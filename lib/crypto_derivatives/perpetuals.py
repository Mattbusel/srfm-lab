"""
perpetuals.py -- Perpetual futures analytics for crypto derivatives signals.

Covers:
  - Funding rate aggregation across Binance, Bybit, OKX, dYdX
  - Open interest analysis and liquidation cascade risk
  - Basis tracking between spot and perpetual prices

All methods accept data as parameters; no live API calls are required.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUNDING_PERIODS_PER_YEAR = 3 * 365  # 8-hour funding, 3 per day

# Default open-interest weights per exchange (relative, not percentages).
# Weights reflect approximate market share and are user-overridable.
DEFAULT_OI_WEIGHTS: Dict[str, float] = {
    "binance": 0.45,
    "bybit": 0.25,
    "okx": 0.20,
    "dydx": 0.10,
}

# Funding signal thresholds (per 8-hour period as decimal fraction)
FUNDING_VERY_NEGATIVE = -0.001   # -0.1% per 8h -> bullish
FUNDING_VERY_POSITIVE = 0.003    # +0.3% per 8h -> bearish

# Basis signal saturation in bps (beyond this the signal saturates to +/-1)
BASIS_SATURATION_BPS = 100.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FundingRate:
    """Single funding rate observation from one exchange."""
    exchange: str
    symbol: str                 # e.g. "BTCUSDT"
    rate_8h: float              # funding rate per 8-hour period (decimal)
    annualized_rate: float      # rate_8h * FUNDING_PERIODS_PER_YEAR
    timestamp: datetime
    next_funding_time: Optional[datetime] = None

    @classmethod
    def from_rate_8h(
        cls,
        exchange: str,
        symbol: str,
        rate_8h: float,
        timestamp: datetime,
        next_funding_time: Optional[datetime] = None,
    ) -> "FundingRate":
        """Construct a FundingRate, computing annualized_rate automatically."""
        annualized = rate_8h * FUNDING_PERIODS_PER_YEAR
        return cls(
            exchange=exchange,
            symbol=symbol,
            rate_8h=rate_8h,
            annualized_rate=annualized,
            timestamp=timestamp,
            next_funding_time=next_funding_time,
        )

    def __repr__(self) -> str:
        return (
            f"FundingRate({self.exchange}, {self.symbol}, "
            f"rate_8h={self.rate_8h:.4%}, ann={self.annualized_rate:.2%})"
        )


@dataclass
class OISnapshot:
    """Open interest snapshot for a single symbol on a single exchange."""
    exchange: str
    symbol: str
    open_interest_usd: float    # notional USD open interest
    timestamp: datetime


# ---------------------------------------------------------------------------
# FundingRateAggregator
# ---------------------------------------------------------------------------

class FundingRateAggregator:
    """
    Aggregates funding rates from multiple exchanges and converts them to
    actionable trading signals for spot market positioning.

    The composite funding rate is a weighted average across exchanges, with
    weights based on open interest (or static defaults if OI is not provided).

    Signal convention:
      positive signal -> bullish (longs expect positive return)
      negative signal -> bearish (shorts expect positive return / crowded longs)
    """

    SUPPORTED_EXCHANGES = frozenset({"binance", "bybit", "okx", "dydx"})

    def __init__(
        self,
        oi_weights: Optional[Dict[str, float]] = None,
        history_window: int = 30,
    ) -> None:
        """
        Parameters
        ----------
        oi_weights : dict, optional
            Per-exchange open interest weights {exchange: weight}.
            If not provided, DEFAULT_OI_WEIGHTS are used.
        history_window : int
            Number of 8-hour periods to retain in the rolling history
            (default 30 days * 3 periods/day = 90 periods).
        """
        self._weights: Dict[str, float] = dict(oi_weights or DEFAULT_OI_WEIGHTS)
        self._history_window = history_window * 3  # convert days to 8h periods

        # {symbol -> deque of FundingRate objects} (all exchanges combined)
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._history_window)
        )

        # Latest reading per (symbol, exchange)
        self._latest: Dict[Tuple[str, str], FundingRate] = {}

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def ingest(self, funding_rate: FundingRate) -> None:
        """Record a new funding rate observation."""
        key = (funding_rate.symbol, funding_rate.exchange)
        self._latest[key] = funding_rate
        self._history[funding_rate.symbol].append(funding_rate)

    def ingest_batch(self, rates: Sequence[FundingRate]) -> None:
        """Record a list of funding rate observations."""
        for rate in rates:
            self.ingest(rate)

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def _normalized_weights(
        self, symbol: str, oi_overrides: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Return per-exchange weights normalized to sum to 1.0.

        oi_overrides, if provided, is a dict {exchange: oi_usd} and takes
        precedence over the static weights for this call.
        """
        if oi_overrides:
            raw = {ex: oi for ex, oi in oi_overrides.items() if oi > 0}
        else:
            raw = dict(self._weights)

        total = sum(raw.values())
        if total == 0:
            n = len(raw) or 1
            return {ex: 1.0 / n for ex in raw}
        return {ex: w / total for ex, w in raw.items()}

    # ------------------------------------------------------------------
    # Core analytics
    # ------------------------------------------------------------------

    def composite_funding_rate(
        self,
        symbol: str,
        oi_overrides: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Weighted average funding rate (per 8h) across available exchanges.

        Parameters
        ----------
        symbol : str
            Perp symbol, e.g. "BTCUSDT".
        oi_overrides : dict, optional
            Real-time OI per exchange {exchange: usd_value} for dynamic
            weighting.

        Returns
        -------
        float
            Composite 8h funding rate. 0.0 if no data available.
        """
        weights = self._normalized_weights(symbol, oi_overrides)
        composite = 0.0
        weight_used = 0.0

        for exchange, weight in weights.items():
            key = (symbol, exchange)
            if key in self._latest:
                composite += self._latest[key].rate_8h * weight
                weight_used += weight

        if weight_used == 0:
            return 0.0

        # Re-normalize if not all exchanges had data
        return composite / weight_used

    def funding_signal(
        self,
        symbol: str,
        oi_overrides: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Convert composite funding rate to a [-1, 1] directional signal.

        Signal logic:
          rate < FUNDING_VERY_NEGATIVE (-0.1%/8h):
            Shorts paying longs -- bullish, shorts likely to close.
            Signal -> +1.0

          rate > FUNDING_VERY_POSITIVE (+0.3%/8h):
            Longs paying shorts -- bearish, crowded long positions.
            Signal -> -1.0

          Between thresholds:
            Linear interpolation through zero at rate = midpoint.

        Returns
        -------
        float in [-1, 1]
        """
        rate = self.composite_funding_rate(symbol, oi_overrides)
        return _funding_rate_to_signal(rate)

    def funding_momentum(
        self,
        symbol: str,
        window: int = 7,
    ) -> float:
        """
        Measure the trend direction of funding rates over the last `window` days.

        Uses a simple linear regression slope on the recent 8h funding rates
        and normalizes the slope relative to typical funding magnitudes.

        Parameters
        ----------
        window : int
            Number of days to look back (default 7). Converted internally
            to 8h periods.

        Returns
        -------
        float
            Positive value -> funding trending higher (increasingly bullish
            crowding or bearish squeeze depending on direction).
            Negative value -> funding trending lower.
            Returns 0.0 if insufficient history.
        """
        periods = window * 3
        history = self._history.get(symbol)
        if not history:
            return 0.0

        # Aggregate per-period averages (multiple exchanges may exist per period)
        period_rates = _aggregate_history_to_periods(list(history), periods)
        if len(period_rates) < 3:
            return 0.0

        slope = _linear_regression_slope(period_rates)

        # Normalize: a slope equal to FUNDING_VERY_POSITIVE per period = 1.0
        normalization = abs(FUNDING_VERY_POSITIVE) or 1e-6
        return max(-1.0, min(1.0, slope / normalization))

    def rolling_distribution(
        self,
        symbol: str,
        days: int = 30,
    ) -> Dict[str, float]:
        """
        Compute descriptive statistics for the rolling funding rate distribution.

        Returns a dict with keys: mean, std, median, p5, p25, p75, p95, min, max.
        Returns an empty dict if fewer than 3 data points are available.
        """
        periods = days * 3
        history = self._history.get(symbol)
        if not history:
            return {}

        recent = [r.rate_8h for r in list(history)[-periods:]]
        if len(recent) < 3:
            return {}

        sorted_rates = sorted(recent)
        n = len(sorted_rates)

        def percentile(p: float) -> float:
            idx = p * (n - 1)
            lo, hi = int(idx), min(int(idx) + 1, n - 1)
            frac = idx - lo
            return sorted_rates[lo] * (1 - frac) + sorted_rates[hi] * frac

        return {
            "mean": statistics.mean(recent),
            "std": statistics.stdev(recent) if len(recent) > 1 else 0.0,
            "median": statistics.median(recent),
            "p5": percentile(0.05),
            "p25": percentile(0.25),
            "p75": percentile(0.75),
            "p95": percentile(0.95),
            "min": sorted_rates[0],
            "max": sorted_rates[-1],
            "count": n,
        }

    def get_latest_rates(self, symbol: str) -> Dict[str, FundingRate]:
        """Return the latest FundingRate per exchange for a symbol."""
        return {
            exchange: rate
            for (sym, exchange), rate in self._latest.items()
            if sym == symbol
        }


# ---------------------------------------------------------------------------
# OpenInterestAnalyzer
# ---------------------------------------------------------------------------

class OpenInterestAnalyzer:
    """
    Analyzes open interest changes to detect leveraged positioning and
    estimate liquidation cascade risk.

    OI + Price analysis:
      Rising OI + Rising price  -> new longs entering, leveraged trend (bullish)
      Rising OI + Falling price -> new shorts entering, short buildup (bearish)
      Falling OI + Rising price -> short squeeze / de-leveraging (weakly bullish)
      Falling OI + Falling price -> long liquidation / de-leveraging (weakly bearish)
    """

    def __init__(self) -> None:
        # {symbol -> deque of (timestamp, oi_usd, price)} tuples
        self._oi_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        # {symbol -> deque of (timestamp, volume_usd)} tuples
        self._volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record_oi(
        self,
        symbol: str,
        oi_usd: float,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record an open interest observation paired with the current price."""
        ts = timestamp or datetime.now(timezone.utc)
        self._oi_history[symbol].append((ts, oi_usd, price))

    def record_volume(
        self,
        symbol: str,
        volume_usd: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a 24h trading volume observation."""
        ts = timestamp or datetime.now(timezone.utc)
        self._volume_history[symbol].append((ts, volume_usd))

    # ------------------------------------------------------------------
    # Core analytics
    # ------------------------------------------------------------------

    def oi_change_signal(
        self,
        symbol: str,
        window_bars: int = 4,
        oi_series: Optional[List[float]] = None,
        price_series: Optional[List[float]] = None,
    ) -> float:
        """
        Detect leveraged positioning from OI and price co-movement.

        Parameters
        ----------
        symbol : str
            Symbol key for internal history lookup.
        window_bars : int
            Number of recent bars to evaluate (default 4).
        oi_series : list of float, optional
            Override: explicit OI values (most recent last).
        price_series : list of float, optional
            Override: explicit price values (most recent last).

        Returns
        -------
        float in [-1, 1]
            +1 strong leveraged long buildup
            -1 strong leveraged short buildup
             0 neutral / unclear
        """
        if oi_series is not None and price_series is not None:
            oi_vals = oi_series[-window_bars:]
            price_vals = price_series[-window_bars:]
        else:
            history = list(self._oi_history[symbol])[-window_bars:]
            if len(history) < 2:
                return 0.0
            oi_vals = [h[1] for h in history]
            price_vals = [h[2] for h in history]

        if len(oi_vals) < 2 or len(price_vals) < 2:
            return 0.0

        oi_change_pct = (oi_vals[-1] - oi_vals[0]) / (abs(oi_vals[0]) + 1e-10)
        price_change_pct = (price_vals[-1] - price_vals[0]) / (abs(price_vals[0]) + 1e-10)

        # Combine: both rising -> bullish leverage; inverse -> bearish leverage
        combined = oi_change_pct * math.copysign(1.0, price_change_pct)

        # Scale: 10% rapid OI change with directional price = ~0.5 signal
        signal = combined / 0.20
        return max(-1.0, min(1.0, signal))

    def oi_to_volume_ratio(
        self,
        symbol: str,
        oi_usd: Optional[float] = None,
        volume_24h_usd: Optional[float] = None,
    ) -> float:
        """
        Compute OI-to-volume ratio.

        A high ratio (> 1.0) indicates leverage is elevated relative to
        actual trading activity -- potential for volatile moves.

        Parameters
        ----------
        symbol : str
            Used for internal history lookup if explicit values not given.
        oi_usd : float, optional
            Override: explicit open interest in USD.
        volume_24h_usd : float, optional
            Override: explicit 24h volume in USD.

        Returns
        -------
        float
            OI / 24h volume. Returns 0.0 if volume is zero or unavailable.
        """
        if oi_usd is not None and volume_24h_usd is not None:
            oi_val = oi_usd
            vol_val = volume_24h_usd
        else:
            oi_hist = list(self._oi_history[symbol])
            vol_hist = list(self._volume_history[symbol])
            if not oi_hist or not vol_hist:
                return 0.0
            oi_val = oi_hist[-1][1]
            vol_val = vol_hist[-1][1]

        if vol_val <= 0:
            return 0.0
        return oi_val / vol_val

    def liquidation_cascade_risk(
        self,
        symbol: str,
        oi_usd: float,
        leverage_estimate: float = 10.0,
        price_move_pct: float = 0.10,
    ) -> float:
        """
        Estimate the notional USD volume at risk of liquidation if the price
        moves by `price_move_pct` in either direction.

        Model:
          At leverage L, a price move of 1/L wipes out the entire position.
          Fraction liquidated = min(1.0, price_move_pct * leverage_estimate)
          Liquidation volume = oi_usd * fraction_liquidated

        This is a simplified, first-order estimate suitable for risk screening.
        For a more accurate model, use a liquidation-level distribution.

        Parameters
        ----------
        symbol : str
            Symbol (used for logging / future extension).
        oi_usd : float
            Notional open interest in USD.
        leverage_estimate : float
            Average platform leverage (default 10x).
        price_move_pct : float
            Hypothetical price move size as a decimal (default 0.10 = 10%).

        Returns
        -------
        float
            Estimated liquidation volume in USD.
        """
        if oi_usd <= 0 or leverage_estimate <= 0:
            return 0.0

        # At leverage L, a 1/L price move liquidates 100% of margined positions.
        # For partial moves, fraction is linear up to 100%.
        liquidation_fraction = min(1.0, price_move_pct * leverage_estimate)
        return oi_usd * liquidation_fraction

    def oi_trend(
        self,
        symbol: str,
        window_bars: int = 12,
        oi_series: Optional[List[float]] = None,
    ) -> float:
        """
        Measure the directional trend in open interest.

        Returns a normalized slope: positive = growing OI, negative = shrinking.
        """
        if oi_series is not None:
            vals = oi_series[-window_bars:]
        else:
            history = list(self._oi_history[symbol])[-window_bars:]
            if len(history) < 3:
                return 0.0
            vals = [h[1] for h in history]

        if len(vals) < 3:
            return 0.0

        slope = _linear_regression_slope(vals)
        mean_oi = statistics.mean(vals) or 1.0
        normalized = slope / (abs(mean_oi) * 0.01)   # 1% per period = ~1.0
        return max(-1.0, min(1.0, normalized))

    def leverage_heatmap(
        self,
        symbol: str,
        oi_series: List[float],
        price_series: List[float],
        volume_series: List[float],
    ) -> Dict[str, float]:
        """
        Return a summary dict of leverage metrics useful for dashboard display.

        Keys: oi_usd, volume_24h_usd, oi_volume_ratio, oi_trend, oi_change_signal.
        """
        oi_last = oi_series[-1] if oi_series else 0.0
        vol_last = volume_series[-1] if volume_series else 0.0
        return {
            "oi_usd": oi_last,
            "volume_24h_usd": vol_last,
            "oi_volume_ratio": self.oi_to_volume_ratio(
                symbol, oi_usd=oi_last, volume_24h_usd=vol_last
            ),
            "oi_trend": self.oi_trend(symbol, oi_series=oi_series),
            "oi_change_signal": self.oi_change_signal(
                symbol, oi_series=oi_series, price_series=price_series
            ),
        }


# ---------------------------------------------------------------------------
# BasisTracker
# ---------------------------------------------------------------------------

class BasisTracker:
    """
    Tracks the basis between spot and perpetual prices and converts it into
    directional and carry signals.

    Basis = (perp price - spot price) / spot price

    Positive basis (perp > spot):
      - Market participants willing to pay a premium for leveraged long exposure.
      - Bullish sentiment signal.

    Negative basis (perp < spot):
      - Market paying a premium for leveraged short exposure.
      - Bearish sentiment signal.

    Annualized basis can be compared against DeFi lending yields as an
    arbitrage reference.
    """

    def __init__(self) -> None:
        # {symbol -> deque of (timestamp, spot, perp, basis_bps)} tuples
        self._basis_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record_basis(
        self,
        symbol: str,
        spot_price: float,
        perp_price: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a spot/perp price pair observation."""
        ts = timestamp or datetime.now(timezone.utc)
        bps = self.basis_bps(spot_price, perp_price)
        self._basis_history[symbol].append((ts, spot_price, perp_price, bps))

    # ------------------------------------------------------------------
    # Core analytics
    # ------------------------------------------------------------------

    @staticmethod
    def basis_bps(spot_price: float, perp_price: float) -> float:
        """
        Compute basis in basis points.

        basis_bps = (perp - spot) / spot * 10_000

        Parameters
        ----------
        spot_price : float
            Current spot market price.
        perp_price : float
            Current perpetual futures price.

        Returns
        -------
        float
            Basis in basis points. Positive -> perp premium (bullish).
        """
        if spot_price <= 0:
            raise ValueError(f"spot_price must be positive, got {spot_price}")
        return (perp_price - spot_price) / spot_price * 10_000.0

    def basis_signal(
        self,
        symbol: str,
        spot_price: Optional[float] = None,
        perp_price: Optional[float] = None,
    ) -> float:
        """
        Convert basis to a [-1, 1] directional signal.

        Signal saturates at +/-1.0 when abs(basis) >= BASIS_SATURATION_BPS.

        Parameters
        ----------
        symbol : str
            Symbol key (used for history if explicit prices not given).
        spot_price : float, optional
            Override spot price for current observation.
        perp_price : float, optional
            Override perp price for current observation.

        Returns
        -------
        float in [-1, 1]
        """
        if spot_price is not None and perp_price is not None:
            bps = self.basis_bps(spot_price, perp_price)
        else:
            history = list(self._basis_history[symbol])
            if not history:
                return 0.0
            bps = history[-1][3]

        signal = bps / BASIS_SATURATION_BPS
        return max(-1.0, min(1.0, signal))

    def annualized_basis(
        self,
        symbol: str,
        days_to_settlement: Optional[float] = None,
        spot_price: Optional[float] = None,
        perp_price: Optional[float] = None,
    ) -> float:
        """
        Compute the annualized carry rate implied by the current basis.

        For a perpetual (no settlement), uses average rolling funding rate as
        a proxy. If days_to_settlement is provided (e.g. for dated futures),
        annualizes the basis directly.

        Parameters
        ----------
        symbol : str
            Symbol key.
        days_to_settlement : float, optional
            Days until contract expiry. If None, treats contract as perpetual
            and annualizes using a 365-day horizon.
        spot_price : float, optional
            Override: spot price.
        perp_price : float, optional
            Override: perp price.

        Returns
        -------
        float
            Annualized basis as a decimal fraction (0.05 = 5% annualized carry).
        """
        if spot_price is not None and perp_price is not None:
            bps = self.basis_bps(spot_price, perp_price)
        else:
            history = list(self._basis_history[symbol])
            if not history:
                return 0.0
            bps = history[-1][3]

        basis_decimal = bps / 10_000.0

        if days_to_settlement is not None and days_to_settlement > 0:
            return basis_decimal * (365.0 / days_to_settlement)

        # Perpetual: annualize over 365 days as a convention
        return basis_decimal * 365.0

    def rolling_basis_stats(
        self,
        symbol: str,
        days: int = 7,
    ) -> Dict[str, float]:
        """
        Descriptive statistics for the basis over the last `days` days.

        Assumes observations are roughly hourly; uses the last days*24 records.

        Returns dict with: mean_bps, std_bps, min_bps, max_bps, trend.
        """
        history = list(self._basis_history[symbol])
        window = days * 24
        recent = history[-window:]
        if len(recent) < 2:
            return {}

        bps_vals = [r[3] for r in recent]
        slope = _linear_regression_slope(bps_vals)
        std = statistics.stdev(bps_vals) if len(bps_vals) > 1 else 0.0

        return {
            "mean_bps": statistics.mean(bps_vals),
            "std_bps": std,
            "min_bps": min(bps_vals),
            "max_bps": max(bps_vals),
            "trend": slope,       # bps per period (positive = widening)
            "count": len(bps_vals),
        }

    def basis_zscore(
        self,
        symbol: str,
        lookback_days: int = 7,
        spot_price: Optional[float] = None,
        perp_price: Optional[float] = None,
    ) -> float:
        """
        Z-score of current basis vs. its rolling distribution.

        Useful for mean-reversion strategies when basis is historically elevated.
        """
        stats = self.rolling_basis_stats(symbol, days=lookback_days)
        if not stats or stats.get("std_bps", 0) == 0:
            return 0.0

        if spot_price is not None and perp_price is not None:
            current_bps = self.basis_bps(spot_price, perp_price)
        else:
            history = list(self._basis_history[symbol])
            if not history:
                return 0.0
            current_bps = history[-1][3]

        return (current_bps - stats["mean_bps"]) / stats["std_bps"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _funding_rate_to_signal(rate: float) -> float:
    """
    Map a raw 8h funding rate to a [-1, 1] signal.

    Piecewise linear:
      rate <= FUNDING_VERY_NEGATIVE  -> +1.0  (bullish)
      rate >= FUNDING_VERY_POSITIVE  -> -1.0  (bearish)
      linear interpolation in between, passing through 0 at the midpoint.
    """
    if rate <= FUNDING_VERY_NEGATIVE:
        return 1.0
    if rate >= FUNDING_VERY_POSITIVE:
        return -1.0

    midpoint = (FUNDING_VERY_NEGATIVE + FUNDING_VERY_POSITIVE) / 2.0
    half_range = (FUNDING_VERY_POSITIVE - FUNDING_VERY_NEGATIVE) / 2.0

    normalized = (rate - midpoint) / half_range
    return max(-1.0, min(1.0, -normalized))


def _aggregate_history_to_periods(
    history: List[FundingRate],
    max_periods: int,
) -> List[float]:
    """
    Collapse per-exchange funding rates into per-period averages.

    Groups by approximate 8h timestamp bucket and averages across exchanges.
    Returns a list of average rates (oldest first), up to max_periods.
    """
    if not history:
        return []

    # Group by timestamp rounded to 8-hour bucket
    buckets: Dict[int, List[float]] = defaultdict(list)
    for fr in history:
        ts_unix = int(fr.timestamp.timestamp())
        bucket = (ts_unix // (8 * 3600)) * (8 * 3600)
        buckets[bucket].append(fr.rate_8h)

    sorted_buckets = sorted(buckets.keys())[-max_periods:]
    return [statistics.mean(buckets[b]) for b in sorted_buckets]


def _linear_regression_slope(values: List[float]) -> float:
    """
    Compute the OLS slope of values against their index positions.

    Returns the slope as float. Returns 0.0 for fewer than 2 values.
    """
    n = len(values)
    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_funding_rate(
    exchange: str,
    symbol: str,
    rate_8h: float,
    timestamp_utc: Optional[datetime] = None,
) -> FundingRate:
    """Convenience factory for constructing FundingRate objects in tests."""
    ts = timestamp_utc or datetime.now(timezone.utc)
    return FundingRate.from_rate_8h(exchange, symbol, rate_8h, ts)
