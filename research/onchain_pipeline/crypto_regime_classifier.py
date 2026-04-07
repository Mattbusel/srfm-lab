"""
crypto_regime_classifier.py -- Crypto market regime detection and position sizing.

CryptoRegimeClassifier identifies the current market regime using a combination
of on-chain signals (MVRV-Z, NUPL, SOPR) and price-based technicals (200-day MA).

Regimes (in rough cycle order):
  ACCUMULATION -> BULL_MARKET -> EUPHORIA -> CORRECTION -> BEAR_MARKET -> CAPITULATION -> ACCUMULATION

RegimePositioningAdapter maps each regime to a position size multiplier so that
upstream portfolio construction can modulate exposure without knowing regime details.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CryptoRegime enum
# ---------------------------------------------------------------------------

class CryptoRegime(Enum):
    """
    Discrete market regimes for the crypto cycle.

    Each regime has associated on-chain and technical conditions:
      BULL_MARKET  -- BTC above 200MA, MVRV-Z > 0, NUPL > 0.25
      EUPHORIA     -- MVRV-Z > 3, NUPL > 0.75, funding rate > 2%
      CORRECTION   -- BTC < 200MA by < 30%, MVRV-Z in [0, 1]
      BEAR_MARKET  -- BTC < 200MA by > 30%, MVRV-Z < 0, NUPL < 0
      CAPITULATION -- NUPL < -0.25, SOPR < 0.98 for 7+ consecutive days
      ACCUMULATION -- NUPL < 0.25, LTH supply rising, low volatility
      UNKNOWN      -- Insufficient data or contradictory signals
    """

    BULL_MARKET = "bull_market"
    EUPHORIA = "euphoria"
    CORRECTION = "correction"
    BEAR_MARKET = "bear_market"
    CAPITULATION = "capitulation"
    ACCUMULATION = "accumulation"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Regime thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeThresholds:
    """Configurable thresholds for regime classification. Defaults are BTC-centric."""

    # BULL_MARKET
    bull_mvrv_z_min: float = 0.0
    bull_nupl_min: float = 0.25
    bull_ma_ratio_min: float = 1.0  # price / 200MA > 1.0

    # EUPHORIA
    euphoria_mvrv_z_min: float = 3.0
    euphoria_nupl_min: float = 0.75
    euphoria_funding_min: float = 0.02  # 2% annualized funding

    # CORRECTION
    correction_ma_ratio_max: float = 1.0   # below 200MA
    correction_ma_ratio_min: float = 0.70  # not more than 30% below
    correction_mvrv_z_min: float = 0.0
    correction_mvrv_z_max: float = 1.0

    # BEAR_MARKET
    bear_ma_ratio_max: float = 0.70  # more than 30% below 200MA
    bear_mvrv_z_max: float = 0.0
    bear_nupl_max: float = 0.0

    # CAPITULATION
    cap_nupl_max: float = -0.25
    cap_sopr_max: float = 0.98
    cap_sopr_days: int = 7  # consecutive days SOPR < threshold

    # ACCUMULATION
    accum_nupl_max: float = 0.25
    accum_vol_max: float = 0.60  # annualized realized vol cap


# ---------------------------------------------------------------------------
# Regime features
# ---------------------------------------------------------------------------

@dataclass
class RegimeFeatures:
    """Snapshot of all features used for regime classification."""

    date: str
    price: Optional[float] = None
    ma_200: Optional[float] = None
    mvrv_z: Optional[float] = None
    nupl: Optional[float] = None
    sopr: Optional[float] = None
    sopr_below_1_days: int = 0  # consecutive days SOPR < 0.98
    funding_rate: Optional[float] = None
    lth_supply: Optional[float] = None
    lth_supply_slope: Optional[float] = None  # positive = rising
    realized_vol_30d: Optional[float] = None
    regime_score: Dict[CryptoRegime, float] = field(default_factory=dict)

    @property
    def ma_ratio(self) -> Optional[float]:
        """price / 200MA ratio. None if either is missing."""
        if self.price is None or self.ma_200 is None or self.ma_200 == 0.0:
            return None
        return self.price / self.ma_200


# ---------------------------------------------------------------------------
# CryptoRegimeClassifier
# ---------------------------------------------------------------------------

class CryptoRegimeClassifier:
    """
    Classifies the current crypto market regime from on-chain and price data.

    Input data is provided as pd.Series objects indexed by UTC date.
    The classifier uses a scoring approach: each regime accumulates evidence
    points, and the regime with the most evidence (above a minimum threshold)
    wins. Ties are broken by recency of confirming signals.

    Usage::

        clf = CryptoRegimeClassifier()
        clf.fit(
            prices=btc_prices,
            mvrv_z=mvrv_z_series,
            nupl=nupl_series,
            sopr=sopr_series,
            funding_rate=funding_series,
            lth_supply=lth_supply_series,
        )
        regime = clf.classify("2024-03-15")
        features = clf.get_features("2024-03-15")
    """

    def __init__(self, thresholds: Optional[RegimeThresholds] = None) -> None:
        self._th = thresholds or RegimeThresholds()
        self._prices: Optional[pd.Series] = None
        self._ma_200: Optional[pd.Series] = None
        self._mvrv_z: Optional[pd.Series] = None
        self._nupl: Optional[pd.Series] = None
        self._sopr: Optional[pd.Series] = None
        self._funding: Optional[pd.Series] = None
        self._lth_supply: Optional[pd.Series] = None
        self._realized_vol: Optional[pd.Series] = None
        self._sopr_consec: Optional[pd.Series] = None  # consecutive days SOPR < 0.98
        self._lth_slope: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        prices: Optional[pd.Series] = None,
        mvrv_z: Optional[pd.Series] = None,
        nupl: Optional[pd.Series] = None,
        sopr: Optional[pd.Series] = None,
        funding_rate: Optional[pd.Series] = None,
        lth_supply: Optional[pd.Series] = None,
    ) -> "CryptoRegimeClassifier":
        """
        Load data series for classification.

        All series should be indexed by pd.DatetimeIndex (UTC preferred).
        Call this once before classifying individual dates.
        """
        self._prices = prices
        self._mvrv_z = mvrv_z
        self._nupl = nupl
        self._sopr = sopr
        self._funding = funding_rate
        self._lth_supply = lth_supply

        if prices is not None and not prices.empty:
            self._ma_200 = prices.rolling(200, min_periods=20).mean()
            log_ret = np.log(prices / prices.shift(1))
            self._realized_vol = log_ret.rolling(30, min_periods=10).std() * np.sqrt(252)

        if sopr is not None and not sopr.empty:
            self._sopr_consec = self._compute_consecutive_below(sopr, self._th.cap_sopr_max)

        if lth_supply is not None and not lth_supply.empty:
            self._lth_slope = lth_supply.pct_change(periods=14).rolling(7).mean()

        return self

    @staticmethod
    def _compute_consecutive_below(series: pd.Series, threshold: float) -> pd.Series:
        """
        For each date, count how many consecutive days (ending at that date)
        the series has been below `threshold`.
        """
        below = (series < threshold).astype(int)
        result = below.copy().astype(float)
        count = 0
        for i, val in enumerate(below):
            if val:
                count += 1
            else:
                count = 0
            result.iloc[i] = float(count)
        return result

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _get_value(self, series: Optional[pd.Series], date_key: str) -> Optional[float]:
        """Safely retrieve the latest value at or before date_key from a series."""
        if series is None or series.empty:
            return None
        try:
            subset = series[:date_key]
        except Exception:
            subset = series
        if subset.empty:
            return None
        val = subset.iloc[-1]
        return float(val) if not (isinstance(val, float) and np.isnan(val)) else None

    def get_features(self, date: str) -> RegimeFeatures:
        """Extract a RegimeFeatures snapshot for the given date string (YYYY-MM-DD)."""
        features = RegimeFeatures(date=date)
        features.price = self._get_value(self._prices, date)
        features.ma_200 = self._get_value(self._ma_200, date)
        features.mvrv_z = self._get_value(self._mvrv_z, date)
        features.nupl = self._get_value(self._nupl, date)
        features.sopr = self._get_value(self._sopr, date)
        features.funding_rate = self._get_value(self._funding, date)
        features.lth_supply = self._get_value(self._lth_supply, date)
        features.realized_vol_30d = self._get_value(self._realized_vol, date)

        if self._sopr_consec is not None:
            days = self._get_value(self._sopr_consec, date)
            features.sopr_below_1_days = int(days) if days is not None else 0

        if self._lth_slope is not None:
            slope = self._get_value(self._lth_slope, date)
            features.lth_supply_slope = slope

        return features

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_regime(self, f: RegimeFeatures) -> Dict[CryptoRegime, float]:
        """
        Assign an evidence score to each regime.

        Each confirming condition adds 1.0 to the score. Contradicting conditions
        add negative weight. The regime with the highest score wins.
        """
        scores: Dict[CryptoRegime, float] = {r: 0.0 for r in CryptoRegime}
        th = self._th

        # --- EUPHORIA ---
        if f.mvrv_z is not None and f.mvrv_z > th.euphoria_mvrv_z_min:
            scores[CryptoRegime.EUPHORIA] += 1.5
        if f.nupl is not None and f.nupl > th.euphoria_nupl_min:
            scores[CryptoRegime.EUPHORIA] += 1.5
        if f.funding_rate is not None and f.funding_rate > th.euphoria_funding_min:
            scores[CryptoRegime.EUPHORIA] += 1.0

        # --- BULL_MARKET ---
        if f.ma_ratio is not None and f.ma_ratio >= th.bull_ma_ratio_min:
            scores[CryptoRegime.BULL_MARKET] += 1.5
        if f.mvrv_z is not None and f.mvrv_z > th.bull_mvrv_z_min:
            scores[CryptoRegime.BULL_MARKET] += 1.0
        if f.nupl is not None and f.nupl > th.bull_nupl_min:
            scores[CryptoRegime.BULL_MARKET] += 1.0
        # Reduce bull score if euphoria conditions are met (euphoria dominates)
        if f.mvrv_z is not None and f.mvrv_z > th.euphoria_mvrv_z_min:
            scores[CryptoRegime.BULL_MARKET] -= 2.0

        # --- CORRECTION ---
        if f.ma_ratio is not None:
            if th.correction_ma_ratio_min <= f.ma_ratio < th.correction_ma_ratio_max:
                scores[CryptoRegime.CORRECTION] += 2.0
        if f.mvrv_z is not None and th.correction_mvrv_z_min <= f.mvrv_z <= th.correction_mvrv_z_max:
            scores[CryptoRegime.CORRECTION] += 1.0

        # --- BEAR_MARKET ---
        if f.ma_ratio is not None and f.ma_ratio < th.bear_ma_ratio_max:
            scores[CryptoRegime.BEAR_MARKET] += 2.0
        if f.mvrv_z is not None and f.mvrv_z < th.bear_mvrv_z_max:
            scores[CryptoRegime.BEAR_MARKET] += 1.5
        if f.nupl is not None and f.nupl < th.bear_nupl_max:
            scores[CryptoRegime.BEAR_MARKET] += 1.0

        # --- CAPITULATION ---
        if f.nupl is not None and f.nupl < th.cap_nupl_max:
            scores[CryptoRegime.CAPITULATION] += 2.0
        if f.sopr_below_1_days >= th.cap_sopr_days:
            scores[CryptoRegime.CAPITULATION] += 2.0
        if f.mvrv_z is not None and f.mvrv_z < -0.5:
            scores[CryptoRegime.CAPITULATION] += 1.0

        # --- ACCUMULATION ---
        if f.nupl is not None and f.nupl < th.accum_nupl_max and (f.nupl is None or f.nupl >= th.cap_nupl_max):
            scores[CryptoRegime.ACCUMULATION] += 1.5
        if f.lth_supply_slope is not None and f.lth_supply_slope > 0:
            scores[CryptoRegime.ACCUMULATION] += 1.5
        if f.realized_vol_30d is not None and f.realized_vol_30d < th.accum_vol_max:
            scores[CryptoRegime.ACCUMULATION] += 0.5
        # Reduce accumulation score if capitulation conditions are met
        if f.sopr_below_1_days >= th.cap_sopr_days:
            scores[CryptoRegime.ACCUMULATION] -= 1.0

        return scores

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, date: str) -> CryptoRegime:
        """
        Classify the market regime for the given date.

        Parameters
        ----------
        date:
            Date string in YYYY-MM-DD format. Uses data at or before this date.

        Returns
        -------
        CryptoRegime enum value.
        """
        features = self.get_features(date)

        # Check if we have enough data to classify
        available_fields = sum([
            features.price is not None,
            features.mvrv_z is not None,
            features.nupl is not None,
            features.sopr is not None,
        ])
        if available_fields < 2:
            logger.warning("Insufficient data for regime classification at %s", date)
            return CryptoRegime.UNKNOWN

        scores = self._score_regime(features)
        features.regime_score = scores

        # Exclude UNKNOWN from competition -- it only wins by default
        valid_regimes = {r: s for r, s in scores.items() if r != CryptoRegime.UNKNOWN}
        best_regime = max(valid_regimes, key=lambda r: valid_regimes[r])
        best_score = valid_regimes[best_regime]

        if best_score < 1.0:
            # No regime has enough evidence
            return CryptoRegime.UNKNOWN

        return best_regime

    def classify_range(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """
        Classify regime for every date between start_date and end_date.

        Returns pd.Series of CryptoRegime values indexed by date.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")
        regimes = []
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            try:
                regime = self.classify(date_str)
            except Exception as exc:
                logger.warning("Classification failed at %s: %s", date_str, exc)
                regime = CryptoRegime.UNKNOWN
            regimes.append(regime)

        return pd.Series(regimes, index=dates, name="regime")

    def regime_durations(self, start_date: str, end_date: str) -> Dict[CryptoRegime, int]:
        """
        Count days spent in each regime over the range.

        Returns dict of CryptoRegime -> day count.
        """
        regime_series = self.classify_range(start_date, end_date)
        counts: Dict[CryptoRegime, int] = {r: 0 for r in CryptoRegime}
        for regime in regime_series:
            counts[regime] += 1
        return counts


# ---------------------------------------------------------------------------
# RegimePositioningAdapter
# ---------------------------------------------------------------------------

class RegimePositioningAdapter:
    """
    Maps CryptoRegime values to position size multipliers.

    The multiplier scales the base position size (1.0 = full base position).
    Direction is encoded separately -- a multiplier does not indicate direction.

    Regime -> Long multiplier / Short multiplier::

      BULL_MARKET  -> 1.0 longs  / 0.0 shorts
      EUPHORIA     -> 0.5 longs  / 0.5 shorts  (reduce longs, start hedging)
      CORRECTION   -> 0.7 longs  / 0.3 shorts
      BEAR_MARKET  -> 0.0 longs  / 1.0 shorts  (only shorts)
      CAPITULATION -> 0.5 longs  / 0.0 shorts  (scale into longs at bottoms)
      ACCUMULATION -> 0.8 longs  / 0.0 shorts  (build positions)
      UNKNOWN      -> 0.5 longs  / 0.0 shorts  (conservative)

    Override via custom_multipliers in __init__.
    """

    _DEFAULT_LONG: Dict[CryptoRegime, float] = {
        CryptoRegime.BULL_MARKET: 1.0,
        CryptoRegime.EUPHORIA: 0.5,
        CryptoRegime.CORRECTION: 0.7,
        CryptoRegime.BEAR_MARKET: 0.3,
        CryptoRegime.CAPITULATION: 0.5,
        CryptoRegime.ACCUMULATION: 0.8,
        CryptoRegime.UNKNOWN: 0.5,
    }

    _DEFAULT_SHORT: Dict[CryptoRegime, float] = {
        CryptoRegime.BULL_MARKET: 0.0,
        CryptoRegime.EUPHORIA: 0.5,
        CryptoRegime.CORRECTION: 0.3,
        CryptoRegime.BEAR_MARKET: 1.0,
        CryptoRegime.CAPITULATION: 0.0,
        CryptoRegime.ACCUMULATION: 0.0,
        CryptoRegime.UNKNOWN: 0.0,
    }

    def __init__(
        self,
        custom_long_multipliers: Optional[Dict[CryptoRegime, float]] = None,
        custom_short_multipliers: Optional[Dict[CryptoRegime, float]] = None,
    ) -> None:
        self._long = {**self._DEFAULT_LONG, **(custom_long_multipliers or {})}
        self._short = {**self._DEFAULT_SHORT, **(custom_short_multipliers or {})}

    def long_multiplier(self, regime: CryptoRegime) -> float:
        """Return long position size multiplier for the given regime."""
        return self._long.get(regime, 0.5)

    def short_multiplier(self, regime: CryptoRegime) -> float:
        """Return short position size multiplier for the given regime."""
        return self._short.get(regime, 0.0)

    def net_direction(self, regime: CryptoRegime) -> float:
        """
        Net directional bias: long_multiplier - short_multiplier.

        Positive = net long bias; negative = net short bias.
        """
        return self.long_multiplier(regime) - self.short_multiplier(regime)

    def scale_position(
        self, base_size: float, regime: CryptoRegime, is_long: bool
    ) -> float:
        """
        Apply regime multiplier to a base position size.

        Parameters
        ----------
        base_size:
            The full-risk position size.
        regime:
            Current regime classification.
        is_long:
            True if the proposed trade is a long; False if short.

        Returns
        -------
        Scaled position size.
        """
        if is_long:
            return base_size * self.long_multiplier(regime)
        else:
            return base_size * self.short_multiplier(regime)

    def regime_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame summarizing multipliers for all regimes.
        """
        rows = []
        for regime in CryptoRegime:
            rows.append({
                "regime": regime.value,
                "long_multiplier": self._long.get(regime, 0.5),
                "short_multiplier": self._short.get(regime, 0.0),
                "net_direction": self.net_direction(regime),
            })
        return pd.DataFrame(rows).set_index("regime")
