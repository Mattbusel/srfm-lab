"""
options_market.py -- Crypto options market analytics (Deribit-style).

Covers:
  - Implied volatility surface construction from option chains
  - ATM vol, vol skew, and term structure analysis
  - Variance risk premium (VRP) signal
  - Composite derivatives signal combining funding, basis, skew, and VRP

All methods accept data as parameters; no live API calls are required.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 365.0
SQRT_YEAR = math.sqrt(TRADING_DAYS_PER_YEAR)

# Vol surface interpolation
MIN_IV = 1e-6
MAX_IV = 50.0    # 5000% -- clamp unreasonable quotes

# VRP signal thresholds
VRP_RICH_RATIO = 1.5      # IV/RV > 1.5 -> options expensive -> fade (sell vol)
VRP_CHEAP_RATIO = 0.7     # IV/RV < 0.7 -> options cheap -> accumulate (buy vol)

# Skew signal saturation in vol points
SKEW_SATURATION_VOL_PTS = 0.15   # 15 vol points 25-delta skew = saturate signal

# Term structure signal
TERM_NORMAL_THRESHOLD = 1.0    # >1 normal contango, <1 inverted (stress)

# Composite weights
COMPOSITE_WEIGHTS = {
    "funding": 0.30,
    "basis": 0.25,
    "skew": 0.25,
    "vrp": 0.20,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OptionQuote:
    """Single option quote from an exchange."""
    strike: float
    expiry_days: float          # days to expiry (can be fractional)
    option_type: str            # "call" or "put"
    bid: float
    ask: float
    iv: Optional[float] = None  # implied volatility (decimal, e.g. 0.80 = 80%)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    open_interest: float = 0.0
    volume: float = 0.0

    @property
    def mid(self) -> float:
        """Mid-market quote."""
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid


@dataclass
class DeribitOptionChain:
    """Full option chain for a single underlying and expiry."""
    symbol: str                 # e.g. "BTC"
    expiry: datetime
    expiry_days: float          # days to expiry from now
    strikes: List[float]
    calls: List[OptionQuote]
    puts: List[OptionQuote]
    spot_price: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def atm_strike(self) -> Optional[float]:
        """Return the strike closest to the current spot price."""
        if not self.strikes or self.spot_price <= 0:
            return None
        return min(self.strikes, key=lambda s: abs(s - self.spot_price))

    def calls_by_strike(self) -> Dict[float, OptionQuote]:
        return {q.strike: q for q in self.calls}

    def puts_by_strike(self) -> Dict[float, OptionQuote]:
        return {q.strike: q for q in self.puts}


@dataclass
class DerivativesSignal:
    """
    Composite signal combining multiple derivatives market inputs.

    All component scores are in [-1, 1].
    composite_score is a weighted combination of the components.

    interpretation is a human-readable string summarizing the overall stance.
    """
    funding_component: float
    basis_component: float
    skew_component: float
    vrp_component: float
    composite_score: float
    interpretation: str = ""

    def to_dict(self) -> Dict[str, float]:
        return {
            "funding": self.funding_component,
            "basis": self.basis_component,
            "skew": self.skew_component,
            "vrp": self.vrp_component,
            "composite": self.composite_score,
        }


# ---------------------------------------------------------------------------
# CryptoImpliedVol
# ---------------------------------------------------------------------------

class CryptoImpliedVol:
    """
    Builds and queries an implied volatility surface from option chains.

    The surface is stored as a nested dict:
      {expiry_days -> {strike -> iv}}

    Interpolation is linear in log-strike space; extrapolation flat.
    """

    def __init__(self) -> None:
        # {symbol -> {expiry_days -> {strike -> iv}}}
        self._surfaces: Dict[str, Dict[float, Dict[float, float]]] = {}
        # {symbol -> spot_price}
        self._spots: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Surface construction
    # ------------------------------------------------------------------

    def vol_surface(self, option_chain: DeribitOptionChain) -> Dict[str, object]:
        """
        Build or update the IV surface from a DeribitOptionChain.

        Uses call IVs for strikes above ATM, put IVs for strikes below ATM,
        and averages them at the ATM strike to reduce skew contamination.

        Parameters
        ----------
        option_chain : DeribitOptionChain
            Populated option chain with IV values on each quote.

        Returns
        -------
        dict
            Summary dict with keys:
              expiry_days, atm_iv, skew_25d, term_slope, strikes, ivs
        """
        symbol = option_chain.symbol
        expiry = option_chain.expiry_days
        spot = option_chain.spot_price or 1.0

        if symbol not in self._surfaces:
            self._surfaces[symbol] = {}
        if symbol not in self._spots or spot > 0:
            self._spots[symbol] = spot

        surface_slice: Dict[float, float] = {}

        calls_by_strike = option_chain.calls_by_strike()
        puts_by_strike = option_chain.puts_by_strike()
        all_strikes = sorted(
            set(calls_by_strike.keys()) | set(puts_by_strike.keys())
        )

        for strike in all_strikes:
            iv_candidates = []
            if strike >= spot and strike in calls_by_strike:
                q = calls_by_strike[strike]
                if q.iv and MIN_IV < q.iv < MAX_IV:
                    iv_candidates.append(q.iv)
            if strike < spot and strike in puts_by_strike:
                q = puts_by_strike[strike]
                if q.iv and MIN_IV < q.iv < MAX_IV:
                    iv_candidates.append(q.iv)
            # Fallback: use whichever side is available
            if not iv_candidates:
                for src in [calls_by_strike, puts_by_strike]:
                    if strike in src and src[strike].iv:
                        iv_val = src[strike].iv
                        if MIN_IV < iv_val < MAX_IV:
                            iv_candidates.append(iv_val)
            if iv_candidates:
                surface_slice[strike] = statistics.mean(iv_candidates)

        self._surfaces[symbol][expiry] = surface_slice

        atm = self.atm_vol_from_chain(option_chain)
        sorted_expiries = sorted(self._surfaces[symbol].keys())

        return {
            "symbol": symbol,
            "expiry_days": expiry,
            "atm_iv": atm,
            "strikes": sorted(surface_slice.keys()),
            "ivs": [surface_slice[s] for s in sorted(surface_slice.keys())],
            "num_strikes": len(surface_slice),
            "term_expiries": sorted_expiries,
        }

    def atm_vol_from_chain(self, option_chain: DeribitOptionChain) -> float:
        """
        Compute ATM vol directly from a chain, using the strike closest to spot.
        """
        spot = option_chain.spot_price or 1.0
        all_quotes = list(option_chain.calls) + list(option_chain.puts)
        if not all_quotes:
            return 0.0

        atm_candidates = sorted(
            [q for q in all_quotes if q.iv and MIN_IV < q.iv < MAX_IV],
            key=lambda q: abs(q.strike - spot),
        )
        if not atm_candidates:
            return 0.0
        # Average the two nearest strikes if available
        n = min(2, len(atm_candidates))
        return statistics.mean(q.iv for q in atm_candidates[:n])

    # ------------------------------------------------------------------
    # Surface queries
    # ------------------------------------------------------------------

    def atm_vol(self, symbol: str, expiry_days: float) -> float:
        """
        At-the-money implied volatility for the given expiry.

        Interpolates linearly between stored expiry slices if the exact
        expiry is not present.

        Parameters
        ----------
        symbol : str
        expiry_days : float
            Target expiry in days.

        Returns
        -------
        float
            ATM IV (decimal). 0.0 if surface not populated.
        """
        surface = self._surfaces.get(symbol)
        if not surface:
            return 0.0
        spot = self._spots.get(symbol, 1.0)
        return self._interpolate_atm_vol(surface, spot, expiry_days)

    def vol_skew(
        self,
        symbol: str,
        expiry_days: float,
        delta_range: float = 0.25,
        option_chain: Optional[DeribitOptionChain] = None,
    ) -> float:
        """
        25-delta skew: IV(25d call) - IV(25d put).

        Sign convention:
          Positive skew -> calls more expensive than puts (market expects
            upside or downside put protection is cheap relative to calls).
          Negative skew -> puts more expensive (fear of downside -- typical
            for equities and crypto during risk-off periods).

        Parameters
        ----------
        symbol : str
        expiry_days : float
        delta_range : float
            Delta level for skew calculation (default 0.25 = 25-delta).
        option_chain : DeribitOptionChain, optional
            If provided, computes skew directly from the chain quotes.

        Returns
        -------
        float
            Skew in vol points (decimal). E.g. -0.05 = -5 vol points.
        """
        if option_chain is not None:
            return self._chain_skew(option_chain, delta_range)

        surface = self._surfaces.get(symbol, {}).get(expiry_days)
        if not surface:
            return 0.0
        spot = self._spots.get(symbol, 1.0)
        return self._surface_skew(surface, spot, expiry_days, delta_range)

    def term_structure_slope(self, symbol: str) -> float:
        """
        Ratio of 30-day ATM vol to 7-day ATM vol.

        Interpretation:
          > 1.0 -> normal term structure (further expiries have higher vol)
          < 1.0 -> inverted term structure (near-term vol elevated -- stress)
          = 1.0 -> flat term structure

        Returns
        -------
        float
            30d_vol / 7d_vol. Returns 1.0 (neutral) if insufficient data.
        """
        vol_7d = self.atm_vol(symbol, 7.0)
        vol_30d = self.atm_vol(symbol, 30.0)

        if vol_7d <= 0:
            return 1.0
        return vol_30d / vol_7d

    # ------------------------------------------------------------------
    # Private interpolation helpers
    # ------------------------------------------------------------------

    def _interpolate_atm_vol(
        self,
        surface: Dict[float, Dict[float, float]],
        spot: float,
        target_expiry: float,
    ) -> float:
        """Linear interpolation of ATM vol across expiry slices."""
        expiries = sorted(surface.keys())
        if not expiries:
            return 0.0

        if target_expiry <= expiries[0]:
            return self._atm_from_slice(surface[expiries[0]], spot)
        if target_expiry >= expiries[-1]:
            return self._atm_from_slice(surface[expiries[-1]], spot)

        for i in range(len(expiries) - 1):
            lo, hi = expiries[i], expiries[i + 1]
            if lo <= target_expiry <= hi:
                frac = (target_expiry - lo) / (hi - lo)
                vol_lo = self._atm_from_slice(surface[lo], spot)
                vol_hi = self._atm_from_slice(surface[hi], spot)
                if vol_lo <= 0 or vol_hi <= 0:
                    return vol_lo or vol_hi
                # Variance interpolation (linear in total variance)
                var_lo = vol_lo ** 2 * lo
                var_hi = vol_hi ** 2 * hi
                var_interp = var_lo * (1 - frac) + var_hi * frac
                return math.sqrt(var_interp / target_expiry) if target_expiry > 0 else 0.0

        return 0.0

    @staticmethod
    def _atm_from_slice(
        slice_: Dict[float, float],
        spot: float,
    ) -> float:
        """Extract ATM vol from a strike->iv slice."""
        if not slice_:
            return 0.0
        nearest = min(slice_.keys(), key=lambda s: abs(s - spot))
        # Average two nearest if possible
        sorted_strikes = sorted(slice_.keys(), key=lambda s: abs(s - spot))
        n = min(2, len(sorted_strikes))
        return statistics.mean(slice_[s] for s in sorted_strikes[:n])

    def _surface_skew(
        self,
        surface: Dict[float, float],
        spot: float,
        expiry_days: float,
        delta_range: float,
    ) -> float:
        """
        Approximate 25-delta skew from surface slice using Black-Scholes
        delta-to-strike conversion.
        """
        if not surface or expiry_days <= 0:
            return 0.0

        atm_iv = self._atm_from_slice(surface, spot)
        if atm_iv <= 0:
            return 0.0

        t = expiry_days / TRADING_DAYS_PER_YEAR
        sqrt_t = math.sqrt(t)

        # Approximate: for a lognormal model, 25d call strike is above ATM
        # and 25d put strike is below ATM.
        # Using: delta = N(d1), d1 = (ln(F/K) + 0.5*sigma^2*t) / (sigma*sqrt(t))
        # Invert numerically: K = F * exp(-N_inv(delta)*sigma*sqrt_t + 0.5*sigma^2*t)
        # Approximate N_inv(0.25) ~ -0.674, N_inv(0.75) ~ 0.674

        n_inv_25_put = -0.6745     # N_inv(0.25) -- put 25d
        n_inv_75_call = 0.6745     # N_inv(0.75) -- call 25d (delta = N(d1))

        # Actually for call: delta ~ N(d1), 25-delta call has d1 = N_inv(0.25)
        # meaning d1 ~ -0.674 -> OTM call

        def approx_strike_from_delta_d1(d1_val: float) -> float:
            """K = spot * exp(-d1*sigma*sqrt_t + 0.5*sigma^2*t)"""
            return spot * math.exp(-d1_val * atm_iv * sqrt_t + 0.5 * atm_iv**2 * t)

        # 25-delta call: d1 = N_inv(0.25) = -0.674 (OTM call)
        k_25c = approx_strike_from_delta_d1(n_inv_25_put)
        # 25-delta put: d1 = N_inv(0.75) = 0.674
        k_25p = approx_strike_from_delta_d1(n_inv_75_call)

        def iv_at_strike(k: float) -> float:
            nearest = min(surface.keys(), key=lambda s: abs(s - k))
            return surface[nearest]

        iv_25c = iv_at_strike(k_25c)
        iv_25p = iv_at_strike(k_25p)
        return iv_25c - iv_25p

    def _chain_skew(
        self,
        chain: DeribitOptionChain,
        delta_range: float,
    ) -> float:
        """
        Compute skew from delta values on individual option quotes.

        Uses the nearest-delta quotes to delta_range for calls and puts.
        """
        spot = chain.spot_price or 1.0

        call_25d = _nearest_delta_quote(chain.calls, delta_range)
        put_25d = _nearest_delta_quote(chain.puts, -(delta_range))

        if call_25d and call_25d.iv and put_25d and put_25d.iv:
            return call_25d.iv - put_25d.iv

        # Fallback: use strikes
        if not chain.strikes:
            return 0.0

        t = chain.expiry_days / TRADING_DAYS_PER_YEAR
        if t <= 0:
            return 0.0

        all_quotes_with_iv = [
            q for q in list(chain.calls) + list(chain.puts)
            if q.iv and MIN_IV < q.iv < MAX_IV
        ]
        if len(all_quotes_with_iv) < 2:
            return 0.0

        atm_iv = self.atm_vol_from_chain(chain)
        if atm_iv <= 0:
            return 0.0

        sqrt_t = math.sqrt(t)
        n_inv = 0.6745

        k_call = spot * math.exp(-(-n_inv) * atm_iv * sqrt_t + 0.5 * atm_iv**2 * t)
        k_put = spot * math.exp(-(n_inv) * atm_iv * sqrt_t + 0.5 * atm_iv**2 * t)

        def nearest_iv(target_k: float, quotes: List[OptionQuote]) -> float:
            valid = [q for q in quotes if q.iv and MIN_IV < q.iv < MAX_IV]
            if not valid:
                return 0.0
            nearest = min(valid, key=lambda q: abs(q.strike - target_k))
            return nearest.iv

        iv_call = nearest_iv(k_call, chain.calls)
        iv_put = nearest_iv(k_put, chain.puts)

        if iv_call <= 0 or iv_put <= 0:
            return 0.0
        return iv_call - iv_put


# ---------------------------------------------------------------------------
# CryptoVolRegime
# ---------------------------------------------------------------------------

class CryptoVolRegime:
    """
    Identifies the volatility regime using the Variance Risk Premium (VRP).

    VRP = Implied Variance - Realized Variance

    When options are expensive relative to realized vol (high IV/RV ratio):
      Market over-prices vol -> fade signal (sell options, short vol)
      Signal -> negative (bearish for long vol positions)

    When options are cheap relative to realized vol (low IV/RV ratio):
      Market under-prices vol -> accumulate (buy options, long vol)
      Signal -> positive (bullish for long vol positions)
    """

    def __init__(self) -> None:
        # {symbol -> list of (timestamp, iv, rv)} tuples
        self._vrp_history: Dict[str, list] = {}

    def vrp_signal(
        self,
        symbol: str,
        implied_vol: float,
        realized_vol: float,
        iv_rv_history: Optional[List[Tuple[float, float]]] = None,
    ) -> float:
        """
        Compute the variance risk premium signal.

        Parameters
        ----------
        symbol : str
        implied_vol : float
            Current ATM implied vol (decimal, e.g. 0.80 = 80%).
        realized_vol : float
            Recent realized vol, same tenor as implied (decimal).
        iv_rv_history : list of (iv, rv) tuples, optional
            Historical IV/RV pairs for normalization.

        Returns
        -------
        float in [-1, 1]
            +1 -> options very cheap (buy vol)
            -1 -> options very expensive (sell vol)
        """
        if realized_vol <= 0 or implied_vol <= 0:
            return 0.0

        ratio = implied_vol / realized_vol
        return _vrp_ratio_to_signal(ratio)

    def vrp_ratio(
        self,
        implied_vol: float,
        realized_vol: float,
    ) -> float:
        """
        Raw IV/RV ratio. >1 means options are expensive relative to realized vol.

        Returns
        -------
        float
            IV / RV ratio. Returns 0.0 if RV is zero.
        """
        if realized_vol <= 0:
            return 0.0
        return implied_vol / realized_vol

    def realized_vol(
        self,
        price_series: List[float],
        window: int = 30,
        annualize: bool = True,
    ) -> float:
        """
        Compute realized vol (close-to-close) from a price series.

        Parameters
        ----------
        price_series : list of float
            Daily closing prices, oldest first.
        window : int
            Number of periods to use (default 30 days).
        annualize : bool
            If True, multiply by sqrt(365) (default True).

        Returns
        -------
        float
            Realized volatility (decimal).
        """
        prices = price_series[-window - 1:]
        if len(prices) < 2:
            return 0.0

        log_returns = [
            math.log(prices[i] / prices[i - 1])
            for i in range(1, len(prices))
            if prices[i - 1] > 0 and prices[i] > 0
        ]
        if len(log_returns) < 2:
            return 0.0

        rv = statistics.stdev(log_returns)
        if annualize:
            rv *= SQRT_YEAR
        return rv

    def vol_regime_label(
        self,
        implied_vol: float,
        realized_vol: float,
    ) -> str:
        """
        Return a human-readable regime label.

        Labels: "expensive", "rich", "fair", "cheap", "very_cheap"
        """
        if realized_vol <= 0:
            return "unknown"
        ratio = implied_vol / realized_vol
        if ratio >= 2.0:
            return "expensive"
        if ratio >= VRP_RICH_RATIO:
            return "rich"
        if ratio >= VRP_CHEAP_RATIO:
            return "fair"
        if ratio >= 0.5:
            return "cheap"
        return "very_cheap"

    def rolling_vrp(
        self,
        iv_series: List[float],
        rv_series: List[float],
        window: int = 30,
    ) -> List[float]:
        """
        Compute rolling VRP (IV - RV) over aligned series.

        Returns a list of vrp values (same length as inputs, or shorter).
        """
        n = min(len(iv_series), len(rv_series))
        if n < 1:
            return []
        return [iv_series[i] - rv_series[i] for i in range(max(0, n - window), n)]


# ---------------------------------------------------------------------------
# PerpOptionsComposite
# ---------------------------------------------------------------------------

class PerpOptionsComposite:
    """
    Combines funding rate, basis, implied vol skew, and VRP into a single
    composite derivatives signal for spot market positioning.

    Component signals are weighted and summed into a composite score in [-1, 1].
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Parameters
        ----------
        weights : dict, optional
            Override the default component weights.
            Keys: "funding", "basis", "skew", "vrp".
            Values are relative (will be normalized to sum to 1).
        """
        raw_weights = weights or COMPOSITE_WEIGHTS
        total = sum(raw_weights.values())
        self._weights = {k: v / total for k, v in raw_weights.items()}

    def composite_derivatives_signal(
        self,
        symbol: str,
        funding_signal: float,
        basis_signal: float,
        skew_signal: float,
        vrp_signal: float,
    ) -> DerivativesSignal:
        """
        Combine component signals into a single DerivativesSignal.

        Parameters
        ----------
        symbol : str
            Asset symbol (used for interpretation label).
        funding_signal : float
            Funding rate signal in [-1, 1]. +1 = bullish (negative funding).
        basis_signal : float
            Basis signal in [-1, 1]. +1 = bullish (perp > spot premium).
        skew_signal : float
            Vol skew signal in [-1, 1]. +1 = bullish (call skew rich).
        vrp_signal : float
            VRP signal in [-1, 1]. +1 = bullish (options cheap).

        Returns
        -------
        DerivativesSignal
        """
        # Clamp all inputs to [-1, 1]
        f = max(-1.0, min(1.0, funding_signal))
        b = max(-1.0, min(1.0, basis_signal))
        s = max(-1.0, min(1.0, skew_signal))
        v = max(-1.0, min(1.0, vrp_signal))

        composite = (
            self._weights["funding"] * f
            + self._weights["basis"] * b
            + self._weights["skew"] * s
            + self._weights["vrp"] * v
        )
        composite = max(-1.0, min(1.0, composite))

        interpretation = _interpret_composite(symbol, composite, f, b, s, v)

        return DerivativesSignal(
            funding_component=f,
            basis_component=b,
            skew_component=s,
            vrp_component=v,
            composite_score=composite,
            interpretation=interpretation,
        )

    def skew_to_signal(self, skew_vol_pts: float) -> float:
        """
        Convert raw 25-delta skew (vol points) to a [-1, 1] signal.

        Positive skew (calls > puts) -> bullish signal.
        Negative skew (puts > calls) -> bearish signal (fear premium).

        Saturates at +/-1.0 when abs(skew) >= SKEW_SATURATION_VOL_PTS.
        """
        signal = skew_vol_pts / SKEW_SATURATION_VOL_PTS
        return max(-1.0, min(1.0, signal))

    def term_structure_to_signal(self, slope: float) -> float:
        """
        Convert term structure slope (30d/7d vol ratio) to a signal.

        slope > 1 (normal): slightly bearish (high long-dated uncertainty)
        slope < 1 (inverted): strongly bearish (front-end stress)

        Returns float in [-1, 1].
        """
        if slope <= 0:
            return -1.0
        # Inversion is more significant than steepening
        if slope >= 1.0:
            # Normal: small negative signal proportional to steepness
            excess = slope - 1.0
            return max(-1.0, -excess * 2.0)
        else:
            # Inverted: bearish, scaling from 0 (slope=1) to -1 (slope=0.5)
            inversion = (1.0 - slope) / 0.5
            return max(-1.0, min(0.0, -inversion))

    def build_full_signal(
        self,
        symbol: str,
        funding_signal: float,
        basis_signal: float,
        iv_skew_vol_pts: float,
        implied_vol: float,
        realized_vol: float,
        term_slope: Optional[float] = None,
    ) -> DerivativesSignal:
        """
        High-level entry point: accepts raw derivatives metrics and returns
        a fully processed composite DerivativesSignal.

        Parameters
        ----------
        symbol : str
        funding_signal : float
            Pre-computed funding signal in [-1, 1].
        basis_signal : float
            Pre-computed basis signal in [-1, 1].
        iv_skew_vol_pts : float
            Raw 25-delta skew in vol points (e.g. -0.05 = -5 vpts).
        implied_vol : float
            ATM implied vol (decimal).
        realized_vol : float
            Recent realized vol (decimal), same tenor.
        term_slope : float, optional
            30d/7d vol ratio. If provided, modifies the skew signal.

        Returns
        -------
        DerivativesSignal
        """
        regime = CryptoVolRegime()
        vrp_sig = regime.vrp_signal(symbol, implied_vol, realized_vol)
        skew_sig = self.skew_to_signal(iv_skew_vol_pts)

        if term_slope is not None:
            ts_sig = self.term_structure_to_signal(term_slope)
            # Blend skew and term structure with equal weight
            skew_sig = 0.5 * skew_sig + 0.5 * ts_sig

        return self.composite_derivatives_signal(
            symbol=symbol,
            funding_signal=funding_signal,
            basis_signal=basis_signal,
            skew_signal=skew_sig,
            vrp_signal=vrp_sig,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _vrp_ratio_to_signal(ratio: float) -> float:
    """
    Map IV/RV ratio to a [-1, 1] signal.

    ratio >= VRP_RICH_RATIO (1.5) -> -1.0 (expensive, sell vol)
    ratio <= VRP_CHEAP_RATIO (0.7) -> +1.0 (cheap, buy vol)
    Linear interpolation in between.
    """
    if ratio >= VRP_RICH_RATIO:
        return -1.0
    if ratio <= VRP_CHEAP_RATIO:
        return 1.0

    # Linear interpolation from cheap (0.7) to rich (1.5)
    normalized = (ratio - VRP_CHEAP_RATIO) / (VRP_RICH_RATIO - VRP_CHEAP_RATIO)
    return 1.0 - 2.0 * normalized


def _nearest_delta_quote(
    quotes: List[OptionQuote],
    target_delta: float,
) -> Optional[OptionQuote]:
    """Return the quote whose delta is closest to target_delta."""
    valid = [q for q in quotes if q.delta is not None]
    if not valid:
        return None
    return min(valid, key=lambda q: abs(q.delta - target_delta))


def _interpret_composite(
    symbol: str,
    composite: float,
    funding: float,
    basis: float,
    skew: float,
    vrp: float,
) -> str:
    """
    Generate a human-readable interpretation string for a composite signal.
    """
    if composite >= 0.5:
        stance = "strongly bullish"
    elif composite >= 0.2:
        stance = "mildly bullish"
    elif composite >= -0.2:
        stance = "neutral"
    elif composite >= -0.5:
        stance = "mildly bearish"
    else:
        stance = "strongly bearish"

    drivers = []
    if abs(funding) >= 0.3:
        direction = "bullish" if funding > 0 else "bearish"
        drivers.append(f"funding ({direction})")
    if abs(basis) >= 0.3:
        direction = "bullish" if basis > 0 else "bearish"
        drivers.append(f"basis ({direction})")
    if abs(skew) >= 0.3:
        direction = "call-rich" if skew > 0 else "put-rich"
        drivers.append(f"skew ({direction})")
    if abs(vrp) >= 0.3:
        direction = "cheap options" if vrp > 0 else "rich options"
        drivers.append(f"vrp ({direction})")

    driver_str = ", ".join(drivers) if drivers else "mixed signals"
    return f"{symbol} derivatives: {stance} (score={composite:.3f}) -- driven by {driver_str}"
