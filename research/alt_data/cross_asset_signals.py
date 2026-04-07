"""
cross_asset_signals.py -- Cross-asset signals that inform crypto positioning.

Crypto does not trade in isolation. Its behavior is strongly influenced by
macro liquidity conditions, dollar strength, equity risk appetite, and
inter-asset correlations. This module provides quantitative signals derived
from cross-asset relationships.

Modules:
  EquityCryptoCorrelation -- Rolling BTC correlation to SPX, NDX, Gold, DXY
  DollarCycleSignal       -- DXY trend as crypto headwind/tailwind gauge
  RateImpact              -- Fed rate regime classification and crypto impact
  CrossAssetMomentum      -- 1/vol weighted momentum across asset classes
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RateRegime(Enum):
    """Federal Reserve interest rate cycle regime."""
    HIKING  = "HIKING"    # Fed actively raising rates
    CUTTING = "CUTTING"   # Fed actively cutting rates
    PAUSED  = "PAUSED"    # Hike/cut cycle on hold
    NEUTRAL = "NEUTRAL"   # Rates near neutral estimate, no clear bias


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return statistics.mean(values)


def _safe_stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 1.0
    return statistics.stdev(values)


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """
    Compute Pearson correlation coefficient between x and y.

    Returns 0.0 if the arrays are too short or have zero variance.
    Both arrays must be the same length.
    """
    n = len(x)
    if n < 3 or len(y) != n:
        return 0.0
    mx = statistics.mean(x)
    my = statistics.mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx  = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy  = math.sqrt(sum((yi - my) ** 2 for yi in y))
    denom = sx * sy
    if denom < 1e-12:
        return 0.0
    return num / denom


def _returns(prices: List[float]) -> List[float]:
    """Compute simple percentage returns from a price series."""
    if len(prices) < 2:
        return []
    return [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]


def _annualized_vol(returns: List[float], periods_per_year: int = 252) -> float:
    """Compute annualized volatility from a returns series."""
    if len(returns) < 2:
        return 1.0
    std = statistics.stdev(returns)
    return std * math.sqrt(periods_per_year)


def _momentum(prices: List[float], window: int) -> float:
    """
    Compute price momentum as the return over the last `window` periods.
    Returns 0.0 if insufficient data.
    """
    if len(prices) < window + 1:
        return 0.0
    old = prices[-(window + 1)]
    new = prices[-1]
    if old < 1e-10:
        return 0.0
    return (new - old) / old


# ---------------------------------------------------------------------------
# EquityCryptoCorrelation
# ---------------------------------------------------------------------------

class EquityCryptoCorrelation:
    """
    Computes rolling Pearson correlations between BTC and other major assets.

    Assets tracked by default:
      SPX  -- S&P 500 (risk-on barometer)
      NDX  -- Nasdaq 100 (tech/growth proxy)
      GLD  -- Gold spot (safe-haven / inflation hedge)
      DXY  -- USD index (dollar funding conditions)
      TLT  -- US 20yr Treasury ETF (duration / rates)

    A high BTC-SPX correlation suggests BTC is trading as a risk asset.
    A high BTC-GLD correlation suggests BTC is acting as a safe haven.
    A high BTC-DXY (negative) correlation signals dollar headwinds.

    Parameters
    ----------
    window : int
        Rolling window length for correlation computation. Default 30.
    history_window : int
        Total price history to retain. Default 252.
    """

    ASSETS = ["SPX", "NDX", "GLD", "DXY", "TLT"]

    def __init__(self, window: int = 30, history_window: int = 252) -> None:
        self.window = window
        self.history_window = history_window
        self._btc_prices:  Deque[float] = deque(maxlen=history_window)
        self._asset_prices: Dict[str, Deque[float]] = {
            asset: deque(maxlen=history_window) for asset in self.ASSETS
        }
        # Correlation history for trend analysis
        self._corr_history: Dict[str, Deque[float]] = {
            asset: deque(maxlen=history_window) for asset in self.ASSETS
        }

    def update(self, btc_price: float, asset_prices: Dict[str, float]) -> None:
        """
        Record a new price observation for BTC and companion assets.

        Parameters
        ----------
        btc_price     : BTC spot price in USD.
        asset_prices  : Dict mapping asset names to current prices.
                        Keys should match ASSETS list (SPX, NDX, GLD, DXY, TLT).
        """
        self._btc_prices.append(btc_price)
        for asset, price in asset_prices.items():
            if asset in self._asset_prices:
                self._asset_prices[asset].append(price)
        # Recompute and store correlations
        for asset in self.ASSETS:
            if asset in asset_prices:
                corr = self._compute_correlation(asset)
                if corr is not None:
                    self._corr_history[asset].append(corr)

    def _compute_correlation(self, asset: str) -> Optional[float]:
        """Compute rolling Pearson correlation between BTC returns and asset returns."""
        btc_prices   = list(self._btc_prices)
        asset_prices = list(self._asset_prices[asset])
        n = min(len(btc_prices), len(asset_prices), self.window + 1)
        if n < 4:
            return None
        btc_ret   = _returns(btc_prices[-n:])
        asset_ret = _returns(asset_prices[-n:])
        m = min(len(btc_ret), len(asset_ret))
        if m < 3:
            return None
        return _pearson_correlation(btc_ret[-m:], asset_ret[-m:])

    def get_correlation(self, asset: str) -> float:
        """
        Return the most recent rolling correlation between BTC and `asset`.
        Returns 0.0 if insufficient data.
        """
        corr = self._compute_correlation(asset)
        return corr if corr is not None else 0.0

    def all_correlations(self) -> Dict[str, float]:
        """Return a dict of current BTC-asset correlations for all tracked assets."""
        return {asset: self.get_correlation(asset) for asset in self.ASSETS}

    def risk_on_signal(self) -> float:
        """
        Compute a risk-on composite signal in [-1, +1].

        Logic:
          When BTC-SPX and BTC-NDX correlations are both high and positive,
          BTC is behaving as a risk asset moving in lockstep with equities.
          This means both assets are responding to the same liquidity conditions
          -- positive equity momentum implies positive crypto momentum.

          Signal = weighted average of BTC-SPX and BTC-NDX correlations.
          High positive correlation + positive equity momentum -> risk-on bullish.
        """
        spx_corr = self.get_correlation("SPX")
        ndx_corr = self.get_correlation("NDX")
        weighted = 0.5 * spx_corr + 0.5 * ndx_corr
        return _clamp(weighted)

    def safe_haven_signal(self) -> float:
        """
        Compute a safe-haven signal indicating whether BTC is trading like gold.

        High BTC-GLD correlation + low BTC-SPX correlation -> BTC acting defensively.
        This regime (though uncommon) suggests BTC could benefit from risk-off flows
        that typically support gold.

        Returns a score in [0, 1] where 1.0 = strong safe-haven characteristics.
        """
        gld_corr = self.get_correlation("GLD")
        spx_corr = self.get_correlation("SPX")
        # Safe haven: high gold correlation, low equity correlation
        haven_score = gld_corr - max(spx_corr, 0.0)
        return _clamp(haven_score, 0.0, 1.0)

    def dxy_drag_signal(self) -> float:
        """
        Return the DXY correlation as a potential headwind indicator.

        Typically BTC and DXY are negatively correlated: strong dollar hurts BTC.
        Returns a value in [-1, +1]:
          Negative DXY correlation is NORMAL -- returns close to 0 or slightly positive.
          Strongly negative (e.g. -0.7) means DXY is a major headwind.
        """
        dxy_corr = self.get_correlation("DXY")
        # Invert: very negative correlation with DXY -> risk to BTC
        return -dxy_corr

    def correlation_trend(self, asset: str, lookback: int = 10) -> float:
        """
        Return the change in correlation over the last `lookback` periods.
        Positive = correlation is rising, Negative = correlation is falling.
        """
        hist = list(self._corr_history.get(asset, []))
        if len(hist) < lookback + 1:
            return 0.0
        return hist[-1] - hist[-(lookback + 1)]

    def regime_classification(self) -> str:
        """
        Classify BTC's current inter-asset regime based on correlations.

        Returns one of:
          RISK_ASSET    -- high SPX/NDX correlation (most common)
          SAFE_HAVEN    -- high GLD correlation, low equity correlation
          DECORRELATED  -- low correlation with all assets (BTC-native drivers)
          DOLLAR_DRIVEN -- high negative DXY correlation dominating
        """
        spx_corr = self.get_correlation("SPX")
        gld_corr = self.get_correlation("GLD")
        dxy_corr = self.get_correlation("DXY")

        if abs(dxy_corr) > 0.6 and abs(dxy_corr) > abs(spx_corr):
            return "DOLLAR_DRIVEN"
        if spx_corr > 0.5:
            return "RISK_ASSET"
        if gld_corr > 0.4 and spx_corr < 0.3:
            return "SAFE_HAVEN"
        return "DECORRELATED"


# ---------------------------------------------------------------------------
# DollarCycleSignal
# ---------------------------------------------------------------------------

class DollarCycleSignal:
    """
    Derives a crypto headwind/tailwind signal from the US Dollar Index (DXY) trend.

    The USD and crypto have a well-documented inverse relationship:
      Strong dollar -> tight global liquidity -> risk asset headwind
      Weak dollar   -> loose global liquidity -> crypto tailwind

    Mechanisms:
      1. Dollar funding: many crypto leveraged positions are dollar-denominated.
      2. EM flows: strong dollar drains EM capital that might otherwise flow to crypto.
      3. Global M2: dollar strength tightens global money supply.

    Parameters
    ----------
    short_window  : int
        Fast EMA/momentum window. Default 10.
    long_window   : int
        Slow EMA/momentum window. Default 50.
    history_window : int
        Total history retained. Default 252.
    """

    def __init__(
        self,
        short_window: int  = 10,
        long_window:  int  = 50,
        history_window: int = 252,
    ) -> None:
        self.short_window  = short_window
        self.long_window   = long_window
        self._dxy_prices: Deque[float] = deque(maxlen=history_window)
        self._signal_history: Deque[float] = deque(maxlen=history_window)

    def update(self, dxy_level: float) -> None:
        """Record a new DXY level observation."""
        self._dxy_prices.append(dxy_level)
        signal = self.get_dollar_headwind_from_history()
        self._signal_history.append(signal)

    def get_dollar_headwind(self, dxy_momentum: float) -> float:
        """
        Compute a crypto headwind score from a pre-computed DXY momentum value.

        Parameters
        ----------
        dxy_momentum : DXY price change over some lookback (e.g., 20-day return).
                       Positive = dollar strengthening.

        Returns
        -------
        float in [-1, +1]:
          -1.0 -> strong dollar headwind (bad for crypto)
          +1.0 -> strong dollar tailwind (good for crypto -- DXY weakening)
        """
        # Invert: positive DXY momentum -> negative for crypto
        return _clamp(-dxy_momentum * 10.0)  # scale: 10% DXY move -> full signal

    def get_dollar_headwind_from_history(self) -> float:
        """
        Compute the dollar headwind signal using the stored DXY price history.

        Combines:
          1. Short-term momentum (fast signal): 10-period return
          2. Long-term trend (slow signal): 50-period return
          3. Trend direction: is dollar making new highs?
        """
        prices = list(self._dxy_prices)
        if len(prices) < 5:
            return 0.0

        short_mom = _momentum(prices, min(self.short_window, len(prices) - 1))
        long_mom  = _momentum(prices, min(self.long_window,  len(prices) - 1))

        # Both signals inverted: dollar rising -> negative for crypto
        short_signal = _clamp(-short_mom * 15.0)
        long_signal  = _clamp(-long_mom  * 8.0)

        # Weighted composite (short-term dominates for trading signals)
        composite = 0.6 * short_signal + 0.4 * long_signal
        return _clamp(composite)

    def dxy_trend(self) -> str:
        """
        Classify current DXY trend.

        Returns one of: STRENGTHENING, WEAKENING, CONSOLIDATING
        """
        prices = list(self._dxy_prices)
        if len(prices) < self.short_window + 1:
            return "CONSOLIDATING"
        short_mom = _momentum(prices, self.short_window)
        if short_mom > 0.01:
            return "STRENGTHENING"
        if short_mom < -0.01:
            return "WEAKENING"
        return "CONSOLIDATING"

    def dxy_at_extreme(self) -> bool:
        """
        Return True if DXY is at a multi-year extreme (above 110 or below 90),
        which typically coincides with structural inflection points.
        """
        if not self._dxy_prices:
            return False
        current = self._dxy_prices[-1]
        return current > 110.0 or current < 90.0

    def rolling_headwind(self, window: int = 20) -> float:
        """Return the average headwind signal over the last `window` periods."""
        sig_list = list(self._signal_history)
        tail = sig_list[-window:] if len(sig_list) >= window else sig_list
        if not tail:
            return 0.0
        return _safe_mean(tail)

    def to_dict(self) -> Dict[str, float]:
        """Return a summary dict of DXY signal metrics."""
        if not self._dxy_prices:
            return {}
        current_dxy = self._dxy_prices[-1]
        return {
            "dxy_level":          current_dxy,
            "headwind_signal":    self.get_dollar_headwind_from_history(),
            "rolling_headwind_20d": self.rolling_headwind(20),
            "dxy_at_extreme":     float(self.dxy_at_extreme()),
        }


# ---------------------------------------------------------------------------
# RateImpact
# ---------------------------------------------------------------------------

class RateImpact:
    """
    Classifies the Federal Reserve rate regime and quantifies its impact
    on crypto risk appetite.

    Crypto is highly sensitive to the Fed's rate cycle:
      Hiking cycle   -> liquidity tightening -> risk assets sold
      Cutting cycle  -> liquidity expansion  -> risk assets bid
      Paused         -> uncertainty resolved -> moderate tailwind
      Neutral        -> rates near neutral r* -> minimal impact

    Parameters
    ----------
    history_window : int
        Number of rate observations to retain. Default 252.
    neutral_rate_est : float
        Estimate of the long-run neutral Fed Funds Rate. Default 2.5%.
        Used to determine whether current rates are restrictive or accommodative.
    """

    # Number of consecutive hikes/cuts before classifying as a cycle
    CYCLE_CONFIRMATION = 2

    def __init__(
        self,
        history_window: int   = 252,
        neutral_rate_est: float = 2.5,
    ) -> None:
        self.history_window  = history_window
        self.neutral_rate    = neutral_rate_est
        self._rate_history:  Deque[float] = deque(maxlen=history_window)
        self._regime_history: Deque[RateRegime] = deque(maxlen=history_window)

    def update(self, fed_funds_rate: float) -> None:
        """
        Record a new Fed Funds Rate observation.

        Parameters
        ----------
        fed_funds_rate : Effective Fed Funds Rate in percent (e.g. 5.25 = 5.25%).
        """
        self._rate_history.append(fed_funds_rate)
        regime = self._classify_regime()
        self._regime_history.append(regime)

    def _classify_regime(self) -> RateRegime:
        """Internal regime classification from stored rate history."""
        rates = list(self._rate_history)
        if len(rates) < self.CYCLE_CONFIRMATION + 1:
            return RateRegime.NEUTRAL

        recent = rates[-(self.CYCLE_CONFIRMATION + 1):]
        changes = [recent[i] - recent[i - 1] for i in range(1, len(recent))]

        all_hikes = all(c > 0.01 for c in changes)  # 1bp minimum change per hike
        all_cuts  = all(c < -0.01 for c in changes)
        no_change = all(abs(c) < 0.01 for c in changes)

        if all_hikes:
            return RateRegime.HIKING
        if all_cuts:
            return RateRegime.CUTTING
        if no_change:
            return RateRegime.PAUSED
        return RateRegime.NEUTRAL

    def get_rate_regime(self) -> str:
        """
        Return the current Fed rate regime as a string.

        Returns one of: HIKING, CUTTING, NEUTRAL, PAUSED
        """
        if not self._regime_history:
            return RateRegime.NEUTRAL.value
        return self._regime_history[-1].value

    def get_rate_regime_enum(self) -> RateRegime:
        """Return the current regime as a RateRegime enum."""
        if not self._regime_history:
            return RateRegime.NEUTRAL
        return self._regime_history[-1]

    def current_rate(self) -> float:
        """Return the most recently recorded Fed Funds Rate."""
        if not self._rate_history:
            raise ValueError("No rate data -- call update() first.")
        return self._rate_history[-1]

    def is_restrictive(self) -> bool:
        """Return True if current rate is above the neutral rate estimate."""
        return self.current_rate() > self.neutral_rate

    def crypto_impact_score(self) -> float:
        """
        Return a score in [-1, +1] representing the rate regime's impact on crypto.

        Interpretation:
          +1.0 -> strongly bullish rate environment (cutting cycle + low rates)
          -1.0 -> strongly bearish rate environment (hiking cycle + high rates)
        """
        regime = self.get_rate_regime_enum()
        base_scores = {
            RateRegime.CUTTING: 0.7,
            RateRegime.PAUSED:  0.2,
            RateRegime.NEUTRAL: 0.0,
            RateRegime.HIKING:  -0.7,
        }
        base = base_scores[regime]

        # Adjust for absolute rate level (high absolute rates = extra negative)
        if self._rate_history:
            rate_level = self.current_rate()
            level_penalty = _clamp(-(rate_level - self.neutral_rate) / 10.0)
            return _clamp(base + 0.3 * level_penalty)
        return base

    def rate_change_velocity(self, lookback: int = 10) -> float:
        """
        Return the average rate change per period over the last `lookback` periods.
        Positive = rates rising, Negative = rates falling.
        """
        rates = list(self._rate_history)
        if len(rates) < lookback + 1:
            return 0.0
        old = rates[-(lookback + 1)]
        new = rates[-1]
        return (new - old) / lookback

    def hike_cycle_duration(self) -> int:
        """
        Return the number of consecutive periods the hiking regime has been active.
        Returns 0 if not currently in a hiking cycle.
        """
        regimes = list(self._regime_history)
        if not regimes or regimes[-1] != RateRegime.HIKING:
            return 0
        count = 0
        for r in reversed(regimes):
            if r == RateRegime.HIKING:
                count += 1
            else:
                break
        return count

    def to_dict(self) -> Dict[str, float]:
        """Return a summary dict of rate regime metrics."""
        if not self._rate_history:
            return {}
        return {
            "fed_funds_rate":      self.current_rate(),
            "neutral_rate_est":    self.neutral_rate,
            "is_restrictive":      float(self.is_restrictive()),
            "crypto_impact_score": self.crypto_impact_score(),
            "rate_velocity_10d":   self.rate_change_velocity(10),
            "hike_cycle_duration": float(self.hike_cycle_duration()),
        }


# ---------------------------------------------------------------------------
# CrossAssetMomentum
# ---------------------------------------------------------------------------

class CrossAssetMomentum:
    """
    Computes momentum-based portfolio signals across multiple asset classes
    simultaneously, with inverse-volatility (1/vol) position sizing.

    The strategy:
      1. Compute N-period momentum for each asset.
      2. Weight assets by 1/annualized_vol (inverse volatility).
      3. Identify the top K momentum assets.
      4. Normalize weights to sum to 1.0.

    This is a simplified version of the AQR time-series momentum (TSMOM)
    framework applied across crypto and macro assets.

    Tracked assets by default: SPX, NDX, GLD, TLT, BTC

    Parameters
    ----------
    assets       : List of asset names to track.
    window       : Momentum lookback window. Default 20 periods.
    top_k        : Number of assets to go long in the momentum portfolio. Default 3.
    history_window : Price history to retain per asset. Default 252.
    """

    DEFAULT_ASSETS = ["SPX", "NDX", "GLD", "TLT", "BTC"]

    def __init__(
        self,
        assets:         Optional[List[str]] = None,
        window:         int = 20,
        top_k:          int = 3,
        history_window: int = 252,
    ) -> None:
        self.assets         = assets or self.DEFAULT_ASSETS
        self.window         = window
        self.top_k          = top_k
        self.history_window = history_window
        self._prices: Dict[str, Deque[float]] = {
            asset: deque(maxlen=history_window) for asset in self.assets
        }

    def update(self, prices: Dict[str, float]) -> None:
        """
        Record new price observations.

        Parameters
        ----------
        prices : Dict mapping asset names to current prices.
                 Keys should match the `assets` list.
        """
        for asset, price in prices.items():
            if asset in self._prices:
                self._prices[asset].append(price)

    def _asset_momentum(self, asset: str, window: Optional[int] = None) -> float:
        """Compute price momentum for a single asset over `window` periods."""
        w = window or self.window
        prices = list(self._prices[asset])
        return _momentum(prices, w)

    def _asset_vol(self, asset: str, vol_window: int = 20) -> float:
        """Compute annualized return volatility for a single asset."""
        prices = list(self._prices[asset])
        if len(prices) < vol_window + 1:
            return 1.0  # fallback to prevent division by zero
        recent = prices[-(vol_window + 1):]
        rets   = _returns(recent)
        if len(rets) < 2:
            return 1.0
        return max(_annualized_vol(rets), 1e-4)

    def get_cross_asset_signal(
        self,
        prices: Dict[str, float],
        window: int = 20,
    ) -> Dict[str, float]:
        """
        Compute the cross-asset momentum signal for the given price snapshot.

        Parameters
        ----------
        prices : Current price dict (same keys as self.assets).
        window : Momentum lookback. Default 20.

        Returns
        -------
        dict with keys:
          "<asset>_momentum"    : Raw momentum for each asset.
          "<asset>_weight"      : 1/vol normalized weight.
          "<asset>_in_portfolio": 1.0 if asset is in top_k momentum portfolio.
          "btc_in_portfolio"    : Whether BTC is in the long portfolio.
          "btc_weight"          : BTC's portfolio weight (0 if not in portfolio).
          "portfolio_score"     : Momentum-weighted portfolio return forecast.
        """
        self.update(prices)

        # Compute momentum and vol for all assets
        momenta: Dict[str, float] = {}
        vols:    Dict[str, float] = {}
        for asset in self.assets:
            if len(self._prices[asset]) < window + 2:
                momenta[asset] = 0.0
                vols[asset]    = 1.0
            else:
                momenta[asset] = self._asset_momentum(asset, window)
                vols[asset]    = self._asset_vol(asset, min(window, 20))

        # 1/vol weights (raw)
        inv_vols = {asset: 1.0 / vols[asset] for asset in self.assets}
        total_inv_vol = sum(inv_vols.values())
        if total_inv_vol < 1e-10:
            total_inv_vol = 1.0
        vol_weights = {asset: inv_vols[asset] / total_inv_vol for asset in self.assets}

        # Rank assets by momentum, select top_k
        sorted_assets = sorted(self.assets, key=lambda a: momenta[a], reverse=True)
        top_assets    = set(sorted_assets[: self.top_k])

        # Portfolio weights: only top_k assets, renormalized
        portfolio_inv_vol = {a: inv_vols[a] for a in top_assets}
        total_portfolio_iv = sum(portfolio_inv_vol.values())
        if total_portfolio_iv < 1e-10:
            total_portfolio_iv = 1.0
        portfolio_weights = {
            asset: (portfolio_inv_vol[asset] / total_portfolio_iv if asset in top_assets else 0.0)
            for asset in self.assets
        }

        # Portfolio-level momentum score: weighted sum of momenta
        portfolio_score = sum(
            portfolio_weights[asset] * momenta[asset]
            for asset in self.assets
        )

        # Build result dict
        result: Dict[str, float] = {}
        for asset in self.assets:
            result[f"{asset}_momentum"]     = momenta[asset]
            result[f"{asset}_vol_weight"]   = vol_weights[asset]
            result[f"{asset}_portfolio_wt"] = portfolio_weights[asset]
            result[f"{asset}_in_portfolio"] = float(asset in top_assets)

        result["portfolio_score"] = portfolio_score
        result["btc_in_portfolio"] = float("BTC" in top_assets)
        result["btc_weight"]       = portfolio_weights.get("BTC", 0.0)

        return result

    def btc_relative_momentum(self, window: Optional[int] = None) -> float:
        """
        Return BTC's momentum relative to the asset-class average momentum.

        Positive -> BTC outperforming the cross-asset universe (bullish).
        Negative -> BTC underperforming (bearish rotation signal).
        """
        w = window or self.window
        valid_assets = [a for a in self.assets if len(self._prices[a]) >= w + 2]
        if not valid_assets:
            return 0.0
        btc_mom = self._asset_momentum("BTC", w) if "BTC" in valid_assets else 0.0
        others  = [self._asset_momentum(a, w) for a in valid_assets if a != "BTC"]
        if not others:
            return 0.0
        avg_other = _safe_mean(others)
        return _clamp(btc_mom - avg_other)

    def momentum_dispersion(self, window: Optional[int] = None) -> float:
        """
        Return the standard deviation of momentum across all assets.

        High dispersion -> large divergence in asset performances -> useful
        for regime detection (high dispersion often coincides with stress).
        """
        w = window or self.window
        valid_assets = [a for a in self.assets if len(self._prices[a]) >= w + 2]
        if len(valid_assets) < 2:
            return 0.0
        momenta = [self._asset_momentum(a, w) for a in valid_assets]
        return statistics.stdev(momenta)

    def correlation_matrix(self, window: Optional[int] = None) -> Dict[Tuple[str, str], float]:
        """
        Compute pairwise return correlations across all tracked assets.

        Returns a dict mapping (asset_a, asset_b) -> Pearson correlation.
        Diagonal entries (same asset) are 1.0.
        """
        w = window or self.window
        corr_matrix: Dict[Tuple[str, str], float] = {}
        for i, a in enumerate(self.assets):
            for j, b in enumerate(self.assets):
                if i == j:
                    corr_matrix[(a, b)] = 1.0
                    continue
                pa = list(self._prices[a])
                pb = list(self._prices[b])
                n  = min(len(pa), len(pb), w + 1)
                if n < 4:
                    corr_matrix[(a, b)] = 0.0
                    continue
                ra = _returns(pa[-n:])
                rb = _returns(pb[-n:])
                m  = min(len(ra), len(rb))
                corr_matrix[(a, b)] = _pearson_correlation(ra[-m:], rb[-m:])
        return corr_matrix

    def risk_parity_weights(self, vol_window: int = 20) -> Dict[str, float]:
        """
        Compute full risk parity (1/vol) weights across all assets.

        Unlike the portfolio weights (which only allocate to top_k assets),
        risk parity allocates to all assets proportional to their inverse volatility.

        Returns a dict of normalized weights summing to 1.0.
        """
        vols = {}
        for asset in self.assets:
            vols[asset] = self._asset_vol(asset, vol_window)
        inv_vols    = {a: 1.0 / v for a, v in vols.items()}
        total_iv    = sum(inv_vols.values())
        if total_iv < 1e-10:
            n = len(self.assets)
            return {a: 1.0 / n for a in self.assets}
        return {a: inv_vols[a] / total_iv for a in self.assets}
