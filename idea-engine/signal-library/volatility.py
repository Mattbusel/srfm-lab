"""
idea-engine/signal-library/volatility.py
=========================================
10 volatility signals for the SRFM Idea Engine Signal Library.

Signals
-------
1.  GARCHVolForecast  — GARCH(1,1) online conditional volatility estimate
2.  RealizedVol       — rolling close-to-close historical volatility
3.  ParkinsonsVol     — high-low based volatility estimator (Parkinson 1980)
4.  GarmanKlassVol    — OHLC-based volatility estimator (Garman-Klass 1980)
5.  ATR               — Average True Range (normalised)
6.  VolRegime         — volatility percentile rank (1-100) over rolling window
7.  VolOfVol          — volatility of volatility
8.  VolCone           — multi-horizon vol vs current vol ratio
9.  VolBreakout       — vol spike detection (> N std devs above rolling mean)
10. VolMeanReversion  — vol's tendency to revert after spikes
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import Signal, SignalResult

logger = logging.getLogger(__name__)

_ANNUALISE_HOURLY = np.sqrt(365.25 * 24)
_ANNUALISE_DAILY  = np.sqrt(252)


def _annualisation_factor(df: pd.DataFrame) -> float:
    """
    Infer annualisation factor from DataFrame index frequency.
    Defaults to daily (√252) if frequency cannot be inferred.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return _ANNUALISE_DAILY
    try:
        freq = pd.infer_freq(df.index)
        if freq is None:
            # Estimate from median diff
            diffs = df.index.to_series().diff().dropna().dt.total_seconds()
            med   = diffs.median()
            if med <= 3700:       # ~1h
                return _ANNUALISE_HOURLY
            if med <= 900:        # ~15m
                return np.sqrt(365.25 * 24 * 4)
        else:
            if "H" in str(freq) or "h" in str(freq):
                return _ANNUALISE_HOURLY
            if "T" in str(freq) or "min" in str(freq):
                return np.sqrt(365.25 * 24 * 4)
    except Exception:
        pass
    return _ANNUALISE_DAILY


# ---------------------------------------------------------------------------
# 1. GARCHVolForecast
# ---------------------------------------------------------------------------

class GARCHVolForecast(Signal):
    """
    GARCH(1,1) conditional volatility via recursive EWMA.

    Model: h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}
    omega is calibrated so that the unconditional variance equals the
    long-run variance of the return series.

    Returns annualised volatility (σ, not variance).

    This matches the implementation in causal/python/feature_extractor.py.
    """

    name:        str = "garch_vol"
    category:    str = "volatility"
    lookback:    int = 60
    signal_type: str = "continuous"

    def __init__(
        self,
        alpha: float = 0.10,
        beta:  float = 0.85,
        warmup: int  = 60,
    ) -> None:
        if alpha + beta >= 1.0:
            raise ValueError("alpha + beta must be < 1 for GARCH stationarity.")
        self.alpha  = alpha
        self.beta   = beta
        self.lookback = warmup

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close   = df["Close"]
        returns = np.log(close / close.shift(1)).fillna(0.0)
        r2      = returns.values ** 2

        long_run_var = float(r2.mean()) if len(r2) > 1 else 1e-8
        omega        = max(long_run_var * (1.0 - self.alpha - self.beta), 1e-10)

        h = np.full(len(returns), long_run_var)
        vals = returns.values
        for t in range(1, len(vals)):
            h[t] = omega + self.alpha * vals[t - 1] ** 2 + self.beta * h[t - 1]

        ann_factor = _annualisation_factor(df)
        result     = pd.Series(np.sqrt(h) * ann_factor, index=df.index, name=self.name)
        return result


# ---------------------------------------------------------------------------
# 2. RealizedVol
# ---------------------------------------------------------------------------

class RealizedVol(Signal):
    """
    Rolling close-to-close realised volatility.

    Computed as the rolling standard deviation of log returns, annualised.
    """

    name:        str = "realized_vol"
    category:    str = "volatility"
    lookback:    int = 20
    signal_type: str = "continuous"

    def __init__(self, window: int = 20) -> None:
        self.window   = window
        self.lookback = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_ret    = np.log(df["Close"] / df["Close"].shift(1))
        ann_factor = _annualisation_factor(df)
        result     = log_ret.rolling(self.window, min_periods=2).std() * ann_factor
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 3. ParkinsonsVol
# ---------------------------------------------------------------------------

class ParkinsonsVol(Signal):
    """
    Parkinson (1980) high-low volatility estimator.

    More efficient than close-to-close vol when intraday data is available.
    Formula: sqrt[ (1 / (4 * ln(2))) * mean(ln(H/L)^2) ] * sqrt(N_periods)
    """

    name:        str = "parkinson_vol"
    category:    str = "volatility"
    lookback:    int = 20
    signal_type: str = "continuous"

    def __init__(self, window: int = 20) -> None:
        self.window   = window
        self.lookback = window

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        factor     = 1.0 / (4.0 * np.log(2.0))
        log_hl_sq  = (np.log(df["High"] / df["Low"].replace(0, np.nan))) ** 2
        roll_mean  = log_hl_sq.rolling(self.window, min_periods=2).mean()
        ann_factor = _annualisation_factor(df)
        result     = np.sqrt(factor * roll_mean) * ann_factor
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 4. GarmanKlassVol
# ---------------------------------------------------------------------------

class GarmanKlassVol(Signal):
    """
    Garman-Klass (1980) OHLC volatility estimator.

    Uses open, high, low, and close for a more efficient vol estimate.
    Formula:
        σ² = 0.5 * (ln H/L)² - (2*ln2 - 1) * (ln C/O)²
    Averaged over a rolling window, then annualised.
    """

    name:        str = "garman_klass_vol"
    category:    str = "volatility"
    lookback:    int = 20
    signal_type: str = "continuous"

    def __init__(self, window: int = 20) -> None:
        self.window   = window
        self.lookback = window

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ln_hl  = np.log(df["High"] / df["Low"].replace(0, np.nan))
        ln_co  = np.log(df["Close"] / df["Open"].replace(0, np.nan))

        gk_sq  = 0.5 * ln_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * ln_co ** 2
        # Clip to avoid sqrt of negative (shouldn't happen but be safe)
        gk_sq  = gk_sq.clip(lower=0.0)

        roll_mean  = gk_sq.rolling(self.window, min_periods=2).mean()
        ann_factor = _annualisation_factor(df)
        result     = np.sqrt(roll_mean) * ann_factor
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 5. ATR — Average True Range (normalised)
# ---------------------------------------------------------------------------

class ATR(Signal):
    """
    Wilder's Average True Range, normalised by closing price.

    Normalising by price makes ATR comparable across assets and time.
    Returns fractional ATR (e.g. 0.02 = 2% ATR).
    """

    name:        str = "atr"
    category:    str = "volatility"
    lookback:    int = 14
    signal_type: str = "continuous"

    def __init__(self, period: int = 14, normalise: bool = True) -> None:
        self.period    = period
        self.normalise = normalise
        self.lookback  = period

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        tr     = self._true_range(df)
        atr    = tr.ewm(com=self.period - 1, min_periods=1, adjust=False).mean()
        if self.normalise:
            atr = atr / df["Close"].replace(0, np.nan)
        atr.name = self.name
        return atr


# ---------------------------------------------------------------------------
# 6. VolRegime
# ---------------------------------------------------------------------------

class VolRegime(Signal):
    """
    Volatility regime percentile rank.

    Computes realised vol and then ranks it in its rolling history (0-100).
        > 80  high-vol regime
        < 20  low-vol regime
        50    median vol
    """

    name:        str = "vol_regime"
    category:    str = "volatility"
    lookback:    int = 252
    signal_type: str = "continuous"

    def __init__(self, vol_window: int = 20, rank_window: int = 252) -> None:
        self.vol_window  = vol_window
        self.rank_window = rank_window
        self.lookback    = max(vol_window, rank_window)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        rvol    = log_ret.rolling(self.vol_window, min_periods=2).std()
        result  = self._percentile_rank(rvol, self.rank_window)
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 7. VolOfVol
# ---------------------------------------------------------------------------

class VolOfVol(Signal):
    """
    Volatility of volatility (vol-of-vol).

    Measures the standard deviation of rolling realised volatility.
    High vol-of-vol indicates regime instability / clustering.
    """

    name:        str = "vol_of_vol"
    category:    str = "volatility"
    lookback:    int = 60
    signal_type: str = "continuous"

    def __init__(self, inner_window: int = 10, outer_window: int = 60) -> None:
        self.inner_window = inner_window
        self.outer_window = outer_window
        self.lookback     = inner_window + outer_window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        rvol    = log_ret.rolling(self.inner_window, min_periods=2).std()
        vov     = rvol.rolling(self.outer_window, min_periods=2).std()
        vov.name = self.name
        return vov


# ---------------------------------------------------------------------------
# 8. VolCone
# ---------------------------------------------------------------------------

class VolCone(Signal):
    """
    Volatility cone: ratio of short-horizon vol to long-horizon vol.

    A ratio > 1 means short-term vol exceeds long-term vol (vol spike).
    A ratio < 1 means short-term vol is below long-term vol (calm period).

    Useful for identifying whether the market is in an elevated vol environment
    relative to its long-run baseline.
    """

    name:        str = "vol_cone"
    category:    str = "volatility"
    lookback:    int = 60
    signal_type: str = "continuous"

    def __init__(self, short_window: int = 5, long_window: int = 60) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window.")
        self.short_window = short_window
        self.long_window  = long_window
        self.lookback     = long_window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_ret   = np.log(df["Close"] / df["Close"].shift(1))
        short_vol = log_ret.rolling(self.short_window, min_periods=2).std()
        long_vol  = log_ret.rolling(self.long_window,  min_periods=2).std()
        result    = short_vol / long_vol.replace(0.0, np.nan)
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 9. VolBreakout
# ---------------------------------------------------------------------------

class VolBreakout(Signal):
    """
    Volatility spike (breakout) detection.

    Returns a binary signal (0 or 1):
        1  if current realised vol is > N std deviations above its rolling mean
        0  otherwise

    High vol breakouts often precede directional moves or reversals.
    """

    name:        str = "vol_breakout"
    category:    str = "volatility"
    lookback:    int = 60
    signal_type: str = "binary"

    def __init__(
        self,
        vol_window:   int   = 10,
        rank_window:  int   = 60,
        n_std:        float = 2.0,
    ) -> None:
        self.vol_window  = vol_window
        self.rank_window = rank_window
        self.n_std       = n_std
        self.lookback    = vol_window + rank_window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_ret  = np.log(df["Close"] / df["Close"].shift(1))
        rvol     = log_ret.rolling(self.vol_window, min_periods=2).std()
        roll_mu  = rvol.rolling(self.rank_window, min_periods=2).mean()
        roll_std = rvol.rolling(self.rank_window, min_periods=2).std()
        z        = (rvol - roll_mu) / roll_std.replace(0.0, np.nan)
        result   = (z > self.n_std).astype(float)
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 10. VolMeanReversion
# ---------------------------------------------------------------------------

class VolMeanReversion(Signal):
    """
    Volatility mean-reversion signal.

    After a vol spike (VolBreakout), vol tends to revert to its mean.
    This signal estimates the reversion potential:

        signal = (rolling_max_vol - current_vol) / rolling_max_vol

    High values → vol has spiked and is now declining (vol selling signal).
    Low/negative values → vol is near its rolling high (vol still elevated).
    """

    name:        str = "vol_mean_reversion"
    category:    str = "volatility"
    lookback:    int = 60
    signal_type: str = "continuous"

    def __init__(self, vol_window: int = 10, reversion_window: int = 60) -> None:
        self.vol_window        = vol_window
        self.reversion_window  = reversion_window
        self.lookback          = vol_window + reversion_window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_ret  = np.log(df["Close"] / df["Close"].shift(1))
        rvol     = log_ret.rolling(self.vol_window, min_periods=2).std()
        max_vol  = rvol.rolling(self.reversion_window, min_periods=2).max()
        result   = (max_vol - rvol) / max_vol.replace(0.0, np.nan)
        result   = result.clip(0.0, 1.0)
        result.name = self.name
        return result
