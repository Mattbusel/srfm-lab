"""
idea-engine/signal-library/microstructure.py
=============================================
6 market microstructure signals for the SRFM Idea Engine Signal Library.

Signals
-------
1.  VolumeWeightedMomentum   — price change × volume weight
2.  VolumeSpike              — volume vs rolling average (>2× = spike)
3.  BuyPressure              — close position within high-low range (Stoch %K)
4.  OBV                      — On-Balance Volume
5.  VPT                      — Volume-Price Trend
6.  VolumeMomentumDivergence — price up but volume declining
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import Signal, SignalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. VolumeWeightedMomentum
# ---------------------------------------------------------------------------

class VolumeWeightedMomentum(Signal):
    """
    Volume-Weighted Momentum: price change scaled by volume weight.

    For each bar:
        vwm = log(Close/Close[-1]) × (Volume / rolling_avg_volume)

    Then summed over N bars for a cumulative measure.

    High positive values: strong up-moves confirmed by volume (bullish).
    High negative values: strong down-moves confirmed by volume (bearish).
    """

    name:        str = "vw_momentum"
    category:    str = "microstructure"
    lookback:    int = 20
    signal_type: str = "continuous"

    def __init__(self, period: int = 20, vol_avg_window: int = 20) -> None:
        self.period         = period
        self.vol_avg_window = vol_avg_window
        self.lookback       = max(period, vol_avg_window)

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_ret  = np.log(df["Close"] / df["Close"].shift(1))
        avg_vol  = df["Volume"].rolling(self.vol_avg_window, min_periods=2).mean()
        vol_wt   = df["Volume"] / avg_vol.replace(0.0, np.nan)
        vwm_bar  = log_ret * vol_wt
        result   = vwm_bar.rolling(self.period, min_periods=1).sum()
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 2. VolumeSpike
# ---------------------------------------------------------------------------

class VolumeSpike(Signal):
    """
    Volume Spike detector: ratio of current volume to rolling average volume.

    Returns the volume ratio (e.g. 2.5 = 2.5× average volume).
        > 2.0  volume spike (significant activity)
        1.0    average volume
        < 0.5  low-volume / illiquid

    Binary variant available via the ``binary`` flag (returns 1 if ratio > threshold).
    """

    name:        str = "volume_spike"
    category:    str = "microstructure"
    lookback:    int = 20
    signal_type: str = "continuous"

    def __init__(
        self,
        window:    int   = 20,
        threshold: float = 2.0,
        binary:    bool  = False,
    ) -> None:
        self.window    = window
        self.threshold = threshold
        self.binary    = binary
        self.lookback  = window

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        avg_vol = df["Volume"].rolling(self.window, min_periods=2).mean()
        ratio   = df["Volume"] / avg_vol.replace(0.0, np.nan)

        if self.binary:
            result = (ratio > self.threshold).astype(float)
        else:
            result = ratio

        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 3. BuyPressure
# ---------------------------------------------------------------------------

class BuyPressure(Signal):
    """
    Buy Pressure: position of close within the high-low range.

    Equivalent to Stochastic %K (without smoothing):
        BuyPressure = (Close - Low_N) / (High_N - Low_N)

    Range: 0 (close at N-bar low) to 1 (close at N-bar high).
    Centred at 0.5 — re-scaled to [-1, +1] for consistency.

    High positive values → close near high end of range (bullish pressure).
    High negative values → close near low end of range (bearish / selling).
    """

    name:        str = "buy_pressure"
    category:    str = "microstructure"
    lookback:    int = 14
    signal_type: str = "continuous"

    def __init__(self, period: int = 14) -> None:
        self.period   = period
        self.lookback = period

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        high_n   = df["High"].rolling(self.period, min_periods=1).max()
        low_n    = df["Low"].rolling(self.period,  min_periods=1).min()
        range_n  = (high_n - low_n).replace(0.0, np.nan)
        stoch_k  = (df["Close"] - low_n) / range_n
        # Rescale from [0,1] → [-1,+1]
        result   = (stoch_k - 0.5) * 2.0
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 4. OBV — On-Balance Volume
# ---------------------------------------------------------------------------

class OBV(Signal):
    """
    On-Balance Volume (Granville 1963).

    Accumulates volume on up-bars and subtracts it on down-bars.
    The running total reveals whether volume is flowing into or out of an asset.

    Raw OBV grows with time (unbounded). For signal use, the z-score of OBV
    over a rolling window is returned (normalised OBV divergence).

    Set ``normalise=False`` to return raw OBV.
    """

    name:        str = "obv"
    category:    str = "microstructure"
    lookback:    int = 30
    signal_type: str = "continuous"

    def __init__(self, normalise: bool = True, norm_window: int = 30) -> None:
        self.normalise   = normalise
        self.norm_window = norm_window
        self.lookback    = norm_window if normalise else 2

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        direction = np.sign(df["Close"].diff().fillna(0.0))
        obv_vals  = (direction * df["Volume"]).cumsum()

        if self.normalise:
            mu     = obv_vals.rolling(self.norm_window, min_periods=2).mean()
            std    = obv_vals.rolling(self.norm_window, min_periods=2).std()
            result = (obv_vals - mu) / std.replace(0.0, np.nan)
        else:
            result = obv_vals

        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 5. VPT — Volume-Price Trend
# ---------------------------------------------------------------------------

class VPT(Signal):
    """
    Volume-Price Trend (VPT).

    VPT_t = VPT_{t-1} + Volume_t × (Close_t - Close_{t-1}) / Close_{t-1}

    Captures the amount of volume flowing in the direction of price change.
    Similar to OBV but weights volume by the magnitude of price change.

    Returned as z-score over rolling window (``normalise=True``) or raw.
    """

    name:        str = "vpt"
    category:    str = "microstructure"
    lookback:    int = 30
    signal_type: str = "continuous"

    def __init__(self, normalise: bool = True, norm_window: int = 30) -> None:
        self.normalise   = normalise
        self.norm_window = norm_window
        self.lookback    = norm_window if normalise else 2

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        price_chg = df["Close"].pct_change().fillna(0.0)
        vpt_bar   = df["Volume"] * price_chg
        vpt_vals  = vpt_bar.cumsum()

        if self.normalise:
            mu     = vpt_vals.rolling(self.norm_window, min_periods=2).mean()
            std    = vpt_vals.rolling(self.norm_window, min_periods=2).std()
            result = (vpt_vals - mu) / std.replace(0.0, np.nan)
        else:
            result = vpt_vals

        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 6. VolumeMomentumDivergence
# ---------------------------------------------------------------------------

class VolumeMomentumDivergence(Signal):
    """
    Volume-Momentum Divergence: detects when price is rising but volume is
    declining (bearish divergence) or price is falling but volume is rising
    (potential selling climax / bullish divergence).

    Returns:
        +1  Bullish divergence: price falling, volume rising (selling climax)
        -1  Bearish divergence: price rising, volume falling (weak rally)
         0  No divergence

    Uses N-bar ROC for both price and volume to assess their relative trends.
    """

    name:        str = "volume_momentum_divergence"
    category:    str = "microstructure"
    lookback:    int = 20
    signal_type: str = "categorical"

    def __init__(self, period: int = 20) -> None:
        self.period   = period
        self.lookback = period

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close     = df["Close"]
        volume    = df["Volume"].replace(0.0, np.nan)

        price_roc = (close - close.shift(self.period)) / close.shift(self.period).replace(0, np.nan)
        vol_roc   = (volume - volume.shift(self.period)) / volume.shift(self.period).replace(0, np.nan)

        signal_vals = np.zeros(len(df), dtype=float)

        # Bearish divergence: price up, volume down
        bearish = (price_roc > 0) & (vol_roc < 0)
        # Bullish divergence: price down, volume up (selling climax)
        bullish = (price_roc < 0) & (vol_roc > 0)

        signal_vals = np.where(bearish.values, -1.0, signal_vals)
        signal_vals = np.where(bullish.values,  1.0, signal_vals)

        result = pd.Series(signal_vals, index=df.index, name=self.name)
        return result
