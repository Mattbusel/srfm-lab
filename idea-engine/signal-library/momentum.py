"""
idea-engine/signal-library/momentum.py
=======================================
12 momentum signals for the SRFM Idea Engine Signal Library.

All signals accept a standard OHLCV DataFrame (Open, High, Low, Close, Volume)
and return a pd.Series of signal values.

Signals
-------
1.  EMAMomentum          — EMA(fast) / EMA(slow) - 1
2.  ROC                  — Rate of change over N bars
3.  RSI                  — Relative Strength Index
4.  MACD                 — MACD line + signal + histogram (returns histogram)
5.  ADX                  — Average Directional Index
6.  AroonOscillator      — Aroon up/down oscillator
7.  TrendIntensity       — fraction of bars closing in trend direction
8.  MomentumDivergence   — price new high but RSI does not (bearish div)
9.  AccelerationMomentum — second derivative of price
10. DualMomentum         — absolute + relative momentum (Antonacci-style)
11. BHMassSignal         — Black-Hole mass physics signal
12. VolAdjMomentum       — momentum scaled by inverse realised volatility
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import Signal, SignalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. EMAMomentum
# ---------------------------------------------------------------------------

class EMAMomentum(Signal):
    """
    EMA-based momentum: EMA(fast) / EMA(slow) - 1.

    Positive values indicate the short-term trend is above the long-term trend
    (bullish momentum). Negative values indicate bearish momentum.
    """

    name:        str = "ema_momentum"
    category:    str = "momentum"
    lookback:    int = 50
    signal_type: str = "continuous"

    def __init__(self, fast: int = 12, slow: int = 26) -> None:
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow
        self.lookback = slow

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        ema_fast = self._ema(close, self.fast)
        ema_slow = self._ema(close, self.slow)
        result = ema_fast / ema_slow.replace(0, np.nan) - 1.0
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 2. ROC — Rate of Change
# ---------------------------------------------------------------------------

class ROC(Signal):
    """
    Rate of change: (Close[t] - Close[t-N]) / Close[t-N].

    A simple N-bar momentum measure. Returns a fractional change (0.05 = 5%).
    """

    name:        str = "roc"
    category:    str = "momentum"
    lookback:    int = 20
    signal_type: str = "continuous"

    def __init__(self, period: int = 20) -> None:
        self.period  = period
        self.lookback = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close  = df["Close"]
        past   = close.shift(self.period)
        result = (close - past) / past.replace(0, np.nan)
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 3. RSI — Relative Strength Index
# ---------------------------------------------------------------------------

class RSI(Signal):
    """
    Wilder's Relative Strength Index (0–100).

    Classic interpretation:
        > 70  overbought (potential reversal down)
        < 30  oversold  (potential reversal up)
    """

    name:        str = "rsi"
    category:    str = "momentum"
    lookback:    int = 14
    signal_type: str = "continuous"

    def __init__(self, period: int = 14) -> None:
        self.period   = period
        self.lookback = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close  = df["Close"]
        delta  = close.diff()
        gain   = delta.clip(lower=0.0)
        loss   = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period,
                            adjust=False).mean()
        avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period,
                            adjust=False).mean()
        rs     = avg_gain / avg_loss.replace(0, np.nan)
        result = 100.0 - (100.0 / (1.0 + rs))
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 4. MACD
# ---------------------------------------------------------------------------

class MACD(Signal):
    """
    MACD (Moving Average Convergence Divergence).

    Returns the MACD histogram: (MACD line - signal line).
    Positive histogram → bullish momentum; negative → bearish.

    Intermediate series are stored in the result metadata when
    compute_result() is used.
    """

    name:        str = "macd"
    category:    str = "momentum"
    lookback:    int = 35
    signal_type: str = "continuous"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast   = fast
        self.slow   = slow
        self.signal = signal
        self.lookback = slow + signal

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close       = df["Close"]
        ema_fast    = self._ema(close, self.fast)
        ema_slow    = self._ema(close, self.slow)
        macd_line   = ema_fast - ema_slow
        signal_line = self._ema(macd_line, self.signal)
        histogram   = macd_line - signal_line
        histogram.name = self.name
        return histogram

    def compute_result(self, df: pd.DataFrame) -> SignalResult:
        self.validate(df)
        close       = df["Close"]
        ema_fast    = self._ema(close, self.fast)
        ema_slow    = self._ema(close, self.slow)
        macd_line   = ema_fast - ema_slow
        signal_line = self._ema(macd_line, self.signal)
        histogram   = macd_line - signal_line
        histogram.name = self.name
        return SignalResult(
            values=histogram,
            signal_name=self.name,
            category=self.category,
            metadata={
                "macd_line":   macd_line,
                "signal_line": signal_line,
                "lookback":    self.lookback,
                "signal_type": self.signal_type,
            },
        )


# ---------------------------------------------------------------------------
# 5. ADX — Average Directional Index
# ---------------------------------------------------------------------------

class ADX(Signal):
    """
    Wilder's Average Directional Index (0–100).

    Measures trend strength regardless of direction.
        > 25  trending (strong trend)
        < 20  ranging / no trend
    """

    name:        str = "adx"
    category:    str = "momentum"
    lookback:    int = 28
    signal_type: str = "continuous"

    def __init__(self, period: int = 14) -> None:
        self.period   = period
        self.lookback = period * 2

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        high  = df["High"]
        low   = df["Low"]
        close = df["Close"]

        tr    = self._true_range(df)

        # Directional movements
        up_move   = high.diff()
        down_move = (-low.diff())
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        pos_dm_s = pd.Series(pos_dm, index=df.index)
        neg_dm_s = pd.Series(neg_dm, index=df.index)

        atr14   = tr.ewm(com=self.period - 1, min_periods=self.period,
                         adjust=False).mean()
        pos_di  = 100.0 * pos_dm_s.ewm(com=self.period - 1,
                                        min_periods=self.period,
                                        adjust=False).mean() / atr14.replace(0, np.nan)
        neg_di  = 100.0 * neg_dm_s.ewm(com=self.period - 1,
                                        min_periods=self.period,
                                        adjust=False).mean() / atr14.replace(0, np.nan)

        dx      = 100.0 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan)
        adx_val = dx.ewm(com=self.period - 1, min_periods=self.period,
                         adjust=False).mean()
        adx_val.name = self.name
        return adx_val


# ---------------------------------------------------------------------------
# 6. AroonOscillator
# ---------------------------------------------------------------------------

class AroonOscillator(Signal):
    """
    Aroon Oscillator = Aroon_Up - Aroon_Down.

    Range: -100 to +100.
        > 0   uptrend dominating
        < 0   downtrend dominating
        Near 0 → no clear trend
    """

    name:        str = "aroon_oscillator"
    category:    str = "momentum"
    lookback:    int = 25
    signal_type: str = "continuous"

    def __init__(self, period: int = 25) -> None:
        self.period   = period
        self.lookback = period

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        high = df["High"]
        low  = df["Low"]
        n    = self.period

        aroon_up   = high.rolling(n + 1, min_periods=n).apply(
            lambda x: float(np.argmax(x)) / n * 100.0, raw=True
        )
        aroon_down = low.rolling(n + 1, min_periods=n).apply(
            lambda x: float(np.argmin(x)) / n * 100.0, raw=True
        )
        result = aroon_up - aroon_down
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 7. TrendIntensity
# ---------------------------------------------------------------------------

class TrendIntensity(Signal):
    """
    Fraction of bars closing above (or below) their opening over N bars.

    A value > 0.5 indicates that more bars have been up-bars than down-bars
    (bullish trend intensity). The signal is centred around 0 by subtracting 0.5
    so it oscillates between -0.5 and +0.5.
    """

    name:        str = "trend_intensity"
    category:    str = "momentum"
    lookback:    int = 20
    signal_type: str = "continuous"

    def __init__(self, period: int = 20) -> None:
        self.period   = period
        self.lookback = period

    def validate(self, df: pd.DataFrame) -> None:
        if "Open" not in df.columns or "Close" not in df.columns:
            raise ValueError(
                f"Signal '{self.name}': requires Open and Close columns."
            )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        up_bar = (df["Close"] > df["Open"]).astype(float)
        result = up_bar.rolling(self.period, min_periods=1).mean() - 0.5
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 8. MomentumDivergence
# ---------------------------------------------------------------------------

class MomentumDivergence(Signal):
    """
    Detects bearish momentum divergence: price makes a new N-bar high but
    RSI does not.

    Returns:
        +1  if bullish divergence (price new low, RSI higher low)
        -1  if bearish divergence (price new high, RSI lower high)
         0  otherwise
    """

    name:        str = "momentum_divergence"
    category:    str = "momentum"
    lookback:    int = 30
    signal_type: str = "categorical"

    def __init__(self, rsi_period: int = 14, lookback_bars: int = 20) -> None:
        self.rsi_period    = rsi_period
        self.lookback_bars = lookback_bars
        self.lookback      = rsi_period + lookback_bars

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]

        # RSI
        delta    = close.diff()
        gain     = delta.clip(lower=0.0)
        loss     = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period,
                            adjust=False).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period,
                            adjust=False).mean()
        rsi      = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss.replace(0, np.nan)))

        n     = self.lookback_bars
        signal_vals = np.zeros(len(df), dtype=float)

        close_arr = close.values
        rsi_arr   = rsi.values

        for i in range(n, len(df)):
            window_close = close_arr[i - n: i + 1]
            window_rsi   = rsi_arr[i - n: i + 1]

            # Bearish divergence: price at new high, RSI lower than its prior high
            if close_arr[i] == np.nanmax(window_close):
                # Find the prior high index (excluding current bar)
                prior_high_idx = int(np.nanargmax(window_close[:-1]))
                if not np.isnan(rsi_arr[i]) and not np.isnan(window_rsi[prior_high_idx]):
                    if rsi_arr[i] < window_rsi[prior_high_idx]:
                        signal_vals[i] = -1.0

            # Bullish divergence: price at new low, RSI higher than its prior low
            elif close_arr[i] == np.nanmin(window_close):
                prior_low_idx = int(np.nanargmin(window_close[:-1]))
                if not np.isnan(rsi_arr[i]) and not np.isnan(window_rsi[prior_low_idx]):
                    if rsi_arr[i] > window_rsi[prior_low_idx]:
                        signal_vals[i] = 1.0

        result = pd.Series(signal_vals, index=df.index, name=self.name)
        return result


# ---------------------------------------------------------------------------
# 9. AccelerationMomentum
# ---------------------------------------------------------------------------

class AccelerationMomentum(Signal):
    """
    Acceleration momentum: second derivative of price, i.e. momentum-of-momentum.

    Computed as the N-bar ROC of the N-bar ROC.  A positive value means
    momentum is accelerating upward; negative means momentum is decelerating
    or reversing.
    """

    name:        str = "acceleration_momentum"
    category:    str = "momentum"
    lookback:    int = 20
    signal_type: str = "continuous"

    def __init__(self, period: int = 10) -> None:
        self.period   = period
        self.lookback = period * 2

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close   = df["Close"]
        roc1    = (close - close.shift(self.period)) / close.shift(self.period).replace(0, np.nan)
        roc2    = (roc1 - roc1.shift(self.period)) / roc1.shift(self.period).abs().replace(0, np.nan)
        roc2.name = self.name
        return roc2


# ---------------------------------------------------------------------------
# 10. DualMomentum
# ---------------------------------------------------------------------------

class DualMomentum(Signal):
    """
    Dual Momentum (Antonacci-style).

    Combines:
        - Absolute momentum: is the asset's own N-bar return positive?
        - Relative momentum: is the asset outperforming its benchmark?

    If a benchmark (Close column of another asset) is provided via
    ``benchmark_series``, relative momentum is computed against it.
    Otherwise, it falls back to pure absolute momentum.

    Returns a composite score in [-1, +1]:
        +1  both absolute and relative are positive  (strong bull)
         0  mixed signals
        -1  both negative                             (strong bear)
    """

    name:        str = "dual_momentum"
    category:    str = "momentum"
    lookback:    int = 252
    signal_type: str = "continuous"

    def __init__(
        self,
        period: int = 252,
        benchmark_series: Optional[pd.Series] = None,
    ) -> None:
        self.period            = period
        self.benchmark_series  = benchmark_series
        self.lookback          = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close  = df["Close"]
        ret    = (close - close.shift(self.period)) / close.shift(self.period).replace(0, np.nan)

        # Absolute momentum component
        abs_mom = ret.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))

        # Relative momentum component
        if self.benchmark_series is not None:
            bench = self.benchmark_series.reindex(close.index).ffill()
            bench_ret = (bench - bench.shift(self.period)) / bench.shift(self.period).replace(0, np.nan)
            rel_mom   = (ret - bench_ret).apply(
                lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
            )
        else:
            rel_mom = abs_mom  # degenerate case

        result = (abs_mom + rel_mom) / 2.0
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 11. BHMassSignal — Black Hole mass physics as a signal class
# ---------------------------------------------------------------------------

class BHMassSignal(Signal):
    """
    Black-Hole mass signal: wraps the SRFM BH mass concept as a Signal class.

    The BH "mass" is an exponentially-weighted momentum proxy modelled as a
    gravitational attractor: mass accumulates when price is trending in the
    same direction as the EMA and decays otherwise.

    Mass interpretation (matches SRFM live system conventions):
        < 1.0   inactive (no BH pull)
        1.0-1.50  warming
        1.50-1.92 early warning (approaching singularity)
        ≥ 1.92   fully activated

    Returns raw mass values (unbounded above 0).
    """

    name:        str = "bh_mass"
    category:    str = "momentum"
    lookback:    int = 50
    signal_type: str = "continuous"

    # Thresholds mirroring the SRFM ingestion config
    WARMING_LOW:  float = 1.00
    EW_LOW:       float = 1.50
    EW_HIGH:      float = 1.92

    def __init__(
        self,
        ema_period:   int   = 20,
        growth_rate:  float = 0.15,
        decay_rate:   float = 0.10,
        initial_mass: float = 1.0,
    ) -> None:
        self.ema_period   = ema_period
        self.growth_rate  = growth_rate
        self.decay_rate   = decay_rate
        self.initial_mass = initial_mass
        self.lookback     = ema_period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        ema   = self._ema(close, self.ema_period)

        mass_vals = np.full(len(close), np.nan)
        mass = self.initial_mass

        for i in range(len(close)):
            if np.isnan(ema.iloc[i]):
                mass_vals[i] = np.nan
                continue
            # Price above EMA → trend alignment → mass grows
            if close.iloc[i] > ema.iloc[i]:
                mass = mass + self.growth_rate * (2.0 - mass / 2.0)
            else:
                mass = max(0.01, mass * (1.0 - self.decay_rate))
            mass_vals[i] = mass

        result = pd.Series(mass_vals, index=df.index, name=self.name)
        return result


# ---------------------------------------------------------------------------
# 12. VolAdjMomentum — Volatility-adjusted momentum
# ---------------------------------------------------------------------------

class VolAdjMomentum(Signal):
    """
    Volatility-adjusted momentum: N-bar return divided by realised vol.

    Scaling by inverse volatility makes momentum signals comparable across
    different market regimes and instruments.  High-vol environments reduce
    signal strength; low-vol environments amplify it.

    Equivalent to a Sharpe-ratio-style momentum estimate.
    """

    name:        str = "vol_adj_momentum"
    category:    str = "momentum"
    lookback:    int = 30
    signal_type: str = "continuous"

    def __init__(self, mom_period: int = 20, vol_period: int = 20) -> None:
        self.mom_period = mom_period
        self.vol_period = vol_period
        self.lookback   = max(mom_period, vol_period)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close   = df["Close"]
        log_ret = np.log(close / close.shift(1))
        mom     = np.log(close / close.shift(self.mom_period))
        vol     = log_ret.rolling(self.vol_period, min_periods=2).std() * np.sqrt(self.vol_period)
        result  = mom / vol.replace(0, np.nan)
        result.name = self.name
        return result
