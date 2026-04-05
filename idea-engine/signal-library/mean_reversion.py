"""
idea-engine/signal-library/mean_reversion.py
=============================================
10 mean-reversion signals for the SRFM Idea Engine Signal Library.

Signals
-------
1.  OUZScore              — Ornstein-Uhlenbeck z-score of price deviation
2.  BollingerBand         — price position within Bollinger bands (-1 to +1)
3.  RSIMeanReversion      — RSI extremes mapped to mean-reversion signal
4.  KeltnerChannel        — position within ATR-based Keltner channel
5.  DonchianBreakout      — N-bar high/low breakout signal
6.  StatArb               — spread z-score between two assets
7.  HalfLifeReversionSpeed — OU half-life estimation
8.  MeanReversionVelocity  — rate of change of z-score (dz/dt)
9.  HighLowMeanReversion   — intraday high-low range vs multi-day average
10. PriceToMovingAverage   — distance from various EMAs as percentile rank
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .base import Signal, SignalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. OUZScore — Ornstein-Uhlenbeck z-score
# ---------------------------------------------------------------------------

class OUZScore(Signal):
    """
    Ornstein-Uhlenbeck z-score: (price - rolling_mean) / rolling_std.

    Positive → price is above its mean (overbought, expect reversion down).
    Negative → price is below its mean (oversold, expect reversion up).

    This is the same computation used in the existing feature_extractor.py.
    """

    name:        str = "ou_zscore"
    category:    str = "mean_reversion"
    lookback:    int = 30
    signal_type: str = "continuous"

    def __init__(self, window: int = 30) -> None:
        self.window   = window
        self.lookback = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close     = df["Close"]
        min_pds   = max(self.window // 2, 2)
        roll_mean = close.rolling(self.window, min_periods=min_pds).mean()
        roll_std  = close.rolling(self.window, min_periods=min_pds).std()
        result    = (close - roll_mean) / roll_std.replace(0.0, np.nan)
        result    = result.fillna(0.0)
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 2. BollingerBand
# ---------------------------------------------------------------------------

class BollingerBand(Signal):
    """
    Bollinger Band position: where is the close relative to the bands?

    Returns a value in [-1, +1]:
        +1   close at upper band (overbought — mean-reversion short signal)
        -1   close at lower band (oversold  — mean-reversion long signal)
         0   close at the middle (at the mean)

    Formula: (close - middle) / (upper - middle)
    """

    name:        str = "bollinger_band"
    category:    str = "mean_reversion"
    lookback:    int = 20
    signal_type: str = "continuous"

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        self.period   = period
        self.num_std  = num_std
        self.lookback = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close  = df["Close"]
        middle = close.rolling(self.period, min_periods=2).mean()
        std    = close.rolling(self.period, min_periods=2).std()
        upper  = middle + self.num_std * std
        lower  = middle - self.num_std * std

        half_width = (upper - lower) / 2.0
        result     = (close - middle) / half_width.replace(0.0, np.nan)
        # Clip to [-1, +1] in case price gaps outside bands
        result     = result.clip(-1.0, 1.0)
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 3. RSIMeanReversion
# ---------------------------------------------------------------------------

class RSIMeanReversion(Signal):
    """
    RSI-based mean-reversion signal.

    Maps RSI to a mean-reversion direction:
        RSI > overbought_thresh → signal = -1 (expect reversion down)
        RSI < oversold_thresh   → signal = +1 (expect reversion up)
        Otherwise               → signal = 0  (neutral)

    For a continuous version, linearly interpolates between thresholds.
    """

    name:        str = "rsi_mean_reversion"
    category:    str = "mean_reversion"
    lookback:    int = 14
    signal_type: str = "continuous"

    def __init__(
        self,
        rsi_period:        int   = 14,
        overbought_thresh: float = 70.0,
        oversold_thresh:   float = 30.0,
        continuous:        bool  = True,
    ) -> None:
        self.rsi_period        = rsi_period
        self.overbought_thresh = overbought_thresh
        self.oversold_thresh   = oversold_thresh
        self.continuous        = continuous
        self.lookback          = rsi_period

    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        delta    = close.diff()
        gain     = delta.clip(lower=0.0)
        loss     = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(com=self.rsi_period - 1,
                            min_periods=self.rsi_period,
                            adjust=False).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1,
                            min_periods=self.rsi_period,
                            adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def compute(self, df: pd.DataFrame) -> pd.Series:
        rsi = self._compute_rsi(df["Close"])

        if self.continuous:
            # Linear mapping: rsi 50 → 0, rsi 70 → -1, rsi 30 → +1
            mid = 50.0
            upper_range = self.overbought_thresh - mid
            lower_range = mid - self.oversold_thresh
            result = np.where(
                rsi >= mid,
                -(rsi - mid) / upper_range,
                (mid - rsi) / lower_range,
            )
            result = pd.Series(result, index=df.index, name=self.name)
            result = result.clip(-1.0, 1.0)
        else:
            result = pd.Series(0.0, index=df.index, name=self.name)
            result = result.where(rsi < self.overbought_thresh, -1.0)
            result = result.where(rsi > self.oversold_thresh,   +1.0)

        return result


# ---------------------------------------------------------------------------
# 4. KeltnerChannel
# ---------------------------------------------------------------------------

class KeltnerChannel(Signal):
    """
    Keltner Channel: ATR-based channel around EMA.

    Returns position of Close relative to channel in [-1, +1]:
        +1  price at or above upper channel (overbought)
        -1  price at or below lower channel (oversold)
         0  price at EMA (neutral)
    """

    name:        str = "keltner_channel"
    category:    str = "mean_reversion"
    lookback:    int = 20
    signal_type: str = "continuous"

    def __init__(self, ema_period: int = 20, atr_period: int = 14,
                 multiplier: float = 2.0) -> None:
        self.ema_period  = ema_period
        self.atr_period  = atr_period
        self.multiplier  = multiplier
        self.lookback    = max(ema_period, atr_period)

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close  = df["Close"]
        ema    = self._ema(close, self.ema_period)
        tr     = self._true_range(df)
        atr    = tr.ewm(com=self.atr_period - 1, min_periods=1,
                        adjust=False).mean()
        upper  = ema + self.multiplier * atr
        lower  = ema - self.multiplier * atr

        half_width = (upper - lower) / 2.0
        result     = (close - ema) / half_width.replace(0.0, np.nan)
        result     = result.clip(-1.0, 1.0)
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 5. DonchianBreakout
# ---------------------------------------------------------------------------

class DonchianBreakout(Signal):
    """
    Donchian Channel breakout signal.

    Returns:
        +1  if close equals N-bar high  (breakout up  → breakout follow-through)
        -1  if close equals N-bar low   (breakout down → breakout follow-through)
         0  otherwise (inside channel)

    For a mean-reversion interpretation, negate the signal.
    The raw signal here is for breakout direction; callers decide interpretation.
    """

    name:        str = "donchian_breakout"
    category:    str = "mean_reversion"
    lookback:    int = 20
    signal_type: str = "categorical"

    def __init__(self, period: int = 20) -> None:
        self.period   = period
        self.lookback = period

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close     = df["Close"]
        high_roll = df["High"].rolling(self.period, min_periods=1).max()
        low_roll  = df["Low"].rolling(self.period, min_periods=1).min()

        signal_vals = np.zeros(len(df), dtype=float)
        signal_vals = np.where(close.values >= high_roll.values,  1.0, signal_vals)
        signal_vals = np.where(close.values <= low_roll.values,  -1.0, signal_vals)

        result = pd.Series(signal_vals, index=df.index, name=self.name)
        return result


# ---------------------------------------------------------------------------
# 6. StatArb — Spread z-score between two assets
# ---------------------------------------------------------------------------

class StatArb(Signal):
    """
    Statistical Arbitrage: z-score of the log-price spread between
    the primary asset (df["Close"]) and a second asset (``pair_series``).

    Positive z → spread is wide (primary rich vs pair) → sell primary
    Negative z → spread is narrow (primary cheap vs pair) → buy primary

    If ``pair_series`` is not provided, falls back to OUZScore.
    """

    name:        str = "stat_arb"
    category:    str = "mean_reversion"
    lookback:    int = 60
    signal_type: str = "continuous"

    def __init__(
        self,
        window:      int                    = 60,
        pair_series: Optional[pd.Series]   = None,
        hedge_ratio: Optional[float]        = None,
    ) -> None:
        self.window      = window
        self.pair_series = pair_series
        self.hedge_ratio = hedge_ratio
        self.lookback    = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_price = np.log(df["Close"].replace(0, np.nan))

        if self.pair_series is not None:
            pair_aligned = self.pair_series.reindex(df.index).ffill()
            log_pair     = np.log(pair_aligned.replace(0, np.nan))
            beta         = self.hedge_ratio if self.hedge_ratio is not None else 1.0
            spread       = log_price - beta * log_pair
        else:
            spread = log_price

        result = self._zscore(spread, self.window)
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 7. HalfLifeReversionSpeed
# ---------------------------------------------------------------------------

class HalfLifeReversionSpeed(Signal):
    """
    Ornstein-Uhlenbeck half-life estimation via rolling OLS.

    Fits: ΔX_t = θ(μ - X_t) + σε  ≡  ΔX_t = α + β X_{t-1} + ε

    Half-life = -ln(2) / β  (in bars).

    A short half-life (fast reversion) is a strong mean-reversion signal.
    Returns the half-life value (bars). Clip at 1-500 for numerical stability.
    """

    name:        str = "half_life_reversion"
    category:    str = "mean_reversion"
    lookback:    int = 60
    signal_type: str = "continuous"

    def __init__(self, window: int = 60) -> None:
        self.window   = window
        self.lookback = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_price = np.log(df["Close"].replace(0, np.nan))
        delta     = log_price.diff()

        half_life_vals = np.full(len(df), np.nan)

        for i in range(self.window, len(df)):
            x_lag = log_price.iloc[i - self.window: i].values[:-1]
            dy    = delta.iloc[i - self.window + 1: i + 1].values

            if len(x_lag) < 5 or np.all(np.isnan(x_lag)) or np.all(np.isnan(dy)):
                continue

            try:
                x_lag_clean = x_lag[~np.isnan(x_lag)]
                dy_clean    = dy[~np.isnan(dy)]
                if len(x_lag_clean) < 4:
                    continue
                # OLS via lstsq: [1, x_lag] @ [alpha, beta] = dy
                X    = np.column_stack([np.ones(len(x_lag_clean)), x_lag_clean])
                coef, *_ = np.linalg.lstsq(X, dy_clean, rcond=None)
                beta = coef[1]
                if beta < 0:
                    hl = float(-np.log(2) / beta)
                    half_life_vals[i] = np.clip(hl, 1.0, 500.0)
            except (np.linalg.LinAlgError, ValueError):
                pass

        result = pd.Series(half_life_vals, index=df.index, name=self.name)
        return result


# ---------------------------------------------------------------------------
# 8. MeanReversionVelocity
# ---------------------------------------------------------------------------

class MeanReversionVelocity(Signal):
    """
    Rate of change of the OU z-score (dz/dt over 1 bar).

    A positive velocity means the z-score is increasing (price moving away from
    mean). A negative velocity means the z-score is decreasing (price reverting).

    Useful as a timing signal: fade moves when z-score is high and velocity
    is starting to reverse.
    """

    name:        str = "mean_reversion_velocity"
    category:    str = "mean_reversion"
    lookback:    int = 31
    signal_type: str = "continuous"

    def __init__(self, window: int = 30, diff_periods: int = 1) -> None:
        self.window       = window
        self.diff_periods = diff_periods
        self.lookback     = window + diff_periods

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close     = df["Close"]
        min_pds   = max(self.window // 2, 2)
        roll_mean = close.rolling(self.window, min_periods=min_pds).mean()
        roll_std  = close.rolling(self.window, min_periods=min_pds).std()
        zscore    = (close - roll_mean) / roll_std.replace(0.0, np.nan)
        result    = zscore.diff(self.diff_periods)
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 9. HighLowMeanReversion
# ---------------------------------------------------------------------------

class HighLowMeanReversion(Signal):
    """
    High-Low mean reversion: compares intraday range to its multi-day average.

    Returns z-score of current high-low range relative to its rolling mean.

    High positive values → unusually wide intraday range (potential exhaustion)
    High negative values → unusually tight range (coiling before breakout)

    This can be combined with directional signals to fade exhaustion moves.
    """

    name:        str = "high_low_mean_reversion"
    category:    str = "mean_reversion"
    lookback:    int = 20
    signal_type: str = "continuous"

    def __init__(self, window: int = 20) -> None:
        self.window   = window
        self.lookback = window

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_ohlcv(df)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        hl_range  = df["High"] - df["Low"]
        min_pds   = max(self.window // 2, 2)
        roll_mean = hl_range.rolling(self.window, min_periods=min_pds).mean()
        roll_std  = hl_range.rolling(self.window, min_periods=min_pds).std()
        result    = (hl_range - roll_mean) / roll_std.replace(0.0, np.nan)
        result.name = self.name
        return result


# ---------------------------------------------------------------------------
# 10. PriceToMovingAverage
# ---------------------------------------------------------------------------

class PriceToMovingAverage(Signal):
    """
    Distance from multiple EMAs as a percentile rank.

    Computes the % distance of close from an EMA and then ranks that
    value in its rolling window (0 = historically close/below, 100 = far above).

    High percentile → price far above EMA historically → mean reversion risk.
    Low percentile  → price far below EMA historically → mean reversion opportunity.

    Returns percentile rank (0–100).
    """

    name:        str = "price_to_moving_average"
    category:    str = "mean_reversion"
    lookback:    int = 200
    signal_type: str = "continuous"

    def __init__(self, ema_period: int = 50, rank_window: int = 200) -> None:
        self.ema_period  = ema_period
        self.rank_window = rank_window
        self.lookback    = max(ema_period, rank_window)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close  = df["Close"]
        ema    = self._ema(close, self.ema_period)
        pct_dev = (close - ema) / ema.replace(0.0, np.nan)
        result  = self._percentile_rank(pct_dev, self.rank_window)
        result.name = self.name
        return result
