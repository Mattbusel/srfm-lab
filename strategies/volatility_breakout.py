"""
volatility_breakout.py -- Volatility expansion breakout strategy.

References:
  - Bollinger (2001): Bollinger on Bollinger Bands
  - Kaufman (2013): Trading Systems and Methods
  - Engle (1982): Autoregressive Conditional Heteroscedasticity (ARCH)
  - Bollerslev (1986): Generalized ARCH (GARCH)

Strategy logic:
  1. Detect volatility compression: ATR < compression_threshold * long_ATR
  2. Compute breakout levels: midpoint +/- breakout_multiplier * ATR
  3. Filter: GARCH vol forecast, volume confirmation, trend direction
  4. Enter on breakout with ATR-based stop and trailing stop

BH constants:
  BH_MASS_THRESH = 1.92
  BH_DECAY       = 0.924
  BH_COLLAPSE    = 0.992

LARSA v18 compatible.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# BH physics constants
# ---------------------------------------------------------------------------
BH_MASS_THRESH = 1.92
BH_DECAY       = 0.924
BH_COLLAPSE    = 0.992


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BreakoutSignal:
    """Output from VolatilityBreakoutStrategy.generate_signal()."""
    direction: int     = 0        -- +1 long, -1 short, 0 flat
    entry_price: float = 0.0
    stop_price: float  = 0.0
    upper_band: float  = 0.0
    lower_band: float  = 0.0
    atr: float         = 0.0
    atr_ratio: float   = 0.0      -- short ATR / long ATR
    garch_vol: float   = 0.0
    realized_vol: float = 0.0
    vol_ratio: float   = 0.0      -- garch / realized
    compressed: bool   = False
    confirmed: bool    = False    -- volume + trend filter passed
    reason: str        = ""


@dataclass
class WalkForwardResult:
    """Results of walk-forward backtest for one instrument."""
    symbol: str         = ""
    total_return: float = 0.0
    cagr: float         = 0.0
    sharpe: float       = 0.0
    sortino: float      = 0.0
    max_drawdown: float = 0.0
    calmar: float       = 0.0
    win_rate: float     = 0.0
    profit_factor: float = 0.0
    n_trades: int       = 0
    avg_trade_return: float = 0.0
    n_windows: int      = 0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series  = field(default_factory=pd.Series)
    signals: pd.Series  = field(default_factory=pd.Series)
    params: dict        = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[{self.symbol}] Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
            f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
            f"Trades={self.n_trades} Windows={self.n_windows}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """True Range and ATR."""
    n  = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    atr = np.full(n, np.nan)
    atr[period - 1] = tr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


def _compute_ema(prices: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    n     = len(prices)
    alpha = 2.0 / (span + 1)
    ema   = np.full(n, np.nan)
    ema[0] = prices[0]
    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def _compute_stats(equity_curve: np.ndarray, trade_returns: List[float]) -> dict:
    n = len(equity_curve)
    initial = equity_curve[0]
    final   = equity_curve[-1]
    total_return = final / initial - 1.0
    n_years = max(1, n / 252)
    cagr    = (final / initial) ** (1.0 / n_years) - 1.0
    rets    = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
    rets    = np.concatenate([[0.0], rets])
    std     = rets.std()
    sharpe  = rets.mean() / std * math.sqrt(252) if std > 0 else 0.0
    down    = rets[rets < 0]
    sortino_d = np.std(down) if len(down) > 0 else 1e-9
    sortino   = rets.mean() / sortino_d * math.sqrt(252)
    pk  = np.maximum.accumulate(equity_curve)
    dd  = (equity_curve - pk) / (pk + 1e-9)
    mdd = dd.min()
    calmar  = cagr / abs(mdd) if mdd != 0 else 0.0
    wins    = [r for r in trade_returns if r > 0]
    losses  = [r for r in trade_returns if r <= 0]
    win_rate = len(wins) / len(trade_returns) if trade_returns else 0.0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    return dict(
        total_return=total_return, cagr=cagr, sharpe=sharpe, sortino=sortino,
        max_drawdown=mdd, calmar=calmar, win_rate=win_rate, profit_factor=pf,
        n_trades=len(trade_returns),
        avg_trade_return=float(np.mean(trade_returns)) if trade_returns else 0.0,
        returns=pd.Series(rets),
    )


# ---------------------------------------------------------------------------
# GARCH(1,1) Filter
# ---------------------------------------------------------------------------

class GARCH11Filter:
    """
    GARCH(1,1) volatility forecast.

    Model: sigma^2_t = omega + alpha * eps_{t-1}^2 + beta * sigma^2_{t-1}

    Fitted via MLE approximation using simple variance targeting.
    Default parameters are typical equity values (alpha=0.09, beta=0.90).

    Parameters
    ----------
    omega   : constant term (default variance targeting)
    alpha   : ARCH coefficient (default 0.09)
    beta    : GARCH coefficient (default 0.90)
    min_periods : warm-up bars before returning valid forecast (default 50)
    vol_ratio_min : reject entry if GARCH / realized < this threshold (default 0.8)
    """

    def __init__(
        self,
        omega: Optional[float]  = None,
        alpha: float             = 0.09,
        beta: float              = 0.90,
        min_periods: int         = 50,
        vol_ratio_min: float     = 0.8,
    ):
        self.omega        = omega
        self.alpha        = alpha
        self.beta         = beta
        self.min_periods  = min_periods
        self.vol_ratio_min = vol_ratio_min
        # Validate GARCH stationarity condition
        if alpha + beta >= 1.0:
            raise ValueError(f"GARCH not stationary: alpha+beta={alpha+beta:.3f} >= 1.0")

    def fit_variance_targeting(self, returns: np.ndarray) -> float:
        """
        Variance targeting: omega = long_run_var * (1 - alpha - beta).
        Returns estimated long-run variance.
        """
        long_run_var = float(np.var(returns))
        return long_run_var * (1.0 - self.alpha - self.beta)

    def forecast_series(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute GARCH(1,1) conditional variance series.
        Returns array of conditional standard deviations (annualized).

        returns: daily return array (not percentage)
        """
        n       = len(returns)
        sigma2  = np.full(n, np.nan)
        if n < self.min_periods:
            return sigma2

        # Initialize omega via variance targeting
        omega = self.omega
        if omega is None:
            omega = self.fit_variance_targeting(returns[:self.min_periods])

        # Initialize sigma2[0] with unconditional variance
        sigma2[0] = float(np.var(returns[:self.min_periods]))

        for i in range(1, n):
            eps2    = returns[i - 1] ** 2
            sigma2[i] = omega + self.alpha * eps2 + self.beta * sigma2[i - 1]

        # Convert variance to annualized vol
        sigma_ann = np.sqrt(sigma2 * 252)
        return sigma_ann

    def allow_entry(
        self,
        returns: np.ndarray,
        at_i: int,
        realized_vol: float,
    ) -> Tuple[bool, float, float]:
        """
        Decide whether to allow a new breakout entry.

        Logic: reject if GARCH vol forecast / realized_vol < vol_ratio_min
        (i.e., GARCH thinks vol is already expanding relative to history).

        Returns (allow, garch_vol, vol_ratio).
        """
        sigma = self.forecast_series(returns[:at_i + 1])
        if np.isnan(sigma[at_i]):
            return True, 0.0, 1.0   -- not enough data, allow
        garch_vol = float(sigma[at_i])
        vol_ratio = garch_vol / (realized_vol + 1e-9)
        allow     = vol_ratio >= self.vol_ratio_min
        return allow, garch_vol, vol_ratio


# ---------------------------------------------------------------------------
# Volatility Compression Detector
# ---------------------------------------------------------------------------

class CompressionDetector:
    """
    Detects periods of volatility compression using ATR ratio.

    Compression = short_period ATR < threshold * long_period ATR.

    Parameters
    ----------
    atr_period        : short ATR period (default 14)
    long_atr_period   : long ATR period for normalization (default 50)
    threshold         : ATR ratio threshold -- must be < this to be compressed (default 0.7)
    """

    def __init__(
        self,
        atr_period: int      = 14,
        long_atr_period: int = 50,
        threshold: float     = 0.7,
    ):
        self.atr_period      = atr_period
        self.long_atr_period = long_atr_period
        self.threshold       = threshold

    def compute_atr_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute ratio of short ATR to long ATR.
        Returns Series of ATR ratios.
        """
        high  = df["high"].values
        low   = df["low"].values
        close = df["close"].values

        short_atr = _compute_atr(high, low, close, self.atr_period)
        long_atr  = _compute_atr(high, low, close, self.long_atr_period)

        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = short_atr / (long_atr + 1e-9)

        return pd.Series(ratio, index=df.index)

    def compression_mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns boolean Series: True = compressed volatility.
        """
        ratio = self.compute_atr_ratio(df)
        return ratio < self.threshold


# ---------------------------------------------------------------------------
# Breakout Level Computer
# ---------------------------------------------------------------------------

class BreakoutLevelComputer:
    """
    Compute upper and lower breakout levels using ATR-based bands.

    Upper band = midpoint + multiplier * ATR
    Lower band = midpoint - multiplier * ATR

    Midpoint = (N-bar high + N-bar low) / 2 (Donchian midpoint).

    Parameters
    ----------
    atr_period   : ATR period (default 14)
    channel_period : lookback for Donchian midpoint (default 20)
    multiplier   : breakout distance in ATR units (default 1.5)
    """

    def __init__(
        self,
        atr_period: int    = 14,
        channel_period: int = 20,
        multiplier: float  = 1.5,
    ):
        self.atr_period     = atr_period
        self.channel_period = channel_period
        self.multiplier     = multiplier

    def compute_bands(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Returns (upper_band, lower_band) as Series aligned to df.index.
        """
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        atr   = _compute_atr(high.values, low.values, close.values, self.atr_period)
        atr_s = pd.Series(atr, index=df.index)

        chan_high = high.rolling(self.channel_period).max()
        chan_low  = low.rolling(self.channel_period).min()
        midpoint  = (chan_high + chan_low) / 2.0

        upper = midpoint + self.multiplier * atr_s
        lower = midpoint - self.multiplier * atr_s
        return upper, lower


# ---------------------------------------------------------------------------
# Trend Filter
# ---------------------------------------------------------------------------

class TrendFilter:
    """
    Only take breakouts in the direction of the 50-bar EMA slope.

    Long breakouts: EMA slope > 0 (uptrend)
    Short breakouts: EMA slope < 0 (downtrend)

    Parameters
    ----------
    ema_span : EMA span for trend detection (default 50)
    slope_threshold : minimum EMA slope (annualized) to qualify (default 0.0)
    """

    def __init__(self, ema_span: int = 50, slope_threshold: float = 0.0):
        self.ema_span        = ema_span
        self.slope_threshold = slope_threshold

    def trend_direction(self, close: pd.Series) -> pd.Series:
        """
        Returns +1 for uptrend, -1 for downtrend, 0 if slope below threshold.
        """
        ema   = close.ewm(span=self.ema_span, adjust=False).mean()
        slope = ema.diff()   -- daily EMA change
        direction = pd.Series(0, index=close.index, dtype=float)
        direction[slope > self.slope_threshold]  =  1.0
        direction[slope < -self.slope_threshold] = -1.0
        return direction

    def filter_direction(self, signal_dir: int, at_idx, close: pd.Series) -> bool:
        """
        Return True if signal_dir is consistent with trend at at_idx.
        """
        trend = self.trend_direction(close)
        if at_idx not in trend.index:
            return True
        t_dir = trend.loc[at_idx]
        if t_dir == 0:
            return False
        return int(np.sign(signal_dir)) == int(t_dir)


# ---------------------------------------------------------------------------
# Breakout Confirmation (volume filter)
# ---------------------------------------------------------------------------

class BreakoutConfirmation:
    """
    Require volume > volume_multiplier * average volume on the breakout bar.

    Parameters
    ----------
    vol_lookback     : bars for average volume (default 20)
    volume_multiplier : minimum volume ratio (default 1.5)
    """

    def __init__(self, vol_lookback: int = 20, volume_multiplier: float = 1.5):
        self.vol_lookback      = vol_lookback
        self.volume_multiplier = volume_multiplier

    def confirm(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns boolean Series: True = volume-confirmed bar.
        Requires 'volume' column in df.
        """
        if "volume" not in df.columns:
            return pd.Series(True, index=df.index)   -- no volume data, pass
        vol     = df["volume"]
        avg_vol = vol.rolling(self.vol_lookback, min_periods=5).mean()
        return vol > self.volume_multiplier * avg_vol


# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------

class RiskManagement:
    """
    ATR-based stop and trail logic for breakout trades.

    Initial stop: entry - stop_atr_mult * ATR (for longs)
    Trailing stop: trail_atr_mult * ATR below highest close seen since entry

    Parameters
    ----------
    stop_atr_mult  : initial stop in ATR units (default 1.0)
    trail_atr_mult : trailing stop in ATR units (default 0.75)
    atr_period     : ATR period (default 14)
    """

    def __init__(
        self,
        stop_atr_mult: float  = 1.0,
        trail_atr_mult: float = 0.75,
        atr_period: int       = 14,
    ):
        self.stop_atr_mult  = stop_atr_mult
        self.trail_atr_mult = trail_atr_mult
        self.atr_period     = atr_period
        self._trail_extreme = None   -- highest/lowest price since entry
        self._current_stop  = None
        self._direction     = 0

    def set_entry(self, entry_price: float, atr: float, direction: int):
        """Initialize stop at entry."""
        self._direction     = direction
        self._trail_extreme = entry_price
        self._current_stop  = entry_price - direction * self.stop_atr_mult * atr

    def update_trail(self, current_price: float, atr: float) -> float:
        """
        Update trailing stop given current price and ATR.
        Returns current stop level.
        """
        if self._direction == 0 or self._trail_extreme is None:
            return float("-inf")

        # Update extreme price
        if self._direction == 1:
            self._trail_extreme = max(self._trail_extreme, current_price)
        else:
            self._trail_extreme = min(self._trail_extreme, current_price)

        # Trailing stop
        trail_stop = self._trail_extreme - self._direction * self.trail_atr_mult * atr
        self._current_stop = trail_stop
        return trail_stop

    def stopped_out(self, current_price: float) -> bool:
        """Return True if current price has hit the stop level."""
        if self._current_stop is None:
            return False
        if self._direction == 1:
            return current_price <= self._current_stop
        else:
            return current_price >= self._current_stop

    def reset(self):
        self._trail_extreme = None
        self._current_stop  = None
        self._direction     = 0


# ---------------------------------------------------------------------------
# Core Strategy
# ---------------------------------------------------------------------------

class VolatilityBreakoutStrategy:
    """
    Enters long/short when price breaks out of a volatility-based band
    after a period of compression. Uses GARCH vol forecast as entry gate.

    Entry conditions (all must pass):
      1. detect_compression() -> True: ATR / long_ATR < compression_threshold
      2. Price crosses breakout level (upper or lower)
      3. GARCH vol forecast / realized_vol >= garch_vol_ratio_min
      4. Volume > 1.5x average on breakout bar
      5. Breakout direction matches 50-bar EMA slope

    Exit conditions:
      - ATR trailing stop hit
      - Opposite breakout signal

    Parameters
    ----------
    config : dict of parameter overrides
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.atr_period            = cfg.get("atr_period", 14)
        self.long_atr_period       = cfg.get("long_atr_period", 50)
        self.compression_threshold = cfg.get("compression_threshold", 0.7)   -- ATR < 70% of 50-bar ATR
        self.breakout_multiplier   = cfg.get("breakout_multiplier", 1.5)      -- break above/below 1.5x ATR
        self.garch_vol_ratio_min   = cfg.get("garch_vol_ratio_min", 0.8)      -- GARCH / realized > 0.8
        self.ema_span              = cfg.get("ema_span", 50)
        self.channel_period        = cfg.get("channel_period", 20)
        self.vol_lookback          = cfg.get("vol_lookback", 20)
        self.volume_multiplier     = cfg.get("volume_multiplier", 1.5)
        self.stop_atr_mult         = cfg.get("stop_atr_mult", 1.0)
        self.trail_atr_mult        = cfg.get("trail_atr_mult", 0.75)

        self._compression  = CompressionDetector(
            atr_period=self.atr_period,
            long_atr_period=self.long_atr_period,
            threshold=self.compression_threshold,
        )
        self._band_computer = BreakoutLevelComputer(
            atr_period=self.atr_period,
            channel_period=self.channel_period,
            multiplier=self.breakout_multiplier,
        )
        self._garch         = GARCH11Filter(
            alpha=cfg.get("garch_alpha", 0.09),
            beta=cfg.get("garch_beta", 0.90),
            vol_ratio_min=self.garch_vol_ratio_min,
        )
        self._trend         = TrendFilter(ema_span=self.ema_span)
        self._vol_confirm   = BreakoutConfirmation(
            vol_lookback=self.vol_lookback,
            volume_multiplier=self.volume_multiplier,
        )
        self._risk          = RiskManagement(
            stop_atr_mult=self.stop_atr_mult,
            trail_atr_mult=self.trail_atr_mult,
            atr_period=self.atr_period,
        )
        self.config = cfg

    def detect_compression(self, bars: pd.DataFrame) -> bool:
        """
        Returns True if volatility is currently compressed.
        ATR / long_ATR < compression_threshold.
        """
        mask = self._compression.compression_mask(bars)
        return bool(mask.iloc[-1]) if len(mask) > 0 else False

    def compute_breakout_levels(self, bars: pd.DataFrame) -> Tuple[float, float]:
        """
        Returns (upper, lower) breakout band levels using latest bar.
        """
        upper, lower = self._band_computer.compute_bands(bars)
        return float(upper.iloc[-1]), float(lower.iloc[-1])

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate full signal series for the DataFrame.

        Returns DataFrame with columns:
          signal, atr_ratio, upper_band, lower_band, garch_vol,
          compressed, confirmed, stop_level
        """
        n      = len(df)
        high   = df["high"].values
        low    = df["low"].values
        close  = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(n)

        atr_short = _compute_atr(high, low, close, self.atr_period)
        atr_long  = _compute_atr(high, low, close, self.long_atr_period)

        returns   = pd.Series(close).pct_change().fillna(0).values
        ema       = _compute_ema(close, self.ema_span)

        # Compute volume average
        avg_vol   = pd.Series(volume).rolling(self.vol_lookback, min_periods=5).mean().values

        # Compute breakout bands
        upper_s, lower_s = self._band_computer.compute_bands(df)
        upper_arr = upper_s.values
        lower_arr = lower_s.values

        # GARCH forecast series
        garch_sigma = self._garch.forecast_series(returns)

        # Realized vol: 21-bar rolling std annualized
        rvol = pd.Series(returns).rolling(21, min_periods=10).std().values * math.sqrt(252)

        signals    = np.zeros(n)
        compressed = np.zeros(n, dtype=bool)
        confirmed  = np.zeros(n, dtype=bool)
        stop_arr   = np.full(n, np.nan)

        # State
        position      = 0      -- current position: +1, -1, 0
        was_compressed = False  -- were we compressed in prior bar
        self._risk.reset()

        warmup = max(self.atr_period, self.long_atr_period, self.channel_period,
                     self.ema_span, self.vol_lookback, 60)

        for i in range(warmup, n):
            if np.isnan(atr_short[i]) or np.isnan(atr_long[i]):
                continue

            # ATR ratio
            atr_ratio = atr_short[i] / (atr_long[i] + 1e-9)
            is_compressed = atr_ratio < self.compression_threshold
            compressed[i] = is_compressed

            # Volume confirmation
            vol_ok = (volume[i] > self.volume_multiplier * avg_vol[i]) if avg_vol[i] > 0 else True
            confirmed[i] = vol_ok

            # GARCH filter
            rv = rvol[i] if not np.isnan(rvol[i]) and rvol[i] > 0 else 0.15
            gv = garch_sigma[i] if not np.isnan(garch_sigma[i]) else rv
            vol_ratio = gv / (rv + 1e-9)
            garch_ok  = vol_ratio >= self.garch_vol_ratio_min

            # EMA trend
            ema_slope = ema[i] - ema[i - 1] if not np.isnan(ema[i]) and not np.isnan(ema[i - 1]) else 0.0
            trend_dir = np.sign(ema_slope)

            # Trailing stop check for existing position
            if position != 0:
                trail_stop = self._risk.update_trail(close[i], atr_short[i])
                stop_arr[i] = trail_stop
                if self._risk.stopped_out(close[i]):
                    position = 0
                    self._risk.reset()

            # Entry logic -- only enter from flat, after compression, with all filters
            if position == 0:
                entry_ok = was_compressed and garch_ok and vol_ok

                # Long breakout: price exceeds upper band and trend is up
                if (entry_ok and not np.isnan(upper_arr[i])
                        and close[i] > upper_arr[i]
                        and trend_dir >= 0):
                    position = 1
                    self._risk.set_entry(close[i], atr_short[i], 1)
                    stop_arr[i] = close[i] - self.stop_atr_mult * atr_short[i]

                # Short breakout: price falls below lower band and trend is down
                elif (entry_ok and not np.isnan(lower_arr[i])
                        and close[i] < lower_arr[i]
                        and trend_dir <= 0):
                    position = -1
                    self._risk.set_entry(close[i], atr_short[i], -1)
                    stop_arr[i] = close[i] + self.stop_atr_mult * atr_short[i]

            signals[i]     = position
            was_compressed = is_compressed

        return pd.DataFrame({
            "signal":     signals,
            "atr_ratio":  atr_short / (atr_long + 1e-9),
            "upper_band": upper_arr,
            "lower_band": lower_arr,
            "garch_vol":  garch_sigma,
            "realized_vol": rvol,
            "compressed": compressed.astype(float),
            "confirmed":  confirmed.astype(float),
            "stop_level": stop_arr,
        }, index=df.index)

    def backtest(
        self,
        df: pd.DataFrame,
        symbol: str = "X",
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0002,
    ) -> WalkForwardResult:
        """Simple full-sample backtest (no walk-forward)."""
        sig_df  = self.generate_signals(df)
        close   = df["close"].values
        signals = sig_df["signal"].values
        n       = len(close)

        equity    = initial_equity
        eq_curve  = np.full(n, initial_equity, dtype=float)
        trade_ret = []
        prev_sig  = 0.0
        entry_px  = None

        for i in range(1, n):
            if np.isnan(signals[i - 1]):
                eq_curve[i] = equity
                continue
            new_sig = float(signals[i - 1])

            if new_sig != prev_sig and entry_px is not None and prev_sig != 0:
                ret = prev_sig * ((close[i] - entry_px) / (entry_px + 1e-9) - commission_pct * 2)
                trade_ret.append(float(ret))

            if new_sig != prev_sig:
                entry_px = close[i] if new_sig != 0 else None
                if new_sig != 0:
                    equity *= (1.0 - commission_pct)

            if new_sig != 0:
                bar_ret  = (close[i] - close[i - 1]) / (close[i - 1] + 1e-9)
                equity  *= (1 + new_sig * bar_ret)

            prev_sig    = new_sig
            eq_curve[i] = equity

        stats = _compute_stats(eq_curve, trade_ret)
        return WalkForwardResult(
            symbol=symbol,
            **{k: v for k, v in stats.items() if k != "returns"},
            equity_curve=pd.Series(eq_curve, index=df.index),
            returns=pd.Series(stats["returns"].values, index=df.index),
            signals=sig_df["signal"],
            params=self.config,
        )


# ---------------------------------------------------------------------------
# Walk-Forward Backtest
# ---------------------------------------------------------------------------

class VolatilityBreakoutBacktest:
    """
    Walk-forward out-of-sample backtest for multiple instruments.

    Walk-forward design:
      - train_window  : bars used to calibrate thresholds (default 252)
      - test_window   : bars used out-of-sample per window (default 63)
      - Rolls forward test_window bars each iteration

    Parameters
    ----------
    config         : passed to VolatilityBreakoutStrategy
    train_window   : in-sample calibration window in bars (default 252)
    test_window    : out-of-sample test window in bars (default 63)
    initial_equity : starting equity per instrument (default 1_000_000)
    commission_pct : per-side commission (default 0.0002)
    """

    def __init__(
        self,
        config: Optional[dict]  = None,
        train_window: int        = 252,
        test_window: int         = 63,
        initial_equity: float   = 1_000_000.0,
        commission_pct: float   = 0.0002,
    ):
        self.config         = config or {}
        self.train_window   = train_window
        self.test_window    = test_window
        self.initial_equity = initial_equity
        self.commission_pct = commission_pct

    def run_instrument(
        self, df: pd.DataFrame, symbol: str = "X"
    ) -> WalkForwardResult:
        """
        Run walk-forward on a single instrument.
        Returns WalkForwardResult with stitched OOS equity curve.
        """
        n        = len(df)
        start    = self.train_window
        n_windows = 0

        all_eq   = np.full(n, self.initial_equity, dtype=float)
        all_sig  = np.zeros(n)
        equity   = self.initial_equity
        all_trade_ret = []

        i = start
        while i < n:
            end = min(i + self.test_window, n)
            # Use all data up to end -- train on [i-train_window : i], test on [i : end]
            train_df = df.iloc[i - self.train_window: i]
            test_df  = df.iloc[i: end]

            if len(train_df) < 50 or len(test_df) < 5:
                i = end
                continue

            strat     = VolatilityBreakoutStrategy(config=self.config)
            # Generate signals on the combined window so warm-up is satisfied
            combined  = pd.concat([train_df, test_df])
            try:
                sig_df   = strat.generate_signals(combined)
            except Exception:
                i = end
                continue

            # Extract just the test-window signals
            test_sigs = sig_df["signal"].iloc[-len(test_df):].values
            test_close = test_df["close"].values

            for j in range(1, len(test_close)):
                sig = float(test_sigs[j - 1])
                if sig != 0:
                    bar_ret = (test_close[j] - test_close[j - 1]) / (test_close[j - 1] + 1e-9)
                    equity *= (1 + sig * bar_ret - self.commission_pct * abs(sig))
                all_eq[i + j]  = equity
                all_sig[i + j] = sig

            all_trade_ret.append(equity / self.initial_equity - 1.0)
            n_windows += 1
            i = end

        stats = _compute_stats(all_eq, all_trade_ret)
        return WalkForwardResult(
            symbol=symbol,
            n_windows=n_windows,
            **{k: v for k, v in stats.items() if k != "returns"},
            equity_curve=pd.Series(all_eq, index=df.index),
            returns=pd.Series(stats["returns"].values, index=df.index),
            signals=pd.Series(all_sig, index=df.index),
            params=self.config,
        )

    def run(
        self,
        instrument_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, WalkForwardResult]:
        """
        Run walk-forward for all instruments.

        Returns dict mapping symbol -> WalkForwardResult.
        """
        results = {}
        for sym, df in instrument_data.items():
            try:
                res = self.run_instrument(df, symbol=sym)
                results[sym] = res
                print(f"  {res.summary()}")
            except Exception as e:
                warnings.warn(f"Walk-forward failed for {sym}: {e}")
        return results

    def summary_table(
        self, results: Dict[str, WalkForwardResult]
    ) -> pd.DataFrame:
        rows = []
        for sym, r in results.items():
            rows.append({
                "symbol":       sym,
                "total_return": r.total_return,
                "cagr":         r.cagr,
                "sharpe":       r.sharpe,
                "max_drawdown": r.max_drawdown,
                "win_rate":     r.win_rate,
                "n_trades":     r.n_trades,
                "n_windows":    r.n_windows,
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(13)
    n   = 1500
    idx = pd.date_range("2019-01-01", periods=n, freq="B")

    # Simulate OHLCV with periodic compression + expansion
    vol_regime  = np.sin(np.linspace(0, 10 * math.pi, n)) * 0.005 + 0.010
    daily_ret   = rng.normal(0.0002, vol_regime, n)
    close       = 100.0 * np.cumprod(1 + daily_ret)
    high        = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low         = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume      = rng.integers(800_000, 2_000_000, n).astype(float)

    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": volume}, index=idx)

    strat  = VolatilityBreakoutStrategy()
    result = strat.backtest(df, symbol="TEST")
    print(result.summary())

    # Walk-forward multi-instrument
    instruments = {}
    for sym in [f"S{k}" for k in range(1, 11)]:
        c = 100.0 * np.cumprod(1 + rng.normal(0.0002, rng.uniform(0.008, 0.015), n))
        h = c * (1 + np.abs(rng.normal(0, 0.003, n)))
        l = c * (1 - np.abs(rng.normal(0, 0.003, n)))
        v = rng.integers(500_000, 3_000_000, n).astype(float)
        instruments[sym] = pd.DataFrame({"open": c, "high": h, "low": l,
                                          "close": c, "volume": v}, index=idx)

    bt      = VolatilityBreakoutBacktest(train_window=252, test_window=63)
    results = bt.run(instruments)
    print(bt.summary_table(results).to_string())
