"""
trend_following.py — Classic trend following strategies.

All strategies share a common interface:
    strategy.generate_signals(df) -> pd.Series of {-1, 0, +1}
    strategy.backtest(df) -> BacktestResult

df must have columns: open, high, low, close, volume (all lowercase).
Index should be DatetimeIndex or integer.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Shared result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Container for backtest statistics."""
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    avg_trade_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    signals: pd.Series = field(default_factory=pd.Series)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Total Return : {self.total_return:.2%}",
            f"CAGR         : {self.cagr:.2%}",
            f"Sharpe       : {self.sharpe:.3f}",
            f"Sortino      : {self.sortino:.3f}",
            f"Max Drawdown : {self.max_drawdown:.2%}",
            f"Calmar       : {self.calmar:.3f}",
            f"Win Rate     : {self.win_rate:.2%}",
            f"Profit Factor: {self.profit_factor:.3f}",
            f"N Trades     : {self.n_trades}",
            f"Avg Trade    : {self.avg_trade_return:.4%}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers (pure numpy/pandas)
# ─────────────────────────────────────────────────────────────────────────────

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _rolling_high(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).max()


def _rolling_low(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).min()


def _keltner_bands(df: pd.DataFrame, period: int = 20, multiplier: float = 2.0):
    """Returns (upper, middle, lower) Keltner Channel bands."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    typical = (high + low + close) / 3
    middle = _ema(typical, period)
    atr_val = _atr(df, period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return upper, middle, lower


def _mesa_adaptive_period(price: pd.Series, base_period: int = 20) -> pd.Series:
    """
    Approximation of MESA adaptive moving average (Ehlers).
    Uses price cycle detection via Hilbert Transform approximation.
    Returns adaptive period series clamped between base_period/2 and base_period*2.
    """
    # Use Hilbert Transform approximation: Dominant Cycle via in-phase/quadrature
    n = len(price)
    smooth = price.copy()
    # 4-bar weighted moving average
    for i in range(3, n):
        smooth.iloc[i] = (4 * price.iloc[i] + 3 * price.iloc[i-1] +
                          2 * price.iloc[i-2] + price.iloc[i-3]) / 10.0

    # Detrender
    detrender = pd.Series(0.0, index=price.index)
    for i in range(6, n):
        detrender.iloc[i] = ((0.0962 * smooth.iloc[i] + 0.5769 * smooth.iloc[i-2] -
                               0.5769 * smooth.iloc[i-4] - 0.0962 * smooth.iloc[i-6]) *
                              (0.075 * 10 + 0.54))

    # In-phase and quadrature
    Q1 = pd.Series(0.0, index=price.index)
    I1 = pd.Series(0.0, index=price.index)
    for i in range(6, n):
        I1.iloc[i] = detrender.iloc[i-3]
        Q1.iloc[i] = ((0.0962 * detrender.iloc[i] + 0.5769 * detrender.iloc[i-2] -
                        0.5769 * detrender.iloc[i-4] - 0.0962 * detrender.iloc[i-6]) *
                       (0.075 * 10 + 0.54))

    # Dominant cycle
    period_out = pd.Series(float(base_period), index=price.index)
    prev_I1 = prev_Q1 = 0.0
    prev_period = float(base_period)
    for i in range(6, n):
        i1 = I1.iloc[i]; q1 = Q1.iloc[i]
        dphase = 0.0
        denom = (prev_I1 * i1 + prev_Q1 * q1)
        if denom != 0.0:
            dphase = (i1 * prev_Q1 - q1 * prev_I1) / denom
        dphase = max(-0.5, min(0.5, dphase))
        alpha = max(0.07, min(0.99, dphase))
        inst_period = 2.0 * math.pi / (math.acos(max(-1, min(1, 1.0 - alpha))) + 1e-9)
        dc = max(base_period / 2.0, min(base_period * 2.0, inst_period))
        prev_period = 0.2 * dc + 0.8 * prev_period
        period_out.iloc[i] = prev_period
        prev_I1 = i1; prev_Q1 = q1

    return period_out


def _backtest_from_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_equity: float = 1_000_000,
    commission_pct: float = 0.0001,
    slippage_pct: float = 0.0001,
) -> BacktestResult:
    """
    Standard backtest engine shared by all trend strategies.
    Signals: +1 long, -1 short, 0 flat. Applied at next-bar open.
    """
    close = df["close"].values
    sig = signals.values
    n = len(close)

    equity = initial_equity
    position = 0.0
    equity_curve = np.zeros(n)
    trade_returns = []
    entry_price = None
    entry_pos = 0.0

    for i in range(1, n):
        ret = (close[i] - close[i-1]) / close[i-1] if close[i-1] != 0 else 0.0
        # Apply position from previous bar signal
        prev_sig = sig[i-1] if not np.isnan(sig[i-1]) else 0.0
        new_pos = float(np.clip(prev_sig, -1, 1))

        if new_pos != position:
            # Close old trade
            if entry_price is not None and entry_pos != 0:
                cost = commission_pct + slippage_pct
                trade_ret = entry_pos * ((close[i] - entry_price) / entry_price - cost * 2)
                trade_returns.append(trade_ret)
            # Open new trade
            if new_pos != 0:
                entry_price = close[i] * (1 + slippage_pct * np.sign(new_pos))
                entry_pos = new_pos
            else:
                entry_price = None
                entry_pos = 0.0
            position = new_pos

        equity *= (1 + position * ret)
        equity_curve[i] = equity

    equity_curve[0] = initial_equity

    returns_series = pd.Series(equity_curve, index=df.index).pct_change().fillna(0)

    # Stats
    total_return = equity / initial_equity - 1.0
    n_years = max(1, n / 252)
    cagr = (equity / initial_equity) ** (1.0 / n_years) - 1.0

    daily_ret = returns_series.values
    std = daily_ret.std()
    sharpe = (daily_ret.mean() / std * math.sqrt(252)) if std > 0 else 0.0

    downside = daily_ret[daily_ret < 0]
    sortino_denom = np.std(downside) if len(downside) > 0 else 1e-9
    sortino = (daily_ret.mean() / sortino_denom * math.sqrt(252))

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / (running_max + 1e-9)
    max_drawdown = drawdowns.min()

    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    win_rate = len(wins) / len(trade_returns) if trade_returns else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
    avg_trade = np.mean(trade_returns) if trade_returns else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    return BacktestResult(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_trades=len(trade_returns),
        avg_trade_return=avg_trade,
        avg_win=avg_win,
        avg_loss=avg_loss,
        equity_curve=pd.Series(equity_curve, index=df.index),
        returns=returns_series,
        signals=signals,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. DualMovingAverage
# ─────────────────────────────────────────────────────────────────────────────

class DualMovingAverage:
    """
    Classic dual moving average crossover system.

    Entry: +1 when fast MA crosses above slow MA
           -1 when fast MA crosses below slow MA
    Exit: opposite crossover or flat when crossing back

    Parameters
    ----------
    fast : int
        Fast MA period (default 20)
    slow : int
        Slow MA period (default 50)
    ma_type : str
        "sma" or "ema" (default "ema")
    """

    def __init__(self, fast: int = 20, slow: int = 50, ma_type: str = "ema"):
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.ma_type = ma_type.lower()

    def _compute_ma(self, series: pd.Series, period: int) -> pd.Series:
        if self.ma_type == "sma":
            return _sma(series, period)
        return _ema(series, period)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns pd.Series of float signals in {-1.0, 0.0, +1.0}.
        NaN where not enough data.
        """
        close = df["close"]
        fast_ma = self._compute_ma(close, self.fast)
        slow_ma = self._compute_ma(close, self.slow)

        # Signal: +1 when fast > slow, -1 when fast < slow
        raw = np.where(fast_ma > slow_ma, 1.0, np.where(fast_ma < slow_ma, -1.0, 0.0))
        signals = pd.Series(raw, index=df.index, dtype=float)

        # NaN warmup period
        signals.iloc[: self.slow - 1] = np.nan
        return signals

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
        slippage_pct: float = 0.0001,
    ) -> BacktestResult:
        signals = self.generate_signals(df)
        result = _backtest_from_signals(df, signals, initial_equity, commission_pct, slippage_pct)
        result.params = {"fast": self.fast, "slow": self.slow, "ma_type": self.ma_type}
        return result

    def optimize(
        self,
        df: pd.DataFrame,
        fast_range: range = range(5, 50, 5),
        slow_range: range = range(20, 200, 10),
        metric: str = "sharpe",
    ) -> Tuple[dict, float]:
        """
        Grid-search over fast/slow combinations. Returns (best_params, best_metric).
        """
        best_val = -np.inf
        best_params = {}
        for f in fast_range:
            for s in slow_range:
                if f >= s:
                    continue
                try:
                    strat = DualMovingAverage(f, s, self.ma_type)
                    res = strat.backtest(df)
                    val = getattr(res, metric, 0.0)
                    if val > best_val:
                        best_val = val
                        best_params = {"fast": f, "slow": s}
                except Exception:
                    continue
        return best_params, best_val


# ─────────────────────────────────────────────────────────────────────────────
# 2. TripleMovingAverage
# ─────────────────────────────────────────────────────────────────────────────

class TripleMovingAverage:
    """
    Triple moving average filter system.

    Entry conditions:
        LONG  : price > mid MA AND fast MA > slow MA AND mid MA > slow MA
        SHORT : price < mid MA AND fast MA < slow MA AND mid MA < slow MA
        FLAT  : otherwise

    This adds a middle-term filter to avoid whipsaws in ambiguous regimes.

    Parameters
    ----------
    fast : int   — fast MA period
    mid  : int   — medium MA period
    slow : int   — slow MA period
    ma_type : str — "sma" or "ema"
    """

    def __init__(self, fast: int = 10, mid: int = 30, slow: int = 100, ma_type: str = "ema"):
        if not (fast < mid < slow):
            raise ValueError(f"Require fast ({fast}) < mid ({mid}) < slow ({slow})")
        self.fast = fast
        self.mid = mid
        self.slow = slow
        self.ma_type = ma_type.lower()

    def _ma(self, series: pd.Series, period: int) -> pd.Series:
        if self.ma_type == "sma":
            return _sma(series, period)
        return _ema(series, period)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        fast_ma = self._ma(close, self.fast)
        mid_ma = self._ma(close, self.mid)
        slow_ma = self._ma(close, self.slow)

        long_cond = (close > mid_ma) & (fast_ma > slow_ma) & (mid_ma > slow_ma)
        short_cond = (close < mid_ma) & (fast_ma < slow_ma) & (mid_ma < slow_ma)

        raw = np.where(long_cond, 1.0, np.where(short_cond, -1.0, 0.0))
        signals = pd.Series(raw, index=df.index, dtype=float)
        signals.iloc[: self.slow - 1] = np.nan
        return signals

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
        slippage_pct: float = 0.0001,
    ) -> BacktestResult:
        signals = self.generate_signals(df)
        result = _backtest_from_signals(df, signals, initial_equity, commission_pct, slippage_pct)
        result.params = {"fast": self.fast, "mid": self.mid, "slow": self.slow}
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. TurtleSystem
# ─────────────────────────────────────────────────────────────────────────────

class TurtleSystem:
    """
    Full Turtle Trading Rules (Dennis & Eckhardt, 1983).

    System 2 rules (20/10 variant is System 1; 55/20 is System 2):
    - Entry: buy new N-bar high (donchian breakout)
    - Exit:  close on new M-bar low (for long) / M-bar high (for short)
    - Stop:  2 * ATR from entry (N-unit stop)
    - Pyramid: add 1 unit per 0.5 ATR move in favour (max 4 units)
    - Position size: risk 1% equity per unit, sized by ATR

    Parameters
    ----------
    atr_period     : period for ATR calculation (default 20)
    entry_period   : Donchian breakout window (default 55)
    exit_period    : Donchian exit window (default 20)
    risk_per_unit  : fraction of equity to risk per unit (default 0.01)
    max_units      : maximum pyramid units (default 4)
    """

    def __init__(
        self,
        atr_period: int = 20,
        entry_period: int = 55,
        exit_period: int = 20,
        risk_per_unit: float = 0.01,
        max_units: int = 4,
    ):
        self.atr_period = atr_period
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.risk_per_unit = risk_per_unit
        self.max_units = max_units

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns signals in {-1, 0, +1} — simplified position direction.
        Full pyramid sizing is in backtest().
        """
        close = df["close"]
        high_entry = _rolling_high(df["high"], self.entry_period)
        low_entry = _rolling_low(df["low"], self.entry_period)
        high_exit = _rolling_high(df["high"], self.exit_period)
        low_exit = _rolling_low(df["low"], self.exit_period)

        # Long: close breaks above entry high; exit: close breaks below exit low
        # Short: close breaks below entry low; exit: close breaks above exit high
        signals = pd.Series(0.0, index=df.index)
        position = 0

        min_idx = max(self.entry_period, self.exit_period, self.atr_period)

        for i in range(min_idx, len(df)):
            if np.isnan(high_entry.iloc[i]) or np.isnan(low_entry.iloc[i]):
                continue
            if position == 0:
                if close.iloc[i] > high_entry.iloc[i - 1]:
                    position = 1
                elif close.iloc[i] < low_entry.iloc[i - 1]:
                    position = -1
            elif position == 1:
                if close.iloc[i] < low_exit.iloc[i - 1]:
                    position = 0
            elif position == -1:
                if close.iloc[i] > high_exit.iloc[i - 1]:
                    position = 0
            signals.iloc[i] = float(position)

        signals.iloc[:min_idx] = np.nan
        return signals

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
        slippage_pct: float = 0.0001,
    ) -> BacktestResult:
        """
        Full Turtle backtest with pyramid entries and ATR-based position sizing.
        """
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(close)

        atr_series = _atr(df, self.atr_period).values
        high_entry_series = _rolling_high(df["high"], self.entry_period).values
        low_entry_series = _rolling_low(df["low"], self.entry_period).values
        high_exit_series = _rolling_high(df["high"], self.exit_period).values
        low_exit_series = _rolling_low(df["low"], self.exit_period).values

        equity = initial_equity
        equity_curve = np.full(n, initial_equity, dtype=float)
        trade_returns = []

        # State
        position_dir = 0      # +1 long, -1 short, 0 flat
        units = 0             # pyramid count
        avg_entry = 0.0       # weighted avg entry price
        last_add_price = 0.0  # price at last pyramid add
        stop_price = 0.0

        warmup = max(self.entry_period, self.exit_period, self.atr_period)

        for i in range(warmup, n):
            if np.isnan(atr_series[i]) or atr_series[i] <= 0:
                equity_curve[i] = equity
                continue

            atr_val = atr_series[i]
            prev_high_e = high_entry_series[i - 1]
            prev_low_e = low_entry_series[i - 1]
            prev_high_x = high_exit_series[i - 1]
            prev_low_x = low_exit_series[i - 1]

            if any(np.isnan(x) for x in [prev_high_e, prev_low_e, prev_high_x, prev_low_x]):
                equity_curve[i] = equity
                continue

            c = close[i]
            cost = commission_pct + slippage_pct

            if position_dir == 0:
                # Check for breakout entry
                if c > prev_high_e:
                    position_dir = 1
                    units = 1
                    avg_entry = c * (1 + cost)
                    last_add_price = c
                    stop_price = c - 2 * atr_val
                elif c < prev_low_e:
                    position_dir = -1
                    units = 1
                    avg_entry = c * (1 - cost)
                    last_add_price = c
                    stop_price = c + 2 * atr_val

            elif position_dir == 1:
                # Pyramid add: add unit per 0.5 ATR in favor
                if units < self.max_units and c >= last_add_price + 0.5 * atr_val:
                    units += 1
                    add_price = c * (1 + cost)
                    avg_entry = ((avg_entry * (units - 1)) + add_price) / units
                    last_add_price = c
                    stop_price = max(stop_price, c - 2 * atr_val)

                # Exit conditions: stop hit OR exit channel breach
                if c < stop_price or c < prev_low_x:
                    # Calculate P&L
                    exit_price = c * (1 - cost)
                    trade_ret = units * self.risk_per_unit * (exit_price - avg_entry) / avg_entry
                    trade_returns.append(trade_ret)
                    equity *= (1 + trade_ret)
                    position_dir = 0
                    units = 0

            elif position_dir == -1:
                # Pyramid short
                if units < self.max_units and c <= last_add_price - 0.5 * atr_val:
                    units += 1
                    add_price = c * (1 - cost)
                    avg_entry = ((avg_entry * (units - 1)) + add_price) / units
                    last_add_price = c
                    stop_price = min(stop_price, c + 2 * atr_val)

                # Exit short: stop or exit high breach
                if c > stop_price or c > prev_high_x:
                    exit_price = c * (1 + cost)
                    trade_ret = units * self.risk_per_unit * (avg_entry - exit_price) / avg_entry
                    trade_returns.append(trade_ret)
                    equity *= (1 + trade_ret)
                    position_dir = 0
                    units = 0

            equity_curve[i] = equity

        # Compute stats
        returns_arr = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
        returns_arr = np.concatenate([[0], returns_arr])
        returns_series = pd.Series(returns_arr, index=df.index)

        total_return = equity / initial_equity - 1.0
        n_years = max(1, n / 252)
        cagr = (equity / initial_equity) ** (1.0 / n_years) - 1.0
        std = returns_arr.std()
        sharpe = (returns_arr.mean() / std * math.sqrt(252)) if std > 0 else 0.0
        downside = returns_arr[returns_arr < 0]
        sortino_d = np.std(downside) if len(downside) > 0 else 1e-9
        sortino = returns_arr.mean() / sortino_d * math.sqrt(252)
        running_max = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - running_max) / (running_max + 1e-9)
        max_dd = dd.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        win_rate = len(wins) / len(trade_returns) if trade_returns else 0.0
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

        return BacktestResult(
            total_return=total_return,
            cagr=cagr,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            calmar=calmar,
            win_rate=win_rate,
            profit_factor=pf,
            n_trades=len(trade_returns),
            avg_trade_return=np.mean(trade_returns) if trade_returns else 0.0,
            avg_win=np.mean(wins) if wins else 0.0,
            avg_loss=np.mean(losses) if losses else 0.0,
            equity_curve=pd.Series(equity_curve, index=df.index),
            returns=returns_series,
            signals=self.generate_signals(df),
            params={"atr_period": self.atr_period, "entry_period": self.entry_period,
                    "exit_period": self.exit_period, "max_units": self.max_units},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. KeltnerBreakout
# ─────────────────────────────────────────────────────────────────────────────

class KeltnerBreakout:
    """
    Keltner Channel breakout system.

    Long  when close > upper Keltner band (momentum breakout).
    Short when close < lower Keltner band.
    Exit  when price returns to midline (EMA).

    Parameters
    ----------
    period     : EMA and ATR period (default 20)
    multiplier : ATR multiplier for band width (default 2.0)
    """

    def __init__(self, period: int = 20, multiplier: float = 2.0):
        self.period = period
        self.multiplier = multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        upper, middle, lower = _keltner_bands(df, self.period, self.multiplier)
        close = df["close"]

        signals = pd.Series(0.0, index=df.index)
        position = 0

        warmup = self.period
        for i in range(warmup, len(df)):
            if any(np.isnan(x.iloc[i]) for x in [upper, middle, lower]):
                continue
            c = close.iloc[i]
            u = upper.iloc[i]
            m = middle.iloc[i]
            lo = lower.iloc[i]

            if position == 0:
                if c > u:
                    position = 1
                elif c < lo:
                    position = -1
            elif position == 1:
                if c < m:
                    position = 0
            elif position == -1:
                if c > m:
                    position = 0
            signals.iloc[i] = float(position)

        signals.iloc[:warmup] = np.nan
        return signals

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
        slippage_pct: float = 0.0001,
    ) -> BacktestResult:
        signals = self.generate_signals(df)
        result = _backtest_from_signals(df, signals, initial_equity, commission_pct, slippage_pct)
        result.params = {"period": self.period, "multiplier": self.multiplier}
        return result

    def get_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame with upper, middle, lower columns."""
        upper, middle, lower = _keltner_bands(df, self.period, self.multiplier)
        return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower}, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DonchianBreakout
# ─────────────────────────────────────────────────────────────────────────────

class DonchianBreakout:
    """
    Donchian Channel breakout system (Richard Donchian, 1970s).

    Entry: price breaks N-period high (long) or N-period low (short).
    Exit:  price crosses M-period opposite (default M = N // 2).

    Parameters
    ----------
    period      : lookback for entry channel (default 20)
    exit_period : lookback for exit channel (default None → period // 2)
    """

    def __init__(self, period: int = 20, exit_period: Optional[int] = None):
        self.period = period
        self.exit_period = exit_period if exit_period is not None else max(1, period // 2)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        high_entry = _rolling_high(df["high"], self.period)
        low_entry = _rolling_low(df["low"], self.period)
        high_exit = _rolling_high(df["high"], self.exit_period)
        low_exit = _rolling_low(df["low"], self.exit_period)
        close = df["close"]

        signals = pd.Series(0.0, index=df.index)
        position = 0
        warmup = max(self.period, self.exit_period)

        for i in range(warmup, len(df)):
            if any(np.isnan(x.iloc[i - 1]) for x in [high_entry, low_entry, high_exit, low_exit]):
                continue

            c = close.iloc[i]
            he = high_entry.iloc[i - 1]
            le = low_entry.iloc[i - 1]
            hx = high_exit.iloc[i - 1]
            lx = low_exit.iloc[i - 1]

            if position == 0:
                if c > he:
                    position = 1
                elif c < le:
                    position = -1
            elif position == 1:
                if c < lx:
                    position = 0
            elif position == -1:
                if c > hx:
                    position = 0
            signals.iloc[i] = float(position)

        signals.iloc[:warmup] = np.nan
        return signals

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
        slippage_pct: float = 0.0001,
    ) -> BacktestResult:
        signals = self.generate_signals(df)
        result = _backtest_from_signals(df, signals, initial_equity, commission_pct, slippage_pct)
        result.params = {"period": self.period, "exit_period": self.exit_period}
        return result

    def channel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame with upper and lower Donchian channel."""
        upper = _rolling_high(df["high"], self.period)
        lower = _rolling_low(df["low"], self.period)
        mid = (upper + lower) / 2
        return pd.DataFrame({"upper": upper, "lower": lower, "mid": mid}, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# 6. AdaptiveTrendFollowing
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveTrendFollowing:
    """
    Adaptive trend following using MESA-inspired adaptive moving average.

    The dominant market cycle length is estimated each bar, and the moving
    average period adapts accordingly (shorter in trending, longer in sideways).

    Entry/exit same as DualMovingAverage but with adaptive periods.

    Parameters
    ----------
    base_period    : base cycle period (adaptive range: base/2 to base*2)
    fast_fraction  : fast MA as fraction of adaptive period (default 0.25)
    """

    def __init__(self, base_period: int = 20, fast_fraction: float = 0.25):
        self.base_period = base_period
        self.fast_fraction = fast_fraction

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        adaptive_period = _mesa_adaptive_period(close, self.base_period)

        # Build adaptive fast and slow MAs
        fast_vals = np.full(len(df), np.nan)
        slow_vals = np.full(len(df), np.nan)
        fast_ema = None
        slow_ema = None

        for i in range(len(df)):
            ap = adaptive_period.iloc[i]
            if np.isnan(ap) or ap < 2:
                continue
            fast_p = max(2, int(ap * self.fast_fraction))
            slow_p = max(fast_p + 1, int(ap))
            fast_alpha = 2.0 / (fast_p + 1)
            slow_alpha = 2.0 / (slow_p + 1)
            c = close.iloc[i]
            fast_ema = c if fast_ema is None else fast_alpha * c + (1 - fast_alpha) * fast_ema
            slow_ema = c if slow_ema is None else slow_alpha * c + (1 - slow_alpha) * slow_ema
            fast_vals[i] = fast_ema
            slow_vals[i] = slow_ema

        fast_series = pd.Series(fast_vals, index=df.index)
        slow_series = pd.Series(slow_vals, index=df.index)

        raw = np.where(
            fast_series > slow_series, 1.0,
            np.where(fast_series < slow_series, -1.0, 0.0)
        )
        signals = pd.Series(raw, index=df.index, dtype=float)
        # NaN first base_period bars
        signals.iloc[:self.base_period] = np.nan
        return signals

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
        slippage_pct: float = 0.0001,
    ) -> BacktestResult:
        signals = self.generate_signals(df)
        result = _backtest_from_signals(df, signals, initial_equity, commission_pct, slippage_pct)
        result.params = {"base_period": self.base_period, "fast_fraction": self.fast_fraction}
        return result

    def get_adaptive_period(self, df: pd.DataFrame) -> pd.Series:
        """Returns the adaptive period estimate for each bar."""
        return _mesa_adaptive_period(df["close"], self.base_period)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: compare strategies on same data
# ─────────────────────────────────────────────────────────────────────────────

def compare_strategies(df: pd.DataFrame, strategies: list) -> pd.DataFrame:
    """
    Run all strategies on df and return a comparison DataFrame.

    Parameters
    ----------
    df         : OHLCV DataFrame
    strategies : list of strategy instances (any with .backtest() method)

    Returns
    -------
    pd.DataFrame with rows = strategies, columns = performance metrics
    """
    rows = []
    for strat in strategies:
        try:
            res = strat.backtest(df)
            rows.append({
                "strategy": strat.__class__.__name__,
                "params": str(res.params),
                "total_return": res.total_return,
                "cagr": res.cagr,
                "sharpe": res.sharpe,
                "sortino": res.sortino,
                "max_drawdown": res.max_drawdown,
                "calmar": res.calmar,
                "win_rate": res.win_rate,
                "profit_factor": res.profit_factor,
                "n_trades": res.n_trades,
            })
        except Exception as e:
            rows.append({"strategy": strat.__class__.__name__, "error": str(e)})

    return pd.DataFrame(rows).set_index("strategy")


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo / smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _make_demo_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    vol = rng.uniform(1_000, 10_000, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


if __name__ == "__main__":
    df = _make_demo_data(2000)
    strategies = [
        DualMovingAverage(20, 50),
        DualMovingAverage(10, 30, "sma"),
        TripleMovingAverage(10, 30, 100),
        TurtleSystem(20, 55, 20),
        KeltnerBreakout(20, 2.0),
        DonchianBreakout(20),
        AdaptiveTrendFollowing(20),
    ]
    cmp = compare_strategies(df, strategies)
    print(cmp.to_string())
