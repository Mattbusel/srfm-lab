"""
momentum.py — Pure momentum strategies.

References:
  - Moskowitz, Ooi, Pedersen (2012): "Time Series Momentum"
  - Jegadeesh, Titman (1993): Cross-sectional momentum
  - Antonacci (2014): "Dual Momentum Investing"
  - Daniel, Hirshleifer, Sun (2020): skewness-adjusted momentum
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Shared BacktestResult (mirrors trend_following.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
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
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    signals: pd.Series = field(default_factory=pd.Series)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
                f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
                f"Trades={self.n_trades}")


def _compute_stats(equity_curve: np.ndarray, trade_returns: List[float]) -> dict:
    n = len(equity_curve)
    initial = equity_curve[0]
    final = equity_curve[-1]
    total_return = final / initial - 1.0
    n_years = max(1, n / 252)
    cagr = (final / initial) ** (1.0 / n_years) - 1.0
    rets = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
    rets = np.concatenate([[0], rets])
    std = rets.std()
    sharpe = rets.mean() / std * math.sqrt(252) if std > 0 else 0.0
    down = rets[rets < 0]
    sortino_d = np.std(down) if len(down) > 0 else 1e-9
    sortino = rets.mean() / sortino_d * math.sqrt(252)
    pk = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - pk) / (pk + 1e-9)
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    win_rate = len(wins) / len(trade_returns) if trade_returns else 0.0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    return dict(
        total_return=total_return, cagr=cagr, sharpe=sharpe, sortino=sortino,
        max_drawdown=max_dd, calmar=calmar, win_rate=win_rate, profit_factor=pf,
        n_trades=len(trade_returns),
        avg_trade_return=float(np.mean(trade_returns)) if trade_returns else 0.0,
        returns=pd.Series(rets),
    )


def _backtest_signal_series(
    close: np.ndarray,
    signals: np.ndarray,
    initial_equity: float = 1_000_000,
    cost: float = 0.0002,
) -> Tuple[np.ndarray, List[float]]:
    """
    Generic signal-to-equity engine. signals applied with 1-bar lag.
    Returns (equity_curve, trade_returns).
    """
    n = len(close)
    equity = initial_equity
    equity_curve = np.full(n, initial_equity, dtype=float)
    trade_returns = []
    prev_sig = 0.0
    position = 0.0
    entry_price = None
    entry_direction = 0.0

    for i in range(1, n):
        if np.isnan(signals[i - 1]):
            equity_curve[i] = equity
            continue

        new_sig = float(signals[i - 1])
        if new_sig != position:
            # Record trade
            if entry_price is not None and entry_direction != 0:
                ret = entry_direction * ((close[i] - entry_price) / entry_price - cost * 2)
                trade_returns.append(ret)
            if new_sig != 0:
                entry_price = close[i]
                entry_direction = np.sign(new_sig)
            else:
                entry_price = None
                entry_direction = 0.0
            position = new_sig

        if position != 0:
            bar_ret = (close[i] - close[i - 1]) / (close[i - 1] + 1e-9)
            equity *= (1 + position * bar_ret)
        equity_curve[i] = equity

    return equity_curve, trade_returns


# ─────────────────────────────────────────────────────────────────────────────
# 1. TimeSeriesMomentum (TSMOM)
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesMomentum:
    """
    Time-Series Momentum: Moskowitz, Ooi, Pedersen (2012).

    Signal = sign of past lookback-period return.
    Position size = scaled by ex-ante volatility to target constant vol.

    The original paper used 12-month lookback (excluding most recent month)
    and targeted 40% annual volatility.

    Parameters
    ----------
    lookback    : lookback in bars for momentum signal (default 252)
    skip_recent : bars to skip at end of lookback to avoid reversal (default 21)
    target_vol  : annualized vol target (default 0.40)
    vol_window  : rolling window for vol estimate (default 63)
    """

    def __init__(
        self,
        lookback: int = 252,
        skip_recent: int = 21,
        target_vol: float = 0.40,
        vol_window: int = 63,
    ):
        self.lookback = lookback
        self.skip_recent = skip_recent
        self.target_vol = target_vol
        self.vol_window = vol_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Returns continuous position size (signed, may exceed -1/+1)."""
        close = df["close"]
        returns = close.pct_change()

        # Past return: lookback bars ago to skip_recent bars ago
        start_ret = close.shift(self.lookback)
        end_ret = close.shift(self.skip_recent)
        past_return = (end_ret - start_ret) / (start_ret.abs() + 1e-9)

        # Realized vol estimate
        realized_vol = returns.rolling(self.vol_window, min_periods=max(5, self.vol_window // 3)).std() * math.sqrt(252)
        realized_vol = realized_vol.replace(0, np.nan).ffill()

        # Signal = sign(past_return) * target_vol / realized_vol
        signal_direction = np.sign(past_return)
        # Avoid dividing by zero
        safe_vol = realized_vol.where(realized_vol > 0.01, 0.01)
        position_size = signal_direction * (self.target_vol / safe_vol)

        # Cap at ±2 to avoid extreme leverage
        position_size = position_size.clip(-2.0, 2.0)
        position_size.iloc[: self.lookback + self.skip_recent] = np.nan
        return position_size

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        signals = self.generate_signals(df)
        close = df["close"].values
        sig = signals.values
        eq, trades = _backtest_signal_series(close, sig, initial_equity, commission_pct)
        stats = _compute_stats(eq, trades)
        idx = df.index
        return BacktestResult(
            **{k: v for k, v in stats.items() if k != "returns"},
            equity_curve=pd.Series(eq, index=idx),
            returns=pd.Series(stats["returns"].values, index=idx),
            signals=signals,
            params={"lookback": self.lookback, "skip_recent": self.skip_recent,
                    "target_vol": self.target_vol},
        )

    def momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """Returns raw signed momentum score (without vol scaling)."""
        close = df["close"]
        start_ret = close.shift(self.lookback)
        end_ret = close.shift(self.skip_recent)
        return (end_ret - start_ret) / (start_ret.abs() + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CrossSectionalMomentum
# ─────────────────────────────────────────────────────────────────────────────

class CrossSectionalMomentum:
    """
    Cross-sectional momentum: rank assets, long top decile, short bottom decile.

    Jegadeesh & Titman (1993): 12-month formation, 1-month holding.

    Parameters
    ----------
    universe    : list of column names in the prices DataFrame
    lookback    : momentum lookback in bars (default 252)
    hold        : holding period in bars (default 21)
    n_top       : number of assets to go long (default None → top 20%)
    n_bottom    : number of assets to short (default None → bottom 20%)
    skip_recent : bars to skip at end (default 21)
    """

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        lookback: int = 252,
        hold: int = 21,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        skip_recent: int = 21,
    ):
        self.universe = universe
        self.lookback = lookback
        self.hold = hold
        self.n_top = n_top
        self.n_bottom = n_bottom
        self.skip_recent = skip_recent

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame of portfolio weights over time.
        prices: columns = assets, rows = time.
        Weights are rebalanced every hold bars.
        """
        cols = self.universe if self.universe is not None else list(prices.columns)
        px = prices[cols]
        n_assets = len(cols)

        n_top = self.n_top if self.n_top is not None else max(1, n_assets // 5)
        n_bottom = self.n_bottom if self.n_bottom is not None else max(1, n_assets // 5)

        weights = pd.DataFrame(0.0, index=px.index, columns=cols)
        warmup = self.lookback + self.skip_recent

        for i in range(warmup, len(px), self.hold):
            row = px.iloc[i]
            past_row = px.iloc[i - self.lookback + self.skip_recent - 1]
            skip_row = px.iloc[i - self.skip_recent] if i >= self.skip_recent else past_row

            mom = (skip_row - past_row) / (past_row.abs() + 1e-9)
            ranked = mom.rank(ascending=True)

            w = pd.Series(0.0, index=cols)
            long_assets = ranked.nlargest(n_top).index
            short_assets = ranked.nsmallest(n_bottom).index
            w[long_assets] = 1.0 / n_top
            w[short_assets] = -1.0 / n_bottom

            end_i = min(i + self.hold, len(px))
            weights.iloc[i:end_i] = w.values

        return weights

    def backtest_universe(
        self,
        prices: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.001,
    ) -> BacktestResult:
        """
        Backtest cross-sectional strategy on a universe of prices.
        prices: pd.DataFrame, columns = assets, rows = dates.
        """
        weights = self.generate_weights(prices)
        cols = list(weights.columns)
        returns = prices[cols].pct_change().fillna(0)

        equity = initial_equity
        eq_curve = np.full(len(prices), initial_equity, dtype=float)
        trade_returns = []

        for i in range(1, len(prices)):
            w = weights.iloc[i].values
            r = returns.iloc[i].values
            port_ret = float(np.dot(w, r))
            # Turnover cost
            prev_w = weights.iloc[i - 1].values
            turnover = np.abs(w - prev_w).sum() / 2
            port_ret -= turnover * commission_pct
            equity *= (1 + port_ret)
            eq_curve[i] = equity
            if port_ret != 0:
                trade_returns.append(port_ret)

        stats = _compute_stats(eq_curve, trade_returns)
        idx = prices.index
        return BacktestResult(
            **{k: v for k, v in stats.items() if k != "returns"},
            equity_curve=pd.Series(eq_curve, index=idx),
            returns=pd.Series(stats["returns"].values, index=idx),
            params={"lookback": self.lookback, "hold": self.hold},
        )

    def rank_assets(self, prices: pd.DataFrame, at_date=None) -> pd.Series:
        """Rank assets by momentum at a specific date (or latest available)."""
        cols = self.universe if self.universe is not None else list(prices.columns)
        px = prices[cols]
        if at_date is not None:
            px = px.loc[:at_date]
        if len(px) < self.lookback:
            raise ValueError(f"Not enough data: need {self.lookback} bars")
        current = px.iloc[-self.skip_recent]
        past = px.iloc[-(self.lookback)]
        mom = (current - past) / (past.abs() + 1e-9)
        return mom.sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DualMomentum
# ─────────────────────────────────────────────────────────────────────────────

class DualMomentum:
    """
    Gary Antonacci's Dual Momentum (2014).

    Combines absolute and relative momentum:
    1. Relative momentum: pick the best-performing asset
    2. Absolute momentum: only hold if the winner has positive absolute return
       Otherwise hold cash/bonds (represented by a cash_return series).

    Parameters
    ----------
    lookback       : momentum lookback in bars (default 252)
    abs_threshold  : minimum return to qualify for absolute momentum (default 0.0)
    """

    def __init__(self, lookback: int = 252, abs_threshold: float = 0.0):
        self.lookback = lookback
        self.abs_threshold = abs_threshold

    def generate_signals(
        self,
        prices: pd.DataFrame,
        risk_asset_col: str = "equity",
        safe_asset_col: Optional[str] = None,
    ) -> pd.Series:
        """
        prices: DataFrame with at least one column for the risky asset.
        Returns signal: +1 = long risk asset, 0 = hold safe asset/cash.

        If prices has two columns, the first is the risky asset,
        the second is the alternative risky asset for relative comparison.
        Then absolute momentum filter is applied.
        """
        if risk_asset_col not in prices.columns:
            risk_asset_col = prices.columns[0]

        risky = prices[risk_asset_col]

        # Absolute momentum: past return over lookback
        past_return = (risky - risky.shift(self.lookback)) / (risky.shift(self.lookback).abs() + 1e-9)

        # If we have a second asset, use relative momentum too
        if len(prices.columns) > 1 and safe_asset_col is not None and safe_asset_col in prices.columns:
            safe = prices[safe_asset_col]
            safe_past_return = (safe - safe.shift(self.lookback)) / (safe.shift(self.lookback).abs() + 1e-9)
            # Relative: prefer risky if risky > safe
            relative_better = past_return > safe_past_return
        else:
            relative_better = pd.Series(True, index=prices.index)

        # Dual momentum: long only if absolute AND relative positive
        signal = pd.Series(0.0, index=prices.index)
        long_cond = (past_return > self.abs_threshold) & relative_better
        signal[long_cond] = 1.0
        signal.iloc[: self.lookback] = np.nan
        return signal

    def backtest(
        self,
        df: pd.DataFrame,
        risk_asset_col: str = "close",
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        """Single asset dual momentum backtest."""
        # For single asset: just absolute momentum
        close = df["close"]
        past_return = (close - close.shift(self.lookback)) / (close.shift(self.lookback).abs() + 1e-9)
        signal = pd.Series(0.0, index=df.index)
        signal[past_return > self.abs_threshold] = 1.0
        signal.iloc[: self.lookback] = np.nan

        eq, trades = _backtest_signal_series(df["close"].values, signal.values, initial_equity, commission_pct)
        stats = _compute_stats(eq, trades)
        return BacktestResult(
            **{k: v for k, v in stats.items() if k != "returns"},
            equity_curve=pd.Series(eq, index=df.index),
            returns=pd.Series(stats["returns"].values, index=df.index),
            signals=signal,
            params={"lookback": self.lookback, "abs_threshold": self.abs_threshold},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. RiskAdjustedMomentum
# ─────────────────────────────────────────────────────────────────────────────

class RiskAdjustedMomentum:
    """
    Momentum scaled by realized volatility: momentum / vol = Sharpe-like signal.

    This adjusts for the fact that high-return assets may have high volatility.
    Signal = past_return / realized_volatility (annualized Sharpe of past returns).

    Parameters
    ----------
    lookback   : momentum lookback (default 252)
    vol_window : rolling volatility window (default 63)
    threshold  : signal threshold to go long/short (default 0)
    """

    def __init__(self, lookback: int = 252, vol_window: int = 63, threshold: float = 0.0):
        self.lookback = lookback
        self.vol_window = vol_window
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        returns = close.pct_change()

        past_return = (close - close.shift(self.lookback)) / (close.shift(self.lookback).abs() + 1e-9)
        realized_vol = returns.rolling(self.vol_window, min_periods=self.vol_window // 2).std() * math.sqrt(252)
        realized_vol = realized_vol.replace(0, np.nan).fillna(method="ffill")

        # Risk-adjusted signal
        ra_signal = past_return / (realized_vol + 1e-9)

        signal = pd.Series(0.0, index=df.index)
        signal[ra_signal > self.threshold] = 1.0
        signal[ra_signal < -self.threshold] = -1.0
        signal.iloc[: self.lookback] = np.nan
        return signal

    def ra_score(self, df: pd.DataFrame) -> pd.Series:
        """Return the raw risk-adjusted momentum score."""
        close = df["close"]
        returns = close.pct_change()
        past_return = (close - close.shift(self.lookback)) / (close.shift(self.lookback).abs() + 1e-9)
        realized_vol = returns.rolling(self.vol_window, min_periods=self.vol_window // 2).std() * math.sqrt(252)
        realized_vol = realized_vol.replace(0, np.nan).fillna(method="ffill")
        return past_return / (realized_vol + 1e-9)

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        signals = self.generate_signals(df)
        eq, trades = _backtest_signal_series(df["close"].values, signals.values, initial_equity, commission_pct)
        stats = _compute_stats(eq, trades)
        return BacktestResult(
            **{k: v for k, v in stats.items() if k != "returns"},
            equity_curve=pd.Series(eq, index=df.index),
            returns=pd.Series(stats["returns"].values, index=df.index),
            signals=signals,
            params={"lookback": self.lookback, "vol_window": self.vol_window},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. MomentumWithFilter
# ─────────────────────────────────────────────────────────────────────────────

class MomentumWithFilter:
    """
    Momentum with a long-term trend filter (200-day MA filter).

    Only take long momentum signals when price > 200d MA.
    Only take short signals when price < 200d MA.
    This classic filter (Faber 2007) dramatically improves risk-adjusted returns.

    Parameters
    ----------
    trend_period : period for trend filter MA (default 200)
    mom_period   : momentum lookback (default 252)
    skip_recent  : bars to skip at end of lookback (default 21)
    """

    def __init__(
        self,
        trend_period: int = 200,
        mom_period: int = 252,
        skip_recent: int = 21,
    ):
        self.trend_period = trend_period
        self.mom_period = mom_period
        self.skip_recent = skip_recent

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        trend_ma = close.ewm(span=self.trend_period, adjust=False).mean()

        # Momentum signal
        past = close.shift(self.mom_period)
        recent = close.shift(self.skip_recent)
        mom_signal = np.sign((recent - past) / (past.abs() + 1e-9))

        # Apply trend filter
        filtered = mom_signal.copy()
        filtered[(mom_signal > 0) & (close < trend_ma)] = 0.0
        filtered[(mom_signal < 0) & (close > trend_ma)] = 0.0

        warmup = max(self.trend_period, self.mom_period + self.skip_recent)
        filtered.iloc[:warmup] = np.nan
        return filtered

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        signals = self.generate_signals(df)
        eq, trades = _backtest_signal_series(df["close"].values, signals.values, initial_equity, commission_pct)
        stats = _compute_stats(eq, trades)
        return BacktestResult(
            **{k: v for k, v in stats.items() if k != "returns"},
            equity_curve=pd.Series(eq, index=df.index),
            returns=pd.Series(stats["returns"].values, index=df.index),
            signals=signals,
            params={"trend_period": self.trend_period, "mom_period": self.mom_period},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. SkewnessAdjustedMomentum
# ─────────────────────────────────────────────────────────────────────────────

class SkewnessAdjustedMomentum:
    """
    Skewness-adjusted momentum to reduce crash risk.

    Inspired by Daniel, Hirshleifer, Sun (2020) — momentum strategies can
    be improved by adjusting for negative skewness (crash risk).

    Adjustment: reduce position when realized skewness is very negative
    (indicating higher crash probability for momentum portfolio).

    Signal = sign(past_return) * max(0, 1 + skew_weight * skewness)

    Parameters
    ----------
    lookback      : momentum lookback (default 252)
    skew_window   : window for skewness estimation (default 63)
    skew_weight   : how much to adjust for skewness (default 0.5)
    """

    def __init__(self, lookback: int = 252, skew_window: int = 63, skew_weight: float = 0.5):
        self.lookback = lookback
        self.skew_window = skew_window
        self.skew_weight = skew_weight

    def _rolling_skewness(self, returns: pd.Series) -> pd.Series:
        """Rolling skewness."""
        def skew_fn(x):
            if len(x) < 3:
                return 0.0
            mu = x.mean()
            sigma = x.std()
            if sigma == 0:
                return 0.0
            return float(((x - mu) ** 3).mean() / (sigma ** 3))
        return returns.rolling(self.skew_window, min_periods=max(5, self.skew_window // 3)).apply(skew_fn, raw=False)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        returns = close.pct_change()

        # Raw momentum signal
        past_return = (close.shift(21) - close.shift(self.lookback)) / (close.shift(self.lookback).abs() + 1e-9)
        mom_direction = np.sign(past_return)

        # Rolling skewness of recent returns
        skewness = self._rolling_skewness(returns)

        # Skewness adjustment: positive skew → increase signal, negative → decrease
        # Clamp adjustment to [0, 2]
        adjustment = (1.0 + self.skew_weight * skewness).clip(0.0, 2.0)

        signal = mom_direction * adjustment
        # Normalize to [-1, 1]
        signal = signal.clip(-1.0, 1.0)

        warmup = max(self.lookback, self.skew_window)
        signal.iloc[:warmup] = np.nan
        return signal

    def skewness_series(self, df: pd.DataFrame) -> pd.Series:
        """Return rolling skewness series."""
        return self._rolling_skewness(df["close"].pct_change())

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        signals = self.generate_signals(df)
        eq, trades = _backtest_signal_series(df["close"].values, signals.values, initial_equity, commission_pct)
        stats = _compute_stats(eq, trades)
        return BacktestResult(
            **{k: v for k, v in stats.items() if k != "returns"},
            equity_curve=pd.Series(eq, index=df.index),
            returns=pd.Series(stats["returns"].values, index=df.index),
            signals=signals,
            params={"lookback": self.lookback, "skew_window": self.skew_window,
                    "skew_weight": self.skew_weight},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1500
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "volume": 1000
    }, index=idx)

    for Cls, kwargs in [
        (TimeSeriesMomentum, {"lookback": 252}),
        (DualMomentum, {"lookback": 252}),
        (RiskAdjustedMomentum, {"lookback": 252}),
        (MomentumWithFilter, {"trend_period": 200, "mom_period": 252}),
        (SkewnessAdjustedMomentum, {"lookback": 252}),
    ]:
        strat = Cls(**kwargs)
        res = strat.backtest(df)
        print(f"{Cls.__name__}: {res.summary()}")
