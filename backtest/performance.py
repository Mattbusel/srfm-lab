"""
performance.py -- Performance analytics for LARSA backtesting.

Implements Sharpe, Sortino, Calmar, drawdown analysis, rolling stats,
trade journal, and bootstrap confidence intervals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# Annualization factors
BARS_PER_DAY = 26        # 15-min bars in a 6.5h trading day
BARS_PER_YEAR = BARS_PER_DAY * 252


# ---------------------------------------------------------------------------
# Drawdown Analyzer
# ---------------------------------------------------------------------------

class DrawdownAnalyzer:
    """
    Computes full drawdown statistics from an equity curve.

    Metrics:
      - Max drawdown (peak to trough)
      - Max drawdown duration (bars)
      - Average drawdown depth
      - Underwater equity curve (fraction below peak)
      - Recovery time (bars to return to prior peak after trough)
    """

    def __init__(self, equity: pd.Series):
        self.equity = equity.copy()
        self._compute()

    def _compute(self) -> None:
        eq = self.equity
        peak = eq.cummax()
        self.underwater = (eq - peak) / peak  # always <= 0
        self.drawdown = self.underwater.copy()

        # Identify drawdown periods
        in_dd = self.underwater < -1e-8
        dd_starts = []
        dd_ends = []
        dd_troughs = []
        dd_recoveries = []

        i = 0
        while i < len(in_dd):
            if in_dd.iloc[i]:
                start = i
                # Find end of drawdown (return to previous peak)
                trough_val = self.underwater.iloc[i]
                trough_idx = i
                j = i
                while j < len(in_dd) and in_dd.iloc[j]:
                    if self.underwater.iloc[j] < trough_val:
                        trough_val = self.underwater.iloc[j]
                        trough_idx = j
                    j += 1
                end = j - 1
                dd_starts.append(start)
                dd_ends.append(end)
                dd_troughs.append(trough_val)
                # Recovery: bars from trough to recovery (or end of data)
                dd_recoveries.append(end - trough_idx)
                i = j
            else:
                i += 1

        self._drawdown_periods = list(zip(dd_starts, dd_ends, dd_troughs, dd_recoveries))

    @property
    def max_drawdown(self) -> float:
        return float(self.underwater.min())

    @property
    def max_drawdown_duration(self) -> int:
        if not self._drawdown_periods:
            return 0
        return max(end - start for start, end, _, _ in self._drawdown_periods)

    @property
    def avg_drawdown(self) -> float:
        if not self._drawdown_periods:
            return 0.0
        return float(np.mean([dd for _, _, dd, _ in self._drawdown_periods]))

    @property
    def avg_recovery_bars(self) -> float:
        if not self._drawdown_periods:
            return 0.0
        return float(np.mean([rec for _, _, _, rec in self._drawdown_periods]))

    @property
    def num_drawdown_periods(self) -> int:
        return len(self._drawdown_periods)

    def summary(self) -> Dict[str, Any]:
        return {
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_bars": self.max_drawdown_duration,
            "avg_drawdown": self.avg_drawdown,
            "avg_recovery_bars": self.avg_recovery_bars,
            "num_drawdown_periods": self.num_drawdown_periods,
        }

    def get_drawdown_series(self) -> pd.Series:
        return self.underwater


# ---------------------------------------------------------------------------
# Rolling Stats
# ---------------------------------------------------------------------------

class RollingStats:
    """
    Rolling Sharpe, Sortino, and volatility with significance testing.

    Uses a 30-day (780-bar) and 90-day (2340-bar) rolling window.
    Significance test: t-test on the rolling mean return vs. zero.
    """

    def __init__(self, returns: pd.Series):
        self.returns = returns
        self._ann_factor = BARS_PER_YEAR

    def rolling_sharpe(self, window_bars: int = 780, min_periods: int = 50) -> pd.Series:
        """Rolling annualized Sharpe ratio."""
        r = self.returns
        roll_mean = r.rolling(window=window_bars, min_periods=min_periods).mean()
        roll_std = r.rolling(window=window_bars, min_periods=min_periods).std()
        sharpe = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(self._ann_factor)
        return sharpe

    def rolling_sortino(self, window_bars: int = 780, min_periods: int = 50) -> pd.Series:
        """Rolling annualized Sortino ratio (downside deviation denominator)."""
        r = self.returns

        def sortino_func(x: np.ndarray) -> float:
            if len(x) < 5:
                return np.nan
            mean_r = np.mean(x)
            downside = x[x < 0]
            if len(downside) < 2:
                return np.nan
            downside_std = np.std(downside, ddof=1)
            if downside_std < 1e-12:
                return np.nan
            return mean_r / downside_std * np.sqrt(self._ann_factor)

        return r.rolling(window=window_bars, min_periods=min_periods).apply(sortino_func, raw=True)

    def rolling_volatility(self, window_bars: int = 260) -> pd.Series:
        """Rolling annualized volatility."""
        return self.returns.rolling(window_bars).std() * np.sqrt(self._ann_factor)

    def sharpe_significance(self, window_bars: int = 780) -> pd.Series:
        """
        Rolling p-value for H0: mean return = 0.
        Low p-value means Sharpe is statistically significant.
        """
        r = self.returns

        def ttest_pval(x: np.ndarray) -> float:
            if len(x) < 10:
                return np.nan
            t_stat, p_val = scipy_stats.ttest_1samp(x, 0)
            return float(p_val)

        return r.rolling(window=window_bars, min_periods=50).apply(ttest_pval, raw=True)

    def ewm_sharpe(self, halflife_bars: int = 260) -> pd.Series:
        """Exponentially weighted Sharpe (more responsive to recent data)."""
        r = self.returns
        ewm_mean = r.ewm(halflife=halflife_bars).mean()
        ewm_std = r.ewm(halflife=halflife_bars).std()
        return (ewm_mean / ewm_std.replace(0, np.nan)) * np.sqrt(self._ann_factor)


# ---------------------------------------------------------------------------
# Trade Journal
# ---------------------------------------------------------------------------

class TradeJournal:
    """
    Maintains a per-trade P&L record with entry/exit metadata.
    Supports signal attribution: tracks which sub-signal drove each trade.
    """

    def __init__(self):
        self._trades: List[Dict[str, Any]] = []
        self._open_trades: Dict[str, Dict[str, Any]] = {}  # symbol -> open trade

    def open_trade(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        direction: str,
        quantity: float,
        entry_price: float,
        entry_signal: str = "",
        bh_mass: float = 0.0,
        hurst: float = 0.5,
        garch_vol: float = 0.0,
        regime: str = "",
    ) -> None:
        self._open_trades[symbol] = {
            "symbol": symbol,
            "entry_time": timestamp,
            "direction": direction,
            "quantity": quantity,
            "entry_price": entry_price,
            "entry_signal": entry_signal,
            "bh_mass_entry": bh_mass,
            "hurst_entry": hurst,
            "garch_vol_entry": garch_vol,
            "regime_entry": regime,
            "commission_entry": 0.0,
            "slippage_entry": 0.0,
        }

    def close_trade(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        exit_price: float,
        exit_qty: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        exit_signal: str = "",
    ) -> Optional[Dict[str, Any]]:
        if symbol not in self._open_trades:
            return None

        trade = dict(self._open_trades[symbol])
        trade["exit_time"] = timestamp
        trade["exit_price"] = exit_price
        trade["exit_qty"] = exit_qty
        trade["exit_signal"] = exit_signal
        trade["commission_exit"] = commission
        trade["slippage_exit"] = slippage

        close_qty = min(abs(exit_qty), abs(trade["quantity"]))
        sign = 1 if trade["direction"] == "LONG" else -1
        trade["gross_pnl"] = sign * (exit_price - trade["entry_price"]) * close_qty
        trade["total_commission"] = trade["commission_entry"] + commission
        trade["total_slippage"] = trade["slippage_entry"] + slippage
        trade["net_pnl"] = trade["gross_pnl"] - trade["total_commission"] - trade["total_slippage"]

        delta = timestamp - trade["entry_time"]
        trade["hold_bars"] = int(delta.total_seconds() / 900)

        notional = abs(trade["entry_price"] * trade["quantity"])
        trade["return_pct"] = trade["net_pnl"] / notional if notional > 0 else 0.0

        self._trades.append(trade)

        # Remove from open trades if fully closed
        if abs(exit_qty) >= abs(trade["quantity"]) - 1e-8:
            del self._open_trades[symbol]
        else:
            self._open_trades[symbol]["quantity"] -= exit_qty

        return trade

    def to_dataframe(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame()
        return pd.DataFrame(self._trades)

    @property
    def num_trades(self) -> int:
        return len(self._trades)

    def win_rate(self) -> float:
        if not self._trades:
            return 0.0
        wins = sum(1 for t in self._trades if t.get("net_pnl", 0) > 0)
        return wins / len(self._trades)

    def avg_hold_bars(self) -> float:
        if not self._trades:
            return 0.0
        return float(np.mean([t.get("hold_bars", 0) for t in self._trades]))

    def profit_factor(self) -> float:
        gross_wins = sum(t["net_pnl"] for t in self._trades if t.get("net_pnl", 0) > 0)
        gross_losses = abs(sum(t["net_pnl"] for t in self._trades if t.get("net_pnl", 0) < 0))
        return gross_wins / (gross_losses + 1e-12)

    def avg_win_loss_ratio(self) -> float:
        wins = [t["net_pnl"] for t in self._trades if t.get("net_pnl", 0) > 0]
        losses = [abs(t["net_pnl"]) for t in self._trades if t.get("net_pnl", 0) < 0]
        if not wins or not losses:
            return 0.0
        return float(np.mean(wins) / np.mean(losses))

    def signal_attribution(self) -> pd.DataFrame:
        """P&L breakdown by entry signal type."""
        df = self.to_dataframe()
        if df.empty or "entry_signal" not in df.columns:
            return pd.DataFrame()
        return df.groupby("entry_signal")["net_pnl"].agg(
            count="count",
            total_pnl="sum",
            avg_pnl="mean",
            win_rate=lambda x: (x > 0).mean(),
        )


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

class BootstrapCI:
    """
    Bootstrap confidence intervals for Sharpe ratio and other statistics.
    Uses the deflated Sharpe ratio methodology (Bailey & Lopez de Prado).
    """

    def __init__(self, n_bootstrap: int = 1000, confidence: float = 0.95, seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.rng = np.random.default_rng(seed)

    def sharpe_ci(
        self, returns: np.ndarray, ann_factor: float = BARS_PER_YEAR
    ) -> Tuple[float, float, float]:
        """
        Returns (sharpe_ratio, lower_CI, upper_CI) using block bootstrap.
        Uses block size = sqrt(n) to preserve autocorrelation structure.
        """
        n = len(returns)
        if n < 20:
            return 0.0, 0.0, 0.0

        sharpe = self._point_sharpe(returns, ann_factor)
        block_size = max(1, int(np.sqrt(n)))

        boot_sharpes = []
        for _ in range(self.n_bootstrap):
            sample = self._block_bootstrap(returns, block_size)
            boot_sharpes.append(self._point_sharpe(sample, ann_factor))

        boot_arr = np.array(boot_sharpes)
        alpha = 1 - self.confidence
        lo = float(np.percentile(boot_arr, alpha / 2 * 100))
        hi = float(np.percentile(boot_arr, (1 - alpha / 2) * 100))
        return float(sharpe), lo, hi

    def _block_bootstrap(self, data: np.ndarray, block_size: int) -> np.ndarray:
        n = len(data)
        n_blocks = int(np.ceil(n / block_size))
        starts = self.rng.integers(0, n - block_size + 1, n_blocks)
        blocks = [data[s : s + block_size] for s in starts]
        sample = np.concatenate(blocks)[:n]
        return sample

    def _point_sharpe(self, returns: np.ndarray, ann_factor: float) -> float:
        std = np.std(returns, ddof=1)
        if std < 1e-12:
            return 0.0
        return float(np.mean(returns) / std * np.sqrt(ann_factor))

    def deflated_sharpe_ratio(
        self,
        sharpe_observed: float,
        returns: np.ndarray,
        n_trials: int = 1,
    ) -> float:
        """
        Deflated Sharpe Ratio (DSR) test (Bailey & Lopez de Prado, 2014).
        Returns the probability that the observed Sharpe is not due to luck.

        DSR = Phi( (SR - SR*) * sqrt(T-1) / sqrt(1 - gamma3*SR + gamma4*SR^2/4) )
        where SR* = adjusted benchmark Sharpe from multiple comparisons.
        """
        n = len(returns)
        if n < 4 or sharpe_observed <= 0:
            return 0.0

        # Higher moments
        skew = float(scipy_stats.skew(returns))
        kurt = float(scipy_stats.kurtosis(returns))  # excess kurtosis

        # Benchmark SR with Bonferroni adjustment for n_trials
        from scipy.special import ndtri
        z = float(ndtri(1 - 1.0 / n_trials)) if n_trials > 1 else 0.0
        sr_star = z / np.sqrt(n - 1)

        # Variance of Sharpe estimator
        sr_var_numerator = (n - 1) * (1 - skew * sharpe_observed + (kurt / 4.0) * sharpe_observed**2)
        sr_se = np.sqrt(max(sr_var_numerator, 0) / (n - 1))

        if sr_se < 1e-12:
            return 1.0 if sharpe_observed > sr_star else 0.0

        z_stat = (sharpe_observed - sr_star) / sr_se
        dsr = float(scipy_stats.norm.cdf(z_stat))
        return dsr

    def cvar(
        self, returns: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Returns (VaR, CVaR) at given confidence level."""
        sorted_r = np.sort(returns)
        cutoff_idx = int((1 - confidence) * len(sorted_r))
        var = float(-sorted_r[max(cutoff_idx, 0)])
        cvar = float(-np.mean(sorted_r[: max(cutoff_idx, 1)]))
        return var, cvar


# ---------------------------------------------------------------------------
# Main Performance Analyzer
# ---------------------------------------------------------------------------

class PerformanceAnalyzer:
    """
    Comprehensive performance analytics for a backtest result.

    Input: equity curve as pd.Series (bar-frequency), trade log as DataFrame.
    Computes all standard quant metrics plus LARSA-specific ones.
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        trade_log: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.05,  # annual
        bars_per_year: int = BARS_PER_YEAR,
    ):
        self.equity = equity_curve.dropna()
        self.trade_log = trade_log if trade_log is not None else pd.DataFrame()
        self.rfr = risk_free_rate
        self.ann_factor = bars_per_year

        # Compute bar returns
        self.returns = self.equity.pct_change().dropna()
        self.log_returns = np.log(self.equity / self.equity.shift(1)).dropna()

        # Sub-analyzers
        self.drawdown = DrawdownAnalyzer(self.equity)
        self.rolling = RollingStats(self.returns)
        self.bootstrap = BootstrapCI()

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def total_return(self) -> float:
        if len(self.equity) < 2:
            return 0.0
        return float((self.equity.iloc[-1] - self.equity.iloc[0]) / self.equity.iloc[0])

    def annualized_return(self) -> float:
        n = len(self.returns)
        if n < 2:
            return 0.0
        total_r = self.total_return()
        years = n / self.ann_factor
        return float((1 + total_r) ** (1 / years) - 1) if years > 0 else 0.0

    def annualized_vol(self) -> float:
        return float(self.returns.std() * np.sqrt(self.ann_factor))

    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio."""
        bar_rfr = self.rfr / self.ann_factor
        excess = self.returns - bar_rfr
        std = float(excess.std())
        if std < 1e-12:
            return 0.0
        return float(excess.mean() / std * np.sqrt(self.ann_factor))

    def sortino_ratio(self) -> float:
        """Annualized Sortino (downside deviation denominator)."""
        bar_rfr = self.rfr / self.ann_factor
        excess = self.returns - bar_rfr
        downside = excess[excess < 0]
        if len(downside) < 2:
            return float("inf") if excess.mean() > 0 else 0.0
        downside_std = float(downside.std())
        if downside_std < 1e-12:
            return 0.0
        return float(excess.mean() / downside_std * np.sqrt(self.ann_factor))

    def calmar_ratio(self) -> float:
        """Annualized return / max drawdown."""
        ann_r = self.annualized_return()
        max_dd = abs(self.drawdown.max_drawdown)
        if max_dd < 1e-8:
            return float("inf") if ann_r > 0 else 0.0
        return ann_r / max_dd

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """Omega ratio: sum of gains above threshold / sum of losses below."""
        bar_thresh = threshold / self.ann_factor
        gains = self.returns[self.returns > bar_thresh] - bar_thresh
        losses = bar_thresh - self.returns[self.returns <= bar_thresh]
        if losses.sum() < 1e-12:
            return float("inf")
        return float(gains.sum() / losses.sum())

    def information_ratio(self, benchmark_returns: pd.Series) -> float:
        """IR against a benchmark return series."""
        active = self.returns - benchmark_returns.reindex(self.returns.index, fill_value=0)
        te = active.std()
        if te < 1e-12:
            return 0.0
        return float(active.mean() / te * np.sqrt(self.ann_factor))

    # ------------------------------------------------------------------
    # Trade statistics
    # ------------------------------------------------------------------

    def win_rate(self) -> float:
        if self.trade_log.empty or "net_pnl" not in self.trade_log.columns:
            return 0.0
        return float((self.trade_log["net_pnl"] > 0).mean())

    def profit_factor(self) -> float:
        if self.trade_log.empty or "net_pnl" not in self.trade_log.columns:
            return 0.0
        wins = self.trade_log.loc[self.trade_log["net_pnl"] > 0, "net_pnl"].sum()
        losses = abs(self.trade_log.loc[self.trade_log["net_pnl"] < 0, "net_pnl"].sum())
        return float(wins / (losses + 1e-12))

    def avg_hold_bars(self) -> float:
        if self.trade_log.empty or "hold_bars" not in self.trade_log.columns:
            return 0.0
        return float(self.trade_log["hold_bars"].mean())

    def avg_trade_return(self) -> float:
        if self.trade_log.empty or "return_pct" not in self.trade_log.columns:
            return 0.0
        return float(self.trade_log["return_pct"].mean())

    def expectancy(self) -> float:
        """Kelly expectancy = win_rate * avg_win - (1-win_rate) * avg_loss."""
        if self.trade_log.empty or "net_pnl" not in self.trade_log.columns:
            return 0.0
        wins = self.trade_log.loc[self.trade_log["net_pnl"] > 0, "net_pnl"]
        losses = self.trade_log.loc[self.trade_log["net_pnl"] < 0, "net_pnl"]
        n = len(self.trade_log)
        if n == 0:
            return 0.0
        wr = len(wins) / n
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
        return wr * avg_win - (1 - wr) * avg_loss

    # ------------------------------------------------------------------
    # Tail risk
    # ------------------------------------------------------------------

    def var(self, confidence: float = 0.95) -> float:
        """Historical VaR at given confidence level."""
        return float(-np.percentile(self.returns, (1 - confidence) * 100))

    def cvar(self, confidence: float = 0.95) -> float:
        """Historical CVaR (Expected Shortfall)."""
        var = self.var(confidence)
        return float(-self.returns[self.returns <= -var].mean())

    def skewness(self) -> float:
        return float(scipy_stats.skew(self.returns.dropna()))

    def kurtosis(self) -> float:
        return float(scipy_stats.kurtosis(self.returns.dropna()))

    # ------------------------------------------------------------------
    # Full summary report
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return all metrics in a flat dict."""
        sharpe, sh_lo, sh_hi = self.bootstrap.sharpe_ci(self.returns.values)
        dsr = self.bootstrap.deflated_sharpe_ratio(
            sharpe_observed=sharpe,
            returns=self.returns.values,
        )
        var95, cvar95 = self.bootstrap.cvar(self.returns.values, 0.95)

        report = {
            # Return metrics
            "total_return": self.total_return(),
            "annualized_return": self.annualized_return(),
            "annualized_vol": self.annualized_vol(),
            # Risk-adjusted
            "sharpe_ratio": self.sharpe_ratio(),
            "sharpe_ci_low": sh_lo,
            "sharpe_ci_high": sh_hi,
            "deflated_sharpe_ratio": dsr,
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "omega_ratio": self.omega_ratio(),
            # Drawdown
            **self.drawdown.summary(),
            # Tail risk
            "var_95": var95,
            "cvar_95": cvar95,
            "skewness": self.skewness(),
            "kurtosis": self.kurtosis(),
            # Trade stats
            "num_trades": len(self.trade_log) if not self.trade_log.empty else 0,
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            "avg_hold_bars": self.avg_hold_bars(),
            "expectancy": self.expectancy(),
            # Meta
            "num_bars": len(self.returns),
            "start_date": str(self.equity.index[0]) if len(self.equity) > 0 else "",
            "end_date": str(self.equity.index[-1]) if len(self.equity) > 0 else "",
        }
        return report

    def print_summary(self) -> None:
        s = self.summary()
        print("\n" + "=" * 60)
        print("LARSA BACKTEST PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Period:              {s['start_date']} to {s['end_date']}")
        print(f"Bars:                {s['num_bars']:,}")
        print("-" * 60)
        print(f"Total Return:        {s['total_return']:.2%}")
        print(f"Annualized Return:   {s['annualized_return']:.2%}")
        print(f"Annualized Vol:      {s['annualized_vol']:.2%}")
        print("-" * 60)
        print(f"Sharpe Ratio:        {s['sharpe_ratio']:.3f}  [{s['sharpe_ci_low']:.3f}, {s['sharpe_ci_high']:.3f}]")
        print(f"Deflated Sharpe:     {s['deflated_sharpe_ratio']:.3f}")
        print(f"Sortino Ratio:       {s['sortino_ratio']:.3f}")
        print(f"Calmar Ratio:        {s['calmar_ratio']:.3f}")
        print("-" * 60)
        print(f"Max Drawdown:        {s['max_drawdown']:.2%}")
        print(f"Max DD Duration:     {s['max_drawdown_duration_bars']} bars")
        print(f"VaR 95%:             {s['var_95']:.4%}")
        print(f"CVaR 95%:            {s['cvar_95']:.4%}")
        print("-" * 60)
        print(f"Trades:              {s['num_trades']}")
        print(f"Win Rate:            {s['win_rate']:.2%}")
        print(f"Profit Factor:       {s['profit_factor']:.2f}")
        print(f"Avg Hold (bars):     {s['avg_hold_bars']:.1f}")
        print("=" * 60 + "\n")

    def rolling_summary(self, window_bars: int = 780) -> pd.DataFrame:
        """Return rolling performance metrics as a DataFrame."""
        roll_sharpe = self.rolling.rolling_sharpe(window_bars)
        roll_sortino = self.rolling.rolling_sortino(window_bars)
        roll_vol = self.rolling.rolling_volatility(window_bars)
        roll_sig = self.rolling.sharpe_significance(window_bars)

        return pd.DataFrame(
            {
                "rolling_sharpe": roll_sharpe,
                "rolling_sortino": roll_sortino,
                "rolling_vol": roll_vol,
                "sharpe_pvalue": roll_sig,
            }
        )

    def attribution(self) -> Optional[pd.DataFrame]:
        """Per-signal-type attribution (if trade log has entry_signal column)."""
        if self.trade_log.empty:
            return None
        journal = TradeJournal()
        # Just use the trade log directly for attribution
        if "entry_signal" not in self.trade_log.columns:
            return None
        return self.trade_log.groupby("entry_signal").agg(
            count=("net_pnl", "count"),
            total_pnl=("net_pnl", "sum"),
            avg_pnl=("net_pnl", "mean"),
            win_rate=("net_pnl", lambda x: (x > 0).mean()),
        )
