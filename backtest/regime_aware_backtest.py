"""
regime_aware_backtest.py -- Regime-conditional backtesting for SRFM.

Extends base BacktestEngine with regime detection, conditional stats,
regime transition analysis, and a markdown report generator.
"""

from __future__ import annotations

import logging
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BARS_PER_YEAR = 252 * 26  # 15-min bars


# ---------------------------------------------------------------------------
# Performance stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class PerformanceStats:
    """Performance statistics for a single regime or overall period."""
    n_bars: int
    total_return: float
    annualized_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    hit_rate: float          # fraction of bars with positive return
    avg_return: float        # mean bar return
    vol: float               # bar-level std dev of returns
    regime: str = "OVERALL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "n_bars": self.n_bars,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "avg_return": self.avg_return,
            "vol": self.vol,
        }


def _compute_stats(returns: pd.Series, regime: str = "OVERALL") -> PerformanceStats:
    """Compute PerformanceStats from a returns series."""
    r = returns.dropna()
    n = len(r)
    if n < 2:
        return PerformanceStats(
            n_bars=n,
            total_return=0.0,
            annualized_return=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            hit_rate=0.0,
            avg_return=0.0,
            vol=0.0,
            regime=regime,
        )

    total_ret = float((1 + r).prod() - 1.0)
    ann_ret = float((1 + total_ret) ** (BARS_PER_YEAR / n) - 1.0) if n > 0 else 0.0
    mu = float(r.mean())
    sigma = float(r.std())
    sharpe = float(mu / sigma * np.sqrt(BARS_PER_YEAR)) if sigma > 1e-10 else 0.0

    downside = r[r < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 1e-10
    sortino = float(mu / downside_std * np.sqrt(BARS_PER_YEAR)) if downside_std > 1e-10 else 0.0

    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    hit_rate = float((r > 0).mean())

    return PerformanceStats(
        n_bars=n,
        total_return=total_ret,
        annualized_return=ann_ret,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        hit_rate=hit_rate,
        avg_return=mu,
        vol=sigma,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# Conditional performance analytics
# ---------------------------------------------------------------------------

class ConditionalPerformance:
    """
    Compute performance statistics conditioned on regime labels.

    Regime series and equity/returns series must share the same index.
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        regime_series: pd.Series,
    ):
        self.equity = equity_curve.copy()
        self.returns = equity_curve.pct_change().dropna()
        self.regime_series = regime_series.copy()

    def by_regime(self) -> Dict[str, PerformanceStats]:
        """
        Return dict mapping regime label -> PerformanceStats.

        Uses the returns series aligned with regime labels.
        """
        aligned = self.regime_series.reindex(self.returns.index).ffill().bfill()
        regimes = aligned.dropna().unique()
        result: Dict[str, PerformanceStats] = {}
        for reg in regimes:
            mask = aligned == reg
            r_subset = self.returns[mask]
            result[str(reg)] = _compute_stats(r_subset, regime=str(reg))
        return result

    def transition_impact(
        self,
        window: int = 5,
    ) -> pd.DataFrame:
        """
        Compute average return in a window around each regime transition.

        window: number of bars before and after each transition to include.

        Returns DataFrame with columns:
          [transition, bars_before_avg, bars_after_avg, pre_window_sharpe, post_window_sharpe]
        """
        aligned = self.regime_series.reindex(self.returns.index).ffill().bfill()
        if aligned.empty:
            return pd.DataFrame()

        regime_arr = aligned.values
        returns_arr = self.returns.values
        index_arr = self.returns.index

        # Find transition points
        transitions: List[Dict[str, Any]] = []
        for i in range(1, len(regime_arr)):
            if regime_arr[i] != regime_arr[i - 1]:
                from_reg = str(regime_arr[i - 1])
                to_reg = str(regime_arr[i])

                pre_start = max(0, i - window)
                post_end = min(len(returns_arr), i + window)

                pre_ret = returns_arr[pre_start:i]
                post_ret = returns_arr[i:post_end]

                pre_avg = float(np.mean(pre_ret)) if len(pre_ret) > 0 else 0.0
                post_avg = float(np.mean(post_ret)) if len(post_ret) > 0 else 0.0

                pre_std = float(np.std(pre_ret)) if len(pre_ret) > 1 else 1e-10
                post_std = float(np.std(post_ret)) if len(post_ret) > 1 else 1e-10

                pre_sharpe = (
                    pre_avg / pre_std * np.sqrt(BARS_PER_YEAR)
                    if pre_std > 1e-10 else 0.0
                )
                post_sharpe = (
                    post_avg / post_std * np.sqrt(BARS_PER_YEAR)
                    if post_std > 1e-10 else 0.0
                )

                transitions.append({
                    "bar": i,
                    "timestamp": index_arr[i] if i < len(index_arr) else None,
                    "from_regime": from_reg,
                    "to_regime": to_reg,
                    "transition": f"{from_reg} -> {to_reg}",
                    "pre_window_avg": pre_avg,
                    "post_window_avg": post_avg,
                    "pre_window_sharpe": pre_sharpe,
                    "post_window_sharpe": post_sharpe,
                })

        if not transitions:
            return pd.DataFrame()

        return pd.DataFrame(transitions)

    def regime_persistence(self) -> Dict[str, float]:
        """
        Compute average run duration (in bars) for each regime.

        Returns dict: regime -> avg_consecutive_bars
        """
        aligned = self.regime_series.reindex(self.returns.index).ffill().bfill()
        if aligned.empty:
            return {}

        durations: Dict[str, List[int]] = defaultdict(list)
        current = None
        run_len = 0

        for val in aligned:
            r = str(val)
            if r == current:
                run_len += 1
            else:
                if current is not None:
                    durations[current].append(run_len)
                current = r
                run_len = 1

        if current is not None and run_len > 0:
            durations[current].append(run_len)

        return {
            reg: float(np.mean(runs))
            for reg, runs in durations.items()
        }

    def regime_distribution(self) -> Dict[str, float]:
        """Return fraction of time spent in each regime."""
        aligned = self.regime_series.reindex(self.returns.index).ffill().bfill().dropna()
        if aligned.empty:
            return {}
        counts = aligned.value_counts(normalize=True)
        return {str(k): float(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Regime-aware backtest
# ---------------------------------------------------------------------------

class RegimeAwareBacktest:
    """
    Backtest that logs regime state at each bar and scales position size
    based on the current regime classification.

    Wraps a base bar-iteration loop rather than inheriting from engine.py
    to avoid tight coupling. Accepts a bars DataFrame and a signal function.

    Usage:
      bt = RegimeAwareBacktest(initial_capital=100_000)
      bt.set_regime_classifier(my_classifier)
      bt.set_regime_sizing({"TRENDING": 1.0, "RANGING": 0.5, "HIGH_VOL": 0.3})
      result = bt.run(bars_df, signal_fn, start, end)
    """

    DEFAULT_REGIME_SIZES: Dict[str, float] = {
        "TRENDING": 1.0,
        "RANGING": 0.5,
        "HIGH_VOL": 0.3,
        "UNKNOWN": 0.5,
    }

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_bps: float = 5.0,
        slippage_bps: float = 3.0,
        max_position_frac: float = 0.25,
        regime_lookback: int = 20,   # bars fed to regime classifier
    ):
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.max_position_frac = max_position_frac
        self.regime_lookback = regime_lookback

        self._regime_fn: Optional[Callable] = None
        self._regime_sizes: Dict[str, float] = dict(self.DEFAULT_REGIME_SIZES)

    def set_regime_classifier(self, fn: Callable[[List[dict]], str]) -> None:
        """
        Set the regime classifier function.

        fn(bar_history: List[dict]) -> str
        bar_history is a list of the last regime_lookback bar dicts.
        Should return a regime label string.
        """
        self._regime_fn = fn

    def set_regime_sizing(self, regime_sizes: Dict[str, float]) -> None:
        """
        Set per-regime position size multipliers.

        Example: {"TRENDING": 1.0, "RANGING": 0.5, "HIGH_VOL": 0.3}
        """
        self._regime_sizes = {**self.DEFAULT_REGIME_SIZES, **regime_sizes}
        logger.info("Regime sizing updated: %s", self._regime_sizes)

    def run(
        self,
        bars_df: pd.DataFrame,
        signal_fn: Callable[[dict], float],
        start: str,
        end: str,
    ) -> Dict[str, Any]:
        """
        Execute regime-aware backtest.

        Returns dict with:
          equity_curve, returns, regime_series, positions, trades, stats
        """
        df = bars_df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        df = df.loc[start_ts:end_ts]

        if df.empty:
            raise ValueError(f"No bars in range [{start}, {end}].")

        cash = self.initial_capital
        position = 0.0
        prev_price = 0.0

        equity_log: Dict[pd.Timestamp, float] = {}
        regime_log: Dict[pd.Timestamp, str] = {}
        position_log: Dict[pd.Timestamp, float] = {}
        trade_log: List[dict] = []
        bar_history: List[dict] = []

        for ts, row in df.iterrows():
            bar = row.to_dict()
            bar["timestamp"] = ts
            price = float(bar.get("close", 0))
            if price <= 0:
                continue

            bar_history.append(bar)
            if len(bar_history) > self.regime_lookback * 2:
                bar_history = bar_history[-self.regime_lookback * 2:]

            # Classify regime
            regime = "UNKNOWN"
            if self._regime_fn is not None:
                history_slice = bar_history[-self.regime_lookback:]
                try:
                    regime = str(self._regime_fn(history_slice))
                except Exception as exc:
                    logger.warning("Regime classifier error at %s: %s", ts, exc)
                    regime = "UNKNOWN"

            regime_log[ts] = regime

            # Get regime size multiplier
            size_mult = self._regime_sizes.get(regime, self._regime_sizes.get("UNKNOWN", 0.5))

            # Generate signal
            try:
                signal = float(signal_fn(bar))
            except Exception as exc:
                logger.warning("signal_fn error at %s: %s", ts, exc)
                signal = 0.0

            signal = max(-1.0, min(1.0, signal))

            # Compute desired position
            notional = self.initial_capital * self.max_position_frac * abs(signal) * size_mult
            desired_units = (notional / price) * (1.0 if signal > 0 else -1.0) if abs(signal) > 1e-8 else 0.0

            # Execute
            delta = desired_units - position
            if abs(delta) > 1e-8:
                direction = 1.0 if delta > 0 else -1.0
                slip_frac = self.slippage_bps / 10_000.0
                fill_price = price * (1.0 + direction * slip_frac)
                comm = abs(delta) * fill_price * self.commission_bps / 10_000.0
                cash -= delta * fill_price + comm
                position = desired_units
                trade_log.append({
                    "timestamp": ts,
                    "regime": regime,
                    "price": price,
                    "fill_price": fill_price,
                    "delta": delta,
                    "position": position,
                    "size_mult": size_mult,
                })

            equity_log[ts] = cash + position * price
            position_log[ts] = position
            prev_price = price

        equity_series = pd.Series(equity_log, name="equity").sort_index()
        regime_series = pd.Series(regime_log, name="regime").sort_index()
        position_series = pd.Series(position_log, name="position").sort_index()
        returns_series = equity_series.pct_change().dropna()

        # Compute conditional performance
        cperf = ConditionalPerformance(equity_series, regime_series)
        by_regime = cperf.by_regime()
        transitions = cperf.transition_impact()
        persistence = cperf.regime_persistence()
        distribution = cperf.regime_distribution()
        overall_stats = _compute_stats(returns_series, "OVERALL")

        return {
            "equity_curve": equity_series,
            "returns": returns_series,
            "regime_series": regime_series,
            "positions": position_series,
            "trades": pd.DataFrame(trade_log),
            "overall_stats": overall_stats,
            "by_regime": by_regime,
            "transitions": transitions,
            "persistence": persistence,
            "distribution": distribution,
        }


# ---------------------------------------------------------------------------
# Regime backtest report generator
# ---------------------------------------------------------------------------

class RegimeBacktestReport:
    """
    Generate a markdown report summarizing regime-aware backtest results.

    Sections:
      1. Overall performance
      2. Per-regime statistics
      3. Regime transition analysis
      4. Regime distribution
    """

    def generate(self, result: Dict[str, Any]) -> str:
        """
        Build and return the full markdown report as a string.

        result: dict as returned by RegimeAwareBacktest.run()
        """
        sections: List[str] = []

        sections.append("# Regime-Aware Backtest Report\n")

        # Section 1: Overall stats
        sections.append(self._section_overall(result))

        # Section 2: Per-regime stats
        sections.append(self._section_per_regime(result))

        # Section 3: Transition analysis
        sections.append(self._section_transitions(result))

        # Section 4: Regime distribution
        sections.append(self._section_distribution(result))

        return "\n".join(sections)

    def _section_overall(self, result: Dict[str, Any]) -> str:
        stats: PerformanceStats = result.get("overall_stats", None)
        if stats is None:
            return "## Overall Stats\n\nNo data.\n"

        eq = result.get("equity_curve", pd.Series(dtype=float))
        start_val = float(eq.iloc[0]) if not eq.empty else 0.0
        end_val = float(eq.iloc[-1]) if not eq.empty else 0.0

        lines = [
            "## Overall Performance\n",
            f"- **Bars**: {stats.n_bars:,}",
            f"- **Total Return**: {stats.total_return*100:.2f}%",
            f"- **Annualized Return**: {stats.annualized_return*100:.2f}%",
            f"- **Sharpe Ratio**: {stats.sharpe:.3f}",
            f"- **Sortino Ratio**: {stats.sortino:.3f}",
            f"- **Max Drawdown**: {stats.max_drawdown*100:.2f}%",
            f"- **Hit Rate**: {stats.hit_rate*100:.1f}%",
            f"- **Start NAV**: ${start_val:,.2f}",
            f"- **End NAV**: ${end_val:,.2f}",
            "",
        ]
        return "\n".join(lines)

    def _section_per_regime(self, result: Dict[str, Any]) -> str:
        by_regime: Dict[str, PerformanceStats] = result.get("by_regime", {})
        if not by_regime:
            return "## Per-Regime Stats\n\nNo regime data.\n"

        lines = ["## Per-Regime Statistics\n"]
        header = "| Regime | Bars | Total Ret | Ann Ret | Sharpe | Sortino | MaxDD | Hit Rate |"
        sep =    "|--------|------|-----------|---------|--------|---------|-------|----------|"
        lines.append(header)
        lines.append(sep)

        for reg, s in sorted(by_regime.items()):
            row = (
                f"| {reg} "
                f"| {s.n_bars:,} "
                f"| {s.total_return*100:.2f}% "
                f"| {s.annualized_return*100:.2f}% "
                f"| {s.sharpe:.3f} "
                f"| {s.sortino:.3f} "
                f"| {s.max_drawdown*100:.2f}% "
                f"| {s.hit_rate*100:.1f}% |"
            )
            lines.append(row)

        lines.append("")
        return "\n".join(lines)

    def _section_transitions(self, result: Dict[str, Any]) -> str:
        df: pd.DataFrame = result.get("transitions", pd.DataFrame())
        if df is None or df.empty:
            return "## Regime Transitions\n\nNo transitions detected.\n"

        lines = ["## Regime Transition Analysis\n"]
        lines.append(f"Total transitions detected: **{len(df)}**\n")

        # Aggregate by transition type
        if "transition" in df.columns:
            agg = df.groupby("transition").agg(
                count=("bar", "count"),
                avg_pre=("pre_window_avg", "mean"),
                avg_post=("post_window_avg", "mean"),
                avg_pre_sharpe=("pre_window_sharpe", "mean"),
                avg_post_sharpe=("post_window_sharpe", "mean"),
            ).reset_index()

            header = "| Transition | Count | Pre Avg Ret | Post Avg Ret | Pre Sharpe | Post Sharpe |"
            sep =    "|------------|-------|-------------|--------------|------------|-------------|"
            lines.append(header)
            lines.append(sep)
            for _, row in agg.iterrows():
                r = (
                    f"| {row['transition']} "
                    f"| {int(row['count'])} "
                    f"| {row['avg_pre']*100:.4f}% "
                    f"| {row['avg_post']*100:.4f}% "
                    f"| {row['avg_pre_sharpe']:.3f} "
                    f"| {row['avg_post_sharpe']:.3f} |"
                )
                lines.append(r)

        lines.append("")
        return "\n".join(lines)

    def _section_distribution(self, result: Dict[str, Any]) -> str:
        distribution: Dict[str, float] = result.get("distribution", {})
        persistence: Dict[str, float] = result.get("persistence", {})

        if not distribution:
            return "## Regime Distribution\n\nNo data.\n"

        lines = [
            "## Regime Distribution\n",
            "| Regime | % Time | Avg Duration (bars) |",
            "|--------|--------|---------------------|",
        ]

        for reg, pct in sorted(distribution.items(), key=lambda x: -x[1]):
            avg_dur = persistence.get(reg, 0.0)
            lines.append(
                f"| {reg} | {pct*100:.1f}% | {avg_dur:.1f} |"
            )

        # ASCII pie chart using block characters
        lines.append("\n### Distribution Chart\n")
        lines.append("```")
        total_chars = 40
        for reg, pct in sorted(distribution.items(), key=lambda x: -x[1]):
            bar_len = int(round(pct * total_chars))
            bar_str = "#" * bar_len
            lines.append(f"{reg:<20} |{bar_str:<40}| {pct*100:.1f}%")
        lines.append("```\n")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in regime classifiers
# ---------------------------------------------------------------------------

def trending_classifier(bar_history: List[dict], adx_threshold: float = 25.0) -> str:
    """
    Simple regime classifier using price momentum and volatility.

    Regimes: TRENDING, RANGING, HIGH_VOL
    """
    if len(bar_history) < 5:
        return "UNKNOWN"

    closes = np.array([float(b.get("close", 0)) for b in bar_history])
    closes = closes[closes > 0]

    if len(closes) < 5:
        return "UNKNOWN"

    # Volatility regime check
    rets = np.diff(closes) / closes[:-1]
    vol = float(np.std(rets))
    annualized_vol = vol * np.sqrt(BARS_PER_YEAR)
    if annualized_vol > 1.0:   # >100% annualized vol = high vol
        return "HIGH_VOL"

    # Trend check: slope of linear regression
    x = np.arange(len(closes), dtype=float)
    x_norm = x / max(len(closes) - 1, 1)
    slope = float(np.polyfit(x_norm, closes, 1)[0])
    slope_pct = slope / closes[0] if closes[0] > 0 else 0.0

    # R-squared
    predicted = np.polyval(np.polyfit(x_norm, closes, 1), x_norm)
    ss_res = float(np.sum((closes - predicted) ** 2))
    ss_tot = float(np.sum((closes - closes.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    if r2 > 0.6 and abs(slope_pct) > 0.005:
        return "TRENDING"
    return "RANGING"


def volatility_regime_classifier(
    bar_history: List[dict],
    low_vol_threshold: float = 0.30,
    high_vol_threshold: float = 0.80,
) -> str:
    """
    Three-state vol regime: LOW_VOL, MED_VOL, HIGH_VOL.
    """
    if len(bar_history) < 5:
        return "MED_VOL"

    closes = np.array([float(b.get("close", 0)) for b in bar_history])
    closes = closes[closes > 0]
    if len(closes) < 2:
        return "MED_VOL"

    rets = np.diff(closes) / closes[:-1]
    ann_vol = float(np.std(rets)) * np.sqrt(BARS_PER_YEAR)

    if ann_vol < low_vol_threshold:
        return "LOW_VOL"
    elif ann_vol > high_vol_threshold:
        return "HIGH_VOL"
    return "MED_VOL"
