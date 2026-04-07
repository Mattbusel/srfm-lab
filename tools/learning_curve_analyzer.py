"""
learning_curve_analyzer.py -- Tracks how LARSA v18 strategy performance evolves
over time as its online learner and RL exit module accumulate experience.

Detects performance regimes (strong/weak periods), estimates the learning
rate of the Sharpe improvement trend, and monitors whether adaptive parameter
changes correlate with improved outcomes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from post_trade_analyzer import (
    PostTradeRecord,
    load_trades_from_db,
    _sharpe,
    _win_rate,
    _avg,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PerformancePeriod:
    """A contiguous period of consistently strong or weak performance."""
    start_date: datetime
    end_date: datetime
    label: str              # "strong" | "weak" | "neutral"
    trade_count: int
    avg_pnl_pct: float
    sharpe: float
    win_rate: float
    dominant_exit_reason: str = ""
    dominant_entry_reason: str = ""

    @property
    def duration_days(self) -> float:
        return (self.end_date - self.start_date).total_seconds() / 86400.0


@dataclass
class ParameterChangeEvent:
    """Records a parameter adaptation event and subsequent performance."""
    timestamp: datetime
    parameter_name: str
    old_value: float
    new_value: float
    pnl_before_n: int       # trades used to measure pre-change performance
    pnl_after_n: int        # trades used to measure post-change performance
    sharpe_before: float = 0.0
    sharpe_after: float = 0.0
    win_rate_before: float = 0.0
    win_rate_after: float = 0.0

    @property
    def sharpe_improvement(self) -> float:
        return self.sharpe_after - self.sharpe_before

    @property
    def was_beneficial(self) -> bool:
        return self.sharpe_after > self.sharpe_before


# ---------------------------------------------------------------------------
# LearningCurveAnalyzer
# ---------------------------------------------------------------------------

class LearningCurveAnalyzer:
    """
    Tracks LARSA v18 learning trajectory over strategy lifetime.

    Analyzes:
    - Rolling 30-day Sharpe to visualize performance evolution
    - ML module convergence: does prediction accuracy improve over time?
    - RL exit improvement: does the Q-table produce better exits over time?
    - Parameter adaptation history vs subsequent performance outcomes
    - Performance regime detection (strong/weak periods)
    - Learning rate estimation via linear regression on cumulative Sharpe

    Usage::

        lca = LearningCurveAnalyzer("trades.db")
        print(lca.rolling_sharpe_series(window=30))
        print(lca.detect_performance_regimes())
        print(lca.estimate_learning_rate())
    """

    def __init__(
        self,
        db_path: str | None = None,
        trades: list[PostTradeRecord] | None = None,
        window_trades: int = 30,
        min_regime_trades: int = 15,
    ):
        if trades is not None:
            self.trades = trades
        elif db_path is not None:
            self.trades = load_trades_from_db(db_path)
        else:
            self.trades = []

        self.window_trades = window_trades
        self.min_regime_trades = min_regime_trades

        # Sort once by entry time
        self.trades = sorted(self.trades, key=lambda t: t.entry_time)
        self._param_changes: list[ParameterChangeEvent] = []

        logger.info(
            "LearningCurveAnalyzer: %d trades, window=%d",
            len(self.trades), window_trades,
        )

    # ------------------------------------------------------------------
    # Rolling Sharpe
    # ------------------------------------------------------------------

    def rolling_sharpe_series(
        self,
        window: int | None = None,
        by: str = "trades",  # "trades" | "days"
        days_window: int = 30,
    ) -> pd.DataFrame:
        """
        Compute rolling Sharpe over strategy lifetime.

        Parameters
        ----------
        window : rolling window size in trades (None uses self.window_trades)
        by : "trades" rolls over N consecutive trades; "days" rolls over calendar days
        days_window : if by="days", size of the rolling calendar window

        Returns DataFrame with columns: date, rolling_sharpe, rolling_win_rate,
        rolling_avg_pnl, n_trades.
        """
        if not self.trades:
            return pd.DataFrame()

        window = window or self.window_trades

        if by == "trades":
            rows = []
            for i in range(window, len(self.trades) + 1):
                subset = self.trades[i - window : i]
                pnls = [t.pnl_pct for t in subset]
                rows.append({
                    "date": subset[-1].exit_time,
                    "trade_index": i,
                    "rolling_sharpe": _sharpe(pnls),
                    "rolling_win_rate": _win_rate(pnls),
                    "rolling_avg_pnl_pct": _avg(pnls),
                    "n_trades": window,
                })
            return pd.DataFrame(rows)

        else:  # by days
            all_dates = [t.exit_time for t in self.trades]
            min_date = min(all_dates)
            max_date = max(all_dates)

            from datetime import timedelta
            rows = []
            cursor = min_date + timedelta(days=days_window)
            while cursor <= max_date + timedelta(days=1):
                start = cursor - timedelta(days=days_window)
                subset = [
                    t for t in self.trades
                    if start <= t.exit_time <= cursor
                ]
                if len(subset) >= 2:
                    pnls = [t.pnl_pct for t in subset]
                    rows.append({
                        "date": cursor,
                        "rolling_sharpe": _sharpe(pnls),
                        "rolling_win_rate": _win_rate(pnls),
                        "rolling_avg_pnl_pct": _avg(pnls),
                        "n_trades": len(subset),
                    })
                cursor += timedelta(days=1)
            return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # ML module convergence proxy
    # ------------------------------------------------------------------

    def ml_convergence_proxy(
        self,
        feature_attr: str = "bh_mass_at_entry",
        bin_size: int = 50,
    ) -> pd.DataFrame:
        """
        Proxy for ML module convergence: measures whether the correlation
        between a feature at entry and subsequent P&L improves over time.

        A rising correlation suggests the model is learning to use the feature
        more discriminatively.

        Parameters
        ----------
        feature_attr : trade attribute to correlate with pnl_pct
        bin_size : number of trades per rolling bin

        Returns DataFrame with bin_index, correlation, p_value, n.
        """
        if len(self.trades) < bin_size:
            return pd.DataFrame()

        rows = []
        for i in range(0, len(self.trades) - bin_size + 1, bin_size // 2):
            subset = self.trades[i : i + bin_size]
            features = [getattr(t, feature_attr) for t in subset]
            pnls = [t.pnl_pct for t in subset]
            if len(set(features)) < 3:
                continue
            try:
                r, p = scipy_stats.pearsonr(features, pnls)
            except Exception:
                r, p = 0.0, 1.0
            rows.append({
                "bin_start_trade": i,
                "bin_end_trade": i + bin_size,
                "end_date": subset[-1].entry_time,
                "feature": feature_attr,
                "correlation": round(r, 4),
                "p_value": round(p, 4),
                "abs_correlation": abs(r),
                "n": bin_size,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # RL exit improvement
    # ------------------------------------------------------------------

    def rl_exit_improvement(
        self,
        window: int | None = None,
    ) -> pd.DataFrame:
        """
        Track whether RL exit quality (exit efficiency) improves over time
        as the Q-table accumulates experience.

        Only includes trades where exit_reason == "rl_exit" and mfe > 0.

        Returns DataFrame with rolling exit_efficiency statistics.
        """
        window = window or self.window_trades
        rl_trades = [
            t for t in self.trades
            if t.exit_reason == "rl_exit" and t.mfe > 0
        ]

        if len(rl_trades) < window:
            logger.warning(
                "Only %d RL exit trades -- need %d for rolling window",
                len(rl_trades), window,
            )
            # Return all we have as one row
            if rl_trades:
                effs = [t.exit_efficiency for t in rl_trades]
                return pd.DataFrame([{
                    "trade_index": len(rl_trades),
                    "date": rl_trades[-1].exit_time,
                    "rolling_exit_efficiency": _avg(effs),
                    "rolling_pnl_pct": _avg([t.pnl_pct for t in rl_trades]),
                    "n_rl_trades": len(rl_trades),
                }])
            return pd.DataFrame()

        rows = []
        for i in range(window, len(rl_trades) + 1):
            subset = rl_trades[i - window : i]
            effs = [t.exit_efficiency for t in subset]
            pnls = [t.pnl_pct for t in subset]
            rows.append({
                "trade_index": i,
                "date": subset[-1].exit_time,
                "rolling_exit_efficiency": _avg(effs),
                "rolling_pnl_pct": _avg(pnls),
                "rolling_sharpe": _sharpe(pnls),
                "n_rl_trades": window,
            })

        df = pd.DataFrame(rows)
        # Add trend direction: is efficiency improving?
        if len(df) > 1:
            slope, _, _, _, _ = scipy_stats.linregress(
                range(len(df)), df["rolling_exit_efficiency"]
            )
            df["efficiency_trend_slope"] = slope
        return df

    # ------------------------------------------------------------------
    # Parameter adaptation tracking
    # ------------------------------------------------------------------

    def register_parameter_change(
        self,
        timestamp: datetime,
        parameter_name: str,
        old_value: float,
        new_value: float,
        lookback_n: int = 20,
        lookahead_n: int = 20,
    ) -> ParameterChangeEvent:
        """
        Register a parameter change event and measure its impact on performance.

        Automatically measures Sharpe before and after using the N trades
        immediately preceding / following the change timestamp.
        """
        before_trades = [
            t for t in self.trades if t.entry_time < timestamp
        ][-lookback_n:]
        after_trades = [
            t for t in self.trades if t.entry_time >= timestamp
        ][:lookahead_n]

        before_pnls = [t.pnl_pct for t in before_trades]
        after_pnls = [t.pnl_pct for t in after_trades]

        event = ParameterChangeEvent(
            timestamp=timestamp,
            parameter_name=parameter_name,
            old_value=old_value,
            new_value=new_value,
            pnl_before_n=len(before_trades),
            pnl_after_n=len(after_trades),
            sharpe_before=_sharpe(before_pnls),
            sharpe_after=_sharpe(after_pnls),
            win_rate_before=_win_rate(before_pnls),
            win_rate_after=_win_rate(after_pnls),
        )
        self._param_changes.append(event)
        logger.info(
            "Param change: %s %.4f -> %.4f | Sharpe %.2f -> %.2f",
            parameter_name, old_value, new_value,
            event.sharpe_before, event.sharpe_after,
        )
        return event

    def parameter_adaptation_history(self) -> pd.DataFrame:
        """Return registered parameter changes as DataFrame."""
        if not self._param_changes:
            return pd.DataFrame()
        rows = [
            {
                "timestamp": e.timestamp,
                "parameter": e.parameter_name,
                "old_value": e.old_value,
                "new_value": e.new_value,
                "sharpe_before": e.sharpe_before,
                "sharpe_after": e.sharpe_after,
                "sharpe_improvement": e.sharpe_improvement,
                "was_beneficial": e.was_beneficial,
                "win_rate_before": e.win_rate_before,
                "win_rate_after": e.win_rate_after,
            }
            for e in self._param_changes
        ]
        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Performance regime detection
    # ------------------------------------------------------------------

    def detect_performance_regimes(
        self,
        strong_sharpe_threshold: float = 0.5,
        weak_sharpe_threshold: float = -0.3,
        window: int | None = None,
    ) -> list[PerformancePeriod]:
        """
        Detect contiguous periods of strong or weak performance by applying
        a rolling window and labeling each window as strong / weak / neutral.

        Merges consecutive windows with the same label into a single period.

        Parameters
        ----------
        strong_sharpe_threshold : rolling Sharpe above this = "strong"
        weak_sharpe_threshold : rolling Sharpe below this = "weak"
        window : rolling window size in trades

        Returns list of PerformancePeriod objects sorted by start_date.
        """
        rolling = self.rolling_sharpe_series(window=window)
        if rolling.empty:
            return []

        def label(sharpe: float) -> str:
            if sharpe >= strong_sharpe_threshold:
                return "strong"
            if sharpe <= weak_sharpe_threshold:
                return "weak"
            return "neutral"

        rolling["period_label"] = rolling["rolling_sharpe"].map(label)

        # Merge consecutive same-label windows into periods
        periods: list[PerformancePeriod] = []
        current_label = None
        current_start: datetime | None = None
        current_rows: list[dict] = []

        for _, row in rolling.iterrows():
            lbl = row["period_label"]
            if lbl != current_label:
                if current_label is not None and current_rows:
                    periods.append(
                        self._rows_to_period(current_label, current_rows)
                    )
                current_label = lbl
                current_start = row["date"]
                current_rows = [row]
            else:
                current_rows.append(row)

        if current_label is not None and current_rows:
            periods.append(self._rows_to_period(current_label, current_rows))

        # Filter out short periods
        periods = [p for p in periods if p.trade_count >= self.min_regime_trades]
        logger.info("Detected %d performance regimes", len(periods))
        return sorted(periods, key=lambda p: p.start_date)

    def _rows_to_period(
        self,
        label: str,
        rows: list,
    ) -> PerformancePeriod:
        """Convert a list of rolling-window rows into a PerformancePeriod."""
        dates = [r["date"] for r in rows]
        sharpes = [r["rolling_sharpe"] for r in rows]
        win_rates = [r["rolling_win_rate"] for r in rows]
        pnls = [r["rolling_avg_pnl_pct"] for r in rows]
        n_trades = int(rows[0]["n_trades"]) if rows else 0

        # Find dominant entry/exit reasons in this period
        period_trades = [
            t for t in self.trades
            if min(dates) <= t.exit_time <= max(dates)
        ]
        entry_reasons = [t.entry_reason for t in period_trades]
        exit_reasons = [t.exit_reason for t in period_trades]

        def mode(lst: list[str]) -> str:
            if not lst:
                return ""
            from collections import Counter
            return Counter(lst).most_common(1)[0][0]

        return PerformancePeriod(
            start_date=min(dates),
            end_date=max(dates),
            label=label,
            trade_count=n_trades,
            avg_pnl_pct=_avg(pnls),
            sharpe=_avg(sharpes),
            win_rate=_avg(win_rates),
            dominant_entry_reason=mode(entry_reasons),
            dominant_exit_reason=mode(exit_reasons),
        )

    # ------------------------------------------------------------------
    # Learning rate estimation
    # ------------------------------------------------------------------

    def estimate_learning_rate(
        self,
        window: int | None = None,
        min_points: int = 5,
    ) -> float:
        """
        Estimate the slope of Sharpe improvement over the strategy's lifetime.

        Uses linear regression on the rolling Sharpe series.

        Returns the slope in Sharpe units per 100 trades (positive = improving).
        """
        rolling = self.rolling_sharpe_series(window=window)
        if len(rolling) < min_points:
            return 0.0

        x = np.arange(len(rolling), dtype=float)
        y = rolling["rolling_sharpe"].values

        # Filter out extreme outliers (> 3 std from mean)
        mean_y, std_y = np.mean(y), np.std(y)
        mask = np.abs(y - mean_y) <= 3 * std_y
        if mask.sum() < min_points:
            return 0.0

        slope, intercept, r, p, se = scipy_stats.linregress(x[mask], y[mask])
        learning_rate_per_100 = slope * 100.0

        logger.info(
            "Learning rate: %.4f Sharpe/100 trades (r=%.3f, p=%.3f)",
            learning_rate_per_100, r, p,
        )
        return float(learning_rate_per_100)

    # ------------------------------------------------------------------
    # Comprehensive learning summary
    # ------------------------------------------------------------------

    def learning_summary(self) -> dict:
        """
        Return a comprehensive summary of learning curve metrics.
        """
        rolling = self.rolling_sharpe_series()
        learning_rate = self.estimate_learning_rate()
        regimes = self.detect_performance_regimes()
        rl_improvement = self.rl_exit_improvement()

        early_sharpe: float = 0.0
        late_sharpe: float = 0.0
        if len(rolling) >= 4:
            n = len(rolling)
            early_sharpe = float(rolling["rolling_sharpe"].iloc[:n // 4].mean())
            late_sharpe = float(rolling["rolling_sharpe"].iloc[3 * n // 4:].mean())

        rl_early_eff: float = 0.0
        rl_late_eff: float = 0.0
        if len(rl_improvement) >= 4:
            n = len(rl_improvement)
            rl_early_eff = float(
                rl_improvement["rolling_exit_efficiency"].iloc[:n // 4].mean()
            )
            rl_late_eff = float(
                rl_improvement["rolling_exit_efficiency"].iloc[3 * n // 4:].mean()
            )

        strong_periods = [r for r in regimes if r.label == "strong"]
        weak_periods = [r for r in regimes if r.label == "weak"]

        return {
            "total_trades": len(self.trades),
            "learning_rate_sharpe_per_100_trades": learning_rate,
            "learning_direction": (
                "improving" if learning_rate > 0.01
                else "declining" if learning_rate < -0.01
                else "flat"
            ),
            "early_sharpe_q1": early_sharpe,
            "late_sharpe_q4": late_sharpe,
            "sharpe_improvement_early_to_late": late_sharpe - early_sharpe,
            "rl_exit_efficiency_early": rl_early_eff,
            "rl_exit_efficiency_late": rl_late_eff,
            "rl_improvement": rl_late_eff - rl_early_eff,
            "strong_periods": len(strong_periods),
            "weak_periods": len(weak_periods),
            "pct_time_in_strong": (
                sum(p.trade_count for p in strong_periods) / len(self.trades) * 100.0
                if self.trades else 0.0
            ),
            "pct_time_in_weak": (
                sum(p.trade_count for p in weak_periods) / len(self.trades) * 100.0
                if self.trades else 0.0
            ),
            "parameter_changes_logged": len(self._param_changes),
            "beneficial_param_changes": sum(
                1 for e in self._param_changes if e.was_beneficial
            ),
        }

    # ------------------------------------------------------------------
    # Cumulative P&L curve
    # ------------------------------------------------------------------

    def cumulative_pnl_curve(self) -> pd.DataFrame:
        """
        Return DataFrame of cumulative P&L over trade history.
        Includes columns for date, trade_index, cumulative_pnl_pct,
        drawdown_pct, and rolling_sharpe.
        """
        if not self.trades:
            return pd.DataFrame()

        rows = []
        cum_pnl = 0.0
        peak_pnl = 0.0
        pnl_history: list[float] = []

        for i, t in enumerate(self.trades):
            cum_pnl += t.pnl_pct
            peak_pnl = max(peak_pnl, cum_pnl)
            drawdown = cum_pnl - peak_pnl
            pnl_history.append(t.pnl_pct)

            w = min(self.window_trades, len(pnl_history))
            rows.append({
                "trade_index": i + 1,
                "date": t.exit_time,
                "symbol": t.symbol,
                "pnl_pct": t.pnl_pct,
                "cumulative_pnl_pct": cum_pnl,
                "peak_pnl_pct": peak_pnl,
                "drawdown_pct": drawdown,
                "rolling_sharpe": _sharpe(pnl_history[-w:]),
            })

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Learning curve analyzer for LARSA v18"
    )
    parser.add_argument("db", help="Path to trades SQLite database")
    parser.add_argument(
        "--window", type=int, default=30, help="Rolling window size (trades)"
    )
    parser.add_argument(
        "--report",
        default="summary",
        choices=["summary", "rolling", "regimes", "rl", "curve"],
    )
    args = parser.parse_args()

    lca = LearningCurveAnalyzer(db_path=args.db, window_trades=args.window)

    if args.report == "summary":
        print(json.dumps(lca.learning_summary(), indent=2, default=str))
    elif args.report == "rolling":
        print(lca.rolling_sharpe_series().to_string())
    elif args.report == "regimes":
        for p in lca.detect_performance_regimes():
            print(
                f"{p.label:8s} | {p.start_date.date()} - {p.end_date.date()} "
                f"| Sharpe={p.sharpe:.2f} | n={p.trade_count}"
            )
    elif args.report == "rl":
        print(lca.rl_exit_improvement().to_string())
    elif args.report == "curve":
        print(lca.cumulative_pnl_curve().to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
