"""
regime_performance_analyzer.py -- Analyzes LARSA v18 strategy performance
conditioned on market regime states (BH mass, Hurst, GARCH vol, NAV omega).

Provides joint regime analysis, best/worst regime identification, and
2-D heatmap DataFrames for interactive inspection.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from post_trade_analyzer import (
    PostTradeRecord,
    load_trades_from_db,
    _sharpe,
    _win_rate,
    _avg,
    _bucket_label,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants -- regime bucket definitions
# ---------------------------------------------------------------------------

BH_THRESHOLDS = [0.2, 0.4, 0.6, 0.8, 1.0]
BH_LABELS = ["bh_q1", "bh_q2", "bh_q3", "bh_q4", "bh_q5"]

HURST_THRESHOLDS = [0.4, 0.55, 1.0]
HURST_LABELS = ["mean_reverting", "neutral", "trending"]

GARCH_THRESHOLDS = [0.015, 0.03, 1.0]
GARCH_LABELS = ["low_vol", "med_vol", "high_vol"]

NAV_THRESHOLDS = [0.25, 0.5, 0.75, 1.0]
NAV_LABELS = ["nav_q1", "nav_q2", "nav_q3", "nav_q4"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats_for_group(records: list[PostTradeRecord]) -> dict:
    if not records:
        return {
            "n": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0,
            "total_pnl_pct": 0.0, "sharpe": 0.0,
            "avg_hold_bars": 0.0, "avg_mfe": 0.0, "avg_mae": 0.0,
        }
    pnls = [r.pnl_pct for r in records]
    return {
        "n": len(records),
        "win_rate": _win_rate(pnls),
        "avg_pnl_pct": _avg(pnls),
        "total_pnl_pct": float(sum(pnls)),
        "sharpe": _sharpe(pnls),
        "avg_hold_bars": _avg([r.hold_bars for r in records]),
        "avg_mfe": _avg([r.mfe for r in records]),
        "avg_mae": _avg([abs(r.mae) for r in records]),
    }


def _stats_df(groups: dict[str, list[PostTradeRecord]]) -> pd.DataFrame:
    rows = []
    for label, records in groups.items():
        row = _stats_for_group(records)
        row["regime"] = label
        rows.append(row)
    df = pd.DataFrame(rows).set_index("regime")
    return df.sort_values("sharpe", ascending=False)


# ---------------------------------------------------------------------------
# RegimePerformanceAnalyzer
# ---------------------------------------------------------------------------

class RegimePerformanceAnalyzer:
    """
    Analyzes LARSA v18 P&L broken down by single and joint regime states.

    Usage::

        rpa = RegimePerformanceAnalyzer("trades.db")
        print(rpa.performance_by_hurst_regime())
        print(rpa.joint_regime_analysis())
        print(rpa.best_regime_combination())
    """

    def __init__(
        self,
        db_path: str | None = None,
        trades: list[PostTradeRecord] | None = None,
    ):
        if trades is not None:
            self.trades = trades
        elif db_path is not None:
            self.trades = load_trades_from_db(db_path)
        else:
            self.trades = []
        logger.info("RegimePerformanceAnalyzer: %d trades loaded", len(self.trades))

    # ------------------------------------------------------------------
    # Single-regime breakdowns
    # ------------------------------------------------------------------

    def performance_by_bh_mass_level(self) -> pd.DataFrame:
        """P&L stats for BH mass quintiles."""
        groups: dict[str, list[PostTradeRecord]] = {lb: [] for lb in BH_LABELS}
        for t in self.trades:
            lb = _bucket_label(t.bh_mass_at_entry, BH_THRESHOLDS, BH_LABELS)
            groups[lb].append(t)
        df = _stats_df(groups)
        df.index.name = "bh_mass_level"
        return df

    def performance_by_hurst_regime(self) -> pd.DataFrame:
        """P&L stats for Hurst trending / neutral / mean-reverting regimes."""
        groups: dict[str, list[PostTradeRecord]] = {lb: [] for lb in HURST_LABELS}
        for t in self.trades:
            lb = _bucket_label(t.hurst_at_entry, HURST_THRESHOLDS, HURST_LABELS)
            groups[lb].append(t)
        df = _stats_df(groups)
        df.index.name = "hurst_regime"
        return df

    def performance_by_vol_regime(self) -> pd.DataFrame:
        """P&L stats for GARCH volatility regimes: low / med / high."""
        groups: dict[str, list[PostTradeRecord]] = {lb: [] for lb in GARCH_LABELS}
        for t in self.trades:
            lb = _bucket_label(t.garch_vol_at_entry, GARCH_THRESHOLDS, GARCH_LABELS)
            groups[lb].append(t)
        df = _stats_df(groups)
        df.index.name = "vol_regime"
        return df

    def performance_by_nav_omega_level(self) -> pd.DataFrame:
        """P&L stats for NAV omega quartiles."""
        groups: dict[str, list[PostTradeRecord]] = {lb: [] for lb in NAV_LABELS}
        for t in self.trades:
            lb = _bucket_label(t.nav_omega_at_entry, NAV_THRESHOLDS, NAV_LABELS)
            groups[lb].append(t)
        df = _stats_df(groups)
        df.index.name = "nav_omega_level"
        return df

    # ------------------------------------------------------------------
    # Joint regime analysis (2-D)
    # ------------------------------------------------------------------

    def joint_regime_analysis(
        self,
        row_regime: str = "bh_mass",
        col_regime: str = "hurst",
        metric: str = "avg_pnl_pct",
    ) -> pd.DataFrame:
        """
        Return 2-D heatmap DataFrame of ``metric`` for two regime dimensions.

        Parameters
        ----------
        row_regime : "bh_mass" | "hurst" | "garch_vol" | "nav_omega"
        col_regime : same choices
        metric : column to display -- "avg_pnl_pct" | "sharpe" | "win_rate" | "n"
        """
        regime_cfg = {
            "bh_mass": (BH_THRESHOLDS, BH_LABELS, "bh_mass_at_entry"),
            "hurst": (HURST_THRESHOLDS, HURST_LABELS, "hurst_at_entry"),
            "garch_vol": (GARCH_THRESHOLDS, GARCH_LABELS, "garch_vol_at_entry"),
            "nav_omega": (NAV_THRESHOLDS, NAV_LABELS, "nav_omega_at_entry"),
        }

        if row_regime not in regime_cfg or col_regime not in regime_cfg:
            raise ValueError(
                f"row_regime and col_regime must be one of: {list(regime_cfg)}"
            )

        row_thresh, row_labels, row_attr = regime_cfg[row_regime]
        col_thresh, col_labels, col_attr = regime_cfg[col_regime]

        # Build 2-D groups
        cells: dict[tuple[str, str], list[PostTradeRecord]] = {}
        for rl in row_labels:
            for cl in col_labels:
                cells[(rl, cl)] = []

        for t in self.trades:
            rv = getattr(t, row_attr)
            cv = getattr(t, col_attr)
            rl = _bucket_label(rv, row_thresh, row_labels)
            cl = _bucket_label(cv, col_thresh, col_labels)
            cells[(rl, cl)].append(t)

        # Compute metric for each cell
        pivot_data: dict[str, dict[str, float]] = {rl: {} for rl in row_labels}
        for (rl, cl), records in cells.items():
            stats = _stats_for_group(records)
            pivot_data[rl][cl] = stats.get(metric, 0.0)

        df = pd.DataFrame(pivot_data).T
        df = df[col_labels]  # enforce column order
        df.index.name = row_regime
        df.columns.name = col_regime
        return df

    # ------------------------------------------------------------------
    # Best / worst regime combinations
    # ------------------------------------------------------------------

    def _all_triple_combinations(self) -> list[dict]:
        """
        Enumerate all (bh_mass x hurst x garch_vol) combinations and
        return stats sorted by Sharpe.
        """
        combos: dict[tuple, list[PostTradeRecord]] = {}
        for t in self.trades:
            bh = _bucket_label(t.bh_mass_at_entry, BH_THRESHOLDS, BH_LABELS)
            hu = _bucket_label(t.hurst_at_entry, HURST_THRESHOLDS, HURST_LABELS)
            gv = _bucket_label(t.garch_vol_at_entry, GARCH_THRESHOLDS, GARCH_LABELS)
            key = (bh, hu, gv)
            combos.setdefault(key, []).append(t)

        rows = []
        for (bh, hu, gv), records in combos.items():
            stats = _stats_for_group(records)
            rows.append({
                "bh_mass_level": bh,
                "hurst_regime": hu,
                "vol_regime": gv,
                **stats,
            })

        rows.sort(key=lambda r: r["sharpe"], reverse=True)
        return rows

    def best_regime_combination(self, min_n: int = 5) -> dict:
        """
        Return the (bh_mass_level, hurst_regime, vol_regime) triple with
        the highest Sharpe ratio, requiring at least ``min_n`` trades.
        """
        combos = [c for c in self._all_triple_combinations() if c["n"] >= min_n]
        if not combos:
            return {}
        return combos[0]

    def worst_regime_combination(self, min_n: int = 5) -> dict:
        """
        Return the combination with the lowest (most negative) Sharpe ratio.
        This combination should be avoided or position-sized down.
        """
        combos = [c for c in self._all_triple_combinations() if c["n"] >= min_n]
        if not combos:
            return {}
        return combos[-1]

    # ------------------------------------------------------------------
    # All-regime summary table
    # ------------------------------------------------------------------

    def full_regime_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame with all triple combinations, sorted by Sharpe.
        Useful for regime-conditional position sizing tables.
        """
        combos = self._all_triple_combinations()
        if not combos:
            return pd.DataFrame()
        return pd.DataFrame(combos).sort_values("sharpe", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Regime stability -- does a regime's edge persist over time?
    # ------------------------------------------------------------------

    def regime_stability_over_time(
        self,
        regime: str = "hurst",
        window_trades: int = 30,
    ) -> pd.DataFrame:
        """
        Rolling-window Sharpe for each regime label over time.

        Parameters
        ----------
        regime : which regime dimension to split on
        window_trades : rolling window size in number of trades
        """
        regime_cfg = {
            "bh_mass": (BH_THRESHOLDS, BH_LABELS, "bh_mass_at_entry"),
            "hurst": (HURST_THRESHOLDS, HURST_LABELS, "hurst_at_entry"),
            "garch_vol": (GARCH_THRESHOLDS, GARCH_LABELS, "garch_vol_at_entry"),
            "nav_omega": (NAV_THRESHOLDS, NAV_LABELS, "nav_omega_at_entry"),
        }
        thresh, labels, attr = regime_cfg[regime]

        sorted_trades = sorted(self.trades, key=lambda t: t.entry_time)
        regime_series: dict[str, list[tuple]] = {lb: [] for lb in labels}

        for t in sorted_trades:
            lb = _bucket_label(getattr(t, attr), thresh, labels)
            regime_series[lb].append((t.entry_time, t.pnl_pct))

        rows = []
        for lb, series in regime_series.items():
            if len(series) < window_trades:
                continue
            for i in range(window_trades, len(series) + 1):
                window = series[i - window_trades : i]
                end_date = window[-1][0]
                pnls = [p for _, p in window]
                rows.append({
                    "regime_label": lb,
                    "end_date": end_date,
                    "rolling_sharpe": _sharpe(pnls),
                    "rolling_win_rate": _win_rate(pnls),
                    "window_n": window_trades,
                })

        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .sort_values(["regime_label", "end_date"])
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Conditional entry filter recommendation
    # ------------------------------------------------------------------

    def regime_filter_recommendations(
        self,
        min_sharpe: float = 0.5,
        min_n: int = 10,
    ) -> dict:
        """
        Recommend which regime combinations to trade and which to avoid,
        based on Sharpe threshold and minimum sample size.

        Returns dict with 'trade' and 'avoid' lists.
        """
        combos = self._all_triple_combinations()
        trade = [
            c for c in combos
            if c["n"] >= min_n and c["sharpe"] >= min_sharpe
        ]
        avoid = [
            c for c in combos
            if c["n"] >= min_n and c["sharpe"] < 0
        ]
        neutral = [
            c for c in combos
            if c["n"] >= min_n and 0 <= c["sharpe"] < min_sharpe
        ]

        return {
            "trade": trade,
            "avoid": avoid,
            "neutral": neutral,
            "threshold_sharpe": min_sharpe,
            "min_n": min_n,
            "total_combos_evaluated": len(combos),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Regime performance analyzer for LARSA v18"
    )
    parser.add_argument("db", help="Path to trades SQLite database")
    parser.add_argument(
        "--analysis",
        default="all",
        choices=["bh", "hurst", "vol", "nav", "joint", "best", "worst", "full"],
    )
    parser.add_argument("--row", default="bh_mass", help="Row regime for joint analysis")
    parser.add_argument("--col", default="hurst", help="Col regime for joint analysis")
    args = parser.parse_args()

    rpa = RegimePerformanceAnalyzer(db_path=args.db)

    if args.analysis == "bh":
        print(rpa.performance_by_bh_mass_level().to_string())
    elif args.analysis == "hurst":
        print(rpa.performance_by_hurst_regime().to_string())
    elif args.analysis == "vol":
        print(rpa.performance_by_vol_regime().to_string())
    elif args.analysis == "nav":
        print(rpa.performance_by_nav_omega_level().to_string())
    elif args.analysis == "joint":
        print(rpa.joint_regime_analysis(args.row, args.col).to_string())
    elif args.analysis == "best":
        import json
        print(json.dumps(rpa.best_regime_combination(), indent=2, default=str))
    elif args.analysis == "worst":
        import json
        print(json.dumps(rpa.worst_regime_combination(), indent=2, default=str))
    elif args.analysis == "full":
        print(rpa.full_regime_summary().to_string())
    else:
        print("=== BH Mass ===")
        print(rpa.performance_by_bh_mass_level().to_string())
        print("\n=== Hurst ===")
        print(rpa.performance_by_hurst_regime().to_string())
        print("\n=== GARCH Vol ===")
        print(rpa.performance_by_vol_regime().to_string())
        print("\n=== NAV Omega ===")
        print(rpa.performance_by_nav_omega_level().to_string())
        print("\n=== Best Combination ===")
        import json
        print(json.dumps(rpa.best_regime_combination(), indent=2, default=str))
        print("\n=== Worst Combination ===")
        print(json.dumps(rpa.worst_regime_combination(), indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
