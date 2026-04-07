"""
tools/backtest_compare.py
=========================
Compare different LARSA v18 configurations on the same historical data.

Usage:
    python tools/backtest_compare.py --start 2024-01-01 --end 2024-12-31

No em dashes used anywhere in this file.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from larsa_v18_backtest import (
    LARSAv18Config,
    LARSAv18Backtest,
    BacktestResult,
    generate_synthetic_data,
)

log = logging.getLogger("backtest_compare")

_REPO_ROOT = Path(__file__).parents[1]


# =============================================================================
# COMPARISON ENGINE
# =============================================================================

class BacktestComparison:
    """
    Runs multiple LARSA v18 configuration variants on the same data and
    produces side-by-side performance tables and equity curve overlays.
    """

    def __init__(self) -> None:
        self._variants: dict[str, LARSAv18Config] = {}
        self._results: dict[str, BacktestResult] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the six pre-defined comparison variants."""
        # v18_full -- all features enabled
        self.add_variant("v18_full", LARSAv18Config(
            USE_QUATNAV=True,
            USE_HURST=True,
            USE_ML=True,
            USE_RL=True,
            USE_GRANGER=True,
            USE_EVENT_CAL=True,
        ))

        # v18_no_nav -- QuatNav disabled
        self.add_variant("v18_no_nav", LARSAv18Config(
            USE_QUATNAV=False,
            USE_HURST=True,
            USE_ML=True,
            USE_RL=True,
            USE_GRANGER=True,
            USE_EVENT_CAL=True,
        ))

        # v18_no_hurst -- Hurst filter disabled
        self.add_variant("v18_no_hurst", LARSAv18Config(
            USE_QUATNAV=True,
            USE_HURST=False,
            USE_ML=True,
            USE_RL=True,
            USE_GRANGER=True,
            USE_EVENT_CAL=True,
        ))

        # v18_no_ml -- ML signal disabled
        self.add_variant("v18_no_ml", LARSAv18Config(
            USE_QUATNAV=True,
            USE_HURST=True,
            USE_ML=False,
            USE_RL=True,
            USE_GRANGER=True,
            USE_EVENT_CAL=True,
        ))

        # v18_no_rl -- RL exit disabled
        self.add_variant("v18_no_rl", LARSAv18Config(
            USE_QUATNAV=True,
            USE_HURST=True,
            USE_ML=True,
            USE_RL=False,
            USE_GRANGER=True,
            USE_EVENT_CAL=True,
        ))

        # baseline_bh_only -- only BH mass signal, no modifiers
        baseline = LARSAv18Config(
            USE_QUATNAV=False,
            USE_HURST=False,
            USE_ML=False,
            USE_RL=False,
            USE_GRANGER=False,
            USE_EVENT_CAL=False,
        )
        self.add_variant("baseline_bh_only", baseline)

    def add_variant(self, name: str, config: LARSAv18Config) -> None:
        """Register a strategy variant by name and config."""
        self._variants[name] = config
        log.debug("Registered variant: %s", name)

    def run_all(
        self,
        data: dict[str, pd.DataFrame],
        initial_equity: float = 100_000.0,
    ) -> dict[str, BacktestResult]:
        """Run all registered variants on the same data."""
        engine = LARSAv18Backtest()
        self._results = {}

        for name, cfg in self._variants.items():
            log.info("Running variant: %s", name)
            try:
                result = engine.run(data, cfg, initial_equity=initial_equity)
                self._results[name] = result
                log.info(
                    "  %s: sharpe=%.3f  dd=%.2f%%  ret=%.2f%%  wr=%.1f%%",
                    name,
                    result.sharpe,
                    result.max_drawdown * 100,
                    result.total_return * 100,
                    result.win_rate * 100,
                )
            except Exception as exc:
                log.error("Variant %s failed: %s", name, exc)

        return self._results

    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Build a comparison DataFrame with columns:
        variant, sharpe, sortino, max_drawdown, calmar,
        total_return, win_rate, n_trades, avg_hold_bars
        """
        if not self._results:
            raise RuntimeError("No results available -- run run_all() first")

        rows = []
        for name, result in self._results.items():
            rows.append({
                "variant": name,
                "sharpe": round(result.sharpe, 4),
                "sortino": round(result.sortino, 4),
                "max_drawdown": round(result.max_drawdown, 4),
                "calmar": round(result.calmar, 4),
                "total_return": round(result.total_return, 4),
                "win_rate": round(result.win_rate, 4),
                "n_trades": result.n_trades,
                "avg_hold_bars": round(result.avg_hold_bars, 1),
            })

        df = pd.DataFrame(rows).set_index("variant")
        df.sort_values("sharpe", ascending=False, inplace=True)
        return df

    def plot_equity_curves(
        self, results: dict[str, BacktestResult] | None = None
    ) -> None:
        """
        Overlay equity curves for all variants using matplotlib.
        Falls back gracefully if matplotlib is not available.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            log.warning("matplotlib not available -- skipping plot")
            return

        r = results or self._results
        if not r:
            log.warning("No results to plot")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        ax_eq, ax_dd = axes

        colors = plt.cm.tab10(np.linspace(0, 1, len(r)))

        for (name, result), color in zip(r.items(), colors):
            eq = result.equity_curve
            if eq is None or len(eq) == 0:
                continue
            label = f"{name} (SR={result.sharpe:.2f}, DD={result.max_drawdown:.1%})"
            ax_eq.plot(eq.index, eq.values, label=label, color=color, linewidth=1.5)

            # Drawdown series
            running_max = eq.cummax()
            dd_series = (eq - running_max) / running_max
            ax_dd.fill_between(dd_series.index, dd_series.values, 0,
                               alpha=0.3, color=color, label=name)

        ax_eq.set_ylabel("Portfolio Equity ($)")
        ax_eq.set_title("LARSA v18 -- Equity Curve Comparison")
        ax_eq.legend(fontsize=8, loc="upper left")
        ax_eq.grid(True, alpha=0.3)

        ax_dd.set_ylabel("Drawdown")
        ax_dd.set_xlabel("Date")
        ax_dd.set_title("Drawdown")
        ax_dd.legend(fontsize=8, loc="lower left")
        ax_dd.grid(True, alpha=0.3)
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        plt.show()

    def best_variant(self) -> str | None:
        """Return the name of the highest-Sharpe variant."""
        if not self._results:
            return None
        return max(self._results, key=lambda n: self._results[n].sharpe)

    def ablation_summary(self) -> dict[str, float]:
        """
        Return the Sharpe contribution of each feature by comparing
        v18_full to ablated variants.
        """
        if "v18_full" not in self._results:
            return {}

        full_sharpe = self._results["v18_full"].sharpe
        ablation_pairs = {
            "QuatNav":   ("v18_no_nav",   "v18_full"),
            "Hurst":     ("v18_no_hurst", "v18_full"),
            "ML Signal": ("v18_no_ml",    "v18_full"),
            "RL Exit":   ("v18_no_rl",    "v18_full"),
        }

        summary: dict[str, float] = {}
        for feature, (ablated, full) in ablation_pairs.items():
            if ablated in self._results and full in self._results:
                delta = self._results[full].sharpe - self._results[ablated].sharpe
                summary[feature] = round(delta, 4)

        return summary

    def print_report(self) -> None:
        """Print a formatted comparison report to stdout."""
        print("\n" + "=" * 70)
        print("LARSA v18 -- Backtest Comparison Report")
        print("=" * 70)

        try:
            table = self.generate_comparison_table()
            print("\nPerformance Table:")
            print(table.to_string())
        except RuntimeError:
            print("  No results yet -- run run_all() first")
            return

        best = self.best_variant()
        if best:
            print(f"\nBest variant by Sharpe: {best}")

        ablation = self.ablation_summary()
        if ablation:
            print("\nAblation (Sharpe contribution vs v18_full):")
            for feature, delta in sorted(ablation.items(), key=lambda x: -abs(x[1])):
                sign = "+" if delta >= 0 else ""
                print(f"  {feature:12s}: {sign}{delta:.4f}")

        print("=" * 70)


# =============================================================================
# COMPARISON RUNNER UTILITIES
# =============================================================================

def run_standard_comparison(
    data: dict[str, pd.DataFrame],
    initial_equity: float = 100_000.0,
    plot: bool = True,
) -> BacktestComparison:
    """Run all standard LARSA v18 variants and print the report."""
    comp = BacktestComparison()
    comp.run_all(data, initial_equity=initial_equity)
    comp.print_report()
    if plot:
        comp.plot_equity_curves()
    return comp


def run_pairwise_comparison(
    data: dict[str, pd.DataFrame],
    name_a: str,
    cfg_a: LARSAv18Config,
    name_b: str,
    cfg_b: LARSAv18Config,
    initial_equity: float = 100_000.0,
) -> tuple[BacktestResult, BacktestResult]:
    """Run two specific variants and return both results."""
    comp = BacktestComparison()
    comp._variants = {}  # clear defaults
    comp.add_variant(name_a, cfg_a)
    comp.add_variant(name_b, cfg_b)
    results = comp.run_all(data, initial_equity=initial_equity)
    return results[name_a], results[name_b]


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LARSA v18 Backtest Comparison")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--symbols", nargs="+",
                        default=["BTC", "ETH", "XRP", "SPY", "QQQ"])
    parser.add_argument("--source", default="synthetic",
                        choices=["auto", "sqlite", "csv", "synthetic"])
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    bt = LARSAv18Backtest()
    data = bt.load_data(args.symbols, start, end, source=args.source)

    log.info("Loaded data: %s bars per symbol", {s: len(df) for s, df in data.items()})

    comp = run_standard_comparison(
        data,
        initial_equity=args.equity,
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
