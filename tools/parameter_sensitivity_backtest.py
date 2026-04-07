"""
tools/parameter_sensitivity_backtest.py
========================================
Fast parameter sensitivity analysis for the LARSA v18 backtest engine.
Uses ProcessPoolExecutor for parallelism.

Usage:
    python tools/parameter_sensitivity_backtest.py --start 2024-01-01 --end 2024-12-31

No em dashes used anywhere in this file.
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
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

log = logging.getLogger("parameter_sensitivity")

_REPO_ROOT = Path(__file__).parents[1]


# =============================================================================
# PARAMETER GRID DEFINITION
# =============================================================================

DEFAULT_PARAMETER_GRIDS: dict[str, list[Any]] = {
    "BH_FORM":             [1.5, 1.7, 1.92, 2.1, 2.3],
    "NAV_OMEGA_SCALE_K":   [0.25, 0.50, 0.75, 1.00],
    "NAV_GEO_ENTRY_GATE":  [2.0, 2.5, 3.0, 3.5],
    "HURST_WINDOW":        [60, 80, 100, 120],
}


# =============================================================================
# WORKER FUNCTION (module-level for pickling)
# =============================================================================

def _run_single_backtest(args: tuple) -> dict[str, Any]:
    """
    Worker function for ProcessPoolExecutor.
    args: (param_name, param_value, config_kwargs, data_dict, initial_equity)
    """
    param_name, param_value, config_kwargs, data_dict, initial_equity = args
    try:
        cfg = LARSAv18Config(**config_kwargs)
        setattr(cfg, param_name, param_value)
        engine = LARSAv18Backtest()
        result = engine.run(data_dict, cfg, initial_equity=initial_equity)
        return {
            "param_name": param_name,
            "param_value": param_value,
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "max_drawdown": result.max_drawdown,
            "calmar": result.calmar,
            "total_return": result.total_return,
            "win_rate": result.win_rate,
            "n_trades": result.n_trades,
            "error": None,
        }
    except Exception as exc:
        return {
            "param_name": param_name,
            "param_value": param_value,
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "total_return": float("nan"),
            "win_rate": float("nan"),
            "n_trades": 0,
            "error": str(exc),
        }


def _run_grid_backtest(args: tuple) -> dict[str, Any]:
    """
    Worker function for full grid search (all params varied simultaneously).
    args: (param_combo, config_kwargs, data_dict, initial_equity)
    """
    param_combo, config_kwargs, data_dict, initial_equity = args
    try:
        cfg = LARSAv18Config(**config_kwargs)
        for param_name, param_value in param_combo.items():
            setattr(cfg, param_name, param_value)
        engine = LARSAv18Backtest()
        result = engine.run(data_dict, cfg, initial_equity=initial_equity)
        row = dict(param_combo)
        row.update({
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "max_drawdown": result.max_drawdown,
            "calmar": result.calmar,
            "total_return": result.total_return,
            "win_rate": result.win_rate,
            "n_trades": result.n_trades,
            "error": None,
        })
        return row
    except Exception as exc:
        row = dict(param_combo)
        row.update({
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "total_return": float("nan"),
            "win_rate": float("nan"),
            "n_trades": 0,
            "error": str(exc),
        })
        return row


# =============================================================================
# SENSITIVITY ANALYSIS CLASS
# =============================================================================

class ParameterSensitivityBacktest:
    """
    Runs LARSA v18 backtests across a parameter grid and reports
    sensitivity of Sharpe ratio to each parameter.

    Parameter grids (defaults):
      BH_FORM:            [1.5, 1.7, 1.92, 2.1, 2.3]
      NAV_OMEGA_SCALE_K:  [0.25, 0.50, 0.75, 1.00]
      NAV_GEO_ENTRY_GATE: [2.0, 2.5, 3.0, 3.5]
      HURST_WINDOW:       [60, 80, 100, 120]
    """

    def __init__(
        self,
        grids: dict[str, list[Any]] | None = None,
        base_config: LARSAv18Config | None = None,
        max_workers: int | None = None,
    ) -> None:
        self._grids = grids or DEFAULT_PARAMETER_GRIDS
        self._base_cfg = base_config or LARSAv18Config()
        self._max_workers = max_workers
        self._sensitivity_records: list[dict[str, Any]] = []
        self._grid_records: list[dict[str, Any]] = []

    def _cfg_as_kwargs(self) -> dict[str, Any]:
        """Serialize the base config to a plain dict for pickling."""
        cfg = self._base_cfg
        return {
            f.name: getattr(cfg, f.name)
            for f in cfg.__dataclass_fields__.values()
            if f.name not in ("TF_CAP", "INSTRUMENTS",
                              "BLOCKED_ENTRY_HOURS_UTC",
                              "BOOST_ENTRY_HOURS_UTC")
        }

    def run_sensitivity(
        self,
        data: dict[str, pd.DataFrame],
        initial_equity: float = 100_000.0,
    ) -> None:
        """
        Run one-at-a-time sensitivity: vary each parameter independently
        while holding all others at their base values.
        Uses ProcessPoolExecutor for parallelism.
        """
        base_kwargs = self._cfg_as_kwargs()
        tasks = []
        for param_name, values in self._grids.items():
            for value in values:
                tasks.append((param_name, value, base_kwargs, data, initial_equity))

        self._sensitivity_records = []
        total = len(tasks)
        log.info("Running %d sensitivity backtests...", total)

        try:
            with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {executor.submit(_run_single_backtest, t): t for t in tasks}
                done = 0
                for future in as_completed(futures):
                    result = future.result()
                    self._sensitivity_records.append(result)
                    done += 1
                    if done % 5 == 0 or done == total:
                        log.info("  Sensitivity: %d/%d done", done, total)
        except Exception as exc:
            # Fall back to sequential execution if multiprocessing fails
            log.warning("Parallel execution failed (%s), running sequentially", exc)
            for task in tasks:
                result = _run_single_backtest(task)
                self._sensitivity_records.append(result)

        log.info("Sensitivity analysis complete: %d results", len(self._sensitivity_records))

    def run_grid_search(
        self,
        data: dict[str, pd.DataFrame],
        initial_equity: float = 100_000.0,
        params: list[str] | None = None,
    ) -> None:
        """
        Run full Cartesian grid search over specified parameters.
        params: subset of self._grids keys. If None, uses all keys.
        WARNING: can be expensive -- use a subset or reduced grids.
        """
        target_params = params or list(self._grids.keys())
        grid_values = [self._grids[p] for p in target_params]
        combos = list(itertools.product(*grid_values))

        base_kwargs = self._cfg_as_kwargs()
        tasks = []
        for combo_vals in combos:
            combo = dict(zip(target_params, combo_vals))
            tasks.append((combo, base_kwargs, data, initial_equity))

        self._grid_records = []
        total = len(tasks)
        log.info("Running %d grid search backtests...", total)

        try:
            with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {executor.submit(_run_grid_backtest, t): t for t in tasks}
                done = 0
                for future in as_completed(futures):
                    result = future.result()
                    self._grid_records.append(result)
                    done += 1
                    if done % 10 == 0 or done == total:
                        log.info("  Grid search: %d/%d done", done, total)
        except Exception as exc:
            log.warning("Parallel grid search failed (%s), running sequentially", exc)
            for task in tasks:
                result = _run_grid_backtest(task)
                self._grid_records.append(result)

        log.info("Grid search complete: %d results", len(self._grid_records))

    def sensitivity_table(self) -> pd.DataFrame:
        """
        Return a DataFrame showing Sharpe by parameter name and value.
        Columns: param_name, param_value, sharpe, sortino, max_drawdown,
                 calmar, total_return, win_rate, n_trades
        """
        if not self._sensitivity_records:
            raise RuntimeError("No sensitivity results -- run run_sensitivity() first")

        df = pd.DataFrame(self._sensitivity_records)
        df = df[df["error"].isna()].drop(columns=["error"])
        df["param_value"] = df["param_value"].astype(float)
        df.sort_values(["param_name", "param_value"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def sensitivity_pivot(self, metric: str = "sharpe") -> pd.DataFrame:
        """
        Return a pivot table: rows = param_name, columns = param_value,
        values = metric.
        """
        df = self.sensitivity_table()
        pivot = df.pivot(index="param_name", columns="param_value", values=metric)
        return pivot

    def optimal_params(self, metric: str = "sharpe") -> dict[str, Any]:
        """
        Return {param_name: optimal_value} where optimal = highest metric.
        Uses one-at-a-time sensitivity results.
        """
        if not self._sensitivity_records:
            raise RuntimeError("No sensitivity results -- run run_sensitivity() first")

        df = self.sensitivity_table()
        optimal: dict[str, Any] = {}
        for param_name in df["param_name"].unique():
            sub = df[df["param_name"] == param_name]
            best_row = sub.loc[sub[metric].idxmax()]
            optimal[param_name] = best_row["param_value"]

        return optimal

    def optimal_config(self, metric: str = "sharpe") -> LARSAv18Config:
        """Return a LARSAv18Config with all parameters set to their optimal values."""
        params = self.optimal_params(metric=metric)
        cfg = copy.deepcopy(self._base_cfg)
        for param_name, value in params.items():
            if hasattr(cfg, param_name):
                # Preserve integer types where appropriate
                existing = getattr(cfg, param_name)
                if isinstance(existing, int):
                    value = int(value)
                elif isinstance(existing, float):
                    value = float(value)
                setattr(cfg, param_name, value)
        return cfg

    def grid_search_table(self) -> pd.DataFrame:
        """Return the full grid search results as a DataFrame."""
        if not self._grid_records:
            raise RuntimeError("No grid search results -- run run_grid_search() first")
        df = pd.DataFrame(self._grid_records)
        df = df[df["error"].isna()].drop(columns=["error"])
        df.sort_values("sharpe", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def optimal_grid_params(self, metric: str = "sharpe") -> dict[str, Any]:
        """Return the parameter combination with the highest grid-search metric."""
        df = self.grid_search_table()
        if df.empty:
            return {}
        best = df.iloc[0]
        param_names = [p for p in self._grids if p in df.columns]
        return {p: best[p] for p in param_names}

    def plot_sensitivity(
        self, metric: str = "sharpe", params: list[str] | None = None
    ) -> None:
        """Plot sensitivity curves for each parameter."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not available -- skipping plot")
            return

        df = self.sensitivity_table()
        target_params = params or sorted(df["param_name"].unique())
        n = len(target_params)
        if n == 0:
            return

        cols = min(2, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
        if n == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = [ax for row in axes for ax in row]

        for i, param_name in enumerate(target_params):
            ax = axes[i]
            sub = df[df["param_name"] == param_name].sort_values("param_value")
            ax.plot(sub["param_value"], sub[metric], "o-", color="steelblue",
                    linewidth=2, markersize=6)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"Sensitivity: {param_name}")
            ax.grid(True, alpha=0.3)

            # Mark optimum
            best_idx = sub[metric].idxmax()
            best_x = sub.loc[best_idx, "param_value"]
            best_y = sub.loc[best_idx, metric]
            ax.annotate(
                f"opt={best_x}",
                xy=(best_x, best_y),
                xytext=(best_x, best_y + abs(best_y) * 0.1 + 0.05),
                fontsize=8,
                color="red",
                ha="center",
            )

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"LARSA v18 -- Parameter Sensitivity ({metric})", fontsize=12)
        plt.tight_layout()
        plt.show()

    def print_report(self) -> None:
        """Print a formatted sensitivity report."""
        print("\n" + "=" * 70)
        print("LARSA v18 -- Parameter Sensitivity Report")
        print("=" * 70)

        if not self._sensitivity_records:
            print("  No results yet -- run run_sensitivity() first")
            return

        try:
            pivot = self.sensitivity_pivot("sharpe")
            print("\nSharpe by Parameter Value:")
            print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
        except Exception as exc:
            print(f"  Could not build pivot: {exc}")

        try:
            optimal = self.optimal_params("sharpe")
            print("\nOptimal Values (highest Sharpe per parameter):")
            for param, val in optimal.items():
                print(f"  {param:25s}: {val}")
        except Exception as exc:
            print(f"  Could not compute optimal params: {exc}")

        if self._grid_records:
            try:
                grid_df = self.grid_search_table()
                print("\nTop 5 Parameter Combinations (grid search):")
                print(grid_df.head(5).to_string())
            except Exception as exc:
                print(f"  Grid search table failed: {exc}")

        print("=" * 70)


# =============================================================================
# CONVENIENCE RUNNER
# =============================================================================

def run_full_sensitivity(
    data: dict[str, pd.DataFrame],
    grids: dict[str, list[Any]] | None = None,
    initial_equity: float = 100_000.0,
    plot: bool = True,
    max_workers: int | None = None,
) -> ParameterSensitivityBacktest:
    """Run sensitivity analysis and print/plot the report."""
    sens = ParameterSensitivityBacktest(grids=grids, max_workers=max_workers)
    sens.run_sensitivity(data, initial_equity=initial_equity)
    sens.print_report()
    if plot:
        sens.plot_sensitivity()
    return sens


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="LARSA v18 Parameter Sensitivity Analysis"
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--symbols", nargs="+",
                        default=["BTC", "ETH", "XRP", "SPY", "QQQ"])
    parser.add_argument("--source", default="synthetic",
                        choices=["auto", "sqlite", "csv", "synthetic"])
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--grid-search", action="store_true",
                        help="Also run a Cartesian grid search (expensive)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    from larsa_v18_backtest import LARSAv18Backtest
    bt = LARSAv18Backtest()
    data = bt.load_data(args.symbols, start, end, source=args.source)
    log.info("Loaded: %s", {s: len(df) for s, df in data.items()})

    sens = ParameterSensitivityBacktest(max_workers=args.workers)
    sens.run_sensitivity(data, initial_equity=args.equity)

    if args.grid_search:
        # Reduced grid for speed
        small_grids = {
            "BH_FORM":            [1.7, 1.92, 2.1],
            "NAV_OMEGA_SCALE_K":  [0.25, 0.50, 1.00],
        }
        reduced = ParameterSensitivityBacktest(
            grids=small_grids, max_workers=args.workers
        )
        reduced.run_grid_search(data, initial_equity=args.equity,
                                params=list(small_grids.keys()))
        sens._grid_records = reduced._grid_records

    sens.print_report()
    if not args.no_plot:
        sens.plot_sensitivity()


if __name__ == "__main__":
    main()
