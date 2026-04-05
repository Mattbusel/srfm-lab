"""
parallel_simulator.py
---------------------
Run N strategy parameter variants simultaneously using multiprocessing.

Usage
-----
    screener = ParallelSimulator(price_data)
    variants = [{"min_hold_bars": k, ...} for k in range(2, 12)]
    results = screener.run(variants)
    top10 = screener.top_n(results, n=10)

Used by IAE to quickly screen parameter combinations before committing to a
full backtest. Returns ranked VariantResult objects sorted by Sharpe.

Design
------
* Each worker runs an independent PaperSimulator instance in its own process.
* Uses multiprocessing.Pool with imap_unordered for efficient CPU utilisation.
* Falls back to single-threaded execution if multiprocessing is unavailable
  (e.g., in REPL, Jupyter, or Windows spawn-mode edge cases).
* Results are collected into VariantResult dataclasses and ranked.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from .paper_simulator import PaperSimulator


# ---------------------------------------------------------------------------
# VariantResult
# ---------------------------------------------------------------------------

@dataclass
class VariantResult:
    """
    Performance summary for one strategy variant after a simulation run.

    Attributes
    ----------
    variant_index    : index into the original variants list
    params           : parameter dict used
    sharpe           : annualised Sharpe ratio
    total_return     : total return over the simulation period
    max_drawdown     : maximum peak-to-trough drawdown (negative)
    win_rate         : fraction of trades with positive P&L
    n_trades         : total number of trades
    final_equity     : ending portfolio value
    error            : error message if simulation failed (else empty string)
    """
    variant_index: int
    params: dict[str, Any]
    sharpe: float
    total_return: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    final_equity: float
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        if self.error:
            return f"Variant {self.variant_index}: ERROR — {self.error}"
        return (
            f"Variant {self.variant_index:3d} | "
            f"Sharpe={self.sharpe:+.3f} | "
            f"Ret={self.total_return:+.1%} | "
            f"DD={self.max_drawdown:.1%} | "
            f"WR={self.win_rate:.1%} | "
            f"N={self.n_trades}"
        )


# ---------------------------------------------------------------------------
# Worker function (module-level for pickling)
# ---------------------------------------------------------------------------

def _simulate_variant(args: tuple) -> VariantResult:
    """
    Worker function: run one PaperSimulator on a price DataFrame.
    args = (variant_index, params, price_data_dict, capital, seed)
    """
    variant_index, params, price_data_dict, capital, seed = args
    try:
        price_data = pd.DataFrame.from_dict(price_data_dict)
        if "index" in price_data.columns:
            price_data.index = pd.to_datetime(price_data["index"])
            price_data = price_data.drop(columns=["index"])
        elif price_data.index.dtype == object:
            price_data.index = pd.to_datetime(price_data.index)

        sim = PaperSimulator(params, capital=capital, seed=seed + variant_index)

        dates = sorted(price_data.index.normalize().unique())
        for day in dates:
            day_data = price_data[price_data.index.normalize() == day]
            sim.step(day_data)

        summary = sim.summary()
        return VariantResult(
            variant_index=variant_index,
            params=params,
            sharpe=summary.get("sharpe", 0.0),
            total_return=summary.get("total_return", 0.0),
            max_drawdown=summary.get("max_drawdown", 0.0),
            win_rate=summary.get("win_rate", 0.0),
            n_trades=summary.get("n_trades", 0),
            final_equity=summary.get("final_equity", capital),
        )
    except Exception as e:
        return VariantResult(
            variant_index=variant_index,
            params=params,
            sharpe=float("-inf"),
            total_return=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            n_trades=0,
            final_equity=capital,
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# ParallelSimulator
# ---------------------------------------------------------------------------

class ParallelSimulator:
    """
    Runs 100+ strategy variants in parallel using multiprocessing.Pool.

    Parameters
    ----------
    price_data   : DataFrame with DatetimeIndex (shared across all variants)
    capital      : starting capital per variant
    n_workers    : CPU workers (default: os.cpu_count())
    seed         : base random seed (each variant gets seed + variant_index)
    """

    def __init__(
        self,
        price_data: pd.DataFrame,
        capital: float = 1_000_000.0,
        n_workers: int | None = None,
        seed: int = 42,
    ) -> None:
        self.price_data = price_data
        self.capital    = capital
        self.n_workers  = n_workers or max(1, (os.cpu_count() or 4) - 1)
        self.seed       = seed

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        variants: list[dict[str, Any]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[VariantResult]:
        """
        Run all variants and return VariantResult list in original order.

        Parameters
        ----------
        variants          : list of parameter dicts
        progress_callback : optional fn(completed, total) called after each result
        """
        if not variants:
            return []

        # Serialise DataFrame for pickling (only send columns actually needed)
        price_dict = self._serialise_price_data()

        args = [
            (i, params, price_dict, self.capital, self.seed)
            for i, params in enumerate(variants)
        ]

        results: list[VariantResult] = [None] * len(variants)  # type: ignore[list-item]
        completed = 0

        try:
            # Use multiprocessing if we can spawn safely
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=min(self.n_workers, len(variants))) as pool:
                for result in pool.imap_unordered(_simulate_variant, args):
                    results[result.variant_index] = result
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(variants))
        except Exception:
            # Fallback: single-threaded
            for arg in args:
                result = _simulate_variant(arg)
                results[result.variant_index] = result
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(variants))

        return results

    def run_grid_search(
        self,
        base_params: dict[str, Any],
        grid: dict[str, list[Any]],
    ) -> list[VariantResult]:
        """
        Grid search over parameter combinations.

        Parameters
        ----------
        base_params : base parameter dict (unchanged params)
        grid        : {param_name: [value1, value2, ...]} to sweep

        Returns results for all combinations.
        """
        variants = list(self._expand_grid(base_params, grid))
        return self.run(variants)

    # ------------------------------------------------------------------
    # Ranking and filtering
    # ------------------------------------------------------------------

    def top_n(
        self,
        results: list[VariantResult],
        n: int = 10,
        metric: str = "sharpe",
        min_trades: int = 10,
    ) -> list[VariantResult]:
        """
        Return top N variants by the given metric.

        Parameters
        ----------
        metric     : "sharpe" | "total_return" | "win_rate" | "max_drawdown"
        min_trades : filter out variants with fewer trades (avoids lucky runs)
        """
        filtered = [r for r in results if not r.error and r.n_trades >= min_trades]
        reverse = metric != "max_drawdown"
        return sorted(filtered, key=lambda r: getattr(r, metric), reverse=reverse)[:n]

    def summary_table(self, results: list[VariantResult]) -> str:
        """Return an ASCII table of all results sorted by Sharpe."""
        sorted_results = sorted(
            results, key=lambda r: r.sharpe if not r.error else float("-inf"), reverse=True
        )
        lines = [
            f"{'Rank':>4}  {'Idx':>4}  {'Sharpe':>8}  {'Return':>8}  "
            f"{'MaxDD':>8}  {'WinRate':>8}  {'Trades':>6}",
            "-" * 62,
        ]
        for rank, r in enumerate(sorted_results, 1):
            if r.error:
                lines.append(f"{rank:4d}  {r.variant_index:4d}  {'ERROR':>8}")
                continue
            lines.append(
                f"{rank:4d}  {r.variant_index:4d}  "
                f"{r.sharpe:+8.3f}  "
                f"{r.total_return:+7.1%}  "
                f"{r.max_drawdown:+7.1%}  "
                f"{r.win_rate:7.1%}  "
                f"{r.n_trades:6d}"
            )
        return "\n".join(lines)

    def pareto_front(
        self, results: list[VariantResult]
    ) -> list[VariantResult]:
        """
        Return the Pareto-optimal variants: those that are not dominated on
        both Sharpe AND max_drawdown simultaneously.
        A variant R1 dominates R2 if R1.sharpe >= R2.sharpe AND R1.max_drawdown <= |R2.max_drawdown|.
        """
        valid = [r for r in results if not r.error]
        pareto: list[VariantResult] = []
        for r in valid:
            dominated = False
            for other in valid:
                if other is r:
                    continue
                if (other.sharpe >= r.sharpe and
                        abs(other.max_drawdown) <= abs(r.max_drawdown)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)
        return sorted(pareto, key=lambda r: r.sharpe, reverse=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _serialise_price_data(self) -> dict:
        """Convert DataFrame to a dict safe for multiprocessing pickling."""
        df = self.price_data.copy()
        df["index"] = df.index.astype(str)
        return df.to_dict(orient="list")

    @staticmethod
    def _expand_grid(
        base: dict[str, Any], grid: dict[str, list[Any]]
    ):
        """
        Yield parameter dicts for every combination in the grid.
        Uses itertools.product over the grid values.
        """
        import itertools
        keys = list(grid.keys())
        for combo in itertools.product(*grid.values()):
            params = dict(base)
            params.update(dict(zip(keys, combo)))
            yield params

    def __repr__(self) -> str:
        shape = self.price_data.shape
        return (
            f"ParallelSimulator("
            f"data={shape}, capital={self.capital:,.0f}, "
            f"workers={self.n_workers})"
        )
