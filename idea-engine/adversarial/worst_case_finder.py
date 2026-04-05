"""
worst_case_finder.py
====================
Systematic worst-case search around the current strategy parameters.

We perform a structured grid search in a ±50% hypercube around each
current parameter value, varying one parameter at a time while holding
all others fixed (one-at-a-time / OAT sensitivity analysis), then
computing a full sensitivity matrix using finite differences.

Results
-------
For each parameter we report:
    - Value that causes the most damage (worst single-parameter perturbation).
    - Estimated P&L drop at that value vs baseline.
    - Sensitivity dP&L / d(param) at the current parameter value.

The sensitivity matrix identifies which parameters the strategy is most
exposed to -- these should be monitored most closely in production.

Usage::

    finder = WorstCaseFinder(current_params=params, trade_stats=stats)
    result = finder.run()
    print(result.sensitivity_ranking())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .fuzzer import _fast_backtest, PARAM_NAMES, FUZZ_BOUNDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default current parameters (matching IAE genome baseline)
# ---------------------------------------------------------------------------

CURRENT_PARAMS: Dict[str, float] = {
    "min_hold_bars":       8.0,
    "blocked_hours_count": 0.0,
    "garch_target_vol":    1.20,
    "corr_factor":         0.25,
    "winner_prot_pct":     0.005,
    "stale_15m_move":      0.005,
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SingleParamResult:
    """Result for one parameter's worst-case analysis."""

    param_name:        str
    current_value:     float
    worst_value:       float
    baseline_pnl:      float
    worst_pnl:         float
    pnl_drop:          float          # worst_pnl - baseline_pnl (negative = loss)
    pnl_drop_pct:      float          # pnl_drop / |baseline_pnl|
    sensitivity:       float          # dP&L / d(param) via finite differences
    n_grid_points:     int = 0


@dataclass
class SensitivityResult:
    """
    Full result of the worst-case sensitivity analysis.

    Attributes
    ----------
    baseline_pnl        : P&L at current (unperturbed) parameters.
    param_results       : per-parameter worst-case analysis.
    sensitivity_matrix  : dict of {param: dP&L/d(param)}.
    most_damaging_param : which single parameter change causes most harm.
    most_damaging_value : value of that parameter at the worst case.
    """

    baseline_pnl:         float
    param_results:        Dict[str, SingleParamResult]
    sensitivity_matrix:   Dict[str, float]
    most_damaging_param:  str
    most_damaging_value:  float

    def sensitivity_ranking(self) -> str:
        """Return a human-readable ranking of parameter sensitivity."""
        ranked = sorted(
            self.sensitivity_matrix.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        lines = [
            "Sensitivity ranking (|dP&L / d(param)|):",
            f"  Baseline P&L: {self.baseline_pnl:+.4f}",
            f"  Most damaging single change: {self.most_damaging_param} -> "
            f"{self.most_damaging_value:.4f}",
            "",
        ]
        for rank, (name, sens) in enumerate(ranked, 1):
            pr = self.param_results[name]
            lines.append(
                f"  {rank}. {name:<25s} sensitivity={sens:+.4f}  "
                f"worst_drop={pr.pnl_drop:+.4f} ({pr.pnl_drop_pct:+.1f}%)  "
                f"at value={pr.worst_value:.4f}"
            )
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return results as a pandas DataFrame."""
        rows = []
        for name, pr in self.param_results.items():
            rows.append({
                "param":          name,
                "current_value":  pr.current_value,
                "worst_value":    pr.worst_value,
                "baseline_pnl":   pr.baseline_pnl,
                "worst_pnl":      pr.worst_pnl,
                "pnl_drop":       pr.pnl_drop,
                "pnl_drop_pct":   pr.pnl_drop_pct,
                "sensitivity":    pr.sensitivity,
            })
        return pd.DataFrame(rows).sort_values("pnl_drop").reset_index(drop=True)


# ---------------------------------------------------------------------------
# WorstCaseFinder
# ---------------------------------------------------------------------------

class WorstCaseFinder:
    """
    One-at-a-time sensitivity analysis for worst-case parameter discovery.

    Parameters
    ----------
    current_params  : dict of current parameter values.
    n_grid_points   : number of grid points per parameter dimension.
    n_replications  : number of random seeds to average over (reduces noise).
    perturbation_pct: fraction of the parameter range to search (±50% default).
    n_days          : length of each mini-backtest in days.
    seed            : base random seed.
    """

    def __init__(
        self,
        current_params: Optional[Dict[str, float]] = None,
        n_grid_points:  int   = 41,
        n_replications: int   = 5,
        perturbation_pct: float = 0.50,
        n_days: int   = 90,
        seed:   int   = 123,
    ):
        self.current_params   = current_params or CURRENT_PARAMS.copy()
        self.n_grid_points    = n_grid_points
        self.n_replications   = n_replications
        self.perturbation_pct = perturbation_pct
        self.n_days           = n_days
        self.seed             = seed

    def _evaluate(self, params_dict: Dict[str, float]) -> float:
        """
        Evaluate average P&L over *n_replications* for a parameter dict.

        Parameters
        ----------
        params_dict : complete set of parameters as a dict.

        Returns
        -------
        Mean P&L across replications.
        """
        params_arr = np.array([params_dict.get(k, CURRENT_PARAMS.get(k, 0.0))
                                for k in PARAM_NAMES])
        pnls = []
        for rep in range(self.n_replications):
            rng = np.random.default_rng(self.seed + rep * 997)
            pnls.append(_fast_backtest(params_arr, n_days=self.n_days, rng=rng))
        return float(np.mean(pnls))

    def _param_grid(self, name: str) -> np.ndarray:
        """
        Build a grid of values for *name* spanning ±perturbation_pct around
        the current value, clipped to the fuzzer bounds.
        """
        cur  = self.current_params.get(name, CURRENT_PARAMS.get(name, 1.0))
        lo_b = FUZZ_BOUNDS[name][0]
        hi_b = FUZZ_BOUNDS[name][1]
        lo   = max(cur * (1 - self.perturbation_pct), lo_b)
        hi   = min(cur * (1 + self.perturbation_pct), hi_b)
        if lo >= hi:
            lo = lo_b
            hi = hi_b
        return np.linspace(lo, hi, self.n_grid_points)

    def run(self) -> SensitivityResult:
        """
        Execute the full OAT sensitivity analysis.

        Returns
        -------
        SensitivityResult with per-parameter reports and sensitivity matrix.
        """
        logger.info("Running worst-case sensitivity analysis (%d params, %d grid points each).",
                    len(self.current_params), self.n_grid_points)

        baseline_pnl = self._evaluate(self.current_params)
        logger.info("Baseline P&L: %.4f", baseline_pnl)

        param_results: Dict[str, SingleParamResult] = {}
        sensitivity_matrix: Dict[str, float] = {}

        for name in PARAM_NAMES:
            if name not in self.current_params:
                continue

            grid       = self._param_grid(name)
            cur        = self.current_params[name]
            pnls       = np.zeros(len(grid))
            test_params = self.current_params.copy()

            for i, val in enumerate(grid):
                test_params[name] = val
                pnls[i]           = self._evaluate(test_params)

            # Reset for next param
            test_params[name] = cur

            worst_idx   = int(np.argmin(pnls))
            worst_val   = float(grid[worst_idx])
            worst_pnl   = float(pnls[worst_idx])
            pnl_drop    = worst_pnl - baseline_pnl
            denom       = max(abs(baseline_pnl), 1e-6)
            pnl_drop_pct = 100.0 * pnl_drop / denom

            # Sensitivity via finite differences around current value
            cur_idx    = int(np.argmin(np.abs(grid - cur)))
            if cur_idx > 0 and cur_idx < len(grid) - 1:
                dp    = float(grid[cur_idx + 1] - grid[cur_idx - 1])
                dpnl  = float(pnls[cur_idx + 1] - pnls[cur_idx - 1])
                sensitivity = dpnl / dp if abs(dp) > 1e-12 else 0.0
            else:
                sensitivity = 0.0

            logger.info(
                "  %s: baseline=%.4f, worst=%.4f (at val=%.4f), drop=%.1f%%, sens=%+.3f",
                name, baseline_pnl, worst_pnl, worst_val, pnl_drop_pct, sensitivity,
            )

            param_results[name] = SingleParamResult(
                param_name=name,
                current_value=cur,
                worst_value=worst_val,
                baseline_pnl=baseline_pnl,
                worst_pnl=worst_pnl,
                pnl_drop=pnl_drop,
                pnl_drop_pct=pnl_drop_pct,
                sensitivity=sensitivity,
                n_grid_points=len(grid),
            )
            sensitivity_matrix[name] = sensitivity

        # Identify most damaging single-parameter change
        most_damaging = min(param_results.values(), key=lambda r: r.pnl_drop)

        result = SensitivityResult(
            baseline_pnl=baseline_pnl,
            param_results=param_results,
            sensitivity_matrix=sensitivity_matrix,
            most_damaging_param=most_damaging.param_name,
            most_damaging_value=most_damaging.worst_value,
        )

        logger.info("\n%s", result.sensitivity_ranking())
        return result

    def report_message(self, result: SensitivityResult) -> str:
        """
        Generate a plain-English risk summary for the sensitivity result.

        Parameters
        ----------
        result : SensitivityResult from run().

        Returns
        -------
        Multi-line string suitable for a Slack alert or email.
        """
        lines = [
            "=== WORST-CASE SENSITIVITY REPORT ===",
            f"Baseline P&L: {result.baseline_pnl:+.4f}",
            f"Most dangerous single parameter: {result.most_damaging_param}",
            f"  -> If {result.most_damaging_param} drifts to {result.most_damaging_value:.4f},"
            f" total P&L could drop by "
            f"{result.param_results[result.most_damaging_param].pnl_drop:+.4f}",
            "",
        ]
        df = result.to_dataframe()
        lines.append("Top 3 risks:")
        for _, row in df.head(3).iterrows():
            lines.append(
                f"  {row['param']}: drop {row['pnl_drop']:+.4f} "
                f"({row['pnl_drop_pct']:+.1f}%) at value {row['worst_value']:.4f}"
            )
        return "\n".join(lines)
