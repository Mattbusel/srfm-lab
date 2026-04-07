"""
iae_performance_tracker.py -- Track IAE adaptation performance over cycles.

The IAE runs discrete adaptation cycles.  After each cycle it emits a result
that the Python layer records here.  This module accumulates those results,
computes rolling improvement rates, detects quality degradation, and generates
plain-text Markdown reports.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IAECycleResult:
    """
    Result produced by one IAE adaptation cycle.

    params_changed maps parameter name to (old_value, new_value).
    live_sharpe_before/after are the realized Sharpe ratios computed from live
    trading performance in the window just before and just after the cycle.
    """

    cycle_id: str
    timestamp: datetime
    n_evaluations: int
    best_fitness: float
    mean_fitness: float
    params_changed: Dict[str, Tuple[float, float]]  # {name: (old, new)}
    live_sharpe_before: float
    live_sharpe_after: float

    @property
    def sharpe_delta(self) -> float:
        return self.live_sharpe_after - self.live_sharpe_before

    @property
    def fitness_gain_per_eval(self) -> float:
        """Average fitness improvement per evaluation in this cycle."""
        if self.n_evaluations <= 0:
            return 0.0
        return self.best_fitness / self.n_evaluations


@dataclass
class QualityReport:
    """Result of AdaptationQualityMonitor.check_adaptation_quality."""

    status: str  # "ok" | "warning" | "critical"
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        return self.status == "ok"

    def to_markdown(self) -> str:
        lines = [
            f"## Adaptation Quality Report",
            f"",
            f"**Status**: {self.status.upper()}",
            f"",
        ]
        if self.metrics:
            lines.append("### Metrics")
            for k, v in sorted(self.metrics.items()):
                lines.append(f"- {k}: {v:.4f}")
            lines.append("")
        if self.warnings:
            lines.append("### Warnings")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")
        if self.recommendations:
            lines.append("### Recommendations")
            for r in self.recommendations:
                lines.append(f"- {r}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# IAEPerformanceTracker
# ---------------------------------------------------------------------------

class IAEPerformanceTracker:
    """
    Accumulate IAE cycle results and compute adaptation performance metrics.

    All results are stored in memory.  Callers should persist them externally
    if needed (e.g., append to a Parquet file after each record_cycle call).
    """

    def __init__(self) -> None:
        self._cycles: List[IAECycleResult] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_cycle(self, cycle_result: IAECycleResult) -> None:
        """Append a new cycle result.  Cycles are kept in insertion order."""
        if not isinstance(cycle_result, IAECycleResult):
            raise TypeError(f"Expected IAECycleResult, got {type(cycle_result)}")
        self._cycles.append(cycle_result)
        logger.debug(
            "Recorded cycle %s  best_fitness=%.6f  n_evals=%d",
            cycle_result.cycle_id,
            cycle_result.best_fitness,
            cycle_result.n_evaluations,
        )

    # ------------------------------------------------------------------
    # Rolling improvement rate
    # ------------------------------------------------------------------

    def rolling_improvement_rate(self, n_cycles: int = 10) -> float:
        """
        Mean per-cycle improvement in best_fitness over the last n_cycles.

        Returns 0.0 if fewer than 2 cycles are available.
        """
        recent = self._cycles[-n_cycles:] if len(self._cycles) >= n_cycles else self._cycles
        if len(recent) < 2:
            return 0.0

        gains = []
        for i in range(1, len(recent)):
            gains.append(recent[i].best_fitness - recent[i - 1].best_fitness)

        return float(np.mean(gains))

    # ------------------------------------------------------------------
    # Time to improvement
    # ------------------------------------------------------------------

    def time_to_improvement(self, threshold: float = 0.05) -> Optional[int]:
        """
        Return the number of cycles it took for best_fitness to improve by
        at least threshold (absolute) from the first recorded cycle.

        Returns None if the threshold was never reached.
        """
        if not self._cycles:
            return None

        baseline = self._cycles[0].best_fitness
        target = baseline + threshold

        for i, c in enumerate(self._cycles):
            if c.best_fitness >= target:
                return i  # 0-indexed cycle count from start

        return None

    # ------------------------------------------------------------------
    # Parameter update frequency
    # ------------------------------------------------------------------

    def parameter_update_frequency(self) -> Dict[str, int]:
        """
        Return a dict mapping each parameter name to the number of cycles in
        which it was updated (i.e., appeared in params_changed).
        """
        freq: Dict[str, int] = {}
        for c in self._cycles:
            for pname in c.params_changed:
                freq[pname] = freq.get(pname, 0) + 1
        return dict(sorted(freq.items(), key=lambda kv: kv[1], reverse=True))

    # ------------------------------------------------------------------
    # Adaptation efficiency
    # ------------------------------------------------------------------

    def adaptation_efficiency(self) -> float:
        """
        Total fitness gain divided by total number of evaluations across all
        recorded cycles.

        Returns 0.0 if no evaluations have been performed.
        """
        if not self._cycles:
            return 0.0

        total_evals = sum(c.n_evaluations for c in self._cycles)
        if total_evals == 0:
            return 0.0

        first_fit = self._cycles[0].best_fitness
        last_fit = self._cycles[-1].best_fitness
        total_gain = last_fit - first_fit

        return total_gain / total_evals

    # ------------------------------------------------------------------
    # DataFrame representation
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all recorded cycles to a DataFrame for analysis."""
        if not self._cycles:
            return pd.DataFrame()

        rows = []
        for c in self._cycles:
            rows.append(
                {
                    "cycle_id": c.cycle_id,
                    "timestamp": c.timestamp,
                    "n_evaluations": c.n_evaluations,
                    "best_fitness": c.best_fitness,
                    "mean_fitness": c.mean_fitness,
                    "n_params_changed": len(c.params_changed),
                    "live_sharpe_before": c.live_sharpe_before,
                    "live_sharpe_after": c.live_sharpe_after,
                    "sharpe_delta": c.sharpe_delta,
                }
            )
        return pd.DataFrame(rows).set_index("cycle_id")

    # ------------------------------------------------------------------
    # Oscillation detection helper
    # ------------------------------------------------------------------

    def _detect_oscillating_params(self, last_n: int = 20) -> List[str]:
        """
        Return parameter names whose values oscillate (alternate direction)
        at least 3 times in the last last_n cycles.
        """
        recent = self._cycles[-last_n:] if len(self._cycles) >= last_n else self._cycles

        # Build per-param value sequence
        param_values: Dict[str, List[float]] = {}
        for c in recent:
            for pname, (old_val, new_val) in c.params_changed.items():
                param_values.setdefault(pname, []).append(new_val)

        oscillating = []
        for pname, vals in param_values.items():
            if len(vals) < 4:
                continue
            sign_changes = 0
            prev_direction = None
            for i in range(1, len(vals)):
                delta = vals[i] - vals[i - 1]
                if delta == 0:
                    continue
                direction = 1 if delta > 0 else -1
                if prev_direction is not None and direction != prev_direction:
                    sign_changes += 1
                prev_direction = direction
            if sign_changes >= 3:
                oscillating.append(pname)

        return oscillating

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, n_cycles: int = 50) -> str:
        """
        Generate a Markdown summary report covering the last n_cycles.
        """
        recent = self._cycles[-n_cycles:]
        total = len(self._cycles)
        n = len(recent)

        lines: List[str] = [
            "# IAE Performance Report",
            "",
            f"**Total cycles recorded**: {total}",
            f"**Cycles in this report**: {n}",
            "",
        ]

        if n == 0:
            lines.append("No cycle data available.")
            return "\n".join(lines)

        # Fitness summary
        fitness_values = [c.best_fitness for c in recent]
        lines += [
            "## Fitness Summary",
            "",
            f"- Best fitness (latest) : {recent[-1].best_fitness:.6f}",
            f"- Best fitness (report) : {max(fitness_values):.6f}",
            f"- Mean fitness (report) : {float(np.mean(fitness_values)):.6f}",
            f"- Std fitness  (report) : {float(np.std(fitness_values)):.6f}",
            "",
        ]

        # Rolling improvement rate
        rir = self.rolling_improvement_rate(n_cycles=min(10, n))
        lines += [
            "## Adaptation Dynamics",
            "",
            f"- Rolling improvement rate (last 10 cycles): {rir:+.6f}",
            f"- Adaptation efficiency (total): {self.adaptation_efficiency():.8f} fitness/eval",
        ]

        tti = self.time_to_improvement(threshold=0.05)
        if tti is not None:
            lines.append(f"- Cycles to +0.05 fitness: {tti}")
        else:
            lines.append("- Cycles to +0.05 fitness: not yet reached")
        lines.append("")

        # Parameter update frequency
        freq = self.parameter_update_frequency()
        if freq:
            lines += ["## Parameter Update Frequency", ""]
            for pname, cnt in list(freq.items())[:10]:
                lines.append(f"- {pname}: {cnt} updates")
            lines.append("")

        # Sharpe impact
        sharpe_deltas = [c.sharpe_delta for c in recent]
        mean_sharpe_delta = float(np.mean(sharpe_deltas)) if sharpe_deltas else 0.0
        pos_cycles = sum(1 for d in sharpe_deltas if d > 0)
        lines += [
            "## Live Sharpe Impact",
            "",
            f"- Mean Sharpe delta per cycle: {mean_sharpe_delta:+.4f}",
            f"- Cycles with positive Sharpe delta: {pos_cycles}/{n}",
            "",
        ]

        # Oscillating parameters
        osc = self._detect_oscillating_params()
        if osc:
            lines += ["## Warnings", ""]
            lines.append(f"- Oscillating parameters detected: {', '.join(osc)}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AdaptationQualityMonitor
# ---------------------------------------------------------------------------

class AdaptationQualityMonitor:
    """
    Monitor IAE adaptation quality and raise warnings when something looks wrong.

    Warning conditions:
        1. No improvement in best_fitness over the last 20 cycles.
        2. Fitness variance is increasing across cycles (adaptation becoming noisier).
        3. One or more parameters are oscillating (changing direction repeatedly).
    """

    NO_IMPROVEMENT_WINDOW = 20
    OSCILLATION_WINDOW = 20

    def check_adaptation_quality(
        self,
        cycles: List[IAECycleResult],
    ) -> QualityReport:
        """
        Analyze the list of cycle results and return a QualityReport.
        """
        if not cycles:
            return QualityReport(
                status="ok",
                warnings=["No cycle data provided -- cannot assess quality."],
                recommendations=["Ensure the IAE is running and recording cycles."],
            )

        warnings: List[str] = []
        recommendations: List[str] = []
        metrics: Dict[str, float] = {}

        # -- Check 1: No improvement in last 20 cycles --
        window = min(self.NO_IMPROVEMENT_WINDOW, len(cycles))
        recent = cycles[-window:]
        best_in_window = max(c.best_fitness for c in recent)
        baseline = recent[0].best_fitness
        improvement = best_in_window - baseline
        metrics["improvement_last_20"] = improvement

        if improvement <= 0.0 and len(recent) >= self.NO_IMPROVEMENT_WINDOW:
            warnings.append(
                f"No fitness improvement detected in the last {self.NO_IMPROVEMENT_WINDOW} cycles "
                f"(baseline={baseline:.6f}, best_in_window={best_in_window:.6f})."
            )
            recommendations.append(
                "Consider increasing mutation rate or broadening the parameter search space."
            )
            recommendations.append(
                "Check that the fitness function is not saturated or returning constant values."
            )

        # -- Check 2: Fitness variance increasing --
        if len(cycles) >= 10:
            first_half = cycles[: len(cycles) // 2]
            second_half = cycles[len(cycles) // 2 :]
            var_first = float(np.var([c.best_fitness for c in first_half]))
            var_second = float(np.var([c.best_fitness for c in second_half]))
            metrics["fitness_var_first_half"] = var_first
            metrics["fitness_var_second_half"] = var_second
            var_ratio = var_second / var_first if var_first > 0 else 0.0
            metrics["fitness_var_ratio"] = var_ratio

            if var_ratio > 2.0:
                warnings.append(
                    f"Fitness variance increased by {var_ratio:.1f}x in the second half of recorded cycles. "
                    "This suggests the adaptation is becoming less stable."
                )
                recommendations.append(
                    "Reduce learning rate or tighten parameter mutation bounds to stabilize adaptation."
                )

        # -- Check 3: Oscillating parameters --
        osc_params = self._detect_oscillating_params(
            cycles, window=self.OSCILLATION_WINDOW
        )
        metrics["n_oscillating_params"] = float(len(osc_params))

        if osc_params:
            warnings.append(
                f"Parameters oscillating (alternating update direction) in last "
                f"{self.OSCILLATION_WINDOW} cycles: {', '.join(osc_params)}."
            )
            recommendations.append(
                "Oscillating parameters may indicate conflicting fitness signals.  "
                "Consider dampening updates for these parameters or reviewing the fitness function."
            )

        # -- Determine overall status --
        if len(warnings) == 0:
            status = "ok"
        elif len(warnings) <= 1 and improvement > 0:
            status = "warning"
        else:
            status = "critical"

        return QualityReport(
            status=status,
            warnings=warnings,
            recommendations=recommendations,
            metrics=metrics,
        )

    @staticmethod
    def _detect_oscillating_params(
        cycles: List[IAECycleResult],
        window: int = 20,
    ) -> List[str]:
        """
        Identify parameters that changed direction 3+ times in the last window cycles.
        """
        recent = cycles[-window:] if len(cycles) >= window else cycles
        param_vals: Dict[str, List[float]] = {}

        for c in recent:
            for pname, (old_v, new_v) in c.params_changed.items():
                param_vals.setdefault(pname, []).append(new_v)

        oscillating = []
        for pname, vals in param_vals.items():
            if len(vals) < 4:
                continue
            sign_changes = 0
            prev_dir: Optional[int] = None
            for i in range(1, len(vals)):
                delta = vals[i] - vals[i - 1]
                if abs(delta) < 1e-12:
                    continue
                direction = 1 if delta > 0 else -1
                if prev_dir is not None and direction != prev_dir:
                    sign_changes += 1
                prev_dir = direction
            if sign_changes >= 3:
                oscillating.append(pname)

        return sorted(oscillating)

    # ------------------------------------------------------------------
    # Historical quality trend
    # ------------------------------------------------------------------

    @staticmethod
    def quality_trend(
        cycles: List[IAECycleResult],
        chunk_size: int = 10,
    ) -> pd.DataFrame:
        """
        Break cycles into chunks of chunk_size and return per-chunk quality
        metrics as a DataFrame.  Useful for spotting when quality degraded.
        """
        rows = []
        for start in range(0, len(cycles), chunk_size):
            chunk = cycles[start : start + chunk_size]
            if not chunk:
                continue
            fitness_vals = [c.best_fitness for c in chunk]
            sharpe_deltas = [c.sharpe_delta for c in chunk]
            rows.append(
                {
                    "chunk_start": start,
                    "chunk_end": start + len(chunk) - 1,
                    "best_fitness": max(fitness_vals),
                    "mean_fitness": float(np.mean(fitness_vals)),
                    "std_fitness": float(np.std(fitness_vals)),
                    "mean_sharpe_delta": float(np.mean(sharpe_deltas)),
                    "positive_sharpe_pct": float(
                        sum(1 for d in sharpe_deltas if d > 0) / len(sharpe_deltas)
                    ),
                }
            )
        return pd.DataFrame(rows)
