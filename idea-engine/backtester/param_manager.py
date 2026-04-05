"""
ParamManager
============
Manages the genome parameter set for backtests.

Provides:
  - BASELINE_PARAMS dict — current production parameters.
  - apply_delta — merge a delta dict onto baseline.
  - validate_params — check parameter bounds.
  - diff_params — show what changed between two param sets.
  - hash_params — deterministic hash for deduplication.
  - canonical_params — sorted, rounded for comparison.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline parameters — current production values
# ---------------------------------------------------------------------------

BASELINE_PARAMS: dict[str, float] = {
    # BH physics
    "bh_form":             1.92,
    "bh_decay":            0.97,
    "bh_collapse":         0.75,
    "bh_ctl_min":          2.0,
    # Trade timing
    "min_hold_bars":       4.0,
    "stale_15m_move":      0.005,
    # Position sizing
    "delta_max_frac":      0.40,
    "corr_factor":         0.25,
    # GARCH / OU
    "garch_target_vol":    1.20,
    "ou_frac":             0.08,
    # Regime-conditional cash-flow scales
    "cf_scale_bull":       1.0,
    "cf_scale_bear":       0.6,
    "cf_scale_neutral":    0.8,
    # Position floor
    "pos_floor_scale":     0.01,
    # Position size cap (additional safety rail)
    "pos_size_cap":        0.10,
}

# ---------------------------------------------------------------------------
# Parameter bounds — valid ranges for each parameter
# ---------------------------------------------------------------------------

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "bh_form":           (1.70,  1.98),
    "bh_decay":          (0.92,  0.99),
    "bh_collapse":       (0.60,  0.95),
    "bh_ctl_min":        (1.0,   5.0),
    "min_hold_bars":     (1.0,   20.0),
    "stale_15m_move":    (0.001, 0.020),
    "delta_max_frac":    (0.10,  0.60),
    "corr_factor":       (0.15,  0.80),
    "garch_target_vol":  (0.60,  2.00),
    "ou_frac":           (0.02,  0.20),
    "pos_floor_scale":   (0.001, 0.050),
    "cf_scale_bull":     (0.5,   2.0),
    "cf_scale_bear":     (0.5,   2.0),
    "cf_scale_neutral":  (0.5,   2.0),
    "pos_size_cap":      (0.01,  0.20),
}

# Parameters that should be treated as integers
INTEGER_PARAMS: frozenset[str] = frozenset({"min_hold_bars", "bh_ctl_min"})

# Precision (decimal places) for rounding in canonical_params
CANONICAL_PRECISION: dict[str, int] = {
    "bh_form":           4,
    "bh_decay":          4,
    "bh_collapse":       4,
    "bh_ctl_min":        1,
    "min_hold_bars":     1,
    "stale_15m_move":    5,
    "delta_max_frac":    4,
    "corr_factor":       4,
    "garch_target_vol":  4,
    "ou_frac":           5,
    "pos_floor_scale":   5,
    "cf_scale_bull":     4,
    "cf_scale_bear":     4,
    "cf_scale_neutral":  4,
    "pos_size_cap":      4,
}


# ---------------------------------------------------------------------------
# ParamManager
# ---------------------------------------------------------------------------

class ParamManager:
    """
    Central manager for backtest parameter sets.

    All methods are pure-functional — they return new dicts rather than
    modifying in-place, so callers can chain transformations safely.
    """

    def __init__(
        self,
        baseline: dict[str, float] | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._baseline = dict(baseline or BASELINE_PARAMS)
        self._bounds = dict(bounds or PARAM_BOUNDS)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get_baseline(self) -> dict[str, float]:
        """Return a copy of the baseline parameter dict."""
        return copy.deepcopy(self._baseline)

    def apply_delta(
        self,
        baseline: dict[str, Any],
        delta: dict[str, Any],
    ) -> dict[str, float]:
        """
        Merge *delta* onto *baseline*, returning a new dict.

        Delta keys override baseline values.  Integer parameters are rounded.
        Values are clipped to their bounds if bounds are defined.

        Parameters
        ----------
        baseline : the base parameter dict (e.g. from get_baseline()).
        delta    : the changes to apply (hypothesis param_delta from DB).

        Returns
        -------
        dict : merged, clipped, rounded parameter dict.
        """
        merged: dict[str, float] = copy.deepcopy(baseline)
        for key, value in delta.items():
            if not isinstance(value, (int, float)):
                logger.warning("Non-numeric delta for '%s': %r — skipped.", key, value)
                continue
            merged[key] = float(value)

        # Round integer params
        for k in INTEGER_PARAMS:
            if k in merged:
                merged[k] = float(round(merged[k]))

        # Clip to bounds
        for k, v in merged.items():
            if k in self._bounds:
                lo, hi = self._bounds[k]
                if v < lo:
                    logger.debug("Clipping %s from %.4f to %.4f (lower bound)", k, v, lo)
                    merged[k] = lo
                elif v > hi:
                    logger.debug("Clipping %s from %.4f to %.4f (upper bound)", k, v, hi)
                    merged[k] = hi

        return merged

    def validate_params(
        self,
        params: dict[str, Any],
    ) -> list[str]:
        """
        Validate a parameter dict against the known bounds.

        Returns a list of error strings (empty if valid).
        """
        errors: list[str] = []

        for key, (lo, hi) in self._bounds.items():
            if key not in params:
                errors.append(f"Missing required parameter: '{key}'")
                continue
            val = params[key]
            try:
                val = float(val)
            except (TypeError, ValueError):
                errors.append(f"Parameter '{key}' is not numeric: {val!r}")
                continue
            if not math.isfinite(val):
                errors.append(f"Parameter '{key}' is not finite: {val}")
                continue
            if val < lo or val > hi:
                errors.append(
                    f"Parameter '{key}' = {val} is out of bounds [{lo}, {hi}]"
                )

        # Check for unknown parameters (informational only)
        known = set(self._bounds) | set(BASELINE_PARAMS)
        for key in params:
            if key not in known:
                errors.append(f"Unknown parameter '{key}' (no bounds defined).")

        return errors

    def diff_params(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """
        Compute the diff from param set *a* to param set *b*.

        Returns a dict of changed keys with structure:
            {param_name: {"from": old_value, "to": new_value, "delta": change}}
        """
        all_keys = set(a) | set(b)
        diff: dict[str, dict[str, Any]] = {}

        for key in sorted(all_keys):
            old = a.get(key)
            new = b.get(key)
            if old is None and new is not None:
                diff[key] = {"from": None, "to": new, "delta": None, "change": "added"}
            elif old is not None and new is None:
                diff[key] = {"from": old, "to": None, "delta": None, "change": "removed"}
            elif old != new:
                try:
                    delta = float(new) - float(old)
                    pct = (delta / float(old) * 100.0) if float(old) != 0 else float("inf")
                    diff[key] = {
                        "from": old,
                        "to": new,
                        "delta": round(delta, 6),
                        "pct_change": round(pct, 2),
                        "change": "modified",
                    }
                except (TypeError, ValueError):
                    diff[key] = {"from": old, "to": new, "delta": None, "change": "modified"}

        return diff

    def hash_params(self, params: dict[str, Any]) -> str:
        """
        Return a deterministic SHA-256 hex digest of *params*.

        The hash is stable: keys are sorted and values rounded to the canonical
        precision before serialisation.
        """
        canonical = self.canonical_params(params)
        serialised = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]

    def canonical_params(self, params: dict[str, Any]) -> dict[str, float]:
        """
        Return a sorted, rounded copy of *params* suitable for comparison and
        hashing.

        Parameters not in CANONICAL_PRECISION get 6 decimal places.
        """
        result: dict[str, float] = {}
        for key in sorted(params):
            val = params[key]
            try:
                precision = CANONICAL_PRECISION.get(key, 6)
                result[key] = round(float(val), precision)
            except (TypeError, ValueError):
                result[key] = val  # type: ignore[assignment]
        return result

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, json_str: str) -> "ParamManager":
        """Create a ParamManager with baseline loaded from a JSON string."""
        data = json.loads(json_str)
        return cls(baseline=data)

    @classmethod
    def from_file(cls, path: str | Path) -> "ParamManager":
        """Create a ParamManager with baseline loaded from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(baseline=data)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def enumerate_neighbors(
        self,
        center: dict[str, Any],
        step_pct: float = 0.05,
        params_to_vary: list[str] | None = None,
    ) -> list[dict[str, float]]:
        """
        Generate parameter sets that vary each parameter in *params_to_vary*
        by ±step_pct of its current value.

        Returns a list of param dicts (2 × len(params_to_vary) in total).
        """
        keys = params_to_vary or [k for k in center if k in self._bounds]
        variants: list[dict[str, float]] = []

        for key in keys:
            if key not in center:
                continue
            val = float(center[key])
            step = abs(val) * step_pct or step_pct  # avoid zero step

            for sign in (-1, +1):
                delta = {key: val + sign * step}
                merged = self.apply_delta(center, delta)
                variants.append(merged)

        return variants

    def random_perturbation(
        self,
        center: dict[str, Any],
        scale: float = 0.10,
        seed: int | None = None,
    ) -> dict[str, float]:
        """
        Return a random perturbation of *center* where each bounded parameter
        is shifted by a Gaussian draw with std = *scale* × (hi - lo).

        Parameters
        ----------
        center : base parameter dict.
        scale  : fraction of parameter range used as noise std (default 0.10).
        seed   : RNG seed for reproducibility.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        delta: dict[str, Any] = {}
        for key in center:
            if key not in self._bounds:
                continue
            lo, hi = self._bounds[key]
            noise = rng.normal(0.0, scale * (hi - lo))
            delta[key] = float(center[key]) + noise

        return self.apply_delta(center, delta)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def param_summary(self, params: dict[str, Any]) -> str:
        """Return a compact string representation of *params*."""
        lines = ["Parameter set:"]
        for key in sorted(params):
            val = params[key]
            baseline_val = self._baseline.get(key)
            if baseline_val is not None:
                delta = float(val) - float(baseline_val)
                indicator = f" (Δ{delta:+.4f})" if abs(delta) > 1e-8 else ""
            else:
                indicator = " (new)"
            lines.append(f"  {key:<22} = {val!r}{indicator}")
        return "\n".join(lines)
