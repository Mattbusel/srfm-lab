"""
ParameterSpace
==============
Defines bounds for all 15 genome parameters and provides sampling strategies:

  - ``latin_hypercube_sample``  — LHS for efficient space coverage
  - ``sobol_sample``            — quasi-random low-discrepancy sequences
  - ``neighborhood_sample``     — Gaussian perturbation around a center point
  - ``gradient_estimate``       — finite-difference gradient of improvement_score
  - ``steepest_ascent``         — follow gradient uphill through parameter space
  - ``clip``                    — project a param dict back inside bounds
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------

#: Canonical parameter bounds used across the whole idea-engine.
#: Each entry: param_name -> (low, high)
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    # BH physics
    "bh_form":           (1.70,  1.98),
    "bh_decay":          (0.92,  0.99),
    "bh_collapse":       (0.60,  0.95),
    "bh_ctl_min":        (1.0,   5.0),
    # Trade timing
    "min_hold_bars":     (1.0,   20.0),
    "stale_15m_move":    (0.001, 0.020),
    # Position sizing
    "delta_max_frac":    (0.10,  0.60),
    "corr_factor":       (0.15,  0.80),
    # GARCH / OU
    "garch_target_vol":  (0.60,  2.00),
    "ou_frac":           (0.02,  0.20),
    "pos_floor_scale":   (0.001, 0.050),
    # Regime-conditional cash-flow scales
    "cf_scale_bull":     (0.5,   2.0),
    "cf_scale_bear":     (0.5,   2.0),
    "cf_scale_neutral":  (0.5,   2.0),
    # Extra — total = 14 continuous + bh_ctl_min (treated as float)
    "pos_size_cap":      (0.01,  0.20),
}

# Integer parameters (rounded during sampling)
INTEGER_PARAMS: frozenset[str] = frozenset({"min_hold_bars", "bh_ctl_min"})


# ---------------------------------------------------------------------------
# Helper: Sobol sequence (base-2 Van der Corput / simple implementation)
# ---------------------------------------------------------------------------

def _van_der_corput(n: int, base: int = 2) -> np.ndarray:
    """Return n points of the Van der Corput sequence in [0, 1)."""
    seq = np.zeros(n)
    for i in range(n):
        q, denom = i + 1, 1
        while q > 0:
            denom *= base
            q, r = divmod(q, base)
            seq[i] += r / denom
    return seq


def _sobol_sequence(n: int, d: int) -> np.ndarray:
    """
    Very simple multi-dimensional quasi-random sequence using distinct prime bases.

    For production use, replace with ``scipy.stats.qmc.Sobol`` if scipy is
    available.  This implementation uses Van der Corput sequences with the
    first ``d`` primes as bases, which gives acceptable low-discrepancy
    properties up to ~20 dimensions.

    Returns shape (n, d) in [0, 1)^d.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if d > len(primes):
        # Fall back to random if dimension exceeds table
        rng = np.random.default_rng(seed=42)
        return rng.random((n, d))
    cols = [_van_der_corput(n, p) for p in primes[:d]]
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# ParameterSpace class
# ---------------------------------------------------------------------------

class ParameterSpace:
    """
    Encapsulates the genome parameter space and provides sampling utilities.

    Parameters
    ----------
    bounds : dict[str, tuple[float, float]] | None
        Override the default ``PARAM_BOUNDS``.  If ``None``, uses the module
        constant.
    seed : int | None
        Random seed for reproducibility (default 42).
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]] | None = None,
        seed: int | None = 42,
    ) -> None:
        self.bounds: dict[str, tuple[float, float]] = bounds or dict(PARAM_BOUNDS)
        self._param_names: list[str] = list(self.bounds.keys())
        self._rng = np.random.default_rng(seed=seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unit_to_params(self, unit: np.ndarray) -> dict[str, Any]:
        """
        Map a point in [0,1]^d to the actual parameter space.

        Parameters
        ----------
        unit : np.ndarray, shape (d,)

        Returns
        -------
        dict mapping param_name -> value
        """
        params: dict[str, Any] = {}
        for i, name in enumerate(self._param_names):
            lo, hi = self.bounds[name]
            val = float(lo + unit[i] * (hi - lo))
            if name in INTEGER_PARAMS:
                val = float(round(val))
            params[name] = val
        return params

    def _params_to_unit(self, params: dict[str, Any]) -> np.ndarray:
        """
        Map a param dict to the [0,1]^d unit hypercube.

        Missing params are placed at the midpoint (0.5).
        """
        unit = np.zeros(len(self._param_names))
        for i, name in enumerate(self._param_names):
            lo, hi = self.bounds[name]
            rng = hi - lo
            v = params.get(name, (lo + hi) / 2.0)
            unit[i] = (v - lo) / rng if rng > 0 else 0.5
        return np.clip(unit, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Latin Hypercube Sampling
    # ------------------------------------------------------------------

    def latin_hypercube_sample(self, n: int) -> list[dict[str, Any]]:
        """
        Generate ``n`` parameter sets using Latin Hypercube Sampling.

        LHS divides each dimension into ``n`` equal strata and places
        exactly one sample per stratum, then shuffles the strata assignments
        across dimensions independently.  This ensures better coverage than
        pure random sampling.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        list of param dicts
        """
        d = len(self._param_names)
        # Create stratified samples: one per stratum, jittered inside
        strata_low = np.arange(n) / n
        samples = np.zeros((n, d))
        for col in range(d):
            perm = self._rng.permutation(n)
            jitter = self._rng.uniform(0.0, 1.0 / n, size=n)
            samples[:, col] = strata_low[perm] + jitter

        return [self._unit_to_params(samples[i]) for i in range(n)]

    # ------------------------------------------------------------------
    # Sobol / quasi-random sampling
    # ------------------------------------------------------------------

    def sobol_sample(self, n: int) -> list[dict[str, Any]]:
        """
        Generate ``n`` parameter sets using a quasi-random low-discrepancy
        sequence (Van der Corput / Sobol approximation).

        If ``scipy`` is available, uses ``scipy.stats.qmc.Sobol`` for true
        Sobol sequences.  Falls back to the internal implementation otherwise.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        list of param dicts
        """
        d = len(self._param_names)
        try:
            from scipy.stats.qmc import Sobol  # type: ignore
            sampler = Sobol(d=d, scramble=True, seed=int(self._rng.integers(0, 2**31)))
            # Sobol requires power-of-two samples; round up and truncate
            n_pow2 = 2 ** math.ceil(math.log2(max(n, 1)))
            unit = sampler.random(n_pow2)[:n]
        except ImportError:
            warnings.warn(
                "scipy not available — using Van der Corput approximation for Sobol sampling.",
                ImportWarning,
                stacklevel=2,
            )
            unit = _sobol_sequence(n, d)

        return [self._unit_to_params(unit[i]) for i in range(n)]

    # ------------------------------------------------------------------
    # Neighborhood sampling
    # ------------------------------------------------------------------

    def neighborhood_sample(
        self,
        center_params: dict[str, Any],
        radius: float = 0.15,
        n: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Generate ``n`` parameter sets by perturbing ``center_params`` with
        Gaussian noise scaled to ``radius`` of each dimension's range.

        Parameters
        ----------
        center_params : dict
            The center of the neighborhood (e.g. a known-good genome).
        radius : float
            Standard deviation of perturbation as a fraction of each
            parameter's full range (default 0.15 = 15 %).
        n : int
            Number of samples.

        Returns
        -------
        list of param dicts clipped to valid bounds
        """
        center_unit = self._params_to_unit(center_params)
        d = len(self._param_names)
        samples = []
        for _ in range(n):
            noise = self._rng.normal(loc=0.0, scale=radius, size=d)
            candidate_unit = np.clip(center_unit + noise, 0.0, 1.0)
            samples.append(self._unit_to_params(candidate_unit))
        return samples

    # ------------------------------------------------------------------
    # Gradient estimation
    # ------------------------------------------------------------------

    def gradient_estimate(
        self,
        score_fn: Callable[[dict[str, Any]], float],
        params: dict[str, Any],
        epsilon: float = 0.05,
    ) -> dict[str, float]:
        """
        Estimate the finite-difference gradient of ``score_fn`` at ``params``.

        For each parameter *p*:

        .. math::
            \\nabla_p \\approx
            \\frac{f(p + \\epsilon \\cdot \\text{range}_p)
                  - f(p - \\epsilon \\cdot \\text{range}_p)}
                 {2 \\epsilon \\cdot \\text{range}_p}

        Parameters
        ----------
        score_fn : callable
            Maps a param dict to a scalar score.
        params : dict
            Point at which to evaluate the gradient.
        epsilon : float
            Step size as a fraction of each param's range (default 0.05).

        Returns
        -------
        dict mapping param_name -> gradient component
        """
        grad: dict[str, float] = {}
        for name in self._param_names:
            lo, hi = self.bounds[name]
            step = epsilon * (hi - lo)
            cur_val = params.get(name, (lo + hi) / 2.0)

            p_plus = copy.deepcopy(params)
            p_minus = copy.deepcopy(params)
            p_plus[name] = min(hi, cur_val + step)
            p_minus[name] = max(lo, cur_val - step)

            score_plus = score_fn(p_plus)
            score_minus = score_fn(p_minus)

            denom = 2.0 * step if step > 1e-12 else 1.0
            grad[name] = (score_plus - score_minus) / denom
        return grad

    # ------------------------------------------------------------------
    # Steepest ascent
    # ------------------------------------------------------------------

    def steepest_ascent(
        self,
        score_fn: Callable[[dict[str, Any]], float],
        start_params: dict[str, Any],
        steps: int = 10,
        step_size: float = 0.05,
        epsilon: float = 0.05,
        tolerance: float = 1e-4,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Follow the gradient of ``score_fn`` uphill from ``start_params``.

        Uses normalised gradient steps so that each step moves at most
        ``step_size`` of the bounding-box diagonal in any single dimension.

        Parameters
        ----------
        score_fn : callable
            Maps param dict -> scalar score.
        start_params : dict
            Starting point in parameter space.
        steps : int
            Maximum number of gradient steps (default 10).
        step_size : float
            Step magnitude as a fraction of each param's range (default 0.05).
        epsilon : float
            Finite-difference epsilon for gradient estimation.
        tolerance : float
            Stop early if improvement between consecutive steps < tolerance.

        Returns
        -------
        List of (params, score) tuples along the ascent path.
        """
        current = copy.deepcopy(start_params)
        current = self.clip(current)
        path: list[tuple[dict[str, Any], float]] = []
        prev_score: float | None = None

        for _ in range(steps):
            score = score_fn(current)
            path.append((copy.deepcopy(current), score))

            if prev_score is not None and abs(score - prev_score) < tolerance:
                break
            prev_score = score

            grad = self.gradient_estimate(score_fn, current, epsilon=epsilon)
            norm = math.sqrt(sum(g ** 2 for g in grad.values())) or 1.0

            next_params = copy.deepcopy(current)
            for name, g in grad.items():
                lo, hi = self.bounds[name]
                rng = hi - lo
                cur_val = current.get(name, (lo + hi) / 2.0)
                new_val = float(np.clip(cur_val + step_size * rng * g / norm, lo, hi))
                if name in INTEGER_PARAMS:
                    new_val = float(round(new_val))
                next_params[name] = new_val

            current = next_params

        return path

    # ------------------------------------------------------------------
    # Utility: clip params to valid bounds
    # ------------------------------------------------------------------

    def clip(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Project a param dict into the valid bounding box.

        Parameters not in ``self.bounds`` are passed through unchanged.

        Returns
        -------
        New dict with all bound params clipped to [lo, hi].
        """
        clipped = dict(params)
        for name, (lo, hi) in self.bounds.items():
            if name in clipped:
                val = float(np.clip(clipped[name], lo, hi))
                if name in INTEGER_PARAMS:
                    val = float(round(val))
                clipped[name] = val
        return clipped

    # ------------------------------------------------------------------
    # Utility: random single sample
    # ------------------------------------------------------------------

    def random_sample(self) -> dict[str, Any]:
        """Return a single uniformly random sample from the parameter space."""
        unit = self._rng.uniform(0.0, 1.0, size=len(self._param_names))
        return self._unit_to_params(unit)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ParameterSpace(n_params={len(self._param_names)}, "
            f"params={self._param_names})"
        )

    # ------------------------------------------------------------------
    # Dimensionality info
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Number of parameters in the space."""
        return len(self._param_names)

    @property
    def param_names(self) -> list[str]:
        """Ordered list of parameter names."""
        return list(self._param_names)

    def range_of(self, param: str) -> float:
        """Return hi − lo for a named parameter."""
        lo, hi = self.bounds[param]
        return hi - lo

    def midpoint(self) -> dict[str, Any]:
        """Return the midpoint of every dimension."""
        return {name: (lo + hi) / 2.0 for name, (lo, hi) in self.bounds.items()}

    def hypercube_volume(self) -> float:
        """Return the product of all ranges (unnormalised volume)."""
        vol = 1.0
        for lo, hi in self.bounds.values():
            vol *= (hi - lo)
        return vol
