"""
research/param_explorer/sensitivity.py
=======================================
Sensitivity analysis tools: OAT, global Sobol, Morris screening,
gradient / Hessian analysis, and associated plot helpers.

Classes
-------
SensitivityAnalyzer : Unified interface for all methods
OATResult           : One-At-a-Time experiment results
SobolResult         : Variance-based Sobol sensitivity indices
MorrisResult        : Morris elementary effects screening results

Stand-alone functions
---------------------
one_at_a_time       : OAT experiment
global_sensitivity_sobol : Full Saltelli scheme
morris_screening    : Morris trajectory method
gradient_sensitivity : Finite-difference gradient
hessian             : Finite-difference Hessian matrix
eigenvalue_analysis : Eigendecomposition of Hessian
plot_oat_curves     : Matplotlib figure for OAT results
plot_sobol_indices  : Bar chart of Si / STi indices
plot_morris_mu_star_sigma : Morris star chart
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm as scipy_norm

from research.param_explorer.space import ParamSpace, ParamSpec, ParamType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OATResult:
    """
    Results of a one-at-a-time (OAT) sensitivity experiment.

    Attributes
    ----------
    base_params : dict
        Parameter dict used as the fixed baseline.
    param_names : list[str]
        Parameters that were varied.
    values : dict[str, np.ndarray]
        Actual parameter values sampled for each varied parameter.
    objectives : dict[str, np.ndarray]
        Objective function values for each varied parameter.
    sensitivity_range : dict[str, float]
        Range of objective values (max - min) for each parameter.
    sensitivity_std : dict[str, float]
        Standard deviation of objective over the sweep.
    sensitivity_rank : dict[str, int]
        Rank of importance (1 = most sensitive).
    n_points : int
        Number of evaluation points per parameter.
    total_evals : int
        Total number of objective function evaluations.
    """

    base_params: Dict[str, Any]
    param_names: List[str]
    values: Dict[str, np.ndarray]
    objectives: Dict[str, np.ndarray]
    sensitivity_range: Dict[str, float]
    sensitivity_std: Dict[str, float]
    sensitivity_rank: Dict[str, int]
    n_points: int
    total_evals: int

    def top_k(self, k: int) -> List[Tuple[str, float]]:
        """Return top *k* parameters by sensitivity range, sorted descending."""
        ranked = sorted(
            self.sensitivity_range.items(), key=lambda kv: kv[1], reverse=True
        )
        return ranked[:k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_params": self.base_params,
            "param_names": self.param_names,
            "sensitivity_range": self.sensitivity_range,
            "sensitivity_std": self.sensitivity_std,
            "sensitivity_rank": self.sensitivity_rank,
            "n_points": self.n_points,
            "total_evals": self.total_evals,
        }


@dataclass
class SobolResult:
    """
    Variance-based Sobol global sensitivity indices.

    Attributes
    ----------
    Si : dict[str, float]
        First-order indices (main effects).
    STi : dict[str, float]
        Total-order indices (main + all interactions).
    Si_conf : dict[str, float]
        95 % confidence half-widths for Si.
    STi_conf : dict[str, float]
        95 % confidence half-widths for STi.
    var_y : float
        Estimated total variance of the objective.
    n_samples : int
        Base sample size N (total evals ≈ N*(2d+2)).
    total_evals : int
        Actual total evaluations.
    """

    Si: Dict[str, float]
    STi: Dict[str, float]
    Si_conf: Dict[str, float]
    STi_conf: Dict[str, float]
    var_y: float
    n_samples: int
    total_evals: int

    def top_k(self, k: int, index: str = "STi") -> List[Tuple[str, float]]:
        """Return top *k* by STi (or Si if index='Si')."""
        src = self.STi if index == "STi" else self.Si
        return sorted(src.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Si": self.Si,
            "STi": self.STi,
            "Si_conf": self.Si_conf,
            "STi_conf": self.STi_conf,
            "var_y": self.var_y,
            "n_samples": self.n_samples,
            "total_evals": self.total_evals,
        }


@dataclass
class MorrisResult:
    """
    Morris elementary effects screening results.

    Attributes
    ----------
    mu : dict[str, float]
        Mean of elementary effects.
    mu_star : dict[str, float]
        Mean of absolute elementary effects (μ*).
    sigma : dict[str, float]
        Standard deviation of elementary effects.
    sensitivity_rank : dict[str, int]
        Rank by μ* (1 = most influential).
    n_trajectories : int
    n_levels : int
    total_evals : int
    """

    mu: Dict[str, float]
    mu_star: Dict[str, float]
    sigma: Dict[str, float]
    sensitivity_rank: Dict[str, int]
    n_trajectories: int
    n_levels: int
    total_evals: int

    def top_k(self, k: int) -> List[Tuple[str, float]]:
        return sorted(self.mu_star.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mu": self.mu,
            "mu_star": self.mu_star,
            "sigma": self.sigma,
            "sensitivity_rank": self.sensitivity_rank,
            "n_trajectories": self.n_trajectories,
            "n_levels": self.n_levels,
            "total_evals": self.total_evals,
        }


# ---------------------------------------------------------------------------
# OAT
# ---------------------------------------------------------------------------

def one_at_a_time(
    base_params: Dict[str, Any],
    param_ranges: Dict[str, Tuple[float, float]],
    objective_fn: Callable[[Dict[str, Any]], float],
    n_points: int = 20,
    log_scale_params: Optional[List[str]] = None,
) -> OATResult:
    """
    One-at-a-time (OAT) sensitivity experiment.

    For each parameter in *param_ranges*, vary it uniformly (or log-uniformly)
    over its range while holding all other parameters at their values in
    *base_params*.

    Parameters
    ----------
    base_params : dict
        Baseline / nominal parameter values.  All parameters in
        *param_ranges* must appear here.
    param_ranges : dict
        Mapping ``param_name → (low, high)`` for each parameter to vary.
    objective_fn : callable
        Maps a parameter dict → scalar float.
    n_points : int
        Number of evaluation points per sweep.
    log_scale_params : list[str] | None
        Parameters that should be swept on a log scale.

    Returns
    -------
    OATResult
    """
    if log_scale_params is None:
        log_scale_params = []

    param_names = list(param_ranges.keys())
    values: Dict[str, np.ndarray] = {}
    objectives: Dict[str, np.ndarray] = {}
    total_evals = 0

    for pname in param_names:
        low, high = param_ranges[pname]
        if pname in log_scale_params and low > 0:
            grid = np.exp(np.linspace(math.log(low), math.log(high), n_points))
        else:
            grid = np.linspace(low, high, n_points)
        values[pname] = grid

        y_vals = np.zeros(n_points)
        for k, v in enumerate(grid):
            p = dict(base_params)
            p[pname] = v
            y_vals[k] = objective_fn(p)
            total_evals += 1

        objectives[pname] = y_vals

    sensitivity_range = {
        pname: float(np.max(objectives[pname]) - np.min(objectives[pname]))
        for pname in param_names
    }
    sensitivity_std = {
        pname: float(np.std(objectives[pname], ddof=1))
        for pname in param_names
    }

    # Rank by range descending
    ranked = sorted(param_names, key=lambda n: sensitivity_range[n], reverse=True)
    sensitivity_rank = {name: rank + 1 for rank, name in enumerate(ranked)}

    return OATResult(
        base_params=dict(base_params),
        param_names=param_names,
        values=values,
        objectives=objectives,
        sensitivity_range=sensitivity_range,
        sensitivity_std=sensitivity_std,
        sensitivity_rank=sensitivity_rank,
        n_points=n_points,
        total_evals=total_evals,
    )


# ---------------------------------------------------------------------------
# Global Sobol
# ---------------------------------------------------------------------------

def global_sensitivity_sobol(
    param_space: ParamSpace,
    objective_fn: Callable[[Dict[str, Any]], float],
    n_samples: int = 2048,
    seed: int = 42,
    conf_level: float = 0.95,
    n_bootstrap: int = 300,
) -> SobolResult:
    """
    Saltelli (2010) variance-based global sensitivity analysis.

    Constructs A and B sample matrices (each of size *n_samples*×d), plus d
    "C" matrices where column j of A is replaced by column j of B.  Total
    evaluations ≈ n_samples × (d + 2).

    The first-order index Si captures the fraction of variance explained by
    parameter i alone; the total-order index STi captures all effects
    involving parameter i, including interactions.

    Parameters
    ----------
    param_space : ParamSpace
    objective_fn : callable
    n_samples : int
        Base N.  Powers of 2 give optimal Sobol properties.
    seed : int
    conf_level : float
    n_bootstrap : int
        Number of bootstrap replicates for CI estimation.

    Returns
    -------
    SobolResult
    """
    d = param_space.n_dims
    rng = np.random.default_rng(seed)

    # Sample A and B using Sobol sequences with different scramble seeds
    A = param_space.sample_sobol(n_samples, seed=seed)
    B = param_space.sample_sobol(n_samples, seed=seed + 1000)

    logger.info("Evaluating A matrix (%d samples)…", n_samples)
    y_A = np.array([objective_fn(param_space.to_params(A[i])) for i in range(n_samples)])

    logger.info("Evaluating B matrix (%d samples)…", n_samples)
    y_B = np.array([objective_fn(param_space.to_params(B[i])) for i in range(n_samples)])

    # Build C_j matrices and evaluate
    y_C: Dict[int, np.ndarray] = {}
    for j in range(d):
        C_j = A.copy()
        C_j[:, j] = B[:, j]
        logger.debug("Evaluating C_%d matrix…", j)
        y_C[j] = np.array(
            [objective_fn(param_space.to_params(C_j[i])) for i in range(n_samples)]
        )

    # Total variance estimate
    y_all = np.concatenate([y_A, y_B])
    var_y = float(np.var(y_all, ddof=1))
    if var_y < 1e-30:
        logger.warning("Near-zero output variance; Sobol indices will be unreliable.")
        var_y = 1e-30

    # Saltelli 2010 estimators
    # Si  = (1/N) Σ y_B * (y_C_j - y_A) / Var(Y)
    # STi = (1/2N) Σ (y_A - y_C_j)^2  / Var(Y)
    Si: Dict[str, float] = {}
    STi: Dict[str, float] = {}
    Si_conf: Dict[str, float] = {}
    STi_conf: Dict[str, float] = {}

    boot_idx = rng.integers(0, n_samples, size=(n_bootstrap, n_samples))

    for j, spec in enumerate(param_space.specs):
        yj = y_C[j]
        si_sample = y_B * (yj - y_A) / var_y
        sti_sample = (y_A - yj) ** 2 / (2.0 * var_y)

        Si[spec.name] = float(np.mean(si_sample))
        STi[spec.name] = float(np.mean(sti_sample))

        si_boot = np.array([np.mean(si_sample[boot_idx[b]]) for b in range(n_bootstrap)])
        sti_boot = np.array([np.mean(sti_sample[boot_idx[b]]) for b in range(n_bootstrap)])
        alpha = 1.0 - conf_level
        Si_conf[spec.name] = float(
            (np.percentile(si_boot, 100 * (1 - alpha / 2))
             - np.percentile(si_boot, 100 * (alpha / 2))) / 2.0
        )
        STi_conf[spec.name] = float(
            (np.percentile(sti_boot, 100 * (1 - alpha / 2))
             - np.percentile(sti_boot, 100 * (alpha / 2))) / 2.0
        )

    total_evals = n_samples * (d + 2)
    return SobolResult(
        Si=Si,
        STi=STi,
        Si_conf=Si_conf,
        STi_conf=STi_conf,
        var_y=var_y,
        n_samples=n_samples,
        total_evals=total_evals,
    )


# ---------------------------------------------------------------------------
# Morris screening
# ---------------------------------------------------------------------------

def morris_screening(
    param_space: ParamSpace,
    objective_fn: Callable[[Dict[str, Any]], float],
    n_trajectories: int = 10,
    n_levels: int = 4,
    seed: int = 0,
) -> MorrisResult:
    """
    Morris (1991) elementary effects screening method.

    Constructs *n_trajectories* one-step-at-a-time trajectories in the
    parameter space.  Each trajectory has d+1 evaluation points, so total
    evaluations = n_trajectories × (d+1).

    The elementary effect for parameter i along trajectory r is:
        EE_i^r = [y(x + Δe_i) - y(x)] / Δ
    where Δ = p / (2*(p-1)) and p = n_levels.

    Parameters
    ----------
    param_space : ParamSpace
    objective_fn : callable
    n_trajectories : int
    n_levels : int
        Number of grid levels.  Typically 4 or 6.
    seed : int

    Returns
    -------
    MorrisResult
    """
    d = param_space.n_dims
    rng = np.random.default_rng(seed)
    p = n_levels
    delta = p / (2.0 * (p - 1.0))

    # Grid levels: {0, 1/(p-1), 2/(p-1), …, 1}
    levels = np.linspace(0.0, 1.0, p)

    all_effects: Dict[int, List[float]] = {j: [] for j in range(d)}

    for _ in range(n_trajectories):
        # Random starting point on the level grid
        x_star = np.array([rng.choice(levels) for _ in range(d)])
        # Clip to [0, 1-delta]
        x_star = np.clip(x_star, 0.0, 1.0 - delta)

        # Random permutation of parameter indices
        perm = rng.permutation(d)

        x = x_star.copy()
        y_prev = objective_fn(param_space.to_params(x))

        for j in perm:
            # Move in +delta or -delta direction with equal probability
            direction = rng.choice([-1, 1])
            x_new = x.copy()
            candidate = x[j] + direction * delta

            # Ensure we stay within [0, 1]
            if candidate < 0 or candidate > 1:
                direction = -direction
                candidate = x[j] + direction * delta

            x_new[j] = np.clip(candidate, 0.0, 1.0)
            y_new = objective_fn(param_space.to_params(x_new))

            ee = (y_new - y_prev) / (direction * delta)
            all_effects[j].append(ee)

            y_prev = y_new
            x = x_new

    mu: Dict[str, float] = {}
    mu_star: Dict[str, float] = {}
    sigma: Dict[str, float] = {}

    for j, spec in enumerate(param_space.specs):
        effects = np.array(all_effects[j])
        mu[spec.name] = float(np.mean(effects))
        mu_star[spec.name] = float(np.mean(np.abs(effects)))
        sigma[spec.name] = float(np.std(effects, ddof=1) if len(effects) > 1 else 0.0)

    ranked = sorted(param_space.names, key=lambda n: mu_star[n], reverse=True)
    sensitivity_rank = {name: rank + 1 for rank, name in enumerate(ranked)}

    total_evals = n_trajectories * (d + 1)
    return MorrisResult(
        mu=mu,
        mu_star=mu_star,
        sigma=sigma,
        sensitivity_rank=sensitivity_rank,
        n_trajectories=n_trajectories,
        n_levels=n_levels,
        total_evals=total_evals,
    )


# ---------------------------------------------------------------------------
# Gradient / Hessian
# ---------------------------------------------------------------------------

def gradient_sensitivity(
    params: Dict[str, Any],
    objective_fn: Callable[[Dict[str, Any]], float],
    param_space: Optional[ParamSpace] = None,
    epsilon: float = 1e-4,
) -> Dict[str, float]:
    """
    Finite-difference gradient of the objective with respect to each parameter.

    Uses central differences:  df/dx_i ≈ [f(x+ε·e_i) - f(x-ε·e_i)] / (2ε).

    Parameters
    ----------
    params : dict
        Point at which to compute the gradient.
    objective_fn : callable
    param_space : ParamSpace | None
        If provided, perturbations are in unit space and rescaled.
    epsilon : float
        Step size (in actual parameter space if param_space is None, else unit space).

    Returns
    -------
    dict mapping parameter name → partial derivative estimate.
    """
    f0 = objective_fn(params)
    grad: Dict[str, float] = {}

    param_names = list(params.keys())

    for name in param_names:
        v0 = params[name]

        if param_space is not None and name in param_space:
            spec = param_space[name]
            u0 = spec._value_to_unit(float(v0))
            u_plus = np.clip(u0 + epsilon, 0.0, 1.0)
            u_minus = np.clip(u0 - epsilon, 0.0, 1.0)
            v_plus = spec._unit_to_value(u_plus)
            v_minus = spec._unit_to_value(u_minus)
        else:
            try:
                v0_f = float(v0)
                v_plus = v0_f + epsilon
                v_minus = v0_f - epsilon
            except (TypeError, ValueError):
                grad[name] = 0.0
                continue

        p_plus = dict(params)
        p_minus = dict(params)
        p_plus[name] = v_plus
        p_minus[name] = v_minus

        f_plus = objective_fn(p_plus)
        f_minus = objective_fn(p_minus)

        actual_step = float(v_plus) - float(v_minus)
        if abs(actual_step) < 1e-30:
            grad[name] = 0.0
        else:
            grad[name] = (f_plus - f_minus) / actual_step

    return grad


def hessian(
    params: Dict[str, Any],
    objective_fn: Callable[[Dict[str, Any]], float],
    param_space: Optional[ParamSpace] = None,
    epsilon: float = 1e-4,
) -> Tuple[List[str], np.ndarray]:
    """
    Finite-difference Hessian matrix of the objective.

    Uses the mixed second-order formula for off-diagonal and the standard
    second central difference for diagonal:
        d²f/dx_i² ≈ [f(x+ε) - 2f(x) + f(x-ε)] / ε²
        d²f/dx_i dx_j ≈ [f(x+εi+εj) - f(x+εi-εj) - f(x-εi+εj) + f(x-εi-εj)] / (4ε²)

    Parameters
    ----------
    params : dict
    objective_fn : callable
    param_space : ParamSpace | None
    epsilon : float

    Returns
    -------
    (names, H) where names is the ordered list of parameter names and
    H is the (d × d) Hessian matrix.
    """
    names = [n for n in params.keys() if _is_numeric(params[n])]
    d = len(names)
    f0 = objective_fn(params)

    # Compute delta in actual space
    def _perturb(name: str, sign: float) -> float:
        """Return perturbed value for *name* with +/- epsilon in unit space."""
        v0 = params[name]
        if param_space is not None and name in param_space:
            spec = param_space[name]
            u0 = spec._value_to_unit(float(v0))
            u_new = np.clip(u0 + sign * epsilon, 0.0, 1.0)
            return float(spec._unit_to_value(u_new))
        return float(v0) + sign * epsilon

    def _eval_at(**overrides: float) -> float:
        p = dict(params)
        p.update(overrides)
        return objective_fn(p)

    H = np.zeros((d, d))
    actual_eps: Dict[str, float] = {}

    for i, ni in enumerate(names):
        vp = _perturb(ni, +1)
        vm = _perturb(ni, -1)
        actual_eps[ni] = (vp - float(params[ni])) or epsilon

        # Diagonal
        H[i, i] = (
            _eval_at(**{ni: vp}) - 2.0 * f0 + _eval_at(**{ni: vm})
        ) / (actual_eps[ni] ** 2)

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if j <= i:
                continue
            vip = _perturb(ni, +1)
            vim = _perturb(ni, -1)
            vjp = _perturb(nj, +1)
            vjm = _perturb(nj, -1)

            f_pp = _eval_at(**{ni: vip, nj: vjp})
            f_pm = _eval_at(**{ni: vip, nj: vjm})
            f_mp = _eval_at(**{ni: vim, nj: vjp})
            f_mm = _eval_at(**{ni: vim, nj: vjm})

            eps_i = actual_eps[ni]
            eps_j = actual_eps.get(nj, epsilon)
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps_i * eps_j)
            H[j, i] = H[i, j]

    return names, H


def _is_numeric(v: Any) -> bool:
    """Return True if *v* can be treated as a float."""
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def eigenvalue_analysis(
    hessian_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Eigendecomposition of a symmetric Hessian matrix.

    Parameters
    ----------
    hessian_matrix : np.ndarray of shape (d, d)

    Returns
    -------
    (eigenvalues, eigenvectors, condition_number)
        eigenvalues : np.ndarray sorted descending by absolute value
        eigenvectors : np.ndarray (columns are eigenvectors)
        condition_number : float  λ_max / λ_min (or inf if λ_min ≈ 0)
    """
    # Symmetrise to mitigate numerical noise
    H = (hessian_matrix + hessian_matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Sort descending by absolute value
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    abs_vals = np.abs(eigenvalues)
    lam_max = abs_vals.max() if len(abs_vals) > 0 else 1.0
    lam_min = abs_vals[abs_vals > 1e-12].min() if np.any(abs_vals > 1e-12) else 0.0
    condition_number = float(lam_max / lam_min) if lam_min > 0 else float("inf")

    return eigenvalues, eigenvectors, condition_number


# ---------------------------------------------------------------------------
# SensitivityAnalyzer class
# ---------------------------------------------------------------------------

class SensitivityAnalyzer:
    """
    Unified sensitivity analysis interface for a given parameter space and
    objective function.

    Parameters
    ----------
    param_space : ParamSpace
    objective_fn : callable
        Maps a parameter dict → scalar float.
    verbose : bool
        If True, log progress messages.
    """

    def __init__(
        self,
        param_space: ParamSpace,
        objective_fn: Callable[[Dict[str, Any]], float],
        verbose: bool = True,
    ) -> None:
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.verbose = verbose

        self._oat_result: Optional[OATResult] = None
        self._sobol_result: Optional[SobolResult] = None
        self._morris_result: Optional[MorrisResult] = None

    # ------------------------------------------------------------------
    # Core analysis methods
    # ------------------------------------------------------------------

    def run_oat(
        self,
        base_params: Optional[Dict[str, Any]] = None,
        n_points: int = 20,
    ) -> OATResult:
        """
        Run a One-At-a-Time sensitivity experiment over the full space.

        Parameters
        ----------
        base_params : dict | None
            Baseline parameter dict.  Uses space defaults if None.
        n_points : int

        Returns
        -------
        OATResult
        """
        if base_params is None:
            base_params = self.param_space.defaults

        param_ranges = {
            s.name: (s.low, s.high)
            for s in self.param_space.specs
            if s.param_type != ParamType.CATEGORICAL
        }
        log_scale_params = [
            s.name for s in self.param_space.specs if s.log_scale
        ]

        if self.verbose:
            logger.info(
                "Running OAT over %d parameters (%d points each)…",
                len(param_ranges), n_points,
            )

        self._oat_result = one_at_a_time(
            base_params=base_params,
            param_ranges=param_ranges,
            objective_fn=self.objective_fn,
            n_points=n_points,
            log_scale_params=log_scale_params,
        )
        return self._oat_result

    def run_sobol(
        self,
        n_samples: int = 2048,
        seed: int = 42,
    ) -> SobolResult:
        """
        Run global Saltelli Sobol sensitivity analysis.

        Parameters
        ----------
        n_samples : int
        seed : int

        Returns
        -------
        SobolResult
        """
        if self.verbose:
            d = self.param_space.n_dims
            logger.info(
                "Running Sobol SA: N=%d, d=%d → ~%d evaluations…",
                n_samples, d, n_samples * (d + 2),
            )

        self._sobol_result = global_sensitivity_sobol(
            self.param_space, self.objective_fn, n_samples=n_samples, seed=seed
        )
        return self._sobol_result

    def run_morris(
        self,
        n_trajectories: int = 10,
        n_levels: int = 4,
        seed: int = 0,
    ) -> MorrisResult:
        """
        Run Morris elementary effects screening.

        Parameters
        ----------
        n_trajectories : int
        n_levels : int
        seed : int

        Returns
        -------
        MorrisResult
        """
        if self.verbose:
            d = self.param_space.n_dims
            logger.info(
                "Running Morris screening: r=%d trajectories, d=%d → %d evals…",
                n_trajectories, d, n_trajectories * (d + 1),
            )

        self._morris_result = morris_screening(
            self.param_space,
            self.objective_fn,
            n_trajectories=n_trajectories,
            n_levels=n_levels,
            seed=seed,
        )
        return self._morris_result

    def run_gradient(
        self,
        params: Optional[Dict[str, Any]] = None,
        epsilon: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Compute the finite-difference gradient at *params*.

        Returns
        -------
        dict[str, float]
        """
        if params is None:
            params = self.param_space.defaults
        return gradient_sensitivity(
            params, self.objective_fn, self.param_space, epsilon
        )

    def run_hessian(
        self,
        params: Optional[Dict[str, Any]] = None,
        epsilon: float = 1e-4,
    ) -> Tuple[List[str], np.ndarray, float]:
        """
        Compute the finite-difference Hessian at *params*.

        Returns
        -------
        (names, H, condition_number)
        """
        if params is None:
            params = self.param_space.defaults
        names, H = hessian(params, self.objective_fn, self.param_space, epsilon)
        _, _, cond = eigenvalue_analysis(H)
        return names, H, cond

    def run_all(
        self,
        base_params: Optional[Dict[str, Any]] = None,
        oat_n_points: int = 20,
        sobol_n_samples: int = 1024,
        morris_n_trajectories: int = 10,
    ) -> Dict[str, Any]:
        """
        Convenience: run OAT + Morris + Sobol and return all results.

        Returns
        -------
        dict with keys 'oat', 'morris', 'sobol'.
        """
        oat = self.run_oat(base_params, n_points=oat_n_points)
        morris = self.run_morris(n_trajectories=morris_n_trajectories)
        sobol = self.run_sobol(n_samples=sobol_n_samples)
        return {"oat": oat, "morris": morris, "sobol": sobol}

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def rank_summary(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Combine rankings from all completed analyses.

        Returns
        -------
        dict with keys 'oat_top', 'morris_top', 'sobol_sti_top'.
        """
        summary: Dict[str, Any] = {}
        if self._oat_result is not None:
            summary["oat_top"] = self._oat_result.top_k(len(self.param_space))
        if self._morris_result is not None:
            summary["morris_top"] = self._morris_result.top_k(len(self.param_space))
        if self._sobol_result is not None:
            summary["sobol_sti_top"] = self._sobol_result.top_k(len(self.param_space))
        return summary


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_oat_curves(
    oat_result: OATResult,
    save_path: Optional[Union[str, Path]] = None,
    max_cols: int = 3,
    figsize_per_panel: Tuple[float, float] = (4.5, 3.0),
) -> plt.Figure:
    """
    Plot sensitivity curves from an OAT experiment.

    Each sub-panel shows how the objective responds to variation in one
    parameter while all others are held fixed.

    Parameters
    ----------
    oat_result : OATResult
    save_path : str | Path | None
    max_cols : int
    figsize_per_panel : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = sorted(
        oat_result.param_names,
        key=lambda n: oat_result.sensitivity_rank[n],
    )
    n = len(names)
    n_cols = min(n, max_cols)
    n_rows = math.ceil(n / n_cols)
    figsize = (figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    cmap = plt.cm.get_cmap("tab10")

    for idx, name in enumerate(names):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        x = oat_result.values[name]
        y = oat_result.objectives[name]
        color = cmap(idx % 10)

        ax.plot(x, y, lw=2, color=color)
        ax.axhline(
            oat_result.objectives[name][len(y) // 2],
            ls="--", color="grey", lw=0.8, alpha=0.6,
        )
        ax.set_title(
            f"{name}\n(rank {oat_result.sensitivity_rank[name]}, "
            f"Δ={oat_result.sensitivity_range[name]:.3g})",
            fontsize=9,
        )
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel("Objective", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("One-At-a-Time Sensitivity Curves", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("OAT curves saved to %s", save_path)

    return fig


def plot_sobol_indices(
    sobol_result: SobolResult,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 5),
    sort_by: str = "STi",
) -> plt.Figure:
    """
    Bar chart showing first-order (Si) and total-order (STi) Sobol indices
    with confidence intervals.

    Parameters
    ----------
    sobol_result : SobolResult
    save_path : str | Path | None
    figsize : tuple
    sort_by : 'STi' | 'Si'

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = list(sobol_result.Si.keys())
    src = sobol_result.STi if sort_by == "STi" else sobol_result.Si
    names_sorted = sorted(names, key=lambda n: src[n], reverse=True)

    si_vals = np.array([sobol_result.Si[n] for n in names_sorted])
    sti_vals = np.array([sobol_result.STi[n] for n in names_sorted])
    si_err = np.array([sobol_result.Si_conf[n] for n in names_sorted])
    sti_err = np.array([sobol_result.STi_conf[n] for n in names_sorted])

    x = np.arange(len(names_sorted))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, si_vals, width, label="Si (first-order)",
           color="#4C72B0", alpha=0.85, yerr=si_err, capsize=3)
    ax.bar(x + width / 2, sti_vals, width, label="STi (total-order)",
           color="#DD8452", alpha=0.85, yerr=sti_err, capsize=3)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names_sorted, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Sensitivity Index", fontsize=10)
    ax.set_title(
        f"Global Sobol Sensitivity Indices\n"
        f"(N={sobol_result.n_samples}, evals={sobol_result.total_evals}, "
        f"Var(Y)={sobol_result.var_y:.4g})",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_ylim(
        min(-0.05, (si_vals - si_err).min() - 0.05),
        max(1.05, (sti_vals + sti_err).max() + 0.05),
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Sobol indices plot saved to %s", save_path)

    return fig


def plot_morris_mu_star_sigma(
    morris_result: MorrisResult,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (7, 6),
    annotate: bool = True,
) -> plt.Figure:
    """
    Morris μ*/σ scatter plot.

    Parameters in the upper-right (high μ*, high σ) are both important and
    involved in non-linear interactions / interactions with other parameters.
    Parameters in the upper-left region (low μ*, high σ) may have cancelling
    effects.  Parameters in the lower region have low influence.

    Parameters
    ----------
    morris_result : MorrisResult
    save_path : str | Path | None
    figsize : tuple
    annotate : bool
        If True, annotate each point with the parameter name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = list(morris_result.mu_star.keys())
    mu_star_vals = np.array([morris_result.mu_star[n] for n in names])
    sigma_vals = np.array([morris_result.sigma[n] for n in names])

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        mu_star_vals, sigma_vals,
        c=np.arange(len(names)), cmap="tab10",
        s=80, zorder=3, edgecolors="black", linewidths=0.5,
    )

    if annotate:
        for i, name in enumerate(names):
            ax.annotate(
                name,
                (mu_star_vals[i], sigma_vals[i]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                alpha=0.85,
            )

    # Reference line σ = μ*
    max_val = max(mu_star_vals.max(), sigma_vals.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], ls="--", color="grey", lw=1, alpha=0.5,
            label="σ = μ*")
    ax.set_xlim(left=-0.01 * max_val)
    ax.set_ylim(bottom=-0.01 * max_val)
    ax.set_xlabel("μ* (mean |elementary effect|)", fontsize=10)
    ax.set_ylabel("σ (std of elementary effects)", fontsize=10)
    ax.set_title(
        f"Morris Screening — μ*/σ Plot\n"
        f"(r={morris_result.n_trajectories} trajectories, "
        f"p={morris_result.n_levels} levels, "
        f"evals={morris_result.total_evals})",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Morris μ*/σ plot saved to %s", save_path)

    return fig


def plot_hessian(
    names: List[str],
    H: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Visualise the Hessian matrix as a heatmap with eigenvalue bar chart.

    Parameters
    ----------
    names : list[str]
    H : np.ndarray of shape (d, d)
    save_path : str | Path | None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    eigenvalues, _, cond = eigenvalue_analysis(H)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_hess, ax_eig = axes

    # Hessian heatmap
    lim = max(abs(H).max(), 1e-6)
    im = ax_hess.imshow(H, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    plt.colorbar(im, ax=ax_hess, label="∂²f/∂xᵢ∂xⱼ")
    ax_hess.set_xticks(range(len(names)))
    ax_hess.set_yticks(range(len(names)))
    ax_hess.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax_hess.set_yticklabels(names, fontsize=8)
    ax_hess.set_title(f"Hessian Matrix\n(cond = {cond:.2g})", fontsize=10)

    for i in range(len(names)):
        for j in range(len(names)):
            ax_hess.text(
                j, i, f"{H[i, j]:.2g}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(H[i, j]) > 0.6 * lim else "black",
            )

    # Eigenvalue bar chart
    d = len(eigenvalues)
    colors = ["#4C72B0" if v >= 0 else "#DD8452" for v in eigenvalues]
    ax_eig.bar(range(d), eigenvalues, color=colors, edgecolor="black", linewidth=0.5)
    ax_eig.axhline(0, color="black", lw=0.8)
    ax_eig.set_xticks(range(d))
    ax_eig.set_xticklabels([f"λ{i+1}" for i in range(d)], fontsize=8)
    ax_eig.set_ylabel("Eigenvalue", fontsize=9)
    ax_eig.set_title("Hessian Eigenvalues\n(+: convex, −: concave)", fontsize=10)
    ax_eig.grid(axis="y", alpha=0.3)

    plt.suptitle("Hessian Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Hessian plot saved to %s", save_path)

    return fig
