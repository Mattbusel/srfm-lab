"""
research/param_explorer/bayesian_opt.py
=========================================
Bayesian optimisation using a Gaussian Process surrogate.

Supports single-objective optimisation (BayesianOptimizer) and
bi-objective Pareto-front discovery (MOBayesianOptimizer).

Classes
-------
GPSurrogate          : sklearn GP wrapper with incremental update
AcquisitionFunction  : Enum of EI / UCB / PI / Thompson Sampling
BayesianOptimizer    : Main single-objective BO loop
MOBayesianOptimizer  : Multi-objective BO for (Sharpe, max_dd)
BayesOptResult       : Results dataclass

Stand-alone acquisition functions
----------------------------------
expected_improvement
upper_confidence_bound
probability_improvement
thompson_sample
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import norm as scipy_norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern,
    ConstantKernel,
    WhiteKernel,
    RBF,
    Sum,
    Product,
)
from sklearn.preprocessing import StandardScaler

from research.param_explorer.space import ParamSpace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AcquisitionFunction enum
# ---------------------------------------------------------------------------

class AcquisitionFunction(str, Enum):
    """Supported acquisition strategies."""
    EI = "ei"                        # Expected Improvement
    UCB = "ucb"                      # Upper Confidence Bound
    PI = "pi"                        # Probability of Improvement
    THOMPSON_SAMPLING = "thompson"   # Thompson Sampling via posterior draw


# ---------------------------------------------------------------------------
# Acquisition function implementations
# ---------------------------------------------------------------------------

def expected_improvement(
    x: np.ndarray,
    surrogate: "GPSurrogate",
    best_f: float,
    xi: float = 0.01,
) -> float:
    """
    Expected Improvement acquisition function.

    EI(x) = E[max(f(x) - f*, 0)]
           = (μ - f* - ξ) Φ(Z) + σ φ(Z)
    where Z = (μ - f* - ξ) / σ

    Parameters
    ----------
    x : np.ndarray of shape (d,)
    surrogate : GPSurrogate
    best_f : float
        Best observed objective so far.
    xi : float
        Exploration-exploitation trade-off parameter.

    Returns
    -------
    float  (higher is better → maximise this acquisition)
    """
    x_2d = x.reshape(1, -1)
    mu, sigma = surrogate.predict(x_2d)
    mu = float(mu[0])
    sigma = float(sigma[0])

    if sigma < 1e-10:
        return 0.0

    improvement = mu - best_f - xi
    Z = improvement / sigma
    ei = improvement * scipy_norm.cdf(Z) + sigma * scipy_norm.pdf(Z)
    return float(max(ei, 0.0))


def upper_confidence_bound(
    x: np.ndarray,
    surrogate: "GPSurrogate",
    beta: float = 2.0,
) -> float:
    """
    Upper Confidence Bound acquisition.

    UCB(x) = μ(x) + β σ(x)

    Parameters
    ----------
    x : np.ndarray of shape (d,)
    surrogate : GPSurrogate
    beta : float

    Returns
    -------
    float
    """
    x_2d = x.reshape(1, -1)
    mu, sigma = surrogate.predict(x_2d)
    return float(mu[0]) + beta * float(sigma[0])


def probability_improvement(
    x: np.ndarray,
    surrogate: "GPSurrogate",
    best_f: float,
    xi: float = 0.01,
) -> float:
    """
    Probability of Improvement acquisition.

    PI(x) = Φ((μ(x) - f* - ξ) / σ(x))

    Parameters
    ----------
    x : np.ndarray of shape (d,)
    surrogate : GPSurrogate
    best_f : float
    xi : float

    Returns
    -------
    float ∈ [0, 1]
    """
    x_2d = x.reshape(1, -1)
    mu, sigma = surrogate.predict(x_2d)
    mu = float(mu[0])
    sigma = float(sigma[0])

    if sigma < 1e-10:
        return float(mu > best_f + xi)

    Z = (mu - best_f - xi) / sigma
    return float(scipy_norm.cdf(Z))


def thompson_sample(
    x: np.ndarray,
    surrogate: "GPSurrogate",
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Thompson Sampling: draw from the GP posterior and return the sample value.

    Parameters
    ----------
    x : np.ndarray of shape (d,)
    surrogate : GPSurrogate
    rng : np.random.Generator | None

    Returns
    -------
    float
    """
    if rng is None:
        rng = np.random.default_rng()
    x_2d = x.reshape(1, -1)
    mu, sigma = surrogate.predict(x_2d)
    return float(rng.normal(float(mu[0]), float(sigma[0])))


# ---------------------------------------------------------------------------
# GPSurrogate
# ---------------------------------------------------------------------------

class GPSurrogate:
    """
    Gaussian Process surrogate model wrapping sklearn's
    :class:`sklearn.gaussian_process.GaussianProcessRegressor`.

    Internally standardises the output (zero mean, unit variance) to improve
    numerical stability.  Predictions are returned in the original scale.

    Parameters
    ----------
    kernel : sklearn kernel | None
        GP kernel.  Defaults to Matern(ν=2.5) × ConstantKernel + WhiteKernel.
    n_restarts_optimizer : int
        Number of random restarts for kernel hyperparameter optimisation.
    alpha : float
        Observation noise (diagonal jitter).
    normalise_y : bool
        Whether to internally normalise y values.
    """

    def __init__(
        self,
        kernel=None,
        n_restarts_optimizer: int = 5,
        alpha: float = 1e-6,
        normalise_y: bool = True,
        random_state: int = 42,
    ) -> None:
        if kernel is None:
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 10.0), nu=2.5)
                + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
            )

        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
            normalize_y=normalise_y,
            random_state=random_state,
        )
        self._fitted = False
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        """
        Fit the GP to observations (X, y).

        Parameters
        ----------
        X : np.ndarray of shape (n, d)  — unit hypercube
        y : np.ndarray of shape (n,)

        Returns
        -------
        self
        """
        self._X = X.copy()
        self._y = y.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gpr.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation at *X*.

        Parameters
        ----------
        X : np.ndarray of shape (m, d)

        Returns
        -------
        (mean, std) each of shape (m,)
        """
        if not self._fitted:
            raise RuntimeError("GPSurrogate.predict() called before fit().")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = self._gpr.predict(X, return_std=True)
        return mean, std

    def update(self, x_new: np.ndarray, y_new: float) -> "GPSurrogate":
        """
        Append a new observation and refit the GP.

        Parameters
        ----------
        x_new : np.ndarray of shape (d,)
        y_new : float

        Returns
        -------
        self
        """
        if self._X is None:
            raise RuntimeError("Call fit() before update().")
        self._X = np.vstack([self._X, x_new.reshape(1, -1)])
        self._y = np.append(self._y, y_new)
        return self.fit(self._X, self._y)

    def sample_y(
        self,
        X: np.ndarray,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Draw *n_samples* function samples from the GP posterior at *X*.

        Returns
        -------
        np.ndarray of shape (m, n_samples)
        """
        if not self._fitted:
            raise RuntimeError("GPSurrogate.sample_y() called before fit().")
        if rng is None:
            rng = np.random.default_rng()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = self._gpr.sample_y(X, n_samples=n_samples, random_state=rng.integers(0, 2**31))
        return samples

    @property
    def X_observed(self) -> Optional[np.ndarray]:
        return self._X

    @property
    def y_observed(self) -> Optional[np.ndarray]:
        return self._y

    @property
    def n_observations(self) -> int:
        return len(self._y) if self._y is not None else 0

    def kernel_summary(self) -> str:
        """Return a string description of the fitted kernel."""
        if not self._fitted:
            return "not fitted"
        return str(self._gpr.kernel_)


# ---------------------------------------------------------------------------
# BayesOptResult
# ---------------------------------------------------------------------------

@dataclass
class BayesOptResult:
    """
    Results of a Bayesian optimisation run.

    Attributes
    ----------
    best_params : dict
        Best parameter dict found.
    best_score : float
        Corresponding objective value.
    history_X : np.ndarray of shape (n_total, d)
        All evaluated points in unit space.
    history_y : np.ndarray of shape (n_total,)
        All corresponding objective values.
    history_params : list[dict]
        All evaluated parameter dicts (decoded).
    surrogate : GPSurrogate
        The final fitted surrogate model.
    convergence : np.ndarray of shape (n_total,)
        Running best objective at each iteration.
    acquisition : AcquisitionFunction
    n_init : int
    n_iter : int
    param_space_name : str
    """

    best_params: Dict[str, Any]
    best_score: float
    history_X: np.ndarray
    history_y: np.ndarray
    history_params: List[Dict[str, Any]]
    surrogate: GPSurrogate
    convergence: np.ndarray
    acquisition: AcquisitionFunction
    n_init: int
    n_iter: int
    param_space_name: str

    @property
    def n_total_evals(self) -> int:
        return len(self.history_y)

    def top_k(self, k: int) -> List[Tuple[Dict[str, Any], float]]:
        """Return the top *k* (params, score) pairs, descending."""
        idx = np.argsort(self.history_y)[::-1][:k]
        return [(self.history_params[i], float(self.history_y[i])) for i in idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": float(self.best_score),
            "n_init": self.n_init,
            "n_iter": self.n_iter,
            "n_total_evals": self.n_total_evals,
            "acquisition": self.acquisition.value,
            "param_space_name": self.param_space_name,
            "convergence": self.convergence.tolist(),
            "history_y": self.history_y.tolist(),
        }


# ---------------------------------------------------------------------------
# BayesianOptimizer
# ---------------------------------------------------------------------------

class BayesianOptimizer:
    """
    Single-objective Bayesian optimisation using a GP surrogate.

    Parameters
    ----------
    param_space : ParamSpace
    objective_fn : callable
        Maps param dict → scalar float (higher = better).
    acquisition : AcquisitionFunction
    n_init : int
        Number of initial random (Sobol) evaluations before fitting the GP.
    gp_kwargs : dict | None
        Extra keyword arguments passed to :class:`GPSurrogate`.
    seed : int
    """

    def __init__(
        self,
        param_space: ParamSpace,
        objective_fn: Callable[[Dict[str, Any]], float],
        acquisition: AcquisitionFunction = AcquisitionFunction.EI,
        n_init: int = 10,
        gp_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        xi: float = 0.01,
        beta: float = 2.0,
        n_restarts_acquisition: int = 10,
    ) -> None:
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.acquisition = acquisition
        self.n_init = n_init
        self.seed = seed
        self.xi = xi
        self.beta = beta
        self.n_restarts_acquisition = n_restarts_acquisition

        gp_kwargs = gp_kwargs or {}
        self.surrogate = GPSurrogate(**gp_kwargs)

        self._rng = np.random.default_rng(seed)
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._params_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initial_design(self, n_init: int) -> np.ndarray:
        """Generate Sobol initial design in unit hypercube."""
        return self.param_space.sample_sobol(n_init, seed=self.seed)

    def _evaluate(self, x_unit: np.ndarray) -> float:
        """Decode *x_unit* and evaluate the objective."""
        params = self.param_space.to_params(x_unit)
        return float(self.objective_fn(params))

    def _acq_value(self, x: np.ndarray) -> float:
        """Scalar acquisition value at *x* (to be maximised)."""
        best_f = float(max(self._y)) if self._y else 0.0
        if self.acquisition == AcquisitionFunction.EI:
            return expected_improvement(x, self.surrogate, best_f, xi=self.xi)
        elif self.acquisition == AcquisitionFunction.UCB:
            return upper_confidence_bound(x, self.surrogate, beta=self.beta)
        elif self.acquisition == AcquisitionFunction.PI:
            return probability_improvement(x, self.surrogate, best_f, xi=self.xi)
        elif self.acquisition == AcquisitionFunction.THOMPSON_SAMPLING:
            return thompson_sample(x, self.surrogate, rng=self._rng)
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition}")

    def _optimize_acquisition(self) -> np.ndarray:
        """
        Maximise the acquisition function via L-BFGS-B with multiple restarts.

        Returns
        -------
        np.ndarray of shape (d,) — best candidate in unit hypercube.
        """
        d = self.param_space.n_dims
        bounds = [(0.0, 1.0)] * d
        best_x = None
        best_acq = -np.inf

        # Multi-start: random starts + one warm start from best observed
        starts = self._rng.uniform(0.0, 1.0, size=(self.n_restarts_acquisition, d))
        if self._X:
            best_obs_idx = int(np.argmax(self._y))
            starts = np.vstack([starts, self._X[best_obs_idx]])

        for x0 in starts:
            try:
                res = scipy_minimize(
                    fun=lambda x: -self._acq_value(x),
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 200, "ftol": 1e-9},
                )
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    best_x = np.clip(res.x, 0.0, 1.0)
            except Exception as exc:
                logger.debug("Acquisition optimisation failed: %s", exc)

        if best_x is None:
            best_x = self._rng.uniform(0.0, 1.0, size=d)

        return best_x

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        n_iter: int = 50,
        verbose: bool = True,
        callback: Optional[Callable[[int, Dict[str, Any], float], None]] = None,
    ) -> BayesOptResult:
        """
        Execute the full Bayesian optimisation loop.

        Parameters
        ----------
        n_iter : int
            Number of BO iterations *after* the initial design.
        verbose : bool
        callback : callable | None
            Called after each iteration with (iteration, params, score).

        Returns
        -------
        BayesOptResult
        """
        # 1. Initial design
        X_init = self._initial_design(self.n_init)
        if verbose:
            logger.info(
                "BayesianOptimizer: evaluating %d initial Sobol points…", self.n_init
            )

        for i in range(len(X_init)):
            x = X_init[i]
            try:
                y = self._evaluate(x)
            except Exception as exc:
                logger.warning("Initial eval %d failed: %s", i, exc)
                y = float("-inf")
            self._X.append(x)
            self._y.append(y)
            self._params_history.append(self.param_space.to_params(x))

            if verbose and (i + 1) % max(1, self.n_init // 5) == 0:
                logger.info("  init %d/%d, y=%.4g", i + 1, self.n_init, y)

        # 2. Fit initial surrogate
        X_arr = np.array(self._X)
        y_arr = np.array(self._y)
        self.surrogate.fit(X_arr, y_arr)

        if verbose:
            logger.info(
                "Initial surrogate fitted. Best init score: %.4g", max(self._y)
            )

        # 3. BO iterations
        for iteration in range(1, n_iter + 1):
            # Suggest next point
            x_next = self._optimize_acquisition()

            # Evaluate
            try:
                y_next = self._evaluate(x_next)
            except Exception as exc:
                logger.warning("BO iter %d eval failed: %s", iteration, exc)
                y_next = float("-inf")

            self._X.append(x_next)
            self._y.append(y_next)
            params_next = self.param_space.to_params(x_next)
            self._params_history.append(params_next)

            # Update surrogate
            try:
                self.surrogate.update(x_next, y_next)
            except Exception as exc:
                logger.warning("Surrogate update failed at iter %d: %s", iteration, exc)
                # Full refit
                self.surrogate.fit(np.array(self._X), np.array(self._y))

            if verbose and iteration % max(1, n_iter // 10) == 0:
                logger.info(
                    "  iter %d/%d, y=%.4g, best=%.4g",
                    iteration, n_iter, y_next, max(self._y),
                )

            if callback is not None:
                callback(iteration, params_next, y_next)

        # 4. Build result
        history_y = np.array(self._y)
        history_X = np.array(self._X)
        best_idx = int(np.argmax(history_y))

        convergence = np.maximum.accumulate(history_y)

        return BayesOptResult(
            best_params=self._params_history[best_idx],
            best_score=float(history_y[best_idx]),
            history_X=history_X,
            history_y=history_y,
            history_params=self._params_history,
            surrogate=self.surrogate,
            convergence=convergence,
            acquisition=self.acquisition,
            n_init=self.n_init,
            n_iter=n_iter,
            param_space_name=self.param_space.name,
        )


# ---------------------------------------------------------------------------
# Multi-objective Bayesian Optimizer
# ---------------------------------------------------------------------------

@dataclass
class MOBayesOptResult:
    """
    Multi-objective Bayesian optimisation result.

    Attributes
    ----------
    pareto_params : list[dict]
        Parameter dicts on the Pareto front.
    pareto_scores : np.ndarray of shape (k, n_objectives)
        Objective values for each Pareto-front point.
    objective_names : list[str]
    history_params : list[dict]
    history_scores : np.ndarray of shape (n_total, n_objectives)
    n_init : int
    n_iter : int
    """

    pareto_params: List[Dict[str, Any]]
    pareto_scores: np.ndarray
    objective_names: List[str]
    history_params: List[Dict[str, Any]]
    history_scores: np.ndarray
    n_init: int
    n_iter: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_pareto": len(self.pareto_params),
            "objective_names": self.objective_names,
            "pareto_scores": self.pareto_scores.tolist(),
            "n_init": self.n_init,
            "n_iter": self.n_iter,
        }


def _is_pareto_efficient(costs: np.ndarray, maximise: bool = True) -> np.ndarray:
    """
    Return a boolean mask of the Pareto-efficient rows.

    Parameters
    ----------
    costs : np.ndarray of shape (n, k)
    maximise : bool
        True → maximise all objectives.

    Returns
    -------
    np.ndarray of shape (n,) dtype bool
    """
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        if maximise:
            dominated = np.all(costs >= costs[i], axis=1) & np.any(costs > costs[i], axis=1)
        else:
            dominated = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        dominated[i] = False
        is_efficient[i] = not np.any(dominated[is_efficient])
    return is_efficient


class MOBayesianOptimizer:
    """
    Multi-objective Bayesian optimisation for two objectives (e.g. Sharpe and
    max drawdown).

    Uses a scalarisation approach: at each iteration, a random weight vector
    w ∈ Δ² is sampled and a single-objective BO step is run to optimise
    w₁·f₁ + w₂·f₂.  Over many iterations this traces out the Pareto front.

    Parameters
    ----------
    param_space : ParamSpace
    objective_fns : list[callable]
        List of exactly two objective functions, each mapping params → float.
    objective_names : list[str]
        Names for the two objectives.
    n_init : int
    gp_kwargs : dict | None
    seed : int
    """

    def __init__(
        self,
        param_space: ParamSpace,
        objective_fns: List[Callable[[Dict[str, Any]], float]],
        objective_names: Optional[List[str]] = None,
        n_init: int = 10,
        gp_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        acquisition: AcquisitionFunction = AcquisitionFunction.EI,
    ) -> None:
        if len(objective_fns) < 2:
            raise ValueError("MOBayesianOptimizer requires at least 2 objectives.")
        self.param_space = param_space
        self.objective_fns = objective_fns
        self.objective_names = objective_names or [f"obj_{i}" for i in range(len(objective_fns))]
        self.n_init = n_init
        self.seed = seed
        self.acquisition = acquisition

        gp_kwargs = gp_kwargs or {}
        # Separate GP for each objective
        self._surrogates = [GPSurrogate(**gp_kwargs) for _ in objective_fns]
        self._rng = np.random.default_rng(seed)
        self._X: List[np.ndarray] = []
        self._scores: List[np.ndarray] = []
        self._params_history: List[Dict[str, Any]] = []

    def _evaluate_all(self, x_unit: np.ndarray) -> np.ndarray:
        params = self.param_space.to_params(x_unit)
        return np.array([float(fn(params)) for fn in self.objective_fns])

    def _scalarised_surrogate_predict(
        self, x: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, float]:
        """Return (scalarised_mean, scalarised_std)."""
        mu_list = []
        var_list = []
        for k, surr in enumerate(self._surrogates):
            mu_k, std_k = surr.predict(x.reshape(1, -1))
            mu_list.append(float(mu_k[0]))
            var_list.append(float(std_k[0]) ** 2)
        # Linear scalarisation
        mu_scalar = float(np.dot(weights, mu_list))
        std_scalar = float(math.sqrt(max(np.dot(weights ** 2, var_list), 0)))
        return mu_scalar, std_scalar

    def _optimize_scalarised_acq(
        self,
        weights: np.ndarray,
        best_f: float,
    ) -> np.ndarray:
        """L-BFGS-B maximisation of EI of the scalarised GP."""
        d = self.param_space.n_dims
        bounds = [(0.0, 1.0)] * d
        best_x = None
        best_acq = -np.inf

        starts = self._rng.uniform(0.0, 1.0, size=(10, d))

        for x0 in starts:
            def neg_ei(x: np.ndarray) -> float:
                mu, sigma = self._scalarised_surrogate_predict(x, weights)
                if sigma < 1e-10:
                    return 0.0
                improvement = mu - best_f - 0.01
                Z = improvement / sigma
                ei = improvement * scipy_norm.cdf(Z) + sigma * scipy_norm.pdf(Z)
                return -float(max(ei, 0.0))

            try:
                res = scipy_minimize(
                    neg_ei, x0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 100},
                )
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    best_x = np.clip(res.x, 0.0, 1.0)
            except Exception:
                pass

        return best_x if best_x is not None else self._rng.uniform(0.0, 1.0, size=d)

    def run(
        self,
        n_iter: int = 50,
        verbose: bool = True,
    ) -> MOBayesOptResult:
        """
        Execute the multi-objective BO loop.

        Returns
        -------
        MOBayesOptResult
        """
        # Initial design
        X_init = self.param_space.sample_sobol(self.n_init, seed=self.seed)
        if verbose:
            logger.info(
                "MOBayesianOptimizer: evaluating %d initial points…", self.n_init
            )

        for i in range(len(X_init)):
            x = X_init[i]
            scores = self._evaluate_all(x)
            self._X.append(x)
            self._scores.append(scores)
            self._params_history.append(self.param_space.to_params(x))

        # Fit initial surrogates
        X_arr = np.array(self._X)
        scores_arr = np.array(self._scores)
        for k, surr in enumerate(self._surrogates):
            surr.fit(X_arr, scores_arr[:, k])

        if verbose:
            logger.info("Initial surrogates fitted.")

        # BO iterations
        for iteration in range(1, n_iter + 1):
            # Sample random weights from simplex
            w_raw = self._rng.exponential(1.0, size=len(self.objective_fns))
            weights = w_raw / w_raw.sum()

            # Current best scalarised
            scalarised_hist = scores_arr @ weights
            best_f = float(scalarised_hist.max())

            x_next = self._optimize_scalarised_acq(weights, best_f)
            scores_next = self._evaluate_all(x_next)

            self._X.append(x_next)
            self._scores.append(scores_next)
            self._params_history.append(self.param_space.to_params(x_next))

            X_arr = np.array(self._X)
            scores_arr = np.array(self._scores)
            for k, surr in enumerate(self._surrogates):
                surr.update(x_next, scores_next[k])

            if verbose and iteration % max(1, n_iter // 10) == 0:
                logger.info(
                    "  MO iter %d/%d", iteration, n_iter
                )

        # Identify Pareto front (maximise all objectives)
        pareto_mask = _is_pareto_efficient(scores_arr, maximise=True)
        pareto_params = [self._params_history[i] for i in np.where(pareto_mask)[0]]
        pareto_scores = scores_arr[pareto_mask]

        return MOBayesOptResult(
            pareto_params=pareto_params,
            pareto_scores=pareto_scores,
            objective_names=self.objective_names,
            history_params=self._params_history,
            history_scores=scores_arr,
            n_init=self.n_init,
            n_iter=n_iter,
        )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_convergence(
    result: BayesOptResult,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (9, 5),
) -> plt.Figure:
    """
    Plot the optimisation convergence (running best) over iterations.

    Parameters
    ----------
    result : BayesOptResult
    save_path : str | Path | None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    iters = np.arange(1, len(result.history_y) + 1)
    init_mask = iters <= result.n_init

    # Panel 1: all evaluations + convergence curve
    ax1.scatter(
        iters[init_mask], result.history_y[init_mask],
        color="#4C72B0", s=25, alpha=0.7, label="Initial design",
    )
    ax1.scatter(
        iters[~init_mask], result.history_y[~init_mask],
        color="#DD8452", s=25, alpha=0.7, label="BO iterations",
    )
    ax1.plot(iters, result.convergence, color="black", lw=2, label="Running best")
    ax1.axvline(result.n_init + 0.5, ls="--", color="grey", lw=1, alpha=0.6)
    ax1.set_xlabel("Evaluation #", fontsize=10)
    ax1.set_ylabel("Objective", fontsize=10)
    ax1.set_title("Convergence Curve", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: histogram of evaluated objective values
    ax2.hist(
        result.history_y[~init_mask], bins=20, color="#DD8452", alpha=0.7,
        edgecolor="white", label="BO evals",
    )
    ax2.hist(
        result.history_y[init_mask], bins=10, color="#4C72B0", alpha=0.6,
        edgecolor="white", label="Initial",
    )
    ax2.axvline(result.best_score, ls="--", color="red", lw=1.5, label=f"Best={result.best_score:.4g}")
    ax2.set_xlabel("Objective", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Score Distribution", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Bayesian Optimisation — {result.param_space_name}\n"
        f"(acquisition={result.acquisition.value}, n_init={result.n_init}, "
        f"n_iter={result.n_iter})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Convergence plot saved to %s", save_path)

    return fig


def plot_surrogate_1d(
    result: BayesOptResult,
    param_name: str,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (9, 5),
    n_grid: int = 100,
) -> plt.Figure:
    """
    Visualise the GP surrogate along one parameter dimension (marginalised).

    The surrogate is evaluated on a grid over *param_name* while all other
    dimensions are fixed at the best observed point.

    Parameters
    ----------
    result : BayesOptResult
    param_name : str
    save_path : str | Path | None
    figsize : tuple
    n_grid : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    param_space = result.surrogate._gpr  # access underlying GP
    # Get index of param_name in the space by scanning history_X columns
    # We need the original param_space — retrieve from best_params + history
    # Build a 1-D grid: use best observed x as base, vary dim j
    best_x = result.history_X[np.argmax(result.history_y)]
    d = result.history_X.shape[1]

    # Find column index for param_name by checking param names
    # (We recover the name order from best_params keys)
    param_names_ordered = list(result.best_params.keys())
    if param_name not in param_names_ordered:
        raise ValueError(f"param_name {param_name!r} not found in result.best_params.")
    j = param_names_ordered.index(param_name)

    grid = np.linspace(0.0, 1.0, n_grid)
    X_grid = np.tile(best_x, (n_grid, 1))
    X_grid[:, j] = grid

    mu, sigma = result.surrogate.predict(X_grid)

    # Actual parameter values on the grid axis
    # Approximate: use best_params to guess spec
    # (Just use the raw unit grid if we can't decode)
    grid_actual = grid  # fallback

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top: surrogate mean ± 2σ
    ax1.plot(grid_actual, mu, color="#4C72B0", lw=2, label="GP mean")
    ax1.fill_between(
        grid_actual,
        mu - 2 * sigma,
        mu + 2 * sigma,
        alpha=0.25, color="#4C72B0", label="±2σ",
    )
    # Scatter observed points projected to this axis
    obs_j = result.history_X[:, j]
    obs_y = result.history_y
    ax1.scatter(obs_j, obs_y, s=20, color="#DD8452", alpha=0.6, zorder=5, label="Observations")
    ax1.set_ylabel("Objective", fontsize=9)
    ax1.set_title(f"GP Surrogate — {param_name} (unit space)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Bottom: acquisition function
    best_f = float(np.max(result.history_y))
    acq_vals = np.array([
        expected_improvement(X_grid[k], result.surrogate, best_f)
        for k in range(n_grid)
    ])
    ax2.plot(grid_actual, acq_vals, color="#55A868", lw=2)
    ax2.set_xlabel(f"{param_name} (unit)", fontsize=9)
    ax2.set_ylabel("EI", fontsize=9)
    ax2.set_title("Expected Improvement", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Surrogate 1-D plot saved to %s", save_path)

    return fig


def plot_pareto_front(
    mo_result: MOBayesOptResult,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Plot the Pareto front for two objectives.

    Parameters
    ----------
    mo_result : MOBayesOptResult
    save_path : str | Path | None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # All evaluated points
    all_scores = mo_result.history_scores
    ax.scatter(
        all_scores[:, 0], all_scores[:, 1],
        s=15, alpha=0.3, color="grey", label="All evaluations",
    )

    # Pareto front
    pf = mo_result.pareto_scores
    # Sort for line plot
    sort_idx = np.argsort(pf[:, 0])
    ax.scatter(
        pf[sort_idx, 0], pf[sort_idx, 1],
        s=60, color="#DD8452", edgecolors="black", linewidths=0.8,
        zorder=5, label="Pareto front",
    )
    ax.plot(pf[sort_idx, 0], pf[sort_idx, 1], color="#DD8452", lw=1.5, alpha=0.7)

    obj_names = mo_result.objective_names
    ax.set_xlabel(obj_names[0], fontsize=10)
    ax.set_ylabel(obj_names[1], fontsize=10)
    ax.set_title(
        f"Pareto Front: {obj_names[0]} vs {obj_names[1]}\n"
        f"({len(pf)} Pareto points from {len(all_scores)} evaluations)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Pareto front plot saved to %s", save_path)

    return fig
