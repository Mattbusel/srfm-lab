"""
optimization/bayesian_optimizer.py
====================================
Gaussian Process-based Bayesian optimization for strategy parameters.

Uses a Matern 5/2 kernel implemented from scratch (no sklearn dependency),
Expected Improvement (EI) acquisition function, and Cholesky-based GP
inference. Designed to minimize the number of expensive backtest evaluations
needed to find high-quality parameter configurations.

Classes:
  MaternKernel52        -- Matern 5/2 covariance function
  GaussianProcess       -- GP regressor with Cholesky solver
  BayesianOptimizer     -- main optimization loop with EI acquisition
  ConvergenceTracker    -- tracks per-iteration best value and GP state

Requires: numpy, pandas, scipy (stats only)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JITTER = 1e-6          -- small diagonal addition for numerical stability
_EI_CANDIDATES = 1000   -- random candidates evaluated per acquisition step
_MIN_SIGMA = 1e-8       -- floor on GP predictive std to avoid div-by-zero
_LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Parameter space helpers
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    """Single parameter specification -- continuous or categorical."""
    name: str
    kind: str                    -- "continuous" or "categorical"
    low: float = 0.0             -- used when kind == "continuous"
    high: float = 1.0            -- used when kind == "continuous"
    choices: List[Any] = field(default_factory=list)  -- used when kind == "categorical"
    log_scale: bool = False      -- sample/scale in log space if True


def parse_param_space(param_space: Dict[str, Any]) -> List[ParamSpec]:
    """
    Convert raw param_space dict to list of ParamSpec.

    Accepted formats:
      {name: (min, max)}         -- continuous
      {name: (min, max, True)}   -- continuous, log-scaled
      {name: [choice1, ...]}     -- categorical
    """
    specs: List[ParamSpec] = []
    for name, spec in param_space.items():
        if isinstance(spec, list):
            specs.append(ParamSpec(name=name, kind="categorical", choices=spec))
        elif isinstance(spec, tuple):
            log_scale = len(spec) == 3 and spec[2] is True
            lo, hi = float(spec[0]), float(spec[1])
            specs.append(ParamSpec(name=name, kind="continuous",
                                   low=lo, high=hi, log_scale=log_scale))
        else:
            raise ValueError(f"Unsupported param spec for '{name}': {spec!r}")
    return specs


def encode_params(params: Dict[str, Any], specs: List[ParamSpec]) -> np.ndarray:
    """Encode a parameter dict to a 1-D float vector in [0, 1]^d."""
    vec = []
    for s in specs:
        val = params[s.name]
        if s.kind == "categorical":
            idx = s.choices.index(val)
            vec.append(float(idx) / max(1.0, len(s.choices) - 1))
        else:
            if s.log_scale:
                lo, hi = math.log(s.low), math.log(s.high)
                vec.append((math.log(float(val)) - lo) / (hi - lo))
            else:
                vec.append((float(val) - s.low) / (s.high - s.low))
    return np.array(vec, dtype=np.float64)


def decode_params(vec: np.ndarray, specs: List[ParamSpec]) -> Dict[str, Any]:
    """Decode a [0,1]^d vector back to a parameter dict."""
    params: Dict[str, Any] = {}
    for i, s in enumerate(specs):
        v = float(np.clip(vec[i], 0.0, 1.0))
        if s.kind == "categorical":
            idx = int(round(v * (len(s.choices) - 1)))
            params[s.name] = s.choices[idx]
        else:
            if s.log_scale:
                lo, hi = math.log(s.low), math.log(s.high)
                params[s.name] = math.exp(lo + v * (hi - lo))
            else:
                params[s.name] = s.low + v * (s.high - s.low)
    return params


def random_candidate(specs: List[ParamSpec], rng: np.random.Generator) -> np.ndarray:
    """Sample a random point uniformly in [0, 1]^d."""
    return rng.uniform(0.0, 1.0, size=len(specs))


# ---------------------------------------------------------------------------
# Matern 5/2 kernel
# ---------------------------------------------------------------------------

class MaternKernel52:
    """
    Matern 5/2 covariance function.

    k(x1, x2) = sigma^2 * (1 + sqrt(5)*d/l + 5*d^2/(3*l^2)) * exp(-sqrt(5)*d/l)

    Parameters
    ----------
    length_scale : float
        Length scale l. Controls smoothness of the function.
    variance : float
        Signal variance sigma^2.
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = max(1e-4, float(length_scale))
        self.variance = max(1e-6, float(variance))

    # -- pairwise squared Euclidean distances
    @staticmethod
    def _sq_dist(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Return (n, m) matrix of squared Euclidean distances."""
        # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2*x1.x2
        sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)   # (n, 1)
        sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)   # (m, 1)
        cross = X1 @ X2.T                               # (n, m)
        return np.maximum(sq1 + sq2.T - 2.0 * cross, 0.0)

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute (n, m) covariance matrix between rows of X1 and X2."""
        sq_d = self._sq_dist(X1, X2)
        d = np.sqrt(sq_d)
        r = math.sqrt(5.0) * d / self.length_scale
        K = self.variance * (1.0 + r + r ** 2 / 3.0) * np.exp(-r)
        return K

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Return diagonal of K(X, X) -- always sigma^2."""
        return np.full(len(X), self.variance)

    def set_params(self, length_scale: float, variance: float) -> None:
        self.length_scale = max(1e-4, float(length_scale))
        self.variance = max(1e-6, float(variance))

    def log_likelihood_grad(
        self, X: np.ndarray, alpha: np.ndarray, K_inv: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute gradient of log-marginal-likelihood w.r.t. length_scale and variance.
        Used by hyperparameter optimization via gradient ascent.
        Returns (d_ll/d_length_scale, d_ll/d_variance).
        """
        n = X.shape[0]
        sq_d = self._sq_dist(X, X)
        d = np.sqrt(sq_d)
        l = self.length_scale
        s2 = self.variance
        r = math.sqrt(5.0) * d / l

        # dK/dl
        dK_dl = (5.0 * sq_d / (3.0 * l ** 3)) * np.exp(-r) * s2
        # dK/ds2
        K_unit = (1.0 + r + r ** 2 / 3.0) * np.exp(-r)
        dK_ds2 = K_unit

        outer = np.outer(alpha, alpha)
        W = outer - K_inv

        grad_l = 0.5 * np.trace(W @ dK_dl)
        grad_s2 = 0.5 * np.trace(W @ dK_ds2)
        return float(grad_l), float(grad_s2)


# ---------------------------------------------------------------------------
# Gaussian Process regressor
# ---------------------------------------------------------------------------

class GaussianProcess:
    """
    Gaussian Process regressor with Matern 5/2 kernel.

    Noise is handled by adding noise_var to the diagonal of K.
    Inference is done via Cholesky decomposition for numerical stability.
    """

    def __init__(
        self,
        kernel: Optional[MaternKernel52] = None,
        noise_var: float = 1e-4,
        normalize_y: bool = True,
    ):
        self.kernel = kernel if kernel is not None else MaternKernel52()
        self.noise_var = max(0.0, noise_var)
        self.normalize_y = normalize_y

        # -- fit state
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None    -- K^{-1} y
        self._L: Optional[np.ndarray] = None        -- Cholesky factor of K
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._fitted: bool = False

    # -- Cholesky-based solve: K*x = b
    @staticmethod
    def _cho_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve (L L^T) x = b using forward/back substitution."""
        # Forward substitution: L v = b
        v = np.linalg.solve(L, b)
        # Back substitution: L^T x = v
        x = np.linalg.solve(L.T, v)
        return x

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        """
        Fit the GP to observations (X, y).

        Builds kernel matrix K, adds jitter + noise, computes Cholesky.
        """
        X = np.atleast_2d(X).copy()
        y = np.asarray(y, dtype=np.float64).copy()

        if self.normalize_y and y.std() > 1e-10:
            self._y_mean = float(y.mean())
            self._y_std = float(y.std())
            y = (y - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0

        n = X.shape[0]
        K = self.kernel(X, X)
        K += (self.noise_var + _JITTER) * np.eye(n)

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # -- add more jitter if Cholesky fails
            K += 1e-4 * np.eye(n)
            L = np.linalg.cholesky(K)

        alpha = self._cho_solve(L, y)

        self._X_train = X
        self._y_train = y
        self._alpha = alpha
        self._L = L
        self._fitted = True
        return self

    def predict(
        self, X_test: np.ndarray, return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict mean and (optionally) standard deviation at X_test.

        Returns (mu, sigma) if return_std else mu.
        Both are in the original (un-normalized) scale.
        """
        if not self._fitted:
            raise RuntimeError("GP must be fitted before calling predict()")

        X_test = np.atleast_2d(X_test)
        K_star = self.kernel(X_test, self._X_train)     -- (m, n)
        mu = K_star @ self._alpha                        -- (m,)

        if return_std:
            # Compute predictive variance
            K_diag = self.kernel.diag(X_test)            -- (m,)
            # v = L^{-1} K_star^T
            v = np.linalg.solve(self._L, K_star.T)       -- (n, m)
            var = K_diag - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 0.0)                   -- numerical clamp
            sigma = np.sqrt(var + _JITTER)

            # Un-normalize
            mu = mu * self._y_std + self._y_mean
            sigma = sigma * self._y_std
            return mu, sigma

        mu = mu * self._y_std + self._y_mean
        return mu

    def log_marginal_likelihood(self) -> float:
        """
        Compute log p(y | X, theta) = -0.5 y^T alpha - sum(log L_ii) - n/2 log(2pi).
        """
        if not self._fitted:
            return -np.inf
        y = self._y_train
        n = len(y)
        log_det = 2.0 * np.sum(np.log(np.diag(self._L)))
        lml = -0.5 * (y @ self._alpha + log_det + n * _LOG_2PI)
        return float(lml)

    def optimize_hyperparams(self, n_restarts: int = 3) -> None:
        """
        Optimize kernel hyperparameters by gradient ascent on log-marginal-likelihood.
        Tries n_restarts random starting points and keeps the best.
        """
        if not self._fitted:
            return

        best_lml = -np.inf
        best_params = (self.kernel.length_scale, self.kernel.variance)
        rng = np.random.default_rng(42)

        for _ in range(n_restarts):
            l = rng.uniform(0.1, 3.0)
            s2 = rng.uniform(0.5, 5.0)
            l, s2 = self._gradient_ascent_hyperparams(l, s2)
            self.kernel.set_params(l, s2)
            self.fit(self._X_train, self._y_train * self._y_std + self._y_mean)
            lml = self.log_marginal_likelihood()
            if lml > best_lml:
                best_lml = lml
                best_params = (l, s2)

        self.kernel.set_params(*best_params)
        self.fit(self._X_train, self._y_train * self._y_std + self._y_mean)

    def _gradient_ascent_hyperparams(
        self, l_init: float, s2_init: float, lr: float = 0.01, n_steps: int = 50
    ) -> Tuple[float, float]:
        """Simple gradient ascent on log-marginal-likelihood for kernel hyperparams."""
        l, s2 = l_init, s2_init
        X = self._X_train
        for _ in range(n_steps):
            self.kernel.set_params(l, s2)
            K = self.kernel(X, X) + (self.noise_var + _JITTER) * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                break
            alpha = self._cho_solve(L, self._y_train)
            K_inv = self._cho_solve(L, np.eye(len(X)))
            gl, gs2 = self.kernel.log_likelihood_grad(X, alpha, K_inv)
            l = max(1e-4, l + lr * gl)
            s2 = max(1e-6, s2 + lr * gs2)
        return l, s2


# ---------------------------------------------------------------------------
# Normal CDF and PDF (scipy-free, accurate to ~1e-7)
# ---------------------------------------------------------------------------

def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """Vectorized standard normal CDF via math.erfc."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    """Vectorized standard normal PDF."""
    return np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Convergence tracking
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    iteration: int
    params: Dict[str, Any]
    value: float             -- raw objective value (higher is better)
    best_so_far: float
    gp_mu: float             -- GP mean at suggested point
    gp_sigma: float          -- GP std at suggested point
    ei: float                -- expected improvement at suggested point
    elapsed_s: float


@dataclass
class ConvergenceTracker:
    records: List[IterationRecord] = field(default_factory=list)

    def add(self, rec: IterationRecord) -> None:
        self.records.append(rec)

    def best_value(self) -> float:
        if not self.records:
            return -np.inf
        return max(r.value for r in self.records)

    def best_params(self) -> Optional[Dict[str, Any]]:
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.value).params

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "iteration": r.iteration,
                "value": r.value,
                "best_so_far": r.best_so_far,
                "gp_mu": r.gp_mu,
                "gp_sigma": r.gp_sigma,
                "ei": r.ei,
                "elapsed_s": r.elapsed_s,
                **{f"param_{k}": v for k, v in r.params.items()},
            }
            for r in self.records
        ])

    def plot_convergence(self) -> Dict[str, List[Any]]:
        """Return dict of lists for convergence plot: iteration, best_value, mu, sigma."""
        return {
            "iteration": [r.iteration for r in self.records],
            "best_value": [r.best_so_far for r in self.records],
            "mu": [r.gp_mu for r in self.records],
            "sigma": [r.gp_sigma for r in self.records],
            "ei": [r.ei for r in self.records],
        }


# ---------------------------------------------------------------------------
# BayesianOptimizer
# ---------------------------------------------------------------------------

class BayesianOptimizer:
    """
    Gaussian Process-based Bayesian optimization for strategy parameters.

    Uses Expected Improvement (EI) acquisition function with a Matern 5/2
    kernel. Handles both continuous and categorical parameters.

    Parameters
    ----------
    param_space : dict
        {name: (min, max)} or {name: [choices]} or {name: (min, max, log_scale)}.
    objective : Callable
        Function that maps a param dict to a scalar score (higher is better).
        May return None to skip a failed evaluation.
    config : dict
        Optional overrides for n_initial, n_iter, xi, noise_var, seed, etc.
    """

    def __init__(
        self,
        param_space: Dict[str, Any],
        objective: Callable[[Dict[str, Any]], Optional[float]],
        config: Optional[Dict[str, Any]] = None,
    ):
        cfg = config or {}
        self.param_space = param_space
        self.specs = parse_param_space(param_space)
        self.objective = objective

        self.gp_kernel = cfg.get("gp_kernel", "matern52")
        self.n_initial = int(cfg.get("n_initial", 10))    -- random initialization
        self.xi = float(cfg.get("xi", 0.01))              -- exploration parameter for EI
        self.noise_var = float(cfg.get("noise_var", 1e-4))
        self.n_ei_candidates = int(cfg.get("n_ei_candidates", _EI_CANDIDATES))
        self.optimize_hyperparams = bool(cfg.get("optimize_hyperparams", True))
        self.hyperopt_every = int(cfg.get("hyperopt_every", 10))
        self.seed = int(cfg.get("seed", 42))
        self.verbose = bool(cfg.get("verbose", True))

        self.X_obs: List[np.ndarray] = []   -- observed parameter vectors (encoded)
        self.y_obs: List[float] = []        -- observed objective values

        kernel = MaternKernel52(
            length_scale=float(cfg.get("length_scale", 1.0)),
            variance=float(cfg.get("kernel_variance", 1.0)),
        )
        self.gp = GaussianProcess(
            kernel=kernel,
            noise_var=self.noise_var,
            normalize_y=True,
        )
        self.tracker = ConvergenceTracker()
        self._rng = np.random.default_rng(self.seed)
        self._iteration = 0

    # -- internal helpers

    def _X_matrix(self) -> np.ndarray:
        return np.array(self.X_obs)   -- (n, d)

    def _evaluate(self, params: Dict[str, Any]) -> Optional[float]:
        """Call the objective and handle None / exceptions."""
        try:
            val = self.objective(params)
            return float(val) if val is not None else None
        except Exception as exc:
            logger.warning("Objective raised %s for params %s", exc, params)
            return None

    def _expected_improvement(
        self, X_candidates: np.ndarray, f_best: float
    ) -> np.ndarray:
        """
        Compute EI(x) = (mu - f_best - xi) * Phi(Z) + sigma * phi(Z)
        where Z = (mu - f_best - xi) / sigma.
        """
        mu, sigma = self.gp.predict(X_candidates, return_std=True)
        sigma = np.maximum(sigma, _MIN_SIGMA)
        improvement = mu - f_best - self.xi
        Z = improvement / sigma
        ei = improvement * _normal_cdf(Z) + sigma * _normal_pdf(Z)
        ei = np.maximum(ei, 0.0)
        return ei

    # -- public API

    def suggest_next(self) -> Dict[str, Any]:
        """
        Suggest the next parameter configuration to evaluate.

        During initialization: pure random sampling.
        After initialization: maximize EI over n_ei_candidates random candidates.
        """
        if len(self.X_obs) < self.n_initial:
            vec = random_candidate(self.specs, self._rng)
            return decode_params(vec, self.specs)

        # -- maximize EI via random search
        candidates = np.array([
            random_candidate(self.specs, self._rng)
            for _ in range(self.n_ei_candidates)
        ])
        f_best = max(self.y_obs) if self.y_obs else 0.0
        ei_vals = self._expected_improvement(candidates, f_best)
        best_idx = int(np.argmax(ei_vals))
        best_vec = candidates[best_idx]
        return decode_params(best_vec, self.specs)

    def register_observation(
        self, params: Dict[str, Any], value: float
    ) -> None:
        """Manually register an (params, value) pair (e.g. from external evaluator)."""
        vec = encode_params(params, self.specs)
        self.X_obs.append(vec)
        self.y_obs.append(float(value))
        if len(self.X_obs) >= 2:
            self.gp.fit(self._X_matrix(), np.array(self.y_obs))

    def optimize(self, n_iter: int = 50) -> Dict[str, Any]:
        """
        Run the full Bayesian optimization loop.

        1. Random initialization for n_initial steps.
        2. GP-guided EI acquisition for remaining steps.
        3. Returns the best parameter dict found.

        Parameters
        ----------
        n_iter : int
            Total number of objective evaluations (including n_initial).

        Returns
        -------
        dict
            Best parameter configuration found.
        """
        logger.info("Starting Bayesian optimization: %d iterations, %d random init",
                    n_iter, self.n_initial)

        for i in range(n_iter):
            t0 = time.perf_counter()
            self._iteration += 1

            # -- suggest next point
            params = self.suggest_next()
            vec = encode_params(params, self.specs)

            # -- evaluate objective
            val = self._evaluate(params)
            if val is None:
                # -- replace failed eval with current worst or 0
                val = min(self.y_obs) - 1.0 if self.y_obs else 0.0

            # -- store observation
            self.X_obs.append(vec)
            self.y_obs.append(val)

            # -- fit / update GP once we have enough points
            gp_mu, gp_sigma, ei_val = 0.0, 0.0, 0.0
            if len(self.X_obs) >= 2:
                self.gp.fit(self._X_matrix(), np.array(self.y_obs))

                # -- optimize GP hyperparameters periodically
                if (
                    self.optimize_hyperparams
                    and len(self.X_obs) >= self.n_initial
                    and (i + 1) % self.hyperopt_every == 0
                ):
                    self.gp.optimize_hyperparams(n_restarts=2)

                # -- record GP prediction at current point
                mu_arr, sig_arr = self.gp.predict(vec.reshape(1, -1), return_std=True)
                gp_mu = float(mu_arr[0])
                gp_sigma = float(sig_arr[0])
                if self.y_obs:
                    f_best = max(self.y_obs[:-1]) if len(self.y_obs) > 1 else self.y_obs[0]
                    ei_arr = self._expected_improvement(vec.reshape(1, -1), f_best)
                    ei_val = float(ei_arr[0])

            best_so_far = max(self.y_obs)
            elapsed = time.perf_counter() - t0

            rec = IterationRecord(
                iteration=self._iteration,
                params=params,
                value=val,
                best_so_far=best_so_far,
                gp_mu=gp_mu,
                gp_sigma=gp_sigma,
                ei=ei_val,
                elapsed_s=elapsed,
            )
            self.tracker.add(rec)

            if self.verbose and (i % 10 == 0 or i == n_iter - 1):
                logger.info(
                    "Iter %3d/%d | val=%.4f | best=%.4f | GP mu=%.4f sigma=%.4f",
                    i + 1, n_iter, val, best_so_far, gp_mu, gp_sigma,
                )

        best = self.tracker.best_params()
        logger.info("Optimization complete. Best value: %.4f", self.tracker.best_value())
        return best or {}

    def plot_convergence(self) -> Dict[str, List[Any]]:
        """
        Return convergence data suitable for plotting.

        Returns a dict with keys:
          iteration, best_value, mu, sigma, ei
        Each maps to a list of per-iteration values.
        """
        return self.tracker.plot_convergence()

    def results_dataframe(self) -> pd.DataFrame:
        """Return full per-iteration results as a DataFrame."""
        return self.tracker.to_dataframe()

    def best_params(self) -> Optional[Dict[str, Any]]:
        """Return the best parameter set found so far."""
        return self.tracker.best_params()

    def best_value(self) -> float:
        """Return the best objective value found so far."""
        return self.tracker.best_value()

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the optimization run."""
        df = self.tracker.to_dataframe()
        return {
            "n_evaluations": len(self.y_obs),
            "best_value": self.best_value(),
            "best_params": self.best_params(),
            "gp_kernel": self.gp_kernel,
            "gp_length_scale": self.gp.kernel.length_scale,
            "gp_variance": self.gp.kernel.variance,
            "n_initial": self.n_initial,
            "xi": self.xi,
            "convergence": self.plot_convergence(),
        }


# ---------------------------------------------------------------------------
# Warm-start helper -- initialize GP from prior run results
# ---------------------------------------------------------------------------

class WarmStartBayesianOptimizer(BayesianOptimizer):
    """
    BayesianOptimizer that pre-loads prior observations from a DataFrame.
    Useful for continuing an interrupted run or seeding from external data.
    """

    def warm_start(self, prior_df: pd.DataFrame) -> None:
        """
        Load prior (params, value) observations.

        prior_df must have columns matching param names + 'value'.
        """
        param_cols = [s.name for s in self.specs]
        for _, row in prior_df.iterrows():
            params = {c: row[c] for c in param_cols if c in row}
            val = float(row["value"])
            vec = encode_params(params, self.specs)
            self.X_obs.append(vec)
            self.y_obs.append(val)

        if len(self.X_obs) >= 2:
            self.gp.fit(self._X_matrix(), np.array(self.y_obs))
            logger.info("Warm-started GP with %d prior observations", len(self.X_obs))


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_bayesian_optimizer(
    param_space: Dict[str, Any],
    objective: Callable,
    n_initial: int = 10,
    xi: float = 0.01,
    seed: int = 42,
    verbose: bool = True,
) -> BayesianOptimizer:
    """
    Convenience factory for BayesianOptimizer with common config.

    Example
    -------
    >>> opt = make_bayesian_optimizer(
    ...     {"cf": (0.0005, 0.05), "bh_form": (0.5, 3.0)},
    ...     my_sharpe_function,
    ...     n_initial=15,
    ... )
    >>> best = opt.optimize(n_iter=60)
    """
    config = {
        "n_initial": n_initial,
        "xi": xi,
        "seed": seed,
        "verbose": verbose,
        "optimize_hyperparams": True,
        "hyperopt_every": 10,
    }
    return BayesianOptimizer(param_space, objective, config)
