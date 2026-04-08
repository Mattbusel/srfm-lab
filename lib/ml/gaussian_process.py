"""
Gaussian Process regression and classification for financial forecasting.

Kernels: RBF, Matern-5/2, Matern-3/2, Periodic, Rational Quadratic, Linear,
         Spectral Mixture.
Models: GPRegressor, GPClassifier (Laplace), SparseGP (FITC), MultiOutputGP,
        VolatilityGP.
Utilities: marginal likelihood, analytic gradients, LOOCV model selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit  # logistic sigmoid


# ---------------------------------------------------------------------------
# Kernel base
# ---------------------------------------------------------------------------

class Kernel:
    """Abstract base class for kernel functions."""

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def gradient(self, X1: np.ndarray, X2: np.ndarray,
                 param_idx: int) -> np.ndarray:
        """Analytic gradient of K wrt log-transformed hyperparameter param_idx."""
        raise NotImplementedError

    @property
    def params(self) -> np.ndarray:
        raise NotImplementedError

    @params.setter
    def params(self, v: np.ndarray):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Kernel implementations
# ---------------------------------------------------------------------------

class RBFKernel(Kernel):
    """Squared Exponential / RBF kernel: k(r) = σ² exp(-r²/(2l²))."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = float(length_scale)
        self.variance = float(variance)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.length_scale, self.variance])

    @params.setter
    def params(self, v: np.ndarray):
        self.length_scale = float(v[0])
        self.variance = float(v[1])

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dist = _sq_dist(X1, X2)
        return self.variance * np.exp(-0.5 * sq_dist / self.length_scale ** 2)

    def gradient(self, X1: np.ndarray, X2: np.ndarray,
                 param_idx: int) -> np.ndarray:
        sq_dist = _sq_dist(X1, X2)
        K = self.variance * np.exp(-0.5 * sq_dist / self.length_scale ** 2)
        if param_idx == 0:  # d/d(log l)
            return K * sq_dist / self.length_scale ** 2
        elif param_idx == 1:  # d/d(log σ²)
            return K
        raise IndexError(param_idx)


class Matern52Kernel(Kernel):
    """Matern 5/2 kernel."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = float(length_scale)
        self.variance = float(variance)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.length_scale, self.variance])

    @params.setter
    def params(self, v: np.ndarray):
        self.length_scale = float(v[0])
        self.variance = float(v[1])

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        r = np.sqrt(np.maximum(_sq_dist(X1, X2), 0.0)) / self.length_scale
        return self.variance * (1.0 + math.sqrt(5) * r + 5.0 / 3.0 * r ** 2) * np.exp(-math.sqrt(5) * r)

    def gradient(self, X1: np.ndarray, X2: np.ndarray,
                 param_idx: int) -> np.ndarray:
        r = np.sqrt(np.maximum(_sq_dist(X1, X2), 0.0)) / self.length_scale
        sq5r = math.sqrt(5) * r
        base = (1.0 + sq5r + 5.0 / 3.0 * r ** 2) * np.exp(-sq5r)
        if param_idx == 0:  # d/d(log l)
            dK_dr = self.variance * np.exp(-sq5r) * (
                math.sqrt(5) + 10.0 / 3.0 * r - math.sqrt(5) * (1.0 + sq5r + 5.0 / 3.0 * r ** 2)
            )
            return dK_dr * r  # chain rule: dr/d(log l) = -r
        elif param_idx == 1:
            return self.variance * base
        raise IndexError(param_idx)


class Matern32Kernel(Kernel):
    """Matern 3/2 kernel."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = float(length_scale)
        self.variance = float(variance)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.length_scale, self.variance])

    @params.setter
    def params(self, v: np.ndarray):
        self.length_scale = float(v[0])
        self.variance = float(v[1])

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        r = np.sqrt(np.maximum(_sq_dist(X1, X2), 0.0)) / self.length_scale
        return self.variance * (1.0 + math.sqrt(3) * r) * np.exp(-math.sqrt(3) * r)

    def gradient(self, X1: np.ndarray, X2: np.ndarray,
                 param_idx: int) -> np.ndarray:
        r = np.sqrt(np.maximum(_sq_dist(X1, X2), 0.0)) / self.length_scale
        if param_idx == 0:
            return self.variance * 3.0 * r ** 2 * np.exp(-math.sqrt(3) * r)
        elif param_idx == 1:
            return self.variance * (1.0 + math.sqrt(3) * r) * np.exp(-math.sqrt(3) * r)
        raise IndexError(param_idx)


class PeriodicKernel(Kernel):
    """Periodic kernel: k = σ² exp(-2 sin²(π|x-x'|/p) / l²)."""

    def __init__(self, length_scale: float = 1.0, period: float = 1.0,
                 variance: float = 1.0):
        self.length_scale = float(length_scale)
        self.period = float(period)
        self.variance = float(variance)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.length_scale, self.period, self.variance])

    @params.setter
    def params(self, v: np.ndarray):
        self.length_scale = float(v[0])
        self.period = float(v[1])
        self.variance = float(v[2])

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dist = _euclidean_dist(X1, X2)
        sin2 = np.sin(math.pi * dist / self.period) ** 2
        return self.variance * np.exp(-2.0 * sin2 / self.length_scale ** 2)

    def gradient(self, X1: np.ndarray, X2: np.ndarray,
                 param_idx: int) -> np.ndarray:
        dist = _euclidean_dist(X1, X2)
        sin2 = np.sin(math.pi * dist / self.period) ** 2
        K = self.variance * np.exp(-2.0 * sin2 / self.length_scale ** 2)
        if param_idx == 0:  # d/d(log l)
            return K * 4.0 * sin2 / self.length_scale ** 2
        elif param_idx == 1:  # d/d(log p)
            sin_cos = np.sin(math.pi * dist / self.period) * np.cos(math.pi * dist / self.period)
            return K * (4.0 * math.pi * dist / self.period) * sin_cos / self.length_scale ** 2
        elif param_idx == 2:
            return K
        raise IndexError(param_idx)


class RationalQuadraticKernel(Kernel):
    """Rational Quadratic kernel: k = σ²(1 + r²/(2αl²))^(-α)."""

    def __init__(self, length_scale: float = 1.0, alpha: float = 1.0,
                 variance: float = 1.0):
        self.length_scale = float(length_scale)
        self.alpha = float(alpha)
        self.variance = float(variance)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.length_scale, self.alpha, self.variance])

    @params.setter
    def params(self, v: np.ndarray):
        self.length_scale = float(v[0])
        self.alpha = float(v[1])
        self.variance = float(v[2])

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dist = _sq_dist(X1, X2)
        return self.variance * (1.0 + sq_dist / (2.0 * self.alpha * self.length_scale ** 2)) ** (-self.alpha)

    def gradient(self, X1: np.ndarray, X2: np.ndarray,
                 param_idx: int) -> np.ndarray:
        sq_dist = _sq_dist(X1, X2)
        t = 1.0 + sq_dist / (2.0 * self.alpha * self.length_scale ** 2)
        K = self.variance * t ** (-self.alpha)
        if param_idx == 0:
            return K * sq_dist / (self.length_scale ** 2 * t)
        elif param_idx == 1:
            return K * (0.5 * sq_dist / (self.alpha * self.length_scale ** 2 * t) - np.log(t))
        elif param_idx == 2:
            return K
        raise IndexError(param_idx)


class LinearKernel(Kernel):
    """Linear kernel: k = σ_b² + σ_v²(x-c)(x'-c)."""

    def __init__(self, variance: float = 1.0, bias: float = 0.0,
                 offset: float = 0.0):
        self.variance = float(variance)
        self.bias = float(bias)
        self.offset = float(offset)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.variance, self.bias, self.offset])

    @params.setter
    def params(self, v: np.ndarray):
        self.variance = float(v[0])
        self.bias = float(v[1])
        self.offset = float(v[2])

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1c = X1 - self.offset
        X2c = X2 - self.offset
        return self.bias ** 2 + self.variance * (X1c @ X2c.T)

    def gradient(self, X1: np.ndarray, X2: np.ndarray,
                 param_idx: int) -> np.ndarray:
        X1c = X1 - self.offset
        X2c = X2 - self.offset
        if param_idx == 0:
            return self.variance * (X1c @ X2c.T)
        elif param_idx == 1:
            return np.full((len(X1), len(X2)), 2.0 * self.bias ** 2)
        elif param_idx == 2:
            return -self.variance * (X1c @ np.ones_like(X2c).T + np.ones_like(X1c) @ X2c.T)
        raise IndexError(param_idx)


class SpectralMixtureKernel(Kernel):
    """Spectral Mixture kernel (Wilson & Adams 2013) for 1-D inputs."""

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        rng = np.random.default_rng(42)
        self.weights = rng.uniform(0.1, 1.0, n_components)
        self.means = rng.uniform(0.0, 1.0, n_components)
        self.scales = rng.uniform(0.5, 2.0, n_components)

    @property
    def params(self) -> np.ndarray:
        return np.concatenate([self.weights, self.means, self.scales])

    @params.setter
    def params(self, v: np.ndarray):
        q = self.n_components
        self.weights = v[:q]
        self.means = v[q:2 * q]
        self.scales = v[2 * q:3 * q]

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = X1.ravel()
        X2 = X2.ravel()
        tau = X1[:, None] - X2[None, :]
        K = np.zeros((len(X1), len(X2)))
        for w, mu, s in zip(self.weights, self.means, self.scales):
            K += w * np.exp(-2.0 * math.pi ** 2 * tau ** 2 * s ** 2) * np.cos(2.0 * math.pi * tau * mu)
        return K

    def gradient(self, X1: np.ndarray, X2: np.ndarray,
                 param_idx: int) -> np.ndarray:
        X1 = X1.ravel()
        X2 = X2.ravel()
        tau = X1[:, None] - X2[None, :]
        q = self.n_components
        if param_idx < q:  # d/d weight_q
            i = param_idx
            mu, s = self.means[i], self.scales[i]
            return self.weights[i] * np.exp(-2 * math.pi ** 2 * tau ** 2 * s ** 2) * np.cos(2 * math.pi * tau * mu)
        elif param_idx < 2 * q:
            i = param_idx - q
            w, mu, s = self.weights[i], self.means[i], self.scales[i]
            env = np.exp(-2 * math.pi ** 2 * tau ** 2 * s ** 2)
            return -w * env * 2 * math.pi * tau * np.sin(2 * math.pi * tau * mu) * mu
        elif param_idx < 3 * q:
            i = param_idx - 2 * q
            w, mu, s = self.weights[i], self.means[i], self.scales[i]
            env = np.exp(-2 * math.pi ** 2 * tau ** 2 * s ** 2)
            cos_part = np.cos(2 * math.pi * tau * mu)
            return w * env * (-4 * math.pi ** 2 * tau ** 2 * s) * cos_part * s
        raise IndexError(param_idx)


# ---------------------------------------------------------------------------
# Helper distance functions
# ---------------------------------------------------------------------------

def _sq_dist(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix."""
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    d1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    d2 = np.sum(X2 ** 2, axis=1, keepdims=True)
    return np.maximum(d1 + d2.T - 2.0 * (X1 @ X2.T), 0.0)


def _euclidean_dist(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    return np.sqrt(_sq_dist(X1, X2))


# ---------------------------------------------------------------------------
# GP Regressor
# ---------------------------------------------------------------------------

@dataclass
class GPRegressor:
    """
    Gaussian Process Regressor with marginal likelihood hyperparameter
    optimization (L-BFGS-B on log-scale parameters).
    """
    kernel: Kernel
    noise_var: float = 1e-3
    normalize_y: bool = True
    max_iter: int = 200

    # Fitted state
    X_train: Optional[np.ndarray] = field(default=None, repr=False)
    y_train: Optional[np.ndarray] = field(default=None, repr=False)
    _alpha: Optional[np.ndarray] = field(default=None, repr=False)
    _L: Optional[np.ndarray] = field(default=None, repr=False)
    _y_mean: float = field(default=0.0, repr=False)
    _y_std: float = field(default=1.0, repr=False)

    def _build_K(self, X: np.ndarray, noise: bool = True) -> np.ndarray:
        K = self.kernel(X, X)
        if noise:
            K += self.noise_var * np.eye(len(X))
        return K

    def _log_marginal_likelihood(self, log_params: np.ndarray) -> float:
        """Negative log marginal likelihood for optimization."""
        params = np.exp(log_params[:-1])
        noise_var = float(np.exp(log_params[-1]))
        self.kernel.params = params
        self.noise_var = noise_var
        n = len(self.X_train)
        K = self._build_K(self.X_train, noise=True)
        try:
            L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
        except np.linalg.LinAlgError:
            return 1e10
        alpha = cho_solve((L, True), self._y_norm)
        lml = (
            -0.5 * self._y_norm @ alpha
            - np.sum(np.log(np.diag(L)))
            - 0.5 * n * math.log(2 * math.pi)
        )
        return -float(lml)

    def _lml_gradient(self, log_params: np.ndarray) -> np.ndarray:
        """Analytic gradient of negative LML wrt log parameters."""
        params = np.exp(log_params[:-1])
        noise_var = float(np.exp(log_params[-1]))
        self.kernel.params = params
        self.noise_var = noise_var
        n = len(self.X_train)
        K = self._build_K(self.X_train, noise=True)
        try:
            L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
        except np.linalg.LinAlgError:
            return np.zeros_like(log_params)
        alpha = cho_solve((L, True), self._y_norm)
        K_inv = cho_solve((L, True), np.eye(n))
        outer = np.outer(alpha, alpha) - K_inv

        grads = []
        for i in range(len(params)):
            dK = self.kernel.gradient(self.X_train, self.X_train, i)
            grads.append(-0.5 * np.trace(outer @ dK) * np.exp(log_params[i]))
        # noise gradient
        dK_noise = noise_var * np.eye(n)
        grads.append(-0.5 * np.trace(outer @ dK_noise))
        return np.array(grads)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPRegressor":
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float)
        self.X_train = X
        if self.normalize_y:
            self._y_mean = float(y.mean())
            self._y_std = float(y.std()) or 1.0
            self._y_norm = (y - self._y_mean) / self._y_std
        else:
            self._y_norm = y.copy()
        self.y_train = y

        log_params0 = np.log(np.concatenate([self.kernel.params, [self.noise_var]]))
        bounds = [(-5, 5)] * len(log_params0)
        res = minimize(
            self._log_marginal_likelihood,
            log_params0,
            jac=self._lml_gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-9},
        )
        opt_params = np.exp(res.x)
        self.kernel.params = opt_params[:-1]
        self.noise_var = float(opt_params[-1])

        K = self._build_K(self.X_train, noise=True)
        n = len(X)
        L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
        self._L = L
        self._alpha = cho_solve((L, True), self._y_norm)
        return self

    def predict(self, X_test: np.ndarray, return_std: bool = True
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_test = np.atleast_2d(X_test)
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test)
        mu = K_s.T @ self._alpha
        v = cho_solve((self._L, True), K_s)
        var = np.diag(K_ss) - np.einsum("ij,ij->j", K_s, v)
        var = np.maximum(var, 0.0)

        if self.normalize_y:
            mu = mu * self._y_std + self._y_mean
            var = var * self._y_std ** 2

        if return_std:
            return mu, np.sqrt(var)
        return mu, None

    def confidence_interval(self, X_test: np.ndarray,
                            alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, lower, upper) at confidence level 1-alpha."""
        from scipy.stats import norm
        z = norm.ppf(1.0 - alpha / 2.0)
        mu, std = self.predict(X_test)
        return mu, mu - z * std, mu + z * std

    def loocv_score(self) -> float:
        """Leave-one-out cross-validation log predictive density."""
        K = self._build_K(self.X_train, noise=True)
        n = len(self.X_train)
        K_inv = cho_solve((self._L, True), np.eye(n))
        diag_inv = np.diag(K_inv)
        loo_mean = self._y_norm - self._alpha / diag_inv
        loo_var = 1.0 / diag_inv
        lpd = -0.5 * np.log(2 * math.pi * loo_var) - 0.5 * (self._y_norm - loo_mean) ** 2 / loo_var
        return float(np.mean(lpd))


# ---------------------------------------------------------------------------
# GP Classifier (Laplace approximation)
# ---------------------------------------------------------------------------

class GPClassifier:
    """
    Binary GP classifier using Laplace approximation.
    Likelihood: Bernoulli with logistic link.
    """

    def __init__(self, kernel: Kernel, max_newton_iter: int = 20,
                 tol: float = 1e-6):
        self.kernel = kernel
        self.max_newton_iter = max_newton_iter
        self.tol = tol
        self.X_train: Optional[np.ndarray] = None
        self._f_hat: Optional[np.ndarray] = None
        self._L: Optional[np.ndarray] = None
        self._K: Optional[np.ndarray] = None

    def _newton_mode_find(self, K: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(y)
        f = np.zeros(n)
        for _ in range(self.max_newton_iter):
            pi = expit(f)
            W = pi * (1 - pi)
            W_sqrt = np.sqrt(W)
            B = np.eye(n) + W_sqrt[:, None] * K * W_sqrt[None, :]
            L = np.linalg.cholesky(B + 1e-8 * np.eye(n))
            b = W * f + (y - pi)
            a = b - W_sqrt * cho_solve((L, True), W_sqrt * (K @ b))
            f_new = K @ a
            if np.max(np.abs(f_new - f)) < self.tol:
                f = f_new
                break
            f = f_new
        return f, L, W, pi

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPClassifier":
        self.X_train = np.atleast_2d(X)
        y = np.asarray(y, dtype=float)
        self.y_train = y
        self._K = self.kernel(self.X_train, self.X_train)
        n = len(y)
        self._f_hat, self._L_lap, self._W, _ = self._newton_mode_find(
            self._K + 1e-8 * np.eye(n), y
        )
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X_test = np.atleast_2d(X_test)
        n = len(self.X_train)
        K_s = self.kernel(self.X_train, X_test)
        pi_hat = expit(self._f_hat)
        W = self._W
        W_sqrt = np.sqrt(W)
        f_mean = K_s.T @ (self.y_train - pi_hat)
        L = self._L_lap
        v = cho_solve((L, True), W_sqrt[:, None] * K_s)
        K_ss_diag = np.sum(self.kernel(X_test, X_test) * np.eye(len(X_test)), axis=1)
        var_f = K_ss_diag - np.sum(v ** 2, axis=0)
        # Sigmoid integration via probit approximation
        kappa = 1.0 / np.sqrt(1.0 + math.pi * var_f / 8.0)
        return expit(kappa * f_mean)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X_test) >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Sparse GP (FITC approximation)
# ---------------------------------------------------------------------------

class SparseGPRegressor:
    """
    Sparse GP using FITC (Fully Independent Training Conditional) approximation.
    Inducing points Z are either provided or selected via k-means.
    """

    def __init__(self, kernel: Kernel, n_inducing: int = 50,
                 noise_var: float = 1e-2, normalize_y: bool = True):
        self.kernel = kernel
        self.n_inducing = n_inducing
        self.noise_var = noise_var
        self.normalize_y = normalize_y
        self.Z: Optional[np.ndarray] = None
        self._m: Optional[np.ndarray] = None
        self._S: Optional[np.ndarray] = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def _select_inducing(self, X: np.ndarray) -> np.ndarray:
        """Simple greedy selection of inducing points via k-means."""
        n = len(X)
        m = min(self.n_inducing, n)
        idx = np.random.choice(n, size=m, replace=False)
        centers = X[idx].copy()
        for _ in range(20):
            dists = _sq_dist(X, centers)
            assignments = np.argmin(dists, axis=1)
            new_centers = np.array([X[assignments == k].mean(axis=0)
                                    if np.any(assignments == k) else centers[k]
                                    for k in range(m)])
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers
        return centers

    def fit(self, X: np.ndarray, y: np.ndarray,
            Z: Optional[np.ndarray] = None) -> "SparseGPRegressor":
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float)
        self.X_train = X
        if self.normalize_y:
            self._y_mean = float(y.mean())
            self._y_std = float(y.std()) or 1.0
            y_norm = (y - self._y_mean) / self._y_std
        else:
            y_norm = y.copy()

        self.Z = np.atleast_2d(Z) if Z is not None else self._select_inducing(X)
        m = len(self.Z)
        n = len(X)

        Kmm = self.kernel(self.Z, self.Z) + 1e-6 * np.eye(m)
        Knm = self.kernel(X, self.Z)
        Knn_diag = np.array([self.kernel(X[i:i+1], X[i:i+1])[0, 0] for i in range(n)])
        Qnn_diag = np.sum(Knm @ np.linalg.inv(Kmm) * Knm, axis=1)
        Lambda_diag = Knn_diag - Qnn_diag + self.noise_var

        Lambda_inv_diag = 1.0 / Lambda_diag
        Lmm = np.linalg.cholesky(Kmm)

        # Posterior over inducing outputs
        Kmm_inv = np.linalg.inv(Kmm)
        A = Kmm + (Knm * Lambda_inv_diag[:, None]).T @ Knm
        A += 1e-8 * np.eye(m)
        LA = np.linalg.cholesky(A)
        m_vec = cho_solve((LA, True), Knm.T @ (Lambda_inv_diag * y_norm))
        self._m = Kmm @ m_vec  # posterior mean at inducing
        self._LA = LA
        self._Kmm = Kmm
        self._Knm_full = Knm
        self._Lambda_inv_diag = Lambda_inv_diag
        self.y_norm = y_norm
        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_test = np.atleast_2d(X_test)
        Ksm = self.kernel(X_test, self.Z)
        Kmm_inv = np.linalg.inv(self._Kmm)
        mu = Ksm @ Kmm_inv @ self._m
        # Posterior variance (diagonal only)
        Kss_diag = np.array([self.kernel(X_test[i:i+1], X_test[i:i+1])[0, 0]
                              for i in range(len(X_test))])
        Qss_diag = np.sum(Ksm @ Kmm_inv * Ksm, axis=1)
        S_diag = np.sum(Ksm @ cho_solve((self._LA, True), Ksm.T).T, axis=1)
        var = Kss_diag - Qss_diag + S_diag
        var = np.maximum(var, 0.0)

        if self.normalize_y:
            mu = mu * self._y_std + self._y_mean
            var = var * self._y_std ** 2
        return mu, np.sqrt(var)


# ---------------------------------------------------------------------------
# Multi-Output GP
# ---------------------------------------------------------------------------

class MultiOutputGP:
    """
    Multi-output GP using the Independent Output (factored) model.
    Each output has its own GP regressor with shared or independent kernels.
    """

    def __init__(self, kernels: List[Kernel], noise_var: float = 1e-3):
        self.gps = [GPRegressor(k, noise_var=noise_var) for k in kernels]
        self.n_outputs = len(kernels)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultiOutputGP":
        """Y shape: (n_samples, n_outputs)."""
        assert Y.shape[1] == self.n_outputs
        for i, gp in enumerate(self.gps):
            gp.fit(X, Y[:, i])
        return self

    def predict(self, X_test: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        mus, stds = [], []
        for gp in self.gps:
            mu, std = gp.predict(X_test)
            mus.append(mu)
            stds.append(std)
        return np.column_stack(mus), np.column_stack(stds)


# ---------------------------------------------------------------------------
# Volatility GP
# ---------------------------------------------------------------------------

class VolatilityGP:
    """
    GP for volatility forecasting.
    Input: log-realized-volatility window.
    Output: next-period log-volatility.
    Uses Matern-5/2 kernel by default (captures rough vol dynamics).
    """

    def __init__(self, lookback: int = 10, horizon: int = 1,
                 kernel: Optional[Kernel] = None,
                 noise_var: float = 1e-3):
        self.lookback = lookback
        self.horizon = horizon
        kernel = kernel or Matern52Kernel(length_scale=1.0, variance=1.0)
        self.gp = GPRegressor(kernel=kernel, noise_var=noise_var,
                              normalize_y=True)

    def _make_features(self, log_vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(log_vol)
        X, y = [], []
        for i in range(self.lookback, n - self.horizon + 1):
            X.append(log_vol[i - self.lookback: i])
            y.append(log_vol[i + self.horizon - 1])
        return np.array(X), np.array(y)

    def fit(self, vol_series: np.ndarray) -> "VolatilityGP":
        """vol_series: raw realized volatility (positive)."""
        log_vol = np.log(np.maximum(vol_series, 1e-8))
        X, y = self._make_features(log_vol)
        self.gp.fit(X, y)
        self._last_log_vol = log_vol
        return self

    def forecast(self, recent_vol: Optional[np.ndarray] = None
                 ) -> Tuple[float, float, float]:
        """
        Returns (forecast_vol, lower_95, upper_95) in original vol space.
        recent_vol: last `lookback` vol observations (raw). If None, uses
                    the tail of the training series.
        """
        if recent_vol is not None:
            x = np.log(np.maximum(recent_vol[-self.lookback:], 1e-8))
        else:
            x = self._last_log_vol[-self.lookback:]
        X_test = x[None, :]
        mu, lower, upper = self.gp.confidence_interval(X_test, alpha=0.05)
        return float(np.exp(mu[0])), float(np.exp(lower[0])), float(np.exp(upper[0]))

    def loocv(self) -> float:
        return self.gp.loocv_score()


# ---------------------------------------------------------------------------
# GP model selection via LOOCV
# ---------------------------------------------------------------------------

def gp_model_selection(kernels: List[Kernel], X: np.ndarray,
                       y: np.ndarray, noise_var: float = 1e-3) -> Tuple[int, List[float]]:
    """
    Select best kernel via LOOCV log predictive density.
    Returns (best_idx, list_of_scores).
    """
    scores = []
    for k in kernels:
        gp = GPRegressor(kernel=k, noise_var=noise_var)
        try:
            gp.fit(X, y)
            scores.append(gp.loocv_score())
        except Exception:
            scores.append(-np.inf)
    best = int(np.argmax(scores))
    return best, scores
