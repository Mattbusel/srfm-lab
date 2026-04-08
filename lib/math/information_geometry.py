"""
information_geometry.py -- Information geometry for financial distributions.

Fisher information, natural gradients, divergences (KL, Wasserstein, alpha),
exponential families, geodesics, Amari-Chentsov tensor, and applications to
portfolio optimization and regime detection.

All numerics via numpy/scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats, linalg, special
from scipy.optimize import minimize, linear_sum_assignment

FloatArray = NDArray[np.float64]


# ===================================================================
# 1.  Fisher Information Matrix
# ===================================================================

def fisher_information_normal(mu: float, sigma: float) -> FloatArray:
    """Fisher information matrix for univariate normal N(mu, sigma^2).
    Parameterized by (mu, sigma).  FIM is diag(1/sigma^2, 2/sigma^2)."""
    return np.array([
        [1.0 / sigma ** 2, 0.0],
        [0.0, 2.0 / sigma ** 2],
    ])


def fisher_information_multivariate_normal(
    mu: FloatArray, Sigma: FloatArray
) -> FloatArray:
    """Fisher information for multivariate normal w.r.t. mean parameters.
    Returns FIM for the mean part only: Sigma^{-1}."""
    return np.linalg.inv(Sigma + np.eye(len(mu)) * 1e-10)


def fisher_information_exponential(lam: float) -> FloatArray:
    """Fisher information for Exp(lambda): 1/lambda^2."""
    return np.array([[1.0 / (lam ** 2 + 1e-12)]])


def fisher_information_gamma(alpha: float, beta: float) -> FloatArray:
    """Fisher information for Gamma(alpha, beta)."""
    psi1 = float(special.polygamma(1, alpha))
    return np.array([
        [psi1, -1.0 / beta],
        [-1.0 / beta, alpha / (beta ** 2)],
    ])


def fisher_information_bernoulli(p: float) -> FloatArray:
    """Fisher information for Bernoulli(p): 1/(p(1-p))."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.array([[1.0 / (p * (1 - p))]])


def fisher_information_categorical(probs: FloatArray) -> FloatArray:
    """Fisher information for Categorical distribution on the simplex.
    FIM_{ij} = delta_{ij}/p_i (using embedded coordinates)."""
    probs = np.clip(probs, 1e-10, None)
    return np.diag(1.0 / probs)


def fisher_information_student_t(nu: float, sigma: float) -> FloatArray:
    """Approximate Fisher information for Student-t with df=nu, scale=sigma.
    For the location-scale parameterization."""
    fim_mu_mu = (nu + 1) / ((nu + 3) * sigma ** 2 + 1e-12)
    fim_sigma_sigma = 2.0 * nu / ((nu + 3) * sigma ** 2 + 1e-12)
    return np.array([
        [fim_mu_mu, 0.0],
        [0.0, fim_sigma_sigma],
    ])


def fisher_information_numerical(
    log_likelihood: Callable[[FloatArray, FloatArray], float],
    theta: FloatArray,
    data: FloatArray,
    eps: float = 1e-5,
) -> FloatArray:
    """Numerically estimate FIM via finite differences of the log-likelihood.
    FIM = -E[H(log p)], approximated by sample Hessian."""
    d = len(theta)
    H = np.zeros((d, d))
    ll0 = log_likelihood(theta, data)
    for i in range(d):
        for j in range(i, d):
            e_i = np.zeros(d)
            e_j = np.zeros(d)
            e_i[i] = eps
            e_j[j] = eps
            ll_pp = log_likelihood(theta + e_i + e_j, data)
            ll_pm = log_likelihood(theta + e_i - e_j, data)
            ll_mp = log_likelihood(theta - e_i + e_j, data)
            ll_mm = log_likelihood(theta - e_i - e_j, data)
            H[i, j] = (ll_pp - ll_pm - ll_mp + ll_mm) / (4 * eps ** 2)
            H[j, i] = H[i, j]
    return -H


# ===================================================================
# 2.  Natural Gradient Descent
# ===================================================================

class NaturalGradientOptimizer:
    """Natural gradient descent on statistical manifolds."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        damping: float = 1e-4,
    ):
        self.lr = learning_rate
        self.damping = damping

    def step(
        self,
        theta: FloatArray,
        grad: FloatArray,
        fim: FloatArray,
    ) -> FloatArray:
        """One natural gradient step: theta -= lr * FIM^{-1} @ grad."""
        n = len(theta)
        fim_reg = fim + self.damping * np.eye(n)
        try:
            nat_grad = np.linalg.solve(fim_reg, grad)
        except np.linalg.LinAlgError:
            nat_grad = grad
        return theta - self.lr * nat_grad

    def optimize(
        self,
        theta0: FloatArray,
        loss_fn: Callable[[FloatArray], float],
        grad_fn: Callable[[FloatArray], FloatArray],
        fim_fn: Callable[[FloatArray], FloatArray],
        n_steps: int = 100,
        callback: Callable[[int, FloatArray, float], None] | None = None,
    ) -> Tuple[FloatArray, List[float]]:
        theta = theta0.copy()
        losses = []
        for step in range(n_steps):
            loss = loss_fn(theta)
            losses.append(loss)
            grad = grad_fn(theta)
            fim = fim_fn(theta)
            theta = self.step(theta, grad, fim)
            if callback:
                callback(step, theta, loss)
        return theta, losses


# ===================================================================
# 3.  Kullback-Leibler Divergence
# ===================================================================

def kl_divergence_normal(
    mu1: float, sigma1: float, mu2: float, sigma2: float
) -> float:
    """KL(N(mu1,sigma1^2) || N(mu2,sigma2^2))."""
    return (
        np.log(sigma2 / sigma1)
        + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2)
        - 0.5
    )


def kl_divergence_multivariate_normal(
    mu1: FloatArray, Sigma1: FloatArray,
    mu2: FloatArray, Sigma2: FloatArray,
) -> float:
    """KL(N(mu1,Sigma1) || N(mu2,Sigma2))."""
    k = len(mu1)
    Sigma2_inv = np.linalg.inv(Sigma2 + np.eye(k) * 1e-10)
    diff = mu2 - mu1
    term1 = np.trace(Sigma2_inv @ Sigma1)
    term2 = diff @ Sigma2_inv @ diff
    term3 = np.log(np.linalg.det(Sigma2) / (np.linalg.det(Sigma1) + 1e-30) + 1e-30)
    return 0.5 * (term1 + term2 - k + term3)


def kl_divergence_categorical(p: FloatArray, q: FloatArray) -> float:
    """KL(p || q) for discrete distributions."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return float(np.sum(p * np.log(p / q)))


def kl_symmetric(p: FloatArray, q: FloatArray) -> float:
    """Symmetric KL (Jensen-Shannon-like): (KL(p||q) + KL(q||p))/2."""
    return 0.5 * (kl_divergence_categorical(p, q) + kl_divergence_categorical(q, p))


def kl_divergence_empirical(
    samples_p: FloatArray, samples_q: FloatArray, k: int = 5
) -> float:
    """KNN-based KL divergence estimate between two sample sets."""
    from scipy.spatial import KDTree
    n_p = len(samples_p)
    n_q = len(samples_q)
    d = samples_p.shape[1] if samples_p.ndim > 1 else 1
    if d == 1:
        samples_p = samples_p.reshape(-1, 1)
        samples_q = samples_q.reshape(-1, 1)
    tree_p = KDTree(samples_p)
    tree_q = KDTree(samples_q)
    r_k = tree_p.query(samples_p, k=k + 1)[0][:, -1]
    s_k = tree_q.query(samples_p, k=k)[0][:, -1]
    r_k = np.maximum(r_k, 1e-12)
    s_k = np.maximum(s_k, 1e-12)
    return float(d * np.mean(np.log(s_k / r_k)) + np.log(n_q / (n_p - 1)))


# ===================================================================
# 4.  Wasserstein Distance
# ===================================================================

def wasserstein_1d(p_samples: FloatArray, q_samples: FloatArray) -> float:
    """Exact 1D Wasserstein-1 distance via sorted quantile matching."""
    p_sorted = np.sort(p_samples)
    q_sorted = np.sort(q_samples)
    n = min(len(p_sorted), len(q_sorted))
    p_q = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(p_sorted)), p_sorted)
    q_q = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(q_sorted)), q_sorted)
    return float(np.mean(np.abs(p_q - q_q)))


def wasserstein_2_1d(p_samples: FloatArray, q_samples: FloatArray) -> float:
    """Exact 1D Wasserstein-2 distance."""
    p_sorted = np.sort(p_samples)
    q_sorted = np.sort(q_samples)
    n = min(len(p_sorted), len(q_sorted))
    p_q = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(p_sorted)), p_sorted)
    q_q = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(q_sorted)), q_sorted)
    return float(np.sqrt(np.mean((p_q - q_q) ** 2)))


def wasserstein_2_normal(
    mu1: FloatArray, Sigma1: FloatArray,
    mu2: FloatArray, Sigma2: FloatArray,
) -> float:
    """Wasserstein-2 between multivariate normals (Bures-Wasserstein)."""
    diff_mu = np.linalg.norm(mu1 - mu2) ** 2
    sqrt_S1 = linalg.sqrtm(Sigma1)
    M = linalg.sqrtm(sqrt_S1 @ Sigma2 @ sqrt_S1)
    if np.iscomplex(M).any():
        M = np.real(M)
    trace_term = np.trace(Sigma1) + np.trace(Sigma2) - 2 * np.trace(M)
    return float(np.sqrt(max(diff_mu + trace_term, 0.0)))


def sinkhorn_divergence(
    X: FloatArray, Y: FloatArray, eps: float = 0.1, n_iter: int = 100
) -> float:
    """Sinkhorn approximation to Wasserstein distance for 2D+ point clouds."""
    n = len(X)
    m = len(Y)
    C = np.zeros((n, m))
    for i in range(n):
        C[i] = np.linalg.norm(X[i] - Y, axis=1) ** 2
    K = np.exp(-C / eps)
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(n_iter):
        u = 1.0 / (K @ v + 1e-12)
        v = 1.0 / (K.T @ u + 1e-12)
    transport = np.diag(u) @ K @ np.diag(v)
    return float(np.sum(transport * C))


# ===================================================================
# 5.  Alpha-divergence family
# ===================================================================

def renyi_divergence(p: FloatArray, q: FloatArray, alpha: float = 0.5) -> float:
    """Renyi divergence of order alpha."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    if abs(alpha - 1.0) < 1e-8:
        return kl_divergence_categorical(p, q)
    integral = np.sum(p ** alpha * q ** (1 - alpha))
    return float(np.log(integral + 1e-30) / (alpha - 1))


def tsallis_divergence(p: FloatArray, q: FloatArray, alpha: float = 0.5) -> float:
    """Tsallis divergence (generalized entropy divergence)."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    if abs(alpha - 1.0) < 1e-8:
        return kl_divergence_categorical(p, q)
    integral = np.sum(p ** alpha * q ** (1 - alpha))
    return float((1.0 - integral) / (alpha - 1))


def alpha_divergence(p: FloatArray, q: FloatArray, alpha: float = 0.5) -> float:
    """Amari alpha-divergence."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    if abs(alpha - 1.0) < 1e-8:
        return kl_divergence_categorical(p, q)
    if abs(alpha + 1.0) < 1e-8:
        return kl_divergence_categorical(q, p)
    a = (1 - alpha) / 2.0
    b = (1 + alpha) / 2.0
    integral = np.sum(p ** a * q ** b)
    return float(4.0 / (1.0 - alpha ** 2) * (1.0 - integral))


def f_divergence(p: FloatArray, q: FloatArray, f: Callable[[float], float]) -> float:
    """General f-divergence: D_f(p||q) = sum q_i * f(p_i/q_i)."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    ratio = p / q
    return float(np.sum(q * np.vectorize(f)(ratio)))


def hellinger_distance(p: FloatArray, q: FloatArray) -> float:
    """Hellinger distance: sqrt(1 - BC(p,q))."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    bc = np.sum(np.sqrt(p * q))
    return float(np.sqrt(max(1.0 - bc, 0.0)))


# ===================================================================
# 6.  Exponential Family
# ===================================================================

@dataclass
class ExponentialFamily:
    """Exponential family: p(x|theta) = h(x) * exp(eta(theta)^T T(x) - A(theta))."""
    name: str
    natural_params: FloatArray          # eta
    sufficient_stats_fn: Callable[[FloatArray], FloatArray]
    log_partition_fn: Callable[[FloatArray], float]
    base_measure_fn: Callable[[FloatArray], FloatArray]

    def log_likelihood(self, data: FloatArray) -> float:
        T = self.sufficient_stats_fn(data)
        A = self.log_partition_fn(self.natural_params)
        h = self.base_measure_fn(data)
        return float(np.sum(self.natural_params @ T.T) - A * len(data) + np.sum(np.log(h + 1e-30)))

    def expected_sufficient_stats(self, data: FloatArray) -> FloatArray:
        return self.sufficient_stats_fn(data).mean(axis=0)

    def fisher_information(self, eps: float = 1e-5) -> FloatArray:
        """FIM = Hessian of log-partition function."""
        d = len(self.natural_params)
        H = np.zeros((d, d))
        A0 = self.log_partition_fn(self.natural_params)
        for i in range(d):
            for j in range(i, d):
                e_i = np.zeros(d); e_i[i] = eps
                e_j = np.zeros(d); e_j[j] = eps
                A_pp = self.log_partition_fn(self.natural_params + e_i + e_j)
                A_pm = self.log_partition_fn(self.natural_params + e_i - e_j)
                A_mp = self.log_partition_fn(self.natural_params - e_i + e_j)
                A_mm = self.log_partition_fn(self.natural_params - e_i - e_j)
                H[i, j] = (A_pp - A_pm - A_mp + A_mm) / (4 * eps ** 2)
                H[j, i] = H[i, j]
        return H


def normal_exponential_family(mu: float, sigma: float) -> ExponentialFamily:
    """Normal distribution as exponential family."""
    eta = np.array([mu / sigma ** 2, -1.0 / (2 * sigma ** 2)])

    def suff_stats(x: FloatArray) -> FloatArray:
        return np.column_stack([x, x ** 2])

    def log_partition(eta: FloatArray) -> float:
        return float(-eta[0] ** 2 / (4 * eta[1]) - 0.5 * np.log(-2 * eta[1] + 1e-12))

    def base_measure(x: FloatArray) -> FloatArray:
        return np.ones(len(x)) / np.sqrt(2 * np.pi)

    return ExponentialFamily("normal", eta, suff_stats, log_partition, base_measure)


def exponential_exponential_family(lam: float) -> ExponentialFamily:
    """Exponential distribution as exponential family."""
    eta = np.array([-lam])

    def suff_stats(x: FloatArray) -> FloatArray:
        return x.reshape(-1, 1)

    def log_partition(eta: FloatArray) -> float:
        return float(-np.log(-eta[0] + 1e-12))

    def base_measure(x: FloatArray) -> FloatArray:
        return np.ones(len(x))

    return ExponentialFamily("exponential", eta, suff_stats, log_partition, base_measure)


def poisson_exponential_family(lam: float) -> ExponentialFamily:
    """Poisson as exponential family."""
    eta = np.array([np.log(lam + 1e-12)])

    def suff_stats(x: FloatArray) -> FloatArray:
        return x.reshape(-1, 1)

    def log_partition(eta: FloatArray) -> float:
        return float(np.exp(eta[0]))

    def base_measure(x: FloatArray) -> FloatArray:
        return 1.0 / (special.factorial(x) + 1e-12)

    return ExponentialFamily("poisson", eta, suff_stats, log_partition, base_measure)


# ===================================================================
# 7.  Information Projection
# ===================================================================

def information_projection_simplex(
    target: FloatArray,
    constraint_fn: Callable[[FloatArray], FloatArray],
    constraint_values: FloatArray,
    n_iter: int = 200,
    lr: float = 0.01,
) -> FloatArray:
    """I-projection: find closest distribution in exponential family
    (on simplex) to target distribution subject to moment constraints.

    Minimizes KL(q || target) s.t. E_q[constraint_fn] = constraint_values.
    """
    n = len(target)
    log_q = np.log(target + 1e-12).copy()
    lam = np.zeros(len(constraint_values))

    for _ in range(n_iter):
        q = np.exp(log_q)
        q /= q.sum()
        # Constraint violation
        c_vals = constraint_fn(q)
        violation = c_vals - constraint_values
        # Update Lagrange multipliers
        lam += lr * violation
        # Update q: q proportional to target * exp(-lam . constraint_fn)
        log_q = np.log(target + 1e-12) - lam @ constraint_fn(np.eye(n)).reshape(len(lam), -1)
        log_q -= log_q.max()
    q = np.exp(log_q)
    q /= q.sum()
    return q


def m_projection_simplex(
    source: FloatArray,
    target_family: Callable[[FloatArray], FloatArray],
    theta0: FloatArray,
    n_iter: int = 200,
    lr: float = 0.01,
) -> FloatArray:
    """M-projection: find q in family closest to source.
    Minimizes KL(source || q(theta))."""
    theta = theta0.copy()
    for _ in range(n_iter):
        q = target_family(theta)
        q = np.clip(q, 1e-12, None)
        q /= q.sum()
        # Gradient of KL(source || q) w.r.t. theta (numerical)
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            eps = 1e-5
            theta_p = theta.copy(); theta_p[i] += eps
            theta_m = theta.copy(); theta_m[i] -= eps
            q_p = target_family(theta_p); q_p = np.clip(q_p, 1e-12, None); q_p /= q_p.sum()
            q_m = target_family(theta_m); q_m = np.clip(q_m, 1e-12, None); q_m /= q_m.sum()
            kl_p = kl_divergence_categorical(source, q_p)
            kl_m = kl_divergence_categorical(source, q_m)
            grad[i] = (kl_p - kl_m) / (2 * eps)
        theta -= lr * grad
    q = target_family(theta)
    q = np.clip(q, 1e-12, None)
    return q / q.sum()


# ===================================================================
# 8.  Amari-Chentsov Tensor
# ===================================================================

def amari_chentsov_tensor(probs: FloatArray) -> FloatArray:
    """Third-order Amari-Chentsov tensor for categorical distribution.
    T_{ijk} = E[d_i d_j d_k log p] on the probability simplex.
    For categorical: T_{iii} = 1/p_i^2, others involve delta functions."""
    n = len(probs)
    probs = np.clip(probs, 1e-12, None)
    T = np.zeros((n, n, n))
    for i in range(n):
        T[i, i, i] = 1.0 / probs[i] ** 2
    return T


def connection_coefficients(
    probs: FloatArray, alpha: float = 1.0
) -> FloatArray:
    """Alpha-connection coefficients for the statistical manifold.
    Gamma^{alpha}_{ijk} = E[d_i d_j log p * d_k log p] + (1-alpha)/2 * T_{ijk}."""
    n = len(probs)
    probs = np.clip(probs, 1e-12, None)
    fim = fisher_information_categorical(probs)
    T = amari_chentsov_tensor(probs)
    Gamma = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Levi-Civita part (alpha=0)
                lc = 0.0
                if i == j == k:
                    lc = -1.0 / probs[i] ** 2
                # Alpha correction
                Gamma[i, j, k] = lc + (1.0 - alpha) / 2.0 * T[i, j, k]
    return Gamma


# ===================================================================
# 9.  Geodesics on probability simplex
# ===================================================================

def geodesic_simplex_e(
    p: FloatArray, q: FloatArray, n_points: int = 50
) -> FloatArray:
    """Exponential (e-)geodesic on simplex: mixture in log space.
    gamma(t) proportional to p^{1-t} * q^t."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    t_vals = np.linspace(0, 1, n_points)
    path = np.zeros((n_points, len(p)))
    for i, t in enumerate(t_vals):
        log_gamma = (1 - t) * np.log(p) + t * np.log(q)
        gamma = np.exp(log_gamma)
        path[i] = gamma / gamma.sum()
    return path


def geodesic_simplex_m(
    p: FloatArray, q: FloatArray, n_points: int = 50
) -> FloatArray:
    """Mixture (m-)geodesic on simplex: linear interpolation.
    gamma(t) = (1-t)*p + t*q."""
    t_vals = np.linspace(0, 1, n_points)
    path = np.zeros((n_points, len(p)))
    for i, t in enumerate(t_vals):
        path[i] = (1 - t) * p + t * q
    return path


def geodesic_fisher_rao(
    p: FloatArray, q: FloatArray, n_points: int = 50
) -> FloatArray:
    """Fisher-Rao geodesic on probability simplex.
    Use the sphere representation: sqrt(p) lives on the unit sphere."""
    sp = np.sqrt(np.clip(p, 1e-12, None))
    sq = np.sqrt(np.clip(q, 1e-12, None))
    cos_d = np.clip(np.dot(sp, sq), -1, 1)
    d = np.arccos(cos_d)
    if d < 1e-10:
        return np.tile(p, (n_points, 1))
    t_vals = np.linspace(0, 1, n_points)
    path = np.zeros((n_points, len(p)))
    for i, t in enumerate(t_vals):
        interp = np.sin((1 - t) * d) / np.sin(d) * sp + np.sin(t * d) / np.sin(d) * sq
        path[i] = interp ** 2
        path[i] /= path[i].sum()
    return path


def fisher_rao_distance(p: FloatArray, q: FloatArray) -> float:
    """Fisher-Rao (Bhattacharyya) distance on simplex."""
    sp = np.sqrt(np.clip(p, 1e-12, None))
    sq = np.sqrt(np.clip(q, 1e-12, None))
    cos_d = np.clip(np.dot(sp, sq), -1, 1)
    return float(2.0 * np.arccos(cos_d))


# ===================================================================
# 10. Portfolio as point on statistical manifold
# ===================================================================

@dataclass
class PortfolioManifold:
    """Treat portfolio weights as a point on the probability simplex.
    The Fisher-Rao geometry gives a natural metric for portfolio comparison."""
    n_assets: int

    def distance(self, w1: FloatArray, w2: FloatArray) -> float:
        """Fisher-Rao distance between two portfolios."""
        w1 = np.abs(w1) / (np.abs(w1).sum() + 1e-12)
        w2 = np.abs(w2) / (np.abs(w2).sum() + 1e-12)
        return fisher_rao_distance(w1, w2)

    def geodesic(self, w1: FloatArray, w2: FloatArray, n_points: int = 20) -> FloatArray:
        """Geodesic path between portfolios."""
        w1 = np.abs(w1) / (np.abs(w1).sum() + 1e-12)
        w2 = np.abs(w2) / (np.abs(w2).sum() + 1e-12)
        return geodesic_fisher_rao(w1, w2, n_points)

    def natural_gradient_step(
        self,
        weights: FloatArray,
        returns: FloatArray,
        learning_rate: float = 0.01,
    ) -> FloatArray:
        """Natural gradient portfolio update: use Fisher information
        of the return distribution induced by the portfolio."""
        w = np.abs(weights) / (np.abs(weights).sum() + 1e-12)
        # Gradient of expected return
        grad = returns.mean(axis=0)
        fim = fisher_information_categorical(w)
        fim_reg = fim + 1e-6 * np.eye(self.n_assets)
        nat_grad = np.linalg.solve(fim_reg, grad)
        w_new = w + learning_rate * nat_grad
        w_new = np.maximum(w_new, 0)
        w_new /= w_new.sum() + 1e-12
        return w_new

    def exponential_map(self, w: FloatArray, v: FloatArray) -> FloatArray:
        """Exponential map on simplex: move from w in direction v."""
        w = np.clip(w, 1e-12, None)
        log_w = np.log(w) + v / (w + 1e-12)
        w_new = np.exp(log_w)
        w_new = np.maximum(w_new, 0)
        return w_new / (w_new.sum() + 1e-12)

    def logarithmic_map(self, w1: FloatArray, w2: FloatArray) -> FloatArray:
        """Logarithmic map: tangent vector at w1 pointing to w2."""
        w1 = np.clip(w1, 1e-12, None)
        w2 = np.clip(w2, 1e-12, None)
        return w1 * (np.log(w2) - np.log(w1))


# ===================================================================
# 11. Regime detection via geodesic distance
# ===================================================================

@dataclass
class DistributionPoint:
    """A distribution described by its sufficient statistics."""
    mean: FloatArray
    cov: FloatArray
    label: str = ""


def regime_distance_matrix(
    return_windows: List[FloatArray],
    metric: str = "wasserstein",
) -> FloatArray:
    """Compute pairwise distances between return distribution windows."""
    n = len(return_windows)
    D = np.zeros((n, n))
    for i in range(n):
        mu_i = return_windows[i].mean(axis=0)
        cov_i = np.cov(return_windows[i].T)
        if cov_i.ndim == 0:
            cov_i = np.array([[cov_i]])
        for j in range(i + 1, n):
            mu_j = return_windows[j].mean(axis=0)
            cov_j = np.cov(return_windows[j].T)
            if cov_j.ndim == 0:
                cov_j = np.array([[cov_j]])
            if metric == "wasserstein":
                d = wasserstein_2_normal(mu_i, cov_i, mu_j, cov_j)
            elif metric == "kl":
                d = 0.5 * (
                    kl_divergence_multivariate_normal(mu_i, cov_i, mu_j, cov_j)
                    + kl_divergence_multivariate_normal(mu_j, cov_j, mu_i, cov_i)
                )
            elif metric == "fisher_rao":
                # Approximate: use diagonal variances as simplex weights
                v_i = np.diag(cov_i)
                v_j = np.diag(cov_j)
                v_i = v_i / (v_i.sum() + 1e-12)
                v_j = v_j / (v_j.sum() + 1e-12)
                d = fisher_rao_distance(v_i, v_j)
            else:
                d = wasserstein_2_normal(mu_i, cov_i, mu_j, cov_j)
            D[i, j] = d
            D[j, i] = d
    return D


def detect_regime_changes(
    returns: FloatArray,
    window: int = 63,
    step: int = 21,
    threshold: float = 0.5,
    metric: str = "wasserstein",
) -> List[int]:
    """Detect regime changes by monitoring geodesic distance between
    consecutive return windows."""
    n = returns.shape[0]
    change_points = []
    prev_window = returns[:window]
    for t in range(window + step, n, step):
        curr_window = returns[t - window : t]
        mu_p = prev_window.mean(axis=0)
        cov_p = np.cov(prev_window.T)
        mu_c = curr_window.mean(axis=0)
        cov_c = np.cov(curr_window.T)
        if cov_p.ndim == 0:
            cov_p = np.array([[cov_p]])
        if cov_c.ndim == 0:
            cov_c = np.array([[cov_c]])
        if metric == "wasserstein":
            d = wasserstein_2_normal(mu_p, cov_p, mu_c, cov_c)
        else:
            d = 0.5 * (
                kl_divergence_multivariate_normal(mu_p, cov_p, mu_c, cov_c)
                + kl_divergence_multivariate_normal(mu_c, cov_c, mu_p, cov_p)
            )
        if d > threshold:
            change_points.append(t)
        prev_window = curr_window
    return change_points


# ===================================================================
# 12. Natural gradient for online portfolio optimization
# ===================================================================

class OnlineNaturalGradientPortfolio:
    """Online portfolio optimization using natural gradient on the simplex."""

    def __init__(
        self,
        n_assets: int,
        learning_rate: float = 0.01,
        regularization: float = 1e-4,
    ):
        self.n_assets = n_assets
        self.lr = learning_rate
        self.reg = regularization
        self.weights = np.ones(n_assets) / n_assets
        self._history: List[FloatArray] = []

    def update(self, returns: FloatArray) -> FloatArray:
        """Update portfolio weights given new return observation."""
        self._history.append(returns)
        w = np.clip(self.weights, 1e-8, None)
        w /= w.sum()
        # Gradient of log portfolio return
        port_ret = w @ returns
        grad = returns / (port_ret + 1e-12)
        # Fisher metric for categorical
        fim = np.diag(1.0 / (w + 1e-12))
        fim_reg = fim + self.reg * np.eye(self.n_assets)
        nat_grad = np.linalg.solve(fim_reg, grad)
        # Multiplicative update (exponentiated gradient)
        log_w = np.log(w) + self.lr * nat_grad
        log_w -= log_w.max()
        w_new = np.exp(log_w)
        w_new /= w_new.sum()
        self.weights = w_new
        return w_new

    def run(self, returns_matrix: FloatArray) -> FloatArray:
        """Run over entire return matrix, return weight history."""
        n = returns_matrix.shape[0]
        weight_history = np.zeros((n, self.n_assets))
        for t in range(n):
            weight_history[t] = self.weights.copy()
            self.update(returns_matrix[t])
        return weight_history

    @property
    def cumulative_wealth(self) -> FloatArray:
        if not self._history:
            return np.array([1.0])
        wealth = [1.0]
        w = np.ones(self.n_assets) / self.n_assets
        for ret in self._history:
            wealth.append(wealth[-1] * (1 + w @ ret))
        return np.array(wealth)


# ===================================================================
# 13. Divergence-based risk measures
# ===================================================================

def divergence_risk_measure(
    returns: FloatArray,
    reference: FloatArray | None = None,
    alpha: float = 0.5,
    divergence: str = "kl",
) -> float:
    """Risk measure based on divergence from reference distribution.
    Higher divergence = higher risk."""
    if reference is None:
        reference = np.random.default_rng(0).normal(0, 0.01, size=len(returns))
    # Estimate densities via histogram
    bins = np.linspace(
        min(returns.min(), reference.min()) - 0.01,
        max(returns.max(), reference.max()) + 0.01,
        50,
    )
    p_hist, _ = np.histogram(returns, bins=bins, density=True)
    q_hist, _ = np.histogram(reference, bins=bins, density=True)
    p_hist = p_hist / (p_hist.sum() + 1e-12)
    q_hist = q_hist / (q_hist.sum() + 1e-12)
    if divergence == "kl":
        return kl_divergence_categorical(p_hist, q_hist)
    elif divergence == "renyi":
        return renyi_divergence(p_hist, q_hist, alpha)
    elif divergence == "hellinger":
        return hellinger_distance(p_hist, q_hist)
    elif divergence == "wasserstein":
        return wasserstein_1d(returns, reference)
    return kl_divergence_categorical(p_hist, q_hist)


def entropic_value_at_risk(
    returns: FloatArray, confidence: float = 0.95, theta: float = 1.0
) -> float:
    """EVaR: tightest upper bound on VaR obtainable from the
    moment generating function.  EVaR = inf_t>0 {t^{-1} * log(MGF(-t)) + VaR_level / t}."""
    sorted_ret = np.sort(returns)
    n = len(sorted_ret)
    var_idx = int((1 - confidence) * n)
    var_val = sorted_ret[var_idx]

    def objective(t: float) -> float:
        if t <= 0:
            return 1e10
        mgf = np.mean(np.exp(-t * returns))
        return float(np.log(mgf + 1e-30) / t + var_val)

    from scipy.optimize import minimize_scalar
    result = minimize_scalar(objective, bounds=(0.01, 100), method="bounded")
    return float(result.fun)


# ===================================================================
# 14. Manifold statistics
# ===================================================================

def frechet_mean_simplex(
    distributions: List[FloatArray], n_iter: int = 100, lr: float = 0.1
) -> FloatArray:
    """Frechet mean on the probability simplex under Fisher-Rao metric."""
    n = len(distributions[0])
    mean = np.ones(n) / n
    for _ in range(n_iter):
        grad = np.zeros(n)
        for p in distributions:
            v = PortfolioManifold(n).logarithmic_map(mean, p)
            grad += v
        grad /= len(distributions)
        mean = PortfolioManifold(n).exponential_map(mean, lr * grad)
    return mean


def frechet_variance_simplex(
    distributions: List[FloatArray], mean: FloatArray
) -> float:
    """Frechet variance: average squared geodesic distance to mean."""
    dists = [fisher_rao_distance(p, mean) ** 2 for p in distributions]
    return float(np.mean(dists))


def parallel_transport_simplex(
    v: FloatArray, p: FloatArray, q: FloatArray
) -> FloatArray:
    """Approximate parallel transport of tangent vector v at p to q
    along the Fisher-Rao geodesic."""
    # Schild's ladder approximation
    mid = geodesic_fisher_rao(p, q, n_points=3)[1]
    # Transport: v at p -> v' at q, approximately
    scale = np.sqrt(np.clip(q, 1e-12, None) / np.clip(p, 1e-12, None))
    return v * scale


# ===================================================================
# __all__
# ===================================================================

__all__ = [
    "fisher_information_normal",
    "fisher_information_multivariate_normal",
    "fisher_information_exponential",
    "fisher_information_gamma",
    "fisher_information_bernoulli",
    "fisher_information_categorical",
    "fisher_information_student_t",
    "fisher_information_numerical",
    "NaturalGradientOptimizer",
    "kl_divergence_normal",
    "kl_divergence_multivariate_normal",
    "kl_divergence_categorical",
    "kl_symmetric",
    "kl_divergence_empirical",
    "wasserstein_1d",
    "wasserstein_2_1d",
    "wasserstein_2_normal",
    "sinkhorn_divergence",
    "renyi_divergence",
    "tsallis_divergence",
    "alpha_divergence",
    "f_divergence",
    "hellinger_distance",
    "ExponentialFamily",
    "normal_exponential_family",
    "exponential_exponential_family",
    "poisson_exponential_family",
    "information_projection_simplex",
    "m_projection_simplex",
    "amari_chentsov_tensor",
    "connection_coefficients",
    "geodesic_simplex_e",
    "geodesic_simplex_m",
    "geodesic_fisher_rao",
    "fisher_rao_distance",
    "PortfolioManifold",
    "regime_distance_matrix",
    "detect_regime_changes",
    "OnlineNaturalGradientPortfolio",
    "divergence_risk_measure",
    "entropic_value_at_risk",
    "frechet_mean_simplex",
    "frechet_variance_simplex",
    "parallel_transport_simplex",
]
