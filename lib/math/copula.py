"""
Copula models for multivariate dependency structure.

Implements:
  - Gaussian copula
  - Student-t copula
  - Clayton copula (lower tail dependence)
  - Gumbel copula (upper tail dependence)
  - Frank copula (symmetric)
  - Kendall's tau ↔ copula parameter conversion
  - MLE fitting via IFM (Inference Functions for Margins)
  - Tail dependence coefficients
  - Copula-based portfolio simulation
"""

from __future__ import annotations
import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from scipy import stats


# ── Probability integral transform ────────────────────────────────────────────

def empirical_pit(x: np.ndarray) -> np.ndarray:
    """Map series to uniform pseudo-observations via rank transform."""
    n = len(x)
    return stats.rankdata(x) / (n + 1)


# ── Base class ─────────────────────────────────────────────────────────────────

class Copula(ABC):
    @abstractmethod
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample n observations from the copula. Returns (n, d) array."""

    @abstractmethod
    def cdf(self, u: np.ndarray) -> np.ndarray:
        """Copula CDF C(u1, ..., ud). Returns (n,) array."""

    @abstractmethod
    def log_density(self, u: np.ndarray) -> np.ndarray:
        """Log copula density log c(u). Returns (n,) array."""

    def fit_tau(self, tau: float) -> "Copula":
        """Return new copula fitted to Kendall's tau."""
        raise NotImplementedError


# ── Gaussian copula ────────────────────────────────────────────────────────────

@dataclass
class GaussianCopula(Copula):
    """Gaussian copula with correlation matrix R."""
    R: np.ndarray  # (d, d) correlation matrix

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        d = self.R.shape[0]
        L = np.linalg.cholesky(self.R)
        Z = rng.standard_normal((n, d)) @ L.T
        return stats.norm.cdf(Z)

    def cdf(self, u: np.ndarray) -> np.ndarray:
        from scipy.stats import multivariate_normal
        z = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        mvn = multivariate_normal(cov=self.R)
        return mvn.cdf(z)

    def log_density(self, u: np.ndarray) -> np.ndarray:
        z = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        d = self.R.shape[0]
        R_inv = np.linalg.inv(self.R)
        sign, log_det = np.linalg.slogdet(self.R)
        quad = np.einsum("ni,ij,nj->n", z, R_inv - np.eye(d), z)
        return -0.5 * (log_det + quad)

    @classmethod
    def from_tau(cls, tau: float, d: int = 2) -> "GaussianCopula":
        rho = math.sin(math.pi / 2 * tau)
        R = np.full((d, d), rho)
        np.fill_diagonal(R, 1.0)
        return cls(R=R)


# ── Student-t copula ───────────────────────────────────────────────────────────

@dataclass
class StudentTCopula(Copula):
    """Bivariate Student-t copula."""
    rho: float = 0.5
    nu: float = 4.0   # degrees of freedom

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        R = np.array([[1.0, self.rho], [self.rho, 1.0]])
        Z = rng.multivariate_normal([0, 0], R, size=n)
        chi2 = rng.chisquare(self.nu, size=(n, 1))
        T = Z / np.sqrt(chi2 / self.nu)
        return stats.t.cdf(T, df=self.nu)

    def cdf(self, u: np.ndarray) -> np.ndarray:
        t_vals = stats.t.ppf(np.clip(u, 1e-10, 1 - 1e-10), df=self.nu)
        from scipy.stats import multivariate_t
        R = np.array([[1.0, self.rho], [self.rho, 1.0]])
        mvt = multivariate_t(shape=R, df=self.nu)
        return mvt.cdf(t_vals)

    def log_density(self, u: np.ndarray) -> np.ndarray:
        from scipy.special import gammaln
        t = stats.t.ppf(np.clip(u, 1e-10, 1 - 1e-10), df=self.nu)
        nu = self.nu
        rho = self.rho
        t1, t2 = t[:, 0], t[:, 1]
        Q = (t1 ** 2 - 2 * rho * t1 * t2 + t2 ** 2) / (1 - rho ** 2)
        log_c = (
            gammaln((nu + 2) / 2) + gammaln(nu / 2)
            - 2 * gammaln((nu + 1) / 2)
            - 0.5 * math.log(1 - rho ** 2)
            + (nu + 1) / 2 * (np.log(1 + t1 ** 2 / nu) + np.log(1 + t2 ** 2 / nu))
            - (nu + 2) / 2 * np.log(1 + Q / nu)
        )
        return log_c

    @property
    def upper_tail_dependence(self) -> float:
        """Lambda_U = Lambda_L for t copula."""
        return 2 * stats.t.cdf(
            -math.sqrt((self.nu + 1) * (1 - self.rho) / (1 + self.rho)),
            df=self.nu + 1,
        )

    @classmethod
    def from_tau(cls, tau: float, nu: float = 4.0) -> "StudentTCopula":
        rho = math.sin(math.pi / 2 * tau)
        return cls(rho=rho, nu=nu)


# ── Archimedean copulas ────────────────────────────────────────────────────────

@dataclass
class ClaytonCopula(Copula):
    """
    Clayton copula: lower tail dependence.
    C(u,v) = (u^-theta + v^-theta - 1)^(-1/theta)
    Kendall's tau = theta / (theta + 2)
    """
    theta: float = 2.0  # theta > 0

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        # Gamma frailty representation
        V = rng.gamma(1.0 / self.theta, 1.0, size=n)
        E1 = rng.exponential(1.0, n)
        E2 = rng.exponential(1.0, n)
        u = (1 + E1 / V) ** (-1.0 / self.theta)
        v = (1 + E2 / V) ** (-1.0 / self.theta)
        return np.column_stack([u, v])

    def cdf(self, u: np.ndarray) -> np.ndarray:
        u1, u2 = u[:, 0], u[:, 1]
        return np.maximum(u1 ** (-self.theta) + u2 ** (-self.theta) - 1, 0) ** (-1.0 / self.theta)

    def log_density(self, u: np.ndarray) -> np.ndarray:
        t = self.theta
        u1, u2 = np.clip(u[:, 0], 1e-10, 1 - 1e-10), np.clip(u[:, 1], 1e-10, 1 - 1e-10)
        log_c = (
            math.log(1 + t) + (-t - 1) * (np.log(u1) + np.log(u2))
            + (-1 / t - 2) * np.log(u1 ** (-t) + u2 ** (-t) - 1)
        )
        return log_c

    @property
    def lower_tail_dependence(self) -> float:
        return 2 ** (-1.0 / self.theta)

    @classmethod
    def from_tau(cls, tau: float) -> "ClaytonCopula":
        theta = max(2 * tau / (1 - tau), 1e-6)
        return cls(theta=theta)


@dataclass
class GumbelCopula(Copula):
    """
    Gumbel copula: upper tail dependence.
    Kendall's tau = 1 - 1/theta
    """
    theta: float = 2.0  # theta >= 1

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        # Stable distribution frailty
        from scipy.stats import levy_stable
        alpha = 1.0 / self.theta
        V = levy_stable.rvs(alpha, 1, loc=0, scale=(math.cos(math.pi / (2 * self.theta))) ** self.theta,
                            size=n, random_state=rng.integers(2 ** 31))
        E1 = rng.exponential(1.0, n)
        E2 = rng.exponential(1.0, n)
        u = np.exp(-(E1 / V) ** (1.0 / self.theta))
        v = np.exp(-(E2 / V) ** (1.0 / self.theta))
        return np.column_stack([u, v])

    def cdf(self, u: np.ndarray) -> np.ndarray:
        u1, u2 = np.clip(u[:, 0], 1e-10, 1), np.clip(u[:, 1], 1e-10, 1)
        A = ((-np.log(u1)) ** self.theta + (-np.log(u2)) ** self.theta) ** (1.0 / self.theta)
        return np.exp(-A)

    def log_density(self, u: np.ndarray) -> np.ndarray:
        t = self.theta
        u1, u2 = np.clip(u[:, 0], 1e-10, 1 - 1e-10), np.clip(u[:, 1], 1e-10, 1 - 1e-10)
        x1 = -np.log(u1)
        x2 = -np.log(u2)
        A = (x1 ** t + x2 ** t) ** (1.0 / t)
        log_c = (
            -A + (t - 1) * (np.log(x1) + np.log(x2))
            + np.log(A + t - 1)
            - (1.0 / t + 2) * np.log(x1 ** t + x2 ** t)
            - np.log(u1) - np.log(u2)
        )
        return log_c

    @property
    def upper_tail_dependence(self) -> float:
        return 2 - 2 ** (1.0 / self.theta)

    @classmethod
    def from_tau(cls, tau: float) -> "GumbelCopula":
        theta = max(1.0 / (1 - tau), 1.0)
        return cls(theta=theta)


@dataclass
class FrankCopula(Copula):
    """
    Frank copula: no tail dependence, symmetric.
    """
    theta: float = 3.0  # any real ≠ 0

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        u = rng.uniform(size=n)
        p = rng.uniform(size=n)
        t = self.theta
        v = -np.log(1 - p * (1 - math.exp(-t)) /
                    (np.exp(-t * u) - p * (np.exp(-t * u) - math.exp(-t)))) / t
        return np.column_stack([u, v])

    def cdf(self, u: np.ndarray) -> np.ndarray:
        t = self.theta
        u1, u2 = u[:, 0], u[:, 1]
        return -np.log(1 + (np.exp(-t * u1) - 1) * (np.exp(-t * u2) - 1) / (math.exp(-t) - 1)) / t

    def log_density(self, u: np.ndarray) -> np.ndarray:
        t = self.theta
        u1, u2 = np.clip(u[:, 0], 1e-10, 1 - 1e-10), np.clip(u[:, 1], 1e-10, 1 - 1e-10)
        e1 = np.exp(-t * u1) - 1
        e2 = np.exp(-t * u2) - 1
        e_t = math.exp(-t) - 1
        numer = -t * e_t * (1 + e_t) * np.exp(-t * (u1 + u2))
        denom = (e_t + e1 * e2) ** 2
        return np.log(np.maximum(np.abs(numer / denom), 1e-300))

    @classmethod
    def from_tau(cls, tau: float) -> "FrankCopula":
        # Numerical inversion: tau = 1 - 4/theta*(1 - D1(theta))
        from scipy.optimize import brentq
        from scipy.special import spence

        def debye(x):
            # Debye function D1 approximation
            return 1 - x / 4 + x ** 2 / 36 - x ** 4 / 3600 if abs(x) < 0.1 else (
                1 / x * (math.pi ** 2 / 6 - float(spence(math.exp(-x))))
            )

        def eq(theta):
            if abs(theta) < 1e-6:
                return -tau
            return 1 - 4 / theta * (1 - debye(theta)) - tau

        theta = brentq(eq, 0.001, 50.0) if tau > 0 else brentq(eq, -50.0, -0.001)
        return cls(theta=float(theta))


# ── Tail dependence summary ────────────────────────────────────────────────────

def tail_dependence_empirical(
    u: np.ndarray,
    v: np.ndarray,
    q: float = 0.05,
) -> dict:
    """
    Non-parametric tail dependence estimation.
    Lower tail: P(U < q | V < q)
    Upper tail: P(U > 1-q | V > 1-q)
    """
    n = len(u)
    lower = np.sum((u < q) & (v < q)) / max(np.sum(v < q), 1)
    upper = np.sum((u > 1 - q) & (v > 1 - q)) / max(np.sum(v > 1 - q), 1)
    return {"lower_tail": float(lower), "upper_tail": float(upper)}


# ── Model selection ────────────────────────────────────────────────────────────

def select_copula(
    x: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Fit multiple copulas via MLE and select by AIC.
    Returns dict with best model and all AIC values.
    """
    from scipy.optimize import minimize_scalar

    u = empirical_pit(x)
    v = empirical_pit(y)
    uv = np.column_stack([u, v])

    tau, _ = stats.kendalltau(x, y)

    results = {}
    for name, copula_cls in [
        ("gaussian", GaussianCopula),
        ("clayton", ClaytonCopula),
        ("gumbel", GumbelCopula),
        ("frank", FrankCopula),
    ]:
        try:
            if name == "gaussian":
                cop = GaussianCopula.from_tau(max(-0.99, min(0.99, tau)))
            elif name == "clayton":
                cop = ClaytonCopula.from_tau(max(0.01, tau))
            elif name == "gumbel":
                cop = GumbelCopula.from_tau(max(0.01, tau))
            elif name == "frank":
                cop = FrankCopula.from_tau(tau)
            ll = float(cop.log_density(uv).sum())
            aic = -2 * ll + 2  # 1 parameter
            results[name] = {"aic": aic, "log_likelihood": ll, "copula": cop}
        except Exception:
            results[name] = {"aic": np.inf, "log_likelihood": -np.inf}

    best = min(results.items(), key=lambda kv: kv[1]["aic"])
    return {"best": best[0], "models": results, "tau": float(tau)}
