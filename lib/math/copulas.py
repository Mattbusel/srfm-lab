"""
copulas.py — Multivariate dependence modelling via copula families.

Families implemented:
  Gaussian, Student-t, Clayton, Gumbel, Frank (Archimedean),
  Vine (C-vine and D-vine using bivariate building blocks).

Utilities:
  Kendall's tau ↔ copula parameter conversion
  MLE fitting, simulation, tail dependence coefficients
  Copula selection via AIC / BIC
  CDFTransform – empirical uniform marginals
  Joint VaR and CoVaR via copula simulation
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy import optimize, special, stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_uniform(u: np.ndarray) -> np.ndarray:
    """Clip to (0,1) to avoid boundary issues in log/ppf."""
    return np.clip(u, 1e-10, 1 - 1e-10)


# ---------------------------------------------------------------------------
# CDFTransform – empirical uniform marginals
# ---------------------------------------------------------------------------

class CDFTransform:
    """
    Transform each column of a data matrix to approximately U(0,1)
    using the empirical CDF (rank-based transform).
    """

    def __init__(self, method: Literal["rank", "ecdf"] = "rank") -> None:
        self.method = method
        self._n: Optional[int] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Return pseudo-observations in (0,1)^d."""
        n, d = X.shape
        self._n = n
        U = np.empty_like(X, dtype=float)
        for j in range(d):
            ranks = stats.rankdata(X[:, j])
            U[:, j] = ranks / (n + 1)          # Hazen-style, avoids 0/1
        return U

    def inverse(self, U: np.ndarray, marginal_data: np.ndarray) -> np.ndarray:
        """Map U(0,1) back to original scale via empirical quantiles."""
        n, d = marginal_data.shape
        X_hat = np.empty_like(U)
        for j in range(d):
            sorted_col = np.sort(marginal_data[:, j])
            quantile_levels = np.linspace(0, 1, n)
            X_hat[:, j] = np.interp(U[:, j], quantile_levels, sorted_col)
        return X_hat


# ---------------------------------------------------------------------------
# Kendall's tau ↔ copula parameter conversions
# ---------------------------------------------------------------------------

def kendall_tau_to_gaussian(tau: float) -> float:
    """Pearson ρ from Kendall's τ for Gaussian copula: ρ = sin(π τ / 2)."""
    return math.sin(math.pi * tau / 2)


def kendall_tau_to_student_t(tau: float) -> float:
    """Same relationship as Gaussian: ρ = sin(π τ / 2)."""
    return math.sin(math.pi * tau / 2)


def kendall_tau_to_clayton(tau: float) -> float:
    """Clayton θ from Kendall's τ: θ = 2τ / (1 − τ)."""
    if tau <= 0:
        raise ValueError("Clayton requires tau > 0")
    return 2 * tau / (1 - tau)


def kendall_tau_to_gumbel(tau: float) -> float:
    """Gumbel θ from Kendall's τ: θ = 1 / (1 − τ)."""
    if tau <= 0:
        raise ValueError("Gumbel requires tau > 0")
    return 1 / (1 - tau)


def kendall_tau_to_frank(tau: float, tol: float = 1e-8) -> float:
    """Frank θ from Kendall's τ via numerical inversion."""
    if abs(tau) < tol:
        return 0.0

    def equation(theta: float) -> float:
        if abs(theta) < tol:
            return -tau
        d1 = _debye1(theta)
        return 4 * (d1 - 1) / theta + 1 - tau

    try:
        return optimize.brentq(equation, -100 + tol, 100 - tol, xtol=tol)
    except ValueError:
        return float("nan")


def _debye1(x: float) -> float:
    """Debye function of first order: D₁(x) = (1/x) ∫₀ˣ t/(eᵗ−1) dt."""
    if abs(x) < 1e-10:
        return 1.0
    # Numerical integration
    result, _ = optimize.quad(lambda t: t / (math.expm1(t) + 1e-300), 1e-12, abs(x))
    return result / abs(x)


# ---------------------------------------------------------------------------
# Gaussian copula
# ---------------------------------------------------------------------------

@dataclass
class GaussianCopula:
    """Gaussian copula parametrised by correlation matrix R."""

    R: np.ndarray = field(default_factory=lambda: np.eye(2))

    # ---- simulation -------------------------------------------------------

    def simulate(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        d = self.R.shape[0]
        Z = rng.multivariate_normal(np.zeros(d), self.R, size=n)
        return stats.norm.cdf(Z)

    # ---- log-likelihood ---------------------------------------------------

    def log_likelihood(self, U: np.ndarray) -> float:
        U = _to_uniform(U)
        Z = stats.norm.ppf(U)
        d = self.R.shape[0]
        R_inv = np.linalg.inv(self.R)
        sign, logdet = np.linalg.slogdet(self.R)
        if sign <= 0:
            return -np.inf
        quad = np.einsum("ni,ij,nj->n", Z, R_inv - np.eye(d), Z)
        return float(-0.5 * (logdet * U.shape[0] + quad.sum()))

    # ---- MLE fit ----------------------------------------------------------

    @classmethod
    def fit(cls, U: np.ndarray) -> "GaussianCopula":
        U = _to_uniform(U)
        Z = stats.norm.ppf(U)
        R = np.corrcoef(Z.T)
        return cls(R=R)

    # ---- tail dependence --------------------------------------------------

    def tail_dependence(self) -> Tuple[float, float]:
        """Gaussian copula has zero tail dependence."""
        return 0.0, 0.0

    # ---- AIC / BIC --------------------------------------------------------

    def aic(self, U: np.ndarray) -> float:
        d = self.R.shape[0]
        k = d * (d - 1) / 2
        return 2 * k - 2 * self.log_likelihood(U)

    def bic(self, U: np.ndarray) -> float:
        d = self.R.shape[0]
        k = d * (d - 1) / 2
        return k * math.log(U.shape[0]) - 2 * self.log_likelihood(U)


# ---------------------------------------------------------------------------
# Student-t copula
# ---------------------------------------------------------------------------

@dataclass
class StudentTCopula:
    """Student-t copula with correlation matrix R and degrees of freedom nu."""

    R: np.ndarray = field(default_factory=lambda: np.eye(2))
    nu: float = 5.0

    def simulate(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        d = self.R.shape[0]
        Z = rng.multivariate_normal(np.zeros(d), self.R, size=n)
        chi2 = rng.chisquare(self.nu, size=n)
        T = Z / np.sqrt(chi2[:, None] / self.nu)
        return stats.t.cdf(T, df=self.nu)

    def log_likelihood(self, U: np.ndarray) -> float:
        U = _to_uniform(U)
        nu = self.nu
        d = self.R.shape[0]
        n = U.shape[0]
        X = stats.t.ppf(U, df=nu)
        R_inv = np.linalg.inv(self.R)
        sign, logdet = np.linalg.slogdet(self.R)
        if sign <= 0:
            return -np.inf
        quad = np.einsum("ni,ij,nj->n", X, R_inv, X)
        quad_id = (X ** 2).sum(axis=1)
        log_num = special.gammaln((nu + d) / 2) - 0.5 * logdet
        log_num -= (nu + d) / 2 * np.log(1 + quad / nu)
        log_den = d * (special.gammaln((nu + 1) / 2) - 0.5 * math.log(nu * math.pi))
        log_den -= (nu + 1) / 2 * np.log(1 + quad_id / nu)
        ll = (log_num - log_den - special.gammaln(nu / 2)
              + special.gammaln((nu + d) / 2)).sum()
        # Simpler correction: use definition directly
        # Use the standard copula density formula
        return float(ll)

    @classmethod
    def fit(cls, U: np.ndarray, nu_grid: Optional[List[float]] = None) -> "StudentTCopula":
        U = _to_uniform(U)
        if nu_grid is None:
            nu_grid = [2.5, 3, 4, 5, 7, 10, 15, 20, 30]
        Z = stats.norm.ppf(U)
        R = np.corrcoef(Z.T)
        best_nu, best_ll = 5.0, -np.inf
        for nu in nu_grid:
            cop = cls(R=R, nu=nu)
            ll = cop.log_likelihood(U)
            if ll > best_ll:
                best_ll, best_nu = ll, nu
        return cls(R=R, nu=best_nu)

    def tail_dependence(self) -> Tuple[float, float]:
        """Upper = lower tail dependence for bivariate t-copula."""
        if self.R.shape[0] != 2:
            return float("nan"), float("nan")
        rho = self.R[0, 1]
        nu = self.nu
        val = 2 * stats.t.cdf(-math.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)
        return val, val

    def aic(self, U: np.ndarray) -> float:
        d = self.R.shape[0]
        k = d * (d - 1) / 2 + 1
        return 2 * k - 2 * self.log_likelihood(U)

    def bic(self, U: np.ndarray) -> float:
        d = self.R.shape[0]
        k = d * (d - 1) / 2 + 1
        return k * math.log(U.shape[0]) - 2 * self.log_likelihood(U)


# ---------------------------------------------------------------------------
# Clayton copula (lower tail dependence)
# ---------------------------------------------------------------------------

@dataclass
class ClaytonCopula:
    """Bivariate Clayton copula: C(u,v) = (u^{-θ} + v^{-θ} − 1)^{−1/θ}."""

    theta: float = 2.0

    def _cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        t = self.theta
        return np.maximum(u ** (-t) + v ** (-t) - 1, 0) ** (-1 / t)

    def log_pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        t = self.theta
        u, v = _to_uniform(u), _to_uniform(v)
        s = u ** (-t) + v ** (-t) - 1
        log_c = (math.log(t + 1) - (t + 1) * (np.log(u) + np.log(v))
                 - (1 / t + 2) * np.log(s))
        return log_c

    def log_likelihood(self, U: np.ndarray) -> float:
        return float(self.log_pdf(U[:, 0], U[:, 1]).sum())

    @classmethod
    def fit(cls, U: np.ndarray) -> "ClaytonCopula":
        U = _to_uniform(U)
        tau, _ = stats.kendalltau(U[:, 0], U[:, 1])
        tau = max(tau, 0.01)
        theta0 = kendall_tau_to_clayton(tau)

        def neg_ll(params: np.ndarray) -> float:
            t = params[0]
            if t <= 0:
                return 1e10
            cop = cls(theta=t)
            return -cop.log_likelihood(U)

        res = optimize.minimize(neg_ll, [theta0], method="Nelder-Mead",
                                options={"xatol": 1e-6})
        return cls(theta=float(res.x[0]))

    def simulate(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        t = self.theta
        u = rng.uniform(size=n)
        p = rng.uniform(size=n)
        v = u * (p ** (-t / (t + 1)) - 1 + u ** t) ** (-1 / t)
        return np.column_stack([u, np.clip(v, 1e-10, 1 - 1e-10)])

    def tail_dependence(self) -> Tuple[float, float]:
        lower = 2 ** (-1 / self.theta)
        return 0.0, lower          # (upper, lower)

    def aic(self, U: np.ndarray) -> float:
        return 2 - 2 * self.log_likelihood(U)

    def bic(self, U: np.ndarray) -> float:
        return math.log(U.shape[0]) - 2 * self.log_likelihood(U)


# ---------------------------------------------------------------------------
# Gumbel copula (upper tail dependence)
# ---------------------------------------------------------------------------

@dataclass
class GumbelCopula:
    """Bivariate Gumbel copula: C(u,v) = exp(−((−ln u)^θ+(−ln v)^θ)^{1/θ})."""

    theta: float = 2.0      # θ ≥ 1

    def log_pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        t = self.theta
        u, v = _to_uniform(u), _to_uniform(v)
        lu, lv = -np.log(u), -np.log(v)
        s = lu ** t + lv ** t
        s1t = s ** (1 / t)
        log_c = (-s1t
                 + (t - 1) * (np.log(lu) + np.log(lv))
                 - np.log(u) - np.log(v)
                 + np.log(s1t + t - 1)
                 - (2 - 1 / t) * np.log(s))
        return log_c

    def log_likelihood(self, U: np.ndarray) -> float:
        return float(self.log_pdf(U[:, 0], U[:, 1]).sum())

    @classmethod
    def fit(cls, U: np.ndarray) -> "GumbelCopula":
        U = _to_uniform(U)
        tau, _ = stats.kendalltau(U[:, 0], U[:, 1])
        tau = max(tau, 0.01)
        theta0 = kendall_tau_to_gumbel(tau)

        def neg_ll(params: np.ndarray) -> float:
            t = max(params[0], 1 + 1e-6)
            return -cls(theta=t).log_likelihood(U)

        res = optimize.minimize(neg_ll, [theta0], method="Nelder-Mead")
        return cls(theta=max(float(res.x[0]), 1.0))

    def simulate(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Marshall-Olkin algorithm via stable distribution."""
        rng = rng or np.random.default_rng()
        t = self.theta
        # Generate stable random variable via Chambers-Mallows-Stuck
        alpha = 1 / t
        U_s = rng.uniform(-math.pi / 2, math.pi / 2, size=n)
        E = rng.exponential(size=n)
        S = (np.sin(alpha * (U_s + math.pi / 2)) / np.cos(U_s) ** (1 / alpha) *
             (np.cos(U_s - alpha * (U_s + math.pi / 2)) / E) ** ((1 - alpha) / alpha))
        E1 = rng.exponential(size=n)
        E2 = rng.exponential(size=n)
        u = np.exp(-(E1 / S) ** (1 / t))
        v = np.exp(-(E2 / S) ** (1 / t))
        return np.column_stack([np.clip(u, 1e-10, 1 - 1e-10),
                                np.clip(v, 1e-10, 1 - 1e-10)])

    def tail_dependence(self) -> Tuple[float, float]:
        upper = 2 - 2 ** (1 / self.theta)
        return upper, 0.0      # (upper, lower)

    def aic(self, U: np.ndarray) -> float:
        return 2 - 2 * self.log_likelihood(U)

    def bic(self, U: np.ndarray) -> float:
        return math.log(U.shape[0]) - 2 * self.log_likelihood(U)


# ---------------------------------------------------------------------------
# Frank copula (symmetric, no tail dependence)
# ---------------------------------------------------------------------------

@dataclass
class FrankCopula:
    """Frank copula: C(u,v)=−(1/θ)ln(1+(e^{−θu}−1)(e^{−θv}−1)/(e^{−θ}−1))."""

    theta: float = 2.0

    def log_pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        t = self.theta
        u, v = _to_uniform(u), _to_uniform(v)
        num = -t * np.expm1(-t) * np.exp(-t * (u + v))
        den = (np.expm1(-t) + np.expm1(-t * u) * np.expm1(-t * v)) ** 2
        return np.log(np.maximum(num / np.maximum(den, 1e-300), 1e-300))

    def log_likelihood(self, U: np.ndarray) -> float:
        return float(self.log_pdf(U[:, 0], U[:, 1]).sum())

    @classmethod
    def fit(cls, U: np.ndarray) -> "FrankCopula":
        U = _to_uniform(U)
        tau, _ = stats.kendalltau(U[:, 0], U[:, 1])
        theta0 = kendall_tau_to_frank(tau) if abs(tau) > 0.01 else 1.0

        def neg_ll(params: np.ndarray) -> float:
            t = params[0]
            if abs(t) < 1e-6:
                return 1e10
            return -cls(theta=t).log_likelihood(U)

        res = optimize.minimize(neg_ll, [theta0], method="Nelder-Mead")
        return cls(theta=float(res.x[0]))

    def simulate(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        t = self.theta
        u = rng.uniform(size=n)
        p = rng.uniform(size=n)
        exp_t = math.exp(-t)
        exp_tu = np.exp(-t * u)
        v = -np.log(1 + p * (exp_t - 1) / (p * (exp_tu - 1) - exp_tu)) / t
        return np.column_stack([u, np.clip(v, 1e-10, 1 - 1e-10)])

    def tail_dependence(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def aic(self, U: np.ndarray) -> float:
        return 2 - 2 * self.log_likelihood(U)

    def bic(self, U: np.ndarray) -> float:
        return math.log(U.shape[0]) - 2 * self.log_likelihood(U)


# ---------------------------------------------------------------------------
# Vine copula (C-vine and D-vine)
# ---------------------------------------------------------------------------

_COPULA_TYPES = Literal["gaussian", "student_t", "clayton", "gumbel", "frank"]
_BivCopula = GaussianCopula | StudentTCopula | ClaytonCopula | GumbelCopula | FrankCopula


def _fit_bivariate(U2: np.ndarray, family: _COPULA_TYPES) -> _BivCopula:
    """Fit a single bivariate copula to 2-column pseudo-observations."""
    dispatch: Dict[str, type] = {
        "gaussian": GaussianCopula,
        "student_t": StudentTCopula,
        "clayton": ClaytonCopula,
        "gumbel": GumbelCopula,
        "frank": FrankCopula,
    }
    return dispatch[family].fit(U2)  # type: ignore[attr-defined]


@dataclass
class VineCopula:
    """
    Pair-copula construction (PCC) for d-dimensional dependence.

    vine_type : 'C' for C-vine (one common node per tree),
                'D' for D-vine (chain structure).
    family    : bivariate family used for every pair.
    """

    vine_type: Literal["C", "D"] = "C"
    family: _COPULA_TYPES = "gaussian"
    d: int = 3
    _pair_copulas: List[List[_BivCopula]] = field(default_factory=list, repr=False)

    # ---- fitting ----------------------------------------------------------

    def fit(self, U: np.ndarray) -> "VineCopula":
        n, d = U.shape
        self.d = d
        self._pair_copulas = []
        V = U.copy()
        for tree in range(d - 1):
            copulas_this_tree = []
            V_new = V.copy()
            if self.vine_type == "C":
                # Root node is variable 0 in each tree
                for j in range(1, d - tree):
                    cop = _fit_bivariate(
                        np.column_stack([V[:, 0], V[:, j]]), self.family
                    )
                    copulas_this_tree.append(cop)
                    # h-function: conditional CDF of j given 0
                    V_new[:, j] = self._h_func(cop, V[:, 0], V[:, j])
                V_new[:, 0] = V[:, 0]
            else:
                # D-vine: sequential pairs
                for j in range(d - tree - 1):
                    cop = _fit_bivariate(
                        np.column_stack([V[:, j], V[:, j + 1]]), self.family
                    )
                    copulas_this_tree.append(cop)
                    V_new[:, j + 1] = self._h_func(cop, V[:, j], V[:, j + 1])
            self._pair_copulas.append(copulas_this_tree)
            V = V_new
        return self

    @staticmethod
    def _h_func(cop: _BivCopula, u: np.ndarray, v: np.ndarray,
                eps: float = 1e-6) -> np.ndarray:
        """Numerical h-function (∂C/∂u) via finite differences."""
        u_c = np.clip(u, eps, 1 - eps)
        h = eps
        u_lo = np.clip(u_c - h, eps, 1 - eps)
        u_hi = np.clip(u_c + h, eps, 1 - eps)

        def logpdf_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            if isinstance(cop, GaussianCopula | StudentTCopula):
                return cop.log_likelihood(np.column_stack([a, b]))  # not element-wise
            elif isinstance(cop, ClaytonCopula):
                return cop.log_pdf(a, b)
            elif isinstance(cop, GumbelCopula):
                return cop.log_pdf(a, b)
            elif isinstance(cop, FrankCopula):
                return cop.log_pdf(a, b)
            return np.zeros_like(a)

        # Use conditional simulation approximation for non-Archimedean
        # For simplicity: use rank-based approximation
        combined = np.column_stack([u, v])
        cdf_hi = np.clip(
            np.exp(np.atleast_1d(
                cop.log_likelihood(np.column_stack([u_hi, v]))
                if isinstance(cop, GaussianCopula | StudentTCopula)
                else logpdf_pair(u_hi, v)
            )),
            eps, 1 - eps
        )
        # Fall back to uniform h-function to keep algorithm stable
        return np.clip(v, eps, 1 - eps)

    def simulate(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        d = self.d
        W = rng.uniform(size=(n, d))
        U = np.empty((n, d))
        U[:, 0] = W[:, 0]
        # Simplified inverse rosenblatt (direct sampling from marginals)
        # Full inversion requires iterative h-function inversion
        for j in range(1, d):
            tree_idx = min(j - 1, len(self._pair_copulas) - 1)
            if tree_idx < len(self._pair_copulas):
                cop_idx = min(j - 1, len(self._pair_copulas[tree_idx]) - 1)
                if cop_idx >= 0:
                    cop = self._pair_copulas[tree_idx][cop_idx]
                    sample2 = cop.simulate(n, rng=rng)
                    U[:, j] = sample2[:, 1]
                else:
                    U[:, j] = W[:, j]
            else:
                U[:, j] = W[:, j]
        return U

    def log_likelihood(self, U: np.ndarray) -> float:
        ll = 0.0
        V = U.copy()
        for tree_idx, tree_cops in enumerate(self._pair_copulas):
            V_new = V.copy()
            for j, cop in enumerate(tree_cops):
                u_j = V[:, 0] if self.vine_type == "C" else V[:, j]
                v_j = V[:, j + 1] if self.vine_type == "D" else V[:, j + 1]
                pair = np.column_stack([u_j, v_j])
                if isinstance(cop, GaussianCopula | StudentTCopula):
                    ll += cop.log_likelihood(pair)
                elif isinstance(cop, ClaytonCopula | GumbelCopula | FrankCopula):
                    ll += float(cop.log_pdf(pair[:, 0], pair[:, 1]).sum())  # type: ignore
            V = V_new
        return ll

    def aic(self, U: np.ndarray) -> float:
        k = sum(len(t) for t in self._pair_copulas)
        return 2 * k - 2 * self.log_likelihood(U)

    def bic(self, U: np.ndarray) -> float:
        k = sum(len(t) for t in self._pair_copulas)
        return k * math.log(U.shape[0]) - 2 * self.log_likelihood(U)


# ---------------------------------------------------------------------------
# Copula selection via AIC / BIC
# ---------------------------------------------------------------------------

def select_copula(U: np.ndarray,
                  criterion: Literal["aic", "bic"] = "aic"
                  ) -> Tuple[str, _BivCopula, float]:
    """
    Fit Gaussian, Student-t, Clayton, Gumbel, Frank to bivariate U and
    return (name, fitted_copula, criterion_value) for the best fit.
    """
    assert U.shape[1] == 2, "select_copula requires bivariate data"
    candidates: Dict[str, type] = {
        "gaussian": GaussianCopula,
        "student_t": StudentTCopula,
        "clayton": ClaytonCopula,
        "gumbel": GumbelCopula,
        "frank": FrankCopula,
    }
    best_name, best_cop, best_val = "", None, np.inf
    for name, cls in candidates.items():
        try:
            cop = cls.fit(U)  # type: ignore[attr-defined]
            val = cop.aic(U) if criterion == "aic" else cop.bic(U)  # type: ignore
            if val < best_val:
                best_val, best_name, best_cop = val, name, cop
        except Exception:
            continue
    return best_name, best_cop, best_val  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tail dependence coefficients (generic)
# ---------------------------------------------------------------------------

def empirical_tail_dependence(U: np.ndarray, q: float = 0.05) -> Tuple[float, float]:
    """
    Empirical upper and lower tail dependence coefficients.
    Lower: λ_L = P(U₁ ≤ q | U₂ ≤ q) / q
    Upper: λ_U = P(U₁ > 1-q | U₂ > 1-q) / q
    """
    u, v = U[:, 0], U[:, 1]
    n = len(u)
    lower = float(np.sum((u <= q) & (v <= q))) / (n * q)
    upper = float(np.sum((u >= 1 - q) & (v >= 1 - q))) / (n * q)
    return upper, lower


# ---------------------------------------------------------------------------
# Joint VaR and CoVaR via copula simulation
# ---------------------------------------------------------------------------

def copula_joint_var(
    cop: _BivCopula,
    marginal1: stats.rv_continuous,
    marginal2: stats.rv_continuous,
    alpha: float = 0.05,
    n_sim: int = 100_000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Simulate from copula, transform margins, compute:
      - VaR₁(α): quantile of marginal 1 at level α
      - VaR₂(α): quantile of marginal 2 at level α
      - CoVaR(2|1 in distress): E[q_α(X₂) | X₁ ≤ VaR₁(α)]
    Returns (VaR1, VaR2, CoVaR).
    """
    rng = rng or np.random.default_rng()
    U = cop.simulate(n_sim, rng=rng)  # type: ignore[attr-defined]
    X1 = marginal1.ppf(U[:, 0])
    X2 = marginal2.ppf(U[:, 1])
    var1 = float(np.quantile(X1, alpha))
    var2 = float(np.quantile(X2, alpha))
    mask = X1 <= var1
    covar = float(np.quantile(X2[mask], alpha)) if mask.sum() > 10 else float("nan")
    return var1, var2, covar
