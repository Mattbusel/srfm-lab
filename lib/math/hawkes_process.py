"""
Hawkes point process for modeling trade arrival, volatility clustering, and
cascading liquidations.

Implements:
  - Univariate Hawkes process (exponential kernel) with MLE fitting
  - Multivariate Hawkes (N-dimensional)
  - Simulation (thinning algorithm)
  - Branching ratio estimation (critical exponent)
  - Reflexivity / Hawkes exogeneity ratio
  - Intensity path inference
  - Goodness of fit (KS test on residuals)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Univariate Hawkes ──────────────────────────────────────────────────────────

@dataclass
class HawkesParams:
    """Parameters of a univariate exponential-kernel Hawkes process."""
    mu: float = 0.5     # baseline intensity
    alpha: float = 0.6  # jump size (excitation)
    beta: float = 1.0   # decay rate
    # branching ratio n = alpha/beta; must be < 1 for stationarity

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta if self.beta > 0 else float("inf")

    @property
    def is_stationary(self) -> bool:
        return self.branching_ratio < 1.0

    @property
    def unconditional_intensity(self) -> float:
        """E[lambda] = mu / (1 - n)."""
        n = self.branching_ratio
        return self.mu / (1 - n) if n < 1 else float("inf")


def hawkes_simulate(
    params: HawkesParams,
    T: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate a univariate Hawkes process via Ogata's thinning algorithm.
    Returns array of event times in [0, T].
    """
    rng = rng or np.random.default_rng()
    p = params
    assert p.is_stationary, "Branching ratio must be < 1 for simulation."

    events: list[float] = []
    t = 0.0
    lambda_t = p.mu  # current upper bound of intensity

    while t < T:
        # Draw waiting time from homogeneous Poisson with rate lambda_t
        dt = rng.exponential(1.0 / lambda_t)
        t += dt
        if t > T:
            break

        # Recompute exact intensity at t
        past = np.array(events)
        intensity = p.mu + p.alpha * np.sum(np.exp(-p.beta * (t - past))) if len(past) > 0 else p.mu

        # Accept/reject
        u = rng.uniform(0, lambda_t)
        if u <= intensity:
            events.append(t)
            lambda_t = intensity + p.alpha  # jump in upper bound
        else:
            lambda_t = intensity  # update upper bound

    return np.array(events)


def hawkes_intensity_path(
    events: np.ndarray,
    params: HawkesParams,
    times: np.ndarray,
) -> np.ndarray:
    """
    Compute lambda(t) at arbitrary time grid given past events.
    """
    p = params
    intensity = np.full(len(times), p.mu)
    for event_t in events:
        dt = times - event_t
        mask = dt >= 0
        intensity[mask] += p.alpha * np.exp(-p.beta * dt[mask])
    return intensity


# ── MLE fitting ───────────────────────────────────────────────────────────────

def hawkes_log_likelihood(
    events: np.ndarray,
    T: float,
    params: HawkesParams,
) -> float:
    """
    Log-likelihood of a Hawkes process.
    LL = sum_i log lambda(t_i) - integral_0^T lambda(t)dt
    """
    p = params
    if not p.is_stationary or p.mu <= 0 or p.alpha < 0 or p.beta <= 0:
        return -1e12

    # Compensator (integral of intensity)
    n = len(events)
    if n == 0:
        return -p.mu * T

    # Recursion for sum of past influences
    R = np.zeros(n)
    for i in range(1, n):
        R[i] = math.exp(-p.beta * (events[i] - events[i - 1])) * (1 + R[i - 1])

    intensities = p.mu + p.alpha * R
    log_lk = np.sum(np.log(np.maximum(intensities, 1e-12)))

    # Integral: mu*T + alpha/beta * sum_i(1 - exp(-beta*(T-t_i)))
    integral = p.mu * T + (p.alpha / p.beta) * np.sum(1 - np.exp(-p.beta * (T - events)))
    return float(log_lk - integral)


def fit_hawkes_mle(
    events: np.ndarray,
    T: float,
    n_restarts: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> tuple[HawkesParams, float]:
    """
    Fit Hawkes parameters via MLE using Nelder-Mead.
    Returns (best_params, log_likelihood).
    """
    from scipy.optimize import minimize

    rng = rng or np.random.default_rng()
    best_ll = -np.inf
    best_params = HawkesParams()

    def neg_ll(x):
        mu, alpha, beta = x
        if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
            return 1e12
        ll = hawkes_log_likelihood(events, T, HawkesParams(mu, alpha, beta))
        return -ll

    for _ in range(n_restarts):
        mu0 = rng.uniform(0.01, 5.0)
        alpha0 = rng.uniform(0.01, 0.9)
        beta0 = rng.uniform(alpha0 + 0.01, alpha0 + 3.0)

        res = minimize(neg_ll, x0=[mu0, alpha0, beta0], method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-6})
        if res.success and -res.fun > best_ll:
            mu, alpha, beta = res.x
            if mu > 0 and alpha >= 0 and beta > 0 and alpha < beta:
                best_ll = -res.fun
                best_params = HawkesParams(float(mu), float(alpha), float(beta))

    return best_params, best_ll


# ── Multivariate Hawkes ────────────────────────────────────────────────────────

@dataclass
class MvHawkesParams:
    """
    N-dimensional Hawkes process with exponential kernels.
    mu: baseline intensities shape (N,)
    alpha: excitation matrix shape (N, N)
    beta: decay matrix shape (N, N)
    """
    mu: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))
    alpha: np.ndarray = field(default_factory=lambda: np.array([[0.3, 0.1], [0.1, 0.3]]))
    beta: np.ndarray = field(default_factory=lambda: np.array([[1.0, 1.0], [1.0, 1.0]]))

    @property
    def N(self) -> int:
        return len(self.mu)

    @property
    def spectral_radius(self) -> float:
        """Stationarity requires spectral_radius(alpha/beta) < 1."""
        M = self.alpha / np.maximum(self.beta, 1e-10)
        eigs = np.linalg.eigvals(M)
        return float(np.max(np.abs(eigs)))

    @property
    def is_stationary(self) -> bool:
        return self.spectral_radius < 1.0


def mv_hawkes_simulate(
    params: MvHawkesParams,
    T: float,
    rng: Optional[np.random.Generator] = None,
) -> list[np.ndarray]:
    """
    Simulate multivariate Hawkes process.
    Returns list of N event-time arrays (one per dimension).
    """
    rng = rng or np.random.default_rng()
    p = params
    N = p.N

    events: list[list[float]] = [[] for _ in range(N)]
    t = 0.0

    # Upper bound = sum of all baseline intensities
    lambda_bar = p.mu.sum() * 2  # rough upper bound

    while t < T:
        dt = rng.exponential(1.0 / max(lambda_bar, 1e-10))
        t += dt
        if t > T:
            break

        # Compute true intensities
        lam = p.mu.copy()
        for j in range(N):
            for s in events[j]:
                lam += p.alpha[:, j] * np.exp(-p.beta[:, j] * (t - s))

        lam_total = lam.sum()
        u = rng.uniform(0, lambda_bar)
        if u <= lam_total:
            # Determine which dimension fires
            probs = lam / lam_total
            dim = rng.choice(N, p=probs)
            events[dim].append(t)
            lambda_bar = lam_total + p.alpha[:, dim].sum()  # update upper bound
        else:
            lambda_bar = lam_total

    return [np.array(e) for e in events]


# ── Branching ratio and reflexivity ──────────────────────────────────────────

def branching_ratio(events: np.ndarray, T: float) -> float:
    """
    Estimate branching ratio n from event data.
    n = 1 - mu / (N/T) where mu is the MLE baseline.
    High n (~0.9+) → high reflexivity (most events are self-excited).
    """
    params, _ = fit_hawkes_mle(events, T)
    mean_rate = len(events) / T if T > 0 else 1.0
    n = params.branching_ratio
    return float(np.clip(n, 0.0, 1.0))


# ── Goodness of fit ────────────────────────────────────────────────────────────

def hawkes_residual_analysis(
    events: np.ndarray,
    T: float,
    params: HawkesParams,
) -> dict:
    """
    Time-rescaling theorem: residuals Lambda(t_i) - Lambda(t_{i-1}) ~ Exp(1).
    Returns KS test p-value and residual statistics.
    """
    from scipy.stats import kstest, expon

    p = params
    n = len(events)
    if n < 2:
        return {"ks_pvalue": np.nan, "mean_residual": np.nan}

    # Compute integrated intensity (compensator) at each event
    Lambda = np.zeros(n)
    Lambda[0] = p.mu * events[0]
    for i in range(1, n):
        # Compensator increment from t_{i-1} to t_i
        prev_t = events[i - 1]
        curr_t = events[i]
        # mu term
        inc = p.mu * (curr_t - prev_t)
        # Hawkes term: sum over past events
        for j in range(i):
            s = events[j]
            inc += (p.alpha / p.beta) * (
                math.exp(-p.beta * (prev_t - s)) - math.exp(-p.beta * (curr_t - s))
            )
        Lambda[i] = Lambda[i - 1] + inc

    residuals = np.diff(Lambda)
    ks_stat, ks_pvalue = kstest(residuals, "expon")

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "mean_residual": float(residuals.mean()),
        "std_residual": float(residuals.std()),
        "well_fit": bool(ks_pvalue > 0.05),
    }
