"""
Jump-diffusion models for fat-tailed price dynamics.

Implements:
  - Merton jump-diffusion (Gaussian jumps)
  - Kou double-exponential jump model
  - Variance Gamma (VG) process
  - Normal Inverse Gaussian (NIG) process
  - Jump intensity estimation from returns
  - Option pricing under jump-diffusion
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy import stats


# ── Merton jump-diffusion ──────────────────────────────────────────────────────

@dataclass
class MertonParams:
    """
    Merton (1976) jump-diffusion.
    dS/S = (mu - lambda*kbar)*dt + sigma*dW + (J-1)*dN
    J ~ LogNormal(mu_J, sigma_J), N ~ Poisson(lambda*dt)
    kbar = exp(mu_J + 0.5*sigma_J^2) - 1 (compensator)
    """
    mu: float = 0.05        # drift
    sigma: float = 0.2      # diffusion vol
    lam: float = 1.0        # jump intensity (jumps/year)
    mu_J: float = -0.05     # mean log jump size
    sigma_J: float = 0.1    # std log jump size

    @property
    def kbar(self) -> float:
        return math.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1


def merton_simulate(
    params: MertonParams,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    S0: float = 100.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate Merton jump-diffusion paths."""
    rng = rng or np.random.default_rng()
    p = params
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    S = np.full((n_paths, n_steps + 1), S0)
    kbar = p.kbar

    for i in range(n_steps):
        # Diffusion
        Z = rng.standard_normal(n_paths)
        # Jumps: Poisson number of jumps per step
        n_jumps = rng.poisson(p.lam * dt, n_paths)
        # Jump sizes: sum of log-normal jumps
        log_jump = np.zeros(n_paths)
        for k in range(n_paths):
            if n_jumps[k] > 0:
                log_jump[k] = np.sum(
                    rng.normal(p.mu_J, p.sigma_J, n_jumps[k])
                )

        log_ret = (
            (p.mu - p.lam * kbar - 0.5 * p.sigma ** 2) * dt
            + p.sigma * sqrt_dt * Z
            + log_jump
        )
        S[:, i + 1] = S[:, i] * np.exp(log_ret)

    return S


def merton_call_mc(
    params: MertonParams,
    K: float,
    T: float,
    r: float,
    n_steps: int = 252,
    n_paths: int = 50_000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Monte Carlo call price under Merton. Returns (price, std_err)."""
    p_risk = MertonParams(mu=r, sigma=params.sigma, lam=params.lam,
                          mu_J=params.mu_J, sigma_J=params.sigma_J)
    S = merton_simulate(p_risk, T, n_steps, n_paths, 100.0, rng)
    payoffs = np.maximum(S[:, -1] - K, 0.0) * math.exp(-r * T)
    return float(payoffs.mean()), float(payoffs.std() / math.sqrt(n_paths))


def merton_call_series(
    params: MertonParams,
    K: float,
    T: float,
    r: float,
    n_terms: int = 20,
) -> float:
    """
    Merton call via Poisson series expansion (closed-form approximation).
    """
    from scipy.stats import norm

    S0 = 100.0
    p = params
    kbar = p.kbar
    mu_tilde = r - p.lam * kbar
    total = 0.0

    for k in range(n_terms):
        # Weight
        lam_T = p.lam * T
        w = math.exp(-lam_T) * lam_T ** k / math.factorial(k)
        # Adjusted params for k jumps
        sigma_k = math.sqrt(p.sigma ** 2 + k * p.sigma_J ** 2 / T)
        mu_k = mu_tilde + k * p.mu_J / T
        # BS price
        d1 = (math.log(S0 / K) + (mu_k + 0.5 * sigma_k ** 2) * T) / (sigma_k * math.sqrt(T))
        d2 = d1 - sigma_k * math.sqrt(T)
        bs = S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        total += w * bs

    return total


# ── Kou double-exponential ────────────────────────────────────────────────────

@dataclass
class KouParams:
    """
    Kou (2002) double-exponential jump model.
    Up jumps ~ Exp(eta1), Down jumps ~ Exp(eta2)
    p = P(up jump), q = 1-p
    """
    sigma: float = 0.2
    lam: float = 0.5        # jump intensity
    p: float = 0.6          # prob of up jump
    eta1: float = 30.0      # up jump rate (mean up = 1/eta1)
    eta2: float = 20.0      # down jump rate (mean down = 1/eta2)
    mu: float = 0.05

    @property
    def kbar(self) -> float:
        return (self.p * self.eta1 / (self.eta1 - 1)
                + (1 - self.p) * self.eta2 / (self.eta2 + 1) - 1)


def kou_simulate(
    params: KouParams,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    S0: float = 100.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate Kou double-exponential jump paths."""
    rng = rng or np.random.default_rng()
    p = params
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    S = np.full((n_paths, n_steps + 1), S0)
    kbar = p.kbar

    for i in range(n_steps):
        Z = rng.standard_normal(n_paths)
        n_jumps = rng.poisson(p.lam * dt, n_paths)

        log_jump = np.zeros(n_paths)
        for k in range(n_paths):
            for _ in range(n_jumps[k]):
                if rng.uniform() < p.p:
                    log_jump[k] += rng.exponential(1.0 / p.eta1)
                else:
                    log_jump[k] -= rng.exponential(1.0 / p.eta2)

        log_ret = (
            (p.mu - p.lam * kbar - 0.5 * p.sigma ** 2) * dt
            + p.sigma * sqrt_dt * Z
            + log_jump
        )
        S[:, i + 1] = S[:, i] * np.exp(log_ret)

    return S


# ── Variance Gamma ────────────────────────────────────────────────────────────

@dataclass
class VGParams:
    """
    Variance Gamma process (Madan, Carr, Chang 1998).
    VG = theta*G + sigma*W(G), G ~ Gamma(t/nu, nu)
    """
    sigma: float = 0.2    # vol parameter
    nu: float = 0.2       # variance of gamma process (kurtosis)
    theta: float = -0.1   # skewness parameter
    mu: float = 0.05

    @property
    def omega(self) -> float:
        """Compensator."""
        return math.log(1 - self.theta * self.nu - 0.5 * self.sigma ** 2 * self.nu) / self.nu


def vg_simulate(
    params: VGParams,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    S0: float = 100.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate Variance Gamma paths."""
    rng = rng or np.random.default_rng()
    p = params
    dt = T / n_steps

    S = np.full((n_paths, n_steps + 1), S0)
    omega = p.omega

    for i in range(n_steps):
        # Gamma time change
        G = rng.gamma(dt / p.nu, p.nu, n_paths)
        # Brownian increment subordinated by G
        Z = rng.standard_normal(n_paths)
        X = p.theta * G + p.sigma * np.sqrt(G) * Z
        log_ret = (p.mu + omega) * dt + X
        S[:, i + 1] = S[:, i] * np.exp(log_ret)

    return S


# ── NIG process ───────────────────────────────────────────────────────────────

@dataclass
class NIGParams:
    """Normal Inverse Gaussian process (Barndorff-Nielsen 1997)."""
    alpha: float = 15.0   # tail heaviness
    beta: float = -5.0    # skewness
    delta: float = 0.5    # scale
    mu: float = 0.0       # location

    @property
    def gamma(self) -> float:
        return math.sqrt(max(self.alpha ** 2 - self.beta ** 2, 1e-10))


def nig_simulate(
    params: NIGParams,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    S0: float = 100.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate NIG process paths."""
    rng = rng or np.random.default_rng()
    p = params
    dt = T / n_steps

    S = np.full((n_paths, n_steps + 1), S0)

    for i in range(n_steps):
        # NIG via inverse Gaussian subordination
        ig_mean = p.delta * dt / p.gamma
        ig_shape = (p.delta * dt) ** 2
        # Inverse Gaussian via Michael-Schucany-Haas algorithm
        v = rng.standard_normal(n_paths)
        y = v ** 2
        x = ig_mean + ig_mean ** 2 * y / (2 * ig_shape) - ig_mean / (2 * ig_shape) * np.sqrt(
            4 * ig_mean * ig_shape * y + ig_mean ** 2 * y ** 2
        )
        u = rng.uniform(0, 1, n_paths)
        ig_sample = np.where(u <= ig_mean / (ig_mean + x), x, ig_mean ** 2 / x)

        Z = rng.standard_normal(n_paths)
        X = p.mu * dt + p.beta * ig_sample + np.sqrt(ig_sample) * Z
        S[:, i + 1] = S[:, i] * np.exp(X)

    return S


# ── Jump intensity estimation ─────────────────────────────────────────────────

def estimate_jump_intensity(
    returns: np.ndarray,
    threshold_sigma: float = 3.0,
    annualize: int = 252,
) -> dict:
    """
    Estimate Poisson jump intensity from returns.
    Jumps defined as |r| > threshold_sigma * rolling_std.
    """
    sigma = returns.std()
    threshold = threshold_sigma * sigma
    jump_mask = np.abs(returns) > threshold
    n_jumps = jump_mask.sum()
    lam = float(n_jumps / len(returns) * annualize)

    jump_returns = returns[jump_mask]
    return {
        "lambda": lam,
        "n_jumps": int(n_jumps),
        "mean_jump": float(jump_returns.mean()) if len(jump_returns) > 0 else 0.0,
        "std_jump": float(jump_returns.std()) if len(jump_returns) > 1 else 0.0,
        "up_fraction": float((jump_returns > 0).mean()) if len(jump_returns) > 0 else 0.5,
        "threshold": float(threshold),
    }
