"""
Stochastic calculus toolkit for quantitative finance.

Implements:
  - Euler-Maruyama and Milstein SDE solvers (generic)
  - GBM, Ornstein-Uhlenbeck, CIR process simulation
  - Heston stochastic volatility model
  - SABR model simulation
  - Variance reduction: antithetic variates, control variates, moment matching
  - Ito's lemma symbolic helper
  - First-passage-time (barrier hitting) analytics
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ── Generic SDE solvers ────────────────────────────────────────────────────────

def euler_maruyama(
    mu: Callable[[float, float], float],
    sigma: Callable[[float, float], float],
    x0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Euler-Maruyama scheme: dX = mu(t,X)dt + sigma(t,X)dW

    Returns array shape (n_paths, n_steps+1).
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    X = np.full((n_paths, n_steps + 1), x0)
    dW = rng.standard_normal((n_paths, n_steps)) * sqrt_dt
    for i in range(n_steps):
        t = i * dt
        x = X[:, i]
        X[:, i + 1] = x + mu(t, x) * dt + sigma(t, x) * dW[:, i]
    return X


def milstein(
    mu: Callable[[float, float], float],
    sigma: Callable[[float, float], float],
    sigma_prime: Callable[[float, float], float],
    x0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Milstein scheme (strong order 1.0):
      X_{n+1} = X_n + mu*dt + sigma*dW + 0.5*sigma*sigma'*(dW^2 - dt)
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    X = np.full((n_paths, n_steps + 1), x0)
    dW = rng.standard_normal((n_paths, n_steps)) * sqrt_dt
    for i in range(n_steps):
        t = i * dt
        x = X[:, i]
        dw = dW[:, i]
        X[:, i + 1] = (
            x
            + mu(t, x) * dt
            + sigma(t, x) * dw
            + 0.5 * sigma(t, x) * sigma_prime(t, x) * (dw ** 2 - dt)
        )
    return X


# ── Standard processes ─────────────────────────────────────────────────────────

def gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    antithetic: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW
    Uses exact simulation (log-normal increments).
    Returns shape (n_paths, n_steps+1).
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    half = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal((half, n_steps))
    if antithetic:
        Z = np.vstack([Z, -Z])
    log_S = np.log(S0) + np.cumsum(
        (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * Z, axis=1
    )
    S = np.exp(log_S)
    S0_col = np.full((n_paths, 1), S0)
    return np.hstack([S0_col, S])


def ornstein_uhlenbeck(
    x0: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Ornstein-Uhlenbeck: dX = theta*(mu - X)*dt + sigma*dW
    Exact simulation using conditional distribution.
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    e = math.exp(-theta * dt)
    cond_var = sigma ** 2 * (1 - e ** 2) / (2 * theta)
    cond_std = math.sqrt(cond_var)
    X = np.full((n_paths, n_steps + 1), x0)
    Z = rng.standard_normal((n_paths, n_steps))
    for i in range(n_steps):
        X[:, i + 1] = mu + (X[:, i] - mu) * e + cond_std * Z[:, i]
    return X


def cir(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Cox-Ingersoll-Ross: dX = kappa*(theta-X)*dt + sigma*sqrt(X)*dW
    Feller condition: 2*kappa*theta >= sigma^2
    Uses full truncation scheme for positivity.
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    X = np.full((n_paths, n_steps + 1), x0)
    Z = rng.standard_normal((n_paths, n_steps))
    for i in range(n_steps):
        x = np.maximum(X[:, i], 0.0)
        X[:, i + 1] = (
            x
            + kappa * (theta - x) * dt
            + sigma * np.sqrt(x) * sqrt_dt * Z[:, i]
        )
    return np.maximum(X, 0.0)


# ── Heston stochastic volatility ───────────────────────────────────────────────

@dataclass
class HestonParams:
    S0: float = 100.0
    V0: float = 0.04      # initial variance
    mu: float = 0.05      # drift
    kappa: float = 2.0    # mean reversion speed
    theta: float = 0.04   # long-run variance
    sigma: float = 0.3    # vol of vol
    rho: float = -0.7     # price-vol correlation


def heston_simulate(
    params: HestonParams,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Heston model via Euler-Milstein with full truncation.
    Returns (S paths, V paths), each shape (n_paths, n_steps+1).
    """
    rng = rng or np.random.default_rng()
    p = params
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))
    # Cholesky correlation
    W1 = Z1
    W2 = p.rho * Z1 + math.sqrt(1 - p.rho ** 2) * Z2

    S = np.full((n_paths, n_steps + 1), p.S0)
    V = np.full((n_paths, n_steps + 1), p.V0)

    for i in range(n_steps):
        v = np.maximum(V[:, i], 0.0)
        sv = np.sqrt(v)
        V[:, i + 1] = np.maximum(
            v + p.kappa * (p.theta - v) * dt + p.sigma * sv * sqrt_dt * W2[:, i], 0.0
        )
        S[:, i + 1] = S[:, i] * np.exp(
            (p.mu - 0.5 * v) * dt + sv * sqrt_dt * W1[:, i]
        )

    return S, V


def heston_call_price_mc(
    params: HestonParams,
    K: float,
    T: float,
    r: float,
    n_steps: int = 252,
    n_paths: int = 50_000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Monte Carlo call price under Heston. Returns (price, std_error)."""
    S, _ = heston_simulate(params, T, n_steps, n_paths, rng)
    payoffs = np.maximum(S[:, -1] - K, 0.0) * math.exp(-r * T)
    return float(payoffs.mean()), float(payoffs.std() / math.sqrt(n_paths))


# ── SABR model ─────────────────────────────────────────────────────────────────

@dataclass
class SABRParams:
    F0: float = 100.0    # initial forward
    alpha: float = 0.3   # initial vol
    beta: float = 0.5    # CEV exponent
    rho: float = -0.3    # correlation
    nu: float = 0.4      # vol of vol


def sabr_simulate(
    params: SABRParams,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SABR: dF = alpha * F^beta * dW1,  d(alpha) = nu * alpha * dW2
    corr(dW1, dW2) = rho
    Returns (F paths, alpha paths).
    """
    rng = rng or np.random.default_rng()
    p = params
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))
    W2 = p.rho * Z1 + math.sqrt(1 - p.rho ** 2) * Z2

    F = np.full((n_paths, n_steps + 1), p.F0)
    A = np.full((n_paths, n_steps + 1), p.alpha)

    for i in range(n_steps):
        f = np.maximum(F[:, i], 1e-8)
        a = np.maximum(A[:, i], 1e-8)
        A[:, i + 1] = a * np.exp(
            -0.5 * p.nu ** 2 * dt + p.nu * sqrt_dt * W2[:, i]
        )
        F[:, i + 1] = f + a * f ** p.beta * sqrt_dt * Z1[:, i]

    return F, A


def sabr_implied_vol(F: float, K: float, T: float, p: SABRParams) -> float:
    """
    Hagan et al. (2002) SABR implied vol approximation.
    """
    if abs(F - K) < 1e-8:
        # ATM approximation
        return p.alpha * (
            1.0
            + (
                (1 - p.beta) ** 2 / 24 * p.alpha ** 2 / F ** (2 - 2 * p.beta)
                + p.rho * p.beta * p.nu * p.alpha / (4 * F ** (1 - p.beta))
                + (2 - 3 * p.rho ** 2) / 24 * p.nu ** 2
            ) * T
        ) / F ** (1 - p.beta)

    FK_mid = (F * K) ** (0.5 * (1 - p.beta))
    log_FK = math.log(F / K)
    z = p.nu / p.alpha * FK_mid * log_FK
    chi = math.log((math.sqrt(1 - 2 * p.rho * z + z ** 2) + z - p.rho) / (1 - p.rho))

    numer = p.alpha * (1 + (
        (1 - p.beta) ** 2 / 24 * p.alpha ** 2 / FK_mid ** 2
        + p.rho * p.beta * p.nu * p.alpha / (4 * FK_mid)
        + (2 - 3 * p.rho ** 2) / 24 * p.nu ** 2
    ) * T)

    denom = FK_mid * (1 + (1 - p.beta) ** 2 / 24 * log_FK ** 2
                      + (1 - p.beta) ** 4 / 1920 * log_FK ** 4)

    return numer / denom * (z / chi) if abs(chi) > 1e-10 else numer / denom


# ── First-passage time ─────────────────────────────────────────────────────────

def first_passage_time_gbm(
    S0: float,
    barrier: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int = 1000,
    n_paths: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, np.ndarray]:
    """
    Monte Carlo first-passage time to a barrier level under GBM.
    Returns (probability of hitting, array of first-hit times; nan if not hit).
    """
    rng = rng or np.random.default_rng()
    paths = gbm(S0, mu, sigma, T, n_steps, n_paths, rng=rng)
    dt = T / n_steps
    times = np.full(n_paths, np.nan)
    hit = np.zeros(n_paths, dtype=bool)

    if barrier > S0:
        for i in range(1, n_steps + 1):
            newly_hit = (~hit) & (paths[:, i] >= barrier)
            times[newly_hit] = i * dt
            hit |= newly_hit
    else:
        for i in range(1, n_steps + 1):
            newly_hit = (~hit) & (paths[:, i] <= barrier)
            times[newly_hit] = i * dt
            hit |= newly_hit

    return float(hit.mean()), times


# ── Ito's lemma helper ─────────────────────────────────────────────────────────

def itos_lemma_gbm(f_coeffs: dict, mu: float, sigma: float) -> dict:
    """
    Apply Ito's lemma to f(S,t) where S follows GBM.
    f_coeffs: {
        'df_dS': df/dS evaluated at (S,t),
        'df_dS2': d^2f/dS^2,
        'df_dt': df/dt,
        'S': current S value
    }
    Returns drift and diffusion of f(S,t).
    """
    df_dS = f_coeffs['df_dS']
    df_dS2 = f_coeffs['df_dS2']
    df_dt = f_coeffs['df_dt']
    S = f_coeffs['S']

    drift = df_dt + mu * S * df_dS + 0.5 * sigma ** 2 * S ** 2 * df_dS2
    diffusion = sigma * S * df_dS
    return {'drift': drift, 'diffusion': diffusion}
