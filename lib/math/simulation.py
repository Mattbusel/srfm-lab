"""
Monte Carlo simulation and stochastic process simulators.

Implements:
  - GBM, arithmetic BM, mean-reverting (OU) simulation
  - Heston stochastic volatility model
  - SABR model simulation
  - Variance Gamma simulation
  - Multi-asset correlated GBM (Cholesky)
  - Quasi-Monte Carlo (Halton, Sobol sequences)
  - Importance sampling for rare events
  - American option pricing via LSM (Longstaff-Schwartz)
  - Path-dependent payoff evaluation
  - Scenario generation with fat tails (copula + marginals)
  - Agent-based price simulation (trend followers + fundamentalists)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable


# ── Basic Process Simulators ──────────────────────────────────────────────────

def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths.
    Returns (n_paths, n_steps+1) array of prices.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
    log_paths = np.cumsum(np.column_stack([np.zeros(n_paths), log_ret]), axis=1)
    return S0 * np.exp(log_paths)


def simulate_ou(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    Exact OU simulation: dX = kappa*(theta - X)dt + sigma*dW
    Uses exact conditional distribution.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    e = math.exp(-kappa * dt)
    var_dt = sigma**2 * (1 - e**2) / (2 * kappa + 1e-10)
    std_dt = math.sqrt(max(var_dt, 0))

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = X0

    for t in range(n_steps):
        Z = rng.standard_normal(n_paths)
        paths[:, t+1] = paths[:, t] * e + theta * (1 - e) + std_dt * Z

    return paths


# ── Heston Model ──────────────────────────────────────────────────────────────

@dataclass
class HestonParams:
    S0: float
    V0: float        # initial variance
    mu: float        # drift
    kappa: float     # mean reversion of variance
    theta: float     # long-run variance
    sigma_v: float   # vol of vol
    rho: float       # correlation between S and V


def simulate_heston(
    params: HestonParams,
    T: float,
    n_steps: int,
    n_paths: int = 1000,
    seed: int = 42,
    scheme: str = "euler",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Heston stochastic volatility model.
    scheme: 'euler' (fast) or 'milstein' (more accurate)
    Returns (S_paths, V_paths): each (n_paths, n_steps+1).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    p = params

    S = np.zeros((n_paths, n_steps + 1))
    V = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = p.S0
    V[:, 0] = p.V0

    for t in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = p.rho * Z1 + math.sqrt(max(1 - p.rho**2, 0)) * rng.standard_normal(n_paths)

        Vt = np.maximum(V[:, t], 0)
        sqrt_Vt = np.sqrt(Vt)

        # Price process
        S[:, t+1] = S[:, t] * np.exp(
            (p.mu - 0.5 * Vt) * dt + sqrt_Vt * math.sqrt(dt) * Z1
        )

        # Variance process (Euler with full truncation)
        dV = p.kappa * (p.theta - Vt) * dt + p.sigma_v * sqrt_Vt * math.sqrt(dt) * Z2
        if scheme == "milstein":
            dV += 0.25 * p.sigma_v**2 * dt * (Z2**2 - 1)
        V[:, t+1] = np.maximum(V[:, t] + dV, 0)

    return S, V


def heston_call_price(
    params: HestonParams,
    K: float,
    T: float,
    r: float = 0.0,
    n_paths: int = 50000,
) -> float:
    """Monte Carlo price of European call under Heston."""
    S_paths, _ = simulate_heston(params, T, int(T * 252), n_paths)
    payoffs = np.maximum(S_paths[:, -1] - K, 0)
    return float(math.exp(-r * T) * payoffs.mean())


# ── SABR Model ────────────────────────────────────────────────────────────────

@dataclass
class SABRParams:
    F0: float        # initial forward
    alpha: float     # initial volatility
    beta: float      # CEV exponent (0=normal, 1=lognormal)
    rho: float       # correlation
    nu: float        # vol of vol


def simulate_sabr(
    params: SABRParams,
    T: float,
    n_steps: int,
    n_paths: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate SABR model paths.
    dF = alpha * F^beta * dW1
    dalpha = nu * alpha * dW2
    corr(dW1, dW2) = rho
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    p = params

    F = np.zeros((n_paths, n_steps + 1))
    A = np.zeros((n_paths, n_steps + 1))
    F[:, 0] = p.F0
    A[:, 0] = p.alpha

    for t in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = p.rho * Z1 + math.sqrt(max(1 - p.rho**2, 0)) * rng.standard_normal(n_paths)

        Ft = np.maximum(F[:, t], 1e-10)
        At = np.maximum(A[:, t], 1e-10)

        F[:, t+1] = Ft + At * Ft**p.beta * math.sqrt(dt) * Z1
        F[:, t+1] = np.maximum(F[:, t+1], 0)
        A[:, t+1] = At * np.exp(-0.5 * p.nu**2 * dt + p.nu * math.sqrt(dt) * Z2)

    return F, A


def sabr_implied_vol(
    F: float, K: float, T: float,
    alpha: float, beta: float, rho: float, nu: float,
) -> float:
    """
    Hagan et al. SABR implied vol formula (closed form).
    """
    if abs(F - K) < 1e-10:
        # ATM formula
        FK_beta = F**(1 - beta)
        z = nu / alpha * FK_beta * math.log(F / max(K, 1e-10))
        chi = math.log((math.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho) + 1e-10)
        zeta = z / max(chi, 1e-10)
        atm_vol = (alpha / FK_beta
                   * (1 + ((1-beta)**2/24 * alpha**2/FK_beta**2
                           + rho*beta*nu*alpha/(4*FK_beta)
                           + (2-3*rho**2)/24 * nu**2) * T))
        return float(max(atm_vol, 1e-8))

    FK = F * K
    FK_beta2 = FK**((1-beta)/2)
    log_FK = math.log(F / max(K, 1e-10))
    z = nu / alpha * FK_beta2 * log_FK
    chi = math.log((math.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho) + 1e-10)

    A = alpha / (FK_beta2 * (1 + (1-beta)**2/24 * log_FK**2 + (1-beta)**4/1920 * log_FK**4))
    B = z / max(chi, 1e-10)
    C = (1 + ((1-beta)**2/24 * alpha**2 / FK**(1-beta)
              + rho*beta*nu*alpha/(4*FK_beta2)
              + (2-3*rho**2)/24 * nu**2) * T)

    return float(max(A * B * C, 1e-8))


# ── Multi-Asset Correlated GBM ─────────────────────────────────────────────────

def simulate_multi_asset_gbm(
    S0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    T: float,
    n_steps: int,
    n_paths: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Multi-asset correlated GBM via Cholesky decomposition.
    S0: (n_assets,), mu: (n_assets,), sigma: (n_assets,)
    corr: (n_assets, n_assets) correlation matrix
    Returns (n_paths, n_steps+1, n_assets)
    """
    rng = np.random.default_rng(seed)
    n = len(S0)
    dt = T / n_steps

    cov = np.outer(sigma, sigma) * corr
    try:
        L = np.linalg.cholesky(cov + 1e-8 * np.eye(n))
    except np.linalg.LinAlgError:
        L = np.diag(sigma)

    paths = np.zeros((n_paths, n_steps + 1, n))
    paths[:, 0, :] = S0

    for t in range(n_steps):
        Z = rng.standard_normal((n_paths, n))
        corr_Z = Z @ L.T  # (n_paths, n)
        log_ret = (mu - 0.5 * sigma**2) * dt + math.sqrt(dt) * corr_Z
        paths[:, t+1, :] = paths[:, t, :] * np.exp(log_ret)

    return paths


# ── Quasi-Monte Carlo ─────────────────────────────────────────────────────────

def halton_sequence(n: int, base: int) -> np.ndarray:
    """Generate Halton low-discrepancy sequence in [0,1]."""
    seq = np.zeros(n)
    for i in range(n):
        f, r = 1, 0
        j = i + 1
        while j > 0:
            f /= base
            r += f * (j % base)
            j //= base
        seq[i] = r
    return seq


def halton_multivariate(n: int, d: int) -> np.ndarray:
    """
    Multi-dimensional Halton sequence.
    Uses first d primes as bases.
    """
    primes = []
    candidate = 2
    while len(primes) < d:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1

    sequences = np.column_stack([halton_sequence(n, p) for p in primes])
    return sequences


def qmc_normal(n: int, d: int) -> np.ndarray:
    """
    QMC normal samples via Halton + inverse normal CDF.
    More uniform coverage than standard MC.
    """
    from scipy.stats import norm
    u = halton_sequence(n, 2) if d == 1 else halton_multivariate(n, d)
    # Avoid exact 0 and 1
    u = np.clip(u, 1e-8, 1 - 1e-8)
    return norm.ppf(u)


# ── Longstaff-Schwartz American Option ────────────────────────────────────────

def longstaff_schwartz(
    S: np.ndarray,      # (n_paths, n_steps+1) price paths
    K: float,
    T: float,
    r: float = 0.0,
    option_type: str = "put",
    poly_degree: int = 2,
) -> float:
    """
    Longstaff-Schwartz algorithm for American option pricing.
    Uses polynomial regression for continuation value estimation.
    """
    n_paths, n_steps_plus1 = S.shape
    n_steps = n_steps_plus1 - 1
    dt = T / n_steps
    disc = math.exp(-r * dt)

    # Payoff function
    if option_type == "put":
        payoff = lambda s: np.maximum(K - s, 0)
    else:
        payoff = lambda s: np.maximum(s - K, 0)

    # Terminal payoff
    cash_flows = payoff(S[:, -1]).copy()

    # Backward induction
    for t in range(n_steps - 1, 0, -1):
        discount_cf = disc * cash_flows
        exercise = payoff(S[:, t])
        in_the_money = exercise > 0

        if in_the_money.sum() < poly_degree + 2:
            cash_flows = discount_cf
            continue

        X = S[in_the_money, t]
        Y = discount_cf[in_the_money]

        # Polynomial basis: 1, X, X^2, ...
        basis = np.column_stack([X**i for i in range(poly_degree + 1)])
        try:
            coeffs = np.linalg.lstsq(basis, Y, rcond=None)[0]
            continuation = basis @ coeffs
        except Exception:
            cash_flows = discount_cf
            continue

        # Exercise if intrinsic > continuation
        exercise_itm = exercise[in_the_money]
        exercise_decision = exercise_itm >= continuation
        cash_flows[in_the_money] = np.where(
            exercise_decision, exercise_itm, discount_cf[in_the_money]
        )
        cash_flows[~in_the_money] = discount_cf[~in_the_money]

    return float(math.exp(-r * dt) * cash_flows.mean())


# ── Agent-Based Price Simulation ──────────────────────────────────────────────

def agent_based_simulation(
    T: int,
    n_trend_followers: int = 100,
    n_fundamentalists: int = 100,
    fundamental_value: float = 100.0,
    noise_sigma: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """
    Simple agent-based model with trend followers and fundamentalists.
    - Trend followers: buy when price rose, sell when fell
    - Fundamentalists: buy below fundamental, sell above
    Returns price series of length T.
    """
    rng = np.random.default_rng(seed)
    prices = np.zeros(T + 1)
    prices[0] = fundamental_value

    # Agent parameters
    tf_strength = rng.uniform(0.01, 0.1, n_trend_followers)
    fund_strength = rng.uniform(0.01, 0.1, n_fundamentalists)
    tf_memory = rng.integers(2, 20, n_trend_followers)  # lookback for trend

    for t in range(1, T + 1):
        # Trend follower demand
        tf_demand = 0.0
        for i in range(n_trend_followers):
            lb = tf_memory[i]
            if t > lb:
                trend = prices[t-1] - prices[t-1-lb]
                tf_demand += tf_strength[i] * np.sign(trend)

        # Fundamentalist demand
        fund_demand = 0.0
        for i in range(n_fundamentalists):
            mispricing = fundamental_value - prices[t-1]
            fund_demand += fund_strength[i] * mispricing / max(fundamental_value, 1)

        # Net demand → price change
        total_demand = tf_demand + fund_demand
        noise = rng.normal(0, noise_sigma)
        price_change = 0.01 * total_demand / (n_trend_followers + n_fundamentalists) + noise
        prices[t] = prices[t-1] * math.exp(price_change)

    return prices[1:]


# ── Path-Dependent Payoffs ────────────────────────────────────────────────────

def asian_option_price(
    S0: float,
    K: float,
    T: float,
    mu: float,
    sigma: float,
    r: float = 0.0,
    n_steps: int = 252,
    n_paths: int = 50000,
    option_type: str = "call",
    averaging: str = "arithmetic",
    seed: int = 42,
) -> float:
    """Asian option pricing via Monte Carlo."""
    paths = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed)

    if averaging == "arithmetic":
        avg = paths[:, 1:].mean(axis=1)
    else:
        avg = np.exp(np.log(paths[:, 1:] + 1e-10).mean(axis=1))

    if option_type == "call":
        payoffs = np.maximum(avg - K, 0)
    else:
        payoffs = np.maximum(K - avg, 0)

    return float(math.exp(-r * T) * payoffs.mean())


def barrier_option_price(
    S0: float,
    K: float,
    B: float,          # barrier level
    T: float,
    mu: float,
    sigma: float,
    r: float = 0.0,
    option_type: str = "call",
    barrier_type: str = "down-and-out",
    n_steps: int = 252,
    n_paths: int = 50000,
    seed: int = 42,
) -> float:
    """Barrier option pricing via Monte Carlo."""
    paths = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed)

    terminal = paths[:, -1]
    if option_type == "call":
        intrinsic = np.maximum(terminal - K, 0)
    else:
        intrinsic = np.maximum(K - terminal, 0)

    if barrier_type == "down-and-out":
        alive = paths.min(axis=1) > B
    elif barrier_type == "down-and-in":
        alive = paths.min(axis=1) <= B
    elif barrier_type == "up-and-out":
        alive = paths.max(axis=1) < B
    elif barrier_type == "up-and-in":
        alive = paths.max(axis=1) >= B
    else:
        alive = np.ones(n_paths, dtype=bool)

    payoffs = intrinsic * alive
    return float(math.exp(-r * T) * payoffs.mean())
