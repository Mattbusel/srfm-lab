"""
stochastic_processes.py -- Advanced stochastic processes for quantitative finance.

Brownian motion, OU, CIR, Heston, SABR, Variance Gamma, NIG, CGMY,
Levy processes, Hawkes process, CTMCs, subordinators, and self-exciting processes.

All numerics via numpy/scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats, special, linalg
from scipy.optimize import minimize, minimize_scalar
from scipy.fft import fft, ifft

FloatArray = NDArray[np.float64]


# ===================================================================
# 1.  Brownian Motion
# ===================================================================

def brownian_motion(
    n_steps: int, dt: float = 1.0 / 252, n_paths: int = 1, seed: int = 42
) -> FloatArray:
    """Standard Brownian motion: W(t+dt) = W(t) + sqrt(dt)*Z."""
    rng = np.random.default_rng(seed)
    dW = rng.standard_normal((n_steps, n_paths)) * np.sqrt(dt)
    W = np.zeros((n_steps + 1, n_paths))
    W[1:] = np.cumsum(dW, axis=0)
    return W


def geometric_brownian_motion(
    S0: float, mu: float, sigma: float,
    n_steps: int, dt: float = 1.0 / 252,
    n_paths: int = 1, seed: int = 42,
) -> FloatArray:
    """GBM: dS = mu*S*dt + sigma*S*dW."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_steps, n_paths))
    log_ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    S = np.zeros((n_steps + 1, n_paths))
    S[0] = S0
    S[1:] = S0 * np.exp(np.cumsum(log_ret, axis=0))
    return S


def fractional_brownian_motion(
    n_steps: int, hurst: float = 0.7, n_paths: int = 1, seed: int = 42
) -> FloatArray:
    """Fractional Brownian motion via Cholesky decomposition.
    H > 0.5: persistent, H < 0.5: anti-persistent, H = 0.5: standard BM."""
    rng = np.random.default_rng(seed)
    H = hurst
    # Build covariance matrix
    n = n_steps
    idx = np.arange(n + 1, dtype=float)
    cov = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            cov[i, j] = 0.5 * (
                abs(i) ** (2 * H) + abs(j) ** (2 * H) - abs(i - j) ** (2 * H)
            )
    # Add small diagonal for numerical stability
    cov += np.eye(n + 1) * 1e-10
    L = np.linalg.cholesky(cov)
    paths = np.zeros((n + 1, n_paths))
    for p in range(n_paths):
        Z = rng.standard_normal(n + 1)
        paths[:, p] = L @ Z
    return paths


def correlated_brownian_motions(
    n_steps: int, corr_matrix: FloatArray, dt: float = 1.0 / 252,
    seed: int = 42,
) -> FloatArray:
    """Generate correlated Brownian motions.
    Returns (n_steps+1, n_assets)."""
    n_assets = corr_matrix.shape[0]
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr_matrix + np.eye(n_assets) * 1e-10)
    Z = rng.standard_normal((n_steps, n_assets))
    dW = Z @ L.T * np.sqrt(dt)
    W = np.zeros((n_steps + 1, n_assets))
    W[1:] = np.cumsum(dW, axis=0)
    return W


# ===================================================================
# 2.  Ornstein-Uhlenbeck Process
# ===================================================================

@dataclass
class OUParams:
    kappa: float = 5.0          # mean-reversion speed
    mu: float = 0.0             # long-run mean
    sigma: float = 0.3          # volatility


def ou_simulate(
    params: OUParams, X0: float, n_steps: int, dt: float = 1.0 / 252,
    n_paths: int = 1, seed: int = 42,
) -> FloatArray:
    """Exact simulation of OU process."""
    rng = np.random.default_rng(seed)
    kappa, mu, sigma = params.kappa, params.mu, params.sigma
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = X0
    exp_k = np.exp(-kappa * dt)
    var_dt = sigma ** 2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt))
    std_dt = np.sqrt(max(var_dt, 1e-12))
    for t in range(n_steps):
        paths[t + 1] = mu + (paths[t] - mu) * exp_k + std_dt * rng.standard_normal(n_paths)
    return paths


def ou_mle(data: FloatArray, dt: float = 1.0 / 252) -> OUParams:
    """Maximum likelihood estimation of OU parameters from time series."""
    x = data.flatten()
    n = len(x) - 1
    if n < 3:
        return OUParams()
    x0 = x[:-1]
    x1 = x[1:]
    Sx = x0.sum()
    Sy = x1.sum()
    Sxx = (x0 ** 2).sum()
    Sxy = (x0 * x1).sum()
    Syy = (x1 ** 2).sum()
    denom = n * Sxx - Sx ** 2
    if abs(denom) < 1e-12:
        return OUParams()
    a = (Sy * Sxx - Sx * Sxy) / denom
    b = (n * Sxy - Sx * Sy) / denom
    if b <= 0 or b >= 1:
        return OUParams()
    kappa = -np.log(b) / dt
    mu = a / (1 - b)
    residuals = x1 - a - b * x0
    sigma_e = residuals.std()
    sigma = sigma_e * np.sqrt(2 * kappa / (1 - np.exp(-2 * kappa * dt)))
    return OUParams(kappa=float(kappa), mu=float(mu), sigma=float(sigma))


def ou_mean_reversion_time(params: OUParams) -> float:
    """Half-life of mean reversion: ln(2) / kappa."""
    return float(np.log(2) / (params.kappa + 1e-12))


def ou_stationary_distribution(params: OUParams) -> Tuple[float, float]:
    """Mean and variance of stationary distribution."""
    var = params.sigma ** 2 / (2 * params.kappa + 1e-12)
    return params.mu, var


# ===================================================================
# 3.  CIR Process
# ===================================================================

@dataclass
class CIRParams:
    kappa: float = 0.5
    theta: float = 0.04
    sigma: float = 0.1
    x0: float = 0.04


def cir_simulate_exact(
    params: CIRParams, n_steps: int, dt: float = 1.0 / 252,
    n_paths: int = 1, seed: int = 42,
) -> FloatArray:
    """Exact simulation of CIR via non-central chi-squared."""
    rng = np.random.default_rng(seed)
    kappa, theta, sigma, x0 = params.kappa, params.theta, params.sigma, params.x0
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = x0
    exp_k = np.exp(-kappa * dt)
    c = sigma ** 2 * (1 - exp_k) / (4 * kappa + 1e-12)
    for t in range(n_steps):
        d = 4 * kappa * theta / (sigma ** 2 + 1e-12)
        lam = paths[t] * exp_k / (c + 1e-12)
        lam = np.maximum(lam, 0)
        if c > 0:
            paths[t + 1] = c * rng.noncentral_chisquare(d, lam, size=n_paths)
        else:
            paths[t + 1] = theta
        paths[t + 1] = np.maximum(paths[t + 1], 0)
    return paths


def cir_simulate_euler(
    params: CIRParams, n_steps: int, dt: float = 1.0 / 252,
    n_paths: int = 1, seed: int = 42,
) -> FloatArray:
    """Euler-Maruyama simulation of CIR with full truncation."""
    rng = np.random.default_rng(seed)
    kappa, theta, sigma, x0 = params.kappa, params.theta, params.sigma, params.x0
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = x0
    for t in range(n_steps):
        x_pos = np.maximum(paths[t], 0)
        dW = rng.standard_normal(n_paths) * np.sqrt(dt)
        dx = kappa * (theta - x_pos) * dt + sigma * np.sqrt(x_pos) * dW
        paths[t + 1] = np.maximum(paths[t] + dx, 0)
    return paths


def cir_bond_price(
    r: float, T: float, params: CIRParams
) -> float:
    """Zero-coupon bond price under CIR short rate model."""
    kappa, theta, sigma = params.kappa, params.theta, params.sigma
    gamma = np.sqrt(kappa ** 2 + 2 * sigma ** 2)
    eg = np.exp(gamma * T)
    denom = (gamma + kappa) * (eg - 1) + 2 * gamma
    B = 2 * (eg - 1) / (denom + 1e-12)
    A_exp = (2 * gamma * np.exp((kappa + gamma) * T / 2) / (denom + 1e-12))
    A = A_exp ** (2 * kappa * theta / (sigma ** 2 + 1e-12))
    return float(A * np.exp(-B * r))


# ===================================================================
# 4.  Heston Model
# ===================================================================

@dataclass
class HestonParams:
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    S0: float = 100.0
    r: float = 0.02


def heston_simulate_qe(
    params: HestonParams, n_steps: int, dt: float = 1.0 / 252,
    n_paths: int = 1, seed: int = 42,
) -> Tuple[FloatArray, FloatArray]:
    """Quadratic-Exponential (QE) scheme for Heston.
    Returns (price_paths, variance_paths)."""
    rng = np.random.default_rng(seed)
    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
    S = np.zeros((n_steps + 1, n_paths))
    V = np.zeros((n_steps + 1, n_paths))
    S[0] = params.S0
    V[0] = params.v0

    for t in range(n_steps):
        v = np.maximum(V[t], 0)
        exp_k = np.exp(-kappa * dt)
        m = theta + (v - theta) * exp_k
        s2 = (
            v * xi ** 2 * exp_k / (kappa + 1e-12) * (1 - exp_k)
            + theta * xi ** 2 / (2 * kappa + 1e-12) * (1 - exp_k) ** 2
        )
        s2 = np.maximum(s2, 1e-12)
        psi = s2 / (m ** 2 + 1e-30)

        U = rng.random(n_paths)
        V_new = np.empty(n_paths)

        # Quadratic: psi <= 1.5
        mask_q = psi <= 1.5
        if mask_q.any():
            b2 = 2.0 / psi[mask_q] - 1.0 + np.sqrt(2.0 / psi[mask_q]) * np.sqrt(np.maximum(2.0 / psi[mask_q] - 1.0, 0.0))
            a_c = m[mask_q] / (1.0 + b2)
            Zv = stats.norm.ppf(np.clip(U[mask_q], 1e-12, 1 - 1e-12))
            V_new[mask_q] = a_c * (np.sqrt(b2) + Zv) ** 2

        # Exponential: psi > 1.5
        mask_e = ~mask_q
        if mask_e.any():
            p_exp = (psi[mask_e] - 1) / (psi[mask_e] + 1e-30)
            beta_e = (1 - p_exp) / (m[mask_e] + 1e-30)
            V_new[mask_e] = np.where(
                U[mask_e] <= p_exp,
                0.0,
                np.log((1 - p_exp) / (1 - U[mask_e] + 1e-30)) / (beta_e + 1e-30),
            )

        V[t + 1] = np.maximum(V_new, 0)

        # Price step
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        Zs = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2
        sqrt_v = np.sqrt(np.maximum(V[t], 0))
        log_S = np.log(S[t]) + (params.r - 0.5 * V[t]) * dt + sqrt_v * np.sqrt(dt) * Zs
        S[t + 1] = np.exp(log_S)

    return S, V


def heston_characteristic_function(
    u: complex, T: float, params: HestonParams
) -> complex:
    """Heston characteristic function phi(u) = E[exp(iu * log(S_T))]."""
    kappa, theta, xi, rho, v0, r = (
        params.kappa, params.theta, params.xi, params.rho, params.v0, params.r
    )
    d = np.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi ** 2 * (1j * u + u ** 2))
    g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d + 1e-30)
    exp_dT = np.exp(-d * T)
    C = (params.r * 1j * u * T
         + kappa * theta / xi ** 2 * (
             (kappa - rho * xi * 1j * u - d) * T
             - 2 * np.log((1 - g * exp_dT) / (1 - g + 1e-30))
         ))
    D = ((kappa - rho * xi * 1j * u - d) / xi ** 2
         * (1 - exp_dT) / (1 - g * exp_dT + 1e-30))
    return np.exp(C + D * v0)


# ===================================================================
# 5.  SABR Model
# ===================================================================

@dataclass
class SABRParams:
    alpha: float = 0.3          # initial vol
    beta: float = 0.5           # CEV exponent
    rho: float = -0.3           # spot-vol correlation
    nu: float = 0.4             # vol of vol


def sabr_simulate(
    params: SABRParams, F0: float, n_steps: int, dt: float = 1.0 / 252,
    n_paths: int = 1, seed: int = 42,
) -> Tuple[FloatArray, FloatArray]:
    """Euler simulation of SABR model."""
    rng = np.random.default_rng(seed)
    alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu
    F = np.zeros((n_steps + 1, n_paths))
    sigma = np.zeros((n_steps + 1, n_paths))
    F[0] = F0
    sigma[0] = alpha
    for t in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2
        f_pos = np.maximum(F[t], 1e-8)
        dF = sigma[t] * f_pos ** beta * np.sqrt(dt) * W1
        dsigma = nu * sigma[t] * np.sqrt(dt) * W2
        F[t + 1] = np.maximum(F[t] + dF, 1e-8)
        sigma[t + 1] = np.maximum(sigma[t] + dsigma, 1e-8)
    return F, sigma


def sabr_implied_vol(
    F: float, K: float, T: float, params: SABRParams
) -> float:
    """Hagan's SABR implied volatility approximation."""
    alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu
    if abs(F - K) < 1e-10:
        # ATM formula
        Fb = F ** (1 - beta)
        vol = alpha / Fb * (
            1 + (
                (1 - beta) ** 2 / 24 * alpha ** 2 / Fb ** 2
                + rho * beta * nu * alpha / (4 * Fb)
                + (2 - 3 * rho ** 2) / 24 * nu ** 2
            ) * T
        )
        return float(vol)
    FK = F * K
    FK_mid = np.sqrt(FK)
    log_FK = np.log(F / K)
    z = nu / alpha * FK_mid ** (1 - beta) * log_FK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho + 1e-12))
    if abs(x_z) < 1e-10:
        x_z = 1.0
    prefix = alpha / (FK_mid ** (1 - beta) * (
        1 + (1 - beta) ** 2 / 24 * log_FK ** 2
        + (1 - beta) ** 4 / 1920 * log_FK ** 4
    ))
    correction = 1 + (
        (1 - beta) ** 2 / 24 * alpha ** 2 / (FK_mid ** (2 - 2 * beta))
        + rho * beta * nu * alpha / (4 * FK_mid ** (1 - beta))
        + (2 - 3 * rho ** 2) / 24 * nu ** 2
    ) * T
    return float(prefix * z / x_z * correction)


# ===================================================================
# 6.  Variance Gamma Process
# ===================================================================

@dataclass
class VarianceGammaParams:
    theta: float = -0.1         # drift of BM
    sigma: float = 0.2          # vol of BM
    nu: float = 0.5             # variance rate of Gamma subordinator


def variance_gamma_simulate(
    params: VarianceGammaParams, n_steps: int, dt: float = 1.0 / 252,
    n_paths: int = 1, seed: int = 42,
) -> FloatArray:
    """Simulate VG process via time-changed Brownian motion.
    X(t) = theta*G(t) + sigma*W(G(t)) where G is Gamma process."""
    rng = np.random.default_rng(seed)
    theta, sigma, nu = params.theta, params.sigma, params.nu
    paths = np.zeros((n_steps + 1, n_paths))
    for t in range(n_steps):
        # Gamma increments: shape = dt/nu, scale = nu
        dG = rng.gamma(shape=dt / nu, scale=nu, size=n_paths)
        Z = rng.standard_normal(n_paths)
        dX = theta * dG + sigma * np.sqrt(dG) * Z
        paths[t + 1] = paths[t] + dX
    return paths


def variance_gamma_characteristic(u: complex, T: float, params: VarianceGammaParams) -> complex:
    """Characteristic function of VG process."""
    theta, sigma, nu = params.theta, params.sigma, params.nu
    return (1 - 1j * u * theta * nu + 0.5 * sigma ** 2 * nu * u ** 2) ** (-T / nu)


# ===================================================================
# 7.  Normal Inverse Gaussian (NIG)
# ===================================================================

@dataclass
class NIGParams:
    alpha: float = 15.0         # tail heaviness
    beta: float = -5.0          # asymmetry
    delta: float = 0.5          # scale
    mu: float = 0.0             # location


def nig_simulate(
    params: NIGParams, n_steps: int, dt: float = 1.0 / 252,
    n_paths: int = 1, seed: int = 42,
) -> FloatArray:
    """Simulate NIG via inverse Gaussian subordinator."""
    rng = np.random.default_rng(seed)
    alpha, beta, delta, mu = params.alpha, params.beta, params.delta, params.mu
    gamma = np.sqrt(alpha ** 2 - beta ** 2)
    paths = np.zeros((n_steps + 1, n_paths))
    for t in range(n_steps):
        # Inverse Gaussian increments
        ig_mu = delta * dt / gamma
        ig_lam = delta ** 2 * dt ** 2
        # Simulate IG via normal
        Y = rng.standard_normal(n_paths) ** 2
        X = ig_mu + ig_mu ** 2 * Y / (2 * ig_lam) - ig_mu / (2 * ig_lam) * np.sqrt(
            4 * ig_mu * ig_lam * Y + ig_mu ** 2 * Y ** 2
        )
        U = rng.random(n_paths)
        dIG = np.where(U <= ig_mu / (ig_mu + X + 1e-12), X, ig_mu ** 2 / (X + 1e-12))
        dIG = np.maximum(dIG, 1e-12)
        Z = rng.standard_normal(n_paths)
        dX = mu * dt + beta * delta ** 2 * dIG + delta * np.sqrt(dIG) * Z
        paths[t + 1] = paths[t] + dX
    return paths


def nig_characteristic(u: complex, T: float, params: NIGParams) -> complex:
    """Characteristic function of NIG process."""
    alpha, beta, delta, mu = params.alpha, params.beta, params.delta, params.mu
    return np.exp(
        1j * u * mu * T
        + delta * T * (np.sqrt(alpha ** 2 - beta ** 2) - np.sqrt(alpha ** 2 - (beta + 1j * u) ** 2))
    )


# ===================================================================
# 8.  CGMY/KoBoL Process
# ===================================================================

@dataclass
class CGMYParams:
    C: float = 1.0              # activity
    G: float = 5.0              # exponential decay, negative jumps
    M: float = 10.0             # exponential decay, positive jumps
    Y: float = 0.5              # fine structure (0 < Y < 2, Y != 1)


def cgmy_characteristic(u: complex, T: float, params: CGMYParams) -> complex:
    """Characteristic function of CGMY process."""
    C, G, M, Y = params.C, params.G, params.M, params.Y
    if abs(Y - 1.0) < 1e-6:
        # Degenerate case
        psi = C * (
            -np.log(1 - 1j * u / M) * M
            - np.log(1 + 1j * u / G) * G
        )
    else:
        psi = C * special.gamma(-Y) * (
            (M - 1j * u) ** Y - M ** Y
            + (G + 1j * u) ** Y - G ** Y
        )
    return np.exp(T * psi)


def cgmy_simulate_series(
    params: CGMYParams, n_steps: int, dt: float = 1.0 / 252,
    n_paths: int = 1, n_terms: int = 1000, seed: int = 42,
) -> FloatArray:
    """CGMY simulation via shot-noise / series representation (approximate)."""
    rng = np.random.default_rng(seed)
    C, G, M, Y = params.C, params.G, params.M, params.Y
    paths = np.zeros((n_steps + 1, n_paths))
    for p in range(n_paths):
        for t in range(n_steps):
            # Approximate by compound Poisson with truncation
            lam_pos = C * M ** (Y - 1) * special.gamma(1 - Y) if Y < 1 else C
            lam_neg = C * G ** (Y - 1) * special.gamma(1 - Y) if Y < 1 else C
            n_pos = rng.poisson(abs(lam_pos) * dt)
            n_neg = rng.poisson(abs(lam_neg) * dt)
            jumps_pos = rng.exponential(1.0 / M, size=max(n_pos, 1))[:n_pos].sum() if n_pos > 0 else 0.0
            jumps_neg = rng.exponential(1.0 / G, size=max(n_neg, 1))[:n_neg].sum() if n_neg > 0 else 0.0
            paths[t + 1, p] = paths[t, p] + jumps_pos - jumps_neg
    return paths


# ===================================================================
# 9.  Levy Process Toolkit
# ===================================================================

def carr_madan_fft_call(
    char_fn: Callable[[complex, float], complex],
    S0: float, K: float, T: float, r: float,
    N: int = 4096, eta: float = 0.25, alpha_cm: float = 1.5,
) -> float:
    """Carr-Madan FFT method for European call pricing.
    char_fn(u, T) -> characteristic function of log(S_T)."""
    lam = 2 * np.pi / (N * eta)
    b = N * lam / 2
    u_vals = np.arange(N) * eta
    k_vals = -b + lam * np.arange(N)

    # Modified characteristic function
    psi_vals = np.zeros(N, dtype=complex)
    for j in range(N):
        u = u_vals[j]
        cf = char_fn(u - (alpha_cm + 1) * 1j, T)
        denom = alpha_cm ** 2 + alpha_cm - u ** 2 + 1j * (2 * alpha_cm + 1) * u
        if abs(denom) < 1e-30:
            psi_vals[j] = 0
        else:
            psi_vals[j] = np.exp(-r * T) * cf / denom

    # Simpson weights
    simpson = 3.0 + (-1.0) ** (np.arange(N) + 1)
    simpson[0] = 1.0
    simpson = simpson / 3.0

    x = np.exp(1j * b * u_vals) * psi_vals * eta * simpson
    fft_result = fft(x)
    call_prices = np.exp(-alpha_cm * k_vals) / np.pi * np.real(fft_result)

    # Interpolate to desired strike
    log_K = np.log(K)
    idx = np.searchsorted(k_vals, log_K)
    idx = np.clip(idx, 1, N - 1)
    w = (log_K - k_vals[idx - 1]) / (k_vals[idx] - k_vals[idx - 1] + 1e-12)
    price = (1 - w) * call_prices[idx - 1] + w * call_prices[idx]
    return float(max(price, 0.0))


def levy_density_via_fft(
    char_fn: Callable[[complex, float], complex],
    T: float, x_min: float = -1.0, x_max: float = 1.0, N: int = 4096,
) -> Tuple[FloatArray, FloatArray]:
    """Recover density from characteristic function via FFT."""
    dx = (x_max - x_min) / N
    du = 2 * np.pi / (N * dx)
    x = x_min + dx * np.arange(N)
    u = du * (np.arange(N) - N // 2)
    cf_vals = np.array([char_fn(ui, T) for ui in u])
    density = np.real(np.fft.fftshift(ifft(np.fft.ifftshift(cf_vals)))) * N * du / (2 * np.pi)
    return x, np.maximum(density, 0)


# ===================================================================
# 10. Hawkes Process
# ===================================================================

@dataclass
class HawkesParams:
    mu: float = 0.5             # baseline intensity
    alpha: float = 0.3          # excitation magnitude
    beta: float = 1.0           # decay rate


def hawkes_simulate_ogata(
    params: HawkesParams, T: float, seed: int = 42,
) -> FloatArray:
    """Simulate Hawkes process via Ogata's thinning algorithm."""
    rng = np.random.default_rng(seed)
    mu, alpha, beta = params.mu, params.alpha, params.beta
    events: List[float] = []
    t = 0.0
    lam_bar = mu + alpha  # upper bound on intensity

    while t < T:
        # Generate candidate
        u = rng.random()
        dt_candidate = -np.log(u + 1e-30) / lam_bar
        t += dt_candidate
        if t >= T:
            break
        # Compute actual intensity
        lam_t = mu
        for s in events:
            lam_t += alpha * np.exp(-beta * (t - s))
        # Accept/reject
        if rng.random() < lam_t / lam_bar:
            events.append(t)
            lam_bar = lam_t + alpha
        else:
            lam_bar = lam_t + alpha * 0.5

    return np.array(events)


def hawkes_intensity(
    events: FloatArray, t: float, params: HawkesParams
) -> float:
    """Compute intensity at time t given event history."""
    mu, alpha, beta = params.mu, params.alpha, params.beta
    lam = mu
    for s in events:
        if s < t:
            lam += alpha * np.exp(-beta * (t - s))
    return float(lam)


def hawkes_mle(events: FloatArray, T: float) -> HawkesParams:
    """Maximum likelihood estimation for univariate Hawkes process."""
    n = len(events)
    if n < 5:
        return HawkesParams()

    def neg_log_lik(params_vec: FloatArray) -> float:
        mu, alpha, beta = params_vec
        if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
            return 1e10
        ll = 0.0
        A = np.zeros(n)
        for i in range(1, n):
            A[i] = np.exp(-beta * (events[i] - events[i - 1])) * (1 + A[i - 1])
        for i in range(n):
            lam_i = mu + alpha * A[i]
            ll += np.log(lam_i + 1e-30)
        # Integral of intensity
        integral = mu * T
        for i in range(n):
            integral += alpha / beta * (1 - np.exp(-beta * (T - events[i])))
        return -(ll - integral)

    result = minimize(
        neg_log_lik,
        x0=np.array([0.5, 0.3, 1.0]),
        method="Nelder-Mead",
        options={"maxiter": 1000},
    )
    mu, alpha, beta = result.x
    return HawkesParams(mu=abs(mu), alpha=abs(alpha), beta=abs(beta))


def hawkes_branching_ratio(params: HawkesParams) -> float:
    """Branching ratio alpha/beta. < 1 for stationarity."""
    return params.alpha / (params.beta + 1e-12)


def hawkes_expected_intensity(params: HawkesParams) -> float:
    """Stationary expected intensity: mu / (1 - alpha/beta)."""
    br = hawkes_branching_ratio(params)
    if br >= 1:
        return float("inf")
    return params.mu / (1 - br)


# ===================================================================
# 11. Continuous-Time Markov Chain
# ===================================================================

@dataclass
class CTMCParams:
    generator: FloatArray       # (n_states, n_states) Q matrix
    state_names: List[str] = field(default_factory=list)


def ctmc_validate_generator(Q: FloatArray) -> bool:
    """Check if Q is a valid generator: off-diag >= 0, rows sum to 0."""
    n = Q.shape[0]
    for i in range(n):
        if abs(Q[i].sum()) > 1e-8:
            return False
        for j in range(n):
            if i != j and Q[i, j] < -1e-10:
                return False
    return True


def ctmc_stationary_distribution(Q: FloatArray) -> FloatArray:
    """Compute stationary distribution: pi @ Q = 0, sum(pi) = 1."""
    n = Q.shape[0]
    A = Q.T.copy()
    A[-1] = np.ones(n)
    b = np.zeros(n)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pi = np.ones(n) / n
    pi = np.maximum(pi, 0)
    return pi / (pi.sum() + 1e-12)


def ctmc_transition_matrix(Q: FloatArray, t: float) -> FloatArray:
    """P(t) = exp(Q*t) -- transition probability matrix."""
    return linalg.expm(Q * t)


def ctmc_simulate(
    Q: FloatArray, initial_state: int, T: float, seed: int = 42
) -> Tuple[FloatArray, IntArray]:
    """Simulate CTMC path. Returns (times, states)."""
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    times = [0.0]
    states = [initial_state]
    t = 0.0
    state = initial_state
    while t < T:
        rate = -Q[state, state]
        if rate <= 0:
            break
        dt = rng.exponential(1.0 / rate)
        t += dt
        if t >= T:
            break
        # Choose next state
        probs = Q[state].copy()
        probs[state] = 0
        probs = probs / (probs.sum() + 1e-12)
        state = int(rng.choice(n, p=probs))
        times.append(t)
        states.append(state)
    return np.array(times), np.array(states, dtype=np.int64)


def ctmc_expected_holding_time(Q: FloatArray, state: int) -> float:
    """Expected time spent in state before transitioning."""
    return float(1.0 / (-Q[state, state] + 1e-12))


def ctmc_absorption_probability(
    Q: FloatArray, transient_states: List[int], absorbing_states: List[int]
) -> FloatArray:
    """Absorption probabilities from transient to absorbing states."""
    n_t = len(transient_states)
    n_a = len(absorbing_states)
    Q_tt = Q[np.ix_(transient_states, transient_states)]
    Q_ta = Q[np.ix_(transient_states, absorbing_states)]
    try:
        N = np.linalg.inv(-Q_tt)  # fundamental matrix
        return N @ Q_ta
    except np.linalg.LinAlgError:
        return np.ones((n_t, n_a)) / n_a


# ===================================================================
# 12. Subordinators
# ===================================================================

def inverse_gaussian_subordinator(
    n_steps: int, mu_ig: float = 1.0, lambda_ig: float = 1.0,
    dt: float = 1.0 / 252, n_paths: int = 1, seed: int = 42,
) -> FloatArray:
    """Simulate Inverse Gaussian subordinator (increasing Levy process)."""
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_steps + 1, n_paths))
    for t in range(n_steps):
        # IG increments
        mu_dt = mu_ig * dt
        lam_dt = lambda_ig * dt ** 2
        Y = rng.standard_normal(n_paths) ** 2
        X = mu_dt + mu_dt ** 2 * Y / (2 * lam_dt) - mu_dt / (2 * lam_dt) * np.sqrt(
            4 * mu_dt * lam_dt * Y + mu_dt ** 2 * Y ** 2
        )
        U = rng.random(n_paths)
        dIG = np.where(U <= mu_dt / (mu_dt + X + 1e-12), X, mu_dt ** 2 / (X + 1e-12))
        paths[t + 1] = paths[t] + np.maximum(dIG, 0)
    return paths


def gamma_subordinator(
    n_steps: int, shape: float = 1.0, rate: float = 1.0,
    dt: float = 1.0 / 252, n_paths: int = 1, seed: int = 42,
) -> FloatArray:
    """Simulate Gamma subordinator."""
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_steps + 1, n_paths))
    for t in range(n_steps):
        dG = rng.gamma(shape=shape * dt, scale=1.0 / rate, size=n_paths)
        paths[t + 1] = paths[t] + dG
    return paths


def tempered_stable_subordinator(
    n_steps: int, alpha_ts: float = 0.5, beta_ts: float = 1.0,
    dt: float = 1.0 / 252, n_paths: int = 1, seed: int = 42,
) -> FloatArray:
    """Approximate tempered stable subordinator via compound Poisson."""
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_steps + 1, n_paths))
    eps = 0.01  # truncation
    # Poisson rate for jumps > eps
    lam = alpha_ts / special.gamma(1 - alpha_ts + 1e-12) * eps ** (-alpha_ts) if alpha_ts < 1 else 1.0
    for t in range(n_steps):
        n_jumps = rng.poisson(lam * dt, size=n_paths)
        for p in range(n_paths):
            if n_jumps[p] > 0:
                # Jump sizes: Pareto-like with exponential tempering
                U = rng.random(n_jumps[p])
                jumps = eps * U ** (-1.0 / alpha_ts) * np.exp(-beta_ts * eps * U ** (-1.0 / alpha_ts))
                paths[t + 1, p] = paths[t, p] + jumps.sum()
            else:
                paths[t + 1, p] = paths[t, p]
    return paths


# ===================================================================
# 13. Self-Exciting Processes
# ===================================================================

@dataclass
class NonlinearHawkesParams:
    mu: float = 0.5
    alpha: float = 0.3
    beta: float = 1.0
    power: float = 1.0          # power-law kernel exponent
    cutoff: float = 100.0       # kernel cutoff time


def nonlinear_hawkes_simulate(
    params: NonlinearHawkesParams, T: float, seed: int = 42,
) -> FloatArray:
    """Simulate nonlinear Hawkes with power-law kernel:
    phi(t) = alpha * (1 + t)^{-power} for t < cutoff."""
    rng = np.random.default_rng(seed)
    mu, alpha, power, cutoff = params.mu, params.alpha, params.power, params.cutoff
    events: List[float] = []
    t = 0.0
    lam_bar = mu + alpha * 10  # generous upper bound

    while t < T:
        u = rng.random()
        dt_candidate = -np.log(u + 1e-30) / lam_bar
        t += dt_candidate
        if t >= T:
            break
        # Compute actual intensity with power-law kernel
        lam_t = mu
        for s in events:
            age = t - s
            if age < cutoff:
                lam_t += alpha * (1 + age) ** (-power)
        if rng.random() < lam_t / (lam_bar + 1e-12):
            events.append(t)
            lam_bar = max(lam_t + alpha, lam_bar)
        else:
            lam_bar = max(lam_t + alpha * 0.5, mu)
    return np.array(events)


def mutually_exciting_hawkes_simulate(
    mu: FloatArray,
    alpha_matrix: FloatArray,
    beta: float,
    T: float,
    seed: int = 42,
) -> List[FloatArray]:
    """Simulate multivariate (mutually exciting) Hawkes process.

    Parameters
    ----------
    mu           : (n_dim,) baseline intensities
    alpha_matrix : (n_dim, n_dim) excitation matrix
    beta         : common decay rate
    """
    rng = np.random.default_rng(seed)
    n_dim = len(mu)
    events: List[List[float]] = [[] for _ in range(n_dim)]
    all_events: List[Tuple[float, int]] = []
    t = 0.0
    lam_bar = mu.sum() + alpha_matrix.sum()

    while t < T:
        u = rng.random()
        dt_cand = -np.log(u + 1e-30) / lam_bar
        t += dt_cand
        if t >= T:
            break
        intensities = mu.copy()
        for s, dim in all_events:
            age = t - s
            intensities += alpha_matrix[:, dim] * np.exp(-beta * age)
        total_lam = intensities.sum()
        if rng.random() < total_lam / (lam_bar + 1e-12):
            probs = intensities / (total_lam + 1e-12)
            dim = int(rng.choice(n_dim, p=probs))
            events[dim].append(t)
            all_events.append((t, dim))
            lam_bar = total_lam + alpha_matrix.sum()
        else:
            lam_bar = max(total_lam + 1, mu.sum())
    return [np.array(e) for e in events]


# ===================================================================
# 14. Process comparison / calibration
# ===================================================================

def fit_process_to_returns(
    returns: FloatArray,
    process_type: str = "ou",
    dt: float = 1.0 / 252,
) -> Dict[str, Any]:
    """Fit a stochastic process to empirical return data."""
    if process_type == "ou":
        cumulative = np.cumsum(returns)
        params = ou_mle(cumulative, dt)
        return {"type": "OU", "kappa": params.kappa, "mu": params.mu, "sigma": params.sigma,
                "half_life": ou_mean_reversion_time(params)}
    elif process_type == "gbm":
        mu_est = returns.mean() / dt
        sigma_est = returns.std() / np.sqrt(dt)
        return {"type": "GBM", "mu": float(mu_est), "sigma": float(sigma_est)}
    elif process_type == "vg":
        mu_r = float(returns.mean())
        var_r = float(returns.var())
        skew_r = float(stats.skew(returns))
        kurt_r = float(stats.kurtosis(returns))
        sigma_est = np.sqrt(var_r / dt)
        nu_est = max(kurt_r * dt / 3.0, 0.01)
        theta_est = skew_r * var_r / (3 * nu_est * dt + 1e-12)
        return {"type": "VG", "theta": theta_est, "sigma": sigma_est, "nu": nu_est}
    elif process_type == "nig":
        mu_r = float(returns.mean())
        var_r = float(returns.var())
        skew_r = float(stats.skew(returns))
        kurt_r = float(stats.kurtosis(returns))
        delta_est = np.sqrt(var_r * dt)
        alpha_est = max(np.sqrt(1 / (var_r * dt + 1e-12)), 1.0)
        beta_est = skew_r * alpha_est / 3.0
        return {"type": "NIG", "alpha": alpha_est, "beta": beta_est,
                "delta": delta_est, "mu": mu_r}
    return {"type": "unknown"}


def compare_process_fits(
    returns: FloatArray, dt: float = 1.0 / 252
) -> Dict[str, Dict[str, Any]]:
    """Fit multiple processes and compare."""
    results = {}
    for ptype in ["gbm", "ou", "vg", "nig"]:
        try:
            results[ptype] = fit_process_to_returns(returns, ptype, dt)
        except Exception:
            results[ptype] = {"type": ptype, "error": True}
    return results


# ===================================================================
# __all__
# ===================================================================

__all__ = [
    "brownian_motion",
    "geometric_brownian_motion",
    "fractional_brownian_motion",
    "correlated_brownian_motions",
    "OUParams",
    "ou_simulate",
    "ou_mle",
    "ou_mean_reversion_time",
    "ou_stationary_distribution",
    "CIRParams",
    "cir_simulate_exact",
    "cir_simulate_euler",
    "cir_bond_price",
    "HestonParams",
    "heston_simulate_qe",
    "heston_characteristic_function",
    "SABRParams",
    "sabr_simulate",
    "sabr_implied_vol",
    "VarianceGammaParams",
    "variance_gamma_simulate",
    "variance_gamma_characteristic",
    "NIGParams",
    "nig_simulate",
    "nig_characteristic",
    "CGMYParams",
    "cgmy_characteristic",
    "cgmy_simulate_series",
    "carr_madan_fft_call",
    "levy_density_via_fft",
    "HawkesParams",
    "hawkes_simulate_ogata",
    "hawkes_intensity",
    "hawkes_mle",
    "hawkes_branching_ratio",
    "hawkes_expected_intensity",
    "CTMCParams",
    "ctmc_validate_generator",
    "ctmc_stationary_distribution",
    "ctmc_transition_matrix",
    "ctmc_simulate",
    "ctmc_expected_holding_time",
    "ctmc_absorption_probability",
    "inverse_gaussian_subordinator",
    "gamma_subordinator",
    "tempered_stable_subordinator",
    "NonlinearHawkesParams",
    "nonlinear_hawkes_simulate",
    "mutually_exciting_hawkes_simulate",
    "fit_process_to_returns",
    "compare_process_fits",
]
