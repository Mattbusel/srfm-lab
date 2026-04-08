"""
jump_diffusion.py — Jump-diffusion and regime-switching models.

Models:
  Merton     – GBM + compound Poisson jumps (analytic call price)
  Kou        – double-exponential jump size distribution
  Bates      – Heston stochastic vol + Merton jumps (Carr-Madan FFT pricing)
  Regime     – 2-state Markov-switching GBM with Hamilton filter
  SVJJ       – stochastic vol + simultaneous jumps in price AND variance

Estimation:
  jump_intensity_mom  – method-of-moments from realised cumulants
  jump_intensity_mle  – MLE over Merton parameter vector
"""

from __future__ import annotations

import cmath as _cmath
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import fft, optimize, stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cexp(z: complex) -> complex:  return _cmath.exp(z)
def _csqrt(z: complex) -> complex: return _cmath.sqrt(z)
def _clog(z: complex) -> complex:  return _cmath.log(z)


def _bs_call(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)


# ---------------------------------------------------------------------------
# Merton jump-diffusion
# ---------------------------------------------------------------------------

def simulate_merton(
    S0: float,
    mu: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate Merton (1976) jump-diffusion paths.

    dS/S = (mu - lambda_j * kappa) dt + sigma dW + (e^J - 1) dN
    where J ~ N(mu_j, sigma_j^2), N ~ Poisson(lambda_j).

    Returns array of shape (n_paths, n_steps + 1).
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    kappa = math.exp(mu_j + 0.5 * sigma_j ** 2) - 1
    drift = (mu - 0.5 * sigma ** 2 - lambda_j * kappa) * dt

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for t in range(n_steps):
        Z = rng.standard_normal(n_paths)
        N_counts = rng.poisson(lambda_j * dt, size=n_paths)
        J_sum = np.zeros(n_paths)
        for idx in np.where(N_counts > 0)[0]:
            jumps = rng.normal(mu_j, sigma_j, size=int(N_counts[idx]))
            J_sum[idx] = jumps.sum()
        log_ret = drift + sigma * math.sqrt(dt) * Z + J_sum
        paths[:, t + 1] = paths[:, t] * np.exp(log_ret)

    return paths


def merton_call_price(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    n_terms: int = 50,
) -> float:
    """
    Merton (1976) closed-form call via Poisson-weighted Black-Scholes series.

    C = sum_{n=0}^inf  w_n * BS(S, K, r_n, T, sigma_n)
    where  lambda' = lambda * exp(mu_j + 0.5*sigma_j^2),
           w_n     = exp(-lambda'*T) * (lambda'*T)^n / n!,
           r_n     = r - lambda*(exp(mu_j+0.5*sigma_j^2)-1) + n*(mu_j+0.5*sigma_j^2)/T,
           sigma_n = sqrt(sigma^2 + n*sigma_j^2 / T).
    """
    kappa = math.exp(mu_j + 0.5 * sigma_j ** 2) - 1
    lambda_prime = lambda_j * (1 + kappa)
    total = 0.0
    log_poisson_term = -lambda_prime * T   # log(w_0)

    for n in range(n_terms):
        if n > 0:
            log_poisson_term += math.log(lambda_prime * T) - math.log(n)
        weight = math.exp(log_poisson_term)
        if weight < 1e-15:
            break
        sigma_n = math.sqrt(sigma ** 2 + n * sigma_j ** 2 / T) if T > 0 else sigma
        r_n = (r - lambda_j * kappa
               + n * (mu_j + 0.5 * sigma_j ** 2) / T) if T > 0 else r
        total += weight * _bs_call(S, K, r_n, T, sigma_n)

    return total


# ---------------------------------------------------------------------------
# Kou double-exponential jump model
# ---------------------------------------------------------------------------

@dataclass
class KouParams:
    """Parameters for the Kou (2002) double-exponential jump model."""
    mu: float = 0.05           # drift
    sigma: float = 0.20        # diffusion volatility
    lambda_j: float = 1.0      # Poisson jump intensity
    p: float = 0.60            # probability of upward jump
    eta1: float = 10.0         # rate of upward jump size ~ Exp(eta1)
    eta2: float = 5.0          # rate of downward jump size ~ Exp(eta2)


def simulate_kou(
    S0: float,
    params: KouParams,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate Kou double-exponential jump-diffusion paths."""
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    p = params
    # mean of jump: kappa = p/eta1 - (1-p)/eta2
    kappa_kou = p.p / p.eta1 - (1 - p.p) / p.eta2
    drift_adj = (p.mu - 0.5 * p.sigma ** 2 - p.lambda_j * kappa_kou) * dt

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for t in range(n_steps):
        Z = rng.standard_normal(n_paths)
        N_counts = rng.poisson(p.lambda_j * dt, size=n_paths)
        J_sum = np.zeros(n_paths)
        for idx in np.where(N_counts > 0)[0]:
            k = int(N_counts[idx])
            up_mask = rng.random(k) < p.p
            j_up = rng.exponential(1.0 / p.eta1, size=k)
            j_dn = rng.exponential(1.0 / p.eta2, size=k)
            J_sum[idx] = np.where(up_mask, j_up, -j_dn).sum()
        log_ret = drift_adj + p.sigma * math.sqrt(dt) * Z + J_sum
        paths[:, t + 1] = paths[:, t] * np.exp(log_ret)

    return paths


def kou_call_price(
    S: float,
    K: float,
    r: float,
    T: float,
    params: KouParams,
    n_terms: int = 40,
) -> float:
    """
    Approximate Kou call price using moment-matched normal approximation
    for jump sizes (Merton series with matched first two moments).
    """
    p = params
    mu_j_approx = p.p / p.eta1 - (1 - p.p) / p.eta2
    var_j = (p.p * (2.0 / p.eta1 ** 2) + (1 - p.p) * (2.0 / p.eta2 ** 2)
             + p.p * (1 - p.p) * (1 / p.eta1 + 1 / p.eta2) ** 2)
    sigma_j_approx = math.sqrt(max(var_j, 1e-8))
    return merton_call_price(S, K, r, T, p.sigma, p.lambda_j,
                             mu_j_approx, sigma_j_approx, n_terms)


# ---------------------------------------------------------------------------
# Bates model: Heston + Merton jumps — characteristic function + FFT pricing
# ---------------------------------------------------------------------------

def bates_characteristic_function(
    u: complex,
    S: float,
    r: float,
    q: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
) -> complex:
    """
    Bates (1996) characteristic function for log(S_T).
    Combines the Heston SV model with Merton log-normal jumps.
    """
    i = complex(0.0, 1.0)
    x = math.log(S) + (r - q) * T
    mu_jump = math.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0

    # Heston terms
    d = _csqrt((kappa - rho * sigma_v * i * u) ** 2
               + sigma_v ** 2 * (i * u + u ** 2))
    g_num = kappa - rho * sigma_v * i * u - d
    g_den = kappa - rho * sigma_v * i * u + d
    g = g_num / g_den
    exp_dT = _cexp(-d * T)
    C = ((r - q) * i * u * T
         + kappa * theta / sigma_v ** 2
         * (g_num * T - 2.0 * _clog((1.0 - g * exp_dT) / (1.0 - g))))
    D = g_num / sigma_v ** 2 * (1.0 - exp_dT) / (1.0 - g * exp_dT)

    # Jump component
    jump_cf = lambda_j * T * (_cexp(i * u * mu_j - 0.5 * sigma_j ** 2 * u ** 2)
                               - 1.0 - i * u * mu_jump)

    return _cexp(i * u * x + C + D * v0 + jump_cf)


def bates_call_fft(
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    N: int = 4096,
    alpha: float = 1.5,
    eta: float = 0.25,
) -> float:
    """
    Carr-Madan (1999) FFT call option pricing for the Bates model.
    alpha is the dampening parameter (must satisfy alpha > 0, alpha+1 < limit).
    """
    lam = 2.0 * math.pi / (N * eta)
    b = N * lam / 2.0
    ku = np.linspace(-b, b - lam, N)
    nu = eta * np.arange(N)

    psi = np.empty(N, dtype=complex)
    for j in range(N):
        v = nu[j]
        u = v - (alpha + 1.0) * 1j
        cf = bates_characteristic_function(
            u, S, r, q, T, v0, kappa, theta, sigma_v, rho,
            lambda_j, mu_j, sigma_j
        )
        denom = alpha ** 2 + alpha - v ** 2 + 1j * (2.0 * alpha + 1.0) * v
        psi[j] = math.exp(-r * T) * cf / denom

    weights = eta / 3.0 * (3.0 + (-1.0) ** np.arange(N) - (np.arange(N) == 0).astype(float))
    x = weights * np.exp(1j * b * nu) * psi
    prices_arr = np.exp(-alpha * ku) / math.pi * fft.fft(x).real
    k_target = math.log(K)
    price = float(np.interp(k_target, ku, prices_arr))
    return max(price, 0.0)


# ---------------------------------------------------------------------------
# Regime-switching GBM (2-state Markov chain)
# ---------------------------------------------------------------------------

@dataclass
class RegimeSwitchingParams:
    """Parameters for 2-state regime-switching GBM."""
    mu: Tuple[float, float] = (0.10, -0.05)
    sigma: Tuple[float, float] = (0.15, 0.35)
    P: np.ndarray = field(
        default_factory=lambda: np.array([[0.98, 0.02],
                                          [0.05, 0.95]])
    )


def simulate_regime_switching(
    S0: float,
    params: RegimeSwitchingParams,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate 2-state Markov-switching GBM.
    Returns (paths, regimes), both shape (n_paths, n_steps+1).
    Regime 0 = bull, regime 1 = bear.
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    mu_arr = np.array(params.mu)
    sig_arr = np.array(params.sigma)
    P = params.P
    # Stationary distribution
    pi0 = np.array([P[1, 0], P[0, 1]]) / (P[0, 1] + P[1, 0])

    paths = np.empty((n_paths, n_steps + 1))
    regimes = np.empty((n_paths, n_steps + 1), dtype=int)
    paths[:, 0] = S0
    regimes[:, 0] = rng.choice(2, size=n_paths, p=pi0)

    for t in range(n_steps):
        s = regimes[:, t]
        Z = rng.standard_normal(n_paths)
        mu_t = mu_arr[s]
        sig_t = sig_arr[s]
        log_ret = (mu_t - 0.5 * sig_t ** 2) * dt + sig_t * math.sqrt(dt) * Z
        paths[:, t + 1] = paths[:, t] * np.exp(log_ret)
        u_trans = rng.uniform(size=n_paths)
        stay_p = P[s, s]                      # vectorised diagonal lookup
        stay_p = np.where(s == 0, P[0, 0], P[1, 1])
        regimes[:, t + 1] = np.where(u_trans < stay_p, s, 1 - s)

    return paths, regimes


def filter_regimes(
    returns: np.ndarray,
    params: RegimeSwitchingParams,
    dt: float = 1.0 / 252,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hamilton (1989) filter + Kim (1994) smoother for hidden regime probabilities.

    Parameters
    ----------
    returns : 1-D array of log-returns
    params  : RegimeSwitchingParams

    Returns
    -------
    filtered : (T, 2)  P(s_t = k | r_1..r_t)
    smoothed : (T, 2)  backward-smoothed probabilities
    """
    T = len(returns)
    mu_arr = np.array(params.mu)
    sig_arr = np.array(params.sigma)
    P = params.P
    pi0 = np.array([P[1, 0], P[0, 1]]) / (P[0, 1] + P[1, 0])

    def emission(r: float) -> np.ndarray:
        loc = (mu_arr - 0.5 * sig_arr ** 2) * dt
        scale = sig_arr * math.sqrt(dt)
        return stats.norm.pdf(r, loc=loc, scale=scale)

    # Forward (filter)
    xi = np.zeros((T, 2))
    xi[0] = pi0 * emission(returns[0])
    xi[0] /= xi[0].sum() + 1e-300

    for t in range(1, T):
        pred = P.T @ xi[t - 1]
        xi[t] = pred * emission(returns[t])
        s = xi[t].sum()
        xi[t] /= s + 1e-300

    # Backward (Kim smoother)
    smooth = xi.copy()
    for t in range(T - 2, -1, -1):
        pred_next = P.T @ xi[t]
        ratio = np.where(pred_next > 1e-300, smooth[t + 1] / (pred_next + 1e-300), 0.0)
        smooth[t] = xi[t] * (P @ ratio)
        s = smooth[t].sum()
        smooth[t] /= s + 1e-300

    return xi, smooth


def regime_switching_forecast(
    params: RegimeSwitchingParams,
    current_regime_prob: np.ndarray,
    h: int = 5,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    h-step-ahead regime probability forecasts and conditional mean/vol.

    Returns (final_regime_prob, (mean_path, vol_path)) each length h.
    """
    mu_arr = np.array(params.mu)
    sig_arr = np.array(params.sigma)
    prob = current_regime_prob.copy().astype(float)
    means = np.empty(h)
    vols = np.empty(h)

    for step in range(h):
        prob = params.P.T @ prob
        m = float(prob @ mu_arr)
        v2 = float(prob @ (sig_arr ** 2 + mu_arr ** 2)) - m ** 2
        means[step] = m
        vols[step] = math.sqrt(max(v2, 0.0))

    return prob, (means, vols)


# ---------------------------------------------------------------------------
# SVJJ — stochastic vol + simultaneous jumps in price and variance
# ---------------------------------------------------------------------------

def svjj_characteristic_function(
    u: complex,
    S: float,
    r: float,
    q: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    lambda_j: float,
    mu_jS: float,
    sigma_jS: float,
    mu_jV: float,
) -> complex:
    """
    SVJJ characteristic function based on Duffie-Pan-Singleton (2000).
    Price jump  J_S ~ N(mu_jS, sigma_jS^2)
    Variance jump J_V ~ Exp(1/mu_jV)  (mean mu_jV, independent of J_S)
    """
    i = complex(0.0, 1.0)
    x = math.log(S) + (r - q) * T

    # Heston component
    d = _csqrt((kappa - rho * sigma_v * i * u) ** 2
               + sigma_v ** 2 * (i * u + u ** 2))
    g_num = kappa - rho * sigma_v * i * u - d
    g_den = kappa - rho * sigma_v * i * u + d
    g = g_num / g_den
    exp_dT = _cexp(-d * T)
    B = g_num / sigma_v ** 2 * (1.0 - exp_dT) / (1.0 - g * exp_dT)
    A = ((r - q) * i * u * T
         + kappa * theta / sigma_v ** 2
         * (g_num * T - 2.0 * _clog((1.0 - g * exp_dT) / (1.0 - g))))

    # Jump component (uncorrelated price & variance jumps)
    cf_S = _cexp(i * u * mu_jS - 0.5 * sigma_jS ** 2 * u ** 2)   # CF of J_S
    denom_V = 1.0 - mu_jV * B
    cf_V = 1.0 / denom_V if abs(denom_V) > 1e-12 else complex(1.0, 0.0)
    mean_jS = math.exp(mu_jS + 0.5 * sigma_jS ** 2) - 1.0
    compensator = -lambda_j * T * i * u * mean_jS
    jump_term = lambda_j * T * (cf_S * cf_V - 1.0) + compensator

    return _cexp(i * u * x + A + B * v0 + jump_term)


# ---------------------------------------------------------------------------
# Jump intensity estimation from returns
# ---------------------------------------------------------------------------

def jump_intensity_mom(
    returns: np.ndarray,
    dt: float = 1.0 / 252,
) -> Tuple[float, float, float]:
    """
    Method-of-moments estimator for Merton jump parameters using
    excess kurtosis and skewness of the return series.

    Returns (lambda_j, mu_j, sigma_j).
    """
    k2 = float(stats.kstat(returns, 2))
    k3 = float(stats.kstat(returns, 3))
    k4 = float(stats.kstat(returns, 4))

    if abs(k4) < 1e-14 or k4 <= 0:
        sigma_total = math.sqrt(max(k2 / dt, 1e-8))
        return 0.0, 0.0, sigma_total

    # Moment equations: k4 = lambda*dt*(3*sj^4 + 6*sj^2*mj^2 + mj^4)
    # k3 = lambda*dt*(mj^3 + 3*mj*sj^2)
    # Estimate sj from k4/k3 ratio
    if abs(k3) > 1e-14:
        mu_j = k3 / k4 * (k4 / max(k2, 1e-12))     # rough
    else:
        mu_j = 0.0
    sigma_j_sq = max(k4 / (3.0 * max(k2, 1e-12)) - mu_j ** 2, 1e-8)
    sigma_j = math.sqrt(sigma_j_sq)
    central_moment_4 = 3.0 * sigma_j ** 4 + 6.0 * sigma_j ** 2 * mu_j ** 2 + mu_j ** 4
    lambda_j = max(k4 / (dt * max(central_moment_4, 1e-12)), 0.0)

    return float(lambda_j), float(mu_j), float(sigma_j)


def jump_intensity_mle(
    returns: np.ndarray,
    dt: float = 1.0 / 252,
    n_terms: int = 20,
) -> Tuple[float, float, float, float, float]:
    """
    Full MLE for Merton jump-diffusion parameters.

    Maximises the Poisson-mixture log-likelihood:
      log L = sum_t log( sum_n P(N=n) * phi(r_t; mu_n, sigma_n^2) )

    Returns (mu_drift, sigma, lambda_j, mu_j, sigma_j).
    """

    def neg_ll(params: np.ndarray) -> float:
        mu_, sig, lam, mj, sj = params
        if sig <= 0.0 or lam < 0.0 or sj <= 0.0:
            return 1e10
        kappa = math.exp(mj + 0.5 * sj ** 2) - 1.0
        drift_adj = (mu_ - 0.5 * sig ** 2 - lam * kappa) * dt
        lam_dt = lam * dt
        ll = 0.0
        for r in returns:
            s = 0.0
            log_pn = -lam_dt
            for n in range(n_terms):
                if n > 0:
                    log_pn += math.log(lam_dt) - math.log(n)
                mn = drift_adj + n * mj
                vn = sig ** 2 * dt + n * sj ** 2
                log_gauss = -0.5 * math.log(2.0 * math.pi * vn) - (r - mn) ** 2 / (2.0 * vn)
                s += math.exp(log_pn + log_gauss)
                if lam_dt < 1.0 and n >= 10:
                    break
            ll += math.log(max(s, 1e-300))
        return -ll

    lj0, mj0, sj0 = jump_intensity_mom(returns, dt)
    sigma0 = max(float(np.std(returns)) / math.sqrt(dt), 0.01)
    mu0 = float(np.mean(returns)) / dt
    x0 = [mu0, sigma0, max(lj0, 0.1), mj0, max(sj0, 0.01)]
    bounds = [(None, None), (1e-4, None), (0.0, None), (None, None), (1e-4, None)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = optimize.minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                                options={"maxiter": 400, "ftol": 1e-10})

    return tuple(float(v) for v in res.x)  # type: ignore[return-value]
