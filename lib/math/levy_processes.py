"""
Lévy processes for financial modeling.

Implements:
  - CGMY (Carr-Geman-Madan-Yor) process
  - Stable distributions (alpha-stable Lévy)
  - Variance Gamma via subordination
  - Normal Inverse Gaussian (NIG) process
  - Lévy-Khintchine representation
  - Subordinators (Gamma, Inverse Gaussian, Tempered Stable)
  - Characteristic function toolkit
  - FFT option pricing via characteristic functions
  - Lévy copulas (tail integral)
  - Self-decomposability tests
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


# ── Characteristic function toolkit ──────────────────────────────────────────

def cf_gbm(u: np.ndarray, mu: float, sigma: float, t: float) -> np.ndarray:
    """Characteristic function of GBM log-return."""
    return np.exp(1j * u * mu * t - 0.5 * sigma**2 * u**2 * t)


def cf_merton(
    u: np.ndarray,
    mu: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    t: float,
) -> np.ndarray:
    """CF of Merton jump-diffusion log-return."""
    jump_cf = np.exp(1j * u * mu_j - 0.5 * sigma_j**2 * u**2) - 1
    return np.exp(
        1j * u * mu * t
        - 0.5 * sigma**2 * u**2 * t
        + lam * t * jump_cf
    )


def cf_vg(
    u: np.ndarray,
    mu: float,
    sigma: float,
    theta: float,
    nu: float,
    t: float,
) -> np.ndarray:
    """
    Characteristic function of Variance Gamma log-return.
    VG(sigma, nu, theta): VG CF = exp(i*mu*u*t) * (1 - i*theta*nu*u + 0.5*sigma^2*nu*u^2)^{-t/nu}
    """
    omega = (1 / nu) * math.log(1 - theta * nu - 0.5 * sigma**2 * nu)  # drift correction
    inner = 1 - 1j * theta * nu * u + 0.5 * sigma**2 * nu * u**2
    inner = np.where(np.abs(inner) < 1e-10, 1e-10, inner)
    return np.exp(1j * u * (mu + omega) * t) * inner ** (-t / nu)


def cf_nig(
    u: np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
    delta: float,
    t: float,
) -> np.ndarray:
    """
    Characteristic function of NIG log-return.
    NIG(alpha, beta, delta, mu): kurtosis > 3, skew via beta.
    """
    gamma = math.sqrt(alpha**2 - beta**2)
    psi = delta * (gamma - np.sqrt(alpha**2 - (beta + 1j * u)**2))
    return np.exp(1j * u * mu * t + psi * t)


def cf_cgmy(
    u: np.ndarray,
    C: float,
    G: float,
    M: float,
    Y: float,
    t: float,
) -> np.ndarray:
    """
    Characteristic function of CGMY process.
    C: overall activity, G: negative jump decay, M: positive jump decay, Y: fine structure.
    Y < 2 required. Y=0: compound Poisson; Y in (0,1): finite activity; Y in [1,2): infinite activity.
    """
    gamma_term = (
        C * math.gamma(-Y)
        * ((M - 1j * u)**Y - M**Y + (G + 1j * u)**Y - G**Y)
    )
    return np.exp(t * gamma_term)


# ── FFT option pricing (Carr-Madan) ──────────────────────────────────────────

def carr_madan_call(
    cf: Callable,        # characteristic function of log-return: cf(u, t)
    S: float,            # spot price
    K_grid: np.ndarray,  # strike grid
    r: float,            # risk-free rate
    t: float,            # time to expiry
    alpha: float = 1.5,  # damping factor
    N: int = 4096,
    eta: float = 0.25,
) -> np.ndarray:
    """
    Carr-Madan FFT option pricing formula.
    Works for any model given its characteristic function.
    Returns call prices for each strike in K_grid.
    """
    lam = 2 * math.pi / (N * eta)
    b = N * lam / 2.0

    # Grid in u-space
    j = np.arange(N)
    u_j = j * eta

    # Modified CF for the damped call
    k_u = np.exp(-r * t) * cf(u_j - (alpha + 1) * 1j, t) / (
        alpha**2 + alpha - u_j**2 + 1j * (2 * alpha + 1) * u_j
    )

    # Simpson weights
    w = eta / 3.0 * (3 + (-1)**(j + 1))
    w[0] = eta / 3.0

    # FFT
    x = np.exp(1j * b * u_j) * k_u * w
    y = np.fft.fft(x).real

    # Log-strike grid
    log_k_grid = -b + lam * j
    calls_fft = (np.exp(-alpha * log_k_grid) / math.pi) * y

    # Interpolate to desired strikes
    log_K = np.log(K_grid / S)
    calls = np.interp(log_K, log_k_grid, calls_fft)
    return np.maximum(calls * S, 0.0)


# ── CGMY process ─────────────────────────────────────────────────────────────

@dataclass
class CGMYParams:
    C: float = 1.0      # activity level
    G: float = 5.0      # negative jump decay
    M: float = 10.0     # positive jump decay
    Y: float = 0.5      # fine structure (0 < Y < 2)
    mu: float = 0.0     # drift

    @property
    def variance(self) -> float:
        """CGMY variance per unit time."""
        return self.C * math.gamma(2 - self.Y) * (self.M**(self.Y - 2) + self.G**(self.Y - 2))

    @property
    def skewness(self) -> float:
        num = self.C * math.gamma(3 - self.Y) * (self.M**(self.Y - 3) - self.G**(self.Y - 3))
        return num / (self.variance ** 1.5)


def cgmy_simulate(
    params: CGMYParams,
    T: float = 1.0,
    n_steps: int = 252,
    n_paths: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate CGMY paths via series representation (small jump approximation).
    Uses subordinated Brownian motion for |Y| < 1 fine structure.
    Returns (n_paths, n_steps+1) array.
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))

    # Approximate via VG-like increments when Y < 1
    # Use Gaussian approximation for small jumps + Poisson for large
    sigma_approx = math.sqrt(params.variance * dt)

    for i in range(1, n_steps + 1):
        # Drift + diffusion approximation
        z = rng.standard_normal(n_paths)
        increments = params.mu * dt + sigma_approx * z

        # Add large jump correction via Poisson
        jump_rate = params.C * (
            math.exp(-params.M) * params.M**(params.Y - 1)
            + math.exp(-params.G) * params.G**(params.Y - 1)
        ) * dt
        n_jumps = rng.poisson(max(jump_rate, 0.01), n_paths)
        for k in range(n_paths):
            if n_jumps[k] > 0:
                # Sample positive and negative jumps from tempered stable
                pos = rng.exponential(1 / params.M, n_jumps[k]).sum()
                neg = -rng.exponential(1 / params.G, n_jumps[k]).sum()
                increments[k] += (pos + neg) * 0.1  # scaled

        paths[:, i] = paths[:, i - 1] + increments

    return paths


# ── Alpha-stable distributions ────────────────────────────────────────────────

@dataclass
class StableParams:
    alpha: float = 1.7   # stability index (0 < alpha <= 2; alpha=2: Gaussian)
    beta: float = 0.0    # skewness (-1 to 1)
    scale: float = 1.0   # scale (sigma)
    loc: float = 0.0     # location (mu)

    def __post_init__(self):
        assert 0 < self.alpha <= 2, "alpha must be in (0, 2]"
        assert -1 <= self.beta <= 1, "beta must be in [-1, 1]"


def stable_cf(u: np.ndarray, params: StableParams) -> np.ndarray:
    """
    Characteristic function of alpha-stable distribution.
    Uses the Zolotarev (S1) parameterization.
    """
    a, b, c, d = params.alpha, params.beta, params.scale, params.loc
    abs_u = np.abs(u * c)
    sign_u = np.sign(u)

    if abs(a - 1.0) < 1e-6:
        # Cauchy case
        psi = -abs_u + 1j * b * sign_u * (2 / math.pi) * c * np.log(abs_u + 1e-15)
    else:
        phi = math.tan(math.pi * a / 2)
        psi = -(abs_u**a) * (1 - 1j * b * sign_u * phi)

    return np.exp(1j * u * d + psi)


def stable_simulate(
    params: StableParams,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate alpha-stable random variables via Chambers-Mallows-Stuck method.
    """
    rng = rng or np.random.default_rng()
    a, b, c, d = params.alpha, params.beta, params.scale, params.loc

    V = rng.uniform(-math.pi / 2, math.pi / 2, n)
    W = rng.exponential(1.0, n)

    if abs(a - 1.0) < 1e-6:
        # Cauchy (alpha=1)
        zeta = math.pi / 2 * b
        X = (1 + b**2 * (math.pi / 2)**2)**0.5 * np.sin(V + math.atan(b * math.pi / 2)) / np.cos(V)**1.0
        X = (1 / (math.pi / 2)) * (X - b * np.log((math.pi / 2) * W * np.cos(V - math.atan(b * math.pi / 2)) / ((1 + b**2 * (math.pi / 2)**2)**0.5)))
    else:
        zeta = -b * math.tan(math.pi * a / 2)
        xi = (1 / a) * math.atan(-zeta)
        X = (
            (1 + zeta**2)**(1 / (2 * a))
            * np.sin(a * (V + xi))
            / np.cos(V)**(1 / a)
            * (np.cos(V - a * (V + xi)) / W)**((1 - a) / a)
        )

    return c * X + d


def fit_stable(returns: np.ndarray) -> StableParams:
    """
    Fit stable distribution to returns via quantile matching.
    McCulloch (1986) method.
    """
    q = np.percentile(returns, [5, 25, 50, 75, 95])
    q5, q25, q50, q75, q95 = q

    # Alpha via tail ratio
    v_alpha = (q95 - q5) / (q75 - q25 + 1e-10)
    # Approximate alpha lookup
    if v_alpha < 2.44:
        alpha = 2.0
    elif v_alpha < 3.44:
        alpha = 1.5 + (3.44 - v_alpha) / 2.0
    elif v_alpha < 6.0:
        alpha = 1.0 + (3.44 - v_alpha) / 2.56
    else:
        alpha = 0.5

    alpha = float(np.clip(alpha, 0.5, 2.0))

    # Beta via asymmetry
    if abs(q75 - q25) > 1e-10:
        beta = float(np.clip((q95 + q5 - 2 * q50) / (q95 - q5), -1, 1))
    else:
        beta = 0.0

    scale = float((q75 - q25) / 1.3490)  # normal quartile factor
    loc = float(q50)

    return StableParams(alpha=alpha, beta=beta, scale=max(scale, 1e-8), loc=loc)


# ── Subordinators ─────────────────────────────────────────────────────────────

def gamma_subordinator(
    mean: float,
    variance: float,
    T: float,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample Gamma subordinator path G_t with E[G_t] = mean*t, Var[G_t] = variance*t.
    Used to time-change Brownian motion for VG processes.
    Returns increments (n_steps,).
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    shape = (mean**2 / variance) * dt
    scale = variance / mean
    return rng.gamma(shape, scale, n_steps)


def inverse_gaussian_subordinator(
    mu: float,
    lam: float,
    T: float,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Inverse Gaussian subordinator increments.
    Used for NIG processes (IG-subordinated Brownian motion).
    mu: mean of each IG increment, lam: shape.
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    ig_mean = mu * dt
    ig_lam = lam * dt**2

    # Michael-Schucany-Haas Wichura algorithm
    y = rng.standard_normal(n_steps)**2
    x = ig_mean + ig_mean**2 * y / (2 * ig_lam) - ig_mean / (2 * ig_lam) * np.sqrt(
        4 * ig_mean * ig_lam * y + ig_mean**2 * y**2
    )
    u = rng.uniform(0, 1, n_steps)
    mask = u <= ig_mean / (ig_mean + x)
    result = np.where(mask, x, ig_mean**2 / x)
    return np.maximum(result, 1e-10)


def tempered_stable_subordinator(
    alpha: float,
    C: float,
    lam: float,
    T: float,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Tempered stable (TS) subordinator increments.
    Used for CTS/CGMY subordinated processes.
    """
    rng = rng or np.random.default_rng()
    # Approximation via gamma subordinator when alpha close to 0
    mean = C * math.gamma(1 - alpha) * lam**(alpha - 1)
    var = C * math.gamma(2 - alpha) * lam**(alpha - 2)
    return gamma_subordinator(mean, var, T, n_steps, rng)


# ── Lévy-Khintchine representation ───────────────────────────────────────────

def levy_khintchine_exponent(
    u: float,
    drift: float,
    diffusion: float,
    levy_measure_cf: Optional[Callable] = None,
) -> complex:
    """
    Lévy-Khintchine formula:
    ψ(u) = i*drift*u - 0.5*diffusion^2*u^2 + ∫(e^{iux} - 1 - iux*1_{|x|<1}) ν(dx)

    levy_measure_cf: callable(u) returning the jump integral term.
    """
    psi = 1j * drift * u - 0.5 * diffusion**2 * u**2
    if levy_measure_cf is not None:
        psi += levy_measure_cf(u)
    return psi


def estimate_levy_triplet(
    returns: np.ndarray,
    dt: float = 1.0,
) -> dict:
    """
    Estimate Lévy triplet (b, c, ν) from return data via empirical moments.
    b: drift, c: Gaussian component variance, ν: jump measure statistics.
    """
    r = returns / math.sqrt(dt)
    mu = float(r.mean())
    var = float(r.var())

    # Estimate jump component via kurtosis
    kurt = float((((r - mu) / (r.std() + 1e-10))**4).mean())
    excess_kurt = max(kurt - 3, 0)

    # Gaussian component: c = variance - jump variance
    # Under Blumenthal-Getoor: excess kurtosis ∝ jump intensity
    jump_intensity = excess_kurt / 4.0  # rough estimate
    jump_variance = min(jump_intensity * var, var * 0.5)
    gaussian_var = max(var - jump_variance, 0)

    return {
        "drift": float(mu),
        "gaussian_variance": float(gaussian_var),
        "jump_intensity": float(jump_intensity),
        "jump_variance": float(jump_variance),
        "excess_kurtosis": float(excess_kurt),
        "blumenthal_getoor_index": float(min(1.5 + 0.5 / max(excess_kurt, 0.1), 2.0)),
    }


# ── Lévy copulas ──────────────────────────────────────────────────────────────

def clayton_levy_copula(u: float, v: float, theta: float) -> float:
    """
    Clayton Lévy copula for bivariate jump dependence.
    F(u,v) = (u^{-theta} + v^{-theta})^{-1/theta}
    theta > 0: positive dependence of jump times.
    """
    if u <= 0 or v <= 0:
        return 0.0
    return float((u**(-theta) + v**(-theta))**(-1 / theta))


def tail_integral_empirical(
    returns1: np.ndarray,
    returns2: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """
    Empirical bivariate tail integral (Lévy copula proxy).
    Estimates P(jump1 > x, jump2 > y) normalized by marginal intensities.
    """
    # Use negative tail (large losses)
    n = min(len(returns1), len(returns2))
    r1, r2 = -returns1[:n], -returns2[:n]

    tail_probs = np.zeros(len(x_grid))
    for i, x in enumerate(x_grid):
        joint = np.mean((r1 > x) & (r2 > x))
        marginal1 = np.mean(r1 > x)
        marginal2 = np.mean(r2 > x)
        denom = math.sqrt(marginal1 * marginal2 + 1e-10)
        tail_probs[i] = joint / denom
    return tail_probs


# ── Self-decomposability ──────────────────────────────────────────────────────

def check_self_decomposability(returns: np.ndarray, n_lags: int = 10) -> dict:
    """
    Test for self-decomposability (ID distributions that are limits of normalized sums).
    Self-decomposable ↔ log CF is concave. Uses empirical CF test.
    """
    from numpy.fft import fft

    n = len(returns)
    # Empirical characteristic function
    u_grid = np.linspace(-5, 5, 100)

    def ecf(u):
        return np.mean(np.exp(1j * u * returns))

    ecf_vals = np.array([ecf(u) for u in u_grid])
    log_ecf = np.log(np.abs(ecf_vals) + 1e-10)

    # Check concavity of |log ECF| via second differences
    d2 = np.diff(log_ecf, 2)
    concavity_score = float(np.mean(d2 <= 0))  # fraction concave

    # Also check Blumenthal-Getoor index
    triplet = estimate_levy_triplet(returns)

    return {
        "concavity_score": concavity_score,
        "likely_self_decomposable": bool(concavity_score > 0.7),
        "blumenthal_getoor_index": triplet["blumenthal_getoor_index"],
        "excess_kurtosis": triplet["excess_kurtosis"],
        "gaussian_fraction": float(
            triplet["gaussian_variance"] / (triplet["gaussian_variance"] + triplet["jump_variance"] + 1e-10)
        ),
    }
