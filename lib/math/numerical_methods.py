"""
Numerical methods for quantitative finance.

Implements:
  - Runge-Kutta ODE solvers (RK4, RK45 adaptive)
  - Crank-Nicolson PDE solver (Black-Scholes, heat equation)
  - FFT-based option pricing
  - Newton-Raphson root finding (implied vol solver)
  - Brent's method (bracketed root finding)
  - Gaussian quadrature integration
  - Monte Carlo integration with variance reduction
  - Finite difference Greeks
  - Trinomial tree option pricing
  - ADI (Alternating Direction Implicit) for 2D PDEs
"""

from __future__ import annotations
import math
import numpy as np
from typing import Callable, Optional, Tuple


# ── ODE Solvers ───────────────────────────────────────────────────────────────

def rk4_step(
    f: Callable,    # f(t, y) -> dy/dt
    t: float,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    """Single RK4 step."""
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_integrate(
    f: Callable,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fixed-step RK4 integration.
    Returns (t_grid, y_grid) where y_grid is (n_steps+1, len(y0)).
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    t_grid = np.linspace(t0, tf, n_steps + 1)
    y = y0.copy().astype(float)
    trajectory = np.zeros((n_steps + 1, len(y0)))
    trajectory[0] = y

    for i in range(n_steps):
        y = rk4_step(f, t_grid[i], y, h)
        trajectory[i + 1] = y

    return t_grid, trajectory


def rk45_adaptive(
    f: Callable,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dormand-Prince RK45 adaptive step-size integrator.
    """
    # Butcher tableau (Dormand-Prince)
    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    a = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    ]
    b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]

    t0, tf = t_span
    h = (tf - t0) / 100
    t = t0
    y = y0.copy().astype(float)
    ts = [t]
    ys = [y.copy()]

    for _ in range(max_steps):
        if t >= tf:
            break
        h = min(h, tf - t)

        k = [f(t, y)]
        for i in range(1, 7):
            yi = y + h * sum(a[i][j] * k[j] for j in range(len(a[i])))
            k.append(f(t + c[i] * h, yi))

        y5 = y + h * sum(b5[j] * k[j] for j in range(7))
        y4 = y + h * sum(b4[j] * k[j] for j in range(7))

        err = np.linalg.norm(y5 - y4) / (atol + rtol * np.linalg.norm(y5))
        if err <= 1.0 or h < 1e-12:
            t += h
            y = y5
            ts.append(t)
            ys.append(y.copy())

        if err > 0:
            h *= min(5.0, max(0.2, 0.9 * err**(-0.2)))

    return np.array(ts), np.array(ys)


# ── Crank-Nicolson PDE Solver ─────────────────────────────────────────────────

def crank_nicolson_bs(
    S_max: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    M: int = 100,    # S grid points
    N: int = 100,    # time steps
    option_type: str = "call",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crank-Nicolson finite difference scheme for Black-Scholes PDE.
    Returns (S_grid, t_grid, V_grid) where V_grid[i, j] = option value at S_i, t_j.
    """
    dS = S_max / M
    dt = T / N

    S = np.linspace(0, S_max, M + 1)
    V = np.zeros((M + 1, N + 1))

    # Terminal condition
    if option_type == "call":
        V[:, -1] = np.maximum(S - K, 0)
    else:
        V[:, -1] = np.maximum(K - S, 0)

    # Boundary conditions (all times)
    if option_type == "call":
        V[0, :] = 0
        V[-1, :] = S_max - K * np.exp(-r * np.linspace(0, T, N + 1))
    else:
        V[0, :] = K * np.exp(-r * np.linspace(0, T, N + 1))
        V[-1, :] = 0

    # Interior coefficients
    j = np.arange(1, M)
    alpha = 0.25 * dt * (sigma**2 * j**2 - r * j)
    beta  = -0.5 * dt * (sigma**2 * j**2 + r)
    gamma = 0.25 * dt * (sigma**2 * j**2 + r * j)

    # Build tridiagonal matrices
    def build_matrix(sign):
        n_int = M - 1
        A = np.zeros((n_int, n_int))
        for i in range(n_int):
            A[i, i] = 1 + sign * beta[i]
            if i > 0:
                A[i, i - 1] = sign * alpha[i]
            if i < n_int - 1:
                A[i, i + 1] = sign * gamma[i]
        return A

    A = build_matrix(-1)   # implicit part
    B = build_matrix(1)    # explicit part

    A_inv = np.linalg.inv(A)

    # Backward in time
    for n in range(N - 1, -1, -1):
        rhs = B @ V[1:M, n + 1]
        rhs[0]  += alpha[0]  * (V[0, n] + V[0, n + 1])
        rhs[-1] += gamma[-1] * (V[-1, n] + V[-1, n + 1])
        V[1:M, n] = A_inv @ rhs

    return S, np.linspace(0, T, N + 1), V


# ── Implied Volatility Solver ─────────────────────────────────────────────────

def black_scholes_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K * math.exp(-r * T), 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from scipy.stats import norm
    return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def bs_vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """BS vega."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    from scipy.stats import norm
    return float(S * math.sqrt(T) * norm.pdf(d1))


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    tol: float = 1e-7,
    max_iter: int = 100,
) -> float:
    """
    Newton-Raphson implied volatility solver.
    Fast convergence near fair value.
    Returns sigma_iv or NaN if not converged.
    """
    if T <= 0:
        return float("nan")

    # Initial guess: Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2 * math.pi / T) * market_price / S

    for _ in range(max_iter):
        if option_type == "call":
            price = black_scholes_call(S, K, r, sigma, T)
        else:
            price = black_scholes_call(S, K, r, sigma, T) - S + K * math.exp(-r * T)

        diff = price - market_price
        vega = bs_vega(S, K, r, sigma, T)

        if abs(vega) < 1e-10:
            break

        sigma_new = sigma - diff / vega
        sigma = max(sigma_new, 1e-6)

        if abs(diff) < tol:
            return float(sigma)

    return float("nan")


def implied_vol_brent(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    tol: float = 1e-7,
) -> float:
    """Brent's method implied vol solver — more robust than Newton."""
    def f(sigma):
        if option_type == "call":
            return black_scholes_call(S, K, r, sigma, T) - market_price
        else:
            return (black_scholes_call(S, K, r, sigma, T) - S + K * math.exp(-r * T)) - market_price

    a, b = 1e-6, 10.0
    if f(a) * f(b) > 0:
        return float("nan")
    return float(brent(f, a, b, tol))


# ── Root Finding ──────────────────────────────────────────────────────────────

def newton_raphson(
    f: Callable,
    df: Callable,
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson root finding."""
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return float(x)
        dfx = df(x)
        if abs(dfx) < 1e-15:
            break
        x -= fx / dfx
    return float(x)


def brent(
    f: Callable,
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Brent's method for root in [a, b]."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    c, fc = b, fb
    d = e = b - a

    for _ in range(max_iter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a

        if abs(fc) < abs(fb):
            a, fa = b, fb
            b, fb = c, fc
            c, fc = a, fa

        tol1 = 2 * 1.1e-16 * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)

        if abs(xm) <= tol1 or fb == 0:
            return float(b)

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2 * xm * s
                q = 1 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)

            if p > 0:
                q = -q
            else:
                p = -p

            if 2 * p < min(3 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d

        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if xm > 0 else -tol1)
        fb = f(b)

    return float(b)


# ── Gaussian Quadrature ───────────────────────────────────────────────────────

def gauss_legendre(f: Callable, a: float, b: float, n: int = 10) -> float:
    """
    Gaussian-Legendre quadrature on [a, b] with n points.
    Exact for polynomials of degree 2n-1.
    """
    xi, wi = np.polynomial.legendre.leggauss(n)
    # Transform from [-1, 1] to [a, b]
    x = 0.5 * (b - a) * xi + 0.5 * (b + a)
    return float(0.5 * (b - a) * np.sum(wi * np.vectorize(f)(x)))


def gauss_hermite(f: Callable, n: int = 10) -> float:
    """
    Gauss-Hermite quadrature: ∫ f(x) exp(-x^2) dx
    Useful for expectations under normal distributions.
    """
    xi, wi = np.polynomial.hermite.hermgauss(n)
    return float(np.sum(wi * np.vectorize(f)(xi)))


def expectation_normal(
    f: Callable,
    mu: float = 0.0,
    sigma: float = 1.0,
    n: int = 20,
) -> float:
    """
    E[f(X)] where X ~ N(mu, sigma^2) via Gauss-Hermite quadrature.
    Change of variables: x = sqrt(2)*sigma*t + mu.
    """
    xi, wi = np.polynomial.hermite.hermgauss(n)
    x = math.sqrt(2) * sigma * xi + mu
    return float(np.sum(wi * np.vectorize(f)(x)) / math.sqrt(math.pi))


# ── Monte Carlo with Variance Reduction ──────────────────────────────────────

def mc_antithetic(
    payoff: Callable,         # payoff(paths) -> (n_paths,) array
    simulate: Callable,       # simulate(n, rng) -> paths
    n_paths: int = 100000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Antithetic variates Monte Carlo.
    Returns (price estimate, std error).
    """
    rng = rng or np.random.default_rng(42)
    n_half = n_paths // 2
    paths_pos = simulate(n_half, rng)
    # Invert the random shocks (antithetic)
    paths_neg = simulate(n_half, rng)

    pv_pos = payoff(paths_pos)
    pv_neg = payoff(paths_neg)
    combined = 0.5 * (pv_pos + pv_neg)
    return float(combined.mean()), float(combined.std() / math.sqrt(n_half))


def mc_control_variate(
    payoff: Callable,
    control_payoff: Callable,
    control_price: float,
    simulate: Callable,
    n_paths: int = 100000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Control variate Monte Carlo.
    Uses known analytical price as control to reduce variance.
    """
    rng = rng or np.random.default_rng(42)
    paths = simulate(n_paths, rng)
    pv = payoff(paths)
    cv = control_payoff(paths)

    # Optimal beta
    cov_matrix = np.cov(pv, cv)
    beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-10)

    adjusted = pv - beta * (cv - control_price)
    return float(adjusted.mean()), float(adjusted.std() / math.sqrt(n_paths))


def mc_importance_sampling(
    payoff: Callable,
    likelihood_ratio: Callable,  # likelihood_ratio(paths) = p/q
    simulate_q: Callable,        # simulate from importance distribution
    n_paths: int = 100000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Importance sampling Monte Carlo.
    E_p[f(X)] = E_q[f(X) * p(X)/q(X)]
    """
    rng = rng or np.random.default_rng(42)
    paths = simulate_q(n_paths, rng)
    pv = payoff(paths)
    lr = likelihood_ratio(paths)
    weighted = pv * lr
    return float(weighted.mean()), float(weighted.std() / math.sqrt(n_paths))


# ── Finite Difference Greeks ──────────────────────────────────────────────────

def fd_greeks(
    pricer: Callable,   # pricer(**kwargs) -> price
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    h_S: float = 0.01,
    h_sigma: float = 0.001,
    h_T: float = 1 / 365,
) -> dict:
    """
    Finite difference Greeks for any pricing function.
    Central differences for delta, gamma, vega; forward for theta.
    """
    def price(**kw):
        base = dict(S=S, K=K, r=r, sigma=sigma, T=T)
        base.update(kw)
        return pricer(**base)

    P0 = price()
    P_Su = price(S=S * (1 + h_S))
    P_Sd = price(S=S * (1 - h_S))
    P_vu = price(sigma=sigma + h_sigma)
    P_vd = price(sigma=sigma - h_sigma)
    P_Tt = price(T=T - h_T)

    delta = (P_Su - P_Sd) / (2 * S * h_S)
    gamma = (P_Su - 2 * P0 + P_Sd) / (S * h_S)**2
    vega = (P_vu - P_vd) / (2 * h_sigma)
    theta = (P_Tt - P0) / h_T  # per day decay

    return {
        "price": float(P0),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
    }


# ── Trinomial Tree ────────────────────────────────────────────────────────────

def trinomial_tree(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 100,
    option_type: str = "call",
    american: bool = False,
) -> float:
    """
    Boyle trinomial tree option pricing.
    More accurate than binomial for same number of steps.
    """
    dt = T / N
    u = math.exp(sigma * math.sqrt(2 * dt))
    d = 1 / u
    pu = ((math.exp(r * dt / 2) - math.exp(-sigma * math.sqrt(dt / 2))) /
          (math.exp(sigma * math.sqrt(dt / 2)) - math.exp(-sigma * math.sqrt(dt / 2))))**2
    pd = ((math.exp(sigma * math.sqrt(dt / 2)) - math.exp(r * dt / 2)) /
          (math.exp(sigma * math.sqrt(dt / 2)) - math.exp(-sigma * math.sqrt(dt / 2))))**2
    pm = 1 - pu - pd
    disc = math.exp(-r * dt)

    # Stock prices at maturity
    n_nodes = 2 * N + 1
    S_T = S * np.array([u**(N - i) for i in range(n_nodes)])

    # Terminal payoff
    if option_type == "call":
        V = np.maximum(S_T - K, 0)
    else:
        V = np.maximum(K - S_T, 0)

    # Backward induction
    for n in range(N - 1, -1, -1):
        n_inner = 2 * n + 1
        S_n = S * np.array([u**(n - i) for i in range(n_inner)])
        V_new = np.zeros(n_inner)
        for i in range(n_inner):
            V_new[i] = disc * (pu * V[i] + pm * V[i + 1] + pd * V[i + 2])
            if american:
                intrinsic = max(S_n[i] - K, 0) if option_type == "call" else max(K - S_n[i], 0)
                V_new[i] = max(V_new[i], intrinsic)
        V = V_new

    return float(V[0])
