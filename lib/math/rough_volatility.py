"""
Rough volatility models.

Implements:
  - Rough Bergomi (rBergomi) model simulation
  - Rough Heston simulation via fractional CIR
  - Fractional Brownian motion for volatility (H < 0.5 empirically)
  - Volterra kernel representation
  - At-the-money forward variance term structure
  - Volatility surface calibration (simplified)
  - Historical Hurst estimation for realized variance
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .fractal import fbm_cholesky


# ── Rough Bergomi ─────────────────────────────────────────────────────────────

@dataclass
class RoughBergomiParams:
    """
    Parameters for the rough Bergomi model (Bayer, Friz, Gatheral 2016).
      dS/S = sqrt(V) dW^1
      V_t  = xi(t) * exp(eta * W^H_t - 0.5 * eta^2 * t^{2H})
    where W^H is fBm with Hurst H, and xi is the forward variance curve.
    """
    H: float = 0.1          # Hurst parameter of vol (empirically ~0.1 for equities)
    eta: float = 1.9        # vol of vol scaling
    rho: float = -0.9       # correlation between price and vol BM
    xi0: float = 0.04       # initial spot variance (atm var for flat term structure)


def rough_bergomi_simulate(
    params: RoughBergomiParams,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    S0: float = 100.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate rough Bergomi model.

    Returns:
      S : shape (n_paths, n_steps+1) — price paths
      V : shape (n_paths, n_steps+1) — instantaneous variance paths
    """
    rng = rng or np.random.default_rng()
    p = params
    dt = T / n_steps
    H = p.H
    alpha = H - 0.5  # H - 0.5 < 0 for rough vol

    # Simulate correlated Brownian motions
    # W^1 (price BM), W^2 (independent), W^H = rho*W^1 + sqrt(1-rho^2)*W^2
    dW1 = rng.standard_normal((n_paths, n_steps)) * math.sqrt(dt)
    dW2 = rng.standard_normal((n_paths, n_steps)) * math.sqrt(dt)
    dWV = p.rho * dW1 + math.sqrt(1 - p.rho ** 2) * dW2

    # Build fractional Brownian motion for log(V) via Volterra convolution
    # W^H_t = int_0^t K(t,s) dW_s,  K(t,s) = (t-s)^alpha
    # Use Riemann sum approximation
    t_grid = np.arange(1, n_steps + 1) * dt
    WH = np.zeros((n_paths, n_steps + 1))  # W^H at each time step

    for i in range(1, n_steps + 1):
        # Volterra: WH_i = sum_{j<i} (i-j)^alpha * dt^{alpha+0.5} * dWV_j
        kernel = np.arange(1, i + 1, dtype=float) ** alpha
        kernel /= (kernel ** 2).sum() ** 0.5 * math.sqrt(dt)  # normalize variance
        # Actually use exact variance: E[(W^H_t)^2] = t^{2H}
        scale = math.sqrt(t_grid[i - 1] ** (2 * H)) if i > 0 else 0.0
        WH[:, i] = np.dot(dWV[:, :i], kernel[::-1] * math.sqrt(dt)) * scale / max(
            math.sqrt(np.sum(kernel ** 2) * dt), 1e-10)

    # Instantaneous variance: V_t = xi(t) * exp(eta*WH_t - 0.5*eta^2*t^{2H})
    V = np.zeros((n_paths, n_steps + 1))
    V[:, 0] = p.xi0
    for i in range(1, n_steps + 1):
        t_i = t_grid[i - 1]
        log_V = math.log(p.xi0) + p.eta * WH[:, i] - 0.5 * p.eta ** 2 * t_i ** (2 * H)
        V[:, i] = np.exp(log_V)

    # Price paths
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for i in range(1, n_steps + 1):
        v = np.maximum(V[:, i - 1], 0.0)
        S[:, i] = S[:, i - 1] * np.exp(-0.5 * v * dt + np.sqrt(v) * dW1[:, i - 1])

    return S, V


def rough_bergomi_implied_vol_atm(
    params: RoughBergomiParams,
    maturities: np.ndarray,
    n_paths: int = 10_000,
    n_steps_per_year: int = 252,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Compute ATM implied vols at given maturities via MC.
    Returns implied vols (annualized).
    """
    from scipy.optimize import brentq

    rng = rng or np.random.default_rng()
    ivols = []
    S0 = 100.0

    for T in maturities:
        n_steps = max(int(n_steps_per_year * T), 10)
        S, _ = rough_bergomi_simulate(params, T, n_steps, n_paths, S0, rng)
        S_T = S[:, -1]
        K = S0  # ATM strike

        mc_call = float(np.mean(np.maximum(S_T - K, 0.0)))

        # Black-Scholes inversion
        def bs_call(sigma):
            d1 = math.log(S0 / K) / (sigma * math.sqrt(T)) + 0.5 * sigma * math.sqrt(T)
            d2 = d1 - sigma * math.sqrt(T)
            from scipy.stats import norm
            return S0 * norm.cdf(d1) - K * norm.cdf(d2)

        try:
            iv = brentq(lambda s: bs_call(s) - mc_call, 0.001, 5.0)
        except Exception:
            iv = math.sqrt(params.xi0)

        ivols.append(iv)

    return np.array(ivols)


# ── Fractional CIR (rough Heston) ─────────────────────────────────────────────

@dataclass
class RoughHestonParams:
    H: float = 0.1          # Hurst index (< 0.5 for rough)
    kappa: float = 0.3      # mean reversion speed
    theta: float = 0.04     # long-run variance
    nu: float = 0.3         # vol of vol
    rho: float = -0.7       # price-vol correlation
    V0: float = 0.04


def rough_heston_simulate(
    params: RoughHestonParams,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    S0: float = 100.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rough Heston via fractional CIR (Euler scheme on fractional integral).
    """
    rng = rng or np.random.default_rng()
    p = params
    dt = T / n_steps
    H = p.H
    alpha = H - 0.5

    dW1 = rng.standard_normal((n_paths, n_steps)) * math.sqrt(dt)
    dW2 = rng.standard_normal((n_paths, n_steps)) * math.sqrt(dt)
    dWV = p.rho * dW1 + math.sqrt(1 - p.rho ** 2) * dW2

    V = np.full((n_paths, n_steps + 1), p.V0)
    S = np.full((n_paths, n_steps + 1), S0)

    # Volterra kernel for rough variance increments
    kernel = np.array([(i + 1) ** alpha - i ** alpha for i in range(n_steps)]) / math.gamma(H + 0.5)

    for i in range(1, n_steps + 1):
        v_pos = np.maximum(V[:, i - 1], 0.0)
        # Fractional increment
        j_max = min(i, 50)  # truncate kernel memory
        frac_inc = np.zeros(n_paths)
        for j in range(1, j_max + 1):
            frac_inc += kernel[j - 1] * dWV[:, i - j] if i - j >= 0 else 0

        V[:, i] = np.maximum(
            V[:, i - 1]
            + p.kappa * (p.theta - v_pos) * dt
            + p.nu * np.sqrt(v_pos) * frac_inc,
            0.0,
        )
        S[:, i] = S[:, i - 1] * np.exp(-0.5 * v_pos * dt + np.sqrt(v_pos) * dW1[:, i - 1])

    return S, V


# ── Historical Hurst for realized variance ────────────────────────────────────

def realized_variance_hurst(
    prices: np.ndarray,
    window: int = 20,
    n_lags: int = 30,
) -> float:
    """
    Estimate Hurst exponent of realized variance series.
    Returns H — empirically ~0.1 for equity vol (rough).
    """
    log_returns = np.diff(np.log(prices))
    # Rolling realized variance
    rv = np.array([
        np.sum(log_returns[max(0, i - window): i] ** 2)
        for i in range(window, len(log_returns) + 1)
    ])
    if len(rv) < 30:
        return 0.1

    from .fractal import hurst_dfa
    return hurst_dfa(np.log(rv + 1e-12))


# ── Vol surface moment matching ────────────────────────────────────────────────

def rough_vol_moment_match(
    realized_vols: np.ndarray,
    H_init: float = 0.1,
) -> dict:
    """
    Match rough vol parameters to observed realized volatility moments.
    Returns estimated (H, eta, xi0).
    """
    log_rv = np.log(realized_vols ** 2 + 1e-12)
    xi0 = float(np.exp(np.mean(log_rv)))
    var_log_rv = float(np.var(log_rv))

    from .fractal import hurst_dfa
    H = hurst_dfa(log_rv)

    # eta^2 * T^{2H} = var(log V_T)
    T = 1.0 / 252  # daily
    eta = math.sqrt(var_log_rv / max(T ** (2 * H), 1e-10))
    eta = float(np.clip(eta, 0.1, 5.0))

    return {"H": float(H), "eta": eta, "xi0": xi0}
