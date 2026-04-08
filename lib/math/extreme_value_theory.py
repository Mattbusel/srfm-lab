"""
Extreme Value Theory (EVT) for tail risk estimation.

Implements:
  - Generalized Extreme Value (GEV) distribution
  - Generalized Pareto Distribution (GPD) — Peaks Over Threshold
  - Maximum Likelihood fitting of GPD / GEV
  - VaR and CVaR via EVT (more accurate than normal assumption)
  - Expected Shortfall (ES) from GPD
  - Hill estimator for tail index
  - Pickands estimator
  - Return level estimation (1-in-N year events)
  - Bivariate extremes (joint tail dependence)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy import stats, optimize


# ── GPD — Peaks Over Threshold ────────────────────────────────────────────────

@dataclass
class GPDParams:
    xi: float = 0.0     # shape (xi>0: heavy tail; xi=0: exponential; xi<0: bounded)
    sigma: float = 1.0  # scale
    threshold: float = 0.0

    @property
    def mean_excess(self) -> float:
        """Mean excess over threshold = sigma/(1-xi) for xi < 1."""
        if self.xi >= 1:
            return float("inf")
        return self.sigma / (1 - self.xi)


def gpd_pdf(x: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    """GPD PDF."""
    x = np.asarray(x)
    sigma = max(sigma, 1e-10)
    if abs(xi) < 1e-8:
        return np.exp(-x / sigma) / sigma
    z = 1 + xi * x / sigma
    mask = z > 0
    pdf = np.zeros_like(x, dtype=float)
    pdf[mask] = (1 / sigma) * z[mask] ** (-(1 / xi + 1))
    return pdf


def gpd_cdf(x: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    x = np.asarray(x)
    sigma = max(sigma, 1e-10)
    if abs(xi) < 1e-8:
        return 1 - np.exp(-x / sigma)
    z = np.maximum(1 + xi * x / sigma, 0)
    return 1 - z ** (-1 / xi)


def fit_gpd(
    exceedances: np.ndarray,
    method: str = "mle",
) -> GPDParams:
    """
    Fit GPD to threshold exceedances.
    exceedances = returns exceeding threshold (positive values).
    """
    x = exceedances[exceedances > 0]
    if len(x) < 10:
        return GPDParams(xi=0.0, sigma=float(x.mean()) if len(x) > 0 else 1.0)

    if method == "mle":
        def neg_ll(params):
            xi, sigma = params
            sigma = max(sigma, 1e-8)
            if abs(xi) < 1e-8:
                return len(x) * math.log(sigma) + x.sum() / sigma
            z = 1 + xi * x / sigma
            if (z <= 0).any():
                return 1e12
            return (len(x) * math.log(sigma)
                    + (1 / xi + 1) * np.log(z).sum())

        # PWM initial estimate
        x_sorted = np.sort(x)
        n = len(x_sorted)
        b0 = x_sorted.mean()
        b1 = sum((n - j) / (n * (n - 1)) * x_sorted[j] for j in range(n))
        xi0 = 2 - b0 / (b0 - 2 * b1)
        sigma0 = 2 * b0 * b1 / (b0 - 2 * b1)

        res = optimize.minimize(neg_ll, x0=[xi0, max(sigma0, 0.01)],
                                method="Nelder-Mead",
                                options={"maxiter": 2000})
        xi, sigma = res.x
        return GPDParams(xi=float(xi), sigma=float(max(sigma, 1e-6)))

    elif method == "pwm":
        # Probability-weighted moments
        x_sorted = np.sort(x)
        n = len(x_sorted)
        b0 = x_sorted.mean()
        b1 = sum((n - j) / (n * (n - 1)) * x_sorted[j] for j in range(n))
        xi = 2 - b0 / (b0 - 2 * b1)
        sigma = 2 * b0 * b1 / (b0 - 2 * b1)
        return GPDParams(xi=float(xi), sigma=float(max(sigma, 1e-6)))

    raise ValueError(f"Unknown method: {method}")


# ── VaR and CVaR via GPD ──────────────────────────────────────────────────────

def gpd_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    threshold_pct: float = 90,
    method: str = "mle",
) -> float:
    """
    VaR at given confidence level via GPD POT method.
    More accurate than Gaussian VaR in the tails.
    """
    losses = -returns  # flip to losses
    u = np.percentile(losses, threshold_pct)
    exceedances = losses[losses > u] - u
    n = len(losses)
    nu = len(exceedances)

    if nu < 10:
        return float(np.percentile(losses, confidence * 100))

    params = fit_gpd(exceedances, method)
    xi, sigma = params.xi, params.sigma

    # VaR formula: u + sigma/xi * ((n/nu * (1-alpha))^{-xi} - 1)
    p = 1 - confidence
    if abs(xi) < 1e-8:
        var = u - sigma * math.log(n / nu * p)
    else:
        var = u + sigma / xi * ((n / nu * p) ** (-xi) - 1)
    return float(var)


def gpd_cvar(
    returns: np.ndarray,
    confidence: float = 0.99,
    threshold_pct: float = 90,
) -> float:
    """
    CVaR (Expected Shortfall) via GPD.
    ES = VaR + E[loss - VaR | loss > VaR]
       = VaR/(1-xi) + (sigma - xi*u)/(1-xi)  for xi < 1
    """
    losses = -returns
    u = np.percentile(losses, threshold_pct)
    exceedances = losses[losses > u] - u

    if len(exceedances) < 10:
        var = float(np.percentile(losses, confidence * 100))
        tail = losses[losses > var]
        return float(tail.mean()) if len(tail) > 0 else var

    params = fit_gpd(exceedances)
    var = gpd_var(returns, confidence, threshold_pct)

    if params.xi >= 1:
        return float("inf")

    es = var / (1 - params.xi) + (params.sigma - params.xi * u) / (1 - params.xi)
    return float(es)


# ── GEV distribution ──────────────────────────────────────────────────────────

@dataclass
class GEVParams:
    xi: float = 0.0       # shape (Gumbel=0, Fréchet>0, Weibull<0)
    mu: float = 0.0       # location
    sigma: float = 1.0    # scale

    @property
    def distribution_type(self) -> str:
        if abs(self.xi) < 0.1:
            return "Gumbel"
        elif self.xi > 0:
            return "Fréchet (heavy tail)"
        else:
            return "Weibull (bounded tail)"


def fit_gev_block_maxima(
    returns: np.ndarray,
    block_size: int = 21,  # ~monthly
) -> GEVParams:
    """
    Fit GEV to block maxima (e.g., monthly maximum losses).
    """
    losses = -returns
    n_blocks = len(losses) // block_size
    if n_blocks < 10:
        return GEVParams()

    block_max = np.array([
        losses[i * block_size: (i + 1) * block_size].max()
        for i in range(n_blocks)
    ])

    result = stats.genextreme.fit(block_max)
    xi, mu, sigma = result[0], result[1], result[2]
    return GEVParams(xi=float(xi), mu=float(mu), sigma=float(sigma))


def gev_return_level(params: GEVParams, return_period: float) -> float:
    """
    Return level for a given return period (in blocks).
    E.g., 100-month return level of monthly maxima.
    """
    p = 1 - 1 / return_period
    if abs(params.xi) < 1e-8:
        return params.mu - params.sigma * math.log(-math.log(p))
    return params.mu + params.sigma / params.xi * ((-math.log(p)) ** (-params.xi) - 1)


# ── Hill estimator ────────────────────────────────────────────────────────────

def hill_estimator(
    returns: np.ndarray,
    k_range: Optional[tuple] = None,
) -> dict:
    """
    Hill estimator for tail index alpha (= 1/xi for Pareto tail).
    Uses upper k order statistics of losses.
    Higher alpha → lighter tail.

    Returns dict with estimated alpha for various k, and stable estimate.
    """
    losses = np.sort(-returns)[::-1]  # descending
    n = len(losses)
    if k_range is None:
        k_range = (max(10, n // 20), min(n // 4, 200))

    k_values = range(k_range[0], k_range[1])
    alphas = []
    for k in k_values:
        if k >= n:
            break
        log_ratios = np.log(losses[:k]) - math.log(losses[k])
        if log_ratios.mean() > 0:
            alphas.append(1.0 / log_ratios.mean())
        else:
            alphas.append(np.nan)

    alphas = np.array(alphas)
    valid = alphas[~np.isnan(alphas)]
    stable_alpha = float(np.median(valid)) if len(valid) > 0 else 2.0

    return {
        "alpha": stable_alpha,
        "xi": 1.0 / stable_alpha,   # tail index
        "k_values": list(k_values),
        "alpha_by_k": alphas.tolist(),
        "tail_type": (
            "heavy" if stable_alpha < 3
            else "medium" if stable_alpha < 6
            else "light"
        ),
    }


# ── Comprehensive tail risk report ────────────────────────────────────────────

def tail_risk_report(returns: np.ndarray, confidences: list = [0.95, 0.99, 0.999]) -> dict:
    """Full EVT-based tail risk report."""
    hill = hill_estimator(returns)
    gev = fit_gev_block_maxima(returns)

    report = {
        "tail_index_alpha": hill["alpha"],
        "shape_xi": hill["xi"],
        "tail_type": hill["tail_type"],
        "gev_params": {"xi": gev.xi, "mu": gev.mu, "sigma": gev.sigma},
        "gev_distribution": gev.distribution_type,
    }

    for conf in confidences:
        label = f"{conf:.1%}"
        report[f"var_{label}"] = gpd_var(returns, conf)
        report[f"cvar_{label}"] = gpd_cvar(returns, conf)

    # 1-in-N year return level
    annual_blocks = int(252 / 21)  # monthly blocks per year
    for n_years in [5, 10, 25, 100]:
        rp = n_years * annual_blocks
        report[f"return_level_{n_years}yr"] = gev_return_level(gev, rp)

    return report
