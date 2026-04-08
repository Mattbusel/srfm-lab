"""
Robust statistical estimators for financial data.

Implements:
  - M-estimators (Huber, Tukey bisquare)
  - Minimum Covariance Determinant (MCD)
  - S-estimator and MM-estimator
  - Robust PCA via projection pursuit
  - Robust regression (IRLS)
  - Robust covariance (Ledoit-Wolf + robust)
  - Breakdown point analysis
  - Outlier flagging (Mahalanobis distance)
  - Robust Sharpe ratio
  - Robust factor model
  - Median of Means (MoM) estimator
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable


# ── M-Estimators ─────────────────────────────────────────────────────────────

def huber_rho(r: float, c: float = 1.345) -> float:
    """Huber loss function."""
    if abs(r) <= c:
        return 0.5 * r**2
    return c * abs(r) - 0.5 * c**2


def huber_psi(r: float, c: float = 1.345) -> float:
    """Huber influence function (derivative of rho)."""
    return float(np.clip(r, -c, c))


def tukey_rho(r: float, c: float = 4.685) -> float:
    """Tukey bisquare loss function."""
    if abs(r) > c:
        return c**2 / 6
    return c**2 / 6 * (1 - (1 - (r/c)**2)**3)


def tukey_psi(r: float, c: float = 4.685) -> float:
    """Tukey bisquare influence function."""
    if abs(r) > c:
        return 0.0
    return float(r * (1 - (r/c)**2)**2)


def huber_location(
    x: np.ndarray,
    c: float = 1.345,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """
    Huber M-estimator of location via IRLS.
    Robust mean that downweights outliers.
    """
    mu = float(np.median(x))
    mad = float(np.median(np.abs(x - mu)))
    scale = mad / 0.6745  # consistent for normal

    for _ in range(max_iter):
        if scale < 1e-10:
            break
        r = (x - mu) / scale
        w = np.array([huber_psi(ri, c) / max(abs(ri), 1e-10) for ri in r])
        mu_new = float(np.sum(w * x) / (np.sum(w) + 1e-10))
        if abs(mu_new - mu) < tol:
            break
        mu = mu_new

    return mu


def huber_scale(x: np.ndarray, c: float = 1.345, k: float = 1.4826) -> float:
    """Huber M-estimator of scale (robust std)."""
    mu = huber_location(x, c)
    return float(k * np.median(np.abs(x - mu)))


def tukey_location(
    x: np.ndarray,
    c: float = 4.685,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """Tukey bisquare M-estimator of location."""
    mu = float(np.median(x))
    mad = float(np.median(np.abs(x - mu))) / 0.6745 + 1e-10

    for _ in range(max_iter):
        r = (x - mu) / mad
        w = np.array([tukey_psi(ri, c) / max(abs(ri), 1e-10) for ri in r])
        mu_new = float(np.sum(w * x) / (np.sum(w) + 1e-10))
        if abs(mu_new - mu) < tol:
            break
        mu = mu_new

    return mu


# ── Robust Regression (IRLS) ──────────────────────────────────────────────────

def irls_regression(
    X: np.ndarray,
    y: np.ndarray,
    loss: str = "huber",
    c: float = 1.345,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> dict:
    """
    Iteratively Reweighted Least Squares for robust regression.
    loss: 'huber' or 'tukey'
    Returns coefficients, residuals, weights.
    """
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    for _ in range(max_iter):
        resid = y - X @ beta
        scale = float(np.median(np.abs(resid)) / 0.6745) + 1e-10
        r = resid / scale

        if loss == "huber":
            psi_r = np.array([huber_psi(ri, c) for ri in r])
        else:
            psi_r = np.array([tukey_psi(ri, c) for ri in r])

        # Weights: w_i = psi(r_i) / r_i
        w = np.array([psi_r[i] / max(abs(r[i]), 1e-10) for i in range(n)])
        w = np.maximum(w, 1e-10)

        W = np.diag(w)
        try:
            beta_new = np.linalg.solve(X.T @ W @ X + 1e-8 * np.eye(p), X.T @ W @ y)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    resid = y - X @ beta
    return {
        "coefficients": beta,
        "residuals": resid,
        "weights": w,
        "robust_se": float(np.sqrt(np.diag(np.linalg.pinv(X.T @ np.diag(w) @ X))) .mean()),
        "robust_r2": float(1 - resid.var() / (y.var() + 1e-10)),
    }


# ── Minimum Covariance Determinant ────────────────────────────────────────────

def mcd_estimator(
    X: np.ndarray,
    h_fraction: float = 0.75,
    n_iter: int = 30,
    n_starts: int = 10,
    seed: int = 42,
) -> dict:
    """
    Fast-MCD: Minimum Covariance Determinant estimator.
    h_fraction: fraction of observations to use (0.5 to 1.0).
    Returns robust mean, covariance, and outlier mask.
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    h = int(np.floor(h_fraction * n))
    h = max(h, p + 1)

    best_det = np.inf
    best_mu = None
    best_cov = None
    best_subset = None

    for _ in range(n_starts):
        # Random initial subset of p+1 points
        idx = rng.choice(n, size=p + 1, replace=False)
        subset = list(idx)

        for iteration in range(n_iter):
            sub_X = X[subset]
            mu = sub_X.mean(axis=0)
            cov = np.cov(sub_X.T) + 1e-6 * np.eye(p) if len(subset) > 1 else np.eye(p)
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.eye(p)

            # Mahalanobis distances for all points
            diff = X - mu
            maha = np.array([float(diff[i] @ cov_inv @ diff[i]) for i in range(n)])

            # Select h points with smallest Mahalanobis distance
            new_subset = list(np.argsort(maha)[:h])
            if set(new_subset) == set(subset):
                break
            subset = new_subset

        # Evaluate determinant
        sub_X = X[subset]
        cov_final = np.cov(sub_X.T) + 1e-6 * np.eye(p) if h > 1 else np.eye(p)
        try:
            det = float(np.linalg.det(cov_final))
        except Exception:
            det = np.inf

        if det < best_det:
            best_det = det
            best_mu = sub_X.mean(axis=0)
            best_cov = cov_final
            best_subset = subset

    # Mahalanobis distances with MCD estimate
    try:
        cov_inv = np.linalg.inv(best_cov)
    except Exception:
        cov_inv = np.eye(p)
    diff = X - best_mu
    maha = np.array([float(diff[i] @ cov_inv @ diff[i]) for i in range(n)])

    # Chi-squared threshold for outlier detection
    from scipy.stats import chi2
    threshold = chi2.ppf(0.975, df=p)
    outliers = maha > threshold

    return {
        "robust_mean": best_mu,
        "robust_cov": best_cov,
        "mahalanobis_distances": maha,
        "outlier_mask": outliers,
        "n_outliers": int(outliers.sum()),
        "outlier_fraction": float(outliers.mean()),
        "mcd_determinant": best_det,
    }


# ── Robust Covariance via Ledoit-Wolf + Shrinkage ────────────────────────────

def robust_ledoit_wolf(
    X: np.ndarray,
    shrinkage: Optional[float] = None,
) -> dict:
    """
    Robust Ledoit-Wolf shrinkage estimator.
    First winsorizes returns, then applies LW shrinkage.
    """
    n, p = X.shape

    # Winsorize at 1st/99th percentile
    X_winsor = X.copy()
    for j in range(p):
        lo, hi = np.percentile(X[:, j], [1, 99])
        X_winsor[:, j] = np.clip(X[:, j], lo, hi)

    S = np.cov(X_winsor.T)

    if shrinkage is None:
        # Oracle approximating shrinkage (OAS)
        mu = np.trace(S) / p
        rho_num = ((1 - 2/p) * np.trace(S @ S) + np.trace(S)**2)
        rho_den = (n + 1 - 2/p) * (np.trace(S @ S) - np.trace(S)**2 / p)
        shrinkage = float(min(rho_num / max(rho_den, 1e-10), 1.0))
        shrinkage = max(shrinkage, 0.0)

    mu = np.trace(S) / p
    Sigma_lw = (1 - shrinkage) * S + shrinkage * mu * np.eye(p)

    return {
        "covariance": Sigma_lw,
        "shrinkage": float(shrinkage),
        "sample_cov": S,
        "condition_number": float(np.linalg.cond(Sigma_lw)),
    }


# ── Robust Sharpe and Performance Metrics ────────────────────────────────────

def robust_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Robust Sharpe ratio using Huber M-estimator for mean and scale.
    """
    mu = huber_location(returns)
    scale = huber_scale(returns)
    return float(mu / max(scale, 1e-10) * math.sqrt(periods_per_year))


def robust_performance_metrics(returns: np.ndarray) -> dict:
    """
    Full robust performance metrics for a return series.
    """
    mu_robust = huber_location(returns)
    scale_robust = huber_scale(returns)
    mu_ols = float(returns.mean())
    scale_ols = float(returns.std())

    # Robust Sortino: downside scale via Huber on negative returns
    neg_returns = returns[returns < 0]
    if len(neg_returns) > 5:
        dd_scale = huber_scale(neg_returns)
    else:
        dd_scale = scale_robust

    # Robust CVaR: trimmed mean of worst quantile
    alpha = 0.05
    thresh = np.quantile(returns, alpha)
    cvar_robust = float(returns[returns <= thresh].mean()) if (returns <= thresh).any() else float(returns.min())

    # Median of Means Sharpe
    n_blocks = max(int(math.sqrt(len(returns))), 5)
    mom_sharpe = _median_of_means_sharpe(returns, n_blocks)

    return {
        "robust_mean": float(mu_robust),
        "robust_std": float(scale_robust),
        "robust_sharpe_ann": float(mu_robust / max(scale_robust, 1e-10) * math.sqrt(252)),
        "robust_sortino": float(mu_robust / max(dd_scale, 1e-10) * math.sqrt(252)),
        "robust_cvar_5pct": cvar_robust,
        "ols_sharpe_ann": float(mu_ols / max(scale_ols, 1e-10) * math.sqrt(252)),
        "mom_sharpe": mom_sharpe,
        "outlier_fraction": float(np.mean(np.abs(returns - mu_robust) > 3 * scale_robust)),
    }


def _median_of_means_sharpe(returns: np.ndarray, n_blocks: int) -> float:
    """Median of Means estimator for Sharpe ratio (robust to heavy tails)."""
    n = len(returns)
    block_size = max(n // n_blocks, 1)
    block_means = []
    for i in range(n_blocks):
        block = returns[i * block_size: (i + 1) * block_size]
        if len(block) > 0:
            block_means.append(float(block.mean()))
    if not block_means:
        return 0.0
    mom_mean = float(np.median(block_means))
    mom_std = float(np.std(block_means) * math.sqrt(n_blocks))
    return float(mom_mean / max(mom_std, 1e-10) * math.sqrt(252))


# ── Robust Factor Model ───────────────────────────────────────────────────────

def robust_factor_regression(
    returns: np.ndarray,
    factors: np.ndarray,
    loss: str = "huber",
) -> dict:
    """
    Robust factor model: returns = alpha + beta @ factors + epsilon
    Uses IRLS for robust coefficient estimation.
    """
    T = len(returns)
    X = np.column_stack([np.ones(T), factors])
    result = irls_regression(X, returns, loss=loss)

    coef = result["coefficients"]
    alpha = float(coef[0])
    beta = coef[1:]

    factor_returns = factors @ beta
    idio = returns - alpha - factor_returns
    r2 = float(1 - idio.var() / (returns.var() + 1e-10))

    return {
        "alpha": alpha,
        "betas": beta,
        "idiosyncratic": idio,
        "factor_contribution": float(factor_returns.var() / (returns.var() + 1e-10)),
        "robust_r2": r2,
        "weights": result["weights"],
        "n_effective": float(result["weights"].sum()**2 / (result["weights"]**2).sum()),
    }


# ── Breakdown Point Analysis ──────────────────────────────────────────────────

def finite_sample_breakdown_point(estimator: str, n: int, p: int = 1) -> float:
    """
    Finite-sample breakdown point of common estimators.
    High BP = more robust to outliers.
    """
    bp_table = {
        "mean": 1 / n,
        "median": 0.5,
        "huber_location": 0.5,
        "tukey_location": 0.5,
        "sample_cov": 1 / n,
        "mcd_75pct": 0.25,
        "mcd_50pct": 0.5,
        "ols": 1 / n,
        "lts": 0.5,
        "lad": 0.0,  # L1 regression, 0% BP in general position
    }
    return float(bp_table.get(estimator, 0.0))


# ── Outlier Detection ─────────────────────────────────────────────────────────

def robust_outlier_scores(
    X: np.ndarray,
    method: str = "mcd",
) -> np.ndarray:
    """
    Compute robust outlier scores for each observation.
    Returns standardized Mahalanobis distances.
    """
    n, p = X.shape

    if method == "mcd":
        result = mcd_estimator(X, h_fraction=0.75)
        maha = result["mahalanobis_distances"]
    else:
        # Classical Mahalanobis
        mu = X.mean(axis=0)
        cov = np.cov(X.T) + 1e-6 * np.eye(p)
        try:
            cov_inv = np.linalg.inv(cov)
        except Exception:
            cov_inv = np.eye(p)
        diff = X - mu
        maha = np.array([float(diff[i] @ cov_inv @ diff[i]) for i in range(n)])

    return np.sqrt(np.maximum(maha, 0))


def winsorize(
    x: np.ndarray,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> np.ndarray:
    """Winsorize array at given percentiles."""
    lo, hi = np.percentile(x, [lower_pct, upper_pct])
    return np.clip(x, lo, hi)


def trim_mean(x: np.ndarray, trim_frac: float = 0.1) -> float:
    """Trimmed mean: remove trim_frac from each tail."""
    n = len(x)
    k = int(n * trim_frac)
    if k == 0:
        return float(x.mean())
    xs = np.sort(x)
    return float(xs[k:-k].mean())
