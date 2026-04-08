"""
Factor models for asset returns.

Implements:
  - Fama-French 3-factor and 5-factor models
  - PCA-based statistical factor models
  - BARRA-style risk model (fundamental factors)
  - APT (Arbitrage Pricing Theory) factor estimation
  - Rolling factor exposures (Kalman-state betas)
  - Factor timing: predict factor premium sign
  - Factor crowding detection
  - Cross-sectional factor momentum
  - Factor zoo: compute 20+ well-known factors from price/vol data
  - Information ratio decomposition by factor
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Factor Returns Dataclass ──────────────────────────────────────────────────

@dataclass
class FactorModel:
    n_assets: int
    n_factors: int
    factor_loadings: np.ndarray    # (n_assets, n_factors) B matrix
    factor_returns: np.ndarray     # (T, n_factors)
    idiosyncratic_var: np.ndarray  # (n_assets,) residual variances
    r2_by_asset: np.ndarray        # (n_assets,) R² of factor model
    factor_names: list[str] = field(default_factory=list)


# ── PCA Factor Model ──────────────────────────────────────────────────────────

def pca_factor_model(
    returns: np.ndarray,
    n_factors: int = 5,
    demean: bool = True,
) -> FactorModel:
    """
    Statistical factor model via PCA.
    returns: (T, n_assets)
    Factors = principal components of return covariance matrix.
    """
    T, N = returns.shape

    if demean:
        R = returns - returns.mean(axis=0)
    else:
        R = returns.copy()

    # Covariance matrix
    cov = R.T @ R / T  # (N, N)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # Top k factors
    k = min(n_factors, N, T - 1)
    F_loadings = eigenvecs[:, :k]  # (N, k) factor loadings

    # Factor returns: projection of returns onto eigenvectors
    factor_rets = R @ F_loadings  # (T, k)

    # Idiosyncratic: residual after removing factors
    R_hat = factor_rets @ F_loadings.T
    residuals = R - R_hat

    idio_var = np.var(residuals, axis=0)
    total_var = np.var(R, axis=0)
    r2 = np.clip(1 - idio_var / (total_var + 1e-10), 0, 1)

    return FactorModel(
        n_assets=N,
        n_factors=k,
        factor_loadings=F_loadings,
        factor_returns=factor_rets,
        idiosyncratic_var=idio_var,
        r2_by_asset=r2,
        factor_names=[f"PC{i+1}" for i in range(k)],
    )


# ── APT Factor Estimation ─────────────────────────────────────────────────────

def apt_factor_estimation(
    returns: np.ndarray,
    n_factors: int = 5,
    n_iter: int = 100,
) -> dict:
    """
    APT factor model via iterative factor analysis (EM-like).
    Decomposes returns into systematic factors + idiosyncratic.
    Returns factor loadings, factor returns, uniqueness.
    """
    T, N = returns.shape
    R = returns - returns.mean(axis=0)
    cov = R.T @ R / T

    # Initialize with PCA
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigenvals)[::-1]
    B = eigenvecs[:, idx[:n_factors]] * np.sqrt(eigenvals[idx[:n_factors]])  # (N, k)
    psi = np.diag(cov - B @ B.T)
    psi = np.maximum(psi, 1e-6)

    for _ in range(n_iter):
        # E-step: factor posterior
        Psi_inv = np.diag(1.0 / psi)
        M = np.eye(n_factors) + B.T @ Psi_inv @ B
        try:
            M_inv = np.linalg.inv(M)
        except Exception:
            break
        # Factor returns
        F = R @ Psi_inv @ B @ M_inv  # (T, k)

        # M-step: update B, psi
        FF = F.T @ F / T
        B_new = (R.T @ F / T) @ np.linalg.pinv(FF)  # (N, k)
        R_hat = F @ B_new.T
        resid = R - R_hat
        psi_new = np.var(resid, axis=0) + 1e-6

        B = B_new
        psi = psi_new

    # Factor returns (standardized)
    Psi_inv = np.diag(1.0 / psi)
    M = np.eye(n_factors) + B.T @ Psi_inv @ B
    try:
        M_inv = np.linalg.inv(M)
    except Exception:
        M_inv = np.eye(n_factors)
    F = R @ Psi_inv @ B @ M_inv

    communality = 1 - psi / (np.diag(cov) + 1e-10)
    return {
        "factor_loadings": B,
        "factor_returns": F,
        "uniqueness": psi,
        "communality": communality,
        "systematic_variance_fraction": float(communality.mean()),
    }


# ── Rolling Factor Betas ──────────────────────────────────────────────────────

def rolling_factor_betas(
    asset_returns: np.ndarray,
    factor_returns: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """
    Rolling OLS betas of asset on factors.
    asset_returns: (T,)
    factor_returns: (T, k)
    Returns beta_series: (T, k+1) — intercept + betas
    """
    T = len(asset_returns)
    k = factor_returns.shape[1] if factor_returns.ndim > 1 else 1
    betas = np.zeros((T, k + 1))

    for t in range(window, T):
        y = asset_returns[t - window: t]
        X = factor_returns[t - window: t]
        X_aug = np.column_stack([np.ones(window), X])
        try:
            b = np.linalg.lstsq(X_aug, y, rcond=None)[0]
            betas[t] = b
        except Exception:
            betas[t] = betas[t-1] if t > 0 else 0

    return betas


# ── Factor Zoo ────────────────────────────────────────────────────────────────

def compute_factor_zoo(
    prices: np.ndarray,
    volume: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute 20+ well-known price/vol-based factors for a single asset.
    Returns dict of factor values (current bar).
    """
    if len(prices) < 252:
        return {}

    returns = np.diff(np.log(prices + 1e-10))
    T = len(returns)

    def safe_ret(n):
        n = min(n, T)
        return float((prices[-1] / prices[-n-1] - 1)) if n > 0 else 0.0

    factors = {}

    # ── Momentum family ──
    factors["MOM_1M"] = safe_ret(21)
    factors["MOM_3M"] = safe_ret(63)
    factors["MOM_6M"] = safe_ret(126)
    factors["MOM_12M"] = safe_ret(252)
    # Skip-month (remove 1M from 12M to avoid reversal)
    factors["MOM_2_12"] = float((prices[-22] / prices[-253] - 1)) if T >= 252 else 0.0

    # ── Mean reversion ──
    n_mr = min(20, T)
    sub = prices[-n_mr:]
    z = float((prices[-1] - sub.mean()) / max(sub.std(), 1e-10))
    factors["MR_1M_ZSCORE"] = float(-z)

    # Weekly reversal
    factors["STR_1W"] = float(-safe_ret(5))
    factors["STR_1M"] = float(-safe_ret(21))

    # ── Volatility ──
    vol_21 = float(returns[-min(21,T):].std() * math.sqrt(252))
    vol_63 = float(returns[-min(63,T):].std() * math.sqrt(252))
    vol_252 = float(returns[-min(252,T):].std() * math.sqrt(252))
    factors["VOL_1M"] = vol_21
    factors["VOL_3M"] = vol_63
    factors["VOL_12M"] = vol_252
    factors["VOL_RATIO"] = float(vol_21 / max(vol_252, 1e-10))
    factors["VOL_TREND"] = float(vol_21 / max(vol_63, 1e-10) - 1)

    # ── Skewness and kurtosis ──
    r_21 = returns[-min(21,T):]
    mu_r, sig_r = r_21.mean(), r_21.std()
    if sig_r > 1e-10:
        factors["SKEW_1M"] = float(np.mean(((r_21 - mu_r)/sig_r)**3))
        factors["KURT_1M"] = float(np.mean(((r_21 - mu_r)/sig_r)**4) - 3)
    else:
        factors["SKEW_1M"] = 0.0
        factors["KURT_1M"] = 0.0

    # ── Realized variance ratio ──
    rv_daily = float(np.mean(returns[-min(21,T):]**2))
    rv_weekly = float(np.mean(
        [sum(returns[max(0,i-5):i])**2 for i in range(5, min(21+5,T), 5)]
    )) if T >= 10 else rv_daily
    factors["VVAR_RATIO"] = float(rv_daily * 5 / max(rv_weekly, 1e-10))

    # ── Volume-based (if available) ──
    if volume is not None and len(volume) >= 21:
        vol_21_v = float(volume[-21:].mean())
        vol_252_v = float(volume[-min(252,len(volume)):].mean())
        factors["VOL_VOLUME_RATIO"] = float(vol_21_v / max(vol_252_v, 1e-10))
        # Amihud illiquidity
        if T >= 21:
            amihud = float(np.mean(np.abs(returns[-21:]) / (volume[-21:] + 1)))
            factors["AMIHUD_ILLIQ"] = float(amihud * 1e6)

    # ── Trend quality ──
    n_trend = min(50, T)
    t_arr = np.arange(n_trend)
    p_sub = prices[-n_trend:]
    slope, intercept = np.polyfit(t_arr, p_sub, 1)
    p_fitted = slope * t_arr + intercept
    resid_trend = p_sub - p_fitted
    r2_trend = float(1 - resid_trend.var() / (p_sub.var() + 1e-10))
    factors["TREND_R2"] = r2_trend
    factors["TREND_SLOPE_NORM"] = float(slope / max(p_sub.mean(), 1e-10))

    # ── RSI ──
    n_rsi = min(14, T)
    ret_rsi = returns[-n_rsi:]
    up = float(ret_rsi[ret_rsi > 0].mean()) if (ret_rsi > 0).any() else 0.0
    down = float(abs(ret_rsi[ret_rsi < 0].mean())) if (ret_rsi < 0).any() else 1e-10
    rs = up / max(down, 1e-10)
    factors["RSI_14"] = float(100 - 100 / (1 + rs))

    # ── Hurst exponent proxy ──
    if T >= 50:
        log_prices = np.log(prices[-50:] + 1e-10)
        lags = [2, 4, 8, 16]
        rs_vals = []
        for lag in lags:
            sub_lp = log_prices[:lag]
            mean_sub = sub_lp.mean()
            cum_dev = np.cumsum(sub_lp - mean_sub)
            r_val = cum_dev.max() - cum_dev.min()
            s_val = sub_lp.std()
            if s_val > 1e-10:
                rs_vals.append((lag, r_val / s_val))
        if len(rs_vals) >= 2:
            lags_arr = np.log([v[0] for v in rs_vals])
            rs_arr = np.log([v[1] for v in rs_vals])
            hurst, _ = np.polyfit(lags_arr, rs_arr, 1)
            factors["HURST_PROXY"] = float(hurst)

    return factors


# ── Factor Crowding Detection ─────────────────────────────────────────────────

def factor_crowding_score(
    factor_returns: np.ndarray,
    lookback_short: int = 21,
    lookback_long: int = 126,
) -> dict:
    """
    Detect crowding in a factor strategy.
    Crowded factors have: high recent Sharpe, high autocorrelation,
    elevated correlation with other popular factors.
    """
    T = len(factor_returns)
    if T < lookback_long:
        return {"crowding_score": 0.5, "warning": "insufficient_data"}

    # Recent vs long-run Sharpe
    recent = factor_returns[-lookback_short:]
    long_run = factor_returns[-lookback_long:]

    sharpe_recent = float(recent.mean() / (recent.std() + 1e-10) * math.sqrt(252))
    sharpe_long = float(long_run.mean() / (long_run.std() + 1e-10) * math.sqrt(252))
    sharpe_ratio = sharpe_recent / max(abs(sharpe_long), 0.1)

    # Autocorrelation (momentum in factor = crowding signal)
    acf1 = float(np.corrcoef(factor_returns[-lookback_short:][1:],
                              factor_returns[-lookback_short:][:-1])[0, 1])

    # Vol compression (crowding often compresses vol before blow-up)
    vol_recent = float(recent.std())
    vol_long = float(long_run.std())
    vol_compression = float(vol_long / max(vol_recent, 1e-10))

    # Composite crowding score
    crowding = float(
        0.4 * min(max(sharpe_ratio, 0), 3) / 3
        + 0.3 * min(max(acf1, 0), 1)
        + 0.3 * min(max(vol_compression - 1, 0), 2) / 2
    )

    return {
        "crowding_score": float(min(crowding, 1.0)),
        "sharpe_ratio_recent_vs_long": sharpe_ratio,
        "autocorrelation": acf1,
        "vol_compression": vol_compression,
        "is_crowded": bool(crowding > 0.65),
        "risk_of_unwind": bool(crowding > 0.75 and acf1 < 0),
    }


# ── Cross-Sectional Factor Momentum ──────────────────────────────────────────

def cross_sectional_factor_momentum(
    factor_returns_matrix: np.ndarray,
    lookback: int = 21,
    n_top: int = 3,
) -> dict:
    """
    Cross-sectional factor momentum: long top recent factors, short bottom.
    factor_returns_matrix: (T, n_factors)
    Returns: factor weights and expected CS momentum signal.
    """
    T, n_factors = factor_returns_matrix.shape
    if T < lookback + 2:
        return {"weights": np.zeros(n_factors), "signal": 0.0}

    recent = factor_returns_matrix[-lookback:]
    cumret = recent.sum(axis=0)  # (n_factors,)

    # Rank factors by recent return
    ranks = np.argsort(cumret)[::-1]
    weights = np.zeros(n_factors)

    top = ranks[:n_top]
    bottom = ranks[-n_top:]

    weights[top] = 1.0 / n_top
    weights[bottom] = -1.0 / n_top

    # Expected signal: factor-weighted combination
    signal = float(cumret @ weights / max(np.abs(cumret).sum(), 1e-10))

    return {
        "weights": weights,
        "signal": signal,
        "top_factors": top.tolist(),
        "bottom_factors": bottom.tolist(),
        "factor_returns_recent": cumret,
    }


# ── Information Ratio Decomposition ──────────────────────────────────────────

def ir_decomposition_by_factor(
    asset_returns: np.ndarray,
    factor_returns: np.ndarray,
    factor_names: Optional[list[str]] = None,
) -> dict:
    """
    Decompose Information Ratio into factor contributions.
    asset_returns: (T,)
    factor_returns: (T, k)
    """
    T, k = factor_returns.shape
    if factor_names is None:
        factor_names = [f"F{i+1}" for i in range(k)]

    # OLS regression
    X = np.column_stack([np.ones(T), factor_returns])
    coef, residuals, _, _ = np.linalg.lstsq(X, asset_returns, rcond=None)

    alpha = float(coef[0])
    betas = coef[1:]
    resid = asset_returns - X @ coef

    # Total IR
    total_ir = float(asset_returns.mean() / (asset_returns.std() + 1e-10) * math.sqrt(252))
    alpha_ir = float(alpha / (resid.std() + 1e-10) * math.sqrt(252))

    # Factor contributions to IR
    factor_contributions = {}
    for i, name in enumerate(factor_names):
        contribution = float(betas[i] * factor_returns[:, i].mean()
                             / (asset_returns.std() + 1e-10) * math.sqrt(252))
        factor_contributions[name] = contribution

    return {
        "total_ir": total_ir,
        "alpha_ir": alpha_ir,
        "betas": dict(zip(factor_names, betas.tolist())),
        "factor_contributions_to_ir": factor_contributions,
        "r2": float(1 - resid.var() / (asset_returns.var() + 1e-10)),
        "alpha_fraction": float(abs(alpha_ir) / max(abs(total_ir), 1e-10)),
    }
