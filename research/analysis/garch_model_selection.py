"""
garch_model_selection.py
------------------------
GARCH model selection and forecasting evaluation for LARSA v18 instruments.

Models compared:
  GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1), GARCH(2,1)

Selection criteria:
  AIC, BIC, log-likelihood

Forecasting evaluation:
  Rolling 1-step-ahead vol forecast vs realized vol (Mincer-Zarnowitz regression)

Tests:
  - Residual iid check: Ljung-Box, ARCH-LM
  - Optimal EWMA decay lambda

Outputs:
  garch_model_selection.json -- best model per instrument
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import chi2, norm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUMENTS = ["BTC", "ETH", "SOL", "ES", "NQ", "CL", "GC", "ZB"]
N_BARS      = 6_000
RANDOM_SEED = 271828
OUT_DIR     = Path(__file__).parent

CF_15M = {
    "BTC": 0.0012, "ETH": 0.0015, "SOL": 0.0020,
    "ES":  0.0003, "NQ":  0.0004, "CL":  0.0015,
    "GC":  0.0008, "ZB":  0.0005,
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_returns(ticker: str, n: int = N_BARS, seed: int = RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed + abs(hash(ticker)) % 9999)
    is_crypto  = ticker in ("BTC", "ETH", "SOL")
    base_vol   = 0.0025 if is_crypto else 0.0008
    omega_true = 1e-6 if is_crypto else 3e-7
    alpha_true = 0.08 if is_crypto else 0.06
    beta_true  = 0.88 if is_crypto else 0.90
    mu         = 3e-5 if is_crypto else 1e-5
    # Cap variance to prevent numerical explosion in extreme random paths
    var_cap    = (base_vol * 20) ** 2

    ret  = np.zeros(n)
    var  = np.full(n, omega_true / (1 - alpha_true - beta_true))
    for i in range(1, n):
        var[i] = omega_true + alpha_true * ret[i - 1] ** 2 + beta_true * var[i - 1]
        # Asymmetry for equity: negative returns increase vol more
        if not is_crypto and ret[i - 1] < 0:
            var[i] += 0.04 * ret[i - 1] ** 2
        var[i] = np.clip(var[i], 1e-12, var_cap)
        ret[i] = mu + np.sqrt(var[i]) * rng.standard_t(df=6)
    return ret


# ---------------------------------------------------------------------------
# GARCH log-likelihood
# ---------------------------------------------------------------------------

def garch11_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    """Negative log-likelihood for GARCH(1,1)."""
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    n   = len(returns)
    var = np.zeros(n)
    var[0] = returns.var()
    for i in range(1, n):
        var[i] = omega + alpha * returns[i - 1] ** 2 + beta * var[i - 1]
        if var[i] <= 0:
            return 1e10
    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + returns ** 2 / var)
    return -ll


def garch21_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    """Negative log-likelihood for GARCH(2,1)."""
    omega, alpha1, alpha2, beta = params
    if omega <= 0 or any(p < 0 for p in [alpha1, alpha2, beta]) or alpha1 + alpha2 + beta >= 1:
        return 1e10
    n   = len(returns)
    var = np.zeros(n)
    var[0] = var[1] = returns.var()
    for i in range(2, n):
        var[i] = omega + alpha1 * returns[i - 1] ** 2 + alpha2 * returns[i - 2] ** 2 + beta * var[i - 1]
        if var[i] <= 0:
            return 1e10
    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + returns ** 2 / var)
    return -ll


def egarch11_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    """Negative log-likelihood for EGARCH(1,1)."""
    omega, alpha, gamma, beta = params
    if abs(beta) >= 1:
        return 1e10
    n   = len(returns)
    lnv = np.zeros(n)
    lnv[0] = np.log(returns.var() + 1e-12)
    for i in range(1, n):
        std_t1 = returns[i - 1] / np.exp(0.5 * lnv[i - 1])
        lnv[i] = omega + beta * lnv[i - 1] + alpha * (abs(std_t1) - np.sqrt(2 / np.pi)) + gamma * std_t1
    var = np.exp(lnv)
    var = np.maximum(var, 1e-12)
    ll  = -0.5 * np.sum(np.log(2 * np.pi * var) + returns ** 2 / var)
    return -ll


def gjr_garch11_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    """Negative log-likelihood for GJR-GARCH(1,1)."""
    omega, alpha, gamma, beta = params
    if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0:
        return 1e10
    if alpha + 0.5 * gamma + beta >= 1:
        return 1e10
    n   = len(returns)
    var = np.zeros(n)
    var[0] = returns.var()
    for i in range(1, n):
        ind = 1.0 if returns[i - 1] < 0 else 0.0
        var[i] = omega + (alpha + gamma * ind) * returns[i - 1] ** 2 + beta * var[i - 1]
        if var[i] <= 0:
            return 1e10
    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + returns ** 2 / var)
    return -ll


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_garch11(returns: np.ndarray) -> Dict:
    """Fits GARCH(1,1) by MLE."""
    sigma2 = returns.var()
    x0     = [sigma2 * 0.05, 0.08, 0.88]
    bounds = [(1e-8, None), (0.001, 0.3), (0.5, 0.999)]
    res    = minimize(garch11_loglik, x0, args=(returns,), method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-9})
    if not res.success:
        return _failed_model("GARCH(1,1)")
    omega, alpha, beta = res.x
    n_params = 3
    return _model_result("GARCH(1,1)", res.fun, n_params, len(returns), {"omega": omega, "alpha": alpha, "beta": beta})


def fit_garch21(returns: np.ndarray) -> Dict:
    """Fits GARCH(2,1) by MLE."""
    sigma2 = returns.var()
    x0     = [sigma2 * 0.05, 0.06, 0.02, 0.88]
    bounds = [(1e-8, None), (0.001, 0.3), (0.001, 0.2), (0.5, 0.999)]
    res    = minimize(garch21_loglik, x0, args=(returns,), method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500})
    if not res.success:
        return _failed_model("GARCH(2,1)")
    omega, alpha1, alpha2, beta = res.x
    n_params = 4
    return _model_result("GARCH(2,1)", res.fun, n_params, len(returns),
                         {"omega": omega, "alpha1": alpha1, "alpha2": alpha2, "beta": beta})


def fit_egarch11(returns: np.ndarray) -> Dict:
    """Fits EGARCH(1,1) by MLE."""
    x0     = [-0.5, 0.1, -0.05, 0.9]
    bounds = [(-5, 0), (-0.5, 0.5), (-0.5, 0.5), (-0.999, 0.999)]
    res    = minimize(egarch11_loglik, x0, args=(returns,), method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500})
    if not res.success:
        return _failed_model("EGARCH(1,1)")
    omega, alpha, gamma, beta = res.x
    n_params = 4
    return _model_result("EGARCH(1,1)", res.fun, n_params, len(returns),
                         {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta})


def fit_gjr_garch11(returns: np.ndarray) -> Dict:
    """Fits GJR-GARCH(1,1) by MLE."""
    sigma2 = returns.var()
    x0     = [sigma2 * 0.05, 0.06, 0.04, 0.88]
    bounds = [(1e-8, None), (0.001, 0.3), (0.001, 0.3), (0.5, 0.999)]
    res    = minimize(gjr_garch11_loglik, x0, args=(returns,), method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500})
    if not res.success:
        return _failed_model("GJR-GARCH(1,1)")
    omega, alpha, gamma, beta = res.x
    n_params = 4
    return _model_result("GJR-GARCH(1,1)", res.fun, n_params, len(returns),
                         {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta})


def _failed_model(name: str) -> Dict:
    return {"model": name, "success": False, "loglik": np.nan, "aic": np.nan, "bic": np.nan, "params": {}}


def _model_result(name: str, neg_ll: float, k: int, n: int, params: Dict) -> Dict:
    ll  = -neg_ll
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    return {
        "model":   name,
        "success": True,
        "loglik":  float(ll),
        "aic":     float(aic),
        "bic":     float(bic),
        "params":  {kk: float(vv) for kk, vv in params.items()},
    }


# ---------------------------------------------------------------------------
# Rolling forecast evaluation
# ---------------------------------------------------------------------------

def rolling_garch11_forecast(
    returns: np.ndarray,
    window: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling 1-step-ahead GARCH(1,1) vol forecast vs realized vol.
    Returns (forecast_vol, realized_vol) arrays (same length).
    """
    n         = len(returns)
    forecast  = np.full(n, np.nan)
    realized  = np.full(n, np.nan)

    for t in range(window, n - 1):
        train = returns[t - window: t]

        # Quick fit
        sigma2 = train.var()
        x0     = [sigma2 * 0.05, 0.08, 0.88]
        bounds = [(1e-8, None), (0.001, 0.3), (0.5, 0.999)]
        try:
            res = minimize(garch11_loglik, x0, args=(train,), method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 200, "ftol": 1e-7})
            if res.success:
                omega_f, alpha_f, beta_f = res.x
                # 1-step forecast
                var_t = sigma2
                for r in train:
                    var_t = omega_f + alpha_f * r ** 2 + beta_f * var_t
                forecast[t + 1] = np.sqrt(max(var_t, 1e-12))
        except Exception:
            pass

        # Realized vol: next-bar absolute return as proxy
        realized[t + 1] = abs(returns[t + 1])

    return forecast, realized


def mincer_zarnowitz_r2(forecast: np.ndarray, realized: np.ndarray) -> float:
    """
    R^2 from OLS regression: realized ~ a + b * forecast.
    MZ R^2 measures how much variance the forecast explains.
    """
    mask = np.isfinite(forecast) & np.isfinite(realized)
    if mask.sum() < 20:
        return np.nan
    X    = np.column_stack([np.ones(mask.sum()), forecast[mask]])
    y    = realized[mask]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------

def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> Tuple[float, float]:
    """Ljung-Box Q-statistic for residual autocorrelation. Returns (Q, p-value)."""
    n   = len(residuals)
    r   = residuals - residuals.mean()
    rho = np.array([
        np.sum(r[k:] * r[:-k]) / np.sum(r ** 2) if k > 0 else 1.0
        for k in range(lags + 1)
    ])
    q = n * (n + 2) * np.sum(rho[1:] ** 2 / (n - np.arange(1, lags + 1)))
    p_val = 1.0 - chi2.cdf(q, df=lags)
    return float(q), float(p_val)


def arch_lm_test(residuals: np.ndarray, lags: int = 5) -> Tuple[float, float]:
    """ARCH-LM test for residual heteroscedasticity. Returns (LM, p-value)."""
    n      = len(residuals)
    sq_res = residuals ** 2
    X = np.column_stack([np.ones(n - lags)] + [sq_res[lags - k: n - k] for k in range(lags)])
    y = sq_res[lags:]
    beta   = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat  = X @ beta
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2     = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
    lm     = (n - lags) * r2
    p_val  = 1.0 - chi2.cdf(lm, df=lags)
    return float(lm), float(p_val)


def compute_garch_residuals(returns: np.ndarray, params: Dict) -> np.ndarray:
    """Compute standardised residuals e_t = r_t / sigma_t."""
    n     = len(returns)
    var   = np.full(n, returns.var())
    omega = params.get("omega", 1e-6)
    alpha = params.get("alpha", 0.08)
    beta  = params.get("beta", 0.88)
    for i in range(1, n):
        var[i] = omega + alpha * returns[i - 1] ** 2 + beta * var[i - 1]
        var[i] = max(var[i], 1e-12)
    return returns / np.sqrt(var)


# ---------------------------------------------------------------------------
# EWMA calibration
# ---------------------------------------------------------------------------

def fit_ewma_lambda(returns: np.ndarray, lambda_grid: np.ndarray | None = None) -> Dict:
    """
    Finds optimal EWMA decay lambda by minimising MSE vs realized variance.
    Realized variance proxy: next-bar squared return.
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0.80, 0.99, 20)

    n      = len(returns)
    best_l = 0.94
    best_mse = np.inf

    for lam in lambda_grid:
        var  = np.zeros(n)
        var[0] = returns[0] ** 2
        for i in range(1, n):
            var[i] = lam * var[i - 1] + (1 - lam) * returns[i - 1] ** 2

        # MSE vs next-bar realized
        sq_ret = returns ** 2
        mse    = np.mean((var[:-1] - sq_ret[1:]) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_l   = lam

    return {"optimal_lambda": float(best_l), "mse": float(best_mse)}


# ---------------------------------------------------------------------------
# Full instrument analysis
# ---------------------------------------------------------------------------

def analyze_instrument(ticker: str, returns: np.ndarray) -> Dict:
    """Fits all four GARCH models and runs diagnostics for one instrument."""
    print(f"  [{ticker}] Fitting models ...")
    models = [
        fit_garch11(returns),
        fit_egarch11(returns),
        fit_gjr_garch11(returns),
        fit_garch21(returns),
    ]

    # Best by BIC
    valid   = [m for m in models if m["success"] and np.isfinite(m["bic"])]
    best_m  = min(valid, key=lambda m: m["bic"]) if valid else models[0]

    # Residual diagnostics for best model
    if best_m["success"] and "alpha" in best_m["params"]:
        std_res = compute_garch_residuals(returns, best_m["params"])
        lb_q, lb_p = ljung_box_test(std_res)
        arch_lm, arch_p = arch_lm_test(std_res)
    else:
        lb_q = lb_p = arch_lm = arch_p = np.nan

    # EWMA calibration
    ewma = fit_ewma_lambda(returns)

    # Rolling forecast (lighter: skip for speed)
    forecast, realized = rolling_garch11_forecast(returns, window=300)
    mz_r2 = mincer_zarnowitz_r2(forecast, realized)

    return {
        "ticker":       ticker,
        "models":       models,
        "best_model":   best_m["model"],
        "best_params":  best_m["params"],
        "lb_q":         float(lb_q) if np.isfinite(lb_q) else None,
        "lb_p":         float(lb_p) if np.isfinite(lb_p) else None,
        "arch_lm":      float(arch_lm) if np.isfinite(arch_lm) else None,
        "arch_p":       float(arch_p) if np.isfinite(arch_p) else None,
        "residuals_iid": bool(lb_p > 0.05 and arch_p > 0.05) if (np.isfinite(lb_p) and np.isfinite(arch_p)) else None,
        "ewma_lambda":  ewma["optimal_lambda"],
        "mz_r2":        float(mz_r2) if np.isfinite(mz_r2) else None,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: Dict[str, Dict],
    ax: plt.Axes,
) -> None:
    """Grouped bar: AIC and BIC per model per instrument (best model highlighted)."""
    model_names = ["GARCH(1,1)", "EGARCH(1,1)", "GJR-GARCH(1,1)", "GARCH(2,1)"]
    colors      = ["#3b82f6", "#22c55e", "#f97316", "#a855f7"]

    tickers = list(results.keys())
    x = np.arange(len(tickers))
    w = 0.8 / len(model_names)

    for mi, mname in enumerate(model_names):
        bic_vals = []
        for ticker in tickers:
            ms = results[ticker]["models"]
            m  = next((mm for mm in ms if mm["model"] == mname), None)
            bic = m["bic"] if (m and np.isfinite(m.get("bic", np.nan))) else 0
            bic_vals.append(bic)
        ax.bar(x + mi * w, bic_vals, w, label=mname, color=colors[mi], alpha=0.8)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(tickers, rotation=20)
    ax.set_title("GARCH Model BIC Comparison", fontsize=9)
    ax.set_ylabel("BIC")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


def plot_mz_r2(results: Dict[str, Dict], ax: plt.Axes) -> None:
    tickers = list(results.keys())
    mz_vals = [results[t].get("mz_r2") or 0 for t in tickers]
    colors  = ["#22c55e" if v > 0.05 else "#ef4444" for v in mz_vals]
    ax.bar(tickers, mz_vals, color=colors, alpha=0.85)
    ax.axhline(0.05, color="gold", linestyle="--", linewidth=1, label="R^2=0.05 threshold")
    ax.set_title("Mincer-Zarnowitz R^2: Rolling GARCH(1,1) Forecast Quality", fontsize=9)
    ax.set_ylabel("MZ R^2")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (t, v) in enumerate(zip(tickers, mz_vals)):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=7)


def plot_ewma_lambda(results: Dict[str, Dict], ax: plt.Axes) -> None:
    tickers  = list(results.keys())
    lambdas  = [results[t].get("ewma_lambda", 0.94) for t in tickers]
    ax.bar(tickers, lambdas, color="#60a5fa", alpha=0.85)
    ax.axhline(0.94, color="orange", linestyle="--", linewidth=1, label="RiskMetrics default=0.94")
    ax.set_ylim(0.75, 1.0)
    ax.set_title("Optimal EWMA Lambda per Instrument", fontsize=9)
    ax.set_ylabel("Lambda")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (t, v) in enumerate(zip(tickers, lambdas)):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=7)


def plot_diagnostic_summary(results: Dict[str, Dict], ax: plt.Axes) -> None:
    """Table-style bar: LB p-value and ARCH-LM p-value per instrument."""
    tickers = list(results.keys())
    lb_ps   = [results[t].get("lb_p") or 0 for t in tickers]
    arch_ps = [results[t].get("arch_p") or 0 for t in tickers]

    x = np.arange(len(tickers))
    w = 0.35
    ax.bar(x - w / 2, lb_ps,   w, label="Ljung-Box p",  color="#34d399", alpha=0.85)
    ax.bar(x + w / 2, arch_ps, w, label="ARCH-LM p",    color="#f472b6", alpha=0.85)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p=0.05")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=20)
    ax.set_title("Residual Diagnostics (p-value > 0.05 = iid)", fontsize=9)
    ax.set_ylabel("p-value")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


def plot_best_model_params(results: Dict[str, Dict], ax: plt.Axes) -> None:
    """Scatter: alpha vs beta per instrument for best model."""
    tickers = list(results.keys())
    alphas  = []
    betas   = []
    for t in tickers:
        p = results[t].get("best_params", {})
        alphas.append(p.get("alpha", np.nan))
        betas.append(p.get("beta", np.nan))

    is_crypto = [t in ("BTC", "ETH", "SOL") for t in tickers]
    colors    = ["#f97316" if c else "#3b82f6" for c in is_crypto]

    for i, t in enumerate(tickers):
        if np.isfinite(alphas[i]) and np.isfinite(betas[i]):
            ax.scatter(alphas[i], betas[i], color=colors[i], s=100, alpha=0.85, zorder=5)
            ax.annotate(t, (alphas[i], betas[i]), textcoords="offset points",
                        xytext=(5, 3), fontsize=8, color="white")

    # Persistence boundary
    ab_vals = np.linspace(0, 0.15, 100)
    ax.plot(ab_vals, 1 - ab_vals, "r--", linewidth=1, label="alpha+beta=1 (unit root)", alpha=0.6)
    ax.set_title("GARCH Alpha vs Beta (best model per instrument)", fontsize=9)
    ax.set_xlabel("Alpha (ARCH term)")
    ax.set_ylabel("Beta (GARCH term)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.20)
    ax.set_ylim(0.70, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[GARCH MODEL SELECTION] Starting ...")
    results: Dict[str, Dict] = {}
    returns_data: Dict[str, np.ndarray] = {}

    for ticker in INSTRUMENTS:
        ret = generate_returns(ticker)
        returns_data[ticker] = ret
        results[ticker] = analyze_instrument(ticker, ret)

    # Save JSON
    json_out: Dict = {}
    for ticker, res in results.items():
        json_out[ticker] = {
            k: v for k, v in res.items()
            if k not in ("models",)
        }
        json_out[ticker]["model_comparison"] = [
            {k2: (float(v2) if isinstance(v2, (float, np.floating)) else v2)
             for k2, v2 in m.items() if k2 != "params"}
            for m in res["models"]
        ]

    json_path = OUT_DIR / "garch_model_selection.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"[GARCH MODEL SELECTION] Saved {json_path}")

    # Build figure
    print("[GARCH MODEL SELECTION] Building charts ...")
    fig = plt.figure(figsize=(18, 22), facecolor="#0d1117")
    fig.suptitle("GARCH Model Selection -- LARSA v18", fontsize=14, color="white", y=0.99)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#161b22")
    plot_model_comparison(results, ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#161b22")
    plot_mz_r2(results, ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#161b22")
    plot_ewma_lambda(results, ax3)

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#161b22")
    plot_diagnostic_summary(results, ax4)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#161b22")
    plot_best_model_params(results, ax5)

    for ax in fig.get_axes():
        ax.tick_params(colors="white")
        for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            item.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    fig_path = OUT_DIR / "garch_model_charts.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[GARCH MODEL SELECTION] Saved {fig_path}")
    print("[GARCH MODEL SELECTION] Done.")


if __name__ == "__main__":
    main()
