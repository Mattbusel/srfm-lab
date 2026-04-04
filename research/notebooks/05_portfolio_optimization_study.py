"""
05_portfolio_optimization_study.py — Portfolio Optimization Study

Builds correlation matrix of BH activation events across instruments.
Runs 5 portfolio optimization methods and compares on walk-forward backtest.

Methods:
  1. Equal weight
  2. Min variance (on BH-activation Jaccard correlation)
  3. Risk parity
  4. HRP (hierarchical risk parity)
  5. Max Sharpe

Walk-forward: monthly rebalance, 1-year rolling in-sample window.

Outputs: research/outputs/portfolio_optimization.png, portfolio_results.json

Run: python research/notebooks/05_portfolio_optimization_study.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "lib"))

from srfm_core import MinkowskiClassifier, BlackHoleDetector

OUTPUTS = _ROOT / "research" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

ASSETS = {
    "ES":     {"annual_vol": 0.15,  "drift": 0.10,  "cf": 0.001},
    "NQ":     {"annual_vol": 0.20,  "drift": 0.12,  "cf": 0.0012},
    "YM":     {"annual_vol": 0.14,  "drift": 0.09,  "cf": 0.0008},
    "BTC":    {"annual_vol": 0.80,  "drift": 0.30,  "cf": 0.005},
    "ETH":    {"annual_vol": 0.90,  "drift": 0.25,  "cf": 0.007},
    "GC":     {"annual_vol": 0.12,  "drift": 0.05,  "cf": 0.008},
    "CL":     {"annual_vol": 0.35,  "drift": 0.03,  "cf": 0.015},
    "EURUSD": {"annual_vol": 0.07,  "drift": 0.01,  "cf": 0.0005},
}

N_BARS_DAILY = 1260   # ~5 years
CORR_MATRIX  = None   # filled in main()


def generate_correlated_daily_returns(n_days: int = 1260, seed: int = 42) -> pd.DataFrame:
    """
    Generate correlated daily returns for all assets.
    Use a realistic cross-asset correlation structure.
    """
    rng = np.random.default_rng(seed)
    syms = list(ASSETS.keys())
    n    = len(syms)

    # Build correlation matrix (realistic structure)
    # Equity indices correlated ~0.7
    # Crypto correlated ~0.6 with each other, ~0.2 with equities
    # Gold slightly negative with equities
    # FX/Oil lower correlation
    corr = np.eye(n)
    corr_map = {
        ("ES", "NQ"):     0.90,
        ("ES", "YM"):     0.85,
        ("NQ", "YM"):     0.80,
        ("BTC", "ETH"):   0.80,
        ("BTC", "ES"):    0.20,
        ("ETH", "ES"):    0.18,
        ("GC", "ES"):    -0.05,
        ("GC", "BTC"):    0.10,
        ("CL", "ES"):     0.30,
        ("EURUSD", "ES"): 0.15,
    }
    sym_to_idx = {s: i for i, s in enumerate(syms)}
    for (s1, s2), c in corr_map.items():
        if s1 in sym_to_idx and s2 in sym_to_idx:
            i1, i2 = sym_to_idx[s1], sym_to_idx[s2]
            corr[i1, i2] = corr[i2, i1] = c

    # Ensure positive definite
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Cholesky
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        L = np.eye(n)

    # Generate returns
    vols = np.array([ASSETS[s]["annual_vol"] / math.sqrt(252) for s in syms])
    drifts = np.array([ASSETS[s]["drift"] / 252 for s in syms])

    Z = rng.standard_normal((n, n_days))
    Z_corr = L @ Z  # shape (n, n_days)

    rets_matrix = drifts[:, None] + vols[:, None] * Z_corr

    idx = pd.date_range("2019-01-02", periods=n_days, freq="B")
    return pd.DataFrame(rets_matrix.T, index=idx, columns=syms)


def returns_to_prices(rets: pd.DataFrame, start: float = 100.0) -> pd.DataFrame:
    """Convert return DataFrame to price DataFrame."""
    return (1 + rets).cumprod() * start


# ─────────────────────────────────────────────────────────────────────────────
# BH activation correlation matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_bh_activation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Jaccard similarity matrix between BH activation series for each asset.
    """
    syms = prices.columns.tolist()
    n    = len(syms)
    act_series = {}

    for sym in syms:
        cf  = ASSETS.get(sym, {}).get("cf", 0.001)
        mc  = MinkowskiClassifier(cf=cf)
        bh  = BlackHoleDetector(1.5, 1.0, 0.95)
        closes = prices[sym].values
        mc.update(float(closes[0]))
        active = []
        for i in range(1, len(closes)):
            bit = mc.update(float(closes[i]))
            act = bh.update(bit, float(closes[i]), float(closes[i-1]))
            active.append(act)
        act_series[sym] = np.array(active, dtype=bool)

    mat = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            a = act_series[syms[i]]
            b = act_series[syms[j]]
            mn = min(len(a), len(b))
            inter = np.sum(a[:mn] & b[:mn])
            union = np.sum(a[:mn] | b[:mn])
            jac = float(inter / union) if union > 0 else 0.0
            mat[i, j] = jac; mat[j, i] = jac

    return pd.DataFrame(mat, index=syms, columns=syms)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio optimization methods
# ─────────────────────────────────────────────────────────────────────────────

def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def min_variance(cov: np.ndarray, reg: float = 1e-4) -> np.ndarray:
    """Minimum variance portfolio (long-only constrained)."""
    n = cov.shape[0]
    cov_reg = cov + reg * np.eye(n)
    try:
        result = minimize(
            lambda w: w @ cov_reg @ w,
            x0=np.full(n, 1/n),
            method="SLSQP",
            bounds=[(0.01, 0.6)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 500, "ftol": 1e-9},
        )
        w = result.x
        return np.maximum(w, 0) / (np.sum(np.maximum(w, 0)) + 1e-10)
    except Exception:
        return equal_weight(n)


def risk_parity(cov: np.ndarray) -> np.ndarray:
    """Risk parity: each asset contributes equally to portfolio variance."""
    n = cov.shape[0]
    def risk_contribution_diff(w: np.ndarray) -> float:
        w = np.maximum(w, 1e-8)
        port_var = w @ cov @ w
        mrc = cov @ w
        rc  = w * mrc / (port_var + 1e-10)
        target = 1.0 / n
        return float(np.sum((rc - target)**2))

    try:
        result = minimize(
            risk_contribution_diff,
            x0=np.full(n, 1/n),
            method="SLSQP",
            bounds=[(0.01, 0.6)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 1000},
        )
        w = np.maximum(result.x, 0)
        return w / (w.sum() + 1e-10)
    except Exception:
        return equal_weight(n)


def hrp(cov: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """
    Hierarchical Risk Parity (Lopez de Prado).
    Uses Ward linkage on correlation distance matrix.
    """
    n = cov.shape[0]
    if n < 2:
        return equal_weight(n)

    dist = np.sqrt((1.0 - corr) / 2.0)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 1.0)

    # Linkage
    condensed = squareform(dist)
    link = linkage(condensed, method="ward")

    # Build sorted leaf order from dendrogram
    dend = dendrogram(link, no_plot=True)
    order = dend["leaves"]

    # Quasi-diagonal bisection
    weights = np.ones(n)

    def _bisect(items: List[int]):
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left, right = items[:mid], items[mid:]

        def _cluster_var(idxs: List[int]) -> float:
            sub_cov = cov[np.ix_(idxs, idxs)]
            inv_diag = 1.0 / (np.diag(sub_cov) + 1e-10)
            w = inv_diag / inv_diag.sum()
            return float(w @ sub_cov @ w)

        var_l = _cluster_var(left)
        var_r = _cluster_var(right)
        alpha = var_r / (var_l + var_r + 1e-10)
        for i in left:  weights[i] *= alpha
        for i in right: weights[i] *= (1 - alpha)
        _bisect(left)
        _bisect(right)

    _bisect(list(range(n)))
    w = weights / (weights.sum() + 1e-10)
    return w


def max_sharpe(rets_mean: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Maximum Sharpe ratio portfolio (long-only)."""
    n = len(rets_mean)
    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ rets_mean)
        port_vol = float(math.sqrt(max(1e-10, w @ cov @ w)))
        return -(port_ret - rf) / port_vol

    try:
        result = minimize(
            neg_sharpe,
            x0=np.full(n, 1/n),
            method="SLSQP",
            bounds=[(0.01, 0.6)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 500},
        )
        w = np.maximum(result.x, 0)
        return w / (w.sum() + 1e-10)
    except Exception:
        return equal_weight(n)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward backtest
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_backtest(
    rets: pd.DataFrame,
    method_fn,
    train_days: int = 252,
    rebal_days: int = 21,
    starting_equity: float = 1_000_000.0,
    method_name: str = "Unknown",
) -> dict:
    """
    Walk-forward backtest of a portfolio optimization method.

    Parameters
    ----------
    rets         : daily returns DataFrame
    method_fn    : callable(rets_train: pd.DataFrame) -> weights: np.ndarray
    train_days   : in-sample window
    rebal_days   : days between rebalancing

    Returns
    -------
    dict with equity_curve, weights_history, sharpe, cagr, max_drawdown, turnover
    """
    n_assets = rets.shape[1]
    syms     = rets.columns.tolist()
    n_days   = len(rets)

    equity = starting_equity
    equity_curve = [equity]
    weights_history = []
    current_weights = np.full(n_assets, 1.0 / n_assets)
    turnovers = []

    for t in range(0, n_days - rebal_days, rebal_days):
        # Training window
        train_start = max(0, t - train_days)
        train_end   = t
        if train_end - train_start < 20:
            pass
        else:
            rets_train = rets.iloc[train_start:train_end]
            try:
                new_weights = method_fn(rets_train)
                # Compute turnover
                turnover = float(np.sum(np.abs(new_weights - current_weights)))
                turnovers.append(turnover)
                current_weights = new_weights
            except Exception:
                pass

        weights_history.append(current_weights.copy())

        # Apply weights for next rebal_days period
        period_rets = rets.iloc[t:t + rebal_days]
        for day_rets in period_rets.values:
            port_ret = float(current_weights @ day_rets)
            equity  *= (1.0 + port_ret)
            equity_curve.append(equity)

    eq_arr = np.array(equity_curve)
    dates  = rets.index[:len(eq_arr)]
    years  = max(0.01, len(eq_arr) / 252)
    cagr   = float((eq_arr[-1] / starting_equity) ** (1 / years) - 1.0)

    # Daily returns of portfolio
    port_daily_rets = np.diff(eq_arr) / (eq_arr[:-1] + 1e-10)
    sharpe = float(port_daily_rets.mean() / (port_daily_rets.std() + 1e-10) * math.sqrt(252))

    peak = np.maximum.accumulate(eq_arr)
    dd   = (eq_arr - peak) / (peak + 1e-10)
    max_dd = float(dd.min())

    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    return {
        "method":       method_name,
        "equity_curve": eq_arr.tolist(),
        "cagr":         round(cagr, 4),
        "sharpe":       round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "total_return": round(float(eq_arr[-1] / starting_equity - 1.0), 4),
        "avg_turnover": round(avg_turnover, 4),
        "final_equity": round(float(eq_arr[-1]), 2),
    }


def build_method_fn(method_name: str, syms: List[str]):
    """Build a callable that takes rets_train and returns weights."""
    def fn(rets_train: pd.DataFrame) -> np.ndarray:
        n = len(syms)
        if method_name == "equal_weight":
            return equal_weight(n)

        cov = rets_train.cov().values * 252
        cov = np.nan_to_num(cov) + 1e-6 * np.eye(n)

        if method_name == "min_variance":
            return min_variance(cov)
        elif method_name == "risk_parity":
            return risk_parity(cov)
        elif method_name == "hrp":
            corr = rets_train.corr().fillna(0).values
            return hrp(cov, corr)
        elif method_name == "max_sharpe":
            mean_rets = rets_train.mean().values * 252
            return max_sharpe(mean_rets, cov)
        else:
            return equal_weight(n)
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_curves(results: List[dict], ax):
    """Plot equity curves for all methods."""
    COLORS = {"equal_weight": "gray", "min_variance": "steelblue",
              "risk_parity": "orange", "hrp": "green", "max_sharpe": "red"}
    for res in results:
        eq = np.array(res["equity_curve"])
        label = f"{res['method']} (Sh={res['sharpe']:.2f}, DD={res['max_drawdown']:.1%})"
        ax.plot(eq, linewidth=1.0, label=label, color=COLORS.get(res["method"], "purple"))
    ax.set_title("Portfolio Equity Curves — Walk-Forward Backtest", fontsize=10)
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Trading Days")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(
        __import__("matplotlib").ticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M")
    )


def plot_metrics_comparison(results: List[dict], ax):
    """Bar chart comparing Sharpe ratios across methods."""
    methods = [r["method"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    cagrs   = [r["cagr"] * 100 for r in results]
    x = np.arange(len(methods))
    ax.bar(x - 0.2, sharpes, 0.4, label="Sharpe", color="steelblue", alpha=0.8)
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, cagrs, 0.4, label="CAGR%", color="orange", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, fontsize=7)
    ax.set_ylabel("Sharpe Ratio"); ax2.set_ylabel("CAGR (%)")
    ax.set_title("Sharpe & CAGR by Method", fontsize=9)
    ax.legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)


def plot_bh_correlation_heatmap(bh_corr: pd.DataFrame, ax):
    """Heatmap of BH activation Jaccard similarity."""
    im = ax.imshow(bh_corr.values, cmap="YlOrRd", vmin=0, vmax=1)
    syms = bh_corr.columns.tolist()
    ax.set_xticks(range(len(syms))); ax.set_xticklabels(syms, rotation=45, fontsize=7)
    ax.set_yticks(range(len(syms))); ax.set_yticklabels(syms, fontsize=7)
    for i in range(len(syms)):
        for j in range(len(syms)):
            ax.text(j, i, f"{bh_corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("BH Activation Jaccard Similarity", fontsize=9)
    import matplotlib.pyplot as plt
    plt.colorbar(im, ax=ax)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("05_portfolio_optimization_study.py")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_PLOT = True
    except ImportError:
        HAS_PLOT = False

    syms = list(ASSETS.keys())
    print(f"\nGenerating {N_BARS_DAILY} days of correlated returns for {len(syms)} assets...")
    rets  = generate_correlated_daily_returns(N_BARS_DAILY)
    prices = returns_to_prices(rets)

    print("\nComputing BH activation correlation matrix...")
    bh_corr = compute_bh_activation_matrix(prices)
    print(bh_corr.round(3).to_string())

    diversification = 1.0 - float(bh_corr.values[~np.eye(len(syms), dtype=bool)].mean())
    print(f"\nDiversification score (1 - avg off-diag Jaccard): {diversification:.3f}")

    # Walk-forward backtest
    method_names = ["equal_weight", "min_variance", "risk_parity", "hrp", "max_sharpe"]
    print(f"\nRunning walk-forward backtest ({N_BARS_DAILY} days, monthly rebal, 1Y train)...")

    wf_results = []
    for method_name in method_names:
        print(f"  {method_name}...")
        fn = build_method_fn(method_name, syms)
        result = walk_forward_backtest(
            rets, fn,
            train_days=252, rebal_days=21,
            starting_equity=1_000_000.0,
            method_name=method_name,
        )
        wf_results.append(result)
        print(f"    CAGR={result['cagr']:.1%}  Sharpe={result['sharpe']:.2f}  "
              f"MaxDD={result['max_drawdown']:.1%}  Turnover={result['avg_turnover']:.2f}")

    # Best method
    best = max(wf_results, key=lambda r: r["sharpe"])
    print(f"\nBest method by Sharpe: {best['method']} (Sharpe={best['sharpe']:.2f})")

    # Plotting
    if HAS_PLOT:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Portfolio Optimization Study", fontsize=13, fontweight="bold")

        plot_equity_curves(wf_results, axes[0, 0])
        plot_metrics_comparison(wf_results, axes[0, 1])
        plot_bh_correlation_heatmap(bh_corr, axes[1, 0])

        # Drawdown comparison
        ax_dd = axes[1, 1]
        COLORS = {"equal_weight": "gray", "min_variance": "steelblue",
                  "risk_parity": "orange", "hrp": "green", "max_sharpe": "red"}
        for res in wf_results:
            eq = np.array(res["equity_curve"])
            peak = np.maximum.accumulate(eq)
            dd   = (eq - peak) / (peak + 1e-10) * 100
            ax_dd.plot(dd, linewidth=0.8, label=res["method"],
                      color=COLORS.get(res["method"], "purple"), alpha=0.8)
        ax_dd.fill_between(range(len(wf_results[0]["equity_curve"])),
                            [r["max_drawdown"]*100 for r in wf_results[:1]] * len(wf_results[0]["equity_curve"]),
                            0, alpha=0.05, color="red")
        ax_dd.set_title("Drawdown by Method (%)", fontsize=9)
        ax_dd.set_ylabel("Drawdown (%)"); ax_dd.set_xlabel("Days")
        ax_dd.legend(fontsize=7); ax_dd.grid(alpha=0.3)
        ax_dd.axhline(0, color="black", linewidth=0.7)

        plt.tight_layout()
        out = OUTPUTS / "portfolio_optimization.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\nPlot → {out}")

    # Save results
    def _clean(obj):
        if isinstance(obj, (float, np.floating)):
            v = float(obj); return v if math.isfinite(v) else None
        if isinstance(obj, (int, np.integer)): return int(obj)
        if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_clean(v) for v in obj[:100]]  # truncate equity curves
        return obj

    summary_results = []
    for r in wf_results:
        sr = dict(r)
        sr["equity_curve"] = [float(v) for v in r["equity_curve"][::10]]  # subsample
        summary_results.append(sr)

    stats = {
        "diversification_score": float(diversification),
        "bh_activation_correlation": bh_corr.to_dict(),
        "method_results": _clean(summary_results),
        "best_method_by_sharpe": best["method"],
    }
    out_json = OUTPUTS / "portfolio_results.json"
    with open(out_json, "w") as f:
        json.dump(_clean(stats), f, indent=2)
    print(f"Results → {out_json}")


if __name__ == "__main__":
    main()
