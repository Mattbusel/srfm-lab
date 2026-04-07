"""
ml_signal_diagnostic.py
------------------------
Diagnostic analysis of the ML signal module in LARSA v18.

Evaluates whether the SGD logistic predictor adds value beyond pure momentum.

Analysis:
  - Rolling 30-day IC of ML signal vs 1-day forward return per instrument
  - Comparison: ML signal IC vs 5-day momentum IC
  - Feature importance: SGD weight coefficient analysis
  - Calibration: reliability diagram (predicted probability vs actual win rate)
  - Decay analysis: does ML signal IC decay faster than BH_MASS?

Outputs:
  ml_signal_diagnostic.html
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUMENTS  = ["BTC", "ETH", "SOL", "ES", "NQ", "CL", "GC", "ZB"]
N_BARS       = 8_000
RANDOM_SEED  = 577215
OUT_DIR      = Path(__file__).parent

# 1-day forward return = 96 x 15-min bars
FWD_1D_BARS  = 96
FWD_1H_BARS  = 4

# Rolling IC window: 30 days = 30 * 96 bars
ROLLING_IC_WINDOW = 30 * 96

FEATURE_NAMES = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5", "garch_vol"]

CF_15M = {
    "BTC": 0.0012, "ETH": 0.0015, "SOL": 0.0020,
    "ES":  0.0003, "NQ":  0.0004, "CL":  0.0015,
    "GC":  0.0008, "ZB":  0.0005,
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_ohlcv(ticker: str, n: int = N_BARS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + abs(hash(ticker)) % 9999)
    is_crypto = ticker in ("BTC", "ETH", "SOL")
    base_vol  = 0.0025 if is_crypto else 0.0008
    drift     = 5e-5   if is_crypto else 2e-5

    regime = np.zeros(n, dtype=int)
    for i in range(1, n):
        regime[i] = (1 - regime[i - 1]) if rng.random() < (0.03 if regime[i - 1] == 0 else 0.07) else regime[i - 1]

    vol = np.where(regime == 1, base_vol * 3, base_vol)
    sh  = rng.standard_t(df=5, size=n) * vol
    ret = np.zeros(n)
    for i in range(1, n):
        ret[i] = drift - 0.04 * ret[i - 1] + sh[i]

    sp    = {"BTC": 42000, "ETH": 2500, "SOL": 100, "ES": 4500,
             "NQ": 15000, "CL": 75, "GC": 1950, "ZB": 120}.get(ticker, 100)
    close = sp * np.exp(np.cumsum(ret))
    hl    = np.abs(ret) * close * rng.uniform(0.5, 1.5, n)
    high  = close + hl * rng.uniform(0.3, 0.7, n)
    low   = close - hl * rng.uniform(0.3, 0.7, n)
    open_ = close * np.exp(rng.normal(0, base_vol * 0.3, n))
    vb    = (10000 * rng.lognormal(0, 0.5, n)).astype(int)

    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vb}, index=ts)


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_garch_vol(returns: np.ndarray, alpha: float = 0.08, beta: float = 0.88) -> np.ndarray:
    n = len(returns)
    var = np.zeros(n)
    omega = max(returns.var() * (1 - alpha - beta), 1e-12)
    v = returns.var()
    for i in range(1, n):
        v = omega + alpha * returns[i - 1] ** 2 + beta * v
        var[i] = max(v, 1e-12)
    return np.sqrt(var)


def compute_ml_signal_with_weights(
    returns: np.ndarray,
    garch_vol: np.ndarray,
    lr: float = 0.005,
    train_start: int = 200,
    weight_record_interval: int = 100,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Online SGD logistic predictor.
    Returns (signal_array, final_weights, weight_history).
    signal = probability(up) - 0.5, centred.
    """
    n        = len(returns)
    n_feat   = 6
    weights  = np.zeros(n_feat)
    signal   = np.zeros(n)
    weight_history: List[np.ndarray] = []

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    for i in range(train_start, n - 1):
        feat = np.array([
            returns[i - 1] if i >= 1 else 0.0,
            returns[i - 2] if i >= 2 else 0.0,
            returns[i - 3] if i >= 3 else 0.0,
            returns[i - 4] if i >= 4 else 0.0,
            returns[i - 5] if i >= 5 else 0.0,
            garch_vol[i],
        ])
        prob      = sigmoid(np.dot(weights, feat))
        signal[i] = prob - 0.5

        label = 1.0 if returns[i + 1] > 0 else 0.0
        grad  = (prob - label) * feat

        # L2 regularisation
        weights = weights * (1 - lr * 0.01) - lr * grad

        if (i - train_start) % weight_record_interval == 0:
            weight_history.append(weights.copy())

    return signal, weights, weight_history


def compute_momentum_signal(returns: np.ndarray, lookback: int = 20) -> np.ndarray:
    """5-day (20-bar) momentum signal: cumulative return over lookback bars."""
    n   = len(returns)
    mom = np.zeros(n)
    for i in range(lookback, n):
        mom[i] = np.sum(returns[i - lookback: i])
    return mom


def compute_bh_mass_signal(close: np.ndarray, cf: float) -> np.ndarray:
    n, mass, ctl = len(close), 0.0, 0
    out = np.zeros(n)
    for i in range(1, n):
        b = abs(close[i] - close[i - 1]) / (close[i - 1] + 1e-9) / (cf + 1e-9)
        if b < 1.0:
            ctl += 1
            mass = mass * 0.97 + 0.03 * min(2.0, 1.0 + ctl * 0.1)
        else:
            ctl  = 0
            mass *= 0.95
        out[i] = mass
    return out


# ---------------------------------------------------------------------------
# Rolling IC computation
# ---------------------------------------------------------------------------

def rolling_ic(
    signal: np.ndarray,
    fwd_ret: np.ndarray,
    window: int = ROLLING_IC_WINDOW,
) -> np.ndarray:
    """Rolling Spearman IC between signal and forward return."""
    n   = len(signal)
    ic  = np.full(n, np.nan)
    for i in range(window, n):
        s = signal[i - window: i]
        f = fwd_ret[i - window: i]
        mask = np.isfinite(s) & np.isfinite(f)
        if mask.sum() < 30:
            continue
        r, _ = spearmanr(s[mask], f[mask])
        ic[i] = r
    return ic


def compute_ic_decay_curve(signal: np.ndarray, returns: np.ndarray, max_lag: int = 20) -> np.ndarray:
    curve = np.zeros(max_lag)
    n     = len(signal)
    for h in range(1, max_lag + 1):
        fwd = np.zeros(n)
        fwd[:-h] = returns[h:]
        fwd[-h:]  = np.nan
        mask = np.isfinite(signal) & np.isfinite(fwd)
        if mask.sum() < 30:
            curve[h - 1] = np.nan
            continue
        r, _ = spearmanr(signal[mask], fwd[mask])
        curve[h - 1] = r
    return curve


def fit_exponential_decay(ic_curve: np.ndarray) -> Tuple[float, float, float]:
    lags  = np.arange(1, len(ic_curve) + 1, dtype=float)
    valid = np.isfinite(ic_curve)
    if valid.sum() < 3:
        return 0.0, 1.0, np.log(2)
    try:
        popt, _ = curve_fit(
            lambda h, ic0, lam: ic0 * np.exp(-lam * h),
            lags[valid], ic_curve[valid],
            p0=[ic_curve[valid][0], 0.1],
            maxfev=3000,
            bounds=([-1, 0.001], [1, 10]),
        )
        ic0, lam = popt
        return float(ic0), float(lam), float(np.log(2) / lam)
    except Exception:
        return 0.0, 0.5, np.log(2) * 2.0


# ---------------------------------------------------------------------------
# Calibration: reliability diagram
# ---------------------------------------------------------------------------

def reliability_diagram_data(
    ml_signal: np.ndarray,
    fwd_ret: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bins ml_signal (probability predictions) and computes actual win rate per bin.
    ml_signal is centred (signal + 0.5 = probability).
    Returns (bin_centers, actual_win_rate, bin_counts).
    """
    prob = ml_signal + 0.5
    prob = np.clip(prob, 0.0, 1.0)

    mask = np.isfinite(prob) & np.isfinite(fwd_ret)
    p    = prob[mask]
    f    = fwd_ret[mask]
    actual_up = (f > 0).astype(float)

    bins     = np.linspace(0, 1, n_bins + 1)
    centers  = 0.5 * (bins[:-1] + bins[1:])
    win_rates = np.full(n_bins, np.nan)
    counts   = np.zeros(n_bins, dtype=int)

    for bi in range(n_bins):
        mask_b = (p >= bins[bi]) & (p < bins[bi + 1])
        if mask_b.sum() >= 5:
            win_rates[bi] = actual_up[mask_b].mean()
            counts[bi]    = int(mask_b.sum())

    return centers, win_rates, counts


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def average_weight_trajectory(weight_history: List[np.ndarray]) -> np.ndarray:
    """Returns time-averaged feature weights from SGD training."""
    if not weight_history:
        return np.zeros(6)
    return np.mean(np.vstack(weight_history), axis=0)


# ---------------------------------------------------------------------------
# Full instrument analysis
# ---------------------------------------------------------------------------

def analyze_instrument(ticker: str, df: pd.DataFrame) -> Dict:
    close   = df["close"].values
    log_ret = np.log(close / np.roll(close, 1))
    log_ret[0] = 0.0

    garch_v = compute_garch_vol(log_ret)
    ml_sig, final_weights, weight_hist = compute_ml_signal_with_weights(log_ret, garch_v)
    mom_sig = compute_momentum_signal(log_ret, lookback=20)
    bh_mass = compute_bh_mass_signal(close, CF_15M.get(ticker, 0.001))

    # 1-day forward return
    fwd_1d = np.log(np.roll(close, -FWD_1D_BARS) / close)
    fwd_1d[-FWD_1D_BARS:] = np.nan

    # 1h forward return
    fwd_1h = np.log(np.roll(close, -FWD_1H_BARS) / close)
    fwd_1h[-FWD_1H_BARS:] = np.nan

    # Rolling IC
    ic_ml  = rolling_ic(ml_sig,  fwd_1d, window=min(ROLLING_IC_WINDOW, N_BARS // 4))
    ic_mom = rolling_ic(mom_sig, fwd_1d, window=min(ROLLING_IC_WINDOW, N_BARS // 4))

    # Overall IC
    mask = np.isfinite(ml_sig) & np.isfinite(fwd_1d)
    ic_ml_overall  = float(spearmanr(ml_sig[mask],  fwd_1d[mask])[0]) if mask.sum() > 30 else np.nan
    mask2 = np.isfinite(mom_sig) & np.isfinite(fwd_1d)
    ic_mom_overall = float(spearmanr(mom_sig[mask2], fwd_1d[mask2])[0]) if mask2.sum() > 30 else np.nan

    # IC decay curves
    ml_decay_curve  = compute_ic_decay_curve(ml_sig,  log_ret, max_lag=20)
    bh_decay_curve  = compute_ic_decay_curve(bh_mass, log_ret, max_lag=20)
    mom_decay_curve = compute_ic_decay_curve(mom_sig, log_ret, max_lag=20)

    _, ml_lam,  ml_hl  = fit_exponential_decay(ml_decay_curve)
    _, bh_lam,  bh_hl  = fit_exponential_decay(bh_decay_curve)
    _, mom_lam, mom_hl = fit_exponential_decay(mom_decay_curve)

    # Reliability diagram
    bin_centers, win_rates, bin_counts = reliability_diagram_data(ml_sig, fwd_1h)

    # Feature importance
    avg_weights = average_weight_trajectory(weight_hist)
    feat_importance = {name: float(w) for name, w in zip(FEATURE_NAMES, avg_weights)}

    # Brier score (calibration quality)
    prob_pred = np.clip(ml_sig + 0.5, 0.01, 0.99)
    actual_up = (fwd_1h > 0).astype(float)
    mask_bs   = np.isfinite(prob_pred) & np.isfinite(actual_up)
    brier     = float(np.mean((prob_pred[mask_bs] - actual_up[mask_bs]) ** 2)) if mask_bs.sum() > 0 else np.nan

    return {
        "ticker":           ticker,
        "ic_ml_overall":    ic_ml_overall,
        "ic_mom_overall":   ic_mom_overall,
        "ml_half_life":     ml_hl,
        "bh_half_life":     bh_hl,
        "mom_half_life":    mom_hl,
        "brier_score":      brier,
        "feat_importance":  feat_importance,
        "final_weights":    {n: float(w) for n, w in zip(FEATURE_NAMES, final_weights)},
        "_ic_ml":           ic_ml,
        "_ic_mom":          ic_mom,
        "_ml_sig":          ml_sig,
        "_bh_mass":         bh_mass,
        "_ml_decay":        ml_decay_curve,
        "_bh_decay":        bh_decay_curve,
        "_mom_decay":       mom_decay_curve,
        "_bin_centers":     bin_centers,
        "_win_rates":       win_rates,
        "_bin_counts":      bin_counts,
        "_close":           close,
        "_log_ret":         log_ret,
        "_fwd_1d":          fwd_1d,
        "_fwd_1h":          fwd_1h,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rolling_ic_comparison(
    results: Dict[str, Dict],
    fig: plt.Figure,
    gs_row,
) -> None:
    """Rolling IC of ML signal vs momentum for BTC and ES side by side."""
    for ci, ticker in enumerate(["BTC", "ES"]):
        ax = fig.add_subplot(gs_row[ci])
        ax.set_facecolor("#161b22")
        res    = results.get(ticker, {})
        ic_ml  = res.get("_ic_ml",  np.array([]))
        ic_mom = res.get("_ic_mom", np.array([]))
        n      = min(len(ic_ml), len(ic_mom))
        if n == 0:
            continue
        idx = np.arange(n)
        ax.plot(idx, ic_ml[:n],  color="#f97316", linewidth=0.8, label="ML Signal IC")
        ax.plot(idx, ic_mom[:n], color="#3b82f6", linewidth=0.8, label="Momentum IC")
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_title(f"{ticker}: Rolling IC (ML vs Momentum)", fontsize=9)
        ax.set_xlabel("Bar")
        ax.set_ylabel("IC (Spearman)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)


def plot_ic_overall_comparison(results: Dict[str, Dict], ax: plt.Axes) -> None:
    """Bar chart: overall IC (ML vs momentum) per instrument."""
    tickers  = list(results.keys())
    ic_ml    = [results[t].get("ic_ml_overall",  0) or 0 for t in tickers]
    ic_mom   = [results[t].get("ic_mom_overall", 0) or 0 for t in tickers]

    x = np.arange(len(tickers))
    w = 0.35
    ax.bar(x - w / 2, ic_ml,  w, label="ML Signal",    color="#f97316", alpha=0.85)
    ax.bar(x + w / 2, ic_mom, w, label="Momentum(20b)", color="#3b82f6", alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=20)
    ax.set_title("Overall IC: ML Signal vs Momentum (1-day fwd return)", fontsize=9)
    ax.set_ylabel("IC (Spearman)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


def plot_ic_decay_comparison(results: Dict[str, Dict], ax: plt.Axes) -> None:
    """IC decay curves: ML vs BH_MASS vs Momentum (averaged across instruments)."""
    lags = np.arange(1, 21)
    ml_curves  = []
    bh_curves  = []
    mom_curves = []
    for res in results.values():
        if "_ml_decay" in res:
            ml_curves.append(np.array(res["_ml_decay"]))
        if "_bh_decay" in res:
            bh_curves.append(np.array(res["_bh_decay"]))
        if "_mom_decay" in res:
            mom_curves.append(np.array(res["_mom_decay"]))

    def avg(curves):
        if not curves:
            return np.zeros(20)
        return np.nanmean(np.vstack(curves), axis=0)

    ax.plot(lags, avg(ml_curves),  "f97316", color="#f97316", label="ML Signal",  linewidth=1.5)
    ax.plot(lags, avg(bh_curves),  color="#22c55e",            label="BH Mass",   linewidth=1.5)
    ax.plot(lags, avg(mom_curves), color="#3b82f6",            label="Momentum",  linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("IC Decay: ML Signal vs BH Mass vs Momentum", fontsize=9)
    ax.set_xlabel("Lag (bars)")
    ax.set_ylabel("IC (Spearman)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_feature_importance_heatmap(results: Dict[str, Dict], ax: plt.Axes) -> None:
    """Heatmap: feature weights per instrument."""
    tickers = list(results.keys())
    matrix  = np.zeros((len(tickers), len(FEATURE_NAMES)))
    for i, t in enumerate(tickers):
        fi = results[t].get("feat_importance", {})
        for j, fname in enumerate(FEATURE_NAMES):
            matrix[i, j] = fi.get(fname, 0)

    # Normalise rows by L1 norm for comparability
    row_norms = np.abs(matrix).sum(axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-12, 1, row_norms)
    matrix_n  = matrix / row_norms

    im = ax.imshow(matrix_n, cmap="RdYlGn", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=8)
    plt.colorbar(im, ax=ax, label="Normalised Weight")
    ax.set_title("ML Signal Feature Importance (SGD Weights)", fontsize=9)
    for i in range(matrix_n.shape[0]):
        for j in range(matrix_n.shape[1]):
            ax.text(j, i, f"{matrix_n[i, j]:.2f}", ha="center", va="center", fontsize=6)


def plot_reliability_diagrams(results: Dict[str, Dict], axes: List[plt.Axes]) -> None:
    """Reliability diagrams for up to 4 instruments."""
    tickers_to_plot = ["BTC", "ETH", "ES", "CL"]
    for ax, ticker in zip(axes, tickers_to_plot):
        ax.set_facecolor("#161b22")
        res = results.get(ticker, {})
        bc  = res.get("_bin_centers", np.array([]))
        wr  = res.get("_win_rates",   np.array([]))
        cnt = res.get("_bin_counts",  np.array([]))

        if len(bc) == 0:
            continue

        valid = np.isfinite(wr)
        ax.plot([0, 1], [0, 1], "gray", linestyle="--", linewidth=1, label="Perfect calibration")
        ax.scatter(bc[valid], wr[valid], s=cnt[valid] / cnt[valid].max() * 200 + 10,
                   color="#f97316", zorder=5, label="Actual win rate")
        ax.set_title(f"{ticker}: Reliability Diagram", fontsize=9)
        ax.set_xlabel("Predicted P(up)")
        ax.set_ylabel("Actual Win Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)


def plot_half_life_comparison(results: Dict[str, Dict], ax: plt.Axes) -> None:
    """Bar chart: signal half-life per instrument."""
    tickers  = list(results.keys())
    ml_hls   = [min(results[t].get("ml_half_life",  99), 30) for t in tickers]
    bh_hls   = [min(results[t].get("bh_half_life",  99), 30) for t in tickers]
    mom_hls  = [min(results[t].get("mom_half_life", 99), 30) for t in tickers]

    x = np.arange(len(tickers))
    w = 0.25
    ax.bar(x - w,     ml_hls,  w, label="ML Signal",  color="#f97316", alpha=0.85)
    ax.bar(x,         bh_hls,  w, label="BH Mass",    color="#22c55e", alpha=0.85)
    ax.bar(x + w,     mom_hls, w, label="Momentum",   color="#3b82f6", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=20)
    ax.set_title("Signal Half-Life (bars) -- ML vs BH Mass vs Momentum", fontsize=9)
    ax.set_ylabel("Half-Life (bars, capped at 30)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


def plot_brier_score(results: Dict[str, Dict], ax: plt.Axes) -> None:
    tickers = list(results.keys())
    briers  = [results[t].get("brier_score", 0) or 0 for t in tickers]
    colors  = ["#22c55e" if b < 0.25 else "#fbbf24" if b < 0.30 else "#ef4444" for b in briers]
    ax.bar(tickers, briers, color=colors, alpha=0.85)
    ax.axhline(0.25, color="gold", linestyle="--", linewidth=1, label="Brier=0.25 (random)")
    ax.set_title("ML Signal Brier Score (lower = better calibration)", fontsize=9)
    ax.set_ylabel("Brier Score")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (t, v) in enumerate(zip(tickers, briers)):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=7)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_html_report(results: Dict[str, Dict], fig_path: str) -> str:
    rows = ""
    for ticker, res in results.items():
        def fmt(x):
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return "N/A"
            return f"{x:.4f}"

        fi = res.get("feat_importance", {})
        top_feat = max(fi, key=lambda k: abs(fi[k])) if fi else "N/A"
        rows += (
            f"<tr><td>{ticker}</td>"
            f"<td>{fmt(res.get('ic_ml_overall'))}</td>"
            f"<td>{fmt(res.get('ic_mom_overall'))}</td>"
            f"<td>{'YES' if (res.get('ic_ml_overall') or 0) > (res.get('ic_mom_overall') or 0) else 'NO'}</td>"
            f"<td>{fmt(min(res.get('ml_half_life', 99), 99))}</td>"
            f"<td>{fmt(min(res.get('bh_half_life', 99), 99))}</td>"
            f"<td>{fmt(res.get('brier_score'))}</td>"
            f"<td>{top_feat}</td></tr>\n"
        )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>ML Signal Diagnostic -- LARSA v18</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #c9d1d9; margin: 20px; }}
  h1 {{ color: #58a6ff; }}
  h2 {{ color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 10px 0; }}
  th, td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: right; }}
  th {{ background: #161b22; color: #f0f6fc; }}
  tr:nth-child(even) {{ background: #161b22; }}
  img {{ max-width: 100%; border: 1px solid #30363d; border-radius: 4px; margin: 12px 0; }}
  .finding {{ background: #21262d; border-left: 3px solid #f97316; padding: 8px 14px; margin: 8px 0; border-radius: 0 4px 4px 0; }}
</style>
</head>
<body>
<h1>ML Signal Diagnostic Report -- LARSA v18</h1>

<h2>Key Questions</h2>
<div class="finding">
  <strong>Does the SGD logistic predictor add value?</strong>
  Compare ML IC vs Momentum IC -- if ML IC is consistently higher, the model adds value.
</div>
<div class="finding">
  <strong>Does ML signal IC decay faster than BH Mass?</strong>
  If ML half-life &lt; BH half-life, use ML signal for short-term timing only.
</div>

<h2>Results Summary</h2>
<table>
  <tr>
    <th>Ticker</th><th>IC (ML)</th><th>IC (Momentum)</th><th>ML Beats Momentum?</th>
    <th>ML Half-Life (bars)</th><th>BH Half-Life (bars)</th>
    <th>Brier Score</th><th>Top Feature</th>
  </tr>
  {rows}
</table>

<h2>Charts</h2>
<img src="{fig_path}" alt="ML Diagnostic Charts">

<h2>Interpretation</h2>
<ul>
  <li>Features with positive weights: lagged positive returns increase P(up) -- momentum contribution.</li>
  <li>GARCH vol feature: if weight is negative, high vol reduces predicted P(up) -- volatility-adjusted sizing.</li>
  <li>Brier score below 0.25 indicates the ML signal is better than random -- approaching 0.20 is good.</li>
  <li>Rolling IC instability suggests ML signal is fragile in regime shifts -- monitor live drift.</li>
</ul>

<footer><p style="color:#484f58;font-size:11px;">Generated by ml_signal_diagnostic.py -- LARSA v18 Research</p></footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[ML SIGNAL DIAGNOSTIC] Starting ...")

    all_data: Dict[str, pd.DataFrame] = {}
    results:  Dict[str, Dict]         = {}

    for t in INSTRUMENTS:
        print(f"  Generating {t} data ...")
        all_data[t] = generate_ohlcv(t)

    for ticker, df in all_data.items():
        print(f"  Analyzing {ticker} ...")
        results[ticker] = analyze_instrument(ticker, df)
        r = results[ticker]
        print(f"    IC_ML={r['ic_ml_overall']:.4f}  IC_Mom={r['ic_mom_overall']:.4f}  "
              f"ML_HL={r['ml_half_life']:.1f}  BH_HL={r['bh_half_life']:.1f}  "
              f"Brier={r['brier_score']:.4f}")

    # Build figure
    print("[ML SIGNAL DIAGNOSTIC] Building charts ...")
    fig = plt.figure(figsize=(20, 30), facecolor="#0d1117")
    fig.suptitle("ML Signal Diagnostic -- LARSA v18", fontsize=14, color="white", y=0.99)
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.4)

    # Rolling IC for BTC and ES
    ax_btc = fig.add_subplot(gs[0, 0])
    ax_btc.set_facecolor("#161b22")
    ax_es  = fig.add_subplot(gs[0, 1])
    ax_es.set_facecolor("#161b22")
    for ax, ticker in [(ax_btc, "BTC"), (ax_es, "ES")]:
        res    = results.get(ticker, {})
        ic_ml  = res.get("_ic_ml",  np.array([]))
        ic_mom = res.get("_ic_mom", np.array([]))
        n = min(len(ic_ml), len(ic_mom))
        if n > 0:
            idx = np.arange(n)
            ax.plot(idx, ic_ml[:n],  color="#f97316", linewidth=0.8, label="ML Signal IC")
            ax.plot(idx, ic_mom[:n], color="#3b82f6", linewidth=0.8, label="Momentum IC")
            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
            ax.set_title(f"{ticker}: Rolling IC", fontsize=9)
            ax.set_xlabel("Bar")
            ax.set_ylabel("IC (Spearman)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor("#161b22")
    plot_ic_overall_comparison(results, ax2)

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor("#161b22")
    # IC decay
    lags = np.arange(1, 21)
    ml_curves  = [np.array(r["_ml_decay"])  for r in results.values() if "_ml_decay"  in r]
    bh_curves  = [np.array(r["_bh_decay"])  for r in results.values() if "_bh_decay"  in r]
    mom_curves = [np.array(r["_mom_decay"]) for r in results.values() if "_mom_decay" in r]
    def avg(curves):
        return np.nanmean(np.vstack(curves), axis=0) if curves else np.zeros(20)
    ax3.plot(lags, avg(ml_curves),  color="#f97316", label="ML Signal",  linewidth=1.5)
    ax3.plot(lags, avg(bh_curves),  color="#22c55e", label="BH Mass",    linewidth=1.5)
    ax3.plot(lags, avg(mom_curves), color="#3b82f6", label="Momentum",   linewidth=1.5)
    ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax3.set_title("IC Decay: ML vs BH Mass vs Momentum", fontsize=9)
    ax3.set_xlabel("Lag (bars)")
    ax3.set_ylabel("IC (Spearman)")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor("#161b22")
    plot_half_life_comparison(results, ax4)

    ax5 = fig.add_subplot(gs[3, :])
    ax5.set_facecolor("#161b22")
    plot_feature_importance_heatmap(results, ax5)

    # Reliability diagrams
    tickers_rel = ["BTC", "ETH", "ES", "CL"]
    ax_rel_axes = [fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 1])]
    for ax in ax_rel_axes:
        ax.set_facecolor("#161b22")

    for ax_idx, ticker in enumerate(["BTC", "ES"]):
        ax = ax_rel_axes[ax_idx]
        res = results.get(ticker, {})
        bc  = res.get("_bin_centers", np.array([]))
        wr  = res.get("_win_rates",   np.array([]))
        cnt = res.get("_bin_counts",  np.array([]))
        if len(bc) > 0:
            valid = np.isfinite(wr)
            ax.plot([0, 1], [0, 1], "gray", linestyle="--", linewidth=1, label="Perfect")
            if valid.any() and cnt[valid].max() > 0:
                ax.scatter(bc[valid], wr[valid],
                           s=cnt[valid] / cnt[valid].max() * 200 + 10,
                           color="#f97316", zorder=5, label="Actual win rate")
            ax.set_title(f"{ticker}: Reliability Diagram", fontsize=9)
            ax.set_xlabel("Predicted P(up)")
            ax.set_ylabel("Actual Win Rate")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    for ax in fig.get_axes():
        ax.tick_params(colors="white")
        for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            item.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    fig_path = OUT_DIR / "ml_signal_charts.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[ML SIGNAL DIAGNOSTIC] Saved {fig_path}")

    html = build_html_report(results, "ml_signal_charts.png")
    html_path = OUT_DIR / "ml_signal_diagnostic.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[ML SIGNAL DIAGNOSTIC] Saved {html_path}")
    print("[ML SIGNAL DIAGNOSTIC] Done.")


if __name__ == "__main__":
    main()
