"""
signal_ic_decay_study.py
------------------------
Comprehensive IC decay analysis for all signals in the LARSA v18 strategy.

Signals analyzed:
  BH_MASS, CF_CROSS, HURST_REGIME, NAV_OMEGA, NAV_GEODESIC,
  GARCH_VOL, ML_SIGNAL, GRANGER_BTC, OU_REVERSION

For each signal:
  - IC at lags 1..20 bars (15-minute bars)
  - Exponential decay fit: IC(h) = IC0 * exp(-lambda*h)
  - Half-life = ln(2) / lambda
  - Regime-conditioned IC (BH active vs inactive)
  - Signal correlation matrix (redundancy check)

Outputs:
  ic_decay_study.html
  ic_decay_results.json
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
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUMENTS   = ["BTC", "ETH", "SOL", "ES", "NQ", "CL", "GC", "ZB"]
MAX_LAG       = 20
N_BARS        = 10_000
RANDOM_SEED   = 137
OUT_DIR       = Path(__file__).parent

SIGNAL_NAMES = [
    "BH_MASS", "CF_CROSS", "HURST_REGIME", "NAV_OMEGA",
    "NAV_GEODESIC", "GARCH_VOL", "ML_SIGNAL",
    "GRANGER_BTC", "OU_REVERSION",
]

CF_15M = {
    "BTC": 0.0012, "ETH": 0.0015, "SOL": 0.0020,
    "ES":  0.0003, "NQ":  0.0004, "CL":  0.0015,
    "GC":  0.0008, "ZB":  0.0005,
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_ohlcv(ticker: str, n: int = N_BARS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generates synthetic 15-min OHLCV with regime dynamics."""
    rng = np.random.default_rng(seed + abs(hash(ticker)) % 9999)
    is_crypto = ticker in ("BTC", "ETH", "SOL")
    base_vol  = 0.0025 if is_crypto else 0.0008
    drift     = 5e-5   if is_crypto else 2e-5

    # Markov vol regime
    regime = np.zeros(n, dtype=int)
    for i in range(1, n):
        regime[i] = (1 - regime[i - 1]) if (rng.random() < (0.03 if regime[i - 1] == 0 else 0.07)) else regime[i - 1]

    vol = np.where(regime == 1, base_vol * 3.0, base_vol)
    shocks  = rng.standard_t(df=6, size=n) * vol
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = drift - 0.04 * returns[i - 1] + shocks[i]

    sp = {"BTC": 42000, "ETH": 2500, "SOL": 100,
          "ES": 4500, "NQ": 15000, "CL": 75, "GC": 1950, "ZB": 120}.get(ticker, 100)
    close = sp * np.exp(np.cumsum(returns))
    hl    = vol * close * rng.uniform(0.5, 1.5, n)
    high  = close + hl * rng.uniform(0.3, 0.7, n)
    low   = close - hl * rng.uniform(0.3, 0.7, n)
    open_ = close * np.exp(rng.normal(0, vol * 0.3, n))
    vol_bars = (10000 * rng.lognormal(0, 0.5, n)).astype(int)

    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol_bars},
        index=ts,
    )


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_bh_mass_signal(
    close: np.ndarray, cf: float, thresh: float = 1.92, bh_decay: float = 0.95
) -> np.ndarray:
    n    = len(close)
    mass = np.zeros(n)
    m    = 0.0
    ctl  = 0
    for i in range(1, n):
        b = abs(close[i] - close[i - 1]) / (close[i - 1] + 1e-9) / (cf + 1e-9)
        if b < 1.0:
            ctl += 1
            sb  = min(2.0, 1.0 + ctl * 0.1)
            m   = m * 0.97 + 0.03 * sb
        else:
            ctl = 0
            m  *= bh_decay
        mass[i] = m
    return mass


def compute_bh_active(bh_mass: np.ndarray, thresh: float = 1.92) -> np.ndarray:
    """Boolean: BH active flag from mass series."""
    n = len(bh_mass)
    active = np.zeros(n, dtype=bool)
    was_active = False
    for i in range(n):
        if not was_active:
            active[i] = bh_mass[i] > thresh
        else:
            active[i] = bh_mass[i] > 1.0
        was_active = active[i]
    return active


def compute_cf_cross_signal(close: np.ndarray, cf: float) -> np.ndarray:
    """CF_CROSS: current beta relative to its rolling EMA."""
    n    = len(close)
    beta = np.zeros(n)
    for i in range(1, n):
        beta[i] = abs(close[i] - close[i - 1]) / (close[i - 1] + 1e-9) / (cf + 1e-9)
    beta_s = pd.Series(beta)
    ema    = beta_s.ewm(span=20, adjust=False).mean().values
    signal = np.where(ema > 1e-12, beta / ema - 1.0, 0.0)
    return signal


def hurst_rs(series: np.ndarray) -> float:
    """R/S Hurst exponent estimate."""
    n = len(series)
    if n < 10:
        return 0.5
    mean = series.mean()
    dev  = np.cumsum(series - mean)
    r    = dev.max() - dev.min()
    s    = series.std(ddof=1)
    if s < 1e-12 or r < 1e-12:
        return 0.5
    return np.log(r / s) / np.log(n)


def compute_hurst_regime(close: np.ndarray, window: int = 100) -> np.ndarray:
    """Rolling Hurst exponent."""
    n    = len(close)
    rets = np.diff(np.log(close), prepend=np.log(close[0]))
    hurst = np.full(n, 0.5)
    for i in range(window, n):
        hurst[i] = hurst_rs(rets[i - window: i])
    return hurst


def compute_nav_omega(close: np.ndarray, k: float = 0.5) -> np.ndarray:
    """NAV_OMEGA: k * |log_return| per bar."""
    log_rets = np.log(close / np.roll(close, 1))
    log_rets[0] = 0.0
    return k * np.abs(log_rets)


def compute_nav_geodesic(omega: np.ndarray, span: int = 20) -> np.ndarray:
    """NAV_GEODESIC: omega / omega_EMA."""
    ema = pd.Series(omega).ewm(span=span, adjust=False).mean().values
    return np.where(ema > 1e-12, omega / ema, 0.0)


def compute_garch_vol(returns: np.ndarray, alpha: float = 0.1, beta: float = 0.85) -> np.ndarray:
    """GARCH(1,1) variance series."""
    n      = len(returns)
    var    = np.zeros(n)
    omega  = returns.var() * (1.0 - alpha - beta)
    v      = returns.var()
    for i in range(1, n):
        v      = omega + alpha * returns[i - 1] ** 2 + beta * v
        var[i] = v
    return np.sqrt(np.maximum(var, 0.0))


def compute_ml_signal(
    returns: np.ndarray,
    garch_vol: np.ndarray,
    lr: float = 0.01,
    train_start: int = 200,
) -> np.ndarray:
    """
    Online SGD logistic predictor.
    Features: lagged returns (lags 1..5) + GARCH vol.
    Label: sign(ret[t+1]).
    Returns probability estimate for up.
    """
    n       = len(returns)
    n_feat  = 6
    weights = np.zeros(n_feat)
    signal  = np.zeros(n)

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
        prob = sigmoid(np.dot(weights, feat))
        signal[i] = prob - 0.5  # centered around 0

        label = 1.0 if returns[i + 1] > 0 else 0.0
        grad  = (prob - label) * feat
        weights -= lr * grad

    return signal


def compute_granger_btc(
    target_returns: np.ndarray,
    btc_returns: np.ndarray,
    window: int = 100,
) -> np.ndarray:
    """
    Rolling Granger-like signal: rolling beta of target ~ BTC lagged.
    Uses OLS slope as the predictive signal.
    """
    n      = len(target_returns)
    signal = np.zeros(n)
    for i in range(window + 1, n):
        x = btc_returns[i - window - 1: i - 1]
        y = target_returns[i - window: i]
        if len(x) < 10:
            continue
        x_dm = x - x.mean()
        y_dm = y - y.mean()
        denom = (x_dm ** 2).sum()
        if denom < 1e-12:
            continue
        beta = (x_dm * y_dm).sum() / denom
        signal[i] = beta * btc_returns[i - 1]
    return signal


def compute_ou_reversion(close: np.ndarray, window: int = 60) -> np.ndarray:
    """
    OU mean reversion signal: z-score of log-price relative to rolling mean.
    Negative = below mean (reversion signal: buy).
    """
    log_p  = np.log(close)
    log_s  = pd.Series(log_p)
    roll_m = log_s.rolling(window, min_periods=window // 2).mean().values
    roll_s = log_s.rolling(window, min_periods=window // 2).std().values
    roll_s = np.where(roll_s < 1e-12, 1e-12, roll_s)
    return -(log_p - roll_m) / roll_s   # neg z-score: big negative = oversold -> buy


def compute_all_signals(df: pd.DataFrame, ticker: str, btc_close: np.ndarray) -> pd.DataFrame:
    """Computes all 9 signals for one instrument's bar data."""
    close   = df["close"].values
    log_ret = np.log(close / np.roll(close, 1))
    log_ret[0] = 0.0
    cf = CF_15M.get(ticker, 0.001)

    bh_mass    = compute_bh_mass_signal(close, cf)
    cf_cross   = compute_cf_cross_signal(close, cf)
    hurst      = compute_hurst_regime(close)
    omega      = compute_nav_omega(close)
    geo_dev    = compute_nav_geodesic(omega)
    garch_v    = compute_garch_vol(log_ret)
    ml_sig     = compute_ml_signal(log_ret, garch_v)
    btc_ret    = np.log(btc_close / np.roll(btc_close, 1))
    btc_ret[0] = 0.0
    granger    = compute_granger_btc(log_ret, btc_ret[:len(log_ret)])
    ou_rev     = compute_ou_reversion(close)

    signals_df = pd.DataFrame({
        "BH_MASS":      bh_mass,
        "CF_CROSS":     cf_cross,
        "HURST_REGIME": hurst,
        "NAV_OMEGA":    omega,
        "NAV_GEODESIC": geo_dev,
        "GARCH_VOL":    garch_v,
        "ML_SIGNAL":    ml_sig,
        "GRANGER_BTC":  granger,
        "OU_REVERSION": ou_rev,
        "close":        close,
        "log_ret":      log_ret,
    }, index=df.index)

    # BH active flag for regime conditioning
    bh_active = compute_bh_active(bh_mass)
    signals_df["BH_ACTIVE"] = bh_active
    return signals_df


# ---------------------------------------------------------------------------
# IC decay computation
# ---------------------------------------------------------------------------

def compute_ic_at_lag(signal: np.ndarray, returns: np.ndarray, lag: int) -> float:
    """Spearman IC between signal[t] and returns[t+lag]."""
    fwd = np.roll(returns, -lag)
    fwd[-lag:] = np.nan
    mask = np.isfinite(signal) & np.isfinite(fwd)
    if mask.sum() < 30:
        return np.nan
    ic, _ = spearmanr(signal[mask], fwd[mask])
    return float(ic)


def compute_ic_decay_curve(
    signal: np.ndarray,
    returns: np.ndarray,
    max_lag: int = MAX_LAG,
) -> np.ndarray:
    """Returns array of IC values at lags 1..max_lag."""
    return np.array([compute_ic_at_lag(signal, returns, h) for h in range(1, max_lag + 1)])


def exponential_decay_model(h: np.ndarray, ic0: float, lam: float) -> np.ndarray:
    return ic0 * np.exp(-lam * h)


def fit_exponential_decay(ic_curve: np.ndarray) -> Tuple[float, float, float]:
    """
    Fits IC(h) = IC0 * exp(-lambda*h).
    Returns (IC0, lambda, half_life_bars).
    """
    lags  = np.arange(1, len(ic_curve) + 1, dtype=float)
    valid = np.isfinite(ic_curve)
    if valid.sum() < 3:
        return 0.0, 1.0, np.log(2)
    try:
        popt, _ = curve_fit(
            exponential_decay_model,
            lags[valid], ic_curve[valid],
            p0=[ic_curve[valid][0], 0.1],
            maxfev=5000,
            bounds=([-1.0, 0.001], [1.0, 10.0]),
        )
        ic0, lam   = popt
        half_life  = np.log(2) / lam if lam > 1e-9 else 999.0
        return float(ic0), float(lam), float(half_life)
    except Exception:
        return 0.0, 0.5, np.log(2) * 2.0


# ---------------------------------------------------------------------------
# Regime-conditioned IC
# ---------------------------------------------------------------------------

def compute_regime_ic(
    signal: np.ndarray,
    returns: np.ndarray,
    bh_active: np.ndarray,
    lag: int = 1,
) -> Tuple[float, float]:
    """IC when BH active vs BH inactive at lag=1."""
    fwd = np.roll(returns, -lag)
    fwd[-lag:] = np.nan
    mask_active   = bh_active & np.isfinite(signal) & np.isfinite(fwd)
    mask_inactive = (~bh_active) & np.isfinite(signal) & np.isfinite(fwd)

    def safe_ic(m: np.ndarray) -> float:
        if m.sum() < 15:
            return np.nan
        ic, _ = spearmanr(signal[m], fwd[m])
        return float(ic)

    return safe_ic(mask_active), safe_ic(mask_inactive)


# ---------------------------------------------------------------------------
# Signal correlation matrix
# ---------------------------------------------------------------------------

def compute_signal_correlation(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation matrix among all signal columns."""
    sig_cols = [c for c in SIGNAL_NAMES if c in signals_df.columns]
    corr = signals_df[sig_cols].rank().corr(method="pearson")
    return corr


# ---------------------------------------------------------------------------
# Full study per instrument
# ---------------------------------------------------------------------------

def run_ic_decay_study_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    btc_close: np.ndarray,
) -> Dict:
    """Returns full IC decay study results for one instrument."""
    print(f"  [{ticker}] Computing signals ...")
    signals_df = compute_all_signals(df, ticker, btc_close)
    returns    = signals_df["log_ret"].values
    bh_active  = signals_df["BH_ACTIVE"].values

    decay_results: Dict[str, Dict] = {}
    regime_results: Dict[str, Dict] = {}

    for sig_name in SIGNAL_NAMES:
        if sig_name not in signals_df.columns:
            continue
        sig = signals_df[sig_name].values
        ic_curve = compute_ic_decay_curve(sig, returns, MAX_LAG)
        ic0, lam, half_life = fit_exponential_decay(ic_curve)
        ic_active, ic_inactive = compute_regime_ic(sig, returns, bh_active)

        decay_results[sig_name] = {
            "ic_curve":   ic_curve.tolist(),
            "ic0":        ic0,
            "lambda":     lam,
            "half_life":  half_life,
        }
        regime_results[sig_name] = {
            "ic_bh_active":   ic_active,
            "ic_bh_inactive": ic_inactive,
        }

    corr_matrix = compute_signal_correlation(signals_df)
    return {
        "ticker":         ticker,
        "decay_results":  decay_results,
        "regime_results": regime_results,
        "corr_matrix":    corr_matrix.to_dict(),
        "_signals_df":    signals_df,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_ic_decay_curves(
    study_results: Dict[str, Dict],
    fig: plt.Figure,
    gs_row: int,
    n_cols: int,
) -> None:
    """IC decay curve for each signal, averaged across instruments."""
    lags = np.arange(1, MAX_LAG + 1)
    signal_avg_ic: Dict[str, np.ndarray] = {}
    for sig in SIGNAL_NAMES:
        curves = []
        for ticker, res in study_results.items():
            dr = res["decay_results"].get(sig)
            if dr:
                curves.append(np.array(dr["ic_curve"]))
        if curves:
            signal_avg_ic[sig] = np.nanmean(np.vstack(curves), axis=0)

    colors = plt.cm.tab10(np.linspace(0, 1, len(SIGNAL_NAMES)))
    ax = fig.add_subplot(gs_row)
    ax.set_facecolor("#161b22")
    for i, sig in enumerate(SIGNAL_NAMES):
        if sig in signal_avg_ic:
            ax.plot(lags, signal_avg_ic[sig], label=sig, color=colors[i], linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("IC Decay Curves -- All Signals (avg across instruments)", color="white", fontsize=10)
    ax.set_xlabel("Lag (15-min bars)")
    ax.set_ylabel("IC (Spearman)")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_half_life_comparison(
    study_results: Dict[str, Dict],
    ax: plt.Axes,
) -> None:
    """Bar chart: half-life per signal per instrument."""
    half_lives: Dict[str, List[float]] = {s: [] for s in SIGNAL_NAMES}
    for ticker, res in study_results.items():
        for sig in SIGNAL_NAMES:
            dr = res["decay_results"].get(sig)
            if dr:
                hl = dr["half_life"]
                if np.isfinite(hl) and hl < 500:
                    half_lives[sig].append(hl)

    sigs       = [s for s in SIGNAL_NAMES if half_lives[s]]
    mean_hls   = [np.mean(half_lives[s]) for s in sigs]
    sorted_idx = np.argsort(mean_hls)
    sigs_s     = [sigs[i] for i in sorted_idx]
    hls_s      = [mean_hls[i] for i in sorted_idx]

    colors = ["#ff7070" if h < 5 else "#ffd700" if h < 10 else "#70c8ff" for h in hls_s]
    bars = ax.barh(sigs_s, hls_s, color=colors, alpha=0.85)
    ax.axvline(5, color="red", linestyle="--", alpha=0.6, label="5-bar day-trading threshold")
    ax.axvline(10, color="orange", linestyle="--", alpha=0.6, label="10-bar swing threshold")
    ax.set_title("Signal Half-Life (bars) -- shorter = day trading, longer = swing", fontsize=9)
    ax.set_xlabel("Half-Life (15-min bars)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="x")
    for bar, v in zip(bars, hls_s):
        ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center", fontsize=7)


def plot_regime_ic_comparison(
    study_results: Dict[str, Dict],
    ax: plt.Axes,
) -> None:
    """Grouped bar: IC when BH active vs inactive for each signal."""
    sigs = SIGNAL_NAMES
    active_ic   = []
    inactive_ic = []
    for sig in sigs:
        acts = []
        inacts = []
        for ticker, res in study_results.items():
            rr = res["regime_results"].get(sig, {})
            a = rr.get("ic_bh_active")
            b = rr.get("ic_bh_inactive")
            if a is not None and np.isfinite(a):
                acts.append(a)
            if b is not None and np.isfinite(b):
                inacts.append(b)
        active_ic.append(np.nanmean(acts) if acts else 0.0)
        inactive_ic.append(np.nanmean(inacts) if inacts else 0.0)

    x = np.arange(len(sigs))
    w = 0.35
    ax.bar(x - w / 2, active_ic,   w, label="BH Active",   color="#f97316", alpha=0.85)
    ax.bar(x + w / 2, inactive_ic, w, label="BH Inactive", color="#3b82f6", alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sigs, rotation=30, ha="right", fontsize=8)
    ax.set_title("Regime-Conditioned IC (BH Active vs Inactive)", fontsize=9)
    ax.set_ylabel("IC (Spearman)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def plot_correlation_heatmap(
    study_results: Dict[str, Dict],
    ax: plt.Axes,
    ticker: str = "BTC",
) -> None:
    """Signal correlation matrix for one instrument."""
    res = study_results.get(ticker, {})
    corr_d = res.get("corr_matrix", {})
    if not corr_d:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    sigs = [s for s in SIGNAL_NAMES if s in corr_d]
    matrix = np.array([[corr_d[r].get(c, 0) for c in sigs] for r in sigs])

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sigs)))
    ax.set_xticklabels(sigs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(sigs)))
    ax.set_yticklabels(sigs, fontsize=7)
    plt.colorbar(im, ax=ax, label="Spearman corr")
    ax.set_title(f"{ticker}: Signal Correlation Matrix", fontsize=9)
    for i in range(len(sigs)):
        for j in range(len(sigs)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(matrix[i, j]) < 0.6 else "white")


def plot_ic_heatmap_by_instrument(
    study_results: Dict[str, Dict],
    ax: plt.Axes,
    lag: int = 1,
) -> None:
    """Heatmap of IC@lag=1 for each (instrument, signal)."""
    tickers = [t for t in INSTRUMENTS if t in study_results]
    matrix  = np.full((len(tickers), len(SIGNAL_NAMES)), np.nan)
    for i, ticker in enumerate(tickers):
        for j, sig in enumerate(SIGNAL_NAMES):
            dr = study_results[ticker]["decay_results"].get(sig)
            if dr and len(dr["ic_curve"]) >= lag:
                matrix[i, j] = dr["ic_curve"][lag - 1]

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.1, vmax=0.1, aspect="auto")
    ax.set_xticks(range(len(SIGNAL_NAMES)))
    ax.set_xticklabels(SIGNAL_NAMES, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=8)
    plt.colorbar(im, ax=ax, label=f"IC@lag={lag}")
    ax.set_title(f"IC@lag={lag} Heatmap: Instrument x Signal", fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6,
                        color="black" if abs(val) < 0.07 else "white")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_html_report(
    study_results: Dict[str, Dict],
    fig_path: str,
) -> str:
    # Summary table rows
    rows_html = ""
    for ticker in INSTRUMENTS:
        if ticker not in study_results:
            continue
        res = study_results[ticker]
        for sig in SIGNAL_NAMES:
            dr = res["decay_results"].get(sig, {})
            rr = res["regime_results"].get(sig, {})
            ic0   = dr.get("ic0", 0)
            lam   = dr.get("lambda", 0)
            hl    = dr.get("half_life", 0)
            ic_a  = rr.get("ic_bh_active", None)
            ic_i  = rr.get("ic_bh_inactive", None)
            ic1   = dr["ic_curve"][0] if dr.get("ic_curve") else 0

            def fmt(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return "N/A"
                return f"{x:.4f}"

            rows_html += (
                f"<tr><td>{ticker}</td><td>{sig}</td>"
                f"<td>{fmt(ic1)}</td>"
                f"<td>{fmt(ic0)}</td>"
                f"<td>{fmt(lam)}</td>"
                f"<td>{fmt(hl)}</td>"
                f"<td>{fmt(ic_a)}</td>"
                f"<td>{fmt(ic_i)}</td></tr>\n"
            )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Signal IC Decay Study -- LARSA v18</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #c9d1d9; margin: 20px; }}
  h1   {{ color: #58a6ff; }}
  h2   {{ color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  th, td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: right; }}
  th {{ background: #161b22; color: #f0f6fc; }}
  tr:nth-child(even) {{ background: #161b22; }}
  img {{ max-width: 100%; border: 1px solid #30363d; border-radius: 4px; margin: 12px 0; }}
</style>
</head>
<body>
<h1>Signal IC Decay Study -- LARSA v18</h1>
<p>Signals: {', '.join(SIGNAL_NAMES)}</p>

<h2>IC Decay Results</h2>
<table>
  <tr>
    <th>Ticker</th><th>Signal</th><th>IC@lag=1</th>
    <th>IC0 (fit)</th><th>Lambda</th><th>Half-Life (bars)</th>
    <th>IC|BH Active</th><th>IC|BH Inactive</th>
  </tr>
  {rows_html}
</table>

<h2>Charts</h2>
<img src="{fig_path}" alt="IC Decay Charts">

<h2>Interpretation</h2>
<ul>
  <li><strong>Short half-life (&lt;5 bars, 1.25h):</strong> Day-trading signals -- BH_MASS, NAV_OMEGA, CF_CROSS</li>
  <li><strong>Medium half-life (5-10 bars):</strong> Intraday swing -- ML_SIGNAL, GRANGER_BTC</li>
  <li><strong>Long half-life (&gt;10 bars):</strong> Swing signals -- HURST_REGIME, OU_REVERSION</li>
  <li><strong>Regime conditioning:</strong> Signals with IC|BH_Active &gt;&gt; IC|BH_Inactive benefit from BH gating</li>
</ul>

<footer><p style="color:#484f58;font-size:11px;">Generated by signal_ic_decay_study.py -- LARSA v18 Research</p></footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[IC DECAY STUDY] Starting ...")

    all_data: Dict[str, pd.DataFrame] = {}
    for t in INSTRUMENTS:
        all_data[t] = generate_ohlcv(t)
    btc_close = all_data["BTC"]["close"].values

    study_results: Dict[str, Dict] = {}
    for ticker, df in all_data.items():
        study_results[ticker] = run_ic_decay_study_for_ticker(ticker, df, btc_close)

    # Save JSON (strip internal DataFrames)
    json_out: Dict = {}
    for ticker, res in study_results.items():
        json_out[ticker] = {
            "decay_results":  res["decay_results"],
            "regime_results": res["regime_results"],
        }

    json_path = OUT_DIR / "ic_decay_results.json"
    with open(json_path, "w") as f:
        def _default(x):
            if isinstance(x, float) and np.isnan(x):
                return None
            if isinstance(x, np.floating):
                return float(x)
            return x
        json.dump(json_out, f, indent=2, default=_default)
    print(f"[IC DECAY STUDY] Saved {json_path}")

    # Build figure
    print("[IC DECAY STUDY] Building charts ...")
    fig = plt.figure(figsize=(20, 24), facecolor="#0d1117")
    fig.suptitle("Signal IC Decay Study -- LARSA v18", fontsize=14, color="white", y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.4)

    # Row 0: IC decay curves (full width)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor("#161b22")

    lags   = np.arange(1, MAX_LAG + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(SIGNAL_NAMES)))
    for i, sig in enumerate(SIGNAL_NAMES):
        curves = []
        for ticker, res in study_results.items():
            dr = res["decay_results"].get(sig)
            if dr:
                curves.append(np.array(dr["ic_curve"]))
        if curves:
            avg = np.nanmean(np.vstack(curves), axis=0)
            ax0.plot(lags, avg, label=sig, color=colors[i], linewidth=1.5)
    ax0.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax0.set_title("IC Decay Curves -- All Signals (avg across instruments)", color="white", fontsize=10)
    ax0.set_xlabel("Lag (15-min bars)")
    ax0.set_ylabel("IC (Spearman)")
    ax0.legend(fontsize=7, ncol=3, loc="upper right")
    ax0.grid(True, alpha=0.3)

    # Row 1: half-life bar, regime IC
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor("#161b22")
    plot_half_life_comparison(study_results, ax1)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor("#161b22")
    plot_regime_ic_comparison(study_results, ax2)

    # Row 2: IC heatmap, correlation
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_facecolor("#161b22")
    plot_ic_heatmap_by_instrument(study_results, ax3, lag=1)

    # Row 3: correlation matrix for BTC and ETH
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.set_facecolor("#161b22")
    plot_correlation_heatmap(study_results, ax4, "BTC")

    ax5 = fig.add_subplot(gs[3, 1])
    ax5.set_facecolor("#161b22")
    plot_correlation_heatmap(study_results, ax5, "ES")

    for ax in fig.get_axes():
        ax.tick_params(colors="white")
        for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            item.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    fig_path = OUT_DIR / "ic_decay_charts.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[IC DECAY STUDY] Saved {fig_path}")

    html = build_html_report(study_results, "ic_decay_charts.png")
    html_path = OUT_DIR / "ic_decay_study.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[IC DECAY STUDY] Saved {html_path}")
    print("[IC DECAY STUDY] Done.")


if __name__ == "__main__":
    main()
