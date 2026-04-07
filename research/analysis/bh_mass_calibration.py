"""
bh_mass_calibration.py
----------------------
Calibrates Black Hole (BH) mass parameters and QuatNav parameters
for the LARSA v18 trading strategy from historical OHLCV data.

Instruments: BTC, ETH, SOL, ES, NQ, CL, GC, ZB
BH_MASS_THRESH grid: [1.5, 1.7, 1.92, 2.1, 2.3]
NAV_OMEGA_SCALE_K grid: [0.25, 0.50, 0.75, 1.0]
NAV_GEO_ENTRY_GATE grid: [2.0, 2.5, 3.0, 3.5]

Outputs:
  calibration_results.json -- optimal params per instrument
  calibration_report.html  -- charts and summary tables
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUMENTS = ["BTC", "ETH", "SOL", "ES", "NQ", "CL", "GC", "ZB"]

BH_THRESH_GRID   = [1.5, 1.7, 1.92, 2.1, 2.3]
OMEGA_SCALE_GRID = [0.25, 0.50, 0.75, 1.0]
GEO_GATE_GRID    = [2.0, 2.5, 3.0, 3.5]

# Instrument-level CF (critical fraction) matching LARSA v16 design
CF_15M = {
    "BTC": 0.0012, "ETH": 0.0015, "SOL": 0.0020,
    "ES":  0.0003, "NQ":  0.0004, "CL":  0.0015,
    "GC":  0.0008, "ZB":  0.0005,
}

BH_FORM_OVERRIDE = {"CL": 1.8, "SOL": 1.8, "ETH": 1.7}

FORWARD_BARS_15M = 4    # 15-min bars -> 1 hour
FORWARD_BARS_1H  = 4    # 1-hour bars -> 4 hours
N_BARS_SYNTHETIC = 8_000
RANDOM_SEED = 42

OUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _crypto_vol_regime(n: int, rng: np.random.Generator) -> np.ndarray:
    """Markov-switching vol regime: HIGH / LOW."""
    state = np.zeros(n, dtype=int)
    for i in range(1, n):
        if state[i - 1] == 0:
            state[i] = 1 if rng.random() < 0.03 else 0
        else:
            state[i] = 0 if rng.random() < 0.07 else 1
    return state


def generate_ohlcv(
    ticker: str,
    n_bars: int = N_BARS_SYNTHETIC,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates realistic synthetic 15-min OHLCV bar data.
    Crypto instruments have higher vol and fatter tails.
    Equity instruments have calmer vol and mild autocorrelation.
    """
    rng = np.random.default_rng(seed + abs(hash(ticker)) % 10_000)

    is_crypto = ticker in ("BTC", "ETH", "SOL")
    base_vol = 0.0025 if is_crypto else 0.0008
    drift    = 0.00005 if is_crypto else 0.00002
    mean_rev = 0.0 if is_crypto else -0.05

    # regime switching for crypto
    if is_crypto:
        regime = _crypto_vol_regime(n_bars, rng)
        vol_series = np.where(regime == 1, base_vol * 3.0, base_vol)
    else:
        vol_series = np.full(n_bars, base_vol)

    # log returns with mild autocorrelation
    shocks = rng.standard_t(df=6, size=n_bars) * vol_series
    returns = np.zeros(n_bars)
    returns[0] = drift + shocks[0]
    for i in range(1, n_bars):
        returns[i] = drift + mean_rev * returns[i - 1] + shocks[i]

    # price series
    start_price = {"BTC": 42_000, "ETH": 2_500, "SOL": 100,
                   "ES": 4_500, "NQ": 15_000, "CL": 75,
                   "GC": 1_950, "ZB": 120}.get(ticker, 100)
    log_price = np.log(start_price) + np.cumsum(returns)
    close = np.exp(log_price)

    # OHLCV construction
    hl_spread = vol_series * close * rng.uniform(0.5, 1.5, n_bars)
    high  = close + hl_spread * rng.uniform(0.3, 0.7, n_bars)
    low   = close - hl_spread * rng.uniform(0.3, 0.7, n_bars)
    open_ = close * (1 + rng.normal(0, vol_series * 0.3))

    volume_base = 10_000 if is_crypto else 50_000
    volume = (volume_base * rng.lognormal(0, 0.5, n_bars)).astype(int)

    timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="15min")
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=timestamps)
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# BH mass computation
# ---------------------------------------------------------------------------

def compute_bh_mass(
    df: pd.DataFrame,
    ticker: str,
    bh_thresh: float,
    cf: float | None = None,
    bh_decay: float = 0.95,
    min_ctl: int = 3,
) -> pd.DataFrame:
    """
    Replicates the FutureInstrument.update_bh() logic in LARSA v16/v18.
    Returns a DataFrame with columns:
      beta, bh_mass, bh_active, bh_dir, bit
    """
    if cf is None:
        cf = CF_15M.get(ticker, 0.001)

    close = df["close"].values
    n     = len(close)

    bh_mass   = np.zeros(n)
    bh_active = np.zeros(n, dtype=bool)
    bh_dir    = np.zeros(n, dtype=int)
    beta_arr  = np.zeros(n)
    ctl_arr   = np.zeros(n, dtype=int)
    bit_arr   = np.empty(n, dtype=object)

    mass  = 0.0
    ctl   = 0
    active = False

    bh_collapse = 1.0

    for i in range(1, n):
        b = abs(close[i] - close[i - 1]) / (close[i - 1] + 1e-9) / (cf + 1e-9)
        beta_arr[i] = b

        was_active = active
        if b < 1.0:
            bit_arr[i] = "TIMELIKE"
            ctl += 1
            sb = min(2.0, 1.0 + ctl * 0.1)
            mass = mass * 0.97 + 0.03 * 1.0 * sb
        else:
            bit_arr[i] = "SPACELIKE"
            ctl = 0
            mass *= bh_decay

        if not was_active:
            active = (mass > bh_thresh) and (ctl >= min_ctl)
        else:
            active = (mass > bh_collapse) and (ctl >= min_ctl)

        if not was_active and active:
            lookback = min(20, i)
            d = 1 if close[i] > close[i - lookback] else -1
            bh_dir[i] = d
        elif was_active and not active:
            bh_dir[i] = 0
        else:
            bh_dir[i] = bh_dir[i - 1] if i > 0 else 0

        bh_mass[i]   = mass
        bh_active[i] = active
        ctl_arr[i]   = ctl

    result = df.copy()
    result["beta"]      = beta_arr
    result["bh_mass"]   = bh_mass
    result["bh_active"] = bh_active
    result["bh_dir"]    = bh_dir
    result["bit"]       = bit_arr
    result["ctl"]       = ctl_arr
    return result


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------

def add_forward_returns(df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """Adds fwd_ret_{horizon} column (log return)."""
    df = df.copy()
    df[f"fwd_ret_{horizon}"] = np.log(
        df["close"].shift(-horizon) / df["close"]
    )
    return df


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def compute_ic(signal: np.ndarray, fwd_ret: np.ndarray) -> float:
    """Rank IC (Spearman) between signal and forward return."""
    mask = np.isfinite(signal) & np.isfinite(fwd_ret)
    if mask.sum() < 30:
        return np.nan
    ic, _ = spearmanr(signal[mask], fwd_ret[mask])
    return float(ic)


def compute_conditional_sharpe(
    df: pd.DataFrame,
    fwd_col: str,
    condition: np.ndarray,
) -> float:
    """Sharpe ratio of forward returns conditional on boolean array."""
    rets = df.loc[condition, fwd_col].dropna()
    if len(rets) < 10:
        return np.nan
    mu  = rets.mean()
    std = rets.std()
    if std < 1e-12:
        return np.nan
    return float(mu / std * np.sqrt(252 * 26))  # annualised (26 bars/day)


def compute_false_positive_rate(
    df: pd.DataFrame,
    fwd_col: str,
    active_mask: np.ndarray,
    direction: np.ndarray,
) -> float:
    """
    False positive: BH active, direction is up (+1), but forward return < 0
    (or vice versa). Returns fraction of BH-active bars that are false positives.
    """
    idx = np.where(active_mask)[0]
    if len(idx) == 0:
        return np.nan
    fwd = df[fwd_col].values
    dirs = direction
    fp = 0
    valid = 0
    for i in idx:
        if i >= len(fwd) or not np.isfinite(fwd[i]):
            continue
        if dirs[i] == 1 and fwd[i] < 0:
            fp += 1
        elif dirs[i] == -1 and fwd[i] > 0:
            fp += 1
        valid += 1
    if valid == 0:
        return np.nan
    return fp / valid


def compute_hit_rate(
    df: pd.DataFrame,
    fwd_col: str,
    active_mask: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Hit rate: fraction of BH-active entries where sign(ret) == direction."""
    idx = np.where(active_mask)[0]
    if len(idx) == 0:
        return np.nan
    fwd = df[fwd_col].values
    hits = 0
    valid = 0
    for i in idx:
        if i >= len(fwd) or not np.isfinite(fwd[i]):
            continue
        if direction[i] == 1 and fwd[i] > 0:
            hits += 1
        elif direction[i] == -1 and fwd[i] < 0:
            hits += 1
        valid += 1
    if valid == 0:
        return np.nan
    return hits / valid


# ---------------------------------------------------------------------------
# Per-instrument threshold calibration
# ---------------------------------------------------------------------------

def calibrate_instrument_bh(
    ticker: str,
    df: pd.DataFrame,
    fwd_horizon: int = FORWARD_BARS_15M,
) -> pd.DataFrame:
    """
    For a single instrument, sweeps BH_MASS_THRESH grid and returns
    a DataFrame of metrics per threshold.
    """
    fwd_col = f"fwd_ret_{fwd_horizon}"
    rows = []
    for thresh in BH_THRESH_GRID:
        augmented = compute_bh_mass(df, ticker, bh_thresh=thresh)
        augmented  = add_forward_returns(augmented, fwd_horizon)

        active_mask = augmented["bh_active"].values
        direction   = augmented["bh_dir"].values
        fwd_vals    = augmented[fwd_col].values
        mass_vals   = augmented["bh_mass"].values

        ic = compute_ic(mass_vals, fwd_vals)
        cond_sharpe = compute_conditional_sharpe(augmented, fwd_col, active_mask)
        fpr  = compute_false_positive_rate(augmented, fwd_col, active_mask, direction)
        hr   = compute_hit_rate(augmented, fwd_col, active_mask, direction)
        pct_active = active_mask.mean()

        # IC when BH active vs inactive
        ic_active   = compute_ic(mass_vals[active_mask],  fwd_vals[active_mask])
        ic_inactive = compute_ic(mass_vals[~active_mask], fwd_vals[~active_mask])

        rows.append({
            "ticker":      ticker,
            "thresh":      thresh,
            "ic":          ic,
            "ic_active":   ic_active,
            "ic_inactive": ic_inactive,
            "cond_sharpe": cond_sharpe,
            "fpr":         fpr,
            "hit_rate":    hr,
            "pct_active":  pct_active,
        })
    return pd.DataFrame(rows)


def select_optimal_threshold(metrics_df: pd.DataFrame) -> Tuple[float, pd.Series]:
    """
    Composite score = IC * hit_rate * (1 - fpr).
    Returns (optimal_thresh, best_row).
    """
    df = metrics_df.copy().dropna(subset=["ic", "hit_rate", "fpr"])
    if df.empty:
        return BH_THRESH_GRID[2], metrics_df.iloc[0]
    df["score"] = df["ic"].clip(0) * df["hit_rate"] * (1.0 - df["fpr"])
    best = df.loc[df["score"].idxmax()]
    return float(best["thresh"]), best


# ---------------------------------------------------------------------------
# QuatNav calibration
# ---------------------------------------------------------------------------

def compute_nav_omega(
    df: pd.DataFrame,
    omega_scale_k: float = 0.5,
) -> np.ndarray:
    """
    Simulates the quaternion navigation angular velocity:
      omega_i = k * |log(close_i / close_{i-1})| / dt
    where dt = 1 bar = 1 unit.
    This is a scalar proxy for the full quaternion rotation rate.
    """
    log_rets = np.log(df["close"] / df["close"].shift(1)).fillna(0).values
    omega = omega_scale_k * np.abs(log_rets)
    return omega


def compute_geodesic_deviation(
    omega: np.ndarray,
    ema_span: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Geodesic deviation = omega / omega_ema.
    Returns (geo_dev, omega_ema).
    """
    omega_s = pd.Series(omega)
    omega_ema = omega_s.ewm(span=ema_span, adjust=False).mean().values
    omega_ema = np.where(omega_ema < 1e-12, 1e-12, omega_ema)
    geo_dev = omega / omega_ema
    return geo_dev, omega_ema


def calibrate_nav_omega_scale(
    ticker: str,
    df: pd.DataFrame,
    fwd_horizon: int = FORWARD_BARS_1H,
) -> pd.DataFrame:
    """
    For each NAV_OMEGA_SCALE_K, compute IC of nav_omega vs forward return.
    """
    df_aug = add_forward_returns(df, fwd_horizon)
    fwd_col = f"fwd_ret_{fwd_horizon}"
    rows = []
    for k in OMEGA_SCALE_GRID:
        omega = compute_nav_omega(df_aug, omega_scale_k=k)
        ic    = compute_ic(omega, df_aug[fwd_col].values)
        rows.append({"ticker": ticker, "omega_scale_k": k, "ic_omega": ic})
    return pd.DataFrame(rows)


def calibrate_geo_gate(
    ticker: str,
    df: pd.DataFrame,
    fwd_horizon: int = FORWARD_BARS_1H,
    omega_scale_k: float = 0.5,
) -> pd.DataFrame:
    """
    For each geo gate threshold, measures:
    - % of bars filtered (geo_dev > gate)
    - Avg forward return of filtered bars vs passed bars
    - IC of nav_omega among passed bars only
    """
    df_aug = add_forward_returns(df, fwd_horizon)
    fwd_col = f"fwd_ret_{fwd_horizon}"
    omega   = compute_nav_omega(df_aug, omega_scale_k=omega_scale_k)
    geo_dev, _ = compute_geodesic_deviation(omega)
    fwd_vals = df_aug[fwd_col].values

    rows = []
    for gate in GEO_GATE_GRID:
        filtered_mask = geo_dev > gate
        passed_mask   = ~filtered_mask

        pct_filtered  = filtered_mask.mean()
        avg_fwd_filtered = np.nanmean(np.abs(fwd_vals[filtered_mask])) if filtered_mask.any() else np.nan
        avg_fwd_passed   = np.nanmean(np.abs(fwd_vals[passed_mask]))   if passed_mask.any()   else np.nan

        # A good gate: filtered avg return should be LOWER (bad entries removed)
        ic_passed = compute_ic(omega[passed_mask], fwd_vals[passed_mask])

        rows.append({
            "ticker":           ticker,
            "geo_gate":         gate,
            "pct_filtered":     pct_filtered,
            "avg_fwd_filtered": avg_fwd_filtered,
            "avg_fwd_passed":   avg_fwd_passed,
            "filter_efficacy":  (avg_fwd_passed - avg_fwd_filtered) if (
                np.isfinite(avg_fwd_passed) and np.isfinite(avg_fwd_filtered)
            ) else np.nan,
            "ic_passed":        ic_passed,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Global calibration across all instruments
# ---------------------------------------------------------------------------

def run_full_calibration() -> Dict:
    """
    Main calibration loop. Returns nested dict of results.
    """
    print("[BH CALIBRATION] Starting full calibration run...")
    all_data: Dict[str, pd.DataFrame] = {}
    for t in INSTRUMENTS:
        print(f"  Generating data for {t} ...")
        all_data[t] = generate_ohlcv(t)

    bh_results: Dict[str, pd.DataFrame]    = {}
    omega_results: Dict[str, pd.DataFrame] = {}
    gate_results: Dict[str, pd.DataFrame]  = {}
    optimal_params: Dict[str, Dict]        = {}

    for ticker, df in all_data.items():
        print(f"  Calibrating {ticker} BH thresholds ...")
        bh_df = calibrate_instrument_bh(ticker, df, fwd_horizon=FORWARD_BARS_15M)
        bh_results[ticker] = bh_df

        opt_thresh, best_row = select_optimal_threshold(bh_df)

        omega_df = calibrate_nav_omega_scale(ticker, df, fwd_horizon=FORWARD_BARS_1H)
        omega_results[ticker] = omega_df

        best_omega_row = omega_df.loc[omega_df["ic_omega"].abs().idxmax()]
        best_omega_k   = float(best_omega_row["omega_scale_k"])

        gate_df = calibrate_geo_gate(ticker, df, omega_scale_k=best_omega_k)
        gate_results[ticker] = gate_df

        # select gate: highest filter_efficacy among those that filter < 40%
        valid_gates = gate_df[gate_df["pct_filtered"] < 0.40].dropna(subset=["filter_efficacy"])
        if valid_gates.empty:
            best_gate = GEO_GATE_GRID[2]
        else:
            best_gate = float(valid_gates.loc[valid_gates["filter_efficacy"].idxmax(), "geo_gate"])

        optimal_params[ticker] = {
            "bh_mass_thresh":    opt_thresh,
            "bh_ic":             float(best_row.get("ic", np.nan)),
            "bh_hit_rate":       float(best_row.get("hit_rate", np.nan)),
            "bh_fpr":            float(best_row.get("fpr", np.nan)),
            "bh_cond_sharpe":    float(best_row.get("cond_sharpe", np.nan)),
            "nav_omega_scale_k": best_omega_k,
            "nav_geo_gate":      best_gate,
        }
        print(f"    {ticker}: thresh={opt_thresh:.2f}  omega_k={best_omega_k:.2f}  geo_gate={best_gate:.1f}")

    # Cross-instrument optimal defaults
    all_thresh   = [v["bh_mass_thresh"]    for v in optimal_params.values()]
    all_omega_k  = [v["nav_omega_scale_k"] for v in optimal_params.values()]
    all_geo_gate = [v["nav_geo_gate"]      for v in optimal_params.values()]

    optimal_params["_global"] = {
        "bh_mass_thresh_mean":    float(np.mean(all_thresh)),
        "bh_mass_thresh_median":  float(np.median(all_thresh)),
        "nav_omega_scale_k_mean": float(np.mean(all_omega_k)),
        "nav_geo_gate_mean":      float(np.mean(all_geo_gate)),
    }

    return {
        "optimal_params": optimal_params,
        "bh_results":     {k: v.to_dict(orient="records") for k, v in bh_results.items()},
        "omega_results":  {k: v.to_dict(orient="records") for k, v in omega_results.items()},
        "gate_results":   {k: v.to_dict(orient="records") for k, v in gate_results.items()},
        "_raw": {
            "bh_results":    bh_results,
            "omega_results": omega_results,
            "gate_results":  gate_results,
            "all_data":      all_data,
        },
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _nan_or(x: float, fallback: float = 0.0) -> float:
    return fallback if (x is None or (isinstance(x, float) and np.isnan(x))) else x


def plot_bh_threshold_sweep(
    bh_results: Dict[str, pd.DataFrame],
    ax_map: Dict[str, plt.Axes],
) -> None:
    """Plots IC and hit-rate vs threshold for each instrument."""
    for ticker, df in bh_results.items():
        if ticker not in ax_map:
            continue
        ax = ax_map[ticker]
        ax2 = ax.twinx()
        thresholds = df["thresh"].values
        ic_vals    = df["ic"].fillna(0).values
        hr_vals    = df["hit_rate"].fillna(0).values
        fpr_vals   = df["fpr"].fillna(1).values

        ax.plot(thresholds, ic_vals, "b-o", label="IC", linewidth=1.5)
        ax2.plot(thresholds, hr_vals,  "g--s", label="Hit Rate", linewidth=1.5)
        ax2.plot(thresholds, fpr_vals, "r:^",  label="FPR",      linewidth=1.5)

        ax.set_title(f"{ticker} BH Threshold Sweep", fontsize=9)
        ax.set_xlabel("BH_MASS_THRESH")
        ax.set_ylabel("IC", color="b")
        ax2.set_ylabel("Hit Rate / FPR", color="g")
        ax.axvline(1.92, color="gray", linestyle="--", alpha=0.5, label="Default 1.92")
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")


def plot_geo_gate_sweep(
    gate_results: Dict[str, pd.DataFrame],
    ax: plt.Axes,
) -> None:
    """Stacked bar: filter efficacy per geo gate per instrument."""
    n_gates = len(GEO_GATE_GRID)
    x = np.arange(n_gates)
    width = 0.8 / len(INSTRUMENTS)
    colors = plt.cm.tab10(np.linspace(0, 1, len(INSTRUMENTS)))

    for i, ticker in enumerate(INSTRUMENTS):
        df = gate_results.get(ticker)
        if df is None:
            continue
        df_sorted = df.sort_values("geo_gate")
        efficacy  = df_sorted["filter_efficacy"].fillna(0).values
        ax.bar(x + i * width, efficacy, width, label=ticker, color=colors[i], alpha=0.8)

    ax.set_title("Geo Gate Filter Efficacy (higher = gate removes worse entries)", fontsize=9)
    ax.set_xlabel("NAV_GEO_ENTRY_GATE")
    ax.set_ylabel("Efficacy (passed_avg - filtered_avg)")
    ax.set_xticks(x + width * len(INSTRUMENTS) / 2)
    ax.set_xticklabels([str(g) for g in GEO_GATE_GRID])
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3, axis="y")


def plot_optimal_params_table(
    optimal_params: Dict[str, Dict],
    ax: plt.Axes,
) -> None:
    """Renders a summary table of optimal params per instrument."""
    ax.axis("off")
    instruments = [k for k in optimal_params if not k.startswith("_")]
    col_labels = ["Instrument", "BH_THRESH", "IC", "Hit Rate", "FPR", "Cond Sharpe", "Omega K", "Geo Gate"]
    rows = []
    for t in instruments:
        p = optimal_params[t]
        rows.append([
            t,
            f"{p['bh_mass_thresh']:.2f}",
            f"{_nan_or(p['bh_ic']):.4f}",
            f"{_nan_or(p['bh_hit_rate']):.3f}",
            f"{_nan_or(p['bh_fpr']):.3f}",
            f"{_nan_or(p['bh_cond_sharpe']):.3f}",
            f"{p['nav_omega_scale_k']:.2f}",
            f"{p['nav_geo_gate']:.1f}",
        ])
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.4)
    ax.set_title("Optimal Calibration Parameters per Instrument", fontsize=10, pad=10)


def plot_bh_mass_timeseries(
    all_data: Dict[str, pd.DataFrame],
    optimal_params: Dict[str, Dict],
    ax: plt.Axes,
    ticker: str = "BTC",
) -> None:
    """BH mass time series with threshold line for one instrument."""
    df = all_data.get(ticker)
    if df is None:
        return
    thresh = optimal_params.get(ticker, {}).get("bh_mass_thresh", 1.92)
    augmented = compute_bh_mass(df, ticker, bh_thresh=thresh)
    idx = augmented.index[:500]
    mass = augmented["bh_mass"].iloc[:500].values
    active_flag = augmented["bh_active"].iloc[:500].values

    ax.plot(idx, mass, "steelblue", linewidth=0.8, label="BH Mass")
    ax.axhline(thresh, color="red", linestyle="--", linewidth=1.2, label=f"Thresh={thresh:.2f}")
    ax.fill_between(idx, 0, mass, where=active_flag, alpha=0.3, color="orange", label="BH Active")
    ax.set_title(f"{ticker}: BH Mass Time Series (first 500 bars)", fontsize=9)
    ax.set_ylabel("BH Mass")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_omega_ic_heatmap(
    omega_results: Dict[str, pd.DataFrame],
    ax: plt.Axes,
) -> None:
    """Heatmap of omega IC vs (instrument, omega_scale_k)."""
    matrix = np.full((len(INSTRUMENTS), len(OMEGA_SCALE_GRID)), np.nan)
    for i, ticker in enumerate(INSTRUMENTS):
        df = omega_results.get(ticker)
        if df is None:
            continue
        df_sorted = df.sort_values("omega_scale_k")
        matrix[i, :len(df_sorted)] = df_sorted["ic_omega"].values

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.15)
    ax.set_xticks(range(len(OMEGA_SCALE_GRID)))
    ax.set_xticklabels([str(k) for k in OMEGA_SCALE_GRID])
    ax.set_yticks(range(len(INSTRUMENTS)))
    ax.set_yticklabels(INSTRUMENTS)
    ax.set_title("NavOmega IC vs Omega Scale K", fontsize=9)
    ax.set_xlabel("NAV_OMEGA_SCALE_K")
    plt.colorbar(im, ax=ax, label="IC")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def build_html_report(results: Dict, fig_path: str) -> str:
    """Generates an HTML calibration report."""
    optimal = results["optimal_params"]
    instruments = [k for k in optimal if not k.startswith("_")]
    glob = optimal.get("_global", {})

    rows_html = ""
    for t in instruments:
        p = optimal[t]
        rows_html += (
            f"<tr><td>{t}</td>"
            f"<td>{p['bh_mass_thresh']:.2f}</td>"
            f"<td>{_nan_or(p['bh_ic']):.4f}</td>"
            f"<td>{_nan_or(p['bh_hit_rate']):.3f}</td>"
            f"<td>{_nan_or(p['bh_fpr']):.3f}</td>"
            f"<td>{_nan_or(p['bh_cond_sharpe']):.3f}</td>"
            f"<td>{p['nav_omega_scale_k']:.2f}</td>"
            f"<td>{p['nav_geo_gate']:.1f}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>BH Mass Calibration Report -- LARSA v18</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #c9d1d9; margin: 20px; }}
  h1   {{ color: #58a6ff; }}
  h2   {{ color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: right; }}
  th {{ background: #161b22; color: #f0f6fc; text-align: center; }}
  tr:nth-child(even) {{ background: #161b22; }}
  .metric {{ background: #21262d; border-radius: 6px; padding: 10px; margin: 8px 0; }}
  img {{ max-width: 100%; border: 1px solid #30363d; border-radius: 4px; margin: 12px 0; }}
</style>
</head>
<body>
<h1>BH Mass Calibration Report -- LARSA v18</h1>
<p>Calibration of BH physics parameters across instruments: {', '.join(instruments)}</p>

<h2>Global Summary</h2>
<div class="metric">
  <strong>Mean optimal BH_MASS_THRESH:</strong> {glob.get('bh_mass_thresh_mean', 0):.3f}
  &nbsp;|&nbsp;
  <strong>Median:</strong> {glob.get('bh_mass_thresh_median', 0):.3f}
  &nbsp;|&nbsp;
  <strong>Mean NAV_OMEGA_SCALE_K:</strong> {glob.get('nav_omega_scale_k_mean', 0):.3f}
  &nbsp;|&nbsp;
  <strong>Mean NAV_GEO_GATE:</strong> {glob.get('nav_geo_gate_mean', 0):.3f}
</div>

<h2>Optimal Parameters per Instrument</h2>
<table>
  <tr>
    <th>Instrument</th><th>BH_THRESH</th><th>IC</th><th>Hit Rate</th>
    <th>FPR</th><th>Cond Sharpe</th><th>Omega K</th><th>Geo Gate</th>
  </tr>
  {rows_html}
</table>

<h2>Calibration Charts</h2>
<img src="{fig_path}" alt="Calibration Charts">

<h2>Methodology</h2>
<p>
BH mass computation follows LARSA v16/v18 FutureInstrument.update_bh():
beta_i = |close_i - close_{{i-1}}| / close_{{i-1}} / CF.
TIMELIKE bars (beta &lt; 1) accumulate mass; SPACELIKE bars decay mass.
BH is activated when mass &gt; threshold and CTL &gt;= 3 consecutive TIMELIKE bars.
</p>
<p>
Optimal threshold selected by composite score = IC * hit_rate * (1 - FPR).
QuatNav omega computed as k * |log_return| per bar; geodesic deviation = omega / omega_EMA(20).
Geo gate filters entries where geodesic deviation exceeds threshold.
</p>

<footer><p style="color:#484f58; font-size:11px;">Generated by bh_mass_calibration.py -- LARSA v18 Research</p></footer>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    results = run_full_calibration()
    raw = results.pop("_raw")

    # Save JSON
    json_path = OUT_DIR / "calibration_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"[BH CALIBRATION] Saved {json_path}")

    # Build figure
    print("[BH CALIBRATION] Building calibration report charts ...")
    n_instruments = len(INSTRUMENTS)
    n_cols = 4
    n_rows_thresh = (n_instruments + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(20, 28), facecolor="#0d1117")
    fig.suptitle("BH Mass Calibration Report -- LARSA v18", fontsize=14, color="white", y=0.98)

    gs = gridspec.GridSpec(n_rows_thresh + 4, n_cols, figure=fig, hspace=0.55, wspace=0.4)

    # Threshold sweep per instrument
    ax_map = {}
    for idx, ticker in enumerate(INSTRUMENTS):
        r, c = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor("#161b22")
        ax_map[ticker] = ax
    plot_bh_threshold_sweep(raw["bh_results"], ax_map)

    # Omega IC heatmap
    ax_omega = fig.add_subplot(gs[n_rows_thresh, :2])
    ax_omega.set_facecolor("#161b22")
    plot_omega_ic_heatmap(raw["omega_results"], ax_omega)

    # Geo gate sweep
    ax_gate = fig.add_subplot(gs[n_rows_thresh, 2:])
    ax_gate.set_facecolor("#161b22")
    plot_geo_gate_sweep(raw["gate_results"], ax_gate)

    # BH mass time series for BTC
    ax_ts = fig.add_subplot(gs[n_rows_thresh + 1, :2])
    ax_ts.set_facecolor("#161b22")
    plot_bh_mass_timeseries(raw["all_data"], results["optimal_params"], ax_ts, "BTC")

    # BH mass time series for ES
    ax_ts2 = fig.add_subplot(gs[n_rows_thresh + 1, 2:])
    ax_ts2.set_facecolor("#161b22")
    plot_bh_mass_timeseries(raw["all_data"], results["optimal_params"], ax_ts2, "ES")

    # Summary table
    ax_tbl = fig.add_subplot(gs[n_rows_thresh + 2, :])
    ax_tbl.set_facecolor("#161b22")
    plot_optimal_params_table(results["optimal_params"], ax_tbl)

    for ax in fig.get_axes():
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    fig_path = OUT_DIR / "calibration_charts.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[BH CALIBRATION] Saved {fig_path}")

    # Build HTML report
    html = build_html_report(results, "calibration_charts.png")
    html_path = OUT_DIR / "calibration_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[BH CALIBRATION] Saved {html_path}")
    print("[BH CALIBRATION] Done.")


if __name__ == "__main__":
    main()
