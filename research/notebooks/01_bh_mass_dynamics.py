"""
01_bh_mass_dynamics.py — Black Hole Mass Dynamics Analysis

Loads ES, NQ, YM, BTC data (real or synthetic), computes BH mass series for
each on daily/hourly/15m timeframes. Produces publication-quality analysis:
  - Mass timeseries with price overlay, colored by active/inactive
  - Distribution of mass values
  - Mass autocorrelation
  - How long does a typical BH last? How does mass peak? When does it collapse?
  - CF sensitivity on mass distribution per asset class
  - Outputs: research/outputs/bh_mass_analysis.png, bh_mass_stats.json

Run: python research/notebooks/01_bh_mass_dynamics.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "lib"))
sys.path.insert(0, str(_ROOT / "spacetime" / "engine"))

from srfm_core import MinkowskiClassifier, BlackHoleDetector

OUTPUTS = _ROOT / "research" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Asset configuration
# ─────────────────────────────────────────────────────────────────────────────

ASSETS = {
    "ES":  {"cf": 0.001,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95, "start_price": 4500.0, "sigma": 0.0008},
    "NQ":  {"cf": 0.0012, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95, "start_price": 15000.0, "sigma": 0.0010},
    "YM":  {"cf": 0.0008, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95, "start_price": 35000.0, "sigma": 0.0007},
    "BTC": {"cf": 0.005,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95, "start_price": 42000.0, "sigma": 0.005},
}

TF_CF_SCALE = {"1d": 5.0, "1h": 1.0, "15m": 0.35}
N_BARS_SYNTHETIC = 2000


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_or_generate(sym: str, cfg: dict, n: int = N_BARS_SYNTHETIC) -> pd.DataFrame:
    """Load real CSV if available; otherwise generate synthetic data."""
    csv_candidates = [
        _ROOT / "data" / f"{sym}_hourly_real.csv",
        _ROOT / "data" / f"{sym}_hourly.csv",
        _ROOT / "data" / f"{sym}.csv",
    ]
    for p in csv_candidates:
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                df.columns = [c.lower() for c in df.columns]
                if "close" in df.columns:
                    print(f"  Loaded {sym} from {p.name}: {len(df)} bars")
                    if "volume" not in df.columns:
                        df["volume"] = 50_000.0
                    return df.sort_index().dropna(subset=["close"])
            except Exception as e:
                print(f"  Failed to load {p}: {e}")

    # Generate synthetic
    print(f"  Generating synthetic {sym}: {n} bars")
    rng = np.random.default_rng(hash(sym) % 2**32)
    drift = 0.0001 + 0.0001 * rng.random()
    sigma = cfg["sigma"]
    prices = np.empty(n)
    prices[0] = cfg["start_price"]
    for i in range(1, n):
        prices[i] = prices[i-1] * max(1e-4, 1.0 + drift + sigma * rng.standard_normal())
    idx = pd.date_range("2021-01-04", periods=n, freq="1h")
    noise = 0.3 * sigma * np.abs(rng.standard_normal(n))
    return pd.DataFrame({
        "open":   prices * (1 - noise / 2),
        "high":   prices * (1 + noise),
        "low":    prices * (1 - noise),
        "close":  prices,
        "volume": np.full(n, 50_000.0),
    }, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# BH mass computation (all 3 timeframes)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bh_mass_series(
    df: pd.DataFrame,
    cf: float,
    bh_form: float = 1.5,
    bh_collapse: float = 1.0,
    bh_decay: float = 0.95,
    tf: str = "1h",
) -> pd.DataFrame:
    """
    Compute BH mass, activation, and direction series for a single timeframe.

    Returns DataFrame with columns:
      price, bh_mass, bh_active, bh_dir, ctl, bit, bh_event
    """
    # Resample to target timeframe
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if tf == "1d":
        df_tf = df.resample("1D").agg(agg).dropna(subset=["close"])
    elif tf == "1h":
        df_tf = df.resample("1h").agg(agg).dropna(subset=["close"])
    elif tf == "15m":
        df_tf = df.resample("15min").agg(agg).dropna(subset=["close"])
    else:
        df_tf = df

    scaled_cf = cf * TF_CF_SCALE.get(tf, 1.0)
    mc = MinkowskiClassifier(cf=scaled_cf)
    bh = BlackHoleDetector(bh_form, bh_collapse, bh_decay)

    closes  = df_tf["close"].values
    n       = len(closes)
    masses  = np.zeros(n)
    active  = np.zeros(n, dtype=bool)
    dirs    = np.zeros(n, dtype=int)
    ctls    = np.zeros(n, dtype=int)
    bits    = ["UNKNOWN"] * n
    events  = [""] * n

    mc.update(float(closes[0]))
    prev_active = False

    for i in range(1, n):
        bit     = mc.update(float(closes[i]))
        is_act  = bh.update(bit, float(closes[i]), float(closes[i-1]))
        masses[i] = bh.bh_mass
        active[i] = is_act
        dirs[i]   = bh.bh_dir
        ctls[i]   = bh.ctl
        bits[i]   = bit

        if is_act and not prev_active:
            events[i] = "formed"
        elif not is_act and prev_active:
            events[i] = "collapsed"
        prev_active = is_act

    return pd.DataFrame({
        "price":     closes,
        "bh_mass":   masses,
        "bh_active": active,
        "bh_dir":    dirs,
        "ctl":       ctls,
        "bit":       bits,
        "bh_event":  events,
    }, index=df_tf.index)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def analyze_bh_duration(mass_df: pd.DataFrame) -> dict:
    """
    Analyze BH well duration statistics.
    Returns dict with: mean_duration, median_duration, max_duration, n_wells.
    Duration measured in bars.
    """
    durations = []
    in_well = False
    start_bar = 0

    for i, (_, row) in enumerate(mass_df.iterrows()):
        if row["bh_active"] and not in_well:
            in_well = True
            start_bar = i
        elif not row["bh_active"] and in_well:
            durations.append(i - start_bar)
            in_well = False
    if in_well:
        durations.append(len(mass_df) - start_bar)

    if not durations:
        return {"mean_duration": 0, "median_duration": 0, "max_duration": 0, "n_wells": 0}

    return {
        "mean_duration":   float(np.mean(durations)),
        "median_duration": float(np.median(durations)),
        "max_duration":    float(np.max(durations)),
        "min_duration":    float(np.min(durations)),
        "std_duration":    float(np.std(durations)),
        "n_wells":         len(durations),
    }


def analyze_mass_distribution(mass_df: pd.DataFrame) -> dict:
    """Compute distribution statistics for BH mass values."""
    m = mass_df["bh_mass"].values
    m_active = m[mass_df["bh_active"].values]
    m_inactive = m[~mass_df["bh_active"].values]

    def _stats(arr: np.ndarray, label: str) -> dict:
        if len(arr) == 0:
            return {f"{label}_{k}": 0.0 for k in ("mean", "std", "p25", "p50", "p75", "p95", "max")}
        return {
            f"{label}_mean": float(np.mean(arr)),
            f"{label}_std":  float(np.std(arr)),
            f"{label}_p25":  float(np.percentile(arr, 25)),
            f"{label}_p50":  float(np.percentile(arr, 50)),
            f"{label}_p75":  float(np.percentile(arr, 75)),
            f"{label}_p95":  float(np.percentile(arr, 95)),
            f"{label}_max":  float(np.max(arr)),
        }

    result = {}
    result.update(_stats(m, "all"))
    result.update(_stats(m_active, "active"))
    result.update(_stats(m_inactive, "inactive"))
    return result


def analyze_mass_autocorrelation(mass_df: pd.DataFrame, max_lag: int = 20) -> dict:
    """Compute autocorrelation of BH mass series up to max_lag."""
    m = mass_df["bh_mass"].values
    m = m - m.mean()
    if m.std() < 1e-10:
        return {"autocorr": [0.0] * max_lag, "lags": list(range(1, max_lag + 1))}
    autocorr = []
    for lag in range(1, max_lag + 1):
        if len(m) > lag:
            c = float(np.corrcoef(m[:-lag], m[lag:])[0, 1])
            autocorr.append(c if math.isfinite(c) else 0.0)
        else:
            autocorr.append(0.0)
    return {"autocorr": autocorr, "lags": list(range(1, max_lag + 1))}


def analyze_peak_mass_on_formation(mass_df: pd.DataFrame) -> dict:
    """
    For each BH formation event, track the peak mass reached before collapse.
    Returns stats on peak mass per activation.
    """
    peak_masses = []
    in_well = False
    current_peak = 0.0

    for _, row in mass_df.iterrows():
        if row["bh_active"] and not in_well:
            in_well = True
            current_peak = row["bh_mass"]
        elif row["bh_active"] and in_well:
            current_peak = max(current_peak, row["bh_mass"])
        elif not row["bh_active"] and in_well:
            peak_masses.append(current_peak)
            in_well = False
            current_peak = 0.0
    if in_well:
        peak_masses.append(current_peak)

    if not peak_masses:
        return {"mean_peak_mass": 0.0, "max_peak_mass": 0.0, "n_activations": 0}
    return {
        "mean_peak_mass":   float(np.mean(peak_masses)),
        "median_peak_mass": float(np.median(peak_masses)),
        "max_peak_mass":    float(np.max(peak_masses)),
        "std_peak_mass":    float(np.std(peak_masses)),
        "n_activations":    len(peak_masses),
    }


def cf_sensitivity_analysis(
    df: pd.DataFrame,
    base_cf: float,
    n_cf: int = 10,
) -> dict:
    """
    Sweep CF from 0.3× to 3× baseline, compute:
      - Fraction of TIMELIKE bars
      - Number of BH activations
      - Mean mass when active
    """
    cf_multipliers = np.linspace(0.3, 3.0, n_cf)
    results = []
    closes  = df.resample("1h").last()["close"].dropna().values

    for mult in cf_multipliers:
        cf = base_cf * mult
        mc = MinkowskiClassifier(cf=cf)
        bh = BlackHoleDetector(1.5, 1.0, 0.95)
        mc.update(float(closes[0]))
        n_tl = 0; n_sl = 0; n_act = 0; mass_sum_active = 0.0; n_act_bars = 0
        prev_active = False
        for i in range(1, len(closes)):
            bit = mc.update(float(closes[i]))
            act = bh.update(bit, float(closes[i]), float(closes[i-1]))
            if bit == "TIMELIKE": n_tl += 1
            else: n_sl += 1
            if act and not prev_active: n_act += 1
            if act: mass_sum_active += bh.bh_mass; n_act_bars += 1
            prev_active = act
        total = n_tl + n_sl
        results.append({
            "cf":           float(cf),
            "cf_mult":      float(mult),
            "tl_fraction":  float(n_tl / total) if total > 0 else 0.0,
            "n_activations": n_act,
            "mean_active_mass": float(mass_sum_active / n_act_bars) if n_act_bars > 0 else 0.0,
        })
    return {"cf_sweep": results}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_mass_timeseries(
    sym: str,
    mass_df: pd.DataFrame,
    tf: str,
    ax_price,
    ax_mass,
):
    """Plot price with BH active shading, and BH mass timeseries."""
    times = np.arange(len(mass_df))
    price = mass_df["price"].values
    mass  = mass_df["bh_mass"].values
    active = mass_df["bh_active"].values

    # Price with BH shading
    ax_price.plot(times, price, color="black", linewidth=0.7, label=f"{sym} Price")
    for i in range(len(times)):
        if active[i]:
            color = "green" if mass_df.iloc[i]["bh_dir"] >= 0 else "red"
            ax_price.axvspan(times[i], times[i]+1, alpha=0.15, color=color)
    ax_price.set_title(f"{sym} {tf.upper()} — Price + BH Active", fontsize=9)
    ax_price.set_ylabel("Price")
    ax_price.legend(fontsize=7)
    ax_price.grid(alpha=0.3)

    # Mass timeseries
    ax_mass.fill_between(times, 0, mass, where=~active, alpha=0.4, color="gray",  label="Inactive")
    ax_mass.fill_between(times, 0, mass, where=active,  alpha=0.5, color="lime",  label="Active")
    ax_mass.axhline(1.5, color="orange", linestyle="--", linewidth=0.8, label="bh_form=1.5")
    ax_mass.axhline(1.0, color="red",    linestyle=":",  linewidth=0.8, label="bh_collapse=1.0")
    ax_mass.set_ylabel("BH Mass")
    ax_mass.legend(fontsize=6)
    ax_mass.grid(alpha=0.3)


def plot_mass_distribution(sym: str, mass_df: pd.DataFrame, ax):
    """Plot histogram of BH mass values, split by active/inactive."""
    import matplotlib.pyplot as plt
    m_all    = mass_df["bh_mass"].values
    m_active = m_all[mass_df["bh_active"].values]
    m_inactive = m_all[~mass_df["bh_active"].values]
    bins = np.linspace(0, min(10.0, m_all.max() + 0.5), 40)
    ax.hist(m_inactive, bins=bins, alpha=0.5, color="gray",  label="Inactive", density=True)
    ax.hist(m_active,   bins=bins, alpha=0.6, color="lime",  label="Active",   density=True)
    ax.axvline(1.5, color="orange", linestyle="--", label="bh_form")
    ax.set_title(f"{sym} Mass Distribution", fontsize=9)
    ax.set_xlabel("BH Mass"); ax.set_ylabel("Density")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)


def plot_autocorrelation(sym: str, autocorr_data: dict, ax):
    """Plot BH mass autocorrelation function."""
    lags = autocorr_data["lags"]
    acf  = autocorr_data["autocorr"]
    ax.bar(lags, acf, color="steelblue", alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=0.8)
    # 95% CI for white noise
    ci = 1.96 / math.sqrt(max(len(lags), 1))
    ax.axhline( ci, color="red", linestyle="--", linewidth=0.7, label="95% CI")
    ax.axhline(-ci, color="red", linestyle="--", linewidth=0.7)
    ax.set_title(f"{sym} Mass Autocorrelation", fontsize=9)
    ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)


def plot_cf_sensitivity(sym: str, cf_data: dict, ax_tl, ax_act):
    """Plot CF sweep: TL fraction and activation count vs CF multiplier."""
    sweep = cf_data["cf_sweep"]
    mults = [e["cf_mult"] for e in sweep]
    tl    = [e["tl_fraction"] for e in sweep]
    acts  = [e["n_activations"] for e in sweep]

    ax_tl.plot(mults, tl, "o-", color="steelblue", markersize=4)
    ax_tl.axhline(0.6, color="orange", linestyle="--", label="60% target")
    ax_tl.set_title(f"{sym} CF Sensitivity — Timelike Frac", fontsize=9)
    ax_tl.set_xlabel("CF Multiplier"); ax_tl.set_ylabel("Timelike Fraction")
    ax_tl.legend(fontsize=7); ax_tl.grid(alpha=0.3)

    ax_act.plot(mults, acts, "s-", color="firebrick", markersize=4)
    ax_act.set_title(f"{sym} CF Sensitivity — Activations", fontsize=9)
    ax_act.set_xlabel("CF Multiplier"); ax_act.set_ylabel("N Activations")
    ax_act.grid(alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("01_bh_mass_dynamics.py")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_PLOT = True
    except ImportError:
        print("[WARN] matplotlib not available; skipping plots")
        HAS_PLOT = False

    all_stats = {}

    for sym, cfg in ASSETS.items():
        print(f"\n[{sym}] Loading data...")
        df = load_or_generate(sym, cfg)
        print(f"  {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")

        sym_stats = {}

        for tf in ("1d", "1h", "15m"):
            print(f"  Computing BH mass [{tf}]...")
            cf_tf = cfg["cf"] * TF_CF_SCALE[tf]
            mass_df = compute_bh_mass_series(
                df, cf=cf_tf,
                bh_form=cfg["bh_form"],
                bh_collapse=cfg["bh_collapse"],
                bh_decay=cfg["bh_decay"],
                tf=tf,
            )

            n_bars    = len(mass_df)
            n_active  = int(mass_df["bh_active"].sum())
            pct_act   = 100.0 * n_active / max(1, n_bars)
            n_events  = (mass_df["bh_event"] == "formed").sum()

            print(f"    Bars={n_bars}, Active={n_active} ({pct_act:.1f}%), Formations={n_events}")

            # Analysis
            duration_stats = analyze_bh_duration(mass_df)
            mass_dist      = analyze_mass_distribution(mass_df)
            autocorr       = analyze_mass_autocorrelation(mass_df, max_lag=10)
            peak_stats     = analyze_peak_mass_on_formation(mass_df)

            sym_stats[tf] = {
                "n_bars":    n_bars,
                "n_active":  n_active,
                "pct_active": round(pct_act, 2),
                "n_formations": int(n_events),
                **duration_stats,
                **mass_dist,
                **peak_stats,
                "autocorr_lag1": autocorr["autocorr"][0] if autocorr["autocorr"] else 0.0,
                "autocorr_lag5": autocorr["autocorr"][4] if len(autocorr["autocorr"]) > 4 else 0.0,
            }

            # Print key stats
            if duration_stats["n_wells"] > 0:
                print(f"    Wells: n={duration_stats['n_wells']}, "
                      f"mean_duration={duration_stats['mean_duration']:.1f} bars, "
                      f"median={duration_stats['median_duration']:.1f}")
            if peak_stats["n_activations"] > 0:
                print(f"    Peak mass: mean={peak_stats['mean_peak_mass']:.2f}, "
                      f"max={peak_stats['max_peak_mass']:.2f}")

        # CF sensitivity analysis (on hourly)
        print(f"  Running CF sensitivity analysis...")
        cf_data = cf_sensitivity_analysis(df, cfg["cf"], n_cf=12)
        sym_stats["cf_sensitivity"] = cf_data["cf_sweep"]

        # Find CF giving ~60% timelike
        best_cf = None
        best_gap = 1.0
        for entry in cf_data["cf_sweep"]:
            gap = abs(entry["tl_fraction"] - 0.60)
            if gap < best_gap:
                best_gap = gap
                best_cf = entry["cf"]
        sym_stats["optimal_cf_for_60pct_tl"] = best_cf
        print(f"  Optimal CF for 60% timelike: {best_cf:.6f} (vs base {cfg['cf']:.6f})")

        all_stats[sym] = sym_stats

    # ── Plotting ─────────────────────────────────────────────────────────────
    if HAS_PLOT:
        n_assets = len(ASSETS)
        fig, axes = plt.subplots(n_assets * 2, 4, figsize=(20, n_assets * 8))
        fig.suptitle("BH Mass Dynamics Analysis", fontsize=14, fontweight="bold")

        for row_idx, (sym, cfg) in enumerate(ASSETS.items()):
            df = load_or_generate(sym, cfg)

            for col_idx, tf in enumerate(("1d", "1h", "15m")):
                cf_tf   = cfg["cf"] * TF_CF_SCALE[tf]
                mass_df = compute_bh_mass_series(df, cf=cf_tf,
                    bh_form=cfg["bh_form"], bh_collapse=cfg["bh_collapse"],
                    bh_decay=cfg["bh_decay"], tf=tf)

                row_price = row_idx * 2
                row_mass  = row_idx * 2 + 1

                ax_p = axes[row_price, col_idx]
                ax_m = axes[row_mass,  col_idx]
                plot_mass_timeseries(sym, mass_df, tf, ax_p, ax_m)

            # Distribution plot in col 3
            cf_tf = cfg["cf"]
            mass_df_1h = compute_bh_mass_series(df, cf=cf_tf, tf="1h",
                bh_form=cfg["bh_form"], bh_collapse=cfg["bh_collapse"],
                bh_decay=cfg["bh_decay"])
            ax_dist = axes[row_idx * 2, 3]
            ax_acf  = axes[row_idx * 2 + 1, 3]
            plot_mass_distribution(sym, mass_df_1h, ax_dist)
            autocorr_data = analyze_mass_autocorrelation(mass_df_1h)
            plot_autocorrelation(sym, autocorr_data, ax_acf)

        plt.tight_layout()
        out_png = OUTPUTS / "bh_mass_analysis.png"
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"\nPlot saved → {out_png}")
        plt.close()

        # CF sensitivity plot
        fig2, axes2 = plt.subplots(len(ASSETS), 2, figsize=(12, len(ASSETS) * 3))
        if len(ASSETS) == 1:
            axes2 = axes2.reshape(1, -1)
        for i, (sym, cfg) in enumerate(ASSETS.items()):
            df = load_or_generate(sym, cfg)
            cf_data = cf_sensitivity_analysis(df, cfg["cf"], n_cf=15)
            plot_cf_sensitivity(sym, cf_data, axes2[i, 0], axes2[i, 1])
        fig2.suptitle("CF Sensitivity Analysis", fontsize=12)
        plt.tight_layout()
        out_cf = OUTPUTS / "cf_sensitivity.png"
        plt.savefig(out_cf, dpi=120)
        plt.close()
        print(f"CF sensitivity plot → {out_cf}")

    # ── Save JSON stats ───────────────────────────────────────────────────────
    out_json = OUTPUTS / "bh_mass_stats.json"
    # Sanitize for JSON serialization
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj) if math.isfinite(float(obj)) else None
        return obj

    with open(out_json, "w") as f:
        json.dump(_sanitize(all_stats), f, indent=2)
    print(f"Stats saved → {out_json}")

    # ── Summary report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY: BH Mass Dynamics")
    print("=" * 60)
    for sym in ASSETS:
        stats = all_stats.get(sym, {})
        h1_stats = stats.get("1h", {})
        print(f"\n{sym}:")
        print(f"  Hourly — Active {h1_stats.get('pct_active', 0):.1f}% of bars")
        print(f"  Formations: {h1_stats.get('n_formations', 0)}")
        print(f"  Mean well duration: {h1_stats.get('mean_duration', 0):.1f} bars")
        print(f"  Mean peak mass:     {h1_stats.get('mean_peak_mass', 0):.2f}")
        print(f"  Mass ACF(lag=1):    {h1_stats.get('autocorr_lag1', 0):.3f}")
        opt_cf = stats.get("optimal_cf_for_60pct_tl")
        if opt_cf:
            print(f"  Optimal CF (60% TL): {opt_cf:.6f}")


if __name__ == "__main__":
    main()
