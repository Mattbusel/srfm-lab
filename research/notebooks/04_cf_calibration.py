"""
04_cf_calibration.py — CF (Causality Factor) Calibration Study

For each of 20 instruments:
  - Load hourly data (real or synthetic)
  - Sweep CF from 0.001 to 0.05 (100 values)
  - At each CF: compute % of bars that are TIMELIKE
  - Objective 1: CF that gives 60% TIMELIKE bars (target ratio)
  - Objective 2: CF that maximizes Sharpe on backtest

Compare: optimal CF by coverage vs by Sharpe — do they agree?
Fit power law: CF ~ sigma^alpha (does optimal CF scale with vol?)

Outputs: research/outputs/cf_calibration.png, calibrated_cfs.yaml

Run: python research/notebooks/04_cf_calibration.py
"""

from __future__ import annotations

import json
import math
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
# 20-instrument universe with known parameters
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSE = {
    "ES":     {"start_price": 4500.0,  "annual_vol": 0.15,  "asset_class": "equity_index"},
    "NQ":     {"start_price": 15000.0, "annual_vol": 0.20,  "asset_class": "equity_index"},
    "YM":     {"start_price": 35000.0, "annual_vol": 0.14,  "asset_class": "equity_index"},
    "RTY":    {"start_price": 2000.0,  "annual_vol": 0.22,  "asset_class": "equity_index"},
    "SPY":    {"start_price": 450.0,   "annual_vol": 0.15,  "asset_class": "equity_index"},
    "QQQ":    {"start_price": 380.0,   "annual_vol": 0.21,  "asset_class": "equity_index"},
    "BTC":    {"start_price": 42000.0, "annual_vol": 0.80,  "asset_class": "crypto"},
    "ETH":    {"start_price": 2500.0,  "annual_vol": 0.90,  "asset_class": "crypto"},
    "SOL":    {"start_price": 100.0,   "annual_vol": 1.20,  "asset_class": "crypto"},
    "GC":     {"start_price": 1900.0,  "annual_vol": 0.12,  "asset_class": "commodity"},
    "SI":     {"start_price": 25.0,    "annual_vol": 0.25,  "asset_class": "commodity"},
    "CL":     {"start_price": 75.0,    "annual_vol": 0.35,  "asset_class": "commodity"},
    "NG":     {"start_price": 3.50,    "annual_vol": 0.55,  "asset_class": "commodity"},
    "ZB":     {"start_price": 115.0,   "annual_vol": 0.07,  "asset_class": "bond"},
    "ZN":     {"start_price": 110.0,   "annual_vol": 0.05,  "asset_class": "bond"},
    "EURUSD": {"start_price": 1.08,    "annual_vol": 0.07,  "asset_class": "forex"},
    "GBPUSD": {"start_price": 1.25,    "annual_vol": 0.08,  "asset_class": "forex"},
    "USDJPY": {"start_price": 150.0,   "annual_vol": 0.07,  "asset_class": "forex"},
    "VIX":    {"start_price": 18.0,    "annual_vol": 1.00,  "asset_class": "volatility"},
    "GLD":    {"start_price": 185.0,   "annual_vol": 0.12,  "asset_class": "commodity"},
}

N_BARS   = 2000
N_CF_PTS = 100
CF_MIN   = 0.0001
CF_MAX   = 0.05
TARGET_TL_FRAC = 0.60


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_hourly(sym: str, cfg: dict, n: int = N_BARS) -> np.ndarray:
    """Generate synthetic hourly close prices for a given instrument."""
    # Try real CSV first
    for csv_name in (f"{sym}_hourly_real.csv", f"{sym}_hourly.csv", f"{sym}.csv"):
        p = _ROOT / "data" / csv_name
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                df.columns = [c.lower() for c in df.columns]
                if "close" in df.columns:
                    return df["close"].dropna().values[:n]
            except Exception:
                pass

    rng = np.random.default_rng(hash(sym) % 2**32)
    annual_vol = cfg["annual_vol"]
    hourly_vol = annual_vol / math.sqrt(252 * 6.5)
    drift = 0.05 / (252 * 6.5)   # 5% annual drift

    closes = np.empty(n)
    closes[0] = cfg["start_price"]
    for i in range(1, n):
        ret = drift + hourly_vol * rng.standard_normal()
        closes[i] = closes[i-1] * max(1e-4, 1.0 + np.clip(ret, -0.10, 0.10))
    return closes


# ─────────────────────────────────────────────────────────────────────────────
# CF sweep
# ─────────────────────────────────────────────────────────────────────────────

def compute_tl_fraction(closes: np.ndarray, cf: float) -> float:
    """Fraction of bars classified as TIMELIKE at given CF."""
    mc = MinkowskiClassifier(cf=cf)
    mc.update(float(closes[0]))
    n_tl = 0
    for i in range(1, len(closes)):
        if mc.update(float(closes[i])) == "TIMELIKE":
            n_tl += 1
    return n_tl / max(1, len(closes) - 1)


def compute_bh_activations(closes: np.ndarray, cf: float,
                            bh_form: float = 1.5) -> int:
    """Count BH activations at given CF."""
    mc = MinkowskiClassifier(cf=cf)
    bh = BlackHoleDetector(bh_form, 1.0, 0.95)
    mc.update(float(closes[0]))
    n_act = 0; prev_active = False
    for i in range(1, len(closes)):
        bit = mc.update(float(closes[i]))
        act = bh.update(bit, float(closes[i]), float(closes[i-1]))
        if act and not prev_active:
            n_act += 1
        prev_active = act
    return n_act


def simple_sharpe_from_bh(closes: np.ndarray, cf: float,
                            bh_form: float = 1.5) -> float:
    """
    Compute a simple Sharpe proxy: go long when BH is active.
    Returns annualized Sharpe of this simple strategy.
    """
    mc = MinkowskiClassifier(cf=cf)
    bh = BlackHoleDetector(bh_form, 1.0, 0.95)
    mc.update(float(closes[0]))
    strat_rets = []
    for i in range(1, len(closes)):
        bar_ret = (closes[i] - closes[i-1]) / (closes[i-1] + 1e-9)
        bit = mc.update(float(closes[i]))
        bh.update(bit, float(closes[i]), float(closes[i-1]))
        if bh.bh_active and bh.bh_dir > 0:
            strat_rets.append(bar_ret)
        else:
            strat_rets.append(0.0)
    if not strat_rets or np.std(strat_rets) < 1e-10:
        return 0.0
    arr = np.array(strat_rets)
    return float(arr.mean() / arr.std() * math.sqrt(252 * 6.5))


def sweep_cf(sym: str, closes: np.ndarray, cfg: dict, n_cf: int = N_CF_PTS) -> dict:
    """
    Sweep CF values, compute TL fraction and Sharpe at each.
    Returns dict with sweep data and optimal CF values.
    """
    cf_values = np.logspace(
        math.log10(CF_MIN), math.log10(CF_MAX), n_cf
    )

    tl_fracs  = []
    sharpes   = []
    n_acts    = []

    for cf in cf_values:
        tl = compute_tl_fraction(closes, float(cf))
        sh = simple_sharpe_from_bh(closes, float(cf))
        na = compute_bh_activations(closes, float(cf))
        tl_fracs.append(tl)
        sharpes.append(sh)
        n_acts.append(na)

    # Find optimal CF for 60% TL
    dists = [abs(tl - TARGET_TL_FRAC) for tl in tl_fracs]
    idx_tl = int(np.argmin(dists))
    cf_opt_tl = float(cf_values[idx_tl])

    # Find optimal CF for max Sharpe
    idx_sh = int(np.argmax(sharpes))
    cf_opt_sharpe = float(cf_values[idx_sh])

    # Realized hourly vol (annualized)
    log_rets = np.diff(np.log(closes + 1e-9))
    realized_vol = float(np.std(log_rets) * math.sqrt(252 * 6.5))

    return {
        "sym":            sym,
        "asset_class":    cfg["asset_class"],
        "annual_vol":     cfg["annual_vol"],
        "realized_vol":   realized_vol,
        "cf_opt_tl60":    cf_opt_tl,
        "cf_opt_sharpe":  cf_opt_sharpe,
        "tl_at_opt_cf":   tl_fracs[idx_tl],
        "sharpe_at_opt":  sharpes[idx_sh],
        "agree":          abs(cf_opt_tl - cf_opt_sharpe) / (cf_opt_sharpe + 1e-10) < 0.5,
        "sweep": {
            "cf":       [float(x) for x in cf_values],
            "tl_frac":  tl_fracs,
            "sharpe":   sharpes,
            "n_activations": n_acts,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Power law fit: CF_opt ~ vol^alpha
# ─────────────────────────────────────────────────────────────────────────────

def fit_power_law(vols: np.ndarray, cf_opts: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit log(CF) = alpha * log(vol) + log(C)
    Returns: (alpha, C, r_squared)
    """
    valid = (vols > 0) & (cf_opts > 0)
    if valid.sum() < 3:
        return 0.0, 0.001, 0.0
    log_vol = np.log(vols[valid])
    log_cf  = np.log(cf_opts[valid])
    # Least squares
    X = np.column_stack([log_vol, np.ones(len(log_vol))])
    try:
        coeffs, residuals, _, _ = np.linalg.lstsq(X, log_cf, rcond=None)
        alpha, log_C = coeffs
        C = math.exp(float(log_C))
        # R²
        ss_res = float(np.sum((log_cf - X @ coeffs)**2))
        ss_tot = float(np.sum((log_cf - log_cf.mean())**2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        return float(alpha), float(C), float(r2)
    except Exception:
        return 0.0, 0.001, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_cf_sweep_grid(results: List[dict], fig, axes_flat):
    """Plot CF sweep curves for each asset (TL fraction and Sharpe)."""
    for i, res in enumerate(results[:len(axes_flat)//2]):
        ax_tl = axes_flat[i * 2]
        ax_sh = axes_flat[i * 2 + 1]
        sweep = res["sweep"]
        cfs   = sweep["cf"]
        tl    = sweep["tl_frac"]
        sh    = sweep["sharpe"]

        ax_tl.plot(cfs, tl, "-", color="steelblue", linewidth=1.0)
        ax_tl.axhline(TARGET_TL_FRAC, color="orange", linestyle="--", linewidth=0.7)
        ax_tl.axvline(res["cf_opt_tl60"],   color="red", linestyle=":", linewidth=0.8, label=f"CF*={res['cf_opt_tl60']:.5f}")
        ax_tl.set_title(f"{res['sym']} TL Frac", fontsize=7)
        ax_tl.set_xscale("log")
        ax_tl.legend(fontsize=5); ax_tl.grid(alpha=0.3)

        ax_sh.plot(cfs, sh, "-", color="firebrick", linewidth=1.0)
        ax_sh.axvline(res["cf_opt_sharpe"], color="red", linestyle=":", linewidth=0.8, label=f"CF*={res['cf_opt_sharpe']:.5f}")
        ax_sh.axhline(0, color="black", linewidth=0.7)
        ax_sh.set_title(f"{res['sym']} Sharpe", fontsize=7)
        ax_sh.set_xscale("log")
        ax_sh.legend(fontsize=5); ax_sh.grid(alpha=0.3)


def plot_power_law(vols, cf_opts, alpha, C, r2, ax):
    """Scatter plot of vol vs optimal CF with power-law fit."""
    ax.scatter(vols, cf_opts, c="steelblue", s=40, alpha=0.7, zorder=3)
    v_range = np.linspace(min(vols) * 0.8, max(vols) * 1.2, 100)
    ax.plot(v_range, C * v_range**alpha, "r-", linewidth=1.2,
            label=f"CF = {C:.5f} × σ^{alpha:.2f}\n R²={r2:.3f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Annualized Volatility"); ax.set_ylabel("Optimal CF")
    ax.set_title("Power Law: Optimal CF vs Volatility", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("04_cf_calibration.py")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_PLOT = True
    except ImportError:
        HAS_PLOT = False

    syms = list(UNIVERSE.keys())
    print(f"\nCalibrating CF for {len(syms)} instruments...")

    all_results = []
    for sym in syms:
        cfg    = UNIVERSE[sym]
        print(f"  {sym:8s} ({cfg['asset_class']:15s}) σ={cfg['annual_vol']:.2f}...")
        closes = generate_hourly(sym, cfg, n=N_BARS)
        result = sweep_cf(sym, closes, cfg, n_cf=N_CF_PTS)
        all_results.append(result)
        agree = "✓" if result["agree"] else "✗"
        print(f"    CF_opt(TL60%)={result['cf_opt_tl60']:.5f}  "
              f"CF_opt(Sharpe)={result['cf_opt_sharpe']:.5f}  "
              f"Agree={agree}")

    # Power law fit
    vols    = np.array([r["annual_vol"] for r in all_results])
    cf_tl   = np.array([r["cf_opt_tl60"] for r in all_results])
    cf_sh   = np.array([r["cf_opt_sharpe"] for r in all_results])

    alpha_tl, C_tl, r2_tl = fit_power_law(vols, cf_tl)
    alpha_sh, C_sh, r2_sh = fit_power_law(vols, cf_sh)

    print(f"\nPower law (TL60% target): CF = {C_tl:.5f} × σ^{alpha_tl:.3f}  R²={r2_tl:.3f}")
    print(f"Power law (Sharpe target): CF = {C_sh:.5f} × σ^{alpha_sh:.3f}  R²={r2_sh:.3f}")

    # Print calibrated CF table
    print("\n" + "=" * 60)
    print(f"{'SYM':8s} {'Asset Class':15s} {'σ':6s} {'CF(TL60)':10s} {'CF(Sharpe)':10s} {'Agree':6s}")
    print("-" * 60)
    for r in all_results:
        agree = "YES" if r["agree"] else "NO"
        print(f"{r['sym']:8s} {r['asset_class']:15s} {r['annual_vol']:6.2f} "
              f"{r['cf_opt_tl60']:10.5f} {r['cf_opt_sharpe']:10.5f} {agree:6s}")

    n_agree = sum(1 for r in all_results if r["agree"])
    print(f"\nAgreement rate: {n_agree}/{len(all_results)} = {n_agree/len(all_results):.1%}")

    # Save YAML (as JSON-compatible dict) + JSON
    calibrated = {}
    for r in all_results:
        calibrated[r["sym"]] = {
            "cf_optimal_tl60":   round(r["cf_opt_tl60"], 6),
            "cf_optimal_sharpe": round(r["cf_opt_sharpe"], 6),
            "annual_vol":        r["annual_vol"],
            "asset_class":       r["asset_class"],
            "agree":             r["agree"],
        }

    yaml_path = OUTPUTS / "calibrated_cfs.yaml"
    with open(yaml_path, "w") as f:
        f.write("# Calibrated CF values per instrument\n")
        f.write(f"# Power law (TL60%): CF = {C_tl:.6f} * sigma^{alpha_tl:.4f} (R2={r2_tl:.3f})\n")
        f.write(f"# Power law (Sharpe): CF = {C_sh:.6f} * sigma^{alpha_sh:.4f} (R2={r2_sh:.3f})\n\n")
        for sym, vals in calibrated.items():
            f.write(f"{sym}:\n")
            for k, v in vals.items():
                f.write(f"  {k}: {v}\n")
    print(f"\nCalibrated CFs → {yaml_path}")

    # JSON stats
    def _clean(obj):
        if isinstance(obj, (float, np.floating)):
            v = float(obj); return v if math.isfinite(v) else None
        if isinstance(obj, (int, np.integer)): return int(obj)
        if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_clean(v) for v in obj]
        return obj

    stats_json = {
        "power_law_tl60":  {"alpha": float(alpha_tl), "C": float(C_tl), "r2": float(r2_tl)},
        "power_law_sharpe":{"alpha": float(alpha_sh), "C": float(C_sh), "r2": float(r2_sh)},
        "calibrated_cfs":  calibrated,
        "agreement_rate":  n_agree / len(all_results),
    }
    json_path = OUTPUTS / "cf_calibration_stats.json"
    with open(json_path, "w") as f:
        json.dump(_clean(stats_json), f, indent=2)
    print(f"Stats → {json_path}")

    # Plotting
    if HAS_PLOT:
        n_syms = len(all_results)
        n_cols  = 8
        n_rows  = math.ceil(n_syms / (n_cols // 2)) * 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
        fig.suptitle("CF Calibration — TL Fraction & Sharpe Sweep", fontsize=12)
        axes_flat = axes.flatten()
        plot_cf_sweep_grid(all_results, fig, axes_flat)
        # Hide unused axes
        for ax in axes_flat[n_syms * 2:]:
            ax.set_visible(False)
        plt.tight_layout()
        out = OUTPUTS / "cf_calibration.png"
        plt.savefig(out, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"CF sweep plot → {out}")

        # Power law plot
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        plot_power_law(vols, cf_tl, alpha_tl, C_tl, r2_tl, axes2[0])
        axes2[0].set_title("Power Law: CF (TL60%) vs σ")
        plot_power_law(vols, cf_sh, alpha_sh, C_sh, r2_sh, axes2[1])
        axes2[1].set_title("Power Law: CF (Sharpe) vs σ")

        # Add symbol labels
        for ax, cf_arr in zip(axes2, [cf_tl, cf_sh]):
            for i, r in enumerate(all_results):
                ax.annotate(r["sym"], (vols[i], cf_arr[i]), fontsize=5, alpha=0.7)

        plt.tight_layout()
        out2 = OUTPUTS / "cf_power_law.png"
        plt.savefig(out2, dpi=120)
        plt.close()
        print(f"Power law plot → {out2}")


if __name__ == "__main__":
    main()
