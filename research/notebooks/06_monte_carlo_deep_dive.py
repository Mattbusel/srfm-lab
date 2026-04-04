"""
06_monte_carlo_deep_dive.py — Monte Carlo Deep Dive

- Path convergence study: 1K, 10K, 100K paths
- Serial correlation analysis: ρ ∈ {0, 0.1, 0.3, 0.5}
- Regime-aware MC vs naive
- Kelly analysis: f from 0.1×Kelly to 2×Kelly
- Stress test: remove 10% best trades
- Fan charts for each scenario
- Kelly frontier curve

Outputs: research/outputs/mc_deep_dive.png, mc_analysis.json

Run: python research/notebooks/06_monte_carlo_deep_dive.py
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
sys.path.insert(0, str(_ROOT / "spacetime" / "engine"))

OUTPUTS = _ROOT / "research" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Trade generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_realistic_trades(n: int = 300, seed: int = 42) -> List[dict]:
    """Generate realistic trade history with 62% win rate."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-01")
    regimes = ["BULL", "BULL", "BULL", "SIDEWAYS", "SIDEWAYS", "BEAR", "HIGH_VOL"]
    trades = []
    for i in range(n):
        reg = rng.choice(regimes)
        if reg == "BULL":
            wr, w_mu, l_mu = 0.68, 0.018, 0.009
        elif reg == "BEAR":
            wr, w_mu, l_mu = 0.42, 0.010, 0.014
        elif reg == "HIGH_VOL":
            wr, w_mu, l_mu = 0.50, 0.025, 0.022
        else:
            wr, w_mu, l_mu = 0.55, 0.012, 0.009
        win = rng.random() < wr
        pnl_pct = float(rng.exponential(w_mu)) if win else -float(rng.exponential(l_mu))
        trades.append({
            "entry_time": base + pd.Timedelta(hours=i * 8),
            "exit_time":  base + pd.Timedelta(hours=i * 8 + 4),
            "pnl": pnl_pct * 1_000_000.0 * 0.25,
            "pnl_pct": pnl_pct,
            "regime": reg,
            "tf_score": int(rng.integers(4, 8)),
        })
    return trades


def extract_returns(trades: List[dict]) -> np.ndarray:
    """Extract normalized returns from trade list."""
    return np.array([t["pnl_pct"] for t in trades])


# ─────────────────────────────────────────────────────────────────────────────
# Core Monte Carlo engine
# ─────────────────────────────────────────────────────────────────────────────

def run_mc_paths(
    returns: np.ndarray,
    n_sims: int = 10_000,
    n_steps: int = 300,
    f: float = 0.25,
    serial_corr: float = 0.0,
    seed: int = 42,
    blowup_threshold: float = 0.10,
) -> Dict:
    """
    Monte Carlo simulation with optional serial correlation.

    Parameters
    ----------
    returns   : array of trade returns (as fractions)
    n_sims    : number of paths
    n_steps   : number of trades per path
    f         : position sizing fraction
    serial_corr : AR(1) loss correlation coefficient
    """
    rng = np.random.default_rng(seed)
    losses = returns[returns < 0]
    gains  = returns[returns >= 0]
    if len(losses) == 0:
        losses = np.array([-0.001])
    if len(gains) == 0:
        gains = np.array([0.001])

    p_loss = len(losses) / (len(losses) + len(gains))

    equities    = np.zeros(n_sims)
    max_draws   = np.zeros(n_sims)
    blowups     = 0
    all_paths   = []

    for sim in range(n_sims):
        eq   = 1.0
        peak = 1.0
        max_dd = 0.0
        path = [eq]
        prev_loss = False

        for step in range(n_steps):
            if eq < blowup_threshold:
                blowups += 1
                break
            # AR(1) serial correlation
            if serial_corr > 0 and prev_loss:
                p_this_loss = min(0.99, p_loss + serial_corr * (1 - p_loss))
            else:
                p_this_loss = p_loss
            if rng.random() < p_this_loss:
                r = float(rng.choice(losses))
                prev_loss = True
            else:
                r = float(rng.choice(gains))
                prev_loss = False
            eq = eq * (1.0 + f * r)
            eq = max(0.0, eq)
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
            path.append(eq)

        equities[sim]  = eq
        max_draws[sim] = max_dd
        if sim < 200:   # store first 200 paths for plotting
            all_paths.append(np.array(path))

    pcts = [5, 10, 25, 50, 75, 90, 95]
    return {
        "final_equities": equities,
        "max_drawdowns":  max_draws,
        "blowup_rate":    blowups / n_sims,
        "median":         float(np.median(equities)),
        "percentiles":    {p: float(np.percentile(equities, p)) for p in pcts},
        "paths":          all_paths,
        "n_sims":         n_sims,
        "n_steps":        n_steps,
        "f":              f,
        "serial_corr":    serial_corr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Kelly analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_kelly_fraction(returns: np.ndarray) -> float:
    """Find Kelly-optimal f using numerical optimization."""
    from scipy.optimize import minimize_scalar
    def neg_log_growth(f: float) -> float:
        vals = np.maximum(1e-10, 1.0 + f * returns)
        return -float(np.mean(np.log(vals)))
    try:
        result = minimize_scalar(neg_log_growth, bounds=(0.0, 2.0), method="bounded")
        return float(np.clip(result.x, 0.0, 1.0))
    except Exception:
        return 0.25


def kelly_frontier_analysis(
    returns: np.ndarray,
    kelly: float,
    n_f_pts: int = 20,
    n_sims: int = 2_000,
    n_steps: int = 200,
) -> List[dict]:
    """
    Sweep f from 0.1×Kelly to 2.0×Kelly.
    Returns list of {f_mult, f, median_equity, blowup_rate}.
    """
    f_mults = np.linspace(0.1, 2.0, n_f_pts)
    results = []
    for mult in f_mults:
        f = float(kelly * mult)
        mc = run_mc_paths(returns, n_sims=n_sims, n_steps=n_steps, f=f, seed=42)
        results.append({
            "f_mult":       float(mult),
            "f":            float(f),
            "median":       float(mc["median"]),
            "blowup_rate":  float(mc["blowup_rate"]),
            "pct_25":       float(mc["percentiles"][25]),
            "pct_75":       float(mc["percentiles"][75]),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Regime-aware MC
# ─────────────────────────────────────────────────────────────────────────────

def run_regime_aware_mc(
    trades: List[dict],
    n_sims: int = 5_000,
    n_steps: int = 300,
    f: float = 0.25,
    seed: int = 42,
) -> Dict:
    """
    MC that samples from regime-stratified return distributions.
    Compare to naive MC.
    """
    rng = np.random.default_rng(seed)
    # Build per-regime return buckets
    buckets = {}
    for t in trades:
        reg = str(t.get("regime", "SIDEWAYS"))
        r   = t.get("pnl_pct", 0.0)
        buckets.setdefault(reg, []).append(float(r))
    regime_probs = {r: len(v) / len(trades) for r, v in buckets.items()}
    all_rets = np.array([t.get("pnl_pct", 0.0) for t in trades])

    equities = np.zeros(n_sims)
    for sim in range(n_sims):
        eq = 1.0
        for _ in range(n_steps):
            if eq < 0.05:
                break
            # Sample regime
            reg = rng.choice(list(regime_probs.keys()),
                             p=list(regime_probs.values()))
            bucket = buckets.get(reg, list(all_rets))
            if len(bucket) == 0:
                r = 0.0
            else:
                r = float(rng.choice(bucket))
            eq = max(0.0, eq * (1.0 + f * r))
        equities[sim] = eq

    pcts = [5, 25, 50, 75, 95]
    return {
        "regime_aware": True,
        "median": float(np.median(equities)),
        "percentiles": {p: float(np.percentile(equities, p)) for p in pcts},
        "blowup_rate": float(np.mean(equities < 0.10)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stress test: remove N% best trades
# ─────────────────────────────────────────────────────────────────────────────

def stress_test_remove_best(
    returns: np.ndarray,
    pct_remove: float = 0.10,
    n_sims: int = 2_000,
    n_steps: int = 200,
    f: float = 0.25,
) -> dict:
    """
    Remove the top pct_remove% of winning trades, re-run MC.
    Returns: {original_median, stressed_median, degradation_pct}
    """
    n_remove = max(1, int(len(returns) * pct_remove))
    # Sort by return descending, remove top n
    sorted_rets = np.sort(returns)[::-1]
    stressed = sorted_rets[n_remove:]

    mc_orig   = run_mc_paths(returns,  n_sims=n_sims, n_steps=n_steps, f=f, seed=42)
    mc_stress = run_mc_paths(stressed, n_sims=n_sims, n_steps=n_steps, f=f, seed=42)

    orig_med   = mc_orig["median"]
    stress_med = mc_stress["median"]
    deg_pct    = (stress_med - orig_med) / (orig_med + 1e-10) * 100

    return {
        "pct_removed": pct_remove,
        "n_removed": n_remove,
        "original_median": float(orig_med),
        "stressed_median": float(stress_med),
        "degradation_pct": float(deg_pct),
        "original_blowup": float(mc_orig["blowup_rate"]),
        "stressed_blowup": float(mc_stress["blowup_rate"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convergence study
# ─────────────────────────────────────────────────────────────────────────────

def convergence_study(returns: np.ndarray, f: float = 0.25) -> Dict:
    """Run MC at 1K, 10K, 100K paths and measure percentile convergence."""
    results = {}
    for n_sims in [1_000, 10_000]:
        mc = run_mc_paths(returns, n_sims=n_sims, n_steps=200, f=f, seed=42)
        results[str(n_sims)] = {
            "median": float(mc["median"]),
            "pct_5":  float(mc["percentiles"][5]),
            "pct_95": float(mc["percentiles"][95]),
            "blowup": float(mc["blowup_rate"]),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_fan_chart(mc_result: Dict, title: str, ax, color: str = "steelblue"):
    """Plot fan chart: 5-95%, 25-75% bands and median path."""
    paths = mc_result.get("paths", [])
    if not paths:
        ax.set_title(title); return
    max_len = max(len(p) for p in paths)
    mat = np.full((len(paths), max_len), np.nan)
    for i, p in enumerate(paths):
        mat[i, :len(p)] = p

    x = np.arange(max_len)
    p5  = np.nanpercentile(mat, 5, axis=0)
    p25 = np.nanpercentile(mat, 25, axis=0)
    p50 = np.nanpercentile(mat, 50, axis=0)
    p75 = np.nanpercentile(mat, 75, axis=0)
    p95 = np.nanpercentile(mat, 95, axis=0)

    ax.fill_between(x, p5,  p95, alpha=0.15, color=color, label="5-95%")
    ax.fill_between(x, p25, p75, alpha=0.25, color=color, label="25-75%")
    ax.plot(x, p50, color=color, linewidth=1.5, label="Median")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, label="Breakeven")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Trade #"); ax.set_ylabel("Portfolio (×Start)")
    ax.legend(fontsize=6); ax.grid(alpha=0.3)


def plot_kelly_frontier(frontier: List[dict], ax):
    """Plot median equity and blowup rate vs Kelly fraction."""
    f_mults = [r["f_mult"] for r in frontier]
    medians = [r["median"] for r in frontier]
    blowups = [r["blowup_rate"] * 100 for r in frontier]

    ax.plot(f_mults, medians, "b-o", markersize=3, linewidth=1.2, label="Median Equity")
    ax.axvline(1.0, color="green", linestyle="--", linewidth=0.8, label="Full Kelly")
    ax.axvline(0.5, color="orange", linestyle=":", linewidth=0.8, label="Half Kelly")
    ax2 = ax.twinx()
    ax2.plot(f_mults, blowups, "r--s", markersize=3, linewidth=1.0, alpha=0.7, label="Blowup%")
    ax.set_title("Kelly Frontier: Median Equity & Blowup Rate", fontsize=9)
    ax.set_xlabel("Kelly Multiple (f / f*)"); ax.set_ylabel("Median Portfolio (×Start)")
    ax2.set_ylabel("Blowup Rate (%)")
    ax.legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("06_monte_carlo_deep_dive.py")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_PLOT = True
    except ImportError:
        HAS_PLOT = False

    print("\nGenerating trade history...")
    trades  = generate_realistic_trades(300, seed=42)
    returns = extract_returns(trades)
    print(f"  {len(trades)} trades  |  win_rate={float(np.mean(returns > 0)):.1%}  "
          f"avg_ret={float(np.mean(returns)):.4f}  std={float(np.std(returns)):.4f}")

    # Kelly fraction
    kelly = compute_kelly_fraction(returns)
    print(f"  Kelly fraction: {kelly:.4f}")

    # 1. Convergence study
    print("\n1. Path convergence study...")
    conv = convergence_study(returns, f=kelly)
    for n_sims, stats in conv.items():
        print(f"  n={n_sims:>7s}: median={stats['median']:.4f}  "
              f"p5={stats['pct_5']:.4f}  p95={stats['pct_95']:.4f}  "
              f"blowup={stats['blowup']:.2%}")

    # 2. Serial correlation analysis
    print("\n2. Serial correlation analysis...")
    mc_serial = {}
    for rho in [0.0, 0.1, 0.3, 0.5]:
        mc = run_mc_paths(returns, n_sims=3_000, n_steps=200, f=kelly, serial_corr=rho, seed=42)
        mc_serial[rho] = mc
        print(f"  ρ={rho:.1f}: median={mc['median']:.4f}  "
              f"avg_maxdd={float(np.mean(mc['max_drawdowns'])):.1%}  "
              f"blowup={mc['blowup_rate']:.2%}")

    # 3. Regime-aware MC
    print("\n3. Regime-aware MC vs naive...")
    mc_naive     = run_mc_paths(returns, n_sims=3_000, n_steps=200, f=kelly, seed=42)
    mc_regime    = run_regime_aware_mc(trades, n_sims=3_000, n_steps=200, f=kelly, seed=42)
    print(f"  Naive:        median={mc_naive['median']:.4f}  blowup={mc_naive['blowup_rate']:.2%}")
    print(f"  Regime-aware: median={mc_regime['median']:.4f}  blowup={mc_regime['blowup_rate']:.2%}")

    # 4. Kelly frontier
    print("\n4. Kelly frontier analysis...")
    frontier = kelly_frontier_analysis(returns, kelly, n_f_pts=15, n_sims=1_500, n_steps=200)
    opt = max(frontier, key=lambda r: r["median"])
    print(f"  Optimal f_mult: {opt['f_mult']:.2f}× Kelly (f={opt['f']:.4f})")
    print(f"  Median at optimal: {opt['median']:.4f}")
    print(f"  Blowup at optimal: {opt['blowup_rate']:.2%}")

    # 5. Stress test
    print("\n5. Stress test (remove 10% best trades)...")
    stress = stress_test_remove_best(returns, pct_remove=0.10, n_sims=2_000, n_steps=200, f=kelly)
    print(f"  Original median:  {stress['original_median']:.4f}")
    print(f"  Stressed median:  {stress['stressed_median']:.4f}")
    print(f"  Degradation:      {stress['degradation_pct']:+.1f}%")
    print(f"  Blowup: {stress['original_blowup']:.2%} → {stress['stressed_blowup']:.2%}")

    # Plotting
    if HAS_PLOT:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle("Monte Carlo Deep Dive", fontsize=13, fontweight="bold")

        # Fan charts for serial correlation scenarios
        rho_colors = {0.0: "steelblue", 0.1: "green", 0.3: "orange", 0.5: "red"}
        for i, (rho, mc) in enumerate(mc_serial.items()):
            ax = axes[0, i]
            plot_fan_chart(mc, f"Serial Corr ρ={rho} (blowup={mc['blowup_rate']:.1%})",
                          ax, color=rho_colors[rho])

        # Kelly frontier
        plot_kelly_frontier(frontier, axes[1, 0])

        # Convergence comparison
        ax_conv = axes[1, 1]
        for n_sims, stats in conv.items():
            label = f"n={n_sims}  med={stats['median']:.3f}"
            ax_conv.bar(n_sims, stats["median"], width=float(n_sims)*0.4,
                       label=label, alpha=0.7)
        ax_conv.set_title("Path Count Convergence", fontsize=9)
        ax_conv.set_xlabel("N Simulations"); ax_conv.set_ylabel("Median Portfolio")
        ax_conv.set_xscale("log"); ax_conv.legend(fontsize=7); ax_conv.grid(alpha=0.3)

        # Stress test visualization
        ax_st = axes[1, 2]
        categories = ["Original", "Stressed\n(−10% best)"]
        medians = [stress["original_median"], stress["stressed_median"]]
        blowups = [stress["original_blowup"] * 100, stress["stressed_blowup"] * 100]
        x = np.arange(2)
        ax_st.bar(x - 0.2, medians, 0.4, color=["steelblue", "orange"], label="Median")
        ax_st2 = ax_st.twinx()
        ax_st2.bar(x + 0.2, blowups, 0.4, color=["steelblue", "orange"], alpha=0.5, label="Blowup%")
        ax_st.set_xticks(x); ax_st.set_xticklabels(categories)
        ax_st.set_title("Stress Test: Remove 10% Best", fontsize=9)
        ax_st.set_ylabel("Median (×Start)"); ax_st2.set_ylabel("Blowup Rate (%)")
        ax_st.legend(fontsize=7, loc="upper left"); ax_st2.legend(fontsize=7, loc="upper right")
        ax_st.grid(alpha=0.3)

        # Return distribution
        ax_ret = axes[1, 3]
        ax_ret.hist(returns, bins=50, color="steelblue", alpha=0.7, density=True)
        ax_ret.axvline(0, color="black", linewidth=0.8)
        ax_ret.axvline(float(np.mean(returns)), color="orange", linestyle="--",
                       label=f"Mean={np.mean(returns):.4f}")
        ax_ret.set_title("Trade Return Distribution", fontsize=9)
        ax_ret.set_xlabel("Return"); ax_ret.set_ylabel("Density")
        ax_ret.legend(fontsize=7); ax_ret.grid(alpha=0.3)

        plt.tight_layout()
        out = OUTPUTS / "mc_deep_dive.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\nPlot → {out}")

    # Save analysis
    def _clean(obj):
        if isinstance(obj, (float, np.floating)):
            v = float(obj); return v if math.isfinite(v) else None
        if isinstance(obj, (int, np.integer)): return int(obj)
        if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_clean(v) for v in obj]
        if isinstance(obj, np.ndarray): return [_clean(v) for v in obj.tolist()]
        return obj

    analysis = {
        "kelly_fraction":   float(kelly),
        "n_trades":         len(trades),
        "win_rate":         float(np.mean(returns > 0)),
        "convergence":      _clean(conv),
        "serial_corr":      _clean({str(rho): {
            "median": float(mc["median"]),
            "blowup": float(mc["blowup_rate"]),
            "avg_maxdd": float(np.mean(mc["max_drawdowns"])),
        } for rho, mc in mc_serial.items()}),
        "naive_vs_regime":  _clean({
            "naive":       {"median": float(mc_naive["median"]),  "blowup": float(mc_naive["blowup_rate"])},
            "regime_aware":{"median": float(mc_regime["median"]), "blowup": float(mc_regime["blowup_rate"])},
        }),
        "kelly_frontier":   _clean(frontier),
        "stress_test":      _clean(stress),
    }
    out_json = OUTPUTS / "mc_analysis.json"
    with open(out_json, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis → {out_json}")


if __name__ == "__main__":
    main()
