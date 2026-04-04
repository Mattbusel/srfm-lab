"""
08_live_trader_analysis.py — Live Trader Analysis

Reads spacetime/cache/live_state.json (or simulates).
Computes: live BH state stats, how often BH is active, comparison to historical,
signal strength (mass >> threshold), expected trades in next 24h.
Outputs a daily monitoring report.

Run: python research/notebooks/08_live_trader_analysis.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "lib"))

from srfm_core import MinkowskiClassifier, BlackHoleDetector

OUTPUTS = _ROOT / "research" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

LIVE_STATE_PATH = _ROOT / "spacetime" / "cache" / "live_state.json"


# ─────────────────────────────────────────────────────────────────────────────
# Live state loading / simulation
# ─────────────────────────────────────────────────────────────────────────────

def load_live_state(path: Path) -> Optional[Dict]:
    """Load live state JSON if it exists."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"  Failed to load live state: {e}")
    return None


def simulate_live_state(n_instruments: int = 9, seed: int = 42) -> Dict:
    """Simulate a realistic live state for analysis."""
    rng = np.random.default_rng(seed)
    syms = ["ES", "NQ", "BTC", "ETH", "GC", "CL", "EURUSD", "ZB", "YM"][:n_instruments]
    cfs  = [0.001, 0.0012, 0.005, 0.007, 0.008, 0.015, 0.0005, 0.003, 0.0008][:n_instruments]

    instruments = {}
    for i, (sym, cf) in enumerate(zip(syms, cfs)):
        bh_mass = float(rng.exponential(1.2))
        active  = bh_mass > 1.5 and rng.random() > 0.4

        instruments[sym] = {
            "bh_mass_1d":      float(rng.exponential(1.0)),
            "bh_mass_1h":      float(bh_mass),
            "bh_mass_15m":     float(rng.exponential(0.8)),
            "bh_active_1d":    bool(rng.random() > 0.5),
            "bh_active_1h":    bool(active),
            "bh_active_15m":   bool(rng.random() > 0.6),
            "bh_dir_1d":       int(rng.choice([-1, 0, 1], p=[0.25, 0.25, 0.50])),
            "bh_dir_1h":       int(rng.choice([-1, 0, 1], p=[0.20, 0.20, 0.60])),
            "ctl_1h":          int(rng.integers(0, 15)),
            "last_price":      float(rng.uniform(100, 50000)),
            "cf":              cf,
            "regime":          str(rng.choice(["BULL", "SIDEWAYS", "BEAR"])),
            "tf_score":        int(
                (4 if rng.random() > 0.4 else 0) |
                (2 if active else 0) |
                (1 if rng.random() > 0.55 else 0)
            ),
        }

    return {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "portfolio_equity": float(1_000_000 * (1 + rng.normal(0.15, 0.05))),
        "n_instruments": n_instruments,
        "instruments": instruments,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_bh_summary(state: Dict) -> Dict:
    """Summarize BH activity across the portfolio."""
    instruments = state.get("instruments", {})
    n = len(instruments)
    if n == 0:
        return {}

    n_active_1d  = sum(1 for v in instruments.values() if v.get("bh_active_1d"))
    n_active_1h  = sum(1 for v in instruments.values() if v.get("bh_active_1h"))
    n_active_15m = sum(1 for v in instruments.values() if v.get("bh_active_15m"))
    tf_scores    = [v.get("tf_score", 0) for v in instruments.values()]
    masses_1h    = [v.get("bh_mass_1h", 0) for v in instruments.values()]

    high_conviction = [sym for sym, v in instruments.items() if v.get("tf_score", 0) >= 6]
    marginal        = [sym for sym, v in instruments.items()
                       if 0 < v.get("bh_mass_1h", 0) < 2.0 and not v.get("bh_active_1h")]

    # Signal strength: how far above bh_form are active instruments?
    bh_form = 1.5
    strength_scores = {}
    for sym, v in instruments.items():
        mass = v.get("bh_mass_1h", 0)
        if v.get("bh_active_1h") and mass > bh_form:
            strength_scores[sym] = round(float(mass / bh_form), 2)

    return {
        "n_instruments":   n,
        "n_active_1d":     n_active_1d,
        "n_active_1h":     n_active_1h,
        "n_active_15m":    n_active_15m,
        "pct_active_1h":   round(n_active_1h / n, 3),
        "avg_tf_score":    round(float(np.mean(tf_scores)), 2),
        "max_tf_score":    int(max(tf_scores)),
        "avg_mass_1h":     round(float(np.mean(masses_1h)), 3),
        "max_mass_1h":     round(float(max(masses_1h)), 3),
        "high_conviction": high_conviction,
        "marginal_wells":  marginal,
        "strength_scores": strength_scores,
    }


def compare_to_historical(state: Dict, n_historical: int = 500) -> Dict:
    """
    Compare current BH state to historical distribution.
    Generates historical baseline by simulating n_historical time steps.
    """
    rng = np.random.default_rng(999)
    # Generate historical mass distribution
    hist_masses = np.random.default_rng(999).exponential(1.0, n_historical)
    hist_masses = hist_masses[hist_masses > 0]

    instruments = state.get("instruments", {})
    current_masses = [v.get("bh_mass_1h", 0) for v in instruments.values()]

    avg_current = float(np.mean(current_masses))
    p75_hist    = float(np.percentile(hist_masses, 75))
    p90_hist    = float(np.percentile(hist_masses, 90))

    signal_level = "NORMAL"
    if avg_current > p90_hist:
        signal_level = "ELEVATED"
    elif avg_current > p75_hist:
        signal_level = "ABOVE_AVERAGE"
    elif avg_current < float(np.percentile(hist_masses, 25)):
        signal_level = "LOW"

    return {
        "avg_current_mass_1h": round(avg_current, 3),
        "hist_p25":            round(float(np.percentile(hist_masses, 25)), 3),
        "hist_p50":            round(float(np.percentile(hist_masses, 50)), 3),
        "hist_p75":            round(float(np.percentile(hist_masses, 75)), 3),
        "hist_p90":            round(float(np.percentile(hist_masses, 90)), 3),
        "signal_level":        signal_level,
        "percentile_rank":     round(float(np.mean(hist_masses < avg_current)) * 100, 1),
    }


def estimate_expected_trades(state: Dict) -> Dict:
    """
    Estimate expected number of new BH activations in the next 24 hours.
    Heuristic: based on mass level and historical formation rate.
    """
    instruments = state.get("instruments", {})
    expected = {}

    for sym, v in instruments.items():
        mass_1h = v.get("bh_mass_1h", 0)
        ctl_1h  = v.get("ctl_1h", 0)
        active  = v.get("bh_active_1h", False)

        bh_form     = 1.5
        bh_collapse = 1.0

        if active:
            # Currently in a BH — estimate how long it will persist
            # Mass decay at 0.95/bar → half life = ln(0.5)/ln(0.95) ≈ 13.5 bars
            bars_to_collapse = max(0, math.log(bh_collapse / max(mass_1h, bh_collapse + 0.001)) /
                                   math.log(0.95)) if mass_1h > bh_collapse else 0
            expected[sym] = {
                "currently_active": True,
                "bars_remaining":   round(bars_to_collapse, 1),
                "hours_remaining":  round(bars_to_collapse, 1),  # hourly bars
                "new_formations_24h": 0,
            }
        else:
            # Not active — estimate formation probability
            # If mass is building (ctl > 0), formation more likely
            mass_gap = bh_form - mass_1h
            if ctl_1h >= 5 and mass_gap < 0.5:
                prob = 0.6  # close to forming
            elif ctl_1h >= 3 and mass_gap < 1.0:
                prob = 0.3
            else:
                prob = 0.1
            expected[sym] = {
                "currently_active": False,
                "formation_probability_24h": round(prob, 3),
                "mass_to_form": round(max(0, mass_gap), 3),
                "new_formations_24h": round(prob * 1.2, 2),
            }

    total_expected = sum(
        v.get("new_formations_24h", 0) for v in expected.values()
        if not v.get("currently_active", False)
    )
    active_count = sum(1 for v in expected.values() if v.get("currently_active", False))

    return {
        "per_instrument":    expected,
        "total_expected_new_24h": round(total_expected, 1),
        "currently_active": active_count,
    }


def classify_signal_strength(state: Dict) -> Dict:
    """
    Classify each instrument's signal as: STRONG, MODERATE, WEAK, INACTIVE.
    Strong: tf_score >= 6, mass > 2.0×bh_form
    Moderate: tf_score >= 4, active
    Weak: mass building (0 < mass < bh_form), ctl > 3
    Inactive: no BH activity
    """
    classifications = {}
    BH_FORM = 1.5

    for sym, v in state.get("instruments", {}).items():
        tf  = v.get("tf_score", 0)
        m1h = v.get("bh_mass_1h", 0)
        act = v.get("bh_active_1h", False)
        ctl = v.get("ctl_1h", 0)

        if act and tf >= 6 and m1h > 2.0 * BH_FORM:
            strength = "STRONG"
        elif act and tf >= 4:
            strength = "MODERATE"
        elif not act and ctl >= 3 and m1h > 0.5:
            strength = "WEAK_BUILDING"
        elif not act:
            strength = "INACTIVE"
        else:
            strength = "MARGINAL"

        classifications[sym] = {
            "strength":   strength,
            "tf_score":   tf,
            "bh_mass_1h": round(m1h, 3),
            "bh_active":  act,
            "ctl_1h":     ctl,
        }
    return classifications


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_monitoring_report(state: Dict) -> str:
    """Generate a daily monitoring text report."""
    ts = state.get("timestamp", "unknown")
    eq = state.get("portfolio_equity", 0)

    summary  = portfolio_bh_summary(state)
    hist_cmp = compare_to_historical(state)
    expected = estimate_expected_trades(state)
    classify = classify_signal_strength(state)

    lines = [
        "=" * 65,
        f"  SRFM LIVE TRADER MONITORING REPORT",
        f"  Generated: {ts}",
        "=" * 65,
        f"\n  Portfolio Equity: ${eq:,.0f}",
        f"\n  BH ACTIVITY SUMMARY",
        f"  {'─'*40}",
        f"  Active instruments (1h):  {summary.get('n_active_1h', 0)}/{summary.get('n_instruments', 0)} "
        f"({summary.get('pct_active_1h', 0):.0%})",
        f"  Average TF score:         {summary.get('avg_tf_score', 0):.2f}",
        f"  Max TF score:             {summary.get('max_tf_score', 0)}",
        f"  Average BH mass (1h):     {summary.get('avg_mass_1h', 0):.3f}",
        f"  Signal level vs history:  {hist_cmp.get('signal_level', 'N/A')} "
        f"(pct_rank={hist_cmp.get('percentile_rank', 0):.0f}th)",
        "",
    ]

    # High conviction signals
    hc = summary.get("high_conviction", [])
    if hc:
        lines.append(f"  HIGH CONVICTION SIGNALS (TF >= 6)")
        for sym in hc:
            sig = classify.get(sym, {})
            lines.append(f"    {sym:8s}: {sig.get('strength', 'N/A'):15s} "
                         f"TF={sig.get('tf_score', 0)}  mass={sig.get('bh_mass_1h', 0):.3f}")
    else:
        lines.append("  No high-conviction signals currently.")

    lines.append("")

    # All signals
    lines.append(f"  SIGNAL CLASSIFICATION")
    lines.append(f"  {'─'*40}")
    for sym, info in sorted(classify.items(), key=lambda x: x[1]["tf_score"], reverse=True):
        lines.append(f"  {sym:8s}: {info['strength']:15s} TF={info['tf_score']}  "
                     f"mass={info['bh_mass_1h']:.3f}  "
                     f"active={'Y' if info['bh_active'] else 'N'}  "
                     f"ctl={info['ctl_1h']}")

    lines.append("")
    lines.append(f"  EXPECTED ACTIVITY (NEXT 24H)")
    lines.append(f"  {'─'*40}")
    lines.append(f"  Currently active:       {expected.get('currently_active', 0)}")
    lines.append(f"  Expected new formations:{expected.get('total_expected_new_24h', 0):.1f}")

    # Instruments close to forming
    for sym, info in expected.get("per_instrument", {}).items():
        if not info.get("currently_active") and info.get("formation_probability_24h", 0) > 0.25:
            lines.append(f"    {sym:8s}: P(form 24h)={info['formation_probability_24h']:.0%}  "
                         f"Δmass={info.get('mass_to_form', 0):.3f}")

    lines.append("")
    lines.append(f"  HISTORICAL CONTEXT")
    lines.append(f"  {'─'*40}")
    lines.append(f"  Current avg mass: {hist_cmp.get('avg_current_mass_1h', 0):.3f}")
    lines.append(f"  Historical p25:   {hist_cmp.get('hist_p25', 0):.3f}")
    lines.append(f"  Historical p50:   {hist_cmp.get('hist_p50', 0):.3f}")
    lines.append(f"  Historical p75:   {hist_cmp.get('hist_p75', 0):.3f}")
    lines.append(f"  Historical p90:   {hist_cmp.get('hist_p90', 0):.3f}")

    lines.append("")
    lines.append("=" * 65)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_live_state(state: Dict, fig, axes):
    """Plot current BH state overview."""
    instruments = state.get("instruments", {})
    syms = list(instruments.keys())
    n = len(syms)

    # BH mass bar chart
    ax = axes[0]
    masses_1d  = [instruments[s].get("bh_mass_1d", 0) for s in syms]
    masses_1h  = [instruments[s].get("bh_mass_1h", 0) for s in syms]
    masses_15m = [instruments[s].get("bh_mass_15m", 0) for s in syms]
    x = np.arange(n)
    ax.bar(x - 0.25, masses_1d,  0.25, label="1D",  alpha=0.8, color="steelblue")
    ax.bar(x,        masses_1h,  0.25, label="1H",  alpha=0.8, color="orange")
    ax.bar(x + 0.25, masses_15m, 0.25, label="15M", alpha=0.8, color="green")
    ax.axhline(1.5, color="red",    linestyle="--", linewidth=0.8, label="bh_form=1.5")
    ax.axhline(1.0, color="orange", linestyle=":",  linewidth=0.8, label="bh_coll=1.0")
    ax.set_xticks(x); ax.set_xticklabels(syms, rotation=30, fontsize=8)
    ax.set_title("Current BH Mass by Instrument & Timeframe", fontsize=9)
    ax.set_ylabel("BH Mass"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    # TF Score chart
    ax = axes[1]
    tf_scores = [instruments[s].get("tf_score", 0) for s in syms]
    colors = ["red" if t < 4 else "orange" if t < 6 else "lime" for t in tf_scores]
    ax.bar(x, tf_scores, color=colors, alpha=0.8)
    ax.axhline(6, color="lime", linestyle="--", linewidth=0.8, label="TF=6 threshold")
    ax.set_xticks(x); ax.set_xticklabels(syms, rotation=30, fontsize=8)
    ax.set_title("Current TF Score by Instrument", fontsize=9)
    ax.set_ylabel("TF Score"); ax.set_ylim(0, 8)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    # Signal strength
    ax = axes[2]
    classify = classify_signal_strength(state)
    strength_map = {"STRONG": 4, "MODERATE": 3, "WEAK_BUILDING": 2, "MARGINAL": 1, "INACTIVE": 0}
    strength_vals  = [strength_map.get(classify.get(s, {}).get("strength", "INACTIVE"), 0) for s in syms]
    strength_colors = {4: "lime", 3: "yellowgreen", 2: "yellow", 1: "orange", 0: "lightgray"}
    sc = [strength_colors[v] for v in strength_vals]
    ax.bar(x, strength_vals, color=sc, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(syms, rotation=30, fontsize=8)
    ax.set_yticks([0,1,2,3,4])
    ax.set_yticklabels(["INACTIVE","MARGINAL","WEAK","MODERATE","STRONG"], fontsize=7)
    ax.set_title("Signal Strength by Instrument", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Active/inactive heatmap
    ax = axes[3]
    bh_active_mat = np.array([
        [int(instruments[s].get("bh_active_1d", False)) for s in syms],
        [int(instruments[s].get("bh_active_1h", False)) for s in syms],
        [int(instruments[s].get("bh_active_15m", False)) for s in syms],
    ])
    im = ax.imshow(bh_active_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(syms, rotation=30, fontsize=7)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(["1D","1H","15M"], fontsize=8)
    ax.set_title("BH Active Status (Green=Active)", fontsize=9)
    import matplotlib.pyplot as plt
    plt.colorbar(im, ax=ax)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("08_live_trader_analysis.py")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_PLOT = True
    except ImportError:
        HAS_PLOT = False

    # Load or simulate state
    print("\nLoading live state...")
    state = load_live_state(LIVE_STATE_PATH)
    if state is None:
        print(f"  {LIVE_STATE_PATH} not found — simulating state")
        state = simulate_live_state(n_instruments=9, seed=42)
    else:
        print(f"  Loaded from {LIVE_STATE_PATH}")

    n_inst = len(state.get("instruments", {}))
    print(f"  {n_inst} instruments in state")
    print(f"  Portfolio equity: ${state.get('portfolio_equity', 0):,.0f}")

    # Analysis
    summary  = portfolio_bh_summary(state)
    hist_cmp = compare_to_historical(state)
    expected = estimate_expected_trades(state)
    classify = classify_signal_strength(state)

    print(f"\n  Active (1h): {summary.get('n_active_1h', 0)}/{n_inst}")
    print(f"  Avg TF score: {summary.get('avg_tf_score', 0):.2f}")
    print(f"  Signal level: {hist_cmp.get('signal_level', 'N/A')}")
    print(f"  Expected new formations (24h): {expected.get('total_expected_new_24h', 0):.1f}")

    if summary.get("high_conviction"):
        print(f"  High conviction: {summary['high_conviction']}")

    # Generate report
    report = generate_monitoring_report(state)
    print("\n" + report)

    # Save report
    report_path = OUTPUTS / "live_monitoring_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport → {report_path}")

    # Plotting
    if HAS_PLOT:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle("SRFM Live Trader State Dashboard", fontsize=12, fontweight="bold")
        plot_live_state(state, fig, axes)
        plt.tight_layout()
        out = OUTPUTS / "live_state_dashboard.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Dashboard → {out}")

    # Save JSON analysis
    def _clean(obj):
        if isinstance(obj, (float, np.floating)):
            v = float(obj); return v if math.isfinite(v) else None
        if isinstance(obj, (int, np.integer)): return int(obj)
        if isinstance(obj, bool): return bool(obj)
        if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_clean(v) for v in obj]
        return obj

    analysis = _clean({
        "summary":        summary,
        "historical_cmp": hist_cmp,
        "expected_trades": expected,
        "signal_strength": classify,
    })
    out_json = OUTPUTS / "live_analysis.json"
    with open(out_json, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis → {out_json}")


if __name__ == "__main__":
    main()
