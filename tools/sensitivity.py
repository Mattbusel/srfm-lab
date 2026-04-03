"""
sensitivity.py — One-at-a-time parameter sensitivity analysis for LARSA arena_v2.

For each parameter in {cf, bh_form, bh_decay, max_lev}:
  - Tests 7 evenly-spaced values across the parameter range
  - Runs arena_v2 + 3 synthetic worlds at each value
  - Records: sharpe, return, drawdown
  - Computes impact score = (max_sharpe - min_sharpe) across the 7 values

Outputs:
  ASCII tornado chart to stdout
  results/sensitivity.md with full tables

Usage:
    python tools/sensitivity.py --csv data/NDX_hourly_poly.csv
    python tools/sensitivity.py --param cf
    python tools/sensitivity.py --param bh_form --csv data/NDX_hourly_poly.csv
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from arena_v2 import run_v2, load_ohlcv, generate_synthetic, CONFIGS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Default "center" config (ABCD on for full v2 experiments)
DEFAULT_CFG = {"cf": 0.005, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95}
DEFAULT_LEV = 0.65
EXP_FLAGS   = "ABCD"
SYNTH_SEEDS = [42, 137, 999]

PARAM_RANGES = {
    "cf":       (0.0005, 0.010),
    "bh_form":  (1.0,    3.0),
    "bh_decay": (0.80,   0.99),
    "max_lev":  (0.30,   0.80),
}
N_STEPS = 7


def make_cfg(base_cfg, param, value):
    cfg = dict(base_cfg)
    if param == "max_lev":
        return cfg, value     # max_lev is passed separately
    cfg[param] = value
    return cfg, DEFAULT_LEV


def run_one(bars, cfg, max_lev):
    try:
        broker, _ = run_v2(bars, cfg, max_leverage=max_lev, exp_flags=EXP_FLAGS)
        s = broker.stats()
        return (
            s.get("sharpe", 0.0),
            s.get("total_return_pct", 0.0),
            s.get("max_drawdown_pct", 0.0),
        )
    except Exception:
        return 0.0, 0.0, 100.0


def sweep_param(param, bars_arena, bars_synths, n_steps=N_STEPS):
    lo, hi = PARAM_RANGES[param]
    values = np.linspace(lo, hi, n_steps).tolist()
    rows = []
    for v in values:
        cfg, lev = make_cfg(DEFAULT_CFG, param, v)
        a_sh, a_ret, a_dd = run_one(bars_arena, cfg, lev)

        synth_sh_list = []
        for sb in bars_synths:
            sh, _, _ = run_one(sb, cfg, lev)
            synth_sh_list.append(sh)
        s_sh = float(np.mean(synth_sh_list)) if synth_sh_list else 0.0

        rows.append({
            "value":        round(v, 6),
            "arena_sharpe": round(a_sh, 4),
            "synth_sharpe": round(s_sh, 4),
            "arena_return": round(a_ret, 2),
            "arena_dd":     round(a_dd, 2),
        })
        print(f"    {param}={v:.5f}  arena_sh={a_sh:.4f}  synth_sh={s_sh:.4f}  "
              f"ret={a_ret:.1f}%  dd={a_dd:.1f}%")
    return rows


def tornado_bar(impact, max_impact, width=24):
    filled = int(round(impact / max_impact * width)) if max_impact > 0 else 0
    return "\u2588" * filled + " " * (width - filled)


def print_tornado(impacts):
    print("\nPARAMETER SENSITIVITY (impact on Sharpe)")
    print("=" * 53)
    max_imp = max(impacts.values()) if impacts else 1.0
    for param, imp in sorted(impacts.items(), key=lambda x: -x[1]):
        bar = tornado_bar(imp, max_imp)
        print(f"{param:<8} {bar}  {imp:.3f}")
    print()
    most   = max(impacts, key=impacts.get)
    least  = min(impacts, key=impacts.get)
    lo_m, hi_m = PARAM_RANGES[most]
    step_m = (hi_m - lo_m) / (N_STEPS - 1)
    print(f"Most sensitive:  {most} (+/- {impacts[most]/2:.3f} Sharpe per {step_m:.4f} change)")
    print(f"Least sensitive: {least} (safe to keep at default)")


def write_md(all_results, impacts, path):
    lines = ["# Parameter Sensitivity Analysis\n"]
    lines.append("Objective metric: Sharpe ratio (arena + synthetic average)\n")
    lines.append("## Impact Summary (Tornado)\n")
    lines.append("| Parameter | Impact (Δ Sharpe) |\n")
    lines.append("|-----------|------------------|\n")
    for p, imp in sorted(impacts.items(), key=lambda x: -x[1]):
        lines.append(f"| {p} | {imp:.4f} |\n")
    lines.append("\n")

    for param, rows in all_results.items():
        lines.append(f"## {param}\n\n")
        lines.append("| Value | Arena Sharpe | Synth Sharpe | Arena Return % | Arena DD % |\n")
        lines.append("|-------|-------------|-------------|----------------|------------|\n")
        for r in rows:
            lines.append(f"| {r['value']:.6f} | {r['arena_sharpe']:.4f} | "
                         f"{r['synth_sharpe']:.4f} | {r['arena_return']:.2f} | "
                         f"{r['arena_dd']:.2f} |\n")
        lines.append("\n")

    with open(path, "w") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(
        description="One-at-a-time parameter sensitivity analysis for LARSA arena_v2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv",   default="data/NDX_hourly_poly.csv",
                        help="Path to OHLCV CSV (default: data/NDX_hourly_poly.csv)")
    parser.add_argument("--param", default=None,
                        choices=list(PARAM_RANGES.keys()),
                        help="Test just one parameter (default: all)")
    parser.add_argument("--steps", type=int, default=N_STEPS,
                        help=f"Number of test values per parameter (default: {N_STEPS})")
    parser.add_argument("--synth-bars", type=int, default=20000,
                        help="Bars per synthetic world (default: 20000)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load arena data
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", csv_path)
    if os.path.exists(csv_path):
        print(f"Loading arena data from {csv_path} ...")
        bars_arena = load_ohlcv(csv_path)
        print(f"  {len(bars_arena)} bars loaded.")
    else:
        print(f"WARNING: {csv_path} not found — using synthetic data for arena.")
        bars_arena = generate_synthetic(args.synth_bars, seed=0)

    bars_synths = [generate_synthetic(args.synth_bars, seed=s) for s in SYNTH_SEEDS]

    params_to_test = [args.param] if args.param else list(PARAM_RANGES.keys())

    all_results = {}
    impacts = {}

    for param in params_to_test:
        print(f"\n--- Sweeping: {param} ({args.steps} values) ---")
        rows = sweep_param(param, bars_arena, bars_synths, n_steps=args.steps)
        all_results[param] = rows
        sharpes = [r["arena_sharpe"] for r in rows]
        impacts[param] = round(max(sharpes) - min(sharpes), 4)

    print_tornado(impacts)

    md_path = os.path.join(RESULTS_DIR, "sensitivity.md")
    write_md(all_results, impacts, md_path)
    print(f"\nDetailed results saved → {md_path}")


if __name__ == "__main__":
    main()
