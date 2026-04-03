"""
experiment_runner.py — Systematic v2 experiment suite.

Tests all 2^4 = 16 flag combinations plus parameter variations.
Outputs to results/v2_experiments.md

Usage:
    python tools/experiment_runner.py
    python tools/experiment_runner.py --quick   (arena only, skip tournament)
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

# Add tools/ to path
tools_dir = os.path.dirname(__file__)
sys.path.insert(0, tools_dir)

from arena_v2 import run_v2, load_ohlcv, generate_synthetic, CONFIGS


def run_experiment(bars, cfg, exp_flags, max_lev=0.65, label="", n_synth=5):
    """Run arena + N synthetic worlds. Returns dict of results."""
    broker, _ = run_v2(bars, cfg, max_leverage=max_lev, exp_flags=exp_flags)
    arena_stats = broker.stats()

    synth_results = []
    for seed in range(n_synth):
        synth_bars = generate_synthetic(20000, seed=seed + 100)
        sb, _ = run_v2(synth_bars, cfg, max_leverage=max_lev, exp_flags=exp_flags)
        synth_results.append(sb.stats())

    synth_sharpe = [r["sharpe"] for r in synth_results]
    synth_return = [r["total_return_pct"] for r in synth_results]
    synth_dd     = [r["max_drawdown_pct"] for r in synth_results]

    return {
        "exp":            label or (exp_flags.upper() or "BASELINE"),
        "flags":          exp_flags.upper(),
        # Arena
        "arena_return":   arena_stats["total_return_pct"],
        "arena_dd":       arena_stats["max_drawdown_pct"],
        "arena_sharpe":   arena_stats["sharpe"],
        "arena_trades":   arena_stats["trade_count"],
        # Synthetic (median across N worlds)
        "synth_sharpe":   sorted(synth_sharpe)[len(synth_sharpe)//2],
        "synth_return":   sorted(synth_return)[len(synth_return)//2],
        "synth_dd":       sorted(synth_dd)[len(synth_dd)//2],
        "synth_worlds":   len(synth_results),
    }


def score_experiment(r, baseline):
    """Score vs baseline. Higher is better."""
    arena_improvement = (
        (r["arena_sharpe"] - baseline["arena_sharpe"]) * 2.0 +
        (r["arena_return"] - baseline["arena_return"]) * 0.01 -
        max(0, r["arena_dd"] - baseline["arena_dd"]) * 0.5
    )
    synth_improvement = (
        (r["synth_sharpe"] - baseline["synth_sharpe"]) * 2.0 +
        (r["synth_return"] - baseline["synth_return"]) * 0.01 -
        max(0, r["synth_dd"] - baseline["synth_dd"]) * 0.5
    )
    # Both must improve for "real" signal (vs overfitting)
    overfit_flag = "OVERFIT" if (arena_improvement > 0.5 and synth_improvement < -0.5) else ""
    return arena_improvement, synth_improvement, overfit_flag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip tournament, 3 synth worlds")
    parser.add_argument("--csv",   default="data/NDX_hourly_poly.csv")
    parser.add_argument("--cf",    type=float, default=0.005)
    args = parser.parse_args()

    n_synth = 3 if args.quick else 10

    print(f"Loading arena data: {args.csv}...")
    if not os.path.exists(args.csv):
        print(f"  [WARN] {args.csv} not found, using synthetic data for arena too")
        arena_bars = generate_synthetic(5000, seed=999)
    else:
        arena_bars = load_ohlcv(args.csv)
        print(f"  {len(arena_bars)} bars")

    cfg = {"cf": args.cf, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95}

    # All 2^4 = 16 combinations + baseline + parameter variations
    flag_combos = [""]  # baseline
    all_flags = ["A", "B", "C", "D"]
    for r in range(1, 5):
        for combo in combinations(all_flags, r):
            flag_combos.append("".join(combo))

    # Additional parameter variations on best combos
    param_variants = [
        ("ABCD_lev50", "ABCD", 0.50),
        ("ABCD_lev80", "ABCD", 0.80),
        ("BCD_lev65",  "BCD",  0.65),
        ("ABD_lev65",  "ABD",  0.65),
    ]

    results = []
    total_exps = len(flag_combos) + len(param_variants)
    print(f"\nRunning {total_exps} experiments ({n_synth} synthetic worlds each)...\n")

    for i, flags in enumerate(flag_combos):
        label = flags.upper() if flags else "BASELINE"
        print(f"  [{i+1:02d}/{total_exps}] {label:<12}", end=" ", flush=True)
        t0 = time.time()
        r = run_experiment(arena_bars, cfg, flags, max_lev=0.65, label=label, n_synth=n_synth)
        print(f"  arena: {r['arena_return']:+6.2f}%  DD:{r['arena_dd']:5.1f}%  Sh:{r['arena_sharpe']:.3f}  "
              f"| synth Sh:{r['synth_sharpe']:.3f}  ({time.time()-t0:.1f}s)")
        results.append(r)

    for label, flags, lev in param_variants:
        idx = len(results) + 1
        print(f"  [{idx:02d}/{total_exps}] {label:<12}", end=" ", flush=True)
        t0 = time.time()
        r = run_experiment(arena_bars, cfg, flags, max_lev=lev, label=label, n_synth=n_synth)
        print(f"  arena: {r['arena_return']:+6.2f}%  DD:{r['arena_dd']:5.1f}%  Sh:{r['arena_sharpe']:.3f}  "
              f"| synth Sh:{r['synth_sharpe']:.3f}  ({time.time()-t0:.1f}s)")
        results.append(r)

    baseline = results[0]

    # Score and rank
    for r in results:
        a_imp, s_imp, of = score_experiment(r, baseline)
        r["arena_improvement"] = a_imp
        r["synth_improvement"] = s_imp
        r["overfit"] = of
        r["combined_score"] = (a_imp + s_imp) / 2

    ranked = sorted(results, key=lambda r: -r["combined_score"])

    # Print summary
    print(f"\n{'='*85}")
    print(f"  RESULTS RANKED BY COMBINED IMPROVEMENT (ARENA + SYNTHETIC)")
    print(f"{'='*85}")
    print(f"  {'Exp':<14}  {'Arena Ret':>10}  {'ArDD':>6}  {'ArSh':>6}  {'SyRet':>8}  {'SySh':>6}  {'Score':>7}  {'Flag'}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*8}")
    for r in ranked[:15]:
        flag = r.get("overfit", "")
        print(f"  {r['exp']:<14}  {r['arena_return']:>+9.2f}%  {r['arena_dd']:>5.1f}%  "
              f"{r['arena_sharpe']:>6.3f}  {r['synth_return']:>+7.2f}%  "
              f"{r['synth_sharpe']:>6.3f}  {r['combined_score']:>+6.3f}  {flag}")

    best = ranked[0]
    print(f"\n  WINNER: {best['exp']}  (arena +{best['arena_improvement']:.3f}  synth +{best['synth_improvement']:.3f})")
    if best.get("overfit"):
        print(f"  WARNING: {best['overfit']} — synthetic does not confirm")
    print()

    # Save markdown report
    os.makedirs("results", exist_ok=True)
    _write_markdown(results, ranked, baseline, args.csv, args.cf, n_synth)

    # Save JSON
    with open("results/v2_experiments.json", "w") as f:
        json.dump(results, f, indent=2)

    return ranked, best


def _write_markdown(results, ranked, baseline, csv_path, cf, n_synth):
    lines = []
    lines.append("# LARSA v2 Experiment Log\n")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    lines.append(f"- Arena data: `{csv_path}` (CF={cf})")
    lines.append(f"- Synthetic: {n_synth} worlds x 20,000 bars (mixed regime)\n")

    lines.append("\n## Experiment Design\n")
    lines.append("| Flag | Description |")
    lines.append("| --- | --- |")
    lines.append("| A | ATR_SCALE: position × min(1, 1.5/atr_ratio) when atr_ratio > 1.5 |")
    lines.append("| B | STOP_LOSS: cut position if per-instrument PnL < -3% portfolio |")
    lines.append("| C | BEAR_FAST: tl_req=2 in BEAR regime (was 3) |")
    lines.append("| D | BH_BOOST: scale by min(1.5, bh_mass/1.5) when BH active |")
    lines.append("| BASELINE | v1 unchanged |")

    lines.append("\n## Results — All Experiments\n")
    lines.append("| Exp | Arena Return | Arena DD | Arena Sharpe | Synth Return | Synth Sharpe | Score | Overfit? |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in ranked:
        lines.append(f"| {r['exp']} | {r['arena_return']:+.2f}% | {r['arena_dd']:.1f}% | "
                     f"{r['arena_sharpe']:.3f} | {r['synth_return']:+.2f}% | "
                     f"{r['synth_sharpe']:.3f} | {r['combined_score']:+.3f} | {r.get('overfit','')} |")

    lines.append("\n## Baseline\n")
    lines.append(f"- Arena: Return={baseline['arena_return']:+.2f}%  DD={baseline['arena_dd']:.1f}%  Sharpe={baseline['arena_sharpe']:.3f}")
    lines.append(f"- Synth: Return={baseline['synth_return']:+.2f}%  DD={baseline['synth_dd']:.1f}%  Sharpe={baseline['synth_sharpe']:.3f}")

    best = ranked[0]
    lines.append("\n## Best Configuration\n")
    lines.append(f"**{best['exp']}**\n")
    lines.append(f"- Arena: Return={best['arena_return']:+.2f}%  DD={best['arena_dd']:.1f}%  Sharpe={best['arena_sharpe']:.3f}")
    lines.append(f"- Synth: Return={best['synth_return']:+.2f}%  DD={best['synth_dd']:.1f}%  Sharpe={best['synth_sharpe']:.3f}")
    lines.append(f"- Arena improvement vs baseline: +{best['arena_improvement']:.3f}")
    lines.append(f"- Synthetic improvement vs baseline: +{best['synth_improvement']:.3f}")
    if best.get("overfit"):
        lines.append(f"- **WARNING: {best['overfit']} — treat result with caution**")

    lines.append("\n## Analysis Per Flag\n")
    for flag in "ABCD":
        flag_results = [r for r in results if r["flags"] == flag]
        if flag_results:
            fr = flag_results[0]
            arena_delta = fr["arena_return"] - baseline["arena_return"]
            synth_delta = fr["synth_return"] - baseline["synth_return"]
            lines.append(f"- **{flag}**: arena {arena_delta:+.2f}%  synth {synth_delta:+.2f}%  "
                         f"{'CONFIRMED' if arena_delta > 0 and synth_delta > 0 else 'MIXED' if arena_delta > 0 else 'REJECTED'}")

    with open("results/v2_experiments.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report -> results/v2_experiments.md")


if __name__ == "__main__":
    main()
