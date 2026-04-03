"""
walkforward.py — Walk-forward validation for LARSA arena_v2.

Splits the bar data into overlapping train/test windows, grid-searches CF on
the train slice, then evaluates the best CF on the test slice.

Answers: "Are we overfitting to the 2018-2024 NDX data?"

Default windows (for ~35,000-bar hourly dataset):
  Window 1: train bars  0–20000, test bars 20000–26000
  Window 2: train bars  5000–25000, test bars 25000–31000
  Window 3: train bars 10000–30000, test bars 30000–35000

CF grid: [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015]

Degradation = (test_sharpe - train_sharpe) / |train_sharpe| * 100

Verdict thresholds:
  degradation > -30%  → OK
  -50% < degradation ≤ -30%  → WARN
  degradation ≤ -50%  → OVERFIT

Outputs:
  ASCII summary to stdout
  results/walkforward.md

Usage:
    python tools/walkforward.py --csv data/NDX_hourly_poly.csv
    python tools/walkforward.py  # uses synthetic data if CSV missing
    python tools/walkforward.py --csv data/NDX_hourly_poly.csv --synth-fallback
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from arena_v2 import run_v2, load_ohlcv, generate_synthetic, CONFIGS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

CF_GRID = [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015]
BASE_CFG = {"bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95}
MAX_LEV   = 0.65
EXP_FLAGS = "ABCD"

WINDOWS = [
    {"name": "1", "train": (0,     20000), "test": (20000, 26000)},
    {"name": "2", "train": (5000,  25000), "test": (25000, 31000)},
    {"name": "3", "train": (10000, 30000), "test": (30000, 35000)},
]


def sharpe_for(bars, cf):
    cfg = {**BASE_CFG, "cf": cf}
    try:
        broker, _ = run_v2(bars, cfg, max_leverage=MAX_LEV, exp_flags=EXP_FLAGS)
        s = broker.stats()
        return s.get("sharpe", 0.0)
    except Exception:
        return 0.0


def grid_search_cf(bars):
    """Return (best_cf, best_sharpe, all_results_list)."""
    results = []
    for cf in CF_GRID:
        sh = sharpe_for(bars, cf)
        results.append((cf, sh))
        print(f"      cf={cf:.3f}  sharpe={sh:.4f}")
    best = max(results, key=lambda x: x[1])
    return best[0], best[1], results


def degradation(train_sh, test_sh):
    if abs(train_sh) < 1e-9:
        return 0.0
    return (test_sh - train_sh) / abs(train_sh) * 100.0


def verdict(deg):
    if deg > -30.0:
        return "OK"
    if deg > -50.0:
        return "WARN"
    return "OVERFIT"


def run_windows(all_bars):
    n = len(all_bars)
    window_results = []

    for w in WINDOWS:
        tr_lo, tr_hi = w["train"]
        te_lo, te_hi = w["test"]

        # Clamp to available bars
        tr_hi = min(tr_hi, n)
        te_lo = min(te_lo, n)
        te_hi = min(te_hi, n)

        if tr_lo >= tr_hi or te_lo >= te_hi:
            print(f"  Window {w['name']}: insufficient bars, skipping.")
            continue

        train_bars = all_bars[tr_lo:tr_hi]
        test_bars  = all_bars[te_lo:te_hi]

        print(f"\n  Window {w['name']}: train [{tr_lo}:{tr_hi}] ({len(train_bars)} bars)  "
              f"test [{te_lo}:{te_hi}] ({len(test_bars)} bars)")
        print(f"    Grid-searching CF on train ...")
        best_cf, train_sh, _ = grid_search_cf(train_bars)

        print(f"    Evaluating best CF={best_cf:.3f} on test ...")
        test_sh = sharpe_for(test_bars, best_cf)

        deg = degradation(train_sh, test_sh)
        verd = verdict(deg)
        window_results.append({
            "window":       w["name"],
            "train_bars":   (tr_lo, tr_hi),
            "test_bars":    (te_lo, te_hi),
            "best_cf":      best_cf,
            "train_sharpe": round(train_sh, 4),
            "test_sharpe":  round(test_sh, 4),
            "degradation":  round(deg, 1),
            "verdict":      verd,
        })

    return window_results


def print_summary(results):
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)
    hdr = f"{'Window':>6}  {'Train Sh':>8}  {'Test Sh':>7}  {'Best CF':>7}  {'Degradation':>12}  Verdict"
    print(hdr)
    print("-" * 60)
    for r in results:
        print(f"  {r['window']:>4}    {r['train_sharpe']:>8.3f}  {r['test_sharpe']:>7.3f}  "
              f"{r['best_cf']:>7.3f}  {r['degradation']:>11.1f}%  {r['verdict']}")

    if results:
        avg_deg = np.mean([r["degradation"] for r in results])
        print(f"\nAvg degradation: {avg_deg:.1f}%")
        overfit_count = sum(1 for r in results if r["verdict"] == "OVERFIT")
        warn_count    = sum(1 for r in results if r["verdict"] == "WARN")
        if overfit_count >= 2:
            overall = "STRONG OVERFITTING (≥2 windows flagged)"
        elif overfit_count == 1 or warn_count >= 2:
            overall = "MODERATE OVERFITTING (>30% degradation typical)"
        elif warn_count == 1:
            overall = "MILD CONCERN (1 warning window)"
        else:
            overall = "ROBUST (degradation within acceptable range)"
        print(f"Verdict: {overall}")


def write_md(results, path):
    lines = ["# Walk-Forward Validation\n\n"]
    lines.append("CF grid: " + ", ".join(str(c) for c in CF_GRID) + "\n\n")
    lines.append("| Window | Train Bars | Test Bars | Train Sharpe | Test Sharpe | Best CF | Degradation | Verdict |\n")
    lines.append("|--------|-----------|----------|-------------|------------|---------|-------------|--------|\n")
    for r in results:
        tr = f"{r['train_bars'][0]}-{r['train_bars'][1]}"
        te = f"{r['test_bars'][0]}-{r['test_bars'][1]}"
        lines.append(f"| {r['window']} | {tr} | {te} | {r['train_sharpe']:.4f} | "
                     f"{r['test_sharpe']:.4f} | {r['best_cf']:.3f} | "
                     f"{r['degradation']:.1f}% | {r['verdict']} |\n")

    if results:
        avg_deg = float(np.mean([r["degradation"] for r in results]))
        lines.append(f"\nAverage degradation: **{avg_deg:.1f}%**\n")

    with open(path, "w") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward validation for LARSA arena_v2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv", default="data/NDX_hourly_poly.csv",
                        help="Path to OHLCV CSV (default: data/NDX_hourly_poly.csv)")
    parser.add_argument("--synth-bars", type=int, default=35000,
                        help="Synthetic bar count if CSV missing (default: 35000)")
    parser.add_argument("--synth-fallback", action="store_true",
                        help="Force synthetic data even if CSV exists")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", csv_path)

    if not args.synth_fallback and os.path.exists(csv_path):
        print(f"Loading data from {csv_path} ...")
        all_bars = load_ohlcv(csv_path)
        print(f"  {len(all_bars)} bars loaded.")
    else:
        if not args.synth_fallback:
            print(f"WARNING: {csv_path} not found — generating {args.synth_bars} synthetic bars.")
        else:
            print(f"Synth-fallback mode: generating {args.synth_bars} synthetic bars.")
        all_bars = generate_synthetic(args.synth_bars, seed=42)

    print(f"\nRunning walk-forward on {len(all_bars)} total bars ...")
    results = run_windows(all_bars)

    print_summary(results)

    md_path = os.path.join(RESULTS_DIR, "walkforward.md")
    write_md(results, md_path)
    print(f"\nDetailed results saved → {md_path}")


if __name__ == "__main__":
    main()
