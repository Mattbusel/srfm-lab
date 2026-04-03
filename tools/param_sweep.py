"""
param_sweep.py — Sweep a single parameter across a range and plot the sensitivity surface.

Usage:
    python tools/param_sweep.py strategies/larsa-v1 bh_form 0.5 3.0 0.25
    python tools/param_sweep.py strategies/larsa-v1 cf 0.8 2.0 0.1 --metric Sharpe

How it works:
    1. For each parameter value, patch the strategy's main.py (regex replace of the constant).
    2. Run `lean backtest` in a temp directory.
    3. Extract the target metric from the result JSON.
    4. Plot the sensitivity surface and save a CSV summary.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple


# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_METRIC = "SharpeRatio"
RESULTS_DIR    = "results"


# ─── Parameter patching ───────────────────────────────────────────────────────

def patch_main(src_path: str, param: str, value: float, dst_path: str):
    """Copy src_path/main.py to dst_path, replacing PARAM = <value>."""
    with open(src_path) as f:
        code = f.read()

    # Match: PARAM_NAME = <number>  (with optional comment)
    pattern = rf"^({re.escape(param)}\s*=\s*)[^\n#]+"
    replacement = rf"\g<1>{value!r}"
    new_code, n = re.subn(pattern, replacement, code, flags=re.MULTILINE)

    if n == 0:
        print(f"[WARN] '{param}' not found in {src_path} — check the constant name.", file=sys.stderr)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        f.write(new_code)


# ─── LEAN runner ──────────────────────────────────────────────────────────────

def run_lean_backtest(strategy_dir: str, output_dir: str) -> str:
    """Run lean backtest and return the path to the result JSON."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "lean", "backtest", strategy_dir,
        "--output", output_dir,
        "--detach",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] lean backtest failed:\n{result.stderr}", file=sys.stderr)
        return ""

    # Find result JSON
    for name in ["result.json", "backtest-results.json"]:
        candidate = os.path.join(output_dir, name)
        if os.path.exists(candidate):
            return candidate

    jsons = list(Path(output_dir).glob("*.json"))
    return str(jsons[0]) if jsons else ""


def extract_metric(result_json: str, metric: str) -> float:
    """Extract a performance metric from the LEAN result JSON."""
    try:
        with open(result_json) as f:
            data = json.load(f)
        stats = (
            data.get("TotalPerformance", {})
                .get("PortfolioStatistics", {})
        )
        val = stats.get(metric)
        if val is None:
            # Try trade stats
            val = (data.get("TotalPerformance", {})
                       .get("TradeStatistics", {})
                       .get(metric))
        if isinstance(val, str):
            val = val.replace("%", "").strip()
            val = float(val)
        return float(val) if val is not None else float("nan")
    except Exception as e:
        print(f"[WARN] Could not extract {metric}: {e}", file=sys.stderr)
        return float("nan")


# ─── Sweep ────────────────────────────────────────────────────────────────────

def sweep(
    strategy_dir: str,
    param: str,
    values: List[float],
    metric: str,
    parallel: bool = False,
) -> List[Tuple[float, float]]:
    results: List[Tuple[float, float]] = []
    strategy_name = Path(strategy_dir).name
    src_main = os.path.join(strategy_dir, "main.py")

    for val in values:
        print(f"  [{param}={val:.4f}] running backtest...", end=" ", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            # Copy full strategy dir; patch main.py
            tmp_strategy = os.path.join(tmp, "strategy")
            shutil.copytree(strategy_dir, tmp_strategy)
            patch_main(src_main, param, val, os.path.join(tmp_strategy, "main.py"))

            out_dir = os.path.join(RESULTS_DIR, strategy_name, f"sweep_{param}_{val:.4f}")
            result_path = run_lean_backtest(tmp_strategy, out_dir)

            if result_path:
                m = extract_metric(result_path, metric)
                print(f"{metric}={m:.4f}")
            else:
                m = float("nan")
                print("FAILED")

        results.append((val, m))

    return results


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_sensitivity(
    param: str,
    values: List[float],
    scores: List[float],
    metric: str,
    strategy_name: str,
    save_path: str,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    clean = [(v, s) for v, s in zip(values, scores) if not math.isnan(s)]
    if not clean:
        print("[WARN] No valid results to plot.")
        return

    xs, ys = zip(*clean)
    ax.plot(xs, ys, "o-", linewidth=2, markersize=6)
    ax.axhline(y=max(ys), color="green", linestyle="--", alpha=0.5, label=f"Best: {max(ys):.4f}")
    ax.set_title(f"Sensitivity: {strategy_name} / {param} → {metric}", fontweight="bold")
    ax.set_xlabel(param)
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved → {save_path}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SRFM parameter sweep")
    parser.add_argument("strategy",    help="Path to strategy directory")
    parser.add_argument("param",       help="Constant name in main.py to sweep")
    parser.add_argument("min",         type=float, help="Minimum value (inclusive)")
    parser.add_argument("max",         type=float, help="Maximum value (inclusive)")
    parser.add_argument("step",        type=float, help="Step size")
    parser.add_argument("--metric",    default=DEFAULT_METRIC, help="Performance metric to optimise")
    parser.add_argument("--no-plot",   action="store_true")
    args = parser.parse_args()

    # Build value list
    values: List[float] = []
    v = args.min
    while v <= args.max + 1e-9:
        values.append(round(v, 8))
        v += args.step

    strategy_name = Path(args.strategy).name
    print(f"\nSweep: {strategy_name} / {args.param} ∈ [{args.min}, {args.max}] step={args.step}")
    print(f"Values: {values}")
    print(f"Metric: {args.metric}\n")

    results = sweep(args.strategy, args.param, values, args.metric)
    scores = [s for _, s in results]

    # CSV summary
    csv_path = os.path.join(RESULTS_DIR, strategy_name, f"sweep_{args.param}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write(f"{args.param},{args.metric}\n")
        for val, score in results:
            f.write(f"{val},{score}\n")
    print(f"CSV saved → {csv_path}")

    # Best
    valid = [(v, s) for v, s in results if not math.isnan(s)]
    if valid:
        best_val, best_score = max(valid, key=lambda x: x[1])
        print(f"\nBest: {args.param}={best_val} → {args.metric}={best_score:.4f}")

    if not args.no_plot:
        plot_path = os.path.join(RESULTS_DIR, strategy_name, f"sweep_{args.param}.png")
        plot_sensitivity(args.param, values, scores, args.metric, strategy_name, plot_path)


if __name__ == "__main__":
    main()
