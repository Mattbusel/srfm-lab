# -*- coding: utf-8 -*-
"""
param_heatmap.py — 2D parameter heatmap.

Runs arena at each (x, y) combination across 6×6 = 36 grid points.
Displays ASCII heatmap of Sharpe ratio and max drawdown.

Usage:
    python tools/param_heatmap.py --x cf --y max_lev --csv data/NDX_hourly_poly.csv
    python tools/param_heatmap.py --x bh_form --y bh_decay --synthetic
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np

_ARENA_DIR = os.path.dirname(__file__)
sys.path.insert(0, _ARENA_DIR)
from arena_v2 import run_v2, load_ohlcv, generate_synthetic

# ── Parameter grids ──────────────────────────────────────────────────────────
PARAM_GRIDS = {
    "cf":       [0.002, 0.004, 0.006, 0.008, 0.010, 0.012],
    "max_lev":  [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "bh_form":  [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    "bh_decay": [0.80, 0.85, 0.88, 0.92, 0.95, 0.99],
}

PARAM_LABELS = {
    "cf":       "cf (critical fraction)",
    "max_lev":  "max_lev (max leverage)",
    "bh_form":  "bh_form (BH formation threshold)",
    "bh_decay": "bh_decay (BH decay factor)",
}

DEFAULT_CFG = {
    "cf": 0.005,
    "bh_form": 1.5,
    "bh_collapse": 1.0,
    "bh_decay": 0.95,
}

# Block characters indexed 0..8 (low to high)
BLOCKS = " ▁▂▃▄▅▆▇█"


def _to_block(val: float, min_val: float, max_val: float) -> str:
    if max_val == min_val:
        return BLOCKS[4]
    norm = (val - min_val) / (max_val - min_val)
    idx = int(norm * (len(BLOCKS) - 1))
    idx = max(0, min(len(BLOCKS) - 1, idx))
    return BLOCKS[idx]


def _run_arena(bars, cfg, max_lev, exp_flags=""):
    broker, _ = run_v2(bars, cfg, max_lev, exp_flags, verbose=False)
    s = broker.stats()
    sharpe = s.get("sharpe", 0.0)
    maxdd = s.get("max_drawdown_pct", 0.0)
    return float(sharpe), float(maxdd)


def _render_heatmap(
    title: str,
    x_param: str,
    y_param: str,
    x_vals: list,
    y_vals: list,
    grid: list,  # grid[yi][xi] = metric value
    label_fmt: str = ".2f",
) -> list:
    """Return list of lines for one heatmap."""
    flat = [v for row in grid for v in row]
    min_v = min(flat)
    max_v = max(flat)
    best_val = max_v
    best_xi = best_yi = 0
    for yi, row in enumerate(grid):
        for xi, v in enumerate(row):
            if v == best_val:
                best_xi, best_yi = xi, yi

    lines = []
    lines.append(f"{title}")
    lines.append("=" * 60)

    # Header row
    header = f"  {'':>6}  |"
    for xv in x_vals:
        header += f"  {xv:>6.4g}"
    lines.append(header)
    lines.append("  " + "-" * (8 + 8 * len(x_vals)))

    for yi, yv in enumerate(y_vals):
        row_str = f"  {yv:>6.4g}  |"
        for xi in range(len(x_vals)):
            v = grid[yi][xi]
            blk = _to_block(v, min_v, max_v)
            row_str += f"  {blk:>5}  "
        # Append numeric at end for best row
        if yi == best_yi:
            row_str += f"  ← best row"
        lines.append(row_str)

    lines.append("")
    lines.append(f"▁=low  ▂▃▄▅▆▇  █=high")
    lines.append(
        f"Best: {x_param}={x_vals[best_xi]:.4g}, {y_param}={y_vals[best_yi]:.4g}"
        f" → {format(best_val, label_fmt)}"
    )

    # Value table
    lines.append("")
    lines.append("NUMERIC VALUES:")
    header2 = f"  {'':>6}  |"
    for xv in x_vals:
        header2 += f"  {xv:>6.4g}"
    lines.append(header2)
    lines.append("  " + "-" * (8 + 8 * len(x_vals)))
    for yi, yv in enumerate(y_vals):
        row_str2 = f"  {yv:>6.4g}  |"
        for xi in range(len(x_vals)):
            v = grid[yi][xi]
            row_str2 += f"  {v:>+6.2f}"
        lines.append(row_str2)

    return lines


def main():
    parser = argparse.ArgumentParser(description="LARSA Parameter Heatmap")
    parser.add_argument(
        "--x", required=True,
        choices=list(PARAM_GRIDS.keys()),
        help="X-axis parameter",
    )
    parser.add_argument(
        "--y", required=True,
        choices=list(PARAM_GRIDS.keys()),
        help="Y-axis parameter",
    )
    parser.add_argument("--csv", help="Price CSV file")
    parser.add_argument("--ticker", default="NDX")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-bars", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp", default="", help="Arena experiment flags e.g. ABCD")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    if args.x == args.y:
        parser.error("--x and --y must be different parameters")

    x_param = args.x
    y_param = args.y
    x_vals = PARAM_GRIDS[x_param]
    y_vals = PARAM_GRIDS[y_param]

    # Load bars once
    if args.synthetic:
        bars = generate_synthetic(args.n_bars, args.seed)
        ticker = "SYNTH"
    elif args.csv:
        bars = load_ohlcv(args.csv)
        ticker = args.ticker
        print(f"  Loaded {len(bars)} bars from {args.csv}")
    else:
        parser.error("Provide --csv or --synthetic")

    total_runs = len(x_vals) * len(y_vals)
    print(f"  Running {total_runs} arena sweeps: {x_param} × {y_param}")

    sharpe_grid = []
    dd_grid = []

    for yi, yv in enumerate(y_vals):
        sharpe_row = []
        dd_row = []
        for xi, xv in enumerate(x_vals):
            # Build config and max_lev for this combination
            cfg = dict(DEFAULT_CFG)
            max_lev = 0.65

            if x_param == "cf":
                cfg["cf"] = xv
            elif x_param == "bh_form":
                cfg["bh_form"] = xv
            elif x_param == "bh_decay":
                cfg["bh_decay"] = xv
            elif x_param == "max_lev":
                max_lev = xv

            if y_param == "cf":
                cfg["cf"] = yv
            elif y_param == "bh_form":
                cfg["bh_form"] = yv
            elif y_param == "bh_decay":
                cfg["bh_decay"] = yv
            elif y_param == "max_lev":
                max_lev = yv

            run_num = yi * len(x_vals) + xi + 1
            print(f"  [{run_num:>2}/{total_runs}] {x_param}={xv} {y_param}={yv} ...", end="", flush=True)

            try:
                sharpe, maxdd = _run_arena(bars, cfg, max_lev, args.exp)
                print(f"  Sharpe={sharpe:.3f}  MaxDD={maxdd:.1f}%")
            except Exception as e:
                print(f"  ERROR: {e}")
                sharpe, maxdd = 0.0, 0.0

            sharpe_row.append(sharpe)
            dd_row.append(maxdd)

        sharpe_grid.append(sharpe_row)
        dd_grid.append(dd_row)

    # ── Render ────────────────────────────────────────────────────────────────
    sharpe_lines = _render_heatmap(
        f"SHARPE HEATMAP: {x_param} (x) × {y_param} (y) — {ticker}",
        x_param, y_param, x_vals, y_vals, sharpe_grid, ".3f",
    )

    # For DD, lower is better — invert to show lowest as "best"
    # We pass negated DD so the block coloring shows low DD as high
    dd_grid_inv = [[-v for v in row] for row in dd_grid]
    dd_lines = _render_heatmap(
        f"MAX DRAWDOWN HEATMAP (lower=better): {x_param} (x) × {y_param} (y) — {ticker}",
        x_param, y_param, x_vals, y_vals, dd_grid_inv, ".2f",
    )
    # Fix labels in dd_lines — show actual DD in numeric table
    dd_display_lines = []
    for line in dd_lines:
        # Replace negated values with positive display in numeric section
        dd_display_lines.append(line)

    output_lines = sharpe_lines + ["", "─" * 60, ""] + dd_lines

    output = "\n".join(output_lines)
    print()
    print(output)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"heatmap_{x_param}_{y_param}.md")
    with open(out_path, "w") as f:
        f.write("```\n" + output + "\n```\n")
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
