"""
frontier.py — Efficient frontier across Sharpe/Drawdown trade-off.

Runs arena_v2 at many (cf × max_leverage) combinations and finds the
Pareto front: for each DD level, the maximum Sharpe achievable.

Usage:
    python tools/frontier.py --csv data/NDX_hourly_poly.csv
    python tools/frontier.py --synthetic          # use synthetic data
    python tools/frontier.py --csv foo.csv --seeds 3
"""

import argparse
import json
import os
import sys

# Allow importing arena_v2 from the same tools/ directory
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TOOLS_DIR)

RESULTS_DIR = os.path.join(_TOOLS_DIR, "..", "results")

# Grid
CF_VALUES  = [0.002, 0.004, 0.006, 0.008, 0.010]
LEV_VALUES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def run_one(bars, cf, max_lev):
    """Import arena lazily to avoid circular import issues."""
    from arena_v2 import run_v2, CONFIGS
    cfg = {"cf": cf, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95}
    broker, _log = run_v2(bars, cfg, max_leverage=max_lev, exp_flags="ABCD")
    s = broker.stats()
    return {
        "cf": cf,
        "max_lev": max_lev,
        "sharpe": round(s["sharpe"], 4),
        "drawdown": round(s["max_drawdown_pct"], 2),
        "ret": round(s["total_return_pct"], 2),
        "trades": s["trade_count"],
    }


def pareto_front(points):
    """
    Find Pareto-optimal points: higher Sharpe AND lower Drawdown.
    Returns sorted list of dicts on the front.
    """
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q["sharpe"] >= p["sharpe"] and q["drawdown"] <= p["drawdown"]:
                if q["sharpe"] > p["sharpe"] or q["drawdown"] < p["drawdown"]:
                    dominated = True
                    break
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda x: x["drawdown"])


def ascii_scatter(points, front, width=60, height=16):
    """Render Sharpe vs Drawdown ASCII scatter plot."""
    all_dd = [p["drawdown"] for p in points]
    all_sr = [p["sharpe"]   for p in points]

    dd_min = min(all_dd);  dd_max = max(all_dd)
    sr_min = min(all_sr);  sr_max = max(all_sr)

    dd_range = dd_max - dd_min if dd_max != dd_min else 1
    sr_range = sr_max - sr_min if sr_max != sr_min else 1

    grid = [[" "] * width for _ in range(height)]

    def to_xy(p):
        x = int((p["drawdown"] - dd_min) / dd_range * (width - 1))
        y = int((p["sharpe"]   - sr_min) / sr_range * (height - 1))
        y = height - 1 - y
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    front_set = {(p["cf"], p["max_lev"]) for p in front}

    for p in points:
        x, y = to_xy(p)
        key = (p["cf"], p["max_lev"])
        grid[y][x] = "P" if key in front_set else "."

    lines = []
    for ri, row in enumerate(grid):
        sr_val = sr_max - (ri / (height - 1)) * sr_range
        lines.append(f"  {sr_val:5.2f} |" + "".join(row))

    x_axis_label = "         +" + "-" * width
    lines.append(x_axis_label)

    # DD axis labels
    left_lbl  = f"{dd_min:.0f}%"
    right_lbl = f"{dd_max:.0f}%"
    pad = width - len(left_lbl) - len(right_lbl)
    lines.append(f"          {left_lbl}" + " " * pad + right_lbl + "  <-- Max Drawdown %")
    lines.append("")
    lines.append("  Legend: P = Pareto-optimal   . = dominated")
    return "\n".join(lines)


def format_report(points, front, n_seeds):
    lines = []
    lines.append("SHARPE / DRAWDOWN EFFICIENT FRONTIER")
    lines.append("=" * 50)
    lines.append(f"Grid: {len(CF_VALUES)} cf values × {len(LEV_VALUES)} leverage values = {len(CF_VALUES)*len(LEV_VALUES)} combos")
    lines.append(f"Seeds per combo: {n_seeds}  (results averaged)")
    lines.append(f"Total experiments: {len(CF_VALUES)*len(LEV_VALUES)*n_seeds}")
    lines.append("")
    lines.append("ALL RESULTS (cf, max_lev → sharpe, dd%, ret%):")
    lines.append(f"  {'cf':>6}  {'lev':>5}  {'Sharpe':>7}  {'MaxDD%':>7}  {'Ret%':>7}  {'Trades':>6}")
    lines.append("  " + "-" * 48)
    for p in sorted(points, key=lambda x: (-x["sharpe"], x["drawdown"])):
        lines.append(f"  {p['cf']:6.3f}  {p['max_lev']:5.2f}  "
                     f"{p['sharpe']:7.3f}  {p['drawdown']:7.1f}  "
                     f"{p['ret']:+7.1f}  {p['trades']:>6}")
    lines.append("")
    lines.append("PARETO FRONTIER (best Sharpe for given DD level):")
    lines.append(f"  {'DD ≤':>7}  {'Sharpe':>7}  {'cf':>6}  {'max_lev':>8}")
    lines.append("  " + "-" * 32)
    for p in front:
        lines.append(f"  {p['drawdown']:6.1f}%  {p['sharpe']:7.3f}  "
                     f"{p['cf']:6.3f}  {p['max_lev']:8.2f}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Sharpe/Drawdown efficient frontier for LARSA arena_v2."
    )
    parser.add_argument("--csv", help="Path to OHLCV price CSV (optional)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (ignores --csv)")
    parser.add_argument("--n-bars", type=int, default=20000,
                        help="Bars for synthetic data (default 20000)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds per config (averaged)")
    args = parser.parse_args()

    from arena_v2 import load_ohlcv, generate_synthetic

    # Build bar datasets
    if args.synthetic or not args.csv:
        if args.csv and not args.synthetic:
            if not os.path.exists(args.csv):
                print(f"WARNING: CSV not found ({args.csv}), falling back to synthetic data.")
        print(f"Using synthetic data ({args.n_bars} bars per seed).")
        bar_sets = [generate_synthetic(args.n_bars, seed=s) for s in range(args.seeds)]
    else:
        print(f"Loading OHLCV from {args.csv} ...")
        base_bars = load_ohlcv(args.csv)
        print(f"  {len(base_bars)} bars loaded.")
        bar_sets = [base_bars]  # single dataset; seeds not meaningful for real data
        args.seeds = 1

    combos = [(cf, lev) for cf in CF_VALUES for lev in LEV_VALUES]
    total = len(combos) * args.seeds
    print(f"Running {total} experiments ({len(combos)} combos × {args.seeds} seeds)...")
    print()

    results = []  # one dict per (cf, lev) — averaged across seeds

    for i, (cf, lev) in enumerate(combos):
        seed_sharpes  = []
        seed_dds      = []
        seed_rets     = []
        seed_trades   = []
        for bars in bar_sets:
            try:
                r = run_one(bars, cf, lev)
                seed_sharpes.append(r["sharpe"])
                seed_dds.append(r["drawdown"])
                seed_rets.append(r["ret"])
                seed_trades.append(r["trades"])
            except Exception as e:
                print(f"  WARNING: cf={cf} lev={lev} failed: {e}")

        if not seed_sharpes:
            continue

        avg = {
            "cf": cf,
            "max_lev": lev,
            "sharpe":   round(sum(seed_sharpes) / len(seed_sharpes), 4),
            "drawdown": round(sum(seed_dds)     / len(seed_dds),     2),
            "ret":      round(sum(seed_rets)    / len(seed_rets),     2),
            "trades":   int(sum(seed_trades)    / len(seed_trades)),
        }
        results.append(avg)
        print(f"  [{i+1:2d}/{len(combos)}] cf={cf:.3f} lev={lev:.2f} → "
              f"Sharpe={avg['sharpe']:.3f}  DD={avg['drawdown']:.1f}%  "
              f"Ret={avg['ret']:+.1f}%")

    if not results:
        print("ERROR: No experiments completed.")
        sys.exit(1)

    front = pareto_front(results)
    scatter = ascii_scatter(results, front)
    report  = format_report(results, front, args.seeds)

    full_output = report + "\nASCII SCATTER (Sharpe vs Drawdown):\n" + scatter + "\n"
    print()
    print(full_output)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    md_path = os.path.join(RESULTS_DIR, "frontier.md")
    with open(md_path, "w") as f:
        f.write("# Efficient Frontier — LARSA arena_v2\n\n```\n")
        f.write(full_output)
        f.write("\n```\n")
    print(f"Saved report: {md_path}")

    json_path = os.path.join(RESULTS_DIR, "frontier.json")
    with open(json_path, "w") as f:
        json.dump({"results": results, "pareto_front": front}, f, indent=2)
    print(f"Saved data:   {json_path}")


if __name__ == "__main__":
    main()
