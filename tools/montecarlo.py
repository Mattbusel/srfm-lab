"""
montecarlo.py — Monte Carlo simulation of LARSA equity curves.

Reads research/trade_analysis_data.json for actual trade P&L distribution,
then bootstraps 10,000 random paths by resampling trades.

Usage:
    python tools/montecarlo.py --paths 10000 --capital 1000000
    python tools/montecarlo.py --paths 1000 --quick
    python tools/montecarlo.py --paths 5000 --leverage 1.5
"""

import argparse
import json
import os
import sys
import math
import random

# ── numpy optional but strongly preferred ──────────────────────────────────────
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "research", "trade_analysis_data.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── stats helpers (pure-python fallbacks) ──────────────────────────────────────

def percentile(sorted_data, p):
    """Return the p-th percentile of already-sorted data."""
    n = len(sorted_data)
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def max_drawdown(equity_curve):
    """Return maximum drawdown fraction (0-1) of an equity curve list."""
    peak = equity_curve[0]
    dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        d = (peak - v) / peak if peak > 0 else 0
        if d > dd:
            dd = d
    return dd


def approx_sharpe(returns_list):
    """Approximate annualised Sharpe from a list of per-trade returns."""
    n = len(returns_list)
    if n < 2:
        return 0.0
    mean_r = sum(returns_list) / n
    var = sum((r - mean_r) ** 2 for r in returns_list) / (n - 1)
    std = math.sqrt(var) if var > 0 else 1e-9
    # Assume ~60 trades/year (263 trades over ~6 years)
    trades_per_year = 263 / 6.0
    return (mean_r / std) * math.sqrt(trades_per_year)


# ── simulation ─────────────────────────────────────────────────────────────────

def run_simulation(pnls, n_paths, capital, leverage, n_trades=None):
    """
    Bootstrap Monte Carlo.

    pnls      : list of dollar P&L per trade
    n_paths   : number of simulation paths
    capital   : starting capital $
    leverage  : scale factor applied to each P&L
    n_trades  : trades per path (defaults to len(pnls))

    Returns dict with:
      final_equities, max_drawdowns, sharpes, percentile_curves
    """
    if n_trades is None:
        n_trades = len(pnls)

    scaled = [p * leverage for p in pnls]
    n_wells = len(scaled)

    # Percentile curve resolution: 50 evenly-spaced points along the path
    CURVE_POINTS = 50

    final_equities = []
    max_dds = []
    sharpes = []
    # bucket percentile curves
    curve_buckets = [[] for _ in range(CURVE_POINTS)]

    for _ in range(n_paths):
        # bootstrap sample
        sampled = [scaled[random.randint(0, n_wells - 1)] for _ in range(n_trades)]
        equity = capital
        curve = [equity]
        returns = []
        for p in sampled:
            r = p / equity if equity > 0 else 0
            equity += p
            if equity < 0:
                equity = 0
            returns.append(r)
            curve.append(equity)

        dd = max_drawdown(curve)
        final_equities.append(equity)
        max_dds.append(dd)
        sharpes.append(approx_sharpe(returns))

        # sample curve at CURVE_POINTS indices
        step = (len(curve) - 1) / (CURVE_POINTS - 1)
        for i in range(CURVE_POINTS):
            idx = int(round(i * step))
            idx = min(idx, len(curve) - 1)
            curve_buckets[i].append(curve[idx])

    # Sort everything
    final_equities.sort()
    max_dds.sort()
    sharpes.sort()
    for bucket in curve_buckets:
        bucket.sort()

    # Build percentile curves (5th, 25th, 50th, 75th, 95th)
    pcts = [5, 10, 25, 50, 75, 90, 95]
    pct_curves = {}
    for p in pcts:
        pct_curves[str(p)] = [percentile(b, p) for b in curve_buckets]

    return {
        "final_equities": final_equities,
        "max_drawdowns": max_dds,
        "sharpes": sharpes,
        "pct_curves": pct_curves,
        "n_paths": n_paths,
        "n_trades": n_trades,
        "capital": capital,
        "leverage": leverage,
    }


# ── ASCII equity chart ─────────────────────────────────────────────────────────

def ascii_equity_chart(pct_curves, capital, width=60, height=12):
    """Render 10th/50th/90th percentile curves as ASCII."""
    c10 = pct_curves["10"]
    c50 = pct_curves["50"]
    c90 = pct_curves["90"]

    n = len(c50)
    all_vals = c10 + c50 + c90
    y_min = min(all_vals)
    y_max = max(all_vals)
    y_range = y_max - y_min if y_max != y_min else 1

    rows = [[" "] * width for _ in range(height)]

    def plot_curve(curve, char):
        for xi in range(width):
            ci = int(xi / (width - 1) * (n - 1))
            ci = min(ci, n - 1)
            val = curve[ci]
            yi = int((val - y_min) / y_range * (height - 1))
            yi = height - 1 - yi
            yi = max(0, min(height - 1, yi))
            if rows[yi][xi] == " ":
                rows[yi][xi] = char

    plot_curve(c10, ".")
    plot_curve(c90, "^")
    plot_curve(c50, "*")

    lines = []
    for ri, row in enumerate(rows):
        val = y_max - (ri / (height - 1)) * y_range
        label = f"  ${val/1e6:5.2f}M |"
        lines.append(label + "".join(row))

    x_axis = "         +" + "-" * width
    lines.append(x_axis)
    lines.append("          Start" + " " * (width - 20) + "End")
    lines.append("")
    lines.append("  Legend: * = 50th pct   ^ = 90th pct   . = 10th pct")
    return "\n".join(lines)


# ── report formatting ──────────────────────────────────────────────────────────

def format_report(res):
    fe = res["final_equities"]
    dds = res["max_drawdowns"]
    sr = res["sharpes"]
    cap = res["capital"]
    n_paths = res["n_paths"]
    n_trades = res["n_trades"]
    lev = res["leverage"]

    pct5  = percentile(fe, 5)
    pct25 = percentile(fe, 25)
    pct50 = percentile(fe, 50)
    pct75 = percentile(fe, 75)
    pct95 = percentile(fe, 95)

    dd5  = percentile(dds, 5)
    dd50 = percentile(dds, 50)
    dd95 = percentile(dds, 95)

    sr5  = percentile(sr, 5)
    sr50 = percentile(sr, 50)
    sr95 = percentile(sr, 95)

    ruin = sum(1 for d in dds if d > 0.50) / n_paths * 100
    loss = sum(1 for e in fe if e < cap) / n_paths * 100

    lines = []
    lines.append("MONTE CARLO SIMULATION")
    lines.append("=" * 54)
    lines.append(f"Paths simulated:    {n_paths:,}")
    lines.append(f"Starting capital:   ${cap:,.0f}")
    lines.append(f"Trades per path:    {n_trades} wells (resampled)")
    if lev != 1.0:
        lines.append(f"Leverage applied:   {lev:.2f}x")
    lines.append("")
    lines.append("RETURN DISTRIBUTION (end of path):")
    for p, v in [(5, pct5), (25, pct25), (50, pct50), (75, pct75), (95, pct95)]:
        ret = (v - cap) / cap * 100
        lines.append(f"  {p:2d}th percentile:  {ret:+7.1f}%   (${v:,.0f})")
    lines.append("")
    lines.append("MAX DRAWDOWN DISTRIBUTION:")
    lines.append(f"  5th pct (best):   {dd5*100:.1f}%")
    lines.append(f"  Median:          {dd50*100:.1f}%")
    lines.append(f"  95th pct (worst): {dd95*100:.1f}%")
    lines.append("")
    lines.append("SHARPE RATIO DISTRIBUTION (approx, annualised):")
    lines.append(f"  5th pct:   {sr5:.2f}")
    lines.append(f"  Median:    {sr50:.2f}")
    lines.append(f"  95th pct:  {sr95:.2f}")
    lines.append("")
    lines.append(f"RISK OF RUIN (DD > 50%): {ruin:.1f}% of paths")
    lines.append(f"PROBABILITY OF LOSS:     {loss:.1f}% of paths")
    lines.append("")
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation of LARSA equity curves."
    )
    parser.add_argument("--paths", type=int, default=10000, help="Number of simulation paths (default 10000)")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Starting capital $ (default 1000000)")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage multiplier on P&L (default 1.0)")
    parser.add_argument("--quick", action="store_true", help="Text output only, skip saving files")
    parser.add_argument(
        "--data", default=DATA_PATH,
        help="Path to trade_analysis_data.json",
    )
    args = parser.parse_args()

    # Load data
    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        print("       Run the LARSA backtest first to generate research/trade_analysis_data.json")
        sys.exit(1)

    with open(data_path) as f:
        data = json.load(f)

    wells = data["wells"]
    pnls = [w["net_pnl"] for w in wells]
    print(f"Loaded {len(pnls)} wells from {data_path}")
    print(f"Running {args.paths:,} Monte Carlo paths...")

    random.seed(42)
    res = run_simulation(pnls, args.paths, args.capital, args.leverage)

    report = format_report(res)
    chart = ascii_equity_chart(res["pct_curves"], args.capital)

    full_output = report + "\nASCII EQUITY CURVE (10th/50th/90th pct):\n" + chart + "\n"
    print(full_output)

    if not args.quick:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        md_path = os.path.join(RESULTS_DIR, "montecarlo.md")
        with open(md_path, "w") as f:
            f.write("# Monte Carlo Simulation — LARSA\n\n```\n")
            f.write(full_output)
            f.write("\n```\n")
        print(f"Saved report: {md_path}")

        json_path = os.path.join(RESULTS_DIR, "montecarlo_paths.json")
        # Convert pct_curves to serialisable form
        save_curves = {}
        for k, v in res["pct_curves"].items():
            save_curves[k] = [round(x, 2) for x in v]
        with open(json_path, "w") as f:
            json.dump({
                "n_paths": res["n_paths"],
                "n_trades": res["n_trades"],
                "capital": res["capital"],
                "leverage": res["leverage"],
                "percentile_curves": save_curves,
            }, f, indent=2)
        print(f"Saved curves:  {json_path}")


if __name__ == "__main__":
    main()
