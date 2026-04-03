"""
compare.py — Compare multiple LEAN backtest result JSONs side-by-side.

Usage:
    python tools/compare.py results/larsa-v1/*/result.json
    python tools/compare.py results/larsa-v1/* results/larsa-v2/*
    python tools/compare.py results/larsa-v1/* --chart

Outputs a ranked table + optional equity curve overlay.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─── Metric extraction ────────────────────────────────────────────────────────

METRIC_KEYS = {
    "Total Return":      ("TotalPerformance", "PortfolioStatistics", "TotalNetProfit"),
    "CAGR":              ("TotalPerformance", "PortfolioStatistics", "CompoundingAnnualReturn"),
    "Sharpe":            ("TotalPerformance", "PortfolioStatistics", "SharpeRatio"),
    "Sortino":           ("TotalPerformance", "PortfolioStatistics", "SortinoRatio"),
    "Max Drawdown":      ("TotalPerformance", "PortfolioStatistics", "Drawdown"),
    "Win Rate":          ("TotalPerformance", "PortfolioStatistics", "WinRate"),
    "Profit Factor":     ("TotalPerformance", "PortfolioStatistics", "ProfitLossRatio"),
    "Total Trades":      ("TotalPerformance", "TradeStatistics", "NumberOfTrades"),
    "Total Fees":        ("TotalPerformance", "PortfolioStatistics", "TotalFees"),
}

HIGHER_IS_BETTER = {"Total Return", "CAGR", "Sharpe", "Sortino", "Win Rate", "Profit Factor", "Total Trades"}
LOWER_IS_BETTER  = {"Max Drawdown", "Total Fees"}


def _dig(d: Dict, *keys: str) -> Optional[Any]:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def load_result(path: str) -> Optional[Dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        name = Path(path).parent.name
        row = {"Name": name, "Path": path}
        for metric, key_path in METRIC_KEYS.items():
            val = _dig(data, *key_path)
            if isinstance(val, str):
                val = val.replace("%", "").strip()
                try:
                    val = float(val)
                except ValueError:
                    pass
            row[metric] = val
        return row
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}", file=sys.stderr)
        return None


# ─── Table rendering ──────────────────────────────────────────────────────────

ANSI_GREEN  = "\033[92m"
ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"


def _fmt(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def print_table(rows: List[Dict]):
    if not rows:
        print("No results to compare.")
        return

    metrics = list(METRIC_KEYS.keys())
    cols = ["Name"] + metrics

    # Find best per metric
    best: Dict[str, Any] = {}
    for m in metrics:
        vals = [(r[m], i) for i, r in enumerate(rows) if isinstance(r.get(m), (int, float))]
        if not vals:
            continue
        if m in HIGHER_IS_BETTER:
            best[m] = max(vals, key=lambda x: x[0])[1]
        elif m in LOWER_IS_BETTER:
            best[m] = min(vals, key=lambda x: x[0])[1]

    # Column widths
    widths = {c: len(c) for c in cols}
    for r in rows:
        widths["Name"] = max(widths["Name"], len(r["Name"]))
        for m in metrics:
            widths[m] = max(widths[m], len(_fmt(r.get(m))))

    # Header
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  ".join("-" * widths[c] for c in cols)
    print(f"\n{ANSI_BOLD}{header}{ANSI_RESET}")
    print(sep)

    for i, r in enumerate(rows):
        parts = [r["Name"].ljust(widths["Name"])]
        for m in metrics:
            cell = _fmt(r.get(m)).ljust(widths[m])
            if best.get(m) == i:
                cell = f"{ANSI_GREEN}{cell}{ANSI_RESET}"
            parts.append(cell)
        print("  ".join(parts))
    print()


# ─── Equity curve chart ───────────────────────────────────────────────────────

def plot_equity_curves(result_paths: List[str]):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
    except ImportError:
        print("[ERROR] matplotlib not installed. Run: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    for path in result_paths:
        try:
            with open(path) as f:
                data = json.load(f)
            charts = data.get("Charts", {})
            equity_chart = charts.get("Strategy Equity", {})
            series = equity_chart.get("Series", {}).get("Equity", {}).get("Values", [])
            if not series:
                continue
            dates = [datetime.utcfromtimestamp(p["x"]) for p in series]
            values = [p["y"] for p in series]
            label = Path(path).parent.name
            ax.plot(dates, values, label=label, linewidth=1.5)
        except Exception as e:
            print(f"[WARN] Could not plot {path}: {e}", file=sys.stderr)

    ax.set_title("Equity Curve Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = "results/equity_comparison.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Chart saved → {out}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare LEAN backtest results")
    parser.add_argument("paths", nargs="+", help="Paths to result JSON files or directories")
    parser.add_argument("--chart", action="store_true", help="Generate equity curve overlay")
    parser.add_argument("--sort", default="Sharpe", help="Sort by metric (default: Sharpe)")
    args = parser.parse_args()

    # Expand globs and directories
    result_files: List[str] = []
    for p in args.paths:
        expanded = glob.glob(p, recursive=True)
        for ep in expanded:
            if os.path.isdir(ep):
                result_files += glob.glob(os.path.join(ep, "**/result.json"), recursive=True)
                result_files += glob.glob(os.path.join(ep, "*.json"))
            else:
                result_files.append(ep)

    result_files = list(dict.fromkeys(result_files))  # deduplicate, preserve order

    rows = [load_result(p) for p in result_files]
    rows = [r for r in rows if r is not None]

    if args.sort in METRIC_KEYS:
        reverse = args.sort in HIGHER_IS_BETTER
        rows.sort(key=lambda r: r.get(args.sort) or (float("-inf") if reverse else float("inf")), reverse=reverse)

    print_table(rows)

    if args.chart:
        plot_equity_curves(result_files)


if __name__ == "__main__":
    main()
