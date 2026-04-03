#!/usr/bin/env python
"""
primitive_builder.py — Computes SRFM physics primitives and writes them as
structured markdown tables to results/primitives.md.

Usage:
    python tools/primitive_builder.py [--csv data/NDX_hourly_poly.csv]
                                      [--cf 0.005]
                                      [--out results/primitives.md]
"""

import argparse
import json
import math
import os
import sys
import warnings
from datetime import datetime

# ---------------------------------------------------------------------------
# optional deps — numpy/pandas may or may not be present
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("pandas/numpy not found — some sections will be skipped")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo(path: str) -> str:
    """Return absolute path relative to repo root."""
    return os.path.join(REPO_ROOT, path)


def fmt(val, decimals=2, width=None):
    """Format a number, returning '—' for None/NaN."""
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "—"
        s = f"{val:,.{decimals}f}"
        return s if width is None else s.rjust(width)
    except (TypeError, ValueError):
        return str(val)


def pct(num, denom, decimals=1):
    if denom == 0:
        return "0.0%"
    return f"{100.0 * num / denom:.{decimals}f}%"


def md_table(headers: list, rows: list) -> str:
    """Render a simple markdown table."""
    col_widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        str_row = [str(c) for c in row]
        for i, cell in enumerate(str_row):
            col_widths[i] = max(col_widths[i], len(cell))
        str_rows.append(str_row)

    def pad(s, w):
        return s.ljust(w)

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header = "| " + " | ".join(pad(h, col_widths[i]) for i, h in enumerate(headers)) + " |"
    lines = [header, sep]
    for row in str_rows:
        lines.append("| " + " | ".join(pad(cell, col_widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 1: Beta Distribution
# ---------------------------------------------------------------------------

BETA_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 5.0, float("inf")]
BETA_LABELS = [
    "[0.0-0.2)",
    "[0.2-0.4)",
    "[0.4-0.6)",
    "[0.6-0.8)",
    "[0.8-1.0)",
    "[1.0-1.5)",
    "[1.5-2.0)",
    "[2.0-5.0)",
    "[5.0-∞)",
]


def compute_beta_section(csv_path: str, cf: float) -> str:
    lines = ["## Section 1: Beta Distribution\n"]

    if not HAS_PANDAS:
        lines.append("_Skipped — pandas not available._\n")
        return "\n".join(lines)

    if not os.path.exists(csv_path):
        lines.append(f"_Skipped — CSV not found: `{csv_path}`_\n")
        return "\n".join(lines)

    df = pd.read_csv(csv_path, parse_dates=["date"])

    # Δclose / close gives fractional move; divide by cf to get beta
    delta = df["close"].diff().abs()
    close_prev = df["close"].shift(1)
    frac = delta / close_prev
    beta = frac / cf

    # Drop first bar (NaN)
    beta = beta.dropna()
    n_total = len(beta)

    # Bin counts
    counts = [0] * (len(BETA_BINS) - 1)
    for b in beta:
        for i in range(len(BETA_BINS) - 1):
            if BETA_BINS[i] <= b < BETA_BINS[i + 1]:
                counts[i] += 1
                break

    # TIMELIKE: beta < 1.0 (bins 0-4), SPACELIKE: beta >= 1.0 (bins 5-8)
    timelike_count = sum(counts[:5])
    spacelike_count = sum(counts[5:])

    median_beta = float(np.median(beta))
    p95_beta = float(np.percentile(beta, 95))

    rows = []
    for i, label in enumerate(BETA_LABELS):
        btype = "TIMELIKE" if i < 5 else "SPACELIKE"
        rows.append([label, str(counts[i]), pct(counts[i], n_total), btype])

    lines.append(md_table(["Beta Range", "Count", "Pct", "Type"], rows))
    lines.append("")
    lines.append(f"- **N bars:** {n_total:,}")
    lines.append(f"- **TIMELIKE fraction:** {pct(timelike_count, n_total)}")
    lines.append(f"- **SPACELIKE fraction:** {pct(spacelike_count, n_total)}")
    lines.append(f"- **Median beta:** {fmt(median_beta, 4)}")
    lines.append(f"- **95th percentile beta:** {fmt(p95_beta, 4)}")
    lines.append(f"- **CF used:** {cf}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 2: Regime Statistics
# ---------------------------------------------------------------------------

def compute_regime_section(regimes_path: str) -> str:
    lines = ["## Section 2: Regime Statistics\n"]

    if not HAS_PANDAS:
        lines.append("_Skipped — pandas not available._\n")
        return "\n".join(lines)

    if not os.path.exists(regimes_path):
        lines.append(f"_Skipped — file not found: `{regimes_path}`_\n")
        return "\n".join(lines)

    df = pd.read_csv(regimes_path)
    total = len(df)

    grouped = df.groupby("regime").agg(
        bars=("regime", "count"),
        avg_conf=("confidence", "mean"),
    ).reset_index()
    grouped = grouped.sort_values("bars", ascending=False)

    rows = []
    for _, row in grouped.iterrows():
        rows.append([
            row["regime"],
            str(int(row["bars"])),
            pct(row["bars"], total),
            fmt(row["avg_conf"], 4),
        ])

    lines.append(md_table(["Regime", "Bars", "Pct", "Avg Confidence"], rows))
    lines.append(f"\n_Total bars: {total:,}_\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 3: BH Well Physics
# ---------------------------------------------------------------------------

def compute_well_section(wells_path: str) -> str:
    lines = ["## Section 3: BH Well Physics\n"]

    if not HAS_PANDAS:
        lines.append("_Skipped — pandas not available._\n")
        return "\n".join(lines)

    if not os.path.exists(wells_path):
        lines.append(f"_Skipped — file not found: `{wells_path}`_\n")
        return "\n".join(lines)

    df = pd.read_csv(wells_path, parse_dates=["formed_at", "collapsed_at"])
    n = len(df)

    def dist_row(label, series):
        s = series.dropna()
        return [
            label,
            fmt(s.min(), 4),
            fmt(float(np.median(s)), 4),
            fmt(s.max(), 4),
            fmt(s.mean(), 4),
        ]

    dist_headers = ["Metric", "Min", "Median", "Max", "Mean"]
    dist_rows = [
        dist_row("mass_peak", df["mass_peak"]),
        dist_row("duration_bars", df["duration_bars"]),
        dist_row("hawking_temp", df["hawking_temp"]),
        dist_row("price_move_pct", df["price_move_pct"]),
    ]

    lines.append("### Distribution Summary\n")
    lines.append(md_table(dist_headers, dist_rows))
    lines.append("")

    # Direction split
    dir_counts = df["direction"].value_counts()
    dir_rows = []
    for direction, cnt in dir_counts.items():
        dir_rows.append([direction, str(cnt), pct(cnt, n)])
    lines.append("### Direction Split\n")
    lines.append(md_table(["Direction", "Count", "Pct"], dir_rows))
    lines.append("")

    # Price move: classify win/lose by direction match
    # LONG well: price_move_pct > 0 is a win; SHORT well: price_move_pct < 0 is a win
    def is_win(row):
        if row["direction"] == "LONG":
            return row["price_move_pct"] > 0
        else:
            return row["price_move_pct"] < 0

    df["well_win"] = df.apply(is_win, axis=1)
    wins = df[df["well_win"]]
    losses = df[~df["well_win"]]

    pm_rows = [
        ["Winners", str(len(wins)), fmt(wins["price_move_pct"].mean(), 4), fmt(wins["price_move_pct"].abs().mean(), 4)],
        ["Losers",  str(len(losses)), fmt(losses["price_move_pct"].mean(), 4), fmt(losses["price_move_pct"].abs().mean(), 4)],
    ]
    lines.append("### Price Move by Outcome\n")
    lines.append(md_table(["Outcome", "Count", "Avg Move %", "Avg |Move| %"], pm_rows))
    lines.append(f"\n_N wells: {n}_\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 4: Convergence Matrix
# ---------------------------------------------------------------------------

def compute_convergence_section(analysis_path: str) -> str:
    lines = ["## Section 4: Convergence Matrix\n"]

    if not os.path.exists(analysis_path):
        lines.append(f"_Skipped — file not found: `{analysis_path}`_\n")
        return "\n".join(lines)

    with open(analysis_path) as f:
        data = json.load(f)

    wells = data.get("wells", [])
    if not wells:
        lines.append("_No wells data found._\n")
        return "\n".join(lines)

    # Group by instrument count
    groups = {"1": [], "2": [], "3+": []}
    for w in wells:
        instr = w.get("instruments", [])
        n_instr = len(instr)
        key = "1" if n_instr == 1 else ("2" if n_instr == 2 else "3+")
        groups[key].append(w)

    # Compute gross pnl across all wells for percentage
    gross_pnl = sum(w.get("total_pnl", 0) for w in wells if w.get("total_pnl", 0) > 0)

    rows = []
    for group_name in ["1", "2", "3+"]:
        wlist = groups[group_name]
        if not wlist:
            continue
        count = len(wlist)
        wins = sum(1 for w in wlist if w.get("is_win", False))
        win_rate = 100.0 * wins / count if count else 0.0
        avg_pnl = sum(w.get("total_pnl", 0) for w in wlist) / count if count else 0.0
        total_pnl = sum(w.get("total_pnl", 0) for w in wlist)
        pct_gross = 100.0 * total_pnl / gross_pnl if gross_pnl else 0.0
        rows.append([
            f"{group_name} instrument{'s' if group_name != '1' else ''}",
            str(count),
            f"{win_rate:.1f}%",
            f"${avg_pnl:,.0f}",
            f"${total_pnl:,.0f}",
            f"{pct_gross:.1f}%",
        ])

    lines.append(md_table(
        ["Group", "Count", "Win%", "Avg P&L", "Total P&L", "% of Gross"],
        rows,
    ))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 5: Experiment Ranking
# ---------------------------------------------------------------------------

def compute_experiment_section(exp_path: str) -> str:
    lines = ["## Section 5: Experiment Ranking\n"]

    if not os.path.exists(exp_path):
        lines.append(f"_Skipped — file not found: `{exp_path}`_\n")
        return "\n".join(lines)

    with open(exp_path) as f:
        exps = json.load(f)

    # Find baseline for delta columns
    baseline = next((e for e in exps if e.get("exp") == "BASELINE"), None)
    base_sharpe = baseline["arena_sharpe"] if baseline else 0.0
    base_return = baseline["arena_return"] if baseline else 0.0

    # Sort by combined_score descending, then arena_sharpe
    ranked = sorted(exps, key=lambda e: (e.get("combined_score", 0), e.get("arena_sharpe", 0)), reverse=True)

    rows = []
    for i, e in enumerate(ranked, 1):
        delta_sharpe = e.get("arena_sharpe", 0) - base_sharpe
        delta_return = e.get("arena_return", 0) - base_return
        overfit_flag = "OVERFIT" if e.get("overfit") else ""
        rows.append([
            str(i),
            e.get("exp", "?"),
            fmt(e.get("combined_score", 0), 3),
            fmt(e.get("arena_sharpe", 0), 3),
            f"{delta_sharpe:+.3f}",
            fmt(e.get("arena_return", 0), 2),
            f"{delta_return:+.2f}",
            fmt(e.get("arena_dd", 0), 2),
            str(e.get("arena_trades", 0)),
            overfit_flag,
        ])

    lines.append(md_table(
        ["#", "Experiment", "Score", "Sharpe", "ΔSharpe", "Return%", "ΔReturn", "MaxDD%", "Trades", "Flags"],
        rows,
    ))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 6: Annual Compounding
# ---------------------------------------------------------------------------

def compute_annual_section(analysis_path: str) -> str:
    lines = ["## Section 6: Annual Compounding\n"]

    if not os.path.exists(analysis_path):
        lines.append(f"_Skipped — file not found: `{analysis_path}`_\n")
        return "\n".join(lines)

    with open(analysis_path) as f:
        data = json.load(f)

    by_year = data.get("by_year", {})
    equity_curve = data.get("equity_curve", [])
    if not by_year:
        lines.append("_No by_year data._\n")
        return "\n".join(lines)

    # Build running equity from equity_curve start or assume 1,000,000 start
    start_equity = equity_curve[0][1] if equity_curve else 1_000_000.0

    running_equity = start_equity
    rows = []
    years = sorted(by_year.keys())
    for yr in years:
        yd = by_year[yr]
        year_pnl = yd.get("pnl", 0)
        year_return_pct = 100.0 * year_pnl / running_equity if running_equity else 0.0
        running_equity += year_pnl
        count = yd.get("count", 0)
        wins = yd.get("wins", 0)
        win_rate = 100.0 * wins / count if count else 0.0
        # Approximate Sharpe: annual_return / (annual_dd proxy = |return| * 0.5) — rough
        approx_sharpe = year_return_pct / (abs(year_return_pct) * 0.5 + 1e-9) if year_return_pct != 0 else 0.0
        approx_sharpe = min(max(approx_sharpe, -9.99), 9.99)  # clamp extreme values
        rows.append([
            yr,
            f"${year_pnl:,.0f}",
            f"{year_return_pct:.2f}%",
            f"${running_equity:,.0f}",
            str(count),
            f"{win_rate:.1f}%",
            fmt(approx_sharpe, 2),
        ])

    lines.append(md_table(
        ["Year", "PnL", "Return%", "Running Equity", "Trades", "Win%", "≈Sharpe"],
        rows,
    ))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_primitives(csv_path: str, cf: float, out_path: str):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    sections = []

    # Metadata header
    sections.append(f"""# SRFM Primitives

> **Generated:** {now}
> **CF value:** {cf}
> **CSV source:** `{csv_path}`
> **Data sources:** wells_ES.csv · regimes_ES.csv · trade_analysis_data.json · v2_experiments.json

---
""")

    # Section 1
    sections.append(compute_beta_section(csv_path, cf))
    sections.append("---\n")

    # Section 2
    sections.append(compute_regime_section(repo("results/regimes_ES.csv")))
    sections.append("---\n")

    # Section 3
    sections.append(compute_well_section(repo("results/wells_ES.csv")))
    sections.append("---\n")

    # Section 4
    sections.append(compute_convergence_section(repo("research/trade_analysis_data.json")))
    sections.append("---\n")

    # Section 5
    sections.append(compute_experiment_section(repo("results/v2_experiments.json")))
    sections.append("---\n")

    # Section 6
    sections.append(compute_annual_section(repo("research/trade_analysis_data.json")))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))

    print(f"[primitive_builder] Written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRFM Primitive Builder")
    parser.add_argument("--csv", default=repo("data/NDX_hourly_poly.csv"), help="OHLCV CSV path")
    parser.add_argument("--cf", type=float, default=0.005, help="CF scaling factor")
    parser.add_argument("--out", default=repo("results/primitives.md"), help="Output markdown path")
    args = parser.parse_args()

    build_primitives(args.csv, args.cf, args.out)
