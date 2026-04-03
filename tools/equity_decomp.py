"""
equity_decomp.py — Equity curve decomposition for LARSA backtest.

Reads research/trade_analysis_data.json and breaks down the equity curve
by instrument, direction, well type, and year.

Usage:
    python tools/equity_decomp.py
    python tools/equity_decomp.py --by instrument
    python tools/equity_decomp.py --by type
    python tools/equity_decomp.py --by year
    python tools/equity_decomp.py --by direction
"""

import argparse
import json
import os
import sys
import collections

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "research", "trade_analysis_data.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

INSTRUMENTS = ["ES", "NQ", "YM"]
STARTING_CAPITAL = 1_000_000.0


# ── helpers ────────────────────────────────────────────────────────────────────

def fmt_dollar(v):
    """Format as compact dollar string e.g. +$2,313k."""
    sign = "+" if v >= 0 else "-"
    av = abs(v)
    if av >= 1_000_000:
        return f"{sign}${av/1_000_000:.2f}M"
    return f"{sign}${av/1000:.0f}k"


def fmt_pct(v, denom=STARTING_CAPITAL):
    """Format as percentage of starting capital."""
    return f"{v/denom*100:+.1f}%"


def win_rate(wells):
    if not wells:
        return 0.0
    return sum(1 for w in wells if w["is_win"]) / len(wells)


def cumulative_pnl_by_year(wells, years):
    """Returns dict year → cumulative pnl up to and including that year."""
    by_year = collections.defaultdict(float)
    for w in wells:
        by_year[w["year"]] += w["net_pnl"]
    cum = 0.0
    result = {}
    for y in sorted(years):
        cum += by_year.get(y, 0.0)
        result[y] = cum
    return result


def ascii_bar(val, max_val, width=30):
    """Simple horizontal bar chart."""
    if max_val == 0:
        return " " * width
    frac = abs(val) / abs(max_val)
    filled = int(frac * width)
    char = "#" if val >= 0 else "-"
    return char * filled + " " * (width - filled)


# ── decomposition sections ─────────────────────────────────────────────────────

def section_instrument(wells, years):
    """BY INSTRUMENT: cumulative P&L per year per instrument."""
    lines = []
    lines.append("BY INSTRUMENT (cumulative net P&L by year):")
    lines.append(f"  {'Year':>5} | {'ES':>12} {'NQ':>12} {'YM':>12} | {'Total':>12}")
    lines.append("  " + "-" * 60)

    for y in sorted(years):
        row_vals = {}
        for inst in INSTRUMENTS:
            grp = [w for w in wells if w["year"] == y and inst in w["instruments"]]
            row_vals[inst] = sum(w["net_pnl"] for w in grp)
        total = sum(row_vals.values())
        lines.append(
            f"  {y:>5} | "
            f"{fmt_dollar(row_vals['ES']):>12} "
            f"{fmt_dollar(row_vals['NQ']):>12} "
            f"{fmt_dollar(row_vals['YM']):>12} | "
            f"{fmt_dollar(total):>12}"
        )

    lines.append("")
    lines.append("  TOTALS (all years):")
    all_total = 0.0
    for inst in INSTRUMENTS:
        grp  = [w for w in wells if inst in w["instruments"]]
        pnl  = sum(w["net_pnl"] for w in grp)
        all_total += pnl
        wr   = win_rate(grp)
        lines.append(f"    {inst}: {len(grp):>3} wells  {fmt_dollar(pnl):>10}  "
                     f"({fmt_pct(pnl)})  WR={wr:.1%}")
    lines.append("")

    # Note: wells can have multiple instruments, so totals won't sum to grand total
    grand = sum(w["net_pnl"] for w in wells)
    lines.append(f"  Grand total: {fmt_dollar(grand)} ({fmt_pct(grand)})")
    lines.append("  (Note: multi-instrument wells counted in each relevant instrument)")
    lines.append("")
    return "\n".join(lines)


def section_direction(wells):
    """BY DIRECTION: long-only vs short-only equity."""
    long_wells  = [w for w in wells if "Buy"  in w["directions"]]
    short_wells = [w for w in wells if "Sell" in w["directions"]]
    both_wells  = [w for w in wells if "Buy" in w["directions"] and "Sell" in w["directions"]]

    long_pnl  = sum(w["net_pnl"] for w in long_wells)
    short_pnl = sum(w["net_pnl"] for w in short_wells)
    grand     = sum(w["net_pnl"] for w in wells)

    lines = []
    lines.append("BY DIRECTION:")
    lines.append(f"  LONG  ({len(long_wells):>3} wells):   {fmt_dollar(long_pnl):>10}  "
                 f"({fmt_pct(long_pnl)})  WR={win_rate(long_wells):.1%}")
    lines.append(f"  SHORT ({len(short_wells):>3} wells):   {fmt_dollar(short_pnl):>10}  "
                 f"({fmt_pct(short_pnl)})  WR={win_rate(short_wells):.1%}")
    if both_wells:
        both_pnl = sum(w["net_pnl"] for w in both_wells)
        lines.append(f"  BOTH  ({len(both_wells):>3} wells):   {fmt_dollar(both_pnl):>10}  "
                     f"(counted in both above)")
    lines.append(f"  Grand total: {fmt_dollar(grand)} ({fmt_pct(grand)})")
    lines.append("")

    max_pnl = max(abs(long_pnl), abs(short_pnl))
    lines.append("  Long  " + ascii_bar(long_pnl,  max_pnl))
    lines.append("  Short " + ascii_bar(short_pnl, max_pnl))
    lines.append("")
    return "\n".join(lines)


def section_type(wells):
    """BY WELL TYPE: convergence (multi-instrument) vs solo."""
    conv = [w for w in wells if len(w["instruments"]) > 1]
    solo = [w for w in wells if len(w["instruments"]) == 1]

    conv_pnl  = sum(w["net_pnl"] for w in conv)
    solo_pnl  = sum(w["net_pnl"] for w in solo)
    grand     = sum(w["net_pnl"] for w in wells)

    conv_avg = conv_pnl / len(conv) if conv else 0
    solo_avg = solo_pnl / len(solo) if solo else 0
    ratio    = (conv_avg / solo_avg) if solo_avg != 0 else float("inf")

    conv_share = conv_pnl / grand * 100 if grand else 0
    conv_well_share = len(conv) / len(wells) * 100

    lines = []
    lines.append("BY WELL TYPE:")
    lines.append(f"  SOLO ONLY        ({len(solo):>3} wells):  "
                 f"{fmt_dollar(solo_pnl):>10}  ({fmt_pct(solo_pnl)})  WR={win_rate(solo):.1%}")
    lines.append(f"  CONVERGENCE ONLY ({len(conv):>3} wells):  "
                 f"{fmt_dollar(conv_pnl):>10}  ({fmt_pct(conv_pnl)})  WR={win_rate(conv):.1%}")
    lines.append("")
    lines.append(f"  Solo avg:        {fmt_dollar(solo_avg)}/well")
    lines.append(f"  Convergence avg: {fmt_dollar(conv_avg)}/well  ({ratio:.1f}x higher than solo)")
    lines.append("")
    lines.append(f"  Convergence events = {conv_well_share:.1f}% of wells but {conv_share:.1f}% of returns.")
    lines.append("")

    max_pnl = max(abs(conv_pnl), abs(solo_pnl))
    lines.append(f"  Solo         {ascii_bar(solo_pnl, max_pnl)}")
    lines.append(f"  Convergence  {ascii_bar(conv_pnl, max_pnl)}")
    lines.append("")
    return "\n".join(lines)


def section_year(wells, years):
    """YEAR-BY-YEAR attribution table."""
    conv = [w for w in wells if len(w["instruments"]) > 1]
    solo = [w for w in wells if len(w["instruments"]) == 1]

    by_year_all  = collections.defaultdict(float)
    by_year_conv = collections.defaultdict(float)
    by_year_solo = collections.defaultdict(float)

    for w in wells:
        by_year_all[w["year"]]  += w["net_pnl"]
    for w in conv:
        by_year_conv[w["year"]] += w["net_pnl"]
    for w in solo:
        by_year_solo[w["year"]] += w["net_pnl"]

    lines = []
    lines.append("YEAR-BY-YEAR ATTRIBUTION:")
    lines.append(f"  {'Year':>5} | {'All Wells':>12} {'Solo':>12} {'Conv':>12} | {'Cum Total':>12}")
    lines.append("  " + "-" * 60)

    cum = 0.0
    for y in sorted(years):
        a = by_year_all.get(y, 0.0)
        s = by_year_solo.get(y, 0.0)
        c = by_year_conv.get(y, 0.0)
        cum += a
        lines.append(
            f"  {y:>5} | "
            f"{fmt_dollar(a):>12} "
            f"{fmt_dollar(s):>12} "
            f"{fmt_dollar(c):>12} | "
            f"{fmt_dollar(cum):>12}"
        )

    lines.append("")
    return "\n".join(lines)


def section_key_insight(wells):
    conv      = [w for w in wells if len(w["instruments"]) > 1]
    solo      = [w for w in wells if len(w["instruments"]) == 1]
    conv_pnl  = sum(w["net_pnl"] for w in conv)
    solo_pnl  = sum(w["net_pnl"] for w in solo)
    grand     = sum(w["net_pnl"] for w in wells)
    conv_avg  = conv_pnl / len(conv) if conv else 0
    solo_avg  = solo_pnl / len(solo) if solo else 0
    ratio     = conv_avg / solo_avg if solo_avg != 0 else float("inf")

    lines = []
    lines.append("KEY INSIGHT:")
    lines.append(f"  If you traded ONLY convergence events ({len(conv)} wells): "
                 f"{fmt_pct(conv_pnl)} return")
    lines.append(f"  If you traded ONLY solo wells ({len(solo)} wells):         "
                 f"{fmt_pct(solo_pnl)} return")
    lines.append(f"  Combined actual:                                            "
                 f"{fmt_pct(grand)} return")
    lines.append("")
    lines.append(f"  Convergence events = {len(conv)/len(wells)*100:.1f}% of wells "
                 f"but {conv_pnl/grand*100:.1f}% of returns.")
    lines.append(f"  Average convergence well = {ratio:.1f}x average solo well.")
    lines.append("")
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Equity curve decomposition for LARSA backtest."
    )
    parser.add_argument(
        "--by",
        choices=["instrument", "direction", "type", "year", "all"],
        default="all",
        help="Which decomposition to show (default: all)",
    )
    parser.add_argument(
        "--data", default=DATA_PATH,
        help="Path to trade_analysis_data.json",
    )
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        print("       Run the LARSA backtest first to generate research/trade_analysis_data.json")
        sys.exit(1)

    with open(data_path) as f:
        data = json.load(f)

    wells = data["wells"]
    years = sorted(set(w["year"] for w in wells))
    grand = sum(w["net_pnl"] for w in wells)

    header = [
        "EQUITY DECOMPOSITION — LARSA backtest",
        "=" * 50,
        f"Wells: {len(wells)}   Years: {years[0]}–{years[-1]}",
        f"Total net P&L: {fmt_dollar(grand)} ({fmt_pct(grand)})",
        f"Starting capital assumed: ${STARTING_CAPITAL:,.0f}",
        "",
    ]

    parts = []
    parts.append("\n".join(header))

    do_all = args.by == "all"

    if do_all or args.by == "instrument":
        parts.append(section_instrument(wells, years))
    if do_all or args.by == "direction":
        parts.append(section_direction(wells))
    if do_all or args.by == "type":
        parts.append(section_type(wells))
    if do_all or args.by == "year":
        parts.append(section_year(wells, years))
    if do_all:
        parts.append(section_key_insight(wells))

    full_output = "\n".join(parts)
    print(full_output)

    if do_all:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        md_path = os.path.join(RESULTS_DIR, "equity_decomp.md")
        with open(md_path, "w") as f:
            f.write("# Equity Decomposition — LARSA backtest\n\n```\n")
            f.write(full_output)
            f.write("\n```\n")
        print(f"\nSaved report: {md_path}")


if __name__ == "__main__":
    main()
