"""
conv_timeline.py — Convergence event timeline.

Shows when convergence events occurred (2+ instruments BH active simultaneously).
Reads research/trade_analysis_data.json.

Usage:
    python tools/conv_timeline.py
    python tools/conv_timeline.py --data research/trade_analysis_data.json
    python tools/conv_timeline.py --pnl-threshold 200000
"""

import argparse
import json
import os
import sys
from collections import defaultdict
import re


DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), "..", "research", "trade_analysis_data.json"
)

MONTH_ABBR = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def _parse_year_month(date_str: str):
    m = re.match(r"(\d{4})-(\d{2})", date_str)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def main():
    parser = argparse.ArgumentParser(description="LARSA Convergence Event Timeline")
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Path to trade_analysis_data.json",
    )
    parser.add_argument(
        "--pnl-threshold",
        type=float,
        default=200_000,
        help="P&L threshold for 'big win' marker (default: 200000)",
    )
    parser.add_argument(
        "--min-instruments",
        type=int,
        default=2,
        help="Minimum instruments for convergence (default: 2)",
    )
    parser.add_argument("--out", default="results/conv_timeline.md")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        print(f"WARNING: Data file not found: {data_path}")
        print("  Expected: research/trade_analysis_data.json")
        print("  Run the LARSA pipeline to generate it.")
        sys.exit(1)

    with open(data_path) as f:
        data = json.load(f)

    wells = data.get("wells", [])
    if not wells:
        print("No wells found in data.")
        sys.exit(1)

    print(f"  Loaded {len(wells)} wells from {data_path}")

    # ── Categorize wells ──────────────────────────────────────────────────────
    # month_events[(year, month)] = list of wells
    month_events: dict = defaultdict(list)
    # month_conv[(year, month)] = list of convergence wells
    month_conv: dict = defaultdict(list)
    # big wins = net_pnl > threshold
    month_big: dict = defaultdict(list)

    all_years = set()

    for w in wells:
        start = w.get("start", "")
        year, month = _parse_year_month(start)
        if not year:
            continue
        all_years.add(year)
        month_events[(year, month)].append(w)

        instruments = w.get("instruments", [])
        net_pnl = w.get("net_pnl", w.get("total_pnl", 0.0))
        is_conv = len(instruments) >= args.min_instruments

        if is_conv:
            month_conv[(year, month)].append(w)
        if net_pnl >= args.pnl_threshold:
            month_big[(year, month)].append(w)

    if not all_years:
        print("No dated wells found.")
        sys.exit(1)

    years = sorted(all_years)

    # ── Build timeline ────────────────────────────────────────────────────────
    lines = []
    lines.append(f"CONVERGENCE EVENT TIMELINE ({min(years)}-{max(years)})")
    lines.append("=" * 60)

    month_label = "     |  " + "  ".join(MONTH_ABBR)
    lines.append(month_label)
    lines.append("     +" + "─" * 40)

    for year in years:
        row = f"{year} | "
        for month in range(1, 13):
            conv = month_conv.get((year, month), [])
            big = month_big.get((year, month), [])
            events = month_events.get((year, month), [])

            has_conv = len(conv) > 0
            has_big = len(big) > 0

            if has_conv and has_big:
                cell = "C■"
            elif has_conv:
                cell = "C "
            elif has_big:
                cell = "■ "
            elif events:
                cell = ". "
            else:
                cell = "  "
            row += " " + cell
        lines.append(row)

    lines.append("     +" + "─" * 40)
    lines.append(month_label)
    lines.append("")
    lines.append("(■ = P&L > ${:,.0f}  C = convergence  . = quiet)".format(args.pnl_threshold))

    # ── Convergence stats ─────────────────────────────────────────────────────
    all_conv = [w for wlist in month_conv.values() for w in wlist]
    total_conv = len(all_conv)

    lines.append("")
    lines.append("CONVERGENCE STATS:")
    lines.append(f"  Total convergence events:  {total_conv}")

    if all_conv:
        pnls = [w.get("net_pnl", w.get("total_pnl", 0.0)) for w in all_conv]
        wins = [p for p in pnls if p > 0]
        avg_pnl = sum(pnls) / len(pnls)
        win_rate = 100.0 * len(wins) / len(pnls) if pnls else 0.0
        durations = [w.get("duration_h", 0.0) for w in all_conv]
        avg_dur = sum(durations) / len(durations) if durations else 0.0
        lines.append(f"  Avg P&L per event:         ${avg_pnl:>10,.0f}")
        lines.append(f"  Win rate:                  {win_rate:.1f}%")
        lines.append(f"  Avg duration:              {avg_dur:.1f} hours")

    # ── Best convergence events ───────────────────────────────────────────────
    if all_conv:
        top = sorted(all_conv, key=lambda w: w.get("net_pnl", w.get("total_pnl", 0.0)), reverse=True)[:10]
        lines.append("")
        lines.append("BEST CONVERGENCE EVENTS:")
        lines.append(f"  {'Date':<12}  {'Instruments':<10}  {'Dur':>5}h  {'Net P&L':>12}")
        lines.append("  " + "-" * 50)
        for w in top:
            start = w.get("start", "")[:10]
            instrs = "+".join(w.get("instruments", []))
            dur = w.get("duration_h", 0.0)
            pnl = w.get("net_pnl", w.get("total_pnl", 0.0))
            lines.append(f"  {start:<12}  {instrs:<10}  {dur:>5.0f}h  ${pnl:>12,.0f}")

    # ── Non-convergence big wins (solo) ───────────────────────────────────────
    all_big_solo = [
        w for w in wells
        if len(w.get("instruments", [])) < args.min_instruments
        and w.get("net_pnl", w.get("total_pnl", 0.0)) >= args.pnl_threshold
    ]
    if all_big_solo:
        top_solo = sorted(
            all_big_solo,
            key=lambda w: w.get("net_pnl", w.get("total_pnl", 0.0)),
            reverse=True
        )[:5]
        lines.append("")
        lines.append("NOTABLE SOLO BIG WINS (single instrument):")
        lines.append(f"  {'Date':<12}  {'Instruments':<10}  {'Dur':>5}h  {'Net P&L':>12}")
        lines.append("  " + "-" * 50)
        for w in top_solo:
            start = w.get("start", "")[:10]
            instrs = "+".join(w.get("instruments", []))
            dur = w.get("duration_h", 0.0)
            pnl = w.get("net_pnl", w.get("total_pnl", 0.0))
            lines.append(f"  {start:<12}  {instrs:<10}  {dur:>5.0f}h  ${pnl:>12,.0f}")

    # ── Year summary ──────────────────────────────────────────────────────────
    lines.append("")
    lines.append("CONVERGENCE BY YEAR:")
    for year in years:
        year_conv = [w for wlist in [month_conv.get((year, m), []) for m in range(1, 13)] for w in wlist]
        cnt = len(year_conv)
        if cnt == 0:
            lines.append(f"  {year}: no convergence events")
        else:
            pnls = [w.get("net_pnl", w.get("total_pnl", 0.0)) for w in year_conv]
            total_pnl = sum(pnls)
            lines.append(f"  {year}: {cnt} events, total P&L ${total_pnl:,.0f}")

    output = "\n".join(lines)
    print()
    print(output)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("```\n" + output + "\n```\n")
    print(f"\n  Saved to {args.out}")


if __name__ == "__main__":
    main()
