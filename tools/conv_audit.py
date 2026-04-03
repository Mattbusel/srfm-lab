#!/usr/bin/env python
"""conv_audit.py — Breakdown of LARSA wells by convergence vs solo category.

Usage:
    python tools/conv_audit.py [path/to/trade_analysis_data.json]

Defaults to research/trade_analysis_data.json relative to the project root.
"""
import json
import sys
import os
from collections import defaultdict

def load_data(path):
    with open(path) as f:
        return json.load(f)

def classify(wells):
    """Return (solo_wells, conv_wells) split."""
    solo = [w for w in wells if len(w["instruments"]) == 1]
    conv = [w for w in wells if len(w["instruments"]) >= 2]
    return solo, conv

def stats(wells):
    if not wells:
        return dict(count=0, wins=0, wr=0.0, total_pnl=0.0, pnl_per_trade=0.0)
    count = len(wells)
    wins = sum(1 for w in wells if w["is_win"])
    wr = wins / count if count else 0.0
    total_pnl = sum(w["total_pnl"] for w in wells)
    pnl_per_trade = total_pnl / count if count else 0.0
    return dict(count=count, wins=wins, wr=wr, total_pnl=total_pnl, pnl_per_trade=pnl_per_trade)

def fmt_pnl(v):
    return f"${v:>12,.0f}"

def fmt_pct(v):
    return f"{v*100:5.1f}%"

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def table_row(label, s, width=22):
    label = label[:width].ljust(width)
    print(f"  {label}  {s['count']:>5}  {fmt_pct(s['wr'])}  "
          f"{fmt_pnl(s['total_pnl'])}  {fmt_pnl(s['pnl_per_trade'])}/trade")

def run(path):
    data = load_data(path)
    wells = data["wells"]

    solo_all, conv_all = classify(wells)
    s_all = stats(solo_all)
    c_all = stats(conv_all)
    grand_pnl = s_all["total_pnl"] + c_all["total_pnl"]

    # ── OVERALL ──────────────────────────────────────────────────────────
    section("OVERALL: Solo vs Convergence Wells")
    hdr = f"  {'Category':<22}  {'Count':>5}  {'  WR':>6}  {'Total P&L':>14}  {'P&L/Trade':>14}"
    print(hdr)
    print("  " + "-"*66)
    table_row("Solo (1 instrument)", s_all)
    table_row("Convergence (2+ instr)", c_all)
    print("  " + "-"*66)
    all_s = stats(wells)
    table_row("ALL", all_s)
    if grand_pnl != 0:
        print(f"\n  Solo share of gross P&L:        {s_all['total_pnl']/grand_pnl*100:5.1f}%")
        print(f"  Convergence share of gross P&L: {c_all['total_pnl']/grand_pnl*100:5.1f}%")

    # ── PER YEAR ─────────────────────────────────────────────────────────
    section("PER-YEAR BREAKDOWN")
    years = sorted(set(w["year"] for w in wells))
    print(f"  {'Year':<6}  {'Cat':<12}  {'Count':>5}  {'  WR':>6}  {'Total P&L':>14}  {'P&L/Trade':>14}")
    print("  " + "-"*66)
    year_solo_drag = {}   # year -> solo P&L (negative = drag)
    for yr in years:
        yw = [w for w in wells if w["year"] == yr]
        solo_y, conv_y = classify(yw)
        s_y = stats(solo_y)
        c_y = stats(conv_y)
        year_solo_drag[yr] = s_y["total_pnl"]
        table_row(f"{yr}  solo", s_y, width=14)
        table_row(f"{yr}  conv", c_y, width=14)
        print()

    # ── PER INSTRUMENT ───────────────────────────────────────────────────
    section("PER-INSTRUMENT BREAKDOWN (solo wells only)")
    instruments = sorted(set(w["instruments"][0] for w in solo_all))
    print(f"  {'Instrument':<22}  {'Count':>5}  {'  WR':>6}  {'Total P&L':>14}  {'P&L/Trade':>14}")
    print("  " + "-"*66)
    for inst in instruments:
        iw = [w for w in solo_all if w["instruments"][0] == inst]
        table_row(inst, stats(iw))

    section("PER-INSTRUMENT BREAKDOWN (convergence wells)")
    inst_conv = defaultdict(list)
    for w in conv_all:
        for inst in w["instruments"]:
            inst_conv[inst].append(w)
    print(f"  {'Instrument (member)':<22}  {'Count':>5}  {'  WR':>6}  {'Total P&L':>14}  {'P&L/Trade':>14}")
    print("  " + "-"*66)
    for inst in sorted(inst_conv):
        table_row(inst, stats(inst_conv[inst]))

    # ── DIAGNOSIS ────────────────────────────────────────────────────────
    section("DIAGNOSIS: Years where solo drag was worst")
    worst = sorted(year_solo_drag.items(), key=lambda kv: kv[1])
    print(f"  {'Year':<6}  {'Solo P&L':>14}  {'Note'}")
    print("  " + "-"*50)
    for yr, pnl in worst:
        flag = "  <<<" if pnl < 0 else ""
        print(f"  {yr:<6}  {fmt_pnl(pnl)}{flag}")

    print(f"\n  Key ratios (solo 46.8% WR ≈ break-even; convergence 74.5% WR = edge):")
    print(f"  Solo    WR: {s_all['wr']*100:.1f}%  —  {s_all['count']} wells  —  {fmt_pnl(s_all['total_pnl'])} gross")
    print(f"  Conv    WR: {c_all['wr']*100:.1f}%  —  {c_all['count']} wells  —  {fmt_pnl(c_all['total_pnl'])} gross")
    print()

if __name__ == "__main__":
    default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "research", "trade_analysis_data.json")
    path = sys.argv[1] if len(sys.argv) > 1 else default
    if not os.path.exists(path):
        print(f"ERROR: data file not found: {path}", file=sys.stderr)
        sys.exit(1)
    run(path)
