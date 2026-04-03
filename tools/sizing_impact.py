#!/usr/bin/env python
"""sizing_impact.py — Simulates halving solo BH cap (0.40 → 0.20).

Because solo P&L is proportional to position size, halving the cap
approximates ×0.5 on solo P&L.  Convergence P&L is unchanged.

Usage:
    python tools/sizing_impact.py [path/to/trade_analysis_data.json]
"""
import json
import sys
import os
from collections import defaultdict

SOLO_SCALE = 0.5  # proposed solo cap 0.20 vs current 0.40

def load_data(path):
    with open(path) as f:
        return json.load(f)

def fmt(v):
    return f"${v:>14,.0f}"

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def run(path):
    data = load_data(path)
    wells = data["wells"]

    solo   = [w for w in wells if len(w["instruments"]) == 1]
    conv   = [w for w in wells if len(w["instruments"]) >= 2]

    solo_pnl_orig = sum(w["total_pnl"] for w in solo)
    conv_pnl_orig = sum(w["total_pnl"] for w in conv)
    total_orig    = solo_pnl_orig + conv_pnl_orig

    solo_pnl_adj  = solo_pnl_orig * SOLO_SCALE
    conv_pnl_adj  = conv_pnl_orig          # unchanged
    total_adj     = solo_pnl_adj + conv_pnl_adj

    section("SIMULATION: Solo BH cap 0.40 → 0.20 (×0.5 on solo P&L)")
    print(f"\n  Assumption: solo P&L scales linearly with position size.")
    print(f"  Convergence P&L unchanged (already uses 0.65 cap).\n")
    print(f"  {'Category':<28}  {'Original':>14}  {'Adjusted':>14}  {'Delta':>14}")
    print(f"  {'-'*70}")
    print(f"  {'Solo P&L':<28}  {fmt(solo_pnl_orig)}  {fmt(solo_pnl_adj)}  {fmt(solo_pnl_adj - solo_pnl_orig)}")
    print(f"  {'Convergence P&L':<28}  {fmt(conv_pnl_orig)}  {fmt(conv_pnl_adj)}  {fmt(0)}")
    print(f"  {'-'*70}")
    print(f"  {'TOTAL':<28}  {fmt(total_orig)}  {fmt(total_adj)}  {fmt(total_adj - total_orig)}")
    print(f"\n  Change: {(total_adj - total_orig)/total_orig*100:+.1f}% on total gross P&L")

    # ── Per-year impact ────────────────────────────────────────────────
    section("PER-YEAR IMPACT")
    years = sorted(set(w["year"] for w in wells))
    print(f"\n  {'Year':<6}  {'Solo orig':>14}  {'Solo adj':>14}  {'Conv orig':>14}  {'Year total orig':>16}  {'Year total adj':>15}  {'Delta':>12}")
    print(f"  {'-'*96}")
    for yr in years:
        solo_y = [w for w in solo if w["year"] == yr]
        conv_y = [w for w in conv if w["year"] == yr]
        sp_o = sum(w["total_pnl"] for w in solo_y)
        cp_o = sum(w["total_pnl"] for w in conv_y)
        sp_a = sp_o * SOLO_SCALE
        t_o  = sp_o + cp_o
        t_a  = sp_a + cp_o
        delta = t_a - t_o
        flag = "  <<<" if delta < -50000 else ""
        print(f"  {yr:<6}  {fmt(sp_o)}  {fmt(sp_a)}  {fmt(cp_o)}  {fmt(t_o):>16}  {fmt(t_a):>15}  {fmt(delta):>12}{flag}")

    # ── Per-instrument impact ──────────────────────────────────────────
    section("PER-INSTRUMENT IMPACT (solo wells only)")
    instruments = sorted(set(w["instruments"][0] for w in solo))
    print(f"\n  {'Instrument':<14}  {'Solo orig':>14}  {'Solo adj':>14}  {'Delta':>14}")
    print(f"  {'-'*58}")
    for inst in instruments:
        iw = [w for w in solo if w["instruments"][0] == inst]
        sp_o = sum(w["total_pnl"] for w in iw)
        sp_a = sp_o * SOLO_SCALE
        print(f"  {inst:<14}  {fmt(sp_o)}  {fmt(sp_a)}  {fmt(sp_a - sp_o)}")

    section("SUMMARY")
    print(f"""
  Current solo cap  : 0.40 (40% of portfolio)
  Proposed solo cap : 0.20 (20% of portfolio)  — v6 uses 0.25/BULL, 0.15/other

  Original total gross P&L : {fmt(total_orig)}
  Adjusted total gross P&L : {fmt(total_adj)}
  Net P&L impact           : {fmt(total_adj - total_orig)}  ({(total_adj-total_orig)/total_orig*100:+.1f}%)

  Rationale: 216 solo wells at ~46.8% WR = near break-even.
  Halving their size costs ~{fmt(solo_pnl_orig * SOLO_SCALE - solo_pnl_orig)} in raw P&L
  but removes outsized risk in the break-even category.
  Convergence trades (74.5% WR, 81.4% of P&L) are untouched.
""")

if __name__ == "__main__":
    default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "research", "trade_analysis_data.json")
    path = sys.argv[1] if len(sys.argv) > 1 else default
    if not os.path.exists(path):
        print(f"ERROR: data file not found: {path}", file=sys.stderr)
        sys.exit(1)
    run(path)
