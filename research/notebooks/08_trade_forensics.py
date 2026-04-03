"""
Notebook 08 — Trade Forensics: LARSA 274% QC Backtest Analysis
Run with: python research/notebooks/08_trade_forensics.py

Produces:
  - Console analysis
  - research/plots/08_equity_curve.txt (ASCII)
  - research/plots/08_well_calendar.txt
"""

import json
import os
import sys
from datetime import datetime
from collections import defaultdict

DATA_PATH = "research/trade_analysis_data.json"
os.makedirs("research/plots", exist_ok=True)

# ─── Load ────────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    data = json.load(f)

summary   = data["summary"]
by_year   = data["by_year"]
by_inst   = data["by_instrument"]
by_dir    = data["by_direction"]
wells     = data["wells"]
curve     = [(datetime.fromisoformat(t), v) for t, v in data["equity_curve"]]
flat      = data["flat_periods"]

print("=" * 65)
print("  LARSA 274% BACKTEST — TRADE FORENSICS")
print("=" * 65)

# ─── 1. Headline numbers ─────────────────────────────────────────────────────
print(f"\n{'HEADLINE NUMBERS':─<50}")
print(f"  Trades:          {summary['n_trades']}")
print(f"  Wells:           {summary['n_wells']}")
print(f"  Gross P&L:       ${summary['total_pnl']:>12,.0f}")
print(f"  Net P&L:         ${summary['net_pnl']:>12,.0f}")
print(f"  Gross Return:    {summary['total_return_pct']:>6.1f}%")
print(f"  Win Rate:        {summary['win_rate_pct']:>6.1f}%")
print(f"  Max Drawdown:    {summary['max_dd_pct']:>6.1f}%")
print(f"  Sharpe:          {summary['sharpe']:>6.3f}")
print(f"  P&L Ratio (W/L): {summary['pnl_ratio']:>6.2f}x")

# ─── 2. Annual breakdown ─────────────────────────────────────────────────────
print(f"\n{'ANNUAL P&L':─<50}")
total_pnl = summary["total_pnl"]
years = sorted(by_year.keys(), key=int)
running = 0
for yr in years:
    d = by_year[yr]
    running += d["pnl"]
    wr = d["wins"] / d["count"] * 100
    pct_total = d["pnl"] / total_pnl * 100
    bar = "█" * int(abs(d["pnl"]) / 50000)
    sign = "+" if d["pnl"] > 0 else ""
    print(f"  {yr}  {d['count']:3d}T  {wr:3.0f}%WR  {sign}${d['pnl']:>10,.0f}  ({pct_total:+5.1f}% of total)  {bar}")

# ─── 3. Key insight: what really drove 274%? ─────────────────────────────────
print(f"\n{'RETURN CONCENTRATION':─<50}")
top3_years = ["2020", "2023", "2024"]
top3_pnl = sum(by_year[y]["pnl"] for y in top3_years)
print(f"  2020+2023+2024 = ${top3_pnl:,.0f}  ({top3_pnl/total_pnl*100:.1f}% of total P&L)")
print(f"  Other 4 years  = ${total_pnl-top3_pnl:,.0f}  ({(total_pnl-top3_pnl)/total_pnl*100:.1f}% of total P&L)")
print(f"\n  2024 ALONE: ${by_year['2024']['pnl']:,.0f}  = {by_year['2024']['pnl']/1e6*100:.1f}% of $1M capital")
print(f"  Note: 2024 trades = {by_year['2024']['count']} with {by_year['2024']['wins']/by_year['2024']['count']*100:.0f}% win rate — unusually high")

# ─── 4. Instrument breakdown ─────────────────────────────────────────────────
print(f"\n{'INSTRUMENT ATTRIBUTION':─<50}")
for inst in sorted(by_inst.keys()):
    d = by_inst[inst]
    wr = d["wins"] / d["count"] * 100
    pct = d["pnl"] / total_pnl * 100
    print(f"  {inst}: {d['count']:3d} trades  {wr:3.0f}% WR  ${d['pnl']:>12,.0f}  ({pct:.1f}% of gross)")

# ─── 5. Direction bias ───────────────────────────────────────────────────────
print(f"\n{'DIRECTION BIAS':─<50}")
for d_name in sorted(by_dir.keys()):
    d = by_dir[d_name]
    wr = d["wins"] / d["count"] * 100
    pct = d["pnl"] / total_pnl * 100
    print(f"  {d_name:4s}: {d['count']:3d} trades  {wr:3.0f}% WR  ${d['pnl']:>12,.0f}  ({pct:.1f}% of gross)")

print(f"\n  >>> Strategy is LONG-DOMINATED: {by_dir['Buy']['pnl']/total_pnl*100:.0f}% of P&L from long positions")
print(f"      Short P&L exists but small: {by_dir.get('Sell',{}).get('pnl',0)/total_pnl*100:.0f}% from shorts")

# ─── 6. Well analysis ────────────────────────────────────────────────────────
print(f"\n{'WELL STATISTICS':─<50}")
win_wells  = [w for w in wells if w["is_win"]]
loss_wells = [w for w in wells if not w["is_win"]]
print(f"  Total wells:      {len(wells)}")
print(f"  Winning wells:    {len(win_wells)}  (avg: ${summary['well_avg_win_pnl']:,.0f})")
print(f"  Losing wells:     {len(loss_wells)}  (avg: ${summary['well_avg_loss_pnl']:,.0f})")
print(f"  Well win rate:    {summary['well_win_rate']:.1f}%")

# Well duration analysis
durations = [w["duration_h"] for w in wells]
short_wells = [w for w in wells if w["duration_h"] <= 2]
long_wells  = [w for w in wells if w["duration_h"] > 24]
print(f"\n  Wells <=2h (intraday flash): {len(short_wells)}")
print(f"  Wells >24h (multi-day):      {len(long_wells)}")

# Multi-instrument convergence
multi_inst = [w for w in wells if len(w["instruments"]) > 1]
single_inst = [w for w in wells if len(w["instruments"]) == 1]
multi_pnl = sum(w["total_pnl"] for w in multi_inst)
single_pnl = sum(w["total_pnl"] for w in single_inst)
print(f"\n  Multi-instrument wells:  {len(multi_inst)} wells  ${multi_pnl:,.0f} P&L")
print(f"  Single-instrument wells: {len(single_inst)} wells  ${single_pnl:,.0f} P&L")
if multi_inst:
    multi_wr = sum(1 for w in multi_inst if w["is_win"]) / len(multi_inst) * 100
    print(f"  Multi-inst win rate:     {multi_wr:.1f}%  (vs {summary['well_win_rate']:.1f}% overall)")
    print(f"  >>> CONVERGENCE EDGE: multi-instrument wells represent {len(multi_inst)/len(wells)*100:.1f}% of wells")
    print(f"      but {multi_pnl/total_pnl*100:.1f}% of gross P&L")

# ─── 7. Top 10 winners/losers ────────────────────────────────────────────────
sorted_wells = sorted(wells, key=lambda w: w["total_pnl"], reverse=True)

print(f"\n{'TOP 10 WINNING WELLS':─<50}")
for w in sorted_wells[:10]:
    dur = f"{w['duration_h']:.0f}h" if w["duration_h"] < 168 else f"{w['duration_h']/24:.1f}d"
    print(f"  {w['start'][:10]}  {dur:>5}  {'+'.join(w['instruments']):<8}  ${w['total_pnl']:>10,.0f}")

print(f"\n{'TOP 10 LOSING WELLS (worst first)':─<50}")
for w in sorted_wells[-10:][::-1]:
    dur = f"{w['duration_h']:.0f}h" if w["duration_h"] < 168 else f"{w['duration_h']/24:.1f}d"
    print(f"  {w['start'][:10]}  {dur:>5}  {'+'.join(w['instruments']):<8}  ${w['total_pnl']:>10,.0f}")

# ─── 8. Flat period analysis ─────────────────────────────────────────────────
print(f"\n{'FLAT PERIOD ANALYSIS':─<50}")
flat_by_year = defaultdict(list)
for g in flat:
    yr = g["start"][:4]
    flat_by_year[yr].append(g["days"])

for yr in sorted(flat_by_year.keys()):
    gaps = flat_by_year[yr]
    print(f"  {yr}: {len(gaps)} flat periods  avg {sum(gaps)/len(gaps):.0f}d  max {max(gaps):.0f}d")

long_gaps = sorted(flat, key=lambda g: -g["days"])[:5]
print(f"\n  Longest flat periods:")
for g in long_gaps:
    print(f"    {g['start'][:10]} -> {g['end'][:10]}  ({g['days']:.0f} days)")

# ─── 9. ASCII equity curve ───────────────────────────────────────────────────
print(f"\n{'EQUITY CURVE (ASCII)':─<50}")
vals = [v for _, v in curve]
min_v = min(vals)
max_v = max(vals)
width = 60
height = 20

rows_out = []
for row_i in range(height):
    threshold = max_v - (row_i / (height - 1)) * (max_v - min_v)
    label = f"${threshold/1e6:.2f}M"
    line = f"{label:>8} |"
    for col_i in range(width):
        idx = int(col_i / width * (len(vals) - 1))
        line += "█" if vals[idx] >= threshold else " "
    rows_out.append(line)
rows_out.append(f"{'':8} +" + "─" * width)
years_label = "        " + "  ".join(str(y)[2:] for y in range(2018, 2026))
rows_out.append(years_label)
curve_txt = "\n".join(rows_out)
print(curve_txt)

# Save ASCII curve
with open("research/plots/08_equity_curve.txt", "w", encoding="utf-8") as f:
    f.write("LARSA Equity Curve — Calm Orange Mule (274% QC Backtest)\n\n")
    f.write(curve_txt)
print(f"\n  Saved -> research/plots/08_equity_curve.txt")

# ─── 10. Key insights summary ────────────────────────────────────────────────
print(f"\n{'KEY INSIGHTS':─<65}")
print("""
  1. RETURN IS BACK-LOADED, NOT 2020-DRIVEN
     - 2020 contributed $629k (21.7% of total) — significant but not dominant
     - 2023 contributed $690k (23.8%) and 2024 contributed $1.116M (38.5%)
     - Prior hypothesis that "2020 was 60-80% of returns" is WRONG
     - The strategy COMPOUNDS: capital base grows, so later trades use more leverage

  2. POSITION SIZING IS THE SECRET
     - With $1M capital in 2018, $93k = 9.3% return
     - With ~$2M+ capital in 2024, $1.116M = same % moves but higher absolute $
     - The 65% 2024 win rate suggests either favorable regime or larger sizing on winners
     - Worst single trades are all 2024 ($258k loss in 1 trade!) — position size risk

  3. CONVERGENCE EVENTS ARE THE EDGE
     - Multi-instrument wells (NQ+YM, ES+NQ, ES+YM) show higher P&L concentration
     - Best well ever: 2023-12-12 NQ+YM $453k in 30 hours (Dec Fed rally)
     - Second best: 2020-11-06 YM $436k (US Election/Vaccine bull run)
     - Multi-instrument convergence = macro regime certainty = size up

  4. ES IS THE WORKHORSE, NQ+YM ARE THE ALPHA GENERATORS
     - ES: 55.3% of P&L but 183 trades (most defensive, most liquid)
     - NQ: 21.3% of P&L, 106 trades (highest per-trade P&L due to momentum)
     - YM: 23.4% of P&L, 88 trades (Dow = macro direction, less noise)

  5. THE BH MECHANISM WORKS EXACTLY AS DESIGNED
     - 263 wells from 377 trades = 1.43 trades/well average (tight clustering)
     - Kill conditions keep flat periods: 85 gaps >=7 days (29.7% of 286 possible weeks flat)
     - 2019: most flat (49% win rate, strategy correctly stayed out of sideways chop)

  6. ARENA CALIBRATION TARGET
     - QC real data implied median ES hourly |return| ~0.00067 (RTH only)
     - CF=0.001 = 1.49x median -> ~72% TIMELIKE rate -> BH formation enabled
     - NDX real data (2023-2025): median |return|=0.00156, need CF ~0.0023-0.0034
     - Arena positive Sharpe at CF=0.005-0.006 is ABOVE optimal range -> fewer wells
""")
