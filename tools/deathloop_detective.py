"""
deathloop_detective.py — Diagnose what causes the v8 death loop.

Loads QC trades CSV + JSON, reconstructs the equity curve bar-by-bar,
finds the peak, then dissects EXACTLY what happened after it.

Usage:
    python tools/deathloop_detective.py
    python tools/deathloop_detective.py --csv "path/to/trades.csv" --json "path/to/results.json"

Outputs:
    Console: full forensic report
    results/deathloop_report.md
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
def load_trades(path: str):
    trades = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pnl  = float(row["P&L"].replace(",", "") or 0)
            fees = float(row["Fees"].replace(",", "") or 0)
            mae  = float(row["MAE"].replace(",", "") or 0)
            mfe  = float(row["MFE"].replace(",", "") or 0)
            qty  = int(row["Quantity"] or 1)
            ep   = float(row["Entry Price"].replace(",", "") or 0)
            xp   = float(row["Exit Price"].replace(",", "") or 0)
            sym  = row["Symbols"].strip()

            # Derive underlying (ES/NQ/YM from contract name)
            underlying = "ES"
            if sym.startswith("NQ"): underlying = "NQ"
            elif sym.startswith("YM"): underlying = "YM"

            try:
                entry_dt = datetime.fromisoformat(row["Entry Time"].replace("Z", "+00:00")).replace(tzinfo=None)
                exit_dt  = datetime.fromisoformat(row["Exit Time"].replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                continue

            dur_h = (exit_dt - entry_dt).total_seconds() / 3600
            is_win = pnl > 0

            trades.append({
                "entry": entry_dt,
                "exit":  exit_dt,
                "sym":   sym,
                "under": underlying,
                "dir":   row["Direction"].strip(),
                "ep":    ep,
                "xp":    xp,
                "qty":   qty,
                "pnl":   pnl,
                "fees":  fees,
                "net":   pnl - fees,
                "mae":   mae,
                "mfe":   mfe,
                "dur_h": dur_h,
                "is_win": is_win,
            })

    trades.sort(key=lambda t: t["entry"])
    return trades


def build_equity_curve(trades, start_equity=1_000_000.0):
    """Reconstruct cumulative equity from trade P&L."""
    equity = start_equity
    curve = []
    for t in trades:
        equity += t["net"]
        curve.append({
            "dt": t["exit"],
            "equity": equity,
            "trade": t,
        })
    return curve


def find_peak(curve):
    peak_idx = max(range(len(curve)), key=lambda i: curve[i]["equity"])
    return peak_idx, curve[peak_idx]["equity"], curve[peak_idx]["dt"]


def classify_phases(curve, peak_idx):
    """Split into: buildup → peak → decline."""
    buildup = curve[:peak_idx + 1]
    decline = curve[peak_idx + 1:]
    return buildup, decline


def trade_cadence(trades):
    """Trades per hour — detect when firing rate explodes."""
    by_hour = defaultdict(list)
    for t in trades:
        hour_key = t["entry"].replace(minute=0, second=0, microsecond=0)
        by_hour[hour_key].append(t)
    return by_hour


def rapid_fire_analysis(trades, threshold_per_hour=3):
    """Find hours where trade count >= threshold — these are death-loop candidates."""
    cadence = trade_cadence(trades)
    hot_hours = [(h, ts) for h, ts in cadence.items() if len(ts) >= threshold_per_hour]
    hot_hours.sort(key=lambda x: x[0])
    return hot_hours


def instrument_flip_analysis(trades):
    """Detect rapid direction flips per instrument — entry then immediate reversal."""
    flips = []
    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t["under"]].append(t)

    for sym, sym_trades in by_sym.items():
        sym_trades.sort(key=lambda t: t["entry"])
        for i in range(1, len(sym_trades)):
            prev = sym_trades[i - 1]
            curr = sym_trades[i]
            gap_h = (curr["entry"] - prev["exit"]).total_seconds() / 3600
            direction_flip = (prev["dir"] != curr["dir"])
            if gap_h < 0.5 and direction_flip:
                flips.append({
                    "sym": sym,
                    "prev": prev,
                    "curr": curr,
                    "gap_h": gap_h,
                })
    return flips


def fee_drag_by_period(trades, start_equity=1_000_000.0):
    """Show cumulative fee drag vs gross P&L over time."""
    periods = defaultdict(lambda: {"pnl": 0.0, "fees": 0.0, "count": 0, "wins": 0})
    for t in trades:
        key = t["entry"].strftime("%Y-%m")
        periods[key]["pnl"]   += t["pnl"]
        periods[key]["fees"]  += t["fees"]
        periods[key]["count"] += 1
        periods[key]["wins"]  += int(t["is_win"])
    return periods


def duration_pnl_buckets(trades):
    """P&L by trade duration — shows whether short trades are the cancer."""
    buckets = {
        "< 15min":  {"trades": [], "label": "< 15min  (noise entries)"},
        "15m-1h":   {"trades": [], "label": "15m–1h"},
        "1h-4h":    {"trades": [], "label": "1h–4h"},
        "4h-1d":    {"trades": [], "label": "4h–1d"},
        "> 1d":     {"trades": [], "label": "> 1d    (multi-session)"},
    }
    for t in trades:
        d = t["dur_h"]
        if d < 0.25:       buckets["< 15min"]["trades"].append(t)
        elif d < 1.0:      buckets["15m-1h"]["trades"].append(t)
        elif d < 4.0:      buckets["1h-4h"]["trades"].append(t)
        elif d < 24.0:     buckets["4h-1d"]["trades"].append(t)
        else:              buckets["> 1d"]["trades"].append(t)
    return buckets


def consecutive_loss_streaks(trades):
    """Find the longest losing streaks and when they occurred."""
    streaks = []
    cur = []
    for t in trades:
        if not t["is_win"]:
            cur.append(t)
        else:
            if len(cur) >= 5:
                streaks.append(cur)
            cur = []
    if len(cur) >= 5:
        streaks.append(cur)
    streaks.sort(key=lambda s: -len(s))
    return streaks


def print_bar(label, value, total, width=30, prefix=""):
    pct = abs(value) / (abs(total) + 1e-9)
    bar = "█" * int(pct * width)
    sign = "+" if value >= 0 else "-"
    print(f"  {label:<12} {prefix}{sign}${abs(value):>10,.0f}  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  default="C:/Users/Matthew/Downloads/Hyper Active Red Orange Seahorse_trades.csv")
    parser.add_argument("--json", default="C:/Users/Matthew/Downloads/Hyper Active Red Orange Seahorse.json")
    parser.add_argument("--start-equity", type=float, default=1_000_000.0)
    args = parser.parse_args()

    print("=" * 70)
    print("  LARSA v8 DEATH LOOP DETECTIVE")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────
    trades = load_trades(args.csv)
    print(f"\n  Loaded {len(trades)} trades  ({trades[0]['entry'].date()} → {trades[-1]['exit'].date()})")

    curve   = build_equity_curve(trades, args.start_equity)
    peak_i, peak_eq, peak_dt = find_peak(curve)
    final_eq = curve[-1]["equity"]
    buildup, decline = classify_phases(curve, peak_i)

    print(f"  Start equity:  ${args.start_equity:>12,.0f}")
    print(f"  Peak equity:   ${peak_eq:>12,.0f}  ({peak_eq/args.start_equity:.2f}x)  @ {peak_dt.date()}")
    print(f"  Final equity:  ${final_eq:>12,.0f}  ({final_eq/args.start_equity:.2f}x)")
    print(f"  Gave back:     ${peak_eq - final_eq:>12,.0f}  ({(peak_eq-final_eq)/peak_eq*100:.1f}% of peak)")

    buildup_trades = [p["trade"] for p in buildup]
    decline_trades = [p["trade"] for p in decline]

    print(f"\n  Buildup phase:  {len(buildup_trades)} trades  (to peak {peak_dt.date()})")
    print(f"  Decline phase:  {len(decline_trades)} trades  (after peak)")

    # ── 1. MONTHLY FEE DRAG ───────────────────────────────────────────────
    print(f"\n{'MONTHLY P&L vs FEES':─<70}")
    periods = fee_drag_by_period(trades)
    cum_pnl  = 0.0
    cum_fees = 0.0
    total_pnl  = sum(t["pnl"]  for t in trades)
    total_fees = sum(t["fees"] for t in trades)
    for mo in sorted(periods.keys()):
        d = periods[mo]
        cum_pnl  += d["pnl"]
        cum_fees += d["fees"]
        wr = d["wins"] / d["count"] * 100 if d["count"] else 0
        net = d["pnl"] - d["fees"]
        fee_pct = d["fees"] / (abs(d["pnl"]) + 1e-9) * 100
        flag = ""
        if d["count"] > 200:  flag = " <<< HYPERACTIVE"
        if net < -50000:      flag += " <<< DEATH LOOP"
        print(f"  {mo}  {d['count']:4d}T  {wr:3.0f}%WR  "
              f"gross={'+' if d['pnl']>=0 else ''}{d['pnl']:>10,.0f}  "
              f"fees={d['fees']:>8,.0f} ({fee_pct:3.0f}%)  "
              f"net={'+' if net>=0 else ''}{net:>10,.0f}{flag}")
    print(f"\n  TOTAL  {len(trades):4d}T  gross={total_pnl:>+12,.0f}  fees={total_fees:>8,.0f}  net={total_pnl-total_fees:>+12,.0f}")

    # ── 2. TRADE DURATION ANALYSIS ────────────────────────────────────────
    print(f"\n{'TRADE DURATION vs P&L':─<70}")
    buckets = duration_pnl_buckets(trades)
    for key, b in buckets.items():
        ts = b["trades"]
        if not ts: continue
        gross = sum(t["pnl"] for t in ts)
        fees  = sum(t["fees"] for t in ts)
        net   = gross - fees
        wins  = sum(1 for t in ts if t["is_win"])
        wr    = wins / len(ts) * 100
        avg_pnl = gross / len(ts)
        flag = " <<< BLEEDING" if net < 0 and len(ts) > 10 else ""
        print(f"  {b['label']:<22}  {len(ts):4d}T  {wr:3.0f}%WR  "
              f"avg={avg_pnl:>+7,.0f}  fees={fees:>8,.0f}  net={net:>+10,.0f}{flag}")

    # ── 3. RAPID FIRE HOURS ───────────────────────────────────────────────
    print(f"\n{'RAPID FIRE HOURS (>=4 trades in one hour)':─<70}")
    hot = rapid_fire_analysis(trades, threshold_per_hour=4)
    print(f"  Found {len(hot)} hyper-active hours")
    hot_pnl  = sum(t["pnl"]  for h, ts in hot for t in ts)
    hot_fees = sum(t["fees"] for h, ts in hot for t in ts)
    hot_net  = hot_pnl - hot_fees
    print(f"  Gross P&L from hot hours: {hot_pnl:>+12,.0f}")
    print(f"  Fees    from hot hours:   {hot_fees:>12,.0f}")
    print(f"  Net     from hot hours:   {hot_net:>+12,.0f}")
    print(f"\n  Worst 10 hot hours:")
    hot.sort(key=lambda x: sum(t["net"] for t in x[1]))
    for h, ts in hot[:10]:
        gpnl = sum(t["pnl"] for t in ts)
        gfees= sum(t["fees"] for t in ts)
        syms = set(t["under"] for t in ts)
        dirs = set(t["dir"] for t in ts)
        print(f"    {h.strftime('%Y-%m-%d %H:%M')}  {len(ts):2d}T  "
              f"syms={'+'.join(sorted(syms))}  dirs={'+'.join(sorted(dirs))}  "
              f"gross={gpnl:>+9,.0f}  fees={gfees:>7,.0f}  net={gpnl-gfees:>+9,.0f}")

    # ── 4. DIRECTION FLIP ANALYSIS ────────────────────────────────────────
    print(f"\n{'DIRECTION FLIPS (< 30min gap, opposite dir)':─<70}")
    flips = instrument_flip_analysis(trades)
    print(f"  Total rapid direction flips: {len(flips)}")
    flip_pnl = sum(f["curr"]["pnl"] + f["prev"]["pnl"] for f in flips)
    flip_fees= sum(f["curr"]["fees"] + f["prev"]["fees"] for f in flips)
    print(f"  Net P&L on flip pairs: {flip_pnl - flip_fees:>+12,.0f}")
    print(f"\n  Sample flip events (worst 10):")
    flips.sort(key=lambda f: f["curr"]["pnl"] + f["prev"]["pnl"])
    for f in flips[:10]:
        print(f"    {f['prev']['entry'].strftime('%Y-%m-%d %H:%M')}  {f['sym']:<6}  "
              f"{f['prev']['dir']}→{f['curr']['dir']}  "
              f"gap={f['gap_h']*60:.0f}min  "
              f"pnl={f['prev']['pnl']+f['curr']['pnl']:>+8,.0f}")

    # ── 5. CONSECUTIVE LOSS STREAKS ───────────────────────────────────────
    print(f"\n{'LOSS STREAKS (>= 5 consecutive losses)':─<70}")
    streaks = consecutive_loss_streaks(trades)
    print(f"  Found {len(streaks)} streaks of 5+ consecutive losses")
    for i, streak in enumerate(streaks[:5]):
        total_loss = sum(t["net"] for t in streak)
        syms = set(t["under"] for t in streak)
        start = streak[0]["entry"].strftime("%Y-%m-%d")
        end   = streak[-1]["exit"].strftime("%Y-%m-%d")
        print(f"  [{i+1}] {len(streak)} losses  {start} → {end}  "
              f"syms={'+'.join(sorted(syms))}  total={total_loss:>+10,.0f}")

    # ── 6. WHAT KILLED THE ACCOUNT ────────────────────────────────────────
    print(f"\n{'ROOT CAUSE SUMMARY':─<70}")

    # Find the inflection point: where did positive expectancy flip negative?
    window = 50
    inflections = []
    for i in range(window, len(trades)):
        window_trades = trades[i - window:i]
        net_window = sum(t["net"] for t in window_trades)
        if net_window < -20000:  # $20k rolling loss over 50 trades
            inflections.append((trades[i]["entry"], net_window, i))

    if inflections:
        first_bad = inflections[0]
        print(f"\n  Strategy turned NEGATIVE (50-trade rolling window < -$20k)")
        print(f"  First inflection: {first_bad[0].strftime('%Y-%m-%d %H:%M')}  "
              f"(trade #{first_bad[2]}  rolling loss=${first_bad[1]:,.0f})")
        print(f"  Peak was:         {peak_dt.strftime('%Y-%m-%d')}  ({peak_i} trades in)")
        gap = (first_bad[0] - peak_dt).days
        print(f"  Gap peak → inflection: {gap} days")
    else:
        print(f"\n  No clear inflection found (rolling window stayed above -$20k)")

    # Short-trade fee analysis
    short_trades = [t for t in trades if t["dur_h"] < 0.25]
    short_fees   = sum(t["fees"] for t in short_trades)
    short_net    = sum(t["net"] for t in short_trades)
    print(f"\n  Sub-15min trades:     {len(short_trades):4d}  fees=${short_fees:,.0f}  net={short_net:>+10,.0f}")

    # Total fee ratio
    print(f"  Fees as % of |gross|: {total_fees / (abs(total_pnl) + 1e-9) * 100:.1f}%")
    print(f"  Avg fee per trade:    ${total_fees / len(trades):.2f}")

    # Per-instrument verdict
    print(f"\n  Per-instrument breakdown:")
    by_inst = defaultdict(lambda: {"pnl": 0.0, "fees": 0.0, "count": 0, "wins": 0})
    for t in trades:
        by_inst[t["under"]]["pnl"]   += t["pnl"]
        by_inst[t["under"]]["fees"]  += t["fees"]
        by_inst[t["under"]]["count"] += 1
        by_inst[t["under"]]["wins"]  += int(t["is_win"])
    for sym in sorted(by_inst.keys()):
        d = by_inst[sym]
        wr  = d["wins"] / d["count"] * 100
        net = d["pnl"] - d["fees"]
        verdict = "OK" if net > 0 else "BLEEDING"
        print(f"    {sym}: {d['count']:4d}T  {wr:.0f}%WR  "
              f"gross={d['pnl']:>+10,.0f}  fees={d['fees']:>8,.0f}  net={net:>+10,.0f}  [{verdict}]")

    # ── 7. ASCII EQUITY CURVE ─────────────────────────────────────────────
    print(f"\n{'EQUITY CURVE (ASCII)':─<70}")
    vals = [p["equity"] for p in curve]
    dts  = [p["dt"] for p in curve]
    min_v, max_v = min(vals), max(vals)
    W, H = 65, 18

    rows_out = []
    for row_i in range(H):
        thresh = max_v - (row_i / (H - 1)) * (max_v - min_v)
        label  = f"${thresh/1e6:.2f}M"
        line   = f"{label:>8} |"
        for col_i in range(W):
            idx = int(col_i / W * (len(vals) - 1))
            line += "█" if vals[idx] >= thresh else " "
        rows_out.append(line)
    rows_out.append(f"{'':8} +" + "─" * W)

    # Month labels
    months_seen = {}
    for idx, dt in enumerate(dts):
        col = int(idx / len(dts) * W)
        mk = dt.strftime("%b%y")
        if mk not in months_seen:
            months_seen[mk] = col
    label_line = " " * 10
    prev_col = 0
    for mk, col in sorted(months_seen.items(), key=lambda x: x[1])[:12]:
        pad = col - prev_col
        label_line += " " * max(0, pad) + mk[:4]
        prev_col = col + 4
    rows_out.append(label_line)
    print("\n".join(rows_out))

    # ── 8. DIAGNOSIS & RECOMMENDED FIX ───────────────────────────────────
    print(f"\n{'DIAGNOSIS':─<70}")

    diagnosis = []

    if len(short_trades) > 50:
        diagnosis.append({
            "issue": "SUB-15MIN CHURN",
            "detail": f"{len(short_trades)} trades closed in < 15 min — ${short_fees:,.0f} in fees for ${short_net:+,.0f} net",
            "fix": "These are noise entries firing on 15m BH when signal barely formed. "
                   "Minimum hold: don't exit a position opened less than 1 hourly bar ago.",
        })

    if len(hot) > 20:
        diagnosis.append({
            "issue": "HYPERACTIVE HOURS",
            "detail": f"{len(hot)} hours with 4+ trades — rebalancing too frequently within same hour",
            "fix": "The hourly gate (current_hour == _last_exec_hour) should already fix this. "
                   "If still happening, the gate is being bypassed somehow.",
        })

    if len(flips) > 30:
        diagnosis.append({
            "issue": "DIRECTION FLIP CHURN",
            "detail": f"{len(flips)} rapid direction reversals — strategy buying then immediately selling same instrument",
            "fix": "BH direction is flipping between 15m bars. Add a minimum position hold: "
                   "track entry_bar per instrument, don't reverse direction within 4 bars (1 hour).",
        })

    fee_ratio = total_fees / (abs(total_pnl) + 1e-9)
    if fee_ratio > 0.5:
        diagnosis.append({
            "issue": "FEE BLOWOUT",
            "detail": f"Fees = {fee_ratio*100:.0f}% of gross P&L — strategy is paying to trade, not to profit",
            "fix": "Reduce trade frequency. The abs(tgt - last_target) > 0.02 threshold "
                   "may need to increase to 0.05 or add a minimum bars-held counter.",
        })

    for i, d in enumerate(diagnosis, 1):
        print(f"\n  [{i}] {d['issue']}")
        print(f"      WHAT:  {d['detail']}")
        print(f"      FIX:   {d['fix']}")

    if not diagnosis:
        print("  No obvious death-loop pattern found in this dataset.")

    # ── Save markdown ─────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open("results/deathloop_report.md", "w", encoding="utf-8") as f:
        f.write(f"# LARSA v8 Death Loop Report\n\n")
        f.write(f"- Trades: {len(trades)}\n")
        f.write(f"- Peak equity: ${peak_eq:,.0f} @ {peak_dt.date()}\n")
        f.write(f"- Final equity: ${final_eq:,.0f}\n")
        f.write(f"- Gave back: ${peak_eq - final_eq:,.0f} ({(peak_eq-final_eq)/peak_eq*100:.1f}% of peak)\n\n")
        f.write(f"## Diagnoses\n\n")
        for d in diagnosis:
            f.write(f"### {d['issue']}\n")
            f.write(f"- **What:** {d['detail']}\n")
            f.write(f"- **Fix:** {d['fix']}\n\n")
    print(f"\n  Report -> results/deathloop_report.md")


if __name__ == "__main__":
    main()
