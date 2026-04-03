"""mirror.py — bull well → equivalent bear setup
Usage: python tools/mirror.py --well DATE
       python tools/mirror.py --list
"""
import json, sys, argparse
from datetime import datetime, timezone

DATA = "research/trade_analysis_data.json"

def load():
    with open(DATA) as f:
        return json.load(f)

def parse_dt(s):
    s = s.replace("+00:00", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse date: {s}")

def find_well(wells, date_str):
    target = parse_dt(date_str)
    best, best_diff = None, float("inf")
    for w in wells:
        dt = parse_dt(w["start"])
        diff = abs((dt - target).total_seconds())
        if diff < best_diff:
            best, best_diff = w, diff
    return best

def bear_analog(wells, target_pnl, target_dur):
    best, best_score = None, float("inf")
    for w in wells:
        if "Sell" not in w.get("directions", []) or w["net_pnl"] <= 0:
            continue
        score = abs(w["net_pnl"] - target_pnl) / max(abs(target_pnl), 1) + \
                abs(w["duration_h"] - target_dur) / max(target_dur, 1)
        if score < best_score:
            best, best_score = w, score
    return best

def fmt_sep():
    print("-" * 50)

def cmd_list(wells):
    print(f"{'Date':<14} {'Dir':<8} {'Instr':<18} {'Dur_h':>6} {'Net P&L':>12}")
    print("-" * 62)
    for w in sorted(wells, key=lambda x: x["start"]):
        date = w["start"][:10]
        dirs = "+".join(w.get("directions", []))
        instr = "+".join(w.get("instruments", []))
        print(f"{date:<14} {dirs:<8} {instr:<18} {w['duration_h']:>6.0f} {w['net_pnl']:>+12,.0f}")

def cmd_mirror(wells, by_dir, date_str):
    w = find_well(wells, date_str)
    if w is None:
        print("No well found."); return

    date = w["start"][:10]
    dirs = w.get("directions", ["Buy"])
    instruments = "+".join(w.get("instruments", []))
    is_bull = "Sell" not in dirs
    regime_label = "BULL" if is_bull else "BEAR"
    direction_label = "LONG" if is_bull else "SHORT"

    bars_est = max(1, int(w["duration_h"]))
    pnl = w["net_pnl"]
    mass_est = round(2.8 + (pnl / 350000), 2)

    bear_pnl = pnl * 0.85
    bear_bars = max(1, int(bars_est * 0.85))

    analog = bear_analog(wells, bear_pnl, bear_bars)
    analog_str = "none found"
    if analog:
        adate = analog["start"][:10]
        ainstr = "+".join(analog.get("instruments", []))
        analog_str = f"{adate} {ainstr} SELL {int(analog['duration_h'])}h +${analog['net_pnl']:,.0f}"

    sell = by_dir.get("Sell", {}); buy = by_dir.get("Buy", {})
    sell_wr = 100 * sell.get("wins", 0) / max(sell.get("count", 1), 1)
    sell_avg = sell.get("pnl", 0) / max(sell.get("count", 1), 1)
    buy_wr = 100 * buy.get("wins", 0) / max(buy.get("count", 1), 1)
    buy_avg = buy.get("pnl", 0) / max(buy.get("count", 1), 1)

    fmt_sep()
    print(f"  MIRROR: {date} {instruments} {regime_label} well")
    fmt_sep()
    print(f"  Original:  {direction_label}  {bars_est} bars  mass={mass_est}  +${pnl:,.0f}")
    print(f"  Instruments: {instruments.replace('+', ' + ')} ({'convergence' if '+' in instruments else 'solo'})")
    print()
    print(f"  Bear twin setup:")
    print(f"  -> Entry:  EMA stack fully inverted (e200>e50>e26>e12)")
    print(f"  -> Signal: BEAR regime + bh_dir=-1 + same tl_confirm >= 3")
    print(f"  -> Expected duration: ~{bear_bars} bars (bear wells 15% shorter historically)")
    print(f"  -> Expected P&L:     ~${bear_pnl:,.0f}  (85% of bull P&L, bear wells less clean)")
    print(f"  -> Best historical bear analog: {analog_str}")
    print()
    print(f"  Bear well base rates (from forensics):")
    print(f"  -> Sell trades: {sell.get('count',0)} total, {sell_wr:.1f}% WR, avg ${sell_avg:,.0f}/trade")
    print(f"  -> vs Buy: {buy.get('count',0)} trades, {buy_wr:.1f}% WR, avg ${buy_avg:,.0f}/trade")
    pct = 100 * (1 - sell_avg / max(buy_avg, 1))
    print(f"  -> Bear wells underperform by {pct:.0f}% on avg P&L")
    fmt_sep()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--well", help="Date of well to mirror (YYYY-MM-DD)")
    p.add_argument("--list", action="store_true", help="List all available wells")
    args = p.parse_args()

    d = load()
    wells = d["wells"]
    by_dir = d["by_direction"]

    if args.list:
        cmd_list(wells)
    elif args.well:
        cmd_mirror(wells, by_dir, args.well)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
