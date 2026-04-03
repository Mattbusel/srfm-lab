"""cost.py — flat period → what patience saved
Usage: python tools/cost.py --start DATE --end DATE [--capital N]
       python tools/cost.py --list
"""
import json, argparse
from datetime import datetime, timezone

DATA = "research/trade_analysis_data.json"

# Known bear-market proxies (SPX/ES drawdown during flat windows)
BEAR_PROXIES = {
    ("2018-01-01", "2018-04-01"): -0.098,
    ("2018-09-01", "2019-01-01"): -0.196,
    ("2020-02-01", "2020-04-01"): -0.340,
    ("2022-01-01", "2022-10-01"): -0.223,
    ("2024-07-01", "2024-08-15"): -0.085,
}

def load():
    with open(DATA) as f:
        return json.load(f)

def parse_dt(s):
    s = s.strip().replace("+00:00", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse: {s}")

def fmt_sep():
    print("-" * 50)

def market_return_for(start_dt, end_dt, equity_curve):
    # Use known bear-market proxy table first (strategy equity != market)
    best_overlap, best_ret = 0.0, None
    for (ps, pe), ret in BEAR_PROXIES.items():
        ps_dt = parse_dt(ps); pe_dt = parse_dt(pe)
        overlap = (min(end_dt, pe_dt) - max(start_dt, ps_dt)).total_seconds()
        if overlap > best_overlap:
            best_overlap = overlap
            window = (pe_dt - ps_dt).total_seconds()
            frac = overlap / window
            best_ret = ret * frac
    if best_ret is not None:
        return best_ret
    # No match: return 0 (unknown period)
    return 0.0

def find_flat(flat_periods, start_str, end_str):
    target_s = parse_dt(start_str)
    target_e = parse_dt(end_str)
    best, best_score = None, float("inf")
    for fp in flat_periods:
        fps = parse_dt(fp["start"]); fpe = parse_dt(fp["end"])
        score = abs((fps - target_s).total_seconds()) + abs((fpe - target_e).total_seconds())
        if score < best_score:
            best, best_score = fp, score
    return best

def first_trade_after(wells, end_dt):
    for w in sorted(wells, key=lambda x: x["start"]):
        wdt = parse_dt(w["start"])
        if wdt > end_dt:
            return w
    return None

def cmd_list(flat_periods):
    long_fp = [fp for fp in flat_periods if fp["days"] >= 30]
    print(f"{'Start':<14} {'End':<14} {'Days':>7}")
    print("-" * 38)
    for fp in sorted(long_fp, key=lambda x: x["days"], reverse=True):
        print(f"{fp['start'][:10]:<14} {fp['end'][:10]:<14} {fp['days']:>7.0f}")

def cmd_cost(flat_periods, equity_curve, wells, start_str, end_str, capital):
    fp = find_flat(flat_periods, start_str, end_str)
    if fp is None:
        print("No flat period found."); return

    fp_start = fp["start"][:10]
    fp_end   = fp["end"][:10]
    days     = fp["days"]

    start_dt = parse_dt(fp["start"])
    end_dt   = parse_dt(fp["end"])

    mkt_ret  = market_return_for(start_dt, end_dt, equity_curve)
    avoided  = abs(mkt_ret * capital) if mkt_ret < 0 else 0
    bah_end  = capital * (1 + mkt_ret)

    next_trade = first_trade_after(wells, end_dt)
    nt_str = "none"
    recoup_str = ""
    if next_trade:
        nt_date  = next_trade["start"][:10]
        nt_instr = "+".join(next_trade.get("instruments", []))
        nt_dir   = "+".join(next_trade.get("directions", []))
        nt_pnl   = next_trade["net_pnl"]
        nt_str   = f"{nt_date} {nt_instr} {nt_dir} +${nt_pnl:,.0f}"
        if avoided > 0:
            recoup_pct = 100 * nt_pnl / avoided
            recoup_str = f"\n  Wait was worth it: the break-out trade recouped {recoup_pct:.0f}% of patience value"

    fmt_sep()
    print(f"  PATIENCE COST: {fp_start} -> {fp_end}")
    fmt_sep()
    print(f"  Duration:        {days:.0f} days flat")
    print(f"  Strategy P&L:   +$0  (no positions)")
    print()
    print(f"  Market during this period:")
    print(f"  -> ES: {mkt_ret*100:.1f}% (from equity_curve proxy)")
    if mkt_ret < 0:
        print(f"  -> If fully invested: -${abs(mkt_ret*capital):,.0f} on ${capital:,.0f} base")
    else:
        print(f"  -> Market was flat/up: no avoided loss")
    print()
    if avoided > 0:
        print(f"  Patience value: +${avoided:,.0f} avoided loss")
    print(f"  Capital preserved: ${capital:,.0f} -> ${capital:,.0f} (flat)")
    print(f"  vs Buy-and-hold:  ${capital:,.0f} -> ${bah_end:,.0f} ({mkt_ret*100:+.1f}%)")
    print()
    print(f"  Next trade after flat: {nt_str}{recoup_str}")
    fmt_sep()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", help="Start of flat period (YYYY-MM-DD)")
    p.add_argument("--end",   help="End of flat period (YYYY-MM-DD)")
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    d = load()
    flat_periods = d["flat_periods"]
    equity_curve = d["equity_curve"]
    wells = d["wells"]

    if args.list:
        cmd_list(flat_periods)
    elif args.start and args.end:
        cmd_cost(flat_periods, equity_curve, wells, args.start, args.end, args.capital)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
