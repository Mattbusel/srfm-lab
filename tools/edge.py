#!/usr/bin/env python
"""edge.py — two trade CSVs → what changed between versions."""
import sys, csv, argparse
from collections import defaultdict
from datetime import datetime

SEP = '-' * 40

def load_trades(path):
    trades = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pnl = float(row.get('P&L', row.get('pnl', 0)))
                entry = row.get('Entry Time', row.get('date', ''))
                sym = row.get('Symbols', row.get('ticker', ''))
                direction = row.get('Direction', row.get('direction', ''))
                # normalize date to YYYY-MM-DD
                try:
                    dt = datetime.fromisoformat(entry.replace('Z',''))
                    date_key = dt.strftime('%Y-%m-%d')
                    year = dt.year
                except Exception:
                    date_key = str(entry)[:10]
                    year = int(date_key[:4]) if len(date_key) >= 4 else 0
                sym_clean = sym.strip('"').strip()[:6]
                trades.append({'date': date_key, 'sym': sym_clean, 'dir': direction, 'pnl': pnl, 'year': year})
            except (ValueError, KeyError):
                continue
    return trades

def match_trades(a_list, b_list):
    """Match by date+sym. Returns matched pairs and unmatched lists."""
    from collections import defaultdict
    b_by_key = defaultdict(list)
    for t in b_list:
        b_by_key[(t['date'], t['sym'])].append(t)

    matched = []
    unmatched_a = []
    used_b = set()

    for ta in a_list:
        key = (ta['date'], ta['sym'])
        candidates = b_by_key.get(key, [])
        found = None
        for idx, tb in enumerate(candidates):
            bid = id(tb)
            if bid not in used_b:
                found = tb
                used_b.add(bid)
                break
        if found:
            matched.append((ta, found))
        else:
            unmatched_a.append(ta)

    unmatched_b = [t for t in b_list if id(t) not in used_b]
    return matched, unmatched_a, unmatched_b

def fmt_pnl(v):
    sign = '+' if v >= 0 else '-'
    return f"{sign}${abs(v):,.0f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file1')
    ap.add_argument('file2')
    ap.add_argument('--by-year', action='store_true')
    args = ap.parse_args()

    a = load_trades(args.file1)
    b = load_trades(args.file2)

    matched, unmatched_a, unmatched_b = match_trades(a, b)

    net_a = sum(t['pnl'] for t in a)
    net_b = sum(t['pnl'] for t in b)

    new_wins  = [t for t in unmatched_b if t['pnl'] > 0]
    new_losses= [t for t in unmatched_b if t['pnl'] <= 0]
    lost_wins = [ta for ta in unmatched_a if ta['pnl'] > 0]
    # fixed losses: matched where b pnl > a pnl and a was negative
    fixed     = [(ta,tb) for ta,tb in matched if ta['pnl'] < 0 and tb['pnl'] > ta['pnl']]

    new_wins_sum   = sum(t['pnl'] for t in new_wins)
    new_losses_sum = sum(t['pnl'] for t in new_losses)
    lost_wins_sum  = sum(ta['pnl'] for ta in lost_wins)
    fixed_sum      = sum(tb['pnl'] - ta['pnl'] for ta, tb in fixed)
    net_edge = net_b - net_a

    # biggest year-level change
    by_year_a = defaultdict(float)
    by_year_b = defaultdict(float)
    for t in a: by_year_a[t['year']] += t['pnl']
    for t in b: by_year_b[t['year']] += t['pnl']
    all_years = set(by_year_a) | set(by_year_b)
    biggest_yr = max(all_years, key=lambda y: abs(by_year_b[y] - by_year_a[y])) if all_years else None
    # infer name from filename
    n1 = args.file1.split('/')[-1].replace('_trades.csv','')
    n2 = args.file2.split('/')[-1].replace('_trades.csv','')

    print(f"\nedge — {n1} vs {n2}")
    print(SEP)
    print(f"v1 trades: {len(a)}  net: {fmt_pnl(net_a)}")
    print(f"v2 trades: {len(b)}  net: {fmt_pnl(net_b)}")
    print()
    nw_tag = '' if new_losses_sum >= 0 else '  [net loser]'
    print(f"New wins  (in v2, not v1):  {fmt_pnl(new_wins_sum)}  ({len(new_wins)} trades)")
    print(f"New losses(in v2, not v1):  {fmt_pnl(new_losses_sum)}  ({len(new_losses)} trades){nw_tag}")
    print(f"Lost wins (in v1, not v2):  {fmt_pnl(lost_wins_sum)}  ({len(lost_wins)} trades)  [missed]")
    print(f"Fixed losses(v1 loss->v2 smaller): {fmt_pnl(fixed_sum)}")
    print()
    better = 'BETTER' if net_edge >= 0 else 'WORSE'
    print(f"Net edge: {fmt_pnl(net_edge)}  v2 {better} by ${abs(net_edge):,.0f}")
    if biggest_yr:
        va = by_year_a[biggest_yr]; vb = by_year_b[biggest_yr]
        print(f"Biggest change: {biggest_yr}  v1 {fmt_pnl(va)}  v2 {fmt_pnl(vb)}  ({fmt_pnl(vb-va)})")
    print(SEP)

    if args.by_year:
        print("\nAnnual breakdown:")
        for yr in sorted(all_years):
            va = by_year_a[yr]; vb = by_year_b[yr]
            print(f"  {yr}  v1 {fmt_pnl(va):>14}  v2 {fmt_pnl(vb):>14}  Δ {fmt_pnl(vb-va):>14}")

if __name__ == '__main__':
    main()
