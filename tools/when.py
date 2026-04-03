#!/usr/bin/env python
"""when.py — dollar amount → find the trade closest to that P&L."""
import sys, csv, argparse, os
from datetime import datetime

DEFAULT_CSVS = [
    "C:/Users/Matthew/Downloads/Calm Orange Mule_trades.csv",
    "C:/Users/Matthew/Downloads/Measured Red Anguilline_trades.csv",
]

def load_trades(path, label=''):
    trades = []
    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pnl = float(row.get('P&L', row.get('pnl', 0)))
                    entry = row.get('Entry Time', row.get('date', ''))
                    exit_t = row.get('Exit Time', '')
                    sym = row.get('Symbols', row.get('ticker', ''))
                    direction = row.get('Direction', row.get('direction', ''))
                    try:
                        dt_in  = datetime.fromisoformat(entry.replace('Z',''))
                        dt_out = datetime.fromisoformat(exit_t.replace('Z','')) if exit_t else dt_in
                        duration = dt_out - dt_in
                        date_str = dt_in.strftime('%Y-%m-%d %H:%M')
                        dur_str = str(duration)
                    except Exception:
                        date_str = str(entry)[:16]
                        dur_str = '?'
                    trades.append({
                        'date': date_str, 'sym': sym.strip('"').strip(),
                        'dir': direction, 'pnl': pnl, 'dur': dur_str, 'source': label
                    })
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return trades

def load_regimes(path):
    regimes = {}
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_key = str(row.get('date', row.get('Date', '')))[:10]
                regimes[date_key] = row.get('regime', 'UNKNOWN')
    except Exception:
        pass
    return regimes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('amount', type=float)
    ap.add_argument('--csv', action='append', dest='csvs')
    ap.add_argument('--tol', type=float, default=5.0, help='tolerance %%')
    args = ap.parse_args()

    target = args.amount
    csv_paths = args.csvs if args.csvs else DEFAULT_CSVS

    all_trades = []
    for i, p in enumerate(csv_paths):
        label = f"v{i+1} ({os.path.basename(p)})"
        all_trades.extend(load_trades(p, label))

    if not all_trades:
        print("No trades loaded.", file=sys.stderr); return

    tol_abs = abs(target) * args.tol / 100.0
    candidates = [t for t in all_trades if abs(t['pnl'] - target) <= max(tol_abs, 1000)]
    candidates.sort(key=lambda t: abs(t['pnl'] - target))

    # load regimes for annotation
    regimes_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'regimes_ES.csv')
    regimes = load_regimes(regimes_path)

    if not candidates:
        print(f"No trades within {args.tol}% of ${target:,.0f}")
        # show top 3 closest regardless
        all_trades.sort(key=lambda t: abs(t['pnl'] - target))
        candidates = all_trades[:3]
        print("Closest matches:")

    for t in candidates[:3]:
        delta = t['pnl'] - target
        sign_d = '+' if delta >= 0 else ''
        date_key = t['date'][:10]
        regime = regimes.get(date_key, '')
        regime_str = f"  regime={regime}" if regime else ''
        print(f"{t['date']}  {t['sym']}  {t['dir']}  P&L: ${t['pnl']:+,.0f}  (d={sign_d}${delta:,.0f} from query)")
        print(f"  duration={t['dur']}{regime_str}")
        print(f"  Source: {t['source']}")
        print()

if __name__ == '__main__':
    main()
