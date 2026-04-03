#!/usr/bin/env python
"""well.py — pipe prices → print every detected BH well, one line each."""
import sys, csv, argparse, os, math
from datetime import datetime

def colorize(text, color, use_color):
    codes = {'green': '\033[92m', 'red': '\033[91m', 'reset': '\033[0m'}
    if not use_color:
        return text
    return f"{codes[color]}{text}{codes['reset']}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cf', type=float, default=0.005)
    ap.add_argument('--min-mass', type=float, default=1.5)
    ap.add_argument('--min-bars', type=int, default=3)
    args = ap.parse_args()

    use_color = sys.stdout.isatty()

    if sys.stdin.isatty():
        default_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'NDX_hourly_poly.csv')
        fh = open(default_csv, newline='')
    else:
        fh = sys.stdin

    reader = csv.DictReader(fh)
    rows = list(reader)
    if not rows:
        print("No data.", file=sys.stderr); return

    mass = 0.0
    ctl = 0          # consecutive timelike bars
    in_bh = False
    bh_start = None
    bh_start_idx = 0
    bh_mass_peak = 0.0
    bh_direction = 'BULL'
    prev_close = None

    for i, row in enumerate(rows):
        close = float(row.get('close', row.get('Close', 0)))
        ts_raw = row.get('date', row.get('timestamp', row.get('Date', '')))
        if prev_close is None:
            prev_close = close
            continue

        delta = abs(close - prev_close)
        beta = delta / (abs(prev_close) * args.cf) if prev_close != 0 else 0
        timelike = beta < 1.0

        if timelike:
            mass = mass * 0.95 + beta * (1 - 0.95) * 10
            ctl += 1
        else:
            mass *= 0.7
            ctl = 0

        if not in_bh:
            if mass >= args.min_mass and ctl >= 5:
                in_bh = True
                bh_start = ts_raw
                bh_start_idx = i
                bh_mass_peak = mass
                # direction from recent slope
                lookback = min(5, i)
                old_close = float(rows[max(0, i - lookback)].get('close', rows[max(0, i-lookback)].get('Close', close)))
                bh_direction = 'BULL' if close > old_close else 'BEAR'
        else:
            bh_mass_peak = max(bh_mass_peak, mass)
            collapsed = mass < 1.0 or not timelike
            last_bar = i == len(rows) - 1
            if collapsed or last_bar:
                duration = i - bh_start_idx
                if duration >= args.min_bars:
                    pnl = bh_mass_peak * duration * args.cf * 1000
                    if bh_direction == 'BEAR':
                        pnl = -pnl
                    sign = '+' if pnl >= 0 else '-'
                    color = 'green' if pnl >= 0 else 'red'
                    pnl_str = f"{sign}${abs(pnl):,.0f}"
                    # Parse date
                    try:
                        dt = datetime.fromisoformat(bh_start.replace('Z',''))
                        date_str = dt.strftime('%Y-%m-%d')
                    except Exception:
                        date_str = str(bh_start)[:10]
                    line = f"{date_str}  {bh_direction}  {duration}bars  mass={bh_mass_peak:.2f}  {pnl_str}"
                    print(colorize(line, color, use_color))
                in_bh = False
                mass = 0.0
                ctl = 0

        prev_close = close

if __name__ == '__main__':
    main()
