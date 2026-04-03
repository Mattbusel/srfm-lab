#!/usr/bin/env python
"""regime.py — pipe prices -> one line of colored regime blocks + summary."""
import sys, csv, argparse, os, shutil
# ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BULL, BEAR, SW = 'BULL', 'BEAR', 'SIDEWAYS'
BLOCK = {'BULL': '█', 'BEAR': '█', 'SIDEWAYS': '░', 'HIGH_VOL': '▓'}
ANSI  = {'BULL': '\033[92m', 'BEAR': '\033[91m', 'SIDEWAYS': '\033[90m',
         'HIGH_VOL': '\033[95m', 'RESET': '\033[0m'}

def ema(prev, val, alpha):
    return alpha * val + (1 - alpha) * prev

def classify(e12, e26, e50, e200):
    if e12 > e26 > e50 > e200:
        return BULL
    if e200 > e50 > e26 > e12:
        return BEAR
    return SW

def compress_runs(runs, width):
    total = sum(r[1] for r in runs)
    if total == 0:
        return []
    scaled = []
    for regime, length in runs:
        chars = max(1, round(length / total * width))
        scaled.append((regime, chars))
    # trim/pad to exact width
    diff = sum(c for _, c in scaled) - width
    if diff > 0:
        for i in range(abs(diff)):
            scaled[-(i % len(scaled)) - 1] = (scaled[-(i % len(scaled)) - 1][0],
                                               max(1, scaled[-(i % len(scaled)) - 1][1] - 1))
    elif diff < 0:
        for i in range(abs(diff)):
            scaled[i % len(scaled)] = (scaled[i % len(scaled)][0],
                                       scaled[i % len(scaled)][1] + 1)
    return scaled

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=None)
    ap.add_argument('--no-color', action='store_true')
    args = ap.parse_args()

    use_color = sys.stdout.isatty() and not args.no_color
    width = args.width or (shutil.get_terminal_size((100, 24)).columns - 2)

    if sys.stdin.isatty():
        default_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'NDX_hourly_poly.csv')
        fh = open(default_csv, newline='')
    else:
        fh = sys.stdin

    reader = csv.DictReader(fh)
    rows = list(reader)
    if not rows:
        print("No data."); return

    a12  = 2/(12+1); a26 = 2/(26+1); a50 = 2/(50+1); a200 = 2/(200+1)
    closes = [float(r.get('close', r.get('Close', 0))) for r in rows]
    e12 = e26 = e50 = e200 = closes[0]

    raw_regimes = []
    for c in closes:
        e12  = ema(e12,  c, a12)
        e26  = ema(e26,  c, a26)
        e50  = ema(e50,  c, a50)
        e200 = ema(e200, c, a200)
        raw_regimes.append(classify(e12, e26, e50, e200))

    # build runs
    runs = []
    for r in raw_regimes:
        if runs and runs[-1][0] == r:
            runs[-1] = (r, runs[-1][1] + 1)
        else:
            runs.append((r, 1))

    scaled = compress_runs(runs, width)

    # build block line
    block_line = ''
    for regime, chars in scaled:
        ch = BLOCK.get(regime, '?') * chars
        if use_color:
            block_line += ANSI[regime] + ch + ANSI['RESET']
        else:
            block_line += ch

    # build label line (abbreviated, skip tiny runs)
    label_line = ''
    for regime, chars in scaled:
        abbr = regime[:2] if chars < 4 else (regime if chars >= len(regime)+1 else regime[:chars])
        label_line += abbr.ljust(chars)[:chars]

    # stats
    counts = {BULL: 0, BEAR: 0, SW: 0}
    for r in raw_regimes:
        counts[r] = counts.get(r, 0) + 1
    total = len(raw_regimes)
    longest = {BULL: 0, BEAR: 0, SW: 0}
    for regime, length in runs:
        if length > longest.get(regime, 0):
            longest[regime] = length

    print(block_line)
    print(label_line)
    print()
    pcts = {k: 100*v/total for k,v in counts.items()}
    print(f"Regime summary: BULL {pcts[BULL]:.1f}%  SIDEWAYS {pcts[SW]:.1f}%  BEAR {pcts[BEAR]:.1f}%")
    print(f"Total: {total} bars  |  Longest BULL: {longest[BULL]} bars  |  Longest BEAR: {longest[BEAR]} bars")

if __name__ == '__main__':
    main()
