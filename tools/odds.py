#!/usr/bin/env python
"""odds.py — current SRFM state → historical win probability from well data."""
import sys, json, argparse, os, math

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'research', 'trade_analysis_data.json')
SEP = '=' * 40

def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    center = (p + z**2/(2*n)) / (1 + z**2/n)
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)
    return center - margin, center + margin

def load_wells(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return d.get('wells', [])

def filter_wells(wells, regime, tl_confirm, bh_active, convergence, min_mass):
    result = []
    for w in wells:
        if regime:
            # regime not directly in well; skip if we have a regime field that doesn't match
            w_regime = w.get('regime', w.get('direction', ''))
            # wells don't carry regime field directly; use direction as proxy or skip filter
            # if well has a regime key, filter on it
            if 'regime' in w and w['regime'].upper() != regime.upper():
                continue
        if convergence is not None:
            instr = w.get('instruments', w.get('n_instruments', []))
            n_instr = len(instr) if isinstance(instr, list) else int(instr or 1)
            if n_instr < convergence:
                continue
        if tl_confirm is not None:
            dur = w.get('duration_h', w.get('duration_bars', 0))
            if dur < tl_confirm * 2:
                continue
        if min_mass is not None:
            mass = w.get('mass_peak', w.get('bh_mass', w.get('max_qty', 0)))
            if mass < min_mass:
                continue
        result.append(w)
    return result

def stats(wells):
    if not wells:
        return {'n': 0, 'wins': 0, 'wr': 0, 'avg_pnl': 0, 'avg_dur': 0, 'best': 0, 'worst': 0,
                'best_date': '', 'worst_date': ''}
    wins = sum(1 for w in wells if w.get('is_win', w.get('net_pnl', 0) > 0))
    pnls = [w.get('net_pnl', w.get('pnl', 0)) for w in wells]
    durs = [w.get('duration_h', w.get('duration_bars', 0)) for w in wells]
    best_idx = max(range(len(pnls)), key=lambda i: pnls[i])
    worst_idx = min(range(len(pnls)), key=lambda i: pnls[i])
    best_date = wells[best_idx].get('start', wells[best_idx].get('formed_at', ''))[:10]
    worst_date = wells[worst_idx].get('start', wells[worst_idx].get('formed_at', ''))[:10]
    return {
        'n': len(wells), 'wins': wins, 'wr': wins/len(wells),
        'avg_pnl': sum(pnls)/len(pnls), 'avg_dur': sum(durs)/len(durs),
        'best': max(pnls), 'worst': min(pnls),
        'best_date': best_date, 'worst_date': worst_date,
    }

def verdict(wr):
    if wr > 0.65: return 'STRONG EDGE - size up to convergence cap (0.65)'
    if wr > 0.55: return 'EDGE - normal sizing'
    if wr > 0.50: return 'MARGINAL - reduce size'
    return 'NO EDGE - stand aside'

def fmt(v):
    sign = '+' if v >= 0 else '-'
    return f"{sign}${abs(v):,.0f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', default=None)
    ap.add_argument('--tl_confirm', type=int, default=None)
    ap.add_argument('--bh_active', default=None)
    ap.add_argument('--convergence', type=int, default=None)
    ap.add_argument('--min-mass', type=float, default=None)
    args = ap.parse_args()

    wells = load_wells(DATA_PATH)
    baseline_stats = stats(wells)
    filtered = filter_wells(wells, args.regime, args.tl_confirm,
                            args.bh_active, args.convergence, args.min_mass)
    s = stats(filtered)

    lo, hi = wilson_ci(s['wins'], s['n'])
    state_parts = []
    if args.regime:       state_parts.append(args.regime)
    if args.bh_active:    state_parts.append('BH active')
    if args.convergence:  state_parts.append(f'convergence={args.convergence}')
    if args.tl_confirm:   state_parts.append(f'tl_confirm={args.tl_confirm}')
    state_str = ' + '.join(state_parts) if state_parts else 'all wells'

    print(SEP)
    print('  Historical odds for current state')
    print(SEP)
    print(f"  State: {state_str}")
    if s['n'] == 0:
        print("  No matching wells found.")
        print(SEP); return

    print(f"  Historical win rate: {s['wr']*100:.1f}%  (n={s['n']})")
    print(f"  95% CI: [{lo*100:.1f}% - {hi*100:.1f}%]  (Wilson interval)")
    print(f"  Avg P&L per event:  {fmt(s['avg_pnl'])}")
    print(f"  Avg duration:        {s['avg_dur']:.1f} bars")
    print(f"  Best outcome:       {fmt(s['best'])}  ({s['best_date']})")
    print(f"  Worst outcome:      {fmt(s['worst'])}  ({s['worst_date']})")
    print()
    bwr = baseline_stats['wr']
    edge_pp = (s['wr'] - bwr) * 100
    sign = '+' if edge_pp >= 0 else ''
    print(f"  Baseline (no filter): {bwr*100:.1f}%  WR")
    print(f"  Edge vs baseline:     {sign}{edge_pp:.1f}pp")
    print()
    print(f"  Verdict: {verdict(s['wr'])}")
    print(SEP)

if __name__ == '__main__':
    main()
