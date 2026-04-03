"""Generate v3 forensics report."""
import json, csv
from datetime import datetime, timezone, timedelta
from collections import defaultdict

CSV_PATH = "C:/Users/Matthew/Downloads/Measured Red Anguilline_trades.csv"
JSON_PATH = "C:/Users/Matthew/Downloads/Measured Red Anguilline.json"
REGIME_PATH = "results/regimes_ES.csv"

# ---- load equity ----
with open(JSON_PATH, encoding='utf-8') as f:
    bt = json.load(f)
eq_vals = bt['charts']['Strategy Equity']['series']['Equity']['values']
equity_ts = sorted([(datetime.fromtimestamp(v[0], tz=timezone.utc), v[4]) for v in eq_vals])

def get_portfolio_at(dt):
    best = equity_ts[0][1]
    for ts, val in equity_ts:
        if ts <= dt:
            best = val
        else:
            break
    return best

# ---- load regimes ----
regimes = {}
with open(REGIME_PATH, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        d = datetime.strptime(row['date'], '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
        regimes[d] = row['regime']

def get_regime_at(dt):
    for delta_h in range(0, 25):
        for sign in [1, -1]:
            candidate = (dt + timedelta(hours=delta_h*sign)).replace(minute=0, second=0, microsecond=0)
            if candidate in regimes:
                return regimes[candidate]
    return 'UNKNOWN'

# ---- load trades ----
trades = []
with open(CSV_PATH, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['pnl'] = float(row['P&L'])
        row['qty'] = int(row['Quantity'])
        row['entry_price'] = float(row['Entry Price'])
        row['exit_price'] = float(row['Exit Price'])
        sym = row['Symbols'].strip().strip('"')
        inst = ''
        for i in sym:
            if i.isdigit():
                break
            inst += i
        row['instrument'] = inst
        row['entry_dt'] = datetime.fromisoformat(row['Entry Time'].replace('Z', '+00:00'))
        row['exit_dt'] = datetime.fromisoformat(row['Exit Time'].replace('Z', '+00:00'))
        trades.append(row)

# ---- load well data ----
with open('research/trade_analysis_v3_data.json') as f:
    v3 = json.load(f)
with open('research/trade_analysis_data.json') as f:
    v1 = json.load(f)

# ---- compute well stats ----
def well_stats(wells):
    total = len(wells)
    wins = sum(1 for w in wells if w['is_win'])
    single = [w for w in wells if len(w['instruments']) == 1]
    multi = [w for w in wells if len(w['instruments']) > 1]
    gross_pnl = sum(w['total_pnl'] for w in wells)
    multi_pnl = sum(w['total_pnl'] for w in multi)
    return {
        'total': total, 'wins': wins, 'win_rate': wins/total*100 if total else 0,
        'single_count': len(single),
        'single_wins': sum(1 for w in single if w['is_win']),
        'single_win_rate': sum(1 for w in single if w['is_win'])/len(single)*100 if single else 0,
        'single_avg_pnl': sum(w['total_pnl'] for w in single)/len(single) if single else 0,
        'single_total_pnl': sum(w['total_pnl'] for w in single),
        'multi_count': len(multi),
        'multi_wins': sum(1 for w in multi if w['is_win']),
        'multi_win_rate': sum(1 for w in multi if w['is_win'])/len(multi)*100 if multi else 0,
        'multi_avg_pnl': sum(w['total_pnl'] for w in multi)/len(multi) if multi else 0,
        'multi_total_pnl': multi_pnl,
        'multi_pct_pnl': multi_pnl/gross_pnl*100 if gross_pnl else 0,
        'gross_pnl': gross_pnl,
    }

s1 = well_stats(v1['wells'])
s3 = well_stats(v3['wells'])

wells_v3 = v3['wells']
top10_win = sorted(wells_v3, key=lambda w: w['total_pnl'], reverse=True)[:10]
top10_lose = sorted(wells_v3, key=lambda w: w['total_pnl'])[:10]

# ---- NQ trades ----
nq = [t for t in trades if t['instrument'] == 'NQ']
nq_by_notional = sorted(nq, key=lambda t: t['qty'] * t['entry_price'] * 20, reverse=True)
top10_nq = nq_by_notional[:10]
peak = nq_by_notional[0]
peak_notional = peak['qty'] * peak['entry_price'] * 20
peak_port = get_portfolio_at(peak['entry_dt'])

# ---- BEAR analysis ----
bear_buys_18 = [t for t in trades if t['entry_dt'].year == 2018 and t['Direction'] == 'Buy' and get_regime_at(t['entry_dt']) == 'BEAR']
bear_buys_19 = [t for t in trades if t['entry_dt'].year == 2019 and t['Direction'] == 'Buy' and get_regime_at(t['entry_dt']) == 'BEAR']
all_bear_buys = [t for t in trades if t['Direction'] == 'Buy' and get_regime_at(t['entry_dt']) == 'BEAR']
bear_all_18_pnl = sum(t['pnl'] for t in bear_buys_18)
bear_all_19_pnl = sum(t['pnl'] for t in bear_buys_19)

# ---- Annual from by_year ----
by_year_v1 = v1['by_year']
by_year_v3 = v3['by_year']

def annual_wells(wells):
    by_yr = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})
    for w in wells:
        yr = datetime.fromisoformat(w['start']).year
        by_yr[yr]['count'] += 1
        by_yr[yr]['pnl'] += w['total_pnl']
        if w['is_win']:
            by_yr[yr]['wins'] += 1
    return dict(by_yr)

aw_v1 = annual_wells(v1['wells'])
aw_v3 = annual_wells(v3['wells'])

# ============================
# BUILD REPORT
# ============================
lines = []

lines.append('# LARSA v3 Well Forensics — "Measured Red Anguilline"')
lines.append('')
lines.append('> Generated 2026-04-03 | 485 trades | 273 wells | 2018-2024')
lines.append('')
lines.append('---')
lines.append('')
lines.append('## Well Statistics vs v1')
lines.append('')
lines.append('| Metric | v1 | v3 | Delta |')
lines.append('|--------|----|----|-------|')
lines.append(f"| Total Wells | {s1['total']} | {s3['total']} | {s3['total']-s1['total']:+d} |")
lines.append(f"| Well Win Rate | {s1['win_rate']:.1f}% | {s3['win_rate']:.1f}% | {s3['win_rate']-s1['win_rate']:+.1f}pp |")
lines.append(f"| Well Avg P&L | ${s1['gross_pnl']/s1['total']:,.0f} | ${s3['gross_pnl']/s3['total']:,.0f} | ${(s3['gross_pnl']/s3['total'])-(s1['gross_pnl']/s1['total']):+,.0f} |")
lines.append(f"| Single-inst wells | {s1['single_count']} | {s3['single_count']} | {s3['single_count']-s1['single_count']:+d} |")
lines.append(f"| Single-inst win rate | {s1['single_win_rate']:.1f}% | {s3['single_win_rate']:.1f}% | {s3['single_win_rate']-s1['single_win_rate']:+.1f}pp |")
lines.append(f"| Single-inst avg P&L | ${s1['single_avg_pnl']:,.0f} | ${s3['single_avg_pnl']:,.0f} | ${s3['single_avg_pnl']-s1['single_avg_pnl']:+,.0f} |")
lines.append(f"| Single-inst total P&L | ${s1['single_total_pnl']:,.0f} | ${s3['single_total_pnl']:,.0f} | ${s3['single_total_pnl']-s1['single_total_pnl']:+,.0f} |")
lines.append(f"| Multi-inst wells | {s1['multi_count']} | {s3['multi_count']} | {s3['multi_count']-s1['multi_count']:+d} |")
lines.append(f"| Multi-inst win rate | {s1['multi_win_rate']:.1f}% | {s3['multi_win_rate']:.1f}% | {s3['multi_win_rate']-s1['multi_win_rate']:+.1f}pp |")
lines.append(f"| Multi-inst avg P&L | ${s1['multi_avg_pnl']:,.0f} | ${s3['multi_avg_pnl']:,.0f} | ${s3['multi_avg_pnl']-s1['multi_avg_pnl']:+,.0f} |")
lines.append(f"| Multi-inst total P&L | ${s1['multi_total_pnl']:,.0f} | ${s3['multi_total_pnl']:,.0f} | ${s3['multi_total_pnl']-s1['multi_total_pnl']:+,.0f} |")
lines.append(f"| Multi-inst % of gross P&L | {s1['multi_pct_pnl']:.1f}% | {s3['multi_pct_pnl']:.1f}% | {s3['multi_pct_pnl']-s1['multi_pct_pnl']:+.1f}pp |")
lines.append(f"| Gross P&L | ${s1['gross_pnl']:,.0f} | ${s3['gross_pnl']:,.0f} | ${s3['gross_pnl']-s1['gross_pnl']:+,.0f} |")
lines.append(f"| Net P&L | ${v1['summary']['net_pnl']:,.0f} | ${v3['summary']['net_pnl']:,.0f} | ${v3['summary']['net_pnl']-v1['summary']['net_pnl']:+,.0f} |")
lines.append(f"| Max Drawdown | {v1['summary']['max_dd_pct']:.1f}% | {v3['summary']['max_dd_pct']:.1f}% | {v3['summary']['max_dd_pct']-v1['summary']['max_dd_pct']:+.1f}pp |")
lines.append(f"| Sharpe | {v1['summary']['sharpe']:.3f} | {v3['summary']['sharpe']:.3f} | {v3['summary']['sharpe']-v1['summary']['sharpe']:+.3f} |")
lines.append('')
lines.append('**Key finding:** v3 has 95 multi-instrument wells vs v1\'s 47 (+48), but multi-inst win rate collapsed from 74.5% to 55.8% (-18.7pp). This is the primary driver of lower gross P&L. v3 also added 108 more trades (485 vs 377) while generating $800k less gross P&L — the extra activity is dilutive.')
lines.append('')
lines.append('---')
lines.append('')
lines.append('## Annual Well Counts and P&L')
lines.append('')
lines.append('| Year | v1 Wells | v1 P&L | v3 Wells | v3 P&L | Delta P&L |')
lines.append('|------|----------|--------|----------|--------|-----------|')
for yr in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
    yrstr = str(yr)
    v1d = aw_v1.get(yr, {'count': 0, 'pnl': 0})
    v3d = aw_v3.get(yr, {'count': 0, 'pnl': 0})
    v1_pnl = by_year_v1.get(yrstr, {}).get('pnl', 0)
    v3_pnl = by_year_v3.get(yrstr, {}).get('pnl', 0)
    lines.append(f'| {yr} | {v1d["count"]} | ${v1_pnl:,.0f} | {v3d["count"]} | ${v3_pnl:,.0f} | ${v3_pnl-v1_pnl:+,.0f} |')
lines.append('')
lines.append('---')
lines.append('')
lines.append('## NQ Notional Exposure Analysis')
lines.append('')
lines.append('NQ futures multiplier: $20/point. All exposures are gross notional at entry.')
lines.append('')
lines.append('### Top 10 NQ Trades by Notional')
lines.append('')
lines.append('| Date | Dir | Qty | Entry | Notional | Portfolio | % Portfolio | P&L |')
lines.append('|------|-----|-----|-------|----------|-----------|-------------|-----|')
for t in top10_nq:
    notional = t['qty'] * t['entry_price'] * 20
    port = get_portfolio_at(t['entry_dt'])
    pct = notional / port * 100
    lines.append(f'| {str(t["entry_dt"])[:19]} | {t["Direction"]} | {t["qty"]} | {t["entry_price"]:,.2f} | ${notional:,.0f} | ${port:,.0f} | {pct:.1f}% | ${t["pnl"]:,.0f} |')
lines.append('')
lines.append(f'**Peak NQ notional: ${peak_notional:,.0f} on {str(peak["entry_dt"])[:10]} ({peak_notional/peak_port*100:.1f}% of ${peak_port:,.0f} portfolio)**')
lines.append('')
lines.append('### The 3 Killer NQ Trades')
lines.append('')
killer_specs = [
    ('2024-10-14', -265000),
    ('2024-06-18', -176000),
    ('2024-12-09', -121000),
]
for date_str, _ in killer_specs:
    candidates = [t for t in nq if t['entry_dt'].date().isoformat() == date_str]
    if not candidates:
        lines.append(f'**{date_str}** — no NQ trade found')
        continue
    t = min(candidates, key=lambda x: x['pnl'])
    notional = t['qty'] * t['entry_price'] * 20
    port = get_portfolio_at(t['entry_dt'])
    pct = notional / port * 100
    move = t['exit_price'] - t['entry_price']
    lines.append(f'#### {date_str} — NQ {t["Direction"]}, ${t["pnl"]:,.0f}')
    lines.append(f'- **Entry:** {t["entry_price"]:,.2f}  **Exit:** {t["exit_price"]:,.2f}  **Move:** {move:+.2f} pts ({move/t["entry_price"]*100:+.2f}%)')
    lines.append(f'- **Quantity:** {t["qty"]} contracts')
    lines.append(f'- **Notional:** ${notional:,.0f}')
    lines.append(f'- **Portfolio at entry:** ${port:,.0f}')
    lines.append(f'- **NQ notional as % of portfolio:** {pct:.1f}%')
    lines.append(f'- **P&L:** ${t["pnl"]:,.0f}')
    lines.append(f'- **Point loss:** {move:+.2f} pts x {t["qty"]} contracts x $20 = ${t["pnl"]:,.0f}')
    lines.append('')
lines.append('**Root cause:** NQ sizing is proportional to portfolio equity. As the portfolio grew to $2.7-3.2M, the position sizer allocated 48-56 NQ contracts. At $20/point, a 276-point adverse move on 48 contracts = $-265,200. The position runs at 700-830% of portfolio NAV in gross notional — making single-trade drawdowns of 4-8% of equity routine and catastrophic tail events inevitable without a notional cap.')
lines.append('')
lines.append('---')
lines.append('')
lines.append('## BEAR Regime Analysis (2018-2019)')
lines.append('')
lines.append('Regime labels from `results/regimes_ES.csv` (ES-based regime classifier). BEAR label = regime state classified as BEAR at trade entry hour.')
lines.append('')
lines.append('### 2018-2019 Trade P&L by Direction')
lines.append('')
lines.append('| Year | Buy Trades | Buy P&L | Sell Trades | Sell P&L | Total P&L |')
lines.append('|------|-----------|---------|------------|---------|-----------|')
for yr in [2018, 2019]:
    yr_trades = [t for t in trades if t['entry_dt'].year == yr]
    buys = [t for t in yr_trades if t['Direction'] == 'Buy']
    sells = [t for t in yr_trades if t['Direction'] == 'Sell']
    lines.append(f'| {yr} | {len(buys)} | ${sum(t["pnl"] for t in buys):,.0f} | {len(sells)} | ${sum(t["pnl"] for t in sells):,.0f} | ${sum(t["pnl"] for t in yr_trades):,.0f} |')
lines.append('')
lines.append('### BEAR-Regime Long Gate Analysis')
lines.append('')
lines.append('If the `rhb > 5` BEAR gate had been applied (blocking all Buy entries when regime = BEAR):')
lines.append('')
lines.append('| Year | Blocked Longs | P&L of Blocked Trades | Recovery (savings) |')
lines.append('|------|--------------|----------------------|-------------------|')
lines.append(f'| 2018 | {len(bear_buys_18)} | ${bear_all_18_pnl:,.0f} | ${-bear_all_18_pnl:+,.0f} |')
lines.append(f'| 2019 | {len(bear_buys_19)} | ${bear_all_19_pnl:,.0f} | ${-bear_all_19_pnl:+,.0f} |')
lines.append(f'| **Total** | **{len(bear_buys_18)+len(bear_buys_19)}** | **${bear_all_18_pnl+bear_all_19_pnl:,.0f}** | **${-(bear_all_18_pnl+bear_all_19_pnl):+,.0f}** |')
lines.append('')
lines.append('**2018 note:** The 2 blocked BEAR-regime longs in 2018 were winners (+$7,060). Blocking them would cost $7k. The 2018 bear losses came from SIDEWAYS-regime longs and Sell trades during the Q4 2018 crash, not from BEAR-labeled periods.')
lines.append('')
lines.append(f'**Net regime-gate recovery: $68,704** (dominated by 21 blocked BEAR-regime longs in 2019 saving $75,764, offset by $7,060 cost in 2018).')
lines.append('')
bear_all_pnl = sum(t['pnl'] for t in all_bear_buys)
lines.append(f'**Full-backtest BEAR-regime longs:** {len(all_bear_buys)} trades across all years, cumulative P&L = ${bear_all_pnl:,.0f}. Blocking all would recover ${-bear_all_pnl:,.0f}.')
lines.append('')
lines.append('---')
lines.append('')
lines.append('## Top 10 Winning Wells (v3)')
lines.append('')
lines.append('| Start | End | Dur | Instruments | Dir | P&L | Trades |')
lines.append('|-------|-----|-----|-------------|-----|-----|--------|')
for w in top10_win:
    insts = '+'.join(w['instruments'])
    lines.append(f'| {w["start"][:10]} | {w["end"][:10]} | {w["duration_h"]:.0f}h | {insts} | {w["directions"][0]} | ${w["total_pnl"]:,.0f} | {w["n_trades"]} |')
lines.append('')
lines.append('---')
lines.append('')
lines.append('## Top 10 Losing Wells (v3)')
lines.append('')
lines.append('| Start | End | Dur | Instruments | Dir | P&L | Trades |')
lines.append('|-------|-----|-----|-------------|-----|-----|--------|')
for w in top10_lose:
    insts = '+'.join(w['instruments'])
    lines.append(f'| {w["start"][:10]} | {w["end"][:10]} | {w["duration_h"]:.0f}h | {insts} | {w["directions"][0]} | ${w["total_pnl"]:,.0f} | {w["n_trades"]} |')
lines.append('')
lines.append('---')
lines.append('')
lines.append('## v3 vs v1 Well Quality Comparison')
lines.append('')
lines.append('| Dimension | v1 | v3 | Interpretation |')
lines.append('|-----------|----|----|----------------|')
lines.append(f"| Gross P&L | ${s1['gross_pnl']:,.0f} | ${s3['gross_pnl']:,.0f} | v3 generates $800k less despite 108 more trades |")
lines.append(f"| Multi-inst wells | {s1['multi_count']} | {s3['multi_count']} | v3 fires multi-inst wells 2x as often |")
lines.append(f"| Multi-inst win rate | {s1['multi_win_rate']:.1f}% | {s3['multi_win_rate']:.1f}% | Quality of multi-inst signals degraded severely |")
lines.append(f"| Avg multi-inst P&L | ${s1['multi_avg_pnl']:,.0f} | ${s3['multi_avg_pnl']:,.0f} | Each multi-inst well earns 65% less on average |")
lines.append(f"| Avg single-inst P&L | ${s1['single_avg_pnl']:,.0f} | ${s3['single_avg_pnl']:,.0f} | Single-inst wells essentially unchanged |")
lines.append(f"| Max drawdown | {v1['summary']['max_dd_pct']:.1f}% | {v3['summary']['max_dd_pct']:.1f}% | v3 has 7.7pp more drawdown |")
lines.append(f"| Sharpe | {v1['summary']['sharpe']:.3f} | {v3['summary']['sharpe']:.3f} | v3 Sharpe slightly higher (+0.47) despite lower P&L |")
lines.append(f"| Well avg win P&L | ${v1['summary']['well_avg_win_pnl']:,.0f} | ${v3['summary']['well_avg_win_pnl']:,.0f} | v3 wins are much smaller |")
lines.append(f"| Well avg loss P&L | ${v1['summary']['well_avg_loss_pnl']:,.0f} | ${v3['summary']['well_avg_loss_pnl']:,.0f} | v3 losses also smaller (more symmetric but lower edge) |")
lines.append('')
lines.append('### Diagnosis')
lines.append('')
lines.append('The v3 configuration has **over-fired multi-instrument coordination**: the simultaneous ES+NQ+YM signal logic triggers more often, but when wrong, three losing legs compound. In v1, 74.5% of multi-inst wells were winners, generating $2.36M from just 47 events. In v3, 55.8% win rate across 95 events generates only $1.66M. The correlated-entry threshold is too permissive in v3.')
lines.append('')
lines.append('The NQ single-trade notional exposure (700-830% of portfolio) is the most dangerous structural flaw. A position sizer without a notional-cap-as-fraction-of-NAV will continue producing catastrophic single-day drawdowns as NAV grows.')
lines.append('')
lines.append('---')
lines.append('')
lines.append('*Report generated by trade forensics pipeline. Source: Measured Red Anguilline (v3 QC backtest).*')

report = '\n'.join(lines)
with open('research/trade_analysis_v3_wells.md', 'w', encoding='utf-8') as f:
    f.write(report)
print(f'Report written: {len(lines)} lines')
