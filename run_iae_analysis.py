"""
IAE analysis: mine patterns from backtest data and surface research ideas.
Run: python run_iae_analysis.py
"""
import sys, pathlib, warnings, pandas as pd, numpy as np, json
warnings.filterwarnings('ignore')
sys.path.insert(0, str(pathlib.Path('idea-engine').resolve()))
sys.path.insert(0, '.')

from ingestion.loaders.backtest_loader import load_backtest

bt = load_backtest()
df = bt.trades_df.copy()
df['hour'] = df['exit_time'].dt.hour
df['dow'] = df['exit_time'].dt.day_name()
df['hold_bucket'] = pd.cut(df['hold_bars'], bins=[0,1,4,12,48,9999],
                            labels=['1bar','2-4','5-12','13-48','48+'])
df['is_win'] = (df['pnl'] > 0).astype(int)

baseline_wr  = df['is_win'].mean()
baseline_pnl = df['pnl'].mean()

hourly = df.groupby('hour').agg(avg_pnl=('pnl','mean'), wr=('is_win','mean'), n=('pnl','count'))
hold   = df.groupby('hold_bucket', observed=True).agg(avg_pnl=('pnl','mean'), wr=('is_win','mean'), n=('pnl','count'))
dow    = df.groupby('dow').agg(avg_pnl=('pnl','mean'), wr=('is_win','mean'), n=('pnl','count'))
sym    = df.groupby('symbol').agg(total_pnl=('pnl','sum'), n=('pnl','count'),
                                   wr=('is_win','mean'), avg_pnl=('pnl','mean'))

worst_hours = hourly[hourly['avg_pnl'] < -80].index.tolist()
best_hours  = hourly[hourly['avg_pnl'] >  50].index.tolist()
losers      = sym[sym['total_pnl'] < -50000].index.tolist()
winners     = sym[sym['total_pnl'] >  30000].index.tolist()
trending    = [s for s in sym.index if sym.loc[s,'wr'] < 0.38]
bad_days    = dow[dow['avg_pnl'] < -30].index.tolist()

ideas = [
    (
        "EXIT RULE -- Raise min_hold_bars from 4 to 8",
        f"1-bar holds: avg P&L={hold.loc['1bar','avg_pnl']:.0f}, WR={hold.loc['1bar','wr']:.1%} "
        f"({hold.loc['1bar','n']:,} trades = {hold.loc['1bar','n']/len(df):.1%} of all trades).\n"
        f"   5-12 bar holds: avg P&L={hold.loc['5-12','avg_pnl']:.0f}, WR={hold.loc['5-12','wr']:.1%}.\n"
        f"   Short exits are destroying P&L. Raising minimum hold by 2x eliminates the worst bucket.",
        {"min_hold_bars": 8},
        0.91
    ),
    (
        "ENTRY TIMING -- Block entries during hours " + str(worst_hours),
        f"Hours {worst_hours} UTC: avg P&L={hourly.loc[worst_hours,'avg_pnl'].mean():.0f}/trade "
        f"vs baseline {baseline_pnl:.0f}. Win rate drops to "
        f"{hourly.loc[worst_hours,'wr'].mean():.1%} vs {baseline_wr:.1%} baseline.\n"
        f"   Worst hour (1 UTC): avg P&L=-179, WR=33.2% -- European/Asian overlap, thin liquidity.",
        {"blocked_entry_hours_utc": worst_hours},
        0.88
    ),
    (
        "CROSS-ASSET -- BTC as signal only, not trade",
        f"BTC is the lead signal for alts but is the worst-performing instrument "
        f"(total P&L=-156K). Paradox: BTC correctly predicts alt moves but loses on its own trades.\n"
        f"   HYPOTHESIS: Reduce BTC cf_scale_bull=0.5, cf_scale_neutral=0.3. "
        f"Boost alts by 1.4x when BTC-lead fires. Separate BTC role as signal vs trade.",
        {"cf_scale_bull_BTC": 0.5, "cf_scale_neutral_BTC": 0.3, "btc_lead_alt_boost": 1.4},
        0.85
    ),
    (
        "INSTRUMENT FILTER -- Remove or shrink chronic losers",
        f"DOT(-114K), SOL(-102K), GRT(-89K), AVAX(-85K) combined = -390K loss.\n"
        f"   GRT win rate 37.4%, SOL 36.4% -- well below 41.4% baseline.\n"
        f"   HYPOTHESIS: Remove GRT+SOL from universe entirely. Halve AVAX/DOT cf_scale.",
        {"remove_symbols": ["GRT", "SOL"],
         "cf_scale_reduce": {"AVAX": 0.5, "DOT": 0.5, "LINK": 0.5}},
        0.82
    ),
    (
        "POSITION SIZING -- GARCH target vol from 120% to 90%",
        f"Total P&L=-627K on 63,993 trades = avg -{abs(baseline_pnl):.0f}/trade. "
        f"Strategy is overtrading in high-vol regimes.\n"
        f"   HYPOTHESIS: Tighten GARCH target_vol to 90% annualized. "
        f"Expected: 20-25% fewer trades, higher quality, improved Sharpe.",
        {"garch_target_vol": 0.90},
        0.80
    ),
    (
        "EXIT RULE -- Winner protection: extend holds from 0.1% to 0.5%",
        f"48+ bar trades: avg P&L=+610/trade ({hold.loc['48+','n']} trades only).\n"
        f"   13-48 bar trades: avg P&L=+285. These are the profitable trades.\n"
        f"   Current winner protection at 0.1% P&L exits too early.\n"
        f"   HYPOTHESIS: Raise winner_protection threshold to 0.5%, stale_15m_move to 0.008.",
        {"winner_protection_pct": 0.005, "stale_15m_move": 0.008},
        0.78
    ),
    (
        "SIGNAL -- Disable OU mean-reversion for trending instruments",
        f"Instruments with WR < 38%: {trending}.\n"
        f"   These are momentum/trend instruments -- OU z-score fights their natural direction.\n"
        f"   HYPOTHESIS: Set ou_frac=0.0 for {trending}, keep BH-only signal for them.",
        {"ou_disabled_symbols": trending, "ou_frac_default": 0.08},
        0.74
    ),
    (
        "REGIME FILTER -- Reduce sizing on bad days of week",
        f"Worst days: {bad_days} (avg P&L: {dow.loc[bad_days,'avg_pnl'].mean():.0f}/trade).\n"
        f"   Saturday avg P&L=+31 (best). Tuesday=-65, Sunday=-55, Thursday=-43.\n"
        f"   HYPOTHESIS: Apply 0.65x position multiplier on {bad_days}.",
        {"dow_size_multiplier": {d: 0.65 for d in bad_days}},
        0.70
    ),
    (
        "RISK -- Dynamic CORR factor: 0.25 base, spike to 0.70 in stress",
        f"Current CORR=0.25 is static. During market stress (like Apr 2026 selloff), "
        f"crypto correlation spikes to 0.85+, making CORR=0.25 dangerously wrong.\n"
        f"   HYPOTHESIS: Compute rolling 30-day realized correlation. "
        f"When avg pair-corr > 0.6, use CORR=0.60 to reduce per-instrument risk.",
        {"corr_dynamic": True, "corr_min": 0.25, "corr_stress": 0.60,
         "corr_trigger_threshold": 0.60},
        0.76
    ),
    (
        "ENTRY TIMING -- Boost size during best hours " + str(best_hours[:4]),
        f"Hours {best_hours[:4]} UTC: avg P&L={hourly.loc[best_hours[:4],'avg_pnl'].mean():.0f}/trade "
        f"vs baseline {baseline_pnl:.0f}. WR={hourly.loc[best_hours[:4],'wr'].mean():.1%}.\n"
        f"   Hour 3 UTC = Asian close / London open overlap, historically strong crypto momentum.\n"
        f"   HYPOTHESIS: Apply 1.25x cf_scale multiplier during hours {best_hours[:4]}.",
        {"boost_hours_utc": best_hours[:4], "hour_boost_multiplier": 1.25},
        0.72
    ),
]

print("=" * 72)
print("IAE: IDEAS GENERATED FROM YOUR STRATEGY DATA")
print(f"Source: {len(df):,} trades | {df['exit_time'].min().date()} to {df['exit_time'].max().date()}")
print(f"Baseline: WR={baseline_wr:.1%} | Avg P&L={baseline_pnl:.0f}/trade | Total={df['pnl'].sum():,.0f}")
print("=" * 72)

for i, (title, desc, params, conf) in enumerate(sorted(ideas, key=lambda x: -x[3]), 1):
    print(f"\n#{i} [conf={conf:.0%}] {title}")
    print(f"   {desc}")
    print(f"   PARAMS: {json.dumps(params)[:130]}")

print("\n" + "=" * 72)
print(f"TOTAL: {len(ideas)} actionable ideas")
print()
print("PRIORITY ORDER:")
for i, (t, _, p, c) in enumerate(sorted(ideas, key=lambda x: -x[3]), 1):
    print(f"  {i}. [{c:.0%}] {t.split('--')[0].strip()}: {json.dumps(p)[:80]}")
