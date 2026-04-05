"""
Full IAE analysis pipeline runner.
Runs: macro regime, on-chain, alternative data, sentiment (fear/greed),
then feeds all findings into the IAE idea miner and prints a unified report.

Usage: python run_full_analysis.py
"""
import sys, warnings, pathlib, types, importlib.util
warnings.filterwarnings('ignore')

IAE = pathlib.Path('idea-engine')

# ── package loader for hyphenated directories ─────────────────────────────────
def load_pkg(dir_path: str, pkg_name: str):
    """Register a directory (possibly hyphenated) as a Python package."""
    parts = pkg_name.split('.')
    # ensure parent packages exist
    for i in range(1, len(parts)):
        parent = '.'.join(parts[:i])
        if parent not in sys.modules:
            m = types.ModuleType(parent)
            m.__path__ = []
            m.__package__ = parent
            sys.modules[parent] = m
    p = pathlib.Path(dir_path).resolve()
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(p)]
    pkg.__package__ = pkg_name
    pkg.__spec__ = importlib.util.spec_from_file_location(pkg_name, str(p / '__init__.py'))
    sys.modules[pkg_name] = pkg
    return pkg

sys.path.insert(0, str(IAE.resolve()))

# Register all hyphenated packages
load_pkg('idea-engine/macro-factor',          'macro_factor')
load_pkg('idea-engine/macro-factor/factors',  'macro_factor.factors')
load_pkg('idea-engine/onchain',               'onchain')
load_pkg('idea-engine/onchain/metrics',       'onchain.metrics')
load_pkg('idea-engine/alternative-data',      'alternative_data')
load_pkg('idea-engine/sentiment-engine',      'sentiment_engine')
load_pkg('idea-engine/sentiment-engine/scrapers', 'sentiment_engine.scrapers')
load_pkg('idea-engine/sentiment-engine/nlp',  'sentiment_engine.nlp')

print("=" * 72)
print("FULL IAE ANALYSIS PIPELINE")
print("=" * 72)

# ── 1. MACRO REGIME ───────────────────────────────────────────────────────────
print("\n[1/5] MACRO REGIME")
print("-" * 40)
try:
    from macro_factor.factors.vix  import compute_vix,  vix_summary
    from macro_factor.factors.dxy  import compute_dxy,  dxy_summary
    from macro_factor.factors.equity_momentum import compute_equity_momentum
    from macro_factor.factors.rates import compute_rates
    from macro_factor.regime_classifier import RegimeClassifier, MacroRegime

    vix_r  = compute_vix()
    dxy_r  = compute_dxy()
    eq_r   = compute_equity_momentum()
    rates_r = compute_rates()

    print(f"  VIX:            {vix_r.vix_current:.1f}  ({vix_r.vix_regime})  signal={vix_r.signal:+.2f}")
    print(f"  DXY:            {dxy_r.dxy_level:.1f}  signal={dxy_r.risk_off_score:+.2f}")
    print(f"  Yield curve:    {rates_r.curve_regime}  slope={rates_r.yield_curve_slope_pct:+.3f}")
    print(f"  SPY 20d mom:    {eq_r.spy_momentum_20d:+.2%}  MA200={eq_r.spy_200d_crossover}")

    clf = RegimeClassifier()
    regime = clf.classify()
    print(f"\n  >> MACRO REGIME:  {regime.regime.value}  (confidence={regime.confidence:.0%})")
    print(f"     Size multiplier: {regime.position_multiplier:.2f}x")
    if regime.crisis_override:
        print(f"     *** CRISIS OVERRIDE ACTIVE ***")
except Exception as e:
    print(f"  ERROR: {e}")
    regime = None

# ── 2. ON-CHAIN ANALYTICS ─────────────────────────────────────────────────────
print("\n[2/5] ON-CHAIN ANALYTICS (BTC)")
print("-" * 40)
try:
    from onchain.composite_signal import OnChainEngine
    engine = OnChainEngine()
    oc = engine.run('BTC')
    print(f"  Composite score:    {oc.composite_score:+.3f}  (range -1 to +1)")
    print(f"  MVRV Z-score:       {oc.mvrv.mvrv_zscore:+.2f}  signal={oc.mvrv.signal:+.2f}")
    print(f"  SOPR:               {oc.sopr.sopr_smooth:.3f}  capitulation={oc.sopr.is_capitulation}")
    print(f"  Exchange reserves:  7d={oc.exchange_reserves.roc_7d:+.2%}  30d={oc.exchange_reserves.roc_30d:+.2%}")
    print(f"  Whale regime:       {oc.whale.dominant_regime}")
    print(f"  HODL wave regime:   {oc.hodl_waves.regime}  STH={oc.hodl_waves.sth_pct:.1%}")
    print(f"  Hash rate ribbon:   {oc.hash_rate.ribbon_crossover}  capitulation={oc.hash_rate.is_capitulation}")
    onchain_signal = oc.composite_score
except Exception as e:
    print(f"  ERROR: {e}")
    onchain_signal = 0.0

# ── 3. ALTERNATIVE DATA ───────────────────────────────────────────────────────
print("\n[3/5] ALTERNATIVE DATA (DERIVATIVES)")
print("-" * 40)
try:
    from alternative_data.futures_oi   import FuturesOIFetcher
    from alternative_data.funding_rates import FundingRateFetcher
    from alternative_data.liquidations  import LiquidationSimulator

    oi = FuturesOIFetcher()
    oi_results = oi.fetch_all()
    btc_oi = next((r for r in oi_results if r.symbol == 'BTC'), None)
    if btc_oi:
        print(f"  BTC OI regime:      {btc_oi.regime}  oi={btc_oi.open_interest_usd/1e9:.2f}B")

    fr = FundingRateFetcher()
    fr_results = fr.fetch_all()
    btc_fr = next((r for r in fr_results if r.symbol == 'BTC'), None)
    if btc_fr:
        print(f"  BTC funding:        {btc_fr.current_rate:+.4%}  signal={btc_fr.signal}")

    liq = LiquidationSimulator()
    liq_results = liq.fetch_all()
    btc_liq = next((r for r in liq_results if r.symbol == 'BTC'), None)
    if btc_liq:
        print(f"  BTC liquidations:   cascade={btc_liq.is_cascade}  Z={btc_liq.z_score:.2f}  signal={btc_liq.signal_type}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 4. FEAR & GREED ──────────────────────────────────────────────────────────
print("\n[4/5] FEAR & GREED INDEX")
print("-" * 40)
try:
    from sentiment_engine.scrapers.fear_greed import FearGreedClient
    fg = FearGreedClient()
    current = fg.get_current()
    hist    = fg.get_history(days=7)
    print(f"  Current:  {current.value}  ({current.label})")
    if hist and hist.readings:
        vals = [r.value for r in hist.readings]
        trend = (vals[-1] - vals[0]) / max(len(vals)-1, 1) if len(vals) > 1 else 0
        print(f"  7d trend: {trend:+.1f} points/day  (7d avg={sum(vals)/len(vals):.0f})")
    fg_value = current.value
except Exception as e:
    print(f"  ERROR: {e}")
    fg_value = 50

# ── 5. IAE IDEA MINER (on actual trade data) ─────────────────────────────────
print("\n[5/5] IAE IDEA MINER (63K trades)")
print("-" * 40)
import pandas as pd, numpy as np, json

try:
    from ingestion.loaders.backtest_loader import load_backtest
    bt = load_backtest()
    df = bt.trades_df.copy()
except Exception:
    try:
        df = pd.read_csv('tools/backtest_output/crypto_trades.csv')
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
    except Exception as e:
        print(f"  Could not load trades: {e}")
        df = None

if df is not None:
    df['hour']       = df['exit_time'].dt.hour
    df['dow']        = df['exit_time'].dt.day_name()
    df['is_win']     = (df['pnl'] > 0).astype(int)
    df['hold_bucket'] = pd.cut(df.get('hold_bars', pd.Series([5]*len(df))),
                                bins=[0,1,4,12,48,9999],
                                labels=['1bar','2-4','5-12','13-48','48+'])

    baseline_wr  = df['is_win'].mean()
    baseline_pnl = df['pnl'].mean()
    total_pnl    = df['pnl'].sum()

    print(f"  Trades: {len(df):,}  WR={baseline_wr:.1%}  avg P&L={baseline_pnl:+.0f}  total={total_pnl:+,.0f}")

    # Hourly breakdown
    hourly = df.groupby('hour').agg(avg_pnl=('pnl','mean'), wr=('is_win','mean'), n=('pnl','count'))
    worst_hours = hourly[hourly['avg_pnl'] < baseline_pnl - 50].index.tolist()
    best_hours  = hourly[hourly['avg_pnl'] > baseline_pnl + 80].index.tolist()

    # Symbol breakdown
    if 'symbol' in df.columns:
        sym = df.groupby('symbol').agg(total=('pnl','sum'), n=('pnl','count'), wr=('is_win','mean'))
        losers  = sym[sym['total'] < -30000].sort_values('total').head(5)
        winners = sym[sym['total'] > 20000].sort_values('total', ascending=False).head(5)
        print(f"\n  TOP LOSERS:")
        for s, row in losers.iterrows():
            print(f"    {s:8s}  P&L={row['total']:>10,.0f}  WR={row['wr']:.1%}  n={row['n']:,}")
        print(f"\n  TOP WINNERS:")
        for s, row in winners.iterrows():
            print(f"    {s:8s}  P&L={row['total']:>10,.0f}  WR={row['wr']:.1%}  n={row['n']:,}")

    print(f"\n  WORST HOURS: {worst_hours}  avg={hourly.loc[worst_hours,'avg_pnl'].mean():.0f}/trade" if worst_hours else "")
    print(f"  BEST HOURS:  {best_hours}  avg={hourly.loc[best_hours,'avg_pnl'].mean():.0f}/trade" if best_hours else "")

# ── UNIFIED SIGNAL SUMMARY ────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("UNIFIED SIGNAL SUMMARY")
print("=" * 72)

try:
    regime_name = regime.regime.value if regime else "UNKNOWN"
    size_mult   = regime.position_multiplier if regime else 1.0
    print(f"  Macro regime:     {regime_name}  ->  {size_mult:.2f}x sizing")
except:
    print(f"  Macro regime:     UNKNOWN")

print(f"  On-chain BTC:     score={onchain_signal:+.3f}  ({'BULLISH' if onchain_signal>0.3 else 'BEARISH' if onchain_signal<-0.3 else 'NEUTRAL'})")
print(f"  Fear & Greed:     {fg_value}  ({'EXTREME FEAR' if fg_value<25 else 'FEAR' if fg_value<45 else 'GREED' if fg_value>55 else 'NEUTRAL'})")

# Combined recommendation
try:
    combined = (onchain_signal * 0.4 + (fg_value - 50) / 100 * 0.2)
    print(f"\n  Combined alpha signal: {combined:+.3f}")
    if size_mult < 0.5:
        print("  ** MACRO OVERRIDE: reduce all positions to 25% normal size **")
    elif size_mult < 0.8:
        print("  ** MACRO CAUTION: reduce all positions to 60% normal size **")
    elif combined > 0.2 and size_mult >= 1.0:
        print("  ** MACRO + ON-CHAIN ALIGNED BULLISH: ok to run full size **")
    else:
        print("  ** NEUTRAL: run normal sizing **")
except:
    pass

print()
