"""
Monte Carlo: v12 baseline vs v15 (TAIL=$22M, TREND=$2M, ARB=$1M)
Run against all available trade datasets.
10,000 simulations per dataset per strategy config.
"""
import csv, random, statistics, os
from pathlib import Path

DOWNLOADS = Path(r"C:\Users\Matthew\Downloads")
TRADE_FILES = [f for f in DOWNLOADS.glob("*_trades.csv") if f.stat().st_size > 10_000]

STARTING_EQUITY = 1_000_000
N_SIMS          = 10_000
BLOWUP_THRESH   = 0.10   # <10% of starting = blown up

# ── gear configs ──────────────────────────────────────────────────────────────
CONFIGS = {
    "v12_baseline": dict(tail_cap=3_000_000, trend_cap=0,         arb_cap=0),
    "v15_22M":      dict(tail_cap=22_000_000, trend_cap=2_000_000, arb_cap=1_000_000),
}

def load_trades(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    pnls = []
    for r in rows:
        try:
            pnls.append(float(r["P&L"]))
        except (KeyError, ValueError):
            pass
    return pnls

def simulate(pnls, tail_cap, trend_cap, arb_cap, n=N_SIMS, start=STARTING_EQUITY):
    """
    Simplified gear model:
    - Gear 1 (tail): scaled by min(tail_cap, equity)/tail_cap
    - Gear 2/3/4: contribute bonus trades proportional to allocation above tail_cap
      We model this as: when equity > tail_cap, replay extra PnL draws scaled by
      (trend+arb+harvest) fraction. Harvest = equity above all caps.
    - Base trades come from resampling actual PnL list with equity-scaled sizing.
    """
    results = []
    blowups = 0
    finals  = []

    for _ in range(n):
        eq = float(start)
        trade_pool = pnls[:]
        random.shuffle(trade_pool)
        idx = 0

        for trade_pnl in trade_pool:
            if eq <= start * BLOWUP_THRESH:
                blowups += 1
                eq = 0
                break

            # Gear 1 scale
            g1_scale = min(tail_cap, eq) / tail_cap if tail_cap > 0 else 0

            # Gear 2/3 bonus: proportional to trend+arb allocation
            trend_alloc   = min(trend_cap, max(0.0, eq - tail_cap))
            arb_alloc     = min(arb_cap,   max(0.0, eq - tail_cap - trend_cap))
            harvest_alloc = max(0.0, eq - tail_cap - trend_cap - arb_cap)
            bonus_alloc   = trend_alloc + arb_alloc + harvest_alloc

            bonus_scale = bonus_alloc / tail_cap if tail_cap > 0 else 0

            # Resample a bonus trade for the gear overlay (mean-reversion tends to reduce vol)
            # Model: bonus trades have 60% win rate, avg win = 0.5x tail avg, avg loss = 0.3x tail avg
            # (conservative — harvest is lower vol than tail)
            bonus_pnl = 0.0
            if bonus_alloc > 0:
                if random.random() < 0.60:
                    bonus_pnl = abs(trade_pnl) * 0.5 * bonus_scale
                else:
                    bonus_pnl = -abs(trade_pnl) * 0.3 * bonus_scale

            scaled_pnl = trade_pnl * g1_scale + bonus_pnl
            eq += scaled_pnl

        finals.append(max(0, eq))

    blowup_rate = sum(1 for f in finals if f < start * BLOWUP_THRESH) / n
    valid = [f for f in finals if f >= start * BLOWUP_THRESH]
    median_eq  = statistics.median(finals)
    mean_eq    = statistics.mean(finals)
    p10        = sorted(finals)[int(n * 0.10)]
    p90        = sorted(finals)[int(n * 0.90)]

    # Sharpe approximation: mean/std of log-returns across paths
    log_rets = [f / start for f in finals if f > 0]
    sharpe_approx = (mean_eq / start - 1) / (statistics.stdev([f/start for f in finals]) + 1e-9)

    return dict(
        blowup_rate=blowup_rate,
        median=median_eq,
        mean=mean_eq,
        p10=p10,
        p90=p90,
        sharpe=sharpe_approx,
        n_trades=len(pnls),
    )

# ── run ───────────────────────────────────────────────────────────────────────
print(f"{'Dataset':<45} {'Trades':>7}  {'Config':<14}  {'Blowup':>7}  {'Median':>12}  {'Mean':>12}  {'Sharpe':>7}")
print("-" * 115)

summary = {}  # config -> list of results

for cfg_name in CONFIGS:
    summary[cfg_name] = []

for fpath in sorted(TRADE_FILES):
    pnls = load_trades(fpath)
    if len(pnls) < 20:
        continue
    name = fpath.stem.replace("_trades", "")
    for cfg_name, cfg in CONFIGS.items():
        r = simulate(pnls, **cfg)
        summary[cfg_name].append(r)
        print(f"{name:<45} {r['n_trades']:>7}  {cfg_name:<14}  {r['blowup_rate']:>6.1%}  "
              f"${r['median']:>11,.0f}  ${r['mean']:>11,.0f}  {r['sharpe']:>7.2f}")
    print()

# ── aggregate ─────────────────────────────────────────────────────────────────
print("=" * 115)
print("AGGREGATE (mean across all datasets)")
print(f"{'Config':<14}  {'Avg Blowup':>10}  {'Avg Median':>12}  {'Avg Mean':>12}  {'Avg Sharpe':>10}")
print("-" * 65)
for cfg_name, results in summary.items():
    if not results:
        continue
    avg_blowup  = statistics.mean([r["blowup_rate"] for r in results])
    avg_median  = statistics.mean([r["median"]      for r in results])
    avg_mean    = statistics.mean([r["mean"]         for r in results])
    avg_sharpe  = statistics.mean([r["sharpe"]       for r in results])
    print(f"{cfg_name:<14}  {avg_blowup:>10.1%}  ${avg_median:>11,.0f}  ${avg_mean:>11,.0f}  {avg_sharpe:>10.2f}")
