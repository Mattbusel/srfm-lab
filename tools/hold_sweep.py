"""
hold_sweep.py — Compare Sharpe across minimum-hold durations: 1x, 10x, 100x, 1000x

The baseline MIN_HOLD in local_backtest.py only blocks sign-reversals.
This sweep enforces a "minimum hold before ANY exit (including to zero)",
which is what forces the strategy to hold trades longer.

Baseline hold unit = 4 bars (same as local_backtest.py MIN_HOLD).
Multipliers: 1x=4, 10x=40, 100x=400, 1000x=4000 bars.

Usage:
    python tools/hold_sweep.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS = Path(__file__).parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import local_backtest as lb

STARTING_EQUITY   = lb.STARTING_EQUITY
BASELINE          = 4
MULTIPLIERS       = [1, 10, 100, 1000]


# ── Patched run_backtest that enforces min_hold_any ───────────────────────────
def run_backtest_with_hold(daily, hourly, intra15m, min_hold_any: int):
    """
    Same as lb.run_backtest but adds a hard floor:
    once a position is entered, no exit/resize is allowed for min_hold_any bars.
    """
    syms = list(lb.INSTRUMENTS.keys())

    d_atr   = {s: lb.atr_s(daily[s]["High"], daily[s]["Low"], daily[s]["Close"]) for s in syms}
    d_scale = {s: lb.bull_scale(daily[s]["Close"]) for s in syms}
    h_atr   = {}
    for s in syms:
        h = hourly[s]
        if not h.empty:
            h_atr[s] = lb.atr_s(h["High"], h["Low"], h["Close"])

    if intra15m is None:
        intra15m = {s: pd.DataFrame() for s in syms}

    d_bh   = {s: lb.BHState(lb.INSTRUMENTS[s]["cf_1d"],  lb.INSTRUMENTS[s].get("bh_form", 1.5)) for s in syms}
    h_bh   = {s: lb.BHState(lb.INSTRUMENTS[s]["cf_1h"],  lb.INSTRUMENTS[s].get("bh_form", 1.5)) for s in syms}
    m15_bh = {s: lb.BHState(lb.INSTRUMENTS[s].get("cf_15m", lb.INSTRUMENTS[s]["cf_1h"] * 0.8),
                             lb.INSTRUMENTS[s].get("bh_form", 1.5)) for s in syms}

    m15_dates    = {s: set(intra15m[s].index.date) for s in syms if not intra15m[s].empty}
    all_days     = daily["ES"].index[daily["ES"].index >= pd.Timestamp(lb.START_DATE)]
    hourly_dates = {s: set(hourly[s].index.date) for s in syms if not hourly[s].empty}

    dollar_pos  = {s: 0.0 for s in syms}
    entry_price = {s: None for s in syms}
    last_frac   = {s: 0.0 for s in syms}
    bars_held   = {s: 0 for s in syms}
    pos_floor   = {s: 0.0 for s in syms}

    equity       = STARTING_EQUITY
    peak         = STARTING_EQUITY
    equity_curve = []
    trades       = []

    WARMUP = 200

    for day_idx, day in enumerate(all_days):
        day_date = day.date()

        for s in syms:
            if day not in daily[s].index:
                continue
            d_bh[s].cf_scale = float(d_scale[s].get(day, 1.0))
            d_bh[s].update(daily[s].loc[day, "Close"])

        if day_idx < WARMUP:
            equity_curve.append((day_date, equity))
            continue

        use_hourly = any(day_date in hourly_dates.get(s, set()) for s in syms)
        use_15m    = any(day_date in m15_dates.get(s, set()) for s in syms)

        if use_hourly:
            h_bars = {s: hourly[s][hourly[s].index.date == day_date]
                      for s in syms if not hourly[s].empty}
            ref = max(h_bars, key=lambda s: len(h_bars[s])) if h_bars else "ES"
            bar_times = h_bars[ref].index if ref in h_bars and not h_bars[ref].empty else [day]
        else:
            bar_times = [day]

        if use_15m:
            m15_bars = {s: intra15m[s][intra15m[s].index.date == day_date] for s in syms}

        for bar_time in bar_times:
            if use_15m:
                for s in syms:
                    if s not in m15_bars:
                        continue
                    if use_hourly:
                        mb = m15_bars[s][(m15_bars[s].index >= bar_time) &
                                         (m15_bars[s].index < bar_time + pd.Timedelta(hours=1))]
                    else:
                        mb = m15_bars[s]
                    for _, row in mb.iterrows():
                        m15_bh[s].cf_scale = float(d_scale[s].get(day, 1.0))
                        m15_bh[s].update(row["Close"])

            if use_hourly:
                for s in syms:
                    if s not in h_bars:
                        continue
                    b = h_bars[s]
                    if bar_time not in b.index:
                        continue
                    h_bh[s].cf_scale = float(d_scale[s].get(day, 1.0))
                    h_bh[s].update(b.loc[bar_time, "Close"])

            curr_price = {}
            for s in syms:
                if use_hourly and s in h_bars and bar_time in h_bars[s].index:
                    curr_price[s] = float(h_bars[s].loc[bar_time, "Close"])
                elif day in daily[s].index:
                    curr_price[s] = float(daily[s].loc[day, "Close"])
                else:
                    curr_price[s] = entry_price[s] if entry_price[s] else 1.0

            mtm_pnl = 0.0
            for s in syms:
                if dollar_pos[s] != 0.0 and entry_price[s] and entry_price[s] > 0:
                    mtm_pnl += dollar_pos[s] * (curr_price[s] - entry_price[s]) / entry_price[s]

            equity_live = equity + mtm_pnl
            if equity_live > peak:
                peak = equity_live

            tail_frac   = min(lb.TAIL_FIXED_CAPITAL, equity_live) / equity_live
            raw_targets = {}

            for s in syms:
                tf = (4 if d_bh[s].active else 0) + \
                     (2 if (use_hourly and h_bh[s].active) else 0) + \
                     (1 if (use_15m and m15_bh[s].active) else 0)
                ceiling = lb.TF_CAP.get(tf, 0.0)

                if tf == 2 and abs(last_frac[s]) < 0.02:
                    ceiling = 0.0

                if ceiling == 0.0:
                    raw_targets[s] = 0.0
                    continue

                direction = 0
                if d_bh[s].active and d_bh[s].bh_dir != 0:
                    direction = d_bh[s].bh_dir
                elif use_hourly and h_bh[s].active and h_bh[s].bh_dir != 0:
                    direction = h_bh[s].bh_dir
                elif d_bh[s].active and len(d_bh[s].prices) >= 5:
                    direction = 1 if d_bh[s].prices[-1] > d_bh[s].prices[-5] else -1

                if direction == 0:
                    raw_targets[s] = 0.0
                    continue

                if use_hourly and s in h_atr and bar_time in h_atr[s].index:
                    vol_pct = float(h_atr[s].loc[bar_time]) / (curr_price[s] + 1e-9) * math.sqrt(6.5)
                elif day in d_atr[s].index:
                    vol_pct = float(d_atr[s].loc[day]) / (curr_price[s] + 1e-9)
                else:
                    vol_pct = 0.01

                cap = min(lb.PER_INST_RISK / (vol_pct + 1e-9), ceiling)
                raw_targets[s] = cap * direction

            for s in syms:
                tgt = raw_targets.get(s, 0.0)
                tf  = (4 if d_bh[s].active else 0) + \
                      (2 if (use_hourly and h_bh[s].active) else 0) + \
                      (1 if (use_15m and m15_bh[s].active) else 0)
                if tf >= 6 and abs(tgt) > 0.15 and not np.isclose(tgt, 0.0) and h_bh[s].ctl >= 5:
                    pos_floor[s] = max(pos_floor[s], 0.70 * abs(tgt))
                if pos_floor[s] > 0.0 and tf >= 4 and not np.isclose(last_frac[s], 0.0):
                    raw_targets[s] = float(np.sign(last_frac[s]) * max(abs(tgt), pos_floor[s]))
                    pos_floor[s] *= 0.95
                if tf < 4 or np.isclose(tgt, 0.0):
                    pos_floor[s] = 0.0
                if not d_bh[s].active and not (use_hourly and h_bh[s].active):
                    pos_floor[s] = 0.0

            total_exp = sum(abs(v) for v in raw_targets.values())
            scale     = 1.0 / total_exp if total_exp > 1.0 else 1.0

            for s in syms:
                tgt   = raw_targets.get(s, 0.0)
                final = tgt * scale * tail_frac

                # ── MIN HOLD ANY: block ALL exits/resizes until bars_held >= min_hold_any ──
                if not np.isclose(last_frac[s], 0.0) and bars_held[s] < min_hold_any:
                    final = last_frac[s]   # lock position — no exit, no resize, no reversal

                if abs(final - last_frac[s]) > 0.02:
                    if dollar_pos[s] != 0.0 and entry_price[s] is not None:
                        ret  = (curr_price[s] - entry_price[s]) / entry_price[s]
                        pnl  = dollar_pos[s] * ret
                        equity += pnl
                        trades.append({
                            "exit_time":   bar_time if use_hourly else day_date,
                            "sym":         s,
                            "direction":   "Long" if dollar_pos[s] > 0 else "Short",
                            "entry_price": entry_price[s],
                            "exit_price":  curr_price[s],
                            "dollar_pos":  dollar_pos[s],
                            "pnl":         pnl,
                        })

                    if np.isclose(final, 0.0):
                        dollar_pos[s]  = 0.0
                        entry_price[s] = None
                        bars_held[s]   = 0
                    else:
                        dollar_pos[s]  = final * equity
                        entry_price[s] = curr_price[s]
                        if np.sign(final) != np.sign(last_frac[s]):
                            bars_held[s] = 0

                    last_frac[s] = final

                if abs(last_frac[s]) > 0.02:
                    bars_held[s] += 1

        eod_pnl = sum(
            dollar_pos[s] * (curr_price[s] - entry_price[s]) / entry_price[s]
            for s in syms
            if dollar_pos[s] != 0.0 and entry_price[s] and entry_price[s] > 0
        )
        equity_curve.append((day_date, max(0.0, equity + eod_pnl)))

    return equity_curve, trades, peak


def compute_stats(equity_curve, trades):
    dates  = [e[0] for e in equity_curve]
    values = np.array([e[1] for e in equity_curve])
    final  = values[-1]
    years  = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    cagr   = (final / STARTING_EQUITY) ** (1 / years) - 1 if years > 0 and final > 0 else 0
    pk     = np.maximum.accumulate(values)
    max_dd = ((values - pk) / pk).min()
    rets   = pd.Series(values).pct_change().dropna()
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * math.sqrt(252)
    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / (len(trades) + 1e-9)
    pf       = sum(wins) / (abs(sum(losses)) + 1e-9) if losses else float("inf")
    return dict(
        sharpe=round(sharpe, 3),
        cagr=round(cagr * 100, 2),
        max_dd=round(max_dd * 100, 2),
        trades=len(trades),
        win_rate=round(win_rate * 100, 1),
        profit_fac=round(pf, 2),
        final_equity=round(final, 0),
    )


# ── Download once ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  Downloading market data (once)...")
print("=" * 60)
daily, hourly, intra15m = lb.download_data()
print()

# ── Run sweep ─────────────────────────────────────────────────────────────────
results = []
for mult in MULTIPLIERS:
    hold = BASELINE * mult
    print(f"  Running min_hold_any = {hold:>5}  ({mult:>4}x)...", flush=True)
    ec, trades, peak = run_backtest_with_hold(daily, hourly, intra15m, hold)
    stats = compute_stats(ec, trades)
    stats["multiplier"] = mult
    stats["min_hold"]   = hold
    results.append(stats)
    print(f"    Sharpe {stats['sharpe']:.3f}  CAGR {stats['cagr']:.1f}%  "
          f"MaxDD {stats['max_dd']:.1f}%  Trades {stats['trades']}")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 82)
print("  HOLD SWEEP RESULTS  (min_hold_any: blocks ALL exits until N bars held)")
print("=" * 82)
print(f"  {'Mult':>6}  {'MinHold':>7}  {'Sharpe':>7}  {'CAGR%':>7}  "
      f"{'MaxDD%':>7}  {'Trades':>7}  {'WinRate%':>9}  {'ProfFac':>8}")
print("  " + "-" * 77)
for r in results:
    print(
        f"  {r['multiplier']:>5}x  {r['min_hold']:>7}  {r['sharpe']:>7.3f}  "
        f"{r['cagr']:>6.1f}%  {r['max_dd']:>6.1f}%  {r['trades']:>7}  "
        f"{r['win_rate']:>8.1f}%  {r['profit_fac']:>8.2f}"
    )
print("=" * 82)

best = max(results, key=lambda r: r["sharpe"])
print(f"\n  Best Sharpe: {best['sharpe']} at {best['multiplier']}x  (min_hold={best['min_hold']} bars)")

out = Path(__file__).parent / "backtest_output" / "hold_sweep.csv"
out.parent.mkdir(exist_ok=True)
pd.DataFrame(results).to_csv(out, index=False)
print(f"  Results saved: {out}")
