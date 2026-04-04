"""
Crypto BH Backtest + Monte Carlo
=================================
Downloads BTC/ETH/SOL full history from Alpaca (2021-present).
Runs the 3-timeframe BH engine (same physics as LARSA v16).
Then runs 10,000-sim Monte Carlo on the resulting trade list.

Long-only (mirrors live trader Alpaca constraint).
Run:  python tools/crypto_backtest_mc.py
"""
import math
import random
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ── Config ────────────────────────────────────────────────────────────────────
# CF tiers: large / mid / small cap volatility
INSTRUMENTS = {
    "BTC":   {"ticker": "BTC/USD",   "cf_4h": 0.008, "cf_15m": 0.005, "cf_1h": 0.015, "cf_1d": 0.05},
    "ETH":   {"ticker": "ETH/USD",   "cf_4h": 0.012, "cf_15m": 0.007, "cf_1h": 0.020, "cf_1d": 0.07},
    "SOL":   {"ticker": "SOL/USD",   "cf_4h": 0.018, "cf_15m": 0.010, "cf_1h": 0.030, "cf_1d": 0.10},
    "XRP":   {"ticker": "XRP/USD",   "cf_4h": 0.018, "cf_15m": 0.010, "cf_1h": 0.030, "cf_1d": 0.10},
    "AVAX":  {"ticker": "AVAX/USD",  "cf_4h": 0.020, "cf_15m": 0.012, "cf_1h": 0.035, "cf_1d": 0.12},
    "LINK":  {"ticker": "LINK/USD",  "cf_4h": 0.020, "cf_15m": 0.012, "cf_1h": 0.035, "cf_1d": 0.12},
    "DOT":   {"ticker": "DOT/USD",   "cf_4h": 0.020, "cf_15m": 0.012, "cf_1h": 0.035, "cf_1d": 0.12},
    "UNI":   {"ticker": "UNI/USD",   "cf_4h": 0.022, "cf_15m": 0.015, "cf_1h": 0.045, "cf_1d": 0.15},
    "AAVE":  {"ticker": "AAVE/USD",  "cf_4h": 0.022, "cf_15m": 0.015, "cf_1h": 0.045, "cf_1d": 0.15},
    "LTC":   {"ticker": "LTC/USD",   "cf_4h": 0.018, "cf_15m": 0.010, "cf_1h": 0.030, "cf_1d": 0.10},
    "BCH":   {"ticker": "BCH/USD",   "cf_4h": 0.020, "cf_15m": 0.012, "cf_1h": 0.035, "cf_1d": 0.12},
    # ADA: only ~51 daily bars on Alpaca — insufficient history, excluded
    # "ADA":   {"ticker": "ADA/USD",   "cf_4h": 0.022, "cf_15m": 0.015, "cf_1h": 0.045, "cf_1d": 0.15},
    "DOGE":  {"ticker": "DOGE/USD",  "cf_4h": 0.030, "cf_15m": 0.020, "cf_1h": 0.060, "cf_1d": 0.20},
    "SHIB":  {"ticker": "SHIB/USD",  "cf_4h": 0.035, "cf_15m": 0.025, "cf_1h": 0.075, "cf_1d": 0.25},
    "GRT":   {"ticker": "GRT/USD",   "cf_4h": 0.030, "cf_15m": 0.020, "cf_1h": 0.060, "cf_1d": 0.20},
    "BAT":   {"ticker": "BAT/USD",   "cf_4h": 0.030, "cf_15m": 0.020, "cf_1h": 0.060, "cf_1d": 0.20},
    "CRV":   {"ticker": "CRV/USD",   "cf_4h": 0.030, "cf_15m": 0.020, "cf_1h": 0.060, "cf_1d": 0.20},
    "SUSHI": {"ticker": "SUSHI/USD", "cf_4h": 0.030, "cf_15m": 0.020, "cf_1h": 0.060, "cf_1d": 0.20},
    # FIL: only ~48 daily bars on Alpaca — insufficient history, excluded
    # "FIL":   {"ticker": "FIL/USD",   "cf_4h": 0.025, "cf_15m": 0.018, "cf_1h": 0.055, "cf_1d": 0.18},
    "MKR":   {"ticker": "MKR/USD",   "cf_4h": 0.022, "cf_15m": 0.015, "cf_1h": 0.045, "cf_1d": 0.15},
    "YFI":   {"ticker": "YFI/USD",   "cf_4h": 0.022, "cf_15m": 0.015, "cf_1h": 0.045, "cf_1d": 0.15},
}

BH_FORM     = 1.2   # lower threshold — fires faster
BH_CTL_MIN  = 1     # activate after just 1 consecutive timelike bar

START_DATE         = datetime(2021, 1, 1, tzinfo=timezone.utc)
STARTING_EQUITY    = 1_000_000.0
TAIL_FIXED_CAPITAL = 1_000_000.0  # no de-risk
DAILY_RISK         = 0.05         # 5x aggression
N_INST             = 20
CORR               = 0.65
CORR_FACTOR        = math.sqrt(N_INST + N_INST * (N_INST - 1) * CORR)
PER_INST_RISK      = DAILY_RISK / CORR_FACTOR

TF_CAP      = {7: 1.0, 6: 1.0, 4: 0.60, 3: 0.50, 2: 0.40, 1: 0.20, 0: 0.0}
CRYPTO_CAP  = 0.40   # 40% per coin
MIN_HOLD      = 1
BH_DECAY      = 0.95
BH_COLLAPSE   = 0.8
WARMUP_DAYS   = 20
STALE_15M_MOVE = 0.001  # full volatility mode: cut losing positions moving < 0.1% per 15m

MC_SIMS     = 10_000
MC_MONTHS   = 12
MC_PROJECT  = 12    # forward projection window in months

OUT_DIR = Path(__file__).parent / "backtest_output"
OUT_DIR.mkdir(exist_ok=True)

# ── BH Physics ────────────────────────────────────────────────────────────────
class BHState:
    def __init__(self, cf, bh_form=None):
        self.cf = cf; self.bh_form = bh_form if bh_form is not None else BH_FORM; self.cf_scale = 1.0
        self.mass = 0.0; self.active = False; self.bh_dir = 0; self.ctl = 0
        self.prices = deque(maxlen=25)

    def update(self, price):
        self.prices.append(float(price))
        if len(self.prices) < 2: return
        px = list(self.prices)
        beta = abs(px[-1] - px[-2]) / (px[-2] + 1e-9) / (self.cf * self.cf_scale + 1e-9)
        was = self.active
        if beta < 1.0:
            self.ctl += 1
            self.mass = self.mass * 0.97 + 0.03 * min(2.0, 1.0 + self.ctl * 0.1)
        else:
            self.ctl = 0; self.mass *= BH_DECAY
        self.active = (self.mass > self.bh_form and self.ctl >= BH_CTL_MIN) if not was else \
                      (self.mass > BH_COLLAPSE and self.ctl >= BH_CTL_MIN)
        if not was and self.active:
            lb = min(20, len(px) - 1)
            self.bh_dir = 1 if px[-1] > px[-1-lb] else -1
        elif was and not self.active:
            self.bh_dir = 0

def _ema(prev, val, a): return val if prev is None else a*val + (1-a)*prev
def _a(n): return 2/(n+1)

class ATRTracker:
    def __init__(self):
        self.atr = self.prev_c = None
    def update(self, h, l, c):
        tr = (h-l) if self.prev_c is None else max(h-l, abs(h-self.prev_c), abs(l-self.prev_c))
        self.atr = _ema(self.atr, tr, _a(14)); self.prev_c = c

class BullScale:
    def __init__(self):
        self.e12=self.e26=self.e50=self.e200=self.last=None
    def update(self, p):
        for attr,n in [("e12",12),("e26",26),("e50",50),("e200",200)]:
            setattr(self, attr, _ema(getattr(self,attr), p, _a(n)))
        self.last = p
    @property
    def scale(self):
        if any(x is None for x in [self.e12,self.e26,self.e50,self.e200]): return 1.0
        return 3.0 if (self.last>self.e200 and self.e12>self.e26 and self.e26>self.e50) else 1.0


# ── Download ──────────────────────────────────────────────────────────────────
def download_crypto():
    client = CryptoHistoricalDataClient()
    now    = datetime.now(timezone.utc)
    data   = {}

    for sym, cfg in INSTRUMENTS.items():
        ticker = cfg["ticker"]
        print(f"  {ticker}...", end=" ", flush=True)
        frames = {}

        for tf_label, tf_obj in [("1d", TimeFrame.Day),
                                  ("4h", TimeFrame(4, TimeFrameUnit.Hour)),
                                  ("1h", TimeFrame.Hour),
                                  ("15m", TimeFrame(15, TimeFrameUnit.Minute))]:
            all_bars = []
            start = START_DATE
            # Paginate in 90-day chunks for sub-daily
            chunk = timedelta(days=90) if tf_label != "1d" else timedelta(days=365*2)
            while start < now:
                end = min(start + chunk, now)
                try:
                    req = CryptoBarsRequest(symbol_or_symbols=[ticker],
                                            timeframe=tf_obj, start=start, end=end)
                    result = client.get_crypto_bars(req)
                    try:
                        bars = result[ticker]
                        all_bars.extend(bars)
                    except KeyError:
                        pass
                    start = end
                    time.sleep(0.2)
                except Exception:
                    time.sleep(2)

            if all_bars:
                df = pd.DataFrame([{
                    "datetime": b.timestamp,
                    "Open": b.open, "High": b.high, "Low": b.low, "Close": b.close
                } for b in all_bars])
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
                df = df.set_index("datetime").sort_index()
                df = df[~df.index.duplicated(keep="last")]
                frames[tf_label] = df
                print(f"{tf_label}:{len(df)} ", end="", flush=True)

        data[sym] = frames
        print()

    return data


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(data):
    syms = list(INSTRUMENTS.keys())

    d_bh   = {s: BHState(INSTRUMENTS[s]["cf_1d"])  for s in syms}
    h4_bh  = {s: BHState(INSTRUMENTS[s]["cf_4h"])  for s in syms}
    h_bh   = {s: BHState(INSTRUMENTS[s]["cf_1h"])  for s in syms}
    m_bh   = {s: BHState(INSTRUMENTS[s]["cf_15m"]) for s in syms}
    d_atr  = {s: ATRTracker() for s in syms}
    h_atr  = {s: ATRTracker() for s in syms}
    bull   = {s: BullScale() for s in syms}

    dollar_pos   = {s: 0.0  for s in syms}
    entry_price  = {s: None for s in syms}
    last_frac    = {s: 0.0  for s in syms}
    bars_held    = {s: 0    for s in syms}
    pos_floor    = {s: 0.0  for s in syms}
    last_15m_px  = {s: None for s in syms}

    equity = STARTING_EQUITY
    peak   = STARTING_EQUITY
    equity_curve = []
    trades = []

    # All daily timestamps as spine
    all_days = data["BTC"]["1d"].index
    warmup_done = {s: False for s in syms}
    d_count = {s: 0 for s in syms}

    for day_idx, day in enumerate(all_days):

        # Update daily BH for all syms
        for s in syms:
            df1d = data[s].get("1d", pd.DataFrame())
            if day not in df1d.index: continue
            row = df1d.loc[day]
            bull[s].update(row["Close"])
            scale = bull[s].scale
            for bh in [d_bh[s], h_bh[s], m_bh[s]]:
                bh.cf_scale = scale
            d_atr[s].update(row["High"], row["Low"], row["Close"])
            d_bh[s].update(row["Close"])
            d_count[s] += 1
            if d_count[s] >= WARMUP_DAYS:
                warmup_done[s] = True

        if not all(warmup_done.values()):
            equity_curve.append((day.date(), equity))
            continue

        # Get hourly bars for this day
        h1_bars = {}
        for s in syms:
            df1h = data[s].get("1h", pd.DataFrame())
            mask  = df1h.index.date == day.date()
            h1_bars[s] = df1h[mask]

        # Get 15m bars for this day
        m15_bars = {}
        for s in syms:
            df15 = data[s].get("15m", pd.DataFrame())
            mask  = df15.index.date == day.date()
            m15_bars[s] = df15[mask]

        # Determine hourly bar times (use BTC as spine)
        bar_times = h1_bars["BTC"].index if not h1_bars["BTC"].empty else [day]

        for bar_time in bar_times:
            # Update 15m BH for bars within this hour
            for s in syms:
                mb = m15_bars[s]
                if mb.empty: continue
                if bar_time != day:  # hourly mode
                    hour_end = bar_time + pd.Timedelta(hours=1)
                    mb = mb[(mb.index >= bar_time) & (mb.index < hour_end)]
                for _, r in mb.iterrows():
                    last_15m_px[s] = r["Close"]
                    m_bh[s].update(r["Close"])

            # Update hourly BH
            for s in syms:
                hb = h1_bars[s]
                if bar_time not in hb.index: continue
                r = hb.loc[bar_time]
                h_atr[s].update(r["High"], r["Low"], r["Close"])
                h_bh[s].update(r["Close"])

            # Current prices
            curr_price = {}
            for s in syms:
                hb = h1_bars[s]
                if bar_time in hb.index:
                    curr_price[s] = float(hb.loc[bar_time, "Close"])
                elif day in data[s]["1d"].index:
                    curr_price[s] = float(data[s]["1d"].loc[day, "Close"])
                else:
                    curr_price[s] = entry_price[s] or 1.0

            # MTM equity
            mtm = sum(dollar_pos[s] * (curr_price[s] - entry_price[s]) / entry_price[s]
                      for s in syms if dollar_pos[s] and entry_price[s])
            equity_live = equity + mtm
            if equity_live > peak: peak = equity_live

            # Compute targets
            tail_frac = min(TAIL_FIXED_CAPITAL, equity_live) / equity_live
            raw = {}
            for s in syms:
                if not warmup_done[s]: raw[s] = 0.0; continue
                d = d_bh[s].active; h = h_bh[s].active; m = m_bh[s].active
                tf = (4 if d else 0) + (2 if h else 0) + (1 if m else 0)
                ceiling = min(TF_CAP.get(tf, 0.0), CRYPTO_CAP)
                # no weak-signal gate in yolo mode
                if ceiling == 0.0: raw[s] = 0.0; continue

                direction = 0
                # Prefer hourly (more current) over daily
                if h and h_bh[s].bh_dir: direction = h_bh[s].bh_dir
                elif d and d_bh[s].bh_dir: direction = d_bh[s].bh_dir
                elif d:
                    px = list(d_bh[s].prices)
                    if len(px) >= 5: direction = 1 if px[-1] > px[-5] else -1

                # Long-only: skip bearish signals
                if direction <= 0: raw[s] = 0.0; continue

                atr = h_atr[s].atr or d_atr[s].atr
                cp  = curr_price[s]
                vol = (atr / cp * math.sqrt(6.5)) if (atr and cp > 0) else 0.01
                raw[s] = min(PER_INST_RISK / (vol + 1e-9), ceiling)

            # Stale-15m exit: losing position + tiny 15m move → close, seek volatility
            for s in syms:
                if raw.get(s, 0.0) == 0.0: continue
                ep   = entry_price[s]
                cp   = curr_price[s]
                px15 = last_15m_px[s]
                if ep and cp and px15:
                    losing = cp < ep  # long-only, so losing = price below entry
                    move15 = abs(cp - px15) / (px15 + 1e-9)
                    if losing and move15 < STALE_15M_MOVE:
                        raw[s] = 0.0

            # pos_floor
            for s in syms:
                tgt = raw.get(s, 0.0)
                d = d_bh[s].active; h = h_bh[s].active
                tf = (4 if d else 0) + (2 if h else 0) + (1 if m_bh[s].active else 0)
                if tf >= 6 and abs(tgt) > 0.15 and h_bh[s].ctl >= 5:
                    pos_floor[s] = max(pos_floor[s], 0.70 * abs(tgt))
                if pos_floor[s] > 0 and tf >= 4 and not np.isclose(last_frac[s], 0.0):
                    raw[s] = max(tgt, pos_floor[s]); pos_floor[s] *= 0.95
                if tf < 4 or np.isclose(tgt, 0.0): pos_floor[s] = 0.0
                if not d and not h: pos_floor[s] = 0.0

            # Normalize + apply
            total = sum(abs(v) for v in raw.values())
            scale = 1.0 / total if total > 1.0 else 1.0

            for s in syms:
                final = raw.get(s, 0.0) * scale * tail_frac

                # Min-hold gate
                if (not np.isclose(last_frac[s], 0.0) and not np.isclose(final, 0.0) and
                        np.sign(final) != np.sign(last_frac[s]) and bars_held[s] < MIN_HOLD):
                    final = last_frac[s]

                if abs(final - last_frac[s]) > 0.02:
                    # Close existing
                    if dollar_pos[s] and entry_price[s]:
                        ret = (curr_price[s] - entry_price[s]) / entry_price[s]
                        pnl = dollar_pos[s] * ret
                        equity += pnl
                        trades.append({
                            "exit_time":   bar_time if bar_time != day else day.date(),
                            "sym":         s,
                            "entry_price": entry_price[s],
                            "exit_price":  curr_price[s],
                            "dollar_pos":  dollar_pos[s],
                            "pnl":         pnl,
                            "hold_bars":   bars_held[s],
                        })

                    if np.isclose(final, 0.0):
                        dollar_pos[s] = 0.0; entry_price[s] = None; bars_held[s] = 0
                    else:
                        dollar_pos[s] = final * equity
                        entry_price[s] = curr_price[s]
                        if np.sign(final) != np.sign(last_frac[s]): bars_held[s] = 0

                    last_frac[s] = final

                if abs(last_frac[s]) > 0.02: bars_held[s] += 1

        # EOD snapshot
        eod = sum(dollar_pos[s] * (curr_price[s] - entry_price[s]) / entry_price[s]
                  for s in syms if dollar_pos[s] and entry_price[s])
        equity_curve.append((day.date(), max(0.0, equity + eod)))

    return equity_curve, trades, peak


# ── Monte Carlo ───────────────────────────────────────────────────────────────
def run_mc(trades, n_sims=MC_SIMS, months=MC_MONTHS):
    if len(trades) < 10:
        print("Not enough trades for MC.")
        return None

    pnls = [t["pnl"] / t["dollar_pos"] for t in trades if t["dollar_pos"] > 0]
    # Trades per month
    df = pd.DataFrame(trades)
    df["exit_time"] = pd.to_datetime(df["exit_time"].astype(str).str[:10])
    span_months = max(1, (df["exit_time"].max() - df["exit_time"].min()).days / 30)
    trades_per_month = len(trades) / span_months

    results = []
    blowups = 0
    for _ in range(n_sims):
        eq = STARTING_EQUITY
        for _ in range(months):
            n = max(1, int(random.gauss(trades_per_month, trades_per_month ** 0.5)))
            for _ in range(n):
                ret   = random.choice(pnls)
                # Position size: same sizing logic simplified
                frac  = min(PER_INST_RISK / 0.02, CRYPTO_CAP) * min(TAIL_FIXED_CAPITAL, eq) / eq
                eq   += frac * eq * ret
                if eq <= 0:
                    eq = 0; blowups += 1; break
            if eq <= 0: break
        results.append(eq)

    results = np.array(results)
    blowup_rate = blowups / n_sims
    return results, blowup_rate, trades_per_month


# ── Stats + plots ─────────────────────────────────────────────────────────────
def print_backtest_stats(equity_curve, trades, peak):
    dates  = [e[0] for e in equity_curve]
    values = np.array([e[1] for e in equity_curve])
    final  = values[-1]
    years  = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    cagr   = (final / STARTING_EQUITY) ** (1 / years) - 1 if years > 0 and final > 0 else 0
    pk     = np.maximum.accumulate(values)
    dd     = (values - pk) / pk
    rets   = pd.Series(values).pct_change().dropna()
    sharpe = rets.mean() / (rets.std() + 1e-9) * math.sqrt(365)

    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf     = sum(wins) / (abs(sum(losses)) + 1e-9) if losses else float("inf")

    print("\n" + "=" * 60)
    print("  CRYPTO BH BACKTEST  (BTC/ETH/SOL, YOLO params, 2021-present)")
    print("=" * 60)
    print(f"  Period:        {dates[0]}  to  {dates[-1]}")
    print(f"  Start equity:  ${STARTING_EQUITY:>13,.0f}")
    print(f"  Peak equity:   ${peak:>13,.0f}")
    print(f"  Final equity:  ${final:>13,.0f}")
    print(f"  CAGR:          {cagr:>13.1%}")
    print(f"  Max drawdown:  {dd.min():>13.1%}")
    print(f"  Sharpe:        {sharpe:>13.2f}")
    print(f"  Trades:        {len(trades):>13,}")
    print(f"  Win rate:      {len(wins)/(len(trades)+1e-9):>13.1%}")
    print(f"  Profit factor: {pf:>13.2f}")
    print("=" * 60)

    # Annual P&L
    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["exit_time"].astype(str).str[:10]).dt.year
    print("\n  ANNUAL P&L:")
    for yr, p in df.groupby("year")["pnl"].sum().items():
        bar = "#" * int(abs(p) / 20000)
        print(f"  {yr}:  {'+'if p>=0 else ''}${p:>10,.0f}  {bar}")

    # Top trades
    df_s = df.sort_values("pnl", ascending=False)
    print(f"\n  TOP 10 TRADES:")
    print(f"  {'Date':<12}  {'Sym':3}  {'P&L':>12}  {'Entry':>10}  {'Exit':>10}")
    print("  " + "-" * 56)
    for _, r in df_s.head(10).iterrows():
        print(f"  {str(r['exit_time'])[:10]:<12}  {r['sym']:3}  "
              f"${r['pnl']:>11,.0f}  {r['entry_price']:>10.2f}  {r['exit_price']:>10.2f}")

    return dates, values, dd


def print_mc_stats(results, blowup_rate, trades_per_month):
    print(f"\n{'='*60}")
    print(f"  MONTE CARLO  ({MC_SIMS:,} sims, {MC_MONTHS}-month forward projection)")
    print(f"  Trades/month: {trades_per_month:.1f}")
    print(f"{'='*60}")
    print(f"  Blowup rate:  {blowup_rate:.1%}")
    print(f"  Median final: ${np.median(results):>13,.0f}")
    print(f"  Mean final:   ${np.mean(results):>13,.0f}")
    print(f"  5th pct:      ${np.percentile(results, 5):>13,.0f}")
    print(f"  25th pct:     ${np.percentile(results, 25):>13,.0f}")
    print(f"  75th pct:     ${np.percentile(results, 75):>13,.0f}")
    print(f"  95th pct:     ${np.percentile(results, 95):>13,.0f}")
    print(f"  Max sim:      ${np.max(results):>13,.0f}")
    print(f"{'='*60}")


def plot_all(dates, values, dd, mc_results):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # equity curve
    ax2 = fig.add_subplot(gs[1, 0])  # drawdown
    ax3 = fig.add_subplot(gs[0, 1])  # MC histogram
    ax4 = fig.add_subplot(gs[1, 1])  # MC percentile bands

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_edgecolor("#333")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    fig.suptitle("LARSA v16 — Crypto BH Engine  (BTC/ETH/SOL, Long-Only)",
                 color="white", fontsize=13)

    # Equity curve
    ax1.plot(dates, values, color="#f7931a", linewidth=1.2)
    ax1.fill_between(dates, values, alpha=0.15, color="#f7931a")
    ax1.axhline(STARTING_EQUITY, color="#555", linewidth=0.5, linestyle="--")
    ax1.set_title("Backtest Equity Curve")
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1e6:.1f}M"))
    ax1.set_yscale("log")
    ax1.grid(alpha=0.12, color="white")

    # Drawdown
    ax2.fill_between(dates, dd * 100, color="#ff4444", alpha=0.7)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("DD %"); ax2.grid(alpha=0.12, color="white")

    # MC histogram
    cap = np.percentile(mc_results, 98)
    clipped = np.clip(mc_results, 0, cap)
    ax3.hist(clipped / 1e6, bins=80, color="#f7931a", alpha=0.7, edgecolor="none")
    ax3.axvline(np.median(mc_results) / 1e6, color="white", linewidth=1.5, linestyle="--",
                label=f"Median ${np.median(mc_results)/1e6:.2f}M")
    ax3.axvline(STARTING_EQUITY / 1e6, color="#ff4444", linewidth=1, linestyle=":",
                label="Start $1M")
    ax3.set_title(f"MC Distribution ({MC_MONTHS}mo forward)")
    ax3.set_xlabel("Final Equity ($M)"); ax3.set_ylabel("Frequency")
    ax3.legend(fontsize=8, labelcolor="white", facecolor="#111")
    ax3.grid(alpha=0.12, color="white")

    # MC percentile fan
    pcts = [5, 25, 50, 75, 95]
    colors = ["#ff4444", "#ff8c00", "#f7931a", "#ffcc00", "#00d4aa"]
    x = list(range(MC_MONTHS + 1))

    # Build monthly equity paths for percentile bands (simplified)
    pnls_list = []
    for sim_eq in mc_results:
        pnls_list.append(sim_eq)

    # Just show bar chart of percentiles
    pct_vals = [np.percentile(mc_results, p) / 1e6 for p in pcts]
    bars = ax4.bar([f"p{p}" for p in pcts], pct_vals, color=colors, alpha=0.8)
    ax4.axhline(STARTING_EQUITY / 1e6, color="#ff4444", linewidth=1, linestyle=":",
                label="Start")
    ax4.set_title(f"MC Percentiles ({MC_MONTHS}mo)")
    ax4.set_ylabel("Final Equity ($M)")
    ax4.grid(alpha=0.12, color="white", axis="y")
    for bar, val in zip(bars, pct_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"${val:.2f}M", ha="center", va="bottom", color="white", fontsize=8)

    png = OUT_DIR / "crypto_bh_mc.png"
    plt.savefig(png, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\n  Chart saved: {png}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Downloading crypto history (2021-present)...")
    data = download_crypto()

    print("\nRunning BH backtest...")
    eq_curve, trades, peak = run_backtest(data)

    dates, values, dd = print_backtest_stats(eq_curve, trades, peak)

    print(f"\nRunning Monte Carlo ({MC_SIMS:,} simulations)...")
    mc_out = run_mc(trades)

    if mc_out:
        mc_results, blowup_rate, tpm = mc_out
        print_mc_stats(mc_results, blowup_rate, tpm)
        plot_all(dates, values, dd, mc_results)
        pd.DataFrame(trades).to_csv(OUT_DIR / "crypto_trades.csv", index=False)
        print(f"  Trades CSV: {OUT_DIR / 'crypto_trades.csv'}")
