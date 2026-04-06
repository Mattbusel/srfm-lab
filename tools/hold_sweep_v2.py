"""
hold_sweep_v2.py — Full 28-instrument hold sweep matching live_trader_alpaca.py v17.

Instruments: 17 crypto + 11 equity (SPY, QQQ, IWM, GLD, TLT, SLV, USO, NVDA, AAPL, TSLA, MSFT)
Hold multipliers tested: 10x, 100x, 1000x  (baseline unit = 4 daily bars)

Key rules matching the live trader:
  - Crypto: long-only
  - Equity: long AND short
  - Equity positions gated to RTH (skipped on weekends/daily bars this is implicit)
  - Shared BH physics + vol sizing + pos_floor + normalization

Usage:
    python tools/hold_sweep_v2.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Instruments (mirrors live_trader_alpaca.py v17) ───────────────────────────
# yf_ticker: what yfinance understands
# asset_class: "crypto" | "equity"
# long_short: True = can go short
INSTRUMENTS = {
    # ── Crypto ────────────────────────────────────────────────────────────────
    "BTC":   {"yf": "BTC-USD",  "asset_class": "crypto", "long_short": False, "cf_1h": 0.030, "cf_1d": 0.10,  "bh_form": 1.9},
    "ETH":   {"yf": "ETH-USD",  "asset_class": "crypto", "long_short": False, "cf_1h": 0.020, "cf_1d": 0.07,  "bh_form": 1.9},
    "XRP":   {"yf": "XRP-USD",  "asset_class": "crypto", "long_short": False, "cf_1h": 0.030, "cf_1d": 0.10,  "bh_form": 1.9},
    "AVAX":  {"yf": "AVAX-USD", "asset_class": "crypto", "long_short": False, "cf_1h": 0.018, "cf_1d": 0.06,  "bh_form": 2.0},
    "LINK":  {"yf": "LINK-USD", "asset_class": "crypto", "long_short": False, "cf_1h": 0.018, "cf_1d": 0.06,  "bh_form": 2.0},
    "DOT":   {"yf": "DOT-USD",  "asset_class": "crypto", "long_short": False, "cf_1h": 0.018, "cf_1d": 0.06,  "bh_form": 2.0},
    "UNI":   {"yf": "UNI7083-USD", "asset_class": "crypto", "long_short": False, "cf_1h": 0.045, "cf_1d": 0.15, "bh_form": 2.0},
    "AAVE":  {"yf": "AAVE-USD", "asset_class": "crypto", "long_short": False, "cf_1h": 0.045, "cf_1d": 0.15,  "bh_form": 2.0},
    "LTC":   {"yf": "LTC-USD",  "asset_class": "crypto", "long_short": False, "cf_1h": 0.030, "cf_1d": 0.10,  "bh_form": 1.9},
    "BCH":   {"yf": "BCH-USD",  "asset_class": "crypto", "long_short": False, "cf_1h": 0.035, "cf_1d": 0.12,  "bh_form": 2.0},
    "DOGE":  {"yf": "DOGE-USD", "asset_class": "crypto", "long_short": False, "cf_1h": 0.060, "cf_1d": 0.20,  "bh_form": 2.0},
    "SHIB":  {"yf": "SHIB-USD", "asset_class": "crypto", "long_short": False, "cf_1h": 0.075, "cf_1d": 0.25,  "bh_form": 2.1},
    "BAT":   {"yf": "BAT-USD",  "asset_class": "crypto", "long_short": False, "cf_1h": 0.060, "cf_1d": 0.20,  "bh_form": 2.0},
    "CRV":   {"yf": "CRV-USD",  "asset_class": "crypto", "long_short": False, "cf_1h": 0.060, "cf_1d": 0.20,  "bh_form": 2.0},
    "SUSHI": {"yf": "SUSHI-USD","asset_class": "crypto", "long_short": False, "cf_1h": 0.060, "cf_1d": 0.20,  "bh_form": 2.0},
    "MKR":   {"yf": "MKR-USD",  "asset_class": "crypto", "long_short": False, "cf_1h": 0.045, "cf_1d": 0.15,  "bh_form": 2.0},
    "YFI":   {"yf": "YFI-USD",  "asset_class": "crypto", "long_short": False, "cf_1h": 0.045, "cf_1d": 0.15,  "bh_form": 2.0},
    # ── Equity ────────────────────────────────────────────────────────────────
    "SPY":   {"yf": "SPY",   "asset_class": "equity", "long_short": True, "cf_1h": 0.001,  "cf_1d": 0.005, "bh_form": 1.5},
    "QQQ":   {"yf": "QQQ",   "asset_class": "equity", "long_short": True, "cf_1h": 0.0012, "cf_1d": 0.006, "bh_form": 1.5},
    "IWM":   {"yf": "IWM",   "asset_class": "equity", "long_short": True, "cf_1h": 0.0015, "cf_1d": 0.007, "bh_form": 1.5},
    "GLD":   {"yf": "GLD",   "asset_class": "equity", "long_short": True, "cf_1h": 0.0012, "cf_1d": 0.005, "bh_form": 1.5},
    "TLT":   {"yf": "TLT",   "asset_class": "equity", "long_short": True, "cf_1h": 0.0010, "cf_1d": 0.004, "bh_form": 1.5},
    "SLV":   {"yf": "SLV",   "asset_class": "equity", "long_short": True, "cf_1h": 0.0018, "cf_1d": 0.008, "bh_form": 1.6},
    "USO":   {"yf": "USO",   "asset_class": "equity", "long_short": True, "cf_1h": 0.0025, "cf_1d": 0.010, "bh_form": 1.8},
    "NVDA":  {"yf": "NVDA",  "asset_class": "equity", "long_short": True, "cf_1h": 0.004,  "cf_1d": 0.015, "bh_form": 1.6},
    "AAPL":  {"yf": "AAPL",  "asset_class": "equity", "long_short": True, "cf_1h": 0.002,  "cf_1d": 0.008, "bh_form": 1.5},
    "TSLA":  {"yf": "TSLA",  "asset_class": "equity", "long_short": True, "cf_1h": 0.005,  "cf_1d": 0.018, "bh_form": 1.7},
    "MSFT":  {"yf": "MSFT",  "asset_class": "equity", "long_short": True, "cf_1h": 0.0016, "cf_1d": 0.007, "bh_form": 1.5},
}

START_DATE      = "2015-01-01"   # crypto data only goes back so far
STARTING_EQUITY = 1_000_000.0
TAIL_FIXED_CAP  = 3_000_000.0
PORT_RISK       = 0.01
N_INST          = len(INSTRUMENTS)
CORR            = 0.30
CORR_FACTOR     = math.sqrt(N_INST + N_INST * (N_INST - 1) * CORR)
PER_INST_RISK   = PORT_RISK / CORR_FACTOR

TF_CAP      = {6: 0.55, 4: 0.35, 2: 0.25, 0: 0.0}
EQUITY_CAP  = 0.20
CRYPTO_CAP  = 0.40
BH_DECAY    = 0.924
BH_FORM     = 1.9
BH_COLLAPSE = 0.992

# Baseline = 3 hourly bars (~20 hours, matching live trader MIN_HOLD=80 x 15m bars)
BASELINE_HOLD = 3
MULTIPLIERS   = [10, 100, 1000]

OUT_DIR = Path(__file__).parent / "backtest_output"
OUT_DIR.mkdir(exist_ok=True)


# ── BH State ──────────────────────────────────────────────────────────────────
class BHState:
    def __init__(self, cf, bh_form=1.9):
        self.cf       = cf
        self.bh_form  = bh_form
        self.mass     = 0.0
        self.active   = False
        self.bh_dir   = 0
        self.ctl      = 0
        self.prices   = []

    def update(self, price):
        self.prices.append(float(price))
        if len(self.prices) > 25:
            self.prices.pop(0)
        if len(self.prices) < 2:
            return
        beta = abs(self.prices[-1] - self.prices[-2]) / (self.prices[-2] + 1e-9) / (self.cf + 1e-9)
        was  = self.active
        if beta < 1.0:
            self.ctl += 1
            self.mass = self.mass * 0.97 + 0.03 * min(2.0, 1.0 + self.ctl * 0.1)
        else:
            self.ctl  = 0
            self.mass *= BH_DECAY
        self.active = (self.mass > self.bh_form and self.ctl >= 3) if not was else \
                      (self.mass > BH_COLLAPSE and self.ctl >= 3)
        if not was and self.active:
            lb = min(20, len(self.prices) - 1)
            self.bh_dir = 1 if self.prices[-1] > self.prices[-1 - lb] else -1
        elif was and not self.active:
            self.bh_dir = 0


def ema_s(s, n):
    return s.ewm(span=n, adjust=False).mean()

def atr_s(hi, lo, cl, n=14):
    tr = pd.concat([(hi - lo), (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def bull_scale(close):
    e12, e26, e50, e200 = ema_s(close, 12), ema_s(close, 26), ema_s(close, 50), ema_s(close, 200)
    return ((close > e200) & (e12 > e26) & (e26 > e50)).map({True: 3.0, False: 1.0})


# ── Download ──────────────────────────────────────────────────────────────────
def download_data():
    daily  = {}
    hourly = {}

    cache_dir = Path(__file__).parent / "data_cache"

    for sym, cfg in INSTRUMENTS.items():
        tk = cfg["yf"]
        print(f"  {sym} ({tk}) daily...", end=" ", flush=True)
        try:
            d = yf.download(tk, start=START_DATE, interval="1d", progress=False, auto_adjust=True)
            d.columns = d.columns.get_level_values(0)
            daily[sym] = d
            print(f"{len(d)} bars")
        except Exception as e:
            print(f"FAILED: {e}")
            daily[sym] = pd.DataFrame()

        # Try cache first for hourly
        h1_path = cache_dir / f"{sym}_1h.csv"
        if h1_path.exists():
            h = pd.read_csv(h1_path, index_col=0, parse_dates=True)
            if h.index.tz is not None:
                h.index = h.index.tz_localize(None)
            print(f"  {sym} 1h (cache)... {len(h)} bars")
            hourly[sym] = h
        else:
            print(f"  {sym} 1h (yfinance)...", end=" ", flush=True)
            try:
                h = yf.download(tk, period="730d", interval="1h", progress=False, auto_adjust=True)
                h.columns = h.columns.get_level_values(0)
                if h.index.tz is not None:
                    h.index = h.index.tz_localize(None)
                print(f"{len(h)} bars")
                hourly[sym] = h
            except Exception as e:
                print(f"FAILED: {e}")
                hourly[sym] = pd.DataFrame()

    return daily, hourly


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(daily, hourly, min_hold_any: int):
    syms = list(INSTRUMENTS.keys())

    d_atr   = {}
    d_scale = {}
    h_atr   = {}
    for s in syms:
        if daily[s].empty:
            continue
        d_atr[s]   = atr_s(daily[s]["High"], daily[s]["Low"], daily[s]["Close"])
        d_scale[s] = bull_scale(daily[s]["Close"])
        if not hourly[s].empty:
            h_atr[s] = atr_s(hourly[s]["High"], hourly[s]["Low"], hourly[s]["Close"])

    d_bh = {s: BHState(INSTRUMENTS[s]["cf_1d"], INSTRUMENTS[s]["bh_form"]) for s in syms}
    h_bh = {s: BHState(INSTRUMENTS[s]["cf_1h"], INSTRUMENTS[s]["bh_form"]) for s in syms}

    hourly_dates = {s: set(hourly[s].index.date) for s in syms if not hourly[s].empty}

    # Use ES/SPY as the reference calendar
    ref_sym = "SPY" if "SPY" in daily and not daily["SPY"].empty else syms[0]
    all_days = daily[ref_sym].index[daily[ref_sym].index >= pd.Timestamp(START_DATE)]

    dollar_pos  = {s: 0.0 for s in syms}
    entry_price = {s: None for s in syms}
    last_frac   = {s: 0.0 for s in syms}
    bars_held   = {s: 0 for s in syms}
    pos_floor   = {s: 0.0 for s in syms}

    equity       = STARTING_EQUITY
    peak         = STARTING_EQUITY
    equity_curve = []
    trades       = []
    WARMUP       = 200

    for day_idx, day in enumerate(all_days):
        day_date = day.date()

        # Update daily BH for all instruments
        for s in syms:
            if daily[s].empty or day not in daily[s].index:
                continue
            scale = float(d_scale[s].get(day, 1.0)) if s in d_scale else 1.0
            d_bh[s].cf = INSTRUMENTS[s]["cf_1d"] * scale
            d_bh[s].update(daily[s].loc[day, "Close"])

        if day_idx < WARMUP:
            equity_curve.append((day_date, equity))
            continue

        use_hourly = any(day_date in hourly_dates.get(s, set()) for s in syms)

        if use_hourly:
            h_bars = {s: hourly[s][hourly[s].index.date == day_date]
                      for s in syms if not hourly[s].empty}
            ref = max(h_bars, key=lambda s: len(h_bars.get(s, pd.DataFrame()))) if h_bars else ref_sym
            bar_times = h_bars[ref].index if ref in h_bars and not h_bars[ref].empty else [day]
        else:
            bar_times = [day]

        for bar_time in bar_times:
            if use_hourly:
                for s in syms:
                    if s not in h_bars:
                        continue
                    b = h_bars[s]
                    if bar_time not in b.index:
                        continue
                    h_bh[s].update(b.loc[bar_time, "Close"])

            curr_price = {}
            for s in syms:
                if use_hourly and s in h_bars and bar_time in h_bars[s].index:
                    curr_price[s] = float(h_bars[s].loc[bar_time, "Close"])
                elif not daily[s].empty and day in daily[s].index:
                    curr_price[s] = float(daily[s].loc[day, "Close"])
                else:
                    curr_price[s] = entry_price[s] if entry_price[s] else 1.0

            # Mark to market
            mtm = sum(
                dollar_pos[s] * (curr_price[s] - entry_price[s]) / entry_price[s]
                for s in syms
                if dollar_pos[s] != 0.0 and entry_price[s] and entry_price[s] > 0
            )
            equity_live = equity + mtm
            if equity_live > peak:
                peak = equity_live

            tail_frac   = min(TAIL_FIXED_CAP, equity_live) / equity_live
            raw_targets = {}

            for s in syms:
                cfg         = INSTRUMENTS[s]
                asset_class = cfg["asset_class"]
                long_short  = cfg["long_short"]

                d_active = d_bh[s].active
                h_active = use_hourly and h_bh[s].active
                tf       = (4 if d_active else 0) + (2 if h_active else 0)

                if asset_class == "equity":
                    ceiling = min(TF_CAP.get(tf, 0.0), EQUITY_CAP)
                else:
                    ceiling = min(TF_CAP.get(tf, 0.0), CRYPTO_CAP)

                if ceiling == 0.0:
                    raw_targets[s] = 0.0
                    continue

                direction = 0
                if h_active and h_bh[s].bh_dir:
                    direction = h_bh[s].bh_dir
                elif d_active and d_bh[s].bh_dir:
                    direction = d_bh[s].bh_dir

                # Crypto long-only; equities long/short
                if not long_short and direction <= 0:
                    raw_targets[s] = 0.0
                    continue
                if direction == 0:
                    raw_targets[s] = 0.0
                    continue

                # Vol sizing
                cp = curr_price.get(s, 1.0)
                if use_hourly and s in h_atr and bar_time in h_atr[s].index:
                    vol_pct = float(h_atr[s].loc[bar_time]) / (cp + 1e-9) * math.sqrt(6.5)
                elif s in d_atr and day in d_atr[s].index:
                    vol_pct = float(d_atr[s].loc[day]) / (cp + 1e-9)
                else:
                    vol_pct = 0.01

                cap = min(PER_INST_RISK / (vol_pct + 1e-9), ceiling)
                raw_targets[s] = cap * direction

            # pos_floor
            for s in syms:
                tgt     = raw_targets.get(s, 0.0)
                d_act   = d_bh[s].active
                h_act   = use_hourly and h_bh[s].active
                tf      = (4 if d_act else 0) + (2 if h_act else 0)
                if tf >= 6 and abs(tgt) > 0.15 and not np.isclose(tgt, 0.0) and h_bh[s].ctl >= 5:
                    pos_floor[s] = max(pos_floor[s], 0.70 * abs(tgt))
                if pos_floor[s] > 0.0 and tf >= 4 and not np.isclose(last_frac[s], 0.0):
                    raw_targets[s] = float(np.sign(last_frac[s]) * max(abs(tgt), pos_floor[s]))
                    pos_floor[s] *= 0.95
                if tf < 4 or np.isclose(tgt, 0.0):
                    pos_floor[s] = 0.0
                if not d_act and not h_act:
                    pos_floor[s] = 0.0

            # Normalize
            total_exp = sum(abs(v) for v in raw_targets.values())
            scale_n   = 1.0 / total_exp if total_exp > 1.0 else 1.0

            for s in syms:
                tgt   = raw_targets.get(s, 0.0)
                final = tgt * scale_n * tail_frac

                # ── Hard min-hold: lock ALL exits until bars_held >= min_hold_any ──
                if not np.isclose(last_frac[s], 0.0) and bars_held[s] < min_hold_any:
                    final = last_frac[s]

                if abs(final - last_frac[s]) > 0.02:
                    if dollar_pos[s] != 0.0 and entry_price[s] is not None:
                        ret   = (curr_price[s] - entry_price[s]) / entry_price[s]
                        pnl   = dollar_pos[s] * ret
                        equity += pnl
                        trades.append({
                            "exit_time":   bar_time if use_hourly else day_date,
                            "sym":         s,
                            "asset_class": INSTRUMENTS[s]["asset_class"],
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

    # Per asset-class breakdown
    crypto_trades = [t for t in trades if t["asset_class"] == "crypto"]
    equity_trades = [t for t in trades if t["asset_class"] == "equity"]
    crypto_pnl    = sum(t["pnl"] for t in crypto_trades)
    equity_pnl    = sum(t["pnl"] for t in equity_trades)

    return dict(
        sharpe=round(sharpe, 3),
        cagr=round(cagr * 100, 2),
        max_dd=round(max_dd * 100, 2),
        trades=len(trades),
        crypto_trades=len(crypto_trades),
        equity_trades=len(equity_trades),
        crypto_pnl=round(crypto_pnl, 0),
        equity_pnl=round(equity_pnl, 0),
        win_rate=round(win_rate * 100, 1),
        profit_fac=round(pf, 2),
        final_equity=round(final, 0),
    )


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  HOLD SWEEP v2 — 28 instruments (17 crypto + 11 equity)")
print("=" * 65)
print("  Downloading data...")
print()
daily, hourly = download_data()
print()

results = []
for mult in MULTIPLIERS:
    hold = BASELINE_HOLD * mult
    print(f"  Running min_hold={hold} ({mult}x)...", flush=True)
    ec, trades, peak = run_backtest(daily, hourly, hold)
    stats = compute_stats(ec, trades)
    stats["multiplier"] = mult
    stats["min_hold"]   = hold
    results.append(stats)
    print(f"    Sharpe {stats['sharpe']:.3f}  CAGR {stats['cagr']:.1f}%  "
          f"MaxDD {stats['max_dd']:.1f}%  Trades {stats['trades']} "
          f"(crypto={stats['crypto_trades']} eq={stats['equity_trades']})")

# ── Results table ─────────────────────────────────────────────────────────────
print()
print("=" * 90)
print("  FULL 28-INSTRUMENT HOLD SWEEP RESULTS")
print("=" * 90)
print(f"  {'Mult':>5}  {'Hold':>5}  {'Sharpe':>7}  {'CAGR%':>6}  {'MaxDD%':>7}  "
      f"{'Trades':>7}  {'CryptoTr':>9}  {'EqTr':>6}  {'WinRate%':>9}  {'ProfFac':>8}")
print("  " + "-" * 86)
for r in results:
    print(f"  {r['multiplier']:>4}x  {r['min_hold']:>5}  {r['sharpe']:>7.3f}  "
          f"{r['cagr']:>5.1f}%  {r['max_dd']:>6.1f}%  {r['trades']:>7}  "
          f"{r['crypto_trades']:>9}  {r['equity_trades']:>6}  "
          f"{r['win_rate']:>8.1f}%  {r['profit_fac']:>8.2f}")
print("=" * 90)

best = max(results, key=lambda r: r["sharpe"])
print(f"\n  Best Sharpe: {best['sharpe']} at {best['multiplier']}x (hold={best['min_hold']} bars)")
print(f"\n  P&L split at best hold ({best['multiplier']}x):")
print(f"    Crypto: ${best['crypto_pnl']:>12,.0f}")
print(f"    Equity: ${best['equity_pnl']:>12,.0f}")

out = OUT_DIR / "hold_sweep_v2.csv"
pd.DataFrame(results).to_csv(out, index=False)
print(f"\n  Results saved: {out}")
