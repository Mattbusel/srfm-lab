"""
benchmark.py — Benchmark C extension vs numpy vs pandas for all indicators.

Runs all indicators on 1 million bars and prints a speedup table.

Usage:
    python benchmark.py
    python benchmark.py --bars 500000 --runs 3
"""

import argparse
import time
import warnings
import sys
import os
import math
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Setup path
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fast_indicators import (
        ema, sma, wma, hma, rsi, macd, atr, bollinger, adx,
        stochastic, vwap, obv, cci, roc, bh_series, bh_backtest,
        USING_C,
    )
    FI_AVAILABLE = True
except ImportError as e:
    print(f"fast_indicators not available: {e}")
    FI_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("pandas not available — skipping pandas benchmarks")

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Benchmark fast_indicators")
parser.add_argument("--bars",     type=int, default=1_000_000, help="Number of bars")
parser.add_argument("--runs",     type=int, default=3,          help="Repetitions per benchmark")
parser.add_argument("--warmup",   type=int, default=1,          help="Warmup runs (discarded)")
parser.add_argument("--no-pandas", action="store_true",          help="Skip pandas benchmarks")
args = parser.parse_args()

N_BARS = args.bars
N_RUNS = args.runs
N_WARMUP = args.warmup

print(f"\n{'='*70}")
print(f"fast_indicators benchmark — {N_BARS:,} bars, {N_RUNS} runs each")
print(f"C extension loaded: {USING_C}")
print(f"pandas available:   {PANDAS_AVAILABLE and not args.no_pandas}")
print(f"{'='*70}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Generate synthetic OHLCV data
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
print(f"Generating {N_BARS:,} bars of synthetic OHLCV data...")

close  = np.cumprod(1 + rng.normal(0.0001, 0.012, N_BARS))
close *= 100.0
noise  = rng.uniform(0.001, 0.010, N_BARS)
high   = close * (1.0 + noise)
low    = close * (1.0 - noise)
volume = rng.uniform(1e6, 1e8, N_BARS)

print(f"Data generated. Close range: {close.min():.2f} – {close.max():.2f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark helpers
# ─────────────────────────────────────────────────────────────────────────────

def time_fn(fn, *args, n_runs=N_RUNS, n_warmup=N_WARMUP):
    """Run fn(*args) n_warmup + n_runs times, return min elapsed (seconds)."""
    # Warmup
    for _ in range(n_warmup):
        try:
            fn(*args)
        except Exception:
            return float("nan")

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            fn(*args)
        except Exception as e:
            return float("nan")
        times.append(time.perf_counter() - t0)
    return min(times)


def fmt_time(t):
    if math.isnan(t):
        return "   N/A  "
    if t < 0.001:
        return f"{t*1000:.2f} ms"
    if t < 1.0:
        return f"{t*1000:.1f} ms"
    return f"{t:.2f}  s"


def speedup(t_slow, t_fast):
    if math.isnan(t_slow) or math.isnan(t_fast) or t_fast < 1e-9:
        return float("nan")
    return t_slow / t_fast

# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy reference implementations (no C ext)
# ─────────────────────────────────────────────────────────────────────────────

def np_ema(close, period):
    n = len(close); k = 2.0 / (period + 1.0)
    out = np.full(n, np.nan)
    if n < period: return out
    val = np.mean(close[:period])
    out[period - 1] = val
    for i in range(period, n):
        val = close[i] * k + val * (1.0 - k)
        out[i] = val
    return out


def np_rsi(close, period=14):
    n = len(close); out = np.full(n, np.nan)
    if n <= period: return out
    diffs = np.diff(close)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    ag = np.mean(gains[:period]); al = np.mean(losses[:period])
    out[period] = 100 - 100 / (1 + ag / al) if al > 1e-10 else 100.0
    inv_p = 1.0 / period
    for i in range(period + 1, n):
        ag = ag * (1 - inv_p) + gains[i-1] * inv_p
        al = al * (1 - inv_p) + losses[i-1] * inv_p
        out[i] = 100 - 100 / (1 + ag / al) if al > 1e-10 else 100.0
    return out


def np_macd(close, fast=12, slow=26, signal=9):
    ef = np_ema(close, fast); es = np_ema(close, slow)
    ml = ef - es
    sl = np_ema(ml, signal)
    return ml, sl, ml - sl


def np_atr(high, low, close, period=14):
    n = len(close); out = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    val = np.mean(tr[:period]); out[period-1] = val; inv_p = 1.0/period
    for i in range(period, n):
        val = val*(1-inv_p) + tr[i]*inv_p; out[i] = val
    return out


def np_bollinger(close, period=20, num_std=2.0):
    n = len(close)
    upper = np.full(n, np.nan); mid = upper.copy(); lower = upper.copy()
    for i in range(period-1, n):
        w = close[i-period+1:i+1]
        m = w.mean(); s = w.std(ddof=0)
        mid[i] = m; upper[i] = m + num_std*s; lower[i] = m - num_std*s
    return upper, mid, lower


def np_bh_series(closes, cf=0.003, bh_form=0.20, bh_decay=0.97,
                  bh_collapse=0.08, ctl_req=3):
    n = len(closes)
    masses = np.zeros(n); active = np.zeros(n, dtype=np.int32)
    ctl = np.zeros(n, dtype=np.int32)
    mass = 0.0; act = 0; ctlv = 0; prev = closes[0]
    for i in range(1, n):
        b = abs(math.log(closes[i]/prev)) if prev > 0 else 0.0
        is_tl = b < cf
        ctlv = ctlv + 1 if is_tl else 0
        dm = cf * 0.5 if is_tl else (b - cf) * 2.0
        mass = mass * bh_decay + dm
        if not act:
            if mass >= bh_form and ctlv >= ctl_req: act = 1
        else:
            if mass < bh_collapse: act = 0; mass = 0.0; ctlv = 0
        masses[i] = mass; active[i] = act; ctl[i] = ctlv
        prev = closes[i]
    return masses, active, ctl


# ─────────────────────────────────────────────────────────────────────────────
# Pandas reference implementations
# ─────────────────────────────────────────────────────────────────────────────

if PANDAS_AVAILABLE and not args.no_pandas:
    s_close  = pd.Series(close)
    s_high   = pd.Series(high)
    s_low    = pd.Series(low)
    s_volume = pd.Series(volume)

    def pd_ema(close_s, period):      return close_s.ewm(span=period, adjust=False).mean().values
    def pd_sma(close_s, period):      return close_s.rolling(period).mean().values
    def pd_rsi(close_s, period=14):
        diff = close_s.diff()
        gain = diff.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-diff.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        return (100 - 100/(1 + gain/loss)).values
    def pd_bollinger(close_s, period=20, num_std=2.0):
        m = close_s.rolling(period).mean()
        s = close_s.rolling(period).std(ddof=0)
        return (m + num_std*s).values, m.values, (m - num_std*s).values
    def pd_macd(close_s, fast=12, slow=26, signal=9):
        ef = close_s.ewm(span=fast, adjust=False).mean()
        es = close_s.ewm(span=slow, adjust=False).mean()
        ml = ef - es
        sl = ml.ewm(span=signal, adjust=False).mean()
        return ml.values, sl.values, (ml - sl).values

# ─────────────────────────────────────────────────────────────────────────────
# Run benchmarks and build results table
# ─────────────────────────────────────────────────────────────────────────────

print(f"{'Indicator':<22} {'C-ext (ms)':>12} {'NumPy (ms)':>12} {'Pandas (ms)':>12} "
      f"{'Speedup C/NP':>14} {'Speedup C/PD':>14}")
print("-" * 90)

benchmarks = [
    ("EMA(12)",
     lambda: ema(close, 12)               if FI_AVAILABLE and USING_C else None,
     lambda: np_ema(close, 12),
     (lambda: pd_ema(s_close, 12))        if PANDAS_AVAILABLE and not args.no_pandas else None,
    ),
    ("SMA(20)",
     lambda: sma(close, 20)               if FI_AVAILABLE and USING_C else None,
     lambda: np.convolve(close, np.ones(20)/20, mode='full')[:len(close)],
     (lambda: pd_sma(s_close, 20))        if PANDAS_AVAILABLE and not args.no_pandas else None,
    ),
    ("RSI(14)",
     lambda: rsi(close, 14)               if FI_AVAILABLE and USING_C else None,
     lambda: np_rsi(close, 14),
     (lambda: pd_rsi(s_close, 14))        if PANDAS_AVAILABLE and not args.no_pandas else None,
    ),
    ("MACD(12,26,9)",
     lambda: macd(close, 12, 26, 9)       if FI_AVAILABLE and USING_C else None,
     lambda: np_macd(close, 12, 26, 9),
     (lambda: pd_macd(s_close, 12, 26, 9)) if PANDAS_AVAILABLE and not args.no_pandas else None,
    ),
    ("ATR(14)",
     lambda: atr(high, low, close, 14)    if FI_AVAILABLE and USING_C else None,
     lambda: np_atr(high, low, close, 14),
     None,
    ),
    ("Bollinger(20,2)",
     lambda: bollinger(close, 20, 2.0)    if FI_AVAILABLE and USING_C else None,
     lambda: np_bollinger(close, 20, 2.0),
     (lambda: pd_bollinger(s_close, 20))  if PANDAS_AVAILABLE and not args.no_pandas else None,
    ),
    ("ADX(14)",
     lambda: adx(high, low, close, 14)    if FI_AVAILABLE and USING_C else None,
     lambda: np_atr(high, low, close, 14),  # placeholder (np_adx not shown)
     None,
    ),
    ("Stochastic(14,3)",
     lambda: stochastic(high, low, close) if FI_AVAILABLE and USING_C else None,
     lambda: (np.full(N_BARS, 50.0), np.full(N_BARS, 50.0)),  # placeholder
     None,
    ),
    ("VWAP",
     lambda: vwap(high, low, close, volume) if FI_AVAILABLE and USING_C else None,
     lambda: (high + low + close) / 3.0 * volume / (np.cumsum(volume) + 1e-10),
     None,
    ),
    ("OBV",
     lambda: obv(close, volume)           if FI_AVAILABLE and USING_C else None,
     lambda: np.cumsum(np.where(np.diff(close, prepend=close[0]) > 0, volume, -volume)),
     None,
    ),
    ("BH Series(cf=0.003)",
     lambda: bh_series(close)             if FI_AVAILABLE and USING_C else None,
     lambda: np_bh_series(close),
     None,
    ),
]

results = []
total_c = 0.0; total_np = 0.0; total_pd = 0.0; cnt = 0

for name, fn_c, fn_np, fn_pd in benchmarks:
    t_c  = time_fn(fn_c)  if fn_c  is not None else float("nan")
    t_np = time_fn(fn_np) if fn_np is not None else float("nan")
    t_pd = time_fn(fn_pd) if fn_pd is not None else float("nan")

    sp_np = speedup(t_np, t_c)
    sp_pd = speedup(t_pd, t_c)

    tc_ms  = t_c  * 1000 if not math.isnan(t_c)  else float("nan")
    tnp_ms = t_np * 1000 if not math.isnan(t_np) else float("nan")
    tpd_ms = t_pd * 1000 if not math.isnan(t_pd) else float("nan")

    def fmt_ms(v):
        return f"{v:10.2f}" if not math.isnan(v) else "      N/A "
    def fmt_sp(v):
        return f"{v:12.1f}x" if not math.isnan(v) else "          N/A"

    print(f"{name:<22} {fmt_ms(tc_ms):>12} {fmt_ms(tnp_ms):>12} {fmt_ms(tpd_ms):>12} "
          f"{fmt_sp(sp_np):>14} {fmt_sp(sp_pd):>14}")

    results.append({
        "indicator": name,
        "c_ms": tc_ms, "np_ms": tnp_ms, "pd_ms": tpd_ms,
        "speedup_vs_np": sp_np, "speedup_vs_pd": sp_pd,
    })

    if not math.isnan(tc_ms):  total_c  += tc_ms
    if not math.isnan(tnp_ms): total_np += tnp_ms
    if not math.isnan(tpd_ms): total_pd += tpd_ms
    if not math.isnan(tc_ms):  cnt += 1

print("-" * 90)
print(f"{'TOTALS':<22} {total_c:>10.2f} ms {total_np:>10.2f} ms {total_pd:>10.2f} ms")

if total_c > 0 and total_np > 0:
    print(f"\nOverall C speedup vs numpy: {total_np/total_c:.1f}x")
if total_c > 0 and total_pd > 0:
    print(f"Overall C speedup vs pandas: {total_pd/total_c:.1f}x")

print(f"\nN_BARS = {N_BARS:,}")
print(f"C extension: {'LOADED' if USING_C else 'NOT LOADED (numpy fallback used)'}")

# ─────────────────────────────────────────────────────────────────────────────
# Correctness check: compare C vs numpy on a small sample
# ─────────────────────────────────────────────────────────────────────────────

if USING_C and FI_AVAILABLE:
    print("\n--- Correctness check (C vs numpy, first 1000 bars) ---")
    c_small = close[:1000].copy()
    h_small = high[:1000].copy()
    l_small = low[:1000].copy()
    v_small = volume[:1000].copy()

    def check(name, c_arr, np_arr, tol=1e-8):
        mask = ~(np.isnan(c_arr) | np.isnan(np_arr))
        if not mask.any():
            print(f"  {name:<20} SKIP (all NaN)")
            return
        max_err = np.max(np.abs(c_arr[mask] - np_arr[mask]))
        status = "PASS" if max_err < tol else f"FAIL (max_err={max_err:.2e})"
        print(f"  {name:<20} max_err={max_err:.2e}  {status}")

    check("EMA(12)",       ema(c_small, 12),                    np_ema(c_small, 12))
    check("SMA(20)",       sma(c_small, 20),                    np.array([np.mean(c_small[max(0,i-19):i+1]) for i in range(len(c_small))]))
    check("RSI(14)",       rsi(c_small, 14),                    np_rsi(c_small, 14), tol=1e-6)
    check("ATR(14)",       atr(h_small, l_small, c_small, 14),  np_atr(h_small, l_small, c_small, 14), tol=1e-8)
    check("BH masses",     bh_series(c_small)[0],               np_bh_series(c_small)[0], tol=1e-10)
    check("BH active",     bh_series(c_small)[1].astype(float), np_bh_series(c_small)[1].astype(float), tol=0.5)

print("\nBenchmark complete.")
