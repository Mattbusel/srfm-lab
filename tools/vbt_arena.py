"""
vbt_arena.py — vectorbt-powered fast backtesting.

Uses vectorbt's vectorized portfolio engine for fast parameter sweeps.
Bypasses the Python event loop entirely — pure NumPy broadcasting.

Usage:
    python tools/vbt_arena.py --csv data/NDX_hourly_poly.csv
    python tools/vbt_arena.py --sweep cf        # sweep cf values, show results
    python tools/vbt_arena.py --compare         # compare vbt vs arena_v2 results
    python tools/vbt_arena.py --synthetic       # use synthetic bars
"""

import argparse
import csv as _csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ohlcv(path: str):
    bars = []
    with open(path) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            def g(*keys):
                for k in keys:
                    v = row.get(k) or row.get(k.lower()) or row.get(k.upper())
                    if v not in (None, "", "null", "None"):
                        try:
                            return float(v)
                        except Exception:
                            pass
                return None
            c = g("close", "Close")
            if c and c > 0:
                bars.append({
                    "date":   row.get("date") or row.get("Date") or "",
                    "open":   g("open", "Open") or c,
                    "high":   g("high", "High") or c,
                    "low":    g("low", "Low") or c,
                    "close":  c,
                    "volume": g("volume", "Volume") or 1000.0,
                })
    return bars


def bars_to_arrays(bars):
    closes  = np.array([b["close"]  for b in bars], dtype=np.float64)
    highs   = np.array([b["high"]   for b in bars], dtype=np.float64)
    lows    = np.array([b["low"]    for b in bars], dtype=np.float64)
    volumes = np.array([b["volume"] for b in bars], dtype=np.float64)
    dates   = [b["date"] for b in bars]
    return closes, highs, lows, volumes, dates


# ---------------------------------------------------------------------------
# Vectorized SRFM signal computation
# ---------------------------------------------------------------------------

def compute_srfm_signals(closes: np.ndarray, cf: float,
                          bh_form: float = 1.5, bh_decay: float = 0.95):
    """
    Vectorized SRFM-inspired signal generator.

    Returns:
        long_entries, long_exits, short_entries, short_exits  (bool arrays, len=n-1)
    """
    n = len(closes)
    if n < 30:
        z = np.zeros(n - 1, dtype=bool)
        return z, z, z, z

    # Beta (price velocity relative to cf threshold)
    betas = np.abs(np.diff(closes)) / (closes[:-1] * cf + 1e-12)

    # BH mass — recurrent, but use numba if available else numpy loop
    bh_mass = np.zeros(n, dtype=np.float64)
    bh_dir  = np.zeros(n, dtype=np.int8)
    try:
        import numba as nb

        @nb.njit
        def _bh_loop(closes, betas, bh_form, bh_decay, bh_mass, bh_dir):
            for i in range(1, len(closes)):
                b = betas[i - 1]
                prev_mass = bh_mass[i - 1] * bh_decay
                if b >= bh_form:
                    direction = 1 if closes[i] > closes[i - 1] else -1
                    bh_mass[i] = prev_mass + b
                    bh_dir[i]  = direction
                else:
                    bh_mass[i] = max(0.0, prev_mass - 0.05)
                    bh_dir[i]  = bh_dir[i - 1]
            return bh_mass, bh_dir

        bh_mass, bh_dir = _bh_loop(closes, betas, bh_form, bh_decay, bh_mass, bh_dir)
    except ImportError:
        # Pure numpy loop — slower but correct
        for i in range(1, n):
            b = betas[i - 1]
            prev_mass = bh_mass[i - 1] * bh_decay
            if b >= bh_form:
                direction = 1 if closes[i] > closes[i - 1] else -1
                bh_mass[i] = prev_mass + b
                bh_dir[i]  = direction
            else:
                bh_mass[i] = max(0.0, prev_mass - 0.05)
                bh_dir[i]  = bh_dir[i - 1]

    # EMA crossover (12/26)
    def ema_series(arr, period):
        k = 2.0 / (period + 1)
        out = np.zeros_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = arr[i] * k + out[i - 1] * (1 - k)
        return out

    ema12 = ema_series(closes, 12)
    ema26 = ema_series(closes, 26)
    macd  = ema12 - ema26

    # Simple RSI (numpy vectorized approximation)
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Rolling mean via cumsum trick (approximate)
    rsi_period = 14
    avg_gain = np.convolve(gains,  np.ones(rsi_period) / rsi_period, mode="full")[:n - 1]
    avg_loss = np.convolve(losses, np.ones(rsi_period) / rsi_period, mode="full")[:n - 1]
    rs  = np.where(avg_loss > 1e-12, avg_gain / avg_loss, 100.0)
    rsi = 100 - 100 / (1 + rs)

    # Signal logic (aligned to closes[:-1], i.e. signal at bar i triggers at bar i+1)
    active_bh  = bh_mass[:-1] > 1.0
    bh_long    = active_bh & (bh_dir[:-1] > 0)
    bh_short   = active_bh & (bh_dir[:-1] < 0)

    macd_bull  = macd[:-1] > 0
    rsi_bull   = rsi > 45
    rsi_bear   = rsi < 55

    # Warm-up mask — skip first 50 bars
    warm      = np.arange(n - 1) < 50
    bh_long  &= ~warm
    bh_short &= ~warm

    long_entries  = bh_long  & macd_bull &  rsi_bull
    long_exits    = ~bh_long | ~macd_bull
    short_entries = bh_short & ~macd_bull & rsi_bear
    short_exits   = ~bh_short | macd_bull

    return long_entries, long_exits, short_entries, short_exits


# ---------------------------------------------------------------------------
# Fallback numpy portfolio simulator
# ---------------------------------------------------------------------------

def numpy_portfolio(closes: np.ndarray,
                    long_entries: np.ndarray,
                    long_exits:   np.ndarray,
                    short_entries: np.ndarray,
                    short_exits:   np.ndarray,
                    init_cash: float = 1_000_000.0,
                    fees: float = 0.0001):
    """
    Minimal vectorized portfolio simulator returning equity curve and stats.
    """
    n       = len(closes)
    equity  = np.full(n, init_cash)
    pos     = 0      # 1=long, -1=short, 0=flat
    shares  = 0.0
    cash    = init_cash

    trade_count = 0
    wins        = 0

    for i in range(1, n):
        ret = (closes[i] - closes[i - 1]) / closes[i - 1]

        # Mark-to-market
        if pos != 0:
            cash += pos * shares * (closes[i] - closes[i - 1])

        sig_idx = i - 1  # signal produced at bar i-1
        if sig_idx < len(long_entries):
            if pos == 0:
                if long_entries[sig_idx]:
                    shares = cash * 0.65 / closes[i]
                    cash  -= shares * closes[i] * (1 + fees)
                    pos    = 1
                    entry_price = closes[i]
                    trade_count += 1
                elif short_entries[sig_idx]:
                    shares = cash * 0.65 / closes[i]
                    cash  += shares * closes[i] * (1 - fees)
                    pos    = -1
                    entry_price = closes[i]
                    trade_count += 1
            elif pos == 1 and long_exits[sig_idx]:
                cash  += shares * closes[i] * (1 - fees)
                if closes[i] > entry_price:
                    wins += 1
                shares = 0.0
                pos    = 0
            elif pos == -1 and short_exits[sig_idx]:
                cash  -= shares * closes[i] * (1 + fees)
                if closes[i] < entry_price:
                    wins += 1
                shares = 0.0
                pos    = 0

        equity[i] = cash + (pos * shares * closes[i] if pos != 0 else 0.0)

    # Compute stats
    daily_rets = np.diff(equity) / equity[:-1]
    total_ret  = (equity[-1] / equity[0] - 1) * 100
    sharpe     = float(daily_rets.mean() / (daily_rets.std() + 1e-12) * np.sqrt(252))
    peak       = np.maximum.accumulate(equity)
    max_dd     = float(((equity - peak) / (peak + 1e-12)).min()) * -100
    win_rate   = wins / max(trade_count, 1)

    return {
        "total_return_pct": total_ret,
        "sharpe":           sharpe,
        "max_drawdown_pct": max_dd,
        "trade_count":      trade_count,
        "win_rate":         win_rate,
        "final_equity":     equity[-1],
        "equity_curve":     equity,
    }


# ---------------------------------------------------------------------------
# VBT-backed portfolio
# ---------------------------------------------------------------------------

def vbt_portfolio(closes: np.ndarray,
                  long_entries: np.ndarray,
                  long_exits:   np.ndarray,
                  short_entries: np.ndarray,
                  short_exits:   np.ndarray,
                  init_cash: float = 1_000_000.0,
                  fees: float = 0.0001):
    """
    Run portfolio via vectorbt. Returns stats dict.
    """
    import vectorbt as vbt

    # VBT needs same-length arrays aligned to 'close'
    n = len(closes)
    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)
    # shift signals: signal at bar i -> action at bar i+1
    le[1:] = long_entries
    lx[1:] = long_exits
    se[1:] = short_entries
    sx[1:] = short_exits

    pf = vbt.Portfolio.from_signals(
        close=closes,
        entries=le,
        exits=lx,
        short_entries=se,
        short_exits=sx,
        init_cash=init_cash,
        fees=fees,
        freq="1h",
    )

    st = pf.stats()
    return {
        "total_return_pct": float(pf.total_return() * 100),
        "sharpe":           float(pf.sharpe_ratio()),
        "max_drawdown_pct": float(pf.max_drawdown() * 100),
        "trade_count":      int(pf.trades.count()),
        "win_rate":         float(pf.trades.win_rate()),
        "final_equity":     float(pf.final_value()),
        "_vbt_stats":       st,
    }


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def run_sweep(closes: np.ndarray, param_name: str, use_vbt: bool):
    """Sweep one parameter and print a table."""
    CF_VALUES      = [0.002, 0.003, 0.005, 0.007, 0.009, 0.012]
    BH_FORM_VALUES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    BH_DECAY_VALUES = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]

    sweep_map = {
        "cf":       (CF_VALUES,       "cf",       1.5,  0.95),
        "bh_form":  (BH_FORM_VALUES,  "bh_form",  None, 0.95),
        "bh_decay": (BH_DECAY_VALUES, "bh_decay", 1.5,  None),
    }

    if param_name not in sweep_map:
        print(f"  Unknown sweep param: {param_name}. Choose: cf, bh_form, bh_decay")
        return

    values, pkey, default_bh_form, default_bh_decay = sweep_map[param_name]

    print(f"\nSWEEP: {param_name.upper()}")
    print(f"{'='*70}")
    print(f"  {'Value':>10}  {'Sharpe':>8}  {'Return%':>9}  {'MaxDD%':>7}  {'Trades':>7}  {'WinRate%':>9}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*9}")

    results = []
    for v in values:
        cf       = v       if pkey == "cf"       else 0.005
        bh_form  = v       if pkey == "bh_form"  else (default_bh_form or 1.5)
        bh_decay = v       if pkey == "bh_decay" else (default_bh_decay or 0.95)

        le, lx, se, sx = compute_srfm_signals(closes, cf, bh_form, bh_decay)

        try:
            if use_vbt:
                s = vbt_portfolio(closes, le, lx, se, sx)
            else:
                s = numpy_portfolio(closes, le, lx, se, sx)
        except Exception as e:
            s = {"sharpe": 0.0, "total_return_pct": 0.0,
                 "max_drawdown_pct": 0.0, "trade_count": 0, "win_rate": 0.0}

        results.append((v, s))
        print(f"  {v:>10.4g}  {s['sharpe']:>8.3f}  "
              f"{s['total_return_pct']:>+9.1f}  {s['max_drawdown_pct']:>7.1f}  "
              f"{s['trade_count']:>7}  {s['win_rate']*100:>8.1f}%")

    best_v, best_s = max(results, key=lambda x: x[1]["sharpe"])
    print(f"\n  Best {param_name} = {best_v:.4g}  (Sharpe={best_s['sharpe']:.3f})")


# ---------------------------------------------------------------------------
# Arena v2 comparison
# ---------------------------------------------------------------------------

def compare_vs_arena(bars, closes: np.ndarray, cf: float = 0.005):
    print(f"\nCOMPARISON: vbt/numpy vs arena_v2  (cf={cf})")
    print("=" * 65)

    # numpy/vbt timing
    t0 = time.perf_counter()
    le, lx, se, sx = compute_srfm_signals(closes, cf)
    try:
        import vectorbt as vbt
        s_vbt = vbt_portfolio(closes, le, lx, se, sx)
        backend = "vectorbt"
    except ImportError:
        s_vbt = numpy_portfolio(closes, le, lx, se, sx)
        backend = "numpy"
    t_fast = time.perf_counter() - t0

    # arena_v2 timing
    t1 = time.perf_counter()
    try:
        try:
            from tools.arena_v2 import run_v2
        except ImportError:
            from arena_v2 import run_v2
        cfg = {"cf": cf, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95}
        broker, _ = run_v2(bars, cfg, max_leverage=0.65)
        s_a2 = broker.stats()
        arena_ok = True
    except Exception as e:
        s_a2 = None
        arena_ok = False
        print(f"  arena_v2 error: {e}")
    t_arena = time.perf_counter() - t1

    print(f"  {'Metric':<22} {'arena_v2':>12} {backend:>12}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*12}")

    def row(label, key, fmt=".2f"):
        a2v = f"{s_a2[key]:{fmt}}" if arena_ok and s_a2 and key in s_a2 else "N/A"
        fbv = f"{s_vbt[key]:{fmt}}" if key in s_vbt else "N/A"
        print(f"  {label:<22} {a2v:>12} {fbv:>12}")

    row("Sharpe",         "sharpe",           ".3f")
    row("Total Return %", "total_return_pct", "+.1f")
    row("Max DD %",       "max_drawdown_pct", ".1f")
    row("Trades",         "trade_count",      "d")
    if "win_rate" in s_vbt:
        wr_a2 = f"{s_a2['win_rate']:.1%}" if arena_ok and s_a2 else "N/A"
        wr_fb = f"{s_vbt['win_rate']:.1%}"
        print(f"  {'Win Rate':<22} {wr_a2:>12} {wr_fb:>12}")

    print(f"\n  Timing:  arena_v2={t_arena:.2f}s   {backend}={t_fast:.3f}s  "
          f"(speedup={t_arena/max(t_fast,0.001):.1f}x)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="vectorbt-accelerated LARSA parameter sweep.")
    parser.add_argument("--csv",       default="data/NDX_hourly_poly.csv", help="Price CSV")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--n-bars",    type=int, default=20000)
    parser.add_argument("--cf",        type=float, default=0.005)
    parser.add_argument("--sweep",     help="Sweep a parameter: cf, bh_form, bh_decay")
    parser.add_argument("--compare",   action="store_true", help="Compare vbt vs arena_v2")
    parser.add_argument("--no-vbt",    action="store_true", help="Skip vectorbt, use numpy")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Load data
    if args.synthetic:
        try:
            try:
                from tools.arena_v2 import generate_synthetic
            except ImportError:
                from arena_v2 import generate_synthetic
            bars = generate_synthetic(args.n_bars, seed=42)
        except ImportError:
            from vbt_arena import load_ohlcv   # fallback won't be called
            bars = []
        print(f"  {len(bars)} synthetic bars")
    else:
        if not os.path.exists(args.csv):
            print(f"  ERROR: {args.csv} not found. Use --synthetic or provide a valid CSV.")
            sys.exit(1)
        bars = load_ohlcv(args.csv)
        print(f"  {len(bars)} bars from {args.csv}")

    closes, highs, lows, volumes, dates = bars_to_arrays(bars)

    use_vbt = not args.no_vbt
    if use_vbt:
        try:
            import vectorbt as vbt
            print(f"  vectorbt {vbt.__version__} detected")
        except ImportError:
            print("  vectorbt not installed — falling back to numpy portfolio")
            print("  Install with: pip install vectorbt")
            use_vbt = False

    if args.sweep:
        run_sweep(closes, args.sweep, use_vbt)

    elif args.compare:
        compare_vs_arena(bars, closes, cf=args.cf)

    else:
        # Default: single run with given cf
        print(f"\nRunning vbt_arena  cf={args.cf} ...")
        t0 = time.perf_counter()
        le, lx, se, sx = compute_srfm_signals(closes, args.cf)

        if use_vbt:
            try:
                s = vbt_portfolio(closes, le, lx, se, sx)
                backend = "vectorbt"
            except Exception as e:
                print(f"  vectorbt error: {e} — falling back to numpy")
                s = numpy_portfolio(closes, le, lx, se, sx)
                backend = "numpy"
        else:
            s = numpy_portfolio(closes, le, lx, se, sx)
            backend = "numpy"

        elapsed = time.perf_counter() - t0

        print(f"\n{'='*55}")
        print(f"  vbt_arena [{backend}]  cf={args.cf}")
        print(f"{'='*55}")
        print(f"  Return   : {s['total_return_pct']:>+.2f}%")
        print(f"  Max DD   : {s['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe   : {s['sharpe']:.3f}")
        print(f"  Trades   : {s['trade_count']}")
        print(f"  Win Rate : {s['win_rate']:.1%}")
        print(f"  Equity   : ${s['final_equity']:,.0f}")
        print(f"  Time     : {elapsed:.3f}s")
        print(f"{'='*55}\n")

        if "_vbt_stats" in s:
            print("  Full vectorbt stats:")
            print(s["_vbt_stats"].to_string())


if __name__ == "__main__":
    main()
