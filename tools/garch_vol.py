"""
garch_vol.py — GARCH(1,1) volatility forecasting for position sizing.

Fits GARCH(1,1) on historical returns to forecast next-bar volatility.
Better than rolling ATR for distinguishing regime vol from noise.

Usage:
    python tools/garch_vol.py --csv data/NDX_hourly_poly.csv
    python tools/garch_vol.py --plot         # ASCII vol chart
    python tools/garch_vol.py --backtest     # show if GARCH sizing improves Sharpe
    python tools/garch_vol.py --synthetic    # run on synthetic bars
"""

import argparse
import csv as _csv
import math
import os
import sys

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


def compute_returns(closes: np.ndarray) -> np.ndarray:
    """Log returns in percentage form (as used by arch_model)."""
    return np.diff(np.log(closes)) * 100.0


# ---------------------------------------------------------------------------
# Rolling ATR (20-bar)
# ---------------------------------------------------------------------------

def rolling_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 20) -> np.ndarray:
    n = len(closes)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i]  - closes[i - 1]))
    atr = np.zeros(n)
    for i in range(period, n):
        atr[i] = tr[i - period:i].mean()
    # Convert ATR to vol % (ATR / close)
    vol = np.where(closes > 0, atr / closes, 0.0)
    return vol


# ---------------------------------------------------------------------------
# GARCH implementation (arch package or pure-numpy fallback)
# ---------------------------------------------------------------------------

class GARCHResult:
    """Container for GARCH fit results."""
    def __init__(self):
        self.omega  = None
        self.alpha  = None
        self.beta   = None
        self.nu     = None          # degrees of freedom (t-dist); None if Gaussian
        self.loglik = None
        self.aic    = None
        self.bic    = None
        self.cond_vol: np.ndarray = None   # conditional volatility series (% scale)
        self.method = "arch"


def fit_garch_arch(returns_pct: np.ndarray) -> GARCHResult:
    """
    Fit GARCH(1,1) with Student-t residuals using the arch package.
    """
    from arch import arch_model

    am  = arch_model(returns_pct, vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(update_freq=0, disp="off")

    gr = GARCHResult()
    gr.omega  = float(res.params.get("omega",  res.params.iloc[0]))
    gr.alpha  = float(res.params.get("alpha[1]", res.params.iloc[1]))
    gr.beta   = float(res.params.get("beta[1]",  res.params.iloc[2]))
    try:
        gr.nu = float(res.params.get("nu", res.params.iloc[-1]))
    except Exception:
        gr.nu = None
    gr.loglik = float(res.loglikelihood)
    gr.aic    = float(res.aic)
    gr.bic    = float(res.bic)
    gr.cond_vol = res.conditional_volatility.values  # percent scale
    gr.method   = "arch"
    return gr


def fit_garch_numpy(returns_pct: np.ndarray) -> GARCHResult:
    """
    Pure-numpy GARCH(1,1) fit via simple MLE coordinate ascent.
    Used as fallback when 'arch' is not installed.
    """
    # Initial parameter guess
    var0  = float(np.var(returns_pct))
    omega = var0 * 0.05
    alpha = 0.08
    beta  = 0.88

    # Compute conditional variance series
    def cond_var(om, al, be):
        n   = len(returns_pct)
        h   = np.zeros(n)
        h[0] = var0
        for i in range(1, n):
            h[i] = om + al * returns_pct[i - 1] ** 2 + be * h[i - 1]
            h[i] = max(h[i], 1e-8)
        return h

    def neg_loglik(om, al, be):
        if om <= 0 or al < 0 or be < 0 or al + be >= 1.0:
            return 1e9
        h = cond_var(om, al, be)
        ll = -0.5 * np.sum(np.log(h) + returns_pct ** 2 / h)
        return -ll

    # Simple grid + gradient descent (SciPy not required)
    try:
        from scipy.optimize import minimize

        def obj(params):
            return neg_loglik(params[0], params[1], params[2])

        res = minimize(
            obj,
            x0=[omega, alpha, beta],
            method="L-BFGS-B",
            bounds=[(1e-8, None), (0, 0.5), (0, 0.99)],
        )
        if res.success:
            omega, alpha, beta = res.x
    except ImportError:
        # Very rough coordinate descent fallback
        best_ll = neg_loglik(omega, alpha, beta)
        for om in [var0 * f for f in [0.01, 0.03, 0.05, 0.08]]:
            for al in [0.05, 0.08, 0.12, 0.15]:
                for be in [0.80, 0.85, 0.88, 0.91, 0.93]:
                    ll = neg_loglik(om, al, be)
                    if ll < best_ll:
                        best_ll = ll
                        omega, alpha, beta = om, al, be

    h = cond_var(omega, alpha, beta)
    ll = float(np.sum(-0.5 * (np.log(h) + returns_pct ** 2 / h)))
    n_params = 3
    n = len(returns_pct)
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * math.log(n)

    gr = GARCHResult()
    gr.omega   = omega
    gr.alpha   = alpha
    gr.beta    = beta
    gr.nu      = None
    gr.loglik  = ll
    gr.aic     = aic
    gr.bic     = bic
    gr.cond_vol = np.sqrt(h)   # still in percent scale
    gr.method   = "numpy_fallback"
    return gr


def rolling_garch_forecast(returns_pct: np.ndarray, train_frac: float = 0.70):
    """
    Rolling 1-step-ahead GARCH vol forecasts on test portion.

    Returns:
        gr       : GARCHResult from training period
        garch_vol: np.ndarray of length len(returns_pct) - n_train
                   forecast vol in decimal (e.g. 0.001 = 0.1%)
        n_train  : number of training bars
    """
    n       = len(returns_pct)
    n_train = int(n * train_frac)

    train_ret = returns_pct[:n_train]
    test_ret  = returns_pct[n_train:]

    try:
        gr = fit_garch_arch(train_ret)
        use_arch = True
    except ImportError:
        gr = fit_garch_numpy(train_ret)
        use_arch = False

    # Rolling forecast on test
    if use_arch:
        try:
            from arch import arch_model
            # Refit on full data up to test start, roll forward
            am   = arch_model(returns_pct[:n_train], vol="GARCH", p=1, q=1, dist="t")
            res  = am.fit(update_freq=0, disp="off")

            forecasts_list = []
            # Use simulation-based rolling forecast
            h_prev = float(res.conditional_volatility.iloc[-1]) ** 2
            r_prev = float(train_ret[-1])

            om = gr.omega; al = gr.alpha; be = gr.beta
            for r in test_ret:
                h_next = om + al * r_prev ** 2 + be * h_prev
                forecasts_list.append(math.sqrt(max(h_next, 1e-8)))
                h_prev = h_next
                r_prev = r

            garch_vol_pct = np.array(forecasts_list)
        except Exception:
            # Fallback: propagate conditional variance manually
            garch_vol_pct = _propagate_garch(gr, train_ret, test_ret)
    else:
        garch_vol_pct = _propagate_garch(gr, train_ret, test_ret)

    # Convert from % to decimal
    garch_vol_dec = garch_vol_pct / 100.0
    return gr, garch_vol_dec, n_train


def _propagate_garch(gr: GARCHResult, train_ret: np.ndarray, test_ret: np.ndarray):
    """Propagate GARCH parameters forward through test returns."""
    h = float(np.var(train_ret))
    r_prev = float(train_ret[-1])
    om = gr.omega; al = gr.alpha; be = gr.beta
    out = []
    for r in test_ret:
        h = om + al * r_prev ** 2 + be * h
        h = max(h, 1e-8)
        out.append(math.sqrt(h))
        r_prev = r
    return np.array(out)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def garch_position_multiplier(garch_vol: np.ndarray,
                               target_vol: float = 0.015,
                               min_mult: float = 0.5,
                               max_mult: float = 2.0) -> np.ndarray:
    """Scale position by target_vol / garch_vol."""
    mult = np.where(garch_vol > 1e-8, target_vol / garch_vol, 1.0)
    return np.clip(mult, min_mult, max_mult)


# ---------------------------------------------------------------------------
# Backtest: fixed vs GARCH sizing
# ---------------------------------------------------------------------------

def run_backtest(closes: np.ndarray, garch_vol: np.ndarray, n_train: int,
                 base_leverage: float = 0.65, target_vol: float = 0.015):
    """
    Simple long-only backtest on the test portion.
    Compare fixed leverage vs GARCH-scaled leverage.
    """
    test_closes = closes[n_train:]
    n_test = len(test_closes)
    if n_test < 10:
        return None

    # Limit to overlapping region
    n = min(n_test, len(garch_vol) + 1)
    closes_t = test_closes[:n]
    returns  = np.diff(closes_t) / closes_t[:-1]   # arithmetic returns

    n_ret = len(returns)
    gv    = garch_vol[:n_ret]
    mult  = garch_position_multiplier(gv, target_vol)

    # Fixed sizing
    equity_fixed  = np.ones(n_ret + 1)
    for i, r in enumerate(returns):
        equity_fixed[i + 1] = equity_fixed[i] * (1 + base_leverage * r)

    # GARCH sizing
    equity_garch  = np.ones(n_ret + 1)
    for i, r in enumerate(returns):
        lev = base_leverage * float(mult[i])
        equity_garch[i + 1] = equity_garch[i] * (1 + lev * r)

    def _stats(equity):
        dr = np.diff(equity) / equity[:-1]
        total_ret = (equity[-1] - 1) * 100
        sharpe    = float(dr.mean() / (dr.std() + 1e-12) * math.sqrt(252))
        peak      = np.maximum.accumulate(equity)
        max_dd    = float(((equity - peak) / (peak + 1e-12)).min()) * -100
        return total_ret, sharpe, max_dd

    tr_f, sh_f, dd_f = _stats(equity_fixed)
    tr_g, sh_g, dd_g = _stats(equity_garch)

    return {
        "fixed":  {"sharpe": sh_f, "return_pct": tr_f, "max_dd_pct": dd_f},
        "garch":  {"sharpe": sh_g, "return_pct": tr_g, "max_dd_pct": dd_g},
        "sharpe_improvement": sh_g - sh_f,
        "equity_fixed": equity_fixed,
        "equity_garch": equity_garch,
        "garch_vol":    gv,
        "mult":         mult,
    }


# ---------------------------------------------------------------------------
# ASCII vol chart
# ---------------------------------------------------------------------------

def ascii_vol_chart(garch_vol: np.ndarray, rolling_atr_vol: np.ndarray,
                    n_points: int = 80):
    """Print ASCII chart comparing GARCH vs ATR vol."""
    step = max(1, len(garch_vol) // n_points)
    gv   = garch_vol[::step][:n_points]
    av   = rolling_atr_vol[::step][:n_points]

    vmax = max(gv.max(), av.max(), 1e-6)
    height = 12

    print("\nVOLATILITY CHART (GARCH=* vs ATR=.)")
    print("=" * (n_points + 10))

    for row in range(height, -1, -1):
        thresh = vmax * row / height
        line   = ""
        for i in range(len(gv)):
            g_above = gv[i] >= thresh
            a_above = av[i] >= thresh
            if g_above and a_above:
                line += "+"
            elif g_above:
                line += "*"
            elif a_above:
                line += "."
            else:
                line += " "
        label = f"{thresh*100:5.3f}%|" if row % 3 == 0 else "      |"
        print(label + line)
    print("      +" + "-" * len(gv))
    print(f"\n  * = GARCH vol   . = ATR vol   + = both")


# ---------------------------------------------------------------------------
# Vol regime stats
# ---------------------------------------------------------------------------

def vol_regime_stats(garch_vol: np.ndarray, mult: np.ndarray):
    low_mask  = garch_vol < 0.0005
    med_mask  = (garch_vol >= 0.0005) & (garch_vol < 0.0015)
    high_mask = garch_vol >= 0.0015

    n = len(garch_vol)

    def stats(mask):
        pct      = mask.sum() / max(n, 1) * 100
        avg_mult = float(mult[mask].mean()) if mask.any() else 0.0
        return pct, avg_mult

    lp, lm = stats(low_mask)
    mp, mm = stats(med_mask)
    hp, hm = stats(high_mask)

    print(f"\n  VOL REGIME STATS:")
    print(f"  Low  vol (<0.05% hourly):   {lp:5.1f}% of bars  avg_position_mult={lm:.2f}")
    print(f"  Med  vol (0.05-0.15%):      {mp:5.1f}% of bars  avg_position_mult={mm:.2f}")
    print(f"  High vol (>0.15%):          {hp:5.1f}% of bars  avg_position_mult={hm:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GARCH(1,1) volatility forecasting for LARSA.")
    parser.add_argument("--csv",       default="data/NDX_hourly_poly.csv", help="Price CSV")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic bars")
    parser.add_argument("--n-bars",    type=int, default=35000, help="Synthetic bar count")
    parser.add_argument("--plot",      action="store_true", help="Show ASCII vol chart")
    parser.add_argument("--backtest",  action="store_true", help="Run GARCH vs fixed sizing backtest")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Load bars
    if args.synthetic:
        try:
            try:
                from tools.arena_v2 import generate_synthetic
            except ImportError:
                from arena_v2 import generate_synthetic
            bars = generate_synthetic(args.n_bars, seed=42)
        except ImportError:
            # Manual fallback
            rng = np.random.default_rng(42)
            price = 4500.0
            bars  = []
            for i in range(args.n_bars):
                ret   = float(rng.normal(0.0001, 0.001))
                close = price * (1 + ret)
                bars.append({"date": f"bar_{i:06d}", "open": price,
                             "high": close * 1.001, "low": close * 0.999,
                             "close": close, "volume": 50000.0})
                price = close
        print(f"  {len(bars)} synthetic bars")
    else:
        if not os.path.exists(args.csv):
            print(f"  ERROR: {args.csv} not found. Use --synthetic or provide a valid CSV.")
            sys.exit(1)
        bars = load_ohlcv(args.csv)
        print(f"  {len(bars)} bars from {args.csv}")

    closes = np.array([b["close"] for b in bars], dtype=np.float64)
    highs  = np.array([b["high"]  for b in bars], dtype=np.float64)
    lows   = np.array([b["low"]   for b in bars], dtype=np.float64)

    # Returns
    rets_pct = compute_returns(closes)  # percent scale for GARCH

    n_total = len(rets_pct)
    n_train = int(n_total * 0.70)
    n_test  = n_total - n_train

    # Check arch
    use_arch = False
    try:
        import arch as _arch
        use_arch = True
        arch_ver = getattr(_arch, "__version__", "?")
    except ImportError:
        arch_ver = "not installed"

    print(f"\nGARCH(1,1) VOLATILITY FORECASTING")
    print("=" * 44)
    print(f"Model:    GARCH(1,1) {'with Student-t residuals' if use_arch else '(numpy fallback)'}")
    print(f"arch pkg: {arch_ver}")
    print(f"Training: {n_train:,} bars   Testing: {n_test:,} bars")

    # Fit and forecast
    gr, garch_vol_dec, n_train = rolling_garch_forecast(rets_pct)

    print(f"\nFIT QUALITY  (method={gr.method}):")
    if gr.loglik is not None:
        print(f"  Log-likelihood:  {gr.loglik:>12,.1f}")
    if gr.aic is not None:
        print(f"  AIC:             {gr.aic:>12,.1f}")
    if gr.bic is not None:
        print(f"  BIC:             {gr.bic:>12,.1f}")
    print(f"  alpha (ARCH):    {gr.alpha:.4f}  <- how quickly vol responds to shocks")
    print(f"  beta  (GARCH):   {gr.beta:.4f}  <- vol persistence (high = slow mean reversion)")
    print(f"  omega (const):   {gr.omega:.8f}")
    if gr.nu is not None:
        print(f"  nu (df, t-dist): {gr.nu:.2f}  <- fat tails confirmed" if gr.nu < 10
              else f"  nu (df, t-dist): {gr.nu:.2f}")
    print(f"  alpha+beta:      {gr.alpha+gr.beta:.4f}  <- persistence (near 1 = high)")

    # Rolling ATR for comparison
    atr_vol = rolling_atr(highs, lows, closes, period=20)
    # align to test portion (offset by n_train+1 for returns)
    atr_test = atr_vol[n_train + 1: n_train + 1 + len(garch_vol_dec)]

    # Position multipliers
    mult = garch_position_multiplier(garch_vol_dec)
    vol_regime_stats(garch_vol_dec, mult)

    # Vol comparison
    if len(atr_test) > 0 and len(garch_vol_dec) > 0:
        n_cmp = min(len(atr_test), len(garch_vol_dec))
        corr  = float(np.corrcoef(garch_vol_dec[:n_cmp], atr_test[:n_cmp])[0, 1])
        print(f"\n  GARCH vs ATR correlation: {corr:.3f}")
        print(f"  GARCH vol (test):  mean={garch_vol_dec.mean()*100:.4f}%  "
              f"std={garch_vol_dec.std()*100:.4f}%  "
              f"max={garch_vol_dec.max()*100:.4f}%")
        if len(atr_test) > 0:
            print(f"  ATR   vol (test):  mean={atr_test.mean()*100:.4f}%  "
                  f"std={atr_test.std()*100:.4f}%  "
                  f"max={atr_test.max()*100:.4f}%")

    # ASCII chart
    if args.plot and len(atr_test) > 0:
        n_cmp = min(len(atr_test), len(garch_vol_dec))
        ascii_vol_chart(garch_vol_dec[:n_cmp], atr_test[:n_cmp])

    # Backtest
    if args.backtest:
        bt = run_backtest(closes, garch_vol_dec, n_train)
        if bt:
            print(f"\nBACKTEST (GARCH position scaling vs fixed):")
            f = bt["fixed"]
            g = bt["garch"]
            print(f"  Fixed sizing:  Sharpe={f['sharpe']:.2f}  "
                  f"Return={f['return_pct']:+.1f}%  DD={f['max_dd_pct']:.1f}%")
            print(f"  GARCH scaling: Sharpe={g['sharpe']:.2f}  "
                  f"Return={g['return_pct']:+.1f}%  DD={g['max_dd_pct']:.1f}%")
            delta = bt["sharpe_improvement"]
            sign  = "+" if delta >= 0 else ""
            print(f"  Improvement:   {sign}{delta:.2f} Sharpe points")

    # Save outputs
    # Markdown report
    lines = [
        f"# GARCH(1,1) Volatility Analysis\n",
        f"- Method: {gr.method}",
        f"- Training bars: {n_train:,}  Testing bars: {n_test:,}",
        f"- alpha: {gr.alpha:.4f}",
        f"- beta:  {gr.beta:.4f}",
        f"- omega: {gr.omega:.8f}",
        f"- alpha+beta: {gr.alpha+gr.beta:.4f}",
    ]
    if gr.loglik:
        lines.append(f"- LogLik: {gr.loglik:.1f}  AIC: {gr.aic:.1f}  BIC: {gr.bic:.1f}")
    if gr.nu:
        lines.append(f"- nu (t-dist df): {gr.nu:.2f}")

    md_path = "results/garch_analysis.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Saved -> {md_path}")

    # CSV: bar_index, garch_vol, rolling_atr
    csv_path = "results/garch_vol_series.csv"
    n_save   = len(garch_vol_dec)
    atr_save = atr_test[:n_save] if len(atr_test) >= n_save else np.zeros(n_save)
    with open(csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["bar_index", "garch_vol", "rolling_atr"])
        for i in range(n_save):
            writer.writerow([n_train + i, f"{garch_vol_dec[i]:.8f}",
                             f"{atr_save[i]:.8f}"])
    print(f"  Saved -> {csv_path}  ({n_save:,} rows)")


if __name__ == "__main__":
    main()
