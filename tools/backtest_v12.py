"""
backtest_v12.py — Full local backtest of LARSA v12 on real hourly data.

Loads data/ES_hourly_real.csv, NQ_hourly_real.csv, YM_hourly_real.csv
and runs the complete v12 strategy bar-by-bar:
  - BH physics with regime-scaled CF (CF×3 in BULL)
  - Multi-timeframe score (15m simulated from hourly, 1h, 1d)
  - Vol-targeted sizing (correlation-adjusted portfolio risk budget)
  - BEAR long-suppression + BULL short-suppression gates
  - 4-bar minimum hold, hourly execution gate, portfolio cap

Outputs:
  Terminal: per-year stats, regime breakdown, Sharpe, MaxDD
  results/backtest_v12.json
  results/backtest_v12.png
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Strategy constants (identical to v12) ────────────────────────────────────
CF_1H = {"ES": 0.001, "NQ": 0.0012, "YM": 0.0008}
CF_15M = {"ES": 0.0003, "NQ": 0.0004, "YM": 0.00025}
CF_1D  = {"ES": 0.005,  "NQ": 0.006,  "YM": 0.004}
TF_CAP = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}
MIN_HOLD = 4
N_INST = 3
INST_CORR = 0.90
PORT_RISK = 0.01
CORR_FACTOR = math.sqrt(N_INST + N_INST * (N_INST - 1) * INST_CORR)  # 2.898
PER_INST_RISK = PORT_RISK / CORR_FACTOR  # 0.00345
START_EQUITY = 1_000_000.0
SYMS = ["ES", "NQ", "YM"]

BULL_CF_SCALE = 3.0


# ── Indicators ────────────────────────────────────────────────────────────────
class WilderEMA:
    def __init__(self, period):
        self.p = period
        self.k = 1.0 / period  # Wilder's smoothing
        self.v = None

    def update(self, x):
        self.v = x if self.v is None else self.v + self.k * (x - self.v)
        return self.v or 0.0


class EMA:
    def __init__(self, period):
        self.p = period
        self.k = 2.0 / (period + 1)
        self.v = None

    def update(self, x):
        self.v = x if self.v is None else self.v + self.k * (x - self.v)
        return self.v or 0.0


class WilderATR:
    def __init__(self, period=14):
        self.p = period
        self.k = 1.0 / period
        self.v = None
        self._pc = None

    def update(self, h, l, c):
        if self._pc is None:
            self._pc = c
            return 0.0
        tr = max(h - l, abs(h - self._pc), abs(l - self._pc))
        self.v = tr if self.v is None else self.v + self.k * (tr - self.v)
        self._pc = c
        return self.v or 0.0

    @property
    def value(self):
        return self.v or 0.0


class ADX:
    def __init__(self, p=14):
        self.p = p
        self.k = 1.0 / p
        self._pDM = 0.0; self._nDM = 0.0; self._TR = 0.0; self._ADX = 0.0
        self._prev_h = None; self._prev_l = None; self._prev_c = None

    def update(self, h, l, c):
        if self._prev_c is None:
            self._prev_h = h; self._prev_l = l; self._prev_c = c
            return 0.0
        ph = self._prev_h; pl = self._prev_l; pc = self._prev_c
        tr = max(h - l, abs(h - pc), abs(l - pc))
        pdm = max(h - ph, 0) if (h - ph) > (pl - l) else 0.0
        ndm = max(pl - l, 0) if (pl - l) > (h - ph) else 0.0
        self._pDM = self._pDM * (1 - self.k) + pdm * self.k
        self._nDM = self._nDM * (1 - self.k) + ndm * self.k
        self._TR  = self._TR  * (1 - self.k) + tr  * self.k
        if self._TR > 0:
            pDI = 100 * self._pDM / self._TR
            nDI = 100 * self._nDM / self._TR
            d = pDI + nDI
            dx = 100 * abs(pDI - nDI) / d if d > 0 else 0.0
            self._ADX = self._ADX * (1 - self.k) + dx * self.k
        self._prev_h = h; self._prev_l = l; self._prev_c = c
        return self._ADX

    @property
    def value(self):
        return self._ADX


# ── BH physics ────────────────────────────────────────────────────────────────
class BH:
    """Black-hole mass accumulator for one resolution."""
    def __init__(self, cf, warmup_bars):
        self.cf = cf
        self.wu = warmup_bars
        self.mass = 0.0
        self.ctl = 0
        self.active = False
        self.direction = 0
        self.prices = []
        self.bc = 0

    def update(self, c, cf_scale=1.0):
        self.bc += 1
        self.prices.append(c)
        if len(self.prices) < 2:
            return
        eff_cf = self.cf * cf_scale
        beta = abs(c - self.prices[-2]) / (self.prices[-2] + 1e-9) / (eff_cf + 1e-9)
        was = self.active
        if beta < 1.0:
            self.ctl += 1
            sb = min(2.0, 1.0 + self.ctl * 0.1)
            self.mass = self.mass * 0.97 + 0.03 * 1.0 * sb
        else:
            self.ctl = 0
            self.mass *= 0.95
        form_thresh = 1.0 if was else 1.5
        self.active = self.mass > form_thresh and self.ctl >= 3
        if not was and self.active:
            lb = min(20, len(self.prices) - 1)
            self.direction = 1 if c > self.prices[-1 - lb] else -1
        elif was and not self.active:
            self.direction = 0
        if self.bc < self.wu:
            self.active = False
            self.direction = 0

    def get_direction(self):
        if self.direction != 0:
            return self.direction
        if len(self.prices) >= 5:
            return 1 if self.prices[-1] > self.prices[-5] else -1
        return 0


# ── Regime detection ─────────────────────────────────────────────────────────
REGIME_BULL = "BULL"
REGIME_BEAR = "BEAR"
REGIME_SIDEWAYS = "SIDEWAYS"
REGIME_HIGH_VOL = "HIGH_VOL"


class RegimeDetector:
    def __init__(self):
        self.e12  = EMA(12)
        self.e26  = EMA(26)
        self.e50  = EMA(50)
        self.e200 = EMA(200)
        self.adx  = ADX(14)
        self.atr  = WilderATR(14)
        self.atr_hist = []
        self.regime = REGIME_SIDEWAYS
        self.cf_scale = 1.0
        self.rhb = 0
        self.bc = 0

    def update(self, o, h, l, c):
        self.bc += 1
        e12  = self.e12.update(c)
        e26  = self.e26.update(c)
        e50  = self.e50.update(c)
        e200 = self.e200.update(c)
        adx  = self.adx.update(h, l, c)
        atr  = self.atr.update(h, l, c)

        self.atr_hist.append(atr)
        if len(self.atr_hist) > 50:
            self.atr_hist.pop(0)

        if self.bc < 200:  # not enough data for e200
            return

        atr_ratio = 1.0
        if len(self.atr_hist) >= 20:
            mean_atr = np.mean(self.atr_hist)
            atr_ratio = atr / mean_atr if mean_atr > 0 else 1.0

        self.rhb += 1
        if atr_ratio >= 1.5:
            nr = REGIME_HIGH_VOL
        elif c > e200 and e12 > e26:
            full_stack = e12 > e26 > e50 > e200
            if adx > (14 if full_stack else 18):
                nr = REGIME_BULL
            else:
                nr = REGIME_SIDEWAYS
        elif c < e200 and e12 < e26:
            full_stack = e200 > e50 > e26 > e12
            if adx > (14 if full_stack else 18):
                nr = REGIME_BEAR
            else:
                nr = REGIME_SIDEWAYS
        else:
            nr = REGIME_SIDEWAYS

        if nr != self.regime:
            self.rhb = 0
            self.regime = nr

        self.cf_scale = BULL_CF_SCALE if self.regime == REGIME_BULL else 1.0


# ── Per-instrument state ──────────────────────────────────────────────────────
class Instrument:
    def __init__(self, sym):
        self.sym = sym
        self.bh_15m = BH(CF_15M[sym], warmup_bars=400)
        self.bh_1h  = BH(CF_1H[sym],  warmup_bars=120)
        self.bh_1d  = BH(CF_1D[sym],  warmup_bars=30)
        self.regime = RegimeDetector()
        self.atr_1h = WilderATR(14)
        self.last_target = 0.0
        self.bars_held = 0

    def tf_score(self):
        return (4 * int(self.bh_1d.active) +
                2 * int(self.bh_1h.active) +
                int(self.bh_15m.active))

    def direction(self):
        if self.bh_1d.active: return self.bh_1d.get_direction()
        if self.bh_1h.active: return self.bh_1h.get_direction()
        if self.bh_15m.active: return self.bh_15m.get_direction()
        return 0


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    dfs = {}
    for sym in SYMS:
        path = os.path.join(DATA, f"{sym}_hourly_real.csv")
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        dfs[sym] = df
    # Align on common timestamps
    common = set(dfs["ES"]["date"])
    for sym in SYMS:
        common &= set(dfs[sym]["date"])
    common = sorted(common)
    aligned = {}
    for sym in SYMS:
        df = dfs[sym].set_index("date").loc[common].reset_index()
        aligned[sym] = df
    return aligned, common


# ── Main backtest ─────────────────────────────────────────────────────────────
def run():
    print(f"{'═'*65}")
    print("  LARSA v12 — Local Backtest on Real Hourly Data")
    print(f"{'═'*65}\n")

    data, timestamps = load_data()
    n = len(timestamps)
    print(f"  Loaded {n} aligned hourly bars per symbol")
    print(f"  Range: {timestamps[0].strftime('%Y-%m-%d')} → {timestamps[-1].strftime('%Y-%m-%d')}\n")

    instruments = {sym: Instrument(sym) for sym in SYMS}

    equity = START_EQUITY
    peak   = START_EQUITY
    positions = {sym: 0.0 for sym in SYMS}  # fraction of equity
    prices    = {sym: 0.0 for sym in SYMS}

    equity_curve = []
    daily_returns = []
    last_daily_equity = equity
    regime_counts = {r: 0 for r in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS, REGIME_HIGH_VOL]}
    regime_pnl    = {r: 0.0 for r in regime_counts}
    bar_count_1d  = 0  # approximate: every 6.5 bars = 1 day

    for i, ts in enumerate(timestamps):
        bars = {sym: data[sym].iloc[i] for sym in SYMS}

        # ── Mark-to-market ───────────────────────────────────────────────────
        bar_pnl = 0.0
        for sym in SYMS:
            b = bars[sym]
            c = float(b["close"])
            pos = positions[sym]
            if abs(pos) > 1e-6 and prices[sym] > 0:
                ret = pos * (c - prices[sym]) / prices[sym]
                equity += ret * equity
                bar_pnl += ret
            prices[sym] = c
        equity = max(equity, 1.0)
        if equity > peak:
            peak = equity

        # ── Update instruments ───────────────────────────────────────────────
        for sym in SYMS:
            b = bars[sym]
            o = float(b["open"]); h = float(b["high"])
            l = float(b["low"]);  c = float(b["close"])
            pc = prices[sym]
            inst = instruments[sym]

            # Simulate 15m sub-bars from OHLC
            for sub in [o, (o + h) / 2, (l + c) / 2, c]:
                inst.bh_15m.update(sub)

            inst.bh_1h.update(c, cf_scale=inst.regime.cf_scale)
            inst.atr_1h.update(h, l, c)
            inst.regime.update(o, h, l, c)

            # Daily bar: approximate every 6 hourly bars
            if inst.bh_1h.bc % 6 == 0:
                inst.bh_1d.update(c)

        # ── Execution (hourly gate is implicit — one bar = one hour) ─────────
        raw_targets = {}
        for sym in SYMS:
            inst = instruments[sym]
            tfs = inst.tf_score()
            ceiling = TF_CAP[tfs]

            if tfs == 1 and abs(inst.last_target) < 0.01:
                ceiling = 0.0

            if ceiling == 0.0:
                tgt = 0.0
            else:
                d = inst.direction()
                if d == 0:
                    tgt = 0.0
                else:
                    atr = inst.atr_1h.value
                    p   = prices[sym]
                    if atr > 0 and p > 0:
                        dv  = (atr / p) * math.sqrt(6.5)
                        raw = PER_INST_RISK / (dv + 1e-9)
                        cap = min(raw, ceiling)
                    else:
                        cap = ceiling
                    tgt = cap * d

                    # BEAR gate: no longs in bear
                    reg = inst.regime.regime
                    if reg == REGIME_BEAR and tgt > 0 and inst.regime.rhb > 5:
                        tgt = 0.0
                    # BULL gate: no shorts in bull (v12)
                    if reg == REGIME_BULL and tgt < 0 and inst.regime.rhb > 5:
                        tgt = 0.0

            # Minimum hold gate
            is_reversal = (
                not np.isclose(inst.last_target, 0.0) and
                not np.isclose(tgt, 0.0) and
                np.sign(tgt) != np.sign(inst.last_target)
            )
            if is_reversal and inst.bars_held < MIN_HOLD:
                tgt = inst.last_target

            raw_targets[sym] = tgt

        # Portfolio cap
        total_exp = sum(abs(v) for v in raw_targets.values())
        scale = 1.0 / total_exp if total_exp > 1.0 else 1.0

        for sym in SYMS:
            inst = instruments[sym]
            tgt = float(raw_targets[sym] * scale)
            if abs(tgt - inst.last_target) > 0.02:
                if np.isclose(tgt, 0.0) or np.sign(tgt) != np.sign(inst.last_target):
                    inst.bars_held = 0
                inst.last_target = tgt
                positions[sym] = tgt
            if abs(inst.last_target) > 0.02:
                inst.bars_held += 1

        # ── Record ───────────────────────────────────────────────────────────
        equity_curve.append((ts, equity))
        # Regime pnl tracking (use ES regime as proxy)
        reg = instruments["ES"].regime.regime
        regime_counts[reg] += 1
        regime_pnl[reg] += bar_pnl

        # Daily return (approx every 6.5 bars)
        bar_count_1d += 1
        if bar_count_1d >= 7:
            bar_count_1d = 0
            daily_returns.append((equity - last_daily_equity) / last_daily_equity)
            last_daily_equity = equity

    # ── Stats ─────────────────────────────────────────────────────────────────
    equities = np.array([e for _, e in equity_curve])
    times    = [t for t, _ in equity_curve]

    total_return = (equity - START_EQUITY) / START_EQUITY
    max_dd = 0.0
    pk = equities[0]
    for v in equities:
        pk = max(pk, v)
        max_dd = max(max_dd, (pk - v) / pk)

    dr = np.array(daily_returns)
    sharpe = float(dr.mean() / (dr.std() + 1e-9) * math.sqrt(252)) if len(dr) > 1 else 0.0
    sortino_neg = dr[dr < 0]
    sortino = float(dr.mean() / (sortino_neg.std() + 1e-9) * math.sqrt(252)) if len(sortino_neg) > 1 else 0.0
    win_days = float(np.sum(dr > 0) / len(dr)) if len(dr) > 0 else 0.0

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"{'─'*65}")
    print("  PERFORMANCE SUMMARY")
    print(f"{'─'*65}")
    print(f"  Start equity:   ${START_EQUITY:>12,.0f}")
    print(f"  Final equity:   ${equity:>12,.0f}")
    print(f"  Total return:   {total_return:>+.1%}")
    print(f"  Max drawdown:   {max_dd:.2%}")
    print(f"  Sharpe ratio:   {sharpe:.2f}")
    print(f"  Sortino ratio:  {sortino:.2f}")
    print(f"  Win rate (day): {win_days:.1%}")
    print(f"  Total bars:     {n:,}")

    print(f"\n{'─'*65}")
    print("  REGIME BREAKDOWN")
    print(f"{'─'*65}")
    print(f"  {'Regime':<12} {'Bars':>6}  {'% Time':>7}  {'Cum PnL':>9}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*7}  {'─'*9}")
    for reg in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS, REGIME_HIGH_VOL]:
        cnt = regime_counts[reg]
        pct = cnt / n * 100
        pnl = regime_pnl[reg] * 100
        print(f"  {reg:<12} {cnt:>6}  {pct:>6.1f}%  {pnl:>+8.2f}%")

    # Per-year breakdown
    print(f"\n{'─'*65}")
    print("  PER-YEAR RETURNS")
    print(f"{'─'*65}")
    year_equity = {}
    for ts, eq in equity_curve:
        y = ts.year
        if y not in year_equity:
            year_equity[y] = [eq, eq]
        year_equity[y][1] = eq
    print(f"  {'Year':>4}  {'Return':>8}")
    for y in sorted(year_equity):
        start_eq, end_eq = year_equity[y]
        ret = (end_eq - start_eq) / start_eq
        print(f"  {y:>4}  {ret:>+7.1%}")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "start_equity": START_EQUITY,
        "final_equity": float(equity),
        "total_return_pct": float(total_return * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "win_rate_daily": float(win_days),
        "n_bars": n,
        "regime_counts": regime_counts,
        "regime_pnl_pct": {k: float(v * 100) for k, v in regime_pnl.items()},
        "per_year": {str(y): float((year_equity[y][1] - year_equity[y][0]) / year_equity[y][0] * 100)
                     for y in sorted(year_equity)},
    }
    json_path = os.path.join(RESULTS, "backtest_v12.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {json_path}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("LARSA v12 — Real Data Backtest", fontsize=13, fontweight="bold")

        ts_plot = [t.replace(tzinfo=None) for t in times]

        # Equity curve
        ax = axes[0, 0]
        ax.plot(ts_plot, equities / START_EQUITY, color="#4c78a8", lw=1.5)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_title("Equity Curve (normalized)")
        ax.set_ylabel("Multiple of Starting Equity")

        # Drawdown
        ax = axes[0, 1]
        pk_arr = np.maximum.accumulate(equities)
        dd_arr = (pk_arr - equities) / pk_arr * 100
        ax.fill_between(ts_plot, -dd_arr, 0, color="#e45756", alpha=0.7)
        ax.set_title("Drawdown %")
        ax.set_ylabel("Drawdown (%)")

        # Regime time split
        ax = axes[1, 0]
        reg_labels = [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS, REGIME_HIGH_VOL]
        reg_colors = ["#54a24b", "#e45756", "#79706e", "#f58518"]
        reg_vals   = [regime_counts[r] for r in reg_labels]
        ax.pie(reg_vals, labels=reg_labels, colors=reg_colors,
               autopct="%1.0f%%", startangle=140)
        ax.set_title("Time in Each Regime")

        # Regime cumulative PnL
        ax = axes[1, 1]
        reg_pnls = [regime_pnl[r] * 100 for r in reg_labels]
        colors_bar = [reg_colors[i] for i in range(len(reg_labels))]
        bars = ax.bar(reg_labels, reg_pnls, color=colors_bar, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Cumulative PnL by Regime (%)")
        ax.set_ylabel("Cumulative Return (%)")
        for bar, val in zip(bars, reg_pnls):
            ax.text(bar.get_x() + bar.get_width() / 2, val + (0.002 if val >= 0 else -0.005),
                    f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)

        plt.tight_layout()
        img_path = os.path.join(RESULTS, "backtest_v12.png")
        plt.savefig(img_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Chart  → {img_path}")
    except Exception as e:
        print(f"  Chart skipped: {e}")

    print(f"\n{'═'*65}")
    print("  Done.")
    print(f"{'═'*65}")


if __name__ == "__main__":
    run()
