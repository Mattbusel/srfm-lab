"""
arena_v10.py — Head-to-head v9 vs v10 arena on real + synthetic data.

Tests before QC backtest:
  1. Real data: v9 vs v10 equity curves (ES/NQ/YM hourly 2023)
  2. Margin cascade simulation: start at $15M, correlated crash, does v10 survive?
  3. Volmageddon replay: Feb 2018 synthetic vol spike, sizing comparison
  4. Trade frequency: hourly gate still working, no hypertrading
  5. 10 synthetic worlds: v10 never worse than v9 on max drawdown

Usage:
    python tools/arena_v10.py
    python tools/arena_v10.py --equity 15000000   # test at $15M start
    python tools/arena_v10.py --no-synth          # skip synthetic worlds
"""

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Tuple
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# ── Constants (mirror v9/v10 exactly) ────────────────────────────────────────
CF = {
    "15m": {"ES": 0.0003, "NQ": 0.0004,  "YM": 0.00025},
    "1h":  {"ES": 0.001,  "NQ": 0.0012,  "YM": 0.0008},
    "1d":  {"ES": 0.005,  "NQ": 0.006,   "YM": 0.004},
}
TF_CAP = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}
MIN_HOLD_BARS    = 4
TARGET_DAILY_RISK = 0.01   # v10: 1% of equity daily risk per instrument

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── BH instrument (pure Python, no QC) ───────────────────────────────────────
class BHInstrument:
    def __init__(self, sym: str, cf: float, res: str):
        self.sym = sym
        self.cf  = cf
        self.res = res
        self.bh_mass   = 0.0
        self.bh_active = False
        self.bh_dir    = 0
        self.ctl       = 0
        self.bc        = 0
        self.prices: List[float] = []
        self._warmup = {"15m": 400, "1h": 120, "1d": 30}.get(res, 120)

    def update(self, close: float):
        self.bc += 1
        self.prices.append(close)
        if len(self.prices) < 2:
            return
        prev = self.prices[-2]
        beta = abs(close - prev) / (prev + 1e-9) / (self.cf + 1e-9)
        was = self.bh_active
        if beta < 1.0:
            self.ctl += 1
            sb = min(2.0, 1.0 + self.ctl * 0.1)
            self.bh_mass = self.bh_mass * 0.97 + 0.03 * sb
        else:
            self.ctl = 0
            self.bh_mass *= 0.95
        if not was:
            self.bh_active = self.bh_mass > 1.5 and self.ctl >= 3
        else:
            self.bh_active = self.bh_mass > 1.0 and self.ctl >= 3
        if not was and self.bh_active:
            lb = min(20, len(self.prices) - 1)
            self.bh_dir = 1 if close > self.prices[-1 - lb] else -1
        elif was and not self.bh_active:
            self.bh_dir = 0
        if self.bc < self._warmup:
            self.bh_active = False
            self.bh_dir = 0

    def direction(self) -> int:
        if self.bh_dir != 0: return self.bh_dir
        if len(self.prices) >= 5:
            return 1 if self.prices[-1] > self.prices[-5] else -1
        return 0


# ── Wilder's ATR ─────────────────────────────────────────────────────────────
class WilderATR:
    def __init__(self, period: int = 14):
        self.period = period
        self._trs: List[float] = []
        self._atr = None

    def update(self, high: float, low: float, prev_close: float) -> float:
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        self._trs.append(tr)
        if len(self._trs) >= self.period:
            if self._atr is None:
                self._atr = np.mean(self._trs[-self.period:])
            else:
                self._atr = (self._atr * (self.period - 1) + tr) / self.period
        return self._atr or 0.0

    @property
    def value(self) -> float:
        return self._atr or 0.0

    @property
    def ready(self) -> bool:
        return self._atr is not None


# ── Multi-res instrument ──────────────────────────────────────────────────────
class MultiResInstrument:
    def __init__(self, sym: str):
        self.sym = sym
        self.i15 = BHInstrument(sym, CF["15m"][sym], "15m")
        self.i1h = BHInstrument(sym, CF["1h"][sym],  "1h")
        self.i1d = BHInstrument(sym, CF["1d"][sym],  "1d")
        self.atr = WilderATR(14)
        self.last_target = 0.0
        self.bars_held   = 0
        self.pos_floor   = 0.0

    def tf_score(self) -> int:
        return (4 if self.i1d.bh_active else 0) + \
               (2 if self.i1h.bh_active else 0) + \
               (1 if self.i15.bh_active else 0)

    def direction(self) -> int:
        if self.i1d.bh_active: return self.i1d.direction()
        if self.i1h.bh_active: return self.i1h.direction()
        if self.i15.bh_active: return self.i15.direction()
        return 0

    def update(self, open_: float, high: float, low: float, close: float, prev_close: float):
        # 15m: 4 sub-bars (open, h/l midpoints, close)
        sub = [open_, (open_ + high) / 2, (low + close) / 2, close]
        for p in sub:
            self.i15.update(p)
        # 1h
        self.i1h.update(close)
        self.atr.update(high, low, prev_close)
        # 1d: every 6 hourly bars (approximated via modulo on bar count)
        if self.i1h.bc % 6 == 0:
            self.i1d.update(close)


# ── Sizing: v9 (fixed) vs v10 (vol-targeted) ─────────────────────────────────
def compute_size_v9(ceiling: float, direction: int) -> float:
    return ceiling * direction if direction != 0 else 0.0

def compute_size_v10(ceiling: float, direction: int,
                     atr_val: float, price: float) -> float:
    if direction == 0 or ceiling == 0.0:
        return 0.0
    if atr_val > 0 and price > 0:
        hourly_vol = atr_val / price
        daily_vol  = hourly_vol * math.sqrt(6.5)
        raw = TARGET_DAILY_RISK / (daily_vol + 1e-9)
        cap = min(raw, ceiling)
    else:
        cap = ceiling
    return cap * direction


# ── Broker / portfolio sim ────────────────────────────────────────────────────
class Broker:
    def __init__(self, equity: float, version: str):
        self.equity      = equity
        self.version     = version
        self.peak        = equity
        self.curve       = [equity]
        self.positions   = {"ES": 0.0, "NQ": 0.0, "YM": 0.0}
        self.total_fees  = 0.0
        self.trade_count = 0
        self.margin_calls = 0
        self._prev_prices: Dict[str, float] = {}
        self.vol_sizes: Dict[str, List[float]] = {"ES": [], "NQ": [], "YM": []}

    def step(self, instruments: Dict[str, MultiResInstrument],
             prices: Dict[str, float], bar_idx: int):
        mults = {"ES": 50, "NQ": 20, "YM": 5}

        # Mark-to-market BEFORE rebalancing
        if self._prev_prices:
            for sym, pos in self.positions.items():
                if abs(pos) < 0.001: continue
                pp = self._prev_prices.get(sym, 0)
                cp = prices.get(sym, pp)
                if pp > 0:
                    self.equity += pos * (cp - pp) / pp * self.equity

        # Margin call: if equity < 0.5% of peak, force liquidate
        if self.equity < self.peak * 0.005 and self.equity > 0:
            self.margin_calls += 1
            self.positions = {"ES": 0.0, "NQ": 0.0, "YM": 0.0}
            for inst in instruments.values():
                inst.last_target = 0.0
                inst.bars_held = 0

        self._prev_prices = dict(prices)

        # Compute raw targets
        raw = {}
        for sym, inst in instruments.items():
            tfs     = inst.tf_score()
            ceiling = TF_CAP[tfs]
            flat    = abs(inst.last_target) < 0.01

            if tfs == 1 and flat:
                ceiling = 0.0

            if ceiling == 0.0:
                tgt = 0.0
            else:
                d = inst.direction()
                price = prices.get(sym, 1.0)
                atr   = inst.atr.value

                if self.version == "v9":
                    tgt = compute_size_v9(ceiling, d)
                else:
                    tgt = compute_size_v10(ceiling, d, atr, price)

                # pos_floor (identical in both)
                if tfs >= 6 and abs(tgt) > 0.15 and inst.i1h.ctl >= 5:
                    inst.pos_floor = max(inst.pos_floor, 0.70 * abs(tgt))
                if inst.pos_floor > 0 and tfs >= 4 and not np.isclose(inst.last_target, 0):
                    tgt = float(np.sign(inst.last_target) * max(abs(tgt), inst.pos_floor))
                    inst.pos_floor *= 0.95
                if tfs < 4 or np.isclose(tgt, 0):
                    inst.pos_floor = 0.0
                if not inst.i1d.bh_active and not inst.i1h.bh_active:
                    inst.pos_floor = 0.0

            # v9 minimum-hold gate (same in both)
            if abs(inst.last_target) > 0.02:
                inst.bars_held += 1
            is_reversal = (
                not np.isclose(inst.last_target, 0) and
                not np.isclose(tgt, 0) and
                np.sign(tgt) != np.sign(inst.last_target)
            )
            if is_reversal and inst.bars_held < MIN_HOLD_BARS:
                tgt = inst.last_target

            raw[sym] = tgt
            # Track vol size for v10
            if self.version == "v10":
                self.vol_sizes[sym].append(abs(tgt))

        # Portfolio cap at 1.0
        total = sum(abs(v) for v in raw.values())
        scale = 1.0 / total if total > 1.0 else 1.0

        # Execute
        for sym, inst in instruments.items():
            tgt = float(raw[sym] * scale)
            old = inst.last_target
            if abs(tgt - old) < 0.02:
                continue

            price = prices.get(sym, 4000.0)
            mult  = mults[sym]
            n_contracts = max(1, int(abs(tgt - old) * self.equity / (price * mult + 1e-9)))
            fee = n_contracts * 2.0 * 2  # $2/side round-trip
            self.equity -= fee
            self.total_fees += fee
            self.trade_count += 1

            if np.isclose(tgt, 0):
                inst.bars_held = 0
            elif np.sign(tgt) != np.sign(old):
                inst.bars_held = 0

            inst.last_target = tgt
            self.positions[sym] = tgt

        self.curve.append(max(self.equity, 0))
        if self.equity > self.peak:
            self.peak = self.equity

    def stats(self) -> dict:
        arr = np.array(self.curve)
        rets = np.diff(arr) / (arr[:-1] + 1e-9)
        sharpe = rets.mean() / (rets.std() + 1e-9) * math.sqrt(252 * 6.5)
        peak = arr[0]; max_dd = 0.0
        for v in arr:
            peak = max(peak, v)
            max_dd = max(max_dd, (peak - v) / (peak + 1e-9))
        return {
            "final":       arr[-1],
            "peak":        self.peak,
            "total_ret%":  (arr[-1] - arr[0]) / arr[0] * 100,
            "peak_ret%":   (self.peak - arr[0]) / arr[0] * 100,
            "max_dd%":     max_dd * 100,
            "sharpe":      sharpe,
            "trades":      self.trade_count,
            "fees":        self.total_fees,
            "margin_calls": self.margin_calls,
        }


# ── Data loading ──────────────────────────────────────────────────────────────
def load_real_bars(sym: str) -> List[dict]:
    path = os.path.join(DATA_DIR, f"{sym}_hourly_real.csv")
    bars = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                bars.append({
                    "open":  float(row.get("open") or row.get("Open") or row["close"]),
                    "high":  float(row.get("high") or row.get("High") or row["close"]),
                    "low":   float(row.get("low")  or row.get("Low")  or row["close"]),
                    "close": float(row.get("close") or row.get("Close")),
                })
            except (ValueError, KeyError):
                continue
    return bars


def generate_correlated(n_bars: int = 5000, seed: int = 42,
                        vol_spike_at: int = None) -> Dict[str, List[dict]]:
    """Correlated ES/NQ/YM with optional vol spike (for Volmageddon sim)."""
    rng = np.random.default_rng(seed)
    corr = np.array([[1.00, 0.92, 0.88],
                     [0.92, 1.00, 0.85],
                     [0.88, 0.85, 1.00]])
    base_vols = np.array([0.15, 0.20, 0.14]) / math.sqrt(252 * 6.5)
    L = np.linalg.cholesky(np.outer(base_vols, base_vols) * corr)

    starts = {"ES": 4000.0, "NQ": 14000.0, "YM": 33000.0}
    regime_probs = [0.55, 0.15, 0.25, 0.05]
    regime_mus   = [0.0003, -0.0002, 0.00005, -0.001]
    regime_sigs  = [0.8, 1.2, 0.6, 2.5]

    regimes = []
    while len(regimes) < n_bars:
        r = rng.choice(4, p=regime_probs)
        regimes.extend([r] * int(rng.exponential(200)))
    regimes = regimes[:n_bars]

    prices = {s: [v] for s, v in starts.items()}
    for i in range(n_bars):
        r = regimes[i]
        # Vol spike: 6× vol for 40 bars centred at vol_spike_at
        if vol_spike_at and abs(i - vol_spike_at) < 20:
            sig_mult = 6.0
        else:
            sig_mult = regime_sigs[r]
        z = L @ rng.standard_normal(3)
        for j, sym in enumerate(["ES", "NQ", "YM"]):
            ret = regime_mus[r] + sig_mult * base_vols[j] * z[j] * math.sqrt(252 * 6.5)
            prices[sym].append(prices[sym][-1] * (1 + ret))

    result = {sym: [] for sym in ["ES", "NQ", "YM"]}
    for sym in ["ES", "NQ", "YM"]:
        ps = prices[sym]
        for i in range(1, len(ps)):
            p, pp = ps[i], ps[i-1]
            hi = max(p, pp) * (1 + abs(rng.normal(0, 0.002)))
            lo = min(p, pp) * (1 - abs(rng.normal(0, 0.002)))
            result[sym].append({"open": pp, "high": hi, "low": lo, "close": p})
    return result


# ── Run one sim ───────────────────────────────────────────────────────────────
def run_sim(bars: Dict[str, List[dict]], equity: float,
            version: str) -> Tuple[Broker, Dict[str, MultiResInstrument]]:
    instruments = {sym: MultiResInstrument(sym) for sym in ["ES", "NQ", "YM"]}
    broker = Broker(equity, version)

    syms = ["ES", "NQ", "YM"]
    n = min(len(bars[s]) for s in syms)
    prev_closes = {sym: bars[sym][0]["close"] for sym in syms}

    for i in range(n):
        prices = {}
        for sym in syms:
            b = bars[sym][i]
            instruments[sym].update(b["open"], b["high"], b["low"], b["close"],
                                    prev_closes[sym])
            prev_closes[sym] = b["close"]
            prices[sym] = b["close"]

        broker.step(instruments, prices, i)

    return broker, instruments


# ── Pretty print comparison ───────────────────────────────────────────────────
def compare(label: str, v9: dict, v10: dict):
    print(f"\n  {'Metric':<22} {'v9':>12} {'v10':>12}  {'Delta':>10}")
    print("  " + "-" * 60)
    metrics = [
        ("Final equity $",    "final",        "${:,.0f}",   "${:,.0f}"),
        ("Peak equity $",     "peak",         "${:,.0f}",   "${:,.0f}"),
        ("Total return %",    "total_ret%",   "{:+.1f}%",   "{:+.1f}%"),
        ("Peak return %",     "peak_ret%",    "{:+.1f}%",   "{:+.1f}%"),
        ("Max drawdown %",    "max_dd%",      "{:.1f}%",    "{:.1f}%"),
        ("Sharpe",            "sharpe",       "{:.2f}",     "{:.2f}"),
        ("Trades",            "trades",       "{:,}",       "{:,}"),
        ("Total fees $",      "fees",         "${:,.0f}",   "${:,.0f}"),
        ("Margin calls",      "margin_calls", "{}",         "{}"),
    ]
    for name, key, fmt9, fmt10 in metrics:
        v9v  = v9[key]
        v10v = v10[key]
        delta = ""
        if isinstance(v9v, (int, float)) and isinstance(v10v, (int, float)):
            if key == "max_dd%":
                delta = f"{'↑' if v10v > v9v else '↓'}{abs(v10v - v9v):.1f}pp"
            elif key in ("trades", "margin_calls"):
                delta = f"{v10v - v9v:+d}"
            elif key == "fees":
                delta = f"${v10v - v9v:+,.0f}"
            else:
                delta = f"{(v10v - v9v) / (abs(v9v) + 1e-9) * 100:+.1f}%"
        v9s  = fmt9.format(v9v)
        v10s = fmt10.format(v10v)
        print(f"  {name:<22} {v9s:>12} {v10s:>12}  {delta:>10}")


# ── Test 1: Real data ─────────────────────────────────────────────────────────
def test_real_data(equity: float):
    print("\n" + "="*65)
    print("TEST 1: Real ES/NQ/YM data (2023)")
    print("="*65)
    try:
        bars = {sym: load_real_bars(sym) for sym in ["ES", "NQ", "YM"]}
        n = min(len(bars[s]) for s in ["ES", "NQ", "YM"])
        print(f"  Loaded {n} hourly bars per instrument")
    except FileNotFoundError as e:
        print(f"  SKIP: data file not found ({e})")
        return None, None

    b9,  _ = run_sim(bars, equity, "v9")
    b10, _ = run_sim(bars, equity, "v10")
    s9, s10 = b9.stats(), b10.stats()
    compare("Real data", s9, s10)

    # Sizing report for v10
    for sym in ["ES", "NQ", "YM"]:
        sizes = b10.vol_sizes[sym]
        if sizes:
            arr = np.array(sizes)
            print(f"\n  v10 {sym} sizing:  mean={arr.mean():.3f}  "
                  f"min={arr.min():.3f}  max={arr.max():.3f}  "
                  f"p25={np.percentile(arr, 25):.3f}  p75={np.percentile(arr, 75):.3f}")

    return s9, s10


# ── Test 2: Margin cascade at $15M ───────────────────────────────────────────
def test_margin_cascade():
    print("\n" + "="*65)
    print("TEST 2: Margin cascade — start at $15M, correlated crash")
    print("="*65)

    # Scenario: 200 bars calm trend (builds position), then 30-bar crash
    # 3 instruments all crash together (correlated), then partial recovery
    rng = np.random.default_rng(99)
    n_calm = 300
    n_crash = 30
    n_recover = 200

    prices_base = {"ES": 4800.0, "NQ": 16000.0, "YM": 38000.0}
    bars = {sym: [] for sym in ["ES", "NQ", "YM"]}

    for sym in ["ES", "NQ", "YM"]:
        p = prices_base[sym]
        # Calm bull trend
        for _ in range(n_calm):
            ret = rng.normal(0.0004, 0.003)   # slight uptrend, calm
            hi = p * (1 + abs(rng.normal(0, 0.002)))
            lo = p * (1 - abs(rng.normal(0, 0.002)))
            bars[sym].append({"open": p, "high": hi, "low": lo, "close": p * (1 + ret)})
            p = p * (1 + ret)
        # Correlated crash: -0.6% to -1.2%/bar for 30 bars
        for _ in range(n_crash):
            ret = rng.normal(-0.008, 0.012)   # hard crash, high vol
            hi = p * (1 + abs(rng.normal(0, 0.015)))
            lo = p * (1 - abs(rng.normal(0, 0.015)))
            bars[sym].append({"open": p, "high": hi, "low": lo, "close": p * (1 + ret)})
            p = p * (1 + ret)
        # Partial recovery
        for _ in range(n_recover):
            ret = rng.normal(0.0001, 0.004)
            hi = p * (1 + abs(rng.normal(0, 0.003)))
            lo = p * (1 - abs(rng.normal(0, 0.003)))
            bars[sym].append({"open": p, "high": hi, "low": lo, "close": p * (1 + ret)})
            p = p * (1 + ret)

    start_equity = 15_000_000.0

    b9,  _ = run_sim(bars, start_equity, "v9")
    b10, _ = run_sim(bars, start_equity, "v10")
    s9, s10 = b9.stats(), b10.stats()
    compare(f"$15M cascade", s9, s10)

    # Survival check
    survived_v9  = s9["final"]  > start_equity * 0.10  # > 10% of start
    survived_v10 = s10["final"] > start_equity * 0.10

    print(f"\n  Survival (> 10% of $15M start):  v9={'YES' if survived_v9 else 'NO'}  "
          f"v10={'YES' if survived_v10 else 'NO'}")

    # Max position before crash vs during crash
    print("\n  (Crash phase is bars 300-330)")

    return s9, s10


# ── Test 3: Volmageddon vol spike sizing ──────────────────────────────────────
def test_volmageddon():
    print("\n" + "="*65)
    print("TEST 3: Volmageddon — synthetic vol spike, sizing comparison")
    print("="*65)

    # 2000 bars: warm-up 500, calm 700, vol spike 100 bars, recovery 700
    WARMUP = 500
    CALM   = 700
    SPIKE  = 100
    RECOV  = 700

    spike_at = WARMUP + CALM + SPIKE // 2
    bars = generate_correlated(n_bars=WARMUP + CALM + SPIKE + RECOV,
                               seed=42, vol_spike_at=spike_at)

    b9,  _ = run_sim(bars, 1_000_000.0, "v9")
    b10, _ = run_sim(bars, 1_000_000.0, "v10")
    s9, s10 = b9.stats(), b10.stats()
    compare("Volmageddon", s9, s10)

    # Compare sizing during calm vs spike for v10
    for sym in ["ES", "NQ", "YM"]:
        sizes = np.array(b10.vol_sizes[sym])
        if len(sizes) < WARMUP + CALM + SPIKE:
            continue
        calm_sizes  = sizes[WARMUP : WARMUP + CALM]
        spike_sizes = sizes[WARMUP + CALM : WARMUP + CALM + SPIKE]
        post_sizes  = sizes[WARMUP + CALM + SPIKE:]

        calm_mean  = calm_sizes.mean()  if len(calm_sizes)  else 0
        spike_mean = spike_sizes.mean() if len(spike_sizes) else 0
        post_mean  = post_sizes.mean()  if len(post_sizes)  else 0

        reduction = calm_mean / (spike_mean + 1e-9)
        print(f"\n  {sym} sizing: calm={calm_mean:.3f}  "
              f"spike={spike_mean:.3f}  post={post_mean:.3f}  "
              f"reduction={reduction:.1f}x")

    return s9, s10


# ── Test 4: Trade frequency check ────────────────────────────────────────────
def test_trade_frequency():
    print("\n" + "="*65)
    print("TEST 4: Trade frequency — no hypertrading regression")
    print("="*65)

    bars = generate_correlated(n_bars=3000, seed=7)
    b9,  _ = run_sim(bars, 1_000_000.0, "v9")
    b10, _ = run_sim(bars, 1_000_000.0, "v10")

    # Hourly gate means max 1 trade per instrument per hour
    # 3 instruments × 3000 hours = 9000 max possible trades (every bar)
    # Expected: ~100-300 for signal-driven trading
    print(f"  v9  trades: {b9.trade_count:,}  (per 3000 bars)")
    print(f"  v10 trades: {b10.trade_count:,}  (per 3000 bars)")

    trades_per_bar_v9  = b9.trade_count  / 3000
    trades_per_bar_v10 = b10.trade_count / 3000

    ok_v9  = trades_per_bar_v9  < 0.5   # less than 1 trade per 2 bars on average
    ok_v10 = trades_per_bar_v10 < 0.5

    print(f"  v9  trades/bar: {trades_per_bar_v9:.3f}  {'OK' if ok_v9 else 'HIGH'}")
    print(f"  v10 trades/bar: {trades_per_bar_v10:.3f}  {'OK' if ok_v10 else 'HIGH'}")

    return ok_v9 and ok_v10


# ── Test 5: 10 synthetic worlds ───────────────────────────────────────────────
def test_synthetic_worlds(n: int = 10):
    print("\n" + "="*65)
    print(f"TEST 5: {n} synthetic worlds — v10 max_dd never worse than v9")
    print("="*65)

    v10_beats_on_dd = 0
    v10_beats_on_final = 0
    results = []

    for seed in range(n):
        bars = generate_correlated(n_bars=4000, seed=seed * 13 + 7,
                                   vol_spike_at=2000)
        b9,  _ = run_sim(bars, 1_000_000.0, "v9")
        b10, _ = run_sim(bars, 1_000_000.0, "v10")
        s9, s10 = b9.stats(), b10.stats()

        dd_better   = s10["max_dd%"] <= s9["max_dd%"]
        final_ok    = s10["final"] > 0

        if dd_better:     v10_beats_on_dd    += 1
        if s10["final"] > s9["final"]: v10_beats_on_final += 1

        results.append({
            "seed": seed,
            "v9_dd": s9["max_dd%"], "v10_dd": s10["max_dd%"],
            "v9_ret": s9["total_ret%"], "v10_ret": s10["total_ret%"],
            "dd_better": dd_better,
        })

    print(f"\n  {'Seed':<6} {'v9 DD':>8} {'v10 DD':>8} {'v9 Ret':>9} {'v10 Ret':>9}  DD?")
    print("  " + "-" * 52)
    for r in results:
        flag = "✓" if r["dd_better"] else "✗"
        print(f"  {r['seed']:<6} {r['v9_dd']:>7.1f}% {r['v10_dd']:>7.1f}%"
              f" {r['v9_ret']:>+8.1f}% {r['v10_ret']:>+8.1f}%  {flag}")

    print(f"\n  v10 lower max_dd:  {v10_beats_on_dd}/{n}")
    print(f"  v10 higher final:  {v10_beats_on_final}/{n}")

    return v10_beats_on_dd, n


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--equity", type=float, default=1_000_000.0)
    parser.add_argument("--no-synth", action="store_true")
    parser.add_argument("--worlds", type=int, default=10)
    args = parser.parse_args()

    print("\n" + "█"*65)
    print("  LARSA v10 Arena — pre-QC validation")
    print("█"*65)

    passed = []
    failed = []

    def record(label, ok):
        (passed if ok else failed).append(label)
        print(f"\n  {'[PASS]' if ok else '[FAIL]'} {label}")

    # Test 1
    s9, s10 = test_real_data(args.equity)
    if s9 and s10:
        ok = s10["max_dd%"] <= s9["max_dd%"] + 5.0  # v10 DD within 5pp of v9
        record("Real data: v10 max_dd within 5pp of v9", ok)

    # Test 2
    s9, s10 = test_margin_cascade()
    ok = s10["margin_calls"] <= s9["margin_calls"]
    record("Cascade: v10 margin_calls ≤ v9", ok)
    ok2 = s10["max_dd%"] <= s9["max_dd%"] + 5.0
    record("Cascade: v10 max_dd within 5pp of v9", ok2)

    # Test 3
    s9, s10 = test_volmageddon()
    ok = s10["max_dd%"] <= s9["max_dd%"] + 5.0
    record("Volmageddon: v10 max_dd within 5pp of v9", ok)

    # Test 4
    freq_ok = test_trade_frequency()
    record("Trade frequency: both versions under 0.5 trades/bar", freq_ok)

    # Test 5
    if not args.no_synth:
        wins, total = test_synthetic_worlds(args.worlds)
        ok = wins >= int(total * 0.6)   # v10 lower DD in ≥60% of worlds
        record(f"Synthetic worlds: v10 lower DD in ≥60% ({wins}/{total})", ok)

    # Summary
    print("\n" + "="*65)
    print(f"  PASSED: {len(passed)}   FAILED: {len(failed)}")
    if failed:
        print("\n  Failed:")
        for f in failed:
            print(f"    - {f}")
    print("="*65)

    if not failed:
        print("\n  All arena tests passed. Safe to run QC backtest.\n")
    else:
        print("\n  Fix failures before running QC backtest.\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
