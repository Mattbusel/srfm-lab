"""
arena_v8.py — Local simulator for LARSA v8 multi-resolution SRFM.

Runs exact v8 BH physics (15min / 1H / 1D) on real CSV data with
correlated ES+NQ+YM synthetic data generation.

Usage:
    python tools/arena_v8.py                         # real data, 3 synth worlds
    python tools/arena_v8.py --mode synth            # synthetic only
    python tools/arena_v8.py --n-synth 10            # more synthetic worlds
    python tools/arena_v8.py --no-lab                # skip dashboard launch
    python tools/arena_v8.py --cf-scale 1.5          # tune CF multiplier

Outputs:
    results/v8_arena.json   — full run stats
    results/v8_equity.csv   — equity curve (for lab.py)
"""

import argparse
import csv
import json
import os
import sys
import webbrowser
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# CF calibration — mirrors v8/main.py exactly
# ---------------------------------------------------------------------------
CF = {
    "15m": {"ES": 0.0003, "NQ": 0.0004,  "YM": 0.00025},
    "1h":  {"ES": 0.001,  "NQ": 0.0012,  "YM": 0.0008},
    "1d":  {"ES": 0.005,  "NQ": 0.006,   "YM": 0.004},
}

TF_CAP = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}

# ---------------------------------------------------------------------------
# Single-resolution BH instrument — pure physics, no QC dependency
# ---------------------------------------------------------------------------
class BHInstrument:
    def __init__(self, label: str, cf: float, res: str,
                 bh_form=1.5, bh_collapse=1.0, bh_decay=0.95):
        self.label = label
        self.cf = cf
        self.res = res
        self.bh_form = bh_form
        self.bh_collapse = bh_collapse
        self.bh_decay = bh_decay

        self.bh_mass = 0.0
        self.bh_active = False
        self.bh_dir = 0
        self.bh_entry_price = 0.0
        self.ctl = 0
        self.bc = 0
        self.prices: List[float] = []

        bc_warmup = {"15m": 400, "1h": 120, "1d": 30}
        self._warmup_bars = bc_warmup.get(res, 120)

    def update(self, close: float):
        self.bc += 1
        self.prices.append(close)

        if len(self.prices) < 2:
            return

        prev = self.prices[-2]
        beta = abs(close - prev) / (prev + 1e-9) / (self.cf + 1e-9)

        was_active = self.bh_active

        if beta < 1.0:
            self.ctl += 1
            sb = min(2.0, 1.0 + self.ctl * 0.1)
            self.bh_mass = self.bh_mass * 0.97 + 0.03 * sb
        else:
            self.ctl = 0
            self.bh_mass *= self.bh_decay

        if not was_active:
            self.bh_active = self.bh_mass > self.bh_form and self.ctl >= 3
        else:
            self.bh_active = self.bh_mass > self.bh_collapse and self.ctl >= 3

        if not was_active and self.bh_active:
            lookback = min(20, len(self.prices) - 1)
            self.bh_dir = 1 if close > self.prices[-1 - lookback] else -1
            self.bh_entry_price = close
        elif was_active and not self.bh_active:
            self.bh_dir = 0

        # Warmup gate
        if self.bc < self._warmup_bars:
            self.bh_active = False
            self.bh_dir = 0

    def direction(self) -> int:
        if self.bh_dir != 0:
            return self.bh_dir
        if len(self.prices) >= 5:
            return 1 if self.prices[-1] > self.prices[-5] else -1
        return 0


# ---------------------------------------------------------------------------
# Multi-resolution instrument (15m + 1h + 1d per underlying)
# ---------------------------------------------------------------------------
class MultiResInstrument:
    def __init__(self, sym: str, cf_scale: float = 1.0):
        self.sym = sym
        self.i15 = BHInstrument(sym, CF["15m"][sym] * cf_scale, "15m")
        self.i1h = BHInstrument(sym, CF["1h"][sym]  * cf_scale, "1h")
        self.i1d = BHInstrument(sym, CF["1d"][sym]  * cf_scale, "1d")
        self.last_target = 0.0
        self.pos_floor = 0.0

    def tf_score(self) -> int:
        return (4 if self.i1d.bh_active else 0) + \
               (2 if self.i1h.bh_active else 0) + \
               (1 if self.i15.bh_active else 0)

    def direction(self) -> int:
        if self.i1d.bh_active: return self.i1d.direction()
        if self.i1h.bh_active: return self.i1h.direction()
        if self.i15.bh_active: return self.i15.direction()
        return 0


# ---------------------------------------------------------------------------
# Multi-resolution broker / portfolio simulator
# ---------------------------------------------------------------------------
class V8Broker:
    def __init__(self, equity: float = 1_000_000.0, commission: float = 2.0):
        self.equity = equity
        self.peak_equity = equity
        self.commission = commission  # $ per contract side
        self.positions: Dict[str, float] = {"ES": 0.0, "NQ": 0.0, "YM": 0.0}
        self.equity_curve: List[float] = [equity]
        self.trades: List[dict] = []
        self._open_trades: Dict[str, dict] = {}
        self.total_fees = 0.0
        self._prev_prices: Dict[str, float] = {}

    def step(self, instruments: Dict[str, MultiResInstrument],
             prices: Dict[str, float], bar_idx: int):
        """Execute one hourly step."""
        multipliers = {"ES": 50, "NQ": 20, "YM": 5}

        # Mark-to-market: apply price returns to existing positions BEFORE rebalancing
        if hasattr(self, "_prev_prices") and self._prev_prices:
            for sym, pos_frac in self.positions.items():
                if abs(pos_frac) < 0.001: continue
                prev_p = self._prev_prices.get(sym)
                curr_p = prices.get(sym)
                if prev_p and curr_p and prev_p > 0:
                    bar_ret = (curr_p - prev_p) / prev_p
                    self.equity += pos_frac * bar_ret * self.equity
        self._prev_prices = dict(prices)

        # Compute targets
        raw = {}
        for sym, inst in instruments.items():
            tfs = inst.tf_score()
            cap = TF_CAP[tfs]

            # tf_score=1 alone can't open new position
            currently_flat = abs(inst.last_target) < 0.01
            if tfs == 1 and currently_flat:
                cap = 0.0

            if cap == 0.0:
                tgt = 0.0
            else:
                d = inst.direction()
                tgt = cap * d if d != 0 else 0.0

            # pos_floor
            if tfs >= 6 and abs(tgt) > 0.15 and inst.i1h.ctl >= 5:
                inst.pos_floor = max(inst.pos_floor, 0.70 * abs(tgt))
            if inst.pos_floor > 0.0 and tfs >= 4 and not np.isclose(inst.last_target, 0.0):
                tgt = float(np.sign(inst.last_target) * max(abs(tgt), inst.pos_floor))
                inst.pos_floor *= 0.95
            if tfs < 4 or np.isclose(tgt, 0.0):
                inst.pos_floor = 0.0
            if not inst.i1d.bh_active and not inst.i1h.bh_active:
                inst.pos_floor = 0.0

            raw[sym] = tgt

        # Portfolio cap at 1.0
        total_exp = sum(abs(v) for v in raw.values())
        scale = 1.0 / total_exp if total_exp > 1.0 else 1.0

        # Execute trades and track equity changes
        for sym, inst in instruments.items():
            tgt = float(raw[sym] * scale)
            old = inst.last_target
            if abs(tgt - old) < 0.02:
                continue

            # Fee: proportional to trade size change * equity / (rough contract value)
            size_change = abs(tgt - old)
            mult = multipliers.get(sym, 50)
            price = prices.get(sym, 4000.0)
            n_contracts = max(1, int(size_change * self.equity / (price * mult)))
            fee = n_contracts * self.commission * 2  # round-trip
            self.equity -= fee
            self.total_fees += fee

            # Log trade
            if abs(old) < 0.01 and abs(tgt) > 0.01:
                self._open_trades[sym] = {
                    "entry_bar": bar_idx, "entry_price": price,
                    "direction": np.sign(tgt), "size": abs(tgt),
                    "tfs": inst.tf_score(),
                }
            elif abs(old) > 0.01 and abs(tgt) < 0.01:
                if sym in self._open_trades:
                    ot = self._open_trades.pop(sym)
                    ep = ot["entry_price"]
                    ret = (price - ep) / ep * ot["direction"]
                    pnl_trade = ret * ot["size"] * self.equity
                    self.trades.append({
                        "sym": sym, "bars": bar_idx - ot["entry_bar"],
                        "ret": ret, "pnl": pnl_trade,
                        "tfs": ot["tfs"], "direction": ot["direction"],
                    })

            inst.last_target = tgt
            self.positions[sym] = tgt

        self.equity_curve.append(self.equity)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def stats(self) -> dict:
        curve = np.array(self.equity_curve)
        returns = np.diff(curve) / (curve[:-1] + 1e-9)
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252 * 6.5)

        peak = curve[0]
        max_dd = 0.0
        for v in curve:
            if v > peak: peak = v
            dd = (peak - v) / (peak + 1e-9)
            if dd > max_dd: max_dd = dd

        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]

        return {
            "total_return_pct": (curve[-1] - curve[0]) / curve[0] * 100,
            "peak_equity": self.peak_equity,
            "peak_return_pct": (self.peak_equity - curve[0]) / curve[0] * 100,
            "max_drawdown_pct": max_dd * 100,
            "sharpe": sharpe,
            "trade_count": len(self.trades),
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0,
            "total_fees": self.total_fees,
            "final_equity": curve[-1],
        }


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_csv(path: str) -> List[dict]:
    bars = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            def g(*keys):
                for k in keys:
                    if k in row and row[k] not in ("", None):
                        return float(row[k])
                return None
            c = g("close", "Close", "c")
            if c is None: continue
            bars.append({
                "open":  g("open",  "Open",  "o") or c,
                "high":  g("high",  "High",  "h") or c,
                "low":   g("low",   "Low",   "l") or c,
                "close": c,
            })
    return bars


def generate_correlated(n_bars: int = 20000, seed: int = 42) -> Dict[str, List[float]]:
    """Generate correlated ES/NQ/YM hourly price series."""
    rng = np.random.default_rng(seed)
    corr = np.array([[1.00, 0.92, 0.88],
                     [0.92, 1.00, 0.85],
                     [0.88, 0.85, 1.00]])
    vols = np.array([0.15, 0.20, 0.14]) / np.sqrt(252 * 6.5)
    L = np.linalg.cholesky(np.outer(vols, vols) * corr)

    starts = {"ES": 4000.0, "NQ": 14000.0, "YM": 33000.0}
    regime_probs = [0.55, 0.15, 0.25, 0.05]
    regime_mus   = [0.0003, -0.0002, 0.00005, -0.001]
    regime_sigs  = [0.8, 1.2, 0.6, 2.5]

    # Build regime sequence
    regimes = []
    while len(regimes) < n_bars:
        r = rng.choice(4, p=regime_probs)
        regimes.extend([r] * int(rng.exponential(200)))
    regimes = regimes[:n_bars]

    prices = {s: [v] for s, v in starts.items()}
    for i in range(n_bars):
        r = regimes[i]
        mu = regime_mus[r]
        sig = regime_sigs[r]
        z = L @ rng.standard_normal(3)
        for j, sym in enumerate(["ES", "NQ", "YM"]):
            ret = mu + sig * vols[j] * z[j] * np.sqrt(252 * 6.5)
            prices[sym].append(prices[sym][-1] * (1 + ret))

    return {s: v[1:] for s, v in prices.items()}


# ---------------------------------------------------------------------------
# Core simulation run
# ---------------------------------------------------------------------------
def run_v8(price_series: Dict[str, List[float]], cf_scale: float = 1.0,
           equity: float = 1_000_000.0, label: str = "") -> Tuple[V8Broker, Dict]:
    """
    Run v8 multi-resolution simulation.
    price_series: {"ES": [...], "NQ": [...], "YM": [...]} — hourly prices.
    15m prices are synthesized by sampling every 4 hourly bars (approximate).
    Daily prices are synthesized by sampling every 6 hourly bars (RTH session).
    """
    n = min(len(v) for v in price_series.values())

    instruments = {
        sym: MultiResInstrument(sym, cf_scale)
        for sym in ["ES", "NQ", "YM"]
    }
    broker = V8Broker(equity=equity)

    # 15m approximation: sub-sample hourly → 4 synthetic 15m bars per hour
    def _make_15m(prev: float, curr: float, rng, vol: float) -> List[float]:
        """Linearly interpolate hourly bar into 4 15-min closes with noise."""
        pts = []
        for i in range(1, 5):
            frac = i / 4
            noise = rng.normal(0, vol * abs(curr - prev) * 0.1)
            pts.append(prev + frac * (curr - prev) + noise)
        pts[-1] = curr  # anchor last 15m to hourly close
        return pts

    rng = np.random.default_rng(42)
    vols_15m = {"ES": 0.0002, "NQ": 0.0003, "YM": 0.00015}

    prev_prices = {sym: price_series[sym][0] for sym in ["ES", "NQ", "YM"]}
    daily_accum = {sym: [] for sym in ["ES", "NQ", "YM"]}

    for bar_idx in range(n):
        curr_prices = {sym: price_series[sym][bar_idx] for sym in ["ES", "NQ", "YM"]}

        for sym, inst in instruments.items():
            prev = prev_prices[sym]
            curr = curr_prices[sym]

            # 15m: 4 synthetic sub-bars per hour
            sub_bars = _make_15m(prev, curr, rng, vols_15m[sym])
            for sb in sub_bars:
                inst.i15.update(sb)

            # 1h: one bar per iteration
            inst.i1h.update(curr)

            # 1d: accumulate 6 hourly bars → one daily bar
            daily_accum[sym].append(curr)
            if len(daily_accum[sym]) >= 6:
                inst.i1d.update(daily_accum[sym][-1])
                daily_accum[sym] = []

        broker.step(instruments, curr_prices, bar_idx)
        prev_prices = curr_prices

    return broker, instruments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LARSA v8 local arena")
    parser.add_argument("--mode",     default="real", choices=["real", "synth", "both"])
    parser.add_argument("--n-synth",  type=int, default=3)
    parser.add_argument("--n-bars",   type=int, default=20000)
    parser.add_argument("--cf-scale", type=float, default=1.0, help="CF multiplier (1.0=default)")
    parser.add_argument("--equity",   type=float, default=1_000_000.0)
    parser.add_argument("--no-lab",   action="store_true")
    parser.add_argument("--es",  default="data/ES_hourly_real.csv")
    parser.add_argument("--nq",  default="data/NQ_hourly_real.csv")
    parser.add_argument("--ym",  default="data/YM_hourly_real.csv")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    all_results = []

    # ── Real data run ──
    if args.mode in ("real", "both"):
        real_available = all(os.path.exists(p) for p in [args.es, args.nq, args.ym])
        if real_available:
            print(f"Loading real data: ES={args.es}  NQ={args.nq}  YM={args.ym}")
            ps = {
                "ES": [b["close"] for b in load_csv(args.es)],
                "NQ": [b["close"] for b in load_csv(args.nq)],
                "YM": [b["close"] for b in load_csv(args.ym)],
            }
            n = min(len(v) for v in ps.values())
            print(f"  {n} hourly bars per instrument")
            broker, instr = run_v8(ps, cf_scale=args.cf_scale, equity=args.equity, label="REAL")
            s = broker.stats()
            s["label"] = "REAL"
            all_results.append((broker, s, "REAL"))
            _print_stats(s, "REAL DATA")

            # Save equity curve
            with open("results/v8_equity.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["bar", "equity"])
                for i, v in enumerate(broker.equity_curve):
                    w.writerow([i, v])
        else:
            print(f"[WARN] Real data not found — run with --mode synth or provide CSVs")
            if args.mode == "real":
                args.mode = "synth"

    # ── Synthetic worlds ──
    if args.mode in ("synth", "both"):
        print(f"\nRunning {args.n_synth} synthetic worlds ({args.n_bars} bars each)...")
        synth_stats = []
        for seed in range(args.n_synth):
            ps = generate_correlated(args.n_bars, seed=seed + 100)
            broker, _ = run_v8(ps, cf_scale=args.cf_scale, equity=args.equity)
            s = broker.stats()
            s["label"] = f"SYNTH_{seed}"
            synth_stats.append(s)
            all_results.append((broker, s, f"SYNTH_{seed}"))
            print(f"  [{seed+1}/{args.n_synth}] return={s['total_return_pct']:+.1f}%  "
                  f"peak={s['peak_return_pct']:+.1f}%  "
                  f"DD={s['max_drawdown_pct']:.1f}%  "
                  f"Sh={s['sharpe']:.3f}  "
                  f"trades={s['trade_count']}  "
                  f"fees=${s['total_fees']:,.0f}")

        # Median synth stats
        if synth_stats:
            med_sh  = sorted(s["sharpe"] for s in synth_stats)[len(synth_stats)//2]
            med_ret = sorted(s["total_return_pct"] for s in synth_stats)[len(synth_stats)//2]
            med_dd  = sorted(s["max_drawdown_pct"] for s in synth_stats)[len(synth_stats)//2]
            print(f"\n  SYNTHETIC MEDIAN  return={med_ret:+.1f}%  DD={med_dd:.1f}%  Sh={med_sh:.3f}")

    # ── tf_score distribution breakdown ──
    print("\n── TF SCORE DISTRIBUTION (real run) ──")
    if all_results:
        broker0, _, _ = all_results[0]
        tfs_counts = defaultdict(int)
        tfs_pnl    = defaultdict(float)
        for t in broker0.trades:
            tfs_counts[t["tfs"]] += 1
            tfs_pnl[t["tfs"]] += t["pnl"]
        for score in sorted(tfs_counts.keys(), reverse=True):
            label_str = f"TF={score}"
            if score == 7: label_str += " (15m+1h+1d)"
            elif score == 6: label_str += " (1h+1d)"
            elif score == 5: label_str += " (15m+1d)"
            elif score == 4: label_str += " (1d only)"
            elif score == 3: label_str += " (15m+1h)"
            elif score == 2: label_str += " (1h only)"
            elif score == 1: label_str += " (15m only)"
            print(f"  {label_str:<22}  {tfs_counts[score]:4d} trades  ${tfs_pnl[score]:>12,.0f} P&L")

    # ── Save JSON ──
    out = [{"label": s["label"], **{k: v for k, v in s.items() if k != "label"}}
           for _, s, _ in all_results]
    with open("results/v8_arena.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results → results/v8_arena.json")
    if os.path.exists("results/v8_equity.csv"):
        print(f"  Equity  → results/v8_equity.csv")

    # ── Launch lab ──
    if not args.no_lab:
        print("\nLaunching lab dashboard (Ctrl+C to stop)...")
        try:
            import subprocess
            subprocess.Popen([sys.executable, "tools/lab.py"])
        except Exception as e:
            print(f"  [lab launch failed: {e}] — run manually: python tools/lab.py")


def _print_stats(s: dict, label: str = ""):
    bar = "=" * 55
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"  Total Return:   {s['total_return_pct']:>+8.2f}%")
    print(f"  Peak Return:    {s['peak_return_pct']:>+8.2f}%")
    print(f"  Max Drawdown:   {s['max_drawdown_pct']:>8.2f}%")
    print(f"  Sharpe:         {s['sharpe']:>8.3f}")
    print(f"  Trades:         {s['trade_count']:>8d}")
    print(f"  Win Rate:       {s['win_rate']:>8.1f}%")
    print(f"  Total Fees:     ${s['total_fees']:>10,.0f}")
    print(f"  Final Equity:   ${s['final_equity']:>12,.0f}")
    print(f"  Peak Equity:    ${s['peak_equity']:>12,.0f}")
    print(bar)


if __name__ == "__main__":
    main()
