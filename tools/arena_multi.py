"""
arena_multi.py — SRFM Multi-Instrument Arena (ES + NQ + YM).

Tests convergence events: the single biggest gap between arena (Sharpe 0.4)
and QC (Sharpe 4.3). When multiple instruments simultaneously form black holes,
the convergence multiplier scales up leverage — matching QC's multi-instrument
convergence logic exactly.

Usage:
    python tools/arena_multi.py --mode synth --n-worlds 5 --n-bars 20000
    python tools/arena_multi.py --mode synth --n-worlds 3 --n-bars 5000 --flags V
    python tools/arena_multi.py --mode real --es data/ES_hourly_real.csv ...

Experiment flags:
    V = CONV_SIZE: solo BH capped at 0.40, multi-BH gets full leverage
    3 = v3_pos_floor: ctl>=5 trigger, 70% retention (was ctl>=3, 90%)
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Correlated synthetic data generation
# ---------------------------------------------------------------------------

def generate_correlated(n_bars: int = 20000, seed: int = 42, regime_mix: Optional[dict] = None) -> List[List[dict]]:
    """
    Generate 3 correlated synthetic price series (ES, NQ, YM).

    Real correlations from QC backtest period:
      ES/NQ: ~0.92 (both tech-heavy index futures)
      ES/YM: ~0.88 (ES vs Dow)
      NQ/YM: ~0.85

    Returns [es_bars, nq_bars, ym_bars] where each is a list of OHLCV dicts.
    """
    rng = np.random.default_rng(seed)

    # Correlation matrix
    corr = np.array([
        [1.00, 0.92, 0.88],
        [0.92, 1.00, 0.85],
        [0.88, 0.85, 1.00]
    ])

    # Volatilities (annualized, hourly scaling: /sqrt(252*6.5))
    # ES ~15% ann, NQ ~20% ann, YM ~14% ann
    vols = np.array([0.15, 0.20, 0.14]) / np.sqrt(252 * 6.5)

    # Covariance matrix + Cholesky decomposition
    cov = np.outer(vols, vols) * corr
    L = np.linalg.cholesky(cov)

    # Default regime mix
    if regime_mix is None:
        regime_mix = {"bull": 0.55, "bear": 0.15, "sideways": 0.25, "crisis": 0.05}

    # Build regime sequence (exponential-length segments)
    regimes = []
    n = 0
    while n < n_bars:
        r = rng.choice(list(regime_mix.keys()), p=list(regime_mix.values()))
        length = int(rng.exponential(200))
        regimes.extend([r] * length)
        n += length
    regimes = regimes[:n_bars]

    # Regime params: (mu, sigma_scale)
    regime_params = {
        "bull":     {"mu": 0.0003,  "sigma": 0.8},
        "bear":     {"mu": -0.0002, "sigma": 1.2},
        "sideways": {"mu": 0.00005, "sigma": 0.6},
        "crisis":   {"mu": -0.001,  "sigma": 2.5},
    }

    # Generate correlated returns per bar
    base_returns = np.zeros((n_bars, 3))
    for i, regime in enumerate(regimes):
        params = regime_params[regime]
        z = rng.standard_normal(3)
        corr_z = L @ z
        base_returns[i] = params["mu"] + corr_z * params["sigma"]

    # Build price series
    prices = np.zeros((n_bars, 3))
    start_prices = np.array([4500.0, 15000.0, 35000.0])  # ES, NQ, YM
    prices[0] = start_prices
    for i in range(1, n_bars):
        prices[i] = prices[i - 1] * (1 + base_returns[i])
    # Guard against negative prices
    prices = np.maximum(prices, 1.0)

    # Build OHLCV bar dicts for each instrument
    bars_list: List[List[dict]] = []
    for inst_i in range(3):
        bars: List[dict] = []
        for i in range(n_bars):
            c = float(prices[i, inst_i])
            o = float(prices[i - 1, inst_i]) if i > 0 else c
            noise = abs(float(rng.standard_normal())) * vols[inst_i] * c
            h = max(o, c) + noise * 0.5
            lo = min(o, c) - noise * 0.5
            bars.append({
                "open": o, "high": h, "low": lo, "close": c,
                "volume": int(rng.integers(1000, 10000)),
                "date": f"bar_{i:06d}",
            })
        bars_list.append(bars)

    return bars_list  # [es_bars, nq_bars, ym_bars]


# ---------------------------------------------------------------------------
# Real data loader
# ---------------------------------------------------------------------------

def load_ohlcv(path: str) -> List[dict]:
    """Load OHLCV bars from a CSV file."""
    bars: List[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
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


def load_real_multi(es_csv: str, nq_csv: str, ym_csv: str) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Load and align three CSV files. Clips all to the shortest series length
    so that step() receives bars at the same index for all instruments.
    """
    es = load_ohlcv(es_csv)
    nq = load_ohlcv(nq_csv)
    ym = load_ohlcv(ym_csv)
    n = min(len(es), len(nq), len(ym))
    return es[:n], nq[:n], ym[:n]


# ---------------------------------------------------------------------------
# Per-instrument SRFM state
# ---------------------------------------------------------------------------

class SRFMInstrument:
    """
    Per-instrument SRFM state machine.

    Implements the same Minkowski/black-hole physics as arena_v2 / srfm_core
    but in a self-contained class suitable for multi-instrument composition.
    """

    def __init__(
        self,
        label: str,
        cf: float,
        bh_form: float = 1.5,
        bh_collapse: float = 1.0,
        bh_decay: float = 0.95,
    ):
        self.label = label
        self.cf = cf
        self.bh_form = bh_form
        self.bh_collapse = bh_collapse
        self.bh_decay = bh_decay

        # --- Physics state ---
        self.bh_mass: float = 0.0
        self.bh_active: bool = False
        self.bh_dir: int = 0
        self.ctl: int = 0           # consecutive TIMELIKE bars
        self.bit: str = "UNKNOWN"
        self.bc: int = 0            # bar count
        self.regime: str = "SIDEWAYS"
        self.tl_confirm: int = 0
        self.pos_floor: float = 0.0
        self.last_target: float = 0.0
        self.atr: float = 0.0
        self.atr_ratio: float = 1.0
        self.atr_avg: float = 0.0

        # Rolling price window (200 bars max)
        self.prices: List[float] = []

        # EMA state
        self.e12: Optional[float] = None
        self.e26: Optional[float] = None
        self.e50: Optional[float] = None
        self.e200: Optional[float] = None

        # ATR history (50-bar window)
        self.atr_history: List[float] = []

    # ------------------------------------------------------------------
    def _ema(self, prev: Optional[float], val: float, n: int) -> float:
        alpha = 2.0 / (n + 1)
        if prev is None:
            return val
        return prev * (1 - alpha) + val * alpha

    def update(self, bar: dict) -> float:
        """
        Process one bar. Returns target signal (fraction of equity, signed).
        0.0 means flat / no signal.
        """
        self.bc += 1
        c = bar["close"]
        self.prices.append(c)
        if len(self.prices) > 200:
            self.prices = self.prices[-200:]

        # Update EMAs
        self.e12  = self._ema(self.e12,  c, 12)
        self.e26  = self._ema(self.e26,  c, 26)
        self.e50  = self._ema(self.e50,  c, 50)
        self.e200 = self._ema(self.e200, c, 200)

        if len(self.prices) < 2:
            return 0.0

        prev_c = self.prices[-2]
        ret = abs(c - prev_c) / (prev_c + 1e-9)
        beta = ret / (self.cf + 1e-12)

        # ATR estimate (|high-low| proxy)
        atr_est = ret * c * 2
        self.atr_history.append(atr_est)
        if len(self.atr_history) > 50:
            self.atr_history = self.atr_history[-50:]
        self.atr = atr_est
        if len(self.atr_history) >= 14:
            self.atr_avg = sum(self.atr_history[-50:]) / len(self.atr_history[-50:])
            self.atr_ratio = self.atr / (self.atr_avg + 1e-9)

        # Minkowski classification
        self.bit = "TIMELIKE" if beta < 1.0 else "SPACELIKE"

        # Black-hole mass dynamics — mirrors srfm_core.BlackHoleDetector exactly:
        #   TIMELIKE : sb = min(2.0, 1 + ctl*0.1); mass = mass*decay + |br|*100*sb
        #   SPACELIKE: ctl = 0; mass *= 0.7
        br = (c - prev_c) / (prev_c + 1e-9)
        if self.bit == "TIMELIKE":
            self.ctl += 1
            self.tl_confirm = min(self.tl_confirm + 1, 5)
            sb = min(2.0, 1.0 + self.ctl * 0.1)
            self.bh_mass = self.bh_mass * self.bh_decay + abs(br) * 100 * sb
        else:
            self.ctl = 0
            self.tl_confirm = 0
            self.bh_mass *= 0.7  # SPACELIKE: faster mass decay

        # Regime detection (needs 200 bars for all EMAs to warm up)
        if self.bc >= 200:
            if self.e12 > self.e26 > self.e50 > self.e200:  # type: ignore[operator]
                self.regime = "BULL"
            elif self.e200 > self.e50 > self.e26 > self.e12:  # type: ignore[operator]
                self.regime = "BEAR"
            else:
                self.regime = "SIDEWAYS"

        # BH formation / collapse — matches srfm_core.BlackHoleDetector:
        #   Formation : mass >= bh_form AND ctl >= 5
        #   Collapse  : mass <= bh_collapse OR ctl < 5
        #   (ctl resets to 0 on any SPACELIKE bar, naturally collapsing the well)
        if not self.bh_active:
            if self.bh_mass >= self.bh_form and self.ctl >= 5:
                self.bh_active = True
                lookback = min(10, len(self.prices))
                self.bh_dir = 1 if c > self.prices[-lookback] else -1
        else:
            if self.bh_mass < self.bh_collapse or self.ctl < 5:
                self.bh_active = False
                self.pos_floor = 0.0

        # --- Signal generation ---
        if not self.bh_active or self.bc < 120:
            return 0.0

        tl_req = 1 if self.regime == "HIGH_VOL" else 3
        if self.tl_confirm < tl_req:
            return 0.0

        tgt = min(2.5, 1.0 + self.bh_mass * 0.30) * self.bh_dir

        # ATR scale (flag A from arena_v2): reduce when vol spikes
        if self.atr_ratio > 1.5:
            atr_scale = max(0.3, 1.5 / self.atr_ratio)
            tgt *= atr_scale

        return float(tgt)


# ---------------------------------------------------------------------------
# Multi-instrument arena
# ---------------------------------------------------------------------------

class MultiArena:
    """
    Runs ES, NQ, YM simultaneously, detects convergence events, manages
    per-instrument positions with a shared equity pool.
    """

    def __init__(self, cfg: dict, max_leverage: float = 0.65, exp_flags: str = ""):
        self.cfg = cfg
        self.max_leverage = max_leverage
        self.exp_flags = exp_flags.upper()

        cf = cfg.get("cf", 0.005)
        bh_form  = cfg.get("bh_form",  1.5)
        bh_decay = cfg.get("bh_decay", 0.95)

        # Per-instrument cf scaled to match QC's per-instrument volatility profiles.
        # Multipliers (1.0, 1.2, 0.9) set so that cf=0.005 (default) produces
        # cf values ≈ typical hourly bar returns in correlated synthetic data,
        # giving a healthy ~50% TIMELIKE fraction needed for BH formation.
        # Equivalent real-data cfs: ES≈0.001, NQ≈0.0012, YM≈0.0009 (matches
        # arena_v2.py CONFIGS when user passes --cf 0.001 for real data).
        self.instruments: Dict[str, SRFMInstrument] = {
            "ES": SRFMInstrument("ES", cf=cf * 1.00, bh_form=bh_form, bh_decay=bh_decay),
            "NQ": SRFMInstrument("NQ", cf=cf * 1.20, bh_form=bh_form, bh_decay=bh_decay),
            "YM": SRFMInstrument("YM", cf=cf * 0.90, bh_form=bh_form, bh_decay=bh_decay),
        }

        # Equity / position state
        self.equity: float = 1_000_000.0
        self.peak_equity: float = 1_000_000.0
        self.positions: Dict[str, float] = {"ES": 0.0, "NQ": 0.0, "YM": 0.0}
        self.last_prices: Dict[str, Optional[float]] = {"ES": None, "NQ": None, "YM": None}

        # Trade log
        self.trades: List[dict] = []

        # Equity curve
        self.equity_curve: List[float] = []

        # Convergence tracking
        self.convergence_events: List[dict] = []
        self.in_convergence: bool = False
        self.conv_entry_equity: float = 0.0
        self.conv_entry_bar: int = 0
        self.bar_count: int = 0

    # ------------------------------------------------------------------
    def _convergence_multiplier(self) -> Tuple[float, int]:
        """
        Compute the convergence leverage multiplier. Mirrors QC strategy exactly.

        Returns (multiplier, bh_count).
        """
        bh_count  = sum(1 for i in self.instruments.values() if i.bh_active)
        tl_count  = sum(
            1 for i in self.instruments.values()
            if i.bit == "TIMELIKE" and i.tl_confirm >= 3
        )
        total_bh_mass = sum(i.bh_mass for i in self.instruments.values() if i.bh_active)

        if bh_count >= 3:
            return 2.5, bh_count
        elif bh_count >= 2 and tl_count >= 3:
            return 2.0, bh_count
        elif bh_count >= 2:
            return 1.7, bh_count
        elif bh_count == 1 and tl_count >= 3 and total_bh_mass > 3.0:
            return 1.5, bh_count
        elif tl_count >= 3:
            return 1.4, bh_count
        elif tl_count >= 2:
            return 1.2, bh_count
        else:
            return 1.0, bh_count

    # ------------------------------------------------------------------
    def step(self, bars_dict: Dict[str, dict]):
        """
        Process one bar for all instruments.

        bars_dict: {"ES": bar, "NQ": bar, "YM": bar}
        """
        self.bar_count += 1

        # 1. Mark-to-market: update equity from price changes on open positions
        pnl_this_bar = 0.0
        for key in self.instruments:
            if self.last_prices[key] is not None and self.positions[key] != 0.0:
                c = bars_dict[key]["close"]
                prev_c = self.last_prices[key]
                ret = (c - prev_c) / (prev_c + 1e-9)
                pnl_this_bar += self.positions[key] * ret * self.equity
        self.equity += pnl_this_bar
        self.equity = max(self.equity, 1.0)  # prevent ruin
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # 2. Update each instrument SRFM state
        signals: Dict[str, float] = {}
        for key, inst in self.instruments.items():
            signals[key] = inst.update(bars_dict[key])
            self.last_prices[key] = bars_dict[key]["close"]

        # 3. Compute convergence multiplier
        conv_mult, bh_count = self._convergence_multiplier()

        # 4. Track convergence events (bh_count >= 2 = converging)
        was_converging = self.in_convergence
        self.in_convergence = bh_count >= 2
        if self.in_convergence and not was_converging:
            self.conv_entry_equity = self.equity
            self.conv_entry_bar    = self.bar_count
        elif not self.in_convergence and was_converging:
            conv_pnl = (self.equity - self.conv_entry_equity) / (self.conv_entry_equity + 1e-9)
            self.convergence_events.append({
                "start_bar": self.conv_entry_bar,
                "end_bar":   self.bar_count,
                "duration":  self.bar_count - self.conv_entry_bar,
                "pnl_pct":   conv_pnl * 100.0,
                "is_win":    conv_pnl > 0,
            })

        # 5. Size and apply positions per instrument
        for key, inst in self.instruments.items():
            tgt_raw = signals[key]

            if tgt_raw == 0.0:
                new_pos = 0.0
                inst.pos_floor = 0.0
            else:
                # Apply convergence multiplier to max leverage cap
                max_lev = self.max_leverage * conv_mult

                # Flag V = CONV_SIZE: solo BH capped at 0.40
                if "V" in self.exp_flags:
                    if bh_count == 1:
                        max_lev = min(max_lev, 0.40)
                    # bh_count >= 2 gets full (multiplied) leverage

                # pos_floor ratchet
                # Flag 3 = v3 settings: ctl>=5 trigger, 70% retention
                # Baseline: ctl>=3, 90% retention
                ctl_trigger = 5 if "3" in self.exp_flags else 3
                retention   = 0.70 if "3" in self.exp_flags else 0.90

                if abs(tgt_raw) > 0.5 and inst.ctl >= ctl_trigger:
                    inst.pos_floor = max(inst.pos_floor, retention * abs(tgt_raw))
                elif inst.pos_floor > 0:
                    inst.pos_floor *= 0.95  # slow decay
                    if inst.pos_floor < 0.05:
                        inst.pos_floor = 0.0

                if inst.pos_floor > 0 and inst.last_target != 0.0:
                    tgt_raw = float(
                        np.sign(inst.last_target) * max(abs(tgt_raw), inst.pos_floor)
                    )

                new_pos = float(np.clip(tgt_raw, -max_lev, max_lev))
                # Hard per-instrument cap (never more than full equity per leg)
                new_pos = float(np.clip(new_pos, -0.65, 0.65))

            # Record trade events
            delta = new_pos - self.positions[key]
            if abs(delta) > 0.02:
                if new_pos == 0.0 and self.positions[key] != 0.0:
                    # Close: find the open trade entry for this instrument
                    for t in reversed(self.trades):
                        if t.get("inst") == key and "exit_bar" not in t:
                            t["exit_bar"]    = self.bar_count
                            t["exit_equity"] = self.equity
                            break
                elif self.positions[key] == 0.0 and new_pos != 0.0:
                    # Open
                    self.trades.append({
                        "inst":               key,
                        "entry_bar":          self.bar_count,
                        "entry_equity":       self.equity,
                        "direction":          1 if new_pos > 0 else -1,
                        "size":               abs(new_pos),
                        "bh_count_at_entry":  bh_count,
                    })

            self.positions[key]   = new_pos
            inst.last_target      = new_pos

        self.equity_curve.append(self.equity)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return performance statistics dict."""
        if len(self.equity_curve) < 2:
            return {}

        eq = np.array(self.equity_curve)
        total_return = (eq[-1] - eq[0]) / eq[0] * 100.0

        # Max drawdown
        peak = np.maximum.accumulate(eq)
        dd   = (eq - peak) / (peak + 1e-9)
        max_dd = float(-dd.min() * 100.0)

        # Sharpe: annualised assuming 6.5 bars/day × 252 trading days
        rets = np.diff(eq) / (eq[:-1] + 1e-9)
        if rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * np.sqrt(252 * 6.5))
        else:
            sharpe = 0.0

        # Convergence stats
        conv_events = self.convergence_events
        conv_wins   = [e for e in conv_events if e["is_win"]]
        conv_pnl_total = sum(e["pnl_pct"] for e in conv_events)

        # Trade stats
        closed_trades = [t for t in self.trades if "exit_equity" in t]
        if closed_trades:
            trade_pnls = [
                (t["exit_equity"] - t["entry_equity"]) / (t["entry_equity"] + 1e-9)
                for t in closed_trades
            ]
            win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)
            conv_trades = [t for t in closed_trades if t.get("bh_count_at_entry", 0) >= 2]
            solo_trades = [t for t in closed_trades if t.get("bh_count_at_entry", 0) < 2]
        else:
            win_rate = 0.0
            conv_trades = []
            solo_trades = []

        def _wr(trade_list: list) -> float:
            if not trade_list:
                return 0.0
            wins = sum(
                1 for t in trade_list
                if t.get("exit_equity", t["entry_equity"]) > t["entry_equity"]
            )
            return round(wins / len(trade_list), 3)

        return {
            "total_return_pct":      round(total_return, 2),
            "max_drawdown_pct":      round(max_dd, 2),
            "sharpe":                round(sharpe, 3),
            "trade_count":           len(self.trades),
            "win_rate":              round(win_rate, 3),
            # Convergence analysis
            "convergence_count":     len(conv_events),
            "convergence_win_rate":  (
                round(len(conv_wins) / len(conv_events), 3) if conv_events else 0.0
            ),
            "convergence_pnl_pct":   round(conv_pnl_total, 2),
            "multi_win_rate":        _wr(conv_trades),
            "solo_win_rate":         _wr(solo_trades),
        }


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SRFM Multi-Instrument Arena (ES+NQ+YM)")
    parser.add_argument("--mode",    choices=["synth", "real"], default="synth",
                        help="Data source: synth (correlated synthetic) or real (CSV)")
    parser.add_argument("--es",      default="data/ES_hourly_real.csv")
    parser.add_argument("--nq",      default="data/NQ_hourly_real.csv")
    parser.add_argument("--ym",      default="data/YM_hourly_real.csv")
    parser.add_argument("--n-bars",  type=int, default=20000,
                        help="Bars per synthetic world")
    parser.add_argument("--n-worlds", type=int, default=5,
                        help="Number of synthetic worlds to simulate")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Base random seed (world i uses seed+i)")
    parser.add_argument("--cf",      type=float, default=0.005,
                        help="SRFM critical fraction cf")
    parser.add_argument("--flags",   default="",
                        help="Experiment flags: V=CONV_SIZE, 3=v3_pos_floor")
    parser.add_argument("--compare", action="store_true",
                        help="Compare baseline vs --flags variant side-by-side")
    args = parser.parse_args()

    cfg = {"cf": args.cf, "bh_form": 1.5, "bh_decay": 0.95}
    flags_label = args.flags.upper() or "BASELINE"
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    if args.mode == "real":
        print(f"Loading real data: {args.es}, {args.nq}, {args.ym}")
        es_bars, nq_bars, ym_bars = load_real_multi(args.es, args.nq, args.ym)
        n = len(es_bars)
        print(f"  {n} aligned bars loaded.")

        arena = MultiArena(cfg, max_leverage=0.65, exp_flags=args.flags)
        for i in range(n):
            arena.step({"ES": es_bars[i], "NQ": nq_bars[i], "YM": ym_bars[i]})

        s = arena.stats()
        _print_stats(s, label=f"REAL [{flags_label}]")

        out = "results/arena_multi_real.json"
        with open(out, "w") as f:
            json.dump({"mode": "real", "flags": flags_label, **s}, f, indent=2)
        print(f"Results -> {out}")
        return

    # ------------------------------------------------------------------
    # Synthetic mode
    print(f"Running {args.n_worlds} synthetic worlds "
          f"(N={args.n_bars} bars each, flags={flags_label})...")

    results = []
    for world_i in range(args.n_worlds):
        world_seed = world_i + args.seed
        bars_list = generate_correlated(args.n_bars, seed=world_seed)
        arena = MultiArena(cfg, max_leverage=0.65, exp_flags=args.flags)

        bars_dict_seq = [
            {"ES": bars_list[0][i], "NQ": bars_list[1][i], "YM": bars_list[2][i]}
            for i in range(args.n_bars)
        ]
        for bar_dict in bars_dict_seq:
            arena.step(bar_dict)

        s = arena.stats()
        s["seed"] = world_seed
        results.append(s)

        print(
            f"  World {world_i} (seed={world_seed}): "
            f"Sharpe={s['sharpe']:.3f}  "
            f"Return={s['total_return_pct']:+.1f}%  "
            f"MaxDD={s['max_drawdown_pct']:.1f}%  "
            f"Conv={s['convergence_count']}  "
            f"ConvWR={s['convergence_win_rate']:.1%}  "
            f"MultiWR={s['multi_win_rate']:.1%}  "
            f"SoloWR={s['solo_win_rate']:.1%}"
        )

    # Aggregate summary
    sharpes      = sorted(r["sharpe"] for r in results)
    returns      = sorted(r["total_return_pct"] for r in results)
    conv_counts  = sorted(r["convergence_count"] for r in results)
    conv_wrs     = sorted(r["convergence_win_rate"] for r in results)

    mid = len(results) // 2
    print(f"\n{'='*65}")
    print(f"  arena_multi [{flags_label}]  --  {args.n_worlds} worlds x {args.n_bars} bars")
    print(f"{'='*65}")
    print(f"  Median Sharpe           : {sharpes[mid]:.3f}")
    print(f"  Median Return           : {returns[mid]:+.1f}%")
    print(f"  Median Convergence Evts : {conv_counts[mid]}")
    print(f"  Median Conv Win Rate    : {conv_wrs[mid]:.1%}")
    print(f"{'='*65}\n")

    out = "results/arena_multi_synth.json"
    with open(out, "w") as f:
        json.dump(
            {"mode": "synth", "flags": flags_label, "n_bars": args.n_bars,
             "n_worlds": args.n_worlds, "results": results},
            f, indent=2,
        )
    print(f"Results -> {out}")

    # ------------------------------------------------------------------
    # Optional compare: re-run with no flags as baseline
    if args.compare and args.flags:
        print(f"\nCompare mode: re-running {args.n_worlds} worlds as BASELINE (no flags)...")
        baseline_results = []
        for world_i in range(args.n_worlds):
            world_seed = world_i + args.seed
            bars_list = generate_correlated(args.n_bars, seed=world_seed)
            arena = MultiArena(cfg, max_leverage=0.65, exp_flags="")
            bars_dict_seq = [
                {"ES": bars_list[0][i], "NQ": bars_list[1][i], "YM": bars_list[2][i]}
                for i in range(args.n_bars)
            ]
            for bar_dict in bars_dict_seq:
                arena.step(bar_dict)
            bs = arena.stats()
            bs["seed"] = world_seed
            baseline_results.append(bs)

        b_sharpes = sorted(r["sharpe"] for r in baseline_results)
        b_returns = sorted(r["total_return_pct"] for r in baseline_results)
        b_mid = len(baseline_results) // 2

        print(f"\n{'='*65}")
        print(f"  BASELINE vs {flags_label}")
        print(f"{'='*65}")
        print(f"  Median Sharpe  BASELINE={b_sharpes[b_mid]:.3f}  {flags_label}={sharpes[mid]:.3f}")
        print(f"  Median Return  BASELINE={b_returns[b_mid]:+.1f}%  {flags_label}={returns[mid]:+.1f}%")
        print(f"{'='*65}")

        out2 = "results/arena_multi_compare.json"
        with open(out2, "w") as f:
            json.dump({
                "baseline": {"flags": "BASELINE", "results": baseline_results},
                "variant":  {"flags": flags_label, "results": results},
            }, f, indent=2)
        print(f"Compare results -> {out2}")


def _print_stats(s: dict, label: str = ""):
    print(f"\n{'='*55}")
    print(f"  arena_multi {label}")
    print(f"{'='*55}")
    print(f"  Return   : {s.get('total_return_pct', 0):+.2f}%")
    print(f"  MaxDD    : {s.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Sharpe   : {s.get('sharpe', 0):.3f}")
    print(f"  Trades   : {s.get('trade_count', 0)}")
    print(f"  Win Rate : {s.get('win_rate', 0):.1%}")
    print(f"  Conv Evt : {s.get('convergence_count', 0)}")
    print(f"  Conv WR  : {s.get('convergence_win_rate', 0):.1%}")
    print(f"  Conv PnL : {s.get('convergence_pnl_pct', 0):+.2f}%")
    print(f"  Multi WR : {s.get('multi_win_rate', 0):.1%}")
    print(f"  Solo WR  : {s.get('solo_win_rate', 0):.1%}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
