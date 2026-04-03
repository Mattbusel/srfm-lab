"""
fast_arena.py — Numba-JIT accelerated SRFM arena.
Falls back to pure numpy if numba not installed.
~10-50x faster than arena_v2 on large bar arrays.

Usage:
    from tools.fast_arena import run_fast, sweep_fast
    broker = run_fast(bars, cf=0.005, bh_form=1.5, bh_decay=0.95, max_lev=0.65)

    # Parallel sweep over cf values
    results = sweep_fast(bars, cf_values=[0.003, 0.005, 0.007], n_jobs=-1)
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Numba setup — optional
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit

    def numba_jit(fn):
        return _njit(fn, cache=True, fastmath=True)

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    def numba_jit(fn):  # type: ignore[misc]
        return fn
    _NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core simulation kernel
# ---------------------------------------------------------------------------

@numba_jit
def _srfm_core(
    closes: np.ndarray,
    cf: float,
    bh_form: float,
    bh_decay: float,
    bh_collapse: float,
    max_lev: float,
) -> tuple:
    """
    Pure numerical SRFM simulation kernel.

    Args:
        closes      : float64 array of close prices
        cf          : critical frequency
        bh_form     : BH formation threshold (default 1.5)
        bh_decay    : decay factor per SPACELIKE bar (default 0.95)
        bh_collapse : collapse threshold (default 1.0)
        max_lev     : maximum leverage fraction

    Returns:
        equity     : float64 array (starts at 1.0)
        positions  : float64 array of position fractions
        bh_mass_arr: float64 array of BH mass per bar
        bh_active_arr: bool array
        regime_arr : int8 array (0=BULL, 1=BEAR, 2=SIDEWAYS, 3=HV)
    """
    n = len(closes)

    equity = np.ones(n, dtype=np.float64)
    positions = np.zeros(n, dtype=np.float64)
    bh_mass_arr = np.zeros(n, dtype=np.float64)
    bh_active_arr = np.zeros(n, dtype=np.uint8)   # bool
    regime_arr = np.zeros(n, dtype=np.int8)        # 0=BULL

    # BH state
    mass = 0.0
    ctl = 0
    bh_active = False

    # EMA-20 for sign-of-trend
    ema20_k = 2.0 / 21.0
    ema20 = closes[0]

    for i in range(1, n):
        prev = closes[i - 1]
        cur = closes[i]

        # Update EMA-20
        ema20 = cur * ema20_k + ema20 * (1.0 - ema20_k)

        # SRFM physics
        if prev > 0.0:
            beta = abs(cur - prev) / (prev * cf)
        else:
            beta = 0.0

        if beta < 1.0:  # TIMELIKE
            mass = mass * 0.97 + 0.03 * 1.0
            ctl += 1
        else:           # SPACELIKE
            mass *= bh_decay
            ctl = 0

        # BH activation / collapse
        if bh_active:
            if mass < bh_collapse:
                bh_active = False
        else:
            if mass >= bh_form and ctl >= 5:
                bh_active = True

        # Regime (simple): BULL/BEAR based on EMA20, HV on high beta
        if beta >= 2.0:
            regime = 3  # HV
        elif cur > ema20:
            regime = 0  # BULL
        else:
            regime = 1  # BEAR

        # Position sizing
        sign = 1.0 if cur > ema20 else -1.0
        if bh_active:
            pos = max_lev * sign
        else:
            pos = max_lev * 0.5 * sign

        bh_mass_arr[i] = mass
        bh_active_arr[i] = 1 if bh_active else 0
        regime_arr[i] = regime
        positions[i] = pos

        # Equity update
        bar_ret = (cur / prev - 1.0) if prev > 0.0 else 0.0
        equity[i] = equity[i - 1] * (1.0 + positions[i - 1] * bar_ret)

    return equity, positions, bh_mass_arr, bh_active_arr, regime_arr


# ---------------------------------------------------------------------------
# Metrics helpers (pure Python / numpy — not in JIT kernel)
# ---------------------------------------------------------------------------

def _sharpe(equity: np.ndarray) -> float:
    """Annualised Sharpe for hourly bars."""
    returns = np.diff(equity) / equity[:-1]
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return 0.0
    return float(math.sqrt(252 * 24) * np.mean(returns) / std)


def _max_dd(equity: np.ndarray) -> float:
    """Maximum drawdown as a fraction (0–1)."""
    peak = equity[0]
    max_drawdown = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = 1.0 - v / peak
        if dd > max_drawdown:
            max_drawdown = dd
    return max_drawdown


def _trade_count(positions: np.ndarray) -> int:
    """Count position direction changes."""
    signs = np.sign(positions)
    changes = np.sum(np.abs(np.diff(signs)) > 0)
    return int(changes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FastBroker:
    """Lightweight result container — mirrors SimulatedBroker.stats() interface."""

    def __init__(
        self,
        equity: np.ndarray,
        positions: np.ndarray,
        bh_mass: np.ndarray,
        bh_active: np.ndarray,
        regime: np.ndarray,
    ):
        self.equity = equity
        self.positions = positions
        self.bh_mass = bh_mass
        self.bh_active = bh_active
        self.regime = regime
        self._sharpe = _sharpe(equity)
        self._max_dd = _max_dd(equity)
        self._trades = _trade_count(positions)
        self.final_equity = float(equity[-1])
        self.total_return = self.final_equity - 1.0

    def stats(self) -> Dict[str, Any]:
        return {
            "sharpe": round(self._sharpe, 4),
            "total_return_pct": round(self.total_return * 100.0, 4),
            "max_drawdown_pct": round(self._max_dd * 100.0, 4),
            "trade_count": self._trades,
            "final_equity": round(self.final_equity, 6),
        }


def run_fast(
    bars: List[dict],
    cf: float = 0.005,
    bh_form: float = 1.5,
    bh_decay: float = 0.95,
    bh_collapse: float = 1.0,
    max_lev: float = 0.65,
) -> FastBroker:
    """
    Run SRFM fast arena on a list of OHLCV bar dicts.

    Returns a FastBroker result object with .stats() method.
    """
    closes = np.array([float(b["close"]) for b in bars], dtype=np.float64)
    equity, positions, bh_mass, bh_active, regime = _srfm_core(
        closes, cf, bh_form, bh_decay, bh_collapse, max_lev
    )
    return FastBroker(equity, positions, bh_mass, bh_active, regime)


def run_fast_closes(
    closes: np.ndarray,
    cf: float = 0.005,
    bh_form: float = 1.5,
    bh_decay: float = 0.95,
    bh_collapse: float = 1.0,
    max_lev: float = 0.65,
) -> FastBroker:
    """Run fast arena on a raw numpy closes array."""
    equity, positions, bh_mass, bh_active, regime = _srfm_core(
        closes.astype(np.float64), cf, bh_form, bh_decay, bh_collapse, max_lev
    )
    return FastBroker(equity, positions, bh_mass, bh_active, regime)


# ---------------------------------------------------------------------------
# Parallel sweep
# ---------------------------------------------------------------------------

def _run_one(closes_arr: np.ndarray, cf: float, max_lev: float) -> Dict[str, Any]:
    br = run_fast_closes(closes_arr, cf=cf, max_lev=max_lev)
    s = br.stats()
    return {"cf": cf, "max_lev": max_lev, **s}


def sweep_fast(
    bars: List[dict],
    param_grid: Optional[Dict[str, List[float]]] = None,
    n_jobs: int = -1,
) -> "Any":
    """
    Parallel parameter sweep.

    param_grid : dict of lists, e.g.
        {"cf": [0.003, 0.005, 0.007], "max_lev": [0.5, 0.65, 0.8]}
    n_jobs     : -1 = all CPU cores (uses joblib if available)

    Returns pandas DataFrame with columns: cf, max_lev, sharpe, return_pct,
    max_dd, trade_count.
    """
    import pandas as pd  # type: ignore[import]

    if param_grid is None:
        param_grid = {
            "cf": [0.003, 0.005, 0.007],
            "max_lev": [0.5, 0.65, 0.80],
        }

    cf_vals = param_grid.get("cf", [0.005])
    lev_vals = param_grid.get("max_lev", [0.65])
    combos = list(itertools.product(cf_vals, lev_vals))

    closes_arr = np.array([float(b["close"]) for b in bars], dtype=np.float64)

    try:
        from joblib import Parallel, delayed  # type: ignore[import]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_one)(closes_arr, cf, lev) for cf, lev in combos
        )
    except ImportError:
        results = [_run_one(closes_arr, cf, lev) for cf, lev in combos]

    df = pd.DataFrame(results)
    # Rename columns to match requested schema
    df = df.rename(columns={
        "total_return_pct": "return_pct",
        "max_drawdown_pct": "max_dd",
    })
    cols = ["cf", "max_lev", "sharpe", "return_pct", "max_dd", "trade_count"]
    return df[[c for c in cols if c in df.columns]]


# ---------------------------------------------------------------------------
# CLI / benchmark
# ---------------------------------------------------------------------------

def _benchmark_vs_v2(n_bars: int = 10_000) -> None:
    """Compare fast_arena vs arena_v2 timing."""
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

    # Generate synthetic bars
    try:
        from arena_v2 import generate_synthetic  # type: ignore[import]
        bars = generate_synthetic(n_bars, seed=42)
    except Exception:
        rng = np.random.default_rng(42)
        price = 4500.0
        bars = []
        for _ in range(n_bars):
            ret = 0.0002 + 0.001 * float(rng.standard_normal())
            ret = max(-0.05, min(0.05, ret))
            close = price * (1.0 + ret)
            bars.append({
                "date": "x", "open": price,
                "high": close * 1.001, "low": close * 0.999,
                "close": close, "volume": 50000.0,
            })
            price = close

    # Warm-up JIT
    _srfm_core(
        np.array([b["close"] for b in bars[:100]], dtype=np.float64),
        0.005, 1.5, 0.95, 1.0, 0.65
    )

    # Time fast_arena
    t0 = time.perf_counter()
    br = run_fast(bars, cf=0.005)
    t_fast = time.perf_counter() - t0

    print(f"\n{'='*55}")
    print(f"  fast_arena — {n_bars:,} bars")
    print(f"  Numba available : {_NUMBA_AVAILABLE}")
    print(f"  Time            : {t_fast*1000:.1f} ms")
    s = br.stats()
    print(f"  Sharpe          : {s['sharpe']:.3f}")
    print(f"  Return          : {s['total_return_pct']:+.2f}%")
    print(f"  MaxDD           : {s['max_drawdown_pct']:.2f}%")
    print(f"  Trades          : {s['trade_count']}")

    # Time arena_v2 if available
    try:
        from arena_v2 import run_v2  # type: ignore[import]
        cfg = {"cf": 0.005, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95}
        t1 = time.perf_counter()
        broker_v2, _ = run_v2(bars, cfg, 0.65, "", verbose=False)
        t_v2 = time.perf_counter() - t1
        speedup = t_v2 / t_fast if t_fast > 0 else float("inf")
        print(f"\n  arena_v2 time   : {t_v2*1000:.1f} ms")
        print(f"  Speedup         : {speedup:.1f}x")
    except Exception as exc:
        print(f"\n  arena_v2 not available for comparison: {exc}")

    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="fast_arena — Numba SRFM benchmark")
    parser.add_argument("--benchmark", action="store_true", help="Run timing benchmark")
    parser.add_argument("--n-bars", type=int, default=10_000)
    parser.add_argument("--cf", type=float, default=0.005)
    parser.add_argument("--max-lev", type=float, default=0.65)
    args = parser.parse_args()

    if args.benchmark:
        _benchmark_vs_v2(args.n_bars)
    else:
        # Quick self-test
        rng = np.random.default_rng(0)
        closes = np.cumprod(1.0 + rng.normal(0, 0.001, 1000)) * 100.0
        bars = [{"close": float(c)} for c in closes]
        br = run_fast(bars, cf=args.cf, max_lev=args.max_lev)
        print(br.stats())


if __name__ == "__main__":
    main()
