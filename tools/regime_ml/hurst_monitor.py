"""
hurst_monitor.py — Hurst exponent and fractal analysis.

Three independent Hurst estimators:
    1.  R/S analysis   (classical Hurst, Hurst 1951)
    2.  Detrended Fluctuation Analysis (DFA, Peng et al. 1994)
    3.  Variogram / structure-function method

Ensemble Hurst: weighted average (weights learned from stability of each
method across bootstrap resamples).

Regime classification
    H < 0.40            mean-reverting  (anti-persistent)
    0.40 ≤ H ≤ 0.60     random walk     (no long memory)
    H > 0.60            trending        (persistent)

Per-symbol rolling computation on configurable window sizes (100/250/500).
Feeds calibration signals to OU-process detectors.

Alerts are raised when H crosses a regime boundary between consecutive
windows.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------


class HurstRegime(str, Enum):
    MEAN_REVERTING = "mean_reverting"   # H < 0.40
    RANDOM_WALK    = "random_walk"      # 0.40 <= H <= 0.60
    TRENDING       = "trending"         # H > 0.60
    UNKNOWN        = "unknown"


def _classify(h: float) -> HurstRegime:
    if not math.isfinite(h):
        return HurstRegime.UNKNOWN
    if h < 0.40:
        return HurstRegime.MEAN_REVERTING
    if h <= 0.60:
        return HurstRegime.RANDOM_WALK
    return HurstRegime.TRENDING


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HurstResult:
    symbol: str
    timeframe: str          # "15m" / "1h" / "4h"
    window: int             # bars used
    h_rs: float             # R/S estimate
    h_dfa: float            # DFA estimate
    h_var: float            # Variogram estimate
    h_ensemble: float       # weighted average
    regime: HurstRegime
    regime_changed: bool    # True if different from previous window
    prev_regime: HurstRegime
    ou_theta_hint: float    # suggested OU mean-reversion speed ≈ -2*ln(2*H-1) (heuristic)


# ---------------------------------------------------------------------------
# R/S analysis
# ---------------------------------------------------------------------------


def hurst_rs(series: np.ndarray, min_chunk: int = 8) -> float:
    """
    Hurst exponent via Rescaled Range (R/S) analysis.

    Fits log(R/S) ~ H * log(n) across multiple chunk sizes.
    """
    if len(series) < 2 * min_chunk:
        return float("nan")

    series = np.asarray(series, dtype=float)
    n = len(series)
    # chunk sizes: powers of 2 up to n/4
    chunk_sizes = []
    s = min_chunk
    while s <= n // 2:
        chunk_sizes.append(s)
        s = int(s * 1.5)
    if not chunk_sizes:
        return float("nan")

    log_n, log_rs = [], []
    for size in chunk_sizes:
        rs_vals = []
        for start in range(0, n - size + 1, size):
            chunk = series[start: start + size]
            mean = chunk.mean()
            devs = np.cumsum(chunk - mean)
            rng  = devs.max() - devs.min()
            std  = chunk.std(ddof=1)
            if std < 1e-12:
                continue
            rs_vals.append(rng / std)
        if rs_vals:
            log_n.append(math.log(size))
            log_rs.append(math.log(np.mean(rs_vals)))

    if len(log_n) < 2:
        return float("nan")

    # OLS slope
    log_n_arr = np.array(log_n)
    log_rs_arr = np.array(log_rs)
    slope = float(np.polyfit(log_n_arr, log_rs_arr, 1)[0])
    return max(0.0, min(1.0, slope))


# ---------------------------------------------------------------------------
# DFA
# ---------------------------------------------------------------------------


def hurst_dfa(series: np.ndarray, min_chunk: int = 8) -> float:
    """
    Hurst exponent via Detrended Fluctuation Analysis.

    Integrates the series first, then fits local linear trends inside
    non-overlapping windows of increasing size.
    """
    if len(series) < 2 * min_chunk:
        return float("nan")

    series = np.asarray(series, dtype=float)
    series = series - series.mean()
    y = np.cumsum(series)          # integrated process
    n = len(y)

    chunk_sizes = []
    s = min_chunk
    while s <= n // 2:
        chunk_sizes.append(s)
        s = int(s * 1.5)

    log_n, log_f = [], []
    for size in chunk_sizes:
        segments = n // size
        if segments < 2:
            continue
        f2 = 0.0
        x = np.arange(size, dtype=float)
        for seg in range(segments):
            chunk = y[seg * size: (seg + 1) * size]
            # linear detrend
            coeffs = np.polyfit(x, chunk, 1)
            trend  = np.polyval(coeffs, x)
            resid  = chunk - trend
            f2 += float(np.mean(resid ** 2))
        f2 /= segments
        log_n.append(math.log(size))
        log_f.append(0.5 * math.log(f2))

    if len(log_n) < 2:
        return float("nan")

    slope = float(np.polyfit(np.array(log_n), np.array(log_f), 1)[0])
    return max(0.0, min(1.0, slope))


# ---------------------------------------------------------------------------
# Variogram (structure function) method
# ---------------------------------------------------------------------------


def hurst_variogram(series: np.ndarray, max_lag: Optional[int] = None) -> float:
    """
    Hurst exponent via the variogram / second-order structure function.

    V(tau) = E[ (X(t+tau) - X(t))^2 ] ~ tau^(2H)
    Fit log V(tau) ~ 2H * log(tau).
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 20:
        return float("nan")

    if max_lag is None:
        max_lag = min(n // 4, 50)

    lags = range(1, max_lag + 1)
    log_tau, log_var = [], []
    for tau in lags:
        diffs = series[tau:] - series[:-tau]
        v = float(np.mean(diffs ** 2))
        if v > 1e-20:
            log_tau.append(math.log(tau))
            log_var.append(math.log(v))

    if len(log_tau) < 2:
        return float("nan")

    slope = float(np.polyfit(np.array(log_tau), np.array(log_var), 1)[0])
    H = slope / 2.0
    return max(0.0, min(1.0, H))


# ---------------------------------------------------------------------------
# Ensemble Hurst
# ---------------------------------------------------------------------------

# Default weights (can be tuned via bootstrap stability)
_DEFAULT_WEIGHTS = {"rs": 0.4, "dfa": 0.4, "var": 0.2}


def hurst_ensemble(
    series: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, float, float]:
    """
    Compute all three Hurst estimates and return a weighted average.

    Returns
    -------
    (h_rs, h_dfa, h_var, h_ensemble)
    """
    w = weights or _DEFAULT_WEIGHTS
    h_rs  = hurst_rs(series)
    h_dfa = hurst_dfa(series)
    h_var = hurst_variogram(series)

    valid = {
        "rs":  h_rs  if math.isfinite(h_rs)  else None,
        "dfa": h_dfa if math.isfinite(h_dfa) else None,
        "var": h_var if math.isfinite(h_var) else None,
    }
    total_w = sum(w[k] for k, v in valid.items() if v is not None)
    if total_w < 1e-9:
        ensemble = float("nan")
    else:
        ensemble = sum(w[k] * v for k, v in valid.items() if v is not None) / total_w

    return h_rs, h_dfa, h_var, ensemble


# ---------------------------------------------------------------------------
# OU theta hint
# ---------------------------------------------------------------------------


def _ou_theta_hint(h: float) -> float:
    """
    Heuristic mapping of H → OU mean-reversion speed.
    For H close to 0 (strong mean-reversion) θ is large; for H→0.5 θ→0.
    """
    if not math.isfinite(h) or h >= 0.5:
        return 0.0
    # θ ≈ -2 * ln(2H - 1) is ill-defined; use empirical mapping instead:
    # θ ≈ 2 * (0.5 - H) / 0.1  (linear in distance from 0.5)
    return max(0.0, 2.0 * (0.5 - h) / 0.1)


# ---------------------------------------------------------------------------
# Per-symbol rolling state
# ---------------------------------------------------------------------------


@dataclass
class _SymbolState:
    symbol: str
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    returns: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    bar_count: int = 0
    # Last known regime per (timeframe, window)
    prev_regimes: Dict[Tuple[str, int], HurstRegime] = field(default_factory=dict)
    # Callbacks
    alert_callbacks: List[Callable[[HurstResult], None]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HurstMonitor
# ---------------------------------------------------------------------------


class HurstMonitor:
    """
    Rolling Hurst exponent monitor for multiple symbols and timeframes.

    Parameters
    ----------
    timeframes : list[str]
        Which timeframe labels to track.  Purely metadata — the caller
        is responsible for feeding bars at the correct frequency.
    windows : list[int]
        Rolling window sizes (in bars) to compute Hurst over.
    weights : dict
        Relative weights for R/S, DFA, and Variogram in the ensemble.
    recompute_every : int
        Recompute Hurst every N bars (not every bar — expensive).
    """

    # Default timeframe → bars mapping (informational only)
    TIMEFRAME_BARS = {"15m": 100, "1h": 250, "4h": 500}

    def __init__(
        self,
        timeframes: Sequence[str] = ("15m", "1h", "4h"),
        windows: Sequence[int] = (100, 250, 500),
        weights: Optional[Dict[str, float]] = None,
        recompute_every: int = 5,
    ) -> None:
        self.timeframes = list(timeframes)
        self.windows = sorted(windows)
        self.weights = weights or _DEFAULT_WEIGHTS
        self.recompute_every = recompute_every

        self._states: Dict[str, _SymbolState] = {}
        self._results_cache: Dict[Tuple[str, str, int], HurstResult] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        symbol: str,
        price: float,
        timeframe: str = "1h",
    ) -> List[HurstResult]:
        """
        Ingest a new price bar.

        Returns a list of HurstResult objects (one per configured window)
        if Hurst was recomputed this bar, else an empty list.
        """
        st = self._get_state(symbol)
        if len(st.prices) > 0 and st.prices[-1] > 0:
            ret = math.log(price / st.prices[-1])
        else:
            ret = 0.0

        st.prices.append(price)
        st.returns.append(ret)
        st.bar_count += 1

        if st.bar_count % self.recompute_every != 0:
            return []

        results: List[HurstResult] = []
        for window in self.windows:
            if len(st.returns) < window:
                continue
            data = np.array(list(st.returns)[-window:], dtype=float)
            result = self._compute(symbol, timeframe, window, data, st)
            self._results_cache[(symbol, timeframe, window)] = result
            results.append(result)
            # Fire alert callbacks on regime change
            if result.regime_changed:
                for cb in st.alert_callbacks:
                    try:
                        cb(result)
                    except Exception:
                        pass
        return results

    def get_latest(
        self,
        symbol: str,
        timeframe: str = "1h",
        window: int = 250,
    ) -> Optional[HurstResult]:
        """Return the most recently computed HurstResult for a symbol."""
        return self._results_cache.get((symbol, timeframe, window))

    def register_alert(
        self,
        symbol: str,
        callback: Callable[[HurstResult], None],
    ) -> None:
        """Register a callback that fires when H crosses a regime boundary."""
        self._get_state(symbol).alert_callbacks.append(callback)

    def ensemble_hurst(self, symbol: str, window: int = 250) -> Optional[float]:
        """Quick accessor — returns ensemble H or None."""
        result = self._results_cache.get((symbol, "1h", window))
        return result.h_ensemble if result else None

    def ou_theta(self, symbol: str, window: int = 250) -> Optional[float]:
        """Return OU mean-reversion hint for the symbol."""
        result = self._results_cache.get((symbol, "1h", window))
        return result.ou_theta_hint if result else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute(
        self,
        symbol: str,
        timeframe: str,
        window: int,
        data: np.ndarray,
        st: _SymbolState,
    ) -> HurstResult:
        h_rs, h_dfa, h_var, h_ens = hurst_ensemble(data, self.weights)
        regime = _classify(h_ens)
        key = (timeframe, window)
        prev = st.prev_regimes.get(key, HurstRegime.UNKNOWN)
        changed = (prev != HurstRegime.UNKNOWN) and (regime != prev)
        st.prev_regimes[key] = regime
        return HurstResult(
            symbol=symbol,
            timeframe=timeframe,
            window=window,
            h_rs=h_rs,
            h_dfa=h_dfa,
            h_var=h_var,
            h_ensemble=h_ens,
            regime=regime,
            regime_changed=changed,
            prev_regime=prev,
            ou_theta_hint=_ou_theta_hint(h_ens),
        )

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState(symbol=symbol)
        return self._states[symbol]

    # ------------------------------------------------------------------
    # Batch / back-fill
    # ------------------------------------------------------------------

    def feed_series(
        self,
        symbol: str,
        prices: Sequence[float],
        timeframe: str = "1h",
    ) -> List[HurstResult]:
        """Feed an entire price series at once and return all results."""
        all_results: List[HurstResult] = []
        for p in prices:
            r = self.update(symbol, float(p), timeframe)
            all_results.extend(r)
        return all_results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, symbol: str) -> str:
        lines = [f"HurstMonitor summary for {symbol}"]
        for (sym, tf, win), res in sorted(self._results_cache.items()):
            if sym != symbol:
                continue
            lines.append(
                f"  [{tf:3s} w={win:4d}]  H_ens={res.h_ensemble:.3f}  "
                f"H_rs={res.h_rs:.3f}  H_dfa={res.h_dfa:.3f}  H_var={res.h_var:.3f}  "
                f"regime={res.regime.value}"
                + ("  *** REGIME CHANGE ***" if res.regime_changed else "")
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import csv
    from pathlib import Path

    csv_path = Path(__file__).parent.parent.parent / "data" / "NDX_hourly_poly.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(open(csv_path)))
        prices = [float(r.get("close", r.get("Close", 0))) for r in rows[:800]]
    else:
        rng = np.random.default_rng(0)
        # trending segment then mean-reverting
        p1 = 15000.0 + np.cumsum(rng.normal(0.5, 10, 400))
        p2 = p1[-1]  + np.cumsum(rng.normal(0.0, 20, 400))
        prices = np.concatenate([p1, p2]).tolist()

    monitor = HurstMonitor(windows=[100, 250], recompute_every=5)
    all_r = monitor.feed_series("NDX", prices, timeframe="1h")
    changes = [r for r in all_r if r.regime_changed]
    print(f"Total Hurst computations: {len(all_r)}")
    print(f"Regime changes detected:  {len(changes)}")
    for r in changes:
        print(
            f"  w={r.window}  {r.prev_regime.value} → {r.regime.value}  "
            f"H={r.h_ensemble:.3f}  θ_hint={r.ou_theta_hint:.2f}"
        )
    print(monitor.summary("NDX"))


if __name__ == "__main__":
    _demo()
