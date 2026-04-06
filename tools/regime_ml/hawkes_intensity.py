"""
hawkes_intensity.py — Hawkes process model for order-flow clustering.

A univariate Hawkes process (self-exciting point process) models the
conditional intensity of trade arrivals:

    λ(t) = μ + Σ_{t_i < t} α · exp(−β · (t − t_i))

Parameters
----------
μ (mu)     : baseline / background intensity  [trades/second]
α (alpha)  : excitation magnitude
β (beta)   : decay rate  (larger β → faster decay)

Branching ratio  n* = α / β   (must be < 1 for stationarity)

Key features
    - MLE fitting on a batch of event times (L-BFGS-B)
    - Real-time recursive intensity update (O(1) per event)
    - Trade clustering index: λ(t) / μ — ratio above baseline
    - Intraday intensity profile: expected intensity by hour
    - Adversarial detection: coefficient of variation of inter-arrival
      times much less than 1.0 suggests algorithmic wash-trading
    - Reads from data/live_trades.db (SQLite) if available

Persistence
    live_trades.db schema expected:
        CREATE TABLE trades (
            id        INTEGER PRIMARY KEY,
            symbol    TEXT,
            ts        REAL,    -- UNIX timestamp
            price     REAL,
            size      REAL
        )
"""

from __future__ import annotations

import math
import sqlite3
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB = Path(__file__).parent.parent.parent / "data" / "live_trades.db"
_MIN_EVENTS  = 50        # minimum events required for MLE
_REFIT_EVERY = 200       # refit every N events
_MAX_BUFFER  = 5_000     # rolling buffer of event timestamps

# Adversarial detection thresholds
_CV_WASH_THRESHOLD  = 0.15   # CV of inter-arrivals < this → suspiciously regular
_CLUSTER_HIGH       = 5.0    # λ/μ ratio above this → crowded / high clustering
_CLUSTER_LOW        = 0.5    # λ/μ ratio below this → quiet


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HawkesIntensityResult:
    symbol: str
    ts: float
    mu: float
    alpha: float
    beta: float
    branching_ratio: float      # alpha / beta
    current_intensity: float    # λ(t) at query time
    clustering_index: float     # λ(t) / μ
    is_crowded: bool
    is_quiet: bool
    adversarial_flag: bool      # suspiciously regular arrivals
    intraday_hour: int          # 0-23 UTC
    intraday_baseline: float    # typical intensity for this hour


# ---------------------------------------------------------------------------
# Recursive intensity tracker (O(1) per event)
# ---------------------------------------------------------------------------


class _RecursiveIntensity:
    """
    Maintains the running sum  A(t) = Σ_{t_i < t} exp(-β*(t-t_i))
    using the recursion A(t_n) = exp(-β*(t_n - t_{n-1})) * (1 + A(t_{n-1})).
    """

    def __init__(self, mu: float, alpha: float, beta: float) -> None:
        self.mu    = mu
        self.alpha = alpha
        self.beta  = beta
        self._A: float = 0.0
        self._last_t: float = 0.0

    def update(self, t: float) -> float:
        """Record a new event at time t and return λ(t^+)."""
        if self._last_t > 0:
            self._A = math.exp(-self.beta * (t - self._last_t)) * (1.0 + self._A)
        else:
            self._A = 0.0
        self._last_t = t
        return self.mu + self.alpha * self._A

    def query(self, t: float) -> float:
        """Return λ(t) without recording a new event."""
        if self._last_t <= 0:
            return self.mu
        A_now = math.exp(-self.beta * (t - self._last_t)) * self._A
        return self.mu + self.alpha * A_now

    def reset(self, mu: float, alpha: float, beta: float) -> None:
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self._A = 0.0
        self._last_t = 0.0


# ---------------------------------------------------------------------------
# MLE fitting
# ---------------------------------------------------------------------------


def _hawkes_log_likelihood(
    params: np.ndarray,
    times: np.ndarray,
    T: float,
) -> float:
    """
    Negative log-likelihood of a univariate Hawkes process.

    Uses the efficient O(n) recursive formula.
    """
    mu, alpha, beta = float(params[0]), float(params[1]), float(params[2])
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return 1e10  # infeasible

    n = len(times)
    log_lam_sum = 0.0
    A = 0.0
    for i in range(n):
        if i > 0:
            A = math.exp(-beta * (times[i] - times[i - 1])) * (1.0 + A)
        lam = mu + alpha * A
        if lam <= 1e-300:
            return 1e10
        log_lam_sum += math.log(lam)

    # Integral term: μT + α/β * Σ (1 - exp(-β*(T-t_i)))
    integral = mu * T
    for ti in times:
        integral += (alpha / beta) * (1.0 - math.exp(-beta * (T - ti)))

    return -(log_lam_sum - integral)


def fit_hawkes_mle(
    times: np.ndarray,
    T: Optional[float] = None,
    n_restarts: int = 5,
    seed: int = 42,
) -> Tuple[float, float, float, bool]:
    """
    Fit Hawkes(μ, α, β) by MLE.

    Parameters
    ----------
    times : sorted array of event times (seconds)
    T     : observation window end time; defaults to times[-1] + 1
    n_restarts : number of random restarts

    Returns
    -------
    (mu, alpha, beta, converged)
    """
    if len(times) < _MIN_EVENTS:
        return 0.01, 0.5, 1.0, False

    times = np.sort(times.astype(float))
    if T is None:
        T = float(times[-1]) + 1.0

    mean_rate = len(times) / T
    rng = np.random.default_rng(seed)
    best_nll = math.inf
    best_params = (mean_rate * 0.5, 0.5, 1.0)

    for _ in range(n_restarts):
        mu0    = rng.uniform(0.01, mean_rate * 0.9)
        alpha0 = rng.uniform(0.01, 0.8)
        beta0  = rng.uniform(alpha0 + 0.01, alpha0 + 2.0)
        x0 = np.array([mu0, alpha0, beta0])
        bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]
        try:
            res = minimize(
                _hawkes_log_likelihood,
                x0,
                args=(times, T),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if res.success and res.fun < best_nll:
                best_nll = res.fun
                best_params = tuple(res.x)
        except Exception:
            continue

    mu, alpha, beta = best_params
    converged = math.isfinite(best_nll)
    # Enforce stationarity
    if alpha >= beta:
        alpha = beta * 0.9
    return float(mu), float(alpha), float(beta), converged


# ---------------------------------------------------------------------------
# Intraday intensity profile
# ---------------------------------------------------------------------------


class _IntradayProfile:
    """
    Tracks the expected intensity for each hour-of-day bucket.
    Uses an exponential moving average.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha
        self._profile: np.ndarray = np.full(24, float("nan"))

    def update(self, hour: int, intensity: float) -> None:
        prev = self._profile[hour]
        if math.isnan(prev):
            self._profile[hour] = intensity
        else:
            self._profile[hour] = (1 - self._alpha) * prev + self._alpha * intensity

    def baseline(self, hour: int) -> float:
        v = self._profile[hour]
        return float(v) if math.isfinite(v) else 0.0

    def as_dict(self) -> Dict[int, float]:
        return {h: float(v) for h, v in enumerate(self._profile) if math.isfinite(v)}


# ---------------------------------------------------------------------------
# Adversarial detection
# ---------------------------------------------------------------------------


def _adversarial_check(
    times: np.ndarray,
    cv_threshold: float = _CV_WASH_THRESHOLD,
) -> bool:
    """
    Return True if inter-arrival spacing is suspiciously regular
    (coefficient of variation far below 1.0 expected for Poisson).
    """
    if len(times) < 20:
        return False
    iats = np.diff(np.sort(times))
    if iats.mean() < 1e-9:
        return False
    cv = iats.std() / iats.mean()
    return bool(cv < cv_threshold)


# ---------------------------------------------------------------------------
# Per-symbol state
# ---------------------------------------------------------------------------


@dataclass
class _SymbolState:
    symbol: str
    events: Deque[float] = field(default_factory=lambda: deque(maxlen=_MAX_BUFFER))
    event_count: int = 0
    last_fit_count: int = 0
    mu: float = 0.01
    alpha: float = 0.3
    beta: float = 1.0
    converged: bool = False
    tracker: _RecursiveIntensity = field(
        default_factory=lambda: _RecursiveIntensity(0.01, 0.3, 1.0)
    )
    profile: _IntradayProfile = field(default_factory=_IntradayProfile)


# ---------------------------------------------------------------------------
# HawkesProcess
# ---------------------------------------------------------------------------


class HawkesProcess:
    """
    Self-exciting Hawkes process for per-symbol trade clustering.

    Parameters
    ----------
    db_path : str | Path
        Path to live_trades.db (SQLite).  Pass None to disable DB reads.
    min_events : int
        Minimum events before fitting.
    refit_every : int
        Re-fit MLE every N new events.
    mle_restarts : int
        Number of random restarts in MLE optimisation.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = _DEFAULT_DB,
        min_events: int = _MIN_EVENTS,
        refit_every: int = _REFIT_EVERY,
        mle_restarts: int = 3,
    ) -> None:
        self.db_path = Path(db_path) if db_path else None
        self.min_events = min_events
        self.refit_every = refit_every
        self.mle_restarts = mle_restarts
        self._states: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # Stream interface
    # ------------------------------------------------------------------

    def update(
        self,
        symbol: str,
        ts: float,
    ) -> Optional[HawkesIntensityResult]:
        """
        Record a new trade event for *symbol* at time *ts*.

        Returns a HawkesIntensityResult after the model has been fitted,
        else None (during warm-up).
        """
        st = self._get_state(symbol)
        st.events.append(ts)
        st.event_count += 1

        # Update recursive intensity tracker
        lam = st.tracker.update(ts)

        # Refit periodically
        should_fit = (
            st.event_count >= self.min_events
            and (st.event_count - st.last_fit_count) >= self.refit_every
        )
        if should_fit:
            self._refit(st)

        if not st.converged:
            return None

        hour = int((ts % 86400) // 3600)
        st.profile.update(hour, lam)

        branching = st.alpha / st.beta if st.beta > 0 else 0.0
        clustering_idx = lam / st.mu if st.mu > 0 else 1.0

        # Adversarial check on last 200 events
        recent = np.array(list(st.events)[-200:])
        adv = _adversarial_check(recent)

        return HawkesIntensityResult(
            symbol=symbol,
            ts=ts,
            mu=st.mu,
            alpha=st.alpha,
            beta=st.beta,
            branching_ratio=branching,
            current_intensity=lam,
            clustering_index=clustering_idx,
            is_crowded=clustering_idx > _CLUSTER_HIGH,
            is_quiet=clustering_idx < _CLUSTER_LOW,
            adversarial_flag=adv,
            intraday_hour=hour,
            intraday_baseline=st.profile.baseline(hour),
        )

    def query_intensity(self, symbol: str, ts: Optional[float] = None) -> float:
        """Return current intensity estimate without recording a new event."""
        st = self._states.get(symbol)
        if st is None or not st.converged:
            return 0.0
        t = ts if ts is not None else time.time()
        return st.tracker.query(t)

    # ------------------------------------------------------------------
    # DB bulk-load
    # ------------------------------------------------------------------

    def load_from_db(
        self,
        symbol: str,
        limit: int = 5000,
        since: Optional[float] = None,
    ) -> int:
        """
        Load recent trades from live_trades.db for *symbol*.
        Returns number of events loaded.
        """
        if self.db_path is None or not self.db_path.exists():
            return 0
        try:
            conn = sqlite3.connect(str(self.db_path))
            q = "SELECT ts FROM trades WHERE symbol=?"
            params: list = [symbol]
            if since is not None:
                q += " AND ts >= ?"
                params.append(since)
            q += " ORDER BY ts DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(q, params).fetchall()
            conn.close()
        except Exception:
            return 0

        for (ts_val,) in reversed(rows):
            self.update(symbol, float(ts_val))
        return len(rows)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState(symbol=symbol)
        return self._states[symbol]

    def _refit(self, st: _SymbolState) -> None:
        times = np.array(list(st.events), dtype=float)
        # Normalise to [0, T] to avoid numerical issues
        t0 = times[0]
        times_norm = times - t0
        T = float(times_norm[-1]) + 1.0
        mu, alpha, beta, converged = fit_hawkes_mle(
            times_norm, T=T, n_restarts=self.mle_restarts
        )
        if converged:
            st.mu = mu
            st.alpha = alpha
            st.beta = beta
            st.converged = True
            st.tracker.reset(mu, alpha, beta)
            # Re-warm tracker with all buffered events
            for t in times_norm:
                st.tracker.update(t)
        st.last_fit_count = st.event_count

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def params(self, symbol: str) -> Optional[Dict]:
        st = self._states.get(symbol)
        if st is None:
            return None
        br = st.alpha / st.beta if st.beta > 0 else 0.0
        return {
            "mu": st.mu,
            "alpha": st.alpha,
            "beta": st.beta,
            "branching_ratio": br,
            "stable": br < 1.0,
            "event_count": st.event_count,
            "converged": st.converged,
        }

    def intraday_profile(self, symbol: str) -> Dict[int, float]:
        st = self._states.get(symbol)
        return st.profile.as_dict() if st else {}

    def summary(self, symbol: str) -> str:
        p = self.params(symbol)
        if p is None:
            return f"No data for {symbol}"
        return (
            f"Hawkes({symbol}): μ={p['mu']:.4f}  α={p['alpha']:.4f}  "
            f"β={p['beta']:.4f}  n*={p['branching_ratio']:.3f}  "
            f"stable={p['stable']}  events={p['event_count']}"
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    rng = np.random.default_rng(7)
    # Simulate a Hawkes process with μ=0.02, α=0.6, β=1.2  (n*=0.5)
    mu, alpha, beta = 0.02, 0.6, 1.2
    times: List[float] = []
    t = 0.0
    T = 5000.0
    A = 0.0
    while t < T:
        lam = mu + alpha * A
        dt = rng.exponential(1.0 / lam)
        t += dt
        if t >= T:
            break
        A = math.exp(-beta * dt) * (A + 1.0)
        times.append(t)

    print(f"Simulated {len(times)} events over T={T}s")

    hp = HawkesProcess(db_path=None, min_events=50, refit_every=100)
    last_result = None
    for ts in times:
        r = hp.update("SIM", ts)
        if r is not None:
            last_result = r

    print(hp.summary("SIM"))
    p = hp.params("SIM")
    if p:
        print(f"True params: μ={mu}  α={alpha}  β={beta}  n*={alpha/beta:.3f}")
        print(f"Fitted:      μ={p['mu']:.4f}  α={p['alpha']:.4f}  β={p['beta']:.4f}  n*={p['branching_ratio']:.3f}")
    if last_result:
        print(f"Last intensity: {last_result.current_intensity:.4f}  "
              f"clustering_idx={last_result.clustering_index:.2f}  "
              f"adversarial={last_result.adversarial_flag}")


if __name__ == "__main__":
    _demo()
