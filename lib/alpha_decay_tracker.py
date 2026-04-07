"""
alpha_decay_tracker.py -- tracks IC decay for active alpha signals across timeframes.

Signals with IC < 0.02 for 5 consecutive cycles are automatically flagged
for retirement. Supports exponential decay fitting and SQLite persistence.
"""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# -- decay constants
DECAY_LAMBDA_THRESHOLD = 0.1       # half-life < 7 bars triggers is_decaying
MIN_OBS_FOR_FIT = 5                # minimum observations required for decay fit
DEFAULT_RETIRE_IC = 0.02           # IC below this triggers retirement check
DEFAULT_RETIRE_STREAK = 5          # consecutive cycles below threshold = retire
DEFAULT_DECAY_WINDOW = 20          # rolling window for IC observations


@dataclass
class ICObservation:
    """Single IC measurement for a signal."""
    signal_id: str
    timestamp: int
    ic: float
    halflife: Optional[float] = None
    status: str = "active"         # active | decaying | retired


@dataclass
class AlphaDecayReport:
    """
    Daily report summarizing IC trend, halflife, and retirement status
    for all tracked signals.
    """
    generated_at: int
    signals: list[dict]            # one entry per signal_id
    retiring_count: int
    decaying_count: int
    healthy_count: int
    total_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class SignalDecayState:
    """
    Per-signal rolling state: IC history, consecutive-below-threshold streak,
    fitted lambda, and status.
    """

    def __init__(self, signal_id: str, window: int, retire_ic: float, retire_streak: int):
        self.signal_id = signal_id
        self.window = window
        self.retire_ic = retire_ic
        self.retire_streak = retire_streak

        # -- circular buffer of (timestamp, ic) pairs
        self.observations: deque[tuple[int, float]] = deque(maxlen=window)
        self.consecutive_below: int = 0
        self.fitted_lambda: Optional[float] = None
        self.status: str = "active"

    def add(self, ts: int, ic: float) -> None:
        """Append new IC observation and update streak counter."""
        self.observations.append((ts, ic))
        if abs(ic) < self.retire_ic:
            self.consecutive_below += 1
        else:
            self.consecutive_below = 0

    def should_retire(self) -> bool:
        return self.consecutive_below >= self.retire_streak

    def compute_lambda(self) -> Optional[float]:
        """
        Fit IC(t) = IC0 * exp(-lambda * t) via log-linear regression.
        Returns lambda if enough non-zero observations exist, else None.
        """
        obs = list(self.observations)
        if len(obs) < MIN_OBS_FOR_FIT:
            return None

        # -- use absolute IC so log is valid; exclude zeros
        pairs = [(i, abs(ic)) for i, (_, ic) in enumerate(obs) if abs(ic) > 1e-9]
        if len(pairs) < MIN_OBS_FOR_FIT:
            return None

        xs = [p[0] for p in pairs]
        ys = [math.log(p[1]) for p in pairs]

        n = len(xs)
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xx = sum(x * x for x in xs)
        sum_xy = sum(x * y for x, y in zip(xs, ys))

        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-12:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denom
        # -- slope is -lambda in IC(t)=IC0*exp(-lambda*t)
        self.fitted_lambda = -slope
        return self.fitted_lambda

    def halflife(self) -> Optional[float]:
        """t_half = ln(2) / lambda"""
        lam = self.compute_lambda()
        if lam is None or lam <= 0:
            return None
        return math.log(2.0) / lam

    def current_ic(self) -> Optional[float]:
        if not self.observations:
            return None
        return self.observations[-1][1]

    def mean_ic(self) -> Optional[float]:
        if not self.observations:
            return None
        return sum(ic for _, ic in self.observations) / len(self.observations)


class AlphaDecayTracker:
    """
    Monitors the information coefficient (IC) decay of active signals.
    Signals with IC < 0.02 for 5 consecutive cycles are retired.

    Thread-safe. Persists to SQLite when db_path is provided.
    """

    def __init__(
        self,
        decay_window: int = DEFAULT_DECAY_WINDOW,
        retire_ic_threshold: float = DEFAULT_RETIRE_IC,
        retire_streak: int = DEFAULT_RETIRE_STREAK,
        db_path: Optional[str] = None,
    ):
        self.decay_window = decay_window
        self.retire_ic_threshold = retire_ic_threshold
        self.retire_streak = retire_streak
        self.db_path = db_path

        self._lock = threading.RLock()
        self._states: dict[str, SignalDecayState] = {}

        if db_path:
            self._init_db()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create alpha_decay table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alpha_decay (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id  TEXT NOT NULL,
                    timestamp  INTEGER NOT NULL,
                    ic         REAL NOT NULL,
                    halflife   REAL,
                    status     TEXT NOT NULL DEFAULT 'active'
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ad_signal_ts "
                "ON alpha_decay(signal_id, timestamp)"
            )
            conn.commit()

    def _persist(self, obs: ICObservation) -> None:
        """Write a single IC observation to the alpha_decay table."""
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO alpha_decay (signal_id, timestamp, ic, halflife, status) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (obs.signal_id, obs.timestamp, obs.ic, obs.halflife, obs.status),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.warning("alpha_decay persistence failed: %s", exc)

    def _load_db(self) -> None:
        """Replay persisted observations into in-memory state on startup."""
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT signal_id, timestamp, ic FROM alpha_decay ORDER BY timestamp ASC"
                ).fetchall()
            for signal_id, ts, ic in rows:
                self._get_or_create(signal_id).add(ts, ic)
        except sqlite3.Error as exc:
            logger.warning("alpha_decay load failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal state access
    # ------------------------------------------------------------------

    def _get_or_create(self, signal_id: str) -> SignalDecayState:
        if signal_id not in self._states:
            self._states[signal_id] = SignalDecayState(
                signal_id,
                self.decay_window,
                self.retire_ic_threshold,
                self.retire_streak,
            )
        return self._states[signal_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_ic(self, signal_id: str, ic: float, timestamp: int) -> None:
        """Store a new IC observation for signal_id."""
        with self._lock:
            state = self._get_or_create(signal_id)
            state.add(timestamp, ic)

            hl = state.halflife()
            status = "retired" if state.should_retire() else (
                "decaying" if self.is_decaying(signal_id) else "active"
            )
            state.status = status

            obs = ICObservation(
                signal_id=signal_id,
                timestamp=timestamp,
                ic=ic,
                halflife=hl,
                status=status,
            )
            self._persist(obs)

    def compute_decay_rate(self, signal_id: str) -> float:
        """
        Return the fitted exponential decay lambda for signal_id.
        Returns 0.0 if insufficient data.
        """
        with self._lock:
            state = self._states.get(signal_id)
            if state is None:
                return 0.0
            lam = state.compute_lambda()
            return lam if lam is not None else 0.0

    def is_decaying(self, signal_id: str) -> bool:
        """
        Returns True if lambda > DECAY_LAMBDA_THRESHOLD (half-life < 7 bars).
        """
        lam = self.compute_decay_rate(signal_id)
        return lam > DECAY_LAMBDA_THRESHOLD

    def get_retire_candidates(self) -> list[str]:
        """
        Return list of signal_ids that have IC below retire_ic_threshold
        for at least retire_streak consecutive cycles.
        """
        with self._lock:
            return [
                sid for sid, state in self._states.items()
                if state.should_retire() and state.status != "retired"
            ]

    def retire_signal(self, signal_id: str) -> None:
        """Explicitly mark a signal as retired."""
        with self._lock:
            state = self._states.get(signal_id)
            if state:
                state.status = "retired"
                logger.info("signal %s retired by alpha_decay_tracker", signal_id)

    def compute_ic_halflife(self, signal_id: str) -> float:
        """
        Return t_half = ln(2) / lambda for signal_id.
        Returns float('inf') if lambda <= 0 or insufficient data.
        """
        with self._lock:
            state = self._states.get(signal_id)
            if state is None:
                return float("inf")
            hl = state.halflife()
            return hl if hl is not None else float("inf")

    def get_signal_summary(self, signal_id: str) -> dict:
        """Return a dict summary for a single signal."""
        with self._lock:
            state = self._states.get(signal_id)
            if state is None:
                return {}
            return {
                "signal_id": signal_id,
                "current_ic": state.current_ic(),
                "mean_ic": state.mean_ic(),
                "lambda": state.compute_lambda(),
                "halflife": state.halflife(),
                "consecutive_below": state.consecutive_below,
                "status": state.status,
                "n_obs": len(state.observations),
            }

    def build_report(self) -> AlphaDecayReport:
        """Generate a full AlphaDecayReport for all tracked signals."""
        with self._lock:
            now = int(time.time())
            signal_summaries = []
            retiring = 0
            decaying = 0
            healthy = 0

            for sid in list(self._states.keys()):
                summary = self.get_signal_summary(sid)
                signal_summaries.append(summary)
                status = summary.get("status", "active")
                if status == "retired":
                    retiring += 1
                elif status == "decaying":
                    decaying += 1
                else:
                    healthy += 1

            return AlphaDecayReport(
                generated_at=now,
                signals=signal_summaries,
                retiring_count=retiring,
                decaying_count=decaying,
                healthy_count=healthy,
                total_count=len(signal_summaries),
            )

    def all_signal_ids(self) -> list[str]:
        with self._lock:
            return list(self._states.keys())

    def active_signals(self) -> list[str]:
        """Return all signal_ids with status != retired."""
        with self._lock:
            return [sid for sid, s in self._states.items() if s.status != "retired"]

    def purge_retired(self) -> int:
        """Remove retired signals from in-memory state. Returns count removed."""
        with self._lock:
            to_remove = [sid for sid, s in self._states.items() if s.status == "retired"]
            for sid in to_remove:
                del self._states[sid]
            return len(to_remove)

    def reset(self) -> None:
        """Clear all in-memory state. Does not touch the database."""
        with self._lock:
            self._states.clear()


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def make_tracker(db_path: Optional[str] = None) -> AlphaDecayTracker:
    """Return a default AlphaDecayTracker, optionally backed by SQLite."""
    return AlphaDecayTracker(
        decay_window=DEFAULT_DECAY_WINDOW,
        retire_ic_threshold=DEFAULT_RETIRE_IC,
        retire_streak=DEFAULT_RETIRE_STREAK,
        db_path=db_path,
    )
