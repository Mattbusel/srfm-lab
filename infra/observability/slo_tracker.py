"""
slo_tracker.py -- Service Level Objective (SLO) tracking for SRFM.

Provides:
    SLO               # definition: name, target, window, metric callable
    SLOStatus         # snapshot of current SLO health
    SLOReport         # full period report across all registered SLOs
    SLOTracker        # register, evaluate, and report on SLOs
    AlertBudgetBurnRate -- fires when error budget will exhaust within 2 days
    SRFM_SLOS         # pre-defined SLO definitions for the trading system

Error budget model
------------------
    error_budget_minutes_total = window_days * 24 * 60 * (1 - target_pct)
    error_budget_minutes_used  = window_days * 24 * 60 * max(0, target_pct - current_pct)
    burn_rate                  = (budget consumed in window) / (budget for window)
"""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SLO:
    """Definition of a single Service Level Objective.

    Parameters
    ----------
    name        # unique identifier used as dict key throughout the tracker
    description -- human-readable description shown in reports
    target_pct  # target success rate, e.g. 0.999 means 99.9 %
    window_days -- rolling calendar window over which the SLO is evaluated
    metric_fn   # zero-argument callable that returns the current success rate
                   as a float in [0.0, 1.0].  Returns None when no data is
                   available (the SLO is treated as met in that case).
    """

    name: str
    description: str
    target_pct: float
    window_days: int
    metric_fn: Optional[Callable[[], Optional[float]]] = None

    @property
    def error_budget_minutes_total(self) -> float:
        """Total allowed downtime minutes in the rolling window."""
        return self.window_days * 24 * 60 * (1.0 - self.target_pct)

    def error_budget_minutes_for_rate(self, current_pct: float) -> float:
        """Minutes of budget consumed given an observed success rate."""
        shortfall = max(0.0, self.target_pct - current_pct)
        return self.window_days * 24 * 60 * shortfall


@dataclass
class SLOStatus:
    """Point-in-time evaluation of a single SLO."""

    name: str
    current_pct: float
    target_pct: float
    is_met: bool

    error_budget_minutes_total: float
    error_budget_minutes_used: float
    error_budget_remaining_pct: float

    burn_rate_1h: float
    burn_rate_6h: float

    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    @property
    def error_budget_minutes_remaining(self) -> float:
        return max(
            0.0,
            self.error_budget_minutes_total - self.error_budget_minutes_used,
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "current_pct": round(self.current_pct, 6),
            "target_pct": self.target_pct,
            "is_met": self.is_met,
            "error_budget_minutes_total": round(self.error_budget_minutes_total, 2),
            "error_budget_minutes_used": round(self.error_budget_minutes_used, 2),
            "error_budget_minutes_remaining": round(
                self.error_budget_minutes_remaining, 2
            ),
            "error_budget_remaining_pct": round(self.error_budget_remaining_pct, 4),
            "burn_rate_1h": round(self.burn_rate_1h, 4),
            "burn_rate_6h": round(self.burn_rate_6h, 4),
            "evaluated_at": self.evaluated_at.isoformat(),
        }


@dataclass
class SLOReport:
    """Aggregated SLO report for a calendar period."""

    period_days: int
    generated_at: datetime
    statuses: Dict[str, SLOStatus]
    slos_met: int
    slos_total: int
    worst_burn_rate_slo: Optional[str]
    worst_burn_rate_value: float

    @property
    def overall_health_pct(self) -> float:
        if self.slos_total == 0:
            return 100.0
        return 100.0 * self.slos_met / self.slos_total

    def to_dict(self) -> Dict:
        return {
            "period_days": self.period_days,
            "generated_at": self.generated_at.isoformat(),
            "overall_health_pct": round(self.overall_health_pct, 2),
            "slos_met": self.slos_met,
            "slos_total": self.slos_total,
            "worst_burn_rate_slo": self.worst_burn_rate_slo,
            "worst_burn_rate_value": round(self.worst_burn_rate_value, 4),
            "statuses": {k: v.to_dict() for k, v in self.statuses.items()},
        }


# ---------------------------------------------------------------------------
# History store -- SQLite ring buffer for SLO measurements
# ---------------------------------------------------------------------------

class _SLOHistoryStore:
    """Persists historical SLO success-rate samples for burn-rate computation.

    Schema
    ------
    slo_history(slo_name TEXT, recorded_at INTEGER, success_rate REAL)

    recorded_at is Unix epoch seconds.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS slo_history (
                    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
                    slo_name    TEXT    NOT NULL,
                    recorded_at INTEGER NOT NULL,
                    success_rate REAL   NOT NULL
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sh_name_time "
                "ON slo_history(slo_name, recorded_at)"
            )
            self._conn.commit()

    def record(self, slo_name: str, success_rate: float) -> None:
        now = int(time.time())
        with self._lock:
            self._conn.execute(
                "INSERT INTO slo_history(slo_name, recorded_at, success_rate) "
                "VALUES (?,?,?)",
                (slo_name, now, success_rate),
            )
            self._conn.commit()

    def samples_since(
        self,
        slo_name: str,
        since_epoch_s: int,
    ) -> List[Tuple[int, float]]:
        """Return (recorded_at, success_rate) tuples newer than since_epoch_s."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT recorded_at, success_rate FROM slo_history "
                "WHERE slo_name = ? AND recorded_at >= ? ORDER BY recorded_at",
                (slo_name, since_epoch_s),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def prune_older_than(self, cutoff_epoch_s: int) -> None:
        """Remove samples older than cutoff_epoch_s."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM slo_history WHERE recorded_at < ?",
                (cutoff_epoch_s,),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# SLOTracker
# ---------------------------------------------------------------------------

class SLOTracker:
    """Register SLOs, evaluate them, and compute error budgets and burn rates.

    Usage
    -----
        tracker = SLOTracker(db_path="/var/data/srfm_slo.db")
        tracker.register(SLO("order_fill_rate", "...", 0.995, 30, my_fn))
        statuses = tracker.check_all()
        remaining = tracker.error_budget_remaining("order_fill_rate")
        report = tracker.generate_report(period_days=30)
    """

    BURN_ALERT_THRESHOLD = 14.0  # x times normal consumption rate

    def __init__(self, db_path: str = ":memory:") -> None:
        self._slos: Dict[str, SLO] = {}
        self._store = _SLOHistoryStore(db_path=db_path)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, slo: SLO) -> None:
        """Register an SLO definition."""
        with self._lock:
            self._slos[slo.name] = slo

    def unregister(self, name: str) -> None:
        """Remove an SLO by name."""
        with self._lock:
            self._slos.pop(name, None)

    def registered_names(self) -> List[str]:
        with self._lock:
            return list(self._slos.keys())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _current_rate(self, slo: SLO) -> Optional[float]:
        """Invoke the SLO metric function safely."""
        if slo.metric_fn is None:
            return None
        try:
            return slo.metric_fn()
        except Exception:
            return None

    def _burn_rate(self, slo: SLO, window_hours: int) -> float:
        """Compute the error budget burn rate over the given window.

        burn_rate = (budget consumed in window) / (expected budget consumption
                    for that window at exactly target_pct success rate).

        A burn_rate of 1.0 means we are consuming the budget at exactly the
        expected rate.  14.0 means the budget will exhaust ~2 calendar days
        ahead of the SLO window end.
        """
        now = int(time.time())
        since = now - window_hours * 3600
        samples = self._store.samples_since(slo.name, since)
        if not samples:
            return 0.0

        avg_rate = sum(r for _, r in samples) / len(samples)
        # error fraction consumed per minute in the observed window
        observed_miss_pct = max(0.0, 1.0 - avg_rate)
        allowed_miss_pct = 1.0 - slo.target_pct

        if allowed_miss_pct <= 0:
            # perfect SLO -- any miss is infinite burn rate, cap at large number
            return 999.9 if observed_miss_pct > 0 else 0.0

        # Normalise: how many times faster than "expected" are we burning?
        # Expected consumption rate assumes uniform distribution across window.
        return observed_miss_pct / allowed_miss_pct

    def check(self, name: str) -> Optional[SLOStatus]:
        """Evaluate a single SLO and return its current status."""
        with self._lock:
            slo = self._slos.get(name)
        if slo is None:
            return None

        current_pct = self._current_rate(slo)
        if current_pct is None:
            # No data available -- treat as met, burn rate 0
            current_pct = slo.target_pct

        # Record sample for burn-rate history
        self._store.record(slo.name, current_pct)

        budget_total = slo.error_budget_minutes_total
        budget_used = slo.error_budget_minutes_for_rate(current_pct)
        budget_remaining_pct = (
            1.0 - (budget_used / budget_total)
            if budget_total > 0
            else 1.0
        )
        budget_remaining_pct = max(0.0, min(1.0, budget_remaining_pct))

        br_1h = self._burn_rate(slo, window_hours=1)
        br_6h = self._burn_rate(slo, window_hours=6)

        return SLOStatus(
            name=slo.name,
            current_pct=current_pct,
            target_pct=slo.target_pct,
            is_met=current_pct >= slo.target_pct,
            error_budget_minutes_total=budget_total,
            error_budget_minutes_used=budget_used,
            error_budget_remaining_pct=budget_remaining_pct,
            burn_rate_1h=br_1h,
            burn_rate_6h=br_6h,
        )

    def check_all(self) -> Dict[str, SLOStatus]:
        """Evaluate every registered SLO and return a name -> status mapping."""
        with self._lock:
            names = list(self._slos.keys())
        results: Dict[str, SLOStatus] = {}
        for name in names:
            status = self.check(name)
            if status is not None:
                results[name] = status
        return results

    # ------------------------------------------------------------------
    # Error budget helpers
    # ------------------------------------------------------------------

    def error_budget_remaining(self, slo_name: str) -> float:
        """Return minutes of error budget remaining for the named SLO.

        Returns the full budget when no data is available.
        """
        status = self.check(slo_name)
        if status is None:
            with self._lock:
                slo = self._slos.get(slo_name)
            if slo is None:
                return 0.0
            return slo.error_budget_minutes_total
        return status.error_budget_minutes_remaining

    def burn_rate(self, slo_name: str, window_hours: int = 1) -> float:
        """Return the current error budget burn rate for the named SLO."""
        with self._lock:
            slo = self._slos.get(slo_name)
        if slo is None:
            return 0.0
        return self._burn_rate(slo, window_hours=window_hours)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, period_days: int = 30) -> SLOReport:
        """Generate a full SLO report for the given period.

        Prunes history older than period_days to keep the store bounded.
        """
        cutoff = int(time.time()) - period_days * 86400
        self._store.prune_older_than(cutoff)

        statuses = self.check_all()
        met_count = sum(1 for s in statuses.values() if s.is_met)

        worst_name: Optional[str] = None
        worst_rate: float = 0.0
        for name, status in statuses.items():
            if status.burn_rate_1h > worst_rate:
                worst_rate = status.burn_rate_1h
                worst_name = name

        return SLOReport(
            period_days=period_days,
            generated_at=datetime.now(tz=timezone.utc),
            statuses=statuses,
            slos_met=met_count,
            slos_total=len(statuses),
            worst_burn_rate_slo=worst_name,
            worst_burn_rate_value=worst_rate,
        )


# ---------------------------------------------------------------------------
# AlertBudgetBurnRate
# ---------------------------------------------------------------------------

class AlertBudgetBurnRate:
    """Fires a callback when any SLO's burn rate exceeds a threshold.

    The default threshold of 14x means the error budget will be exhausted
    within approximately 2 calendar days (window_days / 14 ~ 2 days for a
    30-day window).

    Usage
    -----
        def handle_alert(slo_name, burn_rate, status):
            slack.send(f"SLO {slo_name} burn rate {burn_rate:.1f}x !")

        alert = AlertBudgetBurnRate(tracker, callback=handle_alert)
        alert.start(check_interval_s=60)
        ...
        alert.stop()
    """

    DEFAULT_THRESHOLD = 14.0

    def __init__(
        self,
        tracker: SLOTracker,
        callback: Callable[[str, float, SLOStatus], None],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._tracker = tracker
        self._callback = callback
        self._threshold = threshold
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Track which SLOs are already in alert state to avoid repeat firings
        self._alerting: set = set()

    def start(self, check_interval_s: float = 60.0) -> None:
        """Start background monitoring thread."""
        self._thread = threading.Thread(
            target=self._run,
            args=(check_interval_s,),
            name="srfm-slo-burn-alert",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 10.0) -> None:
        """Stop the monitoring thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def _run(self, interval_s: float) -> None:
        while not self._stop_event.wait(interval_s):
            self._check()

    def _check(self) -> None:
        statuses = self._tracker.check_all()
        for name, status in statuses.items():
            if status.burn_rate_1h >= self._threshold:
                if name not in self._alerting:
                    self._alerting.add(name)
                    try:
                        self._callback(name, status.burn_rate_1h, status)
                    except Exception:
                        pass
            else:
                # Burn rate recovered -- allow re-alerting if it spikes again
                self._alerting.discard(name)

    def check_once(self) -> List[Tuple[str, float, SLOStatus]]:
        """Run a single check synchronously.  Returns list of (name, rate, status)
        for all SLOs currently above the threshold."""
        statuses = self._tracker.check_all()
        return [
            (name, status.burn_rate_1h, status)
            for name, status in statuses.items()
            if status.burn_rate_1h >= self._threshold
        ]


# ---------------------------------------------------------------------------
# Pre-defined SRFM SLOs
# ---------------------------------------------------------------------------

def make_srfm_slos(
    fill_rate_fn: Optional[Callable[[], Optional[float]]] = None,
    signal_freshness_fn: Optional[Callable[[], Optional[float]]] = None,
    risk_latency_fn: Optional[Callable[[], Optional[float]]] = None,
    reconciliation_fn: Optional[Callable[[], Optional[float]]] = None,
    coordination_fn: Optional[Callable[[], Optional[float]]] = None,
) -> List[SLO]:
    """Return the canonical list of SRFM SLO definitions.

    Pass callable metric functions to wire them up; leave as None during
    testing or when the backing metrics are not yet available.

    SLO Definitions
    ---------------
    order_fill_rate          # 99.5 % of submitted orders fill successfully (30d)
    signal_freshness         # 99.9 % of signals computed within 60 s of bar close (7d)
    risk_check_latency       # 99.9 % of pre-trade risk checks complete under 10 ms (7d)
    position_reconciliation  # 99.0 % of positions reconcile with broker within 15 min (30d)
    coordination_availability -- 99.99 % availability of the coordination layer (30d)
    """
    return [
        SLO(
            name="order_fill_rate",
            description="Orders filled successfully",
            target_pct=0.995,
            window_days=30,
            metric_fn=fill_rate_fn,
        ),
        SLO(
            name="signal_freshness",
            description="Signal computed within 60s of bar close",
            target_pct=0.999,
            window_days=7,
            metric_fn=signal_freshness_fn,
        ),
        SLO(
            name="risk_check_latency",
            description="Pre-trade risk check under 10ms",
            target_pct=0.999,
            window_days=7,
            metric_fn=risk_latency_fn,
        ),
        SLO(
            name="position_reconciliation",
            description="Positions reconcile with broker within 15min",
            target_pct=0.99,
            window_days=30,
            metric_fn=reconciliation_fn,
        ),
        SLO(
            name="coordination_availability",
            description="Coordination layer reachable",
            target_pct=0.9999,
            window_days=30,
            metric_fn=coordination_fn,
        ),
    ]


def build_default_tracker(db_path: str = ":memory:") -> SLOTracker:
    """Create an SLOTracker pre-loaded with all SRFM SLO definitions.

    Metric functions are left as None -- inject them at runtime via
    tracker._slos[name].metric_fn = my_callable.
    """
    tracker = SLOTracker(db_path=db_path)
    for slo in make_srfm_slos():
        tracker.register(slo)
    return tracker
