"""
performance_baseline.py -- Performance baseline tracking and regression detection
for the SRFM quantitative trading system.

Provides:
    Regression          # dataclass describing a detected performance regression
    PerformanceBaseline -- tracks rolling baselines and detects regressions
    SRFM_METRICS        # canonical list of SRFM performance metric names

Algorithm
---------
    Each metric maintains a rolling 30-day window of samples stored in SQLite.
    Baseline statistics (mean, std, percentiles) are computed on-demand from
    that window.  A regression is flagged when the current value deviates from
    the baseline by more than threshold_pct OR by more than z_score_threshold
    standard deviations.

    For latency metrics (lower is better):  regression when current > baseline.
    For return metrics (higher is better):  regression when current < baseline.
    Metric direction is configured via the METRIC_DIRECTIONS dict.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------

METRIC_DIRECTION_LOWER_BETTER = "lower"
METRIC_DIRECTION_HIGHER_BETTER = "higher"

#: Maps metric name -> whether lower values are better.
METRIC_DIRECTIONS: Dict[str, str] = {
    "order_fill_latency_ms":          METRIC_DIRECTION_LOWER_BETTER,
    "signal_computation_ms":          METRIC_DIRECTION_LOWER_BETTER,
    "risk_check_ms":                  METRIC_DIRECTION_LOWER_BETTER,
    "position_reconciliation_ms":     METRIC_DIRECTION_LOWER_BETTER,
    "sharpe_ratio_4h":                METRIC_DIRECTION_HIGHER_BETTER,
    "win_rate":                       METRIC_DIRECTION_HIGHER_BETTER,
    "max_drawdown":                   METRIC_DIRECTION_LOWER_BETTER,  # magnitude
}

#: Severity labels keyed by |pct_change| relative to threshold.
_SEVERITY_LEVELS = [
    (0.5,  "critical"),   # >50 % beyond threshold
    (0.25, "high"),       # >25 %
    (0.10, "medium"),     # >10 %
    (0.0,  "low"),        # any threshold breach
]


def _severity(pct_change: float, threshold_pct: float) -> str:
    """Classify regression severity from the magnitude of the change."""
    excess = abs(pct_change) - threshold_pct
    ratio = excess / max(threshold_pct, 1e-9)
    for cutoff, label in _SEVERITY_LEVELS:
        if ratio >= cutoff:
            return label
    return "low"


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

@dataclass
class Regression:
    """Describes a detected performance regression.

    Fields
    ------
    metric          # metric name
    baseline_value  # historical median (or specified percentile) used as reference
    current_value   # the value that triggered the regression
    pct_change      # (current - baseline) / baseline, signed
    severity        # "low", "medium", "high", or "critical"
    z_score         # standard deviations from the rolling mean
    direction       # "lower_better" or "higher_better"
    detected_at     # UTC timestamp
    """

    metric: str
    baseline_value: float
    current_value: float
    pct_change: float
    severity: str
    z_score: float = 0.0
    direction: str = METRIC_DIRECTION_LOWER_BETTER
    detected_at: datetime = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.detected_at is None:
            self.detected_at = datetime.now(tz=timezone.utc)

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric,
            "baseline_value": round(self.baseline_value, 6),
            "current_value": round(self.current_value, 6),
            "pct_change": round(self.pct_change, 6),
            "severity": self.severity,
            "z_score": round(self.z_score, 4),
            "direction": self.direction,
            "detected_at": self.detected_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# SQLite sample store
# ---------------------------------------------------------------------------

class _BaselineStore:
    """Stores raw metric samples used for baseline computation.

    Schema: metric_samples(metric TEXT, recorded_date TEXT, value REAL)
    recorded_date is ISO format YYYY-MM-DD to support daily granularity.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        window_days: int = 30,
    ) -> None:
        self._db_path = db_path
        self._window_days = window_days
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_samples (
                    rowid         INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric        TEXT    NOT NULL,
                    recorded_date TEXT    NOT NULL,
                    recorded_ts   INTEGER NOT NULL,
                    value         REAL    NOT NULL
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ms_metric_date "
                "ON metric_samples(metric, recorded_date)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ms_ts "
                "ON metric_samples(recorded_ts)"
            )
            self._conn.commit()

    def insert(self, metric: str, value: float, date: str) -> None:
        """Insert a sample.  date should be 'YYYY-MM-DD'."""
        ts = int(time.time())
        with self._lock:
            self._conn.execute(
                "INSERT INTO metric_samples(metric, recorded_date, recorded_ts, value) "
                "VALUES (?,?,?,?)",
                (metric, date, ts, value),
            )
            self._conn.commit()

    def window_values(self, metric: str) -> List[float]:
        """Return all values within the rolling window, oldest first."""
        cutoff = int(time.time()) - self._window_days * 86400
        with self._lock:
            rows = self._conn.execute(
                "SELECT value FROM metric_samples "
                "WHERE metric = ? AND recorded_ts >= ? ORDER BY recorded_ts",
                (metric, cutoff),
            ).fetchall()
        return [r[0] for r in rows]

    def prune_old(self) -> None:
        """Remove samples outside the rolling window."""
        cutoff = int(time.time()) - self._window_days * 86400
        with self._lock:
            self._conn.execute(
                "DELETE FROM metric_samples WHERE recorded_ts < ?", (cutoff,)
            )
            self._conn.commit()

    def all_metrics(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT metric FROM metric_samples ORDER BY metric"
            ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _std(values: List[float], mean: Optional[float] = None) -> float:
    if len(values) < 2:
        return 0.0
    m = mean if mean is not None else _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Linear interpolation percentile on a sorted list."""
    if not sorted_values:
        return 0.0
    if pct <= 0.0:
        return sorted_values[0]
    if pct >= 1.0:
        return sorted_values[-1]
    idx = pct * (len(sorted_values) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_values):
        return sorted_values[-1]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


# ---------------------------------------------------------------------------
# PerformanceBaseline
# ---------------------------------------------------------------------------

class PerformanceBaseline:
    """Tracks rolling baselines and detects performance regressions.

    Thread-safe.  All heavy computation is lazy -- baselines are recomputed
    from the SQLite store on each call.

    Alert integration
    -----------------
    Set on_regression to a callable that receives a Regression object; it will
    be invoked immediately when detect_regression returns a non-None value
    internally.  The public detect_regression method also returns the object
    so callers can handle it inline.

    Usage
    -----
        pb = PerformanceBaseline(db_path="/var/data/srfm_perf.db")
        pb.update_baseline("order_fill_latency_ms", 4.2, "2026-04-07")
        reg = pb.detect_regression("order_fill_latency_ms", current_value=18.5)
        if reg:
            slack.post(reg.to_dict())
    """

    DEFAULT_WINDOW_DAYS = 30
    DEFAULT_THRESHOLD_PCT = 0.10   # 10 % deviation
    DEFAULT_Z_THRESHOLD = 3.0      # 3 standard deviations

    def __init__(
        self,
        db_path: str = ":memory:",
        window_days: int = DEFAULT_WINDOW_DAYS,
        z_score_threshold: float = DEFAULT_Z_THRESHOLD,
        on_regression: Optional[Callable[[Regression], None]] = None,
    ) -> None:
        self._store = _BaselineStore(db_path=db_path, window_days=window_days)
        self._window_days = window_days
        self._z_threshold = z_score_threshold
        self._on_regression = on_regression

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def update_baseline(
        self,
        metric: str,
        value: float,
        date: str,
    ) -> None:
        """Record a new sample for the given metric.

        Parameters
        ----------
        metric  # metric name (should be one of METRIC_DIRECTIONS keys)
        value   # observed value
        date    # ISO date string 'YYYY-MM-DD' for daily bucketing
        """
        self._store.insert(metric, value, date)

    # ------------------------------------------------------------------
    # Statistical queries
    # ------------------------------------------------------------------

    def baseline_for(
        self,
        metric: str,
        percentile: float = 0.50,
    ) -> float:
        """Return the historical percentile value for a metric.

        Parameters
        ----------
        metric      # metric name
        percentile  # 0.0-1.0; 0.50 = median (default)

        Returns 0.0 when there is insufficient history.
        """
        values = self._store.window_values(metric)
        if not values:
            return 0.0
        return _percentile(sorted(values), percentile)

    def z_score(self, metric: str, current_value: float) -> float:
        """Return the z-score of current_value relative to the rolling window.

        z = (current - mean) / std

        Returns 0.0 when there is insufficient history.
        """
        values = self._store.window_values(metric)
        if len(values) < 2:
            return 0.0
        m = _mean(values)
        s = _std(values, mean=m)
        if s == 0.0:
            return 0.0
        return (current_value - m) / s

    def rolling_mean(self, metric: str) -> float:
        """Return the rolling window mean for a metric."""
        values = self._store.window_values(metric)
        if not values:
            return 0.0
        return _mean(values)

    def rolling_std(self, metric: str) -> float:
        """Return the rolling window standard deviation for a metric."""
        values = self._store.window_values(metric)
        if len(values) < 2:
            return 0.0
        return _std(values)

    def sample_count(self, metric: str) -> int:
        """Return the number of samples in the rolling window."""
        return len(self._store.window_values(metric))

    # ------------------------------------------------------------------
    # Regression detection
    # ------------------------------------------------------------------

    def detect_regression(
        self,
        metric: str,
        current_value: float,
        threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    ) -> Optional[Regression]:
        """Check whether current_value represents a regression from the baseline.

        A regression is flagged when EITHER condition is true:
            1. |pct_change| > threshold_pct  (percentage-based check)
            2. |z_score|    > z_score_threshold

        Direction is respected: for latency metrics, only increases trigger
        regressions; for return metrics, only decreases trigger regressions.

        Returns a Regression dataclass on regression, None otherwise.
        """
        values = self._store.window_values(metric)
        if len(values) < 5:
            # Not enough history to establish a reliable baseline
            return None

        baseline = _percentile(sorted(values), 0.50)
        if baseline == 0.0:
            return None

        pct_change = (current_value - baseline) / abs(baseline)
        z = self.z_score(metric, current_value)

        direction = METRIC_DIRECTIONS.get(metric, METRIC_DIRECTION_LOWER_BETTER)

        # Determine if the direction of the change is bad
        is_worse: bool
        if direction == METRIC_DIRECTION_LOWER_BETTER:
            is_worse = pct_change > threshold_pct or z > self._z_threshold
        else:
            is_worse = pct_change < -threshold_pct or z < -self._z_threshold

        if not is_worse:
            return None

        regression = Regression(
            metric=metric,
            baseline_value=baseline,
            current_value=current_value,
            pct_change=pct_change,
            severity=_severity(pct_change, threshold_pct),
            z_score=z,
            direction=direction,
        )

        if self._on_regression is not None:
            try:
                self._on_regression(regression)
            except Exception:
                pass

        return regression

    def check_all_metrics(
        self,
        current_values: Dict[str, float],
        threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    ) -> List[Regression]:
        """Batch-check a dict of {metric: current_value} for regressions.

        Returns list of Regression objects for all metrics that regressed.
        """
        regressions: List[Regression] = []
        for metric, value in current_values.items():
            reg = self.detect_regression(metric, value, threshold_pct)
            if reg is not None:
                regressions.append(reg)
        return regressions

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> List[Dict]:
        """Return a summary dict for all metrics that have data."""
        self._store.prune_old()
        result = []
        for metric in self._store.all_metrics():
            values = self._store.window_values(metric)
            if not values:
                continue
            sv = sorted(values)
            result.append({
                "metric": metric,
                "sample_count": len(values),
                "mean": round(_mean(values), 4),
                "std": round(_std(values), 4),
                "p50": round(_percentile(sv, 0.50), 4),
                "p90": round(_percentile(sv, 0.90), 4),
                "p99": round(_percentile(sv, 0.99), 4),
                "min": round(sv[0], 4),
                "max": round(sv[-1], 4),
                "direction": METRIC_DIRECTIONS.get(metric, METRIC_DIRECTION_LOWER_BETTER),
            })
        return result

    def close(self) -> None:
        """Close the underlying database connection."""
        self._store.close()


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

_DEFAULT_BASELINE: Optional[PerformanceBaseline] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_baseline(db_path: str = ":memory:") -> PerformanceBaseline:
    """Return the process-global PerformanceBaseline (lazy singleton)."""
    global _DEFAULT_BASELINE
    with _DEFAULT_LOCK:
        if _DEFAULT_BASELINE is None:
            _DEFAULT_BASELINE = PerformanceBaseline(db_path=db_path)
    return _DEFAULT_BASELINE


#: Canonical SRFM metric names.
SRFM_METRICS: List[str] = list(METRIC_DIRECTIONS.keys())
