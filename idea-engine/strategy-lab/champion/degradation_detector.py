"""
degradation_detector.py
-----------------------
Detect strategy decay over time using statistical tests.

Methods
-------
1. Rolling 30-day performance vs all-time performance (rolling mean comparison)
2. CUSUM test: cumulative sum control chart for step changes in mean return
3. Chow test: structural break detection on rolling regression windows

If decay is detected, a DegradationSignal is raised and stored. The signal
can be used externally to trigger a new IAE idea generation cycle.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd

_DEFAULT_DB           = Path(__file__).parent.parent / "strategy_lab.db"
_ROLLING_WINDOW       = 30   # days
_CUSUM_THRESHOLD      = 4.0  # standard deviations
_CHOW_CONFIDENCE      = 0.95
_MIN_OBSERVATIONS     = 60   # minimum days of data before running tests


# ---------------------------------------------------------------------------
# DegradationSignal
# ---------------------------------------------------------------------------

@dataclass
class DegradationSignal:
    """
    Emitted when a statistical test detects strategy performance decay.

    Attributes
    ----------
    version_id     : strategy version being monitored
    detected_at    : timestamp
    test_name      : which test fired ("rolling" | "cusum" | "chow")
    severity       : "MILD" | "MODERATE" | "SEVERE"
    rolling_sharpe : most recent 30-day Sharpe
    alltime_sharpe : all-time Sharpe
    cusum_stat     : current CUSUM statistic (None if not applicable)
    chow_p_value   : p-value from Chow test (None if not applicable)
    break_index    : estimated index of structural break (None if not applicable)
    message        : human-readable summary
    """
    version_id: str
    detected_at: str
    test_name: str
    severity: str
    rolling_sharpe: float
    alltime_sharpe: float
    cusum_stat: float | None
    chow_p_value: float | None
    break_index: int | None
    message: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DegradationSignal":
        return cls(**d)

    def __str__(self) -> str:
        return (
            f"[{self.severity}] Degradation via {self.test_name}: "
            f"rolling Sharpe={self.rolling_sharpe:.3f} vs all-time={self.alltime_sharpe:.3f}. "
            f"{self.message}"
        )


# ---------------------------------------------------------------------------
# DegradationDetector
# ---------------------------------------------------------------------------

class DegradationDetector:
    """
    Monitors a strategy's daily P&L series for signs of decay.

    Parameters
    ----------
    db_path          : SQLite path
    rolling_window   : days for rolling stats (default 30)
    cusum_threshold  : CUSUM alarm threshold in std devs (default 4.0)
    min_observations : minimum data points before running tests (default 60)
    """

    def __init__(
        self,
        db_path: str | Path = _DEFAULT_DB,
        rolling_window: int = _ROLLING_WINDOW,
        cusum_threshold: float = _CUSUM_THRESHOLD,
        min_observations: int = _MIN_OBSERVATIONS,
    ) -> None:
        self.db_path          = Path(db_path)
        self.rolling_window   = rolling_window
        self.cusum_threshold  = cusum_threshold
        self.min_observations = min_observations
        self._init_db()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def check(
        self,
        version_id: str,
        daily_pnl: list[float],
    ) -> list[DegradationSignal]:
        """
        Run all degradation tests on the daily P&L series.
        Returns list of DegradationSignals (empty = no issues detected).
        """
        if len(daily_pnl) < self.min_observations:
            return []

        arr = np.array(daily_pnl, dtype=float)
        signals: list[DegradationSignal] = []

        # Test 1: Rolling vs all-time
        s = self._test_rolling(version_id, arr)
        if s:
            signals.append(s)

        # Test 2: CUSUM
        s = self._test_cusum(version_id, arr)
        if s:
            signals.append(s)

        # Test 3: Chow structural break
        s = self._test_chow(version_id, arr)
        if s:
            signals.append(s)

        for sig in signals:
            self._persist_signal(sig)

        return signals

    # ------------------------------------------------------------------
    # Test 1: Rolling window vs all-time
    # ------------------------------------------------------------------

    def _test_rolling(
        self, version_id: str, arr: np.ndarray
    ) -> DegradationSignal | None:
        alltime_sharpe = _sharpe(arr)
        rolling = arr[-self.rolling_window:]
        rolling_sharpe = _sharpe(rolling)

        if alltime_sharpe <= 0:
            return None

        degradation = (alltime_sharpe - rolling_sharpe) / abs(alltime_sharpe)

        if degradation < 0.20:
            return None  # less than 20% degradation — OK

        severity = "SEVERE" if degradation > 0.60 else "MODERATE" if degradation > 0.40 else "MILD"

        return DegradationSignal(
            version_id=version_id,
            detected_at=_now_iso(),
            test_name="rolling",
            severity=severity,
            rolling_sharpe=rolling_sharpe,
            alltime_sharpe=alltime_sharpe,
            cusum_stat=None,
            chow_p_value=None,
            break_index=None,
            message=(
                f"Rolling {self.rolling_window}d Sharpe ({rolling_sharpe:.3f}) is "
                f"{degradation:.1%} below all-time Sharpe ({alltime_sharpe:.3f})."
            ),
        )

    # ------------------------------------------------------------------
    # Test 2: CUSUM
    # ------------------------------------------------------------------

    def _test_cusum(
        self, version_id: str, arr: np.ndarray
    ) -> DegradationSignal | None:
        """
        Two-sided CUSUM test for a step change in mean return.
        Uses the first half of the series as the reference period.
        """
        n = len(arr)
        half = n // 2
        reference = arr[:half]
        mu0 = float(np.mean(reference))
        sigma = float(np.std(reference, ddof=1))
        if sigma < 1e-12:
            return None

        # Standardize deviations from mu0
        z = (arr - mu0) / sigma

        # Lower CUSUM (detect downward shift)
        cusum_lo = np.zeros(n)
        for i in range(1, n):
            cusum_lo[i] = max(0.0, cusum_lo[i-1] - z[i] - 0.5)

        max_cusum = float(np.max(cusum_lo))
        alarm_index = int(np.argmax(cusum_lo)) if max_cusum >= self.cusum_threshold else None

        if max_cusum < self.cusum_threshold:
            return None

        alltime_sharpe = _sharpe(arr)
        rolling_sharpe = _sharpe(arr[-self.rolling_window:])
        severity = "SEVERE" if max_cusum > self.cusum_threshold * 2 else "MODERATE"

        return DegradationSignal(
            version_id=version_id,
            detected_at=_now_iso(),
            test_name="cusum",
            severity=severity,
            rolling_sharpe=rolling_sharpe,
            alltime_sharpe=alltime_sharpe,
            cusum_stat=max_cusum,
            chow_p_value=None,
            break_index=alarm_index,
            message=(
                f"CUSUM alarm: stat={max_cusum:.2f} exceeds threshold={self.cusum_threshold:.2f}. "
                f"Likely step-down at observation {alarm_index}."
            ),
        )

    # ------------------------------------------------------------------
    # Test 3: Chow structural break
    # ------------------------------------------------------------------

    def _test_chow(
        self, version_id: str, arr: np.ndarray
    ) -> DegradationSignal | None:
        """
        Scan for the most likely structural break point using a Chow-test approach.
        Tests H0: same mean in both sub-series vs H1: different means.
        Uses F-statistic approximation; tests candidate break points in the
        middle 50% of the series.
        """
        n = len(arr)
        lo = n // 4
        hi = 3 * n // 4

        best_f = 0.0
        best_k = lo

        sse_full = float(np.sum((arr - np.mean(arr)) ** 2))

        for k in range(lo, hi):
            seg1 = arr[:k]
            seg2 = arr[k:]
            sse1 = float(np.sum((seg1 - np.mean(seg1)) ** 2))
            sse2 = float(np.sum((seg2 - np.mean(seg2)) ** 2))
            sse_restricted = sse1 + sse2

            # Chow F-stat: (sse_full - sse_restricted) / k / (sse_restricted / (n - 2*k))
            denom = sse_restricted / max(n - 4, 1)
            if denom < 1e-15:
                continue
            f = ((sse_full - sse_restricted) / 2) / denom
            if f > best_f:
                best_f = f
                best_k = k

        # Convert F-stat to p-value approximation (F(2, n-4) distribution)
        p_value = self._f_pvalue(best_f, dfn=2, dfd=max(n - 4, 1))

        if p_value >= (1 - _CHOW_CONFIDENCE):
            return None  # no significant break

        # Confirm the break is a downward shift (mean2 < mean1)
        mean1 = float(np.mean(arr[:best_k]))
        mean2 = float(np.mean(arr[best_k:]))
        if mean2 >= mean1:
            return None  # break is upward — not a degradation

        alltime_sharpe = _sharpe(arr)
        rolling_sharpe = _sharpe(arr[-self.rolling_window:])

        return DegradationSignal(
            version_id=version_id,
            detected_at=_now_iso(),
            test_name="chow",
            severity="MODERATE",
            rolling_sharpe=rolling_sharpe,
            alltime_sharpe=alltime_sharpe,
            cusum_stat=None,
            chow_p_value=p_value,
            break_index=best_k,
            message=(
                f"Chow structural break detected at observation {best_k} "
                f"(p={p_value:.4f}). "
                f"Mean pre-break={mean1:.5f}, post-break={mean2:.5f}."
            ),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all_signals(self, version_id: str) -> list[DegradationSignal]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT signal_json FROM degradation_signals WHERE version_id = ? ORDER BY rowid",
                (version_id,),
            ).fetchall()
        return [DegradationSignal.from_dict(json.loads(r[0])) for r in rows]

    def latest_signal(self, version_id: str) -> DegradationSignal | None:
        signals = self.all_signals(version_id)
        return signals[-1] if signals else None

    def is_decaying(self, version_id: str) -> bool:
        """Quick check: has any degradation signal been raised recently?"""
        sig = self.latest_signal(version_id)
        return sig is not None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_signal(self, sig: DegradationSignal) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO degradation_signals (version_id, signal_json) VALUES (?,?)",
                (sig.version_id, json.dumps(sig.to_dict(), default=str)),
            )

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS degradation_signals (
                    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id  TEXT NOT NULL,
                    signal_json TEXT NOT NULL
                );
            """)

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _f_pvalue(f: float, dfn: int, dfd: int) -> float:
        """Approximate p-value for F(dfn, dfd) using beta distribution."""
        if f <= 0:
            return 1.0
        x = dfd / (dfd + dfn * f)
        from ..experiments.significance_tester import SignificanceTester
        ibeta = SignificanceTester._reg_incomplete_beta(x, dfd / 2.0, dfn / 2.0)
        return float(ibeta)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sharpe(arr: np.ndarray | list) -> float:
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    return float(np.mean(arr) / std * math.sqrt(252)) if std > 0 else 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
