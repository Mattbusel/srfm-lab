"""
promoter.py
-----------
Automatic promotion logic for A/B test winners.

Promotion pipeline
------------------
1. Challenger beats control at p < 0.01 AND min sample → begin gradual rollout.
2. Gradual rollout: 10 % → 30 % → 50 % → 100 % over 4 weeks (7-day steps).
3. Rollback: if champion performance in first 7 days is > 20 % below baseline, revert.
4. All promotion events written to strategy_lab/events.log.

PromotionEvent
--------------
Logged whenever: new champion declared, rollout step, rollback triggered.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Generator

from ..versioning.strategy_version import StrategyVersion, VersionStatus
from ..versioning.version_store import VersionStore
from .ab_test import ABTest
from .significance_tester import SignificanceTester, SignificanceResult

_DEFAULT_DB    = Path(__file__).parent.parent / "strategy_lab.db"
_EVENTS_LOG    = Path(__file__).parent.parent / "events.log"
_ROLLOUT_STEPS = [0.10, 0.30, 0.50, 1.00]
_STEP_DAYS     = 7
_PROMOTE_ALPHA = 0.01
_ROLLBACK_THRESHOLD = 0.20  # 20 % degradation triggers rollback
_ROLLBACK_WINDOW_DAYS = 7


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class PromotionEventType(str, Enum):
    CANDIDATE    = "CANDIDATE"    # challenger qualified, starting rollout
    ROLLOUT_STEP = "ROLLOUT_STEP" # allocation increased
    CHAMPION     = "CHAMPION"     # reached 100 %, now full champion
    ROLLBACK     = "ROLLBACK"     # performance degraded, reverting
    DEMOTED      = "DEMOTED"      # previous champion archived


@dataclass
class PromotionEvent:
    event_type: str
    test_id: str
    version_id: str
    allocation: float
    timestamp: str
    message: str
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "PromotionEvent":
        return cls(**d)

    def __str__(self) -> str:
        ts = self.timestamp[:19]
        return f"[{ts}] {self.event_type} | {self.version_id[:8]} | alloc={self.allocation:.0%} | {self.message}"


# ---------------------------------------------------------------------------
# Promoter
# ---------------------------------------------------------------------------

class Promoter:
    """
    Manages the full promotion lifecycle from A/B result to production champion.

    Parameters
    ----------
    version_store : VersionStore for reading/updating strategy versions
    db_path       : SQLite path (reuses strategy_lab.db)
    events_log    : path for the human-readable events.log
    """

    def __init__(
        self,
        version_store: VersionStore,
        db_path: str | Path = _DEFAULT_DB,
        events_log: str | Path = _EVENTS_LOG,
        promote_alpha: float = _PROMOTE_ALPHA,
    ) -> None:
        self.version_store  = version_store
        self.db_path        = Path(db_path)
        self.events_log     = Path(events_log)
        self.promote_alpha  = promote_alpha
        self._sig_tester    = SignificanceTester(alpha=promote_alpha, min_trades=50)
        self._init_db()
        self._init_log()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, test: ABTest) -> PromotionEvent | None:
        """
        Evaluate a completed/stopped A/B test.
        If the challenger wins, begin gradual rollout.
        Returns the PromotionEvent if promotion was triggered, else None.
        """
        result = self._sig_tester.test(
            test.daily_pnl_a, test.daily_pnl_b,
            trades_a=len(test.trades_a), trades_b=len(test.trades_b),
        )

        if result.winner != "B":
            return None  # challenger did not win

        # Start rollout
        event = PromotionEvent(
            event_type=PromotionEventType.CANDIDATE,
            test_id=test.test_id,
            version_id=test.version_b_id,
            allocation=_ROLLOUT_STEPS[0],
            timestamp=_now_iso(),
            message=(
                f"Challenger {test.version_b_id[:8]} passed significance test "
                f"(p={result.min_p:.4f}, d={result.cohens_d:.3f}). "
                f"Starting gradual rollout at {_ROLLOUT_STEPS[0]:.0%}."
            ),
            metadata={"significance": result.__dict__},
        )
        self._save_rollout_state(test.test_id, test.version_b_id, step_index=0)
        self._log_event(event)
        return event

    def advance_rollout(self, test_id: str) -> PromotionEvent | None:
        """
        Advance the rollout for a test by one step (call this every _STEP_DAYS days).
        If we reach 100 % allocation, the challenger becomes CHAMPION.
        Returns the event, or None if not yet time or already complete.
        """
        state = self._load_rollout_state(test_id)
        if state is None:
            return None

        step_index  = state["step_index"]
        version_id  = state["version_id"]
        started_at  = datetime.fromisoformat(state["started_at"])

        if step_index >= len(_ROLLOUT_STEPS) - 1:
            return None  # already at 100 %

        elapsed = (datetime.now(timezone.utc) - started_at).days
        if elapsed < _STEP_DAYS * (step_index + 1):
            return None  # not yet time

        new_step = step_index + 1
        new_alloc = _ROLLOUT_STEPS[new_step]
        self._save_rollout_state(test_id, version_id, new_step)

        if new_alloc >= 1.0:
            return self._crown_champion(test_id, version_id)

        event = PromotionEvent(
            event_type=PromotionEventType.ROLLOUT_STEP,
            test_id=test_id,
            version_id=version_id,
            allocation=new_alloc,
            timestamp=_now_iso(),
            message=f"Rollout step {new_step}/{len(_ROLLOUT_STEPS)-1}: allocation -> {new_alloc:.0%}",
            metadata={"step_index": new_step},
        )
        self._log_event(event)
        return event

    def check_rollback(
        self,
        version_id: str,
        recent_sharpe: float,
        baseline_sharpe: float,
    ) -> PromotionEvent | None:
        """
        Compare new champion's recent Sharpe vs baseline.
        If degradation > _ROLLBACK_THRESHOLD, revert to previous champion.
        """
        if baseline_sharpe <= 0:
            return None

        degradation = (baseline_sharpe - recent_sharpe) / abs(baseline_sharpe)
        if degradation < _ROLLBACK_THRESHOLD:
            return None  # still healthy

        # Roll back: set current champion to ACTIVE, restore previous
        current = self.version_store.load(version_id)
        if current:
            self.version_store.update_status(version_id, VersionStatus.ACTIVE)

        # Find previous champion
        prev = self._find_previous_champion(version_id)
        if prev:
            self.version_store.update_status(prev.version_id, VersionStatus.CHAMPION)

        event = PromotionEvent(
            event_type=PromotionEventType.ROLLBACK,
            test_id="",
            version_id=version_id,
            allocation=0.0,
            timestamp=_now_iso(),
            message=(
                f"Rollback triggered: Sharpe degraded {degradation:.1%} "
                f"(baseline={baseline_sharpe:.3f}, recent={recent_sharpe:.3f}). "
                f"Reverted to {prev.version_id[:8] if prev else 'no previous champion'}."
            ),
            metadata={
                "degradation": degradation,
                "baseline_sharpe": baseline_sharpe,
                "recent_sharpe": recent_sharpe,
            },
        )
        self._log_event(event)
        return event

    def all_events(self) -> list[PromotionEvent]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT event_json FROM promotion_events ORDER BY rowid"
            ).fetchall()
        return [PromotionEvent.from_dict(json.loads(r[0])) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _crown_champion(self, test_id: str, version_id: str) -> PromotionEvent:
        """Mark version as CHAMPION, archive the previous one."""
        old_champion = self.version_store.champion()
        if old_champion and old_champion.version_id != version_id:
            self.version_store.update_status(old_champion.version_id, VersionStatus.ARCHIVED)
            demote_event = PromotionEvent(
                event_type=PromotionEventType.DEMOTED,
                test_id=test_id,
                version_id=old_champion.version_id,
                allocation=0.0,
                timestamp=_now_iso(),
                message=f"Previous champion {old_champion.version_id[:8]} archived.",
                metadata={},
            )
            self._log_event(demote_event)

        self.version_store.update_status(version_id, VersionStatus.CHAMPION)

        event = PromotionEvent(
            event_type=PromotionEventType.CHAMPION,
            test_id=test_id,
            version_id=version_id,
            allocation=1.0,
            timestamp=_now_iso(),
            message=f"Version {version_id[:8]} is now CHAMPION at 100% allocation.",
            metadata={},
        )
        self._log_event(event)
        return event

    def _find_previous_champion(self, current_id: str) -> StrategyVersion | None:
        """Find the most recently archived version that was previously champion."""
        archived = self.version_store.by_status(VersionStatus.ARCHIVED)
        # The most recent archived version (by created_at) that isn't current_id
        for v in reversed(archived):
            if v.version_id != current_id:
                return v
        return None

    # ------------------------------------------------------------------
    # Rollout state persistence
    # ------------------------------------------------------------------

    def _save_rollout_state(
        self, test_id: str, version_id: str, step_index: int
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO rollout_state
                   (test_id, version_id, step_index, started_at)
                   VALUES (?,?,?,?)""",
                (test_id, version_id, step_index, _now_iso()),
            )

    def _load_rollout_state(self, test_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM rollout_state WHERE test_id = ?", (test_id,)
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def _log_event(self, event: PromotionEvent) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO promotion_events (event_json) VALUES (?)",
                (event.to_json(),),
            )
        with open(self.events_log, "a", encoding="utf-8") as f:
            f.write(str(event) + "\n")

    # ------------------------------------------------------------------
    # DB init
    # ------------------------------------------------------------------

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
                CREATE TABLE IF NOT EXISTS rollout_state (
                    test_id    TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS promotion_events (
                    rowid      INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_json TEXT NOT NULL
                );
            """)

    def _init_log(self) -> None:
        self.events_log.parent.mkdir(parents=True, exist_ok=True)
        if not self.events_log.exists():
            self.events_log.touch()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
