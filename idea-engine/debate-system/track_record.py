"""
debate-system/track_record.py

AgentTrackRecordManager: tracks agent prediction accuracy over time and
updates credibility scores based on whether their votes were correct.

What counts as "correct"?
-------------------------
A hypothesis is backtested after being promoted to APPROVED.
The backtest result is compared against the hypothesis's prediction:
  - If predicted_sharpe_delta > 0 and backtest improved Sharpe: CONFIRMED
  - If predicted_sharpe_delta > 0 and backtest hurt Sharpe: REFUTED

For each agent's vote on a hypothesis whose backtest is now resolved:
  - Agent voted FOR + hypothesis CONFIRMED → agent was RIGHT → credibility++
  - Agent voted FOR + hypothesis REFUTED   → agent was WRONG → credibility--
  - Agent voted AGAINST + CONFIRMED        → agent was WRONG → credibility--
  - Agent voted AGAINST + REFUTED         → agent was RIGHT → credibility++
  - Agent voted ABSTAIN                   → no update (agent opted out)

Storage
-------
Track records are persisted in debate_agent_track_records table in SQLite.
This table links hypothesis_id → agent_name → vote → was_right → timestamp.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from debate_system.agents.base_agent import Vote
from debate_system.debate.chamber import DebateChamber

logger = logging.getLogger(__name__)

DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")

CREATE_TRACK_RECORD_SQL = """
CREATE TABLE IF NOT EXISTS debate_agent_track_records (
    record_id       TEXT PRIMARY KEY,
    hypothesis_id   TEXT NOT NULL,
    debate_id       TEXT NOT NULL,
    agent_name      TEXT NOT NULL,
    vote            TEXT NOT NULL,
    confidence      REAL NOT NULL,
    was_right       INTEGER,        -- NULL = unresolved, 1 = right, 0 = wrong
    resolved_at     TEXT,
    created_at      TEXT NOT NULL
);
"""

CREATE_TR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_tr_hypothesis
ON debate_agent_track_records (hypothesis_id);
"""


@dataclass
class TrackRecord:
    record_id: str
    hypothesis_id: str
    debate_id: str
    agent_name: str
    vote: str
    confidence: float
    was_right: bool | None = None
    resolved_at: str | None = None
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "hypothesis_id": self.hypothesis_id,
            "debate_id": self.debate_id,
            "agent_name": self.agent_name,
            "vote": self.vote,
            "confidence": self.confidence,
            "was_right": self.was_right,
            "resolved_at": self.resolved_at,
            "created_at": self.created_at,
        }


class AgentTrackRecordManager:
    """
    Manages the prediction track records for all debate agents.

    Usage
    -----
    tracker = AgentTrackRecordManager(chamber)

    # Called after a debate completes:
    tracker.record_debate(debate_result)

    # Called after a hypothesis is backtested:
    tracker.resolve(hypothesis_id, backtest_improved_sharpe=True)
    """

    def __init__(
        self,
        chamber: DebateChamber,
        db_path: Path = DB_PATH,
    ) -> None:
        self.chamber = chamber
        self.db_path = db_path
        self._ensure_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_debate(self, debate_result: Any) -> None:
        """
        Persist each agent's vote from a completed debate.
        Called immediately after DebateChamber.debate() returns.
        """
        from debate_system.debate.chamber import DebateResult
        result: DebateResult = debate_result

        rows: list[tuple] = []
        now = datetime.now(timezone.utc).isoformat()
        for verdict in result.verdicts:
            record_id = f"{result.debate_id}::{verdict.agent_name}"
            rows.append((
                record_id,
                result.hypothesis_id,
                result.debate_id,
                verdict.agent_name,
                verdict.vote.value,
                verdict.confidence,
                None,   # was_right: not yet resolved
                None,   # resolved_at
                now,
            ))

        try:
            with self._connect() as conn:
                conn.executemany(
                    """INSERT OR IGNORE INTO debate_agent_track_records
                       (record_id, hypothesis_id, debate_id, agent_name,
                        vote, confidence, was_right, resolved_at, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                conn.commit()
            logger.debug(
                "Recorded %d agent votes for hypothesis %s.",
                len(rows),
                result.hypothesis_id[:8],
            )
        except sqlite3.Error as exc:
            logger.error("Failed to record debate track records: %s", exc)

    def resolve(
        self,
        hypothesis_id: str,
        backtest_confirmed: bool,
        backtest_sharpe_delta: float | None = None,
    ) -> int:
        """
        Resolve all agent votes for a hypothesis after backtest.

        Parameters
        ----------
        hypothesis_id        : The hypothesis that was backtested.
        backtest_confirmed   : True if the backtest confirmed the hypothesis
                               (improved Sharpe / met its prediction).
        backtest_sharpe_delta: Actual Sharpe delta from backtest (for logging).

        Returns the number of agents whose credibility was updated.
        """
        resolved_at = datetime.now(timezone.utc).isoformat()

        # Load all unresolved votes for this hypothesis
        records = self._load_unresolved(hypothesis_id)
        if not records:
            logger.info(
                "No unresolved track records for hypothesis %s.",
                hypothesis_id[:8],
            )
            return 0

        updated = 0
        for record in records:
            if record.vote == Vote.ABSTAIN.value:
                # Abstainers don't get credited or penalised
                self._mark_resolved(record.record_id, was_right=None, resolved_at=resolved_at)
                continue

            # FOR vote is right iff hypothesis confirmed
            # AGAINST vote is right iff hypothesis refuted
            if record.vote == Vote.FOR.value:
                was_right = backtest_confirmed
            else:  # AGAINST
                was_right = not backtest_confirmed

            # Update in-memory agent credibility
            agent = self.chamber.get_agent(record.agent_name)
            if agent is not None:
                agent.update_credibility(was_right)
                logger.debug(
                    "Updated %s credibility: was_right=%s → new score=%.4f",
                    agent.name,
                    was_right,
                    agent.credibility_score,
                )

            self._mark_resolved(record.record_id, was_right, resolved_at)
            updated += 1

        logger.info(
            "Resolved %d track records for hypothesis %s "
            "(confirmed=%s, sharpe_delta=%s).",
            updated,
            hypothesis_id[:8],
            backtest_confirmed,
            f"{backtest_sharpe_delta:+.4f}" if backtest_sharpe_delta is not None else "N/A",
        )
        return updated

    def agent_accuracy_report(self) -> list[dict[str, Any]]:
        """
        Compute accuracy statistics for each agent from resolved records.
        Returns list sorted by accuracy descending.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT agent_name,
                          COUNT(*) AS total,
                          SUM(CASE WHEN was_right = 1 THEN 1 ELSE 0 END) AS correct,
                          SUM(CASE WHEN was_right = 0 THEN 1 ELSE 0 END) AS wrong
                   FROM debate_agent_track_records
                   WHERE was_right IS NOT NULL
                   GROUP BY agent_name"""
            ).fetchall()

        report = []
        for row in rows:
            total = row["total"]
            correct = row["correct"]
            accuracy = correct / total if total > 0 else 0.0
            agent = self.chamber.get_agent(row["agent_name"])
            report.append({
                "agent": row["agent_name"],
                "total_resolved": total,
                "correct": correct,
                "wrong": row["wrong"],
                "accuracy": round(accuracy, 4),
                "current_credibility": round(agent.credibility_score, 4) if agent else None,
            })

        return sorted(report, key=lambda x: x["accuracy"], reverse=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_TRACK_RECORD_SQL)
            conn.execute(CREATE_TR_INDEX_SQL)
            conn.commit()

    def _load_unresolved(self, hypothesis_id: str) -> list[TrackRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM debate_agent_track_records
                   WHERE hypothesis_id = ? AND was_right IS NULL""",
                (hypothesis_id,),
            ).fetchall()
        return [
            TrackRecord(
                record_id=r["record_id"],
                hypothesis_id=r["hypothesis_id"],
                debate_id=r["debate_id"],
                agent_name=r["agent_name"],
                vote=r["vote"],
                confidence=r["confidence"],
                was_right=r["was_right"],
                resolved_at=r["resolved_at"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def _mark_resolved(
        self,
        record_id: str,
        was_right: bool | None,
        resolved_at: str,
    ) -> None:
        was_right_int = None if was_right is None else (1 if was_right else 0)
        with self._connect() as conn:
            conn.execute(
                """UPDATE debate_agent_track_records
                   SET was_right = ?, resolved_at = ?
                   WHERE record_id = ?""",
                (was_right_int, resolved_at, record_id),
            )
            conn.commit()
