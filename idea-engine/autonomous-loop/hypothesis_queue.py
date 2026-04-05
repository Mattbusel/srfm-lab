"""
idea-engine/autonomous-loop/hypothesis_queue.py

HypothesisQueue: priority queue for hypotheses awaiting processing.

Manages the full hypothesis lifecycle:
  MINED -> DEBATE -> BACKTEST -> VALIDATED -> APPLIED -> ARCHIVED

Enforces concurrency limits and timeouts. Persists all state to SQLite.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_IN_DEBATE = 3
_MAX_IN_BACKTEST = 1
_STAGE_TIMEOUT_HOURS = 24
_DEFAULT_DB = Path(__file__).parent / "loop_state.db"


class HypothesisStage(str, Enum):
    MINED = "mined"
    DEBATE = "debate"
    BACKTEST = "backtest"
    VALIDATED = "validated"
    APPLIED = "applied"
    ARCHIVED = "archived"


class HypothesisQueue:
    """
    Persistent priority queue with stage management and concurrency limits.

    Priority = confidence * effect_size * novelty (penalise recently tried).
    Concurrency: max 3 in DEBATE, max 1 in BACKTEST.
    Stale hypotheses (stuck > 24h in any stage) are auto-archived.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # DB initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hypothesis_queue (
                    hypothesis_id   TEXT PRIMARY KEY,
                    stage           TEXT NOT NULL DEFAULT 'mined',
                    priority_score  REAL NOT NULL DEFAULT 0.0,
                    pattern_type    TEXT,
                    instruments     TEXT,
                    p_value         REAL,
                    effect_size     REAL,
                    novelty_score   REAL DEFAULT 0.5,
                    confidence      REAL DEFAULT 0.5,
                    description     TEXT DEFAULT '',
                    parameters      TEXT DEFAULT '{}',
                    debate_outcome  TEXT,
                    backtest_result TEXT,
                    created_at      TEXT NOT NULL,
                    stage_entered   TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stage ON hypothesis_queue(stage)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_patterns(self, patterns: list) -> list:
        """
        Convert MinedPatterns to hypothesis queue entries via HypothesisGenerator.
        Returns the list of new Hypothesis objects created.
        """
        import sys
        from pathlib import Path as P
        sys.path.insert(0, str(P(__file__).parents[2]))

        new_hypotheses = []
        try:
            from hypothesis.generator import HypothesisGenerator
            generator = HypothesisGenerator()
            new_hypotheses = generator.generate_many(patterns)
        except Exception as exc:
            logger.warning("HypothesisGenerator failed: %s — using stub entries", exc)
            # Fallback: create stub queue entries directly from patterns
            for pat in patterns:
                self._insert_stub(pat)
            return []

        for hyp in new_hypotheses:
            self._upsert_hypothesis(hyp)

        logger.info("HypothesisQueue: ingested %d new hypotheses.", len(new_hypotheses))
        return new_hypotheses

    # ------------------------------------------------------------------
    # Debate stage
    # ------------------------------------------------------------------

    def run_debates(self) -> list:
        """
        Pull hypotheses eligible for debate, respecting concurrency limit.
        Returns list of Hypothesis objects that were approved.
        """
        self._auto_archive_stale()

        # How many slots are available?
        currently_debating = self._count_in_stage(HypothesisStage.DEBATE)
        slots = max(0, _MAX_IN_DEBATE - currently_debating)
        if slots == 0:
            logger.info("HypothesisQueue: debate slots full — skipping.")
            return []

        candidates = self._fetch_by_stage(HypothesisStage.MINED, limit=slots)
        approved = []

        for row in candidates:
            hyp_id = row["hypothesis_id"]
            self._transition(hyp_id, HypothesisStage.DEBATE)
            result = self._debate_one(row)
            if result == "approved":
                self._set_field(hyp_id, "debate_outcome", "approved")
                self._transition(hyp_id, HypothesisStage.BACKTEST)
                approved.append(self._row_to_hypothesis(row))
            elif result == "rejected":
                self._set_field(hyp_id, "debate_outcome", "rejected")
                self._transition(hyp_id, HypothesisStage.ARCHIVED)
            else:
                # needs_more_data: stay in DEBATE until timeout
                self._set_field(hyp_id, "debate_outcome", "needs_more_data")
                logger.info("HypothesisQueue: %s needs more data.", hyp_id[:8])

        return approved

    def _debate_one(self, row: dict) -> str:
        """Run DebateChamber on one hypothesis. Returns 'approved'/'rejected'/'needs_more_data'."""
        try:
            import sys
            from pathlib import Path as P
            sys.path.insert(0, str(P(__file__).parents[2]))
            from debate_system.debate.chamber import DebateChamber
            from hypothesis.types import Hypothesis, HypothesisType, HypothesisStatus
            import json

            hyp = Hypothesis(
                hypothesis_id=row["hypothesis_id"],
                type=HypothesisType.PARAMETER_TWEAK,
                parent_pattern_id=row.get("pattern_type", ""),
                parameters=json.loads(row.get("parameters", "{}")),
                predicted_sharpe_delta=float(row.get("effect_size", 0.0)),
                predicted_dd_delta=0.0,
                novelty_score=float(row.get("novelty_score", 0.5)),
                priority_rank=0,
                status=HypothesisStatus.PENDING,
                created_at=row.get("created_at", ""),
                description=row.get("description", ""),
            )
            chamber = DebateChamber()
            result = chamber.debate(hyp)
            if result.is_approved():
                return "approved"
            elif result.is_rejected():
                return "rejected"
            return "needs_more_data"
        except Exception as exc:
            logger.warning("Debate failed for %s: %s", row["hypothesis_id"][:8], exc)
            return "rejected"

    # ------------------------------------------------------------------
    # Stage transitions
    # ------------------------------------------------------------------

    def mark_validated(self, hypothesis_id: str) -> None:
        self._transition(hypothesis_id, HypothesisStage.VALIDATED)

    def mark_applied(self, hypothesis_id: str) -> None:
        self._transition(hypothesis_id, HypothesisStage.APPLIED)

    def archive(self, hypothesis_id: str) -> None:
        self._transition(hypothesis_id, HypothesisStage.ARCHIVED)

    # ------------------------------------------------------------------
    # Priority scoring
    # ------------------------------------------------------------------

    def _compute_priority(self, row: dict) -> float:
        confidence = float(row.get("confidence", 0.5))
        effect_size = abs(float(row.get("effect_size", 0.0)))
        novelty = float(row.get("novelty_score", 0.5))

        # Penalise recently tried: if updated_at within 12h, reduce novelty
        try:
            updated = datetime.fromisoformat(row.get("updated_at", ""))
            age_h = (datetime.now(timezone.utc) - updated).total_seconds() / 3600
            if age_h < 12:
                novelty *= (age_h / 12)
        except Exception:
            pass

        return confidence * effect_size * novelty

    # ------------------------------------------------------------------
    # Stale hypothesis archival
    # ------------------------------------------------------------------

    def _auto_archive_stale(self) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=_STAGE_TIMEOUT_HOURS)).isoformat()
        active_stages = [
            HypothesisStage.MINED.value,
            HypothesisStage.DEBATE.value,
            HypothesisStage.BACKTEST.value,
        ]
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                f"""
                UPDATE hypothesis_queue
                SET stage = 'archived', updated_at = ?
                WHERE stage IN ({','.join('?' * len(active_stages))})
                  AND stage_entered < ?
                """,
                [datetime.now(timezone.utc).isoformat()] + active_stages + [cutoff],
            )
            if result.rowcount:
                logger.info("HypothesisQueue: archived %d stale hypotheses.", result.rowcount)
            conn.commit()

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _upsert_hypothesis(self, hyp) -> None:
        import json
        now = datetime.now(timezone.utc).isoformat()
        priority = float(hyp.novelty_score) * abs(float(hyp.predicted_sharpe_delta))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO hypothesis_queue
                  (hypothesis_id, stage, priority_score, effect_size, novelty_score,
                   confidence, description, parameters, created_at, stage_entered, updated_at)
                VALUES (?, 'mined', ?, ?, ?, 0.5, ?, ?, ?, ?, ?)
                """,
                (
                    hyp.hypothesis_id,
                    priority,
                    abs(float(hyp.predicted_sharpe_delta)),
                    float(hyp.novelty_score),
                    hyp.description,
                    json.dumps(hyp.parameters),
                    hyp.created_at,
                    now,
                    now,
                ),
            )
            conn.commit()

    def _insert_stub(self, pattern) -> None:
        import json, uuid
        now = datetime.now(timezone.utc).isoformat()
        hyp_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO hypothesis_queue
                  (hypothesis_id, stage, priority_score, pattern_type, instruments,
                   p_value, effect_size, novelty_score, confidence, description,
                   parameters, created_at, stage_entered, updated_at)
                VALUES (?, 'mined', ?, ?, ?, ?, ?, 0.5, 0.5, ?, '{}', ?, ?, ?)
                """,
                (
                    hyp_id,
                    abs(pattern.effect_size),
                    pattern.pattern_type,
                    json.dumps(pattern.instruments),
                    pattern.p_value,
                    pattern.effect_size,
                    f"Auto-mined {pattern.pattern_type} pattern",
                    now, now, now,
                ),
            )
            conn.commit()

    def _count_in_stage(self, stage: HypothesisStage) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM hypothesis_queue WHERE stage = ?",
                (stage.value,),
            ).fetchone()
        return row[0] if row else 0

    def _fetch_by_stage(self, stage: HypothesisStage, limit: int = 10) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM hypothesis_queue WHERE stage = ? ORDER BY priority_score DESC LIMIT ?",
                (stage.value, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def _transition(self, hypothesis_id: str, to_stage: HypothesisStage) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE hypothesis_queue SET stage = ?, stage_entered = ?, updated_at = ? WHERE hypothesis_id = ?",
                (to_stage.value, now, now, hypothesis_id),
            )
            conn.commit()
        logger.debug("HypothesisQueue: %s -> %s", hypothesis_id[:8], to_stage.value)

    def _set_field(self, hypothesis_id: str, field: str, value: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE hypothesis_queue SET {field} = ?, updated_at = ? WHERE hypothesis_id = ?",
                (value, now, hypothesis_id),
            )
            conn.commit()

    def _row_to_hypothesis(self, row: dict):
        """Convert a DB row back to a lightweight stub with required attributes."""
        import json
        from types import SimpleNamespace
        return SimpleNamespace(
            hypothesis_id=row["hypothesis_id"],
            parameters=json.loads(row.get("parameters", "{}")),
            description=row.get("description", ""),
            predicted_sharpe_delta=float(row.get("effect_size", 0.0)),
            novelty_score=float(row.get("novelty_score", 0.5)),
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return current counts per stage."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT stage, COUNT(*) as n FROM hypothesis_queue GROUP BY stage"
            ).fetchall()
        return {r[0]: r[1] for r in rows}
