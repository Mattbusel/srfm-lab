"""
hypothesis/hypothesis_store.py

CRUD layer for the `hypotheses` table in idea_engine.db.
Also ensures the schema exists on first use.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from hypothesis.types import Hypothesis, HypothesisStatus, HypothesisType

DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_HYPOTHESES = """
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id       TEXT PRIMARY KEY,
    type                TEXT NOT NULL,
    parent_pattern_id   TEXT NOT NULL,
    parameters          TEXT NOT NULL,          -- JSON
    predicted_sharpe_delta  REAL NOT NULL DEFAULT 0.0,
    predicted_dd_delta      REAL NOT NULL DEFAULT 0.0,
    novelty_score       REAL NOT NULL DEFAULT 0.5,
    priority_rank       INTEGER NOT NULL DEFAULT 0,
    status              TEXT NOT NULL DEFAULT 'pending',
    created_at          TEXT NOT NULL,
    description         TEXT NOT NULL DEFAULT '',
    compound_child_ids  TEXT NOT NULL DEFAULT '[]'  -- JSON array
);
"""

_CREATE_IDX_STATUS = """
CREATE INDEX IF NOT EXISTS idx_hypotheses_status
    ON hypotheses (status);
"""

_CREATE_IDX_PARENT = """
CREATE INDEX IF NOT EXISTS idx_hypotheses_parent
    ON hypotheses (parent_pattern_id);
"""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class HypothesisStore:
    """
    Thin CRUD wrapper around the hypotheses table.
    Each method opens and closes its own connection for thread-safety.
    """

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema init
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_HYPOTHESES)
            conn.execute(_CREATE_IDX_STATUS)
            conn.execute(_CREATE_IDX_PARENT)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, h: Hypothesis) -> None:
        """Insert a new hypothesis. Raises sqlite3.IntegrityError on duplicate id."""
        row = h.to_dict()
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        sql = f"INSERT INTO hypotheses ({cols}) VALUES ({placeholders})"
        with self._connect() as conn:
            conn.execute(sql, list(row.values()))
            conn.commit()

    def insert_many(self, hypotheses: list[Hypothesis]) -> int:
        """Bulk insert; skips duplicates. Returns number inserted."""
        inserted = 0
        with self._connect() as conn:
            for h in hypotheses:
                row = h.to_dict()
                cols = ", ".join(row.keys())
                placeholders = ", ".join(["?"] * len(row))
                sql = f"INSERT OR IGNORE INTO hypotheses ({cols}) VALUES ({placeholders})"
                cursor = conn.execute(sql, list(row.values()))
                inserted += cursor.rowcount
            conn.commit()
        return inserted

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, hypothesis_id: str) -> Hypothesis | None:
        sql = "SELECT * FROM hypotheses WHERE hypothesis_id = ?"
        with self._connect() as conn:
            row = conn.execute(sql, (hypothesis_id,)).fetchone()
        if row is None:
            return None
        return Hypothesis.from_dict(dict(row))

    def get_pending(self) -> list[Hypothesis]:
        return self.get_by_status(HypothesisStatus.PENDING)

    def get_by_status(self, status: HypothesisStatus) -> list[Hypothesis]:
        sql = "SELECT * FROM hypotheses WHERE status = ? ORDER BY priority_rank DESC, created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, (status.value,)).fetchall()
        return [Hypothesis.from_dict(dict(r)) for r in rows]

    def get_all(self) -> list[Hypothesis]:
        sql = "SELECT * FROM hypotheses ORDER BY priority_rank DESC, created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [Hypothesis.from_dict(dict(r)) for r in rows]

    def get_by_parent(self, parent_pattern_id: str) -> list[Hypothesis]:
        sql = "SELECT * FROM hypotheses WHERE parent_pattern_id = ?"
        with self._connect() as conn:
            rows = conn.execute(sql, (parent_pattern_id,)).fetchall()
        return [Hypothesis.from_dict(dict(r)) for r in rows]

    def get_by_type(self, hypothesis_type: HypothesisType) -> list[Hypothesis]:
        sql = "SELECT * FROM hypotheses WHERE type = ? ORDER BY priority_rank DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, (hypothesis_type.value,)).fetchall()
        return [Hypothesis.from_dict(dict(r)) for r in rows]

    def count_by_status(self) -> dict[str, int]:
        sql = "SELECT status, COUNT(*) AS cnt FROM hypotheses GROUP BY status"
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_status(self, hypothesis_id: str, new_status: HypothesisStatus) -> bool:
        sql = "UPDATE hypotheses SET status = ? WHERE hypothesis_id = ?"
        with self._connect() as conn:
            cursor = conn.execute(sql, (new_status.value, hypothesis_id))
            conn.commit()
        return cursor.rowcount > 0

    def update_priority_rank(self, hypothesis_id: str, rank: int) -> bool:
        sql = "UPDATE hypotheses SET priority_rank = ? WHERE hypothesis_id = ?"
        with self._connect() as conn:
            cursor = conn.execute(sql, (rank, hypothesis_id))
            conn.commit()
        return cursor.rowcount > 0

    def update_scores(
        self,
        hypothesis_id: str,
        predicted_sharpe_delta: float,
        predicted_dd_delta: float,
        novelty_score: float,
    ) -> bool:
        sql = """
            UPDATE hypotheses
            SET predicted_sharpe_delta = ?,
                predicted_dd_delta = ?,
                novelty_score = ?
            WHERE hypothesis_id = ?
        """
        with self._connect() as conn:
            cursor = conn.execute(
                sql,
                (predicted_sharpe_delta, predicted_dd_delta, novelty_score, hypothesis_id),
            )
            conn.commit()
        return cursor.rowcount > 0

    def bulk_update_ranks(self, ranked: list[tuple[str, int]]) -> int:
        """
        Update priority_rank for many hypotheses at once.
        ranked: list of (hypothesis_id, rank) tuples.
        Returns number of rows updated.
        """
        sql = "UPDATE hypotheses SET priority_rank = ? WHERE hypothesis_id = ?"
        with self._connect() as conn:
            cursor = conn.executemany(sql, [(rank, hid) for hid, rank in ranked])
            conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, hypothesis_id: str) -> bool:
        sql = "DELETE FROM hypotheses WHERE hypothesis_id = ?"
        with self._connect() as conn:
            cursor = conn.execute(sql, (hypothesis_id,))
            conn.commit()
        return cursor.rowcount > 0

    def delete_rejected(self) -> int:
        sql = "DELETE FROM hypotheses WHERE status = 'rejected'"
        with self._connect() as conn:
            cursor = conn.execute(sql)
            conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def exists(self, hypothesis_id: str) -> bool:
        sql = "SELECT 1 FROM hypotheses WHERE hypothesis_id = ? LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(sql, (hypothesis_id,)).fetchone()
        return row is not None

    def get_parameters_snapshot(self) -> list[dict[str, Any]]:
        """
        Returns a lightweight list of {hypothesis_id, type, parameters_dict}
        for use by the deduplicator without loading full Hypothesis objects.
        """
        sql = "SELECT hypothesis_id, type, parameters FROM hypotheses WHERE status != 'rejected'"
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        result = []
        for r in rows:
            params = r["parameters"]
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    params = {}
            result.append({
                "hypothesis_id": r["hypothesis_id"],
                "type": r["type"],
                "parameters": params,
            })
        return result
