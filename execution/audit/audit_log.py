"""
execution/audit/audit_log.py
==============================
Immutable, hash-chained audit log stored in SQLite.

Every order event, fill, cancellation, and risk rejection is written as a row
containing:
  - A SHA-256 hash of the row's content.
  - The hash of the *previous* row (chain link).

This means any retroactive modification is detectable: if row N is altered,
its hash will no longer match what row N+1 records as the previous hash.

Schema
------
    audit_log (
        id          INTEGER PRIMARY KEY,
        entry_time  TEXT    NOT NULL,
        event_type  TEXT    NOT NULL,
        order_id    TEXT,
        symbol      TEXT,
        payload     TEXT,          -- JSON-serialised event payload
        row_hash    TEXT    NOT NULL,
        prev_hash   TEXT    NOT NULL
    )

Usage
-----
::

    audit = AuditLog("execution/audit.db")
    audit.log_order(order)
    audit.log_event("RISK_REJECTED", {"symbol": "BTC/USD", "reason": "..."})
    integrity_ok = audit.verify_chain()
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("execution.audit_log")

AUDIT_DB_PATH = Path(__file__).parent.parent / "audit.db"


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------

class AuditLog:
    """
    Append-only, hash-chained audit log backed by SQLite.

    Parameters
    ----------
    db_path : Path | str | None
        Path to the SQLite database file.  Defaults to ``execution/audit.db``.

    Thread safety
    -------------
    Uses a ``threading.Lock`` around every write.  Reads do not need locking
    because SQLite's WAL mode allows concurrent readers.
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self._db_path = Path(db_path) if db_path else AUDIT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock     = threading.Lock()
        self._last_hash: str = "GENESIS"
        self._conn     = self._connect()
        self._init_schema()
        self._load_last_hash()

    # ------------------------------------------------------------------
    # Connection / schema
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time  TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,
                order_id    TEXT,
                symbol      TEXT,
                payload     TEXT    NOT NULL,
                row_hash    TEXT    NOT NULL,
                prev_hash   TEXT    NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_order_id ON audit_log (order_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol ON audit_log (symbol)"
        )
        self._conn.commit()

    def _load_last_hash(self) -> None:
        """Initialise _last_hash from the most recent row in the DB."""
        cur = self._conn.execute(
            "SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row:
            self._last_hash = row[0]

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def log_order(self, order) -> None:
        """
        Write an order state snapshot to the audit log.

        Should be called every time an order transitions state:
        CREATED, SUBMITTED, PARTIAL, FILLED, CANCELLED, REJECTED.
        """
        event_type = f"ORDER_{order.status.value}"
        payload    = order.to_dict()
        self.log_event(
            event_type = event_type,
            payload    = payload,
            order_id   = order.order_id,
            symbol     = order.symbol,
        )

    def log_event(
        self,
        event_type: str,
        payload:    dict,
        order_id:   Optional[str] = None,
        symbol:     Optional[str] = None,
    ) -> int:
        """
        Write a generic event to the audit log.

        Parameters
        ----------
        event_type : str
            Short uppercase string describing the event.
        payload : dict
            JSON-serialisable event details.
        order_id : str | None
        symbol : str | None

        Returns
        -------
        int
            The new row's id.
        """
        with self._lock:
            entry_time   = datetime.now(timezone.utc).isoformat()
            payload_json = json.dumps(payload, default=str, sort_keys=True)

            # Compute row hash covering all content fields
            content_str  = "|".join([
                entry_time, event_type,
                order_id or "", symbol or "",
                payload_json, self._last_hash,
            ])
            row_hash     = _sha256(content_str)

            cur = self._conn.execute(
                """
                INSERT INTO audit_log
                    (entry_time, event_type, order_id, symbol, payload, row_hash, prev_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (entry_time, event_type, order_id, symbol,
                 payload_json, row_hash, self._last_hash),
            )
            self._conn.commit()
            self._last_hash = row_hash
            row_id          = cur.lastrowid

        log.debug("AuditLog: row %d event=%s order=%s", row_id, event_type, order_id or "-")
        return row_id  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> tuple[bool, str]:
        """
        Re-hash every row and confirm the chain is unbroken.

        Returns
        -------
        (True, "OK") if the chain is intact.
        (False, reason) if any row fails its hash check.

        This is O(n) in the number of audit rows and may be slow for large
        databases.  Call it offline, not in the hot path.
        """
        cur = self._conn.execute(
            "SELECT id, entry_time, event_type, order_id, symbol, payload, "
            "row_hash, prev_hash FROM audit_log ORDER BY id ASC"
        )
        rows = cur.fetchall()

        running_prev = "GENESIS"
        for row in rows:
            rid, entry_time, event_type, order_id, symbol, payload, row_hash, prev_hash = row

            if prev_hash != running_prev:
                return False, f"Row {rid}: prev_hash mismatch (expected {running_prev!r})"

            content_str = "|".join([
                entry_time, event_type,
                order_id or "", symbol or "",
                payload, prev_hash,
            ])
            expected = _sha256(content_str)
            if row_hash != expected:
                return False, f"Row {rid}: row_hash mismatch — row has been tampered with"

            running_prev = row_hash

        return True, "OK"

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_events_for_order(self, order_id: str) -> list[dict]:
        """Return all audit events for a given order_id."""
        cur = self._conn.execute(
            "SELECT id, entry_time, event_type, payload FROM audit_log "
            "WHERE order_id = ? ORDER BY id ASC",
            (order_id,),
        )
        return [
            {"id": r[0], "entry_time": r[1], "event_type": r[2],
             "payload": json.loads(r[3])}
            for r in cur.fetchall()
        ]

    def get_recent_events(self, limit: int = 100) -> list[dict]:
        """Return the *limit* most recent audit entries."""
        cur = self._conn.execute(
            "SELECT id, entry_time, event_type, order_id, symbol FROM audit_log "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [
            {"id": r[0], "entry_time": r[1], "event_type": r[2],
             "order_id": r[3], "symbol": r[4]}
            for r in cur.fetchall()
        ]

    def row_count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM audit_log")
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
