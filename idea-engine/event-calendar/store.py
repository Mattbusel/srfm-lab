"""
SQLite storage for calendar events and outcome tracking.

Schema:
  events        -- raw event records
  event_outcomes -- what actually happened to price after each event
  signals       -- generated signals and their resolution status
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .aggregator import AggregatedEvent
from .signal_generator import EventSignal

logger = logging.getLogger(__name__)

DEFAULT_DB = Path(__file__).parent / "event_calendar.db"

DDL = """
CREATE TABLE IF NOT EXISTS events (
    event_id       TEXT PRIMARY KEY,
    event_name     TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    event_date     TEXT NOT NULL,
    category       TEXT,
    impact         TEXT,
    event_risk     REAL,
    sources        TEXT,      -- JSON list
    description    TEXT,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_symbol   ON events(symbol);
CREATE INDEX IF NOT EXISTS idx_events_date     ON events(event_date);
CREATE INDEX IF NOT EXISTS idx_events_impact   ON events(impact);

CREATE TABLE IF NOT EXISTS event_outcomes (
    outcome_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT NOT NULL REFERENCES events(event_id),
    symbol          TEXT NOT NULL,
    event_date      TEXT NOT NULL,
    price_at_event  REAL,
    price_24h_after REAL,
    price_7d_after  REAL,
    actual_pct_24h  REAL,
    actual_pct_7d   REAL,
    recorded_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outcomes_event ON event_outcomes(event_id);

CREATE TABLE IF NOT EXISTS signals (
    signal_id          TEXT PRIMARY KEY,
    signal_type        TEXT NOT NULL,
    source_event_id    TEXT,
    symbol             TEXT NOT NULL,
    direction          TEXT,
    confidence         REAL,
    allocation_modifier REAL,
    rationale          TEXT,
    generated_at       TEXT NOT NULL,
    valid_until        TEXT NOT NULL,
    resolved           INTEGER DEFAULT 0,
    resolution_note    TEXT,
    metadata           TEXT   -- JSON
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_active ON signals(valid_until, resolved);
"""


class EventStore:
    """
    Persistent storage for events and outcome tracking.

    Usage::

        store = EventStore()
        store.upsert_event(aggregated_event)
        store.save_signal(signal)
        outcomes = store.get_outcomes_for_event("event_id_123")
    """

    def __init__(self, db_path: Path = DEFAULT_DB) -> None:
        self._db_path = db_path
        self._init_db()

    # ── lifecycle ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(DDL)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── events ───────────────────────────────────────────────────────────────────

    def upsert_event(self, ev: AggregatedEvent) -> None:
        """Insert or replace an event record."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO events
                   (event_id, event_name, symbol, event_date, category,
                    impact, event_risk, sources, description, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    ev.event_id,
                    ev.event_name,
                    ev.symbol,
                    ev.event_date.isoformat(),
                    ev.category,
                    ev.impact,
                    ev.event_risk_score,
                    json.dumps(ev.sources),
                    ev.description,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def upsert_events(self, events: List[AggregatedEvent]) -> int:
        for ev in events:
            self.upsert_event(ev)
        return len(events)

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_events_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE symbol = ? ORDER BY event_date",
                (symbol.upper(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_events_in_range(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE event_date BETWEEN ? AND ? ORDER BY event_date",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_high_impact_events(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE impact = 'HIGH' ORDER BY event_date"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── outcomes ─────────────────────────────────────────────────────────────────

    def record_outcome(
        self,
        event_id: str,
        symbol: str,
        event_date: datetime,
        price_at_event: float,
        price_24h_after: Optional[float] = None,
        price_7d_after: Optional[float] = None,
    ) -> None:
        """Record what actually happened to price after an event."""
        pct_24h = (
            (price_24h_after - price_at_event) / price_at_event * 100
            if price_24h_after and price_at_event
            else None
        )
        pct_7d = (
            (price_7d_after - price_at_event) / price_at_event * 100
            if price_7d_after and price_at_event
            else None
        )
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO event_outcomes
                   (event_id, symbol, event_date, price_at_event,
                    price_24h_after, price_7d_after, actual_pct_24h,
                    actual_pct_7d, recorded_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    event_id, symbol, event_date.isoformat(),
                    price_at_event, price_24h_after, price_7d_after,
                    pct_24h, pct_7d,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_outcomes_for_event(self, event_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM event_outcomes WHERE event_id = ?", (event_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_category_outcome_stats(self, category: str) -> Dict[str, Any]:
        """
        Return aggregate outcome statistics for all events of a given category.

        Returns: {count, avg_pct_24h, avg_pct_7d, positive_rate_24h}
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT eo.actual_pct_24h, eo.actual_pct_7d
                   FROM event_outcomes eo
                   JOIN events e ON eo.event_id = e.event_id
                   WHERE e.category = ? AND eo.actual_pct_24h IS NOT NULL""",
                (category,),
            ).fetchall()
        if not rows:
            return {"count": 0, "avg_pct_24h": None, "avg_pct_7d": None, "positive_rate_24h": None}
        pct24 = [r["actual_pct_24h"] for r in rows]
        pct7d = [r["actual_pct_7d"] for r in rows if r["actual_pct_7d"] is not None]
        return {
            "count": len(pct24),
            "avg_pct_24h": sum(pct24) / len(pct24),
            "avg_pct_7d": sum(pct7d) / len(pct7d) if pct7d else None,
            "positive_rate_24h": sum(1 for p in pct24 if p > 0) / len(pct24),
        }

    # ── signals ──────────────────────────────────────────────────────────────────

    def save_signal(self, sig: EventSignal) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO signals
                   (signal_id, signal_type, source_event_id, symbol, direction,
                    confidence, allocation_modifier, rationale, generated_at,
                    valid_until, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    sig.signal_id, sig.signal_type, sig.source_event_id,
                    sig.symbol, sig.direction, sig.confidence,
                    sig.allocation_modifier, sig.rationale,
                    sig.generated_at.isoformat(), sig.valid_until.isoformat(),
                    json.dumps(sig.metadata),
                ),
            )

    def get_active_signals(self, ts: Optional[datetime] = None) -> List[Dict[str, Any]]:
        ts = ts or datetime.now(timezone.utc)
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM signals
                   WHERE resolved = 0
                     AND valid_until >= ?
                   ORDER BY generated_at DESC""",
                (ts.isoformat(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def resolve_signal(self, signal_id: str, note: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE signals SET resolved = 1, resolution_note = ? WHERE signal_id = ?",
                (note, signal_id),
            )
