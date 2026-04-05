"""
macro-factor/factor_store.py
──────────────────────────────
SQLite storage for macro factor data and regime history.

Schema
──────
  macro_factors:   One row per (factor, date) — stores the signal and payload.
  macro_regimes:   One row per date — stores the full regime classification.

Design follows the same upsert pattern as onchain/data_store.py for consistency.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")

_DDL = """
CREATE TABLE IF NOT EXISTS macro_factors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    factor      TEXT    NOT NULL,       -- "dxy" | "rates" | "vix" | "gold" | "equity" | "liquidity"
    date        TEXT    NOT NULL,       -- YYYY-MM-DD
    signal      REAL    NOT NULL,       -- [-1, +1]
    payload     TEXT,                   -- full JSON blob
    created_at  TEXT    NOT NULL,
    UNIQUE(factor, date) ON CONFLICT REPLACE
);

CREATE INDEX IF NOT EXISTS idx_macro_factors_date
    ON macro_factors(factor, date DESC);

CREATE TABLE IF NOT EXISTS macro_regimes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    date                TEXT    NOT NULL UNIQUE,
    regime              TEXT    NOT NULL,       -- RISK_ON | RISK_NEUTRAL | RISK_OFF | CRISIS
    composite_score     REAL    NOT NULL,
    position_multiplier REAL    NOT NULL,
    crisis_override     INTEGER NOT NULL DEFAULT 0,
    crisis_reason       TEXT,
    confidence          REAL,
    payload             TEXT,                   -- full JSON of RegimeClassification
    created_at          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_macro_regimes_date
    ON macro_regimes(date DESC);
"""


@contextmanager
def _connect(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class FactorStore:
    """Persistent storage for macro factor data and regime classifications.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with _connect(self.db_path) as conn:
            conn.executescript(_DDL)
        logger.debug("FactorStore schema ready at %s", self.db_path)

    # ── Factor Storage ────────────────────────────────────────────────────

    def upsert_factor(
        self,
        factor: str,
        date: str,
        signal: float,
        payload: Dict[str, Any],
    ) -> None:
        """Insert or replace a single factor reading."""
        now = datetime.now(timezone.utc).isoformat()
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO macro_factors (factor, date, signal, payload, created_at) VALUES (?,?,?,?,?)",
                (factor, date, round(float(signal), 6), json.dumps(payload), now),
            )
        logger.debug("Upserted macro factor %s for %s: %.3f", factor, date, signal)

    def fetch_factor_history(
        self,
        factor: str,
        days: int = 90,
    ) -> pd.DataFrame:
        """Fetch the last `days` readings for a factor."""
        sql = """
            SELECT date, signal, payload
            FROM   macro_factors
            WHERE  factor = ?
            ORDER  BY date DESC
            LIMIT  ?
        """
        with _connect(self.db_path) as conn:
            rows = conn.execute(sql, (factor, days)).fetchall()
        if not rows:
            return pd.DataFrame(columns=["date", "signal", "payload"])
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def fetch_latest_factor(self, factor: str) -> Optional[Dict[str, Any]]:
        """Return the most recent row for a factor."""
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT date, signal, payload FROM macro_factors WHERE factor=? ORDER BY date DESC LIMIT 1",
                (factor,),
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["payload"] = json.loads(d["payload"] or "{}")
        return d

    def upsert_all_factors(self, date: str, factor_signals: Dict[str, float], payloads: Dict[str, Dict]) -> None:
        """Bulk upsert all factor signals for a given date."""
        for factor, signal in factor_signals.items():
            self.upsert_factor(factor, date, signal, payloads.get(factor, {}))

    # ── Regime Storage ────────────────────────────────────────────────────

    def upsert_regime(
        self,
        date: str,
        regime: str,
        composite_score: float,
        position_multiplier: float,
        crisis_override: bool,
        crisis_reason: str,
        confidence: float,
        payload: Dict[str, Any],
    ) -> None:
        """Insert or replace a regime classification for a given date."""
        now = datetime.now(timezone.utc).isoformat()
        sql = """
            INSERT INTO macro_regimes
                (date, regime, composite_score, position_multiplier,
                 crisis_override, crisis_reason, confidence, payload, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """
        with _connect(self.db_path) as conn:
            conn.execute(sql, (
                date, regime,
                round(float(composite_score), 6),
                round(float(position_multiplier), 4),
                int(crisis_override),
                crisis_reason,
                round(float(confidence), 4),
                json.dumps(payload),
                now,
            ))
        logger.info("Upserted macro regime %s for %s (score=%.3f)", regime, date, composite_score)

    def fetch_regime_history(self, days: int = 90) -> pd.DataFrame:
        """Fetch regime classification history."""
        sql = """
            SELECT date, regime, composite_score, position_multiplier,
                   crisis_override, confidence
            FROM   macro_regimes
            ORDER  BY date DESC
            LIMIT  ?
        """
        with _connect(self.db_path) as conn:
            rows = conn.execute(sql, (days,)).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def fetch_latest_regime(self) -> Optional[Dict[str, Any]]:
        """Return the most recent regime classification."""
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM macro_regimes ORDER BY date DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["payload"] = json.loads(d.get("payload") or "{}")
        return d

    def get_regime_transition(self) -> Optional[tuple[str, str]]:
        """Return (previous_regime, current_regime) if there was a transition today."""
        sql = "SELECT regime FROM macro_regimes ORDER BY date DESC LIMIT 2"
        with _connect(self.db_path) as conn:
            rows = conn.execute(sql).fetchall()
        if len(rows) < 2:
            return None
        current, previous = rows[0]["regime"], rows[1]["regime"]
        if current != previous:
            return (previous, current)
        return None

    def purge_old_rows(self, keep_days: int = 730) -> int:
        """Delete rows older than keep_days."""
        cutoff = (pd.Timestamp.now() - pd.Timedelta(days=keep_days)).strftime("%Y-%m-%d")
        with _connect(self.db_path) as conn:
            c1 = conn.execute("DELETE FROM macro_factors WHERE date < ?", (cutoff,)).rowcount
            c2 = conn.execute("DELETE FROM macro_regimes WHERE date < ?", (cutoff,)).rowcount
        total = c1 + c2
        logger.info("FactorStore: purged %d old rows (cutoff %s)", total, cutoff)
        return total

    def get_factor_signal_table(self) -> pd.DataFrame:
        """Return a pivot table of the latest signal for each factor."""
        sql = """
            SELECT f.factor, f.date, f.signal
            FROM   macro_factors f
            WHERE  (f.factor, f.date) IN (
                SELECT factor, MAX(date)
                FROM   macro_factors
                GROUP  BY factor
            )
            ORDER BY f.factor
        """
        with _connect(self.db_path) as conn:
            rows = conn.execute(sql).fetchall()
        if not rows:
            return pd.DataFrame(columns=["factor", "date", "signal"])
        return pd.DataFrame([dict(r) for r in rows])
