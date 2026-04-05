"""
onchain/data_store.py
──────────────────────
SQLite storage for on-chain time series and computed metric results.

Design principles
─────────────────
- Upsert semantics: inserting the same (symbol, metric, date) replaces the row.
- Rolling-window queries: efficiently fetch the last N days of any metric.
- Typed JSON blob for flexible metric payloads; key numeric fields are columns
  for indexing and fast aggregation.
- Schema is auto-created on first use — no migration tooling required.
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
CREATE TABLE IF NOT EXISTS onchain_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    metric      TEXT    NOT NULL,
    date        TEXT    NOT NULL,    -- YYYY-MM-DD
    signal      REAL,               -- [-1, +1] normalised signal
    value       REAL,               -- primary numeric value (ratio, score, etc.)
    source      TEXT,               -- "coinmetrics" | "simulated"
    payload     TEXT,               -- full JSON blob of the result dataclass
    created_at  TEXT    NOT NULL,
    UNIQUE(symbol, metric, date) ON CONFLICT REPLACE
);

CREATE INDEX IF NOT EXISTS idx_onchain_symbol_metric_date
    ON onchain_metrics(symbol, metric, date DESC);

CREATE TABLE IF NOT EXISTS onchain_composite (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    composite_score REAL    NOT NULL,   -- [-1, +1]
    mvrv_signal     REAL,
    nvt_signal      REAL,
    sopr_signal     REAL,
    exchange_signal REAL,
    whale_signal    REAL,
    payload         TEXT,
    created_at      TEXT    NOT NULL,
    UNIQUE(symbol, date) ON CONFLICT REPLACE
);

CREATE INDEX IF NOT EXISTS idx_onchain_composite_date
    ON onchain_composite(symbol, date DESC);
"""


@contextmanager
def _connect(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class OnChainDataStore:
    """Persistent storage for on-chain metric time series.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.  Defaults to the shared idea_engine.db.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with _connect(self.db_path) as conn:
            conn.executescript(_DDL)
        logger.debug("OnChainDataStore schema ready at %s", self.db_path)

    # ── Individual Metric Storage ──────────────────────────────────────────

    def upsert_metric(
        self,
        symbol: str,
        metric: str,
        date: str,
        signal: float,
        value: float,
        source: str,
        payload: Dict[str, Any],
    ) -> None:
        """Insert or replace a single on-chain metric reading.

        Parameters
        ----------
        symbol:  e.g. "BTC-USD"
        metric:  e.g. "mvrv" | "nvt" | "sopr" | ...
        date:    YYYY-MM-DD string
        signal:  normalised [-1, +1] signal
        value:   primary numeric value (ratio, score, etc.)
        source:  "coinmetrics" | "simulated"
        payload: full result dict for archival
        """
        now = datetime.now(timezone.utc).isoformat()
        sql = """
            INSERT INTO onchain_metrics
                (symbol, metric, date, signal, value, source, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with _connect(self.db_path) as conn:
            conn.execute(sql, (
                symbol, metric, date,
                round(float(signal), 6),
                round(float(value), 6),
                source,
                json.dumps(payload),
                now,
            ))
        logger.debug("Upserted %s/%s for %s", symbol, metric, date)

    def fetch_metric_history(
        self,
        symbol: str,
        metric: str,
        days: int = 90,
    ) -> pd.DataFrame:
        """Fetch the last `days` rows for a given symbol/metric.

        Returns a DataFrame with columns: date, signal, value, source, payload.
        """
        sql = """
            SELECT date, signal, value, source, payload
            FROM   onchain_metrics
            WHERE  symbol = ? AND metric = ?
            ORDER  BY date DESC
            LIMIT  ?
        """
        with _connect(self.db_path) as conn:
            rows = conn.execute(sql, (symbol, metric, days)).fetchall()
        if not rows:
            return pd.DataFrame(columns=["date", "signal", "value", "source", "payload"])
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def fetch_latest_metric(
        self,
        symbol: str,
        metric: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the most recent row for a symbol/metric as a dict, or None."""
        sql = """
            SELECT date, signal, value, source, payload
            FROM   onchain_metrics
            WHERE  symbol = ? AND metric = ?
            ORDER  BY date DESC
            LIMIT  1
        """
        with _connect(self.db_path) as conn:
            row = conn.execute(sql, (symbol, metric)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result["payload"] or "{}")
        return result

    # ── Composite Score Storage ─────────────────────────────────────────────

    def upsert_composite(
        self,
        symbol: str,
        date: str,
        composite_score: float,
        component_signals: Dict[str, float],
        payload: Dict[str, Any],
    ) -> None:
        """Store the composite on-chain score and its component signals."""
        now = datetime.now(timezone.utc).isoformat()
        sql = """
            INSERT INTO onchain_composite
                (symbol, date, composite_score,
                 mvrv_signal, nvt_signal, sopr_signal,
                 exchange_signal, whale_signal, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with _connect(self.db_path) as conn:
            conn.execute(sql, (
                symbol, date,
                round(float(composite_score), 6),
                component_signals.get("mvrv"),
                component_signals.get("nvt"),
                component_signals.get("sopr"),
                component_signals.get("exchange_reserves"),
                component_signals.get("whale"),
                json.dumps(payload),
                now,
            ))
        logger.debug("Upserted composite score for %s on %s: %.3f", symbol, date, composite_score)

    def fetch_composite_history(
        self,
        symbol: str,
        days: int = 90,
    ) -> pd.DataFrame:
        """Fetch composite score history as a DataFrame."""
        sql = """
            SELECT date, composite_score,
                   mvrv_signal, nvt_signal, sopr_signal,
                   exchange_signal, whale_signal
            FROM   onchain_composite
            WHERE  symbol = ?
            ORDER  BY date DESC
            LIMIT  ?
        """
        with _connect(self.db_path) as conn:
            rows = conn.execute(sql, (symbol, days)).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def fetch_latest_composite(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the most recent composite score row."""
        sql = """
            SELECT *
            FROM   onchain_composite
            WHERE  symbol = ?
            ORDER  BY date DESC
            LIMIT  1
        """
        with _connect(self.db_path) as conn:
            row = conn.execute(sql, (symbol,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result.get("payload") or "{}")
        return result

    # ── Bulk Queries ────────────────────────────────────────────────────────

    def get_all_metrics_latest(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """Return the latest reading for every metric for a given symbol."""
        sql = """
            SELECT metric, date, signal, value, source, payload
            FROM   onchain_metrics
            WHERE  symbol = ?
              AND  (symbol, metric, date) IN (
                       SELECT symbol, metric, MAX(date)
                       FROM   onchain_metrics
                       WHERE  symbol = ?
                       GROUP  BY metric
                   )
        """
        with _connect(self.db_path) as conn:
            rows = conn.execute(sql, (symbol, symbol)).fetchall()
        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            d = dict(row)
            d["payload"] = json.loads(d["payload"] or "{}")
            result[d["metric"]] = d
        return result

    def purge_old_rows(self, keep_days: int = 730) -> int:
        """Delete rows older than `keep_days` to control DB size."""
        cutoff = (pd.Timestamp.now() - pd.Timedelta(days=keep_days)).strftime("%Y-%m-%d")
        with _connect(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM onchain_metrics WHERE date < ?", (cutoff,)
            )
            n1 = cur.rowcount
            cur = conn.execute(
                "DELETE FROM onchain_composite WHERE date < ?", (cutoff,)
            )
            n2 = cur.rowcount
        total = n1 + n2
        logger.info("Purged %d old on-chain rows (cutoff %s)", total, cutoff)
        return total
