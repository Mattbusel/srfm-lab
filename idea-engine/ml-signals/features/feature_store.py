"""
features/feature_store.py
==========================
SQLite-backed feature cache for incremental feature computation.

Financial rationale
-------------------
Feature engineering over a multi-year crypto dataset is computationally
expensive – rolling correlations, VWAP, ATR, and 200-bar moving averages
all require O(N × lookback) operations.  In a live trading system, we
retrain models daily; recomputing all features from scratch each time
would add minutes of latency.

The FeatureStore solves this by:
1. Persisting computed feature rows keyed on (instrument, timestamp).
2. On each update call, comparing the DataFrame index against stored
   timestamps and only computing features for *new* bars.
3. Providing fast bulk retrieval for a date range.

Schema
------
Table: features_<instrument>
    ts          TEXT PRIMARY KEY   (ISO-8601 timestamp)
    <col_1>     REAL
    <col_2>     REAL
    ...

Table: metadata
    key   TEXT PRIMARY KEY
    value TEXT
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .feature_engineer import FeatureEngineer


# ---------------------------------------------------------------------------
# FeatureStore
# ---------------------------------------------------------------------------

class FeatureStore:
    """Incremental SQLite cache for computed feature DataFrames.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file.  Created if it does not exist.
    engineer : FeatureEngineer, optional
        Feature engineering pipeline.  A default instance is created if
        not supplied.
    """

    def __init__(
        self,
        db_path: str | Path = "feature_store.db",
        engineer: Optional[FeatureEngineer] = None,
    ) -> None:
        self.db_path   = Path(db_path)
        self.engineer  = engineer or FeatureEngineer()
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._ensure_metadata_table()
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "FeatureStore":
        self._get_conn()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _table_name(self, instrument: str) -> str:
        safe = instrument.replace("-", "_").replace("/", "_").upper()
        return f"features_{safe}"

    def _ensure_metadata_table(self) -> None:
        conn = self._get_conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.commit()

    def _ensure_feature_table(self, instrument: str, columns: List[str]) -> None:
        """Create or migrate the feature table for ``instrument``."""
        conn = self._get_conn()
        table = self._table_name(instrument)
        col_defs = ", ".join(f'"{c}" REAL' for c in columns)
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{table}" '
            f'(ts TEXT PRIMARY KEY, {col_defs})'
        )
        # Add any missing columns (schema evolution)
        existing = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        for col in columns:
            if col not in existing:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" REAL DEFAULT NULL')
        conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, instrument: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Compute and cache features for any new bars in ``raw_df``.

        Parameters
        ----------
        instrument : str
            Identifier string (e.g. ``'BTC-USDT'``).
        raw_df : pd.DataFrame
            Raw OHLCV + BH data with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Full feature DataFrame for all cached rows of ``instrument``.
        """
        conn   = self._get_conn()
        table  = self._table_name(instrument)

        # Find which timestamps are already cached
        try:
            cached_ts = set(
                row[0] for row in conn.execute(f'SELECT ts FROM "{table}"')
            )
        except sqlite3.OperationalError:
            cached_ts = set()

        # Identify new rows
        raw_df_ts = set(raw_df.index.astype(str))
        new_ts    = raw_df_ts - cached_ts

        if new_ts:
            # We must compute features over enough history to fill lookback windows.
            # Compute on full df, then only persist the new rows.
            feat_df = self.engineer.transform(raw_df)
            self._ensure_feature_table(instrument, list(feat_df.columns))

            new_rows = feat_df[feat_df.index.astype(str).isin(new_ts)]
            self._bulk_insert(conn, table, new_rows)
            conn.commit()

        return self.load(instrument)

    def load(
        self,
        instrument: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load cached features for ``instrument``.

        Parameters
        ----------
        instrument : str
        start : str, optional
            ISO-8601 start timestamp (inclusive).
        end : str, optional
            ISO-8601 end timestamp (inclusive).

        Returns
        -------
        pd.DataFrame  (empty if no rows cached)
        """
        conn  = self._get_conn()
        table = self._table_name(instrument)

        try:
            query = f'SELECT * FROM "{table}"'
            params: list = []
            if start and end:
                query += " WHERE ts BETWEEN ? AND ?"
                params = [start, end]
            elif start:
                query += " WHERE ts >= ?"
                params = [start]
            elif end:
                query += " WHERE ts <= ?"
                params = [end]
            query += " ORDER BY ts"

            df = pd.read_sql_query(query, conn, params=params, index_col="ts",
                                   parse_dates={"ts": {}})
            df.index = pd.to_datetime(df.index)
            return df

        except (sqlite3.OperationalError, pd.errors.DatabaseError):
            return pd.DataFrame()

    def list_instruments(self) -> List[str]:
        """Return list of instruments that have cached features."""
        conn = self._get_conn()
        tables = [
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'features_%'"
            )
        ]
        return [t[len("features_"):].replace("_", "-") for t in tables]

    def invalidate(self, instrument: str, from_ts: Optional[str] = None) -> int:
        """Delete cached rows for ``instrument``, optionally from a timestamp.

        Useful when upstream raw data is corrected.

        Returns
        -------
        int  number of rows deleted.
        """
        conn  = self._get_conn()
        table = self._table_name(instrument)
        try:
            if from_ts:
                cur = conn.execute(f'DELETE FROM "{table}" WHERE ts >= ?', (from_ts,))
            else:
                cur = conn.execute(f'DELETE FROM "{table}"')
            conn.commit()
            return cur.rowcount
        except sqlite3.OperationalError:
            return 0

    def stats(self, instrument: str) -> dict:
        """Return cache statistics for ``instrument``."""
        conn  = self._get_conn()
        table = self._table_name(instrument)
        try:
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            min_ts    = conn.execute(f'SELECT MIN(ts) FROM "{table}"').fetchone()[0]
            max_ts    = conn.execute(f'SELECT MAX(ts) FROM "{table}"').fetchone()[0]
            return {"instrument": instrument, "rows": row_count,
                    "min_ts": min_ts, "max_ts": max_ts}
        except sqlite3.OperationalError:
            return {"instrument": instrument, "rows": 0}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bulk_insert(
        self, conn: sqlite3.Connection, table: str, df: pd.DataFrame
    ) -> None:
        if df.empty:
            return
        cols   = list(df.columns)
        placeholders = ", ".join(["?"] * (len(cols) + 1))
        sql    = (
            f'INSERT OR REPLACE INTO "{table}" (ts, '
            + ", ".join(f'"{c}"' for c in cols)
            + f") VALUES ({placeholders})"
        )
        rows = []
        for ts, row in df.iterrows():
            vals = [str(ts)] + [
                None if (isinstance(v, float) and np.isnan(v)) else float(v)
                for v in row.values
            ]
            rows.append(vals)
        conn.executemany(sql, rows)

    def set_metadata(self, key: str, value: object) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        conn.commit()

    def get_metadata(self, key: str, default=None):
        conn = self._get_conn()
        row  = conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        return json.loads(row[0])
