"""
idea-engine/feature-store/store.py
=====================================
Core FeatureStore: a persistent cache of pre-computed signal values backed
by SQLite (WAL mode) with optional Parquet file export.

Architecture
------------
    SQLite (idea_engine.db)
        └── feature_cache table    : per-bar signal values
        └── feature_metadata table : per-signal statistics
        └── ic_history table       : rolling IC records

The store is designed to be used by the FeaturePipeline (pipeline.py) and
directly by the hypothesis generator and serendipity engine.

Thread-safety: SQLite WAL mode allows one writer + many readers.
Bulk operations use transactions for performance.

Usage
-----
    from feature_store.store import FeatureStore
    from signal_library import RSI, MACD

    store = FeatureStore("idea_engine.db")
    rsi   = RSI(period=14)

    # Compute and cache
    store.compute_and_cache("BTC", rsi, df)

    # Retrieve
    series = store.get("BTC", "rsi", start_ts="2024-01-01", end_ts="2024-12-31")

    # Get feature matrix (multiple signals × symbols)
    matrix = store.get_feature_matrix(
        symbols=["BTC", "ETH"],
        signal_names=["rsi", "macd", "garch_vol"],
        start_ts="2024-01-01",
        end_ts="2024-12-31",
    )
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH  = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")
SCHEMA_SQL_PATH  = Path(__file__).parent / "schema_extension.sql"
_BULK_CHUNK_SIZE = 5_000   # rows per INSERT batch
_DEFAULT_MAX_AGE = 24.0    # hours before a cached feature is considered stale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_to_str(ts: Any) -> Optional[str]:
    """Convert various timestamp types to ISO-8601 string."""
    if ts is None:
        return None
    if isinstance(ts, str):
        return ts
    if isinstance(ts, pd.Timestamp):
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(ts)


def _value_to_real(v: Any) -> Optional[float]:
    """Convert a Python/numpy value to a SQLite-compatible REAL or None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# FeatureStore
# ---------------------------------------------------------------------------

class FeatureStore:
    """
    Persistent feature cache for the SRFM Idea Engine.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file.
    parquet_dir : str | Path | None
        If provided, features are also persisted as Parquet files under
        this directory (symbol/signal_name.parquet). Useful for fast
        bulk loading.
    max_workers : int
        Thread pool size for bulk_compute.
    auto_init : bool
        If True, apply schema_extension.sql on first connection.
    """

    def __init__(
        self,
        db_path:    Union[str, Path]             = DEFAULT_DB_PATH,
        parquet_dir: Optional[Union[str, Path]]  = None,
        max_workers: int                          = 4,
        auto_init:  bool                          = True,
    ) -> None:
        self.db_path     = Path(db_path)
        self.parquet_dir = Path(parquet_dir) if parquet_dir else None
        self.max_workers = max_workers
        self._local      = threading.local()   # per-thread connection cache

        if auto_init:
            self._apply_schema()

        if self.parquet_dir:
            self.parquet_dir.mkdir(parents=True, exist_ok=True)

    # ── Connection management ─────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _apply_schema(self) -> None:
        """Apply schema_extension.sql if the file exists."""
        if SCHEMA_SQL_PATH.exists():
            sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
            conn = self._conn()
            try:
                conn.executescript(sql)
                conn.commit()
            except sqlite3.OperationalError as exc:
                logger.warning("FeatureStore: schema apply warning: %s", exc)
        else:
            logger.warning(
                "FeatureStore: schema file not found at %s. "
                "Run schema_extension.sql manually.",
                SCHEMA_SQL_PATH,
            )

    def close(self) -> None:
        """Close the thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── Core: compute_and_cache ───────────────────────────────────────────

    def compute_and_cache(
        self,
        symbol:      str,
        signal:      Any,   # Signal instance
        df:          pd.DataFrame,
        overwrite:   bool = True,
    ) -> pd.Series:
        """
        Compute a signal over df and persist results to the cache.

        Parameters
        ----------
        symbol : str
            Asset symbol (e.g. "BTC").
        signal : Signal
            An instantiated Signal object with a .compute(df) method.
        df : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex (or string-convertible index).
        overwrite : bool
            If True, overwrite existing cached values for this symbol/signal.

        Returns
        -------
        pd.Series
            The computed signal values.
        """
        signal_name = signal.name
        try:
            values = signal.compute(df)
        except Exception as exc:
            logger.error(
                "FeatureStore: compute failed for signal '%s' / symbol '%s': %s",
                signal_name, symbol, exc
            )
            raise

        self._persist_series(symbol, signal_name, values, overwrite=overwrite)
        self._upsert_metadata(
            signal_name=signal_name,
            category=getattr(signal, "category", "unknown"),
            lookback=getattr(signal, "lookback", 0),
            signal_type=getattr(signal, "signal_type", "continuous"),
        )
        logger.debug(
            "Cached %d values for %s/%s.", len(values), symbol, signal_name
        )
        return values

    # ── Core: get ─────────────────────────────────────────────────────────

    def get(
        self,
        symbol:       str,
        signal_name:  str,
        start_ts:     Optional[Any] = None,
        end_ts:       Optional[Any] = None,
    ) -> pd.Series:
        """
        Retrieve cached signal values for a symbol.

        Parameters
        ----------
        symbol : str
        signal_name : str
        start_ts, end_ts : str | pd.Timestamp | datetime | None
            Inclusive bounds. If None, returns all cached values.

        Returns
        -------
        pd.Series with DatetimeIndex, or empty Series if not found.
        """
        params: List[Any] = [symbol, signal_name]
        sql = (
            "SELECT ts, value FROM feature_cache "
            "WHERE symbol = ? AND signal_name = ?"
        )
        if start_ts is not None:
            sql += " AND ts >= ?"
            params.append(_ts_to_str(start_ts))
        if end_ts is not None:
            sql += " AND ts <= ?"
            params.append(_ts_to_str(end_ts))
        sql += " ORDER BY ts"

        conn  = self._conn()
        rows  = conn.execute(sql, params).fetchall()
        if not rows:
            return pd.Series(dtype=float, name=signal_name)

        index  = pd.to_datetime([r["ts"] for r in rows], utc=True)
        values = [r["value"] for r in rows]
        return pd.Series(values, index=index, name=signal_name, dtype=float)

    # ── Core: get_feature_matrix ──────────────────────────────────────────

    def get_feature_matrix(
        self,
        symbols:      Sequence[str],
        signal_names: Sequence[str],
        start_ts:     Optional[Any] = None,
        end_ts:       Optional[Any] = None,
        how:          str           = "outer",
    ) -> pd.DataFrame:
        """
        Retrieve a feature matrix for multiple symbols and signals.

        Returns a DataFrame with a MultiIndex of (symbol, ts) and one column
        per signal.  If ``how='outer'``, missing combinations are filled with NaN.

        Parameters
        ----------
        symbols : list[str]
        signal_names : list[str]
        start_ts, end_ts : optional timestamp bounds
        how : 'outer' | 'inner'
            How to join the individual series.

        Returns
        -------
        pd.DataFrame
            Index: pd.DatetimeIndex (when single symbol) or
                   pd.MultiIndex(symbol, ts) (when multiple symbols).
            Columns: signal_names
        """
        if len(symbols) == 1:
            return self._matrix_single_symbol(
                symbols[0], signal_names, start_ts, end_ts
            )

        frames = {}
        for sym in symbols:
            df = self._matrix_single_symbol(sym, signal_names, start_ts, end_ts)
            if not df.empty:
                frames[sym] = df

        if not frames:
            return pd.DataFrame(columns=list(signal_names))

        # Concatenate with symbol in outer index
        combined = pd.concat(frames, axis=0, names=["symbol", "ts"])
        combined.sort_index(inplace=True)
        return combined

    def _matrix_single_symbol(
        self,
        symbol:       str,
        signal_names: Sequence[str],
        start_ts:     Optional[Any],
        end_ts:       Optional[Any],
    ) -> pd.DataFrame:
        """Fetch a feature matrix for a single symbol."""
        series_list = []
        for sig_name in signal_names:
            s = self.get(symbol, sig_name, start_ts, end_ts)
            series_list.append(s.rename(sig_name))

        if not series_list:
            return pd.DataFrame(columns=list(signal_names))

        df = pd.concat(series_list, axis=1)
        df.index.name = "ts"
        return df

    # ── Staleness ─────────────────────────────────────────────────────────

    def is_stale(
        self,
        symbol:       str,
        signal_name:  str,
        max_age_hours: float = _DEFAULT_MAX_AGE,
    ) -> bool:
        """
        Return True if the cached feature is older than max_age_hours
        or does not exist.
        """
        sql = (
            "SELECT MAX(computed_at) AS last "
            "FROM feature_cache "
            "WHERE symbol = ? AND signal_name = ?"
        )
        row = self._conn().execute(sql, [symbol, signal_name]).fetchone()
        if not row or row["last"] is None:
            return True
        try:
            last = pd.Timestamp(row["last"], tz="UTC")
            age  = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 3600.0
            return age > max_age_hours
        except Exception:
            return True

    def staleness_report(
        self,
        symbols:       Optional[List[str]]  = None,
        signal_names:  Optional[List[str]]  = None,
        max_age_hours: float                = _DEFAULT_MAX_AGE,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of (symbol, signal_name, age_hours, is_stale)
        for all symbol/signal pairs matching the filters.
        """
        sql = """
            SELECT symbol, signal_name,
                   MAX(computed_at) AS last_computed,
                   CAST(
                       (julianday('now') - julianday(MAX(computed_at))) * 24
                       AS REAL
                   ) AS age_hours
            FROM feature_cache
            {}
            GROUP BY symbol, signal_name
        """.format(
            "WHERE 1=1"
            + (" AND symbol IN ({})".format(",".join("?" * len(symbols)))
               if symbols else "")
            + (" AND signal_name IN ({})".format(",".join("?" * len(signal_names)))
               if signal_names else "")
        )
        params: List[Any] = []
        if symbols:
            params += list(symbols)
        if signal_names:
            params += list(signal_names)

        rows = self._conn().execute(sql, params).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["symbol", "signal_name", "last_computed", "age_hours", "is_stale"]
            )

        data = [
            {
                "symbol":      r["symbol"],
                "signal_name": r["signal_name"],
                "last_computed": r["last_computed"],
                "age_hours":   r["age_hours"],
                "is_stale":    (r["age_hours"] or 9999) > max_age_hours,
            }
            for r in rows
        ]
        return pd.DataFrame(data)

    # ── Refresh ───────────────────────────────────────────────────────────

    def refresh(
        self,
        symbol:        str,
        df:            pd.DataFrame,
        signal_instances: List[Any],
        signal_names:  Optional[List[str]] = None,
        max_age_hours: float               = _DEFAULT_MAX_AGE,
        force:         bool                = False,
    ) -> Dict[str, bool]:
        """
        Recompute stale features for a symbol.

        Parameters
        ----------
        symbol : str
        df : pd.DataFrame
            OHLCV DataFrame to recompute from.
        signal_instances : list[Signal]
            All signal instances to consider.
        signal_names : list[str] | None
            Restrict to these signal names. If None, consider all instances.
        max_age_hours : float
        force : bool
            If True, recompute even if not stale.

        Returns
        -------
        dict[signal_name, was_refreshed]
        """
        signals_map = {s.name: s for s in signal_instances if s.name}
        if signal_names:
            signals_map = {k: v for k, v in signals_map.items()
                           if k in signal_names}

        results: Dict[str, bool] = {}
        for name, sig in signals_map.items():
            if force or self.is_stale(symbol, name, max_age_hours):
                try:
                    self.compute_and_cache(symbol, sig, df, overwrite=True)
                    results[name] = True
                except Exception as exc:
                    logger.error("refresh: failed for %s/%s: %s", symbol, name, exc)
                    results[name] = False
            else:
                results[name] = False

        return results

    # ── Bulk compute ──────────────────────────────────────────────────────

    def bulk_compute(
        self,
        symbols:          Sequence[str],
        signal_instances: List[Any],
        df_dict:          Dict[str, pd.DataFrame],
        overwrite:        bool  = True,
        max_workers:      Optional[int] = None,
    ) -> Dict[Tuple[str, str], bool]:
        """
        Parallelised bulk computation of all signals for all symbols.

        Parameters
        ----------
        symbols : list[str]
        signal_instances : list[Signal]
        df_dict : dict[symbol, pd.DataFrame]
        overwrite : bool
        max_workers : int | None  (defaults to self.max_workers)

        Returns
        -------
        dict[(symbol, signal_name), success_bool]
        """
        workers = max_workers or self.max_workers
        tasks: List[Tuple[str, Any]] = []

        for sym in symbols:
            if sym not in df_dict:
                logger.warning("bulk_compute: no DataFrame for symbol '%s'. Skipping.", sym)
                continue
            for sig in signal_instances:
                tasks.append((sym, sig))

        results: Dict[Tuple[str, str], bool] = {}

        def _task(args: Tuple[str, Any]) -> Tuple[str, str, bool]:
            sym, sig = args
            df = df_dict[sym]
            try:
                self.compute_and_cache(sym, sig, df, overwrite=overwrite)
                return sym, sig.name, True
            except Exception as exc:
                logger.error(
                    "bulk_compute: error %s/%s: %s", sym, sig.name, exc
                )
                return sym, sig.name, False

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_task, t): t for t in tasks}
            for fut in as_completed(futures):
                sym, sig_name, ok = fut.result()
                results[(sym, sig_name)] = ok

        n_ok  = sum(results.values())
        n_tot = len(results)
        logger.info(
            "bulk_compute: %d/%d tasks succeeded.", n_ok, n_tot
        )
        return results

    # ── Invalidate ────────────────────────────────────────────────────────

    def invalidate(
        self,
        symbol:      str,
        signal_name: Optional[str] = None,
    ) -> int:
        """
        Delete cached features for a symbol (and optionally a specific signal).

        Returns the number of rows deleted.
        """
        conn = self._conn()
        if signal_name is not None:
            sql    = "DELETE FROM feature_cache WHERE symbol = ? AND signal_name = ?"
            params = [symbol, signal_name]
        else:
            sql    = "DELETE FROM feature_cache WHERE symbol = ?"
            params = [symbol]

        cursor = conn.execute(sql, params)
        conn.commit()
        deleted = cursor.rowcount
        logger.info(
            "invalidate: deleted %d rows for symbol='%s' signal='%s'.",
            deleted, symbol, signal_name or "*"
        )
        return deleted

    def invalidate_all(self) -> int:
        """Truncate the entire feature_cache table."""
        conn    = self._conn()
        cursor  = conn.execute("DELETE FROM feature_cache")
        conn.commit()
        return cursor.rowcount

    # ── Symbol / signal discovery ─────────────────────────────────────────

    def list_symbols(self) -> List[str]:
        """Return all symbols that have cached features."""
        rows = self._conn().execute(
            "SELECT DISTINCT symbol FROM feature_cache ORDER BY symbol"
        ).fetchall()
        return [r["symbol"] for r in rows]

    def list_cached_signals(self, symbol: Optional[str] = None) -> List[str]:
        """Return all signal names (optionally filtered by symbol)."""
        if symbol:
            rows = self._conn().execute(
                "SELECT DISTINCT signal_name FROM feature_cache "
                "WHERE symbol = ? ORDER BY signal_name",
                [symbol],
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT DISTINCT signal_name FROM feature_cache ORDER BY signal_name"
            ).fetchall()
        return [r["signal_name"] for r in rows]

    def cache_stats(self) -> pd.DataFrame:
        """
        Return per-symbol / per-signal row count and date range.
        """
        sql = """
            SELECT symbol, signal_name,
                   COUNT(*) AS n_rows,
                   MIN(ts)  AS first_ts,
                   MAX(ts)  AS last_ts
            FROM feature_cache
            GROUP BY symbol, signal_name
            ORDER BY symbol, signal_name
        """
        rows = self._conn().execute(sql).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["symbol", "signal_name", "n_rows", "first_ts", "last_ts"]
            )
        return pd.DataFrame([dict(r) for r in rows])

    # ── Parquet export/import ─────────────────────────────────────────────

    def export_to_parquet(
        self,
        symbol:      str,
        signal_name: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Export cached feature to a Parquet file.

        If output_path is None and parquet_dir is set, saves to
        parquet_dir / symbol / signal_name.parquet.
        """
        series = self.get(symbol, signal_name)
        if series.empty:
            raise ValueError(f"No cached data for {symbol}/{signal_name}.")

        if output_path is None:
            if self.parquet_dir is None:
                raise ValueError("parquet_dir not configured.")
            sym_dir = self.parquet_dir / symbol
            sym_dir.mkdir(parents=True, exist_ok=True)
            output_path = sym_dir / f"{signal_name}.parquet"
        else:
            output_path = Path(output_path)

        df = series.reset_index()
        df.columns = ["ts", "value"]
        df.to_parquet(output_path, index=False)
        logger.info("Exported %d rows to %s", len(df), output_path)
        return output_path

    def load_from_parquet(
        self,
        symbol:      str,
        signal_name: str,
        path:        Optional[Union[str, Path]] = None,
        overwrite:   bool                       = True,
    ) -> int:
        """
        Load a Parquet file into the feature_cache.

        If path is None, looks in parquet_dir / symbol / signal_name.parquet.
        Returns the number of rows inserted.
        """
        if path is None:
            if self.parquet_dir is None:
                raise ValueError("parquet_dir not configured.")
            path = self.parquet_dir / symbol / f"{signal_name}.parquet"
        else:
            path = Path(path)

        df     = pd.read_parquet(path)
        series = pd.Series(
            df["value"].values,
            index=pd.to_datetime(df["ts"]),
            name=signal_name,
        )
        self._persist_series(symbol, signal_name, series, overwrite=overwrite)
        return len(series)

    # ── Internal persistence helpers ──────────────────────────────────────

    def _persist_series(
        self,
        symbol:      str,
        signal_name: str,
        series:      pd.Series,
        overwrite:   bool = True,
    ) -> None:
        """Write a pd.Series to the feature_cache table."""
        if series.empty:
            return

        now_str = _now_utc()
        rows: List[Tuple] = []
        for ts, val in series.items():
            ts_str = _ts_to_str(ts)
            if ts_str is None:
                continue
            rows.append((symbol, signal_name, ts_str, _value_to_real(val), now_str))

        conn = self._conn()
        if overwrite:
            conflict_clause = "OR REPLACE"
        else:
            conflict_clause = "OR IGNORE"

        sql = (
            f"INSERT {conflict_clause} INTO feature_cache "
            "(symbol, signal_name, ts, value, computed_at) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        # Chunked inserts to avoid SQLite parameter limits
        for start in range(0, len(rows), _BULK_CHUNK_SIZE):
            chunk = rows[start: start + _BULK_CHUNK_SIZE]
            conn.executemany(sql, chunk)
        conn.commit()

    def _upsert_metadata(
        self,
        signal_name: str,
        category:    str,
        lookback:    int,
        signal_type: str,
    ) -> None:
        """Insert or update feature_metadata row."""
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO feature_metadata
                (signal_name, category, lookback, signal_type, last_computed,
                 n_symbols, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(signal_name) DO UPDATE SET
                category     = excluded.category,
                lookback     = excluded.lookback,
                signal_type  = excluded.signal_type,
                last_computed= excluded.last_computed,
                n_symbols    = n_symbols + 1,
                updated_at   = excluded.updated_at
            """,
            [signal_name, category, lookback, signal_type, _now_utc(), _now_utc()],
        )
        conn.commit()

    def update_ic_stats(
        self,
        signal_name: str,
        mean_ic:     float,
        icir:        float,
    ) -> None:
        """Update IC statistics in feature_metadata."""
        conn = self._conn()
        conn.execute(
            """
            UPDATE feature_metadata
            SET mean_ic = ?, icir = ?, updated_at = ?
            WHERE signal_name = ?
            """,
            [mean_ic, icir, _now_utc(), signal_name],
        )
        conn.commit()
