"""
idea-engine/ingestion/loaders/equity_curve_loader.py
──────────────────────────────────────────────────────
Normalises equity curves from multiple sources into a consistent pd.Series.

Sources handled
───────────────
  • SQLite equity_snapshots table (ts, equity columns)
  • CSV files with a timestamp column and an equity column
  • JSON list of {ts, equity} dicts or a bare list of floats
  • A raw pd.Series or list of floats (treated as already normalised)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Aliases for common equity column names across different output formats
_EQUITY_COL_ALIASES = [
    "equity", "portfolio_value", "capital", "nav",
    "cumulative_pnl", "cum_equity", "total_equity",
    "value", "balance",
]
_TS_COL_ALIASES = [
    "ts", "timestamp", "datetime", "date", "time",
    "exit_time", "bar_time",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _detect_col(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def _normalise_series(s: pd.Series) -> pd.Series:
    """Sort by index, drop NaN, ensure name='equity'."""
    s = s.dropna().sort_index()
    s.name = "equity"
    return s


# ── SQLite source ─────────────────────────────────────────────────────────────

def from_sqlite(
    db_path: Union[str, Path],
    table: str = "equity_snapshots",
    ts_col: Optional[str] = None,
    equity_col: Optional[str] = None,
) -> Optional[pd.Series]:
    """
    Load equity curve from a SQLite table.

    Parameters
    ----------
    db_path     : path to the .db file
    table       : table name (default: equity_snapshots)
    ts_col      : timestamp column name (auto-detected if None)
    equity_col  : equity column name (auto-detected if None)

    Returns
    -------
    pd.Series indexed by datetime, or None on failure.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        logger.warning("SQLite file not found: %s", db_path)
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        df   = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY rowid", conn)
        conn.close()
    except Exception as exc:
        logger.error("Failed to read %s.%s: %s", db_path.name, table, exc)
        return None

    tc = ts_col or _detect_col(df, _TS_COL_ALIASES)
    ec = equity_col or _detect_col(df, _EQUITY_COL_ALIASES)

    if ec is None:
        # Fall back to first numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            logger.warning("No numeric column found in %s.%s", db_path.name, table)
            return None
        ec = num_cols[0]

    if tc is not None:
        df[tc] = pd.to_datetime(df[tc], errors="coerce")
        df      = df.dropna(subset=[tc])
        s = df.set_index(tc)[ec].astype(float)
    else:
        s = df[ec].astype(float)
        s.index = pd.RangeIndex(len(s))

    logger.debug("Loaded equity from %s.%s: %d points", db_path.name, table, len(s))
    return _normalise_series(s)


# ── CSV source ────────────────────────────────────────────────────────────────

def from_csv(
    csv_path: Union[str, Path],
    ts_col: Optional[str] = None,
    equity_col: Optional[str] = None,
) -> Optional[pd.Series]:
    """
    Load equity curve from a CSV file.

    Parameters
    ----------
    csv_path    : path to CSV
    ts_col      : timestamp column (auto-detected if None)
    equity_col  : equity column (auto-detected if None)

    Returns
    -------
    pd.Series indexed by datetime (or integer if no ts column), or None.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning("CSV not found: %s", csv_path)
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.error("Failed to read CSV %s: %s", csv_path, exc)
        return None

    tc = ts_col or _detect_col(df, _TS_COL_ALIASES)
    ec = equity_col or _detect_col(df, _EQUITY_COL_ALIASES)

    if ec is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            logger.warning("No numeric column in %s", csv_path.name)
            return None
        ec = num_cols[0]

    s = pd.to_numeric(df[ec], errors="coerce")

    if tc is not None and tc in df.columns:
        idx = pd.to_datetime(df[tc], errors="coerce")
        s.index = idx
    else:
        s.index = pd.RangeIndex(len(s))

    logger.debug("Loaded equity from CSV %s: %d points", csv_path.name, len(s))
    return _normalise_series(s)


# ── JSON source ───────────────────────────────────────────────────────────────

def from_json(
    json_path: Union[str, Path],
    ts_key: str = "ts",
    equity_key: str = "equity",
) -> Optional[pd.Series]:
    """
    Load equity curve from a JSON file.

    Handles:
    - List of {ts, equity} dicts
    - Dict with 'equity' key mapping to a list
    - Bare list of floats (integer-indexed)

    Returns
    -------
    pd.Series or None.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        logger.warning("JSON not found: %s", json_path)
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to parse JSON %s: %s", json_path, exc)
        return None

    # List of dicts
    if isinstance(data, list):
        if len(data) == 0:
            return None
        if isinstance(data[0], dict):
            df = pd.DataFrame(data)
            tc = ts_key if ts_key in df.columns else _detect_col(df, _TS_COL_ALIASES)
            ec = equity_key if equity_key in df.columns else _detect_col(df, _EQUITY_COL_ALIASES)
            if ec is None:
                return None
            s = pd.to_numeric(df[ec], errors="coerce")
            if tc:
                s.index = pd.to_datetime(df[tc], errors="coerce")
            return _normalise_series(s)
        else:
            # Bare list of numbers
            s = pd.Series(pd.to_numeric(pd.Series(data), errors="coerce"), dtype=float)
            return _normalise_series(s)

    # Dict with equity key
    if isinstance(data, dict):
        if equity_key in data:
            vals = data[equity_key]
            tss  = data.get(ts_key) or data.get("timestamps") or data.get("dates")
            s    = pd.Series(pd.to_numeric(pd.Series(vals), errors="coerce"), dtype=float)
            if tss:
                s.index = pd.to_datetime(pd.Series(tss), errors="coerce")
            return _normalise_series(s)

    logger.warning("Could not extract equity curve from JSON %s", json_path.name)
    return None


# ── Raw series / list source ──────────────────────────────────────────────────

def from_series(data: Union[pd.Series, list]) -> Optional[pd.Series]:
    """
    Normalise a raw pd.Series or list of floats.

    Returns
    -------
    pd.Series (sorted, no NaN, named 'equity') or None.
    """
    if isinstance(data, list):
        s = pd.Series(data, dtype=float)
    elif isinstance(data, pd.Series):
        s = data.astype(float)
    else:
        logger.warning("from_series: unsupported type %s", type(data))
        return None
    return _normalise_series(s)


# ── Multi-source loader ───────────────────────────────────────────────────────

def load_equity_curve(
    source: Union[str, Path, pd.Series, list],
    source_type: Optional[str] = None,
    **kwargs,
) -> Optional[pd.Series]:
    """
    Universal equity curve loader.  Dispatches to the correct sub-loader
    based on source_type or file extension.

    Parameters
    ----------
    source      : path (str/Path), pd.Series, or list
    source_type : 'sqlite', 'csv', 'json', 'series' — auto-detected if None
    **kwargs    : forwarded to the specific loader

    Returns
    -------
    pd.Series or None.
    """
    if isinstance(source, (pd.Series, list)):
        return from_series(source)

    path = Path(source)
    ext  = path.suffix.lower()

    if source_type == "sqlite" or ext in (".db", ".sqlite", ".sqlite3"):
        return from_sqlite(path, **kwargs)
    if source_type == "csv" or ext == ".csv":
        return from_csv(path, **kwargs)
    if source_type == "json" or ext == ".json":
        return from_json(path, **kwargs)

    logger.warning("Cannot determine source type for %s", source)
    return None
