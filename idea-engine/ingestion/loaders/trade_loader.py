"""
idea-engine/ingestion/loaders/trade_loader.py
──────────────────────────────────────────────
Loads all tables from live_trades.db and returns a LiveTradeData dataclass.

Tables loaded
─────────────
  trades            — per-trade records with PnL and BH annotations
  equity_snapshots  — time-series equity snapshots
  positions         — current / historical positions
  regime_log        — per-bar regime feature log (BH mass, garch_vol, etc.)

Equity normalisation
────────────────────
A normalised pd.Series equity_series is derived from equity_snapshots,
falling back to trade-level reconstruction if snapshots are unavailable.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import LIVE_TRADES_DB
from ..types import LiveTradeData, safe_float

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _open_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _load_table(conn: sqlite3.Connection, table: str, parse_dates: Optional[list] = None) -> Optional[pd.DataFrame]:
    """Load a SQLite table into a DataFrame.  Returns None if table is absent or empty."""
    if not _table_exists(conn, table):
        logger.warning("Table '%s' not found in live_trades.db", table)
        return None
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception as exc:
        logger.error("Failed to read table '%s': %s", table, exc)
        return None
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
    logger.debug("Loaded %d rows from table '%s'", len(df), table)
    return df


# ── trades table ─────────────────────────────────────────────────────────────

def _process_trades(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    df = df.copy()
    # Parse timestamp
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=False)
    # Numeric coercion
    for col in ["qty", "price", "entry_price", "pnl", "bars_held",
                "equity_after", "trade_duration_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Add derived columns if not present
    if "pnl" in df.columns:
        df["is_win"] = df["pnl"] > 0
    if "pnl" in df.columns and "equity_after" in df.columns:
        # Approximate pnl % relative to equity
        prev_equity = df["equity_after"] - df["pnl"]
        prev_equity = prev_equity.replace(0, np.nan)
        df["pnl_pct"] = df["pnl"] / prev_equity
    if "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)
    return df


# ── equity_snapshots table ────────────────────────────────────────────────────

def _process_equity_snapshots(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    df = df.copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=False)
    if "equity" in df.columns:
        df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def _equity_series_from_snapshots(df: pd.DataFrame) -> Optional[pd.Series]:
    """Build a clean pd.Series from the equity_snapshots table."""
    if df is None or df.empty or "equity" not in df.columns or "ts" not in df.columns:
        return None
    s = df.set_index("ts")["equity"].dropna()
    s.index = pd.to_datetime(s.index)
    s.name  = "equity"
    return s.sort_index()


def _equity_series_from_trades(trades: pd.DataFrame, initial_equity: float = 1_000_000.0) -> Optional[pd.Series]:
    """Fallback: reconstruct equity curve from cumulative PnL."""
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return None
    df = trades.dropna(subset=["pnl"]).copy()
    ts_col = "ts" if "ts" in df.columns else None
    if ts_col is None:
        return None
    df = df.sort_values(ts_col)
    equity = initial_equity + df["pnl"].cumsum()
    equity.index = pd.to_datetime(df[ts_col].values)
    equity.name  = "equity"
    return equity


# ── positions table ───────────────────────────────────────────────────────────

def _process_positions(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in ["qty", "avg_entry", "current_price", "unrealized_pnl"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "last_updated" in df.columns:
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    return df


# ── regime_log table ──────────────────────────────────────────────────────────

def _process_regime_log(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    df = df.copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=False)
    numeric_cols = [
        "d_bh_mass", "h_bh_mass", "m15_bh_mass",
        "d_bh_active", "h_bh_active", "m15_bh_active",
        "tf_score", "delta_score", "atr", "garch_vol",
        "ou_zscore", "ou_halflife",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Composite BH signal
    mass_cols = [c for c in ["d_bh_mass", "h_bh_mass", "m15_bh_mass"] if c in df.columns]
    if mass_cols:
        df["avg_bh_mass"] = df[mass_cols].mean(axis=1)
    active_cols = [c for c in ["d_bh_active", "h_bh_active", "m15_bh_active"] if c in df.columns]
    if active_cols:
        df["bh_active_count"] = df[active_cols].sum(axis=1)
    if "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)
    return df


# ── public API ────────────────────────────────────────────────────────────────

def load_live_trades(
    db_path:        Path = LIVE_TRADES_DB,
    initial_equity: float = 1_000_000.0,
) -> LiveTradeData:
    """
    Load all tables from live_trades.db and return a LiveTradeData object.

    Parameters
    ----------
    db_path         : path to live_trades.db
    initial_equity  : starting equity for equity curve reconstruction fallback

    Returns
    -------
    LiveTradeData with all available tables populated.
    """
    if not Path(db_path).exists():
        logger.warning("live_trades.db not found at %s — returning empty LiveTradeData", db_path)
        return LiveTradeData()

    logger.info("Loading live trade data from %s", db_path)
    conn = _open_conn(Path(db_path))

    try:
        raw_trades      = _load_table(conn, "trades")
        raw_snapshots   = _load_table(conn, "equity_snapshots")
        raw_positions   = _load_table(conn, "positions")
        raw_regime_log  = _load_table(conn, "regime_log")
    finally:
        conn.close()

    # Process each table
    trades      = _process_trades(raw_trades)
    snapshots   = _process_equity_snapshots(raw_snapshots)
    positions   = _process_positions(raw_positions)
    regime_log  = _process_regime_log(raw_regime_log)

    # Build equity series
    equity_series: Optional[pd.Series] = None
    if snapshots is not None and not snapshots.empty:
        equity_series = _equity_series_from_snapshots(snapshots)
    if equity_series is None and trades is not None:
        equity_series = _equity_series_from_trades(trades, initial_equity=initial_equity)

    n_trades    = len(trades) if trades is not None else 0
    n_regimes   = len(regime_log) if regime_log is not None else 0
    eq_len      = len(equity_series) if equity_series is not None else 0
    logger.info(
        "LiveTradeData loaded: %d trades | %d regime rows | %d equity points",
        n_trades, n_regimes, eq_len,
    )

    return LiveTradeData(
        trades           = trades,
        equity_snapshots = snapshots,
        positions        = positions,
        regime_log       = regime_log,
        equity_series    = equity_series,
    )
