"""
analytics.py — importable DuckDB + Polars analytics module for SRFM lab.

Usage:
    from tools.analytics import get_db, query_to_polars, profile_convergence_edge
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl

# ---------------------------------------------------------------------------
# Paths — resolve relative to repo root (parent of this file's directory)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent

_SOURCES: dict[str, str] = {
    "regimes":    str(_REPO / "results" / "regimes_ES.csv"),
    "wells_es":   str(_REPO / "results" / "wells_ES.csv"),
    "trades":     str(_REPO / "results" / "survey" / "convergence_pnl.csv"),
    "qc_wells":   str(_REPO / "research" / "trade_analysis_data.json"),
    "experiments":str(_REPO / "results" / "v2_experiments.json"),
    "tournament": str(_REPO / "results" / "tournament" / "leaderboard.csv"),
    "ohlcv":      str(_REPO / "data" / "NDX_hourly_poly.csv"),
}

# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------
_connection: Optional[duckdb.DuckDBPyConnection] = None


def get_db(force_new: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a configured DuckDB in-memory connection with all views registered."""
    global _connection
    if _connection is not None and not force_new:
        return _connection

    con = duckdb.connect()
    registered: list[str] = []
    skipped: list[str] = []

    for table, path in _SOURCES.items():
        if not os.path.exists(path):
            skipped.append(f"{table} ({path})")
            continue
        try:
            _register_view(con, table, path)
            registered.append(table)
        except Exception as exc:
            skipped.append(f"{table} ({exc})")

    if skipped:
        warnings.warn(
            f"[analytics] Skipped tables (source not found): {', '.join(skipped)}",
            stacklevel=2,
        )

    _connection = con
    return con


def _register_view(con: duckdb.DuckDBPyConnection, table: str, path: str) -> None:
    """Register a single view in DuckDB, handling CSV/JSON differences."""
    escaped = path.replace("\\", "/")

    if table == "qc_wells":
        # Extract the 'wells' array from trade_analysis_data.json
        con.execute(f"""
            CREATE OR REPLACE VIEW qc_wells AS
            SELECT w.*
            FROM (
                SELECT unnest(wells) AS w
                FROM read_json_auto('{escaped}', format='auto')
            )
        """)
    elif table == "experiments":
        # JSON array at top level
        con.execute(f"""
            CREATE OR REPLACE VIEW experiments AS
            SELECT * FROM read_json_auto('{escaped}', format='array')
        """)
    elif path.endswith(".json"):
        con.execute(f"""
            CREATE OR REPLACE VIEW {table} AS
            SELECT * FROM read_json_auto('{escaped}', format='auto')
        """)
    else:
        con.execute(f"""
            CREATE OR REPLACE VIEW {table} AS
            SELECT * FROM read_csv_auto('{escaped}')
        """)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def query_to_polars(sql: str) -> pl.DataFrame:
    """Run SQL against the registered views and return a Polars DataFrame."""
    con = get_db()
    return con.execute(sql).pl()


def profile_convergence_edge() -> pl.DataFrame:
    """
    Compute convergence edge stats grouped by bh_count bucket.
    Returns a Polars DataFrame with columns:
        bh_bucket, trade_count, win_rate, avg_pnl, total_pnl
    """
    con = get_db()
    if "trades" not in _available_tables(con):
        raise RuntimeError("Table 'trades' is not registered. Source file missing.")

    df = con.execute("SELECT bh_count, pnl FROM trades").pl()

    result = (
        df.with_columns(
            pl.when(pl.col("bh_count") == 0)
            .then(pl.lit("0 BH"))
            .when(pl.col("bh_count") == 1)
            .then(pl.lit("1 BH"))
            .otherwise(pl.lit("2+ BH"))
            .alias("bh_bucket")
        )
        .group_by("bh_bucket")
        .agg([
            pl.len().alias("trade_count"),
            (pl.col("pnl") > 0).mean().alias("win_rate"),
            pl.col("pnl").mean().alias("avg_pnl"),
            pl.col("pnl").sum().alias("total_pnl"),
        ])
        .sort("bh_bucket")
    )
    return result


def compute_beta_series(csv_path: str, cf: float = 0.005) -> pl.DataFrame:
    """
    Compute beta, bit (TIMELIKE/SPACELIKE), and bh_mass incrementally from OHLCV CSV.

    Returns Polars DataFrame with columns:
        date, close, delta, beta, bit, bh_mass
    """
    df = pl.scan_csv(csv_path).select(["date", "close"]).collect()

    deltas = df["close"].diff().fill_null(0)
    betas = (deltas / df["close"]).abs()

    bits = pl.Series([
        "TIMELIKE" if b < cf else "SPACELIKE"
        for b in betas.to_list()
    ])

    # Accumulate bh_mass: mass grows when SPACELIKE, resets on TIMELIKE
    masses: list[float] = []
    mass = 0.0
    for i, bit_val in enumerate(bits.to_list()):
        if bit_val == "SPACELIKE":
            mass += float(betas[i])
        else:
            mass = 0.0
        masses.append(mass)

    return df.with_columns([
        deltas.alias("delta"),
        betas.alias("beta"),
        bits.alias("bit"),
        pl.Series(masses).alias("bh_mass"),
    ])


def _available_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    """List all views/tables currently registered in the connection."""
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    return [r[0] for r in rows]


def available_tables() -> list[str]:
    """Public helper: list registered table names."""
    return _available_tables(get_db())
