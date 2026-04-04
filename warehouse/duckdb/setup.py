"""
setup.py — DuckDB Analytics Setup for SRFM Lab
==============================================
Initializes the DuckDB analytical database from parquet exports.
Registers UDFs for BH mass calculation.
Pre-computes common aggregations and exports summary reports.

Usage:
    python warehouse/duckdb/setup.py [--db path/to/db] [--data-dir path/to/data]
    python warehouse/duckdb/setup.py --refresh-views
    python warehouse/duckdb/setup.py --export-reports
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT     = Path(__file__).parent.parent.parent
_DATA     = _ROOT / "data"
_REPORTS  = _ROOT / "spacetime" / "reports"
_SQL      = Path(__file__).parent / "analytics.sql"
_DEFAULT_DB = _ROOT / "srfm_analytics.duckdb"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger("duckdb.setup")

# ---------------------------------------------------------------------------
# Instrument CF lookup (mirrors live_trader_alpaca.py and bh_engine.py)
# ---------------------------------------------------------------------------
INSTRUMENT_CF: dict[str, dict[str, float]] = {
    "ES":     {"cf_15m": 0.00030, "cf_1h": 0.00100, "cf_1d": 0.00500, "bh_form": 1.5, "decay": 0.95},
    "NQ":     {"cf_15m": 0.00040, "cf_1h": 0.00120, "cf_1d": 0.00600, "bh_form": 1.5, "decay": 0.95},
    "YM":     {"cf_15m": 0.00025, "cf_1h": 0.00080, "cf_1d": 0.00400, "bh_form": 1.5, "decay": 0.95},
    "CL":     {"cf_15m": 0.00150, "cf_1h": 0.00400, "cf_1d": 0.01500, "bh_form": 1.8, "decay": 0.95},
    "GC":     {"cf_15m": 0.00080, "cf_1h": 0.00250, "cf_1d": 0.00800, "bh_form": 1.5, "decay": 0.95},
    "ZB":     {"cf_15m": 0.00050, "cf_1h": 0.00150, "cf_1d": 0.00500, "bh_form": 1.5, "decay": 0.95},
    "NG":     {"cf_15m": 0.00200, "cf_1h": 0.00600, "cf_1d": 0.02000, "bh_form": 1.8, "decay": 0.92},
    "VX":     {"cf_15m": 0.00300, "cf_1h": 0.00800, "cf_1d": 0.02500, "bh_form": 1.8, "decay": 0.92},
    "BTC":    {"cf_15m": 0.00500, "cf_1h": 0.01500, "cf_1d": 0.05000, "bh_form": 1.5, "decay": 0.95},
    "ETH":    {"cf_15m": 0.00700, "cf_1h": 0.02000, "cf_1d": 0.07000, "bh_form": 1.5, "decay": 0.95},
    "SOL":    {"cf_15m": 0.01000, "cf_1h": 0.03000, "cf_1d": 0.10000, "bh_form": 1.5, "decay": 0.95},
    "DEFAULT":{"cf_15m": 0.01000, "cf_1h": 0.03000, "cf_1d": 0.01000, "bh_form": 1.5, "decay": 0.95},
}


# ---------------------------------------------------------------------------
# UDF: BH mass update (single bar)
# ---------------------------------------------------------------------------
def bh_beta(log_return: float, cf: float) -> float:
    """Relativistic beta = |Δp/p| / CF, capped at 1-epsilon."""
    if cf <= 0:
        return 0.0
    return min(abs(log_return) / cf, 1.0 - 1e-9)


def bh_gamma(beta: float) -> float:
    """Lorentz factor: 1 / sqrt(1 - beta^2)."""
    return 1.0 / math.sqrt(max(1.0 - beta**2, 1e-12))


def bh_mass_update(current_mass: float, log_return: float, cf: float, decay: float) -> float:
    """Apply one bar of BH mass accumulation.
    mass_{t+1} = decay * mass_t + (gamma - 1)
    """
    beta  = bh_beta(log_return, cf)
    gamma = bh_gamma(beta)
    return min(decay * current_mass + (gamma - 1.0), 20.0)


def compute_bh_series(
    log_returns: list[float],
    cf:          float,
    decay:       float = 0.95,
    bh_form:     float = 1.5,
) -> dict[str, list]:
    """Compute full BH mass timeseries from a list of log returns.
    Returns dict of {mass, beta, gamma, active, direction}.
    """
    mass_list   = []
    beta_list   = []
    gamma_list  = []
    active_list = []
    dir_list    = []

    mass = 0.0
    for lr in log_returns:
        beta  = bh_beta(lr, cf)
        gamma = bh_gamma(beta)
        mass  = min(decay * mass + (gamma - 1.0), 20.0)
        active = mass >= bh_form
        direction = 1 if lr > 0 else -1 if lr < 0 else 0

        mass_list.append(mass)
        beta_list.append(beta)
        gamma_list.append(gamma)
        active_list.append(active)
        dir_list.append(direction if active else 0)

    return {
        "mass":      mass_list,
        "beta":      beta_list,
        "gamma":     gamma_list,
        "active":    active_list,
        "direction": dir_list,
    }


# ---------------------------------------------------------------------------
# DuckDB UDF wrappers (scalar functions)
# ---------------------------------------------------------------------------
def udf_bh_mass_update(current_mass: float, log_return: float,
                        cf: float, decay: float) -> float:
    return bh_mass_update(current_mass, log_return, cf, decay)


def udf_bh_beta(log_return: float, cf: float) -> float:
    return bh_beta(log_return, cf)


def udf_bh_gamma(beta: float) -> float:
    return bh_gamma(beta)


def udf_tf_score(active_15m: bool, active_1h: bool, active_1d: bool,
                  dir_15m: int, dir_1h: int, dir_1d: int) -> int:
    """Compute tf_score: 0-7 activation + direction consensus."""
    score = 0
    active_count = 0
    dir_sum = 0

    if active_15m:
        score += 1
        active_count += 1
        dir_sum += (dir_15m or 0)
    if active_1h:
        score += 2
        active_count += 1
        dir_sum += (dir_1h or 0)
    if active_1d:
        score += 4
        active_count += 1
        dir_sum += (dir_1d or 0)

    # Directional bonus: all active TFs agree
    if active_count > 0 and abs(dir_sum) == active_count:
        score = min(score + 1, 7)

    return score


def udf_kelly(win_rate: float, edge_ratio: float) -> float:
    """Full Kelly fraction: p - (1-p)/b."""
    if edge_ratio <= 0 or win_rate <= 0:
        return 0.0
    return win_rate - (1.0 - win_rate) / edge_ratio


def udf_sharpe_from_returns(returns_json: str) -> float:
    """Compute annualized Sharpe from JSON array of daily returns."""
    import json
    try:
        rets = np.array(json.loads(returns_json), dtype=float)
        if len(rets) < 5:
            return 0.0
        std = rets.std()
        if std == 0:
            return 0.0
        return float(rets.mean() / std * math.sqrt(252))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Setup function
# ---------------------------------------------------------------------------
def setup(
    db_path:  Path = _DEFAULT_DB,
    data_dir: Path = _DATA,
    run_sql:  bool = True,
) -> duckdb.DuckDBPyConnection:
    """Create and configure the DuckDB analytical database."""
    log.info(f"Opening DuckDB at: {db_path}")
    con = duckdb.connect(str(db_path))

    # ---------------------------------------------------------------------------
    # Register UDFs
    # ---------------------------------------------------------------------------
    log.info("Registering UDFs...")
    con.create_function("bh_mass_update", udf_bh_mass_update,
                         [float, float, float, float], float)
    con.create_function("bh_beta",        udf_bh_beta,
                         [float, float], float)
    con.create_function("bh_gamma",       udf_bh_gamma,
                         [float], float)
    con.create_function("tf_score",       udf_tf_score,
                         [bool, bool, bool, int, int, int], int)
    con.create_function("kelly",          udf_kelly,
                         [float, float], float)

    # ---------------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------------
    con.execute("SET threads = 8")
    con.execute("SET memory_limit = '8GB'")

    # ---------------------------------------------------------------------------
    # Load parquet data (if it exists)
    # ---------------------------------------------------------------------------
    _load_parquet_tables(con, data_dir)

    # ---------------------------------------------------------------------------
    # Run analytics SQL
    # ---------------------------------------------------------------------------
    if run_sql and _SQL.exists():
        log.info(f"Running analytics SQL: {_SQL}")
        try:
            # Filter out lines that reference non-existent parquet files
            sql = _SQL.read_text()
            # Run without COPY statements (they need actual parquet files)
            statements = [s.strip() for s in sql.split(";") if s.strip()
                          and not s.strip().upper().startswith("COPY")
                          and not s.strip().upper().startswith("PRAGMA")
                          and not s.strip().upper().startswith("SUMMARIZE")]
            for stmt in statements:
                try:
                    con.execute(stmt)
                except Exception as e:
                    if "does not exist" in str(e) or "no such" in str(e).lower():
                        log.debug(f"Skipped (table not loaded): {str(e)[:80]}")
                    else:
                        log.warning(f"SQL warning: {str(e)[:120]}")
        except Exception as e:
            log.error(f"Failed running analytics.sql: {e}")
    else:
        log.info("Registering analytical views inline...")
        _register_views(con)

    log.info("DuckDB setup complete.")
    return con


def _load_parquet_tables(con: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    """Load bar and trade data from parquet files if they exist."""
    tables = {
        "bars_1d":  data_dir / "bars" / "1d",
        "bars_1h":  data_dir / "bars" / "1h",
        "bars_15m": data_dir / "bars" / "15m",
        "trades":   data_dir / "trades",
    }

    for table_name, parquet_path in tables.items():
        glob_path = str(parquet_path / "**" / "*.parquet")
        try:
            # Check if any parquet files exist
            result = con.execute(
                f"SELECT COUNT(*) FROM glob('{glob_path}')"
            ).fetchone()[0]

            if result > 0:
                log.info(f"  Loading {table_name} from {parquet_path}...")
                con.execute(f"""
                    CREATE OR REPLACE TABLE {table_name} AS
                    SELECT * FROM read_parquet('{glob_path}', union_by_name=true)
                """)
                count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                log.info(f"    Loaded {count:,} rows into {table_name}")
            else:
                log.debug(f"  No parquet files found for {table_name}, skipping.")
                # Create empty tables with expected schema
                _create_empty_table(con, table_name)
        except Exception as e:
            log.warning(f"  Could not load {table_name}: {e}")
            _create_empty_table(con, table_name)


def _create_empty_table(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """Create empty tables with correct schemas for views to reference."""
    schemas = {
        "bars_1d": """
            CREATE TABLE IF NOT EXISTS bars_1d (
                symbol VARCHAR, timestamp TIMESTAMPTZ,
                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
                volume DOUBLE, vwap DOUBLE, log_return DOUBLE, hl_range DOUBLE
            )""",
        "bars_1h": """
            CREATE TABLE IF NOT EXISTS bars_1h (
                symbol VARCHAR, timestamp TIMESTAMPTZ,
                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
                volume DOUBLE, vwap DOUBLE, log_return DOUBLE, hl_range DOUBLE
            )""",
        "bars_15m": """
            CREATE TABLE IF NOT EXISTS bars_15m (
                symbol VARCHAR, timestamp TIMESTAMPTZ,
                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
                volume DOUBLE, log_return DOUBLE, hl_range DOUBLE
            )""",
        "trades": """
            CREATE TABLE IF NOT EXISTS trades (
                id BIGINT, run_id INTEGER, symbol VARCHAR, side VARCHAR,
                entry_time TIMESTAMPTZ, exit_time TIMESTAMPTZ,
                entry_price DOUBLE, exit_price DOUBLE,
                qty DOUBLE, pnl_dollar DOUBLE, pnl_pct DOUBLE,
                hold_bars INTEGER, mfe_pct DOUBLE, mae_pct DOUBLE,
                tf_score INTEGER, regime_at_entry VARCHAR,
                bh_mass_1d_at_entry DOUBLE, is_winner BOOLEAN
            )""",
        "equity_snapshots": """
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                run_id INTEGER, snapshot_date DATE,
                equity DOUBLE, cash DOUBLE, day_pnl DOUBLE,
                day_return DOUBLE, drawdown_pct DOUBLE, hwm DOUBLE
            )""",
    }
    if table_name in schemas:
        try:
            con.execute(schemas[table_name])
        except Exception:
            pass


def _register_views(con: duckdb.DuckDBPyConnection) -> None:
    """Register key analytical views without loading parquet."""
    views = {
        "rolling_sharpe": """
            CREATE OR REPLACE VIEW rolling_sharpe AS
            SELECT
                symbol, timestamp, close, log_return,
                AVG(log_return) OVER w20
                    / NULLIF(STDDEV(log_return) OVER w20, 0) * SQRT(252) AS sharpe_20d,
                STDDEV(log_return) OVER w20 * SQRT(252)                  AS vol_20d,
                1 - close / MAX(close) OVER w252                          AS rolling_dd_252d
            FROM bars_1d
            WINDOW
                w20  AS (PARTITION BY symbol ORDER BY timestamp ROWS 19  PRECEDING),
                w252 AS (PARTITION BY symbol ORDER BY timestamp ROWS 251 PRECEDING)
        """,
        "equity_curve_stats": """
            CREATE OR REPLACE VIEW equity_curve_stats AS
            SELECT
                run_id,
                COUNT(*)                                        AS n_days,
                ROUND(AVG(day_return) * 252 * 100, 2)          AS ann_return_pct,
                ROUND(STDDEV(day_return) * SQRT(252) * 100, 2) AS ann_vol_pct,
                ROUND(AVG(day_return) / NULLIF(STDDEV(day_return),0) * SQRT(252), 3) AS sharpe,
                ROUND(MIN(drawdown_pct) * 100, 2)              AS max_dd_pct
            FROM equity_snapshots
            WHERE day_return IS NOT NULL
            GROUP BY run_id
        """,
    }
    for name, sql in views.items():
        try:
            con.execute(sql)
            log.debug(f"  Registered view: {name}")
        except Exception as e:
            log.warning(f"  Could not register {name}: {e}")


# ---------------------------------------------------------------------------
# BH mass computation: run against loaded bar data and export
# ---------------------------------------------------------------------------
def compute_bh_states(con: duckdb.DuckDBPyConnection, symbol: str,
                       timeframe: str = "1d") -> pd.DataFrame:
    """Compute BH mass timeseries for a symbol using the Python BH engine."""
    table = f"bars_{timeframe}"
    cfg = INSTRUMENT_CF.get(symbol, INSTRUMENT_CF["DEFAULT"])
    cf    = cfg[f"cf_{timeframe}"]
    decay = cfg["decay"]

    try:
        df = con.execute(f"""
            SELECT timestamp, close, log_return
            FROM {table}
            WHERE symbol = '{symbol}'
              AND log_return IS NOT NULL
            ORDER BY timestamp
        """).df()
    except Exception as e:
        log.warning(f"Could not fetch {symbol} from {table}: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    bh = compute_bh_series(df["log_return"].tolist(), cf=cf, decay=decay,
                            bh_form=cfg["bh_form"])
    for col, vals in bh.items():
        df[col] = vals

    df["symbol"]    = symbol
    df["timeframe"] = timeframe
    df["cf"]        = cf
    return df


def export_bh_states(con: duckdb.DuckDBPyConnection,
                      output_dir: Path = _DATA / "bh_state") -> None:
    """Compute and export BH states for all instruments to parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        symbols = con.execute("SELECT DISTINCT symbol FROM bars_1d").fetchdf()["symbol"].tolist()
    except Exception:
        symbols = list(INSTRUMENT_CF.keys())
        symbols.remove("DEFAULT")

    for symbol in symbols:
        for tf in ("1d", "1h", "15m"):
            df = compute_bh_states(con, symbol, tf)
            if df.empty:
                continue
            out = output_dir / tf / f"{symbol}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False, compression="zstd")
            log.info(f"  Exported BH state: {symbol} {tf} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def generate_reports(con: duckdb.DuckDBPyConnection,
                      output_dir: Path = _REPORTS) -> None:
    """Export CSV reports for the dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report_queries = {
        "equity_curve_stats.csv": "SELECT * FROM equity_curve_stats ORDER BY sharpe DESC",
        "return_correlations.csv": "SELECT * FROM return_correlation_matrix ORDER BY ABS(correlation) DESC",
        "momentum_ranking.csv": "SELECT * FROM momentum_ranking",
        "drawdown_durations.csv": "SELECT * FROM drawdown_durations ORDER BY trough_dd_pct ASC",
    }

    for filename, query in report_queries.items():
        try:
            df = con.execute(query).df()
            df.to_csv(output_dir / filename, index=False)
            log.info(f"  Wrote {filename}: {len(df)} rows")
        except Exception as e:
            log.warning(f"  Could not generate {filename}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="SRFM Lab DuckDB setup")
    parser.add_argument("--db",          type=Path, default=_DEFAULT_DB,
                        help="Path to DuckDB file")
    parser.add_argument("--data-dir",    type=Path, default=_DATA,
                        help="Root data directory containing parquet files")
    parser.add_argument("--no-sql",      action="store_true",
                        help="Skip running analytics.sql (register views only)")
    parser.add_argument("--export-bh",   action="store_true",
                        help="Compute and export BH state timeseries")
    parser.add_argument("--reports",     action="store_true",
                        help="Generate CSV reports")
    parser.add_argument("--symbol",      type=str, default=None,
                        help="Single symbol for BH export (default: all)")
    args = parser.parse_args()

    con = setup(
        db_path  = args.db,
        data_dir = args.data_dir,
        run_sql  = not args.no_sql,
    )

    if args.export_bh:
        if args.symbol:
            df = compute_bh_states(con, args.symbol.upper())
            if not df.empty:
                print(df.tail(20).to_string())
        else:
            export_bh_states(con)

    if args.reports:
        generate_reports(con)

    # Always show a summary
    log.info("\n=== DuckDB Database Summary ===")
    try:
        tables = con.execute("SHOW TABLES").fetchdf()
        for _, row in tables.iterrows():
            name = row["name"]
            try:
                count = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                log.info(f"  {name:<35} {count:>12,} rows")
            except Exception:
                log.info(f"  {name:<35}  (view)")
    except Exception:
        pass

    con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
