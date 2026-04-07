# execution/tca/tca_store.py -- SQLite persistence layer for TCA results in SRFM
# Stores TCAResult objects with full query, aggregation, and export support.

from __future__ import annotations

import csv
import io
import math
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

_CREATE_TCA_RESULTS = """
CREATE TABLE IF NOT EXISTS tca_results (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id                    TEXT UNIQUE,
    symbol                      TEXT NOT NULL,
    side                        TEXT,
    strategy                    TEXT,
    venue                       TEXT,
    order_type                  TEXT,
    trade_date                  TEXT,

    -- Cost metrics (bps)
    implementation_shortfall_bps REAL,
    market_impact_bps           REAL,
    timing_cost_bps             REAL,
    spread_cost_bps             REAL,
    total_cost_bps              REAL,
    vwap_slippage_bps           REAL,
    twap_slippage_bps           REAL,
    close_slippage_bps          REAL,

    -- Reference prices
    decision_price              REAL,
    arrival_price               REAL,
    fill_price                  REAL,
    benchmark_type              TEXT,

    -- Participation / fill
    participation_rate          REAL,
    fill_rate                   REAL,
    time_to_fill_ms             REAL,

    -- Slippage decomposition
    decomp_spread               REAL,
    decomp_impact               REAL,
    decomp_timing               REAL,
    decomp_alpha                REAL,
    decomp_total                REAL,

    -- Metadata
    inserted_at                 TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_IDX_DATE = "CREATE INDEX IF NOT EXISTS idx_tca_date ON tca_results (trade_date);"
_CREATE_IDX_SYMBOL = "CREATE INDEX IF NOT EXISTS idx_tca_symbol ON tca_results (symbol);"
_CREATE_IDX_STRATEGY = "CREATE INDEX IF NOT EXISTS idx_tca_strategy ON tca_results (strategy);"
_CREATE_IDX_VENUE = "CREATE INDEX IF NOT EXISTS idx_tca_venue ON tca_results (venue);"
_CREATE_IDX_TRADE_ID = "CREATE INDEX IF NOT EXISTS idx_tca_trade_id ON tca_results (trade_id);"

_UPSERT_RESULT = """
INSERT INTO tca_results
    (trade_id, symbol, side, strategy, venue, order_type, trade_date,
     implementation_shortfall_bps, market_impact_bps, timing_cost_bps,
     spread_cost_bps, total_cost_bps, vwap_slippage_bps, twap_slippage_bps,
     close_slippage_bps, decision_price, arrival_price, fill_price,
     benchmark_type, participation_rate, fill_rate, time_to_fill_ms,
     decomp_spread, decomp_impact, decomp_timing, decomp_alpha, decomp_total)
VALUES
    (?, ?, ?, ?, ?, ?, ?,
     ?, ?, ?,
     ?, ?, ?, ?, ?,
     ?, ?, ?,
     ?, ?, ?, ?,
     ?, ?, ?, ?, ?)
ON CONFLICT(trade_id) DO UPDATE SET
    implementation_shortfall_bps = excluded.implementation_shortfall_bps,
    market_impact_bps            = excluded.market_impact_bps,
    timing_cost_bps              = excluded.timing_cost_bps,
    spread_cost_bps              = excluded.spread_cost_bps,
    total_cost_bps               = excluded.total_cost_bps,
    vwap_slippage_bps            = excluded.vwap_slippage_bps,
    twap_slippage_bps            = excluded.twap_slippage_bps,
    close_slippage_bps           = excluded.close_slippage_bps,
    fill_price                   = excluded.fill_price,
    participation_rate           = excluded.participation_rate,
    fill_rate                    = excluded.fill_rate,
    time_to_fill_ms              = excluded.time_to_fill_ms,
    inserted_at                  = datetime('now')
WHERE trade_id IS NOT NULL AND trade_id != ''
;
"""

# Fallback INSERT without UPSERT for rows with empty/null trade_id
_INSERT_RESULT = """
INSERT INTO tca_results
    (trade_id, symbol, side, strategy, venue, order_type, trade_date,
     implementation_shortfall_bps, market_impact_bps, timing_cost_bps,
     spread_cost_bps, total_cost_bps, vwap_slippage_bps, twap_slippage_bps,
     close_slippage_bps, decision_price, arrival_price, fill_price,
     benchmark_type, participation_rate, fill_rate, time_to_fill_ms,
     decomp_spread, decomp_impact, decomp_timing, decomp_alpha, decomp_total)
VALUES
    (?, ?, ?, ?, ?, ?, ?,
     ?, ?, ?,
     ?, ?, ?, ?, ?,
     ?, ?, ?,
     ?, ?, ?, ?,
     ?, ?, ?, ?, ?)
;
"""


def _result_to_row(trade_id: str, result) -> tuple:
    """Convert a TCAResult and trade_id to a flat row tuple for SQL insertion."""
    decomp = result.slippage_decomposition
    ds = decomp.spread_component if decomp else None
    di = decomp.market_impact_component if decomp else None
    dt = decomp.timing_component if decomp else None
    da = decomp.alpha_component if decomp else None
    dtt = decomp.total_bps if decomp else None

    def _f(v):
        """Coerce to None if not finite, else return float."""
        if v is None:
            return None
        try:
            fv = float(v)
            return fv if math.isfinite(fv) else None
        except (TypeError, ValueError):
            return None

    return (
        trade_id or None,
        result.symbol,
        result.side,
        result.strategy,
        result.venue,
        result.order_type,
        result.trade_date,
        _f(result.implementation_shortfall_bps),
        _f(result.market_impact_bps),
        _f(result.timing_cost_bps),
        _f(result.spread_cost_bps),
        _f(result.total_cost_bps),
        _f(result.vwap_slippage_bps),
        _f(result.twap_slippage_bps),
        _f(result.close_slippage_bps),
        _f(result.decision_price),
        _f(result.arrival_price),
        _f(result.fill_price),
        result.benchmark_type,
        _f(result.participation_rate),
        _f(result.fill_rate),
        _f(result.time_to_fill_ms),
        _f(ds),
        _f(di),
        _f(dt),
        _f(da),
        _f(dtt),
    )


def _row_to_result(row: sqlite3.Row):
    """Reconstruct a lightweight TCAResult-like dict from a DB row."""
    # We return a plain dict rather than a TCAResult instance to avoid
    # import cycles. Callers that need TCAResult objects can reconstruct them.
    from .tca_engine import TCAResult, SlippageDecomposition
    decomp = None
    if row["decomp_total"] is not None:
        decomp = SlippageDecomposition(
            spread_component=row["decomp_spread"] or 0.0,
            market_impact_component=row["decomp_impact"] or 0.0,
            timing_component=row["decomp_timing"] or 0.0,
            alpha_component=row["decomp_alpha"] or 0.0,
            total_bps=row["decomp_total"] or 0.0,
        )
    return TCAResult(
        implementation_shortfall_bps=row["implementation_shortfall_bps"] or 0.0,
        market_impact_bps=row["market_impact_bps"] or 0.0,
        timing_cost_bps=row["timing_cost_bps"] or 0.0,
        spread_cost_bps=row["spread_cost_bps"] or 0.0,
        total_cost_bps=row["total_cost_bps"] or 0.0,
        participation_rate=row["participation_rate"] or 0.0,
        vwap_slippage_bps=row["vwap_slippage_bps"] or 0.0,
        twap_slippage_bps=row["twap_slippage_bps"] or 0.0,
        close_slippage_bps=row["close_slippage_bps"] or 0.0,
        decision_price=row["decision_price"] or 0.0,
        arrival_price=row["arrival_price"] or 0.0,
        fill_price=row["fill_price"] or 0.0,
        benchmark_type=row["benchmark_type"] or "ARRIVAL",
        fill_rate=row["fill_rate"] or 0.0,
        time_to_fill_ms=row["time_to_fill_ms"] or 0.0,
        slippage_decomposition=decomp,
        symbol=row["symbol"] or "",
        side=row["side"] or "",
        strategy=row["strategy"] or "",
        venue=row["venue"] or "",
        order_type=row["order_type"] or "",
        trade_id=row["trade_id"],
        trade_date=row["trade_date"],
    )


# ---------------------------------------------------------------------------
# TCA Store
# ---------------------------------------------------------------------------

class TCAStore:
    """
    SQLite-backed persistent store for TCA results.

    All writes use upsert semantics keyed on trade_id to prevent duplicates.
    Rows with empty/null trade_id are always inserted (no deduplication).
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        # For :memory: databases keep a single persistent connection -- SQLite
        # in-memory databases are destroyed when all connections to them close.
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing a connection with row_factory set."""
        if self._persistent_conn is not None:
            # Return the persistent in-memory connection -- do not close it
            yield self._persistent_conn
            return
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent read performance on file DBs
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables and indices if they do not exist."""
        with self._conn() as conn:
            conn.execute(_CREATE_TCA_RESULTS)
            conn.execute(_CREATE_IDX_DATE)
            conn.execute(_CREATE_IDX_SYMBOL)
            conn.execute(_CREATE_IDX_STRATEGY)
            conn.execute(_CREATE_IDX_VENUE)
            conn.execute(_CREATE_IDX_TRADE_ID)
            conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def insert(self, trade_id: str, result) -> None:
        """
        Insert or update a TCAResult record.

        If trade_id is non-empty, performs an upsert (update on conflict).
        If trade_id is empty or None, always inserts a new row.
        """
        row = _result_to_row(trade_id, result)
        with self._conn() as conn:
            if trade_id:
                conn.execute(_UPSERT_RESULT, row)
            else:
                conn.execute(_INSERT_RESULT, row)
            conn.commit()

    def insert_batch(self, records: List[tuple]) -> int:
        """
        Bulk insert a list of (trade_id, TCAResult) tuples.
        Returns number of rows inserted.
        """
        rows = [_result_to_row(tid, r) for tid, r in records]
        with self._conn() as conn:
            for row in rows:
                trade_id = row[0]
                if trade_id:
                    conn.execute(_UPSERT_RESULT, row)
                else:
                    conn.execute(_INSERT_RESULT, row)
            conn.commit()
        return len(rows)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        symbol: Optional[str] = None,
        date_from: Optional[str] = None,    # YYYY-MM-DD inclusive
        date_to: Optional[str] = None,      # YYYY-MM-DD inclusive
        strategy: Optional[str] = None,
        venue: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: str = "trade_date DESC, id DESC",
    ) -> List:
        """
        Query TCA results with optional filters.

        Parameters
        ----------
        symbol    : exact symbol match (case-sensitive)
        date_from : start date string YYYY-MM-DD
        date_to   : end date string YYYY-MM-DD
        strategy  : exact strategy match
        venue     : exact venue match
        limit     : max rows to return
        order_by  : SQL ORDER BY clause

        Returns
        -------
        List[TCAResult]
        """
        conditions = []
        params: List[Any] = []

        if symbol is not None:
            conditions.append("symbol = ?")
            params.append(symbol)
        if date_from is not None:
            conditions.append("trade_date >= ?")
            params.append(date_from)
        if date_to is not None:
            conditions.append("trade_date <= ?")
            params.append(date_to)
        if strategy is not None:
            conditions.append("strategy = ?")
            params.append(strategy)
        if venue is not None:
            conditions.append("venue = ?")
            params.append(venue)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        sql = f"SELECT * FROM tca_results {where} ORDER BY {order_by} {limit_clause};"

        with self._conn() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        return [_row_to_result(row) for row in rows]

    def get_by_trade_id(self, trade_id: str):
        """Retrieve a single TCAResult by trade_id. Returns None if not found."""
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM tca_results WHERE trade_id = ? ORDER BY id DESC LIMIT 1;",
                (trade_id,),
            )
            row = cursor.fetchone()
        return _row_to_result(row) if row else None

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        group_by: str,
        metric: str = "implementation_shortfall_bps",
    ):
        """
        Group TCA results by a dimension and compute mean/std/count.

        Parameters
        ----------
        group_by : column name -- "symbol", "strategy", "venue", "trade_date"
        metric   : numeric column to aggregate

        Returns
        -------
        pandas DataFrame with columns: group_by, mean_{metric}, std_{metric}, count
        If pandas is not available, returns a list of dicts.
        """
        allowed_groups = {"symbol", "strategy", "venue", "trade_date"}
        allowed_metrics = {
            "implementation_shortfall_bps", "market_impact_bps",
            "timing_cost_bps", "spread_cost_bps", "total_cost_bps",
            "vwap_slippage_bps", "participation_rate", "fill_rate",
        }
        if group_by not in allowed_groups:
            raise ValueError(f"group_by must be one of {allowed_groups}")
        if metric not in allowed_metrics:
            raise ValueError(f"metric must be one of {allowed_metrics}")

        sql = f"""
            SELECT
                {group_by} as group_key,
                AVG({metric}) as mean_val,
                -- SQLite has no STDDEV; compute variance manually
                AVG({metric} * {metric}) - AVG({metric}) * AVG({metric}) as var_val,
                COUNT(*) as n
            FROM tca_results
            WHERE {metric} IS NOT NULL
            GROUP BY {group_by}
            ORDER BY mean_val DESC;
        """
        with self._conn() as conn:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()

        import math as _math
        results_list = []
        for row in rows:
            var = row["var_val"] or 0.0
            std = _math.sqrt(max(var, 0.0))
            results_list.append({
                group_by: row["group_key"],
                f"mean_{metric}": row["mean_val"],
                f"std_{metric}": std,
                "count": row["n"],
            })

        try:
            import pandas as pd
            return pd.DataFrame(results_list)
        except ImportError:
            return results_list

    def aggregate_raw(
        self,
        group_by: str,
        metric: str = "implementation_shortfall_bps",
    ) -> List[Dict]:
        """Aggregation returning plain list of dicts (no pandas dependency)."""
        allowed_groups = {"symbol", "strategy", "venue", "trade_date"}
        if group_by not in allowed_groups:
            raise ValueError(f"group_by must be one of {allowed_groups}")

        sql = f"""
            SELECT
                {group_by} as group_key,
                AVG({metric}) as mean_val,
                COUNT(*) as n
            FROM tca_results
            WHERE {metric} IS NOT NULL
            GROUP BY {group_by}
            ORDER BY mean_val DESC;
        """
        with self._conn() as conn:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()

        return [
            {group_by: r["group_key"], f"mean_{metric}": r["mean_val"], "count": r["n"]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(
        self,
        path: str,
        symbol: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        strategy: Optional[str] = None,
        venue: Optional[str] = None,
    ) -> int:
        """
        Export filtered TCA results to a CSV file.

        Returns the number of data rows written (excluding header).
        """
        results = self.query(
            symbol=symbol,
            date_from=date_from,
            date_to=date_to,
            strategy=strategy,
            venue=venue,
        )

        fieldnames = [
            "trade_id", "symbol", "side", "strategy", "venue", "order_type",
            "trade_date", "implementation_shortfall_bps", "market_impact_bps",
            "timing_cost_bps", "spread_cost_bps", "total_cost_bps",
            "vwap_slippage_bps", "twap_slippage_bps", "close_slippage_bps",
            "decision_price", "arrival_price", "fill_price", "benchmark_type",
            "participation_rate", "fill_rate", "time_to_fill_ms",
        ]

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "trade_id": r.trade_id,
                    "symbol": r.symbol,
                    "side": r.side,
                    "strategy": r.strategy,
                    "venue": r.venue,
                    "order_type": r.order_type,
                    "trade_date": r.trade_date,
                    "implementation_shortfall_bps": r.implementation_shortfall_bps,
                    "market_impact_bps": r.market_impact_bps,
                    "timing_cost_bps": r.timing_cost_bps,
                    "spread_cost_bps": r.spread_cost_bps,
                    "total_cost_bps": r.total_cost_bps,
                    "vwap_slippage_bps": r.vwap_slippage_bps,
                    "twap_slippage_bps": r.twap_slippage_bps,
                    "close_slippage_bps": r.close_slippage_bps,
                    "decision_price": r.decision_price,
                    "arrival_price": r.arrival_price,
                    "fill_price": r.fill_price,
                    "benchmark_type": r.benchmark_type,
                    "participation_rate": r.participation_rate,
                    "fill_rate": r.fill_rate,
                    "time_to_fill_ms": r.time_to_fill_ms,
                })

        return len(results)

    # ------------------------------------------------------------------
    # Daily report
    # ------------------------------------------------------------------

    def daily_report(self, date: str) -> Dict:
        """
        Compute a summary report for a single trading day.

        Parameters
        ----------
        date : YYYY-MM-DD string

        Returns
        -------
        Dict with aggregated metrics for the day
        """
        results = self.query(date_from=date, date_to=date)
        n = len(results)
        if n == 0:
            return {
                "date": date,
                "n_trades": 0,
                "avg_is_bps": None,
                "avg_vwap_slippage_bps": None,
                "avg_spread_cost_bps": None,
                "avg_market_impact_bps": None,
                "avg_participation_rate": None,
                "avg_fill_rate": None,
                "n_partial_fills": 0,
            }

        avg_is = sum(r.implementation_shortfall_bps for r in results) / n
        avg_vwap = sum(r.vwap_slippage_bps for r in results) / n
        avg_spread = sum(r.spread_cost_bps for r in results) / n
        avg_impact = sum(r.market_impact_bps for r in results) / n
        avg_part = sum(r.participation_rate for r in results) / n
        avg_fill = sum(r.fill_rate for r in results) / n
        n_partial = sum(1 for r in results if r.fill_rate < 0.9999)

        # Per-venue breakdown
        venue_map: Dict[str, List[float]] = {}
        for r in results:
            venue_map.setdefault(r.venue or "UNKNOWN", []).append(
                r.implementation_shortfall_bps
            )
        venue_avg = {v: sum(vals) / len(vals) for v, vals in venue_map.items()}
        best_venue = min(venue_avg, key=venue_avg.get) if venue_avg else ""
        worst_venue = max(venue_avg, key=venue_avg.get) if venue_avg else ""

        # Per-strategy breakdown
        strat_map: Dict[str, List[float]] = {}
        for r in results:
            strat_map.setdefault(r.strategy or "UNKNOWN", []).append(
                r.implementation_shortfall_bps
            )
        strategy_avg = {s: sum(v) / len(v) for s, v in strat_map.items()}

        return {
            "date": date,
            "n_trades": n,
            "avg_is_bps": avg_is,
            "avg_vwap_slippage_bps": avg_vwap,
            "avg_spread_cost_bps": avg_spread,
            "avg_market_impact_bps": avg_impact,
            "avg_participation_rate": avg_part,
            "avg_fill_rate": avg_fill,
            "n_partial_fills": n_partial,
            "best_venue": best_venue,
            "worst_venue": worst_venue,
            "venue_avg_is": venue_avg,
            "strategy_avg_is": strategy_avg,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def count(
        self,
        symbol: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> int:
        """Return count of records matching optional filters."""
        conditions = []
        params: List[Any] = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if date_from:
            conditions.append("trade_date >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("trade_date <= ?")
            params.append(date_to)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"SELECT COUNT(*) as cnt FROM tca_results {where};"
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return row["cnt"]

    def delete_before(self, date: str) -> int:
        """Delete records with trade_date < date. Returns rows deleted."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM tca_results WHERE trade_date < ?;", (date,)
            )
            conn.commit()
            return cursor.rowcount

    def vacuum(self) -> None:
        """Run VACUUM to reclaim space after large deletes."""
        with self._conn() as conn:
            conn.execute("VACUUM;")
