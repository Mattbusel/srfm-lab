"""
db/query_engine.py -- High-performance query engine for the SRFM SQLite database.

Provides:
  SRFMDatabase   -- connection pool, WAL mode, thread-local connections
  TradeQueries   -- P&L, drawdown, rolling Sharpe, open positions
  SignalQueries  -- signal history, BH mass, regime, NavLog
  PerformanceQueries -- equity curve, regime breakdown, win rates
  ParameterQueries   -- param history, current params, audit log
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator

import pandas as pd

from .schema import Tables

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SRFMDatabase -- connection management
# ---------------------------------------------------------------------------

class SRFMDatabase:
    """
    Thread-safe SQLite connection pool using thread-local connections.

    WAL mode is set on every connection so readers never block writers.
    All write operations must go through execute() or executemany() with
    explicit transactions -- no autocommit surprises.

    Usage
    -----
    db = SRFMDatabase("/path/to/srfm.db")
    df = db.execute("SELECT * FROM trades WHERE symbol=?", ("AAPL",), fetch="df")
    with db.transaction() as conn:
        conn.execute("INSERT INTO trades(...) VALUES (?,...)", row)
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(Path(db_path).resolve())
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-32000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256 MB
            self._local.conn = conn
            log.debug("Opened SQLite connection for thread %s", threading.get_ident())
        return self._local.conn

    def close(self) -> None:
        """Close this thread's connection."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None

    def close_all(self) -> None:
        """Signal all threads to close; each thread must call close() itself."""
        self.close()

    # ------------------------------------------------------------------
    # Core query methods
    # ------------------------------------------------------------------

    def execute(
        self,
        sql: str,
        params: tuple | list = (),
        fetch: str = "none",
    ) -> Any:
        """
        Execute a parameterized SQL statement.

        fetch options
        -------------
        'none'   -- no return value (INSERT/UPDATE/DELETE)
        'one'    -- returns single sqlite3.Row or None
        'all'    -- returns list[sqlite3.Row]
        'df'     -- returns pandas DataFrame
        'scalar' -- returns first column of first row
        """
        conn = self._get_conn()
        try:
            cur = conn.execute(sql, params)
            if fetch == "none":
                conn.commit()
                return None
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            if fetch == "scalar":
                row = cur.fetchone()
                return row[0] if row else None
            if fetch == "df":
                rows = cur.fetchall()
                if not rows:
                    cols = [d[0] for d in cur.description] if cur.description else []
                    return pd.DataFrame(columns=cols)
                cols = list(rows[0].keys())
                return pd.DataFrame([dict(r) for r in rows], columns=cols)
            raise ValueError(f"Unknown fetch mode: {fetch!r}")
        except sqlite3.Error as exc:
            log.error("SQL error in execute(): %s | sql=%s | params=%s", exc, sql, params)
            raise

    def executemany(self, sql: str, params_list: list[tuple | list]) -> int:
        """
        Batch insert/update.  Returns rowcount.
        All rows are committed in a single transaction.
        """
        if not params_list:
            return 0
        conn = self._get_conn()
        try:
            with conn:
                cur = conn.executemany(sql, params_list)
                return cur.rowcount
        except sqlite3.Error as exc:
            log.error("SQL error in executemany(): %s | sql=%s", exc, sql)
            raise

    @contextlib.contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for atomic operations.

        with db.transaction() as conn:
            conn.execute(...)
            conn.execute(...)
        -- auto-commits on success, rolls back on exception
        """
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def table_exists(self, table: str) -> bool:
        row = self.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
            fetch="one",
        )
        return row is not None

    def row_count(self, table: str) -> int:
        return self.execute(f"SELECT COUNT(*) FROM {table}", fetch="scalar") or 0  # noqa: S608


# ---------------------------------------------------------------------------
# TradeQueries
# ---------------------------------------------------------------------------

class TradeQueries:
    """All trade-related read queries."""

    def __init__(self, db: SRFMDatabase) -> None:
        self._db = db

    def get_trades(
        self,
        since: str | None = None,
        until: str | None = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """
        Return trades as a DataFrame.

        Parameters
        ----------
        since  : ISO-8601 datetime string (inclusive lower bound on entry_time)
        until  : ISO-8601 datetime string (inclusive upper bound on entry_time)
        symbol : filter to a single symbol
        """
        clauses: list[str] = []
        params: list[Any] = []

        if since:
            clauses.append("entry_time >= ?")
            params.append(since)
        if until:
            clauses.append("entry_time <= ?")
            params.append(until)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM {Tables.TRADES} {where} ORDER BY entry_time"  # noqa: S608
        df = self._db.execute(sql, params, fetch="df")
        if not df.empty:
            df["entry_time"] = pd.to_datetime(df["entry_time"])
            if "exit_time" in df.columns:
                df["exit_time"] = pd.to_datetime(df["exit_time"])
        return df

    def get_open_positions(self) -> pd.DataFrame:
        """Return currently open positions (exit_time IS NULL)."""
        sql = f"""
            SELECT t.symbol, t.side, t.qty, t.entry_price, t.entry_time,
                   t.strategy_version, t.signal_name, t.regime
            FROM {Tables.TRADES} t
            WHERE t.exit_time IS NULL
            ORDER BY t.entry_time
        """  # noqa: S608
        return self._db.execute(sql, fetch="df")

    def get_trade_pnl_by_symbol(self) -> dict[str, float]:
        """Return total realized P&L keyed by symbol (closed trades only)."""
        sql = f"""
            SELECT symbol, SUM(pnl) AS total_pnl
            FROM {Tables.TRADES}
            WHERE exit_time IS NOT NULL
            GROUP BY symbol
            ORDER BY total_pnl DESC
        """  # noqa: S608
        rows = self._db.execute(sql, fetch="all")
        return {r["symbol"]: r["total_pnl"] for r in rows}

    def get_daily_returns(self, since: str | None = None) -> pd.Series:
        """
        Return a Series of daily returns indexed by date string.
        Reads from daily_performance if available, otherwise computes
        from individual trades.
        """
        if self._db.table_exists(Tables.DAILY_PERFORMANCE):
            clauses = []
            params: list[Any] = []
            if since:
                clauses.append("date >= ?")
                params.append(since)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            sql = f"SELECT date, daily_return FROM {Tables.DAILY_PERFORMANCE} {where} ORDER BY date"  # noqa: S608
            df = self._db.execute(sql, params, fetch="df")
            if not df.empty:
                return df.set_index("date")["daily_return"]

        # Fallback: compute from trades
        sql = f"""
            SELECT date(exit_time) AS trade_date,
                   SUM(pnl) AS daily_pnl
            FROM {Tables.TRADES}
            WHERE exit_time IS NOT NULL
            {"AND exit_time >= ?" if since else ""}
            GROUP BY trade_date
            ORDER BY trade_date
        """  # noqa: S608
        params = [since] if since else []
        df = self._db.execute(sql, params, fetch="df")
        if df.empty:
            return pd.Series(dtype=float)
        # Convert absolute P&L to return (requires capital -- use proxy)
        return df.set_index("trade_date")["daily_pnl"]

    def get_rolling_sharpe(self, window_days: int = 30) -> pd.Series:
        """
        Compute rolling annualized Sharpe ratio over a sliding window.
        Returns a Series indexed by date.
        """
        returns = self.get_daily_returns()
        if returns.empty or len(returns) < 2:
            return pd.Series(dtype=float)

        returns.index = pd.to_datetime(returns.index)
        returns = returns.sort_index()

        rolling_mean = returns.rolling(window=window_days, min_periods=max(5, window_days // 4)).mean()
        rolling_std  = returns.rolling(window=window_days, min_periods=max(5, window_days // 4)).std()

        # Annualize: assume trading days
        sharpe = (rolling_mean / rolling_std.replace(0, float("nan"))) * (252 ** 0.5)
        sharpe.name = f"rolling_sharpe_{window_days}d"
        return sharpe

    def get_largest_drawdown_period(self) -> dict[str, Any]:
        """
        Return a dict with start, end, depth (as fraction), and duration_days
        for the largest peak-to-trough drawdown observed.
        """
        returns = self.get_daily_returns()
        if returns.empty:
            return {"start": None, "end": None, "depth": 0.0, "duration_days": 0}

        returns.index = pd.to_datetime(returns.index)
        returns = returns.sort_index().fillna(0.0)

        # Build equity curve (cumulative product of 1+r, baseline 1.0)
        eq = (1.0 + returns).cumprod()
        running_max = eq.cummax()
        drawdown = (eq - running_max) / running_max

        min_dd_idx = drawdown.idxmin()
        min_dd_val = float(drawdown.min())

        if min_dd_val == 0.0:
            return {"start": None, "end": None, "depth": 0.0, "duration_days": 0}

        # Find peak before trough
        peak_idx = running_max[:min_dd_idx].idxmax()
        duration = (min_dd_idx - peak_idx).days

        return {
            "start": peak_idx.date().isoformat(),
            "end": min_dd_idx.date().isoformat(),
            "depth": round(min_dd_val, 6),
            "duration_days": duration,
        }

    def get_trades_by_strategy(self, strategy_version: str) -> pd.DataFrame:
        """Return all trades for a given strategy version."""
        sql = f"""
            SELECT * FROM {Tables.TRADES}
            WHERE strategy_version = ?
            ORDER BY entry_time
        """  # noqa: S608
        return self._db.execute(sql, (strategy_version,), fetch="df")

    def get_recent_trades(self, n: int = 50) -> pd.DataFrame:
        """Return the n most recent closed trades."""
        sql = f"""
            SELECT * FROM {Tables.TRADES}
            WHERE exit_time IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT ?
        """  # noqa: S608
        return self._db.execute(sql, (n,), fetch="df")

    def count_trades_today(self) -> int:
        """Return count of trades entered today."""
        today = datetime.now(timezone.utc).date().isoformat()
        return self._db.execute(
            f"SELECT COUNT(*) FROM {Tables.TRADES} WHERE date(entry_time) = ?",  # noqa: S608
            (today,),
            fetch="scalar",
        ) or 0

    def get_win_loss_streak(self) -> dict[str, int]:
        """Return current win streak and loss streak (consecutive closed trades)."""
        sql = f"""
            SELECT pnl FROM {Tables.TRADES}
            WHERE exit_time IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT 200
        """  # noqa: S608
        rows = self._db.execute(sql, fetch="all")
        if not rows:
            return {"current_win_streak": 0, "current_loss_streak": 0}

        pnls = [r["pnl"] for r in rows]
        win_streak = loss_streak = 0
        for p in pnls:
            if p is None:
                break
            if p > 0:
                if loss_streak > 0:
                    break
                win_streak += 1
            else:
                if win_streak > 0:
                    break
                loss_streak += 1

        return {"current_win_streak": win_streak, "current_loss_streak": loss_streak}


# ---------------------------------------------------------------------------
# SignalQueries
# ---------------------------------------------------------------------------

class SignalQueries:
    """Queries for signal history, BH mass, regime log, and nav log."""

    def __init__(self, db: SRFMDatabase) -> None:
        self._db = db

    def get_signal_history(
        self,
        signal_name: str,
        since: str | None = None,
    ) -> pd.DataFrame:
        """Return signal_history rows for a given signal."""
        clauses = ["signal_name = ?"]
        params: list[Any] = [signal_name]
        if since:
            clauses.append("bar_time >= ?")
            params.append(since)

        where = "WHERE " + " AND ".join(clauses)
        sql = f"""
            SELECT *
            FROM {Tables.SIGNAL_HISTORY}
            {where}
            ORDER BY bar_time
        """  # noqa: S608
        df = self._db.execute(sql, params, fetch="df")
        if not df.empty:
            df["bar_time"] = pd.to_datetime(df["bar_time"])
        return df

    def get_bh_mass_history(
        self,
        symbol: str,
        timeframe: str,
        since: str | None = None,
    ) -> pd.DataFrame:
        """
        Return BH mass time series for a symbol/timeframe pair from nav_log.
        Returns columns: bar_time, bh_mass, bh_curvature.
        """
        clauses = ["symbol = ?", "timeframe = ?", "bh_mass IS NOT NULL"]
        params: list[Any] = [symbol, timeframe]
        if since:
            clauses.append("bar_time >= ?")
            params.append(since)

        where = "WHERE " + " AND ".join(clauses)
        sql = f"""
            SELECT bar_time, bh_mass, bh_curvature
            FROM {Tables.NAV_LOG}
            {where}
            ORDER BY bar_time
        """  # noqa: S608
        df = self._db.execute(sql, params, fetch="df")
        if not df.empty:
            df["bar_time"] = pd.to_datetime(df["bar_time"])
        return df

    def get_regime_history(self, since: str | None = None) -> pd.DataFrame:
        """Return regime transition log."""
        clauses: list[str] = []
        params: list[Any] = []
        if since:
            clauses.append("transition_time >= ?")
            params.append(since)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT transition_time, regime, previous_regime,
                   regime_confidence, trend_regime, volatility_regime,
                   vix_level, bh_mass_spx, trigger_type
            FROM {Tables.REGIME_LOG}
            {where}
            ORDER BY transition_time
        """  # noqa: S608
        df = self._db.execute(sql, params, fetch="df")
        if not df.empty:
            df["transition_time"] = pd.to_datetime(df["transition_time"])
        return df

    def get_nav_history(
        self,
        symbol: str,
        since: str | None = None,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """Return nav_log history for a symbol."""
        clauses = ["symbol = ?", "timeframe = ?"]
        params: list[Any] = [symbol, timeframe]
        if since:
            clauses.append("bar_time >= ?")
            params.append(since)

        where = "WHERE " + " AND ".join(clauses)
        sql = f"""
            SELECT bar_time, nav, cash, gross_market_value, net_market_value,
                   realized_pnl_today, unrealized_pnl, total_pnl_today,
                   num_long_positions, num_short_positions,
                   bh_mass, regime, close_price
            FROM {Tables.NAV_LOG}
            {where}
            ORDER BY bar_time
        """  # noqa: S608
        df = self._db.execute(sql, params, fetch="df")
        if not df.empty:
            df["bar_time"] = pd.to_datetime(df["bar_time"])
            df = df.set_index("bar_time")
        return df

    def get_active_signals(self) -> list[str]:
        """Return list of currently active signal names from signal_registry."""
        rows = self._db.execute(
            f"SELECT signal_name FROM {Tables.SIGNAL_REGISTRY} WHERE is_active=1 ORDER BY signal_name",  # noqa: S608
            fetch="all",
        )
        return [r["signal_name"] for r in rows]

    def get_signal_win_rate(self, signal_name: str) -> float | None:
        """Return win_rate_pct for a signal from registry."""
        row = self._db.execute(
            f"SELECT win_rate_pct FROM {Tables.SIGNAL_REGISTRY} WHERE signal_name=?",  # noqa: S608
            (signal_name,),
            fetch="one",
        )
        return row["win_rate_pct"] if row else None

    def get_latest_regime(self) -> dict[str, Any] | None:
        """Return the most recent regime log entry."""
        row = self._db.execute(
            f"""
            SELECT * FROM {Tables.REGIME_LOG}
            ORDER BY transition_time DESC LIMIT 1
            """,  # noqa: S608
            fetch="one",
        )
        return dict(row) if row else None

    def get_signals_fired_today(self) -> pd.DataFrame:
        """Return all signals fired today."""
        today = datetime.now(timezone.utc).date().isoformat()
        sql = f"""
            SELECT signal_name, symbol, timeframe, bar_time,
                   signal_value, direction, confidence, acted_on
            FROM {Tables.SIGNAL_HISTORY}
            WHERE date(bar_time) = ?
            ORDER BY bar_time DESC
        """  # noqa: S608
        return self._db.execute(sql, (today,), fetch="df")


# ---------------------------------------------------------------------------
# PerformanceQueries
# ---------------------------------------------------------------------------

class PerformanceQueries:
    """Equity curve, regime breakdown, hourly patterns, win rates."""

    def __init__(self, db: SRFMDatabase) -> None:
        self._db = db

    def get_equity_curve(self, since: str | None = None) -> pd.Series:
        """
        Return NAV time series as a Series indexed by bar_time.
        Uses equity_curve table if available, else nav_log.
        """
        if self._db.table_exists(Tables.EQUITY_CURVE):
            clauses: list[str] = []
            params: list[Any] = []
            if since:
                clauses.append("bar_time >= ?")
                params.append(since)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            sql = f"SELECT bar_time, nav FROM {Tables.EQUITY_CURVE} {where} ORDER BY bar_time"  # noqa: S608
            df = self._db.execute(sql, params, fetch="df")
            if not df.empty:
                df["bar_time"] = pd.to_datetime(df["bar_time"])
                return df.set_index("bar_time")["nav"]

        return pd.Series(dtype=float)

    def get_performance_by_regime(self) -> pd.DataFrame:
        """
        Join trades with regime_log to compute P&L stats grouped by regime.
        Returns columns: regime, num_trades, total_pnl, win_rate, avg_pnl.
        """
        sql = f"""
            SELECT
                t.regime,
                COUNT(*)                              AS num_trades,
                SUM(t.pnl)                            AS total_pnl,
                AVG(t.pnl)                            AS avg_pnl,
                SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_rate,
                MAX(t.pnl)                            AS best_trade,
                MIN(t.pnl)                            AS worst_trade
            FROM {Tables.TRADES} t
            WHERE t.exit_time IS NOT NULL
              AND t.regime IS NOT NULL
            GROUP BY t.regime
            ORDER BY total_pnl DESC
        """  # noqa: S608
        return self._db.execute(sql, fetch="df")

    def get_performance_by_hour(self) -> pd.DataFrame:
        """
        Return P&L stats broken down by hour-of-day.
        Useful for identifying intraday edge patterns.
        """
        sql = f"""
            SELECT
                CAST(strftime('%H', entry_time) AS INTEGER) AS hour_of_day,
                COUNT(*)                                     AS num_trades,
                SUM(pnl)                                     AS total_pnl,
                AVG(pnl)                                     AS avg_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_rate
            FROM {Tables.TRADES}
            WHERE exit_time IS NOT NULL
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """  # noqa: S608
        return self._db.execute(sql, fetch="df")

    def get_win_rate_by_symbol(self) -> dict[str, float]:
        """Return win rate (fraction) keyed by symbol, closed trades only."""
        sql = f"""
            SELECT
                symbol,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_rate
            FROM {Tables.TRADES}
            WHERE exit_time IS NOT NULL
            GROUP BY symbol
        """  # noqa: S608
        rows = self._db.execute(sql, fetch="all")
        return {r["symbol"]: r["win_rate"] for r in rows}

    def get_monthly_summary(self) -> pd.DataFrame:
        """Return monthly performance stats."""
        sql = f"""
            SELECT * FROM {Tables.MONTHLY_PERFORMANCE}
            ORDER BY month
        """  # noqa: S608
        return self._db.execute(sql, fetch="df")

    def get_profit_factor(self, since: str | None = None) -> float:
        """Return overall profit factor (gross_profit / |gross_loss|)."""
        clauses = ["exit_time IS NOT NULL", "pnl IS NOT NULL"]
        params: list[Any] = []
        if since:
            clauses.append("exit_time >= ?")
            params.append(since)
        where = "WHERE " + " AND ".join(clauses)
        sql = f"""
            SELECT
                SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END)  AS gross_profit,
                SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) AS gross_loss
            FROM {Tables.TRADES} {where}
        """  # noqa: S608
        row = self._db.execute(sql, params, fetch="one")
        if not row or not row["gross_loss"]:
            return float("inf")
        return (row["gross_profit"] or 0.0) / row["gross_loss"]

    def get_calmar_ratio(self, years: float = 1.0) -> float | None:
        """
        Return Calmar ratio = CAGR / max_drawdown.
        Uses daily_performance if available.
        """
        sql = f"""
            SELECT MIN(daily_return) AS worst_day,
                   AVG(daily_return) AS avg_return,
                   COUNT(*)          AS num_days
            FROM {Tables.DAILY_PERFORMANCE}
        """  # noqa: S608
        if not self._db.table_exists(Tables.DAILY_PERFORMANCE):
            return None
        row = self._db.execute(sql, fetch="one")
        if not row or not row["num_days"]:
            return None
        ann_return = (1 + row["avg_return"]) ** 252 - 1
        max_dd_sql = f"SELECT MAX(max_drawdown) FROM {Tables.DAILY_PERFORMANCE}"  # noqa: S608
        max_dd = self._db.execute(max_dd_sql, fetch="scalar") or 0.0
        if max_dd == 0.0:
            return None
        return ann_return / max_dd

    def get_execution_quality_summary(self) -> pd.DataFrame:
        """Return average slippage metrics by symbol."""
        sql = f"""
            SELECT
                symbol,
                COUNT(*)                        AS num_fills,
                AVG(arrival_slippage_bps)       AS avg_arrival_slippage_bps,
                AVG(vwap_slippage_bps)          AS avg_vwap_slippage_bps,
                AVG(market_impact_bps)          AS avg_market_impact_bps,
                AVG(time_to_fill_ms)            AS avg_fill_latency_ms,
                SUM(total_fees)                 AS total_fees
            FROM {Tables.EXECUTION_QUALITY}
            GROUP BY symbol
            ORDER BY num_fills DESC
        """  # noqa: S608
        return self._db.execute(sql, fetch="df")


# ---------------------------------------------------------------------------
# ParameterQueries
# ---------------------------------------------------------------------------

class ParameterQueries:
    """Parameter snapshot history, current state, and audit logging."""

    def __init__(self, db: SRFMDatabase) -> None:
        self._db = db

    def get_param_history(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the n most recent parameter snapshots as a list of dicts."""
        sql = f"""
            SELECT id, snapshot_time, source, change_summary,
                   trigger_sharpe, trigger_drawdown, trigger_regime,
                   genome_id, genome_generation, genome_fitness,
                   validation_passed, applied_by
            FROM {Tables.PARAMETER_SNAPSHOTS}
            ORDER BY snapshot_time DESC
            LIMIT ?
        """  # noqa: S608
        rows = self._db.execute(sql, (n,), fetch="all")
        return [dict(r) for r in rows]

    def get_current_params(self) -> dict[str, Any]:
        """
        Return the most recently applied parameter snapshot as a dict.
        The params_json field is decoded from JSON.
        Returns empty dict if no snapshots exist.
        """
        import json

        row = self._db.execute(
            f"""
            SELECT params_json, snapshot_time, source, genome_id
            FROM {Tables.PARAMETER_SNAPSHOTS}
            WHERE validation_passed = 1
            ORDER BY snapshot_time DESC
            LIMIT 1
            """,  # noqa: S608
            fetch="one",
        )
        if not row:
            return {}
        try:
            params = json.loads(row["params_json"])
        except (ValueError, TypeError):
            params = {}
        params["_snapshot_time"] = row["snapshot_time"]
        params["_source"] = row["source"]
        params["_genome_id"] = row["genome_id"]
        return params

    def log_param_update(
        self,
        old_params: dict[str, Any],
        new_params: dict[str, Any],
        source: str,
        applied_by: str | None = None,
        genome_id: str | None = None,
        notes: str | None = None,
    ) -> int:
        """
        Insert a new parameter snapshot.
        Computes a simple change_summary by diffing keys.
        Returns the new row id.
        """
        import json

        changed = [
            k for k in set(list(old_params) + list(new_params))
            if old_params.get(k) != new_params.get(k)
        ]
        change_summary = f"Changed {len(changed)} keys: {', '.join(sorted(changed)[:10])}"

        # Compute RFC-6902-style delta (simplified key/value diff)
        delta = {
            k: {"old": old_params.get(k), "new": new_params.get(k)}
            for k in changed
        }

        conn = self._db._get_conn()
        cur = conn.execute(
            f"""
            INSERT INTO {Tables.PARAMETER_SNAPSHOTS}
              (source, params_json, delta_json, change_summary,
               genome_id, applied_by, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,  # noqa: S608
            (
                source,
                json.dumps(new_params),
                json.dumps(delta),
                change_summary,
                genome_id,
                applied_by,
                notes,
            ),
        )
        conn.commit()
        return cur.lastrowid

    def get_params_at_time(self, timestamp: str) -> dict[str, Any]:
        """
        Return the effective parameter set at a given ISO timestamp.
        Returns the latest snapshot whose snapshot_time <= timestamp.
        """
        import json

        row = self._db.execute(
            f"""
            SELECT params_json FROM {Tables.PARAMETER_SNAPSHOTS}
            WHERE snapshot_time <= ? AND validation_passed = 1
            ORDER BY snapshot_time DESC LIMIT 1
            """,  # noqa: S608
            (timestamp,),
            fetch="one",
        )
        if not row:
            return {}
        try:
            return json.loads(row["params_json"])
        except (ValueError, TypeError):
            return {}

    def get_genome_performance_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame of genome performance sorted by adjusted_fitness.
        """
        sql = f"""
            SELECT genome_id, generation, creation_method,
                   fitness, adjusted_fitness,
                   is_sharpe, oos_sharpe,
                   is_max_drawdown, oos_max_drawdown,
                   is_elite, was_deployed, cycle_id
            FROM {Tables.GENOME_HISTORY}
            WHERE fitness IS NOT NULL
            ORDER BY adjusted_fitness DESC
        """  # noqa: S608
        return self._db.execute(sql, fetch="df")

    def get_feature_importance_latest(
        self,
        model_name: str,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """
        Return the latest feature importance snapshot for a model.
        Decodes importances_json into individual rows.
        """
        import json

        clauses = ["model_name = ?"]
        params: list[Any] = [model_name]
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)

        where = "WHERE " + " AND ".join(clauses)
        sql = f"""
            SELECT importances_json, snapshot_time, model_version
            FROM {Tables.FEATURE_IMPORTANCE}
            {where}
            ORDER BY snapshot_time DESC LIMIT 1
        """  # noqa: S608
        row = self._db.execute(sql, params, fetch="one")
        if not row:
            return pd.DataFrame(columns=["feature", "importance", "rank"])
        try:
            data = json.loads(row["importances_json"])
            df = pd.DataFrame(data)
            df["snapshot_time"] = row["snapshot_time"]
            df["model_version"] = row["model_version"]
            return df.sort_values("importance", ascending=False).reset_index(drop=True)
        except (ValueError, TypeError, KeyError):
            return pd.DataFrame(columns=["feature", "importance", "rank"])

    def get_risk_metrics_latest(self) -> dict[str, Any] | None:
        """Return the most recent risk metrics snapshot."""
        row = self._db.execute(
            f"""
            SELECT * FROM {Tables.RISK_METRICS}
            ORDER BY snapshot_time DESC LIMIT 1
            """,  # noqa: S608
            fetch="one",
        )
        return dict(row) if row else None

    def get_risk_metrics_history(
        self,
        since: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return risk metrics time series for charting."""
        cols = ", ".join(columns) if columns else "*"
        clauses: list[str] = []
        params: list[Any] = []
        if since:
            clauses.append("snapshot_time >= ?")
            params.append(since)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT {cols} FROM {Tables.RISK_METRICS} {where} ORDER BY snapshot_time"  # noqa: S608
        df = self._db.execute(sql, params, fetch="df")
        if not df.empty and "snapshot_time" in df.columns:
            df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])
        return df
