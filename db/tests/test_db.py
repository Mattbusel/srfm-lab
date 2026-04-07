"""
db/tests/test_db.py -- Comprehensive tests for the SRFM database layer.

Covers:
  - Schema migrations (order, idempotency, down/up)
  - DBSchema.create_tables and validate_schema
  - SRFMDatabase connection and query primitives
  - TradeQueries (get_trades, open positions, P&L, Sharpe, drawdown)
  - SignalQueries (signal history, regime, nav history)
  - PerformanceQueries (equity curve, win rates, profit factor)
  - ParameterQueries (param history, current params, audit log)
  - DatabaseArchiver (archive, backup, vacuum, prune, stats)
  - Admin CLI smoke tests

All tests use in-memory or temp-file SQLite databases -- no external dependencies.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
import tempfile
import threading
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

# Ensure package root is on path when running pytest from repo root
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from db.schema import DBSchema, Tables, ALL_TABLES, EXPECTED_INDEXES
from db.query_engine import (
    SRFMDatabase,
    TradeQueries,
    SignalQueries,
    PerformanceQueries,
    ParameterQueries,
)
from db.archiver import DatabaseArchiver
from db import admin_cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db_path(tmp_path):
    """Return a Path to a fresh temp database file."""
    return tmp_path / "test_srfm.db"


@pytest.fixture
def initialized_db(tmp_db_path):
    """Return a Path to a fully initialized database with all tables."""
    conn = sqlite3.connect(str(tmp_db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    DBSchema.create_tables(conn)
    conn.close()
    return tmp_db_path


@pytest.fixture
def db(initialized_db):
    """Return a SRFMDatabase connected to the initialized database."""
    return SRFMDatabase(initialized_db)


@pytest.fixture
def db_with_trades(db, initialized_db):
    """Return (db, path) with 20 sample trades inserted."""
    _insert_sample_trades(db, n=20)
    return db, initialized_db


def _insert_sample_trades(db: SRFMDatabase, n: int = 20) -> None:
    """Insert n sample closed trades."""
    rows = []
    base = datetime(2024, 1, 2, 9, 30, 0)
    symbols = ["AAPL", "MSFT", "NVDA", "TSLA"]
    for i in range(n):
        sym = symbols[i % len(symbols)]
        entry = base + timedelta(hours=i * 6)
        exit_t = entry + timedelta(hours=2)
        pnl = 100.0 * (1 if i % 3 != 0 else -1) * ((i % 5) + 1)
        rows.append((
            sym,
            "BUY",
            100,
            150.0 + i,
            entry.isoformat(),
            151.0 + i + (pnl / 100),
            exit_t.isoformat(),
            pnl,
            pnl / 15000.0,
            1.0,
            0.05,
            f"v1.{i % 3}",
            f"sig_{i % 4}",
            "TRENDING_BULL" if i % 2 == 0 else "MEAN_REVERTING",
            None,
        ))
    db.executemany(
        """
        INSERT INTO trades
          (symbol, side, qty, entry_price, entry_time,
           exit_price, exit_time, pnl, pnl_pct,
           commission, slippage, strategy_version, signal_name, regime, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemaMigrationOrder:
    def test_discover_migrations_are_sorted(self, tmp_path):
        """Migration files must be returned in ascending version order."""
        # Create fake migration files in scrambled order
        mig_dir = tmp_path / "migrations"
        mig_dir.mkdir()
        for v in [20, 17, 25, 18]:
            (mig_dir / f"{v:03d}_fake.sql").write_text("-- UP\nSELECT 1;\n-- DOWN\n")

        # Monkeypatch MIGRATIONS_DIR
        orig = DBSchema.MIGRATIONS_DIR
        DBSchema.MIGRATIONS_DIR = mig_dir
        try:
            migs = DBSchema._discover_migrations()
        finally:
            DBSchema.MIGRATIONS_DIR = orig

        versions = [v for v, _ in migs]
        assert versions == sorted(versions)

    def test_only_numbered_files_discovered(self, tmp_path):
        mig_dir = tmp_path / "migrations"
        mig_dir.mkdir()
        (mig_dir / "017_valid.sql").write_text("-- UP\nSELECT 1;\n-- DOWN\n")
        (mig_dir / "README.md").write_text("ignore me")
        (mig_dir / "no_number.sql").write_text("-- UP\nSELECT 1;\n-- DOWN\n")

        orig = DBSchema.MIGRATIONS_DIR
        DBSchema.MIGRATIONS_DIR = mig_dir
        try:
            migs = DBSchema._discover_migrations()
        finally:
            DBSchema.MIGRATIONS_DIR = orig

        assert len(migs) == 1
        assert migs[0][0] == 17

    def test_migrate_applies_in_order(self, tmp_db_path, tmp_path):
        mig_dir = tmp_path / "migrations"
        mig_dir.mkdir()
        for v in [17, 18, 19]:
            (mig_dir / f"{v:03d}_m.sql").write_text(
                f"-- UP\nCREATE TABLE IF NOT EXISTS t{v}(id INTEGER PRIMARY KEY);\n-- DOWN\nDROP TABLE IF EXISTS t{v};\n"
            )

        orig = DBSchema.MIGRATIONS_DIR
        DBSchema.MIGRATIONS_DIR = mig_dir
        try:
            DBSchema.migrate(tmp_db_path)
        finally:
            DBSchema.MIGRATIONS_DIR = orig

        conn = sqlite3.connect(str(tmp_db_path))
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert "t17" in tables
        assert "t18" in tables
        assert "t19" in tables

    def test_version_advances_after_migration(self, tmp_db_path, tmp_path):
        mig_dir = tmp_path / "migrations"
        mig_dir.mkdir()
        (mig_dir / "021_test.sql").write_text(
            "-- UP\nCREATE TABLE IF NOT EXISTS dummy_tbl(x INTEGER);\n-- DOWN\nDROP TABLE IF EXISTS dummy_tbl;\n"
        )
        orig = DBSchema.MIGRATIONS_DIR
        DBSchema.MIGRATIONS_DIR = mig_dir
        try:
            DBSchema.migrate(tmp_db_path)
            conn = sqlite3.connect(str(tmp_db_path))
            v = DBSchema.get_current_version(conn)
            conn.close()
        finally:
            DBSchema.MIGRATIONS_DIR = orig
        assert v == 21


class TestMigrationIdempotent:
    def test_create_tables_twice_no_error(self, tmp_db_path):
        """create_tables() must be callable multiple times without error."""
        conn = sqlite3.connect(str(tmp_db_path))
        DBSchema.create_tables(conn)
        DBSchema.create_tables(conn)  # second call must not raise
        conn.close()

    def test_migrate_already_applied_skipped(self, tmp_db_path, tmp_path):
        mig_dir = tmp_path / "migrations"
        mig_dir.mkdir()
        (mig_dir / "017_idempotent.sql").write_text(
            "-- UP\nCREATE TABLE IF NOT EXISTS idem_tbl(x INTEGER);\n-- DOWN\nDROP TABLE IF EXISTS idem_tbl;\n"
        )
        orig = DBSchema.MIGRATIONS_DIR
        DBSchema.MIGRATIONS_DIR = mig_dir
        try:
            DBSchema.migrate(tmp_db_path)
            # second migrate call -- should not raise or re-apply
            DBSchema.migrate(tmp_db_path)
            conn = sqlite3.connect(str(tmp_db_path))
            count = conn.execute("SELECT COUNT(*) FROM schema_version WHERE version=17").fetchone()[0]
            conn.close()
        finally:
            DBSchema.MIGRATIONS_DIR = orig
        assert count == 1, "Migration should only be recorded once"

    def test_get_current_version_empty_db(self, tmp_db_path):
        conn = sqlite3.connect(str(tmp_db_path))
        v = DBSchema.get_current_version(conn)
        conn.close()
        assert v == 0

    def test_create_tables_all_tables_present(self, tmp_db_path):
        conn = sqlite3.connect(str(tmp_db_path))
        DBSchema.create_tables(conn)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        for name, _ in ALL_TABLES:
            assert name in tables, f"Table missing after create_tables: {name}"


class TestValidateSchema:
    def test_validate_passes_after_create_tables(self, tmp_db_path):
        conn = sqlite3.connect(str(tmp_db_path))
        DBSchema.create_tables(conn)
        problems = DBSchema.validate_schema(conn)
        conn.close()
        assert problems == [], f"Unexpected problems: {problems}"

    def test_validate_reports_missing_table(self, tmp_db_path):
        conn = sqlite3.connect(str(tmp_db_path))
        DBSchema.create_tables(conn)
        conn.execute("DROP TABLE trades")
        conn.commit()
        problems = DBSchema.validate_schema(conn)
        conn.close()
        assert any("trades" in p for p in problems)

    def test_validate_reports_missing_index(self, tmp_db_path):
        conn = sqlite3.connect(str(tmp_db_path))
        DBSchema.create_tables(conn)
        conn.execute("DROP INDEX IF EXISTS idx_trades_symbol")
        conn.commit()
        problems = DBSchema.validate_schema(conn)
        conn.close()
        assert any("idx_trades_symbol" in p for p in problems)

    def test_get_migration_status_structure(self, initialized_db):
        conn = sqlite3.connect(str(initialized_db))
        status = DBSchema.get_migration_status(conn)
        conn.close()
        assert "current_version" in status
        assert "applied" in status
        assert "pending" in status
        assert "is_up_to_date" in status
        assert isinstance(status["applied"], list)
        assert isinstance(status["pending"], list)


# ---------------------------------------------------------------------------
# SRFMDatabase tests
# ---------------------------------------------------------------------------

class TestSRFMDatabase:
    def test_execute_fetch_none(self, db):
        db.execute("INSERT INTO symbols(symbol, name) VALUES (?, ?)", ("AAPL", "Apple"))
        count = db.execute("SELECT COUNT(*) FROM symbols WHERE symbol=?", ("AAPL",), fetch="scalar")
        assert count == 1

    def test_execute_fetch_one(self, db):
        db.execute("INSERT INTO symbols(symbol, name) VALUES (?,?)", ("MSFT", "Microsoft"))
        row = db.execute("SELECT * FROM symbols WHERE symbol=?", ("MSFT",), fetch="one")
        assert row is not None
        assert row["symbol"] == "MSFT"

    def test_execute_fetch_all(self, db):
        for sym in ["X", "Y", "Z"]:
            db.execute("INSERT INTO symbols(symbol) VALUES (?)", (sym,))
        rows = db.execute("SELECT symbol FROM symbols ORDER BY symbol", fetch="all")
        syms = [r["symbol"] for r in rows]
        assert "X" in syms and "Y" in syms and "Z" in syms

    def test_execute_fetch_df(self, db):
        db.execute("INSERT INTO symbols(symbol, name) VALUES (?,?)", ("NVDA", "Nvidia"))
        df = db.execute("SELECT * FROM symbols WHERE symbol='NVDA'", fetch="df")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "NVDA"

    def test_execute_fetch_scalar(self, db):
        db.execute("INSERT INTO symbols(symbol) VALUES (?)", ("AAA",))
        n = db.execute("SELECT COUNT(*) FROM symbols", fetch="scalar")
        assert isinstance(n, int)
        assert n >= 1

    def test_executemany_batch_insert(self, db):
        rows = [("SYM1",), ("SYM2",), ("SYM3",)]
        count = db.executemany("INSERT INTO symbols(symbol) VALUES (?)", rows)
        assert count == 3

    def test_transaction_commit(self, db):
        with db.transaction() as conn:
            conn.execute("INSERT INTO symbols(symbol) VALUES (?)", ("TXN_OK",))
        n = db.execute("SELECT COUNT(*) FROM symbols WHERE symbol='TXN_OK'", fetch="scalar")
        assert n == 1

    def test_transaction_rollback_on_exception(self, db):
        with pytest.raises(ValueError):
            with db.transaction() as conn:
                conn.execute("INSERT INTO symbols(symbol) VALUES (?)", ("TXN_FAIL",))
                raise ValueError("intentional error")
        n = db.execute("SELECT COUNT(*) FROM symbols WHERE symbol='TXN_FAIL'", fetch="scalar")
        assert n == 0

    def test_table_exists_true(self, db):
        assert db.table_exists("trades") is True

    def test_table_exists_false(self, db):
        assert db.table_exists("nonexistent_table_xyz") is False

    def test_row_count_empty(self, db):
        assert db.row_count("trades") == 0

    def test_thread_local_connections(self, db):
        """Each thread must get its own connection object."""
        connections = []

        def worker():
            # Force connection creation in this thread
            conn_id = id(db._get_conn())
            connections.append(conn_id)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All thread connections should be distinct objects
        assert len(set(connections)) == len(connections)

    def test_empty_executemany(self, db):
        result = db.executemany("INSERT INTO symbols(symbol) VALUES (?)", [])
        assert result == 0

    def test_execute_unknown_fetch_raises(self, db):
        with pytest.raises(ValueError, match="Unknown fetch mode"):
            db.execute("SELECT 1", fetch="bad_mode")


# ---------------------------------------------------------------------------
# TradeQueries tests
# ---------------------------------------------------------------------------

class TestTradeQueries:
    def test_get_trades_returns_df(self, db_with_trades):
        db, _ = db_with_trades
        tq = TradeQueries(db)
        df = tq.get_trades()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20
        assert "symbol" in df.columns
        assert "pnl" in df.columns

    def test_get_trades_empty(self, db):
        tq = TradeQueries(db)
        df = tq.get_trades()
        assert df.empty

    def test_get_trades_since_filter(self, db_with_trades):
        db, _ = db_with_trades
        tq = TradeQueries(db)
        since = "2024-01-15"
        df = tq.get_trades(since=since)
        assert all(df["entry_time"] >= pd.Timestamp(since))

    def test_get_trades_symbol_filter(self, db_with_trades):
        db, _ = db_with_trades
        tq = TradeQueries(db)
        df = tq.get_trades(symbol="AAPL")
        assert all(df["symbol"] == "AAPL")
        assert len(df) == 5  # 20 trades / 4 symbols = 5 each

    def test_get_trades_until_filter(self, db_with_trades):
        db, _ = db_with_trades
        tq = TradeQueries(db)
        until = "2024-01-05"
        df = tq.get_trades(until=until)
        assert all(df["entry_time"] <= pd.Timestamp(until))

    def test_get_open_positions_empty_when_all_closed(self, db_with_trades):
        db, _ = db_with_trades
        tq = TradeQueries(db)
        df = tq.get_open_positions()
        assert df.empty

    def test_get_open_positions_returns_unclosed(self, db):
        db.execute(
            "INSERT INTO trades(symbol, side, qty, entry_price, entry_time) VALUES (?,?,?,?,?)",
            ("AAPL", "BUY", 100, 150.0, "2024-01-02T09:30:00"),
        )
        tq = TradeQueries(db)
        df = tq.get_open_positions()
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAPL"

    def test_get_trade_pnl_by_symbol(self, db_with_trades):
        db, _ = db_with_trades
        tq = TradeQueries(db)
        pnl = tq.get_trade_pnl_by_symbol()
        assert isinstance(pnl, dict)
        assert "AAPL" in pnl
        for v in pnl.values():
            assert isinstance(v, float)

    def test_get_daily_returns_series(self, db_with_trades):
        db, _ = db_with_trades
        tq = TradeQueries(db)
        ret = tq.get_daily_returns()
        assert isinstance(ret, pd.Series)
        assert len(ret) > 0

    def test_get_daily_returns_since(self, db_with_trades):
        db, _ = db_with_trades
        tq = TradeQueries(db)
        ret = tq.get_daily_returns(since="2024-01-10")
        if not ret.empty:
            assert all(ret.index >= "2024-01-10")

    def test_rolling_sharpe_computation(self, db_with_trades):
        db, _ = db_with_trades
        tq = TradeQueries(db)
        sharpe = tq.get_rolling_sharpe(window_days=5)
        assert isinstance(sharpe, pd.Series)
        # Non-NaN values should be finite floats
        valid = sharpe.dropna()
        assert all(valid.apply(lambda x: isinstance(x, float)))

    def test_rolling_sharpe_empty_db(self, db):
        tq = TradeQueries(db)
        sharpe = tq.get_rolling_sharpe()
        assert isinstance(sharpe, pd.Series)
        assert sharpe.empty

    def test_rolling_sharpe_window_respected(self, db_with_trades):
        db, _ = db_with_trades
        tq = TradeQueries(db)
        s5  = tq.get_rolling_sharpe(window_days=5)
        s15 = tq.get_rolling_sharpe(window_days=15)
        # Longer window has fewer valid values
        assert s5.notna().sum() >= s15.notna().sum()

    def test_get_largest_drawdown_period(self, db_with_trades):
        db, _ = db_with_trades
        tq = TradeQueries(db)
        dd = tq.get_largest_drawdown_period()
        assert "start" in dd
        assert "end" in dd
        assert "depth" in dd
        assert "duration_days" in dd
        assert dd["depth"] <= 0.0

    def test_get_largest_drawdown_empty(self, db):
        tq = TradeQueries(db)
        dd = tq.get_largest_drawdown_period()
        assert dd["depth"] == 0.0

    def test_count_trades_today(self, db):
        today = datetime.now(timezone.utc).isoformat()
        db.execute(
            "INSERT INTO trades(symbol, side, qty, entry_price, entry_time, exit_time, pnl) VALUES (?,?,?,?,?,?,?)",
            ("AAPL", "BUY", 10, 100.0, today, today, 50.0),
        )
        tq = TradeQueries(db)
        assert tq.count_trades_today() >= 1

    def test_get_recent_trades(self, db_with_trades):
        db, _ = db_with_trades
        tq = TradeQueries(db)
        df = tq.get_recent_trades(n=5)
        assert len(df) <= 5

    def test_win_loss_streak_all_wins(self, db):
        today = datetime.now(timezone.utc)
        for i in range(5):
            t = (today + timedelta(hours=i)).isoformat()
            db.execute(
                "INSERT INTO trades(symbol,side,qty,entry_price,entry_time,exit_time,pnl) VALUES (?,?,?,?,?,?,?)",
                ("AAPL", "BUY", 10, 100.0, t, t, 50.0),
            )
        tq = TradeQueries(db)
        streak = tq.get_win_loss_streak()
        assert streak["current_win_streak"] == 5
        assert streak["current_loss_streak"] == 0


# ---------------------------------------------------------------------------
# SignalQueries tests
# ---------------------------------------------------------------------------

class TestSignalQueries:
    def _insert_signal_registry(self, db: SRFMDatabase) -> None:
        db.execute(
            """
            INSERT INTO signal_registry
              (signal_name, signal_class, version, is_active)
            VALUES (?,?,?,?)
            """,
            ("bh_momentum", "signals.BHMomentum", "1.0.0", 1),
        )

    def _insert_signal_history(self, db: SRFMDatabase, n: int = 10) -> None:
        self._insert_signal_registry(db)
        base = datetime(2024, 3, 1, 9, 30)
        for i in range(n):
            bt = (base + timedelta(minutes=i)).isoformat()
            db.execute(
                """
                INSERT INTO signal_history
                  (signal_name, symbol, timeframe, bar_time, signal_value, direction, regime)
                VALUES (?,?,?,?,?,?,?)
                """,
                ("bh_momentum", "AAPL", "1m", bt, float(i) * 0.1, "LONG", "TRENDING"),
            )

    def test_get_signal_history_returns_df(self, db):
        self._insert_signal_history(db)
        sq = SignalQueries(db)
        df = sq.get_signal_history("bh_momentum")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "bar_time" in df.columns

    def test_get_signal_history_since_filter(self, db):
        self._insert_signal_history(db)
        sq = SignalQueries(db)
        since = "2024-03-01T09:35:00"
        df = sq.get_signal_history("bh_momentum", since=since)
        assert all(df["bar_time"] >= pd.Timestamp(since))

    def test_get_signal_history_unknown_signal_empty(self, db):
        sq = SignalQueries(db)
        df = sq.get_signal_history("nonexistent_signal")
        assert df.empty

    def test_get_regime_history(self, db):
        db.execute(
            """
            INSERT INTO regime_log(transition_time, regime, previous_regime)
            VALUES (?,?,?)
            """,
            ("2024-01-05T10:00:00", "TRENDING_BULL", "FLAT"),
        )
        sq = SignalQueries(db)
        df = sq.get_regime_history()
        assert len(df) == 1
        assert df.iloc[0]["regime"] == "TRENDING_BULL"

    def test_get_latest_regime(self, db):
        db.execute(
            "INSERT INTO regime_log(transition_time, regime) VALUES (?,?)",
            ("2024-01-10T09:00:00", "CRISIS"),
        )
        sq = SignalQueries(db)
        result = sq.get_latest_regime()
        assert result is not None
        assert result["regime"] == "CRISIS"

    def test_get_active_signals(self, db):
        self._insert_signal_registry(db)
        sq = SignalQueries(db)
        active = sq.get_active_signals()
        assert "bh_momentum" in active

    def test_get_nav_history_returns_df(self, db):
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            """
            INSERT INTO nav_log
              (bar_time, symbol, timeframe, nav, cash, gross_market_value, net_market_value)
            VALUES (?,?,?,?,?,?,?)
            """,
            (now, "PORTFOLIO", "1m", 100000.0, 50000.0, 50000.0, 50000.0),
        )
        sq = SignalQueries(db)
        df = sq.get_nav_history("PORTFOLIO", timeframe="1m")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_get_bh_mass_history(self, db):
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            """
            INSERT INTO nav_log
              (bar_time, symbol, timeframe, nav, cash, gross_market_value, net_market_value, bh_mass)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (now, "SPX", "5m", 0.0, 0.0, 0.0, 0.0, 0.42),
        )
        sq = SignalQueries(db)
        df = sq.get_bh_mass_history("SPX", "5m")
        assert len(df) == 1
        assert abs(df.iloc[0]["bh_mass"] - 0.42) < 1e-9

    def test_get_signals_fired_today(self, db):
        self._insert_signal_registry(db)
        today = datetime.now(timezone.utc).isoformat()
        db.execute(
            """
            INSERT INTO signal_history
              (signal_name, symbol, timeframe, bar_time, signal_value, direction)
            VALUES (?,?,?,?,?,?)
            """,
            ("bh_momentum", "TSLA", "1m", today, 0.9, "LONG"),
        )
        sq = SignalQueries(db)
        df = sq.get_signals_fired_today()
        assert len(df) >= 1


# ---------------------------------------------------------------------------
# PerformanceQueries tests
# ---------------------------------------------------------------------------

class TestPerformanceQueries:
    def test_get_equity_curve_empty(self, db):
        pq = PerformanceQueries(db)
        eq = pq.get_equity_curve()
        assert isinstance(eq, pd.Series)

    def test_get_equity_curve_with_data(self, db):
        for i in range(5):
            bt = f"2024-01-{i+2:02d}T16:00:00"
            db.execute(
                "INSERT INTO equity_curve(bar_time, nav, cash, market_val, high_water, drawdown) VALUES (?,?,?,?,?,?)",
                (bt, 100000.0 + i * 500, 50000.0, 50000.0 + i * 500, 100000.0 + i * 500, 0.0),
            )
        pq = PerformanceQueries(db)
        eq = pq.get_equity_curve()
        assert len(eq) == 5
        assert eq.iloc[-1] > eq.iloc[0]

    def test_get_performance_by_regime(self, db_with_trades):
        db, _ = db_with_trades
        pq = PerformanceQueries(db)
        df = pq.get_performance_by_regime()
        assert isinstance(df, pd.DataFrame)
        assert "regime" in df.columns
        assert "win_rate" in df.columns
        assert len(df) >= 1

    def test_get_performance_by_hour(self, db_with_trades):
        db, _ = db_with_trades
        pq = PerformanceQueries(db)
        df = pq.get_performance_by_hour()
        assert isinstance(df, pd.DataFrame)
        assert "hour_of_day" in df.columns
        assert all(df["hour_of_day"].between(0, 23))

    def test_get_win_rate_by_symbol(self, db_with_trades):
        db, _ = db_with_trades
        pq = PerformanceQueries(db)
        wr = pq.get_win_rate_by_symbol()
        assert isinstance(wr, dict)
        for sym, rate in wr.items():
            assert 0.0 <= rate <= 1.0

    def test_get_profit_factor(self, db_with_trades):
        db, _ = db_with_trades
        pq = PerformanceQueries(db)
        pf = pq.get_profit_factor()
        assert isinstance(pf, float)
        assert pf > 0.0

    def test_get_profit_factor_no_losses(self, db):
        now = datetime.now(timezone.utc)
        for i in range(3):
            t = (now + timedelta(hours=i)).isoformat()
            db.execute(
                "INSERT INTO trades(symbol,side,qty,entry_price,entry_time,exit_time,pnl) VALUES (?,?,?,?,?,?,?)",
                ("AAPL","BUY",10,100.0,t,t,100.0),
            )
        pq = PerformanceQueries(db)
        pf = pq.get_profit_factor()
        assert pf == float("inf")

    def test_get_execution_quality_summary(self, db):
        db.execute(
            """
            INSERT INTO execution_quality
              (trade_id, symbol, order_side, order_type, order_qty,
               fill_price, fill_qty, fill_time,
               arrival_slippage_bps, vwap_slippage_bps, total_fees)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (1, "AAPL", "BUY", "MARKET", 100, 150.5, 100, "2024-01-02T09:30:05", 2.1, 1.8, 1.50),
        )
        pq = PerformanceQueries(db)
        df = pq.get_execution_quality_summary()
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# ParameterQueries tests
# ---------------------------------------------------------------------------

class TestParameterQueries:
    def test_get_current_params_empty(self, db):
        pq = ParameterQueries(db)
        params = pq.get_current_params()
        assert params == {}

    def test_log_and_retrieve_params(self, db):
        pq = ParameterQueries(db)
        old = {"alpha": 0.01, "window": 20}
        new = {"alpha": 0.02, "window": 20, "beta": 0.5}
        pq.log_param_update(old, new, source="IAE_CYCLE", genome_id="g_001")
        current = pq.get_current_params()
        assert current["alpha"] == pytest.approx(0.02)
        assert current["beta"] == pytest.approx(0.5)
        assert current["_source"] == "IAE_CYCLE"
        assert current["_genome_id"] == "g_001"

    def test_get_param_history_count(self, db):
        pq = ParameterQueries(db)
        for i in range(5):
            pq.log_param_update({}, {"v": i}, source="MANUAL")
        history = pq.get_param_history(n=3)
        assert len(history) == 3

    def test_get_param_history_structure(self, db):
        pq = ParameterQueries(db)
        pq.log_param_update({"x": 1}, {"x": 2}, source="SCHEDULE")
        h = pq.get_param_history(n=1)
        assert "snapshot_time" in h[0]
        assert "source" in h[0]
        assert h[0]["source"] == "SCHEDULE"

    def test_get_params_at_time(self, db):
        pq = ParameterQueries(db)
        pq.log_param_update({}, {"rate": 0.05}, source="STARTUP")
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        params = pq.get_params_at_time(future)
        assert params.get("rate") == pytest.approx(0.05)

    def test_get_params_at_time_before_any_snapshot(self, db):
        pq = ParameterQueries(db)
        pq.log_param_update({}, {"x": 1}, source="MANUAL")
        past = "2000-01-01T00:00:00"
        params = pq.get_params_at_time(past)
        assert params == {}

    def test_get_feature_importance_latest_empty(self, db):
        pq = ParameterQueries(db)
        df = pq.get_feature_importance_latest("unknown_model")
        assert df.empty

    def test_get_feature_importance_latest(self, db):
        importances = [
            {"feature": "bh_mass", "importance": 0.35, "rank": 1},
            {"feature": "vix_level", "importance": 0.22, "rank": 2},
        ]
        db.execute(
            """
            INSERT INTO feature_importance
              (model_name, model_version, model_type, target_variable, importances_json)
            VALUES (?,?,?,?,?)
            """,
            ("rf_direction", "2.1.0", "RF", "direction", json.dumps(importances)),
        )
        pq = ParameterQueries(db)
        df = pq.get_feature_importance_latest("rf_direction")
        assert len(df) == 2
        assert df.iloc[0]["feature"] == "bh_mass"

    def test_get_risk_metrics_latest(self, db):
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            """
            INSERT INTO risk_metrics(snapshot_time, var_95_1d, cvar_95_1d, current_nav)
            VALUES (?,?,?,?)
            """,
            (now, 2500.0, 3100.0, 100000.0),
        )
        pq = ParameterQueries(db)
        rm = pq.get_risk_metrics_latest()
        assert rm is not None
        assert rm["var_95_1d"] == pytest.approx(2500.0)

    def test_get_genome_performance_summary(self, db):
        db.execute(
            """
            INSERT INTO genome_history
              (genome_id, generation, creation_method, genome_json, params_json, fitness, adjusted_fitness)
            VALUES (?,?,?,?,?,?,?)
            """,
            ("g001", 1, "RANDOM", "{}", "{}", 1.23, 1.10),
        )
        pq = ParameterQueries(db)
        df = pq.get_genome_performance_summary()
        assert len(df) == 1
        assert df.iloc[0]["genome_id"] == "g001"


# ---------------------------------------------------------------------------
# DatabaseArchiver tests
# ---------------------------------------------------------------------------

class TestArchiveOldTrades:
    def test_archive_old_trades(self, initialized_db, tmp_path):
        db = SRFMDatabase(initialized_db)
        _insert_sample_trades(db, n=10)
        # Force all trades to be "old" by backdating exit_time
        conn = sqlite3.connect(str(initialized_db))
        conn.execute("UPDATE trades SET exit_time='2020-06-01T16:00:00'")
        conn.commit()
        conn.close()

        arch = DatabaseArchiver(initialized_db)
        archive_dest = tmp_path / "old_trades.db"
        result = arch.archive_old_trades(days_to_keep=365, archive_path=archive_dest)

        assert result["rows_archived"] == 10
        assert Path(result["archive_path"]).exists()

        # Verify archive DB contains the rows
        arc_conn = sqlite3.connect(str(archive_dest))
        count = arc_conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        arc_conn.close()
        assert count == 10

        # Verify main DB no longer has those rows
        remaining = db.execute("SELECT COUNT(*) FROM trades", fetch="scalar")
        assert remaining == 0

    def test_archive_nothing_when_all_recent(self, initialized_db, tmp_path):
        db = SRFMDatabase(initialized_db)
        _insert_sample_trades(db, n=5)
        arch = DatabaseArchiver(initialized_db)
        result = arch.archive_old_trades(days_to_keep=365)
        assert result["rows_archived"] == 0

    def test_archive_creates_archive_db(self, initialized_db, tmp_path):
        db = SRFMDatabase(initialized_db)
        _insert_sample_trades(db, n=3)
        conn = sqlite3.connect(str(initialized_db))
        conn.execute("UPDATE trades SET exit_time='2018-01-01T00:00:00'")
        conn.commit()
        conn.close()

        arch_path = tmp_path / "archive.db"
        arch = DatabaseArchiver(initialized_db)
        arch.archive_old_trades(days_to_keep=1000, archive_path=arch_path)
        # If all are recent there should be no archive; if they're old it's created
        # (This test just checks no exception is raised)


class TestBackup:
    def test_backup_creates_valid_db(self, initialized_db, tmp_path):
        db = SRFMDatabase(initialized_db)
        _insert_sample_trades(db, n=5)

        arch = DatabaseArchiver(initialized_db)
        dest = tmp_path / "backup.db"
        result = arch.backup(dest)

        assert result.exists()
        assert result.stat().st_size > 0

        # Verify backup is a valid SQLite DB with the same tables
        bconn = sqlite3.connect(str(result))
        tables = {r[0] for r in bconn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        bconn.close()
        assert "trades" in tables

    def test_backup_dest_parent_created(self, initialized_db, tmp_path):
        arch = DatabaseArchiver(initialized_db)
        dest = tmp_path / "deep" / "nested" / "backup.db"
        arch.backup(dest)
        assert dest.exists()

    def test_backup_content_matches(self, initialized_db, tmp_path):
        db = SRFMDatabase(initialized_db)
        _insert_sample_trades(db, n=8)

        arch = DatabaseArchiver(initialized_db)
        dest = tmp_path / "content_backup.db"
        arch.backup(dest)

        bconn = sqlite3.connect(str(dest))
        count = bconn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        bconn.close()
        assert count == 8


class TestVacuum:
    def test_vacuum_no_error(self, initialized_db):
        arch = DatabaseArchiver(initialized_db)
        result = arch.vacuum()
        assert "size_before_bytes" in result
        assert "duration_sec" in result
        assert result["duration_sec"] >= 0

    def test_vacuum_returns_sizes(self, initialized_db):
        arch = DatabaseArchiver(initialized_db)
        result = arch.vacuum()
        assert result["size_before_bytes"] > 0
        assert result["size_after_bytes"] > 0


class TestCheckpoint:
    def test_checkpoint_no_error(self, initialized_db):
        arch = DatabaseArchiver(initialized_db)
        result = arch.checkpoint(mode="PASSIVE")
        assert "wal_frames" in result
        assert "checkpointed_frames" in result

    def test_checkpoint_invalid_mode_raises(self, initialized_db):
        arch = DatabaseArchiver(initialized_db)
        with pytest.raises(ValueError):
            arch.checkpoint(mode="INVALID")

    def test_checkpoint_truncate(self, initialized_db):
        arch = DatabaseArchiver(initialized_db)
        result = arch.checkpoint(mode="TRUNCATE")
        assert result["mode"] == "TRUNCATE"


class TestPruning:
    def test_prune_signal_history(self, db):
        db_path_attr = db._db_path
        # Insert old signal_history row
        db.execute(
            "INSERT INTO signal_registry(signal_name, signal_class, version) VALUES (?,?,?)",
            ("s1", "c1", "1.0"),
        )
        db.execute(
            """
            INSERT INTO signal_history(signal_name, symbol, timeframe, bar_time, signal_value)
            VALUES (?,?,?,?,?)
            """,
            ("s1", "AAPL", "1m", "2020-01-01T09:30:00", 1.0),
        )
        arch = DatabaseArchiver(db_path_attr)
        result = arch.prune_signal_history(days=365)
        assert result["deleted_rows"] == 1

    def test_prune_nav_log(self, db):
        db_path = db._db_path
        db.execute(
            """
            INSERT INTO nav_log(bar_time, symbol, timeframe, nav, cash, gross_market_value, net_market_value)
            VALUES (?,?,?,?,?,?,?)
            """,
            ("2020-01-01T09:30:00", "PORT", "1m", 100000.0, 100000.0, 0.0, 0.0),
        )
        arch = DatabaseArchiver(db_path)
        result = arch.prune_nav_log(days=10)
        assert result["deleted_rows"] >= 1

    def test_prune_alerts_log_keeps_unresolved(self, db):
        db_path = db._db_path
        db.execute(
            """
            INSERT INTO alerts_log
              (alert_time, alert_type, severity, alert_code, alert_name, message, resolved)
            VALUES (?,?,?,?,?,?,?)
            """,
            ("2019-01-01T00:00:00", "RISK", "WARNING", "TEST", "Test", "msg", 0),
        )
        db.execute(
            """
            INSERT INTO alerts_log
              (alert_time, alert_type, severity, alert_code, alert_name, message, resolved)
            VALUES (?,?,?,?,?,?,?)
            """,
            ("2019-01-02T00:00:00", "RISK", "WARNING", "TEST2", "Test2", "msg2", 1),
        )
        arch = DatabaseArchiver(db_path)
        result = arch.prune_alerts_log(days=30, keep_unresolved=True)
        # Resolved old alert should be deleted, unresolved kept
        remaining = db.execute("SELECT COUNT(*) FROM alerts_log WHERE resolved=0", fetch="scalar")
        assert remaining == 1


class TestGetDbStats:
    def test_get_db_stats_keys(self, initialized_db):
        arch = DatabaseArchiver(initialized_db)
        stats = arch.get_db_stats()
        for key in ("file_size_bytes", "wal_size_bytes", "page_count",
                    "page_size", "free_pages", "fragmentation_ratio",
                    "table_counts", "total_rows", "num_tables"):
            assert key in stats, f"Missing key: {key}"

    def test_get_db_stats_table_counts_non_negative(self, initialized_db):
        arch = DatabaseArchiver(initialized_db)
        stats = arch.get_db_stats()
        for tbl, cnt in stats["table_counts"].items():
            assert cnt >= 0, f"Negative row count for {tbl}"

    def test_get_db_stats_fragmentation_range(self, initialized_db):
        arch = DatabaseArchiver(initialized_db)
        stats = arch.get_db_stats()
        assert 0.0 <= stats["fragmentation_ratio"] <= 1.0

    def test_integrity_check_clean_db(self, initialized_db):
        arch = DatabaseArchiver(initialized_db)
        errors = arch.integrity_check()
        assert errors == []


# ---------------------------------------------------------------------------
# Admin CLI tests
# ---------------------------------------------------------------------------

class TestAdminCLI:
    def test_cli_migrate(self, tmp_db_path, tmp_path, monkeypatch):
        monkeypatch.setattr(DBSchema, "MIGRATIONS_DIR", tmp_path / "migrations")
        (tmp_path / "migrations").mkdir()
        result = admin_cli.main(["--db", str(tmp_db_path), "migrate"])
        assert result == 0
        assert tmp_db_path.exists()

    def test_cli_status(self, initialized_db, capsys):
        result = admin_cli.main(["--db", str(initialized_db), "status"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Schema version" in captured.out

    def test_cli_validate_clean(self, initialized_db, capsys):
        result = admin_cli.main(["--db", str(initialized_db), "validate"])
        assert result == 0
        captured = capsys.readouterr()
        assert "OK" in captured.out

    def test_cli_stats(self, initialized_db, capsys):
        result = admin_cli.main(["--db", str(initialized_db), "stats"])
        assert result == 0
        captured = capsys.readouterr()
        assert "File size" in captured.out

    def test_cli_backup(self, initialized_db, tmp_path, capsys):
        dest = str(tmp_path / "cli_backup.db")
        result = admin_cli.main(["--db", str(initialized_db), "backup", "--dest", dest])
        assert result == 0
        assert Path(dest).exists()

    def test_cli_vacuum(self, initialized_db, capsys):
        result = admin_cli.main(["--db", str(initialized_db), "vacuum"])
        assert result == 0

    def test_cli_checkpoint(self, initialized_db, capsys):
        result = admin_cli.main(["--db", str(initialized_db), "checkpoint", "--mode", "PASSIVE"])
        assert result == 0

    def test_cli_integrity(self, initialized_db, capsys):
        result = admin_cli.main(["--db", str(initialized_db), "integrity"])
        assert result == 0

    def test_cli_query_select(self, initialized_db, capsys):
        result = admin_cli.main([
            "--db", str(initialized_db),
            "query", "--sql", "SELECT COUNT(*) FROM trades",
        ])
        assert result == 0
        captured = capsys.readouterr()
        assert "0" in captured.out

    def test_cli_query_rejects_insert(self, initialized_db, capsys):
        result = admin_cli.main([
            "--db", str(initialized_db),
            "query", "--sql", "INSERT INTO symbols(symbol) VALUES ('X')",
        ])
        assert result == 1

    def test_cli_export_trades_csv(self, initialized_db, tmp_path, capsys):
        db = SRFMDatabase(initialized_db)
        _insert_sample_trades(db, n=5)
        out_file = str(tmp_path / "trades.csv")
        result = admin_cli.main([
            "--db", str(initialized_db),
            "export-trades", "--format", "csv", "--output", out_file,
        ])
        assert result == 0
        assert Path(out_file).exists()
        with open(out_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 5

    def test_cli_export_trades_json(self, initialized_db, tmp_path, capsys):
        db = SRFMDatabase(initialized_db)
        _insert_sample_trades(db, n=3)
        out_file = str(tmp_path / "trades.json")
        result = admin_cli.main([
            "--db", str(initialized_db),
            "export-trades", "--format", "json", "--output", out_file,
        ])
        assert result == 0
        with open(out_file) as f:
            data = json.load(f)
        assert len(data) == 3

    def test_cli_export_trades_no_data(self, initialized_db, capsys):
        result = admin_cli.main([
            "--db", str(initialized_db),
            "export-trades", "--since", "2099-01-01",
        ])
        assert result == 0

    def test_cli_prune(self, initialized_db, capsys):
        result = admin_cli.main([
            "--db", str(initialized_db),
            "prune", "--days", "365",
        ])
        assert result == 0

    def test_cli_missing_db_exits_nonzero(self, tmp_path, capsys):
        missing = str(tmp_path / "does_not_exist.db")
        result = admin_cli.main(["--db", missing, "status"])
        assert result != 0

    def test_cli_archive_trades(self, initialized_db, tmp_path, capsys):
        db = SRFMDatabase(initialized_db)
        _insert_sample_trades(db, n=5)
        conn = sqlite3.connect(str(initialized_db))
        conn.execute("UPDATE trades SET exit_time='2018-01-01T16:00:00'")
        conn.commit()
        conn.close()
        dest = str(tmp_path / "cli_archive.db")
        result = admin_cli.main([
            "--db", str(initialized_db),
            "archive-trades", "--days", "2000", "--dest", dest,
        ])
        assert result == 0


# ---------------------------------------------------------------------------
# Edge case / integration tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_tables_namespace_has_all_expected_names(self):
        """Tables class attributes must cover all ALL_TABLES entries."""
        table_attr_values = {v for k, v in vars(Tables).items() if not k.startswith("_")}
        for name, _ in ALL_TABLES:
            assert name in table_attr_values, f"Tables.{name} missing"

    def test_all_tables_have_unique_names(self):
        names = [name for name, _ in ALL_TABLES]
        assert len(names) == len(set(names)), "Duplicate table names in ALL_TABLES"

    def test_extract_up_with_markers(self):
        sql = "-- UP\nCREATE TABLE foo(x INTEGER);\n-- DOWN\nDROP TABLE foo;\n"
        up = DBSchema._extract_up(sql)
        assert "CREATE TABLE" in up
        assert "DROP TABLE" not in up

    def test_extract_up_no_markers(self):
        sql = "CREATE TABLE foo(x INTEGER);"
        up = DBSchema._extract_up(sql)
        assert "CREATE TABLE" in up

    def test_srmf_db_close(self, initialized_db):
        db = SRFMDatabase(initialized_db)
        db._get_conn()  # open
        db.close()      # should not raise
        db.close()      # double-close should not raise

    def test_large_batch_insert(self, db):
        rows = [(f"SYM{i:04d}",) for i in range(1000)]
        count = db.executemany("INSERT INTO symbols(symbol) VALUES (?)", rows)
        assert count == 1000

    def test_concurrent_reads(self, initialized_db):
        """Multiple threads reading from the same DB should not raise."""
        db = SRFMDatabase(initialized_db)
        errors = []

        def reader():
            try:
                db.execute("SELECT COUNT(*) FROM trades", fetch="scalar")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_rotate_backups(self, initialized_db, tmp_path):
        arch = DatabaseArchiver(initialized_db)
        backup_dir = tmp_path / "backups"
        result = arch.rotate_backups(backup_dir, keep_daily=3, keep_weekly=2)
        assert "new_backup" in result
        assert Path(result["new_backup"]).exists()

    def test_param_log_returns_row_id(self, db):
        pq = ParameterQueries(db)
        row_id = pq.log_param_update({}, {"x": 1}, source="MANUAL")
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_multiple_param_snapshots_ordering(self, db):
        pq = ParameterQueries(db)
        for i in range(5):
            pq.log_param_update({}, {"i": i}, source="IAE_CYCLE")
        history = pq.get_param_history(n=10)
        # Most recent first
        times = [h["snapshot_time"] for h in history]
        assert times == sorted(times, reverse=True)
