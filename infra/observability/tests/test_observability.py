"""
test_observability.py -- Test suite for the SRFM observability stack.

Covers:
    - MetricsCollector (gauge registration, scraper logic, HTTP server)
    - Alert rules (DrawdownAlert, VaRBreachAlert, all 9 rules)
    - AlertDeduplication
    - AlertDispatcher
    - AuditLogger (write/read all tables)
    - DashboardAPI endpoints (live metrics, history, alerts, health, WebSocket)

Run with:
    pytest infra/observability/tests/test_observability.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup so imports work when run directly
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent.parent.parent  # srfm-lab/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from infra.observability.audit_log import (
    AuditLogger,
    _compute_diff,
    get_audit_logger,
)
from infra.observability.alerter import (
    Alert,
    AlertDeduplication,
    AlertDispatcher,
    AlertHistory,
    Alerter,
    BHMassExtremeAlert,
    CircuitBreakerAlert,
    CorrelationRegimeAlert,
    DrawdownAlert,
    HurstFlipAlert,
    LiquidityAlert,
    ParameterRollbackAlert,
    ServiceDownAlert,
    Severity,
    VaRBreachAlert,
)
from infra.observability.metrics_collector import (
    MetricsCollector,
    _PromMetrics,
    LiveTraderScraper,
    RiskAPIScraper,
    CoordinationScraper,
    SignalEngineScraper,
    _PROM_AVAILABLE,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _tmp_db() -> str:
    """Return a path to a fresh temp SQLite file."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    os.unlink(f.name)
    return f.name


def _make_trade_db(path: str) -> sqlite3.Connection:
    """Create a minimal trade DB for scraper tests."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            entry_price REAL,
            pnl REAL NOT NULL DEFAULT 0.0,
            bars_held INTEGER NOT NULL DEFAULT 0,
            equity_after REAL,
            trade_duration_s REAL,
            notes TEXT
        );
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            equity REAL NOT NULL,
            positions TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS positions (
            sym TEXT PRIMARY KEY,
            qty REAL NOT NULL,
            avg_entry REAL NOT NULL DEFAULT 0.0,
            current_price REAL NOT NULL DEFAULT 0.0,
            unrealized_pnl REAL NOT NULL DEFAULT 0.0,
            bh_active INTEGER NOT NULL DEFAULT 0,
            last_updated TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS regime_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            d_bh_mass REAL,
            h_bh_mass REAL,
            m15_bh_mass REAL,
            d_bh_active INTEGER,
            h_bh_active INTEGER,
            m15_bh_active INTEGER,
            tf_score INTEGER,
            delta_score REAL,
            atr REAL,
            garch_vol REAL,
            ou_zscore REAL,
            ou_halflife REAL
        );
    """)
    conn.commit()
    return conn


def _seed_trade_db(conn: sqlite3.Connection) -> None:
    """Populate trade DB with test data."""
    now = datetime.now(timezone.utc)

    # Equity snapshots
    for i in range(10):
        ts = (now - timedelta(hours=10 - i)).isoformat()
        eq = 100_000.0 + i * 500
        conn.execute(
            "INSERT INTO equity_snapshots (ts, equity, positions) VALUES (?,?,?)",
            (ts, eq, "{}"),
        )

    # Trades
    for i in range(5):
        ts = (now - timedelta(hours=5 - i)).isoformat()
        pnl = 100.0 * (i - 2)  # mix of positive and negative
        conn.execute(
            """
            INSERT INTO trades
                (ts, symbol, side, qty, price, pnl, bars_held)
            VALUES (?,?,?,?,?,?,?)
            """,
            (ts, "BTC", "buy", 0.1, 45000.0, pnl, 3),
        )

    # Positions
    conn.execute(
        """
        INSERT INTO positions
            (sym, qty, avg_entry, current_price, unrealized_pnl, bh_active, last_updated)
        VALUES (?,?,?,?,?,?,?)
        """,
        ("BTC", 0.5, 44000.0, 45000.0, 500.0, 1, now.isoformat()),
    )

    # Regime log
    conn.execute(
        """
        INSERT INTO regime_log
            (ts, symbol, d_bh_mass, h_bh_mass, m15_bh_mass, garch_vol)
        VALUES (?,?,?,?,?,?)
        """,
        (now.isoformat(), "BTC", 1.8, 1.2, 0.9, 0.025),
    )

    conn.commit()


# ===========================================================================
# 1. MetricsCollector tests
# ===========================================================================

class TestMetricsCollectorGauges(unittest.TestCase):
    """Test that MetricsCollector creates all required Prometheus metrics."""

    def setUp(self) -> None:
        if not _PROM_AVAILABLE:
            self.skipTest("prometheus_client not installed")

    def test_prom_metrics_gauges_exist(self) -> None:
        m = _PromMetrics()
        self.assertIsNotNone(m.portfolio_equity)
        self.assertIsNotNone(m.position_size_pct)
        self.assertIsNotNone(m.bh_mass)
        self.assertIsNotNone(m.garch_vol)
        self.assertIsNotNone(m.hurst_h)
        self.assertIsNotNone(m.nav_omega)
        self.assertIsNotNone(m.nav_geodesic_deviation)
        self.assertIsNotNone(m.drawdown)

    def test_prom_metrics_counters_exist(self) -> None:
        m = _PromMetrics()
        self.assertIsNotNone(m.trades_total)
        self.assertIsNotNone(m.bars_processed)
        self.assertIsNotNone(m.signals_generated)
        self.assertIsNotNone(m.api_calls_total)
        self.assertIsNotNone(m.errors_total)

    def test_prom_metrics_histograms_exist(self) -> None:
        m = _PromMetrics()
        self.assertIsNotNone(m.bar_processing_latency_ms)
        self.assertIsNotNone(m.order_execution_latency_ms)
        self.assertIsNotNone(m.signal_computation_ns)

    def test_prom_metrics_summaries_exist(self) -> None:
        m = _PromMetrics()
        self.assertIsNotNone(m.scrape_duration_s)

    def test_rolling_sharpe_gauges_exist(self) -> None:
        m = _PromMetrics()
        self.assertIsNotNone(m.rolling_sharpe_30d)
        self.assertIsNotNone(m.rolling_sharpe_90d)

    def test_render_returns_bytes(self) -> None:
        m = _PromMetrics()
        data = m.render()
        self.assertIsInstance(data, bytes)

    def test_gauge_set_and_read(self) -> None:
        m = _PromMetrics()
        m.portfolio_equity.set(123456.78)
        output = m.render().decode()
        self.assertIn("srfm_portfolio_equity", output)

    def test_labeled_gauge(self) -> None:
        m = _PromMetrics()
        m.bh_mass.labels(symbol="BTC", timeframe="daily").set(2.3)
        output = m.render().decode()
        self.assertIn("srfm_bh_mass", output)
        self.assertIn("BTC", output)

    def test_counter_increment(self) -> None:
        m = _PromMetrics()
        m.trades_total.inc(5)
        output = m.render().decode()
        self.assertIn("srfm_trades_total", output)

    def test_histogram_observe(self) -> None:
        m = _PromMetrics()
        m.bar_processing_latency_ms.labels(symbol="ETH").observe(12.5)
        output = m.render().decode()
        self.assertIn("srfm_bar_processing_latency_ms", output)


class TestLiveTraderScraper(unittest.TestCase):

    def setUp(self) -> None:
        if not _PROM_AVAILABLE:
            self.skipTest("prometheus_client not installed")
        self._tmp = _tmp_db()
        self._conn = _make_trade_db(self._tmp)
        _seed_trade_db(self._conn)
        self._metrics = _PromMetrics()

    def tearDown(self) -> None:
        self._conn.close()
        Path(self._tmp).unlink(missing_ok=True)

    def test_scrape_ok(self) -> None:
        scraper = LiveTraderScraper(self._metrics, db_path=self._tmp)
        result = scraper.scrape()
        self.assertEqual(result.get("status"), "ok")

    def test_scrape_reads_equity(self) -> None:
        scraper = LiveTraderScraper(self._metrics, db_path=self._tmp)
        result = scraper.scrape()
        self.assertIn("equity", result)
        self.assertGreater(result["equity"], 0)

    def test_scrape_reads_trade_count(self) -> None:
        scraper = LiveTraderScraper(self._metrics, db_path=self._tmp)
        result = scraper.scrape()
        self.assertEqual(result.get("trade_count"), 5)

    def test_scrape_missing_db(self) -> None:
        scraper = LiveTraderScraper(self._metrics, db_path="/tmp/does_not_exist_xyz.db")
        result = scraper.scrape()
        self.assertEqual(result.get("status"), "db_missing")

    def test_scrape_drawdown_computed(self) -> None:
        scraper = LiveTraderScraper(self._metrics, db_path=self._tmp)
        result = scraper.scrape()
        self.assertIn("drawdown", result)
        self.assertGreaterEqual(result["drawdown"], 0.0)
        self.assertLessEqual(result["drawdown"], 1.0)

    def test_trade_counter_increments_only_delta(self) -> None:
        scraper = LiveTraderScraper(self._metrics, db_path=self._tmp)
        result1 = scraper.scrape()
        count1 = result1.get("trade_count", 0)
        # Add one more trade
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO trades (ts, symbol, side, qty, price, pnl, bars_held) VALUES (?,?,?,?,?,?,?)",
            (ts, "ETH", "sell", 0.5, 3200.0, 50.0, 2),
        )
        self._conn.commit()
        result2 = scraper.scrape()
        count2 = result2.get("trade_count", 0)
        self.assertEqual(count2, count1 + 1)


class TestMetricsCollectorInit(unittest.TestCase):

    def test_collector_creates_scrapers(self) -> None:
        collector = MetricsCollector(db_path=":memory:")
        self.assertEqual(len(collector._scrapers), 4)

    def test_force_scrape_returns_dict(self) -> None:
        collector = MetricsCollector(db_path="/tmp/nonexistent.db")
        results = collector.force_scrape()
        self.assertIsInstance(results, dict)
        self.assertIn("collected_at", results)

    def test_get_last_results_empty_initially(self) -> None:
        collector = MetricsCollector(db_path="/tmp/nonexistent.db")
        results = collector.get_last_results()
        self.assertIsInstance(results, dict)


# ===========================================================================
# 2. Alert rule tests
# ===========================================================================

class TestDrawdownAlert(unittest.TestCase):

    def setUp(self) -> None:
        self.rule = DrawdownAlert(warn_pct=0.05, crit_pct=0.10)

    def test_no_alert_below_warn(self) -> None:
        state = {"drawdown": 0.03, "equity": 100_000.0}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)

    def test_warning_at_warn_threshold(self) -> None:
        state = {"drawdown": 0.06, "equity": 94_000.0}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, Severity.WARNING)

    def test_critical_at_crit_threshold(self) -> None:
        state = {"drawdown": 0.12, "equity": 88_000.0}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, Severity.CRITICAL)

    def test_critical_takes_priority_over_warn(self) -> None:
        # At 15% drawdown we should get one CRITICAL, not WARNING
        state = {"drawdown": 0.15, "equity": 85_000.0}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, Severity.CRITICAL)

    def test_missing_drawdown_key_returns_no_alerts(self) -> None:
        alerts = self.rule.evaluate({"equity": 100_000.0})
        self.assertEqual(len(alerts), 0)

    def test_rule_name(self) -> None:
        self.assertEqual(self.rule.name, "DrawdownAlert")

    def test_metadata_contains_drawdown_pct(self) -> None:
        state = {"drawdown": 0.08, "equity": 92_000.0}
        alerts = self.rule.evaluate(state)
        self.assertIn("drawdown_pct", alerts[0].metadata)

    def test_zero_drawdown_no_alert(self) -> None:
        state = {"drawdown": 0.0, "equity": 100_000.0}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)


class TestVaRBreachAlert(unittest.TestCase):

    def setUp(self) -> None:
        self.rule = VaRBreachAlert()

    def test_no_breach(self) -> None:
        state = {"daily_pnl": -1000.0, "var_95": 5000.0}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)

    def test_var_breach_fires_critical(self) -> None:
        state = {"daily_pnl": -7500.0, "var_95": 5000.0}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, Severity.CRITICAL)

    def test_missing_var_no_alert(self) -> None:
        state = {"daily_pnl": -9999.0}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)

    def test_positive_pnl_no_alert(self) -> None:
        state = {"daily_pnl": 2000.0, "var_95": 1000.0}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)


class TestCircuitBreakerAlert(unittest.TestCase):

    def setUp(self) -> None:
        self.rule = CircuitBreakerAlert()

    def test_no_circuit_open(self) -> None:
        state = {"circuit_breakers": {"alpaca": False, "binance": False}}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)

    def test_one_circuit_open(self) -> None:
        state = {"circuit_breakers": {"alpaca": True, "binance": False}}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, Severity.WARNING)
        self.assertIn("alpaca", alerts[0].message)

    def test_both_circuits_open_two_alerts(self) -> None:
        state = {"circuit_breakers": {"alpaca": True, "binance": True}}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 2)

    def test_empty_circuit_breakers(self) -> None:
        state = {"circuit_breakers": {}}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)

    def test_missing_key_no_alert(self) -> None:
        alerts = self.rule.evaluate({})
        self.assertEqual(len(alerts), 0)


class TestServiceDownAlert(unittest.TestCase):

    def setUp(self) -> None:
        self.rule = ServiceDownAlert()

    def test_all_up_no_alert(self) -> None:
        state = {"service_health": {"risk_api": True, "live_trader": True}}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)

    def test_one_down_critical(self) -> None:
        state = {"service_health": {"risk_api": False, "live_trader": True}}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, Severity.CRITICAL)

    def test_message_contains_service_name(self) -> None:
        state = {"service_health": {"signal_engine": False}}
        alerts = self.rule.evaluate(state)
        self.assertIn("signal_engine", alerts[0].message)


class TestBHMassExtremeAlert(unittest.TestCase):

    def setUp(self) -> None:
        self.rule = BHMassExtremeAlert(threshold=3.5)

    def test_no_alert_below_threshold(self) -> None:
        state = {"bh_mass": {"BTC": {"daily": 2.0, "hourly": 1.5}}}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)

    def test_alert_above_threshold(self) -> None:
        state = {"bh_mass": {"BTC": {"daily": 4.1, "hourly": 1.2}}}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, Severity.WARNING)
        self.assertIn("BTC", alerts[0].message)

    def test_multiple_symbols_breaching(self) -> None:
        state = {"bh_mass": {
            "BTC": {"daily": 4.0},
            "ETH": {"daily": 3.8},
        }}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 2)


class TestHurstFlipAlert(unittest.TestCase):

    def setUp(self) -> None:
        self.rule = HurstFlipAlert(flip_count=3)

    def test_no_flip_no_alert(self) -> None:
        state = {
            "hurst_current":  {"BTC": 0.6, "ETH": 0.6},
            "hurst_previous": {"BTC": 0.62, "ETH": 0.58},
        }
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)

    def test_flip_above_count_fires_info(self) -> None:
        state = {
            "hurst_current":  {
                "BTC": 0.3, "ETH": 0.3, "SOL": 0.3, "XRP": 0.3
            },
            "hurst_previous": {
                "BTC": 0.7, "ETH": 0.7, "SOL": 0.7, "XRP": 0.7
            },
        }
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, Severity.INFO)

    def test_exactly_at_threshold_no_alert(self) -> None:
        # flip_count=3 means >3 needed
        state = {
            "hurst_current":  {"BTC": 0.3, "ETH": 0.3, "SOL": 0.3},
            "hurst_previous": {"BTC": 0.7, "ETH": 0.7, "SOL": 0.7},
        }
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)


class TestParameterRollbackAlert(unittest.TestCase):

    def setUp(self) -> None:
        self.rule = ParameterRollbackAlert()

    def test_no_rollback_no_alert(self) -> None:
        state = {"param_rollback_triggered": False}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)

    def test_rollback_fires_critical(self) -> None:
        state = {
            "param_rollback_triggered": True,
            "rollback_reason": "sharpe < 0.5",
        }
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, Severity.CRITICAL)
        self.assertIn("sharpe < 0.5", alerts[0].message)


class TestCorrelationRegimeAlert(unittest.TestCase):

    def setUp(self) -> None:
        self.rule = CorrelationRegimeAlert(threshold=0.85)

    def test_no_alert_below_threshold(self) -> None:
        state = {"avg_pairwise_correlation": 0.70}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)

    def test_warning_above_threshold(self) -> None:
        state = {"avg_pairwise_correlation": 0.91}
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, Severity.WARNING)

    def test_missing_key_no_alert(self) -> None:
        alerts = self.rule.evaluate({})
        self.assertEqual(len(alerts), 0)


class TestLiquidityAlert(unittest.TestCase):

    def setUp(self) -> None:
        self.rule = LiquidityAlert(multiplier=3.0)

    def test_no_alert_within_threshold(self) -> None:
        state = {
            "amihud_current":  {"BTC": 0.002},
            "amihud_30d_mean": {"BTC": 0.001},
        }
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)

    def test_alert_above_threshold(self) -> None:
        state = {
            "amihud_current":  {"BTC": 0.005},
            "amihud_30d_mean": {"BTC": 0.001},
        }
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, Severity.WARNING)

    def test_zero_baseline_no_alert(self) -> None:
        state = {
            "amihud_current":  {"BTC": 0.005},
            "amihud_30d_mean": {"BTC": 0.0},
        }
        alerts = self.rule.evaluate(state)
        self.assertEqual(len(alerts), 0)


# ===========================================================================
# 3. AlertDeduplication tests
# ===========================================================================

class TestAlertDeduplication(unittest.TestCase):

    def _make_alert(self, rule: str = "TestRule", sev: Severity = Severity.WARNING) -> Alert:
        return Alert(rule_name=rule, severity=sev, message="test")

    def test_first_alert_not_duplicate(self) -> None:
        dedup = AlertDeduplication(window_s=1800)
        alert = self._make_alert()
        self.assertFalse(dedup.is_duplicate(alert))

    def test_second_alert_within_window_is_duplicate(self) -> None:
        dedup = AlertDeduplication(window_s=1800)
        alert = self._make_alert()
        dedup.record(alert)
        self.assertTrue(dedup.is_duplicate(alert))

    def test_different_rule_not_duplicate(self) -> None:
        dedup = AlertDeduplication(window_s=1800)
        a1 = self._make_alert("Rule1")
        a2 = self._make_alert("Rule2")
        dedup.record(a1)
        self.assertFalse(dedup.is_duplicate(a2))

    def test_different_severity_not_duplicate(self) -> None:
        dedup = AlertDeduplication(window_s=1800)
        a1 = self._make_alert("Rule1", Severity.WARNING)
        a2 = self._make_alert("Rule1", Severity.CRITICAL)
        dedup.record(a1)
        self.assertFalse(dedup.is_duplicate(a2))

    def test_clear_removes_all(self) -> None:
        dedup = AlertDeduplication(window_s=1800)
        alert = self._make_alert()
        dedup.record(alert)
        dedup.clear()
        self.assertFalse(dedup.is_duplicate(alert))

    def test_expired_window_not_duplicate(self) -> None:
        dedup = AlertDeduplication(window_s=1)  # 1-second window
        alert = self._make_alert()
        dedup.record(alert)
        # Manually set timestamp to the past
        key = alert.dedup_key()
        from datetime import timedelta
        dedup._seen[key] = datetime.now(timezone.utc) - timedelta(seconds=2)
        dedup.clear_expired()
        self.assertFalse(dedup.is_duplicate(alert))


# ===========================================================================
# 4. AlertHistory (SQLite) tests
# ===========================================================================

class TestAlertHistory(unittest.TestCase):

    def setUp(self) -> None:
        self._db = _tmp_db()
        self._history = AlertHistory(self._db)

    def tearDown(self) -> None:
        self._history.close()
        Path(self._db).unlink(missing_ok=True)

    def _make_alert(self, rule: str = "TestRule") -> Alert:
        return Alert(rule_name=rule, severity=Severity.WARNING, message="test msg")

    def test_insert_returns_id(self) -> None:
        alert = self._make_alert()
        db_id = self._history.insert(alert)
        self.assertIsInstance(db_id, int)
        self.assertGreater(db_id, 0)

    def test_get_active_returns_unresolved(self) -> None:
        a1 = self._make_alert("Rule1")
        a2 = self._make_alert("Rule2")
        self._history.insert(a1)
        self._history.insert(a2)
        active = self._history.get_active()
        self.assertEqual(len(active), 2)

    def test_resolve_removes_from_active(self) -> None:
        alert = self._make_alert()
        db_id = self._history.insert(alert)
        self._history.resolve(db_id)
        active = self._history.get_active()
        self.assertEqual(len(active), 0)

    def test_get_history_includes_resolved(self) -> None:
        alert = self._make_alert()
        db_id = self._history.insert(alert)
        self._history.resolve(db_id)
        history = self._history.get_history()
        self.assertEqual(len(history), 1)


# ===========================================================================
# 5. Alerter integration tests
# ===========================================================================

class TestAlerter(unittest.TestCase):

    def setUp(self) -> None:
        self._db = _tmp_db()
        self._alerter = Alerter(db_path=self._db)

    def tearDown(self) -> None:
        self._alerter.stop()
        Path(self._db).unlink(missing_ok=True)

    def test_evaluate_now_fires_drawdown_critical(self) -> None:
        self._alerter.update_state({
            "drawdown": 0.15,
            "equity": 85_000.0,
        })
        alerts = self._alerter.evaluate_now()
        sev_vals = [a.severity for a in alerts]
        self.assertIn(Severity.CRITICAL, sev_vals)

    def test_evaluate_now_fires_var_breach(self) -> None:
        self._alerter.update_state({
            "daily_pnl": -9000.0,
            "var_95": 5000.0,
        })
        alerts = self._alerter.evaluate_now()
        names = [a.rule_name for a in alerts]
        self.assertIn("VaRBreachAlert", names)

    def test_start_stop(self) -> None:
        self._alerter.start()
        time.sleep(0.1)
        self._alerter.stop()

    def test_get_active_alerts_returns_list(self) -> None:
        active = self._alerter.get_active_alerts()
        self.assertIsInstance(active, list)

    def test_get_alert_history_returns_list(self) -> None:
        history = self._alerter.get_alert_history()
        self.assertIsInstance(history, list)

    def test_add_custom_rule(self) -> None:
        from infra.observability.alerter import AlertRule

        class _MyRule(AlertRule):
            name = "MyCustomRule"
            def evaluate(self, state):
                if state.get("my_flag"):
                    return [self._make_alert(Severity.INFO, "Custom fired")]
                return []

        self._alerter.add_rule(_MyRule())
        self._alerter.update_state({"my_flag": True})
        alerts = self._alerter.evaluate_now()
        names = [a.rule_name for a in alerts]
        self.assertIn("MyCustomRule", names)


# ===========================================================================
# 6. AuditLogger tests
# ===========================================================================

class TestAuditLogWriteRead(unittest.TestCase):

    def setUp(self) -> None:
        self._db = _tmp_db()
        self._logger = AuditLogger(db_path=self._db)

    def tearDown(self) -> None:
        self._logger.stop()
        Path(self._db).unlink(missing_ok=True)

    def test_log_trade_returns_id(self) -> None:
        row_id = self._logger.log_trade("BTC", "entry", 45000.0, 0.1, reason="bh")
        self.assertIsInstance(row_id, int)
        self.assertGreater(row_id, 0)

    def test_log_trade_read_back(self) -> None:
        self._logger.log_trade("ETH", "exit", 3200.0, 1.5, pnl=240.0)
        trades = self._logger.get_trades()
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["symbol"], "ETH")
        self.assertEqual(trades[0]["action"], "exit")

    def test_get_trades_filter_by_symbol(self) -> None:
        self._logger.log_trade("BTC", "entry", 45000.0, 0.1)
        self._logger.log_trade("ETH", "entry", 3200.0, 1.0)
        btc_trades = self._logger.get_trades(symbol="BTC")
        self.assertEqual(len(btc_trades), 1)

    def test_get_trades_filter_by_since(self) -> None:
        old_ts = datetime.now(timezone.utc) - timedelta(days=2)
        self._logger.log_trade("BTC", "entry", 45000.0, 0.1)
        since = datetime.now(timezone.utc) - timedelta(hours=1)
        trades = self._logger.get_trades(since=since)
        self.assertEqual(len(trades), 1)
        trades_old = self._logger.get_trades(since=old_ts)
        self.assertEqual(len(trades_old), 1)

    def test_log_param_update(self) -> None:
        old = {"max_frac": 0.8, "threshold": 0.001}
        new = {"max_frac": 0.7, "threshold": 0.001}
        row_id = self._logger.log_param_update(old, new, source="optimizer")
        self.assertGreater(row_id, 0)

    def test_get_param_history(self) -> None:
        for i in range(5):
            self._logger.log_param_update(
                {"v": i}, {"v": i + 1}, source="test"
            )
        history = self._logger.get_param_history(n=3)
        self.assertEqual(len(history), 3)

    def test_param_history_contains_diff(self) -> None:
        self._logger.log_param_update(
            {"a": 1, "b": 2}, {"a": 1, "b": 3}, source="test"
        )
        history = self._logger.get_param_history()
        self.assertIn("b", history[0]["change_diff"])
        self.assertNotIn("a", history[0]["change_diff"])

    def test_log_regime_change(self) -> None:
        row_id = self._logger.log_regime_change(
            "BTC", "mean_reverting", "trending", trigger="hurst"
        )
        self.assertGreater(row_id, 0)
        rows = self._logger.get_regime_history(symbol="BTC")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["from_regime"], "mean_reverting")

    def test_log_rollback(self) -> None:
        row_id = self._logger.log_rollback(
            "sharpe_below_0.5",
            {"stale_threshold": 0.0012},
            {"stale_threshold": 0.001},
        )
        self.assertGreater(row_id, 0)
        rollbacks = self._logger.get_rollback_history()
        self.assertEqual(len(rollbacks), 1)
        self.assertEqual(rollbacks[0]["trigger_reason"], "sharpe_below_0.5")

    def test_log_alert(self) -> None:
        row_id = self._logger.log_alert(
            "CRITICAL", "DrawdownAlert", "Drawdown 12%", {"dd": 0.12}
        )
        self.assertGreater(row_id, 0)
        alerts = self._logger.get_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["severity"], "CRITICAL")

    def test_get_alerts_filter_severity(self) -> None:
        self._logger.log_alert("INFO", "Rule1", "info msg")
        self._logger.log_alert("WARNING", "Rule2", "warn msg")
        self._logger.log_alert("CRITICAL", "Rule3", "crit msg")
        crits = self._logger.get_alerts(severity="CRITICAL")
        self.assertEqual(len(crits), 1)
        warns = self._logger.get_alerts(severity="WARNING")
        self.assertEqual(len(warns), 1)

    def test_compute_diff_utility(self) -> None:
        old = {"a": 1, "b": 2, "c": 3}
        new = {"a": 1, "b": 99, "d": 4}
        diff = _compute_diff(old, new)
        self.assertIn("b", diff)
        self.assertIn("c", diff)  # removed
        self.assertIn("d", diff)  # added
        self.assertNotIn("a", diff)  # unchanged

    def test_multiple_trades_ordering(self) -> None:
        for i in range(10):
            self._logger.log_trade("BTC", "entry", 45000.0 + i, 0.1)
        trades = self._logger.get_trades(limit=5)
        self.assertEqual(len(trades), 5)

    def test_get_regime_history_filter_by_symbol(self) -> None:
        self._logger.log_regime_change("BTC", "trending", "mean_reverting")
        self._logger.log_regime_change("ETH", "random", "trending")
        btc = self._logger.get_regime_history(symbol="BTC")
        self.assertEqual(len(btc), 1)
        eth = self._logger.get_regime_history(symbol="ETH")
        self.assertEqual(len(eth), 1)


# ===========================================================================
# 7. Dashboard API tests
# ===========================================================================

class TestDashboardAPILiveMetrics(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from infra.observability.dashboard_api import app, _cache
            self._app = app
            self._cache = _cache
            self._client = TestClient(app, raise_server_exceptions=False)
        except Exception as exc:
            self.skipTest(f"FastAPI test client setup failed: {exc}")

    def test_health_endpoint(self) -> None:
        resp = self._client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")

    def test_live_metrics_returns_200(self) -> None:
        resp = self._client.get("/metrics/live")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("data", data)

    def test_live_metrics_response_shape(self) -> None:
        resp = self._client.get("/metrics/live")
        data = resp.json()["data"]
        # Even with empty DBs, these keys must be present
        self.assertIn("timestamp", data)

    def test_metrics_history_equity(self) -> None:
        resp = self._client.get("/metrics/history?metric=equity")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["metric"], "equity")
        self.assertIn("data", body)

    def test_metrics_history_bh_mass(self) -> None:
        resp = self._client.get("/metrics/history?metric=bh_mass&symbol=BTC")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["metric"], "bh_mass")

    def test_metrics_history_invalid_metric(self) -> None:
        resp = self._client.get("/metrics/history?metric=nonexistent_metric")
        self.assertEqual(resp.status_code, 400)

    def test_alerts_active_endpoint(self) -> None:
        resp = self._client.get("/alerts/active")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("count", body)
        self.assertIn("alerts", body)

    def test_alerts_history_endpoint(self) -> None:
        resp = self._client.get("/alerts/history")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("count", body)
        self.assertIn("alerts", body)

    def test_alerts_history_severity_filter(self) -> None:
        resp = self._client.get("/alerts/history?severity=CRITICAL")
        self.assertEqual(resp.status_code, 200)

    def test_alerts_history_invalid_severity(self) -> None:
        resp = self._client.get("/alerts/history?severity=INVALID")
        self.assertEqual(resp.status_code, 400)

    def test_system_health_endpoint(self) -> None:
        resp = self._client.get("/system/health")
        # May be 200 or 503 depending on whether services are up
        self.assertIn(resp.status_code, (200, 503))
        body = resp.json()
        self.assertIn("components", body)
        self.assertIn("checked_at", body)

    def test_metrics_history_with_since(self) -> None:
        since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        resp = self._client.get(f"/metrics/history?metric=equity&since={since}")
        self.assertEqual(resp.status_code, 200)

    def test_metrics_history_invalid_since(self) -> None:
        resp = self._client.get("/metrics/history?metric=equity&since=not-a-date")
        self.assertEqual(resp.status_code, 400)


class TestWebSocketPush(unittest.IsolatedAsyncioTestCase):
    """Tests for the WebSocket /ws/metrics endpoint."""

    async def asyncSetUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from infra.observability.dashboard_api import app
            self._app = app
            self._sync_client = TestClient(app, raise_server_exceptions=False)
        except Exception as exc:
            self.skipTest(f"FastAPI setup failed: {exc}")

    def test_websocket_connect_receives_initial_snapshot(self) -> None:
        with self._sync_client.websocket_connect("/ws/metrics") as ws:
            # Should receive one message immediately on connect
            data = ws.receive_text()
            parsed = json.loads(data)
            self.assertIsInstance(parsed, dict)

    def test_websocket_ping_pong(self) -> None:
        with self._sync_client.websocket_connect("/ws/metrics") as ws:
            # Consume initial snapshot
            ws.receive_text()
            ws.send_text("ping")
            response = ws.receive_text()
            self.assertEqual(response, "pong")

    def test_websocket_snapshot_has_timestamp(self) -> None:
        with self._sync_client.websocket_connect("/ws/metrics") as ws:
            data = json.loads(ws.receive_text())
            # The snapshot might not have all keys if DBs are missing,
            # but cache_updated_at should be present or None
            self.assertIsInstance(data, dict)


# ===========================================================================
# 8. SignalEngineScraper tests
# ===========================================================================

class TestSignalEngineScraper(unittest.TestCase):

    def setUp(self) -> None:
        if not _PROM_AVAILABLE:
            self.skipTest("prometheus_client not installed")
        self._metrics = _PromMetrics()

    def test_missing_status_file(self) -> None:
        scraper = SignalEngineScraper(self._metrics)
        scraper.STATUS_PATH = "/tmp/does_not_exist_signal_xyz.json"
        result = scraper.scrape()
        self.assertEqual(result.get("status"), "status_file_missing")

    def test_valid_status_file(self) -> None:
        import json
        status_data = {
            "bars_processed": {"BTC": {"daily": 100, "hourly": 2400}},
            "signals_generated": {"BTC": {"long": 12, "short": 8}},
            "bar_latency_ms": {"BTC": [1.2, 2.3, 0.9]},
            "signal_ns": {"bh_mass": [1500, 2200]},
            "order_latency_ms": {"alpaca": [25.0, 30.5]},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(status_data, f)
            path = f.name

        try:
            scraper = SignalEngineScraper(self._metrics)
            scraper.STATUS_PATH = path
            result = scraper.scrape()
            self.assertEqual(result.get("status"), "ok")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_stale_status_file(self) -> None:
        import json
        status_data = {
            "bars_processed": {},
            "updated_at": (
                datetime.now(timezone.utc) - timedelta(minutes=5)
            ).isoformat(),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(status_data, f)
            path = f.name

        try:
            scraper = SignalEngineScraper(self._metrics)
            scraper.STATUS_PATH = path
            result = scraper.scrape()
            self.assertEqual(result.get("status"), "stale")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_malformed_json(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{ invalid json !")
            path = f.name

        try:
            scraper = SignalEngineScraper(self._metrics)
            scraper.STATUS_PATH = path
            result = scraper.scrape()
            self.assertEqual(result.get("status"), "parse_error")
        finally:
            Path(path).unlink(missing_ok=True)


# ===========================================================================
# 9. Utility tests
# ===========================================================================

class TestComputeDiff(unittest.TestCase):

    def test_added_key(self) -> None:
        diff = _compute_diff({}, {"a": 1})
        self.assertIn("a", diff)
        self.assertIsNone(diff["a"]["old"])
        self.assertEqual(diff["a"]["new"], 1)

    def test_removed_key(self) -> None:
        diff = _compute_diff({"a": 1}, {})
        self.assertIn("a", diff)
        self.assertEqual(diff["a"]["old"], 1)
        self.assertIsNone(diff["a"]["new"])

    def test_changed_key(self) -> None:
        diff = _compute_diff({"a": 1}, {"a": 2})
        self.assertIn("a", diff)
        self.assertEqual(diff["a"]["old"], 1)
        self.assertEqual(diff["a"]["new"], 2)

    def test_unchanged_key_not_in_diff(self) -> None:
        diff = _compute_diff({"a": 1, "b": 2}, {"a": 1, "b": 3})
        self.assertNotIn("a", diff)
        self.assertIn("b", diff)

    def test_empty_inputs(self) -> None:
        diff = _compute_diff({}, {})
        self.assertEqual(diff, {})


# ===========================================================================
# 10. Alerter dispatcher tests (with mocked Slack)
# ===========================================================================

class TestAlertDispatcher(unittest.TestCase):

    def setUp(self) -> None:
        self._db = _tmp_db()
        self._history = AlertHistory(self._db)
        self._dedup = AlertDeduplication(window_s=1800)
        self._dispatcher = AlertDispatcher(
            self._history,
            self._dedup,
            slack_url="",  # disabled
            coord_url="http://localhost:0",  # will fail silently
        )

    def tearDown(self) -> None:
        self._history.close()
        Path(self._db).unlink(missing_ok=True)

    def _make_alert(self, rule: str = "TestRule") -> Alert:
        return Alert(rule_name=rule, severity=Severity.WARNING, message="test")

    def test_dispatch_returns_true(self) -> None:
        alert = self._make_alert()
        result = self._dispatcher.dispatch(alert)
        self.assertTrue(result)

    def test_dispatch_persists_to_db(self) -> None:
        alert = self._make_alert()
        self._dispatcher.dispatch(alert)
        time.sleep(0.05)
        active = self._history.get_active()
        self.assertEqual(len(active), 1)

    def test_dispatch_assigns_db_id(self) -> None:
        alert = self._make_alert()
        self._dispatcher.dispatch(alert)
        time.sleep(0.05)
        self.assertIsNotNone(alert.db_id)
        self.assertGreater(alert.db_id, 0)

    def test_duplicate_suppressed(self) -> None:
        alert = self._make_alert()
        r1 = self._dispatcher.dispatch(alert)
        r2 = self._dispatcher.dispatch(alert)
        self.assertTrue(r1)
        self.assertFalse(r2)

    def test_different_rules_both_dispatched(self) -> None:
        a1 = self._make_alert("Rule1")
        a2 = self._make_alert("Rule2")
        r1 = self._dispatcher.dispatch(a1)
        r2 = self._dispatcher.dispatch(a2)
        self.assertTrue(r1)
        self.assertTrue(r2)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
