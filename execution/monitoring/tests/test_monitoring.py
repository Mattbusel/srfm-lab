"""
Tests for execution/monitoring -- circuit breaker, order monitor,
position reconciler, and performance degradation alerts.
Run with: pytest execution/monitoring/tests/test_monitoring.py -v
"""

from __future__ import annotations

import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from execution.monitoring.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    circuit_protected,
)
from execution.monitoring.order_monitor import (
    FillRecord,
    OrderAlertEngine,
    OrderMonitor,
    OrderRecord,
)
from execution.monitoring.performance_monitor import (
    DegradationAlert,
    LivePerformanceMonitor,
    NAVPoint,
    PerformanceDegradationAlert,
    TradeRecord,
)
from execution.monitoring.position_reconciler import (
    Discrepancy,
    PositionReconciler,
    ReconciliationResult,
    ReconciliationStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_order(
    order_id: str = "O001",
    symbol: str = "AAPL",
    side: str = "buy",
    qty: float = 100.0,
    price: float = 150.0,
    strategy_id: str = "strat_01",
    offset_seconds: float = 0.0,
) -> OrderRecord:
    submitted = datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)
    return OrderRecord(
        order_id=order_id,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        order_type="limit",
        strategy_id=strategy_id,
        submitted_at=submitted,
    )


def _make_fill(
    fill_id: str = "F001",
    qty: float = 100.0,
    price: float = 150.0,
    venue: str = "alpaca",
) -> FillRecord:
    return FillRecord(
        fill_id=fill_id,
        qty=qty,
        price=price,
        timestamp=datetime.now(timezone.utc),
        venue=venue,
    )


def _make_trade(
    trade_id: str = "T001",
    symbol: str = "AAPL",
    net_pnl: float = 100.0,
    duration_bars: float = 4.0,
) -> TradeRecord:
    entry = datetime(2025, 1, 6, 10, 0, tzinfo=timezone.utc)
    exit_dt = entry + timedelta(seconds=int(duration_bars * 900))
    side = "buy"
    qty = 10.0
    entry_price = 100.0
    exit_price = entry_price + net_pnl / qty
    return TradeRecord(
        trade_id=trade_id,
        symbol=symbol,
        side=side,
        qty=qty,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=entry,
        exit_time=exit_dt,
        strategy_id="strat_01",
        pnl=net_pnl,
        commission=0.0,
    )


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerStateTransitions(unittest.TestCase):

    def _make_cb(
        self,
        threshold: int = 3,
        window: float = 60.0,
        timeout: float = 0.1,   # short timeout for tests
        probe_threshold: int = 2,
    ) -> CircuitBreaker:
        cfg = CircuitBreakerConfig(
            failure_threshold=threshold,
            window_seconds=window,
            timeout_seconds=timeout,
            probe_success_threshold=probe_threshold,
        )
        return CircuitBreaker("test_cb", cfg)

    def test_initial_state_is_closed(self) -> None:
        cb = self._make_cb()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_failures_below_threshold_stay_closed(self) -> None:
        cb = self._make_cb(threshold=3)
        cb.record_failure(ValueError("err"))
        cb.record_failure(ValueError("err"))
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_failures_at_threshold_open_circuit(self) -> None:
        cb = self._make_cb(threshold=3)
        for _ in range(3):
            cb.record_failure(ValueError("err"))
        self.assertEqual(cb.state, CircuitState.OPEN)

    def test_open_circuit_raises_on_call(self) -> None:
        cb = self._make_cb(threshold=1)
        cb.record_failure(RuntimeError("fail"))
        with self.assertRaises(CircuitOpenError):
            cb.call(lambda: None)

    def test_open_transitions_to_half_open_after_timeout(self) -> None:
        cb = self._make_cb(threshold=1, timeout=0.05)
        cb.record_failure(RuntimeError("fail"))
        self.assertEqual(cb.state, CircuitState.OPEN)
        time.sleep(0.1)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)

    def test_half_open_success_closes_circuit(self) -> None:
        cb = self._make_cb(threshold=1, timeout=0.05, probe_threshold=2)
        cb.record_failure(RuntimeError("fail"))
        time.sleep(0.1)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        cb.record_success()
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_half_open_failure_reopens_circuit(self) -> None:
        cb = self._make_cb(threshold=1, timeout=0.05, probe_threshold=3)
        cb.record_failure(RuntimeError("fail"))
        time.sleep(0.1)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        cb.record_failure(RuntimeError("probe fail"))
        self.assertEqual(cb.state, CircuitState.OPEN)

    def test_manual_reset_closes_open_circuit(self) -> None:
        cb = self._make_cb(threshold=1)
        cb.record_failure(RuntimeError("fail"))
        self.assertEqual(cb.state, CircuitState.OPEN)
        cb.reset()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_call_passes_through_when_closed(self) -> None:
        cb = self._make_cb()
        result = cb.call(lambda x: x * 2, 5)
        self.assertEqual(result, 10)

    def test_call_records_failure_on_exception(self) -> None:
        cb = self._make_cb(threshold=5)
        def failing_fn():
            raise ValueError("boom")
        with self.assertRaises(ValueError):
            cb.call(failing_fn)
        self.assertEqual(cb.status()["recent_failures"], 1)

    def test_failures_outside_window_evicted(self) -> None:
        cb = self._make_cb(threshold=3, window=0.05)
        cb.record_failure(ValueError("old"))
        cb.record_failure(ValueError("old"))
        time.sleep(0.1)
        # these two new failures should not combine with the old ones
        cb.record_failure(ValueError("new"))
        cb.record_failure(ValueError("new"))
        # still below threshold after eviction
        self.assertEqual(cb.state, CircuitState.CLOSED)


class TestCircuitBreakerRegistry(unittest.TestCase):

    def setUp(self) -> None:
        # use a fresh registry per test
        self.registry = CircuitBreakerRegistry()

    def test_default_breakers_registered(self) -> None:
        names = self.registry.list_names()
        for expected in ("alpaca_orders", "alpaca_data", "binance_orders", "database", "coordination"):
            self.assertIn(expected, names)

    def test_get_or_create_returns_same_instance(self) -> None:
        cb1 = self.registry.get_or_create("alpaca_orders")
        cb2 = self.registry.get_or_create("alpaca_orders")
        self.assertIs(cb1, cb2)

    def test_status_all_returns_state_strings(self) -> None:
        statuses = self.registry.status_all()
        for state in statuses.values():
            self.assertIn(state, ("CLOSED", "OPEN", "HALF_OPEN"))

    def test_reset_closes_open_circuit(self) -> None:
        cb = self.registry.get_or_create("database")
        cfg = CircuitBreakerConfig(failure_threshold=1, window_seconds=60.0, timeout_seconds=120.0)
        cb2 = self.registry.register("test_reset_db", cfg)
        cb2.record_failure(RuntimeError("db down"))
        self.assertEqual(cb2.state, CircuitState.OPEN)
        self.registry.reset("test_reset_db")
        self.assertEqual(cb2.state, CircuitState.CLOSED)

    def test_reset_all_closes_all_circuits(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, window_seconds=60.0, timeout_seconds=120.0)
        for name in ("ra_1", "ra_2"):
            cb = self.registry.register(name, cfg)
            cb.record_failure(RuntimeError("err"))
        self.registry.reset_all()
        for name in ("ra_1", "ra_2"):
            self.assertEqual(self.registry.get_or_create(name).state, CircuitState.CLOSED)


# ---------------------------------------------------------------------------
# Order monitor tests
# ---------------------------------------------------------------------------

class TestOrderMonitor(unittest.TestCase):

    def setUp(self) -> None:
        self.monitor = OrderMonitor()

    def test_track_order_adds_to_pending(self) -> None:
        order = _make_order("O1")
        self.monitor.track_order(order)
        self.assertEqual(len(self.monitor.pending_orders()), 1)

    def test_on_fill_completes_order(self) -> None:
        order = _make_order("O2", qty=100.0)
        self.monitor.track_order(order)
        fill = _make_fill("F2", qty=100.0)
        self.monitor.on_fill("O2", fill)
        self.assertEqual(len(self.monitor.pending_orders()), 0)
        o = self.monitor.get_order("O2")
        self.assertIsNotNone(o)
        self.assertEqual(o.status, "filled")  # type: ignore[union-attr]

    def test_partial_fill_stays_pending(self) -> None:
        order = _make_order("O3", qty=100.0)
        self.monitor.track_order(order)
        fill = _make_fill("F3", qty=50.0)
        self.monitor.on_fill("O3", fill)
        self.assertEqual(len(self.monitor.pending_orders()), 1)
        self.assertEqual(self.monitor.get_order("O3").status, "partial")  # type: ignore[union-attr]

    def test_on_cancel_removes_from_pending(self) -> None:
        order = _make_order("O4")
        self.monitor.track_order(order)
        self.monitor.on_cancel("O4", "user cancelled")
        self.assertEqual(len(self.monitor.pending_orders()), 0)
        self.assertEqual(self.monitor.get_order("O4").status, "cancelled")  # type: ignore[union-attr]

    def test_on_reject_records_status(self) -> None:
        order = _make_order("O5")
        self.monitor.track_order(order)
        self.monitor.on_reject("O5", "insufficient funds")
        o = self.monitor.get_order("O5")
        self.assertEqual(o.status, "rejected")  # type: ignore[union-attr]
        self.assertEqual(o.rejection_reason, "insufficient funds")  # type: ignore[union-attr]

    def test_stale_order_detection(self) -> None:
        fresh = _make_order("O6", offset_seconds=5.0)
        old = _make_order("O7", offset_seconds=60.0)
        self.monitor.track_order(fresh)
        self.monitor.track_order(old)
        stale = self.monitor.stale_orders(max_age_seconds=30.0)
        ids = [o.order_id for o in stale]
        self.assertIn("O7", ids)
        self.assertNotIn("O6", ids)

    def test_fill_rate_calculation(self) -> None:
        for i in range(8):
            o = _make_order(f"FR{i}", qty=10.0)
            self.monitor.track_order(o)
            self.monitor.on_fill(f"FR{i}", _make_fill(f"F{i}", qty=10.0))
        for i in range(2):
            o = _make_order(f"FR_rej{i}", qty=10.0)
            self.monitor.track_order(o)
            self.monitor.on_reject(f"FR_rej{i}", "test")
        rate = self.monitor.fill_rate(window_minutes=60)
        self.assertAlmostEqual(rate, 0.8, places=5)

    def test_rejection_rate_calculation(self) -> None:
        for i in range(3):
            o = _make_order(f"RR{i}", qty=10.0)
            self.monitor.track_order(o)
            self.monitor.on_reject(f"RR{i}", "test reject")
        for i in range(7):
            o = _make_order(f"RR_ok{i}", qty=10.0)
            self.monitor.track_order(o)
            self.monitor.on_fill(f"RR_ok{i}", _make_fill(f"RF{i}", qty=10.0))
        rate = self.monitor.rejection_rate(window_minutes=60)
        self.assertAlmostEqual(rate, 0.3, places=5)


# ---------------------------------------------------------------------------
# Position reconciler tests
# ---------------------------------------------------------------------------

class TestPositionReconciler(unittest.TestCase):

    def setUp(self) -> None:
        self.store = ReconciliationStore(db_path=":memory:")
        self.reconciler = PositionReconciler(store=self.store)

    def test_matched_positions(self) -> None:
        internal = {"AAPL": 100.0, "MSFT": 200.0}
        broker = {"AAPL": 100.0, "MSFT": 200.0}
        result = self.reconciler.reconcile(internal, broker)
        self.assertEqual(result.status, "MATCHED")
        self.assertEqual(result.discrepancy_count, 0)

    def test_info_discrepancy_auto_corrected(self) -> None:
        corrections = []
        def on_correction(symbol, old, new):
            corrections.append((symbol, old, new))
        reconciler = PositionReconciler(store=self.store, on_correction=on_correction)
        # 0.5% difference -- should be INFO
        internal = {"AAPL": 1000.0}
        broker = {"AAPL": 1005.0}
        result = reconciler.reconcile(internal, broker)
        self.assertEqual(result.status, "DISCREPANCY")
        self.assertEqual(result.discrepancies[0].severity, "INFO")
        self.assertEqual(len(corrections), 1)

    def test_warning_discrepancy(self) -> None:
        # 3% difference -- WARNING
        internal = {"MSFT": 100.0}
        broker = {"MSFT": 103.0}
        result = self.reconciler.reconcile(internal, broker)
        self.assertEqual(result.discrepancies[0].severity, "WARNING")

    def test_critical_discrepancy_halts_symbol(self) -> None:
        halt_calls = []
        def on_halt(symbol, disc):
            halt_calls.append(symbol)
        reconciler = PositionReconciler(store=self.store, on_halt_symbol=on_halt)
        # 10% difference -- CRITICAL
        internal = {"TSLA": 100.0}
        broker = {"TSLA": 110.0}
        result = reconciler.reconcile(internal, broker)
        self.assertEqual(result.discrepancies[0].severity, "CRITICAL")
        self.assertIn("TSLA", halt_calls)
        self.assertTrue(reconciler.is_halted("TSLA"))

    def test_reconciliation_history_persisted(self) -> None:
        for _ in range(3):
            self.reconciler.reconcile({"X": 10.0}, {"X": 10.0})
        history = self.reconciler.reconciliation_history(n=10)
        self.assertGreaterEqual(len(history), 3)

    def test_last_reconciliation_time_updated(self) -> None:
        self.assertIsNone(self.reconciler.last_reconciliation_time())
        self.reconciler.reconcile({"A": 1.0}, {"A": 1.0})
        self.assertIsNotNone(self.reconciler.last_reconciliation_time())

    def test_discrepancy_severity_boundaries(self) -> None:
        # exactly at 1% boundary
        self.assertEqual(Discrepancy.compute_severity(0.99), "INFO")
        self.assertEqual(Discrepancy.compute_severity(1.0), "WARNING")
        self.assertEqual(Discrepancy.compute_severity(4.99), "WARNING")
        self.assertEqual(Discrepancy.compute_severity(5.0), "CRITICAL")


# ---------------------------------------------------------------------------
# Performance monitor tests
# ---------------------------------------------------------------------------

class TestLivePerformanceMonitor(unittest.TestCase):

    def setUp(self) -> None:
        self.monitor = LivePerformanceMonitor()

    def _add_nav_series(self, values, base_time=None) -> None:
        if base_time is None:
            base_time = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)
        for i, v in enumerate(values):
            ts = base_time + timedelta(minutes=15 * i)
            self.monitor.update_nav(v, ts)

    def test_win_rate_all_wins(self) -> None:
        for i in range(10):
            self.monitor.record_trade(_make_trade(f"T{i}", net_pnl=50.0))
        self.assertAlmostEqual(self.monitor.win_rate(10), 1.0)

    def test_win_rate_mixed(self) -> None:
        for i in range(6):
            self.monitor.record_trade(_make_trade(f"W{i}", net_pnl=50.0))
        for i in range(4):
            self.monitor.record_trade(_make_trade(f"L{i}", net_pnl=-30.0))
        self.assertAlmostEqual(self.monitor.win_rate(10), 0.6)

    def test_current_streak_win(self) -> None:
        for i in range(3):
            self.monitor.record_trade(_make_trade(f"W{i}", net_pnl=10.0))
        kind, count = self.monitor.current_streak()
        self.assertEqual(kind, "win")
        self.assertEqual(count, 3)

    def test_current_streak_loss(self) -> None:
        self.monitor.record_trade(_make_trade("W1", net_pnl=10.0))
        for i in range(4):
            self.monitor.record_trade(_make_trade(f"L{i}", net_pnl=-10.0))
        kind, count = self.monitor.current_streak()
        self.assertEqual(kind, "loss")
        self.assertEqual(count, 4)

    def test_rolling_sharpe_returns_float(self) -> None:
        navs = [1000.0 + i * 0.5 + (i % 3) * 0.1 for i in range(100)]
        self._add_nav_series(navs)
        sharpe = self.monitor.rolling_sharpe(96)
        self.assertIsInstance(sharpe, float)

    def test_rolling_vol_insufficient_data(self) -> None:
        self._add_nav_series([1000.0, 1001.0])
        self.assertEqual(self.monitor.rolling_vol(20), 0.0)

    def test_pnl_by_hour_groups_correctly(self) -> None:
        base = datetime(2025, 1, 6, tzinfo=timezone.utc)
        for h in [9, 10, 9]:
            entry = base.replace(hour=h)
            exit_dt = entry + timedelta(hours=1)
            t = TradeRecord(
                trade_id=f"H{h}",
                symbol="AAPL",
                side="buy",
                qty=10.0,
                entry_price=100.0,
                exit_price=110.0,
                entry_time=entry,
                exit_time=exit_dt,
                strategy_id="s",
                pnl=100.0,
            )
            self.monitor.record_trade(t)
        by_hour = self.monitor.pnl_by_hour()
        self.assertIn(9, by_hour)
        self.assertIn(10, by_hour)
        self.assertAlmostEqual(by_hour[9], 200.0)
        self.assertAlmostEqual(by_hour[10], 100.0)


class TestPerformanceDegradationAlert(unittest.TestCase):

    def setUp(self) -> None:
        self.monitor = LivePerformanceMonitor()
        self.alerter = PerformanceDegradationAlert()

    def test_no_alerts_when_performance_good(self) -> None:
        for i in range(15):
            self.monitor.record_trade(_make_trade(f"T{i}", net_pnl=50.0))
        navs = [1000.0 + i * 1.0 for i in range(100)]
        base = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)
        for i, v in enumerate(navs):
            self.monitor.update_nav(v, base + timedelta(minutes=15 * i))
        alerts = self.alerter.check(self.monitor)
        self.assertEqual(len(alerts), 0)

    def test_low_win_rate_triggers_warning(self) -> None:
        for i in range(10):
            self.monitor.record_trade(_make_trade(f"W{i}", net_pnl=-50.0))
        alerts = self.alerter.check(self.monitor)
        metrics = [a.metric for a in alerts]
        self.assertIn("win_rate_50", metrics)
        win_alert = next(a for a in alerts if a.metric == "win_rate_50")
        self.assertEqual(win_alert.severity, "WARNING")

    def test_consecutive_losses_triggers_critical(self) -> None:
        for i in range(6):
            self.monitor.record_trade(_make_trade(f"L{i}", net_pnl=-100.0))
        alerts = self.alerter.check(self.monitor)
        metrics = [a.metric for a in alerts]
        self.assertIn("consecutive_losses", metrics)
        streak_alert = next(a for a in alerts if a.metric == "consecutive_losses")
        self.assertEqual(streak_alert.severity, "CRITICAL")

    def test_negative_sharpe_triggers_critical(self) -> None:
        # Build a declining NAV series to produce negative Sharpe
        base = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)
        navs = [1000.0 - i * 2.0 for i in range(100)]
        for i, v in enumerate(navs):
            self.monitor.update_nav(v, base + timedelta(minutes=15 * i))
        alerts = self.alerter.check(self.monitor)
        metrics = [a.metric for a in alerts]
        self.assertIn("rolling_sharpe_96", metrics)
        sharpe_alert = next(a for a in alerts if a.metric == "rolling_sharpe_96")
        self.assertEqual(sharpe_alert.severity, "CRITICAL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
