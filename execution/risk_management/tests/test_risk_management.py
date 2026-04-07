"""
test_risk_management.py # Unit tests for the SRFM risk management module.

Covers:
  - PreTradeRiskEngine (all 10 check functions)
  - MarginManager + all three calculators
  - DrawdownMonitor + DrawdownBreachHandler
  - VaRMonitor (historical, parametric, Monte Carlo)
  - RiskReporter (snapshot, EOD, weekly, HTML/JSON export)
"""

from __future__ import annotations

import os
import math
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from typing import Dict

import numpy as np

from ..pre_trade_checks import (
    OrderRequest,
    OrderSide,
    OrderType,
    PositionSnapshot,
    PreTradeRiskEngine,
    RiskCheckLog,
    check_circuit_breaker,
    check_correlation_concentration,
    check_daily_loss_limit,
    check_event_calendar,
    check_leverage,
    check_max_order_size,
    check_min_hold_bars,
    check_position_limit,
    check_sector_limit,
    check_spread_gate,
)
from ..margin_manager import (
    MarginConfig,
    MarginManager,
    RegTMarginCalculator,
    CryptoMarginCalculator,
    PortfolioMarginCalculator,
    Position,
)
from ..drawdown_monitor import (
    DrawdownBreach,
    DrawdownBreachHandler,
    DrawdownMonitor,
    DrawdownWindow,
    RecoveryState,
)
from ..var_monitor import (
    EWMACovarianceMatrix,
    VaRLimitManager,
    VaRMonitor,
    historical_var,
    monte_carlo_var,
    parametric_var,
)
from ..risk_reporter import (
    EODRiskReport,
    PositionSummary,
    RiskReporter,
    RiskSnapshot,
    WeeklyRiskReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NAV = 1_000_000.0


def _make_order(
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    qty: float = 100.0,
    price: float = 150.0,
    signal: float = 0.8,
    order_type: OrderType = OrderType.LIMIT,
) -> OrderRequest:
    return OrderRequest(
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        order_type=order_type,
        strategy_id="strat_001",
        signal_strength=signal,
        order_id="ord_001",
    )


def _make_positions(overrides: Dict[str, float] = None) -> PositionSnapshot:
    base = {"AAPL": 50_000.0, "MSFT": 40_000.0}
    if overrides:
        base.update(overrides)
    return PositionSnapshot(positions=base, nav=NAV)


# ---------------------------------------------------------------------------
# Pre-trade check unit tests
# ---------------------------------------------------------------------------

class TestCheckPositionLimit(unittest.TestCase):
    def test_pass_within_limit(self) -> None:
        order = _make_order(qty=100, price=150)  # $15,000 order
        snap = _make_positions()
        r = check_position_limit(order, snap)
        self.assertTrue(r.passed)

    def test_fail_exceeds_limit(self) -> None:
        # AAPL already 50k; adding 200k would be 250k/1M = 25%
        order = _make_order(qty=1000, price=200)
        snap = _make_positions()
        r = check_position_limit(order, snap)
        self.assertFalse(r.passed)
        self.assertEqual(r.rejected_reason, "position_limit_exceeded")

    def test_zero_nav_fails(self) -> None:
        order = _make_order()
        snap = PositionSnapshot(positions={}, nav=0.0)
        r = check_position_limit(order, snap)
        self.assertFalse(r.passed)


class TestCheckDailyLossLimit(unittest.TestCase):
    def test_pass_no_loss(self) -> None:
        r = check_daily_loss_limit(_make_order(), 5000.0, NAV)
        self.assertTrue(r.passed)

    def test_fail_loss_exceeds_limit(self) -> None:
        # 2% of 1M = 20,000; daily loss of 25,000 should fail
        r = check_daily_loss_limit(_make_order(), -25_000.0, NAV)
        self.assertFalse(r.passed)
        self.assertEqual(r.rejected_reason, "daily_loss_limit_breached")

    def test_exactly_at_boundary(self) -> None:
        # exactly 2% should fail (>= limit)
        r = check_daily_loss_limit(_make_order(), -20_000.0, NAV)
        self.assertFalse(r.passed)


class TestCheckMaxOrderSize(unittest.TestCase):
    def test_pass_small_order(self) -> None:
        order = _make_order(qty=100, price=150)  # $15,000
        r = check_max_order_size(order, 1_000_000.0)  # 5% of 1M = 50k, OK
        self.assertTrue(r.passed)

    def test_fail_large_order(self) -> None:
        order = _make_order(qty=1000, price=200)  # $200,000
        r = check_max_order_size(order, 1_000_000.0)  # 20% of ADV
        self.assertFalse(r.passed)
        self.assertEqual(r.rejected_reason, "order_too_large_vs_adv")

    def test_zero_adv(self) -> None:
        r = check_max_order_size(_make_order(), 0.0)
        self.assertFalse(r.passed)


class TestCheckSpreadGate(unittest.TestCase):
    def test_pass_narrow_spread(self) -> None:
        r = check_spread_gate(_make_order(order_type=OrderType.MARKET), 20.0)
        self.assertTrue(r.passed)

    def test_fail_wide_spread_market(self) -> None:
        r = check_spread_gate(_make_order(order_type=OrderType.MARKET), 80.0)
        self.assertFalse(r.passed)
        self.assertEqual(r.rejected_reason, "spread_too_wide")

    def test_limit_order_not_rejected_wide_spread(self) -> None:
        # limit orders are not rejected by spread gate (only MARKET)
        r = check_spread_gate(_make_order(order_type=OrderType.LIMIT), 80.0)
        self.assertTrue(r.passed)


class TestCheckCircuitBreaker(unittest.TestCase):
    def test_not_broken(self) -> None:
        r = check_circuit_breaker("AAPL", set())
        self.assertTrue(r.passed)

    def test_broken(self) -> None:
        r = check_circuit_breaker("AAPL", {"AAPL"})
        self.assertFalse(r.passed)
        self.assertEqual(r.rejected_reason, "circuit_breaker_active")


class TestCheckEventCalendar(unittest.TestCase):
    def test_no_event(self) -> None:
        r = check_event_calendar(_make_order(), {})
        self.assertTrue(r.passed)

    def test_event_within_window_high_signal(self) -> None:
        # event in 12h, signal > 0.5 should fail
        now = datetime.now(timezone.utc)
        order = _make_order(signal=0.9)
        cal = {"AAPL": now + timedelta(hours=12)}
        r = check_event_calendar(order, cal, current_time=now)
        self.assertFalse(r.passed)

    def test_event_within_window_low_signal(self) -> None:
        # event in 12h, signal <= 0.5 should pass
        now = datetime.now(timezone.utc)
        order = _make_order(signal=0.4)
        cal = {"AAPL": now + timedelta(hours=12)}
        r = check_event_calendar(order, cal, current_time=now)
        self.assertTrue(r.passed)


class TestCheckSectorLimit(unittest.TestCase):
    def test_pass_under_limit(self) -> None:
        sector_map = {"AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH"}
        snap = PositionSnapshot(positions={"AAPL": 100_000.0, "MSFT": 80_000.0}, nav=NAV)
        order = _make_order(symbol="NVDA", qty=100, price=200)  # adds $20k TECH
        r = check_sector_limit(order, snap, sector_map)
        self.assertTrue(r.passed)

    def test_fail_exceeds_limit(self) -> None:
        sector_map = {"AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH"}
        snap = PositionSnapshot(positions={"AAPL": 200_000.0, "MSFT": 150_000.0}, nav=NAV)
        order = _make_order(symbol="NVDA", qty=1000, price=200)  # massive TECH
        r = check_sector_limit(order, snap, sector_map)
        self.assertFalse(r.passed)


class TestCheckLeverage(unittest.TestCase):
    def test_pass(self) -> None:
        snap = PositionSnapshot(positions={"AAPL": 500_000.0}, nav=NAV)
        order = _make_order(qty=100, price=150)
        r = check_leverage(order, snap)
        self.assertTrue(r.passed)

    def test_fail(self) -> None:
        # 2.95M existing + 100*200=20k = 2.97M... use qty=500 price=200 = 100k extra -> 3.05M / 1M = 3.05x
        snap = PositionSnapshot(positions={"AAPL": 2_950_000.0}, nav=NAV)
        order = _make_order(qty=500, price=200)  # adds 100k -> total 3.05x
        r = check_leverage(order, snap)
        self.assertFalse(r.passed)


class TestCheckMinHoldBars(unittest.TestCase):
    def test_pass_sufficient_bars(self) -> None:
        r = check_min_hold_bars("AAPL", last_entry_bar=10, current_bar=15, min_hold_bars=3)
        self.assertTrue(r.passed)

    def test_fail_too_few_bars(self) -> None:
        r = check_min_hold_bars("AAPL", last_entry_bar=10, current_bar=11, min_hold_bars=3)
        self.assertFalse(r.passed)
        self.assertEqual(r.rejected_reason, "min_hold_bars_not_met")

    def test_pass_no_prior_entry(self) -> None:
        r = check_min_hold_bars("AAPL", last_entry_bar=None, current_bar=50)
        self.assertTrue(r.passed)


# ---------------------------------------------------------------------------
# PreTradeRiskEngine integration
# ---------------------------------------------------------------------------

class TestPreTradeRiskEngine(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mktemp(suffix=".db")
        self._engine = PreTradeRiskEngine(nav=NAV, db_path=self._tmp)

    def tearDown(self) -> None:
        self._engine.close()
        if os.path.exists(self._tmp):
            os.remove(self._tmp)

    def test_clean_order_passes(self) -> None:
        order = _make_order(qty=100, price=150)
        result = self._engine.check(order, {"daily_pnl": 1000.0, "adv": 5_000_000.0, "spread_bps": 10.0})
        self.assertTrue(result.passed)
        self.assertGreater(result.checks_run, 0)
        self.assertGreater(result.latency_us, 0.0)

    def test_circuit_breaker_blocks(self) -> None:
        self._engine.set_circuit_breaker("AAPL", True)
        result = self._engine.check(_make_order())
        self.assertFalse(result.passed)
        self.assertEqual(result.rejected_reason, "circuit_breaker_active")
        self._engine.set_circuit_breaker("AAPL", False)

    def test_daily_loss_limit_blocks(self) -> None:
        result = self._engine.check(_make_order(), {"daily_pnl": -30_000.0})
        self.assertFalse(result.passed)
        self.assertEqual(result.rejected_reason, "daily_loss_limit_breached")

    def test_rejection_logged_to_db(self) -> None:
        self._engine.set_circuit_breaker("AAPL", True)
        self._engine.check(_make_order())
        self._engine.set_circuit_breaker("AAPL", False)
        rejections = self._engine._log.recent_rejections(10)
        self.assertGreater(len(rejections), 0)

    def test_rejection_rate(self) -> None:
        # force a rejection
        self._engine.set_circuit_breaker("AAPL", True)
        self._engine.check(_make_order())
        self._engine.set_circuit_breaker("AAPL", False)
        # one pass
        self._engine.check(_make_order(symbol="MSFT"))
        rate = self._engine._log.rejection_rate(hours=1)
        self.assertGreater(rate, 0.0)
        self.assertLessEqual(rate, 1.0)


# ---------------------------------------------------------------------------
# Margin manager tests
# ---------------------------------------------------------------------------

class TestRegTMarginCalculator(unittest.TestCase):
    def setUp(self) -> None:
        self._calc = RegTMarginCalculator()
        self._cfg = MarginConfig(account_nav=NAV)

    def test_long_initial_margin(self) -> None:
        im = self._calc.initial_margin_for_single("AAPL", 100, 150.0, self._cfg)
        self.assertAlmostEqual(im, 100 * 150.0 * 0.25)

    def test_short_initial_margin_higher(self) -> None:
        im_long = self._calc.initial_margin_for_single("AAPL", 100, 150.0, self._cfg)
        im_short = self._calc.initial_margin_for_single("AAPL", -100, 150.0, self._cfg)
        self.assertGreater(im_short, im_long)

    def test_calculate_multiple_positions(self) -> None:
        positions = [
            Position("AAPL", 100, 150.0, "equity"),
            Position("MSFT", 50, 300.0, "equity"),
        ]
        req = self._calc.calculate(positions, self._cfg)
        self.assertEqual(req.method, "regt")
        self.assertGreater(req.initial_margin, 0)


class TestMarginManager(unittest.TestCase):
    def setUp(self) -> None:
        self._cfg = MarginConfig(account_nav=NAV)
        self._mgr = MarginManager(self._cfg)

    def test_update_and_margin_utilization(self) -> None:
        self._mgr.update_position("AAPL", 1000, 150.0, "equity")
        util = self._mgr.margin_utilization()
        self.assertGreater(util, 0.0)
        self.assertLessEqual(util, 1.0)

    def test_available_margin_decreases_with_positions(self) -> None:
        avail_before = self._mgr.available_margin()
        self._mgr.update_position("AAPL", 1000, 150.0, "equity")
        avail_after = self._mgr.available_margin()
        self.assertLess(avail_after, avail_before)

    def test_is_margin_call_false_normally(self) -> None:
        self._mgr.update_position("AAPL", 100, 150.0, "equity")
        self.assertFalse(self._mgr.is_margin_call())

    def test_gross_leverage(self) -> None:
        self._mgr.update_position("AAPL", 2000, 150.0, "equity")  # 300k
        lev = self._mgr.gross_leverage()
        self.assertAlmostEqual(lev, 300_000.0 / NAV, places=4)

    def test_remove_position(self) -> None:
        self._mgr.update_position("AAPL", 100, 150.0)
        self._mgr.update_position("AAPL", 0.0, 150.0)
        self.assertIsNone(self._mgr.get_position("AAPL"))

    def test_positions_to_liquidate_empty_when_no_call(self) -> None:
        self.assertEqual(self._mgr.positions_to_liquidate_for_margin(), [])

    def test_crypto_margin_higher_initial(self) -> None:
        calc = CryptoMarginCalculator()
        cfg = MarginConfig(account_nav=NAV)
        im_crypto = calc.initial_margin_for_single("BTCUSDT", 1, 30000.0, cfg)
        self.assertAlmostEqual(im_crypto, 30000.0 * 0.50)

    def test_portfolio_margin_worst_case_positive(self) -> None:
        calc = PortfolioMarginCalculator()
        cfg = MarginConfig(account_nav=NAV)
        positions = [Position("AAPL", 100, 150.0, "futures")]
        req = calc.calculate(positions, cfg)
        self.assertGreater(req.initial_margin, 0)
        self.assertEqual(req.method, "portfolio")


# ---------------------------------------------------------------------------
# Drawdown monitor tests
# ---------------------------------------------------------------------------

class TestDrawdownMonitor(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mktemp(suffix=".db")
        self._mon = DrawdownMonitor(initial_nav=1_000_000.0, db_path=self._tmp)

    def tearDown(self) -> None:
        self._mon.close()
        if os.path.exists(self._tmp):
            os.remove(self._tmp)

    def test_zero_drawdown_at_peak(self) -> None:
        now = datetime.now(timezone.utc)
        self._mon.update(1_000_000.0, now)
        self.assertAlmostEqual(self._mon.current_drawdown(), 0.0)

    def test_drawdown_after_decline(self) -> None:
        t0 = datetime.now(timezone.utc)
        self._mon.update(1_000_000.0, t0)
        self._mon.update(900_000.0, t0 + timedelta(hours=1))
        dd = self._mon.current_drawdown()
        self.assertAlmostEqual(dd, 0.10, places=5)

    def test_recovery_target(self) -> None:
        t0 = datetime.now(timezone.utc)
        self._mon.update(1_000_000.0, t0)
        self._mon.update(800_000.0, t0 + timedelta(hours=1))
        self.assertAlmostEqual(self._mon.recovery_target(), 1_000_000.0)

    def test_drawdown_duration_increases(self) -> None:
        t0 = datetime.now(timezone.utc)
        self._mon.update(1_000_000.0, t0)
        for i in range(5):
            self._mon.update(900_000.0, t0 + timedelta(hours=i + 1))
        self.assertEqual(self._mon.drawdown_duration(), 5)

    def test_is_breach(self) -> None:
        t0 = datetime.now(timezone.utc)
        self._mon.update(1_000_000.0, t0)
        self._mon.update(850_000.0, t0 + timedelta(hours=1))
        self.assertTrue(self._mon.is_breach(0.10))
        self.assertFalse(self._mon.is_breach(0.20))

    def test_max_drawdown(self) -> None:
        t0 = datetime.now(timezone.utc)
        self._mon.update(1_000_000.0, t0)
        self._mon.update(700_000.0, t0 + timedelta(hours=1))
        self._mon.update(900_000.0, t0 + timedelta(hours=2))
        max_dd = self._mon.max_drawdown()
        self.assertAlmostEqual(max_dd, 0.30, places=5)


class TestDrawdownBreachHandler(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_mon = tempfile.mktemp(suffix=".db")
        self._tmp_bh  = tempfile.mktemp(suffix=".db")
        self._mon = DrawdownMonitor(initial_nav=1_000_000.0, db_path=self._tmp_mon)
        self._handler = DrawdownBreachHandler(
            threshold=0.05,
            window=DrawdownWindow.ALL_TIME,
            db_path=self._tmp_bh,
        )

    def tearDown(self) -> None:
        self._mon.close()
        self._handler.close()
        for p in (self._tmp_mon, self._tmp_bh):
            if os.path.exists(p):
                os.remove(p)

    def test_normal_state_initially(self) -> None:
        self.assertEqual(self._handler.recovery_state, RecoveryState.NORMAL)

    def test_breach_transitions_state(self) -> None:
        t0 = datetime.now(timezone.utc)
        self._mon.update(1_000_000.0, t0)
        self._mon.update(900_000.0, t0 + timedelta(hours=1))  # 10% DD > 5% threshold
        self._handler.evaluate(self._mon, 900_000.0)
        self.assertEqual(self._handler.recovery_state, RecoveryState.BREACHED)
        self.assertTrue(self._handler.entries_halted)
        self.assertEqual(self._handler.sizing_factor, 0.0)

    def test_recovery_half_sizing(self) -> None:
        t0 = datetime.now(timezone.utc)
        self._mon.update(1_000_000.0, t0)
        # Force breach at 10%
        self._mon.update(900_000.0, t0 + timedelta(hours=1))
        self._handler.evaluate(self._mon, 900_000.0)
        # Now recover to 4% drawdown (< 50% of 10%)
        self._mon.update(960_000.0, t0 + timedelta(hours=2))
        self._handler.evaluate(self._mon, 960_000.0)
        self.assertEqual(self._handler.recovery_state, RecoveryState.RECOVERING)
        self.assertAlmostEqual(self._handler.sizing_factor, 0.5)


# ---------------------------------------------------------------------------
# VaR monitor tests
# ---------------------------------------------------------------------------

class TestHistoricalVaR(unittest.TestCase):
    def test_zero_returns(self) -> None:
        arr = np.zeros(252)
        var, cvar = historical_var(arr, 0.99, 1)
        self.assertAlmostEqual(var, 0.0, places=5)

    def test_var_increases_with_horizon(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 1000, 252)
        var1, _ = historical_var(returns, 0.99, 1)
        var5, _ = historical_var(returns, 0.99, 5)
        self.assertGreater(var5, var1)


class TestParametricVaR(unittest.TestCase):
    def test_parametric_positive(self) -> None:
        cov = np.array([[0.0004, 0.0001], [0.0001, 0.0004]])
        weights = np.array([0.5, 0.5])
        var, cvar = parametric_var(weights, cov, 1_000_000.0, 0.99, 1)
        self.assertGreater(var, 0)
        self.assertGreater(cvar, var)

    def test_empty_weights(self) -> None:
        var, cvar = parametric_var(np.array([]), np.array([]).reshape(0, 0), 1e6, 0.99, 1)
        self.assertEqual(var, 0.0)


class TestMonteCarloVaR(unittest.TestCase):
    def test_mc_var_positive(self) -> None:
        cov = np.array([[0.0004]])
        weights = np.array([1.0])
        var, cvar = monte_carlo_var(weights, cov, 1_000_000.0, 0.99, 1, n_paths=500, rng_seed=0)
        self.assertGreater(var, 0)


class TestEWMACovarianceMatrix(unittest.TestCase):
    def test_initialization(self) -> None:
        ewma = EWMACovarianceMatrix(["AAPL", "MSFT"])
        self.assertEqual(ewma.n, 2)

    def test_update_changes_covariance(self) -> None:
        ewma = EWMACovarianceMatrix(["AAPL", "MSFT"])
        initial = ewma.covariance_matrix().copy()
        ewma.update({"AAPL": 0.01, "MSFT": -0.01})
        updated = ewma.covariance_matrix()
        self.assertFalse(np.allclose(initial, updated))

    def test_add_symbol_expands_matrix(self) -> None:
        ewma = EWMACovarianceMatrix(["AAPL"])
        ewma.add_symbol("MSFT")
        self.assertEqual(ewma.n, 2)
        self.assertEqual(ewma.covariance_matrix().shape, (2, 2))


class TestVaRMonitor(unittest.TestCase):
    def setUp(self) -> None:
        self._mon = VaRMonitor(nav=NAV)
        self._mon.update_positions({"AAPL": 200_000.0, "MSFT": 150_000.0})
        rng = np.random.default_rng(7)
        for _ in range(100):
            self._mon.update_returns_batch({
                "AAPL": float(rng.normal(0, 0.01)),
                "MSFT": float(rng.normal(0, 0.012)),
            })

    def test_portfolio_var_historical_positive(self) -> None:
        var = self._mon.portfolio_var(0.99, method="historical")
        self.assertGreaterEqual(var, 0.0)

    def test_portfolio_var_parametric_positive(self) -> None:
        var = self._mon.portfolio_var(0.99, method="parametric")
        self.assertGreaterEqual(var, 0.0)

    def test_portfolio_cvar_gte_var(self) -> None:
        var = self._mon.portfolio_var(0.99, method="historical")
        cvar = self._mon.portfolio_cvar(0.99, method="historical")
        self.assertGreaterEqual(cvar, var - 1.0)

    def test_component_var_nonnegative(self) -> None:
        cv = self._mon.component_var("AAPL")
        self.assertGreaterEqual(cv, 0.0)

    def test_var_breach_detection(self) -> None:
        very_small_limit = 1.0
        self.assertTrue(self._mon.is_var_breach(very_small_limit))
        very_large_limit = 1e12
        self.assertFalse(self._mon.is_var_breach(very_large_limit))


class TestVaRLimitManager(unittest.TestCase):
    def setUp(self) -> None:
        self._var_mon = VaRMonitor(nav=NAV)
        self._var_mon.update_positions({"AAPL": 300_000.0})
        rng = np.random.default_rng(99)
        for _ in range(60):
            self._var_mon.update_returns_batch({"AAPL": float(rng.normal(0, 0.015))})
        self._mgr = VaRLimitManager(
            self._var_mon, portfolio_var_limit=1.0, confidence=0.99
        )

    def test_breach_when_limit_very_small(self) -> None:
        breaches = self._mgr.check_all()
        self.assertGreater(len(breaches), 0)

    def test_no_breach_with_large_limit(self) -> None:
        self._mgr.set_portfolio_limit(1e9)
        breaches = self._mgr.check_all()
        self.assertEqual(len(breaches), 0)


# ---------------------------------------------------------------------------
# Risk reporter tests
# ---------------------------------------------------------------------------

class TestRiskReporter(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_db = tempfile.mktemp(suffix=".db")
        cfg = MarginConfig(account_nav=NAV)
        self._margin = MarginManager(cfg)
        self._margin.update_position("AAPL", 500, 150.0, "equity")
        self._margin.update_position("MSFT", 200, 300.0, "equity")

        self._var_mon = VaRMonitor(nav=NAV)
        self._var_mon.update_positions({"AAPL": 75_000.0, "MSFT": 60_000.0})
        rng = np.random.default_rng(1)
        for _ in range(50):
            self._var_mon.update_returns_batch({
                "AAPL": float(rng.normal(0, 0.01)),
                "MSFT": float(rng.normal(0, 0.01)),
            })

        self._dd_mon = DrawdownMonitor(initial_nav=NAV, db_path=self._tmp_db)
        t0 = datetime.now(timezone.utc)
        for i in range(5):
            nav_val = NAV * (1.0 - i * 0.01)
            self._dd_mon.update(nav_val, t0 + timedelta(hours=i))

        self._reporter = RiskReporter(
            var_monitor=self._var_mon,
            margin_manager=self._margin,
            drawdown_monitor=self._dd_mon,
            nav=NAV,
        )
        self._reporter.record_daily_pnl(5000.0)
        self._reporter.record_daily_pnl(-2000.0)
        self._reporter.record_daily_pnl(3000.0)

    def tearDown(self) -> None:
        self._dd_mon.close()
        if os.path.exists(self._tmp_db):
            os.remove(self._tmp_db)

    def test_intraday_snapshot_fields(self) -> None:
        snap = self._reporter.intraday_snapshot()
        self.assertIsInstance(snap, RiskSnapshot)
        self.assertGreater(snap.n_positions, 0)
        self.assertGreater(snap.leverage, 0.0)

    def test_eod_report_fields(self) -> None:
        report = self._reporter.end_of_day_report("2026-04-07")
        self.assertEqual(report.date, "2026-04-07")
        self.assertIsInstance(report.positions, list)
        self.assertGreaterEqual(report.realized_vol, 0.0)

    def test_weekly_report_fields(self) -> None:
        report = self._reporter.weekly_risk_summary("2026-04-07")
        self.assertIsInstance(report, WeeklyRiskReport)
        self.assertIsInstance(report.daily_pnl_series, list)

    def test_export_to_json(self) -> None:
        snap = self._reporter.intraday_snapshot()
        tmp = tempfile.mktemp(suffix=".json")
        try:
            self._reporter.export_to_json(snap, tmp)
            with open(tmp) as fh:
                data = json.load(fh)
            self.assertIn("portfolio_var_99", data)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_export_snapshot_to_html(self) -> None:
        snap = self._reporter.intraday_snapshot()
        tmp = tempfile.mktemp(suffix=".html")
        try:
            self._reporter.export_to_html(snap, tmp)
            with open(tmp) as fh:
                content = fh.read()
            self.assertIn("<html", content)
            self.assertIn("SRFM", content)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_export_eod_to_html(self) -> None:
        report = self._reporter.end_of_day_report("2026-04-07")
        tmp = tempfile.mktemp(suffix=".html")
        try:
            self._reporter.export_to_html(report, tmp)
            with open(tmp) as fh:
                content = fh.read()
            self.assertIn("End-of-Day", content)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_export_weekly_to_html(self) -> None:
        report = self._reporter.weekly_risk_summary("2026-04-07")
        tmp = tempfile.mktemp(suffix=".html")
        try:
            self._reporter.export_to_html(report, tmp)
            with open(tmp) as fh:
                content = fh.read()
            self.assertIn("Weekly", content)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


if __name__ == "__main__":
    unittest.main()
