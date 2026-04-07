"""
execution/tests/test_execution.py
==================================
Production test suite for the execution infrastructure.

Covers:
  - cost_model: impact models, slippage, CostEstimator, Almgren-Chriss,
                CostTracker, venue configs
  - smart_router: urgency routing, large-order splitting, liquidity map,
                  dark pool checker, implementation shortfall, circuit breakers
  - position_manager: open/close, P&L, mark-to-market, persistence, async API

Run with::

    pytest execution/tests/test_execution.py -v

"""

from __future__ import annotations

import asyncio
import math
import os
import sqlite3
import tempfile
import time
import threading
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from execution.cost_model import (
    CostEstimator,
    CostEstimate,
    CostTracker,
    ImpactModel,
    PermanentImpact,
    SlippageModel,
    TemporaryImpact,
    VenueConfig,
    VENUES,
    FillRecord,
    _almgren_chriss_trajectory,
    _intraday_volume_profile,
)

from execution.smart_router import (
    DarkPoolChecker,
    ExecutionLogger,
    LiquidityMap,
    OrderIntent,
    QuoteRecord,
    RoutingDecision,
    SmartRouter,
    _URGENCY_MARKET,
    _URGENCY_PATIENT,
)

from execution.position_manager import (
    Position,
    PositionManager,
    PositionPersistence,
)


# ===========================================================================
# HELPER UTILITIES
# ===========================================================================

def _tmp_db() -> Path:
    """Return a temp-file path for a SQLite database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return Path(path)


def _run(coro):
    """Execute a coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ===========================================================================
# COST MODEL TESTS
# ===========================================================================


class TestVenueConfig(unittest.TestCase):

    def test_all_venues_present(self):
        expected = {"alpaca_equity", "alpaca_crypto", "binance_spot", "coinbase"}
        self.assertEqual(set(VENUES.keys()), expected)

    def test_alpaca_equity_zero_commission(self):
        v = VENUES["alpaca_equity"]
        self.assertEqual(v.maker_fee_bps, 0.0)
        self.assertEqual(v.taker_fee_bps, 0.0)

    def test_alpaca_crypto_fees(self):
        v = VENUES["alpaca_crypto"]
        self.assertEqual(v.maker_fee_bps, 15.0)
        self.assertEqual(v.taker_fee_bps, 25.0)

    def test_binance_spot_fees(self):
        v = VENUES["binance_spot"]
        self.assertEqual(v.maker_fee_bps, 7.0)
        self.assertEqual(v.taker_fee_bps, 10.0)

    def test_coinbase_fees(self):
        v = VENUES["coinbase"]
        self.assertEqual(v.maker_fee_bps, 50.0)
        self.assertEqual(v.taker_fee_bps, 50.0)

    def test_venue_config_fields(self):
        v = VenueConfig(
            name="test_venue",
            maker_fee_bps=5.0,
            taker_fee_bps=10.0,
            min_order_size=100.0,
            max_order_pct_adv=0.02,
        )
        self.assertEqual(v.name, "test_venue")
        self.assertEqual(v.min_order_size, 100.0)
        self.assertEqual(v.max_order_pct_adv, 0.02)


class TestImpactModel(unittest.TestCase):

    def setUp(self):
        self.model = ImpactModel(eta=0.1)

    def test_impact_zero_for_zero_adv(self):
        result = self.model.estimate_bps(10_000, adv_usd=0, sigma_daily=0.02)
        self.assertEqual(result, 0.0)

    def test_impact_zero_for_zero_sigma(self):
        result = self.model.estimate_bps(10_000, adv_usd=1_000_000, sigma_daily=0.0)
        self.assertEqual(result, 0.0)

    def test_impact_model_scaling(self):
        """Doubling order size should roughly sqrt(2)-scale the impact."""
        base   = self.model.estimate_bps(10_000, 1_000_000, 0.02)
        double = self.model.estimate_bps(40_000, 1_000_000, 0.02)
        # sqrt(4) = 2x
        self.assertAlmostEqual(double / base, 2.0, places=4)

    def test_impact_model_adv_scaling(self):
        """Doubling ADV should halve the impact by sqrt(2)."""
        small_adv = self.model.estimate_bps(10_000, 500_000, 0.02)
        large_adv = self.model.estimate_bps(10_000, 2_000_000, 0.02)
        self.assertAlmostEqual(small_adv / large_adv, 2.0, places=4)

    def test_impact_model_sigma_scaling(self):
        """Doubling sigma should double the impact."""
        low_sig  = self.model.estimate_bps(10_000, 1_000_000, 0.01)
        high_sig = self.model.estimate_bps(10_000, 1_000_000, 0.02)
        self.assertAlmostEqual(high_sig / low_sig, 2.0, places=4)

    def test_impact_formula_explicit(self):
        """Manually verify: eta=0.1, sigma=0.02, Q/V=0.01 -> 0.002 frac -> 20 bps."""
        # 0.1 * 0.02 * sqrt(0.01) = 0.1 * 0.02 * 0.1 = 0.0002 -> 2 bps
        result = self.model.estimate_bps(
            order_size_usd=10_000,
            adv_usd=1_000_000,
            sigma_daily=0.02,
        )
        expected = 0.1 * 0.02 * math.sqrt(0.01) * 10_000
        self.assertAlmostEqual(result, expected, places=6)

    def test_calibrate_adjusts_eta(self):
        model = ImpactModel(eta=0.1)
        # Provide fills where actual is twice the model prediction
        fills = [(10.0, 20.0)] * 10
        model.calibrate(fills)
        # After calibration eta should roughly double
        self.assertGreater(model.eta, 0.15)

    def test_calibrate_requires_minimum_fills(self):
        model = ImpactModel(eta=0.1)
        model.calibrate([(10.0, 20.0)] * 3)  # only 3 points -> no change
        self.assertAlmostEqual(model.eta, 0.1)


class TestTemporaryImpact(unittest.TestCase):

    def setUp(self):
        self.model = TemporaryImpact(half_life_bars=5)

    def test_decay_at_zero_bars(self):
        impact = self.model.estimate_bps(10_000, 1_000_000, 0.02)
        remaining = self.model.decay(impact, 0)
        self.assertAlmostEqual(remaining, impact)

    def test_decay_at_half_life(self):
        """After 5 bars, exactly half the impact should remain."""
        remaining_frac = self.model.remaining_fraction(5)
        self.assertAlmostEqual(remaining_frac, 0.5, places=4)

    def test_decay_monotone_decrease(self):
        """Impact must strictly decrease over time."""
        initial = 20.0
        prev = initial
        for t in range(1, 20):
            curr = self.model.decay(initial, t)
            self.assertLess(curr, prev)
            prev = curr

    def test_decay_approaches_zero(self):
        remaining = self.model.decay(100.0, elapsed_bars=200)
        self.assertLess(remaining, 0.01)

    def test_negative_elapsed_returns_initial(self):
        initial = 20.0
        result = self.model.decay(initial, elapsed_bars=-1)
        self.assertEqual(result, initial)


class TestPermanentImpact(unittest.TestCase):

    def test_default_fifty_percent(self):
        model = PermanentImpact()
        temp  = 20.0
        perm  = model.estimate_bps(temp)
        self.assertAlmostEqual(perm, 10.0)

    def test_custom_fraction(self):
        model = PermanentImpact(permanent_fraction=0.3)
        perm  = model.estimate_bps(100.0)
        self.assertAlmostEqual(perm, 30.0)

    def test_total_decomposition(self):
        model = PermanentImpact(permanent_fraction=0.5)
        perm, temp = model.total_impact_bps(10_000, 1_000_000, 0.02)
        self.assertGreater(perm, 0)
        self.assertGreater(temp, 0)
        self.assertAlmostEqual(perm, temp * 0.5, places=4)


class TestSlippageModel(unittest.TestCase):

    def setUp(self):
        self.model = SlippageModel()

    def test_spread_cost_roundtrip(self):
        self.assertEqual(self.model.spread_cost_bps(3.5), 3.5)

    def test_timing_slippage_zero_for_zero_bars(self):
        result = self.model.timing_slippage_bps(0.02, execution_time_bars=0)
        self.assertEqual(result, 0.0)

    def test_timing_slippage_zero_for_zero_sigma(self):
        result = self.model.timing_slippage_bps(0.0, execution_time_bars=4)
        self.assertEqual(result, 0.0)

    def test_timing_slippage_scaling(self):
        """Doubling execution time should sqrt(2)-scale timing slippage."""
        t1 = self.model.timing_slippage_bps(0.02, 1)
        t4 = self.model.timing_slippage_bps(0.02, 4)
        self.assertAlmostEqual(t4 / t1, 2.0, places=4)

    def test_estimate_combines_components(self):
        spread = 2.0
        timing = self.model.timing_slippage_bps(0.02, 1)
        total  = self.model.estimate_bps(spread, 0.02, 1)
        self.assertAlmostEqual(total, spread + timing, places=6)


class TestCostEstimatorCryptoVsEquity(unittest.TestCase):
    """Crypto should be more expensive than equity (Alpaca) due to non-zero commissions."""

    def setUp(self):
        self.est = CostEstimator()
        self.kwargs = dict(
            symbol="BTC",
            order_size_usd=50_000,
            side="buy",
            adv_usd=500_000_000,
            sigma_daily=0.03,
        )

    def test_crypto_more_expensive_than_equity(self):
        crypto  = self.est.estimate(venue="alpaca_crypto", **self.kwargs)
        equity  = self.est.estimate(venue="alpaca_equity", **self.kwargs)
        self.assertGreater(crypto.total_bps, equity.total_bps)

    def test_commission_zero_for_alpaca_equity(self):
        est = self.est.estimate(
            symbol="SPY", order_size_usd=10_000, side="buy",
            venue="alpaca_equity", adv_usd=25_000_000, sigma_daily=0.01,
        )
        self.assertEqual(est.commission_bps, 0.0)

    def test_commission_nonzero_for_alpaca_crypto(self):
        est = self.est.estimate(
            symbol="BTC", order_size_usd=10_000, side="buy",
            venue="alpaca_crypto", adv_usd=500_000_000, sigma_daily=0.03,
        )
        self.assertGreater(est.commission_bps, 0.0)

    def test_maker_cheaper_than_taker(self):
        taker = self.est.estimate(venue="alpaca_crypto", is_maker=False, **self.kwargs)
        maker = self.est.estimate(venue="alpaca_crypto", is_maker=True,  **self.kwargs)
        self.assertLess(maker.total_bps, taker.total_bps)

    def test_unknown_venue_raises(self):
        with self.assertRaises(ValueError):
            self.est.estimate(
                symbol="X", order_size_usd=100, side="buy",
                venue="unknown_venue", adv_usd=1_000_000, sigma_daily=0.02,
            )

    def test_dollar_cost_proportional_to_order_size(self):
        e1 = self.est.estimate(
            symbol="SPY", order_size_usd=10_000, side="buy",
            venue="alpaca_equity", adv_usd=25_000_000, sigma_daily=0.01,
        )
        e2 = self.est.estimate(
            symbol="SPY", order_size_usd=20_000, side="buy",
            venue="alpaca_equity", adv_usd=25_000_000, sigma_daily=0.01,
        )
        # Dollar cost should scale with order size (not perfectly linear due to sqrt)
        self.assertGreater(e2.dollar_cost, e1.dollar_cost)

    def test_cost_estimate_to_dict(self):
        est = self.est.estimate(
            symbol="SPY", order_size_usd=10_000, side="buy",
            venue="alpaca_equity", adv_usd=25_000_000, sigma_daily=0.01,
        )
        d = est.to_dict()
        self.assertIn("total_bps",    d)
        self.assertIn("dollar_cost",  d)
        self.assertIn("impact_bps",   d)
        self.assertEqual(d["symbol"], "SPY")

    def test_impact_increases_with_order_size(self):
        small = self.est.estimate(
            symbol="BTC", order_size_usd=10_000, side="buy",
            venue="alpaca_crypto", adv_usd=500_000_000, sigma_daily=0.03,
        )
        large = self.est.estimate(
            symbol="BTC", order_size_usd=5_000_000, side="buy",
            venue="alpaca_crypto", adv_usd=500_000_000, sigma_daily=0.03,
        )
        self.assertGreater(large.impact_bps, small.impact_bps)


class TestAlmgrenChrissTrajectory(unittest.TestCase):

    def test_single_bar(self):
        trades = _almgren_chriss_trajectory(100, 1, 0.01, 0.1, 0.05, 1e-6)
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0], 100.0, places=6)

    def test_sum_equals_total(self):
        trades = _almgren_chriss_trajectory(1000, 8, 0.005, 0.1, 0.05, 1e-6)
        self.assertAlmostEqual(sum(trades), 1000.0, places=4)

    def test_all_trades_positive(self):
        trades = _almgren_chriss_trajectory(500, 10, 0.005, 0.1, 0.05, 1e-6)
        for t in trades:
            self.assertGreater(t, 0)

    def test_front_loading_with_high_risk_aversion(self):
        """High risk aversion should front-load execution (larger early trades)."""
        # High lambda -> urgency -> front-load
        high_lam = _almgren_chriss_trajectory(1000, 10, 0.005, 0.1, 0.05, lam=1.0)
        # Low lambda -> patient -> more uniform
        low_lam  = _almgren_chriss_trajectory(1000, 10, 0.005, 0.1, 0.05, lam=1e-9)
        # First trade should be larger under high risk aversion
        self.assertGreater(high_lam[0], low_lam[0])

    def test_length_matches_n_bars(self):
        for n in [2, 4, 8, 12, 20]:
            trades = _almgren_chriss_trajectory(100, n, 0.005, 0.1, 0.05, 1e-6)
            self.assertEqual(len(trades), n)

    def test_zero_total_shares(self):
        trades = _almgren_chriss_trajectory(0, 5, 0.005, 0.1, 0.05, 1e-6)
        self.assertEqual(trades, [0])


class TestCostEstimatorSchedule(unittest.TestCase):

    def setUp(self):
        self.est = CostEstimator()

    def test_twap_equal_slices(self):
        fracs = self.est.optimize_execution_schedule(
            order_size=100_000, adv=10_000_000, target_bars=4,
            sigma_daily=0.02, strategy="twap",
        )
        self.assertEqual(len(fracs), 4)
        for f in fracs:
            self.assertAlmostEqual(f, 0.25, places=6)

    def test_vwap_sums_to_one(self):
        fracs = self.est.optimize_execution_schedule(
            order_size=100_000, adv=10_000_000, target_bars=8,
            sigma_daily=0.02, strategy="vwap",
        )
        self.assertAlmostEqual(sum(fracs), 1.0, places=6)
        self.assertEqual(len(fracs), 8)

    def test_almgren_chriss_sums_to_one(self):
        fracs = self.est.optimize_execution_schedule(
            order_size=100_000, adv=10_000_000, target_bars=8,
            sigma_daily=0.02, strategy="almgren_chriss",
        )
        self.assertAlmostEqual(sum(fracs), 1.0, places=4)

    def test_zero_target_bars(self):
        fracs = self.est.optimize_execution_schedule(
            order_size=100_000, adv=10_000_000, target_bars=0,
        )
        self.assertEqual(fracs, [1.0])

    def test_cheapest_venue(self):
        venue, est = self.est.cheapest_venue(
            symbol="SPY",
            order_size_usd=50_000,
            side="buy",
            adv_usd=25_000_000,
            sigma_daily=0.01,
        )
        self.assertIn(venue, VENUES)
        self.assertIsInstance(est, CostEstimate)


class TestIntradayVolumeProfile(unittest.TestCase):

    def test_single_bar(self):
        profile = _intraday_volume_profile(1)
        self.assertEqual(len(profile), 1)
        self.assertEqual(profile[0], 1.0)

    def test_all_positive(self):
        for n in [2, 4, 8, 26]:
            profile = _intraday_volume_profile(n)
            for v in profile:
                self.assertGreater(v, 0)

    def test_u_shape(self):
        """Endpoints should have higher volume than midpoint."""
        profile = _intraday_volume_profile(11)  # odd length so midpoint is exact
        mid = len(profile) // 2
        self.assertGreater(profile[0],  profile[mid])
        self.assertGreater(profile[-1], profile[mid])


class TestCostTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = CostTracker()

    def test_record_and_report(self):
        self.tracker.record_fill("SPY", estimated_bps=5.0, actual_bps=6.0, venue="alpaca_equity")
        report = self.tracker.get_slippage_report()
        self.assertEqual(report["n_fills"], 1)
        self.assertIn("alpaca_equity", report["per_venue"])
        self.assertIn("SPY",           report["per_symbol"])

    def test_slippage_bps_calculation(self):
        self.tracker.record_fill("BTC", estimated_bps=20.0, actual_bps=25.0, venue="alpaca_crypto")
        report = self.tracker.get_slippage_report()
        spy_data = report["per_symbol"]["BTC"]
        self.assertAlmostEqual(spy_data["avg_slippage_bps"], 5.0)

    def test_efficiency_ratio_perfect(self):
        """When estimated == actual, efficiency ratio should be 1.0."""
        for i in range(5):
            self.tracker.record_fill("SPY", 10.0, 10.0, "alpaca_equity")
        ratio = self.tracker.efficiency_ratio()
        self.assertAlmostEqual(ratio, 1.0, places=4)

    def test_efficiency_ratio_underestimate(self):
        """When actual > estimated, efficiency < 1."""
        for i in range(5):
            self.tracker.record_fill("SPY", 5.0, 10.0, "alpaca_equity")
        ratio = self.tracker.efficiency_ratio()
        self.assertLess(ratio, 1.0)

    def test_empty_report_defaults(self):
        report = self.tracker.get_slippage_report()
        self.assertEqual(report["n_fills"], 0)
        self.assertAlmostEqual(report["efficiency_ratio"], 1.0)

    def test_multi_venue_per_venue_report(self):
        self.tracker.record_fill("SPY", 5.0, 5.5, "alpaca_equity")
        self.tracker.record_fill("BTC", 20.0, 22.0, "alpaca_crypto")
        report = self.tracker.get_slippage_report()
        self.assertIn("alpaca_equity", report["per_venue"])
        self.assertIn("alpaca_crypto", report["per_venue"])

    def test_clear_removes_records(self):
        self.tracker.record_fill("SPY", 5.0, 5.0, "alpaca_equity")
        self.tracker.clear()
        report = self.tracker.get_slippage_report()
        self.assertEqual(report["n_fills"], 0)

    def test_sqlite_persistence(self):
        db = _tmp_db()
        try:
            t1 = CostTracker(db_path=db)
            t1.record_fill("SPY", 5.0, 6.0, "alpaca_equity")
            t1.record_fill("BTC", 20.0, 22.0, "alpaca_crypto")

            t2 = CostTracker(db_path=db)
            loaded = t2.load_from_db()
            self.assertEqual(loaded, 2)
        finally:
            db.unlink(missing_ok=True)

    def test_since_filter(self):
        past_ts = time.time() - 90 * 86400  # 90 days ago
        self.tracker.record_fill("SPY", 5.0, 6.0, "alpaca_equity", ts=past_ts)
        self.tracker.record_fill("SPY", 5.0, 6.0, "alpaca_equity")  # recent
        # Only 30-day window should see the recent one
        report = self.tracker.get_slippage_report()
        self.assertEqual(report["n_fills"], 1)


# ===========================================================================
# SMART ROUTER TESTS
# ===========================================================================


class TestLiquidityMap(unittest.TestCase):

    def setUp(self):
        self.lm = LiquidityMap()

    def test_update_and_get_quote(self):
        self.lm.update_quote("SPY", "alpaca_equity", bid=450.0, ask=450.02, bid_size=1000, ask_size=800)
        q = self.lm.get_quote("SPY", "alpaca_equity")
        self.assertIsNotNone(q)
        self.assertEqual(q.bid, 450.0)
        self.assertEqual(q.ask, 450.02)

    def test_mid_price(self):
        self.lm.update_quote("SPY", "alpaca_equity", bid=450.0, ask=450.04, bid_size=1000, ask_size=500)
        q = self.lm.get_quote("SPY", "alpaca_equity")
        self.assertAlmostEqual(q.mid, 450.02, places=4)

    def test_half_spread_bps(self):
        self.lm.update_quote("SPY", "alpaca_equity", bid=450.0, ask=450.10, bid_size=1000, ask_size=500)
        q = self.lm.get_quote("SPY", "alpaca_equity")
        # spread = 0.10, half = 0.05, mid = 450.05
        expected = (0.05 / 450.05) * 10_000
        self.assertAlmostEqual(q.half_spread_bps, expected, places=2)

    def test_stale_quote_returns_none(self):
        stale_ts = time.time() - 200   # 200 seconds ago, > 60s stale threshold
        self.lm.update_quote("SPY", "alpaca_equity", bid=450.0, ask=450.02,
                             bid_size=1000, ask_size=800, ts=stale_ts)
        q = self.lm.get_quote("SPY", "alpaca_equity")
        self.assertIsNone(q)

    def test_missing_quote_returns_none(self):
        q = self.lm.get_quote("NOTEXIST", "alpaca_equity")
        self.assertIsNone(q)

    def test_circuit_breaker_trip_and_reset(self):
        self.lm.trip_circuit_breaker("alpaca_equity")
        self.assertTrue(self.lm.is_circuit_broken("alpaca_equity"))
        self.lm.reset_circuit_breaker("alpaca_equity")
        self.assertFalse(self.lm.is_circuit_broken("alpaca_equity"))

    def test_best_venue_with_no_quotes_returns_valid_venue(self):
        venue = self.lm.best_venue(
            symbol="SPY", order_size_usd=10_000,
            candidates=["alpaca_equity"],
            adv_usd=25_000_000, sigma_daily=0.01,
        )
        self.assertEqual(venue, "alpaca_equity")

    def test_best_venue_excludes_circuit_broken(self):
        self.lm.trip_circuit_breaker("binance_spot")
        venue = self.lm.best_venue(
            symbol="BTC", order_size_usd=10_000,
            candidates=["alpaca_crypto", "binance_spot"],
            adv_usd=500_000_000, sigma_daily=0.03,
        )
        self.assertEqual(venue, "alpaca_crypto")

    def test_all_quotes_returns_list(self):
        self.lm.update_quote("SPY", "alpaca_equity", 450, 450.1, 1000, 800)
        self.lm.update_quote("BTC", "alpaca_crypto", 60000, 60020, 1, 0.5)
        quotes = self.lm.all_quotes()
        self.assertEqual(len(quotes), 2)


class TestSmartRouterUrgencyRouting(unittest.TestCase):

    def setUp(self):
        self.router = SmartRouter()
        self.lm     = LiquidityMap()
        self.lm.update_quote("SPY", "alpaca_equity",
                             bid=450.0, ask=450.1, bid_size=1000, ask_size=800)

    def _make_intent(self, urgency: float, qty: float = 100) -> OrderIntent:
        return OrderIntent(
            symbol="SPY",
            target_qty=qty,
            urgency=urgency,
            max_slippage_bps=20.0,
            time_limit_bars=4,
            adv_usd=25_000_000,
            sigma_daily=0.01,
            asset_class="equity",
        )

    def test_high_urgency_produces_market_order(self):
        intent    = self._make_intent(urgency=0.9)
        decisions = self.router.route(intent, self.lm)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].order_type, "market")

    def test_low_urgency_produces_twap(self):
        intent    = self._make_intent(urgency=0.1)
        decisions = self.router.route(intent, self.lm)
        self.assertGreater(len(decisions), 1)
        for d in decisions:
            self.assertEqual(d.order_type, "twap_slice")

    def test_medium_urgency_produces_limit(self):
        intent    = self._make_intent(urgency=0.5)
        decisions = self.router.route(intent, self.lm)
        self.assertEqual(len(decisions), 1)
        self.assertIn(decisions[0].order_type, {"limit", "market"})

    def test_limit_price_between_bid_and_ask_for_buy(self):
        intent    = self._make_intent(urgency=0.5, qty=100)
        decisions = self.router.route(intent, self.lm)
        if decisions[0].order_type == "limit":
            # Buy limit should be above bid and at or below ask
            self.assertGreater(decisions[0].limit_price, 450.0)
            self.assertLessEqual(decisions[0].limit_price, 450.1)

    def test_sell_limit_below_ask(self):
        intent = self._make_intent(urgency=0.5, qty=-100)
        decisions = self.router.route(intent, self.lm)
        if decisions[0].order_type == "limit":
            self.assertLess(decisions[0].limit_price, 450.1)

    def test_urgency_exactly_at_market_threshold(self):
        intent    = self._make_intent(urgency=_URGENCY_MARKET)
        decisions = self.router.route(intent, self.lm)
        self.assertEqual(decisions[0].order_type, "market")

    def test_urgency_exactly_at_patient_threshold(self):
        intent    = self._make_intent(urgency=_URGENCY_PATIENT)
        decisions = self.router.route(intent, self.lm)
        for d in decisions:
            self.assertEqual(d.order_type, "twap_slice")

    def test_twap_qty_sums_to_total(self):
        intent    = self._make_intent(urgency=0.1, qty=400)
        decisions = self.router.route(intent, self.lm)
        total_qty = sum(d.qty for d in decisions)
        self.assertAlmostEqual(total_qty, 400.0, places=4)

    def test_crypto_routes_to_crypto_venue(self):
        lm = LiquidityMap()
        lm.update_quote("BTC", "alpaca_crypto", 60000, 60020, 1, 0.5)
        intent = OrderIntent(
            symbol="BTC",
            target_qty=0.1,
            urgency=0.9,
            adv_usd=500_000_000,
            sigma_daily=0.03,
            asset_class="crypto",
        )
        decisions = self.router.route(intent, lm)
        self.assertIn(decisions[0].venue, {"alpaca_crypto", "binance_spot", "coinbase"})


class TestSmartRouterLargeOrderSplit(unittest.TestCase):

    def setUp(self):
        self.router = SmartRouter()
        self.lm     = LiquidityMap()
        self.lm.update_quote("SPY", "alpaca_equity", 450.0, 450.1, 1000, 800)

    def test_large_order_splits_into_multiple_bars(self):
        # Order is 5% of ADV -> triggers large order path
        intent = OrderIntent(
            symbol="SPY",
            target_qty=25_000,    # $25k of a $500k ADV = 5%
            urgency=0.5,
            adv_usd=500_000,
            sigma_daily=0.01,
            time_limit_bars=8,
            asset_class="equity",
        )
        decisions = self.router.route(intent, self.lm)
        self.assertGreater(len(decisions), 1)

    def test_large_order_qty_sums_correctly(self):
        intent = OrderIntent(
            symbol="SPY",
            target_qty=10_000,
            urgency=0.5,
            adv_usd=500_000,
            sigma_daily=0.01,
            time_limit_bars=6,
            asset_class="equity",
        )
        decisions = self.router.route(intent, self.lm)
        total_qty = sum(d.qty for d in decisions)
        self.assertAlmostEqual(total_qty, 10_000.0, places=2)

    def test_large_order_bar_indices_sequential(self):
        intent = OrderIntent(
            symbol="SPY",
            target_qty=50_000,
            urgency=0.5,
            adv_usd=1_000_000,
            sigma_daily=0.01,
            time_limit_bars=5,
            asset_class="equity",
        )
        decisions = self.router.route(intent, self.lm)
        bar_indices = [d.bar_index for d in decisions]
        self.assertEqual(bar_indices, list(range(len(decisions))))

    def test_estimated_cost_bps_populated(self):
        intent = OrderIntent(
            symbol="SPY",
            target_qty=10_000,
            urgency=0.5,
            adv_usd=500_000,
            sigma_daily=0.01,
            time_limit_bars=4,
            asset_class="equity",
        )
        decisions = self.router.route(intent, self.lm)
        for d in decisions:
            self.assertGreaterEqual(d.estimated_cost_bps, 0.0)

    def test_routing_decision_to_dict(self):
        d = RoutingDecision(
            venue="alpaca_equity", order_type="limit",
            limit_price=450.0, qty=100, bar_index=0,
            estimated_cost_bps=5.2, reasoning="test",
        )
        row = d.to_dict()
        self.assertEqual(row["venue"], "alpaca_equity")
        self.assertEqual(row["order_type"], "limit")
        self.assertIn("estimated_cost_bps", row)


class TestImplementationShortfall(unittest.TestCase):

    def setUp(self):
        self.router = SmartRouter()

    def test_positive_is_for_adverse_buy(self):
        """Bought above mid -> positive IS (bad for buyer)."""
        is_bps = self.router.implementation_shortfall(
            decision_mid=100.0, fill_price=100.05, side="buy"
        )
        self.assertGreater(is_bps, 0)

    def test_negative_is_for_favorable_buy(self):
        """Bought below mid -> negative IS (good for buyer)."""
        is_bps = self.router.implementation_shortfall(
            decision_mid=100.0, fill_price=99.95, side="buy"
        )
        self.assertLess(is_bps, 0)

    def test_positive_is_for_adverse_sell(self):
        """Sold below mid -> positive IS (bad for seller)."""
        is_bps = self.router.implementation_shortfall(
            decision_mid=100.0, fill_price=99.95, side="sell"
        )
        self.assertGreater(is_bps, 0)

    def test_zero_mid_returns_zero(self):
        is_bps = self.router.implementation_shortfall(
            decision_mid=0.0, fill_price=100.0, side="buy"
        )
        self.assertEqual(is_bps, 0.0)


class TestDarkPoolChecker(unittest.TestCase):

    def setUp(self):
        self.checker = DarkPoolChecker()

    def test_large_cap_eligible(self):
        self.assertTrue(self.checker.has_dark_pool("SPY"))
        self.assertTrue(self.checker.has_dark_pool("AAPL"))

    def test_unknown_symbol_not_eligible(self):
        self.assertFalse(self.checker.has_dark_pool("SMALLCAP123"))

    def test_fill_probability_decreases_with_size(self):
        p_small = self.checker.fill_probability("SPY", 10_000, 25_000_000)
        p_large = self.checker.fill_probability("SPY", 5_000_000, 25_000_000)
        self.assertGreater(p_small, p_large)

    def test_fill_probability_zero_for_ineligible(self):
        p = self.checker.fill_probability("SMALLCAP", 10_000, 1_000_000)
        self.assertEqual(p, 0.0)

    def test_recommend_dark_pool_small_order(self):
        result = self.checker.recommend_dark_pool("SPY", 1_000, 25_000_000)
        self.assertTrue(result)

    def test_recommend_dark_pool_very_large_order(self):
        # Huge order relative to ADV: fill probability drops below 50%
        result = self.checker.recommend_dark_pool("SPY", 50_000_000, 25_000_000)
        self.assertFalse(result)

    def test_add_eligible_symbol(self):
        self.checker.add_eligible_symbol("MYCOIN")
        self.assertTrue(self.checker.has_dark_pool("MYCOIN"))


class TestExecutionLogger(unittest.TestCase):

    def _make_intent(self) -> OrderIntent:
        return OrderIntent(symbol="SPY", target_qty=100, urgency=0.5,
                          adv_usd=25_000_000, sigma_daily=0.01)

    def _make_decision(self) -> RoutingDecision:
        return RoutingDecision(
            venue="alpaca_equity", order_type="limit",
            limit_price=450.0, qty=100, estimated_cost_bps=5.0,
            reasoning="test decision",
        )

    def test_log_and_retrieve_decision(self):
        logger  = ExecutionLogger()
        intent  = self._make_intent()
        decision = self._make_decision()
        logger.log_decision(intent, decision)
        since = time.time() - 60
        rows  = logger.decisions_since(since)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["venue"], "alpaca_equity")

    def test_log_fill_and_quality_report(self):
        logger = ExecutionLogger()
        logger.log_fill("alpaca_equity", "SPY", 100, 450.05, 450.0, 1.1, 0)
        quality = logger.get_execution_quality()
        self.assertEqual(quality["n_fills"], 1)
        self.assertAlmostEqual(quality["avg_is_bps"], 1.1)

    def test_execution_quality_by_symbol(self):
        logger = ExecutionLogger()
        logger.log_fill("alpaca_equity", "SPY", 100, 450.05, 450.0, 1.0, 0)
        logger.log_fill("alpaca_equity", "QQQ", 50, 380.01, 380.0, 0.3, 0)
        q_spy = logger.get_execution_quality(symbol="SPY")
        self.assertEqual(q_spy["n_fills"], 1)
        q_qqq = logger.get_execution_quality(symbol="QQQ")
        self.assertEqual(q_qqq["n_fills"], 1)

    def test_sqlite_logger_persists(self):
        db = _tmp_db()
        try:
            logger  = ExecutionLogger(db_path=db)
            intent  = self._make_intent()
            decision = self._make_decision()
            logger.log_decision(intent, decision)
            logger.log_fill("alpaca_equity", "SPY", 100, 450.05, 450.0, 1.0, 0)

            # New logger instance reading same DB
            logger2 = ExecutionLogger(db_path=db)
            con = sqlite3.connect(str(db))
            decisions_n = con.execute("SELECT COUNT(*) FROM routing_decisions").fetchone()[0]
            fills_n     = con.execute("SELECT COUNT(*) FROM fill_outcomes").fetchone()[0]
            con.close()
            self.assertEqual(decisions_n, 1)
            self.assertEqual(fills_n, 1)
        finally:
            db.unlink(missing_ok=True)

    def test_clear_removes_memory_records(self):
        logger = ExecutionLogger()
        logger.log_fill("alpaca_equity", "SPY", 100, 450.05, 450.0, 1.0, 0)
        logger.clear()
        quality = logger.get_execution_quality()
        self.assertEqual(quality["n_fills"], 0)


# ===========================================================================
# POSITION MANAGER TESTS
# ===========================================================================


class TestPositionOpenClose(unittest.TestCase):

    def setUp(self):
        self.mgr = PositionManager(equity_usd=100_000.0)

    def _open(self, symbol, qty, px, comm=0.0):
        return _run(self.mgr.open_position(symbol, qty, px, comm))

    def _close(self, symbol, qty, px, comm=0.0):
        return _run(self.mgr.close_position(symbol, qty, px, comm))

    def test_open_creates_position(self):
        pos = self._open("SPY", 100, 450.0)
        self.assertEqual(pos.symbol, "SPY")
        self.assertAlmostEqual(pos.qty, 100.0)
        self.assertAlmostEqual(pos.avg_entry_px, 450.0)

    def test_open_adds_to_existing(self):
        self._open("SPY", 100, 450.0)
        pos = self._open("SPY", 50, 452.0)
        expected_avg = (100 * 450.0 + 50 * 452.0) / 150
        self.assertAlmostEqual(pos.qty, 150.0)
        self.assertAlmostEqual(pos.avg_entry_px, expected_avg, places=4)

    def test_close_partial_position(self):
        self._open("SPY", 100, 450.0)
        pnl = self._close("SPY", 50, 452.0)
        pos = self.mgr.get_position("SPY")
        self.assertAlmostEqual(pos.qty, 50.0)
        self.assertAlmostEqual(pnl, 50 * (452.0 - 450.0), places=4)

    def test_close_full_position(self):
        self._open("SPY", 100, 450.0)
        pnl = self._close("SPY", 100, 455.0)
        pos = self.mgr.get_position("SPY")
        self.assertIsNone(pos)   # should be flat
        self.assertAlmostEqual(pnl, 100 * (455.0 - 450.0), places=4)

    def test_close_nonexistent_returns_zero(self):
        pnl = self._close("NOTEXIST", 10, 100.0)
        self.assertEqual(pnl, 0.0)

    def test_close_more_than_held_caps_at_held(self):
        self._open("SPY", 50, 450.0)
        pnl = self._close("SPY", 200, 452.0)   # want to close 200 but only have 50
        pos = self.mgr.get_position("SPY")
        self.assertIsNone(pos)
        self.assertAlmostEqual(pnl, 50 * (452.0 - 450.0), places=4)


class TestPositionPnlCalculation(unittest.TestCase):

    def setUp(self):
        self.mgr = PositionManager(equity_usd=100_000.0)

    def _open(self, symbol, qty, px, comm=0.0):
        return _run(self.mgr.open_position(symbol, qty, px, comm))

    def _close(self, symbol, qty, px, comm=0.0):
        return _run(self.mgr.close_position(symbol, qty, px, comm))

    def test_unrealized_pnl_after_price_update(self):
        self._open("BTC", 1.0, 60_000.0)
        _run(self.mgr.update_prices({"BTC": 61_000.0}))
        pos = self.mgr.get_position("BTC")
        self.assertAlmostEqual(pos.unrealized_pnl, 1_000.0, places=2)

    def test_unrealized_pnl_negative_when_price_falls(self):
        self._open("BTC", 1.0, 60_000.0)
        _run(self.mgr.update_prices({"BTC": 59_000.0}))
        pos = self.mgr.get_position("BTC")
        self.assertAlmostEqual(pos.unrealized_pnl, -1_000.0, places=2)

    def test_commission_reduces_total_pnl(self):
        self._open("SPY", 100, 450.0, comm=5.0)
        pos = self.mgr.get_position("SPY")
        self.assertAlmostEqual(pos.commission_paid, 5.0)
        self.assertLess(pos.total_pnl, pos.unrealized_pnl)

    def test_realized_pnl_accumulates(self):
        self._open("SPY", 100, 450.0)
        pnl1 = self._close("SPY", 50, 455.0)
        # Re-open and close again
        self._open("SPY", 50, 455.0)
        pnl2 = self._close("SPY", 50, 460.0)
        pos = self.mgr.get_position("SPY")
        # After both closes, some realized pnl from first SPY close is in the pos object
        # Note: second open creates a new position object after first is flat
        self.assertGreater(pnl1, 0)
        self.assertGreater(pnl2, 0)

    def test_cost_basis_correct(self):
        self._open("AAPL", 10, 170.0)
        pos = self.mgr.get_position("AAPL")
        self.assertAlmostEqual(pos.cost_basis, 10 * 170.0, places=4)

    def test_market_value_correct(self):
        self._open("AAPL", 10, 170.0)
        _run(self.mgr.update_prices({"AAPL": 175.0}))
        pos = self.mgr.get_position("AAPL")
        self.assertAlmostEqual(pos.market_value, 10 * 175.0, places=4)

    def test_portfolio_summary(self):
        self._open("SPY", 100, 450.0)
        self._open("BTC", 0.5, 60_000.0)
        summary = _run(self.mgr.get_portfolio_summary())
        self.assertEqual(summary["n_positions"], 2)
        self.assertGreater(summary["total_market_value"], 0)

    def test_concentration_calculation(self):
        self._open("SPY", 100, 450.0)
        _run(self.mgr.update_prices({"SPY": 450.0}))
        conc = self.mgr.get_concentration("SPY")
        expected = (100 * 450.0) / 100_000.0
        self.assertAlmostEqual(conc, expected, places=4)

    def test_position_age_bars_zero_just_opened(self):
        self._open("SPY", 100, 450.0)
        age = self.mgr.get_position_age_bars("SPY")
        self.assertEqual(age, 0)  # just opened -- should be 0 bars

    def test_position_age_bars_no_position(self):
        age = self.mgr.get_position_age_bars("NOTEXIST")
        self.assertEqual(age, 0)

    def test_export_to_dict(self):
        self._open("SPY", 100, 450.0)
        self._open("BTC", 0.1, 60_000.0)
        rows = self.mgr.export_to_dict()
        self.assertEqual(len(rows), 2)
        symbols = {r["symbol"] for r in rows}
        self.assertEqual(symbols, {"SPY", "BTC"})

    def test_all_positions_excludes_flat(self):
        self._open("SPY", 100, 450.0)
        self._open("BTC", 0.1, 60_000.0)
        _run(self.mgr.close_position("SPY", 100, 451.0))
        positions = self.mgr.all_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].symbol, "BTC")


class TestPositionPersistenceRoundtrip(unittest.TestCase):

    def test_save_and_load_snapshot(self):
        db = _tmp_db()
        try:
            p = PositionPersistence(db)
            positions = {
                "SPY": Position(
                    symbol="SPY", qty=100, avg_entry_px=450.0,
                    current_px=452.0, unrealized_pnl=200.0,
                    realized_pnl=50.0, cost_basis=45_000.0,
                    commission_paid=0.0,
                ),
                "BTC": Position(
                    symbol="BTC", qty=0.5, avg_entry_px=60_000.0,
                    current_px=61_000.0, unrealized_pnl=500.0,
                    realized_pnl=0.0, cost_basis=30_000.0,
                    commission_paid=10.0,
                ),
            }
            p.save_snapshot(positions)
            loaded = p.load_snapshot()

            self.assertIn("SPY", loaded)
            self.assertIn("BTC", loaded)
            self.assertAlmostEqual(loaded["SPY"].qty, 100.0)
            self.assertAlmostEqual(loaded["SPY"].avg_entry_px, 450.0)
            self.assertAlmostEqual(loaded["BTC"].qty, 0.5)
            self.assertAlmostEqual(loaded["BTC"].avg_entry_px, 60_000.0)
        finally:
            db.unlink(missing_ok=True)

    def test_snapshot_overwrites_stale_data(self):
        db = _tmp_db()
        try:
            p = PositionPersistence(db)
            # First snapshot
            positions = {
                "SPY": Position(symbol="SPY", qty=100, avg_entry_px=450.0,
                                current_px=450.0),
            }
            p.save_snapshot(positions)

            # Update and save again
            positions["SPY"].qty = 200
            positions["SPY"].avg_entry_px = 451.0
            p.save_snapshot(positions)

            loaded = p.load_snapshot()
            self.assertAlmostEqual(loaded["SPY"].qty, 200.0)
            self.assertAlmostEqual(loaded["SPY"].avg_entry_px, 451.0)
        finally:
            db.unlink(missing_ok=True)

    def test_trade_log_roundtrip(self):
        db = _tmp_db()
        try:
            p = PositionPersistence(db)
            p.log_trade("SPY", "buy",  100, 450.0, 0.0,   "entry signal")
            p.log_trade("SPY", "sell", 100, 455.0, 500.0, "exit signal")

            history = p.get_trade_history(symbol="SPY")
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]["action"], "buy")
            self.assertEqual(history[1]["action"], "sell")
            self.assertAlmostEqual(history[1]["pnl"], 500.0)
        finally:
            db.unlink(missing_ok=True)

    def test_trade_history_since_filter(self):
        db = _tmp_db()
        try:
            p = PositionPersistence(db)
            p.log_trade("SPY", "buy", 100, 450.0, 0.0, "")
            cutoff = time.time()
            time.sleep(0.01)
            p.log_trade("SPY", "sell", 100, 455.0, 500.0, "")

            history = p.get_trade_history(symbol="SPY", since=cutoff)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["action"], "sell")
        finally:
            db.unlink(missing_ok=True)

    def test_get_cumulative_pnl(self):
        db = _tmp_db()
        try:
            p = PositionPersistence(db)
            p.log_trade("SPY", "close_long", 100, 455.0, 500.0, "")
            p.log_trade("SPY", "close_long",  50, 458.0, 200.0, "")
            total = p.get_cumulative_pnl(symbol="SPY")
            self.assertAlmostEqual(total, 700.0, places=4)
        finally:
            db.unlink(missing_ok=True)

    def test_delete_snapshot(self):
        db = _tmp_db()
        try:
            p = PositionPersistence(db)
            positions = {
                "SPY": Position(symbol="SPY", qty=100, avg_entry_px=450.0,
                                current_px=450.0),
            }
            p.save_snapshot(positions)
            p.delete_snapshot("SPY")
            loaded = p.load_snapshot()
            self.assertNotIn("SPY", loaded)
        finally:
            db.unlink(missing_ok=True)

    def test_get_trade_count(self):
        db = _tmp_db()
        try:
            p = PositionPersistence(db)
            for i in range(7):
                p.log_trade("BTC", "buy", 0.1, 60000.0, 0.0, "")
            self.assertEqual(p.get_trade_count("BTC"), 7)
        finally:
            db.unlink(missing_ok=True)


class TestPositionManagerWithPersistence(unittest.TestCase):

    def test_position_manager_saves_and_loads(self):
        db = _tmp_db()
        try:
            p   = PositionPersistence(db)
            mgr = PositionManager(persistence=p, equity_usd=100_000.0)

            _run(mgr.open_position("SPY", 100, 450.0, reason="test"))
            mgr.save_snapshot()

            mgr2   = PositionManager(persistence=p, equity_usd=100_000.0)
            n      = mgr2.load_from_db()
            self.assertEqual(n, 1)
            pos    = mgr2.get_position("SPY")
            self.assertIsNotNone(pos)
            self.assertAlmostEqual(pos.qty, 100.0)
        finally:
            db.unlink(missing_ok=True)

    def test_trade_logged_on_close(self):
        db = _tmp_db()
        try:
            p   = PositionPersistence(db)
            mgr = PositionManager(persistence=p)
            _run(mgr.open_position("BTC", 1.0, 60_000.0))
            _run(mgr.close_position("BTC", 1.0, 61_000.0, reason="tp"))
            history = p.get_trade_history(symbol="BTC")
            self.assertEqual(len(history), 2)
            close_records = [h for h in history if "close" in h["action"]]
            self.assertEqual(len(close_records), 1)
            self.assertAlmostEqual(close_records[0]["pnl"], 1_000.0, places=2)
        finally:
            db.unlink(missing_ok=True)


class TestPositionManagerConcurrency(unittest.TestCase):
    """Ensure thread-safety via sync API under concurrent access."""

    def test_concurrent_opens_do_not_corrupt_position(self):
        mgr = PositionManager(equity_usd=1_000_000.0)
        errors = []

        def worker(i):
            try:
                mgr.open_position_sync("SPY", 1.0, 450.0 + i * 0.01)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        pos = mgr.get_position("SPY")
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos.qty, 20.0, places=4)


class TestPositionDataclass(unittest.TestCase):

    def test_is_long_short_flat(self):
        p_long  = Position(symbol="SPY", qty=100.0)
        p_short = Position(symbol="SPY", qty=-50.0)
        p_flat  = Position(symbol="SPY", qty=0.0)
        self.assertTrue(p_long.is_long)
        self.assertFalse(p_long.is_short)
        self.assertTrue(p_short.is_short)
        self.assertFalse(p_short.is_long)
        self.assertTrue(p_flat.is_flat)

    def test_mark_updates_fields(self):
        p = Position(symbol="SPY", qty=100.0, avg_entry_px=450.0)
        p.mark(455.0)
        self.assertAlmostEqual(p.current_px, 455.0)
        self.assertAlmostEqual(p.unrealized_pnl, 500.0)
        self.assertAlmostEqual(p.cost_basis, 45_000.0)

    def test_total_pnl_deducts_commission(self):
        p = Position(symbol="SPY", qty=100.0, avg_entry_px=450.0, commission_paid=25.0)
        p.mark(455.0)
        self.assertAlmostEqual(p.total_pnl, 500.0 - 25.0, places=4)

    def test_to_dict_contains_all_keys(self):
        p = Position(symbol="SPY", qty=100.0, avg_entry_px=450.0, current_px=452.0)
        d = p.to_dict()
        for key in ("symbol", "qty", "avg_entry_px", "current_px", "unrealized_pnl",
                    "realized_pnl", "cost_basis", "commission_paid", "market_value",
                    "total_pnl"):
            self.assertIn(key, d)


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================


class TestEndToEndWorkflow(unittest.TestCase):
    """
    Simulate a simplified trade lifecycle:
    intent -> routing decision -> fill -> position update -> cost tracking
    """

    def test_full_trade_lifecycle(self):
        db   = _tmp_db()
        try:
            # Setup
            estimator = CostEstimator()
            tracker   = CostTracker(db_path=db)
            logger    = ExecutionLogger()
            router    = SmartRouter(estimator=estimator, logger=logger)
            lm        = LiquidityMap()
            lm.update_quote("SPY", "alpaca_equity", 450.0, 450.05, 10_000, 8_000)

            pers_db = Path(str(db) + ".pos.db")
            persistence = PositionPersistence(pers_db)
            mgr     = PositionManager(persistence=persistence, equity_usd=100_000.0)

            # Route the order
            intent = OrderIntent(
                symbol="SPY", target_qty=100, urgency=0.9,
                adv_usd=25_000_000, sigma_daily=0.01, asset_class="equity",
            )
            decisions = router.route(intent, lm)
            self.assertGreater(len(decisions), 0)

            # Simulate fill at ask price
            fill_px = 450.05
            for d in decisions:
                _run(mgr.open_position("SPY", d.qty, fill_px))

            # Mark to market
            _run(mgr.update_prices({"SPY": 450.50}))

            # Record cost
            pre_trade_est = estimator.estimate(
                "SPY", 100 * fill_px, "buy", "alpaca_equity", 25_000_000, 0.01
            )
            actual_cost_bps = router.implementation_shortfall(450.025, fill_px, "buy")
            tracker.record_fill("SPY", pre_trade_est.total_bps, actual_cost_bps, "alpaca_equity")

            # Verify positions
            pos = mgr.get_position("SPY")
            self.assertIsNotNone(pos)
            self.assertAlmostEqual(pos.qty, 100.0, places=4)
            self.assertGreater(pos.unrealized_pnl, 0)

            # Verify cost report
            report = tracker.get_slippage_report()
            self.assertEqual(report["n_fills"], 1)
        finally:
            db.unlink(missing_ok=True)
            Path(str(db) + ".pos.db").unlink(missing_ok=True)
            for f in [db.with_suffix(".db-shm"), db.with_suffix(".db-wal")]:
                f.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
