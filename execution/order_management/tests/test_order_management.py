"""
Tests for execution/order_management -- 25+ unit tests.

Run with: pytest execution/order_management/tests/test_order_management.py -v
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import List

import pytest

# -- module under test
from execution.order_management.order_types import (
    Fill,
    IcebergOrder,
    LimitOrder,
    MarketOrder,
    OrderFactory,
    OrderStatus,
    StopOrder,
    TWAPOrder,
    VWAPOrder,
    make_fill,
)
from execution.order_management.order_book_tracker import (
    MAX_OPEN_ORDERS_PER_SYMBOL,
    OrderBookTracker,
    OrderConflictChecker,
    OrderStateStore,
)
from execution.order_management.twap_engine import (
    EQUITY_VOLUME_PROFILE,
    TWAPEngine,
    VWAPEngine,
)
from execution.order_management.algo_scheduler import (
    AlgoScheduler,
    IcebergEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker() -> OrderBookTracker:
    t = OrderBookTracker()
    return t


@pytest.fixture
def db_store(tmp_path):
    db_path = str(tmp_path / "test_orders.db")
    return OrderStateStore(db_path=db_path)


@pytest.fixture
def checker() -> OrderConflictChecker:
    return OrderConflictChecker()


@pytest.fixture
def twap_order() -> TWAPOrder:
    now = datetime.utcnow()
    return OrderFactory.create_twap(
        symbol="AAPL",
        side="buy",
        qty=1000.0,
        start=now,
        end=now + timedelta(seconds=0.5),
        n_slices=5,
        strategy_id="strat-1",
    )


@pytest.fixture
def vwap_order() -> VWAPOrder:
    return OrderFactory.create_vwap(
        symbol="MSFT",
        side="sell",
        qty=500.0,
        strategy_id="strat-2",
        target_pct=0.05,
    )


@pytest.fixture
def iceberg_order() -> IcebergOrder:
    return OrderFactory.create_iceberg(
        symbol="GOOG",
        side="buy",
        qty=200.0,
        strategy_id="strat-3",
        display_pct=0.10,
        limit_price=150.0,
    )


# ---------------------------------------------------------------------------
# 1. OrderFactory -- create_market
# ---------------------------------------------------------------------------

class TestOrderFactory:

    def test_create_market_type(self):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        assert isinstance(o, MarketOrder)
        assert o.order_type == "MARKET"

    def test_create_market_fields(self):
        o = OrderFactory.create_market("SPY", "sell", 50.0, "s1", signal_strength=0.8)
        assert o.symbol == "SPY"
        assert o.side == "sell"
        assert o.qty == 50.0
        assert o.strategy_id == "s1"
        assert o.signal_strength == 0.8
        assert o.status == OrderStatus.NEW
        assert isinstance(o.order_id, str)
        assert len(o.order_id) == 36  # UUID format

    def test_create_limit_type(self):
        o = OrderFactory.create_limit("AAPL", "buy", 10.0, 150.0, "s2")
        assert isinstance(o, LimitOrder)
        assert o.order_type == "LIMIT"
        assert o.limit_price == 150.0
        assert o.time_in_force == "DAY"

    def test_create_limit_tif_variants(self):
        for tif in ("DAY", "GTC", "IOC", "FOK"):
            o = OrderFactory.create_limit("AAPL", "buy", 10.0, 150.0, "s2", tif=tif)
            assert o.time_in_force == tif

    def test_create_stop_plain(self):
        o = OrderFactory.create_stop("TSLA", "sell", 5.0, 200.0, "s3")
        assert isinstance(o, StopOrder)
        assert o.stop_price == 200.0
        assert o.limit_price is None
        assert not o.is_stop_limit

    def test_create_stop_limit(self):
        o = OrderFactory.create_stop("TSLA", "sell", 5.0, 200.0, "s3", limit_price=198.0)
        assert o.is_stop_limit
        assert o.limit_price == 198.0

    def test_create_twap_slices(self):
        now = datetime.utcnow()
        o = OrderFactory.create_twap(
            "NVDA", "buy", 600.0,
            start=now, end=now + timedelta(seconds=60),
            n_slices=6, strategy_id="s4"
        )
        assert isinstance(o, TWAPOrder)
        assert o.n_slices == 6
        assert abs(o.slice_interval_s - 10.0) < 1e-6
        assert abs(o.slice_qty - 100.0) < 1e-9

    def test_create_twap_interval_computed(self):
        now = datetime.utcnow()
        o = OrderFactory.create_twap(
            "NVDA", "buy", 1000.0,
            start=now, end=now + timedelta(seconds=100),
            n_slices=10, strategy_id="s4"
        )
        assert abs(o.slice_interval_s - 10.0) < 1e-6

    def test_create_vwap_default_profile(self):
        o = OrderFactory.create_vwap("IBM", "buy", 300.0, "s5")
        assert isinstance(o, VWAPOrder)
        assert len(o.volume_curve) == 48
        assert abs(sum(o.volume_curve) - 1.0) < 1e-4

    def test_create_iceberg_display_qty(self):
        o = OrderFactory.create_iceberg("AMZN", "buy", 1000.0, "s6", display_pct=0.05)
        assert isinstance(o, IcebergOrder)
        assert abs(o.display_qty - 50.0) < 1.0
        assert o.total_qty == 1000.0

    def test_factory_assigns_unique_ids(self):
        ids = {OrderFactory.create_market("SPY", "buy", 10.0, "s").order_id for _ in range(50)}
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# 2. BaseOrder fill tracking
# ---------------------------------------------------------------------------

class TestBaseOrderFills:

    def test_apply_fill_updates_qty(self):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        f = make_fill(o, 40.0, 450.0)
        o.apply_fill(f)
        assert abs(o.filled_qty - 40.0) < 1e-9
        assert abs(o.avg_fill_price - 450.0) < 1e-9
        assert o.status == OrderStatus.PARTIAL

    def test_apply_full_fill_sets_filled(self):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        f = make_fill(o, 100.0, 451.0)
        o.apply_fill(f)
        assert o.status == OrderStatus.FILLED
        assert abs(o.remaining_qty) < 1e-9

    def test_apply_multiple_fills_vwap(self):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        f1 = make_fill(o, 60.0, 400.0)
        f2 = make_fill(o, 40.0, 500.0)
        o.apply_fill(f1)
        o.apply_fill(f2)
        # VWAP = (60*400 + 40*500) / 100 = 440
        assert abs(o.avg_fill_price - 440.0) < 1e-6
        assert o.status == OrderStatus.FILLED

    def test_apply_fill_wrong_order_id_raises(self):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        bad_fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id="wrong-id",
            symbol="SPY", side="buy",
            qty=10.0, price=450.0,
            timestamp=datetime.utcnow(),
            venue="TEST",
            commission_bps=1.0,
        )
        with pytest.raises(ValueError, match="does not match"):
            o.apply_fill(bad_fill)

    def test_remaining_qty(self):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        assert abs(o.remaining_qty - 100.0) < 1e-9
        o.apply_fill(make_fill(o, 30.0, 450.0))
        assert abs(o.remaining_qty - 70.0) < 1e-9


# ---------------------------------------------------------------------------
# 3. OrderBookTracker
# ---------------------------------------------------------------------------

class TestOrderBookTracker:

    def test_add_and_get(self, tracker):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        tracker.add_order(o)
        assert tracker.get_order(o.order_id) is o

    def test_add_duplicate_raises(self, tracker):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        tracker.add_order(o)
        with pytest.raises(ValueError, match="already tracked"):
            tracker.add_order(o)

    def test_pending_orders_all(self, tracker):
        o1 = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        o2 = OrderFactory.create_market("AAPL", "sell", 50.0, "s1")
        tracker.add_order(o1)
        tracker.add_order(o2)
        pending = tracker.pending_orders()
        assert len(pending) == 2

    def test_pending_orders_by_symbol(self, tracker):
        o1 = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        o2 = OrderFactory.create_market("AAPL", "sell", 50.0, "s1")
        tracker.add_order(o1)
        tracker.add_order(o2)
        assert len(tracker.pending_orders("SPY")) == 1
        assert len(tracker.pending_orders("AAPL")) == 1
        assert len(tracker.pending_orders("TSLA")) == 0

    def test_update_status_fill(self, tracker):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        tracker.add_order(o)
        f = make_fill(o, 100.0, 450.0)
        tracker.update_status(o.order_id, OrderStatus.FILLED, fill=f)
        retrieved = tracker.get_order(o.order_id)
        assert retrieved.status == OrderStatus.FILLED
        assert abs(retrieved.filled_qty - 100.0) < 1e-9

    def test_filled_orders(self, tracker):
        o1 = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        o2 = OrderFactory.create_market("SPY", "buy", 50.0, "s1")
        tracker.add_order(o1)
        tracker.add_order(o2)
        tracker.update_status(o1.order_id, OrderStatus.FILLED, fill=make_fill(o1, 100.0, 450.0))
        filled = tracker.filled_orders()
        assert len(filled) == 1
        assert filled[0].order_id == o1.order_id

    def test_open_qty(self, tracker):
        o1 = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        o2 = OrderFactory.create_market("SPY", "sell", 60.0, "s1")
        tracker.add_order(o1)
        tracker.add_order(o2)
        buy_qty, sell_qty = tracker.open_qty("SPY")
        assert abs(buy_qty - 100.0) < 1e-9
        assert abs(sell_qty - 60.0) < 1e-9

    def test_fill_rate_zero_initially(self, tracker):
        assert tracker.fill_rate() == 0.0

    def test_fill_rate_calculation(self, tracker):
        for _ in range(8):
            o = OrderFactory.create_market("SPY", "buy", 10.0, "s1")
            tracker.add_order(o)
            tracker.update_status(o.order_id, OrderStatus.FILLED, fill=make_fill(o, 10.0, 100.0))
        for _ in range(2):
            o = OrderFactory.create_market("SPY", "buy", 10.0, "s1")
            tracker.add_order(o)
        # 8 filled out of 10
        assert abs(tracker.fill_rate() - 0.8) < 1e-9

    def test_daily_fills_returns_list(self, tracker):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        tracker.add_order(o)
        tracker.update_status(o.order_id, OrderStatus.FILLED, fill=make_fill(o, 100.0, 450.0))
        fills = tracker.daily_fills()
        assert len(fills) >= 1


# ---------------------------------------------------------------------------
# 4. OrderConflictChecker
# ---------------------------------------------------------------------------

class TestOrderConflictChecker:

    def test_no_conflict_empty(self, checker):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        result = checker.check_conflict(o, [])
        assert result is None

    def test_duplicate_conflict(self, checker):
        o1 = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        o1.status = OrderStatus.PENDING
        o2 = OrderFactory.create_market("SPY", "buy", 102.0, "s1")  # within 5%
        result = checker.check_conflict(o2, [o1])
        assert result is not None
        assert "Duplicate" in result

    def test_no_conflict_different_side(self, checker):
        o1 = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        o1.status = OrderStatus.PENDING
        o2 = OrderFactory.create_market("SPY", "sell", 100.0, "s1")
        result = checker.check_conflict(o2, [o1])
        assert result is None

    def test_concentration_limit(self, checker):
        open_orders = []
        for _ in range(MAX_OPEN_ORDERS_PER_SYMBOL):
            o = OrderFactory.create_market("SPY", "buy", 10.0, "s1")
            o.status = OrderStatus.PENDING
            open_orders.append(o)
        new_order = OrderFactory.create_market("SPY", "sell", 10.0, "s1")
        result = checker.check_conflict(new_order, open_orders)
        assert result is not None
        assert "Concentration" in result

    def test_no_conflict_different_symbol(self, checker):
        o1 = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        o1.status = OrderStatus.PENDING
        o2 = OrderFactory.create_market("AAPL", "buy", 100.0, "s1")
        result = checker.check_conflict(o2, [o1])
        assert result is None

    def test_filled_order_not_conflict(self, checker):
        o1 = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        o1.status = OrderStatus.FILLED
        o2 = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        result = checker.check_conflict(o2, [o1])
        assert result is None


# ---------------------------------------------------------------------------
# 5. SQLite persistence (OrderStateStore)
# ---------------------------------------------------------------------------

class TestOrderStateStore:

    def test_persist_and_load(self, db_store):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        db_store.persist(o)
        open_orders = db_store.load_open_orders()
        ids = [ord_.order_id for ord_ in open_orders]
        assert o.order_id in ids

    def test_persist_limit_order(self, db_store):
        o = OrderFactory.create_limit("AAPL", "buy", 10.0, 150.0, "s2", tif="GTC")
        db_store.persist(o)
        open_orders = db_store.load_open_orders()
        restored = next((x for x in open_orders if x.order_id == o.order_id), None)
        assert restored is not None
        assert isinstance(restored, LimitOrder)
        assert abs(restored.limit_price - 150.0) < 1e-9
        assert restored.time_in_force == "GTC"

    def test_persist_twap_order(self, db_store):
        now = datetime.utcnow()
        o = OrderFactory.create_twap(
            "NVDA", "buy", 1000.0,
            start=now, end=now + timedelta(seconds=60),
            n_slices=6, strategy_id="s3"
        )
        db_store.persist(o)
        open_orders = db_store.load_open_orders()
        restored = next((x for x in open_orders if x.order_id == o.order_id), None)
        assert restored is not None
        assert isinstance(restored, TWAPOrder)
        assert restored.n_slices == 6

    def test_update_status(self, db_store):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        db_store.persist(o)
        db_store.update(o.order_id, OrderStatus.FILLED)
        # FILLED orders should not appear in load_open_orders
        open_orders = db_store.load_open_orders()
        ids = [ord_.order_id for ord_ in open_orders]
        assert o.order_id not in ids

    def test_persist_and_retrieve_fill(self, db_store):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        db_store.persist(o)
        f = make_fill(o, 100.0, 450.0)
        db_store.update(o.order_id, OrderStatus.FILLED, fill=f)
        fills = db_store.load_fills_since(datetime.utcfromtimestamp(0))
        fill_ids = [fil.fill_id for fil in fills]
        assert f.fill_id in fill_ids

    def test_load_fills_since_filter(self, db_store):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        db_store.persist(o)
        f = make_fill(o, 100.0, 450.0)
        db_store.update(o.order_id, OrderStatus.FILLED, fill=f)
        # Query with future timestamp -- should return nothing
        future = datetime(2099, 1, 1)
        fills = db_store.load_fills_since(future)
        assert len(fills) == 0

    def test_persist_duplicate_ignored(self, db_store):
        o = OrderFactory.create_market("SPY", "buy", 100.0, "s1")
        db_store.persist(o)
        db_store.persist(o)  # Should not raise
        open_orders = db_store.load_open_orders()
        matching = [x for x in open_orders if x.order_id == o.order_id]
        assert len(matching) == 1


# ---------------------------------------------------------------------------
# 6. TWAPEngine
# ---------------------------------------------------------------------------

class TestTWAPEngine:

    def test_submit_returns_execution_id(self, twap_order):
        engine = TWAPEngine()
        eid = engine.submit(twap_order)
        assert isinstance(eid, str)
        assert len(eid) == 36

    def test_status_initial(self, twap_order):
        engine = TWAPEngine()
        eid = engine.submit(twap_order)
        s = engine.status(eid)
        assert s.total_qty == twap_order.qty
        assert s.slices_total == twap_order.n_slices

    def test_twap_fills_over_time(self, twap_order):
        fills: List[Fill] = []
        engine = TWAPEngine(fill_callback=fills.append)
        eid = engine.submit(twap_order)
        # Wait for all 5 slices (interval=2s, but simulated instantly)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            s = engine.status(eid)
            if s.is_complete:
                break
            time.sleep(0.1)
        s = engine.status(eid)
        assert s.is_complete
        assert s.slices_completed == twap_order.n_slices
        assert abs(s.filled_qty - twap_order.qty) < 1e-6

    def test_cancel_stops_execution(self):
        now = datetime.utcnow()
        o = OrderFactory.create_twap(
            "SPY", "buy", 1000.0,
            start=now, end=now + timedelta(seconds=10),
            n_slices=50, strategy_id="s1"
        )
        engine = TWAPEngine()
        eid = engine.submit(o)
        time.sleep(0.05)
        engine.cancel(eid)
        time.sleep(0.1)
        s = engine.status(eid)
        # Not all slices should be complete
        assert s.slices_completed < 50

    def test_status_raises_for_unknown(self):
        engine = TWAPEngine()
        with pytest.raises(KeyError):
            engine.status("nonexistent-id")

    def test_slice_qty_correct(self, twap_order):
        assert abs(twap_order.slice_qty - 200.0) < 1e-9  # 1000 / 5


# ---------------------------------------------------------------------------
# 7. VWAPEngine
# ---------------------------------------------------------------------------

class TestVWAPEngine:

    def test_equity_volume_profile_sums_to_one(self):
        assert abs(sum(EQUITY_VOLUME_PROFILE) - 1.0) < 1e-4
        assert len(EQUITY_VOLUME_PROFILE) == 48

    def test_submit_returns_id(self, vwap_order):
        engine = VWAPEngine()
        eid = engine.submit(vwap_order)
        assert isinstance(eid, str)

    def test_vwap_qty_for_bucket(self, vwap_order):
        # Bucket 0 should have qty proportional to EQUITY_VOLUME_PROFILE[0]
        expected = vwap_order.qty * EQUITY_VOLUME_PROFILE[0]
        assert abs(vwap_order.qty_for_bucket(0) - expected) < 1e-6

    def test_cancel_vwap(self, vwap_order):
        engine = VWAPEngine()
        eid = engine.submit(vwap_order)
        time.sleep(0.02)
        engine.cancel(eid)  # Should not raise


# ---------------------------------------------------------------------------
# 8. IcebergEngine
# ---------------------------------------------------------------------------

class TestIcebergEngine:

    def test_submit_creates_display_order(self, iceberg_order):
        engine = IcebergEngine()
        eid = engine.submit(iceberg_order, limit_price=150.0)
        ctx = engine._executions[eid]
        assert ctx.current_display_order is not None

    def test_fill_triggers_resubmit(self, iceberg_order):
        submitted: List[LimitOrder] = []
        engine = IcebergEngine(submit_order_callback=submitted.append)
        eid = engine.submit(iceberg_order, limit_price=150.0)
        assert len(submitted) == 1  # first display order

        # Simulate fill of the first display order
        engine.simulate_fill_current_display(eid)
        assert len(submitted) == 2  # second display order submitted

    def test_iceberg_completes_after_all_fills(self):
        o = OrderFactory.create_iceberg(
            "TEST", "buy", 20.0, "s1", display_pct=0.50
        )
        submitted: List[LimitOrder] = []
        engine = IcebergEngine(submit_order_callback=submitted.append)
        eid = engine.submit(o, limit_price=100.0)

        # Keep filling until complete
        for _ in range(20):
            fill = engine.simulate_fill_current_display(eid)
            if fill is None:
                break

        s = engine.status(eid)
        assert s.is_complete
        assert abs(s.filled_qty - o.total_qty) < 1.0

    def test_display_qty_jitter(self, iceberg_order):
        """Jittered display_qty should be within [0.9 * base, 1.1 * base]."""
        engine = IcebergEngine()
        eid = engine.submit(iceberg_order, limit_price=150.0)
        ctx = engine._executions[eid]
        base = iceberg_order.display_qty
        for _ in range(50):
            jittered = ctx.jitter_display_qty()
            assert jittered >= base * 0.9 - 1.0
            assert jittered <= min(base * 1.1 + 1.0, iceberg_order.total_qty)

    def test_cancel_iceberg(self, iceberg_order):
        engine = IcebergEngine()
        eid = engine.submit(iceberg_order, limit_price=150.0)
        engine.cancel(eid)
        ctx = engine._executions[eid]
        assert ctx.cancelled


# ---------------------------------------------------------------------------
# 9. AlgoScheduler
# ---------------------------------------------------------------------------

class TestAlgoScheduler:

    def test_submit_twap(self, twap_order):
        scheduler = AlgoScheduler()
        eid = scheduler.submit_algo(twap_order)
        assert eid is not None
        ae = scheduler.get_algo_execution(eid)
        assert ae is not None
        assert ae.order_type == "TWAP"

    def test_submit_vwap(self, vwap_order):
        scheduler = AlgoScheduler()
        eid = scheduler.submit_algo(vwap_order)
        ae = scheduler.get_algo_execution(eid)
        assert ae.order_type == "VWAP"

    def test_submit_iceberg(self, iceberg_order):
        scheduler = AlgoScheduler()
        eid = scheduler.submit_algo(iceberg_order)
        ae = scheduler.get_algo_execution(eid)
        assert ae.order_type == "ICEBERG"

    def test_cancel_all_no_symbol(self, twap_order, vwap_order):
        now = datetime.utcnow()
        t1 = OrderFactory.create_twap(
            "SPY", "buy", 1000.0,
            start=now, end=now + timedelta(seconds=60),
            n_slices=30, strategy_id="s1"
        )
        t2 = OrderFactory.create_twap(
            "AAPL", "buy", 500.0,
            start=now, end=now + timedelta(seconds=60),
            n_slices=30, strategy_id="s1"
        )
        scheduler = AlgoScheduler()
        scheduler.submit_algo(t1)
        scheduler.submit_algo(t2)
        n = scheduler.cancel_all()
        assert n == 2

    def test_daily_algo_summary_keys(self, twap_order):
        fills: List[Fill] = []
        scheduler = AlgoScheduler(fill_callback=fills.append)
        scheduler.submit_algo(twap_order)
        time.sleep(0.05)
        summary = scheduler.daily_algo_summary()
        for key in ("total_algo_volume_usd", "total_fills", "avg_fill_price",
                    "avg_slippage_bps", "completion_rate"):
            assert key in summary

    def test_get_all_active(self):
        now = datetime.utcnow()
        o = OrderFactory.create_twap(
            "SPY", "buy", 1000.0,
            start=now, end=now + timedelta(seconds=60),
            n_slices=30, strategy_id="s1"
        )
        scheduler = AlgoScheduler()
        eid = scheduler.submit_algo(o)
        active = scheduler.get_all_active()
        eids = [ae.execution_id for ae in active]
        assert eid in eids

    def test_unsupported_order_type_raises(self):
        scheduler = AlgoScheduler()
        o = OrderFactory.create_market("SPY", "buy", 10.0, "s1")
        with pytest.raises(TypeError, match="Unsupported algo order type"):
            scheduler.submit_algo(o)  # type: ignore[arg-type]
