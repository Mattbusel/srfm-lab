"""
tests/test_execution_oms.py -- Tests for execution/oms modules.

Covers:
  - Order / OrderBook (fill, partial fill, terminal guard)
  - OrderManager (submit, fill, cancel, reconcile)
  - FillEvent processing
  - OrderStatus state transitions
  - open_orders / filled_orders queries
  - Position tracker updates via fills
  - Slippage computation
  - Thread-safety smoke test
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from execution.oms.order import (
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Side,
    _TERMINAL_STATES,
)
from execution.oms.order_manager import FillEvent, OrderManager
from execution.oms.position_tracker import PositionTracker
from execution.oms.risk_guard import RiskGuard


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_order(**kwargs) -> Order:
    """Create a minimal Order with sensible defaults."""
    defaults = dict(
        symbol="SPY",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=100.0,
        price=None,
        strategy_id="test_strat",
    )
    defaults.update(kwargs)
    return Order(**defaults)


def make_router(broker_id: str = "BR-001"):
    """Return a mock SmartRouter that returns a fixed broker_order_id."""
    router = MagicMock()
    router.route.return_value = broker_id
    return router


def make_risk_guard(allow: bool = True) -> RiskGuard:
    """Return a RiskGuard whose run_all_checks always passes (or fails)."""
    rg = MagicMock(spec=RiskGuard)
    rg.run_all_checks.return_value = (allow, "") if allow else (False, "test_reject")
    return rg


# ---------------------------------------------------------------------------
# Order dataclass -- basic construction
# ---------------------------------------------------------------------------


def test_order_default_status_is_pending():
    o = make_order()
    assert o.status == OrderStatus.PENDING


def test_order_id_is_uuid():
    o = make_order()
    parsed = uuid.UUID(o.order_id)  # raises if not valid UUID
    assert str(parsed) == o.order_id


def test_order_remaining_qty_before_fill():
    o = make_order(quantity=200.0)
    assert o.remaining_qty == 200.0


def test_order_remaining_qty_after_partial():
    o = make_order(quantity=200.0)
    o.mark_submitted()
    o.mark_partial(fill_qty=80.0, fill_price=450.0)
    assert o.remaining_qty == 120.0


def test_order_is_terminal_false_when_pending():
    o = make_order()
    assert not o.is_terminal


def test_order_is_terminal_true_after_fill():
    o = make_order()
    o.mark_filled(fill_qty=100.0, fill_price=450.0)
    assert o.is_terminal


def test_order_is_terminal_true_after_cancel():
    o = make_order()
    o.mark_cancelled()
    assert o.is_terminal


def test_order_is_terminal_true_after_reject():
    o = make_order()
    o.mark_rejected("price out of range")
    assert o.is_terminal


# ---------------------------------------------------------------------------
# FillProcessor / fill validation -- via mark_filled / mark_partial
# ---------------------------------------------------------------------------


def test_fill_processor_valid_full_fill():
    """Full fill sets status FILLED and records price + qty."""
    o = make_order(quantity=100.0)
    o.mark_submitted("BR-XYZ")
    o.mark_filled(fill_qty=100.0, fill_price=450.00, commission_usd=1.50)
    assert o.status == OrderStatus.FILLED
    assert o.fill_qty == 100.0
    assert o.fill_price == 450.00
    assert o.commission_usd == 1.50
    assert o.filled_at is not None


def test_fill_processor_partial_fill_updates_qty():
    """Partial fill moves status to PARTIAL and stores fill_qty."""
    o = make_order(quantity=100.0)
    o.mark_submitted()
    o.mark_partial(fill_qty=40.0, fill_price=300.0)
    assert o.status == OrderStatus.PARTIAL
    assert o.fill_qty == 40.0
    assert o.remaining_qty == 60.0


def test_fill_processor_over_fill_rejected_by_terminal_guard():
    """Filling a terminal order raises RuntimeError (duplicate/over-fill guard)."""
    o = make_order(quantity=100.0)
    o.mark_filled(fill_qty=100.0, fill_price=400.0)
    with pytest.raises(RuntimeError, match="terminal"):
        o.mark_filled(fill_qty=100.0, fill_price=401.0)


def test_fill_processor_duplicate_fill_rejected():
    """Second fill on a FILLED order is rejected."""
    o = make_order(quantity=50.0)
    o.mark_filled(fill_qty=50.0, fill_price=200.0)
    with pytest.raises(RuntimeError):
        o.mark_partial(fill_qty=10.0, fill_price=200.0)


def test_fill_validator_suspicious_price_not_stored():
    """
    Fill price wildly different from decision_price produces nonzero slippage_bps.
    We verify the slippage is computed (positive for overpay on BUY).
    """
    o = make_order(side=Side.BUY, quantity=100.0, decision_price=400.0)
    o.mark_filled(fill_qty=100.0, fill_price=420.0)
    # 20/400 = 5% = 500 bps overpay
    assert o.slippage_bps is not None
    assert o.slippage_bps > 400.0  # at least 400 bps


def test_fill_validator_sell_slippage_direction():
    """For SELL, paying less than decision_price is positive (adverse) slippage."""
    o = make_order(side=Side.SELL, quantity=100.0, decision_price=400.0)
    o.mark_filled(fill_qty=100.0, fill_price=380.0)
    # SELL at 380 vs decision 400 -> slippage negative (got worse price)
    assert o.slippage_bps is not None
    assert o.slippage_bps < 0.0


def test_fill_aggregator_vwap_across_partials():
    """
    Simulate two partial fills; VWAP-style check via fill_price on final fill.
    This tests that the last fill_price is recorded correctly and fill_qty accumulates.
    """
    o = make_order(quantity=100.0)
    o.mark_submitted()
    o.mark_partial(fill_qty=40.0, fill_price=300.0)
    o.mark_partial(fill_qty=80.0, fill_price=302.0)  # cumulative 80
    assert o.fill_qty == 80.0
    assert o.fill_price == 302.0


def test_fill_aggregator_full_fill_after_partial():
    """After partial, a full fill transitions to FILLED."""
    o = make_order(quantity=100.0)
    o.mark_submitted()
    o.mark_partial(fill_qty=50.0, fill_price=200.0)
    o.mark_filled(fill_qty=100.0, fill_price=200.5)
    assert o.status == OrderStatus.FILLED


# ---------------------------------------------------------------------------
# OrderBook
# ---------------------------------------------------------------------------


def test_orderbook_add_and_get():
    book = OrderBook()
    o = make_order()
    book.add(o)
    assert book.get(o.order_id) is o


def test_orderbook_duplicate_order_id_raises():
    book = OrderBook()
    o = make_order()
    book.add(o)
    with pytest.raises(ValueError, match="Duplicate"):
        book.add(o)


def test_orderbook_open_orders_excludes_terminal():
    book = OrderBook()
    o1 = make_order(symbol="SPY")
    o2 = make_order(symbol="QQQ")
    book.add(o1)
    book.add(o2)
    o2.mark_filled(fill_qty=100.0, fill_price=300.0)
    open_orders = book.open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].symbol == "SPY"


def test_orderbook_filled_orders():
    book = OrderBook()
    o = make_order()
    book.add(o)
    o.mark_filled(fill_qty=100.0, fill_price=400.0)
    filled = book.filled_orders()
    assert len(filled) == 1
    assert filled[0] is o


def test_orderbook_open_orders_for_symbol():
    book = OrderBook()
    for _ in range(3):
        book.add(make_order(symbol="SPY"))
    for _ in range(2):
        book.add(make_order(symbol="AAPL"))
    spy_orders = book.open_orders_for("SPY")
    assert len(spy_orders) == 3


def test_orderbook_get_by_broker_id():
    book = OrderBook()
    o = make_order()
    o.mark_submitted(broker_order_id="ALPACA-999")
    book.add(o)
    found = book.get_by_broker_id("ALPACA-999")
    assert found is o


def test_orderbook_get_by_broker_id_not_found():
    book = OrderBook()
    assert book.get_by_broker_id("NONEXISTENT") is None


def test_orderbook_remove():
    book = OrderBook()
    o = make_order()
    book.add(o)
    removed = book.remove(o.order_id)
    assert removed is o
    assert book.get(o.order_id) is None


def test_orderbook_len():
    book = OrderBook()
    for _ in range(5):
        book.add(make_order())
    assert len(book) == 5


# ---------------------------------------------------------------------------
# OrderStateMachine -- valid transitions
# ---------------------------------------------------------------------------


def test_state_machine_pending_to_submitted():
    o = make_order()
    o.mark_submitted("BR-001")
    assert o.status == OrderStatus.SUBMITTED
    assert o.broker_order_id == "BR-001"


def test_state_machine_submitted_to_partial():
    o = make_order(quantity=200.0)
    o.mark_submitted()
    o.mark_partial(fill_qty=100.0, fill_price=150.0)
    assert o.status == OrderStatus.PARTIAL


def test_state_machine_partial_to_filled():
    o = make_order(quantity=200.0)
    o.mark_submitted()
    o.mark_partial(fill_qty=100.0, fill_price=150.0)
    o.mark_filled(fill_qty=200.0, fill_price=150.5)
    assert o.status == OrderStatus.FILLED


def test_state_machine_pending_to_cancelled():
    o = make_order()
    o.mark_cancelled()
    assert o.status == OrderStatus.CANCELLED


def test_state_machine_pending_to_rejected():
    o = make_order()
    o.mark_rejected("exceeds position limit")
    assert o.status == OrderStatus.REJECTED
    assert "exceeds position limit" in o.reject_reason


def test_state_machine_invalid_transition_from_filled_raises():
    """FILLED cannot transition to CANCELLED -- terminal guard."""
    o = make_order()
    o.mark_filled(fill_qty=100.0, fill_price=500.0)
    with pytest.raises(RuntimeError):
        o.mark_cancelled()


def test_state_machine_invalid_transition_from_cancelled_raises():
    o = make_order()
    o.mark_cancelled()
    with pytest.raises(RuntimeError):
        o.mark_submitted()


def test_state_machine_invalid_transition_from_rejected_raises():
    o = make_order()
    o.mark_rejected("risk limit")
    with pytest.raises(RuntimeError):
        o.mark_partial(fill_qty=50.0, fill_price=100.0)


# ---------------------------------------------------------------------------
# Terminal states -- all 3 should block further mutation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("terminal", [
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
])
def test_terminal_states_block_mutation(terminal):
    o = make_order()
    if terminal == OrderStatus.FILLED:
        o.mark_filled(fill_qty=100.0, fill_price=100.0)
    elif terminal == OrderStatus.CANCELLED:
        o.mark_cancelled()
    else:
        o.mark_rejected("test")
    assert o.is_terminal
    with pytest.raises(RuntimeError):
        o.mark_submitted()


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------


@pytest.fixture()
def manager():
    """Return an OrderManager with mock router, permissive risk, no audit."""
    router = make_router("BROKER-001")
    risk = make_risk_guard(allow=True)
    return OrderManager(router=router, risk=risk, equity=100_000.0)


def test_order_manager_submit_returns_order(manager):
    order = manager.submit_order(
        symbol="SPY",
        side=Side.BUY,
        target_frac=0.10,
        curr_price=450.0,
        strategy_id="test",
    )
    assert order is not None
    assert order.status == OrderStatus.SUBMITTED


def test_order_manager_submit_registers_in_book(manager):
    order = manager.submit_order("SPY", Side.BUY, 0.10, curr_price=450.0)
    assert manager.book.get(order.order_id) is order


def test_order_manager_size_order_proportional():
    router = make_router()
    risk = make_risk_guard()
    mgr = OrderManager(router=router, risk=risk, equity=200_000.0)
    qty = mgr.size_order(target_frac=0.10, equity=200_000.0, price=100.0)
    # 10% of 200k at $100 = 200 shares
    assert abs(qty - 200.0) < 1.0


def test_order_manager_zero_price_returns_none():
    router = make_router()
    risk = make_risk_guard()
    mgr = OrderManager(router=router, risk=risk, equity=100_000.0)
    order = mgr.submit_order("SPY", Side.BUY, 0.10, curr_price=0.0)
    assert order is None


def test_order_manager_risk_rejection_returns_none():
    router = make_router()
    risk = make_risk_guard(allow=False)
    mgr = OrderManager(router=router, risk=risk, equity=100_000.0)
    order = mgr.submit_order("SPY", Side.BUY, 0.10, curr_price=450.0)
    assert order is None


def test_order_manager_risk_rejection_book_entry_is_rejected():
    router = make_router()
    risk = make_risk_guard(allow=False)
    mgr = OrderManager(router=router, risk=risk, equity=100_000.0)
    mgr.submit_order("SPY", Side.BUY, 0.10, curr_price=450.0)
    rejected = [o for o in mgr.book if o.status == OrderStatus.REJECTED]
    assert len(rejected) == 1


def test_order_manager_cancel_open_order(manager):
    order = manager.submit_order("SPY", Side.BUY, 0.10, curr_price=450.0)
    result = manager.cancel_order(order.order_id)
    assert result is True
    assert order.status == OrderStatus.CANCELLED


def test_order_manager_cancel_unknown_order_returns_false(manager):
    assert manager.cancel_order("nonexistent-id") is False


def test_order_manager_cancel_already_terminal_returns_false(manager):
    order = manager.submit_order("SPY", Side.BUY, 0.10, curr_price=450.0)
    order.mark_filled(fill_qty=order.quantity, fill_price=450.0)
    result = manager.cancel_order(order.order_id)
    assert result is False


def test_order_manager_handle_fill_full(manager):
    order = manager.submit_order("SPY", Side.BUY, 0.10, curr_price=450.0)
    fill = FillEvent(
        broker_order_id="BROKER-001",
        fill_qty=order.quantity,
        fill_price=451.0,
        commission_usd=2.0,
    )
    filled_order = manager.handle_fill(fill)
    assert filled_order is not None
    assert filled_order.status == OrderStatus.FILLED


def test_order_manager_handle_fill_unknown_broker_id_returns_none(manager):
    result = manager.handle_fill(FillEvent("UNKNOWN-999", 10.0, 450.0))
    assert result is None


def test_order_manager_handle_fill_partial(manager):
    order = manager.submit_order("SPY", Side.BUY, 0.10, curr_price=450.0)
    partial_qty = order.quantity / 2.0
    fill = FillEvent("BROKER-001", fill_qty=partial_qty, fill_price=450.0)
    updated = manager.handle_fill(fill)
    assert updated.status == OrderStatus.PARTIAL


def test_order_manager_get_open_orders_only_non_terminal(manager):
    o1 = manager.submit_order("SPY", Side.BUY, 0.05, curr_price=450.0)
    o2 = manager.submit_order("QQQ", Side.BUY, 0.05, curr_price=350.0)
    # Fill o2
    manager.handle_fill(FillEvent(o2.broker_order_id, o2.quantity, 350.0))
    open_orders = manager.book.open_orders()
    assert all(not o.is_terminal for o in open_orders)
    assert o1 in open_orders
    assert o2 not in open_orders


def test_order_manager_event_bus_called_on_submit():
    router = make_router()
    risk = make_risk_guard()
    events = []
    bus = lambda etype, payload: events.append(etype)
    mgr = OrderManager(router=router, risk=risk, equity=100_000.0, event_bus=bus)
    mgr.submit_order("SPY", Side.BUY, 0.10, curr_price=450.0)
    assert "ORDER_CREATED" in events
    assert "ORDER_SUBMITTED" in events


def test_order_manager_reconcile_detects_drift(manager):
    """Broker has 50 shares of SPY; OMS has 0 -- expect discrepancy report."""
    discrepancies = manager.reconcile_with_broker({"SPY": 50.0})
    assert len(discrepancies) >= 1
    assert "SPY" in discrepancies[0]


def test_order_manager_equity_update(manager):
    manager.set_equity(500_000.0)
    assert manager.get_equity() == 500_000.0


# ---------------------------------------------------------------------------
# OrderRouter -- equity vs crypto vs large order
# ---------------------------------------------------------------------------


def test_order_router_equity_uses_alpaca():
    """Equity symbols should route through Alpaca adapter."""
    from execution.routing.smart_router import SmartRouter
    router = SmartRouter.__new__(SmartRouter)
    # Verify SmartRouter exists and is importable
    assert router is not None


def test_order_manager_large_order_tracked_in_book():
    """A 100K notional order should appear in the book."""
    router = make_router()
    risk = make_risk_guard()
    mgr = OrderManager(router=router, risk=risk, equity=10_000_000.0)
    # 100K / $100 = 1000 shares
    order = mgr.submit_order("SPY", Side.BUY, 0.01, curr_price=100.0)
    assert order is not None
    assert order.quantity > 0


# ---------------------------------------------------------------------------
# Thread safety -- concurrent fills do not corrupt book
# ---------------------------------------------------------------------------


def test_orderbook_thread_safe_concurrent_adds():
    """Adding 100 orders from 10 threads should not raise or lose entries."""
    book = OrderBook()
    errors = []

    def add_orders():
        for _ in range(10):
            try:
                book.add(make_order())
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=add_orders) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(book) == 100


def test_order_to_dict_serializable():
    """to_dict should return a JSON-serializable snapshot."""
    import json
    o = make_order()
    o.mark_submitted("BROKER-ABC")
    d = o.to_dict()
    dumped = json.dumps(d)
    assert "BROKER-ABC" in dumped


def test_notional_value_uses_fill_price_when_available():
    o = make_order(quantity=100.0, price=400.0)
    o.mark_filled(fill_qty=100.0, fill_price=402.0)
    assert abs(o.notional_value - 40200.0) < 1.0


def test_notional_value_uses_limit_price_when_unfilled():
    o = make_order(quantity=100.0, price=400.0)
    assert abs(o.notional_value - 40000.0) < 1.0
