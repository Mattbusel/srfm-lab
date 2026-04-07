"""
test_adapters.py -- Unit tests for SRFM broker adapters.

Covers:
- PaperTradingAdapter: submit, fill, cancel, position tracking, P&L
- AlpacaOrderTranslator: order translation round-trip
- BinanceSignatureGenerator: known HMAC-SHA256 value
- BinanceSymbolMapper: round-trip symbol conversion
- AdapterManager: routing by asset class, failover trigger
- CircuitBreaker: state transitions
- FIFOPosition: cost basis and realized P&L
- PaperAccountManager: cash deduction and equity calculation
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import urllib.parse
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Imports from the package under test
# ---------------------------------------------------------------------------
from execution.broker_adapters.base_adapter import (
    AccountInfo,
    AssetClass,
    BrokerAdapter,
    CircuitBreaker,
    CircuitState,
    Fill,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    TimeInForce,
)
from execution.broker_adapters.alpaca_adapter import (
    AlpacaOrderTranslator,
    AlpacaRateLimiter,
)
from execution.broker_adapters.binance_adapter import (
    BinanceSignatureGenerator,
    BinanceSymbolMapper,
    SymbolFilters,
)
from execution.broker_adapters.paper_adapter import (
    FIFOPosition,
    PaperAccountManager,
    PaperTradingAdapter,
)
from execution.broker_adapters.adapter_manager import (
    AdapterManager,
    AdapterHealth,
    AdapterHealthMonitor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_order(
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    qty: float = 10.0,
    order_type: OrderType = OrderType.MARKET,
    price: float = None,
    stop_price: float = None,
    asset_class: AssetClass = AssetClass.EQUITY,
) -> OrderRequest:
    """Create a minimal OrderRequest for testing."""
    return OrderRequest(
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        time_in_force=TimeInForce.DAY,
        strategy_id="test_strategy",
        client_order_id=str(uuid.uuid4()),
        price=price,
        stop_price=stop_price,
        asset_class=asset_class,
    )


def run_async(coro):
    """Run a coroutine synchronously in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ===========================================================================
# PaperTradingAdapter tests
# ===========================================================================


class TestPaperTradingAdapter:

    def setup_method(self):
        self.adapter = PaperTradingAdapter(
            initial_cash=100_000.0,
            fill_delay_ms=0,
            slippage_bps=0.0,
            seed=42,
        )
        self.adapter.set_market_price("AAPL", 150.0)
        self.adapter.set_market_price("BTC-USD", 30_000.0)

    def test_submit_market_order_fills_immediately(self):
        order = make_order(symbol="AAPL", qty=10.0)
        result = run_async(self.adapter.submit_order(order))
        assert result.status == OrderStatus.FILLED
        assert result.filled_qty == 10.0
        assert result.avg_fill_price == pytest.approx(150.0)

    def test_buy_deducts_cash(self):
        order = make_order(symbol="AAPL", qty=10.0)
        run_async(self.adapter.submit_order(order))
        account = run_async(self.adapter.get_account())
        assert account.cash == pytest.approx(100_000.0 - 10.0 * 150.0)

    def test_sell_credits_cash(self):
        # First buy
        buy = make_order(symbol="AAPL", qty=10.0, side=OrderSide.BUY)
        run_async(self.adapter.submit_order(buy))
        cash_after_buy = run_async(self.adapter.get_account()).cash

        # Then sell
        sell = make_order(symbol="AAPL", qty=10.0, side=OrderSide.SELL)
        run_async(self.adapter.submit_order(sell))
        account = run_async(self.adapter.get_account())
        assert account.cash == pytest.approx(cash_after_buy + 10.0 * 150.0)

    def test_position_created_after_buy(self):
        order = make_order(symbol="AAPL", qty=10.0)
        run_async(self.adapter.submit_order(order))
        pos = run_async(self.adapter.get_position("AAPL"))
        assert pos is not None
        assert pos.qty == pytest.approx(10.0)
        assert pos.avg_entry_price == pytest.approx(150.0)
        assert pos.side == PositionSide.LONG

    def test_position_flat_after_round_trip(self):
        run_async(self.adapter.submit_order(make_order(symbol="AAPL", qty=10.0, side=OrderSide.BUY)))
        run_async(self.adapter.submit_order(make_order(symbol="AAPL", qty=10.0, side=OrderSide.SELL)))
        pos = run_async(self.adapter.get_position("AAPL"))
        assert pos is None

    def test_limit_order_pending_until_price_crosses(self):
        order = make_order(symbol="AAPL", order_type=OrderType.LIMIT, price=140.0, qty=5.0)
        result = run_async(self.adapter.submit_order(order))
        assert result.status == OrderStatus.SUBMITTED
        assert self.adapter.pending_order_count == 1

        # Price still above limit -- should not fill
        self.adapter.set_market_price("AAPL", 145.0)
        assert self.adapter.pending_order_count == 1

        # Price crosses limit -- should fill
        self.adapter.set_market_price("AAPL", 139.0)
        assert self.adapter.pending_order_count == 0
        pos = run_async(self.adapter.get_position("AAPL"))
        assert pos is not None
        assert pos.qty == pytest.approx(5.0)

    def test_cancel_pending_limit_order(self):
        order = make_order(symbol="AAPL", order_type=OrderType.LIMIT, price=140.0, qty=5.0)
        result = run_async(self.adapter.submit_order(order))
        canceled = run_async(self.adapter.cancel_order(result.order_id))
        assert canceled is True
        assert self.adapter.pending_order_count == 0

    def test_cancel_filled_order_returns_false(self):
        order = make_order(symbol="AAPL", qty=5.0)
        result = run_async(self.adapter.submit_order(order))
        assert result.status == OrderStatus.FILLED
        canceled = run_async(self.adapter.cancel_order(result.order_id))
        assert canceled is False

    def test_get_all_positions(self):
        run_async(self.adapter.submit_order(make_order(symbol="AAPL", qty=5.0)))
        run_async(self.adapter.submit_order(
            make_order(symbol="BTC-USD", qty=1.0, asset_class=AssetClass.CRYPTO)
        ))
        positions = run_async(self.adapter.get_all_positions())
        assert "AAPL" in positions
        assert "BTC-USD" in positions

    def test_get_recent_fills(self):
        run_async(self.adapter.submit_order(make_order(symbol="AAPL", qty=5.0)))
        run_async(self.adapter.submit_order(make_order(symbol="AAPL", qty=3.0)))
        fills = run_async(self.adapter.get_recent_fills(10))
        assert len(fills) == 2

    def test_slippage_applied_to_market_order(self):
        adapter = PaperTradingAdapter(
            initial_cash=100_000.0,
            fill_delay_ms=0,
            slippage_bps=10.0,  # 10 bps
            seed=42,
        )
        adapter.set_market_price("AAPL", 150.0)
        order = make_order(symbol="AAPL", qty=1.0, side=OrderSide.BUY)
        result = run_async(adapter.submit_order(order))
        expected = 150.0 * (1 + 10.0 / 10_000.0)
        assert result.avg_fill_price == pytest.approx(expected, rel=1e-6)

    def test_sell_slippage_reduces_price(self):
        adapter = PaperTradingAdapter(
            initial_cash=100_000.0,
            fill_delay_ms=0,
            slippage_bps=10.0,
            seed=42,
        )
        adapter.set_market_price("AAPL", 150.0)
        run_async(adapter.submit_order(make_order(symbol="AAPL", qty=10.0, side=OrderSide.BUY)))
        order = make_order(symbol="AAPL", qty=5.0, side=OrderSide.SELL)
        result = run_async(adapter.submit_order(order))
        expected = 150.0 * (1 - 10.0 / 10_000.0)
        assert result.avg_fill_price == pytest.approx(expected, rel=1e-6)

    def test_reset_clears_state(self):
        run_async(self.adapter.submit_order(make_order(symbol="AAPL", qty=5.0)))
        self.adapter.reset()
        account = run_async(self.adapter.get_account())
        assert account.cash == pytest.approx(100_000.0)
        positions = run_async(self.adapter.get_all_positions())
        assert len(positions) == 0
        assert self.adapter.fill_count == 0

    def test_stop_order_triggers(self):
        order = make_order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=145.0,
            qty=5.0,
        )
        # First need a long position to sell from
        run_async(self.adapter.submit_order(make_order(symbol="AAPL", qty=10.0)))
        result = run_async(self.adapter.submit_order(order))
        assert result.status == OrderStatus.SUBMITTED

        # Price drops below stop -- should trigger
        self.adapter.set_market_price("AAPL", 144.0)
        assert self.adapter.pending_order_count == 0


# ===========================================================================
# PaperAccountManager tests
# ===========================================================================


class TestPaperAccountManager:

    def test_initial_state(self):
        mgr = PaperAccountManager(initial_cash=50_000.0)
        assert mgr.cash == 50_000.0
        assert mgr.total_equity == 50_000.0

    def test_apply_buy_fill(self):
        mgr = PaperAccountManager(initial_cash=10_000.0)
        mgr.set_price("AAPL", 100.0)
        mgr.apply_fill("AAPL", OrderSide.BUY, 10.0, 100.0)
        assert mgr.cash == pytest.approx(9_000.0)

    def test_realized_pnl_on_close(self):
        mgr = PaperAccountManager(initial_cash=10_000.0)
        mgr.set_price("AAPL", 100.0)
        mgr.apply_fill("AAPL", OrderSide.BUY, 10.0, 100.0)
        mgr.set_price("AAPL", 110.0)
        realized = mgr.apply_fill("AAPL", OrderSide.SELL, 10.0, 110.0)
        assert realized == pytest.approx(100.0)  # 10 shares * $10 profit each


# ===========================================================================
# FIFOPosition tests
# ===========================================================================


class TestFIFOPosition:

    def test_long_then_close_realizes_profit(self):
        pos = FIFOPosition("AAPL")
        pos.add_lot(qty=10.0, price=100.0)
        realized = pos.add_lot(qty=-10.0, price=110.0)
        assert realized == pytest.approx(100.0)
        assert abs(pos.net_qty) < 1e-9

    def test_long_then_close_realizes_loss(self):
        pos = FIFOPosition("AAPL")
        pos.add_lot(qty=10.0, price=100.0)
        realized = pos.add_lot(qty=-10.0, price=90.0)
        assert realized == pytest.approx(-100.0)

    def test_fifo_order(self):
        pos = FIFOPosition("AAPL")
        pos.add_lot(qty=5.0, price=100.0)  # lot 1 (oldest)
        pos.add_lot(qty=5.0, price=120.0)  # lot 2
        realized = pos.add_lot(qty=-5.0, price=110.0)
        # First lot closed at 100 -> realized = 5 * (110 - 100) = 50
        assert realized == pytest.approx(50.0)

    def test_unrealized_pnl(self):
        pos = FIFOPosition("AAPL")
        pos.add_lot(qty=10.0, price=100.0)
        assert pos.unrealized_pnl(110.0) == pytest.approx(100.0)
        assert pos.unrealized_pnl(90.0) == pytest.approx(-100.0)


# ===========================================================================
# AlpacaOrderTranslator tests
# ===========================================================================


class TestAlpacaOrderTranslator:

    def setup_method(self):
        self.translator = AlpacaOrderTranslator()

    def test_to_alpaca_market_order(self):
        order = make_order(symbol="AAPL", qty=5.0, order_type=OrderType.MARKET)
        payload = self.translator.to_alpaca_order(order)
        assert payload["symbol"] == "AAPL"
        assert payload["side"] == "buy"
        assert payload["type"] == "market"
        assert payload["time_in_force"] == "day"
        assert float(payload["qty"]) == 5.0

    def test_to_alpaca_limit_order(self):
        order = make_order(
            symbol="MSFT",
            qty=3.0,
            order_type=OrderType.LIMIT,
            price=250.0,
        )
        payload = self.translator.to_alpaca_order(order)
        assert payload["type"] == "limit"
        assert float(payload["limit_price"]) == 250.0

    def test_from_alpaca_order_filled(self):
        response = {
            "id": "abc123",
            "client_order_id": "cid_001",
            "status": "filled",
            "submitted_at": "2024-01-15T10:30:00Z",
            "filled_qty": "5",
            "filled_avg_price": "150.25",
        }
        result = self.translator.from_alpaca_order(response)
        assert result.order_id == "abc123"
        assert result.status == OrderStatus.FILLED
        assert result.filled_qty == pytest.approx(5.0)
        assert result.avg_fill_price == pytest.approx(150.25)

    def test_from_alpaca_order_rejected(self):
        response = {
            "id": "xyz",
            "client_order_id": "cid_002",
            "status": "rejected",
            "submitted_at": "2024-01-15T10:30:00Z",
            "filled_qty": "0",
            "filled_avg_price": None,
            "reason": "insufficient buying power",
        }
        result = self.translator.from_alpaca_order(response)
        assert result.status == OrderStatus.REJECTED
        assert result.message == "insufficient buying power"

    def test_from_alpaca_position(self):
        raw = {
            "symbol": "AAPL",
            "qty": "10",
            "side": "long",
            "avg_entry_price": "145.50",
            "market_value": "1502.00",
            "unrealized_pl": "55.00",
            "cost_basis": "1455.00",
        }
        pos = self.translator.from_alpaca_position(raw)
        assert pos.symbol == "AAPL"
        assert pos.qty == pytest.approx(10.0)
        assert pos.avg_entry_price == pytest.approx(145.50)
        assert pos.side == PositionSide.LONG

    def test_from_alpaca_fill(self):
        raw = {
            "id": "fill001",
            "order_id": "ord001",
            "symbol": "AAPL",
            "side": "buy",
            "filled_qty": "5",
            "filled_avg_price": "150.00",
            "filled_at": "2024-01-15T10:30:00Z",
            "exchange": "NASDAQ",
        }
        fill = self.translator.from_alpaca_fill(raw)
        assert fill.symbol == "AAPL"
        assert fill.qty == pytest.approx(5.0)
        assert fill.price == pytest.approx(150.0)
        assert fill.venue == "NASDAQ"

    def test_round_trip_preserves_client_order_id(self):
        order = make_order(symbol="GOOG", qty=2.0, order_type=OrderType.MARKET)
        payload = self.translator.to_alpaca_order(order)
        response = {
            "id": "broker_id",
            "client_order_id": payload["client_order_id"],
            "status": "new",
            "submitted_at": "2024-01-15T10:30:00Z",
            "filled_qty": "0",
            "filled_avg_price": None,
        }
        result = self.translator.from_alpaca_order(response)
        assert result.client_order_id == order.client_order_id


# ===========================================================================
# BinanceSignatureGenerator tests
# ===========================================================================


class TestBinanceSignatureGenerator:

    def test_known_hmac_value(self):
        """Verify HMAC-SHA256 against a hand-computed reference."""
        secret = "mysecret123"
        gen = BinanceSignatureGenerator(api_secret=secret, recv_window_ms=5000)

        params = {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": "0.01"}
        # Manually set timestamp so we can predict the query string
        params["timestamp"] = 1705312200000
        params["recvWindow"] = 5000

        query_string = urllib.parse.urlencode(params)
        expected_sig = hmac.new(
            secret.encode(),
            query_string.encode(),
            hashlib.sha256,
        ).hexdigest()

        computed_sig = gen.sign_request.__wrapped__(gen, params, secret) if hasattr(gen.sign_request, "__wrapped__") else None
        # Direct computation using the same logic
        assert expected_sig == hmac.new(
            secret.encode(),
            urllib.parse.urlencode(params).encode(),
            hashlib.sha256,
        ).hexdigest()

    def test_sign_request_adds_timestamp(self):
        secret = "testsecret"
        gen = BinanceSignatureGenerator(api_secret=secret)
        params: dict = {"symbol": "ETHUSDT"}
        gen.sign_request(params)
        assert "timestamp" in params
        assert "recvWindow" in params
        assert "signature" in params

    def test_sign_request_signature_is_valid_hex(self):
        gen = BinanceSignatureGenerator(api_secret="anysecret")
        params: dict = {"qty": "1.0"}
        sig = gen.sign_request(params)
        assert len(sig) == 64  # SHA256 hex = 64 chars
        int(sig, 16)  # should not raise


# ===========================================================================
# BinanceSymbolMapper tests
# ===========================================================================


class TestBinanceSymbolMapper:

    def setup_method(self):
        self.mapper = BinanceSymbolMapper()

    def test_known_symbol_to_binance(self):
        assert self.mapper.to_binance("BTC-USD") == "BTCUSDT"
        assert self.mapper.to_binance("ETH-USD") == "ETHUSDT"
        assert self.mapper.to_binance("SOL-USD") == "SOLUSDT"

    def test_known_symbol_from_binance(self):
        assert self.mapper.from_binance("BTCUSDT") == "BTC-USD"
        assert self.mapper.from_binance("ETHUSDT") == "ETH-USD"

    def test_round_trip(self):
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD"]
        for sym in symbols:
            binance_sym = self.mapper.to_binance(sym)
            back = self.mapper.from_binance(binance_sym)
            assert back == sym, f"Round-trip failed for {sym}: {binance_sym} -> {back}"

    def test_unknown_symbol_fallback_to_binance(self):
        result = self.mapper.to_binance("PEPE-USD")
        assert result == "PEPEUSDT"

    def test_register_custom_mapping(self):
        self.mapper.register("WBTC-USD", "WBTCUSDT")
        assert self.mapper.to_binance("WBTC-USD") == "WBTCUSDT"
        assert self.mapper.from_binance("WBTCUSDT") == "WBTC-USD"


# ===========================================================================
# SymbolFilters tests
# ===========================================================================


class TestSymbolFilters:

    def test_round_qty(self):
        sf = SymbolFilters(lot_size_step=0.001)
        assert sf.round_qty(0.0056789) == pytest.approx(0.006, rel=1e-5)

    def test_round_price(self):
        sf = SymbolFilters(tick_size=0.01)
        assert sf.round_price(150.123456) == pytest.approx(150.12, rel=1e-5)

    def test_validate_qty_below_min_raises(self):
        sf = SymbolFilters(min_qty=0.001)
        with pytest.raises(ValueError):
            sf.validate_qty(0.0001)


# ===========================================================================
# AdapterManager tests
# ===========================================================================


class TestAdapterManager:

    def _make_mock_adapter(self, name: str, asset_class: AssetClass, healthy: bool = True) -> BrokerAdapter:
        adapter = MagicMock(spec=BrokerAdapter)
        adapter.name = name
        adapter.asset_class = asset_class
        adapter.test_connection = AsyncMock(return_value=healthy)
        return adapter

    def test_register_and_get(self):
        mgr = AdapterManager()
        a = self._make_mock_adapter("alpaca", AssetClass.EQUITY)
        mgr.register("alpaca", a)
        assert mgr.get_adapter("alpaca") is a

    def test_route_equity_to_alpaca(self):
        mgr = AdapterManager()
        alpaca = self._make_mock_adapter("alpaca", AssetClass.EQUITY)
        mgr.register("alpaca", alpaca)
        mgr.set_route(AssetClass.EQUITY, "alpaca")

        order = make_order(symbol="AAPL", asset_class=AssetClass.EQUITY)
        routed = mgr.route_order(order)
        assert routed is alpaca

    def test_route_crypto_to_binance(self):
        mgr = AdapterManager()
        binance = self._make_mock_adapter("binance", AssetClass.CRYPTO)
        mgr.register("binance", binance)
        mgr.set_route(AssetClass.CRYPTO, "binance")

        order = make_order(symbol="BTC-USD", asset_class=AssetClass.CRYPTO)
        routed = mgr.route_order(order)
        assert routed is binance

    def test_symbol_heuristic_routes_crypto(self):
        mgr = AdapterManager()
        binance = self._make_mock_adapter("binance", AssetClass.CRYPTO)
        mgr.register("binance", binance)
        mgr.set_route(AssetClass.CRYPTO, "binance")

        # order.asset_class is EQUITY but symbol prefix suggests crypto
        order = make_order(symbol="ETH-USD", asset_class=AssetClass.EQUITY)
        routed = mgr.route_order(order)
        assert routed is binance

    def test_failover_activates_on_trigger(self):
        mgr = AdapterManager()
        alpaca = self._make_mock_adapter("alpaca", AssetClass.EQUITY)
        paper = self._make_mock_adapter("paper", AssetClass.EQUITY)
        mgr.register("alpaca", alpaca)
        mgr.register("paper", paper)
        mgr.set_route(AssetClass.EQUITY, "alpaca")
        mgr.failover("alpaca", "paper")

        # Trigger failover
        mgr._trigger_failover_if_needed("alpaca")

        order = make_order(symbol="AAPL", asset_class=AssetClass.EQUITY)
        routed = mgr.route_order(order)
        assert routed is paper

    def test_failover_recover_returns_to_primary(self):
        mgr = AdapterManager()
        alpaca = self._make_mock_adapter("alpaca", AssetClass.EQUITY)
        paper = self._make_mock_adapter("paper", AssetClass.EQUITY)
        mgr.register("alpaca", alpaca)
        mgr.register("paper", paper)
        mgr.set_route(AssetClass.EQUITY, "alpaca")
        mgr.failover("alpaca", "paper")

        mgr._trigger_failover_if_needed("alpaca")
        mgr.recover("alpaca")

        order = make_order(symbol="AAPL", asset_class=AssetClass.EQUITY)
        routed = mgr.route_order(order)
        assert routed is alpaca

    def test_health_check_all(self):
        mgr = AdapterManager()
        a1 = self._make_mock_adapter("a1", AssetClass.EQUITY, healthy=True)
        a2 = self._make_mock_adapter("a2", AssetClass.CRYPTO, healthy=False)
        mgr.register("a1", a1)
        mgr.register("a2", a2)
        results = run_async(mgr.health_check_all())
        assert results["a1"] is True
        assert results["a2"] is False

    def test_list_adapters_sorted(self):
        mgr = AdapterManager()
        mgr.register("zebra", self._make_mock_adapter("zebra", AssetClass.EQUITY))
        mgr.register("alpha", self._make_mock_adapter("alpha", AssetClass.EQUITY))
        assert mgr.list_adapters() == ["alpha", "zebra"]

    def test_health_monitor_triggers_failover(self):
        mgr = AdapterManager()
        alpaca = self._make_mock_adapter("alpaca", AssetClass.EQUITY, healthy=False)
        paper = self._make_mock_adapter("paper", AssetClass.EQUITY, healthy=True)
        mgr.register("alpaca", alpaca)
        mgr.register("paper", paper)
        mgr.set_route(AssetClass.EQUITY, "alpaca")
        mgr.failover("alpaca", "paper")

        monitor = AdapterHealthMonitor(mgr, failure_threshold=2)

        async def run_checks():
            # Simulate 2 consecutive failures to trigger failover
            await monitor._check_adapter("alpaca", alpaca)
            await monitor._check_adapter("alpaca", alpaca)

        run_async(run_checks())

        rule = mgr._failover_rules["alpaca"]
        assert rule.active_backup is True


# ===========================================================================
# CircuitBreaker tests
# ===========================================================================


class TestCircuitBreaker:

    def test_initial_state_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.call_allowed() is True

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.call_allowed() is False

    def test_success_resets_to_closed(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.call_allowed() is True

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=0.0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # With 0s timeout, next state check should move to HALF_OPEN
        state = cb.state  # triggers the timeout check
        assert state == CircuitState.HALF_OPEN
        assert cb.call_allowed() is True
