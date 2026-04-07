"""
paper_adapter.py -- Paper trading adapter for SRFM backtesting and live sim.

Simulates a broker with configurable fill model. Supports:
- Market orders: fill immediately at current_price +/- slippage
- Limit orders: fill when market price crosses the limit level
- Configurable partial fill probability
- FIFO cost basis accounting
- Daily P&L reset at midnight UTC

Usage:
    adapter = PaperTradingAdapter(initial_cash=100_000.0, slippage_bps=3.0)
    adapter.set_market_price("AAPL", 150.00)
    result = await adapter.submit_order(order)
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List, Optional, Tuple

from .base_adapter import (
    AccountInfo,
    AssetClass,
    BrokerAdapter,
    BrokerAdapterError,
    Fill,
    InsufficientFundsError,
    OrderRejectedError,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    TimeInForce,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FIFO cost basis lot tracker
# ---------------------------------------------------------------------------


@dataclass
class CostLot:
    """A single FIFO cost lot for position accounting.

    Fields
    ------
    qty          -- quantity in this lot
    entry_price  -- price at which this lot was acquired
    acquired_at  -- UTC timestamp when lot was opened
    """

    qty: float
    entry_price: float
    acquired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FIFOPosition:
    """Tracks a single symbol's position using FIFO cost basis.

    Long and short lots are tracked separately. When a trade reduces
    a position, the oldest lots are consumed first (FIFO).

    Parameters
    ----------
    symbol -- SRFM symbol
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._long_lots: Deque[CostLot] = deque()
        self._short_lots: Deque[CostLot] = deque()
        self.realized_pnl: float = 0.0

    @property
    def net_qty(self) -> float:
        long_qty = sum(lot.qty for lot in self._long_lots)
        short_qty = sum(lot.qty for lot in self._short_lots)
        return long_qty - short_qty

    @property
    def avg_entry_price(self) -> float:
        """Weighted average entry price across all lots on the dominant side."""
        if self.net_qty > 0:
            lots = self._long_lots
        elif self.net_qty < 0:
            lots = self._short_lots
        else:
            return 0.0
        total_qty = sum(lot.qty for lot in lots)
        if total_qty == 0:
            return 0.0
        total_cost = sum(lot.qty * lot.entry_price for lot in lots)
        return total_cost / total_qty

    @property
    def cost_basis(self) -> float:
        return abs(self.net_qty) * self.avg_entry_price

    def add_lot(self, qty: float, price: float) -> float:
        """Add a new lot. Returns realized P&L from any position reduction.

        For a buy when short (or sell when long), consumes opposite-side lots
        via FIFO and realizes P&L on the closed portion.

        Parameters
        ----------
        qty   -- positive for buy, negative for sell
        price -- fill price

        Returns
        -------
        Realized P&L from this trade.
        """
        realized = 0.0

        if qty > 0:  # buying
            remaining = qty
            while remaining > 0 and self._short_lots:
                oldest = self._short_lots[0]
                close_qty = min(remaining, oldest.qty)
                realized += close_qty * (oldest.entry_price - price)
                oldest.qty -= close_qty
                remaining -= close_qty
                if oldest.qty <= 1e-9:
                    self._short_lots.popleft()
            if remaining > 1e-9:
                self._long_lots.append(CostLot(qty=remaining, entry_price=price))
        else:  # selling
            sell_qty = abs(qty)
            remaining = sell_qty
            while remaining > 0 and self._long_lots:
                oldest = self._long_lots[0]
                close_qty = min(remaining, oldest.qty)
                realized += close_qty * (price - oldest.entry_price)
                oldest.qty -= close_qty
                remaining -= close_qty
                if oldest.qty <= 1e-9:
                    self._long_lots.popleft()
            if remaining > 1e-9:
                self._short_lots.append(CostLot(qty=remaining, entry_price=price))

        self.realized_pnl += realized
        return realized

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at the given market price."""
        if current_price <= 0:
            return 0.0
        long_pnl = sum(lot.qty * (current_price - lot.entry_price) for lot in self._long_lots)
        short_pnl = sum(lot.qty * (lot.entry_price - current_price) for lot in self._short_lots)
        return long_pnl + short_pnl

    def to_position(self, market_price: float, asset_class: AssetClass = AssetClass.EQUITY) -> Optional[Position]:
        """Convert to Position snapshot. Returns None if flat."""
        qty = self.net_qty
        if abs(qty) < 1e-9:
            return None
        side = PositionSide.LONG if qty > 0 else PositionSide.SHORT
        mkt_val = abs(qty) * market_price if market_price > 0 else 0.0
        return Position(
            symbol=self.symbol,
            qty=abs(qty),
            avg_entry_price=self.avg_entry_price,
            market_value=mkt_val,
            unrealized_pnl=self.unrealized_pnl(market_price),
            side=side,
            cost_basis=self.cost_basis,
            asset_class=asset_class,
        )


# ---------------------------------------------------------------------------
# Paper account manager
# ---------------------------------------------------------------------------


class PaperAccountManager:
    """Manages cash, positions, and P&L for the paper trading adapter.

    Tracks:
    - Cash balance (buying power for longs)
    - FIFO positions per symbol
    - Day P&L (resets at midnight UTC)
    - Total lifetime realized P&L

    Parameters
    ----------
    initial_cash -- starting cash balance
    """

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self._positions: Dict[str, FIFOPosition] = {}
        self._market_prices: Dict[str, float] = {}
        self._total_realized_pnl: float = 0.0
        self._day_start_equity: float = initial_cash
        self._day_start_date: Optional[str] = None

    def set_price(self, symbol: str, price: float) -> None:
        self._market_prices[symbol] = price
        self._maybe_reset_day()

    def get_price(self, symbol: str) -> Optional[float]:
        return self._market_prices.get(symbol)

    def _maybe_reset_day(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._day_start_date != today:
            self._day_start_equity = self.total_equity
            self._day_start_date = today

    @property
    def total_equity(self) -> float:
        """Cash + sum of all position market values."""
        equity = self.cash
        for sym, pos_tracker in self._positions.items():
            price = self._market_prices.get(sym, 0.0)
            equity += pos_tracker.unrealized_pnl(price)
            equity += pos_tracker.cost_basis
        return equity

    @property
    def day_pnl(self) -> float:
        return self.total_equity - self._day_start_equity

    def apply_fill(self, symbol: str, side: OrderSide, qty: float, price: float) -> float:
        """Apply a fill to cash and positions. Returns realized P&L.

        For buys: deducts qty * price from cash, adds lot to position.
        For sells: credits qty * price to cash, removes lot from position.

        Parameters
        ----------
        symbol -- SRFM symbol
        side   -- buy or sell
        qty    -- positive quantity filled
        price  -- fill execution price

        Returns
        -------
        Realized P&L from position reduction, if any.
        """
        if symbol not in self._positions:
            self._positions[symbol] = FIFOPosition(symbol)

        pos = self._positions[symbol]
        notional = qty * price

        if side == OrderSide.BUY:
            if self.cash < notional:
                raise InsufficientFundsError(
                    f"Insufficient cash: need {notional:.2f}, have {self.cash:.2f}"
                )
            self.cash -= notional
            realized = pos.add_lot(qty, price)
        else:
            self.cash += notional
            realized = pos.add_lot(-qty, price)

        self._total_realized_pnl += realized
        return realized

    def get_position(self, symbol: str) -> Optional[Position]:
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        price = self._market_prices.get(symbol, 0.0)
        return pos.to_position(price)

    def get_all_positions(self) -> Dict[str, Position]:
        result: Dict[str, Position] = {}
        for sym, pos in self._positions.items():
            price = self._market_prices.get(sym, 0.0)
            snapshot = pos.to_position(price)
            if snapshot is not None:
                result[sym] = snapshot
        return result

    def get_account(self) -> AccountInfo:
        equity = self.total_equity
        return AccountInfo(
            equity=equity,
            cash=self.cash,
            buying_power=self.cash,
            margin_used=max(0.0, equity - self.cash),
            leverage=equity / self.cash if self.cash > 0 else 1.0,
            day_pnl=self.day_pnl,
            total_pnl=self._total_realized_pnl,
            currency="USD",
        )

    def reset(self, initial_cash: Optional[float] = None) -> None:
        """Reset to initial state for testing."""
        self.cash = initial_cash or self.initial_cash
        self._positions.clear()
        self._market_prices.clear()
        self._total_realized_pnl = 0.0
        self._day_start_equity = self.cash
        self._day_start_date = None


# ---------------------------------------------------------------------------
# Pending limit order tracker
# ---------------------------------------------------------------------------


@dataclass
class PendingOrder:
    """An open limit or stop order waiting to be triggered.

    Fields
    ------
    order         -- original OrderRequest
    order_id      -- assigned paper order ID
    pending_since -- UTC timestamp when order was placed
    """

    order: OrderRequest
    order_id: str
    pending_since: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Paper trading adapter
# ---------------------------------------------------------------------------


class PaperTradingAdapter(BrokerAdapter):
    """Simulated broker adapter for paper trading and strategy testing.

    Fills market orders immediately at current_price +/- slippage.
    Fills limit orders when the market price crosses the limit level.
    Supports configurable partial fill probability.

    Parameters
    ----------
    initial_cash        -- starting cash (default 100,000 USD)
    fill_delay_ms       -- simulated fill latency in milliseconds
    slippage_bps        -- market order slippage in basis points (default 3 bps)
    partial_fill_prob   -- probability [0, 1] that a fill is partial (default 0.0)
    seed                -- random seed for reproducibility (None for random)
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        fill_delay_ms: int = 50,
        slippage_bps: float = 3.0,
        partial_fill_prob: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name="paper", asset_class=AssetClass.EQUITY)
        self.initial_cash = initial_cash
        self.fill_delay_ms = fill_delay_ms
        self.slippage_bps = slippage_bps
        self.partial_fill_prob = max(0.0, min(1.0, partial_fill_prob))
        self._rng = random.Random(seed)
        self._account = PaperAccountManager(initial_cash)
        self._pending_orders: Dict[str, PendingOrder] = {}
        self._order_statuses: Dict[str, OrderStatus] = {}
        self._fills: List[Fill] = []
        self._fill_callbacks: List[Callable[[Fill], None]] = []
        self._connected = True

    # ------------------------------------------------------------------
    # Price injection
    # ------------------------------------------------------------------

    def set_market_price(self, symbol: str, price: float) -> None:
        """Inject the current market price for a symbol.

        Also triggers pending limit order checks for the symbol.

        Parameters
        ----------
        symbol -- SRFM symbol
        price  -- current market price
        """
        if price <= 0:
            raise ValueError(f"Market price must be positive, got {price}")
        self._account.set_price(symbol, price)
        self._check_pending_orders(symbol, price)

    def _check_pending_orders(self, symbol: str, price: float) -> None:
        """Check if any pending limit/stop orders should fill at the new price."""
        to_fill: List[str] = []

        for order_id, pending in self._pending_orders.items():
            order = pending.order
            if order.symbol != symbol:
                continue

            should_fill = False
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= order.price:  # type: ignore[operator]
                    should_fill = True
                elif order.side == OrderSide.SELL and price >= order.price:  # type: ignore[operator]
                    should_fill = True
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and price >= order.stop_price:  # type: ignore[operator]
                    should_fill = True
                elif order.side == OrderSide.SELL and price <= order.stop_price:  # type: ignore[operator]
                    should_fill = True

            if should_fill:
                to_fill.append(order_id)

        for order_id in to_fill:
            pending = self._pending_orders.pop(order_id)
            fill_price = self._compute_fill_price(pending.order, price)
            self._execute_fill(pending.order, order_id, fill_price)

    # ------------------------------------------------------------------
    # Fill mechanics
    # ------------------------------------------------------------------

    def _compute_fill_price(self, order: OrderRequest, market_price: float) -> float:
        """Calculate execution price including slippage.

        Market orders: market_price * (1 +/- slippage)
        Limit orders: the limit price (or better)
        Stop orders: stop_price (triggered at market)

        Parameters
        ----------
        order        -- the order being filled
        market_price -- current market price

        Returns
        -------
        Execution price.
        """
        slip = self.slippage_bps / 10_000.0  # convert bps to decimal

        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                return market_price * (1.0 + slip)
            else:
                return market_price * (1.0 - slip)
        elif order.order_type == OrderType.LIMIT:
            # Limit fills at limit price or better
            return order.price  # type: ignore[return-value]
        elif order.order_type == OrderType.STOP:
            # Stop fills at market price when triggered
            if order.side == OrderSide.BUY:
                return market_price * (1.0 + slip)
            else:
                return market_price * (1.0 - slip)
        elif order.order_type == OrderType.STOP_LIMIT:
            return order.price  # type: ignore[return-value]
        else:
            return market_price

    def _compute_fill_qty(self, requested_qty: float) -> float:
        """Compute actual fill quantity accounting for partial fill probability.

        Parameters
        ----------
        requested_qty -- originally requested quantity

        Returns
        -------
        Actual fill quantity (may be less than requested if partial fill).
        """
        if self._rng.random() < self.partial_fill_prob:
            # Fill between 20% and 80% of requested
            fraction = self._rng.uniform(0.2, 0.8)
            return round(requested_qty * fraction, 6)
        return requested_qty

    def _execute_fill(
        self,
        order: OrderRequest,
        order_id: str,
        fill_price: float,
        qty_override: Optional[float] = None,
    ) -> Fill:
        """Record a fill, update positions, and notify callbacks.

        Parameters
        ----------
        order       -- original order
        order_id    -- paper order ID
        fill_price  -- execution price
        qty_override -- force a specific quantity (ignores partial fill logic)

        Returns
        -------
        Fill object.
        """
        fill_qty = qty_override if qty_override is not None else self._compute_fill_qty(order.qty)
        realized = self._account.apply_fill(order.symbol, order.side, fill_qty, fill_price)

        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            qty=fill_qty,
            price=fill_price,
            timestamp=datetime.now(timezone.utc),
            venue="PAPER",
            commission=0.0,
            liquidity="taker",
        )

        self._fills.append(fill)
        if fill_qty >= order.qty:
            self._order_statuses[order_id] = OrderStatus.FILLED
        else:
            self._order_statuses[order_id] = OrderStatus.PARTIALLY_FILLED

        self.logger.debug(
            "Paper fill: %s %s %s qty=%.4f price=%.4f realized_pnl=%.2f",
            order.side.value, order.symbol, order_id,
            fill_qty, fill_price, realized,
        )

        for cb in self._fill_callbacks:
            try:
                cb(fill)
            except Exception as exc:
                self.logger.warning("Fill callback error: %s", exc)

        return fill

    # ------------------------------------------------------------------
    # BrokerAdapter interface
    # ------------------------------------------------------------------

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        """Submit an order. Market orders fill after fill_delay_ms."""
        order_id = str(uuid.uuid4())
        self._order_statuses[order_id] = OrderStatus.PENDING

        submitted_at = datetime.now(timezone.utc)

        if order.order_type == OrderType.MARKET:
            market_price = self._account.get_price(order.symbol)
            if market_price is None:
                self._order_statuses[order_id] = OrderStatus.REJECTED
                raise OrderRejectedError(
                    f"No market price set for {order.symbol} -- call set_market_price() first",
                    order_id=order_id,
                )

            # Simulate fill delay
            if self.fill_delay_ms > 0:
                await asyncio.sleep(self.fill_delay_ms / 1000.0)

            fill_price = self._compute_fill_price(order, market_price)
            self._execute_fill(order, order_id, fill_price)
            status = self._order_statuses[order_id]
            avg_fill = fill_price

        elif order.order_type in (OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT):
            # Queue as pending -- will fill when price crosses
            self._pending_orders[order_id] = PendingOrder(order=order, order_id=order_id)
            self._order_statuses[order_id] = OrderStatus.SUBMITTED
            status = OrderStatus.SUBMITTED
            avg_fill = None

        else:
            self._order_statuses[order_id] = OrderStatus.REJECTED
            raise OrderRejectedError(
                f"Paper adapter does not support order_type={order.order_type}",
                order_id=order_id,
            )

        return OrderResult(
            order_id=order_id,
            client_order_id=order.client_order_id,
            status=status,
            submitted_at=submitted_at,
            message="",
            filled_qty=order.qty if status == OrderStatus.FILLED else 0.0,
            avg_fill_price=avg_fill if status == OrderStatus.FILLED else None,
        )

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
            self._order_statuses[order_id] = OrderStatus.CANCELED
            self.logger.info("Paper order %s canceled", order_id)
            return True
        current = self._order_statuses.get(order_id)
        if current in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            return False  # cannot cancel already-filled order
        return False

    async def get_position(self, symbol: str) -> Optional[Position]:
        return self._account.get_position(symbol)

    async def get_all_positions(self) -> Dict[str, Position]:
        return self._account.get_all_positions()

    async def get_account(self) -> AccountInfo:
        return self._account.get_account()

    async def get_order_status(self, order_id: str) -> OrderStatus:
        return self._order_statuses.get(order_id, OrderStatus.UNKNOWN)

    async def get_recent_fills(self, n: int = 100) -> List[Fill]:
        return list(reversed(self._fills[-n:]))

    def is_connected(self) -> bool:
        return self._connected

    async def test_connection(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Paper-specific extras
    # ------------------------------------------------------------------

    async def stream_fills(self, callback: Callable[[Fill], None]) -> None:
        """Register a fill callback. Paper adapter calls it synchronously on fill."""
        self._fill_callbacks.append(callback)
        # Keep running indefinitely (caller manages lifecycle)
        while True:
            await asyncio.sleep(1.0)

    def reset(self, initial_cash: Optional[float] = None) -> None:
        """Reset adapter to initial state for testing.

        Parameters
        ----------
        initial_cash -- override starting cash (uses constructor value if None)
        """
        self._account.reset(initial_cash or self.initial_cash)
        self._pending_orders.clear()
        self._order_statuses.clear()
        self._fills.clear()
        self._fill_callbacks.clear()
        self.logger.info(
            "PaperTradingAdapter reset, cash=%.2f",
            self._account.cash,
        )

    @property
    def pending_order_count(self) -> int:
        """Number of open pending limit/stop orders."""
        return len(self._pending_orders)

    @property
    def fill_count(self) -> int:
        """Total number of fills recorded."""
        return len(self._fills)
