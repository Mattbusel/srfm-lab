"""
execution.py -- Simulated execution handler for LARSA backtesting.

Fills orders at next-bar open with realistic slippage and spread costs.
Supports partial fills for large orders and per-symbol spread lookup.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .engine import (
    Direction,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    TransactionCostModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spreads Model
# ---------------------------------------------------------------------------

class SpreadsModel:
    """
    Per-symbol bid-ask spread lookup.

    Spreads are expressed as a fraction of mid price.
    Used to compute realistic fill prices around the theoretical mid.
    """

    # Default spreads (as fraction of price)
    DEFAULT_SPREADS: Dict[str, float] = {
        "BTC/USDT": 0.00005,    # 0.5 bps -- very liquid
        "ETH/USDT": 0.00008,
        "SOL/USDT": 0.0001,
        "BNB/USDT": 0.0001,
        "AVAX/USDT": 0.00015,
        "MATIC/USDT": 0.0002,
        "DOGE/USDT": 0.0003,
        # Equities
        "SPY": 0.00002,
        "QQQ": 0.00002,
        "AAPL": 0.00003,
        "NVDA": 0.00004,
        "TSLA": 0.00005,
    }

    def __init__(self, custom_spreads: Optional[Dict[str, float]] = None):
        self._spreads = dict(self.DEFAULT_SPREADS)
        if custom_spreads:
            self._spreads.update(custom_spreads)
        self._fallback_spread = 0.0002  # 2 bps default

    def get_spread(self, symbol: str) -> float:
        return self._spreads.get(symbol, self._fallback_spread)

    def get_half_spread(self, symbol: str) -> float:
        return self.get_spread(symbol) / 2.0

    def ask_price(self, symbol: str, mid: float) -> float:
        return mid * (1 + self.get_half_spread(symbol))

    def bid_price(self, symbol: str, mid: float) -> float:
        return mid * (1 - self.get_half_spread(symbol))

    def fill_price(self, symbol: str, mid: float, is_buy: bool) -> float:
        """Buyer pays ask, seller receives bid."""
        return self.ask_price(symbol, mid) if is_buy else self.bid_price(symbol, mid)

    def add_spread(self, symbol: str, spread_frac: float) -> None:
        self._spreads[symbol] = spread_frac


# ---------------------------------------------------------------------------
# Partial Fill Simulator
# ---------------------------------------------------------------------------

class PartialFillSimulator:
    """
    Splits large orders across multiple bars to simulate realistic
    execution of block-sized orders that exceed a fraction of ADV.

    If order size > max_participation * ADV per bar, the remainder
    is queued for the next bar(s).
    """

    def __init__(
        self,
        max_participation: float = 0.05,   # max 5% of ADV per bar
        min_fill_notional: float = 100.0,  # ignore fills below $100
    ):
        self.max_participation = max_participation
        self.min_fill_notional = min_fill_notional
        self._pending: List[Tuple[OrderEvent, float]] = []  # (order, remaining_qty)

    def submit(self, order: OrderEvent, adv: float) -> Tuple[float, float]:
        """
        Returns (fill_qty, remaining_qty).
        fill_qty may be < order.quantity if the order is too large.
        """
        max_qty = self.max_participation * adv
        abs_qty = abs(order.quantity)

        if abs_qty <= max_qty:
            return order.quantity, 0.0

        fill_qty = max_qty * np.sign(order.quantity)
        remaining = (abs_qty - max_qty) * np.sign(order.quantity)
        return fill_qty, remaining

    def queue_remainder(self, order: OrderEvent, remaining_qty: float) -> None:
        """Queue the unfilled portion for the next bar."""
        if abs(remaining_qty) > 1e-8:
            self._pending.append((order, remaining_qty))

    def get_pending_fills(self) -> List[Tuple[OrderEvent, float]]:
        """Retrieve and clear the pending fill queue."""
        pending = list(self._pending)
        self._pending.clear()
        return pending

    def has_pending(self) -> bool:
        return len(self._pending) > 0

    def clear(self) -> None:
        self._pending.clear()


# ---------------------------------------------------------------------------
# Simulated Execution Handler
# ---------------------------------------------------------------------------

class SimulatedExecutionHandler:
    """
    Simulates order execution at the open of the bar following the order.

    Execution logic:
      - MARKET orders: fill at next bar open + spread
      - LIMIT orders: fill only if next bar's range crosses the limit price
      - STOP orders: fill if price crosses the stop level
      - Slippage: square-root market impact applied on top of spread

    Generates FillEvents that are pushed back to the event queue.
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        spreads_model: Optional[SpreadsModel] = None,
        partial_fill_sim: Optional[PartialFillSimulator] = None,
        enable_partial_fills: bool = True,
    ):
        self.cost_model = cost_model or TransactionCostModel()
        self.spreads = spreads_model or SpreadsModel()
        self.partial_sim = partial_fill_sim or PartialFillSimulator()
        self.enable_partial_fills = enable_partial_fills

        # Pending orders keyed by symbol
        self._pending_orders: Dict[str, List[OrderEvent]] = defaultdict(list)
        # Most recent bar per symbol
        self._last_bars: Dict[str, MarketEvent] = {}
        # ADV estimates per symbol
        self._adv: Dict[str, float] = defaultdict(lambda: 1e6)
        self._daily_vol: Dict[str, float] = defaultdict(lambda: 0.02)
        # Recent volume for ADV estimation
        self._vol_window: Dict[str, deque] = defaultdict(lambda: deque(maxlen=26))

        # Fill callback (set by engine)
        self._fill_callbacks: List[Callable[[FillEvent], None]] = []

        # Statistics
        self.total_fills: int = 0
        self.total_commission: float = 0.0
        self.total_slippage: float = 0.0

    def register_fill_callback(self, fn: Callable[[FillEvent], None]) -> None:
        self._fill_callbacks.append(fn)

    def on_order_event(self, event: OrderEvent) -> None:
        """Accept an order for deferred execution."""
        if not isinstance(event, OrderEvent):
            return
        self._pending_orders[event.symbol].append(event)
        logger.debug(
            "Order queued: %s %s qty=%.4f",
            event.symbol,
            event.order_type.value,
            event.quantity,
        )

    def on_market_event(self, event: MarketEvent) -> List[FillEvent]:
        """
        Process pending orders using the current bar's OHLCV data.
        Returns FillEvents for all executed orders.
        """
        if not isinstance(event, MarketEvent):
            return []

        sym = event.symbol
        self._update_adv(sym, event)
        self._last_bars[sym] = event

        fills = []

        # Process pending orders for this symbol
        pending = self._pending_orders.pop(sym, [])

        # Also check partial fills from prior bars
        for order, remaining_qty in self.partial_sim.get_pending_fills():
            if order.symbol == sym:
                order_copy = OrderEvent(
                    event_type=EventType.ORDER,
                    timestamp=event.timestamp,
                    symbol=order.symbol,
                    order_type=order.order_type,
                    quantity=remaining_qty,
                    price=order.price,
                    direction=order.direction,
                    order_id=order.order_id,
                )
                pending.append(order_copy)

        for order in pending:
            fill = self._execute_order(order, event)
            if fill is not None:
                fills.append(fill)
                for cb in self._fill_callbacks:
                    cb(fill)

        return fills

    def _execute_order(self, order: OrderEvent, bar: MarketEvent) -> Optional[FillEvent]:
        """Compute fill for a single order against the current bar."""
        if order.order_type == OrderType.MARKET:
            return self._fill_market(order, bar)
        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit(order, bar)
        elif order.order_type == OrderType.STOP:
            return self._fill_stop(order, bar)
        elif order.order_type == OrderType.MOC:
            return self._fill_moc(order, bar)
        else:
            logger.warning("Unknown order type: %s", order.order_type)
            return None

    def _fill_market(self, order: OrderEvent, bar: MarketEvent) -> FillEvent:
        """
        Market order: fill at bar open with spread and slippage.
        Use bar open as execution price (next-bar execution model).
        """
        sym = order.symbol
        mid_price = bar.open
        is_buy = order.quantity > 0

        # Apply spread
        spread_price = self.spreads.fill_price(sym, mid_price, is_buy)

        # Partial fill logic
        fill_qty = order.quantity
        if self.enable_partial_fills:
            adv = self._adv[sym]
            fill_qty, remaining = self.partial_sim.submit(order, adv)
            if abs(remaining) > 1e-8:
                self.partial_sim.queue_remainder(order, remaining)

        if abs(fill_qty) < 1e-10:
            return None

        # Slippage (market impact)
        adv = self._adv[sym]
        daily_vol = self._daily_vol[sym]
        comm, slip = self.cost_model.total_cost(
            price=spread_price,
            quantity=fill_qty,
            adv=adv,
            daily_vol=daily_vol,
            is_maker=False,
            is_buy=is_buy,
        )

        # Effective fill price = spread_price + per-unit slippage
        if abs(fill_qty) > 1e-10:
            slip_per_unit = slip / abs(fill_qty)
        else:
            slip_per_unit = 0.0
        effective_price = spread_price + (slip_per_unit if is_buy else -slip_per_unit)
        effective_price = max(effective_price, 1e-8)

        self.total_fills += 1
        self.total_commission += comm
        self.total_slippage += slip

        return FillEvent(
            event_type=EventType.FILL,
            timestamp=bar.timestamp,
            symbol=sym,
            order_id=order.order_id,
            direction=order.direction,
            quantity=fill_qty,
            fill_price=effective_price,
            commission=comm,
            slippage=slip,
        )

    def _fill_limit(self, order: OrderEvent, bar: MarketEvent) -> Optional[FillEvent]:
        """
        Limit order: fill only if bar range crosses the limit price.
        Assumes fill at the limit price (conservative estimate).
        """
        sym = order.symbol
        limit_price = order.price
        is_buy = order.quantity > 0

        if is_buy and bar.low <= limit_price:
            fill_price = limit_price
        elif not is_buy and bar.high >= limit_price:
            fill_price = limit_price
        else:
            # Limit not hit -- re-queue for next bar
            self._pending_orders[sym].append(order)
            return None

        fill_qty = order.quantity
        if self.enable_partial_fills:
            fill_qty, remaining = self.partial_sim.submit(order, self._adv[sym])
            if abs(remaining) > 1e-8:
                self.partial_sim.queue_remainder(order, remaining)

        comm, slip = self.cost_model.total_cost(
            price=fill_price,
            quantity=fill_qty,
            adv=self._adv[sym],
            daily_vol=self._daily_vol[sym],
            is_maker=True,   # limit orders are makers
            is_buy=is_buy,
        )

        self.total_fills += 1
        self.total_commission += comm
        self.total_slippage += slip

        return FillEvent(
            event_type=EventType.FILL,
            timestamp=bar.timestamp,
            symbol=sym,
            order_id=order.order_id,
            direction=order.direction,
            quantity=fill_qty,
            fill_price=fill_price,
            commission=comm,
            slippage=slip,
        )

    def _fill_stop(self, order: OrderEvent, bar: MarketEvent) -> Optional[FillEvent]:
        """
        Stop order: triggered when bar crosses stop price.
        Fill at stop price + additional slippage (stop hunt premium).
        """
        sym = order.symbol
        stop_price = order.price
        is_buy = order.quantity > 0  # buy stops above market, sell stops below

        if is_buy and bar.high >= stop_price:
            fill_price = stop_price * 1.0005  # small stop-out premium
        elif not is_buy and bar.low <= stop_price:
            fill_price = stop_price * 0.9995
        else:
            # Stop not triggered -- re-queue
            self._pending_orders[sym].append(order)
            return None

        fill_qty = order.quantity
        comm, slip = self.cost_model.total_cost(
            price=fill_price,
            quantity=fill_qty,
            adv=self._adv[sym],
            daily_vol=self._daily_vol[sym],
            is_maker=False,
            is_buy=is_buy,
        )

        self.total_fills += 1
        self.total_commission += comm
        self.total_slippage += slip

        return FillEvent(
            event_type=EventType.FILL,
            timestamp=bar.timestamp,
            symbol=sym,
            order_id=order.order_id,
            direction=order.direction,
            quantity=fill_qty,
            fill_price=fill_price,
            commission=comm,
            slippage=slip,
        )

    def _fill_moc(self, order: OrderEvent, bar: MarketEvent) -> FillEvent:
        """Market-on-close: fill at bar close with spread."""
        sym = order.symbol
        mid_price = bar.close
        is_buy = order.quantity > 0
        fill_price = self.spreads.fill_price(sym, mid_price, is_buy)

        comm, slip = self.cost_model.total_cost(
            price=fill_price,
            quantity=order.quantity,
            adv=self._adv[sym],
            daily_vol=self._daily_vol[sym],
            is_maker=False,
            is_buy=is_buy,
        )

        self.total_fills += 1
        self.total_commission += comm
        self.total_slippage += slip

        return FillEvent(
            event_type=EventType.FILL,
            timestamp=bar.timestamp,
            symbol=sym,
            order_id=order.order_id,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=comm,
            slippage=slip,
        )

    def _update_adv(self, symbol: str, bar: MarketEvent) -> None:
        """Update ADV and daily vol estimates for a symbol."""
        self._vol_window[symbol].append(bar.volume)
        # Scale from 15m volume to daily (26 bars per day)
        if len(self._vol_window[symbol]) >= 26:
            self._adv[symbol] = float(np.mean(self._vol_window[symbol])) * 26

        # Estimate daily vol from bar returns
        if hasattr(self, "_prev_close") and symbol in self._prev_close:
            prev = self._prev_close[symbol]
            if prev > 0:
                ret = np.log(bar.close / prev)
                # Update EMA of squared returns (EWMA vol)
                prev_var = self._ewma_var.get(symbol, 0.02**2)
                alpha = 0.06  # ~ 32-bar EWMA
                new_var = (1 - alpha) * prev_var + alpha * ret**2
                self._ewma_var[symbol] = new_var
                # Scale to daily vol
                self._daily_vol[symbol] = float(np.sqrt(new_var * 26))

        if not hasattr(self, "_prev_close"):
            self._prev_close: Dict[str, float] = {}
            self._ewma_var: Dict[str, float] = {}
        self._prev_close[symbol] = bar.close

    def get_stats(self) -> Dict[str, float]:
        return {
            "total_fills": self.total_fills,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "avg_commission_per_fill": (
                self.total_commission / self.total_fills if self.total_fills > 0 else 0
            ),
            "avg_slippage_per_fill": (
                self.total_slippage / self.total_fills if self.total_fills > 0 else 0
            ),
        }

    def cancel_pending(self, symbol: Optional[str] = None) -> int:
        """Cancel pending orders. Returns count of cancelled orders."""
        if symbol:
            n = len(self._pending_orders.pop(symbol, []))
        else:
            n = sum(len(v) for v in self._pending_orders.values())
            self._pending_orders.clear()
        return n

    def reset(self) -> None:
        self._pending_orders.clear()
        self._last_bars.clear()
        self.partial_sim.clear()
        self.total_fills = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
