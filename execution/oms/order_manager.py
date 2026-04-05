"""
execution/oms/order_manager.py
==============================
Central order lifecycle management for the SRFM execution layer.

OrderManager is the single entry-point that strategy code calls when it wants
to enter or exit a position.  It:

1. Converts a target portfolio fraction into a concrete share/unit quantity
   (``size_order``).
2. Runs pre-trade risk checks via ``RiskGuard``.
3. Routes the order to the broker via the injected ``SmartRouter``.
4. Tracks fills and updates ``PositionTracker``.
5. Emits events to an optional ``event_bus`` callable for downstream
   subscribers (TCA, audit log, monitoring).

Thread safety
-------------
All public methods are guarded by an ``RLock`` so they can be called from
both the main async loop and the monitoring background thread.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from .order import Order, OrderBook, OrderStatus, OrderType, Side
from .position_tracker import PositionTracker
from .risk_guard import RiskGuard

log = logging.getLogger("execution.order_manager")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MIN_ORDER_SIZE:  float = 1e-6    # minimum tradeable unit (fractional crypto)
MAX_ORDER_SIZE:  float = 1e6     # hard cap on any single order (safety)
LOT_SIZE:        float = 1e-8    # smallest allowed increment (Alpaca crypto)
MIN_NOTIONAL:    float = 1.0     # orders below $1 notional are skipped


# ---------------------------------------------------------------------------
# Fill event
# ---------------------------------------------------------------------------

class FillEvent:
    """
    Lightweight value-object emitted by the broker adapter when a fill arrives.

    Attributes
    ----------
    broker_order_id : str
        The broker's own order identifier.
    fill_qty : float
        Quantity filled in this event (may be partial).
    fill_price : float
        Execution price.
    commission_usd : float
        Commission charged for this fill (default 0 for paper trading).
    """

    __slots__ = ("broker_order_id", "fill_qty", "fill_price", "commission_usd")

    def __init__(
        self,
        broker_order_id: str,
        fill_qty: float,
        fill_price: float,
        commission_usd: float = 0.0,
    ) -> None:
        self.broker_order_id = broker_order_id
        self.fill_qty        = fill_qty
        self.fill_price      = fill_price
        self.commission_usd  = commission_usd


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class OrderManager:
    """
    Central order lifecycle manager.

    Parameters
    ----------
    router : object
        A SmartRouter (or any object with a ``route(order) -> str`` method
        that returns the broker_order_id).
    risk : RiskGuard
        Pre-trade risk checker.
    position_tracker : PositionTracker | None
        If provided, fills are forwarded here for real-time P&L.
    audit : object | None
        An AuditLog with a ``log_order(order)`` method.
    event_bus : Callable[[str, dict], None] | None
        Optional callback invoked on every order state change with
        ``(event_type, payload)``.
    equity : float
        Starting equity used for sizing.  Update with ``set_equity()``.
    """

    def __init__(
        self,
        router,
        risk: RiskGuard,
        position_tracker: Optional[PositionTracker] = None,
        audit=None,
        event_bus: Optional[Callable[[str, dict], None]] = None,
        equity: float = 100_000.0,
    ) -> None:
        self._router           = router
        self._risk             = risk
        self._position_tracker = position_tracker or PositionTracker()
        self._audit            = audit
        self._event_bus        = event_bus
        self._equity           = equity
        self._book             = OrderBook()
        self._lock             = threading.RLock()

    # ------------------------------------------------------------------
    # Equity management
    # ------------------------------------------------------------------

    def set_equity(self, equity: float) -> None:
        """Update the equity used for position sizing."""
        with self._lock:
            self._equity = equity

    def get_equity(self) -> float:
        with self._lock:
            return self._equity

    # ------------------------------------------------------------------
    # Order sizing
    # ------------------------------------------------------------------

    def size_order(
        self,
        target_frac: float,
        equity: float,
        price: float,
    ) -> float:
        """
        Convert a portfolio target fraction into a tradeable quantity.

        Parameters
        ----------
        target_frac : float
            Desired exposure as a fraction of *equity* (e.g. 0.10 = 10 %).
        equity : float
            Current portfolio equity in USD.
        price : float
            Current market price of the instrument.

        Returns
        -------
        float
            Quantity in base units, rounded to ``LOT_SIZE``, clamped to
            [MIN_ORDER_SIZE, MAX_ORDER_SIZE].  Returns 0.0 if the computed
            notional is below MIN_NOTIONAL.
        """
        if price <= 0 or equity <= 0:
            return 0.0
        raw_qty = (target_frac * equity) / price
        # Round to lot size
        qty = round(raw_qty / LOT_SIZE) * LOT_SIZE
        qty = max(MIN_ORDER_SIZE, min(MAX_ORDER_SIZE, qty))
        if qty * price < MIN_NOTIONAL:
            return 0.0
        return qty

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: Side,
        target_frac: float,
        equity: Optional[float] = None,
        curr_price: float = 0.0,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        strategy_id: str = "default",
        decision_price: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Create, validate, size, and submit an order.

        Parameters
        ----------
        symbol : str
            Instrument symbol.
        side : Side
            BUY or SELL.
        target_frac : float
            Desired allocation fraction (e.g. 0.10).
        equity : float | None
            Portfolio equity; uses internal value if None.
        curr_price : float
            Current mid-price of the instrument.
        order_type : OrderType
            MARKET (default), LIMIT, or STOP.
        limit_price : float | None
            Required when order_type is LIMIT or STOP.
        strategy_id : str
            Tag for attribution.
        decision_price : float | None
            Mid-price at signal time (stored for TCA).

        Returns
        -------
        Order | None
            The registered Order on success, or None if risk-rejected.
        """
        with self._lock:
            eq = equity if equity is not None else self._equity

            # ── 1. Size the order ────────────────────────────────────
            qty = self.size_order(target_frac, eq, curr_price)
            if qty == 0.0:
                log.warning(
                    "size_order returned 0 for %s %.4f frac @ %.4f price — skipping",
                    symbol, target_frac, curr_price,
                )
                return None

            # ── 2. Build order object ────────────────────────────────
            order = Order(
                symbol         = symbol,
                side           = side,
                order_type     = order_type,
                quantity       = qty,
                price          = limit_price,
                strategy_id    = strategy_id,
                decision_price = decision_price or curr_price,
            )

            # ── 3. Pre-trade risk checks ─────────────────────────────
            passed, reason = self._risk.run_all_checks(
                symbol      = symbol,
                new_frac    = target_frac,
                price       = curr_price,
                order       = order,
                equity      = eq,
                positions   = self._position_tracker.positions,
            )
            if not passed:
                log.warning("Risk rejected order for %s: %s", symbol, reason)
                order.mark_rejected(reason)
                self._book.add(order)
                self._emit("ORDER_REJECTED", order.to_dict())
                if self._audit:
                    self._audit.log_order(order)
                return None

            # ── 4. Register in book ──────────────────────────────────
            self._book.add(order)
            self._emit("ORDER_CREATED", order.to_dict())

            # ── 5. Route to broker ───────────────────────────────────
            try:
                broker_id = self._router.route(order)
                order.mark_submitted(broker_id)
                self._emit("ORDER_SUBMITTED", order.to_dict())
            except Exception as exc:
                log.error("Router error for %s: %s", symbol, exc)
                order.mark_rejected(str(exc))
                self._emit("ORDER_REJECTED", order.to_dict())

            if self._audit:
                self._audit.log_order(order)
            return order

    def cancel_order(self, order_id: str) -> bool:
        """
        Request cancellation of an open order.

        Returns True if the cancellation was dispatched, False if the order
        was already terminal or not found.
        """
        with self._lock:
            order = self._book.get(order_id)
            if order is None:
                log.warning("cancel_order: order %s not found", order_id)
                return False
            if order.is_terminal:
                log.info("cancel_order: order %s already terminal (%s)", order_id, order.status)
                return False
            try:
                if order.broker_order_id:
                    self._router.cancel(order.broker_order_id)
                order.mark_cancelled()
                self._emit("ORDER_CANCELLED", order.to_dict())
                if self._audit:
                    self._audit.log_order(order)
                return True
            except Exception as exc:
                log.error("cancel_order failed for %s: %s", order_id, exc)
                return False

    def handle_fill(self, fill_event: FillEvent) -> Optional[Order]:
        """
        Process an inbound fill from the broker.

        Updates the order status and position tracker, then emits events.

        Parameters
        ----------
        fill_event : FillEvent
            Broker fill notification.

        Returns
        -------
        Order | None
            The updated Order, or None if the broker_order_id is unknown.
        """
        with self._lock:
            order = self._book.get_by_broker_id(fill_event.broker_order_id)
            if order is None:
                log.warning(
                    "handle_fill: unknown broker_order_id %s",
                    fill_event.broker_order_id,
                )
                return None

            is_full = fill_event.fill_qty >= order.remaining_qty - 1e-9
            if is_full:
                order.mark_filled(
                    fill_qty       = order.quantity,
                    fill_price     = fill_event.fill_price,
                    commission_usd = fill_event.commission_usd,
                )
            else:
                order.mark_partial(
                    fill_qty   = order.fill_qty + fill_event.fill_qty,
                    fill_price = fill_event.fill_price,
                )

            # Update position tracker
            self._position_tracker.record_fill(order)

            event_type = "ORDER_FILLED" if is_full else "ORDER_PARTIAL"
            self._emit(event_type, order.to_dict())
            if self._audit:
                self._audit.log_order(order)

            log.info(
                "Fill: %s %s %s qty=%.6f @ %.4f slip=%.1f bps",
                order.side.value, order.symbol,
                "FULL" if is_full else "PARTIAL",
                fill_event.fill_qty,
                fill_event.fill_price,
                order.slippage_bps or 0.0,
            )
            return order

    # ------------------------------------------------------------------
    # Position / reconciliation helpers
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> float:
        """Return current position fraction for *symbol*."""
        pos = self._position_tracker.positions.get(symbol)
        if pos is None:
            return 0.0
        equity = self._equity or 1.0
        return (pos.quantity * (pos.avg_entry_price or 0.0)) / equity

    def reconcile_with_broker(self, broker_positions: dict[str, float]) -> list[str]:
        """
        Detect and correct drift between OMS and broker.

        Parameters
        ----------
        broker_positions : dict[str, float]
            Symbol -> quantity as reported by the broker.

        Returns
        -------
        list[str]
            Human-readable descriptions of any discrepancies found.
        """
        with self._lock:
            discrepancies: list[str] = []
            oms_positions = {
                sym: pos.quantity
                for sym, pos in self._position_tracker.positions.items()
            }
            all_symbols = set(broker_positions) | set(oms_positions)

            for sym in all_symbols:
                broker_qty = broker_positions.get(sym, 0.0)
                oms_qty    = oms_positions.get(sym, 0.0)
                delta      = abs(broker_qty - oms_qty)

                if delta < 1e-9:
                    continue  # no drift

                rel_delta = delta / max(abs(broker_qty), 1e-9)
                msg = (
                    f"RECONCILE {sym}: oms={oms_qty:.6f} broker={broker_qty:.6f} "
                    f"delta={delta:.6f} ({rel_delta:.2%})"
                )
                discrepancies.append(msg)
                log.warning(msg)

                # Auto-correct: overwrite OMS position to match broker
                self._position_tracker.set_quantity(sym, broker_qty)

            return discrepancies

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, payload: dict) -> None:
        if self._event_bus:
            try:
                self._event_bus(event_type, payload)
            except Exception as exc:
                log.error("event_bus error (%s): %s", event_type, exc)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def book(self) -> OrderBook:
        return self._book

    @property
    def position_tracker(self) -> PositionTracker:
        return self._position_tracker
