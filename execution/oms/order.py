"""
execution/oms/order.py
======================
Core Order dataclass and OrderBook for the SRFM execution layer.

The Order dataclass is the single source of truth for every order's lifecycle.
OrderBook provides a thread-safe registry of all open and historical orders,
indexed for O(1) lookups by order_id and O(n) scans by symbol.

Design notes
------------
- RLock is used (not Lock) so that methods that call other methods on the same
  thread never deadlock.
- Orders are frozen after reaching a terminal state (FILLED / CANCELLED /
  REJECTED) — mutation attempts raise RuntimeError.
- fill_price and fill_qty are the *actual* executed values; price is the
  *requested* limit/stop price (None for MARKET orders).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Side(str, Enum):
    """Order direction."""
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Execution instruction type."""
    MARKET = "MARKET"
    LIMIT  = "LIMIT"
    STOP   = "STOP"


class OrderStatus(str, Enum):
    """Order lifecycle states (linear progression with possible bypass to terminal)."""
    PENDING   = "PENDING"    # created locally, not yet sent to broker
    SUBMITTED = "SUBMITTED"  # sent to broker, awaiting acknowledgment
    PARTIAL   = "PARTIAL"    # partially filled
    FILLED    = "FILLED"     # fully filled — terminal
    CANCELLED = "CANCELLED"  # cancelled — terminal
    REJECTED  = "REJECTED"   # broker or risk rejected — terminal


_TERMINAL_STATES: frozenset[OrderStatus] = frozenset(
    {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}
)


# ---------------------------------------------------------------------------
# Order dataclass
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """
    Represents a single order across its entire lifecycle.

    Parameters
    ----------
    symbol : str
        Instrument symbol, e.g. ``"BTC/USD"`` or ``"SPY"``.
    side : Side
        BUY or SELL.
    order_type : OrderType
        MARKET, LIMIT, or STOP.
    quantity : float
        Requested quantity in base units (shares or coins).
    price : float | None
        Limit/stop price; ``None`` for MARKET orders.
    strategy_id : str
        Tag identifying the originating strategy (e.g. ``"larsa_v16"``).
    parent_order_id : str | None
        Set on child TWAP slices that belong to a parent order.
    """

    symbol:          str
    side:            Side
    order_type:      OrderType
    quantity:        float
    price:           Optional[float]        = None
    strategy_id:     str                    = "default"
    parent_order_id: Optional[str]          = None

    # --- auto-populated fields ---
    order_id:        str                    = field(default_factory=lambda: str(uuid.uuid4()))
    status:          OrderStatus            = field(default=OrderStatus.PENDING)
    created_at:      datetime               = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_at:    Optional[datetime]     = field(default=None)
    filled_at:       Optional[datetime]     = field(default=None)
    fill_price:      Optional[float]        = field(default=None)
    fill_qty:        float                  = field(default=0.0)
    slippage_bps:    Optional[float]        = field(default=None)
    commission_usd:  float                  = field(default=0.0)
    broker_order_id: Optional[str]          = field(default=None)
    reject_reason:   Optional[str]          = field(default=None)
    # decision_price is the mid-price at signal generation time (used for TCA)
    decision_price:  Optional[float]        = field(default=None)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """Return True when the order can no longer change state."""
        return self.status in _TERMINAL_STATES

    @property
    def remaining_qty(self) -> float:
        """How much of the order is still unfilled."""
        return max(0.0, self.quantity - self.fill_qty)

    @property
    def notional_value(self) -> Optional[float]:
        """Approximate notional at fill price (or limit price if unfilled)."""
        ref = self.fill_price if self.fill_price is not None else self.price
        if ref is None:
            return None
        return abs(self.fill_qty or self.quantity) * ref

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _guard_mutable(self) -> None:
        if self.is_terminal:
            raise RuntimeError(
                f"Order {self.order_id} is in terminal state {self.status.value} "
                "and cannot be mutated."
            )

    def mark_submitted(self, broker_order_id: Optional[str] = None) -> None:
        """Transition PENDING -> SUBMITTED."""
        self._guard_mutable()
        self.status          = OrderStatus.SUBMITTED
        self.submitted_at    = datetime.now(timezone.utc)
        self.broker_order_id = broker_order_id

    def mark_partial(self, fill_qty: float, fill_price: float) -> None:
        """Record a partial fill."""
        self._guard_mutable()
        self.status     = OrderStatus.PARTIAL
        self.fill_qty   = fill_qty
        self.fill_price = fill_price

    def mark_filled(self, fill_qty: float, fill_price: float,
                    commission_usd: float = 0.0) -> None:
        """Transition any non-terminal state -> FILLED."""
        self._guard_mutable()
        ref_price = self.decision_price or self.price
        self.fill_qty       = fill_qty
        self.fill_price     = fill_price
        self.commission_usd = commission_usd
        self.filled_at      = datetime.now(timezone.utc)
        self.status         = OrderStatus.FILLED
        if ref_price and ref_price > 0:
            side_sign = 1.0 if self.side == Side.BUY else -1.0
            self.slippage_bps = (
                (fill_price - ref_price) / ref_price * 10_000 * side_sign
            )

    def mark_cancelled(self) -> None:
        """Transition -> CANCELLED."""
        self._guard_mutable()
        self.status = OrderStatus.CANCELLED

    def mark_rejected(self, reason: str) -> None:
        """Transition -> REJECTED with a human-readable reason."""
        self._guard_mutable()
        self.status        = OrderStatus.REJECTED
        self.reject_reason = reason

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary snapshot of this order."""
        return {
            "order_id":        self.order_id,
            "symbol":          self.symbol,
            "side":            self.side.value,
            "order_type":      self.order_type.value,
            "quantity":        self.quantity,
            "price":           self.price,
            "status":          self.status.value,
            "strategy_id":     self.strategy_id,
            "parent_order_id": self.parent_order_id,
            "broker_order_id": self.broker_order_id,
            "created_at":      self.created_at.isoformat(),
            "submitted_at":    self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at":       self.filled_at.isoformat() if self.filled_at else None,
            "fill_price":      self.fill_price,
            "fill_qty":        self.fill_qty,
            "slippage_bps":    self.slippage_bps,
            "commission_usd":  self.commission_usd,
            "decision_price":  self.decision_price,
            "reject_reason":   self.reject_reason,
        }

    def __repr__(self) -> str:
        return (
            f"<Order {self.order_id[:8]} {self.side.value} {self.quantity} "
            f"{self.symbol} @ {self.price or 'MKT'} [{self.status.value}]>"
        )


# ---------------------------------------------------------------------------
# OrderBook
# ---------------------------------------------------------------------------

class OrderBook:
    """
    Thread-safe registry of all orders regardless of state.

    Internal storage
    ----------------
    _orders : dict[order_id, Order]
        All orders ever registered.

    Indexing
    --------
    For performance, a secondary index maps symbol -> set[order_id] so
    symbol-scoped queries avoid a full scan.

    Thread safety
    -------------
    All public methods acquire ``_lock`` (RLock) before touching shared state.
    """

    def __init__(self) -> None:
        self._orders: dict[str, Order]         = {}
        self._by_symbol: dict[str, set[str]]   = {}
        self._lock: threading.RLock            = threading.RLock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, order: Order) -> None:
        """Register a new order. Raises ValueError if order_id already exists."""
        with self._lock:
            if order.order_id in self._orders:
                raise ValueError(f"Duplicate order_id: {order.order_id}")
            self._orders[order.order_id] = order
            self._by_symbol.setdefault(order.symbol, set()).add(order.order_id)

    def remove(self, order_id: str) -> Optional[Order]:
        """Remove and return an order, or None if not found."""
        with self._lock:
            order = self._orders.pop(order_id, None)
            if order:
                self._by_symbol.get(order.symbol, set()).discard(order_id)
            return order

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, order_id: str) -> Optional[Order]:
        """Return the Order for *order_id*, or None."""
        with self._lock:
            return self._orders.get(order_id)

    def get_by_broker_id(self, broker_order_id: str) -> Optional[Order]:
        """Linear scan to find order by broker-assigned ID."""
        with self._lock:
            for o in self._orders.values():
                if o.broker_order_id == broker_order_id:
                    return o
            return None

    def open_orders(self) -> list[Order]:
        """Return all orders that are NOT in a terminal state."""
        with self._lock:
            return [o for o in self._orders.values() if not o.is_terminal]

    def open_orders_for(self, symbol: str) -> list[Order]:
        """Return open orders for a specific symbol."""
        with self._lock:
            ids = self._by_symbol.get(symbol, set())
            return [
                self._orders[oid]
                for oid in ids
                if oid in self._orders and not self._orders[oid].is_terminal
            ]

    def filled_orders(self) -> list[Order]:
        """Return all FILLED orders (historical record)."""
        with self._lock:
            return [o for o in self._orders.values() if o.status == OrderStatus.FILLED]

    def all_orders(self) -> list[Order]:
        """Return a snapshot list of every order ever registered."""
        with self._lock:
            return list(self._orders.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._orders)

    def __iter__(self) -> Iterator[Order]:
        with self._lock:
            snapshot = list(self._orders.values())
        return iter(snapshot)

    def __repr__(self) -> str:
        with self._lock:
            n_open   = sum(1 for o in self._orders.values() if not o.is_terminal)
            n_filled = sum(1 for o in self._orders.values() if o.status == OrderStatus.FILLED)
        return f"<OrderBook open={n_open} filled={n_filled} total={len(self._orders)}>"
