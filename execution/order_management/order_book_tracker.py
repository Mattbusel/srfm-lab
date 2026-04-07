"""
In-memory order book tracker with SQLite persistence and conflict detection.

OrderBookTracker  -- primary runtime index of all live orders
OrderStateStore   -- SQLite-backed persistence (crash recovery)
OrderConflictChecker -- pre-submission duplicate/concentration guard
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

from .order_types import (
    BaseOrder,
    Fill,
    IcebergOrder,
    LimitOrder,
    MarketOrder,
    OrderStatus,
    StopOrder,
    TWAPOrder,
    VWAPOrder,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OrderBookTracker
# ---------------------------------------------------------------------------

class OrderBookTracker:
    """
    In-memory index of all orders submitted during the session.

    Thread-safe via a single reentrant lock.  Methods that read or mutate
    the internal maps acquire the lock; callers should not hold their own
    locks when calling into this class.
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        # primary index: order_id -> order
        self._orders: Dict[str, BaseOrder] = {}
        # secondary index: symbol -> list of order_ids
        self._by_symbol: Dict[str, List[str]] = {}
        # all fills received this session
        self._fills: List[Fill] = []
        # rolling window for fill_rate calculation (capacity: last 100 orders)
        self._recent_order_ids: Deque[str] = deque(maxlen=100)

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def add_order(self, order: BaseOrder) -> None:
        """
        Register a new order in the tracker.

        Raises ValueError if an order with the same order_id already exists.
        """
        with self._lock:
            if order.order_id in self._orders:
                raise ValueError(
                    f"Order {order.order_id!r} already tracked"
                )
            self._orders[order.order_id] = order
            self._by_symbol.setdefault(order.symbol, []).append(order.order_id)
            self._recent_order_ids.append(order.order_id)
        logger.debug(
            "add_order order_id=%s symbol=%s side=%s qty=%s",
            order.order_id, order.symbol, order.side, order.qty,
        )

    def update_status(
        self,
        order_id: str,
        status: OrderStatus,
        fill: Optional[Fill] = None,
    ) -> None:
        """
        Update order status and optionally apply a fill.

        If fill is provided it is applied via order.apply_fill() which also
        updates filled_qty and avg_fill_price.
        """
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                raise KeyError(f"Order {order_id!r} not found in tracker")
            if fill is not None:
                order.apply_fill(fill)
                self._fills.append(fill)
                logger.debug(
                    "fill applied order_id=%s fill_id=%s qty=%s price=%s",
                    order_id, fill.fill_id, fill.qty, fill.price,
                )
            else:
                order.status = status
        logger.debug(
            "update_status order_id=%s status=%s", order_id, status.value
        )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_order(self, order_id: str) -> Optional[BaseOrder]:
        """Return order by ID or None if not found."""
        with self._lock:
            return self._orders.get(order_id)

    def pending_orders(self, symbol: Optional[str] = None) -> List[BaseOrder]:
        """
        Return all active (NEW / PENDING / PARTIAL) orders.

        If symbol is specified, filter to that symbol only.
        """
        with self._lock:
            result: List[BaseOrder] = []
            if symbol is not None:
                ids = self._by_symbol.get(symbol, [])
                candidates = [self._orders[oid] for oid in ids if oid in self._orders]
            else:
                candidates = list(self._orders.values())
            for o in candidates:
                if o.is_active:
                    result.append(o)
            return result

    def filled_orders(self, since: Optional[datetime] = None) -> List[BaseOrder]:
        """
        Return all FILLED orders.

        If since is given, filter to orders whose last fill timestamp >=
        since.  Orders with no fills are excluded.
        """
        with self._lock:
            result: List[BaseOrder] = []
            for o in self._orders.values():
                if o.status != OrderStatus.FILLED:
                    continue
                if since is not None:
                    if not o.fills:
                        continue
                    last_fill_ts = max(f.timestamp for f in o.fills)
                    if last_fill_ts < since:
                        continue
                result.append(o)
            return result

    def open_qty(self, symbol: str) -> Tuple[float, float]:
        """
        Return (open_buy_qty, open_sell_qty) for symbol across all active orders.

        Both values are non-negative.
        """
        with self._lock:
            buy_qty = 0.0
            sell_qty = 0.0
            for oid in self._by_symbol.get(symbol, []):
                o = self._orders.get(oid)
                if o is None or not o.is_active:
                    continue
                if o.side == "buy":
                    buy_qty += o.remaining_qty
                else:
                    sell_qty += o.remaining_qty
            return buy_qty, sell_qty

    def net_open_qty(self, symbol: str) -> float:
        """
        Return signed net open qty: positive = net buy, negative = net sell.
        """
        buy, sell = self.open_qty(symbol)
        return buy - sell

    def daily_fills(self) -> List[Fill]:
        """
        Return all fills whose timestamp falls on today (UTC).
        """
        today = datetime.now(timezone.utc).date()
        with self._lock:
            return [
                f for f in self._fills
                if f.timestamp.date() == today
            ]

    def fill_rate(self) -> float:
        """
        Fill rate = filled orders / total orders for the last 100 orders.

        Returns 0.0 if no orders have been submitted.
        """
        with self._lock:
            if not self._recent_order_ids:
                return 0.0
            filled = sum(
                1
                for oid in self._recent_order_ids
                if self._orders.get(oid, None) is not None
                and self._orders[oid].status == OrderStatus.FILLED
            )
            return filled / len(self._recent_order_ids)

    def all_orders(self) -> List[BaseOrder]:
        """Return a snapshot of all tracked orders."""
        with self._lock:
            return list(self._orders.values())

    def order_count(self) -> int:
        with self._lock:
            return len(self._orders)

    def clear(self) -> None:
        """Reset tracker -- for testing only."""
        with self._lock:
            self._orders.clear()
            self._by_symbol.clear()
            self._fills.clear()
            self._recent_order_ids.clear()


# ---------------------------------------------------------------------------
# OrderStateStore -- SQLite persistence
# ---------------------------------------------------------------------------

_CREATE_ORDERS_TABLE = """
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    qty             REAL NOT NULL,
    strategy_id     TEXT NOT NULL,
    signal_strength REAL NOT NULL,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL,
    order_type      TEXT NOT NULL,
    extra_json      TEXT NOT NULL DEFAULT '{}'
);
"""

_CREATE_FILLS_TABLE = """
CREATE TABLE IF NOT EXISTS fills (
    fill_id         TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    qty             REAL NOT NULL,
    price           REAL NOT NULL,
    timestamp       TEXT NOT NULL,
    venue           TEXT NOT NULL,
    commission_bps  REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
);
"""

_CREATE_ORDER_IDX = """
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status
    ON orders (symbol, status);
"""

_CREATE_FILL_IDX = """
CREATE INDEX IF NOT EXISTS idx_fills_order_id
    ON fills (order_id);
"""


def _dt_to_str(dt: datetime) -> str:
    return dt.isoformat()


def _str_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _order_to_extra_json(order: BaseOrder) -> str:
    """Serialize type-specific fields to JSON for the extra_json column."""
    extra: dict = {}
    if isinstance(order, LimitOrder):
        extra["limit_price"] = order.limit_price
        tif = order.time_in_force
        extra["time_in_force"] = tif.value if hasattr(tif, "value") else tif
    elif isinstance(order, StopOrder):
        extra["stop_price"] = order.stop_price
        extra["limit_price"] = order.limit_price
    elif isinstance(order, TWAPOrder):
        extra["start_time"] = _dt_to_str(order.start_time)
        extra["end_time"] = _dt_to_str(order.end_time)
        extra["n_slices"] = order.n_slices
        extra["slice_interval_s"] = order.slice_interval_s
    elif isinstance(order, VWAPOrder):
        extra["volume_curve"] = order.volume_curve
        extra["target_participation_rate"] = order.target_participation_rate
    elif isinstance(order, IcebergOrder):
        extra["total_qty"] = order.total_qty
        extra["display_qty"] = order.display_qty
    return json.dumps(extra)


def _row_to_order(row: sqlite3.Row, fills: List[Fill]) -> Optional[BaseOrder]:
    """Reconstruct an order from a DB row plus associated fills."""
    order_id = row["order_id"]
    symbol = row["symbol"]
    side = row["side"]
    qty = row["qty"]
    strategy_id = row["strategy_id"]
    signal_strength = row["signal_strength"]
    created_at = _str_to_dt(row["created_at"])
    status = OrderStatus(row["status"])
    order_type = row["order_type"]
    extra = json.loads(row["extra_json"] or "{}")

    base_kwargs = dict(
        order_id=order_id,
        symbol=symbol,
        side=side,
        qty=qty,
        strategy_id=strategy_id,
        signal_strength=signal_strength,
        created_at=created_at,
        status=status,
    )

    order: Optional[BaseOrder] = None
    if order_type == "MARKET":
        order = MarketOrder(**base_kwargs)
    elif order_type == "LIMIT":
        order = LimitOrder(
            limit_price=extra.get("limit_price", 0.0),
            time_in_force=extra.get("time_in_force", "DAY"),
            **base_kwargs,
        )
    elif order_type == "STOP":
        order = StopOrder(
            stop_price=extra.get("stop_price", 0.0),
            limit_price=extra.get("limit_price"),
            **base_kwargs,
        )
    elif order_type == "TWAP":
        order = TWAPOrder(
            start_time=_str_to_dt(extra.get("start_time", created_at.isoformat())),
            end_time=_str_to_dt(extra.get("end_time", created_at.isoformat())),
            n_slices=extra.get("n_slices", 1),
            **base_kwargs,
        )
    elif order_type == "VWAP":
        order = VWAPOrder(
            volume_curve=extra.get("volume_curve", [1.0]),
            target_participation_rate=extra.get("target_participation_rate", 0.10),
            **base_kwargs,
        )
    elif order_type == "ICEBERG":
        order = IcebergOrder(
            total_qty=extra.get("total_qty", qty),
            display_qty=extra.get("display_qty", qty * 0.10),
            **base_kwargs,
        )
    else:
        logger.warning("Unknown order_type %r for order_id %s -- skipping", order_type, order_id)
        return None

    # re-attach fills so filled_qty / avg_fill_price are restored
    for f in fills:
        order.fills.append(f)
        prev_notional = order.filled_qty * order.avg_fill_price
        order.filled_qty += f.qty
        if order.filled_qty > 0:
            order.avg_fill_price = (prev_notional + f.qty * f.price) / order.filled_qty

    return order


class OrderStateStore:
    """
    SQLite-backed persistence for order states and fills.

    Designed for single-process use.  Connection is opened per-call to allow
    safe use from multiple threads (check_same_thread=False with WAL mode).
    """

    def __init__(self, db_path: str = "order_state.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_ORDERS_TABLE)
            conn.execute(_CREATE_FILLS_TABLE)
            conn.execute(_CREATE_ORDER_IDX)
            conn.execute(_CREATE_FILL_IDX)
            conn.commit()
        logger.info("OrderStateStore initialized at %s", self._db_path)

    def persist(self, order: BaseOrder) -> None:
        """
        Insert a new order row.  Ignores duplicates (IGNORE conflict).
        """
        order_type = getattr(order, "order_type", type(order).__name__.upper())
        extra = _order_to_extra_json(order)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO orders
                        (order_id, symbol, side, qty, strategy_id,
                         signal_strength, created_at, status, order_type, extra_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        order.order_id,
                        order.symbol,
                        order.side,
                        order.qty,
                        order.strategy_id,
                        order.signal_strength,
                        _dt_to_str(order.created_at),
                        order.status.value,
                        order_type,
                        extra,
                    ),
                )
                conn.commit()

    def update(
        self,
        order_id: str,
        status: OrderStatus,
        fill: Optional[Fill] = None,
    ) -> None:
        """Update order status and optionally persist a fill row."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE orders SET status = ? WHERE order_id = ?",
                    (status.value, order_id),
                )
                if fill is not None:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO fills
                            (fill_id, order_id, symbol, side, qty, price,
                             timestamp, venue, commission_bps)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fill.fill_id,
                            fill.order_id,
                            fill.symbol,
                            fill.side,
                            fill.qty,
                            fill.price,
                            _dt_to_str(fill.timestamp),
                            fill.venue,
                            fill.commission_bps,
                        ),
                    )
                conn.commit()

    def load_open_orders(self) -> List[BaseOrder]:
        """
        Reload orders whose status is NEW / PENDING / PARTIAL.

        Used on startup to restore state after a crash or restart.
        """
        open_statuses = (
            OrderStatus.NEW.value,
            OrderStatus.PENDING.value,
            OrderStatus.PARTIAL.value,
        )
        placeholder = ",".join("?" * len(open_statuses))
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT * FROM orders WHERE status IN ({placeholder})",
                    open_statuses,
                ).fetchall()

                result: List[BaseOrder] = []
                for row in rows:
                    fill_rows = conn.execute(
                        "SELECT * FROM fills WHERE order_id = ? ORDER BY timestamp",
                        (row["order_id"],),
                    ).fetchall()
                    fills = [
                        Fill(
                            fill_id=fr["fill_id"],
                            order_id=fr["order_id"],
                            symbol=fr["symbol"],
                            side=fr["side"],
                            qty=fr["qty"],
                            price=fr["price"],
                            timestamp=_str_to_dt(fr["timestamp"]),
                            venue=fr["venue"],
                            commission_bps=fr["commission_bps"],
                        )
                        for fr in fill_rows
                    ]
                    order = _row_to_order(row, fills)
                    if order is not None:
                        result.append(order)
        logger.info(
            "load_open_orders restored %d open orders from %s",
            len(result), self._db_path,
        )
        return result

    def load_fills_since(self, since: datetime) -> List[Fill]:
        """Return all fills with timestamp >= since."""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM fills WHERE timestamp >= ? ORDER BY timestamp",
                    (_dt_to_str(since),),
                ).fetchall()
                return [
                    Fill(
                        fill_id=r["fill_id"],
                        order_id=r["order_id"],
                        symbol=r["symbol"],
                        side=r["side"],
                        qty=r["qty"],
                        price=r["price"],
                        timestamp=_str_to_dt(r["timestamp"]),
                        venue=r["venue"],
                        commission_bps=r["commission_bps"],
                    )
                    for r in rows
                ]


# ---------------------------------------------------------------------------
# OrderConflictChecker
# ---------------------------------------------------------------------------

MAX_OPEN_ORDERS_PER_SYMBOL = 3


class OrderConflictChecker:
    """
    Pre-submission guard that enforces:

    1. No duplicate orders: same (symbol, side) with status NEW/PENDING/PARTIAL
       and qty within 5% of the new order's qty.
    2. Concentration limit: no more than MAX_OPEN_ORDERS_PER_SYMBOL active
       orders for the same symbol.

    Returns None if no conflict is detected, or a human-readable string
    describing the conflict.
    """

    def check_conflict(
        self,
        new_order: BaseOrder,
        open_orders: List[BaseOrder],
    ) -> Optional[str]:
        """
        Check new_order against the list of open orders.

        Returns None on success, or an error string on conflict.
        """
        symbol_orders = [
            o for o in open_orders
            if o.symbol == new_order.symbol and o.is_active
        ]

        # Rule 1: concentration limit
        if len(symbol_orders) >= MAX_OPEN_ORDERS_PER_SYMBOL:
            return (
                f"Concentration limit: {len(symbol_orders)} open orders already "
                f"exist for {new_order.symbol!r} "
                f"(max {MAX_OPEN_ORDERS_PER_SYMBOL})"
            )

        # Rule 2: near-duplicate detection
        same_side = [o for o in symbol_orders if o.side == new_order.side]
        for existing in same_side:
            qty_ratio = (
                abs(existing.qty - new_order.qty) / max(existing.qty, new_order.qty)
            )
            if qty_ratio <= 0.05:
                return (
                    f"Duplicate order: active {existing.side} order "
                    f"{existing.order_id!r} for {existing.symbol!r} "
                    f"has qty={existing.qty} (within 5% of new qty={new_order.qty})"
                )

        return None

    def check_conflict_strict(
        self,
        new_order: BaseOrder,
        open_orders: List[BaseOrder],
    ) -> Optional[str]:
        """
        Strict variant: any same-side same-symbol active order is a conflict,
        regardless of quantity.  Used for risk-critical strategies.
        """
        symbol_orders = [
            o for o in open_orders
            if o.symbol == new_order.symbol and o.is_active
        ]

        if len(symbol_orders) >= MAX_OPEN_ORDERS_PER_SYMBOL:
            return (
                f"Concentration limit: {len(symbol_orders)} open orders "
                f"for {new_order.symbol!r}"
            )

        same_side = [o for o in symbol_orders if o.side == new_order.side]
        if same_side:
            ids = [o.order_id for o in same_side]
            return (
                f"Strict conflict: {len(same_side)} open {new_order.side} "
                f"order(s) already exist for {new_order.symbol!r}: {ids}"
            )

        return None
