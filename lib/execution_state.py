"""
execution_state.py -- single source of truth for active orders and positions.

Thread-safe via RLock. Crash-recoverable via SQLite WAL mode.
Supports full snapshot serialization and restore.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# -- order status constants
STATUS_PENDING   = "PENDING"
STATUS_FILLED    = "FILLED"
STATUS_CANCELLED = "CANCELLED"
STATUS_PARTIAL   = "PARTIAL"


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str            # BUY | SELL
    qty: float
    order_type: str      # MARKET | LIMIT | STOP
    limit_price: Optional[float]
    status: str          # PENDING | FILLED | CANCELLED | PARTIAL
    created_ts: int
    filled_qty: float = 0.0
    fill_price: Optional[float] = None
    fill_ts: Optional[int] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class Position:
    symbol: str
    qty: float           # positive = long, negative = short
    avg_price: float
    updated_ts: int
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ExecutionState:
    """
    Single source of truth for all active orders and positions during live trading.
    Thread-safe with RLock. Persistent via SQLite WAL for crash recovery.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._lock = threading.RLock()
        self._orders: dict[str, Order] = {}        # order_id -> Order
        self._positions: dict[str, Position] = {}  # symbol -> Position
        self._realized_pnl: float = 0.0
        self.db_path = db_path

        if db_path:
            self._init_db()

    # ------------------------------------------------------------------
    # SQLite setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS es_orders (
                    order_id    TEXT PRIMARY KEY,
                    symbol      TEXT NOT NULL,
                    side        TEXT NOT NULL,
                    qty         REAL NOT NULL,
                    order_type  TEXT NOT NULL,
                    limit_price REAL,
                    status      TEXT NOT NULL,
                    created_ts  INTEGER NOT NULL,
                    filled_qty  REAL NOT NULL DEFAULT 0,
                    fill_price  REAL,
                    fill_ts     INTEGER,
                    meta_json   TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS es_positions (
                    symbol      TEXT PRIMARY KEY,
                    qty         REAL NOT NULL,
                    avg_price   REAL NOT NULL,
                    updated_ts  INTEGER NOT NULL,
                    meta_json   TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS es_realized_pnl (
                    id          INTEGER PRIMARY KEY CHECK (id = 1),
                    pnl         REAL NOT NULL DEFAULT 0
                )
            """)
            conn.execute(
                "INSERT OR IGNORE INTO es_realized_pnl (id, pnl) VALUES (1, 0)"
            )
            conn.commit()

    def _write_order(self, o: Order) -> None:
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """INSERT OR REPLACE INTO es_orders
                       (order_id,symbol,side,qty,order_type,limit_price,status,
                        created_ts,filled_qty,fill_price,fill_ts,meta_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        o.order_id, o.symbol, o.side, o.qty, o.order_type,
                        o.limit_price, o.status, o.created_ts, o.filled_qty,
                        o.fill_price, o.fill_ts, json.dumps(o.meta),
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.error("es_orders write error: %s", exc)

    def _write_position(self, p: Position) -> None:
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """INSERT OR REPLACE INTO es_positions
                       (symbol,qty,avg_price,updated_ts,meta_json)
                       VALUES (?,?,?,?,?)""",
                    (p.symbol, p.qty, p.avg_price, p.updated_ts, json.dumps(p.meta)),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.error("es_positions write error: %s", exc)

    def _write_rpnl(self) -> None:
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    "UPDATE es_realized_pnl SET pnl=? WHERE id=1",
                    (self._realized_pnl,),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.error("es_realized_pnl write error: %s", exc)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def add_order(self, order: dict) -> str:
        """
        Record a pending order. If order_id is absent a UUID is assigned.
        Returns the order_id.
        """
        with self._lock:
            order_id = order.get("order_id") or str(uuid.uuid4())
            o = Order(
                order_id=order_id,
                symbol=order["symbol"],
                side=order["side"],
                qty=float(order["qty"]),
                order_type=order.get("order_type", "MARKET"),
                limit_price=order.get("limit_price"),
                status=STATUS_PENDING,
                created_ts=order.get("created_ts", int(time.time())),
                meta=order.get("meta", {}),
            )
            self._orders[order_id] = o
            self._write_order(o)
            logger.debug("order added: %s %s %s", order_id, o.symbol, o.side)
            return order_id

    def fill_order(
        self,
        order_id: str,
        fill_price: float,
        fill_qty: float,
        ts: int,
    ) -> None:
        """Mark an order as FILLED (or PARTIAL if fill_qty < order qty)."""
        with self._lock:
            o = self._orders.get(order_id)
            if o is None:
                logger.warning("fill_order: unknown order_id %s", order_id)
                return
            o.filled_qty += fill_qty
            o.fill_price = fill_price
            o.fill_ts = ts
            o.status = STATUS_FILLED if o.filled_qty >= o.qty else STATUS_PARTIAL
            self._write_order(o)
            logger.debug(
                "order filled: %s @ %.4f qty=%.2f status=%s",
                order_id, fill_price, fill_qty, o.status,
            )

    def cancel_order(self, order_id: str) -> None:
        """Mark order as CANCELLED."""
        with self._lock:
            o = self._orders.get(order_id)
            if o is None:
                logger.warning("cancel_order: unknown order_id %s", order_id)
                return
            o.status = STATUS_CANCELLED
            self._write_order(o)

    def get_open_orders(self) -> list[dict]:
        """Return all orders with status PENDING or PARTIAL."""
        with self._lock:
            return [
                o.to_dict()
                for o in self._orders.values()
                if o.status in (STATUS_PENDING, STATUS_PARTIAL)
            ]

    def get_order(self, order_id: str) -> Optional[dict]:
        with self._lock:
            o = self._orders.get(order_id)
            return o.to_dict() if o else None

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def update_position(self, symbol: str, qty: float, avg_price: float) -> None:
        """
        Overwrite the position for symbol. If qty is 0 the position is removed.
        """
        with self._lock:
            ts = int(time.time())
            if qty == 0.0:
                self._positions.pop(symbol, None)
                return
            pos = Position(
                symbol=symbol,
                qty=qty,
                avg_price=avg_price,
                updated_ts=ts,
            )
            self._positions[symbol] = pos
            self._write_position(pos)

    def get_position(self, symbol: str) -> dict:
        """Return position dict for symbol, or empty dict if no position."""
        with self._lock:
            pos = self._positions.get(symbol)
            return pos.to_dict() if pos else {}

    def all_positions(self) -> list[dict]:
        with self._lock:
            return [p.to_dict() for p in self._positions.values()]

    # ------------------------------------------------------------------
    # P&L
    # ------------------------------------------------------------------

    def compute_unrealized_pnl(self, prices: dict[str, float]) -> float:
        """
        Sum unrealized P&L across all open positions given current prices.
        Long: (current - avg) * qty; Short: (avg - current) * abs(qty).
        """
        with self._lock:
            total = 0.0
            for symbol, pos in self._positions.items():
                price = prices.get(symbol)
                if price is None:
                    continue
                if pos.qty > 0:
                    total += (price - pos.avg_price) * pos.qty
                else:
                    total += (pos.avg_price - price) * abs(pos.qty)
            return total

    def add_realized_pnl(self, amount: float) -> None:
        """Accumulate realized P&L (called externally after a trade closes)."""
        with self._lock:
            self._realized_pnl += amount
            self._write_rpnl()

    def compute_realized_pnl(self) -> float:
        with self._lock:
            return self._realized_pnl

    def reset_realized_pnl(self) -> None:
        """Reset accumulated realized P&L (e.g. at session open)."""
        with self._lock:
            self._realized_pnl = 0.0
            self._write_rpnl()

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def to_snapshot(self) -> dict:
        """Full state as JSON-serializable dict for crash recovery."""
        with self._lock:
            return {
                "orders": {oid: o.to_dict() for oid, o in self._orders.items()},
                "positions": {sym: p.to_dict() for sym, p in self._positions.items()},
                "realized_pnl": self._realized_pnl,
                "snapshot_ts": int(time.time()),
            }

    def restore_from_snapshot(self, snapshot: dict) -> None:
        """Restore state from a snapshot dict produced by to_snapshot()."""
        with self._lock:
            self._orders.clear()
            self._positions.clear()

            for oid, od in snapshot.get("orders", {}).items():
                self._orders[oid] = Order(**od)

            for sym, pd in snapshot.get("positions", {}).items():
                self._positions[sym] = Position(**pd)

            self._realized_pnl = snapshot.get("realized_pnl", 0.0)
            logger.info(
                "execution_state restored: %d orders, %d positions",
                len(self._orders),
                len(self._positions),
            )

    def restore_from_db(self) -> None:
        """Reload state from SQLite tables (used after crash recovery)."""
        if not self.db_path:
            return
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("PRAGMA journal_mode=WAL")
                    rows = conn.execute(
                        "SELECT order_id,symbol,side,qty,order_type,limit_price,"
                        "status,created_ts,filled_qty,fill_price,fill_ts,meta_json "
                        "FROM es_orders"
                    ).fetchall()
                    for row in rows:
                        meta = json.loads(row[11]) if row[11] else {}
                        o = Order(
                            order_id=row[0], symbol=row[1], side=row[2],
                            qty=row[3], order_type=row[4], limit_price=row[5],
                            status=row[6], created_ts=row[7], filled_qty=row[8],
                            fill_price=row[9], fill_ts=row[10], meta=meta,
                        )
                        self._orders[o.order_id] = o

                    prows = conn.execute(
                        "SELECT symbol,qty,avg_price,updated_ts,meta_json FROM es_positions"
                    ).fetchall()
                    for row in prows:
                        meta = json.loads(row[4]) if row[4] else {}
                        p = Position(
                            symbol=row[0], qty=row[1], avg_price=row[2],
                            updated_ts=row[3], meta=meta,
                        )
                        self._positions[p.symbol] = p

                    rpnl_row = conn.execute(
                        "SELECT pnl FROM es_realized_pnl WHERE id=1"
                    ).fetchone()
                    if rpnl_row:
                        self._realized_pnl = rpnl_row[0]

                logger.info(
                    "execution_state loaded from db: %d orders, %d positions",
                    len(self._orders), len(self._positions),
                )
            except sqlite3.Error as exc:
                logger.error("restore_from_db failed: %s", exc)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        with self._lock:
            open_count = sum(
                1 for o in self._orders.values()
                if o.status in (STATUS_PENDING, STATUS_PARTIAL)
            )
            return {
                "total_orders": len(self._orders),
                "open_orders": open_count,
                "positions": len(self._positions),
                "realized_pnl": self._realized_pnl,
            }
