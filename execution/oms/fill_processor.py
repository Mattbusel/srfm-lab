"""
execution/oms/fill_processor.py
================================
Fill processing and reconciliation for the SRFM execution layer.

Fills arrive from broker adapters as raw event dicts.  This module:
  1. Validates the fill (order exists, qty in range, price sanity).
  2. Computes commission using asset-class-aware rates.
  3. Updates order status via the OrderBook.
  4. Updates positions via PositionTracker (including realized P&L).
  5. Persists every fill to SQLite for end-of-day reconciliation.
  6. Aggregates fills per order into VWAP fill prices.
  7. Produces a DailyFillReport at session close.

Commission schedule
-------------------
  Equity  -- $0.005 per share (flat, SEC/FINRA rebate model)
  Crypto  -- 0.04 % of notional (maker/taker blend)
  Futures -- $1.25 per contract side (exchange + NFA)

Price sanity gate
-----------------
  Fills where |fill_price - midpoint| / midpoint > 5 % are flagged
  as SUSPICIOUS and persisted, but an alert is raised.

Duplicate detection
-------------------
  FillValidator maintains a seen-fill-ids set backed by the SQLite
  fills table.  Any fill_id already in the DB is rejected as a duplicate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("execution.fill_processor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EQUITY_COMMISSION_PER_SHARE: float = 0.005    # USD per share
CRYPTO_COMMISSION_RATE:       float = 0.0004   # 0.04 % of notional
FUTURES_COMMISSION_PER_SIDE:  float = 1.25     # USD per contract

PRICE_SANITY_THRESHOLD: float = 0.05   # 5 % deviation triggers SUSPICIOUS flag
OVERFILL_TOLERANCE:     float = 1e-6   # floating-point slop for qty checks

FILLS_DB_PATH = Path(__file__).parent.parent / "fills.db"


# ---------------------------------------------------------------------------
# Result / value types
# ---------------------------------------------------------------------------

@dataclass
class FillResult:
    """
    Value object returned by FillProcessor.process_fill().

    Attributes
    ----------
    order_id      : OMS order ID (UUID string)
    fill_id       : Broker-assigned fill identifier
    fill_qty      : Quantity filled in this event
    fill_price    : Execution price
    commission    : Commission charged (USD)
    realized_pnl  : Realized P&L for closing fills (0.0 for opening fills)
    status        : New order status after applying this fill
                    ('PARTIAL_FILL' or 'FILLED')
    suspicious    : True if the price deviated more than 5% from midpoint
    db_row_id     : SQLite rowid of the persisted fills record
    """
    order_id:     str
    fill_id:      str
    fill_qty:     float
    fill_price:   float
    commission:   float
    realized_pnl: float
    status:       str
    suspicious:   bool     = False
    db_row_id:    int      = 0

    def to_dict(self) -> dict:
        return {
            "order_id":     self.order_id,
            "fill_id":      self.fill_id,
            "fill_qty":     self.fill_qty,
            "fill_price":   self.fill_price,
            "commission":   self.commission,
            "realized_pnl": self.realized_pnl,
            "status":       self.status,
            "suspicious":   self.suspicious,
            "db_row_id":    self.db_row_id,
        }


# ---------------------------------------------------------------------------
# FillValidator
# ---------------------------------------------------------------------------

class FillValidator:
    """
    Stateful validator for incoming fill events.

    Checks performed
    ----------------
    1. fill_id uniqueness -- duplicate fill detection via SQLite-backed set.
    2. Quantity sanity    -- fill_qty must be > 0 and <= remaining order qty.
    3. Price sanity       -- |fill_price - midpoint| / midpoint <= 10 %
                             (flags SUSPICIOUS if > 5 %, rejects if > 10 %).

    Parameters
    ----------
    conn : sqlite3.Connection
        Shared fills database connection.  The validator reads the fills
        table to pre-load known fill IDs on construction.
    """

    SUSPICIOUS_THRESH: float = 0.05   # flag as suspicious above 5 %
    REJECT_THRESH:     float = 0.10   # hard reject above 10 %

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._seen_fills: set[str] = set()
        self._lock = threading.Lock()
        self._load_seen_fills()

    def _load_seen_fills(self) -> None:
        """Pre-populate seen set from persisted fills table."""
        try:
            cur = self._conn.execute("SELECT fill_id FROM fills")
            for row in cur.fetchall():
                self._seen_fills.add(row[0])
        except sqlite3.OperationalError:
            pass  -- table not yet created; will be populated on first write

    def register_fill(self, fill_id: str) -> None:
        """Mark a fill_id as processed so duplicates are caught."""
        with self._lock:
            self._seen_fills.add(fill_id)

    def is_duplicate(self, fill_id: str) -> bool:
        with self._lock:
            return fill_id in self._seen_fills

    def validate(
        self,
        fill_event: dict,
        order,             -- Order object from OrderBook
        midpoint: float,
    ) -> Tuple[bool, str, bool]:
        """
        Validate a fill event.

        Parameters
        ----------
        fill_event : dict
            Raw fill dict from broker adapter.  Expected keys:
            fill_id, fill_qty, fill_price, [asset_class].
        order : Order
            The matching OMS order.
        midpoint : float
            Current mid-price (bid+ask)/2 used for price sanity.
            Pass 0.0 to skip price check.

        Returns
        -------
        (valid, reason, suspicious)
        """
        fill_id    = fill_event.get("fill_id", "")
        fill_qty   = float(fill_event.get("fill_qty", 0.0))
        fill_price = float(fill_event.get("fill_price", 0.0))

        -- duplicate check
        if self.is_duplicate(fill_id):
            return False, f"Duplicate fill_id={fill_id}", False

        -- quantity sanity
        if fill_qty <= 0:
            return False, f"fill_qty={fill_qty} <= 0", False

        remaining = order.remaining_qty
        if fill_qty > remaining + OVERFILL_TOLERANCE:
            return False, (
                f"Over-fill: fill_qty={fill_qty:.6f} > remaining={remaining:.6f} "
                f"for order={order.order_id}"
            ), False

        -- price sanity
        suspicious = False
        if midpoint > 0 and fill_price > 0:
            dev = abs(fill_price - midpoint) / midpoint
            if dev > self.REJECT_THRESH:
                return False, (
                    f"Price deviation {dev:.2%} > hard reject threshold "
                    f"{self.REJECT_THRESH:.0%} -- fill_price={fill_price:.4f} "
                    f"midpoint={midpoint:.4f}"
                ), False
            if dev > self.SUSPICIOUS_THRESH:
                suspicious = True
                log.warning(
                    "SUSPICIOUS fill: order=%s fill_price=%.4f midpoint=%.4f dev=%.2%%",
                    order.order_id, fill_price, midpoint, dev * 100,
                )

        return True, "", suspicious


# ---------------------------------------------------------------------------
# FillAggregator
# ---------------------------------------------------------------------------

class FillAggregator:
    """
    Aggregates multiple partial fills per order into VWAP statistics.

    Use case: when an order is filled across multiple execution slices or
    time intervals, the aggregator computes the volume-weighted average
    fill price and total volume.

    Thread safety: all methods acquire an internal RLock.
    """

    def __init__(self) -> None:
        -- order_id -> list of (qty, price) tuples
        self._fills: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self._lock = threading.RLock()

    def record(self, order_id: str, qty: float, price: float) -> None:
        """Record a partial fill for VWAP aggregation."""
        with self._lock:
            self._fills[order_id].append((qty, price))

    def vwap(self, order_id: str) -> Optional[float]:
        """
        Return the volume-weighted average price for all fills on order_id.
        Returns None if no fills have been recorded.
        """
        with self._lock:
            entries = self._fills.get(order_id)
            if not entries:
                return None
            total_qty    = sum(q for q, _ in entries)
            total_notional = sum(q * p for q, p in entries)
            if total_qty <= 0:
                return None
            return total_notional / total_qty

    def total_qty(self, order_id: str) -> float:
        """Return cumulative filled quantity for order_id."""
        with self._lock:
            return sum(q for q, _ in self._fills.get(order_id, []))

    def total_notional(self, order_id: str) -> float:
        """Return cumulative notional (sum of qty * price) for order_id."""
        with self._lock:
            return sum(q * p for q, p in self._fills.get(order_id, []))

    def fill_count(self, order_id: str) -> int:
        """Return number of partial fills recorded for order_id."""
        with self._lock:
            return len(self._fills.get(order_id, []))

    def get_all_vwaps(self) -> Dict[str, float]:
        """Return a snapshot dict of {order_id: vwap} for all orders."""
        with self._lock:
            result = {}
            for oid, entries in self._fills.items():
                total_qty = sum(q for q, _ in entries)
                if total_qty > 0:
                    result[oid] = sum(q * p for q, p in entries) / total_qty
            return result

    def clear(self, order_id: str) -> None:
        """Remove all aggregated fills for order_id (e.g. after EOD export)."""
        with self._lock:
            self._fills.pop(order_id, None)

    def clear_all(self) -> None:
        """Reset entire aggregator state."""
        with self._lock:
            self._fills.clear()


# ---------------------------------------------------------------------------
# DailyFillReport
# ---------------------------------------------------------------------------

@dataclass
class DailyFillSummary:
    """Per-symbol summary entry in the DailyFillReport."""
    symbol:         str
    total_qty:      float
    total_notional: float
    vwap:           float
    fill_count:     int
    total_commission: float
    total_slippage_bps: float   -- signed average slippage in bps
    realized_pnl:   float


class DailyFillReport:
    """
    End-of-day fill summary.

    Reads all fills from the fills SQLite table for a given trade date and
    computes per-symbol and portfolio-level statistics.

    Parameters
    ----------
    conn : sqlite3.Connection
        Shared fills DB connection.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def generate(self, trade_date: Optional[date] = None) -> dict:
        """
        Build the EOD fill report.

        Parameters
        ----------
        trade_date : date | None
            Date to generate the report for.  Defaults to today (UTC).

        Returns
        -------
        dict with keys: date, symbols (list of DailyFillSummary dicts),
        totals (portfolio-level aggregates).
        """
        target = trade_date or date.today()
        date_str = target.isoformat()

        cur = self._conn.execute(
            """
            SELECT order_id, symbol, side, fill_qty, fill_price,
                   commission, realized_pnl, slippage_bps
            FROM fills
            WHERE trade_date = ?
            ORDER BY symbol, fill_ts ASC
            """,
            (date_str,),
        )
        rows = cur.fetchall()

        -- aggregate per symbol
        sym_data: Dict[str, dict] = {}
        for (order_id, symbol, side, fill_qty, fill_price,
             commission, realized_pnl, slippage_bps) in rows:
            if symbol not in sym_data:
                sym_data[symbol] = {
                    "total_qty":        0.0,
                    "total_notional":   0.0,
                    "fill_count":       0,
                    "total_commission": 0.0,
                    "total_slippage_bps_weighted": 0.0,
                    "realized_pnl":     0.0,
                }
            d = sym_data[symbol]
            d["total_qty"]      += fill_qty
            d["total_notional"] += fill_qty * fill_price
            d["fill_count"]     += 1
            d["total_commission"] += commission
            d["total_slippage_bps_weighted"] += (slippage_bps or 0.0) * fill_qty
            d["realized_pnl"]   += realized_pnl or 0.0

        summaries = []
        for symbol, d in sym_data.items():
            qty = d["total_qty"]
            vwap = d["total_notional"] / qty if qty > 0 else 0.0
            avg_slip = d["total_slippage_bps_weighted"] / qty if qty > 0 else 0.0
            summaries.append({
                "symbol":           symbol,
                "total_qty":        qty,
                "total_notional":   d["total_notional"],
                "vwap":             vwap,
                "fill_count":       d["fill_count"],
                "total_commission": d["total_commission"],
                "avg_slippage_bps": avg_slip,
                "realized_pnl":     d["realized_pnl"],
            })

        total_commission = sum(d["total_commission"] for d in sym_data.values())
        total_realized   = sum(d["realized_pnl"]     for d in sym_data.values())
        total_volume     = sum(d["total_notional"]    for d in sym_data.values())
        total_fills      = sum(d["fill_count"]        for d in sym_data.values())

        return {
            "date":             date_str,
            "symbols":          summaries,
            "total_fills":      total_fills,
            "total_volume_usd": total_volume,
            "total_commission": total_commission,
            "total_realized_pnl": total_realized,
            "net_pnl":          total_realized - total_commission,
        }


# ---------------------------------------------------------------------------
# FillProcessor
# ---------------------------------------------------------------------------

class FillProcessor:
    """
    Processes fill events from broker adapters and updates OMS state.

    Handles partial fills, over-fills, and fill rejections.  All fills are
    persisted to SQLite for reconciliation and end-of-day reporting.

    Parameters
    ----------
    order_book : OrderBook
        The OMS OrderBook holding live order state.
    position_tracker : PositionTracker
        Real-time position tracker updated on every fill.
    db_path : Path | str | None
        SQLite database for fill persistence.  Defaults to fills.db.
    event_bus : callable | None
        Optional event callback invoked with (event_type, payload) on
        every processed fill.
    """

    def __init__(
        self,
        order_book,
        position_tracker,
        db_path: Optional[Path | str] = None,
        event_bus=None,
    ) -> None:
        self._book            = order_book
        self._position_tracker = position_tracker
        self._event_bus       = event_bus
        self._db_path         = Path(db_path) if db_path else FILLS_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn            = self._connect()
        self._init_schema()
        self._validator       = FillValidator(self._conn)
        self._aggregator      = FillAggregator()
        self._lock            = threading.RLock()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                fill_id       TEXT    NOT NULL UNIQUE,
                order_id      TEXT    NOT NULL,
                symbol        TEXT    NOT NULL,
                side          TEXT    NOT NULL,
                fill_qty      REAL    NOT NULL,
                fill_price    REAL    NOT NULL,
                commission    REAL    NOT NULL DEFAULT 0.0,
                realized_pnl  REAL    NOT NULL DEFAULT 0.0,
                slippage_bps  REAL,
                status        TEXT    NOT NULL,
                suspicious    INTEGER NOT NULL DEFAULT 0,
                asset_class   TEXT    NOT NULL DEFAULT 'equity',
                trade_date    TEXT    NOT NULL,
                fill_ts       TEXT    NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fills_order ON fills (order_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills (symbol)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fills_date ON fills (trade_date)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Commission calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_commission(
        fill_qty: float,
        fill_price: float,
        asset_class: str,
    ) -> float:
        """
        Compute commission for a fill.

        asset_class values: 'equity', 'crypto', 'futures'.
        """
        cls = asset_class.lower()
        if cls == "equity":
            return round(fill_qty * EQUITY_COMMISSION_PER_SHARE, 6)
        elif cls == "crypto":
            return round(fill_qty * fill_price * CRYPTO_COMMISSION_RATE, 6)
        elif cls == "futures":
            return FUTURES_COMMISSION_PER_SIDE
        -- default: treat as equity
        return round(fill_qty * EQUITY_COMMISSION_PER_SHARE, 6)

    # ------------------------------------------------------------------
    # Realized P&L helper
    # ------------------------------------------------------------------

    def _compute_realized_pnl(self, order, fill_qty: float, fill_price: float) -> float:
        """
        Compute realized P&L for a closing fill.

        For BUY orders, there is no realized P&L (opening or adding).
        For SELL orders, realized P&L = (fill_price - avg_entry_price) * fill_qty,
        fetched from PositionTracker before recording the fill.

        Returns 0.0 for opening trades.
        """
        from .order import Side
        if order.side != Side.SELL:
            return 0.0

        pos = self._position_tracker.positions.get(order.symbol)
        if pos is None or pos.avg_entry_price <= 0:
            return 0.0

        realized = (fill_price - pos.avg_entry_price) * fill_qty
        return round(realized, 6)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_fill(
        self,
        fill_id:     str,
        order,
        fill_qty:    float,
        fill_price:  float,
        commission:  float,
        realized_pnl: float,
        status:      str,
        suspicious:  bool,
        asset_class: str,
    ) -> int:
        """Write fill to SQLite.  Returns the new rowid."""
        now     = datetime.now(timezone.utc)
        slippage_bps = order.slippage_bps  -- may be None until FILLED

        self._conn.execute(
            """
            INSERT INTO fills
                (fill_id, order_id, symbol, side, fill_qty, fill_price,
                 commission, realized_pnl, slippage_bps, status, suspicious,
                 asset_class, trade_date, fill_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill_id, order.order_id, order.symbol,
                order.side.value if hasattr(order.side, "value") else str(order.side),
                fill_qty, fill_price, commission, realized_pnl,
                slippage_bps, status,
                1 if suspicious else 0,
                asset_class,
                now.date().isoformat(),
                now.isoformat(),
            ),
        )
        self._conn.commit()
        cur = self._conn.execute("SELECT last_insert_rowid()")
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def process_fill(
        self,
        fill_event: dict,
        midpoint: float = 0.0,
    ) -> Optional[FillResult]:
        """
        Process a fill event from a broker adapter.

        Parameters
        ----------
        fill_event : dict
            Required keys: order_id (or broker_order_id), fill_id,
            fill_qty, fill_price.
            Optional keys: asset_class ('equity'|'crypto'|'futures').
        midpoint : float
            Current bid/ask midpoint for price sanity check.  Pass 0 to skip.

        Returns
        -------
        FillResult on success, or None if the fill is rejected (duplicate,
        over-fill, hard price violation, or order not found).
        """
        with self._lock:
            -- resolve order
            order_id = fill_event.get("order_id")
            broker_id = fill_event.get("broker_order_id")
            order = None
            if order_id:
                order = self._book.get(order_id)
            if order is None and broker_id:
                order = self._book.get_by_broker_id(broker_id)
            if order is None:
                log.error(
                    "process_fill: order not found -- order_id=%s broker_id=%s",
                    order_id, broker_id,
                )
                return None

            fill_id    = fill_event.get("fill_id") or str(uuid.uuid4())
            fill_qty   = float(fill_event.get("fill_qty", 0.0))
            fill_price = float(fill_event.get("fill_price", 0.0))
            asset_class = fill_event.get("asset_class", "equity")

            -- validate
            valid, reason, suspicious = self._validator.validate(
                fill_event, order, midpoint
            )
            if not valid:
                log.warning(
                    "process_fill REJECTED: order=%s fill_id=%s reason=%s",
                    order.order_id, fill_id, reason,
                )
                return None

            -- compute commission before updating order state
            commission = self._compute_commission(fill_qty, fill_price, asset_class)

            -- compute realized P&L before updating position
            realized_pnl = self._compute_realized_pnl(order, fill_qty, fill_price)

            -- determine if this is a full or partial fill
            remaining = order.remaining_qty
            is_full   = fill_qty >= remaining - OVERFILL_TOLERANCE

            -- update order state
            if is_full:
                order.mark_filled(
                    fill_qty       = order.fill_qty + fill_qty,
                    fill_price     = fill_price,
                    commission_usd = order.commission_usd + commission,
                )
                new_status = "FILLED"
            else:
                order.mark_partial(
                    fill_qty   = order.fill_qty + fill_qty,
                    fill_price = fill_price,
                )
                order.commission_usd += commission
                new_status = "PARTIAL_FILL"

            -- update position tracker
            self._position_tracker.record_fill(order)

            -- update aggregator
            self._aggregator.record(order.order_id, fill_qty, fill_price)

            -- persist
            row_id = self._persist_fill(
                fill_id, order, fill_qty, fill_price,
                commission, realized_pnl, new_status,
                suspicious, asset_class,
            )

            -- register fill_id as seen
            self._validator.register_fill(fill_id)

            result = FillResult(
                order_id     = order.order_id,
                fill_id      = fill_id,
                fill_qty     = fill_qty,
                fill_price   = fill_price,
                commission   = commission,
                realized_pnl = realized_pnl,
                status       = new_status,
                suspicious   = suspicious,
                db_row_id    = row_id,
            )

            -- emit event
            if self._event_bus:
                try:
                    self._event_bus("FILL_PROCESSED", result.to_dict())
                except Exception as exc:
                    log.error("event_bus error (FILL_PROCESSED): %s", exc)

            log.info(
                "Fill processed: order=%s fill_id=%s qty=%.6f @ %.4f "
                "comm=%.4f rpnl=%.4f status=%s%s",
                order.order_id, fill_id, fill_qty, fill_price,
                commission, realized_pnl, new_status,
                " [SUSPICIOUS]" if suspicious else "",
            )

            return result

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def aggregator(self) -> FillAggregator:
        """Access the FillAggregator for VWAP queries."""
        return self._aggregator

    @property
    def validator(self) -> FillValidator:
        """Access the FillValidator for duplicate checks."""
        return self._validator

    def get_daily_report(self, trade_date: Optional[date] = None) -> dict:
        """Generate and return the DailyFillReport for the given date."""
        reporter = DailyFillReport(self._conn)
        return reporter.generate(trade_date)

    def get_fills_for_order(self, order_id: str) -> List[dict]:
        """
        Return all persisted fills for an order.

        Returns list of dicts with fill fields.
        """
        cur = self._conn.execute(
            """
            SELECT fill_id, fill_qty, fill_price, commission,
                   realized_pnl, status, suspicious, fill_ts
            FROM fills
            WHERE order_id = ?
            ORDER BY fill_ts ASC
            """,
            (order_id,),
        )
        cols = ["fill_id", "fill_qty", "fill_price", "commission",
                "realized_pnl", "status", "suspicious", "fill_ts"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        """Close SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
