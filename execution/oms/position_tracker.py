"""
execution/oms/position_tracker.py
==================================
Real-time position tracking with FIFO cost-basis accounting.

Each ``Position`` stores the current quantity, average entry price, and
running P&L.  When a fill arrives the FIFO lot queue is updated so that
partial exits accurately compute realized P&L.

Thread safety
-------------
All public methods acquire a ``threading.RLock``.  ``update_price`` is
designed to be called from a high-frequency market-data callback and is
O(n) in the number of positions (typically < 30), which is acceptable.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("execution.position_tracker")


# ---------------------------------------------------------------------------
# Lot (FIFO queue element)
# ---------------------------------------------------------------------------

@dataclass
class Lot:
    """A single acquired lot for FIFO cost-basis tracking."""
    qty:         float
    entry_price: float
    acquired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    Real-time state of a single instrument holding.

    Attributes
    ----------
    symbol : str
        Instrument symbol.
    quantity : float
        Net signed quantity (positive = long).
    avg_entry_price : float
        Weighted average entry price of current lots.
    market_value : float
        quantity * last_price (mark-to-market).
    unrealized_pnl : float
        market_value - cost_basis.
    realized_pnl : float
        Running sum of closed P&L (all time).
    cost_basis : float
        abs(quantity) * avg_entry_price.
    last_updated : datetime
        Timestamp of last mark-to-market.
    last_price : float
        Most recent market price used for MTM.
    """

    symbol:          str
    quantity:        float         = 0.0
    avg_entry_price: float         = 0.0
    market_value:    float         = 0.0
    unrealized_pnl:  float         = 0.0
    realized_pnl:    float         = 0.0
    cost_basis:      float         = 0.0
    last_updated:    datetime      = field(default_factory=lambda: datetime.now(timezone.utc))
    last_price:      float         = 0.0
    _lots:           deque         = field(default_factory=deque, repr=False)

    def mark_to_market(self, price: float) -> None:
        """Update market_value and unrealized_pnl using *price*."""
        self.last_price     = price
        self.market_value   = self.quantity * price
        self.unrealized_pnl = self.market_value - self.cost_basis
        self.last_updated   = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "symbol":          self.symbol,
            "quantity":        self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "market_value":    self.market_value,
            "unrealized_pnl":  self.unrealized_pnl,
            "realized_pnl":    self.realized_pnl,
            "cost_basis":      self.cost_basis,
            "last_price":      self.last_price,
            "last_updated":    self.last_updated.isoformat(),
        }


# ---------------------------------------------------------------------------
# PositionTracker
# ---------------------------------------------------------------------------

class PositionTracker:
    """
    Maintains a real-time view of all open positions.

    FIFO cost-basis
    ---------------
    Each ``record_fill`` for a BUY appends a Lot to the FIFO queue.
    A SELL pops lots from the front, computing realized P&L for each
    lot consumed.

    Usage
    -----
    ::

        tracker = PositionTracker(initial_equity=100_000)
        tracker.record_fill(filled_order)
        tracker.update_price("BTC/USD", 65_000.0)
        snapshot = tracker.export_snapshot()
    """

    def __init__(self, initial_equity: float = 100_000.0) -> None:
        self.positions: dict[str, Position] = {}
        self._equity   = initial_equity
        self._lock     = threading.RLock()

    # ------------------------------------------------------------------
    # Equity
    # ------------------------------------------------------------------

    def set_equity(self, equity: float) -> None:
        with self._lock:
            self._equity = equity

    # ------------------------------------------------------------------
    # Fill processing
    # ------------------------------------------------------------------

    def record_fill(self, order) -> None:
        """
        Update position state from a filled (or partially-filled) Order.

        Handles both BUY (open/add) and SELL (close/reduce) directions.
        """
        from .order import Side, OrderStatus

        with self._lock:
            sym   = order.symbol
            qty   = order.fill_qty
            price = order.fill_price

            if qty is None or price is None or qty <= 0:
                return

            pos = self.positions.setdefault(sym, Position(symbol=sym))

            if order.side == Side.BUY:
                self._apply_buy(pos, qty, price)
            else:
                self._apply_sell(pos, qty, price)

    def _apply_buy(self, pos: Position, qty: float, price: float) -> None:
        """Add a BUY lot using weighted-average entry price."""
        total_cost     = pos.avg_entry_price * pos.quantity + price * qty
        pos.quantity  += qty
        pos.avg_entry_price = total_cost / pos.quantity if pos.quantity > 0 else price
        pos.cost_basis      = abs(pos.quantity) * pos.avg_entry_price
        pos._lots.append(Lot(qty=qty, entry_price=price))
        pos.last_updated = datetime.now(timezone.utc)

    def _apply_sell(self, pos: Position, qty: float, price: float) -> None:
        """Consume FIFO lots for a SELL, computing realized P&L."""
        remaining = qty
        realized  = 0.0

        while remaining > 1e-9 and pos._lots:
            lot = pos._lots[0]
            if lot.qty <= remaining + 1e-9:
                # Consume entire lot
                realized  += lot.qty * (price - lot.entry_price)
                remaining -= lot.qty
                pos._lots.popleft()
            else:
                # Partially consume front lot
                realized        += remaining * (price - lot.entry_price)
                lot.qty         -= remaining
                remaining        = 0.0

        pos.realized_pnl += realized
        pos.quantity      = max(0.0, pos.quantity - qty)
        pos.cost_basis    = abs(pos.quantity) * pos.avg_entry_price
        pos.last_updated  = datetime.now(timezone.utc)

        if pos.quantity < 1e-9:
            # Position fully closed
            pos.quantity        = 0.0
            pos.avg_entry_price = 0.0
            pos.cost_basis      = 0.0

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float) -> None:
        """
        Mark a position to market.  Creates a zero-quantity stub if the
        symbol has never been seen (e.g. for watchlist display).
        """
        with self._lock:
            pos = self.positions.get(symbol)
            if pos:
                pos.mark_to_market(price)

    def update_all_prices(self, prices: dict[str, float]) -> None:
        """Batch mark-to-market update from a price snapshot dict."""
        with self._lock:
            for sym, price in prices.items():
                if sym in self.positions:
                    self.positions[sym].mark_to_market(price)

    # ------------------------------------------------------------------
    # Direct quantity override (used by reconciler)
    # ------------------------------------------------------------------

    def set_quantity(self, symbol: str, quantity: float) -> None:
        """Forcibly set a position's quantity (used by reconciliation)."""
        with self._lock:
            pos = self.positions.setdefault(symbol, Position(symbol=symbol))
            pos.quantity     = quantity
            pos.cost_basis   = abs(quantity) * pos.avg_entry_price
            pos.last_updated = datetime.now(timezone.utc)
            log.info("set_quantity: %s -> %.6f (reconcile override)", symbol, quantity)

    # ------------------------------------------------------------------
    # Portfolio metrics
    # ------------------------------------------------------------------

    def get_portfolio_value(self) -> float:
        """
        Return total market value of all positions.

        This is gross notional — add cash to get net equity.
        """
        with self._lock:
            return sum(p.market_value for p in self.positions.values())

    def get_total_realized_pnl(self) -> float:
        """Sum of realized P&L across all positions."""
        with self._lock:
            return sum(p.realized_pnl for p in self.positions.values())

    def get_total_unrealized_pnl(self) -> float:
        """Sum of unrealized P&L across all positions."""
        with self._lock:
            return sum(p.unrealized_pnl for p in self.positions.values())

    def get_leverage(self) -> float:
        """
        Return current leverage ratio: total gross notional / equity.

        A value of 1.0 means fully invested with no leverage.
        """
        with self._lock:
            equity = self._equity
            if equity <= 0:
                return 0.0
            gross = sum(abs(p.market_value) for p in self.positions.values())
            return gross / equity

    def get_position_fractions(self) -> dict[str, float]:
        """Return {symbol: fraction_of_equity} for all non-zero positions."""
        with self._lock:
            equity = self._equity or 1.0
            return {
                sym: abs(p.market_value) / equity
                for sym, p in self.positions.items()
                if abs(p.quantity) > 1e-9
            }

    # ------------------------------------------------------------------
    # Snapshot export
    # ------------------------------------------------------------------

    def export_snapshot(self) -> dict:
        """
        Return a JSON-serialisable snapshot of the entire portfolio.

        Suitable for writing to ``execution/status.json`` by the live monitor.
        """
        with self._lock:
            positions_dict = {sym: p.to_dict() for sym, p in self.positions.items()}
            return {
                "timestamp":        datetime.now(timezone.utc).isoformat(),
                "equity":           self._equity,
                "portfolio_value":  self.get_portfolio_value(),
                "leverage":         self.get_leverage(),
                "realized_pnl":     self.get_total_realized_pnl(),
                "unrealized_pnl":   self.get_total_unrealized_pnl(),
                "positions":        positions_dict,
                "position_count":   sum(
                    1 for p in self.positions.values() if abs(p.quantity) > 1e-9
                ),
            }
