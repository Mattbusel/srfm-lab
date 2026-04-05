"""
execution/orderbook/orderbook.py
=================================
Pure-Python L2 order-book data structure.

Thread-safe; all public properties and methods acquire a lock so an async feed
writer and a synchronous strategy reader can coexist safely.

Design notes
------------
- Bids and asks are stored as ``dict[float, float]`` (price -> cumulative qty).
- ``best_bid`` / ``best_ask`` scan sorted keys — O(n) but n is always small
  (≤ 25 levels in practice) so this avoids the overhead of a heap.
- ``vwap_to_fill`` walks the book and raises ``InsufficientLiquidityError`` if
  the book does not have enough depth to fill the requested quantity.
"""

from __future__ import annotations

import math
import threading
from typing import Optional


class InsufficientLiquidityError(Exception):
    """Raised when the book cannot absorb the requested fill quantity."""


class OrderBook:
    """
    Level-2 order book for a single trading symbol.

    Parameters
    ----------
    symbol : str
        Instrument identifier (e.g. ``"BTC/USD"``).
    max_levels : int
        Maximum number of price levels to retain per side (default 25).
    """

    def __init__(self, symbol: str, max_levels: int = 25) -> None:
        self.symbol = symbol
        self._max_levels = max_levels
        self._bids: dict[float, float] = {}   # price -> qty
        self._asks: dict[float, float] = {}
        self._lock = threading.Lock()
        self._last_update_ts: float = 0.0      # epoch seconds

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(self, side: str, price: float, qty: float) -> None:
        """
        Apply a single level update.

        Parameters
        ----------
        side : str
            ``"bid"`` or ``"ask"`` (case-insensitive).
        price : float
            Price level to update.
        qty : float
            New quantity at this level.  ``qty == 0`` removes the level.
        """
        import time as _time

        side = side.lower()
        if side not in ("bid", "ask"):
            raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")

        with self._lock:
            book = self._bids if side == "bid" else self._asks
            if qty == 0.0:
                book.pop(price, None)
            else:
                book[price] = qty
            # Trim to max levels, keeping best prices
            if len(book) > self._max_levels:
                if side == "bid":
                    # Keep highest bids
                    for p in sorted(book.keys())[: -self._max_levels]:
                        del book[p]
                else:
                    # Keep lowest asks
                    for p in sorted(book.keys())[self._max_levels :]:
                        del book[p]
            self._last_update_ts = _time.time()

    def apply_snapshot(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> None:
        """
        Replace the entire book with a fresh snapshot.

        Parameters
        ----------
        bids : list of (price, qty)
        asks : list of (price, qty)
        """
        import time as _time

        with self._lock:
            self._bids = {p: q for p, q in bids if q > 0}
            self._asks = {p: q for p, q in asks if q > 0}
            self._last_update_ts = _time.time()

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> Optional[float]:
        """Highest bid price, or None if empty."""
        with self._lock:
            return max(self._bids.keys()) if self._bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Lowest ask price, or None if empty."""
        with self._lock:
            return min(self._asks.keys()) if self._asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Arithmetic midpoint of best bid and ask, or None."""
        with self._lock:
            if not self._bids or not self._asks:
                return None
            bb = max(self._bids.keys())
            ba = min(self._asks.keys())
            return (bb + ba) / 2.0

    @property
    def spread_bps(self) -> Optional[float]:
        """
        Bid-ask spread expressed in basis points.

        ``spread_bps = (ask - bid) / mid * 10_000``

        Returns None if either side is empty or mid is zero.
        """
        with self._lock:
            if not self._bids or not self._asks:
                return None
            bb = max(self._bids.keys())
            ba = min(self._asks.keys())
            mid = (bb + ba) / 2.0
            if mid == 0:
                return None
            return (ba - bb) / mid * 10_000.0

    @property
    def last_update_ts(self) -> float:
        """Unix timestamp of the most recent update."""
        return self._last_update_ts

    # ------------------------------------------------------------------
    # Depth snapshot
    # ------------------------------------------------------------------

    def depth(self, n: int = 5) -> dict[str, list[tuple[float, float]]]:
        """
        Return top-N levels for each side.

        Returns
        -------
        dict with keys ``"bids"`` and ``"asks"``, each a list of
        ``(price, qty)`` tuples ordered best-to-worst.
        """
        with self._lock:
            bids = sorted(self._bids.items(), reverse=True)[:n]
            asks = sorted(self._asks.items())[:n]
        return {"bids": bids, "asks": asks}

    # ------------------------------------------------------------------
    # Imbalance
    # ------------------------------------------------------------------

    @property
    def imbalance(self) -> Optional[float]:
        """
        Order-flow imbalance at the top-5 levels.

        ``imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)``

        Range [-1, 1].  Positive values indicate bid pressure.
        Returns None when either side is empty.
        """
        with self._lock:
            bid_qty = sum(
                q for _, q in sorted(self._bids.items(), reverse=True)[:5]
            )
            ask_qty = sum(
                q for _, q in sorted(self._asks.items())[:5]
            )
        total = bid_qty + ask_qty
        if total == 0:
            return None
        return (bid_qty - ask_qty) / total

    # ------------------------------------------------------------------
    # VWAP fill estimator
    # ------------------------------------------------------------------

    def vwap_to_fill(self, side: str, qty: float) -> float:
        """
        Walk the book and compute the volume-weighted average fill price
        for a given quantity.

        Parameters
        ----------
        side : str
            ``"buy"`` — walks the ask side (ascending price).
            ``"sell"`` — walks the bid side (descending price).
        qty : float
            Quantity to fill.

        Returns
        -------
        float
            Expected VWAP fill price.

        Raises
        ------
        InsufficientLiquidityError
            If available depth is smaller than ``qty``.
        ValueError
            If ``side`` is invalid or ``qty`` <= 0.
        """
        if qty <= 0:
            raise ValueError("qty must be positive")
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        with self._lock:
            if side == "buy":
                levels = sorted(self._asks.items())          # ascending ask
            else:
                levels = sorted(self._bids.items(), reverse=True)  # descending bid

        remaining = qty
        notional = 0.0
        for price, level_qty in levels:
            fill = min(remaining, level_qty)
            notional += fill * price
            remaining -= fill
            if remaining <= 1e-12:
                break

        if remaining > 1e-12:
            raise InsufficientLiquidityError(
                f"Book for {self.symbol} has insufficient {side} liquidity: "
                f"need {qty}, only {qty - remaining:.6f} available"
            )

        return notional / qty

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def bid_depth_usd(self, n: int = 5) -> float:
        """Total USD value of the top-N bid levels."""
        with self._lock:
            return sum(
                p * q for p, q in sorted(self._bids.items(), reverse=True)[:n]
            )

    def ask_depth_usd(self, n: int = 5) -> float:
        """Total USD value of the top-N ask levels."""
        with self._lock:
            return sum(
                p * q for p, q in sorted(self._asks.items())[:n]
            )

    def __repr__(self) -> str:
        bb = self.best_bid
        ba = self.best_ask
        sp = self.spread_bps
        return (
            f"<OrderBook {self.symbol} "
            f"bid={bb:.4f} ask={ba:.4f} spread={sp:.2f}bps>"
            if bb and ba and sp
            else f"<OrderBook {self.symbol} empty>"
        )
