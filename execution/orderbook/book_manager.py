"""
execution/orderbook/book_manager.py
=====================================
Manages both L2 feeds and provides a unified, thread-safe interface to the
rest of the execution layer.

Architecture
------------
- Starts the Alpaca feed immediately.
- Starts the Binance feed concurrently as a hot-standby.
- Every ``_HEALTH_CHECK_INTERVAL`` seconds a background coroutine checks
  whether the Alpaca feed has been silent for > 30 s.  If so, BookManager
  promotes Binance as the primary source and logs an alert.
- ``get_spread_bps``, ``get_mid``, ``is_liquid``, ``estimate_impact_bps``
  always use the currently active feed.

Market-impact model
-------------------
``estimate_impact_bps`` uses the square-root model:

    impact_bps = k * sqrt(notional / adv) * vol_bps * 1e4

where
- k = 0.5 (empirical constant)
- adv is derived from ``daily_volume_estimates`` (USD), defaulting to $5 M
- vol_bps is approximated as the current spread in bps (tight proxy for
  short-term vol in liquid crypto markets)
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Optional

from .alpaca_l2_feed import AlpacaL2Feed, SILENCE_TIMEOUT as ALPACA_SILENCE
from .binance_l2_feed import BinanceL2Feed
from .orderbook import OrderBook

log = logging.getLogger("execution.book_manager")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_HEALTH_CHECK_INTERVAL = 5.0    # seconds between feed health checks
_IMPACT_K = 0.5                  # sqrt-model constant
_DEFAULT_ADV_USD = 5_000_000.0   # fallback ADV if not provided
_MIN_DEPTH_USD_DEFAULT = 10_000.0


# ---------------------------------------------------------------------------
# BookManager
# ---------------------------------------------------------------------------

class BookManager:
    """
    Unified L2 book interface for the execution layer.

    Parameters
    ----------
    symbols : list[str]
        Symbols to track (Alpaca format, e.g. ``["BTC/USD", "ETH/USD"]``).
    alpaca_key : str | None
        Alpaca API key (env fallback: APCA_API_KEY_ID / ALPACA_KEY).
    alpaca_secret : str | None
        Alpaca API secret.
    daily_volume_estimates : dict[str, float] | None
        Symbol -> estimated daily volume in USD, used for impact model.
    """

    def __init__(
        self,
        symbols: list[str],
        alpaca_key: Optional[str] = None,
        alpaca_secret: Optional[str] = None,
        daily_volume_estimates: Optional[dict[str, float]] = None,
    ) -> None:
        self._symbols = list(symbols)
        self._adv = daily_volume_estimates or {}

        self._alpaca = AlpacaL2Feed(symbols, alpaca_key, alpaca_secret)
        self._binance = BinanceL2Feed(symbols)

        self._use_alpaca = True          # True = prefer Alpaca; False = prefer Binance
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start both feeds and the health-monitor background task."""
        if self._running:
            return
        self._running = True
        await self._alpaca.start()
        await self._binance.start()
        self._monitor_task = asyncio.create_task(
            self._health_monitor(), name="book_manager_health"
        )
        log.info("BookManager started (symbols=%s)", self._symbols)

    async def stop(self) -> None:
        """Stop all feeds."""
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        await self._alpaca.stop()
        await self._binance.stop()
        log.info("BookManager stopped.")

    # ------------------------------------------------------------------
    # Feed selection
    # ------------------------------------------------------------------

    async def _health_monitor(self) -> None:
        """Periodically check feed liveness and switch if Alpaca goes silent."""
        while self._running:
            await asyncio.sleep(_HEALTH_CHECK_INTERVAL)
            alpaca_silent = self._alpaca.is_silent
            if self._use_alpaca and alpaca_silent:
                log.warning(
                    "BookManager: Alpaca feed silent for >%.0fs — switching to Binance",
                    ALPACA_SILENCE,
                )
                self._use_alpaca = False
            elif not self._use_alpaca and not alpaca_silent:
                log.info("BookManager: Alpaca feed recovered — switching back from Binance")
                self._use_alpaca = True

    def _active_feed(self):
        """Return the currently preferred feed object."""
        if self._use_alpaca:
            return self._alpaca
        return self._binance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_book(self, symbol: str) -> Optional[OrderBook]:
        """Return the current OrderBook for ``symbol`` from the active feed."""
        return self._active_feed().get_book(symbol)

    def get_spread_bps(self, symbol: str) -> Optional[float]:
        """
        Return the current bid-ask spread in basis points.

        Falls back to the secondary feed if the primary returns None.
        """
        spread = self._active_feed().get_spread_bps(symbol)
        if spread is not None:
            return spread
        # Fallback to whichever feed we're NOT using
        fallback = self._binance if self._use_alpaca else self._alpaca
        return fallback.get_spread_bps(symbol)

    def get_mid(self, symbol: str) -> Optional[float]:
        """Return midprice for ``symbol``."""
        mid = self._active_feed().get_mid(symbol)
        if mid is not None:
            return mid
        fallback = self._binance if self._use_alpaca else self._alpaca
        return fallback.get_mid(symbol)

    def is_liquid(
        self,
        symbol: str,
        min_depth_usd: float = _MIN_DEPTH_USD_DEFAULT,
    ) -> bool:
        """
        Return True if at least ``min_depth_usd`` of liquidity exists on
        both the bid and ask side (top-5 levels).

        Returns False if the book is not available or too thin.
        """
        book = self.get_book(symbol)
        if book is None:
            log.debug("is_liquid(%s): no book available", symbol)
            return False
        bid_depth = book.bid_depth_usd(n=5)
        ask_depth = book.ask_depth_usd(n=5)
        ok = bid_depth >= min_depth_usd and ask_depth >= min_depth_usd
        if not ok:
            log.debug(
                "is_liquid(%s) False: bid_usd=%.0f ask_usd=%.0f threshold=%.0f",
                symbol, bid_depth, ask_depth, min_depth_usd,
            )
        return ok

    def estimate_impact_bps(self, symbol: str, notional_usd: float) -> float:
        """
        Estimate market-impact in basis points using the square-root model.

        ``impact_bps = k * sqrt(notional / adv) * vol_bps``

        where ``vol_bps`` is proxied by the current spread in bps.

        Parameters
        ----------
        symbol : str
        notional_usd : float
            Trade size in USD.

        Returns
        -------
        float
            Estimated impact in basis points.  Returns a large value (999 bps)
            if the book is not available.
        """
        adv = self._adv.get(symbol, _DEFAULT_ADV_USD)
        if adv <= 0:
            adv = _DEFAULT_ADV_USD

        spread = self.get_spread_bps(symbol)
        if spread is None or spread <= 0:
            log.warning(
                "estimate_impact_bps(%s): no spread available, returning 999 bps",
                symbol,
            )
            return 999.0

        vol_bps = spread  # tight proxy for short-term realised vol
        impact = _IMPACT_K * math.sqrt(notional_usd / adv) * vol_bps
        log.debug(
            "estimate_impact_bps(%s, %.0f): adv=%.0f spread=%.2fbps -> %.2fbps",
            symbol, notional_usd, adv, spread, impact,
        )
        return impact

    # ------------------------------------------------------------------
    # Convenience: bid/ask tuple (matches legacy spread_feed interface)
    # ------------------------------------------------------------------

    def get_bid_ask(self, symbol: str) -> Optional[tuple[float, float]]:
        """
        Return ``(bid, ask)`` tuple for ``symbol``, or None.

        Suitable for use as a ``spread_feed`` callable in SmartRouter:

            router = SmartRouter(
                broker=broker,
                spread_feed=lambda sym: book_manager.get_bid_ask(sym),
            )
        """
        book = self.get_book(symbol)
        if book is None:
            return None
        bb = book.best_bid
        ba = book.best_ask
        if bb is None or ba is None:
            return None
        return (bb, ba)

    @property
    def active_feed_name(self) -> str:
        """Name of the currently active feed (for logging/monitoring)."""
        return "alpaca" if self._use_alpaca else "binance"
