"""
execution/orderbook/alpaca_l2_feed.py
======================================
Alpaca crypto WebSocket L2 feed.

Connects to:
    wss://stream.data.alpaca.markets/v1beta3/crypto/us

Message types handled
----------------------
- ``"o"``  — full orderbook snapshot (replaces the entire book)
- ``"q"``  — quote update (best bid/ask only; applied as single-level updates)

Authentication uses the APCA_API_KEY_ID / APCA_API_SECRET_KEY environment
variables, falling back to ALPACA_KEY / ALPACA_SECRET.

Reconnect
---------
Exponential backoff starting at 1 s, capped at 60 s, with ±10 % jitter.
The feed is considered "silent" if no message is received for ``SILENCE_TIMEOUT``
seconds; callers can check ``last_message_ts`` to detect this.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
from typing import Optional

try:
    import websockets
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
except ImportError as _e:
    raise ImportError(
        "websockets library is required: pip install websockets"
    ) from _e

from .orderbook import OrderBook

log = logging.getLogger("execution.alpaca_l2_feed")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WS_URL = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
SILENCE_TIMEOUT = 30.0          # seconds — used by BookManager to detect stale feed
MIN_BACKOFF = 1.0
MAX_BACKOFF = 60.0
PING_INTERVAL = 20              # websockets library heartbeat


# ---------------------------------------------------------------------------
# AlpacaL2Feed
# ---------------------------------------------------------------------------

class AlpacaL2Feed:
    """
    Maintains a live L2 orderbook for each subscribed crypto symbol via the
    Alpaca market-data WebSocket.

    Parameters
    ----------
    symbols : list[str]
        Symbols in Alpaca format, e.g. ``["BTC/USD", "ETH/USD"]``.
    api_key : str | None
        Alpaca API key.  Falls back to env vars if None.
    api_secret : str | None
        Alpaca API secret.  Falls back to env vars if None.
    """

    def __init__(
        self,
        symbols: list[str],
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        self._symbols = list(symbols)
        self._api_key = (
            api_key
            or os.getenv("APCA_API_KEY_ID")
            or os.getenv("ALPACA_KEY", "")
        )
        self._api_secret = (
            api_secret
            or os.getenv("APCA_API_SECRET_KEY")
            or os.getenv("ALPACA_SECRET", "")
        )
        self._books: dict[str, OrderBook] = {s: OrderBook(s) for s in symbols}
        self._last_message_ts: float = 0.0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_book(self, symbol: str) -> Optional[OrderBook]:
        """Return the current OrderBook for ``symbol``, or None if unknown."""
        return self._books.get(symbol)

    def get_spread_bps(self, symbol: str) -> Optional[float]:
        """Return spread in basis points for ``symbol``, or None."""
        book = self._books.get(symbol)
        return book.spread_bps if book else None

    def get_mid(self, symbol: str) -> Optional[float]:
        """Return midprice for ``symbol``, or None."""
        book = self._books.get(symbol)
        return book.mid_price if book else None

    @property
    def last_message_ts(self) -> float:
        """Unix timestamp of the last received message."""
        return self._last_message_ts

    @property
    def is_silent(self) -> bool:
        """True if no message received in the last SILENCE_TIMEOUT seconds."""
        if self._last_message_ts == 0:
            return True
        return (time.time() - self._last_message_ts) > SILENCE_TIMEOUT

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the feed as a background asyncio task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_forever(), name="alpaca_l2_feed")
        log.info("AlpacaL2Feed started for symbols: %s", self._symbols)

    async def stop(self) -> None:
        """Gracefully stop the feed."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("AlpacaL2Feed stopped.")

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    async def _run_forever(self) -> None:
        backoff = MIN_BACKOFF
        while self._running:
            try:
                await self._connect_and_consume()
                backoff = MIN_BACKOFF          # reset on clean exit
            except asyncio.CancelledError:
                break
            except Exception as exc:
                jitter = random.uniform(0.9, 1.1)
                wait = backoff * jitter
                log.warning(
                    "AlpacaL2Feed connection error (%s) — reconnecting in %.1fs",
                    exc, wait,
                )
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, MAX_BACKOFF)

    async def _connect_and_consume(self) -> None:
        log.info("AlpacaL2Feed connecting to %s", WS_URL)
        async with websockets.connect(
            WS_URL,
            ping_interval=PING_INTERVAL,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            # ── Authenticate ────────────────────────────────────────
            await self._authenticate(ws)
            # ── Subscribe ───────────────────────────────────────────
            await self._subscribe(ws)
            # ── Consume messages ────────────────────────────────────
            async for raw in ws:
                if not self._running:
                    break
                self._last_message_ts = time.time()
                try:
                    msgs = json.loads(raw)
                    if not isinstance(msgs, list):
                        msgs = [msgs]
                    for msg in msgs:
                        self._dispatch(msg)
                except Exception as exc:
                    log.debug("AlpacaL2Feed parse error: %s | raw=%r", exc, raw[:200])

    async def _authenticate(self, ws) -> None:
        auth = json.dumps({
            "action": "auth",
            "key": self._api_key,
            "secret": self._api_secret,
        })
        await ws.send(auth)
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        msgs = json.loads(raw)
        for msg in (msgs if isinstance(msgs, list) else [msgs]):
            if msg.get("T") == "success" and msg.get("msg") == "authenticated":
                log.info("AlpacaL2Feed authenticated")
                return
            if msg.get("T") == "error":
                raise ConnectionError(
                    f"Alpaca auth error: {msg.get('msg', msg)}"
                )
        log.warning("AlpacaL2Feed: unexpected auth response: %s", msgs)

    async def _subscribe(self, ws) -> None:
        sub = json.dumps({
            "action": "subscribe",
            "orderbooks": self._symbols,
            "quotes": self._symbols,
        })
        await ws.send(sub)
        log.info("AlpacaL2Feed subscribed: %s", self._symbols)

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, msg: dict) -> None:
        msg_type = msg.get("T", "")
        symbol = msg.get("S", "")

        if msg_type == "o":
            self._handle_snapshot(symbol, msg)
        elif msg_type == "q":
            self._handle_quote(symbol, msg)
        elif msg_type in ("subscription", "success", "error"):
            log.debug("AlpacaL2Feed control msg: %s", msg)
        # else silently ignore unknown message types

    def _handle_snapshot(self, symbol: str, msg: dict) -> None:
        """Full orderbook snapshot — replaces current book."""
        book = self._books.get(symbol)
        if book is None:
            book = OrderBook(symbol)
            self._books[symbol] = book

        def _parse_levels(raw: list) -> list[tuple[float, float]]:
            out = []
            for item in raw:
                if isinstance(item, dict):
                    out.append((float(item["p"]), float(item["s"])))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append((float(item[0]), float(item[1])))
            return out

        bids = _parse_levels(msg.get("b", []) or msg.get("bids", []))
        asks = _parse_levels(msg.get("a", []) or msg.get("asks", []))
        book.apply_snapshot(bids, asks)
        log.debug("AlpacaL2Feed snapshot %s: %d bids, %d asks", symbol, len(bids), len(asks))

    def _handle_quote(self, symbol: str, msg: dict) -> None:
        """Quote update — updates best bid and ask levels."""
        book = self._books.get(symbol)
        if book is None:
            book = OrderBook(symbol)
            self._books[symbol] = book

        bp = msg.get("bp") or msg.get("bid_price")
        bs = msg.get("bs") or msg.get("bid_size")
        ap = msg.get("ap") or msg.get("ask_price")
        as_ = msg.get("as") or msg.get("ask_size")

        if bp is not None and bs is not None:
            book.update("bid", float(bp), float(bs))
        if ap is not None and as_ is not None:
            book.update("ask", float(ap), float(as_))
