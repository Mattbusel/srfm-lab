"""
execution/orderbook/binance_l2_feed.py
========================================
Binance WebSocket depth feed — fallback when Alpaca is unavailable.

Subscribes to:
    wss://stream.binance.com:9443/ws/<symbol>@depth10@100ms

Each message is a partial book (top-10 levels per side) that replaces the
current book state for that symbol.  We do **not** use the incremental diff
stream here — the 100 ms partial book is simpler and sufficient for spread /
liquidity checks.

Symbol mapping
--------------
Alpaca-style symbols (``"BTC/USD"``, ``"ETH/USD"``) are converted to Binance
perpetual spot symbols (``"BTCUSDT"``, ``"ETHUSDT"``).  Add entries to
``SYMBOL_MAP`` below for any exotic pairs.

Reconnect
---------
Same exponential-backoff strategy as AlpacaL2Feed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Optional

try:
    import websockets
except ImportError as _e:
    raise ImportError(
        "websockets library is required: pip install websockets"
    ) from _e

from .orderbook import OrderBook

log = logging.getLogger("execution.binance_l2_feed")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WS_BASE = "wss://stream.binance.com:9443/ws"
PING_INTERVAL = 20
MIN_BACKOFF = 1.0
MAX_BACKOFF = 60.0
SILENCE_TIMEOUT = 30.0

# Alpaca symbol -> Binance spot symbol
SYMBOL_MAP: dict[str, str] = {
    "BTC/USD":  "BTCUSDT",
    "ETH/USD":  "ETHUSDT",
    "SOL/USD":  "SOLUSDT",
    "AVAX/USD": "AVAXUSDT",
    "LINK/USD": "LINKUSDT",
    "LTC/USD":  "LTCUSDT",
    "BCH/USD":  "BCHUSDT",
    "DOGE/USD": "DOGEUSDT",
    "MATIC/USD":"MATICUSDT",
    "UNI/USD":  "UNIUSDT",
    "AAVE/USD": "AAVEUSDT",
    "XRP/USD":  "XRPUSDT",
    "ADA/USD":  "ADAUSDT",
    "DOT/USD":  "DOTUSDT",
    "ATOM/USD": "ATOMUSDT",
}


def _to_binance(symbol: str) -> str:
    """Convert Alpaca symbol to Binance symbol (best-effort)."""
    if symbol in SYMBOL_MAP:
        return SYMBOL_MAP[symbol]
    # Generic: strip '/' and append USDT
    base = symbol.split("/")[0]
    return base + "USDT"


# ---------------------------------------------------------------------------
# BinanceL2Feed
# ---------------------------------------------------------------------------

class BinanceL2Feed:
    """
    Maintains a live top-10 L2 orderbook for each subscribed symbol via the
    Binance public WebSocket stream.

    Parameters
    ----------
    symbols : list[str]
        Symbols in Alpaca format (``"BTC/USD"`` etc.).
    """

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = list(symbols)
        # Map Alpaca symbol -> OrderBook
        self._books: dict[str, OrderBook] = {s: OrderBook(s) for s in symbols}
        # Map Binance stream key -> Alpaca symbol
        self._stream_to_alpaca: dict[str, str] = {
            _to_binance(s).lower(): s for s in symbols
        }
        self._last_message_ts: float = 0.0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public accessors (mirror AlpacaL2Feed interface)
    # ------------------------------------------------------------------

    def get_book(self, symbol: str) -> Optional[OrderBook]:
        return self._books.get(symbol)

    def get_spread_bps(self, symbol: str) -> Optional[float]:
        book = self._books.get(symbol)
        return book.spread_bps if book else None

    def get_mid(self, symbol: str) -> Optional[float]:
        book = self._books.get(symbol)
        return book.mid_price if book else None

    @property
    def last_message_ts(self) -> float:
        return self._last_message_ts

    @property
    def is_silent(self) -> bool:
        if self._last_message_ts == 0:
            return True
        return (time.time() - self._last_message_ts) > SILENCE_TIMEOUT

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_forever(), name="binance_l2_feed")
        log.info("BinanceL2Feed started for symbols: %s", self._symbols)

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("BinanceL2Feed stopped.")

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    async def _run_forever(self) -> None:
        backoff = MIN_BACKOFF
        while self._running:
            try:
                await self._connect_and_consume()
                backoff = MIN_BACKOFF
            except asyncio.CancelledError:
                break
            except Exception as exc:
                jitter = random.uniform(0.9, 1.1)
                wait = backoff * jitter
                log.warning(
                    "BinanceL2Feed connection error (%s) — reconnecting in %.1fs",
                    exc, wait,
                )
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, MAX_BACKOFF)

    async def _connect_and_consume(self) -> None:
        # Binance combined stream supports multiple symbols in one connection
        streams = "/".join(
            f"{_to_binance(s).lower()}@depth10@100ms" for s in self._symbols
        )
        url = f"{WS_BASE}/{streams}" if len(self._symbols) == 1 else \
              f"wss://stream.binance.com:9443/stream?streams={streams}"

        log.info("BinanceL2Feed connecting to %s", url)
        async with websockets.connect(
            url,
            ping_interval=PING_INTERVAL,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            async for raw in ws:
                if not self._running:
                    break
                self._last_message_ts = time.time()
                try:
                    msg = json.loads(raw)
                    # Combined stream wraps in {"stream": "...", "data": {...}}
                    if "stream" in msg:
                        stream_name = msg["stream"].split("@")[0]  # e.g. "btcusdt"
                        alpaca_sym = self._stream_to_alpaca.get(stream_name)
                        data = msg.get("data", {})
                    else:
                        # Single-stream: derive symbol from URL
                        alpaca_sym = self._symbols[0] if len(self._symbols) == 1 else None
                        data = msg

                    if alpaca_sym:
                        self._apply_depth(alpaca_sym, data)
                except Exception as exc:
                    log.debug("BinanceL2Feed parse error: %s | raw=%r", exc, raw[:200])

    # ------------------------------------------------------------------
    # Book application
    # ------------------------------------------------------------------

    def _apply_depth(self, alpaca_sym: str, data: dict) -> None:
        """Apply a Binance depth10 message to the corresponding OrderBook."""
        book = self._books.get(alpaca_sym)
        if book is None:
            book = OrderBook(alpaca_sym)
            self._books[alpaca_sym] = book

        def _parse(levels: list) -> list[tuple[float, float]]:
            return [(float(p), float(q)) for p, q in levels]

        bids = _parse(data.get("bids", []))
        asks = _parse(data.get("asks", []))
        book.apply_snapshot(bids, asks)
        log.debug(
            "BinanceL2Feed depth %s: %d bids, %d asks | spread=%.2fbps",
            alpaca_sym, len(bids), len(asks),
            book.spread_bps or 0,
        )
