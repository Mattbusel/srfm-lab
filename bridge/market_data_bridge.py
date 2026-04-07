"""
bridge/market_data_bridge.py
=============================
Bridge between the Go market data service (:8780/ws/bars) and the Python
live trader (tools/live_trader_alpaca.py).

Architecture
------------
  MarketDataBridge
    - Connects to Go market data WebSocket (:8780/ws/bars)
    - Receives OHLCV bars for all instruments in real time
    - Maintains in-memory bar buffers: last 500 bars per (symbol, timeframe)
    - Thread-safe access via asyncio.Lock
    - Fallback: if Go service unreachable -> connects directly to Alpaca WebSocket
    - Feed health monitoring: no bar for 30+ minutes -> alert + fallback

  BarBroadcaster
    - Receives bars from MarketDataBridge
    - Fan-out to registered consumer queues (one asyncio.Queue per consumer)
    - Back-pressure: if consumer queue full -> warn + drop oldest item

Usage::

    import asyncio
    from bridge.market_data_bridge import MarketDataBridge, BarBroadcaster

    broadcaster = BarBroadcaster()
    bridge = MarketDataBridge(broadcaster=broadcaster)

    consumer_q = broadcaster.register_consumer("live_trader")
    asyncio.run(bridge.run())
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Optional

import aiohttp

log = logging.getLogger("bridge.market_data_bridge")

_REPO_ROOT = Path(__file__).parents[1]

# Primary Go market data service
_GO_WS_URL = "ws://localhost:8780/ws/bars"

# Alpaca WebSocket fallback (crypto + equity)
_ALPACA_CRYPTO_WS = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
_ALPACA_EQUITY_WS = "wss://stream.data.alpaca.markets/v2/iex"

_MAX_BARS_PER_BUFFER = 500
_FEED_STALE_SECS = 1800      # 30 minutes -- trigger fallback
_FEED_WARN_SECS = 600        # 10 minutes -- log warning
_CONSUMER_QUEUE_SIZE = 2000
_RECONNECT_BACKOFF_MAX = 120.0


# ---------------------------------------------------------------------------
# Bar dataclass
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    """Normalised OHLCV bar from any data source."""

    symbol: str
    timeframe: str       # "1m", "5m", "15m", "1h", "4h", "1d"
    timestamp: str       # ISO-8601 UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0
    trade_count: int = 0
    source: str = "go_market_data"

    def to_dict(self) -> dict:
        return {
            "symbol":      self.symbol,
            "timeframe":   self.timeframe,
            "timestamp":   self.timestamp,
            "open":        self.open,
            "high":        self.high,
            "low":         self.low,
            "close":       self.close,
            "volume":      self.volume,
            "vwap":        self.vwap,
            "trade_count": self.trade_count,
            "source":      self.source,
        }

    @classmethod
    def from_go_message(cls, msg: dict) -> "Bar":
        """Parse a bar message from the Go market data WebSocket."""
        return cls(
            symbol=str(msg.get("symbol", msg.get("S", ""))),
            timeframe=str(msg.get("timeframe", msg.get("tf", "1m"))),
            timestamp=str(msg.get("timestamp", msg.get("t", ""))),
            open=float(msg.get("open",  msg.get("o", 0.0))),
            high=float(msg.get("high",  msg.get("h", 0.0))),
            low=float(msg.get("low",   msg.get("l", 0.0))),
            close=float(msg.get("close", msg.get("c", 0.0))),
            volume=float(msg.get("volume", msg.get("v", 0.0))),
            vwap=float(msg.get("vwap", msg.get("vw", 0.0))),
            trade_count=int(msg.get("trade_count", msg.get("n", 0))),
            source="go_market_data",
        )

    @classmethod
    def from_alpaca_bar(cls, msg: dict, timeframe: str = "1m") -> "Bar":
        """Parse an Alpaca stream bar message."""
        return cls(
            symbol=str(msg.get("S", msg.get("symbol", ""))),
            timeframe=timeframe,
            timestamp=str(msg.get("t", msg.get("timestamp", ""))),
            open=float(msg.get("o", msg.get("open", 0.0))),
            high=float(msg.get("h", msg.get("high", 0.0))),
            low=float(msg.get("l", msg.get("low", 0.0))),
            close=float(msg.get("c", msg.get("close", 0.0))),
            volume=float(msg.get("v", msg.get("volume", 0.0))),
            vwap=float(msg.get("vw", msg.get("vwap", 0.0))),
            trade_count=int(msg.get("n", msg.get("trade_count", 0))),
            source="alpaca_fallback",
        )


# ---------------------------------------------------------------------------
# BarBuffer -- thread-safe in-memory OHLCV store
# ---------------------------------------------------------------------------

class BarBuffer:
    """
    Thread-safe in-memory buffer of the last N bars for a single
    (symbol, timeframe) combination.

    All methods are safe to call from any asyncio task or thread.
    """

    def __init__(self, maxlen: int = _MAX_BARS_PER_BUFFER) -> None:
        self._buf: Deque[Bar] = deque(maxlen=maxlen)
        self._lock = asyncio.Lock()

    async def append(self, bar: Bar) -> None:
        async with self._lock:
            self._buf.append(bar)

    async def get_latest(self) -> Bar | None:
        async with self._lock:
            return self._buf[-1] if self._buf else None

    async def get_history(self, n: int) -> list[Bar]:
        async with self._lock:
            data = list(self._buf)
        return data[-n:] if n < len(data) else data

    async def __len_async__(self) -> int:
        async with self._lock:
            return len(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# FeedHealthMonitor
# ---------------------------------------------------------------------------

class FeedHealthMonitor:
    """
    Tracks last-seen timestamps per (symbol, timeframe) and flags feeds
    that have gone stale.
    """

    def __init__(
        self,
        warn_threshold_secs: float = _FEED_WARN_SECS,
        stale_threshold_secs: float = _FEED_STALE_SECS,
    ) -> None:
        self._warn_secs = warn_threshold_secs
        self._stale_secs = stale_threshold_secs
        self._last_seen: dict[tuple[str, str], float] = {}
        self._lock = asyncio.Lock()

    async def record_bar(self, symbol: str, timeframe: str) -> None:
        key = (symbol, timeframe)
        async with self._lock:
            self._last_seen[key] = time.monotonic()

    async def check_staleness(self) -> list[tuple[str, str, float]]:
        """
        Return list of (symbol, timeframe, elapsed_secs) for stale feeds.
        A feed is stale if elapsed > stale_threshold_secs.
        """
        now = time.monotonic()
        stale: list[tuple[str, str, float]] = []
        async with self._lock:
            for (sym, tf), last in self._last_seen.items():
                elapsed = now - last
                if elapsed > self._stale_secs:
                    stale.append((sym, tf, elapsed))
        return stale

    async def check_warnings(self) -> list[tuple[str, str, float]]:
        """Return (symbol, timeframe, elapsed_secs) for warned but not yet stale feeds."""
        now = time.monotonic()
        warned: list[tuple[str, str, float]] = []
        async with self._lock:
            for (sym, tf), last in self._last_seen.items():
                elapsed = now - last
                if self._warn_secs < elapsed <= self._stale_secs:
                    warned.append((sym, tf, elapsed))
        return warned

    async def get_all_last_seen(self) -> dict[tuple[str, str], float]:
        async with self._lock:
            return dict(self._last_seen)


# ---------------------------------------------------------------------------
# BarBroadcaster
# ---------------------------------------------------------------------------

class BarBroadcaster:
    """
    Fan-out broadcaster: receives bars from MarketDataBridge and distributes
    them to all registered consumers via per-consumer asyncio.Queue.

    Back-pressure policy
    --------------------
    If a consumer's queue is full (size >= _CONSUMER_QUEUE_SIZE), the oldest
    item is dropped and a warning is logged.  This prevents a slow consumer
    from blocking the real-time feed.
    """

    def __init__(self, queue_size: int = _CONSUMER_QUEUE_SIZE) -> None:
        self._queue_size = queue_size
        self._consumers: dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
        self._drop_counts: dict[str, int] = {}

    async def register_consumer(self, name: str) -> asyncio.Queue:
        """Register a named consumer. Returns its dedicated asyncio.Queue."""
        async with self._lock:
            if name in self._consumers:
                log.warning("BarBroadcaster: consumer '%s' already registered -- returning existing queue", name)
                return self._consumers[name]
            q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
            self._consumers[name] = q
            self._drop_counts[name] = 0
            log.info("BarBroadcaster: registered consumer '%s'", name)
            return q

    def register_consumer_sync(self, name: str) -> asyncio.Queue:
        """
        Synchronous version -- must be called from the same event loop thread
        before the event loop is running, or use register_consumer() instead.
        """
        if name in self._consumers:
            return self._consumers[name]
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._consumers[name] = q
        self._drop_counts[name] = 0
        log.info("BarBroadcaster: registered consumer '%s' (sync)", name)
        return q

    async def unregister_consumer(self, name: str) -> None:
        async with self._lock:
            self._consumers.pop(name, None)
            self._drop_counts.pop(name, None)
            log.info("BarBroadcaster: unregistered consumer '%s'", name)

    async def broadcast(self, bar: Bar) -> None:
        """
        Deliver a bar to every registered consumer.

        If a consumer's queue is full, drop its oldest item and enqueue
        the new bar.  Log a warning at most once per 100 drops per consumer.
        """
        async with self._lock:
            consumers = dict(self._consumers)

        for name, q in consumers.items():
            if q.full():
                try:
                    q.get_nowait()  # drop oldest
                except asyncio.QueueEmpty:
                    pass
                self._drop_counts[name] = self._drop_counts.get(name, 0) + 1
                drops = self._drop_counts[name]
                if drops % 100 == 1:
                    log.warning(
                        "BarBroadcaster: consumer '%s' queue full -- dropped %d bar(s)",
                        name,
                        drops,
                    )
            try:
                q.put_nowait(bar)
            except asyncio.QueueFull:
                pass  # already dropped oldest above; should not reach here

    async def get_consumer_names(self) -> list[str]:
        async with self._lock:
            return list(self._consumers.keys())

    def get_drop_counts(self) -> dict[str, int]:
        return dict(self._drop_counts)


# ---------------------------------------------------------------------------
# AlpacaFallbackClient
# ---------------------------------------------------------------------------

class AlpacaFallbackClient:
    """
    Minimal Alpaca WebSocket client used as fallback when the Go market data
    service is unavailable.

    Connects to both crypto and equity WebSocket streams.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.
    """

    def __init__(
        self,
        on_bar: Callable[[Bar], None],
        crypto_symbols: list[str] | None = None,
        equity_symbols: list[str] | None = None,
    ) -> None:
        import os
        self._api_key = os.environ.get("ALPACA_API_KEY", "")
        self._api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
        self._on_bar = on_bar
        self._crypto_syms = crypto_symbols or ["BTC/USD", "ETH/USD"]
        self._equity_syms = equity_symbols or ["SPY", "QQQ"]
        self._running = False

    async def run(self) -> None:
        self._running = True
        tasks = [
            asyncio.create_task(self._run_stream(_ALPACA_CRYPTO_WS, self._crypto_syms, "crypto")),
            asyncio.create_task(self._run_stream(_ALPACA_EQUITY_WS, self._equity_syms, "equity")),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    async def _run_stream(self, url: str, symbols: list[str], asset_class: str) -> None:
        backoff = 5.0
        while self._running:
            try:
                await self._connect_and_stream(url, symbols, asset_class)
                backoff = 5.0
            except Exception as exc:
                log.warning(
                    "AlpacaFallbackClient [%s]: stream error: %s -- reconnecting in %.0fs",
                    asset_class,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX)

    async def _connect_and_stream(
        self, url: str, symbols: list[str], asset_class: str
    ) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url, heartbeat=30) as ws:
                # Auth
                await ws.send_json({"action": "auth", "key": self._api_key, "secret": self._api_secret})
                auth_msg = await ws.receive_json()
                log.debug("AlpacaFallbackClient [%s]: auth response: %s", asset_class, auth_msg)

                # Subscribe to bars
                await ws.send_json({"action": "subscribe", "bars": symbols})
                sub_msg = await ws.receive_json()
                log.info(
                    "AlpacaFallbackClient [%s]: subscribed to %d symbols",
                    asset_class,
                    len(symbols),
                )

                async for msg in ws:
                    if not self._running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if isinstance(data, list):
                            for item in data:
                                if item.get("T") == "b":  # bar message
                                    bar = Bar.from_alpaca_bar(item, timeframe="1m")
                                    self._on_bar(bar)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        log.warning("AlpacaFallbackClient [%s]: WebSocket closed", asset_class)
                        break


# ---------------------------------------------------------------------------
# MarketDataBridge
# ---------------------------------------------------------------------------

class MarketDataBridge:
    """
    Central bar ingestion hub.

    Connects to the Go market data WebSocket as primary source.
    If that service is down or stale, falls back to direct Alpaca WebSocket.

    All bar data is stored in BarBuffer instances and broadcast via BarBroadcaster.
    """

    def __init__(
        self,
        go_ws_url: str = _GO_WS_URL,
        broadcaster: BarBroadcaster | None = None,
        max_bars: int = _MAX_BARS_PER_BUFFER,
        stale_threshold_secs: float = _FEED_STALE_SECS,
        warn_threshold_secs: float = _FEED_WARN_SECS,
        crypto_symbols: list[str] | None = None,
        equity_symbols: list[str] | None = None,
    ) -> None:
        self._go_ws_url = go_ws_url
        self._broadcaster = broadcaster or BarBroadcaster()
        self._max_bars = max_bars
        self._buffers: dict[tuple[str, str], BarBuffer] = {}
        self._buffers_lock = asyncio.Lock()
        self._health = FeedHealthMonitor(warn_threshold_secs, stale_threshold_secs)
        self._stale_secs = stale_threshold_secs
        self._crypto_symbols = crypto_symbols
        self._equity_symbols = equity_symbols
        self._running = False
        self._using_fallback = False
        self._fallback_client: AlpacaFallbackClient | None = None
        self._total_bars_received: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_latest_bar(self, symbol: str, timeframe: str) -> Bar | None:
        """Return the most recent bar for (symbol, timeframe), or None."""
        buf = await self._get_buffer(symbol, timeframe)
        return await buf.get_latest()

    async def get_bar_history(self, symbol: str, timeframe: str, n: int = 100) -> list[Bar]:
        """Return up to n most recent bars for (symbol, timeframe), oldest first."""
        buf = await self._get_buffer(symbol, timeframe)
        return await buf.get_history(n)

    async def get_latest_bar_dict(self, symbol: str, timeframe: str) -> dict | None:
        """Convenience: return latest bar as dict, or None."""
        bar = await self.get_latest_bar(symbol, timeframe)
        return bar.to_dict() if bar else None

    def is_using_fallback(self) -> bool:
        return self._using_fallback

    def get_total_bars_received(self) -> int:
        return self._total_bars_received

    def get_broadcaster(self) -> BarBroadcaster:
        return self._broadcaster

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run bridge -- primary Go WS with automatic Alpaca fallback."""
        self._running = True
        log.info("MarketDataBridge: starting -- primary source: %s", self._go_ws_url)

        tasks = [
            asyncio.create_task(self._go_ws_loop(), name="go_ws"),
            asyncio.create_task(self._health_monitor_loop(), name="health_monitor"),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._running = False
            if self._fallback_client:
                self._fallback_client.stop()
            log.info("MarketDataBridge: stopped. Total bars received: %d", self._total_bars_received)

    def stop(self) -> None:
        self._running = False
        if self._fallback_client:
            self._fallback_client.stop()

    # ------------------------------------------------------------------
    # Go WebSocket loop
    # ------------------------------------------------------------------

    async def _go_ws_loop(self) -> None:
        backoff = 5.0
        while self._running:
            try:
                await self._connect_go_ws()
                backoff = 5.0
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.warning(
                    "MarketDataBridge: Go WS error: %s -- reconnecting in %.0fs",
                    exc,
                    backoff,
                )
                if not self._using_fallback:
                    log.warning("MarketDataBridge: switching to Alpaca fallback")
                    asyncio.create_task(self._run_fallback(), name="alpaca_fallback")
                    self._using_fallback = True
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX)

    async def _connect_go_ws(self) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self._go_ws_url,
                heartbeat=30,
                timeout=aiohttp.ClientWSTimeout(ws_close=10),
            ) as ws:
                if self._using_fallback:
                    log.info("MarketDataBridge: Go WS reconnected -- disabling fallback")
                    self._using_fallback = False
                    if self._fallback_client:
                        self._fallback_client.stop()

                log.info("MarketDataBridge: connected to Go WS %s", self._go_ws_url)

                async for msg in ws:
                    if not self._running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_go_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        await self._handle_go_message(msg.data.decode("utf-8", errors="replace"))
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        log.warning("MarketDataBridge: Go WS closed")
                        break

    async def _handle_go_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.debug("MarketDataBridge: JSON parse error on: %s", raw[:100])
            return

        # Go service may send a list of bars or a single bar object
        items: list[dict] = data if isinstance(data, list) else [data]
        for item in items:
            msg_type = item.get("type", item.get("T", "bar"))
            if msg_type in ("bar", "b", "ohlcv"):
                bar = Bar.from_go_message(item)
                await self._ingest_bar(bar)

    # ------------------------------------------------------------------
    # Alpaca fallback
    # ------------------------------------------------------------------

    async def _run_fallback(self) -> None:
        self._fallback_client = AlpacaFallbackClient(
            on_bar=self._on_fallback_bar,
            crypto_symbols=self._crypto_symbols,
            equity_symbols=self._equity_symbols,
        )
        log.info("MarketDataBridge: Alpaca fallback client starting")
        try:
            await self._fallback_client.run()
        except Exception as exc:
            log.error("MarketDataBridge: Alpaca fallback crashed: %s", exc)

    def _on_fallback_bar(self, bar: Bar) -> None:
        """Callback from AlpacaFallbackClient -- schedule bar ingestion."""
        asyncio.create_task(self._ingest_bar(bar))

    # ------------------------------------------------------------------
    # Bar ingestion
    # ------------------------------------------------------------------

    async def _ingest_bar(self, bar: Bar) -> None:
        """Store bar in buffer, update health, broadcast to consumers."""
        buf = await self._get_buffer(bar.symbol, bar.timeframe)
        await buf.append(bar)
        await self._health.record_bar(bar.symbol, bar.timeframe)
        await self._broadcaster.broadcast(bar)
        self._total_bars_received += 1
        log.debug(
            "MarketDataBridge: %s [%s] c=%.4f v=%.0f src=%s",
            bar.symbol,
            bar.timeframe,
            bar.close,
            bar.volume,
            bar.source,
        )

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    async def _get_buffer(self, symbol: str, timeframe: str) -> BarBuffer:
        key = (symbol, timeframe)
        async with self._buffers_lock:
            if key not in self._buffers:
                self._buffers[key] = BarBuffer(maxlen=self._max_bars)
            return self._buffers[key]

    async def get_tracked_symbols(self) -> list[tuple[str, str]]:
        """Return list of (symbol, timeframe) pairs currently buffered."""
        async with self._buffers_lock:
            return list(self._buffers.keys())

    # ------------------------------------------------------------------
    # Health monitoring loop
    # ------------------------------------------------------------------

    async def _health_monitor_loop(self) -> None:
        while self._running:
            await asyncio.sleep(60.0)
            await self._check_feed_health()

    async def _check_feed_health(self) -> None:
        warnings = await self._health.check_warnings()
        for sym, tf, elapsed in warnings:
            log.warning(
                "MarketDataBridge: feed warning -- %s [%s] no bar for %.0fs",
                sym,
                tf,
                elapsed,
            )

        stale = await self._health.check_staleness()
        for sym, tf, elapsed in stale:
            log.error(
                "MarketDataBridge: feed STALE -- %s [%s] no bar for %.0fm -- triggering fallback",
                sym,
                tf,
                elapsed / 60,
            )
            if not self._using_fallback:
                log.warning("MarketDataBridge: activating Alpaca fallback due to stale feed")
                asyncio.create_task(self._run_fallback(), name="alpaca_fallback_health")
                self._using_fallback = True
            break  # one alert is enough per cycle


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s UTC [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    broadcaster = BarBroadcaster()
    bridge = MarketDataBridge(broadcaster=broadcaster)
    consumer_q = broadcaster.register_consumer_sync("cli_consumer")

    async def consume() -> None:
        while True:
            bar: Bar = await consumer_q.get()
            log.info("CONSUMER: %s [%s] close=%.4f", bar.symbol, bar.timeframe, bar.close)

    try:
        await asyncio.gather(bridge.run(), consume())
    except KeyboardInterrupt:
        bridge.stop()


if __name__ == "__main__":
    asyncio.run(_main())
