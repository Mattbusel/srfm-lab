"""
alpaca_adapter.py -- Alpaca broker adapter for SRFM execution layer.

Implements BrokerAdapter for Alpaca Markets REST API (paper and live).
Uses aiohttp for async HTTP and supports WebSocket streaming for fills
and quotes via Alpaca's Data/Trading stream endpoints.

Rate limit: 200 requests/minute enforced by AlpacaRateLimiter (token bucket).
All calls pass through a circuit breaker. Retries on 429; raises on 401/403.

Alpaca API docs: https://docs.alpaca.markets/
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import aiohttp  # HTTP client -- not instantiated at import time

from .base_adapter import (
    AccountInfo,
    AssetClass,
    AuthenticationError,
    BrokerAdapter,
    BrokerAdapterError,
    CircuitBreaker,
    Fill,
    InsufficientFundsError,
    OrderRejectedError,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    RateLimitError,
    TimeInForce,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_BASE_URL = "https://api.alpaca.markets"
ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"
ALPACA_STREAM_URL = "wss://stream.data.alpaca.markets/v2"
ALPACA_TRADING_STREAM_URL = "wss://paper-api.alpaca.markets/stream"
ALPACA_LIVE_TRADING_STREAM_URL = "wss://api.alpaca.markets/stream"

MAX_REQUESTS_PER_MINUTE = 200
RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_S = 0.5


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class AlpacaRateLimiter:
    """Token bucket rate limiter enforcing Alpaca's 200 req/min cap.

    Tokens refill continuously at rate = max_tokens / window_s per second.
    If no tokens are available, acquire() sleeps until one refills.

    Parameters
    ----------
    max_requests_per_minute -- bucket capacity and refill rate
    """

    def __init__(self, max_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE) -> None:
        self.max_tokens = float(max_requests_per_minute)
        self.refill_rate = self.max_tokens / 60.0  # tokens per second
        self._tokens = self.max_tokens
        self._last_refill_ts = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill_ts
        self._tokens = min(self.max_tokens, self._tokens + elapsed * self.refill_rate)
        self._last_refill_ts = now

    async def acquire(self) -> None:
        """Block until a token is available, then consume it."""
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Sleep until one token refills
                wait_s = (1.0 - self._tokens) / self.refill_rate
                await asyncio.sleep(wait_s)

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens


# ---------------------------------------------------------------------------
# Order translator
# ---------------------------------------------------------------------------


class AlpacaOrderTranslator:
    """Translates between SRFM data models and Alpaca API dictionaries.

    All methods are pure (no network calls). The translator normalises
    Alpaca's snake_case JSON keys and handles missing/null fields gracefully.
    """

    # Map SRFM OrderType -> Alpaca type string
    _ORDER_TYPE_MAP: Dict[OrderType, str] = {
        OrderType.MARKET: "market",
        OrderType.LIMIT: "limit",
        OrderType.STOP: "stop",
        OrderType.STOP_LIMIT: "stop_limit",
        OrderType.TRAILING_STOP: "trailing_stop",
    }

    # Map SRFM TimeInForce -> Alpaca tif string
    _TIF_MAP: Dict[TimeInForce, str] = {
        TimeInForce.DAY: "day",
        TimeInForce.GTC: "gtc",
        TimeInForce.IOC: "ioc",
        TimeInForce.FOK: "fok",
        TimeInForce.OPG: "opg",
        TimeInForce.CLS: "cls",
    }

    # Reverse map for Alpaca status -> SRFM OrderStatus
    _STATUS_MAP: Dict[str, OrderStatus] = {
        "new": OrderStatus.SUBMITTED,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "filled": OrderStatus.FILLED,
        "done_for_day": OrderStatus.CANCELED,
        "canceled": OrderStatus.CANCELED,
        "expired": OrderStatus.EXPIRED,
        "replaced": OrderStatus.CANCELED,
        "pending_cancel": OrderStatus.SUBMITTED,
        "pending_replace": OrderStatus.SUBMITTED,
        "held": OrderStatus.PENDING,
        "accepted": OrderStatus.SUBMITTED,
        "pending_new": OrderStatus.PENDING,
        "accepted_for_bidding": OrderStatus.SUBMITTED,
        "stopped": OrderStatus.CANCELED,
        "rejected": OrderStatus.REJECTED,
        "suspended": OrderStatus.REJECTED,
        "calculated": OrderStatus.SUBMITTED,
    }

    def to_alpaca_order(self, order: OrderRequest) -> Dict[str, Any]:
        """Convert an OrderRequest to an Alpaca API POST body.

        Parameters
        ----------
        order -- SRFM OrderRequest

        Returns
        -------
        Dict suitable for JSON serialisation and POST to /v2/orders
        """
        payload: Dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side.value,
            "type": self._ORDER_TYPE_MAP[order.order_type],
            "time_in_force": self._TIF_MAP[order.time_in_force],
            "client_order_id": order.client_order_id,
            "extended_hours": order.extended_hours,
        }

        if order.notional is not None:
            payload["notional"] = str(order.notional)
        else:
            payload["qty"] = str(order.qty)

        if order.price is not None:
            payload["limit_price"] = str(order.price)

        if order.stop_price is not None:
            payload["stop_price"] = str(order.stop_price)

        return payload

    def from_alpaca_order(self, response: Dict[str, Any]) -> OrderResult:
        """Parse an Alpaca order response into an OrderResult.

        Parameters
        ----------
        response -- raw dict from Alpaca /v2/orders endpoint

        Returns
        -------
        OrderResult
        """
        raw_status = response.get("status", "unknown")
        status = self._STATUS_MAP.get(raw_status, OrderStatus.UNKNOWN)

        submitted_at_str = response.get("submitted_at") or response.get("created_at", "")
        try:
            submitted_at = datetime.fromisoformat(submitted_at_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            submitted_at = datetime.now(timezone.utc)

        filled_qty_str = response.get("filled_qty", "0") or "0"
        avg_fill_str = response.get("filled_avg_price")

        return OrderResult(
            order_id=response.get("id", ""),
            client_order_id=response.get("client_order_id", ""),
            status=status,
            submitted_at=submitted_at,
            message=response.get("reason", ""),
            filled_qty=float(filled_qty_str),
            avg_fill_price=float(avg_fill_str) if avg_fill_str else None,
        )

    def from_alpaca_position(self, pos: Dict[str, Any]) -> Position:
        """Parse an Alpaca position dict into a Position.

        Parameters
        ----------
        pos -- raw dict from Alpaca /v2/positions endpoint
        """
        qty = float(pos.get("qty", 0) or 0)
        side_str = pos.get("side", "long")
        side = PositionSide.LONG if side_str == "long" else PositionSide.SHORT

        return Position(
            symbol=pos.get("symbol", ""),
            qty=abs(qty),
            avg_entry_price=float(pos.get("avg_entry_price", 0) or 0),
            market_value=float(pos.get("market_value", 0) or 0),
            unrealized_pnl=float(pos.get("unrealized_pl", 0) or 0),
            side=side,
            cost_basis=float(pos.get("cost_basis", 0) or 0),
            asset_class=AssetClass.EQUITY,
        )

    def from_alpaca_fill(self, fill: Dict[str, Any]) -> Fill:
        """Parse an Alpaca trade/fill event into a Fill.

        Alpaca sends fills via the trading stream as 'trade_updates' events.

        Parameters
        ----------
        fill -- raw dict from trading stream or /v2/orders/:id endpoint
        """
        ts_str = fill.get("timestamp") or fill.get("filled_at") or ""
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.now(timezone.utc)

        return Fill(
            fill_id=fill.get("id", fill.get("order_id", "")),
            order_id=fill.get("order_id", ""),
            symbol=fill.get("symbol", ""),
            side=OrderSide(fill.get("side", "buy")),
            qty=float(fill.get("filled_qty", fill.get("qty", 0)) or 0),
            price=float(fill.get("filled_avg_price", fill.get("price", 0)) or 0),
            timestamp=ts,
            venue=fill.get("exchange", "ALPACA"),
            commission=float(fill.get("commission", 0) or 0),
        )


# ---------------------------------------------------------------------------
# Alpaca adapter
# ---------------------------------------------------------------------------


class AlpacaAdapter(BrokerAdapter):
    """Broker adapter for Alpaca Markets.

    Supports paper and live trading. Uses aiohttp for REST calls and the
    Alpaca WebSocket for real-time fill and quote streaming.

    Parameters
    ----------
    api_key    -- Alpaca API key ID
    api_secret -- Alpaca API secret key
    base_url   -- override REST base URL (defaults to paper or live)
    paper      -- True for paper trading, False for live
    circuit_breaker -- optional pre-built circuit breaker
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: Optional[str] = None,
        paper: bool = True,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        super().__init__(
            name="alpaca_paper" if paper else "alpaca_live",
            asset_class=AssetClass.EQUITY,
            circuit_breaker=circuit_breaker,
        )
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.base_url = base_url or (ALPACA_PAPER_BASE_URL if paper else ALPACA_LIVE_BASE_URL)
        self._stream_url = (
            ALPACA_TRADING_STREAM_URL if paper else ALPACA_LIVE_TRADING_STREAM_URL
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = AlpacaRateLimiter()
        self._translator = AlpacaOrderTranslator()
        self._connected = False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._get_headers())
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._connected = False

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict] = None,
        params: Optional[Dict] = None,
        base_url: Optional[str] = None,
    ) -> Any:
        """Make a rate-limited, circuit-broken HTTP request.

        Retries on 429 (rate limit) up to RETRY_MAX_ATTEMPTS times with
        exponential backoff. Raises on 401/403. Logs 5xx errors.

        Parameters
        ----------
        method   -- HTTP method (GET, POST, DELETE, etc.)
        path     -- URL path, e.g. /v2/orders
        payload  -- JSON body for POST/PATCH
        params   -- query parameters
        base_url -- override base URL for data API calls

        Returns
        -------
        Parsed JSON response as dict or list.
        """
        self._check_circuit()
        await self._rate_limiter.acquire()

        url = (base_url or self.base_url) + path
        session = await self._ensure_session()

        last_exc: Optional[Exception] = None
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                async with session.request(
                    method,
                    url,
                    json=payload,
                    params=params,
                ) as resp:
                    if resp.status == 200 or resp.status == 201 or resp.status == 204:
                        self._on_success()
                        if resp.status == 204:
                            return {}
                        return await resp.json()

                    if resp.status == 401 or resp.status == 403:
                        body = await resp.text()
                        self._on_failure(AuthenticationError(body))
                        raise AuthenticationError(
                            f"Alpaca auth failed ({resp.status}): {body}"
                        )

                    if resp.status == 422:
                        body = await resp.text()
                        self._on_failure(OrderRejectedError(body))
                        raise OrderRejectedError(f"Alpaca rejected order: {body}")

                    if resp.status == 403 and "insufficient" in (await resp.text()).lower():
                        raise InsufficientFundsError("Insufficient buying power")

                    if resp.status == 429:
                        retry_after = float(resp.headers.get("Retry-After", RETRY_BASE_DELAY_S))
                        self.logger.warning(
                            "Alpaca rate limit hit, waiting %.1fs (attempt %d/%d)",
                            retry_after,
                            attempt + 1,
                            RETRY_MAX_ATTEMPTS,
                        )
                        await asyncio.sleep(retry_after)
                        last_exc = RateLimitError("rate limit", retry_after_s=retry_after)
                        continue

                    if resp.status >= 500:
                        body = await resp.text()
                        self.logger.error("Alpaca 5xx error %d: %s", resp.status, body)
                        delay = RETRY_BASE_DELAY_S * (2 ** attempt)
                        await asyncio.sleep(delay)
                        last_exc = BrokerAdapterError(f"Alpaca {resp.status}: {body}")
                        self._on_failure(last_exc)
                        continue

                    body = await resp.text()
                    raise BrokerAdapterError(f"Alpaca unexpected {resp.status}: {body}")

            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                self.logger.warning("Alpaca connection error (attempt %d): %s", attempt + 1, exc)
                self._on_failure(exc)
                last_exc = exc
                await asyncio.sleep(RETRY_BASE_DELAY_S * (2 ** attempt))

        raise last_exc or BrokerAdapterError("Alpaca request failed after retries")

    # ------------------------------------------------------------------
    # BrokerAdapter interface
    # ------------------------------------------------------------------

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        payload = self._translator.to_alpaca_order(order)
        self.logger.info(
            "Submitting order symbol=%s side=%s qty=%s type=%s",
            order.symbol,
            order.side.value,
            order.qty,
            order.order_type.value,
        )
        response = await self._request("POST", "/v2/orders", payload=payload)
        result = self._translator.from_alpaca_order(response)
        self.logger.info("Order submitted order_id=%s status=%s", result.order_id, result.status)
        return result

    async def cancel_order(self, order_id: str) -> bool:
        self.logger.info("Canceling order order_id=%s", order_id)
        try:
            await self._request("DELETE", f"/v2/orders/{order_id}")
            return True
        except BrokerAdapterError as exc:
            if "422" in str(exc) or "not cancelable" in str(exc).lower():
                self.logger.info("Order %s not cancelable: %s", order_id, exc)
                return False
            raise

    async def get_position(self, symbol: str) -> Optional[Position]:
        try:
            resp = await self._request("GET", f"/v2/positions/{symbol}")
            return self._translator.from_alpaca_position(resp)
        except BrokerAdapterError as exc:
            if "404" in str(exc) or "position does not exist" in str(exc).lower():
                return None
            raise

    async def get_all_positions(self) -> Dict[str, Position]:
        resp = await self._request("GET", "/v2/positions")
        positions: Dict[str, Position] = {}
        for raw in resp:
            pos = self._translator.from_alpaca_position(raw)
            positions[pos.symbol] = pos
        return positions

    async def get_account(self) -> AccountInfo:
        resp = await self._request("GET", "/v2/account")
        return AccountInfo(
            equity=float(resp.get("equity", 0) or 0),
            cash=float(resp.get("cash", 0) or 0),
            buying_power=float(resp.get("buying_power", 0) or 0),
            margin_used=float(resp.get("initial_margin", 0) or 0),
            leverage=float(resp.get("multiplier", 1) or 1),
            day_pnl=float(resp.get("equity_previous_close", 0) or 0),
            currency="USD",
        )

    async def get_order_status(self, order_id: str) -> OrderStatus:
        resp = await self._request("GET", f"/v2/orders/{order_id}")
        raw_status = resp.get("status", "unknown")
        return self._translator._STATUS_MAP.get(raw_status, OrderStatus.UNKNOWN)

    async def get_recent_fills(self, n: int = 100) -> List[Fill]:
        resp = await self._request(
            "GET",
            "/v2/orders",
            params={"status": "filled", "limit": str(min(n, 500)), "direction": "desc"},
        )
        fills: List[Fill] = []
        for raw in resp[:n]:
            fills.append(self._translator.from_alpaca_fill(raw))
        return fills

    def is_connected(self) -> bool:
        return self._connected and self._session is not None and not self._session.closed

    async def test_connection(self) -> bool:
        try:
            await self._request("GET", "/v2/clock")
            self._connected = True
            return True
        except Exception as exc:
            self.logger.warning("Alpaca connection test failed: %s", exc)
            self._connected = False
            return False

    # ------------------------------------------------------------------
    # WebSocket streaming
    # ------------------------------------------------------------------

    async def stream_fills(self, callback: Callable[[Fill], None]) -> None:
        """Stream real-time order fill events from Alpaca trading stream.

        Connects to the Alpaca trading WebSocket, authenticates, subscribes
        to trade_updates, and calls callback for every fill event received.

        Parameters
        ----------
        callback -- async or sync callable that receives Fill objects
        """
        session = await self._ensure_session()
        auth_msg = json.dumps({
            "action": "auth",
            "key": self.api_key,
            "secret": self.api_secret,
        })
        listen_msg = json.dumps({"action": "listen", "data": {"streams": ["trade_updates"]}})

        while True:
            try:
                async with session.ws_connect(self._stream_url) as ws:
                    self._connected = True
                    self.logger.info("Alpaca trading stream connected")

                    await ws.send_str(auth_msg)
                    await ws.send_str(listen_msg)

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get("stream") == "trade_updates":
                                event = data.get("data", {})
                                if event.get("event") in ("fill", "partial_fill"):
                                    order = event.get("order", {})
                                    fill = self._translator.from_alpaca_fill(order)
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(fill)
                                    else:
                                        callback(fill)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            self.logger.warning("Alpaca trading stream closed/error")
                            break

            except Exception as exc:
                self._connected = False
                self.logger.error("Alpaca fill stream error: %s -- reconnecting in 5s", exc)
                await asyncio.sleep(5.0)

    async def stream_quotes(
        self,
        symbols: List[str],
        callback: Callable[[Dict], None],
    ) -> None:
        """Stream real-time quotes from Alpaca Market Data stream.

        Parameters
        ----------
        symbols  -- list of SRFM symbols (e.g. ['AAPL', 'MSFT'])
        callback -- callable receiving raw quote dicts
        """
        session = await self._ensure_session()
        stream_url = f"{ALPACA_STREAM_URL}/iex"

        auth_msg = json.dumps({"action": "auth", "key": self.api_key, "secret": self.api_secret})
        sub_msg = json.dumps({"action": "subscribe", "quotes": symbols})

        while True:
            try:
                async with session.ws_connect(stream_url) as ws:
                    self.logger.info("Alpaca quote stream connected for %d symbols", len(symbols))
                    await ws.send_str(auth_msg)
                    await ws.send_str(sub_msg)

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            events = json.loads(msg.data)
                            if isinstance(events, list):
                                for event in events:
                                    if event.get("T") == "q":  # quote event
                                        if asyncio.iscoroutinefunction(callback):
                                            await callback(event)
                                        else:
                                            callback(event)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            self.logger.warning("Alpaca quote stream closed/error")
                            break

            except Exception as exc:
                self.logger.error("Alpaca quote stream error: %s -- reconnecting in 5s", exc)
                await asyncio.sleep(5.0)
