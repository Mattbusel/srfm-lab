"""
binance_adapter.py -- Binance broker adapter for SRFM execution layer.

Implements BrokerAdapter for Binance Spot and USD-M Futures APIs.
Handles HMAC-SHA256 request signing, lot size / tick size precision,
USDT-denominated positions, and perpetual funding rates.

Testnet endpoints are used by default (testnet=True). Switch to live
by passing testnet=False and valid production credentials.

Binance API docs:
  Spot:    https://binance-docs.github.io/apidocs/spot/en/
  Futures: https://binance-docs.github.io/apidocs/futures/en/
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp  # async HTTP -- not instantiated at import time

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

BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_SPOT_TESTNET_BASE = "https://testnet.binance.vision"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BINANCE_FUTURES_TESTNET_BASE = "https://testnet.binancefuture.com"

BINANCE_SPOT_WS = "wss://stream.binance.com:9443/ws"
BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"

RECV_WINDOW_MS = 5000
RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_S = 0.5


# ---------------------------------------------------------------------------
# Signature generator
# ---------------------------------------------------------------------------


class BinanceSignatureGenerator:
    """Generates HMAC-SHA256 signatures for Binance signed endpoints.

    Binance requires all signed requests to include:
    - timestamp: server-synced millisecond timestamp
    - recvWindow: allowed clock drift in ms
    - signature: HMAC-SHA256 of the full query string with the API secret

    Server time sync is performed lazily on first use and refreshed every
    5 minutes to avoid clock drift rejections.

    Parameters
    ----------
    api_secret       -- Binance API secret key
    recv_window_ms   -- maximum allowed timestamp deviation
    sync_interval_s  -- how often to resync server time
    """

    def __init__(
        self,
        api_secret: str,
        recv_window_ms: int = RECV_WINDOW_MS,
        sync_interval_s: float = 300.0,
    ) -> None:
        self.api_secret = api_secret.encode()
        self.recv_window_ms = recv_window_ms
        self.sync_interval_s = sync_interval_s
        self._time_offset_ms: int = 0
        self._last_sync_ts: float = 0.0

    def sign_request(self, params: Dict[str, Any], secret: Optional[str] = None) -> str:
        """Compute HMAC-SHA256 signature for the given parameter dict.

        Adds timestamp and recvWindow to params in-place, then returns
        the hex-encoded signature of the URL-encoded query string.

        Parameters
        ----------
        params -- mutable dict of request parameters (modified in-place)
        secret -- override the instance secret (for testing)

        Returns
        -------
        Hex-encoded HMAC-SHA256 signature string.
        """
        params["timestamp"] = self._get_timestamp()
        params["recvWindow"] = self.recv_window_ms

        query_string = urllib.parse.urlencode(params)
        key = secret.encode() if secret else self.api_secret
        sig = hmac.new(key, query_string.encode(), hashlib.sha256).hexdigest()
        return sig

    def _get_timestamp(self) -> int:
        """Return current UTC timestamp in ms adjusted for server clock offset."""
        return int(time.time() * 1000) + self._time_offset_ms

    async def sync_time(self, session: aiohttp.ClientSession, base_url: str) -> None:
        """Sync local clock with Binance server time.

        Parameters
        ----------
        session  -- active aiohttp session
        base_url -- Binance base URL (spot or futures)
        """
        now = time.monotonic()
        if now - self._last_sync_ts < self.sync_interval_s:
            return

        try:
            local_before = int(time.time() * 1000)
            async with session.get(f"{base_url}/api/v3/time") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    server_time = data.get("serverTime", local_before)
                    local_after = int(time.time() * 1000)
                    rtt = local_after - local_before
                    self._time_offset_ms = server_time - local_before - rtt // 2
                    self._last_sync_ts = time.monotonic()
                    logger.debug("Binance time offset: %dms", self._time_offset_ms)
        except Exception as exc:
            logger.warning("Binance time sync failed: %s", exc)


# ---------------------------------------------------------------------------
# Symbol mapper
# ---------------------------------------------------------------------------


class BinanceSymbolMapper:
    """Maps between SRFM canonical symbols and Binance format.

    SRFM uses dash-separated pairs (BTC-USD, ETH-USD).
    Binance Spot uses concatenated USDT pairs (BTCUSDT, ETHUSDT).
    Binance Futures uses the same format but may include PERP suffixes.

    Custom mappings can be registered via register().
    """

    # Default SRFM -> Binance overrides for non-standard symbols
    _DEFAULT_OVERRIDES: Dict[str, str] = {
        "BTC-USD": "BTCUSDT",
        "ETH-USD": "ETHUSDT",
        "SOL-USD": "SOLUSDT",
        "BNB-USD": "BNBUSDT",
        "XRP-USD": "XRPUSDT",
        "ADA-USD": "ADAUSDT",
        "AVAX-USD": "AVAXUSDT",
        "DOGE-USD": "DOGEUSDT",
        "MATIC-USD": "MATICUSDT",
        "LTC-USD": "LTCUSDT",
        "LINK-USD": "LINKUSDT",
        "DOT-USD": "DOTUSDT",
        "UNI-USD": "UNIUSDT",
        "ATOM-USD": "ATOMUSDT",
        "ETC-USD": "ETCUSDT",
    }

    def __init__(self) -> None:
        self._to_binance: Dict[str, str] = dict(self._DEFAULT_OVERRIDES)
        self._from_binance: Dict[str, str] = {v: k for k, v in self._to_binance.items()}

    def register(self, srfm_symbol: str, binance_symbol: str) -> None:
        """Register a custom symbol mapping.

        Parameters
        ----------
        srfm_symbol    -- SRFM canonical symbol (e.g. BTC-USD)
        binance_symbol -- Binance symbol (e.g. BTCUSDT)
        """
        self._to_binance[srfm_symbol] = binance_symbol
        self._from_binance[binance_symbol] = srfm_symbol

    def to_binance(self, symbol: str) -> str:
        """Convert SRFM symbol to Binance format.

        For unknown symbols, converts 'BASE-QUOTE' to 'BASEQUOTE'
        by stripping the dash and appending T if quote is USD.

        Parameters
        ----------
        symbol -- SRFM symbol (e.g. BTC-USD)

        Returns
        -------
        Binance symbol string (e.g. BTCUSDT)
        """
        if symbol in self._to_binance:
            return self._to_binance[symbol]
        # Fallback: strip dash, replace USD with USDT
        parts = symbol.split("-")
        if len(parts) == 2:
            base, quote = parts
            if quote == "USD":
                quote = "USDT"
            return f"{base}{quote}"
        return symbol

    def from_binance(self, symbol: str) -> str:
        """Convert Binance symbol to SRFM format.

        For unknown symbols, attempts to split on USDT/BTC/ETH/BNB suffix.

        Parameters
        ----------
        symbol -- Binance symbol (e.g. BTCUSDT)

        Returns
        -------
        SRFM symbol string (e.g. BTC-USD)
        """
        if symbol in self._from_binance:
            return self._from_binance[symbol]
        # Fallback: attempt USDT split
        for quote_suffix, srfm_quote in [("USDT", "USD"), ("BUSD", "USD"), ("BTC", "BTC"), ("ETH", "ETH")]:
            if symbol.endswith(quote_suffix):
                base = symbol[: -len(quote_suffix)]
                return f"{base}-{srfm_quote}"
        return symbol


# ---------------------------------------------------------------------------
# Lot size / tick size precision helper
# ---------------------------------------------------------------------------


@dataclass
class SymbolFilters:
    """Precision constraints for a Binance trading symbol.

    Fields
    ------
    lot_size_step   -- minimum order size increment (e.g. 0.001 BTC)
    min_qty         -- minimum order quantity
    max_qty         -- maximum order quantity
    tick_size       -- minimum price increment (e.g. 0.01 USDT)
    min_notional    -- minimum order notional value in USDT
    """

    lot_size_step: float = 0.001
    min_qty: float = 0.001
    max_qty: float = 9000000.0
    tick_size: float = 0.01
    min_notional: float = 10.0

    def round_qty(self, qty: float) -> float:
        """Round quantity to the nearest allowed lot size step."""
        if self.lot_size_step <= 0:
            return qty
        factor = 1.0 / self.lot_size_step
        return round(round(qty * factor) / factor, 8)

    def round_price(self, price: float) -> float:
        """Round price to the nearest tick size."""
        if self.tick_size <= 0:
            return price
        factor = 1.0 / self.tick_size
        return round(round(price * factor) / factor, 8)

    def validate_qty(self, qty: float) -> None:
        """Raise ValueError if qty violates lot size constraints."""
        if qty < self.min_qty:
            raise ValueError(f"qty {qty} below min_qty {self.min_qty}")
        if qty > self.max_qty:
            raise ValueError(f"qty {qty} above max_qty {self.max_qty}")


# ---------------------------------------------------------------------------
# Binance adapter
# ---------------------------------------------------------------------------


class BinanceAdapter(BrokerAdapter):
    """Broker adapter for Binance Spot and USD-M Futures.

    Handles HMAC signing, symbol translation, lot size rounding, and
    USDT-denominated position accounting.

    Parameters
    ----------
    api_key    -- Binance API key
    api_secret -- Binance API secret
    testnet    -- use testnet endpoints (default True)
    futures    -- use USD-M Futures API (default False = Spot)
    circuit_breaker -- optional pre-built circuit breaker
    """

    # Map SRFM OrderType -> Binance type string
    _ORDER_TYPE_MAP: Dict[OrderType, str] = {
        OrderType.MARKET: "MARKET",
        OrderType.LIMIT: "LIMIT",
        OrderType.STOP: "STOP_MARKET",
        OrderType.STOP_LIMIT: "STOP",
        OrderType.TRAILING_STOP: "TRAILING_STOP_MARKET",
    }

    _TIF_MAP: Dict[TimeInForce, str] = {
        TimeInForce.GTC: "GTC",
        TimeInForce.IOC: "IOC",
        TimeInForce.FOK: "FOK",
        TimeInForce.DAY: "GTC",  # Binance has no DAY; use GTC
    }

    _STATUS_MAP: Dict[str, OrderStatus] = {
        "NEW": OrderStatus.SUBMITTED,
        "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
        "FILLED": OrderStatus.FILLED,
        "CANCELED": OrderStatus.CANCELED,
        "PENDING_CANCEL": OrderStatus.SUBMITTED,
        "REJECTED": OrderStatus.REJECTED,
        "EXPIRED": OrderStatus.EXPIRED,
    }

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        futures: bool = False,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        name = f"binance_{'futures' if futures else 'spot'}_{'testnet' if testnet else 'live'}"
        super().__init__(
            name=name,
            asset_class=AssetClass.CRYPTO,
            circuit_breaker=circuit_breaker,
        )
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.futures = futures

        if futures:
            self.base_url = BINANCE_FUTURES_TESTNET_BASE if testnet else BINANCE_FUTURES_BASE
        else:
            self.base_url = BINANCE_SPOT_TESTNET_BASE if testnet else BINANCE_SPOT_BASE

        self._session: Optional[aiohttp.ClientSession] = None
        self._signer = BinanceSignatureGenerator(api_secret)
        self._symbol_mapper = BinanceSymbolMapper()
        self._filters: Dict[str, SymbolFilters] = {}
        self._connected = False
        self._listen_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_headers(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key, "Content-Type": "application/json"}

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._get_headers())
        return self._session

    async def close(self) -> None:
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
        params: Optional[Dict] = None,
        payload: Optional[Dict] = None,
        signed: bool = False,
    ) -> Any:
        """Make a signed or unsigned HTTP request to Binance.

        Parameters
        ----------
        method  -- HTTP method
        path    -- URL path
        params  -- query parameters (modified in-place if signed=True)
        payload -- JSON body (for POST)
        signed  -- if True, adds timestamp + HMAC signature to params

        Returns
        -------
        Parsed JSON response.
        """
        self._check_circuit()

        session = await self._ensure_session()
        await self._signer.sync_time(session, self.base_url)

        url = self.base_url + path
        req_params: Dict[str, Any] = dict(params or {})

        if signed:
            sig = self._signer.sign_request(req_params)
            req_params["signature"] = sig

        last_exc: Optional[Exception] = None
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                async with session.request(
                    method,
                    url,
                    params=req_params if method == "GET" else None,
                    data=urllib.parse.urlencode(req_params) if method != "GET" and not payload else None,
                    json=payload,
                ) as resp:
                    body_text = await resp.text()

                    if resp.status in (200, 201):
                        self._on_success()
                        return json.loads(body_text)

                    if resp.status == 401 or resp.status == 403:
                        exc = AuthenticationError(f"Binance auth error {resp.status}: {body_text}")
                        self._on_failure(exc)
                        raise exc

                    # Parse Binance error codes
                    try:
                        err = json.loads(body_text)
                        code = err.get("code", 0)
                        msg = err.get("msg", body_text)
                    except json.JSONDecodeError:
                        code = 0
                        msg = body_text

                    # -2010: insufficient balance, -1013: lot size, -1111: precision
                    if code in (-2010,):
                        raise InsufficientFundsError(f"Binance: {msg}")

                    if code in (-1013, -1111, -2014, -1100):
                        raise OrderRejectedError(f"Binance rejected order ({code}): {msg}")

                    if resp.status == 429 or code == -1003:
                        retry_after = float(resp.headers.get("Retry-After", RETRY_BASE_DELAY_S))
                        self.logger.warning(
                            "Binance rate limit hit, waiting %.1fs (attempt %d/%d)",
                            retry_after, attempt + 1, RETRY_MAX_ATTEMPTS,
                        )
                        await asyncio.sleep(retry_after)
                        last_exc = RateLimitError("rate limit", retry_after_s=retry_after)
                        continue

                    if resp.status >= 500:
                        self.logger.error("Binance 5xx %d: %s", resp.status, body_text)
                        delay = RETRY_BASE_DELAY_S * (2 ** attempt)
                        last_exc = BrokerAdapterError(f"Binance {resp.status}: {body_text}")
                        self._on_failure(last_exc)
                        await asyncio.sleep(delay)
                        continue

                    raise BrokerAdapterError(f"Binance {resp.status} code={code}: {msg}")

            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                self.logger.warning("Binance connection error (attempt %d): %s", attempt + 1, exc)
                self._on_failure(exc)
                last_exc = exc
                await asyncio.sleep(RETRY_BASE_DELAY_S * (2 ** attempt))

        raise last_exc or BrokerAdapterError("Binance request failed after retries")

    # ------------------------------------------------------------------
    # Symbol filter cache
    # ------------------------------------------------------------------

    async def _get_symbol_filters(self, binance_symbol: str) -> SymbolFilters:
        """Fetch and cache exchange filters for a symbol.

        Parameters
        ----------
        binance_symbol -- Binance-format symbol (e.g. BTCUSDT)

        Returns
        -------
        SymbolFilters with precision constraints.
        """
        if binance_symbol in self._filters:
            return self._filters[binance_symbol]

        path = "/fapi/v1/exchangeInfo" if self.futures else "/api/v3/exchangeInfo"
        try:
            resp = await self._request("GET", path, params={"symbol": binance_symbol})
            symbols = resp.get("symbols", [])
            if not symbols:
                return SymbolFilters()

            sym_info = symbols[0]
            filters_raw = sym_info.get("filters", [])

            sf = SymbolFilters()
            for f in filters_raw:
                ft = f.get("filterType", "")
                if ft == "LOT_SIZE":
                    sf.lot_size_step = float(f.get("stepSize", sf.lot_size_step))
                    sf.min_qty = float(f.get("minQty", sf.min_qty))
                    sf.max_qty = float(f.get("maxQty", sf.max_qty))
                elif ft == "PRICE_FILTER":
                    sf.tick_size = float(f.get("tickSize", sf.tick_size))
                elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
                    sf.min_notional = float(f.get("minNotional", sf.min_notional))

            self._filters[binance_symbol] = sf
            return sf
        except Exception as exc:
            self.logger.warning("Could not fetch filters for %s: %s", binance_symbol, exc)
            return SymbolFilters()

    # ------------------------------------------------------------------
    # BrokerAdapter interface
    # ------------------------------------------------------------------

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        binance_sym = self._symbol_mapper.to_binance(order.symbol)
        filters = await self._get_symbol_filters(binance_sym)

        qty = filters.round_qty(order.qty)
        filters.validate_qty(qty)

        order_type = self._ORDER_TYPE_MAP[order.order_type]
        side = "BUY" if order.side == OrderSide.BUY else "SELL"
        tif = self._TIF_MAP.get(order.time_in_force, "GTC")

        params: Dict[str, Any] = {
            "symbol": binance_sym,
            "side": side,
            "type": order_type,
            "quantity": qty,
            "newClientOrderId": order.client_order_id,
        }

        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            params["price"] = filters.round_price(order.price)  # type: ignore[arg-type]
            params["timeInForce"] = tif

        if order.stop_price is not None:
            params["stopPrice"] = filters.round_price(order.stop_price)

        path = "/fapi/v1/order" if self.futures else "/api/v3/order"

        self.logger.info(
            "Submitting Binance order symbol=%s side=%s qty=%s type=%s",
            binance_sym, side, qty, order_type,
        )

        resp = await self._request("POST", path, params=params, signed=True)
        return self._parse_order_result(resp, order.client_order_id)

    def _parse_order_result(self, resp: Dict, client_order_id: str) -> OrderResult:
        raw_status = resp.get("status", "NEW")
        status = self._STATUS_MAP.get(raw_status, OrderStatus.UNKNOWN)
        filled_qty = float(resp.get("executedQty", 0) or 0)
        avg_price_str = resp.get("avgPrice") or resp.get("price")
        avg_price = float(avg_price_str) if avg_price_str and float(avg_price_str) > 0 else None

        ts_ms = resp.get("transactTime") or resp.get("updateTime") or int(time.time() * 1000)
        submitted_at = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        return OrderResult(
            order_id=str(resp.get("orderId", "")),
            client_order_id=resp.get("clientOrderId", client_order_id),
            status=status,
            submitted_at=submitted_at,
            filled_qty=filled_qty,
            avg_fill_price=avg_price,
        )

    async def cancel_order(self, order_id: str) -> bool:
        path = "/fapi/v1/order" if self.futures else "/api/v3/order"
        self.logger.info("Canceling Binance order order_id=%s", order_id)
        try:
            await self._request("DELETE", path, params={"orderId": order_id}, signed=True)
            return True
        except BrokerAdapterError as exc:
            if "-2011" in str(exc) or "unknown order" in str(exc).lower():
                return False
            raise

    async def get_position(self, symbol: str) -> Optional[Position]:
        positions = await self.get_all_positions()
        return positions.get(symbol)

    async def get_all_positions(self) -> Dict[str, Position]:
        if self.futures:
            path = "/fapi/v2/positionRisk"
            resp = await self._request("GET", path, signed=True)
            positions: Dict[str, Position] = {}
            for raw in resp:
                qty = float(raw.get("positionAmt", 0) or 0)
                if qty == 0.0:
                    continue
                srfm_sym = self._symbol_mapper.from_binance(raw.get("symbol", ""))
                side = PositionSide.LONG if qty > 0 else PositionSide.SHORT
                positions[srfm_sym] = Position(
                    symbol=srfm_sym,
                    qty=abs(qty),
                    avg_entry_price=float(raw.get("entryPrice", 0) or 0),
                    market_value=abs(qty) * float(raw.get("markPrice", 0) or 0),
                    unrealized_pnl=float(raw.get("unRealizedProfit", 0) or 0),
                    side=side,
                    asset_class=AssetClass.CRYPTO,
                )
            return positions
        else:
            # Spot: derive from account balances
            resp = await self._request("GET", "/api/v3/account", signed=True)
            balances = resp.get("balances", [])
            positions = {}
            for bal in balances:
                free = float(bal.get("free", 0) or 0)
                locked = float(bal.get("locked", 0) or 0)
                total = free + locked
                if total <= 0:
                    continue
                asset = bal.get("asset", "")
                if asset in ("USDT", "BUSD", "USD"):
                    continue
                srfm_sym = f"{asset}-USD"
                positions[srfm_sym] = Position(
                    symbol=srfm_sym,
                    qty=total,
                    avg_entry_price=0.0,  # spot has no avg entry from balance endpoint
                    market_value=0.0,
                    unrealized_pnl=0.0,
                    side=PositionSide.LONG,
                    asset_class=AssetClass.CRYPTO,
                )
            return positions

    async def get_account(self) -> AccountInfo:
        if self.futures:
            resp = await self._request("GET", "/fapi/v2/account", signed=True)
            return AccountInfo(
                equity=float(resp.get("totalMarginBalance", 0) or 0),
                cash=float(resp.get("availableBalance", 0) or 0),
                buying_power=float(resp.get("maxWithdrawAmount", 0) or 0),
                margin_used=float(resp.get("totalInitialMargin", 0) or 0),
                leverage=float(resp.get("totalMaintMargin", 1) or 1),
                day_pnl=float(resp.get("totalUnrealizedProfit", 0) or 0),
                currency="USDT",
            )
        else:
            resp = await self._request("GET", "/api/v3/account", signed=True)
            usdt_balance = 0.0
            for bal in resp.get("balances", []):
                if bal.get("asset") in ("USDT", "BUSD"):
                    usdt_balance += float(bal.get("free", 0) or 0) + float(bal.get("locked", 0) or 0)
            return AccountInfo(
                equity=usdt_balance,
                cash=usdt_balance,
                buying_power=usdt_balance,
                margin_used=0.0,
                leverage=1.0,
                currency="USDT",
            )

    async def get_order_status(self, order_id: str) -> OrderStatus:
        path = "/fapi/v1/order" if self.futures else "/api/v3/order"
        resp = await self._request("GET", path, params={"orderId": order_id}, signed=True)
        raw = resp.get("status", "NEW")
        return self._STATUS_MAP.get(raw, OrderStatus.UNKNOWN)

    async def get_recent_fills(self, n: int = 100) -> List[Fill]:
        path = "/fapi/v1/userTrades" if self.futures else "/api/v3/myTrades"
        # Requires a symbol -- return empty list if no specific symbol given
        # For a production system, this would iterate over known symbols
        return []

    def is_connected(self) -> bool:
        return self._connected and self._session is not None and not self._session.closed

    async def test_connection(self) -> bool:
        try:
            path = "/fapi/v1/ping" if self.futures else "/api/v3/ping"
            await self._request("GET", path)
            self._connected = True
            return True
        except Exception as exc:
            self.logger.warning("Binance connection test failed: %s", exc)
            self._connected = False
            return False

    # ------------------------------------------------------------------
    # Binance-specific extras
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> float:
        """Return the current perpetual funding rate for a futures symbol.

        Parameters
        ----------
        symbol -- SRFM symbol (e.g. BTC-USD)

        Returns
        -------
        Funding rate as a decimal (e.g. 0.0001 = 0.01%).
        Requires futures=True; returns 0.0 for spot.
        """
        if not self.futures:
            self.logger.warning("get_funding_rate called on spot adapter -- returning 0.0")
            return 0.0

        binance_sym = self._symbol_mapper.to_binance(symbol)
        resp = await self._request(
            "GET",
            "/fapi/v1/premiumIndex",
            params={"symbol": binance_sym},
        )
        return float(resp.get("lastFundingRate", 0.0) or 0.0)

    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Return the top-N orderbook for a symbol.

        Parameters
        ----------
        symbol -- SRFM symbol (e.g. BTC-USD)
        depth  -- number of bid/ask levels to return (max 5000 for spot)

        Returns
        -------
        Dict with keys 'bids' and 'asks', each a list of [price, qty] pairs.
        """
        binance_sym = self._symbol_mapper.to_binance(symbol)
        path = "/fapi/v1/depth" if self.futures else "/api/v3/depth"
        resp = await self._request("GET", path, params={"symbol": binance_sym, "limit": depth})

        return {
            "symbol": symbol,
            "bids": [[float(p), float(q)] for p, q in resp.get("bids", [])],
            "asks": [[float(p), float(q)] for p, q in resp.get("asks", [])],
            "last_update_id": resp.get("lastUpdateId"),
        }

    async def stream_fills(self, callback: Callable[[Fill], None]) -> None:
        """Stream real-time fill events via Binance user data stream.

        Opens a listen key, connects to the user data WebSocket, and
        delivers executionReport events as Fill objects to callback.
        Refreshes the listen key every 30 minutes (Binance requires this).

        Parameters
        ----------
        callback -- callable receiving Fill objects
        """
        session = await self._ensure_session()

        # Create listen key
        path = "/fapi/v1/listenKey" if self.futures else "/api/v3/userDataStream"
        try:
            resp = await self._request("POST", path)
            listen_key = resp.get("listenKey", "")
        except Exception as exc:
            self.logger.error("Could not create Binance listen key: %s", exc)
            return

        ws_base = BINANCE_FUTURES_WS if self.futures else BINANCE_SPOT_WS
        ws_url = f"{ws_base}/{listen_key}"

        keepalive_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

        async def keepalive() -> None:
            """Refresh listen key every 30 minutes."""
            while True:
                await asyncio.sleep(1800)
                try:
                    await self._request("PUT", path, params={"listenKey": listen_key})
                    self.logger.debug("Binance listen key refreshed")
                except Exception as exc2:
                    self.logger.warning("Listen key refresh failed: %s", exc2)

        while True:
            try:
                async with session.ws_connect(ws_url) as ws:
                    self._connected = True
                    self.logger.info("Binance user data stream connected")
                    if keepalive_task is None or keepalive_task.done():
                        keepalive_task = asyncio.create_task(keepalive())

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            event_type = data.get("e", "")
                            if event_type == "executionReport":
                                order_status = data.get("X", "")
                                if order_status in ("FILLED", "PARTIALLY_FILLED"):
                                    fill = self._parse_execution_report(data)
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(fill)
                                    else:
                                        callback(fill)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break

            except Exception as exc:
                self._connected = False
                self.logger.error("Binance fill stream error: %s -- reconnecting in 5s", exc)
                await asyncio.sleep(5.0)

    def _parse_execution_report(self, data: Dict[str, Any]) -> Fill:
        """Parse a Binance executionReport WebSocket event into a Fill.

        Parameters
        ----------
        data -- raw executionReport event dict from Binance stream
        """
        ts_ms = data.get("T", int(time.time() * 1000))
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        srfm_sym = self._symbol_mapper.from_binance(data.get("s", ""))
        side = OrderSide.BUY if data.get("S", "BUY") == "BUY" else OrderSide.SELL

        return Fill(
            fill_id=str(data.get("t", "")),   # trade ID
            order_id=str(data.get("i", "")),   # order ID
            symbol=srfm_sym,
            side=side,
            qty=float(data.get("l", 0) or 0),   # last filled qty
            price=float(data.get("L", 0) or 0), # last filled price
            timestamp=ts,
            venue="BINANCE",
            commission=float(data.get("n", 0) or 0),
            liquidity="maker" if data.get("m", False) else "taker",
        )

    async def stream_quotes(
        self,
        symbols: List[str],
        callback: Callable[[Dict], None],
    ) -> None:
        """Stream best bid/ask quotes for a list of symbols.

        Parameters
        ----------
        symbols  -- SRFM symbols to subscribe to
        callback -- callable receiving quote dicts with keys: symbol, bid, ask, timestamp
        """
        session = await self._ensure_session()
        binance_symbols = [self._symbol_mapper.to_binance(s).lower() for s in symbols]
        streams = "/".join(f"{s}@bookTicker" for s in binance_symbols)
        ws_base = BINANCE_FUTURES_WS if self.futures else "wss://stream.binance.com:9443/stream"
        ws_url = f"{ws_base}?streams={streams}"

        while True:
            try:
                async with session.ws_connect(ws_url) as ws:
                    self.logger.info("Binance quote stream connected for %d symbols", len(symbols))
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            envelope = json.loads(msg.data)
                            data = envelope.get("data", envelope)
                            event_type = data.get("e", "bookTicker")
                            if event_type == "bookTicker" or "b" in data:
                                quote = {
                                    "symbol": self._symbol_mapper.from_binance(data.get("s", "")),
                                    "bid": float(data.get("b", 0) or 0),
                                    "ask": float(data.get("a", 0) or 0),
                                    "bid_qty": float(data.get("B", 0) or 0),
                                    "ask_qty": float(data.get("A", 0) or 0),
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(quote)
                                else:
                                    callback(quote)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except Exception as exc:
                self.logger.error("Binance quote stream error: %s -- reconnecting in 5s", exc)
                await asyncio.sleep(5.0)
