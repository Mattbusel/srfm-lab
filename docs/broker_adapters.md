# Broker Adapters

## Overview

The `execution/broker_adapters/` package implements a unified adapter pattern for
multi-broker execution. All order submission, cancellation, and status queries
flow through a common interface regardless of which broker handles the trade. The
`AdapterManager` handles routing, failover, and aggregate position tracking across
venues.

```
execution/broker_adapters/
  base_adapter.py
  alpaca_adapter.py
  binance_adapter.py
  paper_adapter.py
  adapter_manager.py
  exceptions.py
```

The Elixir coordination layer provides circuit breaker gates
(`CircuitBreaker[alpaca]`, `CircuitBreaker[binance]`) that the adapters consult
before every submission. No order reaches a broker if its circuit is open.

---

## BaseAdapter -- base_adapter.py

Abstract base class that all adapters must subclass. Enforces the interface,
performs `OrderRequest` validation, and integrates the circuit breaker.

### OrderRequest

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class OrderRequest:
    symbol:   str                                  # e.g. "AAPL", "BTC"
    qty:      float                                # positive = buy, negative = sell
    side:     Literal["buy", "sell"]
    type:     Literal["market", "limit", "stop"]
    price:    float | None = None                  # required for limit/stop
    tif:      Literal["day", "gtc", "ioc"] = "day"
    client_id: str | None = None                   # optional idempotency key

    def validate(self) -> None:
        if not self.symbol:
            raise InvalidSymbolError("symbol must not be empty")
        if self.qty == 0:
            raise ValueError("qty must not be zero")
        if self.type in ("limit", "stop") and self.price is None:
            raise ValueError(f"{self.type} order requires a price")
```

### Abstract Interface

```python
import abc
from typing import AsyncIterator

class BaseAdapter(abc.ABC):

    def __init__(self, circuit_breaker_name: str) -> None:
        self._cb_name = circuit_breaker_name

    # ------------------------------------------------------------------
    # Circuit breaker gate -- called before every submission
    # ------------------------------------------------------------------

    def _assert_circuit_closed(self) -> None:
        from coordination.circuit_breaker import CircuitBreaker
        if CircuitBreaker(self._cb_name).is_open():
            raise CircuitOpenError(f"circuit {self._cb_name!r} is open")

    # ------------------------------------------------------------------
    # Abstract methods -- must be implemented by each adapter
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def submit_order(self, req: OrderRequest) -> str:
        """Submit an order. Returns a broker-assigned order ID."""

    @abc.abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID. Returns True if successfully cancelled."""

    @abc.abstractmethod
    async def get_order_status(self, order_id: str) -> dict:
        """Return a dict with keys: status, filled_qty, avg_price, ..."""

    @abc.abstractmethod
    async def get_positions(self) -> list[dict]:
        """Return current open positions."""

    @abc.abstractmethod
    def stream_fills(self) -> AsyncIterator[dict]:
        """Async generator yielding fill events as they arrive."""
```

### Status Codes

All adapters normalize broker-specific status strings to a common vocabulary:

| Code | Meaning |
|---|---|
| `pending` | Order accepted, not yet routed to exchange |
| `open` | Resting on the book |
| `partially_filled` | Some shares/contracts filled |
| `filled` | Fully executed |
| `cancelled` | Cancelled by user or broker |
| `rejected` | Broker rejected the order (see error field) |
| `expired` | TIF expired without fill |

---

## AlpacaAdapter -- alpaca_adapter.py

Connects to Alpaca's REST API and WebSocket feed. Handles equities and crypto
via the same interface.

### Rate Limiting

A token bucket limiter caps at 200 requests per minute (Alpaca's default limit
for live accounts). The bucket refills continuously so short bursts are allowed
as long as the rolling average stays under the cap.

```python
import asyncio
import time

class TokenBucketRateLimiter:
    """200 req/min = 3.333... tokens/sec"""

    def __init__(self, rate: float = 200 / 60, capacity: float = 20) -> None:
        self._rate     = rate
        self._capacity = capacity
        self._tokens   = capacity
        self._last     = time.monotonic()

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            self._tokens = min(
                self._capacity,
                self._tokens + (now - self._last) * self._rate,
            )
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)
```

### Retry Logic

HTTP 429 (rate limit) and 5xx responses trigger exponential backoff with jitter:

```python
import random

async def _request_with_retry(self, method: str, url: str, **kwargs) -> dict:
    max_attempts = 5
    for attempt in range(max_attempts):
        await self._limiter.acquire()
        resp = await self._session.request(method, url, **kwargs)
        if resp.status == 429 or resp.status >= 500:
            if attempt == max_attempts - 1:
                raise BrokerError(f"HTTP {resp.status} after {max_attempts} attempts")
            backoff = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(backoff)
            continue
        resp.raise_for_status()
        return await resp.json()
```

### WebSocket Fill Streaming

Fills are streamed via Alpaca's WebSocket account updates feed:

```python
async def stream_fills(self):
    async with websockets.connect(self._ws_url) as ws:
        await ws.send(json.dumps({"action": "auth",
                                   "key": self._key, "secret": self._secret}))
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("stream") == "trade_updates":
                event = msg["data"]
                if event["event"] == "fill":
                    yield {
                        "order_id":   event["order"]["id"],
                        "symbol":     event["order"]["symbol"],
                        "filled_qty": float(event["order"]["filled_qty"]),
                        "avg_price":  float(event["order"]["filled_avg_price"]),
                        "side":       event["order"]["side"],
                        "timestamp":  event["timestamp"],
                    }
```

---

## BinanceAdapter -- binance_adapter.py

Supports Spot and USD-M Futures with a unified interface. All requests are
HMAC-SHA256 signed.

### Request Signing

```python
import hashlib
import hmac
import time
import urllib.parse

def _sign(self, params: dict) -> dict:
    params["timestamp"] = int(time.time() * 1000)
    query = urllib.parse.urlencode(params)
    sig = hmac.new(
        self._secret.encode(),
        query.encode(),
        hashlib.sha256,
    ).hexdigest()
    params["signature"] = sig
    return params
```

The API key is sent in the `X-MBX-APIKEY` header on every request.

### Spot vs. Futures Routing

Symbol routing is handled by inspecting the `OrderRequest.symbol` field:

```python
FUTURES_SYMBOLS = {"BTC", "ETH", "SOL", "BNB", "AVAX"}  # configurable

def _base_url(self, symbol: str) -> str:
    if symbol in FUTURES_SYMBOLS:
        if self._testnet:
            return "https://testnet.binancefuture.com"
        return "https://fapi.binance.com"
    return "https://api.binance.com"
```

### Symbol Mapping

Binance uses concatenated pairs rather than slash-separated tickers:

```python
SYMBOL_MAP = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "SOL":  "SOLUSDT",
    "AAPL": "AAPL",    # pass-through for any equity handled via Alpaca
}

def _map_symbol(self, symbol: str) -> str:
    return SYMBOL_MAP.get(symbol, symbol)
```

### Position Mode

Futures accounts may be in one-way or hedge mode. The adapter reads the current
mode on initialization and adjusts the `positionSide` parameter accordingly:

```python
async def _get_position_mode(self) -> str:
    data = await self._get("/fapi/v1/positionSide/dual")
    return "BOTH" if not data["dualSidePosition"] else "LONG_SHORT"
```

### Testnet Support

Pass `testnet=True` to route all traffic to `testnet.binancefuture.com` and
`testnet.binance.vision`. Credentials for the testnet are configured separately
in `config/secrets.toml` under `[binance_testnet]`.

---

## PaperAdapter -- paper_adapter.py

In-memory paper trading adapter. Used for backtesting, strategy development, and
pre-live verification. No network calls are made -- all fills are simulated
internally.

### P&L Tracking

Uses a FIFO cost basis queue per symbol:

```python
from collections import deque

class FifoCostBasis:
    def __init__(self) -> None:
        self._lots: deque[tuple[float, float]] = deque()  # (qty, price)
        self.realized_pnl: float = 0.0

    def buy(self, qty: float, price: float) -> None:
        self._lots.append((qty, price))

    def sell(self, qty: float, price: float) -> float:
        """Return realized P&L for this sale."""
        remaining = qty
        pnl = 0.0
        while remaining > 0 and self._lots:
            lot_qty, lot_price = self._lots[0]
            fill = min(remaining, lot_qty)
            pnl += fill * (price - lot_price)
            remaining -= fill
            if fill < lot_qty:
                self._lots[0] = (lot_qty - fill, lot_price)
            else:
                self._lots.popleft()
        self.realized_pnl += pnl
        return pnl
```

### Partial Fill Simulation

Market orders are partially filled with a random fill fraction in `[0.10, 0.90]`
on the first bar and the remainder on the next bar. This simulates realistic
execution uncertainty:

```python
import random

def _simulate_fill(self, req: OrderRequest) -> tuple[float, float]:
    """Return (filled_qty, avg_price) for a market order."""
    fill_pct   = random.uniform(0.10, 0.90)
    filled_qty = abs(req.qty) * fill_pct
    slippage   = self._slippage_model(req.symbol, filled_qty, req.side)
    mid        = self._market_data.mid_price(req.symbol)
    avg_price  = mid + slippage
    return filled_qty, avg_price
```

### Slippage Model

Slippage is proportional to order size relative to average daily volume (ADV):

```python
def _slippage_model(self, symbol: str, qty: float, side: str) -> float:
    adv         = self._market_data.adv(symbol)
    participation = qty / adv
    impact_bps  = 10.0 * (participation ** 0.6)   # power-law market impact
    mid         = self._market_data.mid_price(symbol)
    direction   = 1 if side == "buy" else -1
    return direction * mid * impact_bps / 10_000
```

### Limit Order Book Matching

Limit orders are stored in a priority queue and matched against incoming bar
midpoints:

```python
# Simplified -- bids matched when mid <= limit, asks when mid >= limit
def _match_limit_orders(self, symbol: str, mid: float) -> list[dict]:
    fills = []
    for order in self._pending_limits[symbol]:
        if order["side"] == "buy"  and mid <= order["price"]:
            fills.append(self._fill_order(order, order["price"]))
        elif order["side"] == "sell" and mid >= order["price"]:
            fills.append(self._fill_order(order, order["price"]))
    return fills
```

---

## AdapterManager -- adapter_manager.py

Routes orders across multiple adapters with automatic failover, health-based
weighting, and aggregate position tracking.

### Routing Logic

```python
class AdapterManager:
    def __init__(self, adapters: dict[str, BaseAdapter]) -> None:
        self._adapters  = adapters          # name -> adapter
        self._health    = {k: 1.0 for k in adapters}    # 0.0 = down, 1.0 = healthy
        self._latencies = {k: [] for k in adapters}      # rolling window (ms)
        self._positions: dict[str, float] = {}           # symbol -> net qty

    async def submit_order(self, req: OrderRequest,
                           preferred: str | None = None) -> str:
        chain = self._routing_chain(req.symbol, preferred)
        last_exc: Exception | None = None
        for name in chain:
            try:
                order_id = await self._adapters[name].submit_order(req)
                self._record_success(name)
                self._update_position(req)
                return order_id
            except CircuitOpenError:
                continue
            except BrokerError as exc:
                last_exc = exc
                self._record_failure(name)
        raise last_exc or BrokerError("all adapters in chain failed")
```

### Failover Chain

The routing chain is ordered by health score * inverse latency:

```python
def _routing_chain(self, symbol: str, preferred: str | None) -> list[str]:
    def score(name: str) -> float:
        avg_lat = sum(self._latencies[name][-10:]) / max(1, len(self._latencies[name][-10:]))
        return self._health[name] / (avg_lat + 1.0)

    candidates = sorted(self._adapters.keys(), key=score, reverse=True)
    if preferred and preferred in candidates:
        candidates.remove(preferred)
        candidates.insert(0, preferred)
    return candidates
```

### Aggregate Position Tracking

`AdapterManager` maintains a unified position book across all venues by
accumulating net fills:

```python
def _update_position(self, req: OrderRequest) -> None:
    delta = req.qty if req.side == "buy" else -req.qty
    self._positions[req.symbol] = self._positions.get(req.symbol, 0.0) + delta

def get_position(self, symbol: str) -> float:
    return self._positions.get(symbol, 0.0)

def get_all_positions(self) -> dict[str, float]:
    return dict(self._positions)
```

---

## Circuit Breaker Integration

The Elixir coordination layer maintains two named circuit breakers:

- `CircuitBreaker[alpaca]` -- gates all Alpaca submissions
- `CircuitBreaker[binance]` -- gates all Binance submissions

The Python adapters call `_assert_circuit_closed()` before every `submit_order`
call. If the circuit is open (tripped by repeated errors or an operator command),
the call raises `CircuitOpenError` immediately and the `AdapterManager` moves to
the next adapter in the failover chain.

Circuit state transitions:

```
closed  -> open    : error threshold exceeded (e.g. 5 errors in 60 s)
open    -> half-open: cooldown period elapsed (default 30 s)
half-open -> closed : probe request succeeds
half-open -> open  : probe request fails
```

Operators can manually trip or reset a circuit from the Elixir console:

```elixir
# Force open
CircuitBreaker.trip("alpaca")

# Force reset
CircuitBreaker.reset("alpaca")
```

---

## Error Taxonomy

All broker-specific errors are normalized to typed Python exceptions defined in
`execution/broker_adapters/exceptions.py`:

| Exception | Error code | Cause |
|---|---|---|
| `RateLimitError` | `rate_limit` | HTTP 429 -- too many requests |
| `AuthFailureError` | `auth_failure` | Invalid API key or signature |
| `InsufficientFundsError` | `insufficient_funds` | Broker rejected due to buying power |
| `InvalidSymbolError` | `invalid_symbol` | Symbol not found or not tradeable |
| `MarketClosedError` | `market_closed` | Order submitted outside market hours |
| `CircuitOpenError` | `circuit_open` | Elixir circuit breaker is open |
| `BrokerError` | `broker_error` | Generic broker-side error |

```python
# exceptions.py

class BrokerError(Exception):
    """Base class for all broker adapter errors."""
    code: str = "broker_error"

class RateLimitError(BrokerError):      code = "rate_limit"
class AuthFailureError(BrokerError):    code = "auth_failure"
class InsufficientFundsError(BrokerError): code = "insufficient_funds"
class InvalidSymbolError(BrokerError):  code = "invalid_symbol"
class MarketClosedError(BrokerError):   code = "market_closed"
class CircuitOpenError(BrokerError):    code = "circuit_open"
```

---

## Adding a New Broker

To add a new broker adapter:

1. Create `execution/broker_adapters/<broker>_adapter.py`
2. Subclass `BaseAdapter` and implement the 5 abstract methods:
   - `submit_order(req: OrderRequest) -> str`
   - `cancel_order(order_id: str) -> bool`
   - `get_order_status(order_id: str) -> dict`
   - `get_positions() -> list[dict]`
   - `stream_fills() -> AsyncIterator[dict]`
3. Map all broker-specific HTTP error codes to the typed exceptions in
   `exceptions.py`
4. Register a circuit breaker name in the Elixir coordination layer
5. Add the adapter to `AdapterManager` in `adapter_manager.py`
6. Add testnet credentials to `config/secrets.toml` under `[<broker>_testnet]`
7. Write a `PaperAdapter` round-trip test to verify the submit/status/cancel
   cycle before connecting live

Minimal skeleton:

```python
# execution/broker_adapters/newbroker_adapter.py

from .base_adapter import BaseAdapter, OrderRequest
from .exceptions import BrokerError, RateLimitError
import aiohttp

class NewBrokerAdapter(BaseAdapter):

    def __init__(self, api_key: str, api_secret: str) -> None:
        super().__init__(circuit_breaker_name="newbroker")
        self._key    = api_key
        self._secret = api_secret
        self._session: aiohttp.ClientSession | None = None

    async def _session_or_create(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def submit_order(self, req: OrderRequest) -> str:
        self._assert_circuit_closed()
        req.validate()
        # ... broker-specific HTTP call ...
        return order_id

    async def cancel_order(self, order_id: str) -> bool:
        self._assert_circuit_closed()
        # ...
        return True

    async def get_order_status(self, order_id: str) -> dict:
        # ...
        return {}

    async def get_positions(self) -> list[dict]:
        # ...
        return []

    async def stream_fills(self):
        # ... async generator over WebSocket or polling ...
        yield {}
```

---

## Configuration Reference

```toml
# config/broker_adapters.toml

[alpaca]
api_key    = "${ALPACA_API_KEY}"
api_secret = "${ALPACA_API_SECRET}"
base_url   = "https://api.alpaca.markets"
ws_url     = "wss://stream.alpaca.markets/stream"
rate_limit = 200   # req/min

[binance]
api_key    = "${BINANCE_API_KEY}"
api_secret = "${BINANCE_API_SECRET}"
testnet    = false
futures_symbols = ["BTC", "ETH", "SOL", "BNB", "AVAX"]

[paper]
slippage_model = "power_law"
partial_fill   = true
fill_pct_low   = 0.10
fill_pct_high  = 0.90

[adapter_manager]
default_preferred = "alpaca"
health_window     = 10       # number of recent requests to average
latency_window    = 10

[circuit_breaker]
error_threshold  = 5
window_seconds   = 60
cooldown_seconds = 30
```
