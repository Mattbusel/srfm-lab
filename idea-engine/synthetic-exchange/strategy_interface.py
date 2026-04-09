"""
strategy_interface.py - Bridge between Event Horizon system and synthetic exchange.

Provides ExchangeClient, StrategyRunner, MultiStrategyRunner,
StrategyTournamentOnExchange, and EventHorizonExchangeTest.
"""

import time
import json
import uuid
import threading
import queue
import logging
import traceback
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
)
from enum import Enum, auto
from collections import deque
import math
import statistics

import numpy as np

try:
    import requests
except ImportError:
    requests = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_API_BASE = "http://localhost:11438"
DEFAULT_TIMEOUT = 5.0
MAX_RETRIES = 3
RETRY_DELAY = 0.25


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SimulationState(Enum):
    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()


@dataclass
class OrderBookLevel:
    price: float
    quantity: float
    num_orders: int = 1


@dataclass
class OrderBook:
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: float = 0.0

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        mid = self.mid_price
        sp = self.spread
        if mid and sp and mid > 0:
            return (sp / mid) * 10000.0
        return None

    def bid_depth(self, levels: int = 5) -> float:
        return sum(b.quantity for b in self.bids[:levels])

    def ask_depth(self, levels: int = 5) -> float:
        return sum(a.quantity for a in self.asks[:levels])

    def imbalance(self, levels: int = 5) -> float:
        bd = self.bid_depth(levels)
        ad = self.ask_depth(levels)
        total = bd + ad
        if total == 0:
            return 0.0
        return (bd - ad) / total

    def weighted_mid(self, levels: int = 1) -> Optional[float]:
        if not self.bids or not self.asks:
            return None
        bid_vol = sum(b.quantity for b in self.bids[:levels])
        ask_vol = sum(a.quantity for a in self.asks[:levels])
        total = bid_vol + ask_vol
        if total == 0:
            return self.mid_price
        return (self.bids[0].price * ask_vol + self.asks[0].price * bid_vol) / total


@dataclass
class MarketBar:
    symbol: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0
    num_trades: int = 0
    spread_avg: float = 0.0
    bid_depth_avg: float = 0.0
    ask_depth_avg: float = 0.0

    @property
    def range_pct(self) -> float:
        if self.open == 0:
            return 0.0
        return (self.high - self.low) / self.open * 100.0

    @property
    def return_pct(self) -> float:
        if self.open == 0:
            return 0.0
        return (self.close - self.open) / self.open * 100.0


@dataclass
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    side: str
    price: float
    quantity: float
    timestamp: float
    fee: float = 0.0
    is_maker: bool = False

    @property
    def notional(self) -> float:
        return self.price * self.quantity


@dataclass
class OrderResponse:
    order_id: str
    status: str
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    timestamp: float = 0.0
    reject_reason: str = ""


@dataclass
class AgentStats:
    agent_id: str
    agent_type: str
    orders_placed: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    total_volume: float = 0.0
    pnl: float = 0.0
    position: float = 0.0
    inventory_value: float = 0.0


@dataclass
class SimulationConfig:
    symbols: List[str] = field(default_factory=lambda: ["SIM-USD"])
    num_market_makers: int = 5
    num_trend_followers: int = 8
    num_mean_reversion: int = 4
    num_noise_traders: int = 20
    num_hft_agents: int = 3
    initial_price: float = 100.0
    tick_size: float = 0.01
    lot_size: float = 1.0
    volatility: float = 0.02
    drift: float = 0.0
    mean_spread_bps: float = 10.0
    book_depth_levels: int = 20
    duration_bars: int = 1000
    bar_interval_ms: int = 1000
    latency_mean_us: int = 50
    latency_std_us: int = 10
    maker_fee_bps: float = -0.5
    taker_fee_bps: float = 1.0
    circuit_breaker_pct: float = 10.0
    max_order_size: float = 1000.0
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": self.symbols,
            "num_market_makers": self.num_market_makers,
            "num_trend_followers": self.num_trend_followers,
            "num_mean_reversion": self.num_mean_reversion,
            "num_noise_traders": self.num_noise_traders,
            "num_hft_agents": self.num_hft_agents,
            "initial_price": self.initial_price,
            "tick_size": self.tick_size,
            "lot_size": self.lot_size,
            "volatility": self.volatility,
            "drift": self.drift,
            "mean_spread_bps": self.mean_spread_bps,
            "book_depth_levels": self.book_depth_levels,
            "duration_bars": self.duration_bars,
            "bar_interval_ms": self.bar_interval_ms,
            "latency_mean_us": self.latency_mean_us,
            "latency_std_us": self.latency_std_us,
            "maker_fee_bps": self.maker_fee_bps,
            "taker_fee_bps": self.taker_fee_bps,
            "circuit_breaker_pct": self.circuit_breaker_pct,
            "max_order_size": self.max_order_size,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# ExchangeClient
# ---------------------------------------------------------------------------

class ExchangeClient:
    """HTTP client connecting to the Go synthetic exchange API at localhost:11438."""

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        api_key: Optional[str] = None,
    ):
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._api_key = api_key
        self._session_id: Optional[str] = None
        self._state = SimulationState.STOPPED
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_running = False
        self._order_log: List[Dict[str, Any]] = []
        self._fill_cache: deque = deque(maxlen=10000)
        self._request_count = 0
        self._error_count = 0
        self._last_request_time = 0.0
        self._rate_limit_window: deque = deque(maxlen=1000)
        self._lock = threading.Lock()

    # -- internal helpers ---------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        if self._session_id:
            h["X-Session-ID"] = self._session_id
        return h

    def _url(self, path: str) -> str:
        return f"{self._api_base}{path}"

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if requests is None:
            raise RuntimeError("requests library not installed")

        url = self._url(path)
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                now = time.time()
                self._rate_limit_window.append(now)
                # prune old entries
                while self._rate_limit_window and self._rate_limit_window[0] < now - 1.0:
                    self._rate_limit_window.popleft()
                if len(self._rate_limit_window) > 500:
                    wait = 1.0 - (now - self._rate_limit_window[0])
                    if wait > 0:
                        time.sleep(wait)

                kwargs: Dict[str, Any] = {
                    "headers": self._headers(),
                    "timeout": self._timeout,
                }
                if payload is not None:
                    kwargs["json"] = payload
                if params is not None:
                    kwargs["params"] = params

                resp = getattr(requests, method.lower())(url, **kwargs)
                self._request_count += 1
                self._last_request_time = time.time()

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    time.sleep(retry_after)
                    continue

                if resp.status_code >= 500:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                    continue

                resp.raise_for_status()
                return resp.json()

            except Exception as exc:
                last_exc = exc
                self._error_count += 1
                if attempt < self._max_retries - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))

        raise ConnectionError(
            f"Failed after {self._max_retries} attempts to {method} {path}: {last_exc}"
        )

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", path, params=params)

    def _post(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("POST", path, payload=payload)

    def _delete(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("DELETE", path, payload=payload)

    # -- simulation lifecycle ------------------------------------------------

    def start_simulation(self, config: Optional[SimulationConfig] = None) -> str:
        """Start a new simulation and return the session ID."""
        if config is None:
            config = SimulationConfig()
        resp = self._post("/api/v1/simulation/start", config.to_dict())
        self._session_id = resp.get("session_id", str(uuid.uuid4()))
        self._state = SimulationState.RUNNING
        logger.info("Simulation started: session=%s", self._session_id)
        return self._session_id

    def stop_simulation(self) -> Dict[str, Any]:
        """Stop the running simulation and return summary."""
        resp = self._post("/api/v1/simulation/stop", {"session_id": self._session_id})
        self._state = SimulationState.STOPPED
        self._stream_running = False
        logger.info("Simulation stopped: session=%s", self._session_id)
        return resp

    def pause_simulation(self) -> Dict[str, Any]:
        resp = self._post("/api/v1/simulation/pause", {"session_id": self._session_id})
        self._state = SimulationState.PAUSED
        return resp

    def resume_simulation(self) -> Dict[str, Any]:
        resp = self._post("/api/v1/simulation/resume", {"session_id": self._session_id})
        self._state = SimulationState.RUNNING
        return resp

    def get_simulation_status(self) -> Dict[str, Any]:
        return self._get("/api/v1/simulation/status")

    @property
    def is_running(self) -> bool:
        return self._state == SimulationState.RUNNING

    # -- market data --------------------------------------------------------

    def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get current order book for a symbol."""
        resp = self._get(f"/api/v1/orderbook/{symbol}", params={"depth": depth})
        bids = [
            OrderBookLevel(
                price=lvl["price"],
                quantity=lvl["quantity"],
                num_orders=lvl.get("num_orders", 1),
            )
            for lvl in resp.get("bids", [])
        ]
        asks = [
            OrderBookLevel(
                price=lvl["price"],
                quantity=lvl["quantity"],
                num_orders=lvl.get("num_orders", 1),
            )
            for lvl in resp.get("asks", [])
        ]
        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=resp.get("timestamp", time.time()),
        )

    def get_market_data(self, symbol: str, bars: int = 100) -> List[MarketBar]:
        """Get OHLCV bars with spread and depth information."""
        resp = self._get(f"/api/v1/market/{symbol}/bars", params={"count": bars})
        result: List[MarketBar] = []
        for b in resp.get("bars", []):
            result.append(MarketBar(
                symbol=symbol,
                timestamp=b.get("timestamp", 0.0),
                open=b.get("open", 0.0),
                high=b.get("high", 0.0),
                low=b.get("low", 0.0),
                close=b.get("close", 0.0),
                volume=b.get("volume", 0.0),
                vwap=b.get("vwap", 0.0),
                num_trades=b.get("num_trades", 0),
                spread_avg=b.get("spread_avg", 0.0),
                bid_depth_avg=b.get("bid_depth_avg", 0.0),
                ask_depth_avg=b.get("ask_depth_avg", 0.0),
            ))
        return result

    def get_latest_bar(self, symbol: str) -> Optional[MarketBar]:
        bars = self.get_market_data(symbol, bars=1)
        return bars[-1] if bars else None

    def get_trade_history(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        return self._get(f"/api/v1/market/{symbol}/trades", params={"limit": limit}).get("trades", [])

    # -- order management ---------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: Union[str, Side],
        qty: float,
        price: Optional[float] = None,
        order_type: Optional[str] = None,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Submit an order and return the response."""
        if isinstance(side, Side):
            side = side.value
        if order_type is None:
            order_type = "limit" if price is not None else "market"
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())[:12]

        payload = {
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "order_type": order_type,
            "time_in_force": time_in_force,
            "client_order_id": client_order_id,
        }
        if price is not None:
            payload["price"] = price

        resp = self._post("/api/v1/orders", payload)

        order_resp = OrderResponse(
            order_id=resp.get("order_id", client_order_id),
            status=resp.get("status", "new"),
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            filled_qty=resp.get("filled_qty", 0.0),
            avg_fill_price=resp.get("avg_fill_price", 0.0),
            timestamp=resp.get("timestamp", time.time()),
            reject_reason=resp.get("reject_reason", ""),
        )
        with self._lock:
            self._order_log.append({
                "order_id": order_resp.order_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "status": order_resp.status,
                "ts": order_resp.timestamp,
            })
        return order_resp

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order."""
        return self._delete(f"/api/v1/orders/{order_id}")

    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if symbol:
            payload["symbol"] = symbol
        return self._delete("/api/v1/orders", payload=payload)

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        return self._get(f"/api/v1/orders/{order_id}")

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._get("/api/v1/orders/open", params=params).get("orders", [])

    # -- fills & positions --------------------------------------------------

    def get_fills(self, symbol: Optional[str] = None, limit: int = 500) -> List[Fill]:
        """Get recent fills."""
        params: Dict[str, Any] = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        resp = self._get("/api/v1/fills", params=params)
        fills: List[Fill] = []
        for f in resp.get("fills", []):
            fill = Fill(
                fill_id=f.get("fill_id", ""),
                order_id=f.get("order_id", ""),
                symbol=f.get("symbol", ""),
                side=f.get("side", ""),
                price=f.get("price", 0.0),
                quantity=f.get("quantity", 0.0),
                timestamp=f.get("timestamp", 0.0),
                fee=f.get("fee", 0.0),
                is_maker=f.get("is_maker", False),
            )
            fills.append(fill)
            self._fill_cache.append(fill)
        return fills

    def get_position(self, symbol: str) -> Dict[str, Any]:
        return self._get(f"/api/v1/positions/{symbol}")

    def get_all_positions(self) -> List[Dict[str, Any]]:
        return self._get("/api/v1/positions").get("positions", [])

    # -- agent statistics ---------------------------------------------------

    def get_agent_stats(self) -> List[AgentStats]:
        """Get statistics for all simulated agents."""
        resp = self._get("/api/v1/agents/stats")
        stats: List[AgentStats] = []
        for a in resp.get("agents", []):
            stats.append(AgentStats(
                agent_id=a.get("agent_id", ""),
                agent_type=a.get("agent_type", ""),
                orders_placed=a.get("orders_placed", 0),
                orders_filled=a.get("orders_filled", 0),
                orders_cancelled=a.get("orders_cancelled", 0),
                total_volume=a.get("total_volume", 0.0),
                pnl=a.get("pnl", 0.0),
                position=a.get("position", 0.0),
                inventory_value=a.get("inventory_value", 0.0),
            ))
        return stats

    def get_agent_detail(self, agent_id: str) -> Dict[str, Any]:
        return self._get(f"/api/v1/agents/{agent_id}")

    # -- event injection ----------------------------------------------------

    def inject_event(self, event_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Inject a market event (flash crash, liquidity drain, etc.)."""
        payload = {
            "event_type": event_type,
            "params": params or {},
            "session_id": self._session_id,
        }
        return self._post("/api/v1/events/inject", payload)

    def inject_flash_crash(self, magnitude_pct: float = 10.0, duration_bars: int = 60) -> Dict[str, Any]:
        return self.inject_event("flash_crash", {
            "magnitude_pct": magnitude_pct,
            "duration_bars": duration_bars,
        })

    def inject_liquidity_drain(self, drain_pct: float = 80.0, duration_bars: int = 120) -> Dict[str, Any]:
        return self.inject_event("liquidity_drain", {
            "drain_pct": drain_pct,
            "duration_bars": duration_bars,
        })

    def inject_volatility_spike(self, multiplier: float = 3.0, duration_bars: int = 60) -> Dict[str, Any]:
        return self.inject_event("volatility_spike", {
            "multiplier": multiplier,
            "duration_bars": duration_bars,
        })

    def inject_circuit_breaker(self) -> Dict[str, Any]:
        return self.inject_event("circuit_breaker", {})

    # -- streaming ----------------------------------------------------------

    def stream_market_data(
        self,
        callback: Callable[[Dict[str, Any]], None],
        symbol: Optional[str] = None,
    ) -> None:
        """Start streaming market data in a background thread."""
        if self._stream_running:
            logger.warning("Stream already running")
            return

        self._stream_running = True

        def _stream_worker():
            url = self._url("/api/v1/stream/market")
            params: Dict[str, Any] = {}
            if symbol:
                params["symbol"] = symbol
            if self._session_id:
                params["session_id"] = self._session_id

            try:
                if requests is None:
                    return
                with requests.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    stream=True,
                    timeout=None,
                ) as resp:
                    resp.raise_for_status()
                    buffer = ""
                    for chunk in resp.iter_content(chunk_size=1024, decode_unicode=True):
                        if not self._stream_running:
                            break
                        if chunk:
                            buffer += chunk
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if line:
                                    try:
                                        data = json.loads(line)
                                        callback(data)
                                    except json.JSONDecodeError:
                                        pass
            except Exception as exc:
                if self._stream_running:
                    logger.error("Stream error: %s", exc)
            finally:
                self._stream_running = False

        self._stream_thread = threading.Thread(target=_stream_worker, daemon=True)
        self._stream_thread.start()

    def stop_stream(self) -> None:
        self._stream_running = False
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=5.0)
            self._stream_thread = None

    # -- diagnostics --------------------------------------------------------

    def ping(self) -> bool:
        try:
            resp = self._get("/api/v1/ping")
            return resp.get("status") == "ok"
        except Exception:
            return False

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "api_base": self._api_base,
            "session_id": self._session_id,
            "state": self._state.name,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "order_log_size": len(self._order_log),
            "fill_cache_size": len(self._fill_cache),
            "stream_running": self._stream_running,
        }


# ---------------------------------------------------------------------------
# Position / P&L tracking
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    trade_count: int = 0

    def update(self, fill: Fill) -> float:
        """Update position from fill and return realized P&L from this fill."""
        sign = 1.0 if fill.side == "buy" else -1.0
        fill_qty = fill.quantity * sign
        realized = 0.0

        if self.quantity == 0:
            self.avg_entry_price = fill.price
            self.quantity = fill_qty
        elif (self.quantity > 0 and sign > 0) or (self.quantity < 0 and sign < 0):
            # adding to position
            total_cost = self.avg_entry_price * abs(self.quantity) + fill.price * fill.quantity
            self.quantity += fill_qty
            if abs(self.quantity) > 0:
                self.avg_entry_price = total_cost / abs(self.quantity)
        else:
            # reducing / flipping position
            close_qty = min(abs(self.quantity), fill.quantity)
            realized = close_qty * (fill.price - self.avg_entry_price) * (1.0 if self.quantity > 0 else -1.0)
            self.realized_pnl += realized
            remaining = fill.quantity - close_qty
            self.quantity += fill_qty
            if remaining > 0 and abs(self.quantity) > 0:
                self.avg_entry_price = fill.price

        self.total_fees += fill.fee
        self.trade_count += 1
        return realized

    def unrealized_pnl(self, current_price: float) -> float:
        if self.quantity == 0:
            return 0.0
        return self.quantity * (current_price - self.avg_entry_price)

    def total_pnl(self, current_price: float) -> float:
        return self.realized_pnl + self.unrealized_pnl(current_price) - self.total_fees

    def notional(self, current_price: float) -> float:
        return abs(self.quantity) * current_price


# ---------------------------------------------------------------------------
# Performance Tracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """Track strategy performance metrics in real time."""

    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate
        self.equity_curve: List[float] = []
        self.returns: List[float] = []
        self.pnl_series: List[float] = []
        self.trade_pnls: List[float] = []
        self.timestamps: List[float] = []
        self.high_water_mark: float = 0.0
        self.drawdown_series: List[float] = []
        self.trade_count: int = 0
        self.win_count: int = 0
        self.loss_count: int = 0
        self.gross_profit: float = 0.0
        self.gross_loss: float = 0.0
        self.max_consecutive_wins: int = 0
        self.max_consecutive_losses: int = 0
        self._current_streak: int = 0
        self._last_streak_positive: bool = True

    def update_equity(self, equity: float, ts: float = 0.0) -> None:
        self.equity_curve.append(equity)
        self.timestamps.append(ts or time.time())

        if len(self.equity_curve) >= 2:
            prev = self.equity_curve[-2]
            if prev != 0:
                ret = (equity - prev) / abs(prev)
            else:
                ret = 0.0
            self.returns.append(ret)

        if equity > self.high_water_mark:
            self.high_water_mark = equity
        dd = 0.0
        if self.high_water_mark > 0:
            dd = (self.high_water_mark - equity) / self.high_water_mark
        self.drawdown_series.append(dd)

    def record_trade(self, pnl: float) -> None:
        self.trade_pnls.append(pnl)
        self.trade_count += 1
        if pnl >= 0:
            self.win_count += 1
            self.gross_profit += pnl
            if self._last_streak_positive:
                self._current_streak += 1
            else:
                self._current_streak = 1
                self._last_streak_positive = True
            self.max_consecutive_wins = max(self.max_consecutive_wins, self._current_streak)
        else:
            self.loss_count += 1
            self.gross_loss += abs(pnl)
            if not self._last_streak_positive:
                self._current_streak += 1
            else:
                self._current_streak = 1
                self._last_streak_positive = False
            self.max_consecutive_losses = max(self.max_consecutive_losses, self._current_streak)

    @property
    def total_pnl(self) -> float:
        return sum(self.trade_pnls) if self.trade_pnls else 0.0

    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float("inf") if self.gross_profit > 0 else 0.0
        return self.gross_profit / self.gross_loss

    @property
    def avg_win(self) -> float:
        if self.win_count == 0:
            return 0.0
        return self.gross_profit / self.win_count

    @property
    def avg_loss(self) -> float:
        if self.loss_count == 0:
            return 0.0
        return self.gross_loss / self.loss_count

    @property
    def expectancy(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.total_pnl / self.trade_count

    @property
    def max_drawdown(self) -> float:
        if not self.drawdown_series:
            return 0.0
        return max(self.drawdown_series)

    @property
    def sharpe_ratio(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        arr = np.array(self.returns)
        mu = np.mean(arr)
        sigma = np.std(arr, ddof=1)
        if sigma == 0:
            return 0.0
        excess = mu - self.risk_free_rate / 252.0
        return excess / sigma * np.sqrt(252.0)

    @property
    def sortino_ratio(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        arr = np.array(self.returns)
        mu = np.mean(arr)
        downside = arr[arr < 0]
        if len(downside) < 2:
            return float("inf") if mu > 0 else 0.0
        down_std = np.std(downside, ddof=1)
        if down_std == 0:
            return 0.0
        return (mu - self.risk_free_rate / 252.0) / down_std * np.sqrt(252.0)

    @property
    def calmar_ratio(self) -> float:
        if self.max_drawdown == 0 or len(self.returns) < 2:
            return 0.0
        annual_return = np.mean(self.returns) * 252.0
        return annual_return / self.max_drawdown

    def summary(self) -> Dict[str, Any]:
        return {
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "expectancy": self.expectancy,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
        }


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:
    """Enforce risk limits for a strategy."""

    def __init__(
        self,
        max_position: float = 100.0,
        max_notional: float = 100000.0,
        max_drawdown_pct: float = 0.10,
        max_order_rate_per_sec: float = 10.0,
        max_daily_loss: float = 5000.0,
        max_open_orders: int = 50,
    ):
        self.max_position = max_position
        self.max_notional = max_notional
        self.max_drawdown_pct = max_drawdown_pct
        self.max_order_rate_per_sec = max_order_rate_per_sec
        self.max_daily_loss = max_daily_loss
        self.max_open_orders = max_open_orders
        self._order_times: deque = deque(maxlen=1000)
        self._halted = False
        self._halt_reason = ""
        self._daily_pnl = 0.0
        self._open_order_count = 0

    def check_order(
        self,
        side: str,
        qty: float,
        price: float,
        current_position: float,
        current_drawdown: float,
    ) -> Tuple[bool, str]:
        """Check if an order is allowed. Returns (allowed, reason)."""
        if self._halted:
            return False, f"Trading halted: {self._halt_reason}"

        # drawdown check
        if current_drawdown >= self.max_drawdown_pct:
            self._halted = True
            self._halt_reason = f"Max drawdown {current_drawdown:.2%} >= {self.max_drawdown_pct:.2%}"
            return False, self._halt_reason

        # daily loss check
        if self._daily_pnl < -self.max_daily_loss:
            self._halted = True
            self._halt_reason = f"Daily loss {self._daily_pnl:.2f} exceeds limit {self.max_daily_loss:.2f}"
            return False, self._halt_reason

        # position check
        sign = 1.0 if side == "buy" else -1.0
        new_pos = current_position + qty * sign
        if abs(new_pos) > self.max_position:
            return False, f"Position would be {new_pos:.2f}, max is {self.max_position:.2f}"

        # notional check
        notional = abs(new_pos) * price
        if notional > self.max_notional:
            return False, f"Notional {notional:.2f} exceeds max {self.max_notional:.2f}"

        # order rate check
        now = time.time()
        self._order_times.append(now)
        while self._order_times and self._order_times[0] < now - 1.0:
            self._order_times.popleft()
        if len(self._order_times) > self.max_order_rate_per_sec:
            return False, f"Order rate {len(self._order_times)}/s exceeds limit {self.max_order_rate_per_sec}"

        # open orders check
        if self._open_order_count >= self.max_open_orders:
            return False, f"Open orders {self._open_order_count} >= max {self.max_open_orders}"

        return True, ""

    def update_daily_pnl(self, pnl_delta: float) -> None:
        self._daily_pnl += pnl_delta

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0
        self._halted = False
        self._halt_reason = ""

    def update_open_orders(self, count: int) -> None:
        self._open_order_count = count

    def force_halt(self, reason: str) -> None:
        self._halted = True
        self._halt_reason = reason

    def resume(self) -> None:
        self._halted = False
        self._halt_reason = ""

    @property
    def is_halted(self) -> bool:
        return self._halted


# ---------------------------------------------------------------------------
# StrategyRunner
# ---------------------------------------------------------------------------

@dataclass
class StrategyRunConfig:
    symbol: str = "SIM-USD"
    initial_capital: float = 100000.0
    max_position: float = 100.0
    max_drawdown_pct: float = 0.10
    max_order_rate: float = 10.0
    max_daily_loss: float = 5000.0
    warmup_bars: int = 20
    bar_sleep_sec: float = 0.0


class StrategyRunner:
    """
    Run an Event Horizon signal function against the simulated exchange.

    The signal function signature:
        signal_fn(bars: List[MarketBar], position: Position, bar_idx: int) -> Optional[Dict]

    Returns None for no action, or:
        {"side": "buy"/"sell", "qty": float, "price": Optional[float]}
    """

    def __init__(
        self,
        signal_fn: Callable,
        client: ExchangeClient,
        config: Optional[StrategyRunConfig] = None,
        name: str = "strategy",
    ):
        self.signal_fn = signal_fn
        self.client = client
        self.config = config or StrategyRunConfig()
        self.name = name
        self.position = Position(symbol=self.config.symbol)
        self.risk_mgr = RiskManager(
            max_position=self.config.max_position,
            max_drawdown_pct=self.config.max_drawdown_pct,
            max_order_rate_per_sec=self.config.max_order_rate,
            max_daily_loss=self.config.max_daily_loss,
        )
        self.tracker = PerformanceTracker()
        self._bars: List[MarketBar] = []
        self._running = False
        self._bar_count = 0
        self._errors: List[str] = []

    def run(self, num_bars: Optional[int] = None) -> Dict[str, Any]:
        """Run bar-by-bar loop for num_bars (or until simulation ends)."""
        self._running = True
        capital = self.config.initial_capital
        self.tracker.update_equity(capital)
        target_bars = num_bars or 999999

        logger.info("StrategyRunner '%s' starting for %d bars", self.name, target_bars)

        for bar_idx in range(target_bars):
            if not self._running:
                break

            try:
                bar = self.client.get_latest_bar(self.config.symbol)
                if bar is None:
                    if self.config.bar_sleep_sec > 0:
                        time.sleep(self.config.bar_sleep_sec)
                    continue

                self._bars.append(bar)
                self._bar_count += 1
                current_price = bar.close

                # process fills
                fills = self.client.get_fills(self.config.symbol, limit=100)
                for fill in fills:
                    rpnl = self.position.update(fill)
                    if rpnl != 0:
                        self.tracker.record_trade(rpnl)
                        self.risk_mgr.update_daily_pnl(rpnl)

                # update equity
                equity = capital + self.position.total_pnl(current_price)
                self.tracker.update_equity(equity, bar.timestamp)

                # skip warmup period
                if bar_idx < self.config.warmup_bars:
                    if self.config.bar_sleep_sec > 0:
                        time.sleep(self.config.bar_sleep_sec)
                    continue

                # generate signal
                signal = self.signal_fn(self._bars, self.position, bar_idx)

                if signal is not None:
                    side = signal.get("side", "buy")
                    qty = signal.get("qty", 1.0)
                    price = signal.get("price")
                    order_price = price if price else current_price

                    allowed, reason = self.risk_mgr.check_order(
                        side, qty, order_price,
                        self.position.quantity,
                        self.tracker.max_drawdown,
                    )
                    if allowed:
                        self.client.submit_order(
                            self.config.symbol, side, qty, price=price
                        )
                    else:
                        logger.debug("Order rejected by risk: %s", reason)

            except Exception as exc:
                self._errors.append(f"Bar {bar_idx}: {exc}")
                logger.error("Error at bar %d: %s", bar_idx, exc)

            if self.config.bar_sleep_sec > 0:
                time.sleep(self.config.bar_sleep_sec)

        self._running = False
        return self.results()

    def stop(self) -> None:
        self._running = False

    def results(self) -> Dict[str, Any]:
        last_price = self._bars[-1].close if self._bars else 0.0
        return {
            "name": self.name,
            "bars_processed": self._bar_count,
            "position": self.position.quantity,
            "realized_pnl": self.position.realized_pnl,
            "unrealized_pnl": self.position.unrealized_pnl(last_price),
            "total_pnl": self.position.total_pnl(last_price),
            "total_fees": self.position.total_fees,
            "errors": len(self._errors),
            "performance": self.tracker.summary(),
        }


# ---------------------------------------------------------------------------
# MultiStrategyRunner
# ---------------------------------------------------------------------------

class MultiStrategyRunner:
    """Run multiple strategies simultaneously against the same exchange."""

    def __init__(
        self,
        strategies: List[Tuple[str, Callable]],
        client: ExchangeClient,
        config: Optional[StrategyRunConfig] = None,
    ):
        self.client = client
        self.config = config or StrategyRunConfig()
        self.runners: List[StrategyRunner] = []
        for name, fn in strategies:
            runner = StrategyRunner(
                signal_fn=fn,
                client=client,
                config=self.config,
                name=name,
            )
            self.runners.append(runner)

    def run(self, num_bars: Optional[int] = None) -> Dict[str, Any]:
        """Run all strategies in parallel threads."""
        threads: List[threading.Thread] = []
        results: Dict[str, Any] = {}
        lock = threading.Lock()
        target_bars = num_bars or 1000

        def _run_strategy(runner: StrategyRunner):
            res = runner.run(target_bars)
            with lock:
                results[runner.name] = res

        for runner in self.runners:
            t = threading.Thread(target=_run_strategy, args=(runner,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return self._compile_results(results)

    def stop_all(self) -> None:
        for runner in self.runners:
            runner.stop()

    def _compile_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comparative results across strategies."""
        summary: Dict[str, Any] = {"strategies": results}

        if not results:
            return summary

        # rankings
        by_pnl = sorted(results.items(), key=lambda x: x[1].get("total_pnl", 0), reverse=True)
        by_sharpe = sorted(
            results.items(),
            key=lambda x: x[1].get("performance", {}).get("sharpe_ratio", 0),
            reverse=True,
        )
        by_dd = sorted(
            results.items(),
            key=lambda x: x[1].get("performance", {}).get("max_drawdown", 1),
        )

        summary["ranking_by_pnl"] = [name for name, _ in by_pnl]
        summary["ranking_by_sharpe"] = [name for name, _ in by_sharpe]
        summary["ranking_by_drawdown"] = [name for name, _ in by_dd]

        # best/worst
        summary["best_pnl"] = by_pnl[0][0] if by_pnl else None
        summary["worst_pnl"] = by_pnl[-1][0] if by_pnl else None
        summary["best_sharpe"] = by_sharpe[0][0] if by_sharpe else None

        # aggregate
        pnls = [r.get("total_pnl", 0) for r in results.values()]
        summary["aggregate_pnl"] = sum(pnls)
        summary["mean_pnl"] = np.mean(pnls) if pnls else 0.0
        summary["pnl_std"] = np.std(pnls) if pnls else 0.0

        sharpes = [r.get("performance", {}).get("sharpe_ratio", 0) for r in results.values()]
        summary["mean_sharpe"] = np.mean(sharpes) if sharpes else 0.0

        return summary


# ---------------------------------------------------------------------------
# Tournament Strategy Templates (12 archetypes)
# ---------------------------------------------------------------------------

def _momentum_signal(bars: List[MarketBar], pos: Position, idx: int) -> Optional[Dict]:
    """Trend-following momentum strategy."""
    if len(bars) < 20:
        return None
    closes = np.array([b.close for b in bars[-20:]])
    ret_10 = (closes[-1] / closes[-10] - 1.0) if closes[-10] != 0 else 0
    ret_20 = (closes[-1] / closes[0] - 1.0) if closes[0] != 0 else 0
    score = 0.6 * ret_10 + 0.4 * ret_20
    if score > 0.005 and pos.quantity <= 0:
        return {"side": "buy", "qty": 5.0}
    elif score < -0.005 and pos.quantity >= 0:
        return {"side": "sell", "qty": 5.0}
    return None


def _mean_reversion_signal(bars: List[MarketBar], pos: Position, idx: int) -> Optional[Dict]:
    """Mean-reversion with Bollinger Bands."""
    if len(bars) < 30:
        return None
    closes = np.array([b.close for b in bars[-30:]])
    ma = np.mean(closes)
    std = np.std(closes)
    if std == 0:
        return None
    z = (closes[-1] - ma) / std
    if z < -2.0 and pos.quantity <= 0:
        return {"side": "buy", "qty": 3.0}
    elif z > 2.0 and pos.quantity >= 0:
        return {"side": "sell", "qty": 3.0}
    elif abs(z) < 0.5 and pos.quantity != 0:
        side = "sell" if pos.quantity > 0 else "buy"
        return {"side": side, "qty": abs(pos.quantity)}
    return None


def _breakout_signal(bars: List[MarketBar], pos: Position, idx: int) -> Optional[Dict]:
    """Donchian channel breakout."""
    if len(bars) < 50:
        return None
    highs = np.array([b.high for b in bars[-50:]])
    lows = np.array([b.low for b in bars[-50:]])
    upper = np.max(highs[:-1])
    lower = np.min(lows[:-1])
    price = bars[-1].close
    if price > upper and pos.quantity <= 0:
        return {"side": "buy", "qty": 4.0}
    elif price < lower and pos.quantity >= 0:
        return {"side": "sell", "qty": 4.0}
    return None


def _vol_targeting_signal(bars: List[MarketBar], pos: Position, idx: int) -> Optional[Dict]:
    """Volatility-targeted momentum."""
    if len(bars) < 30:
        return None
    closes = np.array([b.close for b in bars[-30:]])
    rets = np.diff(np.log(closes + 1e-10))
    vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 0.01
    target_vol = 0.15
    scale = target_vol / max(vol, 0.01)
    scale = min(scale, 3.0)
    mom = np.mean(rets[-10:]) if len(rets) >= 10 else 0
    qty = max(1.0, round(abs(mom) * scale * 100, 0))
    qty = min(qty, 20.0)
    if mom > 0.001 and pos.quantity <= 0:
        return {"side": "buy", "qty": qty}
    elif mom < -0.001 and pos.quantity >= 0:
        return {"side": "sell", "qty": qty}
    return None


def _rsi_signal(bars: List[MarketBar], pos: Position, idx: int) -> Optional[Dict]:
    """RSI mean-reversion."""
    if len(bars) < 16:
        return None
    closes = np.array([b.close for b in bars[-16:]])
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)

    if rsi < 25 and pos.quantity <= 0:
        return {"side": "buy", "qty": 3.0}
    elif rsi > 75 and pos.quantity >= 0:
        return {"side": "sell", "qty": 3.0}
    return None


def _macd_signal(bars: List[MarketBar], pos: Position, idx: int) -> Optional[Dict]:
    """MACD crossover."""
    if len(bars) < 35:
        return None
    closes = np.array([b.close for b in bars[-35:]])
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    hist = macd_line - signal_line
    if len(hist) < 2:
        return None
    if hist[-1] > 0 and hist[-2] <= 0 and pos.quantity <= 0:
        return {"side": "buy", "qty": 4.0}
    elif hist[-1] < 0 and hist[-2] >= 0 and pos.quantity >= 0:
        return {"side": "sell", "qty": 4.0}
    return None


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (period + 1)
    result = np.empty_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def _pairs_signal(bars: List[MarketBar], pos: Position, idx: int) -> Optional[Dict]:
    """Spread-based: buy when spread widens (value), sell when narrow."""
    if len(bars) < 20:
        return None
    spreads = np.array([b.spread_avg for b in bars[-20:]])
    mu = np.mean(spreads)
    std = np.std(spreads)
    if std == 0:
        return None
    z = (spreads[-1] - mu) / std
    if z > 1.5 and pos.quantity <= 0:
        return {"side": "buy", "qty": 2.0}
    elif z < -1.5 and pos.quantity >= 0:
        return {"side": "sell", "qty": 2.0}
    return None


def _vwap_revert_signal(bars: List[MarketBar], pos: Position, idx: int) -> Optional[Dict]:
    """Revert toward VWAP."""
    if len(bars) < 10:
        return None
    bar = bars[-1]
    if bar.vwap == 0:
        return None
    deviation = (bar.close - bar.vwap) / bar.vwap
    if deviation < -0.003 and pos.quantity <= 0:
        return {"side": "buy", "qty": 3.0}
    elif deviation > 0.003 and pos.quantity >= 0:
        return {"side": "sell", "qty": 3.0}
    return None


def _imbalance_signal(bars: List[MarketBar], pos: Position, idx: int) -> Optional[Dict]:
    """Order book imbalance signal."""
    if len(bars) < 5:
        return None
    bid_depths = np.array([b.bid_depth_avg for b in bars[-5:]])
    ask_depths = np.array([b.ask_depth_avg for b in bars[-5:]])
    total = bid_depths + ask_depths
    total = np.where(total == 0, 1, total)
    imb = (bid_depths - ask_depths) / total
    avg_imb = np.mean(imb)
    if avg_imb > 0.3 and pos.quantity <= 0:
        return {"side": "buy", "qty": 2.0}
    elif avg_imb < -0.3 and pos.quantity >= 0:
        return {"side": "sell", "qty": 2.0}
    return None


def _microstructure_signal(bars: List[MarketBar], pos: Position, idx: int) -> Optional[Dict]:
    """Trade flow / microstructure signal."""
    if len(bars) < 15:
        return None
    volumes = np.array([b.volume for b in bars[-15:]])
    closes = np.array([b.close for b in bars[-15:]])
    rets = np.diff(np.log(closes + 1e-10))
    if len(rets) < 2:
        return None
    signed_vol = rets * volumes[1:]
    cum_flow = np.cumsum(signed_vol)
    flow_ma = np.mean(cum_flow[-5:])
    flow_std = np.std(cum_flow) if len(cum_flow) > 1 else 1.0
    if flow_std == 0:
        return None
    z = flow_ma / flow_std
    if z > 1.5 and pos.quantity <= 0:
        return {"side": "buy", "qty": 2.0}
    elif z < -1.5 and pos.quantity >= 0:
        return {"side": "sell", "qty": 2.0}
    return None


def _regime_switch_signal(bars: List[MarketBar], pos: Position, idx: int) -> Optional[Dict]:
    """Simple regime detection: trend vs mean-revert based on Hurst exponent estimate."""
    if len(bars) < 50:
        return None
    closes = np.array([b.close for b in bars[-50:]])
    log_prices = np.log(closes + 1e-10)
    # Hurst exponent via R/S
    n = len(log_prices)
    rets = np.diff(log_prices)
    mean_r = np.mean(rets)
    deviations = np.cumsum(rets - mean_r)
    r = np.max(deviations) - np.min(deviations)
    s = np.std(rets, ddof=1)
    if s == 0:
        return None
    rs = r / s
    hurst = np.log(rs) / np.log(n) if rs > 0 and n > 1 else 0.5

    if hurst > 0.6:
        # trending regime: use momentum
        mom = np.mean(rets[-10:])
        if mom > 0 and pos.quantity <= 0:
            return {"side": "buy", "qty": 3.0}
        elif mom < 0 and pos.quantity >= 0:
            return {"side": "sell", "qty": 3.0}
    elif hurst < 0.4:
        # mean reverting: use z-score
        z = (closes[-1] - np.mean(closes)) / (np.std(closes) + 1e-10)
        if z < -1.5 and pos.quantity <= 0:
            return {"side": "buy", "qty": 3.0}
        elif z > 1.5 and pos.quantity >= 0:
            return {"side": "sell", "qty": 3.0}
    return None


def _adaptive_signal(bars: List[MarketBar], pos: Position, idx: int) -> Optional[Dict]:
    """Adaptive: blend momentum and mean-reversion based on recent performance."""
    if len(bars) < 40:
        return None
    closes = np.array([b.close for b in bars[-40:]])
    rets = np.diff(np.log(closes + 1e-10))
    # compute rolling strategy returns
    mom_returns = rets[:-1] * np.sign(rets[:-1])  # momentum pnl proxy
    mr_returns = -rets[:-1] * np.sign(rets[:-1])  # mean-rev pnl proxy
    mom_sharpe = np.mean(mom_returns[-10:]) / (np.std(mom_returns[-10:]) + 1e-10)
    mr_sharpe = np.mean(mr_returns[-10:]) / (np.std(mr_returns[-10:]) + 1e-10)

    weight = 1.0 / (1.0 + np.exp(-(mom_sharpe - mr_sharpe)))  # sigmoid blend

    # momentum component
    mom_sig = np.mean(rets[-10:])
    # mean-rev component
    z = (closes[-1] - np.mean(closes[-20:])) / (np.std(closes[-20:]) + 1e-10)
    mr_sig = -z * 0.001

    blended = weight * mom_sig + (1 - weight) * mr_sig

    if blended > 0.001 and pos.quantity <= 0:
        return {"side": "buy", "qty": 3.0}
    elif blended < -0.001 and pos.quantity >= 0:
        return {"side": "sell", "qty": 3.0}
    return None


TOURNAMENT_TEMPLATES = {
    "momentum": _momentum_signal,
    "mean_reversion": _mean_reversion_signal,
    "breakout": _breakout_signal,
    "vol_targeting": _vol_targeting_signal,
    "rsi": _rsi_signal,
    "macd": _macd_signal,
    "pairs_spread": _pairs_signal,
    "vwap_revert": _vwap_revert_signal,
    "imbalance": _imbalance_signal,
    "microstructure": _microstructure_signal,
    "regime_switch": _regime_switch_signal,
    "adaptive": _adaptive_signal,
}


# ---------------------------------------------------------------------------
# StrategyTournamentOnExchange
# ---------------------------------------------------------------------------

class StrategyTournamentOnExchange:
    """
    Run all 12 tournament strategy templates on SIMULATED data and compare
    rankings with historical-data results to detect overfitting.
    """

    def __init__(
        self,
        client: ExchangeClient,
        config: Optional[StrategyRunConfig] = None,
        historical_rankings: Optional[Dict[str, int]] = None,
    ):
        self.client = client
        self.config = config or StrategyRunConfig()
        self.historical_rankings = historical_rankings or {}
        self._results: Dict[str, Dict[str, Any]] = {}

    def run_tournament(self, num_bars: int = 500) -> Dict[str, Any]:
        """Run all 12 templates and produce ranking comparison."""
        strategies = list(TOURNAMENT_TEMPLATES.items())
        multi = MultiStrategyRunner(strategies, self.client, self.config)
        raw_results = multi.run(num_bars)
        self._results = raw_results.get("strategies", {})

        sim_rankings = self._compute_rankings()
        stability = self._compute_stability(sim_rankings)
        overfit_score = self._compute_overfit_score(sim_rankings)

        return {
            "sim_results": self._results,
            "sim_rankings": sim_rankings,
            "historical_rankings": self.historical_rankings,
            "rank_correlation": stability.get("spearman_rho", None),
            "ranking_stable": stability.get("is_stable", False),
            "overfit_score": overfit_score,
            "interpretation": self._interpret(stability, overfit_score),
        }

    def _compute_rankings(self) -> Dict[str, int]:
        sorted_by_sharpe = sorted(
            self._results.items(),
            key=lambda x: x[1].get("performance", {}).get("sharpe_ratio", 0),
            reverse=True,
        )
        return {name: rank + 1 for rank, (name, _) in enumerate(sorted_by_sharpe)}

    def _compute_stability(self, sim_rankings: Dict[str, int]) -> Dict[str, Any]:
        if not self.historical_rankings:
            return {"is_stable": False, "spearman_rho": None, "reason": "no historical rankings"}
        common = set(sim_rankings.keys()) & set(self.historical_rankings.keys())
        if len(common) < 3:
            return {"is_stable": False, "spearman_rho": None, "reason": "too few common strategies"}
        sim_ranks = [sim_rankings[k] for k in common]
        hist_ranks = [self.historical_rankings[k] for k in common]
        n = len(sim_ranks)
        d_sq = sum((s - h) ** 2 for s, h in zip(sim_ranks, hist_ranks))
        rho = 1.0 - 6.0 * d_sq / (n * (n ** 2 - 1))
        return {
            "spearman_rho": rho,
            "is_stable": rho > 0.6,
            "d_squared_sum": d_sq,
            "n_strategies": n,
        }

    def _compute_overfit_score(self, sim_rankings: Dict[str, int]) -> float:
        """0 = no overfit, 1 = completely overfit. Based on rank distance."""
        if not self.historical_rankings:
            return -1.0
        common = set(sim_rankings.keys()) & set(self.historical_rankings.keys())
        if len(common) < 2:
            return -1.0
        n = len(common)
        max_d_sq = 2 * n * (n ** 2 - 1) / 6.0  # worst case
        d_sq = sum((sim_rankings[k] - self.historical_rankings[k]) ** 2 for k in common)
        if max_d_sq == 0:
            return 0.0
        return d_sq / max_d_sq

    def _interpret(self, stability: Dict[str, Any], overfit_score: float) -> str:
        rho = stability.get("spearman_rho")
        if rho is None:
            return "Cannot assess stability without historical rankings."
        if rho > 0.8:
            return f"ROBUST: Rank correlation {rho:.3f} is very high. Strategies generalize well."
        elif rho > 0.6:
            return f"MODERATE: Rank correlation {rho:.3f} is decent. Some strategies may be overfit."
        elif rho > 0.3:
            return f"UNSTABLE: Rank correlation {rho:.3f} is low. Rankings shift significantly on simulated data."
        else:
            return f"OVERFIT: Rank correlation {rho:.3f} is very low. Historical rankings do not hold."


# ---------------------------------------------------------------------------
# EventHorizonExchangeTest
# ---------------------------------------------------------------------------

class EventHorizonModule:
    """Stub for a single EH module that produces or consumes data."""

    def __init__(self, name: str, module_fn: Optional[Callable] = None):
        self.name = name
        self.module_fn = module_fn
        self.state: Dict[str, Any] = {}
        self.enabled = True
        self.error_count = 0
        self.call_count = 0
        self.last_output: Any = None

    def tick(self, context: Dict[str, Any]) -> Any:
        if not self.enabled:
            return None
        self.call_count += 1
        try:
            if self.module_fn:
                self.last_output = self.module_fn(context, self.state)
            return self.last_output
        except Exception as exc:
            self.error_count += 1
            logger.error("Module %s error: %s", self.name, exc)
            return None


class EventHorizonExchangeTest:
    """
    Integration test: run all 27 Event Horizon modules against the synthetic exchange.

    The full autonomous loop:
      signals -> dreams -> debate -> trade -> risk -> adapt -> repeat

    Modules (27):
      1-3: Price signal generators (momentum, mean_rev, breakout)
      4-6: Volume / flow analyzers
      7-9: Volatility estimators (GARCH, realized, implied proxy)
      10-12: Regime detectors (HMM, clustering, Hurst)
      13-15: Dream / scenario generators (monte carlo, GAN proxy, copula)
      16-18: Debate / signal aggregation (voting, stacking, Bayesian)
      19-21: Position sizers (Kelly, risk parity, vol target)
      22-24: Risk managers (drawdown, correlation, tail risk)
      25-27: Adaptation (parameter tuning, regime adaptation, strategy selection)
    """

    def __init__(self, client: ExchangeClient, config: Optional[StrategyRunConfig] = None):
        self.client = client
        self.config = config or StrategyRunConfig()
        self.modules: List[EventHorizonModule] = self._build_modules()
        self.position = Position(symbol=self.config.symbol)
        self.tracker = PerformanceTracker()
        self.risk_mgr = RiskManager(
            max_position=self.config.max_position,
            max_drawdown_pct=self.config.max_drawdown_pct,
        )
        self._context: Dict[str, Any] = {
            "bars": [],
            "signals": {},
            "dreams": {},
            "debates": {},
            "sizes": {},
            "risks": {},
            "adaptations": {},
            "position": 0.0,
            "equity": self.config.initial_capital,
            "bar_idx": 0,
        }
        self._running = False
        self._errors: List[str] = []

    def _build_modules(self) -> List[EventHorizonModule]:
        modules: List[EventHorizonModule] = []

        # 1-3: Signal generators
        modules.append(EventHorizonModule("signal_momentum", self._mod_signal_momentum))
        modules.append(EventHorizonModule("signal_mean_rev", self._mod_signal_mean_rev))
        modules.append(EventHorizonModule("signal_breakout", self._mod_signal_breakout))

        # 4-6: Volume / flow
        modules.append(EventHorizonModule("flow_volume_profile", self._mod_volume_profile))
        modules.append(EventHorizonModule("flow_trade_intensity", self._mod_trade_intensity))
        modules.append(EventHorizonModule("flow_imbalance", self._mod_flow_imbalance))

        # 7-9: Volatility
        modules.append(EventHorizonModule("vol_realized", self._mod_vol_realized))
        modules.append(EventHorizonModule("vol_garch", self._mod_vol_garch))
        modules.append(EventHorizonModule("vol_range", self._mod_vol_range))

        # 10-12: Regime
        modules.append(EventHorizonModule("regime_hmm", self._mod_regime_hmm))
        modules.append(EventHorizonModule("regime_cluster", self._mod_regime_cluster))
        modules.append(EventHorizonModule("regime_hurst", self._mod_regime_hurst))

        # 13-15: Dream / scenario
        modules.append(EventHorizonModule("dream_montecarlo", self._mod_dream_mc))
        modules.append(EventHorizonModule("dream_bootstrap", self._mod_dream_bootstrap))
        modules.append(EventHorizonModule("dream_stress", self._mod_dream_stress))

        # 16-18: Debate / aggregation
        modules.append(EventHorizonModule("debate_vote", self._mod_debate_vote))
        modules.append(EventHorizonModule("debate_stack", self._mod_debate_stack))
        modules.append(EventHorizonModule("debate_bayesian", self._mod_debate_bayesian))

        # 19-21: Position sizing
        modules.append(EventHorizonModule("size_kelly", self._mod_size_kelly))
        modules.append(EventHorizonModule("size_risk_parity", self._mod_size_risk_parity))
        modules.append(EventHorizonModule("size_vol_target", self._mod_size_vol_target))

        # 22-24: Risk management
        modules.append(EventHorizonModule("risk_drawdown", self._mod_risk_drawdown))
        modules.append(EventHorizonModule("risk_correlation", self._mod_risk_correlation))
        modules.append(EventHorizonModule("risk_tail", self._mod_risk_tail))

        # 25-27: Adaptation
        modules.append(EventHorizonModule("adapt_params", self._mod_adapt_params))
        modules.append(EventHorizonModule("adapt_regime", self._mod_adapt_regime))
        modules.append(EventHorizonModule("adapt_select", self._mod_adapt_select))

        return modules

    # --- Module implementations ---

    def _mod_signal_momentum(self, ctx: Dict, state: Dict) -> float:
        bars = ctx.get("bars", [])
        if len(bars) < 20:
            return 0.0
        closes = np.array([b.close for b in bars[-20:]])
        return float(np.mean(np.diff(np.log(closes + 1e-10))))

    def _mod_signal_mean_rev(self, ctx: Dict, state: Dict) -> float:
        bars = ctx.get("bars", [])
        if len(bars) < 30:
            return 0.0
        closes = np.array([b.close for b in bars[-30:]])
        z = (closes[-1] - np.mean(closes)) / (np.std(closes) + 1e-10)
        return float(-z * 0.01)

    def _mod_signal_breakout(self, ctx: Dict, state: Dict) -> float:
        bars = ctx.get("bars", [])
        if len(bars) < 40:
            return 0.0
        highs = np.array([b.high for b in bars[-40:]])
        lows = np.array([b.low for b in bars[-40:]])
        price = bars[-1].close
        upper = np.max(highs[:-1])
        lower = np.min(lows[:-1])
        rng = upper - lower
        if rng == 0:
            return 0.0
        return float((price - (upper + lower) / 2.0) / rng)

    def _mod_volume_profile(self, ctx: Dict, state: Dict) -> float:
        bars = ctx.get("bars", [])
        if len(bars) < 10:
            return 0.0
        vols = np.array([b.volume for b in bars[-10:]])
        if len(vols) < 2:
            return 0.0
        return float(vols[-1] / (np.mean(vols[:-1]) + 1e-10))

    def _mod_trade_intensity(self, ctx: Dict, state: Dict) -> float:
        bars = ctx.get("bars", [])
        if len(bars) < 10:
            return 0.0
        trades = np.array([b.num_trades for b in bars[-10:]])
        return float(np.mean(trades))

    def _mod_flow_imbalance(self, ctx: Dict, state: Dict) -> float:
        bars = ctx.get("bars", [])
        if len(bars) < 5:
            return 0.0
        bd = np.array([b.bid_depth_avg for b in bars[-5:]])
        ad = np.array([b.ask_depth_avg for b in bars[-5:]])
        total = bd + ad
        total = np.where(total == 0, 1, total)
        return float(np.mean((bd - ad) / total))

    def _mod_vol_realized(self, ctx: Dict, state: Dict) -> float:
        bars = ctx.get("bars", [])
        if len(bars) < 20:
            return 0.02
        closes = np.array([b.close for b in bars[-20:]])
        rets = np.diff(np.log(closes + 1e-10))
        return float(np.std(rets) * np.sqrt(252))

    def _mod_vol_garch(self, ctx: Dict, state: Dict) -> float:
        """Simple GARCH(1,1) proxy: exponentially weighted variance."""
        bars = ctx.get("bars", [])
        if len(bars) < 20:
            return 0.02
        closes = np.array([b.close for b in bars[-20:]])
        rets = np.diff(np.log(closes + 1e-10))
        omega, alpha, beta = 0.00001, 0.1, 0.85
        var = np.var(rets)
        for r in rets:
            var = omega + alpha * r * r + beta * var
        return float(np.sqrt(var * 252))

    def _mod_vol_range(self, ctx: Dict, state: Dict) -> float:
        bars = ctx.get("bars", [])
        if len(bars) < 10:
            return 0.02
        ranges = np.array([(b.high - b.low) / (b.close + 1e-10) for b in bars[-10:]])
        return float(np.mean(ranges) * np.sqrt(252))

    def _mod_regime_hmm(self, ctx: Dict, state: Dict) -> int:
        """Simplified 2-state regime via volatility threshold."""
        vol = ctx.get("signals", {}).get("vol_realized", 0.02)
        if isinstance(vol, (int, float)):
            return 0 if vol < 0.25 else 1
        return 0

    def _mod_regime_cluster(self, ctx: Dict, state: Dict) -> int:
        bars = ctx.get("bars", [])
        if len(bars) < 20:
            return 0
        closes = np.array([b.close for b in bars[-20:]])
        rets = np.diff(np.log(closes + 1e-10))
        vol = np.std(rets)
        trend = np.mean(rets)
        # cluster: 0=low-vol-trend, 1=high-vol-trend, 2=low-vol-mean-rev, 3=high-vol-mean-rev
        high_vol = vol > 0.015
        trending = abs(trend) > 0.001
        return int(high_vol) * 2 + int(trending)

    def _mod_regime_hurst(self, ctx: Dict, state: Dict) -> float:
        bars = ctx.get("bars", [])
        if len(bars) < 40:
            return 0.5
        closes = np.array([b.close for b in bars[-40:]])
        rets = np.diff(np.log(closes + 1e-10))
        n = len(rets)
        mean_r = np.mean(rets)
        devs = np.cumsum(rets - mean_r)
        r = np.max(devs) - np.min(devs)
        s = np.std(rets, ddof=1)
        if s == 0 or r == 0 or n <= 1:
            return 0.5
        return float(np.log(r / s) / np.log(n))

    def _mod_dream_mc(self, ctx: Dict, state: Dict) -> Dict[str, float]:
        """Monte Carlo forward paths."""
        bars = ctx.get("bars", [])
        if len(bars) < 20:
            return {"mean_final": 0, "p5": 0, "p95": 0}
        closes = np.array([b.close for b in bars[-20:]])
        rets = np.diff(np.log(closes + 1e-10))
        mu, sigma = np.mean(rets), np.std(rets)
        rng = np.random.default_rng(42)
        n_paths, n_steps = 200, 20
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = closes[-1]
        for t in range(1, n_steps):
            shocks = rng.normal(mu, sigma + 1e-10, n_paths)
            paths[:, t] = paths[:, t - 1] * np.exp(shocks)
        finals = paths[:, -1]
        return {
            "mean_final": float(np.mean(finals)),
            "p5": float(np.percentile(finals, 5)),
            "p95": float(np.percentile(finals, 95)),
        }

    def _mod_dream_bootstrap(self, ctx: Dict, state: Dict) -> Dict[str, float]:
        bars = ctx.get("bars", [])
        if len(bars) < 20:
            return {"mean_final": 0, "p5": 0, "p95": 0}
        closes = np.array([b.close for b in bars[-20:]])
        rets = np.diff(np.log(closes + 1e-10))
        rng = np.random.default_rng(43)
        n_paths, n_steps = 200, 20
        finals = []
        for _ in range(n_paths):
            idx = rng.choice(len(rets), size=n_steps, replace=True)
            sampled = rets[idx]
            final = closes[-1] * np.exp(np.sum(sampled))
            finals.append(final)
        finals_arr = np.array(finals)
        return {
            "mean_final": float(np.mean(finals_arr)),
            "p5": float(np.percentile(finals_arr, 5)),
            "p95": float(np.percentile(finals_arr, 95)),
        }

    def _mod_dream_stress(self, ctx: Dict, state: Dict) -> Dict[str, float]:
        """Stress test: what if vol triples."""
        bars = ctx.get("bars", [])
        if len(bars) < 20:
            return {"stress_p5": 0}
        closes = np.array([b.close for b in bars[-20:]])
        rets = np.diff(np.log(closes + 1e-10))
        mu, sigma = np.mean(rets), np.std(rets)
        rng = np.random.default_rng(44)
        stressed = rng.normal(mu, sigma * 3 + 1e-10, 500)
        final_prices = closes[-1] * np.exp(np.cumsum(stressed.reshape(50, 10), axis=1)[:, -1])
        return {"stress_p5": float(np.percentile(final_prices, 5))}

    def _mod_debate_vote(self, ctx: Dict, state: Dict) -> float:
        """Majority vote across signal modules."""
        sigs = ctx.get("signals", {})
        votes = []
        for k, v in sigs.items():
            if k.startswith("signal_") and isinstance(v, (int, float)):
                votes.append(1 if v > 0 else (-1 if v < 0 else 0))
        if not votes:
            return 0.0
        return float(np.sign(np.sum(votes)))

    def _mod_debate_stack(self, ctx: Dict, state: Dict) -> float:
        """Weighted average of signals."""
        sigs = ctx.get("signals", {})
        weights = {"signal_momentum": 0.4, "signal_mean_rev": 0.3, "signal_breakout": 0.3}
        total_w, total_s = 0.0, 0.0
        for k, w in weights.items():
            v = sigs.get(k, 0)
            if isinstance(v, (int, float)):
                total_s += v * w
                total_w += w
        return total_s / total_w if total_w > 0 else 0.0

    def _mod_debate_bayesian(self, ctx: Dict, state: Dict) -> float:
        """Bayesian updating: each signal updates a prior."""
        sigs = ctx.get("signals", {})
        prior = 0.5  # neutral
        for k, v in sigs.items():
            if k.startswith("signal_") and isinstance(v, (int, float)):
                likelihood = 1.0 / (1.0 + np.exp(-v * 100))
                prior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior) + 1e-10)
        return float(prior - 0.5) * 2.0  # map [0,1] -> [-1,1]

    def _mod_size_kelly(self, ctx: Dict, state: Dict) -> float:
        """Half-Kelly position sizing."""
        debate = ctx.get("debates", {}).get("debate_stack", 0)
        vol = ctx.get("signals", {}).get("vol_realized", 0.02)
        if not isinstance(debate, (int, float)) or not isinstance(vol, (int, float)):
            return 0.0
        if vol == 0:
            return 0.0
        edge = abs(debate) * 0.01
        kelly = edge / (vol ** 2 + 1e-10)
        half_kelly = kelly * 0.5
        return float(min(half_kelly, 20.0))

    def _mod_size_risk_parity(self, ctx: Dict, state: Dict) -> float:
        vol = ctx.get("signals", {}).get("vol_realized", 0.02)
        if not isinstance(vol, (int, float)) or vol == 0:
            return 1.0
        target_risk = 0.01
        return float(min(target_risk / (vol / np.sqrt(252) + 1e-10), 20.0))

    def _mod_size_vol_target(self, ctx: Dict, state: Dict) -> float:
        vol = ctx.get("signals", {}).get("vol_realized", 0.02)
        if not isinstance(vol, (int, float)) or vol == 0:
            return 1.0
        target_annual_vol = 0.15
        return float(min(target_annual_vol / (vol + 1e-10), 5.0))

    def _mod_risk_drawdown(self, ctx: Dict, state: Dict) -> float:
        """Scale factor [0,1] based on current drawdown."""
        equity = ctx.get("equity", 100000)
        hwm = state.get("hwm", equity)
        if equity > hwm:
            state["hwm"] = equity
            hwm = equity
        dd = (hwm - equity) / (hwm + 1e-10)
        if dd > 0.08:
            return 0.0
        elif dd > 0.05:
            return 0.5
        return 1.0

    def _mod_risk_correlation(self, ctx: Dict, state: Dict) -> float:
        """Penalize if signals are too correlated (herding risk)."""
        sigs = ctx.get("signals", {})
        vals = [v for k, v in sigs.items() if k.startswith("signal_") and isinstance(v, (int, float))]
        if len(vals) < 2:
            return 1.0
        signs = [np.sign(v) for v in vals]
        agreement = abs(np.mean(signs))
        if agreement > 0.9:
            return 0.7  # all agree => herding, reduce size
        return 1.0

    def _mod_risk_tail(self, ctx: Dict, state: Dict) -> float:
        """Reduce exposure if tail risk is elevated."""
        bars = ctx.get("bars", [])
        if len(bars) < 30:
            return 1.0
        closes = np.array([b.close for b in bars[-30:]])
        rets = np.diff(np.log(closes + 1e-10))
        kurt = float(np.mean(rets ** 4) / (np.var(rets) ** 2 + 1e-20) - 3.0) if len(rets) > 3 else 0.0
        if kurt > 6:
            return 0.5
        elif kurt > 3:
            return 0.75
        return 1.0

    def _mod_adapt_params(self, ctx: Dict, state: Dict) -> Dict[str, float]:
        """Adjust lookback windows based on regime."""
        regime = ctx.get("signals", {}).get("regime_cluster", 0)
        if regime in (1, 3):  # high vol
            return {"lookback_mult": 0.7}
        return {"lookback_mult": 1.0}

    def _mod_adapt_regime(self, ctx: Dict, state: Dict) -> str:
        regime = ctx.get("signals", {}).get("regime_cluster", 0)
        mapping = {0: "trend", 1: "volatile_trend", 2: "mean_rev", 3: "volatile_mean_rev"}
        return mapping.get(regime, "unknown")

    def _mod_adapt_select(self, ctx: Dict, state: Dict) -> str:
        """Select best strategy template for current regime."""
        regime = ctx.get("adaptations", {}).get("adapt_regime", "trend")
        selection = {
            "trend": "momentum",
            "volatile_trend": "vol_targeting",
            "mean_rev": "mean_reversion",
            "volatile_mean_rev": "rsi",
        }
        return selection.get(regime, "adaptive")

    # --- Main run loop ---

    def run(self, num_bars: int = 500) -> Dict[str, Any]:
        """Run the full 27-module autonomous loop."""
        self._running = True
        capital = self.config.initial_capital
        self.tracker.update_equity(capital)
        bars_list: List[MarketBar] = []

        for bar_idx in range(num_bars):
            if not self._running:
                break

            try:
                bar = self.client.get_latest_bar(self.config.symbol)
                if bar is None:
                    continue
                bars_list.append(bar)
                self._context["bars"] = bars_list
                self._context["bar_idx"] = bar_idx
                self._context["position"] = self.position.quantity

                # Phase 1: signals (modules 0-11)
                for mod in self.modules[:12]:
                    out = mod.tick(self._context)
                    self._context["signals"][mod.name] = out

                # Phase 2: dreams (modules 12-14)
                for mod in self.modules[12:15]:
                    out = mod.tick(self._context)
                    self._context["dreams"][mod.name] = out

                # Phase 3: debate (modules 15-17)
                for mod in self.modules[15:18]:
                    out = mod.tick(self._context)
                    self._context["debates"][mod.name] = out

                # Phase 4: sizing (modules 18-20)
                for mod in self.modules[18:21]:
                    out = mod.tick(self._context)
                    self._context["sizes"][mod.name] = out

                # Phase 5: risk (modules 21-23)
                for mod in self.modules[21:24]:
                    out = mod.tick(self._context)
                    self._context["risks"][mod.name] = out

                # Phase 6: adaptation (modules 24-26)
                for mod in self.modules[24:27]:
                    out = mod.tick(self._context)
                    self._context["adaptations"][mod.name] = out

                # --- Trade decision ---
                consensus = self._context["debates"].get("debate_stack", 0)
                if not isinstance(consensus, (int, float)):
                    consensus = 0
                size_kelly = self._context["sizes"].get("size_kelly", 0)
                size_rp = self._context["sizes"].get("size_risk_parity", 1)
                size_vt = self._context["sizes"].get("size_vol_target", 1)
                for v in (size_kelly, size_rp, size_vt):
                    if not isinstance(v, (int, float)):
                        v = 1.0
                base_size = float(np.mean([
                    size_kelly if isinstance(size_kelly, (int, float)) else 0,
                    size_rp if isinstance(size_rp, (int, float)) else 1,
                    size_vt if isinstance(size_vt, (int, float)) else 1,
                ]))

                risk_dd = self._context["risks"].get("risk_drawdown", 1)
                risk_corr = self._context["risks"].get("risk_correlation", 1)
                risk_tail = self._context["risks"].get("risk_tail", 1)
                for v in (risk_dd, risk_corr, risk_tail):
                    if not isinstance(v, (int, float)):
                        v = 1.0
                risk_scale = float(np.prod([
                    risk_dd if isinstance(risk_dd, (int, float)) else 1,
                    risk_corr if isinstance(risk_corr, (int, float)) else 1,
                    risk_tail if isinstance(risk_tail, (int, float)) else 1,
                ]))

                final_qty = abs(consensus) * base_size * risk_scale
                final_qty = max(0, min(final_qty, self.config.max_position))

                if final_qty >= 1.0 and abs(consensus) > 0.001:
                    side = "buy" if consensus > 0 else "sell"
                    current_dd = self.tracker.max_drawdown
                    allowed, reason = self.risk_mgr.check_order(
                        side, final_qty, bar.close,
                        self.position.quantity, current_dd,
                    )
                    if allowed:
                        self.client.submit_order(self.config.symbol, side, final_qty)

                # process fills
                fills = self.client.get_fills(self.config.symbol, limit=50)
                for fill in fills:
                    rpnl = self.position.update(fill)
                    if rpnl != 0:
                        self.tracker.record_trade(rpnl)
                        self.risk_mgr.update_daily_pnl(rpnl)

                equity = capital + self.position.total_pnl(bar.close)
                self._context["equity"] = equity
                self.tracker.update_equity(equity, bar.timestamp)

            except Exception as exc:
                self._errors.append(f"Bar {bar_idx}: {exc}")
                logger.error("EH test error bar %d: %s", bar_idx, exc)

        self._running = False
        return self.results()

    def stop(self) -> None:
        self._running = False

    def results(self) -> Dict[str, Any]:
        last_price = self._context["bars"][-1].close if self._context["bars"] else 0
        module_stats = {}
        for m in self.modules:
            module_stats[m.name] = {
                "calls": m.call_count,
                "errors": m.error_count,
                "enabled": m.enabled,
                "last_output_type": type(m.last_output).__name__,
            }
        return {
            "bars_processed": self._context["bar_idx"] + 1,
            "position": self.position.quantity,
            "realized_pnl": self.position.realized_pnl,
            "unrealized_pnl": self.position.unrealized_pnl(last_price),
            "total_pnl": self.position.total_pnl(last_price),
            "performance": self.tracker.summary(),
            "module_stats": module_stats,
            "errors": self._errors,
            "total_errors": len(self._errors),
            "modules_with_errors": sum(1 for m in self.modules if m.error_count > 0),
        }
