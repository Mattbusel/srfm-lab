"""
base_adapter.py -- Abstract base class and shared data models for broker adapters.

All broker adapters in SRFM inherit from BrokerAdapter and implement the abstract methods
defined here. Data models are frozen dataclasses to prevent accidental mutation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"       # good till canceled
    IOC = "ioc"       # immediate or cancel
    FOK = "fok"       # fill or kill
    OPG = "opg"       # at open
    CLS = "cls"       # at close


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class AssetClass(str, Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    FUTURES = "futures"
    FOREX = "forex"
    OPTION = "option"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class OrderRequest:
    """Represents an order to be submitted to a broker.

    Fields
    ------
    symbol          -- SRFM canonical symbol (e.g. AAPL, BTC-USD)
    side            -- buy or sell
    qty             -- number of shares/contracts/units
    order_type      -- market, limit, stop, etc.
    time_in_force   -- day, gtc, ioc, fok
    strategy_id     -- identifier for the originating strategy
    client_order_id -- unique idempotency key set by the caller
    price           -- limit price (required for limit/stop_limit orders)
    stop_price      -- stop trigger price (required for stop/stop_limit)
    notional        -- dollar notional (alternative to qty for fractional)
    asset_class     -- equity, crypto, futures (used for routing)
    extended_hours  -- allow pre/after-market fills
    """

    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    time_in_force: TimeInForce
    strategy_id: str
    client_order_id: str
    price: Optional[float] = None
    stop_price: Optional[float] = None
    notional: Optional[float] = None
    asset_class: AssetClass = AssetClass.EQUITY
    extended_hours: bool = False

    def __post_init__(self) -> None:
        if self.qty <= 0 and self.notional is None:
            raise ValueError(f"OrderRequest qty must be positive, got {self.qty}")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.price is None:
            raise ValueError(f"order_type={self.order_type} requires a price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"order_type={self.order_type} requires a stop_price")


@dataclass
class OrderResult:
    """Represents the broker's response after submitting an order.

    Fields
    ------
    order_id        -- broker-assigned order ID
    client_order_id -- echoed from OrderRequest for correlation
    status          -- current status of the order
    submitted_at    -- UTC timestamp when the broker acknowledged the order
    message         -- human-readable status message or error description
    filled_qty      -- quantity filled so far (0 for new submissions)
    avg_fill_price  -- volume-weighted average fill price (None if not yet filled)
    """

    order_id: str
    client_order_id: str
    status: OrderStatus
    submitted_at: datetime
    message: str = ""
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None


@dataclass
class Position:
    """Represents a current open position.

    Fields
    ------
    symbol          -- SRFM canonical symbol
    qty             -- absolute quantity held (positive)
    avg_entry_price -- average cost basis per unit
    market_value    -- current mark-to-market value
    unrealized_pnl  -- unrealized profit/loss at current market price
    side            -- long or short
    cost_basis      -- total cost at entry (qty * avg_entry_price)
    asset_class     -- equity, crypto, futures
    """

    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float
    side: PositionSide
    cost_basis: float = 0.0
    asset_class: AssetClass = AssetClass.EQUITY

    def __post_init__(self) -> None:
        if self.cost_basis == 0.0 and self.qty != 0.0:
            self.cost_basis = self.qty * self.avg_entry_price

    @property
    def return_pct(self) -> float:
        """Unrealized return as a percentage of cost basis."""
        if self.cost_basis == 0.0:
            return 0.0
        return (self.unrealized_pnl / abs(self.cost_basis)) * 100.0


@dataclass
class AccountInfo:
    """Snapshot of account-level financial state.

    Fields
    ------
    equity          -- total account equity (cash + market value of positions)
    cash            -- settled cash available
    buying_power    -- purchasing power including margin
    margin_used     -- margin currently in use
    leverage        -- current leverage ratio (equity / net_liquidation)
    day_pnl         -- realized + unrealized P&L since midnight UTC
    total_pnl       -- lifetime realized P&L
    currency        -- base currency (USD, USDT, etc.)
    """

    equity: float
    cash: float
    buying_power: float
    margin_used: float
    leverage: float
    day_pnl: float = 0.0
    total_pnl: float = 0.0
    currency: str = "USD"

    @property
    def net_exposure(self) -> float:
        """Net dollar exposure = equity - cash."""
        return self.equity - self.cash


@dataclass
class Fill:
    """Represents a single execution event (partial or full fill).

    Fields
    ------
    fill_id     -- broker-assigned fill identifier
    order_id    -- broker-assigned order identifier
    symbol      -- SRFM canonical symbol
    side        -- buy or sell
    qty         -- quantity filled in this event
    price       -- execution price for this fill
    timestamp   -- UTC timestamp of fill
    venue       -- execution venue (e.g. NYSE, NASDAQ, BINANCE)
    commission  -- commission charged for this fill
    liquidity   -- maker or taker
    """

    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    timestamp: datetime
    venue: str = ""
    commission: float = 0.0
    liquidity: str = "taker"

    @property
    def gross_value(self) -> float:
        """Dollar value of this fill before commission."""
        return self.qty * self.price

    @property
    def net_value(self) -> float:
        """Dollar value after commission."""
        return self.gross_value - self.commission


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BrokerAdapterError(Exception):
    """Base exception for all broker adapter errors."""
    pass


class AuthenticationError(BrokerAdapterError):
    """Raised when API credentials are invalid or expired."""
    pass


class RateLimitError(BrokerAdapterError):
    """Raised when the broker rate limit is exceeded."""

    def __init__(self, message: str, retry_after_s: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class OrderRejectedError(BrokerAdapterError):
    """Raised when the broker rejects an order."""

    def __init__(self, message: str, order_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.order_id = order_id


class ConnectionError(BrokerAdapterError):
    """Raised when the adapter cannot connect to the broker."""
    pass


class InsufficientFundsError(BrokerAdapterError):
    """Raised when there is not enough buying power to submit an order."""
    pass


class SymbolNotFoundError(BrokerAdapterError):
    """Raised when the broker does not recognize the given symbol."""
    pass


# ---------------------------------------------------------------------------
# Circuit breaker (simple state machine used by adapters)
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    CLOSED = "closed"       # normal operation
    OPEN = "open"           # failing, blocking requests
    HALF_OPEN = "half_open" # testing recovery


class CircuitBreaker:
    """Simple circuit breaker to protect adapters from cascading failures.

    States
    ------
    CLOSED    -- all requests pass through
    OPEN      -- requests are blocked; after recovery_timeout_s, moves to HALF_OPEN
    HALF_OPEN -- one probe request allowed; success -> CLOSED, failure -> OPEN

    Parameters
    ----------
    failure_threshold   -- number of consecutive failures to open the circuit
    recovery_timeout_s  -- seconds to wait before attempting recovery
    name                -- identifier used in log messages
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        name: str = "circuit",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.name = name
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: Optional[datetime] = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout_s:
                    logger.info("CircuitBreaker[%s] transitioning OPEN -> HALF_OPEN", self.name)
                    self._state = CircuitState.HALF_OPEN
        return self._state

    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def record_success(self) -> None:
        if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            logger.info("CircuitBreaker[%s] recovered -> CLOSED", self.name)
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = datetime.utcnow()
        if self._failure_count >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                logger.warning(
                    "CircuitBreaker[%s] opening after %d failures",
                    self.name,
                    self._failure_count,
                )
            self._state = CircuitState.OPEN

    def call_allowed(self) -> bool:
        """Return True if a request should be allowed through."""
        s = self.state
        if s == CircuitState.CLOSED:
            return True
        if s == CircuitState.HALF_OPEN:
            return True
        return False


# ---------------------------------------------------------------------------
# Abstract broker adapter
# ---------------------------------------------------------------------------


class BrokerAdapter(ABC):
    """Abstract base class for all SRFM broker adapters.

    Concrete subclasses must implement every abstract method. The adapter
    is responsible for translating SRFM-internal data models to and from
    broker-specific API formats, handling authentication, rate limiting,
    and connection lifecycle.

    All I/O methods are async. Implementations should use aiohttp for HTTP
    and the appropriate WebSocket library for streaming.

    Parameters
    ----------
    name            -- human-readable adapter name used in logs
    asset_class     -- primary asset class handled by this adapter
    circuit_breaker -- optional external circuit breaker; one is created if not provided
    """

    def __init__(
        self,
        name: str,
        asset_class: AssetClass = AssetClass.EQUITY,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self.name = name
        self.asset_class = asset_class
        self._cb = circuit_breaker or CircuitBreaker(name=name)
        self._connected: bool = False
        self.logger = logging.getLogger(f"srfm.adapter.{name}")

    # ------------------------------------------------------------------
    # Abstract interface -- must be implemented by all adapters
    # ------------------------------------------------------------------

    @abstractmethod
    async def submit_order(self, order: OrderRequest) -> OrderResult:
        """Submit an order to the broker.

        Parameters
        ----------
        order -- fully-populated OrderRequest

        Returns
        -------
        OrderResult with broker-assigned order_id and initial status.

        Raises
        ------
        AuthenticationError     -- invalid credentials
        InsufficientFundsError  -- not enough buying power
        OrderRejectedError      -- broker rejected the order
        RateLimitError          -- too many requests
        BrokerAdapterError      -- other broker-side errors
        """
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by broker order ID.

        Returns True if successfully canceled, False if already filled/expired.

        Raises
        ------
        AuthenticationError  -- invalid credentials
        BrokerAdapterError   -- other errors
        """
        ...

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Return current position for symbol, or None if flat."""
        ...

    @abstractmethod
    async def get_all_positions(self) -> Dict[str, Position]:
        """Return all open positions keyed by SRFM symbol."""
        ...

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Return current account snapshot."""
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Return current status of an order by broker order ID."""
        ...

    @abstractmethod
    async def get_recent_fills(self, n: int = 100) -> List[Fill]:
        """Return up to n most recent fills, newest first."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the adapter has an active connection to the broker."""
        ...

    @abstractmethod
    async def test_connection(self) -> bool:
        """Perform a lightweight connectivity check.

        Returns True on success, False on failure. Should not raise.
        """
        ...

    # ------------------------------------------------------------------
    # Optional streaming interface -- adapters override as applicable
    # ------------------------------------------------------------------

    async def stream_fills(self, callback: Callable[[Fill], None]) -> None:
        """Stream real-time fill events via WebSocket.

        Default implementation raises NotImplementedError. Adapters that
        support streaming should override this method.
        """
        raise NotImplementedError(f"{self.name} does not support fill streaming")

    async def stream_quotes(
        self,
        symbols: List[str],
        callback: Callable[[Dict], None],
    ) -> None:
        """Stream real-time quotes for a list of symbols.

        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(f"{self.name} does not support quote streaming")

    # ------------------------------------------------------------------
    # Circuit breaker helpers
    # ------------------------------------------------------------------

    def _check_circuit(self) -> None:
        """Raise ConnectionError if the circuit breaker is open."""
        if not self._cb.call_allowed():
            raise ConnectionError(
                f"Adapter {self.name} circuit breaker is OPEN -- requests blocked"
            )

    def _on_success(self) -> None:
        self._cb.record_success()

    def _on_failure(self, exc: Exception) -> None:
        self._cb.record_failure()
        self.logger.error("Adapter %s recorded failure: %s", self.name, exc)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"asset_class={self.asset_class.value}, "
            f"connected={self.is_connected()}, "
            f"circuit={self._cb.state.value})"
        )
