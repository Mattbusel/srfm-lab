"""
Order type definitions for SRFM execution layer.
Covers market, limit, stop, TWAP, VWAP, and iceberg order types.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OrderStatus(Enum):
    """Lifecycle states for any order."""
    NEW = "NEW"
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(str, Enum):
    """Standard time-in-force values."""
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


# ---------------------------------------------------------------------------
# Fill record
# ---------------------------------------------------------------------------

@dataclass
class Fill:
    """
    Represents a single execution against an order.

    commission_bps is expressed as basis points (1 bps = 0.01%).
    """
    fill_id: str
    order_id: str
    symbol: str
    side: str              # "buy" or "sell"
    qty: float
    price: float
    timestamp: datetime
    venue: str
    commission_bps: float  # e.g. 1.0 means 1 bps round-trip

    @property
    def notional(self) -> float:
        return self.qty * self.price

    @property
    def commission_dollars(self) -> float:
        return self.notional * (self.commission_bps / 10_000.0)

    def __repr__(self) -> str:
        return (
            f"Fill(fill_id={self.fill_id!r}, order_id={self.order_id!r}, "
            f"symbol={self.symbol!r}, side={self.side!r}, qty={self.qty}, "
            f"price={self.price}, venue={self.venue!r})"
        )


# ---------------------------------------------------------------------------
# Base order
# ---------------------------------------------------------------------------

@dataclass
class BaseOrder:
    """
    Common fields shared by all order types.

    order_id is a UUID string assigned at creation time.
    signal_strength is a float in [-1.0, 1.0] from the upstream signal.
    filled_qty tracks cumulative fills; avg_fill_price is the VWAP of fills.
    """
    order_id: str
    symbol: str
    side: str              # "buy" or "sell"
    qty: float             # target quantity (shares / contracts)
    strategy_id: str
    signal_strength: float  # in [-1.0, 1.0]
    created_at: datetime
    status: OrderStatus = field(default=OrderStatus.NEW)

    # mutable fill tracking -- updated by order book tracker
    filled_qty: float = field(default=0.0)
    avg_fill_price: float = field(default=0.0)
    fills: List[Fill] = field(default_factory=list)

    # optional metadata
    notes: str = field(default="")

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    @property
    def is_done(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def is_active(self) -> bool:
        return self.status in (
            OrderStatus.NEW,
            OrderStatus.PENDING,
            OrderStatus.PARTIAL,
        )

    def apply_fill(self, fill: Fill) -> None:
        """Update filled_qty and avg_fill_price after receiving a Fill."""
        if fill.order_id != self.order_id:
            raise ValueError(
                f"Fill order_id {fill.order_id!r} does not match order "
                f"{self.order_id!r}"
            )
        prev_notional = self.filled_qty * self.avg_fill_price
        new_notional = fill.qty * fill.price
        self.filled_qty += fill.qty
        if self.filled_qty > 0:
            self.avg_fill_price = (prev_notional + new_notional) / self.filled_qty
        self.fills.append(fill)
        if self.filled_qty >= self.qty - 1e-9:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL

    def validate(self) -> None:
        """Raise ValueError if order fields are invalid."""
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if self.qty <= 0:
            raise ValueError(f"qty must be positive, got {self.qty}")
        if not (-1.0 <= self.signal_strength <= 1.0):
            raise ValueError(
                f"signal_strength must be in [-1,1], got {self.signal_strength}"
            )
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.strategy_id:
            raise ValueError("strategy_id must not be empty")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(order_id={self.order_id!r}, "
            f"symbol={self.symbol!r}, side={self.side!r}, qty={self.qty}, "
            f"status={self.status.value!r})"
        )


# ---------------------------------------------------------------------------
# Concrete order types
# ---------------------------------------------------------------------------

@dataclass
class MarketOrder(BaseOrder):
    """
    Market order -- executes immediately at best available price.
    No price guarantee; used for high-urgency fills.
    """
    order_type: str = field(default="MARKET", init=False)

    def validate(self) -> None:
        super().validate()


@dataclass
class LimitOrder(BaseOrder):
    """
    Limit order -- executes at limit_price or better.
    time_in_force controls expiration behavior.
    """
    order_type: str = field(default="LIMIT", init=False)
    limit_price: float = field(default=0.0)
    time_in_force: str = field(default=TimeInForce.DAY)

    def validate(self) -> None:
        super().validate()
        if self.limit_price <= 0:
            raise ValueError(
                f"limit_price must be positive, got {self.limit_price}"
            )
        valid_tif = {t.value for t in TimeInForce}
        tif_val = (
            self.time_in_force.value
            if isinstance(self.time_in_force, TimeInForce)
            else self.time_in_force
        )
        if tif_val not in valid_tif:
            raise ValueError(
                f"time_in_force must be one of {valid_tif}, got {tif_val!r}"
            )


@dataclass
class StopOrder(BaseOrder):
    """
    Stop or stop-limit order.

    If limit_price is None this is a plain stop (market) order that triggers
    when the last trade crosses stop_price.  If limit_price is set this
    becomes a stop-limit order.
    """
    order_type: str = field(default="STOP", init=False)
    stop_price: float = field(default=0.0)
    limit_price: Optional[float] = field(default=None)

    @property
    def is_stop_limit(self) -> bool:
        return self.limit_price is not None

    def validate(self) -> None:
        super().validate()
        if self.stop_price <= 0:
            raise ValueError(
                f"stop_price must be positive, got {self.stop_price}"
            )
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError(
                f"limit_price must be positive when set, got {self.limit_price}"
            )


@dataclass
class TWAPOrder(BaseOrder):
    """
    TWAP algorithmic order.

    The TWAP engine divides total qty into n_slices equal child limit orders
    spaced evenly between start_time and end_time.
    slice_interval_s is derived from the time window and n_slices.
    child_orders is populated by the engine as slices are created.
    """
    order_type: str = field(default="TWAP", init=False)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime = field(default_factory=datetime.utcnow)
    n_slices: int = field(default=10)
    slice_interval_s: float = field(default=0.0)
    child_orders: List[LimitOrder] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_slices > 0:
            duration_s = (self.end_time - self.start_time).total_seconds()
            if duration_s > 0:
                self.slice_interval_s = duration_s / self.n_slices
        else:
            self.slice_interval_s = 0.0

    @property
    def slice_qty(self) -> float:
        """Quantity per slice -- may not be integer for fractional shares."""
        if self.n_slices <= 0:
            return self.qty
        return self.qty / self.n_slices

    def validate(self) -> None:
        super().validate()
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        if self.n_slices < 1:
            raise ValueError(f"n_slices must be >= 1, got {self.n_slices}")


@dataclass
class VWAPOrder(BaseOrder):
    """
    VWAP algorithmic order.

    volume_curve is a list of fractional weights (must sum to 1.0) one per
    time bucket.  target_participation_rate is the fraction of bucket volume
    we target (e.g. 0.10 means we want to be 10% of market volume).
    """
    order_type: str = field(default="VWAP", init=False)
    volume_curve: List[float] = field(default_factory=list)
    target_participation_rate: float = field(default=0.10)

    def validate(self) -> None:
        super().validate()
        if not self.volume_curve:
            raise ValueError("volume_curve must not be empty")
        total = sum(self.volume_curve)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"volume_curve weights must sum to 1.0, got {total:.6f}"
            )
        if not (0.0 < self.target_participation_rate <= 1.0):
            raise ValueError(
                f"target_participation_rate must be in (0,1], "
                f"got {self.target_participation_rate}"
            )

    def qty_for_bucket(self, bucket_index: int) -> float:
        """Return the target qty for a given volume bucket."""
        if bucket_index < 0 or bucket_index >= len(self.volume_curve):
            raise IndexError(f"bucket_index {bucket_index} out of range")
        return self.qty * self.volume_curve[bucket_index]


@dataclass
class IcebergOrder(BaseOrder):
    """
    Iceberg (reserve) order.

    Only display_qty shares are shown on the book at once.  As each displayed
    slice fills the engine re-submits a new display order until total_qty is
    exhausted.  display_qty is randomized slightly at runtime to avoid
    pattern detection.
    """
    order_type: str = field(default="ICEBERG", init=False)
    total_qty: float = field(default=0.0)
    display_qty: float = field(default=0.0)
    shown_orders: List[LimitOrder] = field(default_factory=list)

    def __post_init__(self) -> None:
        # keep qty and total_qty in sync -- qty is the display slice
        if self.total_qty == 0.0 and self.qty > 0:
            self.total_qty = self.qty

    @property
    def hidden_qty(self) -> float:
        return max(0.0, self.total_qty - self.filled_qty - self.display_qty)

    @property
    def n_slices_estimated(self) -> int:
        if self.display_qty <= 0:
            return 0
        import math
        return math.ceil(self.total_qty / self.display_qty)

    def validate(self) -> None:
        super().validate()
        if self.total_qty <= 0:
            raise ValueError(f"total_qty must be positive, got {self.total_qty}")
        if self.display_qty <= 0:
            raise ValueError(f"display_qty must be positive, got {self.display_qty}")
        if self.display_qty > self.total_qty:
            raise ValueError(
                f"display_qty ({self.display_qty}) cannot exceed "
                f"total_qty ({self.total_qty})"
            )


# ---------------------------------------------------------------------------
# Order factory
# ---------------------------------------------------------------------------

class OrderFactory:
    """
    Convenience factory for creating typed orders with sensible defaults.

    All methods assign a fresh UUID order_id and set created_at to utcnow().
    The caller can override fields after creation if needed.
    """

    @staticmethod
    def _base_kwargs(
        symbol: str,
        side: str,
        qty: float,
        strategy_id: str,
        signal_strength: float = 0.5,
    ) -> dict:
        return {
            "order_id": str(uuid.uuid4()),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "strategy_id": strategy_id,
            "signal_strength": signal_strength,
            "created_at": datetime.utcnow(),
            "status": OrderStatus.NEW,
        }

    @staticmethod
    def create_market(
        symbol: str,
        side: str,
        qty: float,
        strategy_id: str,
        signal_strength: float = 0.5,
    ) -> MarketOrder:
        """Create a market order for immediate execution."""
        kwargs = OrderFactory._base_kwargs(
            symbol, side, qty, strategy_id, signal_strength
        )
        order = MarketOrder(**kwargs)
        order.validate()
        return order

    @staticmethod
    def create_limit(
        symbol: str,
        side: str,
        qty: float,
        price: float,
        strategy_id: str,
        tif: str = "DAY",
        signal_strength: float = 0.5,
    ) -> LimitOrder:
        """Create a limit order with the given price and time-in-force."""
        kwargs = OrderFactory._base_kwargs(
            symbol, side, qty, strategy_id, signal_strength
        )
        order = LimitOrder(limit_price=price, time_in_force=tif, **kwargs)
        order.validate()
        return order

    @staticmethod
    def create_stop(
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        strategy_id: str,
        limit_price: Optional[float] = None,
        signal_strength: float = 0.5,
    ) -> StopOrder:
        """Create a stop or stop-limit order."""
        kwargs = OrderFactory._base_kwargs(
            symbol, side, qty, strategy_id, signal_strength
        )
        order = StopOrder(
            stop_price=stop_price, limit_price=limit_price, **kwargs
        )
        order.validate()
        return order

    @staticmethod
    def create_twap(
        symbol: str,
        side: str,
        qty: float,
        start: datetime,
        end: datetime,
        n_slices: int,
        strategy_id: str,
        signal_strength: float = 0.5,
    ) -> TWAPOrder:
        """
        Create a TWAP order.  slice_interval_s is computed in __post_init__.
        """
        kwargs = OrderFactory._base_kwargs(
            symbol, side, qty, strategy_id, signal_strength
        )
        order = TWAPOrder(
            start_time=start,
            end_time=end,
            n_slices=n_slices,
            **kwargs,
        )
        order.validate()
        return order

    @staticmethod
    def create_vwap(
        symbol: str,
        side: str,
        qty: float,
        strategy_id: str,
        target_pct: float = 0.10,
        volume_curve: Optional[List[float]] = None,
        signal_strength: float = 0.5,
    ) -> VWAPOrder:
        """
        Create a VWAP order.

        If volume_curve is not supplied the 48-bucket U-shaped equity profile
        from twap_engine.EQUITY_VOLUME_PROFILE is used.
        """
        if volume_curve is None:
            from .twap_engine import EQUITY_VOLUME_PROFILE
            volume_curve = EQUITY_VOLUME_PROFILE
        kwargs = OrderFactory._base_kwargs(
            symbol, side, qty, strategy_id, signal_strength
        )
        order = VWAPOrder(
            volume_curve=volume_curve,
            target_participation_rate=target_pct,
            **kwargs,
        )
        order.validate()
        return order

    @staticmethod
    def create_iceberg(
        symbol: str,
        side: str,
        qty: float,
        strategy_id: str,
        display_pct: float = 0.10,
        limit_price: float = 0.0,
        signal_strength: float = 0.5,
    ) -> IcebergOrder:
        """
        Create an iceberg order where display_qty = qty * display_pct.

        limit_price is the price at which child display orders will be placed.
        """
        display_qty = max(1.0, round(qty * display_pct, 2))
        kwargs = OrderFactory._base_kwargs(
            symbol, side, qty, strategy_id, signal_strength
        )
        order = IcebergOrder(
            total_qty=qty,
            display_qty=display_qty,
            **kwargs,
        )
        order.validate()
        return order


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def make_fill(
    order: BaseOrder,
    qty: float,
    price: float,
    venue: str = "INTERNAL",
    commission_bps: float = 1.0,
) -> Fill:
    """Convenience function to construct a Fill for a given order."""
    return Fill(
        fill_id=str(uuid.uuid4()),
        order_id=order.order_id,
        symbol=order.symbol,
        side=order.side,
        qty=qty,
        price=price,
        timestamp=datetime.utcnow(),
        venue=venue,
        commission_bps=commission_bps,
    )
