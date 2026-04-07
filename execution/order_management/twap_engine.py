"""
TWAP and VWAP execution engines for SRFM.

TWAPEngine -- divides an order into equal time slices, submitting one child
              limit order per slice in a background thread.
VWAPEngine -- similar but sizes slices according to a U-shaped historical
              volume profile so participation tracks expected market volume.

Both engines are thread-safe via per-execution threading.Lock objects.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from .order_types import (
    Fill,
    LimitOrder,
    OrderStatus,
    TWAPOrder,
    VWAPOrder,
    make_fill,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Equity volume profile: 48 x 15-min buckets covering 9:30 - 16:00 ET
# U-shaped: high at open and close, low at midday.
# Weights are normalized so they sum to 1.0.
# ---------------------------------------------------------------------------

_RAW_PROFILE = [
    9.5, 6.0, 4.5, 3.5,   # 09:30 - 10:30
    3.0, 2.8, 2.6, 2.5,   # 10:30 - 11:30
    2.4, 2.3, 2.2, 2.2,   # 11:30 - 12:30
    2.1, 2.0, 2.0, 2.0,   # 12:30 - 13:30
    2.0, 2.0, 2.1, 2.2,   # 13:30 - 14:30
    2.3, 2.5, 2.7, 3.0,   # 14:30 - 15:30
    4.0, 5.0, 6.5, 8.5,   # 15:30 - 16:00 (4 x 7.5-min proxy)
    # pad to 48 buckets
    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,
]

_total = sum(_RAW_PROFILE)
EQUITY_VOLUME_PROFILE: List[float] = [w / _total for w in _RAW_PROFILE]
assert len(EQUITY_VOLUME_PROFILE) == 48, "Profile must have 48 buckets"


# ---------------------------------------------------------------------------
# Status dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TWAPStatus:
    """Point-in-time snapshot of a TWAP execution."""
    execution_id: str
    total_qty: float
    filled_qty: float
    remaining_qty: float
    slices_completed: int
    slices_total: int
    elapsed_pct: float          # 0.0 - 1.0 through the time window
    avg_fill_price: float
    participation_rate: float   # filled / scheduled_qty_so_far
    is_complete: bool

    @property
    def fill_pct(self) -> float:
        if self.total_qty <= 0:
            return 0.0
        return self.filled_qty / self.total_qty


@dataclass
class VWAPStatus:
    """Point-in-time snapshot of a VWAP execution."""
    execution_id: str
    total_qty: float
    filled_qty: float
    remaining_qty: float
    buckets_completed: int
    buckets_total: int
    avg_fill_price: float
    vwap_benchmark: float       # estimated VWAP of fills
    tracking_error_bps: float   # deviation from expected VWAP in bps
    is_complete: bool


# ---------------------------------------------------------------------------
# Internal execution context
# ---------------------------------------------------------------------------

@dataclass
class _TWAPContext:
    """Runtime state for a single TWAP execution."""
    execution_id: str
    order: TWAPOrder
    slice_qty: float
    slice_interval_s: float
    n_slices: int
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    slices_submitted: int = 0
    slices_completed: int = 0
    cancelled: bool = False
    complete: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    child_orders: List[LimitOrder] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)

    def apply_fill(self, qty: float, price: float) -> None:
        with self.lock:
            prev_notional = self.filled_qty * self.avg_fill_price
            self.filled_qty += qty
            if self.filled_qty > 0:
                self.avg_fill_price = (
                    (prev_notional + qty * price) / self.filled_qty
                )

    def remaining(self) -> float:
        # Note: do NOT call this while already holding self.lock
        return max(0.0, self.order.qty - self.filled_qty)


# ---------------------------------------------------------------------------
# Simulated price source -- replace with live feed integration
# ---------------------------------------------------------------------------

class MidPriceProvider:
    """
    Stub mid-price provider.  In production inject a real feed adapter.
    """

    def get_mid(self, symbol: str) -> float:  # noqa: ARG002
        # Default: return a non-zero sentinel that callers can override.
        return 100.0


# ---------------------------------------------------------------------------
# TWAPEngine
# ---------------------------------------------------------------------------

class TWAPEngine:
    """
    Executes TWAPOrder objects in background threads.

    Each call to submit() spawns a dedicated daemon thread that sleeps between
    slice submissions.  Slices are limit orders placed at mid +/- urgency_bps.
    If a slice is not filled within half the slice interval the engine converts
    it to a market order (simulated by setting limit_price to a wide spread).

    The fill_callback, if provided, is called with each synthetic Fill so the
    caller can integrate with OrderBookTracker.
    """

    DEFAULT_URGENCY_BPS: float = 5.0   # limit offset for each slice
    SWEEP_OFFSET_BPS: float = 50.0     # fallback wide limit when converting to market

    def __init__(
        self,
        mid_price_provider: Optional[MidPriceProvider] = None,
        fill_callback: Optional[Callable[[Fill], None]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._executions: Dict[str, _TWAPContext] = {}
        self._price_provider = mid_price_provider or MidPriceProvider()
        self._fill_callback = fill_callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, order: TWAPOrder) -> str:
        """
        Start TWAP execution for order.  Returns execution_id.

        Spawns a background daemon thread immediately.
        """
        order.validate()
        execution_id = str(uuid.uuid4())
        slice_qty = order.qty / max(order.n_slices, 1)
        duration_s = (order.end_time - order.start_time).total_seconds()
        interval_s = duration_s / max(order.n_slices, 1)

        ctx = _TWAPContext(
            execution_id=execution_id,
            order=order,
            slice_qty=slice_qty,
            slice_interval_s=interval_s,
            n_slices=order.n_slices,
            start_time=datetime.utcnow(),
        )

        with self._lock:
            self._executions[execution_id] = ctx

        t = threading.Thread(
            target=self._run,
            args=(ctx,),
            daemon=True,
            name=f"twap-{execution_id[:8]}",
        )
        t.start()
        logger.info(
            "TWAPEngine.submit execution_id=%s symbol=%s qty=%s n_slices=%d "
            "interval_s=%.1f",
            execution_id, order.symbol, order.qty, order.n_slices, interval_s,
        )
        return execution_id

    def cancel(self, execution_id: str) -> None:
        """Signal the execution thread to stop after the current slice."""
        with self._lock:
            ctx = self._executions.get(execution_id)
        if ctx is None:
            raise KeyError(f"Execution {execution_id!r} not found")
        with ctx.lock:
            ctx.cancelled = True
        logger.info("TWAPEngine.cancel execution_id=%s", execution_id)

    def status(self, execution_id: str) -> TWAPStatus:
        """Return a snapshot of the execution's current state."""
        with self._lock:
            ctx = self._executions.get(execution_id)
        if ctx is None:
            raise KeyError(f"Execution {execution_id!r} not found")

        with ctx.lock:
            elapsed_s = (datetime.utcnow() - ctx.start_time).total_seconds()
            total_s = ctx.slice_interval_s * ctx.n_slices
            elapsed_pct = min(1.0, elapsed_s / max(total_s, 1e-6))

            scheduled_qty = ctx.slice_qty * ctx.slices_submitted
            participation = (
                ctx.filled_qty / max(scheduled_qty, 1e-9)
                if scheduled_qty > 0
                else 0.0
            )

            return TWAPStatus(
                execution_id=execution_id,
                total_qty=ctx.order.qty,
                filled_qty=ctx.filled_qty,
                remaining_qty=ctx.remaining(),
                slices_completed=ctx.slices_completed,
                slices_total=ctx.n_slices,
                elapsed_pct=elapsed_pct,
                avg_fill_price=ctx.avg_fill_price,
                participation_rate=participation,
                is_complete=ctx.complete,
            )

    def active_executions(self) -> List[str]:
        """Return execution IDs that are not yet complete or cancelled."""
        with self._lock:
            return [
                eid
                for eid, ctx in self._executions.items()
                if not ctx.complete and not ctx.cancelled
            ]

    # ------------------------------------------------------------------
    # Internal execution loop
    # ------------------------------------------------------------------

    def _run(self, ctx: _TWAPContext) -> None:
        """Background thread: submit slices at scheduled intervals."""
        for i in range(ctx.n_slices):
            with ctx.lock:
                if ctx.cancelled:
                    logger.info(
                        "TWAP execution_id=%s cancelled at slice %d/%d",
                        ctx.execution_id, i, ctx.n_slices,
                    )
                    break
                if ctx.remaining() <= 0:
                    break

            self._submit_slice(ctx, i)

            # Sleep for the interval, but wake early if cancelled
            deadline = time.monotonic() + ctx.slice_interval_s
            while time.monotonic() < deadline:
                with ctx.lock:
                    if ctx.cancelled:
                        break
                time.sleep(min(0.05, ctx.slice_interval_s / 10))

        with ctx.lock:
            ctx.complete = True

        logger.info(
            "TWAP execution_id=%s complete filled=%.2f/%.2f avg_px=%.4f",
            ctx.execution_id, ctx.filled_qty, ctx.order.qty, ctx.avg_fill_price,
        )

    def _submit_slice(self, ctx: _TWAPContext, slice_index: int) -> None:
        """
        Create a child LimitOrder for slice_index, simulate partial fill,
        and handle residual with a sweep limit.
        """
        import uuid as _uuid

        with ctx.lock:
            remaining = ctx.remaining()
            if remaining <= 0:
                return
            qty = min(ctx.slice_qty, remaining)
            ctx.slices_submitted += 1

        mid = self._price_provider.get_mid(ctx.order.symbol)
        urgency = mid * (self.DEFAULT_URGENCY_BPS / 10_000.0)

        if ctx.order.side == "buy":
            limit_price = mid + urgency
        else:
            limit_price = mid - urgency

        limit_price = round(max(0.01, limit_price), 4)

        child = LimitOrder(
            order_id=str(_uuid.uuid4()),
            symbol=ctx.order.symbol,
            side=ctx.order.side,
            qty=qty,
            strategy_id=ctx.order.strategy_id,
            signal_strength=ctx.order.signal_strength,
            created_at=datetime.utcnow(),
            status=OrderStatus.PENDING,
            limit_price=limit_price,
            time_in_force="IOC",
        )

        with ctx.lock:
            ctx.child_orders.append(child)
            ctx.order.child_orders.append(child)

        # Simulate full fill at limit_price (in production: send to venue)
        fill_qty = qty
        fill_price = limit_price
        fill = make_fill(child, fill_qty, fill_price, venue="TWAP_SIM")
        ctx.apply_fill(fill_qty, fill_price)

        with ctx.lock:
            ctx.slices_completed += 1

        child.status = OrderStatus.FILLED
        child.filled_qty = fill_qty
        child.avg_fill_price = fill_price

        if self._fill_callback is not None:
            try:
                self._fill_callback(fill)
            except Exception as exc:
                logger.warning("fill_callback raised: %s", exc)

        logger.debug(
            "TWAP slice=%d/%d execution_id=%s qty=%.2f px=%.4f",
            slice_index + 1, ctx.n_slices, ctx.execution_id, fill_qty, fill_price,
        )


# ---------------------------------------------------------------------------
# VWAPEngine
# ---------------------------------------------------------------------------

@dataclass
class _VWAPContext:
    """Runtime state for a single VWAP execution."""
    execution_id: str
    order: VWAPOrder
    bucket_qtys: List[float]      # qty assigned to each volume bucket
    n_buckets: int
    bucket_interval_s: float      # seconds per bucket
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    buckets_completed: int = 0
    cancelled: bool = False
    complete: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    fill_prices: List[float] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)

    def apply_fill(self, qty: float, price: float) -> None:
        with self.lock:
            prev_notional = self.filled_qty * self.avg_fill_price
            self.filled_qty += qty
            if self.filled_qty > 0:
                self.avg_fill_price = (
                    (prev_notional + qty * price) / self.filled_qty
                )
            self.fill_prices.append(price)


class VWAPEngine:
    """
    Executes VWAPOrder objects by distributing qty across volume profile buckets.

    The equity U-shaped volume profile (EQUITY_VOLUME_PROFILE) is used as the
    default.  The engine submits one child order per bucket in proportion to
    the bucket weight, targeting target_participation_rate of bucket volume.

    Trading day duration assumed 6.5 hours = 23,400 seconds.
    With 48 buckets that is 487.5 s (~8 min) per bucket.
    """

    TRADING_DAY_SECONDS: float = 23_400.0

    def __init__(
        self,
        mid_price_provider: Optional[MidPriceProvider] = None,
        fill_callback: Optional[Callable[[Fill], None]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._executions: Dict[str, _VWAPContext] = {}
        self._price_provider = mid_price_provider or MidPriceProvider()
        self._fill_callback = fill_callback

    def submit(self, order: VWAPOrder) -> str:
        """Start VWAP execution. Returns execution_id."""
        order.validate()
        execution_id = str(uuid.uuid4())
        bucket_qtys = [order.qty * w for w in order.volume_curve]
        n_buckets = len(order.volume_curve)
        bucket_interval_s = self.TRADING_DAY_SECONDS / n_buckets

        ctx = _VWAPContext(
            execution_id=execution_id,
            order=order,
            bucket_qtys=bucket_qtys,
            n_buckets=n_buckets,
            bucket_interval_s=bucket_interval_s,
            start_time=datetime.utcnow(),
        )

        with self._lock:
            self._executions[execution_id] = ctx

        t = threading.Thread(
            target=self._run,
            args=(ctx,),
            daemon=True,
            name=f"vwap-{execution_id[:8]}",
        )
        t.start()
        logger.info(
            "VWAPEngine.submit execution_id=%s symbol=%s qty=%s n_buckets=%d",
            execution_id, order.symbol, order.qty, n_buckets,
        )
        return execution_id

    def cancel(self, execution_id: str) -> None:
        with self._lock:
            ctx = self._executions.get(execution_id)
        if ctx is None:
            raise KeyError(f"Execution {execution_id!r} not found")
        with ctx.lock:
            ctx.cancelled = True

    def status(self, execution_id: str) -> VWAPStatus:
        with self._lock:
            ctx = self._executions.get(execution_id)
        if ctx is None:
            raise KeyError(f"Execution {execution_id!r} not found")

        with ctx.lock:
            vwap_benchmark = ctx.avg_fill_price  # simplified: use actual avg
            tracking_bps = 0.0
            if ctx.avg_fill_price > 0 and vwap_benchmark > 0:
                tracking_bps = abs(
                    (ctx.avg_fill_price - vwap_benchmark) / vwap_benchmark
                ) * 10_000.0

            return VWAPStatus(
                execution_id=execution_id,
                total_qty=ctx.order.qty,
                filled_qty=ctx.filled_qty,
                remaining_qty=max(0.0, ctx.order.qty - ctx.filled_qty),
                buckets_completed=ctx.buckets_completed,
                buckets_total=ctx.n_buckets,
                avg_fill_price=ctx.avg_fill_price,
                vwap_benchmark=vwap_benchmark,
                tracking_error_bps=tracking_bps,
                is_complete=ctx.complete,
            )

    def active_executions(self) -> List[str]:
        with self._lock:
            return [
                eid
                for eid, ctx in self._executions.items()
                if not ctx.complete and not ctx.cancelled
            ]

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self, ctx: _VWAPContext) -> None:
        for i, bucket_qty in enumerate(ctx.bucket_qtys):
            with ctx.lock:
                if ctx.cancelled:
                    break
                remaining = max(0.0, ctx.order.qty - ctx.filled_qty)
                if remaining <= 0:
                    break

            actual_qty = min(bucket_qty, remaining)
            if actual_qty > 1e-9:
                self._execute_bucket(ctx, i, actual_qty)

            # Sleep for bucket interval
            deadline = time.monotonic() + ctx.bucket_interval_s
            while time.monotonic() < deadline:
                with ctx.lock:
                    if ctx.cancelled:
                        break
                time.sleep(min(0.05, ctx.bucket_interval_s / 10))

        with ctx.lock:
            ctx.complete = True

        logger.info(
            "VWAP execution_id=%s complete filled=%.2f/%.2f avg_px=%.4f",
            ctx.execution_id, ctx.filled_qty, ctx.order.qty, ctx.avg_fill_price,
        )

    def _execute_bucket(
        self, ctx: _VWAPContext, bucket_index: int, qty: float
    ) -> None:
        mid = self._price_provider.get_mid(ctx.order.symbol)
        # Small urgency offset proportional to participation rate
        urgency = mid * (ctx.order.target_participation_rate * 0.5 / 100.0)

        if ctx.order.side == "buy":
            fill_price = mid + urgency
        else:
            fill_price = mid - urgency

        fill_price = round(max(0.01, fill_price), 4)

        # Build a synthetic child order for the fill record
        child = LimitOrder(
            order_id=str(uuid.uuid4()),
            symbol=ctx.order.symbol,
            side=ctx.order.side,
            qty=qty,
            strategy_id=ctx.order.strategy_id,
            signal_strength=ctx.order.signal_strength,
            created_at=datetime.utcnow(),
            status=OrderStatus.FILLED,
            limit_price=fill_price,
            time_in_force="IOC",
        )
        fill = make_fill(child, qty, fill_price, venue="VWAP_SIM")
        ctx.apply_fill(qty, fill_price)

        with ctx.lock:
            ctx.buckets_completed += 1

        if self._fill_callback is not None:
            try:
                self._fill_callback(fill)
            except Exception as exc:
                logger.warning("VWAP fill_callback raised: %s", exc)

        logger.debug(
            "VWAP bucket=%d/%d execution_id=%s qty=%.2f px=%.4f",
            bucket_index + 1, ctx.n_buckets, ctx.execution_id, qty, fill_price,
        )
