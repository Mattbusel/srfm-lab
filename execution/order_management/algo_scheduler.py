"""
Algorithmic order scheduler for SRFM execution layer.

AlgoScheduler  -- top-level manager for concurrent TWAP/VWAP/Iceberg runs
IcebergEngine  -- shows only display_qty at a time, re-submitting as fills arrive
"""

from __future__ import annotations

import logging
import math
import random
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Union

from .order_types import (
    Fill,
    IcebergOrder,
    LimitOrder,
    OrderStatus,
    TWAPOrder,
    VWAPOrder,
    make_fill,
)
from .twap_engine import (
    MidPriceProvider,
    TWAPEngine,
    TWAPStatus,
    VWAPEngine,
    VWAPStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AlgoExecution -- lightweight status record returned to callers
# ---------------------------------------------------------------------------

@dataclass
class AlgoExecution:
    """
    Represents a single algorithmic execution managed by AlgoScheduler.

    progress is in [0.0, 1.0].  estimated_completion is UTC datetime or None
    if the algo has already finished or the estimate is unavailable.
    """
    execution_id: str
    order_type: str          # "TWAP" / "VWAP" / "ICEBERG"
    symbol: str
    side: str
    total_qty: float
    filled_qty: float
    status: str              # "ACTIVE" / "COMPLETE" / "CANCELLED" / "ERROR"
    progress: float          # filled_qty / total_qty
    estimated_completion: Optional[datetime]
    avg_fill_price: float
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.total_qty - self.filled_qty)


# ---------------------------------------------------------------------------
# IcebergStatus
# ---------------------------------------------------------------------------

@dataclass
class IcebergStatus:
    """Point-in-time snapshot of an iceberg execution."""
    execution_id: str
    total_qty: float
    filled_qty: float
    remaining_qty: float
    display_qty: float            # current displayed slice size
    slices_completed: int
    slices_total_estimated: int
    avg_fill_price: float
    is_complete: bool


# ---------------------------------------------------------------------------
# Internal iceberg context
# ---------------------------------------------------------------------------

@dataclass
class _IcebergContext:
    """Runtime state for a single iceberg execution."""
    execution_id: str
    order: IcebergOrder
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    slices_completed: int = 0
    cancelled: bool = False
    complete: bool = False
    current_display_order: Optional[LimitOrder] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    fill_prices: List[float] = field(default_factory=list)
    limit_price: float = 0.0    # price at which display orders are placed

    def apply_fill(self, qty: float, price: float) -> None:
        with self.lock:
            prev_notional = self.filled_qty * self.avg_fill_price
            self.filled_qty += qty
            if self.filled_qty > 0:
                self.avg_fill_price = (
                    (prev_notional + qty * price) / self.filled_qty
                )
            self.fill_prices.append(price)

    def remaining(self) -> float:
        # Note: do NOT call this while already holding self.lock
        return max(0.0, self.order.total_qty - self.filled_qty)

    def jitter_display_qty(self) -> float:
        """
        Return display_qty +/- 10% random noise to avoid pattern detection.
        Result is rounded to 1 decimal place and clipped to (0, remaining].
        """
        base = self.order.display_qty
        jitter = random.uniform(-0.10, 0.10) * base
        raw = base + jitter
        remaining = self.remaining()
        return max(1.0, round(min(raw, remaining), 1))


# ---------------------------------------------------------------------------
# IcebergEngine
# ---------------------------------------------------------------------------

class IcebergEngine:
    """
    Iceberg (reserve) order engine.

    Only display_qty shares are visible on the book at a time.  When a display
    order fills, on_fill() is called and a new display order is submitted
    immediately for the next slice until total_qty is exhausted.

    The display qty is randomized by +/-10% each slice to reduce detectable
    patterns.  This is a passive strategy -- no background thread; callers
    must route fill events back via on_fill().
    """

    def __init__(
        self,
        mid_price_provider: Optional[MidPriceProvider] = None,
        fill_callback: Optional[Callable[[Fill], None]] = None,
        submit_order_callback: Optional[Callable[[LimitOrder], None]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._executions: Dict[str, _IcebergContext] = {}
        self._price_provider = mid_price_provider or MidPriceProvider()
        self._fill_callback = fill_callback
        # Called when a new display order is ready to be routed to the venue
        self._submit_order_callback = submit_order_callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, order: IcebergOrder, limit_price: float = 0.0) -> str:
        """
        Begin iceberg execution.  Returns execution_id.

        limit_price is the price at which all display orders are placed.
        If 0.0 the engine queries MidPriceProvider for the current mid.
        """
        order.validate()
        execution_id = str(uuid.uuid4())

        if limit_price <= 0:
            limit_price = self._price_provider.get_mid(order.symbol)

        ctx = _IcebergContext(
            execution_id=execution_id,
            order=order,
            limit_price=limit_price,
        )

        with self._lock:
            self._executions[execution_id] = ctx

        self._submit_next_display(ctx)

        logger.info(
            "IcebergEngine.submit execution_id=%s symbol=%s total_qty=%s "
            "display_qty=%s limit_px=%.4f",
            execution_id, order.symbol, order.total_qty,
            order.display_qty, limit_price,
        )
        return execution_id

    def on_fill(self, fill: Fill) -> None:
        """
        Route a fill event to the relevant iceberg execution.

        Looks up the execution by matching fill.order_id against current
        display orders.  If the display order is fully filled a new slice
        is submitted.
        """
        with self._lock:
            ctx = self._find_ctx_by_display_order(fill.order_id)

        if ctx is None:
            logger.debug(
                "IcebergEngine.on_fill no ctx for order_id=%s", fill.order_id
            )
            return

        ctx.apply_fill(fill.qty, fill.price)

        # Propagate fill upward
        if self._fill_callback is not None:
            try:
                self._fill_callback(fill)
            except Exception as exc:
                logger.warning("IcebergEngine fill_callback raised: %s", exc)

        with ctx.lock:
            display_order = ctx.current_display_order
            if display_order is not None:
                display_order.apply_fill(fill)

        # Check if display slice fully filled
        display_filled = False
        with ctx.lock:
            do = ctx.current_display_order
            if do is not None and do.filled_qty >= do.qty - 1e-9:
                ctx.slices_completed += 1
                display_filled = True
                ctx.current_display_order = None

        if display_filled and ctx.remaining() > 1e-9:
            self._submit_next_display(ctx)
        elif ctx.remaining() <= 1e-9:
            with ctx.lock:
                ctx.complete = True
            logger.info(
                "IcebergEngine execution_id=%s complete filled=%.2f avg_px=%.4f",
                ctx.execution_id, ctx.filled_qty, ctx.avg_fill_price,
            )

    def cancel(self, execution_id: str) -> None:
        """Cancel remaining iceberg slices."""
        with self._lock:
            ctx = self._executions.get(execution_id)
        if ctx is None:
            raise KeyError(f"IcebergExecution {execution_id!r} not found")
        with ctx.lock:
            ctx.cancelled = True
            ctx.current_display_order = None
        logger.info("IcebergEngine.cancel execution_id=%s", execution_id)

    def status(self, execution_id: str) -> IcebergStatus:
        with self._lock:
            ctx = self._executions.get(execution_id)
        if ctx is None:
            raise KeyError(f"IcebergExecution {execution_id!r} not found")

        with ctx.lock:
            return IcebergStatus(
                execution_id=execution_id,
                total_qty=ctx.order.total_qty,
                filled_qty=ctx.filled_qty,
                remaining_qty=ctx.remaining(),
                display_qty=ctx.order.display_qty,
                slices_completed=ctx.slices_completed,
                slices_total_estimated=ctx.order.n_slices_estimated,
                avg_fill_price=ctx.avg_fill_price,
                is_complete=ctx.complete,
            )

    def simulate_fill_current_display(self, execution_id: str) -> Optional[Fill]:
        """
        Test/simulation helper: immediately fill the current display order
        at the limit_price and route it through on_fill().

        Returns the Fill or None if no display order is active.
        """
        with self._lock:
            ctx = self._executions.get(execution_id)
        if ctx is None:
            return None

        with ctx.lock:
            do = ctx.current_display_order
        if do is None:
            return None

        fill = make_fill(do, do.qty, ctx.limit_price, venue="ICEBERG_SIM")
        self.on_fill(fill)
        return fill

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit_next_display(self, ctx: _IcebergContext) -> None:
        """Create and register a new display LimitOrder for the next slice."""
        with ctx.lock:
            if ctx.cancelled or ctx.complete:
                return
            remaining = ctx.remaining()
            if remaining <= 1e-9:
                ctx.complete = True
                return
            display_qty = ctx.jitter_display_qty()
            display_qty = min(display_qty, remaining)

        child = LimitOrder(
            order_id=str(uuid.uuid4()),
            symbol=ctx.order.symbol,
            side=ctx.order.side,
            qty=display_qty,
            strategy_id=ctx.order.strategy_id,
            signal_strength=ctx.order.signal_strength,
            created_at=datetime.utcnow(),
            status=OrderStatus.PENDING,
            limit_price=ctx.limit_price,
            time_in_force="GTC",
        )

        with ctx.lock:
            ctx.current_display_order = child
            ctx.order.shown_orders.append(child)

        if self._submit_order_callback is not None:
            try:
                self._submit_order_callback(child)
            except Exception as exc:
                logger.warning("IcebergEngine submit_order_callback raised: %s", exc)

        logger.debug(
            "Iceberg display order execution_id=%s order_id=%s qty=%.2f px=%.4f",
            ctx.execution_id, child.order_id, display_qty, ctx.limit_price,
        )

    def _find_ctx_by_display_order(
        self, order_id: str
    ) -> Optional[_IcebergContext]:
        """Return the context whose current display order matches order_id."""
        for ctx in self._executions.values():
            with ctx.lock:
                do = ctx.current_display_order
                if do is not None and do.order_id == order_id:
                    return ctx
        return None


# ---------------------------------------------------------------------------
# AlgoScheduler
# ---------------------------------------------------------------------------

class AlgoScheduler:
    """
    Central manager for concurrent algorithmic executions.

    Maintains separate TWAPEngine, VWAPEngine, and IcebergEngine instances.
    Provides a unified submit/cancel/status interface and computes daily
    summary statistics.
    """

    def __init__(
        self,
        mid_price_provider: Optional[MidPriceProvider] = None,
        fill_callback: Optional[Callable[[Fill], None]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._fill_callback = fill_callback

        provider = mid_price_provider or MidPriceProvider()

        self._twap_engine = TWAPEngine(
            mid_price_provider=provider,
            fill_callback=self._on_fill,
        )
        self._vwap_engine = VWAPEngine(
            mid_price_provider=provider,
            fill_callback=self._on_fill,
        )
        self._iceberg_engine = IcebergEngine(
            mid_price_provider=provider,
            fill_callback=self._on_fill,
        )

        # execution_id -> (order_type, original_order)
        self._registry: Dict[str, tuple] = {}
        # all fills received this session
        self._fills: List[Fill] = []
        # total slippage bps accumulated this session
        self._slippage_bps_list: List[float] = []

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_algo(
        self, order: Union[TWAPOrder, VWAPOrder, IcebergOrder]
    ) -> str:
        """
        Submit any algorithmic order for execution.

        Returns execution_id.
        """
        if isinstance(order, TWAPOrder):
            execution_id = self._twap_engine.submit(order)
            order_type = "TWAP"
        elif isinstance(order, VWAPOrder):
            execution_id = self._vwap_engine.submit(order)
            order_type = "VWAP"
        elif isinstance(order, IcebergOrder):
            execution_id = self._iceberg_engine.submit(order)
            order_type = "ICEBERG"
        else:
            raise TypeError(
                f"Unsupported algo order type: {type(order).__name__}"
            )

        with self._lock:
            self._registry[execution_id] = (order_type, order)

        logger.info(
            "AlgoScheduler.submit_algo execution_id=%s type=%s symbol=%s qty=%s",
            execution_id, order_type, order.symbol, order.qty,
        )
        return execution_id

    def cancel(self, execution_id: str) -> None:
        """Cancel a specific algorithmic execution."""
        with self._lock:
            entry = self._registry.get(execution_id)
        if entry is None:
            raise KeyError(f"AlgoExecution {execution_id!r} not found")
        order_type, _ = entry
        if order_type == "TWAP":
            self._twap_engine.cancel(execution_id)
        elif order_type == "VWAP":
            self._vwap_engine.cancel(execution_id)
        elif order_type == "ICEBERG":
            self._iceberg_engine.cancel(execution_id)

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all active algorithmic executions, optionally filtering by symbol.

        Returns the number of executions cancelled.
        """
        count = 0
        with self._lock:
            ids = list(self._registry.keys())
        for eid in ids:
            try:
                exec_info = self.get_algo_execution(eid)
                if exec_info is None:
                    continue
                if symbol is not None and exec_info.symbol != symbol:
                    continue
                if exec_info.status == "ACTIVE":
                    self.cancel(eid)
                    count += 1
            except Exception as exc:
                logger.warning("cancel_all error for %s: %s", eid, exc)
        logger.info(
            "AlgoScheduler.cancel_all symbol=%s cancelled=%d", symbol, count
        )
        return count

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def get_algo_execution(self, execution_id: str) -> Optional[AlgoExecution]:
        """Return a snapshot AlgoExecution for the given execution_id."""
        with self._lock:
            entry = self._registry.get(execution_id)
        if entry is None:
            return None
        order_type, order = entry

        if order_type == "TWAP":
            try:
                s: TWAPStatus = self._twap_engine.status(execution_id)
                status_str = "COMPLETE" if s.is_complete else "ACTIVE"
                return AlgoExecution(
                    execution_id=execution_id,
                    order_type="TWAP",
                    symbol=order.symbol,
                    side=order.side,
                    total_qty=s.total_qty,
                    filled_qty=s.filled_qty,
                    status=status_str,
                    progress=s.fill_pct,
                    estimated_completion=None,
                    avg_fill_price=s.avg_fill_price,
                )
            except KeyError:
                pass

        elif order_type == "VWAP":
            try:
                vs: VWAPStatus = self._vwap_engine.status(execution_id)
                status_str = "COMPLETE" if vs.is_complete else "ACTIVE"
                fill_pct = vs.filled_qty / max(vs.total_qty, 1e-9)
                return AlgoExecution(
                    execution_id=execution_id,
                    order_type="VWAP",
                    symbol=order.symbol,
                    side=order.side,
                    total_qty=vs.total_qty,
                    filled_qty=vs.filled_qty,
                    status=status_str,
                    progress=fill_pct,
                    estimated_completion=None,
                    avg_fill_price=vs.avg_fill_price,
                )
            except KeyError:
                pass

        elif order_type == "ICEBERG":
            try:
                from .algo_scheduler import IcebergStatus
                ist: IcebergStatus = self._iceberg_engine.status(execution_id)
                status_str = "COMPLETE" if ist.is_complete else "ACTIVE"
                fill_pct = ist.filled_qty / max(ist.total_qty, 1e-9)
                return AlgoExecution(
                    execution_id=execution_id,
                    order_type="ICEBERG",
                    symbol=order.symbol,
                    side=order.side,
                    total_qty=ist.total_qty,
                    filled_qty=ist.filled_qty,
                    status=status_str,
                    progress=fill_pct,
                    estimated_completion=None,
                    avg_fill_price=ist.avg_fill_price,
                )
            except KeyError:
                pass

        return None

    def get_all_active(self) -> List[AlgoExecution]:
        """Return AlgoExecution snapshots for all non-complete executions."""
        result: List[AlgoExecution] = []
        with self._lock:
            ids = list(self._registry.keys())
        for eid in ids:
            ae = self.get_algo_execution(eid)
            if ae is not None and ae.status == "ACTIVE":
                result.append(ae)
        return result

    def get_all_executions(self) -> List[AlgoExecution]:
        """Return AlgoExecution snapshots for all tracked executions."""
        result: List[AlgoExecution] = []
        with self._lock:
            ids = list(self._registry.keys())
        for eid in ids:
            ae = self.get_algo_execution(eid)
            if ae is not None:
                result.append(ae)
        return result

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    def daily_algo_summary(self) -> dict:
        """
        Compute session-level summary statistics.

        Returns a dict with keys:
          total_algo_volume_usd, total_fills, avg_fill_price,
          avg_slippage_bps, completion_rate, active_count, complete_count
        """
        with self._lock:
            fills_snapshot = list(self._fills)
            ids = list(self._registry.keys())

        total_notional = sum(f.qty * f.price for f in fills_snapshot)
        total_fills = len(fills_snapshot)

        total_qty = sum(f.qty for f in fills_snapshot)
        avg_fill_price = total_notional / max(total_qty, 1e-9) if total_qty > 0 else 0.0

        avg_slippage_bps = 0.0
        if self._slippage_bps_list:
            avg_slippage_bps = sum(self._slippage_bps_list) / len(self._slippage_bps_list)

        active_count = 0
        complete_count = 0
        for eid in ids:
            ae = self.get_algo_execution(eid)
            if ae is None:
                continue
            if ae.status == "ACTIVE":
                active_count += 1
            elif ae.status == "COMPLETE":
                complete_count += 1

        total = active_count + complete_count
        completion_rate = complete_count / max(total, 1) if total > 0 else 0.0

        return {
            "total_algo_volume_usd": round(total_notional, 2),
            "total_fills": total_fills,
            "avg_fill_price": round(avg_fill_price, 4),
            "avg_slippage_bps": round(avg_slippage_bps, 4),
            "completion_rate": round(completion_rate, 4),
            "active_count": active_count,
            "complete_count": complete_count,
        }

    # ------------------------------------------------------------------
    # Fill routing
    # ------------------------------------------------------------------

    def _on_fill(self, fill: Fill) -> None:
        """Internal fill sink -- aggregates fills and forwards to external callback."""
        with self._lock:
            self._fills.append(fill)

        if self._fill_callback is not None:
            try:
                self._fill_callback(fill)
            except Exception as exc:
                logger.warning("AlgoScheduler external fill_callback raised: %s", exc)

    # ------------------------------------------------------------------
    # Iceberg passthrough helpers
    # ------------------------------------------------------------------

    def route_fill_to_iceberg(self, fill: Fill) -> None:
        """
        Forward a fill event from the venue to the IcebergEngine.

        Call this whenever a fill arrives for an order submitted by the
        iceberg engine so the engine can re-submit the next display slice.
        """
        self._iceberg_engine.on_fill(fill)
