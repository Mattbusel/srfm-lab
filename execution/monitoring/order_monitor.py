"""
Real-time order monitoring for SRFM.
Tracks lifecycle of every order: submission -> fills -> terminal state.
Provides fill/rejection rate analytics and stale-order detection.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FillRecord:
    """A single fill event associated with an order."""

    fill_id: str
    qty: float
    price: float
    timestamp: datetime
    venue: str  # e.g. "alpaca", "binance"


@dataclass
class OrderRecord:
    """Full lifecycle record for one order."""

    order_id: str
    symbol: str
    side: str           # "buy" | "sell"
    qty: float
    price: float
    order_type: str     # "limit" | "market" | "stop" ...
    strategy_id: str
    submitted_at: datetime
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    cancel_reason: Optional[str] = None
    status: str = "pending"  # pending | partial | filled | cancelled | rejected
    fills: List[FillRecord] = field(default_factory=list)

    @property
    def filled_qty(self) -> float:
        return sum(f.qty for f in self.fills)

    @property
    def avg_fill_price(self) -> Optional[float]:
        if not self.fills:
            return None
        total_value = sum(f.qty * f.price for f in self.fills)
        total_qty = sum(f.qty for f in self.fills)
        return total_value / total_qty if total_qty else None

    @property
    def is_terminal(self) -> bool:
        return self.status in ("filled", "cancelled", "rejected")

    @property
    def age_seconds(self) -> float:
        now = datetime.now(timezone.utc)
        submitted = self.submitted_at
        if submitted.tzinfo is None:
            submitted = submitted.replace(tzinfo=timezone.utc)
        return (now - submitted).total_seconds()


# ---------------------------------------------------------------------------
# Core monitor
# ---------------------------------------------------------------------------

class OrderMonitor:
    """
    Real-time tracker for order lifecycle events.

    All methods are thread-safe via a single RLock so callers from broker
    callbacks and strategy threads can safely interleave.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # live orders (not yet terminal)
        self._active: Dict[str, OrderRecord] = {}
        # terminal orders -- keyed by order_id
        self._history: Dict[str, OrderRecord] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def track_order(self, order: OrderRecord) -> None:
        """Add an order to the monitoring system."""
        with self._lock:
            if order.order_id in self._active or order.order_id in self._history:
                logger.debug("order %s already tracked -- ignoring duplicate", order.order_id)
                return
            self._active[order.order_id] = order
            logger.debug("tracking order %s %s %s x%.2f", order.order_id, order.side, order.symbol, order.qty)

    def on_fill(self, order_id: str, fill: FillRecord) -> None:
        """Record a fill against an active order."""
        with self._lock:
            order = self._active.get(order_id)
            if order is None:
                logger.warning("on_fill: unknown order_id %s", order_id)
                return

            order.fills.append(fill)
            filled = order.filled_qty

            if filled >= order.qty:
                order.status = "filled"
                order.filled_at = fill.timestamp
                self._archive(order)
                logger.info(
                    "Order %s FILLED %.2f @ %.4f (venue=%s)",
                    order_id, filled, fill.price, fill.venue,
                )
            else:
                order.status = "partial"
                logger.debug(
                    "Order %s partial fill %.2f/%.2f",
                    order_id, filled, order.qty,
                )

    def on_cancel(self, order_id: str, reason: str = "") -> None:
        """Mark an order as cancelled."""
        with self._lock:
            order = self._active.get(order_id)
            if order is None:
                logger.warning("on_cancel: unknown order_id %s", order_id)
                return
            order.status = "cancelled"
            order.cancelled_at = datetime.now(timezone.utc)
            order.cancel_reason = reason
            self._archive(order)
            logger.info("Order %s CANCELLED: %s", order_id, reason)

    def on_reject(self, order_id: str, reason: str = "") -> None:
        """Mark an order as rejected."""
        with self._lock:
            order = self._active.get(order_id)
            if order is None:
                logger.warning("on_reject: unknown order_id %s", order_id)
                return
            order.status = "rejected"
            order.rejected_at = datetime.now(timezone.utc)
            order.rejection_reason = reason
            self._archive(order)
            logger.warning("Order %s REJECTED: %s", order_id, reason)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def pending_orders(self) -> List[OrderRecord]:
        """Return all orders currently in pending or partial state."""
        with self._lock:
            return [o for o in self._active.values() if o.status in ("pending", "partial")]

    def stale_orders(self, max_age_seconds: float = 30.0) -> List[OrderRecord]:
        """Return pending orders older than max_age_seconds."""
        with self._lock:
            return [
                o for o in self._active.values()
                if o.status in ("pending", "partial") and o.age_seconds > max_age_seconds
            ]

    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        with self._lock:
            return self._active.get(order_id) or self._history.get(order_id)

    def fill_rate(self, window_minutes: float = 60.0) -> float:
        """
        Fraction of submitted orders that were filled within window_minutes.
        Excludes still-active orders.  Returns 0.0 if no completed orders.
        """
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            completed = [
                o for o in self._history.values()
                if _submitted_after(o, cutoff)
            ]
            if not completed:
                return 0.0
            filled = sum(1 for o in completed if o.status == "filled")
            return filled / len(completed)

    def rejection_rate(self, window_minutes: float = 60.0) -> float:
        """Fraction of completed orders that were rejected within window_minutes."""
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            completed = [
                o for o in self._history.values()
                if _submitted_after(o, cutoff)
            ]
            if not completed:
                return 0.0
            rejected = sum(1 for o in completed if o.status == "rejected")
            return rejected / len(completed)

    def cancel_rate(self, window_minutes: float = 60.0) -> float:
        """Fraction of completed orders that were cancelled within window_minutes."""
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            completed = [
                o for o in self._history.values()
                if _submitted_after(o, cutoff)
            ]
            if not completed:
                return 0.0
            cancelled = sum(1 for o in completed if o.status == "cancelled")
            return cancelled / len(completed)

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def history_count(self) -> int:
        with self._lock:
            return len(self._history)

    def recent_history(self, n: int = 100) -> List[OrderRecord]:
        """Return the n most recently completed orders, newest first."""
        with self._lock:
            orders = sorted(
                self._history.values(),
                key=lambda o: o.submitted_at,
                reverse=True,
            )
            return orders[:n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _archive(self, order: OrderRecord) -> None:
        """Move an order from active to history.  Must hold self._lock."""
        self._active.pop(order.order_id, None)
        self._history[order.order_id] = order


# ---------------------------------------------------------------------------
# Alert engine
# ---------------------------------------------------------------------------

@dataclass
class OrderAlert:
    """A single alert generated by the OrderAlertEngine."""

    alert_id: str
    order_id: Optional[str]
    level: str          # "WARNING" | "CRITICAL"
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metric: str = ""
    value: float = 0.0


class OrderAlertEngine:
    """
    Monitors an OrderMonitor and generates alerts for:
    - Stale orders (pending > 30s = WARNING, > 120s = CRITICAL)
    - Rejection spikes (rejection rate > 20% in 5 min)
    - Fill rate degradation (fill rate < 80% in 30 min)
    """

    STALE_WARNING_SECONDS: float = 30.0
    STALE_CRITICAL_SECONDS: float = 120.0
    REJECTION_SPIKE_THRESHOLD: float = 0.20   # 20%
    REJECTION_SPIKE_WINDOW_MIN: float = 5.0
    FILL_RATE_DEGRADATION_THRESHOLD: float = 0.80  # 80%
    FILL_RATE_WINDOW_MIN: float = 30.0

    def __init__(self, monitor: OrderMonitor) -> None:
        self._monitor = monitor
        self._lock = threading.Lock()
        self._alert_counter: int = 0
        # track which stale orders we have already alerted to avoid spam
        self._alerted_warning: set = set()
        self._alerted_critical: set = set()

    def check_all(self) -> List[OrderAlert]:
        """Run all checks and return any new alerts generated."""
        alerts: List[OrderAlert] = []
        alerts.extend(self._check_stale_orders())
        alerts.extend(self._check_rejection_spike())
        alerts.extend(self._check_fill_rate())
        return alerts

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_stale_orders(self) -> List[OrderAlert]:
        alerts: List[OrderAlert] = []
        stale_warning = self._monitor.stale_orders(self.STALE_WARNING_SECONDS)

        for order in stale_warning:
            age = order.age_seconds
            if age > self.STALE_CRITICAL_SECONDS:
                if order.order_id not in self._alerted_critical:
                    self._alerted_critical.add(order.order_id)
                    alerts.append(OrderAlert(
                        alert_id=self._next_id(),
                        order_id=order.order_id,
                        level="CRITICAL",
                        message=(
                            f"Order {order.order_id} ({order.symbol} {order.side}) "
                            f"has been pending for {age:.0f}s -- exceeds 120s threshold"
                        ),
                        metric="stale_order_age_seconds",
                        value=age,
                    ))
            else:
                if order.order_id not in self._alerted_warning:
                    self._alerted_warning.add(order.order_id)
                    alerts.append(OrderAlert(
                        alert_id=self._next_id(),
                        order_id=order.order_id,
                        level="WARNING",
                        message=(
                            f"Order {order.order_id} ({order.symbol} {order.side}) "
                            f"has been pending for {age:.0f}s -- exceeds 30s threshold"
                        ),
                        metric="stale_order_age_seconds",
                        value=age,
                    ))

        # clear resolved orders from tracked sets
        pending_ids = {o.order_id for o in self._monitor.pending_orders()}
        self._alerted_warning &= pending_ids
        self._alerted_critical &= pending_ids

        return alerts

    def _check_rejection_spike(self) -> List[OrderAlert]:
        rate = self._monitor.rejection_rate(self.REJECTION_SPIKE_WINDOW_MIN)
        if rate > self.REJECTION_SPIKE_THRESHOLD:
            return [OrderAlert(
                alert_id=self._next_id(),
                order_id=None,
                level="CRITICAL",
                message=(
                    f"Rejection rate {rate:.1%} exceeds threshold "
                    f"{self.REJECTION_SPIKE_THRESHOLD:.0%} "
                    f"over last {self.REJECTION_SPIKE_WINDOW_MIN:.0f} min"
                ),
                metric="rejection_rate",
                value=rate,
            )]
        return []

    def _check_fill_rate(self) -> List[OrderAlert]:
        rate = self._monitor.fill_rate(self.FILL_RATE_WINDOW_MIN)
        # only alert if there is meaningful volume (rate > 0 means orders exist)
        history_count = self._monitor.history_count()
        if history_count < 5:
            # not enough data to be meaningful
            return []
        if 0.0 < rate < self.FILL_RATE_DEGRADATION_THRESHOLD:
            return [OrderAlert(
                alert_id=self._next_id(),
                order_id=None,
                level="WARNING",
                message=(
                    f"Fill rate {rate:.1%} below threshold "
                    f"{self.FILL_RATE_DEGRADATION_THRESHOLD:.0%} "
                    f"over last {self.FILL_RATE_WINDOW_MIN:.0f} min"
                ),
                metric="fill_rate",
                value=rate,
            )]
        return []

    def _next_id(self) -> str:
        with self._lock:
            self._alert_counter += 1
            return f"OA-{self._alert_counter:06d}"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _submitted_after(order: OrderRecord, cutoff: datetime) -> bool:
    submitted = order.submitted_at
    if submitted.tzinfo is None:
        submitted = submitted.replace(tzinfo=timezone.utc)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    return submitted >= cutoff
