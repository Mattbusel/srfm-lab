"""
execution/monitoring/live_monitor.py
======================================
Real-time execution monitoring running in a background thread.

Responsibilities
----------------
Every 30 seconds:
  - Scan for orders stuck in SUBMITTED for > 5 minutes; cancel and resubmit.
  - Check daily P&L, current drawdown from peak, and position count.
  - Alert if: daily loss > 3 %, any position > 35 % of portfolio, or order
    rejection rate > 20 %.

Every 60 seconds:
  - Write a status snapshot to ``execution/status.json`` for dashboard pickup.

Alerts are logged at WARNING or ERROR level and can also be routed to an
external alerting callable (e.g. Slack webhook, email).

Usage
-----
::

    monitor = LiveMonitor(
        order_manager = mgr,
        position_tracker = tracker,
        circuit_breaker  = cb,
    )
    monitor.start()
    # ... trading ...
    monitor.stop()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("execution.live_monitor")

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

STUCK_ORDER_THRESHOLD_SEC:  int   = 300    # 5 minutes
DAILY_LOSS_ALERT_FRAC:      float = 0.03   # 3 %
POSITION_ALERT_FRAC:        float = 0.35   # 35 %
REJECTION_RATE_ALERT:       float = 0.20   # 20 %
CHECK_INTERVAL_SEC:         int   = 30
STATUS_WRITE_INTERVAL_SEC:  int   = 60

STATUS_FILE = Path(__file__).parent.parent / "status.json"


# ---------------------------------------------------------------------------
# LiveMonitor
# ---------------------------------------------------------------------------

class LiveMonitor:
    """
    Background thread that monitors execution health.

    Parameters
    ----------
    order_manager : OrderManager
        The live OMS order manager.
    position_tracker : PositionTracker
        Real-time position state.
    circuit_breaker : CircuitBreaker | None
        If provided, alerts are also forwarded as halt triggers when thresholds
        are breached.
    alert_callback : Callable[[str, str], None] | None
        Optional external alert function ``(level, message) -> None``.
    initial_equity : float
        Equity at session start (for daily P&L calculation).
    status_file : Path | None
        Override the default status.json path.
    """

    def __init__(
        self,
        order_manager,
        position_tracker,
        circuit_breaker=None,
        alert_callback: Optional[Callable[[str, str], None]] = None,
        initial_equity: float = 100_000.0,
        status_file: Optional[Path] = None,
    ) -> None:
        self._mgr              = order_manager
        self._tracker          = position_tracker
        self._cb               = circuit_breaker
        self._alert_callback   = alert_callback
        self._initial_equity   = initial_equity
        self._peak_equity      = initial_equity
        self._status_file      = status_file or STATUS_FILE

        self._running          = False
        self._thread:  Optional[threading.Thread] = None
        self._last_status_write: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the monitoring background thread."""
        if self._running:
            log.warning("LiveMonitor already running")
            return
        self._running = True
        self._thread  = threading.Thread(
            target = self._loop,
            daemon = True,
            name   = "live-monitor",
        )
        self._thread.start()
        log.info("LiveMonitor started (interval=%ds)", CHECK_INTERVAL_SEC)

    def stop(self) -> None:
        """Signal the monitoring thread to stop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info("LiveMonitor stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            try:
                self._run_checks()
            except Exception as exc:
                log.error("LiveMonitor loop error: %s", exc, exc_info=True)

            now = time.monotonic()
            if now - self._last_status_write >= STATUS_WRITE_INTERVAL_SEC:
                self._write_status()
                self._last_status_write = now

            time.sleep(CHECK_INTERVAL_SEC)

    # ------------------------------------------------------------------
    # Check routines
    # ------------------------------------------------------------------

    def _run_checks(self) -> None:
        """Execute all health checks in sequence."""
        self._check_stuck_orders()
        self._check_daily_loss()
        self._check_position_concentration()
        self._check_rejection_rate()

    def _check_stuck_orders(self) -> None:
        """Cancel and resubmit orders that have been SUBMITTED for > 5 minutes."""
        from ..oms.order import OrderStatus

        now = datetime.now(timezone.utc)
        for order in self._mgr.book.open_orders():
            if order.status != OrderStatus.SUBMITTED:
                continue
            if order.submitted_at is None:
                continue
            age_sec = (now - order.submitted_at).total_seconds()
            if age_sec > STUCK_ORDER_THRESHOLD_SEC:
                log.warning(
                    "Stuck order detected: %s %s %.4f age=%.0fs — cancelling",
                    order.symbol, order.side.value, order.quantity, age_sec,
                )
                self._alert("WARNING", f"Stuck order cancelled: {order.symbol} age={age_sec:.0f}s")
                cancelled = self._mgr.cancel_order(order.order_id)
                if cancelled:
                    # Resubmit with same parameters
                    try:
                        self._mgr.submit_order(
                            symbol       = order.symbol,
                            side         = order.side,
                            target_frac  = order.quantity * (order.price or 1.0) / self._mgr.get_equity(),
                            curr_price   = order.price or 0.0,
                            order_type   = order.order_type,
                            strategy_id  = order.strategy_id,
                            decision_price = order.decision_price,
                        )
                        log.info("Stuck order resubmitted: %s", order.symbol)
                    except Exception as exc:
                        log.error("Failed to resubmit stuck order %s: %s", order.order_id[:8], exc)

    def _check_daily_loss(self) -> None:
        """Alert if daily P&L < -3 % of initial equity."""
        current_equity = self._mgr.get_equity()
        daily_pnl_frac = (current_equity - self._initial_equity) / max(self._initial_equity, 1.0)
        # Update peak
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        drawdown = (self._peak_equity - current_equity) / max(self._peak_equity, 1.0)

        if daily_pnl_frac < -DAILY_LOSS_ALERT_FRAC:
            msg = (
                f"DAILY LOSS ALERT: {daily_pnl_frac:.2%} vs threshold "
                f"-{DAILY_LOSS_ALERT_FRAC:.2%}"
            )
            log.warning(msg)
            self._alert("WARNING", msg)
            if self._cb:
                self._cb.trigger("DAILY_LOSS_HALT", msg)

        if drawdown > 0.15:
            msg = f"DRAWDOWN ALERT: {drawdown:.2%} from peak"
            log.warning(msg)
            self._alert("WARNING", msg)

    def _check_position_concentration(self) -> None:
        """Alert if any single position exceeds 35 % of portfolio."""
        fracs = self._tracker.get_position_fractions()
        for sym, frac in fracs.items():
            if frac > POSITION_ALERT_FRAC:
                msg = f"CONCENTRATION ALERT: {sym} = {frac:.2%} > {POSITION_ALERT_FRAC:.2%}"
                log.warning(msg)
                self._alert("WARNING", msg)

    def _check_rejection_rate(self) -> None:
        """Alert if order rejection rate exceeds 20 %."""
        from ..oms.order import OrderStatus

        all_orders = self._mgr.book.all_orders()
        if len(all_orders) < 10:
            return
        rejected = sum(1 for o in all_orders if o.status == OrderStatus.REJECTED)
        rate     = rejected / len(all_orders)
        if rate > REJECTION_RATE_ALERT:
            msg = f"HIGH REJECTION RATE: {rate:.1%} ({rejected}/{len(all_orders)})"
            log.error(msg)
            self._alert("ERROR", msg)

    # ------------------------------------------------------------------
    # Status file
    # ------------------------------------------------------------------

    def _write_status(self) -> None:
        """Write a JSON status snapshot to ``execution/status.json``."""
        try:
            snapshot = self._tracker.export_snapshot()
            equity   = self._mgr.get_equity()
            daily_pnl = (equity - self._initial_equity) / max(self._initial_equity, 1.0)
            drawdown  = (self._peak_equity - equity) / max(self._peak_equity, 1.0)

            status = {
                "timestamp":       datetime.now(timezone.utc).isoformat(),
                "equity":          equity,
                "daily_pnl_frac":  daily_pnl,
                "drawdown_frac":   drawdown,
                "peak_equity":     self._peak_equity,
                "open_orders":     len(self._mgr.book.open_orders()),
                "position_count":  snapshot.get("position_count", 0),
                "leverage":        snapshot.get("leverage", 0.0),
                "positions":       snapshot.get("positions", {}),
            }
            self._status_file.parent.mkdir(parents=True, exist_ok=True)
            self._status_file.write_text(json.dumps(status, indent=2))
        except Exception as exc:
            log.error("Failed to write status.json: %s", exc)

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    def _alert(self, level: str, message: str) -> None:
        """Route an alert to the external callback if configured."""
        if self._alert_callback:
            try:
                self._alert_callback(level, message)
            except Exception as exc:
                log.error("alert_callback failed: %s", exc)
