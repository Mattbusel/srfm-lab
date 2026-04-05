"""
execution/monitoring/reconciler.py
=====================================
Position reconciliation between OMS and Alpaca broker.

Runs every 5 minutes and compares what the OMS thinks it holds against what
the broker actually holds.  Discrepancies are classified as:

- **Small** (< 1 % of position): auto-corrected by updating OMS state.
- **Large** (> 5 % of position or > $100 notional mismatch): alert raised,
  human intervention required.

All reconciliation events are logged to the module logger and to a
structured list that can be retrieved via ``get_events()``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("execution.reconciler")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RECONCILE_INTERVAL_SEC:    int   = 300    # 5 minutes
AUTO_CORRECT_THRESHOLD:    float = 0.01   # 1% — auto-correct below this
ALERT_THRESHOLD_FRAC:      float = 0.05   # 5% — alert above this
ALERT_THRESHOLD_USD:       float = 100.0  # $100 mismatch always alerts


# ---------------------------------------------------------------------------
# ReconcileEvent
# ---------------------------------------------------------------------------

@dataclass
class ReconcileEvent:
    """Record of a single position discrepancy."""
    symbol:        str
    oms_qty:       float
    broker_qty:    float
    delta:         float
    delta_pct:     float
    delta_usd:     float
    action:        str           # "AUTO_CORRECTED", "ALERT", "OK"
    timestamp:     datetime      = field(default_factory=lambda: datetime.now(timezone.utc))
    last_price:    float         = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol":      self.symbol,
            "oms_qty":     self.oms_qty,
            "broker_qty":  self.broker_qty,
            "delta":       self.delta,
            "delta_pct":   self.delta_pct,
            "delta_usd":   self.delta_usd,
            "action":      self.action,
            "timestamp":   self.timestamp.isoformat(),
            "last_price":  self.last_price,
        }


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

class Reconciler:
    """
    Periodic position reconciler between OMS and broker.

    Parameters
    ----------
    order_manager : OrderManager
        The OMS order manager (provides ``reconcile_with_broker``).
    broker : AlpacaAdapter
        Broker adapter for ``get_positions()``.
    position_tracker : PositionTracker
        Used for per-position price lookups.
    alert_callback : Callable[[str], None] | None
        Called with a human-readable string when a large discrepancy is found.
    interval_sec : int
        How often to reconcile (default: 300 s = 5 minutes).
    """

    def __init__(
        self,
        order_manager,
        broker,
        position_tracker,
        alert_callback=None,
        interval_sec: int = RECONCILE_INTERVAL_SEC,
    ) -> None:
        self._mgr      = order_manager
        self._broker   = broker
        self._tracker  = position_tracker
        self._alert_cb = alert_callback
        self._interval = interval_sec

        self._events:  list[ReconcileEvent] = []
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._lock     = threading.Lock()
        self._reconcile_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the periodic reconciliation background thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target = self._loop,
            daemon = True,
            name   = "reconciler",
        )
        self._thread.start()
        log.info("Reconciler started (interval=%ds)", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10.0)
        log.info("Reconciler stopped after %d runs", self._reconcile_count)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            try:
                self.run_once()
            except Exception as exc:
                log.error("Reconciler error: %s", exc, exc_info=True)
            time.sleep(self._interval)

    # ------------------------------------------------------------------
    # Core reconciliation
    # ------------------------------------------------------------------

    def run_once(self) -> list[ReconcileEvent]:
        """
        Execute one reconciliation pass synchronously.

        Returns
        -------
        list[ReconcileEvent]
            Events generated in this pass (may be empty if all match).
        """
        self._reconcile_count += 1
        log.debug("Reconciler pass #%d", self._reconcile_count)

        try:
            broker_positions = self._broker.get_positions()
        except Exception as exc:
            log.error("Reconciler: failed to fetch broker positions: %s", exc)
            return []

        oms_positions = {
            sym: pos.quantity
            for sym, pos in self._tracker.positions.items()
            if abs(pos.quantity) > 1e-9
        }

        all_symbols = set(broker_positions) | set(oms_positions)
        events: list[ReconcileEvent] = []

        for sym in all_symbols:
            broker_qty = broker_positions.get(sym, 0.0)
            oms_qty    = oms_positions.get(sym, 0.0)
            delta      = broker_qty - oms_qty

            if abs(delta) < 1e-9:
                continue   # no drift

            last_price = self._get_last_price(sym)
            delta_usd  = abs(delta) * last_price
            delta_pct  = abs(delta) / max(abs(broker_qty), 1e-9)

            action = self._classify_and_act(
                sym       = sym,
                oms_qty   = oms_qty,
                broker_qty = broker_qty,
                delta_pct = delta_pct,
                delta_usd = delta_usd,
            )

            evt = ReconcileEvent(
                symbol     = sym,
                oms_qty    = oms_qty,
                broker_qty = broker_qty,
                delta      = delta,
                delta_pct  = delta_pct,
                delta_usd  = delta_usd,
                action     = action,
                last_price = last_price,
            )
            events.append(evt)
            with self._lock:
                self._events.append(evt)

            log.info(
                "Reconcile %s: oms=%.6f broker=%.6f delta=%.6f "
                "(%.2f%%, $%.2f) action=%s",
                sym, oms_qty, broker_qty, delta, delta_pct * 100, delta_usd, action,
            )

        if not events:
            log.debug("Reconciler: all positions match broker")

        return events

    # ------------------------------------------------------------------
    # Classification + action
    # ------------------------------------------------------------------

    def _classify_and_act(
        self,
        sym:        str,
        oms_qty:    float,
        broker_qty: float,
        delta_pct:  float,
        delta_usd:  float,
    ) -> str:
        """
        Decide action based on discrepancy size.

        Returns
        -------
        str
            "AUTO_CORRECTED", "ALERT", or "OK".
        """
        is_large = delta_pct > ALERT_THRESHOLD_FRAC or delta_usd > ALERT_THRESHOLD_USD
        is_small = delta_pct < AUTO_CORRECT_THRESHOLD

        if is_small:
            # Auto-correct: trust the broker
            self._tracker.set_quantity(sym, broker_qty)
            return "AUTO_CORRECTED"

        if is_large:
            msg = (
                f"RECONCILE ALERT: {sym} OMS={oms_qty:.6f} vs "
                f"Broker={broker_qty:.6f} delta={delta_pct:.2%} / "
                f"${delta_usd:.2f} — HUMAN REVIEW REQUIRED"
            )
            log.error(msg)
            if self._alert_cb:
                try:
                    self._alert_cb(msg)
                except Exception:
                    pass
            return "ALERT"

        # Medium discrepancy: log and auto-correct
        log.warning(
            "Reconcile medium discrepancy %s %.2f%% — auto-correcting",
            sym, delta_pct * 100,
        )
        self._tracker.set_quantity(sym, broker_qty)
        return "AUTO_CORRECTED"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_last_price(self, symbol: str) -> float:
        pos = self._tracker.positions.get(symbol)
        if pos:
            return pos.last_price or pos.avg_entry_price or 1.0
        return 1.0

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_events(self, limit: int = 100) -> list[ReconcileEvent]:
        """Return the most recent reconciliation events."""
        with self._lock:
            return list(self._events[-limit:])

    def summary(self) -> dict:
        with self._lock:
            total   = len(self._events)
            alerts  = sum(1 for e in self._events if e.action == "ALERT")
            corrected = sum(1 for e in self._events if e.action == "AUTO_CORRECTED")
        return {
            "reconcile_runs":      self._reconcile_count,
            "total_discrepancies": total,
            "auto_corrected":      corrected,
            "alerts_raised":       alerts,
        }
