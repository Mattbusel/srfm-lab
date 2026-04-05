"""
execution/routing/twap_executor.py
====================================
TWAP (Time-Weighted Average Price) execution engine using asyncio.

Large orders are split into N equal slices submitted at regular intervals
over a T-minute window.  Mid-execution the engine watches for adverse price
moves > 1 % and pauses if detected.

Metrics tracked
---------------
- TWAP benchmark: arrival_price + VWAP of fills
- Actual fill VWAP across all child slices
- Execution shortfall: (actual VWAP - decision_price) * side_sign / decision_price

Usage
-----
::

    executor = TWAPExecutor(broker=adapter, n_slices=5, window_minutes=10)
    # Blocking (for use from sync thread):
    broker_id = executor.submit(order)
    # From async context:
    broker_id = await executor.submit_async(order)
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("execution.twap_executor")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_N_SLICES:      int   = 5
DEFAULT_WINDOW_MIN:    float = 10.0    # minutes
ADVERSE_MOVE_THRESH:   float = 0.01   # 1% adverse move triggers pause
PAUSE_DURATION_SEC:    float = 60.0   # pause duration after adverse move
MAX_PAUSES:            int   = 3      # max number of pauses before aborting


# ---------------------------------------------------------------------------
# TWAPRun — tracks a single TWAP execution
# ---------------------------------------------------------------------------

@dataclass
class TWAPRun:
    """State container for one TWAP execution."""
    parent_order_id:  str
    symbol:           str
    side:             str          # "BUY" or "SELL"
    total_qty:        float
    n_slices:         int
    window_minutes:   float
    decision_price:   float
    arrival_price:    float        = 0.0
    slice_qty:        float        = 0.0
    submitted_slices: int          = 0
    filled_qty:       float        = 0.0
    fill_notional:    float        = 0.0
    pauses:           int          = 0
    aborted:          bool         = False
    completed:        bool         = False
    child_broker_ids: list[str]    = field(default_factory=list)
    start_time:       datetime     = field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    @property
    def actual_vwap(self) -> float:
        """Volume-weighted average price of all fills so far."""
        if self.filled_qty < 1e-9:
            return 0.0
        return self.fill_notional / self.filled_qty

    @property
    def execution_shortfall_bps(self) -> float:
        """
        Execution shortfall in basis points relative to decision price.

        Positive = paid more than expected (bad for buys, good for sells).
        """
        if self.decision_price <= 0 or self.filled_qty < 1e-9:
            return 0.0
        side_sign = 1.0 if self.side == "BUY" else -1.0
        vwap = self.actual_vwap
        return (vwap - self.decision_price) / self.decision_price * 10_000 * side_sign

    @property
    def twap_benchmark(self) -> float:
        """Average of (arrival_price + ... + last_slice_ref) — simplified as arrival."""
        return self.arrival_price

    def to_dict(self) -> dict:
        return {
            "parent_order_id":       self.parent_order_id,
            "symbol":                self.symbol,
            "side":                  self.side,
            "total_qty":             self.total_qty,
            "n_slices":              self.n_slices,
            "window_minutes":        self.window_minutes,
            "decision_price":        self.decision_price,
            "arrival_price":         self.arrival_price,
            "submitted_slices":      self.submitted_slices,
            "filled_qty":            self.filled_qty,
            "actual_vwap":           self.actual_vwap,
            "execution_shortfall_bps": self.execution_shortfall_bps,
            "pauses":                self.pauses,
            "aborted":               self.aborted,
            "completed":             self.completed,
        }


# ---------------------------------------------------------------------------
# TWAPExecutor
# ---------------------------------------------------------------------------

class TWAPExecutor:
    """
    Executes large orders as TWAP slices using asyncio.

    Parameters
    ----------
    broker : AlpacaAdapter
        Broker adapter used to submit each slice.
    n_slices : int
        Number of child orders to split the parent into.
    window_minutes : float
        Total time window over which slices are distributed.
    price_feed : Callable[[str], float] | None
        Optional function ``symbol -> current_price`` for adverse-move monitoring.
        If None, adverse-move check is skipped.
    event_bus : Callable[[str, dict], None] | None
        Optional callback for slice events.
    """

    def __init__(
        self,
        broker,
        n_slices:       int   = DEFAULT_N_SLICES,
        window_minutes: float = DEFAULT_WINDOW_MIN,
        price_feed=None,
        event_bus=None,
    ) -> None:
        self._broker         = broker
        self._n_slices       = n_slices
        self._window_minutes = window_minutes
        self._price_feed     = price_feed
        self._event_bus      = event_bus
        self._active_runs: dict[str, TWAPRun] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Synchronous entry point (called from SmartRouter)
    # ------------------------------------------------------------------

    def submit(self, order) -> str:
        """
        Start a TWAP execution in a background thread.

        Returns the parent_order_id immediately (not a broker ID — the
        OrderManager stores this as the broker_order_id for tracking).
        """
        run = self._build_run(order)
        with self._lock:
            self._active_runs[run.parent_order_id] = run

        t = threading.Thread(
            target = self._run_sync,
            args   = (run,),
            daemon = True,
            name   = f"twap-{run.parent_order_id[:8]}",
        )
        t.start()
        log.info(
            "TWAP started: %s %s %.4f in %d slices over %.1f min",
            order.side.value, order.symbol, order.quantity,
            run.n_slices, run.window_minutes,
        )
        return run.parent_order_id

    # ------------------------------------------------------------------
    # Async entry point
    # ------------------------------------------------------------------

    async def submit_async(self, order) -> str:
        """Async version of submit — runs the slice loop in the current event loop."""
        run = self._build_run(order)
        with self._lock:
            self._active_runs[run.parent_order_id] = run
        await self._run_async(run)
        return run.parent_order_id

    # ------------------------------------------------------------------
    # Internal run helpers
    # ------------------------------------------------------------------

    def _build_run(self, order) -> TWAPRun:
        """Construct a TWAPRun from an OMS Order."""
        run = TWAPRun(
            parent_order_id = order.order_id,
            symbol          = order.symbol,
            side            = order.side.value,
            total_qty       = order.quantity,
            n_slices        = self._n_slices,
            window_minutes  = self._window_minutes,
            decision_price  = order.decision_price or order.price or 0.0,
        )
        run.slice_qty     = run.total_qty / run.n_slices
        run.arrival_price = self._get_price(order.symbol)
        return run

    def _get_price(self, symbol: str) -> float:
        if self._price_feed:
            try:
                return float(self._price_feed(symbol))
            except Exception:
                pass
        return 0.0

    def _run_sync(self, run: TWAPRun) -> None:
        """Run the TWAP loop synchronously (in a background thread)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_async(run))
        finally:
            loop.close()

    async def _run_async(self, run: TWAPRun) -> None:
        """Core async TWAP loop: submit one slice every interval seconds."""
        interval_sec = (run.window_minutes * 60.0) / run.n_slices

        for i in range(run.n_slices):
            if run.aborted:
                log.warning("TWAP aborted for %s after %d slices", run.symbol, i)
                break

            # ── Adverse move check ───────────────────────────────────
            if run.arrival_price > 0 and self._price_feed:
                current_price = self._get_price(run.symbol)
                if current_price > 0:
                    move = (current_price - run.arrival_price) / run.arrival_price
                    # Adverse = price rising for BUY, falling for SELL
                    side_sign = 1.0 if run.side == "BUY" else -1.0
                    if move * side_sign > ADVERSE_MOVE_THRESH:
                        run.pauses += 1
                        log.warning(
                            "TWAP adverse move %.2f%% for %s — pausing %ds (pause %d/%d)",
                            move * 100, run.symbol, PAUSE_DURATION_SEC,
                            run.pauses, MAX_PAUSES,
                        )
                        if run.pauses >= MAX_PAUSES:
                            log.error("TWAP max pauses reached for %s — aborting", run.symbol)
                            run.aborted = True
                            break
                        await asyncio.sleep(PAUSE_DURATION_SEC)
                        # Update arrival price after pause so subsequent slices
                        # compare against the new baseline
                        run.arrival_price = self._get_price(run.symbol)

            # ── Submit slice ─────────────────────────────────────────
            side_str = "buy" if run.side == "BUY" else "sell"
            try:
                broker_id = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._broker.submit_market_order,
                    run.symbol,
                    run.slice_qty,
                    side_str,
                )
                run.child_broker_ids.append(broker_id)
                run.submitted_slices += 1
                fill_price = self._get_price(run.symbol) or run.arrival_price
                run.filled_qty    += run.slice_qty
                run.fill_notional += run.slice_qty * fill_price
                log.info(
                    "TWAP slice %d/%d: %s %s %.6f @ ~%.4f",
                    i + 1, run.n_slices, run.side, run.symbol, run.slice_qty, fill_price,
                )
                if self._event_bus:
                    self._event_bus("TWAP_SLICE", {
                        "slice": i + 1,
                        "of":    run.n_slices,
                        **run.to_dict(),
                    })
            except Exception as exc:
                log.error("TWAP slice %d failed for %s: %s", i + 1, run.symbol, exc)

            # ── Wait for next interval (skip after last slice) ───────
            if i < run.n_slices - 1:
                await asyncio.sleep(interval_sec)

        run.completed = not run.aborted
        log.info(
            "TWAP complete: %s %s filled=%.6f vwap=%.4f shortfall=%.1f bps",
            run.side, run.symbol,
            run.filled_qty, run.actual_vwap, run.execution_shortfall_bps,
        )
        if self._event_bus:
            self._event_bus("TWAP_COMPLETE", run.to_dict())

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def get_run(self, parent_order_id: str) -> Optional[TWAPRun]:
        """Return live TWAPRun state for a given parent order."""
        with self._lock:
            return self._active_runs.get(parent_order_id)

    def get_active_runs(self) -> list[TWAPRun]:
        """Return all currently active (incomplete) TWAP runs."""
        with self._lock:
            return [r for r in self._active_runs.values() if not r.completed and not r.aborted]
