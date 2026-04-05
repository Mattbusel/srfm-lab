"""
scheduler.py
============
Periodic update scheduler for the Bayesian updater.

Runs the full update cycle every *interval_hours* (default 4 hours).
Each cycle:
    1. Load new trades from live_trades.db.
    2. Update posteriors.
    3. Detect drift.
    4. Emit IAE hypotheses for any significant findings.
    5. Check for regime change.
    6. Persist state to JSON.
    7. Log a structured summary.

Design
------
The scheduler uses Python's ``threading.Timer`` for lightweight periodic
execution that works inside the IAE process without a separate daemon.
A threading.Event is used for clean shutdown.

If the idea-engine scheduler module (``idea-engine/scheduler/``) is
available, we integrate with it; otherwise we run standalone.

Logging
-------
Each update cycle logs a JSON-structured summary at INFO level and a
human-readable summary at DEBUG level.  All posterior estimates (mean,
std, CI) are logged for audit purposes.

Usage::

    scheduler = UpdateScheduler(interval_hours=4)
    scheduler.start()
    # ... runs in background ...
    scheduler.stop()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .updater import BayesianUpdater
from .drift_monitor import DriftMonitor, DriftAlert
from .hypothesis_emitter import HypothesisEmitter, IAEHypothesis
from .priors import build_default_priors

logger = logging.getLogger(__name__)

_HERE  = Path(__file__).resolve()
_REPO  = _HERE.parents[3]
_LOG_FILE = _REPO / "idea-engine" / "bayesian-updater" / "update_log.jsonl"


# ---------------------------------------------------------------------------
# Cycle result container
# ---------------------------------------------------------------------------

@dataclass
class CycleResult:
    """
    Summary of one update cycle.

    Attributes
    ----------
    cycle_number        : sequential cycle counter.
    timestamp           : ISO-8601 string of cycle start.
    n_new_trades        : trades processed in this cycle.
    posterior_means     : dict of {param: posterior_mean}.
    posterior_stds      : dict of {param: posterior_std}.
    drift_flags         : list of parameter names that drifted.
    regime_alert        : DriftAlert if regime change was detected.
    hypotheses_emitted  : number of new hypotheses emitted.
    duration_seconds    : wall-clock time for the cycle.
    error               : error message if the cycle failed.
    """

    cycle_number:       int
    timestamp:          str
    n_new_trades:       int = 0
    posterior_means:    Dict[str, float] = field(default_factory=dict)
    posterior_stds:     Dict[str, float] = field(default_factory=dict)
    drift_flags:        List[str]  = field(default_factory=list)
    regime_alert:       Optional[str] = None
    hypotheses_emitted: int = 0
    duration_seconds:   float = 0.0
    error:              Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "cycle":              self.cycle_number,
            "timestamp":          self.timestamp,
            "n_new_trades":       self.n_new_trades,
            "posterior_means":    self.posterior_means,
            "posterior_stds":     self.posterior_stds,
            "drift_flags":        self.drift_flags,
            "regime_alert":       self.regime_alert,
            "hypotheses_emitted": self.hypotheses_emitted,
            "duration_seconds":   round(self.duration_seconds, 2),
            "error":              self.error,
        }

    def log_summary(self) -> str:
        """Human-readable one-liner for this cycle."""
        if self.error:
            return f"[Cycle {self.cycle_number}] ERROR: {self.error}"
        drift_str = ", ".join(self.drift_flags) if self.drift_flags else "none"
        return (
            f"[Cycle {self.cycle_number}] "
            f"trades={self.n_new_trades}, "
            f"drift=[{drift_str}], "
            f"hypotheses={self.hypotheses_emitted}, "
            f"regime_alert={'YES' if self.regime_alert else 'no'}, "
            f"duration={self.duration_seconds:.1f}s"
        )


# ---------------------------------------------------------------------------
# UpdateScheduler
# ---------------------------------------------------------------------------

class UpdateScheduler:
    """
    Periodic Bayesian update scheduler.

    Parameters
    ----------
    interval_hours   : how often to run the update cycle (default 4).
    updater          : BayesianUpdater instance.  Created from default
                       config if not provided.
    drift_monitor    : DriftMonitor instance.  Created with empty baseline
                       if not provided.
    emitter          : HypothesisEmitter instance.
    hypothesis_callback: optional callable(List[IAEHypothesis]) invoked
                       after each cycle with any new hypotheses.
    log_file         : path to the JSONL log file.
    auto_reset_on_regime: if True, automatically reset posteriors when a
                       regime change is detected.
    """

    def __init__(
        self,
        interval_hours: float = 4.0,
        updater: Optional[BayesianUpdater] = None,
        drift_monitor: Optional[DriftMonitor] = None,
        emitter: Optional[HypothesisEmitter] = None,
        hypothesis_callback: Optional[Callable[[List[IAEHypothesis]], None]] = None,
        log_file: Optional[Path] = None,
        auto_reset_on_regime: bool = True,
    ):
        self.interval_hours       = interval_hours
        self.updater              = updater or BayesianUpdater.from_state_file()
        self.drift_monitor        = drift_monitor or DriftMonitor()
        self.emitter              = emitter or HypothesisEmitter()
        self.hypothesis_callback  = hypothesis_callback
        self.log_file             = log_file or _LOG_FILE
        self.auto_reset_on_regime = auto_reset_on_regime

        self._stop_event:  threading.Event  = threading.Event()
        self._timer:       Optional[threading.Timer] = None
        self._cycle_count: int              = 0
        self._cycle_history: List[CycleResult] = []

        logger.info(
            "UpdateScheduler initialised: interval=%.1fh, log=%s",
            interval_hours, self.log_file,
        )

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the periodic update loop.

        The first cycle runs immediately, then every *interval_hours*.
        Non-blocking: returns after scheduling.
        """
        self._stop_event.clear()
        logger.info("UpdateScheduler starting (interval=%.1fh).", self.interval_hours)
        self._schedule_next(delay=0)

    def stop(self) -> None:
        """
        Stop the scheduler gracefully.

        Cancels the pending timer.  Any currently-running cycle is
        allowed to complete.
        """
        self._stop_event.set()
        if self._timer is not None:
            self._timer.cancel()
        logger.info("UpdateScheduler stopped after %d cycles.", self._cycle_count)

    def run_once(self) -> CycleResult:
        """
        Run a single update cycle synchronously.

        Useful for testing or manual invocation.

        Returns
        -------
        CycleResult summarising what happened.
        """
        return self._run_cycle()

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    def _schedule_next(self, delay: Optional[float] = None) -> None:
        """Schedule the next cycle using a threading.Timer."""
        if self._stop_event.is_set():
            return
        delay = delay if delay is not None else self.interval_hours * 3600
        self._timer = threading.Timer(delay, self._cycle_wrapper)
        self._timer.daemon = True
        self._timer.start()

    def _cycle_wrapper(self) -> None:
        """Wrapper that calls _run_cycle and then re-schedules."""
        try:
            result = self._run_cycle()
            self._append_log(result)
        except Exception as exc:
            logger.exception("Unhandled error in update cycle: %s", exc)
        finally:
            self._schedule_next()

    def _run_cycle(self) -> CycleResult:
        """
        Execute one full update cycle.

        Steps:
        1. Load new trades.
        2. Update posteriors.
        3. Detect drift.
        4. Check regime change.
        5. Emit hypotheses.
        6. Save state.

        Returns
        -------
        CycleResult.
        """
        import numpy as np

        self._cycle_count += 1
        cycle_num  = self._cycle_count
        ts_start   = time.monotonic()
        timestamp  = datetime.utcnow().isoformat()

        result = CycleResult(cycle_number=cycle_num, timestamp=timestamp)

        try:
            # 1. Update posteriors
            posteriors = self.updater.update_from_db()
            result.n_new_trades = posteriors.n_trades

            # 2. Collect posterior summaries for logging
            for name, est in posteriors.estimates.items():
                result.posterior_means[name] = round(est.mean, 6)
                result.posterior_stds[name]  = round(est.std, 6)

            # 3. Detect parameter drift
            drift_flags = self.updater.detect_drift()
            result.drift_flags = [f.param_name for f in drift_flags]

            # 4. Regime change check
            recent_pnl = self.updater._batcher.get_recent_pnl(n=100)
            regime_alert: Optional[DriftAlert] = None
            if len(recent_pnl) > 0:
                regime_alert = self.drift_monitor.check(recent_pnl)

            if regime_alert:
                result.regime_alert = regime_alert.alert_type
                logger.warning("Regime change alert: %s", regime_alert.message)

                if self.auto_reset_on_regime:
                    self.updater._computer.reset_to_priors()
                    logger.info("Posteriors reset to priors due to regime change.")

                    # Update drift monitor baseline with the recent data
                    if len(recent_pnl) >= 30:
                        self.drift_monitor.update_historical_baseline(recent_pnl)

            # 5. Emit hypotheses
            hypotheses: List[IAEHypothesis] = []
            if drift_flags:
                hypotheses += self.emitter.emit_from_drift_flags(
                    drift_flags, posteriors=posteriors, n_trades=posteriors.n_trades
                )
            if regime_alert:
                hyp = self.emitter.emit_from_regime_alert(
                    regime_alert, n_trades=posteriors.n_trades
                )
                hypotheses.append(hyp)

            # Also scan posteriors for informative updates
            hypotheses += self.emitter.emit_from_posteriors(
                posteriors, n_trades=posteriors.n_trades
            )

            result.hypotheses_emitted = len(hypotheses)

            if hypotheses and self.hypothesis_callback:
                try:
                    self.hypothesis_callback(hypotheses)
                except Exception as cb_exc:
                    logger.error("Hypothesis callback failed: %s", cb_exc)

            # 6. Save state
            self.updater.save_state()

        except Exception as exc:
            result.error = str(exc)
            logger.exception("Error in update cycle %d: %s", cycle_num, exc)

        result.duration_seconds = time.monotonic() - ts_start
        self._cycle_history.append(result)

        logger.info(result.log_summary())
        logger.debug(
            "Posterior means: %s",
            json.dumps(result.posterior_means, default=str),
        )

        return result

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _append_log(self, result: CycleResult) -> None:
        """Append a JSON line to the log file."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(result.to_dict(), default=str) + "\n")
        except Exception as exc:
            logger.error("Failed to write cycle log: %s", exc)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def cycle_history(self) -> List[CycleResult]:
        """Return all completed cycle results."""
        return list(self._cycle_history)

    def next_run_at(self) -> Optional[str]:
        """Return the approximate ISO timestamp of the next scheduled run."""
        if self._timer is None or self._stop_event.is_set():
            return None
        eta = datetime.utcnow() + timedelta(seconds=self.interval_hours * 3600)
        return eta.isoformat()

    def status(self) -> dict:
        """Return a summary of scheduler status."""
        return {
            "running":         not self._stop_event.is_set(),
            "interval_hours":  self.interval_hours,
            "cycles_completed": self._cycle_count,
            "next_run_at":     self.next_run_at(),
            "n_hypotheses_emitted": len(self.emitter.emitted_hypotheses()),
        }
