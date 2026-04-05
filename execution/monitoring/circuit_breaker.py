"""
execution/monitoring/circuit_breaker.py
=========================================
Automated trading halt system with multiple independent circuit breakers.

Halt types
----------
DAILY_LOSS_HALT   — stop new entries if daily loss > 5 %
DRAWDOWN_HALT     — stop new entries if drawdown from peak > 20 %
VOLATILITY_HALT   — stop if realized vol (last 2h) > 5× historical baseline
BROKER_ERROR_HALT — stop if 3+ consecutive broker errors
MANUAL_HALT       — read halt flag from ``execution/halt.flag``

Each halt type has an independent cooldown and resume condition.
Halt events are persisted to ``execution/halts.log``.

Usage
-----
::

    cb = CircuitBreaker()
    cb.start()   # starts background flag-file watcher
    # In order submission path:
    if cb.is_halted():
        raise RuntimeError(f"Trading halted: {cb.halt_reason}")
    # Triggering from monitor:
    cb.trigger("DAILY_LOSS_HALT", "daily loss -5.2%")
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("execution.circuit_breaker")

HALT_FLAG_FILE = Path(__file__).parent.parent / "halt.flag"
HALTS_LOG_FILE = Path(__file__).parent.parent / "halts.log"

# ---------------------------------------------------------------------------
# Halt type definitions
# ---------------------------------------------------------------------------

HALT_CONFIGS: dict[str, dict] = {
    "DAILY_LOSS_HALT": {
        "description":    "Daily loss exceeds 5% of equity",
        "cooldown_min":   60,
        "auto_resume":    True,
    },
    "DRAWDOWN_HALT": {
        "description":    "Drawdown from peak exceeds 20%",
        "cooldown_min":   120,
        "auto_resume":    True,
    },
    "VOLATILITY_HALT": {
        "description":    "Realized volatility exceeds 5x historical baseline",
        "cooldown_min":   30,
        "auto_resume":    True,
    },
    "BROKER_ERROR_HALT": {
        "description":    "3+ consecutive broker errors",
        "cooldown_min":   15,
        "auto_resume":    True,
    },
    "MANUAL_HALT": {
        "description":    "Manual halt flag file detected",
        "cooldown_min":   0,
        "auto_resume":    False,   # must be cleared by removing the flag file
    },
}


# ---------------------------------------------------------------------------
# HaltState
# ---------------------------------------------------------------------------

class HaltState:
    """Mutable state for one halt type."""

    def __init__(self, halt_type: str, config: dict) -> None:
        self.halt_type    = halt_type
        self.config       = config
        self.active       = False
        self.triggered_at: Optional[datetime] = None
        self.reason:       Optional[str]       = None

    @property
    def cooldown_minutes(self) -> int:
        return self.config.get("cooldown_min", 60)

    @property
    def can_auto_resume(self) -> bool:
        return self.config.get("auto_resume", False)

    @property
    def resume_time(self) -> Optional[datetime]:
        if self.triggered_at is None:
            return None
        return self.triggered_at + timedelta(minutes=self.cooldown_minutes)

    def is_cooldown_elapsed(self) -> bool:
        if not self.active or not self.triggered_at:
            return False
        return datetime.now(timezone.utc) >= self.resume_time  # type: ignore[operator]


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Thread-safe trading halt manager.

    Parameters
    ----------
    halt_flag_file : Path | None
        Override path for the manual halt flag file.
    halts_log_file : Path | None
        Override path for the halt event log.
    vol_baseline : float
        Historical daily volatility baseline for the VOLATILITY_HALT.
    vol_multiplier : float
        Trigger VOLATILITY_HALT if realized_vol > vol_baseline * vol_multiplier.
    broker_error_threshold : int
        Consecutive broker errors that trigger BROKER_ERROR_HALT.
    """

    def __init__(
        self,
        halt_flag_file:         Optional[Path]  = None,
        halts_log_file:         Optional[Path]  = None,
        vol_baseline:           float           = 0.02,
        vol_multiplier:         float           = 5.0,
        broker_error_threshold: int             = 3,
    ) -> None:
        self._halt_flag_file        = halt_flag_file or HALT_FLAG_FILE
        self._halts_log_file        = halts_log_file or HALTS_LOG_FILE
        self._vol_baseline          = vol_baseline
        self._vol_multiplier        = vol_multiplier
        self._broker_error_threshold = broker_error_threshold

        self._states: dict[str, HaltState] = {
            ht: HaltState(ht, cfg) for ht, cfg in HALT_CONFIGS.items()
        }
        self._consecutive_broker_errors = 0
        self._lock    = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background thread that watches the flag file and auto-resumes."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target = self._background_loop,
            daemon = True,
            name   = "circuit-breaker",
        )
        self._thread.start()
        log.info("CircuitBreaker started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Halt / resume API
    # ------------------------------------------------------------------

    def trigger(self, halt_type: str, reason: str = "") -> None:
        """
        Activate a halt.

        Parameters
        ----------
        halt_type : str
            One of DAILY_LOSS_HALT, DRAWDOWN_HALT, VOLATILITY_HALT,
            BROKER_ERROR_HALT, MANUAL_HALT.
        reason : str
            Human-readable description for the log.
        """
        with self._lock:
            state = self._states.get(halt_type)
            if state is None:
                log.error("Unknown halt type: %s", halt_type)
                return
            if state.active:
                return   # already active — don't re-trigger

            state.active       = True
            state.triggered_at = datetime.now(timezone.utc)
            state.reason       = reason

        self._write_halt_log(halt_type, "TRIGGERED", reason)
        log.error(
            "CIRCUIT BREAKER TRIGGERED: %s — %s (resume after %d min)",
            halt_type, reason, state.cooldown_minutes,
        )

    def resume(self, halt_type: str, reason: str = "manual") -> None:
        """Manually clear a halt."""
        with self._lock:
            state = self._states.get(halt_type)
            if state is None or not state.active:
                return
            state.active       = False
            state.triggered_at = None
            state.reason       = None

        self._write_halt_log(halt_type, "RESUMED", reason)
        log.info("CircuitBreaker: %s resumed (%s)", halt_type, reason)

    def resume_all(self) -> None:
        """Clear all active halts (for end-of-session reset)."""
        with self._lock:
            for ht in list(self._states.keys()):
                self.resume(ht, "resume_all")

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_halted(self) -> bool:
        """Return True if any halt is currently active."""
        with self._lock:
            return any(s.active for s in self._states.values())

    @property
    def halt_reason(self) -> str:
        """Return a description of the first active halt, or empty string."""
        with self._lock:
            for ht, s in self._states.items():
                if s.active:
                    return f"{ht}: {s.reason}"
            return ""

    def status(self) -> dict:
        """Return a dict snapshot of all halt states."""
        with self._lock:
            return {
                ht: {
                    "active":       s.active,
                    "triggered_at": s.triggered_at.isoformat() if s.triggered_at else None,
                    "reason":       s.reason,
                    "resume_time":  s.resume_time.isoformat() if s.resume_time else None,
                }
                for ht, s in self._states.items()
            }

    # ------------------------------------------------------------------
    # Specialized triggers
    # ------------------------------------------------------------------

    def record_broker_error(self) -> None:
        """Increment broker error counter; trigger halt at threshold."""
        with self._lock:
            self._consecutive_broker_errors += 1
            count = self._consecutive_broker_errors
        if count >= self._broker_error_threshold:
            self.trigger(
                "BROKER_ERROR_HALT",
                f"{count} consecutive broker errors",
            )

    def record_broker_success(self) -> None:
        """Reset broker error counter after a successful call."""
        with self._lock:
            self._consecutive_broker_errors = 0

    def check_volatility(self, realized_vol: float) -> None:
        """Trigger VOLATILITY_HALT if realized_vol > baseline * multiplier."""
        threshold = self._vol_baseline * self._vol_multiplier
        if realized_vol > threshold:
            self.trigger(
                "VOLATILITY_HALT",
                f"realized_vol={realized_vol:.4f} > threshold={threshold:.4f}",
            )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _background_loop(self) -> None:
        while self._running:
            try:
                self._check_flag_file()
                self._check_auto_resume()
            except Exception as exc:
                log.error("CircuitBreaker background error: %s", exc)
            time.sleep(10)

    def _check_flag_file(self) -> None:
        """Trigger MANUAL_HALT if halt.flag exists; resume if removed."""
        exists = self._halt_flag_file.exists()
        with self._lock:
            state = self._states["MANUAL_HALT"]
            if exists and not state.active:
                self.trigger("MANUAL_HALT", f"halt.flag present at {self._halt_flag_file}")
            elif not exists and state.active:
                self.resume("MANUAL_HALT", "halt.flag removed")

    def _check_auto_resume(self) -> None:
        """Auto-resume halts whose cooldown has elapsed."""
        with self._lock:
            for ht, state in self._states.items():
                if state.active and state.can_auto_resume and state.is_cooldown_elapsed():
                    self.resume(ht, "cooldown_elapsed")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _write_halt_log(self, halt_type: str, event: str, reason: str) -> None:
        try:
            self._halts_log_file.parent.mkdir(parents=True, exist_ok=True)
            line = (
                f"{datetime.now(timezone.utc).isoformat()} "
                f"{event:10s} {halt_type:25s} {reason}\n"
            )
            with open(self._halts_log_file, "a") as f:
                f.write(line)
        except Exception as exc:
            log.error("Failed to write halts.log: %s", exc)
