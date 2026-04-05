"""
bridge/live_param_bridge.py

LiveParamBridge: watch params.json for changes and signal the live trader
to reload its parameters atomically and safely.

Uses polling (watchdog-optional) to detect changes. Validates parameters
before applying. Writes atomically via tmp-then-rename. Tracks version.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal as _signal
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parents[1]
_PARAMS_FILE = _REPO_ROOT / "config" / "live_params.json"
_POLL_INTERVAL = 5.0   # seconds between polls when watchdog unavailable

# Validation constraints: key -> (min, max)
_PARAM_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "min_hold_bars": (1, 200),
    "max_hold_bars": (1, 2000),
    "min_correlation": (-1.0, 1.0),
    "max_correlation": (-1.0, 1.0),
    "position_size_pct": (0.001, 0.5),
    "stop_loss_pct": (0.001, 0.5),
    "take_profit_pct": (0.001, 2.0),
    "regime_filter_threshold": (0.0, 1.0),
}


class LiveParamBridge:
    """
    Watch params.json for changes and apply them to the live trader safely.

    - Validates values against known constraints before applying
    - Writes atomically (tmp → rename) to prevent partial reads
    - Signals the live trader via SIGUSR1 (Unix) or a reload socket
    - Tracks version numbers and timestamps in the params file
    """

    def __init__(
        self,
        params_file: Path | str | None = None,
        reload_socket_path: str = "/tmp/live_trader_reload.sock",
        poll_interval: float = _POLL_INTERVAL,
    ) -> None:
        self.params_file = Path(params_file) if params_file else _PARAMS_FILE
        self.params_file.parent.mkdir(parents=True, exist_ok=True)
        self.reload_socket_path = reload_socket_path
        self.poll_interval = poll_interval
        self._last_mtime: float = 0.0
        self._last_version: int = 0
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Poll params.json and apply changes as they arrive."""
        self._running = True
        logger.info("LiveParamBridge: watching %s (poll every %.1fs)", self.params_file, self.poll_interval)

        try:
            while self._running:
                await self._check_for_changes()
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            logger.info("LiveParamBridge: stopped.")

    def stop(self) -> None:
        self._running = False

    def push_params(
        self,
        params: dict[str, Any],
        source: str = "autonomous_loop",
        hypothesis_id: str | None = None,
    ) -> bool:
        """
        Write new parameters to params.json and signal the live trader.
        Returns True on success.
        """
        validation_errors = self._validate(params)
        if validation_errors:
            logger.warning(
                "LiveParamBridge: validation failed — %s", validation_errors
            )
            return False

        versioned = {
            "version": self._last_version + 1,
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "hypothesis_id": hypothesis_id or "",
            "params": params,
        }

        if self._atomic_write(versioned):
            self._last_version += 1
            self._signal_trader()
            logger.info(
                "LiveParamBridge: pushed version %d from %s", self._last_version, source
            )
            return True
        return False

    def read_current(self) -> dict[str, Any]:
        """Read and return the current params.json content."""
        if not self.params_file.exists():
            return {}
        try:
            return json.loads(self.params_file.read_text())
        except Exception as exc:
            logger.warning("LiveParamBridge: could not read params: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    async def _check_for_changes(self) -> None:
        if not self.params_file.exists():
            return
        try:
            mtime = self.params_file.stat().st_mtime
        except OSError:
            return

        if mtime <= self._last_mtime:
            return

        self._last_mtime = mtime
        data = self.read_current()
        version = int(data.get("version", 0))

        if version <= self._last_version:
            return

        logger.info(
            "LiveParamBridge: detected new version %d (was %d)", version, self._last_version
        )
        params = data.get("params", {})
        errors = self._validate(params)
        if errors:
            logger.warning("LiveParamBridge: ignoring invalid params: %s", errors)
            return

        self._last_version = version
        self._signal_trader()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, params: dict[str, Any]) -> list[str]:
        """Check params against known constraints. Returns list of error messages."""
        errors = []
        for key, value in params.items():
            if key in _PARAM_CONSTRAINTS:
                lo, hi = _PARAM_CONSTRAINTS[key]
                try:
                    fv = float(value)
                    if not (lo <= fv <= hi):
                        errors.append(f"{key}={fv} out of range [{lo}, {hi}]")
                except (TypeError, ValueError):
                    errors.append(f"{key}: non-numeric value '{value}'")

        # Hard sanity checks
        if "min_hold_bars" in params and "max_hold_bars" in params:
            try:
                if float(params["min_hold_bars"]) >= float(params["max_hold_bars"]):
                    errors.append("min_hold_bars must be < max_hold_bars")
            except Exception:
                pass

        if "min_correlation" in params and "max_correlation" in params:
            try:
                if float(params["min_correlation"]) > float(params["max_correlation"]):
                    errors.append("min_correlation must be <= max_correlation")
            except Exception:
                pass

        return errors

    # ------------------------------------------------------------------
    # Atomic write
    # ------------------------------------------------------------------

    def _atomic_write(self, data: dict[str, Any]) -> bool:
        """Write to a tmp file then atomically rename to the target path."""
        tmp = self.params_file.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self.params_file)
            return True
        except Exception as exc:
            logger.error("LiveParamBridge: atomic write failed: %s", exc)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return False

    # ------------------------------------------------------------------
    # Reload signalling
    # ------------------------------------------------------------------

    def _signal_trader(self) -> None:
        """Signal the live trader to reload parameters."""
        # Try Unix signal first (works on Linux/macOS)
        trader_pid = self._find_trader_pid()
        if trader_pid:
            try:
                os.kill(trader_pid, _signal.SIGUSR1)
                logger.info("LiveParamBridge: sent SIGUSR1 to PID %d", trader_pid)
                return
            except (OSError, AttributeError):
                pass  # SIGUSR1 not available on Windows or PID gone

        # Fall back to Unix domain socket
        self._socket_signal()

    def _socket_signal(self) -> None:
        """Send reload command via Unix domain socket."""
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(2.0)
                sock.connect(self.reload_socket_path)
                sock.sendall(b"RELOAD\n")
            logger.info("LiveParamBridge: sent RELOAD via socket")
        except Exception as exc:
            logger.debug("LiveParamBridge: socket signal failed (trader may not support it): %s", exc)

    def _find_trader_pid(self) -> int | None:
        """Try to find the PID of the live trader process."""
        pid_file = _REPO_ROOT / "tools" / "live_trader.pid"
        if pid_file.exists():
            try:
                return int(pid_file.read_text().strip())
            except Exception:
                pass
        return None
