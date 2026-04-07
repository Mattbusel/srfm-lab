# infra/deployment/process_manager.py -- OS process management for SRFM services
from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RestartPolicy(Enum):
    NEVER = "never"
    ON_FAILURE = "on_failure"
    ALWAYS = "always"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ServiceDefinition:
    name: str
    command: List[str]
    cwd: str
    env: Dict[str, str] = field(default_factory=dict)
    log_file: str = ""
    pid_file: str = ""
    restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE
    restart_delay_s: float = 3.0
    max_restarts: int = 10

    def __post_init__(self) -> None:
        if not self.log_file:
            self.log_file = f"/tmp/srfm_{self.name}.log"
        if not self.pid_file:
            self.pid_file = f"/tmp/srfm_{self.name}.pid"


@dataclass
class ProcessStatus:
    name: str
    pid: Optional[int]
    running: bool
    uptime_s: float
    restarts: int
    last_exit_code: Optional[int]
    started_at: Optional[datetime] = None

    def summary(self) -> str:
        state = "RUNNING" if self.running else "STOPPED"
        pid_str = str(self.pid) if self.pid else "N/A"
        uptime = f"{self.uptime_s:.0f}s" if self.running else "-"
        return (
            f"[{state}] {self.name:<22} pid={pid_str:<8} "
            f"uptime={uptime:<8} restarts={self.restarts}"
        )


# ---------------------------------------------------------------------------
# Internal runtime state per managed service
# ---------------------------------------------------------------------------

@dataclass
class _ServiceState:
    definition: ServiceDefinition
    process: Optional[subprocess.Popen] = None
    started_at: Optional[float] = None  # monotonic time
    restarts: int = 0
    last_exit_code: Optional[int] = None
    supervisor_enabled: bool = False


# ---------------------------------------------------------------------------
# Log rotation helpers
# ---------------------------------------------------------------------------

LOG_MAX_BYTES: int = 100 * 1024 * 1024  # 100 MB
LOG_KEEP_ROTATIONS: int = 5


def _rotate_log_if_needed(log_path: str) -> None:
    """Rotate log file if it exceeds LOG_MAX_BYTES, keeping LOG_KEEP_ROTATIONS copies."""
    p = Path(log_path)
    if not p.exists():
        return
    if p.stat().st_size < LOG_MAX_BYTES:
        return

    logger.info("Rotating log file: %s", log_path)
    # Shift existing rotations: .4 -> dropped, .3 -> .4, ..., .0 -> .1
    for i in range(LOG_KEEP_ROTATIONS - 1, 0, -1):
        src = Path(f"{log_path}.{i - 1}")
        dst = Path(f"{log_path}.{i}")
        if src.exists():
            shutil.move(str(src), str(dst))
    shutil.move(log_path, f"{log_path}.0")
    logger.debug("Log rotated: %s -> %s.0", log_path, log_path)


# ---------------------------------------------------------------------------
# PID file helpers
# ---------------------------------------------------------------------------

def _write_pid(pid_file: str, pid: int) -> None:
    try:
        Path(pid_file).write_text(str(pid), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write PID file %s: %s", pid_file, exc)


def _read_pid(pid_file: str) -> Optional[int]:
    try:
        text = Path(pid_file).read_text(encoding="utf-8").strip()
        return int(text) if text else None
    except (OSError, ValueError):
        return None


def _remove_pid(pid_file: str) -> None:
    try:
        Path(pid_file).unlink(missing_ok=True)
    except OSError:
        pass


def _process_alive(pid: int) -> bool:
    """Return True if a process with the given PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


# ---------------------------------------------------------------------------
# ProcessManager
# ---------------------------------------------------------------------------

class ProcessManager:
    """Manages lifecycle of SRFM service processes.

    -- Spawns processes via subprocess.Popen
    -- Tracks PIDs via pid files
    -- Handles graceful shutdown with SIGTERM -> SIGKILL fallback
    -- Rotates log files automatically
    """

    def __init__(self) -> None:
        self._states: Dict[str, _ServiceState] = {}
        self._lock = threading.Lock()

    # -- registration --------------------------------------------------------

    def register(self, definition: ServiceDefinition) -> None:
        """Register a service definition. Idempotent."""
        with self._lock:
            if definition.name in self._states:
                # Update definition but keep runtime state
                self._states[definition.name].definition = definition
            else:
                self._states[definition.name] = _ServiceState(definition=definition)
        logger.debug("Registered service definition: %s", definition.name)

    def registered(self) -> List[str]:
        with self._lock:
            return list(self._states.keys())

    # -- lifecycle -----------------------------------------------------------

    def start_service(self, name: str) -> bool:
        """Start a registered service. Returns True on success."""
        with self._lock:
            state = self._states.get(name)
        if state is None:
            logger.error("Cannot start unknown service: %s", name)
            return False

        if self._is_running(state):
            logger.info("Service '%s' is already running (pid=%s)", name, state.process and state.process.pid)
            return True

        defn = state.definition
        _rotate_log_if_needed(defn.log_file)

        # Build environment: inherit current env, then overlay service env
        env = dict(os.environ)
        env.update(defn.env)

        # Ensure log directory exists
        Path(defn.log_file).parent.mkdir(parents=True, exist_ok=True)

        try:
            log_fh = open(defn.log_file, "a", encoding="utf-8")
        except OSError as exc:
            logger.error("Cannot open log file %s for service %s: %s", defn.log_file, name, exc)
            return False

        try:
            proc = subprocess.Popen(
                defn.command,
                cwd=defn.cwd,
                env=env,
                stdout=log_fh,
                stderr=log_fh,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, ValueError) as exc:
            log_fh.close()
            logger.error("Failed to spawn service '%s': %s", name, exc)
            return False

        log_fh.close()

        with self._lock:
            state.process = proc
            state.started_at = time.monotonic()
            state.last_exit_code = None

        _write_pid(defn.pid_file, proc.pid)
        logger.info("Started service '%s' pid=%d", name, proc.pid)
        return True

    def stop_service(self, name: str, graceful_timeout_s: float = 30.0) -> bool:
        """Stop a service gracefully, falling back to SIGKILL after timeout."""
        with self._lock:
            state = self._states.get(name)
        if state is None:
            logger.error("Cannot stop unknown service: %s", name)
            return False

        proc = state.process
        if proc is None or not self._is_running(state):
            logger.info("Service '%s' is not running", name)
            return True

        logger.info("Stopping service '%s' (pid=%d, timeout=%.1fs)", name, proc.pid, graceful_timeout_s)

        # Try SIGTERM first (SIGINT on Windows)
        try:
            if sys.platform == "win32":
                proc.terminate()
            else:
                os.kill(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        # Wait for graceful exit
        deadline = time.monotonic() + graceful_timeout_s
        while time.monotonic() < deadline:
            ret = proc.poll()
            if ret is not None:
                with self._lock:
                    state.last_exit_code = ret
                    state.process = None
                _remove_pid(state.definition.pid_file)
                logger.info("Service '%s' stopped with exit code %d", name, ret)
                return True
            time.sleep(0.2)

        # Escalate to SIGKILL
        logger.warning("Service '%s' did not stop gracefully; sending SIGKILL", name)
        try:
            if sys.platform == "win32":
                proc.kill()
            else:
                os.kill(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

        proc.wait(timeout=5.0)
        with self._lock:
            state.last_exit_code = proc.returncode
            state.process = None
        _remove_pid(state.definition.pid_file)
        logger.info("Service '%s' killed", name)
        return True

    def restart_service(self, name: str) -> bool:
        """Stop then start a service."""
        stopped = self.stop_service(name)
        if not stopped:
            logger.warning("restart_service: stop phase failed for '%s', continuing anyway", name)
        time.sleep(1.0)
        started = self.start_service(name)
        if started:
            with self._lock:
                state = self._states.get(name)
                if state:
                    state.restarts += 1
        return started

    # -- status --------------------------------------------------------------

    def status(self, name: str) -> ProcessStatus:
        """Return current ProcessStatus for a single service."""
        with self._lock:
            state = self._states.get(name)
        if state is None:
            return ProcessStatus(
                name=name,
                pid=None,
                running=False,
                uptime_s=0.0,
                restarts=0,
                last_exit_code=None,
            )

        running = self._is_running(state)
        pid: Optional[int] = None
        uptime = 0.0
        started_dt: Optional[datetime] = None

        if running and state.process:
            pid = state.process.pid
            if state.started_at is not None:
                uptime = time.monotonic() - state.started_at
                started_dt = datetime.utcfromtimestamp(
                    time.time() - uptime
                )
        else:
            # Try PID file
            pid = _read_pid(state.definition.pid_file)
            if pid and _process_alive(pid):
                running = True

        return ProcessStatus(
            name=name,
            pid=pid,
            running=running,
            uptime_s=uptime,
            restarts=state.restarts,
            last_exit_code=state.last_exit_code,
            started_at=started_dt,
        )

    def status_all(self) -> Dict[str, ProcessStatus]:
        with self._lock:
            names = list(self._states.keys())
        return {name: self.status(name) for name in names}

    # -- log access ----------------------------------------------------------

    def tail_logs(self, name: str, n_lines: int = 100) -> List[str]:
        """Return the last n_lines from the service log file."""
        with self._lock:
            state = self._states.get(name)
        if state is None:
            return [f"Unknown service: {name}"]

        log_path = Path(state.definition.log_file)
        if not log_path.exists():
            return [f"Log file not found: {log_path}"]

        try:
            # Efficient tail: read from end of file
            with open(log_path, "rb") as f:
                f.seek(0, 2)  # seek to end
                file_size = f.tell()
                # Read up to 1 MB from the end to find n_lines
                read_size = min(file_size, 1 * 1024 * 1024)
                f.seek(-read_size, 2)
                raw = f.read()

            lines = raw.decode("utf-8", errors="replace").splitlines()
            return lines[-n_lines:]
        except OSError as exc:
            return [f"Error reading log: {exc}"]

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _is_running(state: _ServiceState) -> bool:
        if state.process is None:
            return False
        return state.process.poll() is None


# ---------------------------------------------------------------------------
# ProcessSupervisor
# ---------------------------------------------------------------------------

class ProcessSupervisor:
    """Background thread that monitors and auto-restarts services per policy.

    -- Checks process liveness every `poll_interval_s` seconds.
    -- Respects RestartPolicy: NEVER, ON_FAILURE, ALWAYS.
    -- Caps restarts at ServiceDefinition.max_restarts.
    """

    def __init__(
        self,
        manager: ProcessManager,
        poll_interval_s: float = 5.0,
    ) -> None:
        self._manager = manager
        self._poll_interval_s = poll_interval_s
        self._enabled_services: set = set()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def enable(self, name: str) -> None:
        """Enable supervision for a service."""
        with self._lock:
            self._enabled_services.add(name)
        logger.info("Supervisor enabled for service: %s", name)

    def disable(self, name: str) -> None:
        with self._lock:
            self._enabled_services.discard(name)

    def start(self) -> None:
        """Start the supervisor background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="ProcessSupervisor",
            daemon=True,
        )
        self._thread.start()
        logger.info("ProcessSupervisor started (poll=%.1fs)", self._poll_interval_s)

    def stop(self, timeout_s: float = 10.0) -> None:
        """Stop the supervisor thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout_s)
        logger.info("ProcessSupervisor stopped")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- internal ------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._poll_interval_s)
            if self._stop_event.is_set():
                break
            self._check_all()

    def _check_all(self) -> None:
        with self._lock:
            names = list(self._enabled_services)

        for name in names:
            try:
                self._check_service(name)
            except Exception:
                logger.exception("Supervisor error checking service '%s'", name)

    def _check_service(self, name: str) -> None:
        # Access internal state directly via manager's private dict
        with self._manager._lock:
            state = self._manager._states.get(name)
        if state is None:
            return

        defn = state.definition
        running = ProcessManager._is_running(state)

        if running:
            return  # all good

        # Service is down -- check policy
        exit_code = state.process.returncode if state.process else None
        was_failure = exit_code is not None and exit_code != 0

        should_restart = False
        if defn.restart_policy == RestartPolicy.ALWAYS:
            should_restart = True
        elif defn.restart_policy == RestartPolicy.ON_FAILURE and was_failure:
            should_restart = True

        if not should_restart:
            return

        if state.restarts >= defn.max_restarts:
            logger.error(
                "Service '%s' exceeded max_restarts=%d; not restarting",
                name,
                defn.max_restarts,
            )
            return

        logger.warning(
            "Supervisor restarting '%s' (exit=%s, restarts_so_far=%d)",
            name,
            exit_code,
            state.restarts,
        )
        time.sleep(defn.restart_delay_s)
        self._manager.start_service(name)
        with self._manager._lock:
            s2 = self._manager._states.get(name)
            if s2:
                s2.restarts += 1
