#!/usr/bin/env python3
"""
scripts/supervisor.py — SRFM Lab Python Process Supervisor

Features:
  - Monitors all 5 trading services
  - Auto-restarts crashed services with exponential backoff (5s → 10s → 20s → 60s max)
  - Writes logs/supervisor.json with current state
  - HTTP API on port 8790:
      GET  /status
      POST /restart/{service}
      POST /stop/{service}

Usage:
    python scripts/supervisor.py
    python scripts/supervisor.py --port 8790
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
ENV_FILE = REPO_ROOT / "tools" / ".env"
STATE_FILE = LOGS_DIR / "supervisor.json"

# ── Service definitions ───────────────────────────────────────────────────────
# Each entry: (name, cwd, cmd_list, health_url_or_None)
SERVICES = [
    {
        "name": "market-data",
        "cwd": str(REPO_ROOT / "market-data"),
        "cmd": ["./market-data.exe"],
        "health_url": "http://localhost:8780/health",
        "port": 8780,
    },
    {
        "name": "coordination",
        "cwd": str(REPO_ROOT / "coordination"),
        "cmd": ["mix", "run", "--no-halt"],
        "health_url": "http://localhost:8781/health",
        "port": 8781,
    },
    {
        "name": "bridge",
        "cwd": str(REPO_ROOT),
        "cmd": [sys.executable, "bridge/heartbeat.py"],
        "health_url": "http://localhost:8783/health",
        "port": 8783,
    },
    {
        "name": "autonomous-loop",
        "cwd": str(REPO_ROOT),
        "cmd": [sys.executable, "-m", "idea_engine.autonomous_loop.orchestrator"],
        "health_url": None,
        "port": None,
    },
    {
        "name": "live-trader",
        "cwd": str(REPO_ROOT),
        "cmd": [sys.executable, "tools/live_trader_alpaca.py"],
        "health_url": None,
        "port": None,
    },
]

BACKOFF_SEQUENCE = [5, 10, 20, 40, 60]  # seconds; stays at max after last step

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [supervisor] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("supervisor")


# ── State ─────────────────────────────────────────────────────────────────────
class ServiceState:
    def __init__(self, defn: dict):
        self.defn = defn
        self.name: str = defn["name"]
        self.process: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.start_time: Optional[float] = None
        self.restart_count: int = 0
        self.backoff_index: int = 0
        self.next_restart_at: Optional[float] = None
        self.status: str = "stopped"   # stopped | starting | running | backoff | user-stopped
        self.last_exit_code: Optional[int] = None
        self.log_fh = None

    def to_dict(self) -> dict:
        uptime = None
        if self.start_time and self.status == "running":
            uptime = round(time.time() - self.start_time, 1)
        return {
            "name": self.name,
            "status": self.status,
            "pid": self.pid,
            "restart_count": self.restart_count,
            "uptime_seconds": uptime,
            "last_exit_code": self.last_exit_code,
            "next_restart_at": (
                datetime.fromtimestamp(self.next_restart_at, tz=timezone.utc).isoformat()
                if self.next_restart_at
                else None
            ),
        }


class Supervisor:
    def __init__(self, env: dict):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.env = env
        self.states: dict[str, ServiceState] = {
            svc["name"]: ServiceState(svc) for svc in SERVICES
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    # ── Process control ───────────────────────────────────────────────────────
    def _open_log(self, state: ServiceState):
        if state.log_fh:
            try:
                state.log_fh.close()
            except Exception:
                pass
        log_path = LOGS_DIR / f"{state.name}.log"
        state.log_fh = open(log_path, "a", buffering=1)

    def _start(self, state: ServiceState):
        defn = state.defn
        self._open_log(state)
        log.info("Starting %s: %s (cwd=%s)", state.name, " ".join(defn["cmd"]), defn["cwd"])
        try:
            proc = subprocess.Popen(
                defn["cmd"],
                cwd=defn["cwd"],
                env=self.env,
                stdout=state.log_fh,
                stderr=state.log_fh,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            log.error("Could not start %s: %s", state.name, exc)
            state.status = "backoff"
            self._schedule_backoff(state)
            return

        state.process = proc
        state.pid = proc.pid
        state.start_time = time.time()
        state.status = "running"
        state.next_restart_at = None

        # Write pid file
        pid_path = LOGS_DIR / f"{state.name}.pid"
        pid_path.write_text(str(proc.pid))
        log.info("%s started (PID %d)", state.name, proc.pid)

    def _schedule_backoff(self, state: ServiceState):
        delay = BACKOFF_SEQUENCE[min(state.backoff_index, len(BACKOFF_SEQUENCE) - 1)]
        state.backoff_index = min(state.backoff_index + 1, len(BACKOFF_SEQUENCE) - 1)
        state.next_restart_at = time.time() + delay
        state.status = "backoff"
        log.info("%s will restart in %ds (attempt #%d)", state.name, delay, state.restart_count + 1)

    def _stop_process(self, state: ServiceState, remove_pid: bool = True):
        if state.process and state.process.poll() is None:
            log.info("Stopping %s (PID %d) ...", state.name, state.pid)
            try:
                state.process.terminate()
                try:
                    state.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    state.process.kill()
            except OSError:
                pass
        state.process = None
        state.pid = None
        state.start_time = None
        if remove_pid:
            pid_path = LOGS_DIR / f"{state.name}.pid"
            pid_path.unlink(missing_ok=True)

    # ── Monitor loop ──────────────────────────────────────────────────────────
    def _poll_once(self):
        with self._lock:
            for state in self.states.values():
                if state.status == "user-stopped":
                    continue

                # Check if running process has died
                if state.status == "running" and state.process:
                    rc = state.process.poll()
                    if rc is not None:
                        state.last_exit_code = rc
                        state.process = None
                        state.pid = None
                        log.warning("%s exited with code %d", state.name, rc)
                        state.restart_count += 1
                        self._schedule_backoff(state)

                # Attempt restart after backoff delay
                elif state.status == "backoff":
                    if state.next_restart_at and time.time() >= state.next_restart_at:
                        self._start(state)

                # Not started yet
                elif state.status == "stopped":
                    self._start(state)

            self._write_state()

    def _write_state(self):
        data = {
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            "services": [s.to_dict() for s in self.states.values()],
        }
        try:
            STATE_FILE.write_text(json.dumps(data, indent=2))
        except OSError:
            pass

    def run(self):
        log.info("Supervisor starting — repo: %s", REPO_ROOT)
        while not self._stop_event.is_set():
            self._poll_once()
            self._stop_event.wait(timeout=5)

        # Graceful shutdown
        log.info("Supervisor shutting down ...")
        with self._lock:
            for state in self.states.values():
                self._stop_process(state)

    def stop(self):
        self._stop_event.set()

    # ── HTTP API helpers ──────────────────────────────────────────────────────
    def api_status(self) -> dict:
        with self._lock:
            return {
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
                "services": [s.to_dict() for s in self.states.values()],
            }

    def api_restart(self, name: str) -> tuple[int, str]:
        with self._lock:
            state = self.states.get(name)
            if not state:
                return 404, f"Unknown service: {name}"
            self._stop_process(state, remove_pid=False)
            state.status = "stopped"
            state.backoff_index = 0
            state.next_restart_at = None
            self._start(state)
            return 200, f"{name} restarted"

    def api_stop(self, name: str) -> tuple[int, str]:
        with self._lock:
            state = self.states.get(name)
            if not state:
                return 404, f"Unknown service: {name}"
            self._stop_process(state)
            state.status = "user-stopped"
            return 200, f"{name} stopped"


# ── HTTP server ───────────────────────────────────────────────────────────────
def make_handler(supervisor: Supervisor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug("HTTP %s", fmt % args)

        def _send_json(self, code: int, data):
            body = json.dumps(data, indent=2).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/status", "/"):
                self._send_json(200, supervisor.api_status())
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            parsed = urlparse(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) == 2 and parts[0] == "restart":
                code, msg = supervisor.api_restart(parts[1])
                self._send_json(code, {"message": msg})
            elif len(parts) == 2 and parts[0] == "stop":
                code, msg = supervisor.api_stop(parts[1])
                self._send_json(code, {"message": msg})
            else:
                self._send_json(404, {"error": "not found"})

    return Handler


# ── Environment loading ───────────────────────────────────────────────────────
def load_env() -> dict:
    env = os.environ.copy()
    if ENV_FILE.exists():
        log.info("Loading env from %s", ENV_FILE)
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                # Strip surrounding quotes
                val = val.strip().strip('"').strip("'")
                env[key.strip()] = val
    return env


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SRFM Lab Supervisor")
    parser.add_argument("--port", type=int, default=8790, help="HTTP API port (default 8790)")
    args = parser.parse_args()

    env = load_env()
    supervisor = Supervisor(env)

    # Write our own PID so health_check.sh can find it
    (LOGS_DIR / "supervisor.pid").write_text(str(os.getpid()))

    # HTTP server in background thread
    handler_class = make_handler(supervisor)
    httpd = HTTPServer(("0.0.0.0", args.port), handler_class)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()
    log.info("HTTP API listening on http://0.0.0.0:%d", args.port)
    log.info("  GET  /status")
    log.info("  POST /restart/{service}")
    log.info("  POST /stop/{service}")

    # Graceful shutdown on SIGTERM/SIGINT
    def _handle_sig(signum, _frame):
        log.info("Received signal %d — shutting down", signum)
        supervisor.stop()
        httpd.shutdown()

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    supervisor.run()
    (LOGS_DIR / "supervisor.pid").unlink(missing_ok=True)
    log.info("Supervisor exited cleanly.")


if __name__ == "__main__":
    main()
