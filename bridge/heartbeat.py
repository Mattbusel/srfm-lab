"""
bridge/heartbeat.py

HeartbeatServer: lightweight HTTP server on port 8783 that receives
pings from the live trader and alerts if the trader goes silent.

The live trader POSTs to /heartbeat every 30 seconds with:
  {"equity": X, "positions": N}

If no heartbeat for 90 seconds, fires an alert (trader may have crashed).
Tracks uptime, last_seen, total_uptime_pct.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

logger = logging.getLogger(__name__)

_PORT = 8783
_HEARTBEAT_TIMEOUT_S = 90.0      # alert if silent for this long
_HEARTBEAT_INTERVAL_S = 30.0     # expected ping interval
_STATUS_FILE = Path(__file__).parent / "heartbeat_status.json"


class HeartbeatState:
    """Thread-safe state container shared between the HTTP handler and monitor."""

    def __init__(self) -> None:
        self.last_seen: float = 0.0
        self.last_equity: float = 0.0
        self.last_positions: int = 0
        self.total_beats: int = 0
        self.started_at: float = time.monotonic()
        self.down_seconds: float = 0.0
        self._alerted: bool = False

    def record_beat(self, equity: float, positions: int) -> None:
        self.last_seen = time.monotonic()
        self.last_equity = equity
        self.last_positions = positions
        self.total_beats += 1
        self._alerted = False

    def is_alive(self) -> bool:
        if self.last_seen == 0:
            return False
        return (time.monotonic() - self.last_seen) < _HEARTBEAT_TIMEOUT_S

    def seconds_since_beat(self) -> float:
        if self.last_seen == 0:
            return float("inf")
        return time.monotonic() - self.last_seen

    def uptime_pct(self) -> float:
        elapsed = time.monotonic() - self.started_at
        if elapsed < 1:
            return 100.0
        up = elapsed - self.down_seconds
        return 100.0 * max(0.0, up) / elapsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "alive": self.is_alive(),
            "last_seen_iso": (
                datetime.now(timezone.utc)
                .fromtimestamp(
                    time.time() - self.seconds_since_beat(), tz=timezone.utc
                )
                .isoformat()
                if self.last_seen > 0
                else None
            ),
            "seconds_since_beat": round(self.seconds_since_beat(), 1)
            if self.last_seen > 0
            else None,
            "last_equity": self.last_equity,
            "last_positions": self.last_positions,
            "total_beats": self.total_beats,
            "uptime_pct": round(self.uptime_pct(), 2),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


_STATE = HeartbeatState()


class _HeartbeatHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler: POST /heartbeat, GET /status."""

    def log_message(self, format: str, *args: Any) -> None:  # type: ignore[override]
        logger.debug("Heartbeat HTTP: " + format, *args)

    def do_POST(self) -> None:
        if self.path != "/heartbeat":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
            equity = float(data.get("equity", 0.0))
            positions = int(data.get("positions", 0))
            _STATE.record_beat(equity, positions)
            logger.debug(
                "Heartbeat: equity=%.2f positions=%d", equity, positions
            )
        except Exception as exc:
            logger.warning("Heartbeat: malformed body: %s", exc)
            self.send_error(400, str(exc))
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def do_GET(self) -> None:
        if self.path not in ("/status", "/healthz", "/"):
            self.send_error(404)
            return

        payload = json.dumps(_STATE.to_dict(), indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class HeartbeatServer:
    """
    HTTP server that receives trader heartbeats and monitors liveness.

    Usage:
      server = HeartbeatServer()
      await server.run()   # runs until cancelled

    Or as a background thread:
      server.start_thread()
    """

    def __init__(
        self,
        port: int = _PORT,
        status_file: Path | str | None = None,
    ) -> None:
        self.port = port
        self.status_file = Path(status_file) if status_file else _STATUS_FILE
        self._server: HTTPServer | None = None
        self._thread: Thread | None = None
        self._monitor_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the HTTP server and the liveness monitor concurrently."""
        self._server = HTTPServer(("0.0.0.0", self.port), _HeartbeatHandler)
        logger.info("HeartbeatServer: listening on port %d", self.port)

        loop = asyncio.get_event_loop()
        # Run blocking HTTPServer in executor
        serve_task = loop.run_in_executor(None, self._serve_forever)
        monitor_task = asyncio.ensure_future(self._monitor_loop())
        self._monitor_task = monitor_task

        try:
            await asyncio.gather(serve_task, monitor_task)
        except asyncio.CancelledError:
            pass
        finally:
            self._shutdown()

    def start_thread(self) -> None:
        """Start the HTTP server in a background daemon thread (non-async contexts)."""
        self._server = HTTPServer(("0.0.0.0", self.port), _HeartbeatHandler)
        self._thread = Thread(target=self._serve_forever, daemon=True)
        self._thread.start()
        logger.info("HeartbeatServer: background thread started on port %d", self.port)

    def get_state(self) -> dict[str, Any]:
        return _STATE.to_dict()

    def is_trader_alive(self) -> bool:
        return _STATE.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _serve_forever(self) -> None:
        if self._server:
            try:
                self._server.serve_forever()
            except Exception as exc:
                logger.error("HeartbeatServer: serve_forever error: %s", exc)

    async def _monitor_loop(self) -> None:
        """Check liveness every 15 seconds. Alert if trader goes quiet."""
        while True:
            await asyncio.sleep(15)
            status = _STATE.to_dict()
            self._write_status(status)

            if _STATE.last_seen > 0 and not _STATE.is_alive():
                lag = _STATE.seconds_since_beat()
                if not _STATE._alerted:
                    logger.critical(
                        "HeartbeatServer ALERT: No heartbeat for %.0fs "
                        "(threshold=%ds). Trader may have crashed!",
                        lag,
                        _HEARTBEAT_TIMEOUT_S,
                    )
                    _STATE._alerted = True
                    _STATE.down_seconds += 15.0
            else:
                # Reset alert flag when trader recovers
                if _STATE._alerted and _STATE.is_alive():
                    logger.info("HeartbeatServer: trader heartbeat recovered.")

    def _write_status(self, status: dict[str, Any]) -> None:
        try:
            tmp = self.status_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(status, indent=2))
            tmp.replace(self.status_file)
        except Exception as exc:
            logger.debug("HeartbeatServer: could not write status: %s", exc)

    def _shutdown(self) -> None:
        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass
        logger.info("HeartbeatServer: shutdown complete.")


# ------------------------------------------------------------------
# Convenience: run standalone
# ------------------------------------------------------------------

async def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    server = HeartbeatServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(_main())
