"""
tools/observability/health.py
================================
Comprehensive HTTP health-check server for the LARSA live trader.

Endpoints
---------
GET /health
    Full health report: all checks, uptime, version, overall status.
    Returns HTTP 200 when healthy, 503 when unhealthy.

GET /ready
    Kubernetes readiness probe — 200 when all critical checks pass,
    503 otherwise.

GET /live
    Kubernetes liveness probe — 200 as long as the process is running
    and the health server itself is responsive.

GET /metrics/json
    All registered metrics as a JSON object.  Does not require
    prometheus_client; uses the PrometheusMetrics singleton if available,
    otherwise returns an empty dict.

The server runs in a daemon thread on :8799 and is thread-safe.

Usage::

    from tools.observability.health import HealthServer

    health = HealthServer(port=8799, version="larsa_v17")
    health.start()

    # Register custom checks:
    health.register_check("broker_connected", health.check_broker_connected)

    # Periodically update shared state from the trading loop:
    health.set_state(
        broker_connected=True,
        stream_connected=True,
        db_accessible=True,
        last_bar_age_secs=12.3,
        equity=102_500,
        equity_timestamp=time.time(),
    )

    health.stop()

Dependencies:
    Standard library only (http.server, threading, json).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Thread-per-request HTTP server — handles concurrent health probes."""
    daemon_threads = True
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

_DEFAULT_PORT = 8799
_BAR_AGE_WARN_SECS = 90       # warn if last bar older than this
_BAR_AGE_CRIT_SECS = 300      # critical if last bar older than this
_EQUITY_STALE_SECS = 600      # equity is "stale" after this many seconds

# ---------------------------------------------------------------------------
# HealthServer
# ---------------------------------------------------------------------------

class HealthServer:
    """
    Thread-safe HTTP health-check server.

    State is stored in ``_state`` dict, updated via ``set_state()``.
    Custom check callables can be registered with ``register_check()``.

    All response bodies are JSON.
    """

    def __init__(
        self,
        port: int = _DEFAULT_PORT,
        version: str = "unknown",
        service_name: str = "larsa-live-trader",
    ) -> None:
        self._port = port
        self._version = version
        self._service_name = service_name
        self._start_time = time.time()
        self._lock = threading.RLock()

        # Mutable shared state updated by the trading loop
        self._state: Dict[str, Any] = {
            "broker_connected": False,
            "stream_connected": False,
            "db_accessible": False,
            "last_bar_age_secs": float("inf"),
            "equity": 0.0,
            "equity_timestamp": 0.0,
            "circuit_breaker_active": False,
        }

        # Custom check registry: name → callable() → (ok: bool, detail: str)
        self._checks: Dict[str, Callable[[], tuple[bool, str]]] = {}
        self._register_builtin_checks()

        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Optional metrics reference
        self._metrics_ref: Optional[Any] = None

    # ----------------------------------------------------------------- state
    def set_state(self, **kwargs: Any) -> None:
        """
        Update health-check state variables.

        Keys correspond to fields in ``self._state``.  Unknown keys are
        accepted and stored (future-proofing).
        """
        with self._lock:
            self._state.update(kwargs)

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of the current state dict."""
        with self._lock:
            return dict(self._state)

    def attach_metrics(self, metrics_instance: Any) -> None:
        """
        Attach a ``PrometheusMetrics`` (or duck-typed equivalent) instance
        so that ``/metrics/json`` can read live values.
        """
        self._metrics_ref = metrics_instance

    # ----------------------------------------------------------------- checks
    def register_check(
        self, name: str, fn: Callable[[], tuple[bool, str]]
    ) -> None:
        """
        Register a custom health-check function.

        Parameters
        ----------
        name:
            Short identifier, e.g. ``"redis_connected"``.
        fn:
            Zero-argument callable returning ``(ok: bool, detail: str)``.
        """
        with self._lock:
            self._checks[name] = fn

    def _register_builtin_checks(self) -> None:
        self._checks["broker_connected"] = self._check_broker
        self._checks["stream_connected"] = self._check_stream
        self._checks["db_accessible"] = self._check_db
        self._checks["last_bar_age"] = self._check_bar_age
        self._checks["equity_stale"] = self._check_equity_stale

    def _check_broker(self) -> tuple[bool, str]:
        ok = bool(self._state.get("broker_connected", False))
        return ok, "connected" if ok else "disconnected"

    def _check_stream(self) -> tuple[bool, str]:
        ok = bool(self._state.get("stream_connected", False))
        return ok, "connected" if ok else "disconnected"

    def _check_db(self) -> tuple[bool, str]:
        ok = bool(self._state.get("db_accessible", False))
        return ok, "accessible" if ok else "not accessible"

    def _check_bar_age(self) -> tuple[bool, str]:
        age = float(self._state.get("last_bar_age_secs", float("inf")))
        if age >= _BAR_AGE_CRIT_SECS:
            return False, f"no bar for {age:.0f}s (threshold {_BAR_AGE_CRIT_SECS}s)"
        if age >= _BAR_AGE_WARN_SECS:
            return True, f"warning: last bar {age:.0f}s ago"
        return True, f"{age:.1f}s ago"

    def _check_equity_stale(self) -> tuple[bool, str]:
        ts = float(self._state.get("equity_timestamp", 0))
        if ts == 0:
            return False, "equity never updated"
        age = time.time() - ts
        if age > _EQUITY_STALE_SECS:
            return False, f"equity stale ({age:.0f}s)"
        return True, f"updated {age:.1f}s ago"

    # ----------------------------------------------------------------- run checks
    def _run_all_checks(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        all_ok = True
        with self._lock:
            checks = dict(self._checks)

        for name, fn in checks.items():
            try:
                ok, detail = fn()
            except Exception as exc:
                ok, detail = False, f"exception: {exc}"
            results[name] = {"ok": ok, "detail": detail}
            if not ok:
                all_ok = False

        return {"checks": results, "ok": all_ok}

    def _build_health_response(self) -> Dict[str, Any]:
        check_data = self._run_all_checks()
        uptime = time.time() - self._start_time
        return {
            "service": self._service_name,
            "version": self._version,
            "status": "healthy" if check_data["ok"] else "unhealthy",
            "ok": check_data["ok"],
            "checks": check_data["checks"],
            "uptime_secs": round(uptime, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _build_metrics_json(self) -> Dict[str, Any]:
        """Collect live metric values from the attached PrometheusMetrics."""
        if self._metrics_ref is None:
            try:
                from tools.observability.metrics import get_metrics
                self._metrics_ref = get_metrics()
            except Exception:
                return {}

        state = self.get_state()
        return {
            "equity_usd": state.get("equity", 0.0),
            "last_bar_age_secs": state.get("last_bar_age_secs", None),
            "broker_connected": state.get("broker_connected", False),
            "stream_connected": state.get("stream_connected", False),
            "circuit_breaker_active": state.get("circuit_breaker_active", False),
            "uptime_secs": round(time.time() - self._start_time, 1),
        }

    # ----------------------------------------------------------------- HTTP server
    def start(self) -> None:
        """Start the health-check HTTP server in a daemon thread."""
        if self._running:
            return

        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args) -> None:  # silence access logs
                pass

            def _send_json(self, data: dict, status: int = 200) -> None:
                body = json.dumps(data, indent=2).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?")[0].rstrip("/")

                if path == "/health":
                    data = server_ref._build_health_response()
                    self._send_json(data, 200 if data["ok"] else 503)

                elif path == "/ready":
                    data = server_ref._run_all_checks()
                    body = {
                        "ready": data["ok"],
                        "checks": data["checks"],
                    }
                    self._send_json(body, 200 if data["ok"] else 503)

                elif path == "/live":
                    self._send_json({"alive": True}, 200)

                elif path == "/metrics/json":
                    data = server_ref._build_metrics_json()
                    self._send_json(data, 200)

                else:
                    self._send_json(
                        {"error": "not found", "path": path}, 404
                    )

        try:
            httpd = _ThreadingHTTPServer(("0.0.0.0", self._port), Handler)
        except OSError as exc:
            log.error("HealthServer failed to bind on port %d: %s",
                      self._port, exc)
            return

        self._server = httpd
        self._running = True

        def _serve():
            log.info("HealthServer: listening on :%d", self._port)
            try:
                httpd.serve_forever(poll_interval=0.5)
            except Exception as exc:
                if self._running:
                    log.error("HealthServer crashed: %s", exc)
            finally:
                try:
                    httpd.server_close()
                except Exception:
                    pass
                log.info("HealthServer: stopped")

        self._thread = threading.Thread(
            target=_serve, name="health-server", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the health-check server."""
        self._running = False
        if self._server:
            try:
                self._server.shutdown()   # unblocks serve_forever()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    # ----------------------------------------------------------------- context manager
    def __enter__(self) -> "HealthServer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"<HealthServer port={self._port} "
            f"running={self._running} version={self._version!r}>"
        )
