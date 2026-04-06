"""
tools/observability/log_streamer.py
=====================================
Real-time structured log streaming for the LARSA live trader.

Features
--------
- Intercepts Python's standard ``logging`` module via a custom handler.
- Enriches every log record with:
    trace_id, span_id, symbol (from context vars), equity, timestamp_ns
- Writes one JSON object per line to ``logs/structured/{YYYY-MM-DD}.jsonl``
- WebSocket broadcaster: pushes log lines to all connected clients on
  ``ws://localhost:8798/logs``
- Log level filtering and rate limiting (max 1000 lines/sec)
- HTTP search endpoint: ``GET /logs/search?q=BTC&level=ERROR&last=100``

Usage::

    from tools.observability.log_streamer import LogStreamer

    streamer = LogStreamer(ws_port=8798, log_dir="logs/structured")
    streamer.start()

    # (optional) set shared equity reference so logs include equity value
    streamer.set_equity_ref(lambda: my_trader.equity)

    streamer.stop()

Dependencies:
    Standard library (threading, http.server, logging, json, socket).
    WebSocket support requires ``websockets`` (optional; falls back to no-op).
"""

from __future__ import annotations

import asyncio
import collections
import gzip
import io
import json
import logging
import os
import queue
import re
import socket
import struct
import threading
import time
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import parse_qs, urlparse

# Context-variable imports (populated by TracingMiddleware)
try:
    from tools.observability.tracing import get_trace_id, get_span_id, get_current_symbol
    _TRACING_AVAILABLE = True
except Exception:
    _TRACING_AVAILABLE = False
    def get_trace_id(): return None  # type: ignore
    def get_span_id(): return None  # type: ignore
    def get_current_symbol(): return None  # type: ignore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Simple token-bucket rate limiter."""

    def __init__(self, rate: float) -> None:
        self._rate = rate          # tokens per second
        self._tokens = rate        # start full
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def consume(self) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self._rate,
                               self._tokens + (now - self._last) * self._rate)
            self._last = now
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False


# ---------------------------------------------------------------------------
# Minimal WebSocket server (RFC 6455) — no external dependency
# ---------------------------------------------------------------------------

_WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_handshake(client_sock: socket.socket, request_lines: List[bytes]) -> bool:
    """Perform the WebSocket upgrade handshake. Returns True on success."""
    import base64
    import hashlib

    key = None
    for line in request_lines:
        if line.lower().startswith(b"sec-websocket-key:"):
            key = line.split(b":", 1)[1].strip().decode()
            break
    if not key:
        return False

    accept = base64.b64encode(
        hashlib.sha1((key + _WS_MAGIC).encode()).digest()
    ).decode()

    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n"
        "\r\n"
    )
    try:
        client_sock.sendall(response.encode())
        return True
    except OSError:
        return False


def _ws_send(client_sock: socket.socket, message: str) -> bool:
    """Send a text frame to a WebSocket client. Returns False on error."""
    try:
        data = message.encode("utf-8")
        length = len(data)
        if length < 126:
            header = bytes([0x81, length])
        elif length < 65536:
            header = struct.pack(">BBH", 0x81, 126, length)
        else:
            header = struct.pack(">BBQ", 0x81, 127, length)
        client_sock.sendall(header + data)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# StructuredLogHandler
# ---------------------------------------------------------------------------

class StructuredLogHandler(logging.Handler):
    """
    Python logging handler that enriches records with tracing context and
    pushes them to a JSON queue for writing to disk and WebSocket broadcast.
    """

    def __init__(
        self,
        output_queue: "queue.Queue[str]",
        rate_limiter: _TokenBucket,
        equity_ref: Optional[Callable[[], float]] = None,
    ) -> None:
        super().__init__()
        self._queue = output_queue
        self._limiter = rate_limiter
        self._equity_ref = equity_ref
        self._dropped = 0

    def emit(self, record: logging.LogRecord) -> None:
        if not self._limiter.consume():
            self._dropped += 1
            return

        equity: Optional[float] = None
        if self._equity_ref is not None:
            try:
                equity = self._equity_ref()
            except Exception:
                pass

        entry = {
            "timestamp_ns": time.time_ns(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": self.format(record),
            "trace_id": get_trace_id(),
            "span_id": get_span_id(),
            "symbol": get_current_symbol(),
            "equity": equity,
            "thread": record.threadName,
            "filename": record.filename,
            "lineno": record.lineno,
        }

        if record.exc_info:
            import traceback
            entry["exception"] = "".join(traceback.format_exception(*record.exc_info))

        try:
            self._queue.put_nowait(json.dumps(entry, default=str))
        except queue.Full:
            self._dropped += 1


# ---------------------------------------------------------------------------
# LogStreamer
# ---------------------------------------------------------------------------

class LogStreamer:
    """
    Intercepts Python logging, writes structured JSONL files, and broadcasts
    over a minimal built-in WebSocket server.

    HTTP search endpoint
    --------------------
    ``GET /logs/search?q=<query>&level=<LEVEL>&last=<N>``

    Returns a JSON array of matching log lines (most-recent first).
    """

    def __init__(
        self,
        ws_port: int = 8798,
        log_dir: Optional[str] = None,
        max_rate: float = 1000.0,
        min_level: int = logging.DEBUG,
        in_memory_lines: int = 5000,
    ) -> None:
        if log_dir is None:
            _repo = Path(__file__).parents[2]
            log_dir = str(_repo / "logs" / "structured")
        self._log_dir = Path(log_dir)
        self._ws_port = ws_port
        self._min_level = min_level
        self._in_memory_lines = in_memory_lines

        # Shared queue between handler and writer thread
        self._queue: queue.Queue[str] = queue.Queue(maxsize=10_000)
        self._limiter = _TokenBucket(max_rate)
        self._handler = StructuredLogHandler(self._queue, self._limiter)
        self._handler.setLevel(min_level)

        # In-memory ring buffer for search
        self._buffer: collections.deque[str] = collections.deque(
            maxlen=in_memory_lines
        )
        self._buffer_lock = threading.Lock()

        # WebSocket client set
        self._ws_clients: Set[socket.socket] = set()
        self._ws_lock = threading.Lock()

        # Equity reference callable
        self._equity_ref: Optional[Callable[[], float]] = None

        # Threads
        self._running = False
        self._writer_thread: Optional[threading.Thread] = None
        self._ws_server_thread: Optional[threading.Thread] = None
        self._http_thread: Optional[threading.Thread] = None
        self._http_server: Optional[HTTPServer] = None

        self._current_log_date: Optional[date] = None
        self._log_file: Optional[io.TextIOWrapper] = None

    # ----------------------------------------------------------------- start/stop
    def start(self) -> None:
        if self._running:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._running = True

        # Install logging handler on root logger
        self._handler._equity_ref = self._equity_ref
        root_logger = logging.getLogger()
        root_logger.addHandler(self._handler)

        # Writer thread
        self._writer_thread = threading.Thread(
            target=self._write_loop, name="log-writer", daemon=True
        )
        self._writer_thread.start()

        # WebSocket server
        self._ws_server_thread = threading.Thread(
            target=self._ws_server_loop, name="log-ws-server", daemon=True
        )
        self._ws_server_thread.start()

        # HTTP search server
        self._start_http_server()

        log.info("LogStreamer started (ws=:%d, log_dir=%s)",
                 self._ws_port, self._log_dir)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        # Remove handler
        try:
            logging.getLogger().removeHandler(self._handler)
        except Exception:
            pass

        # Flush remaining queue items
        try:
            self._flush_queue()
        except Exception:
            pass

        # Close current log file
        if self._log_file:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass

        # Close WebSocket clients
        with self._ws_lock:
            for sock in list(self._ws_clients):
                try:
                    sock.close()
                except Exception:
                    pass
            self._ws_clients.clear()

        if self._http_server:
            try:
                self._http_server.server_close()
            except Exception:
                pass

        log.info("LogStreamer stopped")

    def set_equity_ref(self, fn: Callable[[], float]) -> None:
        """Set a callable that returns the current equity for log enrichment."""
        self._equity_ref = fn
        self._handler._equity_ref = fn

    # ----------------------------------------------------------------- writer loop
    def _get_log_file(self) -> io.TextIOWrapper:
        today = date.today()
        if self._current_log_date != today or self._log_file is None:
            if self._log_file:
                try:
                    self._log_file.flush()
                    self._log_file.close()
                except Exception:
                    pass
            path = self._log_dir / f"{today.isoformat()}.jsonl"
            self._log_file = open(path, "a", buffering=1, encoding="utf-8")
            self._current_log_date = today
        return self._log_file

    def _flush_queue(self) -> None:
        while True:
            try:
                line = self._queue.get_nowait()
                self._write_line(line)
            except queue.Empty:
                break

    def _write_line(self, line: str) -> None:
        # Write to file
        try:
            f = self._get_log_file()
            f.write(line + "\n")
        except Exception as exc:
            pass  # can't use log.* here — would recurse

        # Store in memory ring buffer
        with self._buffer_lock:
            self._buffer.append(line)

        # Broadcast to WebSocket clients
        self._ws_broadcast(line)

    def _write_loop(self) -> None:
        while self._running:
            try:
                line = self._queue.get(timeout=0.5)
                self._write_line(line)
            except queue.Empty:
                continue
            except Exception:
                pass

    # ----------------------------------------------------------------- WebSocket
    def _ws_broadcast(self, message: str) -> None:
        with self._ws_lock:
            dead: Set[socket.socket] = set()
            for sock in self._ws_clients:
                if not _ws_send(sock, message):
                    dead.add(sock)
            for sock in dead:
                self._ws_clients.discard(sock)
                try:
                    sock.close()
                except Exception:
                    pass

    def _ws_server_loop(self) -> None:
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind(("0.0.0.0", self._ws_port))
            server_sock.listen(10)
            server_sock.settimeout(1.0)
        except OSError as exc:
            log.error("LogStreamer WebSocket server bind error :%d — %s",
                      self._ws_port, exc)
            return

        log.info("LogStreamer WS server listening on :%d/logs", self._ws_port)

        while self._running:
            try:
                client_sock, addr = server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            t = threading.Thread(
                target=self._ws_client_handler,
                args=(client_sock, addr),
                daemon=True,
            )
            t.start()

        try:
            server_sock.close()
        except Exception:
            pass

    def _ws_client_handler(
        self, client_sock: socket.socket, addr: tuple
    ) -> None:
        client_sock.settimeout(60.0)
        try:
            # Read the HTTP upgrade request
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = client_sock.recv(1024)
                if not chunk:
                    return
                data += chunk
            lines = data.split(b"\r\n")

            # Only handle /logs path
            request_line = lines[0].decode("utf-8", errors="replace")
            if "/logs" not in request_line:
                client_sock.close()
                return

            if not _ws_handshake(client_sock, lines[1:]):
                client_sock.close()
                return

            with self._ws_lock:
                self._ws_clients.add(client_sock)

            # Keep the socket open until client disconnects or we stop
            client_sock.settimeout(5.0)
            while self._running:
                try:
                    data = client_sock.recv(128)
                    if not data:
                        break
                except socket.timeout:
                    continue
                except OSError:
                    break

        except Exception:
            pass
        finally:
            with self._ws_lock:
                self._ws_clients.discard(client_sock)
            try:
                client_sock.close()
            except Exception:
                pass

    # ----------------------------------------------------------------- HTTP search server
    def _start_http_server(self) -> None:
        streamer_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != "/logs/search":
                    self._send_json({"error": "not found"}, 404)
                    return

                params = parse_qs(parsed.query)
                query = params.get("q", [""])[0].lower()
                level_filter = params.get("level", [""])[0].upper()
                last_n = int(params.get("last", ["100"])[0])

                results = streamer_ref._search(query, level_filter, last_n)
                self._send_json(results, 200)

            def _send_json(self, data, status: int) -> None:
                body = json.dumps(data, indent=2).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # Bind on a different port to avoid conflict with the WS server
        http_port = self._ws_port + 1  # :8799 — but HealthServer uses that
        # Use 8800 for search
        http_port = 8800
        try:
            httpd = HTTPServer(("0.0.0.0", http_port), Handler)
            httpd.timeout = 1.0
            self._http_server = httpd
        except OSError as exc:
            log.warning("LogStreamer HTTP search server bind failed :%d — %s",
                        http_port, exc)
            return

        def _serve():
            log.info("LogStreamer search endpoint on :%d/logs/search", http_port)
            while self._running:
                try:
                    httpd.handle_request()
                except Exception:
                    pass

        self._http_thread = threading.Thread(
            target=_serve, name="log-search-http", daemon=True
        )
        self._http_thread.start()

    # ----------------------------------------------------------------- search
    def _search(
        self, query: str, level: str, last_n: int
    ) -> List[Dict[str, Any]]:
        with self._buffer_lock:
            lines = list(self._buffer)

        results: List[Dict[str, Any]] = []
        for raw in reversed(lines):
            try:
                obj = json.loads(raw)
            except Exception:
                continue

            if level and obj.get("level", "").upper() != level:
                continue
            if query:
                # Search in message + symbol + logger
                haystack = " ".join(filter(None, [
                    obj.get("message", ""),
                    obj.get("symbol", ""),
                    obj.get("logger", ""),
                ])).lower()
                if query not in haystack:
                    continue
            results.append(obj)
            if len(results) >= last_n:
                break

        return results

    # ----------------------------------------------------------------- context manager
    def __enter__(self) -> "LogStreamer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        n_clients = len(self._ws_clients)
        return (
            f"<LogStreamer ws_port={self._ws_port} "
            f"running={self._running} ws_clients={n_clients}>"
        )
