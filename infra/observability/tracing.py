"""
tracing.py -- Distributed tracing for the SRFM quantitative trading system.

Provides:
    TraceContext      # immutable span data container
    Tracer            # span lifecycle management with context manager support
    TraceExporter     # export to Jaeger thrift, Zipkin V2 JSON, or local SQLite
    TraceStore        # SQLite-backed trace storage and query engine
    Pre-instrumented helpers for SRFM hot paths

No external tracing libraries -- pure stdlib + sqlite3.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, List, Optional, Tuple
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# TraceContext
# ---------------------------------------------------------------------------

@dataclass
class TraceContext:
    """Single span in a distributed trace.

    Fields
    ------
    trace_id         # globally unique trace identifier (hex string)
    span_id          # identifier for this specific span (hex string)
    parent_span_id   # span_id of the parent span, None for root spans
    service_name     # name of the emitting service (e.g. "order_router")
    operation        # human-readable operation name (e.g. "submit_order")
    start_time_ns    # epoch time in nanoseconds when span was started
    end_time_ns      # epoch time in nanoseconds when span was finished
    tags             # key-value metadata attached to the span
    logs             # timestamped structured log events inside the span
    status           # terminal state: "ok", "error", or "timeout"
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    service_name: str
    operation: str
    start_time_ns: int
    end_time_ns: Optional[int] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict] = field(default_factory=list)
    status: str = "ok"

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def duration_ns(self) -> Optional[int]:
        """Span duration in nanoseconds, None if not yet finished."""
        if self.end_time_ns is None:
            return None
        return self.end_time_ns - self.start_time_ns

    @property
    def duration_ms(self) -> Optional[float]:
        """Span duration in milliseconds, None if not yet finished."""
        d = self.duration_ns
        if d is None:
            return None
        return d / 1_000_000.0

    @property
    def is_root(self) -> bool:
        """True when this span has no parent."""
        return self.parent_span_id is None

    def to_dict(self) -> Dict:
        """Serialize to a plain dictionary suitable for JSON encoding."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "service_name": self.service_name,
            "operation": self.operation,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
            "duration_ms": self.duration_ms,
        }


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class Tracer:
    """Manages span creation, annotation, and lifecycle.

    Thread-safe -- each thread automatically gets its own active-span stack
    via threading.local().

    Usage
    -----
        tracer = Tracer(service_name="order_router")

        # Low-level API
        ctx = tracer.start_span("submit_order")
        tracer.tag(ctx, "symbol", "AAPL")
        tracer.log_event(ctx, "validation_passed", {"checks": 5})
        tracer.finish_span(ctx)

        # Context-manager API (preferred)
        with tracer.trace("submit_order") as ctx:
            tracer.tag(ctx, "symbol", "AAPL")
            ...  # span auto-finished on exit, status="error" on exception
    """

    def __init__(
        self,
        service_name: str,
        on_finish: Optional[Callable[[TraceContext], None]] = None,
    ) -> None:
        self.service_name = service_name
        self._on_finish = on_finish
        self._local = threading.local()
        self._completed: List[TraceContext] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex

    def _stack(self) -> List[TraceContext]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_span(
        self,
        operation: str,
        parent: Optional[TraceContext] = None,
    ) -> TraceContext:
        """Create and activate a new span.

        If parent is None, the top of the current thread's stack is used as
        the implicit parent.  Pass parent=False (or call with an explicit
        root trace_id) to force a root span.
        """
        stack = self._stack()

        if parent is None and stack:
            parent = stack[-1]

        trace_id = parent.trace_id if parent else self._new_id()
        parent_span_id = parent.span_id if parent else None

        ctx = TraceContext(
            trace_id=trace_id,
            span_id=self._new_id(),
            parent_span_id=parent_span_id,
            service_name=self.service_name,
            operation=operation,
            start_time_ns=time.time_ns(),
        )
        stack.append(ctx)
        return ctx

    def finish_span(self, ctx: TraceContext, status: str = "ok") -> None:
        """Mark span complete and deliver to the on_finish callback."""
        ctx.end_time_ns = time.time_ns()
        ctx.status = status

        stack = self._stack()
        if ctx in stack:
            stack.remove(ctx)

        with self._lock:
            self._completed.append(ctx)

        if self._on_finish is not None:
            try:
                self._on_finish(ctx)
            except Exception:
                pass  # never let callback kill caller

    def log_event(
        self,
        ctx: TraceContext,
        event: str,
        payload: Optional[Dict] = None,
    ) -> None:
        """Append a timestamped log entry to the span."""
        entry: Dict = {
            "timestamp_ns": time.time_ns(),
            "event": event,
        }
        if payload:
            entry.update(payload)
        ctx.logs.append(entry)

    def tag(self, ctx: TraceContext, key: str, value: str) -> None:
        """Attach a string tag to the span."""
        ctx.tags[key] = value

    @contextmanager
    def trace(
        self,
        operation: str,
        parent: Optional[TraceContext] = None,
    ) -> Generator[TraceContext, None, None]:
        """Context manager that auto-finishes a span.

        Sets status="error" if an exception propagates through.
        """
        ctx = self.start_span(operation, parent=parent)
        try:
            yield ctx
            self.finish_span(ctx, status="ok")
        except Exception as exc:
            self.tag(ctx, "error.message", str(exc)[:256])
            self.tag(ctx, "error.type", type(exc).__name__)
            self.finish_span(ctx, status="error")
            raise

    def flush(self) -> List[TraceContext]:
        """Return and clear completed spans."""
        with self._lock:
            out = list(self._completed)
            self._completed.clear()
        return out

    def current_span(self) -> Optional[TraceContext]:
        """Return the innermost active span for the current thread."""
        stack = self._stack()
        return stack[-1] if stack else None


# ---------------------------------------------------------------------------
# TraceExporter
# ---------------------------------------------------------------------------

class TraceExporter:
    """Export completed traces to external backends.

    Supports:
        export_jaeger  # HTTP Thrift compact encoding to Jaeger collector
        export_zipkin  # Zipkin V2 JSON over HTTP
        export_sqlite  # local SQLite file (useful for dev / CI)
    """

    # ------------------------------------------------------------------
    # Jaeger
    # ------------------------------------------------------------------

    @staticmethod
    def _span_to_jaeger_dict(ctx: TraceContext) -> Dict:
        """Convert a TraceContext to a Jaeger-compatible span dict."""
        tags = [
            {"key": k, "vType": "STRING", "vStr": v}
            for k, v in ctx.tags.items()
        ]
        tags.append({"key": "status", "vType": "STRING", "vStr": ctx.status})

        logs = []
        for log_entry in ctx.logs:
            fields = [
                {"key": k, "vType": "STRING", "vStr": str(v)}
                for k, v in log_entry.items()
                if k != "timestamp_ns"
            ]
            logs.append({
                "timestamp": log_entry.get("timestamp_ns", ctx.start_time_ns) // 1000,
                "fields": fields,
            })

        references = []
        if ctx.parent_span_id:
            references.append({
                "refType": "CHILD_OF",
                "traceId": ctx.trace_id,
                "spanId": ctx.parent_span_id,
            })

        return {
            "traceId": ctx.trace_id,
            "spanId": ctx.span_id,
            "operationName": ctx.operation,
            "references": references,
            "startTime": ctx.start_time_ns // 1000,  # microseconds
            "duration": (ctx.duration_ns or 0) // 1000,
            "tags": tags,
            "logs": logs,
            "processId": "p1",
        }

    @staticmethod
    def export_jaeger(
        traces: List[TraceContext],
        endpoint: str,
        timeout_s: float = 5.0,
    ) -> bool:
        """POST traces to a Jaeger collector HTTP endpoint.

        endpoint example: "http://localhost:14268/api/traces"

        Returns True on success, False on network/HTTP failure.
        Uses Jaeger's HTTP JSON API (not Thrift binary) for simplicity
        without requiring the thrift library.
        """
        if not traces:
            return True

        # Group by trace_id
        trace_groups: Dict[str, List[Dict]] = {}
        for ctx in traces:
            jspan = TraceExporter._span_to_jaeger_dict(ctx)
            trace_groups.setdefault(ctx.trace_id, []).append(jspan)

        payload = {
            "data": [
                {
                    "traceID": tid,
                    "spans": spans,
                    "processes": {
                        "p1": {
                            "serviceName": traces[0].service_name,
                            "tags": [],
                        }
                    },
                }
                for tid, spans in trace_groups.items()
            ]
        }

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.status < 400
        except (urllib.error.URLError, OSError):
            return False

    # ------------------------------------------------------------------
    # Zipkin
    # ------------------------------------------------------------------

    @staticmethod
    def _span_to_zipkin_dict(ctx: TraceContext) -> Dict:
        """Convert a TraceContext to Zipkin V2 span format."""
        span: Dict = {
            "traceId": ctx.trace_id,
            "id": ctx.span_id,
            "name": ctx.operation,
            "timestamp": ctx.start_time_ns // 1000,  # microseconds
            "duration": (ctx.duration_ns or 0) // 1000,
            "localEndpoint": {"serviceName": ctx.service_name},
            "tags": dict(ctx.tags),
        }
        if ctx.parent_span_id:
            span["parentId"] = ctx.parent_span_id
        if ctx.status != "ok":
            span["tags"]["error"] = ctx.status
        if ctx.logs:
            span["annotations"] = [
                {
                    "timestamp": e.get("timestamp_ns", ctx.start_time_ns) // 1000,
                    "value": e.get("event", ""),
                }
                for e in ctx.logs
            ]
        return span

    @staticmethod
    def export_zipkin(
        traces: List[TraceContext],
        endpoint: str,
        timeout_s: float = 5.0,
    ) -> bool:
        """POST spans to a Zipkin V2 HTTP endpoint.

        endpoint example: "http://localhost:9411/api/v2/spans"

        Returns True on success, False on failure.
        """
        if not traces:
            return True

        payload = [TraceExporter._span_to_zipkin_dict(ctx) for ctx in traces]
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.status < 400
        except (urllib.error.URLError, OSError):
            return False

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------

    @staticmethod
    def export_sqlite(
        traces: List[TraceContext],
        db_path: str,
    ) -> None:
        """Persist traces to a local SQLite database.

        Creates the traces table if it does not already exist.
        Suitable for development, CI, and offline analysis.
        """
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id        TEXT NOT NULL,
                    span_id         TEXT NOT NULL PRIMARY KEY,
                    parent_span_id  TEXT,
                    service_name    TEXT NOT NULL,
                    operation       TEXT NOT NULL,
                    start_time_ns   INTEGER NOT NULL,
                    end_time_ns     INTEGER,
                    duration_ns     INTEGER,
                    tags            TEXT,
                    logs            TEXT,
                    status          TEXT NOT NULL DEFAULT 'ok'
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_trace_id ON traces(trace_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_service ON traces(service_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_start ON traces(start_time_ns)"
            )

            rows = [
                (
                    ctx.trace_id,
                    ctx.span_id,
                    ctx.parent_span_id,
                    ctx.service_name,
                    ctx.operation,
                    ctx.start_time_ns,
                    ctx.end_time_ns,
                    ctx.duration_ns,
                    json.dumps(ctx.tags),
                    json.dumps(ctx.logs),
                    ctx.status,
                )
                for ctx in traces
            ]
            conn.executemany(
                """INSERT OR REPLACE INTO traces
                   (trace_id, span_id, parent_span_id, service_name, operation,
                    start_time_ns, end_time_ns, duration_ns, tags, logs, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# TraceStore
# ---------------------------------------------------------------------------

class TraceStore:
    """SQLite-backed trace storage and query engine.

    Designed as a ring buffer -- old spans are pruned once the store exceeds
    max_spans entries.

    All public methods are thread-safe.
    """

    DEFAULT_DB = ":memory:"
    DEFAULT_MAX_SPANS = 1_000_000

    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        max_spans: int = DEFAULT_MAX_SPANS,
    ) -> None:
        self._db_path = db_path
        self._max_spans = max_spans
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    rowid           INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id        TEXT NOT NULL,
                    span_id         TEXT NOT NULL UNIQUE,
                    parent_span_id  TEXT,
                    service_name    TEXT NOT NULL,
                    operation       TEXT NOT NULL,
                    start_time_ns   INTEGER NOT NULL,
                    end_time_ns     INTEGER,
                    duration_ns     INTEGER,
                    tags            TEXT DEFAULT '{}',
                    logs            TEXT DEFAULT '[]',
                    status          TEXT NOT NULL DEFAULT 'ok'
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts_trace   ON traces(trace_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts_service ON traces(service_name)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts_op      ON traces(operation)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts_start   ON traces(start_time_ns)"
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(self, ctx: TraceContext) -> None:
        """Persist a single span."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO traces
                   (trace_id, span_id, parent_span_id, service_name, operation,
                    start_time_ns, end_time_ns, duration_ns, tags, logs, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ctx.trace_id,
                    ctx.span_id,
                    ctx.parent_span_id,
                    ctx.service_name,
                    ctx.operation,
                    ctx.start_time_ns,
                    ctx.end_time_ns,
                    ctx.duration_ns,
                    json.dumps(ctx.tags),
                    json.dumps(ctx.logs),
                    ctx.status,
                ),
            )
            self._conn.commit()
            self._prune()

    def store_many(self, spans: List[TraceContext]) -> None:
        """Bulk-insert a list of spans."""
        if not spans:
            return
        rows = [
            (
                ctx.trace_id,
                ctx.span_id,
                ctx.parent_span_id,
                ctx.service_name,
                ctx.operation,
                ctx.start_time_ns,
                ctx.end_time_ns,
                ctx.duration_ns,
                json.dumps(ctx.tags),
                json.dumps(ctx.logs),
                ctx.status,
            )
            for ctx in spans
        ]
        with self._lock:
            self._conn.executemany(
                """INSERT OR REPLACE INTO traces
                   (trace_id, span_id, parent_span_id, service_name, operation,
                    start_time_ns, end_time_ns, duration_ns, tags, logs, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            self._conn.commit()
            self._prune()

    def _prune(self) -> None:
        """Remove oldest rows when max_spans is exceeded (call with lock held)."""
        count = self._conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        if count > self._max_spans:
            excess = count - self._max_spans
            self._conn.execute(
                "DELETE FROM traces WHERE rowid IN "
                "(SELECT rowid FROM traces ORDER BY rowid ASC LIMIT ?)",
                (excess,),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_ctx(row: Tuple) -> TraceContext:
        (
            trace_id, span_id, parent_span_id, service_name, operation,
            start_time_ns, end_time_ns, duration_ns, tags_json, logs_json, status,
        ) = row
        return TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            service_name=service_name,
            operation=operation,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            tags=json.loads(tags_json or "{}"),
            logs=json.loads(logs_json or "[]"),
            status=status,
        )

    _SELECT = (
        "SELECT trace_id, span_id, parent_span_id, service_name, operation, "
        "start_time_ns, end_time_ns, duration_ns, tags, logs, status FROM traces"
    )

    def get_trace(self, trace_id: str) -> List[TraceContext]:
        """Return all spans belonging to a trace."""
        with self._lock:
            rows = self._conn.execute(
                f"{self._SELECT} WHERE trace_id = ? ORDER BY start_time_ns",
                (trace_id,),
            ).fetchall()
        return [self._row_to_ctx(r) for r in rows]

    def get_by_service(
        self,
        service_name: str,
        limit: int = 200,
    ) -> List[TraceContext]:
        """Return recent spans for a given service."""
        with self._lock:
            rows = self._conn.execute(
                f"{self._SELECT} WHERE service_name = ? "
                "ORDER BY start_time_ns DESC LIMIT ?",
                (service_name, limit),
            ).fetchall()
        return [self._row_to_ctx(r) for r in rows]

    def get_by_operation(
        self,
        operation: str,
        limit: int = 200,
    ) -> List[TraceContext]:
        """Return recent spans for a given operation."""
        with self._lock:
            rows = self._conn.execute(
                f"{self._SELECT} WHERE operation = ? "
                "ORDER BY start_time_ns DESC LIMIT ?",
                (operation, limit),
            ).fetchall()
        return [self._row_to_ctx(r) for r in rows]

    def get_by_time_range(
        self,
        start_ns: int,
        end_ns: int,
        service_name: Optional[str] = None,
    ) -> List[TraceContext]:
        """Return all spans whose start_time falls within [start_ns, end_ns]."""
        with self._lock:
            if service_name:
                rows = self._conn.execute(
                    f"{self._SELECT} WHERE start_time_ns >= ? "
                    "AND start_time_ns <= ? AND service_name = ? "
                    "ORDER BY start_time_ns",
                    (start_ns, end_ns, service_name),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"{self._SELECT} WHERE start_time_ns >= ? "
                    "AND start_time_ns <= ? ORDER BY start_time_ns",
                    (start_ns, end_ns),
                ).fetchall()
        return [self._row_to_ctx(r) for r in rows]

    def get_errors(
        self,
        service_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[TraceContext]:
        """Return recent error spans, optionally filtered by service."""
        with self._lock:
            if service_name:
                rows = self._conn.execute(
                    f"{self._SELECT} WHERE status != 'ok' AND service_name = ? "
                    "ORDER BY start_time_ns DESC LIMIT ?",
                    (service_name, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"{self._SELECT} WHERE status != 'ok' "
                    "ORDER BY start_time_ns DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_ctx(r) for r in rows]

    def p99_latency_ms(self, operation: str, last_n: int = 1000) -> Optional[float]:
        """Compute p99 latency in milliseconds for a given operation."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT duration_ns FROM traces WHERE operation = ? "
                "AND duration_ns IS NOT NULL ORDER BY start_time_ns DESC LIMIT ?",
                (operation, last_n),
            ).fetchall()
        if not rows:
            return None
        values = sorted(r[0] for r in rows)
        idx = int(len(values) * 0.99)
        idx = min(idx, len(values) - 1)
        return values[idx] / 1_000_000.0

    def span_count(self) -> int:
        """Return total number of spans in the store."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM traces"
            ).fetchone()[0]

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Pre-instrumented SRFM hot-path helpers
# ---------------------------------------------------------------------------

def _default_tracer() -> Tracer:
    """Return the module-level default tracer (lazy singleton)."""
    global _MODULE_TRACER
    if _MODULE_TRACER is None:
        _MODULE_TRACER = Tracer(service_name="srfm")
    return _MODULE_TRACER


_MODULE_TRACER: Optional[Tracer] = None


def set_default_tracer(tracer: Tracer) -> None:
    """Override the module-level default tracer."""
    global _MODULE_TRACER
    _MODULE_TRACER = tracer


def trace_order_submission(
    symbol: str,
    qty: float,
    price: float,
    parent: Optional[TraceContext] = None,
) -> TraceContext:
    """Start a pre-configured span for order submission.

    Attaches symbol, qty, and price as tags so they appear in all backends.
    """
    tracer = _default_tracer()
    ctx = tracer.start_span("order_submission", parent=parent)
    tracer.tag(ctx, "symbol", symbol)
    tracer.tag(ctx, "qty", str(qty))
    tracer.tag(ctx, "price", str(price))
    tracer.tag(ctx, "component", "order_router")
    return ctx


def trace_signal_computation(
    symbol: str,
    timeframe: str,
    parent: Optional[TraceContext] = None,
) -> TraceContext:
    """Start a pre-configured span for signal computation."""
    tracer = _default_tracer()
    ctx = tracer.start_span("signal_computation", parent=parent)
    tracer.tag(ctx, "symbol", symbol)
    tracer.tag(ctx, "timeframe", timeframe)
    tracer.tag(ctx, "component", "signal_engine")
    return ctx


def trace_risk_check(
    order_id: str,
    parent: Optional[TraceContext] = None,
) -> TraceContext:
    """Start a pre-configured span for pre-trade risk check."""
    tracer = _default_tracer()
    ctx = tracer.start_span("risk_check", parent=parent)
    tracer.tag(ctx, "order_id", order_id)
    tracer.tag(ctx, "component", "risk_manager")
    return ctx


def trace_param_update(
    source: str,
    parent: Optional[TraceContext] = None,
) -> TraceContext:
    """Start a pre-configured span for parameter update propagation."""
    tracer = _default_tracer()
    ctx = tracer.start_span("param_update", parent=parent)
    tracer.tag(ctx, "source", source)
    tracer.tag(ctx, "component", "param_coordinator")
    return ctx


# ---------------------------------------------------------------------------
# Background exporter -- periodically flushes tracer to a store
# ---------------------------------------------------------------------------

class BackgroundExporter:
    """Periodically flushes completed spans from a Tracer to a TraceStore.

    Also optionally forwards to Jaeger or Zipkin endpoints.

    Usage
    -----
        store = TraceStore("/var/data/srfm_traces.db")
        exporter = BackgroundExporter(tracer, store, interval_s=5.0)
        exporter.start()
        ...
        exporter.stop()
    """

    def __init__(
        self,
        tracer: Tracer,
        store: TraceStore,
        interval_s: float = 5.0,
        jaeger_endpoint: Optional[str] = None,
        zipkin_endpoint: Optional[str] = None,
    ) -> None:
        self._tracer = tracer
        self._store = store
        self._interval_s = interval_s
        self._jaeger_endpoint = jaeger_endpoint
        self._zipkin_endpoint = zipkin_endpoint
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background flush thread."""
        self._thread = threading.Thread(
            target=self._run,
            name="srfm-trace-exporter",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 10.0) -> None:
        """Signal stop and wait for the flush thread to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self._flush()
        self._flush()  # final flush on shutdown

    def _flush(self) -> None:
        spans = self._tracer.flush()
        if not spans:
            return
        self._store.store_many(spans)
        if self._jaeger_endpoint:
            TraceExporter.export_jaeger(spans, self._jaeger_endpoint)
        if self._zipkin_endpoint:
            TraceExporter.export_zipkin(spans, self._zipkin_endpoint)
