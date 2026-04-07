"""
log_aggregator.py -- Structured log aggregation for the SRFM trading system.

Provides:
    LogEntry          # immutable structured log record
    StructuredLogger  # service-level logger that emits JSON log entries
    LogAggregator     # multi-source collector (file tail + HTTP push)
    LogAnalyzer       # error-rate analysis, pattern alerting, anomaly detection
    LogHTTPHandler    # wsgiref-based HTTP server for push endpoints

Architecture
------------
    Services write JSON lines via StructuredLogger.
    LogAggregator tails those files in background threads and indexes them
    into a local SQLite database.
    LogAnalyzer queries that database for dashboards and alert evaluation.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Deque, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LEVEL_NUM: Dict[str, int] = {lvl: i for i, lvl in enumerate(LEVELS)}


@dataclass
class LogEntry:
    """Single structured log record.

    Fields
    ------
    timestamp_ns  # epoch time in nanoseconds
    level         # log level string: DEBUG, INFO, WARNING, ERROR, CRITICAL
    service       # emitting service name
    message       # human-readable log message
    fields        # additional structured key-value context
    trace_id      # optional distributed trace correlation ID
    """

    timestamp_ns: int
    level: str
    service: str
    message: str
    fields: Dict = field(default_factory=dict)
    trace_id: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict) -> "LogEntry":
        return cls(
            timestamp_ns=int(d.get("timestamp_ns", 0)),
            level=d.get("level", "INFO").upper(),
            service=d.get("service", "unknown"),
            message=d.get("message", ""),
            fields=d.get("fields", {}),
            trace_id=d.get("trace_id"),
        )

    def to_dict(self) -> Dict:
        return {
            "timestamp_ns": self.timestamp_ns,
            "level": self.level,
            "service": self.service,
            "message": self.message,
            "fields": self.fields,
            "trace_id": self.trace_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ns / 1_000_000_000.0


# ---------------------------------------------------------------------------
# StructuredLogger
# ---------------------------------------------------------------------------

class StructuredLogger:
    """Writes structured JSON log entries to a file or stream.

    All **context keyword arguments become fields in the log entry so they
    can be queried later.

    Usage
    -----
        logger = StructuredLogger("order_router", path="/var/log/srfm/order_router.log")
        logger.info("Order submitted", symbol="AAPL", qty=100, price=175.5)
        logger.error("Fill rejected", reason="circuit_breaker", order_id="ord-001")
    """

    def __init__(
        self,
        service: str,
        path: Optional[str] = None,
        trace_id: Optional[str] = None,
        min_level: str = "DEBUG",
    ) -> None:
        self.service = service
        self._trace_id = trace_id
        self._min_level_num = LEVEL_NUM.get(min_level.upper(), 0)
        self._lock = threading.Lock()
        self._file: Optional[io.TextIOWrapper] = None
        if path:
            self._file = open(path, "a", encoding="utf-8", buffering=1)

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def bind(self, **context) -> "StructuredLogger":
        """Return a child logger with extra fields always attached."""
        child = StructuredLogger(
            service=self.service,
            min_level=LEVELS[self._min_level_num],
        )
        child._trace_id = self._trace_id
        child._file = self._file
        child._lock = self._lock
        child._bound: Dict = {**getattr(self, "_bound", {}), **context}
        return child

    def set_trace_id(self, trace_id: str) -> None:
        self._trace_id = trace_id

    # ------------------------------------------------------------------
    # Core emit
    # ------------------------------------------------------------------

    def _emit(self, level: str, msg: str, **context) -> LogEntry:
        if LEVEL_NUM.get(level, 0) < self._min_level_num:
            # Create the entry anyway so callers can inspect it in tests
            pass
        all_fields = {**getattr(self, "_bound", {}), **context}
        entry = LogEntry(
            timestamp_ns=time.time_ns(),
            level=level,
            service=self.service,
            message=msg,
            fields=all_fields,
            trace_id=self._trace_id,
        )
        if self._file is not None:
            with self._lock:
                self._file.write(entry.to_json() + "\n")
        return entry

    # ------------------------------------------------------------------
    # Level methods
    # ------------------------------------------------------------------

    def debug(self, msg: str, **context) -> LogEntry:
        return self._emit("DEBUG", msg, **context)

    def info(self, msg: str, **context) -> LogEntry:
        return self._emit("INFO", msg, **context)

    def warning(self, msg: str, **context) -> LogEntry:
        return self._emit("WARNING", msg, **context)

    def error(self, msg: str, **context) -> LogEntry:
        return self._emit("ERROR", msg, **context)

    def critical(self, msg: str, **context) -> LogEntry:
        return self._emit("CRITICAL", msg, **context)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


# ---------------------------------------------------------------------------
# SQLite log store
# ---------------------------------------------------------------------------

class _LogStore:
    """SQLite ring-buffer for structured log entries.

    Schema: logs(rowid, timestamp_ns, level, service, message, fields_json, trace_id)
    """

    DEFAULT_MAX_ENTRIES = 5_000_000

    def __init__(
        self,
        db_path: str = ":memory:",
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._db_path = db_path
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()
        self._write_count = 0
        self._prune_interval = 50_000

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    rowid        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_ns INTEGER NOT NULL,
                    level        TEXT    NOT NULL,
                    service      TEXT    NOT NULL,
                    message      TEXT    NOT NULL,
                    fields_json  TEXT    DEFAULT '{}',
                    trace_id     TEXT
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_level   ON logs(level)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_service ON logs(service)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_ts      ON logs(timestamp_ns)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_trace   ON logs(trace_id)"
            )
            self._conn.commit()

    def insert(self, entry: LogEntry) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO logs(timestamp_ns, level, service, message, fields_json, trace_id) "
                "VALUES (?,?,?,?,?,?)",
                (
                    entry.timestamp_ns,
                    entry.level,
                    entry.service,
                    entry.message,
                    json.dumps(entry.fields),
                    entry.trace_id,
                ),
            )
            self._conn.commit()
            self._write_count += 1
            if self._write_count % self._prune_interval == 0:
                self._prune()

    def insert_many(self, entries: List[LogEntry]) -> None:
        if not entries:
            return
        rows = [
            (e.timestamp_ns, e.level, e.service, e.message,
             json.dumps(e.fields), e.trace_id)
            for e in entries
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO logs(timestamp_ns, level, service, message, fields_json, trace_id) "
                "VALUES (?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()
            self._write_count += len(rows)
            if self._write_count % self._prune_interval == 0:
                self._prune()

    def _prune(self) -> None:
        """Remove oldest rows when count exceeds max_entries (lock held by caller)."""
        count = self._conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        if count > self._max_entries:
            excess = count - self._max_entries
            self._conn.execute(
                "DELETE FROM logs WHERE rowid IN "
                "(SELECT rowid FROM logs ORDER BY rowid ASC LIMIT ?)",
                (excess,),
            )
            self._conn.commit()

    def query(
        self,
        level: Optional[str] = None,
        service: Optional[str] = None,
        since_ns: Optional[int] = None,
        text_search: Optional[str] = None,
        n: int = 100,
    ) -> List[LogEntry]:
        clauses: List[str] = []
        params: List = []

        if level is not None:
            lvl_num = LEVEL_NUM.get(level.upper(), 0)
            level_list = [lvl for lvl, num in LEVEL_NUM.items() if num >= lvl_num]
            placeholders = ",".join("?" * len(level_list))
            clauses.append(f"level IN ({placeholders})")
            params.extend(level_list)

        if service is not None:
            clauses.append("service = ?")
            params.append(service)

        if since_ns is not None:
            clauses.append("timestamp_ns >= ?")
            params.append(since_ns)

        if text_search is not None:
            clauses.append("message LIKE ?")
            params.append(f"%{text_search}%")

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(n)

        with self._lock:
            rows = self._conn.execute(
                f"SELECT timestamp_ns, level, service, message, fields_json, trace_id "
                f"FROM logs {where} ORDER BY timestamp_ns DESC LIMIT ?",
                params,
            ).fetchall()

        return [
            LogEntry(
                timestamp_ns=r[0],
                level=r[1],
                service=r[2],
                message=r[3],
                fields=json.loads(r[4] or "{}"),
                trace_id=r[5],
            )
            for r in rows
        ]

    def count_since(
        self,
        since_ns: int,
        level: Optional[str] = None,
        service: Optional[str] = None,
    ) -> int:
        clauses = ["timestamp_ns >= ?"]
        params: List = [since_ns]
        if level:
            clauses.append("level = ?")
            params.append(level.upper())
        if service:
            clauses.append("service = ?")
            params.append(service)
        where = "WHERE " + " AND ".join(clauses)
        with self._lock:
            return self._conn.execute(
                f"SELECT COUNT(*) FROM logs {where}", params
            ).fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# LogAggregator
# ---------------------------------------------------------------------------

class LogAggregator:
    """Multi-source structured log collector.

    Sources
    -------
    tail_file(path, service)     # background thread reads new lines as they appear
    register_push_endpoint(...)  # start an HTTP server that accepts POST batches
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._store = _LogStore(db_path=db_path)
        self._tail_threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._http_server: Optional[HTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Push ingest -- direct
    # ------------------------------------------------------------------

    def ingest(self, entry: LogEntry) -> None:
        """Directly insert a log entry (used by StructuredLogger integration)."""
        self._store.insert(entry)

    def ingest_many(self, entries: List[LogEntry]) -> None:
        self._store.insert_many(entries)

    # ------------------------------------------------------------------
    # File tail
    # ------------------------------------------------------------------

    def tail_file(self, path: str, service: str) -> None:
        """Start a background thread that tails a JSON-lines log file.

        Each line must be a JSON object matching the LogEntry schema.
        Tailing is resilient to log rotation (file recreated / truncated).
        """
        key = f"{path}:{service}"
        if key in self._tail_threads and self._tail_threads[key].is_alive():
            return  # already tailing this file

        stop_event = threading.Event()
        self._stop_events[key] = stop_event

        def _tail() -> None:
            last_inode = -1
            last_pos = 0
            while not stop_event.is_set():
                try:
                    import os
                    stat = os.stat(path)
                    current_inode = stat.st_ino
                    if current_inode != last_inode:
                        # File was rotated or created
                        last_inode = current_inode
                        last_pos = 0
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        fh.seek(last_pos)
                        while not stop_event.is_set():
                            line = fh.readline()
                            if not line:
                                last_pos = fh.tell()
                                time.sleep(0.2)
                                continue
                            line = line.strip()
                            if line:
                                try:
                                    d = json.loads(line)
                                    d.setdefault("service", service)
                                    entry = LogEntry.from_dict(d)
                                    self._store.insert(entry)
                                except (json.JSONDecodeError, KeyError):
                                    # Plain text fallback
                                    entry = LogEntry(
                                        timestamp_ns=time.time_ns(),
                                        level="INFO",
                                        service=service,
                                        message=line,
                                    )
                                    self._store.insert(entry)
                except FileNotFoundError:
                    time.sleep(1.0)
                except Exception:
                    time.sleep(1.0)

        t = threading.Thread(
            target=_tail,
            name=f"srfm-logtail-{service}",
            daemon=True,
        )
        self._tail_threads[key] = t
        t.start()

    def stop_tail(self, path: str, service: str) -> None:
        """Stop tailing a specific file."""
        key = f"{path}:{service}"
        ev = self._stop_events.get(key)
        if ev:
            ev.set()

    # ------------------------------------------------------------------
    # HTTP push endpoint
    # ------------------------------------------------------------------

    def register_push_endpoint(
        self,
        host: str = "127.0.0.1",
        port: int = 9200,
    ) -> int:
        """Start an HTTP server on host:port that accepts POST /logs.

        Body: JSON array of log entry objects.
        Returns the actual port the server bound to.
        """
        store_ref = self._store

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    data = json.loads(body)
                    entries = [LogEntry.from_dict(d) for d in data]
                    store_ref.insert_many(entries)
                    self.send_response(204)
                    self.end_headers()
                except Exception as exc:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(str(exc).encode())

            def log_message(self, *args):  # noqa: N802
                pass  # suppress default access logging

        server = HTTPServer((host, port), _Handler)
        self._http_server = server
        actual_port = server.server_address[1]

        self._http_thread = threading.Thread(
            target=server.serve_forever,
            name="srfm-log-push",
            daemon=True,
        )
        self._http_thread.start()
        return actual_port

    def stop_push_endpoint(self) -> None:
        """Shutdown the HTTP push server."""
        if self._http_server:
            self._http_server.shutdown()
            self._http_server = None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        level: Optional[str] = None,
        service: Optional[str] = None,
        since: Optional[float] = None,  # epoch seconds float
        text_search: Optional[str] = None,
        n: int = 100,
    ) -> List[LogEntry]:
        """Query the log store with optional filters.

        Parameters
        ----------
        level       # minimum log level (e.g. "WARNING" returns WARNING+ERROR+CRITICAL)
        service     # exact service name filter
        since       # Unix epoch seconds; only return entries after this time
        text_search -- substring search on message field (case-insensitive via LIKE)
        n           # maximum number of entries to return (most recent first)
        """
        since_ns = int(since * 1_000_000_000) if since is not None else None
        return self._store.query(
            level=level,
            service=service,
            since_ns=since_ns,
            text_search=text_search,
            n=n,
        )

    def close(self) -> None:
        self.stop_push_endpoint()
        for ev in self._stop_events.values():
            ev.set()
        self._store.close()


# ---------------------------------------------------------------------------
# LogAnalyzer
# ---------------------------------------------------------------------------

@dataclass
class _PatternAlert:
    pattern: str
    compiled: re.Pattern
    threshold: int    # occurrences within window_s
    window_s: int
    callback: Callable[[str, List[LogEntry]], None]
    service: Optional[str]
    recent: Deque[float] = field(default_factory=deque)  # timestamps of matches


class LogAnalyzer:
    """Higher-level analysis on top of a LogAggregator.

    Usage
    -----
        analyzer = LogAnalyzer(aggregator)
        rate = analyzer.error_rate("order_router", window_minutes=5)
        top = analyzer.top_errors("order_router", n=10)
        analyzer.alert_on_pattern(r"circuit.?breaker", 3, 60, my_callback)
        spike = analyzer.detect_log_anomaly("order_router")
    """

    def __init__(self, aggregator: LogAggregator) -> None:
        self._agg = aggregator
        self._pattern_alerts: List[_PatternAlert] = []
        self._alert_lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Error rate
    # ------------------------------------------------------------------

    def error_rate(
        self,
        service: str,
        window_minutes: int = 5,
    ) -> float:
        """Return errors per minute for the given service in the recent window."""
        since_s = time.time() - window_minutes * 60
        count = self._agg._store.count_since(
            since_ns=int(since_s * 1e9),
            level="ERROR",
            service=service,
        )
        return count / max(1, window_minutes)

    # ------------------------------------------------------------------
    # Top errors
    # ------------------------------------------------------------------

    def top_errors(
        self,
        service: str,
        n: int = 10,
        window_hours: int = 24,
    ) -> List[Tuple[str, int]]:
        """Return the n most frequent error messages in the window.

        Returns list of (message, count) tuples sorted by count descending.
        """
        since_s = time.time() - window_hours * 3600
        entries = self._agg.query(
            level="ERROR",
            service=service,
            since=since_s,
            n=10_000,
        )
        counts: Counter = Counter(e.message for e in entries)
        return counts.most_common(n)

    # ------------------------------------------------------------------
    # Pattern alerts
    # ------------------------------------------------------------------

    def alert_on_pattern(
        self,
        pattern: str,
        threshold: int,
        window_s: int,
        callback: Callable[[str, List[LogEntry]], None],
        service: Optional[str] = None,
    ) -> None:
        """Fire callback when pattern appears >= threshold times within window_s.

        callback receives (pattern, matching_entries).

        The monitor loop must be running (call start_monitor()).
        """
        pa = _PatternAlert(
            pattern=pattern,
            compiled=re.compile(pattern, re.IGNORECASE),
            threshold=threshold,
            window_s=window_s,
            callback=callback,
            service=service,
        )
        with self._alert_lock:
            self._pattern_alerts.append(pa)

    def start_monitor(self, check_interval_s: float = 10.0) -> None:
        """Start the background thread that evaluates pattern alerts."""
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(check_interval_s,),
            name="srfm-log-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def stop_monitor(self, timeout_s: float = 5.0) -> None:
        """Stop the background monitor thread."""
        self._monitor_stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=timeout_s)

    def _monitor_loop(self, interval_s: float) -> None:
        while not self._monitor_stop.wait(interval_s):
            self._evaluate_patterns()

    def _evaluate_patterns(self) -> None:
        now = time.time()
        with self._alert_lock:
            alerts_snap = list(self._pattern_alerts)

        for pa in alerts_snap:
            since_s = now - pa.window_s
            entries = self._agg.query(
                service=pa.service,
                since=since_s,
                n=10_000,
            )
            matches = [e for e in entries if pa.compiled.search(e.message)]
            if len(matches) >= pa.threshold:
                try:
                    pa.callback(pa.pattern, matches)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_log_anomaly(
        self,
        service: str,
        short_window_min: int = 5,
        long_window_min: int = 60,
        spike_factor: float = 3.0,
    ) -> Optional[str]:
        """Detect a sudden spike in error rate.

        Compares the short-window error rate to the long-window baseline.
        Returns a human-readable description if a spike is detected, else None.

        Parameters
        ----------
        service           # service to evaluate
        short_window_min  # recent window for spike detection (minutes)
        long_window_min   # baseline window (minutes)
        spike_factor      # how many times the baseline rate triggers an anomaly
        """
        short_rate = self.error_rate(service, window_minutes=short_window_min)
        long_rate = self.error_rate(service, window_minutes=long_window_min)

        baseline = max(long_rate, 0.01)  # avoid division by zero
        if short_rate >= baseline * spike_factor:
            return (
                f"Anomaly detected for service '{service}': "
                f"{short_rate:.2f} errors/min in last {short_window_min} min "
                f"vs baseline {long_rate:.2f} errors/min "
                f"(spike factor {short_rate / baseline:.1f}x)"
            )
        return None

    # ------------------------------------------------------------------
    # Convenience aggregation queries
    # ------------------------------------------------------------------

    def services(self) -> List[str]:
        """Return distinct service names present in the log store."""
        with self._agg._store._lock:
            rows = self._agg._store._conn.execute(
                "SELECT DISTINCT service FROM logs ORDER BY service"
            ).fetchall()
        return [r[0] for r in rows]

    def log_volume(
        self,
        service: str,
        bucket_minutes: int = 5,
        window_hours: int = 1,
    ) -> List[Tuple[float, int]]:
        """Return (bucket_start_epoch_s, count) tuples for time-bucketed log volume.

        Useful for plotting log rate timelines.
        """
        since_s = time.time() - window_hours * 3600
        bucket_ns = bucket_minutes * 60 * 1_000_000_000
        with self._agg._store._lock:
            rows = self._agg._store._conn.execute(
                "SELECT (timestamp_ns / ?) * ? as bucket, COUNT(*) "
                "FROM logs WHERE service = ? AND timestamp_ns >= ? "
                "GROUP BY bucket ORDER BY bucket",
                (bucket_ns, bucket_ns, service, int(since_s * 1e9)),
            ).fetchall()
        return [(r[0] / 1_000_000_000.0, r[1]) for r in rows]
