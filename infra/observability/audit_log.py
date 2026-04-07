"""
audit_log.py -- Append-only audit log for all SRFM system actions.

Writes to SQLite (audit.db). Subscribes to the Elixir EventBus via HTTP
polling to capture param updates, rollbacks, and circuit breaker events.

Tables:
    trades            -- every trade entry/exit
    parameter_updates -- param changes with old/new values
    regime_changes    -- Hurst/BH regime transitions per symbol
    alerts            -- (shared with alerter.py, same schema)
    rollbacks         -- parameter rollback events

Usage:
    logger = AuditLogger()
    logger.log_trade("BTC", "buy", 45000.0, 0.1, "bh_signal")
    logger.log_param_update(old, new, source="elixir_coordinator")
    logger.start_event_polling()   -- background thread
    logger.stop()
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("srfm.audit_log")

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AUDIT_DB_PATH  = os.environ.get("SRFM_AUDIT_DB",          "data/audit.db")
COORD_URL      = os.environ.get("SRFM_COORD_URL",          "http://localhost:8781")
POLL_INTERVAL  = int(os.environ.get("SRFM_AUDIT_POLL_S",  "15"))
HTTP_TIMEOUT   = float(os.environ.get("SRFM_HTTP_TIMEOUT", "5"))

# EventBus topics we subscribe to
EVENTBUS_TOPICS = ["params_updated", "rollback_triggered", "circuit_open"]

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    action      TEXT    NOT NULL,          -- entry | exit | partial
    price       REAL    NOT NULL,
    size        REAL    NOT NULL,
    reason      TEXT,
    pnl         REAL,
    equity_after REAL
);
CREATE INDEX IF NOT EXISTS idx_aud_trades_ts     ON trades(ts);
CREATE INDEX IF NOT EXISTS idx_aud_trades_symbol ON trades(symbol);

CREATE TABLE IF NOT EXISTS parameter_updates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    old_params  TEXT    NOT NULL,          -- JSON
    new_params  TEXT    NOT NULL,          -- JSON
    change_diff TEXT    NOT NULL DEFAULT '{}'  -- JSON diff
);
CREATE INDEX IF NOT EXISTS idx_params_ts ON parameter_updates(ts);

CREATE TABLE IF NOT EXISTS regime_changes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    from_regime TEXT    NOT NULL,
    to_regime   TEXT    NOT NULL,
    trigger     TEXT                       -- e.g. "bh_mass", "hurst", "manual"
);
CREATE INDEX IF NOT EXISTS idx_regime_ts     ON regime_changes(ts);
CREATE INDEX IF NOT EXISTS idx_regime_symbol ON regime_changes(symbol);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    severity    TEXT    NOT NULL,
    rule_name   TEXT    NOT NULL,
    message     TEXT    NOT NULL,
    metadata    TEXT    NOT NULL DEFAULT '{}',
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_aud_alerts_ts ON alerts(timestamp);

CREATE TABLE IF NOT EXISTS rollbacks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    trigger_reason TEXT   NOT NULL,
    old_params    TEXT    NOT NULL,        -- JSON
    new_params    TEXT    NOT NULL,        -- JSON
    triggered_by  TEXT    NOT NULL DEFAULT 'elixir_coordinator'
);
CREATE INDEX IF NOT EXISTS idx_rollbacks_ts ON rollbacks(ts);

CREATE TABLE IF NOT EXISTS event_bus_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    topic       TEXT    NOT NULL,
    payload     TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_ts    ON event_bus_events(ts);
CREATE INDEX IF NOT EXISTS idx_events_topic ON event_bus_events(topic);
"""


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Thread-safe, append-only audit logger backed by SQLite WAL mode.

    All writes are synchronous from the caller's perspective. The EventBus
    polling runs on a separate daemon thread.
    """

    def __init__(
        self,
        db_path: str = AUDIT_DB_PATH,
        coord_url: str = COORD_URL,
        poll_interval: int = POLL_INTERVAL,
    ) -> None:
        self._db_path = Path(db_path)
        self._coord_url = coord_url.rstrip("/")
        self._poll_interval = poll_interval

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._open_db()
        self._lock = threading.Lock()
        self._init_schema()

        # EventBus polling state
        self._stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        # Track last seen event id per topic to avoid reprocessing
        self._last_event_ids: Dict[str, int] = {t: 0 for t in EVENTBUS_TOPICS}
        self._session: Optional[Any] = None

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_DDL)
            self._conn.commit()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _fetchall(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def log_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        size: float,
        reason: Optional[str] = None,
        pnl: Optional[float] = None,
        equity_after: Optional[float] = None,
    ) -> int:
        """
        Record a trade entry, exit, or partial fill.

        Parameters
        ----------
        symbol      : instrument (e.g. "BTC")
        action      : "entry" | "exit" | "partial"
        price       : fill price
        size        : fill size (positive = long, negative = short)
        reason      : signal name or rationale
        pnl         : realized P&L for exits
        equity_after: NAV after the trade
        """
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._execute(
            """
            INSERT INTO trades (ts, symbol, action, price, size, reason, pnl, equity_after)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, symbol, action, float(price), float(size), reason,
             float(pnl) if pnl is not None else None,
             float(equity_after) if equity_after is not None else None),
        )
        log.debug(f"Audit: trade {action} {size} {symbol} @ {price:.4f}")
        return cur.lastrowid  # type: ignore[return-value]

    def log_param_update(
        self,
        old_params: Dict[str, Any],
        new_params: Dict[str, Any],
        source: str = "unknown",
    ) -> int:
        """
        Record a parameter update event.

        Parameters
        ----------
        old_params : parameter dict before change
        new_params : parameter dict after change
        source     : originator (e.g. "elixir_coordinator", "manual", "optimizer")
        """
        ts = datetime.now(timezone.utc).isoformat()
        diff = _compute_diff(old_params, new_params)
        cur = self._execute(
            """
            INSERT INTO parameter_updates (ts, source, old_params, new_params, change_diff)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts, source,
             json.dumps(old_params, default=str),
             json.dumps(new_params, default=str),
             json.dumps(diff, default=str)),
        )
        changed = list(diff.keys())
        log.info(f"Audit: param update from {source} -- changed: {changed}")
        return cur.lastrowid  # type: ignore[return-value]

    def log_regime_change(
        self,
        symbol: str,
        old_regime: str,
        new_regime: str,
        trigger: Optional[str] = None,
    ) -> int:
        """
        Record a regime transition for a symbol.

        Parameters
        ----------
        symbol      : instrument (e.g. "BTC")
        old_regime  : previous regime label
        new_regime  : new regime label
        trigger     : what caused the change ("bh_mass", "hurst", etc.)
        """
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._execute(
            """
            INSERT INTO regime_changes (ts, symbol, from_regime, to_regime, trigger)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts, symbol, old_regime, new_regime, trigger),
        )
        log.info(f"Audit: regime change {symbol}: {old_regime} -> {new_regime} (trigger={trigger})")
        return cur.lastrowid  # type: ignore[return-value]

    def log_rollback(
        self,
        trigger_reason: str,
        old_params: Dict[str, Any],
        new_params: Dict[str, Any],
        triggered_by: str = "elixir_coordinator",
    ) -> int:
        """Record a parameter rollback event."""
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._execute(
            """
            INSERT INTO rollbacks (ts, trigger_reason, old_params, new_params, triggered_by)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts, trigger_reason,
             json.dumps(old_params, default=str),
             json.dumps(new_params, default=str),
             triggered_by),
        )
        log.warning(f"Audit: rollback triggered by {triggered_by} -- reason: {trigger_reason}")
        return cur.lastrowid  # type: ignore[return-value]

    def log_alert(
        self,
        severity: str,
        rule_name: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record an alert in the audit log."""
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._execute(
            """
            INSERT INTO alerts (timestamp, severity, rule_name, message, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts, severity.upper(), rule_name, message,
             json.dumps(metadata or {}, default=str)),
        )
        return cur.lastrowid  # type: ignore[return-value]

    def log_event_bus(self, topic: str, payload: Dict[str, Any]) -> int:
        """Record a raw EventBus event."""
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._execute(
            "INSERT INTO event_bus_events (ts, topic, payload) VALUES (?, ?, ?)",
            (ts, topic, json.dumps(payload, default=str)),
        )
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_trades(
        self,
        since: Optional[datetime] = None,
        symbol: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve trades from the audit log.

        Parameters
        ----------
        since  : only return trades after this UTC datetime
        symbol : filter by symbol
        limit  : max rows
        """
        conditions = []
        params: list = []

        if since:
            conditions.append("ts >= ?")
            params.append(since.isoformat())
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        rows = self._fetchall(
            f"SELECT * FROM trades {where} ORDER BY ts DESC LIMIT ?",
            tuple(params),
        )
        return [dict(r) for r in rows]

    def get_param_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the n most recent parameter update records."""
        rows = self._fetchall(
            "SELECT * FROM parameter_updates ORDER BY ts DESC LIMIT ?", (n,)
        )
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["old_params"]  = json.loads(d["old_params"])
                d["new_params"]  = json.loads(d["new_params"])
                d["change_diff"] = json.loads(d["change_diff"])
            except (json.JSONDecodeError, KeyError):
                pass
            result.append(d)
        return result

    def get_alerts(
        self,
        severity: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Retrieve alerts, optionally filtered by severity and time."""
        conditions = []
        params: list = []

        if severity:
            conditions.append("severity = ?")
            params.append(severity.upper())
        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        rows = self._fetchall(
            f"SELECT * FROM alerts {where} ORDER BY timestamp DESC LIMIT ?",
            tuple(params),
        )
        return [dict(r) for r in rows]

    def get_regime_history(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return regime change history."""
        conditions = []
        params: list = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if since:
            conditions.append("ts >= ?")
            params.append(since.isoformat())

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        rows = self._fetchall(
            f"SELECT * FROM regime_changes {where} ORDER BY ts DESC LIMIT ?",
            tuple(params),
        )
        return [dict(r) for r in rows]

    def get_rollback_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return rollback event history."""
        rows = self._fetchall(
            "SELECT * FROM rollbacks ORDER BY ts DESC LIMIT ?", (limit,)
        )
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["old_params"] = json.loads(d["old_params"])
                d["new_params"] = json.loads(d["new_params"])
            except (json.JSONDecodeError, KeyError):
                pass
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # EventBus polling
    # ------------------------------------------------------------------

    def start_event_polling(self) -> None:
        """Start background thread that polls Elixir EventBus for new events."""
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="audit_event_poller",
        )
        self._poll_thread.start()
        log.info(f"EventBus poller started -- interval={self._poll_interval}s")

    def stop(self) -> None:
        """Stop background threads and close DB connection."""
        self._stop_event.set()
        with self._lock:
            self._conn.close()
        log.info("AuditLogger stopped")

    def _get_session(self):
        if not _REQUESTS_AVAILABLE:
            return None
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_eventbus_once()
            except Exception as exc:
                log.debug(f"EventBus poll error: {exc}")
            self._stop_event.wait(timeout=self._poll_interval)

    def _poll_eventbus_once(self) -> None:
        sess = self._get_session()
        if sess is None:
            return

        for topic in EVENTBUS_TOPICS:
            last_id = self._last_event_ids.get(topic, 0)
            try:
                resp = sess.get(
                    f"{self._coord_url}/events",
                    params={"topic": topic, "after_id": last_id, "limit": 50},
                    timeout=HTTP_TIMEOUT,
                )
                if resp.status_code != 200:
                    continue
                events = resp.json().get("events", [])
                for event in events:
                    event_id = event.get("id", 0)
                    payload  = event.get("payload", {})
                    self.log_event_bus(topic, payload)
                    self._dispatch_event(topic, payload)
                    if event_id > last_id:
                        last_id = event_id
                self._last_event_ids[topic] = last_id
            except Exception as exc:
                log.debug(f"Poll topic {topic!r}: {exc}")

    def _dispatch_event(self, topic: str, payload: Dict[str, Any]) -> None:
        """Process an EventBus event and write to the appropriate audit table."""
        if topic == "params_updated":
            old = payload.get("old_params", {})
            new = payload.get("new_params", {})
            source = payload.get("source", "elixir_coordinator")
            if old or new:
                self.log_param_update(old, new, source=source)

        elif topic == "rollback_triggered":
            reason   = payload.get("reason", "unknown")
            old      = payload.get("old_params", {})
            new      = payload.get("new_params", {})
            by_whom  = payload.get("triggered_by", "elixir_coordinator")
            self.log_rollback(reason, old, new, triggered_by=by_whom)

        elif topic == "circuit_open":
            venue   = payload.get("venue", "unknown")
            reason  = payload.get("reason", "")
            self.log_alert(
                "WARNING",
                "CircuitBreakerEvent",
                f"Circuit breaker OPEN for {venue}: {reason}",
                payload,
            )

        else:
            log.debug(f"Unhandled EventBus topic: {topic!r}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _compute_diff(
    old: Dict[str, Any],
    new: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute a simple diff between two param dicts.
    Returns {key: {"old": v, "new": v}} for every changed key.
    """
    diff: Dict[str, Any] = {}
    all_keys = set(old) | set(new)
    for k in all_keys:
        o = old.get(k)
        n = new.get(k)
        if o != n:
            diff[k] = {"old": o, "new": n}
    return diff


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_logger: Optional[AuditLogger] = None


def get_audit_logger(db_path: str = AUDIT_DB_PATH) -> AuditLogger:
    """Return the module-level AuditLogger singleton."""
    global _default_logger
    if _default_logger is None:
        _default_logger = AuditLogger(db_path=db_path)
    return _default_logger


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    )

    logger = AuditLogger(db_path=":memory:")

    # Smoke-test all write methods
    logger.log_trade("BTC", "entry", 45_000.0, 0.1, reason="bh_signal")
    logger.log_trade("ETH", "exit", 3_200.0, 1.5, reason="tp_hit", pnl=240.0)
    logger.log_param_update(
        {"stale_threshold": 0.001, "max_frac": 0.8},
        {"stale_threshold": 0.0012, "max_frac": 0.75},
        source="optimizer",
    )
    logger.log_regime_change("BTC", "mean_reverting", "trending", trigger="hurst")
    logger.log_rollback(
        "sharpe_below_0.5",
        {"stale_threshold": 0.0012},
        {"stale_threshold": 0.001},
    )
    logger.log_alert("CRITICAL", "DrawdownAlert", "Drawdown 12% exceeds threshold")

    print("\n-- Trades --")
    for t in logger.get_trades():
        print(f"  {t['ts']} {t['action']} {t['symbol']} {t['size']} @ {t['price']}")

    print("\n-- Param history --")
    for p in logger.get_param_history():
        print(f"  {p['ts']} source={p['source']} diff={p['change_diff']}")

    print("\n-- Rollbacks --")
    for r in logger.get_rollback_history():
        print(f"  {r['ts']} reason={r['trigger_reason']}")

    print("\n-- Alerts --")
    for a in logger.get_alerts():
        print(f"  [{a['severity']}] {a['rule_name']}: {a['message']}")

    logger.stop()
