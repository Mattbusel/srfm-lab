"""
alerter.py -- Alert manager for the SRFM quantitative trading system.

Monitors system state across all microservices and dispatches alerts to
Slack, the SQLite audit log, and the Elixir EventBus.

Alert rules:
    DrawdownAlert          -- equity drawdown thresholds
    VaRBreachAlert         -- realized loss vs VaR
    CircuitBreakerAlert    -- Alpaca / Binance circuit open
    ServiceDownAlert       -- microservice health check failing
    BHMassExtremeAlert     -- BH mass > 3.5 on any instrument
    HurstFlipAlert         -- Hurst regime flip on >3 instruments simultaneously
    ParameterRollbackAlert -- Elixir coordinator triggered rollback
    CorrelationRegimeAlert -- avg pairwise correlation > 0.85
    LiquidityAlert         -- Amihud illiquidity 3x above 30-day mean

Usage:
    alerter = Alerter()
    alerter.start()         # launches background evaluation loop
    alerter.stop()
"""

from __future__ import annotations

import enum
import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("srfm.alerter")

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
ALERT_DB_PATH        = os.environ.get("SRFM_ALERT_DB",           "data/audit.db")
SLACK_WEBHOOK_URL    = os.environ.get("SRFM_SLACK_WEBHOOK",       "")
RISK_API_URL         = os.environ.get("SRFM_RISK_API_URL",        "http://localhost:8791")
COORD_URL            = os.environ.get("SRFM_COORD_URL",           "http://localhost:8781")
TRADE_DB_PATH        = os.environ.get("SRFM_TRADE_DB",            "data/larsa_trades.db")
EVAL_INTERVAL_S      = int(os.environ.get("SRFM_ALERT_INTERVAL",  "15"))
DEDUP_WINDOW_S       = int(os.environ.get("SRFM_ALERT_DEDUP_S",   "1800"))  # 30 min
HTTP_TIMEOUT_S       = float(os.environ.get("SRFM_HTTP_TIMEOUT",  "5"))

# Alert thresholds -- all env-overridable
DRAWDOWN_WARN_PCT    = float(os.environ.get("SRFM_DD_WARN",       "0.05"))
DRAWDOWN_CRIT_PCT    = float(os.environ.get("SRFM_DD_CRIT",       "0.10"))
BH_MASS_EXTREME      = float(os.environ.get("SRFM_BH_EXTREME",    "3.5"))
HURST_FLIP_COUNT     = int(os.environ.get("SRFM_HURST_FLIP_N",    "3"))
CORRELATION_WARN     = float(os.environ.get("SRFM_CORR_WARN",     "0.85"))
AMIHUD_MULTIPLIER    = float(os.environ.get("SRFM_AMIHUD_MULT",   "3.0"))


# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------

class Severity(enum.Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"

    def __lt__(self, other: "Severity") -> bool:
        order = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}
        return order[self] < order[other]


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    rule_name:   str
    severity:    Severity
    message:     str
    metadata:    Dict[str, Any] = field(default_factory=dict)
    timestamp:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    db_id:       Optional[int] = None

    # Unique key for deduplication
    def dedup_key(self) -> str:
        return f"{self.rule_name}:{self.severity.value}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name":   self.rule_name,
            "severity":    self.severity.value,
            "message":     self.message,
            "metadata":    self.metadata,
            "timestamp":   self.timestamp.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "db_id":       self.db_id,
        }


# ---------------------------------------------------------------------------
# Alert history (SQLite)
# ---------------------------------------------------------------------------

_DDL_ALERTS = """
CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    severity    TEXT    NOT NULL,
    rule_name   TEXT    NOT NULL,
    message     TEXT    NOT NULL,
    metadata    TEXT    NOT NULL DEFAULT '{}',
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts        ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_severity  ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_rule      ON alerts(rule_name);
"""


class AlertHistory:
    """Append-only SQLite store for fired alerts."""

    def __init__(self, db_path: str = ALERT_DB_PATH) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_DDL_ALERTS)
            self._conn.commit()

    def insert(self, alert: Alert) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO alerts (timestamp, severity, rule_name, message, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    alert.timestamp.isoformat(),
                    alert.severity.value,
                    alert.rule_name,
                    alert.message,
                    json.dumps(alert.metadata),
                ),
            )
            self._conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def resolve(self, db_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE alerts SET resolved_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), db_id),
            )
            self._conn.commit()

    def get_active(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM alerts
                WHERE resolved_at IS NULL
                ORDER BY timestamp DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_history(
        self,
        since: Optional[datetime] = None,
        severity: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        conditions = []
        params: List[Any] = []
        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        if severity:
            conditions.append("severity = ?")
            params.append(severity.upper())
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM alerts {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class AlertDeduplication:
    """
    Suppress repeated identical alerts within a rolling time window.

    An alert is considered a duplicate if its dedup_key matches a recently
    fired alert and the window has not expired.
    """

    def __init__(self, window_s: int = DEDUP_WINDOW_S) -> None:
        self._window_s = window_s
        self._seen: Dict[str, datetime] = {}  # key -> last_fired_at
        self._lock = threading.Lock()

    def is_duplicate(self, alert: Alert) -> bool:
        key = alert.dedup_key()
        now = datetime.now(timezone.utc)
        with self._lock:
            last = self._seen.get(key)
            if last is None:
                return False
            return (now - last).total_seconds() < self._window_s

    def record(self, alert: Alert) -> None:
        key = alert.dedup_key()
        with self._lock:
            self._seen[key] = alert.timestamp

    def clear_expired(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._window_s)
        with self._lock:
            self._seen = {
                k: v for k, v in self._seen.items() if v > cutoff
            }

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()


# ---------------------------------------------------------------------------
# Alert dispatcher
# ---------------------------------------------------------------------------

class AlertDispatcher:
    """
    Routes fired alerts to configured sinks:
        1. Slack webhook (if SRFM_SLACK_WEBHOOK is set)
        2. SQLite audit log (always)
        3. Elixir EventBus (HTTP POST to :8781/events)
    """

    SEVERITY_EMOJI = {
        Severity.INFO:     ":information_source:",
        Severity.WARNING:  ":warning:",
        Severity.CRITICAL: ":rotating_light:",
    }
    SEVERITY_COLOR = {
        Severity.INFO:     "#36a64f",
        Severity.WARNING:  "#ffae00",
        Severity.CRITICAL: "#e01e5a",
    }

    def __init__(
        self,
        history: AlertHistory,
        dedup: AlertDeduplication,
        slack_url: str = SLACK_WEBHOOK_URL,
        coord_url: str = COORD_URL,
    ) -> None:
        self._history = history
        self._dedup = dedup
        self._slack_url = slack_url
        self._coord_url = coord_url.rstrip("/")
        self._session: Optional[Any] = None

    def _sess(self):
        if not _REQUESTS_AVAILABLE:
            return None
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers["Content-Type"] = "application/json"
        return self._session

    def dispatch(self, alert: Alert) -> bool:
        """
        Dispatch a single alert. Returns True if dispatched, False if suppressed.
        """
        if self._dedup.is_duplicate(alert):
            log.debug(f"Suppressed duplicate alert: {alert.dedup_key()}")
            return False

        # Persist to SQLite first
        try:
            db_id = self._history.insert(alert)
            alert.db_id = db_id
        except Exception as exc:
            log.error(f"Alert DB insert failed: {exc}")

        # Record in dedup tracker
        self._dedup.record(alert)

        # Dispatch async to avoid blocking evaluation loop
        t = threading.Thread(
            target=self._dispatch_async,
            args=(alert,),
            daemon=True,
        )
        t.start()

        log.info(
            f"[{alert.severity.value}] {alert.rule_name}: {alert.message}"
        )
        return True

    def _dispatch_async(self, alert: Alert) -> None:
        self._send_slack(alert)
        self._publish_event_bus(alert)

    def _send_slack(self, alert: Alert) -> None:
        if not self._slack_url or not _REQUESTS_AVAILABLE:
            return
        sess = self._sess()
        if sess is None:
            return

        emoji = self.SEVERITY_EMOJI.get(alert.severity, "")
        color = self.SEVERITY_COLOR.get(alert.severity, "#cccccc")
        ts_str = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji} [{alert.severity.value}] {alert.rule_name}",
                    "text": alert.message,
                    "footer": f"SRFM Alerter | {ts_str}",
                    "fields": [
                        {"title": k, "value": str(v), "short": True}
                        for k, v in alert.metadata.items()
                    ],
                }
            ]
        }

        try:
            resp = sess.post(
                self._slack_url,
                json=payload,
                timeout=HTTP_TIMEOUT_S,
            )
            resp.raise_for_status()
            log.debug(f"Slack alert sent: {alert.rule_name}")
        except Exception as exc:
            log.warning(f"Slack dispatch failed: {exc}")

    def _publish_event_bus(self, alert: Alert) -> None:
        if not _REQUESTS_AVAILABLE:
            return
        sess = self._sess()
        if sess is None:
            return
        payload = {
            "topic":   "alert_fired",
            "payload": alert.to_dict(),
        }
        try:
            sess.post(
                f"{self._coord_url}/events",
                json=payload,
                timeout=HTTP_TIMEOUT_S,
            )
        except Exception as exc:
            log.debug(f"EventBus publish failed (non-critical): {exc}")


# ---------------------------------------------------------------------------
# Base alert rule
# ---------------------------------------------------------------------------

class AlertRule(ABC):
    """
    Base class for all alert rules.

    Subclasses implement evaluate() which receives the current system state
    dict and returns a list of Alerts (empty if no alerts).
    """

    name: str = "base_rule"

    def __init__(self) -> None:
        self._active_alerts: Dict[str, Alert] = {}

    @abstractmethod
    def evaluate(self, state: Dict[str, Any]) -> List[Alert]:
        """Evaluate state, return list of new alerts (may be empty)."""
        ...

    def _make_alert(
        self,
        severity: Severity,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        return Alert(
            rule_name=self.name,
            severity=severity,
            message=message,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# Concrete alert rules
# ---------------------------------------------------------------------------

class DrawdownAlert(AlertRule):
    """
    WARNING  when equity drawdown > DRAWDOWN_WARN_PCT (default 5%)
    CRITICAL when equity drawdown > DRAWDOWN_CRIT_PCT (default 10%)

    State keys:
        drawdown: float   -- current drawdown [0, 1]
        equity:   float   -- current NAV in USD
    """

    name = "DrawdownAlert"

    def __init__(
        self,
        warn_pct: float = DRAWDOWN_WARN_PCT,
        crit_pct: float = DRAWDOWN_CRIT_PCT,
    ) -> None:
        super().__init__()
        self._warn = warn_pct
        self._crit = crit_pct

    def evaluate(self, state: Dict[str, Any]) -> List[Alert]:
        dd = state.get("drawdown")
        if dd is None:
            return []

        dd = float(dd)
        equity = state.get("equity", 0.0)
        meta = {"drawdown_pct": f"{dd:.2%}", "equity_usd": f"{equity:,.0f}"}

        if dd >= self._crit:
            return [self._make_alert(
                Severity.CRITICAL,
                f"Portfolio drawdown {dd:.2%} exceeds critical threshold {self._crit:.0%}",
                meta,
            )]
        if dd >= self._warn:
            return [self._make_alert(
                Severity.WARNING,
                f"Portfolio drawdown {dd:.2%} exceeds warning threshold {self._warn:.0%}",
                meta,
            )]
        return []


class VaRBreachAlert(AlertRule):
    """
    CRITICAL when realized daily loss exceeds the 95% VaR estimate.

    State keys:
        daily_pnl: float   -- today's P&L in USD (negative = loss)
        var_95:    float   -- 95% VaR threshold in USD (positive number)
    """

    name = "VaRBreachAlert"

    def evaluate(self, state: Dict[str, Any]) -> List[Alert]:
        daily_pnl = state.get("daily_pnl")
        var_95 = state.get("var_95")
        if daily_pnl is None or var_95 is None:
            return []

        loss = -float(daily_pnl)
        var = abs(float(var_95))
        if loss > var and var > 0:
            return [self._make_alert(
                Severity.CRITICAL,
                f"Realized daily loss ${loss:,.0f} exceeds 95% VaR ${var:,.0f}",
                {"daily_pnl": f"{daily_pnl:,.2f}", "var_95": f"{var:,.2f}"},
            )]
        return []


class CircuitBreakerAlert(AlertRule):
    """
    WARNING when any venue circuit breaker is open.

    State keys:
        circuit_breakers: {venue: bool}   -- True = open (halted)
    """

    name = "CircuitBreakerAlert"

    def evaluate(self, state: Dict[str, Any]) -> List[Alert]:
        alerts = []
        cb = state.get("circuit_breakers", {})
        for venue, is_open in cb.items():
            if is_open:
                alerts.append(self._make_alert(
                    Severity.WARNING,
                    f"Circuit breaker OPEN for venue: {venue}",
                    {"venue": venue},
                ))
        return alerts


class ServiceDownAlert(AlertRule):
    """
    CRITICAL when any microservice health check is failing.

    State keys:
        service_health: {service_name: bool}   -- True = healthy
    """

    name = "ServiceDownAlert"

    def evaluate(self, state: Dict[str, Any]) -> List[Alert]:
        alerts = []
        health = state.get("service_health", {})
        for service, is_up in health.items():
            if not is_up:
                alerts.append(self._make_alert(
                    Severity.CRITICAL,
                    f"Service {service!r} is DOWN -- health check failing",
                    {"service": service},
                ))
        return alerts


class BHMassExtremeAlert(AlertRule):
    """
    WARNING when BH mass > BH_MASS_EXTREME (default 3.5) on any instrument.

    State keys:
        bh_mass: {symbol: {timeframe: float}}
    """

    name = "BHMassExtremeAlert"

    def __init__(self, threshold: float = BH_MASS_EXTREME) -> None:
        super().__init__()
        self._threshold = threshold

    def evaluate(self, state: Dict[str, Any]) -> List[Alert]:
        alerts = []
        bh_mass = state.get("bh_mass", {})
        for sym, tf_map in bh_mass.items():
            if not isinstance(tf_map, dict):
                continue
            for tf, mass in tf_map.items():
                if mass is not None and float(mass) > self._threshold:
                    alerts.append(self._make_alert(
                        Severity.WARNING,
                        (
                            f"BH mass {float(mass):.2f} on {sym}/{tf} "
                            f"exceeds extreme threshold {self._threshold}"
                        ),
                        {"symbol": sym, "timeframe": tf, "bh_mass": float(mass)},
                    ))
        return alerts


class HurstFlipAlert(AlertRule):
    """
    INFO when Hurst regime flip detected on more than HURST_FLIP_COUNT instruments
    simultaneously.

    State keys:
        hurst_current:  {symbol: float}   -- current Hurst exponents
        hurst_previous: {symbol: float}   -- previous Hurst exponents
    """

    name = "HurstFlipAlert"

    TRENDING_THRESHOLD  = 0.55
    MEANREV_THRESHOLD   = 0.45

    def __init__(self, flip_count: int = HURST_FLIP_COUNT) -> None:
        super().__init__()
        self._flip_count = flip_count

    def _classify(self, h: float) -> str:
        if h > self.TRENDING_THRESHOLD:
            return "trending"
        if h < self.MEANREV_THRESHOLD:
            return "mean_reverting"
        return "random"

    def evaluate(self, state: Dict[str, Any]) -> List[Alert]:
        current  = state.get("hurst_current", {})
        previous = state.get("hurst_previous", {})
        if not current or not previous:
            return []

        flipped = []
        for sym, h_now in current.items():
            h_prev = previous.get(sym)
            if h_prev is None:
                continue
            reg_now  = self._classify(float(h_now))
            reg_prev = self._classify(float(h_prev))
            if reg_now != reg_prev:
                flipped.append({
                    "symbol":   sym,
                    "from":     reg_prev,
                    "to":       reg_now,
                    "h_prev":   round(float(h_prev), 3),
                    "h_now":    round(float(h_now), 3),
                })

        if len(flipped) > self._flip_count:
            syms = ", ".join(f["symbol"] for f in flipped)
            return [self._make_alert(
                Severity.INFO,
                (
                    f"Hurst regime flip on {len(flipped)} instruments "
                    f"simultaneously: {syms}"
                ),
                {"flipped_instruments": flipped, "count": len(flipped)},
            )]
        return []


class ParameterRollbackAlert(AlertRule):
    """
    CRITICAL when the Elixir coordinator has triggered a parameter rollback.

    State keys:
        param_rollback_triggered: bool
        rollback_reason:          str  (optional)
        rollback_params:          dict (optional)
    """

    name = "ParameterRollbackAlert"

    def evaluate(self, state: Dict[str, Any]) -> List[Alert]:
        if not state.get("param_rollback_triggered", False):
            return []
        reason = state.get("rollback_reason", "unknown")
        params = state.get("rollback_params", {})
        return [self._make_alert(
            Severity.CRITICAL,
            f"Elixir coordinator triggered parameter rollback -- reason: {reason}",
            {"reason": reason, "rollback_params": params},
        )]


class CorrelationRegimeAlert(AlertRule):
    """
    WARNING when average pairwise correlation exceeds CORRELATION_WARN (default 0.85).
    Indicates a crowded trade regime where diversification benefits collapse.

    State keys:
        avg_pairwise_correlation: float
    """

    name = "CorrelationRegimeAlert"

    def __init__(self, threshold: float = CORRELATION_WARN) -> None:
        super().__init__()
        self._threshold = threshold

    def evaluate(self, state: Dict[str, Any]) -> List[Alert]:
        corr = state.get("avg_pairwise_correlation")
        if corr is None:
            return []
        corr = float(corr)
        if corr > self._threshold:
            return [self._make_alert(
                Severity.WARNING,
                (
                    f"Average pairwise correlation {corr:.3f} exceeds {self._threshold} "
                    f"-- possible crowded trade regime"
                ),
                {"avg_correlation": corr, "threshold": self._threshold},
            )]
        return []


class LiquidityAlert(AlertRule):
    """
    WARNING when Amihud illiquidity ratio is AMIHUD_MULTIPLIER x (default 3x)
    above the 30-day mean for any instrument.

    State keys:
        amihud_current: {symbol: float}
        amihud_30d_mean: {symbol: float}
    """

    name = "LiquidityAlert"

    def __init__(self, multiplier: float = AMIHUD_MULTIPLIER) -> None:
        super().__init__()
        self._mult = multiplier

    def evaluate(self, state: Dict[str, Any]) -> List[Alert]:
        alerts = []
        current = state.get("amihud_current", {})
        mean_30d = state.get("amihud_30d_mean", {})

        for sym, illiq in current.items():
            baseline = mean_30d.get(sym)
            if baseline is None or float(baseline) <= 0:
                continue
            ratio = float(illiq) / float(baseline)
            if ratio >= self._mult:
                alerts.append(self._make_alert(
                    Severity.WARNING,
                    (
                        f"Amihud illiquidity on {sym} is {ratio:.1f}x the 30-day mean "
                        f"(current={illiq:.4f}, mean={baseline:.4f})"
                    ),
                    {
                        "symbol":        sym,
                        "illiq_current": float(illiq),
                        "illiq_30d_mean": float(baseline),
                        "ratio":         round(ratio, 2),
                    },
                ))
        return alerts


# ---------------------------------------------------------------------------
# Alerter -- main orchestrator
# ---------------------------------------------------------------------------

# All built-in rules registered by default
_DEFAULT_RULES = [
    DrawdownAlert,
    VaRBreachAlert,
    CircuitBreakerAlert,
    ServiceDownAlert,
    BHMassExtremeAlert,
    HurstFlipAlert,
    ParameterRollbackAlert,
    CorrelationRegimeAlert,
    LiquidityAlert,
]


class Alerter:
    """
    Evaluates all alert rules against current system state on a background
    thread and dispatches fired alerts.

    Usage::

        alerter = Alerter()
        alerter.start()
        # Feed state from your monitoring loop:
        alerter.update_state(new_state_dict)
        # ...
        alerter.stop()
    """

    def __init__(
        self,
        db_path: str = ALERT_DB_PATH,
        slack_url: str = SLACK_WEBHOOK_URL,
        coord_url: str = COORD_URL,
        eval_interval: int = EVAL_INTERVAL_S,
        extra_rules: Optional[List[AlertRule]] = None,
    ) -> None:
        self._interval = eval_interval
        self._state: Dict[str, Any] = {}
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._eval_thread: Optional[threading.Thread] = None

        self._history = AlertHistory(db_path)
        self._dedup = AlertDeduplication()
        self._dispatcher = AlertDispatcher(
            self._history, self._dedup, slack_url=slack_url, coord_url=coord_url
        )

        self._rules: List[AlertRule] = [cls() for cls in _DEFAULT_RULES]
        if extra_rules:
            self._rules.extend(extra_rules)

        # Track which alerts are currently firing (for resolution)
        self._firing: Dict[str, Alert] = {}  # dedup_key -> alert
        self._firing_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stop_event.clear()
        self._eval_thread = threading.Thread(
            target=self._eval_loop,
            daemon=True,
            name="alerter_eval",
        )
        self._eval_thread.start()
        log.info(f"Alerter started -- eval_interval={self._interval}s")

    def stop(self) -> None:
        self._stop_event.set()
        self._history.close()
        log.info("Alerter stopped")

    def update_state(self, state: Dict[str, Any]) -> None:
        """Thread-safe state update. Call from monitoring loop."""
        with self._state_lock:
            self._state.update(state)

    def replace_state(self, state: Dict[str, Any]) -> None:
        """Replace entire state dict atomically."""
        with self._state_lock:
            self._state = dict(state)

    def evaluate_now(self) -> List[Alert]:
        """Run one evaluation cycle synchronously. Returns all fired alerts."""
        with self._state_lock:
            state = dict(self._state)
        return self._run_evaluation(state)

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        return self._history.get_active()

    def get_alert_history(
        self,
        since: Optional[datetime] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self._history.get_history(since=since, severity=severity)

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _eval_loop(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                with self._state_lock:
                    state = dict(self._state)
                self._run_evaluation(state)
                self._dedup.clear_expired()
            except Exception as exc:
                log.error(f"Alerter eval loop error: {exc}", exc_info=True)

            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, self._interval - elapsed)
            self._stop_event.wait(timeout=sleep_for)

    def _run_evaluation(self, state: Dict[str, Any]) -> List[Alert]:
        fired: List[Alert] = []
        for rule in self._rules:
            try:
                new_alerts = rule.evaluate(state)
                for alert in new_alerts:
                    dispatched = self._dispatcher.dispatch(alert)
                    if dispatched:
                        fired.append(alert)
                        with self._firing_lock:
                            self._firing[alert.dedup_key()] = alert
            except Exception as exc:
                log.error(f"Rule {rule.name} evaluation error: {exc}", exc_info=True)

        # Auto-resolve alerts whose conditions are no longer met
        self._auto_resolve(state)
        return fired

    def _auto_resolve(self, state: Dict[str, Any]) -> None:
        """
        For each firing alert, re-evaluate whether its condition still holds.
        If not, mark it resolved in the DB.
        """
        with self._firing_lock:
            keys = list(self._firing.keys())

        for key in keys:
            with self._firing_lock:
                alert = self._firing.get(key)
            if alert is None:
                continue

            # Find the rule that owns this alert
            rule = next(
                (r for r in self._rules if r.name == alert.rule_name), None
            )
            if rule is None:
                continue

            try:
                still_firing = rule.evaluate(state)
                still_keys = {a.dedup_key() for a in still_firing}
                if key not in still_keys:
                    # Condition cleared -- resolve in DB
                    if alert.db_id is not None:
                        self._history.resolve(alert.db_id)
                    with self._firing_lock:
                        self._firing.pop(key, None)
                    log.info(f"Alert resolved: {key}")
            except Exception as exc:
                log.debug(f"Auto-resolve check failed for {key}: {exc}")


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    )

    alerter = Alerter(db_path=":memory:")
    alerter.start()

    # Inject a test state with a critical drawdown
    test_state = {
        "equity":       90_000.0,
        "drawdown":     0.12,          # > 10% -- should fire CRITICAL
        "var_95":       5_000.0,
        "daily_pnl":    -8_000.0,      # exceeds VaR -- should fire CRITICAL
        "circuit_breakers": {"alpaca": True, "binance": False},
        "service_health": {"risk_api": True, "live_trader": False},
        "bh_mass":      {"BTC": {"daily": 4.1, "hourly": 1.2}},
        "avg_pairwise_correlation": 0.91,
    }

    alerter.update_state(test_state)
    fired = alerter.evaluate_now()
    print(f"\nFired {len(fired)} alerts:")
    for a in fired:
        print(f"  [{a.severity.value}] {a.rule_name}: {a.message}")

    time.sleep(1)
    alerter.stop()
