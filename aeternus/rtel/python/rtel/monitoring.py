"""
AETERNUS Real-Time Execution Layer (RTEL)
monitoring.py — System health monitoring, alerting, and diagnostics

Provides:
- Component health registry with heartbeat tracking
- Prometheus-compatible metrics aggregation
- Alert rules and alert manager
- Live dashboard (ASCII terminal output)
- Log aggregation and structured logging
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Health check registry
# ---------------------------------------------------------------------------

class HealthStatus:
    OK      = "ok"
    WARNING = "warning"
    CRITICAL= "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    name:           str
    status:         str     = HealthStatus.UNKNOWN
    last_heartbeat: float   = 0.0
    message:        str     = ""
    n_checks:       int     = 0
    n_failures:     int     = 0

    @property
    def uptime_pct(self) -> float:
        if self.n_checks == 0:
            return 100.0
        return (self.n_checks - self.n_failures) / self.n_checks * 100.0

    @property
    def staleness_s(self) -> float:
        return time.time() - self.last_heartbeat


CheckFn = Callable[[], Tuple[str, str]]  # returns (status, message)


class HealthRegistry:
    """Registers and polls component health checks."""

    def __init__(self, stale_threshold_s: float = 30.0):
        self._components:  Dict[str, ComponentHealth] = {}
        self._checks:      Dict[str, CheckFn] = {}
        self._stale_threshold = stale_threshold_s
        self._lock = threading.Lock()

    def register(self, name: str, check_fn: CheckFn) -> None:
        with self._lock:
            self._components[name] = ComponentHealth(name=name)
            self._checks[name]     = check_fn

    def heartbeat(self, name: str, status: str = HealthStatus.OK,
                  message: str = "") -> None:
        with self._lock:
            if name not in self._components:
                self._components[name] = ComponentHealth(name=name)
            comp = self._components[name]
            comp.last_heartbeat = time.time()
            comp.status         = status
            comp.message        = message
            comp.n_checks      += 1
            if status != HealthStatus.OK:
                comp.n_failures += 1

    def check_all(self) -> Dict[str, ComponentHealth]:
        results = {}
        with self._lock:
            names = list(self._checks.keys())
        for name in names:
            try:
                status, msg = self._checks[name]()
                self.heartbeat(name, status, msg)
            except Exception as e:
                self.heartbeat(name, HealthStatus.CRITICAL, str(e))
        with self._lock:
            # Also mark stale components
            now = time.time()
            for name, comp in self._components.items():
                if comp.last_heartbeat > 0 and (now - comp.last_heartbeat) > self._stale_threshold:
                    comp.status = HealthStatus.CRITICAL
            results = dict(self._components)
        return results

    def overall_status(self) -> str:
        with self._lock:
            statuses = [c.status for c in self._components.values()]
        if any(s == HealthStatus.CRITICAL for s in statuses):
            return HealthStatus.CRITICAL
        if any(s == HealthStatus.WARNING for s in statuses):
            return HealthStatus.WARNING
        if all(s == HealthStatus.OK for s in statuses):
            return HealthStatus.OK
        return HealthStatus.UNKNOWN

    def to_dict(self) -> dict:
        with self._lock:
            return {
                name: {
                    "status":     comp.status,
                    "staleness":  comp.staleness_s,
                    "uptime_pct": comp.uptime_pct,
                    "message":    comp.message,
                }
                for name, comp in self._components.items()
            }


# ---------------------------------------------------------------------------
# Metrics aggregator
# ---------------------------------------------------------------------------

@dataclass
class MetricValue:
    name:       str
    value:      float
    labels:     Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"   # gauge | counter | histogram
    help_text:  str = ""


class MetricsAggregator:
    """Aggregates metrics from multiple sources and exports Prometheus format."""

    def __init__(self):
        self._gauges:   Dict[str, float]           = {}
        self._counters: Dict[str, float]           = {}
        self._histograms: Dict[str, List[float]]   = defaultdict(list)
        self._labels:   Dict[str, Dict[str, str]]  = {}
        self._lock      = threading.Lock()

    def set_gauge(self, name: str, value: float,
                  labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key]  = value
            self._labels[key]  = labels or {}

    def inc_counter(self, name: str, amount: float = 1.0,
                    labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key]  = self._counters.get(key, 0.0) + amount
            self._labels[key]    = labels or {}

    def observe_histogram(self, name: str, value: float,
                          labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            key = self._make_key(name, labels)
            hist = self._histograms[key]
            hist.append(value)
            if len(hist) > 10000:
                hist.pop(0)
            self._labels[key] = labels or {}

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def export_prometheus(self) -> str:
        out = []
        with self._lock:
            for key, value in self._gauges.items():
                out.append(f"{key} {value:.6g}")
            for key, value in self._counters.items():
                out.append(f"{key} {value:.6g}")
            for key, values in self._histograms.items():
                if values:
                    arr = sorted(values)
                    n   = len(arr)
                    def pct(p: float) -> float:
                        idx = int(p / 100.0 * n)
                        return arr[min(idx, n-1)]
                    base = key.replace("{", "_").replace("}", "").replace('"', "").replace(",", "_").replace("=", "_")
                    out.append(f"{base}_count {n}")
                    out.append(f"{base}_sum {sum(arr):.6g}")
                    out.append(f"{base}_p50 {pct(50):.6g}")
                    out.append(f"{base}_p95 {pct(95):.6g}")
                    out.append(f"{base}_p99 {pct(99):.6g}")
        return "\n".join(out) + "\n"

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "gauges":   dict(self._gauges),
                "counters": dict(self._counters),
                "histogram_keys": list(self._histograms.keys()),
            }


# ---------------------------------------------------------------------------
# Alert rule and alert manager
# ---------------------------------------------------------------------------

@dataclass
class AlertRule:
    name:       str
    condition:  Callable[[MetricsAggregator], bool]
    severity:   str     = "warning"    # warning | critical
    message:    str     = ""
    cooldown_s: float   = 60.0         # minimum seconds between alerts


@dataclass
class Alert:
    rule_name:  str
    severity:   str
    message:    str
    timestamp:  float   = field(default_factory=time.time)
    resolved:   bool    = False
    resolved_at: Optional[float] = None


class AlertManager:
    """Evaluates alert rules and manages active alerts."""

    def __init__(self, metrics: MetricsAggregator, health: HealthRegistry):
        self._metrics    = metrics
        self._health     = health
        self._rules:     List[AlertRule]    = []
        self._active:    Dict[str, Alert]   = {}
        self._history:   deque              = deque(maxlen=1000)
        self._last_fired: Dict[str, float] = {}
        self._handlers:  List[Callable[[Alert], None]] = []

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        self._handlers.append(handler)

    def evaluate(self) -> List[Alert]:
        fired = []
        now   = time.time()

        for rule in self._rules:
            try:
                triggered = rule.condition(self._metrics)
            except Exception as e:
                logger.warning("Alert rule %s error: %s", rule.name, e)
                continue

            cooldown_ok = (now - self._last_fired.get(rule.name, 0)) >= rule.cooldown_s

            if triggered and cooldown_ok:
                alert = Alert(
                    rule_name = rule.name,
                    severity  = rule.severity,
                    message   = rule.message or f"Alert: {rule.name}",
                )
                self._active[rule.name] = alert
                self._history.append(alert)
                self._last_fired[rule.name] = now
                fired.append(alert)
                for handler in self._handlers:
                    try:
                        handler(alert)
                    except Exception:
                        pass
            elif not triggered and rule.name in self._active:
                # Resolve
                self._active[rule.name].resolved    = True
                self._active[rule.name].resolved_at = now
                del self._active[rule.name]

        return fired

    def active_alerts(self) -> List[Alert]:
        return list(self._active.values())

    def alert_history(self, n: int = 50) -> List[Alert]:
        return list(self._history)[-n:]

    def add_standard_rules(self) -> None:
        """Register standard AETERNUS alert rules."""

        def high_latency_rule(m: MetricsAggregator) -> bool:
            gauges = m.snapshot()["gauges"]
            for k, v in gauges.items():
                if "latency_p99" in k and v > 10_000_000:  # > 10ms
                    return True
            return False

        def high_error_rate(m: MetricsAggregator) -> bool:
            counters = m.snapshot()["counters"]
            errors = sum(v for k, v in counters.items() if "error" in k)
            total  = sum(v for k, v in counters.items() if "total" in k)
            return total > 100 and errors / total > 0.01

        self.add_rule(AlertRule(
            name="high_publish_latency",
            condition=high_latency_rule,
            severity="warning",
            message="SHM publish latency p99 > 10ms",
        ))
        self.add_rule(AlertRule(
            name="high_error_rate",
            condition=high_error_rate,
            severity="critical",
            message="Error rate > 1%",
        ))


# ---------------------------------------------------------------------------
# Live monitoring dashboard
# ---------------------------------------------------------------------------

class MonitoringDashboard:
    """ASCII terminal dashboard for live monitoring."""

    def __init__(self, health: HealthRegistry, metrics: MetricsAggregator,
                 alerts: AlertManager):
        self._health  = health
        self._metrics = metrics
        self._alerts  = alerts

    def render(self) -> str:
        now    = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        lines  = []
        lines.append("=" * 72)
        lines.append(f"  AETERNUS RTEL Monitor   {now}")
        lines.append("=" * 72)

        # Health
        lines.append("\n[Health Status]")
        health = self._health.to_dict()
        overall = self._health.overall_status()
        status_icon = {"ok": "✓", "warning": "⚠", "critical": "✗",
                       "unknown": "?"}.get(overall, "?")
        lines.append(f"  Overall: {status_icon} {overall.upper()}")
        for name, info in sorted(health.items()):
            icon = {"ok": "✓", "warning": "⚠", "critical": "✗",
                    "unknown": "?"}.get(info["status"], "?")
            stale = f"(stale {info['staleness']:.0f}s)" if info["staleness"] > 5 else ""
            up    = f"uptime={info['uptime_pct']:.1f}%"
            lines.append(f"  {icon} {name:<30} {info['status']:<10} {up}  {stale}")

        # Metrics
        lines.append("\n[Key Metrics]")
        snap    = self._metrics.snapshot()
        gauges  = snap["gauges"]
        counters= snap["counters"]
        display_gauges = sorted([
            (k, v) for k, v in gauges.items()
            if any(kw in k for kw in ["latency", "lag", "equity", "vol", "rate"])
        ], key=lambda x: x[0])[:10]
        for k, v in display_gauges:
            lines.append(f"  {k:<40} {v:>12.4g}")

        display_counters = sorted([
            (k, v) for k, v in counters.items()
        ], key=lambda x: x[0])[:10]
        for k, v in display_counters:
            lines.append(f"  {k:<40} {v:>12.0f}")

        # Alerts
        active = self._alerts.active_alerts()
        lines.append(f"\n[Active Alerts: {len(active)}]")
        if not active:
            lines.append("  No active alerts")
        for alert in active[:5]:
            ts = time.strftime("%H:%M:%S", time.gmtime(alert.timestamp))
            lines.append(f"  [{alert.severity.upper()}] {alert.rule_name} @ {ts}: {alert.message}")

        lines.append("=" * 72)
        return "\n".join(lines)

    def print_dashboard(self) -> None:
        print(self.render())


# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------

class StructuredLogger:
    """JSON-structured logger for AETERNUS events."""

    def __init__(self, component: str, sink: Optional[deque] = None):
        self.component = component
        self._sink     = sink or deque(maxlen=10000)
        self._logger   = logging.getLogger(f"rtel.{component}")

    def _log(self, level: str, event: str, **kwargs) -> None:
        entry = {
            "ts":        time.time(),
            "level":     level,
            "component": self.component,
            "event":     event,
            **kwargs,
        }
        self._sink.append(entry)
        msg = f"[{event}] " + " ".join(f"{k}={v}" for k, v in kwargs.items())
        if level == "DEBUG":    self._logger.debug(msg)
        elif level == "INFO":   self._logger.info(msg)
        elif level == "WARN":   self._logger.warning(msg)
        elif level == "ERROR":  self._logger.error(msg)

    def debug(self, event: str, **kwargs) -> None:
        self._log("DEBUG", event, **kwargs)

    def info(self, event: str, **kwargs) -> None:
        self._log("INFO", event, **kwargs)

    def warn(self, event: str, **kwargs) -> None:
        self._log("WARN", event, **kwargs)

    def error(self, event: str, **kwargs) -> None:
        self._log("ERROR", event, **kwargs)

    def recent(self, n: int = 20) -> List[dict]:
        return list(self._sink)[-n:]


# ---------------------------------------------------------------------------
# MonitoringSystem — top-level integration
# ---------------------------------------------------------------------------

class MonitoringSystem:
    """
    Integrates health checks, metrics, alerts, and dashboard
    into a single monitoring system for AETERNUS RTEL.
    """

    def __init__(self):
        self.health    = HealthRegistry()
        self.metrics   = MetricsAggregator()
        self.alerts    = AlertManager(self.metrics, self.health)
        self.dashboard = MonitoringDashboard(self.health, self.metrics, self.alerts)
        self._logger   = StructuredLogger("monitoring")
        self._running  = False
        self._thread: Optional[threading.Thread] = None

    def register_component(self, name: str, check_fn: CheckFn) -> None:
        self.health.register(name, check_fn)
        self._logger.info("component_registered", name=name)

    def start_background_polling(self, interval_s: float = 5.0) -> None:
        """Start background thread for health checks and alert evaluation."""
        self._running = True
        def poll_loop():
            while self._running:
                try:
                    self.health.check_all()
                    fired = self.alerts.evaluate()
                    for alert in fired:
                        self._logger.warn("alert_fired",
                                          rule=alert.rule_name,
                                          severity=alert.severity)
                    self.metrics.set_gauge(
                        "rtel_monitor_health_ok",
                        1.0 if self.health.overall_status() == HealthStatus.OK else 0.0)
                    self.metrics.set_gauge("rtel_monitor_active_alerts",
                                          float(len(self.alerts.active_alerts())))
                except Exception as e:
                    self._logger.error("poll_error", error=str(e))
                time.sleep(interval_s)

        self._thread = threading.Thread(target=poll_loop, daemon=True)
        self._thread.start()
        self._logger.info("background_polling_started", interval_s=interval_s)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._logger.info("monitoring_stopped")

    def report(self) -> str:
        return self.dashboard.render()

    def export_prometheus(self) -> str:
        return self.metrics.export_prometheus()

    def add_standard_alerts(self) -> None:
        self.alerts.add_standard_rules()
        # Health-based alert
        def critical_component(m):
            health = self.health.to_dict()
            return any(info["status"] == HealthStatus.CRITICAL
                       for info in health.values())
        self.alerts.add_rule(AlertRule(
            name      = "critical_component",
            condition = critical_component,
            severity  = "critical",
            message   = "One or more components are in CRITICAL state",
        ))
