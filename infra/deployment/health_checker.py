# infra/deployment/health_checker.py -- comprehensive service health checking for SRFM
from __future__ import annotations

import json
import logging
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ServiceHealth:
    name: str
    status: HealthStatus
    latency_ms: float
    last_check: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def is_ok(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    def summary(self) -> str:
        tag = self.status.value.upper().ljust(11)
        latency = f"{self.latency_ms:7.1f}ms"
        ts = self.last_check.strftime("%H:%M:%S")
        err = f"  err={self.error}" if self.error else ""
        return f"[{tag}] {self.name:<22} {latency}  checked={ts}{err}"


@dataclass
class ServiceEndpoint:
    name: str
    url: str
    http_timeout_s: float = 5.0
    expected_status_codes: Tuple[int, ...] = (200, 204)
    tags: List[str] = field(default_factory=list)


@dataclass
class DegradationAlert:
    service_name: str
    alert_type: str  # "failure_rate" or "latency_spike"
    message: str
    triggered_at: datetime


# ---------------------------------------------------------------------------
# Default service registry
# ---------------------------------------------------------------------------

DEFAULT_SERVICES: List[ServiceEndpoint] = [
    ServiceEndpoint(
        name="live-trader",
        url="http://localhost:8080/health",
        tags=["trading", "critical"],
    ),
    ServiceEndpoint(
        name="coordination",
        url="http://localhost:8781/health",
        tags=["core"],
    ),
    ServiceEndpoint(
        name="risk-api",
        url="http://localhost:8783/health",
        tags=["risk", "critical"],
    ),
    ServiceEndpoint(
        name="market-data",
        url="http://localhost:8784/health",
        tags=["data"],
    ),
    ServiceEndpoint(
        name="idea-engine",
        url="http://localhost:8785/health",
        tags=["alpha"],
    ),
    ServiceEndpoint(
        name="dashboard-api",
        url="http://localhost:9091/health",
        tags=["ui"],
    ),
    ServiceEndpoint(
        name="metrics-collector",
        url="http://localhost:9090/health",
        tags=["observability"],
    ),
]


# ---------------------------------------------------------------------------
# HealthDegradationDetector
# ---------------------------------------------------------------------------

class HealthDegradationDetector:
    """Tracks rolling health windows and fires alerts on degradation.

    -- Window size: last 10 checks per service.
    -- Alert when > 3 failures in window.
    -- Alert when latest latency > 2x rolling mean.
    """

    WINDOW_SIZE: int = 10
    FAILURE_THRESHOLD: int = 3
    LATENCY_SPIKE_MULTIPLIER: float = 2.0

    def __init__(self, alert_callback: Optional[Callable[[DegradationAlert], None]] = None) -> None:
        self._alert_callback = alert_callback or self._default_alert
        # service_name -> deque of (status_ok: bool, latency_ms: float)
        self._windows: Dict[str, deque] = {}
        self._lock = threading.Lock()
        self._fired_alerts: List[DegradationAlert] = []

    # -- public API ----------------------------------------------------------

    def record(self, health: ServiceHealth) -> None:
        """Record a new health observation and evaluate for degradation."""
        ok = health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        with self._lock:
            if health.name not in self._windows:
                self._windows[health.name] = deque(maxlen=self.WINDOW_SIZE)
            self._windows[health.name].append((ok, health.latency_ms))
            window = list(self._windows[health.name])

        self._evaluate_failure_rate(health.name, window)
        self._evaluate_latency_spike(health.name, window, health.latency_ms)

    def recent_alerts(self, n: int = 20) -> List[DegradationAlert]:
        return self._fired_alerts[-n:]

    def clear_alerts(self) -> None:
        self._fired_alerts.clear()

    def window_summary(self, service_name: str) -> Dict[str, Any]:
        with self._lock:
            window = list(self._windows.get(service_name, []))
        if not window:
            return {"size": 0, "failure_count": 0, "mean_latency_ms": 0.0}
        failures = sum(1 for ok, _ in window if not ok)
        latencies = [lat for _, lat in window]
        return {
            "size": len(window),
            "failure_count": failures,
            "mean_latency_ms": statistics.mean(latencies),
        }

    # -- internal ------------------------------------------------------------

    def _evaluate_failure_rate(self, name: str, window: List[Tuple[bool, float]]) -> None:
        failures = sum(1 for ok, _ in window if not ok)
        if failures > self.FAILURE_THRESHOLD:
            alert = DegradationAlert(
                service_name=name,
                alert_type="failure_rate",
                message=(
                    f"Service '{name}' has {failures}/{len(window)} failures "
                    f"in last {self.WINDOW_SIZE}-check window"
                ),
                triggered_at=datetime.utcnow(),
            )
            self._fire(alert)

    def _evaluate_latency_spike(
        self,
        name: str,
        window: List[Tuple[bool, float]],
        latest_ms: float,
    ) -> None:
        if len(window) < 3:
            return
        latencies = [lat for _, lat in window[:-1]]  # exclude latest for baseline
        if not latencies:
            return
        mean_ms = statistics.mean(latencies)
        if mean_ms > 0 and latest_ms > mean_ms * self.LATENCY_SPIKE_MULTIPLIER:
            alert = DegradationAlert(
                service_name=name,
                alert_type="latency_spike",
                message=(
                    f"Service '{name}' latency spike: {latest_ms:.1f}ms "
                    f"vs rolling mean {mean_ms:.1f}ms "
                    f"(>{self.LATENCY_SPIKE_MULTIPLIER}x threshold)"
                ),
                triggered_at=datetime.utcnow(),
            )
            self._fire(alert)

    def _fire(self, alert: DegradationAlert) -> None:
        self._fired_alerts.append(alert)
        logger.warning("DegradationAlert [%s] %s: %s", alert.alert_type, alert.service_name, alert.message)
        try:
            self._alert_callback(alert)
        except Exception:
            logger.exception("Alert callback raised an exception")

    @staticmethod
    def _default_alert(alert: DegradationAlert) -> None:
        print(f"[ALERT] {alert.alert_type.upper()} -- {alert.service_name}: {alert.message}")


# ---------------------------------------------------------------------------
# ServiceHealthChecker
# ---------------------------------------------------------------------------

class ServiceHealthChecker:
    """HTTP-based health checker for all registered SRFM services.

    Usage::

        checker = ServiceHealthChecker()
        results = checker.check_all()
        print(checker.health_report())
    """

    def __init__(
        self,
        services: Optional[List[ServiceEndpoint]] = None,
        degradation_callback: Optional[Callable[[DegradationAlert], None]] = None,
    ) -> None:
        self._services: Dict[str, ServiceEndpoint] = {}
        self._last_results: Dict[str, ServiceHealth] = {}
        self._lock = threading.Lock()
        self._degradation_detector = HealthDegradationDetector(
            alert_callback=degradation_callback
        )

        base = services if services is not None else DEFAULT_SERVICES
        for svc in base:
            self.register(svc)

    # -- registration --------------------------------------------------------

    def register(self, endpoint: ServiceEndpoint) -> None:
        """Add or replace a service endpoint definition."""
        with self._lock:
            self._services[endpoint.name] = endpoint
        logger.debug("Registered health endpoint: %s -> %s", endpoint.name, endpoint.url)

    def unregister(self, name: str) -> bool:
        with self._lock:
            if name in self._services:
                del self._services[name]
                return True
        return False

    def registered_names(self) -> List[str]:
        with self._lock:
            return list(self._services.keys())

    # -- core checks ---------------------------------------------------------

    def check_service(self, name: str) -> ServiceHealth:
        """Perform a single HTTP health check and return a ServiceHealth result."""
        with self._lock:
            endpoint = self._services.get(name)
        if endpoint is None:
            return ServiceHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                latency_ms=0.0,
                last_check=datetime.utcnow(),
                error=f"Service '{name}' not registered",
            )
        return self._http_check(endpoint)

    def check_all(self, parallel: bool = True) -> Dict[str, ServiceHealth]:
        """Check all registered services, optionally in parallel threads."""
        with self._lock:
            names = list(self._services.keys())

        results: Dict[str, ServiceHealth] = {}

        if parallel:
            threads: List[threading.Thread] = []
            result_bucket: Dict[str, ServiceHealth] = {}
            result_lock = threading.Lock()

            def worker(n: str) -> None:
                h = self.check_service(n)
                with result_lock:
                    result_bucket[n] = h

            for name in names:
                t = threading.Thread(target=worker, args=(name,), daemon=True)
                threads.append(t)
                t.start()
            for t in threads:
                t.join(timeout=10.0)
            results = result_bucket
        else:
            for name in names:
                results[name] = self.check_service(name)

        with self._lock:
            self._last_results = dict(results)

        for health in results.values():
            self._degradation_detector.record(health)

        return results

    def last_results(self) -> Dict[str, ServiceHealth]:
        with self._lock:
            return dict(self._last_results)

    # -- wait helpers --------------------------------------------------------

    def wait_until_healthy(
        self,
        name: str,
        timeout_s: float = 60.0,
        poll_interval_s: float = 2.0,
    ) -> bool:
        """Poll a service until it reports HEALTHY or timeout elapses."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            health = self.check_service(name)
            if health.status == HealthStatus.HEALTHY:
                logger.info("Service '%s' is healthy after polling", name)
                return True
            remaining = deadline - time.monotonic()
            logger.debug(
                "Service '%s' is %s, %.1fs remaining",
                name,
                health.status.value,
                remaining,
            )
            if remaining <= 0:
                break
            time.sleep(min(poll_interval_s, remaining))
        logger.warning("Service '%s' did not become healthy within %.1fs", name, timeout_s)
        return False

    def wait_all_healthy(
        self,
        timeout_s: float = 120.0,
        poll_interval_s: float = 3.0,
    ) -> Dict[str, bool]:
        """Wait for every registered service to be healthy. Returns per-service result."""
        with self._lock:
            names = list(self._services.keys())
        return {
            name: self.wait_until_healthy(name, timeout_s=timeout_s, poll_interval_s=poll_interval_s)
            for name in names
        }

    # -- reporting -----------------------------------------------------------

    def health_report(self) -> str:
        """Produce a formatted ASCII health report from the last check results."""
        with self._lock:
            results = dict(self._last_results)

        if not results:
            return "No health data available -- run check_all() first."

        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("  SRFM SERVICE HEALTH REPORT")
        lines.append(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append("=" * 72)

        # Count summary
        counts: Dict[HealthStatus, int] = {s: 0 for s in HealthStatus}
        for h in results.values():
            counts[h.status] += 1

        summary_parts = []
        for status in HealthStatus:
            if counts[status] > 0:
                summary_parts.append(f"{status.value.upper()}={counts[status]}")
        lines.append("  " + "  ".join(summary_parts))
        lines.append("-" * 72)

        # Per-service lines, sorted by status severity then name
        def sort_key(h: ServiceHealth) -> Tuple[int, str]:
            order = {
                HealthStatus.UNHEALTHY: 0,
                HealthStatus.UNREACHABLE: 1,
                HealthStatus.DEGRADED: 2,
                HealthStatus.UNKNOWN: 3,
                HealthStatus.HEALTHY: 4,
            }
            return (order.get(h.status, 99), h.name)

        for health in sorted(results.values(), key=sort_key):
            lines.append("  " + health.summary())

            # Emit key details if present
            for key in ("version", "uptime_s", "error_rate", "db_status", "queue_depth"):
                val = health.details.get(key)
                if val is not None:
                    lines.append(f"    {key}: {val}")

        lines.append("-" * 72)

        # Degradation alerts
        alerts = self._degradation_detector.recent_alerts(10)
        if alerts:
            lines.append("  RECENT DEGRADATION ALERTS:")
            for a in alerts[-5:]:
                ts = a.triggered_at.strftime("%H:%M:%S")
                lines.append(f"    [{ts}] {a.alert_type.upper()} {a.service_name}: {a.message}")
        else:
            lines.append("  No degradation alerts.")

        lines.append("=" * 72)
        return "\n".join(lines)

    def degradation_detector(self) -> HealthDegradationDetector:
        return self._degradation_detector

    # -- internal HTTP machinery ---------------------------------------------

    def _http_check(self, endpoint: ServiceEndpoint) -> ServiceHealth:
        """Execute one HTTP GET to the health endpoint and interpret the response."""
        t_start = time.monotonic()
        try:
            req = urllib.request.Request(
                endpoint.url,
                headers={"Accept": "application/json", "User-Agent": "srfm-health-checker/1.0"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=endpoint.http_timeout_s) as resp:
                latency_ms = (time.monotonic() - t_start) * 1000.0
                status_code: int = resp.status
                raw_body = resp.read()

            details = self._parse_body(raw_body)
            http_ok = status_code in endpoint.expected_status_codes

            if not http_ok:
                return ServiceHealth(
                    name=endpoint.name,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=latency_ms,
                    last_check=datetime.utcnow(),
                    details=details,
                    error=f"HTTP {status_code}",
                )

            # Interpret JSON body for fine-grained status
            health_status = self._interpret_details(details, http_ok)

            return ServiceHealth(
                name=endpoint.name,
                status=health_status,
                latency_ms=latency_ms,
                last_check=datetime.utcnow(),
                details=details,
            )

        except urllib.error.URLError as exc:
            latency_ms = (time.monotonic() - t_start) * 1000.0
            reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)
            return ServiceHealth(
                name=endpoint.name,
                status=HealthStatus.UNREACHABLE,
                latency_ms=latency_ms,
                last_check=datetime.utcnow(),
                error=f"URLError: {reason}",
            )
        except TimeoutError:
            latency_ms = (time.monotonic() - t_start) * 1000.0
            return ServiceHealth(
                name=endpoint.name,
                status=HealthStatus.UNREACHABLE,
                latency_ms=latency_ms,
                last_check=datetime.utcnow(),
                error="Connection timed out",
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - t_start) * 1000.0
            return ServiceHealth(
                name=endpoint.name,
                status=HealthStatus.UNKNOWN,
                latency_ms=latency_ms,
                last_check=datetime.utcnow(),
                error=f"Unexpected error: {exc}",
            )

    @staticmethod
    def _parse_body(raw_body: bytes) -> Dict[str, Any]:
        """Try to parse JSON from response body; return empty dict on failure."""
        if not raw_body:
            return {}
        try:
            data = json.loads(raw_body.decode("utf-8", errors="replace"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return {}

    @staticmethod
    def _interpret_details(details: Dict[str, Any], http_ok: bool) -> HealthStatus:
        """Map JSON body fields to a HealthStatus."""
        if not http_ok:
            return HealthStatus.UNHEALTHY
        if not details:
            return HealthStatus.HEALTHY

        # Standard status field
        raw_status = str(details.get("status", "")).lower()
        if raw_status in ("ok", "healthy", "up"):
            return HealthStatus.HEALTHY
        if raw_status in ("degraded", "warn", "warning"):
            return HealthStatus.DEGRADED
        if raw_status in ("down", "unhealthy", "error", "fail", "failed"):
            return HealthStatus.UNHEALTHY

        # Error rate heuristic
        error_rate = details.get("error_rate")
        if isinstance(error_rate, (int, float)):
            if error_rate > 0.10:
                return HealthStatus.UNHEALTHY
            if error_rate > 0.01:
                return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_default_checker(
    degradation_callback: Optional[Callable[[DegradationAlert], None]] = None,
) -> ServiceHealthChecker:
    """Return a checker pre-loaded with all SRFM default service endpoints."""
    return ServiceHealthChecker(
        services=DEFAULT_SERVICES,
        degradation_callback=degradation_callback,
    )
