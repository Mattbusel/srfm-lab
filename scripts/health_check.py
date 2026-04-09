#!/usr/bin/env python3
"""
scripts/health_check.py
=======================
Comprehensive health check for all SRFM Lab services.

Polls all HTTP endpoints, verifies SQLite DB accessibility, checks that the
C++ signal engine is writing to its IPC ring buffer, and optionally tests
Alpaca API connectivity.

Output: JSON report to stdout
Exit codes:
  0 -- all services healthy
  1 -- some services degraded (non-critical failures)
  2 -- critical failure (trader or DB inaccessible)

Usage:
    python scripts/health_check.py [--json-only] [--timeout 5]
    python scripts/health_check.py --check alpaca,risk_api

Environment variables used (from .env or shell):
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
    SRFM_DB_PATH       path to live_trades.db
    SRFM_IPC_PATH      path to signal engine ring buffer sentinel
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import httpx
    _HTTPX = True
except ImportError:
    try:
        import urllib.request
        _HTTPX = False
    except ImportError:
        _HTTPX = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parents[1]

SERVICES: Dict[str, dict] = {
    "coordination": {
        "url": os.environ.get("SRFM_COORD_URL", "http://localhost:8781/health"),
        "critical": False,
        "timeout": 3.0,
    },
    "market_data": {
        "url": os.environ.get("SRFM_MKT_URL", "http://localhost:8780/health"),
        "critical": False,
        "timeout": 3.0,
    },
    "risk_api": {
        "url": os.environ.get("SRFM_RISK_URL", "http://localhost:8791/risk/health"),
        "critical": True,
        "timeout": 5.0,
    },
    "risk_aggregator": {
        "url": os.environ.get("SRFM_RISK_AGG_URL", "http://localhost:8792/health"),
        "critical": False,
        "timeout": 3.0,
    },
    "observability_api": {
        "url": os.environ.get("SRFM_OBS_URL", "http://localhost:9091/health"),
        "critical": False,
        "timeout": 3.0,
    },
    "metrics_collector": {
        "url": os.environ.get("SRFM_METRICS_URL", "http://localhost:9090/metrics"),
        "critical": False,
        "timeout": 3.0,
    },
}

DB_PATH = Path(os.environ.get("SRFM_DB_PATH", str(REPO_ROOT / "execution" / "live_trades.db")))
IPC_SENTINEL = Path(os.environ.get("SRFM_IPC_PATH", "/tmp/srfm_signal_engine.ready"))
IPC_RING_PATH = Path(os.environ.get("SRFM_RING_PATH", "/dev/shm/srfm_signals"))

ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ServiceResult:
    name: str
    status: str           # "ok" | "degraded" | "down" | "skipped"
    latency_ms: float = 0.0
    message: str = ""
    critical: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    status: str           # "ok" | "degraded" | "down"
    timestamp: str = ""
    services: Dict[str, Any] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# HTTP probe
# ---------------------------------------------------------------------------

def http_probe(name: str, url: str, timeout: float, critical: bool) -> ServiceResult:
    """Issue an HTTP GET and return a ServiceResult."""
    start = time.monotonic()
    try:
        if _HTTPX:
            import httpx
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url)
            elapsed_ms = (time.monotonic() - start) * 1000
            if resp.status_code < 400:
                try:
                    body = resp.json()
                except Exception:
                    body = {}
                svc_status = body.get("status", "ok") if isinstance(body, dict) else "ok"
                return ServiceResult(
                    name=name,
                    status="ok" if svc_status in ("ok", "healthy", "up") else "degraded",
                    latency_ms=round(elapsed_ms, 2),
                    message=f"HTTP {resp.status_code}",
                    critical=critical,
                    details=body if isinstance(body, dict) else {},
                )
            else:
                elapsed_ms = (time.monotonic() - start) * 1000
                return ServiceResult(
                    name=name,
                    status="down",
                    latency_ms=round(elapsed_ms, 2),
                    message=f"HTTP {resp.status_code}",
                    critical=critical,
                )
        else:
            import urllib.request
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed_ms = (time.monotonic() - start) * 1000
                body_bytes = resp.read()
                try:
                    body = json.loads(body_bytes)
                except Exception:
                    body = {}
                return ServiceResult(
                    name=name,
                    status="ok",
                    latency_ms=round(elapsed_ms, 2),
                    message=f"HTTP {resp.status}",
                    critical=critical,
                    details=body if isinstance(body, dict) else {},
                )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        return ServiceResult(
            name=name,
            status="down",
            latency_ms=round(elapsed_ms, 2),
            message=str(exc)[:120],
            critical=critical,
        )


# ---------------------------------------------------------------------------
# Database probe
# ---------------------------------------------------------------------------

def check_database(db_path: Path) -> ServiceResult:
    """Verify the SQLite database is accessible and has expected tables."""
    if not db_path.exists():
        return ServiceResult(
            name="database",
            status="down",
            message=f"DB file not found: {db_path}",
            critical=True,
        )
    start = time.monotonic()
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode")  # lightweight probe
        tables_raw = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        elapsed_ms = (time.monotonic() - start) * 1000
        tables = [r[0] for r in tables_raw]

        expected = {"live_trades", "positions"}
        missing = expected - set(tables)

        # Check WAL health: if db is locked, this will raise
        conn.execute("SELECT COUNT(*) FROM live_trades LIMIT 1")
        trade_count = conn.execute("SELECT COUNT(*) FROM live_trades").fetchone()[0]
        conn.close()

        if missing:
            return ServiceResult(
                name="database",
                status="degraded",
                latency_ms=round(elapsed_ms, 2),
                message=f"Missing tables: {missing}",
                critical=True,
                details={"tables": tables, "missing": list(missing)},
            )
        return ServiceResult(
            name="database",
            status="ok",
            latency_ms=round(elapsed_ms, 2),
            message=f"{trade_count} trades on record",
            critical=True,
            details={"tables": tables, "trade_count": trade_count},
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        return ServiceResult(
            name="database",
            status="down",
            latency_ms=round(elapsed_ms, 2),
            message=str(exc)[:120],
            critical=True,
        )


# ---------------------------------------------------------------------------
# Signal engine / IPC ring buffer probe
# ---------------------------------------------------------------------------

def check_signal_engine() -> ServiceResult:
    """
    Verify the C++ signal engine is alive by checking:
    1. Sentinel file exists and was modified recently (< 60s)
    2. Ring buffer file exists and has non-zero size
    """
    details: Dict[str, Any] = {}

    # Check sentinel
    if IPC_SENTINEL.exists():
        age_s = time.time() - IPC_SENTINEL.stat().st_mtime
        details["sentinel_age_s"] = round(age_s, 1)
        if age_s > 120:
            return ServiceResult(
                name="signal_engine",
                status="degraded",
                message=f"Sentinel file stale: {age_s:.0f}s ago",
                critical=False,
                details=details,
            )
    else:
        details["sentinel_found"] = False
        # Not a hard failure -- engine might write to ring without sentinel on some platforms

    # Check ring buffer file
    if IPC_RING_PATH.exists():
        size = IPC_RING_PATH.stat().st_size
        age_s = time.time() - IPC_RING_PATH.stat().st_mtime
        details["ring_size_bytes"] = size
        details["ring_age_s"] = round(age_s, 1)
        if size == 0:
            return ServiceResult(
                name="signal_engine",
                status="degraded",
                message="Ring buffer file is empty",
                critical=False,
                details=details,
            )
        if age_s > 300:
            return ServiceResult(
                name="signal_engine",
                status="degraded",
                message=f"Ring buffer not updated in {age_s:.0f}s",
                critical=False,
                details=details,
            )
        return ServiceResult(
            name="signal_engine",
            status="ok",
            message=f"Ring buffer active ({size} bytes, updated {age_s:.0f}s ago)",
            critical=False,
            details=details,
        )

    # Neither sentinel nor ring buffer found
    if not IPC_SENTINEL.exists():
        return ServiceResult(
            name="signal_engine",
            status="down",
            message="Neither IPC sentinel nor ring buffer found",
            critical=False,
            details=details,
        )

    return ServiceResult(
        name="signal_engine",
        status="ok",
        message="Sentinel found, ring buffer path not present (may be platform-specific)",
        critical=False,
        details=details,
    )


# ---------------------------------------------------------------------------
# Alpaca connectivity probe
# ---------------------------------------------------------------------------

def check_alpaca(timeout: float = 5.0) -> ServiceResult:
    """
    Probe Alpaca API connectivity.
    Uses /v2/account endpoint with API key headers.
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return ServiceResult(
            name="alpaca",
            status="skipped",
            message="ALPACA_API_KEY / ALPACA_SECRET_KEY not set",
            critical=False,
        )

    url = f"{ALPACA_BASE_URL}/v2/account"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    start = time.monotonic()
    try:
        if _HTTPX:
            import httpx
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url, headers=headers)
            elapsed_ms = (time.monotonic() - start) * 1000
            if resp.status_code == 200:
                body = resp.json()
                return ServiceResult(
                    name="alpaca",
                    status="ok",
                    latency_ms=round(elapsed_ms, 2),
                    message=f"Account status: {body.get('status', 'unknown')}",
                    critical=True,
                    details={
                        "account_status": body.get("status"),
                        "equity": body.get("equity"),
                        "buying_power": body.get("buying_power"),
                    },
                )
            elif resp.status_code == 403:
                return ServiceResult(
                    name="alpaca",
                    status="down",
                    latency_ms=round(elapsed_ms, 2),
                    message="Alpaca: authentication failed (check API keys)",
                    critical=True,
                )
            else:
                return ServiceResult(
                    name="alpaca",
                    status="degraded",
                    latency_ms=round(elapsed_ms, 2),
                    message=f"Alpaca HTTP {resp.status_code}",
                    critical=True,
                )
        else:
            import urllib.request
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed_ms = (time.monotonic() - start) * 1000
                body = json.loads(resp.read())
                return ServiceResult(
                    name="alpaca",
                    status="ok",
                    latency_ms=round(elapsed_ms, 2),
                    message=f"Account: {body.get('status', 'unknown')}",
                    critical=True,
                    details=body,
                )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        return ServiceResult(
            name="alpaca",
            status="down",
            latency_ms=round(elapsed_ms, 2),
            message=str(exc)[:120],
            critical=True,
        )


# ---------------------------------------------------------------------------
# Coordination circuit breaker probe
# ---------------------------------------------------------------------------

def check_circuit_breakers(timeout: float = 3.0) -> ServiceResult:
    """
    Query coordination layer for circuit breaker states.
    Returns degraded if any breaker is open.
    """
    coord_url = os.environ.get("SRFM_COORD_URL", "http://localhost:8781").rstrip("/")
    # Fix: rstrip('/health') strips individual chars, not the suffix. Use removesuffix.
    if coord_url.endswith("/health"):
        coord_url = coord_url[:-7]
    url = f"{coord_url}/health"
    result = http_probe("circuit_breakers", url, timeout, critical=False)

    if result.status == "ok" and result.details:
        open_circuits = [
            svc for svc, state in result.details.get("services", {}).items()
            if state == "open"
        ]
        if open_circuits:
            return ServiceResult(
                name="circuit_breakers",
                status="degraded",
                latency_ms=result.latency_ms,
                message=f"Open circuits: {', '.join(open_circuits)}",
                critical=False,
                details={"open_circuits": open_circuits, "all_states": result.details.get("services", {})},
            )
        return ServiceResult(
            name="circuit_breakers",
            status="ok",
            latency_ms=result.latency_ms,
            message="All circuits closed",
            critical=False,
            details=result.details,
        )
    return result


# ---------------------------------------------------------------------------
# Main health check orchestrator
# ---------------------------------------------------------------------------

def run_health_check(
    checks: Optional[List[str]] = None,
    timeout_override: Optional[float] = None,
) -> HealthReport:
    """
    Run all (or selected) health checks.
    Returns a populated HealthReport.
    """
    results: List[ServiceResult] = []

    # HTTP service probes
    for name, cfg in SERVICES.items():
        if checks and name not in checks:
            continue
        t = timeout_override or cfg["timeout"]
        results.append(http_probe(name, cfg["url"], t, cfg["critical"]))

    # Database
    if not checks or "database" in checks:
        results.append(check_database(DB_PATH))

    # Signal engine
    if not checks or "signal_engine" in checks:
        results.append(check_signal_engine())

    # Alpaca
    if not checks or "alpaca" in checks:
        results.append(check_alpaca(timeout_override or 5.0))

    # Circuit breakers
    if not checks or "circuit_breakers" in checks:
        results.append(check_circuit_breakers(timeout_override or 3.0))

    # Determine overall status
    counts = {"ok": 0, "degraded": 0, "down": 0, "skipped": 0}
    alerts: List[str] = []
    services_dict: Dict[str, Any] = {}
    critical_failures: List[str] = []

    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        services_dict[r.name] = {
            "status": r.status,
            "latency_ms": r.latency_ms,
            "message": r.message,
            "critical": r.critical,
            "details": r.details,
        }
        if r.status == "down":
            msg = f"{r.name}: DOWN -- {r.message}"
            alerts.append(msg)
            if r.critical:
                critical_failures.append(r.name)
        elif r.status == "degraded":
            alerts.append(f"{r.name}: DEGRADED -- {r.message}")

    if critical_failures:
        overall_status = "down"
    elif counts["degraded"] > 0 or counts["down"] > 0:
        overall_status = "degraded"
    else:
        overall_status = "ok"

    return HealthReport(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=services_dict,
        alerts=alerts,
        summary=counts,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SRFM Lab health check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        type=str,
        default="",
        help="Comma-separated list of checks to run (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override request timeout in seconds",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only emit JSON, no human-readable output",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    args = parser.parse_args()

    checks = [c.strip() for c in args.check.split(",") if c.strip()] or None
    report = run_health_check(checks=checks, timeout_override=args.timeout)

    # JSON output
    indent = 2 if args.pretty else None
    json_str = json.dumps(report.to_dict(), indent=indent)

    if args.json_only:
        print(json_str)
    else:
        # Human-readable summary first
        STATUS_ICONS = {"ok": "\u2705", "degraded": "\u26a0\ufe0f", "down": "\u274c", "skipped": "\u23e9"}
        icon = STATUS_ICONS.get(report.status, "?")
        print(f"\n{icon} SRFM Lab Health: {report.status.upper()}")
        print(f"  Timestamp: {report.timestamp}")
        print(f"  Summary: {report.summary}")
        print("")
        print(f"{'Service':<25} {'Status':<12} {'Latency (ms)':<15} Message")
        print("-" * 80)
        for svc_name, svc_info in report.services.items():
            status = svc_info["status"]
            lat = f"{svc_info['latency_ms']:.1f}" if svc_info["latency_ms"] else "--"
            msg = svc_info["message"][:50]
            icon_svc = STATUS_ICONS.get(status, "?")
            print(f"{icon_svc} {svc_name:<23} {status:<12} {lat:<15} {msg}")

        if report.alerts:
            print("\nAlerts:")
            for alert in report.alerts:
                print(f"  - {alert}")

        print("\n--- JSON Report ---")
        print(json_str)
        print("")

    # Exit code
    if report.status == "ok":
        return 0
    elif report.status == "degraded":
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
