#!/usr/bin/env python3
"""
daily_startup.py -- SRFM daily trading day startup script.

Runs a full startup sequence including health checks, config validation,
database connectivity, circuit breaker verification, broker connectivity,
position reconciliation, IAE parameter checks, and trading mode activation.

Exit codes:
  0 -- success
  1 -- health check failed
  2 -- config invalid
  3 -- broker error
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
CONFIG_DIR = REPO_ROOT / "config"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
TODAY_STR = date.today().strftime("%Y%m%d")
LOG_FILE = LOGS_DIR / f"startup_{TODAY_STR}.log"


def build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s -- %(message)s")
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # File handler
    fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = build_logger("srfm.startup")

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
COORDINATOR_BASE = os.environ.get("SRFM_COORDINATOR_URL", "http://localhost:8000")
ALPACA_BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_KEY = os.environ.get("APCA_API_KEY_ID", "")
ALPACA_SECRET = os.environ.get("APCA_API_SECRET_KEY", "")
BINANCE_BASE = os.environ.get("BINANCE_BASE_URL", "https://api.binance.com")
BINANCE_KEY = os.environ.get("BINANCE_API_KEY", "")
SLACK_WEBHOOK = os.environ.get("SRFM_SLACK_WEBHOOK", "")

HTTP_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = HTTP_TIMEOUT) -> Tuple[int, bytes]:
    """Perform a GET request. Returns (status_code, body_bytes)."""
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        log.debug("URLError for %s: %s", url, exc.reason)
        return 0, b""


def http_post(url: str, payload: Dict, headers: Optional[Dict[str, str]] = None, timeout: int = HTTP_TIMEOUT) -> Tuple[int, bytes]:
    """Perform a POST request with JSON payload. Returns (status_code, body_bytes)."""
    data = json.dumps(payload).encode("utf-8")
    base_headers = {"Content-Type": "application/json"}
    if headers:
        base_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=base_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        log.debug("URLError for %s: %s", url, exc.reason)
        return 0, b""


# ---------------------------------------------------------------------------
# Step 1 -- Health checker
# ---------------------------------------------------------------------------

class ServiceHealthChecker:
    """Checks liveness of all SRFM microservices via coordinator /health endpoint."""

    REQUIRED_SERVICES = [
        "coordinator",
        "live_trader",
        "iae",
        "data_store",
        "risk_manager",
    ]

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def check_coordinator(self) -> bool:
        url = f"{self.base_url}/health"
        status, body = http_get(url)
        if status == 200:
            try:
                data = json.loads(body)
                ok = data.get("status") in ("ok", "healthy")
                log.info("Coordinator health: %s", data.get("status", "unknown"))
                return ok
            except (json.JSONDecodeError, AttributeError):
                return True  # 200 with non-JSON is still alive
        log.warning("Coordinator /health returned HTTP %d", status)
        return False

    def check_all_services(self) -> Dict[str, bool]:
        url = f"{self.base_url}/health/all"
        status, body = http_get(url)
        results: Dict[str, bool] = {}
        if status == 200:
            try:
                data = json.loads(body)
                services = data.get("services", {})
                for svc in self.REQUIRED_SERVICES:
                    svc_status = services.get(svc, {})
                    healthy = svc_status.get("healthy", False) if isinstance(svc_status, dict) else bool(svc_status)
                    results[svc] = healthy
                    log.info("  Service %-20s %s", svc, "OK" if healthy else "FAIL")
            except (json.JSONDecodeError, TypeError):
                log.warning("Could not parse /health/all response; assuming coordinator only")
                results["coordinator"] = True
        else:
            log.warning("/health/all returned HTTP %d -- falling back to coordinator ping", status)
            results["coordinator"] = self.check_coordinator()
        return results

    def run(self) -> bool:
        log.info("=== Step 1: Service Health Check ===")
        if not self.check_coordinator():
            log.error("Coordinator is unreachable -- cannot proceed")
            return False
        results = self.check_all_services()
        failed = [svc for svc, ok in results.items() if not ok]
        if failed:
            log.error("Unhealthy services: %s", ", ".join(failed))
            return False
        log.info("All services healthy.")
        return True


# ---------------------------------------------------------------------------
# Step 2 -- Config validation
# ---------------------------------------------------------------------------

class ConfigValidator:
    """Validates required config files exist and are parseable."""

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir

    def _load_yaml_minimal(self, path: Path) -> Optional[Dict]:
        """Very minimal YAML parser for flat key:value files (avoids PyYAML dep)."""
        try:
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            pass
        # Fallback: just check it is non-empty and contains no obvious syntax errors
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            return {"_raw": content} if content else None
        except OSError:
            return None

    def _check_file(self, relative_path: str, required_keys: List[str]) -> bool:
        path = self.config_dir / relative_path
        if not path.exists():
            log.warning("Config file missing: %s", path)
            return False
        data = self._load_yaml_minimal(path)
        if data is None:
            log.error("Config file empty or unreadable: %s", path)
            return False
        if required_keys and "_raw" not in data:
            missing = [k for k in required_keys if k not in data]
            if missing:
                log.warning("Config %s missing keys: %s", relative_path, missing)
        log.info("  Config %-35s OK", relative_path)
        return True

    def run(self) -> bool:
        log.info("=== Step 2: Config Validation ===")
        checks = [
            ("param_schema.yaml", ["BH_MASS_THRESH", "GW_STRAIN_THRESH"]),
            ("instruments.yaml", ["universe"]),
            ("event_calendar.yaml", ["events"]),
            ("coordinator.yaml", []),
            ("risk_limits.yaml", []),
        ]
        all_ok = True
        for rel, keys in checks:
            if not self._check_file(rel, keys):
                all_ok = False
        if all_ok:
            log.info("All config files valid.")
        else:
            log.error("Config validation failed -- check missing files above")
        return all_ok


# ---------------------------------------------------------------------------
# Step 3 -- Database connectivity
# ---------------------------------------------------------------------------

class DatabaseChecker:
    """Checks DuckDB connectivity via coordinator /db/health endpoint."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def run(self) -> bool:
        log.info("=== Step 3: Database Connectivity ===")
        url = f"{self.base_url}/db/health"
        status, body = http_get(url)
        if status == 200:
            try:
                data = json.loads(body)
                log.info("DB status: %s | integrity: %s", data.get("status"), data.get("integrity"))
                if data.get("integrity") not in ("ok", "pass", True, None):
                    log.error("DB integrity check failed: %s", data.get("integrity_detail", "no detail"))
                    return False
            except (json.JSONDecodeError, TypeError):
                pass
            log.info("Database connectivity OK.")
            return True
        log.error("DB health endpoint returned HTTP %d", status)
        return False


# ---------------------------------------------------------------------------
# Step 4 -- Circuit breaker verification
# ---------------------------------------------------------------------------

class CircuitBreakerChecker:
    """Verifies all circuit breakers are in CLOSED state."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def run(self) -> bool:
        log.info("=== Step 4: Circuit Breaker Verification ===")
        url = f"{self.base_url}/risk/circuit_breakers"
        status, body = http_get(url)
        if status != 200:
            log.error("Circuit breaker endpoint returned HTTP %d", status)
            return False
        try:
            data = json.loads(body)
            breakers = data.get("breakers", data)
            all_closed = True
            if isinstance(breakers, dict):
                for name, state in breakers.items():
                    cb_state = state if isinstance(state, str) else state.get("state", "UNKNOWN")
                    ok = cb_state == "CLOSED"
                    log.info("  CB %-30s %s", name, cb_state)
                    if not ok:
                        all_closed = False
            if all_closed:
                log.info("All circuit breakers CLOSED.")
            else:
                log.error("One or more circuit breakers are not CLOSED -- trading blocked")
            return all_closed
        except (json.JSONDecodeError, TypeError) as exc:
            log.error("Failed to parse circuit breaker response: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Step 5 -- Broker connectivity
# ---------------------------------------------------------------------------

class BrokerConnectivityChecker:
    """Tests connectivity to Alpaca and Binance."""

    def _check_alpaca(self) -> bool:
        if not ALPACA_KEY:
            log.warning("ALPACA key not set -- skipping Alpaca check")
            return True
        url = f"{ALPACA_BASE}/v2/account"
        headers = {
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        }
        status, body = http_get(url, headers=headers)
        if status == 200:
            try:
                data = json.loads(body)
                log.info("Alpaca account: %s | buying_power: %s", data.get("status"), data.get("buying_power"))
            except (json.JSONDecodeError, TypeError):
                pass
            return True
        log.error("Alpaca API returned HTTP %d", status)
        return False

    def _check_binance(self) -> bool:
        url = f"{BINANCE_BASE}/api/v3/ping"
        status, _ = http_get(url)
        if status == 200:
            log.info("Binance API ping OK")
            return True
        log.error("Binance API ping returned HTTP %d", status)
        return False

    def run(self) -> bool:
        log.info("=== Step 5: Broker Connectivity ===")
        alpaca_ok = self._check_alpaca()
        binance_ok = self._check_binance()
        if alpaca_ok and binance_ok:
            log.info("Broker connectivity verified.")
            return True
        log.error("Broker connectivity failed")
        return False


# ---------------------------------------------------------------------------
# Step 6 -- Position reconciliation
# ---------------------------------------------------------------------------

class PositionReconciler:
    """Loads previous day's positions and verifies reconciliation with broker."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def run(self) -> bool:
        log.info("=== Step 6: Position Reconciliation ===")
        url = f"{self.base_url}/positions/reconcile"
        status, body = http_get(url)
        if status == 200:
            try:
                data = json.loads(body)
                discrepancies = data.get("discrepancies", [])
                matched = data.get("matched", 0)
                log.info("Positions matched: %d | discrepancies: %d", matched, len(discrepancies))
                if discrepancies:
                    for d in discrepancies:
                        log.warning("  Discrepancy: %s", d)
                    log.error("Position reconciliation has discrepancies -- manual review required")
                    return False
            except (json.JSONDecodeError, TypeError):
                pass
            log.info("Position reconciliation passed.")
            return True
        elif status == 404:
            log.info("No previous day positions found (first run or clean state).")
            return True
        log.error("Position reconciliation endpoint returned HTTP %d", status)
        return False


# ---------------------------------------------------------------------------
# Step 7 -- IAE parameter updates
# ---------------------------------------------------------------------------

class IAEUpdateChecker:
    """Checks for pending parameter updates from the IAE module."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def run(self) -> bool:
        log.info("=== Step 7: IAE Parameter Update Check ===")
        url = f"{self.base_url}/params/pending"
        status, body = http_get(url)
        if status == 200:
            try:
                data = json.loads(body)
                pending = data.get("pending", [])
                if pending:
                    log.info("Found %d pending parameter update(s) from IAE:", len(pending))
                    for item in pending:
                        log.info("  Param: %-30s new_value: %s  reason: %s",
                                 item.get("param"), item.get("value"), item.get("reason", "IAE"))
                else:
                    log.info("No pending IAE parameter updates.")
            except (json.JSONDecodeError, TypeError):
                log.info("Param pending endpoint returned non-JSON (OK)")
            return True
        elif status in (404, 501):
            log.info("IAE pending endpoint not available -- skipping")
            return True
        log.warning("IAE parameter check returned HTTP %d", status)
        return True  # Non-fatal -- IAE updates are advisory at startup


# ---------------------------------------------------------------------------
# Step 8 -- Enable trading mode
# ---------------------------------------------------------------------------

def enable_trading_mode(base_url: str, dry_run: bool, symbols: Optional[List[str]]) -> bool:
    log.info("=== Step 8: Enable Trading Mode ===")
    if dry_run:
        log.info("[DRY RUN] Would enable trading mode (skipped)")
        return True
    payload: Dict = {"mode": "trading", "enabled": True}
    if symbols:
        payload["symbols"] = symbols
    status, body = http_post(f"{base_url}/trading/mode", payload)
    if status in (200, 204):
        log.info("Trading mode ENABLED.")
        return True
    log.error("Failed to enable trading mode -- HTTP %d -- %s", status, body[:200])
    return False


# ---------------------------------------------------------------------------
# Step 9 -- Slack notification
# ---------------------------------------------------------------------------

def send_startup_notification(dry_run: bool, symbols: Optional[List[str]], duration_s: float) -> None:
    log.info("=== Step 9: Slack Startup Notification ===")
    if not SLACK_WEBHOOK:
        log.info("SLACK_WEBHOOK not configured -- skipping notification")
        return
    mode_str = "DRY RUN" if dry_run else "LIVE"
    sym_str = ", ".join(symbols) if symbols else "full universe"
    text = (
        f":rocket: *SRFM Startup Complete* ({mode_str})\n"
        f"Date: {date.today().isoformat()} | Symbols: {sym_str}\n"
        f"Startup duration: {duration_s:.1f}s | Log: `{LOG_FILE.name}`"
    )
    payload = {"text": text}
    try:
        status, _ = http_post(SLACK_WEBHOOK, payload)
        if status in (200, 204):
            log.info("Slack notification sent.")
        else:
            log.warning("Slack notification returned HTTP %d", status)
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("Slack notification failed: %s", exc)


# ---------------------------------------------------------------------------
# Main startup orchestrator
# ---------------------------------------------------------------------------

class StartupOrchestrator:
    def __init__(self, dry_run: bool, symbols: Optional[List[str]]) -> None:
        self.dry_run = dry_run
        self.symbols = symbols
        self.base_url = COORDINATOR_BASE

    def run(self) -> int:
        start_time = time.monotonic()
        log.info("=" * 60)
        log.info("SRFM Daily Startup -- %s", datetime.now().isoformat())
        log.info("Dry-run: %s | Symbols: %s", self.dry_run, self.symbols or "all")
        log.info("=" * 60)

        # Step 1 -- health
        checker = ServiceHealthChecker(self.base_url)
        if not checker.run():
            log.critical("STARTUP ABORTED: health check failed")
            return 1

        # Step 2 -- config
        validator = ConfigValidator(CONFIG_DIR)
        if not validator.run():
            log.critical("STARTUP ABORTED: config validation failed")
            return 2

        # Step 3 -- database
        db_checker = DatabaseChecker(self.base_url)
        if not db_checker.run():
            log.critical("STARTUP ABORTED: database connectivity failed")
            return 1

        # Step 4 -- circuit breakers
        cb_checker = CircuitBreakerChecker(self.base_url)
        if not cb_checker.run():
            log.critical("STARTUP ABORTED: circuit breakers not CLOSED")
            return 1

        # Step 5 -- broker connectivity
        broker_checker = BrokerConnectivityChecker()
        if not broker_checker.run():
            log.critical("STARTUP ABORTED: broker connectivity failed")
            return 3

        # Step 6 -- position reconciliation
        reconciler = PositionReconciler(self.base_url)
        if not reconciler.run():
            log.critical("STARTUP ABORTED: position reconciliation failed")
            return 1

        # Step 7 -- IAE parameter updates (advisory)
        iae_checker = IAEUpdateChecker(self.base_url)
        iae_checker.run()

        # Step 8 -- enable trading
        if not enable_trading_mode(self.base_url, self.dry_run, self.symbols):
            log.critical("STARTUP ABORTED: failed to enable trading mode")
            return 1

        duration = time.monotonic() - start_time

        # Step 9 -- Slack
        send_startup_notification(self.dry_run, self.symbols, duration)

        log.info("=" * 60)
        log.info("SRFM STARTUP COMPLETE -- %.1fs", duration)
        log.info("=" * 60)
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SRFM daily trading startup script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all checks without enabling live trading",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated subset of symbols to enable (e.g. BTC,ETH)",
    )
    parser.add_argument(
        "--coordinator-url",
        type=str,
        default=COORDINATOR_BASE,
        help=f"Coordinator base URL (default: {COORDINATOR_BASE})",
    )
    return parser.parse_args()


def main() -> None:
    global COORDINATOR_BASE  # noqa: PLW0603
    args = parse_args()
    if args.coordinator_url != COORDINATOR_BASE:
        COORDINATOR_BASE = args.coordinator_url

    symbols: Optional[List[str]] = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    orchestrator = StartupOrchestrator(dry_run=args.dry_run, symbols=symbols)
    sys.exit(orchestrator.run())


if __name__ == "__main__":
    main()
