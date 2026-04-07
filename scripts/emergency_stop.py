#!/usr/bin/env python3
"""
emergency_stop.py -- SRFM emergency trading halt.

Immediately halts trading, optionally cancels pending orders and/or
flattens all positions. Sends alerts to Slack and PagerDuty.
Writes incident log to logs/incidents/YYYYMMDD_HHMMSS_emergency.json.

Usage:
  python scripts/emergency_stop.py --reason "drawdown_breach" --confirm
  python scripts/emergency_stop.py --reason "api_disconnect" --flatten --cancel-orders --confirm
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
INCIDENTS_DIR = LOGS_DIR / "incidents"
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s -- %(message)s")
    ch = logging.StreamHandler(sys.stderr)  # emergency output goes to stderr
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


log = build_logger("srfm.emergency")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COORDINATOR_BASE = os.environ.get("SRFM_COORDINATOR_URL", "http://localhost:8000")
ALPACA_BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_KEY = os.environ.get("APCA_API_KEY_ID", "")
ALPACA_SECRET = os.environ.get("APCA_API_SECRET_KEY", "")
BINANCE_BASE = os.environ.get("BINANCE_BASE_URL", "https://api.binance.com")
BINANCE_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_SECRET = os.environ.get("BINANCE_API_SECRET", "")
SLACK_WEBHOOK = os.environ.get("SRFM_SLACK_WEBHOOK", "")
PAGERDUTY_KEY = os.environ.get("SRFM_PAGERDUTY_KEY", "")

HTTP_TIMEOUT = 8  # faster timeout for emergency operations


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get(url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError) as exc:
        log.debug("http_get error %s: %s", url, exc)
        return 0, b""


def http_post(url: str, payload: Dict, headers: Optional[Dict[str, str]] = None) -> Tuple[int, bytes]:
    data = json.dumps(payload).encode("utf-8")
    base_headers = {"Content-Type": "application/json"}
    if headers:
        base_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=base_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError) as exc:
        log.debug("http_post error %s: %s", url, exc)
        return 0, b""


def http_delete(url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers or {}, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError) as exc:
        log.debug("http_delete error %s: %s", url, exc)
        return 0, b""


# ---------------------------------------------------------------------------
# EmergencyStop class
# ---------------------------------------------------------------------------

class EmergencyStop:
    """
    Performs emergency trading halt operations.

    All methods are designed to be as robust as possible -- they continue
    even if individual API calls fail and record all actions.
    """

    def __init__(self, base_url: str = COORDINATOR_BASE) -> None:
        self.base_url = base_url
        self._actions_taken: List[Dict[str, Any]] = []
        self._start_time = datetime.utcnow()

    def _record(self, action: str, success: bool, detail: str = "") -> None:
        self._actions_taken.append({
            "ts": datetime.utcnow().isoformat(),
            "action": action,
            "success": success,
            "detail": detail,
        })

    # ------------------------------------------------------------------
    # halt_all_trading
    # ------------------------------------------------------------------

    def halt_all_trading(self) -> bool:
        """
        Immediately disable trading and block all new orders at the coordinator.
        Returns True if the halt was confirmed, False if it could not be verified.
        """
        log.critical("!!! HALTING ALL TRADING !!!")
        payload = {
            "mode": "emergency_halt",
            "enabled": False,
            "reason": "emergency_stop_script",
            "timestamp": datetime.utcnow().isoformat(),
        }
        status, body = http_post(f"{self.base_url}/trading/halt", payload)
        if status in (200, 204):
            log.critical("Trading HALTED via coordinator.")
            self._record("halt_all_trading", True, f"HTTP {status}")
            return True

        # Try fallback endpoint
        status2, body2 = http_post(f"{self.base_url}/trading/mode", {"mode": "disabled", "enabled": False})
        if status2 in (200, 204):
            log.critical("Trading DISABLED via fallback mode endpoint.")
            self._record("halt_all_trading", True, f"fallback HTTP {status2}")
            return True

        log.error("halt_all_trading: both halt (%d) and mode (%d) endpoints failed", status, status2)
        self._record("halt_all_trading", False, f"halt={status} mode={status2}")
        return False

    # ------------------------------------------------------------------
    # cancel_pending_orders
    # ------------------------------------------------------------------

    def cancel_pending_orders(self, broker: str) -> bool:
        """
        Cancel all pending orders for the specified broker.
        broker: "alpaca" or "binance"
        """
        log.critical("Cancelling all pending orders on broker: %s", broker)
        broker_lower = broker.lower()

        if broker_lower == "alpaca":
            return self._cancel_alpaca_orders()
        elif broker_lower == "binance":
            return self._cancel_binance_orders()
        elif broker_lower == "all":
            a = self._cancel_alpaca_orders()
            b = self._cancel_binance_orders()
            return a and b
        else:
            # Try coordinator generic endpoint
            status, body = http_delete(f"{self.base_url}/orders/all?broker={broker}")
            ok = status in (200, 204)
            self._record("cancel_pending_orders", ok, f"broker={broker} HTTP {status}")
            if ok:
                log.critical("All pending orders cancelled via coordinator (broker=%s).", broker)
            else:
                log.error("Order cancellation via coordinator failed: HTTP %d", status)
            return ok

    def _cancel_alpaca_orders(self) -> bool:
        if not ALPACA_KEY:
            log.warning("Alpaca API keys not set -- cannot cancel Alpaca orders directly")
            return False
        url = f"{ALPACA_BASE}/v2/orders"
        headers = {
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        }
        status, body = http_delete(url, headers=headers)
        ok = status in (200, 204, 207)
        self._record("cancel_alpaca_orders", ok, f"HTTP {status}")
        if ok:
            log.critical("All Alpaca orders cancelled (HTTP %d).", status)
        else:
            log.error("Alpaca order cancellation failed: HTTP %d -- %s", status, body[:200])
        return ok

    def _cancel_binance_orders(self) -> bool:
        """Cancel open orders on Binance via coordinator proxy (avoids HMAC signing complexity)."""
        status, body = http_delete(f"{self.base_url}/orders/all?broker=binance")
        ok = status in (200, 204)
        self._record("cancel_binance_orders", ok, f"HTTP {status}")
        if ok:
            log.critical("Binance orders cancel request submitted (HTTP %d).", status)
        else:
            log.error("Binance order cancellation failed: HTTP %d -- %s", status, body[:200])
        return ok

    # ------------------------------------------------------------------
    # flatten_positions
    # ------------------------------------------------------------------

    def flatten_positions(self, broker: str, market_order: bool = True) -> bool:
        """
        Close all open positions for the given broker.
        Uses market orders by default for immediate execution.
        """
        order_type = "market" if market_order else "limit"
        log.critical("Flattening all positions (broker=%s, order_type=%s)...", broker, order_type)

        if broker.lower() == "alpaca":
            return self._flatten_alpaca(market_order)

        # Generic coordinator flatten
        payload = {
            "broker": broker,
            "method": order_type,
            "timestamp": datetime.utcnow().isoformat(),
        }
        status, body = http_post(f"{self.base_url}/positions/flatten", payload)
        ok = status in (200, 204)
        self._record("flatten_positions", ok, f"broker={broker} method={order_type} HTTP {status}")
        if ok:
            log.critical("Flatten request submitted for broker=%s (HTTP %d).", broker, status)
        else:
            log.error("Flatten failed for broker=%s: HTTP %d -- %s", broker, status, body[:200])
        return ok

    def _flatten_alpaca(self, market_order: bool) -> bool:
        if not ALPACA_KEY:
            log.warning("Alpaca API keys not set -- using coordinator proxy")
            return self.flatten_positions("alpaca_proxy", market_order)
        url = f"{ALPACA_BASE}/v2/positions"
        headers = {
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        }
        status, body = http_delete(url, headers=headers)
        ok = status in (200, 204, 207)
        self._record("flatten_alpaca", ok, f"HTTP {status}")
        if ok:
            log.critical("All Alpaca positions submitted for closure (HTTP %d).", status)
            try:
                closed = json.loads(body)
                if isinstance(closed, list):
                    log.critical("  Closing %d positions.", len(closed))
            except (json.JSONDecodeError, TypeError):
                pass
        else:
            log.error("Alpaca flatten failed: HTTP %d -- %s", status, body[:200])
        return ok

    # ------------------------------------------------------------------
    # send_emergency_alert
    # ------------------------------------------------------------------

    def send_emergency_alert(self, reason: str) -> None:
        """Send immediate Slack and PagerDuty alerts."""
        log.critical("Sending emergency alerts (reason: %s)...", reason)
        self._alert_slack(reason)
        self._alert_pagerduty(reason)

    def _alert_slack(self, reason: str) -> None:
        if not SLACK_WEBHOOK:
            log.warning("SLACK_WEBHOOK not set -- Slack alert skipped")
            return
        ts = datetime.utcnow().isoformat()
        text = (
            f":rotating_light: *SRFM EMERGENCY STOP* :rotating_light:\n"
            f"*Reason:* `{reason}`\n"
            f"*Time:* {ts}Z\n"
            f"Trading has been HALTED. Investigate immediately."
        )
        payload = {"text": text, "username": "SRFM-EmergencyBot"}
        try:
            status, _ = http_post(SLACK_WEBHOOK, payload)
            if status in (200, 204):
                log.critical("Slack emergency alert sent.")
                self._record("slack_alert", True, f"HTTP {status}")
            else:
                log.error("Slack alert failed: HTTP %d", status)
                self._record("slack_alert", False, f"HTTP {status}")
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Slack alert exception: %s", exc)
            self._record("slack_alert", False, str(exc))

    def _alert_pagerduty(self, reason: str) -> None:
        if not PAGERDUTY_KEY:
            log.warning("PAGERDUTY_KEY not set -- PagerDuty alert skipped")
            return
        url = "https://events.pagerduty.com/v2/enqueue"
        payload = {
            "routing_key": PAGERDUTY_KEY,
            "event_action": "trigger",
            "payload": {
                "summary": f"SRFM Emergency Stop: {reason}",
                "severity": "critical",
                "source": "srfm-emergency-stop",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "custom_details": {"reason": reason},
            },
        }
        try:
            status, body = http_post(url, payload)
            if status in (200, 202):
                log.critical("PagerDuty alert sent.")
                self._record("pagerduty_alert", True, f"HTTP {status}")
            else:
                log.error("PagerDuty alert failed: HTTP %d -- %s", status, body[:100])
                self._record("pagerduty_alert", False, f"HTTP {status}")
        except Exception as exc:  # pylint: disable=broad-except
            log.error("PagerDuty alert exception: %s", exc)
            self._record("pagerduty_alert", False, str(exc))

    # ------------------------------------------------------------------
    # log_incident
    # ------------------------------------------------------------------

    def log_incident(self, reason: str, context: Dict) -> Path:
        """
        Write a structured incident log to logs/incidents/YYYYMMDD_HHMMSS_emergency.json.
        Returns the path of the written file.
        """
        ts_str = self._start_time.strftime("%Y%m%d_%H%M%S")
        incident_path = INCIDENTS_DIR / f"{ts_str}_emergency.json"
        incident = {
            "incident_type": "emergency_stop",
            "reason": reason,
            "started_at": self._start_time.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "context": context,
            "actions_taken": self._actions_taken,
            "coordinator_url": self.base_url,
        }
        try:
            with open(incident_path, "w", encoding="utf-8") as f:
                json.dump(incident, f, indent=2)
            log.critical("Incident log written: %s", incident_path)
        except OSError as exc:
            log.error("Failed to write incident log: %s", exc)
        return incident_path

    # ------------------------------------------------------------------
    # full_stop -- convenience method combining all steps
    # ------------------------------------------------------------------

    def full_stop(
        self,
        reason: str,
        flatten: bool = False,
        cancel_orders: bool = False,
        brokers: Optional[List[str]] = None,
    ) -> bool:
        """Run the full emergency stop sequence."""
        _brokers = brokers or ["alpaca", "binance"]
        context: Dict[str, Any] = {
            "reason": reason,
            "flatten": flatten,
            "cancel_orders": cancel_orders,
            "brokers": _brokers,
        }

        # Alert first -- before any actions that might fail
        self.send_emergency_alert(reason)

        # Halt trading
        halt_ok = self.halt_all_trading()

        # Cancel orders if requested
        if cancel_orders:
            for broker in _brokers:
                self.cancel_pending_orders(broker)

        # Flatten if requested
        if flatten:
            for broker in _brokers:
                self.flatten_positions(broker, market_order=True)

        # Write incident log
        self.log_incident(reason, context)

        return halt_ok


# ---------------------------------------------------------------------------
# CLI confirmation helpers
# ---------------------------------------------------------------------------

def _require_confirmation(reason: str, flatten: bool, cancel_orders: bool) -> bool:
    """Prompt user to confirm the emergency stop action."""
    print("\n" + "=" * 60)
    print("!!! SRFM EMERGENCY STOP !!!")
    print(f"Reason       : {reason}")
    print(f"Cancel orders: {cancel_orders}")
    print(f"Flatten pos  : {flatten}")
    print("=" * 60)
    print("This will IMMEDIATELY halt all trading.")
    if flatten:
        print("WARNING: All positions will be closed via MARKET ORDERS.")
    ans = input("\nType 'CONFIRM' to proceed: ").strip()
    return ans == "CONFIRM"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SRFM emergency trading halt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reason",
        type=str,
        required=True,
        help="Reason for emergency stop (e.g. drawdown_breach)",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Close all open positions via market orders",
    )
    parser.add_argument(
        "--cancel-orders",
        action="store_true",
        dest="cancel_orders",
        help="Cancel all pending orders",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip interactive confirmation prompt",
    )
    parser.add_argument(
        "--brokers",
        type=str,
        default="alpaca,binance",
        help="Comma-separated list of brokers to act on (default: alpaca,binance)",
    )
    parser.add_argument(
        "--coordinator-url",
        type=str,
        default=COORDINATOR_BASE,
        help=f"Coordinator base URL (default: {COORDINATOR_BASE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    brokers = [b.strip().lower() for b in args.brokers.split(",") if b.strip()]

    if not args.confirm:
        confirmed = _require_confirmation(args.reason, args.flatten, args.cancel_orders)
        if not confirmed:
            print("Emergency stop CANCELLED by operator.")
            sys.exit(0)

    log.critical("Emergency stop CONFIRMED -- executing sequence...")
    es = EmergencyStop(base_url=args.coordinator_url)
    success = es.full_stop(
        reason=args.reason,
        flatten=args.flatten,
        cancel_orders=args.cancel_orders,
        brokers=brokers,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
