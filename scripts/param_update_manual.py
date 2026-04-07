#!/usr/bin/env python3
"""
param_update_manual.py -- Manual parameter update tool for SRFM.

Allows an operator to update live system parameters with validation,
diff display, and audit logging. All updates go through the coordination
layer's propose endpoint.

Usage:
  python scripts/param_update_manual.py --param BH_MASS_THRESH --value 2.1 --reason "calibration"
  python scripts/param_update_manual.py --file params.yaml --reason "weekly refit"
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
AUDIT_LOG = LOGS_DIR / "param_audit.jsonl"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s -- %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


log = build_logger("srfm.param_update")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COORDINATOR_BASE = os.environ.get("SRFM_COORDINATOR_URL", "http://localhost:8000")
HTTP_TIMEOUT = 10

# ANSI color codes
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _colored(text: str, color: str) -> str:
    if _supports_color():
        return f"{color}{text}{_RESET}"
    return text


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get(url: str) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError) as exc:
        log.debug("http_get %s: %s", url, exc)
        return 0, b""


def http_post(url: str, payload: Dict) -> Tuple[int, bytes]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError) as exc:
        log.debug("http_post %s: %s", url, exc)
        return 0, b""


# ---------------------------------------------------------------------------
# Diff display
# ---------------------------------------------------------------------------

def _coerce_numeric(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def display_diff(param: str, old_val: Any, new_val: Any) -> None:
    """Print a colored diff for a single parameter change."""
    old_num = _coerce_numeric(old_val)
    new_num = _coerce_numeric(new_val)

    if old_num is not None and new_num is not None:
        if new_num > old_num:
            arrow = _colored(f"{old_val} -> {new_val} (+{new_num - old_num:.4g})", _GREEN)
        elif new_num < old_num:
            arrow = _colored(f"{old_val} -> {new_val} ({new_num - old_num:.4g})", _RED)
        else:
            arrow = _colored(f"{old_val} -> {new_val} (unchanged)", _YELLOW)
    else:
        arrow = _colored(f'"{old_val}" -> "{new_val}"', _YELLOW)

    print(f"  {_colored(_BOLD + param, _BOLD):<35} {arrow}")


def display_diff_table(current_params: Dict, proposed_params: Dict) -> None:
    """Print a full diff table for a set of proposed parameter changes."""
    print()
    print(_colored("Parameter Diff", _BOLD))
    print("-" * 60)
    for param, new_val in proposed_params.items():
        old_val = current_params.get(param, "<not set>")
        display_diff(param, old_val, new_val)
    # Show params not in proposed (will remain unchanged)
    unchanged = [k for k in current_params if k not in proposed_params]
    if unchanged:
        print(f"  [+{len(unchanged)} parameters unchanged]")
    print("-" * 60)
    print()


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def write_audit_entry(
    operator: str,
    params: Dict,
    reason: str,
    accepted: bool,
    response_detail: str,
) -> None:
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "operator": operator,
        "params": params,
        "reason": reason,
        "accepted": accepted,
        "response": response_detail,
    }
    try:
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        log.warning("Could not write audit log: %s", exc)


# ---------------------------------------------------------------------------
# ManualParamUpdater
# ---------------------------------------------------------------------------

class ManualParamUpdater:
    """
    Proposes and submits manual parameter updates to the SRFM coordination layer.
    Handles diff display, validation, and audit logging.
    """

    def __init__(self, base_url: str = COORDINATOR_BASE) -> None:
        self.base_url = base_url

    def get_current_params(self) -> Optional[Dict]:
        """Fetch current live parameters from coordinator."""
        status, body = http_get(f"{self.base_url}/params/current")
        if status == 200:
            try:
                data = json.loads(body)
                return data.get("params", data)
            except (json.JSONDecodeError, TypeError):
                pass
        elif status == 404:
            log.info("Params endpoint not found -- using empty baseline")
            return {}
        log.warning("Could not fetch current params (HTTP %d)", status)
        return None

    def validate_with_coordinator(self, params: Dict) -> Tuple[bool, List[str]]:
        """
        Ask the coordinator to validate proposed params.
        Returns (valid, list_of_error_strings).
        """
        status, body = http_post(f"{self.base_url}/params/validate", {"params": params})
        if status == 200:
            try:
                data = json.loads(body)
                errors = data.get("errors", [])
                warnings = data.get("warnings", [])
                for w in warnings:
                    log.warning("  Validation warning: %s", w)
                return len(errors) == 0, errors
            except (json.JSONDecodeError, TypeError):
                return True, []
        elif status in (400, 422):
            try:
                data = json.loads(body)
                errors = data.get("errors", [str(body[:200])])
                return False, errors
            except (json.JSONDecodeError, TypeError):
                return False, [f"Validation HTTP {status}"]
        log.warning("Validation endpoint returned HTTP %d -- assuming valid", status)
        return True, []

    def propose(self, params: Dict, reason: str) -> bool:
        """
        Propose a parameter update.

        1. Fetch current params and display diff.
        2. Validate with coordinator.
        3. Prompt for confirmation.
        4. POST to /params/propose.
        5. Write audit entry.

        Returns True if the proposal was accepted.
        """
        log.info("Fetching current parameters...")
        current = self.get_current_params()
        if current is None:
            log.error("Could not retrieve current parameters -- aborting")
            return False

        # Show diff
        display_diff_table(current, params)

        # Validate
        log.info("Validating proposed parameters...")
        valid, errors = self.validate_with_coordinator(params)
        if not valid:
            print(_colored("Validation FAILED:", _RED))
            for err in errors:
                print(f"  - {err}")
            return False

        print(_colored("Validation passed.", _GREEN))
        print(f"Reason: {reason}")

        # Confirmation
        ans = input("\nSubmit this parameter update? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Update cancelled.")
            write_audit_entry(
                operator=os.environ.get("USER", "unknown"),
                params=params,
                reason=reason,
                accepted=False,
                response_detail="operator_cancelled",
            )
            return False

        # Submit
        payload = {
            "params": params,
            "reason": reason,
            "operator": os.environ.get("USER", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
        }
        status, body = http_post(f"{self.base_url}/params/propose", payload)
        if status in (200, 201, 202, 204):
            try:
                resp_data = json.loads(body)
                detail = resp_data.get("message", f"HTTP {status}")
            except (json.JSONDecodeError, TypeError):
                detail = f"HTTP {status}"
            log.info("Parameter update ACCEPTED: %s", detail)
            write_audit_entry(
                operator=os.environ.get("USER", "unknown"),
                params=params,
                reason=reason,
                accepted=True,
                response_detail=detail,
            )
            return True
        else:
            try:
                resp_data = json.loads(body)
                detail = resp_data.get("detail", str(body[:200]))
            except (json.JSONDecodeError, TypeError):
                detail = body[:200].decode("utf-8", errors="replace")
            log.error("Parameter update REJECTED (HTTP %d): %s", status, detail)
            write_audit_entry(
                operator=os.environ.get("USER", "unknown"),
                params=params,
                reason=reason,
                accepted=False,
                response_detail=f"HTTP {status}: {detail}",
            )
            return False


# ---------------------------------------------------------------------------
# YAML loader (minimal, avoids PyYAML dep)
# ---------------------------------------------------------------------------

def _load_yaml_file(path: Path) -> Dict:
    """
    Load a simple flat or nested YAML file.
    Uses PyYAML if available, otherwise raises ImportError with a hint.
    """
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {"_data": data}
    except ImportError:
        raise ImportError(
            "PyYAML is required for --file mode. Install with: pip install pyyaml"
        ) from None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SRFM manual parameter update tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--param",
        type=str,
        help="Single parameter name to update",
    )
    group.add_argument(
        "--file",
        type=str,
        dest="yaml_file",
        help="YAML file containing parameter name -> value mapping",
    )
    parser.add_argument(
        "--value",
        type=str,
        help="New value for --param (required with --param)",
    )
    parser.add_argument(
        "--reason",
        type=str,
        required=True,
        help="Reason for the parameter update (required)",
    )
    parser.add_argument(
        "--coordinator-url",
        type=str,
        default=COORDINATOR_BASE,
        help=f"Coordinator base URL (default: {COORDINATOR_BASE})",
    )
    return parser.parse_args()


def _parse_value(raw: str) -> Any:
    """Try to coerce a string value to int, float, bool, or leave as string."""
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def main() -> None:
    args = parse_args()

    if args.param and not args.value:
        print("Error: --value is required when using --param", file=sys.stderr)
        sys.exit(2)

    updater = ManualParamUpdater(base_url=args.coordinator_url)

    if args.param:
        params = {args.param: _parse_value(args.value)}
    else:
        yaml_path = Path(args.yaml_file)
        if not yaml_path.exists():
            print(f"Error: YAML file not found: {yaml_path}", file=sys.stderr)
            sys.exit(2)
        try:
            params = _load_yaml_file(yaml_path)
        except ImportError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(2)

    success = updater.propose(params, reason=args.reason)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
