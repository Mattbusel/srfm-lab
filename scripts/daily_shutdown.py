#!/usr/bin/env python3
"""
daily_shutdown.py -- SRFM end-of-day trading shutdown script.

Runs the full EOD sequence:
  1. Stop accepting new signals (pause trading)
  2. Wait for pending orders to complete (max 10 min)
  3. Compute EOD P&L and performance metrics
  4. Run position reconciliation with broker
  5. Save performance snapshot to database
  6. Generate EOD report (HTML + JSON)
  7. Checkpoint database and run WAL vacuum
  8. Backup database to backups/ directory
  9. Send EOD summary to Slack
  10. Disable trading mode

Handles SIGINT/SIGTERM gracefully.
"""

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
BACKUPS_DIR = REPO_ROOT / "backups"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIG_DIR = REPO_ROOT / "config"

for _d in (LOGS_DIR, BACKUPS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
TODAY_STR = date.today().strftime("%Y%m%d")
LOG_FILE = LOGS_DIR / f"shutdown_{TODAY_STR}.log"


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
    fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = build_logger("srfm.shutdown")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COORDINATOR_BASE = os.environ.get("SRFM_COORDINATOR_URL", "http://localhost:8000")
SLACK_WEBHOOK = os.environ.get("SRFM_SLACK_WEBHOOK", "")
HTTP_TIMEOUT = 10
ORDER_WAIT_MAX_SECONDS = 600  # 10 minutes
ORDER_POLL_INTERVAL = 10      # seconds between polls

# ---------------------------------------------------------------------------
# Shutdown state
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _signal_handler(signum: int, frame) -> None:
    global _shutdown_requested  # noqa: PLW0603
    sig_name = signal.Signals(signum).name
    log.warning("Received signal %s -- finishing current step and shutting down cleanly...", sig_name)
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


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
    except urllib.error.URLError as exc:
        log.debug("URLError for %s: %s", url, exc.reason)
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
    except urllib.error.URLError as exc:
        log.debug("URLError for %s: %s", url, exc.reason)
        return 0, b""


# ---------------------------------------------------------------------------
# Step 1 -- Pause trading
# ---------------------------------------------------------------------------

def pause_trading(base_url: str) -> bool:
    log.info("=== Step 1: Pausing Trading Mode ===")
    status, body = http_post(f"{base_url}/trading/mode", {"mode": "paused", "enabled": False})
    if status in (200, 204):
        log.info("Trading paused -- no new signals will be accepted.")
        return True
    log.error("Failed to pause trading -- HTTP %d -- %s", status, body[:200])
    return False


# ---------------------------------------------------------------------------
# Step 2 -- Wait for pending orders
# ---------------------------------------------------------------------------

def wait_for_pending_orders(base_url: str) -> bool:
    log.info("=== Step 2: Waiting for Pending Orders (max %ds) ===", ORDER_WAIT_MAX_SECONDS)
    deadline = time.monotonic() + ORDER_WAIT_MAX_SECONDS
    elapsed = 0
    while time.monotonic() < deadline:
        if _shutdown_requested:
            log.warning("Signal received during order wait -- proceeding with shutdown")
            return False
        status, body = http_get(f"{base_url}/orders/pending")
        if status == 200:
            try:
                data = json.loads(body)
                count = data.get("count", 0)
                log.info("Pending orders: %d (elapsed: %ds)", count, int(elapsed))
                if count == 0:
                    log.info("All orders settled.")
                    return True
            except (json.JSONDecodeError, TypeError):
                pass
        elif status == 404:
            log.info("No pending orders endpoint -- assuming clear.")
            return True
        time.sleep(ORDER_POLL_INTERVAL)
        elapsed += ORDER_POLL_INTERVAL

    log.warning("Timeout waiting for orders -- proceeding with EOD sequence")
    return False


# ---------------------------------------------------------------------------
# Step 3 -- Compute EOD P&L
# ---------------------------------------------------------------------------

def compute_eod_pnl(base_url: str) -> Optional[Dict]:
    log.info("=== Step 3: Computing EOD P&L and Performance Metrics ===")
    status, body = http_get(f"{base_url}/analytics/eod_pnl")
    if status == 200:
        try:
            data = json.loads(body)
            pnl = data.get("total_pnl", 0.0)
            sharpe = data.get("sharpe_today", None)
            drawdown = data.get("max_drawdown_today", None)
            log.info("  Total P&L: $%.2f", pnl)
            if sharpe is not None:
                log.info("  Sharpe (today): %.3f", sharpe)
            if drawdown is not None:
                log.info("  Max drawdown (today): %.2f%%", drawdown * 100)
            return data
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning("Could not parse EOD P&L response: %s", exc)
    else:
        log.warning("EOD P&L endpoint returned HTTP %d", status)
    return None


# ---------------------------------------------------------------------------
# Step 4 -- Position reconciliation
# ---------------------------------------------------------------------------

def reconcile_positions(base_url: str) -> bool:
    log.info("=== Step 4: EOD Position Reconciliation ===")
    status, body = http_get(f"{base_url}/positions/reconcile")
    if status == 200:
        try:
            data = json.loads(body)
            discreps = data.get("discrepancies", [])
            matched = data.get("matched", 0)
            log.info("  Positions matched: %d | discrepancies: %d", matched, len(discreps))
            for d in discreps:
                log.warning("  Discrepancy: %s", d)
            if discreps:
                log.warning("Discrepancies found -- recorded in log, will require manual review")
        except (json.JSONDecodeError, TypeError):
            pass
        return True
    elif status == 404:
        log.info("No positions to reconcile (flat book).")
        return True
    log.error("Position reconciliation endpoint returned HTTP %d", status)
    return False


# ---------------------------------------------------------------------------
# Step 5 -- Save performance snapshot
# ---------------------------------------------------------------------------

def save_performance_snapshot(base_url: str, pnl_data: Optional[Dict]) -> bool:
    log.info("=== Step 5: Saving Performance Snapshot ===")
    payload: Dict = {
        "date": date.today().isoformat(),
        "timestamp": datetime.utcnow().isoformat(),
        "pnl_data": pnl_data or {},
    }
    status, body = http_post(f"{base_url}/analytics/snapshot", payload)
    if status in (200, 201, 204):
        log.info("Performance snapshot saved to database.")
        return True
    log.warning("Snapshot endpoint returned HTTP %d -- %s", status, body[:100])
    return False


# ---------------------------------------------------------------------------
# Step 6 -- Generate EOD report
# ---------------------------------------------------------------------------

def _build_html_report(pnl_data: Optional[Dict], date_str: str) -> str:
    pnl = pnl_data.get("total_pnl", 0.0) if pnl_data else 0.0
    sharpe = pnl_data.get("sharpe_today", "N/A") if pnl_data else "N/A"
    drawdown = pnl_data.get("max_drawdown_today", "N/A") if pnl_data else "N/A"
    trades = pnl_data.get("num_trades", "N/A") if pnl_data else "N/A"
    color = "#27ae60" if pnl >= 0 else "#e74c3c"
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>SRFM EOD Report {date_str}</title>
  <style>
    body {{ font-family: monospace; background: #1a1a2e; color: #eee; padding: 2em; }}
    h1 {{ color: #00d4aa; }}
    .metric {{ margin: 0.4em 0; }}
    .val {{ font-weight: bold; color: {color}; }}
    table {{ border-collapse: collapse; margin-top: 1em; }}
    td, th {{ border: 1px solid #555; padding: 6px 14px; }}
    th {{ background: #2d2d44; }}
  </style>
</head>
<body>
  <h1>SRFM End-of-Day Report</h1>
  <p>Date: {date_str}</p>
  <div class="metric">Total P&amp;L: <span class="val">${pnl:.2f}</span></div>
  <div class="metric">Sharpe (today): <span class="val">{sharpe}</span></div>
  <div class="metric">Max Drawdown: <span class="val">{drawdown}</span></div>
  <div class="metric">Num Trades: <span class="val">{trades}</span></div>
  <p><em>Generated: {datetime.utcnow().isoformat()}Z</em></p>
</body>
</html>"""
    return html


def generate_eod_report(pnl_data: Optional[Dict]) -> bool:
    log.info("=== Step 6: Generating EOD Report ===")
    date_str = date.today().isoformat()
    # JSON report
    json_path = REPORTS_DIR / f"eod_{TODAY_STR}.json"
    report_payload = {
        "date": date_str,
        "generated_at": datetime.utcnow().isoformat(),
        "pnl": pnl_data or {},
    }
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        log.info("  JSON report: %s", json_path)
    except OSError as exc:
        log.error("Failed to write JSON report: %s", exc)
        return False

    # HTML report
    html_path = REPORTS_DIR / f"eod_{TODAY_STR}.html"
    html_content = _build_html_report(pnl_data, date_str)
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        log.info("  HTML report: %s", html_path)
    except OSError as exc:
        log.error("Failed to write HTML report: %s", exc)
        return False

    log.info("EOD reports generated.")
    return True


# ---------------------------------------------------------------------------
# Step 7 -- Database checkpoint and WAL vacuum
# ---------------------------------------------------------------------------

def checkpoint_database(base_url: str) -> bool:
    log.info("=== Step 7: Database Checkpoint and WAL Vacuum ===")
    status, body = http_post(f"{base_url}/db/checkpoint", {"vacuum_wal": True})
    if status in (200, 204):
        log.info("Database checkpoint and WAL vacuum complete.")
        return True
    log.warning("DB checkpoint returned HTTP %d -- %s", status, body[:100])
    return False


# ---------------------------------------------------------------------------
# Step 8 -- Backup database
# ---------------------------------------------------------------------------

def backup_database() -> bool:
    log.info("=== Step 8: Backing Up Database ===")
    # Locate DuckDB file
    db_candidates = [
        REPO_ROOT / "data" / "srfm.duckdb",
        REPO_ROOT / "srfm.duckdb",
        REPO_ROOT / "db" / "srfm.duckdb",
    ]
    db_path: Optional[Path] = None
    for candidate in db_candidates:
        if candidate.exists():
            db_path = candidate
            break

    if db_path is None:
        log.warning("No DuckDB file found in standard locations -- skipping backup")
        return True  # Non-fatal if DB file doesn't exist yet

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUPS_DIR / f"srfm_{ts}.duckdb"
    try:
        shutil.copy2(str(db_path), str(backup_path))
        size_mb = backup_path.stat().st_size / (1024 * 1024)
        log.info("Database backed up to %s (%.1f MB)", backup_path, size_mb)
        # Prune backups older than 30 days
        _prune_old_backups()
        return True
    except (OSError, shutil.Error) as exc:
        log.error("Database backup failed: %s", exc)
        return False


def _prune_old_backups(max_days: int = 30) -> None:
    cutoff = time.time() - (max_days * 86400)
    removed = 0
    for f in BACKUPS_DIR.glob("srfm_*.duckdb"):
        if f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        log.info("Pruned %d old backup(s) (>%d days)", removed, max_days)


# ---------------------------------------------------------------------------
# Step 9 -- Slack notification
# ---------------------------------------------------------------------------

def send_eod_notification(pnl_data: Optional[Dict], duration_s: float) -> None:
    log.info("=== Step 9: Slack EOD Notification ===")
    if not SLACK_WEBHOOK:
        log.info("SLACK_WEBHOOK not configured -- skipping")
        return
    pnl = pnl_data.get("total_pnl", 0.0) if pnl_data else 0.0
    sharpe = pnl_data.get("sharpe_today", "N/A") if pnl_data else "N/A"
    trades = pnl_data.get("num_trades", "N/A") if pnl_data else "N/A"
    emoji = ":chart_with_upwards_trend:" if pnl >= 0 else ":chart_with_downwards_trend:"
    text = (
        f"{emoji} *SRFM EOD Shutdown Complete*\n"
        f"Date: {date.today().isoformat()} | P&L: `${pnl:.2f}` | "
        f"Sharpe: `{sharpe}` | Trades: `{trades}`\n"
        f"Shutdown duration: {duration_s:.1f}s"
    )
    payload = {"text": text}
    try:
        status, _ = http_post(SLACK_WEBHOOK, payload)
        if status in (200, 204):
            log.info("Slack EOD notification sent.")
        else:
            log.warning("Slack notification returned HTTP %d", status)
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("Slack notification failed: %s", exc)


# ---------------------------------------------------------------------------
# Step 10 -- Disable trading
# ---------------------------------------------------------------------------

def disable_trading(base_url: str) -> bool:
    log.info("=== Step 10: Disabling Trading Mode ===")
    status, body = http_post(f"{base_url}/trading/mode", {"mode": "disabled", "enabled": False})
    if status in (200, 204):
        log.info("Trading mode DISABLED.")
        return True
    log.error("Failed to disable trading mode -- HTTP %d -- %s", status, body[:200])
    return False


# ---------------------------------------------------------------------------
# Position flattener
# ---------------------------------------------------------------------------

def flatten_all_positions(base_url: str) -> bool:
    log.info("[FLATTEN] Closing all open positions via market orders...")
    status, body = http_post(f"{base_url}/positions/flatten", {"method": "market"})
    if status in (200, 204):
        log.info("[FLATTEN] All positions submitted for closure.")
        return True
    log.error("[FLATTEN] Flatten endpoint returned HTTP %d -- %s", status, body[:200])
    return False


# ---------------------------------------------------------------------------
# Main shutdown orchestrator
# ---------------------------------------------------------------------------

class ShutdownOrchestrator:
    def __init__(self, flatten: bool) -> None:
        self.flatten = flatten
        self.base_url = COORDINATOR_BASE

    def run(self) -> int:
        start_time = time.monotonic()
        log.info("=" * 60)
        log.info("SRFM End-of-Day Shutdown -- %s", datetime.now().isoformat())
        log.info("Flatten positions: %s", self.flatten)
        log.info("=" * 60)

        # Step 1 -- pause
        if not pause_trading(self.base_url):
            log.error("Could not pause trading -- proceeding anyway")

        # Optional pre-step: flatten positions
        if self.flatten:
            flatten_all_positions(self.base_url)

        # Step 2 -- wait for orders
        wait_for_pending_orders(self.base_url)
        if _shutdown_requested:
            log.warning("Emergency shutdown -- skipping analytics steps")
            disable_trading(self.base_url)
            return 1

        # Step 3 -- compute P&L
        pnl_data = compute_eod_pnl(self.base_url)

        # Step 4 -- reconcile
        reconcile_positions(self.base_url)

        # Step 5 -- save snapshot
        save_performance_snapshot(self.base_url, pnl_data)

        # Step 6 -- generate report
        generate_eod_report(pnl_data)

        # Step 7 -- checkpoint
        checkpoint_database(self.base_url)

        # Step 8 -- backup
        backup_database()

        duration = time.monotonic() - start_time

        # Step 9 -- Slack
        send_eod_notification(pnl_data, duration)

        # Step 10 -- disable
        disable_trading(self.base_url)

        log.info("=" * 60)
        log.info("SRFM SHUTDOWN COMPLETE -- %.1fs", duration)
        log.info("=" * 60)
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SRFM end-of-day shutdown script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten (close) all open positions before shutdown",
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

    orchestrator = ShutdownOrchestrator(flatten=args.flatten)
    sys.exit(orchestrator.run())


if __name__ == "__main__":
    main()
