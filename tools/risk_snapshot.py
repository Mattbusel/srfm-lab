"""
risk_snapshot.py -- Quick risk snapshot tool for SRFM.

Fetches all risk metrics and prints a formatted Rich report, then optionally
saves a JSON snapshot to logs/risk_snapshot_YYYYMMDD_HHMMSS.json.

Usage:
    python tools/risk_snapshot.py
    python tools/risk_snapshot.py --format table
    python tools/risk_snapshot.py --format json
    python tools/risk_snapshot.py --format compact
    python tools/risk_snapshot.py --no-save
    python tools/risk_snapshot.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text
from rich import box

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RISK_AGG_BASE        = "http://risk-aggregator:8783"
LIVE_TRADER_BASE     = "http://live-trader:8080"
COORDINATION_BASE    = "http://coordination:8781"
REQUEST_TIMEOUT      = 3.0

console = Console()


# ---------------------------------------------------------------------------
# Raw fetch helpers
# ---------------------------------------------------------------------------

def _get(url: str) -> Optional[dict]:
    """GET url and return parsed JSON dict, or None on failure."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return None


def fetch_risk_metrics() -> dict:
    data = _get(f"{RISK_AGG_BASE}/metrics") or {}
    return data


def fetch_positions() -> list:
    data = _get(f"{LIVE_TRADER_BASE}/positions") or {}
    return data.get("positions", [])


def fetch_position_limits() -> list:
    data = _get(f"{RISK_AGG_BASE}/limits") or {}
    return data.get("limits", [])


def fetch_circuit_breakers() -> dict:
    data = _get(f"{COORDINATION_BASE}/circuit/all") or {}
    return data.get("breakers", {})


def fetch_alerts() -> list:
    data = _get(f"{RISK_AGG_BASE}/alerts?n=20") or {}
    return data.get("alerts", [])


def fetch_all() -> dict:
    """Fetch all data and return as a unified dict."""
    return {
        "risk_metrics":      fetch_risk_metrics(),
        "positions":         fetch_positions(),
        "position_limits":   fetch_position_limits(),
        "circuit_breakers":  fetch_circuit_breakers(),
        "alerts":            fetch_alerts(),
        "snapshot_time":     datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _util_style(pct: float) -> str:
    """Green < 50%, yellow 50-80%, red > 80%."""
    if pct >= 80:
        return "bright_red"
    if pct >= 50:
        return "yellow"
    return "bright_green"


def _pnl_style(v: float) -> str:
    return "bright_green" if v >= 0 else "bright_red"


def _pnl_str(v: float) -> str:
    if v >= 0:
        return f"+${v:,.0f}"
    return f"-${abs(v):,.0f}"


def _cb_style(state: str) -> str:
    return {"CLOSED": "bright_green", "OPEN": "bright_red",
            "HALF_OPEN": "yellow"}.get(state, "white")


def _alert_style(level: str) -> str:
    return {"BREACH": "bright_red bold", "WARN": "yellow", "INFO": "dim"}.get(level, "white")


# ---------------------------------------------------------------------------
# Inline progress bar (no live context needed)
# ---------------------------------------------------------------------------

def _pct_bar(pct: float, width: int = 24) -> Text:
    """Returns a Rich Text progress bar."""
    filled = int(min(pct, 100) / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    style = _util_style(pct)
    return Text(bar, style=style)


# ---------------------------------------------------------------------------
# RiskSnapshotPrinter
# ---------------------------------------------------------------------------

class RiskSnapshotPrinter:
    """Prints formatted risk snapshot reports using Rich."""

    def __init__(self, data: dict):
        self._data = data
        self._metrics = data.get("risk_metrics", {})
        self._positions = data.get("positions", [])
        self._limits = data.get("position_limits", [])
        self._breakers = data.get("circuit_breakers", {})
        self._alerts = data.get("alerts", [])
        self._snap_time = data.get("snapshot_time", datetime.now().isoformat())

    # -- Header --------------------------------------------------------------

    def _print_header(self) -> None:
        ts = self._snap_time[:19].replace("T", " ")
        console.rule(f"[bold bright_blue]SRFM Risk Snapshot[/bold bright_blue]  {ts}")

    # -- Risk metrics panel --------------------------------------------------

    def print_metrics(self) -> None:
        """Print NAV-level risk gauges."""
        m = self._metrics
        if not m:
            console.print(Panel("[dim]-- no risk metrics available --[/dim]",
                                title="Risk Metrics", border_style="bright_black"))
            return

        var_95      = float(m.get("var_95", 0))
        var_limit   = float(m.get("var_limit", 1))
        drawdown    = float(m.get("drawdown", 0))      # fraction 0-1
        dd_limit    = float(m.get("drawdown_limit", 0.15))
        margin_used = float(m.get("margin_used", 0))
        margin_lim  = float(m.get("margin_limit", 1))
        nav         = float(m.get("nav", 0))
        daily_pnl   = float(m.get("daily_pnl", 0))
        ytd_pnl     = float(m.get("ytd_pnl", 0))

        var_pct = var_95 / max(var_limit, 1) * 100
        dd_pct  = drawdown / max(dd_limit, 0.0001) * 100
        mar_pct = margin_used / max(margin_lim, 1) * 100

        lines = []
        lines.append(
            f"  [bold]NAV[/bold]          "
            f"[bright_green]${nav:>14,.0f}[/bright_green]"
        )
        lines.append(
            f"  [bold]Daily P&L[/bold]    "
            f"[{_pnl_style(daily_pnl)}]{_pnl_str(daily_pnl):>14}[/{_pnl_style(daily_pnl)}]"
        )
        lines.append(
            f"  [bold]YTD P&L[/bold]      "
            f"[{_pnl_style(ytd_pnl)}]{_pnl_str(ytd_pnl):>14}[/{_pnl_style(ytd_pnl)}]"
        )
        lines.append("")
        lines.append(
            f"  [bold]VaR 95%[/bold]      "
            f"[{_util_style(var_pct)}]{var_pct:5.1f}%[/{_util_style(var_pct)}]  "
            f"{_pct_bar(var_pct).markup}  "
            f"[dim]${var_95:,.0f} / ${var_limit:,.0f}[/dim]"
        )
        lines.append(
            f"  [bold]Drawdown[/bold]     "
            f"[{_util_style(dd_pct)}]{dd_pct:5.1f}%[/{_util_style(dd_pct)}]  "
            f"{_pct_bar(dd_pct).markup}  "
            f"[dim]{drawdown*100:.2f}% / {dd_limit*100:.1f}% limit[/dim]"
        )
        lines.append(
            f"  [bold]Margin[/bold]       "
            f"[{_util_style(mar_pct)}]{mar_pct:5.1f}%[/{_util_style(mar_pct)}]  "
            f"{_pct_bar(mar_pct).markup}  "
            f"[dim]${margin_used:,.0f} / ${margin_lim:,.0f}[/dim]"
        )

        console.print(Panel(
            "\n".join(lines),
            title="[bold bright_blue]RISK GAUGES[/bold bright_blue]",
            border_style="blue",
        ))

    # -- Positions -----------------------------------------------------------

    def print_positions(self) -> None:
        """Print current positions with P&L."""
        positions = self._positions
        if not positions:
            console.print(Panel("[dim]-- no open positions --[/dim]",
                                title="Positions", border_style="bright_black"))
            return

        table = Table(
            title="Open Positions",
            box=box.ROUNDED,
            header_style="bold bright_blue",
            border_style="blue",
        )
        table.add_column("Symbol", style="bold white", width=10)
        table.add_column("Qty", justify="right", width=10)
        table.add_column("Avg Cost", justify="right", width=10)
        table.add_column("Mark", justify="right", width=10)
        table.add_column("Unreal P&L", justify="right", width=12)
        table.add_column("%", justify="right", width=8)
        table.add_column("Daily P&L", justify="right", width=12)

        for p in positions:
            unreal = float(p.get("unreal_pnl", 0))
            pct    = float(p.get("pct_pnl", 0))
            daily  = float(p.get("daily_pnl", 0))

            table.add_row(
                str(p.get("symbol", "")),
                f"{p.get('qty', 0):.2f}",
                f"{p.get('avg_cost', 0):.4f}",
                f"{p.get('mark', 0):.4f}",
                Text(_pnl_str(unreal), style=_pnl_style(unreal)),
                Text(f"{pct:+.2f}%", style=_pnl_style(pct)),
                Text(_pnl_str(daily), style=_pnl_style(daily)),
            )

        console.print(table)

    # -- Position limits -----------------------------------------------------

    def print_limits(self) -> None:
        """Print limit utilization table."""
        limits = self._limits
        if not limits:
            console.print(Panel("[dim]-- no limit data available --[/dim]",
                                title="Position Limits", border_style="bright_black"))
            return

        table = Table(
            title="Position Limits",
            box=box.ROUNDED,
            header_style="bold bright_blue",
            border_style="blue",
        )
        table.add_column("Symbol", style="bold white", width=10)
        table.add_column("Position", justify="right", width=12)
        table.add_column("Limit", justify="right", width=12)
        table.add_column("Utilization", justify="right", width=10)
        table.add_column("Bar", width=26)

        for lim in limits:
            pos   = float(lim.get("position", 0))
            limit = float(lim.get("limit", 1))
            util  = float(lim.get("utilization", abs(pos) / max(limit, 1) * 100))
            sty   = _util_style(util)

            table.add_row(
                str(lim.get("symbol", "")),
                f"{abs(pos):,.0f}",
                f"{limit:,.0f}",
                Text(f"{util:.1f}%", style=sty),
                _pct_bar(util, width=20),
            )

        console.print(table)

    # -- Circuit breakers ----------------------------------------------------

    def print_circuit_breakers(self) -> None:
        """Print all circuit breaker states."""
        breakers = self._breakers
        if not breakers:
            console.print(Panel("[dim]-- no circuit breaker data --[/dim]",
                                title="Circuit Breakers", border_style="bright_black"))
            return

        table = Table(
            title="Circuit Breakers",
            box=box.ROUNDED,
            header_style="bold bright_blue",
            border_style="blue",
        )
        table.add_column("Name", style="bold white", width=28)
        table.add_column("State", width=12)
        table.add_column("Trips", justify="right", width=6)
        table.add_column("Reason / Tripped At", width=40)

        for name, cb in breakers.items():
            if isinstance(cb, dict):
                state      = str(cb.get("state", "UNKNOWN"))
                trip_count = int(cb.get("trip_count", 0))
                reason     = str(cb.get("reason", ""))
                tripped_at = str(cb.get("tripped_at", ""))
            else:
                # simple string state
                state = str(cb)
                trip_count = 0
                reason = ""
                tripped_at = ""

            note = reason
            if tripped_at:
                note += f" @{tripped_at[:19]}"

            table.add_row(
                name,
                Text(state, style=_cb_style(state)),
                str(trip_count),
                Text(note, style="dim"),
            )

        console.print(table)

    # -- Full report ---------------------------------------------------------

    def print_full_report(self) -> None:
        """Print the complete risk state."""
        self._print_header()
        self.print_metrics()
        console.print()
        self.print_positions()
        console.print()
        self.print_limits()
        console.print()
        self.print_circuit_breakers()
        console.print()
        self._print_alerts()

    # -- Alerts (internal helper) -------------------------------------------

    def _print_alerts(self) -> None:
        alerts = self._alerts
        if not alerts:
            return

        table = Table(
            title="Recent Alerts",
            box=box.ROUNDED,
            header_style="bold bright_blue",
            border_style="blue",
        )
        table.add_column("Time", width=20)
        table.add_column("Level", width=8)
        table.add_column("Message")

        for a in alerts[:10]:
            ts    = str(a.get("timestamp", ""))[:19]
            level = str(a.get("level", "INFO"))
            msg   = str(a.get("message", ""))
            sty   = _alert_style(level)
            table.add_row(
                Text(ts, style="dim"),
                Text(level, style=sty),
                Text(msg, style="white"),
            )

        console.print(table)

    # -- Compact mode --------------------------------------------------------

    def print_compact(self) -> None:
        """One-line-per-section compact summary."""
        self._print_header()
        m = self._metrics
        if m:
            var_95    = float(m.get("var_95", 0))
            var_limit = float(m.get("var_limit", 1))
            dd        = float(m.get("drawdown", 0)) * 100
            nav       = float(m.get("nav", 0))
            dpnl      = float(m.get("daily_pnl", 0))
            var_pct   = var_95 / max(var_limit, 1) * 100
            console.print(
                f"  NAV=[bright_green]${nav:,.0f}[/bright_green]  "
                f"DailyP&L=[{_pnl_style(dpnl)}]{_pnl_str(dpnl)}[/{_pnl_style(dpnl)}]  "
                f"VaR=[{_util_style(var_pct)}]{var_pct:.0f}%[/{_util_style(var_pct)}]  "
                f"DD=[{_util_style(dd/15*100)}]{dd:.2f}%[/{_util_style(dd/15*100)}]"
            )

        # breaker summary
        open_cbs = []
        for name, cb in self._breakers.items():
            state = cb.get("state", cb) if isinstance(cb, dict) else cb
            if state == "OPEN":
                open_cbs.append(name)
        if open_cbs:
            console.print(f"  [bright_red]OPEN BREAKERS: {', '.join(open_cbs)}[/bright_red]")
        else:
            console.print("  [bright_green]All circuit breakers CLOSED[/bright_green]")

        n_pos = len(self._positions)
        console.print(f"  Positions: {n_pos}  |  Limits tracked: {len(self._limits)}")


# ---------------------------------------------------------------------------
# JSON snapshot saver
# ---------------------------------------------------------------------------

def save_json_snapshot(data: dict, logs_dir: str = "logs") -> str:
    """Save data as JSON to logs/risk_snapshot_YYYYMMDD_HHMMSS.json.
    Returns the path written."""
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(logs_dir, f"risk_snapshot_{ts}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SRFM Risk Snapshot Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--format", choices=["table", "json", "compact"], default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving JSON snapshot to disk",
    )
    parser.add_argument(
        "--logs-dir", default="logs", dest="logs_dir",
        help="Directory for JSON snapshots (default: logs)",
    )
    parser.add_argument(
        "--section", choices=["metrics", "positions", "limits", "breakers", "all"],
        default="all",
        help="Which section to show (default: all)",
    )
    parser.add_argument(
        "--risk-url", default=RISK_AGG_BASE,
        help=f"risk-aggregator base URL (default: {RISK_AGG_BASE})",
    )
    parser.add_argument(
        "--trader-url", default=LIVE_TRADER_BASE,
        help=f"live-trader base URL (default: {LIVE_TRADER_BASE})",
    )
    parser.add_argument(
        "--coord-url", default=COORDINATION_BASE,
        help=f"coordination base URL (default: {COORDINATION_BASE})",
    )

    args = parser.parse_args(argv)

    # override module-level URLs if provided
    global RISK_AGG_BASE, LIVE_TRADER_BASE, COORDINATION_BASE
    RISK_AGG_BASE       = args.risk_url
    LIVE_TRADER_BASE    = args.trader_url
    COORDINATION_BASE   = args.coord_url

    data = fetch_all()
    printer = RiskSnapshotPrinter(data)

    if args.format == "json":
        print(json.dumps(data, indent=2, default=str))
    elif args.format == "compact":
        printer.print_compact()
    else:
        # table format -- respect --section
        if args.section == "all":
            printer.print_full_report()
        elif args.section == "metrics":
            printer._print_header()
            printer.print_metrics()
        elif args.section == "positions":
            printer._print_header()
            printer.print_positions()
        elif args.section == "limits":
            printer._print_header()
            printer.print_limits()
        elif args.section == "breakers":
            printer._print_header()
            printer.print_circuit_breakers()

    if not args.no_save:
        path = save_json_snapshot(data, args.logs_dir)
        console.print(f"\n[dim]Snapshot saved: {path}[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
