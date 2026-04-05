"""
dashboard.py — Rich Terminal Dashboard for Live Trading Monitor
===============================================================

Provides a live-updating terminal UI using the `rich` library.
Falls back to simple periodic print() output if rich is not available.

Layout
------
  Top row    : Equity | P&L today | P&L this week | Drawdown | Sharpe
  Middle     : Positions table (sym, qty, entry, current, unrealized P&L, BH)
  Bottom     : Recent trades log | Alerts | Regime status
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Rich availability check
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.info("rich not installed — will use plain text fallback")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pnl(value: float, width: int = 12) -> str:
    """Format a P&L value with +/- and colour tag for rich markup."""
    sign = "+" if value >= 0 else ""
    return f"{sign}${value:,.0f}"


def _pnl_colour(value: float) -> str:
    """Return rich colour string based on sign."""
    return "green" if value >= 0 else "red"


def _pct_colour(value: float) -> str:
    return "green" if value >= 0 else "red"


# ---------------------------------------------------------------------------
# Rich dashboard
# ---------------------------------------------------------------------------

class LiveDashboard:
    """
    Interactive terminal dashboard for the live Alpaca trader.

    Parameters
    ----------
    db_path : str
        Path to live_trades.db.
    refresh_interval : int
        Seconds between display refreshes.
    max_recent_trades : int
        How many recent trades to display.
    """

    def __init__(
        self,
        db_path: str,
        refresh_interval: int = 60,
        max_recent_trades: int = 20,
    ) -> None:
        self.db_path = db_path
        self.refresh_interval = refresh_interval
        self.max_recent_trades = max_recent_trades
        self._running = False

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Start the dashboard. Uses rich if available, else plain fallback."""
        if RICH_AVAILABLE:
            self._run_rich()
        else:
            self._run_plain()

    def stop(self) -> None:
        """Signal the dashboard to stop on the next refresh."""
        self._running = False

    # -----------------------------------------------------------------------
    # Data fetching
    # -----------------------------------------------------------------------

    def _fetch_data(self) -> dict[str, Any]:
        """Fetch all data needed for a single dashboard render."""
        from .monitor import LiveTraderMonitor

        monitor = LiveTraderMonitor(self.db_path, poll_interval_seconds=self.refresh_interval)

        data: dict[str, Any] = {
            "fetched_at": datetime.utcnow(),
            "positions": [],
            "metrics": None,
            "health": None,
            "today_pnl": {},
            "recent_trades": [],
            "errors": [],
        }

        try:
            data["positions"] = monitor.get_current_positions()
        except Exception as exc:
            data["errors"].append(f"positions: {exc}")

        try:
            data["metrics"] = monitor.get_live_metrics()
        except Exception as exc:
            data["errors"].append(f"metrics: {exc}")

        try:
            data["health"] = monitor.check_health(data["metrics"])
        except Exception as exc:
            data["errors"].append(f"health: {exc}")

        try:
            data["today_pnl"] = monitor.get_todays_pnl()
        except Exception as exc:
            data["errors"].append(f"today_pnl: {exc}")

        try:
            data["recent_trades"] = self._load_recent_trades()
        except Exception as exc:
            data["errors"].append(f"recent_trades: {exc}")

        return data

    def _load_recent_trades(self) -> list[dict[str, Any]]:
        """Load recent trades from the DB."""
        import sqlite3
        import pandas as pd
        from pathlib import Path

        db = Path(self.db_path)
        if not db.exists():
            return []

        conn = sqlite3.connect(str(db))
        try:
            df = pd.read_sql_query(
                f"SELECT symbol as sym, side, price as fill_price, qty, pnl, ts as fill_time "
                f"FROM trades ORDER BY ts DESC LIMIT {self.max_recent_trades}",
                conn,
            )
        except Exception:
            conn.close()
            return []
        conn.close()
        return df.to_dict(orient="records")

    # -----------------------------------------------------------------------
    # Rich render
    # -----------------------------------------------------------------------

    def _run_rich(self) -> None:
        """Run the rich terminal dashboard."""
        console = Console()
        self._running = True

        try:
            with Live(
                self._build_rich_layout({}),
                console=console,
                refresh_per_second=0.5,
                screen=True,
            ) as live:
                while self._running:
                    data = self._fetch_data()
                    layout = self._build_rich_layout(data)
                    live.update(layout)
                    # Sleep in small increments so we can respond to stop()
                    for _ in range(self.refresh_interval * 2):
                        if not self._running:
                            break
                        time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

    def _build_rich_layout(self, data: dict[str, Any]) -> Layout:
        """Build the full Rich Layout from data."""
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="summary", size=6),
            Layout(name="body"),
            Layout(name="footer", size=12),
        )

        # Header
        ts = data.get("fetched_at", datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S UTC")
        header_text = Text(
            f"  srfm-lab Live Trading Monitor  |  {ts}",
            style="bold white on blue",
            justify="center",
        )
        layout["header"].update(Panel(header_text, box=box.HORIZONTALS))

        # Summary row
        layout["summary"].update(self._build_summary_panel(data))

        # Body: positions table
        layout["body"].update(self._build_positions_panel(data))

        # Footer: recent trades + alerts
        layout["footer"].split_row(
            Layout(name="trades_panel"),
            Layout(name="alerts_panel"),
        )
        layout["footer"]["trades_panel"].update(self._build_trades_panel(data))
        layout["footer"]["alerts_panel"].update(self._build_alerts_panel(data))

        return layout

    def _build_summary_panel(self, data: dict[str, Any]) -> Panel:
        """Build top summary panel with key metrics."""
        metrics = data.get("metrics")
        health = data.get("health")

        table = Table(box=box.SIMPLE, show_header=False, expand=True)
        table.add_column("", style="bold", min_width=18)
        table.add_column("", min_width=18)
        table.add_column("", style="bold", min_width=18)
        table.add_column("", min_width=18)
        table.add_column("", style="bold", min_width=18)
        table.add_column("", min_width=18)

        if metrics:
            equity = health.equity if health else 1_000_000
            dd = metrics.current_drawdown_pct
            daily = metrics.daily_pnl
            weekly = metrics.weekly_pnl
            sharpe = metrics.sharpe_ratio
            win_rate = metrics.win_rate

            table.add_row(
                "Equity",
                f"[bold cyan]${equity:,.0f}[/bold cyan]",
                "P&L Today",
                f"[{_pnl_colour(daily)}]{_fmt_pnl(daily)}[/{_pnl_colour(daily)}]",
                "P&L Week",
                f"[{_pnl_colour(weekly)}]{_fmt_pnl(weekly)}[/{_pnl_colour(weekly)}]",
            )
            table.add_row(
                "Drawdown",
                f"[{'red' if dd > 10 else 'yellow' if dd > 5 else 'green'}]{dd:.1f}%[/]",
                "Sharpe (30d)",
                f"[{'green' if sharpe > 1 else 'yellow' if sharpe > 0 else 'red'}]{sharpe:.2f}[/]",
                "Win Rate",
                f"[{'green' if win_rate > 0.5 else 'red'}]{win_rate*100:.0f}%[/]",
            )
        else:
            table.add_row("Status", "[yellow]Loading...[/yellow]", "", "", "", "")

        status = health.status if health else "LOADING"
        status_color = {"OK": "green", "WARN": "yellow", "CRITICAL": "red"}.get(status, "white")

        return Panel(
            table,
            title=f"[bold {status_color}]● {status}[/bold {status_color}]  Summary",
            border_style=status_color,
            box=box.ROUNDED,
        )

    def _build_positions_panel(self, data: dict[str, Any]) -> Panel:
        """Build the positions table."""
        positions = data.get("positions", [])

        table = Table(
            show_header=True,
            box=box.SIMPLE_HEAVY,
            expand=True,
            header_style="bold magenta",
        )
        table.add_column("Symbol", style="bold", min_width=12)
        table.add_column("Side", justify="center", min_width=6)
        table.add_column("Qty", justify="right", min_width=12)
        table.add_column("Entry", justify="right", min_width=12)
        table.add_column("Current", justify="right", min_width=12)
        table.add_column("Notional", justify="right", min_width=14)
        table.add_column("Unrealized P&L", justify="right", min_width=16)
        table.add_column("P&L %", justify="right", min_width=8)
        table.add_column("BH", justify="center", min_width=5)

        if not positions:
            table.add_row(
                "[dim]No open positions[/dim]", "", "", "", "", "", "", "", ""
            )
        else:
            for pos in sorted(positions, key=lambda p: -abs(p.notional)):
                side_color = "green" if pos.qty > 0 else "red"
                side = "LONG" if pos.qty > 0 else "SHORT"
                pnl_color = _pnl_colour(pos.unrealized_pnl)
                pct_color = _pct_colour(pos.pnl_pct)
                bh_icon = "[green]●[/green]" if pos.bh_active else "[dim]○[/dim]"

                table.add_row(
                    pos.sym,
                    f"[{side_color}]{side}[/{side_color}]",
                    f"{pos.qty:,.4f}",
                    f"${pos.avg_entry:,.2f}",
                    f"${pos.current_price:,.2f}",
                    f"${pos.notional:,.0f}",
                    f"[{pnl_color}]{_fmt_pnl(pos.unrealized_pnl)}[/{pnl_color}]",
                    f"[{pct_color}]{pos.pnl_pct:+.1f}%[/{pct_color}]",
                    bh_icon,
                )

        n_pos = len(positions)
        total_notional = sum(abs(p.notional) for p in positions)
        total_unrealized = sum(p.unrealized_pnl for p in positions)
        pnl_color = _pnl_colour(total_unrealized)

        return Panel(
            table,
            title=f"Open Positions  ({n_pos} positions, "
                  f"${total_notional:,.0f} notional, "
                  f"[{pnl_color}]{_fmt_pnl(total_unrealized)} unrealized[/{pnl_color}])",
            border_style="blue",
            box=box.ROUNDED,
        )

    def _build_trades_panel(self, data: dict[str, Any]) -> Panel:
        """Build recent trades panel."""
        trades = data.get("recent_trades", [])

        table = Table(
            show_header=True,
            box=box.SIMPLE,
            expand=True,
            header_style="bold cyan",
        )
        table.add_column("Time", min_width=10)
        table.add_column("Sym", min_width=10)
        table.add_column("Side", min_width=6)
        table.add_column("Price", justify="right", min_width=10)
        table.add_column("P&L", justify="right", min_width=10)

        if not trades:
            table.add_row("[dim]No recent trades[/dim]", "", "", "", "")
        else:
            for t in trades[:10]:
                side = str(t.get("side", "?")).upper()
                side_color = "green" if side == "BUY" else "red"
                pnl = float(t.get("pnl", 0.0) or 0.0)
                pnl_color = _pnl_colour(pnl)
                fill_time = str(t.get("fill_time", ""))[:16]

                table.add_row(
                    fill_time,
                    str(t.get("sym", "")),
                    f"[{side_color}]{side}[/{side_color}]",
                    f"${float(t.get('fill_price', 0)):,.2f}",
                    f"[{pnl_color}]{_fmt_pnl(pnl)}[/{pnl_color}]",
                )

        return Panel(table, title="Recent Trades", border_style="cyan", box=box.ROUNDED)

    def _build_alerts_panel(self, data: dict[str, Any]) -> Panel:
        """Build alerts + regime panel."""
        health = data.get("health")
        metrics = data.get("metrics")

        lines: list[str] = []

        if health:
            for alert in health.alerts:
                if "CRITICAL" in alert:
                    lines.append(f"[bold red]{alert}[/bold red]")
                elif "WARN" in alert:
                    lines.append(f"[yellow]{alert}[/yellow]")
                else:
                    lines.append(f"[green]{alert}[/green]")
        else:
            lines.append("[dim]No health data[/dim]")

        if metrics:
            lines.append("")
            lines.append(
                f"[dim]Last trade: "
                f"{metrics.last_trade_time.strftime('%H:%M:%S') if metrics.last_trade_time else 'N/A'}[/dim]"
            )
            lines.append(f"[dim]Order failures: {metrics.order_failure_rate*100:.1f}%[/dim]")
            lines.append(f"[dim]N trades (30d): {metrics.n_trades}[/dim]")

        errors = data.get("errors", [])
        if errors:
            lines.append("")
            lines.append("[bold red]Data Errors:[/bold red]")
            for err in errors[:3]:
                lines.append(f"[red]  {err}[/red]")

        content = "\n".join(lines) or "[dim]No alerts[/dim]"
        return Panel(
            Text.from_markup(content),
            title="Alerts & Status",
            border_style="yellow",
            box=box.ROUNDED,
        )

    # -----------------------------------------------------------------------
    # Plain fallback
    # -----------------------------------------------------------------------

    def _run_plain(self) -> None:
        """Simple print-based fallback when rich is not available."""
        self._running = True
        print("srfm-lab Live Monitor (plain mode — install 'rich' for full dashboard)")
        print("=" * 70)

        try:
            while self._running:
                data = self._fetch_data()
                self._print_plain(data)
                for _ in range(self.refresh_interval):
                    if not self._running:
                        break
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            print("\nDashboard stopped.")

    def _print_plain(self, data: dict[str, Any]) -> None:
        """Print a plain-text snapshot of the current state."""
        ts = data.get("fetched_at", datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S UTC")
        metrics = data.get("metrics")
        health = data.get("health")
        positions = data.get("positions", [])
        trades = data.get("recent_trades", [])

        print(f"\n[{ts}]  STATUS: {health.status if health else 'UNKNOWN'}")
        print("-" * 70)

        if metrics:
            eq = health.equity if health else 0
            print(f"  Equity:     ${eq:>12,.0f}   "
                  f"P&L Today: {_fmt_pnl(metrics.daily_pnl):>12}   "
                  f"P&L Week: {_fmt_pnl(metrics.weekly_pnl):>12}")
            print(f"  Drawdown:   {metrics.current_drawdown_pct:>10.1f}%   "
                  f"Sharpe:    {metrics.sharpe_ratio:>10.2f}   "
                  f"Win Rate: {metrics.win_rate*100:>9.0f}%")

        if health and health.alerts:
            print("\n  Alerts:")
            for alert in health.alerts:
                print(f"    {alert}")

        if positions:
            print(f"\n  Positions ({len(positions)}):")
            print(f"    {'Sym':<12} {'Side':<6} {'Qty':>10} {'Entry':>10} "
                  f"{'Current':>10} {'Unreal P&L':>12} {'BH'}")
            print("    " + "-" * 65)
            for pos in positions:
                side = "LONG" if pos.qty > 0 else "SHORT"
                bh = "Y" if pos.bh_active else "N"
                print(
                    f"    {pos.sym:<12} {side:<6} {pos.qty:>10.4f} "
                    f"{pos.avg_entry:>10.2f} {pos.current_price:>10.2f} "
                    f"{_fmt_pnl(pos.unrealized_pnl):>12} {bh}"
                )

        if trades:
            print(f"\n  Recent Trades (last {min(5, len(trades))}):")
            for t in trades[:5]:
                pnl = float(t.get("pnl", 0.0) or 0.0)
                print(
                    f"    [{str(t.get('fill_time',''))[:16]}] "
                    f"{t.get('sym','')} {str(t.get('side','')).upper()} "
                    f"@ ${float(t.get('fill_price', 0)):,.2f}  "
                    f"P&L: {_fmt_pnl(pnl)}"
                )

        for err in data.get("errors", []):
            print(f"  ERROR: {err}")

        print("-" * 70)
