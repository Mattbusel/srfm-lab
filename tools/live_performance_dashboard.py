"""
tools/live_performance_dashboard.py
=====================================
Rich terminal dashboard for live SRFM trading performance.

Reads SRFM SQLite every 30 seconds and displays:
  - Equity curve sparkline
  - Today's P&L by symbol
  - Open positions with unrealized P&L and bars held
  - Current BH mass per symbol (all timeframes)
  - Current Hurst H per symbol
  - Recent trades (last 10)
  - Rolling Sharpe (1d, 7d, 30d)
  - System health: last bar time per symbol, stale feeds

Usage:
    python tools/live_performance_dashboard.py
    python tools/live_performance_dashboard.py --db execution/live_trades.db
    python tools/live_performance_dashboard.py --refresh 10
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    from rich.align import Align
    from rich.rule import Rule
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _REPO / "execution" / "live_trades.db"
_BARS_DB = _REPO / "data" / "bars.db"

SYMBOLS = ["BTC", "ETH", "SOL", "AAPL", "SPY", "QQQ"]
REFRESH_SECS = 30

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _connect(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _ts_n_days_ago(n: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=n)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_closed_trades(conn: sqlite3.Connection, since_iso: str) -> list[dict]:
    for table in ("trade_pnl", "trades"):
        try:
            rows = conn.execute(
                f"SELECT * FROM {table} WHERE exit_time >= ? ORDER BY exit_time",
                (since_iso,)
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            continue
    return []


def load_all_trades(conn: sqlite3.Connection) -> list[dict]:
    for table in ("trade_pnl", "trades"):
        try:
            rows = conn.execute(
                f"SELECT * FROM {table} ORDER BY exit_time"
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            continue
    return []


def load_open_positions(conn: sqlite3.Connection) -> list[dict]:
    """Load open positions from live_trades or positions table."""
    try:
        rows = conn.execute(
            """SELECT * FROM live_trades
               WHERE side IN ('buy','sell') AND qty != 0
               ORDER BY timestamp DESC"""
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        pass
    try:
        rows = conn.execute("SELECT * FROM positions").fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def load_regime_state(conn: sqlite3.Connection) -> dict[str, dict]:
    """Load current regime state from regime_state or nav_state table."""
    result: dict[str, dict] = {}
    for table in ("regime_state", "nav_state", "regime"):
        try:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            for r in rows:
                d = dict(r)
                sym = d.get("symbol", "UNKNOWN")
                result[sym] = d
            if result:
                return result
        except sqlite3.OperationalError:
            continue
    return result


def load_last_bar_times(conn: sqlite3.Connection) -> dict[str, str]:
    """Return symbol -> last_bar_ts from bars table."""
    result: dict[str, str] = {}
    try:
        rows = conn.execute(
            "SELECT symbol, MAX(ts) AS last_ts FROM bars GROUP BY symbol"
        ).fetchall()
        for r in rows:
            result[r["symbol"]] = r["last_ts"]
    except sqlite3.OperationalError:
        pass
    return result


# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------

def _safe_std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(max(var, 0.0))


def rolling_sharpe(trades: list[dict], days: int) -> float:
    since = _ts_n_days_ago(days)
    recent = [
        float(t.get("pnl", 0.0) or 0.0)
        for t in trades
        if (t.get("exit_time") or "") >= since
    ]
    if len(recent) < 2:
        return 0.0
    mean = sum(recent) / len(recent)
    std = _safe_std(recent)
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(252 * 26)  # 15m bars/day


def today_pnl_by_symbol(trades: list[dict]) -> dict[str, float]:
    today = _today_iso()
    result: dict[str, float] = defaultdict(float)
    for t in trades:
        exit_time = t.get("exit_time") or ""
        if exit_time.startswith(today):
            sym = t.get("symbol", "UNKNOWN")
            result[sym] += float(t.get("pnl", 0.0) or 0.0)
    return dict(result)


def equity_curve(all_trades: list[dict], initial: float = 100_000.0) -> list[float]:
    """Build cumulative equity curve from all trades."""
    curve = [initial]
    for t in all_trades:
        pnl = float(t.get("pnl", 0.0) or 0.0)
        curve.append(curve[-1] + pnl)
    return curve


SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 40) -> str:
    """Return a Unicode sparkline string."""
    if len(values) < 2:
        return "-- no data --"
    # Downsample to width
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span == 0:
        return SPARKLINE_CHARS[4] * len(values)
    chars = []
    for v in values:
        idx = int((v - lo) / span * (len(SPARKLINE_CHARS) - 1))
        chars.append(SPARKLINE_CHARS[idx])
    return "".join(chars)


def _is_stale(last_ts: str | None, minutes: int = 20) -> bool:
    """Return True if last_ts is older than `minutes` minutes."""
    if not last_ts:
        return True
    try:
        # Try common formats
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S+00:00"):
            try:
                dt = datetime.strptime(last_ts[:19], fmt[:len(last_ts[:19])])
                dt = dt.replace(tzinfo=timezone.utc)
                return (datetime.now(timezone.utc) - dt).total_seconds() > minutes * 60
            except ValueError:
                continue
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Rich dashboard rendering
# ---------------------------------------------------------------------------

def _pnl_color(val: float) -> str:
    return "green" if val >= 0 else "red"


def build_rich_layout(
    equity: list[float],
    today_pnl: dict[str, float],
    open_positions: list[dict],
    regime_state: dict[str, dict],
    recent_trades: list[dict],
    sharpes: dict[str, float],
    last_bar_times: dict[str, str],
    all_trades: list[dict],
) -> Any:
    """Build a rich Layout object for the dashboard."""
    console = Console()

    # -- Header
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_today = sum(today_pnl.values())
    total_pnl_all = sum(float(t.get("pnl", 0.0) or 0.0) for t in all_trades)
    spark = sparkline(equity[-60:] if len(equity) > 60 else equity)

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top", size=14),
        Layout(name="middle", size=14),
        Layout(name="bottom"),
    )
    layout["top"].split_row(
        Layout(name="today_pnl"),
        Layout(name="positions"),
        Layout(name="sharpe"),
    )
    layout["middle"].split_row(
        Layout(name="regime"),
        Layout(name="recent_trades"),
    )
    layout["bottom"].split_row(
        Layout(name="health"),
        Layout(name="equity_spark"),
    )

    # Header
    color = _pnl_color(total_today)
    header_text = Text(justify="center")
    header_text.append("SRFM Live Dashboard  ", style="bold white")
    header_text.append(f"[{now_str}]  ", style="dim")
    header_text.append(f"Today: ", style="bold")
    header_text.append(f"{total_today:+.2f}  ", style=f"bold {color}")
    header_text.append(f"All-time: {total_pnl_all:+.2f}", style="dim")
    layout["header"].update(Panel(Align.center(header_text), style="bold blue"))

    # Today's P&L by symbol
    pnl_table = Table(title="Today P&L by Symbol", box=None, padding=(0, 1))
    pnl_table.add_column("Symbol", style="bold")
    pnl_table.add_column("P&L", justify="right")
    for sym in sorted(set(list(today_pnl.keys()) + SYMBOLS[:4])):
        pnl = today_pnl.get(sym, 0.0)
        pnl_table.add_row(sym, Text(f"{pnl:+.2f}", style=_pnl_color(pnl)))
    layout["today_pnl"].update(Panel(pnl_table, title="[b]Today P&L[/b]"))

    # Open positions
    pos_table = Table(title="Open Positions", box=None, padding=(0, 1))
    pos_table.add_column("Symbol")
    pos_table.add_column("Qty", justify="right")
    pos_table.add_column("Entry")
    pos_table.add_column("Unr. P&L", justify="right")
    pos_table.add_column("Bars", justify="right")
    for pos in open_positions[:6]:
        sym = pos.get("symbol", "?")
        qty = float(pos.get("qty", 0.0) or 0.0)
        entry = float(pos.get("price", 0.0) or pos.get("entry_price", 0.0) or 0.0)
        upnl = float(pos.get("unrealized_pnl", 0.0) or 0.0)
        bars = int(pos.get("bars_held", 0) or pos.get("hold_bars", 0) or 0)
        pos_table.add_row(
            sym, f"{qty:.3f}", f"{entry:.2f}",
            Text(f"{upnl:+.2f}", style=_pnl_color(upnl)),
            str(bars),
        )
    if not open_positions:
        pos_table.add_row("--", "--", "--", "--", "--")
    layout["positions"].update(Panel(pos_table, title="[b]Open Positions[/b]"))

    # Rolling Sharpe
    sharpe_table = Table(box=None, padding=(0, 1))
    sharpe_table.add_column("Period")
    sharpe_table.add_column("Sharpe", justify="right")
    for period, val in sharpes.items():
        color = "green" if val > 0 else "red"
        sharpe_table.add_row(period, Text(f"{val:.3f}", style=color))
    layout["sharpe"].update(Panel(sharpe_table, title="[b]Rolling Sharpe[/b]"))

    # Regime state
    reg_table = Table(box=None, padding=(0, 1))
    reg_table.add_column("Symbol")
    reg_table.add_column("BH Mass")
    reg_table.add_column("15m")
    reg_table.add_column("1h")
    reg_table.add_column("1d")
    reg_table.add_column("Hurst H")
    for sym in SYMBOLS[:6]:
        rs = regime_state.get(sym, {})
        bh = float(rs.get("bh_mass", 0.0) or 0.0)
        bh15 = "Y" if rs.get("bh_15m") else "N"
        bh1h = "Y" if rs.get("bh_1h") else "N"
        bh1d = "Y" if rs.get("bh_1d") else "N"
        hurst = float(rs.get("hurst_h", 0.5) or 0.5)
        reg_table.add_row(
            sym,
            Text(f"{bh:.2f}", style="green" if bh > 1.0 else "yellow"),
            Text(bh15, style="green" if bh15 == "Y" else "dim"),
            Text(bh1h, style="green" if bh1h == "Y" else "dim"),
            Text(bh1d, style="green" if bh1d == "Y" else "dim"),
            f"{hurst:.3f}",
        )
    layout["regime"].update(Panel(reg_table, title="[b]Current Regime State[/b]"))

    # Recent trades
    rt_table = Table(box=None, padding=(0, 1))
    rt_table.add_column("Symbol")
    rt_table.add_column("Exit")
    rt_table.add_column("P&L", justify="right")
    rt_table.add_column("Bars", justify="right")
    for t in recent_trades[-10:]:
        sym = t.get("symbol", "?")
        et = str(t.get("exit_time", ""))[:16]
        pnl = float(t.get("pnl", 0.0) or 0.0)
        bars = int(t.get("hold_bars", 0) or 0)
        rt_table.add_row(sym, et, Text(f"{pnl:+.2f}", style=_pnl_color(pnl)), str(bars))
    if not recent_trades:
        rt_table.add_row("--", "--", "--", "--")
    layout["recent_trades"].update(Panel(rt_table, title="[b]Recent Trades (last 10)[/b]"))

    # System health
    health_table = Table(box=None, padding=(0, 1))
    health_table.add_column("Symbol")
    health_table.add_column("Last Bar")
    health_table.add_column("Status")
    for sym in SYMBOLS:
        last_ts = last_bar_times.get(sym)
        stale = _is_stale(last_ts)
        status = Text("STALE", style="bold red") if stale else Text("OK", style="green")
        health_table.add_row(sym, last_ts or "never", status)
    layout["health"].update(Panel(health_table, title="[b]System Health[/b]"))

    # Equity sparkline
    spark_text = Text()
    spark_text.append("Equity Curve (last 60 trades)\n", style="bold")
    spark_text.append(spark, style="cyan")
    nav = equity[-1] if equity else 0.0
    spark_text.append(f"\nNAV: {nav:,.2f}", style="bold")
    layout["equity_spark"].update(Panel(spark_text, title="[b]Equity[/b]"))

    return layout


# ---------------------------------------------------------------------------
# Plain-text fallback
# ---------------------------------------------------------------------------

def print_plain_dashboard(
    equity: list[float],
    today_pnl: dict[str, float],
    open_positions: list[dict],
    regime_state: dict[str, dict],
    recent_trades: list[dict],
    sharpes: dict[str, float],
    last_bar_times: dict[str, str],
) -> None:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_today = sum(today_pnl.values())
    spark = sparkline(equity[-60:] if len(equity) > 60 else equity)
    nav = equity[-1] if equity else 0.0

    print("\n" + "=" * 70)
    print(f"  SRFM Live Dashboard  [{now_str}]")
    print(f"  NAV: {nav:,.2f}  Today P&L: {total_today:+.2f}")
    print(f"  Equity: {spark}")
    print("=" * 70)

    print("\n--- Today P&L by Symbol ---")
    for sym, pnl in sorted(today_pnl.items()):
        sign = "+" if pnl >= 0 else ""
        print(f"  {sym:<8} {sign}{pnl:.2f}")

    print("\n--- Rolling Sharpe ---")
    for period, val in sharpes.items():
        print(f"  {period:<8} {val:.3f}")

    print("\n--- Open Positions ---")
    if open_positions:
        for pos in open_positions[:6]:
            sym = pos.get("symbol", "?")
            qty = pos.get("qty", 0)
            entry = pos.get("price", pos.get("entry_price", 0))
            bars = pos.get("bars_held", pos.get("hold_bars", 0))
            print(f"  {sym:<8} qty={qty:.3f}  entry={entry:.2f}  bars={bars}")
    else:
        print("  (no open positions)")

    print("\n--- Current Regime State ---")
    fmt = "  {:<8} bh_mass={:.2f}  hurst={:.3f}"
    for sym in SYMBOLS[:6]:
        rs = regime_state.get(sym, {})
        bh = float(rs.get("bh_mass", 0.0) or 0.0)
        hurst = float(rs.get("hurst_h", 0.5) or 0.5)
        print(fmt.format(sym, bh, hurst))

    print("\n--- Recent Trades (last 10) ---")
    for t in (recent_trades or [])[-10:]:
        sym = t.get("symbol", "?")
        et = str(t.get("exit_time", ""))[:16]
        pnl = float(t.get("pnl", 0.0) or 0.0)
        bars = int(t.get("hold_bars", 0) or 0)
        print(f"  {sym:<8} {et}  pnl={pnl:+.2f}  bars={bars}")

    print("\n--- System Health ---")
    for sym in SYMBOLS:
        last_ts = last_bar_times.get(sym, "never")
        stale = _is_stale(last_ts)
        status = "STALE" if stale else "OK"
        print(f"  {sym:<8} {last_ts or 'never':<26} {status}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _gather_data(db_path: Path) -> dict:
    """Load all data from DB. Returns dict of results."""
    conn = _connect(db_path)
    if conn is None:
        # Demo / synthetic mode
        return _synthetic_data()

    try:
        since_30d = _ts_n_days_ago(30)
        all_trades = load_all_trades(conn)
        today_trades = load_closed_trades(conn, _today_iso() + "T00:00:00")
        open_positions = load_open_positions(conn)
        regime_state = load_regime_state(conn)
        last_bar_times = load_last_bar_times(conn)
    finally:
        conn.close()

    equity = equity_curve(all_trades)
    today_pnl = today_pnl_by_symbol(today_trades)
    sharpes = {
        "1d": rolling_sharpe(all_trades, 1),
        "7d": rolling_sharpe(all_trades, 7),
        "30d": rolling_sharpe(all_trades, 30),
    }
    recent_trades = all_trades[-10:]

    return {
        "equity": equity,
        "today_pnl": today_pnl,
        "open_positions": open_positions,
        "regime_state": regime_state,
        "recent_trades": recent_trades,
        "sharpes": sharpes,
        "last_bar_times": last_bar_times,
        "all_trades": all_trades,
    }


def _synthetic_data() -> dict:
    """Return plausible synthetic data for demo/testing."""
    import random
    rng = random.Random(99)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trades: list[dict] = []
    for i in range(200):
        t = base + timedelta(hours=i * 2)
        pnl = rng.gauss(5.0, 60.0)
        trades.append({
            "symbol": rng.choice(["BTC", "ETH", "SOL", "AAPL"]),
            "exit_time": t.isoformat(),
            "entry_time": (t - timedelta(hours=1)).isoformat(),
            "pnl": round(pnl, 2),
            "hold_bars": rng.randint(1, 20),
        })

    equity = equity_curve(trades)
    today_pnl = {
        "BTC": round(rng.gauss(100, 300), 2),
        "ETH": round(rng.gauss(50, 200), 2),
        "SOL": round(rng.gauss(20, 100), 2),
        "AAPL": round(rng.gauss(10, 50), 2),
    }
    regime_state = {
        sym: {
            "bh_mass": round(rng.uniform(0.5, 2.0), 2),
            "bh_15m": rng.random() > 0.5,
            "bh_1h": rng.random() > 0.5,
            "bh_1d": rng.random() > 0.7,
            "hurst_h": round(rng.uniform(0.3, 0.8), 3),
        }
        for sym in SYMBOLS
    }
    sharpes = {"1d": 0.85, "7d": 1.23, "30d": 0.97}
    last_bar_times = {
        sym: (datetime.now(timezone.utc) - timedelta(minutes=rng.randint(1, 30))).isoformat()[:19]
        for sym in SYMBOLS
    }
    return {
        "equity": equity,
        "today_pnl": today_pnl,
        "open_positions": [],
        "regime_state": regime_state,
        "recent_trades": trades[-10:],
        "sharpes": sharpes,
        "last_bar_times": last_bar_times,
        "all_trades": trades,
    }


def run_dashboard(db_path: Path, refresh_secs: int = REFRESH_SECS) -> None:
    """Main dashboard loop."""
    if HAS_RICH:
        console = Console()
        console.print(
            f"[bold blue]SRFM Live Dashboard[/bold blue] -- "
            f"refresh every {refresh_secs}s -- Ctrl+C to exit"
        )
        try:
            with Live(console=console, refresh_per_second=0.5, screen=True) as live:
                while True:
                    data = _gather_data(db_path)
                    layout = build_rich_layout(
                        equity=data["equity"],
                        today_pnl=data["today_pnl"],
                        open_positions=data["open_positions"],
                        regime_state=data["regime_state"],
                        recent_trades=data["recent_trades"],
                        sharpes=data["sharpes"],
                        last_bar_times=data["last_bar_times"],
                        all_trades=data["all_trades"],
                    )
                    live.update(layout)
                    time.sleep(refresh_secs)
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped.[/yellow]")
    else:
        print(f"SRFM Live Dashboard -- refresh every {refresh_secs}s -- Ctrl+C to exit")
        print("(Install 'rich' for a better UI: pip install rich)\n")
        try:
            while True:
                data = _gather_data(db_path)
                print_plain_dashboard(
                    equity=data["equity"],
                    today_pnl=data["today_pnl"],
                    open_positions=data["open_positions"],
                    regime_state=data["regime_state"],
                    recent_trades=data["recent_trades"],
                    sharpes=data["sharpes"],
                    last_bar_times=data["last_bar_times"],
                )
                time.sleep(refresh_secs)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SRFM live performance dashboard"
    )
    p.add_argument("--db", default=str(_DEFAULT_DB),
                   help="Path to SRFM SQLite DB")
    p.add_argument("--refresh", type=int, default=REFRESH_SECS,
                   help=f"Refresh interval in seconds (default: {REFRESH_SECS})")
    p.add_argument("--once", action="store_true",
                   help="Print one snapshot and exit (no loop)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    db_path = Path(args.db)

    if args.once:
        data = _gather_data(db_path)
        if HAS_RICH:
            console = Console()
            layout = build_rich_layout(
                equity=data["equity"],
                today_pnl=data["today_pnl"],
                open_positions=data["open_positions"],
                regime_state=data["regime_state"],
                recent_trades=data["recent_trades"],
                sharpes=data["sharpes"],
                last_bar_times=data["last_bar_times"],
                all_trades=data["all_trades"],
            )
            console.print(layout)
        else:
            print_plain_dashboard(
                equity=data["equity"],
                today_pnl=data["today_pnl"],
                open_positions=data["open_positions"],
                regime_state=data["regime_state"],
                recent_trades=data["recent_trades"],
                sharpes=data["sharpes"],
                last_bar_times=data["last_bar_times"],
            )
        return 0

    run_dashboard(db_path, args.refresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
