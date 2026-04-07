"""
tools/live_monitor_v2.py
========================
Enhanced live monitoring dashboard for LARSA v18.

Reads SRFM SQLite every 15 seconds, polls coordination layer (:8781),
risk API (:8791), and observability API (:9091).

Run:
    python tools/live_monitor_v2.py
    python tools/live_monitor_v2.py --no-rich   # plain text fallback
    python tools/live_monitor_v2.py --interval 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import aiohttp

# ── Rich terminal UI (optional) ───────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.table import Table
    from rich.text import Text
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

log = logging.getLogger("live_monitor_v2")
logging.basicConfig(
    stream=sys.stderr,
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT   = Path(__file__).parents[1]
_DB_PATH     = _REPO_ROOT / "execution" / "live_trades.db"
_COORD_URL   = "http://127.0.0.1:8781"
_RISK_URL    = "http://127.0.0.1:8791"
_OBS_URL     = "http://127.0.0.1:9091"

STRATEGY_VERSION = "larsa_v18"
DEFAULT_INTERVAL = 15  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Data layer -- async fetchers
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_json(session: aiohttp.ClientSession, url: str, timeout: float = 3.0) -> dict:
    """Fetch JSON from a URL, returning empty dict on failure."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                return await resp.json(content_type=None)
    except Exception as exc:
        log.debug("fetch_json %s failed: %s", url, exc)
    return {}


async def fetch_coordination(session: aiohttp.ClientSession) -> dict:
    data = await _fetch_json(session, f"{_COORD_URL}/state")
    params = await _fetch_json(session, f"{_COORD_URL}/params")
    breakers = await _fetch_json(session, f"{_COORD_URL}/circuit_breakers")
    return {"state": data, "params": params, "circuit_breakers": breakers}


async def fetch_risk(session: aiohttp.ClientSession) -> dict:
    summary = await _fetch_json(session, f"{_RISK_URL}/summary")
    limits = await _fetch_json(session, f"{_RISK_URL}/limits")
    return {"summary": summary, "limits": limits}


async def fetch_observability(session: aiohttp.ClientSession) -> dict:
    metrics = await _fetch_json(session, f"{_OBS_URL}/metrics")
    alerts  = await _fetch_json(session, f"{_OBS_URL}/alerts")
    return {"metrics": metrics, "alerts": alerts}


# ─────────────────────────────────────────────────────────────────────────────
# SQLite queries
# ─────────────────────────────────────────────────────────────────────────────

def _open_db(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:
        log.warning("Cannot open DB %s: %s", db_path, exc)
        return None


def query_equity(conn: sqlite3.Connection) -> dict:
    """Return equity snapshot from trade history."""
    result = {
        "current_equity": None,
        "daily_pnl": 0.0,
        "seven_day_pnl": 0.0,
        "max_drawdown": 0.0,
        "last_bar_time": None,
    }
    try:
        # Cumulative P&L from trade_pnl
        rows = conn.execute(
            "SELECT exit_time, pnl FROM trade_pnl ORDER BY exit_time"
        ).fetchall()
        if rows:
            cumulative = 0.0
            peak = 0.0
            max_dd = 0.0
            now_utc = datetime.now(timezone.utc)
            today_str = now_utc.strftime("%Y-%m-%d")
            seven_ago = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")
            daily_pnl = 0.0
            seven_day_pnl = 0.0
            for row in rows:
                cumulative += row["pnl"]
                if cumulative > peak:
                    peak = cumulative
                dd = peak - cumulative
                if dd > max_dd:
                    max_dd = dd
                if row["exit_time"] and row["exit_time"][:10] >= today_str:
                    daily_pnl += row["pnl"]
                if row["exit_time"] and row["exit_time"][:10] >= seven_ago:
                    seven_day_pnl += row["pnl"]
            result["daily_pnl"] = daily_pnl
            result["seven_day_pnl"] = seven_day_pnl
            result["max_drawdown"] = max_dd
            result["last_bar_time"] = rows[-1]["exit_time"]
    except Exception as exc:
        log.debug("query_equity: %s", exc)
    return result


def query_positions(conn: sqlite3.Connection) -> list[dict]:
    """Return open positions with signal state."""
    positions = []
    try:
        # Get open trades (buy without matching sell)
        rows = conn.execute("""
            SELECT symbol, side, qty, price, fill_time, strategy_version
            FROM live_trades
            ORDER BY fill_time DESC
        """).fetchall()
        # Simple FIFO: net position per symbol
        net: dict[str, dict] = {}
        for row in rows:
            sym = row["symbol"]
            if sym not in net:
                net[sym] = {"symbol": sym, "qty": 0.0, "entry_price": 0.0,
                             "entry_time": row["fill_time"], "fills": 0}
            if row["side"] == "buy":
                net[sym]["qty"] += row["qty"]
                net[sym]["entry_price"] = (
                    (net[sym]["entry_price"] * (net[sym]["fills"]) + row["price"])
                    / (net[sym]["fills"] + 1)
                )
                net[sym]["fills"] += 1
            else:
                net[sym]["qty"] -= row["qty"]
        # Only symbols with nonzero qty are open
        for sym, p in net.items():
            if abs(p["qty"]) > 1e-9:
                positions.append(p)

        # Try to get nav_state info if nav_state table exists
        try:
            nav_rows = conn.execute("""
                SELECT symbol, bh_mass_15m, bh_mass_1h, hurst_h, nav_omega, signal_strength,
                       bars_held, strategy_version
                FROM nav_state
                WHERE ts = (SELECT MAX(ts) FROM nav_state)
            """).fetchall()
            nav_map = {r["symbol"]: dict(r) for r in nav_rows}
            for pos in positions:
                ns = nav_map.get(pos["symbol"], {})
                pos["bh_mass_15m"]    = ns.get("bh_mass_15m", 0.0)
                pos["bh_mass_1h"]     = ns.get("bh_mass_1h", 0.0)
                pos["hurst_regime"]   = _hurst_label(ns.get("hurst_h"))
                pos["nav_omega"]      = ns.get("nav_omega", 0.0)
                pos["signal_strength"] = ns.get("signal_strength", 0.0)
                pos["bars_held"]      = ns.get("bars_held", 0)
        except Exception:
            for pos in positions:
                pos.setdefault("bh_mass_15m", 0.0)
                pos.setdefault("bh_mass_1h", 0.0)
                pos.setdefault("hurst_regime", "N/A")
                pos.setdefault("nav_omega", 0.0)
                pos.setdefault("signal_strength", 0.0)
                pos.setdefault("bars_held", 0)

    except Exception as exc:
        log.debug("query_positions: %s", exc)
    return positions


def query_recent_trades(conn: sqlite3.Connection, n: int = 5) -> list[dict]:
    """Return last N completed trades."""
    try:
        rows = conn.execute("""
            SELECT symbol, entry_time, exit_time, entry_price, exit_price,
                   qty, pnl, hold_bars
            FROM trade_pnl
            ORDER BY exit_time DESC
            LIMIT ?
        """, (n,)).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.debug("query_recent_trades: %s", exc)
        return []


def query_signal_state(conn: sqlite3.Connection) -> dict:
    """Return latest signal gate states if stored."""
    defaults = {
        "quatnav_gate": None,
        "hurst_filter": None,
        "calendar_filter": None,
        "granger_boost": None,
        "ml_signal_dir": None,
        "macro_regime": "UNKNOWN",
    }
    try:
        row = conn.execute("""
            SELECT * FROM signal_state ORDER BY ts DESC LIMIT 1
        """).fetchone()
        if row:
            d = dict(row)
            defaults.update(d)
    except Exception:
        pass
    return defaults


def _hurst_label(h: float | None) -> str:
    if h is None:
        return "N/A"
    if h > 0.58:
        return f"TREND({h:.2f})"
    if h < 0.42:
        return f"MR({h:.2f})"
    return f"RAND({h:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
# State container
# ─────────────────────────────────────────────────────────────────────────────

class MonitorState:
    def __init__(self) -> None:
        self.start_time   = time.time()
        self.last_refresh = 0.0
        self.equity_data: dict        = {}
        self.positions: list[dict]    = []
        self.recent_trades: list[dict] = []
        self.signal_state: dict       = {}
        self.coordination: dict       = {}
        self.risk: dict               = {}
        self.observability: dict      = {}
        self.error_counts: dict[str, int] = {}
        self.refresh_count = 0

    def uptime_str(self) -> str:
        elapsed = int(time.time() - self.start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        return f"{h:02d}:{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Refresh logic
# ─────────────────────────────────────────────────────────────────────────────

async def refresh(state: MonitorState, db_path: Path) -> None:
    """Pull all data sources and update state."""
    conn = _open_db(db_path)
    if conn:
        state.equity_data    = query_equity(conn)
        state.positions      = query_positions(conn)
        state.recent_trades  = query_recent_trades(conn)
        state.signal_state   = query_signal_state(conn)
        conn.close()

    async with aiohttp.ClientSession() as session:
        coord_task = asyncio.create_task(fetch_coordination(session))
        risk_task  = asyncio.create_task(fetch_risk(session))
        obs_task   = asyncio.create_task(fetch_observability(session))
        state.coordination  = await coord_task
        state.risk          = await risk_task
        state.observability = await obs_task

    state.last_refresh  = time.time()
    state.refresh_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Rich rendering
# ─────────────────────────────────────────────────────────────────────────────

def _ascii_bar(value: float, width: int = 20, max_val: float = 1.0) -> str:
    """Return ASCII progress bar string."""
    if max_val <= 0:
        max_val = 1.0
    filled = int(min(value / max_val, 1.0) * width)
    filled = max(0, filled)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _fmt_pnl(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:,.2f}"


def _gate_str(v: Any) -> str:
    if v is None:
        return "[dim]--[/dim]"
    return "[green]YES[/green]" if v else "[red]NO[/red]"


def _gate_str_plain(v: Any) -> str:
    if v is None:
        return "--"
    return "YES" if v else "NO"


def build_rich_display(state: MonitorState, console: "Console") -> "Layout":
    """Build the full rich Layout from current state."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top_row", size=10),
        Layout(name="positions", size=12),
        Layout(name="middle_row", size=10),
        Layout(name="bottom_row", size=12),
        Layout(name="footer", size=5),
    )
    layout["top_row"].split_row(
        Layout(name="equity", ratio=2),
        Layout(name="risk", ratio=2),
        Layout(name="params", ratio=2),
    )
    layout["middle_row"].split_row(
        Layout(name="bh_physics", ratio=3),
        Layout(name="regime", ratio=2),
    )
    layout["bottom_row"].split_row(
        Layout(name="recent_trades", ratio=2),
        Layout(name="signals", ratio=2),
        Layout(name="alerts", ratio=1),
    )

    # -- Header --
    last_bar = state.equity_data.get("last_bar_time") or "N/A"
    header_text = Text(justify="center")
    header_text.append(f"  {STRATEGY_VERSION}  ", style="bold cyan")
    header_text.append(f"uptime: {state.uptime_str()}  ", style="green")
    header_text.append(f"last bar: {last_bar}  ", style="yellow")
    header_text.append(f"refresh #{state.refresh_count}", style="dim")
    layout["header"].update(Panel(header_text, box=box.SIMPLE))

    # -- Equity --
    eq = state.equity_data
    equity_tbl = Table(box=box.SIMPLE, padding=(0, 1))
    equity_tbl.add_column("Metric", style="cyan")
    equity_tbl.add_column("Value", style="white", justify="right")
    equity_val = eq.get("current_equity")
    equity_tbl.add_row("Equity", f"${equity_val:,.2f}" if equity_val else "N/A")
    dpnl = eq.get("daily_pnl", 0.0)
    equity_tbl.add_row("Daily P&L",   _fmt_pnl(dpnl), )
    s7pnl = eq.get("seven_day_pnl", 0.0)
    equity_tbl.add_row("7-Day P&L",   _fmt_pnl(s7pnl))
    mdd = eq.get("max_drawdown", 0.0)
    equity_tbl.add_row("Max Drawdown", f"-{mdd:,.2f}")
    layout["equity"].update(Panel(equity_tbl, title="Equity", border_style="green"))

    # -- Positions table --
    pos_tbl = Table(box=box.SIMPLE, padding=(0, 1), show_header=True)
    for col in ("Symbol", "Pos%", "Unreal PNL", "Bars", "BH 15m", "BH 1h", "Hurst", "NavOmega", "SigStr"):
        pos_tbl.add_column(col, style="cyan" if col == "Symbol" else "white", justify="right" if col != "Symbol" else "left")
    for p in state.positions:
        sym = p.get("symbol", "?")
        pos_pct = f"{p.get('last_frac', 0.0)*100:.1f}%"
        upnl = p.get("unrealized_pnl", 0.0)
        bars = str(p.get("bars_held", 0))
        bh15 = f"{p.get('bh_mass_15m', 0.0):.3f}"
        bh1h = f"{p.get('bh_mass_1h', 0.0):.3f}"
        hurst = p.get("hurst_regime", "N/A")
        omega = f"{p.get('nav_omega', 0.0):.4f}"
        sigstr = f"{p.get('signal_strength', 0.0):.3f}"
        pos_tbl.add_row(sym, pos_pct, _fmt_pnl(upnl), bars, bh15, bh1h, hurst, omega, sigstr)
    if not state.positions:
        pos_tbl.add_row("[dim]no open positions[/dim]", "", "", "", "", "", "", "", "")
    layout["positions"].update(Panel(pos_tbl, title="Open Positions", border_style="blue"))

    # -- BH Physics --
    bh_lines: list[str] = []
    for p in state.positions:
        sym = p.get("symbol", "?")
        m15 = p.get("bh_mass_15m", 0.0)
        m1h = p.get("bh_mass_1h", 0.0)
        bar15 = _ascii_bar(m15)
        bar1h = _ascii_bar(m1h)
        bh_lines.append(f"{sym:>6}  15m {bar15} {m15:.3f}")
        bh_lines.append(f"{'':>6}   1h {bar1h} {m1h:.3f}")
    bh_content = "\n".join(bh_lines) if bh_lines else "[dim]no positions[/dim]"
    layout["bh_physics"].update(Panel(bh_content, title="BH Mass Accumulation", border_style="magenta"))

    # -- Regime --
    ss = state.signal_state
    macro_regime = ss.get("macro_regime", "UNKNOWN")
    hursts = [p.get("hurst_regime", "") for p in state.positions]
    hurst_dist = {}
    for h in hursts:
        if h.startswith("TREND"):
            hurst_dist["TREND"] = hurst_dist.get("TREND", 0) + 1
        elif h.startswith("MR"):
            hurst_dist["MR"] = hurst_dist.get("MR", 0) + 1
        else:
            hurst_dist["RAND"] = hurst_dist.get("RAND", 0) + 1
    regime_lines = [f"Macro: [bold]{macro_regime}[/bold]", ""]
    for k, v in hurst_dist.items():
        regime_lines.append(f"  {k}: {v} symbol(s)")
    if not hurst_dist:
        regime_lines.append("[dim]no regime data[/dim]")
    layout["regime"].update(Panel("\n".join(regime_lines), title="Regime", border_style="yellow"))

    # -- Risk --
    risk_summary = state.risk.get("summary", {})
    risk_limits  = state.risk.get("limits", {})
    breakers     = state.coordination.get("circuit_breakers", {})
    risk_tbl = Table(box=box.SIMPLE, padding=(0, 1))
    risk_tbl.add_column("Item", style="cyan")
    risk_tbl.add_column("Value", style="white", justify="right")
    var95 = risk_summary.get("var_95", None)
    risk_tbl.add_row("VaR (95%)", f"{var95:.4f}" if var95 else "N/A")
    breaches = risk_limits.get("active_breaches", [])
    breach_str = ", ".join(breaches) if breaches else "[green]none[/green]"
    risk_tbl.add_row("Limit Breaches", breach_str)
    for name, active in (breakers or {}).items():
        style = "[red]OPEN[/red]" if active else "[green]CLOSED[/green]"
        risk_tbl.add_row(f"CB: {name}", style)
    layout["risk"].update(Panel(risk_tbl, title="Risk", border_style="red"))

    # -- Parameters --
    params      = state.coordination.get("params", {})
    param_vals  = params.get("values", {})
    param_src   = params.get("last_source", "N/A")
    param_time  = params.get("last_updated", "N/A")
    param_tbl = Table(box=box.SIMPLE, padding=(0, 1), show_header=False)
    param_tbl.add_column("K")
    param_tbl.add_column("V")
    for k, v in list(param_vals.items())[:8]:
        param_tbl.add_row(str(k), str(v))
    param_tbl.add_row("[dim]source[/dim]", str(param_src))
    param_tbl.add_row("[dim]updated[/dim]", str(param_time))
    layout["params"].update(Panel(param_tbl, title="Parameters", border_style="cyan"))

    # -- Recent Trades --
    trade_tbl = Table(box=box.SIMPLE, padding=(0, 1))
    trade_tbl.add_column("Symbol", style="cyan")
    trade_tbl.add_column("P&L", justify="right")
    trade_tbl.add_column("Bars", justify="right")
    trade_tbl.add_column("Exit", style="dim")
    for t in state.recent_trades:
        pnl_val = t.get("pnl", 0.0)
        pnl_style = "green" if pnl_val >= 0 else "red"
        trade_tbl.add_row(
            t.get("symbol", "?"),
            f"[{pnl_style}]{_fmt_pnl(pnl_val)}[/{pnl_style}]",
            str(t.get("hold_bars", 0)),
            (t.get("exit_time") or "")[:16],
        )
    if not state.recent_trades:
        trade_tbl.add_row("[dim]no exits yet[/dim]", "", "", "")
    layout["recent_trades"].update(Panel(trade_tbl, title="Recent Trades (last 5)", border_style="green"))

    # -- Signal Breakdown --
    sig_tbl = Table(box=box.SIMPLE, padding=(0, 1), show_header=False)
    sig_tbl.add_column("Gate")
    sig_tbl.add_column("State")
    sig_tbl.add_row("QuatNav Gate",   _gate_str(ss.get("quatnav_gate")))
    sig_tbl.add_row("Hurst Filter",   _gate_str(ss.get("hurst_filter")))
    sig_tbl.add_row("Calendar Filter",_gate_str(ss.get("calendar_filter")))
    sig_tbl.add_row("Granger Boost",  _gate_str(ss.get("granger_boost")))
    ml_dir = ss.get("ml_signal_dir")
    ml_str = str(ml_dir) if ml_dir is not None else "[dim]--[/dim]"
    sig_tbl.add_row("ML Signal Dir",  ml_str)
    layout["signals"].update(Panel(sig_tbl, title="Signal Breakdown", border_style="yellow"))

    # -- Alerts --
    obs_alerts = state.observability.get("alerts", {})
    alert_items = obs_alerts.get("active", []) if isinstance(obs_alerts, dict) else []
    if alert_items:
        alert_lines = "\n".join(f"[red]! {a}[/red]" for a in alert_items[:6])
    else:
        alert_lines = "[green]No active alerts[/green]"
    layout["alerts"].update(Panel(alert_lines, title="Alerts", border_style="red"))

    # -- Footer --
    footer_text = Text(justify="center")
    footer_text.append(f"  DB: {_DB_PATH}  |  COORD: {_COORD_URL}  |  RISK: {_RISK_URL}  |  OBS: {_OBS_URL}  ", style="dim")
    layout["footer"].update(Panel(footer_text, box=box.SIMPLE))

    return layout


# ─────────────────────────────────────────────────────────────────────────────
# Plain-text fallback renderer
# ─────────────────────────────────────────────────────────────────────────────

def print_plain(state: MonitorState) -> None:
    """Print plain text dashboard to stdout."""
    eq   = state.equity_data
    sep  = "-" * 70

    print(sep)
    print(f"  {STRATEGY_VERSION}  |  uptime: {state.uptime_str()}  |  refresh #{state.refresh_count}")
    last_bar = eq.get("last_bar_time") or "N/A"
    print(f"  last bar: {last_bar}")
    print(sep)

    # Equity
    print("EQUITY")
    print(f"  Daily P&L:   {_fmt_pnl(eq.get('daily_pnl', 0.0))}")
    print(f"  7-Day P&L:   {_fmt_pnl(eq.get('seven_day_pnl', 0.0))}")
    print(f"  Max Drawdown: -{eq.get('max_drawdown', 0.0):,.2f}")

    # Positions
    print(sep)
    print("OPEN POSITIONS")
    print(f"  {'Symbol':<8} {'Bars':>5} {'BH15m':>7} {'BH1h':>7} {'Hurst':<14} {'NavOmega':>9} {'SigStr':>7}")
    for p in state.positions:
        print(
            f"  {p.get('symbol','?'):<8}"
            f" {p.get('bars_held',0):>5}"
            f" {p.get('bh_mass_15m',0.0):>7.3f}"
            f" {p.get('bh_mass_1h',0.0):>7.3f}"
            f" {p.get('hurst_regime','N/A'):<14}"
            f" {p.get('nav_omega',0.0):>9.4f}"
            f" {p.get('signal_strength',0.0):>7.3f}"
        )
    if not state.positions:
        print("  (no open positions)")

    # BH Physics bars
    print(sep)
    print("BH MASS ACCUMULATION")
    for p in state.positions:
        sym = p.get("symbol", "?")
        m15 = p.get("bh_mass_15m", 0.0)
        m1h = p.get("bh_mass_1h", 0.0)
        print(f"  {sym:>6}  15m {_ascii_bar(m15)} {m15:.3f}")
        print(f"  {'':>6}   1h {_ascii_bar(m1h)} {m1h:.3f}")

    # Regime
    print(sep)
    ss = state.signal_state
    print(f"REGIME  macro={ss.get('macro_regime','UNKNOWN')}")

    # Risk
    print(sep)
    risk_summary = state.risk.get("summary", {})
    var95 = risk_summary.get("var_95")
    print(f"RISK  VaR(95%)={var95:.4f}" if var95 else "RISK  VaR(95%)=N/A")
    breaches = state.risk.get("limits", {}).get("active_breaches", [])
    print(f"  Limit breaches: {breaches or 'none'}")
    for name, active in (state.coordination.get("circuit_breakers") or {}).items():
        print(f"  CB {name}: {'OPEN' if active else 'CLOSED'}")

    # Parameters
    print(sep)
    params    = state.coordination.get("params", {})
    param_vals = params.get("values", {})
    print("PARAMS")
    for k, v in list(param_vals.items())[:6]:
        print(f"  {k} = {v}")
    print(f"  source={params.get('last_source','N/A')}  updated={params.get('last_updated','N/A')}")

    # Signals
    print(sep)
    print("SIGNAL GATES")
    print(f"  QuatNav gate:   {_gate_str_plain(ss.get('quatnav_gate'))}")
    print(f"  Hurst filter:   {_gate_str_plain(ss.get('hurst_filter'))}")
    print(f"  Calendar filter:{_gate_str_plain(ss.get('calendar_filter'))}")
    print(f"  Granger boost:  {_gate_str_plain(ss.get('granger_boost'))}")
    print(f"  ML signal dir:  {ss.get('ml_signal_dir','--')}")

    # Recent Trades
    print(sep)
    print("RECENT TRADES (last 5)")
    for t in state.recent_trades:
        pnl_val = t.get("pnl", 0.0)
        print(
            f"  {t.get('symbol','?'):<8}"
            f"  pnl={_fmt_pnl(pnl_val):>10}"
            f"  bars={t.get('hold_bars',0):>4}"
            f"  exit={str(t.get('exit_time',''))[:16]}"
        )
    if not state.recent_trades:
        print("  (no exits yet)")

    # Alerts
    print(sep)
    obs_alerts = state.observability.get("alerts", {})
    alert_items = obs_alerts.get("active", []) if isinstance(obs_alerts, dict) else []
    print("ALERTS")
    if alert_items:
        for a in alert_items:
            print(f"  ! {a}")
    else:
        print("  No active alerts")
    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

async def run_monitor(interval: int, db_path: Path, use_rich: bool) -> None:
    state   = MonitorState()
    console = Console() if (use_rich and _RICH_AVAILABLE) else None

    if console:
        with Live(build_rich_display(state, console), console=console, refresh_per_second=1, screen=True) as live:
            while True:
                await refresh(state, db_path)
                live.update(build_rich_display(state, console))
                await asyncio.sleep(interval)
    else:
        if use_rich and not _RICH_AVAILABLE:
            print("WARNING: rich library not available, using plain text fallback.")
        while True:
            await refresh(state, db_path)
            os.system("cls" if sys.platform == "win32" else "clear")
            print_plain(state)
            await asyncio.sleep(interval)


def main() -> None:
    global _COORD_URL, _RISK_URL, _OBS_URL
    parser = argparse.ArgumentParser(description="LARSA v18 live monitoring dashboard")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help="Refresh interval in seconds (default: 15)")
    parser.add_argument("--db", type=str, default=str(_DB_PATH),
                        help="Path to live_trades.db")
    parser.add_argument("--no-rich", action="store_true",
                        help="Disable rich terminal UI, use plain text")
    parser.add_argument("--coord-url", type=str, default=_COORD_URL)
    parser.add_argument("--risk-url",  type=str, default=_RISK_URL)
    parser.add_argument("--obs-url",   type=str, default=_OBS_URL)
    args = parser.parse_args()

    _COORD_URL = args.coord_url
    _RISK_URL  = args.risk_url
    _OBS_URL   = args.obs_url

    use_rich = (not args.no_rich)
    db_path  = Path(args.db)

    try:
        asyncio.run(run_monitor(args.interval, db_path, use_rich))
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
