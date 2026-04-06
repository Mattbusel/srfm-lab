"""
tools/pnl_inspector.py
======================
Live P&L inspector — rich terminal dashboard reading execution/live_trades.db.

Usage:
    python tools/pnl_inspector.py               # snapshot
    python tools/pnl_inspector.py --watch        # refresh every 10s
    python tools/pnl_inspector.py --since 1h     # last 1 hour
    python tools/pnl_inspector.py --export csv   # dump to CSV
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

_DB = Path(__file__).parents[1] / "execution" / "live_trades.db"
console = Console() if HAS_RICH else None


# ── helpers ───────────────────────────────────────────────────────────────────

def _since_ts(since: str) -> str | None:
    """Convert '1h', '4h', '1d', or ISO string to UTC ISO string."""
    units = {"m": 60, "h": 3600, "d": 86400}
    for suffix, secs in units.items():
        if since.endswith(suffix):
            try:
                n = float(since[:-1])
                dt = datetime.now(timezone.utc) - timedelta(seconds=n * secs)
                return dt.isoformat()
            except ValueError:
                pass
    return since  # assume ISO string


def _pct(val: float, total: float) -> str:
    if total == 0:
        return "  n/a"
    return f"{val / total * 100:+5.1f}%"


def _color(val: float) -> str:
    if not HAS_RICH:
        return ""
    return "green" if val >= 0 else "red"


# ── data loading ──────────────────────────────────────────────────────────────

def load_summary(since_iso: str | None = None) -> dict:
    if not _DB.exists():
        return {}

    conn = sqlite3.connect(str(_DB))
    conn.row_factory = sqlite3.Row

    where = f"WHERE exit_time >= '{since_iso}'" if since_iso else ""

    # Overall P&L
    row = conn.execute(f"""
        SELECT
            COUNT(*)                                                       AS n_trades,
            COALESCE(SUM(pnl), 0)                                         AS total_pnl,
            COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0)      AS gross_win,
            COALESCE(SUM(CASE WHEN pnl <= 0 THEN pnl ELSE 0 END), 0)     AS gross_loss,
            COALESCE(COUNT(CASE WHEN pnl > 0 THEN 1 END), 0)             AS n_wins,
            COALESCE(AVG(hold_bars), 0)                                    AS avg_hold,
            COALESCE(MIN(pnl), 0)                                          AS worst,
            COALESCE(MAX(pnl), 0)                                          AS best
        FROM trade_pnl {where}
    """).fetchone()

    # Per-symbol breakdown
    sym_rows = conn.execute(f"""
        SELECT
            symbol,
            COUNT(*)                                                      AS n,
            ROUND(SUM(pnl), 2)                                            AS pnl,
            ROUND(AVG(pnl), 2)                                            AS avg_pnl,
            ROUND(100.0 * COUNT(CASE WHEN pnl > 0 THEN 1 END) / COUNT(*), 1) AS wr,
            ROUND(AVG(hold_bars), 1)                                      AS avg_hold,
            ROUND(MIN(pnl), 2)                                            AS worst,
            ROUND(MAX(pnl), 2)                                            AS best
        FROM trade_pnl {where}
        GROUP BY symbol
        ORDER BY pnl ASC
    """).fetchall()

    # Fill volume by symbol
    fill_where = f"WHERE fill_time >= '{since_iso}'" if since_iso else ""
    fill_rows = conn.execute(f"""
        SELECT symbol, side, COUNT(*) AS fills, ROUND(SUM(notional), 0) AS notional
        FROM live_trades {fill_where}
        GROUP BY symbol, side
        ORDER BY notional DESC
    """).fetchall()

    # Equity curve (last 200 points from live_trades timestamps)
    eq_rows = conn.execute(f"""
        SELECT fill_time, pnl
        FROM trade_pnl {where}
        ORDER BY exit_time
    """).fetchall()

    # Recent trades
    recent = conn.execute(f"""
        SELECT symbol, entry_time, exit_time, entry_price, exit_price, qty, pnl, hold_bars
        FROM trade_pnl {where}
        ORDER BY exit_time DESC
        LIMIT 20
    """).fetchall()

    conn.close()

    # Sharpe from daily P&L buckets
    sharpe = _compute_sharpe(eq_rows)

    return dict(
        n_trades   = row["n_trades"],
        total_pnl  = row["total_pnl"],
        gross_win  = row["gross_win"],
        gross_loss = row["gross_loss"],
        n_wins     = row["n_wins"],
        avg_hold   = row["avg_hold"],
        worst      = row["worst"],
        best       = row["best"],
        win_rate   = row["n_wins"] / max(row["n_trades"], 1) * 100,
        profit_fac = abs(row["gross_win"] / row["gross_loss"]) if row["gross_loss"] != 0 else float("inf"),
        sharpe     = sharpe,
        symbols    = [dict(r) for r in sym_rows],
        fills      = [dict(r) for r in fill_rows],
        recent     = [dict(r) for r in recent],
    )


def _compute_sharpe(eq_rows) -> float:
    if len(eq_rows) < 10:
        return 0.0
    pnls = [r["pnl"] for r in eq_rows]
    n = len(pnls)
    if n < 2:
        return 0.0
    mean = sum(pnls) / n
    var  = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std  = math.sqrt(var) if var > 0 else 1e-9
    # Annualise assuming ~390 trades/day (1-min bars, rough proxy)
    return round((mean / std) * math.sqrt(252 * 390), 3)


# ── rich display ──────────────────────────────────────────────────────────────

def _render(data: dict, since_label: str) -> None:
    if not data:
        console.print("[red]No database found or no trades yet.[/red]")
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pnl_color = "green" if data["total_pnl"] >= 0 else "red"

    # ── Summary panel ─────────────────────────────────────────────────────────
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan")
    summary.add_column()

    def fmt_money(v: float) -> Text:
        t = Text(f"${v:>+14,.2f}")
        t.stylize("green" if v >= 0 else "red")
        return t

    summary.add_row("Total Realized P&L",  fmt_money(data["total_pnl"]))
    summary.add_row("Gross Wins",           fmt_money(data["gross_win"]))
    summary.add_row("Gross Losses",         fmt_money(data["gross_loss"]))
    summary.add_row("Profit Factor",        f"{data['profit_fac']:.2f}x" if math.isfinite(data['profit_fac']) else "∞")
    summary.add_row("Win Rate",             f"{data['win_rate']:.1f}%  ({data['n_wins']}/{data['n_trades']})")
    summary.add_row("Sharpe (annualised)",  f"{data['sharpe']:.3f}")
    summary.add_row("Best Trade",           fmt_money(data["best"]))
    summary.add_row("Worst Trade",          fmt_money(data["worst"]))
    summary.add_row("Avg Hold",             f"{data['avg_hold']:.1f} min")

    # ── Per-symbol table ──────────────────────────────────────────────────────
    sym_table = Table(title="P&L by Symbol", show_header=True, header_style="bold magenta",
                      border_style="dim", show_lines=False)
    sym_table.add_column("Symbol",    style="bold", width=8)
    sym_table.add_column("Trades",    justify="right", width=7)
    sym_table.add_column("P&L",       justify="right", width=14)
    sym_table.add_column("Avg P&L",   justify="right", width=12)
    sym_table.add_column("Win%",      justify="right", width=7)
    sym_table.add_column("Avg Hold",  justify="right", width=9)
    sym_table.add_column("Worst",     justify="right", width=12)
    sym_table.add_column("Best",      justify="right", width=12)

    for s in data["symbols"]:
        c = "green" if s["pnl"] >= 0 else "red"
        sym_table.add_row(
            s["symbol"],
            str(s["n"]),
            f"[{c}]${s['pnl']:>+,.2f}[/{c}]",
            f"[{c}]${s['avg_pnl']:>+,.2f}[/{c}]",
            f"{s['wr']:.1f}%",
            f"{s['avg_hold']:.1f}m",
            f"[red]${s['worst']:>+,.2f}[/red]",
            f"[green]${s['best']:>+,.2f}[/green]",
        )

    # ── Fill volume table ─────────────────────────────────────────────────────
    fill_table = Table(title="Fill Volume by Symbol", header_style="bold blue",
                       border_style="dim", show_lines=False)
    fill_table.add_column("Symbol",    style="bold", width=8)
    fill_table.add_column("Side",      width=5)
    fill_table.add_column("Fills",     justify="right", width=8)
    fill_table.add_column("Notional",  justify="right", width=16)

    for f in data["fills"][:20]:  # top 20
        c = "green" if f["side"] == "buy" else "red"
        fill_table.add_row(
            f["symbol"],
            f"[{c}]{f['side'].upper()}[/{c}]",
            str(f["fills"]),
            f"${f['notional']:>,.0f}",
        )

    # ── Recent trades ─────────────────────────────────────────────────────────
    recent_table = Table(title="Recent Closed Trades (last 20)", header_style="bold yellow",
                         border_style="dim", show_lines=False)
    recent_table.add_column("Symbol",   width=8)
    recent_table.add_column("Entry px", justify="right", width=12)
    recent_table.add_column("Exit px",  justify="right", width=12)
    recent_table.add_column("Qty",      justify="right", width=14)
    recent_table.add_column("P&L",      justify="right", width=12)
    recent_table.add_column("Hold",     justify="right", width=7)
    recent_table.add_column("Exit time",               width=26)

    for t in data["recent"]:
        c = "green" if t["pnl"] >= 0 else "red"
        recent_table.add_row(
            t["symbol"],
            f"${t['entry_price']:.4f}",
            f"${t['exit_price']:.4f}",
            f"{t['qty']:.4f}",
            f"[{c}]${t['pnl']:>+,.2f}[/{c}]",
            f"{t['hold_bars']}m",
            str(t["exit_time"])[:25],
        )

    console.print()
    console.print(Panel(
        summary,
        title=f"[bold]LARSA v17 — P&L Inspector  [dim]{since_label}[/dim]  @ {now}[/bold]",
        border_style=pnl_color,
        padding=(1, 2),
    ))
    console.print()
    console.print(Columns([sym_table, fill_table], equal=False, expand=True))
    console.print()
    console.print(recent_table)
    console.print()


def _render_plain(data: dict, since_label: str) -> None:
    """Fallback for when rich is not installed."""
    print(f"\n=== LARSA P&L Inspector  [{since_label}] ===")
    print(f"Total P&L:     ${data['total_pnl']:>+14,.2f}")
    print(f"Profit Factor: {data['profit_fac']:.2f}x")
    print(f"Win Rate:      {data['win_rate']:.1f}%  ({data['n_wins']}/{data['n_trades']})")
    print(f"Sharpe:        {data['sharpe']:.3f}")
    print(f"Best:          ${data['best']:>+,.2f}")
    print(f"Worst:         ${data['worst']:>+,.2f}")
    print()
    print(f"{'Symbol':8s}  {'Trades':>6}  {'P&L':>14}  {'WR%':>6}  {'AvgHold':>8}")
    print("-" * 55)
    for s in data["symbols"]:
        print(f"{s['symbol']:8s}  {s['n']:>6}  ${s['pnl']:>13,.2f}  {s['wr']:>5.1f}%  {s['avg_hold']:>7.1f}m")


def _export_csv(data: dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "n", "pnl", "avg_pnl", "wr", "avg_hold", "worst", "best"])
        w.writeheader()
        w.writerows(data["symbols"])
    print(f"Exported to {path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="LARSA live P&L inspector")
    p.add_argument("--watch",  "-w", action="store_true",  help="Refresh every 10s")
    p.add_argument("--since",  "-s", default=None,         help="Time filter: 1h, 4h, 1d, or ISO timestamp")
    p.add_argument("--export",        default=None,         help="Export to: csv (path optional)")
    p.add_argument("--interval",      type=int, default=10, help="Watch interval seconds (default 10)")
    args = p.parse_args()

    since_iso   = _since_ts(args.since) if args.since else None
    since_label = args.since or "all time"

    render_fn = _render if HAS_RICH else _render_plain

    if args.export:
        path = args.export if args.export != "csv" else "pnl_export.csv"
        _export_csv(load_summary(since_iso), path)
        return

    if args.watch and HAS_RICH:
        try:
            while True:
                console.clear()
                render_fn(load_summary(since_iso), since_label)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped.[/dim]")
    else:
        render_fn(load_summary(since_iso), since_label)


if __name__ == "__main__":
    main()
