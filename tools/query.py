"""
query.py — DuckDB + Polars analytics CLI for SRFM trading lab.

Commands:
    query   Run arbitrary SQL
    ask     Natural-language query (maps to SQL)
    profile Full analytics report
    schema  List registered tables and columns
    export  Export a table to parquet/csv/json
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Optional

import io
import click
import duckdb
import polars as pl
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Ensure repo root is on sys.path so `tools.analytics` resolves
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.analytics import (
    _available_tables,
    available_tables,
    get_db,
    profile_convergence_edge,
    query_to_polars,
)

import sys
# Force UTF-8 output on Windows to avoid cp1252 encoding errors with box chars
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console(file=sys.stdout, force_terminal=True, highlight=False)

# ---------------------------------------------------------------------------
# Rich table rendering helpers
# ---------------------------------------------------------------------------

_NUMERIC_TYPES = {
    pl.Float32, pl.Float64,
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}
_DATE_TYPES = {pl.Date, pl.Datetime, pl.Time, pl.Duration}


def _col_icon(dtype: pl.DataType) -> str:
    if type(dtype) in _NUMERIC_TYPES:
        return "[#]"
    if type(dtype) in _DATE_TYPES:
        return "[D]"
    return "[T]"


def _fmt_value(val, dtype: pl.DataType) -> Text:
    """Format a single cell value with colour for P&L columns."""
    if val is None:
        return Text("null", style="dim")

    is_numeric = type(dtype) in _NUMERIC_TYPES

    if is_numeric:
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return Text(str(val))

        # Colour sign for float columns that look like P&L / returns
        formatted = f"{fval:,.4f}" if abs(fval) < 1000 else f"{fval:,.2f}"
        if fval > 0:
            return Text(formatted, style="bold green")
        if fval < 0:
            return Text(formatted, style="bold red")
        return Text(formatted)

    return Text(str(val))


def render_df(df: pl.DataFrame, title: str = "", max_rows: int = 200) -> None:
    """Render a Polars DataFrame as a rich Table."""
    if df.is_empty():
        console.print("[yellow]No rows returned.[/yellow]")
        return

    t = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        row_styles=["", "dim"],
        show_footer=False,
    )

    for col, dtype in zip(df.columns, df.dtypes):
        icon = _col_icon(dtype)
        t.add_column(f"{icon} {col}", no_wrap=False)

    display = df.head(max_rows)
    for row in display.iter_rows():
        cells = [
            _fmt_value(val, dtype)
            for val, dtype in zip(row, df.dtypes)
        ]
        t.add_row(*cells)

    console.print(t)
    if len(df) > max_rows:
        console.print(f"[dim]... {len(df) - max_rows} more rows not shown[/dim]")
    console.print(f"[dim]{len(df)} row(s)[/dim]")


def _run_sql(sql: str, label: str = "") -> Optional[pl.DataFrame]:
    """Execute SQL, time it, render result. Returns DataFrame or None on error."""
    con = get_db()
    tables = _available_tables(con)

    t0 = time.perf_counter()
    try:
        df = con.execute(sql).pl()
    except Exception as exc:
        console.print(Panel(
            f"[bold red]SQL Error:[/bold red] {exc}\n\n"
            f"[bold]Query:[/bold]\n[yellow]{sql}[/yellow]\n\n"
            f"[bold]Available tables:[/bold] {', '.join(tables) or 'none'}",
            title="Error",
            border_style="red",
        ))
        return None
    elapsed = time.perf_counter() - t0

    title = f"{label}  [dim]({elapsed*1000:.1f} ms)[/dim]" if label else f"[dim]{elapsed*1000:.1f} ms[/dim]"
    render_df(df, title=title)
    return df


# ---------------------------------------------------------------------------
# Natural-language → SQL mapping
# ---------------------------------------------------------------------------

def _nl_to_sql(question: str) -> Optional[str]:
    """Map a natural language question to SQL. Returns None if no pattern matches."""
    q = question.lower().strip()

    # Extract number from "top N"
    top_n_match = re.search(r"top\s+(\d+)", q)
    top_n = int(top_n_match.group(1)) if top_n_match else 10

    if re.search(r"convergence wells.*2024|wells.*convergence.*2024|2024.*convergence", q):
        return (
            "SELECT * FROM qc_wells "
            "WHERE YEAR(TRY_CAST(date AS TIMESTAMP)) = 2024 "
            "ORDER BY date DESC"
        )

    if re.search(r"best regime|regime pnl|pnl by regime|regime.*pnl", q):
        return (
            "SELECT regime, COUNT(*) as trade_count, "
            "AVG(pnl) as avg_pnl, SUM(pnl) as total_pnl "
            "FROM trades "
            "GROUP BY regime ORDER BY avg_pnl DESC"
        )

    if re.search(r"regime distribution|time in regime|regime.*time|regime.*count", q):
        return (
            "SELECT regime, COUNT(*) as bars, "
            "ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct "
            "FROM regimes GROUP BY regime ORDER BY bars DESC"
        )

    if re.search(r"top.*wells|best wells|wells.*rank", q):
        return (
            f"SELECT * FROM qc_wells ORDER BY total_pnl DESC LIMIT {top_n}"
        )

    if re.search(r"win rate by|win_rate by|winrate by", q):
        col_match = re.search(r"win.?rate by (\w+)", q)
        col = col_match.group(1) if col_match else "regime"
        return (
            f"SELECT {col}, "
            f"COUNT(*) as trade_count, "
            f"ROUND(100.0 * SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate_pct, "
            f"AVG(pnl) as avg_pnl "
            f"FROM trades GROUP BY {col} ORDER BY win_rate_pct DESC"
        )

    if re.search(r"show experiments|list experiments|experiments.*score|top experiments", q):
        return (
            f"SELECT exp, combined_score, arena_sharpe, synth_sharpe, arena_return, "
            f"arena_dd, overfit "
            f"FROM experiments ORDER BY combined_score DESC LIMIT {top_n}"
        )

    if re.search(r"tournament top|top.*tournament|leaderboard", q):
        return (
            f"SELECT * FROM tournament ORDER BY sharpe DESC LIMIT {top_n}"
        )

    if re.search(r"flat periods|inactive|duration.*long|long.*duration", q):
        threshold = 50
        t_match = re.search(r"duration.*?(\d+)|(\d+).*?duration", q)
        if t_match:
            threshold = int(t_match.group(1) or t_match.group(2))
        return (
            f"SELECT * FROM trades WHERE duration > {threshold} ORDER BY duration DESC"
        )

    if re.search(r"trades by year|year.*trade|trade count.*year", q):
        return (
            "SELECT YEAR(TRY_CAST(date AS TIMESTAMP)) as year, "
            "COUNT(*) as trade_count, "
            "SUM(pnl) as total_pnl, AVG(pnl) as avg_pnl "
            "FROM trades GROUP BY year ORDER BY year"
        )

    if re.search(r"bh.?count|black hole count|bh formation", q):
        return (
            "SELECT bh_count, COUNT(*) as n, "
            "ROUND(100.0 * SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate_pct, "
            "AVG(pnl) as avg_pnl "
            "FROM trades GROUP BY bh_count ORDER BY bh_count"
        )

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """SRFM Analytics — DuckDB + Polars query engine."""
    pass


@cli.command("query")
@click.argument("sql")
def cmd_query(sql: str):
    """Run arbitrary SQL against registered tables."""
    console.print(Panel(f"[bold cyan]{sql}[/bold cyan]", title="SQL", border_style="cyan"))
    _run_sql(sql, label="Result")


@cli.command("ask")
@click.argument("question")
def cmd_ask(question: str):
    """Ask a natural-language question; maps to SQL automatically."""
    sql = _nl_to_sql(question)

    if sql is None:
        tables = available_tables()
        console.print(Panel(
            f"[yellow]No pattern matched for:[/yellow] \"{question}\"\n\n"
            f"[bold]Available tables:[/bold] {', '.join(tables) or 'none'}\n\n"
            "[bold]Try:[/bold]\n"
            "  • best regime\n"
            "  • top 5 wells\n"
            "  • win rate by ticker\n"
            "  • show experiments\n"
            "  • tournament top 10\n"
            "  • trades by year\n"
            "  • bh_count analysis\n"
            "  • regime distribution",
            title="No match",
            border_style="yellow",
        ))
        return

    console.print(Panel(f"[bold cyan]{sql}[/bold cyan]", title="Generated SQL", border_style="cyan"))
    _run_sql(sql, label=f'"{question}"')


@cli.command("schema")
def cmd_schema():
    """List all registered tables with columns and row counts."""
    con = get_db()
    tables = _available_tables(con)

    if not tables:
        console.print("[yellow]No tables registered. Check that source files exist.[/yellow]")
        return

    console.print(Panel(
        f"[bold green]{len(tables)} table(s) registered[/bold green]",
        title="SRFM Analytics — Schema",
        border_style="green",
    ))

    for tname in sorted(tables):
        try:
            info = con.execute(
                f"SELECT column_name, data_type "
                f"FROM information_schema.columns "
                f"WHERE table_name='{tname}' AND table_schema='main'"
            ).fetchall()

            count_row = con.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()
            row_count = count_row[0] if count_row else "?"

            t = Table(
                title=f"[bold]{tname}[/bold]  [dim]({row_count} rows)[/dim]",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold magenta",
            )
            t.add_column("Column")
            t.add_column("Type", style="dim")

            for col_name, col_type in info:
                icon = "[#]" if any(x in col_type.upper() for x in ("INT","FLOAT","DOUBLE","DECIMAL","NUMERIC","HUGE")) else \
                       "[D]" if any(x in col_type.upper() for x in ("DATE","TIME","STAMP")) else "[T]"
                t.add_row(f"{icon} {col_name}", col_type)

            console.print(t)
        except Exception as exc:
            console.print(f"[red]Error reading schema for {tname}: {exc}[/red]")


@cli.command("profile")
def cmd_profile():
    """Run full analytics profile report."""
    con = get_db()
    tables = _available_tables(con)

    console.print(Panel(
        "[bold]SRFM Lab — Analytics Profile Report[/bold]",
        border_style="blue",
    ))

    # 1. Regime distribution
    if "regimes" in tables:
        console.rule("[bold cyan]1. Regime Distribution[/bold cyan]")
        _run_sql(
            "SELECT regime, COUNT(*) as bars, "
            "ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct_time, "
            "AVG(confidence) as avg_confidence "
            "FROM regimes GROUP BY regime ORDER BY bars DESC",
            label="Regime Distribution",
        )
    else:
        console.print("[dim]Skipping regime distribution — table not available[/dim]")

    # 2. BH formation rate by ticker
    if "trades" in tables:
        console.rule("[bold cyan]2. BH Formation Rate by Ticker[/bold cyan]")
        _run_sql(
            "SELECT ticker, COUNT(*) as trades, "
            "ROUND(100.0 * SUM(CASE WHEN bh_count > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as bh_rate_pct, "
            "AVG(bh_count) as avg_bh_count "
            "FROM trades GROUP BY ticker ORDER BY bh_rate_pct DESC",
            label="BH Formation by Ticker",
        )
    else:
        console.print("[dim]Skipping BH formation — trades table not available[/dim]")

    # 3. Convergence edge: bh_count buckets
    if "trades" in tables:
        console.rule("[bold cyan]3. Convergence Edge (bh_count Buckets)[/bold cyan]")
        try:
            edge_df = profile_convergence_edge()
            render_df(edge_df, title="Convergence Edge by BH Count")
        except Exception as exc:
            console.print(f"[red]Error computing convergence edge: {exc}[/red]")

    # 4. Best/worst 5 wells
    if "wells_es" in tables:
        console.rule("[bold cyan]4. Best 5 Wells[/bold cyan]")
        _run_sql(
            "SELECT formed_at, direction, mass_peak, duration_bars, "
            "hawking_temp, price_move_pct "
            "FROM wells_es ORDER BY mass_peak DESC LIMIT 5",
            label="Top 5 Wells by Mass",
        )
        console.rule("[bold cyan]4b. Worst 5 Wells (Smallest Mass)[/bold cyan]")
        _run_sql(
            "SELECT formed_at, direction, mass_peak, duration_bars, "
            "hawking_temp, price_move_pct "
            "FROM wells_es ORDER BY mass_peak ASC LIMIT 5",
            label="Bottom 5 Wells by Mass",
        )
    else:
        console.print("[dim]Skipping wells — table not available[/dim]")

    # 5. Trade count by year
    if "trades" in tables:
        console.rule("[bold cyan]5. Trade Count by Year[/bold cyan]")
        _run_sql(
            "SELECT YEAR(TRY_CAST(date AS TIMESTAMP)) as year, "
            "COUNT(*) as trade_count, "
            "ROUND(SUM(pnl), 4) as total_pnl, "
            "ROUND(AVG(pnl), 6) as avg_pnl "
            "FROM trades GROUP BY year ORDER BY year",
            label="Trades by Year",
        )

    # 6. Sharpe estimate from trade returns (via Polars lazy)
    if "trades" in tables:
        console.rule("[bold cyan]6. Sharpe Estimate from Trade Returns[/bold cyan]")
        try:
            pnl_df = con.execute("SELECT pnl FROM trades").pl()
            pnl = pnl_df["pnl"].drop_nulls()
            mean_r = float(pnl.mean())
            std_r = float(pnl.std())
            sharpe = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0.0
            win_rate = float((pnl > 0).mean()) * 100

            t = Table(box=box.ROUNDED, header_style="bold cyan", title="Return Statistics")
            t.add_column("Metric")
            t.add_column("Value", justify="right")
            rows = [
                ("Trade count", f"{len(pnl):,}"),
                ("Mean return", f"{mean_r:.6f}"),
                ("Std return", f"{std_r:.6f}"),
                ("Annualised Sharpe (252 trades/yr)", f"{sharpe:.3f}"),
                ("Win rate", f"{win_rate:.2f}%"),
                ("Total P&L", f"{float(pnl.sum()):.4f}"),
            ]
            for metric, value in rows:
                t.add_row(metric, value)
            console.print(t)
        except Exception as exc:
            console.print(f"[red]Error computing Sharpe: {exc}[/red]")

    # 7. Top experiments
    if "experiments" in tables:
        console.rule("[bold cyan]7. Top Experiments by Combined Score[/bold cyan]")
        _run_sql(
            "SELECT exp, combined_score, arena_sharpe, synth_sharpe, "
            "arena_return, arena_dd, overfit "
            "FROM experiments ORDER BY combined_score DESC LIMIT 10",
            label="Top Experiments",
        )

    console.print(Panel("[bold green]Profile complete.[/bold green]", border_style="green"))


@cli.command("export")
@click.option("--table", required=True, help="Table name to export")
@click.option("--format", "fmt", default="parquet",
              type=click.Choice(["parquet", "csv", "json"]),
              help="Output format")
@click.option("--out", required=True, help="Output file path")
def cmd_export(table: str, fmt: str, out: str):
    """Export a table to parquet, csv, or json using Polars."""
    con = get_db()
    tables = _available_tables(con)

    if table not in tables:
        console.print(
            f"[red]Table '{table}' not found.[/red] "
            f"Available: {', '.join(tables) or 'none'}"
        )
        sys.exit(1)

    t0 = time.perf_counter()
    df = con.execute(f"SELECT * FROM {table}").pl()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        df.write_parquet(str(out_path))
    elif fmt == "csv":
        df.write_csv(str(out_path))
    elif fmt == "json":
        df.write_json(str(out_path))

    elapsed = time.perf_counter() - t0
    console.print(
        f"[green]Exported[/green] {len(df):,} rows from "
        f"[bold]{table}[/bold] → [bold]{out_path}[/bold] "
        f"[dim]({fmt}, {elapsed*1000:.1f} ms)[/dim]"
    )


if __name__ == "__main__":
    cli()
