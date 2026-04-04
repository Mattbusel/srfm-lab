"""
cli.py — Click CLI for live_monitor
=====================================

Commands
--------
  monitor run       : Start the live terminal dashboard
  monitor diagnose  : Print a full diagnostic report to stdout
  monitor report    : Generate a markdown/JSON report file
  monitor alerts    : Show current alerts
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click

logger = logging.getLogger(__name__)

DEFAULT_DB = "live_trades.db"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------

@click.group()
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """srfm-lab Live Monitor CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@cli.group("monitor")
def monitor_group() -> None:
    """Live monitor commands."""


# ---------------------------------------------------------------------------
# monitor run — starts dashboard
# ---------------------------------------------------------------------------

@monitor_group.command("run")
@click.option(
    "--db",
    "db_path",
    default=DEFAULT_DB,
    show_default=True,
    help="Path to live_trades.db",
)
@click.option(
    "--interval",
    default=60,
    show_default=True,
    type=int,
    help="Refresh interval in seconds",
)
@click.option(
    "--plain",
    is_flag=True,
    default=False,
    help="Force plain text output (no rich)",
)
def monitor_run(db_path: str, interval: int, plain: bool) -> None:
    """
    Start the live terminal dashboard.

    Example:

        monitor run --db /data/live_trades.db --interval 30
    """
    if plain:
        # Monkey-patch to disable rich
        import research.live_monitor.dashboard as dash_mod
        dash_mod.RICH_AVAILABLE = False

    from .dashboard import LiveDashboard

    click.echo(f"Starting live monitor dashboard (DB: {db_path}, interval: {interval}s)")
    click.echo("Press Ctrl+C to exit.\n")

    dashboard = LiveDashboard(db_path=db_path, refresh_interval=interval)
    try:
        dashboard.run()
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped.")


# ---------------------------------------------------------------------------
# monitor diagnose — prints diagnostic report
# ---------------------------------------------------------------------------

@monitor_group.command("diagnose")
@click.option(
    "--db",
    "db_path",
    default=DEFAULT_DB,
    show_default=True,
    help="Path to live_trades.db",
)
@click.option(
    "--days",
    default=90,
    show_default=True,
    type=int,
    help="Lookback days for analysis",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    help="Save JSON report to this path (optional)",
)
@click.option(
    "--delta-max-frac",
    default=0.20,
    show_default=True,
    type=float,
    help="Max single-position fraction for concentration alert",
)
def monitor_diagnose(
    db_path: str,
    days: int,
    output_path: str | None,
    delta_max_frac: float,
) -> None:
    """
    Run all diagnostics and print a structured report.

    Example:

        monitor diagnose --db live_trades.db --days 60

        monitor diagnose --db live_trades.db --output diag_report.json
    """
    from .diagnostics import LiveDiagnostics

    if not Path(db_path).exists():
        click.echo(f"ERROR: DB not found: {db_path}", err=True)
        sys.exit(1)

    click.echo(f"Running diagnostics on {db_path} (last {days} days) ...\n")
    diag = LiveDiagnostics(db_path=db_path, delta_max_frac=delta_max_frac)
    report = diag.full_diagnostic_report(days=days)

    _print_diagnostic_report(report)

    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2, default=str))
        click.echo(f"\nReport saved to {output_path}")


def _print_diagnostic_report(report: dict) -> None:
    """Pretty-print the diagnostic report."""
    sep = "=" * 60

    click.echo(f"{sep}")
    click.echo("  ORDER FAILURES")
    click.echo(sep)
    fail = report.get("failures", {})
    if "error" in fail:
        click.echo(f"  [ERROR] {fail['error']}")
    else:
        click.echo(f"  Total failures:          {fail.get('n_total', 0)}")
        click.echo(f"  Notional violation rate: {fail.get('notional_violation_rate', 0)*100:.1f}%")
        click.echo(f"  Most common reason:      {fail.get('most_common', 'n/a')}")
        for reason, count in (fail.get("by_reason") or {}).items():
            click.echo(f"    {reason:<30} {count}")

    click.echo(f"\n{sep}")
    click.echo("  SIGNAL QUALITY")
    click.echo(sep)
    sig = report.get("signal", {})
    if "error" in sig:
        click.echo(f"  [ERROR] {sig['error']}")
    else:
        ic = sig.get("ic", 0)
        p = sig.get("ic_p_value", 1)
        predictive = sig.get("is_predictive", False)
        flag = "✓ PREDICTIVE" if predictive else "✗ NOT PREDICTIVE"
        click.echo(f"  IC:                      {ic:.4f}")
        click.echo(f"  IC p-value:              {p:.4f}")
        click.echo(f"  Hit rate:                {sig.get('hit_rate', 0)*100:.1f}%")
        click.echo(f"  Status:                  {flag}")
        click.echo(f"  N signal trades:         {sig.get('n_trades', 0)}")

    click.echo(f"\n{sep}")
    click.echo("  REGIME EXPOSURE")
    click.echo(sep)
    reg = report.get("regime", {})
    if "error" in reg:
        click.echo(f"  [ERROR] {reg['error']}")
    else:
        click.echo(f"  Current regime:          {reg.get('current', 'UNKNOWN')}")
        click.echo(f"  HIGH_VOL fraction:       {reg.get('high_vol_fraction', 0)*100:.1f}%")
        click.echo(f"  Over-exposed HIGH_VOL:   {reg.get('is_over_exposed', False)}")
        for rec in (reg.get("recommendations") or []):
            click.echo(f"  → {rec}")

    click.echo(f"\n{sep}")
    click.echo("  ENSEMBLE USAGE")
    click.echo(sep)
    ens = report.get("ensemble", {})
    if "error" in ens:
        click.echo(f"  [ERROR] {ens['error']}")
    else:
        better = ens.get("ensemble_is_better", False)
        click.echo(f"  Ensemble Sharpe:         {ens.get('ensemble_sharpe', 0):.2f}")
        click.echo(f"  Single-model Sharpe:     {ens.get('single_model_sharpe', 0):.2f}")
        click.echo(f"  Best threshold:          {ens.get('best_threshold', 0.5):.2f}")
        click.echo(f"  Ensemble better:         {'Yes ✓' if better else 'No ✗'}")

    click.echo(f"\n{sep}")
    click.echo("  CONCENTRATION")
    click.echo(sep)
    conc = report.get("concentration", {})
    if "error" in conc:
        click.echo(f"  [ERROR] {conc['error']}")
    else:
        hhi = conc.get("current_hhi", 0)
        max_frac = conc.get("max_position_frac", 0)
        click.echo(f"  Current HHI:             {hhi:.3f}")
        click.echo(f"  Max position fraction:   {max_frac*100:.1f}%")
        for alert_msg in (conc.get("alerts") or []):
            click.echo(f"  → {alert_msg}")

    click.echo(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# monitor report — generate file report
# ---------------------------------------------------------------------------

@monitor_group.command("report")
@click.option("--db", "db_path", default=DEFAULT_DB, show_default=True)
@click.option("--output", "output_path", default="monitor_report.json", show_default=True)
@click.option("--days", default=30, show_default=True, type=int)
def monitor_report(db_path: str, output_path: str, days: int) -> None:
    """
    Generate a JSON health + performance report.

    Example:

        monitor report --db live_trades.db --output report.json --days 30
    """
    from .monitor import LiveTraderMonitor

    if not Path(db_path).exists():
        click.echo(f"ERROR: DB not found: {db_path}", err=True)
        sys.exit(1)

    monitor = LiveTraderMonitor(db_path)
    metrics = monitor.get_live_metrics(lookback_days=days)
    health = monitor.check_health(metrics)
    positions = monitor.get_current_positions()
    today_pnl = monitor.get_todays_pnl()

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "db_path": db_path,
        "lookback_days": days,
        "health": {
            "status": health.status,
            "alerts": health.alerts,
            "equity": health.equity,
            "current_drawdown_pct": health.current_drawdown_pct,
            "order_failure_rate": health.order_failure_rate,
            "n_open_positions": health.n_open_positions,
            "is_market_hours": health.is_market_hours,
        },
        "metrics": {
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "current_drawdown_pct": metrics.current_drawdown_pct,
            "win_rate": metrics.win_rate,
            "n_trades": metrics.n_trades,
            "total_pnl": metrics.total_pnl,
            "daily_pnl": metrics.daily_pnl,
            "weekly_pnl": metrics.weekly_pnl,
            "avg_trade_pnl": metrics.avg_trade_pnl,
            "best_trade_pnl": metrics.best_trade_pnl,
            "worst_trade_pnl": metrics.worst_trade_pnl,
            "order_failure_rate": metrics.order_failure_rate,
            "last_trade_time": metrics.last_trade_time.isoformat() if metrics.last_trade_time else None,
        },
        "positions": [
            {
                "sym": p.sym,
                "qty": p.qty,
                "avg_entry": p.avg_entry,
                "current_price": p.current_price,
                "unrealized_pnl": p.unrealized_pnl,
                "notional": p.notional,
                "pnl_pct": p.pnl_pct,
                "bh_active": p.bh_active,
            }
            for p in positions
        ],
        "today_pnl_by_sym": today_pnl,
    }

    Path(output_path).write_text(json.dumps(report, indent=2, default=str))
    click.echo(f"Report written to {output_path}")
    click.echo(f"  Status: {health.status}  |  Equity: ${health.equity:,.0f}")


# ---------------------------------------------------------------------------
# monitor alerts — show current alerts
# ---------------------------------------------------------------------------

@monitor_group.command("alerts")
@click.option("--db", "db_path", default=DEFAULT_DB, show_default=True)
@click.option("--days", default=30, show_default=True, type=int)
@click.option(
    "--min-severity",
    default="INFO",
    show_default=True,
    type=click.Choice(["INFO", "WARN", "CRITICAL"]),
)
def monitor_alerts(db_path: str, days: int, min_severity: str) -> None:
    """
    Show current alerts for the live trader.

    Example:

        monitor alerts --db live_trades.db --min-severity WARN
    """
    import pandas as pd
    from .monitor import LiveTraderMonitor
    from .alerts import AlertSystem, AlertConfig

    monitor = LiveTraderMonitor(db_path)
    metrics = monitor.get_live_metrics(lookback_days=days)
    health = monitor.check_health(metrics)
    positions = monitor.get_current_positions()
    equity_curve = monitor.get_equity_curve(days=days)

    alert_system = AlertSystem()

    # Load trades for frequency/streak checks
    cutoff = (datetime.utcnow() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    trades_df = monitor._query(
        "SELECT * FROM trades WHERE DATE(fill_time) >= ? ORDER BY fill_time",
        (cutoff,),
    )

    all_alerts = alert_system.run_all_checks(
        equity_curve=equity_curve,
        positions=positions,
        trades=trades_df,
        equity=health.equity,
        last_trade_time=metrics.last_trade_time,
    )

    severity_rank = {"INFO": 0, "WARN": 1, "CRITICAL": 2}
    min_rank = severity_rank.get(min_severity, 0)
    filtered = [a for a in all_alerts if severity_rank.get(a.severity, 0) >= min_rank]

    if not filtered:
        click.echo(f"No alerts at or above {min_severity} severity.")
        return

    click.echo(f"Alerts (>= {min_severity})  —  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")
    click.echo(f"{'Severity':<12} {'Check':<22} {'Message'}")
    click.echo("-" * 80)
    for a in filtered:
        click.echo(f"{a.severity:<12} {a.check_name:<22} {a.message}")
    click.echo(f"\nTotal: {len(filtered)} alert(s)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
