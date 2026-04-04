"""
cli.py — Click CLI for execution_research
==========================================

Commands
--------
  tca analyze   : Run full TCA on a trades database
  tca report    : Generate HTML report
  split plan    : Generate an order-splitting schedule
  impact calibrate : Calibrate impact model from trade history
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


def _load_trades_db(db_path: str):
    """Load trades from SQLite and return list[Trade]."""
    from .tca import TCAAnalyzer
    return TCAAnalyzer.load_trades_from_db(db_path)


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------

@click.group()
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """srfm-lab Execution Research CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# tca group
# ---------------------------------------------------------------------------

@cli.group("tca")
def tca_group() -> None:
    """Transaction Cost Analysis commands."""


@tca_group.command("analyze")
@click.option(
    "--trades",
    "trades_db",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to live_trades.db SQLite database",
)
@click.option(
    "--output",
    "output_path",
    default="tca_report.html",
    show_default=True,
    help="Output HTML report path",
)
@click.option(
    "--json-output",
    "json_output",
    default=None,
    help="Optional JSON output path for machine-readable results",
)
@click.option(
    "--n-buckets",
    default=5,
    show_default=True,
    help="Number of size buckets for size-vs-cost analysis",
)
@click.pass_context
def tca_analyze(
    ctx: click.Context,
    trades_db: str,
    output_path: str,
    json_output: str | None,
    n_buckets: int,
) -> None:
    """
    Run full TCA on a trades database and generate HTML report.

    Example:

        tca analyze --trades live_trades.db --output tca_report.html
    """
    from .tca import TCAAnalyzer, generate_html_report

    click.echo(f"Loading trades from {trades_db} ...")
    trades = _load_trades_db(trades_db)
    click.echo(f"  → Loaded {len(trades)} trades")

    if not trades:
        click.echo("No trades found. Exiting.", err=True)
        sys.exit(1)

    analyzer = TCAAnalyzer()
    click.echo("Running TCA ...")
    portfolio_report = analyzer.analyze_portfolio(trades)

    click.echo(
        f"\n{'='*50}\n"
        f"  Trades analyzed:        {portfolio_report.n_trades}\n"
        f"  Total notional:         ${portfolio_report.total_notional:,.0f}\n"
        f"  Mean IS:                {portfolio_report.mean_is_bps:.2f} bps\n"
        f"  Median IS:              {portfolio_report.median_is_bps:.2f} bps\n"
        f"  P95 IS:                 {portfolio_report.p95_is_bps:.2f} bps\n"
        f"  Mean market impact:     {portfolio_report.mean_market_impact_bps:.2f} bps\n"
        f"  Mean spread cost:       {portfolio_report.mean_spread_cost_bps:.2f} bps\n"
        f"  Total cost:             ${portfolio_report.total_cost_dollars:,.0f}\n"
        f"  NW total cost:          {portfolio_report.notional_weighted_total_cost_bps:.2f} bps\n"
        f"{'='*50}"
    )

    click.echo(f"\nGenerating HTML report → {output_path}")
    generate_html_report(portfolio_report, output_path)
    click.echo("Done.")

    if json_output:
        # Build JSON-serializable dict
        data = {
            "n_trades": portfolio_report.n_trades,
            "total_notional": portfolio_report.total_notional,
            "mean_is_bps": portfolio_report.mean_is_bps,
            "median_is_bps": portfolio_report.median_is_bps,
            "p95_is_bps": portfolio_report.p95_is_bps,
            "mean_market_impact_bps": portfolio_report.mean_market_impact_bps,
            "mean_spread_cost_bps": portfolio_report.mean_spread_cost_bps,
            "total_cost_dollars": portfolio_report.total_cost_dollars,
            "notional_weighted_total_cost_bps": portfolio_report.notional_weighted_total_cost_bps,
            "by_sym": {
                sym: {
                    "n_trades": s.n_trades,
                    "total_notional": s.total_notional,
                    "mean_is_bps": s.mean_is_bps,
                    "mean_total_cost_bps": s.mean_total_cost_bps,
                    "total_cost_dollars": s.total_cost_dollars,
                }
                for sym, s in portfolio_report.by_sym.items()
            },
        }
        Path(json_output).write_text(json.dumps(data, indent=2))
        click.echo(f"JSON output written to {json_output}")


@tca_group.command("report")
@click.option(
    "--trades",
    "trades_db",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to live_trades.db",
)
@click.option(
    "--output",
    "output_path",
    default="tca_report.html",
    show_default=True,
)
@click.option("--plots-dir", default="tca_plots", show_default=True)
def tca_report(trades_db: str, output_path: str, plots_dir: str) -> None:
    """
    Generate full TCA report including plots.

    Example:

        tca report --trades live_trades.db --output tca_report.html --plots-dir ./plots
    """
    from .tca import TCAAnalyzer, generate_html_report

    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    trades = _load_trades_db(trades_db)
    if not trades:
        click.echo("No trades found.", err=True)
        sys.exit(1)

    analyzer = TCAAnalyzer()
    portfolio_report = analyzer.analyze_portfolio(trades)

    # Generate plots
    is_plot = plots_path / "is_distribution.png"
    size_plot = plots_path / "cost_vs_size.png"
    analyzer.plot_is_distribution(trades, is_plot)
    analyzer.plot_cost_vs_size(trades, size_plot)
    click.echo(f"Plots saved to {plots_dir}/")

    generate_html_report(portfolio_report, output_path)
    click.echo(f"Full report written to {output_path}")


# ---------------------------------------------------------------------------
# split group
# ---------------------------------------------------------------------------

@cli.group("split")
def split_group() -> None:
    """Order splitting and scheduling commands."""


@split_group.command("plan")
@click.option("--sym", required=True, help="Symbol to trade, e.g. BTC/USD")
@click.option("--side", required=True, type=click.Choice(["buy", "sell"]), help="Order side")
@click.option(
    "--notional",
    required=True,
    type=float,
    help="Total notional in USD",
)
@click.option(
    "--method",
    default="twap",
    show_default=True,
    type=click.Choice(["even", "twap", "vwap", "almgren_chriss"]),
    help="Scheduling method",
)
@click.option(
    "--n-slices",
    default=10,
    show_default=True,
    help="Number of slices (for TWAP/A-C)",
)
@click.option(
    "--interval",
    default=5.0,
    show_default=True,
    help="Interval between slices in minutes (TWAP)",
)
@click.option(
    "--horizon",
    default=1.0,
    show_default=True,
    help="Trading horizon in days (A-C)",
)
@click.option(
    "--risk-aversion",
    default=1e-6,
    show_default=True,
    help="Risk aversion lambda (A-C)",
)
@click.option(
    "--sigma",
    default=0.02,
    show_default=True,
    help="Daily volatility estimate (A-C)",
)
@click.option("--price", default=None, type=float, help="Current price estimate")
@click.option("--adv", default=None, type=float, help="Average daily volume in USD")
@click.option("--json", "json_out", is_flag=True, default=False, help="Output JSON")
def split_plan(
    sym: str,
    side: str,
    notional: float,
    method: str,
    n_slices: int,
    interval: float,
    horizon: float,
    risk_aversion: float,
    sigma: float,
    price: float | None,
    adv: float | None,
    json_out: bool,
) -> None:
    """
    Generate an order execution schedule.

    Example:

        split plan --sym BTC/USD --side buy --notional 500000 --method twap --n-slices 5

        split plan --sym AAPL --side sell --notional 1000000 --method almgren_chriss
                   --horizon 0.5 --sigma 0.018 --n-slices 12
    """
    from .order_splitting import OrderSplitter

    splitter = OrderSplitter()

    click.echo(
        f"\nPlanning {method.upper()} schedule: {side.upper()} {sym} ${notional:,.0f}\n"
    )

    if method == "even":
        orders = splitter.split_order(
            sym=sym,
            side=side,
            target_notional=notional,
            interval_minutes=interval,
        )
        if json_out:
            data = [
                {
                    "slice": o.slice_index,
                    "schedule_time": o.schedule_time.isoformat(),
                    "notional": o.notional,
                }
                for o in orders
            ]
            click.echo(json.dumps(data, indent=2))
        else:
            click.echo(f"{'Slice':>6} {'Time':>22} {'Notional':>14}")
            click.echo("-" * 50)
            for o in orders:
                click.echo(
                    f"{o.slice_index:>6d} "
                    f"{o.schedule_time.strftime('%Y-%m-%d %H:%M:%S'):>22} "
                    f"${o.notional:>12,.0f}"
                )

    elif method == "twap":
        qty = notional / price if price else notional  # if no price, treat as qty
        scheduled = splitter.twap_schedule(
            sym=sym,
            side=side,
            total_qty=qty,
            n_slices=n_slices,
            interval_minutes=interval,
            price_estimate=price,
            adv=adv,
        )
        if json_out:
            _print_schedule_json(scheduled)
        else:
            OrderSplitter.print_schedule(scheduled)

    elif method == "vwap":
        from .order_splitting import DEFAULT_EQUITY_VOLUME_PROFILE, DEFAULT_CRYPTO_VOLUME_PROFILE
        qty = notional / price if price else notional
        vol_profile = (
            DEFAULT_CRYPTO_VOLUME_PROFILE
            if "/" in sym or sym in ("BTC", "ETH", "SOL")
            else DEFAULT_EQUITY_VOLUME_PROFILE
        )
        scheduled = splitter.vwap_schedule(
            sym=sym,
            side=side,
            total_qty=qty,
            volume_profile=vol_profile,
            price_estimate=price,
        )
        if json_out:
            _print_schedule_json(scheduled)
        else:
            OrderSplitter.print_schedule(scheduled)

    elif method == "almgren_chriss":
        qty = notional / price if price else notional
        scheduled = splitter.almgren_chriss_schedule(
            sym=sym,
            side=side,
            total_qty=qty,
            T=horizon,
            n_slices=n_slices,
            risk_aversion=risk_aversion,
            sigma=sigma,
            price_estimate=price,
        )
        if json_out:
            _print_schedule_json(scheduled)
        else:
            OrderSplitter.print_schedule(scheduled)

    n = len(scheduled) if method != "even" else len(orders)  # type: ignore[possibly-undefined]
    click.echo(f"\nTotal: {n} order slices generated.")


def _print_schedule_json(scheduled) -> None:
    """Print a ScheduledOrder list as JSON."""
    data = [
        {
            "slice": s.order.slice_index,
            "expected_time": s.expected_time.isoformat(),
            "qty": s.order.qty,
            "notional": s.order.notional,
            "expected_impact_bps": s.expected_impact_bps,
            "cumulative_pct": s.cumulative_qty_pct,
        }
        for s in scheduled
    ]
    click.echo(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# impact group
# ---------------------------------------------------------------------------

@cli.group("impact")
def impact_group() -> None:
    """Market impact calibration commands."""


@impact_group.command("calibrate")
@click.option(
    "--trades",
    "trades_csv",
    required=True,
    type=click.Path(exists=True),
    help="CSV with columns: participation_rate, actual_impact_bps, [daily_vol]",
)
@click.option(
    "--model",
    default="sqrt",
    show_default=True,
    type=click.Choice(["linear", "sqrt", "almgren_chriss"]),
)
@click.option("--plot-dir", default=None, help="Directory for calibration plots")
@click.option("--adv", default=1_000_000.0, show_default=True, help="ADV for size plot ($)")
def impact_calibrate(
    trades_csv: str,
    model: str,
    plot_dir: str | None,
    adv: float,
) -> None:
    """
    Calibrate a market impact model from trade CSV data.

    The CSV must have at minimum:
      participation_rate, actual_impact_bps

    Optionally: daily_vol, notional, adv

    Example:

        impact calibrate --trades fills.csv --model sqrt --plot-dir ./plots
    """
    import pandas as pd
    from .market_impact import MarketImpactCalibrator

    df = pd.read_csv(trades_csv)
    click.echo(f"Loaded {len(df)} trades from {trades_csv}")

    calibrator = MarketImpactCalibrator()

    if model == "linear":
        alpha, beta = calibrator.calibrate_linear(df)
        click.echo(f"\nLinear model: impact = {alpha:.2f} + {beta:.2f} × (Q/V)  bps")
    elif model == "sqrt":
        alpha, beta = calibrator.calibrate_sqrt(df)
        click.echo(f"\nSqrt model: impact = {alpha:.2f} + {beta:.2f} × sqrt(Q/V)  bps")
    elif model == "almgren_chriss":
        eta, gamma = calibrator.calibrate_almgren_chriss(df)
        click.echo(f"\nA-C model: eta = {eta:.4f}, gamma = {gamma:.4f}")

    m = calibrator.get_model(model)
    click.echo(f"R² = {m.r_squared:.3f}  |  n_obs = {m.n_observations}  |  residual_std = {m.residual_std:.2f} bps")

    if plot_dir:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        import numpy as np
        participation_rates = df["participation_rate"].values
        if model == "linear":
            predicted = m.alpha + m.beta * participation_rates
        elif model == "sqrt":
            predicted = m.alpha + m.beta * np.sqrt(np.maximum(participation_rates, 0))
        else:
            sigma = df.get("daily_vol", pd.Series([0.02] * len(df))).values
            predicted = (m.eta * np.sqrt(np.maximum(participation_rates, 0)) + m.gamma * participation_rates) * sigma * 10_000

        calibrator.plot_impact_calibration(
            df["actual_impact_bps"].values,
            predicted,
            save_path=f"{plot_dir}/calibration_{model}.png",
            title=f"{model.title()} Impact Model Calibration",
        )
        calibrator.plot_impact_vs_size(
            m,
            adv=adv,
            save_path=f"{plot_dir}/impact_vs_size_{model}.png",
        )
        click.echo(f"Plots saved to {plot_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
