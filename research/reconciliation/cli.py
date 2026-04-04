"""
research/reconciliation/cli.py
================================
Click-based command-line interface for the live-vs-backtest reconciliation
pipeline.

Usage
-----
From the repo root::

    python -m research.reconciliation.cli --help

    # Full reconciliation run
    python -m research.reconciliation.cli run \\
        --live tools/backtest_output/live_trades.db \\
        --backtest tools/backtest_output/crypto_trades.csv \\
        --output research/reconciliation/output

    # Individual sub-commands
    python -m research.reconciliation.cli slippage --live ... --backtest ...
    python -m research.reconciliation.cli drift    --live ... --backtest ...
    python -m research.reconciliation.cli attribution --live ... --backtest ...
    python -m research.reconciliation.cli report   --input research/reconciliation/output

Or install the package and use `recon` as the entry-point command.

Command groups
--------------
recon run          – full pipeline run
recon slippage     – slippage analysis only
recon drift        – signal/regime drift only
recon attribution  – PnL attribution only
recon report       – generate HTML/console report from saved data
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("recon.cli")


# ── Rich helpers (optional) ───────────────────────────────────────────────────

def _try_rich():
    """Return (console, Progress) from rich if available, else (None, None)."""
    try:
        from rich.console import Console
        from rich.progress import (
            Progress, SpinnerColumn, TextColumn,
            BarColumn, TimeElapsedColumn,
        )
        return Console(), Progress
    except ImportError:
        return None, None


def _print_ok(msg: str) -> None:
    console, _ = _try_rich()
    if console:
        console.print(f"[bold green]✓[/] {msg}")
    else:
        click.echo(f"  OK  {msg}")


def _print_warn(msg: str) -> None:
    console, _ = _try_rich()
    if console:
        console.print(f"[bold yellow]⚠[/] {msg}")
    else:
        click.echo(f" WARN {msg}", err=True)


def _print_error(msg: str) -> None:
    console, _ = _try_rich()
    if console:
        console.print(f"[bold red]✗[/] {msg}")
    else:
        click.echo(f"  ERR {msg}", err=True)


def _print_header(title: str) -> None:
    console, _ = _try_rich()
    if console:
        console.rule(f"[bold cyan]{title}[/]")
    else:
        w = max(len(title) + 4, 60)
        click.echo("=" * w)
        click.echo(f"  {title}")
        click.echo("=" * w)


# ── Shared loader helper ──────────────────────────────────────────────────────

def _load_trades(live_path: str, backtest_path: str):
    """
    Load live and backtest trades from the provided paths.
    Returns (live_df, bt_df, merged_df).
    """
    import pandas as pd
    from research.reconciliation.loader import (
        LiveTradeLoader, BacktestTradeLoader,
        merge_live_backtest, _records_to_df,
    )

    _print_header("Loading Trades")

    # Live trades
    if live_path:
        lp = Path(live_path)
        if not lp.exists():
            _print_error(f"Live trades file not found: {lp}")
            sys.exit(1)
        try:
            live_loader = LiveTradeLoader(lp)
            live_records = live_loader.load()
            live_df = _records_to_df(live_records)
            _print_ok(f"Loaded {len(live_df)} live trades from {lp.name}")
        except Exception as exc:
            _print_error(f"Failed to load live trades: {exc}")
            sys.exit(1)
    else:
        live_df = pd.DataFrame()
        _print_warn("No live trades path provided; live DataFrame is empty.")

    # Backtest trades
    if backtest_path:
        bp = Path(backtest_path)
        if not bp.exists():
            _print_error(f"Backtest file not found: {bp}")
            sys.exit(1)
        try:
            bt_loader = BacktestTradeLoader(bp)
            bt_records = bt_loader.load()
            bt_df = _records_to_df(bt_records)
            _print_ok(f"Loaded {len(bt_df)} backtest trades from {bp.name}")
        except Exception as exc:
            _print_error(f"Failed to load backtest trades: {exc}")
            sys.exit(1)
    else:
        bt_df = pd.DataFrame()
        _print_warn("No backtest path provided; backtest DataFrame is empty.")

    # Merge
    merged = pd.DataFrame()
    if not live_df.empty and not bt_df.empty:
        try:
            merged = merge_live_backtest(live_df, bt_df)
            _print_ok(f"Merged frame: {len(merged)} rows")
        except Exception as exc:
            _print_warn(f"Trade merge failed: {exc}")

    return live_df, bt_df, merged


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.version_option(version="1.0.0", prog_name="recon")
def recon(verbose: bool) -> None:
    """
    srfm-lab Live vs Backtest Reconciliation Pipeline.

    Run 'recon COMMAND --help' for detailed usage of each sub-command.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Debug logging enabled.")


# ── recon run ─────────────────────────────────────────────────────────────────

@recon.command("run")
@click.option(
    "--live", "-l",
    default="tools/backtest_output/live_trades.db",
    show_default=True,
    help="Path to live_trades.db (SQLite).",
)
@click.option(
    "--backtest", "-b",
    default="tools/backtest_output/crypto_trades.csv",
    show_default=True,
    help="Path to backtest CSV or SQLite.",
)
@click.option(
    "--output", "-o",
    default="research/reconciliation/output",
    show_default=True,
    help="Output directory for all artefacts.",
)
@click.option(
    "--json-export", is_flag=True, default=False,
    help="Also write a JSON summary to <output>/reconciliation_summary.json.",
)
@click.option(
    "--no-html", is_flag=True, default=False,
    help="Skip HTML report generation.",
)
@click.option(
    "--no-plots", is_flag=True, default=False,
    help="Skip all chart generation (faster).",
)
@click.option(
    "--annual-factor", default=252.0, show_default=True,
    help="Annualisation factor (252 for daily, 8760 for hourly crypto).",
)
@click.option(
    "--dpi", default=150, show_default=True,
    help="Chart DPI / resolution.",
)
def run_command(
    live: str,
    backtest: str,
    output: str,
    json_export: bool,
    no_html: bool,
    no_plots: bool,
    annual_factor: float,
    dpi: int,
) -> None:
    """
    Run the full live-vs-backtest reconciliation pipeline.

    Performs: data loading → merge → slippage analysis → PnL attribution →
    signal drift detection → leakage audit → HTML + JSON report.

    Examples
    --------
    \\b
    # Basic run with defaults
    recon run

    \\b
    # Custom paths
    recon run --live path/to/live_trades.db --backtest path/to/bt.csv -o output/

    \\b
    # Fast run without plots
    recon run --no-plots --json-export
    """
    from research.reconciliation.report import generate_full_report, to_console

    _print_header("Full Reconciliation Pipeline")
    click.echo(f"  Live:     {live}")
    click.echo(f"  Backtest: {backtest}")
    click.echo(f"  Output:   {output}")
    click.echo()

    live_df, bt_df, merged = _load_trades(live, backtest)

    _print_header("Running Analysis")

    console, Progress = _try_rich()
    steps = [
        "Slippage analysis",
        "PnL attribution",
        "Signal drift",
        "Leakage audit",
        "Report generation",
    ]

    if Progress and console:
        from rich.progress import SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating full report…", total=1)
            report = generate_full_report(
                live_df, bt_df,
                output_dir=output,
                dpi=dpi,
                annual_factor=annual_factor,
                suppress_plots=no_plots,
            )
            progress.advance(task)
    else:
        click.echo("  Running generate_full_report …")
        report = generate_full_report(
            live_df, bt_df,
            output_dir=output,
            dpi=dpi,
            annual_factor=annual_factor,
            suppress_plots=no_plots,
        )

    _print_ok("Analysis complete.")

    to_console(report)

    # Artefacts
    click.echo()
    if not no_html:
        html = report.plot_paths.get("html_report", "")
        if html:
            _print_ok(f"HTML report: {html}")
    if json_export or True:  # always write JSON
        js = report.plot_paths.get("json_summary", "")
        if js:
            _print_ok(f"JSON summary: {js}")

    grade = report.overall_grade()
    click.echo(f"\n  Overall grade: {grade}")
    if grade in ("D", "F"):
        _print_warn("Poor grade — review leakage and overfitting flags.")
    elif grade in ("A", "B"):
        _print_ok("Strategy passes reconciliation quality checks.")


# ── recon slippage ────────────────────────────────────────────────────────────

@recon.command("slippage")
@click.option("--live", "-l", default="tools/backtest_output/live_trades.db")
@click.option("--backtest", "-b", default="tools/backtest_output/crypto_trades.csv")
@click.option("--output", "-o", default="research/reconciliation/output")
@click.option("--spread-bps", default=5.0, show_default=True, help="Assumed half-spread (bps).")
@click.option("--impact-coeff", default=0.1, show_default=True, help="Almgren-Chriss η coefficient.")
@click.option("--plot/--no-plot", default=True, show_default=True)
@click.option("--json-out", is_flag=True, default=False)
def slippage_command(
    live: str,
    backtest: str,
    output: str,
    spread_bps: float,
    impact_coeff: float,
    plot: bool,
    json_out: bool,
) -> None:
    """
    Run slippage and fill-quality analysis only.

    Computes per-trade slippage decomposed into spread, timing, and
    market-impact components.  Generates a distribution plot.

    Examples
    --------
    \\b
    recon slippage --spread-bps 10 --plot
    """
    from research.reconciliation.slippage import SlippageAnalyzer

    _, _, merged = _load_trades(live, backtest)
    if merged.empty:
        _print_warn("No merged trades available; cannot compute slippage.")
        return

    _print_header("Slippage Analysis")
    analyzer = SlippageAnalyzer(spread_bps=spread_bps, impact_coeff=impact_coeff)
    report = analyzer.analyze_fill_quality(merged)

    _print_ok(f"N trades: {report.n_trades}")

    overall = report.overall
    click.echo(f"  Mean slippage:   {overall.mean_bps:.2f} bps")
    click.echo(f"  P50  slippage:   {overall.median_bps:.2f} bps")
    click.echo(f"  P95  slippage:   {overall.p95_bps:.2f} bps")
    click.echo(f"  P99  slippage:   {overall.p99_bps:.2f} bps")
    click.echo(f"  Total cost USD:  ${overall.total_cost_usd:,.0f}")
    click.echo(f"  Spread component: {overall.spread_component_bps:.2f} bps")
    click.echo(f"  Timing component: {overall.timing_component_bps:.2f} bps")
    click.echo(f"  Impact component: {overall.impact_component_bps:.2f} bps")

    click.echo("\n  By Regime:")
    for regime, stats in report.by_regime.items():
        click.echo(f"    {regime:<12} mean={stats.mean_bps:.1f} bps  "
                   f"n={stats.n_trades}  cost=${stats.total_cost_usd:,.0f}")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if plot:
        save_path = out_dir / "slippage_distribution.png"
        try:
            analyzer.plot_slippage_distribution(merged, save_path)
            _print_ok(f"Plot: {save_path}")
        except Exception as exc:
            _print_warn(f"Plot failed: {exc}")

    if json_out:
        data = {
            "overall": overall.to_dict(),
            "by_regime": {k: v.to_dict() for k, v in report.by_regime.items()},
        }
        jp = out_dir / "slippage_report.json"
        jp.write_text(json.dumps(data, indent=2, default=str))
        _print_ok(f"JSON: {jp}")


# ── recon drift ───────────────────────────────────────────────────────────────

@recon.command("drift")
@click.option("--live", "-l", default="tools/backtest_output/live_trades.db")
@click.option("--backtest", "-b", default="tools/backtest_output/crypto_trades.csv")
@click.option("--output", "-o", default="research/reconciliation/output")
@click.option("--window", default=20, show_default=True, help="Rolling window for drift detection.")
@click.option("--lags", default=20, show_default=True, help="Max lags for Ljung-Box test.")
@click.option("--stability-window", default=500, show_default=True)
@click.option("--plot/--no-plot", default=True)
@click.option("--json-out", is_flag=True, default=False)
@click.option("--signal-col", default="delta_score", show_default=True,
              help="Signal column for ACF/Ljung-Box analysis.")
def drift_command(
    live: str,
    backtest: str,
    output: str,
    window: int,
    lags: int,
    stability_window: int,
    plot: bool,
    json_out: bool,
    signal_col: str,
) -> None:
    """
    Detect signal and regime drift between live and backtest.

    Runs rolling Jaccard activation overlap, Ljung-Box test, and
    Chow-test parameter stability checks.

    Examples
    --------
    \\b
    recon drift --window 30 --lags 20 --signal-col tf_score
    """
    from research.reconciliation.drift import SignalDriftDetector

    live_df, bt_df, merged = _load_trades(live, backtest)
    _print_header("Signal & Regime Drift Detection")

    detector = SignalDriftDetector()
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Distribution comparison
    for sc in ("delta_score", "tf_score", "mass", "ensemble_signal"):
        res = detector.compare_signal_distributions(live_df, bt_df, sc)
        if not any(v != v for v in (res.get("ks_statistic"), res.get("ks_p_value"))):  # not NaN
            status = "DIVERGED ⚠" if res.get("diverged") else "OK"
            click.echo(
                f"  {sc:<20}  KS={res.get('ks_statistic', float('nan')):.3f}  "
                f"p={res.get('ks_p_value', float('nan')):.3f}  "
                f"shift={res.get('distribution_shift', float('nan')):.2f}  {status}"
            )

    # Ljung-Box test on signal
    sig_col_live = signal_col if signal_col in live_df.columns else None
    if sig_col_live:
        lb_result = detector.ljung_box_test(live_df[sig_col_live].dropna(), lags=lags)
        click.echo(f"\n  Ljung-Box ({signal_col}): "
                   f"autocorrelation_present={lb_result.autocorrelation_present}  "
                   f"max_sig_lag={lb_result.max_significant_lag}")
    else:
        click.echo(f"\n  Signal column '{signal_col}' not found in live trades.")

    # Activation overlap (from trades)
    try:
        overlap_result = detector.compute_activation_overlap_from_trades(
            live_df, bt_df, window=window,
        )
        click.echo(f"\n  Activation overlap: mean={overlap_result.mean_overlap:.3f}  "
                   f"min={overlap_result.min_overlap:.3f}  "
                   f"low_periods={len(overlap_result.low_overlap_periods)}")

        if plot and len(overlap_result.overlap_series) > 0:
            op_path = out_dir / "activation_overlap.png"
            detector.plot_activation_overlap(overlap_result, op_path)
            _print_ok(f"Overlap plot: {op_path}")
    except Exception as exc:
        _print_warn(f"Activation overlap failed: {exc}")

    # Parameter stability
    pnl_df = bt_df if not bt_df.empty else live_df
    stability = detector.parameter_stability_test(pnl_df, window=stability_window)
    summary = detector.parameter_stability_summary(stability)
    click.echo(f"\n  Parameter stability: stable={summary['is_stable']}  "
               f"breakpoints={summary['n_breakpoints']}")
    if summary["breakpoints"]:
        click.echo(f"  Breakpoint indices: {summary['breakpoints']}")

    if plot and not pnl_df.empty:
        try:
            sig_col_bt = signal_col if signal_col in pnl_df.columns else \
                ("delta_score" if "delta_score" in pnl_df.columns else None)
            if sig_col_bt:
                acf_path = out_dir / "acf_pacf.png"
                detector.plot_acf_pacf(pnl_df[sig_col_bt].dropna(), acf_path,
                                       title=f"ACF/PACF: {sig_col_bt}")
                _print_ok(f"ACF/PACF plot: {acf_path}")
        except Exception as exc:
            _print_warn(f"ACF plot failed: {exc}")

    if json_out:
        data = {
            "signal_drift": {sc: detector.compare_signal_distributions(live_df, bt_df, sc)
                             for sc in ("delta_score", "tf_score", "mass")},
            "parameter_stability": summary,
        }
        if sig_col_live:
            lb = detector.ljung_box_test(live_df[signal_col].dropna(), lags=lags)
            data["ljung_box"] = lb.to_dict()
        jp = out_dir / "drift_report.json"
        jp.write_text(json.dumps(data, indent=2, default=str))
        _print_ok(f"JSON: {jp}")


# ── recon attribution ─────────────────────────────────────────────────────────

@recon.command("attribution")
@click.option("--live", "-l", default="tools/backtest_output/live_trades.db")
@click.option("--backtest", "-b", default="tools/backtest_output/crypto_trades.csv")
@click.option("--output", "-o", default="research/reconciliation/output")
@click.option("--ic-window", default=60, show_default=True,
              help="Rolling window for IC calculation.")
@click.option("--plot/--no-plot", default=True)
@click.option("--json-out", is_flag=True, default=False)
def attribution_command(
    live: str,
    backtest: str,
    output: str,
    ic_window: int,
    plot: bool,
    json_out: bool,
) -> None:
    """
    Run PnL attribution analysis only.

    Decomposes PnL into selection, timing, sizing, regime, and signal
    effects using a BHB-style framework.

    Examples
    --------
    \\b
    recon attribution --ic-window 30 --plot
    """
    from research.reconciliation.attribution import PnLAttributionEngine

    live_df, bt_df, _ = _load_trades(live, backtest)
    _print_header("PnL Attribution Analysis")

    engine = PnLAttributionEngine()
    report = engine.attribute_pnl(
        live_df if not live_df.empty else bt_df,
        benchmark_trades=bt_df,
    )

    effects = report.effect_table()
    click.echo(f"\n  Total PnL: ${report.total_pnl:,.2f}  N={report.n_trades}")
    click.echo("\n  Attribution Breakdown:")
    for effect, row in effects.iterrows():
        pct = f"{row.get('Pct_of_Total', float('nan')):.1f}%" \
            if not (row.get("Pct_of_Total") != row.get("Pct_of_Total")) else "—"
        click.echo(f"    {effect:<20} ${float(row.get('Value', 0)):>10,.2f}  ({pct})")

    click.echo(f"\n  ICIR: {report.icir:.3f}")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if plot:
        for fn, method in [
            ("attribution_waterfall.png", lambda: engine.plot_attribution_waterfall(report, out_dir / "attribution_waterfall.png")),
            ("rolling_ic.png", lambda: engine.plot_rolling_ic(report, out_dir / "rolling_ic.png")),
            ("regime_attribution.png", lambda: engine.plot_regime_attribution(report, out_dir / "regime_attribution.png")),
        ]:
            try:
                path = method()
                _print_ok(f"Plot: {path}")
            except Exception as exc:
                _print_warn(f"{fn} failed: {exc}")

    if json_out:
        jp = out_dir / "attribution_report.json"
        jp.write_text(
            json.dumps(report.summary_dict(), indent=2, default=str)
        )
        _print_ok(f"JSON: {jp}")


# ── recon report ──────────────────────────────────────────────────────────────

@recon.command("report")
@click.option("--live", "-l", default="tools/backtest_output/live_trades.db")
@click.option("--backtest", "-b", default="tools/backtest_output/crypto_trades.csv")
@click.option(
    "--input-dir", "-i",
    default=None,
    help="Load existing JSON summary from a prior run instead of re-computing.",
)
@click.option("--output", "-o", default="research/reconciliation/output")
@click.option("--format", "fmt",
              type=click.Choice(["html", "console", "both"], case_sensitive=False),
              default="both", show_default=True)
@click.option("--no-plots", is_flag=True, default=False)
def report_command(
    live: str,
    backtest: str,
    input_dir: Optional[str],
    output: str,
    fmt: str,
    no_plots: bool,
) -> None:
    """
    Generate or re-render the reconciliation report.

    If --input-dir is supplied, load the JSON summary produced by a prior
    ``recon run`` instead of re-computing everything (faster).

    Examples
    --------
    \\b
    # Full re-run and render both HTML and console
    recon report --live live.db --backtest bt.csv --format both

    \\b
    # Re-render from saved JSON without reloading trades
    recon report --input-dir research/reconciliation/output --format html
    """
    from research.reconciliation.report import generate_full_report, to_html, to_console

    _print_header("Report Generation")

    if input_dir:
        jp = Path(input_dir) / "reconciliation_summary.json"
        if jp.exists():
            click.echo(f"  Loading existing summary from {jp}")
            data = json.loads(jp.read_text(encoding="utf-8"))
            click.echo(json.dumps(data, indent=2, default=str))
            return
        else:
            _print_warn(f"No summary JSON found at {jp}; re-running analysis.")

    live_df, bt_df, _ = _load_trades(live, backtest)
    report = generate_full_report(
        live_df, bt_df,
        output_dir=output,
        suppress_plots=no_plots,
    )

    if fmt in ("console", "both"):
        to_console(report)

    if fmt in ("html", "both"):
        html_path = Path(output) / "reconciliation_report.html"
        to_html(report, html_path)
        _print_ok(f"HTML report: {html_path}")


# ── recon leakage ─────────────────────────────────────────────────────────────

@recon.command("leakage")
@click.option("--live", "-l", default="tools/backtest_output/live_trades.db")
@click.option("--backtest", "-b", default="tools/backtest_output/crypto_trades.csv")
@click.option("--output", "-o", default="research/reconciliation/output")
@click.option("--embargo-bars", default=5, show_default=True)
@click.option("--n-trials", default=100, show_default=True,
              help="Equivalent number of parameter-search trials (for DSR).")
@click.option("--plot/--no-plot", default=True)
@click.option("--json-out", is_flag=True, default=False)
def leakage_command(
    live: str,
    backtest: str,
    output: str,
    embargo_bars: int,
    n_trials: int,
    plot: bool,
    json_out: bool,
) -> None:
    """
    Run the data-leakage and overfitting audit only.

    Checks for lookahead bias, computes the Deflated Sharpe Ratio,
    runs VIF analysis, and estimates autocorrelation-adjusted Sharpe.

    Examples
    --------
    \\b
    recon leakage --embargo-bars 10 --n-trials 200 --plot
    """
    from research.reconciliation.leakage import DataLeakageAuditor

    live_df, bt_df, _ = _load_trades(live, backtest)
    _print_header("Leakage & Overfitting Audit")

    auditor = DataLeakageAuditor(
        embargo_bars=embargo_bars,
        n_trials_equivalent=n_trials,
    )

    import pandas as pd
    # Build factor matrix from backtest numeric columns
    num_cols = [c for c in bt_df.select_dtypes(include=["number"]).columns
                if c not in ("pnl", "return_pct")]
    factor_matrix = bt_df[num_cols].dropna() if num_cols else None

    report = auditor.audit(live_df, bt_df, factor_matrix=factor_matrix)
    summary = report.summary_dict()

    # Print
    click.echo(f"\n  Leakage Score:       {summary['leakage_score']:.3f}")
    click.echo(f"  Lookahead Suspected: {summary['lookahead_suspected']}")
    click.echo(f"  Live Sharpe (period): {summary['live_sharpe_in_period']:.2f}")
    click.echo(f"  BT Sharpe (period):  {summary['bt_sharpe_in_period']:.2f}")
    click.echo(f"  Sharpe Excess:       {summary['sharpe_excess']:.2f}")
    click.echo(f"  Purged:              {summary['n_purged']} trades ({summary['purge_fraction']:.1%})")
    click.echo(f"  Deflated Sharpe:     {summary['deflated_sharpe']:.3f}")
    click.echo(f"  Overfit Prob:        {summary['overfitting_probability']:.1%}")
    click.echo(f"  NW-Adj Sharpe:       {summary['ac_adjusted_sharpe']:.2f}")
    click.echo(f"  AC Inflation:        {summary['autocorrelation_score']:.3f}")

    if summary["high_vif_factors"]:
        _print_warn(f"High-VIF factors: {', '.join(summary['high_vif_factors'])}")
    else:
        _print_ok("No high-VIF factors detected.")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if plot:
        lp = out_dir / "leakage_summary.png"
        try:
            auditor.plot_leakage_summary(report, lp)
            _print_ok(f"Plot: {lp}")
        except Exception as exc:
            _print_warn(f"Plot failed: {exc}")

    if json_out:
        jp = out_dir / "leakage_report.json"
        jp.write_text(json.dumps(summary, indent=2, default=str))
        _print_ok(f"JSON: {jp}")


# ── recon impact ──────────────────────────────────────────────────────────────

@recon.command("impact")
@click.option("--dollar-size", "-d", required=True, type=float,
              help="Order size in USD.")
@click.option("--adv", "-a", default=0.0, type=float,
              help="Average daily volume in USD (0 = use symbol default).")
@click.option("--sigma", "-s", default=0.03, type=float, show_default=True,
              help="Daily volatility (decimal).")
@click.option("--sym", default="BTC", show_default=True, help="Symbol (for ADV lookup).")
@click.option("--spread-bps", default=5.0, show_default=True)
@click.option("--impact-coeff", default=0.1, show_default=True)
def impact_command(
    dollar_size: float,
    adv: float,
    sigma: float,
    sym: str,
    spread_bps: float,
    impact_coeff: float,
) -> None:
    """
    Estimate market impact for a single hypothetical order.

    Uses the Almgren-Chriss square-root model.

    Examples
    --------
    \\b
    recon impact --dollar-size 1000000 --sym BTC --sigma 0.04

    \\b
    recon impact -d 500000 --adv 5000000000 --sigma 0.025
    """
    from research.reconciliation.slippage import SlippageAnalyzer

    analyzer = SlippageAnalyzer(spread_bps=spread_bps, impact_coeff=impact_coeff)
    est = analyzer.estimate_market_impact(dollar_size, adv, sigma, sym)

    _print_header("Market Impact Estimate")
    click.echo(f"  Symbol:             {sym}")
    click.echo(f"  Order size:         ${dollar_size:,.0f}")
    click.echo(f"  ADV:                ${est.adv:,.0f}")
    click.echo(f"  Daily vol (σ):      {est.sigma:.2%}")
    click.echo(f"  Participation rate: {est.participation_rate:.4%}")
    click.echo(f"  Linear impact:      {est.linear_impact_bps:.2f} bps")
    click.echo(f"  √ impact:           {est.sqrt_impact_bps:.2f} bps")
    click.echo(f"  Total impact:       {est.total_impact_bps:.2f} bps  "
               f"(${est.total_impact_usd:,.0f})")
    click.echo(f"  + Spread (both sides): {spread_bps * 2:.1f} bps")
    click.echo(f"  Total round-trip:   {est.total_impact_bps + spread_bps * 2:.2f} bps")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for `recon` CLI when installed as a package script."""
    recon(standalone_mode=True)


if __name__ == "__main__":
    main()
