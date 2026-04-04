"""
research/signal_analytics/cli.py
==================================
Click command-line interface for signal analytics.

Commands
--------
  signal ic          – Compute IC / ICIR for a trades CSV
  signal factors     – Fama-MacBeth factor attribution
  signal decay       – Alpha decay analysis and half-life
  signal bh-quality  – BH-specific signal quality diagnostics
  signal report      – Full signal analytics tearsheet

Examples
--------
  python -m research.signal_analytics.cli ic \\
      --trades data/crypto_trades.csv \\
      --signal ensemble_signal --horizon 20

  python -m research.signal_analytics.cli bh-quality \\
      --trades data/crypto_trades.csv \\
      --output results/bh_quality/

  python -m research.signal_analytics.cli report \\
      --trades data/crypto_trades.csv \\
      --output results/signal_report/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI entry-point group
# ---------------------------------------------------------------------------

@click.group(name="signal", help="Signal analytics and IC/ICIR diagnostics for SRFM-Lab.")
@click.version_option("0.1.0", prog_name="signal")
def cli() -> None:
    """SRFM-Lab Signal Analytics CLI."""


# ---------------------------------------------------------------------------
# Helper: load trades CSV
# ---------------------------------------------------------------------------

def _load_trades(trades_path: str) -> pd.DataFrame:
    """Load a trades CSV with sensible defaults."""
    path = Path(trades_path)
    if not path.exists():
        click.echo(f"[ERROR] Trades file not found: {trades_path}", err=True)
        sys.exit(1)

    trades = pd.read_csv(path)
    # Try to parse exit_time
    if "exit_time" in trades.columns:
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")
    elif trades.columns[0].lower() in ("date", "datetime", "timestamp", "time"):
        trades = trades.rename(columns={trades.columns[0]: "exit_time"})
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")

    click.echo(f"Loaded {len(trades)} trades from {trades_path}")
    click.echo(f"Columns: {', '.join(trades.columns.tolist())}")
    return trades


def _infer_signal_col(trades: pd.DataFrame, hint: Optional[str]) -> str:
    """Pick a signal column, preferring the user hint over defaults."""
    if hint and hint in trades.columns:
        return hint
    for candidate in ["ensemble_signal", "delta_score", "tf_score", "mass"]:
        if candidate in trades.columns:
            return candidate
    raise click.ClickException(
        "No recognised signal column found. "
        "Pass --signal <col> or add ensemble_signal/delta_score to the trades CSV."
    )


# ---------------------------------------------------------------------------
# signal ic
# ---------------------------------------------------------------------------

@cli.command("ic")
@click.option("--trades", required=True, type=click.Path(exists=True), help="Trades CSV path.")
@click.option("--signal", default=None, help="Signal column name.")
@click.option("--horizon", default=20, show_default=True, help="Max IC decay horizon (bars).")
@click.option("--method", default="spearman", type=click.Choice(["pearson", "spearman", "kendall"]),
              show_default=True, help="Correlation method.")
@click.option("--output", default=None, help="Directory to save IC decay plot.")
@click.option("--return-col", default="pnl", show_default=True, help="Return/P&L column.")
@click.option("--pos-col", default="dollar_pos", show_default=True, help="Dollar position column.")
def cmd_ic(
    trades: str,
    signal: Optional[str],
    horizon: int,
    method: str,
    output: Optional[str],
    return_col: str,
    pos_col: str,
) -> None:
    """Compute IC / ICIR for a trades CSV.

    Outputs IC summary, IC decay curve, and optional plots.

    \b
    Example:
      signal ic --trades crypto_trades.csv --horizon 20
    """
    from research.signal_analytics.ic_framework import ICCalculator

    df = _load_trades(trades)
    sig_col = _infer_signal_col(df, signal)

    # Normalise returns
    if pos_col in df.columns:
        pos = df[pos_col].abs().replace(0, np.nan)
        df["_ret"] = df[return_col] / pos
    else:
        df["_ret"] = df[return_col]

    calc = ICCalculator()

    # Overall IC
    from scipy import stats
    sub = df[[sig_col, "_ret"]].dropna()
    if len(sub) < 3:
        click.echo("[WARNING] Too few observations to compute IC.")
        return

    r_sp, p_sp = stats.spearmanr(sub[sig_col], sub["_ret"])
    t_stat = float(r_sp) * np.sqrt(len(sub) - 2) / max(np.sqrt(max(1 - r_sp**2, 1e-10)), 1e-10)

    click.echo("\n" + "=" * 55)
    click.echo("IC ANALYSIS")
    click.echo("=" * 55)
    click.echo(f"  Signal column   : {sig_col}")
    click.echo(f"  N observations  : {len(sub)}")
    click.echo(f"  IC ({method:8s})  : {float(r_sp):.4f}")
    click.echo(f"  t-stat          : {float(t_stat):.2f}")
    click.echo(f"  p-value         : {float(p_sp):.4f}")
    click.echo(f"  Significant 5%  : {'YES' if abs(t_stat) > 1.96 else 'no'}")
    click.echo("=" * 55)

    # IC Decay
    if "hold_bars" in df.columns:
        click.echo(f"\nComputing IC decay (max horizon={horizon})...")
        decay_result = calc.ic_decay_from_trades(
            df, signal_col=sig_col, return_col=return_col,
            hold_col="hold_bars", dollar_pos_col=pos_col, max_horizon=horizon, method=method,
        )
        click.echo(f"\n{'Horizon':>8}  {'IC':>10}  {'Stderr':>10}")
        click.echo("-" * 32)
        for h, ic, se in zip(decay_result.horizons, decay_result.ic_values, decay_result.ic_stderr):
            ic_str = f"{ic:.4f}" if not np.isnan(ic) else "    —"
            se_str = f"{se:.4f}" if not np.isnan(se) else "    —"
            click.echo(f"{h:>8}  {ic_str:>10}  {se_str:>10}")

        click.echo(f"\n  Half-life  : {decay_result.half_life:.1f} bars")
        click.echo(f"  Decay rate : {decay_result.decay_rate:.4f} per bar")
        click.echo(f"  Fit R²     : {decay_result.r_squared:.3f}")

        if output:
            out_path = Path(output)
            out_path.mkdir(parents=True, exist_ok=True)
            fig = calc.plot_ic_decay(decay_result, save_path=out_path / "ic_decay.png")
            import matplotlib.pyplot as plt
            plt.close(fig)
            click.echo(f"\nIC decay plot saved to: {out_path / 'ic_decay.png'}")


# ---------------------------------------------------------------------------
# signal factors
# ---------------------------------------------------------------------------

@cli.command("factors")
@click.option("--trades", required=True, type=click.Path(exists=True), help="Trades CSV path.")
@click.option("--signal", default=None, help="Signal column name.")
@click.option("--output", default=None, help="Directory to save factor plots.")
@click.option("--return-col", default="pnl", show_default=True, help="Return/P&L column.")
@click.option("--pos-col", default="dollar_pos", show_default=True, help="Dollar position column.")
@click.option("--nw-lags", default=5, show_default=True, help="Newey-West lag truncation.")
def cmd_factors(
    trades: str,
    signal: Optional[str],
    output: Optional[str],
    return_col: str,
    pos_col: str,
    nw_lags: int,
) -> None:
    """Run factor attribution and IC analysis.

    Computes per-factor IC and Fama-MacBeth (cross-sectional OLS).

    \b
    Example:
      signal factors --trades crypto_trades.csv --output results/factors/
    """
    from research.signal_analytics.factor_model import FactorModel

    df = _load_trades(trades)
    fm = FactorModel()

    click.echo("\nBuilding factor matrix...")
    factor_df = fm.build_factor_matrix(df)
    click.echo(f"Factors available: {', '.join(factor_df.columns.tolist())}")

    if pos_col in df.columns:
        pos = df[pos_col].abs().replace(0, np.nan)
        df["_ret"] = df[return_col] / pos
    else:
        df["_ret"] = df[return_col]

    ret = df["_ret"].loc[factor_df.index].dropna()
    factor_aligned = factor_df.loc[ret.index]

    # Factor ICs
    factor_ics = fm.factor_ic(factor_aligned, ret)

    click.echo("\n" + "=" * 55)
    click.echo("FACTOR ICs")
    click.echo("=" * 55)
    click.echo(f"{'Factor':>20}  {'IC':>10}")
    click.echo("-" * 35)
    for fname, ic in sorted(factor_ics.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True):
        ic_str = f"{ic:.4f}" if not np.isnan(ic) else "     —"
        click.echo(f"{fname:>20}  {ic_str:>10}")

    # Attribution
    try:
        click.echo("\nRunning factor attribution...")
        attr = fm.factor_attribution(df, return_col=return_col, dollar_pos_col=pos_col)
        click.echo(f"\n  Total return     : {attr.total_return:.4f}")
        click.echo(f"  Systematic       : {attr.systematic_return:.4f}")
        click.echo(f"  Idiosyncratic    : {attr.idiosyncratic_return:.4f}")
        click.echo(f"  R²               : {attr.r_squared:.3f}")
        click.echo("\n  Factor contributions:")
        for fname, contrib in attr.factor_contributions.items():
            click.echo(f"    {fname:>20} : {contrib:.4f}")
    except Exception as e:
        click.echo(f"[WARNING] Attribution failed: {e}", err=True)

    if output:
        out_path = Path(output)
        out_path.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        fig_fic = fm.plot_factor_ic_bar(factor_ics, save_path=out_path / "factor_ic_bar.png")
        plt.close(fig_fic)
        click.echo(f"\nFactor IC bar chart saved to: {out_path / 'factor_ic_bar.png'}")


# ---------------------------------------------------------------------------
# signal decay
# ---------------------------------------------------------------------------

@cli.command("decay")
@click.option("--trades", required=True, type=click.Path(exists=True), help="Trades CSV path.")
@click.option("--signal", default=None, help="Signal column name.")
@click.option("--horizon", default=20, show_default=True, help="Max decay horizon (bars).")
@click.option("--cost", default=0.0002, show_default=True, help="One-way transaction cost.")
@click.option("--output", default=None, help="Directory to save decay plots.")
@click.option("--return-col", default="pnl", show_default=True, help="Return/P&L column.")
@click.option("--pos-col", default="dollar_pos", show_default=True, help="Dollar position column.")
def cmd_decay(
    trades: str,
    signal: Optional[str],
    horizon: int,
    cost: float,
    output: Optional[str],
    return_col: str,
    pos_col: str,
) -> None:
    """Analyse alpha decay and optimal holding period.

    Fits an exponential IC decay model and computes the holding period
    that maximises net IC after transaction costs.

    \b
    Example:
      signal decay --trades crypto_trades.csv --horizon 20 --cost 0.0002
    """
    from research.signal_analytics.ic_framework import ICCalculator
    from research.signal_analytics.alpha_decay import AlphaDecayAnalyzer

    df = _load_trades(trades)
    sig_col = _infer_signal_col(df, signal)

    if "hold_bars" not in df.columns:
        click.echo("[ERROR] 'hold_bars' column required for decay analysis.", err=True)
        sys.exit(1)

    calc = ICCalculator()
    ada = AlphaDecayAnalyzer(default_cost=cost)

    click.echo(f"\nFitting IC decay (max_horizon={horizon}, cost={cost:.4%})...")
    decay_result = calc.ic_decay_from_trades(
        df, signal_col=sig_col, return_col=return_col,
        hold_col="hold_bars", dollar_pos_col=pos_col, max_horizon=horizon,
    )

    decay_model = ada.signal_decay_model(decay_result)
    opt_hold = ada.optimal_holding_period(decay_result, transaction_cost=cost)
    half_life = ada.compute_signal_halflife(decay_result)

    click.echo("\n" + "=" * 55)
    click.echo("ALPHA DECAY ANALYSIS")
    click.echo("=" * 55)
    click.echo(f"  Signal column    : {sig_col}")
    click.echo(f"  Peak IC          : {decay_result.peak_ic:.4f} @ h={decay_result.peak_horizon} bars")
    click.echo(f"  Half-life        : {half_life:.1f} bars")
    click.echo(f"  Decay rate λ     : {decay_model.decay_rate:.4f} per bar")
    click.echo(f"  IC at h=0 (IC₀)  : {decay_model.ic_at_zero:.4f}")
    click.echo(f"  Model R²         : {decay_model.r_squared:.3f}")
    click.echo(f"  Optimal holding  : {opt_hold} bars")
    click.echo(f"  Transaction cost : {cost:.4%} (one-way)")
    click.echo("=" * 55)

    if output:
        out_path = Path(output)
        out_path.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        fig = ada.plot_alpha_decay(decay_result, save_path=out_path / "alpha_decay.png", transaction_cost=cost)
        plt.close(fig)
        click.echo(f"\nAlpha decay plot saved to: {out_path / 'alpha_decay.png'}")


# ---------------------------------------------------------------------------
# signal bh-quality
# ---------------------------------------------------------------------------

@cli.command("bh-quality")
@click.option("--trades", required=True, type=click.Path(exists=True), help="Trades CSV path.")
@click.option("--signal", default="ensemble_signal", show_default=True, help="Signal column name.")
@click.option("--output", required=True, type=click.Path(), help="Directory to save results.")
@click.option("--return-col", default="pnl", show_default=True, help="Return/P&L column.")
@click.option("--pos-col", default="dollar_pos", show_default=True, help="Dollar position column.")
@click.option("--mass-sweep/--no-mass-sweep", default=True, show_default=True,
              help="Run mass threshold sweep.")
def cmd_bh_quality(
    trades: str,
    signal: str,
    output: str,
    return_col: str,
    pos_col: str,
    mass_sweep: bool,
) -> None:
    """BH-specific signal quality diagnostics.

    Computes IC of each BH signal component, mass threshold sweep,
    tf_score analysis, and ensemble calibration.

    \b
    Example:
      signal bh-quality --trades crypto_trades.csv --output results/bh/
    """
    from research.signal_analytics.bh_signal_quality import BHSignalAnalyzer
    import matplotlib.pyplot as plt

    df = _load_trades(trades)
    out_path = Path(output)
    out_path.mkdir(parents=True, exist_ok=True)

    bh = BHSignalAnalyzer(
        signal_col=signal,
        return_col=return_col,
        dollar_pos_col=pos_col,
    )

    click.echo("\nComputing BH activation quality...")
    aq = bh.activation_quality(df)

    click.echo("\n" + "=" * 55)
    click.echo("BH ACTIVATION QUALITY")
    click.echo("=" * 55)
    click.echo(f"  Total activations     : {aq.total_activations}")
    click.echo(f"  Profitable fraction   : {aq.profitable_fraction:.1%}")
    click.echo(f"  Win rate              : {aq.win_rate:.1%}")
    click.echo(f"  Total PnL             : ${aq.total_pnl:,.2f}")
    click.echo(f"  Sharpe                : {aq.sharpe:.2f}")
    click.echo(f"\n  IC — ensemble_signal  : {aq.ic_ensemble:.4f}")
    click.echo(f"  IC — tf_score         : {aq.ic_tf_score:.4f}")
    click.echo(f"  IC — mass             : {aq.ic_mass:.4f}")
    click.echo(f"  IC — delta_score      : {aq.ic_delta_score:.4f}")
    click.echo(f"  IC — ATR              : {aq.ic_atr:.4f}")
    click.echo(f"\n  Long trades           : {aq.long_trades}  (win={aq.long_win_rate:.1%}  IC={aq.long_ic:.4f})")
    click.echo(f"  Short trades          : {aq.short_trades}  (win={aq.short_win_rate:.1%}  IC={aq.short_ic:.4f})")
    click.echo("=" * 55)

    # Mass sweep
    if mass_sweep and "mass" in df.columns:
        click.echo("\nRunning mass threshold sweep...")
        thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        sweep = bh.mass_threshold_sweep(df, thresholds)
        click.echo(f"\n{'Threshold':>12}  {'N Trades':>10}  {'Win Rate':>10}  {'Sharpe':>10}")
        click.echo("-" * 48)
        for thr in thresholds:
            ps = sweep[thr]
            wr = f"{ps.win_rate:.1%}" if not np.isnan(ps.win_rate) else "  —"
            sh = f"{ps.sharpe:.2f}" if not np.isnan(ps.sharpe) else " —"
            click.echo(f"{thr:>12.2f}  {ps.n_trades:>10}  {wr:>10}  {sh:>10}")

    # TF-score analysis
    if "tf_score" in df.columns:
        click.echo("\nTF-Score Analysis...")
        tf_result = bh.tf_score_analysis(df)
        click.echo(f"\n{'TF Score':>10}  {'N':>8}  {'Win Rate':>10}  {'Mean Ret':>12}")
        click.echo("-" * 46)
        for sc, n, wr, mr in zip(
            tf_result.scores, tf_result.n_trades,
            tf_result.win_rates, tf_result.mean_returns
        ):
            click.echo(f"{sc:>10}  {n:>8}  {wr:>10.1%}  {mr:>12.4f}")
        click.echo(f"\nBest TF score: {tf_result.best_score}")

    # Plots
    fig_dash = bh.plot_signal_quality_dashboard(aq, trades=df, save_path=out_path / "bh_dashboard.png")
    plt.close(fig_dash)
    click.echo(f"\nDashboard saved to: {out_path / 'bh_dashboard.png'}")

    if signal in df.columns:
        fig_calib = bh.plot_ensemble_calibration(df, save_path=out_path / "ensemble_calibration.png")
        plt.close(fig_calib)
        click.echo(f"Calibration plot saved to: {out_path / 'ensemble_calibration.png'}")

    # Signal summary table
    summary_tbl = bh.signal_summary_table(df)
    click.echo(f"\nSignal Summary:\n{summary_tbl.to_string()}")
    summary_tbl.to_csv(out_path / "signal_summary.csv")
    click.echo(f"\nSignal summary CSV saved to: {out_path / 'signal_summary.csv'}")


# ---------------------------------------------------------------------------
# signal report
# ---------------------------------------------------------------------------

@cli.command("report")
@click.option("--trades", required=True, type=click.Path(exists=True), help="Trades CSV path.")
@click.option("--output", required=True, type=click.Path(), help="Output directory.")
@click.option("--signal", default=None, help="Signal column name.")
@click.option("--horizon", default=20, show_default=True, help="Max IC decay horizon.")
@click.option("--quantiles", default=5, show_default=True, help="Number of quantile buckets.")
@click.option("--rolling-window", default=60, show_default=True, help="Rolling IC window.")
@click.option("--cost", default=0.0002, show_default=True, help="Transaction cost (one-way).")
@click.option("--return-col", default="pnl", show_default=True, help="Return/P&L column.")
@click.option("--pos-col", default="dollar_pos", show_default=True, help="Dollar position column.")
@click.option("--html/--no-html", "emit_html", default=True, show_default=True,
              help="Generate HTML report.")
def cmd_report(
    trades: str,
    output: str,
    signal: Optional[str],
    horizon: int,
    quantiles: int,
    rolling_window: int,
    cost: float,
    return_col: str,
    pos_col: str,
    emit_html: bool,
) -> None:
    """Generate a comprehensive signal analytics tearsheet.

    Produces IC tearsheet, factor attribution, quintile charts,
    BH activation quality, and an HTML report.

    \b
    Example:
      signal report --trades crypto_trades.csv --output results/signal_report/
    """
    from research.signal_analytics.report import generate_signal_report, to_html, to_console

    df = _load_trades(trades)
    sig_col = _infer_signal_col(df, signal)
    out_path = Path(output)

    click.echo(f"\nGenerating comprehensive signal analytics report...")
    click.echo(f"  Signal column     : {sig_col}")
    click.echo(f"  Output directory  : {out_path}")
    click.echo(f"  Max decay horizon : {horizon} bars")
    click.echo(f"  Quantiles         : {quantiles}")
    click.echo(f"  Rolling IC window : {rolling_window}")
    click.echo(f"  Transaction cost  : {cost:.4%}")

    rpt = generate_signal_report(
        df,
        price_history=None,
        output_dir=out_path,
        signal_col=sig_col,
        return_col=return_col,
        dollar_pos_col=pos_col,
        max_decay_horizon=horizon,
        n_quantiles=quantiles,
        rolling_ic_window=rolling_window,
        transaction_cost=cost,
    )

    to_console(rpt)

    if emit_html:
        html_path = out_path / "signal_report.html"
        to_html(rpt, path=html_path)
        click.echo(f"\nHTML report saved to: {html_path}")

    click.echo(f"\nAll outputs saved to: {out_path}")


# ---------------------------------------------------------------------------
# signal quintile
# ---------------------------------------------------------------------------

@cli.command("quintile")
@click.option("--trades", required=True, type=click.Path(exists=True), help="Trades CSV path.")
@click.option("--signal", default=None, help="Signal column name.")
@click.option("--quantiles", default=5, show_default=True, help="Number of quantile buckets.")
@click.option("--output", default=None, help="Directory to save quintile plots.")
@click.option("--return-col", default="pnl", show_default=True, help="Return/P&L column.")
@click.option("--pos-col", default="dollar_pos", show_default=True, help="Dollar position column.")
def cmd_quintile(
    trades: str,
    signal: Optional[str],
    quantiles: int,
    output: Optional[str],
    return_col: str,
    pos_col: str,
) -> None:
    """Quintile portfolio analysis.

    Sorts trades by signal strength and reports return statistics per quintile.

    \b
    Example:
      signal quintile --trades crypto_trades.csv --quantiles 5
    """
    from research.signal_analytics.quantile_analysis import QuantileAnalyzer
    import matplotlib.pyplot as plt

    df = _load_trades(trades)
    sig_col = _infer_signal_col(df, signal)

    if pos_col in df.columns:
        pos = df[pos_col].abs().replace(0, np.nan)
        df["_ret"] = df[return_col] / pos
    else:
        df["_ret"] = df[return_col]

    qa = QuantileAnalyzer(n_quantiles=quantiles)
    try:
        result = qa.compute_quintile_returns(df[sig_col], df["_ret"], signal_col=sig_col)
    except Exception as e:
        click.echo(f"[ERROR] Quintile analysis failed: {e}", err=True)
        sys.exit(1)

    click.echo("\n" + "=" * 65)
    click.echo("QUINTILE ANALYSIS")
    click.echo("=" * 65)
    click.echo(f"  Signal         : {sig_col}")
    click.echo(f"  N quantiles    : {quantiles}")
    click.echo(f"  Q5-Q1 Spread   : {result.spread:.4f}  (t={result.spread_t_stat:.2f}  p={result.spread_p_value:.4f})")
    click.echo(f"  Monotonicity   : {result.monotonicity_score:.2f}")
    click.echo()
    click.echo(f"{'Quantile':>10}  {'N':>8}  {'Mean Ret':>12}  {'Hit Rate':>10}  {'Std':>10}")
    click.echo("-" * 58)
    for q, n, mr, hr, sd in zip(
        result.quantile_labels, result.n_obs, result.mean_returns,
        result.hit_rates, result.std_returns
    ):
        click.echo(f"{q:>10}  {n:>8}  {mr:>12.4f}  {hr:>10.1%}  {sd:>10.4f}")
    click.echo("=" * 65)

    if output:
        out_path = Path(output)
        out_path.mkdir(parents=True, exist_ok=True)
        fig = qa.plot_quintile_bar(result, save_path=out_path / "quintile_returns.png")
        plt.close(fig)
        click.echo(f"\nQuintile plot saved to: {out_path / 'quintile_returns.png'}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main CLI entry-point."""
    cli()


# ---------------------------------------------------------------------------
# signal portfolio
# ---------------------------------------------------------------------------

@cli.command("portfolio")
@click.option("--trades", required=True, type=click.Path(exists=True), help="Trades CSV path.")
@click.option("--signal", default=None, help="Signal column name (delta_score etc).")
@click.option("--output", default=None, help="Output directory.")
@click.option("--return-col", default="pnl", show_default=True, help="Return/P&L column.")
@click.option("--pos-col", default="dollar_pos", show_default=True, help="Dollar position column.")
@click.option("--max-capacity", default=1e7, show_default=True, help="Max AUM for capacity curve.")
def cmd_portfolio(
    trades: str,
    signal: Optional[str],
    output: Optional[str],
    return_col: str,
    pos_col: str,
    max_capacity: float,
) -> None:
    """Portfolio-level signal analytics.

    Computes signal concentration, diversification, capacity, and
    turnover-cost-adjusted IC.

    \b
    Example:
      signal portfolio --trades crypto_trades.csv --output results/portfolio/
    """
    from research.signal_analytics.portfolio_signal import PortfolioSignalAnalyzer
    import matplotlib.pyplot as plt

    df = _load_trades(trades)
    sig_col = _infer_signal_col(df, signal)
    psa = PortfolioSignalAnalyzer(signal_col=sig_col, dollar_pos_col=pos_col, return_col=return_col)

    # Concentration (single-period: use whole trades table)
    sig_series = df[sig_col].dropna() if sig_col in df.columns else pd.Series(dtype=float)

    click.echo("\n" + "=" * 55)
    click.echo("PORTFOLIO SIGNAL ANALYTICS")
    click.echo("=" * 55)
    click.echo(f"  Signal column : {sig_col}")
    click.echo(f"  N trades      : {len(df)}")

    if sig_col in df.columns:
        hhi = psa.signal_concentration(sig_series)
        click.echo(f"  HHI           : {hhi:.4f}")

        # Turnover-cost impact
        cost_impact = psa.turnover_cost_impact(df, transaction_cost=0.0002)
        click.echo(f"\n  Gross IC      : {cost_impact.get('gross_ic', float('nan')):.4f}")
        click.echo(f"  Net IC        : {cost_impact.get('net_ic', float('nan')):.4f}")
        click.echo(f"  Cost drag     : {cost_impact.get('cost_drag', float('nan')):.6f}")
        click.echo(f"  N obs         : {cost_impact.get('n_trades', 0)}")

        # Capacity (if we have return data)
        from research.signal_analytics.ic_framework import ICCalculator
        from scipy import stats as _stats
        if return_col in df.columns:
            if pos_col in df.columns:
                pos = df[pos_col].abs().replace(0, float("nan"))
                df["_ret"] = df[return_col] / pos
            else:
                df["_ret"] = df[return_col]
            sub = df[[sig_col, "_ret"]].dropna()
            if len(sub) >= 5:
                cap = psa.signal_capacity(sub[sig_col], sub["_ret"], max_capacity=max_capacity)
                click.echo(f"\n  Estimated capacity   : ${cap.estimated_capacity_usd:,.0f}")
                click.echo(f"  Decay onset          : ${cap.decay_onset_usd:,.0f}")
                click.echo(f"  Current IC           : {cap.current_ic:.4f}")
                click.echo(f"  IC at capacity       : {cap.capacity_ic:.4f}")

                if output:
                    out_path = Path(output)
                    out_path.mkdir(parents=True, exist_ok=True)
                    fig_cap = psa.plot_capacity_curve(cap, max_capacity=max_capacity,
                                                      save_path=out_path / "capacity_curve.png")
                    plt.close(fig_cap)
                    click.echo(f"\nCapacity curve saved to: {out_path / 'capacity_curve.png'}")

    click.echo("=" * 55)


# ---------------------------------------------------------------------------
# signal compare
# ---------------------------------------------------------------------------

@cli.command("compare")
@click.option("--trades", required=True, type=click.Path(exists=True), help="Trades CSV path.")
@click.option("--signals", required=True, help="Comma-separated list of signal columns to compare.")
@click.option("--output", default=None, help="Output directory.")
@click.option("--return-col", default="pnl", show_default=True, help="Return/P&L column.")
@click.option("--pos-col", default="dollar_pos", show_default=True, help="Dollar position column.")
def cmd_compare(
    trades: str,
    signals: str,
    output: Optional[str],
    return_col: str,
    pos_col: str,
) -> None:
    """Compare multiple signal columns side-by-side.

    Computes IC, t-stat, half-life, win rate, and Sharpe for each signal.

    \b
    Example:
      signal compare --trades crypto_trades.csv \\
          --signals "ensemble_signal,delta_score,tf_score,mass"
    """
    from research.signal_analytics.report import compare_signals_report

    df = _load_trades(trades)
    sig_cols = [s.strip() for s in signals.split(",")]
    missing = [c for c in sig_cols if c not in df.columns]
    if missing:
        click.echo(f"[WARNING] Signals not in trades: {missing}", err=True)

    sig_cols = [c for c in sig_cols if c in df.columns]
    if not sig_cols:
        click.echo("[ERROR] No valid signal columns found.", err=True)
        sys.exit(1)

    out_dir = output or "results/signal_compare/"
    comparison_df = compare_signals_report(
        df, sig_cols, output_dir=out_dir,
        return_col=return_col, dollar_pos_col=pos_col,
    )

    click.echo("\n" + "=" * 80)
    click.echo("SIGNAL COMPARISON")
    click.echo("=" * 80)
    click.echo(comparison_df.to_string())
    click.echo("=" * 80)
    click.echo(f"\nResults saved to: {out_dir}")


# ---------------------------------------------------------------------------
# signal validate
# ---------------------------------------------------------------------------

@cli.command("validate")
@click.option("--trades", required=True, type=click.Path(exists=True), help="Trades CSV path.")
@click.option("--strict/--no-strict", default=False, show_default=True,
              help="Fail on any missing optional column.")
def cmd_validate(trades: str, strict: bool) -> None:
    """Validate a trades CSV for signal analytics compatibility.

    Checks for required and recommended columns, NaN rates, and
    minimum observation counts.

    \b
    Example:
      signal validate --trades crypto_trades.csv
    """
    from research.signal_analytics.utils import validate_trades

    df = _load_trades(trades)
    is_valid, issues = validate_trades(df, raise_on_error=False)

    click.echo("\n" + "=" * 60)
    click.echo("TRADE DATA VALIDATION")
    click.echo("=" * 60)
    click.echo(f"  File      : {trades}")
    click.echo(f"  Rows      : {len(df)}")
    click.echo(f"  Columns   : {', '.join(df.columns.tolist())}")
    click.echo(f"  Valid     : {'YES' if is_valid else 'NO'}")

    if issues:
        click.echo("\n  Issues found:")
        for issue in issues:
            level = "[ERROR]" if "required" in issue.lower() else "[WARN] "
            click.echo(f"    {level} {issue}")
    else:
        click.echo("\n  No issues found.")

    # NaN rates
    click.echo("\n  NaN rates per column:")
    for col in df.columns:
        nan_rate = df[col].isna().mean()
        if nan_rate > 0:
            click.echo(f"    {col:<30} {nan_rate:.1%} NaN")

    # Basic stats for numeric columns
    click.echo("\n  Numeric column statistics:")
    numeric = df.select_dtypes(include=[np.number])
    for col in numeric.columns[:10]:  # Limit to first 10
        click.echo(
            f"    {col:<30} "
            f"mean={numeric[col].mean():.4f}  "
            f"std={numeric[col].std():.4f}  "
            f"min={numeric[col].min():.4f}  "
            f"max={numeric[col].max():.4f}"
        )

    click.echo("=" * 60)
    if not is_valid and strict:
        sys.exit(1)


# ---------------------------------------------------------------------------
# signal synthetic
# ---------------------------------------------------------------------------

@cli.command("synthetic")
@click.option("--n-trades", default=500, show_default=True, help="Number of synthetic trades.")
@click.option("--ic", default=0.05, show_default=True, help="Target IC (signal quality).")
@click.option("--output", required=True, type=click.Path(), help="Output CSV path for trades.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--n-symbols", default=10, show_default=True, help="Number of symbols.")
def cmd_synthetic(
    n_trades: int,
    ic: float,
    output: str,
    seed: int,
    n_symbols: int,
) -> None:
    """Generate synthetic BH-style trade data for testing.

    Creates a CSV with all BH signal columns (tf_score, mass, ATR,
    ensemble_signal, delta_score) with a controllable IC.

    \b
    Example:
      signal synthetic --n-trades 1000 --ic 0.08 --output data/synthetic_trades.csv
    """
    from research.signal_analytics.utils import generate_synthetic_trades

    click.echo(f"\nGenerating {n_trades} synthetic trades with IC={ic}...")
    trades = generate_synthetic_trades(
        n_trades=n_trades, n_symbols=n_symbols,
        ic=ic, seed=seed, include_bh_signals=True,
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_path, index=False)

    click.echo(f"Saved to: {out_path}")
    click.echo(f"Columns : {', '.join(trades.columns.tolist())}")
    click.echo(f"Shape   : {trades.shape}")

    # Verify IC
    from scipy import stats as _stats
    sub = trades[["ensemble_signal", "pnl"]].dropna()
    if len(sub) > 3:
        r, p = _stats.spearmanr(sub["ensemble_signal"], sub["pnl"])
        click.echo(f"\nVerified IC (signal vs raw pnl): {float(r):.4f}  (target={ic})")


# ---------------------------------------------------------------------------
# signal regime – regime-conditioned signal analysis
# ---------------------------------------------------------------------------


@main.command("regime")
@click.argument("trades_csv", type=click.Path(exists=True))
@click.option("--signal", "signal_col", default="delta_score", show_default=True,
              help="Signal column name.")
@click.option("--regime-col", default="regime", show_default=True,
              help="Column containing discrete regime labels.")
@click.option("--vol-window", default=20, show_default=True,
              help="Rolling window for volatility regime detection.")
@click.option("--auto-detect", is_flag=True, default=False,
              help="Auto-detect volatility regimes even if --regime-col exists.")
@click.option("--adaptive-filters", is_flag=True, default=False,
              help="Run adaptive BH filter grid search per regime.")
@click.option("--mass-col", default="mass", show_default=True,
              help="Mass column for adaptive filter search.")
@click.option("--tf-col", default="tf_score", show_default=True,
              help="TF-score column for adaptive filter search.")
@click.option("--output-dir", "-o", default=".", show_default=True,
              help="Directory for output files and plots.")
@click.option("--plot/--no-plot", default=True, show_default=True,
              help="Generate regime IC bar chart and transition heatmap.")
def regime_cmd(
    trades_csv: str,
    signal_col: str,
    regime_col: str,
    vol_window: int,
    auto_detect: bool,
    adaptive_filters: bool,
    mass_col: str,
    tf_col: str,
    output_dir: str,
    plot: bool,
) -> None:
    """Regime-conditioned signal quality analysis.

    Conditions IC, ICIR and hit-rate on market regime (volatility, trend,
    or custom discrete labels supplied in the trade log).

    \b
    Examples
    --------
    signal regime trades.csv --regime-col regime
    signal regime trades.csv --auto-detect --vol-window 30
    signal regime trades.csv --adaptive-filters --mass-col mass --tf-col tf_score
    """
    import json
    from pathlib import Path

    from research.signal_analytics.utils import load_trades, validate_trades
    from research.signal_analytics.regime_signals import RegimeSignalAnalyzer

    trades = load_trades(trades_csv)
    errs = validate_trades(trades)
    if errs:
        click.echo("Trade validation issues:", err=True)
        for e in errs:
            click.echo(f"  - {e}", err=True)

    analyzer = RegimeSignalAnalyzer(trades, signal_col=signal_col)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if auto_detect or regime_col not in trades.columns:
        click.echo(f"Auto-detecting volatility regimes (window={vol_window})...")
        summary = analyzer.analyse_vol_regimes(window=vol_window)
        regime_col = "_vol_regime"
    else:
        click.echo(f"Using regime column: '{regime_col}'")
        summary = analyzer.analyse_by_regime_column(regime_col)

    click.echo(f"\nRegime Analysis Summary ({summary.regime_type})")
    click.echo(f"  Total bars   : {summary.n_total_bars:,}")
    click.echo(f"  N regimes    : {summary.n_regimes}")
    click.echo(f"  Best regime  : {summary.best_regime}  (IC={summary.per_regime.get(summary.best_regime, type('', (), {'signal_ic': float('nan')})()).signal_ic:.4f})")
    click.echo(f"  Worst regime : {summary.worst_regime}")
    click.echo(f"  IC range     : {summary.ic_range:.4f}")
    click.echo(f"  IC sensitivity (std): {summary.regime_sensitivity:.4f}")

    click.echo("\nPer-Regime Details:")
    click.echo(f"  {'Regime':<20} {'N':>6}  {'Freq':>6}  {'IC':>7}  {'Sharpe':>7}  {'HitRate':>8}")
    click.echo("  " + "-" * 60)
    for rname, rs in sorted(summary.per_regime.items()):
        click.echo(
            f"  {rname:<20} {rs.n_bars:>6,}  {rs.freq:>6.1%}  "
            f"{rs.signal_ic:>7.4f}  {rs.sharpe:>7.3f}  {rs.hit_rate:>8.2%}"
        )

    if adaptive_filters and all(c in trades.columns for c in [mass_col, tf_col, regime_col]):
        click.echo("\nRunning adaptive BH filter grid search per regime...")
        filter_res = analyzer.adaptive_signal_filter_by_regime(
            mass_col=mass_col, tf_score_col=tf_col, regime_col=regime_col
        )
        click.echo(f"\n{'Regime':<20}  {'BestMass':>10}  {'BestTF':>8}  {'ICIR':>8}")
        click.echo("  " + "-" * 52)
        for rname, afr in filter_res.items():
            click.echo(f"  {rname:<20}  {afr.best_mass_min:>10.2f}  "
                       f"{afr.best_tf_score_min:>8d}  {afr.best_icir:>8.3f}")

    if plot:
        try:
            fig_ic = analyzer.plot_ic_by_regime(regime_col)
            fig_path = out_path / "regime_ic.png"
            fig_ic.savefig(fig_path, dpi=150, bbox_inches="tight")
            click.echo(f"\nSaved regime IC chart -> {fig_path}")
            import matplotlib.pyplot as plt
            plt.close(fig_ic)

            fig_tm = analyzer.plot_regime_transition_heatmap(regime_col)
            tm_path = out_path / "regime_transitions.png"
            fig_tm.savefig(tm_path, dpi=150, bbox_inches="tight")
            click.echo(f"Saved transition heatmap -> {tm_path}")
            plt.close(fig_tm)
        except Exception as exc:
            click.echo(f"[warning] Plot failed: {exc}", err=True)


# ---------------------------------------------------------------------------
# signal score – score signals with SignalScorer
# ---------------------------------------------------------------------------


@main.command("score")
@click.argument("trades_csv", type=click.Path(exists=True))
@click.option("--signals", "-s", multiple=True, default=["delta_score"],
              show_default=True, help="Signal columns to score (repeatable).")
@click.option("--ic-window", default=30, show_default=True,
              help="Rolling window for IC weight computation.")
@click.option("--transaction-cost", default=0.0002, show_default=True,
              help="Per-bar transaction cost (fraction of notional).")
@click.option("--min-obs", default=20, show_default=True,
              help="Minimum observations per signal to score.")
@click.option("--output-dir", "-o", default=".", show_default=True,
              help="Directory for output files.")
@click.option("--plot/--no-plot", default=True, show_default=True,
              help="Generate signal score summary plots.")
def score_cmd(
    trades_csv: str,
    signals: tuple[str, ...],
    ic_window: int,
    transaction_cost: float,
    min_obs: int,
    output_dir: str,
    plot: bool,
) -> None:
    """Score and rank multiple signals using IC-weighted metrics.

    Computes composite signal scores (0-100) incorporating IC, ICIR, decay-
    adjusted sizing, and optional transaction cost adjustment.

    \b
    Examples
    --------
    signal score trades.csv -s delta_score -s ensemble_signal
    signal score trades.csv -s delta_score --ic-window 20 --transaction-cost 0.0005
    """
    from pathlib import Path

    from research.signal_analytics.utils import load_trades, validate_trades
    from research.signal_analytics.scoring import SignalScorer

    trades = load_trades(trades_csv)
    errs = validate_trades(trades)
    if errs:
        for e in errs:
            click.echo(f"[warn] {e}", err=True)

    signal_list = list(signals)
    available = [s for s in signal_list if s in trades.columns]
    missing = [s for s in signal_list if s not in trades.columns]
    if missing:
        click.echo(f"[warn] Signals not found in trades: {missing}", err=True)
    if not available:
        click.echo("[error] No valid signal columns found.", err=True)
        raise click.Abort()

    scorer = SignalScorer(
        trades=trades,
        signal_cols=available,
        ic_window=ic_window,
        transaction_cost=transaction_cost,
        min_obs=min_obs,
    )

    click.echo(f"\nScoring {len(available)} signal(s): {available}")
    scores_df = scorer.score_all_signals()

    click.echo("\nSignal Scores (0-100):")
    click.echo(scores_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_out = out_path / "signal_scores.csv"
    scores_df.to_csv(csv_out, index=False)
    click.echo(f"\nSaved scores -> {csv_out}")

    if plot:
        try:
            fig = scorer.plot_score_summary()
            fig_path = out_path / "signal_score_summary.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            click.echo(f"Saved score chart -> {fig_path}")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as exc:
            click.echo(f"[warning] Plot failed: {exc}", err=True)


# ---------------------------------------------------------------------------
# signal hurst – Hurst exponent analysis
# ---------------------------------------------------------------------------


@main.command("hurst")
@click.argument("trades_csv", type=click.Path(exists=True))
@click.option("--min-lag", default=10, show_default=True,
              help="Minimum lag for R/S analysis.")
@click.option("--max-lag", default=None, type=int, show_default=True,
              help="Maximum lag for R/S analysis (default n//4).")
@click.option("--return-col", default=None,
              help="Return column to analyse (default: auto-compute from prices).")
@click.option("--plot/--no-plot", default=True, show_default=True,
              help="Generate R/S log-log plot.")
@click.option("--output-dir", "-o", default=".", show_default=True,
              help="Directory for output files.")
def hurst_cmd(
    trades_csv: str,
    min_lag: int,
    max_lag: int | None,
    return_col: str | None,
    plot: bool,
    output_dir: str,
) -> None:
    """Estimate Hurst exponent of trade returns for regime classification.

    The Hurst exponent H indicates the memory in the return series:
    H > 0.55 -> trending / persistent signal
    H ~ 0.50 -> random walk
    H < 0.45 -> mean-reverting / anti-persistent

    \b
    Examples
    --------
    signal hurst trades.csv
    signal hurst trades.csv --min-lag 5 --max-lag 100
    """
    from pathlib import Path

    from research.signal_analytics.utils import load_trades
    from research.signal_analytics.regime_signals import hurst_exponent

    trades = load_trades(trades_csv)

    if return_col and return_col in trades.columns:
        ret_series = trades[return_col].dropna()
    elif "entry_price" in trades.columns and "exit_price" in trades.columns:
        ret_series = (
            (trades["exit_price"] - trades["entry_price"]) / trades["entry_price"].replace(0, float("nan"))
        ).dropna()
        click.echo("Auto-computed percentage returns from entry/exit prices.")
    elif "pnl" in trades.columns and "dollar_pos" in trades.columns:
        ret_series = (trades["pnl"] / trades["dollar_pos"].replace(0, float("nan"))).dropna()
        click.echo("Auto-computed pnl/dollar_pos returns.")
    else:
        click.echo("[error] Cannot derive return series. Provide --return-col.", err=True)
        raise click.Abort()

    result = hurst_exponent(ret_series, min_lag=min_lag, max_lag=max_lag)

    click.echo(f"\nHurst Exponent Analysis")
    click.echo(f"  Series length  : {len(ret_series):,}")
    click.echo(f"  Hurst (H)      : {result.hurst:.4f}")
    click.echo(f"  Interpretation : {result.interpretation}")
    click.echo(f"  R^2 (log-log)  : {result.r_squared:.4f}")
    click.echo(f"  Lag range      : [{min(result.lag_series)}, {max(result.lag_series)}]  ({len(result.lag_series)} points)")

    if result.hurst > 0.55:
        click.echo("\n  => Return series shows TREND persistence. Momentum signals may be effective.")
    elif result.hurst < 0.45:
        click.echo("\n  => Return series is MEAN-REVERTING. Counter-trend signals may outperform.")
    else:
        click.echo("\n  => Return series is approximately a RANDOM WALK. Signal IC may be low.")

    if plot and len(result.lag_series) >= 3:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(7, 5))
            log_lags = np.log(result.lag_series)
            log_rs = np.log(result.rs_series)
            ax.scatter(log_lags, log_rs, color="steelblue", s=30, zorder=3, label="R/S")
            m, b = np.polyfit(log_lags, log_rs, 1)
            ax.plot(log_lags, m * log_lags + b, "r--", label=f"H={m:.3f}")
            ax.set_xlabel("log(lag)")
            ax.set_ylabel("log(R/S)")
            ax.set_title(f"R/S Analysis — Hurst H={result.hurst:.4f} ({result.interpretation})")
            ax.legend()
            fig.tight_layout()

            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / "hurst_rs.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            click.echo(f"\nSaved R/S plot -> {path}")
            plt.close(fig)
        except Exception as exc:
            click.echo(f"[warning] Plot failed: {exc}", err=True)


# ---------------------------------------------------------------------------
# signal ic-grid – IC significance grid across multiple signals/horizons
# ---------------------------------------------------------------------------


@main.command("ic-grid")
@click.argument("trades_csv", type=click.Path(exists=True))
@click.option("--signals", "-s", multiple=True,
              help="Signal columns to include (default: auto-detect numeric columns).")
@click.option("--max-horizon", default=10, show_default=True,
              help="Maximum forward-return horizon in bars.")
@click.option("--alpha", default=0.05, show_default=True,
              help="Significance level for IC t-test.")
@click.option("--output", "-o", default="ic_grid.csv", show_default=True,
              help="Output CSV path for the IC grid.")
@click.option("--plot/--no-plot", default=True, show_default=True,
              help="Generate heatmap of IC x horizon grid.")
def ic_grid_cmd(
    trades_csv: str,
    signals: tuple[str, ...],
    max_horizon: int,
    alpha: float,
    output: str,
    plot: bool,
) -> None:
    """IC significance grid across signals and forward-return horizons.

    Produces a matrix showing which (signal, horizon) pairs have statistically
    significant IC, helping identify the optimal holding period per signal.

    \b
    Examples
    --------
    signal ic-grid trades.csv -s delta_score -s ensemble_signal --max-horizon 15
    signal ic-grid trades.csv --alpha 0.01 --no-plot
    """
    from pathlib import Path

    import numpy as np
    from scipy import stats as _stats

    from research.signal_analytics.utils import load_trades, ic_significance_grid

    trades = load_trades(trades_csv)

    if signals:
        signal_list = [s for s in signals if s in trades.columns]
    else:
        # Auto-detect numeric columns that are not schema columns
        schema_cols = {"exit_time", "sym", "entry_price", "exit_price",
                       "dollar_pos", "pnl", "hold_bars", "regime"}
        signal_list = [c for c in trades.select_dtypes(include=[float, int]).columns
                       if c not in schema_cols]

    if not signal_list:
        click.echo("[error] No valid signal columns found.", err=True)
        raise click.Abort()

    click.echo(f"Computing IC grid: {len(signal_list)} signals x {max_horizon} horizons")

    # Compute forward returns at each horizon via hold_bars proxy
    if "hold_bars" not in trades.columns:
        click.echo("[error] 'hold_bars' column required for horizon IC grid.", err=True)
        raise click.Abort()

    ret_col = "_ret_pct"
    if ret_col not in trades.columns:
        if "entry_price" in trades.columns and "exit_price" in trades.columns:
            trades[ret_col] = (trades["exit_price"] - trades["entry_price"]) / trades["entry_price"].replace(0, float("nan"))
        elif "pnl" in trades.columns and "dollar_pos" in trades.columns:
            trades[ret_col] = trades["pnl"] / trades["dollar_pos"].replace(0, float("nan"))

    grid_data: dict[str, list[float]] = {s: [] for s in signal_list}
    horizons = list(range(1, max_horizon + 1))

    for h in horizons:
        mask = trades["hold_bars"] >= h
        sub = trades[mask]
        for sig_col in signal_list:
            if sig_col not in sub.columns:
                grid_data[sig_col].append(float("nan"))
                continue
            sig = sub[sig_col].values.astype(float)
            ret = sub[ret_col].values.astype(float)
            valid = np.isfinite(sig) & np.isfinite(ret)
            if valid.sum() < 10:
                grid_data[sig_col].append(float("nan"))
            else:
                rho, pval = _stats.spearmanr(sig[valid], ret[valid])
                # Annotate with significance
                grid_data[sig_col].append(float(rho) if pval <= alpha else 0.0)

    import pandas as pd
    grid_df = pd.DataFrame(grid_data, index=horizons)
    grid_df.index.name = "horizon"

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(out_path)
    click.echo(f"\nIC grid (alpha={alpha}, zeros = not significant):")
    click.echo(grid_df.to_string(float_format=lambda x: f"{x:+.4f}"))
    click.echo(f"\nSaved grid -> {out_path}")

    if plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(max(6, len(signal_list) * 1.5), max(5, max_horizon * 0.4)))
            sns.heatmap(grid_df.T, ax=ax, cmap="RdYlGn", center=0,
                        annot=True, fmt=".2f", linewidths=0.3,
                        cbar_kws={"label": "IC (0=not sig.)"})
            ax.set_title(f"IC Significance Grid (alpha={alpha})", fontsize=13, fontweight="bold")
            ax.set_xlabel("Forward Return Horizon (bars)")
            ax.set_ylabel("Signal")
            fig.tight_layout()
            img_path = out_path.with_suffix(".png")
            fig.savefig(img_path, dpi=150, bbox_inches="tight")
            click.echo(f"Saved heatmap -> {img_path}")
            plt.close(fig)
        except Exception as exc:
            click.echo(f"[warning] Plot failed: {exc}", err=True)


if __name__ == "__main__":
    main()
