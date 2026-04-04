"""
research/signal_analytics/report.py
=====================================
Signal analytics report generation.

Produces a comprehensive IC tearsheet integrating:
  - IC analysis (overall, by regime, by quantile)
  - Alpha decay curve and half-life
  - Factor attribution waterfall
  - Quintile bar chart
  - BH activation quality
  - HTML and console output

Usage example
-------------
>>> report = generate_signal_report(trades, price_history, output_dir="results/signal/")
>>> to_html(report, path="results/signal/report.html")
>>> to_console(report)
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.signal_analytics.ic_framework import ICCalculator, ICDecayResult
from research.signal_analytics.alpha_decay import AlphaDecayAnalyzer, DecayModel
from research.signal_analytics.factor_model import FactorModel, AttributionResult
from research.signal_analytics.quantile_analysis import QuantileAnalyzer, QuantileResult
from research.signal_analytics.bh_signal_quality import BHSignalAnalyzer, ActivationQualityReport
from research.signal_analytics.portfolio_signal import PortfolioSignalAnalyzer


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignalAnalyticsReport:
    """Comprehensive signal analytics report container."""

    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    n_trades: int = 0
    date_range: str = ""

    # IC Analysis
    ic_summary: Dict[str, float] = field(default_factory=dict)
    ic_decay_result: Optional[ICDecayResult] = None
    decay_model: Optional[DecayModel] = None
    rolling_ic_path: str = ""
    ic_decay_path: str = ""
    ic_distribution_path: str = ""

    # Factor Attribution
    attribution_result: Optional[AttributionResult] = None
    factor_ics: Dict[str, float] = field(default_factory=dict)
    factor_ic_bar_path: str = ""
    attribution_waterfall_path: str = ""

    # Quintile Analysis
    quintile_result: Optional[QuantileResult] = None
    quintile_bar_path: str = ""
    quintile_cumulative_path: str = ""

    # BH Signal Quality
    activation_quality: Optional[ActivationQualityReport] = None
    bh_dashboard_path: str = ""

    # Alpha Decay
    optimal_holding_bars: int = 0
    halflife_bars: float = float("nan")
    alpha_decay_path: str = ""

    # Portfolio
    signal_concentration_hhi: float = float("nan")
    portfolio_signal_path: str = ""

    # Raw data references
    output_dir: str = ""

    def summary_dict(self) -> Dict[str, Any]:
        """Produce a flat summary dictionary for serialisation."""
        return {
            "generated_at": self.generated_at,
            "n_trades": self.n_trades,
            "date_range": self.date_range,
            "ic_mean": self.ic_summary.get("mean_ic", float("nan")),
            "icir": self.ic_summary.get("icir", float("nan")),
            "ic_pct_positive": self.ic_summary.get("pct_positive", float("nan")),
            "ic_t_stat": self.ic_summary.get("t_stat", float("nan")),
            "ic_p_value": self.ic_summary.get("p_value", float("nan")),
            "half_life_bars": self.halflife_bars,
            "optimal_holding_bars": self.optimal_holding_bars,
            "signal_concentration_hhi": self.signal_concentration_hhi,
            "activation_win_rate": self.activation_quality.win_rate if self.activation_quality else float("nan"),
            "activation_total_pnl": self.activation_quality.total_pnl if self.activation_quality else float("nan"),
            "quintile_spread": self.quintile_result.spread if self.quintile_result else float("nan"),
            "quintile_monotonicity": self.quintile_result.monotonicity_score if self.quintile_result else float("nan"),
        }


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_signal_report(
    trades: pd.DataFrame,
    price_history: Optional[pd.DataFrame] = None,
    output_dir: str | Path = "results/signal_analytics/",
    signal_col: str = "ensemble_signal",
    return_col: str = "pnl",
    dollar_pos_col: str = "dollar_pos",
    max_decay_horizon: int = 20,
    n_quantiles: int = 5,
    rolling_ic_window: int = 60,
    transaction_cost: float = 0.0002,
) -> SignalAnalyticsReport:
    """Generate a comprehensive signal analytics report.

    Parameters
    ----------
    trades           : trade DataFrame with BH signal columns
    price_history    : optional wide price panel for factor construction
    output_dir       : directory to save figures and data
    signal_col       : primary signal column
    return_col       : P&L column
    dollar_pos_col   : position size column
    max_decay_horizon: maximum IC decay horizon
    n_quantiles      : quantile analysis buckets
    rolling_ic_window: rolling IC window in periods
    transaction_cost : one-way cost fraction

    Returns
    -------
    SignalAnalyticsReport
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = SignalAnalyticsReport(output_dir=str(output_path))
    report.n_trades = len(trades)

    # Date range
    if "exit_time" in trades.columns:
        dates = pd.to_datetime(trades["exit_time"], errors="coerce").dropna()
        if len(dates) > 0:
            report.date_range = f"{dates.min().date()} → {dates.max().date()}"

    # Normalised returns
    df = trades.copy()
    if dollar_pos_col in df.columns:
        pos = df[dollar_pos_col].abs().replace(0, np.nan)
        df["_ret"] = df[return_col] / pos
    else:
        df["_ret"] = df[return_col]

    # ------------------------------------------------------------------ #
    # 1. IC Analysis
    # ------------------------------------------------------------------ #
    calc = ICCalculator()
    report.ic_summary = _compute_ic_summary(calc, df, signal_col)

    # IC Decay from trades
    if signal_col in df.columns and "hold_bars" in df.columns:
        decay_result = calc.ic_decay_from_trades(
            df,
            signal_col=signal_col,
            return_col=return_col,
            hold_col="hold_bars",
            dollar_pos_col=dollar_pos_col,
            max_horizon=max_decay_horizon,
        )
        report.ic_decay_result = decay_result

        decay_fig = calc.plot_ic_decay(decay_result, save_path=output_path / "ic_decay.png")
        plt.close(decay_fig)
        report.ic_decay_path = str(output_path / "ic_decay.png")

    # Rolling IC (time-based if exit_time available)
    if "exit_time" in df.columns and signal_col in df.columns:
        rolling_ic_series = _compute_rolling_ic_from_trades(df, signal_col, window=rolling_ic_window)
        if len(rolling_ic_series) > 5:
            fig_ric = calc.plot_rolling_ic(rolling_ic_series, save_path=output_path / "rolling_ic.png")
            plt.close(fig_ric)
            report.rolling_ic_path = str(output_path / "rolling_ic.png")

            # IC distribution
            fig_dist = calc.plot_ic_distribution(rolling_ic_series, save_path=output_path / "ic_distribution.png")
            plt.close(fig_dist)
            report.ic_distribution_path = str(output_path / "ic_distribution.png")

    # ------------------------------------------------------------------ #
    # 2. Alpha Decay
    # ------------------------------------------------------------------ #
    ada = AlphaDecayAnalyzer(default_cost=transaction_cost)
    if report.ic_decay_result is not None:
        decay_model = ada.signal_decay_model(report.ic_decay_result)
        report.decay_model = decay_model
        report.halflife_bars = ada.compute_signal_halflife(report.ic_decay_result)
        report.optimal_holding_bars = ada.optimal_holding_period(
            report.ic_decay_result, transaction_cost=transaction_cost
        )
        decay_fig2 = ada.plot_alpha_decay(
            report.ic_decay_result,
            save_path=output_path / "alpha_decay.png",
            transaction_cost=transaction_cost,
        )
        plt.close(decay_fig2)
        report.alpha_decay_path = str(output_path / "alpha_decay.png")

    # ------------------------------------------------------------------ #
    # 3. Quintile Analysis
    # ------------------------------------------------------------------ #
    if signal_col in df.columns:
        qa = QuantileAnalyzer(n_quantiles=n_quantiles)
        try:
            q_result = qa.compute_quintile_returns(
                df[signal_col], df["_ret"], signal_col=signal_col
            )
            report.quintile_result = q_result
            fig_q = qa.plot_quintile_bar(q_result, save_path=output_path / "quintile_returns.png")
            plt.close(fig_q)
            report.quintile_bar_path = str(output_path / "quintile_returns.png")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # 4. BH Signal Quality
    # ------------------------------------------------------------------ #
    bh = BHSignalAnalyzer(
        signal_col=signal_col,
        return_col=return_col,
        dollar_pos_col=dollar_pos_col,
    )
    try:
        aq_report = bh.activation_quality(trades)
        report.activation_quality = aq_report
        fig_bh = bh.plot_signal_quality_dashboard(
            aq_report, trades=trades, save_path=output_path / "bh_dashboard.png"
        )
        plt.close(fig_bh)
        report.bh_dashboard_path = str(output_path / "bh_dashboard.png")
    except Exception:
        pass

    # ------------------------------------------------------------------ #
    # 5. Factor Attribution
    # ------------------------------------------------------------------ #
    fm = FactorModel()
    try:
        attr = fm.factor_attribution(
            trades,
            price_history=price_history,
            return_col=return_col,
            dollar_pos_col=dollar_pos_col,
        )
        report.attribution_result = attr

        # Factor ICs
        factor_df = fm.build_factor_matrix(trades, price_history)
        fret = df["_ret"].loc[factor_df.index].dropna()
        report.factor_ics = fm.factor_ic(factor_df.loc[fret.index], fret)

        fig_fic = fm.plot_factor_ic_bar(
            report.factor_ics, save_path=output_path / "factor_ic_bar.png"
        )
        plt.close(fig_fic)
        report.factor_ic_bar_path = str(output_path / "factor_ic_bar.png")

        fig_attr = fm.plot_attribution_waterfall(
            attr, save_path=output_path / "attribution_waterfall.png"
        )
        plt.close(fig_attr)
        report.attribution_waterfall_path = str(output_path / "attribution_waterfall.png")
    except Exception:
        pass

    # ------------------------------------------------------------------ #
    # 6. Portfolio Signal Concentration
    # ------------------------------------------------------------------ #
    if signal_col in trades.columns and "sym" in trades.columns:
        psa = PortfolioSignalAnalyzer(signal_col=signal_col)
        try:
            sig_pivot = trades.pivot_table(
                index=trades.index, columns="sym", values=signal_col, aggfunc="mean"
            )
            if sig_pivot.shape[1] >= 2:
                report.signal_concentration_hhi = psa.signal_concentration(sig_pivot)
                fig_conc = psa.plot_signal_concentration(
                    sig_pivot, save_path=output_path / "signal_concentration.png"
                )
                plt.close(fig_conc)
                report.portfolio_signal_path = str(output_path / "signal_concentration.png")
        except Exception:
            pass

    # Save summary JSON
    summary = report.summary_dict()
    with open(output_path / "signal_summary.json", "w") as f:
        json.dump(
            {k: (v if not (isinstance(v, float) and np.isnan(v)) else None) for k, v in summary.items()},
            f, indent=2,
        )

    return report


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

def to_html(report: SignalAnalyticsReport, path: str | Path) -> None:
    """Render the SignalAnalyticsReport as a self-contained HTML file.

    Parameters
    ----------
    report : SignalAnalyticsReport from generate_signal_report()
    path   : output path for the HTML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v: Any, decimals: int = 4) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            if np.isnan(v):
                return "—"
            return f"{v:.{decimals}f}"
        return html.escape(str(v))

    def img_section(title: str, img_path: str) -> str:
        if not img_path or not Path(img_path).exists():
            return ""
        rel = Path(img_path).name
        return f"""
        <div class="section">
            <h2>{html.escape(title)}</h2>
            <img src="{rel}" alt="{html.escape(title)}" style="max-width:100%; border:1px solid #ddd;">
        </div>
        """

    sum_dict = report.summary_dict()
    rows = ""
    for k, v in sum_dict.items():
        rows += f"<tr><td>{html.escape(k)}</td><td>{fmt(v)}</td></tr>\n"

    ic_rows = ""
    for k, v in report.ic_summary.items():
        ic_rows += f"<tr><td>{html.escape(k)}</td><td>{fmt(v)}</td></tr>\n"

    factor_ic_rows = ""
    for k, v in report.factor_ics.items():
        factor_ic_rows += f"<tr><td>{html.escape(k)}</td><td>{fmt(v)}</td></tr>\n"

    aq = report.activation_quality
    aq_rows = ""
    if aq is not None:
        aq_data = {
            "total_activations": aq.total_activations,
            "profitable_fraction": aq.profitable_fraction,
            "win_rate": aq.win_rate,
            "sharpe": aq.sharpe,
            "total_pnl": aq.total_pnl,
            "ic_ensemble": aq.ic_ensemble,
            "ic_tf_score": aq.ic_tf_score,
            "ic_mass": aq.ic_mass,
            "ic_delta_score": aq.ic_delta_score,
            "long_ic": aq.long_ic,
            "short_ic": aq.short_ic,
            "long_trades": aq.long_trades,
            "short_trades": aq.short_trades,
        }
        for k, v in aq_data.items():
            aq_rows += f"<tr><td>{html.escape(k)}</td><td>{fmt(v)}</td></tr>\n"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Analytics Report</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f9f9f9; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; }}
        .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #e0e0e0; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f5f5f5; }}
        .meta {{ color: #666; font-size: 0.9em; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        img {{ border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Signal Analytics Report</h1>
    <p class="meta">Generated: {html.escape(report.generated_at)} | Trades: {report.n_trades} | Period: {html.escape(report.date_range)}</p>

    <div class="section">
        <h2>Executive Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {rows}
        </table>
    </div>

    <div class="section">
        <h2>IC Analysis</h2>
        <table>
            <tr><th>IC Metric</th><th>Value</th></tr>
            {ic_rows}
        </table>
    </div>

    {img_section("IC Decay", report.ic_decay_path)}
    {img_section("Rolling IC", report.rolling_ic_path)}
    {img_section("IC Distribution", report.ic_distribution_path)}
    {img_section("Alpha Decay Analysis", report.alpha_decay_path)}

    <div class="section">
        <h2>BH Activation Quality</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {aq_rows}
        </table>
    </div>

    {img_section("BH Signal Quality Dashboard", report.bh_dashboard_path)}

    <div class="section">
        <h2>Factor ICs</h2>
        <table>
            <tr><th>Factor</th><th>IC</th></tr>
            {factor_ic_rows}
        </table>
    </div>

    {img_section("Factor IC Bar Chart", report.factor_ic_bar_path)}
    {img_section("Return Attribution Waterfall", report.attribution_waterfall_path)}
    {img_section("Quintile Returns", report.quintile_bar_path)}
    {img_section("Signal Concentration", report.portfolio_signal_path)}

</body>
</html>
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def to_console(report: SignalAnalyticsReport) -> None:
    """Print a formatted signal analytics summary to the console.

    Uses rich if available, else plain text.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        console = Console()
        console.print(
            Panel(
                f"[bold]Signal Analytics Report[/bold]\n"
                f"Generated: {report.generated_at}\n"
                f"Trades: {report.n_trades}  |  Period: {report.date_range}",
                title="SRFM-Lab",
                border_style="blue",
            )
        )

        # IC Summary
        ic_tbl = Table("Metric", "Value", box=box.SIMPLE, title="IC Analysis")
        for k, v in report.ic_summary.items():
            v_str = f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else str(v)
            ic_tbl.add_row(k, v_str)
        console.print(ic_tbl)

        # Alpha decay
        console.print(
            f"[cyan]Half-life:[/cyan] {report.halflife_bars:.1f} bars  |  "
            f"[cyan]Optimal holding:[/cyan] {report.optimal_holding_bars} bars"
        )

        # Activation quality
        if report.activation_quality is not None:
            aq = report.activation_quality
            aq_tbl = Table("Metric", "Value", box=box.SIMPLE, title="BH Activation Quality")
            aq_tbl.add_row("N Activations", str(aq.total_activations))
            aq_tbl.add_row("Win Rate", f"{aq.win_rate:.1%}")
            aq_tbl.add_row("Total PnL", f"${aq.total_pnl:,.2f}")
            aq_tbl.add_row("Sharpe", f"{aq.sharpe:.2f}")
            aq_tbl.add_row("IC Ensemble", f"{aq.ic_ensemble:.4f}")
            aq_tbl.add_row("IC TF-score", f"{aq.ic_tf_score:.4f}")
            aq_tbl.add_row("IC Mass", f"{aq.ic_mass:.4f}")
            aq_tbl.add_row("Long IC", f"{aq.long_ic:.4f}")
            aq_tbl.add_row("Short IC", f"{aq.short_ic:.4f}")
            console.print(aq_tbl)

        # Quintile
        if report.quintile_result is not None:
            qr = report.quintile_result
            console.print(
                f"[cyan]Quintile Spread:[/cyan] {qr.spread:.4f}  |  "
                f"[cyan]Monotonicity:[/cyan] {qr.monotonicity_score:.2f}  |  "
                f"[cyan]Spread t:[/cyan] {qr.spread_t_stat:.2f} (p={qr.spread_p_value:.4f})"
            )

        console.print(f"\n[green]Outputs saved to:[/green] {report.output_dir}")

    except ImportError:
        # Fallback plain text
        print("=" * 60)
        print("SIGNAL ANALYTICS REPORT")
        print(f"Generated: {report.generated_at}")
        print(f"Trades: {report.n_trades}  |  Period: {report.date_range}")
        print("=" * 60)

        print("\nIC SUMMARY")
        for k, v in report.ic_summary.items():
            v_str = f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else str(v)
            print(f"  {k:<25} {v_str}")

        print(f"\nHalf-life: {report.halflife_bars:.1f} bars")
        print(f"Optimal holding: {report.optimal_holding_bars} bars")

        if report.activation_quality is not None:
            aq = report.activation_quality
            print("\nBH ACTIVATION QUALITY")
            print(f"  N Activations:  {aq.total_activations}")
            print(f"  Win Rate:       {aq.win_rate:.1%}")
            print(f"  Total PnL:      ${aq.total_pnl:,.2f}")
            print(f"  Sharpe:         {aq.sharpe:.2f}")
            print(f"  IC Ensemble:    {aq.ic_ensemble:.4f}")

        print(f"\nOutputs saved to: {report.output_dir}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_ic_summary(
    calc: ICCalculator,
    df: pd.DataFrame,
    signal_col: str,
) -> Dict[str, float]:
    """Compute IC summary stats from trades DataFrame."""
    if signal_col not in df.columns or "_ret" not in df.columns:
        return {}
    sub = df[[signal_col, "_ret"]].dropna()
    if len(sub) < 5:
        return {"mean_ic": float("nan"), "n_obs": float(len(sub))}

    sig = sub[signal_col]
    ret = sub["_ret"]

    from scipy import stats as _stats
    r_sp, p_sp = _stats.spearmanr(sig, ret)
    r_pe, _ = _stats.pearsonr(sig, ret)

    ic_series = pd.Series([float(r_sp)], name="IC")
    icir_val = calc.icir(ic_series)

    return {
        "mean_ic": float(r_sp),
        "pearson_ic": float(r_pe),
        "icir": icir_val,
        "pct_positive": float(float(r_sp) > 0),
        "t_stat": float(float(r_sp) * np.sqrt(max(len(sub) - 2, 1)) / max(np.sqrt(max(1 - r_sp**2, 1e-10)), 1e-10)),
        "p_value": float(p_sp),
        "n_obs": float(len(sub)),
    }


def _compute_rolling_ic_from_trades(
    df: pd.DataFrame,
    signal_col: str,
    window: int = 60,
) -> pd.Series:
    """Compute a rolling IC series from a flat trades DataFrame sorted by exit_time."""
    if "exit_time" not in df.columns or signal_col not in df.columns or "_ret" not in df.columns:
        return pd.Series(dtype=float)

    df_sorted = df.sort_values("exit_time")[[signal_col, "_ret", "exit_time"]].dropna()
    if len(df_sorted) < window + 1:
        return pd.Series(dtype=float)

    from scipy import stats as _stats

    ics: list[float] = []
    dates: list = []

    for i in range(window - 1, len(df_sorted)):
        window_df = df_sorted.iloc[i - window + 1 : i + 1]
        sv, rv = window_df[signal_col].values, window_df["_ret"].values
        if len(sv) < 3:
            ics.append(float("nan"))
        else:
            r, _ = _stats.spearmanr(sv, rv)
            ics.append(float(r))
        dates.append(df_sorted["exit_time"].iloc[i])

    return pd.Series(ics, index=dates, name="rolling_ic")


# ---------------------------------------------------------------------------
# Extended report utilities
# ---------------------------------------------------------------------------

def generate_ic_tearsheet(
    trades: pd.DataFrame,
    signal_col: str = "ensemble_signal",
    return_col: str = "pnl",
    dollar_pos_col: str = "dollar_pos",
    output_dir: str | Path = "results/ic_tearsheet/",
    max_horizon: int = 20,
    rolling_window: int = 60,
    transaction_cost: float = 0.0002,
) -> Dict[str, Any]:
    """Generate a standalone IC tearsheet without the full report.

    Produces IC decay, rolling IC, IC distribution, and IC heatmap.

    Parameters
    ----------
    trades           : trade records
    signal_col       : signal column
    return_col       : P&L column
    dollar_pos_col   : position column
    output_dir       : output directory
    max_horizon      : max IC decay horizon
    rolling_window   : rolling IC window
    transaction_cost : one-way cost

    Returns
    -------
    Dict with paths to generated plots and IC statistics
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = trades.copy()
    if dollar_pos_col in df.columns:
        pos = df[dollar_pos_col].abs().replace(0, np.nan)
        df["_ret"] = df[return_col] / pos
    else:
        df["_ret"] = df[return_col]

    calc = ICCalculator()
    ada = AlphaDecayAnalyzer(default_cost=transaction_cost)
    result: dict[str, Any] = {}

    # IC summary
    result["ic_summary"] = _compute_ic_summary(calc, df, signal_col)

    # IC decay
    if signal_col in df.columns and "hold_bars" in df.columns:
        decay = calc.ic_decay_from_trades(
            df, signal_col=signal_col, return_col=return_col,
            hold_col="hold_bars", dollar_pos_col=dollar_pos_col,
            max_horizon=max_horizon,
        )
        result["decay_result"] = decay
        result["decay_model"] = ada.signal_decay_model(decay)
        result["half_life_bars"] = ada.compute_signal_halflife(decay)
        result["optimal_holding_bars"] = ada.optimal_holding_period(decay, transaction_cost)

        fig_decay = calc.plot_ic_decay(decay, save_path=out_path / "ic_decay.png")
        plt.close(fig_decay)
        result["ic_decay_path"] = str(out_path / "ic_decay.png")

        fig_alpha = ada.plot_alpha_decay(decay, save_path=out_path / "alpha_decay.png",
                                          transaction_cost=transaction_cost)
        plt.close(fig_alpha)
        result["alpha_decay_path"] = str(out_path / "alpha_decay.png")

    # Rolling IC
    if "exit_time" in df.columns and signal_col in df.columns:
        ric = _compute_rolling_ic_from_trades(df, signal_col, window=rolling_window)
        if len(ric) > 5:
            fig_ric = calc.plot_rolling_ic(ric, save_path=out_path / "rolling_ic.png")
            plt.close(fig_ric)
            result["rolling_ic_path"] = str(out_path / "rolling_ic.png")

            fig_dist = calc.plot_ic_distribution(ric, save_path=out_path / "ic_distribution.png")
            plt.close(fig_dist)
            result["ic_distribution_path"] = str(out_path / "ic_distribution.png")

    # IC heatmap
    if "exit_time" in df.columns and signal_col in df.columns:
        try:
            fig_heat = calc.plot_ic_heatmap(df, signal_col, return_col, dollar_pos_col,
                                             save_path=out_path / "ic_heatmap.png")
            plt.close(fig_heat)
            result["ic_heatmap_path"] = str(out_path / "ic_heatmap.png")
        except Exception:
            pass

    return result


def generate_factor_tearsheet(
    trades: pd.DataFrame,
    price_history: Optional[pd.DataFrame] = None,
    return_col: str = "pnl",
    dollar_pos_col: str = "dollar_pos",
    output_dir: str | Path = "results/factor_tearsheet/",
) -> Dict[str, Any]:
    """Generate a standalone factor attribution tearsheet.

    Parameters
    ----------
    trades        : trade records
    price_history : optional price panel for momentum/vol factors
    return_col    : P&L column
    dollar_pos_col: position column
    output_dir    : output directory

    Returns
    -------
    Dict with factor ICs, attribution result, and plot paths
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fm = FactorModel()
    df = trades.copy()
    if dollar_pos_col in df.columns:
        pos = df[dollar_pos_col].abs().replace(0, np.nan)
        df["_ret"] = df[return_col] / pos
    else:
        df["_ret"] = df[return_col]

    result: dict[str, Any] = {}

    try:
        factor_df = fm.build_factor_matrix(df, price_history)
        ret = df["_ret"].loc[factor_df.index].dropna()
        factor_aligned = factor_df.loc[ret.index]

        result["factor_ics"] = fm.factor_ic(factor_aligned, ret)
        result["vif"] = fm.variance_inflation_factors(factor_aligned).to_dict()
        result["factor_corr"] = fm.factor_correlation_matrix(factor_aligned).to_dict()

        fig_fic = fm.plot_factor_ic_bar(result["factor_ics"],
                                         save_path=out_path / "factor_ic_bar.png")
        plt.close(fig_fic)
        result["factor_ic_bar_path"] = str(out_path / "factor_ic_bar.png")

        attr = fm.factor_attribution(df, price_history=price_history,
                                      return_col=return_col, dollar_pos_col=dollar_pos_col)
        result["attribution"] = attr

        fig_attr = fm.plot_attribution_waterfall(attr,
                                                  save_path=out_path / "attribution_waterfall.png")
        plt.close(fig_attr)
        result["attribution_path"] = str(out_path / "attribution_waterfall.png")

        # PCA
        try:
            if price_history is not None:
                pca_res = fm.pca_factors(price_history.pct_change().dropna())
                result["pca_result"] = pca_res
                fig_pca = fm.plot_pca_scree(pca_res, save_path=out_path / "pca_scree.png")
                plt.close(fig_pca)
                result["pca_scree_path"] = str(out_path / "pca_scree.png")
        except Exception:
            pass

    except Exception as e:
        result["error"] = str(e)

    return result


def compare_signals_report(
    trades: pd.DataFrame,
    signal_cols: List[str],
    return_col: str = "pnl",
    dollar_pos_col: str = "dollar_pos",
    output_dir: str | Path = "results/signal_comparison/",
    transaction_cost: float = 0.0002,
) -> pd.DataFrame:
    """Compare multiple signals side-by-side.

    For each signal, computes IC, ICIR, half-life, win rate, and Sharpe.

    Parameters
    ----------
    trades           : trade records
    signal_cols      : list of signal columns to compare
    return_col       : P&L column
    dollar_pos_col   : position column
    output_dir       : output directory for comparison chart
    transaction_cost : one-way cost

    Returns
    -------
    pd.DataFrame[signal x (ic, icir, half_life, win_rate, sharpe, n_obs)]
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = trades.copy()
    if dollar_pos_col in df.columns:
        pos = df[dollar_pos_col].abs().replace(0, np.nan)
        df["_ret"] = df[return_col] / pos
    else:
        df["_ret"] = df[return_col]

    calc = ICCalculator()
    ada = AlphaDecayAnalyzer(default_cost=transaction_cost)

    records: list[dict] = []
    for col in signal_cols:
        if col not in df.columns:
            continue
        sub = df[[col, "_ret"]].dropna()
        n = len(sub)

        if n < 5:
            records.append({"signal": col, "n_obs": n})
            continue

        from scipy import stats as _stats
        r, p = _stats.spearmanr(sub[col], sub["_ret"])
        ic = float(r)
        t = ic * np.sqrt(n - 2) / max(np.sqrt(max(1 - ic**2, 1e-10)), 1e-10)

        # Approximate ICIR from bootstrap
        ic_series = pd.Series([ic])  # degenerate
        icir_val = float("nan")

        # Half-life from trades
        half_life = float("nan")
        if "hold_bars" in df.columns:
            try:
                decay = calc.ic_decay_from_trades(
                    df, signal_col=col, return_col=return_col,
                    hold_col="hold_bars", dollar_pos_col=dollar_pos_col,
                    max_horizon=20,
                )
                half_life = ada.compute_signal_halflife(decay)
            except Exception:
                pass

        win_rate = float((df[return_col] > 0).mean())
        r_arr = sub["_ret"].values
        sharpe_val = float("nan")
        if len(r_arr) > 1 and r_arr.std(ddof=1) > 0:
            sharpe_val = float(r_arr.mean() / r_arr.std(ddof=1) * np.sqrt(252))

        records.append({
            "signal": col,
            "ic": ic,
            "t_stat": float(t),
            "p_value": float(p),
            "half_life_bars": half_life,
            "win_rate": win_rate,
            "sharpe": sharpe_val,
            "n_obs": n,
        })

    result_df = pd.DataFrame(records).set_index("signal")

    # Save comparison table
    result_df.to_csv(out_path / "signal_comparison.csv")

    # Plot IC comparison bar chart
    if len(result_df) > 0 and "ic" in result_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ic_vals = result_df["ic"].values
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ic_vals]
        ax.bar(result_df.index, ic_vals, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Signal")
        ax.set_ylabel("IC (Spearman)")
        ax.set_title("Signal IC Comparison")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_path / "signal_ic_comparison.png", dpi=150)
        plt.close(fig)

    return result_df


def to_pdf(report: SignalAnalyticsReport, path: str | Path) -> None:
    """Export report to PDF (requires weasyprint).

    Falls back to HTML if weasyprint is not installed.

    Parameters
    ----------
    report : SignalAnalyticsReport
    path   : output PDF path
    """
    path = Path(path)
    html_path = path.with_suffix(".html")
    to_html(report, html_path)

    try:
        from weasyprint import HTML  # type: ignore[import]
        HTML(filename=str(html_path)).write_pdf(str(path))
    except ImportError:
        import warnings
        warnings.warn(
            "weasyprint not installed; PDF export unavailable. "
            "HTML report saved to: " + str(html_path)
        )


def batch_signal_report(
    trades_list: List[pd.DataFrame],
    names: List[str],
    output_dir: str | Path = "results/batch_signals/",
    signal_col: str = "ensemble_signal",
    return_col: str = "pnl",
    dollar_pos_col: str = "dollar_pos",
) -> pd.DataFrame:
    """Run signal quality analysis on a list of trade DataFrames and compare.

    Useful for comparing signals across different instruments, time periods,
    or parameter sets.

    Parameters
    ----------
    trades_list   : list of trade DataFrames
    names         : corresponding names for each DataFrame
    output_dir    : base output directory
    signal_col    : signal column
    return_col    : P&L column
    dollar_pos_col: position column

    Returns
    -------
    pd.DataFrame[name x (ic, icir, win_rate, sharpe, total_pnl)]
    """
    out_path = Path(output_dir)
    records: list[dict] = []

    for name, trades in zip(names, trades_list):
        sub_dir = out_path / name
        try:
            rpt = generate_signal_report(
                trades,
                output_dir=sub_dir,
                signal_col=signal_col,
                return_col=return_col,
                dollar_pos_col=dollar_pos_col,
            )
            aq = rpt.activation_quality
            records.append({
                "name": name,
                "n_trades": rpt.n_trades,
                "ic": rpt.ic_summary.get("mean_ic", float("nan")),
                "icir": rpt.ic_summary.get("icir", float("nan")),
                "half_life": rpt.halflife_bars,
                "optimal_hold": rpt.optimal_holding_bars,
                "win_rate": aq.win_rate if aq else float("nan"),
                "sharpe": aq.sharpe if aq else float("nan"),
                "total_pnl": aq.total_pnl if aq else float("nan"),
            })
        except Exception as e:
            records.append({"name": name, "error": str(e)})

    return pd.DataFrame(records).set_index("name")
