"""
research/walk_forward/report.py
─────────────────────────────────
HTML and terminal reporting for walk-forward analysis results.

Generates:
  • HTML report with IS vs OOS table, fold breakdown, param stability chart
  • Console-formatted summary table
  • IS vs OOS Sharpe scatter plot
  • Equity curves by fold
  • Parameter selection frequency chart
"""

from __future__ import annotations

import logging
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .engine import WFResult, CPCVResult, FoldResult, is_oos_degradation_summary
from .metrics import PerformanceStats, compute_performance_stats
from .stability import StabilityAnalyzer, StabilityReport, chow_test

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# WFReport dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WFReport:
    """
    Aggregated walk-forward analysis report.

    Attributes
    ----------
    wf_result          : source WFResult.
    stability_report   : StabilityReport for parameter stability.
    fold_summary_df    : DataFrame with per-fold IS/OOS metrics.
    html_path          : path to the generated HTML report (if any).
    plot_paths         : dict of plot_name → file path.
    summary_stats      : dict of top-level summary statistics.
    chow_test_result   : (F, p_value) structural break test between first/last half.
    """
    wf_result:         WFResult
    stability_report:  StabilityReport
    fold_summary_df:   pd.DataFrame
    html_path:         Optional[str]    = None
    plot_paths:        Dict[str, str]   = field(default_factory=dict)
    summary_stats:     Dict[str, Any]   = field(default_factory=dict)
    chow_test_result:  Tuple[float, float] = (np.nan, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# generate_wf_report
# ─────────────────────────────────────────────────────────────────────────────

def generate_wf_report(
    wf_result:  WFResult,
    output_dir: str,
    title:      str = "Walk-Forward Analysis Report",
    save_plots: bool = True,
    open_html:  bool = False,
) -> WFReport:
    """
    Generate a comprehensive walk-forward analysis report.

    Creates an output directory with:
    - report.html : full interactive HTML report
    - plots/      : all charts as PNG files
    - summary.csv : fold-level summary table

    Parameters
    ----------
    wf_result  : WFResult from WalkForwardEngine.run().
    output_dir : directory to write output files.
    title      : report title.
    save_plots : if True, generate and save all plot PNGs.
    open_html  : if True, open the HTML file in the default browser.

    Returns
    -------
    WFReport dataclass.

    Examples
    --------
    >>> report = generate_wf_report(wf_result, "results/wf_report/")
    >>> print(report.summary_stats["oos_sharpe"])
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots_path  = output_path / "plots"
    plots_path.mkdir(exist_ok=True)

    logger.info("Generating WF report in: %s", output_path)

    # Stability analysis
    analyzer      = StabilityAnalyzer()
    stability_rpt = analyzer.parameter_stability(wf_result)

    # Fold summary DataFrame
    fold_df = is_oos_degradation_summary(wf_result)

    # Summary statistics
    successful = [fr for fr in wf_result.fold_results if fr.success]
    oos_sharpes = [fr.oos_sharpe for fr in successful]
    is_sharpes  = [fr.is_stats.sharpe for fr in successful]

    # Chow test: first half vs second half of folds
    if len(oos_sharpes) >= 4:
        mid        = len(oos_sharpes) // 2
        chow_stat, chow_p = chow_test(oos_sharpes[:mid], oos_sharpes[mid:])
    else:
        chow_stat, chow_p = np.nan, np.nan

    summary_stats = {
        "n_folds":                 wf_result.n_folds,
        "oos_sharpe":              wf_result.oos_sharpe,
        "oos_sharpe_ci_lower":     wf_result.sharpe_ci[0],
        "oos_sharpe_ci_upper":     wf_result.sharpe_ci[1],
        "oos_cagr":                wf_result.oos_cagr,
        "oos_max_dd":              wf_result.oos_max_dd,
        "param_stability_score":   wf_result.param_stability_score,
        "is_oos_sharpe_ratio":     wf_result.is_oos_sharpe_ratio,
        "sharpe_degradation":      wf_result.sharpe_degradation,
        "mean_is_sharpe":          float(np.mean(is_sharpes)) if is_sharpes else 0.0,
        "chow_f_stat":             chow_stat,
        "chow_p_value":            chow_p,
        "total_oos_trades":        len(wf_result.combined_oos_trades) if not wf_result.combined_oos_trades.empty else 0,
        "best_params":             str(wf_result.best_params),
        "total_elapsed_sec":       wf_result.total_elapsed_sec,
        "combined_oos_sharpe":     wf_result.combined_oos_stats.sharpe,
        "combined_oos_win_rate":   wf_result.combined_oos_stats.win_rate_,
        "combined_oos_pf":         wf_result.combined_oos_stats.profit_factor_,
    }

    # Generate plots
    plot_paths: Dict[str, str] = {}
    if save_plots:
        try:
            p = str(plots_path / "is_vs_oos.png")
            plot_is_vs_oos(wf_result, save_path=p, show=False)
            plot_paths["is_vs_oos"] = p
        except Exception as e:
            logger.warning("is_vs_oos plot failed: %s", e)

        try:
            p = str(plots_path / "equity_curves.png")
            plot_equity_curves_by_fold(wf_result, save_path=p, show=False)
            plot_paths["equity_curves"] = p
        except Exception as e:
            logger.warning("equity_curves plot failed: %s", e)

        try:
            p = str(plots_path / "param_frequency.png")
            plot_param_selection_frequency(wf_result, save_path=p, show=False)
            plot_paths["param_frequency"] = p
        except Exception as e:
            logger.warning("param_frequency plot failed: %s", e)

        try:
            p = str(plots_path / "stability_dashboard.png")
            analyzer.plot_stability_dashboard(wf_result, save_path=p, show=False)
            plot_paths["stability_dashboard"] = p
        except Exception as e:
            logger.warning("stability dashboard plot failed: %s", e)

    # Save fold summary CSV
    if not fold_df.empty:
        csv_path = output_path / "fold_summary.csv"
        fold_df.to_csv(csv_path, index=False)
        logger.info("Saved fold summary: %s", csv_path)

    # Generate HTML report
    html_path = str(output_path / "report.html")
    try:
        _write_html_report(
            html_path     = html_path,
            title         = title,
            wf_result     = wf_result,
            summary_stats = summary_stats,
            fold_df       = fold_df,
            stability_rpt = stability_rpt,
            plot_paths    = plot_paths,
        )
        logger.info("Saved HTML report: %s", html_path)
    except Exception as e:
        logger.error("HTML report generation failed: %s", e)
        html_path = None

    if open_html and html_path and Path(html_path).exists():
        import webbrowser
        webbrowser.open(f"file://{Path(html_path).resolve()}")

    report = WFReport(
        wf_result        = wf_result,
        stability_report = stability_rpt,
        fold_summary_df  = fold_df,
        html_path        = html_path,
        plot_paths       = plot_paths,
        summary_stats    = summary_stats,
        chow_test_result = (chow_stat, chow_p),
    )

    logger.info(
        "WF report complete: OOS Sharpe=%.3f, Stability=%.1f%%, %d folds",
        wf_result.oos_sharpe, wf_result.param_stability_score * 100, wf_result.n_folds,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_is_vs_oos(
    wf_result: WFResult,
    save_path: Optional[str] = None,
    show:      bool = True,
) -> None:
    """
    Scatter plot of IS Sharpe vs OOS Sharpe per fold.

    Points above the diagonal indicate OOS outperformance relative to IS.
    Points below (typical) indicate overfitting.

    Parameters
    ----------
    wf_result : WFResult.
    save_path : optional save path.
    show      : if True, display figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return

    successful = [fr for fr in wf_result.fold_results if fr.success]
    if not successful:
        return

    is_s  = np.array([fr.is_stats.sharpe for fr in successful])
    oos_s = np.array([fr.oos_sharpe       for fr in successful])
    folds = [fr.fold_id for fr in successful]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Colour by OOS Sharpe quality
    colors = ["#2ecc71" if s > 0.5 else "#f1c40f" if s > 0.0 else "#e74c3c" for s in oos_s]

    scatter = ax.scatter(is_s, oos_s, c=colors, s=120, edgecolors="black", linewidths=0.8, zorder=3)

    # Annotate folds
    for fid, xi, yi in zip(folds, is_s, oos_s):
        ax.annotate(f"F{fid}", (xi, yi), textcoords="offset points",
                    xytext=(8, 4), fontsize=8)

    # 45-degree line
    all_vals = np.concatenate([is_s, oos_s])
    vmin, vmax = all_vals.min() - 0.2, all_vals.max() + 0.2
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, alpha=0.5, label="IS = OOS")
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)

    # Mean lines
    ax.axhline(np.mean(oos_s), color="#e74c3c", linewidth=1.5, linestyle="-.",
               alpha=0.8, label=f"Mean OOS Sharpe = {np.mean(oos_s):.3f}")
    ax.axvline(np.mean(is_s),  color="#3498db", linewidth=1.5, linestyle="-.",
               alpha=0.8, label=f"Mean IS Sharpe  = {np.mean(is_s):.3f}")

    ax.set_xlabel("IS Sharpe", fontsize=12)
    ax.set_ylabel("OOS Sharpe", fontsize=12)
    ax.set_title(f"IS vs OOS Sharpe  |  Degradation = {float(np.mean(is_s) - np.mean(oos_s)):.3f}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved IS vs OOS plot: %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_equity_curves_by_fold(
    wf_result:       WFResult,
    save_path:       Optional[str] = None,
    show:            bool  = True,
    starting_equity: float = 100_000.0,
) -> None:
    """
    Plot OOS equity curve for each fold, plus the combined equity curve.

    Parameters
    ----------
    wf_result       : WFResult.
    save_path       : optional save path.
    show            : display figure.
    starting_equity : initial equity.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib not available")
        return

    successful = [fr for fr in wf_result.fold_results if fr.success]
    if not successful:
        return

    n_folds = len(successful)
    colors  = cm.tab10(np.linspace(0, 1, n_folds))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})

    ax_curves  = axes[0]
    ax_per_fold = axes[1]

    # ── Combined OOS equity (top panel) ───────────────────────────────────
    eq = starting_equity
    all_eq = [eq]
    fold_boundaries = [0]

    for fr in sorted(successful, key=lambda f: f.fold_id):
        for trade in fr.oos_trades:
            pnl = float(trade.get("pnl", 0.0)) if isinstance(trade, dict) else 0.0
            eq += pnl
            all_eq.append(eq)
        fold_boundaries.append(len(all_eq) - 1)

    eq_series = pd.Series(all_eq)
    final_ret = (eq_series.iloc[-1] / starting_equity - 1.0) * 100.0
    color     = "#2ecc71" if eq_series.iloc[-1] >= starting_equity else "#e74c3c"

    ax_curves.plot(eq_series.index, eq_series.values, color=color, linewidth=2.0,
                   label=f"Combined OOS ({final_ret:+.1f}%)")

    # Shade fold regions
    for fi, (b_start, b_end) in enumerate(zip(fold_boundaries[:-1], fold_boundaries[1:])):
        alpha = 0.05 if fi % 2 == 0 else 0.10
        ax_curves.axvspan(b_start, b_end, alpha=alpha, color=colors[fi % len(colors)])

    # Mark fold boundaries
    for b in fold_boundaries[1:-1]:
        ax_curves.axvline(b, color="gray", linewidth=0.5, alpha=0.5)

    ax_curves.axhline(starting_equity, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_curves.set_ylabel("Portfolio Equity ($)")
    ax_curves.set_title("Combined OOS Equity Curve (all folds stitched)")
    ax_curves.legend(fontsize=9)
    ax_curves.grid(True, alpha=0.3)
    ax_curves.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # ── Per-fold OOS Sharpe bars (bottom panel) ─────────────────────────
    fold_ids    = [fr.fold_id        for fr in sorted(successful, key=lambda f: f.fold_id)]
    fold_sharpes = [fr.oos_sharpe    for fr in sorted(successful, key=lambda f: f.fold_id)]
    fold_colors = ["#2ecc71" if s > 0.5 else "#f1c40f" if s > 0 else "#e74c3c" for s in fold_sharpes]

    bars = ax_per_fold.bar(range(n_folds), fold_sharpes, color=fold_colors,
                           alpha=0.85, edgecolor="black", linewidth=0.5)
    ax_per_fold.axhline(0, color="black", linewidth=0.8)
    ax_per_fold.axhline(wf_result.oos_sharpe, color="navy", linewidth=1.5,
                        linestyle="--", alpha=0.7,
                        label=f"Mean OOS Sharpe = {wf_result.oos_sharpe:.3f}")
    ax_per_fold.set_xticks(range(n_folds))
    ax_per_fold.set_xticklabels([f"Fold {fid}" for fid in fold_ids])
    ax_per_fold.set_ylabel("OOS Sharpe")
    ax_per_fold.set_title("OOS Sharpe by Fold")
    ax_per_fold.legend(fontsize=9)
    ax_per_fold.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, fold_sharpes):
        ypos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.07
        ax_per_fold.text(bar.get_x() + bar.get_width() / 2, ypos,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved equity curves plot: %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_param_selection_frequency(
    wf_result: WFResult,
    save_path: Optional[str] = None,
    show:      bool = True,
) -> None:
    """
    Plot the frequency of parameter values selected across folds.

    For each parameter in the param grid, shows a bar chart of how often
    each value was selected as the best IS parameter.

    Parameters
    ----------
    wf_result : WFResult.
    save_path : optional save path.
    show      : display figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return

    successful = [fr for fr in wf_result.fold_results if fr.success]
    if not successful:
        return

    # Get all unique params
    all_params = sorted(set().union(*[set(fr.params.keys()) for fr in successful]))
    if not all_params:
        return

    n_params = len(all_params)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 5), sharey=False)

    if n_params == 1:
        axes = [axes]

    for ax, param in zip(axes, all_params):
        values = [fr.params.get(param) for fr in successful if param in fr.params]
        from collections import Counter
        counts = Counter(str(v) for v in values)
        labels = sorted(counts.keys(), key=lambda k: float(k) if _is_numeric_str(k) else k)
        freqs  = [counts[l] for l in labels]
        total  = sum(freqs)

        colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(labels)))
        bars   = ax.bar(range(len(labels)), freqs, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Frequency (folds)")
        ax.set_title(f"{param}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, freq in zip(bars, freqs):
            pct = 100.0 * freq / max(1, total)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Parameter Selection Frequency Across Folds", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved param frequency plot: %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Console terminal report
# ─────────────────────────────────────────────────────────────────────────────

def to_console(report: WFReport) -> None:
    """
    Print a formatted summary of the WFReport to the console.

    Attempts to use `rich` for formatted tables; falls back to plain text
    if rich is not installed.

    Parameters
    ----------
    report : WFReport from generate_wf_report().
    """
    try:
        from rich.console import Console
        from rich.table   import Table
        from rich.panel   import Panel
        from rich         import box

        _rich_console(report)
    except ImportError:
        _plain_console(report)


def _rich_console(report: WFReport) -> None:
    """Rich-formatted console output."""
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich         import box

    console = Console()
    ss      = report.summary_stats

    # Summary panel
    summary_lines = [
        f"[bold]OOS Sharpe:[/bold]      {ss.get('oos_sharpe', 0):.4f}  "
        f"(95% CI: [{ss.get('oos_sharpe_ci_lower', 0):.3f}, {ss.get('oos_sharpe_ci_upper', 0):.3f}])",
        f"[bold]OOS CAGR:[/bold]        {ss.get('oos_cagr', 0):.1%}",
        f"[bold]OOS Max DD:[/bold]      {ss.get('oos_max_dd', 0):.1%}",
        f"[bold]IS/OOS Ratio:[/bold]    {ss.get('is_oos_sharpe_ratio', 0):.3f}  "
        f"(degradation: {ss.get('sharpe_degradation', 0):.3f})",
        f"[bold]Param Stability:[/bold] {ss.get('param_stability_score', 0):.1%}",
        f"[bold]Folds:[/bold]           {ss.get('n_folds', 0)}",
        f"[bold]Chow Test:[/bold]       F={ss.get('chow_f_stat', float('nan')):.3f}  "
        f"p={ss.get('chow_p_value', float('nan')):.4f}",
        f"[bold]Best Params:[/bold]     {ss.get('best_params', '')}",
    ]
    console.print(Panel(
        "\n".join(summary_lines),
        title="[bold cyan]Walk-Forward Analysis Summary[/bold cyan]",
        border_style="cyan",
    ))

    # Fold summary table
    if not report.fold_summary_df.empty:
        table = Table(
            title="Fold Breakdown",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        df = report.fold_summary_df
        for col in df.columns:
            table.add_column(col, justify="right")

        for _, row in df.iterrows():
            formatted = []
            for val in row:
                if isinstance(val, float):
                    formatted.append(f"{val:.4f}")
                else:
                    formatted.append(str(val))
            table.add_row(*formatted)

        console.print(table)

    # Param stability
    if report.stability_report.param_drift:
        drift_table = Table(
            title="Parameter Drift",
            box=box.SIMPLE,
            header_style="bold yellow",
        )
        drift_table.add_column("Parameter",   justify="left")
        drift_table.add_column("Drift (%)",   justify="right")
        drift_table.add_column("Modal Value", justify="right")

        for param, drift in sorted(report.stability_report.param_drift.items()):
            modal = report.stability_report.most_common_params.get(param, "N/A")
            color = "green" if drift < 0.2 else "yellow" if drift < 0.5 else "red"
            drift_table.add_row(
                param,
                f"[{color}]{drift:.1%}[/{color}]",
                str(modal),
            )
        console.print(drift_table)

    console.print(f"\n[dim]Report saved to: {report.html_path or 'N/A'}[/dim]")


def _plain_console(report: WFReport) -> None:
    """Plain-text fallback console output."""
    ss = report.summary_stats
    sep = "─" * 60

    print(f"\n{sep}")
    print("  WALK-FORWARD ANALYSIS SUMMARY")
    print(sep)
    print(f"  OOS Sharpe:        {ss.get('oos_sharpe', 0):.4f}"
          f"  (95% CI: [{ss.get('oos_sharpe_ci_lower', 0):.3f}, "
          f"{ss.get('oos_sharpe_ci_upper', 0):.3f}])")
    print(f"  OOS CAGR:          {ss.get('oos_cagr', 0):.1%}")
    print(f"  OOS Max DD:        {ss.get('oos_max_dd', 0):.1%}")
    print(f"  IS/OOS Ratio:      {ss.get('is_oos_sharpe_ratio', 0):.3f}")
    print(f"  Sharpe Degradation:{ss.get('sharpe_degradation', 0):.3f}")
    print(f"  Param Stability:   {ss.get('param_stability_score', 0):.1%}")
    print(f"  Folds:             {ss.get('n_folds', 0)}")
    chow_f = ss.get('chow_f_stat', float('nan'))
    chow_p = ss.get('chow_p_value', float('nan'))
    print(f"  Chow Test:         F={chow_f:.3f}  p={chow_p:.4f}")
    print(f"  Best Params:       {ss.get('best_params', '')}")
    print(sep)

    # Fold table
    df = report.fold_summary_df
    if not df.empty:
        print("\n  FOLD BREAKDOWN")
        print(df.to_string(index=False, float_format="%.4f"))
        print()

    # Param drift
    drift = report.stability_report.param_drift
    if drift:
        print("\n  PARAMETER DRIFT")
        print(f"  {'Parameter':<20} {'Drift':>8}  {'Modal Value'}")
        for param, d in sorted(drift.items()):
            modal = report.stability_report.most_common_params.get(param, "N/A")
            print(f"  {param:<20} {d:>8.1%}  {modal}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# HTML report generator
# ─────────────────────────────────────────────────────────────────────────────

def _write_html_report(
    html_path:     str,
    title:         str,
    wf_result:     WFResult,
    summary_stats: Dict[str, Any],
    fold_df:       pd.DataFrame,
    stability_rpt: StabilityReport,
    plot_paths:    Dict[str, str],
) -> None:
    """Write the complete HTML report to disk."""

    def _fmt(val: Any, decimals: int = 4) -> str:
        if isinstance(val, float) and not np.isfinite(val):
            return "N/A"
        if isinstance(val, float):
            return f"{val:.{decimals}f}"
        return str(val)

    def _pct(val: Any) -> str:
        if isinstance(val, float) and np.isfinite(val):
            return f"{val:.1%}"
        return "N/A"

    # Build plot img tags (relative paths)
    def _img_tag(plot_key: str) -> str:
        path = plot_paths.get(plot_key)
        if path and os.path.exists(path):
            rel = os.path.relpath(path, os.path.dirname(html_path))
            return f'<img src="{rel}" style="max-width:100%;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,0.1)">'
        return "<p><em>Plot not available</em></p>"

    # Fold table HTML
    if not fold_df.empty:
        fold_html = fold_df.to_html(
            classes="data-table", border=0, index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    else:
        fold_html = "<p>No fold data available.</p>"

    # Param drift table
    drift_rows = ""
    for param, drift in sorted(stability_rpt.param_drift.items()):
        modal = stability_rpt.most_common_params.get(param, "N/A")
        color = "#2ecc71" if drift < 0.2 else "#f39c12" if drift < 0.5 else "#e74c3c"
        drift_rows += (
            f'<tr>'
            f'<td>{param}</td>'
            f'<td style="color:{color};font-weight:bold">{drift:.1%}</td>'
            f'<td>{modal}</td>'
            f'</tr>'
        )

    ss = summary_stats
    chow_sig = "Yes (structural break detected)" if (
        isinstance(ss.get("chow_p_value"), float) and
        np.isfinite(ss.get("chow_p_value", float("nan"))) and
        ss["chow_p_value"] < 0.05
    ) else "No"

    html = textwrap.dedent(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f4f6f8; color: #2c3e50; margin: 0; padding: 0; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  h1 {{ color: #1a252f; border-bottom: 3px solid #3498db; padding-bottom: 12px; }}
  h2 {{ color: #2980b9; margin-top: 32px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
  .metric-card {{ background: white; border-radius: 8px; padding: 16px;
                  box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
  .metric-card .value {{ font-size: 2em; font-weight: 700; color: #2980b9; }}
  .metric-card .label {{ font-size: 0.85em; color: #7f8c8d; margin-top: 4px; }}
  .metric-card.good  .value {{ color: #2ecc71; }}
  .metric-card.warn  .value {{ color: #f39c12; }}
  .metric-card.bad   .value {{ color: #e74c3c; }}
  .data-table {{ width: 100%; border-collapse: collapse; background: white;
                 border-radius: 8px; overflow: hidden;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .data-table th {{ background: #2980b9; color: white; padding: 10px 12px; text-align: right; }}
  .data-table td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid #ecf0f1; }}
  .data-table tr:last-child td {{ border-bottom: none; }}
  .data-table tr:nth-child(even) td {{ background: #f8f9fa; }}
  .plot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0; }}
  .plot-card {{ background: white; border-radius: 8px; padding: 16px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .plot-card h3 {{ margin-top: 0; color: #2c3e50; }}
  .drift-table {{ width: 100%; border-collapse: collapse; }}
  .drift-table th {{ background: #ecf0f1; padding: 8px 12px; text-align: left; }}
  .drift-table td {{ padding: 6px 12px; border-bottom: 1px solid #ecf0f1; }}
  .info-box {{ background: #eaf2ff; border-left: 4px solid #3498db; padding: 12px 16px;
               border-radius: 4px; margin: 16px 0; }}
  footer {{ text-align: center; color: #95a5a6; font-size: 0.8em; margin-top: 48px; padding: 16px; }}
</style>
</head>
<body>
<div class="container">

<h1>📊 {title}</h1>

<div class="info-box">
  Generated by SRFM-Lab Walk-Forward Analysis Platform &nbsp;|&nbsp;
  Folds: <strong>{ss.get('n_folds', 0)}</strong> &nbsp;|&nbsp;
  Total OOS Trades: <strong>{ss.get('total_oos_trades', 0)}</strong> &nbsp;|&nbsp;
  Elapsed: <strong>{ss.get('total_elapsed_sec', 0):.1f}s</strong>
</div>

<!-- ── Summary Metrics ── -->
<h2>Summary Metrics</h2>
<div class="summary-grid">
  <div class="metric-card {'good' if (ss.get('oos_sharpe', 0) or 0) > 0.5 else 'warn' if (ss.get('oos_sharpe', 0) or 0) > 0 else 'bad'}">
    <div class="value">{_fmt(ss.get('oos_sharpe', 0), 3)}</div>
    <div class="label">OOS Sharpe</div>
  </div>
  <div class="metric-card">
    <div class="value">{_pct(ss.get('oos_cagr', 0))}</div>
    <div class="label">OOS CAGR</div>
  </div>
  <div class="metric-card {'bad' if abs(ss.get('oos_max_dd', 0) or 0) > 0.20 else 'warn' if abs(ss.get('oos_max_dd', 0) or 0) > 0.10 else 'good'}">
    <div class="value">{_pct(ss.get('oos_max_dd', 0))}</div>
    <div class="label">OOS Max Drawdown</div>
  </div>
  <div class="metric-card {'good' if (ss.get('param_stability_score', 0) or 0) > 0.7 else 'warn' if (ss.get('param_stability_score', 0) or 0) > 0.4 else 'bad'}">
    <div class="value">{_pct(ss.get('param_stability_score', 0))}</div>
    <div class="label">Param Stability</div>
  </div>
  <div class="metric-card">
    <div class="value">{_fmt(ss.get('is_oos_sharpe_ratio', 0), 2)}×</div>
    <div class="label">IS/OOS Ratio</div>
  </div>
  <div class="metric-card">
    <div class="value">{_fmt(ss.get('sharpe_degradation', 0), 3)}</div>
    <div class="label">Sharpe Degradation</div>
  </div>
  <div class="metric-card">
    <div class="value">{_pct(ss.get('combined_oos_win_rate', 0))}</div>
    <div class="label">OOS Win Rate</div>
  </div>
  <div class="metric-card">
    <div class="value">{_fmt(ss.get('combined_oos_pf', 0), 2)}</div>
    <div class="label">OOS Profit Factor</div>
  </div>
</div>

<p><strong>Best Params:</strong> <code>{ss.get('best_params', 'N/A')}</code></p>
<p><strong>95% CI for OOS Sharpe:</strong>
   [{_fmt(ss.get('oos_sharpe_ci_lower', 0), 3)}, {_fmt(ss.get('oos_sharpe_ci_upper', 0), 3)}]</p>
<p><strong>Chow Structural Break Test:</strong>
   F={_fmt(ss.get('chow_f_stat'), 3)}, p={_fmt(ss.get('chow_p_value'), 4)}
   — Significant: {chow_sig}</p>

<!-- ── Fold Breakdown ── -->
<h2>Fold-by-Fold Breakdown</h2>
{fold_html}

<!-- ── Plots ── -->
<h2>Charts</h2>
<div class="plot-grid">
  <div class="plot-card">
    <h3>IS vs OOS Sharpe Scatter</h3>
    {_img_tag('is_vs_oos')}
  </div>
  <div class="plot-card">
    <h3>OOS Equity Curves by Fold</h3>
    {_img_tag('equity_curves')}
  </div>
  <div class="plot-card">
    <h3>Parameter Selection Frequency</h3>
    {_img_tag('param_frequency')}
  </div>
  <div class="plot-card">
    <h3>Stability Dashboard</h3>
    {_img_tag('stability_dashboard')}
  </div>
</div>

<!-- ── Parameter Stability ── -->
<h2>Parameter Stability</h2>
<p>Stability Score: <strong>{_pct(stability_rpt.stability_score)}</strong> &nbsp;|&nbsp;
   IS/OOS Ratio: <strong>{_fmt(stability_rpt.is_oos_ratio, 3)}</strong> &nbsp;|&nbsp;
   Sharpe Dispersion: <strong>{_fmt(stability_rpt.sharpe_dispersion, 4)}</strong> &nbsp;|&nbsp;
   Sharpe CV: <strong>{_fmt(stability_rpt.sharpe_cv, 4)}</strong></p>
<table class="drift-table">
  <tr><th>Parameter</th><th>Drift (fraction not selecting modal)</th><th>Modal Value</th></tr>
  {drift_rows}
</table>

<footer>
  SRFM-Lab Walk-Forward Analysis Platform &nbsp;|&nbsp;
  Generated 2026 &nbsp;|&nbsp;
  All analysis is out-of-sample (OOS data never seen during IS optimization)
</footer>
</div>
</body>
</html>
""")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_numeric_str(s: str) -> bool:
    """Return True if string s can be interpreted as a number."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False
