"""
generator.py — PDF report generator for Spacetime Arena.

Generates professional research reports using ReportLab.
"""

from __future__ import annotations

import io
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "output"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Colors
DARK_BG     = (0.051, 0.067, 0.090)
ORANGE      = (0.969, 0.580, 0.102)
LIGHT_GRAY  = (0.85, 0.85, 0.85)
MID_GRAY    = (0.45, 0.45, 0.45)
WHITE       = (1.0, 1.0, 1.0)
RED         = (0.863, 0.196, 0.184)
GREEN       = (0.133, 0.694, 0.298)
BLUE        = (0.204, 0.596, 0.859)


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(
    run_names: List[str],
    backtest_results: Optional[List[Dict[str, Any]]] = None,
    mc_result: Optional[Any] = None,
    sensitivity_report: Optional[Any] = None,
    correlation_result: Optional[Any] = None,
    include_mc: bool = True,
    include_sensitivity: bool = True,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Generate a PDF research report.

    Parameters
    ----------
    run_names           : list of run/strategy names for the cover page
    backtest_results    : list of BacktestResult objects or dicts
    mc_result           : MCResult object
    sensitivity_report  : SensitivityReport object
    correlation_result  : CorrelationResult object
    include_mc          : include MC section
    include_sensitivity : include sensitivity section
    output_path         : override output file path

    Returns
    -------
    Path to the generated PDF.
    """
    from reportlab.lib.pagesizes import letter  # type: ignore
    from reportlab.lib.units import inch         # type: ignore
    from reportlab.pdfgen.canvas import Canvas   # type: ignore
    from reportlab.lib import colors             # type: ignore
    from reportlab.platypus import (            # type: ignore
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, Image,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT          # type: ignore

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"larsa_report_{ts}.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    page_w, page_h = letter
    margin = 0.75 * inch

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin + 0.3 * inch,  # room for footer
    )

    styles = getSampleStyleSheet()

    # Custom styles
    def _style(name: str, **kwargs) -> ParagraphStyle:
        return ParagraphStyle(name=name, parent=styles["Normal"], **kwargs)

    s_title     = _style("Title",    fontSize=22, textColor=colors.Color(*ORANGE),
                          alignment=TA_CENTER, spaceAfter=12, leading=26)
    s_h1        = _style("H1",       fontSize=16, textColor=colors.Color(*ORANGE),
                          spaceBefore=18, spaceAfter=8, leading=20)
    s_h2        = _style("H2",       fontSize=13, textColor=colors.Color(*LIGHT_GRAY),
                          spaceBefore=12, spaceAfter=6)
    s_body      = _style("Body",     fontSize=10, textColor=colors.Color(*LIGHT_GRAY),
                          leading=14, spaceAfter=6)
    s_small     = _style("Small",    fontSize=8,  textColor=colors.Color(*MID_GRAY))
    s_center    = _style("Center",   fontSize=10, textColor=colors.Color(*LIGHT_GRAY),
                          alignment=TA_CENTER)
    s_mono      = _style("Mono",     fontSize=9,  textColor=colors.Color(*LIGHT_GRAY),
                          fontName="Courier", leading=13)

    story: list = []

    def _hr() -> HRFlowable:
        return HRFlowable(width="100%", thickness=0.5,
                          color=colors.Color(*MID_GRAY), spaceAfter=8)

    # ── Cover Page ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("LARSA v16", s_title))
    story.append(Paragraph("Spacetime Arena Research Report", s_title))
    story.append(Spacer(1, 0.3 * inch))
    story.append(_hr())
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", s_center))
    story.append(Paragraph(f"Runs: {', '.join(run_names) or 'N/A'}", s_center))
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("CONFIDENTIAL — SRFM Research", _style("Conf",
        fontSize=11, textColor=colors.Color(*MID_GRAY), alignment=TA_CENTER)))
    story.append(PageBreak())

    # ── Section 1: Strategy Overview ──────────────────────────────────────────
    story.append(Paragraph("1. Strategy Overview", s_h1))
    story.append(_hr())
    story.append(Paragraph(
        "The LARSA v16 Black Hole (BH) engine classifies market moves using "
        "Minkowski spacetime geometry. Each price bar is tagged TIMELIKE (sub-luminal, "
        "β < 1) or SPACELIKE (super-luminal, β ≥ 1). Consecutive TIMELIKE bars "
        "accumulate gravitational mass; when mass crosses a formation threshold and "
        "the TIMELIKE count exceeds 5, a Black Hole is declared active — signaling a "
        "strong, causal momentum regime. The strategy enters long (or short) aligned "
        "with the BH direction and scales position size by the 3-timeframe alignment "
        "score (daily × 4 + hourly × 2 + 15m × 1).",
        s_body,
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Instrument Universe", s_h2))
    instrument_data = [
        ["Asset Class", "Instruments", "CF (causal factor)"],
        ["Equity Index", "ES, NQ, YM, RTY", "0.0008 – 0.0012"],
        ["Commodities",  "CL, NG, GC, SI",  "0.008 – 0.020"],
        ["Bonds",        "ZB, ZN",           "0.003"],
        ["Forex",        "EUR/USD, GBP/USD", "0.0005"],
        ["Crypto",       "BTC, ETH, SOL",    "0.005 – 0.010"],
        ["Volatility",   "VIX",              "0.025"],
    ]
    story.append(_make_table(instrument_data, col_widths=[1.5*inch, 2*inch, 1.8*inch]))
    story.append(PageBreak())

    # ── Section 2: Equity Curves ───────────────────────────────────────────────
    story.append(Paragraph("2. Equity Curves", s_h1))
    story.append(_hr())

    if backtest_results:
        for bt in backtest_results:
            sym = _get(bt, "sym", "Unknown")
            story.append(Paragraph(f"Instrument: {sym}", s_h2))
            try:
                img_buf = _plot_equity_curve(bt)
                if img_buf:
                    story.append(Image(img_buf, width=6.5*inch, height=2.8*inch))
            except Exception as e:
                logger.warning("Equity curve plot failed for %s: %s", sym, e)
                story.append(Paragraph(f"[Chart unavailable: {e}]", s_small))
            story.append(Spacer(1, 0.2 * inch))
    else:
        story.append(Paragraph("No backtest results provided.", s_body))

    story.append(PageBreak())

    # ── Section 3: Statistical Metrics ────────────────────────────────────────
    story.append(Paragraph("3. Statistical Metrics", s_h1))
    story.append(_hr())

    if backtest_results:
        metrics_header = ["Symbol", "CAGR", "Sharpe", "Max DD", "Win Rate", "PF", "Trades", "Avg Hold"]
        metrics_rows   = [metrics_header]
        for bt in backtest_results:
            s = _get(bt, "stats", {})
            if callable(getattr(bt, "__getitem__", None)) or hasattr(bt, "__dict__"):
                pass
            metrics_rows.append([
                str(_get(bt, "sym", "?")),
                f"{_get(s, 'cagr', 0):.1%}",
                f"{_get(s, 'sharpe', 0):.2f}",
                f"{_get(s, 'max_drawdown', 0):.1%}",
                f"{_get(s, 'win_rate', 0):.1%}",
                f"{min(_get(s, 'profit_factor', 0), 99):.2f}",
                str(_get(s, "trade_count", 0)),
                f"{_get(s, 'avg_hold_bars', 0):.1f}",
            ])
        story.append(_make_table(metrics_rows, col_widths=[0.9*inch]*8))
    else:
        story.append(Paragraph("No backtest results provided.", s_body))

    story.append(PageBreak())

    # ── Section 4: Benchmark Comparison ───────────────────────────────────────
    story.append(Paragraph("4. Benchmark Comparison", s_h1))
    story.append(_hr())

    sp500_cagr = _fetch_sp500_cagr()
    bench_data = [
        ["Benchmark / Strategy", "CAGR (approx.)"],
        ["S&P 500 Buy-Hold",     f"{sp500_cagr:.1%}"],
        ["Medallion Fund",       "~66%"],
    ]
    if backtest_results:
        for bt in backtest_results:
            s = _get(bt, "stats", {})
            bench_data.append([
                f"LARSA {_get(bt, 'sym', '?')}",
                f"{_get(s, 'cagr', 0):.1%}",
            ])
    story.append(_make_table(bench_data, col_widths=[3*inch, 2*inch]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "S&P 500 CAGR sourced from yfinance (SPY). Medallion Fund ~66% is a "
        "widely cited estimate for the Renaissance Technologies flagship fund.",
        s_small,
    ))
    story.append(PageBreak())

    # ── Section 5: Regime Analysis ────────────────────────────────────────────
    story.append(Paragraph("5. Regime Analysis", s_h1))
    story.append(_hr())

    if backtest_results:
        for bt in backtest_results:
            sym    = _get(bt, "sym", "?")
            trades = _get(bt, "trades", [])
            if trades:
                regime_stats = _regime_stats(trades)
                story.append(Paragraph(f"{sym} — Trades by Regime", s_h2))
                r_header = ["Regime", "Trades", "Win Rate", "Avg PnL"]
                r_rows   = [r_header]
                for regime, rs in sorted(regime_stats.items()):
                    r_rows.append([
                        regime,
                        str(rs["count"]),
                        f"{rs['win_rate']:.1%}",
                        f"${rs['avg_pnl']:,.0f}",
                    ])
                story.append(_make_table(r_rows, col_widths=[1.5*inch, 1*inch, 1.2*inch, 1.2*inch]))
                story.append(Spacer(1, 0.15 * inch))
    else:
        story.append(Paragraph("No backtest results provided.", s_body))

    story.append(PageBreak())

    # ── Section 6: Drawdown Analysis ──────────────────────────────────────────
    story.append(Paragraph("6. Drawdown Analysis", s_h1))
    story.append(_hr())

    if backtest_results:
        for bt in backtest_results:
            sym   = _get(bt, "sym", "?")
            curve = _get(bt, "equity_curve", [])
            if curve:
                story.append(Paragraph(f"{sym} — Drawdown", s_h2))
                try:
                    img_buf = _plot_drawdown(curve)
                    if img_buf:
                        story.append(Image(img_buf, width=6.5*inch, height=2.2*inch))
                except Exception as e:
                    story.append(Paragraph(f"[Chart unavailable: {e}]", s_small))

                dd_stats = _drawdown_stats(curve)
                dd_data  = [
                    ["Metric", "Value"],
                    ["Max Drawdown",   f"{dd_stats['max_dd']:.1%}"],
                    ["Avg Drawdown",   f"{dd_stats['avg_dd']:.1%}"],
                    ["Longest DD (bars)", str(dd_stats["longest_dd"])],
                ]
                story.append(_make_table(dd_data, col_widths=[2.5*inch, 1.5*inch]))
                story.append(Spacer(1, 0.15 * inch))
    else:
        story.append(Paragraph("No backtest results provided.", s_body))

    story.append(PageBreak())

    # ── Section 7: BH Correlation Matrix ──────────────────────────────────────
    story.append(Paragraph("7. BH Activation Correlation", s_h1))
    story.append(_hr())

    if correlation_result is not None:
        try:
            img_buf = _plot_correlation_heatmap(correlation_result)
            if img_buf:
                story.append(Image(img_buf, width=5*inch, height=4*inch))
            story.append(Spacer(1, 0.1 * inch))
            opt = getattr(correlation_result, "optimal_portfolio", [])
            div = getattr(correlation_result, "diversification_score", 0.0)
            story.append(Paragraph(
                f"Optimal diversified portfolio: {', '.join(opt)}  "
                f"(diversification score: {div:.3f})",
                s_body,
            ))
        except Exception as e:
            story.append(Paragraph(f"[Correlation heatmap unavailable: {e}]", s_small))
    else:
        story.append(Paragraph("Correlation analysis not included in this run.", s_body))

    story.append(PageBreak())

    # ── Section 8: Monte Carlo ─────────────────────────────────────────────────
    story.append(Paragraph("8. Monte Carlo Analysis", s_h1))
    story.append(_hr())

    if include_mc and mc_result is not None:
        try:
            img_buf = _plot_mc(mc_result)
            if img_buf:
                story.append(Image(img_buf, width=6.5*inch, height=3.2*inch))
        except Exception as e:
            story.append(Paragraph(f"[MC chart unavailable: {e}]", s_small))

        mc_data = [
            ["Metric", "Value"],
            ["Blowup Rate",         f"{getattr(mc_result, 'blowup_rate', 0):.1%}"],
            ["Median Final Equity", f"${getattr(mc_result, 'median_equity', 0):,.0f}"],
            ["5th Percentile",      f"${getattr(mc_result, 'pct_5', 0):,.0f}"],
            ["25th Percentile",     f"${getattr(mc_result, 'pct_25', 0):,.0f}"],
            ["75th Percentile",     f"${getattr(mc_result, 'pct_75', 0):,.0f}"],
            ["95th Percentile",     f"${getattr(mc_result, 'pct_95', 0):,.0f}"],
            ["Kelly Fraction",      f"{getattr(mc_result, 'kelly_fraction', 0):.3f}"],
            ["Trades/Month",        f"{getattr(mc_result, 'trades_per_month', 0):.1f}"],
        ]
        story.append(_make_table(mc_data, col_widths=[2.5*inch, 2*inch]))
    else:
        story.append(Paragraph("Monte Carlo not included in this report.", s_body))

    story.append(PageBreak())

    # ── Section 9: Parameter Sensitivity ──────────────────────────────────────
    story.append(Paragraph("9. Parameter Sensitivity", s_h1))
    story.append(_hr())

    if include_sensitivity and sensitivity_report is not None:
        summary = getattr(sensitivity_report, "edge_summary", "")
        if summary:
            story.append(Paragraph(summary.replace("\n", "<br/>"), s_mono))
        try:
            img_buf = _plot_sensitivity_heatmap(sensitivity_report)
            if img_buf:
                story.append(Spacer(1, 0.2 * inch))
                story.append(Image(img_buf, width=6*inch, height=3*inch))
        except Exception as e:
            story.append(Paragraph(f"[Sensitivity heatmap unavailable: {e}]", s_small))
    else:
        story.append(Paragraph("Sensitivity analysis not included in this report.", s_body))

    # ── Build with footer ──────────────────────────────────────────────────────
    def _add_footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColorRGB(*MID_GRAY)
        canvas.drawCentredString(
            page_w / 2,
            0.4 * inch,
            "CONFIDENTIAL — SRFM Research",
        )
        canvas.drawRightString(
            page_w - margin,
            0.4 * inch,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_add_footer, onLaterPages=_add_footer)
    logger.info("Report generated: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Table helper
# ---------------------------------------------------------------------------

def _make_table(data: List[List[str]], col_widths: Optional[List[Any]] = None) -> Any:
    from reportlab.platypus import Table, TableStyle  # type: ignore
    from reportlab.lib import colors                  # type: ignore

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.Color(0.15, 0.15, 0.18)),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.Color(*ORANGE)),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",    (0, 1), (-1, -1), colors.Color(*LIGHT_GRAY)),
        ("BACKGROUND",   (0, 1), (-1, -1), colors.Color(0.08, 0.09, 0.11)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.Color(0.09, 0.10, 0.12), colors.Color(0.07, 0.08, 0.10)]),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.Color(0.25, 0.25, 0.28)),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return t


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_equity_curve(bt: Any) -> Optional[io.BytesIO]:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.figure as mfig

        curve = _get(bt, "equity_curve", [])
        if not curve:
            return None

        dates  = [str(t)[:10] for t, _ in curve]
        values = [v for _, v in curve]

        fig, ax = plt.subplots(figsize=(8, 3.2))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        ax.plot(range(len(values)), values, color="#f7931a", linewidth=1.2)
        ax.fill_between(range(len(values)), values, alpha=0.12, color="#f7931a")
        ax.axhline(values[0], color="#555", linewidth=0.5, linestyle="--")

        stats = _get(bt, "stats", {})
        ax.set_title(
            f"{_get(bt, 'sym', '')}  CAGR: {_get(stats, 'cagr', 0):.1%}  "
            f"Sharpe: {_get(stats, 'sharpe', 0):.2f}  "
            f"MaxDD: {_get(stats, 'max_drawdown', 0):.1%}",
            color="white", fontsize=9,
        )
        ax.tick_params(colors="white", labelsize=7)
        ax.set_ylabel("Equity ($)", color="white", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.grid(alpha=0.1, color="white")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        buf.seek(0)
        return buf
    except ImportError:
        return None


def _plot_drawdown(curve: List[Tuple]) -> Optional[io.BytesIO]:
    try:
        import matplotlib.pyplot as plt

        values = np.array([v for _, v in curve])
        pk     = np.maximum.accumulate(values)
        dd     = (values - pk) / (pk + 1e-9) * 100

        fig, ax = plt.subplots(figsize=(8, 2.5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax.fill_between(range(len(dd)), dd, color="#ff4444", alpha=0.7)
        ax.set_ylabel("DD %", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.grid(alpha=0.1, color="white", axis="y")
        ax.set_title("Drawdown", color="white", fontsize=9)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        buf.seek(0)
        return buf
    except ImportError:
        return None


def _plot_mc(mc_result: Any) -> Optional[io.BytesIO]:
    try:
        import matplotlib.pyplot as plt

        eq = getattr(mc_result, "final_equities", np.array([]))
        if len(eq) == 0:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
        fig.patch.set_facecolor("#0d1117")

        for ax in [ax1, ax2]:
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

        cap = np.percentile(eq, 98)
        cl  = np.clip(eq, 0, cap)
        ax1.hist(cl / 1e6, bins=60, color="#f7931a", alpha=0.7, edgecolor="none")
        ax1.axvline(float(np.median(eq)) / 1e6, color="white", linewidth=1.2,
                    linestyle="--", label=f"Median ${np.median(eq)/1e6:.2f}M")
        ax1.set_title("MC Distribution", color="white", fontsize=9)
        ax1.set_xlabel("Final Equity ($M)", color="white", fontsize=8)
        ax1.legend(fontsize=7, labelcolor="white", facecolor="#111")
        ax1.grid(alpha=0.1, color="white")

        pcts    = [5, 25, 50, 75, 95]
        colors  = ["#ff4444", "#ff8c00", "#f7931a", "#ffcc00", "#00d4aa"]
        pct_vals = [np.percentile(eq, p) / 1e6 for p in pcts]
        bars = ax2.bar([f"p{p}" for p in pcts], pct_vals, color=colors, alpha=0.85)
        ax2.set_title("MC Percentiles", color="white", fontsize=9)
        ax2.set_ylabel("Final Equity ($M)", color="white", fontsize=8)
        ax2.grid(alpha=0.1, color="white", axis="y")
        for bar, val in zip(bars, pct_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"${val:.2f}M", ha="center", va="bottom", color="white", fontsize=7)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        buf.seek(0)
        return buf
    except ImportError:
        return None


def _plot_correlation_heatmap(corr_result: Any) -> Optional[io.BytesIO]:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        syms    = getattr(corr_result, "instruments", [])
        matrix  = getattr(corr_result, "jaccard_matrix", None)
        if matrix is None or len(syms) == 0:
            return None

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Jaccard Similarity")
        ax.set_xticks(range(len(syms)))
        ax.set_yticks(range(len(syms)))
        ax.set_xticklabels(syms, rotation=45, ha="right", fontsize=7, color="white")
        ax.set_yticklabels(syms, fontsize=7, color="white")
        ax.set_title("BH Activation Jaccard Similarity", color="white", fontsize=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        # Annotate cells
        for i in range(len(syms)):
            for j in range(len(syms)):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="black" if matrix[i, j] > 0.5 else "white")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        buf.seek(0)
        return buf
    except ImportError:
        return None


def _plot_sensitivity_heatmap(sensitivity_report: Any) -> Optional[io.BytesIO]:
    try:
        import matplotlib.pyplot as plt

        params_dict = getattr(sensitivity_report, "params", {})
        if not params_dict:
            return None

        param_names = list(params_dict.keys())
        mults       = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
        n_params    = len(param_names)
        n_mults     = len(mults)

        sharpe_grid = np.zeros((n_params, n_mults))
        for i, pname in enumerate(param_names):
            ps  = params_dict[pname]
            mvs = getattr(ps, "metric_values", {}).get("sharpe", [])
            for j, v in enumerate(mvs[:n_mults]):
                sharpe_grid[i, j] = v

        fig, ax = plt.subplots(figsize=(7, max(2.5, n_params * 0.6 + 1)))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        im = ax.imshow(sharpe_grid, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax, label="Sharpe")
        ax.set_xticks(range(n_mults))
        ax.set_xticklabels([f"×{m}" for m in mults], fontsize=8, color="white")
        ax.set_yticks(range(n_params))
        ax.set_yticklabels(param_names, fontsize=8, color="white")
        ax.set_title("Sharpe Sensitivity Heatmap", color="white", fontsize=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        for i in range(n_params):
            for j in range(n_mults):
                ax.text(j, i, f"{sharpe_grid[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if abs(sharpe_grid[i, j]) < 2 else "white")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        buf.seek(0)
        return buf
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute from dataclass or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _regime_stats(trades: list) -> Dict[str, Dict[str, Any]]:
    by_regime: Dict[str, list] = {}
    for t in trades:
        r   = str(_get(t, "regime", "UNKNOWN"))
        pnl = float(_get(t, "pnl", 0.0))
        by_regime.setdefault(r, []).append(pnl)

    result = {}
    for r, pnls in by_regime.items():
        wins = [p for p in pnls if p > 0]
        result[r] = {
            "count":    len(pnls),
            "win_rate": len(wins) / len(pnls) if pnls else 0.0,
            "avg_pnl":  float(np.mean(pnls)) if pnls else 0.0,
        }
    return result


def _drawdown_stats(curve: List[Tuple]) -> Dict[str, Any]:
    values = np.array([v for _, v in curve])
    pk     = np.maximum.accumulate(values)
    dd     = (values - pk) / (pk + 1e-9)
    max_dd  = float(dd.min())
    avg_dd  = float(dd[dd < -0.001].mean()) if (dd < -0.001).any() else 0.0

    # Longest drawdown
    in_dd = False
    dd_len = curr_len = 0
    for d in dd:
        if d < -0.001:
            curr_len += 1
            in_dd = True
        else:
            if in_dd:
                dd_len = max(dd_len, curr_len)
                curr_len = 0
                in_dd = False
    dd_len = max(dd_len, curr_len)

    return {"max_dd": max_dd, "avg_dd": avg_dd, "longest_dd": dd_len}


def _fetch_sp500_cagr() -> float:
    """Fetch SPY CAGR over last 5 years from yfinance."""
    try:
        import yfinance as yf  # type: ignore
        from datetime import timedelta

        end   = datetime.now()
        start = end - timedelta(days=5 * 365)
        df    = yf.download("SPY", start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
        if df.empty:
            return 0.10
        prices = df["Close"].values
        cagr   = (prices[-1] / prices[0]) ** (1 / 5) - 1
        return float(cagr)
    except Exception:
        return 0.10  # fallback
