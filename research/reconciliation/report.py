"""
research/reconciliation/report.py
====================================
Report generation for the live-vs-backtest reconciliation pipeline.

Produces:
  1. A ``ReconciliationReport`` dataclass containing all computed metrics.
  2. An HTML report with embedded charts, comparison tables, and summaries.
  3. Rich terminal output with colour-coded performance tables.

Entry points
------------
``generate_full_report(live_trades, bt_trades, output_dir)`` → ReconciliationReport
    Orchestrates all sub-analyses and writes artefacts to disk.

``to_html(report, path)``
    Render the report to a self-contained HTML file.

``to_console(report)``
    Print a colour-coded summary to the terminal using `rich`.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from research.reconciliation.loader import (
    LiveTradeLoader,
    BacktestTradeLoader,
    merge_live_backtest,
    compute_rolling_metrics,
    stratify_by_regime,
    _records_to_df,
)
from research.reconciliation.slippage import SlippageAnalyzer, FillReport
from research.reconciliation.drift import SignalDriftDetector, ParameterStabilityResult
from research.reconciliation.attribution import PnLAttributionEngine, AttributionReport
from research.reconciliation.leakage import DataLeakageAuditor, LeakageReport

log = logging.getLogger(__name__)

# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class PerformanceMetrics:
    """Core performance metrics for one set of trades."""
    source: str
    n_trades: int
    total_pnl: float
    win_rate: float
    profit_factor: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    cagr: float
    avg_hold_hours: float
    best_trade: float
    worst_trade: float
    expectancy: float    # win_rate * avg_win + (1 - win_rate) * avg_loss
    skewness: float
    kurtosis: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ReconciliationReport:
    """
    Full live-vs-backtest reconciliation report.

    All sub-reports and DataFrames are stored here so that downstream
    rendering (HTML, console, JSON) can access any level of detail.
    """
    generated_at: str
    live_metrics: PerformanceMetrics
    bt_metrics: PerformanceMetrics
    regime_comparison: pd.DataFrame    # regime-stratified side-by-side
    merged_trades: pd.DataFrame        # raw merged trade table
    fill_report: FillReport
    attribution_report: AttributionReport
    leakage_report: LeakageReport
    parameter_stability: ParameterStabilityResult
    slippage_summary: dict[str, Any]
    drift_summary: dict[str, Any]
    plot_paths: dict[str, str]         # label → file path
    output_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def overall_grade(self) -> str:
        """
        Assign an overall A-F quality grade based on:
          * Leakage score
          * Overfitting probability
          * Sharpe degradation (live vs BT)
          * Parameter stability
        """
        score = 100.0
        if self.leakage_report.lookahead_suspected:
            score -= 30
        if self.leakage_report.overfitting_probability > 0.5:
            score -= 25
        if not self.parameter_stability.is_stable:
            score -= 15

        if not math.isnan(self.live_metrics.sharpe) and not math.isnan(self.bt_metrics.sharpe):
            if self.bt_metrics.sharpe > 0:
                degradation = (self.bt_metrics.sharpe - self.live_metrics.sharpe) / abs(self.bt_metrics.sharpe)
                if degradation > 0.5:
                    score -= 20
                elif degradation > 0.25:
                    score -= 10

        if score >= 85:
            return "A"
        if score >= 70:
            return "B"
        if score >= 55:
            return "C"
        if score >= 40:
            return "D"
        return "F"


# ── Performance metric computation ───────────────────────────────────────────


def _compute_metrics(
    df: pd.DataFrame,
    source: str,
    pnl_col: str = "pnl",
    annual_factor: float = 252.0,
) -> PerformanceMetrics:
    """
    Compute all core performance metrics from a trade DataFrame.
    """
    if df.empty or pnl_col not in df.columns:
        nan = float("nan")
        return PerformanceMetrics(
            source=source, n_trades=0, total_pnl=0.0,
            win_rate=nan, profit_factor=nan, avg_pnl=nan,
            avg_win=nan, avg_loss=nan, max_drawdown=nan,
            sharpe=nan, sortino=nan, calmar=nan, cagr=nan,
            avg_hold_hours=nan, best_trade=nan, worst_trade=nan,
            expectancy=nan, skewness=nan, kurtosis=nan,
        )

    pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna()
    n = len(pnl)
    if n == 0:
        nan = float("nan")
        return PerformanceMetrics(
            source=source, n_trades=0, total_pnl=0.0,
            win_rate=nan, profit_factor=nan, avg_pnl=nan,
            avg_win=nan, avg_loss=nan, max_drawdown=nan,
            sharpe=nan, sortino=nan, calmar=nan, cagr=nan,
            avg_hold_hours=nan, best_trade=nan, worst_trade=nan,
            expectancy=nan, skewness=nan, kurtosis=nan,
        )

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    win_rate = float((pnl > 0).mean())
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(losses.abs().sum()) if len(losses) > 0 else 1e-8
    profit_factor = gross_profit / gross_loss

    # Sharpe
    std_pnl = float(pnl.std())
    sharpe = float(pnl.mean() / std_pnl * math.sqrt(annual_factor)) if std_pnl > 0 else float("nan")

    # Sortino (downside std)
    downside = pnl[pnl < 0]
    dstd = float(downside.std()) if len(downside) > 1 else float("nan")
    sortino = float(pnl.mean() / dstd * math.sqrt(annual_factor)) if dstd and dstd > 0 else float("nan")

    # Max drawdown
    cum = pnl.cumsum()
    running_max = cum.cummax()
    drawdowns = cum - running_max
    max_dd = float(drawdowns.min())

    # CAGR (rough, using total return over calendar span)
    time_col = None
    for tc in ("exit_time", "live_exit_time", "bt_exit_time", "ts"):
        if tc in df.columns:
            time_col = tc
            break

    cagr = float("nan")
    avg_hold = float("nan")

    if time_col and time_col in df.columns:
        times = pd.to_datetime(df[time_col], utc=True, errors="coerce").dropna()
        if len(times) >= 2:
            span_years = (times.max() - times.min()).total_seconds() / (365.25 * 86400)
            if span_years > 0.01:
                initial = 100_000.0  # assumed starting equity
                final = initial + float(pnl.sum())
                if final > 0 and initial > 0:
                    cagr = (final / initial) ** (1 / span_years) - 1

    hold_col = None
    for hc in ("hold_hours", "live_hold_hours", "bt_hold_hours"):
        if hc in df.columns:
            hold_col = hc
            break
    if hold_col:
        avg_hold = float(pd.to_numeric(df[hold_col], errors="coerce").mean())

    # Calmar
    calmar = float("nan")
    if not math.isnan(cagr) and max_dd < 0:
        calmar = cagr / abs(max_dd)

    # Distribution stats
    skewness = float(stats.skew(pnl.values))
    kurt = float(stats.kurtosis(pnl.values, fisher=False))  # full kurtosis

    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return PerformanceMetrics(
        source=source,
        n_trades=n,
        total_pnl=float(pnl.sum()),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_pnl=float(pnl.mean()),
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown=max_dd,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        cagr=cagr,
        avg_hold_hours=avg_hold,
        best_trade=float(pnl.max()),
        worst_trade=float(pnl.min()),
        expectancy=expectancy,
        skewness=skewness,
        kurtosis=kurt,
    )


def _regime_comparison_table(
    live_df: pd.DataFrame,
    bt_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a side-by-side regime comparison table.
    """
    all_regimes = {"BULL", "BEAR", "SIDEWAYS", "HIGH_VOL", "UNKNOWN"}

    def _regime_col(df: pd.DataFrame) -> str:
        for c in ("regime", "live_regime", "bt_regime"):
            if c in df.columns:
                return c
        return "regime"

    def _pnl_col(df: pd.DataFrame) -> str:
        for c in ("pnl", "live_pnl", "bt_pnl", "return_pct"):
            if c in df.columns:
                return c
        return "pnl"

    rows = []
    live_reg = _regime_col(live_df)
    bt_reg = _regime_col(bt_df)
    live_pnl_c = _pnl_col(live_df)
    bt_pnl_c = _pnl_col(bt_df)

    regimes = set()
    if live_reg in live_df.columns:
        regimes |= set(live_df[live_reg].dropna().astype(str).unique())
    if bt_reg in bt_df.columns:
        regimes |= set(bt_df[bt_reg].dropna().astype(str).unique())
    if not regimes:
        regimes = all_regimes

    for regime in sorted(regimes):
        live_sub = live_df[live_df.get(live_reg, pd.Series(dtype=str)) == regime] if live_reg in live_df.columns else pd.DataFrame()
        bt_sub = bt_df[bt_df.get(bt_reg, pd.Series(dtype=str)) == regime] if bt_reg in bt_df.columns else pd.DataFrame()

        live_pnl = pd.to_numeric(live_sub.get(live_pnl_c, pd.Series(dtype=float)), errors="coerce").dropna()
        bt_pnl = pd.to_numeric(bt_sub.get(bt_pnl_c, pd.Series(dtype=float)), errors="coerce").dropna()

        def _sr(p: pd.Series) -> float:
            if len(p) < 2 or p.std() < 1e-10:
                return float("nan")
            return float(p.mean() / p.std() * math.sqrt(252))

        rows.append({
            "regime": regime,
            "live_n": len(live_pnl),
            "bt_n": len(bt_pnl),
            "live_total_pnl": float(live_pnl.sum()) if len(live_pnl) > 0 else float("nan"),
            "bt_total_pnl": float(bt_pnl.sum()) if len(bt_pnl) > 0 else float("nan"),
            "live_win_rate": float((live_pnl > 0).mean()) if len(live_pnl) > 0 else float("nan"),
            "bt_win_rate": float((bt_pnl > 0).mean()) if len(bt_pnl) > 0 else float("nan"),
            "live_sharpe": _sr(live_pnl),
            "bt_sharpe": _sr(bt_pnl),
            "live_avg_pnl": float(live_pnl.mean()) if len(live_pnl) > 0 else float("nan"),
            "bt_avg_pnl": float(bt_pnl.mean()) if len(bt_pnl) > 0 else float("nan"),
            "pnl_diff": float(live_pnl.sum() - bt_pnl.sum())
            if len(live_pnl) > 0 and len(bt_pnl) > 0 else float("nan"),
        })

    return pd.DataFrame(rows).set_index("regime")


# ── Chart generation helpers ──────────────────────────────────────────────────


def _figure_to_base64() -> str:
    """Return the current matplotlib figure as a base64-encoded PNG."""
    import matplotlib.pyplot as plt
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return data


def _plot_equity_curves(
    live_df: pd.DataFrame,
    bt_df: pd.DataFrame,
    save_path: Path,
    dpi: int = 150,
) -> Path:
    """Plot cumulative PnL for live and backtest side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))

    def _plot_cum(df: pd.DataFrame, label: str, color: str) -> None:
        pnl_col = None
        for c in ("pnl", "live_pnl", "bt_pnl", "return_pct"):
            if c in df.columns:
                pnl_col = c
                break
        if pnl_col is None or df.empty:
            return
        pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0)
        cum = pnl.cumsum().values
        ax.plot(range(len(cum)), cum, label=label, color=color, linewidth=1.5)
        ax.fill_between(range(len(cum)), cum, alpha=0.07, color=color)

    _plot_cum(live_df, "Live", "steelblue")
    _plot_cum(bt_df, "Backtest", "orange")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("Equity Curve: Live vs Backtest")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _plot_pnl_histogram(
    live_df: pd.DataFrame,
    bt_df: pd.DataFrame,
    save_path: Path,
    dpi: int = 150,
) -> Path:
    """Overlapping PnL histograms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    def _pnl(df: pd.DataFrame) -> np.ndarray:
        for c in ("pnl", "live_pnl", "bt_pnl"):
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce").dropna().values
        return np.array([])

    lp = _pnl(live_df)
    bp = _pnl(bt_df)

    if len(lp) > 0:
        ax.hist(lp, bins=50, alpha=0.5, color="steelblue", label=f"Live (n={len(lp)})", density=True)
    if len(bp) > 0:
        ax.hist(bp, bins=50, alpha=0.5, color="orange", label=f"Backtest (n={len(bp)})", density=True)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("PnL ($)")
    ax.set_ylabel("Density")
    ax.set_title("PnL Distribution: Live vs Backtest")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _plot_regime_bar(regime_df: pd.DataFrame, save_path: Path, dpi: int = 150) -> Path:
    """Side-by-side bars of win rate by regime."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if not regime_df.empty:
        for ax, col_pair, title in [
            (axes[0], ("live_win_rate", "bt_win_rate"), "Win Rate by Regime"),
            (axes[1], ("live_avg_pnl", "bt_avg_pnl"), "Avg PnL by Regime"),
        ]:
            regimes = regime_df.index.tolist()
            x = np.arange(len(regimes))
            width = 0.35
            l_vals = pd.to_numeric(regime_df.get(col_pair[0], pd.Series()), errors="coerce").fillna(0)
            b_vals = pd.to_numeric(regime_df.get(col_pair[1], pd.Series()), errors="coerce").fillna(0)
            ax.bar(x - width / 2, l_vals, width, label="Live", alpha=0.8, color="steelblue")
            ax.bar(x + width / 2, b_vals, width, label="Backtest", alpha=0.8, color="orange")
            ax.set_xticks(x)
            ax.set_xticklabels(regimes, rotation=30, ha="right")
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.axhline(0, color="black", linewidth=0.5)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No regime data", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ── HTML rendering ────────────────────────────────────────────────────────────


def _metrics_to_html_rows(m: PerformanceMetrics) -> list[tuple[str, str]]:
    def _fmt(v: Any, fmt: str = ".2f") -> str:
        try:
            return f"{v:{fmt}}" if not math.isnan(float(v)) else "—"
        except (TypeError, ValueError):
            return str(v)

    return [
        ("N Trades", _fmt(m.n_trades, "d")),
        ("Total PnL", f"${_fmt(m.total_pnl, ',.0f')}"),
        ("Win Rate", f"{_fmt(m.win_rate * 100, '.1f')}%"),
        ("Profit Factor", _fmt(m.profit_factor)),
        ("Avg PnL / Trade", f"${_fmt(m.avg_pnl, ',.2f')}"),
        ("Avg Win", f"${_fmt(m.avg_win, ',.2f')}"),
        ("Avg Loss", f"${_fmt(m.avg_loss, ',.2f')}"),
        ("Max Drawdown", f"${_fmt(m.max_drawdown, ',.0f')}"),
        ("Sharpe", _fmt(m.sharpe)),
        ("Sortino", _fmt(m.sortino)),
        ("Calmar", _fmt(m.calmar)),
        ("CAGR", f"{_fmt(m.cagr * 100, '.1f')}%" if not math.isnan(m.cagr) else "—"),
        ("Avg Hold (h)", _fmt(m.avg_hold_hours)),
        ("Best Trade", f"${_fmt(m.best_trade, ',.0f')}"),
        ("Worst Trade", f"${_fmt(m.worst_trade, ',.0f')}"),
        ("Expectancy", f"${_fmt(m.expectancy, ',.2f')}"),
        ("Skewness", _fmt(m.skewness)),
        ("Kurtosis", _fmt(m.kurtosis)),
    ]


def to_html(report: ReconciliationReport, path: str | Path) -> Path:
    """
    Render the ReconciliationReport to a self-contained HTML file with
    embedded charts (base64 PNG) and styled tables.

    Parameters
    ----------
    report : ReconciliationReport
    path : str | Path
        Output file path.

    Returns
    -------
    Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grade = report.overall_grade()
    grade_color = {
        "A": "#4CAF50", "B": "#8BC34A", "C": "#FF9800",
        "D": "#FF5722", "F": "#F44336",
    }.get(grade, "#607D8B")

    # Embed plots as base64
    def _embed_img(plot_key: str) -> str:
        img_path = report.plot_paths.get(plot_key, "")
        if img_path and Path(img_path).exists():
            with open(img_path, "rb") as fh:
                data = base64.b64encode(fh.read()).decode()
            return f'<img src="data:image/png;base64,{data}" style="max-width:100%;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.15);">'
        return '<p style="color:#999;font-style:italic;">Chart not available.</p>'

    # Build metrics comparison table
    live_rows = _metrics_to_html_rows(report.live_metrics)
    bt_rows = _metrics_to_html_rows(report.bt_metrics)
    metrics_table_rows = ""
    for (label, lv), (_, bv) in zip(live_rows, bt_rows):
        metrics_table_rows += f"<tr><td>{label}</td><td>{lv}</td><td>{bv}</td></tr>\n"

    # Regime table
    regime_html = report.regime_comparison.to_html(
        classes="data-table",
        float_format=lambda x: f"{x:.3f}" if not math.isnan(x) else "—",
        border=0,
    ) if not report.regime_comparison.empty else "<p>No regime data.</p>"

    # Leakage summary
    lr = report.leakage_report
    leakage_color = "#F44336" if lr.lookahead_suspected else "#4CAF50"
    leakage_label = "⚠ SUSPECTED" if lr.lookahead_suspected else "✓ OK"

    # Attribution
    ar = report.attribution_report
    attr_table = ar.effect_table().to_html(
        classes="data-table",
        float_format=lambda x: f"{x:.2f}" if not math.isnan(x) else "—",
        border=0,
    ) if ar.n_trades > 0 else "<p>No attribution data.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>srfm-lab Reconciliation Report – {report.generated_at}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f1117; color: #e0e0e0; padding: 20px; }}
  h1 {{ color: #64b5f6; margin-bottom: 8px; font-size: 1.6rem; }}
  h2 {{ color: #90caf9; margin: 24px 0 10px; font-size: 1.2rem; border-bottom: 1px solid #2a2a3a; padding-bottom: 6px; }}
  h3 {{ color: #bbdefb; margin: 16px 0 8px; font-size: 1.0rem; }}
  .meta {{ color: #9e9e9e; font-size: 0.85rem; margin-bottom: 20px; }}
  .grade {{ display: inline-block; padding: 6px 16px; border-radius: 20px;
            background: {grade_color}; color: white; font-size: 1.4rem; font-weight: bold; }}
  .card {{ background: #1a1c27; border-radius: 8px; padding: 16px; margin: 12px 0;
           box-shadow: 0 2px 6px rgba(0,0,0,.3); }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
  table.data-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  table.data-table th {{ background: #252840; color: #90caf9; padding: 8px 10px;
                         text-align: left; position: sticky; top: 0; }}
  table.data-table td {{ padding: 6px 10px; border-bottom: 1px solid #252840; }}
  table.data-table tr:hover td {{ background: #252840; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.78rem; font-weight:bold; }}
  .badge-ok {{ background:#1b5e20; color:#a5d6a7; }}
  .badge-warn {{ background:#e65100; color:#ffe0b2; }}
  .stat-box {{ text-align:center; }}
  .stat-box .val {{ font-size:1.5rem; font-weight:bold; color:#64b5f6; }}
  .stat-box .lbl {{ font-size:0.75rem; color:#9e9e9e; margin-top:2px; }}
  .section-divider {{ height:2px; background: linear-gradient(90deg,#3949ab,#1565c0,transparent); margin:24px 0; }}
  pre {{ background:#252840; padding:12px; border-radius:6px; font-size:0.8rem; overflow-x:auto; }}
  .leakage-badge {{ display:inline-block; padding:4px 12px; border-radius:4px;
                    background:{leakage_color}33; color:{leakage_color}; font-weight:bold; border:1px solid {leakage_color}; }}
</style>
</head>
<body>

<h1>srfm-lab Live vs Backtest Reconciliation</h1>
<div class="meta">
  Generated: {report.generated_at} &nbsp;|&nbsp;
  Output Dir: <code>{report.output_dir}</code> &nbsp;|&nbsp;
  Grade: <span class="grade">{grade}</span>
</div>

<div class="section-divider"></div>

<h2>1. Performance Overview</h2>
<div class="card">
  <table class="data-table">
    <thead><tr><th>Metric</th><th>Live</th><th>Backtest</th></tr></thead>
    <tbody>{metrics_table_rows}</tbody>
  </table>
</div>

<div class="section-divider"></div>

<h2>2. Equity Curves</h2>
<div class="card">{_embed_img("equity_curves")}</div>

<h2>3. PnL Distribution</h2>
<div class="card">{_embed_img("pnl_histogram")}</div>

<div class="section-divider"></div>

<h2>4. Regime-Stratified Comparison</h2>
<div class="card">{regime_html}</div>
<div class="card">{_embed_img("regime_comparison")}</div>

<div class="section-divider"></div>

<h2>5. Slippage Analysis</h2>
<div class="card">
  <div class="grid-3">
    <div class="stat-box">
      <div class="val">{report.slippage_summary.get('avg_cost_per_trade_bps', float('nan')):.1f}</div>
      <div class="lbl">Avg Slippage (bps)</div>
    </div>
    <div class="stat-box">
      <div class="val">${report.slippage_summary.get('total_cost_usd', 0):,.0f}</div>
      <div class="lbl">Total Slippage Cost</div>
    </div>
    <div class="stat-box">
      <div class="val">{report.slippage_summary.get('annualised_drag_bps', float('nan')):.0f}</div>
      <div class="lbl">Annualised Drag (bps)</div>
    </div>
  </div>
</div>
<div class="card">{_embed_img("slippage_distribution")}</div>

<div class="section-divider"></div>

<h2>6. PnL Attribution</h2>
<div class="card">{attr_table}</div>
<div class="card">{_embed_img("attribution_waterfall")}</div>
<div class="card">{_embed_img("regime_attribution")}</div>
<div class="card">{_embed_img("rolling_ic")}</div>

<div class="section-divider"></div>

<h2>7. Data Leakage &amp; Overfitting Audit</h2>
<div class="card">
  <div class="grid-3">
    <div class="stat-box">
      <div class="val"><span class="leakage-badge">{leakage_label}</span></div>
      <div class="lbl">Lookahead Bias</div>
    </div>
    <div class="stat-box">
      <div class="val">{lr.deflated_sharpe:.3f}</div>
      <div class="lbl">Deflated Sharpe (DSR)</div>
    </div>
    <div class="stat-box">
      <div class="val">{lr.overfitting_probability:.1%}</div>
      <div class="lbl">Overfitting Probability</div>
    </div>
  </div>
  <div style="margin-top:12px;">
    <strong>Purged Trades:</strong> {lr.n_purged} / {lr.n_purged + lr.n_remaining}
    ({lr.purge_fraction:.1%}) &nbsp;|&nbsp;
    <strong>AC-Adj Sharpe:</strong> {lr.ac_adjusted_sharpe:.2f} &nbsp;|&nbsp;
    <strong>AC Inflation Score:</strong> {lr.autocorrelation_score:.3f}
  </div>
  {"<div style='margin-top:8px;'><strong>High-VIF Factors:</strong> " + ", ".join(lr.high_vif_factors) + "</div>" if lr.high_vif_factors else ""}
</div>
<div class="card">{_embed_img("leakage_summary")}</div>

<div class="section-divider"></div>

<h2>8. Parameter Stability</h2>
<div class="card">
  <p>
    <strong>Stable:</strong>
    {'<span class="badge badge-ok">YES</span>' if report.parameter_stability.is_stable else '<span class="badge badge-warn">NO</span>'}
    &nbsp;&nbsp;
    <strong>Breakpoints detected:</strong> {len(report.parameter_stability.breakpoints)}
    {"&nbsp;&nbsp;<strong>Breakpoint indices:</strong> " + str(report.parameter_stability.breakpoints) if report.parameter_stability.breakpoints else ""}
  </p>
</div>

<div class="section-divider"></div>

<h2>9. Signal Drift</h2>
<div class="card">
<pre>{json.dumps(report.drift_summary, indent=2, default=str)}</pre>
</div>

<div class="section-divider"></div>

<footer style="color:#555; font-size:0.75rem; margin-top:32px; text-align:center;">
  srfm-lab Reconciliation Pipeline v1.0.0 &nbsp;|&nbsp;
  {report.generated_at}
</footer>
</body>
</html>
"""

    path.write_text(html, encoding="utf-8")
    log.info("HTML report written to %s", path)
    return path


# ── Console output ────────────────────────────────────────────────────────────


def to_console(report: ReconciliationReport) -> None:
    """
    Print a colour-coded summary to the terminal using `rich`.
    Falls back to plain print if rich is unavailable.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box as rbox
        _rich = True
    except ImportError:
        _rich = False

    grade = report.overall_grade()

    if not _rich:
        # Fallback plain output
        print("\n" + "=" * 70)
        print(" srfm-lab Live vs Backtest Reconciliation Report")
        print(f" Generated: {report.generated_at}")
        print(f" Grade: {grade}")
        print("=" * 70)
        _print_metrics_plain(report.live_metrics, "LIVE")
        _print_metrics_plain(report.bt_metrics, "BACKTEST")
        return

    console = Console()
    grade_style = {
        "A": "bold green", "B": "bold bright_green", "C": "bold yellow",
        "D": "bold red", "F": "bold bright_red",
    }.get(grade, "bold white")

    console.print(Panel(
        f"[bold cyan]srfm-lab Live vs Backtest Reconciliation[/]\n"
        f"[dim]Generated: {report.generated_at}[/]\n"
        f"[{grade_style}]Overall Grade: {grade}[/]",
        border_style="bright_blue",
    ))

    # Performance comparison table
    perf_table = Table(title="Performance Comparison", box=rbox.ROUNDED, show_header=True)
    perf_table.add_column("Metric", style="bold cyan", no_wrap=True)
    perf_table.add_column("Live", style="bright_blue", justify="right")
    perf_table.add_column("Backtest", style="bright_yellow", justify="right")

    live_rows = _metrics_to_html_rows(report.live_metrics)
    bt_rows = _metrics_to_html_rows(report.bt_metrics)
    for (label, lv), (_, bv) in zip(live_rows, bt_rows):
        perf_table.add_row(label, lv, bv)

    console.print(perf_table)

    # Regime comparison
    if not report.regime_comparison.empty:
        reg_table = Table(title="Regime Comparison", box=rbox.SIMPLE, show_header=True)
        reg_table.add_column("Regime", style="bold")
        for col in report.regime_comparison.columns:
            reg_table.add_column(col, justify="right")
        for idx, row in report.regime_comparison.iterrows():
            def _fmt(v: Any) -> str:
                try:
                    return f"{float(v):.2f}" if not math.isnan(float(v)) else "—"
                except (TypeError, ValueError):
                    return str(v)
            reg_table.add_row(str(idx), *[_fmt(v) for v in row])
        console.print(reg_table)

    # Leakage summary
    lr = report.leakage_report
    leak_style = "bold red" if lr.lookahead_suspected else "bold green"
    console.print(Panel(
        f"[{leak_style}]Lookahead: {'SUSPECTED ⚠' if lr.lookahead_suspected else 'OK ✓'}[/]\n"
        f"Leakage Score: [bold]{lr.leakage_score:.3f}[/]  "
        f"DSR: [bold]{lr.deflated_sharpe:.3f}[/]  "
        f"Overfit Prob: [bold]{lr.overfitting_probability:.1%}[/]\n"
        f"Purged: {lr.n_purged}/{lr.n_purged + lr.n_remaining} ({lr.purge_fraction:.1%})\n"
        f"NW-Adj Sharpe: [bold]{lr.ac_adjusted_sharpe:.2f}[/]  "
        f"AC Inflation: [bold]{lr.autocorrelation_score:.3f}[/]",
        title="[bold yellow]Leakage & Overfitting Audit[/]",
        border_style="yellow",
    ))

    # Attribution
    ar = report.attribution_report
    if ar.n_trades > 0:
        attr_table = Table(title="PnL Attribution", box=rbox.SIMPLE)
        attr_table.add_column("Effect", style="bold cyan")
        attr_table.add_column("Value ($)", justify="right")
        attr_table.add_column("% of Total", justify="right")
        for _, row in ar.effect_table().reset_index().iterrows():
            pct = f"{row.get('Pct_of_Total', float('nan')):.1f}%" \
                  if not math.isnan(float(row.get("Pct_of_Total", float("nan")))) else "—"
            val = f"${float(row.get('Value', 0)):,.0f}"
            attr_table.add_row(str(row["Effect"]), val, pct)
        console.print(attr_table)

    # Slippage
    console.print(Panel(
        f"Avg Slippage: [bold]{report.slippage_summary.get('avg_cost_per_trade_bps', float('nan')):.1f} bps[/]\n"
        f"Total Cost: [bold]${report.slippage_summary.get('total_cost_usd', 0):,.0f}[/]\n"
        f"Annualised Drag: [bold]{report.slippage_summary.get('annualised_drag_bps', float('nan')):.0f} bps[/]",
        title="[bold magenta]Slippage Summary[/]",
        border_style="magenta",
    ))

    # Stability
    stab_style = "green" if report.parameter_stability.is_stable else "red"
    bp_count = len(report.parameter_stability.breakpoints)
    console.print(Panel(
        f"[{stab_style}]Parameter Stability: {'STABLE ✓' if report.parameter_stability.is_stable else 'UNSTABLE ⚠'}[/]\n"
        f"Breakpoints detected: [bold]{bp_count}[/]",
        title="[bold cyan]Parameter Stability[/]",
        border_style="cyan",
    ))

    console.print(f"\n[dim]Output directory: {report.output_dir}[/]")
    console.print(f"[dim]HTML report: {report.plot_paths.get('html_report', 'N/A')}[/]\n")


def _print_metrics_plain(m: PerformanceMetrics, label: str) -> None:
    """Plain-text fallback metrics printer."""
    print(f"\n  [{label}]")
    for k, v in _metrics_to_html_rows(m):
        print(f"    {k:<25} {v}")


# ── Main orchestrator ─────────────────────────────────────────────────────────


def generate_full_report(
    live_trades: list | pd.DataFrame,
    bt_trades: list | pd.DataFrame,
    output_dir: str | Path = "research/reconciliation/output",
    dpi: int = 150,
    annual_factor: float = 252.0,
    suppress_plots: bool = False,
) -> ReconciliationReport:
    """
    Orchestrate all reconciliation sub-analyses and produce a full report.

    Parameters
    ----------
    live_trades : list[TradeRecord] | pd.DataFrame
    bt_trades : list[TradeRecord] | pd.DataFrame
    output_dir : str | Path
        Directory for all output artefacts (charts, HTML, JSON).
    dpi : int
        Chart resolution.
    annual_factor : float
        Annualisation factor for Sharpe etc.
    suppress_plots : bool
        If True, skip matplotlib chart generation (faster for tests).

    Returns
    -------
    ReconciliationReport
    """
    from research.reconciliation.loader import TradeRecord

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrames
    if isinstance(live_trades, list):
        live_df = _records_to_df(live_trades)
    else:
        live_df = live_trades.copy()

    if isinstance(bt_trades, list):
        bt_df = _records_to_df(bt_trades)
    else:
        bt_df = bt_trades.copy()

    log.info("Generating reconciliation report: %d live, %d bt trades.",
             len(live_df), len(bt_df))

    # Merge
    merged = merge_live_backtest(live_df, bt_df)

    # Performance metrics
    live_pnl_col = "pnl" if "pnl" in live_df.columns else "live_pnl"
    bt_pnl_col = "pnl" if "pnl" in bt_df.columns else "bt_pnl"
    live_metrics = _compute_metrics(live_df, "live", live_pnl_col, annual_factor)
    bt_metrics = _compute_metrics(bt_df, "backtest", bt_pnl_col, annual_factor)

    # Regime comparison
    regime_comparison = _regime_comparison_table(live_df, bt_df)

    # Slippage
    slip_analyzer = SlippageAnalyzer()
    fill_report = slip_analyzer.analyze_fill_quality(merged)
    slippage_summary = slip_analyzer.compute_turnover_cost(merged, annual_factor=annual_factor)

    # Attribution
    attrib_engine = PnLAttributionEngine()
    attribution_report = attrib_engine.attribute_pnl(
        live_df if not live_df.empty else merged,
        benchmark_trades=bt_df,
    )

    # Leakage
    leakage_auditor = DataLeakageAuditor()
    leakage_report = leakage_auditor.audit(live_df, bt_df)

    # Parameter stability
    drift_detector = SignalDriftDetector()
    stability_result = drift_detector.parameter_stability_test(
        bt_df if not bt_df.empty else merged,
        window=min(500, max(20, len(bt_df) // 2)),
    )

    # Signal drift summary
    drift_summary: dict[str, Any] = {}
    for signal_col in ("delta_score", "tf_score", "ensemble_signal"):
        res = drift_detector.compare_signal_distributions(live_df, bt_df, signal_col)
        drift_summary[signal_col] = res

    # Plots
    plot_paths: dict[str, str] = {}

    if not suppress_plots:
        # Equity curves
        eq_path = output_dir / "equity_curves.png"
        try:
            _plot_equity_curves(live_df, bt_df, eq_path, dpi=dpi)
            plot_paths["equity_curves"] = str(eq_path)
        except Exception as exc:
            log.warning("equity_curves plot failed: %s", exc)

        # PnL histogram
        hist_path = output_dir / "pnl_histogram.png"
        try:
            _plot_pnl_histogram(live_df, bt_df, hist_path, dpi=dpi)
            plot_paths["pnl_histogram"] = str(hist_path)
        except Exception as exc:
            log.warning("pnl_histogram plot failed: %s", exc)

        # Regime bar
        reg_path = output_dir / "regime_comparison.png"
        try:
            _plot_regime_bar(regime_comparison, reg_path, dpi=dpi)
            plot_paths["regime_comparison"] = str(reg_path)
        except Exception as exc:
            log.warning("regime_comparison plot failed: %s", exc)

        # Slippage distribution
        slip_path = output_dir / "slippage_distribution.png"
        try:
            slip_analyzer.plot_slippage_distribution(merged, slip_path, dpi=dpi)
            plot_paths["slippage_distribution"] = str(slip_path)
        except Exception as exc:
            log.warning("slippage_distribution plot failed: %s", exc)

        # Attribution waterfall
        attr_path = output_dir / "attribution_waterfall.png"
        try:
            attrib_engine.plot_attribution_waterfall(attribution_report, attr_path, dpi=dpi)
            plot_paths["attribution_waterfall"] = str(attr_path)
        except Exception as exc:
            log.warning("attribution_waterfall plot failed: %s", exc)

        # Rolling IC
        ic_path = output_dir / "rolling_ic.png"
        try:
            attrib_engine.plot_rolling_ic(attribution_report, ic_path, dpi=dpi)
            plot_paths["rolling_ic"] = str(ic_path)
        except Exception as exc:
            log.warning("rolling_ic plot failed: %s", exc)

        # Regime attribution
        ra_path = output_dir / "regime_attribution.png"
        try:
            attrib_engine.plot_regime_attribution(attribution_report, ra_path, dpi=dpi)
            plot_paths["regime_attribution"] = str(ra_path)
        except Exception as exc:
            log.warning("regime_attribution plot failed: %s", exc)

        # Leakage summary
        leak_path = output_dir / "leakage_summary.png"
        try:
            leakage_auditor.plot_leakage_summary(leakage_report, leak_path, dpi=dpi)
            plot_paths["leakage_summary"] = str(leak_path)
        except Exception as exc:
            log.warning("leakage_summary plot failed: %s", exc)

    # Build report
    report = ReconciliationReport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        live_metrics=live_metrics,
        bt_metrics=bt_metrics,
        regime_comparison=regime_comparison,
        merged_trades=merged,
        fill_report=fill_report,
        attribution_report=attribution_report,
        leakage_report=leakage_report,
        parameter_stability=stability_result,
        slippage_summary=slippage_summary,
        drift_summary=drift_summary,
        plot_paths=plot_paths,
        output_dir=str(output_dir),
        metadata={
            "n_live": len(live_df),
            "n_bt": len(bt_df),
            "annual_factor": annual_factor,
        },
    )

    # Write HTML
    html_path = output_dir / "reconciliation_report.html"
    try:
        to_html(report, html_path)
        plot_paths["html_report"] = str(html_path)
        report.plot_paths["html_report"] = str(html_path)
    except Exception as exc:
        log.warning("HTML report generation failed: %s", exc)

    # Write JSON summary
    json_path = output_dir / "reconciliation_summary.json"
    try:
        summary = {
            "generated_at": report.generated_at,
            "grade": report.overall_grade(),
            "live": live_metrics.to_dict(),
            "backtest": bt_metrics.to_dict(),
            "slippage": slippage_summary,
            "leakage": leakage_report.summary_dict(),
            "attribution": attribution_report.summary_dict(),
            "parameter_stability": stability_result.to_dict(),
            "drift": drift_summary,
        }
        json_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        plot_paths["json_summary"] = str(json_path)
    except Exception as exc:
        log.warning("JSON summary failed: %s", exc)

    log.info("Report complete. Grade: %s", report.overall_grade())
    return report
