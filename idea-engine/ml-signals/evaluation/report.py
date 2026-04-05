"""
evaluation/report.py
=====================
Generate model performance reports as text tables and summaries.

Financial rationale
-------------------
Model monitoring in systematic trading requires fast, readable output
that can be scanned in a terminal or logged to a file.  We avoid
external plotting libraries (matplotlib, plotext) so that this module
works in any environment (remote server, CI pipeline, cron job) without
a display.

Reports cover:
1. Feature importance tables: which features drive each model, sorted
   by importance, with a simple ASCII bar chart.
2. Rolling IC chart: ASCII sparkline of IC over time.
3. Out-of-sample vs in-sample IC ratio: values close to 1.0 indicate
   no backtest overfitting.
4. Walk-forward result table: per-step IC, ICIR, Sharpe.
5. Cross-model comparison: side-by-side metrics for LSTM, Transformer,
   XGBoost, and Ensemble.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models.base import MLSignal, SignalMetrics
from .backtester import BacktestResult


# ---------------------------------------------------------------------------
# ASCII helpers
# ---------------------------------------------------------------------------

def _bar(value: float, max_val: float, width: int = 30) -> str:
    """Render a normalised horizontal ASCII bar."""
    if max_val < 1e-9:
        return " " * width
    filled = int(round(width * min(1.0, value / max_val)))
    return "#" * filled + "." * (width - filled)


def _sparkline(values: np.ndarray, width: int = 60) -> str:
    """Render a fixed-width sparkline using block characters."""
    blocks = " .,:;+=xX#"
    if len(values) == 0:
        return ""
    # Down-sample to width
    if len(values) > width:
        idx = np.linspace(0, len(values) - 1, width, dtype=int)
        vals = values[idx]
    else:
        vals = values
    vmin, vmax = vals.min(), vals.max()
    if abs(vmax - vmin) < 1e-9:
        return blocks[4] * len(vals)
    normalised = (vals - vmin) / (vmax - vmin)
    return "".join(blocks[int(round(v * (len(blocks) - 1)))] for v in normalised)


def _table(
    headers: List[str],
    rows: List[List],
    col_widths: Optional[List[int]] = None,
) -> str:
    """Render a simple text table."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]
    sep    = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    h_row  = "|" + "|".join(f" {str(h):<{w}} " for h, w in zip(headers, col_widths)) + "|"
    lines  = [sep, h_row, sep]
    for row in rows:
        r_line = "|" + "|".join(f" {str(v):<{w}} " for v, w in zip(row, col_widths)) + "|"
        lines.append(r_line)
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class Report:
    """Generate model performance reports as plain text.

    Usage::

        rpt = Report()
        print(rpt.feature_importance(model))
        print(rpt.rolling_ic(predictions, returns))
        print(rpt.comparison_table(results_dict))
    """

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(
        self,
        model:      MLSignal,
        top_n:      int = 20,
        bar_width:  int = 30,
    ) -> str:
        """Generate feature importance table with ASCII bars.

        Parameters
        ----------
        model : MLSignal  must be fitted
        top_n : int       number of features to show
        bar_width : int   width of the importance bar

        Returns
        -------
        str  multi-line text table
        """
        if not model.is_fitted:
            return f"[Report] {model.name} is not fitted."

        imp = model.feature_importance()
        if not imp:
            return f"[Report] {model.name} returned empty feature importance."

        # Sort by importance descending
        items = sorted(imp.items(), key=lambda x: -x[1])[:top_n]
        max_imp = items[0][1] if items else 1.0

        title   = f"\n{'=' * 60}\nFeature Importance: {model.name}\n{'=' * 60}\n"
        header  = f"{'Feature':<30}  {'Importance':>10}  {'Bar'}"
        divider = "-" * (30 + 12 + bar_width + 4)
        lines   = [title, header, divider]
        for feat, val in items:
            bar = _bar(val, max_imp, bar_width)
            lines.append(f"{feat:<30}  {val:>10.4f}  {bar}")
        lines.append(divider)
        lines.append(f"Total features: {len(imp)}, showing top {len(items)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Rolling IC chart
    # ------------------------------------------------------------------

    def rolling_ic(
        self,
        predictions: np.ndarray,
        returns:     np.ndarray,
        window:      int = 20,
        width:       int = 60,
    ) -> str:
        """Generate rolling IC sparkline and summary statistics.

        Parameters
        ----------
        predictions : np.ndarray
        returns     : np.ndarray
        window      : int  rolling IC window
        width       : int  sparkline character width

        Returns
        -------
        str  text chart + summary
        """
        from scipy.stats import spearmanr

        n = min(len(predictions), len(returns))
        ics = []
        for i in range(window, n):
            p = predictions[i - window : i]
            r = returns[i - window : i]
            if np.std(p) > 1e-9 and np.std(r) > 1e-9:
                ic, _ = spearmanr(p, r)
                if not np.isnan(ic):
                    ics.append(ic)

        ics_arr = np.array(ics)
        spark   = _sparkline(ics_arr, width)
        mean_ic = float(np.mean(ics_arr)) if len(ics_arr) else 0.0
        std_ic  = float(np.std(ics_arr)) if len(ics_arr) else 0.0
        icir    = mean_ic / (std_ic + 1e-9)
        pct_pos = float(np.mean(ics_arr > 0)) if len(ics_arr) else 0.0

        lines = [
            f"\n{'=' * 60}",
            f"Rolling IC (window={window})",
            f"{'=' * 60}",
            spark,
            f"min={ics_arr.min():.3f}  max={ics_arr.max():.3f}" if len(ics_arr) else "",
            f"",
            f"Mean IC    : {mean_ic:+.4f}",
            f"Std IC     : {std_ic:.4f}",
            f"ICIR       : {icir:+.4f}",
            f"% Positive : {pct_pos:.1%}",
            f"N windows  : {len(ics_arr)}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Backtest comparison table
    # ------------------------------------------------------------------

    def comparison_table(
        self, results: Dict[str, BacktestResult]
    ) -> str:
        """Side-by-side comparison of multiple strategy backtest results.

        Parameters
        ----------
        results : dict  {label: BacktestResult}
        """
        if not results:
            return "[Report] No results to compare."

        labels  = list(results.keys())
        metrics = ["ic", "icir", "sharpe", "calmar", "max_drawdown",
                   "annual_return", "hit_rate"]
        header  = ["Metric"] + labels
        col_w   = [max(len(m), 14) for m in [" " * 16] + labels]

        rows = []
        for m in metrics:
            row = [m]
            for lbl in labels:
                val = getattr(results[lbl], m, None)
                if val is None:
                    row.append("N/A")
                elif m in ("max_drawdown", "annual_return", "hit_rate"):
                    row.append(f"{val:.2%}")
                else:
                    row.append(f"{val:.4f}")
            rows.append(row)

        title = f"\n{'=' * 60}\nStrategy Comparison\n{'=' * 60}\n"
        return title + _table(header, rows)

    # ------------------------------------------------------------------
    # Walk-forward results table
    # ------------------------------------------------------------------

    def walkforward_table(self, results: list) -> str:
        """Format walk-forward training results as a text table.

        Parameters
        ----------
        results : list[TrainingResult]  from MLTrainer.train_walkforward()
        """
        if not results:
            return "[Report] No walk-forward results."

        headers = ["Step", "Train End", "Val Start", "Val End",
                   "IC", "ICIR", "Sec"]
        rows = []
        for i, r in enumerate(results):
            rows.append([
                str(i + 1),
                str(r.train_end)[:10],
                str(r.val_start)[:10],
                str(r.val_end)[:10],
                f"{r.val_ic:+.4f}",
                f"{r.val_icir:+.4f}",
                f"{r.train_sec:.1f}",
            ])
        title = f"\n{'=' * 60}\nWalk-Forward Training Results\n{'=' * 60}\n"
        summary_rows = [r for r in results if hasattr(r, "val_ic")]
        mean_ic  = np.mean([r.val_ic for r in summary_rows]) if summary_rows else 0.0
        mean_icir = np.mean([r.val_icir for r in summary_rows]) if summary_rows else 0.0
        summary = (f"\nSummary: mean_IC={mean_ic:+.4f}  "
                   f"mean_ICIR={mean_icir:+.4f}  n_steps={len(results)}")
        return title + _table(headers, rows) + summary

    # ------------------------------------------------------------------
    # IS / OOS ratio
    # ------------------------------------------------------------------

    def is_oos_report(
        self,
        is_ic:  float,
        oos_ic: float,
        n_is:   int,
        n_oos:  int,
    ) -> str:
        """Report in-sample vs out-of-sample IC ratio.

        A ratio of OOS/IS close to 1.0 indicates no overfitting.
        Ratios < 0.5 are a red flag (possible backtest fitting).
        """
        ratio = oos_ic / (is_ic + 1e-9)
        flag  = ("[OK] Healthy" if ratio >= 0.7 else
                 "[WARN] Moderate overfit" if ratio >= 0.4 else
                 "[FAIL] Severe overfit")
        lines = [
            f"\n{'=' * 60}",
            f"In-Sample vs Out-of-Sample IC",
            f"{'=' * 60}",
            f"  IS  IC  ({n_is:>5} bars): {is_ic:+.4f}",
            f"  OOS IC  ({n_oos:>5} bars): {oos_ic:+.4f}",
            f"  OOS/IS ratio         : {ratio:.3f}",
            f"  Assessment           : {flag}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Full model report
    # ------------------------------------------------------------------

    def full_report(
        self,
        model:      MLSignal,
        predictions: np.ndarray,
        returns:    np.ndarray,
        bt_results: Optional[Dict[str, BacktestResult]] = None,
        wf_results: Optional[list] = None,
        is_ic:      float = 0.0,
        oos_ic:     float = 0.0,
    ) -> str:
        """Assemble a complete model performance report."""
        buf = io.StringIO()
        buf.write(f"\n{'#' * 60}\n")
        buf.write(f"# SRFM ML Signal Report: {model.name}\n")
        buf.write(f"{'#' * 60}\n")

        buf.write(self.feature_importance(model))
        buf.write("\n")
        buf.write(self.rolling_ic(predictions, returns))
        buf.write("\n")

        if bt_results:
            buf.write(self.comparison_table(bt_results))
            buf.write("\n")

        if wf_results:
            buf.write(self.walkforward_table(wf_results))
            buf.write("\n")

        if is_ic != 0.0 or oos_ic != 0.0:
            buf.write(self.is_oos_report(
                is_ic, oos_ic,
                n_is  = int(len(predictions) * 0.7),
                n_oos = int(len(predictions) * 0.3),
            ))

        return buf.getvalue()
