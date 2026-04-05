"""
experiment-tracker/reporter.py
================================
Experiment reporting and analysis.

Generates human-readable summaries, parameter importance rankings, learning
curves, regime-split breakdowns, and top-N leaderboards for sets of
experiments stored in ``idea_engine.db``.

All formatting targets GitHub-Flavored Markdown so outputs can be dropped
directly into pull-request descriptions, Notion pages, or the IAE dashboard.

Key design decisions
--------------------
* Reports are **pure strings** — no file I/O is performed here.
  Callers write them wherever appropriate (DB artifact, file, dashboard).
* Parameter importance uses Pearson correlation between param values and
  the target metric — simple, interpretable, and requires no external libs.
* Learning curves use a linear regression over time to quantify whether
  metric quality is improving (positive slope) or degrading.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB = Path(__file__).resolve().parents[1] / "idea_engine.db"

# ---------------------------------------------------------------------------
# ExperimentReporter
# ---------------------------------------------------------------------------


class ExperimentReporter:
    """
    Generates Markdown reports and analytical summaries for experiment sets.

    Parameters
    ----------
    db_path : str | Path
        Path to ``idea_engine.db``.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB) -> None:
        self._db_path = Path(db_path)
        self._conn = self._open_connection()
        self._ensure_schema()
        logger.info("ExperimentReporter connected to %s.", self._db_path)

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        sql_path = Path(__file__).parent / "schema_extension.sql"
        if sql_path.exists():
            self._conn.executescript(sql_path.read_text(encoding="utf-8"))
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ExperimentReporter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary_table(
        self,
        experiments: list[Any],   # list of ExperimentRecord or dicts
        *,
        param_cols: list[str] | None = None,
        metric_cols: list[str] | None = None,
    ) -> str:
        """
        Render a Markdown table showing selected params and metrics for a
        list of experiments.

        Parameters
        ----------
        experiments : list of ExperimentRecord (or flat dicts from to_flat_dict())
        param_cols  : list[str] | None
            Param keys to include (without the ``param_`` prefix).
            If None, all params found in the first experiment are used.
        metric_cols : list[str] | None
            Metric keys to include (without the ``metric_`` prefix).
            Defaults to: sharpe, max_dd, calmar, win_rate, total_return.

        Returns
        -------
        str — Markdown-formatted table.
        """
        if not experiments:
            return "_No experiments to display._\n"

        # Normalise to flat dicts
        rows: list[dict[str, Any]] = []
        for exp in experiments:
            if hasattr(exp, "to_flat_dict"):
                rows.append(exp.to_flat_dict())
            elif isinstance(exp, dict):
                rows.append(exp)
            else:
                rows.append({"id": str(exp)})

        df = pd.DataFrame(rows)

        # Determine columns to show
        if metric_cols is None:
            metric_cols = ["sharpe", "max_dd", "calmar", "win_rate", "total_return"]
        metric_display = [c for c in (f"metric_{m}" for m in metric_cols) if c in df.columns]

        if param_cols is None:
            param_display = sorted(c for c in df.columns if c.startswith("param_"))
        else:
            param_display = [f"param_{p}" for p in param_cols if f"param_{p}" in df.columns]

        core_cols = ["id", "name", "status", "duration_seconds"]
        display_cols = (
            [c for c in core_cols if c in df.columns]
            + param_display
            + metric_display
        )

        df = df[display_cols].copy()
        # Round floats
        for col in df.select_dtypes(include="float").columns:
            df[col] = df[col].round(4)

        # Rename columns for display
        rename = {"duration_seconds": "duration_s"}
        rename.update({c: c.replace("param_", "").replace("metric_", "") for c in display_cols})
        df.rename(columns=rename, inplace=True)

        return _df_to_markdown(df)

    # ------------------------------------------------------------------
    # Parameter importance
    # ------------------------------------------------------------------

    def parameter_importance(
        self,
        experiments: list[Any],
        target_metric: str = "sharpe",
    ) -> pd.DataFrame:
        """
        Compute the Pearson correlation between each numeric parameter and
        the target metric across a set of experiments.

        Parameters that vary across experiments and correlate strongly with
        the target metric are the most "important" levers to optimise.

        Parameters
        ----------
        experiments   : list of ExperimentRecord (or flat dicts)
        target_metric : str — metric key to correlate against (default 'sharpe')

        Returns
        -------
        pd.DataFrame with columns:
            parameter, correlation, abs_correlation, n_experiments
        sorted by abs_correlation descending.
        """
        if not experiments:
            return pd.DataFrame(
                columns=["parameter", "correlation", "abs_correlation", "n_experiments"]
            )

        rows = [
            exp.to_flat_dict() if hasattr(exp, "to_flat_dict") else exp
            for exp in experiments
        ]
        df = pd.DataFrame(rows)

        target_col = f"metric_{target_metric}"
        if target_col not in df.columns:
            logger.warning("Metric '%s' not found in experiments.", target_metric)
            return pd.DataFrame(
                columns=["parameter", "correlation", "abs_correlation", "n_experiments"]
            )

        param_cols = [c for c in df.columns if c.startswith("param_")]
        results: list[dict[str, Any]] = []

        for col in param_cols:
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                valid = df[[col, target_col]].copy()
                valid[col] = numeric
                valid = valid.dropna()
                if len(valid) < 3 or valid[col].nunique() < 2:
                    continue
                corr = float(valid[col].corr(valid[target_col]))
                results.append(
                    {
                        "parameter": col.replace("param_", ""),
                        "correlation": round(corr, 4),
                        "abs_correlation": round(abs(corr), 4),
                        "n_experiments": len(valid),
                    }
                )
            except Exception:
                continue

        if not results:
            return pd.DataFrame(
                columns=["parameter", "correlation", "abs_correlation", "n_experiments"]
            )

        return (
            pd.DataFrame(results)
            .sort_values("abs_correlation", ascending=False)
            .reset_index(drop=True)
        )

    def parameter_importance_markdown(
        self,
        experiments: list[Any],
        target_metric: str = "sharpe",
    ) -> str:
        """
        Render the parameter importance table as Markdown.

        Parameters
        ----------
        experiments   : list of ExperimentRecord
        target_metric : str

        Returns
        -------
        str — Markdown table.
        """
        df = self.parameter_importance(experiments, target_metric)
        if df.empty:
            return "_No parameter importance data available._\n"
        header = f"## Parameter Importance (target: `{target_metric}`)\n\n"
        return header + _df_to_markdown(df)

    # ------------------------------------------------------------------
    # Learning curve
    # ------------------------------------------------------------------

    def learning_curve(
        self,
        experiments_by_date: list[Any],
        target_metric: str = "sharpe",
    ) -> pd.DataFrame:
        """
        Analyse whether experiment quality (measured by ``target_metric``) is
        improving over time.

        Fits a simple linear regression of ``metric_value ~ experiment_index``
        to estimate the learning trend.

        Parameters
        ----------
        experiments_by_date : list of ExperimentRecord sorted by started_at
        target_metric       : str

        Returns
        -------
        pd.DataFrame with columns:
            experiment_id, started_at, metric_value, trend_value, residual
        Plus a ``slope`` and ``r_squared`` attribute accessible via the
        DataFrame's ``attrs`` dict.
        """
        rows_raw = [
            exp.to_flat_dict() if hasattr(exp, "to_flat_dict") else exp
            for exp in experiments_by_date
        ]
        df = pd.DataFrame(rows_raw)
        target_col = f"metric_{target_metric}"

        if target_col not in df.columns or "started_at" not in df.columns:
            return pd.DataFrame(
                columns=["experiment_id", "started_at", "metric_value",
                         "trend_value", "residual"]
            )

        df = df[["id", "started_at", target_col]].dropna().copy()
        df.rename(columns={"id": "experiment_id", target_col: "metric_value"}, inplace=True)
        df.sort_values("started_at", inplace=True)
        df.reset_index(drop=True, inplace=True)

        x = np.arange(len(df), dtype=float)
        y = df["metric_value"].astype(float).values

        slope, intercept, r_squared = _linear_regression(x, y)
        df["trend_value"] = intercept + slope * x
        df["residual"] = y - df["trend_value"]

        df.attrs["slope"] = slope
        df.attrs["r_squared"] = r_squared
        df.attrs["target_metric"] = target_metric

        return df

    def learning_curve_markdown(
        self,
        experiments_by_date: list[Any],
        target_metric: str = "sharpe",
    ) -> str:
        """
        Render a Markdown summary of the learning curve analysis.

        Parameters
        ----------
        experiments_by_date : list
        target_metric       : str

        Returns
        -------
        str — Markdown report.
        """
        df = self.learning_curve(experiments_by_date, target_metric)
        if df.empty:
            return "_Not enough data for learning curve analysis._\n"

        slope = df.attrs.get("slope", 0.0)
        r_sq = df.attrs.get("r_squared", 0.0)
        direction = "improving" if slope > 0 else ("stable" if abs(slope) < 1e-4 else "degrading")

        lines = [
            f"## Learning Curve — `{target_metric}`",
            "",
            f"**Trend**: {direction}  ",
            f"**Slope**: {slope:+.4f} per experiment  ",
            f"**R²**: {r_sq:.4f}",
            "",
        ]
        lines.append(_df_to_markdown(df[["experiment_id", "started_at", "metric_value", "trend_value"]].round(4)))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Regime breakdown
    # ------------------------------------------------------------------

    def regime_breakdown(
        self,
        experiments: list[Any],
        regime_metric_prefix: str = "regime_",
    ) -> pd.DataFrame:
        """
        Summarise performance split by market regime at the time of each
        experiment test.

        Assumes experiments have metrics logged with names like
        ``regime_bull_sharpe``, ``regime_bear_sharpe``, etc.

        Parameters
        ----------
        experiments          : list of ExperimentRecord
        regime_metric_prefix : str — prefix identifying regime-split metrics

        Returns
        -------
        pd.DataFrame with columns:
            regime, mean_sharpe, std_sharpe, n_experiments, best_sharpe
        """
        rows_raw = [
            exp.to_flat_dict() if hasattr(exp, "to_flat_dict") else exp
            for exp in experiments
        ]
        df = pd.DataFrame(rows_raw)

        regime_cols = [c for c in df.columns if c.startswith(f"metric_{regime_metric_prefix}")]
        if not regime_cols:
            return pd.DataFrame(
                columns=["regime", "mean_sharpe", "std_sharpe",
                         "n_experiments", "best_sharpe"]
            )

        records: list[dict[str, Any]] = []
        for col in regime_cols:
            regime_name = col.replace(f"metric_{regime_metric_prefix}", "")
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if values.empty:
                continue
            records.append(
                {
                    "regime": regime_name,
                    "mean_sharpe": round(float(values.mean()), 4),
                    "std_sharpe": round(float(values.std(ddof=1)), 4),
                    "n_experiments": int(values.count()),
                    "best_sharpe": round(float(values.max()), 4),
                }
            )

        return pd.DataFrame(records).sort_values("mean_sharpe", ascending=False).reset_index(drop=True)

    def regime_breakdown_markdown(self, experiments: list[Any]) -> str:
        """Render regime breakdown as Markdown."""
        df = self.regime_breakdown(experiments)
        if df.empty:
            return "_No regime-split metrics found in these experiments._\n"
        return "## Regime Breakdown\n\n" + _df_to_markdown(df)

    # ------------------------------------------------------------------
    # Top N leaderboard
    # ------------------------------------------------------------------

    def top_n_experiments(
        self,
        n: int = 10,
        metric: str = "sharpe",
        higher_is_better: bool = True,
        *,
        status: str = "completed",
    ) -> str:
        """
        Return a formatted Markdown leaderboard of the top-N experiments
        ranked by the specified metric.

        Parameters
        ----------
        n                : int   — number of experiments to return
        metric           : str   — metric key to rank by
        higher_is_better : bool
        status           : str   — filter by experiment status

        Returns
        -------
        str — Markdown-formatted leaderboard.
        """
        order = "DESC" if higher_is_better else "ASC"
        rows = self._conn.execute(
            f"""
            SELECT e.id, e.name, e.status, e.started_at,
                   e.hypothesis_id, e.duration_seconds,
                   m.value AS metric_value
            FROM experiments e
            JOIN experiment_metrics m ON m.experiment_id = e.id
            WHERE m.key = ?
              AND e.status = ?
              AND (e.id, m.key, m.logged_at) IN (
                  SELECT experiment_id, key, MAX(logged_at)
                  FROM experiment_metrics
                  GROUP BY experiment_id, key
              )
            ORDER BY m.value {order}
            LIMIT ?
            """,
            (metric, status, n),
        ).fetchall()

        if not rows:
            return f"_No {status!r} experiments with metric `{metric}` found._\n"

        lines: list[str] = [
            f"## Top {n} Experiments — `{metric}`",
            "",
            f"| Rank | ID | Name | {metric.capitalize()} | Hypothesis | Duration | Started |",
            f"|------|----|------|{'─' * len(metric)}|------------|----------|---------|",
        ]
        for rank, row in enumerate(rows, start=1):
            dur = _fmt_duration(row["duration_seconds"])
            lines.append(
                f"| {rank} | {row['id']} | {row['name']} "
                f"| {row['metric_value']:.4f} | {row['hypothesis_id'] or '—'} "
                f"| {dur} | {row['started_at'][:10]} |"
            )
        lines.append("")
        lines.append(
            f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(
        self,
        experiments: list[Any],
        *,
        top_n: int = 10,
        target_metric: str = "sharpe",
    ) -> str:
        """
        Assemble a complete Markdown report covering all sections:
        summary table, parameter importance, learning curve, regime
        breakdown, and top-N leaderboard.

        Parameters
        ----------
        experiments   : list of ExperimentRecord
        top_n         : int — passed to top_n_experiments
        target_metric : str — primary metric for rankings

        Returns
        -------
        str — full Markdown report.
        """
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        sections = [
            f"# Experiment Report",
            f"*Generated {ts} — {len(experiments)} experiment(s)*",
            "",
            self.summary_table(experiments),
            "",
            self.parameter_importance_markdown(experiments, target_metric),
            "",
            self.learning_curve_markdown(experiments, target_metric),
            "",
            self.regime_breakdown_markdown(experiments),
            "",
            self.top_n_experiments(top_n, target_metric),
        ]
        return "\n".join(sections)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _df_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a GitHub-Flavored Markdown table string."""
    if df.empty:
        return "_Empty result set._\n"

    headers = list(df.columns)
    sep = ["-" * max(len(h), 5) for h in headers]

    lines: list[str] = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float) and math.isnan(val):
                cells.append("—")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def _linear_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """
    Fit OLS y = intercept + slope·x.

    Returns
    -------
    (slope, intercept, r_squared)
    """
    n = len(x)
    if n < 2:
        return 0.0, float(y[0]) if n == 1 else 0.0, 0.0

    x_mean = float(x.mean())
    y_mean = float(y.mean())
    ss_xy = float(((x - x_mean) * (y - y_mean)).sum())
    ss_xx = float(((x - x_mean) ** 2).sum())

    if ss_xx == 0:
        return 0.0, y_mean, 0.0

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    y_pred = intercept + slope * x
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    return slope, intercept, max(0.0, r_squared)


def _fmt_duration(seconds: float | None) -> str:
    """Format seconds as a human-readable duration string."""
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"
