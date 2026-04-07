# research/pipeline/alpha_research_db.py
# SRFM -- SQLite database for alpha research results, live performance tracking,
# and signal lifecycle management

from __future__ import annotations

import json
import logging
import sqlite3
import textwrap
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .signal_research_pipeline import ResearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AlphaResearchDB
# ---------------------------------------------------------------------------

class AlphaResearchDB:
    """
    Persistent SQLite store for signal research results, per-date IC history,
    and live production performance tracking.

    Schema
    ------
    signals          -- one row per evaluated signal (latest result)
    ic_history       -- time series of IC values from backtest
    live_performance -- daily realized IC in production
    """

    # DDL
    _CREATE_SIGNALS = """
        CREATE TABLE IF NOT EXISTS signals (
            signal_name         TEXT    PRIMARY KEY,
            category            TEXT,
            description         TEXT,
            ic_mean             REAL,
            ic_std              REAL,
            icir                REAL,
            ic_decay_halflife   REAL,
            regime_bull         REAL,
            regime_bear         REAL,
            regime_ranging      REAL,
            turnover_daily      REAL,
            capacity_usd        REAL,
            is_significant      INTEGER,
            deflated_sharpe     REAL,
            recommendation      TEXT,
            p_value             REAL,
            n_observations      INTEGER,
            created_at          TEXT,
            updated_at          TEXT
        );
    """

    _CREATE_IC_HISTORY = """
        CREATE TABLE IF NOT EXISTS ic_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_name TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            ic_value    REAL,
            UNIQUE(signal_name, date),
            FOREIGN KEY (signal_name) REFERENCES signals(signal_name)
        );
    """

    _CREATE_LIVE_PERFORMANCE = """
        CREATE TABLE IF NOT EXISTS live_performance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_name TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            realized_ic REAL,
            UNIQUE(signal_name, date),
            FOREIGN KEY (signal_name) REFERENCES signals(signal_name)
        );
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(self._CREATE_SIGNALS)
            conn.execute(self._CREATE_IC_HISTORY)
            conn.execute(self._CREATE_LIVE_PERFORMANCE)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    # ------------------------------------------------------------------
    # Save / upsert
    # ------------------------------------------------------------------

    def save_result(self, result: ResearchResult) -> None:
        """
        Insert or replace a ResearchResult into the signals table.
        Also persists the IC time series to ic_history.
        """
        now = datetime.utcnow().isoformat()
        regime = result.regime_conditional_ic or {}

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signals (
                    signal_name, category, description, ic_mean, ic_std, icir,
                    ic_decay_halflife, regime_bull, regime_bear, regime_ranging,
                    turnover_daily, capacity_usd, is_significant, deflated_sharpe,
                    recommendation, p_value, n_observations, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(signal_name) DO UPDATE SET
                    category          = excluded.category,
                    description       = excluded.description,
                    ic_mean           = excluded.ic_mean,
                    ic_std            = excluded.ic_std,
                    icir              = excluded.icir,
                    ic_decay_halflife = excluded.ic_decay_halflife,
                    regime_bull       = excluded.regime_bull,
                    regime_bear       = excluded.regime_bear,
                    regime_ranging    = excluded.regime_ranging,
                    turnover_daily    = excluded.turnover_daily,
                    capacity_usd      = excluded.capacity_usd,
                    is_significant    = excluded.is_significant,
                    deflated_sharpe   = excluded.deflated_sharpe,
                    recommendation    = excluded.recommendation,
                    p_value           = excluded.p_value,
                    n_observations    = excluded.n_observations,
                    updated_at        = excluded.updated_at
                """,
                (
                    result.signal_name,
                    result.category,
                    result.description,
                    result.ic_mean,
                    result.ic_std,
                    result.icir,
                    result.ic_decay_halflife,
                    regime.get("bull"),
                    regime.get("bear"),
                    regime.get("ranging"),
                    result.turnover_daily,
                    result.capacity_estimate_usd,
                    int(result.is_significant),
                    result.deflated_sharpe,
                    result.recommendation,
                    result.p_value,
                    result.n_observations,
                    now,
                    now,
                ),
            )

            # Persist IC history
            if result.ic_series is not None and not result.ic_series.empty:
                rows = [
                    (result.signal_name, str(dt.date()), float(ic))
                    for dt, ic in result.ic_series.items()
                    if pd.notna(ic)
                ]
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO ic_history (signal_name, date, ic_value)
                    VALUES (?, ?, ?)
                    """,
                    rows,
                )

        logger.debug("Saved result for signal: %s", result.signal_name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_results(
        self,
        category: Optional[str] = None,
        min_icir: Optional[float] = None,
        recommendation: Optional[str] = None,
    ) -> List[ResearchResult]:
        """
        Retrieve research results with optional filtering.

        Parameters
        ----------
        category : str, optional
            Filter by signal category (e.g. "momentum", "reversal").
        min_icir : float, optional
            Minimum absolute ICIR to include.
        recommendation : str, optional
            Filter by recommendation string: "PROMOTE", "WATCH", "RETIRE".

        Returns
        -------
        List[ResearchResult]
        """
        clauses = []
        params: list = []

        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if min_icir is not None:
            clauses.append("ABS(icir) >= ?")
            params.append(min_icir)
        if recommendation is not None:
            clauses.append("recommendation = ?")
            params.append(recommendation)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM signals {where} ORDER BY icir DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_result(r) for r in rows]

    def get_best_signals(self, n: int = 10) -> List[ResearchResult]:
        """Return the top n signals by ICIR with recommendation = PROMOTE or WATCH."""
        sql = """
            SELECT * FROM signals
            WHERE recommendation IN ('PROMOTE', 'WATCH')
            ORDER BY icir DESC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (n,)).fetchall()
        return [self._row_to_result(r) for r in rows]

    def get_ic_history(self, signal_name: str) -> pd.Series:
        """Return the backtest IC time series for a given signal."""
        sql = """
            SELECT date, ic_value FROM ic_history
            WHERE signal_name = ?
            ORDER BY date
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (signal_name,)).fetchall()

        if not rows:
            return pd.Series(dtype=float, name="ic")

        dates = pd.to_datetime([r["date"] for r in rows])
        values = [r["ic_value"] for r in rows]
        return pd.Series(values, index=dates, name="ic")

    # ------------------------------------------------------------------
    # Live performance tracking
    # ------------------------------------------------------------------

    def update_live_performance(
        self,
        signal_name: str,
        date: str,
        realized_ic: float,
    ) -> None:
        """
        Record a single day's realized IC for a signal running in production.

        Parameters
        ----------
        signal_name : str
        date : str
            ISO format date string "YYYY-MM-DD".
        realized_ic : float
        """
        sql = """
            INSERT OR REPLACE INTO live_performance (signal_name, date, realized_ic)
            VALUES (?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(sql, (signal_name, date, realized_ic))

    def get_live_performance(
        self,
        signal_name: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        """Return live realized IC series for a signal."""
        clauses = ["signal_name = ?"]
        params: list = [signal_name]
        if start:
            clauses.append("date >= ?")
            params.append(start)
        if end:
            clauses.append("date <= ?")
            params.append(end)
        where = " AND ".join(clauses)
        sql = f"SELECT date, realized_ic FROM live_performance WHERE {where} ORDER BY date"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        if not rows:
            return pd.Series(dtype=float, name="realized_ic")
        dates = pd.to_datetime([r["date"] for r in rows])
        values = [r["realized_ic"] for r in rows]
        return pd.Series(values, index=dates, name="realized_ic")

    # ------------------------------------------------------------------
    # Decay in production
    # ------------------------------------------------------------------

    def compute_decay_in_production(self, signal_name: str) -> float:
        """
        Compare backtest IC mean to live IC mean to estimate performance decay.

        Returns
        -------
        float
            Ratio of live IC mean to backtest IC mean.
            1.0 = no decay; 0.5 = half the backtest IC; <0 = sign flip.
        """
        backtest_ic = self.get_ic_history(signal_name)
        live_ic = self.get_live_performance(signal_name)

        if backtest_ic.empty or live_ic.empty:
            logger.warning("compute_decay_in_production: no data for %s", signal_name)
            return float("nan")

        bt_mean = float(backtest_ic.mean())
        live_mean = float(live_ic.mean())

        if abs(bt_mean) < 1e-6:
            return float("nan")

        return live_mean / bt_mean

    # ------------------------------------------------------------------
    # Retirement candidates
    # ------------------------------------------------------------------

    def retirement_candidates(
        self,
        decay_threshold: float = 0.5,
        min_live_obs: int = 20,
    ) -> List[str]:
        """
        Identify signals whose live IC has fallen below decay_threshold * backtest IC.

        Parameters
        ----------
        decay_threshold : float
            Signals with live/backtest IC ratio below this are retirement candidates.
            Default 0.5 (live IC < 50% of backtest IC).
        min_live_obs : int
            Minimum number of live observations required to flag retirement.

        Returns
        -------
        List[str]
            Signal names recommended for retirement.
        """
        # Get all signals
        sql = "SELECT signal_name FROM signals"
        with self._connect() as conn:
            all_signals = [r["signal_name"] for r in conn.execute(sql).fetchall()]

        candidates = []
        for name in all_signals:
            live = self.get_live_performance(name)
            if len(live) < min_live_obs:
                continue
            decay = self.compute_decay_in_production(name)
            if np.isnan(decay):
                continue
            if decay < decay_threshold:
                candidates.append(name)
                logger.info(
                    "Retirement candidate: %s (decay=%.3f, threshold=%.3f)",
                    name, decay, decay_threshold,
                )

        return candidates

    # ------------------------------------------------------------------
    # Internal: row to ResearchResult
    # ------------------------------------------------------------------

    def _row_to_result(self, row: sqlite3.Row) -> ResearchResult:
        return ResearchResult(
            signal_name=row["signal_name"],
            ic_mean=row["ic_mean"] or 0.0,
            ic_std=row["ic_std"] or 0.0,
            icir=row["icir"] or 0.0,
            ic_decay_halflife=row["ic_decay_halflife"] or float("inf"),
            regime_conditional_ic={
                "bull": row["regime_bull"] or 0.0,
                "bear": row["regime_bear"] or 0.0,
                "ranging": row["regime_ranging"] or 0.0,
            },
            turnover_daily=row["turnover_daily"] or 0.0,
            capacity_estimate_usd=row["capacity_usd"] or 0.0,
            is_significant=bool(row["is_significant"]),
            deflated_sharpe=row["deflated_sharpe"] or 0.0,
            recommendation=row["recommendation"] or "WATCH",
            p_value=row["p_value"] or 1.0,
            n_observations=row["n_observations"] or 0,
            category=row["category"] or "",
            description=row["description"] or "",
        )


# ---------------------------------------------------------------------------
# ResearchReport -- text and HTML report generation
# ---------------------------------------------------------------------------

class ResearchReport:
    """
    Generates research reports from a list of ResearchResult objects.

    Supports HTML, Markdown, and ASCII comparison table output.
    """

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_html(
        self,
        results: List[ResearchResult],
        output_path: str,
    ) -> None:
        """
        Write a styled HTML report to disk.

        Parameters
        ----------
        results : list of ResearchResult
        output_path : str
            File path to write the HTML report.
        """
        sorted_results = sorted(results, key=lambda r: r.icir, reverse=True)

        rows_html = ""
        for r in sorted_results:
            color = (
                "#d4edda" if r.recommendation == "PROMOTE"
                else "#fff3cd" if r.recommendation == "WATCH"
                else "#f8d7da"
            )
            rows_html += f"""
            <tr style="background-color:{color}">
                <td>{r.signal_name}</td>
                <td>{r.category}</td>
                <td>{r.ic_mean:.4f}</td>
                <td>{r.ic_std:.4f}</td>
                <td>{r.icir:.3f}</td>
                <td>{r.ic_decay_halflife:.1f}</td>
                <td>{r.turnover_daily:.2%}</td>
                <td>${r.capacity_estimate_usd:,.0f}</td>
                <td>{"Yes" if r.is_significant else "No"}</td>
                <td>{r.deflated_sharpe:.3f}</td>
                <td><b>{r.recommendation}</b></td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>SRFM Alpha Research Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1 {{ color: #333; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #4CAF50; color: white; }}
    tr:hover {{ filter: brightness(0.95); }}
  </style>
</head>
<body>
  <h1>SRFM Alpha Research Report</h1>
  <p>Generated: {datetime.utcnow().isoformat()} UTC | Signals evaluated: {len(results)}</p>
  <table>
    <thead>
      <tr>
        <th>Signal</th><th>Category</th>
        <th>IC Mean</th><th>IC Std</th><th>ICIR</th>
        <th>Decay HL</th><th>Turnover</th><th>Capacity</th>
        <th>Significant</th><th>DSR</th><th>Recommendation</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        logger.info("HTML report written to %s", output_path)

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------

    def generate_markdown(self, results: List[ResearchResult]) -> str:
        """
        Return a Markdown-formatted research report string.
        """
        sorted_results = sorted(results, key=lambda r: r.icir, reverse=True)
        lines = [
            "# SRFM Alpha Research Report",
            f"Generated: {datetime.utcnow().isoformat()} UTC",
            f"Signals evaluated: {len(results)}",
            "",
            "## Summary Table",
            "",
            "| Signal | Category | IC Mean | ICIR | Decay HL | Turnover | Significant | DSR | Recommendation |",
            "|--------|----------|---------|------|----------|----------|-------------|-----|----------------|",
        ]
        for r in sorted_results:
            lines.append(
                f"| {r.signal_name} | {r.category} | {r.ic_mean:.4f} "
                f"| {r.icir:.3f} | {r.ic_decay_halflife:.1f} "
                f"| {r.turnover_daily:.2%} "
                f"| {'Yes' if r.is_significant else 'No'} "
                f"| {r.deflated_sharpe:.3f} "
                f"| **{r.recommendation}** |"
            )

        lines += [
            "",
            "## Regime Conditional IC",
            "",
            "| Signal | Bull | Bear | Ranging |",
            "|--------|------|------|---------|",
        ]
        for r in sorted_results:
            reg = r.regime_conditional_ic or {}
            lines.append(
                f"| {r.signal_name} "
                f"| {reg.get('bull', float('nan')):.4f} "
                f"| {reg.get('bear', float('nan')):.4f} "
                f"| {reg.get('ranging', float('nan')):.4f} |"
            )

        lines += [
            "",
            "## PROMOTE Signals",
            "",
        ]
        promote = [r for r in sorted_results if r.recommendation == "PROMOTE"]
        if promote:
            for r in promote:
                lines.append(f"- **{r.signal_name}** (ICIR={r.icir:.3f}, DSR={r.deflated_sharpe:.3f})")
        else:
            lines.append("_No signals reached PROMOTE status._")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # ASCII comparison table
    # ------------------------------------------------------------------

    def generate_comparison_table(self, results: List[ResearchResult]) -> str:
        """
        Return an ASCII table of all results sorted by ICIR descending.
        """
        sorted_results = sorted(results, key=lambda r: r.icir, reverse=True)

        col_widths = {
            "name": max(20, max((len(r.signal_name) for r in sorted_results), default=6)),
            "cat":  10,
            "icmn": 9,
            "icir": 8,
            "hl":   8,
            "to":   9,
            "sig":  5,
            "dsr":  7,
            "rec":  10,
        }

        def fmt_row(name, cat, icmn, icir, hl, to, sig, dsr, rec):
            return (
                f"| {name:<{col_widths['name']}} "
                f"| {cat:<{col_widths['cat']}} "
                f"| {icmn:>{col_widths['icmn']}} "
                f"| {icir:>{col_widths['icir']}} "
                f"| {hl:>{col_widths['hl']}} "
                f"| {to:>{col_widths['to']}} "
                f"| {sig:>{col_widths['sig']}} "
                f"| {dsr:>{col_widths['dsr']}} "
                f"| {rec:<{col_widths['rec']}} |"
            )

        sep_parts = [
            "-" * (col_widths["name"] + 2),
            "-" * (col_widths["cat"] + 2),
            "-" * (col_widths["icmn"] + 2),
            "-" * (col_widths["icir"] + 2),
            "-" * (col_widths["hl"] + 2),
            "-" * (col_widths["to"] + 2),
            "-" * (col_widths["sig"] + 2),
            "-" * (col_widths["dsr"] + 2),
            "-" * (col_widths["rec"] + 2),
        ]
        separator = "+" + "+".join(sep_parts) + "+"

        header = fmt_row(
            "Signal", "Category", "IC Mean", "ICIR",
            "Decay HL", "Turnover", "Sig?", "DSR", "Recommend"
        )

        lines = [
            separator,
            header,
            separator,
        ]

        for r in sorted_results:
            reg = r.regime_conditional_ic or {}
            line = fmt_row(
                r.signal_name[:col_widths["name"]],
                (r.category or "")[:col_widths["cat"]],
                f"{r.ic_mean:.4f}",
                f"{r.icir:.3f}",
                f"{r.ic_decay_halflife:.1f}",
                f"{r.turnover_daily:.2%}",
                "Y" if r.is_significant else "N",
                f"{r.deflated_sharpe:.3f}",
                r.recommendation,
            )
            lines.append(line)

        lines.append(separator)
        lines.append(f"  Total signals: {len(sorted_results)}")
        promote_n = sum(1 for r in sorted_results if r.recommendation == "PROMOTE")
        watch_n = sum(1 for r in sorted_results if r.recommendation == "WATCH")
        retire_n = sum(1 for r in sorted_results if r.recommendation == "RETIRE")
        lines.append(f"  PROMOTE: {promote_n}  WATCH: {watch_n}  RETIRE: {retire_n}")

        return "\n".join(lines)
