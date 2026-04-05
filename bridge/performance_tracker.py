"""
bridge/performance_tracker.py

PerformanceTracker: track live performance vs backtest expectations.

Computes z-score of live win-rate deviation from the backtest baseline.
Rolls by week to detect trends. Triggers parameter review when performance
degrades for 2+ consecutive weeks.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parents[1]
_LIVE_TRADES_DB = _REPO_ROOT / "execution" / "live_trades.db"
_TRACKER_DB = Path(__file__).parent / "performance_tracker.db"
_REPORT_FILE = Path(__file__).parent / "performance_report.json"

# Backtest expectations
_EXPECTED_WR = 0.422
_EXPECTED_AVG_PNL = -6.0   # basis points

# Trigger: 2+ consecutive weeks below this threshold
_TRIGGER_WR_DEFICIT = 0.03       # live_wr < expected_wr - 3%
_TRIGGER_CONSECUTIVE_WEEKS = 2
_MIN_TRADES_PER_WEEK = 10


class PerformanceTracker:
    """
    Weekly rolling tracker comparing live performance to backtest expectations.

    Each week:
      - Computes live win-rate from live_trades.db
      - Computes z-score vs expected win-rate
      - Records in SQLite for trend analysis
      - Checks for 2+ consecutive underperforming weeks → parameter review trigger

    Reports are written to performance_report.json.
    """

    def __init__(
        self,
        trades_db: Path | str | None = None,
        tracker_db: Path | str | None = None,
        report_file: Path | str | None = None,
    ) -> None:
        self.trades_db = Path(trades_db) if trades_db else _LIVE_TRADES_DB
        self.tracker_db = Path(tracker_db) if tracker_db else _TRACKER_DB
        self.report_file = Path(report_file) if report_file else _REPORT_FILE
        self.tracker_db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # DB setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.tracker_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS weekly_performance (
                    week_start  TEXT PRIMARY KEY,
                    week_end    TEXT NOT NULL,
                    n_trades    INTEGER,
                    win_rate    REAL,
                    avg_pnl     REAL,
                    expected_wr REAL,
                    wr_z_score  REAL,
                    underperforming INTEGER DEFAULT 0,
                    review_triggered INTEGER DEFAULT 0,
                    computed_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self) -> dict[str, Any]:
        """
        Record this week's performance, compute z-score, check triggers.
        Returns the full report dict.
        """
        week_start, week_end = self._current_week()
        perf = self._compute_weekly_perf(week_start, week_end)

        if perf["n_trades"] < _MIN_TRADES_PER_WEEK:
            logger.info(
                "PerformanceTracker: only %d trades this week — need %d minimum.",
                perf["n_trades"],
                _MIN_TRADES_PER_WEEK,
            )
        else:
            self._store_week(week_start, week_end, perf)

        review_needed = self._check_review_trigger()
        report = self._build_report(perf, review_needed)
        self._write_report(report)
        return report

    def check_review_needed(self) -> bool:
        """Return True if a parameter review should be triggered."""
        return self._check_review_trigger()

    def weekly_trend(self, last_n: int = 8) -> list[dict[str, Any]]:
        """Return the last N weeks of performance data."""
        with sqlite3.connect(self.tracker_db) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM weekly_performance ORDER BY week_start DESC LIMIT ?",
                (last_n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def full_report(self) -> dict[str, Any]:
        """Return the most recently written report."""
        if not self.report_file.exists():
            return self.update()
        try:
            return json.loads(self.report_file.read_text())
        except Exception:
            return self.update()

    # ------------------------------------------------------------------
    # Weekly computation
    # ------------------------------------------------------------------

    def _compute_weekly_perf(
        self, week_start: datetime, week_end: datetime
    ) -> dict[str, Any]:
        """Compute performance metrics for the given week window."""
        trades = self._load_trades_in_window(week_start, week_end)
        if not trades:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "wr_z_score": 0.0,
                "expected_wr": _EXPECTED_WR,
            }

        pnls = [float(t.get("pnl_pct", 0)) for t in trades]
        n = len(pnls)
        wr = sum(1 for p in pnls if p > 0) / n
        avg_pnl = float(np.mean(pnls))

        # Z-score: binomial test
        expected_wins = _EXPECTED_WR * n
        actual_wins = wr * n
        std_est = (n * _EXPECTED_WR * (1 - _EXPECTED_WR)) ** 0.5
        z_score = (actual_wins - expected_wins) / max(std_est, 1e-9)

        return {
            "n_trades": n,
            "win_rate": round(wr, 4),
            "avg_pnl": round(avg_pnl, 6),
            "expected_wr": _EXPECTED_WR,
            "wr_z_score": round(z_score, 3),
        }

    def _store_week(
        self,
        week_start: datetime,
        week_end: datetime,
        perf: dict[str, Any],
    ) -> None:
        wr = perf["win_rate"]
        underperforming = int(wr < _EXPECTED_WR - _TRIGGER_WR_DEFICIT)
        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.tracker_db) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO weekly_performance
                  (week_start, week_end, n_trades, win_rate, avg_pnl,
                   expected_wr, wr_z_score, underperforming, review_triggered, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    week_start.isoformat(),
                    week_end.isoformat(),
                    perf["n_trades"],
                    wr,
                    perf["avg_pnl"],
                    _EXPECTED_WR,
                    perf["wr_z_score"],
                    underperforming,
                    now,
                ),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Review trigger
    # ------------------------------------------------------------------

    def _check_review_trigger(self) -> bool:
        """
        Return True if the last _TRIGGER_CONSECUTIVE_WEEKS are ALL underperforming.
        """
        with sqlite3.connect(self.tracker_db) as conn:
            rows = conn.execute(
                """
                SELECT underperforming FROM weekly_performance
                WHERE n_trades >= ?
                ORDER BY week_start DESC LIMIT ?
                """,
                (_MIN_TRADES_PER_WEEK, _TRIGGER_CONSECUTIVE_WEEKS),
            ).fetchall()

        if len(rows) < _TRIGGER_CONSECUTIVE_WEEKS:
            return False

        all_under = all(r[0] == 1 for r in rows)
        if all_under:
            logger.warning(
                "PerformanceTracker: %d consecutive underperforming weeks — REVIEW TRIGGERED",
                _TRIGGER_CONSECUTIVE_WEEKS,
            )
        return all_under

    def _mark_review_triggered(self, week_start: datetime) -> None:
        with sqlite3.connect(self.tracker_db) as conn:
            conn.execute(
                "UPDATE weekly_performance SET review_triggered = 1 WHERE week_start = ?",
                (week_start.isoformat(),),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Report building
    # ------------------------------------------------------------------

    def _build_report(self, perf: dict[str, Any], review_needed: bool) -> dict[str, Any]:
        trend = self.weekly_trend(last_n=8)
        trend_direction = self._compute_trend_direction(trend)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "current_week": perf,
            "expected_win_rate": _EXPECTED_WR,
            "expected_avg_pnl": _EXPECTED_AVG_PNL,
            "review_needed": review_needed,
            "trend_direction": trend_direction,
            "weekly_history": trend,
            "summary": self._build_summary(perf, review_needed, trend_direction),
        }

    def _compute_trend_direction(self, trend: list[dict]) -> str:
        if len(trend) < 3:
            return "insufficient_data"
        wrs = [t["win_rate"] for t in reversed(trend) if t.get("n_trades", 0) >= _MIN_TRADES_PER_WEEK]
        if len(wrs) < 3:
            return "insufficient_data"
        # Simple linear regression slope
        x = list(range(len(wrs)))
        slope = float(np.polyfit(x, wrs, 1)[0])
        if slope > 0.002:
            return "improving"
        elif slope < -0.002:
            return "degrading"
        return "stable"

    def _build_summary(
        self, perf: dict[str, Any], review_needed: bool, trend: str
    ) -> str:
        wr = perf.get("win_rate", 0)
        n = perf.get("n_trades", 0)
        z = perf.get("wr_z_score", 0)
        parts = [
            f"Live WR: {wr:.1%} vs expected {_EXPECTED_WR:.1%} (z={z:+.2f})",
            f"n_trades: {n}",
            f"trend: {trend}",
        ]
        if review_needed:
            parts.append("ACTION REQUIRED: parameter review triggered")
        return " | ".join(parts)

    def _write_report(self, report: dict[str, Any]) -> None:
        try:
            tmp = self.report_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(report, indent=2, default=str))
            tmp.replace(self.report_file)
        except Exception as exc:
            logger.warning("PerformanceTracker: could not write report: %s", exc)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _load_trades_in_window(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        if not self.trades_db.exists():
            return []
        try:
            with sqlite3.connect(self.trades_db) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM trades WHERE entry_time >= ? AND entry_time < ?",
                    (start.isoformat(), end.isoformat()),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.debug("PerformanceTracker: DB read error: %s", exc)
            return []

    @staticmethod
    def _current_week() -> tuple[datetime, datetime]:
        now = datetime.now(timezone.utc)
        # ISO week: Monday is day 0
        week_start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        week_end = week_start + timedelta(days=7)
        return week_start, week_end
