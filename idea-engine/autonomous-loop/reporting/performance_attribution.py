"""
idea-engine/autonomous-loop/reporting/performance_attribution.py

PerformanceAttributor: tracks which autonomous parameter changes led to
measurable live performance improvements.

Maintains a running scorecard: "Autonomous loop has made N changes.
Net impact: +X% performance." Persists everything to SQLite so the
attribution history survives restarts.
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

_DEFAULT_DB = Path(__file__).parent.parent / "loop_state.db"
_ATTRIBUTION_WINDOW_DAYS = 7    # how long after an application to measure its impact
_MIN_TRADES_FOR_ATTRIBUTION = 20


class PerformanceAttributor:
    """
    Tracks the causal chain: hypothesis → parameter change → live outcome.

    After each parameter application, records the expected improvement
    (from backtests). After _ATTRIBUTION_WINDOW_DAYS, samples actual live
    performance during that window and computes the realised delta.

    Writes a running scorecard that can be queried at any time.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # DB initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parameter_applications (
                    application_id      TEXT PRIMARY KEY,
                    hypothesis_id       TEXT NOT NULL,
                    applied_at          TEXT NOT NULL,
                    expected_sharpe_delta REAL,
                    expected_wr_delta   REAL,
                    params_changed      TEXT,
                    backtest_sharpe     REAL,
                    backtest_wr         REAL,
                    realised_wr_before  REAL,
                    realised_wr_after   REAL,
                    realised_delta      REAL,
                    attribution_status  TEXT DEFAULT 'pending',
                    attributed_at       TEXT
                )
                """
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_application(self, hypothesis, backtest_result) -> None:
        """Record a new parameter application for future attribution."""
        import uuid
        app_id = str(uuid.uuid4())
        applied_at = datetime.now(timezone.utc).isoformat()
        params = json.dumps(getattr(hypothesis, "parameters", {}))
        expected_sharpe_delta = float(
            getattr(backtest_result, "sharpe_ratio", 0.0)
        )
        expected_wr = float(getattr(backtest_result, "win_rate", 0.0))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO parameter_applications
                  (application_id, hypothesis_id, applied_at, expected_sharpe_delta,
                   expected_wr_delta, params_changed, backtest_sharpe, backtest_wr,
                   attribution_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    app_id,
                    getattr(hypothesis, "hypothesis_id", "unknown"),
                    applied_at,
                    expected_sharpe_delta,
                    expected_wr,
                    params,
                    expected_sharpe_delta,
                    expected_wr,
                ),
            )
            conn.commit()

        logger.info(
            "PerformanceAttributor: recorded application %s for hypothesis %s",
            app_id[:8],
            getattr(hypothesis, "hypothesis_id", "?")[:8],
        )

    # ------------------------------------------------------------------
    # Attribution settlement
    # ------------------------------------------------------------------

    def settle_pending(self, live_trades_db: Path | None = None) -> int:
        """
        For each pending application older than _ATTRIBUTION_WINDOW_DAYS,
        compute the realised win-rate delta and mark as settled.
        Returns the number of applications settled.
        """
        from pathlib import Path as P
        trades_db = live_trades_db or (
            P(__file__).parents[4] / "execution" / "live_trades.db"
        )

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=_ATTRIBUTION_WINDOW_DAYS)
        ).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT application_id, hypothesis_id, applied_at, backtest_wr
                FROM parameter_applications
                WHERE attribution_status = 'pending' AND applied_at < ?
                """,
                (cutoff,),
            ).fetchall()

        settled = 0
        for app_id, hyp_id, applied_at_str, bt_wr in rows:
            applied_at = datetime.fromisoformat(applied_at_str)
            before_start = applied_at - timedelta(days=_ATTRIBUTION_WINDOW_DAYS)
            after_end = applied_at + timedelta(days=_ATTRIBUTION_WINDOW_DAYS)

            wr_before = self._compute_live_wr(trades_db, before_start, applied_at)
            wr_after = self._compute_live_wr(trades_db, applied_at, after_end)

            if wr_before is None or wr_after is None:
                continue

            delta = wr_after - wr_before
            now = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE parameter_applications
                    SET realised_wr_before = ?, realised_wr_after = ?,
                        realised_delta = ?, attribution_status = 'settled',
                        attributed_at = ?
                    WHERE application_id = ?
                    """,
                    (wr_before, wr_after, delta, now, app_id),
                )
                conn.commit()

            logger.info(
                "PerformanceAttributor: settled %s — delta=%.3f", app_id[:8], delta
            )
            settled += 1

        return settled

    # ------------------------------------------------------------------
    # Scorecard
    # ------------------------------------------------------------------

    def scorecard(self) -> dict[str, Any]:
        """
        Build the running scorecard.

        Returns:
          total_applications, settled, net_wr_delta, avg_wr_delta_per_change,
          positive_changes, negative_changes, neutral_changes, best_change, worst_change
        """
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM parameter_applications"
            ).fetchone()[0]
            settled_rows = conn.execute(
                "SELECT realised_delta FROM parameter_applications WHERE attribution_status = 'settled'"
            ).fetchall()

        deltas = [r[0] for r in settled_rows if r[0] is not None]

        if not deltas:
            return {
                "total_applications": total,
                "settled": 0,
                "net_wr_delta": 0.0,
                "avg_wr_delta_per_change": 0.0,
                "positive_changes": 0,
                "negative_changes": 0,
                "neutral_changes": 0,
                "best_change": None,
                "worst_change": None,
                "summary": f"Autonomous loop has made {total} changes. No settled attributions yet.",
            }

        net_delta = float(np.sum(deltas))
        avg_delta = float(np.mean(deltas))
        positive = sum(1 for d in deltas if d > 0.005)
        negative = sum(1 for d in deltas if d < -0.005)
        neutral = len(deltas) - positive - negative

        summary = (
            f"Autonomous loop has made {total} changes. "
            f"Net impact: {net_delta:+.1%} win-rate. "
            f"({positive} improved, {negative} degraded, {neutral} neutral)"
        )

        return {
            "total_applications": total,
            "settled": len(deltas),
            "net_wr_delta": round(net_delta, 4),
            "avg_wr_delta_per_change": round(avg_delta, 4),
            "positive_changes": positive,
            "negative_changes": negative,
            "neutral_changes": neutral,
            "best_change": round(max(deltas), 4),
            "worst_change": round(min(deltas), 4),
            "summary": summary,
        }

    def print_scorecard(self) -> None:
        """Pretty-print the scorecard to the logger."""
        sc = self.scorecard()
        logger.info("=== PERFORMANCE ATTRIBUTION SCORECARD ===")
        logger.info(sc["summary"])
        logger.info(
            "  Settled: %d | Best: %s | Worst: %s",
            sc["settled"],
            f"{sc['best_change']:+.1%}" if sc["best_change"] is not None else "n/a",
            f"{sc['worst_change']:+.1%}" if sc["worst_change"] is not None else "n/a",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_live_wr(
        self, trades_db: Path, start: datetime, end: datetime
    ) -> float | None:
        """Compute win rate from live_trades.db within a time window."""
        if not trades_db.exists():
            return None
        try:
            with sqlite3.connect(trades_db) as conn:
                rows = conn.execute(
                    """
                    SELECT pnl_pct FROM trades
                    WHERE entry_time >= ? AND entry_time < ?
                    """,
                    (start.isoformat(), end.isoformat()),
                ).fetchall()
            pnls = [float(r[0]) for r in rows]
            if len(pnls) < _MIN_TRADES_FOR_ATTRIBUTION:
                return None
            return float(sum(1 for p in pnls if p > 0) / len(pnls))
        except Exception as exc:
            logger.debug("Attribution wr computation failed: %s", exc)
            return None

    def get_recent_applications(self, last_n: int = 10) -> list[dict[str, Any]]:
        """Return the last N applications with attribution status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT application_id, hypothesis_id, applied_at, expected_wr_delta,
                       realised_delta, attribution_status, attributed_at
                FROM parameter_applications
                ORDER BY applied_at DESC LIMIT ?
                """,
                (last_n,),
            ).fetchall()
        return [dict(r) for r in rows]
