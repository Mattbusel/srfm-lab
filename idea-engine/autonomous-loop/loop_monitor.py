"""
idea-engine/autonomous-loop/loop_monitor.py

LoopMonitor: monitors the autonomous loop itself and detects live performance drift.

Writes loop_health.json every cycle. Runs Bayesian drift detection via the
existing DriftMonitor. Checks live win-rate vs backtest expectations.
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

_REPO_ROOT = Path(__file__).parents[3]
_LIVE_TRADES_DB = _REPO_ROOT / "execution" / "live_trades.db"
_HEALTH_FILE = Path(__file__).parent / "loop_health.json"

# Backtest expectations (from crypto_backtest_mc results)
_EXPECTED_WIN_RATE = 0.422
_EXPECTED_AVG_PNL = -6.0  # avg pnl in basis points (net of fees)

# Alert thresholds
_ALERT_LOSING_STREAK = 10
_ALERT_WIN_RATE_MIN = 0.35
_ALERT_SAMPLE_SIZE = 50
_ALERT_CYCLE_SLOWDOWN_FACTOR = 2.0
_ALERT_NO_HYPOTHESES_HOURS = 24
_ALERT_BACKTEST_CLOG_HOURS = 12


class LoopMonitor:
    """
    Health monitor for the autonomous loop and live trading performance.

    Per cycle:
      - Checks loop timing (alerts if > 2x expected cycle time)
      - Checks hypothesis generation rate
      - Checks backtest pipeline for clogs
      - Runs Bayesian drift detection on live returns
      - Writes loop_health.json
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        health_file: Path | str | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else (_LIVE_TRADES_DB.parent / "loop_state.db")
        self.health_file = Path(health_file) if health_file else _HEALTH_FILE
        self._cycle_times: list[float] = []
        self._expected_cycle_s: float = 3600.0
        self._last_cycle_time: str = ""
        self._drift_monitor = self._build_drift_monitor()

    # ------------------------------------------------------------------
    # Main health check
    # ------------------------------------------------------------------

    def record_cycle(
        self,
        cycle_number: int,
        elapsed_seconds: float,
        patterns_found: int,
        hypotheses_generated: int,
        drift_alerts: int,
    ) -> None:
        """Record metrics for a completed cycle and write health file."""
        self._cycle_times.append(elapsed_seconds)
        if len(self._cycle_times) > 20:
            self._cycle_times.pop(0)
        self._last_cycle_time = datetime.now(timezone.utc).isoformat()

        alerts = []

        # Timing alert
        avg_cycle = float(np.mean(self._cycle_times)) if self._cycle_times else elapsed_seconds
        if elapsed_seconds > self._expected_cycle_s * _ALERT_CYCLE_SLOWDOWN_FACTOR:
            alerts.append(
                f"Cycle #{cycle_number} took {elapsed_seconds:.0f}s ({_ALERT_CYCLE_SLOWDOWN_FACTOR}x expected)"
            )

        # Hypothesis generation rate
        if hypotheses_generated == 0:
            alerts.append(f"Cycle #{cycle_number}: 0 new hypotheses generated")

        health = {
            "cycle_number": cycle_number,
            "last_cycle_at": self._last_cycle_time,
            "elapsed_seconds": elapsed_seconds,
            "avg_cycle_seconds": avg_cycle,
            "patterns_found": patterns_found,
            "hypotheses_generated": hypotheses_generated,
            "drift_alerts": drift_alerts,
            "alerts": alerts,
            "status": "ALERT" if alerts else "OK",
        }

        self._write_health(health)
        if alerts:
            for alert in alerts:
                logger.warning("LoopMonitor ALERT: %s", alert)

    # ------------------------------------------------------------------
    # Live performance monitoring
    # ------------------------------------------------------------------

    def check_live_performance(self) -> dict[str, Any]:
        """
        Compare live win-rate to backtest expectations.
        Returns a health dict with status, metrics, and any alerts.
        """
        trades = self._load_recent_trades(n=_ALERT_SAMPLE_SIZE)
        if not trades:
            return {"status": "no_data", "n_trades": 0, "alerts": []}

        pnls = [float(t.get("pnl_pct", 0)) for t in trades]
        wins = [p for p in pnls if p > 0]
        live_wr = len(wins) / len(pnls) if pnls else 0.0
        avg_pnl = float(np.mean(pnls)) if pnls else 0.0

        # Losing streak
        streak = 0
        for p in reversed(pnls):
            if p < 0:
                streak += 1
            else:
                break

        alerts = []
        if streak >= _ALERT_LOSING_STREAK:
            alerts.append(f"Losing streak: {streak} consecutive losses")
        if live_wr < _ALERT_WIN_RATE_MIN and len(pnls) >= 20:
            alerts.append(
                f"Win rate {live_wr:.1%} < threshold {_ALERT_WIN_RATE_MIN:.1%} "
                f"over last {len(pnls)} trades"
            )

        # Z-score of win rate deviation from expectation
        if len(pnls) >= 20:
            expected_wins = _EXPECTED_WIN_RATE * len(pnls)
            actual_wins = len(wins)
            std_est = (len(pnls) * _EXPECTED_WIN_RATE * (1 - _EXPECTED_WIN_RATE)) ** 0.5
            z_score = (actual_wins - expected_wins) / max(std_est, 1e-9)
        else:
            z_score = 0.0

        result = {
            "status": "ALERT" if alerts else "OK",
            "n_trades": len(pnls),
            "live_win_rate": round(live_wr, 4),
            "expected_win_rate": _EXPECTED_WIN_RATE,
            "win_rate_z_score": round(z_score, 3),
            "avg_pnl": round(avg_pnl, 4),
            "losing_streak": streak,
            "alerts": alerts,
        }

        for alert in alerts:
            logger.warning("LoopMonitor live_perf ALERT: %s", alert)

        return result

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(self) -> list[str]:
        """
        Run Bayesian / CUSUM drift detection on recent live returns.
        Returns list of alert message strings.
        """
        trades = self._load_recent_trades(n=200)
        if not trades:
            return []

        pnls = np.array([float(t.get("pnl_pct", 0)) for t in trades])
        wins = (pnls > 0).astype(float)

        alerts = []
        if self._drift_monitor is not None:
            try:
                drift_result = self._drift_monitor.check(
                    new_live=wins,
                    historical=wins[:50] if len(wins) >= 50 else wins,
                )
                if drift_result.is_drifting:
                    msg = f"Drift detected: {drift_result.message}"
                    alerts.append(msg)
                    logger.warning("LoopMonitor: %s", msg)
            except Exception as exc:
                logger.debug("DriftMonitor.check failed: %s", exc)

        return alerts

    # ------------------------------------------------------------------
    # Backtest pipeline clog detection
    # ------------------------------------------------------------------

    def check_backtest_clog(self) -> bool:
        """
        Return True if any hypothesis has been stuck in BACKTEST stage > 12h.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=_ALERT_BACKTEST_CLOG_HOURS)
        ).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) FROM hypothesis_queue
                    WHERE stage = 'backtest' AND stage_entered < ?
                    """,
                    (cutoff,),
                ).fetchone()
            return (row[0] if row else 0) > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_recent_trades(self, n: int = 100) -> list[dict]:
        if not _LIVE_TRADES_DB.exists():
            return []
        try:
            with sqlite3.connect(_LIVE_TRADES_DB) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT pnl_pct, entry_time FROM trades ORDER BY entry_time DESC LIMIT ?",
                    (n,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.debug("LoopMonitor: could not load trades: %s", exc)
            return []

    def _build_drift_monitor(self):
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parents[2]))
            from bayesian_updater.drift_monitor import DriftMonitor
            import numpy as np
            return DriftMonitor(historical_win_rates=np.full(50, _EXPECTED_WIN_RATE))
        except Exception as exc:
            logger.debug("DriftMonitor not available: %s", exc)
            return None

    def _write_health(self, health: dict[str, Any]) -> None:
        try:
            self.health_file.write_text(json.dumps(health, indent=2))
        except Exception as exc:
            logger.warning("LoopMonitor: could not write health file: %s", exc)
