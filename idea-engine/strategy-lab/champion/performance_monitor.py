"""
performance_monitor.py
----------------------
Monitor the champion strategy's live performance vs expectations.

Expectations are set at promotion time (from backtest or A/B test):
  * Expected win rate, average P&L per trade, Sharpe.

The monitor computes actual live metrics daily and raises a PerformanceAlert
if actual performance falls > 20% below expected for 5+ consecutive days.

Regime adjustment
-----------------
Performance is macro-adjusted using a simple beta model:
  regime_factor = correlation(strategy_daily_pnl, market_daily_returns)
If the strategy underperforms mainly when the market is in a bad regime,
that's "macro-driven". If it underperforms even in a neutral/good regime,
that's "signal-driven".
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd

_DEFAULT_DB             = Path(__file__).parent.parent / "strategy_lab.db"
_ALERT_THRESHOLD        = 0.20   # 20 % below expected triggers alert
_CONSECUTIVE_DAYS_LIMIT = 5      # must persist for 5 days before alerting


# ---------------------------------------------------------------------------
# PerformanceAlert
# ---------------------------------------------------------------------------

@dataclass
class PerformanceAlert:
    """
    Fired when live performance meaningfully underperforms expectations.

    Attributes
    ----------
    version_id          : champion version being monitored
    metric              : "win_rate" | "avg_pnl" | "sharpe"
    expected_value      : expectation at promotion time
    actual_value        : live value over alert window
    pct_below_expected  : how far below expected (positive = bad)
    consecutive_days    : how many days the underperformance has persisted
    is_macro_driven     : True if underperformance correlates with market regime
    timestamp           : when the alert was raised
    message             : human-readable explanation
    """
    version_id: str
    metric: str
    expected_value: float
    actual_value: float
    pct_below_expected: float
    consecutive_days: int
    is_macro_driven: bool
    timestamp: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        regime = "[MACRO]" if self.is_macro_driven else "[SIGNAL]"
        return (
            f"ALERT {regime} {self.metric}: expected={self.expected_value:.4f}, "
            f"actual={self.actual_value:.4f} ({self.pct_below_expected:.1%} below) "
            f"for {self.consecutive_days} consecutive days"
        )


# ---------------------------------------------------------------------------
# Expectation profile
# ---------------------------------------------------------------------------

@dataclass
class PerformanceExpectation:
    """Baseline expectations for a champion strategy."""
    version_id: str
    expected_win_rate: float
    expected_avg_pnl: float
    expected_sharpe: float
    set_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PerformanceExpectation":
        return cls(**d)


# ---------------------------------------------------------------------------
# PerformanceMonitor
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """
    Daily performance monitor for the live champion strategy.

    Parameters
    ----------
    db_path              : SQLite path
    alert_threshold      : fraction below expected to trigger alert (default 0.20)
    consecutive_required : number of consecutive bad days before alerting (default 5)
    """

    def __init__(
        self,
        db_path: str | Path = _DEFAULT_DB,
        alert_threshold: float = _ALERT_THRESHOLD,
        consecutive_required: int = _CONSECUTIVE_DAYS_LIMIT,
    ) -> None:
        self.db_path             = Path(db_path)
        self.alert_threshold     = alert_threshold
        self.consecutive_required = consecutive_required
        self._init_db()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_expectations(
        self,
        version_id: str,
        *,
        win_rate: float,
        avg_pnl: float,
        sharpe: float,
    ) -> PerformanceExpectation:
        """Register baseline expectations for a champion version."""
        exp = PerformanceExpectation(
            version_id=version_id,
            expected_win_rate=win_rate,
            expected_avg_pnl=avg_pnl,
            expected_sharpe=sharpe,
            set_at=_now_iso(),
        )
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO perf_expectations
                   (version_id, expectation_json) VALUES (?,?)""",
                (version_id, json.dumps(exp.to_dict(), default=str)),
            )
        return exp

    def get_expectations(self, version_id: str) -> PerformanceExpectation | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT expectation_json FROM perf_expectations WHERE version_id = ?",
                (version_id,),
            ).fetchone()
        return PerformanceExpectation.from_dict(json.loads(row[0])) if row else None

    # ------------------------------------------------------------------
    # Daily update
    # ------------------------------------------------------------------

    def record_daily(
        self,
        version_id: str,
        date_str: str,
        trades: list[dict[str, Any]],
        market_daily_return: float = 0.0,
    ) -> list[PerformanceAlert]:
        """
        Called once per trading day with that day's live trades.
        Returns any new alerts triggered.
        """
        # Compute daily metrics
        n_trades  = len(trades)
        pnl_list  = [float(t.get("pnl", 0.0)) for t in trades]
        daily_pnl = sum(pnl_list)
        win_rate  = sum(1 for p in pnl_list if p > 0) / n_trades if n_trades > 0 else 0.0
        avg_pnl   = daily_pnl / n_trades if n_trades > 0 else 0.0

        row = {
            "version_id": version_id,
            "date": date_str,
            "n_trades": n_trades,
            "daily_pnl": daily_pnl,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "market_return": market_daily_return,
        }
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO daily_perf
                   (version_id, date, n_trades, daily_pnl, win_rate, avg_pnl, market_return)
                   VALUES (:version_id,:date,:n_trades,:daily_pnl,:win_rate,:avg_pnl,:market_return)""",
                row,
            )

        return self._check_alerts(version_id)

    # ------------------------------------------------------------------
    # Alert checking
    # ------------------------------------------------------------------

    def _check_alerts(self, version_id: str) -> list[PerformanceAlert]:
        exp = self.get_expectations(version_id)
        if exp is None:
            return []

        alerts: list[PerformanceAlert] = []
        history = self._load_recent_days(version_id, n=self.consecutive_required + 5)

        if len(history) < self.consecutive_required:
            return []

        # Rolling window of last `consecutive_required` days
        window = history[-self.consecutive_required:]

        metrics = {
            "win_rate": (
                float(np.mean([r["win_rate"] for r in window])),
                exp.expected_win_rate,
            ),
            "avg_pnl": (
                float(np.mean([r["avg_pnl"] for r in window])),
                exp.expected_avg_pnl,
            ),
        }

        # Compute rolling Sharpe from full history
        pnl_series = [r["daily_pnl"] for r in history]
        actual_sharpe = _sharpe(pnl_series)
        metrics["sharpe"] = (actual_sharpe, exp.expected_sharpe)

        market_returns = [r["market_return"] for r in window]
        strategy_pnls  = [r["daily_pnl"] for r in window]

        for metric_name, (actual, expected) in metrics.items():
            if expected <= 0:
                continue
            pct_below = (expected - actual) / abs(expected)
            if pct_below < self.alert_threshold:
                continue

            is_macro = self._is_macro_driven(strategy_pnls, market_returns)
            alert = PerformanceAlert(
                version_id=version_id,
                metric=metric_name,
                expected_value=expected,
                actual_value=actual,
                pct_below_expected=pct_below,
                consecutive_days=self.consecutive_required,
                is_macro_driven=is_macro,
                timestamp=_now_iso(),
                message=self._alert_message(
                    metric_name, expected, actual, pct_below, is_macro
                ),
            )
            alerts.append(alert)
            self._persist_alert(alert)

        return alerts

    @staticmethod
    def _is_macro_driven(strategy_pnls: list[float], market_returns: list[float]) -> bool:
        """
        Returns True if strategy underperformance co-moves with market declines.
        Uses Pearson correlation between strategy P&L and market returns.
        """
        if len(strategy_pnls) < 3 or len(market_returns) < 3:
            return False
        s = np.array(strategy_pnls, dtype=float)
        m = np.array(market_returns, dtype=float)
        if np.std(s) < 1e-9 or np.std(m) < 1e-9:
            return False
        corr = float(np.corrcoef(s, m)[0, 1])
        return corr > 0.5  # strategy moves with market — macro-driven

    @staticmethod
    def _alert_message(
        metric: str, expected: float, actual: float,
        pct_below: float, is_macro: bool,
    ) -> str:
        cause = "macro regime" if is_macro else "strategy signal decay"
        return (
            f"{metric} degraded: expected={expected:.4f}, actual={actual:.4f} "
            f"({pct_below:.1%} below). Likely cause: {cause}."
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def recent_performance(
        self, version_id: str, days: int = 30
    ) -> pd.DataFrame:
        """Return a DataFrame of the last N days of daily performance metrics."""
        rows = self._load_recent_days(version_id, n=days)
        return pd.DataFrame(rows)

    def all_alerts(self, version_id: str) -> list[PerformanceAlert]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT alert_json FROM perf_alerts WHERE version_id = ? ORDER BY rowid",
                (version_id,),
            ).fetchall()
        return [PerformanceAlert(**json.loads(r[0])) for r in rows]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_recent_days(self, version_id: str, n: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT date, n_trades, daily_pnl, win_rate, avg_pnl, market_return
                   FROM daily_perf WHERE version_id = ?
                   ORDER BY date DESC LIMIT ?""",
                (version_id, n),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def _persist_alert(self, alert: PerformanceAlert) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO perf_alerts (version_id, alert_json) VALUES (?,?)",
                (alert.version_id, json.dumps(alert.to_dict(), default=str)),
            )

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS perf_expectations (
                    version_id       TEXT PRIMARY KEY,
                    expectation_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS daily_perf (
                    version_id    TEXT NOT NULL,
                    date          TEXT NOT NULL,
                    n_trades      INTEGER NOT NULL DEFAULT 0,
                    daily_pnl     REAL NOT NULL DEFAULT 0,
                    win_rate      REAL NOT NULL DEFAULT 0,
                    avg_pnl       REAL NOT NULL DEFAULT 0,
                    market_return REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (version_id, date)
                );
                CREATE TABLE IF NOT EXISTS perf_alerts (
                    rowid      INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id TEXT NOT NULL,
                    alert_json TEXT NOT NULL
                );
            """)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sharpe(daily_pnl: list[float]) -> float:
    if len(daily_pnl) < 2:
        return 0.0
    arr = np.array(daily_pnl, dtype=float)
    std = float(np.std(arr, ddof=1))
    return float(np.mean(arr) / std * math.sqrt(252)) if std > 0 else 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
