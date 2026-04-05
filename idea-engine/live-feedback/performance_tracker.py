"""
live-feedback/performance_tracker.py
=====================================
Real-time streaming performance tracker.

All metrics are updated *online* as individual trades arrive, so there is no
need to replay the full trade history on every call.  Long-window statistics
use a fixed-size circular buffer to bound memory usage.

Metrics computed
----------------
* Annualised running Sharpe ratio (rolling window)
* Running maximum drawdown from equity peak
* Rolling win rate
* Annualised running Calmar ratio
* Trade expectancy (avg_win × win_rate − avg_loss × loss_rate)
* Daily P&L series (for charting)
* Alpha / beta / correlation vs. a benchmark equity curve

Snapshots of all metrics are persisted to ``performance_snapshots`` in
``idea_engine.db`` every ``snapshot_interval_seconds`` (default: 3600).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, date, timezone
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANNUALISATION = math.sqrt(252)       # daily → annualised Sharpe
DEFAULT_WINDOW = 252                  # bars for rolling Sharpe / Calmar
WIN_RATE_WINDOW = 50                  # bars for rolling win rate
EXPECTANCY_WINDOW = 50                # bars for rolling expectancy
SNAPSHOT_INTERVAL = 3600             # seconds between DB snapshots


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PerformanceSnapshot:
    """
    Point-in-time snapshot of all performance metrics.
    Matches the ``performance_snapshots`` table schema.
    """

    ts: str
    equity: float
    running_sharpe: float
    running_dd: float
    win_rate: float
    calmar: float
    expectancy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkStats:
    """Alpha / beta / correlation vs. a single benchmark."""

    benchmark_name: str
    alpha: float
    beta: float
    correlation: float
    information_ratio: float
    tracking_error: float


# ---------------------------------------------------------------------------
# PerformanceTracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """
    Streaming performance tracker updated trade-by-trade.

    Parameters
    ----------
    iae_conn            : sqlite3.Connection
        Open connection to ``idea_engine.db`` for snapshot persistence.
    sharpe_window       : int
        Number of observations for the rolling Sharpe / Calmar window.
    win_rate_window     : int
        Number of trades for the rolling win rate.
    expectancy_window   : int
        Number of trades for the rolling expectancy.
    snapshot_interval   : float
        Seconds between automatic DB snapshots (default: 3600).
    initial_equity      : float
        Starting equity value (default: 100 000).
    """

    def __init__(
        self,
        iae_conn: sqlite3.Connection,
        *,
        sharpe_window: int = DEFAULT_WINDOW,
        win_rate_window: int = WIN_RATE_WINDOW,
        expectancy_window: int = EXPECTANCY_WINDOW,
        snapshot_interval: float = SNAPSHOT_INTERVAL,
        initial_equity: float = 100_000.0,
    ) -> None:
        self._conn = iae_conn
        self._sharpe_window = sharpe_window
        self._win_rate_window = win_rate_window
        self._expectancy_window = expectancy_window
        self._snapshot_interval = snapshot_interval

        # State
        self._equity = initial_equity
        self._peak_equity = initial_equity
        self._trade_count = 0
        self._last_snapshot_ts = time.monotonic()

        # Circular buffers
        self._returns: deque[float] = deque(maxlen=sharpe_window)
        self._pnl_window: deque[float] = deque(maxlen=win_rate_window)
        self._win_buffer: deque[bool] = deque(maxlen=win_rate_window)
        self._wins_for_expectancy: deque[float] = deque(maxlen=expectancy_window)
        self._losses_for_expectancy: deque[float] = deque(maxlen=expectancy_window)

        # Daily P&L accumulator
        self._daily_pnl: dict[date, float] = {}

        # Running max drawdown tracking
        self._max_dd: float = 0.0

        logger.info(
            "PerformanceTracker initialised. initial_equity=%.2f, sharpe_window=%d.",
            initial_equity,
            sharpe_window,
        )

    # ------------------------------------------------------------------
    # Streaming update
    # ------------------------------------------------------------------

    def update(self, trade: dict[str, Any] | pd.Series) -> None:
        """
        Process a single closed trade and update all running statistics.

        Expected keys in ``trade``:
            pnl      : float — absolute P&L in quote currency
            pnl_pct  : float — percentage return (e.g. 0.02 = 2 %)
            closed_at : str  — ISO-8601 UTC timestamp

        Parameters
        ----------
        trade : dict or pd.Series
        """
        if isinstance(trade, pd.Series):
            trade = trade.to_dict()

        pnl = float(trade.get("pnl", 0.0))
        pnl_pct = float(trade.get("pnl_pct", 0.0))
        closed_at_str = str(trade.get("closed_at", ""))

        # Update equity
        self._equity += pnl
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        # Current drawdown
        if self._peak_equity > 0:
            current_dd = (self._peak_equity - self._equity) / self._peak_equity
            if current_dd > self._max_dd:
                self._max_dd = current_dd

        # Buffers
        self._returns.append(pnl_pct)
        self._pnl_window.append(pnl)
        self._win_buffer.append(pnl > 0)

        if pnl > 0:
            self._wins_for_expectancy.append(pnl)
        else:
            self._losses_for_expectancy.append(abs(pnl))

        # Daily P&L
        try:
            closed_dt = datetime.fromisoformat(closed_at_str.replace("Z", "+00:00"))
            trade_date = closed_dt.date()
        except (ValueError, AttributeError):
            trade_date = date.today()
        self._daily_pnl[trade_date] = self._daily_pnl.get(trade_date, 0.0) + pnl

        self._trade_count += 1

        # Periodic snapshot
        if time.monotonic() - self._last_snapshot_ts >= self._snapshot_interval:
            self._take_snapshot()

    # ------------------------------------------------------------------
    # Metric accessors
    # ------------------------------------------------------------------

    def running_sharpe(self, window: int | None = None) -> float:
        """
        Annualised Sharpe ratio over the most recent ``window`` observations.

        Uses the circular buffer of percentage returns.  Returns 0.0 if
        there are fewer than 2 observations or zero standard deviation.

        Parameters
        ----------
        window : int | None
            If given, only the last ``window`` returns are used.
            Defaults to the tracker's configured ``sharpe_window``.

        Returns
        -------
        float — annualised Sharpe ratio.
        """
        returns = list(self._returns)
        if window is not None:
            returns = returns[-window:]
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns, dtype=float)
        std = arr.std(ddof=1)
        if std == 0:
            return 0.0
        return float(arr.mean() / std * ANNUALISATION)

    def running_drawdown(self) -> float:
        """
        Current drawdown from the equity high-water mark.

        Returns
        -------
        float — drawdown as a fraction (e.g. 0.15 = 15 % drawdown).
        """
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._equity) / self._peak_equity)

    def running_win_rate(self, window: int | None = None) -> float:
        """
        Fraction of trades that were profitable over the rolling window.

        Parameters
        ----------
        window : int | None
            If given, only the last ``window`` trades are used.
            Defaults to ``win_rate_window``.

        Returns
        -------
        float in [0, 1].
        """
        wins = list(self._win_buffer)
        if window is not None:
            wins = wins[-window:]
        if not wins:
            return 0.0
        return float(sum(wins) / len(wins))

    def running_calmar(self, window: int | None = None) -> float:
        """
        Annualised Calmar ratio = annualised return / max drawdown.

        Uses the daily P&L series to compute the annualised return.
        Returns 0.0 if max drawdown is zero or there are no trades.

        Parameters
        ----------
        window : int | None
            Days of history to include.  Defaults to ``sharpe_window``.

        Returns
        -------
        float — Calmar ratio.
        """
        if self._max_dd == 0 or not self._daily_pnl:
            return 0.0

        sorted_dates = sorted(self._daily_pnl.keys())
        daily_pnl_vals = [self._daily_pnl[d] for d in sorted_dates]

        if window is not None:
            daily_pnl_vals = daily_pnl_vals[-window:]

        if not daily_pnl_vals or self._equity == 0:
            return 0.0

        total_return = sum(daily_pnl_vals) / (self._equity - sum(daily_pnl_vals) + 1e-9)
        n_days = max(len(daily_pnl_vals), 1)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        return float(annual_return / self._max_dd)

    def expectancy(self, window: int | None = None) -> float:
        """
        Trade expectancy = avg_win × win_rate − avg_loss × loss_rate.

        A positive expectancy means the strategy makes money on average per
        trade.

        Parameters
        ----------
        window : int | None
            If given, the calculation uses only the last ``window`` trades.

        Returns
        -------
        float — expectancy in quote currency per trade.
        """
        pnl_list = list(self._pnl_window)
        if window is not None:
            pnl_list = pnl_list[-window:]
        if not pnl_list:
            return 0.0

        wins = [p for p in pnl_list if p > 0]
        losses = [abs(p) for p in pnl_list if p < 0]
        n = len(pnl_list)

        win_rate = len(wins) / n
        loss_rate = len(losses) / n
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        return avg_win * win_rate - avg_loss * loss_rate

    def daily_pnl_series(self) -> pd.Series:
        """
        Return the accumulated daily P&L as a ``pd.Series`` indexed by date.

        Returns
        -------
        pd.Series — index is ``datetime.date``, values are float P&L.
        """
        if not self._daily_pnl:
            return pd.Series(dtype=float)
        sorted_dates = sorted(self._daily_pnl.keys())
        return pd.Series(
            [self._daily_pnl[d] for d in sorted_dates],
            index=pd.to_datetime(sorted_dates),
            name="daily_pnl",
        )

    def performance_vs_benchmark(
        self,
        benchmark: str = "btc_buy_hold",
        benchmark_returns: list[float] | np.ndarray | None = None,
    ) -> BenchmarkStats:
        """
        Compute alpha, beta, correlation, information ratio, and tracking
        error relative to a benchmark.

        If ``benchmark_returns`` is not supplied the method attempts to
        load daily returns from ``benchmark_prices`` table in
        ``idea_engine.db``.  If neither is available it returns zeros.

        Parameters
        ----------
        benchmark       : str
            Benchmark identifier (used for DB lookup and the returned label).
        benchmark_returns : list[float] | np.ndarray | None
            Pre-supplied benchmark return series.  Must be the same length
            as the tracker's daily return series.

        Returns
        -------
        BenchmarkStats
        """
        strat_series = self.daily_pnl_series()
        if strat_series.empty:
            return BenchmarkStats(
                benchmark_name=benchmark,
                alpha=0.0, beta=0.0, correlation=0.0,
                information_ratio=0.0, tracking_error=0.0,
            )

        # Convert daily P&L to percentage returns
        start_equity = self._equity - float(strat_series.sum())
        if abs(start_equity) < 1e-9:
            start_equity = 1.0
        strat_pct = strat_series / start_equity

        # Load / align benchmark returns
        if benchmark_returns is not None:
            bm_arr = np.asarray(benchmark_returns, dtype=float)
        else:
            bm_arr = self._load_benchmark_returns(benchmark, len(strat_pct))

        n = min(len(strat_pct), len(bm_arr))
        if n < 2:
            return BenchmarkStats(
                benchmark_name=benchmark,
                alpha=0.0, beta=0.0, correlation=0.0,
                information_ratio=0.0, tracking_error=0.0,
            )

        strat_arr = strat_pct.values[-n:]
        bm_arr = bm_arr[-n:]

        # Beta = Cov(strat, bm) / Var(bm)
        bm_var = float(np.var(bm_arr, ddof=1))
        cov = float(np.cov(strat_arr, bm_arr, ddof=1)[0, 1])
        beta = cov / bm_var if bm_var != 0 else 0.0

        # Alpha = mean(strat) - beta * mean(bm), annualised
        alpha = float((np.mean(strat_arr) - beta * np.mean(bm_arr)) * 252)

        # Correlation
        std_s = float(np.std(strat_arr, ddof=1))
        std_b = float(np.std(bm_arr, ddof=1))
        corr = cov / (std_s * std_b) if std_s > 0 and std_b > 0 else 0.0

        # Tracking error and IR
        excess = strat_arr - bm_arr
        tracking_error = float(np.std(excess, ddof=1) * ANNUALISATION)
        ir = float(np.mean(excess) * 252 / (tracking_error + 1e-9))

        return BenchmarkStats(
            benchmark_name=benchmark,
            alpha=alpha,
            beta=beta,
            correlation=corr,
            information_ratio=ir,
            tracking_error=tracking_error,
        )

    # ------------------------------------------------------------------
    # Snapshot persistence
    # ------------------------------------------------------------------

    def take_snapshot_now(self) -> PerformanceSnapshot:
        """
        Force an immediate performance snapshot and persist it to the DB.

        Returns
        -------
        PerformanceSnapshot — the snapshot written.
        """
        return self._take_snapshot()

    def _take_snapshot(self) -> PerformanceSnapshot:
        snap = PerformanceSnapshot(
            ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            equity=self._equity,
            running_sharpe=self.running_sharpe(),
            running_dd=self.running_drawdown(),
            win_rate=self.running_win_rate(),
            calmar=self.running_calmar(),
            expectancy=self.expectancy(),
        )
        self._persist_snapshot(snap)
        self._last_snapshot_ts = time.monotonic()
        return snap

    def _persist_snapshot(self, snap: PerformanceSnapshot) -> None:
        try:
            self._conn.execute(
                """
                INSERT INTO performance_snapshots
                    (ts, equity, running_sharpe, running_dd, win_rate, calmar, expectancy)
                VALUES
                    (:ts, :equity, :running_sharpe, :running_dd, :win_rate, :calmar, :expectancy)
                """,
                snap.to_dict(),
            )
            self._conn.commit()
            logger.debug("Performance snapshot persisted at %s.", snap.ts)
        except sqlite3.Error as exc:
            logger.warning("Could not persist performance snapshot: %s", exc)

    # ------------------------------------------------------------------
    # DB helper
    # ------------------------------------------------------------------

    def _load_benchmark_returns(self, benchmark: str, n: int) -> np.ndarray:
        """
        Attempt to load the last ``n`` daily percentage returns for the
        named benchmark from ``benchmark_prices`` table.
        Falls back to zeros if the table does not exist.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT pct_change
                FROM benchmark_prices
                WHERE benchmark = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (benchmark, n),
            ).fetchall()
            if rows:
                arr = np.array([float(r[0]) for r in reversed(rows)], dtype=float)
                return arr
        except sqlite3.OperationalError:
            pass
        return np.zeros(n)

    # ------------------------------------------------------------------
    # Convenience reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """
        Return a snapshot of all current metrics as a plain dict.

        Useful for logging or passing to external dashboards.

        Returns
        -------
        dict with keys: equity, running_sharpe, running_drawdown,
        win_rate, calmar, expectancy, trade_count, max_dd.
        """
        return {
            "equity": self._equity,
            "peak_equity": self._peak_equity,
            "trade_count": self._trade_count,
            "running_sharpe": self.running_sharpe(),
            "running_drawdown": self.running_drawdown(),
            "max_drawdown": self._max_dd,
            "win_rate": self.running_win_rate(),
            "calmar": self.running_calmar(),
            "expectancy": self.expectancy(),
        }

    def reset(self, initial_equity: float = 100_000.0) -> None:
        """
        Reset all accumulators and buffers.

        Useful for paper-trading restarts or unit tests.

        Parameters
        ----------
        initial_equity : float
        """
        self._equity = initial_equity
        self._peak_equity = initial_equity
        self._trade_count = 0
        self._max_dd = 0.0
        self._returns.clear()
        self._pnl_window.clear()
        self._win_buffer.clear()
        self._wins_for_expectancy.clear()
        self._losses_for_expectancy.clear()
        self._daily_pnl.clear()
        self._last_snapshot_ts = time.monotonic()
        logger.info("PerformanceTracker reset. initial_equity=%.2f", initial_equity)
