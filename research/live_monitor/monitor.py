"""
monitor.py — Live Trader Monitor
=================================

Polls live_trades.db SQLite every 60 seconds and provides:
  - Current positions
  - Daily / cumulative P&L
  - Rolling performance metrics (Sharpe, drawdown, win rate)
  - Health checks with alerting

Database schema assumed
-----------------------
  trades (order_id TEXT, sym TEXT, side TEXT, fill_price REAL, qty REAL,
          fill_time TEXT, pnl REAL, status TEXT)
  positions (sym TEXT, qty REAL, avg_entry REAL, current_price REAL,
             unrealized_pnl REAL, bh_active INTEGER, last_updated TEXT)
  equity_curve (ts TEXT, equity REAL)
  order_failures (order_id TEXT, sym TEXT, reason TEXT, ts TEXT, code TEXT)
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Current open position."""
    sym: str
    qty: float
    avg_entry: float
    current_price: float
    unrealized_pnl: float
    bh_active: bool           # Bollinger–Hurst signal active
    notional: float = field(init=False)
    pnl_pct: float = field(init=False)

    def __post_init__(self) -> None:
        self.notional = abs(self.qty * self.current_price)
        direction = 1.0 if self.qty > 0 else -1.0
        self.pnl_pct = (
            direction * (self.current_price - self.avg_entry) / self.avg_entry * 100
            if self.avg_entry > 0
            else 0.0
        )


@dataclass
class LiveMetrics:
    """Rolling performance metrics."""
    sharpe_ratio: float
    max_drawdown_pct: float
    current_drawdown_pct: float
    win_rate: float
    n_trades: int
    total_pnl: float
    daily_pnl: float
    weekly_pnl: float
    avg_trade_pnl: float
    best_trade_pnl: float
    worst_trade_pnl: float
    avg_holding_minutes: float
    order_failure_rate: float
    last_trade_time: datetime | None
    lookback_days: int


@dataclass
class HealthReport:
    """Health status of the live trader."""
    status: str                      # "OK" | "WARN" | "CRITICAL"
    alerts: list[str]
    last_trade_time: datetime | None
    current_drawdown_pct: float
    order_failure_rate: float
    is_market_hours: bool
    equity: float
    n_open_positions: int
    checked_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_healthy(self) -> bool:
        return self.status == "OK"


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------

class LiveTraderMonitor:
    """
    Monitors the live Alpaca trader by polling live_trades.db.

    Parameters
    ----------
    db_path : str | Path
        Path to live_trades.db SQLite file.
    poll_interval_seconds : int
        How often to re-read the database (default 60s).
    """

    def __init__(
        self,
        db_path: str | Path,
        poll_interval_seconds: int = 60,
    ) -> None:
        self.db_path = Path(db_path)
        self.poll_interval_seconds = poll_interval_seconds
        self._last_polled: datetime | None = None
        self._cache: dict[str, Any] = {}

        if not self.db_path.exists():
            logger.warning("DB not found at %s — will retry on first poll", self.db_path)

    # -----------------------------------------------------------------------
    # DB helpers
    # -----------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"live_trades.db not found: {self.db_path}")
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        try:
            conn = self._connect()
            df = pd.read_sql_query(sql, conn, params=params)
            conn.close()
            return df
        except Exception as exc:
            logger.error("DB query failed: %s", exc)
            return pd.DataFrame()

    def _table_exists(self, table: str) -> bool:
        df = self._query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        return len(df) > 0

    # -----------------------------------------------------------------------
    # Positions
    # -----------------------------------------------------------------------

    def get_current_positions(self) -> list[Position]:
        """
        Return all current open positions from the positions table.

        Returns
        -------
        list[Position]
            Empty list if table missing or no open positions.
        """
        if not self._table_exists("positions"):
            logger.warning("positions table not found in DB")
            return []

        df = self._query(
            "SELECT sym, qty, avg_entry, current_price, unrealized_pnl, bh_active "
            "FROM positions WHERE qty != 0"
        )

        if df.empty:
            return []

        positions = []
        for _, row in df.iterrows():
            try:
                pos = Position(
                    sym=str(row["sym"]),
                    qty=float(row["qty"]),
                    avg_entry=float(row["avg_entry"]),
                    current_price=float(row["current_price"]),
                    unrealized_pnl=float(row["unrealized_pnl"]),
                    bh_active=bool(int(row.get("bh_active", 0))),
                )
                positions.append(pos)
            except Exception as exc:
                logger.warning("Skipping malformed position row: %s", exc)

        return positions

    # -----------------------------------------------------------------------
    # P&L
    # -----------------------------------------------------------------------

    def get_todays_pnl(self) -> dict[str, float]:
        """
        Return realized P&L per symbol for today (UTC).

        Returns
        -------
        dict[sym, pnl_usd]
        """
        if not self._table_exists("trades"):
            return {}

        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_str = today_start.strftime("%Y-%m-%d")

        df = self._query(
            "SELECT sym, SUM(pnl) as sym_pnl FROM trades "
            "WHERE DATE(fill_time) = ? GROUP BY sym",
            (today_str,),
        )

        if df.empty:
            return {}

        return {str(row["sym"]): float(row["sym_pnl"]) for _, row in df.iterrows()}

    def get_cumulative_pnl(self, days: int = 30) -> pd.Series:
        """
        Return cumulative P&L as a daily time series over the last N days.

        Parameters
        ----------
        days : int
            Lookback in calendar days.

        Returns
        -------
        pd.Series
            Index: date, Values: cumulative P&L in USD.
        """
        if not self._table_exists("trades"):
            return pd.Series(dtype=float)

        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        df = self._query(
            "SELECT DATE(fill_time) as trade_date, SUM(pnl) as daily_pnl "
            "FROM trades WHERE DATE(fill_time) >= ? "
            "GROUP BY trade_date ORDER BY trade_date",
            (cutoff,),
        )

        if df.empty:
            return pd.Series(dtype=float)

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()
        cumulative = df["daily_pnl"].cumsum()
        return cumulative

    def get_equity_curve(self, days: int = 30) -> pd.Series:
        """
        Return equity curve from equity_curve table.

        Falls back to cumulative P&L + initial equity if table missing.
        """
        if self._table_exists("equity_curve"):
            cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            df = self._query(
                "SELECT ts, equity FROM equity_curve WHERE DATE(ts) >= ? ORDER BY ts",
                (cutoff,),
            )
            if not df.empty:
                df["ts"] = pd.to_datetime(df["ts"])
                return df.set_index("ts")["equity"]

        # Fallback: reconstruct from trades
        logger.warning("equity_curve table missing — reconstructing from trade P&L")
        cum_pnl = self.get_cumulative_pnl(days)
        if cum_pnl.empty:
            return pd.Series(dtype=float)
        initial_equity = 1_000_000.0  # $1M paper account
        return initial_equity + cum_pnl

    # -----------------------------------------------------------------------
    # Live metrics
    # -----------------------------------------------------------------------

    def get_live_metrics(
        self,
        lookback_days: int = 30,
        n_recent_trades: int = 100,
    ) -> LiveMetrics:
        """
        Compute rolling performance metrics.

        Parameters
        ----------
        lookback_days : int
        n_recent_trades : int

        Returns
        -------
        LiveMetrics
        """
        equity_curve = self.get_equity_curve(lookback_days)
        trades_df = self._get_recent_trades(lookback_days)

        # Sharpe
        sharpe = self._compute_sharpe(equity_curve)

        # Drawdown
        max_dd, current_dd = self._compute_drawdown(equity_curve)

        # Win rate and trade stats
        n_trades = 0
        win_rate = 0.0
        avg_pnl = 0.0
        best_pnl = 0.0
        worst_pnl = 0.0
        total_pnl = 0.0
        last_trade_time: datetime | None = None
        avg_holding_minutes = 0.0

        if not trades_df.empty and "pnl" in trades_df.columns:
            pnls = trades_df["pnl"].dropna().values.astype(float)
            n_trades = len(pnls)
            total_pnl = float(pnls.sum())
            win_rate = float((pnls > 0).mean()) if n_trades > 0 else 0.0
            avg_pnl = float(pnls.mean()) if n_trades > 0 else 0.0
            best_pnl = float(pnls.max()) if n_trades > 0 else 0.0
            worst_pnl = float(pnls.min()) if n_trades > 0 else 0.0

            if "fill_time" in trades_df.columns:
                fill_times = pd.to_datetime(trades_df["fill_time"]).dropna()
                if not fill_times.empty:
                    last_trade_time = fill_times.max().to_pydatetime()

        # Daily / weekly PnL
        today_pnl = sum(self.get_todays_pnl().values())
        weekly_cutoff = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        weekly_df = self._query(
            "SELECT SUM(pnl) as total FROM trades WHERE DATE(fill_time) >= ?",
            (weekly_cutoff,),
        )
        weekly_pnl = float(weekly_df.iloc[0]["total"]) if not weekly_df.empty and weekly_df.iloc[0]["total"] is not None else 0.0

        # Order failure rate
        failure_rate = self._compute_failure_rate()

        return LiveMetrics(
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            current_drawdown_pct=current_dd,
            win_rate=win_rate,
            n_trades=n_trades,
            total_pnl=total_pnl,
            daily_pnl=today_pnl,
            weekly_pnl=weekly_pnl,
            avg_trade_pnl=avg_pnl,
            best_trade_pnl=best_pnl,
            worst_trade_pnl=worst_pnl,
            avg_holding_minutes=avg_holding_minutes,
            order_failure_rate=failure_rate,
            last_trade_time=last_trade_time,
            lookback_days=lookback_days,
        )

    # -----------------------------------------------------------------------
    # Health check
    # -----------------------------------------------------------------------

    def check_health(
        self,
        live_metrics: LiveMetrics | None = None,
        expected_trades_per_day: int = 5,
        max_drawdown_threshold: float = 0.20,
        max_failure_rate: float = 0.10,
    ) -> HealthReport:
        """
        Run health checks and return a HealthReport.

        Checks
        ------
        1. No trades in last 24h during market hours → WARN
        2. Drawdown > max_drawdown_threshold → CRITICAL
        3. Order failure rate > max_failure_rate → WARN
        4. Zero open positions during market hours → INFO

        Parameters
        ----------
        live_metrics : LiveMetrics, optional
            Pre-computed. Will call get_live_metrics() if None.
        expected_trades_per_day : int
        max_drawdown_threshold : float
            0.20 = 20%
        max_failure_rate : float

        Returns
        -------
        HealthReport
        """
        if live_metrics is None:
            live_metrics = self.get_live_metrics()

        alerts: list[str] = []
        status = "OK"

        now_utc = datetime.utcnow()
        is_market_hours = self._is_equity_market_hours(now_utc)

        # Check 1: No recent trades
        if live_metrics.last_trade_time:
            hours_since_last = (now_utc - live_metrics.last_trade_time).total_seconds() / 3600
            if hours_since_last > 24 and is_market_hours:
                alerts.append(
                    f"WARN: No trades in last {hours_since_last:.0f}h during market hours"
                )
                status = "WARN"
        else:
            if is_market_hours:
                alerts.append("WARN: No trades on record and market is open")
                status = "WARN"

        # Check 2: Drawdown
        dd = live_metrics.current_drawdown_pct / 100  # convert pct to fraction
        if dd > max_drawdown_threshold:
            alerts.append(
                f"CRITICAL: Current drawdown {live_metrics.current_drawdown_pct:.1f}% "
                f"exceeds threshold {max_drawdown_threshold*100:.0f}%"
            )
            status = "CRITICAL"

        # Check 3: Order failure rate
        if live_metrics.order_failure_rate > max_failure_rate:
            alerts.append(
                f"WARN: Order failure rate {live_metrics.order_failure_rate*100:.1f}% "
                f"exceeds threshold {max_failure_rate*100:.0f}%"
            )
            if status == "OK":
                status = "WARN"

        # Check 4: Positions
        positions = self.get_current_positions()
        n_positions = len(positions)
        if n_positions == 0 and is_market_hours:
            alerts.append("INFO: No open positions during market hours")

        # Compute current equity
        eq_curve = self.get_equity_curve(days=1)
        equity = float(eq_curve.iloc[-1]) if not eq_curve.empty else 1_000_000.0

        if not alerts:
            alerts.append("All systems nominal")

        return HealthReport(
            status=status,
            alerts=alerts,
            last_trade_time=live_metrics.last_trade_time,
            current_drawdown_pct=live_metrics.current_drawdown_pct,
            order_failure_rate=live_metrics.order_failure_rate,
            is_market_hours=is_market_hours,
            equity=equity,
            n_open_positions=n_positions,
            checked_at=now_utc,
        )

    # -----------------------------------------------------------------------
    # Continuous polling
    # -----------------------------------------------------------------------

    def poll_forever(
        self,
        callback: Any | None = None,
        poll_interval: int | None = None,
    ) -> None:
        """
        Poll the database continuously and call callback(HealthReport) each tick.

        Parameters
        ----------
        callback : callable(HealthReport) | None
            If None, logs health status.
        poll_interval : int, optional
            Override self.poll_interval_seconds.
        """
        interval = poll_interval or self.poll_interval_seconds
        logger.info("Starting LiveTraderMonitor polling every %ds", interval)

        while True:
            try:
                health = self.check_health()
                if callback:
                    callback(health)
                else:
                    status_sym = {"OK": "✓", "WARN": "⚠", "CRITICAL": "✗"}.get(health.status, "?")
                    logger.info(
                        "[%s %s] Equity=$%.0f  DD=%.1f%%  Positions=%d  Alerts: %s",
                        status_sym,
                        health.status,
                        health.equity,
                        health.current_drawdown_pct,
                        health.n_open_positions,
                        " | ".join(health.alerts),
                    )
            except Exception as exc:
                logger.error("Monitor poll error: %s", exc, exc_info=True)

            time.sleep(interval)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_recent_trades(self, days: int) -> pd.DataFrame:
        """Load recent trades from DB."""
        if not self._table_exists("trades"):
            return pd.DataFrame()
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self._query(
            "SELECT * FROM trades WHERE DATE(fill_time) >= ? ORDER BY fill_time DESC",
            (cutoff,),
        )

    @staticmethod
    def _compute_sharpe(equity_curve: pd.Series, annualization: float = 252.0) -> float:
        """Compute annualised Sharpe ratio from equity curve."""
        if equity_curve.empty or len(equity_curve) < 2:
            return 0.0
        returns = equity_curve.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * math.sqrt(annualization))

    @staticmethod
    def _compute_drawdown(equity_curve: pd.Series) -> tuple[float, float]:
        """
        Compute max and current drawdown.

        Returns
        -------
        (max_drawdown_pct, current_drawdown_pct) — both as percentages (0–100).
        """
        if equity_curve.empty:
            return 0.0, 0.0
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        max_dd = float(drawdown.min())
        current_dd = float(drawdown.iloc[-1])
        return abs(max_dd), abs(current_dd)

    def _compute_failure_rate(self) -> float:
        """Compute fraction of orders that failed."""
        if not self._table_exists("order_failures"):
            return 0.0

        total_df = self._query("SELECT COUNT(*) as n FROM trades")
        fail_df = self._query("SELECT COUNT(*) as n FROM order_failures")

        total = int(total_df.iloc[0]["n"]) if not total_df.empty else 0
        failed = int(fail_df.iloc[0]["n"]) if not fail_df.empty else 0

        if total == 0:
            return 0.0
        return failed / (total + failed)

    @staticmethod
    def _is_equity_market_hours(dt: datetime) -> bool:
        """
        Rough check: is this a weekday between 9:30–16:00 ET?
        (UTC-5 in EST, UTC-4 in EDT — use UTC-4 as approximation)
        """
        dt_et = dt - timedelta(hours=4)  # approximate ET
        if dt_et.weekday() >= 5:          # Saturday/Sunday
            return False
        market_open = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= dt_et <= market_close
