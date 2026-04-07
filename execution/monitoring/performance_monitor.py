"""
Live performance monitoring for SRFM.
Tracks NAV, trade statistics, rolling Sharpe/volatility, win rate, and
streak metrics.  Issues alerts when performance degrades below thresholds.
"""

from __future__ import annotations

import logging
import math
import statistics
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Annualization factor for 15-minute bars:  252 trading days * 26 bars/day
_BARS_PER_YEAR = 252 * 26  # = 6552


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """A completed (round-trip or one-way) trade for performance tracking."""

    trade_id: str
    symbol: str
    side: str               # "buy" | "sell"
    qty: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    strategy_id: str
    pnl: float              # realized P&L in account currency
    commission: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.pnl - self.commission

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0.0

    @property
    def duration_bars(self) -> float:
        """Duration expressed in 15-minute bars."""
        delta_seconds = (self.exit_time - self.entry_time).total_seconds()
        return delta_seconds / 900.0  # 900 seconds per 15-min bar


@dataclass
class NAVPoint:
    """A single NAV observation."""

    nav: float
    timestamp: datetime


@dataclass
class DegradationAlert:
    """Alert produced by PerformanceDegradationAlert."""

    alert_id: str
    metric: str
    value: float
    threshold: float
    severity: str           # "WARNING" | "CRITICAL"
    action_required: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""


# ---------------------------------------------------------------------------
# Core monitor
# ---------------------------------------------------------------------------

class LivePerformanceMonitor:
    """
    Real-time performance metrics computed over rolling windows of trades
    and NAV observations.

    Thread-safe -- all public methods acquire self._lock.
    """

    def __init__(
        self,
        max_trades: int = 1000,
        max_nav_points: int = 10000,
    ) -> None:
        self._lock = threading.Lock()
        self._trades: Deque[TradeRecord] = deque(maxlen=max_trades)
        self._nav_points: Deque[NAVPoint] = deque(maxlen=max_nav_points)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def record_trade(self, trade: TradeRecord) -> None:
        """Add a completed trade to the performance history."""
        with self._lock:
            self._trades.append(trade)
            logger.debug(
                "Trade recorded: %s %s %.2f pnl=%.4f",
                trade.trade_id, trade.symbol, trade.qty, trade.net_pnl,
            )

    def update_nav(self, nav: float, timestamp: Optional[datetime] = None) -> None:
        """Record a NAV observation."""
        ts = timestamp or datetime.now(timezone.utc)
        with self._lock:
            self._nav_points.append(NAVPoint(nav=nav, timestamp=ts))

    # ------------------------------------------------------------------
    # Rolling return / risk metrics
    # ------------------------------------------------------------------

    def rolling_sharpe(self, window_bars: int = 96) -> float:
        """
        Annualized Sharpe ratio over the last window_bars NAV observations.
        Returns 0.0 if there are fewer than 3 data points.
        Assumes risk-free rate of zero.
        """
        with self._lock:
            points = list(self._nav_points)[-window_bars - 1:]

        if len(points) < 3:
            return 0.0

        returns = _bar_returns(points)
        if not returns:
            return 0.0

        mean_r = statistics.mean(returns)
        if len(returns) < 2:
            return 0.0
        std_r = statistics.stdev(returns)
        if std_r < 1e-12:
            return 0.0

        annualized_mean = mean_r * _BARS_PER_YEAR
        annualized_std = std_r * math.sqrt(_BARS_PER_YEAR)
        return annualized_mean / annualized_std

    def rolling_vol(self, window_bars: int = 20) -> float:
        """
        Annualized volatility (std dev of bar returns) over the last
        window_bars NAV observations.  Returns 0.0 if insufficient data.
        """
        with self._lock:
            points = list(self._nav_points)[-window_bars - 1:]

        if len(points) < 3:
            return 0.0

        returns = _bar_returns(points)
        if len(returns) < 2:
            return 0.0

        std_r = statistics.stdev(returns)
        return std_r * math.sqrt(_BARS_PER_YEAR)

    # ------------------------------------------------------------------
    # Trade-level metrics
    # ------------------------------------------------------------------

    def win_rate(self, window_trades: int = 50) -> float:
        """Fraction of the last window_trades that were winning trades."""
        with self._lock:
            recent = list(self._trades)[-window_trades:]
        if not recent:
            return 0.0
        wins = sum(1 for t in recent if t.is_win)
        return wins / len(recent)

    def avg_trade_duration(self, window: int = 50) -> float:
        """Average duration in 15-minute bars of the last window trades."""
        with self._lock:
            recent = list(self._trades)[-window:]
        if not recent:
            return 0.0
        return statistics.mean(t.duration_bars for t in recent)

    def total_pnl(self, window_trades: Optional[int] = None) -> float:
        with self._lock:
            trades = list(self._trades)
        if window_trades is not None:
            trades = trades[-window_trades:]
        return sum(t.net_pnl for t in trades)

    def avg_win(self, window_trades: int = 50) -> float:
        with self._lock:
            recent = list(self._trades)[-window_trades:]
        wins = [t.net_pnl for t in recent if t.is_win]
        return statistics.mean(wins) if wins else 0.0

    def avg_loss(self, window_trades: int = 50) -> float:
        with self._lock:
            recent = list(self._trades)[-window_trades:]
        losses = [t.net_pnl for t in recent if not t.is_win]
        return statistics.mean(losses) if losses else 0.0

    def profit_factor(self, window_trades: int = 50) -> float:
        """Gross profit / gross loss.  Returns 0.0 if no losses."""
        with self._lock:
            recent = list(self._trades)[-window_trades:]
        gross_profit = sum(t.net_pnl for t in recent if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in recent if t.net_pnl < 0))
        if gross_loss < 1e-9:
            return 0.0
        return gross_profit / gross_loss

    # ------------------------------------------------------------------
    # Time-of-day / day-of-week analytics
    # ------------------------------------------------------------------

    def pnl_by_hour(self) -> Dict[int, float]:
        """
        Aggregate net P&L by hour-of-day (0-23) using trade entry_time.
        Returns a dict with only hours that have at least one trade.
        """
        with self._lock:
            trades = list(self._trades)
        result: Dict[int, float] = {}
        for t in trades:
            hour = t.entry_time.hour
            result[hour] = result.get(hour, 0.0) + t.net_pnl
        return result

    def pnl_by_day_of_week(self) -> Dict[int, float]:
        """
        Aggregate net P&L by ISO day-of-week (1=Monday, 7=Sunday).
        Returns a dict with only days that have at least one trade.
        """
        with self._lock:
            trades = list(self._trades)
        result: Dict[int, float] = {}
        for t in trades:
            dow = t.entry_time.isoweekday()  # 1=Mon ... 7=Sun
            result[dow] = result.get(dow, 0.0) + t.net_pnl
        return result

    # ------------------------------------------------------------------
    # Streak
    # ------------------------------------------------------------------

    def current_streak(self) -> Tuple[str, int]:
        """
        Return the current win/loss streak as ("win", n) or ("loss", n).
        Returns ("none", 0) if there are no trades.
        """
        with self._lock:
            trades = list(self._trades)

        if not trades:
            return ("none", 0)

        kind = "win" if trades[-1].is_win else "loss"
        count = 0
        for t in reversed(trades):
            if t.is_win == (kind == "win"):
                count += 1
            else:
                break
        return (kind, count)

    def max_consecutive_losses(self, window_trades: int = 100) -> int:
        """Largest consecutive loss streak in the last window_trades trades."""
        with self._lock:
            recent = list(self._trades)[-window_trades:]
        max_streak = 0
        current = 0
        for t in recent:
            if not t.is_win:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    # ------------------------------------------------------------------
    # Convenience snapshots
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict:
        """Return a dict of key metrics for dashboards / logging."""
        return {
            "rolling_sharpe_96": self.rolling_sharpe(96),
            "rolling_vol_20": self.rolling_vol(20),
            "win_rate_50": self.win_rate(50),
            "avg_duration_bars_50": self.avg_trade_duration(50),
            "profit_factor_50": self.profit_factor(50),
            "total_pnl": self.total_pnl(),
            "streak": self.current_streak(),
            "trade_count": len(self._trades),
            "nav_points": len(self._nav_points),
        }


# ---------------------------------------------------------------------------
# Degradation alert checker
# ---------------------------------------------------------------------------

class PerformanceDegradationAlert:
    """
    Evaluates a LivePerformanceMonitor and returns degradation alerts.

    Thresholds:
    - rolling_sharpe < -0.5 -> CRITICAL
    - win_rate < 0.40       -> WARNING
    - consecutive losses > 5 -> CRITICAL
    """

    SHARPE_THRESHOLD: float = -0.5
    WIN_RATE_THRESHOLD: float = 0.40
    CONSECUTIVE_LOSS_THRESHOLD: int = 5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counter: int = 0

    def check(self, monitor: LivePerformanceMonitor) -> List[DegradationAlert]:
        """Run all degradation checks and return any triggered alerts."""
        alerts: List[DegradationAlert] = []

        sharpe = monitor.rolling_sharpe(96)
        if sharpe < self.SHARPE_THRESHOLD:
            alerts.append(DegradationAlert(
                alert_id=self._next_id(),
                metric="rolling_sharpe_96",
                value=sharpe,
                threshold=self.SHARPE_THRESHOLD,
                severity="CRITICAL",
                action_required="Review strategy -- consider pausing live trading",
                message=(
                    f"Rolling Sharpe {sharpe:.3f} is below threshold "
                    f"{self.SHARPE_THRESHOLD:.1f}"
                ),
            ))

        win_rate = monitor.win_rate(50)
        # only alert if we have meaningful trade count
        if len(monitor._trades) >= 10 and win_rate < self.WIN_RATE_THRESHOLD:
            alerts.append(DegradationAlert(
                alert_id=self._next_id(),
                metric="win_rate_50",
                value=win_rate,
                threshold=self.WIN_RATE_THRESHOLD,
                severity="WARNING",
                action_required="Investigate signal quality -- reduce position sizing",
                message=(
                    f"Win rate {win_rate:.1%} is below threshold "
                    f"{self.WIN_RATE_THRESHOLD:.0%} over last 50 trades"
                ),
            ))

        streak_kind, streak_count = monitor.current_streak()
        if streak_kind == "loss" and streak_count > self.CONSECUTIVE_LOSS_THRESHOLD:
            alerts.append(DegradationAlert(
                alert_id=self._next_id(),
                metric="consecutive_losses",
                value=float(streak_count),
                threshold=float(self.CONSECUTIVE_LOSS_THRESHOLD),
                severity="CRITICAL",
                action_required="Halt strategy and perform immediate review",
                message=(
                    f"Consecutive loss streak of {streak_count} exceeds "
                    f"threshold of {self.CONSECUTIVE_LOSS_THRESHOLD}"
                ),
            ))

        if alerts:
            logger.warning(
                "PerformanceDegradationAlert: %d alert(s) triggered", len(alerts)
            )
        return alerts

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"PDA-{self._counter:06d}"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _bar_returns(points: List[NAVPoint]) -> List[float]:
    """Compute simple bar-over-bar returns from a list of NAVPoint."""
    returns: List[float] = []
    for i in range(1, len(points)):
        prev = points[i - 1].nav
        curr = points[i].nav
        if abs(prev) < 1e-12:
            continue
        returns.append((curr - prev) / prev)
    return returns
