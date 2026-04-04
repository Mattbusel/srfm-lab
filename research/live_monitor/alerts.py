"""
alerts.py — Configurable Alert System for Live Trader
======================================================

Provides:
  - Alert dataclass with severity levels INFO / WARN / CRITICAL
  - AlertSystem class with per-check threshold configuration
  - Individual check methods returning Alert | None
  - Convenience method to run all checks at once
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

class Severity:
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"

    @staticmethod
    def rank(severity: str) -> int:
        return {"INFO": 0, "WARN": 1, "CRITICAL": 2}.get(severity, -1)


@dataclass
class Alert:
    """A single alert produced by the AlertSystem."""
    severity: str           # INFO | WARN | CRITICAL
    message: str
    timestamp: datetime
    value: float            # The numeric value that triggered the alert
    threshold: float        # The threshold that was violated
    check_name: str         # Which check produced this
    sym: str = ""           # Optional: which symbol

    def __str__(self) -> str:
        sym_str = f" [{self.sym}]" if self.sym else ""
        return (
            f"[{self.severity}]{sym_str} {self.message} "
            f"(value={self.value:.4g}, threshold={self.threshold:.4g})"
        )

    def is_critical(self) -> bool:
        return self.severity == Severity.CRITICAL

    def is_warn_or_above(self) -> bool:
        return Severity.rank(self.severity) >= Severity.rank(Severity.WARN)


@dataclass
class AlertConfig:
    """Configuration thresholds for AlertSystem."""
    # Drawdown
    drawdown_warn_pct: float = 0.10        # 10%
    drawdown_critical_pct: float = 0.20    # 20%

    # Concentration
    max_position_frac: float = 0.75        # single position max fraction

    # Trade frequency (per day)
    min_trades_per_day: float = 1.0
    max_trades_per_day: float = 100.0

    # Losing streak
    max_losing_streak: int = 10

    # Order failures
    max_failure_rate: float = 0.05         # 5%

    # Latency
    max_p95_latency_ms: float = 2000.0

    # Equity
    min_equity: float = 500_000.0          # Stop-loss at 50% of $1M

    # Time without trades (market hours)
    max_idle_hours: float = 24.0


# ---------------------------------------------------------------------------
# AlertSystem
# ---------------------------------------------------------------------------

class AlertSystem:
    """
    Configurable alert system that checks various health metrics.

    Parameters
    ----------
    config : AlertConfig, optional
        Override default thresholds.
    """

    def __init__(self, config: AlertConfig | None = None) -> None:
        self.config = config or AlertConfig()
        self._alert_history: list[Alert] = []

    # -----------------------------------------------------------------------
    # Individual checks
    # -----------------------------------------------------------------------

    def check_drawdown(
        self,
        equity_curve: pd.Series,
        warn_threshold: float | None = None,
        critical_threshold: float | None = None,
    ) -> Alert | None:
        """
        Check if drawdown exceeds warn or critical threshold.

        Parameters
        ----------
        equity_curve : pd.Series
            Time-indexed equity values.
        warn_threshold : float, optional
            Fraction (e.g. 0.10 = 10%). Defaults to config.drawdown_warn_pct.
        critical_threshold : float, optional
            Fraction. Defaults to config.drawdown_critical_pct.

        Returns
        -------
        Alert or None
        """
        warn_th = warn_threshold if warn_threshold is not None else self.config.drawdown_warn_pct
        crit_th = critical_threshold if critical_threshold is not None else self.config.drawdown_critical_pct

        if equity_curve.empty:
            return None

        rolling_max = equity_curve.cummax()
        current_dd = float((equity_curve.iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1])
        dd_abs = abs(current_dd)

        if dd_abs >= crit_th:
            alert = Alert(
                severity=Severity.CRITICAL,
                message=f"Drawdown {dd_abs*100:.1f}% exceeds critical threshold {crit_th*100:.0f}%",
                timestamp=datetime.utcnow(),
                value=dd_abs,
                threshold=crit_th,
                check_name="drawdown",
            )
            self._alert_history.append(alert)
            return alert

        if dd_abs >= warn_th:
            alert = Alert(
                severity=Severity.WARN,
                message=f"Drawdown {dd_abs*100:.1f}% exceeds warn threshold {warn_th*100:.0f}%",
                timestamp=datetime.utcnow(),
                value=dd_abs,
                threshold=warn_th,
                check_name="drawdown",
            )
            self._alert_history.append(alert)
            return alert

        return None

    def check_concentration(
        self,
        positions: list[Any],  # list[Position] from monitor
        max_frac: float | None = None,
    ) -> Alert | None:
        """
        Alert if any single position exceeds max_frac of total portfolio notional.

        Parameters
        ----------
        positions : list[Position]
        max_frac : float, optional
            Defaults to config.max_position_frac.

        Returns
        -------
        Alert or None
        """
        threshold = max_frac if max_frac is not None else self.config.max_position_frac

        if not positions:
            return None

        total_notional = sum(getattr(p, "notional", 0.0) for p in positions)
        if total_notional <= 0:
            return None

        worst_sym = ""
        worst_frac = 0.0
        for pos in positions:
            notional = getattr(pos, "notional", 0.0)
            frac = notional / total_notional
            if frac > worst_frac:
                worst_frac = frac
                worst_sym = getattr(pos, "sym", "?")

        if worst_frac > threshold:
            alert = Alert(
                severity=Severity.CRITICAL,
                message=(
                    f"{worst_sym} concentration {worst_frac*100:.1f}% "
                    f"exceeds limit {threshold*100:.0f}%"
                ),
                timestamp=datetime.utcnow(),
                value=worst_frac,
                threshold=threshold,
                check_name="concentration",
                sym=worst_sym,
            )
            self._alert_history.append(alert)
            return alert

        return None

    def check_trade_frequency(
        self,
        trades: pd.DataFrame,
        min_per_day: float | None = None,
        max_per_day: float | None = None,
        lookback_days: int = 7,
    ) -> Alert | None:
        """
        Alert if average daily trade frequency is outside expected range.

        Parameters
        ----------
        trades : pd.DataFrame
            Must have 'fill_time' column.
        min_per_day : float, optional
        max_per_day : float, optional
        lookback_days : int

        Returns
        -------
        Alert or None
        """
        lo = min_per_day if min_per_day is not None else self.config.min_trades_per_day
        hi = max_per_day if max_per_day is not None else self.config.max_trades_per_day

        if trades.empty or "fill_time" not in trades.columns:
            # No trades at all
            alert = Alert(
                severity=Severity.WARN,
                message=f"No trades recorded in last {lookback_days} days",
                timestamp=datetime.utcnow(),
                value=0.0,
                threshold=lo,
                check_name="trade_frequency",
            )
            self._alert_history.append(alert)
            return alert

        df = trades.copy()
        df["fill_time"] = pd.to_datetime(df["fill_time"])
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        recent = df[df["fill_time"] >= cutoff]
        n_days = max(lookback_days, 1)
        avg_per_day = len(recent) / n_days

        if avg_per_day < lo:
            alert = Alert(
                severity=Severity.WARN,
                message=(
                    f"Avg trades/day {avg_per_day:.1f} below minimum {lo:.0f} "
                    f"(lookback {lookback_days}d)"
                ),
                timestamp=datetime.utcnow(),
                value=avg_per_day,
                threshold=lo,
                check_name="trade_frequency_min",
            )
            self._alert_history.append(alert)
            return alert

        if avg_per_day > hi:
            alert = Alert(
                severity=Severity.WARN,
                message=(
                    f"Avg trades/day {avg_per_day:.1f} above maximum {hi:.0f} "
                    f"(possible runaway trading)"
                ),
                timestamp=datetime.utcnow(),
                value=avg_per_day,
                threshold=hi,
                check_name="trade_frequency_max",
            )
            self._alert_history.append(alert)
            return alert

        return None

    def check_losing_streak(
        self,
        trades: pd.DataFrame,
        max_streak: int | None = None,
    ) -> Alert | None:
        """
        Alert if the current losing streak exceeds max_streak consecutive trades.

        Parameters
        ----------
        trades : pd.DataFrame
            Must have 'pnl' and 'fill_time' columns.
        max_streak : int, optional

        Returns
        -------
        Alert or None
        """
        threshold = max_streak if max_streak is not None else self.config.max_losing_streak

        if trades.empty or "pnl" not in trades.columns:
            return None

        df = trades.copy()
        if "fill_time" in df.columns:
            df["fill_time"] = pd.to_datetime(df["fill_time"])
            df = df.sort_values("fill_time")

        pnls = df["pnl"].dropna().values.astype(float)
        if len(pnls) == 0:
            return None

        # Compute current streak from end of series
        current_streak = 0
        max_obs_streak = 0
        current_run = 0

        for pnl in reversed(pnls):
            if pnl < 0:
                current_run += 1
                if current_streak == 0:
                    current_streak = current_run
            else:
                if current_streak == 0:
                    current_streak = 0  # already broke
                break

        # Also compute overall max streak
        run = 0
        for pnl in pnls:
            if pnl < 0:
                run += 1
                max_obs_streak = max(max_obs_streak, run)
            else:
                run = 0

        if current_streak >= threshold:
            alert = Alert(
                severity=Severity.CRITICAL if current_streak >= threshold * 1.5 else Severity.WARN,
                message=(
                    f"Losing streak of {current_streak} consecutive trades "
                    f"(threshold: {threshold})"
                ),
                timestamp=datetime.utcnow(),
                value=float(current_streak),
                threshold=float(threshold),
                check_name="losing_streak",
            )
            self._alert_history.append(alert)
            return alert

        return None

    def check_order_failures(
        self,
        failures: pd.DataFrame | dict[str, int],
        total_orders: int | None = None,
        max_rate: float | None = None,
    ) -> Alert | None:
        """
        Alert if order failure rate exceeds threshold.

        Parameters
        ----------
        failures : pd.DataFrame | dict[str, int]
            Either a DataFrame with order failure records,
            or a dict of {reason: count}.
        total_orders : int, optional
            Total number of order attempts (used if failures is a dict).
        max_rate : float, optional

        Returns
        -------
        Alert or None
        """
        threshold = max_rate if max_rate is not None else self.config.max_failure_rate

        if isinstance(failures, pd.DataFrame):
            n_failures = len(failures)
            if "total_orders" in failures.columns:
                n_total = int(failures["total_orders"].sum())
            elif total_orders is not None:
                n_total = total_orders
            else:
                n_total = n_failures * 10  # rough estimate
        elif isinstance(failures, dict):
            n_failures = sum(failures.values())
            n_total = total_orders if total_orders is not None else n_failures * 10
        else:
            return None

        if n_total == 0:
            return None

        rate = n_failures / (n_total + n_failures)

        if rate > threshold:
            severity = Severity.CRITICAL if rate > threshold * 2 else Severity.WARN
            alert = Alert(
                severity=severity,
                message=(
                    f"Order failure rate {rate*100:.1f}% exceeds threshold "
                    f"{threshold*100:.0f}% ({n_failures} failures / {n_total} total)"
                ),
                timestamp=datetime.utcnow(),
                value=rate,
                threshold=threshold,
                check_name="order_failures",
            )
            self._alert_history.append(alert)
            return alert

        return None

    def check_equity_floor(
        self,
        equity: float,
        min_equity: float | None = None,
    ) -> Alert | None:
        """
        Alert if equity falls below the minimum floor.

        Parameters
        ----------
        equity : float
        min_equity : float, optional

        Returns
        -------
        Alert or None
        """
        floor = min_equity if min_equity is not None else self.config.min_equity

        if equity < floor:
            alert = Alert(
                severity=Severity.CRITICAL,
                message=(
                    f"Equity ${equity:,.0f} below floor ${floor:,.0f}. "
                    f"Consider halting trading."
                ),
                timestamp=datetime.utcnow(),
                value=equity,
                threshold=floor,
                check_name="equity_floor",
            )
            self._alert_history.append(alert)
            return alert

        return None

    def check_idle(
        self,
        last_trade_time: datetime | None,
        max_idle_hours: float | None = None,
    ) -> Alert | None:
        """
        Alert if the system has been idle for too long during market hours.

        Parameters
        ----------
        last_trade_time : datetime | None
        max_idle_hours : float, optional

        Returns
        -------
        Alert or None
        """
        threshold = max_idle_hours if max_idle_hours is not None else self.config.max_idle_hours

        if last_trade_time is None:
            alert = Alert(
                severity=Severity.WARN,
                message="No trade has ever been executed",
                timestamp=datetime.utcnow(),
                value=float("inf"),
                threshold=threshold,
                check_name="idle",
            )
            self._alert_history.append(alert)
            return alert

        idle_hours = (datetime.utcnow() - last_trade_time).total_seconds() / 3600

        if idle_hours > threshold:
            alert = Alert(
                severity=Severity.WARN,
                message=(
                    f"No trades in {idle_hours:.1f}h "
                    f"(threshold: {threshold:.0f}h)"
                ),
                timestamp=datetime.utcnow(),
                value=idle_hours,
                threshold=threshold,
                check_name="idle",
            )
            self._alert_history.append(alert)
            return alert

        return None

    # -----------------------------------------------------------------------
    # Run all checks
    # -----------------------------------------------------------------------

    def run_all_checks(
        self,
        equity_curve: pd.Series | None = None,
        positions: list[Any] | None = None,
        trades: pd.DataFrame | None = None,
        failures: pd.DataFrame | None = None,
        equity: float | None = None,
        last_trade_time: datetime | None = None,
    ) -> list[Alert]:
        """
        Run all configured checks and return a list of triggered alerts.

        Parameters
        ----------
        equity_curve : pd.Series, optional
        positions : list[Position], optional
        trades : pd.DataFrame, optional
        failures : pd.DataFrame, optional
        equity : float, optional
        last_trade_time : datetime, optional

        Returns
        -------
        list[Alert]
            Sorted by severity (CRITICAL first).
        """
        alerts: list[Alert] = []

        if equity_curve is not None and not equity_curve.empty:
            a = self.check_drawdown(equity_curve)
            if a:
                alerts.append(a)

        if positions is not None:
            a = self.check_concentration(positions)
            if a:
                alerts.append(a)

        if trades is not None:
            a = self.check_trade_frequency(trades)
            if a:
                alerts.append(a)

            a = self.check_losing_streak(trades)
            if a:
                alerts.append(a)

        if failures is not None:
            a = self.check_order_failures(failures)
            if a:
                alerts.append(a)

        if equity is not None:
            a = self.check_equity_floor(equity)
            if a:
                alerts.append(a)

        if last_trade_time is not None or last_trade_time == datetime.min:
            a = self.check_idle(last_trade_time)
            if a:
                alerts.append(a)

        # Sort by severity (CRITICAL first)
        alerts.sort(key=lambda a: -Severity.rank(a.severity))
        return alerts

    # -----------------------------------------------------------------------
    # Alert history
    # -----------------------------------------------------------------------

    def get_alert_history(
        self,
        since: datetime | None = None,
        severity: str | None = None,
    ) -> list[Alert]:
        """
        Return historical alerts, optionally filtered.

        Parameters
        ----------
        since : datetime, optional
        severity : str, optional
            Filter to this severity level only.

        Returns
        -------
        list[Alert]
        """
        alerts = self._alert_history
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    def clear_history(self) -> None:
        """Clear the alert history."""
        self._alert_history = []

    def summary(self) -> dict[str, int]:
        """Return count of alerts by severity."""
        counts = {Severity.INFO: 0, Severity.WARN: 0, Severity.CRITICAL: 0}
        for a in self._alert_history:
            if a.severity in counts:
                counts[a.severity] += 1
        return counts
