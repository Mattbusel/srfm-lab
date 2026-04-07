"""
drawdown_monitor.py # Real-time drawdown tracking for SRFM.

Tracks multiple drawdown windows:
  intraday | daily | weekly | monthly | all_time

DrawdownBreachHandler implements auto-recovery logic:
  - Once drawdown < 50% of breach level, re-enable entries with half sizing.
  - SQLite persistence for all drawdown history.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations and data structures
# ---------------------------------------------------------------------------

class DrawdownWindow(str, Enum):
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ALL_TIME = "all_time"


@dataclass
class DrawdownBreach:
    """Emitted when drawdown exceeds a threshold."""
    window: DrawdownWindow
    level: float              # current drawdown as positive fraction (e.g. 0.05 = 5%)
    threshold: float          # threshold that was crossed
    timestamp: datetime
    nav_at_breach: float
    peak_nav: float
    breach_id: Optional[int] = None


@dataclass
class DrawdownState:
    """Internal per-window state."""
    window: DrawdownWindow
    peak_nav: float = 0.0
    current_nav: float = 0.0
    trough_nav: float = 0.0
    peak_ts: Optional[datetime] = None
    trough_ts: Optional[datetime] = None
    bars_since_peak: int = 0
    max_drawdown_observed: float = 0.0

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak as a positive fraction."""
        if self.peak_nav <= 0:
            return 0.0
        return max(0.0, (self.peak_nav - self.current_nav) / self.peak_nav)

    @property
    def max_drawdown(self) -> float:
        """Maximum observed drawdown in this window."""
        return self.max_drawdown_observed


# ---------------------------------------------------------------------------
# Recovery state machine
# ---------------------------------------------------------------------------

class RecoveryState(str, Enum):
    NORMAL = "normal"           # no active breach
    BREACHED = "breached"       # drawdown > threshold, halt new entries
    RECOVERING = "recovering"   # DD < 50% of breach level, half sizing


# ---------------------------------------------------------------------------
# SQLite-backed persistence
# ---------------------------------------------------------------------------

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS drawdown_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    window      TEXT    NOT NULL,
    nav         REAL    NOT NULL,
    peak_nav    REAL    NOT NULL,
    drawdown    REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS drawdown_breaches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    window          TEXT    NOT NULL,
    level           REAL    NOT NULL,
    threshold       REAL    NOT NULL,
    nav_at_breach   REAL    NOT NULL,
    peak_nav        REAL    NOT NULL,
    resolved_ts     REAL
);
CREATE INDEX IF NOT EXISTS idx_ddh_ts  ON drawdown_history(ts);
CREATE INDEX IF NOT EXISTS idx_ddb_ts  ON drawdown_breaches(ts);
"""


class DrawdownDB:
    """Thin SQLite wrapper for drawdown persistence."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_CREATE_TABLES)
        self._conn.commit()

    def insert_nav(self, window: str, nav: float, peak_nav: float, drawdown: float) -> None:
        self._conn.execute(
            "INSERT INTO drawdown_history (ts, window, nav, peak_nav, drawdown) VALUES (?,?,?,?,?)",
            (time.time(), window, nav, peak_nav, drawdown),
        )
        self._conn.commit()

    def insert_breach(
        self, window: str, level: float, threshold: float, nav: float, peak_nav: float
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO drawdown_breaches
               (ts, window, level, threshold, nav_at_breach, peak_nav)
               VALUES (?,?,?,?,?,?)""",
            (time.time(), window, level, threshold, nav, peak_nav),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def resolve_breach(self, breach_id: int) -> None:
        self._conn.execute(
            "UPDATE drawdown_breaches SET resolved_ts=? WHERE id=?",
            (time.time(), breach_id),
        )
        self._conn.commit()

    def history(
        self,
        window: str,
        since_ts: float = 0.0,
        limit: int = 1000,
    ) -> List[dict]:
        cur = self._conn.execute(
            "SELECT ts, nav, peak_nav, drawdown FROM drawdown_history "
            "WHERE window=? AND ts>=? ORDER BY ts DESC LIMIT ?",
            (window, since_ts, limit),
        )
        return [{"ts": r[0], "nav": r[1], "peak_nav": r[2], "drawdown": r[3]} for r in cur]

    def max_drawdown_since(self, window: str, since_ts: float) -> float:
        cur = self._conn.execute(
            "SELECT MAX(drawdown) FROM drawdown_history WHERE window=? AND ts>=?",
            (window, since_ts),
        )
        row = cur.fetchone()
        return row[0] or 0.0

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# DrawdownMonitor
# ---------------------------------------------------------------------------

class DrawdownMonitor:
    """
    Real-time drawdown tracking across five time windows.

    Call `update(nav, timestamp)` on every bar.
    """

    _WINDOW_LOOKBACKS: Dict[DrawdownWindow, Optional[timedelta]] = {
        DrawdownWindow.INTRADAY: timedelta(hours=8),
        DrawdownWindow.DAILY:    timedelta(days=1),
        DrawdownWindow.WEEKLY:   timedelta(weeks=1),
        DrawdownWindow.MONTHLY:  timedelta(days=30),
        DrawdownWindow.ALL_TIME: None,
    }

    def __init__(
        self,
        initial_nav: float,
        db_path: str = "drawdown.db",
        breach_callbacks: Optional[List[Callable[[DrawdownBreach], None]]] = None,
    ) -> None:
        self._states: Dict[DrawdownWindow, DrawdownState] = {
            w: DrawdownState(window=w, peak_nav=initial_nav, current_nav=initial_nav, trough_nav=initial_nav)
            for w in DrawdownWindow
        }
        self._db = DrawdownDB(db_path)
        self._breach_callbacks: List[Callable[[DrawdownBreach], None]] = breach_callbacks or []
        self._last_update_ts: Optional[datetime] = None

    # # Core update ---------------------------------------------------------

    def update(self, nav: float, timestamp: datetime) -> None:
        """
        Ingest a new NAV observation.  Must be called sequentially per bar.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        self._last_update_ts = timestamp

        for window, state in self._states.items():
            self._update_window(window, state, nav, timestamp)

    def _update_window(
        self,
        window: DrawdownWindow,
        state: DrawdownState,
        nav: float,
        ts: datetime,
    ) -> None:
        # Reset peak if we are starting a new window period
        lookback = self._WINDOW_LOOKBACKS[window]
        if lookback is not None and state.peak_ts is not None:
            age = ts - state.peak_ts
            if age > lookback:
                # Roll the window # reset peak to current nav
                state.peak_nav = nav
                state.peak_ts = ts
                state.bars_since_peak = 0

        # Update peak
        if nav >= state.peak_nav:
            state.peak_nav = nav
            state.peak_ts = ts
            state.bars_since_peak = 0
        else:
            state.bars_since_peak += 1

        state.current_nav = nav

        dd = state.current_drawdown
        if dd > state.max_drawdown_observed:
            state.max_drawdown_observed = dd
            state.trough_nav = nav
            state.trough_ts = ts

        # Persist every update to all_time; others sampled for performance
        if window == DrawdownWindow.ALL_TIME:
            self._db.insert_nav(window.value, nav, state.peak_nav, dd)

    # # Accessors -----------------------------------------------------------

    def current_drawdown(
        self, window: DrawdownWindow = DrawdownWindow.ALL_TIME
    ) -> float:
        """Current drawdown from peak as a positive fraction."""
        return self._states[window].current_drawdown

    def max_drawdown(
        self,
        lookback_days: Optional[int] = None,
        window: DrawdownWindow = DrawdownWindow.ALL_TIME,
    ) -> float:
        """
        Maximum drawdown.  If lookback_days is set, queries the DB for
        the historical maximum in that period.  Otherwise returns the
        in-memory maximum for the given window.
        """
        if lookback_days is not None:
            since_ts = time.time() - lookback_days * 86400
            return self._db.max_drawdown_since(DrawdownWindow.ALL_TIME.value, since_ts)
        return self._states[window].max_drawdown_observed

    def drawdown_duration(
        self, window: DrawdownWindow = DrawdownWindow.ALL_TIME
    ) -> int:
        """Number of bars since the last peak in the given window."""
        return self._states[window].bars_since_peak

    def is_breach(
        self,
        threshold: float,
        window: DrawdownWindow = DrawdownWindow.ALL_TIME,
    ) -> bool:
        """True when current drawdown exceeds `threshold`."""
        return self._states[window].current_drawdown > threshold

    def recovery_target(
        self, window: DrawdownWindow = DrawdownWindow.ALL_TIME
    ) -> float:
        """NAV level required to fully recover to peak."""
        return self._states[window].peak_nav

    def peak_nav(self, window: DrawdownWindow = DrawdownWindow.ALL_TIME) -> float:
        return self._states[window].peak_nav

    def window_state(self, window: DrawdownWindow) -> DrawdownState:
        return self._states[window]

    def summary(self) -> Dict:
        """Return a dict of all window drawdowns for dashboards."""
        return {
            w.value: {
                "current_dd": s.current_drawdown,
                "max_dd": s.max_drawdown_observed,
                "peak_nav": s.peak_nav,
                "bars_since_peak": s.bars_since_peak,
            }
            for w, s in self._states.items()
        }

    def add_breach_callback(self, fn: Callable[[DrawdownBreach], None]) -> None:
        self._breach_callbacks.append(fn)

    def _fire_breach(self, breach: DrawdownBreach) -> None:
        for fn in self._breach_callbacks:
            try:
                fn(breach)
            except Exception:
                logger.exception("Breach callback raised an exception")

    def close(self) -> None:
        self._db.close()


# ---------------------------------------------------------------------------
# DrawdownBreachHandler
# ---------------------------------------------------------------------------

class DrawdownBreachHandler:
    """
    Monitors thresholds and manages recovery state.

    Integrates with DrawdownMonitor via callbacks.  Call
    `evaluate(monitor, nav)` on each bar to run the state machine.

    Auto-recovery rules:
    - Once DD < 50% of breach level, transition to RECOVERING.
    - In RECOVERING, sizing_factor = 0.5.
    - Full recovery when DD < 25% of original breach level.
    """

    def __init__(
        self,
        threshold: float,
        window: DrawdownWindow = DrawdownWindow.DAILY,
        alert_fn: Optional[Callable[[DrawdownBreach], None]] = None,
        db_path: str = "drawdown.db",
    ) -> None:
        self._threshold = threshold
        self._window = window
        self._alert_fn = alert_fn
        self._state = RecoveryState.NORMAL
        self._active_breach: Optional[DrawdownBreach] = None
        self._sizing_factor: float = 1.0
        self._halt_entries: bool = False
        self._db = DrawdownDB(db_path)

    # # State machine -------------------------------------------------------

    def evaluate(self, monitor: DrawdownMonitor, current_nav: float) -> None:
        """
        Call once per bar after DrawdownMonitor.update().
        Drives the breach/recovery state machine.
        """
        dd = monitor.current_drawdown(self._window)

        if self._state == RecoveryState.NORMAL:
            if dd > self._threshold:
                self._enter_breach(dd, monitor, current_nav)

        elif self._state == RecoveryState.BREACHED:
            recovery_dd_threshold = self._active_breach.level * 0.50  # type: ignore[union-attr]
            if dd <= recovery_dd_threshold:
                self._enter_recovering(dd)
            elif dd > (self._active_breach.level + 0.01):  # type: ignore[union-attr]
                # Worsening # update breach record
                self._active_breach.level = dd  # type: ignore[union-attr]

        elif self._state == RecoveryState.RECOVERING:
            full_recovery_threshold = self._active_breach.level * 0.25  # type: ignore[union-attr]
            if dd <= full_recovery_threshold:
                self._enter_normal()
            elif dd > self._threshold:
                # Re-entered breach territory
                self._enter_breach(dd, monitor, current_nav)

    def _enter_breach(
        self, dd: float, monitor: DrawdownMonitor, current_nav: float
    ) -> None:
        logger.warning("Drawdown breach: %.2f%% > threshold %.2f%%", dd * 100, self._threshold * 100)
        breach = DrawdownBreach(
            window=self._window,
            level=dd,
            threshold=self._threshold,
            timestamp=datetime.now(timezone.utc),
            nav_at_breach=current_nav,
            peak_nav=monitor.peak_nav(self._window),
        )
        breach.breach_id = self._db.insert_breach(
            self._window.value, dd, self._threshold, current_nav, breach.peak_nav
        )
        self._active_breach = breach
        self._state = RecoveryState.BREACHED
        self._sizing_factor = 0.0
        self._halt_entries = True
        self.send_alert(breach)
        self.on_breach(breach)

    def _enter_recovering(self, dd: float) -> None:
        logger.info("Drawdown recovering: %.2f%% # re-enabling at half size", dd * 100)
        self._state = RecoveryState.RECOVERING
        self._sizing_factor = 0.5
        self._halt_entries = False

    def _enter_normal(self) -> None:
        logger.info("Drawdown fully recovered # restoring normal sizing")
        if self._active_breach and self._active_breach.breach_id:
            self._db.resolve_breach(self._active_breach.breach_id)
        self._state = RecoveryState.NORMAL
        self._active_breach = None
        self._sizing_factor = 1.0
        self._halt_entries = False

    # # Actions -------------------------------------------------------------

    def on_breach(self, breach: DrawdownBreach) -> None:
        """Called when a breach threshold is crossed.  Subclass or override."""
        self.halt_new_entries()
        self.reduce_risk(0.0)

    def reduce_risk(self, factor: float) -> None:
        """
        Signal to all downstream components to scale position sizes by `factor`.
        factor=0.0 means do not open new positions; existing positions remain.
        """
        self._sizing_factor = factor
        logger.info("Risk reduced to %.0f%% sizing", factor * 100)

    def halt_new_entries(self) -> None:
        """Block all new position entries; allow exits."""
        self._halt_entries = True
        logger.warning("New entries halted due to drawdown breach")

    def send_alert(self, breach: DrawdownBreach) -> None:
        """Route alert through the injected alert function if provided."""
        if self._alert_fn is not None:
            try:
                self._alert_fn(breach)
            except Exception:
                logger.exception("Alert function raised an exception")

    # # Properties ----------------------------------------------------------

    @property
    def sizing_factor(self) -> float:
        """Current position sizing factor (0.0-1.0)."""
        return self._sizing_factor

    @property
    def entries_halted(self) -> bool:
        """True when new entries should be blocked."""
        return self._halt_entries

    @property
    def recovery_state(self) -> RecoveryState:
        return self._state

    @property
    def active_breach(self) -> Optional[DrawdownBreach]:
        return self._active_breach

    def status(self) -> Dict:
        return {
            "state": self._state.value,
            "sizing_factor": self._sizing_factor,
            "entries_halted": self._halt_entries,
            "threshold": self._threshold,
            "window": self._window.value,
            "active_breach": (
                {
                    "level": self._active_breach.level,
                    "threshold": self._active_breach.threshold,
                    "timestamp": self._active_breach.timestamp.isoformat(),
                    "nav_at_breach": self._active_breach.nav_at_breach,
                }
                if self._active_breach
                else None
            ),
        }

    def close(self) -> None:
        self._db.close()
