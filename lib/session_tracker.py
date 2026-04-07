"""
lib/session_tracker.py
=======================
LARSA v18 -- Trading session tracker and session-aware sizing rules.

Defines the trading day as five named sessions based on UTC time, and
provides utilities for:
  - Classifying a bar timestamp into its session
  - Retrieving a per-session position-size multiplier
  - Detecting session transitions (for signal resets)
  - Computing per-session performance statistics from historical trades

Sessions (UTC):
  ASIAN        : 00:00 - 08:00
  LONDON       : 08:00 - 13:00
  US_OPEN      : 13:00 - 17:00
  US_AFTERNOON : 17:00 - 21:00
  OVERNIGHT    : 21:00 - 24:00 (wraps to 00:00)

Size multipliers reflect liquidity and signal quality by session.
Crypto runs 24/7 so all sessions are used. For equity, session multipliers
are informational -- the equity gate (RTH hours) is enforced elsewhere.

Classes:
  TradingSession    -- Enum of session labels
  SessionTracker    -- Main tracker class
  SessionStats      -- Dataclass for per-session performance

Usage::

    from datetime import datetime, timezone
    from lib.session_tracker import SessionTracker, TradingSession

    tracker = SessionTracker()
    session = tracker.current_session(bar_time)
    mult    = tracker.session_multiplier(session)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# TradingSession enum
# ---------------------------------------------------------------------------

class TradingSession(str, Enum):
    """
    Named trading sessions based on UTC clock.

    Values are used as dict keys so str inheritance is useful.
    """
    ASIAN        = "ASIAN"         # 00:00 - 08:00 UTC
    LONDON       = "LONDON"        # 08:00 - 13:00 UTC
    US_OPEN      = "US_OPEN"       # 13:00 - 17:00 UTC
    US_AFTERNOON = "US_AFTERNOON"  # 17:00 - 21:00 UTC
    OVERNIGHT    = "OVERNIGHT"     # 21:00 - 00:00 UTC

    def display_name(self) -> str:
        """Human-readable name with UTC hours."""
        _DISPLAY = {
            "ASIAN":        "Asian (00:00-08:00 UTC)",
            "LONDON":       "London (08:00-13:00 UTC)",
            "US_OPEN":      "US Open (13:00-17:00 UTC)",
            "US_AFTERNOON": "US Afternoon (17:00-21:00 UTC)",
            "OVERNIGHT":    "Overnight (21:00-00:00 UTC)",
        }
        return _DISPLAY[self.value]


# ---------------------------------------------------------------------------
# Session boundary definitions
# ---------------------------------------------------------------------------

# (start_hour_utc, end_hour_utc) -- end is exclusive
# Note: OVERNIGHT wraps midnight; handled specially in current_session()
_SESSION_HOURS: dict[TradingSession, tuple[int, int]] = {
    TradingSession.ASIAN:        (0,  8),
    TradingSession.LONDON:       (8,  13),
    TradingSession.US_OPEN:      (13, 17),
    TradingSession.US_AFTERNOON: (17, 21),
    TradingSession.OVERNIGHT:    (21, 24),  # 24 treated as 00:00 next day
}

# ---------------------------------------------------------------------------
# SessionStats dataclass
# ---------------------------------------------------------------------------

@dataclass
class SessionStats:
    """
    Performance statistics for a single trading session.

    Attributes
    ----------
    session:
        The TradingSession this stats object describes.
    trade_count:
        Total number of trades in this session.
    win_count:
        Number of winning trades (P&L > 0).
    total_pnl:
        Sum of all trade P&L values.
    avg_pnl:
        Mean P&L per trade (0.0 if no trades).
    win_rate:
        Fraction of winning trades (0.0 if no trades).
    best_trade:
        Highest single-trade P&L.
    worst_trade:
        Lowest single-trade P&L.
    """
    session:     TradingSession
    trade_count: int   = 0
    win_count:   int   = 0
    total_pnl:   float = 0.0
    avg_pnl:     float = 0.0
    win_rate:    float = 0.0
    best_trade:  float = 0.0
    worst_trade: float = 0.0

    def update(self, pnl: float) -> None:
        """Add one trade's P&L to the stats."""
        self.trade_count += 1
        self.total_pnl   += pnl
        self.avg_pnl      = self.total_pnl / self.trade_count
        if pnl > 0.0:
            self.win_count += 1
        self.win_rate   = self.win_count / self.trade_count
        self.best_trade  = max(self.best_trade, pnl)
        self.worst_trade = min(self.worst_trade, pnl)

    def to_dict(self) -> dict:
        return {
            "session":     self.session.value,
            "trade_count": self.trade_count,
            "win_count":   self.win_count,
            "total_pnl":   round(self.total_pnl, 4),
            "avg_pnl":     round(self.avg_pnl, 4),
            "win_rate":    round(self.win_rate, 4),
            "best_trade":  round(self.best_trade, 4),
            "worst_trade": round(self.worst_trade, 4),
        }

    def __str__(self) -> str:
        return (
            f"{self.session.value}: "
            f"{self.trade_count} trades, "
            f"WR={self.win_rate:.1%}, "
            f"avg_pnl={self.avg_pnl:+.4f}"
        )


# ---------------------------------------------------------------------------
# SessionTracker
# ---------------------------------------------------------------------------

class SessionTracker:
    """
    Tracks the current trading session and provides session-aware
    size multipliers and transition detection.

    Session multipliers (rationale):
      US_OPEN      : 1.20 -- highest volume, tightest spreads, LARSA signals
                             historically most reliable during this window
      LONDON       : 1.10 -- strong institutional activity, good trend follow-
                             through; second-best for crypto and EUR-correlated
                             assets
      ASIAN        : 0.80 -- lower volume for US-listed assets (crypto and
                             US equities); mean-reversion more common, BH
                             signals noisier
      US_AFTERNOON : 0.90 -- adequate liquidity; gamma-driven moves in equities;
                             moderate for crypto
      OVERNIGHT    : 0.70 -- thinnest order books; widest spreads; highest
                             slippage risk

    Usage::

        tracker = SessionTracker()
        session  = tracker.current_session(bar_dt)
        mult     = tracker.session_multiplier(session)
        if tracker.is_session_transition(prev_dt, curr_dt):
            await order_flow.reset_session()
    """

    # Per-session multipliers
    _MULTIPLIERS: dict[TradingSession, float] = {
        TradingSession.US_OPEN:      1.20,
        TradingSession.LONDON:       1.10,
        TradingSession.US_AFTERNOON: 0.90,
        TradingSession.ASIAN:        0.80,
        TradingSession.OVERNIGHT:    0.70,
    }

    def __init__(self) -> None:
        # Per-session stats accumulator
        self._stats: dict[TradingSession, SessionStats] = {
            sess: SessionStats(session=sess) for sess in TradingSession
        }
        # Bar counter per session (lightweight usage tracking)
        self._bar_counts: dict[TradingSession, int] = {
            sess: 0 for sess in TradingSession
        }
        # Last session seen (for transition detection)
        self._last_session: Optional[TradingSession] = None

    # ------------------------------------------------------------------
    # Core classification
    # ------------------------------------------------------------------

    @staticmethod
    def current_session(bar_time: datetime) -> TradingSession:
        """
        Return the TradingSession corresponding to bar_time.

        If bar_time is timezone-naive it is assumed to be UTC.
        If bar_time has a timezone, it is converted to UTC first.

        Parameters
        ----------
        bar_time:
            The bar's timestamp (typically the bar open or close time).

        Returns
        -------
        TradingSession
        """
        # Normalise to UTC
        if bar_time.tzinfo is None:
            utc_time = bar_time
        else:
            utc_time = bar_time.astimezone(timezone.utc).replace(tzinfo=None)

        hour = utc_time.hour
        minute = utc_time.minute

        # Use fractional hours for minute-level precision
        frac_hour = hour + minute / 60.0

        if frac_hour < 8.0:
            return TradingSession.ASIAN
        elif frac_hour < 13.0:
            return TradingSession.LONDON
        elif frac_hour < 17.0:
            return TradingSession.US_OPEN
        elif frac_hour < 21.0:
            return TradingSession.US_AFTERNOON
        else:
            return TradingSession.OVERNIGHT

    # ------------------------------------------------------------------
    # Multiplier
    # ------------------------------------------------------------------

    def session_multiplier(self, session: TradingSession) -> float:
        """
        Return the position-size multiplier for the given session.

        US_OPEN -> 1.20, LONDON -> 1.10, US_AFTERNOON -> 0.90,
        ASIAN -> 0.80, OVERNIGHT -> 0.70.
        """
        return self._MULTIPLIERS[session]

    def multiplier_for_time(self, bar_time: datetime) -> float:
        """
        Convenience method: classify bar_time and return its multiplier.
        """
        return self.session_multiplier(self.current_session(bar_time))

    # ------------------------------------------------------------------
    # Transition detection
    # ------------------------------------------------------------------

    def is_session_transition(self, prev_time: datetime, curr_time: datetime) -> bool:
        """
        Return True if the bar pair crosses a session boundary.

        Uses current_session() on both timestamps. A transition fires when
        the session label changes, including the midnight wrap
        (OVERNIGHT -> ASIAN).

        Parameters
        ----------
        prev_time:
            Timestamp of the previous bar.
        curr_time:
            Timestamp of the current bar.

        Returns
        -------
        bool
        """
        prev_session = self.current_session(prev_time)
        curr_session = self.current_session(curr_time)
        return prev_session != curr_session

    def get_transition(
        self, prev_time: datetime, curr_time: datetime
    ) -> Optional[tuple[TradingSession, TradingSession]]:
        """
        Return (from_session, to_session) if a transition occurred, else None.
        """
        prev_session = self.current_session(prev_time)
        curr_session = self.current_session(curr_time)
        if prev_session != curr_session:
            return prev_session, curr_session
        return None

    # ------------------------------------------------------------------
    # Bar tracking
    # ------------------------------------------------------------------

    def record_bar(self, bar_time: datetime) -> TradingSession:
        """
        Record a bar occurrence and return its session.

        Internally increments the per-session bar count.
        """
        session = self.current_session(bar_time)
        self._bar_counts[session] += 1
        self._last_session = session
        return session

    @property
    def last_session(self) -> Optional[TradingSession]:
        """Most recently recorded session (None if no bars recorded)."""
        return self._last_session

    def bar_count(self, session: TradingSession) -> int:
        """Number of bars recorded for the given session."""
        return self._bar_counts[session]

    # ------------------------------------------------------------------
    # Historical trade statistics
    # ------------------------------------------------------------------

    def record_trade(self, bar_time: datetime, pnl: float) -> None:
        """
        Record a completed trade P&L for the session containing bar_time.

        Parameters
        ----------
        bar_time:
            Time at which the trade closed (used to classify session).
        pnl:
            Realised P&L of the trade (positive = profit, negative = loss).
        """
        session = self.current_session(bar_time)
        self._stats[session].update(pnl)

    def session_stats(self, trades: list) -> dict[str, dict]:
        """
        Compute per-session win rate and average P&L from a list of trade dicts.

        Each trade dict must contain:
          'close_time' : datetime -- when the trade was closed
          'pnl'        : float    -- realised profit/loss

        Returns a dict mapping session name -> stats dict.
        This method does NOT mutate internal state; it computes fresh stats
        from the provided list only.

        Parameters
        ----------
        trades:
            List of trade dicts as described above.

        Returns
        -------
        dict mapping session value string -> SessionStats.to_dict()
        """
        # Build temporary stats objects
        temp_stats: dict[TradingSession, SessionStats] = {
            sess: SessionStats(session=sess) for sess in TradingSession
        }

        for trade in trades:
            close_time = trade.get("close_time")
            pnl        = trade.get("pnl", 0.0)
            if close_time is None:
                continue
            session = self.current_session(close_time)
            temp_stats[session].update(pnl)

        return {sess.value: stats.to_dict() for sess, stats in temp_stats.items()}

    def running_stats(self) -> dict[str, dict]:
        """
        Return running (accumulated via record_trade) per-session stats.
        """
        return {sess.value: stats.to_dict() for sess, stats in self._stats.items()}

    # ------------------------------------------------------------------
    # Utility / inspection
    # ------------------------------------------------------------------

    def session_schedule(self) -> dict[str, dict]:
        """
        Return a summary of all sessions with their UTC hours and multipliers.
        """
        result = {}
        for sess, (start, end) in _SESSION_HOURS.items():
            result[sess.value] = {
                "utc_start": start,
                "utc_end":   end if end != 24 else 0,
                "multiplier": self._MULTIPLIERS[sess],
                "display":    sess.display_name(),
            }
        return result

    @staticmethod
    def session_for_utc_hour(hour: int) -> TradingSession:
        """
        Classify an integer UTC hour [0, 23] into a session.

        Convenience for quick lookups without a full datetime object.
        """
        if hour < 8:
            return TradingSession.ASIAN
        elif hour < 13:
            return TradingSession.LONDON
        elif hour < 17:
            return TradingSession.US_OPEN
        elif hour < 21:
            return TradingSession.US_AFTERNOON
        else:
            return TradingSession.OVERNIGHT

    @staticmethod
    def all_sessions() -> list[TradingSession]:
        """Return all sessions in chronological order."""
        return [
            TradingSession.ASIAN,
            TradingSession.LONDON,
            TradingSession.US_OPEN,
            TradingSession.US_AFTERNOON,
            TradingSession.OVERNIGHT,
        ]

    def best_session(self) -> TradingSession:
        """Return the session with the highest multiplier."""
        return max(self._MULTIPLIERS, key=self._MULTIPLIERS.__getitem__)

    def worst_session(self) -> TradingSession:
        """Return the session with the lowest multiplier."""
        return min(self._MULTIPLIERS, key=self._MULTIPLIERS.__getitem__)

    def __repr__(self) -> str:
        return (
            f"SessionTracker("
            f"last={self._last_session}, "
            f"total_bars={sum(self._bar_counts.values())})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def classify_time(bar_time: datetime) -> TradingSession:
    """Module-level shortcut for SessionTracker.current_session()."""
    return SessionTracker.current_session(bar_time)


def size_multiplier(bar_time: datetime) -> float:
    """
    Module-level shortcut: return the session multiplier for bar_time.

    Creates a throwaway SessionTracker instance -- use SessionTracker
    directly if you need to track state.
    """
    tracker = SessionTracker()
    return tracker.multiplier_for_time(bar_time)


# ---------------------------------------------------------------------------
# Stand-alone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime, timezone

    tracker = SessionTracker()

    test_times = [
        ("00:00 UTC", datetime(2026, 4, 6,  0,  0, tzinfo=timezone.utc)),
        ("07:59 UTC", datetime(2026, 4, 6,  7, 59, tzinfo=timezone.utc)),
        ("08:00 UTC", datetime(2026, 4, 6,  8,  0, tzinfo=timezone.utc)),
        ("12:59 UTC", datetime(2026, 4, 6, 12, 59, tzinfo=timezone.utc)),
        ("13:00 UTC", datetime(2026, 4, 6, 13,  0, tzinfo=timezone.utc)),
        ("16:59 UTC", datetime(2026, 4, 6, 16, 59, tzinfo=timezone.utc)),
        ("17:00 UTC", datetime(2026, 4, 6, 17,  0, tzinfo=timezone.utc)),
        ("20:59 UTC", datetime(2026, 4, 6, 20, 59, tzinfo=timezone.utc)),
        ("21:00 UTC", datetime(2026, 4, 6, 21,  0, tzinfo=timezone.utc)),
        ("23:59 UTC", datetime(2026, 4, 6, 23, 59, tzinfo=timezone.utc)),
    ]

    print("=== Session classification ===")
    for label, dt in test_times:
        sess = tracker.current_session(dt)
        mult = tracker.session_multiplier(sess)
        print(f"  {label:12s} -> {sess.value:12s}  mult={mult:.2f}")

    print("\n=== Transition detection ===")
    t1 = datetime(2026, 4, 6, 12, 45, tzinfo=timezone.utc)
    t2 = datetime(2026, 4, 6, 13,  0, tzinfo=timezone.utc)
    print(f"  12:45 -> 13:00 transition? {tracker.is_session_transition(t1, t2)}")

    t3 = datetime(2026, 4, 6, 13,  0, tzinfo=timezone.utc)
    t4 = datetime(2026, 4, 6, 13, 15, tzinfo=timezone.utc)
    print(f"  13:00 -> 13:15 transition? {tracker.is_session_transition(t3, t4)}")

    print("\n=== Trade stats ===")
    from datetime import datetime, timezone
    sample_trades = [
        {"close_time": datetime(2026, 4, 6, 14, 0, tzinfo=timezone.utc), "pnl": 120.0},
        {"close_time": datetime(2026, 4, 6, 14, 15, tzinfo=timezone.utc), "pnl": -30.0},
        {"close_time": datetime(2026, 4, 6, 9, 0, tzinfo=timezone.utc),  "pnl": 50.0},
        {"close_time": datetime(2026, 4, 6, 22, 0, tzinfo=timezone.utc), "pnl": -10.0},
    ]
    stats = tracker.session_stats(sample_trades)
    for sess_name, s in stats.items():
        if s["trade_count"] > 0:
            print(f"  {sess_name}: {s}")
