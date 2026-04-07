"""
lib/signal_registry.py
=======================
Registry of all active trading signals with rolling IC / ICIR tracking
and status management.

Signals tracked:
    BH_MASS_15m, BH_MASS_1h, BH_MASS_4h  -- Black-Hole mass per timeframe
    CF_CROSS                               -- Cross-timeframe CF alignment
    GARCH_VOL                              -- GARCH volatility signal
    HURST_REGIME                           -- Hurst exponent regime filter
    NAV_OMEGA                              -- QuatNav angular velocity signal
    NAV_GEODESIC                           -- QuatNav geodesic deviation signal
    ML_SIGNAL                              -- ML model output signal
    GRANGER_BTC                            -- Granger BTC lead signal
    RL_EXIT                                -- RL policy exit signal
    EVENT_CALENDAR                         -- Event calendar filter signal

Usage:
    from lib.signal_registry import get_signal_registry

    sr = get_signal_registry()
    sr.update_ic("BH_MASS_15m", 0.045)
    report = sr.get_report()
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("signal_registry")

_REPO_ROOT  = Path(__file__).parents[1]
_DEFAULT_DB = _REPO_ROOT / "execution" / "live_trades.db"

# ---------------------------------------------------------------------------
# Signal names
# ---------------------------------------------------------------------------

ALL_SIGNALS: list[str] = [
    "BH_MASS_15m",
    "BH_MASS_1h",
    "BH_MASS_4h",
    "CF_CROSS",
    "GARCH_VOL",
    "HURST_REGIME",
    "NAV_OMEGA",
    "NAV_GEODESIC",
    "ML_SIGNAL",
    "GRANGER_BTC",
    "RL_EXIT",
    "EVENT_CALENDAR",
]

# Status constants
STATUS_ACTIVE    = "ACTIVE"
STATUS_PROBATION = "PROBATION"
STATUS_RETIRED   = "RETIRED"

# IC rolling window
IC_WINDOW = 30

# Thresholds for automatic status transitions
IC_PROBATION_THRESHOLD  =  0.03   # IC < this for IC_WINDOW consecutive days -> PROBATION
IC_RETIREMENT_THRESHOLD =  0.00   # IC <= 0 rolling average -> RETIRED
ICIR_GOOD_THRESHOLD     =  0.50   # ICIR > this -> ACTIVE
ICIR_WARN_THRESHOLD     =  0.20   # ICIR between 0.20 and 0.50 -> PROBATION


# ---------------------------------------------------------------------------
# SignalState dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignalState:
    """Current state of a single signal."""
    name:          str
    is_active:     bool         = True
    status:        str          = STATUS_ACTIVE
    ic_rolling_30d: Optional[float] = None   # mean IC over last 30 observations
    icir_30d:      Optional[float] = None    # IC / std(IC) over last 30 obs
    ic_history:    list[float]  = field(default_factory=list)  # raw IC values (kept in memory)
    last_ic:       Optional[float] = None
    last_update:   Optional[datetime] = None
    manual_override: bool       = False      # if True, auto-status transitions disabled
    description:   str          = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":           self.name,
            "is_active":      self.is_active,
            "status":         self.status,
            "ic_rolling_30d": round(self.ic_rolling_30d, 6) if self.ic_rolling_30d is not None else None,
            "icir_30d":       round(self.icir_30d, 4)      if self.icir_30d       is not None else None,
            "last_ic":        self.last_ic,
            "last_update":    self.last_update.isoformat() if self.last_update else None,
            "manual_override":self.manual_override,
            "description":    self.description,
        }


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signal_registry (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    signal_name TEXT    NOT NULL,
    ic_value    REAL,
    icir        REAL,
    status      TEXT,
    note        TEXT
);
CREATE INDEX IF NOT EXISTS idx_signal_registry_name ON signal_registry(signal_name);
CREATE INDEX IF NOT EXISTS idx_signal_registry_ts   ON signal_registry(ts);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# SignalRegistry
# ---------------------------------------------------------------------------

class SignalRegistry:
    """
    Thread-safe registry of all trading signals with IC/ICIR tracking.

    IC history is maintained in a rolling deque (last IC_WINDOW observations).
    ICIR = mean(IC) / std(IC) over the window -- measures information ratio.

    Status transitions:
        ACTIVE -> PROBATION if ICIR < ICIR_WARN_THRESHOLD
        PROBATION -> RETIRED if rolling IC mean <= IC_RETIREMENT_THRESHOLD
        RETIRED -> ACTIVE only via manual set_signal_status() call
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB
        self._lock    = threading.RLock()
        self._signals: dict[str, SignalState]           = {}
        self._ic_deque: dict[str, deque[float]]         = {}
        self._conn: Optional[sqlite3.Connection]        = None

        self._init_signals()
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_signals(self) -> None:
        """Populate registry with all known signals at default state."""
        descriptions = {
            "BH_MASS_15m":     "Black-Hole mass signal on 15-minute bars",
            "BH_MASS_1h":      "Black-Hole mass signal on 1-hour bars",
            "BH_MASS_4h":      "Black-Hole mass signal on 4-hour bars",
            "CF_CROSS":        "Cross-timeframe centrifugal force alignment",
            "GARCH_VOL":       "GARCH(1,1) annualised volatility forecast",
            "HURST_REGIME":    "Hurst exponent regime filter (H > 0.5 = trending)",
            "NAV_OMEGA":       "QuatNav angular velocity size penalty",
            "NAV_GEODESIC":    "QuatNav geodesic path length entry gate",
            "ML_SIGNAL":       "Gradient-boosted ML model output",
            "GRANGER_BTC":     "Granger causality BTC lead signal for alts",
            "RL_EXIT":         "Reinforcement learning exit policy",
            "EVENT_CALENDAR":  "Economic event calendar filter",
        }
        for name in ALL_SIGNALS:
            self._signals[name] = SignalState(
                name        = name,
                description = descriptions.get(name, ""),
            )
            self._ic_deque[name] = deque(maxlen=IC_WINDOW)

    # ------------------------------------------------------------------
    # IC / ICIR updates
    # ------------------------------------------------------------------

    def update_ic(self, signal_name: str, ic_value: float) -> None:
        """
        Record a new IC observation for a signal.

        Automatically recomputes ICIR and updates status.
        Persists to SQLite.
        """
        with self._lock:
            if signal_name not in self._signals:
                log.warning("signal_registry: unknown signal '%s'", signal_name)
                return

            sig   = self._signals[signal_name]
            deq   = self._ic_deque[signal_name]

            deq.append(ic_value)
            sig.ic_history.append(ic_value)
            sig.last_ic     = ic_value
            sig.last_update = datetime.now(timezone.utc)

            # Recompute rolling stats
            self._recompute_icir(signal_name)

            # Auto status transition (skipped if manual override)
            if not sig.manual_override:
                self._auto_transition(signal_name)

            # Persist
            self._persist_ic(signal_name, ic_value, sig.icir_30d, sig.status)

    def update_icir(self, signal_name: str) -> None:
        """Force recompute ICIR from existing IC history (without adding new IC)."""
        with self._lock:
            self._recompute_icir(signal_name)

    def _recompute_icir(self, signal_name: str) -> None:
        """Internal: recompute rolling IC mean and ICIR. Must be called under lock."""
        sig = self._signals.get(signal_name)
        if sig is None:
            return
        deq = self._ic_deque.get(signal_name)
        if not deq:
            return
        vals = list(deq)
        n    = len(vals)
        mean = sum(vals) / n
        sig.ic_rolling_30d = mean
        if n >= 2:
            variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
            std      = math.sqrt(variance) if variance > 0 else 0.0
            sig.icir_30d = mean / std if std > 1e-9 else 0.0
        else:
            sig.icir_30d = 0.0

    def _auto_transition(self, signal_name: str) -> None:
        """Auto status transition logic. Must be called under lock."""
        sig    = self._signals[signal_name]
        icir   = sig.icir_30d   or 0.0
        ic_avg = sig.ic_rolling_30d or 0.0
        n_obs  = len(self._ic_deque[signal_name])

        if n_obs < 5:
            return   # not enough data yet

        if sig.status == STATUS_ACTIVE:
            if icir < ICIR_WARN_THRESHOLD:
                sig.status    = STATUS_PROBATION
                sig.is_active = True   # still fires, but flagged
                log.warning(
                    "signal_registry: %s -> PROBATION (ICIR=%.3f < %.2f)",
                    signal_name, icir, ICIR_WARN_THRESHOLD,
                )

        elif sig.status == STATUS_PROBATION:
            if icir >= ICIR_GOOD_THRESHOLD:
                sig.status = STATUS_ACTIVE
                log.info(
                    "signal_registry: %s PROBATION -> ACTIVE (ICIR=%.3f)",
                    signal_name, icir,
                )
            elif ic_avg <= IC_RETIREMENT_THRESHOLD:
                sig.status    = STATUS_RETIRED
                sig.is_active = False
                log.warning(
                    "signal_registry: %s -> RETIRED (rolling IC=%.4f <= %.4f)",
                    signal_name, ic_avg, IC_RETIREMENT_THRESHOLD,
                )

        elif sig.status == STATUS_RETIRED:
            # Only manual override can revive a retired signal
            pass

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def set_signal_status(
        self,
        signal_name: str,
        status: str,
        reason: str = "",
    ) -> None:
        """
        Manually set signal status. Also sets manual_override=True to
        prevent automatic status transitions from overwriting it.

        status must be one of: ACTIVE, PROBATION, RETIRED.
        """
        if status not in (STATUS_ACTIVE, STATUS_PROBATION, STATUS_RETIRED):
            raise ValueError(f"Invalid status '{status}'. Must be ACTIVE/PROBATION/RETIRED")
        with self._lock:
            sig = self._signals.get(signal_name)
            if sig is None:
                raise KeyError(f"signal_registry: unknown signal '{signal_name}'")
            old_status        = sig.status
            sig.status        = status
            sig.is_active     = status != STATUS_RETIRED
            sig.manual_override = True
            self._persist_ic(signal_name, None, sig.icir_30d, status, note=f"manual:{reason}")
            log.info(
                "signal_registry: %s status %s -> %s (manual) -- %s",
                signal_name, old_status, status, reason,
            )

    def clear_manual_override(self, signal_name: str) -> None:
        """Re-enable automatic status transitions for a signal."""
        with self._lock:
            sig = self._signals.get(signal_name)
            if sig is None:
                raise KeyError(f"signal_registry: unknown signal '{signal_name}'")
            sig.manual_override = False
            log.info("signal_registry: cleared manual override for %s", signal_name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, signal_name: str) -> SignalState:
        """Return SignalState for a signal. Raises KeyError if not found."""
        with self._lock:
            sig = self._signals.get(signal_name)
            if sig is None:
                raise KeyError(f"signal_registry: unknown signal '{signal_name}'")
            return sig

    def get_active_signals(self) -> list[str]:
        """Return names of all non-retired signals."""
        with self._lock:
            return [
                name for name, sig in self._signals.items()
                if sig.status != STATUS_RETIRED
            ]

    def get_fired_signals(self) -> list[str]:
        """Return names of signals that are ACTIVE (not PROBATION/RETIRED)."""
        with self._lock:
            return [
                name for name, sig in self._signals.items()
                if sig.status == STATUS_ACTIVE and sig.is_active
            ]

    def get_all_states(self) -> dict[str, SignalState]:
        """Return a snapshot copy of all signal states."""
        with self._lock:
            return {k: v for k, v in self._signals.items()}

    def get_report(self) -> dict[str, Any]:
        """Return a formatted status report dict for all signals."""
        with self._lock:
            rows = []
            for name in ALL_SIGNALS:
                sig = self._signals.get(name)
                if sig is None:
                    continue
                rows.append(sig.to_dict())
            return {
                "generated_at":    _now_iso(),
                "total_signals":   len(rows),
                "active_count":    sum(1 for r in rows if r["status"] == STATUS_ACTIVE),
                "probation_count": sum(1 for r in rows if r["status"] == STATUS_PROBATION),
                "retired_count":   sum(1 for r in rows if r["status"] == STATUS_RETIRED),
                "signals":         rows,
            }

    def get_ic_history(self, signal_name: str, n: int = IC_WINDOW) -> list[float]:
        """Return last N IC values for a signal."""
        with self._lock:
            sig = self._signals.get(signal_name)
            if sig is None:
                return []
            return list(sig.ic_history[-n:])

    def get_icir_by_signal(self) -> dict[str, Optional[float]]:
        """Return {signal_name: icir_30d} for all signals."""
        with self._lock:
            return {
                name: sig.icir_30d
                for name, sig in self._signals.items()
            }

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            self._conn = conn
        except Exception as exc:
            log.warning("signal_registry: DB init failed -- %s", exc)
            self._conn = None

    def _persist_ic(
        self,
        signal_name: str,
        ic_value: Optional[float],
        icir: Optional[float],
        status: str,
        note: str = "",
    ) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT INTO signal_registry (ts, signal_name, ic_value, icir, status, note) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_now_iso(), signal_name, ic_value, icir, status, note),
            )
            self._conn.commit()
        except Exception as exc:
            log.debug("signal_registry: persist error -- %s", exc)

    def get_ic_from_db(
        self,
        signal_name: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve IC history for a signal from the database."""
        if self._conn is None:
            return []
        try:
            cur = self._conn.execute(
                "SELECT ts, ic_value, icir, status FROM signal_registry "
                "WHERE signal_name = ? ORDER BY id DESC LIMIT ?",
                (signal_name, limit),
            )
            return [
                {"ts": ts, "ic_value": ic, "icir": icir, "status": st}
                for ts, ic, icir, st in cur.fetchall()
            ]
        except Exception as exc:
            log.debug("signal_registry: get_ic_from_db error -- %s", exc)
            return []

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_singleton: Optional[SignalRegistry] = None
_singleton_lock = threading.Lock()


def get_signal_registry(db_path: Optional[Path] = None) -> SignalRegistry:
    """Return (or create) the module-level SignalRegistry singleton."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = SignalRegistry(db_path=db_path)
    return _singleton


def reset_singleton_for_testing(db_path: Optional[Path] = None) -> SignalRegistry:
    """Force-create a new singleton. For use in tests only."""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.close()
        _singleton = SignalRegistry(db_path=db_path)
    return _singleton
