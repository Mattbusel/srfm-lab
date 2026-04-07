"""
lib/session_state.py
====================
Centralized, thread-safe session state for the LARSA live trader.

Tracks:
  - Global trading enabled/disabled toggle
  - Pause until datetime (blocks new entries for N minutes)
  - Per-symbol blocks with optional expiry
  - Emergency mode (hard disable all trading)

All state changes are persisted to SQLite (sessions table) for audit.
The REST API in tools/live_controls_v2.py should read/write through this
module rather than directly touching the database.

Usage:
    from lib.session_state import get_session_state

    ss = get_session_state()
    ss.enable_trading()
    ss.pause(minutes=30)
    ss.block_symbol("TSLA", duration_hours=2)
    if ss.is_tradeable("BTC"):
        ...
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("session_state")

_REPO_ROOT = Path(__file__).parents[1]
_DEFAULT_DB = _REPO_ROOT / "execution" / "live_trades.db"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    symbol      TEXT,
    reason      TEXT,
    extra_json  TEXT
);

CREATE TABLE IF NOT EXISTS session_symbol_blocks (
    symbol      TEXT    PRIMARY KEY,
    blocked_at  TEXT    NOT NULL,
    expires_at  TEXT,
    reason      TEXT
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# SymbolBlock dataclass
# ---------------------------------------------------------------------------

@dataclass
class SymbolBlock:
    symbol:     str
    blocked_at: datetime
    expires_at: Optional[datetime]
    reason:     str

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.expires_at is None:
            return False
        ts = now or datetime.now(timezone.utc)
        return ts >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":     self.symbol,
            "blocked_at": self.blocked_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reason":     self.reason,
        }


# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------

class SessionState:
    """
    Singleton thread-safe session state container.

    Do not instantiate directly -- use get_session_state().

    State fields:
        trading_enabled  -- global kill switch
        paused_until     -- datetime until which new entries are blocked (None = not paused)
        blocked_symbols  -- set of symbols with active blocks
        emergency_mode   -- hard stop: disables all trading, requires explicit reset
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path    = Path(db_path) if db_path else _DEFAULT_DB
        self._lock       = threading.RLock()
        self._trading_enabled: bool                  = True
        self._paused_until:    Optional[datetime]    = None
        self._symbol_blocks:   dict[str, SymbolBlock]= {}
        self._emergency_mode:  bool                  = False
        self._conn: Optional[sqlite3.Connection]     = None

        self._init_db()
        self._restore_symbol_blocks()

    # ------------------------------------------------------------------
    # Global trading toggle
    # ------------------------------------------------------------------

    def enable_trading(self, reason: str = "") -> None:
        """Enable global trading. Clears emergency mode if set."""
        with self._lock:
            if self._emergency_mode:
                log.warning(
                    "session_state: enable_trading called while in emergency mode -- "
                    "clearing emergency mode"
                )
                self._emergency_mode = False
            was = self._trading_enabled
            self._trading_enabled = True
            if not was:
                self._persist_event("TRADING_ENABLED", reason=reason)
                log.info("session_state: trading ENABLED -- %s", reason or "(no reason)")

    def disable_trading(self, reason: str = "") -> None:
        """Disable global trading (all new entries blocked)."""
        with self._lock:
            was = self._trading_enabled
            self._trading_enabled = False
            if was:
                self._persist_event("TRADING_DISABLED", reason=reason)
                log.warning("session_state: trading DISABLED -- %s", reason or "(no reason)")

    def is_trading_enabled(self) -> bool:
        with self._lock:
            return self._trading_enabled and not self._emergency_mode

    # ------------------------------------------------------------------
    # Pause (temporary entry block)
    # ------------------------------------------------------------------

    def pause(self, minutes: float, reason: str = "") -> None:
        """
        Pause new entries for N minutes.
        Can be extended by calling again before expiry.
        """
        with self._lock:
            expiry              = datetime.now(timezone.utc) + timedelta(minutes=minutes)
            self._paused_until  = expiry
            self._persist_event(
                "PAUSE_SET",
                reason=reason,
                extra={"minutes": minutes, "expires_at": expiry.isoformat()},
            )
            log.info(
                "session_state: trading PAUSED for %.1f minutes (until %s) -- %s",
                minutes,
                expiry.strftime("%H:%M:%S"),
                reason or "(no reason)",
            )

    def resume(self, reason: str = "") -> None:
        """Clear pause immediately."""
        with self._lock:
            if self._paused_until is not None:
                self._paused_until = None
                self._persist_event("PAUSE_CLEARED", reason=reason)
                log.info("session_state: pause CLEARED -- %s", reason or "(no reason)")

    def is_paused(self) -> bool:
        """Return True if currently in a pause window."""
        with self._lock:
            if self._paused_until is None:
                return False
            now = datetime.now(timezone.utc)
            if now >= self._paused_until:
                # Auto-expire
                self._paused_until = None
                return False
            return True

    def paused_until(self) -> Optional[datetime]:
        """Return the datetime when pause expires, or None."""
        with self._lock:
            if self._paused_until is None:
                return None
            now = datetime.now(timezone.utc)
            if now >= self._paused_until:
                self._paused_until = None
                return None
            return self._paused_until

    # ------------------------------------------------------------------
    # Per-symbol blocks
    # ------------------------------------------------------------------

    def block_symbol(
        self,
        symbol:         str,
        reason:         str          = "",
        duration_hours: Optional[float] = None,
    ) -> None:
        """
        Block trading of a specific symbol.

        If duration_hours is None the block is permanent until unblock_symbol().
        Otherwise the block expires after the given duration.
        """
        with self._lock:
            sym   = symbol.upper()
            now   = datetime.now(timezone.utc)
            expiry = (now + timedelta(hours=duration_hours)) if duration_hours else None
            self._symbol_blocks[sym] = SymbolBlock(
                symbol     = sym,
                blocked_at = now,
                expires_at = expiry,
                reason     = reason,
            )
            self._persist_symbol_block(sym, now, expiry, reason)
            self._persist_event(
                "SYMBOL_BLOCKED",
                symbol  = sym,
                reason  = reason,
                extra   = {"duration_hours": duration_hours},
            )
            if expiry:
                log.warning(
                    "session_state: blocked %s for %.1fh (until %s) -- %s",
                    sym, duration_hours, expiry.strftime("%H:%M:%S UTC"), reason,
                )
            else:
                log.warning("session_state: blocked %s (permanent) -- %s", sym, reason)

    def unblock_symbol(self, symbol: str, reason: str = "") -> None:
        """Remove block on a symbol."""
        with self._lock:
            sym = symbol.upper()
            if sym in self._symbol_blocks:
                del self._symbol_blocks[sym]
                self._remove_symbol_block(sym)
                self._persist_event("SYMBOL_UNBLOCKED", symbol=sym, reason=reason)
                log.info("session_state: unblocked %s -- %s", sym, reason or "(no reason)")

    def is_symbol_blocked(self, symbol: str) -> bool:
        """Return True if the symbol has an active (non-expired) block."""
        with self._lock:
            sym   = symbol.upper()
            block = self._symbol_blocks.get(sym)
            if block is None:
                return False
            if block.is_expired():
                del self._symbol_blocks[sym]
                self._remove_symbol_block(sym)
                return False
            return True

    def get_blocked_symbols(self) -> dict[str, SymbolBlock]:
        """Return a snapshot of currently blocked symbols (expired blocks removed)."""
        with self._lock:
            now     = datetime.now(timezone.utc)
            expired = [s for s, b in self._symbol_blocks.items() if b.is_expired(now)]
            for s in expired:
                del self._symbol_blocks[s]
                self._remove_symbol_block(s)
            return dict(self._symbol_blocks)

    # ------------------------------------------------------------------
    # Emergency mode
    # ------------------------------------------------------------------

    def set_emergency_mode(self, reason: str = "EMERGENCY") -> None:
        """
        Hard disable all trading. Sets emergency_mode flag.
        Can only be cleared by calling enable_trading().
        """
        with self._lock:
            self._emergency_mode  = True
            self._trading_enabled = False
            self._persist_event("EMERGENCY_MODE", reason=reason)
            log.critical("session_state: *** EMERGENCY MODE ACTIVATED *** -- %s", reason)

    def is_emergency(self) -> bool:
        with self._lock:
            return self._emergency_mode

    # ------------------------------------------------------------------
    # Composite tradeable check
    # ------------------------------------------------------------------

    def is_tradeable(self, symbol: str) -> bool:
        """
        Return True if it is safe to trade this symbol right now.

        Checks (in order):
          1. Emergency mode disabled
          2. Global trading enabled
          3. Not in pause window
          4. Symbol not blocked
        """
        with self._lock:
            if self._emergency_mode:
                return False
            if not self._trading_enabled:
                return False
        # Outside lock for pause/block checks (they are self-locking)
        if self.is_paused():
            return False
        if self.is_symbol_blocked(symbol):
            return False
        return True

    def tradeable_reason(self, symbol: str) -> str:
        """Return a human-readable reason why a symbol is not tradeable, or 'OK'."""
        with self._lock:
            if self._emergency_mode:
                return "emergency mode active"
            if not self._trading_enabled:
                return "trading globally disabled"
        if self.is_paused():
            expiry = self.paused_until()
            if expiry:
                return f"paused until {expiry.strftime('%H:%M:%S UTC')}"
            return "paused"
        if self.is_symbol_blocked(symbol):
            block = self._symbol_blocks.get(symbol.upper())
            if block:
                return f"symbol blocked: {block.reason}"
            return "symbol blocked"
        return "OK"

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a complete snapshot of current session state."""
        blocks = self.get_blocked_symbols()
        paused = self.paused_until()
        with self._lock:
            return {
                "trading_enabled":  self._trading_enabled,
                "emergency_mode":   self._emergency_mode,
                "paused_until":     paused.isoformat() if paused else None,
                "blocked_symbols":  {s: b.to_dict() for s, b in blocks.items()},
                "snapshot_at":      _now_iso(),
            }

    # ------------------------------------------------------------------
    # Event log query
    # ------------------------------------------------------------------

    def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return most recent session events from the database."""
        conn = self._get_conn()
        cur  = conn.execute(
            "SELECT ts, event_type, symbol, reason, extra_json FROM session_events "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = []
        for ts, etype, sym, reason, extra in cur.fetchall():
            rows.append({
                "ts":         ts,
                "event_type": etype,
                "symbol":     sym,
                "reason":     reason,
                "extra":      extra,
            })
        return rows

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            self._conn = conn
        except Exception as exc:
            log.warning("session_state: DB init failed (state will be in-memory only) -- %s", exc)
            self._conn = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("session_state: database not available")
        return self._conn

    def _persist_event(
        self,
        event_type: str,
        symbol: Optional[str] = None,
        reason: str = "",
        extra: Optional[dict] = None,
    ) -> None:
        if self._conn is None:
            return
        import json
        try:
            self._conn.execute(
                "INSERT INTO session_events (ts, event_type, symbol, reason, extra_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    _now_iso(),
                    event_type,
                    symbol,
                    reason,
                    json.dumps(extra) if extra else None,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            log.debug("session_state: persist_event error -- %s", exc)

    def _persist_symbol_block(
        self,
        symbol: str,
        blocked_at: datetime,
        expires_at: Optional[datetime],
        reason: str,
    ) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO session_symbol_blocks
                   (symbol, blocked_at, expires_at, reason)
                   VALUES (?, ?, ?, ?)""",
                (
                    symbol,
                    blocked_at.isoformat(),
                    expires_at.isoformat() if expires_at else None,
                    reason,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            log.debug("session_state: persist_symbol_block error -- %s", exc)

    def _remove_symbol_block(self, symbol: str) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "DELETE FROM session_symbol_blocks WHERE symbol = ?", (symbol,)
            )
            self._conn.commit()
        except Exception as exc:
            log.debug("session_state: remove_symbol_block error -- %s", exc)

    def _restore_symbol_blocks(self) -> None:
        """On startup, restore any non-expired symbol blocks from DB."""
        if self._conn is None:
            return
        try:
            cur = self._conn.execute(
                "SELECT symbol, blocked_at, expires_at, reason FROM session_symbol_blocks"
            )
            now = datetime.now(timezone.utc)
            restored = 0
            for sym, bat, eat, reason in cur.fetchall():
                blocked_at = _parse_dt(bat) or now
                expires_at = _parse_dt(eat)
                if expires_at and now >= expires_at:
                    # Expired -- clean up
                    self._conn.execute(
                        "DELETE FROM session_symbol_blocks WHERE symbol = ?", (sym,)
                    )
                    continue
                self._symbol_blocks[sym] = SymbolBlock(
                    symbol     = sym,
                    blocked_at = blocked_at,
                    expires_at = expires_at,
                    reason     = reason or "",
                )
                restored += 1
            if restored:
                self._conn.commit()
                log.info("session_state: restored %d symbol blocks from DB", restored)
        except Exception as exc:
            log.debug("session_state: restore_symbol_blocks error -- %s", exc)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_singleton: Optional[SessionState] = None
_singleton_lock = threading.Lock()


def get_session_state(db_path: Optional[Path] = None) -> SessionState:
    """Return (or create) the module-level SessionState singleton."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = SessionState(db_path=db_path)
    return _singleton


def reset_singleton_for_testing(db_path: Optional[Path] = None) -> SessionState:
    """Force-create a new singleton. For use in tests only."""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.close()
        _singleton = SessionState(db_path=db_path)
    return _singleton
