"""
execution/oms/state_machine.py
================================
Order lifecycle state machine for the SRFM OMS layer.

The state machine enforces that every order follows a valid transition path.
Invalid transitions are rejected with a clear error message.  Every
successful transition is written to an immutable audit trail in SQLite.

Valid transitions
-----------------
  NEW             -> PENDING_SUBMIT
  PENDING_SUBMIT  -> SUBMITTED | REJECTED
  SUBMITTED       -> PARTIAL_FILL | FILLED | CANCELLED | REJECTED
  PARTIAL_FILL    -> FILLED | CANCELLED
  FILLED          -> (terminal -- no transitions allowed)
  CANCELLED       -> (terminal -- no transitions allowed)
  REJECTED        -> (terminal -- no transitions allowed)

Design notes
------------
- All state is stored in memory (dict) plus SQLite for recovery.
- Thread safety: RLock on all public methods.
- StateTransitionLogger is embedded -- no separate persistence object needed.
- The machine can be reconstructed from SQLite on restart via load_from_db().
- get_open_orders() returns order_ids in any non-terminal state, suitable
  for the monitoring layer to poll.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger("execution.state_machine")

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

-- all valid states
STATE_NEW:            str = "NEW"
STATE_PENDING_SUBMIT: str = "PENDING_SUBMIT"
STATE_SUBMITTED:      str = "SUBMITTED"
STATE_PARTIAL_FILL:   str = "PARTIAL_FILL"
STATE_FILLED:         str = "FILLED"
STATE_CANCELLED:      str = "CANCELLED"
STATE_REJECTED:       str = "REJECTED"

TERMINAL_STATES: frozenset[str] = frozenset({
    STATE_FILLED,
    STATE_CANCELLED,
    STATE_REJECTED,
})

VALID_TRANSITIONS: Dict[str, List[str]] = {
    STATE_NEW:            [STATE_PENDING_SUBMIT],
    STATE_PENDING_SUBMIT: [STATE_SUBMITTED, STATE_REJECTED],
    STATE_SUBMITTED:      [STATE_PARTIAL_FILL, STATE_FILLED,
                           STATE_CANCELLED, STATE_REJECTED],
    STATE_PARTIAL_FILL:   [STATE_FILLED, STATE_CANCELLED],
    STATE_FILLED:         [],
    STATE_CANCELLED:      [],
    STATE_REJECTED:       [],
}

SM_DB_PATH = Path(__file__).parent.parent / "state_machine.db"


# ---------------------------------------------------------------------------
# StateTransitionLogger
# ---------------------------------------------------------------------------

class StateTransitionLogger:
    """
    Immutable SQLite-backed log of every order state transition.

    Schema
    ------
        state_transitions (
            id          INTEGER PRIMARY KEY,
            order_id    TEXT    NOT NULL,
            from_state  TEXT    NOT NULL,
            to_state    TEXT    NOT NULL,
            ts_epoch_ns INTEGER NOT NULL,   -- nanosecond epoch timestamp
            ts_iso      TEXT    NOT NULL,
            reason      TEXT
        )

    The logger is append-only.  Rows are never updated or deleted.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS state_transitions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id    TEXT    NOT NULL,
                from_state  TEXT    NOT NULL,
                to_state    TEXT    NOT NULL,
                ts_epoch_ns INTEGER NOT NULL,
                ts_iso      TEXT    NOT NULL,
                reason      TEXT
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_st_order ON state_transitions (order_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_st_ts ON state_transitions (ts_epoch_ns)"
        )
        self._conn.commit()

    def log_transition(
        self,
        order_id:   str,
        from_state: str,
        to_state:   str,
        reason:     str = "",
    ) -> None:
        """Write a state transition record."""
        now     = datetime.now(timezone.utc)
        ts_ns   = int(now.timestamp() * 1_000_000_000)
        ts_iso  = now.isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO state_transitions
                    (order_id, from_state, to_state, ts_epoch_ns, ts_iso, reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (order_id, from_state, to_state, ts_ns, ts_iso, reason or None),
            )
            self._conn.commit()

    def get_history(self, order_id: str) -> List[Tuple[str, int]]:
        """
        Return state transition history for order_id.

        Returns list of (state, ts_epoch_ns) tuples ordered by time.
        The first element represents the initial NEW state arrival,
        subsequent elements are each transition's to_state.
        """
        cur = self._conn.execute(
            """
            SELECT to_state, ts_epoch_ns
            FROM state_transitions
            WHERE order_id = ?
            ORDER BY ts_epoch_ns ASC
            """,
            (order_id,),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]

    def get_all_transitions(self, order_id: str) -> List[dict]:
        """Return full transition records for order_id."""
        cur = self._conn.execute(
            """
            SELECT id, order_id, from_state, to_state,
                   ts_epoch_ns, ts_iso, reason
            FROM state_transitions
            WHERE order_id = ?
            ORDER BY ts_epoch_ns ASC
            """,
            (order_id,),
        )
        cols = ["id", "order_id", "from_state", "to_state",
                "ts_epoch_ns", "ts_iso", "reason"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_recent_transitions(self, limit: int = 100) -> List[dict]:
        """Return the most recent transitions across all orders."""
        cur = self._conn.execute(
            """
            SELECT id, order_id, from_state, to_state,
                   ts_epoch_ns, ts_iso, reason
            FROM state_transitions
            ORDER BY ts_epoch_ns DESC LIMIT ?
            """,
            (limit,),
        )
        cols = ["id", "order_id", "from_state", "to_state",
                "ts_epoch_ns", "ts_iso", "reason"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def count_transitions(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM state_transitions")
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# OrderStateMachine
# ---------------------------------------------------------------------------

@dataclass
class _OrderState:
    """Internal state record for a single order."""
    order_id: str
    status:   str = STATE_NEW
    -- list of (status, ts_epoch_ns) for in-memory history
    history:  List[Tuple[str, int]] = field(default_factory=list)


class OrderStateMachine:
    """
    Enforces valid order state transitions.

    NEW -> PENDING_SUBMIT -> SUBMITTED -> PARTIAL_FILL -> FILLED
                                      -> CANCELLED
                                      -> REJECTED

    Parameters
    ----------
    db_path : Path | str | None
        Path to the SQLite state machine database.  Defaults to
        execution/state_machine.db.

    Thread safety
    -------------
    All public methods acquire self._lock (RLock) before accessing
    internal state.
    """

    VALID_TRANSITIONS = VALID_TRANSITIONS

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self._db_path   = Path(db_path) if db_path else SM_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn      = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._logger    = StateTransitionLogger(self._conn)
        self._states:   Dict[str, _OrderState] = {}
        self._lock      = threading.RLock()

    # ------------------------------------------------------------------
    # Order registration
    # ------------------------------------------------------------------

    def register(self, order_id: str) -> bool:
        """
        Register a new order in state NEW.

        Returns True on success, False if the order_id is already registered.
        """
        with self._lock:
            if order_id in self._states:
                log.warning("state_machine.register: order %s already registered", order_id)
                return False

            now_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
            state  = _OrderState(
                order_id = order_id,
                status   = STATE_NEW,
                history  = [(STATE_NEW, now_ns)],
            )
            self._states[order_id] = state

            -- log the initial NEW state as a transition from GENESIS
            self._logger.log_transition(
                order_id   = order_id,
                from_state = "GENESIS",
                to_state   = STATE_NEW,
                reason     = "order_registered",
            )
            log.debug("state_machine.register: order %s -> NEW", order_id)
            return True

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def transition(
        self,
        order_id:  str,
        new_status: str,
        reason:    str = "",
    ) -> bool:
        """
        Validate and apply a state transition.

        Parameters
        ----------
        order_id   : OMS order ID.
        new_status : Target state (must be in VALID_TRANSITIONS[current]).
        reason     : Optional human-readable rationale (logged to audit trail).

        Returns
        -------
        True on success, False if the transition is invalid or the order
        is unknown.
        """
        with self._lock:
            state = self._states.get(order_id)
            if state is None:
                log.error(
                    "state_machine.transition: unknown order_id=%s", order_id
                )
                return False

            current = state.status
            allowed = VALID_TRANSITIONS.get(current, [])

            if new_status not in allowed:
                log.warning(
                    "state_machine.transition INVALID: order=%s %s -> %s "
                    "(allowed: %s)",
                    order_id, current, new_status, allowed,
                )
                return False

            now_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
            state.status = new_status
            state.history.append((new_status, now_ns))

            self._logger.log_transition(
                order_id   = order_id,
                from_state = current,
                to_state   = new_status,
                reason     = reason,
            )

            log.info(
                "state_machine: order=%s %s -> %s%s",
                order_id, current, new_status,
                f" ({reason})" if reason else "",
            )
            return True

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_status(self, order_id: str) -> Optional[str]:
        """
        Return the current status string for order_id.

        Returns None if the order_id is not registered.
        """
        with self._lock:
            state = self._states.get(order_id)
            return state.status if state else None

    def get_history(self, order_id: str) -> List[Tuple[str, int]]:
        """
        Return the in-memory state history for order_id.

        Returns a list of (status, ts_epoch_ns) tuples in chronological order.
        For the full persisted history (including across restarts), use
        get_persisted_history().

        Returns empty list if order_id is unknown.
        """
        with self._lock:
            state = self._states.get(order_id)
            if state is None:
                return []
            return list(state.history)

    def get_persisted_history(self, order_id: str) -> List[Tuple[str, int]]:
        """
        Return the full state history from SQLite for order_id.

        Useful for reconstruction after a restart.  Returns list of
        (to_state, ts_epoch_ns) tuples.
        """
        return self._logger.get_history(order_id)

    def is_terminal(self, order_id: str) -> bool:
        """
        Return True if the order is in a terminal state (FILLED, CANCELLED,
        REJECTED) or if the order_id is unknown.
        """
        with self._lock:
            state = self._states.get(order_id)
            if state is None:
                return True  -- unknown orders are treated as terminal
            return state.status in TERMINAL_STATES

    def get_open_orders(self) -> List[str]:
        """
        Return a list of all order_ids currently in non-terminal states.

        Suitable for the monitoring layer to iterate over live orders.
        """
        with self._lock:
            return [
                oid for oid, state in self._states.items()
                if state.status not in TERMINAL_STATES
            ]

    def get_orders_in_state(self, status: str) -> List[str]:
        """Return all order_ids with the given status."""
        with self._lock:
            return [
                oid for oid, state in self._states.items()
                if state.status == status
            ]

    def get_terminal_orders(self) -> List[str]:
        """Return all order_ids that have reached a terminal state."""
        with self._lock:
            return [
                oid for oid, state in self._states.items()
                if state.status in TERMINAL_STATES
            ]

    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------

    def status_counts(self) -> Dict[str, int]:
        """Return {status: count} for all registered orders."""
        with self._lock:
            counts: Dict[str, int] = {}
            for state in self._states.values():
                counts[state.status] = counts.get(state.status, 0) + 1
            return counts

    def is_registered(self, order_id: str) -> bool:
        """Return True if order_id is in the state machine."""
        with self._lock:
            return order_id in self._states

    def order_count(self) -> int:
        """Return total number of registered orders."""
        with self._lock:
            return len(self._states)

    # ------------------------------------------------------------------
    # Persistence and recovery
    # ------------------------------------------------------------------

    def snapshot_to_db(self) -> None:
        """
        Persist current in-memory state to a snapshot table in SQLite.

        This is a convenience method for checkpointing.  The primary
        source of truth is the state_transitions table; this snapshot
        allows faster recovery.
        """
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS order_state_snapshot (
                    order_id    TEXT PRIMARY KEY,
                    status      TEXT NOT NULL,
                    snapshot_ts TEXT NOT NULL
                )
            """)
            now_iso = datetime.now(timezone.utc).isoformat()
            for oid, state in self._states.items():
                self._conn.execute(
                    """
                    INSERT INTO order_state_snapshot (order_id, status, snapshot_ts)
                    VALUES (?, ?, ?)
                    ON CONFLICT(order_id) DO UPDATE SET
                        status      = excluded.status,
                        snapshot_ts = excluded.snapshot_ts
                    """,
                    (oid, state.status, now_iso),
                )
            self._conn.commit()
        log.info("state_machine snapshot written: %d orders", len(self._states))

    def load_from_snapshot(self) -> int:
        """
        Reload in-memory state from the snapshot table.

        Returns the number of orders loaded.  Only loads orders that are
        not already in self._states so this is safe to call on a warm machine.
        """
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT order_id, status FROM order_state_snapshot"
                )
            except sqlite3.OperationalError:
                return 0  -- snapshot table does not yet exist

            count = 0
            for order_id, status in cur.fetchall():
                if order_id not in self._states:
                    self._states[order_id] = _OrderState(
                        order_id = order_id,
                        status   = status,
                        history  = [(status, 0)],
                    )
                    count += 1
            log.info("state_machine loaded %d orders from snapshot", count)
            return count

    # ------------------------------------------------------------------
    # Validation helpers (static -- callable without instance)
    # ------------------------------------------------------------------

    @staticmethod
    def is_valid_transition(from_state: str, to_state: str) -> bool:
        """Return True if the transition from_state -> to_state is valid."""
        return to_state in VALID_TRANSITIONS.get(from_state, [])

    @staticmethod
    def reachable_from(from_state: str) -> List[str]:
        """Return all states directly reachable from from_state."""
        return list(VALID_TRANSITIONS.get(from_state, []))

    # ------------------------------------------------------------------
    # Transition logger access
    # ------------------------------------------------------------------

    @property
    def transition_logger(self) -> StateTransitionLogger:
        """Access the underlying StateTransitionLogger for direct queries."""
        return self._logger

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        with self._lock:
            counts = self.status_counts()
        return f"<OrderStateMachine orders={self.order_count()} states={counts}>"
