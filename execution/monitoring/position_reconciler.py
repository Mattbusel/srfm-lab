"""
Position reconciliation for SRFM.
Compares internal ledger positions against broker-reported positions and
flags, corrects, or halts based on severity of discrepancies.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Discrepancy:
    """A single symbol whose internal and broker quantities differ."""

    symbol: str
    internal_qty: float
    broker_qty: float
    difference: float       # broker_qty - internal_qty
    difference_pct: float   # abs(difference) / max(abs(internal_qty), 1e-9) * 100
    severity: str           # "INFO" | "WARNING" | "CRITICAL"

    @staticmethod
    def compute_severity(difference_pct: float) -> str:
        if difference_pct < 1.0:
            return "INFO"
        if difference_pct < 5.0:
            return "WARNING"
        return "CRITICAL"


@dataclass
class ReconciliationResult:
    """Outcome of one full reconciliation pass."""

    status: str                              # "MATCHED" | "DISCREPANCY" | "FAILED"
    discrepancies: List[Discrepancy]
    matched_count: int
    total_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None

    @property
    def discrepancy_count(self) -> int:
        return len(self.discrepancies)

    @property
    def critical_count(self) -> int:
        return sum(1 for d in self.discrepancies if d.severity == "CRITICAL")

    @property
    def warning_count(self) -> int:
        return sum(1 for d in self.discrepancies if d.severity == "WARNING")


@dataclass
class ReconciliationRecord:
    """A row stored in the reconciliation history database."""

    record_id: int
    status: str
    matched_count: int
    total_count: int
    discrepancy_count: int
    critical_count: int
    warning_count: int
    timestamp: datetime
    discrepancies_json: str   # serialized list of Discrepancy dicts
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# SQLite-backed store
# ---------------------------------------------------------------------------

class ReconciliationStore:
    """Persists reconciliation history to SQLite."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS reconciliation_history (
        record_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        status            TEXT    NOT NULL,
        matched_count     INTEGER NOT NULL,
        total_count       INTEGER NOT NULL,
        discrepancy_count INTEGER NOT NULL,
        critical_count    INTEGER NOT NULL,
        warning_count     INTEGER NOT NULL,
        timestamp         TEXT    NOT NULL,
        discrepancies_json TEXT   NOT NULL,
        error_message     TEXT
    )
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(self._SCHEMA)
        self._conn.commit()

    def save(self, result: ReconciliationResult) -> int:
        """Persist a ReconciliationResult and return the new record_id."""
        discrepancies_json = json.dumps([asdict(d) for d in result.discrepancies])
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO reconciliation_history
                    (status, matched_count, total_count, discrepancy_count,
                     critical_count, warning_count, timestamp,
                     discrepancies_json, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.status,
                    result.matched_count,
                    result.total_count,
                    result.discrepancy_count,
                    result.critical_count,
                    result.warning_count,
                    result.timestamp.isoformat(),
                    discrepancies_json,
                    result.error_message,
                ),
            )
            self._conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def load_recent(self, n: int = 20) -> List[ReconciliationRecord]:
        """Return the n most recent reconciliation records."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT record_id, status, matched_count, total_count,
                       discrepancy_count, critical_count, warning_count,
                       timestamp, discrepancies_json, error_message
                FROM reconciliation_history
                ORDER BY record_id DESC
                LIMIT ?
                """,
                (n,),
            ).fetchall()

        records: List[ReconciliationRecord] = []
        for row in rows:
            ts = datetime.fromisoformat(row[7])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            records.append(
                ReconciliationRecord(
                    record_id=row[0],
                    status=row[1],
                    matched_count=row[2],
                    total_count=row[3],
                    discrepancy_count=row[4],
                    critical_count=row[5],
                    warning_count=row[6],
                    timestamp=ts,
                    discrepancies_json=row[8],
                    error_message=row[9],
                )
            )
        return records

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Core reconciler
# ---------------------------------------------------------------------------

class PositionReconciler:
    """
    Compares internal position ledger against broker-reported positions.

    Auto-correction policy:
    - INFO   (< 1% diff): silently update internal ledger to match broker
    - WARNING (1-5%): log warning, update internal ledger, notify
    - CRITICAL (> 5%): halt trading for affected symbol, alert immediately

    Positions are dicts of { symbol -> qty } (float).
    """

    def __init__(
        self,
        store: Optional[ReconciliationStore] = None,
        on_halt_symbol: Optional[Callable[[str, Discrepancy], None]] = None,
        on_critical_alert: Optional[Callable[[Discrepancy], None]] = None,
        on_correction: Optional[Callable[[str, float, float], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        store            -- persistence layer; defaults to in-memory SQLite
        on_halt_symbol   -- callback(symbol, discrepancy) when CRITICAL found
        on_critical_alert -- callback(discrepancy) for notifications
        on_correction    -- callback(symbol, old_qty, new_qty) after auto-correct
        """
        self._store = store or ReconciliationStore()
        self._on_halt_symbol = on_halt_symbol
        self._on_critical_alert = on_critical_alert
        self._on_correction = on_correction

        self._lock = threading.Lock()
        self._last_reconciliation_time: Optional[datetime] = None
        self._halted_symbols: Set[str] = set()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_stop = threading.Event()

        # internal positions cache -- updated by auto-correction
        self._internal_positions: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Core reconciliation
    # ------------------------------------------------------------------

    def reconcile(
        self,
        internal_positions: Dict[str, float],
        broker_positions: Dict[str, float],
    ) -> ReconciliationResult:
        """
        Perform a full reconciliation between two position snapshots.
        Applies auto-correction for INFO-level discrepancies.
        Fires halt callbacks for CRITICAL discrepancies.
        """
        with self._lock:
            self._internal_positions = dict(internal_positions)

        all_symbols = set(internal_positions) | set(broker_positions)
        discrepancies: List[Discrepancy] = []
        matched = 0

        for symbol in sorted(all_symbols):
            internal_qty = internal_positions.get(symbol, 0.0)
            broker_qty = broker_positions.get(symbol, 0.0)
            diff = broker_qty - internal_qty
            ref = max(abs(internal_qty), 1e-9)
            diff_pct = abs(diff) / ref * 100.0

            if abs(diff) < 1e-9:
                matched += 1
                continue

            severity = Discrepancy.compute_severity(diff_pct)
            d = Discrepancy(
                symbol=symbol,
                internal_qty=internal_qty,
                broker_qty=broker_qty,
                difference=diff,
                difference_pct=diff_pct,
                severity=severity,
            )
            discrepancies.append(d)
            self._handle_discrepancy(d)

        total = len(all_symbols)
        status = "MATCHED" if not discrepancies else "DISCREPANCY"

        result = ReconciliationResult(
            status=status,
            discrepancies=discrepancies,
            matched_count=matched,
            total_count=total,
        )

        with self._lock:
            self._last_reconciliation_time = result.timestamp

        try:
            self._store.save(result)
        except Exception as exc:
            logger.error("Failed to persist reconciliation result: %s", exc)

        if discrepancies:
            logger.warning(
                "Reconciliation: %d discrepancies (%d critical, %d warning) of %d symbols",
                len(discrepancies),
                result.critical_count,
                result.warning_count,
                total,
            )
        else:
            logger.info("Reconciliation: all %d symbols matched", total)

        return result

    def _handle_discrepancy(self, d: Discrepancy) -> None:
        """Apply the appropriate action for a discrepancy."""
        if d.severity == "INFO":
            # auto-correct -- silently update internal ledger
            old_qty = self._internal_positions.get(d.symbol, 0.0)
            self._internal_positions[d.symbol] = d.broker_qty
            logger.debug(
                "Auto-corrected %s: %.4f -> %.4f (%.2f%% diff)",
                d.symbol, old_qty, d.broker_qty, d.difference_pct,
            )
            if self._on_correction:
                try:
                    self._on_correction(d.symbol, old_qty, d.broker_qty)
                except Exception as exc:
                    logger.error("on_correction callback failed: %s", exc)

        elif d.severity == "WARNING":
            old_qty = self._internal_positions.get(d.symbol, 0.0)
            self._internal_positions[d.symbol] = d.broker_qty
            logger.warning(
                "WARNING discrepancy %s: internal=%.4f broker=%.4f diff=%.2f%%",
                d.symbol, d.internal_qty, d.broker_qty, d.difference_pct,
            )
            if self._on_correction:
                try:
                    self._on_correction(d.symbol, old_qty, d.broker_qty)
                except Exception as exc:
                    logger.error("on_correction callback failed: %s", exc)

        elif d.severity == "CRITICAL":
            logger.error(
                "CRITICAL discrepancy %s: internal=%.4f broker=%.4f diff=%.2f%% "
                "-- halting trading for this symbol",
                d.symbol, d.internal_qty, d.broker_qty, d.difference_pct,
            )
            with self._lock:
                self._halted_symbols.add(d.symbol)

            if self._on_halt_symbol:
                try:
                    self._on_halt_symbol(d.symbol, d)
                except Exception as exc:
                    logger.error("on_halt_symbol callback failed: %s", exc)

            if self._on_critical_alert:
                try:
                    self._on_critical_alert(d)
                except Exception as exc:
                    logger.error("on_critical_alert callback failed: %s", exc)

    # ------------------------------------------------------------------
    # Scheduled reconciliation
    # ------------------------------------------------------------------

    def schedule_reconciliation(
        self,
        interval_minutes: int = 15,
        position_fetcher: Optional[Callable[[], tuple]] = None,
    ) -> None:
        """
        Start a background thread that triggers reconciliation every
        interval_minutes.

        position_fetcher should be a callable() -> (internal_dict, broker_dict).
        If not supplied, reconcile() must be called manually.
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("Reconciliation scheduler already running")
            return

        self._scheduler_stop.clear()
        interval_seconds = interval_minutes * 60

        def _run() -> None:
            logger.info(
                "Reconciliation scheduler started -- interval %d min",
                interval_minutes,
            )
            while not self._scheduler_stop.wait(timeout=interval_seconds):
                if position_fetcher:
                    try:
                        internal, broker = position_fetcher()
                        self.reconcile(internal, broker)
                    except Exception as exc:
                        logger.error("Scheduled reconciliation error: %s", exc)
            logger.info("Reconciliation scheduler stopped")

        self._scheduler_thread = threading.Thread(
            target=_run,
            name="ReconciliationScheduler",
            daemon=True,
        )
        self._scheduler_thread.start()

    def stop_scheduler(self) -> None:
        """Stop the background reconciliation scheduler."""
        self._scheduler_stop.set()

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def last_reconciliation_time(self) -> Optional[datetime]:
        with self._lock:
            return self._last_reconciliation_time

    def reconciliation_history(self, n: int = 20) -> List[ReconciliationRecord]:
        """Return the n most recent reconciliation records from the store."""
        return self._store.load_recent(n)

    def halted_symbols(self) -> Set[str]:
        """Return set of symbols currently halted due to CRITICAL discrepancies."""
        with self._lock:
            return set(self._halted_symbols)

    def is_halted(self, symbol: str) -> bool:
        with self._lock:
            return symbol in self._halted_symbols

    def clear_halt(self, symbol: str) -> None:
        """Manually clear the trading halt for a symbol after manual review."""
        with self._lock:
            self._halted_symbols.discard(symbol)
            logger.info("Trading halt cleared for %s", symbol)

    def clear_all_halts(self) -> None:
        with self._lock:
            count = len(self._halted_symbols)
            self._halted_symbols.clear()
            logger.info("Cleared %d trading halt(s)", count)

    def corrected_positions(self) -> Dict[str, float]:
        """Return the internal positions dict after any auto-corrections."""
        with self._lock:
            return dict(self._internal_positions)
