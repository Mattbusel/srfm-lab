"""
execution/audit/trade_reconciler.py
=====================================
Trade reconciliation between OMS fill records and broker-reported fills.

The reconciler compares two sources of truth:
  1. OMS fills -- from the fills SQLite table (FillProcessor output)
  2. Broker fills -- from the broker adapter's fill report API

Discrepancy categories
----------------------
  matched         -- fills present in both sources with matching qty and price
  oms_only        -- fills in OMS but not reported by broker
  broker_only     -- fills broker reported but absent from OMS
  qty_mismatches  -- matched by order_id+timestamp but qty differs > 0.01
  price_mismatches-- matched by order_id+timestamp but price differs > 0.01%

ShadowPortfolio
---------------
Maintains a second position ledger built exclusively from broker-reported
fills.  Compared against OMS positions every hour to detect drift.
Alert threshold: > 1 share (equity) or > 0.001 units (crypto).

ReconciliationScheduler
-----------------------
Runs reconciliation at 10:00, 14:00, and 16:30 ET each trading day.
Uses a background thread with a 60-second tick loop.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytz

log = logging.getLogger("execution.trade_reconciler")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QTY_MISMATCH_THRESHOLD:   float = 0.01     -- shares / units
PRICE_MISMATCH_THRESHOLD: float = 0.0001   -- 0.01 %

EQUITY_ALERT_THRESHOLD: float = 1.0        -- shares
CRYPTO_ALERT_THRESHOLD: float = 0.001      -- crypto units

-- ET reconciliation schedule (hour, minute)
RECONCILE_SCHEDULE_ET = [(10, 0), (14, 0), (16, 30)]

ET_TZ = pytz.timezone("America/New_York")

RECON_DB_PATH = Path(__file__).parent.parent / "reconciliation.db"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass
class FillRecord:
    """
    Normalized fill record used for reconciliation matching.

    Both OMS fills and broker fills are normalized to this structure
    before comparison.
    """
    order_id:   str
    fill_id:    str
    symbol:     str
    side:       str
    fill_qty:   float
    fill_price: float
    ts_epoch:   float   -- Unix timestamp (seconds) -- used for approximate matching
    source:     str     -- 'oms' or 'broker'

    def notional(self) -> float:
        return self.fill_qty * self.fill_price


@dataclass
class QtyMismatch:
    order_id:    str
    symbol:      str
    oms_qty:     float
    broker_qty:  float
    delta:       float

    def to_dict(self) -> dict:
        return {
            "order_id":   self.order_id,
            "symbol":     self.symbol,
            "oms_qty":    self.oms_qty,
            "broker_qty": self.broker_qty,
            "delta":      self.delta,
        }


@dataclass
class PriceMismatch:
    order_id:      str
    symbol:        str
    oms_price:     float
    broker_price:  float
    delta_bps:     float

    def to_dict(self) -> dict:
        return {
            "order_id":     self.order_id,
            "symbol":       self.symbol,
            "oms_price":    self.oms_price,
            "broker_price": self.broker_price,
            "delta_bps":    self.delta_bps,
        }


@dataclass
class ReconciliationReport:
    """
    Result of a single reconciliation run.

    Attributes
    ----------
    run_ts          : UTC timestamp when the reconciliation ran
    trade_date      : Date being reconciled
    matched         : Fills present and matching in both OMS and broker
    oms_only        : Fills in OMS but absent from broker
    broker_only     : Fills broker reported but missing in OMS
    qty_mismatches  : Matched fills with qty difference > 0.01
    price_mismatches: Matched fills with price difference > 0.01%
    clean           : True if all lists except 'matched' are empty
    """
    run_ts:           str
    trade_date:       str
    matched:          List[FillRecord]       = field(default_factory=list)
    oms_only:         List[FillRecord]       = field(default_factory=list)
    broker_only:      List[FillRecord]       = field(default_factory=list)
    qty_mismatches:   List[QtyMismatch]      = field(default_factory=list)
    price_mismatches: List[PriceMismatch]    = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return (
            not self.oms_only
            and not self.broker_only
            and not self.qty_mismatches
            and not self.price_mismatches
        )

    def summary(self) -> str:
        return (
            f"Reconciliation {self.trade_date}: "
            f"matched={len(self.matched)} "
            f"oms_only={len(self.oms_only)} "
            f"broker_only={len(self.broker_only)} "
            f"qty_mismatch={len(self.qty_mismatches)} "
            f"price_mismatch={len(self.price_mismatches)} "
            f"{'CLEAN' if self.clean else 'DISCREPANCIES_FOUND'}"
        )

    def to_dict(self) -> dict:
        return {
            "run_ts":           self.run_ts,
            "trade_date":       self.trade_date,
            "matched_count":    len(self.matched),
            "oms_only":         [f.__dict__ for f in self.oms_only],
            "broker_only":      [f.__dict__ for f in self.broker_only],
            "qty_mismatches":   [q.to_dict() for q in self.qty_mismatches],
            "price_mismatches": [p.to_dict() for p in self.price_mismatches],
            "clean":            self.clean,
        }


# ---------------------------------------------------------------------------
# ShadowPortfolio
# ---------------------------------------------------------------------------

class ShadowPortfolio:
    """
    Broker-sourced position ledger maintained independently of OMS.

    Positions are built by replaying broker fill reports.  Every hour,
    the shadow portfolio is compared against OMS positions and alerts are
    raised for any differences above the threshold.

    Parameters
    ----------
    alert_callback : callable | None
        Called with (symbol, shadow_qty, oms_qty, delta) when a drift
        above threshold is detected.
    """

    def __init__(self, alert_callback=None) -> None:
        -- symbol -> net quantity (positive=long, negative=short)
        self._positions: Dict[str, float] = defaultdict(float)
        self._lock        = threading.RLock()
        self._alert_cb    = alert_callback
        self._last_check  = 0.0  -- epoch seconds of last comparison

    def apply_fill(self, fill: FillRecord) -> None:
        """
        Update shadow positions from a broker fill.

        BUY fills increase the position.  SELL fills decrease it.
        """
        with self._lock:
            side = fill.side.upper()
            if side in ("BUY", "BUY_TO_COVER"):
                self._positions[fill.symbol] += fill.fill_qty
            elif side in ("SELL", "SELL_SHORT"):
                self._positions[fill.symbol] -= fill.fill_qty
            -- clamp near-zero positions to exactly 0
            if abs(self._positions[fill.symbol]) < 1e-9:
                self._positions[fill.symbol] = 0.0

    def apply_fills(self, fills: List[FillRecord]) -> None:
        """Batch apply a list of broker fills."""
        for fill in fills:
            self.apply_fill(fill)

    def get_position(self, symbol: str) -> float:
        with self._lock:
            return self._positions.get(symbol, 0.0)

    def get_all_positions(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._positions)

    def compare_with_oms(
        self,
        oms_positions: Dict[str, float],
    ) -> List[dict]:
        """
        Compare shadow positions against OMS positions.

        Parameters
        ----------
        oms_positions : dict
            symbol -> quantity from PositionTracker (or OMS positions dict).

        Returns
        -------
        List of drift dicts: {symbol, shadow_qty, oms_qty, delta, alert_level}.
        Also calls self._alert_cb for drifts above threshold.
        """
        with self._lock:
            all_symbols = set(self._positions) | set(oms_positions)
            drifts = []

            for symbol in all_symbols:
                shadow_qty = self._positions.get(symbol, 0.0)
                oms_qty    = oms_positions.get(symbol, 0.0)
                delta      = abs(shadow_qty - oms_qty)

                if delta < 1e-9:
                    continue

                -- determine threshold based on crypto vs equity heuristic
                is_crypto = "/" in symbol
                threshold = CRYPTO_ALERT_THRESHOLD if is_crypto else EQUITY_ALERT_THRESHOLD

                alert_level = "WARN" if delta < threshold * 5 else "CRITICAL"
                if delta >= threshold:
                    alert_level = "ALERT"

                drift = {
                    "symbol":     symbol,
                    "shadow_qty": shadow_qty,
                    "oms_qty":    oms_qty,
                    "delta":      delta,
                    "alert_level": alert_level,
                }
                drifts.append(drift)

                if delta >= threshold:
                    log.warning(
                        "ShadowPortfolio DRIFT %s: shadow=%.6f oms=%.6f delta=%.6f",
                        symbol, shadow_qty, oms_qty, delta,
                    )
                    if self._alert_cb:
                        try:
                            self._alert_cb(symbol, shadow_qty, oms_qty, delta)
                        except Exception as exc:
                            log.error("ShadowPortfolio alert_callback error: %s", exc)

            self._last_check = time.time()
            return drifts

    def reset(self) -> None:
        """Clear all shadow positions (e.g. at start of new trading day)."""
        with self._lock:
            self._positions.clear()

    def seconds_since_last_check(self) -> float:
        return time.time() - self._last_check if self._last_check > 0 else float("inf")


# ---------------------------------------------------------------------------
# TradeReconciler
# ---------------------------------------------------------------------------

class TradeReconciler:
    """
    Compares OMS fill records against broker-reported fills.

    Identifies discrepancies: missing fills, quantity mismatches,
    price differences.

    Parameters
    ----------
    fills_db_path : Path | str | None
        Path to the OMS fills SQLite database.  Reads from fills table.
    recon_db_path : Path | str | None
        Path where reconciliation reports are stored.
    alert_callback : callable | None
        Called when a critical discrepancy is found.
        Signature: (report: ReconciliationReport) -> None.
    """

    def __init__(
        self,
        fills_db_path:  Optional[Path | str] = None,
        recon_db_path:  Optional[Path | str] = None,
        alert_callback: Optional[object]     = None,
    ) -> None:
        self._fills_db    = Path(fills_db_path) if fills_db_path \
                            else Path(__file__).parent.parent / "fills.db"
        self._recon_db    = Path(recon_db_path) if recon_db_path \
                            else RECON_DB_PATH
        self._recon_db.parent.mkdir(parents=True, exist_ok=True)
        self._alert_cb    = alert_callback
        self._conn        = self._connect()
        self._init_schema()
        self._lock        = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._recon_db), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS reconciliation_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_ts      TEXT    NOT NULL,
                trade_date  TEXT    NOT NULL,
                matched_cnt INTEGER NOT NULL DEFAULT 0,
                oms_only_cnt INTEGER NOT NULL DEFAULT 0,
                broker_only_cnt INTEGER NOT NULL DEFAULT 0,
                qty_mismatch_cnt INTEGER NOT NULL DEFAULT 0,
                price_mismatch_cnt INTEGER NOT NULL DEFAULT 0,
                clean       INTEGER NOT NULL DEFAULT 1,
                report_json TEXT
            )
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # OMS fill loading
    # ------------------------------------------------------------------

    def _load_oms_fills(self, trade_date: date) -> List[FillRecord]:
        """Load fills from OMS fills.db for the given trade date."""
        if not self._fills_db.exists():
            log.warning("TradeReconciler: fills.db not found at %s", self._fills_db)
            return []

        try:
            fills_conn = sqlite3.connect(str(self._fills_db), check_same_thread=False)
            fills_conn.execute("PRAGMA journal_mode=WAL")
            cur = fills_conn.execute(
                """
                SELECT order_id, fill_id, symbol, side,
                       fill_qty, fill_price, fill_ts
                FROM fills
                WHERE trade_date = ?
                ORDER BY fill_ts ASC
                """,
                (trade_date.isoformat(),),
            )
            records = []
            for row in cur.fetchall():
                order_id, fill_id, symbol, side, qty, price, fill_ts = row
                try:
                    ts = datetime.fromisoformat(fill_ts).timestamp()
                except Exception:
                    ts = 0.0
                records.append(FillRecord(
                    order_id   = order_id,
                    fill_id    = fill_id,
                    symbol     = symbol,
                    side       = side,
                    fill_qty   = qty,
                    fill_price = price,
                    ts_epoch   = ts,
                    source     = "oms",
                ))
            fills_conn.close()
            return records
        except Exception as exc:
            log.error("TradeReconciler: error loading OMS fills: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Fill matching logic
    # ------------------------------------------------------------------

    @staticmethod
    def _make_fill_key(fill: FillRecord) -> str:
        """Primary match key: order_id + fill_id."""
        return f"{fill.order_id}::{fill.fill_id}"

    @staticmethod
    def _approximate_time_match(ts_a: float, ts_b: float, window_s: float = 5.0) -> bool:
        """True if two timestamps are within window_s seconds of each other."""
        return abs(ts_a - ts_b) <= window_s

    def reconcile_fills(
        self,
        oms_fills:    List[FillRecord],
        broker_fills: List[FillRecord],
    ) -> ReconciliationReport:
        """
        Compare OMS fill records against broker-reported fills.

        Matching is primary by fill_id.  If fill_id is absent or mismatched,
        falls back to order_id + approximate timestamp (5-second window).

        Parameters
        ----------
        oms_fills    : Normalized FillRecord list from OMS.
        broker_fills : Normalized FillRecord list from broker report.

        Returns
        -------
        ReconciliationReport with all discrepancy categories populated.
        """
        run_ts = datetime.now(timezone.utc).isoformat()
        trade_date = date.today().isoformat()

        -- build lookup dicts
        oms_by_id:    Dict[str, FillRecord] = {self._make_fill_key(f): f for f in oms_fills}
        broker_by_id: Dict[str, FillRecord] = {self._make_fill_key(f): f for f in broker_fills}

        matched:          List[FillRecord]    = []
        oms_only:         List[FillRecord]    = []
        broker_only:      List[FillRecord]    = []
        qty_mismatches:   List[QtyMismatch]   = []
        price_mismatches: List[PriceMismatch] = []

        all_keys = set(oms_by_id) | set(broker_by_id)

        for key in all_keys:
            oms_fill    = oms_by_id.get(key)
            broker_fill = broker_by_id.get(key)

            if oms_fill and broker_fill:
                -- both present -- check for value discrepancies
                qty_delta = abs(oms_fill.fill_qty - broker_fill.fill_qty)
                if qty_delta > QTY_MISMATCH_THRESHOLD:
                    qty_mismatches.append(QtyMismatch(
                        order_id    = oms_fill.order_id,
                        symbol      = oms_fill.symbol,
                        oms_qty     = oms_fill.fill_qty,
                        broker_qty  = broker_fill.fill_qty,
                        delta       = qty_delta,
                    ))
                    log.warning(
                        "QTY_MISMATCH order=%s oms=%.6f broker=%.6f delta=%.6f",
                        oms_fill.order_id, oms_fill.fill_qty,
                        broker_fill.fill_qty, qty_delta,
                    )

                ref_price = broker_fill.fill_price
                if ref_price > 0:
                    price_dev = abs(oms_fill.fill_price - broker_fill.fill_price) / ref_price
                    if price_dev > PRICE_MISMATCH_THRESHOLD:
                        delta_bps = price_dev * 10_000
                        price_mismatches.append(PriceMismatch(
                            order_id     = oms_fill.order_id,
                            symbol       = oms_fill.symbol,
                            oms_price    = oms_fill.fill_price,
                            broker_price = broker_fill.fill_price,
                            delta_bps    = delta_bps,
                        ))
                        log.warning(
                            "PRICE_MISMATCH order=%s oms=%.4f broker=%.4f %.2fbps",
                            oms_fill.order_id, oms_fill.fill_price,
                            broker_fill.fill_price, delta_bps,
                        )

                matched.append(oms_fill)

            elif oms_fill and not broker_fill:
                -- attempt approximate time-based match in broker_fills
                found = False
                for bf in broker_fills:
                    if (bf.order_id == oms_fill.order_id
                            and self._approximate_time_match(oms_fill.ts_epoch, bf.ts_epoch)):
                        found = True
                        matched.append(oms_fill)
                        break
                if not found:
                    oms_only.append(oms_fill)
                    log.warning(
                        "OMS_ONLY fill: order=%s fill_id=%s symbol=%s qty=%.4f",
                        oms_fill.order_id, oms_fill.fill_id,
                        oms_fill.symbol, oms_fill.fill_qty,
                    )

            else:
                -- broker_fill only
                broker_only.append(broker_fill)
                log.warning(
                    "BROKER_ONLY fill: order=%s fill_id=%s symbol=%s qty=%.4f",
                    broker_fill.order_id, broker_fill.fill_id,
                    broker_fill.symbol, broker_fill.fill_qty,
                )

        report = ReconciliationReport(
            run_ts           = run_ts,
            trade_date       = trade_date,
            matched          = matched,
            oms_only         = oms_only,
            broker_only      = broker_only,
            qty_mismatches   = qty_mismatches,
            price_mismatches = price_mismatches,
        )

        self._persist_report(report)

        if not report.clean and self._alert_cb:
            try:
                self._alert_cb(report)
            except Exception as exc:
                log.error("TradeReconciler alert_callback error: %s", exc)

        log.info(report.summary())
        return report

    def reconcile_for_date(
        self,
        broker_fills: List[FillRecord],
        trade_date:   Optional[date] = None,
    ) -> ReconciliationReport:
        """
        Load OMS fills for trade_date and reconcile against broker_fills.

        Parameters
        ----------
        broker_fills : Broker-sourced FillRecord list.
        trade_date   : Date to reconcile.  Defaults to today.
        """
        target = trade_date or date.today()
        oms_fills = self._load_oms_fills(target)
        return self.reconcile_fills(oms_fills, broker_fills)

    # ------------------------------------------------------------------
    # Report persistence
    # ------------------------------------------------------------------

    def _persist_report(self, report: ReconciliationReport) -> None:
        import json
        try:
            self._conn.execute(
                """
                INSERT INTO reconciliation_runs
                    (run_ts, trade_date, matched_cnt, oms_only_cnt,
                     broker_only_cnt, qty_mismatch_cnt, price_mismatch_cnt,
                     clean, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.run_ts,
                    report.trade_date,
                    len(report.matched),
                    len(report.oms_only),
                    len(report.broker_only),
                    len(report.qty_mismatches),
                    len(report.price_mismatches),
                    1 if report.clean else 0,
                    json.dumps(report.to_dict(), default=str),
                ),
            )
            self._conn.commit()
        except Exception as exc:
            log.error("TradeReconciler: failed to persist report: %s", exc)

    def get_recent_reports(self, limit: int = 20) -> List[dict]:
        """Return the most recent reconciliation report summaries."""
        cur = self._conn.execute(
            """
            SELECT run_ts, trade_date, matched_cnt, oms_only_cnt,
                   broker_only_cnt, qty_mismatch_cnt, price_mismatch_cnt, clean
            FROM reconciliation_runs
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        )
        cols = ["run_ts", "trade_date", "matched_cnt", "oms_only_cnt",
                "broker_only_cnt", "qty_mismatch_cnt", "price_mismatch_cnt", "clean"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ReconciliationScheduler
# ---------------------------------------------------------------------------

class ReconciliationScheduler:
    """
    Runs trade reconciliation at 10:00, 14:00, and 16:30 ET each trading day.

    Parameters
    ----------
    reconciler : TradeReconciler
        The reconciler instance to invoke.
    broker_fill_fetcher : callable
        Callable(trade_date: date) -> List[FillRecord] that fetches
        broker fills for the given date.
    oms_position_fetcher : callable | None
        Callable() -> Dict[str, float] returning OMS positions.
        Used for the ShadowPortfolio hourly comparison.
    """

    def __init__(
        self,
        reconciler:          TradeReconciler,
        broker_fill_fetcher: object,
        oms_position_fetcher: Optional[object] = None,
    ) -> None:
        self._reconciler         = reconciler
        self._fill_fetcher       = broker_fill_fetcher
        self._position_fetcher   = oms_position_fetcher
        self._shadow             = ShadowPortfolio()
        self._running            = False
        self._thread: Optional[threading.Thread] = None
        self._lock               = threading.Lock()

    def start(self) -> None:
        """Start the background scheduler thread."""
        with self._lock:
            if self._running:
                log.warning("ReconciliationScheduler already running")
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._loop,
                name="recon-scheduler",
                daemon=True,
            )
            self._thread.start()
            log.info("ReconciliationScheduler started")

    def stop(self) -> None:
        """Stop the scheduler thread gracefully."""
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        log.info("ReconciliationScheduler stopped")

    def _should_run_now(self, now_et: datetime) -> bool:
        """True if now_et matches any scheduled reconciliation time (within 60s)."""
        for hour, minute in RECONCILE_SCHEDULE_ET:
            sched = now_et.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if abs((now_et - sched).total_seconds()) < 60:
                return True
        return False

    def _loop(self) -> None:
        """Background thread: 60-second tick, fires reconciliation at scheduled times."""
        last_run_minute = -1

        while self._running:
            try:
                now_utc = datetime.now(timezone.utc)
                now_et  = now_utc.astimezone(ET_TZ)

                -- deduplicate: only run once per scheduled minute
                current_minute = now_et.hour * 60 + now_et.minute
                if current_minute != last_run_minute and self._should_run_now(now_et):
                    last_run_minute = current_minute
                    self._run_scheduled_reconciliation(now_utc.date())

                -- hourly shadow portfolio comparison
                if self._position_fetcher and self._shadow.seconds_since_last_check() > 3600:
                    try:
                        oms_pos = self._position_fetcher()
                        drifts  = self._shadow.compare_with_oms(oms_pos)
                        if drifts:
                            log.warning(
                                "ShadowPortfolio hourly check: %d drift(s) found",
                                len(drifts),
                            )
                    except Exception as exc:
                        log.error("ShadowPortfolio comparison error: %s", exc)

            except Exception as exc:
                log.error("ReconciliationScheduler loop error: %s", exc)

            time.sleep(60)

    def _run_scheduled_reconciliation(self, trade_date: date) -> None:
        """Execute a reconciliation run and update the shadow portfolio."""
        log.info(
            "ReconciliationScheduler: running scheduled reconciliation for %s",
            trade_date.isoformat(),
        )
        try:
            broker_fills = self._fill_fetcher(trade_date)
            -- update shadow portfolio with any new broker fills
            self._shadow.apply_fills(broker_fills)
            -- run full reconciliation
            report = self._reconciler.reconcile_for_date(broker_fills, trade_date)
            log.info("Scheduled reconciliation complete: %s", report.summary())
        except Exception as exc:
            log.error("Scheduled reconciliation failed: %s", exc)

    def run_now(self, trade_date: Optional[date] = None) -> Optional[ReconciliationReport]:
        """
        Trigger an immediate reconciliation run (blocking).

        Parameters
        ----------
        trade_date : date | None -- defaults to today.

        Returns
        -------
        ReconciliationReport or None on error.
        """
        target = trade_date or date.today()
        try:
            broker_fills = self._fill_fetcher(target)
            self._shadow.apply_fills(broker_fills)
            return self._reconciler.reconcile_for_date(broker_fills, target)
        except Exception as exc:
            log.error("ReconciliationScheduler.run_now error: %s", exc)
            return None

    @property
    def shadow_portfolio(self) -> ShadowPortfolio:
        return self._shadow
