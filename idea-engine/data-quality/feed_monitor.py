"""
FeedMonitor
===========
Live data feed health monitoring for the Idea Automation Engine.

Tracks:
  - Feed staleness (time since last received bar)
  - Feed outages (gap > threshold)
  - Cross-exchange price deviation
  - Continuous heartbeat monitoring loop

Results are persisted in the ``feed_health`` table and written as narrative
alerts into ``narrative_alerts`` (if the table exists).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH_ENV = "IDEA_ENGINE_DB"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "idea_engine.db"

DEFAULT_MAX_STALENESS_SECONDS = 120
DEFAULT_OUTAGE_THRESHOLD_MINUTES = 5
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30
EXCHANGE_DEVIATION_PCT_THRESHOLD = 0.005  # 0.5 %


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeedHealth:
    """Snapshot health state for one symbol."""

    symbol: str
    is_healthy: bool
    latency_ms: float | None        # round-trip latency if available
    price: float | None             # last known price
    last_update: str | None         # ISO timestamp of last received bar
    staleness_seconds: float | None # seconds since last bar
    issue_type: str | None          # None | "stale" | "outage" | "missing"
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def status_emoji(self) -> str:
        return "OK" if self.is_healthy else f"UNHEALTHY({self.issue_type})"


@dataclass
class PriceDeviation:
    """Cross-exchange price comparison result."""

    symbol: str
    exchange_a: str
    exchange_b: str
    price_a: float
    price_b: float
    deviation_pct: float
    is_divergent: bool
    threshold_pct: float
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# FeedMonitor
# ---------------------------------------------------------------------------

class FeedMonitor:
    """
    Monitors live data feed health for a set of symbols.

    Typical usage:

    .. code-block:: python

        monitor = FeedMonitor(db_path="idea_engine.db")
        monitor.register_price_callback("BTC/USDT", my_price_getter)
        monitor.heartbeat_monitor(["BTC/USDT", "ETH/USDT"], interval=30)

    Parameters
    ----------
    db_path : path to idea_engine.db.
    max_staleness_seconds : default staleness threshold.
    outage_threshold_minutes : default outage threshold.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        max_staleness_seconds: int = DEFAULT_MAX_STALENESS_SECONDS,
        outage_threshold_minutes: int = DEFAULT_OUTAGE_THRESHOLD_MINUTES,
    ) -> None:
        self.db_path = Path(
            db_path
            or __import__("os").environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)
        )
        self.max_staleness_seconds = max_staleness_seconds
        self.outage_threshold_minutes = outage_threshold_minutes

        # Internal state
        self._last_seen: dict[str, datetime] = {}      # symbol → last update time
        self._last_price: dict[str, float] = {}        # symbol → last price
        self._price_callbacks: dict[str, Callable[[], float | None]] = {}
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()
        self._lock = threading.Lock()

        self._ensure_schema()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_price_callback(
        self,
        symbol: str,
        callback: Callable[[], float | None],
    ) -> None:
        """
        Register a callable that returns the latest price for *symbol*.

        The callback is invoked during heartbeat checks.  It should return
        ``None`` if the price is unavailable.
        """
        with self._lock:
            self._price_callbacks[symbol] = callback
        logger.info("Registered price callback for %s.", symbol)

    def update_price(self, symbol: str, price: float) -> None:
        """Push a new price tick for *symbol* (thread-safe)."""
        with self._lock:
            self._last_price[symbol] = price
            self._last_seen[symbol] = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_feed_health(
        self,
        symbol: str,
        max_staleness_seconds: int | None = None,
    ) -> FeedHealth:
        """
        Return a :class:`FeedHealth` snapshot for *symbol*.

        If a price callback is registered, it is invoked to get the latest
        price.  Otherwise the most recently pushed price is used.

        Parameters
        ----------
        symbol : ticker symbol.
        max_staleness_seconds : override the instance default.
        """
        threshold = max_staleness_seconds or self.max_staleness_seconds
        now = datetime.now(timezone.utc)

        # Try callback first
        price: float | None = None
        latency_ms: float | None = None

        callback = self._price_callbacks.get(symbol)
        if callback is not None:
            t0 = time.perf_counter()
            try:
                price = callback()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                if price is not None:
                    with self._lock:
                        self._last_price[symbol] = price
                        self._last_seen[symbol] = now
            except Exception as exc:  # noqa: BLE001
                logger.warning("Price callback for %s raised: %s", symbol, exc)

        with self._lock:
            last_seen = self._last_seen.get(symbol)
            if price is None:
                price = self._last_price.get(symbol)

        staleness: float | None = None
        issue_type: str | None = None
        is_healthy = True
        last_update_str: str | None = None

        if last_seen is None:
            issue_type = "missing"
            is_healthy = False
        else:
            staleness = (now - last_seen).total_seconds()
            last_update_str = last_seen.strftime("%Y-%m-%dT%H:%M:%SZ")
            if staleness > threshold:
                is_healthy = False
                if staleness > self.outage_threshold_minutes * 60:
                    issue_type = "outage"
                else:
                    issue_type = "stale"

        health = FeedHealth(
            symbol=symbol,
            is_healthy=is_healthy,
            latency_ms=latency_ms,
            price=price,
            last_update=last_update_str,
            staleness_seconds=staleness,
            issue_type=issue_type,
        )

        self._persist_health(health)

        if not is_healthy:
            logger.warning(
                "Feed unhealthy for %s: %s (staleness=%.0fs)",
                symbol, issue_type, staleness or -1,
            )

        return health

    def detect_feed_outage(
        self,
        symbol: str,
        outage_threshold_minutes: int | None = None,
    ) -> bool:
        """
        Return True if *symbol* has been silent longer than *outage_threshold_minutes*.
        """
        threshold = outage_threshold_minutes or self.outage_threshold_minutes
        with self._lock:
            last_seen = self._last_seen.get(symbol)

        if last_seen is None:
            return True

        elapsed_minutes = (datetime.now(timezone.utc) - last_seen).total_seconds() / 60
        return elapsed_minutes >= threshold

    def compare_exchanges(
        self,
        symbol: str,
        exchange_a_price: float,
        exchange_b_price: float,
        exchange_a: str = "exchange_a",
        exchange_b: str = "exchange_b",
        threshold_pct: float = EXCHANGE_DEVIATION_PCT_THRESHOLD,
    ) -> PriceDeviation:
        """
        Compare prices from two exchanges and flag if they diverge > *threshold_pct*.

        Parameters
        ----------
        symbol          : ticker string.
        exchange_a_price, exchange_b_price : current mid prices.
        exchange_a, exchange_b : exchange labels for reporting.
        threshold_pct   : fractional threshold (default 0.005 = 0.5 %).
        """
        if exchange_a_price <= 0 or exchange_b_price <= 0:
            raise ValueError("Prices must be positive.")

        mid = (exchange_a_price + exchange_b_price) / 2.0
        deviation_pct = abs(exchange_a_price - exchange_b_price) / mid
        is_divergent = deviation_pct > threshold_pct

        result = PriceDeviation(
            symbol=symbol,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            price_a=exchange_a_price,
            price_b=exchange_b_price,
            deviation_pct=deviation_pct,
            is_divergent=is_divergent,
            threshold_pct=threshold_pct,
        )

        if is_divergent:
            msg = (
                f"Price divergence on {symbol}: {exchange_a}={exchange_a_price:.4f} "
                f"vs {exchange_b}={exchange_b_price:.4f} "
                f"({deviation_pct:.2%} > {threshold_pct:.2%} threshold)"
            )
            logger.warning(msg)
            self.generate_feed_alert(
                symbol,
                issue_type="price_deviation",
                detail=msg,
            )

        return result

    # ------------------------------------------------------------------
    # Heartbeat monitoring loop
    # ------------------------------------------------------------------

    def heartbeat_monitor(
        self,
        symbols: list[str],
        interval: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        on_unhealthy: Callable[[str, FeedHealth], None] | None = None,
        max_iterations: int | None = None,
    ) -> None:
        """
        Blocking continuous monitoring loop that checks each symbol in *symbols*
        every *interval* seconds.

        To run non-blocking, call ``start_heartbeat_thread`` instead.

        Parameters
        ----------
        symbols        : list of ticker symbols to monitor.
        interval       : seconds between full sweeps.
        on_unhealthy   : optional callback invoked with (symbol, FeedHealth) for
                         each unhealthy symbol.
        max_iterations : if set, stop after this many sweeps (useful for testing).
        """
        logger.info(
            "Heartbeat monitor started: %d symbols, interval=%ds.",
            len(symbols), interval,
        )
        iteration = 0
        while not self._heartbeat_stop.is_set():
            for symbol in symbols:
                try:
                    health = self.check_feed_health(symbol)
                    if not health.is_healthy:
                        if on_unhealthy:
                            on_unhealthy(symbol, health)
                        self.generate_feed_alert(symbol, health.issue_type or "unknown")
                except Exception as exc:  # noqa: BLE001
                    logger.error("Heartbeat check failed for %s: %s", symbol, exc)

            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break
            self._heartbeat_stop.wait(timeout=interval)

        logger.info("Heartbeat monitor stopped.")

    def start_heartbeat_thread(
        self,
        symbols: list[str],
        interval: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        on_unhealthy: Callable[[str, FeedHealth], None] | None = None,
    ) -> threading.Thread:
        """
        Start ``heartbeat_monitor`` in a daemon thread and return it.

        Call ``stop_heartbeat()`` to request graceful shutdown.
        """
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self.heartbeat_monitor,
            args=(symbols,),
            kwargs={"interval": interval, "on_unhealthy": on_unhealthy},
            daemon=True,
            name="FeedMonitor-heartbeat",
        )
        self._heartbeat_thread.start()
        return self._heartbeat_thread

    def stop_heartbeat(self, timeout: float = 5.0) -> None:
        """Request the heartbeat thread to stop and wait for it."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    def generate_feed_alert(
        self,
        symbol: str,
        issue_type: str,
        detail: str = "",
    ) -> None:
        """
        Write a feed alert to the ``narrative_alerts`` table in the DB, if it
        exists.  Also emits a logger.warning.
        """
        msg = (
            f"FEED ALERT [{symbol}] issue_type={issue_type}"
            + (f": {detail}" if detail else "")
        )
        logger.warning(msg)

        if not self.db_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self.db_path))
            # Check table exists first
            tbl_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='narrative_alerts'"
            ).fetchone()
            if tbl_exists:
                conn.execute(
                    """
                    INSERT INTO narrative_alerts
                        (alert_type, symbol, message, severity, created_at)
                    VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
                    """,
                    ("feed_health", symbol, msg, "warning"),
                )
                conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.warning("Failed to persist feed alert: %s", exc)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_history(
        self,
        symbol: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return recent heartbeat records for *symbol* from the DB."""
        if not self.db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM feed_health WHERE symbol = ? "
                "ORDER BY ts DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logger.warning("Failed to load feed history: %s", exc)
            return []

    def unhealthy_symbols(self) -> list[str]:
        """Return symbols that have had any unhealthy record in the last hour."""
        if not self.db_path.exists():
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        try:
            conn = sqlite3.connect(str(self.db_path))
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM feed_health "
                "WHERE is_healthy = 0 AND ts > ?",
                (cutoff,),
            ).fetchall()
            conn.close()
            return [r[0] for r in rows]
        except sqlite3.Error as exc:
            logger.warning("Failed to query unhealthy symbols: %s", exc)
            return []

    def uptime_pct(self, symbol: str, hours: int = 24) -> float:
        """
        Return the percentage of heartbeat checks in the last *hours* hours
        where *symbol* was healthy.
        """
        if not self.db_path.exists():
            return 100.0
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        try:
            conn = sqlite3.connect(str(self.db_path))
            total = conn.execute(
                "SELECT COUNT(*) FROM feed_health WHERE symbol = ? AND ts > ?",
                (symbol, cutoff),
            ).fetchone()[0]
            healthy = conn.execute(
                "SELECT COUNT(*) FROM feed_health "
                "WHERE symbol = ? AND ts > ? AND is_healthy = 1",
                (symbol, cutoff),
            ).fetchone()[0]
            conn.close()
            return (healthy / total * 100.0) if total > 0 else 100.0
        except sqlite3.Error as exc:
            logger.warning("Failed to compute uptime: %s", exc)
            return 100.0

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        sql_path = Path(__file__).parent / "schema_extension.sql"
        if not sql_path.exists() or not self.db_path.exists():
            return
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.executescript(sql_path.read_text(encoding="utf-8"))
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.warning("Could not apply feed_health schema: %s", exc)

    def _persist_health(self, health: FeedHealth) -> None:
        if not self.db_path.exists():
            return
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """
                INSERT INTO feed_health
                    (symbol, ts, is_healthy, latency_ms, price, issue_type)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    health.symbol,
                    ts,
                    1 if health.is_healthy else 0,
                    health.latency_ms,
                    health.price,
                    health.issue_type,
                ),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.warning("Failed to persist feed health record: %s", exc)
