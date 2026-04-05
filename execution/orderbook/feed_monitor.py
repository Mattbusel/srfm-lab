"""
execution/orderbook/feed_monitor.py
=====================================
Periodic metrics logger for L2 orderbook health.

Every 60 seconds (configurable) writes a JSON-Lines record to
``logs/orderbook_metrics.jsonl`` containing spread, depth, and imbalance for
each tracked symbol.

Alert logic
-----------
If the current spread for a symbol is more than 3× its rolling average over
the last ``ALERT_WINDOW`` observations, an alert is logged at WARNING level
and an "alert" flag is set in that observation's JSON record.

Usage
-----
Instantiate ``FeedMonitor`` and call ``await monitor.start()`` from your
async main loop.  ``BookManager`` must already be started.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import time
from typing import Optional

from .book_manager import BookManager

log = logging.getLogger("execution.feed_monitor")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_INTERVAL_SEC = 60          # write a record every N seconds
ALERT_WINDOW = 20              # rolling window for average spread
ALERT_MULTIPLIER = 3.0         # trigger alert when spread > N × average
DEFAULT_LOG_PATH = os.path.join("logs", "orderbook_metrics.jsonl")


# ---------------------------------------------------------------------------
# FeedMonitor
# ---------------------------------------------------------------------------

class FeedMonitor:
    """
    Logs spread / depth / imbalance metrics from a BookManager and alerts on
    anomalous spread widening.

    Parameters
    ----------
    book_manager : BookManager
        The live BookManager instance.
    symbols : list[str]
        Symbols to monitor.
    log_path : str
        Path to the JSONL output file.
    interval_sec : float
        How often to sample and write metrics (default 60 s).
    alert_window : int
        Number of past observations used to compute rolling average spread.
    alert_multiplier : float
        Alert threshold as a multiple of the rolling average.
    """

    def __init__(
        self,
        book_manager: BookManager,
        symbols: list[str],
        log_path: str = DEFAULT_LOG_PATH,
        interval_sec: float = LOG_INTERVAL_SEC,
        alert_window: int = ALERT_WINDOW,
        alert_multiplier: float = ALERT_MULTIPLIER,
    ) -> None:
        self._bm = book_manager
        self._symbols = list(symbols)
        self._log_path = log_path
        self._interval = interval_sec
        self._alert_window = alert_window
        self._alert_multiplier = alert_multiplier

        # Rolling deque of recent spread observations per symbol
        self._spread_history: dict[str, collections.deque] = {
            s: collections.deque(maxlen=alert_window) for s in symbols
        }

        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Ensure log directory exists
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the periodic monitor task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="feed_monitor")
        log.info(
            "FeedMonitor started: interval=%.0fs path=%s", self._interval, self._log_path
        )

    async def stop(self) -> None:
        """Stop the monitor."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("FeedMonitor stopped.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        while self._running:
            await asyncio.sleep(self._interval)
            try:
                self._sample_and_log()
            except Exception as exc:
                log.error("FeedMonitor sample error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_and_log(self) -> None:
        ts = time.time()
        records = []

        for symbol in self._symbols:
            record = self._build_record(symbol, ts)
            records.append(record)

            # Alert check
            spread = record.get("spread_bps")
            if spread is not None:
                hist = self._spread_history[symbol]
                if hist:
                    avg = sum(hist) / len(hist)
                    if spread > self._alert_multiplier * avg and avg > 0:
                        log.warning(
                            "FeedMonitor ALERT: %s spread=%.2fbps is >%.0fx avg=%.2fbps "
                            "(active_feed=%s)",
                            symbol, spread, self._alert_multiplier, avg,
                            self._bm.active_feed_name,
                        )
                        record["alert"] = True
                        record["avg_spread_bps"] = round(avg, 4)
                hist.append(spread)
            else:
                log.debug("FeedMonitor: no spread data for %s", symbol)

        # Write all records for this tick
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
        except OSError as exc:
            log.error("FeedMonitor: failed to write %s: %s", self._log_path, exc)

    def _build_record(self, symbol: str, ts: float) -> dict:
        book = self._bm.get_book(symbol)
        record: dict = {
            "ts": round(ts, 3),
            "symbol": symbol,
            "active_feed": self._bm.active_feed_name,
        }

        if book is None:
            record["error"] = "no_book"
            return record

        spread = book.spread_bps
        mid = book.mid_price
        imb = book.imbalance
        depth = book.depth(n=5)
        bid_usd = book.bid_depth_usd(n=5)
        ask_usd = book.ask_depth_usd(n=5)
        age = round(ts - book.last_update_ts, 3) if book.last_update_ts else None

        record.update({
            "spread_bps": round(spread, 4) if spread is not None else None,
            "mid_price":  round(mid, 6) if mid is not None else None,
            "imbalance":  round(imb, 4) if imb is not None else None,
            "bid_depth_usd": round(bid_usd, 2),
            "ask_depth_usd": round(ask_usd, 2),
            "best_bid":   book.best_bid,
            "best_ask":   book.best_ask,
            "book_age_sec": age,
            "bids_top5":  [[round(p, 6), round(q, 8)] for p, q in depth["bids"]],
            "asks_top5":  [[round(p, 6), round(q, 8)] for p, q in depth["asks"]],
            "alert": False,
        })
        return record
