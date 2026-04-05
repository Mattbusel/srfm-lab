"""
sentiment_engine/scheduler.py
================================
Async scheduler that runs the full sentiment scrape-score-aggregate-bridge
pipeline every 15 minutes using asyncio + aiohttp.

Design notes
------------
The core scraping logic (Reddit PRAW, Twitter requests, RSS parsing) is
synchronous.  We run it in a ThreadPoolExecutor so it doesn't block the
event loop.  The result is then handed to the async-safe pipeline steps.

Schedule: run every CYCLE_INTERVAL_S seconds (default 900 = 15 minutes).
On startup, run immediately, then on schedule.

Signal handling: SIGTERM / SIGINT cause clean shutdown after the current
cycle completes.

Usage::

    import asyncio
    from sentiment_engine.scheduler import SentimentScheduler

    scheduler = SentimentScheduler(db_path="idea_engine.db")
    asyncio.run(scheduler.run_forever())

Or from the CLI::

    python -m sentiment_engine.scheduler
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .aggregator    import SentimentAggregator, SentimentSignal, DEFAULT_DB
from .signal_bridge import SignalBridge

logger = logging.getLogger(__name__)

CYCLE_INTERVAL_S: int = 900   # 15 minutes
MAX_CONSECUTIVE_ERRORS: int = 5


# ---------------------------------------------------------------------------
# Cycle statistics
# ---------------------------------------------------------------------------

class CycleStats:
    """Tracks runtime statistics across scheduler cycles."""

    def __init__(self) -> None:
        self.cycles_run:        int   = 0
        self.cycles_failed:     int   = 0
        self.signals_generated: int   = 0
        self.hypotheses_created: int  = 0
        self.last_cycle_ts:     Optional[float] = None
        self.last_cycle_duration_s: Optional[float] = None
        self.start_ts:          float = time.monotonic()

    def record_cycle(
        self,
        duration_s:    float,
        n_signals:     int,
        n_hypotheses:  int,
        failed:        bool,
    ) -> None:
        self.cycles_run             += 1
        self.last_cycle_ts           = time.monotonic()
        self.last_cycle_duration_s   = duration_s
        self.signals_generated      += n_signals
        self.hypotheses_created     += n_hypotheses
        if failed:
            self.cycles_failed += 1

    def to_dict(self) -> dict:
        uptime = time.monotonic() - self.start_ts
        return {
            "cycles_run":              self.cycles_run,
            "cycles_failed":           self.cycles_failed,
            "signals_generated":       self.signals_generated,
            "hypotheses_created":      self.hypotheses_created,
            "uptime_hours":            round(uptime / 3600, 2),
            "last_cycle_duration_s":   self.last_cycle_duration_s,
        }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class SentimentScheduler:
    """
    Async scheduler for the sentiment engine pipeline.

    Parameters
    ----------
    db_path          : Path to idea_engine.db
    cycle_interval_s : Seconds between cycles (default 900)
    twitter_mock     : Use mock Twitter scraper
    reddit_mock      : Use mock Reddit scraper
    max_workers      : ThreadPoolExecutor workers for blocking I/O
    """

    def __init__(
        self,
        db_path:          Path | str = DEFAULT_DB,
        cycle_interval_s: int  = CYCLE_INTERVAL_S,
        twitter_mock:     bool = True,
        reddit_mock:      bool = True,
        max_workers:      int  = 4,
    ) -> None:
        self.db_path          = Path(db_path)
        self.cycle_interval_s = cycle_interval_s
        self._executor        = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="senteng")
        self._aggregator      = SentimentAggregator(
            db_path=db_path,
            twitter_mock=twitter_mock,
            reddit_mock=reddit_mock,
        )
        self._bridge          = SignalBridge(db_path=db_path)
        self._stats           = CycleStats()
        self._shutdown        = asyncio.Event()
        self._consecutive_errors = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def run_forever(self) -> None:
        """
        Run the scheduler indefinitely.

        Listens for SIGTERM/SIGINT to trigger graceful shutdown.
        After MAX_CONSECUTIVE_ERRORS consecutive failures, aborts.
        """
        self._register_signal_handlers()
        logger.info(
            "SentimentScheduler starting — cycle every %ds, db=%s",
            self.cycle_interval_s, self.db_path,
        )

        try:
            # Run first cycle immediately
            await self._run_cycle()

            while not self._shutdown.is_set():
                # Wait for next cycle or shutdown signal
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(),
                        timeout=float(self.cycle_interval_s),
                    )
                except asyncio.TimeoutError:
                    pass   # expected — time for next cycle

                if self._shutdown.is_set():
                    break

                await self._run_cycle()

                if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.critical(
                        "SentimentScheduler: %d consecutive errors — aborting.",
                        self._consecutive_errors,
                    )
                    break

        finally:
            self._executor.shutdown(wait=False)
            logger.info("SentimentScheduler stopped. Stats: %s", self._stats.to_dict())

    async def run_once(self) -> tuple[list[SentimentSignal], list]:
        """
        Run a single cycle synchronously (for testing / CLI use).

        Returns (signals, hypotheses).
        """
        return await self._run_cycle()

    def get_stats(self) -> dict:
        """Return current scheduler statistics."""
        return self._stats.to_dict()

    def request_shutdown(self) -> None:
        """Signal the scheduler to stop after the current cycle."""
        logger.info("SentimentScheduler: shutdown requested.")
        self._shutdown.set()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    async def _run_cycle(self) -> tuple[list[SentimentSignal], list]:
        """Execute one full pipeline cycle in a thread pool."""
        ts_start = time.monotonic()
        cycle_num = self._stats.cycles_run + 1
        logger.info("SentimentScheduler: starting cycle #%d …", cycle_num)

        signals: list[SentimentSignal] = []
        hypotheses: list = []
        failed = False

        try:
            loop = asyncio.get_event_loop()

            # Run blocking aggregation in thread pool
            signals = await loop.run_in_executor(
                self._executor,
                self._aggregator.run_cycle,
            )

            # Run blocking signal bridge in thread pool
            hypotheses = await loop.run_in_executor(
                self._executor,
                self._bridge.convert,
                signals,
            )

            self._consecutive_errors = 0
            logger.info(
                "Cycle #%d: %d signals → %d hypotheses.", cycle_num, len(signals), len(hypotheses)
            )

        except Exception as exc:
            failed = True
            self._consecutive_errors += 1
            logger.error(
                "Cycle #%d failed (%d/%d consecutive): %s",
                cycle_num, self._consecutive_errors, MAX_CONSECUTIVE_ERRORS, exc,
                exc_info=True,
            )

        duration = time.monotonic() - ts_start
        self._stats.record_cycle(
            duration_s=duration,
            n_signals=len(signals),
            n_hypotheses=len(hypotheses),
            failed=failed,
        )
        logger.info("Cycle #%d complete in %.1fs.", cycle_num, duration)
        return signals, hypotheses

    def _register_signal_handlers(self) -> None:
        """Register OS signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        def _handle_stop(sig_name: str) -> None:
            logger.info("Received %s — initiating graceful shutdown.", sig_name)
            loop.call_soon_threadsafe(self._shutdown.set)

        if sys.platform != "win32":
            loop.add_signal_handler(signal.SIGTERM, lambda: _handle_stop("SIGTERM"))
            loop.add_signal_handler(signal.SIGINT,  lambda: _handle_stop("SIGINT"))
        else:
            # On Windows, signal.signal must be used from the main thread
            signal.signal(signal.SIGINT,  lambda s, f: _handle_stop("SIGINT"))
            signal.signal(signal.SIGTERM, lambda s, f: _handle_stop("SIGTERM"))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the IAE Sentiment Engine scheduler")
    parser.add_argument("--db",       default=str(DEFAULT_DB), help="Path to idea_engine.db")
    parser.add_argument("--interval", type=int, default=CYCLE_INTERVAL_S, help="Cycle interval in seconds")
    parser.add_argument("--once",     action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--no-mock",  action="store_true", dest="no_mock",
                        help="Use live API scrapers (requires credentials in env)")
    args = parser.parse_args()

    use_mock = not args.no_mock
    scheduler = SentimentScheduler(
        db_path=args.db,
        cycle_interval_s=args.interval,
        twitter_mock=use_mock,
        reddit_mock=use_mock,
    )

    if args.once:
        signals, hyps = asyncio.run(scheduler.run_once())
        print(f"Signals: {len(signals)}, Hypotheses: {len(hyps)}")
        for sig in signals:
            print(f"  {sig.symbol:6s}  score={sig.score:+.3f}  conf={sig.confidence:.2f}  F&G={sig.fear_greed_index}")
    else:
        asyncio.run(scheduler.run_forever())


if __name__ == "__main__":
    _main()
