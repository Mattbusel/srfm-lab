"""
BacktestQueue
=============
Priority queue for backtest jobs with persistent backing in idea_engine.db.

Priority levels (lower integer = higher urgency):
  URGENT  = 10  — hypothesis is currently degraded, need results fast
  HIGH    = 30  — promising hypothesis, run soon
  NORMAL  = 50  — standard research run
  LOW     = 70  — exploratory / low-confidence hypotheses

The worker loop pulls jobs from the DB in priority order and runs them
using BacktestRunner (respects n_workers for parallelism).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable

from .runner import BacktestRunner
from .result_parser import BacktestResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH_ENV     = "IDEA_ENGINE_DB"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "idea_engine.db"
POLL_INTERVAL   = 5.0   # seconds between queue polls in worker_loop


# ---------------------------------------------------------------------------
# Priority enum
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    URGENT = 10
    HIGH   = 30
    NORMAL = 50
    LOW    = 70


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestJob:
    """One enqueued backtest job."""

    job_id:       int
    priority:     int
    params:       dict[str, Any]
    label:        str | None
    hypothesis_id: int | None
    status:       str                # "queued" | "running" | "completed" | "failed" | "cancelled"
    created_at:   str
    started_at:   str | None = None
    completed_at: str | None = None

    @property
    def priority_name(self) -> str:
        try:
            return Priority(self.priority).name
        except ValueError:
            return str(self.priority)


# ---------------------------------------------------------------------------
# BacktestQueue
# ---------------------------------------------------------------------------

class BacktestQueue:
    """
    Priority-ordered backtest queue backed by the ``backtest_queue`` table.

    Parameters
    ----------
    db_path        : path to idea_engine.db.
    runner         : BacktestRunner instance.  If None, a default runner is
                     created.
    n_workers      : max parallel backtest workers in worker_loop.
    on_complete    : optional callback(job_id, BacktestResult) after each run.
    on_error       : optional callback(job_id, error_str) on job failure.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        runner: BacktestRunner | None = None,
        n_workers: int = 4,
        on_complete: Callable[[int, BacktestResult], None] | None = None,
        on_error: Callable[[int, str], None] | None = None,
    ) -> None:
        self.db_path = Path(
            db_path or os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)
        )
        self.runner = runner or BacktestRunner(db_path=self.db_path)
        self.n_workers = n_workers
        self.on_complete = on_complete
        self.on_error = on_error

        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._active_jobs: dict[int, threading.Thread] = {}   # job_id → thread
        self._lock = threading.Lock()
        self._avg_run_time: float = 30.0   # rolling estimate in seconds

        self._ensure_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        params: dict[str, Any],
        priority: Priority | int = Priority.NORMAL,
        label: str | None = None,
        hypothesis_id: int | None = None,
    ) -> int:
        """
        Add a new backtest job to the queue.

        Parameters
        ----------
        params       : complete parameter dict.
        priority     : Priority level (lower = sooner).
        label        : optional human label.
        hypothesis_id: optional DB hypothesis link.

        Returns
        -------
        int : job_id (row ID in backtest_queue table).
        """
        p = int(priority)
        params_json = json.dumps(params, sort_keys=True)

        conn = self._connect()
        cur = conn.execute(
            """
            INSERT INTO backtest_queue
                (priority, params_json, label, hypothesis_id, status)
            VALUES (?, ?, ?, ?, 'queued')
            """,
            (p, params_json, label, hypothesis_id),
        )
        job_id = cur.lastrowid
        conn.commit()
        conn.close()

        logger.info(
            "Enqueued job #%d priority=%s label=%r",
            job_id, Priority(p).name if p in [e.value for e in Priority] else p, label,
        )
        return job_id

    def next_job(self) -> BacktestJob | None:
        """
        Return (and lock) the next queued job by priority, or None if the
        queue is empty.

        The job status is atomically set to 'running' so concurrent callers
        do not pick the same job.
        """
        conn = self._connect()
        conn.execute("BEGIN EXCLUSIVE")
        try:
            row = conn.execute(
                """
                SELECT id, priority, params_json, label, hypothesis_id,
                       status, created_at, started_at, completed_at
                FROM backtest_queue
                WHERE status = 'queued'
                ORDER BY priority ASC, id ASC
                LIMIT 1
                """
            ).fetchone()

            if row is None:
                conn.rollback()
                return None

            job_id = row[0]
            started_at = _now_iso()
            conn.execute(
                "UPDATE backtest_queue SET status='running', started_at=? WHERE id=?",
                (started_at, job_id),
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
        finally:
            conn.close()

        params = json.loads(row[2])
        return BacktestJob(
            job_id=row[0],
            priority=row[1],
            params=params,
            label=row[3],
            hypothesis_id=row[4],
            status="running",
            created_at=row[6],
            started_at=started_at,
        )

    def cancel(self, job_id: int) -> bool:
        """
        Cancel a queued job by ID.

        Returns True if the job was found and cancelled, False otherwise.
        Only 'queued' jobs can be cancelled (running jobs cannot be interrupted
        via this method).
        """
        conn = self._connect()
        cur = conn.execute(
            "UPDATE backtest_queue SET status='cancelled' WHERE id=? AND status='queued'",
            (job_id,),
        )
        affected = cur.rowcount
        conn.commit()
        conn.close()

        if affected:
            logger.info("Cancelled job #%d.", job_id)
        else:
            logger.warning("Could not cancel job #%d (not found or already running).", job_id)
        return bool(affected)

    def cancel_all_queued(self) -> int:
        """Cancel all queued (not yet running) jobs.  Returns number cancelled."""
        conn = self._connect()
        cur = conn.execute(
            "UPDATE backtest_queue SET status='cancelled' WHERE status='queued'"
        )
        n = cur.rowcount
        conn.commit()
        conn.close()
        logger.info("Cancelled %d queued jobs.", n)
        return n

    def estimate_queue_time(self) -> float:
        """
        Estimate total wait time for the current queue in seconds.

        Calculated as: queued_jobs × avg_run_time / n_workers.
        """
        depth = self.queue_depth()
        return depth * self._avg_run_time / max(self.n_workers, 1)

    def queue_depth(self) -> int:
        """Return the number of jobs currently in 'queued' status."""
        conn = self._connect()
        n = conn.execute(
            "SELECT COUNT(*) FROM backtest_queue WHERE status='queued'"
        ).fetchone()[0]
        conn.close()
        return int(n)

    def running_count(self) -> int:
        """Return the number of jobs in 'running' status."""
        conn = self._connect()
        n = conn.execute(
            "SELECT COUNT(*) FROM backtest_queue WHERE status='running'"
        ).fetchone()[0]
        conn.close()
        return int(n)

    def get_job(self, job_id: int) -> BacktestJob | None:
        """Fetch a job by ID from the queue table."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM backtest_queue WHERE id=?", (job_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return BacktestJob(
            job_id=row["id"],
            priority=row["priority"],
            params=json.loads(row["params_json"]),
            label=row["label"],
            hypothesis_id=row["hypothesis_id"],
            status=row["status"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    def list_jobs(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[BacktestJob]:
        """List jobs, optionally filtered by status."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        if status:
            rows = conn.execute(
                "SELECT * FROM backtest_queue WHERE status=? "
                "ORDER BY priority ASC, id ASC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM backtest_queue "
                "ORDER BY priority ASC, id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [
            BacktestJob(
                job_id=r["id"],
                priority=r["priority"],
                params=json.loads(r["params_json"]),
                label=r["label"],
                hypothesis_id=r["hypothesis_id"],
                status=r["status"],
                created_at=r["created_at"],
                started_at=r["started_at"],
                completed_at=r["completed_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def worker_loop(
        self,
        n_workers: int | None = None,
        poll_interval: float = POLL_INTERVAL,
        max_iterations: int | None = None,
    ) -> None:
        """
        Blocking loop that continuously drains the queue.

        Jobs are dispatched to daemon threads up to *n_workers* concurrently.
        Call ``stop_workers()`` to signal graceful shutdown.

        Parameters
        ----------
        n_workers      : max concurrent running jobs (overrides instance default).
        poll_interval  : seconds to wait between queue polls when no job is found.
        max_iterations : if set, stop after this many poll iterations (for testing).
        """
        workers = n_workers or self.n_workers
        self._stop_event.clear()
        iteration = 0

        logger.info("BacktestQueue worker loop started (n_workers=%d).", workers)

        while not self._stop_event.is_set():
            # Clean up finished threads
            with self._lock:
                finished = [jid for jid, t in self._active_jobs.items()
                            if not t.is_alive()]
                for jid in finished:
                    del self._active_jobs[jid]
                active_count = len(self._active_jobs)

            # Dispatch new jobs up to the worker limit
            while active_count < workers:
                job = self.next_job()
                if job is None:
                    break
                t = threading.Thread(
                    target=self._execute_job,
                    args=(job,),
                    daemon=True,
                    name=f"bt-queue-{job.job_id}",
                )
                with self._lock:
                    self._active_jobs[job.job_id] = t
                t.start()
                active_count += 1
                logger.info(
                    "Dispatched job #%d (%s) | active=%d",
                    job.job_id, job.label or "unlabelled", active_count,
                )

            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break
            self._stop_event.wait(timeout=poll_interval)

        # Wait for active jobs to finish
        with self._lock:
            threads = list(self._active_jobs.values())
        for t in threads:
            t.join(timeout=self.runner.timeout_sec + 10)

        logger.info("BacktestQueue worker loop stopped.")

    def start_worker_thread(
        self,
        n_workers: int | None = None,
        poll_interval: float = POLL_INTERVAL,
    ) -> threading.Thread:
        """Start the worker loop in a daemon thread.  Returns the Thread."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self.worker_loop,
            kwargs={"n_workers": n_workers, "poll_interval": poll_interval},
            daemon=True,
            name="BacktestQueue-worker",
        )
        self._worker_thread.start()
        return self._worker_thread

    def stop_workers(self, timeout: float = 10.0) -> None:
        """Signal the worker loop to stop and wait for it."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _execute_job(self, job: BacktestJob) -> None:
        """Run *job* and update its DB record when done."""
        t0 = time.perf_counter()
        try:
            result = self.runner.run(
                params=job.params,
                label=job.label,
                hypothesis_id=job.hypothesis_id,
            )
            duration = time.perf_counter() - t0
            self._update_avg_run_time(duration)

            new_status = "completed" if result.is_success else "failed"
            self._finish_job(job.job_id, new_status)

            if self.on_complete:
                try:
                    self.on_complete(job.job_id, result)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("on_complete callback raised: %s", exc)

            logger.info(
                "Job #%d %s in %.1fs: sharpe=%.3f",
                job.job_id, new_status, duration,
                result.metrics.sharpe,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("Unhandled exception in job #%d: %s", job.job_id, exc, exc_info=True)
            self._finish_job(job.job_id, "failed")
            if self.on_error:
                try:
                    self.on_error(job.job_id, str(exc))
                except Exception:  # noqa: BLE001
                    pass

    def _finish_job(self, job_id: int, status: str) -> None:
        """Mark a job as done in the DB."""
        completed_at = _now_iso()
        try:
            conn = self._connect()
            conn.execute(
                "UPDATE backtest_queue SET status=?, completed_at=? WHERE id=?",
                (status, completed_at, job_id),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.warning("Failed to finish job #%d in DB: %s", job_id, exc)

    def _update_avg_run_time(self, duration: float) -> None:
        """Exponential moving average of run times (for estimate_queue_time)."""
        alpha = 0.2
        self._avg_run_time = alpha * duration + (1.0 - alpha) * self._avg_run_time

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        sql_path = Path(__file__).parent / "schema_extension.sql"
        if not sql_path.exists() or not self.db_path.exists():
            return
        try:
            conn = self._connect()
            conn.executescript(sql_path.read_text(encoding="utf-8"))
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.warning("Could not apply backtester schema: %s", exc)

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return a summary of queue statistics."""
        if not self.db_path.exists():
            return {}
        try:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS n
                FROM backtest_queue
                GROUP BY status
                """
            ).fetchall()
            avg_duration = conn.execute(
                """
                SELECT AVG(CAST(
                    (julianday(completed_at) - julianday(started_at)) * 86400
                AS REAL))
                FROM backtest_queue
                WHERE status='completed' AND completed_at IS NOT NULL
                """
            ).fetchone()[0]
            conn.close()

            by_status = {r["status"]: r["n"] for r in rows}
            return {
                "by_status": by_status,
                "avg_run_seconds": round(avg_duration, 1) if avg_duration else None,
                "estimated_queue_time_seconds": self.estimate_queue_time(),
                "active_workers": len(self._active_jobs),
            }
        except sqlite3.Error as exc:
            logger.warning("Failed to compute queue stats: %s", exc)
            return {}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
