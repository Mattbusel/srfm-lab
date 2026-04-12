"""
AETERNUS Real-Time Execution Layer (RTEL)
pipeline_client.py — High-Level Python Pipeline Client

Subscribes to market ticks, runs inference pipeline, publishes results.
Supports async event loop and latency tracking.

Usage:
    client = PipelineClient()
    client.add_handler("lumina", lumina_module.forward)
    await client.run_async()
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import numpy as np

from .shm_reader import ChannelCursor, LobSnapshot, ShmReader
from .shm_writer import ShmWriter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PipelineStage — tracks timing of one stage
# ---------------------------------------------------------------------------
@dataclass
class StageMetrics:
    name:      str
    count:     int   = 0
    total_ns:  int   = 0
    min_ns:    int   = 10**18
    max_ns:    int   = 0
    p99_buf:   deque = field(default_factory=lambda: deque(maxlen=10000))

    def record(self, elapsed_ns: int) -> None:
        self.count    += 1
        self.total_ns += elapsed_ns
        self.min_ns    = min(self.min_ns, elapsed_ns)
        self.max_ns    = max(self.max_ns, elapsed_ns)
        self.p99_buf.append(elapsed_ns)

    @property
    def mean_us(self) -> float:
        return (self.total_ns / self.count / 1000) if self.count else 0.0

    @property
    def p99_us(self) -> float:
        if not self.p99_buf:
            return 0.0
        s = sorted(self.p99_buf)
        return s[int(len(s) * 0.99)] / 1000


# ---------------------------------------------------------------------------
# PipelineRun — one complete pipeline execution
# ---------------------------------------------------------------------------
@dataclass
class PipelineRun:
    run_id:         int
    market_recv_ns: int = 0
    stages:         Dict[str, Tuple[int, int]] = field(default_factory=dict)
    total_ns:       int = 0
    sla_met:        bool = False
    error:          Optional[str] = None

    SLA_TARGET_NS = 1_000_000  # 1ms

    def stage_start(self, name: str) -> None:
        self.stages[name] = (time.perf_counter_ns(), 0)

    def stage_end(self, name: str) -> None:
        start, _ = self.stages.get(name, (time.perf_counter_ns(), 0))
        self.stages[name] = (start, time.perf_counter_ns())

    def stage_duration_ns(self, name: str) -> int:
        start, end = self.stages.get(name, (0, 0))
        return max(0, end - start)

    def finalize(self) -> None:
        end_ns = time.perf_counter_ns()
        self.total_ns = end_ns - self.market_recv_ns
        self.sla_met  = self.total_ns <= self.SLA_TARGET_NS


# ---------------------------------------------------------------------------
# Handler — a stage handler registered in the pipeline
# ---------------------------------------------------------------------------
@dataclass
class Handler:
    name:     str
    fn:       Callable
    is_async: bool
    timeout_ms: float = 500.0
    enabled:  bool = True


# ---------------------------------------------------------------------------
# PipelineClient
# ---------------------------------------------------------------------------
class PipelineClient:
    """
    High-level pipeline client for Python modules (Lumina, TensorNet, OmniGraph).

    Pipeline:
        LOB snapshot → [handlers in order] → publish results

    Latency tracking:
        Per-stage nanosecond timing via time.perf_counter_ns()
        Rolling p99 window (last 10,000 observations)
    """

    def __init__(self,
                 base_path:     Path = Path("/tmp"),
                 poll_interval: float = 0.001,
                 sla_target_ns: int = 1_000_000):
        self._reader       = ShmReader(base_path=base_path, auto_open=True)
        self._writer       = ShmWriter(base_path=base_path)
        self._handlers:    List[Handler] = []
        self._running      = False
        self._poll_interval = poll_interval
        self._sla_target_ns = sla_target_ns

        # Cursors for each input channel
        self._lob_cursor = ChannelCursor(
            channel_name=ShmReader.LOB_SNAPSHOT, next_seq=1)

        # Metrics
        self._runs:            int = 0
        self._sla_violations:  int = 0
        self._stage_metrics:   Dict[str, StageMetrics] = {}
        self._pipeline_latencies: deque = deque(maxlen=10000)
        self._last_run:        Optional[PipelineRun] = None

        logger.info("PipelineClient initialized at %s", base_path)

    def add_handler(self, name: str, fn: Callable,
                    timeout_ms: float = 500.0) -> "PipelineClient":
        """Register a pipeline handler. Handlers are called in registration order."""
        is_async = asyncio.iscoroutinefunction(fn)
        self._handlers.append(Handler(name, fn, is_async, timeout_ms))
        self._stage_metrics[name] = StageMetrics(name)
        logger.debug("Registered handler '%s' (async=%s)", name, is_async)
        return self

    def remove_handler(self, name: str) -> bool:
        before = len(self._handlers)
        self._handlers = [h for h in self._handlers if h.name != name]
        return len(self._handlers) < before

    def set_handler_enabled(self, name: str, enabled: bool) -> None:
        for h in self._handlers:
            if h.name == name:
                h.enabled = enabled

    # -----------------------------------------------------------------------
    # Synchronous event loop
    # -----------------------------------------------------------------------

    def run(self, max_iterations: Optional[int] = None) -> None:
        """Synchronous event loop. Blocks until stop() is called or max_iter reached."""
        self._running = True
        logger.info("PipelineClient starting synchronous loop")
        i = 0
        try:
            while self._running:
                snap = self._reader.read_lob(self._lob_cursor)
                if snap is not None:
                    self._execute_pipeline_sync(snap)
                else:
                    time.sleep(self._poll_interval)
                i += 1
                if max_iterations and i >= max_iterations:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
        logger.info("PipelineClient stopped after %d iterations", i)

    def _execute_pipeline_sync(self, snap: LobSnapshot) -> PipelineRun:
        run = PipelineRun(
            run_id=self._runs,
            market_recv_ns=time.perf_counter_ns(),
        )

        prev_result: Any = snap
        for handler in self._handlers:
            if not handler.enabled:
                continue
            run.stage_start(handler.name)
            t0 = time.perf_counter_ns()
            try:
                prev_result = handler.fn(prev_result)
            except Exception as e:
                logger.error("Handler '%s' error: %s", handler.name, e)
                run.error = str(e)
            t1 = time.perf_counter_ns()
            run.stage_end(handler.name)
            elapsed = t1 - t0
            self._stage_metrics[handler.name].record(elapsed)

        run.finalize()
        self._runs += 1
        self._pipeline_latencies.append(run.total_ns)
        if not run.sla_met:
            self._sla_violations += 1
        self._last_run = run
        return run

    # -----------------------------------------------------------------------
    # Async event loop
    # -----------------------------------------------------------------------

    async def run_async(self, max_iterations: Optional[int] = None) -> None:
        """Async event loop for use with asyncio.run()."""
        self._running = True
        logger.info("PipelineClient starting async loop")
        i = 0
        try:
            while self._running:
                snap = self._reader.read_lob(self._lob_cursor)
                if snap is not None:
                    await self._execute_pipeline_async(snap)
                else:
                    await asyncio.sleep(self._poll_interval)
                i += 1
                if max_iterations and i >= max_iterations:
                    break
        finally:
            self._running = False

    async def _execute_pipeline_async(self, snap: LobSnapshot) -> PipelineRun:
        run = PipelineRun(
            run_id=self._runs,
            market_recv_ns=time.perf_counter_ns(),
        )

        prev_result: Any = snap
        for handler in self._handlers:
            if not handler.enabled:
                continue
            run.stage_start(handler.name)
            t0 = time.perf_counter_ns()
            try:
                if handler.is_async:
                    coro = handler.fn(prev_result)
                    prev_result = await asyncio.wait_for(
                        coro, timeout=handler.timeout_ms / 1000)
                else:
                    loop = asyncio.get_event_loop()
                    prev_result = await loop.run_in_executor(
                        None, handler.fn, prev_result)
            except asyncio.TimeoutError:
                logger.warning("Handler '%s' timed out after %.0fms",
                               handler.name, handler.timeout_ms)
                run.error = f"{handler.name} timeout"
            except Exception as e:
                logger.error("Handler '%s' error: %s", handler.name, e)
                run.error = str(e)
            t1 = time.perf_counter_ns()
            run.stage_end(handler.name)
            self._stage_metrics[handler.name].record(t1 - t0)

        run.finalize()
        self._runs += 1
        self._pipeline_latencies.append(run.total_ns)
        if not run.sla_met:
            self._sla_violations += 1
        self._last_run = run
        return run

    def stop(self) -> None:
        self._running = False

    # -----------------------------------------------------------------------
    # Stats and reporting
    # -----------------------------------------------------------------------

    @property
    def pipeline_p99_us(self) -> float:
        if not self._pipeline_latencies:
            return 0.0
        s = sorted(self._pipeline_latencies)
        return s[int(len(s) * 0.99)] / 1000

    @property
    def pipeline_p50_us(self) -> float:
        if not self._pipeline_latencies:
            return 0.0
        s = sorted(self._pipeline_latencies)
        return s[len(s) // 2] / 1000

    @property
    def sla_violation_rate(self) -> float:
        if self._runs == 0:
            return 0.0
        return self._sla_violations / self._runs

    def summary(self) -> Dict[str, Any]:
        return {
            "runs":               self._runs,
            "sla_violations":     self._sla_violations,
            "sla_violation_rate": f"{self.sla_violation_rate:.2%}",
            "pipeline_p50_us":    round(self.pipeline_p50_us, 2),
            "pipeline_p99_us":    round(self.pipeline_p99_us, 2),
            "handlers":           [h.name for h in self._handlers],
            "stages":             {
                name: {
                    "count":    m.count,
                    "mean_us":  round(m.mean_us, 2),
                    "p99_us":   round(m.p99_us, 2),
                    "min_us":   round(m.min_ns / 1000, 2),
                    "max_us":   round(m.max_ns / 1000, 2),
                }
                for name, m in self._stage_metrics.items()
            },
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("=== Pipeline Client Summary ===")
        print(f"  Runs:            {s['runs']}")
        print(f"  SLA violations:  {s['sla_violations']} ({s['sla_violation_rate']})")
        print(f"  Pipeline p50:    {s['pipeline_p50_us']} µs")
        print(f"  Pipeline p99:    {s['pipeline_p99_us']} µs")
        print(f"  Handlers:        {s['handlers']}")
        if s["stages"]:
            print(f"  {'Stage':<20} {'Count':>8} {'Mean µs':>8} {'p99 µs':>8}")
            print(f"  {'-'*50}")
            for name, sm in s["stages"].items():
                print(f"  {name:<20} {sm['count']:>8} {sm['mean_us']:>8.1f} {sm['p99_us']:>8.1f}")

    def close(self) -> None:
        self._reader.close()
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
