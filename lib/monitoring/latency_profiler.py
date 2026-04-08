"""
Bar-Processing Latency Profiler (A4)
Instruments the full bar-processing pipeline with microsecond-resolution timing.

Identifies:
  - Critical path (longest sequential computation chain)
  - Candidates for async background computation
  - Bar-to-order latency target: < 100ms for all 19 instruments
"""
import time
import logging
import threading
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger(__name__)

@dataclass
class LatencyRecord:
    stage: str
    symbol: str
    duration_us: float    # microseconds
    bar_seq: int
    is_critical_path: bool = True

class LatencyProfiler:
    """
    Context-manager-based latency profiler for bar processing stages.

    Usage:
        profiler = LatencyProfiler(warn_threshold_ms=50)

        with profiler.stage("bh_physics", "BTC", bar_seq=100):
            # BH physics computation
            pass

        with profiler.stage("garch", "BTC", bar_seq=100):
            # GARCH update
            pass

        # After processing all bars:
        profiler.report()  # logs critical path analysis
    """

    # Stages known to be async-safe (can be moved to background workers)
    ASYNC_SAFE_STAGES = {"tda", "hmm_regime", "hurst", "gr_analog", "ads_cft"}

    def __init__(self, warn_threshold_ms: float = 50.0, report_every: int = 100):
        self._records: list[LatencyRecord] = []
        self._lock = threading.Lock()
        self._warn_threshold_us = warn_threshold_ms * 1000
        self._report_every = report_every
        self._bar_count = 0
        self._per_stage_stats: dict[str, list[float]] = defaultdict(list)
        self._bar_total_latencies: list[float] = []
        self._bar_start: dict[str, float] = {}  # symbol -> bar start time

    @contextmanager
    def stage(self, stage_name: str, symbol: str, bar_seq: int = 0):
        """Context manager for timing a processing stage."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            duration_us = (t1 - t0) * 1e6

            with self._lock:
                self._per_stage_stats[stage_name].append(duration_us)
                if len(self._per_stage_stats[stage_name]) > 10000:
                    self._per_stage_stats[stage_name] = self._per_stage_stats[stage_name][-5000:]

            if duration_us > self._warn_threshold_us:
                log.warning(
                    "LatencyProfiler: SLOW stage '%s' for %s: %.1f ms (threshold %.1f ms)",
                    stage_name, symbol, duration_us / 1000, self._warn_threshold_us / 1000
                )

    def bar_start(self, symbol: str):
        """Mark the start of bar processing for a symbol."""
        self._bar_start[symbol] = time.perf_counter()

    def bar_end(self, symbol: str, bar_seq: int = 0):
        """Mark the end of bar processing. Logs if total latency exceeds threshold."""
        t0 = self._bar_start.pop(symbol, None)
        if t0 is None:
            return
        total_us = (time.perf_counter() - t0) * 1e6

        with self._lock:
            self._bar_total_latencies.append(total_us)
            self._bar_count += 1

        if total_us > 100_000:  # 100ms threshold
            log.warning(
                "LatencyProfiler: bar-to-signal latency %.1f ms for %s (target <100ms)",
                total_us / 1000, symbol
            )

        if self._bar_count % self._report_every == 0:
            self.report()

    def report(self) -> dict:
        """Generate and log a latency report. Returns summary dict."""
        with self._lock:
            stats = {}
            for stage, latencies in self._per_stage_stats.items():
                if not latencies:
                    continue
                n = len(latencies)
                sorted_l = sorted(latencies)
                stats[stage] = {
                    "mean_ms": sum(latencies) / n / 1000,
                    "p50_ms": sorted_l[n // 2] / 1000,
                    "p95_ms": sorted_l[int(n * 0.95)] / 1000,
                    "p99_ms": sorted_l[int(n * 0.99)] / 1000,
                    "is_async_safe": stage in self.ASYNC_SAFE_STAGES,
                }

            bar_lats = self._bar_total_latencies
            if bar_lats:
                n = len(bar_lats)
                sorted_b = sorted(bar_lats)
                bar_stats = {
                    "mean_ms": sum(bar_lats) / n / 1000,
                    "p95_ms": sorted_b[int(n * 0.95)] / 1000,
                    "p99_ms": sorted_b[int(n * 0.99)] / 1000,
                }
            else:
                bar_stats = {}

        # Log critical path
        if stats:
            slowest = sorted(stats.items(), key=lambda x: -x[1]["p95_ms"])[:5]
            slow_strs = [f"{name}={s['p95_ms']:.1f}ms" for name, s in slowest]
            log.info("LatencyProfiler p95 critical path: %s", " -> ".join(slow_strs))

            async_candidates = [name for name, s in stats.items() if s["is_async_safe"] and s["p95_ms"] > 5.0]
            if async_candidates:
                log.info("LatencyProfiler: async-safe candidates (>5ms): %s", async_candidates)

        if bar_stats:
            log.info("LatencyProfiler: bar-total p95=%.1f ms, p99=%.1f ms",
                     bar_stats.get("p95_ms", 0), bar_stats.get("p99_ms", 0))

        return {"stages": stats, "bar_total": bar_stats}

    def get_stage_p95_ms(self, stage: str) -> float:
        """Get p95 latency for a specific stage in ms."""
        with self._lock:
            lats = self._per_stage_stats.get(stage, [])
        if not lats:
            return 0.0
        sorted_l = sorted(lats)
        return sorted_l[int(len(sorted_l) * 0.95)] / 1000
