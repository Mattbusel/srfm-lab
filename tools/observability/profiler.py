"""
tools/observability/profiler.py
================================
Continuous performance profiler for the LARSA live trader.

Uses cProfile to instrument hot-path functions and accumulates rolling
percentile statistics.  Every 5 minutes a JSON snapshot is written to
``profiles/hotspots.json`` relative to the repo root.

Also tracks:
  - Memory via ``tracemalloc`` (top-20 allocation sites)
  - Per-thread CPU % via ``psutil``

Usage::

    from tools.observability.profiler import ProfilerDaemon

    prof = ProfilerDaemon(output_dir="profiles")
    prof.start()

    # Instrument a function:
    @prof.profile("bar_handler")
    def my_bar_handler(bar):
        ...

    # Or measure manually:
    with prof.measure("compute_targets"):
        targets = compute_all_targets(bars)

    prof.stop()

Dependencies:
    pip install psutil   (optional but recommended)
"""

from __future__ import annotations

import cProfile
import io
import json
import logging
import os
import pstats
import threading
import time
import tracemalloc
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, TypeVar

log = logging.getLogger(__name__)

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    log.warning("psutil not installed — CPU-per-thread metrics unavailable")

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Rolling percentile helper
# ---------------------------------------------------------------------------

_WINDOW = 1000   # keep last N samples per function


def _percentile(sorted_data: list, pct: float) -> float:
    if not sorted_data:
        return 0.0
    idx = max(0, int(len(sorted_data) * pct / 100) - 1)
    return sorted_data[min(idx, len(sorted_data) - 1)]


# ---------------------------------------------------------------------------
# ProfilerDaemon
# ---------------------------------------------------------------------------

class ProfilerDaemon:
    """
    Continuous cProfile-based profiler with rolling statistics.

    Each instrumented function accumulates a deque of elapsed-ms samples.
    A background thread periodically flushes statistics + memory snapshots
    to ``{output_dir}/hotspots.json``.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        flush_interval_secs: int = 300,
        window: int = _WINDOW,
        enable_tracemalloc: bool = True,
        top_allocations: int = 20,
    ) -> None:
        if output_dir is None:
            _repo = Path(__file__).parents[2]
            output_dir = str(_repo / "profiles")
        self._output_dir = Path(output_dir)
        self._flush_interval = flush_interval_secs
        self._window = window
        self._enable_tracemalloc = enable_tracemalloc
        self._top_allocs = top_allocations

        # samples[func_name] → deque of float (ms)
        self._samples: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

        self._running = False
        self._flush_thread: Optional[threading.Thread] = None
        self._start_time = time.time()

    # ----------------------------------------------------------------- start/stop
    def start(self) -> None:
        """Start the background flush thread and tracemalloc (if enabled)."""
        if self._running:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)

        if self._enable_tracemalloc:
            try:
                tracemalloc.start(10)
                log.info("ProfilerDaemon: tracemalloc enabled")
            except Exception as exc:
                log.warning("tracemalloc.start() failed: %s", exc)

        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="profiler-flush",
            daemon=True,
        )
        self._flush_thread.start()
        log.info("ProfilerDaemon started (output=%s, flush=%ds)",
                 self._output_dir, self._flush_interval)

    def stop(self) -> None:
        """Stop the flush thread and write a final snapshot."""
        if not self._running:
            return
        self._running = False
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=10.0)
        self._flush_snapshot()
        if self._enable_tracemalloc:
            try:
                tracemalloc.stop()
            except Exception:
                pass
        log.info("ProfilerDaemon stopped")

    # ----------------------------------------------------------------- sample recording
    def _record(self, func_name: str, elapsed_ms: float) -> None:
        with self._lock:
            if func_name not in self._samples:
                self._samples[func_name] = deque(maxlen=self._window)
            self._samples[func_name].append(elapsed_ms)

    # ----------------------------------------------------------------- context manager
    @contextmanager
    def measure(self, func_name: str) -> Generator[None, None, None]:
        """
        Context manager that measures wall-clock time of the enclosed block
        and records it under ``func_name``.

        Example::

            with profiler.measure("compute_targets"):
                targets = compute_all_targets(bars)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._record(func_name, elapsed_ms)

    # ----------------------------------------------------------------- decorator
    def profile(self, func_name: Optional[str] = None) -> Callable[[F], F]:
        """
        Decorator that wraps a function with timing measurement.

        Parameters
        ----------
        func_name:
            Label for this function in the stats output.
            Defaults to the decorated function's qualified name.

        Example::

            @profiler.profile("bar_handler")
            def handle_bar(bar: dict) -> None:
                ...
        """
        import functools

        def decorator(fn: F) -> F:
            name = func_name or fn.__qualname__

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    self._record(name, (time.perf_counter() - t0) * 1000.0)

            return wrapper  # type: ignore

        return decorator

    # ----------------------------------------------------------------- cProfile helper
    def profile_call(self, func_name: str, fn: Callable, *args, **kwargs) -> Any:
        """
        Run ``fn(*args, **kwargs)`` under cProfile, record wall time, and
        store the top cProfile stats in the snapshot.

        Returns the function's return value.
        """
        profiler = cProfile.Profile()
        t0 = time.perf_counter()
        profiler.enable()
        try:
            result = fn(*args, **kwargs)
        finally:
            profiler.disable()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._record(func_name, elapsed_ms)
            self._store_cprofile_stats(func_name, profiler)
        return result

    def _store_cprofile_stats(self, func_name: str, profiler: cProfile.Profile) -> None:
        """Extract and store top-10 cProfile lines for a function."""
        buf = io.StringIO()
        try:
            stats = pstats.Stats(profiler, stream=buf)
            stats.sort_stats(pstats.SortKey.CUMULATIVE)
            stats.print_stats(10)
            with self._lock:
                if not hasattr(self, "_cprofile_snapshots"):
                    self._cprofile_snapshots: Dict[str, str] = {}
                self._cprofile_snapshots[func_name] = buf.getvalue()
        except Exception as exc:
            log.debug("cProfile stats error for %s: %s", func_name, exc)

    # ----------------------------------------------------------------- statistics
    def get_stats(self) -> Dict[str, Any]:
        """
        Return a dict of per-function rolling statistics.

        Returns
        -------
        dict
            Keys are function names; values are dicts with
            ``count``, ``avg_ms``, ``p50_ms``, ``p95_ms``, ``p99_ms``,
            ``min_ms``, ``max_ms``.
        """
        with self._lock:
            snapshot = {k: list(v) for k, v in self._samples.items()}

        result: Dict[str, Any] = {}
        for name, samples in snapshot.items():
            if not samples:
                continue
            s = sorted(samples)
            result[name] = {
                "count": len(s),
                "avg_ms": round(sum(s) / len(s), 3),
                "min_ms": round(s[0], 3),
                "p50_ms": round(_percentile(s, 50), 3),
                "p95_ms": round(_percentile(s, 95), 3),
                "p99_ms": round(_percentile(s, 99), 3),
                "max_ms": round(s[-1], 3),
            }
        return result

    # ----------------------------------------------------------------- memory snapshot
    def get_memory_snapshot(self) -> List[Dict[str, Any]]:
        """
        Return top-N tracemalloc allocation sites.

        Each entry has ``filename``, ``lineno``, ``size_kb``, ``count``.
        Returns an empty list if tracemalloc is not running.
        """
        if not tracemalloc.is_tracing():
            return []
        try:
            snapshot = tracemalloc.take_snapshot()
            stats = snapshot.statistics("lineno")
            result = []
            for stat in stats[: self._top_allocs]:
                frame = stat.traceback[0]
                result.append({
                    "filename": frame.filename,
                    "lineno": frame.lineno,
                    "size_kb": round(stat.size / 1024, 2),
                    "count": stat.count,
                })
            return result
        except Exception as exc:
            log.debug("tracemalloc snapshot error: %s", exc)
            return []

    # ----------------------------------------------------------------- CPU per thread
    def get_thread_cpu(self) -> List[Dict[str, Any]]:
        """
        Return per-thread CPU time via psutil.

        Returns an empty list if psutil is unavailable.
        """
        if not _PSUTIL_AVAILABLE:
            return []
        try:
            proc = psutil.Process()
            threads = proc.threads()
            all_threads = {t.ident: t.name for t in threading.enumerate()}
            result = []
            for t in threads:
                name = all_threads.get(t.id, f"thread-{t.id}")
                result.append({
                    "thread_id": t.id,
                    "name": name,
                    "user_time_secs": round(t.user_time, 3),
                    "system_time_secs": round(t.system_time, 3),
                })
            return result
        except Exception as exc:
            log.debug("psutil thread CPU error: %s", exc)
            return []

    # ----------------------------------------------------------------- flush
    def _flush_snapshot(self) -> None:
        """Write hotspots.json to disk."""
        try:
            output = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "uptime_secs": round(time.time() - self._start_time, 1),
                "function_stats": self.get_stats(),
                "memory_top_allocations": self.get_memory_snapshot(),
                "thread_cpu": self.get_thread_cpu(),
            }
            if hasattr(self, "_cprofile_snapshots"):
                with self._lock:
                    output["cprofile_text"] = dict(self._cprofile_snapshots)

            path = self._output_dir / "hotspots.json"
            tmp = self._output_dir / "hotspots.json.tmp"
            tmp.write_text(json.dumps(output, indent=2))
            tmp.replace(path)
            log.debug("ProfilerDaemon: flushed %s", path)
        except Exception as exc:
            log.warning("ProfilerDaemon flush error: %s", exc)

    def _flush_loop(self) -> None:
        last_flush = time.time()
        while self._running:
            time.sleep(1.0)
            if time.time() - last_flush >= self._flush_interval:
                self._flush_snapshot()
                last_flush = time.time()

    # ----------------------------------------------------------------- context manager
    def __enter__(self) -> "ProfilerDaemon":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        funcs = len(self._samples)
        return (
            f"<ProfilerDaemon output={self._output_dir} "
            f"running={self._running} tracked_functions={funcs}>"
        )
