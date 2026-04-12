"""
AETERNUS Real-Time Execution Layer (RTEL)
replay_engine.py — Historical data replay for strategy evaluation

Provides:
- ReplayEngine: event-driven replay of historical market data
- CSV/binary tick data loaders
- Synchronization and rate control
- Cross-asset time alignment
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .data_pipeline import RawTick

logger = logging.getLogger(__name__)

_EPS = 1e-12


@dataclass
class ReplayConfig:
    speed:           float = 0.0        # 0 = max speed, 1.0 = realtime
    start_time:      float = 0.0
    end_time:        float = float("inf")
    loop:            bool  = False
    n_warmup_ticks:  int   = 0


@dataclass
class ReplayStats:
    n_ticks:         int   = 0
    elapsed_real_s:  float = 0.0
    elapsed_sim_s:   float = 0.0
    ticks_per_sec:   float = 0.0
    n_assets:        int   = 0
    start_ts:        float = 0.0
    end_ts:          float = 0.0


TickHandler = Callable[[RawTick], None]


class ReplayEngine:
    """
    Replays synthetic or pre-generated tick data through registered handlers.
    Supports time synchronization and speed control.
    """

    def __init__(self, config: Optional[ReplayConfig] = None):
        self.config     = config or ReplayConfig()
        self._ticks:    List[RawTick] = []
        self._handlers: List[TickHandler] = []
        self._stats     = ReplayStats()
        self._is_running= False

    def load_ticks(self, ticks: List[RawTick]) -> None:
        """Load pre-generated ticks sorted by timestamp."""
        self._ticks = sorted(ticks, key=lambda t: t.timestamp)
        if self._ticks:
            self._stats.start_ts = self._ticks[0].timestamp
            self._stats.end_ts   = self._ticks[-1].timestamp
            self._stats.n_assets = len(set(t.asset_id for t in self._ticks))

    def add_handler(self, fn: TickHandler) -> None:
        self._handlers.append(fn)

    def run(self) -> ReplayStats:
        """Run the full replay. Returns stats."""
        if not self._ticks:
            return self._stats

        t0     = time.perf_counter()
        n      = 0
        prev_ts: Optional[float] = None

        for tick in self._ticks:
            if tick.timestamp < self.config.start_time:
                continue
            if tick.timestamp > self.config.end_time:
                break

            # Rate control
            if self.config.speed > _EPS and prev_ts is not None:
                dt_sim  = tick.timestamp - prev_ts
                dt_real = dt_sim / self.config.speed
                if dt_real > 0:
                    time.sleep(min(dt_real, 0.1))

            for handler in self._handlers:
                try:
                    handler(tick)
                except Exception as e:
                    logger.warning("Replay handler error: %s", e)

            prev_ts = tick.timestamp
            n      += 1

        elapsed = time.perf_counter() - t0
        self._stats.n_ticks      = n
        self._stats.elapsed_real_s = elapsed
        self._stats.ticks_per_sec  = n / max(elapsed, _EPS)
        if self._ticks:
            self._stats.elapsed_sim_s = self._stats.end_ts - self._stats.start_ts
        return self._stats

    def iter_ticks(self) -> Iterator[RawTick]:
        """Iterate over ticks one by one."""
        for tick in self._ticks:
            yield tick

    @staticmethod
    def from_synthetic(n_assets: int, n_steps: int, sigma: float = 0.01,
                       seed: int = 42) -> "ReplayEngine":
        """Create a replay engine with synthetic GBM data."""
        from .data_pipeline import SyntheticDataSource
        src    = SyntheticDataSource(n_assets=n_assets, sigma=sigma)
        ticks  = []
        for batch in src.generate(n_steps):
            ticks.extend(batch)
        engine = ReplayEngine()
        engine.load_ticks(ticks)
        return engine


class MultiAssetAligner:
    """
    Aligns multi-asset tick streams to a common time grid
    using forward-fill for missing observations.
    """

    def __init__(self, n_assets: int, max_gap_s: float = 5.0):
        self.n_assets  = n_assets
        self.max_gap_s = max_gap_s
        self._last:    Dict[int, RawTick] = {}
        self._buffers: Dict[int, deque]   = defaultdict(lambda: deque(maxlen=1000))

    def push(self, tick: RawTick) -> Optional[Dict[int, RawTick]]:
        """
        Push a tick. Returns aligned snapshot if all assets updated recently.
        """
        self._last[tick.asset_id] = tick
        self._buffers[tick.asset_id].append(tick)

        if len(self._last) < self.n_assets:
            return None

        now = tick.timestamp
        snapshot = {}
        for aid in range(self.n_assets):
            if aid in self._last:
                t = self._last[aid]
                if now - t.timestamp <= self.max_gap_s:
                    snapshot[aid] = t
                else:
                    return None  # stale asset
            else:
                return None

        return snapshot

    def aligned_prices(self) -> Optional[Dict[int, float]]:
        if len(self._last) < self.n_assets:
            return None
        return {aid: t.mid() for aid, t in self._last.items()
                if aid < self.n_assets}


class EventQueue:
    """Priority queue of timed events for simulation."""

    def __init__(self):
        import heapq
        self._heap:  List = []
        self._count: int  = 0

    def schedule(self, timestamp: float, callback: Callable, *args) -> None:
        import heapq
        heapq.heappush(self._heap, (timestamp, self._count, callback, args))
        self._count += 1

    def run_until(self, end_time: float) -> int:
        import heapq
        n = 0
        while self._heap and self._heap[0][0] <= end_time:
            ts, _, cb, args = heapq.heappop(self._heap)
            try:
                cb(*args)
            except Exception as e:
                logger.warning("EventQueue callback error: %s", e)
            n += 1
        return n

    def __len__(self) -> int:
        return len(self._heap)


# ---------------------------------------------------------------------------
# Mini integration test
# ---------------------------------------------------------------------------

def _self_test() -> bool:
    """Quick internal sanity check."""
    engine = ReplayEngine.from_synthetic(n_assets=3, n_steps=50)
    received = []
    engine.add_handler(lambda t: received.append(t))
    stats = engine.run()
    assert stats.n_ticks > 0
    assert len(received) == stats.n_ticks
    assert stats.n_assets == 3
    return True


if __name__ == "__main__":
    ok = _self_test()
    print("replay_engine self-test:", "PASS" if ok else "FAIL")
