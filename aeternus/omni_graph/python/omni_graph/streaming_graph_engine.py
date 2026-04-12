"""
streaming_graph_engine.py — Streaming Graph Construction Engine
===============================================================
Part of the Omni-Graph incremental graph construction suite.

Design goals
------------
* Dedicated background thread/process for graph construction, fully
  decoupled from the inference thread.
* Producer-consumer queue: the construction thread publishes completed
  graph snapshots; the inference thread consumes them with zero-copy
  where possible.
* Double buffering: while inference uses buffer A, construction writes
  to buffer B, then swaps atomically.
* Adaptive update frequency: slow graph updates during low-volatility
  periods; fast during high-vol (prevents CPU/GPU waste in quiet markets).
* Priority queue of pending edge updates: most impactful edges are
  processed first to reduce worst-case latency for critical edges.
* Fallback mechanism: if the graph becomes disconnected (Fiedler value → 0),
  switch to a minimum spanning tree to preserve connectivity.
* Density guard: if edge density exceeds threshold, prune to k-nearest
  by weight to keep the graph sparse.
* Async integration with RTEL shm-bus: reads LOB snapshots from shared
  memory, publishes graph snapshots to GSR.

Architecture
------------
    LOBFeed (mock/real) ──► ConstructionThread ──► DoubleBuffer ──► InferenceThread
                                    │                    ▲
                          PriorityEdgeQueue         atomic swap
                          FiedlerMonitor
                          DensityGuard
                          AdaptiveUpdateScheduler
"""

from __future__ import annotations

import heapq
import math
import queue
import threading
import time
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict, List, Any, Callable

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EngineState(Enum):
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()


class GraphRegime(Enum):
    LOW_VOL = auto()
    NORMAL = auto()
    HIGH_VOL = auto()
    CRISIS = auto()


class FallbackMode(Enum):
    FULL_GRAPH = auto()
    MINIMUM_SPANNING_TREE = auto()
    K_NEAREST = auto()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SGEConfig:
    """Configuration for the Streaming Graph Engine."""

    n_assets: int = 500
    feature_dim: int = 64
    device: str = "cuda"

    # Queue capacities
    input_queue_size: int = 512
    output_queue_size: int = 16

    # Adaptive update frequencies (ticks between graph rebuilds)
    update_freq_low_vol: int = 10
    update_freq_normal: int = 5
    update_freq_high_vol: int = 1
    update_freq_crisis: int = 1

    # Volatility thresholds for regime detection
    vol_high_threshold: float = 0.02
    vol_low_threshold: float = 0.005
    vol_crisis_threshold: float = 0.05

    # Connectivity / fallback
    fiedler_threshold: float = 1e-4
    mst_fallback: bool = True

    # Density guard: if density > this, prune to k-nearest
    max_density: float = 0.15
    density_knn: int = 20

    # Double-buffer: wait at most this many ms for swap
    buffer_swap_timeout_ms: float = 5.0

    # Priority queue: recompute top-K edges per update cycle
    priority_queue_top_k: int = 500

    # Thread affinity (Linux only, ignored on Windows)
    construction_cpu_affinity: Optional[int] = None

    # Publish to GSR (set False if not using RTEL)
    publish_to_gsr: bool = False
    gsr_topic: str = "omni_graph.snapshot"

    # Background thread priority
    thread_daemon: bool = True


# ---------------------------------------------------------------------------
# Graph snapshot (immutable, passed between threads)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphSnapshot:
    """Immutable graph snapshot published by the construction thread."""

    tick_id: int
    timestamp_ns: int
    edge_index: torch.Tensor      # (2, E) int64 — CPU tensor for safe IPC
    edge_weight: torch.Tensor     # (E,) float32 — CPU tensor
    n_nodes: int
    n_edges: int
    density: float
    fiedler_value: float
    regime: GraphRegime
    fallback_mode: FallbackMode
    construction_time_ms: float

    def to_device(self, device: torch.device) -> "GraphSnapshot":
        """Move edge tensors to device (returns new object, data is shared)."""
        object.__setattr__(self, "edge_index", self.edge_index.to(device))
        object.__setattr__(self, "edge_weight", self.edge_weight.to(device))
        return self

    def __repr__(self) -> str:
        return (
            f"GraphSnapshot(tick={self.tick_id}, edges={self.n_edges}, "
            f"density={self.density:.4f}, fiedler={self.fiedler_value:.4f}, "
            f"regime={self.regime.name}, "
            f"fallback={self.fallback_mode.name}, "
            f"{self.construction_time_ms:.2f}ms)"
        )


# ---------------------------------------------------------------------------
# Priority edge queue
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _PriorityEdge:
    """Entry in the priority edge queue."""
    priority: float          # negative weight so max-heap via min-heap
    sort_index: int = field(compare=False)
    src: int = field(compare=False)
    dst: int = field(compare=False)
    weight: float = field(compare=False)
    tick_id: int = field(compare=False)


class PriorityEdgeQueue:
    """
    Max-priority queue of pending edge updates.

    Edges are prioritised by |weight_change| so the most impactful
    updates are processed first within each construction cycle.
    """

    def __init__(self, maxsize: int = 100_000) -> None:
        self._heap: List[_PriorityEdge] = []
        self._counter: int = 0
        self._maxsize = maxsize
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def push(self, src: int, dst: int, weight: float, delta: float, tick_id: int) -> None:
        """Push an edge update with given weight and delta magnitude."""
        with self._lock:
            if len(self._heap) >= self._maxsize:
                # Evict lowest priority if full
                heapq.heapreplace(
                    self._heap,
                    _PriorityEdge(
                        priority=-abs(delta),
                        sort_index=self._counter,
                        src=src, dst=dst,
                        weight=weight,
                        tick_id=tick_id,
                    ),
                )
            else:
                heapq.heappush(
                    self._heap,
                    _PriorityEdge(
                        priority=-abs(delta),
                        sort_index=self._counter,
                        src=src, dst=dst,
                        weight=weight,
                        tick_id=tick_id,
                    ),
                )
            self._counter += 1

    # ------------------------------------------------------------------
    def pop_top_k(self, k: int) -> List[_PriorityEdge]:
        """Pop the top-K highest priority edges."""
        with self._lock:
            result: List[_PriorityEdge] = []
            for _ in range(min(k, len(self._heap))):
                result.append(heapq.heappop(self._heap))
            return result

    # ------------------------------------------------------------------
    def size(self) -> int:
        with self._lock:
            return len(self._heap)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        with self._lock:
            self._heap.clear()


# ---------------------------------------------------------------------------
# Double buffer
# ---------------------------------------------------------------------------

class DoubleBuffer:
    """
    Thread-safe double buffer for graph snapshots.

    The construction thread writes to the inactive buffer, then atomically
    promotes it to active.  The inference thread always reads from the
    active buffer without blocking.
    """

    def __init__(self) -> None:
        self._buffers: List[Optional[GraphSnapshot]] = [None, None]
        self._active: int = 0          # index of buffer currently read by inference
        self._lock = threading.Lock()
        self._write_event = threading.Event()

    # ------------------------------------------------------------------
    def write(self, snapshot: GraphSnapshot) -> None:
        """Write a new snapshot and swap the active buffer."""
        inactive = 1 - self._active
        self._buffers[inactive] = snapshot
        # Atomic swap
        with self._lock:
            self._active = inactive
        self._write_event.set()

    # ------------------------------------------------------------------
    def read(self) -> Optional[GraphSnapshot]:
        """Read the current active snapshot (non-blocking)."""
        return self._buffers[self._active]

    # ------------------------------------------------------------------
    def wait_for_update(self, timeout_ms: float = 100.0) -> Optional[GraphSnapshot]:
        """Block until a new snapshot is available or timeout."""
        self._write_event.wait(timeout=timeout_ms / 1000.0)
        self._write_event.clear()
        return self.read()

    # ------------------------------------------------------------------
    def has_snapshot(self) -> bool:
        return self._buffers[self._active] is not None


# ---------------------------------------------------------------------------
# Regime detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Detects the current market volatility regime from a rolling returns
    series and outputs a GraphRegime enum.
    """

    def __init__(
        self,
        vol_window: int = 20,
        low_threshold: float = 0.005,
        high_threshold: float = 0.02,
        crisis_threshold: float = 0.05,
    ) -> None:
        self.vol_window = vol_window
        self.low_thresh = low_threshold
        self.high_thresh = high_threshold
        self.crisis_thresh = crisis_threshold

        self._returns_history: List[float] = []
        self._current_regime = GraphRegime.NORMAL

    # ------------------------------------------------------------------
    def update(self, market_return: float) -> GraphRegime:
        """
        Update with the latest cross-sectional mean return and detect regime.
        """
        self._returns_history.append(abs(market_return))
        if len(self._returns_history) > self.vol_window:
            self._returns_history.pop(0)

        if len(self._returns_history) < 3:
            return self._current_regime

        vol = float(np.std(self._returns_history))

        if vol >= self.crisis_thresh:
            self._current_regime = GraphRegime.CRISIS
        elif vol >= self.high_thresh:
            self._current_regime = GraphRegime.HIGH_VOL
        elif vol <= self.low_thresh:
            self._current_regime = GraphRegime.LOW_VOL
        else:
            self._current_regime = GraphRegime.NORMAL

        return self._current_regime

    # ------------------------------------------------------------------
    @property
    def current_regime(self) -> GraphRegime:
        return self._current_regime

    # ------------------------------------------------------------------
    def current_vol(self) -> float:
        if not self._returns_history:
            return 0.0
        return float(np.std(self._returns_history))


# ---------------------------------------------------------------------------
# Adaptive update scheduler
# ---------------------------------------------------------------------------

class AdaptiveUpdateScheduler:
    """
    Controls how often the construction thread runs based on the current
    market regime.
    """

    def __init__(
        self,
        freq_low_vol: int = 10,
        freq_normal: int = 5,
        freq_high_vol: int = 1,
        freq_crisis: int = 1,
    ) -> None:
        self._freq: Dict[GraphRegime, int] = {
            GraphRegime.LOW_VOL: freq_low_vol,
            GraphRegime.NORMAL: freq_normal,
            GraphRegime.HIGH_VOL: freq_high_vol,
            GraphRegime.CRISIS: freq_crisis,
        }
        self._tick_counter: int = 0
        self._current_regime = GraphRegime.NORMAL

    # ------------------------------------------------------------------
    def should_update(self, regime: GraphRegime) -> bool:
        """Return True if an update should be triggered this tick."""
        self._current_regime = regime
        self._tick_counter += 1
        freq = self._freq.get(regime, 5)
        if self._tick_counter >= freq:
            self._tick_counter = 0
            return True
        return False

    # ------------------------------------------------------------------
    def set_frequency(self, regime: GraphRegime, freq: int) -> None:
        self._freq[regime] = freq


# ---------------------------------------------------------------------------
# Fiedler value estimator
# ---------------------------------------------------------------------------

class FiedlerMonitor:
    """
    Estimates the Fiedler value (second-smallest eigenvalue of the Laplacian)
    using a fast power iteration approximation.

    Detects near-zero Fiedler values as graph disconnection signals.
    """

    def __init__(
        self,
        n_nodes: int,
        threshold: float = 1e-4,
        n_iter: int = 10,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.n = n_nodes
        self.threshold = threshold
        self.n_iter = n_iter
        self.device = device

        # Warm-start vector
        self._v: Optional[torch.Tensor] = None
        self._last_fiedler: float = 1.0

    # ------------------------------------------------------------------
    def estimate(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        weight: torch.Tensor,
    ) -> float:
        """
        Estimate Fiedler value from COO edges using power iteration.

        Parameters
        ----------
        row, col : (E,) int64
        weight   : (E,) float32

        Returns
        -------
        fiedler : float
        """
        n = self.n
        dev = self.device

        # Build degree vector
        deg = torch.zeros(n, device=dev)
        deg.scatter_add_(0, row.to(dev), weight.to(dev))

        # Warm-start or random init
        if self._v is None or self._v.shape[0] != n:
            v = torch.randn(n, device=dev)
        else:
            v = self._v.clone()

        # Remove component along constant vector (1/sqrt(n))
        ones = torch.ones(n, device=dev) / math.sqrt(n)
        v -= (v @ ones) * ones
        v /= v.norm().clamp(min=1e-10)

        # Power iteration on L = D - A
        for _ in range(self.n_iter):
            # Lv = Dv - Av
            Dv = deg * v
            Av = torch.zeros(n, device=dev)
            Av.scatter_add_(0, row.to(dev), weight.to(dev) * v[col.to(dev)])
            Lv = Dv - Av

            # Project out constant component
            Lv -= (Lv @ ones) * ones

            # Rayleigh quotient
            rq = float((v @ Lv).item()) / float((v @ v).item() + 1e-10)

            # Normalise
            norm = Lv.norm()
            if norm < 1e-10:
                break
            v = Lv / norm

        # Final Rayleigh quotient
        Dv = deg * v
        Av = torch.zeros(n, device=dev)
        Av.scatter_add_(0, row.to(dev), weight.to(dev) * v[col.to(dev)])
        Lv = Dv - Av
        fiedler = float((v @ Lv).item()) / float((v @ v).item() + 1e-10)
        fiedler = max(0.0, fiedler)

        self._v = v
        self._last_fiedler = fiedler
        return fiedler

    # ------------------------------------------------------------------
    def is_disconnected(self) -> bool:
        return self._last_fiedler < self.threshold

    # ------------------------------------------------------------------
    @property
    def last_fiedler(self) -> float:
        return self._last_fiedler


# ---------------------------------------------------------------------------
# MST fallback (Kruskal's via sorted edges)
# ---------------------------------------------------------------------------

class MSTFallback:
    """
    Computes a minimum spanning tree using Kruskal's algorithm when the
    graph becomes disconnected.

    Uses a union-find data structure for O(E * alpha(N)) complexity.
    """

    def __init__(self, n_nodes: int) -> None:
        self.n = n_nodes

    # ------------------------------------------------------------------
    def compute(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the maximum spanning tree (keep high-weight edges).

        Returns
        -------
        mst_row, mst_col, mst_weight : (N-1,) each
        """
        n = self.n
        # Sort descending by weight (max spanning tree)
        sorted_idx = torch.argsort(weight, descending=True)
        sorted_row = row[sorted_idx].cpu().tolist()
        sorted_col = col[sorted_idx].cpu().tolist()
        sorted_w = weight[sorted_idx].cpu().tolist()

        # Union-find
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True

        mst_rows, mst_cols, mst_ws = [], [], []
        for s, d, w in zip(sorted_row, sorted_col, sorted_w):
            if union(s, d):
                mst_rows.append(s)
                mst_rows.append(d)
                mst_cols.append(d)
                mst_cols.append(s)
                mst_ws.append(w)
                mst_ws.append(w)
            if len(mst_rows) >= 2 * (n - 1):
                break

        dev = row.device
        if not mst_rows:
            return (
                torch.zeros(0, dtype=torch.int64, device=dev),
                torch.zeros(0, dtype=torch.int64, device=dev),
                torch.zeros(0, dtype=torch.float32, device=dev),
            )

        return (
            torch.tensor(mst_rows, dtype=torch.int64, device=dev),
            torch.tensor(mst_cols, dtype=torch.int64, device=dev),
            torch.tensor(mst_ws, dtype=torch.float32, device=dev),
        )


# ---------------------------------------------------------------------------
# Density guard
# ---------------------------------------------------------------------------

class DensityGuard:
    """
    Prunes the graph if density exceeds the configured threshold.
    Keeps the k-nearest edges by weight for each node.
    """

    def __init__(self, max_density: float = 0.15, k: int = 20) -> None:
        self.max_density = max_density
        self.k = k

    # ------------------------------------------------------------------
    def check_and_prune(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        weight: torch.Tensor,
        n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        Prune edges if density exceeds threshold.

        Returns
        -------
        (row, col, weight, was_pruned)
        """
        n_edges = row.shape[0] // 2  # undirected
        max_edges = n * (n - 1) // 2
        density = n_edges / max(max_edges, 1)

        if density <= self.max_density:
            return row, col, weight, False

        logger.info(
            "DensityGuard: density=%.4f > max=%.4f; pruning to k=%d",
            density, self.max_density, self.k
        )

        # Keep top-k edges per node
        keep_set: set = set()
        dev = row.device

        row_cpu = row.cpu().tolist()
        col_cpu = col.cpu().tolist()
        w_cpu = weight.cpu().tolist()

        # Group by source node
        from collections import defaultdict
        node_edges: Dict[int, List[Tuple[float, int, int]]] = defaultdict(list)
        for r, c, w in zip(row_cpu, col_cpu, w_cpu):
            if r < c:  # upper triangle only
                node_edges[r].append((-w, r, c))

        for node, edges in node_edges.items():
            edges.sort()
            for _, r, c in edges[: self.k]:
                keep_set.add((r, c))

        # Build mask
        mask = torch.tensor(
            [
                (int(r), int(c)) in keep_set or (int(c), int(r)) in keep_set
                for r, c in zip(row_cpu, col_cpu)
            ],
            dtype=torch.bool,
            device=dev,
        )

        return row[mask], col[mask], weight[mask], True


# ---------------------------------------------------------------------------
# GSR publisher (stub — real implementation would use RTEL shm-bus)
# ---------------------------------------------------------------------------

class GSRPublisher:
    """
    Publishes graph snapshots to the RTEL Global State Registry (GSR).

    In production this writes to a shared memory ring buffer.
    This stub logs the publish event.
    """

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self._publish_count: int = 0

    # ------------------------------------------------------------------
    def publish(self, snapshot: GraphSnapshot) -> None:
        """Publish a graph snapshot."""
        self._publish_count += 1
        logger.debug(
            "GSR publish [%s] tick=%d edges=%d density=%.4f",
            self.topic, snapshot.tick_id, snapshot.n_edges, snapshot.density,
        )

    # ------------------------------------------------------------------
    @property
    def publish_count(self) -> int:
        return self._publish_count


# ---------------------------------------------------------------------------
# LOB snapshot (input data type from the shm-bus)
# ---------------------------------------------------------------------------

@dataclass
class LOBSnapshot:
    """
    LOB (Limit Order Book) snapshot from the RTEL shm-bus.
    Contains pre-computed features for each asset.
    """
    tick_id: int
    timestamp_ns: int
    features: torch.Tensor          # (N, D) float32
    returns: torch.Tensor           # (N,) float32 — last tick returns
    mid_prices: torch.Tensor        # (N,) float32


# ---------------------------------------------------------------------------
# Construction worker
# ---------------------------------------------------------------------------

class ConstructionWorker:
    """
    The main graph construction logic, run in a background thread.

    Reads LOBSnapshot objects from the input queue, runs the IAUK kernel,
    checks connectivity, applies density guard, and publishes snapshots.
    """

    def __init__(self, cfg: SGEConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # Import here to avoid circular imports at module level
        from omni_graph.incremental_adjacency import IAUKernel, IAUKConfig
        iauk_cfg = IAUKConfig(
            n_assets=cfg.n_assets,
            feature_dim=cfg.feature_dim,
            device=str(self.device),
        )
        self.kernel = IAUKernel(iauk_cfg)

        self.fiedler = FiedlerMonitor(
            n_nodes=cfg.n_assets,
            threshold=cfg.fiedler_threshold,
            device=self.device,
        )
        self.mst = MSTFallback(n_nodes=cfg.n_assets)
        self.density_guard = DensityGuard(
            max_density=cfg.max_density, k=cfg.density_knn
        )
        self.regime_detector = RegimeDetector(
            low_threshold=cfg.vol_low_threshold,
            high_threshold=cfg.vol_high_threshold,
            crisis_threshold=cfg.vol_crisis_threshold,
        )
        self.scheduler = AdaptiveUpdateScheduler(
            freq_low_vol=cfg.update_freq_low_vol,
            freq_normal=cfg.update_freq_normal,
            freq_high_vol=cfg.update_freq_high_vol,
            freq_crisis=cfg.update_freq_crisis,
        )
        self.priority_queue = PriorityEdgeQueue()
        self.gsr = GSRPublisher(cfg.gsr_topic) if cfg.publish_to_gsr else None

        # Metrics
        self._n_updates: int = 0
        self._n_mst_fallbacks: int = 0
        self._n_density_prunes: int = 0
        self._update_times_ms: List[float] = []

    # ------------------------------------------------------------------
    def process(self, lob: LOBSnapshot) -> Optional[GraphSnapshot]:
        """
        Process a single LOB snapshot and return a graph snapshot
        if an update was triggered this tick.
        """
        t0 = time.perf_counter()

        # Detect regime
        mean_return = float(lob.returns.mean().item())
        regime = self.regime_detector.update(mean_return)

        # Adaptive scheduling: skip update this tick?
        if not self.scheduler.should_update(regime):
            return None

        # Run IAUK kernel
        stats = self.kernel.update(lob.features, tick_id=lob.tick_id)
        ei, ew = self.kernel.get_pyg_edge_index()

        # Move to CPU for safe cross-thread sharing
        row = ei[0].cpu()
        col = ei[1].cpu()
        weight = ew.cpu()
        n = self.cfg.n_assets

        # Fiedler connectivity check
        fiedler = self.fiedler.estimate(row, col, weight)
        fallback_mode = FallbackMode.FULL_GRAPH

        if self.fiedler.is_disconnected() and self.cfg.mst_fallback:
            logger.warning(
                "Fiedler=%.6f < threshold=%.6f at tick=%d; MST fallback",
                fiedler, self.cfg.fiedler_threshold, lob.tick_id,
            )
            row, col, weight = self.mst.compute(row, col, weight)
            fiedler = self.fiedler.estimate(row, col, weight)
            fallback_mode = FallbackMode.MINIMUM_SPANNING_TREE
            self._n_mst_fallbacks += 1

        # Density guard
        row, col, weight, pruned = self.density_guard.check_and_prune(
            row, col, weight, n
        )
        if pruned:
            fallback_mode = FallbackMode.K_NEAREST
            self._n_density_prunes += 1

        n_edges = row.shape[0] // 2
        max_edges = n * (n - 1) // 2
        density = n_edges / max(max_edges, 1)

        elapsed = (time.perf_counter() - t0) * 1000.0
        self._update_times_ms.append(elapsed)
        self._n_updates += 1

        snapshot = GraphSnapshot(
            tick_id=lob.tick_id,
            timestamp_ns=lob.timestamp_ns,
            edge_index=torch.stack([row, col], dim=0),
            edge_weight=weight,
            n_nodes=n,
            n_edges=n_edges,
            density=density,
            fiedler_value=fiedler,
            regime=regime,
            fallback_mode=fallback_mode,
            construction_time_ms=elapsed,
        )

        if self.gsr is not None:
            self.gsr.publish(snapshot)

        return snapshot

    # ------------------------------------------------------------------
    def metrics(self) -> Dict[str, Any]:
        times = self._update_times_ms
        if not times:
            return {"n_updates": 0}
        return {
            "n_updates": self._n_updates,
            "n_mst_fallbacks": self._n_mst_fallbacks,
            "n_density_prunes": self._n_density_prunes,
            "mean_update_ms": sum(times) / len(times),
            "max_update_ms": max(times),
            "p99_update_ms": _pct(times, 99),
        }


# ---------------------------------------------------------------------------
# Streaming Graph Engine
# ---------------------------------------------------------------------------

class StreamingGraphEngine:
    """
    Streaming Graph Construction Engine — main public API.

    Runs a background construction thread that reads LOB snapshots from
    an input queue, builds graph snapshots, and makes them available to
    the inference thread via a double buffer.

    Usage
    -----
        cfg = SGEConfig(n_assets=500, feature_dim=64)
        engine = StreamingGraphEngine(cfg)
        engine.start()

        # Producer side (data feed):
        engine.push_lob(lob_snapshot)

        # Consumer side (inference):
        snapshot = engine.get_latest_graph()
        if snapshot:
            gnn_forward(snapshot.edge_index, snapshot.edge_weight)

        engine.stop()
    """

    def __init__(self, cfg: Optional[SGEConfig] = None, **kwargs: Any) -> None:
        if cfg is None:
            cfg = SGEConfig(**kwargs)
        self.cfg = cfg

        # Queues
        self._input_q: queue.Queue[LOBSnapshot] = queue.Queue(
            maxsize=cfg.input_queue_size
        )
        self._output_q: queue.Queue[GraphSnapshot] = queue.Queue(
            maxsize=cfg.output_queue_size
        )

        # Double buffer for lock-free inference access
        self._double_buf = DoubleBuffer()

        # Construction worker
        self._worker = ConstructionWorker(cfg)

        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._state = EngineState.IDLE
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Callbacks
        self._on_snapshot: Optional[Callable[[GraphSnapshot], None]] = None

        # Telemetry
        self._dropped_lob_count: int = 0
        self._snapshots_published: int = 0
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background construction thread."""
        with self._state_lock:
            if self._state not in (EngineState.IDLE, EngineState.STOPPED):
                raise RuntimeError(f"Cannot start engine in state {self._state}")
            self._stop_event.clear()
            self._state = EngineState.RUNNING

        self._start_time = time.perf_counter()
        self._thread = threading.Thread(
            target=self._construction_loop,
            name="GraphConstructionThread",
            daemon=self.cfg.thread_daemon,
        )
        self._thread.start()
        logger.info("StreamingGraphEngine started")

    # ------------------------------------------------------------------
    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the background thread gracefully."""
        with self._state_lock:
            if self._state != EngineState.RUNNING:
                return
            self._state = EngineState.STOPPING

        self._stop_event.set()
        # Poison pill
        try:
            self._input_q.put_nowait(None)  # type: ignore[arg-type]
        except queue.Full:
            pass

        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                logger.warning("Construction thread did not stop within %.1f s", timeout_s)

        with self._state_lock:
            self._state = EngineState.STOPPED

        logger.info("StreamingGraphEngine stopped")

    # ------------------------------------------------------------------
    def pause(self) -> None:
        """Pause graph construction (input queue keeps accumulating)."""
        with self._state_lock:
            if self._state == EngineState.RUNNING:
                self._state = EngineState.PAUSED

    # ------------------------------------------------------------------
    def resume(self) -> None:
        """Resume graph construction after a pause."""
        with self._state_lock:
            if self._state == EngineState.PAUSED:
                self._state = EngineState.RUNNING

    # ------------------------------------------------------------------
    def push_lob(self, lob: LOBSnapshot, block: bool = False, timeout: float = 0.001) -> bool:
        """
        Push a LOB snapshot to the construction queue.

        Parameters
        ----------
        lob     : LOBSnapshot
        block   : if True, wait up to timeout seconds
        timeout : seconds to wait if block=True

        Returns
        -------
        True if successfully enqueued, False if dropped
        """
        try:
            if block:
                self._input_q.put(lob, timeout=timeout)
            else:
                self._input_q.put_nowait(lob)
            return True
        except queue.Full:
            self._dropped_lob_count += 1
            return False

    # ------------------------------------------------------------------
    def get_latest_graph(self) -> Optional[GraphSnapshot]:
        """
        Get the latest completed graph snapshot (non-blocking).
        Returns None if no snapshot has been built yet.
        """
        return self._double_buf.read()

    # ------------------------------------------------------------------
    def wait_for_graph(self, timeout_ms: float = 100.0) -> Optional[GraphSnapshot]:
        """
        Block until a new graph snapshot is available.

        Parameters
        ----------
        timeout_ms : max wait time in milliseconds

        Returns
        -------
        GraphSnapshot or None on timeout
        """
        return self._double_buf.wait_for_update(timeout_ms)

    # ------------------------------------------------------------------
    def register_snapshot_callback(self, fn: Callable[[GraphSnapshot], None]) -> None:
        """Register a callback invoked each time a new snapshot is published."""
        self._on_snapshot = fn

    # ------------------------------------------------------------------
    def _construction_loop(self) -> None:
        """Main loop for the background construction thread."""
        logger.debug("Construction loop started")

        while not self._stop_event.is_set():
            # Check for pause
            with self._state_lock:
                if self._state == EngineState.PAUSED:
                    time.sleep(0.001)
                    continue
                if self._state == EngineState.STOPPING:
                    break

            # Drain input queue
            try:
                lob = self._input_q.get(timeout=0.005)
            except queue.Empty:
                continue

            if lob is None:
                # Poison pill
                break

            # Process
            try:
                snapshot = self._worker.process(lob)
            except Exception as exc:
                logger.exception("Error in construction worker: %s", exc)
                continue

            if snapshot is not None:
                # Write to double buffer
                self._double_buf.write(snapshot)
                self._snapshots_published += 1

                # Push to output queue (non-blocking drop if full)
                try:
                    self._output_q.put_nowait(snapshot)
                except queue.Full:
                    pass

                # Fire callback
                if self._on_snapshot is not None:
                    try:
                        self._on_snapshot(snapshot)
                    except Exception as exc:
                        logger.exception("Snapshot callback error: %s", exc)

        logger.debug("Construction loop exited")

    # ------------------------------------------------------------------
    @property
    def state(self) -> EngineState:
        return self._state

    # ------------------------------------------------------------------
    def metrics(self) -> Dict[str, Any]:
        """Return engine performance metrics."""
        worker_m = self._worker.metrics()
        uptime = time.perf_counter() - self._start_time if self._start_time else 0.0
        return {
            **worker_m,
            "state": self._state.name,
            "snapshots_published": self._snapshots_published,
            "dropped_lob": self._dropped_lob_count,
            "input_queue_size": self._input_q.qsize(),
            "output_queue_size": self._output_q.qsize(),
            "uptime_s": uptime,
            "current_regime": self._worker.regime_detector.current_regime.name,
            "current_vol": self._worker.regime_detector.current_vol(),
            "fiedler": self._worker.fiedler.last_fiedler,
        }

    # ------------------------------------------------------------------
    def drain_output_queue(self, max_items: int = 100) -> List[GraphSnapshot]:
        """Drain pending snapshots from the output queue."""
        snapshots: List[GraphSnapshot] = []
        for _ in range(max_items):
            try:
                snapshots.append(self._output_q.get_nowait())
            except queue.Empty:
                break
        return snapshots

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"StreamingGraphEngine("
            f"n={self.cfg.n_assets}, "
            f"state={self._state.name}, "
            f"published={self._snapshots_published})"
        )

    # ------------------------------------------------------------------
    def __enter__(self) -> "StreamingGraphEngine":
        self.start()
        return self

    # ------------------------------------------------------------------
    def __exit__(self, *_: Any) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Multi-engine load balancer (shard by asset subset)
# ---------------------------------------------------------------------------

class ShardedGraphEngine:
    """
    Runs multiple StreamingGraphEngines, each responsible for a shard
    of the asset universe.  Used for very large N (e.g. 5000+ assets).

    Each shard handles assets [i*shard_size : (i+1)*shard_size].
    Inter-shard edges are handled by a cross-shard connector.
    """

    def __init__(
        self,
        total_assets: int,
        n_shards: int,
        feature_dim: int = 64,
        device: str = "cuda",
    ) -> None:
        self.total_assets = total_assets
        self.n_shards = n_shards
        self.shard_size = math.ceil(total_assets / n_shards)

        self._engines: List[StreamingGraphEngine] = []
        for i in range(n_shards):
            shard_n = min(self.shard_size, total_assets - i * self.shard_size)
            if shard_n <= 0:
                break
            cfg = SGEConfig(n_assets=shard_n, feature_dim=feature_dim, device=device)
            self._engines.append(StreamingGraphEngine(cfg))

    # ------------------------------------------------------------------
    def start_all(self) -> None:
        for eng in self._engines:
            eng.start()

    # ------------------------------------------------------------------
    def stop_all(self) -> None:
        for eng in self._engines:
            eng.stop()

    # ------------------------------------------------------------------
    def push_lob_sharded(self, features: torch.Tensor, tick_id: int) -> None:
        """Distribute a full feature matrix across shards."""
        T_ns = time.time_ns()
        N = features.shape[0]
        D = features.shape[1] if features.dim() > 1 else 1

        for i, eng in enumerate(self._engines):
            start = i * self.shard_size
            end = min(start + self.shard_size, N)
            shard_feat = features[start:end]
            lob = LOBSnapshot(
                tick_id=tick_id,
                timestamp_ns=T_ns,
                features=shard_feat,
                returns=torch.zeros(end - start),
                mid_prices=torch.ones(end - start),
            )
            eng.push_lob(lob)

    # ------------------------------------------------------------------
    def get_all_snapshots(self) -> List[Optional[GraphSnapshot]]:
        return [eng.get_latest_graph() for eng in self._engines]

    # ------------------------------------------------------------------
    def merge_snapshots(
        self,
        snapshots: List[Optional[GraphSnapshot]],
    ) -> Optional[GraphSnapshot]:
        """
        Merge shard snapshots into a single global snapshot by offsetting
        node indices.
        """
        valid = [s for s in snapshots if s is not None]
        if not valid:
            return None

        all_row, all_col, all_w = [], [], []
        node_offset = 0

        for i, s in enumerate(valid):
            row = s.edge_index[0] + node_offset
            col = s.edge_index[1] + node_offset
            all_row.append(row)
            all_col.append(col)
            all_w.append(s.edge_weight)
            node_offset += s.n_nodes

        merged_row = torch.cat(all_row)
        merged_col = torch.cat(all_col)
        merged_w = torch.cat(all_w)
        merged_ei = torch.stack([merged_row, merged_col], dim=0)

        n_edges = merged_row.shape[0] // 2
        n_total = node_offset
        density = n_edges / max(n_total * (n_total - 1) // 2, 1)

        return GraphSnapshot(
            tick_id=valid[0].tick_id,
            timestamp_ns=valid[0].timestamp_ns,
            edge_index=merged_ei,
            edge_weight=merged_w,
            n_nodes=n_total,
            n_edges=n_edges,
            density=density,
            fiedler_value=min(s.fiedler_value for s in valid),
            regime=valid[0].regime,
            fallback_mode=valid[0].fallback_mode,
            construction_time_ms=max(s.construction_time_ms for s in valid),
        )

    # ------------------------------------------------------------------
    @property
    def n_engines(self) -> int:
        return len(self._engines)


# ---------------------------------------------------------------------------
# Replay engine (for backtesting)
# ---------------------------------------------------------------------------

class ReplayGraphEngine:
    """
    Synchronous replay of historical LOB data.
    Useful for backtesting and strategy simulation without threading.
    """

    def __init__(self, cfg: Optional[SGEConfig] = None, **kwargs: Any) -> None:
        if cfg is None:
            cfg = SGEConfig(**kwargs)
        self.cfg = cfg
        self._worker = ConstructionWorker(cfg)
        self._snapshots: List[GraphSnapshot] = []

    # ------------------------------------------------------------------
    def replay(
        self,
        features_sequence: torch.Tensor,   # (T, N, D)
        returns_sequence: Optional[torch.Tensor] = None,  # (T, N)
    ) -> List[GraphSnapshot]:
        """
        Replay a full feature sequence and collect snapshots.

        Parameters
        ----------
        features_sequence : (T, N, D) float32
        returns_sequence  : optional (T, N) float32

        Returns
        -------
        snapshots : list of GraphSnapshot (one per construction tick)
        """
        T = features_sequence.shape[0]
        T_start_ns = time.time_ns()

        self._snapshots = []
        for t in range(T):
            feat = features_sequence[t]
            ret = returns_sequence[t] if returns_sequence is not None else torch.zeros(
                feat.shape[0]
            )
            lob = LOBSnapshot(
                tick_id=t,
                timestamp_ns=T_start_ns + t * 1_000_000,
                features=feat,
                returns=ret,
                mid_prices=torch.ones(feat.shape[0]),
            )
            snap = self._worker.process(lob)
            if snap is not None:
                self._snapshots.append(snap)

        return self._snapshots

    # ------------------------------------------------------------------
    def get_snapshots(self) -> List[GraphSnapshot]:
        return self._snapshots

    # ------------------------------------------------------------------
    def metrics(self) -> Dict[str, Any]:
        return self._worker.metrics()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _pct(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def make_lob_snapshot(
    tick_id: int,
    features: torch.Tensor,
    returns: Optional[torch.Tensor] = None,
) -> LOBSnapshot:
    """Convenience constructor for LOBSnapshot."""
    N = features.shape[0]
    return LOBSnapshot(
        tick_id=tick_id,
        timestamp_ns=time.time_ns(),
        features=features,
        returns=returns if returns is not None else torch.zeros(N),
        mid_prices=torch.ones(N),
    )


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "SGEConfig",
    "StreamingGraphEngine",
    "GraphSnapshot",
    "LOBSnapshot",
    "EngineState",
    "GraphRegime",
    "FallbackMode",
    "PriorityEdgeQueue",
    "DoubleBuffer",
    "RegimeDetector",
    "AdaptiveUpdateScheduler",
    "FiedlerMonitor",
    "MSTFallback",
    "DensityGuard",
    "ConstructionWorker",
    "ShardedGraphEngine",
    "ReplayGraphEngine",
    "GSRPublisher",
    "make_lob_snapshot",
]
