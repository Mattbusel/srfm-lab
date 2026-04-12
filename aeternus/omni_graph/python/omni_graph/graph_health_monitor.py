"""
graph_health_monitor.py — Real-time Graph Health Monitor
=========================================================
Part of the Omni-Graph incremental graph construction suite.

Design goals
------------
* Connectivity check every N ticks using union-find (O(α(N))).
* Dead node detection: nodes with no edges updated in > T seconds.
* Density shock alert: graph density change > X% in one tick.
* Spectral gap collapse alert: Fiedler value drops below threshold
  (used as a liquidity crisis signal).
* Auto-repair: reconnect isolated nodes via minimum distance edge to
  nearest cluster centroid.
* Health metrics published to RTEL GSR (stub + real interface).

Public API
----------
    monitor = GraphHealthMonitor(n_nodes=500)
    alert = monitor.check(edge_index, edge_weight, tick_id)
    if alert.has_alerts:
        monitor.auto_repair(edge_index, edge_weight, node_features)
    monitor.publish_metrics(tick_id)
"""

from __future__ import annotations

import math
import time
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict, List, Any, Set

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert types
# ---------------------------------------------------------------------------

class AlertSeverity(Enum):
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


class AlertType(Enum):
    DISCONNECTED = auto()
    DEAD_NODE = auto()
    DENSITY_SHOCK = auto()
    SPECTRAL_GAP_COLLAPSE = auto()
    DENSITY_EXPLOSION = auto()
    ISOLATED_CLUSTER = auto()
    EDGE_WEIGHT_COLLAPSE = auto()


@dataclass
class GraphAlert:
    """A single health alert raised by the monitor."""

    alert_type: AlertType
    severity: AlertSeverity
    tick_id: int
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        return (
            f"[{self.severity.name}] {self.alert_type.name} "
            f"@ tick={self.tick_id}: {self.message}"
        )


@dataclass
class HealthCheckResult:
    """Aggregated result of a health check cycle."""

    tick_id: int
    timestamp_ns: int
    alerts: List[GraphAlert]
    n_connected_components: int
    n_dead_nodes: int
    density: float
    fiedler_estimate: float
    mean_edge_weight: float
    is_healthy: bool
    check_time_ms: float

    # ------------------------------------------------------------------
    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0

    # ------------------------------------------------------------------
    @property
    def has_critical(self) -> bool:
        return any(a.severity == AlertSeverity.CRITICAL for a in self.alerts)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"HealthCheckResult(tick={self.tick_id}, "
            f"alerts={len(self.alerts)}, "
            f"components={self.n_connected_components}, "
            f"density={self.density:.4f}, "
            f"fiedler={self.fiedler_estimate:.4f}, "
            f"{'CRITICAL' if self.has_critical else 'OK'})"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GHMConfig:
    """Configuration for GraphHealthMonitor."""

    n_nodes: int = 500
    device: str = "cuda"

    # Connectivity check frequency (every N ticks)
    connectivity_check_interval: int = 10

    # Dead node: no edge update in > T seconds
    dead_node_timeout_s: float = 60.0

    # Density shock: alert if density changes by > X fraction in one tick
    density_shock_threshold: float = 0.05

    # Density explosion: alert if absolute density > this
    density_max_threshold: float = 0.50

    # Spectral gap collapse: Fiedler < this → crisis signal
    fiedler_crisis_threshold: float = 0.001
    fiedler_warning_threshold: float = 0.01

    # Edge weight collapse: mean weight < this
    min_mean_weight: float = 0.05

    # Auto-repair: connect isolated nodes to nearest centroid
    auto_repair_enabled: bool = True
    repair_min_weight: float = 0.15

    # Publish interval (every N health checks)
    publish_interval: int = 50

    # Fiedler estimation: Lanczos iterations
    fiedler_lanczos_k: int = 15

    # History buffer for metric tracking
    history_length: int = 1000


# ---------------------------------------------------------------------------
# Union-Find for connectivity
# ---------------------------------------------------------------------------

class UnionFind:
    """
    Path-compressed, union-by-rank union-find data structure.
    O(α(N)) per operation.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self._parent = list(range(n))
        self._rank = [0] * n
        self._n_components = n

    # ------------------------------------------------------------------
    def find(self, x: int) -> int:
        """Path-compressed find."""
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path halving
            x = self._parent[x]
        return x

    # ------------------------------------------------------------------
    def union(self, x: int, y: int) -> bool:
        """
        Union the sets containing x and y.
        Returns True if they were in different sets.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        self._n_components -= 1
        return True

    # ------------------------------------------------------------------
    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    # ------------------------------------------------------------------
    @property
    def n_components(self) -> int:
        return self._n_components

    # ------------------------------------------------------------------
    def get_components(self) -> Dict[int, List[int]]:
        """Return {root: [node, ...]} mapping."""
        comps: Dict[int, List[int]] = {}
        for i in range(self.n):
            r = self.find(i)
            comps.setdefault(r, []).append(i)
        return comps

    # ------------------------------------------------------------------
    def isolated_nodes(self) -> List[int]:
        """Return nodes in singleton components."""
        comps = self.get_components()
        return [nodes[0] for nodes in comps.values() if len(nodes) == 1]

    # ------------------------------------------------------------------
    @classmethod
    def from_edge_index(cls, n: int, row: torch.Tensor, col: torch.Tensor) -> "UnionFind":
        """Build union-find from edge COO tensors."""
        uf = cls(n)
        row_list = row.cpu().tolist()
        col_list = col.cpu().tolist()
        for r, c in zip(row_list, col_list):
            uf.union(int(r), int(c))
        return uf


# ---------------------------------------------------------------------------
# Dead node tracker
# ---------------------------------------------------------------------------

class DeadNodeTracker:
    """
    Tracks the last time each node's edges were updated.
    Raises alerts for nodes that haven't been touched in > timeout seconds.
    """

    def __init__(self, n: int, timeout_s: float) -> None:
        self.n = n
        self.timeout_s = timeout_s
        self._last_update: List[float] = [0.0] * n
        self._birth_time = time.monotonic()

    # ------------------------------------------------------------------
    def update_nodes(self, node_indices: List[int]) -> None:
        """Record that the given nodes were updated now."""
        now = time.monotonic()
        for i in node_indices:
            if 0 <= i < self.n:
                self._last_update[i] = now

    # ------------------------------------------------------------------
    def update_from_edges(self, row: torch.Tensor) -> None:
        """Update all nodes that appear as sources in edge list."""
        nodes = row.cpu().unique().tolist()
        self.update_nodes([int(n) for n in nodes])

    # ------------------------------------------------------------------
    def get_dead_nodes(self) -> List[int]:
        """Return nodes not updated within the timeout window."""
        now = time.monotonic()
        threshold = now - self.timeout_s
        dead = [
            i for i in range(self.n)
            if self._last_update[i] < threshold and
            (now - self._birth_time) > self.timeout_s
        ]
        return dead

    # ------------------------------------------------------------------
    def time_since_update(self, node: int) -> float:
        """Return seconds since node was last updated."""
        return time.monotonic() - self._last_update[node]

    # ------------------------------------------------------------------
    def is_dead(self, node: int) -> bool:
        return self.time_since_update(node) > self.timeout_s

    # ------------------------------------------------------------------
    def reset(self) -> None:
        now = time.monotonic()
        self._last_update = [now] * self.n


# ---------------------------------------------------------------------------
# Density tracker
# ---------------------------------------------------------------------------

class DensityTracker:
    """Tracks graph density over time and detects shocks."""

    def __init__(self, n: int, shock_threshold: float, max_threshold: float) -> None:
        self.n = n
        self.max_possible = n * (n - 1) // 2
        self.shock_threshold = shock_threshold
        self.max_threshold = max_threshold

        self._history: List[float] = []
        self._current: float = 0.0

    # ------------------------------------------------------------------
    def update(self, n_edges: int) -> Tuple[float, float]:
        """
        Update with current number of edges.

        Returns
        -------
        density : current density
        delta   : change from previous tick
        """
        density = n_edges / max(self.max_possible, 1)
        delta = abs(density - self._current)
        self._current = density
        self._history.append(density)
        if len(self._history) > 1000:
            self._history.pop(0)
        return density, delta

    # ------------------------------------------------------------------
    def is_shock(self, delta: float) -> bool:
        return delta > self.shock_threshold

    # ------------------------------------------------------------------
    def is_explosion(self, density: float) -> bool:
        return density > self.max_threshold

    # ------------------------------------------------------------------
    def rolling_mean(self, window: int = 20) -> float:
        tail = self._history[-window:]
        return sum(tail) / max(len(tail), 1)

    # ------------------------------------------------------------------
    def trend(self, window: int = 10) -> float:
        """Positive = growing, negative = shrinking."""
        if len(self._history) < window + 1:
            return 0.0
        recent = self._history[-window:]
        older = self._history[-2 * window:-window]
        if not older:
            return 0.0
        return sum(recent) / len(recent) - sum(older) / len(older)

    # ------------------------------------------------------------------
    @property
    def current(self) -> float:
        return self._current


# ---------------------------------------------------------------------------
# Fiedler estimator (lightweight, inline)
# ---------------------------------------------------------------------------

class InlineFiedlerEstimator:
    """
    Fast inline Fiedler estimator using 10 Lanczos iterations.
    Designed for the health monitor's lightweight budget.
    """

    def __init__(self, n: int, k: int = 15, device: torch.device = torch.device("cpu")) -> None:
        self.n = n
        self.k = k
        self.device = device
        self._v0: Optional[torch.Tensor] = None
        self._last: float = 1.0

    # ------------------------------------------------------------------
    def estimate(
        self, row: torch.Tensor, col: torch.Tensor, weight: torch.Tensor
    ) -> float:
        n = self.n
        dev = self.device
        k = min(self.k, n - 1)
        if k < 2:
            return 0.0

        row_d = row.to(dev)
        col_d = col.to(dev)
        w_d = weight.to(dev, dtype=torch.float32)

        deg = torch.zeros(n, device=dev)
        if row_d.numel() > 0:
            deg.scatter_add_(0, row_d, w_d)

        ones = torch.ones(n, device=dev) / math.sqrt(n)

        if self._v0 is not None and self._v0.shape[0] == n:
            v = self._v0.clone()
        else:
            v = torch.randn(n, device=dev)

        v -= (v @ ones) * ones
        nrm = v.norm()
        if nrm < 1e-10:
            return 0.0
        v /= nrm

        def Lv(u: torch.Tensor) -> torch.Tensor:
            Du = deg * u
            Au = torch.zeros(n, device=dev)
            if row_d.numel() > 0:
                Au.scatter_add_(0, col_d, w_d * u[row_d])
            return Du - Au

        alphas: List[float] = []
        betas: List[float] = []
        prev_v = torch.zeros(n, device=dev)
        prev_beta = 0.0

        for _ in range(k):
            w_vec = Lv(v) - prev_beta * prev_v
            a = float((v @ w_vec).item())
            alphas.append(a)
            w_vec = w_vec - a * v
            b = float(w_vec.norm().item())
            if b < 1e-10:
                break
            betas.append(b)
            prev_v, prev_beta = v, b
            v = w_vec / b

        m = len(alphas)
        if m < 2:
            return 0.0

        T = torch.zeros(m, m, device=dev)
        for i in range(m):
            T[i, i] = alphas[i]
        for i in range(len(betas)):
            if i + 1 < m:
                T[i, i + 1] = betas[i]
                T[i + 1, i] = betas[i]

        try:
            ev = torch.linalg.eigvalsh(T).sort().values
        except Exception:
            return 0.0

        fiedler = max(0.0, float(ev[1].item())) if ev.numel() > 1 else 0.0
        self._last = fiedler
        self._v0 = v
        return fiedler


# ---------------------------------------------------------------------------
# Auto-repair engine
# ---------------------------------------------------------------------------

class AutoRepairEngine:
    """
    Reconnects isolated or weakly connected nodes by adding minimum-weight
    edges to the nearest cluster centroid.
    """

    def __init__(self, n: int, min_weight: float = 0.15, device: torch.device = torch.device("cpu")) -> None:
        self.n = n
        self.min_weight = min_weight
        self.device = device
        self._repair_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def repair_isolated(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        weight: torch.Tensor,
        isolated_nodes: List[int],
        node_features: Optional[torch.Tensor] = None,
        cluster_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add edges from isolated nodes to their nearest cluster.

        Parameters
        ----------
        row, col       : (E,) int64
        weight         : (E,) float32
        isolated_nodes : list of isolated node indices
        node_features  : (N, D) float32 optional — used for distance
        cluster_labels : (N,) int64 optional — cluster assignments

        Returns
        -------
        new_row, new_col, new_weight : augmented COO tensors
        """
        if not isolated_nodes:
            return row, col, weight

        dev = self.device
        n_repaired = 0
        add_row, add_col, add_w = [], [], []

        # Build cluster centroid map
        if node_features is not None and cluster_labels is not None:
            feats = node_features.to(dev, dtype=torch.float32)
            labels = cluster_labels.to(dev)
            k = int(labels.max().item()) + 1
            centroids = torch.zeros(k, feats.shape[1], device=dev)
            counts = torch.zeros(k, device=dev)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centroids[c] = feats[mask].mean(dim=0)
                    counts[c] = mask.sum()

        for iso in isolated_nodes:
            # Find nearest non-isolated node by feature similarity or random
            if node_features is not None and cluster_labels is not None:
                feat_iso = node_features[iso].to(dev, dtype=torch.float32)
                # Nearest cluster centroid
                dists = torch.cdist(feat_iso.unsqueeze(0), centroids)[0]
                nearest_cluster = int(dists.argmin().item())
                # Pick a node in that cluster
                cluster_nodes = (cluster_labels == nearest_cluster).nonzero(as_tuple=True)[0]
                if cluster_nodes.numel() == 0:
                    continue
                target = int(cluster_nodes[0].item())
                # Weight based on feature similarity
                feat_target = node_features[target].to(dev, dtype=torch.float32)
                sim = float(
                    F_cosine_similarity(feat_iso.unsqueeze(0), feat_target.unsqueeze(0)).item()
                )
                w = max(self.min_weight, abs(sim))
            else:
                # Random non-isolated node
                candidates = [j for j in range(self.n) if j != iso and j not in isolated_nodes]
                if not candidates:
                    continue
                target = candidates[0]
                w = self.min_weight

            add_row.extend([iso, target])
            add_col.extend([target, iso])
            add_w.extend([w, w])
            n_repaired += 1

        if not add_row:
            return row, col, weight

        add_r = torch.tensor(add_row, dtype=torch.int64, device=row.device)
        add_c = torch.tensor(add_col, dtype=torch.int64, device=col.device)
        add_wt = torch.tensor(add_w, dtype=torch.float32, device=weight.device)

        new_row = torch.cat([row, add_r])
        new_col = torch.cat([col, add_c])
        new_weight = torch.cat([weight, add_wt])

        self._repair_history.append({
            "n_repaired": n_repaired,
            "isolated": isolated_nodes[:10],  # truncate for logging
        })
        logger.info("AutoRepair: connected %d isolated nodes", n_repaired)
        return new_row, new_col, new_weight

    # ------------------------------------------------------------------
    def n_repairs(self) -> int:
        return sum(r["n_repaired"] for r in self._repair_history)


def F_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two batches of vectors."""
    a_norm = a / a.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    b_norm = b / b.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    return (a_norm * b_norm).sum(dim=1)


# ---------------------------------------------------------------------------
# Metric history store
# ---------------------------------------------------------------------------

class MetricHistory:
    """Circular buffer of health metric time series."""

    def __init__(self, capacity: int = 1000) -> None:
        self._capacity = capacity
        self._data: Dict[str, List[float]] = {}
        self._ticks: List[int] = []

    # ------------------------------------------------------------------
    def record(self, tick_id: int, metrics: Dict[str, float]) -> None:
        self._ticks.append(tick_id)
        for k, v in metrics.items():
            self._data.setdefault(k, []).append(v)
        if len(self._ticks) > self._capacity:
            self._ticks.pop(0)
            for k in self._data:
                if self._data[k]:
                    self._data[k].pop(0)

    # ------------------------------------------------------------------
    def get(self, key: str, window: int = 50) -> List[float]:
        return self._data.get(key, [])[-window:]

    # ------------------------------------------------------------------
    def rolling_mean(self, key: str, window: int = 20) -> float:
        vals = self.get(key, window)
        return sum(vals) / max(len(vals), 1)

    # ------------------------------------------------------------------
    def latest(self, key: str) -> Optional[float]:
        vals = self._data.get(key)
        return vals[-1] if vals else None

    # ------------------------------------------------------------------
    def keys(self) -> List[str]:
        return list(self._data.keys())


# ---------------------------------------------------------------------------
# GSR publisher (stub)
# ---------------------------------------------------------------------------

class HealthGSRPublisher:
    """Publishes health metrics to the RTEL GSR."""

    def __init__(self, topic: str = "omni_graph.health") -> None:
        self.topic = topic
        self._count = 0

    # ------------------------------------------------------------------
    def publish(self, result: HealthCheckResult) -> None:
        self._count += 1
        logger.debug(
            "GSR health [%s] tick=%d alerts=%d fiedler=%.4f density=%.4f",
            self.topic, result.tick_id, len(result.alerts),
            result.fiedler_estimate, result.density,
        )

    # ------------------------------------------------------------------
    @property
    def publish_count(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Main Graph Health Monitor
# ---------------------------------------------------------------------------

class GraphHealthMonitor:
    """
    Real-time graph health monitoring system.

    Usage
    -----
        monitor = GraphHealthMonitor(n_nodes=500)
        for tick in range(T):
            result = monitor.check(edge_index, edge_weight, tick_id=tick)
            if result.has_critical:
                ei, ew = monitor.auto_repair(ei, ew, node_feats)
    """

    def __init__(
        self,
        n_nodes: int,
        cfg: Optional[GHMConfig] = None,
        **kwargs: Any,
    ) -> None:
        if cfg is None:
            cfg = GHMConfig(n_nodes=n_nodes, **kwargs)
        self.cfg = cfg
        self.n = n_nodes
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )

        # Sub-components
        self._dead_node_tracker = DeadNodeTracker(n_nodes, cfg.dead_node_timeout_s)
        self._density_tracker = DensityTracker(
            n_nodes, cfg.density_shock_threshold, cfg.density_max_threshold
        )
        self._fiedler = InlineFiedlerEstimator(n_nodes, cfg.fiedler_lanczos_k, self.device)
        self._repair = AutoRepairEngine(n_nodes, cfg.repair_min_weight, self.device)
        self._history = MetricHistory(cfg.history_length)
        self._gsr = HealthGSRPublisher()

        # State
        self._tick_counter: int = 0
        self._last_result: Optional[HealthCheckResult] = None
        self._alert_log: List[GraphAlert] = []
        self._alert_counts: Dict[AlertType, int] = {}

        # Thread safety
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def check(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        tick_id: int,
        node_features: Optional[torch.Tensor] = None,
    ) -> HealthCheckResult:
        """
        Run a full health check cycle.

        Parameters
        ----------
        edge_index    : (2, E) int64
        edge_weight   : (E,) float32
        tick_id       : current tick
        node_features : optional (N, D) for auto-repair

        Returns
        -------
        HealthCheckResult
        """
        t0 = time.perf_counter()
        T_ns = time.time_ns()

        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device, dtype=torch.float32)
        row, col = edge_index[0], edge_index[1]
        n_edges_half = row.shape[0] // 2

        alerts: List[GraphAlert] = []

        # ---- 1. Connectivity check (every N ticks) ----
        n_components = 1
        n_dead = 0
        isolated: List[int] = []

        if self._tick_counter % self.cfg.connectivity_check_interval == 0:
            uf = UnionFind.from_edge_index(self.n, row, col)
            n_components = uf.n_components
            isolated = uf.isolated_nodes()

            if n_components > 1:
                severity = (
                    AlertSeverity.CRITICAL if n_components > self.n // 4
                    else AlertSeverity.WARNING
                )
                alerts.append(GraphAlert(
                    alert_type=AlertType.DISCONNECTED,
                    severity=severity,
                    tick_id=tick_id,
                    message=f"Graph has {n_components} connected components; "
                            f"{len(isolated)} isolated nodes",
                    metadata={"n_components": n_components, "isolated": isolated[:20]},
                ))

            if isolated:
                alerts.append(GraphAlert(
                    alert_type=AlertType.ISOLATED_CLUSTER,
                    severity=AlertSeverity.WARNING,
                    tick_id=tick_id,
                    message=f"{len(isolated)} isolated nodes detected",
                    metadata={"nodes": isolated[:20]},
                ))

        # ---- 2. Dead node detection ----
        self._dead_node_tracker.update_from_edges(row)
        if self._tick_counter % self.cfg.connectivity_check_interval == 0:
            dead_nodes = self._dead_node_tracker.get_dead_nodes()
            n_dead = len(dead_nodes)
            if n_dead > 0:
                frac = n_dead / self.n
                severity = (
                    AlertSeverity.CRITICAL if frac > 0.1
                    else AlertSeverity.WARNING
                )
                alerts.append(GraphAlert(
                    alert_type=AlertType.DEAD_NODE,
                    severity=severity,
                    tick_id=tick_id,
                    message=f"{n_dead} dead nodes ({frac:.1%} of graph)",
                    metadata={"dead_nodes": dead_nodes[:20], "frac": frac},
                ))

        # ---- 3. Density shock detection ----
        density, density_delta = self._density_tracker.update(n_edges_half)

        if self._density_tracker.is_shock(density_delta) and self._tick_counter > 0:
            alerts.append(GraphAlert(
                alert_type=AlertType.DENSITY_SHOCK,
                severity=AlertSeverity.WARNING,
                tick_id=tick_id,
                message=f"Density shock: Δdensity={density_delta:.4f} > "
                        f"threshold={self.cfg.density_shock_threshold:.4f}",
                metadata={"density": density, "delta": density_delta},
            ))

        if self._density_tracker.is_explosion(density):
            alerts.append(GraphAlert(
                alert_type=AlertType.DENSITY_EXPLOSION,
                severity=AlertSeverity.WARNING,
                tick_id=tick_id,
                message=f"Density explosion: density={density:.4f} > "
                        f"max={self.cfg.density_max_threshold:.4f}",
                metadata={"density": density},
            ))

        # ---- 4. Fiedler / spectral gap ----
        fiedler = 0.0
        if self._tick_counter % self.cfg.connectivity_check_interval == 0 and row.numel() > 0:
            fiedler = self._fiedler.estimate(row, col, edge_weight)
        else:
            fiedler = self._fiedler._last

        if fiedler < self.cfg.fiedler_crisis_threshold:
            alerts.append(GraphAlert(
                alert_type=AlertType.SPECTRAL_GAP_COLLAPSE,
                severity=AlertSeverity.CRITICAL,
                tick_id=tick_id,
                message=f"Spectral gap collapse: Fiedler={fiedler:.6f} < "
                        f"crisis_threshold={self.cfg.fiedler_crisis_threshold:.6f}",
                metadata={"fiedler": fiedler},
            ))
        elif fiedler < self.cfg.fiedler_warning_threshold:
            alerts.append(GraphAlert(
                alert_type=AlertType.SPECTRAL_GAP_COLLAPSE,
                severity=AlertSeverity.WARNING,
                tick_id=tick_id,
                message=f"Spectral gap low: Fiedler={fiedler:.6f} < "
                        f"warning_threshold={self.cfg.fiedler_warning_threshold:.6f}",
                metadata={"fiedler": fiedler},
            ))

        # ---- 5. Edge weight collapse ----
        mean_weight = float(edge_weight.mean().item()) if edge_weight.numel() > 0 else 0.0
        if mean_weight < self.cfg.min_mean_weight and edge_weight.numel() > 0:
            alerts.append(GraphAlert(
                alert_type=AlertType.EDGE_WEIGHT_COLLAPSE,
                severity=AlertSeverity.WARNING,
                tick_id=tick_id,
                message=f"Edge weight collapse: mean_weight={mean_weight:.4f} < "
                        f"min={self.cfg.min_mean_weight:.4f}",
                metadata={"mean_weight": mean_weight},
            ))

        # ---- 6. Update alert log and counts ----
        with self._lock:
            self._alert_log.extend(alerts)
            if len(self._alert_log) > 10_000:
                self._alert_log = self._alert_log[-5_000:]
            for a in alerts:
                self._alert_counts[a.alert_type] = \
                    self._alert_counts.get(a.alert_type, 0) + 1

        # ---- 7. Record metrics ----
        self._history.record(tick_id, {
            "density": density,
            "fiedler": fiedler,
            "mean_weight": mean_weight,
            "n_components": float(n_components),
            "n_dead": float(n_dead),
            "n_alerts": float(len(alerts)),
        })

        elapsed = (time.perf_counter() - t0) * 1000.0
        is_healthy = not any(a.severity == AlertSeverity.CRITICAL for a in alerts)

        result = HealthCheckResult(
            tick_id=tick_id,
            timestamp_ns=T_ns,
            alerts=alerts,
            n_connected_components=n_components,
            n_dead_nodes=n_dead,
            density=density,
            fiedler_estimate=fiedler,
            mean_edge_weight=mean_weight,
            is_healthy=is_healthy,
            check_time_ms=elapsed,
        )

        self._last_result = result
        self._tick_counter += 1

        # ---- 8. Publish to GSR if interval reached ----
        if self._tick_counter % self.cfg.publish_interval == 0:
            self._gsr.publish(result)

        # Log critical alerts
        for a in alerts:
            if a.severity == AlertSeverity.CRITICAL:
                logger.critical("%s", a)
            elif a.severity == AlertSeverity.WARNING:
                logger.warning("%s", a)

        return result

    # ------------------------------------------------------------------
    def auto_repair(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        cluster_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attempt to auto-repair graph health issues.

        Currently handles: isolated node reconnection.

        Returns
        -------
        repaired_edge_index  : (2, E') int64
        repaired_edge_weight : (E',) float32
        """
        if self._last_result is None:
            return edge_index, edge_weight

        # Get isolated nodes from last check
        isolated: List[int] = []
        for alert in self._last_result.alerts:
            if alert.alert_type == AlertType.ISOLATED_CLUSTER:
                isolated = alert.metadata.get("nodes", [])
                break
            if alert.alert_type == AlertType.DISCONNECTED:
                isolated = alert.metadata.get("isolated", [])

        if not isolated:
            return edge_index, edge_weight

        row, col = edge_index[0], edge_index[1]
        new_row, new_col, new_w = self._repair.repair_isolated(
            row, col, edge_weight, isolated, node_features, cluster_labels
        )
        new_ei = torch.stack([new_row, new_col], dim=0)
        return new_ei, new_w

    # ------------------------------------------------------------------
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return a summary of current health metrics."""
        return {
            "tick_counter": self._tick_counter,
            "total_alerts": sum(self._alert_counts.values()),
            "alert_counts": {k.name: v for k, v in self._alert_counts.items()},
            "last_density": self._history.latest("density"),
            "last_fiedler": self._history.latest("fiedler"),
            "last_mean_weight": self._history.latest("mean_weight"),
            "last_n_components": self._history.latest("n_components"),
            "rolling_density_20": self._history.rolling_mean("density", 20),
            "rolling_fiedler_20": self._history.rolling_mean("fiedler", 20),
            "n_auto_repairs": self._repair.n_repairs(),
            "gsr_publishes": self._gsr.publish_count,
        }

    # ------------------------------------------------------------------
    def get_recent_alerts(self, n: int = 10, alert_type: Optional[AlertType] = None) -> List[GraphAlert]:
        """Return the last n alerts, optionally filtered by type."""
        with self._lock:
            alerts = list(self._alert_log)
        if alert_type is not None:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        return alerts[-n:]

    # ------------------------------------------------------------------
    def is_crisis(self) -> bool:
        """Return True if the most recent check had a CRITICAL alert."""
        if self._last_result is None:
            return False
        return self._last_result.has_critical

    # ------------------------------------------------------------------
    def fiedler_history(self, window: int = 100) -> List[float]:
        """Return recent Fiedler value history."""
        return self._history.get("fiedler", window)

    # ------------------------------------------------------------------
    def density_history(self, window: int = 100) -> List[float]:
        return self._history.get("density", window)

    # ------------------------------------------------------------------
    def reset_dead_node_tracker(self) -> None:
        """Reset all dead node timestamps (e.g., after a full graph rebuild)."""
        self._dead_node_tracker.reset()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"GraphHealthMonitor("
            f"n={self.n}, "
            f"tick={self._tick_counter}, "
            f"alerts={sum(self._alert_counts.values())})"
        )


# ---------------------------------------------------------------------------
# Composite monitor (combines all sub-monitors)
# ---------------------------------------------------------------------------

class CompositeHealthMonitor:
    """
    Runs multiple health monitors for different graph types
    (e.g., equity graph, rates graph, FX graph) and aggregates alerts.
    """

    def __init__(self, monitors: Dict[str, GraphHealthMonitor]) -> None:
        self.monitors = monitors

    # ------------------------------------------------------------------
    def check_all(
        self,
        edge_index_dict: Dict[str, torch.Tensor],
        edge_weight_dict: Dict[str, torch.Tensor],
        tick_id: int,
    ) -> Dict[str, HealthCheckResult]:
        """Run health checks on all registered graphs."""
        results: Dict[str, HealthCheckResult] = {}
        for name, monitor in self.monitors.items():
            if name in edge_index_dict and name in edge_weight_dict:
                results[name] = monitor.check(
                    edge_index_dict[name], edge_weight_dict[name], tick_id
                )
        return results

    # ------------------------------------------------------------------
    def any_crisis(self, results: Dict[str, HealthCheckResult]) -> bool:
        return any(r.has_critical for r in results.values())

    # ------------------------------------------------------------------
    def aggregate_metrics(
        self, results: Dict[str, HealthCheckResult]
    ) -> Dict[str, Any]:
        """Return aggregated health metrics across all graphs."""
        if not results:
            return {}
        total_alerts = sum(len(r.alerts) for r in results.values())
        min_fiedler = min(r.fiedler_estimate for r in results.values())
        mean_density = sum(r.density for r in results.values()) / len(results)
        return {
            "total_alerts": total_alerts,
            "min_fiedler": min_fiedler,
            "mean_density": mean_density,
            "any_crisis": self.any_crisis(results),
            "n_graphs": len(results),
        }

    # ------------------------------------------------------------------
    def add_monitor(self, name: str, monitor: GraphHealthMonitor) -> None:
        self.monitors[name] = monitor

    # ------------------------------------------------------------------
    def remove_monitor(self, name: str) -> None:
        self.monitors.pop(name, None)


# ---------------------------------------------------------------------------
# Alert rate limiter
# ---------------------------------------------------------------------------

class AlertRateLimiter:
    """
    Prevents alert spam by rate-limiting alerts of the same type.
    Each alert type is allowed at most max_per_window alerts within
    the time window.
    """

    def __init__(self, max_per_window: int = 5, window_s: float = 60.0) -> None:
        self.max_per_window = max_per_window
        self.window_s = window_s
        self._timestamps: Dict[AlertType, List[float]] = {}

    # ------------------------------------------------------------------
    def should_emit(self, alert_type: AlertType) -> bool:
        """Return True if this alert should be emitted (not rate-limited)."""
        now = time.monotonic()
        if alert_type not in self._timestamps:
            self._timestamps[alert_type] = []

        # Prune old timestamps
        window_start = now - self.window_s
        self._timestamps[alert_type] = [
            t for t in self._timestamps[alert_type] if t > window_start
        ]

        if len(self._timestamps[alert_type]) < self.max_per_window:
            self._timestamps[alert_type].append(now)
            return True
        return False

    # ------------------------------------------------------------------
    def filter_alerts(self, alerts: List[GraphAlert]) -> List[GraphAlert]:
        """Filter a list of alerts through the rate limiter."""
        return [a for a in alerts if self.should_emit(a.alert_type)]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_health_monitor(
    n_nodes: int = 500,
    n_ticks: int = 500,
    n_edges: int = 5000,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Benchmark GraphHealthMonitor check() latency.

    Returns
    -------
    dict with mean/max/p99 latencies in ms
    """
    import random
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    cfg = GHMConfig(n_nodes=n_nodes, device=str(dev))
    monitor = GraphHealthMonitor(n_nodes=n_nodes, cfg=cfg)

    times: List[float] = []
    for tick in range(n_ticks):
        src = torch.randint(0, n_nodes, (n_edges,))
        dst = torch.randint(0, n_nodes, (n_edges,))
        ei = torch.stack([src, dst], dim=0)
        ew = torch.rand(n_edges)

        t0 = time.perf_counter()
        _ = monitor.check(ei, ew, tick_id=tick)
        times.append((time.perf_counter() - t0) * 1000.0)

    s = sorted(times)
    return {
        "mean_ms": sum(s) / len(s),
        "max_ms": s[-1],
        "p99_ms": s[int(0.99 * len(s))],
        "p95_ms": s[int(0.95 * len(s))],
        "n_ticks": n_ticks,
    }


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "GHMConfig",
    "GraphHealthMonitor",
    "HealthCheckResult",
    "GraphAlert",
    "AlertType",
    "AlertSeverity",
    "UnionFind",
    "DeadNodeTracker",
    "DensityTracker",
    "InlineFiedlerEstimator",
    "AutoRepairEngine",
    "MetricHistory",
    "CompositeHealthMonitor",
    "AlertRateLimiter",
    "benchmark_health_monitor",
]
