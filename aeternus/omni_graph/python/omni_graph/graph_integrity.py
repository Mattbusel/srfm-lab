"""
graph_integrity.py
==================
Graph health monitoring for dynamic financial networks.

Implements:
  - Disconnected component detection
  - Dead node identification
  - Edge weight degeneration alerts
  - Graph density tracking
  - Spectral gap monitoring
  - Fiedler value tracking as liquidity proxy
  - Graph anomaly scoring
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch import Tensor

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ---------------------------------------------------------------------------
# Health check result container
# ---------------------------------------------------------------------------

@dataclass
class GraphHealthReport:
    """
    Container for graph health check results.
    """
    timestamp: int = 0
    num_nodes: int = 0
    num_edges: int = 0
    density: float = 0.0
    n_components: int = 1
    largest_component_size: int = 0
    isolated_nodes: List[int] = field(default_factory=list)
    dead_nodes: List[int] = field(default_factory=list)
    degenerate_edges: List[Tuple[int, int]] = field(default_factory=list)
    fiedler_value: float = 0.0
    spectral_gap: float = 0.0
    anomaly_score: float = 0.0
    alerts: List[str] = field(default_factory=list)
    is_healthy: bool = True

    def summary(self) -> str:
        status = "HEALTHY" if self.is_healthy else "DEGRADED"
        return (
            f"[t={self.timestamp}] GraphHealth({status}): "
            f"N={self.num_nodes}, E={self.num_edges}, density={self.density:.3f}, "
            f"components={self.n_components}, fiedler={self.fiedler_value:.4f}, "
            f"anomaly={self.anomaly_score:.3f}, alerts={len(self.alerts)}"
        )


# ---------------------------------------------------------------------------
# Union-Find for component detection
# ---------------------------------------------------------------------------

class UnionFind:
    """Efficient Union-Find (disjoint sets) for connectivity analysis."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.n_components = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_components -= 1
        return True

    def component_sizes(self) -> Dict[int, int]:
        roots: Dict[int, int] = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            roots[r] = roots.get(r, 0) + 1
        return roots

    def get_components(self) -> List[List[int]]:
        comp: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            comp[self.find(i)].append(i)
        return list(comp.values())


# ---------------------------------------------------------------------------
# Component detector
# ---------------------------------------------------------------------------

class DisconnectedComponentDetector:
    """
    Detect and analyse disconnected components in a financial graph.

    Raises alerts when:
      - Number of components exceeds threshold
      - Largest component shrinks below size threshold
      - Isolated nodes appear (assets disconnecting from the market)
    """

    def __init__(
        self,
        max_components: int = 3,
        min_lcc_fraction: float = 0.7,
        alert_on_isolated: bool = True,
    ):
        self.max_components = max_components
        self.min_lcc_fraction = min_lcc_fraction
        self.alert_on_isolated = alert_on_isolated

    def detect(
        self,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tuple[int, List[List[int]], List[int]]:
        """
        Parameters
        ----------
        edge_index : (2, E)
        num_nodes  : N

        Returns
        -------
        n_components : int
        components   : list of lists (each a connected component)
        isolated     : list of isolated node IDs
        """
        uf = UnionFind(num_nodes)
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            if 0 <= i < num_nodes and 0 <= j < num_nodes and i != j:
                uf.union(i, j)

        components = uf.get_components()
        isolated = [c[0] for c in components if len(c) == 1]

        return uf.n_components, components, isolated

    def check(
        self,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Dict[str, Any]:
        n_comp, components, isolated = self.detect(edge_index, num_nodes)
        lcc_size = max((len(c) for c in components), default=0)
        lcc_fraction = lcc_size / (num_nodes + 1e-8)

        alerts = []
        if n_comp > self.max_components:
            alerts.append(f"HIGH_FRAGMENTATION: {n_comp} components (max={self.max_components})")
        if lcc_fraction < self.min_lcc_fraction:
            alerts.append(f"LCC_TOO_SMALL: {lcc_fraction:.2%} < {self.min_lcc_fraction:.2%}")
        if self.alert_on_isolated and isolated:
            alerts.append(f"ISOLATED_NODES: {len(isolated)} nodes isolated ({isolated[:5]}...)")

        return {
            "n_components": n_comp,
            "lcc_size": lcc_size,
            "lcc_fraction": lcc_fraction,
            "isolated_nodes": isolated,
            "components": components,
            "alerts": alerts,
        }


# ---------------------------------------------------------------------------
# Dead node detector
# ---------------------------------------------------------------------------

class DeadNodeDetector:
    """
    Identify 'dead' nodes — assets that have become effectively inactive.

    An asset is dead if:
      1. It has degree 0 (isolated)
      2. Its edge weights have all fallen below a threshold
      3. It has had no edge updates for `staleness_threshold` time steps
    """

    def __init__(
        self,
        weight_threshold: float = 0.01,
        staleness_threshold: int = 10,
    ):
        self.weight_threshold = weight_threshold
        self.staleness_threshold = staleness_threshold
        self._last_active: Dict[int, int] = {}
        self._t: int = 0

    def update(
        self,
        edge_index: Tensor,
        edge_weights: Tensor,
        num_nodes: int,
    ) -> List[int]:
        """
        Update activity tracking and return list of dead nodes.

        Parameters
        ----------
        edge_index  : (2, E)
        edge_weights : (E,)
        num_nodes   : N

        Returns
        -------
        dead_nodes : list of node IDs
        """
        self._t += 1
        active_nodes: Set[int] = set()

        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            w = float(edge_weights[k])
            if abs(w) >= self.weight_threshold:
                active_nodes.add(i)
                active_nodes.add(j)

        for node in active_nodes:
            self._last_active[node] = self._t

        dead_nodes = []
        for node in range(num_nodes):
            last_t = self._last_active.get(node, 0)
            if self._t - last_t > self.staleness_threshold:
                dead_nodes.append(node)

        return dead_nodes

    def reset_node(self, node: int) -> None:
        """Mark a node as active (e.g., on re-listing)."""
        self._last_active[node] = self._t

    def staleness_scores(self, num_nodes: int) -> np.ndarray:
        """Return per-node staleness (steps since last active)."""
        scores = np.zeros(num_nodes)
        for node in range(num_nodes):
            last_t = self._last_active.get(node, 0)
            scores[node] = self._t - last_t
        return scores


# ---------------------------------------------------------------------------
# Edge weight degeneration monitor
# ---------------------------------------------------------------------------

class EdgeWeightDegenerationMonitor:
    """
    Monitor for edge weight degeneration patterns.

    Detects:
      1. Weight collapse: edges approaching zero across the board
      2. Weight explosion: unnormally large weights
      3. Sign flip storms: many edges changing correlation sign rapidly
      4. Uniformity collapse: all weights becoming similar (loss of structure)
    """

    def __init__(
        self,
        collapse_threshold: float = 0.02,
        explosion_threshold: float = 0.99,
        uniformity_threshold: float = 0.05,
        window: int = 10,
    ):
        self.collapse_threshold = collapse_threshold
        self.explosion_threshold = explosion_threshold
        self.uniformity_threshold = uniformity_threshold
        self.window = window

        self._weight_stats_history: deque = deque(maxlen=window)
        self._sign_history: deque = deque(maxlen=window)

    def update(
        self,
        edge_weights: Tensor,
    ) -> Dict[str, Any]:
        """
        Monitor edge weights and return degeneration report.

        Parameters
        ----------
        edge_weights : (E,) or (E, F) — uses first column if 2D

        Returns
        -------
        dict with degeneration flags and alerts
        """
        if edge_weights.dim() > 1:
            w = edge_weights[:, 0]
        else:
            w = edge_weights

        if w.shape[0] == 0:
            return {"alerts": ["NO_EDGES"], "is_degenerate": True}

        w_np = w.detach().cpu().numpy()
        mean_w = float(np.mean(w_np))
        std_w = float(np.std(w_np))
        abs_mean = float(np.mean(np.abs(w_np)))
        max_abs = float(np.max(np.abs(w_np)))
        pos_fraction = float(np.mean(w_np > 0))

        stats = {
            "mean": mean_w,
            "std": std_w,
            "abs_mean": abs_mean,
            "max_abs": max_abs,
            "pos_fraction": pos_fraction,
        }
        self._weight_stats_history.append(stats)

        signs = np.sign(w_np)
        self._sign_history.append(signs.copy())

        alerts = []
        is_degenerate = False

        # Weight collapse
        if abs_mean < self.collapse_threshold:
            alerts.append(f"WEIGHT_COLLAPSE: mean|w|={abs_mean:.4f} < {self.collapse_threshold}")
            is_degenerate = True

        # Weight explosion
        if max_abs > self.explosion_threshold:
            alerts.append(f"WEIGHT_EXPLOSION: max|w|={max_abs:.4f} > {self.explosion_threshold}")

        # Uniformity collapse (std very low relative to mean)
        if abs_mean > 0.01 and std_w / (abs_mean + 1e-8) < self.uniformity_threshold:
            alerts.append(f"UNIFORMITY_COLLAPSE: std/mean={std_w/(abs_mean+1e-8):.4f}")

        # Sign flip storm: compare current signs to previous
        if len(self._sign_history) >= 2:
            prev_signs = self._sign_history[-2]
            curr_signs = self._sign_history[-1]
            min_len = min(len(prev_signs), len(curr_signs))
            if min_len > 0:
                flip_rate = float(np.mean(prev_signs[:min_len] != curr_signs[:min_len]))
                if flip_rate > 0.4:
                    alerts.append(f"SIGN_FLIP_STORM: {flip_rate:.1%} of edges flipped sign")

        return {
            "stats": stats,
            "alerts": alerts,
            "is_degenerate": is_degenerate,
        }


# ---------------------------------------------------------------------------
# Graph density tracker
# ---------------------------------------------------------------------------

class GraphDensityTracker:
    """
    Track graph density over time and detect density regime changes.

    Density = 2E / (N*(N-1)) for undirected graphs.

    Monitors:
      - Current density vs rolling average
      - Density trend (increasing/decreasing connectivity)
      - Density shock detection (sudden large change)
    """

    def __init__(
        self,
        window: int = 20,
        shock_threshold: float = 0.3,
    ):
        self.window = window
        self.shock_threshold = shock_threshold
        self._density_history: deque = deque(maxlen=window)
        self._t: int = 0

    def update(
        self,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Dict[str, Any]:
        """Update density tracker and return analysis."""
        self._t += 1
        E = edge_index.shape[1]
        max_edges = num_nodes * (num_nodes - 1)  # directed count / 2 for undirected
        density = E / max(max_edges, 1) * 2  # undirected density

        self._density_history.append(density)
        hist = list(self._density_history)

        result: Dict[str, Any] = {
            "density": density,
            "t": self._t,
            "n_edges": E,
            "alerts": [],
        }

        if len(hist) >= 5:
            rolling_mean = float(np.mean(hist[:-1]))
            rolling_std = float(np.std(hist[:-1])) + 1e-8
            z_score = (density - rolling_mean) / rolling_std
            result["rolling_mean_density"] = rolling_mean
            result["density_z_score"] = z_score

            if abs(z_score) > self.shock_threshold * 3.0:
                direction = "INCREASE" if density > rolling_mean else "DECREASE"
                result["alerts"].append(
                    f"DENSITY_SHOCK_{direction}: z={z_score:.2f}, density={density:.4f}"
                )

            # Trend: last 5 vs first 5
            if len(hist) >= 10:
                early = np.mean(hist[:5])
                late = np.mean(hist[-5:])
                trend = (late - early) / (abs(early) + 1e-8)
                result["density_trend"] = trend

        return result

    @property
    def current_density(self) -> float:
        if self._density_history:
            return self._density_history[-1]
        return 0.0

    def density_series(self) -> np.ndarray:
        return np.array(list(self._density_history))


# ---------------------------------------------------------------------------
# Spectral gap & Fiedler value tracker
# ---------------------------------------------------------------------------

class SpectralGapMonitor:
    """
    Monitor spectral gap and Fiedler value (λ₂) of the graph Laplacian.

    The Fiedler value measures algebraic connectivity — a proxy for
    market liquidity and integration. A falling Fiedler value indicates
    fragmentation or liquidity deterioration.

    Spectral gap = λ₂ - λ₁ (where λ₁ = 0 for connected graphs).
    """

    def __init__(
        self,
        window: int = 30,
        fiedler_alert_threshold: float = 0.01,
        fiedler_drop_threshold: float = 0.3,
        max_nodes_for_exact: int = 300,
    ):
        self.window = window
        self.fiedler_alert_threshold = fiedler_alert_threshold
        self.fiedler_drop_threshold = fiedler_drop_threshold
        self.max_nodes_for_exact = max_nodes_for_exact

        self._fiedler_history: deque = deque(maxlen=window)
        self._spectral_gap_history: deque = deque(maxlen=window)
        self._t: int = 0

    def compute_fiedler(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weights: Optional[Tensor] = None,
    ) -> Tuple[float, float]:
        """
        Compute Fiedler value and spectral gap.

        Returns (fiedler, spectral_gap).
        """
        n = min(num_nodes, self.max_nodes_for_exact)
        A = np.zeros((n, n), dtype=np.float32)

        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            if 0 <= i < n and 0 <= j < n and i != j:
                w = float(edge_weights[k]) if edge_weights is not None else 1.0
                A[i, j] = abs(w)
                A[j, i] = abs(w)

        deg = A.sum(axis=1)
        D = np.diag(deg)
        L = D - A

        try:
            eigvals = np.linalg.eigvalsh(L)
            eigvals_sorted = np.sort(eigvals)
            fiedler = float(max(eigvals_sorted[1], 0.0)) if n > 1 else 0.0
            gap = float(max(eigvals_sorted[1] - eigvals_sorted[0], 0.0)) if n > 1 else 0.0
        except np.linalg.LinAlgError:
            fiedler, gap = 0.0, 0.0

        return fiedler, gap

    def update(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weights: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Update monitor and return analysis."""
        self._t += 1
        fiedler, gap = self.compute_fiedler(edge_index, num_nodes, edge_weights)

        self._fiedler_history.append(fiedler)
        self._spectral_gap_history.append(gap)

        result: Dict[str, Any] = {
            "fiedler_value": fiedler,
            "spectral_gap": gap,
            "t": self._t,
            "alerts": [],
            "liquidity_signal": self._fiedler_to_liquidity(fiedler),
        }

        # Alert: Fiedler below absolute threshold
        if fiedler < self.fiedler_alert_threshold:
            result["alerts"].append(
                f"LOW_FIEDLER: λ₂={fiedler:.6f} < {self.fiedler_alert_threshold}"
            )

        # Alert: Significant drop from recent average
        hist = list(self._fiedler_history)
        if len(hist) >= 5:
            prev_mean = float(np.mean(hist[:-1]))
            if prev_mean > 0 and (prev_mean - fiedler) / prev_mean > self.fiedler_drop_threshold:
                result["alerts"].append(
                    f"FIEDLER_DROP: {fiedler:.4f} is {(prev_mean-fiedler)/prev_mean:.1%} below average"
                )

        return result

    def _fiedler_to_liquidity(self, fiedler: float) -> float:
        """
        Convert Fiedler value to a normalised liquidity score in [0, 1].
        Uses logistic transform centred at 0.1.
        """
        return float(1.0 / (1.0 + math.exp(-10.0 * (fiedler - 0.1))))

    def fiedler_series(self) -> np.ndarray:
        return np.array(list(self._fiedler_history))

    def spectral_gap_series(self) -> np.ndarray:
        return np.array(list(self._spectral_gap_history))


# ---------------------------------------------------------------------------
# Graph anomaly scorer
# ---------------------------------------------------------------------------

class GraphAnomalyScorer:
    """
    Compute a composite anomaly score for a graph snapshot.

    Combines multiple signals:
      1. Density deviation from rolling mean
      2. Fiedler value drop
      3. Number of isolated nodes
      4. Edge weight degeneration
      5. Component fragmentation

    Score is in [0, 1] where 1.0 = highly anomalous.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        window: int = 20,
    ):
        self.weights = weights or {
            "density": 0.20,
            "fiedler": 0.30,
            "isolated": 0.20,
            "weight_degen": 0.15,
            "fragmentation": 0.15,
        }
        self.window = window

        self._density_history: deque = deque(maxlen=window)
        self._fiedler_history: deque = deque(maxlen=window)
        self._component_history: deque = deque(maxlen=window)

    def score(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weights: Optional[Tensor] = None,
        fiedler: Optional[float] = None,
    ) -> float:
        """
        Compute anomaly score for current graph snapshot.

        Parameters
        ----------
        edge_index  : (2, E)
        num_nodes   : N
        edge_weights : (E,) optional
        fiedler     : pre-computed Fiedler value (optional)

        Returns
        -------
        anomaly_score : float in [0, 1]
        """
        E = edge_index.shape[1]
        density = 2 * E / max(num_nodes * (num_nodes - 1), 1)
        self._density_history.append(density)

        # Component analysis
        uf = UnionFind(num_nodes)
        for k in range(E):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            if 0 <= i < num_nodes and 0 <= j < num_nodes and i != j:
                uf.union(i, j)
        n_comp = uf.n_components
        self._component_history.append(n_comp)

        components = uf.get_components()
        n_isolated = sum(1 for c in components if len(c) == 1)

        # Fiedler
        if fiedler is None:
            # Quick proxy: 1/n_components
            fiedler = 1.0 / n_comp
        self._fiedler_history.append(fiedler)

        # Sub-scores
        density_score = self._density_anomaly_score(density)
        fiedler_score = self._fiedler_anomaly_score(fiedler)
        isolated_score = min(n_isolated / max(num_nodes, 1), 1.0)
        weight_score = self._weight_degen_score(edge_weights)
        frag_score = min((n_comp - 1) / max(num_nodes, 1), 1.0)

        total = (
            self.weights["density"] * density_score
            + self.weights["fiedler"] * fiedler_score
            + self.weights["isolated"] * isolated_score
            + self.weights["weight_degen"] * weight_score
            + self.weights["fragmentation"] * frag_score
        )

        return float(np.clip(total, 0.0, 1.0))

    def _density_anomaly_score(self, density: float) -> float:
        hist = list(self._density_history)
        if len(hist) < 5:
            return 0.0
        mu = np.mean(hist[:-1])
        sigma = np.std(hist[:-1]) + 1e-8
        z = abs((density - mu) / sigma)
        return float(np.clip(z / 3.0, 0.0, 1.0))

    def _fiedler_anomaly_score(self, fiedler: float) -> float:
        hist = list(self._fiedler_history)
        if len(hist) < 5:
            return 0.0
        mu = np.mean(hist[:-1])
        sigma = np.std(hist[:-1]) + 1e-8
        # Fiedler dropping is bad → one-sided z
        z = (mu - fiedler) / sigma
        return float(np.clip(z / 3.0, 0.0, 1.0))

    def _weight_degen_score(self, edge_weights: Optional[Tensor]) -> float:
        if edge_weights is None or edge_weights.shape[0] == 0:
            return 0.5  # unknown → moderate concern
        w = edge_weights.detach().cpu().numpy()
        abs_mean = float(np.mean(np.abs(w)))
        if abs_mean < 0.02:
            return 1.0
        elif abs_mean < 0.1:
            return (0.1 - abs_mean) / 0.08
        return 0.0


# ---------------------------------------------------------------------------
# Centralised graph health monitor
# ---------------------------------------------------------------------------

class GraphHealthMonitor:
    """
    Centralised monitor combining all health checks into a single API.

    Usage:
        monitor = GraphHealthMonitor(num_nodes=100)
        report = monitor.check(edge_index, edge_weights, t=42)
        if not report.is_healthy:
            for alert in report.alerts:
                print(alert)
    """

    def __init__(
        self,
        num_nodes: int,
        max_components: int = 3,
        min_lcc_fraction: float = 0.7,
        weight_collapse_threshold: float = 0.02,
        fiedler_alert_threshold: float = 0.01,
        anomaly_alert_threshold: float = 0.6,
        window: int = 20,
    ):
        self.num_nodes = num_nodes
        self.anomaly_alert_threshold = anomaly_alert_threshold

        self.component_detector = DisconnectedComponentDetector(
            max_components=max_components,
            min_lcc_fraction=min_lcc_fraction,
        )
        self.dead_node_detector = DeadNodeDetector(
            weight_threshold=weight_collapse_threshold,
        )
        self.weight_monitor = EdgeWeightDegenerationMonitor(
            collapse_threshold=weight_collapse_threshold,
        )
        self.density_tracker = GraphDensityTracker(window=window)
        self.spectral_monitor = SpectralGapMonitor(
            window=window,
            fiedler_alert_threshold=fiedler_alert_threshold,
        )
        self.anomaly_scorer = GraphAnomalyScorer(window=window)

        self._report_history: List[GraphHealthReport] = []

    def check(
        self,
        edge_index: Tensor,
        edge_weights: Optional[Tensor] = None,
        t: int = 0,
        compute_spectral: bool = True,
    ) -> GraphHealthReport:
        """
        Run all health checks and return a consolidated report.

        Parameters
        ----------
        edge_index   : (2, E)
        edge_weights : (E,) or (E, F)
        t            : current time step
        compute_spectral : whether to compute Fiedler/spectral metrics

        Returns
        -------
        GraphHealthReport
        """
        num_nodes = self.num_nodes
        E = edge_index.shape[1]

        all_alerts: List[str] = []

        # 1. Connectivity
        comp_result = self.component_detector.check(edge_index, num_nodes)
        all_alerts.extend(comp_result["alerts"])

        # 2. Dead nodes
        if edge_weights is not None:
            w_1d = edge_weights[:, 0] if edge_weights.dim() > 1 else edge_weights
        else:
            w_1d = torch.ones(E)

        dead_nodes = self.dead_node_detector.update(edge_index, w_1d, num_nodes)

        # 3. Weight degeneration
        if edge_weights is not None:
            weight_result = self.weight_monitor.update(edge_weights)
            all_alerts.extend(weight_result["alerts"])
        else:
            weight_result = {"is_degenerate": False}

        # 4. Density
        density_result = self.density_tracker.update(edge_index, num_nodes)
        all_alerts.extend(density_result["alerts"])
        density = density_result["density"]

        # 5. Spectral
        fiedler = 0.0
        spectral_gap = 0.0
        if compute_spectral:
            spectral_result = self.spectral_monitor.update(
                edge_index, num_nodes,
                w_1d if edge_weights is not None else None,
            )
            fiedler = spectral_result["fiedler_value"]
            spectral_gap = spectral_result["spectral_gap"]
            all_alerts.extend(spectral_result["alerts"])
        else:
            fiedler = 1.0 / comp_result["n_components"]

        # 6. Anomaly score
        anomaly_score = self.anomaly_scorer.score(
            edge_index, num_nodes,
            w_1d if edge_weights is not None else None,
            fiedler=fiedler,
        )
        if anomaly_score > self.anomaly_alert_threshold:
            all_alerts.append(f"HIGH_ANOMALY_SCORE: {anomaly_score:.3f}")

        # Identify degenerate edges
        degenerate_edges = []
        if edge_weights is not None:
            for k in range(edge_index.shape[1]):
                i, j = int(edge_index[0, k]), int(edge_index[1, k])
                w = float(w_1d[k]) if k < len(w_1d) else 0.0
                if abs(w) < 0.005:
                    degenerate_edges.append((i, j))

        is_healthy = (
            len(all_alerts) == 0
            or not weight_result.get("is_degenerate", False)
        ) and anomaly_score < self.anomaly_alert_threshold

        report = GraphHealthReport(
            timestamp=t,
            num_nodes=num_nodes,
            num_edges=E,
            density=density,
            n_components=comp_result["n_components"],
            largest_component_size=comp_result["lcc_size"],
            isolated_nodes=comp_result["isolated_nodes"],
            dead_nodes=dead_nodes,
            degenerate_edges=degenerate_edges[:20],  # cap list length
            fiedler_value=fiedler,
            spectral_gap=spectral_gap,
            anomaly_score=anomaly_score,
            alerts=all_alerts,
            is_healthy=is_healthy,
        )
        self._report_history.append(report)
        return report

    def get_report_history(self) -> List[GraphHealthReport]:
        return self._report_history

    def anomaly_series(self) -> np.ndarray:
        return np.array([r.anomaly_score for r in self._report_history])

    def fiedler_series(self) -> np.ndarray:
        return np.array([r.fiedler_value for r in self._report_history])

    def density_series(self) -> np.ndarray:
        return np.array([r.density for r in self._report_history])


# ---------------------------------------------------------------------------
# Node-level health metrics
# ---------------------------------------------------------------------------

def compute_node_health_scores(
    edge_index: Tensor,
    num_nodes: int,
    edge_weights: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute per-node health scores (higher = healthier).

    Metrics combined:
      - Degree normalised
      - Mean edge weight magnitude (local connectivity strength)
      - Clustering coefficient proxy

    Returns
    -------
    scores : (N,) tensor in [0, 1]
    """
    N = num_nodes
    degree = torch.zeros(N)
    weight_sum = torch.zeros(N)

    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if 0 <= i < N and 0 <= j < N:
            degree[i] += 1
            if edge_weights is not None and k < edge_weights.shape[0]:
                w = float(edge_weights[k, 0] if edge_weights.dim() > 1 else edge_weights[k])
                weight_sum[i] += abs(w)

    max_deg = degree.max() + 1e-8
    max_ws = weight_sum.max() + 1e-8

    norm_degree = degree / max_deg
    norm_weight = weight_sum / max_ws

    # Simple average
    scores = 0.6 * norm_degree + 0.4 * norm_weight
    return scores.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "GraphHealthReport",
    "UnionFind",
    "DisconnectedComponentDetector",
    "DeadNodeDetector",
    "EdgeWeightDegenerationMonitor",
    "GraphDensityTracker",
    "SpectralGapMonitor",
    "GraphAnomalyScorer",
    "GraphHealthMonitor",
    "compute_node_health_scores",
]
