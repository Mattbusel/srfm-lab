"""
Market Topology Mapper: build and evolve a live map of the market ecosystem.

Maps the relationships between ALL market participants by analyzing:
  - Correlation clusters (which assets move together)
  - Lead-lag networks (which assets predict others)
  - Liquidity flow paths (how liquidity moves between assets)
  - Volatility contagion (how vol spills from one asset to another)
  - Sentiment contagion (how narratives spread)

The topology changes over time. Detecting topology changes BEFORE price changes
is a source of alpha: if the correlation structure is shifting, it means
institutional flows are repositioning before the price reflects it.

This module feeds directly into the Portfolio Brain for:
  - Diversification: ensure positions span different topology clusters
  - Contagion risk: reduce exposure when topology shows stress spreading
  - Alpha: trade assets that lead in the current topology structure
"""

from __future__ import annotations
import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TopologyNode:
    """An asset in the market topology."""
    symbol: str
    cluster_id: int = -1
    centrality: float = 0.0       # how connected (0=peripheral, 1=hub)
    leadership_score: float = 0.0  # how much it leads other assets
    vulnerability: float = 0.0     # how susceptible to contagion
    current_regime: str = "normal"


@dataclass
class TopologyEdge:
    """Relationship between two assets."""
    source: str
    target: str
    correlation: float
    lead_lag: int                  # positive = source leads target by N bars
    granger_strength: float       # how predictive is source of target?
    vol_transmission: float       # how much vol spills from source to target


@dataclass
class TopologySnapshot:
    """Complete market topology at a point in time."""
    timestamp: float
    n_nodes: int
    n_edges: int
    n_clusters: int
    nodes: List[TopologyNode]
    edges: List[TopologyEdge]
    avg_correlation: float
    max_eigenvalue_ratio: float   # concentration of risk
    topology_change_score: float  # how much topology changed from last snapshot
    stress_indicator: float       # 0=healthy, 1=systemic stress


class MarketTopologyMapper:
    """
    Build and evolve a real-time market topology.

    The topology captures WHO is connected to WHOM, HOW STRONGLY,
    and in WHICH DIRECTION. Changes in topology precede price moves.
    """

    def __init__(self, symbols: List[str], correlation_window: int = 63,
                  correlation_threshold: float = 0.3):
        self.symbols = symbols
        self.window = correlation_window
        self.threshold = correlation_threshold
        self.n = len(symbols)

        self._returns_buffer: Dict[str, deque] = {
            s: deque(maxlen=correlation_window * 2) for s in symbols
        }
        self._topology_history: List[TopologySnapshot] = []

    def update(self, returns: Dict[str, float]) -> None:
        """Update with new bar returns for all symbols."""
        for sym, ret in returns.items():
            if sym in self._returns_buffer:
                self._returns_buffer[sym].append(ret)

    def compute_topology(self) -> TopologySnapshot:
        """Compute current market topology."""
        # Build return matrix
        symbols_with_data = []
        return_matrix = []

        for sym in self.symbols:
            rets = list(self._returns_buffer[sym])
            if len(rets) >= self.window:
                symbols_with_data.append(sym)
                return_matrix.append(rets[-self.window:])

        if len(symbols_with_data) < 3:
            return TopologySnapshot(
                timestamp=0, n_nodes=0, n_edges=0, n_clusters=0,
                nodes=[], edges=[], avg_correlation=0, max_eigenvalue_ratio=0,
                topology_change_score=0, stress_indicator=0,
            )

        R = np.array(return_matrix)  # (n_assets, window)
        n = len(symbols_with_data)

        # Correlation matrix
        corr = np.corrcoef(R)
        np.fill_diagonal(corr, 0)

        # Edges: significant correlations
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) > self.threshold:
                    # Lead-lag: cross-correlation at lag 1
                    if len(R[i]) > 2 and len(R[j]) > 2:
                        lead_corr = float(np.corrcoef(R[i][:-1], R[j][1:])[0, 1])
                        lag_corr = float(np.corrcoef(R[i][1:], R[j][:-1])[0, 1])
                        if lead_corr > lag_corr + 0.05:
                            lead_lag = 1  # i leads j
                            granger = abs(lead_corr - lag_corr)
                        elif lag_corr > lead_corr + 0.05:
                            lead_lag = -1  # j leads i
                            granger = abs(lag_corr - lead_corr)
                        else:
                            lead_lag = 0
                            granger = 0.0
                    else:
                        lead_lag = 0
                        granger = 0.0

                    # Vol transmission
                    vol_i = float(np.std(R[i][-21:]))
                    vol_j = float(np.std(R[j][-21:]))
                    vol_trans = abs(vol_i - vol_j) / max(vol_i + vol_j, 1e-10)

                    edges.append(TopologyEdge(
                        source=symbols_with_data[i],
                        target=symbols_with_data[j],
                        correlation=float(corr[i, j]),
                        lead_lag=lead_lag,
                        granger_strength=granger,
                        vol_transmission=vol_trans,
                    ))

        # Nodes: compute centrality and leadership
        nodes = []
        adj = (np.abs(corr) > self.threshold).astype(float)
        degree = adj.sum(axis=1)

        # Eigenvector centrality (simplified: use degree normalized)
        centrality = degree / max(degree.max(), 1)

        # Leadership: how many assets does each one lead?
        leadership = np.zeros(n)
        for e in edges:
            if e.lead_lag > 0:
                idx = symbols_with_data.index(e.source)
                leadership[idx] += e.granger_strength
            elif e.lead_lag < 0:
                idx = symbols_with_data.index(e.target)
                leadership[idx] += e.granger_strength

        if leadership.max() > 0:
            leadership /= leadership.max()

        # Simple clustering: spectral clustering from correlation
        clusters = self._cluster(corr, n_clusters=min(5, n // 2 + 1))

        for i, sym in enumerate(symbols_with_data):
            nodes.append(TopologyNode(
                symbol=sym,
                cluster_id=int(clusters[i]),
                centrality=float(centrality[i]),
                leadership_score=float(leadership[i]),
                vulnerability=float(centrality[i] * 0.5 + degree[i] / max(n, 1) * 0.5),
            ))

        # Aggregate metrics
        upper = corr[np.triu_indices(n, k=1)]
        avg_corr = float(upper.mean()) if len(upper) > 0 else 0

        # Max eigenvalue ratio (risk concentration)
        try:
            eigvals = np.linalg.eigvalsh(corr + np.eye(n))
            eigvals = eigvals[eigvals > 0]
            max_eig_ratio = float(eigvals[-1] / eigvals.sum()) if len(eigvals) > 0 else 0
        except:
            max_eig_ratio = 0

        # Topology change from last snapshot
        change = 0.0
        if self._topology_history:
            prev = self._topology_history[-1]
            change = abs(avg_corr - prev.avg_correlation) + abs(max_eig_ratio - prev.max_eigenvalue_ratio)

        # Stress indicator: high correlation + high eigenvalue concentration = systemic stress
        stress = float(min(1.0, avg_corr * 2 + max(max_eig_ratio - 0.3, 0) * 3))

        snapshot = TopologySnapshot(
            timestamp=float(len(self._topology_history)),
            n_nodes=n,
            n_edges=len(edges),
            n_clusters=len(set(clusters)),
            nodes=nodes,
            edges=edges,
            avg_correlation=avg_corr,
            max_eigenvalue_ratio=max_eig_ratio,
            topology_change_score=change,
            stress_indicator=stress,
        )

        self._topology_history.append(snapshot)
        return snapshot

    def _cluster(self, corr: np.ndarray, n_clusters: int) -> np.ndarray:
        """Simple spectral clustering."""
        n = corr.shape[0]
        if n <= n_clusters:
            return np.arange(n)

        try:
            # Use eigenvectors of correlation for clustering
            eigvals, eigvecs = np.linalg.eigh(corr + np.eye(n))
            features = eigvecs[:, -n_clusters:]

            # K-means on eigenvectors
            rng = np.random.default_rng(42)
            centroids = features[rng.choice(n, n_clusters, replace=False)]

            for _ in range(20):
                dists = np.array([[np.linalg.norm(f - c) for c in centroids] for f in features])
                labels = dists.argmin(axis=1)
                for k in range(n_clusters):
                    mask = labels == k
                    if mask.sum() > 0:
                        centroids[k] = features[mask].mean(axis=0)

            return labels
        except:
            return np.zeros(n, dtype=int)

    def get_leaders(self) -> List[Dict]:
        """Get assets that are currently leading the market."""
        if not self._topology_history:
            return []
        latest = self._topology_history[-1]
        leaders = sorted(latest.nodes, key=lambda n: n.leadership_score, reverse=True)
        return [
            {"symbol": n.symbol, "leadership": n.leadership_score, "centrality": n.centrality,
             "cluster": n.cluster_id}
            for n in leaders[:5]
        ]

    def get_diversification_map(self) -> Dict[int, List[str]]:
        """Map clusters for portfolio diversification."""
        if not self._topology_history:
            return {}
        latest = self._topology_history[-1]
        clusters = defaultdict(list)
        for node in latest.nodes:
            clusters[node.cluster_id].append(node.symbol)
        return dict(clusters)

    def detect_topology_shift(self, threshold: float = 0.1) -> Optional[Dict]:
        """Detect when the market topology is changing."""
        if len(self._topology_history) < 5:
            return None

        recent_changes = [s.topology_change_score for s in self._topology_history[-5:]]
        avg_change = float(np.mean(recent_changes))

        if avg_change > threshold:
            return {
                "detected": True,
                "change_score": avg_change,
                "implication": "Market structure is reorganizing. Correlations are shifting.",
                "action": "Review portfolio diversification. Update cluster assignments.",
            }
        return None
