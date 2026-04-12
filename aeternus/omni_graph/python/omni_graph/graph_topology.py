"""
graph_topology.py
=================
Graph construction from financial data for the Omni-Graph library.

Supports:
  - Pearson / Spearman / distance correlation graphs
  - Minimum Spanning Tree (MST)
  - Planar Maximally Filtered Graph (PMFG)
  - k-Nearest Neighbour graph
  - Lead-lag graph via cross-correlation
  - Limit Order Book (LOB) graph (price levels as nodes, order flow as edges)
  - Node feature engineering (return, volatility, volume, bid-ask spread)
  - Edge feature engineering (correlation strength, direction, lag)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from torch import Tensor

# Optional PyG imports
try:
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.utils import (
        to_undirected, remove_self_loops, add_self_loops,
        dense_to_sparse, coalesce, to_networkx
    )
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn("torch_geometric not found; some functions will be unavailable.")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GraphBuildConfig:
    """Configuration for graph construction."""
    corr_method: str = "pearson"          # pearson | spearman | distance
    corr_threshold: float = 0.3
    use_absolute_corr: bool = True
    k_neighbours: int = 5
    max_lag: int = 5                       # for lead-lag graph
    pmfg_max_edges: Optional[int] = None  # None → auto (3*(n-2))
    lob_depth: int = 10                   # LOB levels per side
    min_edge_weight: float = 1e-6
    self_loops: bool = False
    directed: bool = False
    device: str = "cpu"


@dataclass
class FinancialGraphData:
    """Container for a constructed financial graph."""
    edge_index: Tensor                    # [2, E]
    edge_attr: Optional[Tensor] = None   # [E, edge_feat_dim]
    node_attr: Optional[Tensor] = None   # [N, node_feat_dim]
    num_nodes: int = 0
    asset_names: List[str] = field(default_factory=list)
    graph_type: str = "unknown"
    metadata: Dict = field(default_factory=dict)

    def to_pyg(self) -> "Data":
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for to_pyg()")
        return Data(
            x=self.node_attr,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self.num_nodes,
        )


# ---------------------------------------------------------------------------
# Correlation utilities
# ---------------------------------------------------------------------------

def pearson_correlation_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Pearson correlation from a (T, N) returns matrix.

    Parameters
    ----------
    returns : ndarray, shape (T, N)

    Returns
    -------
    corr : ndarray, shape (N, N)
    """
    if returns.shape[0] < 3:
        raise ValueError("Need at least 3 time steps for Pearson correlation.")
    corr = np.corrcoef(returns.T)
    # Clip to [-1, 1] for numerical safety
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def spearman_correlation_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Spearman rank correlation from a (T, N) returns matrix.
    """
    n = returns.shape[1]
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = stats.spearmanr(returns[:, i], returns[:, j])
            corr[i, j] = rho
            corr[j, i] = rho
    return np.clip(corr, -1.0, 1.0)


def distance_correlation_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance correlation from a (T, N) returns matrix.
    Distance correlation is always in [0, 1].
    """
    n = returns.shape[1]
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dcor = _distance_correlation(returns[:, i], returns[:, j])
            corr[i, j] = dcor
            corr[j, i] = dcor
    return corr


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute distance correlation between two 1-D arrays."""
    n = len(x)
    if n < 2:
        return 0.0

    def _cent_dist(z: np.ndarray) -> np.ndarray:
        d = np.abs(z[:, None] - z[None, :])
        row_mean = d.mean(axis=1, keepdims=True)
        col_mean = d.mean(axis=0, keepdims=True)
        grand_mean = d.mean()
        return d - row_mean - col_mean + grand_mean

    A = _cent_dist(x)
    B = _cent_dist(y)

    dcov2_xy = (A * B).mean()
    dcov2_xx = (A * A).mean()
    dcov2_yy = (B * B).mean()

    denom = math.sqrt(max(dcov2_xx * dcov2_yy, 0.0))
    if denom < 1e-12:
        return 0.0
    return float(math.sqrt(max(dcov2_xy / denom, 0.0)))


def correlation_to_distance(corr: np.ndarray) -> np.ndarray:
    """
    Convert Pearson/Spearman correlation matrix to Mantegna's distance metric:
        d(i, j) = sqrt(2 * (1 - rho(i, j)))
    """
    dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return dist


# ---------------------------------------------------------------------------
# Correlation graph
# ---------------------------------------------------------------------------

class CorrelationGraphBuilder:
    """
    Build a correlation-based graph from asset return time series.

    Nodes = assets, edges = pairs with |corr| > threshold.
    """

    def __init__(self, config: Optional[GraphBuildConfig] = None):
        self.config = config or GraphBuildConfig()

    def build(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        asset_names: Optional[List[str]] = None,
    ) -> FinancialGraphData:
        """
        Parameters
        ----------
        returns : (T, N) array or DataFrame
        asset_names : list of length N

        Returns
        -------
        FinancialGraphData
        """
        if isinstance(returns, pd.DataFrame):
            asset_names = asset_names or list(returns.columns)
            returns = returns.values.astype(np.float32)
        else:
            returns = returns.astype(np.float32)

        n = returns.shape[1]
        asset_names = asset_names or [f"asset_{i}" for i in range(n)]

        corr = self._compute_corr(returns)

        rows, cols, weights = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                c = corr[i, j]
                val = abs(c) if self.config.use_absolute_corr else c
                if val >= self.config.corr_threshold:
                    rows.append(i)
                    cols.append(j)
                    weights.append(c)

        if not rows:
            # Fallback: keep top-k% of edges
            flat = []
            for i in range(n):
                for j in range(i + 1, n):
                    flat.append((i, j, corr[i, j]))
            flat.sort(key=lambda x: abs(x[2]), reverse=True)
            keep = max(n, len(flat) // 5)
            for i, j, c in flat[:keep]:
                rows.append(i)
                cols.append(j)
                weights.append(c)

        edge_index, edge_attr = self._make_edge_tensors(rows, cols, weights, n)

        node_attr = self._node_features(returns)

        return FinancialGraphData(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            num_nodes=n,
            asset_names=asset_names,
            graph_type=f"correlation_{self.config.corr_method}",
            metadata={"corr_matrix": corr, "threshold": self.config.corr_threshold},
        )

    def _compute_corr(self, returns: np.ndarray) -> np.ndarray:
        method = self.config.corr_method.lower()
        if method == "pearson":
            return pearson_correlation_matrix(returns)
        elif method == "spearman":
            return spearman_correlation_matrix(returns)
        elif method == "distance":
            return distance_correlation_matrix(returns)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

    def _make_edge_tensors(
        self,
        rows: List[int],
        cols: List[int],
        weights: List[float],
        n: int,
    ) -> Tuple[Tensor, Tensor]:
        if not rows:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, 3)
            return edge_index, edge_attr

        src = torch.tensor(rows, dtype=torch.long)
        dst = torch.tensor(cols, dtype=torch.long)
        w = torch.tensor(weights, dtype=torch.float32)

        if not self.config.directed:
            src2 = torch.cat([src, dst])
            dst2 = torch.cat([dst, src])
            w2 = torch.cat([w, w])
            src, dst, w = src2, dst2, w2

        edge_index = torch.stack([src, dst], dim=0)

        # Edge features: [correlation, |correlation|, sign]
        edge_attr = torch.stack([
            w,
            w.abs(),
            w.sign(),
        ], dim=1)

        return edge_index, edge_attr

    def _node_features(self, returns: np.ndarray) -> Tensor:
        """Compute per-node features from return series."""
        return compute_node_features(returns)


# ---------------------------------------------------------------------------
# Minimum Spanning Tree
# ---------------------------------------------------------------------------

class MSTGraphBuilder:
    """
    Build a Minimum Spanning Tree on Mantegna distance metric.

    MST gives the most significant structural connections in the market.
    """

    def __init__(self, config: Optional[GraphBuildConfig] = None):
        self.config = config or GraphBuildConfig()

    def build(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        asset_names: Optional[List[str]] = None,
    ) -> FinancialGraphData:
        if isinstance(returns, pd.DataFrame):
            asset_names = asset_names or list(returns.columns)
            returns = returns.values.astype(np.float32)
        else:
            returns = returns.astype(np.float32)

        n = returns.shape[1]
        asset_names = asset_names or [f"asset_{i}" for i in range(n)]

        corr = pearson_correlation_matrix(returns)
        dist = correlation_to_distance(corr)

        edges = self._kruskal_mst(dist, n)

        rows = [e[0] for e in edges]
        cols = [e[1] for e in edges]
        weights_dist = [dist[e[0], e[1]] for e in edges]
        weights_corr = [corr[e[0], e[1]] for e in edges]

        src = torch.tensor(rows, dtype=torch.long)
        dst = torch.tensor(cols, dtype=torch.long)
        wd = torch.tensor(weights_dist, dtype=torch.float32)
        wc = torch.tensor(weights_corr, dtype=torch.float32)

        if not self.config.directed:
            src = torch.cat([src, dst])
            dst = torch.cat([dst, torch.tensor(rows, dtype=torch.long)])
            wd = torch.cat([wd, wd])
            wc = torch.cat([wc, wc])

        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.stack([wc, wd, wc.abs()], dim=1)

        node_attr = compute_node_features(returns)

        return FinancialGraphData(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            num_nodes=n,
            asset_names=asset_names,
            graph_type="mst",
            metadata={"corr_matrix": corr, "dist_matrix": dist},
        )

    def _kruskal_mst(self, dist: np.ndarray, n: int) -> List[Tuple[int, int]]:
        """Kruskal's algorithm for MST on full distance matrix."""
        # Collect all edges
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((dist[i, j], i, j))
        edge_list.sort()

        # Union-Find
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

        mst_edges = []
        for d, i, j in edge_list:
            if union(i, j):
                mst_edges.append((i, j))
            if len(mst_edges) == n - 1:
                break

        return mst_edges


# ---------------------------------------------------------------------------
# Planar Maximally Filtered Graph (PMFG)
# ---------------------------------------------------------------------------

class PMFGBuilder:
    """
    Build the Planar Maximally Filtered Graph (PMFG) from financial correlations.

    PMFG keeps 3*(N-2) edges (max for planar graph) by greedily adding
    highest-correlation edges that maintain planarity.

    Note: Full planarity testing is O(N) per edge via Boyer-Myrvold; here we use
    a heuristic (genus check via Euler's formula on connected components).
    """

    def __init__(self, config: Optional[GraphBuildConfig] = None):
        self.config = config or GraphBuildConfig()

    def build(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        asset_names: Optional[List[str]] = None,
    ) -> FinancialGraphData:
        if isinstance(returns, pd.DataFrame):
            asset_names = asset_names or list(returns.columns)
            returns = returns.values.astype(np.float32)
        else:
            returns = returns.astype(np.float32)

        n = returns.shape[1]
        asset_names = asset_names or [f"asset_{i}" for i in range(n)]

        corr = pearson_correlation_matrix(returns)
        dist = correlation_to_distance(corr)

        max_edges = self.config.pmfg_max_edges or (3 * (n - 2))
        pmfg_edges = self._build_pmfg(dist, n, max_edges)

        rows = [e[0] for e in pmfg_edges]
        cols = [e[1] for e in pmfg_edges]
        ws = [corr[e[0], e[1]] for e in pmfg_edges]

        src = torch.tensor(rows, dtype=torch.long)
        dst = torch.tensor(cols, dtype=torch.long)
        wt = torch.tensor(ws, dtype=torch.float32)

        if not self.config.directed:
            src = torch.cat([src, torch.tensor(cols, dtype=torch.long)])
            dst = torch.cat([dst, torch.tensor(rows, dtype=torch.long)])
            wt = torch.cat([wt, wt])

        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.stack([wt, wt.abs(), wt.sign()], dim=1)

        node_attr = compute_node_features(returns)

        return FinancialGraphData(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            num_nodes=n,
            asset_names=asset_names,
            graph_type="pmfg",
            metadata={"max_edges": max_edges, "actual_edges": len(pmfg_edges)},
        )

    def _build_pmfg(
        self, dist: np.ndarray, n: int, max_edges: int
    ) -> List[Tuple[int, int]]:
        """Greedy PMFG construction using planarity heuristic."""
        if not HAS_NX:
            # Fallback: just build MST-like structure limited to max_edges
            edge_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    edge_list.append((dist[i, j], i, j))
            edge_list.sort()
            return [(e[1], e[2]) for e in edge_list[:max_edges]]

        G = nx.Graph()
        G.add_nodes_from(range(n))

        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((dist[i, j], i, j))
        edge_list.sort()

        pmfg_edges = []
        for d, i, j in edge_list:
            if len(pmfg_edges) >= max_edges:
                break
            G.add_edge(i, j)
            if nx.check_planarity(G)[0]:
                pmfg_edges.append((i, j))
            else:
                G.remove_edge(i, j)

        return pmfg_edges


# ---------------------------------------------------------------------------
# k-Nearest Neighbour graph
# ---------------------------------------------------------------------------

class KNNGraphBuilder:
    """
    Build a k-Nearest Neighbour graph in correlation distance space.

    Each asset is connected to its k most correlated peers.
    """

    def __init__(self, config: Optional[GraphBuildConfig] = None):
        self.config = config or GraphBuildConfig()

    def build(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        asset_names: Optional[List[str]] = None,
    ) -> FinancialGraphData:
        if isinstance(returns, pd.DataFrame):
            asset_names = asset_names or list(returns.columns)
            returns = returns.values.astype(np.float32)
        else:
            returns = returns.astype(np.float32)

        n = returns.shape[1]
        k = min(self.config.k_neighbours, n - 1)
        asset_names = asset_names or [f"asset_{i}" for i in range(n)]

        corr = pearson_correlation_matrix(returns)
        dist = correlation_to_distance(corr)

        rows, cols, ws = [], [], []
        for i in range(n):
            nbrs = np.argsort(dist[i])
            # skip self (dist=0)
            nbrs = [j for j in nbrs if j != i][:k]
            for j in nbrs:
                rows.append(i)
                cols.append(j)
                ws.append(corr[i, j])

        src = torch.tensor(rows, dtype=torch.long)
        dst = torch.tensor(cols, dtype=torch.long)
        wt = torch.tensor(ws, dtype=torch.float32)

        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.stack([wt, wt.abs(), wt.sign()], dim=1)

        node_attr = compute_node_features(returns)

        return FinancialGraphData(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            num_nodes=n,
            asset_names=asset_names,
            graph_type="knn",
            metadata={"k": k},
        )


# ---------------------------------------------------------------------------
# Lead-lag graph
# ---------------------------------------------------------------------------

class LeadLagGraphBuilder:
    """
    Build a directed lead-lag graph via cross-correlation analysis.

    For each pair (i, j), find the lag tau* that maximises |xcorr(i, j, tau)|.
    If tau* > 0, asset i leads asset j; edge i → j with weight xcorr(i, j, tau*).
    """

    def __init__(self, config: Optional[GraphBuildConfig] = None):
        self.config = config or GraphBuildConfig()

    def build(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        asset_names: Optional[List[str]] = None,
    ) -> FinancialGraphData:
        if isinstance(returns, pd.DataFrame):
            asset_names = asset_names or list(returns.columns)
            returns = returns.values.astype(np.float32)
        else:
            returns = returns.astype(np.float32)

        T, n = returns.shape
        max_lag = self.config.max_lag
        asset_names = asset_names or [f"asset_{i}" for i in range(n)]

        rows, cols, ws, lags = [], [], [], []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                xcorr, best_lag = self._cross_corr(returns[:, i], returns[:, j], max_lag)
                if best_lag > 0 and abs(xcorr) >= self.config.corr_threshold:
                    rows.append(i)
                    cols.append(j)
                    ws.append(xcorr)
                    lags.append(best_lag)

        src = torch.tensor(rows, dtype=torch.long)
        dst = torch.tensor(cols, dtype=torch.long)
        wt = torch.tensor(ws, dtype=torch.float32)
        lag_t = torch.tensor(lags, dtype=torch.float32)

        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.stack([wt, wt.abs(), lag_t], dim=1)

        node_attr = compute_node_features(returns)

        return FinancialGraphData(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            num_nodes=n,
            asset_names=asset_names,
            graph_type="lead_lag",
            metadata={"max_lag": max_lag},
        )

    def _cross_corr(
        self, x: np.ndarray, y: np.ndarray, max_lag: int
    ) -> Tuple[float, int]:
        """Return (max_xcorr_value, best_lag > 0) for lags 1..max_lag."""
        best_corr = 0.0
        best_lag = 0
        for lag in range(1, max_lag + 1):
            if lag >= len(x):
                break
            xi = x[:-lag]
            yj = y[lag:]
            if len(xi) < 5:
                continue
            c = np.corrcoef(xi, yj)[0, 1]
            if not np.isfinite(c):
                continue
            if abs(c) > abs(best_corr):
                best_corr = c
                best_lag = lag
        return best_corr, best_lag


# ---------------------------------------------------------------------------
# LOB (Limit Order Book) graph
# ---------------------------------------------------------------------------

@dataclass
class LOBSnapshot:
    """Single snapshot of a limit order book."""
    bid_prices: np.ndarray   # shape (depth,)
    bid_sizes: np.ndarray    # shape (depth,)
    ask_prices: np.ndarray   # shape (depth,)
    ask_sizes: np.ndarray    # shape (depth,)
    timestamp: float = 0.0
    asset_id: int = 0


class LOBGraphBuilder:
    """
    Build a graph from Limit Order Book data.

    Node types:
      - Bid price levels (depth levels on buy side)
      - Ask price levels (depth levels on sell side)
      - Mid price node (aggregated)

    Edge types:
      - Intra-side adjacency: consecutive bid levels, consecutive ask levels
      - Cross-spread: best bid ↔ best ask
      - Order flow: weighted by size imbalance
    """

    def __init__(self, config: Optional[GraphBuildConfig] = None):
        self.config = config or GraphBuildConfig()

    def build_from_snapshot(self, snapshot: LOBSnapshot) -> FinancialGraphData:
        depth = self.config.lob_depth
        bid_p = snapshot.bid_prices[:depth]
        bid_s = snapshot.bid_sizes[:depth]
        ask_p = snapshot.ask_prices[:depth]
        ask_s = snapshot.ask_sizes[:depth]

        nb = len(bid_p)
        na = len(ask_p)
        n_total = nb + na + 1  # +1 for mid-price node

        # Node features: [price, size, side, level_normalized]
        node_feats = []
        for lvl, (p, s) in enumerate(zip(bid_p, bid_s)):
            node_feats.append([p, s, 0.0, lvl / max(nb - 1, 1)])  # side=0 for bid
        for lvl, (p, s) in enumerate(zip(ask_p, ask_s)):
            node_feats.append([p, s, 1.0, lvl / max(na - 1, 1)])  # side=1 for ask
        # mid-price node
        mid = (bid_p[0] + ask_p[0]) / 2.0 if nb > 0 and na > 0 else 0.0
        total_size = float(bid_s.sum() + ask_s.sum())
        node_feats.append([mid, total_size, 0.5, 0.0])

        node_attr = torch.tensor(node_feats, dtype=torch.float32)

        # Edges
        rows, cols, edge_ws = [], [], []

        # Intra-bid adjacency
        for lvl in range(nb - 1):
            rows.append(lvl); cols.append(lvl + 1)
            w = float(bid_s[lvl] + bid_s[lvl + 1]) / (total_size + 1e-8)
            edge_ws.append(w)

        # Intra-ask adjacency
        for lvl in range(na - 1):
            i = nb + lvl; j = nb + lvl + 1
            rows.append(i); cols.append(j)
            w = float(ask_s[lvl] + ask_s[lvl + 1]) / (total_size + 1e-8)
            edge_ws.append(w)

        # Cross-spread: best bid ↔ best ask
        if nb > 0 and na > 0:
            rows.append(0); cols.append(nb)
            spread = float(ask_p[0] - bid_p[0])
            edge_ws.append(1.0 / (spread + 1e-8))

        # Mid node to best bid & best ask
        mid_idx = nb + na
        if nb > 0:
            rows.append(mid_idx); cols.append(0)
            edge_ws.append(1.0)
        if na > 0:
            rows.append(mid_idx); cols.append(nb)
            edge_ws.append(1.0)

        src = torch.tensor(rows, dtype=torch.long)
        dst = torch.tensor(cols, dtype=torch.long)
        wt = torch.tensor(edge_ws, dtype=torch.float32)

        # Make undirected
        src = torch.cat([src, dst])
        dst = torch.cat([dst, torch.tensor(rows, dtype=torch.long)])
        wt = torch.cat([wt, wt])

        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = wt.unsqueeze(1)

        return FinancialGraphData(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            num_nodes=n_total,
            asset_names=[f"bid_{i}" for i in range(nb)]
                        + [f"ask_{i}" for i in range(na)]
                        + ["mid"],
            graph_type="lob",
            metadata={"depth": depth, "spread": float(ask_p[0] - bid_p[0]) if nb > 0 and na > 0 else None},
        )

    def build_from_multi_asset(
        self, snapshots: List[LOBSnapshot]
    ) -> FinancialGraphData:
        """
        Build a combined LOB graph across multiple assets.
        Each asset's LOB becomes a sub-graph; cross-asset edges connect mid-price nodes.
        """
        sub_graphs = [self.build_from_snapshot(s) for s in snapshots]
        return self._merge_sub_graphs(sub_graphs, snapshots)

    def _merge_sub_graphs(
        self,
        sub_graphs: List[FinancialGraphData],
        snapshots: List[LOBSnapshot],
    ) -> FinancialGraphData:
        offsets = []
        offset = 0
        for sg in sub_graphs:
            offsets.append(offset)
            offset += sg.num_nodes

        all_edges_src, all_edges_dst, all_edge_w = [], [], []
        all_node_feats = []

        for sg, off in zip(sub_graphs, offsets):
            all_edges_src.append(sg.edge_index[0] + off)
            all_edges_dst.append(sg.edge_index[1] + off)
            all_edge_w.append(sg.edge_attr.squeeze(-1) if sg.edge_attr is not None else torch.ones(sg.edge_index.shape[1]))
            all_node_feats.append(sg.node_attr)

        # Cross-asset mid-node edges (correlation-like)
        n_assets = len(sub_graphs)
        depth = self.config.lob_depth
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                mid_i = offsets[i] + sub_graphs[i].num_nodes - 1
                mid_j = offsets[j] + sub_graphs[j].num_nodes - 1
                all_edges_src.append(torch.tensor([mid_i, mid_j]))
                all_edges_dst.append(torch.tensor([mid_j, mid_i]))
                all_edge_w.append(torch.tensor([0.5, 0.5]))

        src_all = torch.cat(all_edges_src)
        dst_all = torch.cat(all_edges_dst)
        w_all = torch.cat(all_edge_w)
        edge_index = torch.stack([src_all, dst_all], dim=0)
        edge_attr = w_all.unsqueeze(1)
        node_attr = torch.cat(all_node_feats, dim=0)

        return FinancialGraphData(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            num_nodes=offset,
            graph_type="multi_asset_lob",
            metadata={"n_assets": n_assets, "depth": depth},
        )


# ---------------------------------------------------------------------------
# Node feature engineering
# ---------------------------------------------------------------------------

def compute_node_features(
    returns: np.ndarray,
    prices: Optional[np.ndarray] = None,
    volumes: Optional[np.ndarray] = None,
    bid_ask_spreads: Optional[np.ndarray] = None,
    window: int = 20,
) -> Tensor:
    """
    Compute rich per-asset node features from time-series data.

    Features computed (per asset):
      0: mean return (full series)
      1: std return (full series)
      2: mean return (last `window` steps)
      3: std return (last `window` steps)
      4: skewness
      5: kurtosis
      6: Sharpe ratio (annualised, sqrt(252))
      7: max drawdown
      8: autocorrelation lag-1
      9: mean volume (if provided, else 0)
     10: mean bid-ask spread (if provided, else 0)
     11: realised volatility (last window)
     12: up-ratio (fraction of positive returns)
     13: normalised price level (if provided, else 0)

    Parameters
    ----------
    returns : (T, N)
    prices  : (T, N) optional
    volumes : (T, N) optional
    bid_ask_spreads : (T, N) optional
    window : lookback for recent statistics

    Returns
    -------
    Tensor of shape (N, 14)
    """
    T, N = returns.shape
    feats = np.zeros((N, 14), dtype=np.float32)

    for i in range(N):
        r = returns[:, i]
        r_w = r[-window:] if T >= window else r

        feats[i, 0] = float(np.mean(r))
        feats[i, 1] = float(np.std(r) + 1e-8)
        feats[i, 2] = float(np.mean(r_w))
        feats[i, 3] = float(np.std(r_w) + 1e-8)

        if len(r) >= 3:
            feats[i, 4] = float(stats.skew(r))
            feats[i, 5] = float(stats.kurtosis(r))
        else:
            feats[i, 4] = 0.0
            feats[i, 5] = 0.0

        std_r = float(np.std(r)) + 1e-8
        feats[i, 6] = float(np.mean(r) / std_r * math.sqrt(252))

        # Max drawdown
        cum = np.cumprod(1.0 + np.clip(r, -0.99, None))
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / (peak + 1e-8)
        feats[i, 7] = float(np.min(dd))

        # Autocorrelation lag-1
        if T > 2:
            ac = np.corrcoef(r[:-1], r[1:])[0, 1]
            feats[i, 8] = float(ac) if np.isfinite(ac) else 0.0

        if volumes is not None:
            feats[i, 9] = float(np.mean(volumes[:, i]))

        if bid_ask_spreads is not None:
            feats[i, 10] = float(np.mean(bid_ask_spreads[:, i]))

        feats[i, 11] = float(np.std(r_w) * math.sqrt(252))
        feats[i, 12] = float(np.mean(r > 0))

        if prices is not None:
            p = prices[:, i]
            feats[i, 13] = float(p[-1] / (p.mean() + 1e-8))

    # Clip extreme values
    feats = np.clip(feats, -10.0, 10.0)
    return torch.tensor(feats, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Edge feature engineering (standalone helper)
# ---------------------------------------------------------------------------

def compute_edge_features(
    edge_index: Tensor,
    corr_matrix: np.ndarray,
    dist_matrix: Optional[np.ndarray] = None,
    lag_matrix: Optional[np.ndarray] = None,
) -> Tensor:
    """
    Compute rich per-edge features.

    Features:
      0: correlation value (signed)
      1: |correlation| (strength)
      2: sign of correlation (direction)
      3: distance correlation (from dist_matrix, or 1-|corr|)
      4: lead-lag (from lag_matrix, or 0)
      5: normalised correlation rank

    Parameters
    ----------
    edge_index : (2, E)
    corr_matrix : (N, N)
    dist_matrix : (N, N) optional
    lag_matrix  : (N, N) optional

    Returns
    -------
    Tensor (E, 6)
    """
    E = edge_index.shape[1]
    feats = torch.zeros(E, 6, dtype=torch.float32)

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    for k in range(E):
        i, j = int(src[k]), int(dst[k])
        c = corr_matrix[i, j]
        feats[k, 0] = c
        feats[k, 1] = abs(c)
        feats[k, 2] = 1.0 if c > 0 else (-1.0 if c < 0 else 0.0)

        if dist_matrix is not None:
            feats[k, 3] = float(dist_matrix[i, j])
        else:
            feats[k, 3] = float(1.0 - abs(c))

        if lag_matrix is not None:
            feats[k, 4] = float(lag_matrix[i, j])

    # Normalised rank
    strengths = feats[:, 1].clone()
    ranks = torch.argsort(torch.argsort(strengths)).float()
    feats[:, 5] = ranks / (E + 1e-8)

    return feats


# ---------------------------------------------------------------------------
# Composite builder — auto-select best graph type
# ---------------------------------------------------------------------------

class AdaptiveGraphBuilder:
    """
    Automatically selects and combines graph construction methods.

    Strategy:
      1. Compute correlation matrix
      2. Build MST as backbone
      3. Augment with PMFG (more edges)
      4. Add lead-lag directed edges
      5. Merge into final graph
    """

    def __init__(self, config: Optional[GraphBuildConfig] = None):
        self.config = config or GraphBuildConfig()
        self._corr_builder = CorrelationGraphBuilder(config)
        self._mst_builder = MSTGraphBuilder(config)
        self._pmfg_builder = PMFGBuilder(config)
        self._ll_builder = LeadLagGraphBuilder(config)

    def build(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        asset_names: Optional[List[str]] = None,
        include_lead_lag: bool = True,
    ) -> FinancialGraphData:
        if isinstance(returns, pd.DataFrame):
            asset_names = asset_names or list(returns.columns)
            returns = returns.values.astype(np.float32)
        else:
            returns = returns.astype(np.float32)

        n = returns.shape[1]
        asset_names = asset_names or [f"asset_{i}" for i in range(n)]

        mst = self._mst_builder.build(returns, asset_names)
        pmfg = self._pmfg_builder.build(returns, asset_names)

        # Merge undirected edges from MST and PMFG
        ei = torch.cat([mst.edge_index, pmfg.edge_index], dim=1)
        ea = torch.cat([mst.edge_attr, pmfg.edge_attr], dim=0)

        if include_lead_lag and returns.shape[0] > self.config.max_lag + 5:
            ll = self._ll_builder.build(returns, asset_names)
            if ll.edge_index.shape[1] > 0:
                # Pad edge_attr to same dim
                ea_ll = ll.edge_attr[:, :3]
                ei = torch.cat([ei, ll.edge_index], dim=1)
                ea = torch.cat([ea, ea_ll], dim=0)

        # Deduplicate edges
        if HAS_PYG:
            ei, ea = coalesce(ei, ea, num_nodes=n)

        node_attr = compute_node_features(returns)

        return FinancialGraphData(
            edge_index=ei,
            edge_attr=ea,
            node_attr=node_attr,
            num_nodes=n,
            asset_names=asset_names,
            graph_type="adaptive_composite",
            metadata={"n_mst_edges": mst.edge_index.shape[1], "n_pmfg_edges": pmfg.edge_index.shape[1]},
        )


# ---------------------------------------------------------------------------
# Graph normalisation utilities
# ---------------------------------------------------------------------------

def normalise_edge_weights(
    edge_attr: Tensor,
    method: str = "zscore",
    eps: float = 1e-8,
) -> Tensor:
    """
    Normalise edge weight features.

    Parameters
    ----------
    edge_attr : (E, F)
    method : zscore | minmax | softmax
    """
    if edge_attr.shape[0] == 0:
        return edge_attr

    if method == "zscore":
        mean = edge_attr.mean(dim=0, keepdim=True)
        std = edge_attr.std(dim=0, keepdim=True) + eps
        return (edge_attr - mean) / std

    elif method == "minmax":
        mn = edge_attr.min(dim=0, keepdim=True).values
        mx = edge_attr.max(dim=0, keepdim=True).values
        return (edge_attr - mn) / (mx - mn + eps)

    elif method == "softmax":
        return F.softmax(edge_attr, dim=0)

    else:
        raise ValueError(f"Unknown normalisation method: {method}")


def normalise_node_features(
    node_attr: Tensor,
    method: str = "zscore",
    eps: float = 1e-8,
) -> Tensor:
    """Normalise node feature matrix column-wise."""
    if node_attr.shape[0] == 0:
        return node_attr

    if method == "zscore":
        mean = node_attr.mean(dim=0, keepdim=True)
        std = node_attr.std(dim=0, keepdim=True) + eps
        return (node_attr - mean) / std

    elif method == "minmax":
        mn = node_attr.min(dim=0, keepdim=True).values
        mx = node_attr.max(dim=0, keepdim=True).values
        return (node_attr - mn) / (mx - mn + eps)

    else:
        raise ValueError(f"Unknown normalisation method: {method}")


# ---------------------------------------------------------------------------
# Sector / hierarchical grouping
# ---------------------------------------------------------------------------

def build_sector_graph(
    returns: np.ndarray,
    sector_map: Dict[int, int],
    asset_names: Optional[List[str]] = None,
    config: Optional[GraphBuildConfig] = None,
) -> FinancialGraphData:
    """
    Build a two-level hierarchical graph:
      Level 0: asset nodes
      Level 1: sector nodes (aggregated)

    Edges:
      - Asset-asset: correlation within sector
      - Asset-sector: membership (always present)
      - Sector-sector: correlation between sector return indices
    """
    config = config or GraphBuildConfig()
    T, N = returns.shape
    asset_names = asset_names or [f"asset_{i}" for i in range(N)]

    sectors = sorted(set(sector_map.values()))
    S = len(sectors)
    sector_idx = {s: N + i for i, s in enumerate(sectors)}
    n_total = N + S

    # Asset-asset edges (intra-sector correlation)
    rows, cols, ws = [], [], []
    corr = pearson_correlation_matrix(returns)
    for i in range(N):
        for j in range(i + 1, N):
            if sector_map.get(i) == sector_map.get(j):
                c = corr[i, j]
                if abs(c) >= config.corr_threshold:
                    rows.append(i); cols.append(j); ws.append(c)

    # Asset-sector membership edges
    for asset_i, sector_id in sector_map.items():
        if asset_i < N:
            si = sector_idx[sector_id]
            rows.append(asset_i); cols.append(si); ws.append(1.0)

    # Sector-sector edges: compute sector returns
    sector_returns = np.zeros((T, S), dtype=np.float32)
    for asset_i, sector_id in sector_map.items():
        if asset_i < N:
            si = sectors.index(sector_id)
            sector_returns[:, si] += returns[:, asset_i]

    if S > 1:
        scorr = pearson_correlation_matrix(sector_returns)
        for i in range(S):
            for j in range(i + 1, S):
                c = scorr[i, j]
                if abs(c) >= config.corr_threshold:
                    si = N + i; sj = N + j
                    rows.append(si); cols.append(sj); ws.append(c)

    src = torch.tensor(rows, dtype=torch.long)
    dst = torch.tensor(cols, dtype=torch.long)
    wt = torch.tensor(ws, dtype=torch.float32)

    src_all = torch.cat([src, dst])
    dst_all = torch.cat([dst, src])
    wt_all = torch.cat([wt, wt])

    edge_index = torch.stack([src_all, dst_all], dim=0)
    edge_attr = wt_all.unsqueeze(1)

    asset_feats = compute_node_features(returns)
    sector_feats = compute_node_features(sector_returns)
    node_attr = torch.cat([asset_feats, sector_feats], dim=0)

    return FinancialGraphData(
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_attr=node_attr,
        num_nodes=n_total,
        asset_names=asset_names + [f"sector_{s}" for s in sectors],
        graph_type="sector_hierarchical",
        metadata={"n_assets": N, "n_sectors": S, "sector_map": sector_map},
    )


# ---------------------------------------------------------------------------
# Temporal graph snapshot sequence
# ---------------------------------------------------------------------------

class TemporalGraphSequenceBuilder:
    """
    Build a sequence of graph snapshots over rolling windows.

    Each snapshot is a FinancialGraphData built from a rolling window of returns.
    """

    def __init__(
        self,
        window: int = 60,
        stride: int = 5,
        builder: Optional[CorrelationGraphBuilder] = None,
        config: Optional[GraphBuildConfig] = None,
    ):
        self.window = window
        self.stride = stride
        self.builder = builder or CorrelationGraphBuilder(config)

    def build_sequence(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        asset_names: Optional[List[str]] = None,
    ) -> List[FinancialGraphData]:
        if isinstance(returns, pd.DataFrame):
            asset_names = asset_names or list(returns.columns)
            returns = returns.values.astype(np.float32)
        else:
            returns = returns.astype(np.float32)

        T = returns.shape[0]
        snapshots = []
        t = self.window
        while t <= T:
            chunk = returns[t - self.window : t]
            g = self.builder.build(chunk, asset_names)
            g.metadata["t_start"] = t - self.window
            g.metadata["t_end"] = t
            snapshots.append(g)
            t += self.stride

        return snapshots


# ---------------------------------------------------------------------------
# Graph summary statistics
# ---------------------------------------------------------------------------

def graph_summary(g: FinancialGraphData) -> Dict:
    """Return basic statistics about a FinancialGraphData object."""
    n = g.num_nodes
    e = g.edge_index.shape[1]
    density = 2 * e / max(n * (n - 1), 1)

    summary = {
        "num_nodes": n,
        "num_edges": e,
        "density": density,
        "graph_type": g.graph_type,
    }

    if g.edge_attr is not None and g.edge_attr.shape[0] > 0:
        w = g.edge_attr[:, 0]
        summary["mean_edge_weight"] = float(w.mean())
        summary["std_edge_weight"] = float(w.std())
        summary["min_edge_weight"] = float(w.min())
        summary["max_edge_weight"] = float(w.max())

    if HAS_NX and n < 5000:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        ei = g.edge_index.numpy()
        for k in range(ei.shape[1]):
            G.add_edge(int(ei[0, k]), int(ei[1, k]))
        summary["n_connected_components"] = nx.number_connected_components(G)
        if nx.is_connected(G):
            summary["diameter"] = nx.diameter(G)
        summary["avg_clustering"] = nx.average_clustering(G)

    return summary


# ---------------------------------------------------------------------------
# Utility: returns → PyG Data batch
# ---------------------------------------------------------------------------

def returns_to_pyg_data(
    returns: Union[np.ndarray, pd.DataFrame],
    method: str = "pearson",
    threshold: float = 0.3,
    k: int = 5,
    asset_names: Optional[List[str]] = None,
) -> "Data":
    """
    Convenience function: raw returns → PyG Data object.

    Parameters
    ----------
    returns : (T, N) array or DataFrame
    method : graph construction method — pearson | spearman | distance | mst | knn | pmfg
    threshold : correlation threshold (for pearson/spearman/distance)
    k : number of neighbours (for knn)
    """
    if not HAS_PYG:
        raise ImportError("torch_geometric required for returns_to_pyg_data")

    cfg = GraphBuildConfig(
        corr_method=method if method in ("pearson", "spearman", "distance") else "pearson",
        corr_threshold=threshold,
        k_neighbours=k,
    )

    if method in ("pearson", "spearman", "distance"):
        builder = CorrelationGraphBuilder(cfg)
    elif method == "mst":
        builder = MSTGraphBuilder(cfg)
    elif method == "knn":
        builder = KNNGraphBuilder(cfg)
    elif method == "pmfg":
        builder = PMFGBuilder(cfg)
    else:
        raise ValueError(f"Unknown method: {method}")

    fg = builder.build(returns, asset_names)
    return fg.to_pyg()


# ---------------------------------------------------------------------------
# Correlation matrix regularisation
# ---------------------------------------------------------------------------

def ledoit_wolf_shrinkage(returns: np.ndarray) -> np.ndarray:
    """
    Estimate the Ledoit-Wolf shrinkage correlation matrix.
    Shrinks sample correlation toward the identity matrix.
    """
    T, N = returns.shape
    sample_corr = pearson_correlation_matrix(returns)

    # Oracle shrinkage intensity (simplified Ledoit-Wolf)
    mu = np.trace(sample_corr) / N
    delta2 = np.sum((sample_corr - mu * np.eye(N)) ** 2) / N
    beta2 = 0.0
    for t in range(T):
        r = returns[t]
        outer = np.outer(r, r)
        beta2 += np.sum((outer - sample_corr) ** 2)
    beta2 /= T ** 2

    alpha = min(beta2 / delta2, 1.0)
    shrunk = (1 - alpha) * sample_corr + alpha * mu * np.eye(N)
    return shrunk


def random_matrix_filter(
    corr: np.ndarray,
    T: int,
    q_threshold_factor: float = 1.0,
) -> np.ndarray:
    """
    Filter noise from a correlation matrix using Random Matrix Theory.

    Eigenvalues below the Marchenko-Pastur upper bound are treated as noise
    and replaced by their average (noise floor).

    Parameters
    ----------
    corr : (N, N) correlation matrix
    T : number of time steps
    q_threshold_factor : multiplier on Marchenko-Pastur edge
    """
    N = corr.shape[0]
    q = T / N  # ratio
    lambda_max = (1 + math.sqrt(1 / q)) ** 2 * q_threshold_factor

    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    noise_mask = eigenvalues < lambda_max
    noise_eigenvalues = eigenvalues[noise_mask]
    noise_mean = noise_eigenvalues.mean() if noise_mask.sum() > 0 else 1.0

    filtered_eigenvalues = eigenvalues.copy()
    filtered_eigenvalues[noise_mask] = noise_mean

    filtered_corr = eigenvectors @ np.diag(filtered_eigenvalues) @ eigenvectors.T
    # Renormalise diagonal to 1
    diag = np.sqrt(np.diag(filtered_corr))
    filtered_corr /= np.outer(diag, diag)
    np.fill_diagonal(filtered_corr, 1.0)
    return filtered_corr


# ---------------------------------------------------------------------------
# Financial graph metrics
# ---------------------------------------------------------------------------

class GraphMetricsComputer:
    """
    Compute financial-graph-specific metrics:
      - Degree distribution
      - Betweenness centrality
      - Fiedler value (algebraic connectivity)
      - Spectral gap
      - Community structure (via modularity)
    """

    def __init__(self, g: FinancialGraphData):
        self.g = g
        self._G: Optional["nx.Graph"] = None

    @property
    def nx_graph(self) -> "nx.Graph":
        if self._G is None:
            if not HAS_NX:
                raise ImportError("networkx is required for graph metrics.")
            self._G = nx.Graph()
            n = self.g.num_nodes
            self._G.add_nodes_from(range(n))
            ei = self.g.edge_index.numpy()
            for k in range(ei.shape[1]):
                i, j = int(ei[0, k]), int(ei[1, k])
                if i != j:
                    w = float(self.g.edge_attr[k, 0]) if self.g.edge_attr is not None else 1.0
                    self._G.add_edge(i, j, weight=abs(w))
        return self._G

    def degree_distribution(self) -> np.ndarray:
        degrees = np.array([d for _, d in self.nx_graph.degree()])
        return degrees

    def fiedler_value(self) -> float:
        """Compute the Fiedler value (second-smallest Laplacian eigenvalue)."""
        L = nx.normalized_laplacian_matrix(self.nx_graph).toarray()
        eigvals = np.linalg.eigvalsh(L)
        eigvals_sorted = np.sort(eigvals)
        return float(eigvals_sorted[1]) if len(eigvals_sorted) > 1 else 0.0

    def spectral_gap(self) -> float:
        """Spectral gap = lambda_2 - lambda_1 of Laplacian."""
        L = nx.laplacian_matrix(self.nx_graph).toarray().astype(float)
        eigvals = np.linalg.eigvalsh(L)
        eigvals_sorted = np.sort(eigvals)
        return float(eigvals_sorted[1] - eigvals_sorted[0]) if len(eigvals_sorted) > 1 else 0.0

    def betweenness_centrality(self) -> Dict[int, float]:
        return nx.betweenness_centrality(self.nx_graph, weight="weight")

    def modularity_communities(self, resolution: float = 1.0) -> List[List[int]]:
        """Detect communities via greedy modularity maximisation."""
        communities = list(
            nx.community.greedy_modularity_communities(
                self.nx_graph, weight="weight", resolution=resolution
            )
        )
        return [sorted(c) for c in communities]

    def all_metrics(self) -> Dict:
        deg = self.degree_distribution()
        return {
            "fiedler_value": self.fiedler_value(),
            "spectral_gap": self.spectral_gap(),
            "mean_degree": float(deg.mean()),
            "max_degree": int(deg.max()),
            "density": nx.density(self.nx_graph),
            "n_connected_components": nx.number_connected_components(self.nx_graph),
        }


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "GraphBuildConfig",
    "FinancialGraphData",
    "CorrelationGraphBuilder",
    "MSTGraphBuilder",
    "PMFGBuilder",
    "KNNGraphBuilder",
    "LeadLagGraphBuilder",
    "LOBGraphBuilder",
    "LOBSnapshot",
    "AdaptiveGraphBuilder",
    "TemporalGraphSequenceBuilder",
    "GraphMetricsComputer",
    "compute_node_features",
    "compute_edge_features",
    "normalise_edge_weights",
    "normalise_node_features",
    "build_sector_graph",
    "returns_to_pyg_data",
    "pearson_correlation_matrix",
    "spearman_correlation_matrix",
    "distance_correlation_matrix",
    "correlation_to_distance",
    "ledoit_wolf_shrinkage",
    "random_matrix_filter",
    "graph_summary",
]
