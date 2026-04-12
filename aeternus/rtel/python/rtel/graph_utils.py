"""
AETERNUS Real-Time Execution Layer (RTEL)
graph_utils.py — Graph construction and analysis utilities for OmniGraph

Used by OmniGraph to build and analyze correlation/causality networks.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph representations
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    src: int
    dst: int
    weight: float


@dataclass
class GraphCSR:
    """Compressed Sparse Row graph representation."""
    n_nodes:     int
    n_edges:     int
    row_ptr:     np.ndarray   # [n_nodes+1] int32
    col_idx:     np.ndarray   # [n_edges]   int32
    edge_weight: np.ndarray   # [n_edges]   float32

    def degree(self, node: int) -> int:
        return int(self.row_ptr[node + 1] - self.row_ptr[node])

    def neighbors(self, node: int) -> np.ndarray:
        start = int(self.row_ptr[node])
        end   = int(self.row_ptr[node + 1])
        return self.col_idx[start:end]

    def neighbor_weights(self, node: int) -> np.ndarray:
        start = int(self.row_ptr[node])
        end   = int(self.row_ptr[node + 1])
        return self.edge_weight[start:end]

    def to_dense(self) -> np.ndarray:
        mat = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        for i in range(self.n_nodes):
            start = int(self.row_ptr[i])
            end   = int(self.row_ptr[i + 1])
            for j_idx in range(start, end):
                j = int(self.col_idx[j_idx])
                mat[i, j] = self.edge_weight[j_idx]
        return mat

    @classmethod
    def from_dense(cls, mat: np.ndarray, threshold: float = 0.0) -> "GraphCSR":
        n = mat.shape[0]
        row_ptr = np.zeros(n + 1, dtype=np.int32)
        col_list, wgt_list = [], []
        for i in range(n):
            for j in range(n):
                if i != j and abs(mat[i, j]) > threshold:
                    col_list.append(j)
                    wgt_list.append(mat[i, j])
            row_ptr[i + 1] = len(col_list)
        col_idx    = np.array(col_list, dtype=np.int32)
        edge_weight= np.array(wgt_list, dtype=np.float32)
        return cls(n, len(col_list), row_ptr, col_idx, edge_weight)

    @classmethod
    def from_edges(cls, n_nodes: int, edges: List[Edge]) -> "GraphCSR":
        # Sort by source node
        edges = sorted(edges, key=lambda e: e.src)
        row_ptr    = np.zeros(n_nodes + 1, dtype=np.int32)
        col_idx    = np.array([e.dst    for e in edges], dtype=np.int32)
        edge_weight= np.array([e.weight for e in edges], dtype=np.float32)
        for e in edges:
            row_ptr[e.src + 1] += 1
        np.cumsum(row_ptr, out=row_ptr)
        return cls(n_nodes, len(edges), row_ptr, col_idx, edge_weight)


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def build_correlation_graph(returns: np.ndarray,
                              threshold: float = 0.3,
                              window: int = 60) -> GraphCSR:
    """Build asset correlation graph from returns matrix [T × N]."""
    T, N = returns.shape
    start = max(0, T - window)
    r = returns[start:]
    if len(r) < 2:
        return GraphCSR(N, 0, np.zeros(N+1,dtype=np.int32),
                        np.array([],dtype=np.int32),
                        np.array([],dtype=np.float32))
    corr = np.corrcoef(r.T)
    np.fill_diagonal(corr, 0.0)
    return GraphCSR.from_dense(corr, threshold)


def build_minimum_spanning_tree(corr_matrix: np.ndarray) -> GraphCSR:
    """Build MST from correlation matrix using Kruskal's algorithm."""
    n = corr_matrix.shape[0]
    # Convert correlations to distances: d = sqrt(2*(1-corr))
    dist = np.sqrt(np.clip(2 * (1 - corr_matrix), 0, None))
    np.fill_diagonal(dist, np.inf)

    # Kruskal's
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist[i, j], i, j))
    edges.sort()

    # Union-Find
    parent = list(range(n))
    rank   = [0] * n

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
    for d, i, j in edges:
        if len(mst_edges) >= n - 1:
            break
        if union(i, j):
            mst_edges.append(Edge(i, j, 1.0 - d))  # back to correlation
            mst_edges.append(Edge(j, i, 1.0 - d))  # undirected

    return GraphCSR.from_edges(n, mst_edges)


def build_granger_causality_graph(returns: np.ndarray,
                                   max_lag: int = 5,
                                   threshold: float = 0.05) -> GraphCSR:
    """Build approximate Granger causality graph using OLS F-tests."""
    T, N = returns.shape
    if T < max_lag + 10:
        return GraphCSR(N, 0, np.zeros(N+1,dtype=np.int32),
                        np.array([],dtype=np.int32),
                        np.array([],dtype=np.float32))

    edges = []
    for target in range(N):
        y = returns[max_lag:, target]
        # Restricted: only lags of y
        X_r = np.column_stack([returns[max_lag-l-1:-l-1 if l+1 < T else T-l-1, target]
                                for l in range(max_lag)])
        X_r = np.column_stack([np.ones(len(y)), X_r])

        for cause in range(N):
            if cause == target:
                continue
            # Unrestricted: lags of y + lags of cause
            X_u = np.column_stack([X_r,
                                    *[returns[max_lag-l-1:-l-1 if l+1 < T else T-l-1, cause]
                                      for l in range(max_lag)]])
            try:
                # F-test approximation using RSS comparison
                beta_r, rss_r, _, _ = np.linalg.lstsq(X_r, y, rcond=None)
                beta_u, rss_u, _, _ = np.linalg.lstsq(X_u, y, rcond=None)
                if len(rss_r) == 0 or len(rss_u) == 0:
                    continue
                f_stat = ((rss_r[0] - rss_u[0]) / max_lag) / (rss_u[0] / (T - max_lag * 2 - 1) + 1e-10)
                # Approximate p-value using chi-squared
                import scipy.stats
                p_val = float(1 - scipy.stats.f.cdf(f_stat, max_lag, T - max_lag * 2 - 1))
                if p_val < threshold:
                    edges.append(Edge(cause, target, 1.0 - p_val))
            except Exception:
                pass

    return GraphCSR.from_edges(N, edges)


# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------

def degree_centrality(g: GraphCSR) -> np.ndarray:
    """Normalized degree centrality."""
    n = g.n_nodes
    if n <= 1:
        return np.zeros(n)
    degrees = np.array([g.degree(i) for i in range(n)], dtype=np.float32)
    return degrees / (n - 1)


def strength_centrality(g: GraphCSR) -> np.ndarray:
    """Sum of edge weights per node (strength)."""
    strength = np.zeros(g.n_nodes, dtype=np.float32)
    for i in range(g.n_nodes):
        strength[i] = float(np.abs(g.neighbor_weights(i)).sum())
    return strength


def clustering_coefficient(g: GraphCSR) -> np.ndarray:
    """Local clustering coefficient for each node."""
    n = g.n_nodes
    cc = np.zeros(n, dtype=np.float32)
    for i in range(n):
        nbrs = set(g.neighbors(i).tolist())
        k    = len(nbrs)
        if k < 2:
            continue
        # Count triangles
        triangles = 0
        for j in nbrs:
            j_nbrs = set(g.neighbors(j).tolist())
            triangles += len(nbrs & j_nbrs)
        cc[i] = triangles / (k * (k - 1))
    return cc


def page_rank(g: GraphCSR, damping: float = 0.85,
               max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """PageRank using power iteration."""
    n = g.n_nodes
    rank = np.ones(n, dtype=np.float64) / n
    for _ in range(max_iter):
        new_rank = np.full(n, (1 - damping) / n)
        for i in range(n):
            d = g.degree(i)
            if d == 0:
                new_rank += rank[i] / n  # dangling node
            else:
                for j in g.neighbors(i):
                    new_rank[j] += damping * rank[i] / d
        delta = np.abs(new_rank - rank).sum()
        rank  = new_rank
        if delta < tol:
            break
    return rank.astype(np.float32)


def betweenness_centrality(g: GraphCSR) -> np.ndarray:
    """Approximate betweenness centrality using BFS."""
    n = g.n_nodes
    bc = np.zeros(n, dtype=np.float32)
    for s in range(n):
        # BFS
        visited = [-1] * n
        visited[s] = 0
        sigma = [0.0] * n
        sigma[s] = 1.0
        queue = [s]
        stack = []
        pred  = [[] for _ in range(n)]
        dist  = [-1] * n
        dist[s] = 0

        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in g.neighbors(v).tolist():
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = [0.0] * n
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]

    # Normalize
    if n > 2:
        bc /= ((n - 1) * (n - 2))
    return bc


def spectral_gap(adj_matrix: np.ndarray) -> float:
    """Algebraic connectivity (second smallest Laplacian eigenvalue)."""
    n = adj_matrix.shape[0]
    D = np.diag(adj_matrix.sum(axis=1))
    L = D - adj_matrix
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues.sort()
    return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

def louvain_communities(adj_matrix: np.ndarray,
                         resolution: float = 1.0,
                         n_iter: int = 10) -> np.ndarray:
    """
    Simplified Louvain-like community detection.
    Returns community label per node [n_nodes] int32.
    """
    n = adj_matrix.shape[0]
    labels = np.arange(n, dtype=np.int32)
    total_weight = adj_matrix.sum()
    if total_weight < 1e-10:
        return labels

    for _ in range(n_iter):
        improved = False
        for i in range(n):
            nbrs = np.where(adj_matrix[i] > 0)[0]
            if len(nbrs) == 0:
                continue
            # Try moving node i to each neighbor's community
            ki     = adj_matrix[i].sum()
            best_gain = 0.0
            best_comm = labels[i]
            for j in nbrs:
                comm_j = labels[j]
                # Compute modularity gain (simplified)
                in_wgt = sum(adj_matrix[i, k] for k in np.where(labels == comm_j)[0])
                comm_wgt = sum(adj_matrix[k1, k2]
                               for k1 in np.where(labels == comm_j)[0]
                               for k2 in range(n))
                gain = in_wgt - resolution * ki * comm_wgt / (2 * total_weight + 1e-10)
                if gain > best_gain:
                    best_gain = gain
                    best_comm = comm_j

            if best_comm != labels[i]:
                labels[i] = best_comm
                improved = True

        if not improved:
            break

    # Relabel to consecutive integers
    unique, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int32)


def graph_summary(g: GraphCSR) -> Dict:
    """Compute summary statistics of a graph."""
    dc  = degree_centrality(g)
    cc  = clustering_coefficient(g)
    pr  = page_rank(g)
    return {
        "n_nodes":        g.n_nodes,
        "n_edges":        g.n_edges,
        "density":        g.n_edges / max(1, g.n_nodes * (g.n_nodes - 1)),
        "mean_degree":    float(dc.mean() * (g.n_nodes - 1)),
        "max_degree":     float(dc.max() * (g.n_nodes - 1)),
        "mean_clustering":float(cc.mean()),
        "max_pagerank":   float(pr.max()),
        "min_pagerank":   float(pr.min()),
    }
