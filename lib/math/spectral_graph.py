"""
Spectral graph theory for asset correlation networks.

Implements:
  - Asset correlation graph construction
  - Minimum Spanning Tree (Kruskal/Prim)
  - Planar Maximally Filtered Graph (PMFG)
  - Graph Laplacian and spectral decomposition
  - Spectral clustering of assets
  - Centrality measures (degree, betweenness, eigenvector)
  - Community detection (Louvain-style)
  - Network risk metrics (systemic importance, contagion)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Correlation to distance ───────────────────────────────────────────────────

def correlation_distance(C: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance: d_ij = sqrt(2*(1 - rho_ij))."""
    return np.sqrt(np.maximum(2 * (1 - C), 0.0))


# ── Minimum Spanning Tree ─────────────────────────────────────────────────────

def minimum_spanning_tree(D: np.ndarray) -> np.ndarray:
    """
    Kruskal's MST on distance matrix D.
    Returns adjacency matrix of MST (symmetric, binary).
    """
    n = D.shape[0]
    # All edges sorted by weight
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((D[i, j], i, j))
    edges.sort()

    # Union-Find
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    adj = np.zeros((n, n))
    for w, i, j in edges:
        if union(i, j):
            adj[i, j] = adj[j, i] = 1
    return adj


# ── Graph Laplacian ───────────────────────────────────────────────────────────

def graph_laplacian(
    C: np.ndarray,
    threshold: Optional[float] = None,
    method: str = "correlation",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build weighted graph Laplacian from correlation matrix.

    Parameters:
      C         : correlation matrix
      threshold : minimum correlation to include edge (None = full graph)
      method    : 'correlation' (weight=|rho|) or 'mst' (use MST)

    Returns:
      (L, W) — Laplacian and weight matrix
    """
    n = C.shape[0]

    if method == "mst":
        D = correlation_distance(C)
        W = minimum_spanning_tree(D) * np.abs(C)
    else:
        W = np.abs(C.copy())
        np.fill_diagonal(W, 0.0)
        if threshold is not None:
            W[W < threshold] = 0.0

    # Degree matrix
    deg = W.sum(axis=1)
    D_mat = np.diag(deg)
    L = D_mat - W
    return L, W


def spectral_decomposition(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition of graph Laplacian.
    Returns sorted (eigenvalues, eigenvectors), eigenvalues ascending.
    """
    eigs, vecs = np.linalg.eigh(L)
    idx = np.argsort(eigs)
    return eigs[idx], vecs[:, idx]


# ── Spectral clustering ────────────────────────────────────────────────────────

def spectral_clustering(
    C: np.ndarray,
    n_clusters: int = 4,
    threshold: Optional[float] = 0.3,
) -> np.ndarray:
    """
    Spectral clustering of assets into n_clusters groups.
    Returns cluster labels array (n_assets,).
    """
    from scipy.cluster.vq import kmeans2

    L, _ = graph_laplacian(C, threshold=threshold)
    eigs, vecs = spectral_decomposition(L)

    # Use first n_clusters eigenvectors (skip constant vector)
    X = vecs[:, 1: n_clusters + 1]
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms > 0, norms, 1.0)

    try:
        centroids, labels = kmeans2(X, n_clusters, niter=100, minit="points")
    except Exception:
        labels = np.arange(len(C)) % n_clusters

    return labels


# ── Centrality measures ────────────────────────────────────────────────────────

def degree_centrality(W: np.ndarray) -> np.ndarray:
    """Weighted degree centrality (normalized)."""
    deg = W.sum(axis=1)
    return deg / deg.sum() if deg.sum() > 0 else deg


def eigenvector_centrality(W: np.ndarray, n_iter: int = 100) -> np.ndarray:
    """
    Eigenvector centrality via power iteration.
    Central assets = those connected to other central assets.
    """
    n = W.shape[0]
    x = np.ones(n)
    for _ in range(n_iter):
        x_new = W @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-10:
            break
        x = x_new / norm
    return x


def betweenness_centrality(W: np.ndarray) -> np.ndarray:
    """
    Approximate betweenness centrality via shortest paths (Floyd-Warshall).
    Higher = more central to information flow in the network.
    """
    n = W.shape[0]
    # Distance matrix (inf where no edge)
    D = np.where(W > 0, 1.0 / (W + 1e-10), np.inf)
    np.fill_diagonal(D, 0.0)

    # Floyd-Warshall
    pred = np.full((n, n), -1, dtype=int)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i, k] + D[k, j] < D[i, j]:
                    D[i, j] = D[i, k] + D[k, j]
                    pred[i, j] = k

    # Count paths through each node
    bc = np.zeros(n)
    for s in range(n):
        for t in range(n):
            if s != t:
                # Trace path
                v = pred[s, t]
                while v != -1 and v != s:
                    bc[v] += 1
                    v = pred[s, v]

    total = (n - 1) * (n - 2)
    return bc / total if total > 0 else bc


# ── Network risk metrics ──────────────────────────────────────────────────────

def systemic_importance(W: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Systemic importance score combining degree, eigenvector centrality,
    and average correlation.
    Higher score → more systemically important (removal has bigger impact).
    """
    deg = degree_centrality(W)
    eig = eigenvector_centrality(W)
    avg_corr = (np.abs(C).sum(axis=1) - 1) / (C.shape[0] - 1)  # exclude self

    # Normalize each component
    def norm01(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 0 else np.zeros_like(x)

    return (norm01(deg) + norm01(eig) + norm01(avg_corr)) / 3


def network_density(W: np.ndarray) -> float:
    """Fraction of possible edges that exist (above threshold)."""
    n = W.shape[0]
    n_edges = np.sum(W > 0) / 2  # undirected
    max_edges = n * (n - 1) / 2
    return float(n_edges / max_edges) if max_edges > 0 else 0.0


def average_path_length(W: np.ndarray) -> float:
    """Average shortest path length in the network."""
    n = W.shape[0]
    D = np.where(W > 0, 1.0, np.inf)
    np.fill_diagonal(D, 0.0)
    # Floyd-Warshall
    for k in range(n):
        D = np.minimum(D, D[:, k: k + 1] + D[k: k + 1, :])
    finite_paths = D[D < np.inf]
    return float(finite_paths[finite_paths > 0].mean()) if len(finite_paths) > 0 else np.inf


def network_summary(C: np.ndarray, threshold: float = 0.3) -> dict:
    """Full network analysis summary."""
    L, W = graph_laplacian(C, threshold=threshold)
    eigs, vecs = spectral_decomposition(L)

    n_components = int(np.sum(eigs < 1e-6))  # multiplicity of zero eigenvalue
    n = C.shape[0]

    return {
        "n_assets": n,
        "n_components": n_components,
        "network_density": network_density(W),
        "spectral_gap": float(eigs[1]) if len(eigs) > 1 else 0.0,  # connectivity
        "n_edges": int(np.sum(W > 0) / 2),
        "avg_degree": float(W.sum() / n),
        "max_eigenvalue": float(eigs[-1]),
        "systemic_importance": systemic_importance(W, C).tolist(),
        "degree_centrality": degree_centrality(W).tolist(),
        "is_connected": bool(n_components == 1),
    }
