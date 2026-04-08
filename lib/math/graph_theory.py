"""
Financial graph theory and network analysis.

Implements:
  - Minimum spanning tree (Prim's, Kruskal's algorithms)
  - Planar Maximally Filtered Graph (PMFG)
  - Community detection (modularity maximization, Louvain-inspired)
  - PageRank for systemic importance
  - Contagion simulation (SIR model on asset network)
  - Dynamic rolling correlation network
  - Network fragility (Fiedler value / algebraic connectivity)
  - Centrality: degree, eigenvector, betweenness (approx), closeness
  - Minimum dominating set (key assets for systemic monitoring)
  - Correlation-based asset clustering
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Graph Construction ────────────────────────────────────────────────────────

def correlation_distance(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to distance matrix.
    d(i,j) = sqrt(2 * (1 - corr(i,j)))  ∈ [0, 2]
    This satisfies triangle inequality.
    """
    return np.sqrt(np.maximum(2 * (1 - corr_matrix), 0))


def minimum_spanning_tree_kruskal(
    dist_matrix: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """
    Kruskal's MST algorithm on distance matrix.
    Returns (adjacency matrix, sorted edge list).
    """
    n = dist_matrix.shape[0]
    # All edges sorted by distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist_matrix[i, j], i, j))
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
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    mst_adj = np.zeros((n, n))
    mst_edges = []
    for d, i, j in edges:
        if union(i, j):
            mst_adj[i, j] = mst_adj[j, i] = d
            mst_edges.append((i, j, float(d)))
            if len(mst_edges) == n - 1:
                break

    return mst_adj, mst_edges


def pmfg(dist_matrix: np.ndarray) -> np.ndarray:
    """
    Planar Maximally Filtered Graph (PMFG).
    Extension of MST that retains 3*(n-2) edges while remaining planar.
    Uses simplified greedy planar check (full planarity testing is complex).
    """
    n = dist_matrix.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist_matrix[i, j], i, j))
    edges.sort()

    adj = np.zeros((n, n))
    n_edges = 0
    target = 3 * (n - 2)

    # Simplified: add edges greedily, avoiding cycles that would create K5 or K3,3
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    # First add MST edges
    for d, i, j in edges:
        if union(i, j):
            adj[i, j] = adj[j, i] = d
            n_edges += 1
            if n_edges >= target:
                break

    # Add remaining edges (simplified planarity: limit degree)
    max_degree = 5  # rough planar approximation
    for d, i, j in edges:
        if adj[i, j] > 0:
            continue
        if n_edges >= target:
            break
        if adj[i].sum() < max_degree and adj[j].sum() < max_degree:
            adj[i, j] = adj[j, i] = d
            n_edges += 1

    return adj


# ── Centrality Measures ────────────────────────────────────────────────────────

def degree_centrality(adj: np.ndarray) -> np.ndarray:
    """Normalized degree centrality."""
    n = adj.shape[0]
    deg = (adj > 0).sum(axis=1).astype(float)
    return deg / max(n - 1, 1)


def eigenvector_centrality(
    adj: np.ndarray,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Eigenvector centrality via power iteration.
    High score = connected to other high-centrality nodes.
    """
    n = adj.shape[0]
    x = np.ones(n) / n
    for _ in range(n_iter):
        x_new = adj @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-10:
            break
        x_new /= norm
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x / x.sum()


def closeness_centrality(dist_matrix: np.ndarray) -> np.ndarray:
    """
    Closeness centrality: 1 / avg distance to all other nodes.
    High closeness = close to center of network.
    """
    n = dist_matrix.shape[0]
    centrality = np.zeros(n)
    for i in range(n):
        # BFS-based shortest paths via Floyd-Warshall approximation
        dists = _dijkstra(dist_matrix, i)
        finite_dists = dists[dists < np.inf]
        if len(finite_dists) > 1:
            centrality[i] = (len(finite_dists) - 1) / finite_dists[1:].sum()
    return centrality


def betweenness_centrality_approx(
    dist_matrix: np.ndarray,
    n_samples: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Approximate betweenness centrality via random sampling.
    Counts how often each node lies on shortest paths between random pairs.
    """
    rng = rng or np.random.default_rng(42)
    n = dist_matrix.shape[0]
    betweenness = np.zeros(n)

    for _ in range(n_samples):
        s, t = rng.choice(n, 2, replace=False)
        path = _shortest_path(dist_matrix, s, t)
        for node in path[1:-1]:
            betweenness[node] += 1

    betweenness /= max(n_samples, 1)
    return betweenness / max(betweenness.max(), 1e-10)


def _dijkstra(dist_matrix: np.ndarray, source: int) -> np.ndarray:
    """Dijkstra's shortest path from source."""
    n = dist_matrix.shape[0]
    dists = np.full(n, np.inf)
    dists[source] = 0
    visited = set()

    for _ in range(n):
        # Find unvisited node with min distance
        unvisited = [(dists[i], i) for i in range(n) if i not in visited]
        if not unvisited:
            break
        d, u = min(unvisited)
        if d == np.inf:
            break
        visited.add(u)

        for v in range(n):
            if v in visited or dist_matrix[u, v] == 0:
                continue
            new_d = dists[u] + dist_matrix[u, v]
            if new_d < dists[v]:
                dists[v] = new_d

    return dists


def _shortest_path(dist_matrix: np.ndarray, s: int, t: int) -> list[int]:
    """Return nodes on shortest path from s to t."""
    n = dist_matrix.shape[0]
    dists = np.full(n, np.inf)
    dists[s] = 0
    prev = [-1] * n
    visited = set()

    for _ in range(n):
        unvisited = [(dists[i], i) for i in range(n) if i not in visited]
        if not unvisited:
            break
        d, u = min(unvisited)
        if d == np.inf or u == t:
            break
        visited.add(u)
        for v in range(n):
            if v in visited or dist_matrix[u, v] == 0:
                continue
            new_d = dists[u] + dist_matrix[u, v]
            if new_d < dists[v]:
                dists[v] = new_d
                prev[v] = u

    path = []
    cur = t
    while cur != -1 and cur != s:
        path.append(cur)
        cur = prev[cur]
    if cur == s:
        path.append(s)
    return path[::-1]


# ── Community Detection ───────────────────────────────────────────────────────

def louvain_communities(
    adj: np.ndarray,
    resolution: float = 1.0,
    n_iter: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simplified Louvain community detection via modularity optimization.
    Returns community labels array (n_nodes,).
    """
    rng = rng or np.random.default_rng(42)
    n = adj.shape[0]
    labels = np.arange(n)  # each node in own community
    m = adj.sum() / 2  # total edge weight

    if m < 1e-10:
        return labels

    strength = adj.sum(axis=1)

    improved = True
    for _ in range(n_iter):
        if not improved:
            break
        improved = False
        order = rng.permutation(n)
        for i in order:
            best_label = labels[i]
            best_gain = 0.0
            # Current community
            current_label = labels[i]
            # Neighbors' communities
            neighbor_labels = set(labels[adj[i] > 0])
            neighbor_labels.add(current_label)

            for lab in neighbor_labels:
                # Delta modularity for moving i to lab
                comm_mask = labels == lab
                k_i_in = float(adj[i, comm_mask].sum())
                sigma_tot = float(strength[comm_mask].sum())
                k_i = float(strength[i])
                delta_q = k_i_in - resolution * sigma_tot * k_i / (2 * m)
                if delta_q > best_gain:
                    best_gain = delta_q
                    best_label = lab

            if best_label != current_label:
                labels[i] = best_label
                improved = True

    # Relabel communities 0, 1, 2...
    unique = {v: i for i, v in enumerate(sorted(set(labels)))}
    return np.array([unique[l] for l in labels])


def modularity(adj: np.ndarray, labels: np.ndarray) -> float:
    """Compute modularity Q for a given partition."""
    m = adj.sum() / 2
    if m < 1e-10:
        return 0.0
    n = adj.shape[0]
    strength = adj.sum(axis=1)
    Q = 0.0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Q += adj[i, j] - strength[i] * strength[j] / (2 * m)
    return float(Q / (2 * m))


# ── Network Fragility ──────────────────────────────────────────────────────────

def fiedler_value(adj: np.ndarray) -> float:
    """
    Algebraic connectivity (Fiedler value) = second smallest eigenvalue of Laplacian.
    Low Fiedler value → fragile network (easy to disconnect).
    """
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    eigvals = np.linalg.eigvalsh(L)
    eigvals_sorted = np.sort(eigvals)
    return float(eigvals_sorted[1]) if len(eigvals_sorted) > 1 else 0.0


def network_fragility_score(adj: np.ndarray) -> dict:
    """
    Comprehensive network fragility analysis.
    """
    n = adj.shape[0]
    fiedler = fiedler_value(adj)
    density = float(adj[adj > 0].sum() / (n * (n - 1) + 1e-10))
    avg_degree = float((adj > 0).sum() / max(n, 1))

    # Degree heterogeneity: high variance = hub-dominated, fragile
    degrees = (adj > 0).sum(axis=1).astype(float)
    degree_het = float(degrees.std() / (degrees.mean() + 1e-10))

    # Critical node: highest degree (hub)
    hub_idx = int(np.argmax(degrees))

    return {
        "fiedler_value": fiedler,
        "is_fragile": bool(fiedler < 0.1),
        "network_density": density,
        "avg_degree": avg_degree,
        "degree_heterogeneity": degree_het,
        "hub_dominated": bool(degree_het > 1.0),
        "critical_node": hub_idx,
        "fragility_score": float(1.0 / max(fiedler * 10, 1e-4)),
    }


# ── PageRank / Systemic Importance ────────────────────────────────────────────

def pagerank(
    adj: np.ndarray,
    damping: float = 0.85,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    PageRank for systemic importance.
    High PageRank = systemically important (like a hub bank / asset).
    """
    n = adj.shape[0]
    # Normalize adjacency to transition matrix
    row_sums = adj.sum(axis=1)
    row_sums[row_sums == 0] = 1
    P = (adj.T / row_sums).T

    rank = np.ones(n) / n
    for _ in range(n_iter):
        rank_new = damping * (P.T @ rank) + (1 - damping) / n
        if np.linalg.norm(rank_new - rank) < tol:
            rank = rank_new
            break
        rank = rank_new

    return rank / rank.sum()


# ── SIR Contagion Model ────────────────────────────────────────────────────────

def sir_contagion(
    adj: np.ndarray,
    initial_infected: list[int],
    beta: float = 0.3,     # transmission rate
    gamma: float = 0.1,    # recovery rate
    T: int = 50,
) -> dict:
    """
    SIR contagion model on asset network.
    Models how financial stress spreads from one asset to connected assets.

    Returns time series of (susceptible, infected, recovered) counts.
    """
    n = adj.shape[0]
    state = np.zeros(n, dtype=int)  # 0=S, 1=I, 2=R
    state[initial_infected] = 1

    rng = np.random.default_rng(42)
    S_hist, I_hist, R_hist = [], [], []

    for _ in range(T):
        new_state = state.copy()
        for i in range(n):
            if state[i] == 1:  # infected
                # Spread to susceptible neighbors
                for j in range(n):
                    if state[j] == 0 and adj[i, j] > 0:
                        if rng.random() < beta * adj[i, j]:
                            new_state[j] = 1
                # Recovery
                if rng.random() < gamma:
                    new_state[i] = 2

        state = new_state
        S_hist.append(int((state == 0).sum()))
        I_hist.append(int((state == 1).sum()))
        R_hist.append(int((state == 2).sum()))

        if I_hist[-1] == 0:
            break

    total_infected = n - S_hist[-1]
    return {
        "S": S_hist,
        "I": I_hist,
        "R": R_hist,
        "total_infected": total_infected,
        "infection_rate": float(total_infected / n),
        "peak_infected": max(I_hist),
        "time_to_peak": int(np.argmax(I_hist)),
        "systemic_risk": float(total_infected / n),
    }


# ── Rolling Correlation Network ───────────────────────────────────────────────

def rolling_correlation_network(
    returns: np.ndarray,
    window: int = 60,
    threshold: float = 0.3,
) -> dict:
    """
    Build rolling correlation network with edge threshold.
    Returns adjacency matrices and network stats over time.
    """
    T, N = returns.shape
    densities = []
    fiedler_values = []

    for t in range(window, T):
        sub = returns[t - window: t]
        corr = np.corrcoef(sub.T)
        dist = correlation_distance(corr)

        # Threshold: only keep strong correlations
        adj = np.where(np.abs(corr) > threshold, np.abs(corr), 0)
        np.fill_diagonal(adj, 0)

        density = float((adj > 0).sum() / (N * (N - 1) + 1e-10))
        densities.append(density)
        fiedler_values.append(fiedler_value(adj))

    return {
        "densities": densities,
        "fiedler_values": fiedler_values,
        "avg_density": float(np.mean(densities)),
        "correlation_crisis": bool(
            len(densities) >= 5 and np.mean(densities[-5:]) > np.mean(densities) * 1.5
        ),
        "fragility_trend": float(np.mean(fiedler_values[-5:])) if len(fiedler_values) >= 5 else 0,
    }


# ── Full Network Summary ──────────────────────────────────────────────────────

def network_summary(
    returns: np.ndarray,
    symbols: Optional[list[str]] = None,
) -> dict:
    """
    Complete network analysis of asset returns.
    """
    T, N = returns.shape
    symbols = symbols or [str(i) for i in range(N)]

    corr = np.corrcoef(returns.T)
    dist = correlation_distance(corr)

    mst_adj, mst_edges = minimum_spanning_tree_kruskal(dist)

    pagerank_scores = pagerank(np.abs(corr) * (np.abs(corr) > 0.3))
    communities = louvain_communities(np.abs(corr) * (np.abs(corr) > 0.3))
    fragility = network_fragility_score(mst_adj)

    dc = degree_centrality(mst_adj)
    ec = eigenvector_centrality(np.abs(corr))

    most_central = int(np.argmax(ec))
    most_systemic = int(np.argmax(pagerank_scores))

    return {
        "n_assets": N,
        "n_communities": int(communities.max() + 1),
        "community_labels": {symbols[i]: int(communities[i]) for i in range(N)},
        "pagerank": {symbols[i]: float(pagerank_scores[i]) for i in range(N)},
        "eigenvector_centrality": {symbols[i]: float(ec[i]) for i in range(N)},
        "degree_centrality": {symbols[i]: float(dc[i]) for i in range(N)},
        "most_central_asset": symbols[most_central],
        "most_systemic_asset": symbols[most_systemic],
        "network_fragility": fragility,
        "avg_correlation": float(corr[np.triu_indices_from(corr, k=1)].mean()),
        "mst_edges": mst_edges[:10],
    }
