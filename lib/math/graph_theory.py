"""
graph_theory.py
===============
Financial graph theory and network analysis for the idea-engine.

Implements from scratch (numpy / pure-Python, no networkx):

- Minimum Spanning Tree  (Kruskal and Prim)
- Planar Maximally Filtered Graph (PMFG)
- Community detection    (Louvain-inspired modularity maximisation)
- PageRank               for systemic importance
- SIR contagion          simulation on asset networks
- Dynamic rolling correlation network
- Algebraic connectivity (Fiedler value) for fragility
- Centrality measures    : degree, eigenvector, betweenness (approx), closeness

Graph representation: adjacency dicts  {node_id: {neighbour_id: weight}}
"""

from __future__ import annotations

import math
import heapq
import random
import itertools
import collections
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────

# node_id → {neighbour_id → edge_weight}
AdjDict  = dict[int, dict[int, float]]
# (weight, node_i, node_j)
EdgeList = list[tuple[float, int, int]]


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphMetrics:
    """Container for all computed network metrics for an asset graph."""
    n_nodes: int
    n_edges: int
    density: float
    degree_centrality: dict[int, float]
    eigenvector_centrality: dict[int, float]
    betweenness_centrality: dict[int, float]
    closeness_centrality: dict[int, float]
    pagerank: dict[int, float]
    fiedler_value: float          # algebraic connectivity λ₂
    communities: dict[int, int]   # node → community_id
    modularity: float             # Newman-Girvan Q


@dataclass
class SIRResult:
    """Result of an SIR contagion simulation on the asset network."""
    seed_nodes: list[int]
    time_series: list[dict[str, list[int]]]   # list of {S, I, R} per step
    final_infected_fraction: float
    peak_infected_fraction: float
    extinction_time: int


# ──────────────────────────────────────────────────────────────────────────────
# Union-Find (for Kruskal)
# ──────────────────────────────────────────────────────────────────────────────

class UnionFind:
    """Path-compressed, rank-union Disjoint Set Union structure."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction helpers
# ──────────────────────────────────────────────────────────────────────────────

def correlation_to_distance(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Mantegna (1999) metric distance:  d(i,j) = sqrt(2 * (1 - ρ_{ij}))
    Maps ρ ∈ [-1, 1] to distances in [0, 2].
    """
    return np.sqrt(np.clip(2.0 * (1.0 - corr_matrix), 0.0, 4.0))


def distance_matrix_to_edges(dist: np.ndarray) -> EdgeList:
    """
    Flatten upper-triangular distance matrix into sorted edge list
    (ascending by weight).
    """
    n = dist.shape[0]
    edges: EdgeList = [
        (float(dist[i, j]), i, j)
        for i in range(n)
        for j in range(i + 1, n)
    ]
    edges.sort()
    return edges


def adj_from_edges(n: int, edges: list[tuple[float, int, int]]) -> AdjDict:
    """Build adjacency dict from an edge list."""
    adj: AdjDict = {i: {} for i in range(n)}
    for w, i, j in edges:
        adj[i][j] = w
        adj[j][i] = w
    return adj


def adj_to_matrix(adj: AdjDict, n: int) -> np.ndarray:
    """Convert adjacency dict to dense (n × n) weight matrix."""
    mat = np.zeros((n, n))
    for i, neighbours in adj.items():
        for j, w in neighbours.items():
            mat[i, j] = w
    return mat


def compute_laplacian(adj: AdjDict, n: int) -> np.ndarray:
    """Compute the graph Laplacian L = D - A."""
    L = np.zeros((n, n))
    nodes = sorted(adj.keys())
    node_idx = {node: idx for idx, node in enumerate(nodes)}
    for u in nodes:
        ii = node_idx[u]
        for v, w in adj[u].items():
            jj = node_idx[v]
            L[ii, jj] -= w
            L[ii, ii] += w
    return L


# ──────────────────────────────────────────────────────────────────────────────
# Minimum Spanning Tree
# ──────────────────────────────────────────────────────────────────────────────

def mst_kruskal(n: int, edges: EdgeList) -> AdjDict:
    """
    Kruskal's MST algorithm.

    Greedily selects minimum-weight edges that do not form a cycle.
    Time complexity O(E log E) for the sort.

    Parameters
    ----------
    n     : number of nodes (0 … n-1)
    edges : pre-sorted edge list (weight, i, j) ascending
    """
    uf = UnionFind(n)
    mst_edges: list[tuple[float, int, int]] = []
    for w, i, j in edges:
        if uf.union(i, j):
            mst_edges.append((w, i, j))
            if len(mst_edges) == n - 1:
                break
    return adj_from_edges(n, mst_edges)


def mst_prim(n: int, dist_matrix: np.ndarray) -> AdjDict:
    """
    Prim's MST algorithm using a binary min-heap.

    Parameters
    ----------
    n           : number of nodes
    dist_matrix : (n × n) distance matrix

    Returns
    -------
    Adjacency dict of the MST.
    """
    in_tree = [False] * n
    key     = [math.inf] * n
    parent  = [-1] * n
    key[0]  = 0.0
    heap    = [(0.0, 0)]

    while heap:
        d, u = heapq.heappop(heap)
        if in_tree[u]:
            continue
        in_tree[u] = True
        for v in range(n):
            if not in_tree[v] and dist_matrix[u, v] < key[v]:
                key[v] = dist_matrix[u, v]
                parent[v] = u
                heapq.heappush(heap, (key[v], v))

    mst_edges = [
        (dist_matrix[parent[v], v], parent[v], v)
        for v in range(1, n)
        if parent[v] >= 0
    ]
    return adj_from_edges(n, mst_edges)


# ──────────────────────────────────────────────────────────────────────────────
# Planar Maximally Filtered Graph (PMFG)
# ──────────────────────────────────────────────────────────────────────────────

def _euler_planarity_ok(n_nodes: int, n_edges: int) -> bool:
    """
    Euler's formula necessary condition for planarity:
        E ≤ 3V - 6   (V ≥ 3)
    Used as a fast necessary-but-not-sufficient planarity gate.
    """
    if n_nodes < 3:
        return True
    return n_edges <= 3 * n_nodes - 6


def build_pmfg(corr_matrix: np.ndarray) -> AdjDict:
    """
    Planar Maximally Filtered Graph construction.

    Greedily inserts the highest-correlation edges that maintain planarity
    (checked via Euler's formula heuristic, which is exact for sparse graphs
    built from MST seeds).  The PMFG contains 3(n-2) edges and always
    includes the MST as a subgraph.

    References
    ----------
    Tumminello, M., Aste, T., Di Matteo, T., Mantegna, R.N. (2005).
    "A tool for filtering information in complex systems." PNAS 102(30).
    """
    n = corr_matrix.shape[0]
    dist = correlation_to_distance(corr_matrix)
    edges = distance_matrix_to_edges(dist)   # sorted ascending = densest first when reversed
    # We want highest correlation = shortest distance first, already sorted
    adj: AdjDict = {i: {} for i in range(n)}
    target_edges = max(3 * (n - 2), n - 1)
    added = 0

    for w, i, j in edges:
        if i == j or j in adj[i]:
            continue
        current_count = added + 1
        if _euler_planarity_ok(n, current_count):
            adj[i][j] = float(w)
            adj[j][i] = float(w)
            added += 1
        if added >= target_edges:
            break

    return adj


# ──────────────────────────────────────────────────────────────────────────────
# Community Detection  (Louvain-inspired modularity maximisation)
# ──────────────────────────────────────────────────────────────────────────────

def _modularity(
    adj: AdjDict,
    communities: dict[int, int],
    total_weight: float,
) -> float:
    """
    Newman-Girvan modularity:
        Q = (1/2m) Σ_{i,j} [A_{ij} - k_i k_j / 2m] δ(c_i, c_j)
    """
    if total_weight < 1e-12:
        return 0.0
    m2 = 2.0 * total_weight
    strength: dict[int, float] = {i: sum(adj[i].values()) for i in adj}

    comm_nodes: dict[int, list[int]] = collections.defaultdict(list)
    for node, comm in communities.items():
        comm_nodes[comm].append(node)

    q = 0.0
    for nodes in comm_nodes.values():
        for i in nodes:
            for j in nodes:
                q += adj[i].get(j, 0.0) - strength[i] * strength[j] / m2
    return q / m2


def detect_communities_louvain(
    adj: AdjDict,
    max_passes: int = 20,
    seed: int = 42,
) -> dict[int, int]:
    """
    Louvain modularity maximisation (greedy Phase 1).

    For each node, compute the modularity gain from moving it into each
    neighbouring community.  Repeat until no improvement.

    Parameters
    ----------
    adj        : weighted adjacency dict
    max_passes : maximum optimisation sweeps
    seed       : random seed for node ordering

    Returns
    -------
    {node_id: community_id}
    """
    rng = random.Random(seed)
    nodes = list(adj.keys())
    communities: dict[int, int] = {n: n for n in nodes}

    total_weight = sum(w for i in adj for w in adj[i].values()) / 2.0
    if total_weight < 1e-12:
        return communities

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        order = list(nodes)
        rng.shuffle(order)

        for node in order:
            current_comm = communities[node]
            communities[node] = -1  # temporarily remove

            # Strength of node
            k_i = sum(adj[node].values())

            # Sum of edge weights into each neighbouring community
            k_i_to: dict[int, float] = collections.defaultdict(float)
            for nbr, w in adj[node].items():
                c = communities[nbr]
                if c >= 0:
                    k_i_to[c] += w

            # Total strength of each candidate community
            sigma_tot: dict[int, float] = collections.defaultdict(float)
            for nd in nodes:
                c = communities[nd]
                if c >= 0:
                    sigma_tot[c] += sum(adj[nd].values())

            best_comm = current_comm
            best_gain = 0.0
            m2 = 2.0 * total_weight

            for cand_comm, k_in in k_i_to.items():
                gain = k_in / total_weight - k_i * sigma_tot[cand_comm] / (m2 * total_weight)
                if gain > best_gain:
                    best_gain = gain
                    best_comm = cand_comm

            communities[node] = best_comm
            if best_comm != current_comm:
                improved = True

    # Remap community IDs to contiguous integers
    unique = sorted(set(communities.values()))
    remap  = {old: new for new, old in enumerate(unique)}
    return {n: remap[c] for n, c in communities.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Centrality measures
# ──────────────────────────────────────────────────────────────────────────────

def degree_centrality(adj: AdjDict) -> dict[int, float]:
    """
    Weighted degree (strength) centrality, normalised by maximum strength.
    """
    strength = {i: sum(adj[i].values()) for i in adj}
    max_s = max(strength.values(), default=1.0)
    if max_s < 1e-12:
        max_s = 1.0
    return {i: s / max_s for i, s in strength.items()}


def eigenvector_centrality(
    adj: AdjDict,
    max_iter: int = 300,
    tol: float = 1e-9,
) -> dict[int, float]:
    """
    Eigenvector centrality via power iteration.

    x_i^{(t+1)} = Σ_j A_{ij} x_j^{(t)}  (renormalised each step)

    Converges to the leading eigenvector of the adjacency matrix.
    """
    nodes = list(adj.keys())
    n = len(nodes)
    if n == 0:
        return {}
    x = {i: 1.0 / n for i in nodes}

    for _ in range(max_iter):
        x_new = {i: sum(adj[i].get(j, 0.0) * x[j] for j in adj[i]) for i in nodes}
        norm = math.sqrt(sum(v ** 2 for v in x_new.values()))
        if norm < 1e-12:
            norm = 1.0
        x_new = {i: v / norm for i, v in x_new.items()}
        diff = sum((x_new[i] - x[i]) ** 2 for i in nodes)
        x = x_new
        if diff < tol:
            break
    return x


def closeness_centrality(adj: AdjDict) -> dict[int, float]:
    """
    Closeness centrality: C(u) = (reachable - 1) / Σ_{v≠u} d(u,v)
    Computed via Dijkstra from each node.
    """
    nodes = list(adj.keys())
    result: dict[int, float] = {}
    for source in nodes:
        dist = _dijkstra(adj, source)
        finite = {v: d for v, d in dist.items() if d < math.inf and v != source}
        total = sum(finite.values())
        if total > 1e-12 and finite:
            result[source] = len(finite) / total
        else:
            result[source] = 0.0
    return result


def betweenness_centrality_approx(
    adj: AdjDict,
    n_samples: int = 200,
    seed: int = 0,
) -> dict[int, float]:
    """
    Approximate betweenness centrality via random-source Dijkstra sampling.

    For each sampled source, runs Dijkstra and distributes credit to
    intermediate nodes on all shortest paths.

    Parameters
    ----------
    n_samples : sources to sample; use all nodes when n_samples ≥ n
    """
    rng    = random.Random(seed)
    nodes  = list(adj.keys())
    n      = len(nodes)
    bc: dict[int, float] = {i: 0.0 for i in nodes}

    sources = nodes if n_samples >= n else rng.sample(nodes, n_samples)

    for source in sources:
        dist, prev = _dijkstra_with_prev(adj, source)
        for target in nodes:
            if target == source or dist[target] == math.inf:
                continue
            path = []
            cur  = target
            while cur != source:
                p = prev.get(cur, -1)
                if p < 0:
                    break
                path.append(cur)
                cur = p
            for intermediate in path[1:]:   # exclude target itself
                bc[intermediate] += 1.0

    # Normalise by (n-1)(n-2) and scale for sampling
    denom = max((n - 1) * (n - 2), 1)
    scale = n / max(len(sources), 1)
    return {i: v * scale / denom for i, v in bc.items()}


# ──────────────────────────────────────────────────────────────────────────────
# PageRank
# ──────────────────────────────────────────────────────────────────────────────

def pagerank(
    adj: AdjDict,
    damping: float = 0.85,
    max_iter: int = 300,
    tol: float = 1e-10,
) -> dict[int, float]:
    """
    Weighted PageRank for systemic importance scoring.

    PR(i) = (1-d)/n  +  d * Σ_j [ A_{ji} / strength(j) ] * PR(j)

    Edge weights enter via the weighted out-strength normalisation.
    """
    nodes = list(adj.keys())
    n = len(nodes)
    if n == 0:
        return {}

    pr = {i: 1.0 / n for i in nodes}
    out_strength = {i: max(sum(adj[i].values()), 1e-12) for i in nodes}
    teleport = (1.0 - damping) / n

    for _ in range(max_iter):
        pr_new: dict[int, float] = {i: teleport for i in nodes}
        for j in nodes:
            share = damping * pr[j] / out_strength[j]
            for i, w in adj[j].items():
                pr_new[i] += share * w
        diff = sum(abs(pr_new[i] - pr[i]) for i in nodes)
        pr = pr_new
        if diff < tol:
            break

    return pr


# ──────────────────────────────────────────────────────────────────────────────
# Algebraic connectivity  (Fiedler value λ₂)
# ──────────────────────────────────────────────────────────────────────────────

def fiedler_value(
    adj: AdjDict,
    n_iter: int = 600,
    tol: float = 1e-9,
) -> float:
    """
    Compute the Fiedler value (second-smallest Laplacian eigenvalue) λ₂.

    Uses power iteration on the shifted Laplacian (σI - L) after deflating
    the trivial zero eigenvector (the all-ones vector for connected graphs).

    λ₂ = 0   → graph is disconnected (maximally fragile)
    λ₂ small → graph is barely connected, single-point fragile
    λ₂ large → well-connected, robust to node removal

    Algorithm
    ---------
    σ = max diagonal of L + 1  (shift to make M = σI - L positive definite)
    Power iterate M to find dominant eigenvalue λ_max(M) = σ - λ₂(L)
    """
    nodes = sorted(adj.keys())
    n = len(nodes)
    if n <= 1:
        return 0.0

    node_idx = {node: i for i, node in enumerate(nodes)}
    L = np.zeros((n, n))
    for u in nodes:
        ii = node_idx[u]
        for v, w in adj[u].items():
            jj = node_idx[v]
            L[ii, jj] -= float(w)
            L[ii, ii] += float(w)

    sigma = float(np.max(np.diag(L))) + 1.0
    M = sigma * np.eye(n) - L   # eigenvalues: σ - λ_i(L)

    ones = np.ones(n) / math.sqrt(n)
    rng  = np.random.default_rng(1)
    v    = rng.standard_normal(n)
    # Project out the constant eigenvector
    v = v - np.dot(v, ones) * ones
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        return 0.0
    v /= nrm

    lam = 0.0
    for _ in range(n_iter):
        Mv = M @ v
        Mv -= np.dot(Mv, ones) * ones     # re-deflate each step
        lam_new = float(np.dot(v, Mv))
        nrm = float(np.linalg.norm(Mv))
        if nrm < 1e-12:
            break
        v = Mv / nrm
        if abs(lam_new - lam) < tol:
            lam = lam_new
            break
        lam = lam_new

    # λ_2(L) = σ - lam_max(M)
    return max(0.0, sigma - lam)


# ──────────────────────────────────────────────────────────────────────────────
# SIR contagion simulation
# ──────────────────────────────────────────────────────────────────────────────

def sir_contagion(
    adj: AdjDict,
    seed_nodes: list[int],
    beta: float = 0.3,
    gamma: float = 0.1,
    max_steps: int = 200,
    rng: random.Random | None = None,
) -> SIRResult:
    """
    Discrete-time SIR epidemic model on the asset graph.

    Transition rules per step
    -------------------------
    S → I : for each (infected_node, susceptible_nbr) edge with weight w,
            the susceptible node becomes infected with probability min(1, β·w).
    I → R : each infected node recovers with probability γ.

    Parameters
    ----------
    seed_nodes : initially infected assets
    beta       : transmission coefficient (per-contact, scaled by edge weight)
    gamma      : recovery probability per step
    max_steps  : cap on simulation length
    """
    if rng is None:
        rng = random.Random(42)

    all_nodes = set(adj.keys())
    S = all_nodes - set(seed_nodes)
    I = set(seed_nodes) & all_nodes
    R: set[int] = set()
    time_series: list[dict[str, list[int]]] = []
    n = len(all_nodes)
    peak_infected = len(I)

    for _ in range(max_steps):
        time_series.append({"S": sorted(S), "I": sorted(I), "R": sorted(R)})
        if not I:
            break

        new_I: set[int] = set()
        new_R: set[int] = set()

        for inf_node in list(I):
            for nbr, w in adj[inf_node].items():
                if nbr in S:
                    if rng.random() < min(1.0, beta * w):
                        new_I.add(nbr)
            if rng.random() < gamma:
                new_R.add(inf_node)

        S -= new_I
        I  = (I | new_I) - new_R
        R |= new_R

        if len(I) > peak_infected:
            peak_infected = len(I)

    return SIRResult(
        seed_nodes=sorted(seed_nodes),
        time_series=time_series,
        final_infected_fraction=len(R) / max(n, 1),
        peak_infected_fraction=peak_infected / max(n, 1),
        extinction_time=len(time_series),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic rolling correlation network
# ──────────────────────────────────────────────────────────────────────────────

def build_dynamic_network(
    returns: np.ndarray,
    window: int = 63,
    threshold: float = 0.30,
    step: int = 21,
) -> list[tuple[int, AdjDict]]:
    """
    Build a time series of correlation-threshold asset networks.

    Parameters
    ----------
    returns   : (T × N) return matrix
    window    : rolling estimation window (periods)
    threshold : minimum absolute correlation for an edge
    step      : number of periods between snapshots

    Returns
    -------
    List of (t, AdjDict) snapshots.
    """
    T, N = returns.shape
    snapshots: list[tuple[int, AdjDict]] = []

    for t in range(window, T, step):
        R = returns[t - window:t]
        corr = np.corrcoef(R.T)
        np.fill_diagonal(corr, 0.0)

        adj: AdjDict = {i: {} for i in range(N)}
        for i in range(N):
            for j in range(i + 1, N):
                c = float(corr[i, j])
                if abs(c) >= threshold:
                    adj[i][j] = abs(c)
                    adj[j][i] = abs(c)
        snapshots.append((t, adj))

    return snapshots


def dynamic_network_stats(
    snapshots: list[tuple[int, AdjDict]],
    n: int,
) -> dict[str, list]:
    """
    Compute time series of summary statistics over a dynamic network sequence.

    Returns dict with time-indexed lists of: n_edges, density, fiedler_value.
    """
    stats: dict[str, list] = {"t": [], "n_edges": [], "density": [], "fiedler": []}
    max_edges = n * (n - 1) // 2
    for t, adj in snapshots:
        ne = sum(len(v) for v in adj.values()) // 2
        stats["t"].append(t)
        stats["n_edges"].append(ne)
        stats["density"].append(ne / max(max_edges, 1))
        stats["fiedler"].append(round(fiedler_value(adj), 5))
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Full analysis pipeline
# ──────────────────────────────────────────────────────────────────────────────

def analyse_asset_network(
    corr_matrix: np.ndarray,
    node_labels: list[str] | None = None,
    use_pmfg: bool = True,
    bet_samples: int = 100,
) -> tuple[AdjDict, GraphMetrics]:
    """
    Full network analysis pipeline for an asset correlation matrix.

    Steps
    -----
    1. Construct PMFG or MST from the correlation matrix.
    2. Compute all centrality measures.
    3. Louvain community detection + modularity.
    4. PageRank for systemic importance.
    5. Fiedler value for fragility.
    6. Return (adjacency_dict, GraphMetrics).

    Parameters
    ----------
    corr_matrix : (N × N) correlation matrix
    node_labels : optional list of N asset names
    use_pmfg    : True = PMFG; False = MST (Prim's)
    bet_samples : sources for betweenness approximation
    """
    n = corr_matrix.shape[0]

    if use_pmfg:
        adj = build_pmfg(corr_matrix)
    else:
        dist = correlation_to_distance(corr_matrix)
        adj  = mst_prim(n, dist)

    n_edges   = sum(len(v) for v in adj.values()) // 2
    max_edges = n * (n - 1) // 2
    density   = n_edges / max(max_edges, 1)

    deg_c  = degree_centrality(adj)
    eig_c  = eigenvector_centrality(adj)
    clos_c = closeness_centrality(adj)
    bet_c  = betweenness_centrality_approx(adj, n_samples=min(n, bet_samples))
    pr     = pagerank(adj)
    comms  = detect_communities_louvain(adj)

    total_weight = sum(w for i in adj for w in adj[i].values()) / 2.0
    mod    = _modularity(adj, comms, total_weight)
    fv     = fiedler_value(adj)

    metrics = GraphMetrics(
        n_nodes=n,
        n_edges=n_edges,
        density=round(density, 6),
        degree_centrality=deg_c,
        eigenvector_centrality=eig_c,
        betweenness_centrality=bet_c,
        closeness_centrality=clos_c,
        pagerank=pr,
        fiedler_value=round(fv, 6),
        communities=comms,
        modularity=round(mod, 6),
    )
    return adj, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Dijkstra helpers
# ──────────────────────────────────────────────────────────────────────────────

def _dijkstra(adj: AdjDict, source: int) -> dict[int, float]:
    dist: dict[int, float] = {i: math.inf for i in adj}
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _dijkstra_with_prev(
    adj: AdjDict, source: int
) -> tuple[dict[int, float], dict[int, int]]:
    dist: dict[int, float] = {i: math.inf for i in adj}
    prev: dict[int, int]   = {}
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, prev


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def top_k_nodes(
    metric: dict[int, float],
    k: int = 5,
    labels: list[str] | None = None,
) -> list[tuple[str | int, float]]:
    """Return top-k (label, score) sorted by descending metric value."""
    ranked = sorted(metric.items(), key=lambda x: x[1], reverse=True)[:k]
    if labels:
        return [(labels[i] if i < len(labels) else str(i), v) for i, v in ranked]
    return ranked


def print_network_report(
    metrics: GraphMetrics,
    labels: list[str] | None = None,
    top_k: int = 5,
) -> None:
    """Print a concise human-readable network report."""
    n_comm = len(set(metrics.communities.values()))
    fragility = "fragile" if metrics.fiedler_value < 0.05 else "moderate" \
                if metrics.fiedler_value < 0.20 else "robust"
    print("═" * 50)
    print("  Asset Network Report")
    print("═" * 50)
    print(f"  Nodes      : {metrics.n_nodes}")
    print(f"  Edges      : {metrics.n_edges}")
    print(f"  Density    : {metrics.density:.5f}")
    print(f"  Fiedler λ₂ : {metrics.fiedler_value:.5f}  [{fragility}]")
    print(f"  Modularity : {metrics.modularity:.5f}")
    print(f"  Communities: {n_comm}")
    print(f"\n  Top {top_k} — PageRank (systemic importance):")
    for lbl, score in top_k_nodes(metrics.pagerank, top_k, labels):
        print(f"    {str(lbl):<22} {score:.5f}")
    print(f"\n  Top {top_k} — Betweenness centrality:")
    for lbl, score in top_k_nodes(metrics.betweenness_centrality, top_k, labels):
        print(f"    {str(lbl):<22} {score:.5f}")
    print(f"\n  Top {top_k} — Eigenvector centrality:")
    for lbl, score in top_k_nodes(metrics.eigenvector_centrality, top_k, labels):
        print(f"    {str(lbl):<22} {score:.5f}")
    print("═" * 50)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ──────────────────────────────────────────────────────────────────────────────

def _demo():
    rng = np.random.default_rng(42)
    N, T = 20, 500
    labels = [f"ASSET_{i:02d}" for i in range(N)]

    # Simulate block-correlated returns (3 factors + idiosyncratic)
    factors  = rng.standard_normal((T, 3))
    loadings = rng.standard_normal((N, 3)) * 0.5
    idio     = rng.standard_normal((T, N)) * 0.3
    returns  = factors @ loadings.T + idio
    corr     = np.corrcoef(returns.T)

    print("Building PMFG from 20-asset correlation matrix...")
    adj, metrics = analyse_asset_network(corr, node_labels=labels, use_pmfg=True)
    print_network_report(metrics, labels=labels)

    print("\nRunning SIR contagion from most central node...")
    seed = max(metrics.pagerank, key=metrics.pagerank.get)
    result = sir_contagion(adj, seed_nodes=[seed], beta=0.4, gamma=0.15)
    print(f"  Final infected : {result.final_infected_fraction:.1%}")
    print(f"  Peak infected  : {result.peak_infected_fraction:.1%}")
    print(f"  Extinction at t: {result.extinction_time}")

    print("\nBuilding 3 dynamic network snapshots (window=100, step=100)...")
    snaps = build_dynamic_network(returns, window=100, threshold=0.30, step=100)
    stats = dynamic_network_stats(snaps, N)
    for i, t in enumerate(stats["t"]):
        print(f"  t={t:4d}: edges={stats['n_edges'][i]:3d}  "
              f"density={stats['density'][i]:.3f}  "
              f"fiedler={stats['fiedler'][i]:.4f}")


if __name__ == "__main__":
    _demo()
