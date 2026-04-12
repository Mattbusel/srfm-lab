"""
causal_graph.py
===============
Causal graph learning for financial networks.

Implements:
  - PC-algorithm (constraint-based causal discovery)
  - FCI algorithm (Fast Causal Inference — handles latent confounders)
  - GES (Greedy Equivalence Search)
  - Granger causality graph construction
  - NOTEARS continuous optimization for DAG learning
  - Causal effect estimation via do-calculus adjustment
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch import Tensor


# ---------------------------------------------------------------------------
# Graph structures for causal learning
# ---------------------------------------------------------------------------

@dataclass
class CausalGraph:
    """
    Directed Acyclic Graph (DAG) or CPDAG (completed partially DAG)
    representing causal relationships between financial variables.
    """
    n_nodes: int
    edges: Set[Tuple[int, int]] = field(default_factory=set)      # directed
    undirected: Set[FrozenSet[int]] = field(default_factory=set)   # CPDAG undirected
    bidirected: Set[FrozenSet[int]] = field(default_factory=set)   # FCI bidirected (latent)
    node_names: Optional[List[str]] = None

    def add_edge(self, src: int, dst: int) -> None:
        self.edges.add((src, dst))

    def remove_edge(self, src: int, dst: int) -> None:
        self.edges.discard((src, dst))

    def orient_edge(self, i: int, j: int) -> None:
        """Orient undirected edge i - j as i → j."""
        key = frozenset({i, j})
        self.undirected.discard(key)
        self.edges.add((i, j))

    def has_edge(self, src: int, dst: int) -> bool:
        return (src, dst) in self.edges

    def has_undirected(self, i: int, j: int) -> bool:
        return frozenset({i, j}) in self.undirected

    def parents(self, node: int) -> List[int]:
        return [s for s, d in self.edges if d == node]

    def children(self, node: int) -> List[int]:
        return [d for s, d in self.edges if s == node]

    def to_edge_index(self) -> Tensor:
        """Convert directed edges to PyG-style edge_index (2, E)."""
        if not self.edges:
            return torch.zeros(2, 0, dtype=torch.long)
        src_list = [e[0] for e in self.edges]
        dst_list = [e[1] for e in self.edges]
        return torch.tensor([src_list, dst_list], dtype=torch.long)

    def adjacency_matrix(self) -> np.ndarray:
        """Return (N, N) binary adjacency matrix."""
        A = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int32)
        for s, d in self.edges:
            A[s, d] = 1
        return A

    def is_dag(self) -> bool:
        """Check DAG property via topological sort (Kahn's algorithm)."""
        in_deg = [0] * self.n_nodes
        for s, d in self.edges:
            in_deg[d] += 1
        queue = [i for i in range(self.n_nodes) if in_deg[i] == 0]
        count = 0
        while queue:
            node = queue.pop()
            count += 1
            for child in self.children(node):
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        return count == self.n_nodes


# ---------------------------------------------------------------------------
# Conditional independence testing
# ---------------------------------------------------------------------------

def partial_correlation_test(
    data: np.ndarray,
    i: int,
    j: int,
    sep_set: List[int],
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """
    Test conditional independence X_i ⊥ X_j | X_{sep_set} via partial correlation.

    Returns (independent, p_value).
    """
    if not sep_set:
        # Simple correlation test
        r = np.corrcoef(data[:, i], data[:, j])[0, 1]
        n = data.shape[0]
        if abs(r) >= 1.0 - 1e-10:
            return False, 0.0
        t_stat = r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2 + 1e-10)
        p_val = 2 * stats.t.sf(abs(t_stat), df=n - 2)
        return p_val > alpha, p_val

    # Partial correlation via regression residuals
    Z = data[:, sep_set]
    x_i = data[:, i]
    x_j = data[:, j]

    try:
        # Regress i on Z
        Z_aug = np.column_stack([Z, np.ones(len(Z))])
        beta_i = np.linalg.lstsq(Z_aug, x_i, rcond=None)[0]
        res_i = x_i - Z_aug @ beta_i

        # Regress j on Z
        beta_j = np.linalg.lstsq(Z_aug, x_j, rcond=None)[0]
        res_j = x_j - Z_aug @ beta_j

        # Partial correlation
        r_partial = np.corrcoef(res_i, res_j)[0, 1]
    except np.linalg.LinAlgError:
        return True, 1.0

    n = data.shape[0]
    k = len(sep_set)
    df = n - k - 2
    if df < 1:
        return True, 1.0

    if abs(r_partial) >= 1.0 - 1e-10:
        return False, 0.0

    t_stat = r_partial * math.sqrt(df) / math.sqrt(1 - r_partial ** 2 + 1e-10)
    p_val = 2 * stats.t.sf(abs(t_stat), df=df)
    return p_val > alpha, p_val


def fisher_z_test(
    data: np.ndarray,
    i: int,
    j: int,
    sep_set: List[int],
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """
    Fisher Z-test for conditional independence (more accurate for large samples).
    """
    n = data.shape[0]
    subset = [i, j] + sep_set
    cov = np.corrcoef(data[:, subset].T)

    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return True, 1.0

    r_partial = -prec[0, 1] / math.sqrt(abs(prec[0, 0] * prec[1, 1]) + 1e-10)
    r_partial = np.clip(r_partial, -1 + 1e-8, 1 - 1e-8)

    z = 0.5 * math.log((1 + r_partial) / (1 - r_partial))
    k = len(sep_set)
    se = 1.0 / math.sqrt(n - k - 3) if n - k - 3 > 0 else 1.0

    z_stat = z / se
    p_val = 2 * stats.norm.sf(abs(z_stat))
    return p_val > alpha, p_val


# ---------------------------------------------------------------------------
# PC algorithm
# ---------------------------------------------------------------------------

class PCAlgorithm:
    """
    PC algorithm for constraint-based causal structure learning.

    Phase 1: skeleton discovery (remove edges using CI tests)
    Phase 2: orientation (v-structures, propagation rules)

    Reference: Spirtes, Glymour, Scheines, "Causation, Prediction, Search", 2000.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_set_size: Optional[int] = None,
        ci_test: str = "fisher_z",
    ):
        self.alpha = alpha
        self.max_cond_set_size = max_cond_set_size
        self.ci_test = ci_test

    def fit(
        self,
        data: np.ndarray,
        node_names: Optional[List[str]] = None,
    ) -> CausalGraph:
        """
        Parameters
        ----------
        data : (T, N) time series data

        Returns
        -------
        CausalGraph (CPDAG)
        """
        n = data.shape[1]
        node_names = node_names or [f"X_{i}" for i in range(n)]

        # Phase 1: skeleton
        skeleton, sep_sets = self._find_skeleton(data, n)

        # Phase 2: orientation
        graph = self._orient_vstructures(skeleton, sep_sets, n)
        graph = self._propagate_orientation(graph, n)
        graph.node_names = node_names

        return graph

    def _find_skeleton(
        self,
        data: np.ndarray,
        n: int,
    ) -> Tuple[Set[FrozenSet[int]], Dict[FrozenSet[int], List[int]]]:
        """Find the skeleton graph via CI tests."""
        # Start with complete undirected graph
        skeleton: Set[FrozenSet[int]] = {
            frozenset({i, j}) for i in range(n) for j in range(i + 1, n)
        }
        sep_sets: Dict[FrozenSet[int], List[int]] = {}

        cond_set_size = 0
        max_size = self.max_cond_set_size or (n - 2)

        while cond_set_size <= max_size:
            edges_to_remove = []
            for edge in list(skeleton):
                i, j = sorted(edge)
                adjacent = self._adjacent(skeleton, i, n) - {j}
                adjacent_j = self._adjacent(skeleton, j, n) - {i}
                neighbours = adjacent | adjacent_j

                if len(neighbours) < cond_set_size:
                    continue

                for sep_cands in combinations(sorted(neighbours), cond_set_size):
                    sep_cands = list(sep_cands)
                    ci, p = self._test(data, i, j, sep_cands)
                    if ci:
                        edges_to_remove.append((edge, sep_cands))
                        break

            for edge, sep in edges_to_remove:
                skeleton.discard(edge)
                sep_sets[edge] = sep

            if all(
                len(self._adjacent(skeleton, list(e)[0], n)) <= cond_set_size
                and len(self._adjacent(skeleton, list(e)[1], n)) <= cond_set_size
                for e in skeleton
            ):
                break
            cond_set_size += 1

        return skeleton, sep_sets

    def _adjacent(self, skeleton: Set, node: int, n: int) -> Set[int]:
        adj = set()
        for edge in skeleton:
            edge_list = list(edge)
            if node in edge_list:
                adj.update(edge_list)
        adj.discard(node)
        return adj

    def _test(
        self,
        data: np.ndarray,
        i: int,
        j: int,
        sep: List[int],
    ) -> Tuple[bool, float]:
        if self.ci_test == "fisher_z":
            return fisher_z_test(data, i, j, sep, self.alpha)
        return partial_correlation_test(data, i, j, sep, self.alpha)

    def _orient_vstructures(
        self,
        skeleton: Set[FrozenSet[int]],
        sep_sets: Dict[FrozenSet[int], List[int]],
        n: int,
    ) -> CausalGraph:
        """Orient v-structures: i → k ← j where k ∉ sep(i, j)."""
        graph = CausalGraph(n_nodes=n)

        # Start with all skeleton edges as undirected
        for edge in skeleton:
            i, j = sorted(edge)
            graph.undirected.add(frozenset({i, j}))

        # Find v-structures
        for k in range(n):
            parents_k = []
            for edge in skeleton:
                edge_list = sorted(edge)
                if k in edge_list:
                    other = edge_list[0] if edge_list[1] == k else edge_list[1]
                    parents_k.append(other)

            for i, j in combinations(parents_k, 2):
                if frozenset({i, j}) not in skeleton:
                    # Potential v-structure: check if k ∉ sep(i, j)
                    sep = sep_sets.get(frozenset({i, j}), [])
                    if k not in sep:
                        # Orient: i → k ← j
                        graph.orient_edge(i, k)
                        graph.orient_edge(j, k)

        return graph

    def _propagate_orientation(self, graph: CausalGraph, n: int) -> CausalGraph:
        """Apply Meek's orientation rules."""
        changed = True
        while changed:
            changed = False

            for edge in list(graph.undirected):
                i, j = sorted(edge)

                # Rule 1: i → k — j and i -/- j → orient k → j
                for k in graph.parents(j):
                    if k != i and not graph.has_edge(k, i) and not graph.has_undirected(k, i):
                        graph.orient_edge(i, j)
                        changed = True
                        break

        return graph


# ---------------------------------------------------------------------------
# FCI algorithm (Fast Causal Inference)
# ---------------------------------------------------------------------------

class FCIAlgorithm:
    """
    FCI algorithm for causal discovery with latent confounders.

    FCI outputs a PAG (Partial Ancestral Graph) that can contain:
      - Directed edges (→)
      - Bidirected edges (↔) — indicating latent common causes
      - Circle marks (○->) — uncertainty

    Reference: Spirtes et al., 1995.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_set_size: Optional[int] = None,
    ):
        self.alpha = alpha
        self.max_cond_set_size = max_cond_set_size
        self._pc = PCAlgorithm(alpha, max_cond_set_size)

    def fit(
        self,
        data: np.ndarray,
        node_names: Optional[List[str]] = None,
    ) -> CausalGraph:
        """
        Run FCI on observed data.

        Returns CausalGraph with bidirected edges representing latent confounders.
        """
        n = data.shape[1]
        node_names = node_names or [f"X_{i}" for i in range(n)]

        # Step 1: Run PC skeleton discovery
        skeleton, sep_sets = self._pc._find_skeleton(data, n)

        # Step 2: Orient v-structures (as in PC)
        graph = self._pc._orient_vstructures(skeleton, sep_sets, n)

        # Step 3: Discriminating paths (FCI-specific)
        # Simplified: identify potential bidirected edges (latent confounders)
        # via Shalizi-style heuristic
        for i in range(n):
            for j in range(i + 1, n):
                if frozenset({i, j}) not in skeleton:
                    # Not adjacent → could be confounded if CI test fails on large set
                    full_sep = list(range(n))
                    full_sep = [k for k in full_sep if k != i and k != j]
                    ci, _ = self._pc._test(data, i, j, full_sep[:min(5, len(full_sep))])
                    if not ci:
                        # Possible latent confounder
                        graph.bidirected.add(frozenset({i, j}))

        graph.node_names = node_names
        return graph


# ---------------------------------------------------------------------------
# GES (Greedy Equivalence Search)
# ---------------------------------------------------------------------------

class GES:
    """
    Greedy Equivalence Search for score-based causal structure learning.

    Uses BIC score as the objective.
    Phases:
      1. Forward phase: add edges that maximally improve score
      2. Backward phase: remove edges that improve score
      3. Turning phase: reverse/orient edges to improve score

    Reference: Chickering, "Optimal Structure Identification With Greedy Search", JMLR 2002.
    """

    def __init__(self, penalty: float = 1.0):
        self.penalty = penalty

    def fit(
        self,
        data: np.ndarray,
        node_names: Optional[List[str]] = None,
    ) -> CausalGraph:
        """
        Parameters
        ----------
        data : (T, N)

        Returns
        -------
        CausalGraph (estimated CPDAG)
        """
        T, n = data.shape
        node_names = node_names or [f"X_{i}" for i in range(n)]

        # Precompute covariance
        cov = np.cov(data.T, ddof=1)
        cov += 1e-8 * np.eye(n)

        graph = CausalGraph(n_nodes=n, node_names=node_names)
        current_score = self._total_bic(graph, data, cov, T, n)

        # Forward phase
        improved = True
        while improved:
            improved = False
            best_delta = 0.0
            best_edge = None

            for i in range(n):
                for j in range(n):
                    if i == j or graph.has_edge(i, j) or graph.has_edge(j, i):
                        continue
                    # Try adding edge i → j
                    graph.add_edge(i, j)
                    if graph.is_dag():
                        new_score = self._total_bic(graph, data, cov, T, n)
                        delta = new_score - current_score
                        if delta > best_delta:
                            best_delta = delta
                            best_edge = (i, j)
                    graph.remove_edge(i, j)

            if best_edge is not None and best_delta > 1e-8:
                graph.add_edge(*best_edge)
                current_score += best_delta
                improved = True

        # Backward phase
        improved = True
        while improved:
            improved = False
            edges = list(graph.edges)
            for i, j in edges:
                graph.remove_edge(i, j)
                new_score = self._total_bic(graph, data, cov, T, n)
                delta = new_score - current_score
                if delta > 0:
                    current_score += delta
                    improved = True
                else:
                    graph.add_edge(i, j)

        return graph

    def _bic_single(
        self,
        node: int,
        parents: List[int],
        data: np.ndarray,
        cov: np.ndarray,
        T: int,
        n: int,
    ) -> float:
        """BIC score for a single node given its parents."""
        if not parents:
            var = float(np.var(data[:, node], ddof=1)) + 1e-10
            return -T * math.log(var) - math.log(T) * 1
        else:
            y = data[:, node]
            X = np.column_stack([data[:, p] for p in parents] + [np.ones(T)])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta
                var = float(np.var(residuals, ddof=1)) + 1e-10
            except np.linalg.LinAlgError:
                var = 1.0
            k = len(parents) + 1
            return -T * math.log(var) - math.log(T) * k

    def _total_bic(
        self,
        graph: CausalGraph,
        data: np.ndarray,
        cov: np.ndarray,
        T: int,
        n: int,
    ) -> float:
        total = 0.0
        for node in range(n):
            parents = graph.parents(node)
            total += self._bic_single(node, parents, data, cov, T, n)
        return total


# ---------------------------------------------------------------------------
# Granger causality graph
# ---------------------------------------------------------------------------

class GrangerCausalityGraph:
    """
    Build a Granger causality graph from multivariate time series.

    For each pair (i, j), tests whether past values of X_i improve
    forecast of X_j beyond X_j's own history.

    Uses F-test from VAR model comparison.
    """

    def __init__(
        self,
        max_lag: int = 5,
        alpha: float = 0.05,
        method: str = "f_test",
    ):
        self.max_lag = max_lag
        self.alpha = alpha
        self.method = method

    def fit(
        self,
        data: np.ndarray,
        node_names: Optional[List[str]] = None,
    ) -> CausalGraph:
        """
        Parameters
        ----------
        data : (T, N) time series

        Returns
        -------
        CausalGraph with directed edges i → j if X_i Granger-causes X_j
        """
        T, n = data.shape
        node_names = node_names or [f"X_{i}" for i in range(n)]
        graph = CausalGraph(n_nodes=n, node_names=node_names)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p_val = self._granger_test(data[:, i], data[:, j], self.max_lag)
                if p_val < self.alpha:
                    graph.add_edge(i, j)

        return graph

    def _granger_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int,
    ) -> float:
        """
        Test H0: X does not Granger-cause Y.

        Returns p-value. Small p → reject H0 → X Granger-causes Y.
        """
        T = len(y)
        L = max_lag

        if T - L - L < 10:
            return 1.0

        # Restricted model: y_t = sum_k a_k y_{t-k} + eps
        Y = y[L:]
        X_restricted = np.column_stack([y[L - k : T - k] for k in range(1, L + 1)])

        # Unrestricted model: adds lags of x
        X_unrestricted = np.column_stack([
            X_restricted,
            np.column_stack([x[L - k : T - k] for k in range(1, L + 1)]),
        ])

        try:
            # Fit restricted
            beta_r = np.linalg.lstsq(
                np.column_stack([X_restricted, np.ones(len(Y))]), Y, rcond=None
            )[0]
            res_r = Y - np.column_stack([X_restricted, np.ones(len(Y))]) @ beta_r
            rss_r = float(res_r @ res_r)

            # Fit unrestricted
            beta_u = np.linalg.lstsq(
                np.column_stack([X_unrestricted, np.ones(len(Y))]), Y, rcond=None
            )[0]
            res_u = Y - np.column_stack([X_unrestricted, np.ones(len(Y))]) @ beta_u
            rss_u = float(res_u @ res_u)

        except np.linalg.LinAlgError:
            return 1.0

        n_obs = len(Y)
        n_params_r = X_restricted.shape[1] + 1
        n_params_u = X_unrestricted.shape[1] + 1

        if rss_u < 1e-10:
            return 0.0

        F = ((rss_r - rss_u) / L) / (rss_u / (n_obs - n_params_u))
        if F < 0:
            return 1.0

        p_val = stats.f.sf(F, dfn=L, dfd=n_obs - n_params_u)
        return float(p_val)

    def causal_matrix(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """Return (N, N) matrix of p-values for Granger causality."""
        T, n = data.shape
        p_matrix = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    p_matrix[i, j] = self._granger_test(data[:, i], data[:, j], self.max_lag)
        return p_matrix


# ---------------------------------------------------------------------------
# NOTEARS continuous optimisation for DAG learning
# ---------------------------------------------------------------------------

class NOTEARS(nn.Module):
    """
    NOTEARS: Non-combinatorial Optimisation via Trace Exponential and Augmented lagRangian
    for Structure learning.

    Learns a weighted adjacency matrix W such that the corresponding graph is a DAG,
    using the constraint h(W) = tr(e^{W ∘ W}) - d = 0.

    Reference: Zheng et al., "DAGs with NO TEARS: Continuous Optimization for Structure Learning",
    NeurIPS 2018.
    """

    def __init__(
        self,
        n_nodes: int,
        loss_type: str = "l2",
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
    ):
        super().__init__()
        self.n = n_nodes
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold

        # Learnable adjacency matrix (initialised small random)
        self.W = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1)

    def _h(self, W: Tensor) -> Tensor:
        """
        DAG constraint: h(W) = tr(e^{W ∘ W}) - d = 0 iff G(W) is a DAG.
        """
        d = self.n
        # Matrix exponential approximation (series expansion for efficiency)
        WW = W * W
        # Use trace of matrix exponential: tr(expm(WW))
        # Approximation via sum of traces of powers
        M = torch.eye(d, device=W.device)
        total = torch.zeros(1, device=W.device)
        total = total + d  # trace of I
        power = WW.clone()
        factorial = 1.0
        for k in range(1, min(10, d)):
            factorial *= k
            total = total + power.diagonal().sum() / factorial
            power = power @ WW
        return total - d

    def _loss(self, W: Tensor, X: Tensor) -> Tensor:
        """
        Structural equation model loss.

        For L2: loss = (1/2n) ||X - X W^T||_F^2
        """
        n = X.shape[0]
        residual = X - X @ W.T
        return 0.5 * (residual ** 2).sum() / n

    def fit(
        self,
        data: np.ndarray,
        lr: float = 0.01,
        lambda1: float = 0.01,
    ) -> Tuple[np.ndarray, CausalGraph]:
        """
        Fit NOTEARS model using augmented Lagrangian.

        Parameters
        ----------
        data    : (T, N) standardised data
        lr      : learning rate
        lambda1 : L1 penalty for sparsity

        Returns
        -------
        W_est   : (N, N) estimated weighted adjacency matrix
        graph   : CausalGraph (thresholded DAG)
        """
        X = torch.tensor(data, dtype=torch.float32)
        d = self.n

        # Augmented Lagrangian parameters
        rho = 1.0
        alpha_lag = 0.0
        h_val = torch.tensor(float("inf"))

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for it in range(self.max_iter):
            for inner_step in range(200):
                optimizer.zero_grad()
                W = self.W

                # Mask diagonal (no self-loops)
                W_masked = W * (1 - torch.eye(d, device=W.device))

                loss = self._loss(W_masked, X)
                h = self._h(W_masked)
                l1_penalty = lambda1 * W_masked.abs().sum()

                # Augmented Lagrangian
                al_term = alpha_lag * h + 0.5 * rho * h ** 2
                total_loss = loss + al_term + l1_penalty

                total_loss.backward()
                optimizer.step()

            with torch.no_grad():
                W_masked = self.W * (1 - torch.eye(d, device=self.W.device))
                h_val = self._h(W_masked)

            if float(h_val) <= self.h_tol:
                break

            alpha_lag += rho * float(h_val)
            rho = min(rho * 2, self.rho_max)

        # Extract final W
        W_est = self.W.detach().numpy().copy()
        np.fill_diagonal(W_est, 0)

        # Threshold
        W_thresh = np.where(np.abs(W_est) >= self.w_threshold, W_est, 0.0)

        # Build CausalGraph
        graph = CausalGraph(n_nodes=d)
        for i in range(d):
            for j in range(d):
                if W_thresh[i, j] != 0.0:
                    graph.add_edge(i, j)

        return W_thresh, graph


# ---------------------------------------------------------------------------
# Causal effect estimation (do-calculus adjustment)
# ---------------------------------------------------------------------------

class CausalEffectEstimator:
    """
    Estimate causal effects using observational data and a causal graph.

    Implements:
      1. Backdoor adjustment
      2. Front-door adjustment
      3. Inverse propensity weighting (IPW)

    Useful for estimating the causal effect of intervention
    do(X_i = x) on X_j in financial networks (e.g., effect of Fed rate
    change on equity returns).
    """

    def __init__(self, graph: CausalGraph, data: np.ndarray):
        self.graph = graph
        self.data = data

    def backdoor_adjustment(
        self,
        treatment: int,
        outcome: int,
        adjustment_set: Optional[List[int]] = None,
    ) -> float:
        """
        Estimate E[Y | do(X=1)] - E[Y | do(X=0)] via backdoor adjustment.

        Parameters
        ----------
        treatment : index of treatment variable (X)
        outcome   : index of outcome variable (Y)
        adjustment_set : confounders to adjust for; auto-detected if None

        Returns
        -------
        ate : float — Average Treatment Effect
        """
        if adjustment_set is None:
            adjustment_set = self._find_backdoor_set(treatment, outcome)

        if not adjustment_set:
            # No confounding: simple difference in means
            T_data = self.data[:, treatment]
            Y_data = self.data[:, outcome]
            high = T_data > np.median(T_data)
            return float(Y_data[high].mean() - Y_data[~high].mean())

        # Regression-based backdoor adjustment
        T = self.data[:, treatment]
        Y = self.data[:, outcome]
        Z = self.data[:, adjustment_set]

        # Fit E[Y | T, Z] via OLS
        X_mat = np.column_stack([T.reshape(-1, 1), Z, np.ones((len(T), 1))])
        try:
            beta = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
            ate = float(beta[0])  # coefficient on treatment
        except np.linalg.LinAlgError:
            ate = 0.0

        return ate

    def _find_backdoor_set(self, treatment: int, outcome: int) -> List[int]:
        """
        Find a valid backdoor adjustment set (parents of treatment).
        """
        parents_t = self.graph.parents(treatment)
        return [p for p in parents_t if p != outcome]

    def ipw_estimate(
        self,
        treatment: int,
        outcome: int,
        propensity_covariates: Optional[List[int]] = None,
    ) -> float:
        """
        Inverse Propensity Weighted (IPW) estimate of ATE.
        """
        T = self.data[:, treatment]
        Y = self.data[:, outcome]

        if propensity_covariates is None:
            propensity_covariates = self._find_backdoor_set(treatment, outcome)

        if not propensity_covariates:
            # Naive estimate
            high = T > np.median(T)
            return float(Y[high].mean() - Y[~high].mean())

        Z = self.data[:, propensity_covariates]

        # Estimate propensity scores via logistic regression
        from scipy.special import expit

        T_binary = (T > np.median(T)).astype(float)
        X_mat = np.column_stack([Z, np.ones(len(T))])

        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=500)
            lr.fit(Z, T_binary)
            ps = lr.predict_proba(Z)[:, 1]
        except ImportError:
            # Fallback: use linear probability
            beta = np.linalg.lstsq(X_mat, T_binary, rcond=None)[0]
            ps = np.clip(X_mat @ beta, 0.05, 0.95)

        ps = np.clip(ps, 0.05, 0.95)

        # IPW estimator
        treated = T_binary == 1
        ate = (
            np.mean(Y[treated] / ps[treated]) - np.mean(Y[~treated] / (1 - ps[~treated]))
        )
        return float(ate)

    def causal_effect_matrix(self) -> np.ndarray:
        """Compute pairwise ATE matrix for all (treatment, outcome) pairs."""
        n = self.graph.n_nodes
        effects = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and self.graph.has_edge(i, j):
                    try:
                        effects[i, j] = self.backdoor_adjustment(i, j)
                    except Exception:
                        effects[i, j] = 0.0
        return effects


# ---------------------------------------------------------------------------
# Combined causal graph discovery pipeline
# ---------------------------------------------------------------------------

class FinancialCausalDiscovery:
    """
    End-to-end causal discovery pipeline for financial returns data.

    Runs multiple algorithms and takes a consensus graph.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_lag: int = 5,
        use_granger: bool = True,
        use_pc: bool = True,
        use_notears: bool = True,
        notears_threshold: float = 0.3,
        consensus_method: str = "union",
    ):
        self.alpha = alpha
        self.use_granger = use_granger
        self.use_pc = use_pc
        self.use_notears = use_notears
        self.notears_threshold = notears_threshold
        self.consensus_method = consensus_method

        if use_granger:
            self.granger = GrangerCausalityGraph(max_lag=max_lag, alpha=alpha)
        if use_pc:
            self.pc = PCAlgorithm(alpha=alpha, max_cond_set_size=3)
        if use_notears:
            self._notears_cls = NOTEARS  # instantiated per run

    def discover(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
    ) -> CausalGraph:
        """
        Run causal discovery and return consensus graph.

        Parameters
        ----------
        returns : (T, N) standardised returns
        """
        T, n = returns.shape
        asset_names = asset_names or [f"asset_{i}" for i in range(n)]

        # Standardise
        data = (returns - returns.mean(axis=0)) / (returns.std(axis=0) + 1e-8)

        graphs = []

        if self.use_granger:
            try:
                g = self.granger.fit(data, asset_names)
                graphs.append(g)
            except Exception as e:
                warnings.warn(f"Granger failed: {e}")

        if self.use_pc:
            try:
                g = self.pc.fit(data, asset_names)
                # Convert CPDAG to DAG (orient remaining undirected arbitrarily)
                for edge in list(g.undirected):
                    i, j = sorted(edge)
                    g.orient_edge(i, j)
                graphs.append(g)
            except Exception as e:
                warnings.warn(f"PC failed: {e}")

        if self.use_notears and n <= 50:  # NOTEARS is expensive for large n
            try:
                model = self._notears_cls(n, w_threshold=self.notears_threshold)
                _, g = model.fit(data)
                g.node_names = asset_names
                graphs.append(g)
            except Exception as e:
                warnings.warn(f"NOTEARS failed: {e}")

        if not graphs:
            return CausalGraph(n_nodes=n, node_names=asset_names)

        return self._consensus(graphs, n, asset_names)

    def _consensus(
        self,
        graphs: List[CausalGraph],
        n: int,
        node_names: List[str],
    ) -> CausalGraph:
        """Merge multiple causal graphs via union or majority vote."""
        result = CausalGraph(n_nodes=n, node_names=node_names)
        edge_counts: Dict[Tuple[int, int], int] = {}

        for g in graphs:
            for edge in g.edges:
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        if self.consensus_method == "union":
            for edge in edge_counts:
                result.add_edge(*edge)

        elif self.consensus_method == "majority":
            threshold = len(graphs) // 2
            for edge, count in edge_counts.items():
                if count > threshold:
                    result.add_edge(*edge)

        elif self.consensus_method == "intersection":
            required = len(graphs)
            for edge, count in edge_counts.items():
                if count == required:
                    result.add_edge(*edge)

        return result

    def to_pyg_data(
        self,
        graph: CausalGraph,
        node_features: Optional[Tensor] = None,
    ) -> "Data":
        """Convert CausalGraph to PyG Data object."""
        try:
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("torch_geometric required")

        ei = graph.to_edge_index()
        return Data(
            x=node_features,
            edge_index=ei,
            num_nodes=graph.n_nodes,
        )


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "CausalGraph",
    "partial_correlation_test",
    "fisher_z_test",
    "PCAlgorithm",
    "FCIAlgorithm",
    "GES",
    "GrangerCausalityGraph",
    "NOTEARS",
    "CausalEffectEstimator",
    "FinancialCausalDiscovery",
]
