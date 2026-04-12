"""
fast_graph_ops.py — Fast Graph Operations
==========================================
Part of the Omni-Graph incremental graph construction suite.

Design goals
------------
* Cached Laplacian decomposition: reuse eigenvectors when the graph
  changes slowly (change < threshold).
* Approximate PageRank: power iteration with early stopping when the
  delta between iterations falls below epsilon.
* Streaming GNN inference: process only changed node neighbourhoods
  and cache results for unchanged nodes.
* Incremental spectral clustering: update cluster assignments when
  edges change using warm-started k-means on Laplacian eigenvectors.
* O(1) Fiedler value approximation using Lanczos iteration with warm
  restart from the previous eigenvector.
* All ops benchmarked and profiled; target <500 μs for 500-node graph.

Public API
----------
    ops = FastGraphOps(n_nodes=500, device="cuda")
    ops.update_graph(edge_index, edge_weight)
    pr = ops.pagerank(damping=0.85, max_iter=50)
    clusters = ops.spectral_clusters(k=5)
    fiedler = ops.fiedler_value()
    node_emb = ops.gnn_inference(x, model)
"""

from __future__ import annotations

import math
import time
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FastGraphOpsConfig:
    """Configuration for FastGraphOps."""

    n_nodes: int = 500
    device: str = "cuda"

    # Laplacian cache: reuse eigenvectors if graph change < this fraction
    eigen_cache_threshold: float = 0.05

    # Number of eigenvectors to cache
    n_eigen: int = 10

    # PageRank: early stopping delta
    pagerank_epsilon: float = 1e-6
    pagerank_max_iter: int = 100
    pagerank_damping: float = 0.85

    # Spectral clustering: number of clusters
    n_clusters: int = 5
    kmeans_max_iter: int = 100
    kmeans_tol: float = 1e-4

    # Fiedler: Lanczos iterations
    lanczos_k: int = 20
    fiedler_warm_restart: bool = True

    # GNN inference: neighbourhood cache size
    gnn_cache_size: int = 500
    gnn_dirty_threshold: float = 0.01

    # Benchmark: print every N calls
    benchmark_interval: int = 200


# ---------------------------------------------------------------------------
# Laplacian cache
# ---------------------------------------------------------------------------

class LaplacianEigenCache:
    """
    Caches the eigendecomposition of the graph Laplacian.
    Reuses the cached result if the graph hasn't changed significantly.

    Change is measured as the Frobenius norm of the difference in the
    degree sequence divided by the total degree.
    """

    def __init__(
        self,
        n_nodes: int,
        n_eigen: int,
        change_threshold: float,
        device: torch.device,
    ) -> None:
        self.n = n_nodes
        self.n_eigen = n_eigen
        self.threshold = change_threshold
        self.device = device

        # Cached values
        self._eigenvalues: Optional[torch.Tensor] = None   # (k,)
        self._eigenvectors: Optional[torch.Tensor] = None  # (N, k)
        self._prev_degree: Optional[torch.Tensor] = None   # (N,)
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    # ------------------------------------------------------------------
    def get_or_compute(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (eigenvalues, eigenvectors), computing or using cache.

        Returns
        -------
        eigenvalues  : (k,) float32
        eigenvectors : (N, k) float32
        """
        n = self.n
        dev = self.device

        # Compute current degree
        deg = torch.zeros(n, device=dev)
        if row.numel() > 0:
            deg.scatter_add_(0, row.to(dev), weight.to(dev, dtype=torch.float32))

        # Check if graph changed significantly
        if self._prev_degree is not None and self._eigenvalues is not None:
            delta = (deg - self._prev_degree).norm() / (self._prev_degree.norm() + 1e-8)
            if float(delta.item()) < self.threshold:
                self._cache_hits += 1
                return self._eigenvalues, self._eigenvectors  # type: ignore[return-value]

        self._cache_misses += 1

        # Recompute Laplacian eigenvectors
        L = self._build_normalised_laplacian(row, col, weight, n, dev, deg)

        try:
            # Use symmetric eigensolver
            eigvals, eigvecs = torch.linalg.eigh(L)
            # Take smallest k (skip first = constant vector for connected graph)
            eigvals = eigvals[:self.n_eigen]
            eigvecs = eigvecs[:, :self.n_eigen]
        except Exception as exc:
            logger.warning("Eigensolver failed: %s; returning zeros", exc)
            eigvals = torch.zeros(self.n_eigen, device=dev)
            eigvecs = torch.zeros(n, self.n_eigen, device=dev)

        self._eigenvalues = eigvals
        self._eigenvectors = eigvecs
        self._prev_degree = deg
        return eigvals, eigvecs

    # ------------------------------------------------------------------
    def invalidate(self) -> None:
        """Force recomputation on next call."""
        self._prev_degree = None
        self._eigenvalues = None
        self._eigenvectors = None

    # ------------------------------------------------------------------
    @property
    def cache_hit_rate(self) -> float:
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / max(total, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_normalised_laplacian(
        row: torch.Tensor,
        col: torch.Tensor,
        weight: torch.Tensor,
        n: int,
        dev: torch.device,
        deg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the normalised Laplacian L = I - D^{-1/2} A D^{-1/2} (dense)."""
        A = torch.zeros(n, n, dtype=torch.float32, device=dev)
        if row.numel() > 0:
            A[row.to(dev), col.to(dev)] = weight.to(dev, dtype=torch.float32)

        if deg is None:
            deg = A.sum(dim=1)

        d_inv_sqrt = deg.pow(-0.5).clamp(max=1e4)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        L = torch.eye(n, device=dev) - D_inv_sqrt @ A @ D_inv_sqrt
        return L


# ---------------------------------------------------------------------------
# Approximate PageRank
# ---------------------------------------------------------------------------

class ApproximatePageRank:
    """
    Computes approximate PageRank via power iteration with early stopping.

    Power iteration:
        r ← alpha * (A_hat @ r) + (1 - alpha) * v
    where A_hat = D^{-1} A (row-normalised), v is the personalisation vector.

    Early stop when ||r_t - r_{t-1}||_1 < epsilon.
    """

    def __init__(
        self,
        n_nodes: int,
        damping: float = 0.85,
        epsilon: float = 1e-6,
        max_iter: int = 100,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.n = n_nodes
        self.damping = damping
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.device = device

        # Warm-start PageRank vector
        self._prev_pr: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def compute(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        weight: torch.Tensor,
        personalisation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute PageRank vector.

        Parameters
        ----------
        row, col      : (E,) int64
        weight        : (E,) float32
        personalisation : (N,) float32 optional — default uniform

        Returns
        -------
        pr : (N,) float32
        """
        n = self.n
        dev = self.device

        if personalisation is None:
            v = torch.full((n,), 1.0 / n, dtype=torch.float32, device=dev)
        else:
            v = personalisation.to(dev, dtype=torch.float32)
            v = v / v.sum().clamp(min=1e-8)

        # Build row-normalised adjacency: A_hat[i, j] = w[i,j] / deg[i]
        deg = torch.zeros(n, device=dev)
        if row.numel() > 0:
            deg.scatter_add_(0, row.to(dev), weight.to(dev, dtype=torch.float32))
        deg_safe = deg.clamp(min=1e-8)

        # Warm start
        if self._prev_pr is not None and self._prev_pr.shape[0] == n:
            r = self._prev_pr.clone()
        else:
            r = v.clone()

        alpha = self.damping
        one_minus_alpha = 1.0 - alpha

        for _ in range(self.max_iter):
            # A_hat @ r via scatter
            if row.numel() > 0:
                r_src = row.to(dev)
                c_src = col.to(dev)
                w_norm = weight.to(dev, dtype=torch.float32) / deg_safe[r_src]
                new_r = torch.zeros(n, device=dev)
                new_r.scatter_add_(0, c_src, w_norm * r[r_src])
            else:
                new_r = torch.zeros(n, device=dev)

            new_r = alpha * new_r + one_minus_alpha * v

            # Check convergence
            delta = (new_r - r).abs().sum()
            r = new_r

            if float(delta.item()) < self.epsilon:
                break

        # Normalise
        r = r / r.sum().clamp(min=1e-8)
        self._prev_pr = r
        return r

    # ------------------------------------------------------------------
    def personalised(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        weight: torch.Tensor,
        seed_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute personalised PageRank from a set of seed nodes.

        Parameters
        ----------
        seed_nodes : (K,) int64

        Returns
        -------
        pr : (N,) float32
        """
        n = self.n
        v = torch.zeros(n, dtype=torch.float32, device=self.device)
        v[seed_nodes.to(self.device)] = 1.0 / seed_nodes.numel()
        return self.compute(row, col, weight, personalisation=v)


# ---------------------------------------------------------------------------
# Streaming GNN inference with neighbourhood cache
# ---------------------------------------------------------------------------

class StreamingGNNInference:
    """
    Incremental GNN inference that only recomputes node embeddings
    for nodes whose neighbourhood has changed.

    For unchanged nodes, the cached embedding from the previous tick
    is reused.  This exploits temporal locality in financial graphs
    (most edges don't change every tick).
    """

    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int,
        dirty_threshold: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.n = n_nodes
        self.hidden_dim = hidden_dim
        self.threshold = dirty_threshold
        self.device = device

        # Embedding cache
        self._cached_embeddings: Optional[torch.Tensor] = None   # (N, H)
        self._cached_edge_index: Optional[torch.Tensor] = None   # (2, E)
        self._cached_edge_weight: Optional[torch.Tensor] = None  # (E,)

        # Dirty flags for nodes
        self._dirty: torch.Tensor = torch.ones(n_nodes, dtype=torch.bool, device=device)

        # Metrics
        self._n_recomputed: int = 0
        self._n_cached: int = 0

    # ------------------------------------------------------------------
    def update_graph(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update the graph and return the dirty node mask.

        Nodes are marked dirty if:
        1. Any of their edges changed weight by > threshold, or
        2. They gained or lost neighbours.

        Returns
        -------
        dirty_mask : (N,) bool
        """
        if self._cached_edge_index is None:
            self._dirty.fill_(True)
            self._cached_edge_index = edge_index
            self._cached_edge_weight = edge_weight
            return self._dirty

        new_ei = edge_index.to(self.device)
        new_ew = edge_weight.to(self.device, dtype=torch.float32)
        old_ei = self._cached_edge_index
        old_ew = self._cached_edge_weight

        n = self.n
        dirty = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Compare edge sets: changed topology → dirty source nodes
        if new_ei.shape[1] != old_ei.shape[1]:
            dirty.fill_(True)
        else:
            # Check weight changes
            try:
                w_delta = (new_ew - old_ew).abs()
                changed_edge_mask = w_delta > self.threshold
                if changed_edge_mask.any():
                    changed_src = new_ei[0][changed_edge_mask]
                    dirty[changed_src] = True
            except Exception:
                dirty.fill_(True)

        self._dirty = dirty
        self._cached_edge_index = new_ei
        self._cached_edge_weight = new_ew
        return dirty

    # ------------------------------------------------------------------
    def infer(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        dirty_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run GNN inference, reusing cached embeddings for clean nodes.

        Parameters
        ----------
        x           : (N, D) float32 input features
        edge_index  : (2, E) int64
        edge_weight : (E,) float32
        model       : callable(x, edge_index, edge_weight) → (N, H)
        dirty_mask  : optional override; if None, uses internal dirty mask

        Returns
        -------
        embeddings : (N, H) float32
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device, dtype=torch.float32)

        if dirty_mask is None:
            dirty_mask = self._dirty

        n_dirty = int(dirty_mask.sum().item())
        n_clean = self.n - n_dirty

        if self._cached_embeddings is None or n_dirty == self.n:
            # Full recompute
            embeddings = model(x, edge_index, edge_weight)
            self._cached_embeddings = embeddings.detach()
            self._n_recomputed += self.n
            return embeddings

        if n_dirty == 0:
            self._n_cached += self.n
            return self._cached_embeddings

        # Partial recompute: run full forward but only update dirty nodes
        # (In a true streaming GNN, we'd extract sub-graphs, but for
        # correctness we run the full forward and splice results.)
        new_embeddings = model(x, edge_index, edge_weight)
        result = self._cached_embeddings.clone()
        result[dirty_mask] = new_embeddings[dirty_mask].detach()
        self._cached_embeddings = result
        self._n_recomputed += n_dirty
        self._n_cached += n_clean
        return result

    # ------------------------------------------------------------------
    @property
    def cache_hit_rate(self) -> float:
        total = self._n_recomputed + self._n_cached
        return self._n_cached / max(total, 1)


# ---------------------------------------------------------------------------
# Incremental spectral clustering
# ---------------------------------------------------------------------------

class IncrementalSpectralClustering:
    """
    Maintains spectral cluster assignments and updates them incrementally
    when the graph Laplacian eigenvectors change.

    Uses warm-started k-means on the (N, k) Laplacian eigenvector matrix.
    """

    def __init__(
        self,
        n_nodes: int,
        n_clusters: int,
        n_eigen: int = 10,
        kmeans_max_iter: int = 100,
        kmeans_tol: float = 1e-4,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.n = n_nodes
        self.k = n_clusters
        self.n_eigen = n_eigen
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol
        self.device = device

        # Current cluster assignments (N,) int64
        self._labels: Optional[torch.Tensor] = None

        # Previous eigenvectors for warm start
        self._prev_eigvecs: Optional[torch.Tensor] = None

        # K-means centroids (k, n_eigen)
        self._centroids: Optional[torch.Tensor] = None

        # Metrics
        self._n_full_recluster: int = 0
        self._n_incremental: int = 0

    # ------------------------------------------------------------------
    def update(
        self,
        eigvecs: torch.Tensor,
        force_recluster: bool = False,
    ) -> torch.Tensor:
        """
        Update cluster assignments given new eigenvectors.

        Parameters
        ----------
        eigvecs        : (N, k) float32 — Laplacian eigenvectors
        force_recluster : if True, run full k-means from scratch

        Returns
        -------
        labels : (N,) int64 cluster assignments
        """
        eigvecs = eigvecs[:, : self.n_eigen].to(self.device, dtype=torch.float32)

        # Normalise rows (standard spectral clustering step)
        norms = eigvecs.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        X = eigvecs / norms

        if self._centroids is None or force_recluster:
            self._labels, self._centroids = self._kmeans_full(X)
            self._n_full_recluster += 1
        else:
            # Warm-start: reassign nodes then update centroids
            self._labels, self._centroids = self._kmeans_warm(X, self._centroids)
            self._n_incremental += 1

        self._prev_eigvecs = eigvecs
        return self._labels

    # ------------------------------------------------------------------
    def _kmeans_full(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """K-means++ initialisation followed by Lloyd's algorithm."""
        n, d = X.shape
        k = min(self.k, n)

        # K-means++ init
        centroids = self._kmeans_plus_plus_init(X, k)
        return self._lloyd(X, centroids)

    # ------------------------------------------------------------------
    def _kmeans_warm(
        self,
        X: torch.Tensor,
        centroids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Lloyd's from existing centroids (warm start)."""
        return self._lloyd(X, centroids)

    # ------------------------------------------------------------------
    def _lloyd(
        self,
        X: torch.Tensor,
        centroids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lloyd's algorithm."""
        for _ in range(self.kmeans_max_iter):
            # Assignment step
            dists = torch.cdist(X, centroids)  # (N, k)
            labels = dists.argmin(dim=1)       # (N,)

            # Update step
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(centroids.shape[0], device=self.device)
            for j in range(centroids.shape[0]):
                mask = labels == j
                if mask.any():
                    new_centroids[j] = X[mask].mean(dim=0)
                    counts[j] = mask.sum()
                else:
                    new_centroids[j] = centroids[j]

            # Convergence
            delta = (new_centroids - centroids).norm()
            centroids = new_centroids
            if float(delta.item()) < self.kmeans_tol:
                break

        # Final assignment
        dists = torch.cdist(X, centroids)
        labels = dists.argmin(dim=1)
        return labels, centroids

    # ------------------------------------------------------------------
    def _kmeans_plus_plus_init(self, X: torch.Tensor, k: int) -> torch.Tensor:
        """K-means++ centroid initialisation."""
        n = X.shape[0]
        # Choose first centroid uniformly at random
        idx = torch.randint(0, n, (1,), device=self.device).item()
        centroids = [X[idx].unsqueeze(0)]

        for _ in range(1, k):
            C = torch.cat(centroids, dim=0)  # (current_k, d)
            dists = torch.cdist(X, C).min(dim=1).values  # (N,)
            probs = dists ** 2
            probs /= probs.sum().clamp(min=1e-8)
            chosen = int(torch.multinomial(probs, 1).item())
            centroids.append(X[chosen].unsqueeze(0))

        return torch.cat(centroids, dim=0)  # (k, d)

    # ------------------------------------------------------------------
    @property
    def labels(self) -> Optional[torch.Tensor]:
        return self._labels

    # ------------------------------------------------------------------
    def cluster_sizes(self) -> Optional[Dict[int, int]]:
        if self._labels is None:
            return None
        sizes: Dict[int, int] = {}
        for lbl in self._labels.cpu().tolist():
            sizes[lbl] = sizes.get(lbl, 0) + 1
        return sizes


# ---------------------------------------------------------------------------
# Lanczos Fiedler approximation
# ---------------------------------------------------------------------------

class LanczosFiedler:
    """
    Estimates the Fiedler value (λ₂ of the Laplacian) using the Lanczos
    algorithm with warm restart.

    Warm restart: reuses the previous Lanczos vector as the starting
    point so fewer iterations are needed when the graph changes slowly.

    Target: O(k * N) per call where k = lanczos_k << N.
    """

    def __init__(
        self,
        n_nodes: int,
        k: int = 20,
        warm_restart: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.n = n_nodes
        self.k = k
        self.warm_restart = warm_restart
        self.device = device

        # Warm-start vector
        self._v0: Optional[torch.Tensor] = None
        self._last_fiedler: float = 0.0

    # ------------------------------------------------------------------
    def estimate(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        weight: torch.Tensor,
    ) -> float:
        """
        Estimate Fiedler value.

        Parameters
        ----------
        row, col : (E,) int64
        weight   : (E,) float32

        Returns
        -------
        fiedler : float (λ₂ ≥ 0)
        """
        n = self.n
        dev = self.device
        k = min(self.k, n - 1)

        row = row.to(dev)
        col = col.to(dev)
        weight = weight.to(dev, dtype=torch.float32)

        # Degree
        deg = torch.zeros(n, device=dev)
        if row.numel() > 0:
            deg.scatter_add_(0, row, weight)

        # Starting vector
        if self.warm_restart and self._v0 is not None and self._v0.shape[0] == n:
            v = self._v0.clone()
        else:
            v = torch.randn(n, device=dev)

        # Project out constant component
        ones = torch.ones(n, device=dev) / math.sqrt(n)
        v -= (v @ ones) * ones
        norm = v.norm()
        if norm < 1e-10:
            return 0.0
        v /= norm

        # Lanczos iteration
        alpha_list: List[float] = []
        beta_list: List[float] = []
        V = [v]  # Lanczos vectors

        def Lv_matvec(u: torch.Tensor) -> torch.Tensor:
            """Apply L = D - A to vector u."""
            Du = deg * u
            Au = torch.zeros(n, device=dev)
            if row.numel() > 0:
                Au.scatter_add_(0, col, weight * u[row])
            return Du - Au

        prev_beta = 0.0
        prev_v = torch.zeros(n, device=dev)

        for j in range(k):
            w = Lv_matvec(v)
            alpha = float((v @ w).item())
            alpha_list.append(alpha)

            w = w - alpha * v - prev_beta * prev_v

            beta = float(w.norm().item())
            if beta < 1e-10:
                break
            beta_list.append(beta)

            prev_v = v
            prev_beta = beta
            v = w / beta
            V.append(v)

        # Build tridiagonal matrix
        m = len(alpha_list)
        if m == 0:
            return 0.0

        T = torch.zeros(m, m, device=dev)
        for i in range(m):
            T[i, i] = alpha_list[i]
        for i in range(len(beta_list)):
            if i + 1 < m:
                T[i, i + 1] = beta_list[i]
                T[i + 1, i] = beta_list[i]

        # Eigenvalues of tridiagonal
        try:
            eigvals = torch.linalg.eigvalsh(T)
        except Exception:
            self._last_fiedler = 0.0
            return 0.0

        # λ₂ is the second smallest (skip λ₁ ≈ 0)
        sorted_eigvals = eigvals.sort().values
        if sorted_eigvals.numel() > 1:
            fiedler = max(0.0, float(sorted_eigvals[1].item()))
        else:
            fiedler = max(0.0, float(sorted_eigvals[0].item()))

        self._last_fiedler = fiedler
        self._v0 = v  # save for warm restart
        return fiedler

    # ------------------------------------------------------------------
    @property
    def last_fiedler(self) -> float:
        return self._last_fiedler

    # ------------------------------------------------------------------
    def is_connected(self, threshold: float = 1e-4) -> bool:
        return self._last_fiedler > threshold


# ---------------------------------------------------------------------------
# Main FastGraphOps class
# ---------------------------------------------------------------------------

class FastGraphOps:
    """
    Fast graph operations orchestrating all sub-components.

    Usage
    -----
        ops = FastGraphOps(n_nodes=500, device="cuda")
        ops.update_graph(edge_index, edge_weight)
        pr = ops.pagerank()
        clusters = ops.spectral_clusters()
        fiedler = ops.fiedler_value()
    """

    def __init__(
        self,
        n_nodes: int,
        cfg: Optional[FastGraphOpsConfig] = None,
        **kwargs: Any,
    ) -> None:
        if cfg is None:
            cfg = FastGraphOpsConfig(n_nodes=n_nodes, **kwargs)
        self.cfg = cfg
        self.n = n_nodes
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )

        # Sub-components
        self._eigen_cache = LaplacianEigenCache(
            n_nodes=n_nodes,
            n_eigen=cfg.n_eigen,
            change_threshold=cfg.eigen_cache_threshold,
            device=self.device,
        )
        self._pagerank = ApproximatePageRank(
            n_nodes=n_nodes,
            damping=cfg.pagerank_damping,
            epsilon=cfg.pagerank_epsilon,
            max_iter=cfg.pagerank_max_iter,
            device=self.device,
        )
        self._spectral = IncrementalSpectralClustering(
            n_nodes=n_nodes,
            n_clusters=cfg.n_clusters,
            n_eigen=cfg.n_eigen,
            kmeans_max_iter=cfg.kmeans_max_iter,
            kmeans_tol=cfg.kmeans_tol,
            device=self.device,
        )
        self._lanczos = LanczosFiedler(
            n_nodes=n_nodes,
            k=cfg.lanczos_k,
            warm_restart=cfg.fiedler_warm_restart,
            device=self.device,
        )
        self._gnn_cache = StreamingGNNInference(
            n_nodes=n_nodes,
            hidden_dim=64,
            dirty_threshold=cfg.gnn_dirty_threshold,
            device=self.device,
        )

        # Current graph state
        self._row: Optional[torch.Tensor] = None
        self._col: Optional[torch.Tensor] = None
        self._weight: Optional[torch.Tensor] = None

        # Benchmark counters
        self._op_times: Dict[str, List[float]] = {
            "pagerank": [], "fiedler": [], "clusters": [], "gnn": []
        }
        self._call_count: int = 0

    # ------------------------------------------------------------------
    def update_graph(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> None:
        """
        Update the stored graph state.

        Parameters
        ----------
        edge_index  : (2, E) int64
        edge_weight : (E,) float32
        """
        self._row = edge_index[0].to(self.device)
        self._col = edge_index[1].to(self.device)
        self._weight = edge_weight.to(self.device, dtype=torch.float32)
        self._gnn_cache.update_graph(edge_index, edge_weight)

    # ------------------------------------------------------------------
    def pagerank(
        self,
        damping: Optional[float] = None,
        personalisation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute approximate PageRank.

        Returns
        -------
        (N,) float32 — PageRank scores
        """
        self._check_graph()
        if damping is not None:
            self._pagerank.damping = damping

        t0 = time.perf_counter()
        pr = self._pagerank.compute(
            self._row, self._col, self._weight, personalisation  # type: ignore[arg-type]
        )
        self._record_time("pagerank", t0)
        return pr

    # ------------------------------------------------------------------
    def personalised_pagerank(
        self,
        seed_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute personalised PageRank from seed nodes."""
        self._check_graph()
        return self._pagerank.personalised(
            self._row, self._col, self._weight, seed_nodes  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    def spectral_clusters(
        self,
        k: Optional[int] = None,
        force_recluster: bool = False,
    ) -> torch.Tensor:
        """
        Compute spectral cluster assignments.

        Returns
        -------
        labels : (N,) int64
        """
        self._check_graph()
        if k is not None:
            self._spectral.k = k

        t0 = time.perf_counter()
        _, eigvecs = self._eigen_cache.get_or_compute(
            self._row, self._col, self._weight  # type: ignore[arg-type]
        )
        labels = self._spectral.update(eigvecs, force_recluster=force_recluster)
        self._record_time("clusters", t0)
        return labels

    # ------------------------------------------------------------------
    def fiedler_value(self) -> float:
        """
        Estimate the Fiedler value (λ₂ of the Laplacian).

        Returns
        -------
        fiedler : float
        """
        self._check_graph()
        t0 = time.perf_counter()
        fiedler = self._lanczos.estimate(
            self._row, self._col, self._weight  # type: ignore[arg-type]
        )
        self._record_time("fiedler", t0)
        return fiedler

    # ------------------------------------------------------------------
    def is_connected(self, threshold: float = 1e-4) -> bool:
        """Return True if the graph is (approximately) connected."""
        self.fiedler_value()
        return self._lanczos.is_connected(threshold)

    # ------------------------------------------------------------------
    def laplacian_eigenvectors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the cached Laplacian eigenvectors.

        Returns
        -------
        eigenvalues  : (k,) float32
        eigenvectors : (N, k) float32
        """
        self._check_graph()
        return self._eigen_cache.get_or_compute(
            self._row, self._col, self._weight  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    def gnn_inference(
        self,
        x: torch.Tensor,
        model: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        dirty_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run streaming GNN inference with neighbourhood caching.

        Parameters
        ----------
        x           : (N, D) float32 input features
        model       : callable(x, edge_index, edge_weight) → (N, H)
        dirty_mask  : optional (N,) bool override

        Returns
        -------
        embeddings : (N, H) float32
        """
        self._check_graph()
        t0 = time.perf_counter()
        emb = self._gnn_cache.infer(
            x, torch.stack([self._row, self._col], dim=0),  # type: ignore[arg-type]
            self._weight, model, dirty_mask  # type: ignore[arg-type]
        )
        self._record_time("gnn", t0)
        return emb

    # ------------------------------------------------------------------
    def degree_centrality(self) -> torch.Tensor:
        """Return normalised degree centrality for all nodes."""
        self._check_graph()
        n = self.n
        deg = torch.zeros(n, dtype=torch.float32, device=self.device)
        if self._weight is not None and self._row is not None:
            deg.scatter_add_(0, self._row, self._weight)
        max_deg = float(deg.max().item()) if deg.numel() > 0 else 1.0
        return deg / max(max_deg, 1e-8)

    # ------------------------------------------------------------------
    def betweenness_approx(self, n_samples: int = 20) -> torch.Tensor:
        """
        Approximate betweenness centrality via random-walk sampling.
        (Crude approximation — for fast monitoring only.)
        """
        self._check_graph()
        n = self.n
        betweenness = torch.zeros(n, dtype=torch.float32, device=self.device)

        if self._row is None or self._row.numel() == 0:
            return betweenness

        # Build adjacency list on CPU for BFS
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for r, c in zip(self._row.cpu().tolist(), self._col.cpu().tolist()):
            adj[r].append(c)

        for _ in range(n_samples):
            src = int(torch.randint(0, n, (1,)).item())
            # BFS
            dist = [-1] * n
            path_count = [0] * n
            path_count[src] = 1
            dist[src] = 0
            q = [src]
            order = []
            while q:
                v = q.pop(0)
                order.append(v)
                for u in adj[v]:
                    if dist[u] == -1:
                        dist[u] = dist[v] + 1
                        q.append(u)
                    if dist[u] == dist[v] + 1:
                        path_count[u] += path_count[v]

            # Accumulate
            dep = [0.0] * n
            for v in reversed(order):
                for u in adj[v]:
                    if dist[u] == dist[v] + 1 and path_count[u] > 0:
                        dep[v] += path_count[v] / path_count[u] * (1 + dep[u])
                if v != src:
                    betweenness[v] += dep[v]

        return betweenness / max(n_samples, 1)

    # ------------------------------------------------------------------
    def node_similarity(
        self,
        x: torch.Tensor,
        i: int,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the top-k most similar nodes to node i by feature cosine similarity.

        Returns
        -------
        indices : (k,) int64
        scores  : (k,) float32
        """
        x = x.to(self.device, dtype=torch.float32)
        norms = x.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        xn = x / norms
        sims = xn @ xn[i]     # (N,)
        sims[i] = -1.0        # exclude self
        top_vals, top_idx = sims.topk(top_k)
        return top_idx, top_vals

    # ------------------------------------------------------------------
    def benchmark_summary(self) -> Dict[str, Dict[str, float]]:
        """Return timing stats for all operations."""
        result: Dict[str, Dict[str, float]] = {}
        for op, times in self._op_times.items():
            if not times:
                continue
            s = sorted(times)
            result[op] = {
                "mean_us": sum(s) / len(s) * 1e6,
                "p99_us": s[int(0.99 * len(s))] * 1e6,
                "max_us": s[-1] * 1e6,
                "n_calls": len(s),
            }
        return result

    # ------------------------------------------------------------------
    def _check_graph(self) -> None:
        if self._row is None:
            raise RuntimeError("No graph loaded. Call update_graph() first.")

    # ------------------------------------------------------------------
    def _record_time(self, op: str, t0: float) -> None:
        self._op_times[op].append(time.perf_counter() - t0)
        self._call_count += 1
        if self._call_count % self.cfg.benchmark_interval == 0:
            summary = self.benchmark_summary()
            for op_name, stats in summary.items():
                logger.debug(
                    "FastGraphOps[%s] mean=%.0f μs p99=%.0f μs",
                    op_name, stats["mean_us"], stats["p99_us"],
                )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n_edges = self._row.numel() if self._row is not None else 0
        return (
            f"FastGraphOps(n={self.n}, edges={n_edges}, "
            f"device={self.device})"
        )


# ---------------------------------------------------------------------------
# Normalised cuts computation
# ---------------------------------------------------------------------------

def normalised_cut(
    labels: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    weight: torch.Tensor,
    n: int,
) -> float:
    """
    Compute the normalised cut value for a given graph partitioning.

    Parameters
    ----------
    labels         : (N,) int64 cluster assignments
    row, col       : (E,) int64
    weight         : (E,) float32
    n              : number of nodes

    Returns
    -------
    ncut : float (lower is better)
    """
    if row.numel() == 0:
        return 0.0

    k = int(labels.max().item()) + 1
    ncut = 0.0

    # Total volume per cluster
    deg = torch.zeros(n, device=row.device)
    deg.scatter_add_(0, row, weight)
    label_vol = torch.zeros(k, device=row.device)
    for c in range(k):
        mask = labels == c
        label_vol[c] = deg[mask].sum()

    # Cut between clusters
    for c in range(k):
        cut_c = 0.0
        src_in_c = labels[row] == c
        dst_not_in_c = labels[col] != c
        cross_mask = src_in_c & dst_not_in_c
        cut_c = float(weight[cross_mask].sum().item())
        vol_c = float(label_vol[c].item())
        if vol_c > 1e-8:
            ncut += cut_c / vol_c

    return ncut / k


# ---------------------------------------------------------------------------
# Power iteration for dominant eigenvector (fast community detection)
# ---------------------------------------------------------------------------

def power_iteration_dominant(
    row: torch.Tensor,
    col: torch.Tensor,
    weight: torch.Tensor,
    n: int,
    n_iter: int = 30,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Find the dominant eigenvector of the adjacency matrix via power
    iteration.  Useful for community detection (modularity maximisation).

    Returns
    -------
    v : (N,) float32 — dominant eigenvector
    """
    v = torch.randn(n, device=device)
    v /= v.norm().clamp(min=1e-10)

    row_d = row.to(device)
    col_d = col.to(device)
    w_d = weight.to(device, dtype=torch.float32)

    for _ in range(n_iter):
        # Av
        new_v = torch.zeros(n, device=device)
        if row_d.numel() > 0:
            new_v.scatter_add_(0, col_d, w_d * v[row_d])
        norm = new_v.norm().clamp(min=1e-10)
        v = new_v / norm

    return v


# ---------------------------------------------------------------------------
# Chebyshev graph convolution (fast, no eigen needed)
# ---------------------------------------------------------------------------

class ChebyshevConv(nn.Module):
    """
    Chebyshev spectral graph convolution.

    Approximates graph convolution using K-th order Chebyshev polynomials:
        y = Σ_{k=0}^{K} θ_k T_k(L̃) x
    where L̃ = 2L/λ_max - I (scaled Laplacian).

    No eigendecomposition required — O(K * E) per forward pass.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K

        self.weight = nn.Parameter(
            torch.Tensor(K, in_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self._init_params()

    # ------------------------------------------------------------------
    def _init_params(self) -> None:
        nn.init.xavier_uniform_(self.weight.view(self.K * self.in_channels, self.out_channels))
        nn.init.zeros_(self.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        lambda_max: float = 2.0,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x           : (N, in_channels)
        edge_index  : (2, E) int64
        edge_weight : (E,) float32
        lambda_max  : max eigenvalue estimate (default 2 for normalised L)

        Returns
        -------
        (N, out_channels)
        """
        N = x.shape[0]
        dev = x.device
        row, col = edge_index[0], edge_index[1]

        # Build normalised Laplacian: L̃ = 2/λ_max * L - I
        deg = torch.zeros(N, device=dev)
        deg.scatter_add_(0, row, edge_weight)
        d_inv_sqrt = deg.pow(-0.5).clamp(max=1e4)

        # L_norm = I - D^{-1/2} A D^{-1/2}
        # L_tilde = 2/lambda_max * L_norm - I
        # We apply L_tilde as a sparse operator

        def apply_L_tilde(v: torch.Tensor) -> torch.Tensor:
            """Apply L̃ to each column of v: (N, C) → (N, C)"""
            # A v
            Av = torch.zeros_like(v)
            Av.scatter_add_(0, col.unsqueeze(1).expand(-1, v.shape[1]),
                            (edge_weight * d_inv_sqrt[row] * d_inv_sqrt[col]).unsqueeze(1) * v[row])
            # L_norm v = v - Av
            Lv = v - Av
            # L_tilde v = 2/lambda_max * Lv - v
            return (2.0 / lambda_max) * Lv - v

        # Chebyshev recursion: T_0=I, T_1=L̃, T_k=2*L̃*T_{k-1} - T_{k-2}
        T_prev_prev = x                        # T_0 x
        T_prev = apply_L_tilde(x)              # T_1 x

        out = x @ self.weight[0] + T_prev @ self.weight[1]

        for k in range(2, self.K):
            T_cur = 2.0 * apply_L_tilde(T_prev) - T_prev_prev
            out = out + T_cur @ self.weight[k % self.K]
            T_prev_prev = T_prev
            T_prev = T_cur

        return out + self.bias


# ---------------------------------------------------------------------------
# Degree sequence statistics
# ---------------------------------------------------------------------------

def degree_statistics(
    row: torch.Tensor,
    weight: torch.Tensor,
    n: int,
) -> Dict[str, float]:
    """
    Compute summary statistics of the degree sequence.

    Returns
    -------
    dict with mean, std, min, max, gini coefficient
    """
    deg = torch.zeros(n, dtype=torch.float32, device=row.device)
    if row.numel() > 0:
        deg.scatter_add_(0, row, weight)

    deg_float = deg.cpu()
    d_sorted = deg_float.sort().values
    n_nodes = float(n)

    # Gini coefficient
    cumsum = d_sorted.cumsum(0)
    total = float(cumsum[-1].item()) if n > 0 else 1.0
    if total < 1e-8:
        gini = 0.0
    else:
        gini = float(
            (2 * (torch.arange(1, n + 1, dtype=torch.float32) * d_sorted).sum()
             - (n + 1) * d_sorted.sum()) / (n * total)
        )

    return {
        "mean": float(deg_float.mean().item()),
        "std": float(deg_float.std().item()) if n > 1 else 0.0,
        "min": float(d_sorted[0].item()) if n > 0 else 0.0,
        "max": float(d_sorted[-1].item()) if n > 0 else 0.0,
        "gini": max(0.0, min(1.0, gini)),
    }


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "FastGraphOpsConfig",
    "FastGraphOps",
    "LaplacianEigenCache",
    "ApproximatePageRank",
    "StreamingGNNInference",
    "IncrementalSpectralClustering",
    "LanczosFiedler",
    "ChebyshevConv",
    "normalised_cut",
    "power_iteration_dominant",
    "degree_statistics",
]
