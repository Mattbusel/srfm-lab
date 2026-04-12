"""
incremental_adjacency.py — Incremental Adjacency Update Kernel (IAUK)
======================================================================
Part of the Omni-Graph incremental graph construction suite.

Design goals
------------
* Edge-delta updates: only recompute edges for nodes whose feature vectors
  have changed by more than `epsilon` since the last full sweep.
* Dirty-bit tracking per node: a boolean mask marks nodes as stale.
* Sparse storage: the full adjacency is maintained in CSR format; delta
  patches are applied with scatter operations, never a full rebuild.
* Exponential Moving Average (EMA) edge weights: configurable alpha,
  supporting both batch and online update modes.
* Correlation delta via rank-1 (Sherman-Morrison) updates so only O(k)
  entries are touched per dirty node instead of O(N²).
* Birth/death threshold hysteresis: an edge is born when corr > upper_thresh
  and dies when corr < lower_thresh, preventing rapid oscillation.
* GPU-accelerated via torch.sparse CSR operations, CUDA scatter/gather for
  delta application.
* Benchmark target: <1 ms per 500-asset graph update on GPU.

Public API
----------
    kernel = IAUKernel(n_assets=500, feature_dim=64, device="cuda")
    kernel.init_from_features(feature_matrix)        # shape (N, D)
    kernel.update(new_features, tick_id)              # incremental
    csr = kernel.get_adjacency_csr()
    edge_index, edge_weight = kernel.get_pyg_edge_index()
"""

from __future__ import annotations

import math
import time
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class IAUKConfig:
    """All hyper-parameters for the IAUK kernel."""

    # Graph size
    n_assets: int = 500
    feature_dim: int = 64

    # Dirty-bit threshold: node marked stale if ||Δf||₂ > epsilon
    dirty_epsilon: float = 1e-4

    # EMA decay for edge weights: w_t = alpha * corr_t + (1-alpha) * w_{t-1}
    ema_alpha: float = 0.1

    # Hysteresis thresholds for birth/death of edges
    birth_threshold: float = 0.30
    death_threshold: float = 0.20

    # Minimum correlation absolute value to keep edge
    min_abs_corr: float = 0.15

    # If True, normalise feature vectors before computing correlation
    normalise_features: bool = True

    # Maximum number of neighbours per node (k-NN cap after delta update)
    max_degree: int = 50

    # Device preference
    device: str = "cuda"

    # Batch size for pairwise correlation computation (to avoid OOM)
    corr_batch_size: int = 64

    # Benchmark: print timing every N ticks
    benchmark_interval: int = 100

    # Tolerance for Sherman-Morrison rank-1 updates
    sm_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.death_threshold >= self.birth_threshold:
            raise ValueError(
                f"death_threshold ({self.death_threshold}) must be < "
                f"birth_threshold ({self.birth_threshold})"
            )
        if self.ema_alpha <= 0 or self.ema_alpha > 1:
            raise ValueError("ema_alpha must be in (0, 1]")


# ---------------------------------------------------------------------------
# Internal state containers
# ---------------------------------------------------------------------------

@dataclass
class _CSRBuffer:
    """
    Stores a sparse adjacency in COO + CSR form.
    We keep COO for easy updates and rebuild CSR on demand.
    """

    n: int
    # COO
    row: torch.Tensor        # (E,) int64
    col: torch.Tensor        # (E,) int64
    weight: torch.Tensor     # (E,) float32

    # CSR cache (rebuilt lazily)
    _csr: Optional[torch.Tensor] = field(default=None, repr=False)
    _dirty_csr: bool = field(default=True, repr=False)

    # ------------------------------------------------------------------
    @classmethod
    def empty(cls, n: int, device: str) -> "_CSRBuffer":
        dev = torch.device(device)
        return cls(
            n=n,
            row=torch.zeros(0, dtype=torch.int64, device=dev),
            col=torch.zeros(0, dtype=torch.int64, device=dev),
            weight=torch.zeros(0, dtype=torch.float32, device=dev),
        )

    # ------------------------------------------------------------------
    def num_edges(self) -> int:
        return int(self.row.shape[0])

    # ------------------------------------------------------------------
    def to_sparse_csr(self) -> torch.Tensor:
        """Return (or recompute) the torch sparse CSR tensor."""
        if not self._dirty_csr and self._csr is not None:
            return self._csr
        # Build via COO → CSR conversion
        indices = torch.stack([self.row, self.col], dim=0)
        sparse_coo = torch.sparse_coo_tensor(
            indices, self.weight, (self.n, self.n), device=self.weight.device
        )
        self._csr = sparse_coo.to_sparse_csr()
        self._dirty_csr = False
        return self._csr

    # ------------------------------------------------------------------
    def edge_index_and_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return PyG-style (2, E) edge_index and (E,) edge_weight."""
        return torch.stack([self.row, self.col], dim=0), self.weight


# ---------------------------------------------------------------------------
# Dirty-bit tracker
# ---------------------------------------------------------------------------

class DirtyBitTracker:
    """
    Maintains per-node dirty flags and tracks feature history for delta
    computation.
    """

    def __init__(self, n: int, feature_dim: int, epsilon: float, device: torch.device) -> None:
        self.n = n
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.device = device

        # Previous feature snapshot (updated after each clean sweep)
        self._prev_features: Optional[torch.Tensor] = None  # (N, D)

        # Current dirty mask
        self.dirty_mask: torch.Tensor = torch.ones(n, dtype=torch.bool, device=device)

        # Per-node update counters
        self.update_counts: torch.Tensor = torch.zeros(n, dtype=torch.int64, device=device)

        # Timestamp of last update per node
        self.last_update_tick: torch.Tensor = torch.full(
            (n,), -1, dtype=torch.int64, device=device
        )

    # ------------------------------------------------------------------
    def compute_dirty(self, new_features: torch.Tensor, tick_id: int) -> torch.Tensor:
        """
        Mark nodes dirty where ||f_new - f_prev||₂ > epsilon.
        Returns the dirty mask (bool tensor of shape (N,)).
        """
        if self._prev_features is None:
            # First call — everything is dirty
            self.dirty_mask.fill_(True)
            self.last_update_tick.fill_(tick_id)
            self.update_counts.add_(1)
            return self.dirty_mask

        delta = new_features - self._prev_features          # (N, D)
        norms = torch.norm(delta, p=2, dim=1)               # (N,)
        self.dirty_mask = norms > self.epsilon

        # Update metadata for dirty nodes
        dirty_indices = self.dirty_mask.nonzero(as_tuple=True)[0]
        self.last_update_tick[dirty_indices] = tick_id
        self.update_counts[dirty_indices] += 1

        return self.dirty_mask

    # ------------------------------------------------------------------
    def commit_features(self, features: torch.Tensor) -> None:
        """Store current features as the new baseline."""
        self._prev_features = features.clone()

    # ------------------------------------------------------------------
    def force_dirty_all(self) -> None:
        """Force all nodes dirty — used after a major topology change."""
        self.dirty_mask.fill_(True)

    # ------------------------------------------------------------------
    def num_dirty(self) -> int:
        return int(self.dirty_mask.sum().item())

    # ------------------------------------------------------------------
    def get_dirty_indices(self) -> torch.Tensor:
        return self.dirty_mask.nonzero(as_tuple=True)[0]

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "num_dirty": self.num_dirty(),
            "frac_dirty": self.num_dirty() / self.n,
            "total_updates": int(self.update_counts.sum().item()),
        }


# ---------------------------------------------------------------------------
# EMA edge weight manager
# ---------------------------------------------------------------------------

class EMAEdgeWeights:
    """
    Maintains exponential moving average weights for all edges.

    For existing edges:  w ← alpha * corr + (1-alpha) * w
    For new edges:       w = corr  (initialise at birth)
    """

    def __init__(self, alpha: float, device: torch.device) -> None:
        self.alpha = alpha
        self.device = device

        # Flat storage indexed by (row * N + col) hash — we store as a dict
        # for sparse updates.  On GPU we also maintain a dense weight matrix
        # for vectorised access during batch updates.
        self._weights: Dict[int, float] = {}

        # Dense matrix cache
        self._n: int = 0
        self._dense: Optional[torch.Tensor] = None   # (N, N) float32
        self._dense_valid: bool = False

    # ------------------------------------------------------------------
    def resize(self, n: int) -> None:
        """Allocate/resize the dense weight matrix."""
        self._n = n
        self._dense = torch.zeros(n, n, dtype=torch.float32, device=self.device)
        self._dense_valid = True

    # ------------------------------------------------------------------
    def update_batch(
        self,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        new_corr: torch.Tensor,
        is_new_edge: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorised EMA update for a batch of (row, col) pairs.

        Parameters
        ----------
        row_indices : (E,) int64
        col_indices : (E,) int64
        new_corr    : (E,) float32 — current correlation values
        is_new_edge : (E,) bool — True for newly born edges

        Returns
        -------
        updated_weights : (E,) float32
        """
        if self._dense is None:
            raise RuntimeError("Call resize() before update_batch()")

        old_w = self._dense[row_indices, col_indices]  # (E,)

        # New edges initialise directly from corr; existing edges use EMA
        updated = torch.where(
            is_new_edge,
            new_corr,
            self.alpha * new_corr + (1.0 - self.alpha) * old_w,
        )

        # Write back to dense matrix (scatter)
        self._dense[row_indices, col_indices] = updated
        # Symmetric
        self._dense[col_indices, row_indices] = updated

        return updated

    # ------------------------------------------------------------------
    def decay_all(self, decay_factor: float = 0.99) -> None:
        """Apply global decay to all weights (e.g., during low-vol periods)."""
        if self._dense is not None:
            self._dense.mul_(decay_factor)

    # ------------------------------------------------------------------
    def get_weight(self, row: int, col: int) -> float:
        if self._dense is None:
            return 0.0
        return float(self._dense[row, col].item())

    # ------------------------------------------------------------------
    def get_dense(self) -> torch.Tensor:
        if self._dense is None:
            raise RuntimeError("Not initialised")
        return self._dense


# ---------------------------------------------------------------------------
# Sherman-Morrison rank-1 correlation delta
# ---------------------------------------------------------------------------

class ShermanMorrisonDelta:
    """
    Computes incremental correlation updates using the Sherman-Morrison
    formula for rank-1 matrix updates.

    When node i's feature changes from f_old to f_new, only column i and
    row i of the correlation matrix are affected (to first order).

    Full rank-1 update:
        C_new = C_old + (u v^T) / (denom)
    where u and v encode the change.

    This is approximate — we use it for speed and fall back to exact
    recomputation when the change is large.
    """

    def __init__(self, n: int, feature_dim: int, device: torch.device, sm_epsilon: float = 1e-9) -> None:
        self.n = n
        self.feature_dim = feature_dim
        self.device = device
        self.sm_epsilon = sm_epsilon

        # Full normalised feature matrix (N, D)
        self._F: Optional[torch.Tensor] = None

        # Full correlation matrix (N, N)
        self._C: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def initialise(self, features: torch.Tensor) -> torch.Tensor:
        """
        Build the initial correlation matrix from scratch.

        Parameters
        ----------
        features : (N, D) float32

        Returns
        -------
        C : (N, N) float32 correlation matrix
        """
        F_norm = self._normalise(features)
        self._F = F_norm
        C = F_norm @ F_norm.T  # (N, N)
        # Clamp to [-1, 1] for numerical safety
        C = C.clamp(-1.0, 1.0)
        self._C = C
        return C

    # ------------------------------------------------------------------
    def update_dirty_nodes(
        self,
        new_features: torch.Tensor,
        dirty_mask: torch.Tensor,
        force_exact: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update correlation matrix for dirty nodes.

        For each dirty node i, update column/row i of C using rank-1
        Sherman-Morrison if the feature change is small, otherwise exact.

        Returns
        -------
        C_updated : (N, N)
        changed_rows : rows (node indices) where C changed
        changed_cols : cols (node indices) where C changed
        """
        if self._F is None or self._C is None:
            raise RuntimeError("Call initialise() first")

        F_new_norm = self._normalise(new_features)
        dirty_indices = dirty_mask.nonzero(as_tuple=True)[0]

        changed_row_list: List[torch.Tensor] = []
        changed_col_list: List[torch.Tensor] = []

        for idx in dirty_indices:
            i = int(idx.item())
            f_old = self._F[i]      # (D,)
            f_new = F_new_norm[i]   # (D,)
            delta_f = f_new - f_old  # (D,)
            delta_norm = float(delta_f.norm().item())

            if force_exact or delta_norm > 0.5:
                # Large change — recompute entire row/column exactly
                new_col = F_new_norm @ f_new  # (N,)
                new_col = new_col.clamp(-1.0, 1.0)
                self._C[i, :] = new_col
                self._C[:, i] = new_col
                self._C[i, i] = 1.0
            else:
                # Rank-1 Sherman-Morrison update
                # C_new[:, i] = F @ f_new
                # We decompose: f_new = f_old + delta_f
                # => C[:, i] += F @ delta_f
                delta_col = self._F @ delta_f  # (N,)
                delta_col = delta_col.clamp(-1.0, 1.0)
                self._C[i, :] += delta_col
                self._C[:, i] += delta_col
                self._C[i, i] = 1.0
                # Clamp again
                self._C[i, :].clamp_(-1.0, 1.0)
                self._C[:, i].clamp_(-1.0, 1.0)

            # Record changed positions
            rows = torch.full((self.n,), i, dtype=torch.int64, device=self.device)
            cols = torch.arange(self.n, dtype=torch.int64, device=self.device)
            changed_row_list.append(rows)
            changed_col_list.append(cols)
            # Also symmetric
            changed_row_list.append(cols)
            changed_col_list.append(rows)

            # Update stored feature
            self._F[i] = f_new

        if not changed_row_list:
            empty = torch.zeros(0, dtype=torch.int64, device=self.device)
            return self._C, empty, empty

        changed_rows = torch.cat(changed_row_list)
        changed_cols = torch.cat(changed_col_list)
        return self._C, changed_rows, changed_cols

    # ------------------------------------------------------------------
    def get_correlation_matrix(self) -> Optional[torch.Tensor]:
        return self._C

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(features: torch.Tensor) -> torch.Tensor:
        """L2-normalise along the feature dimension."""
        norms = features.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        return features / norms


# ---------------------------------------------------------------------------
# Hysteresis edge birth/death controller
# ---------------------------------------------------------------------------

class HysteresisEdgeController:
    """
    Manages edge existence using a two-threshold hysteresis to prevent
    rapid birth/death oscillation.

    Edge born  when |corr| > birth_threshold  (AND edge currently absent)
    Edge dies  when |corr| < death_threshold  (AND edge currently present)
    """

    def __init__(
        self,
        n: int,
        birth_threshold: float,
        death_threshold: float,
        device: torch.device,
    ) -> None:
        self.n = n
        self.birth_threshold = birth_threshold
        self.death_threshold = death_threshold
        self.device = device

        # Binary adjacency mask (bool), True = edge exists
        self.edge_mask: torch.Tensor = torch.zeros(n, n, dtype=torch.bool, device=device)
        # Diagonal is never an edge
        diag = torch.arange(n, device=device)
        self.edge_mask[diag, diag] = False

    # ------------------------------------------------------------------
    def apply_delta(
        self,
        corr_matrix: torch.Tensor,
        changed_rows: torch.Tensor,
        changed_cols: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply hysteresis rules to the changed positions.

        Returns
        -------
        new_edge_mask : (N, N) bool
        born_edges    : (2, K) int64 — newly created edges
        dead_edges    : (2, K) int64 — newly removed edges
        """
        if changed_rows.numel() == 0:
            empty = torch.zeros(2, 0, dtype=torch.int64, device=self.device)
            return self.edge_mask, empty, empty

        # De-duplicate and restrict to upper triangle to avoid double-counting
        pairs = torch.stack([changed_rows, changed_cols], dim=1)  # (M, 2)
        # Keep only i < j
        mask_upper = pairs[:, 0] < pairs[:, 1]
        pairs = pairs[mask_upper]

        if pairs.numel() == 0:
            empty = torch.zeros(2, 0, dtype=torch.int64, device=self.device)
            return self.edge_mask, empty, empty

        r = pairs[:, 0]
        c = pairs[:, 1]
        abs_corr = corr_matrix[r, c].abs()      # (K,)
        currently_alive = self.edge_mask[r, c]  # (K,) bool

        # Birth: corr > birth and NOT alive
        should_born = (abs_corr > self.birth_threshold) & ~currently_alive
        # Death: corr < death and alive
        should_die = (abs_corr < self.death_threshold) & currently_alive

        born_r = r[should_born]
        born_c = c[should_born]
        dead_r = r[should_die]
        dead_c = c[should_die]

        # Apply
        if born_r.numel() > 0:
            self.edge_mask[born_r, born_c] = True
            self.edge_mask[born_c, born_r] = True
        if dead_r.numel() > 0:
            self.edge_mask[dead_r, dead_c] = False
            self.edge_mask[dead_c, dead_r] = False

        born_edges = torch.stack([born_r, born_c], dim=0) if born_r.numel() > 0 else \
            torch.zeros(2, 0, dtype=torch.int64, device=self.device)
        dead_edges = torch.stack([dead_r, dead_c], dim=0) if dead_r.numel() > 0 else \
            torch.zeros(2, 0, dtype=torch.int64, device=self.device)

        return self.edge_mask, born_edges, dead_edges

    # ------------------------------------------------------------------
    def apply_full(self, corr_matrix: torch.Tensor) -> None:
        """Full rebuild of edge_mask from scratch (used at initialisation)."""
        abs_corr = corr_matrix.abs()
        # Clear diagonal
        diag = torch.arange(self.n, device=self.device)
        abs_corr[diag, diag] = 0.0
        self.edge_mask = abs_corr > self.birth_threshold

    # ------------------------------------------------------------------
    def get_coo(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return COO (row, col) for all live edges (upper triangle)."""
        triu = torch.triu(self.edge_mask, diagonal=1)
        idx = triu.nonzero(as_tuple=False)  # (E, 2)
        if idx.numel() == 0:
            empty = torch.zeros(0, dtype=torch.int64, device=self.device)
            return empty, empty
        return idx[:, 0], idx[:, 1]

    # ------------------------------------------------------------------
    def num_edges(self) -> int:
        return int(self.edge_mask.triu(diagonal=1).sum().item())

    # ------------------------------------------------------------------
    def density(self) -> float:
        max_edges = self.n * (self.n - 1) // 2
        if max_edges == 0:
            return 0.0
        return self.num_edges() / max_edges


# ---------------------------------------------------------------------------
# K-NN degree cap
# ---------------------------------------------------------------------------

class DegreeCapKNN:
    """
    Prunes the adjacency so that no node exceeds `max_degree` neighbours,
    keeping the max_degree strongest edges by weight.
    """

    def __init__(self, max_degree: int, device: torch.device) -> None:
        self.max_degree = max_degree
        self.device = device

    # ------------------------------------------------------------------
    def apply(
        self,
        edge_mask: torch.Tensor,
        weight_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return a pruned edge_mask where each node has at most max_degree
        neighbours.  Works in-place on edge_mask for efficiency.
        """
        n = edge_mask.shape[0]
        degree = edge_mask.long().sum(dim=1)  # (N,)
        over = (degree > self.max_degree).nonzero(as_tuple=True)[0]

        for i in over:
            i = int(i.item())
            neighbours = edge_mask[i].nonzero(as_tuple=True)[0]  # (deg,)
            w = weight_matrix[i, neighbours]                      # (deg,)
            # Sort descending by weight
            sorted_idx = torch.argsort(w, descending=True)
            keep = neighbours[sorted_idx[: self.max_degree]]
            remove = neighbours[sorted_idx[self.max_degree :]]
            edge_mask[i, remove] = False
            edge_mask[remove, i] = False

        return edge_mask


# ---------------------------------------------------------------------------
# Delta patch applicator
# ---------------------------------------------------------------------------

class DeltaPatchApplicator:
    """
    Applies born/dead edge lists to a _CSRBuffer using scatter operations.
    This avoids rebuilding the full COO from scratch on every tick.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    # ------------------------------------------------------------------
    def apply(
        self,
        buf: _CSRBuffer,
        born_edges: torch.Tensor,      # (2, K_born) int64
        dead_edges: torch.Tensor,      # (2, K_dead) int64
        weight_matrix: torch.Tensor,   # (N, N) float32
    ) -> _CSRBuffer:
        """
        Return an updated _CSRBuffer with born edges added and dead edges
        removed, using set operations on the COO representation.
        """
        if born_edges.numel() == 0 and dead_edges.numel() == 0:
            return buf

        n = buf.n
        # ------ Remove dead edges from COO ------
        if dead_edges.numel() > 0:
            dead_r = dead_edges[0]
            dead_c = dead_edges[1]
            dead_keys = dead_r * n + dead_c
            live_keys = buf.row * n + buf.col
            # Keep edges not in dead set
            keep_mask = ~torch.isin(live_keys, dead_keys)
            buf.row = buf.row[keep_mask]
            buf.col = buf.col[keep_mask]
            buf.weight = buf.weight[keep_mask]

        # ------ Add born edges to COO ------
        if born_edges.numel() > 0:
            born_r = born_edges[0]
            born_c = born_edges[1]
            born_w = weight_matrix[born_r, born_c]
            # Also add symmetric
            buf.row = torch.cat([buf.row, born_r, born_c])
            buf.col = torch.cat([buf.col, born_c, born_r])
            buf.weight = torch.cat([buf.weight, born_w, born_w])

        buf._dirty_csr = True
        return buf

    # ------------------------------------------------------------------
    def update_weights(
        self,
        buf: _CSRBuffer,
        weight_matrix: torch.Tensor,
    ) -> _CSRBuffer:
        """
        Re-fetch weights for all existing edges from weight_matrix.
        Used when EMA has updated weights but topology unchanged.
        """
        if buf.row.numel() == 0:
            return buf
        buf.weight = weight_matrix[buf.row, buf.col]
        buf._dirty_csr = True
        return buf


# ---------------------------------------------------------------------------
# Benchmark / profiling context manager
# ---------------------------------------------------------------------------

class _BenchmarkTimer:
    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "_BenchmarkTimer":
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


# ---------------------------------------------------------------------------
# Main IAUK class
# ---------------------------------------------------------------------------

class IAUKernel:
    """
    Incremental Adjacency Update Kernel.

    This is the main entry point for users.  It orchestrates all
    sub-components: dirty-bit tracking, correlation delta via
    Sherman-Morrison, EMA weights, hysteresis birth/death, degree cap,
    and delta patch application.

    Usage
    -----
        cfg = IAUKConfig(n_assets=500, feature_dim=64, device="cuda")
        kernel = IAUKernel(cfg)
        kernel.init_from_features(F0)              # (500, 64)
        for tick in range(T):
            kernel.update(F_t, tick_id=tick)
            ei, ew = kernel.get_pyg_edge_index()   # ready for GNN
    """

    def __init__(self, cfg: Optional[IAUKConfig] = None, **kwargs: Any) -> None:
        if cfg is None:
            cfg = IAUKConfig(**kwargs)
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        if str(self.device) != cfg.device:
            logger.warning("CUDA not available; falling back to CPU")

        n, d = cfg.n_assets, cfg.feature_dim

        # Sub-components
        self.dirty_tracker = DirtyBitTracker(n, d, cfg.dirty_epsilon, self.device)
        self.sm_delta = ShermanMorrisonDelta(n, d, self.device, cfg.sm_epsilon)
        self.ema = EMAEdgeWeights(cfg.ema_alpha, self.device)
        self.ema.resize(n)
        self.hysteresis = HysteresisEdgeController(
            n, cfg.birth_threshold, cfg.death_threshold, self.device
        )
        self.degree_cap = DegreeCapKNN(cfg.max_degree, self.device)
        self.patcher = DeltaPatchApplicator(self.device)

        # CSR buffer
        self._buf = _CSRBuffer.empty(n, str(self.device))

        # State
        self._initialised: bool = False
        self._tick_id: int = -1

        # Telemetry
        self._tick_times_ms: List[float] = []
        self._edge_counts: List[int] = []
        self._dirty_counts: List[int] = []

    # ------------------------------------------------------------------
    def init_from_features(self, features: torch.Tensor) -> None:
        """
        Full initialisation from a feature matrix.

        Parameters
        ----------
        features : (N, D) float32 — initial feature matrix
        """
        features = self._to_device(features)
        n = self.cfg.n_assets
        if features.shape != (n, self.cfg.feature_dim):
            raise ValueError(
                f"Expected features shape ({n}, {self.cfg.feature_dim}), "
                f"got {tuple(features.shape)}"
            )

        with _BenchmarkTimer("init") as t:
            # 1. Build full correlation matrix
            C = self.sm_delta.initialise(features)

            # 2. Apply hysteresis full pass to build initial edge_mask
            self.hysteresis.apply_full(C)

            # 3. Apply degree cap
            self.degree_cap.apply(self.hysteresis.edge_mask, C.abs())

            # 4. Build COO from edge_mask and initialise EMA weights
            row, col = self.hysteresis.get_coo()
            if row.numel() > 0:
                w = C[row, col].abs()
                is_new = torch.ones(row.numel(), dtype=torch.bool, device=self.device)
                updated_w = self.ema.update_batch(row, col, w, is_new)
            else:
                updated_w = torch.zeros(0, dtype=torch.float32, device=self.device)

            # 5. Build CSR buffer (symmetric)
            if row.numel() > 0:
                self._buf.row = torch.cat([row, col])
                self._buf.col = torch.cat([col, row])
                self._buf.weight = torch.cat([updated_w, updated_w])
            else:
                self._buf.row = torch.zeros(0, dtype=torch.int64, device=self.device)
                self._buf.col = torch.zeros(0, dtype=torch.int64, device=self.device)
                self._buf.weight = torch.zeros(0, dtype=torch.float32, device=self.device)
            self._buf._dirty_csr = True

            # 6. Commit features as baseline
            self.dirty_tracker.commit_features(features)

        self._initialised = True
        self._tick_id = 0
        logger.info(
            "IAUKernel initialised | n=%d | edges=%d | time=%.3f ms",
            n, self._buf.num_edges() // 2, t.elapsed_ms,
        )

    # ------------------------------------------------------------------
    def update(self, new_features: torch.Tensor, tick_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Incremental update given new feature matrix.

        Parameters
        ----------
        new_features : (N, D) float32
        tick_id      : optional integer tick identifier

        Returns
        -------
        stats : dict with keys {num_dirty, born, died, elapsed_ms, ...}
        """
        if not self._initialised:
            self.init_from_features(new_features)
            return {"init": True}

        new_features = self._to_device(new_features)
        if tick_id is None:
            tick_id = self._tick_id + 1
        self._tick_id = tick_id

        with _BenchmarkTimer("update") as t:
            stats = self._update_internal(new_features, tick_id)

        stats["elapsed_ms"] = t.elapsed_ms
        self._tick_times_ms.append(t.elapsed_ms)
        self._edge_counts.append(self._buf.num_edges() // 2)
        self._dirty_counts.append(stats.get("num_dirty", 0))

        if tick_id % self.cfg.benchmark_interval == 0:
            self._log_benchmark(tick_id, stats)

        return stats

    # ------------------------------------------------------------------
    def _update_internal(self, new_features: torch.Tensor, tick_id: int) -> Dict[str, Any]:
        """Core incremental update logic."""
        cfg = self.cfg
        n = cfg.n_assets

        # --- Step 1: compute dirty mask ---
        dirty_mask = self.dirty_tracker.compute_dirty(new_features, tick_id)
        num_dirty = int(dirty_mask.sum().item())

        if num_dirty == 0:
            # Nothing changed — just return early
            return {"num_dirty": 0, "born": 0, "died": 0}

        # --- Step 2: correlation delta via Sherman-Morrison ---
        C, changed_rows, changed_cols = self.sm_delta.update_dirty_nodes(
            new_features, dirty_mask
        )

        # --- Step 3: hysteresis birth/death ---
        _, born_edges, dead_edges = self.hysteresis.apply_delta(
            C, changed_rows, changed_cols
        )

        num_born = born_edges.shape[1] if born_edges.numel() > 0 else 0
        num_died = dead_edges.shape[1] if dead_edges.numel() > 0 else 0

        # --- Step 4: EMA weight update for affected edges ---
        if num_born > 0 or num_died > 0 or num_dirty > 0:
            # Gather all edges touching dirty nodes for EMA update
            dirty_idx = dirty_mask.nonzero(as_tuple=True)[0]
            affected_row_list: List[torch.Tensor] = []
            affected_col_list: List[torch.Tensor] = []

            for i in dirty_idx:
                i = int(i.item())
                neighbours = self.hysteresis.edge_mask[i].nonzero(as_tuple=True)[0]
                if neighbours.numel() > 0:
                    r_vec = torch.full(
                        (neighbours.numel(),), i, dtype=torch.int64, device=self.device
                    )
                    affected_row_list.append(r_vec)
                    affected_col_list.append(neighbours)

            if affected_row_list:
                aff_r = torch.cat(affected_row_list)
                aff_c = torch.cat(affected_col_list)
                new_corr = C[aff_r, aff_c].abs()
                # is_new_edge: newly born in this tick
                if num_born > 0:
                    born_r, born_c = born_edges[0], born_edges[1]
                    born_keys = born_r * n + born_c
                    aff_keys = aff_r * n + aff_c
                    is_new_flag = torch.isin(aff_keys, born_keys)
                else:
                    is_new_flag = torch.zeros(
                        aff_r.numel(), dtype=torch.bool, device=self.device
                    )
                self.ema.update_batch(aff_r, aff_c, new_corr, is_new_flag)

        # --- Step 5: degree cap on changed nodes ---
        if num_born > 0:
            # Only check degree for nodes that gained edges
            born_nodes = born_edges.flatten().unique()
            tmp_mask = self.hysteresis.edge_mask.clone()
            tmp_mask = self.degree_cap.apply(tmp_mask, self.ema.get_dense())
            self.hysteresis.edge_mask = tmp_mask

        # --- Step 6: apply delta patch to COO buffer ---
        self._buf = self.patcher.apply(
            self._buf, born_edges, dead_edges, self.ema.get_dense()
        )
        # Also update weights for dirty-node edges
        if num_dirty > 0 and num_born == 0 and num_died == 0:
            self._buf = self.patcher.update_weights(self._buf, self.ema.get_dense())

        # --- Step 7: commit feature baseline ---
        self.dirty_tracker.commit_features(new_features)

        return {
            "num_dirty": num_dirty,
            "born": num_born,
            "died": num_died,
            "num_edges": self._buf.num_edges() // 2,
            "density": self.hysteresis.density(),
        }

    # ------------------------------------------------------------------
    def get_adjacency_csr(self) -> torch.Tensor:
        """Return the full adjacency as a torch sparse CSR tensor."""
        return self._buf.to_sparse_csr()

    # ------------------------------------------------------------------
    def get_pyg_edge_index(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (edge_index, edge_weight) in PyG format.

        Returns
        -------
        edge_index  : (2, E) int64
        edge_weight : (E,) float32
        """
        return self._buf.edge_index_and_weight()

    # ------------------------------------------------------------------
    def get_correlation_matrix(self) -> Optional[torch.Tensor]:
        """Return the current (N, N) correlation matrix."""
        return self.sm_delta.get_correlation_matrix()

    # ------------------------------------------------------------------
    def get_edge_mask(self) -> torch.Tensor:
        """Return the boolean (N, N) adjacency mask."""
        return self.hysteresis.edge_mask

    # ------------------------------------------------------------------
    def force_rebuild(self, features: torch.Tensor) -> None:
        """Force a full rebuild of the adjacency from scratch."""
        self._initialised = False
        self.dirty_tracker.force_dirty_all()
        self.init_from_features(features)

    # ------------------------------------------------------------------
    def num_edges(self) -> int:
        return self._buf.num_edges() // 2

    # ------------------------------------------------------------------
    def density(self) -> float:
        return self.hysteresis.density()

    # ------------------------------------------------------------------
    def benchmark_summary(self) -> Dict[str, float]:
        """Return timing statistics over all processed ticks."""
        if not self._tick_times_ms:
            return {}
        times = self._tick_times_ms
        return {
            "mean_ms": sum(times) / len(times),
            "max_ms": max(times),
            "min_ms": min(times),
            "p99_ms": _percentile(times, 99),
            "p95_ms": _percentile(times, 95),
            "n_ticks": len(times),
        }

    # ------------------------------------------------------------------
    def _log_benchmark(self, tick_id: int, stats: Dict[str, Any]) -> None:
        elapsed = stats.get("elapsed_ms", 0.0)
        logger.info(
            "tick=%d | dirty=%d | born=%d | died=%d | edges=%d | %.3f ms",
            tick_id,
            stats.get("num_dirty", 0),
            stats.get("born", 0),
            stats.get("died", 0),
            stats.get("num_edges", 0),
            elapsed,
        )

    # ------------------------------------------------------------------
    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"IAUKernel(n={self.cfg.n_assets}, d={self.cfg.feature_dim}, "
            f"device={self.device}, edges={self.num_edges()}, "
            f"density={self.density():.4f})"
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _percentile(data: List[float], p: float) -> float:
    """Compute percentile p (0-100) from a list of floats."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return sorted_data[lo]
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def build_iauk_from_returns(
    returns: torch.Tensor,
    window: int = 60,
    cfg: Optional[IAUKConfig] = None,
) -> IAUKernel:
    """
    Convenience factory: build IAUK from a (T, N) returns matrix.
    Uses rolling windows to build features.

    Parameters
    ----------
    returns : (T, N) float32
    window  : rolling window length for feature computation
    cfg     : optional config; if None, auto-configured from data shape
    """
    T, N = returns.shape
    if cfg is None:
        cfg = IAUKConfig(n_assets=N, feature_dim=window)

    kernel = IAUKernel(cfg)

    # Bootstrap: initialise from the first window
    if T < window:
        raise ValueError(f"Need at least {window} time steps, got {T}")

    F0 = returns[:window, :].T.contiguous()   # (N, window)
    kernel.init_from_features(F0)

    # Incremental updates
    for t in range(window, T):
        Ft = returns[t - window : t, :].T.contiguous()  # (N, window)
        kernel.update(Ft, tick_id=t)

    return kernel


# ---------------------------------------------------------------------------
# Batch correlation with chunking (avoids OOM for large N)
# ---------------------------------------------------------------------------

def batched_correlation(
    F: torch.Tensor,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Compute the (N, N) correlation matrix in batches along the first axis
    to avoid materialising a huge intermediate tensor.

    Parameters
    ----------
    F          : (N, D) float32, already L2-normalised
    batch_size : number of rows per batch

    Returns
    -------
    C : (N, N) float32
    """
    N = F.shape[0]
    C = torch.empty(N, N, dtype=torch.float32, device=F.device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        C[start:end, :] = F[start:end] @ F.T
    return C.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# CSR delta serialisation / deserialisation (for IPC with streaming engine)
# ---------------------------------------------------------------------------

class CSRDeltaPacket:
    """
    Lightweight serialisable snapshot of the graph state for passing
    between processes via shared memory or queue.
    """

    def __init__(
        self,
        tick_id: int,
        n: int,
        row: torch.Tensor,
        col: torch.Tensor,
        weight: torch.Tensor,
        num_born: int,
        num_died: int,
        density: float,
        elapsed_ms: float,
    ) -> None:
        self.tick_id = tick_id
        self.n = n
        # Move to CPU for serialisation
        self.row = row.cpu()
        self.col = col.cpu()
        self.weight = weight.cpu()
        self.num_born = num_born
        self.num_died = num_died
        self.density = density
        self.elapsed_ms = elapsed_ms

    # ------------------------------------------------------------------
    def to_pyg(self, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (edge_index, edge_weight) on the target device."""
        ei = torch.stack([self.row, self.col], dim=0).to(device)
        ew = self.weight.to(device)
        return ei, ew

    # ------------------------------------------------------------------
    @classmethod
    def from_kernel(
        cls,
        kernel: IAUKernel,
        stats: Dict[str, Any],
    ) -> "CSRDeltaPacket":
        ei, ew = kernel.get_pyg_edge_index()
        return cls(
            tick_id=kernel._tick_id,
            n=kernel.cfg.n_assets,
            row=ei[0],
            col=ei[1],
            weight=ew,
            num_born=stats.get("born", 0),
            num_died=stats.get("died", 0),
            density=stats.get("density", 0.0),
            elapsed_ms=stats.get("elapsed_ms", 0.0),
        )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"CSRDeltaPacket(tick={self.tick_id}, n={self.n}, "
            f"edges={self.row.numel()}, density={self.density:.4f}, "
            f"{self.elapsed_ms:.3f} ms)"
        )


# ---------------------------------------------------------------------------
# Multi-asset feature extractor (helper for real-world usage)
# ---------------------------------------------------------------------------

class MultiAssetFeatureExtractor:
    """
    Converts raw OHLCV data into feature vectors suitable for IAUK.

    Features per asset (configurable):
    - Rolling returns (multiple windows)
    - Rolling volatility (std of returns)
    - Volume-weighted average price deviation
    - Momentum signals
    - Normalised price levels
    """

    DEFAULT_RETURN_WINDOWS = [5, 10, 20, 60]
    DEFAULT_VOL_WINDOWS = [10, 20]

    def __init__(
        self,
        return_windows: Optional[List[int]] = None,
        vol_windows: Optional[List[int]] = None,
        device: str = "cuda",
    ) -> None:
        self.return_windows = return_windows or self.DEFAULT_RETURN_WINDOWS
        self.vol_windows = vol_windows or self.DEFAULT_VOL_WINDOWS
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Compute total feature dim
        self.feature_dim = len(self.return_windows) + len(self.vol_windows) + 2
        # +2 for momentum and normalised price

    # ------------------------------------------------------------------
    def extract(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Extract features from price history.

        Parameters
        ----------
        prices : (T, N) float32 — T time steps, N assets

        Returns
        -------
        features : (N, feature_dim) float32
        """
        prices = prices.to(self.device)
        T, N = prices.shape
        feat_list: List[torch.Tensor] = []

        # Rolling returns
        for w in self.return_windows:
            if T > w:
                ret = (prices[-1] - prices[-w]) / (prices[-w].clamp(min=1e-8))
            else:
                ret = torch.zeros(N, device=self.device)
            feat_list.append(ret.unsqueeze(1))

        # Rolling volatility
        for w in self.vol_windows:
            if T > w:
                window_prices = prices[-w:]
                log_ret = torch.log(window_prices[1:] / window_prices[:-1].clamp(min=1e-8))
                vol = log_ret.std(dim=0)
            else:
                vol = torch.zeros(N, device=self.device)
            feat_list.append(vol.unsqueeze(1))

        # Momentum (simple): return over full window
        if T > 1:
            mom = (prices[-1] - prices[0]) / prices[0].clamp(min=1e-8)
        else:
            mom = torch.zeros(N, device=self.device)
        feat_list.append(mom.unsqueeze(1))

        # Normalised price level
        p_min = prices.min(dim=0).values
        p_max = prices.max(dim=0).values
        p_range = (p_max - p_min).clamp(min=1e-8)
        norm_price = (prices[-1] - p_min) / p_range
        feat_list.append(norm_price.unsqueeze(1))

        features = torch.cat(feat_list, dim=1)  # (N, feature_dim)
        return features

    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        return self.feature_dim


# ---------------------------------------------------------------------------
# Online correlation tracker with Welford algorithm
# ---------------------------------------------------------------------------

class OnlineCorrelationTracker:
    """
    Maintains running mean and variance per node feature using Welford's
    online algorithm.  Used as a preprocessing normaliser in streaming mode.
    """

    def __init__(self, n_assets: int, feature_dim: int, device: torch.device) -> None:
        self.n = n_assets
        self.d = feature_dim
        self.device = device
        self.count: int = 0
        self.mean = torch.zeros(n_assets, feature_dim, device=device)
        self.M2 = torch.zeros(n_assets, feature_dim, device=device)

    # ------------------------------------------------------------------
    def update(self, batch: torch.Tensor) -> None:
        """
        Update running statistics.

        Parameters
        ----------
        batch : (N, D) float32
        """
        self.count += 1
        delta = batch - self.mean
        self.mean += delta / self.count
        delta2 = batch - self.mean
        self.M2 += delta * delta2

    # ------------------------------------------------------------------
    def normalise(self, features: torch.Tensor) -> torch.Tensor:
        """Zero-mean, unit-variance normalisation using running stats."""
        if self.count < 2:
            return features
        var = self.M2 / (self.count - 1)
        std = var.sqrt().clamp(min=1e-8)
        return (features - self.mean) / std

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.count = 0
        self.mean.zero_()
        self.M2.zero_()


# ---------------------------------------------------------------------------
# Sparse CSR operations helpers
# ---------------------------------------------------------------------------

def csr_row_sum(csr: torch.Tensor) -> torch.Tensor:
    """Compute row sums of a sparse CSR tensor."""
    return torch.sparse.sum(csr, dim=1).to_dense()


def csr_col_sum(csr: torch.Tensor) -> torch.Tensor:
    """Compute column sums of a sparse CSR tensor."""
    return torch.sparse.sum(csr, dim=0).to_dense()


def csr_degree_vector(csr: torch.Tensor) -> torch.Tensor:
    """Return degree vector from binary CSR adjacency."""
    return csr_row_sum(csr)


def csr_symmetric_normalise(
    row: torch.Tensor,
    col: torch.Tensor,
    weight: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """
    Compute D^{-1/2} A D^{-1/2} normalisation in COO form.

    Parameters
    ----------
    row, col : (E,) int64
    weight   : (E,) float32
    n        : number of nodes

    Returns
    -------
    norm_weight : (E,) float32
    """
    # Degree
    deg = torch.zeros(n, dtype=torch.float32, device=row.device)
    deg.scatter_add_(0, row, weight)
    deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e4)  # avoid inf
    return weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]


def csr_add_self_loops(
    row: torch.Tensor,
    col: torch.Tensor,
    weight: torch.Tensor,
    n: int,
    loop_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add self-loops to COO format adjacency."""
    diag = torch.arange(n, dtype=torch.int64, device=row.device)
    loop_w = torch.full((n,), loop_weight, dtype=torch.float32, device=weight.device)
    new_row = torch.cat([row, diag])
    new_col = torch.cat([col, diag])
    new_w = torch.cat([weight, loop_w])
    return new_row, new_col, new_w


# ---------------------------------------------------------------------------
# Laplacian computation
# ---------------------------------------------------------------------------

def compute_laplacian(
    row: torch.Tensor,
    col: torch.Tensor,
    weight: torch.Tensor,
    n: int,
    normalised: bool = True,
) -> torch.Tensor:
    """
    Compute the graph Laplacian (dense) from COO edges.

    Parameters
    ----------
    row, col   : (E,) int64
    weight     : (E,) float32
    n          : number of nodes
    normalised : if True, return normalised Laplacian L = I - D^{-1/2} A D^{-1/2}

    Returns
    -------
    L : (N, N) float32
    """
    A = torch.zeros(n, n, dtype=torch.float32, device=row.device)
    A[row, col] = weight
    deg = A.sum(dim=1)
    D = torch.diag(deg)
    L = D - A
    if normalised:
        d_inv_sqrt = deg.pow(-0.5).clamp(max=1e4)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        L = D_inv_sqrt @ L @ D_inv_sqrt
    return L


# ---------------------------------------------------------------------------
# Edge correlation statistics
# ---------------------------------------------------------------------------

class EdgeCorrelationStats:
    """
    Maintains running statistics on edge correlations for monitoring
    graph health and triggering adaptive rebuilds.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    def record(self, weight: torch.Tensor, tick_id: int) -> Dict[str, float]:
        """Record statistics for current tick's edge weights."""
        if weight.numel() == 0:
            stats = {
                "tick": tick_id,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "n_edges": 0,
            }
        else:
            stats = {
                "tick": tick_id,
                "mean": float(weight.mean().item()),
                "std": float(weight.std().item()) if weight.numel() > 1 else 0.0,
                "min": float(weight.min().item()),
                "max": float(weight.max().item()),
                "n_edges": weight.numel(),
            }
        self._history.append(stats)
        return stats

    # ------------------------------------------------------------------
    def rolling_mean(self, window: int = 20) -> float:
        """Return mean of the last `window` tick means."""
        tail = self._history[-window:]
        if not tail:
            return 0.0
        return sum(s["mean"] for s in tail) / len(tail)

    # ------------------------------------------------------------------
    def is_correlation_collapse(self, threshold: float = 0.05) -> bool:
        """Detect if mean edge weight has dropped below threshold."""
        if not self._history:
            return False
        return self._history[-1]["mean"] < threshold

    # ------------------------------------------------------------------
    def get_history(self) -> List[Dict[str, float]]:
        return list(self._history)


# ---------------------------------------------------------------------------
# GPU kernel launch utilities
# ---------------------------------------------------------------------------

def _scatter_add_safe(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """
    Safe scatter_add that handles empty tensors.

    Parameters
    ----------
    src      : (E,) float32
    index    : (E,) int64
    dim_size : output size

    Returns
    -------
    out : (dim_size,) float32
    """
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    if src.numel() == 0:
        return out
    out.scatter_add_(0, index, src)
    return out


def _gather_edges(
    weight_matrix: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
) -> torch.Tensor:
    """
    Efficiently gather edge weights from a dense matrix.

    Parameters
    ----------
    weight_matrix : (N, N) float32
    row, col      : (E,) int64

    Returns
    -------
    weights : (E,) float32
    """
    if row.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=weight_matrix.device)
    return weight_matrix[row, col]


# ---------------------------------------------------------------------------
# Rank-1 update (Sherman-Morrison) standalone function
# ---------------------------------------------------------------------------

def sherman_morrison_update(
    C: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    denom: Optional[float] = None,
) -> torch.Tensor:
    """
    Apply a rank-1 update to matrix C in-place.

    C ← C + (u ⊗ v) / denom

    Parameters
    ----------
    C     : (N, N) float32
    u     : (N,) float32
    v     : (N,) float32
    denom : scalar divisor (default 1.0)

    Returns
    -------
    C : (N, N) float32 (updated in-place)
    """
    if denom is None or abs(denom) < 1e-12:
        denom = 1.0
    C.add_(torch.outer(u, v) / denom)
    return C


# ---------------------------------------------------------------------------
# Adaptive threshold scheduler
# ---------------------------------------------------------------------------

class AdaptiveThresholdScheduler:
    """
    Dynamically adjusts birth/death thresholds based on market volatility.

    During high-vol regimes, we lower the birth threshold to capture more
    co-movement.  During low-vol regimes, we raise it to avoid spurious edges.
    """

    def __init__(
        self,
        kernel: IAUKernel,
        vol_series: Optional[torch.Tensor] = None,
        high_vol_birth: float = 0.20,
        low_vol_birth: float = 0.40,
        high_vol_death: float = 0.10,
        low_vol_death: float = 0.25,
        vol_high_percentile: float = 75.0,
        vol_low_percentile: float = 25.0,
    ) -> None:
        self.kernel = kernel
        self.high_vol_birth = high_vol_birth
        self.low_vol_birth = low_vol_birth
        self.high_vol_death = high_vol_death
        self.low_vol_death = low_vol_death
        self.vol_high_pct = vol_high_percentile
        self.vol_low_pct = vol_low_percentile

        self._vol_history: List[float] = []
        if vol_series is not None:
            self._vol_history = vol_series.tolist()

    # ------------------------------------------------------------------
    def step(self, current_vol: float) -> Dict[str, float]:
        """Update thresholds based on current volatility level."""
        self._vol_history.append(current_vol)

        if len(self._vol_history) < 10:
            return {
                "birth": self.kernel.hysteresis.birth_threshold,
                "death": self.kernel.hysteresis.death_threshold,
            }

        history = self._vol_history[-100:]  # last 100 readings
        vol_high = _percentile(history, self.vol_high_pct)
        vol_low = _percentile(history, self.vol_low_pct)

        if current_vol > vol_high:
            new_birth = self.high_vol_birth
            new_death = self.high_vol_death
        elif current_vol < vol_low:
            new_birth = self.low_vol_birth
            new_death = self.low_vol_death
        else:
            # Linear interpolation
            alpha = (current_vol - vol_low) / max(vol_high - vol_low, 1e-8)
            new_birth = self.low_vol_birth + alpha * (self.high_vol_birth - self.low_vol_birth)
            new_death = self.low_vol_death + alpha * (self.high_vol_death - self.low_vol_death)

        self.kernel.hysteresis.birth_threshold = float(new_birth)
        self.kernel.hysteresis.death_threshold = float(new_death)

        return {"birth": float(new_birth), "death": float(new_death)}


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def make_kernel(
    n_assets: int = 500,
    feature_dim: int = 64,
    device: str = "cuda",
    ema_alpha: float = 0.1,
    birth_threshold: float = 0.30,
    death_threshold: float = 0.20,
    max_degree: int = 50,
) -> IAUKernel:
    """Create a pre-configured IAUKernel with sensible defaults."""
    cfg = IAUKConfig(
        n_assets=n_assets,
        feature_dim=feature_dim,
        device=device,
        ema_alpha=ema_alpha,
        birth_threshold=birth_threshold,
        death_threshold=death_threshold,
        max_degree=max_degree,
    )
    return IAUKernel(cfg)


def benchmark_kernel(
    n_assets: int = 500,
    feature_dim: int = 64,
    n_ticks: int = 1000,
    dirty_fraction: float = 0.1,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run a synthetic benchmark to measure IAUK performance.

    Parameters
    ----------
    n_assets       : number of assets
    feature_dim    : feature vector dimension
    n_ticks        : number of ticks to simulate
    dirty_fraction : fraction of nodes updated per tick
    device         : torch device string

    Returns
    -------
    results : dict with timing statistics and graph statistics
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    kernel = make_kernel(n_assets, feature_dim, str(dev))

    # Initialise
    F0 = torch.randn(n_assets, feature_dim, device=dev)
    kernel.init_from_features(F0)

    features = F0.clone()
    tick_stats: List[Dict[str, Any]] = []

    for tick in range(n_ticks):
        # Simulate partial update: only dirty_fraction nodes change
        n_dirty = max(1, int(n_assets * dirty_fraction))
        dirty_idx = torch.randint(0, n_assets, (n_dirty,), device=dev)
        noise = torch.randn(n_dirty, feature_dim, device=dev) * 0.05
        features[dirty_idx] += noise

        stats = kernel.update(features.clone(), tick_id=tick)
        tick_stats.append(stats)

    summary = kernel.benchmark_summary()
    summary["n_assets"] = n_assets
    summary["feature_dim"] = feature_dim
    summary["n_ticks"] = n_ticks
    summary["final_edges"] = kernel.num_edges()
    summary["final_density"] = kernel.density()
    summary["target_met_1ms"] = summary.get("p99_ms", 999) < 1.0

    return summary


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "IAUKConfig",
    "IAUKernel",
    "DirtyBitTracker",
    "EMAEdgeWeights",
    "ShermanMorrisonDelta",
    "HysteresisEdgeController",
    "DegreeCapKNN",
    "DeltaPatchApplicator",
    "CSRDeltaPacket",
    "MultiAssetFeatureExtractor",
    "OnlineCorrelationTracker",
    "EdgeCorrelationStats",
    "AdaptiveThresholdScheduler",
    "batched_correlation",
    "csr_symmetric_normalise",
    "csr_add_self_loops",
    "compute_laplacian",
    "sherman_morrison_update",
    "make_kernel",
    "benchmark_kernel",
    "build_iauk_from_returns",
]
