"""
temporal_edge_buffer.py — Temporal Edge Buffer (TEB)
=====================================================
Part of the Omni-Graph incremental graph construction suite.

Design goals
------------
* Sliding window buffer storing recent edge weights in contiguous memory
  (torch.Tensor, no pointer chasing) so GPU can access entire history in
  one contiguous read.
* Fixed-size circular buffer: configurable window (e.g. 100 ticks per edge).
* Pre-allocated GPU memory: no dynamic allocation in the hot path.
* Vectorised edge weight lookup: batch query all edges at time T in a
  single GPU kernel call.
* Edge weight statistics: mean/variance/min/max/trend over window without
  a full scan via incremental Welford tracking.
* Temporal attention weights: learned importance weighting over window
  positions (soft attention per edge).
* Integration with PyG TemporalData format.

Public API
----------
    teb = TemporalEdgeBuffer(max_edges=50000, window=100, device="cuda")
    teb.push(edge_index, edge_weight, tick_id)
    weights_t = teb.query_at(tick_id)
    stats = teb.edge_stats(edge_index)
    td = teb.to_temporal_data(node_features)
"""

from __future__ import annotations

import math
import time
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TEBConfig:
    """All hyper-parameters for the Temporal Edge Buffer."""

    # Maximum number of distinct edges tracked
    max_edges: int = 50_000

    # Circular window length (ticks per edge)
    window: int = 100

    # Device
    device: str = "cuda"

    # Attention model hidden dim
    attn_hidden_dim: int = 32

    # Number of attention heads
    n_attn_heads: int = 4

    # Whether to train the temporal attention module
    learn_attention: bool = True

    # EMA for incremental statistics (for trend estimation)
    stats_ema_alpha: float = 0.05

    # Fill value for empty buffer slots
    fill_value: float = 0.0

    # If True, store absolute edge weights (unsigned)
    store_abs: bool = True

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be > 0")
        if self.max_edges <= 0:
            raise ValueError("max_edges must be > 0")


# ---------------------------------------------------------------------------
# Named tuple for query results
# ---------------------------------------------------------------------------

class EdgeQueryResult(NamedTuple):
    """Result of a temporal edge buffer query."""
    edge_index: torch.Tensor      # (2, E) int64
    weight_history: torch.Tensor  # (E, W) float32 — W = window
    current_weight: torch.Tensor  # (E,) float32
    tick_ids: torch.Tensor        # (W,) int64


# ---------------------------------------------------------------------------
# Incremental Welford statistics per edge
# ---------------------------------------------------------------------------

class IncrementalEdgeStats:
    """
    Maintains per-edge running statistics using Welford's online algorithm.
    All state is stored in pre-allocated GPU tensors for speed.

    Tracks: mean, M2 (for variance), min, max, count, EMA (for trend).
    """

    def __init__(self, max_edges: int, ema_alpha: float, device: torch.device) -> None:
        self.max_edges = max_edges
        self.ema_alpha = ema_alpha
        self.device = device

        self.count = torch.zeros(max_edges, dtype=torch.int64, device=device)
        self.mean = torch.zeros(max_edges, dtype=torch.float32, device=device)
        self.M2 = torch.zeros(max_edges, dtype=torch.float32, device=device)
        self.min_val = torch.full((max_edges,), float("inf"), dtype=torch.float32, device=device)
        self.max_val = torch.full((max_edges,), float("-inf"), dtype=torch.float32, device=device)
        self.ema = torch.zeros(max_edges, dtype=torch.float32, device=device)
        self.prev_ema = torch.zeros(max_edges, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    def update(self, edge_ids: torch.Tensor, weights: torch.Tensor) -> None:
        """
        Update statistics for given edge IDs.

        Parameters
        ----------
        edge_ids : (E,) int64 — indices into [0, max_edges)
        weights  : (E,) float32
        """
        if edge_ids.numel() == 0:
            return

        self.count[edge_ids] += 1
        n = self.count[edge_ids].float()

        delta = weights - self.mean[edge_ids]
        self.mean[edge_ids] += delta / n
        delta2 = weights - self.mean[edge_ids]
        self.M2[edge_ids] += delta * delta2

        # Min/max
        self.min_val[edge_ids] = torch.minimum(self.min_val[edge_ids], weights)
        self.max_val[edge_ids] = torch.maximum(self.max_val[edge_ids], weights)

        # EMA for trend
        self.prev_ema[edge_ids] = self.ema[edge_ids].clone()
        self.ema[edge_ids] = (
            self.ema_alpha * weights + (1 - self.ema_alpha) * self.ema[edge_ids]
        )

    # ------------------------------------------------------------------
    def variance(self, edge_ids: torch.Tensor) -> torch.Tensor:
        """Return variance for given edge IDs."""
        n = self.count[edge_ids].float().clamp(min=2)
        return self.M2[edge_ids] / (n - 1)

    # ------------------------------------------------------------------
    def std(self, edge_ids: torch.Tensor) -> torch.Tensor:
        return self.variance(edge_ids).clamp(min=0).sqrt()

    # ------------------------------------------------------------------
    def trend(self, edge_ids: torch.Tensor) -> torch.Tensor:
        """Return EMA trend direction (+/-) for given edges."""
        return self.ema[edge_ids] - self.prev_ema[edge_ids]

    # ------------------------------------------------------------------
    def get_stats(self, edge_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return all stats as a dict for given edge IDs."""
        return {
            "mean": self.mean[edge_ids],
            "std": self.std(edge_ids),
            "min": self.min_val[edge_ids],
            "max": self.max_val[edge_ids],
            "trend": self.trend(edge_ids),
            "count": self.count[edge_ids],
        }

    # ------------------------------------------------------------------
    def reset(self, edge_ids: Optional[torch.Tensor] = None) -> None:
        """Reset stats (optionally for specific edges)."""
        if edge_ids is None:
            self.count.zero_()
            self.mean.zero_()
            self.M2.zero_()
            self.min_val.fill_(float("inf"))
            self.max_val.fill_(float("-inf"))
            self.ema.zero_()
            self.prev_ema.zero_()
        else:
            self.count[edge_ids] = 0
            self.mean[edge_ids] = 0.0
            self.M2[edge_ids] = 0.0
            self.min_val[edge_ids] = float("inf")
            self.max_val[edge_ids] = float("-inf")
            self.ema[edge_ids] = 0.0
            self.prev_ema[edge_ids] = 0.0


# ---------------------------------------------------------------------------
# Edge index registry
# ---------------------------------------------------------------------------

class EdgeRegistry:
    """
    Maps (src, dst) edge pairs to contiguous integer IDs in [0, max_edges).
    Uses a hash-map approach backed by a dense collision table on CPU
    (lookups are rare; the hot path uses pre-cached IDs).
    """

    def __init__(self, n_nodes: int, max_edges: int, device: torch.device) -> None:
        self.n_nodes = n_nodes
        self.max_edges = max_edges
        self.device = device

        # Compact hash: key = src * N + dst → edge_id
        self._map: Dict[int, int] = {}
        self._next_id: int = 0

        # Reverse map: edge_id → (src, dst)
        self._src = torch.zeros(max_edges, dtype=torch.int64, device=device)
        self._dst = torch.zeros(max_edges, dtype=torch.int64, device=device)

        # Active mask
        self._active = torch.zeros(max_edges, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------
    def get_or_create(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """
        Get edge IDs for (src, dst) pairs, creating new entries as needed.

        Parameters
        ----------
        src, dst : (E,) int64 on CPU or GPU

        Returns
        -------
        ids : (E,) int64
        """
        src_cpu = src.cpu().tolist()
        dst_cpu = dst.cpu().tolist()
        ids: List[int] = []
        N = self.n_nodes

        for s, d in zip(src_cpu, dst_cpu):
            key = int(s) * N + int(d)
            if key not in self._map:
                if self._next_id >= self.max_edges:
                    # Eviction: reuse oldest (simple FIFO — for production
                    # a proper LRU would be used)
                    warnings.warn(
                        f"EdgeRegistry full ({self.max_edges} edges); "
                        "evicting oldest entry.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._next_id = 0
                eid = self._next_id
                self._map[key] = eid
                self._src[eid] = int(s)
                self._dst[eid] = int(d)
                self._active[eid] = True
                self._next_id += 1
            ids.append(self._map[key])

        return torch.tensor(ids, dtype=torch.int64, device=self.device)

    # ------------------------------------------------------------------
    def lookup(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """
        Look up edge IDs; return -1 for unknown edges.
        """
        src_cpu = src.cpu().tolist()
        dst_cpu = dst.cpu().tolist()
        N = self.n_nodes
        ids = [
            self._map.get(int(s) * N + int(d), -1)
            for s, d in zip(src_cpu, dst_cpu)
        ]
        return torch.tensor(ids, dtype=torch.int64, device=self.device)

    # ------------------------------------------------------------------
    def deactivate(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Mark edges as inactive (dead)."""
        ids = self.lookup(src, dst)
        valid = ids >= 0
        self._active[ids[valid]] = False

    # ------------------------------------------------------------------
    def num_active(self) -> int:
        return int(self._active.sum().item())

    # ------------------------------------------------------------------
    def get_active_edge_index(self) -> torch.Tensor:
        """Return (2, E_active) int64 edge_index for all active edges."""
        active_ids = self._active.nonzero(as_tuple=True)[0]
        if active_ids.numel() == 0:
            return torch.zeros(2, 0, dtype=torch.int64, device=self.device)
        src = self._src[active_ids]
        dst = self._dst[active_ids]
        return torch.stack([src, dst], dim=0)


# ---------------------------------------------------------------------------
# Circular buffer core
# ---------------------------------------------------------------------------

class _CircularBuffer:
    """
    Fixed-size circular buffer for edge weight history.

    Layout: (max_edges, window) float32 — contiguous in memory.
    Write pointer per edge is tracked in a (max_edges,) int64 tensor.
    """

    def __init__(
        self,
        max_edges: int,
        window: int,
        fill_value: float,
        device: torch.device,
    ) -> None:
        self.max_edges = max_edges
        self.window = window
        self.device = device

        # Main buffer: (E, W)
        self.data = torch.full(
            (max_edges, window), fill_value, dtype=torch.float32, device=device
        )

        # Write pointer per edge: next position to overwrite
        self.write_ptr = torch.zeros(max_edges, dtype=torch.int64, device=device)

        # How many ticks have been written per edge
        self.fill_count = torch.zeros(max_edges, dtype=torch.int64, device=device)

        # Tick ID of each slot: (E, W) int64
        self.tick_ids = torch.full(
            (max_edges, window), -1, dtype=torch.int64, device=device
        )

    # ------------------------------------------------------------------
    def write(
        self,
        edge_ids: torch.Tensor,   # (E,) int64
        weights: torch.Tensor,    # (E,) float32
        tick_id: int,
    ) -> None:
        """Write new weights at current write position (circular)."""
        if edge_ids.numel() == 0:
            return

        ptrs = self.write_ptr[edge_ids]   # (E,)

        # Scatter write using advanced indexing
        self.data[edge_ids, ptrs] = weights
        self.tick_ids[edge_ids, ptrs] = tick_id

        # Advance write pointer (mod window)
        self.write_ptr[edge_ids] = (ptrs + 1) % self.window

        # Update fill count (capped at window)
        self.fill_count[edge_ids] = (self.fill_count[edge_ids] + 1).clamp(max=self.window)

    # ------------------------------------------------------------------
    def read_latest(self, edge_ids: torch.Tensor) -> torch.Tensor:
        """Read the most recent weight for each edge."""
        ptrs = (self.write_ptr[edge_ids] - 1) % self.window
        return self.data[edge_ids, ptrs]

    # ------------------------------------------------------------------
    def read_window(self, edge_ids: torch.Tensor) -> torch.Tensor:
        """
        Read the full window for each edge, ordered oldest→newest.

        Returns
        -------
        history : (|edge_ids|, W) float32
        """
        # Each edge may have a different write_ptr so we can't simply slice.
        # We use an index rotation approach.
        E = edge_ids.numel()
        W = self.window

        raw = self.data[edge_ids]  # (E, W) — current circular layout

        # Create rotation indices to put oldest first
        ptrs = self.write_ptr[edge_ids]  # (E,) — next write position = oldest
        idx = (torch.arange(W, device=self.device).unsqueeze(0) + ptrs.unsqueeze(1)) % W
        # idx[e, t] = physical position of logical time t for edge e
        ordered = torch.gather(raw, 1, idx)  # (E, W)
        return ordered

    # ------------------------------------------------------------------
    def read_at_offset(self, edge_ids: torch.Tensor, offset: int) -> torch.Tensor:
        """
        Read weight at relative offset from latest (0 = latest, -1 = one tick ago).
        """
        offset = offset % self.window
        ptrs = (self.write_ptr[edge_ids] - 1 - offset) % self.window
        return self.data[edge_ids, ptrs]

    # ------------------------------------------------------------------
    def reset_edges(self, edge_ids: torch.Tensor, fill_value: float = 0.0) -> None:
        """Reset buffer for specific edges."""
        self.data[edge_ids] = fill_value
        self.write_ptr[edge_ids] = 0
        self.fill_count[edge_ids] = 0
        self.tick_ids[edge_ids] = -1


# ---------------------------------------------------------------------------
# Temporal attention module
# ---------------------------------------------------------------------------

class TemporalAttentionWeights(nn.Module):
    """
    Learned soft-attention weights over window positions.

    Produces a (W,) weighting vector for each edge based on its weight
    history and optional edge features.

    Architecture:
        Linear(W → hidden) → ReLU → Linear(hidden → W) → Softmax
    """

    def __init__(
        self,
        window: int,
        hidden_dim: int = 32,
        n_heads: int = 4,
        edge_feat_dim: int = 0,
    ) -> None:
        super().__init__()
        self.window = window
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        input_dim = window + edge_feat_dim

        self.proj = nn.Linear(input_dim, hidden_dim * n_heads)
        self.attn_head_proj = nn.Linear(hidden_dim * n_heads, n_heads * window)
        self.head_combine = nn.Linear(n_heads, 1, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_head_proj.weight)
        nn.init.ones_(self.head_combine.weight)

    # ------------------------------------------------------------------
    def forward(
        self,
        weight_history: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        weight_history : (E, W) float32
        edge_features  : (E, F) float32, optional

        Returns
        -------
        attn_weights : (E, W) float32 — sum to 1 across W
        """
        E, W = weight_history.shape

        if edge_features is not None:
            x = torch.cat([weight_history, edge_features], dim=-1)
        else:
            x = weight_history

        h = F.relu(self.proj(x))                          # (E, H*heads)
        raw = self.attn_head_proj(h)                      # (E, heads*W)
        raw = raw.view(E, self.n_heads, W)                # (E, heads, W)
        attn = F.softmax(raw, dim=-1)                     # (E, heads, W)
        # Combine heads
        attn = attn.permute(0, 2, 1)                      # (E, W, heads)
        attn = self.head_combine(attn).squeeze(-1)        # (E, W)
        attn = F.softmax(attn, dim=-1)                    # (E, W) normalised
        return attn

    # ------------------------------------------------------------------
    def weighted_sum(
        self,
        weight_history: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention-weighted sum of weight history.

        Returns
        -------
        (E,) float32 — temporally weighted edge weight
        """
        attn = self.forward(weight_history, edge_features)  # (E, W)
        return (attn * weight_history).sum(dim=-1)           # (E,)


# ---------------------------------------------------------------------------
# Main Temporal Edge Buffer
# ---------------------------------------------------------------------------

class TemporalEdgeBuffer:
    """
    Temporal Edge Buffer: fixed-size circular GPU buffer of edge weight
    history.

    This is the main entry point for users.

    Usage
    -----
        teb = TemporalEdgeBuffer(n_nodes=500, max_edges=50000, window=100)
        for tick in range(T):
            teb.push(edge_index, edge_weight, tick_id=tick)
        hist = teb.get_window(edge_index)           # (E, 100)
        stats = teb.edge_stats(edge_index)          # dict of (E,) tensors
        td = teb.to_temporal_data(node_feats)       # PyG TemporalData
    """

    def __init__(
        self,
        n_nodes: int,
        cfg: Optional[TEBConfig] = None,
        **kwargs: Any,
    ) -> None:
        if cfg is None:
            cfg = TEBConfig(**kwargs)
        self.cfg = cfg
        self.n_nodes = n_nodes
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )

        # Core buffer
        self._buf = _CircularBuffer(
            max_edges=cfg.max_edges,
            window=cfg.window,
            fill_value=cfg.fill_value,
            device=self.device,
        )

        # Edge registry
        self._registry = EdgeRegistry(n_nodes, cfg.max_edges, self.device)

        # Incremental statistics
        self._stats = IncrementalEdgeStats(
            cfg.max_edges, cfg.stats_ema_alpha, self.device
        )

        # Temporal attention module (on same device)
        if cfg.learn_attention:
            self._attn = TemporalAttentionWeights(
                window=cfg.window,
                hidden_dim=cfg.attn_hidden_dim,
                n_heads=cfg.n_attn_heads,
            ).to(self.device)
        else:
            self._attn = None

        # Metadata
        self._current_tick: int = -1
        self._push_count: int = 0
        self._tick_to_push_time: Dict[int, float] = {}

    # ------------------------------------------------------------------
    def push(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        tick_id: int,
    ) -> int:
        """
        Push new edge weights into the buffer.

        Parameters
        ----------
        edge_index  : (2, E) int64
        edge_weight : (E,) float32
        tick_id     : integer tick identifier

        Returns
        -------
        n_edges_written : int
        """
        t0 = time.perf_counter()

        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device, dtype=torch.float32)

        if self.cfg.store_abs:
            edge_weight = edge_weight.abs()

        E = edge_index.shape[1]
        if E == 0:
            self._current_tick = tick_id
            return 0

        src, dst = edge_index[0], edge_index[1]

        # Resolve edge IDs (get or create)
        edge_ids = self._registry.get_or_create(src, dst)

        # Write to circular buffer
        self._buf.write(edge_ids, edge_weight, tick_id)

        # Update incremental statistics
        self._stats.update(edge_ids, edge_weight)

        self._current_tick = tick_id
        self._push_count += 1
        self._tick_to_push_time[tick_id] = (time.perf_counter() - t0) * 1000.0

        return E

    # ------------------------------------------------------------------
    def get_latest(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the most recent weight for each edge in edge_index.

        Returns
        -------
        (E,) float32
        """
        edge_index = edge_index.to(self.device)
        src, dst = edge_index[0], edge_index[1]
        ids = self._registry.lookup(src, dst)
        valid = ids >= 0
        result = torch.zeros(edge_index.shape[1], dtype=torch.float32, device=self.device)
        if valid.any():
            result[valid] = self._buf.read_latest(ids[valid])
        return result

    # ------------------------------------------------------------------
    def get_window(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the full weight history for each edge.

        Returns
        -------
        history : (E, W) float32 — ordered oldest→newest
        """
        edge_index = edge_index.to(self.device)
        src, dst = edge_index[0], edge_index[1]
        ids = self._registry.lookup(src, dst)
        E = edge_index.shape[1]
        W = self.cfg.window
        result = torch.full((E, W), self.cfg.fill_value, dtype=torch.float32, device=self.device)
        valid = ids >= 0
        if valid.any():
            result[valid] = self._buf.read_window(ids[valid])
        return result

    # ------------------------------------------------------------------
    def query_at(self, tick_id: int, edge_index: Optional[torch.Tensor] = None) -> EdgeQueryResult:
        """
        Query buffer state at a specific tick.

        Parameters
        ----------
        tick_id    : the tick to query
        edge_index : optional (2, E) — if None, use all active edges

        Returns
        -------
        EdgeQueryResult
        """
        if edge_index is None:
            edge_index = self._registry.get_active_edge_index()

        if edge_index.shape[1] == 0:
            empty_w = torch.zeros(0, self.cfg.window, device=self.device)
            empty_cw = torch.zeros(0, device=self.device)
            tick_arr = torch.arange(tick_id - self.cfg.window, tick_id, device=self.device)
            return EdgeQueryResult(edge_index, empty_w, empty_cw, tick_arr)

        edge_index = edge_index.to(self.device)
        src, dst = edge_index[0], edge_index[1]
        ids = self._registry.lookup(src, dst)
        valid = ids >= 0

        E = edge_index.shape[1]
        W = self.cfg.window
        history = torch.full((E, W), self.cfg.fill_value, dtype=torch.float32, device=self.device)
        current = torch.zeros(E, dtype=torch.float32, device=self.device)

        if valid.any():
            history[valid] = self._buf.read_window(ids[valid])
            current[valid] = self._buf.read_latest(ids[valid])

        tick_arr = torch.arange(
            tick_id - W, tick_id, dtype=torch.int64, device=self.device
        )

        return EdgeQueryResult(edge_index, history, current, tick_arr)

    # ------------------------------------------------------------------
    def edge_stats(self, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return per-edge statistics without a full window scan.

        Returns dict with keys: mean, std, min, max, trend, count.
        """
        edge_index = edge_index.to(self.device)
        src, dst = edge_index[0], edge_index[1]
        ids = self._registry.lookup(src, dst)
        valid = ids >= 0

        E = edge_index.shape[1]
        result: Dict[str, torch.Tensor] = {
            "mean":  torch.zeros(E, device=self.device),
            "std":   torch.zeros(E, device=self.device),
            "min":   torch.zeros(E, device=self.device),
            "max":   torch.zeros(E, device=self.device),
            "trend": torch.zeros(E, device=self.device),
            "count": torch.zeros(E, dtype=torch.int64, device=self.device),
        }

        if valid.any():
            s = self._stats.get_stats(ids[valid])
            for k in result:
                result[k][valid] = s[k]

        return result

    # ------------------------------------------------------------------
    def edge_stats_from_window(self, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute statistics by scanning the full window (more accurate
        but slower than edge_stats()).  Useful for validation.
        """
        history = self.get_window(edge_index)      # (E, W)
        fill = self.cfg.fill_value
        mask = history != fill                     # (E, W) bool

        mean = history.sum(dim=1) / mask.sum(dim=1).float().clamp(min=1)
        var_num = ((history - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1)
        var = var_num / mask.sum(dim=1).float().clamp(min=2)
        std = var.clamp(min=0).sqrt()

        min_val = history.masked_fill(~mask, float("inf")).min(dim=1).values
        max_val = history.masked_fill(~mask, float("-inf")).max(dim=1).values

        # Trend: difference between second-half and first-half mean
        W = self.cfg.window
        half = W // 2
        h1 = history[:, :half]
        h2 = history[:, half:]
        m1 = h1.masked_fill(h1 == fill, 0).mean(dim=1)
        m2 = h2.masked_fill(h2 == fill, 0).mean(dim=1)
        trend = m2 - m1

        return {
            "mean": mean,
            "std": std,
            "min": min_val.clamp(min=0),
            "max": max_val.clamp(min=0),
            "trend": trend,
            "count": mask.sum(dim=1),
        }

    # ------------------------------------------------------------------
    def attention_weights(
        self,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute temporal attention weights over window positions.

        Requires learn_attention=True in config.

        Returns
        -------
        (E, W) float32 — attention weights summing to 1 along W axis
        """
        if self._attn is None:
            # Uniform attention as fallback
            E = edge_index.shape[1]
            W = self.cfg.window
            return torch.full((E, W), 1.0 / W, device=self.device)

        history = self.get_window(edge_index)       # (E, W)
        with torch.no_grad():
            attn = self._attn(history, edge_features)
        return attn

    # ------------------------------------------------------------------
    def attended_weights(
        self,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute temporally attended edge weights (scalar per edge).

        Returns
        -------
        (E,) float32
        """
        history = self.get_window(edge_index)       # (E, W)
        if self._attn is None:
            return history.mean(dim=1)
        with torch.no_grad():
            return self._attn.weighted_sum(history, edge_features)

    # ------------------------------------------------------------------
    def to_temporal_data(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        use_attended_weights: bool = False,
    ) -> Any:
        """
        Package buffer state as a PyG TemporalData-like object.

        Parameters
        ----------
        node_features         : (N, D) float32
        edge_index            : optional (2, E) — if None, use all active
        use_attended_weights  : if True, use temporal attention for edge_attr

        Returns
        -------
        TEBTemporalData (dict-like container compatible with PyG)
        """
        if edge_index is None:
            edge_index = self._registry.get_active_edge_index()

        if edge_index.shape[1] == 0:
            return TEBTemporalData(
                x=node_features.to(self.device),
                edge_index=edge_index,
                edge_attr=torch.zeros(0, self.cfg.window, device=self.device),
                edge_weight=torch.zeros(0, device=self.device),
                t=torch.tensor([self._current_tick], dtype=torch.int64, device=self.device),
            )

        edge_index = edge_index.to(self.device)
        history = self.get_window(edge_index)   # (E, W)

        if use_attended_weights:
            edge_weight = self.attended_weights(edge_index)
        else:
            edge_weight = self.get_latest(edge_index)

        return TEBTemporalData(
            x=node_features.to(self.device),
            edge_index=edge_index,
            edge_attr=history,
            edge_weight=edge_weight,
            t=torch.tensor([self._current_tick], dtype=torch.int64, device=self.device),
        )

    # ------------------------------------------------------------------
    def num_active_edges(self) -> int:
        return self._registry.num_active()

    # ------------------------------------------------------------------
    def remove_edges(self, edge_index: torch.Tensor) -> None:
        """
        Mark edges as dead (deactivate in registry, reset buffer).
        """
        edge_index = edge_index.to(self.device)
        src, dst = edge_index[0], edge_index[1]
        ids = self._registry.lookup(src, dst)
        valid = ids >= 0
        if valid.any():
            self._buf.reset_edges(ids[valid], fill_value=self.cfg.fill_value)
            self._stats.reset(ids[valid])
            self._registry.deactivate(src[valid], dst[valid])

    # ------------------------------------------------------------------
    def get_push_latencies(self) -> List[float]:
        """Return per-tick push latencies in ms."""
        return list(self._tick_to_push_time.values())

    # ------------------------------------------------------------------
    def memory_usage_mb(self) -> float:
        """Estimate GPU memory used by the buffer in megabytes."""
        buf_bytes = self._buf.data.numel() * 4  # float32
        tick_bytes = self._buf.tick_ids.numel() * 8  # int64
        stats_bytes = (
            self._stats.mean.numel() * 4 * 5  # mean, M2, min, max, ema
            + self._stats.count.numel() * 8
        )
        total_bytes = buf_bytes + tick_bytes + stats_bytes
        return total_bytes / (1024 ** 2)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"TemporalEdgeBuffer("
            f"n_nodes={self.n_nodes}, "
            f"max_edges={self.cfg.max_edges}, "
            f"window={self.cfg.window}, "
            f"active_edges={self.num_active_edges()}, "
            f"mem={self.memory_usage_mb():.1f}MB, "
            f"device={self.device})"
        )


# ---------------------------------------------------------------------------
# TEBTemporalData — PyG-compatible container
# ---------------------------------------------------------------------------

class TEBTemporalData:
    """
    Lightweight container compatible with PyG's TemporalData API.
    Stores graph snapshot at a single tick.
    """

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_weight: torch.Tensor,
        t: torch.Tensor,
    ) -> None:
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_weight = edge_weight
        self.t = t

    # ------------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    # ------------------------------------------------------------------
    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    # ------------------------------------------------------------------
    def to(self, device: torch.device) -> "TEBTemporalData":
        """Move all tensors to device."""
        return TEBTemporalData(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            edge_weight=self.edge_weight.to(device),
            t=self.t.to(device),
        )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"TEBTemporalData(nodes={self.num_nodes}, "
            f"edges={self.num_edges}, "
            f"window={self.edge_attr.shape[-1] if self.edge_attr.dim() > 1 else 1}, "
            f"t={self.t.item()})"
        )


# ---------------------------------------------------------------------------
# Sliding-window correlation builder
# ---------------------------------------------------------------------------

class SlidingWindowCorrelationBuilder:
    """
    Builds correlation-based edge weights from a sliding window of returns,
    outputting them in a form ready for TemporalEdgeBuffer.push().

    Maintains an online covariance matrix updated each tick.
    """

    def __init__(
        self,
        n_assets: int,
        window: int = 60,
        min_periods: int = 20,
        device: str = "cuda",
    ) -> None:
        self.n_assets = n_assets
        self.window = window
        self.min_periods = min_periods
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Ring buffer of returns: (window, N)
        self._returns_buf = torch.zeros(
            window, n_assets, dtype=torch.float32, device=self.device
        )
        self._write_ptr = 0
        self._fill_count = 0

    # ------------------------------------------------------------------
    def push_returns(self, returns: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Push new returns (N,) and recompute correlation if enough data.

        Returns
        -------
        corr_matrix : (N, N) float32 or None if insufficient data
        """
        returns = returns.to(self.device, dtype=torch.float32)
        self._returns_buf[self._write_ptr] = returns
        self._write_ptr = (self._write_ptr + 1) % self.window
        self._fill_count = min(self._fill_count + 1, self.window)

        if self._fill_count < self.min_periods:
            return None

        # Extract valid window
        n_valid = self._fill_count
        if n_valid == self.window:
            R = self._returns_buf
        else:
            # Build ordered slice
            idxs = [(self._write_ptr + i) % self.window for i in range(n_valid)]
            idx_t = torch.tensor(idxs, dtype=torch.int64, device=self.device)
            R = self._returns_buf[idx_t]

        # Compute correlation: (N, N)
        R = R - R.mean(dim=0, keepdim=True)
        std = R.std(dim=0, keepdim=True).clamp(min=1e-8)
        R_norm = R / std
        corr = (R_norm.T @ R_norm) / n_valid
        corr = corr.clamp(-1.0, 1.0)
        return corr

    # ------------------------------------------------------------------
    def build_edge_tensors(
        self,
        corr_matrix: torch.Tensor,
        threshold: float = 0.20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert correlation matrix to edge_index and edge_weight.

        Parameters
        ----------
        corr_matrix : (N, N) float32
        threshold   : minimum |corr| to include edge

        Returns
        -------
        edge_index  : (2, E) int64
        edge_weight : (E,) float32
        """
        abs_corr = corr_matrix.abs()
        # Upper triangle only
        triu = torch.triu(abs_corr, diagonal=1)
        mask = triu > threshold
        idx = mask.nonzero(as_tuple=False)  # (E, 2)
        if idx.numel() == 0:
            dev = corr_matrix.device
            return torch.zeros(2, 0, dtype=torch.int64, device=dev), \
                   torch.zeros(0, dtype=torch.float32, device=dev)
        row, col = idx[:, 0], idx[:, 1]
        w = abs_corr[row, col]
        # Symmetric
        row_sym = torch.cat([row, col])
        col_sym = torch.cat([col, row])
        w_sym = torch.cat([w, w])
        edge_index = torch.stack([row_sym, col_sym], dim=0)
        return edge_index, w_sym


# ---------------------------------------------------------------------------
# Vectorised batch lookups
# ---------------------------------------------------------------------------

class BatchEdgeLookup:
    """
    Utility for performing batch lookups across multiple TEBs
    (e.g. different edge types in a heterogeneous graph).
    """

    def __init__(self, buffers: Dict[str, TemporalEdgeBuffer]) -> None:
        self.buffers = buffers

    # ------------------------------------------------------------------
    def batch_query(
        self,
        edge_dict: Dict[str, torch.Tensor],
        tick_id: int,
    ) -> Dict[str, EdgeQueryResult]:
        """
        Query all buffers simultaneously.

        Parameters
        ----------
        edge_dict : {edge_type: (2, E) int64}
        tick_id   : current tick

        Returns
        -------
        {edge_type: EdgeQueryResult}
        """
        results: Dict[str, EdgeQueryResult] = {}
        for etype, ei in edge_dict.items():
            if etype in self.buffers:
                results[etype] = self.buffers[etype].query_at(tick_id, ei)
        return results

    # ------------------------------------------------------------------
    def push_all(
        self,
        edge_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        tick_id: int,
    ) -> None:
        """Push updates to all buffers."""
        for etype, (ei, ew) in edge_dict.items():
            if etype in self.buffers:
                self.buffers[etype].push(ei, ew, tick_id)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_teb(
    n_nodes: int = 500,
    max_edges: int = 50_000,
    window: int = 100,
    n_ticks: int = 1000,
    edges_per_tick: int = 5000,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Benchmark the TemporalEdgeBuffer push and query latencies.

    Returns
    -------
    dict with mean/max/p99 push and query latencies in ms.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    cfg = TEBConfig(max_edges=max_edges, window=window, device=str(dev))
    teb = TemporalEdgeBuffer(n_nodes=n_nodes, cfg=cfg)

    # Pre-generate random edges
    src = torch.randint(0, n_nodes, (edges_per_tick,), device=dev)
    dst = torch.randint(0, n_nodes, (edges_per_tick,), device=dev)
    ei = torch.stack([src, dst], dim=0)

    push_times: List[float] = []
    query_times: List[float] = []

    for tick in range(n_ticks):
        w = torch.rand(edges_per_tick, device=dev)

        # Push
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        teb.push(ei, w, tick_id=tick)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        push_times.append((time.perf_counter() - t0) * 1000.0)

        # Query
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = teb.get_window(ei)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        query_times.append((time.perf_counter() - t0) * 1000.0)

    def stats(lst: List[float]) -> Dict[str, float]:
        s = sorted(lst)
        n = len(s)
        return {
            "mean": sum(s) / n,
            "max": s[-1],
            "p99": s[int(0.99 * n)],
            "p95": s[int(0.95 * n)],
        }

    return {
        "push": stats(push_times),
        "query": stats(query_times),
        "n_ticks": n_ticks,
        "edges_per_tick": edges_per_tick,
        "memory_mb": teb.memory_usage_mb(),
    }


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "TEBConfig",
    "TemporalEdgeBuffer",
    "TEBTemporalData",
    "EdgeQueryResult",
    "IncrementalEdgeStats",
    "EdgeRegistry",
    "TemporalAttentionWeights",
    "SlidingWindowCorrelationBuilder",
    "BatchEdgeLookup",
    "benchmark_teb",
]
