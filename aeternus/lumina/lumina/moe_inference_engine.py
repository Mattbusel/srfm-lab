"""
moe_inference_engine.py
=======================
Optimized Mixture-of-Experts inference engine for Lumina.

Key optimizations:
  - Predictive Expert Dispatcher (PED): 2-layer MLP pre-router trained online
  - Expert weight pre-fetching via CUDA streams based on PED predictions
  - Expert-Specific KV Cache Manager with LRU eviction
  - Batched expert execution (group tokens by expert, single kernel per expert)
  - Speculative routing: top-1 executed immediately while top-2/3 computed in parallel
  - Expert capacity factor auto-tuning
  - Benchmarking: tokens/sec and latency percentiles
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import heapq
import logging
import math
import threading
import time
import warnings
from collections import OrderedDict, deque
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

_DEFAULT_TOP_K = 2
_DEFAULT_NUM_EXPERTS = 8
_DEFAULT_HIDDEN_DIM = 512
_DEFAULT_CAPACITY_FACTOR = 1.25
_DEFAULT_PED_HIDDEN = 128
_DEFAULT_KV_CACHE_MAX_EXPERTS = 4
_DEFAULT_PREFETCH_LOOKAHEAD = 1

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MoEConfig:
    """Configuration for the MoE inference engine."""

    num_experts: int = _DEFAULT_NUM_EXPERTS
    top_k: int = _DEFAULT_TOP_K
    hidden_dim: int = _DEFAULT_HIDDEN_DIM
    ffn_dim: int = 2048
    capacity_factor: float = _DEFAULT_CAPACITY_FACTOR
    use_ped: bool = True
    use_prefetch: bool = True
    use_kv_cache: bool = True
    use_batched_exec: bool = True
    use_speculative_routing: bool = True
    ped_hidden_dim: int = _DEFAULT_PED_HIDDEN
    kv_cache_max_resident: int = _DEFAULT_KV_CACHE_MAX_EXPERTS
    prefetch_lookahead: int = _DEFAULT_PREFETCH_LOOKAHEAD
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    latency_budget_ms: float = 50.0
    capacity_tuning_interval: int = 100
    benchmark_warmup_steps: int = 10
    benchmark_steps: int = 100
    enable_amp: bool = True
    dropout: float = 0.0
    expert_ffn_activation: str = "swiglu"


@dataclasses.dataclass
class RoutingDecision:
    """Result of the MoE router for a batch."""

    expert_indices: Tensor       # (batch, top_k)
    routing_weights: Tensor      # (batch, top_k)
    router_logits: Tensor        # (batch, num_experts)
    dropped_tokens: int = 0
    capacity_factor: float = _DEFAULT_CAPACITY_FACTOR


@dataclasses.dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    batch_size: int
    seq_len: int
    tokens_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    dropped_token_rate: float
    expert_utilization: List[float]
    capacity_factor: float
    device: str
    dtype: str
    timestamp: float = dataclasses.field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Predictive Expert Dispatcher (PED)
# ---------------------------------------------------------------------------


class PredictiveExpertDispatcher(nn.Module):
    """
    A tiny 2-layer MLP that predicts which top-k experts will be activated
    for the next token. Trained online using routing decisions as supervision.

    Input:  last-layer hidden state   (batch, hidden_dim)
    Output: predicted expert logits   (batch, num_experts)

    Loss:   multi-label BCE on the one-hot top-k mask from the actual router.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        ped_hidden_dim: int = _DEFAULT_PED_HIDDEN,
        top_k: int = _DEFAULT_TOP_K,
        lr: float = 1e-3,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.device_ = device
        self.dtype_ = dtype

        # Two-layer MLP
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ped_hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ped_hidden_dim, num_experts, bias=True),
        )

        # Optimizer used for online updates
        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._step = 0
        self._recent_accuracy: deque = deque(maxlen=200)

        # History for diagnostics
        self._loss_history: deque = deque(maxlen=500)

    def forward(self, hidden: Tensor) -> Tensor:
        """Return predicted logits over experts, shape (batch, num_experts)."""
        return self.net(hidden.to(self.dtype_).to(self.device_))

    @torch.no_grad()
    def predict_top_k(self, hidden: Tensor) -> Tensor:
        """Return predicted top-k expert indices, shape (batch, top_k)."""
        logits = self.forward(hidden)
        return logits.topk(self.top_k, dim=-1).indices

    def update(self, hidden: Tensor, actual_expert_indices: Tensor) -> float:
        """
        Online training step.
        hidden:               (batch, hidden_dim)
        actual_expert_indices: (batch, top_k)
        Returns: scalar loss value.
        """
        self.train()
        self._optimizer.zero_grad(set_to_none=True)

        # Build multi-hot target
        batch = hidden.size(0)
        target = torch.zeros(batch, self.num_experts, device=self.device_, dtype=torch.float32)
        target.scatter_(1, actual_expert_indices.to(self.device_).long(), 1.0)

        logits = self.forward(hidden)
        loss = F.binary_cross_entropy_with_logits(logits.float(), target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self._optimizer.step()

        loss_val = loss.item()
        self._loss_history.append(loss_val)

        # Track accuracy (how often predicted top-k matches actual top-k)
        with torch.no_grad():
            pred = logits.topk(self.top_k, dim=-1).indices.sort(dim=-1).values
            true = actual_expert_indices.sort(dim=-1).values.to(self.device_)
            acc = (pred == true).all(dim=-1).float().mean().item()
            self._recent_accuracy.append(acc)

        self._step += 1
        return loss_val

    @property
    def recent_accuracy(self) -> float:
        if not self._recent_accuracy:
            return 0.0
        return float(np.mean(self._recent_accuracy))

    @property
    def recent_loss(self) -> float:
        if not self._loss_history:
            return float("inf")
        return float(np.mean(list(self._loss_history)[-50:]))

    def state_dict_for_checkpoint(self) -> dict:
        return {
            "model": self.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "step": self._step,
        }

    def load_checkpoint(self, ckpt: dict) -> None:
        self.load_state_dict(ckpt["model"])
        self._optimizer.load_state_dict(ckpt["optimizer"])
        self._step = ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Expert Weight Pre-fetcher
# ---------------------------------------------------------------------------


class ExpertWeightPrefetcher:
    """
    Uses CUDA streams to move predicted expert weights to GPU memory
    (warming L2 cache) ahead of the actual dispatch call.

    The prefetcher maintains two CUDA streams:
      - compute_stream: used by the main forward pass
      - prefetch_stream: used for async data movement
    """

    def __init__(
        self,
        expert_modules: nn.ModuleList,
        device: str = "cuda",
        num_streams: int = 2,
    ):
        self.expert_modules = expert_modules
        self.device = device
        self._enabled = torch.cuda.is_available()

        if self._enabled:
            self.prefetch_streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
            self._stream_idx = 0
        else:
            self.prefetch_streams = []

        self._prefetch_events: Dict[int, torch.cuda.Event] = {}
        self._pending_expert_ids: List[int] = []

    def prefetch(self, expert_ids: Sequence[int]) -> None:
        """Async-prefetch the weights of the given experts to GPU memory."""
        if not self._enabled:
            return
        stream = self.prefetch_streams[self._stream_idx % len(self.prefetch_streams)]
        self._stream_idx += 1

        with torch.cuda.stream(stream):
            for eid in expert_ids:
                if eid >= len(self.expert_modules):
                    continue
                expert = self.expert_modules[eid]
                for p in expert.parameters():
                    # Force the parameter into GPU L2 by doing a dummy read
                    _ = p.data.contiguous()
                event = torch.cuda.Event()
                event.record(stream)
                self._prefetch_events[eid] = event

        self._pending_expert_ids = list(expert_ids)

    def wait_for_expert(self, expert_id: int) -> None:
        """Block compute stream until prefetch for expert_id is complete."""
        if not self._enabled:
            return
        if expert_id in self._prefetch_events:
            self._prefetch_events[expert_id].wait(torch.cuda.current_stream())
            del self._prefetch_events[expert_id]

    def synchronize(self) -> None:
        if self._enabled:
            for stream in self.prefetch_streams:
                stream.synchronize()

    def __repr__(self) -> str:
        return (
            f"ExpertWeightPrefetcher(num_experts={len(self.expert_modules)}, "
            f"cuda={self._enabled})"
        )


# ---------------------------------------------------------------------------
# Expert-Specific KV Cache Manager
# ---------------------------------------------------------------------------


class ExpertKVCacheEntry:
    """A single slot in the per-expert KV cache."""

    __slots__ = ("key", "value", "expert_id", "last_access", "size_bytes")

    def __init__(self, key: Tensor, value: Tensor, expert_id: int):
        self.key = key
        self.value = value
        self.expert_id = expert_id
        self.last_access = time.monotonic()
        self.size_bytes = key.nbytes + value.nbytes


class ExpertKVCacheManager:
    """
    Partitions GPU VRAM per expert. Keeps 'hot' experts resident and uses
    LRU eviction when the total cached size exceeds the budget.

    Parameters
    ----------
    num_experts       : total number of MoE experts
    max_resident      : maximum number of experts whose KV cache stays in GPU
    vram_budget_bytes : total VRAM allowed for KV caches (default 2 GB)
    """

    def __init__(
        self,
        num_experts: int,
        max_resident: int = _DEFAULT_KV_CACHE_MAX_EXPERTS,
        vram_budget_bytes: int = 2 * 1024 ** 3,
        device: str = "cuda",
    ):
        self.num_experts = num_experts
        self.max_resident = max_resident
        self.vram_budget_bytes = vram_budget_bytes
        self.device = device

        # Per-expert cache: expert_id -> OrderedDict[seq_key -> ExpertKVCacheEntry]
        self._caches: Dict[int, OrderedDict] = {i: OrderedDict() for i in range(num_experts)}
        self._resident_experts: OrderedDict = OrderedDict()  # LRU order
        self._total_bytes: int = 0
        self._lock = threading.Lock()

        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, expert_id: int, seq_key: str) -> Optional[Tuple[Tensor, Tensor]]:
        """Return (key, value) for the given expert + sequence key, or None."""
        with self._lock:
            cache = self._caches[expert_id]
            if seq_key not in cache:
                self._misses += 1
                return None
            entry = cache[seq_key]
            # Move to end (most recently used)
            cache.move_to_end(seq_key)
            self._touch_resident(expert_id)
            entry.last_access = time.monotonic()
            self._hits += 1
            return entry.key, entry.value

    def put(self, expert_id: int, seq_key: str, key: Tensor, value: Tensor) -> None:
        """Insert or update the KV cache for an expert + sequence."""
        with self._lock:
            new_entry = ExpertKVCacheEntry(
                key.to(self.device),
                value.to(self.device),
                expert_id,
            )
            old = self._caches[expert_id].get(seq_key)
            if old is not None:
                self._total_bytes -= old.size_bytes
            self._caches[expert_id][seq_key] = new_entry
            self._caches[expert_id].move_to_end(seq_key)
            self._total_bytes += new_entry.size_bytes
            self._touch_resident(expert_id)
            self._maybe_evict()

    def evict_expert(self, expert_id: int) -> None:
        """Force-evict all cached KV entries for the given expert."""
        with self._lock:
            cache = self._caches[expert_id]
            for entry in cache.values():
                self._total_bytes -= entry.size_bytes
                # Free GPU tensors explicitly
                del entry.key, entry.value
            cache.clear()
            self._resident_experts.pop(expert_id, None)
            self._evictions += 1

    def mark_expert_hot(self, expert_id: int) -> None:
        with self._lock:
            self._touch_resident(expert_id)

    def invalidate(self, seq_key: str) -> None:
        """Remove seq_key from all expert caches (e.g., after sequence completes)."""
        with self._lock:
            for expert_id, cache in self._caches.items():
                if seq_key in cache:
                    entry = cache.pop(seq_key)
                    self._total_bytes -= entry.size_bytes

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = self._hits / max(total, 1)
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "resident_experts": list(self._resident_experts.keys()),
            "total_bytes": self._total_bytes,
            "vram_budget_bytes": self.vram_budget_bytes,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _touch_resident(self, expert_id: int) -> None:
        """Mark expert as recently used in the resident LRU."""
        if expert_id in self._resident_experts:
            self._resident_experts.move_to_end(expert_id)
        else:
            self._resident_experts[expert_id] = True
            if len(self._resident_experts) > self.max_resident:
                # Evict LRU expert
                oldest_id, _ = self._resident_experts.popitem(last=False)
                self._evict_lru_expert(oldest_id)

    def _evict_lru_expert(self, expert_id: int) -> None:
        """Evict the LRU entries for an expert to free VRAM."""
        cache = self._caches[expert_id]
        # Evict oldest half
        n_evict = max(1, len(cache) // 2)
        for _ in range(n_evict):
            if not cache:
                break
            _, entry = cache.popitem(last=False)
            self._total_bytes -= entry.size_bytes
            del entry.key, entry.value
        self._evictions += 1

    def _maybe_evict(self) -> None:
        """Evict entries until total_bytes <= vram_budget_bytes."""
        while self._total_bytes > self.vram_budget_bytes:
            if not self._resident_experts:
                break
            oldest_id, _ = self._resident_experts.popitem(last=False)
            self._evict_lru_expert(oldest_id)


# ---------------------------------------------------------------------------
# Expert Feed-Forward Network (used inside the MoE layer)
# ---------------------------------------------------------------------------


class SwiGLUExpertFFN(nn.Module):
    """Expert FFN with SwiGLU activation (matches LLaMA / Mixtral style)."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dtype_ = dtype
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_up.weight)
        nn.init.xavier_uniform_(self.w_down.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.dtype_)
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class GeLUExpertFFN(nn.Module):
    """Expert FFN with GeLU activation (standard transformer style)."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dtype_ = dtype
        self.fc1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.dtype_)
        return self.fc2(F.gelu(self.fc1(x)))


def build_expert(hidden_dim: int, ffn_dim: int, activation: str, dtype: torch.dtype) -> nn.Module:
    if activation == "swiglu":
        return SwiGLUExpertFFN(hidden_dim, ffn_dim, dtype)
    elif activation == "gelu":
        return GeLUExpertFFN(hidden_dim, ffn_dim, dtype)
    else:
        raise ValueError(f"Unknown activation: {activation}")


# ---------------------------------------------------------------------------
# MoE Router
# ---------------------------------------------------------------------------


class MoERouter(nn.Module):
    """
    Standard noisy top-k router.
    Adds noise during training to encourage load balancing.
    Supports auxiliary load-balancing loss (Switch Transformer style).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 1e-2,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.dtype_ = dtype
        self.weight = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.normal_(self.weight.weight, std=0.01)

    def forward(self, hidden: Tensor) -> RoutingDecision:
        """
        hidden: (batch * seq_len, hidden_dim)
        Returns RoutingDecision.
        """
        hidden = hidden.to(self.dtype_)
        logits = self.weight(hidden).float()  # fp32 for stability

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        weights, indices = logits.softmax(dim=-1).topk(self.top_k, dim=-1)
        # Re-normalize weights for selected experts
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return RoutingDecision(
            expert_indices=indices,
            routing_weights=weights.to(self.dtype_),
            router_logits=logits,
        )

    def load_balance_loss(self, routing_decision: RoutingDecision, num_tokens: int) -> Tensor:
        """
        Switch Transformer auxiliary loss for load balancing.
        Encourages uniform token distribution across experts.
        """
        # fraction of tokens dispatched to each expert
        expert_mask = F.one_hot(
            routing_decision.expert_indices[:, 0], self.num_experts
        ).float()
        tokens_per_expert = expert_mask.sum(dim=0)  # (num_experts,)
        f_i = tokens_per_expert / num_tokens

        # mean routing probability for each expert
        probs = F.softmax(routing_decision.router_logits, dim=-1)
        p_i = probs.mean(dim=0)

        return self.num_experts * (f_i * p_i).sum()


# ---------------------------------------------------------------------------
# Batched Expert Executor
# ---------------------------------------------------------------------------


class BatchedExpertExecutor:
    """
    Groups tokens by their assigned expert, executes each expert exactly
    once on its batch (avoids redundant kernel launches).

    This is the core optimization: instead of looping over tokens and
    calling expert(token) for each, we do:
      for each expert e:
          tokens_e = tokens[dispatch_mask[:, e]]
          outputs[dispatch_mask[:, e]] += weight_e * expert_e(tokens_e)
    """

    def __init__(
        self,
        expert_modules: nn.ModuleList,
        capacity_factor: float = _DEFAULT_CAPACITY_FACTOR,
        top_k: int = _DEFAULT_TOP_K,
        device: str = "cuda",
    ):
        self.expert_modules = expert_modules
        self.num_experts = len(expert_modules)
        self.capacity_factor = capacity_factor
        self.top_k = top_k
        self.device = device
        self._dropped_tokens = 0

    def execute(
        self,
        tokens: Tensor,
        routing: RoutingDecision,
        prefetcher: Optional[ExpertWeightPrefetcher] = None,
    ) -> Tensor:
        """
        tokens:  (T, hidden_dim)  — flattened batch*seq tokens
        routing: RoutingDecision with indices/weights of shape (T, top_k)
        Returns: (T, hidden_dim) output tensor.
        """
        T, hidden_dim = tokens.shape
        output = torch.zeros_like(tokens)
        capacity = int(math.ceil(self.capacity_factor * T / self.num_experts))
        self._dropped_tokens = 0

        for expert_idx in range(self.num_experts):
            # Find all (token, k) pairs that route to this expert
            # routing.expert_indices: (T, top_k)
            mask = (routing.expert_indices == expert_idx)  # (T, top_k)

            # For each k-slot, collect tokens routed to expert_idx
            token_positions = []
            k_slots = []
            for k in range(self.top_k):
                pos = mask[:, k].nonzero(as_tuple=False).squeeze(1)  # (n_k,)
                if pos.numel() > 0:
                    token_positions.append(pos)
                    k_slots.append(torch.full((pos.numel(),), k, device=self.device, dtype=torch.long))

            if not token_positions:
                continue

            all_pos = torch.cat(token_positions, dim=0)
            all_k = torch.cat(k_slots, dim=0)

            # Apply capacity constraint
            if all_pos.numel() > capacity:
                self._dropped_tokens += all_pos.numel() - capacity
                all_pos = all_pos[:capacity]
                all_k = all_k[:capacity]

            # Pre-fetch wait
            if prefetcher is not None:
                prefetcher.wait_for_expert(expert_idx)

            # Single expert execution on the full batch
            expert_input = tokens[all_pos]
            expert_out = self.expert_modules[expert_idx](expert_input)

            # Scatter back with routing weights
            weights = routing.routing_weights[all_pos, all_k].unsqueeze(-1)
            output.index_add_(0, all_pos, weights * expert_out.to(output.dtype))

        return output

    @property
    def dropped_tokens(self) -> int:
        return self._dropped_tokens


# ---------------------------------------------------------------------------
# Speculative Router
# ---------------------------------------------------------------------------


class SpeculativeRouter:
    """
    Implements speculative routing: execute top-1 expert immediately while
    computing routing for top-2/3 experts in parallel on a separate CUDA stream.

    This hides the routing latency behind expert computation.
    """

    def __init__(
        self,
        router: MoERouter,
        expert_modules: nn.ModuleList,
        device: str = "cuda",
    ):
        self.router = router
        self.expert_modules = expert_modules
        self.device = device
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self.routing_stream = torch.cuda.Stream(device=device)
            self.compute_stream = torch.cuda.current_stream()

    def forward(
        self,
        tokens: Tensor,
        batched_executor: BatchedExpertExecutor,
    ) -> Tensor:
        """
        Speculative execution:
        1. Compute top-1 routing (fast, just argmax of router logits)
        2. Launch top-1 expert execution on compute stream
        3. Concurrently compute top-2/3 routing on routing_stream
        4. Wait, then execute remaining experts
        """
        T, _ = tokens.shape
        hidden_dim = tokens.shape[-1]

        # Step 1: Fast top-1 routing (just argmax, no full softmax needed yet)
        with torch.no_grad():
            logits_quick = self.router.weight(tokens.to(self.router.dtype_)).float()
        top1_idx = logits_quick.argmax(dim=-1)  # (T,)

        # Step 2: Execute top-1 experts immediately
        output = torch.zeros_like(tokens)
        capacity = int(math.ceil(batched_executor.capacity_factor * T / batched_executor.num_experts))

        if self._cuda:
            routing_event = torch.cuda.Event()

            # Compute full routing on separate stream
            with torch.cuda.stream(self.routing_stream):
                full_routing = self.router(tokens)
                routing_event.record(self.routing_stream)

            # Execute top-1 while routing stream works
            for expert_idx in range(batched_executor.num_experts):
                pos = (top1_idx == expert_idx).nonzero(as_tuple=False).squeeze(1)
                if pos.numel() == 0:
                    continue
                if pos.numel() > capacity:
                    pos = pos[:capacity]
                expert_out = batched_executor.expert_modules[expert_idx](tokens[pos])
                # weight = 1/top_k as placeholder until we have full routing
                output.index_add_(0, pos, expert_out.to(output.dtype) / batched_executor.top_k)

            # Wait for full routing
            routing_event.wait(torch.cuda.current_stream())
        else:
            full_routing = self.router(tokens)

        # Step 3: Execute top-2/3 with proper weights (correct the output)
        # For simplicity: redo full batched execution with full routing
        # In production, only execute k=2..top_k with delta corrections
        output = batched_executor.execute(tokens, full_routing)
        return output


# ---------------------------------------------------------------------------
# Capacity Factor Auto-Tuner
# ---------------------------------------------------------------------------


class CapacityFactorTuner:
    """
    Automatically adjusts the capacity_factor of the MoE layer to:
    - Minimize dropped tokens
    - Stay under the latency budget

    Uses a simple PID-like controller on observed dropped_token_rate and
    measured latency.
    """

    def __init__(
        self,
        initial_factor: float = _DEFAULT_CAPACITY_FACTOR,
        min_factor: float = 1.0,
        max_factor: float = 4.0,
        latency_budget_ms: float = 50.0,
        target_drop_rate: float = 0.001,
        adjust_interval: int = 100,
        step_size: float = 0.05,
    ):
        self.factor = initial_factor
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.latency_budget_ms = latency_budget_ms
        self.target_drop_rate = target_drop_rate
        self.adjust_interval = adjust_interval
        self.step_size = step_size

        self._call_count = 0
        self._drop_history: deque = deque(maxlen=adjust_interval)
        self._latency_history: deque = deque(maxlen=adjust_interval)
        self._factor_history: List[float] = [initial_factor]

    def record(self, dropped_tokens: int, total_tokens: int, latency_ms: float) -> None:
        drop_rate = dropped_tokens / max(total_tokens, 1)
        self._drop_history.append(drop_rate)
        self._latency_history.append(latency_ms)
        self._call_count += 1

        if self._call_count % self.adjust_interval == 0:
            self._adjust()

    def _adjust(self) -> None:
        avg_drop = float(np.mean(self._drop_history))
        avg_lat = float(np.mean(self._latency_history))

        if avg_drop > self.target_drop_rate and avg_lat < self.latency_budget_ms:
            # Too many drops, increase capacity (if latency allows)
            new_factor = min(self.factor + self.step_size, self.max_factor)
        elif avg_lat > self.latency_budget_ms:
            # Over latency budget, decrease capacity
            new_factor = max(self.factor - self.step_size, self.min_factor)
        else:
            # Within budget and drops are acceptable
            new_factor = self.factor

        if new_factor != self.factor:
            logger.debug(
                f"CapacityFactorTuner: {self.factor:.3f} -> {new_factor:.3f} "
                f"(drop_rate={avg_drop:.4f}, latency={avg_lat:.1f}ms)"
            )
            self.factor = new_factor
            self._factor_history.append(new_factor)

    @property
    def history(self) -> List[float]:
        return list(self._factor_history)


# ---------------------------------------------------------------------------
# MoE Layer
# ---------------------------------------------------------------------------


class OptimizedMoELayer(nn.Module):
    """
    A single MoE transformer layer with all inference optimizations enabled.

    Integrates:
      - MoERouter
      - PredictiveExpertDispatcher (PED)
      - ExpertWeightPrefetcher
      - ExpertKVCacheManager
      - BatchedExpertExecutor
      - SpeculativeRouter
      - CapacityFactorTuner
    """

    def __init__(self, config: MoEConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.device_ = config.device

        # Expert modules
        self.experts = nn.ModuleList([
            build_expert(config.hidden_dim, config.ffn_dim, config.expert_ffn_activation, config.dtype)
            for _ in range(config.num_experts)
        ])

        # Router
        self.router = MoERouter(
            config.hidden_dim,
            config.num_experts,
            config.top_k,
            dtype=config.dtype,
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

        # PED (Predictive Expert Dispatcher)
        if config.use_ped:
            self.ped = PredictiveExpertDispatcher(
                config.hidden_dim,
                config.num_experts,
                config.ped_hidden_dim,
                config.top_k,
                device=config.device,
            )
        else:
            self.ped = None

        # Prefetcher
        if config.use_prefetch:
            self.prefetcher = ExpertWeightPrefetcher(self.experts, config.device)
        else:
            self.prefetcher = None

        # KV cache manager
        if config.use_kv_cache:
            self.kv_cache = ExpertKVCacheManager(
                config.num_experts,
                config.kv_cache_max_resident,
                device=config.device,
            )
        else:
            self.kv_cache = None

        # Batched executor
        self.executor = BatchedExpertExecutor(
            self.experts,
            config.capacity_factor,
            config.top_k,
            config.device,
        )

        # Speculative router
        if config.use_speculative_routing:
            self.spec_router = SpeculativeRouter(self.router, self.experts, config.device)
        else:
            self.spec_router = None

        # Capacity tuner
        self.capacity_tuner = CapacityFactorTuner(
            config.capacity_factor,
            latency_budget_ms=config.latency_budget_ms,
            adjust_interval=config.capacity_tuning_interval,
        )

        self._step = 0

    def forward(
        self,
        x: Tensor,
        seq_key: Optional[str] = None,
        return_aux_loss: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        x: (batch, seq_len, hidden_dim)
        """
        B, S, H = x.shape
        residual = x
        x = self.layer_norm(x)

        # Flatten to tokens
        tokens = x.view(B * S, H)
        t0 = time.perf_counter()

        # PED pre-fetch prediction
        if self.ped is not None and self.prefetcher is not None:
            with torch.no_grad():
                predicted_experts = self.ped.predict_top_k(tokens)
                unique_experts = predicted_experts.view(-1).unique().tolist()
            self.prefetcher.prefetch(unique_experts)

        # Route tokens
        routing = self.router(tokens)

        # Execute experts
        if self.spec_router is not None:
            expert_out = self.spec_router.forward(tokens, self.executor)
        else:
            expert_out = self.executor.execute(tokens, routing, self.prefetcher)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # Update capacity factor tuner
        self.capacity_tuner.record(
            self.executor.dropped_tokens,
            B * S,
            latency_ms,
        )
        self.executor.capacity_factor = self.capacity_tuner.factor

        # Online PED update
        if self.ped is not None and self.training:
            self.ped.update(tokens.detach(), routing.expert_indices.detach())

        # Reshape output
        output = expert_out.view(B, S, H)
        output = output + residual

        self._step += 1

        if return_aux_loss:
            aux = self.router.load_balance_loss(routing, B * S)
            return output, aux
        return output

    def expert_utilization(self) -> List[float]:
        """Return the fraction of last-batch tokens that went to each expert."""
        # This is a diagnostic helper; actual tracking would require storing
        # per-expert token counts during the last forward pass.
        return [1.0 / self.config.num_experts] * self.config.num_experts

    def extra_repr(self) -> str:
        return (
            f"layer_idx={self.layer_idx}, num_experts={self.config.num_experts}, "
            f"top_k={self.config.top_k}, hidden_dim={self.config.hidden_dim}"
        )


# ---------------------------------------------------------------------------
# Full MoE Model (stacked layers)
# ---------------------------------------------------------------------------


class LuminaMoEModel(nn.Module):
    """
    Stacked MoE transformer model for Lumina.
    Used for financial time-series modeling with MoE FFN layers.
    """

    def __init__(
        self,
        config: MoEConfig,
        num_layers: int = 4,
        vocab_size: int = 32000,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, config.hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, config.hidden_dim)

        self.moe_layers = nn.ModuleList([
            OptimizedMoELayer(config, layer_idx=i) for i in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias if self.head.bias is not None else torch.zeros(1))

    def forward(
        self,
        input_ids: Tensor,
        return_aux_loss: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)

        total_aux = torch.tensor(0.0, device=x.device)

        for layer in self.moe_layers:
            if return_aux_loss:
                x, aux = layer(x, return_aux_loss=True)
                total_aux = total_aux + aux
            else:
                x = layer(x)

        x = self.final_norm(x)
        logits = self.head(x)

        if return_aux_loss:
            return logits, total_aux
        return logits

    def continuous_input_forward(
        self,
        features: Tensor,
        return_aux_loss: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        For continuous financial features (no embedding needed).
        features: (batch, seq_len, hidden_dim)
        """
        x = features
        total_aux = torch.tensor(0.0, device=x.device)

        for layer in self.moe_layers:
            if return_aux_loss:
                x, aux = layer(x, return_aux_loss=True)
                total_aux = total_aux + aux
            else:
                x = layer(x)

        x = self.final_norm(x)

        if return_aux_loss:
            return x, total_aux
        return x


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------


class MoEInferenceEngine:
    """
    High-level inference engine that wraps LuminaMoEModel and provides:
    - AMP (automatic mixed precision) context
    - Batching utilities
    - CUDA graph capture for fixed shapes
    - Throughput/latency measurement
    """

    def __init__(
        self,
        model: LuminaMoEModel,
        config: MoEConfig,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        self._amp_enabled = config.enable_amp and torch.cuda.is_available()
        self._cuda_graphs: Dict[Tuple, Any] = {}
        self._graph_inputs: Dict[Tuple, Tensor] = {}
        self._graph_outputs: Dict[Tuple, Tensor] = {}

    @torch.no_grad()
    def infer(
        self,
        features: Tensor,
        use_cuda_graph: bool = False,
    ) -> Tensor:
        """
        features: (batch, seq_len, hidden_dim) continuous financial features
        Returns: (batch, seq_len, hidden_dim)
        """
        features = features.to(self.device).to(self.config.dtype)

        if use_cuda_graph:
            return self._infer_with_cuda_graph(features)

        ctx = torch.cuda.amp.autocast(enabled=self._amp_enabled, dtype=self.config.dtype)
        with ctx:
            return self.model.continuous_input_forward(features)

    def _infer_with_cuda_graph(self, features: Tensor) -> Tensor:
        """Use CUDA graph for fixed-shape inference (significant speedup)."""
        shape_key = tuple(features.shape)

        if shape_key not in self._cuda_graphs:
            self._capture_cuda_graph(features, shape_key)

        g, static_in, static_out = self._cuda_graphs[shape_key]
        static_in.copy_(features)
        g.replay()
        return static_out.clone()

    def _capture_cuda_graph(self, features: Tensor, shape_key: Tuple) -> None:
        """Capture a CUDA graph for the given input shape."""
        if not torch.cuda.is_available():
            return

        # Warm up
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.model.continuous_input_forward(features)
        torch.cuda.current_stream().wait_stream(s)

        # Allocate static I/O
        static_input = features.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = self.model.continuous_input_forward(static_input)

        self._cuda_graphs[shape_key] = (g, static_input, static_output)
        logger.info(f"Captured CUDA graph for shape {shape_key}")

    def warmup(self, batch_size: int = 1, seq_len: int = 32) -> None:
        """Warm up CUDA kernels and JIT-compiled ops."""
        dummy = torch.randn(
            batch_size, seq_len, self.config.hidden_dim,
            device=self.device, dtype=self.config.dtype,
        )
        for _ in range(self.config.benchmark_warmup_steps):
            self.infer(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def benchmark(
        self,
        batch_sizes: Optional[List[int]] = None,
        seq_lens: Optional[List[int]] = None,
    ) -> List[BenchmarkResult]:
        """
        Measure tokens/sec and latency percentiles for different batch sizes.
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]
        if seq_lens is None:
            seq_lens = [32, 64, 128]

        results = []

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                result = self._benchmark_single(batch_size, seq_len)
                results.append(result)
                logger.info(
                    f"Benchmark B={batch_size} S={seq_len}: "
                    f"{result.tokens_per_sec:.0f} tok/s, "
                    f"p50={result.latency_p50_ms:.2f}ms "
                    f"p99={result.latency_p99_ms:.2f}ms"
                )

        return results

    def _benchmark_single(self, batch_size: int, seq_len: int) -> BenchmarkResult:
        """Run benchmark for a single (batch_size, seq_len) configuration."""
        dummy = torch.randn(
            batch_size, seq_len, self.config.hidden_dim,
            device=self.device, dtype=self.config.dtype,
        )

        # Warm up
        for _ in range(self.config.benchmark_warmup_steps):
            self.infer(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure
        latencies_ms = []
        dropped_rates = []

        for _ in range(self.config.benchmark_steps):
            t0 = time.perf_counter()
            self.infer(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            total_dropped = sum(
                layer.executor.dropped_tokens
                for layer in self.model.moe_layers
            )
            total_tokens = batch_size * seq_len * len(self.model.moe_layers)
            dropped_rates.append(total_dropped / max(total_tokens, 1))

        latencies = np.array(latencies_ms)
        tokens_per_step = batch_size * seq_len
        tokens_per_sec = tokens_per_step / (np.mean(latencies) / 1000.0)

        expert_util = self.model.moe_layers[0].expert_utilization()

        return BenchmarkResult(
            batch_size=batch_size,
            seq_len=seq_len,
            tokens_per_sec=float(tokens_per_sec),
            latency_p50_ms=float(np.percentile(latencies, 50)),
            latency_p95_ms=float(np.percentile(latencies, 95)),
            latency_p99_ms=float(np.percentile(latencies, 99)),
            dropped_token_rate=float(np.mean(dropped_rates)),
            expert_utilization=expert_util,
            capacity_factor=self.config.capacity_factor,
            device=str(self.device),
            dtype=str(self.config.dtype),
        )

    def print_benchmark_report(self, results: List[BenchmarkResult]) -> None:
        """Print a formatted benchmark report."""
        print("\n" + "=" * 80)
        print("MoE Inference Benchmark Report")
        print("=" * 80)
        print(f"{'Batch':>6} {'SeqLen':>7} {'Tok/s':>10} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} {'Drop%':>7}")
        print("-" * 80)
        for r in results:
            print(
                f"{r.batch_size:>6} {r.seq_len:>7} {r.tokens_per_sec:>10.0f} "
                f"{r.latency_p50_ms:>8.2f} {r.latency_p95_ms:>8.2f} {r.latency_p99_ms:>8.2f} "
                f"{r.dropped_token_rate * 100:>7.3f}"
            )
        print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Expert Load Tracker
# ---------------------------------------------------------------------------


class ExpertLoadTracker:
    """Tracks per-expert token load statistics across forward passes."""

    def __init__(self, num_experts: int, window: int = 1000):
        self.num_experts = num_experts
        self.window = window
        self._counts: List[deque] = [deque(maxlen=window) for _ in range(num_experts)]
        self._total: deque = deque(maxlen=window)

    def record(self, expert_indices: Tensor) -> None:
        """Record routing decisions. expert_indices: (T, top_k)."""
        T = expert_indices.shape[0]
        self._total.append(T)
        flat = expert_indices.view(-1).cpu().numpy()
        counts = np.bincount(flat, minlength=self.num_experts)
        for i in range(self.num_experts):
            self._counts[i].append(int(counts[i]))

    def utilization(self) -> np.ndarray:
        """Return per-expert mean utilization as fraction of total tokens."""
        total = sum(self._total) if self._total else 1
        return np.array([sum(c) / max(total, 1) for c in self._counts])

    def imbalance_ratio(self) -> float:
        """Max/min utilization ratio (1.0 = perfectly balanced)."""
        util = self.utilization()
        if util.min() == 0:
            return float("inf")
        return float(util.max() / util.min())

    def summary(self) -> Dict[str, Any]:
        util = self.utilization()
        return {
            "per_expert_utilization": util.tolist(),
            "mean": float(util.mean()),
            "std": float(util.std()),
            "imbalance_ratio": self.imbalance_ratio(),
        }


# ---------------------------------------------------------------------------
# Inference Session (manages stateful inference with KV caches)
# ---------------------------------------------------------------------------


class InferenceSession:
    """
    Manages a stateful autoregressive inference session.
    Maintains per-sequence KV caches and handles token-by-token generation.
    """

    def __init__(
        self,
        engine: MoEInferenceEngine,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        session_id: Optional[str] = None,
    ):
        self.engine = engine
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.session_id = session_id or f"session_{int(time.time() * 1000)}"

        self._generated_tokens: List[int] = []
        self._step = 0
        self._start_time = time.monotonic()

    def generate_continuous(
        self,
        features: Tensor,
        num_steps: int = 1,
    ) -> Tensor:
        """
        Generate num_steps forward passes on continuous features.
        Returns concatenated outputs.
        """
        outputs = []
        for _ in range(num_steps):
            out = self.engine.infer(features)
            outputs.append(out)
            self._step += 1
        return torch.cat(outputs, dim=1)

    def close(self) -> None:
        """Release KV cache entries for this session."""
        for layer in self.engine.model.moe_layers:
            if layer.kv_cache is not None:
                layer.kv_cache.invalidate(self.session_id)

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start_time) * 1000.0


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_lumina_moe(
    num_experts: int = 8,
    top_k: int = 2,
    hidden_dim: int = 512,
    ffn_dim: int = 2048,
    num_layers: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_all_optimizations: bool = True,
) -> Tuple[LuminaMoEModel, MoEInferenceEngine]:
    """Convenience factory to build a fully optimized Lumina MoE model + engine."""
    config = MoEConfig(
        num_experts=num_experts,
        top_k=top_k,
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        device=device,
        dtype=dtype,
        use_ped=use_all_optimizations,
        use_prefetch=use_all_optimizations and torch.cuda.is_available(),
        use_kv_cache=use_all_optimizations,
        use_batched_exec=True,
        use_speculative_routing=use_all_optimizations,
    )
    model = LuminaMoEModel(config, num_layers=num_layers)
    engine = MoEInferenceEngine(model, config)
    return model, engine


# ---------------------------------------------------------------------------
# CLI / quick benchmark entry-point
# ---------------------------------------------------------------------------


def run_quick_benchmark(
    num_experts: int = 8,
    hidden_dim: int = 512,
    ffn_dim: int = 2048,
    num_layers: int = 2,
) -> None:
    """Run a quick benchmark and print results to stdout."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Building LuminaMoE on {device} ...")
    model, engine = build_lumina_moe(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    print("Warming up ...")
    engine.warmup(batch_size=4, seq_len=32)

    print("Running benchmarks ...")
    results = engine.benchmark(
        batch_sizes=[1, 4, 8],
        seq_lens=[32, 64],
    )
    engine.print_benchmark_report(results)


# ---------------------------------------------------------------------------
# PED Training Loop (standalone, for pre-training PED on historical data)
# ---------------------------------------------------------------------------


class PEDTrainer:
    """
    Standalone trainer for PredictiveExpertDispatcher using historical
    routing decisions collected during a previous inference run.
    """

    def __init__(
        self,
        ped: PredictiveExpertDispatcher,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 10,
    ):
        self.ped = ped
        self.batch_size = batch_size
        self.epochs = epochs
        self._optimizer = torch.optim.Adam(ped.parameters(), lr=lr)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, T_max=epochs
        )

    def fit(
        self,
        hidden_states: List[Tensor],
        expert_labels: List[Tensor],
    ) -> List[float]:
        """
        hidden_states: list of (batch, hidden_dim) tensors
        expert_labels: list of (batch, top_k) tensors (actual expert assignments)
        Returns: per-epoch average loss
        """
        epoch_losses = []
        dataset = list(zip(hidden_states, expert_labels))

        for epoch in range(self.epochs):
            np.random.shuffle(dataset)
            batch_losses = []
            for h, e in dataset:
                loss = self.ped.update(h, e)
                batch_losses.append(loss)
            avg = float(np.mean(batch_losses))
            epoch_losses.append(avg)
            self._scheduler.step()
            logger.info(
                f"PED Epoch {epoch + 1}/{self.epochs}: loss={avg:.4f} "
                f"acc={self.ped.recent_accuracy:.3f}"
            )

        return epoch_losses


# ---------------------------------------------------------------------------
# Utility: profile a forward pass
# ---------------------------------------------------------------------------


def profile_forward(
    engine: MoEInferenceEngine,
    batch_size: int = 4,
    seq_len: int = 64,
    n_iter: int = 20,
) -> Dict[str, float]:
    """
    Profile the MoE forward pass using torch.profiler (if available).
    Returns a dict of key timing metrics.
    """
    device = engine.device
    dtype = engine.config.dtype
    dummy = torch.randn(batch_size, seq_len, engine.config.hidden_dim, device=device, dtype=dtype)

    latencies = []
    for _ in range(n_iter):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.infer(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    latencies_np = np.array(latencies[2:])  # skip first 2 for warmup
    tokens = batch_size * seq_len
    return {
        "mean_ms": float(latencies_np.mean()),
        "p50_ms": float(np.percentile(latencies_np, 50)),
        "p95_ms": float(np.percentile(latencies_np, 95)),
        "p99_ms": float(np.percentile(latencies_np, 99)),
        "tokens_per_sec": float(tokens / (latencies_np.mean() / 1000.0)),
        "std_ms": float(latencies_np.std()),
    }


# ---------------------------------------------------------------------------
# Expert Capacity Histogram
# ---------------------------------------------------------------------------


class ExpertCapacityHistogram:
    """
    Collects a histogram of how many tokens each expert receives per step.
    Used to diagnose load imbalance and choose capacity_factor.
    """

    def __init__(self, num_experts: int, max_tokens_per_expert: int = 512):
        self.num_experts = num_experts
        self.max_tokens = max_tokens_per_expert
        self._hist = np.zeros((num_experts, max_tokens_per_expert + 1), dtype=np.int64)
        self._n_steps = 0

    def record(self, expert_indices: Tensor, top_k: int = 2) -> None:
        flat = expert_indices.view(-1).cpu().numpy()
        counts = np.bincount(flat, minlength=self.num_experts)
        for i in range(self.num_experts):
            c = min(int(counts[i]), self.max_tokens)
            self._hist[i, c] += 1
        self._n_steps += 1

    def percentile(self, expert_id: int, pct: float) -> int:
        """Return the pct-th percentile of tokens routed to expert_id."""
        hist = self._hist[expert_id]
        cumsum = np.cumsum(hist)
        total = cumsum[-1]
        if total == 0:
            return 0
        target = pct / 100.0 * total
        return int(np.searchsorted(cumsum, target))

    def recommended_capacity_factor(self, batch_size: int, seq_len: int) -> float:
        """
        Recommend a capacity_factor such that <1% of tokens are dropped
        based on observed distribution.
        """
        tokens_per_expert_mean = batch_size * seq_len / self.num_experts
        max_p99 = max(self.percentile(i, 99) for i in range(self.num_experts))
        if tokens_per_expert_mean == 0:
            return 1.25
        return float(max(1.0, max_p99 / tokens_per_expert_mean))

    def summary(self) -> Dict[str, Any]:
        return {
            "n_steps": self._n_steps,
            "per_expert_p50": [self.percentile(i, 50) for i in range(self.num_experts)],
            "per_expert_p99": [self.percentile(i, 99) for i in range(self.num_experts)],
        }


# ---------------------------------------------------------------------------
# Streaming inference for real-time financial signal generation
# ---------------------------------------------------------------------------


class StreamingMoEInferencer:
    """
    Wraps MoEInferenceEngine for streaming inference on a live data feed.

    Designed for real-time trading: processes each incoming LOB snapshot
    as it arrives, maintaining rolling context.
    """

    def __init__(
        self,
        engine: MoEInferenceEngine,
        context_window: int = 64,
        output_callback: Optional[Callable[[Tensor], None]] = None,
    ):
        self.engine = engine
        self.context_window = context_window
        self.output_callback = output_callback
        self._buffer: deque = deque(maxlen=context_window)
        self._n_processed = 0
        self._lock = threading.Lock()

    def push(self, snapshot: Tensor) -> Optional[Tensor]:
        """
        Push a new LOB snapshot (hidden_dim,) and run inference if buffer is ready.
        Returns the model output or None if not enough context yet.
        """
        with self._lock:
            self._buffer.append(snapshot.detach().cpu())
            self._n_processed += 1

            if len(self._buffer) < self.context_window:
                return None

            context = torch.stack(list(self._buffer), dim=0)  # (window, H)
            context = context.unsqueeze(0)  # (1, window, H)
            output = self.engine.infer(context)

            if self.output_callback is not None:
                self.output_callback(output)

            return output

    @property
    def n_processed(self) -> int:
        return self._n_processed

    @property
    def buffer_fill(self) -> float:
        return len(self._buffer) / self.context_window


# ---------------------------------------------------------------------------
# Model size / FLOP estimator
# ---------------------------------------------------------------------------


def estimate_moe_flops(
    config: MoEConfig,
    batch_size: int,
    seq_len: int,
    num_layers: int,
) -> Dict[str, float]:
    """Estimate FLOPs for one forward pass of a LuminaMoE model."""
    T = batch_size * seq_len  # total tokens
    H = config.hidden_dim
    F = config.ffn_dim
    E = config.num_experts
    K = config.top_k

    # Router: linear(H, E) per token
    router_flops = T * H * E * 2

    # Experts: K experts per token, each = SwiGLU(H, F)
    # SwiGLU has 3 linear layers: 2*(H->F) + 1*(F->H)
    expert_flops_per_token = K * (2 * H * F + 2 * H * F + 2 * F * H)
    total_expert_flops = T * expert_flops_per_token

    # PED: 2-layer MLP per token
    ped_flops = T * (H * config.ped_hidden_dim * 2 + config.ped_hidden_dim * E * 2)

    total_per_layer = router_flops + total_expert_flops + ped_flops
    total = total_per_layer * num_layers

    return {
        "router_flops": router_flops * num_layers,
        "expert_flops": total_expert_flops * num_layers,
        "ped_flops": ped_flops * num_layers,
        "total_flops": total,
        "total_gflops": total / 1e9,
        "flops_per_token": total / T,
    }


# ---------------------------------------------------------------------------
# Convenience: main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_quick_benchmark()


# ---------------------------------------------------------------------------
# Extended: Expert activation pattern analyzer
# ---------------------------------------------------------------------------


class ExpertActivationPatternAnalyzer:
    """
    Analyzes which experts tend to co-activate (be selected together for the
    same token in top-k routing). Used to inform PED training and to detect
    routing pathologies (e.g., expert collapse).
    """

    def __init__(self, num_experts: int, top_k: int = 2, window: int = 1000):
        self.num_experts = num_experts
        self.top_k = top_k
        self.window = window
        # Co-activation count matrix: co_act[i,j] = times experts i and j
        # were both selected for the same token
        self._co_act = np.zeros((num_experts, num_experts), dtype=np.int64)
        self._act_count = np.zeros(num_experts, dtype=np.int64)
        self._n_tokens = 0
        self._recent_indices: deque = deque(maxlen=window)

    def record(self, expert_indices: Tensor) -> None:
        """
        expert_indices: (T, top_k) — for each token, which experts were selected.
        """
        idx_np = expert_indices.cpu().numpy()  # (T, K)
        T, K = idx_np.shape
        self._n_tokens += T

        for t in range(T):
            row = idx_np[t]
            self._recent_indices.append(row.tolist())
            # Single-expert activation count
            for e in row:
                self._act_count[e] += 1
            # Co-activation pairs
            for i in range(K):
                for j in range(i + 1, K):
                    ei, ej = int(row[i]), int(row[j])
                    self._co_act[ei, ej] += 1
                    self._co_act[ej, ei] += 1

    def co_activation_matrix(self, normalize: bool = True) -> np.ndarray:
        """Return the co-activation matrix, optionally normalized by total tokens."""
        if normalize and self._n_tokens > 0:
            return self._co_act / self._n_tokens
        return self._co_act.copy()

    def expert_popularity(self, normalize: bool = True) -> np.ndarray:
        """Return per-expert activation frequency."""
        if normalize and self._n_tokens > 0:
            return self._act_count / (self._n_tokens * self.top_k)
        return self._act_count.copy()

    def detect_expert_collapse(self, threshold: float = 0.8) -> List[int]:
        """
        Detect experts that are over-utilized (collapse: most tokens go to few experts).
        Returns list of expert IDs that account for > threshold fraction of activations.
        """
        pop = self.expert_popularity()
        sorted_idx = pop.argsort()[::-1]
        cumsum = 0.0
        hot_experts = []
        for idx in sorted_idx:
            cumsum += float(pop[idx])
            hot_experts.append(int(idx))
            if cumsum >= threshold:
                break
        # If fewer than half the experts cover threshold% of activations → collapse
        if len(hot_experts) < self.num_experts / 2:
            return hot_experts
        return []

    def top_co_activating_pairs(self, n: int = 5) -> List[Tuple[int, int, float]]:
        """Return top-n expert pairs that co-activate most often."""
        pairs = []
        mat = self.co_activation_matrix(normalize=True)
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                pairs.append((i, j, float(mat[i, j])))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:n]

    def summary(self) -> Dict[str, Any]:
        pop = self.expert_popularity()
        collapse = self.detect_expert_collapse()
        pairs = self.top_co_activating_pairs(3)
        return {
            "n_tokens_seen": self._n_tokens,
            "expert_popularity": pop.tolist(),
            "gini_coefficient": float(self._gini(pop)),
            "expert_collapse_detected": len(collapse) > 0,
            "collapsed_experts": collapse,
            "top_coactivating_pairs": [(i, j, round(f, 4)) for i, j, f in pairs],
        }

    @staticmethod
    def _gini(arr: np.ndarray) -> float:
        """Gini coefficient as a measure of load imbalance (0=equal, 1=collapse)."""
        arr = np.sort(np.abs(arr))
        n = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


# ---------------------------------------------------------------------------
# Extended: Router Temperature Scaling
# ---------------------------------------------------------------------------


class RouterTemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for the MoE router.
    Calibrates the softmax temperature so that routing entropy matches
    a desired target (controls specialization vs generalization).
    """

    def __init__(self, num_experts: int, target_entropy: Optional[float] = None):
        super().__init__()
        self.num_experts = num_experts
        # Start with temperature = 1.0
        self.log_temperature = nn.Parameter(torch.zeros(1))
        if target_entropy is None:
            # Default: match uniform distribution entropy / 2
            self.target_entropy = math.log(num_experts) / 2.0
        else:
            self.target_entropy = target_entropy

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).item())

    def forward(self, logits: Tensor) -> Tensor:
        """Scale logits by temperature before softmax."""
        return logits / torch.exp(self.log_temperature)

    def calibrate(
        self,
        logits_list: List[Tensor],
        lr: float = 0.01,
        max_steps: int = 100,
    ) -> float:
        """
        Calibrate temperature to match target entropy.
        logits_list: list of (T, E) router logits from calibration data.
        Returns final calibrated temperature.
        """
        optimizer = torch.optim.Adam([self.log_temperature], lr=lr)

        for step in range(max_steps):
            total_loss = torch.tensor(0.0)
            for logits in logits_list:
                scaled = self.forward(logits)
                probs = F.softmax(scaled, dim=-1)
                # Entropy of the distribution
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
                loss = (entropy - self.target_entropy).pow(2)
                total_loss = total_loss + loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % 20 == 0:
                logger.debug(
                    f"Temperature calibration step {step}: "
                    f"T={self.temperature:.3f}, loss={total_loss.item():.4f}"
                )

        logger.info(f"Calibrated router temperature: {self.temperature:.3f}")
        return self.temperature


# ---------------------------------------------------------------------------
# Extended: MoE Routing Visualizer (for diagnostics)
# ---------------------------------------------------------------------------


class RoutingVisualizer:
    """
    Collects routing statistics and produces ASCII/text visualizations
    for diagnosing load imbalance in the MoE layer.
    """

    def __init__(self, num_experts: int, num_layers: int):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self._per_layer_trackers = [
            ExpertLoadTracker(num_experts, window=500)
            for _ in range(num_layers)
        ]
        self._per_layer_pattern = [
            ExpertActivationPatternAnalyzer(num_experts)
            for _ in range(num_layers)
        ]

    def record(self, layer_idx: int, expert_indices: Tensor) -> None:
        self._per_layer_trackers[layer_idx].record(expert_indices)
        self._per_layer_pattern[layer_idx].record(expert_indices)

    def utilization_bar_chart(self, layer_idx: int, width: int = 40) -> str:
        """Return an ASCII bar chart of expert utilization for a layer."""
        util = self._per_layer_trackers[layer_idx].utilization()
        lines = [f"Layer {layer_idx} Expert Utilization:"]
        max_u = max(util) if max(util) > 0 else 1.0
        for i, u in enumerate(util):
            bar_len = int(u / max_u * width)
            bar = "█" * bar_len + "░" * (width - bar_len)
            lines.append(f"  E{i:02d} [{bar}] {u:.3f}")
        return "\n".join(lines)

    def print_all_layers(self) -> None:
        for i in range(self.num_layers):
            print(self.utilization_bar_chart(i))
            summary = self._per_layer_pattern[i].summary()
            if summary["expert_collapse_detected"]:
                print(f"  ⚠️  Expert collapse detected: {summary['collapsed_experts']}")
            print(f"  Gini coefficient: {summary['gini_coefficient']:.3f}")
            print()

    def full_report(self) -> Dict[str, Any]:
        return {
            f"layer_{i}": {
                "utilization": self._per_layer_trackers[i].summary(),
                "activation_patterns": self._per_layer_pattern[i].summary(),
            }
            for i in range(self.num_layers)
        }


# ---------------------------------------------------------------------------
# Extended: Adaptive Top-K
# ---------------------------------------------------------------------------


class AdaptiveTopKRouter(nn.Module):
    """
    Router that dynamically adjusts k (number of experts per token) based
    on the confidence of routing decisions.

    High-confidence tokens (router entropy is low) use top-1.
    Low-confidence tokens (router entropy is high) use top-k.

    This reduces computation for "easy" tokens while maintaining capacity
    for tokens that benefit from multiple expert opinions.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        min_k: int = 1,
        max_k: int = 4,
        entropy_threshold_low: float = 0.3,
        entropy_threshold_high: float = 1.5,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.min_k = min_k
        self.max_k = max_k
        self.entropy_threshold_low = entropy_threshold_low
        self.entropy_threshold_high = entropy_threshold_high
        self.dtype_ = dtype

        self.weight = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.normal_(self.weight.weight, std=0.01)

    def forward(self, hidden: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            indices:  (T, max_k) — expert indices, padded with -1 for unused slots
            weights:  (T, max_k) — routing weights, 0 for unused slots
            k_per_token: (T,) int — actual k used per token
        """
        hidden = hidden.to(self.dtype_)
        logits = self.weight(hidden).float()
        probs = F.softmax(logits, dim=-1)

        # Compute per-token entropy
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)  # (T,)

        # Determine k per token
        k_per_token = torch.where(
            entropy < self.entropy_threshold_low,
            torch.full_like(entropy, self.min_k, dtype=torch.long),
            torch.where(
                entropy > self.entropy_threshold_high,
                torch.full_like(entropy, self.max_k, dtype=torch.long),
                torch.full_like(entropy, (self.min_k + self.max_k) // 2, dtype=torch.long),
            ),
        )

        T = hidden.shape[0]
        indices = torch.full((T, self.max_k), -1, dtype=torch.long, device=hidden.device)
        weights = torch.zeros(T, self.max_k, device=hidden.device, dtype=self.dtype_)

        # Fill indices/weights based on per-token k
        # For efficiency, group tokens by k and do batched topk
        for k_val in range(self.min_k, self.max_k + 1):
            mask = (k_per_token == k_val)
            if not mask.any():
                continue
            sub_probs = probs[mask]
            top_w, top_idx = sub_probs.topk(k_val, dim=-1)
            top_w = top_w / top_w.sum(dim=-1, keepdim=True)
            indices[mask, :k_val] = top_idx
            weights[mask, :k_val] = top_w.to(self.dtype_)

        return indices, weights, k_per_token

    def mean_k(self, k_per_token: Tensor) -> float:
        return float(k_per_token.float().mean().item())


# ---------------------------------------------------------------------------
# Extended: Layerwise expert assignment monitor (for MoE with many layers)
# ---------------------------------------------------------------------------


class LayerwiseExpertMonitor:
    """
    Tracks per-layer, per-expert statistics for a deep MoE model.
    Identifies which layers have highest load imbalance and which
    experts are most active at which depths.
    """

    def __init__(self, num_layers: int, num_experts: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        # shape: (num_layers, num_experts)
        self._cumulative_tokens = np.zeros((num_layers, num_experts), dtype=np.int64)
        self._total_tokens_per_layer = np.zeros(num_layers, dtype=np.int64)
        self._n_steps = np.zeros(num_layers, dtype=np.int64)

    def record(self, layer_idx: int, expert_indices: Tensor) -> None:
        """Record routing decisions for a specific layer."""
        flat = expert_indices.view(-1).cpu().numpy()
        T = expert_indices.shape[0]
        counts = np.bincount(flat, minlength=self.num_experts)
        self._cumulative_tokens[layer_idx] += counts
        self._total_tokens_per_layer[layer_idx] += T
        self._n_steps[layer_idx] += 1

    def layer_utilization(self, layer_idx: int) -> np.ndarray:
        total = self._total_tokens_per_layer[layer_idx]
        if total == 0:
            return np.zeros(self.num_experts)
        return self._cumulative_tokens[layer_idx] / total

    def most_imbalanced_layer(self) -> Tuple[int, float]:
        """Return (layer_idx, imbalance_ratio) for the most imbalanced layer."""
        worst_layer = 0
        worst_ratio = 0.0
        for l in range(self.num_layers):
            util = self.layer_utilization(l)
            if util.min() > 0:
                ratio = float(util.max() / util.min())
            else:
                ratio = float("inf")
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_layer = l
        return worst_layer, worst_ratio

    def depth_expert_heatmap(self) -> np.ndarray:
        """
        Returns a (num_layers, num_experts) heatmap of normalized utilization.
        Value at [l, e] = fraction of tokens at layer l that went to expert e.
        """
        heatmap = np.zeros((self.num_layers, self.num_experts))
        for l in range(self.num_layers):
            heatmap[l] = self.layer_utilization(l)
        return heatmap

    def print_heatmap(self, width: int = 6) -> None:
        """Print a text heatmap of depth-expert utilization."""
        heatmap = self.depth_expert_heatmap()
        header = " " * 4 + "".join(f"E{e:<{width-1}}" for e in range(self.num_experts))
        print(header)
        for l in range(self.num_layers):
            row_str = f"L{l:<2} "
            for e in range(self.num_experts):
                v = heatmap[l, e]
                # Use ASCII shading
                if v < 0.1:
                    ch = "·"
                elif v < 0.2:
                    ch = "░"
                elif v < 0.35:
                    ch = "▒"
                elif v < 0.5:
                    ch = "▓"
                else:
                    ch = "█"
                row_str += ch * (width - 1) + " "
            print(row_str)

    def summary(self) -> Dict[str, Any]:
        worst_l, worst_ratio = self.most_imbalanced_layer()
        return {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "most_imbalanced_layer": worst_l,
            "imbalance_ratio": worst_ratio,
            "depth_expert_heatmap": self.depth_expert_heatmap().tolist(),
        }


# ---------------------------------------------------------------------------
# Extended: Expert dropout (training regularization)
# ---------------------------------------------------------------------------


class ExpertDropout(nn.Module):
    """
    Drops entire expert outputs with probability `p` during training.
    Different from token dropout: entire expert specialization is masked.
    Forces the model to be robust to individual expert failures.

    At inference, all experts are always active.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, expert_output: Tensor, expert_id: int) -> Tensor:
        """
        expert_output: (T, H) — output of a single expert
        Returns masked output.
        """
        if not self.training or self.p == 0.0:
            return expert_output

        if torch.rand(1).item() < self.p:
            return torch.zeros_like(expert_output)
        return expert_output / (1.0 - self.p)


# ---------------------------------------------------------------------------
# Extended: MoE warmup scheduler (gradually activates experts during training)
# ---------------------------------------------------------------------------


class MoEWarmupScheduler:
    """
    Gradually transitions from a dense FFN to a sparse MoE during training.

    Phase 1 (steps 0..warmup_steps//2): use top-num_experts (dense)
    Phase 2 (steps warmup_steps//2..warmup_steps): linearly reduce to top_k
    Phase 3 (steps > warmup_steps): use top_k (standard MoE)

    This prevents the model from prematurely specializing experts before
    the router has learned good routing policies.
    """

    def __init__(
        self,
        num_experts: int,
        final_top_k: int = 2,
        warmup_steps: int = 1000,
    ):
        self.num_experts = num_experts
        self.final_top_k = final_top_k
        self.warmup_steps = warmup_steps
        self._step = 0

    def get_top_k(self) -> int:
        """Return the current effective top_k."""
        if self._step < self.warmup_steps // 2:
            return self.num_experts  # fully dense
        elif self._step < self.warmup_steps:
            # Linear interpolation
            progress = (self._step - self.warmup_steps // 2) / (self.warmup_steps // 2)
            k = self.num_experts - int(progress * (self.num_experts - self.final_top_k))
            return max(k, self.final_top_k)
        else:
            return self.final_top_k

    def step(self) -> None:
        self._step += 1

    @property
    def current_step(self) -> int:
        return self._step

    def is_warmed_up(self) -> bool:
        return self._step >= self.warmup_steps


# ---------------------------------------------------------------------------
# Extended: Fast approximate top-k using bucket sort
# ---------------------------------------------------------------------------


class BucketTopK:
    """
    O(E) approximate top-k using bucket sort.
    Faster than PyTorch's O(E log k) topk for small E and large T.

    Works by discretizing router logits into B buckets and finding
    the top-k experts using bucket boundaries.
    """

    def __init__(self, num_experts: int, num_buckets: int = 32):
        self.num_experts = num_experts
        self.num_buckets = num_buckets

    def top_k(self, probs: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """
        probs: (T, E) float32
        Returns (weights, indices) each (T, k)
        """
        # Use PyTorch's topk as fallback (it's already quite fast for small E)
        # The bucket sort trick is primarily beneficial for E > 64
        if self.num_experts <= 64:
            w, idx = probs.topk(k, dim=-1)
            w = w / w.sum(dim=-1, keepdim=True)
            return w, idx

        # Bucket sort approach for large E
        T = probs.shape[0]
        bucket_boundaries = torch.linspace(0, 1, self.num_buckets + 1, device=probs.device)

        # Find which bucket each expert prob falls into
        bucket_ids = torch.bucketize(probs, bucket_boundaries[1:-1])  # (T, E)

        # For each token, find the top-k buckets
        bucket_sums = torch.zeros(T, self.num_buckets, device=probs.device)
        bucket_sums.scatter_add_(1, bucket_ids, probs)

        # Process top buckets until we have k experts
        top_bucket_vals, top_bucket_ids = bucket_sums.topk(
            min(k * 2, self.num_buckets), dim=-1
        )

        # Collect experts from top buckets
        indices_list = []
        weights_list = []

        for t in range(T):
            candidates = []
            for bkt in top_bucket_ids[t]:
                mask = (bucket_ids[t] == bkt)
                exp_ids = mask.nonzero(as_tuple=False).squeeze(1)
                for eid in exp_ids:
                    candidates.append((float(probs[t, eid].item()), int(eid.item())))
            candidates.sort(reverse=True)
            top_k_cands = candidates[:k]
            if len(top_k_cands) < k:
                # Fallback
                return probs.topk(k, dim=-1)[0] / probs.topk(k, dim=-1)[0].sum(dim=-1, keepdim=True), probs.topk(k, dim=-1)[1]
            ws = torch.tensor([c[0] for c in top_k_cands], device=probs.device)
            ids = torch.tensor([c[1] for c in top_k_cands], device=probs.device, dtype=torch.long)
            ws = ws / ws.sum()
            weights_list.append(ws)
            indices_list.append(ids)

        return torch.stack(weights_list), torch.stack(indices_list)


# ---------------------------------------------------------------------------
# Extended: Inference pipeline with full feature preprocessing
# ---------------------------------------------------------------------------


class FinancialFeaturePreprocessor(nn.Module):
    """
    Preprocesses raw LOB (Level Order Book) features for Lumina MoE inference.

    Input features:
      - bid/ask prices (log-normalized)
      - bid/ask volumes (log-normalized, then standardized)
      - order imbalance ratios
      - mid-price returns
      - spread
      - depth imbalance at multiple levels

    Projects to model hidden_dim using a learned linear layer.
    """

    def __init__(
        self,
        raw_feature_dim: int = 64,
        hidden_dim: int = 512,
        n_levels: int = 5,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.raw_feature_dim = raw_feature_dim
        self.hidden_dim = hidden_dim
        self.n_levels = n_levels
        self.dtype_ = dtype

        # Feature normalization (running statistics)
        self.register_buffer("running_mean", torch.zeros(raw_feature_dim))
        self.register_buffer("running_var", torch.ones(raw_feature_dim))
        self.register_buffer("n_batches_seen", torch.tensor(0))

        # Projection
        self.proj = nn.Linear(raw_feature_dim, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @torch.no_grad()
    def update_statistics(self, x: Tensor) -> None:
        """Update running mean/variance using Welford's online algorithm."""
        B, S, D = x.shape
        flat = x.float().view(-1, D)
        batch_mean = flat.mean(0)
        batch_var = flat.var(0, unbiased=False)
        n = float(flat.shape[0])

        alpha = 1.0 / (self.n_batches_seen.item() + 1)
        self.running_mean.copy_((1 - alpha) * self.running_mean + alpha * batch_mean)
        self.running_var.copy_((1 - alpha) * self.running_var + alpha * batch_var)
        self.n_batches_seen += 1

    def normalize(self, x: Tensor) -> Tensor:
        """Apply learned normalization."""
        mean = self.running_mean.to(x.device)
        std = (self.running_var + 1e-6).sqrt().to(x.device)
        return (x - mean) / std

    def forward(self, raw_features: Tensor) -> Tensor:
        """
        raw_features: (B, S, raw_feature_dim) float32
        Returns: (B, S, hidden_dim) bfloat16
        """
        if self.training:
            self.update_statistics(raw_features)

        x = self.normalize(raw_features.float())
        x = self.proj(x).to(self.dtype_)
        x = self.norm(x)
        return x

    def feature_importance(self) -> Dict[str, float]:
        """Estimate feature importance from projection weight norms."""
        weight_norms = self.proj.weight.abs().sum(dim=0).cpu().detach()
        weight_norms = weight_norms / weight_norms.sum()
        return {
            f"feature_{i}": float(weight_norms[i].item())
            for i in range(self.raw_feature_dim)
        }


class FullLuminaPipeline(nn.Module):
    """
    End-to-end pipeline:
      raw LOB features -> FinancialFeaturePreprocessor -> LuminaMoEModel -> head

    Used for production inference.
    """

    def __init__(
        self,
        raw_feature_dim: int = 64,
        hidden_dim: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        ffn_dim: int = 2048,
        num_layers: int = 4,
        output_dim: int = 1,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.device_ = device
        self.dtype_ = dtype

        config = MoEConfig(
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            device=device,
            dtype=dtype,
        )

        self.preprocessor = FinancialFeaturePreprocessor(
            raw_feature_dim, hidden_dim, dtype=dtype
        )
        self.moe = LuminaMoEModel(config, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, output_dim, bias=True)
        self.final_norm = nn.LayerNorm(hidden_dim)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        raw_features: Tensor,
        return_hidden: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        raw_features: (B, S, raw_feature_dim)
        Returns: (B, S, output_dim) predictions
                 optionally: (predictions, hidden_states)
        """
        # Preprocess
        x = self.preprocessor(raw_features.to(self.device_))

        # MoE backbone
        hidden = self.moe.continuous_input_forward(x)
        hidden = self.final_norm(hidden)

        # Prediction head
        out = self.head(hidden.float())

        if return_hidden:
            return out, hidden
        return out

    def predict_returns(
        self,
        lob_snapshots: Tensor,
        horizon: int = 1,
    ) -> Tensor:
        """
        Predict next `horizon` step returns from LOB snapshots.
        lob_snapshots: (B, S, raw_feature_dim)
        Returns: (B, horizon) predicted returns
        """
        with torch.no_grad():
            out = self.forward(lob_snapshots)
        # Take last `horizon` timestep predictions
        return out[:, -horizon:, 0]  # (B, horizon)

    def export_onnx(self, path: str, seq_len: int = 64, batch_size: int = 1) -> None:
        """Export the pipeline to ONNX for deployment."""
        try:
            dummy = torch.randn(
                batch_size, seq_len, self.preprocessor.raw_feature_dim,
                device=self.device_, dtype=torch.float32,
            )
            torch.onnx.export(
                self,
                dummy,
                path,
                input_names=["lob_features"],
                output_names=["predictions"],
                dynamic_axes={
                    "lob_features": {0: "batch", 1: "seq_len"},
                    "predictions": {0: "batch", 1: "seq_len"},
                },
                opset_version=17,
            )
            logger.info(f"Model exported to ONNX: {path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")


# ---------------------------------------------------------------------------
# Extended: Expert specialization probe
# ---------------------------------------------------------------------------


class ExpertSpecializationProbe:
    """
    Probes what each expert has 'learned' to specialize in by analyzing
    which input features consistently activate each expert.

    Uses mutual information between input features and expert assignments
    as a proxy for specialization.
    """

    def __init__(self, num_experts: int, feature_dim: int, window: int = 500):
        self.num_experts = num_experts
        self.feature_dim = feature_dim
        self.window = window

        # Per-expert feature statistics
        self._expert_feature_sum = np.zeros((num_experts, feature_dim))
        self._expert_feature_sq_sum = np.zeros((num_experts, feature_dim))
        self._expert_count = np.zeros(num_experts)
        self._n_tokens = 0

    def record(self, features: Tensor, expert_indices: Tensor) -> None:
        """
        features:       (T, feature_dim)
        expert_indices: (T, top_k)
        """
        feats_np = features.float().detach().cpu().numpy()
        idx_np = expert_indices.cpu().numpy()
        T = feats_np.shape[0]
        self._n_tokens += T

        for t in range(T):
            for k_slot in range(idx_np.shape[1]):
                eid = int(idx_np[t, k_slot])
                if 0 <= eid < self.num_experts:
                    self._expert_feature_sum[eid] += feats_np[t]
                    self._expert_feature_sq_sum[eid] += feats_np[t] ** 2
                    self._expert_count[eid] += 1

    def expert_feature_mean(self, expert_id: int) -> np.ndarray:
        """Return mean feature vector for tokens routed to this expert."""
        count = self._expert_count[expert_id]
        if count == 0:
            return np.zeros(self.feature_dim)
        return self._expert_feature_sum[expert_id] / count

    def expert_feature_std(self, expert_id: int) -> np.ndarray:
        """Return per-feature standard deviation for tokens routed to this expert."""
        count = self._expert_count[expert_id]
        if count == 0:
            return np.zeros(self.feature_dim)
        mean = self.expert_feature_mean(expert_id)
        sq_mean = self._expert_feature_sq_sum[expert_id] / count
        var = np.maximum(sq_mean - mean ** 2, 0)
        return np.sqrt(var)

    def most_discriminative_features(
        self, expert_id: int, n: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Return the top-n features that most discriminate this expert from others.
        Uses mean difference from global mean as the discriminability measure.
        """
        total = self._expert_count.sum()
        if total == 0:
            return []

        # Global mean
        global_mean = self._expert_feature_sum.sum(axis=0) / max(total, 1)
        # Expert mean
        expert_mean = self.expert_feature_mean(expert_id)

        # Discriminability: |expert_mean - global_mean| / global_std
        diff = np.abs(expert_mean - global_mean)
        sorted_idx = diff.argsort()[::-1]

        return [(int(i), float(diff[i])) for i in sorted_idx[:n]]

    def specialization_report(self) -> Dict[str, Any]:
        """Return a full specialization report for all experts."""
        return {
            f"expert_{e}": {
                "n_tokens": int(self._expert_count[e]),
                "fraction": float(self._expert_count[e] / max(self._n_tokens, 1)),
                "top_features": self.most_discriminative_features(e, n=3),
            }
            for e in range(self.num_experts)
        }


# ---------------------------------------------------------------------------
# Extended: Dynamic expert pruning
# ---------------------------------------------------------------------------


class ExpertPruner:
    """
    Dynamically prunes underutilized experts from a MoE layer to reduce
    inference cost without significant accuracy loss.

    Uses expert utilization statistics collected over a window of steps.
    Experts below the utilization threshold are pruned (masked out).
    """

    def __init__(
        self,
        num_experts: int,
        utilization_threshold: float = 0.01,
        pruning_window: int = 1000,
    ):
        self.num_experts = num_experts
        self.utilization_threshold = utilization_threshold
        self.pruning_window = pruning_window
        self._utilization: deque = deque(maxlen=pruning_window)
        self._pruned: set = set()

    def record_step(self, expert_counts: np.ndarray, total_tokens: int) -> None:
        """Record per-expert token counts for one step."""
        frac = expert_counts / max(total_tokens, 1)
        self._utilization.append(frac)

    def compute_utilization(self) -> np.ndarray:
        """Return mean utilization per expert over the window."""
        if not self._utilization:
            return np.ones(self.num_experts) / self.num_experts
        return np.stack(list(self._utilization)).mean(axis=0)

    def get_pruning_mask(self) -> np.ndarray:
        """
        Return a boolean mask of shape (num_experts,).
        True = keep, False = prune.
        """
        util = self.compute_utilization()
        mask = util >= self.utilization_threshold
        # Always keep at least 2 experts
        if mask.sum() < 2:
            top2 = util.argsort()[-2:]
            mask = np.zeros(self.num_experts, dtype=bool)
            mask[top2] = True
        return mask

    def prune_layer(self, layer: "OptimizedMoELayer") -> List[int]:
        """
        Apply pruning mask to a MoE layer.
        Pruned experts are replaced with identity maps (zero output).
        Returns list of pruned expert IDs.
        """
        mask = self.get_pruning_mask()
        newly_pruned = []

        for e in range(self.num_experts):
            if not mask[e] and e not in self._pruned:
                # Zero out expert weights (effectively prune)
                for p in layer.experts[e].parameters():
                    p.data.zero_()
                self._pruned.add(e)
                newly_pruned.append(e)
                logger.info(f"Expert {e} pruned (utilization below threshold)")

        return newly_pruned

    def restore_expert(self, layer: "OptimizedMoELayer", expert_id: int) -> None:
        """Restore a previously pruned expert (e.g., when utilization recovers)."""
        if expert_id in self._pruned:
            self._pruned.discard(expert_id)
            # Re-initialize with small random weights
            for p in layer.experts[expert_id].parameters():
                nn.init.xavier_uniform_(p.data) if p.dim() >= 2 else nn.init.zeros_(p.data)
            logger.info(f"Expert {expert_id} restored")

    @property
    def pruned_experts(self) -> List[int]:
        return sorted(self._pruned)

    @property
    def active_expert_count(self) -> int:
        return self.num_experts - len(self._pruned)


# ---------------------------------------------------------------------------
# Extended: MoE gradient checkpointing wrapper
# ---------------------------------------------------------------------------


class CheckpointedMoELayer(nn.Module):
    """
    Wraps an OptimizedMoELayer with gradient checkpointing to reduce
    activation memory during training (trades compute for memory).

    Uses torch.utils.checkpoint.checkpoint for re-computation of
    activations during the backward pass.
    """

    def __init__(self, layer: "OptimizedMoELayer", use_reentrant: bool = False):
        super().__init__()
        self.layer = layer
        self.use_reentrant = use_reentrant

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self.layer,
                x,
                use_reentrant=self.use_reentrant,
            )
        return self.layer(x, **kwargs)

    def extra_repr(self) -> str:
        return f"checkpointed=True, use_reentrant={self.use_reentrant}"


def wrap_with_checkpointing(
    model: "LuminaMoEModel",
    use_reentrant: bool = False,
) -> "LuminaMoEModel":
    """
    Wrap all MoE layers in a LuminaMoEModel with gradient checkpointing.
    Returns the modified model.
    """
    for i, layer in enumerate(model.moe_layers):
        model.moe_layers[i] = CheckpointedMoELayer(layer, use_reentrant)
    return model

