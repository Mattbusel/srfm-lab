

# ============================================================
# Extended MoE Components
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
import torch.distributed as dist


class ExpertCapacity:
    """Computes per-expert capacity for load-balanced dispatch."""

    @staticmethod
    def compute(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity: int = 1) -> int:
        capacity = math.ceil(capacity_factor * num_tokens / num_experts)
        return max(capacity, min_capacity)


class RouterOutput(NamedTuple):
    dispatch_mask: torch.Tensor      # (B*T, E, C) one-hot
    combine_weights: torch.Tensor    # (B*T, E, C) float
    router_probs: torch.Tensor       # (B*T, E)
    load_loss: torch.Tensor          # scalar


class TopKRouter(nn.Module):
    """Top-K router with auxiliary load balancing loss (Switch/GShard style)."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        noisy_gate_policy: str = "rsample",  # rsample | none
        aux_loss_weight: float = 1e-2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.noisy_gate_policy = noisy_gate_policy
        self.aux_loss_weight = aux_loss_weight

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01)

        if noisy_gate_policy == "rsample":
            self.noise_gate = nn.Linear(d_model, num_experts, bias=False)
            nn.init.normal_(self.noise_gate.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """x: (N, D) where N = B*T"""
        N, D = x.shape
        E = self.num_experts
        C = ExpertCapacity.compute(N, E, self.capacity_factor)

        logits = self.gate(x)  # (N, E)

        if self.training and self.noisy_gate_policy == "rsample":
            noise_std = F.softplus(self.noise_gate(x))
            noise = torch.randn_like(logits) * noise_std
            logits = logits + noise

        router_probs = F.softmax(logits, dim=-1)  # (N, E)

        # Top-k selection
        topk_vals, topk_idx = router_probs.topk(self.top_k, dim=-1)  # (N, k)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # renormalize

        # Build dispatch and combine tensors
        dispatch_mask = torch.zeros(N, E, C, device=x.device, dtype=torch.bool)
        combine_weights = torch.zeros(N, E, C, device=x.device, dtype=x.dtype)

        # Assign tokens to experts with capacity constraint
        expert_counts = torch.zeros(E, dtype=torch.long, device=x.device)
        for token_idx in range(N):
            for k_idx in range(self.top_k):
                e = topk_idx[token_idx, k_idx].item()
                c = expert_counts[e].item()
                if c < C:
                    dispatch_mask[token_idx, e, c] = True
                    combine_weights[token_idx, e, c] = topk_vals[token_idx, k_idx]
                    expert_counts[e] += 1

        # Auxiliary load balancing loss
        # L_aux = alpha * E * sum_e(f_e * p_e)
        f = (topk_idx == torch.arange(E, device=x.device).unsqueeze(0).unsqueeze(0)).float().mean(0).mean(0)
        p = router_probs.mean(0)
        load_loss = self.aux_loss_weight * E * (f * p).sum()

        return RouterOutput(dispatch_mask, combine_weights, router_probs, load_loss)


class ExpertFFN(nn.Module):
    """Single expert FFN (feed-forward network)."""

    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu", dropout: float = 0.0):
        super().__init__()
        act_fn = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }.get(activation, nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMoELayer(nn.Module):
    """Sparse Mixture-of-Experts layer with top-K routing and capacity constraints."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        capacity_factor: float = 1.25,
        activation: str = "gelu",
        dropout: float = 0.0,
        aux_loss_weight: float = 1e-2,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_experts, top_k, capacity_factor, aux_loss_weight=aux_loss_weight)
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff, activation, dropout) for _ in range(num_experts)
        ])
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)

        router_out = self.router(x_flat)
        self.aux_loss = router_out.load_loss

        dispatch = router_out.dispatch_mask   # (N, E, C)
        combine = router_out.combine_weights  # (N, E, C)
        C = dispatch.shape[-1]

        output = torch.zeros_like(x_flat)

        for e, expert in enumerate(self.experts):
            # Gather tokens dispatched to expert e
            # dispatch[:, e, :] -> (N, C) bool
            token_indices = dispatch[:, e, :].any(dim=-1).nonzero(as_tuple=True)[0]  # (K,)
            if token_indices.numel() == 0:
                continue
            expert_in = x_flat[token_indices]  # (K, D)
            expert_out = expert(expert_in)       # (K, D)
            # Accumulate with combine weights
            for slot in range(C):
                dispatched = dispatch[token_indices, e, slot]  # (K,)
                weight = combine[token_indices, e, slot]        # (K,)
                output[token_indices] += dispatched.float().unsqueeze(-1) * weight.unsqueeze(-1) * expert_out

        return output.reshape(B, T, D)


class FusedMoELayer(nn.Module):
    """Fused MoE using batched matrix multiply for efficiency (no Python loop over experts)."""

    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ff = d_ff

        # All expert weights stacked
        self.w1 = nn.Parameter(torch.empty(num_experts, d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_model, d_ff))
        self.router = nn.Linear(d_model, num_experts, bias=False)

        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        logits = self.router(x_flat)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            idx = indices[:, k]     # (N,)
            w = weights[:, k]       # (N,)
            # Gather expert weights per token
            w1_k = self.w1[idx]     # (N, d_ff, D)
            w2_k = self.w2[idx]     # (N, D, d_ff)
            h = torch.bmm(w1_k, x_flat.unsqueeze(-1)).squeeze(-1)  # (N, d_ff)
            h = F.gelu(h)
            out_k = torch.bmm(w2_k, h.unsqueeze(-1)).squeeze(-1)   # (N, D)
            output = output + w.unsqueeze(-1) * out_k

        return output.reshape(B, T, D)


class ExpertChoiceLayer(nn.Module):
    """Expert-Choice MoE (Zhou et al. 2022): each expert selects top-k tokens.

    Guarantees perfect load balance; each expert processes exactly C tokens.
    """

    def __init__(self, d_model: int, num_experts: int = 8, expert_capacity: int = 32, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)
        C = min(self.expert_capacity, N)

        # Router: (N, E) -> each expert picks top-C tokens
        logits = self.router(x_flat)  # (N, E)
        probs = F.softmax(logits, dim=0)  # softmax over tokens dimension for expert-choice

        # Expert i selects top-C tokens by its column
        output = torch.zeros_like(x_flat)
        for e, expert in enumerate(self.experts):
            expert_probs = probs[:, e]            # (N,)
            topC_vals, topC_idx = expert_probs.topk(C)
            selected = x_flat[topC_idx]           # (C, D)
            expert_out = expert(selected)          # (C, D)
            output.scatter_add_(0, topC_idx.unsqueeze(-1).expand(-1, D),
                                topC_vals.unsqueeze(-1) * expert_out)

        return output.reshape(B, T, D)


class SwitchTransformerLayer(nn.Module):
    """Switch Transformer FFN layer (Fedus et al. 2021) with top-1 routing."""

    def __init__(self, d_model: int, num_experts: int = 8, d_ff: Optional[int] = None,
                 capacity_factor: float = 1.5, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff, dropout=dropout) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)
        C = ExpertCapacity.compute(N, self.num_experts, self.capacity_factor)

        logits = self.router(x_flat)              # (N, E)
        probs = F.softmax(logits, dim=-1)          # (N, E)
        chosen_e = probs.argmax(dim=-1)            # (N,)
        chosen_p = probs.max(dim=-1).values        # (N,)

        # Aux loss
        f = F.one_hot(chosen_e, self.num_experts).float().mean(0)
        p = probs.mean(0)
        self.aux_loss = self.num_experts * (f * p).sum()

        output = torch.zeros_like(x_flat)
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)

        for i in range(N):
            e = chosen_e[i].item()
            if counts[e] < C:
                expert_out = self.experts[e](x_flat[i:i+1])
                output[i] = chosen_p[i] * expert_out.squeeze(0)
                counts[e] += 1

        return output.reshape(B, T, D)


class MoETransformerBlock(nn.Module):
    """Transformer block with every-N-th FFN replaced by MoE layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        moe_layer: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        if moe_layer:
            self.ffn = SparseMoELayer(d_model, num_experts, top_k, d_ff)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
            )
        self.moe_layer = moe_layer

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)
        # FFN / MoE
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

    @property
    def aux_loss(self) -> torch.Tensor:
        if self.moe_layer and hasattr(self.ffn, "aux_loss"):
            return self.ffn.aux_loss
        return torch.tensor(0.0)


class MoELanguageModel(nn.Module):
    """Simple MoE language model with alternating dense/MoE FFN layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        num_experts: int = 8,
        top_k: int = 2,
        moe_every_n: int = 2,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            MoETransformerBlock(
                d_model, num_heads, num_experts, top_k,
                moe_layer=(i % moe_every_n == (moe_every_n - 1)),
                dropout=dropout,
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.dropout(self.embed(input_ids) + self.pos_embed(pos))
        aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x = layer(x)
            if hasattr(layer, "aux_loss"):
                aux_loss = aux_loss + layer.aux_loss
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, aux_loss


class DistributedMoE(nn.Module):
    """Distributed MoE where each rank holds a subset of experts (Expert Parallelism)."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.world_size = world_size
        self.rank = rank

        assert num_experts % world_size == 0
        self.local_num_experts = num_experts // world_size

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.local_experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff) for _ in range(self.local_num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified: run local experts only (in real usage, would all_to_all)
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        local_start = self.rank * self.local_num_experts
        local_end = local_start + self.local_num_experts

        for k in range(self.top_k):
            idx = topk_idx[:, k]
            w = topk_vals[:, k]
            # Filter for local experts
            local_mask = (idx >= local_start) & (idx < local_end)
            if local_mask.any():
                local_idx = idx[local_mask] - local_start
                x_sel = x_flat[local_mask]
                for le in range(self.local_num_experts):
                    le_mask = local_idx == le
                    if le_mask.any():
                        out = self.local_experts[le](x_sel[le_mask])
                        tok_idx = local_mask.nonzero(as_tuple=True)[0][le_mask]
                        output[tok_idx] += w[tok_idx].unsqueeze(-1) * out

        return output.reshape(B, T, D)


class MoELoadBalancer:
    """Monitors and adjusts expert load balance during training."""

    def __init__(self, num_experts: int, ema_decay: float = 0.99):
        self.num_experts = num_experts
        self.ema_decay = ema_decay
        self.expert_counts = torch.zeros(num_experts)
        self.total_tokens = 0

    def update(self, dispatch_mask: torch.Tensor):
        """dispatch_mask: (N, E) binary"""
        counts = dispatch_mask.float().sum(0).detach().cpu()
        self.expert_counts = self.ema_decay * self.expert_counts + (1 - self.ema_decay) * counts
        self.total_tokens += dispatch_mask.shape[0]

    def load_imbalance(self) -> float:
        """Returns coefficient of variation of expert loads."""
        mean = self.expert_counts.mean()
        std = self.expert_counts.std()
        return (std / (mean + 1e-8)).item()

    def underutilized_experts(self, threshold: float = 0.5) -> List[int]:
        mean = self.expert_counts.mean()
        return [i for i, c in enumerate(self.expert_counts) if c < threshold * mean]

    def report(self) -> Dict[str, Any]:
        return {
            "expert_counts": self.expert_counts.tolist(),
            "load_imbalance_cv": self.load_imbalance(),
            "underutilized": self.underutilized_experts(),
        }


# ============================================================
# Extended MoE Components
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
import torch.distributed as dist


class ExpertCapacity:
    """Computes per-expert capacity for load-balanced dispatch."""

    @staticmethod
    def compute(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity: int = 1) -> int:
        capacity = math.ceil(capacity_factor * num_tokens / num_experts)
        return max(capacity, min_capacity)


class RouterOutput(NamedTuple):
    dispatch_mask: torch.Tensor      # (B*T, E, C) one-hot
    combine_weights: torch.Tensor    # (B*T, E, C) float
    router_probs: torch.Tensor       # (B*T, E)
    load_loss: torch.Tensor          # scalar


class TopKRouter(nn.Module):
    """Top-K router with auxiliary load balancing loss (Switch/GShard style)."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        noisy_gate_policy: str = "rsample",  # rsample | none
        aux_loss_weight: float = 1e-2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.noisy_gate_policy = noisy_gate_policy
        self.aux_loss_weight = aux_loss_weight

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01)

        if noisy_gate_policy == "rsample":
            self.noise_gate = nn.Linear(d_model, num_experts, bias=False)
            nn.init.normal_(self.noise_gate.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """x: (N, D) where N = B*T"""
        N, D = x.shape
        E = self.num_experts
        C = ExpertCapacity.compute(N, E, self.capacity_factor)

        logits = self.gate(x)  # (N, E)

        if self.training and self.noisy_gate_policy == "rsample":
            noise_std = F.softplus(self.noise_gate(x))
            noise = torch.randn_like(logits) * noise_std
            logits = logits + noise

        router_probs = F.softmax(logits, dim=-1)  # (N, E)

        # Top-k selection
        topk_vals, topk_idx = router_probs.topk(self.top_k, dim=-1)  # (N, k)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # renormalize

        # Build dispatch and combine tensors
        dispatch_mask = torch.zeros(N, E, C, device=x.device, dtype=torch.bool)
        combine_weights = torch.zeros(N, E, C, device=x.device, dtype=x.dtype)

        # Assign tokens to experts with capacity constraint
        expert_counts = torch.zeros(E, dtype=torch.long, device=x.device)
        for token_idx in range(N):
            for k_idx in range(self.top_k):
                e = topk_idx[token_idx, k_idx].item()
                c = expert_counts[e].item()
                if c < C:
                    dispatch_mask[token_idx, e, c] = True
                    combine_weights[token_idx, e, c] = topk_vals[token_idx, k_idx]
                    expert_counts[e] += 1

        # Auxiliary load balancing loss
        # L_aux = alpha * E * sum_e(f_e * p_e)
        f = (topk_idx == torch.arange(E, device=x.device).unsqueeze(0).unsqueeze(0)).float().mean(0).mean(0)
        p = router_probs.mean(0)
        load_loss = self.aux_loss_weight * E * (f * p).sum()

        return RouterOutput(dispatch_mask, combine_weights, router_probs, load_loss)


class ExpertFFN(nn.Module):
    """Single expert FFN (feed-forward network)."""

    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu", dropout: float = 0.0):
        super().__init__()
        act_fn = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }.get(activation, nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMoELayer(nn.Module):
    """Sparse Mixture-of-Experts layer with top-K routing and capacity constraints."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        capacity_factor: float = 1.25,
        activation: str = "gelu",
        dropout: float = 0.0,
        aux_loss_weight: float = 1e-2,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_experts, top_k, capacity_factor, aux_loss_weight=aux_loss_weight)
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff, activation, dropout) for _ in range(num_experts)
        ])
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)

        router_out = self.router(x_flat)
        self.aux_loss = router_out.load_loss

        dispatch = router_out.dispatch_mask   # (N, E, C)
        combine = router_out.combine_weights  # (N, E, C)
        C = dispatch.shape[-1]

        output = torch.zeros_like(x_flat)

        for e, expert in enumerate(self.experts):
            # Gather tokens dispatched to expert e
            # dispatch[:, e, :] -> (N, C) bool
            token_indices = dispatch[:, e, :].any(dim=-1).nonzero(as_tuple=True)[0]  # (K,)
            if token_indices.numel() == 0:
                continue
            expert_in = x_flat[token_indices]  # (K, D)
            expert_out = expert(expert_in)       # (K, D)
            # Accumulate with combine weights
            for slot in range(C):
                dispatched = dispatch[token_indices, e, slot]  # (K,)
                weight = combine[token_indices, e, slot]        # (K,)
                output[token_indices] += dispatched.float().unsqueeze(-1) * weight.unsqueeze(-1) * expert_out

        return output.reshape(B, T, D)


class FusedMoELayer(nn.Module):
    """Fused MoE using batched matrix multiply for efficiency (no Python loop over experts)."""

    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ff = d_ff

        # All expert weights stacked
        self.w1 = nn.Parameter(torch.empty(num_experts, d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_model, d_ff))
        self.router = nn.Linear(d_model, num_experts, bias=False)

        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        logits = self.router(x_flat)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            idx = indices[:, k]     # (N,)
            w = weights[:, k]       # (N,)
            # Gather expert weights per token
            w1_k = self.w1[idx]     # (N, d_ff, D)
            w2_k = self.w2[idx]     # (N, D, d_ff)
            h = torch.bmm(w1_k, x_flat.unsqueeze(-1)).squeeze(-1)  # (N, d_ff)
            h = F.gelu(h)
            out_k = torch.bmm(w2_k, h.unsqueeze(-1)).squeeze(-1)   # (N, D)
            output = output + w.unsqueeze(-1) * out_k

        return output.reshape(B, T, D)


class ExpertChoiceLayer(nn.Module):
    """Expert-Choice MoE (Zhou et al. 2022): each expert selects top-k tokens.

    Guarantees perfect load balance; each expert processes exactly C tokens.
    """

    def __init__(self, d_model: int, num_experts: int = 8, expert_capacity: int = 32, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)
        C = min(self.expert_capacity, N)

        # Router: (N, E) -> each expert picks top-C tokens
        logits = self.router(x_flat)  # (N, E)
        probs = F.softmax(logits, dim=0)  # softmax over tokens dimension for expert-choice

        # Expert i selects top-C tokens by its column
        output = torch.zeros_like(x_flat)
        for e, expert in enumerate(self.experts):
            expert_probs = probs[:, e]            # (N,)
            topC_vals, topC_idx = expert_probs.topk(C)
            selected = x_flat[topC_idx]           # (C, D)
            expert_out = expert(selected)          # (C, D)
            output.scatter_add_(0, topC_idx.unsqueeze(-1).expand(-1, D),
                                topC_vals.unsqueeze(-1) * expert_out)

        return output.reshape(B, T, D)


class SwitchTransformerLayer(nn.Module):
    """Switch Transformer FFN layer (Fedus et al. 2021) with top-1 routing."""

    def __init__(self, d_model: int, num_experts: int = 8, d_ff: Optional[int] = None,
                 capacity_factor: float = 1.5, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff, dropout=dropout) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)
        C = ExpertCapacity.compute(N, self.num_experts, self.capacity_factor)

        logits = self.router(x_flat)              # (N, E)
        probs = F.softmax(logits, dim=-1)          # (N, E)
        chosen_e = probs.argmax(dim=-1)            # (N,)
        chosen_p = probs.max(dim=-1).values        # (N,)

        # Aux loss
        f = F.one_hot(chosen_e, self.num_experts).float().mean(0)
        p = probs.mean(0)
        self.aux_loss = self.num_experts * (f * p).sum()

        output = torch.zeros_like(x_flat)
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)

        for i in range(N):
            e = chosen_e[i].item()
            if counts[e] < C:
                expert_out = self.experts[e](x_flat[i:i+1])
                output[i] = chosen_p[i] * expert_out.squeeze(0)
                counts[e] += 1

        return output.reshape(B, T, D)


class MoETransformerBlock(nn.Module):
    """Transformer block with every-N-th FFN replaced by MoE layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        moe_layer: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        if moe_layer:
            self.ffn = SparseMoELayer(d_model, num_experts, top_k, d_ff)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
            )
        self.moe_layer = moe_layer

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)
        # FFN / MoE
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

    @property
    def aux_loss(self) -> torch.Tensor:
        if self.moe_layer and hasattr(self.ffn, "aux_loss"):
            return self.ffn.aux_loss
        return torch.tensor(0.0)


class MoELanguageModel(nn.Module):
    """Simple MoE language model with alternating dense/MoE FFN layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        num_experts: int = 8,
        top_k: int = 2,
        moe_every_n: int = 2,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            MoETransformerBlock(
                d_model, num_heads, num_experts, top_k,
                moe_layer=(i % moe_every_n == (moe_every_n - 1)),
                dropout=dropout,
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.dropout(self.embed(input_ids) + self.pos_embed(pos))
        aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x = layer(x)
            if hasattr(layer, "aux_loss"):
                aux_loss = aux_loss + layer.aux_loss
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, aux_loss


class DistributedMoE(nn.Module):
    """Distributed MoE where each rank holds a subset of experts (Expert Parallelism)."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.world_size = world_size
        self.rank = rank

        assert num_experts % world_size == 0
        self.local_num_experts = num_experts // world_size

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.local_experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff) for _ in range(self.local_num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified: run local experts only (in real usage, would all_to_all)
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        local_start = self.rank * self.local_num_experts
        local_end = local_start + self.local_num_experts

        for k in range(self.top_k):
            idx = topk_idx[:, k]
            w = topk_vals[:, k]
            # Filter for local experts
            local_mask = (idx >= local_start) & (idx < local_end)
            if local_mask.any():
                local_idx = idx[local_mask] - local_start
                x_sel = x_flat[local_mask]
                for le in range(self.local_num_experts):
                    le_mask = local_idx == le
                    if le_mask.any():
                        out = self.local_experts[le](x_sel[le_mask])
                        tok_idx = local_mask.nonzero(as_tuple=True)[0][le_mask]
                        output[tok_idx] += w[tok_idx].unsqueeze(-1) * out

        return output.reshape(B, T, D)


class MoELoadBalancer:
    """Monitors and adjusts expert load balance during training."""

    def __init__(self, num_experts: int, ema_decay: float = 0.99):
        self.num_experts = num_experts
        self.ema_decay = ema_decay
        self.expert_counts = torch.zeros(num_experts)
        self.total_tokens = 0

    def update(self, dispatch_mask: torch.Tensor):
        """dispatch_mask: (N, E) binary"""
        counts = dispatch_mask.float().sum(0).detach().cpu()
        self.expert_counts = self.ema_decay * self.expert_counts + (1 - self.ema_decay) * counts
        self.total_tokens += dispatch_mask.shape[0]

    def load_imbalance(self) -> float:
        """Returns coefficient of variation of expert loads."""
        mean = self.expert_counts.mean()
        std = self.expert_counts.std()
        return (std / (mean + 1e-8)).item()

    def underutilized_experts(self, threshold: float = 0.5) -> List[int]:
        mean = self.expert_counts.mean()
        return [i for i, c in enumerate(self.expert_counts) if c < threshold * mean]

    def report(self) -> Dict[str, Any]:
        return {
            "expert_counts": self.expert_counts.tolist(),
            "load_imbalance_cv": self.load_imbalance(),
            "underutilized": self.underutilized_experts(),
        }
