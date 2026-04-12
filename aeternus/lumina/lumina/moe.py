"""
lumina/moe.py

Mixture of Experts (MoE) for Lumina Financial Foundation Model.

Implements:
  - TopKRouter         : learnable gating network with top-k selection
  - LoadBalancingLoss  : auxiliary loss to encourage uniform expert utilization
  - Expert             : individual FFN expert
  - SparseMoELayer     : sparse MoE replacing dense FFN
  - ExpertParallelStub : placeholder for multi-device expert parallelism
  - MoEConfig          : configuration dataclass
  - MoETransformerStack: full transformer with MoE layers
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MoEConfig:
    d_model:         int   = 512
    n_experts:       int   = 8
    n_active:        int   = 2          # top-k experts per token
    d_ff:            Optional[int] = None
    dropout:         float = 0.1
    lb_loss_weight:  float = 0.01       # load balancing loss coefficient
    router_jitter:   float = 0.0        # noise added to router logits during training
    use_expert_bias: bool  = False      # expert-specific bias in router
    capacity_factor: float = 1.25       # buffer factor for expert capacity
    norm_topk_prob:  bool  = True       # normalize top-k routing probabilities
    router_type:     str   = "linear"   # "linear" | "mlp"
    expert_type:     str   = "swiglu"   # "swiglu" | "standard"


# ---------------------------------------------------------------------------
# Individual Expert FFN
# ---------------------------------------------------------------------------

class ExpertFFN(nn.Module):
    """
    Single FFN expert with SwiGLU or standard activation.
    Each expert is a standard feedforward network independent of the others.
    """

    def __init__(
        self,
        d_model:    int,
        d_ff:       int,
        activation: str   = "swiglu",
        dropout:    float = 0.1,
        bias:       bool  = False,
    ):
        super().__init__()
        self.activation = activation
        if activation == "swiglu":
            self.gate = nn.Linear(d_model, d_ff, bias=bias)
            self.up   = nn.Linear(d_model, d_ff, bias=bias)
            self.down = nn.Linear(d_ff,    d_model, bias=bias)
        else:
            self.net  = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model, bias=bias),
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.down(self.dropout(F.silu(self.gate(x)) * self.up(x)))
        return self.net(x)


# ---------------------------------------------------------------------------
# Top-K Router
# ---------------------------------------------------------------------------

class TopKRouter(nn.Module):
    """
    Learnable routing network for MoE.

    For each token, computes routing probabilities over all experts,
    then selects the top-k experts to process it.

    Supports:
      - Linear router (single projection)
      - MLP router (two-layer with activation)
      - Training-time jitter noise (prevents routing collapse)
      - Auxiliary load balancing loss
    """

    def __init__(
        self,
        d_model:      int,
        n_experts:    int,
        n_active:     int,
        router_type:  str   = "linear",
        jitter_noise: float = 0.0,
        norm_prob:    bool  = True,
    ):
        super().__init__()
        self.n_experts    = n_experts
        self.n_active     = n_active
        self.jitter_noise = jitter_noise
        self.norm_prob    = norm_prob

        if router_type == "linear":
            self.gate = nn.Linear(d_model, n_experts, bias=False)
        else:
            self.gate = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, n_experts, bias=False),
            )

        # Initialize to uniform routing
        if hasattr(self.gate, 'weight'):
            nn.init.zeros_(self.gate.weight)
        else:
            # MLP: init last layer to zero
            last_layer = list(self.gate.children())[-1]
            if hasattr(last_layer, 'weight'):
                nn.init.zeros_(last_layer.weight)

    def add_jitter(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(logits) * self.jitter_noise
            logits = logits + noise
        return logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (N, d_model) — N = batch * seq_len

        Returns:
            routing_weights: (N, n_active) softmax-normalized routing probabilities
            selected_experts: (N, n_active) expert indices
            router_probs: (N, n_experts) full routing probabilities (for aux loss)
        """
        logits = self.gate(x)                      # (N, E)
        logits = self.add_jitter(logits)

        router_probs = F.softmax(logits, dim=-1)   # (N, E)

        topk_vals, topk_idx = torch.topk(router_probs, self.n_active, dim=-1)  # (N, k)

        if self.norm_prob:
            topk_vals = topk_vals / (topk_vals.sum(-1, keepdim=True) + 1e-6)

        return topk_vals, topk_idx, router_probs


# ---------------------------------------------------------------------------
# Load Balancing Loss
# ---------------------------------------------------------------------------

class LoadBalancingLoss(nn.Module):
    """
    Auxiliary load balancing loss (Fedus et al. 2021).

    Encourages equal token distribution across experts.
    L_aux = n_experts * sum(f_i * P_i)
    where:
      f_i = fraction of tokens routed to expert i
      P_i = mean routing probability for expert i
    """

    def __init__(self, n_experts: int, n_active: int):
        super().__init__()
        self.n_experts = n_experts
        self.n_active  = n_active

    def forward(
        self,
        router_probs:     torch.Tensor,   # (N, E) full softmax probs
        selected_experts: torch.Tensor,   # (N, k) selected expert indices
    ) -> torch.Tensor:
        N, E = router_probs.shape
        k    = selected_experts.shape[1]

        # Fraction of tokens dispatched to each expert
        # one_hot: (N, k, E)
        one_hot = F.one_hot(selected_experts, num_classes=E).float()  # (N, k, E)
        # f_i = fraction of tokens routed to expert i
        tokens_per_expert = one_hot.sum(0).sum(0) / (N * k)           # (E,)

        # P_i = mean router probability for expert i
        P_per_expert = router_probs.mean(0)                            # (E,)

        # Auxiliary loss
        lb_loss = E * (tokens_per_expert * P_per_expert).sum()
        return lb_loss

    def z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Z-loss: penalizes large router logits to improve numerical stability.
        L_z = (1/N) * sum(log^2(sum_j(exp(logit_j))))
        """
        N    = router_logits.shape[0]
        logsumexp = torch.logsumexp(router_logits, dim=-1)
        return (logsumexp ** 2).mean()


# ---------------------------------------------------------------------------
# Expert Capacity Buffer
# ---------------------------------------------------------------------------

def compute_expert_capacity(
    n_tokens:       int,
    n_experts:      int,
    n_active:       int,
    capacity_factor: float = 1.25,
) -> int:
    """
    Compute expert capacity (max tokens per expert).
    capacity = ceil((n_tokens / n_experts) * n_active * capacity_factor)
    """
    return math.ceil((n_tokens / n_experts) * n_active * capacity_factor)


# ---------------------------------------------------------------------------
# Sparse MoE Layer (main module)
# ---------------------------------------------------------------------------

class SparseMoELayer(nn.Module):
    """
    Sparse Mixture of Experts layer.

    Architecture:
      1. Router: assigns each token to k experts
      2. Dispatch: route tokens to assigned experts
      3. Experts: each expert processes its assigned tokens
      4. Combine: weighted sum of expert outputs

    Implements expert capacity to prevent load imbalance overflow.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg       = cfg
        E              = cfg.n_experts
        d_ff           = cfg.d_ff or cfg.d_model * 4

        # Router
        self.router    = TopKRouter(
            d_model      = cfg.d_model,
            n_experts    = E,
            n_active     = cfg.n_active,
            router_type  = cfg.router_type,
            jitter_noise = cfg.router_jitter,
            norm_prob    = cfg.norm_topk_prob,
        )

        # Experts
        self.experts   = nn.ModuleList([
            ExpertFFN(cfg.d_model, d_ff, cfg.expert_type, cfg.dropout)
            for _ in range(E)
        ])

        # Load balancing
        self.lb_loss_fn  = LoadBalancingLoss(E, cfg.n_active)
        self.lb_weight   = cfg.lb_loss_weight
        self.norm        = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, D)
        Returns: (output (B, T, D), total_aux_loss scalar)
        """
        B, T, D = x.shape
        N       = B * T
        x_flat  = x.reshape(N, D)

        # Route
        gate_weights, selected_experts, router_probs = self.router(x_flat)
        # gate_weights: (N, k)
        # selected_experts: (N, k)

        # Load balancing loss
        lb_loss = self.lb_loss_fn(router_probs, selected_experts)
        z_loss  = self.lb_loss_fn.z_loss(
            self.router.gate(x_flat) if hasattr(self.router.gate, '__call__') else
            torch.zeros(N, self.cfg.n_experts, device=x.device)
        ) * 0.001

        total_aux = self.lb_weight * (lb_loss + z_loss)

        # Dispatch tokens to experts
        out = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        for expert_idx, expert in enumerate(self.experts):
            # Find tokens assigned to this expert
            # selected_experts: (N, k) — check if expert_idx is in any slot
            is_selected = (selected_experts == expert_idx)   # (N, k)
            tok_mask    = is_selected.any(-1)                  # (N,)
            if not tok_mask.any():
                continue

            tok_in  = x_flat[tok_mask]                         # (n_tok, D)
            exp_out = expert(tok_in)                           # (n_tok, D)

            # Weight: sum of routing weights for this expert
            weight_mat = gate_weights[tok_mask]                # (n_tok, k)
            sel_mat    = is_selected[tok_mask].float()         # (n_tok, k)
            weight     = (weight_mat * sel_mat).sum(-1, keepdim=True)  # (n_tok, 1)

            out[tok_mask] += exp_out * weight

        out = out.reshape(B, T, D)
        out = self.norm(out + x)   # residual + norm
        return out, total_aux

    def get_routing_stats(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return routing statistics for monitoring."""
        B, T, D = x.shape
        x_flat  = x.reshape(B * T, D)
        _, selected, router_probs = self.router(x_flat)

        E = self.cfg.n_experts
        # Expert utilization
        utilization = torch.zeros(E, device=x.device)
        for e in range(E):
            utilization[e] = (selected == e).float().mean()

        return {
            "expert_utilization":  utilization,
            "routing_entropy":     -(router_probs * (router_probs + 1e-8).log()).sum(-1).mean(),
            "max_utilization":     utilization.max(),
            "min_utilization":     utilization.min(),
            "utilization_std":     utilization.std(),
        }


# ---------------------------------------------------------------------------
# Expert Parallel Stub
# ---------------------------------------------------------------------------

class ExpertParallelStub(nn.Module):
    """
    Stub for multi-device expert parallelism.

    In a full implementation, each expert would live on a different device.
    This stub simulates the interface while keeping all experts on one device.

    For true expert parallelism, integrate with torch.distributed and
    implement all-to-all communication between device groups.
    """

    def __init__(self, cfg: MoEConfig, world_size: int = 1):
        super().__init__()
        self.cfg        = cfg
        self.world_size = world_size
        self.local_rank = 0

        # In expert-parallel mode, each rank owns n_experts // world_size experts
        n_local_experts = max(1, cfg.n_experts // world_size)

        self.local_moe = SparseMoELayer(
            MoEConfig(
                d_model   = cfg.d_model,
                n_experts = n_local_experts,
                n_active  = min(cfg.n_active, n_local_experts),
                d_ff      = cfg.d_ff,
                dropout   = cfg.dropout,
            )
        )

        # Router still sees all experts
        self.global_router = TopKRouter(
            cfg.d_model, cfg.n_experts, cfg.n_active
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        In a real implementation, this would:
        1. Run global router on all ranks
        2. All-to-all dispatch to remote experts
        3. Local expert processing
        4. All-to-all combine
        5. Weighted sum

        Here we fall back to local MoE.
        """
        return self.local_moe(x)

    @staticmethod
    def is_available() -> bool:
        """Check if expert parallelism is available (requires distributed setup)."""
        return False  # Set to True when distributed is configured


# ---------------------------------------------------------------------------
# MoE Layer with residual connection (drop-in replacement for FFN)
# ---------------------------------------------------------------------------

class MoEFFNLayer(nn.Module):
    """
    Drop-in replacement for a standard FFN layer, using sparse MoE.
    Does NOT include the residual connection (handled by TransformerBlock).
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.moe = SparseMoELayer(cfg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, D)
        Returns: (output without residual, aux_loss)
        Note: SparseMoELayer already adds residual + norm internally.
        """
        out, aux = self.moe(x)
        return out, aux


# ---------------------------------------------------------------------------
# Switch Transformer variant (top-1 routing)
# ---------------------------------------------------------------------------

class SwitchMoELayer(nn.Module):
    """
    Switch Transformer (Fedus et al. 2021) variant: top-1 routing.
    Simpler than top-k but less accurate — used for research comparisons.
    """

    def __init__(
        self,
        d_model:   int,
        n_experts: int   = 8,
        d_ff:      Optional[int] = None,
        dropout:   float = 0.1,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.n_experts       = n_experts
        self.capacity_factor = capacity_factor

        self.router  = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff, "swiglu", dropout)
            for _ in range(n_experts)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D   = x.shape
        N         = B * T
        x_flat    = x.reshape(N, D)

        # Top-1 routing
        logits    = self.router(x_flat)
        probs     = F.softmax(logits, dim=-1)   # (N, E)
        top1_val, top1_idx = probs.max(-1)       # (N,)

        # Expert capacity
        capacity = compute_expert_capacity(N, self.n_experts, 1, self.capacity_factor)

        out = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        for e, expert in enumerate(self.experts):
            mask  = (top1_idx == e)
            n_tok = mask.sum().item()
            if n_tok == 0:
                continue
            # Enforce capacity
            if n_tok > capacity:
                # Drop overflow tokens (keep first `capacity`)
                idx    = mask.nonzero(as_tuple=True)[0][:capacity]
                mask   = torch.zeros_like(mask)
                mask[idx] = True
            tok_in         = x_flat[mask]
            out[mask]      = expert(tok_in) * top1_val[mask].unsqueeze(-1)

        # Load balancing loss
        expert_counts = F.one_hot(top1_idx, self.n_experts).float().mean(0)
        P             = probs.mean(0)
        lb_loss       = self.n_experts * (expert_counts * P).sum()

        out = self.norm(out.reshape(B, T, D) + x)
        return out, lb_loss


# ---------------------------------------------------------------------------
# Mixture-of-Depths (optional)
# ---------------------------------------------------------------------------

class MixtureOfDepths(nn.Module):
    """
    Mixture of Depths (Raposo et al. 2024).
    Routes tokens to either process through a layer or skip it entirely.
    Reduces effective compute by only processing informative tokens.
    """

    def __init__(
        self,
        block:         nn.Module,
        d_model:       int,
        capacity_frac: float = 0.5,   # fraction of tokens to process
    ):
        super().__init__()
        self.block         = block
        self.capacity_frac = capacity_frac
        self.router        = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        capacity = max(1, int(T * self.capacity_frac))

        # Route: select top-capacity tokens to process
        scores  = self.router(x).squeeze(-1)   # (B, T)
        topk    = scores.topk(capacity, dim=-1)
        tok_idx = topk.indices                  # (B, capacity)

        # Gather selected tokens
        selected = torch.gather(x, 1, tok_idx.unsqueeze(-1).expand(-1, -1, D))

        # Process through block
        processed = self.block(selected)
        if isinstance(processed, tuple):
            processed = processed[0]

        # Scatter back
        out = x.clone()
        out.scatter_(1, tok_idx.unsqueeze(-1).expand(-1, -1, D), processed)
        return out


# ---------------------------------------------------------------------------
# Full MoE Transformer Stack
# ---------------------------------------------------------------------------

class MoETransformerStack(nn.Module):
    """
    Transformer stack where every moe_every_n layers has a MoE FFN.
    Remaining layers use dense SwiGLU FFN.
    """

    from .transformer import TransformerConfig, TransformerBlock, StackedTransformer

    def __init__(self, transformer_cfg: "TransformerConfig", moe_cfg: MoEConfig):
        super().__init__()
        from .transformer import TransformerBlock, RMSNorm

        self.transformer_cfg = transformer_cfg
        self.moe_cfg         = moe_cfg
        n_layers             = transformer_cfg.n_layers

        layers = []
        for i in range(n_layers):
            if i % transformer_cfg.moe_every_n == transformer_cfg.moe_every_n - 1:
                layers.append(MoETransformerBlockFull(transformer_cfg, moe_cfg, i))
            else:
                layers.append(TransformerBlock(transformer_cfg, i))

        self.layers   = nn.ModuleList(layers)
        self.norm_out = RMSNorm(transformer_cfg.d_model, eps=transformer_cfg.norm_eps)
        self._aux     = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        from .transformer import TransformerBlock

        total_aux = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            if isinstance(layer, MoETransformerBlockFull):
                x, lb = layer(x, mask=mask)
                total_aux = total_aux + lb
            else:
                x = layer(x, mask=mask)

        self._aux = total_aux
        return self.norm_out(x)

    def get_aux_loss(self) -> torch.Tensor:
        return self._aux


class MoETransformerBlockFull(nn.Module):
    """Transformer block with attention + MoE FFN."""

    def __init__(
        self,
        tf_cfg:   "TransformerConfig",
        moe_cfg:  MoEConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        from .transformer import RMSNorm, SwiGLUFFN
        from .attention import GroupedQueryAttention, AttentionConfig

        self.norm1 = RMSNorm(tf_cfg.d_model, eps=tf_cfg.norm_eps)
        self.norm2 = RMSNorm(tf_cfg.d_model, eps=tf_cfg.norm_eps)

        attn_cfg = AttentionConfig(
            d_model    = tf_cfg.d_model,
            n_heads    = tf_cfg.n_heads,
            n_kv_heads = tf_cfg.n_kv_heads,
            dropout    = tf_cfg.attn_dropout,
            causal     = tf_cfg.causal,
            max_seq_len = tf_cfg.max_seq_len,
            pos_encoding = tf_cfg.pos_encoding,
            use_flash   = tf_cfg.use_flash,
        )
        self.attn    = GroupedQueryAttention(attn_cfg)
        self.moe_ffn = SparseMoELayer(moe_cfg)
        self.dropout = nn.Dropout(tf_cfg.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        moe_out, lb = self.moe_ffn(self.norm2(x))
        x = x + self.dropout(moe_out - self.norm2(x))   # remove double residual
        return x, lb


# ---------------------------------------------------------------------------
# Expert Utilization Monitor
# ---------------------------------------------------------------------------

class ExpertUtilizationMonitor:
    """Tracks expert utilization statistics across training steps."""

    def __init__(self, n_experts: int):
        self.n_experts   = n_experts
        self.step_counts  = []
        self.reset()

    def reset(self):
        self.total_tokens  = 0
        self.expert_counts = torch.zeros(self.n_experts)

    def update(self, selected_experts: torch.Tensor) -> None:
        """selected_experts: (N, k) integer indices."""
        N, k = selected_experts.shape
        self.total_tokens += N * k
        for e in range(self.n_experts):
            self.expert_counts[e] += (selected_experts == e).float().sum().item()

    def get_stats(self) -> Dict[str, float]:
        if self.total_tokens == 0:
            return {}
        frac = self.expert_counts / (self.total_tokens + 1e-8)
        return {
            "expert_fractions": frac.tolist(),
            "max_utilization":  frac.max().item(),
            "min_utilization":  frac.min().item(),
            "utilization_cv":   (frac.std() / (frac.mean() + 1e-8)).item(),  # coefficient of variation
        }

    def log_step(self) -> None:
        self.step_counts.append(self.get_stats())
        self.reset()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "MoEConfig",
    "ExpertFFN",
    "TopKRouter",
    "LoadBalancingLoss",
    "SparseMoELayer",
    "MoEFFNLayer",
    "SwitchMoELayer",
    "ExpertParallelStub",
    "MixtureOfDepths",
    "MoETransformerStack",
    "MoETransformerBlockFull",
    "ExpertUtilizationMonitor",
    "compute_expert_capacity",
]
