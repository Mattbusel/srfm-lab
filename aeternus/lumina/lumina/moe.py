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
# Expert Routing Analysis
# ---------------------------------------------------------------------------

class ExpertRoutingAnalyzer:
    """Analyze expert routing patterns for diagnosis and interpretability.

    Tracks which tokens get routed to which experts, detects routing
    collapse, and computes routing statistics over a training run.

    Args:
        n_experts: number of experts in the MoE layer

    Example:
        >>> analyzer = ExpertRoutingAnalyzer(n_experts=8)
        >>> for batch in dataloader:
        ...     out, aux = moe_layer(batch)
        ...     analyzer.update(moe_layer)
        >>> print(analyzer.summary())
    """

    def __init__(self, n_experts: int):
        self.n_experts = n_experts
        self._routing_history: List[Dict] = []
        self._step = 0

    def update(self, moe_layer) -> None:
        """Record current routing state from a MoE layer.

        Args:
            moe_layer: SparseMoELayer or similar with expert_counts buffer
        """
        self._step += 1
        if hasattr(moe_layer, "expert_counts"):
            counts = moe_layer.expert_counts.detach().cpu().float()
            total = counts.sum().item()
            if total > 0:
                fracs = (counts / total).tolist()
            else:
                fracs = [0.0] * self.n_experts

            self._routing_history.append({
                "step": self._step,
                "fracs": fracs,
                "entropy": -sum(
                    max(f, 1e-10) * math.log(max(f, 1e-10))
                    for f in fracs
                ),
                "max_frac": max(fracs),
                "min_frac": min(fracs),
                "collapse": any(f < 0.01 / self.n_experts for f in fracs),  # dead expert
            })

    def summary(self) -> Dict:
        """Return summary statistics over all recorded steps."""
        if not self._routing_history:
            return {}

        n = len(self._routing_history)
        avg_entropy = sum(r["entropy"] for r in self._routing_history) / n
        n_collapsed = sum(1 for r in self._routing_history if r["collapse"])
        max_entropy = math.log(self.n_experts)  # uniform = max entropy
        uniformity = avg_entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            "n_steps": n,
            "avg_routing_entropy": avg_entropy,
            "max_possible_entropy": max_entropy,
            "routing_uniformity": uniformity,
            "n_steps_with_collapsed_expert": n_collapsed,
            "collapse_fraction": n_collapsed / n,
            "last_fracs": self._routing_history[-1]["fracs"],
        }

    def get_history(self) -> List[Dict]:
        """Return full routing history."""
        return self._routing_history.copy()

    def reset(self) -> None:
        """Clear routing history."""
        self._routing_history = []
        self._step = 0


# ---------------------------------------------------------------------------
# Expert Specialization Metrics
# ---------------------------------------------------------------------------

class ExpertSpecializationMetrics:
    """Measure the degree of expert specialization in a MoE model.

    Specialization metrics:
    1. Routing Entropy: how uniformly are tokens spread across experts?
    2. Jaccard Similarity: token sets overlap between expert pairs
    3. Token Type Consistency: do same-type tokens route to same experts?

    Args:
        n_experts: number of experts

    Example:
        >>> metrics = ExpertSpecializationMetrics(n_experts=8)
        >>> route_ids = torch.randint(0, 8, (256,))  # token-to-expert assignments
        >>> metrics.update(route_ids)
        >>> print(metrics.compute())
    """

    def __init__(self, n_experts: int):
        self.n_experts = n_experts
        self._all_routes: List[torch.Tensor] = []

    def update(self, route_ids: torch.Tensor) -> None:
        """Record token routing decisions.

        Args:
            route_ids: (N,) long tensor of expert assignments per token
        """
        self._all_routes.append(route_ids.cpu())

    def compute(self) -> Dict[str, float]:
        """Compute specialization metrics from recorded routes."""
        if not self._all_routes:
            return {}

        all_routes = torch.cat(self._all_routes)  # (N,)
        N = all_routes.shape[0]

        # Expert fractions
        fracs = torch.zeros(self.n_experts)
        for e in range(self.n_experts):
            fracs[e] = (all_routes == e).float().mean()

        # Routing entropy
        fracs_pos = fracs.clamp(min=1e-10)
        entropy = -(fracs_pos * fracs_pos.log()).sum().item()
        max_entropy = math.log(self.n_experts)

        # Gini coefficient of expert load (inequality measure)
        sorted_fracs = fracs.sort().values
        n = self.n_experts
        gini = (2 * sum((i + 1) * sorted_fracs[i].item() for i in range(n))
                / (n * sorted_fracs.sum().item() + 1e-10) - (n + 1) / n)

        return {
            "routing_entropy": entropy,
            "max_entropy": max_entropy,
            "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0,
            "gini_coefficient": gini,
            "expert_fracs": fracs.tolist(),
            "most_loaded_expert": fracs.argmax().item(),
            "least_loaded_expert": fracs.argmin().item(),
            "n_dead_experts": (fracs < 0.01 / n).sum().item(),
        }

    def reset(self) -> None:
        """Clear accumulated routes."""
        self._all_routes = []


# ---------------------------------------------------------------------------
# Dynamic Expert Allocation
# ---------------------------------------------------------------------------

class DynamicExpertAllocation(nn.Module):
    """Dynamically allocate expert capacity based on routing distribution.

    Instead of a fixed capacity per expert, allocates more capacity to
    overloaded experts and less to underloaded ones, within total budget.

    Uses an exponential moving average of token fractions to track load
    and adjusts per-expert capacity fractions accordingly.

    Args:
        n_experts:       number of experts
        total_capacity:  total token capacity budget (as fraction of batch size)
        ema_alpha:       EMA smoothing factor for load tracking
        min_capacity:    minimum capacity fraction per expert

    Example:
        >>> dea = DynamicExpertAllocation(n_experts=8, total_capacity=2.0)
        >>> capacities = dea.get_capacities(batch_size=256)
        >>> # Returns per-expert token budgets
    """

    def __init__(
        self,
        n_experts: int,
        total_capacity: float = 2.0,
        ema_alpha: float = 0.1,
        min_capacity: float = 0.5,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.total_capacity = total_capacity
        self.ema_alpha = ema_alpha
        self.min_capacity = min_capacity

        # EMA of load fractions
        self.register_buffer(
            "load_ema",
            torch.ones(n_experts) / n_experts
        )

    def update_load(self, route_ids: torch.Tensor) -> None:
        """Update EMA with observed routing."""
        N = route_ids.shape[0]
        counts = torch.zeros(self.n_experts, device=route_ids.device)
        for e in range(self.n_experts):
            counts[e] = (route_ids == e).float().sum() / N
        self.load_ema = (1 - self.ema_alpha) * self.load_ema + self.ema_alpha * counts

    def get_capacities(self, batch_size: int) -> torch.Tensor:
        """Get per-expert token capacities for current batch.

        Args:
            batch_size: number of tokens in batch

        Returns:
            capacities: (n_experts,) int tensor of token budgets
        """
        # Invert load fractions: overloaded experts get more capacity
        inv_load = 1.0 / (self.load_ema + 1e-6)
        inv_load = inv_load / inv_load.sum()

        # Mix uniform and load-based allocation
        uniform = torch.ones(self.n_experts, device=self.load_ema.device) / self.n_experts
        capacity_frac = 0.7 * uniform + 0.3 * inv_load

        # Apply minimum capacity floor
        capacity_frac = capacity_frac.clamp(min=self.min_capacity / self.n_experts)
        capacity_frac = capacity_frac / capacity_frac.sum()

        # Total budget
        total_tokens = int(batch_size * self.total_capacity)
        return (capacity_frac * total_tokens).long().clamp(min=1)


# ---------------------------------------------------------------------------
# Mixture of Granularities
# ---------------------------------------------------------------------------

class MixtureOfGranularities(nn.Module):
    """MoE variant where experts operate at different temporal granularities.

    For financial data where signals exist at different time scales:
    - Micro-experts: process individual ticks/bars
    - Macro-experts: process aggregated multi-bar representations
    - Trend-experts: process momentum signals

    Each granularity level has dedicated experts, and a router decides
    which granularity level to activate per token.

    Args:
        d_model:           model dimension
        n_experts_per_scale: experts per granularity level
        n_scales:           number of granularity levels
        pool_sizes:         pooling sizes for each scale (except first)
        top_k:             tokens to route per token
        dropout:           dropout

    Example:
        >>> mog = MixtureOfGranularities(
        ...     d_model=512,
        ...     n_experts_per_scale=4,
        ...     n_scales=3,
        ...     pool_sizes=[1, 4, 16]
        ... )
        >>> x = torch.randn(2, 128, 512)
        >>> out, aux = mog(x)
    """

    def __init__(
        self,
        d_model: int,
        n_experts_per_scale: int = 4,
        n_scales: int = 3,
        pool_sizes: Optional[List[int]] = None,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_scales = n_scales
        self.n_experts_per_scale = n_experts_per_scale

        if pool_sizes is None:
            pool_sizes = [2 ** i for i in range(n_scales)]
        self.pool_sizes = pool_sizes

        # Scale-specific expert groups
        self.scale_experts = nn.ModuleList([
            nn.ModuleList([
                SwiGLUFFN(d_model, dropout=dropout)
                for _ in range(n_experts_per_scale)
            ])
            for _ in range(n_scales)
        ])

        # Multi-scale router
        total_experts = n_scales * n_experts_per_scale
        self.router = nn.Linear(d_model, total_experts, bias=False)
        self.top_k = top_k
        self.n_total = total_experts

        # Scale pooling projections
        self.scale_projs = nn.ModuleList([
            nn.Linear(d_model * ps, d_model, bias=False) if ps > 1 else nn.Identity()
            for ps in pool_sizes
        ])

    def _get_scale_repr(
        self, x: torch.Tensor, pool_size: int, proj: nn.Module
    ) -> torch.Tensor:
        """Get pooled representation at given scale."""
        if pool_size == 1:
            return x
        B, T, D = x.shape
        T_padded = ((T + pool_size - 1) // pool_size) * pool_size
        if T_padded > T:
            pad = torch.zeros(B, T_padded - T, D, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=1)
        else:
            x_padded = x
        x_grouped = x_padded.view(B, T_padded // pool_size, pool_size * D)
        x_scaled = proj(x_grouped)  # (B, T//pool_size, D)
        # Upsample back to T
        x_up = x_scaled.unsqueeze(2).expand(-1, -1, pool_size, -1)
        x_up = x_up.reshape(B, T_padded, D)[:, :T, :]
        return x_up

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            out: (B, T, d_model)
            aux_loss: scalar load balance loss
        """
        B, T, D = x.shape

        # Get scale representations
        scale_reprs = []
        for ps, proj in zip(self.pool_sizes, self.scale_projs):
            scale_reprs.append(self._get_scale_repr(x, ps, proj))

        # Route tokens
        gate_logits = self.router(x)  # (B, T, total_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_probs, topk_ids = gate_probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        output = torch.zeros_like(x)
        expert_load = gate_probs.mean(dim=0).mean(dim=0)  # (total_experts,)

        flat_x = x.view(B * T, D)

        for scale_idx, scale_repr in enumerate(scale_reprs):
            flat_repr = scale_repr.reshape(B * T, D)
            for exp_idx in range(self.n_experts_per_scale):
                global_exp_idx = scale_idx * self.n_experts_per_scale + exp_idx
                expert = self.scale_experts[scale_idx][exp_idx]

                # Find tokens routed to this expert
                for k in range(self.top_k):
                    mask = (topk_ids.view(B * T, self.top_k)[:, k] == global_exp_idx)
                    if mask.sum() == 0:
                        continue
                    weights = topk_probs.view(B * T, self.top_k)[mask, k]
                    exp_out = expert(flat_repr[mask])
                    output.view(B * T, D)[mask] += weights.unsqueeze(-1) * exp_out

        # Load balance loss
        expert_frac = torch.zeros(self.n_total, device=x.device)
        for e in range(self.n_total):
            expert_frac[e] = (topk_ids.view(-1) == e).float().mean()
        aux_loss = self.n_total * (expert_frac * expert_load).sum()

        return output, aux_loss


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------
import math

def SwiGLUFFN(d_model: int, d_ffn: Optional[int] = None, dropout: float = 0.0) -> nn.Module:
    """Convenience factory for SwiGLU FFN (import from transformer module)."""
    try:
        from .transformer import SwiGLUFFN as _SwiGLUFFN
        return _SwiGLUFFN(d_model, d_ffn, dropout)
    except ImportError:
        # Fallback inline implementation
        class _Inline(nn.Module):
            def __init__(self):
                super().__init__()
                _d_ffn = d_ffn or int(8/3 * d_model + 63) // 64 * 64
                self.g = nn.Linear(d_model, _d_ffn, bias=False)
                self.u = nn.Linear(d_model, _d_ffn, bias=False)
                self.d = nn.Linear(_d_ffn, d_model, bias=False)
                self.drop = nn.Dropout(dropout)
            def forward(self, x):
                import torch.nn.functional as F
                return self.d(self.drop(F.silu(self.g(x)) * self.u(x)))
        return _Inline()


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
    "ExpertRoutingAnalyzer",
    "ExpertSpecializationMetrics",
    "DynamicExpertAllocation",
    "MixtureOfGranularities",
    "compute_expert_capacity",
]


# =============================================================================
# SECTION: Advanced MoE Architectures
# =============================================================================

class SoftMoE(nn.Module):
    """Soft Mixture of Experts: all tokens processed by all experts.

    Unlike hard-routing MoE, Soft MoE creates 'slots' that aggregate
    information from all tokens, processes them with experts, then
    disperses outputs back. Avoids discrete routing decisions.

    Reference: Puigcerver et al., "From Sparse to Soft Mixtures of Experts"
    (ICLR 2024)

    Args:
        d_model: Model dimension
        num_experts: Number of expert networks
        num_slots: Number of aggregation slots per expert
        d_ff: Expert hidden dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        num_slots: int = 1,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.d_model = d_model
        # Slot embeddings: (num_experts * num_slots, d_model)
        self.slot_embeds = nn.Parameter(
            torch.randn(num_experts * num_slots, d_model) * 0.02
        )
        # Expert networks (one per expert)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model, bias=False),
            )
            for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        E, S = self.num_experts, self.num_slots
        total_slots = E * S

        # Dispatch weights: how much each token contributes to each slot
        # slots: (total_slots, D); x: (B, T, D)
        slots = self.slot_embeds.unsqueeze(0).expand(B, -1, -1)  # (B, E*S, D)
        # logits: (B, T, E*S)
        logits = torch.matmul(x, slots.transpose(-2, -1)) / (D ** 0.5)
        dispatch_weights = torch.softmax(logits, dim=1)  # (B, T, E*S) - over tokens per slot

        # Aggregate tokens into slots
        # slot_inputs: (B, E*S, D) = dispatch_weights^T @ x
        slot_inputs = torch.matmul(dispatch_weights.transpose(-2, -1), x)  # (B, E*S, D)
        slot_inputs = self.norm(slot_inputs)

        # Process each expert's slots
        slot_outputs = torch.zeros_like(slot_inputs)
        for e in range(E):
            start, end = e * S, (e + 1) * S
            slot_outputs[:, start:end, :] = self.experts[e](slot_inputs[:, start:end, :])

        # Combine weights: how each slot contributes to each token output
        combine_weights = torch.softmax(logits, dim=2)  # (B, T, E*S) - over slots per token

        # Scatter: output = combine_weights @ slot_outputs
        output = torch.matmul(combine_weights, slot_outputs)  # (B, T, D)
        return output


class HierarchicalMoE(nn.Module):
    """Hierarchical Mixture of Experts with two-level routing.

    Level 1: Route to expert group (coarse)
    Level 2: Route to specific expert within group (fine)

    This creates a tree structure of experts that can specialize
    at different levels of abstraction.

    Args:
        d_model: Model dimension
        num_groups: Number of expert groups (Level 1)
        experts_per_group: Number of experts per group (Level 2)
        d_ff: Expert hidden dimension
        top_k_groups: Top-K groups to activate (Level 1)
        top_k_experts: Top-K experts to activate per selected group
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_groups: int = 4,
        experts_per_group: int = 4,
        d_ff: Optional[int] = None,
        top_k_groups: int = 2,
        top_k_experts: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.top_k_groups = top_k_groups
        self.top_k_experts = top_k_experts
        total_experts = num_groups * experts_per_group

        # Level 1 router (to groups)
        self.group_router = nn.Linear(d_model, num_groups, bias=False)
        # Level 2 routers (one per group, to experts within group)
        self.expert_routers = nn.ModuleList([
            nn.Linear(d_model, experts_per_group, bias=False)
            for _ in range(num_groups)
        ])
        # Expert networks
        self.experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_ff, bias=False),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model, bias=False),
                )
                for _ in range(experts_per_group)
            ])
            for _ in range(num_groups)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        # Level 1: group routing
        group_logits = self.group_router(x_flat)  # (B*T, G)
        group_weights, group_ids = group_logits.topk(self.top_k_groups, dim=-1)
        group_weights = torch.softmax(group_weights, dim=-1)

        output = torch.zeros_like(x_flat)

        for g_rank in range(self.top_k_groups):
            g_idx = group_ids[:, g_rank]  # (B*T,)
            g_w = group_weights[:, g_rank:g_rank+1]  # (B*T, 1)

            # Process each group
            for g in range(self.num_groups):
                mask = (g_idx == g)
                if mask.sum() == 0:
                    continue

                x_g = x_flat[mask]  # (n_g, D)
                # Level 2: expert routing within group
                expert_logits = self.expert_routers[g](x_g)  # (n_g, E_g)
                top_e_weights, top_e_ids = expert_logits.topk(self.top_k_experts, dim=-1)
                top_e_weights = torch.softmax(top_e_weights, dim=-1)  # (n_g, top_k_experts)

                expert_out = torch.zeros_like(x_g)
                for e_rank in range(self.top_k_experts):
                    e_idx = top_e_ids[:, e_rank]
                    e_w = top_e_weights[:, e_rank:e_rank+1]
                    for e in range(self.experts_per_group):
                        e_mask = (e_idx == e)
                        if e_mask.sum() == 0:
                            continue
                        expert_out[e_mask] += e_w[e_mask] * self.experts[g][e](x_g[e_mask])

                output[mask] += g_w[mask] * expert_out

        output = output.view(B, T, D)
        routing_info = {
            "group_logits": group_logits.view(B, T, -1),
            "group_entropy": -(torch.softmax(group_logits, -1) *
                               torch.log_softmax(group_logits, -1)).sum(-1).mean(),
        }
        return output, routing_info


class SharedExpertMoE(nn.Module):
    """MoE with shared experts (always active) + routing experts.

    Some experts (shared) process all tokens, while routing experts
    process based on learned dispatch. The shared experts capture
    common patterns while routing experts specialize.

    Reference: Dai et al., "DeepSeekMoE" (2024)

    Args:
        d_model: Model dimension
        num_shared_experts: Always-active expert count
        num_routing_experts: Pool of routing experts
        top_k: Number of routing experts activated per token
        d_ff: Expert FFN dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_shared_experts: int = 2,
        num_routing_experts: int = 16,
        top_k: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.num_shared = num_shared_experts
        self.num_routing = num_routing_experts
        self.top_k = top_k

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff // num_shared_experts, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff // num_shared_experts, d_model, bias=False),
            )
            for _ in range(num_shared_experts)
        ])

        # Routing experts
        self.routing_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff // top_k, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff // top_k, d_model, bias=False),
            )
            for _ in range(num_routing_experts)
        ])

        # Router
        self.router = nn.Linear(d_model, num_routing_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        # Shared expert pass (all tokens)
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)

        # Routing expert dispatch
        logits = self.router(x_flat)  # (B*T, E_r)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)

        routing_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            e_idx = indices[:, k]
            e_w = weights[:, k:k+1]
            for e in range(self.num_routing):
                mask = (e_idx == e)
                if mask.sum() == 0:
                    continue
                routing_out[mask] += e_w[mask] * self.routing_experts[e](x_flat[mask])

        # Combine shared + routing
        output = (shared_out + routing_out).view(B, T, D)

        # Load balancing loss
        router_probs = torch.softmax(logits, dim=-1)
        load = router_probs.mean(0)
        balance_loss = self.num_routing * (load * load).sum()

        return output, balance_loss


class ExpertMerging(nn.Module):
    """Expert merging for efficient MoE inference.

    At inference time, merges multiple experts into fewer (or 1)
    weighted expert to reduce memory footprint while preserving quality.

    Supports:
    - Task-vector merging: weighted average of expert weights
    - TIES merging: trim, elect, and merge conflicting parameters
    - LoRA-based merging: merge via shared low-rank factors

    Args:
        num_experts: Source number of experts
        expert_dim: Expert network dimension
    """

    def __init__(self, num_experts: int, expert_dim: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.expert_dim = expert_dim

    def merge_weights(
        self,
        expert_weights: List[Dict[str, torch.Tensor]],
        merge_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Merge expert state dicts via weighted average.

        Args:
            expert_weights: List of expert state dicts
            merge_weights: (num_experts,) weight tensor, default uniform
        Returns:
            Merged state dict
        """
        if merge_weights is None:
            merge_weights = torch.ones(len(expert_weights)) / len(expert_weights)

        merged = {}
        for key in expert_weights[0].keys():
            merged[key] = sum(
                w * ew[key] for w, ew in zip(merge_weights, expert_weights)
            )
        return merged

    def ties_merge(
        self,
        expert_weights: List[Dict[str, torch.Tensor]],
        density: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """TIES merging: Trim, Elect Sign, Merge.

        1. Trim small magnitudes to zero (density fraction kept)
        2. Resolve sign conflicts by majority vote
        3. Average non-conflicting parameters

        Reference: Yadav et al., "TIES-Merging" NeurIPS 2023

        Args:
            expert_weights: List of expert state dicts
            density: Fraction of parameters to keep (top-density by magnitude)
        Returns:
            Merged state dict
        """
        merged = {}
        for key in expert_weights[0].keys():
            stacked = torch.stack([ew[key] for ew in expert_weights], dim=0)  # (E, ...)

            # Step 1: Trim - keep only top density fraction
            flat = stacked.view(len(expert_weights), -1)
            thresh = torch.quantile(flat.abs(), 1 - density, dim=1, keepdim=True)
            trimmed = torch.where(flat.abs() >= thresh, flat, torch.zeros_like(flat))
            trimmed = trimmed.view_as(stacked)

            # Step 2: Elect sign via majority vote
            pos_count = (trimmed > 0).float().sum(0)
            neg_count = (trimmed < 0).float().sum(0)
            elected_sign = torch.where(pos_count >= neg_count, torch.ones_like(pos_count), -torch.ones_like(pos_count))

            # Step 3: Average where sign agrees with elected
            sign_match = (trimmed.sign() == elected_sign.unsqueeze(0)).float()
            weighted = trimmed * sign_match
            count = sign_match.sum(0).clamp(min=1)
            merged[key] = weighted.sum(0) / count

        return merged


class BalancedMoELayer(nn.Module):
    """MoE layer with auxiliary load-balancing loss and z-loss.

    Implements the expert routing from Switch Transformer + z-loss
    regularization from ST-MoE to prevent router collapse.

    Reference:
    - Fedus et al., "Switch Transformers" JMLR 2022
    - Zoph et al., "ST-MoE: Designing Stable and Transferable
      Sparse Expert Models" 2022

    Args:
        d_model: Model dimension
        num_experts: Total experts
        top_k: Number of active experts per token
        d_ff: Expert FFN width
        capacity_factor: Token capacity per expert (overflow dropped)
        aux_loss_coeff: Load balance loss coefficient
        z_loss_coeff: Z-loss coefficient
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 1,
        d_ff: Optional[int] = None,
        capacity_factor: float = 1.25,
        aux_loss_coeff: float = 0.01,
        z_loss_coeff: float = 0.001,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_coeff = aux_loss_coeff
        self.z_loss_coeff = z_loss_coeff

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model, bias=False),
            )
            for _ in range(num_experts)
        ])

    def _load_balance_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Auxiliary load balance loss from Switch Transformer."""
        E = self.num_experts
        # Fraction of tokens routed to each expert
        one_hot = F.one_hot(expert_indices, E).float()  # (T, E)
        fraction = one_hot.mean(0)  # (E,)
        # Mean router probability per expert
        mean_prob = router_probs.mean(0)  # (E,)
        return E * (fraction * mean_prob).sum()

    def _z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Z-loss: penalize large logits to prevent router collapse."""
        return torch.log(torch.exp(logits).sum(dim=-1)).pow(2).mean()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        # Router
        logits = self.router(x_flat)  # (B*T, E)
        router_probs = torch.softmax(logits, dim=-1)
        top_weights, top_indices = router_probs.topk(self.top_k, dim=-1)
        # Normalize top-k weights
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Compute auxiliary losses
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            lbl = self._load_balance_loss(router_probs, top_indices[:, 0], B * T)
            z = self._z_loss(logits)
            aux_loss = self.aux_loss_coeff * lbl + self.z_loss_coeff * z

        # Dispatch to experts
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_id = top_indices[:, k]
            expert_w = top_weights[:, k:k+1]
            for e in range(self.num_experts):
                mask = (expert_id == e)
                if mask.sum() == 0:
                    continue
                expert_out = self.experts[e](x_flat[mask])
                output[mask] += expert_w[mask] * expert_out

        return output.view(B, T, D), aux_loss


_NEW_MOE_EXPORTS = [
    "SoftMoE", "HierarchicalMoE", "SharedExpertMoE", "ExpertMerging", "BalancedMoELayer",
]
