"""
lumina/transformer.py

Core transformer components for Lumina:

  - RMSNorm
  - MultiHeadSelfAttention (with optional flash-attn path)
  - GroupedQueryAttention (GQA)
  - SwiGLUFFN
  - TransformerBlock (pre-norm)
  - CausalTransformer (decoder)
  - BidirectionalTransformer (encoder)
  - MixtureOfExpertsLayer (sparse MoE, top-2)
  - LuminaModel (full stack)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .positional_encoding import (
    RotaryPositionalEncoding,
    ALiBiPositionalBias,
    TemporalEncoding,
    CrossModalPositionalEncoding,
)

# Optional flash-attention
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Faster than LayerNorm: no mean subtraction, only RMS scaling.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x.float() / rms
        return (self.weight * x_norm).to(x.dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward as in LLaMA/PaLM.

    FFN(x) = (SiLU(W_gate(x)) * W_up(x)) @ W_down

    The intermediate dimension is typically 8/3 * d_model (or closest multiple of 64).
    """

    def __init__(self, d_model: int, d_ffn: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if d_ffn is None:
            d_ffn = int(8 / 3 * d_model)
            d_ffn = ((d_ffn + 63) // 64) * 64

        self.d_model = d_model
        self.d_ffn = d_ffn

        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.w_down(hidden)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ffn={self.d_ffn}"


# ---------------------------------------------------------------------------
# MultiHeadSelfAttention
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention with optional flash attention.

    Supports:
      - Causal masking (for autoregressive generation)
      - RoPE positional encoding on Q, K
      - ALiBi bias
      - KV-cache for efficient inference
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_alibi: bool = False,
        use_flash: bool = False,
        max_seq_len: int = 4096,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.use_alibi = use_alibi
        self.use_flash = use_flash and FLASH_AVAILABLE

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len)
        if use_alibi:
            self.alibi = ALiBiPositionalBias(n_heads, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (B, T, d_model)
            attention_mask: (B, T) or (B, 1, T, T) bool (True = attend)
            causal: whether to apply causal mask
            kv_cache: optional (K, V) from previous step
            position_ids: (B, T) for RoPE

        Returns:
            output: (B, T, d_model)
            new_kv_cache: (K, V) tuple
        """
        B, T, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, "b t (h d) -> b h t d", h=self.n_heads)
        K = rearrange(K, "b t (h d) -> b h t d", h=self.n_heads)
        V = rearrange(V, "b t (h d) -> b h t d", h=self.n_heads)

        if self.use_rope:
            Q, K = self.rope(Q, K, seq_len=T, position_ids=position_ids)

        if kv_cache is not None:
            K_past, V_past = kv_cache
            K = torch.cat([K_past, K], dim=2)
            V = torch.cat([V_past, V], dim=2)

        new_kv_cache = (K, V)
        T_k = K.shape[2]

        if self.use_flash and x.is_cuda:
            q = rearrange(Q, "b h t d -> b t h d").contiguous()
            k = rearrange(K, "b h t d -> b t h d").contiguous()
            v = rearrange(V, "b h t d -> b t h d").contiguous()
            attn_out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                causal=causal,
            )
            out = rearrange(attn_out, "b t h d -> b t (h d)")
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

            if self.use_alibi:
                scores = scores + self.alibi.forward(T_k, x.device)

            if causal:
                causal_mask = torch.tril(
                    torch.ones(T, T_k, device=x.device, dtype=torch.bool)
                )
                scores = scores.masked_fill(
                    ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(~mask, float("-inf"))
                elif attention_mask.dim() == 4:
                    scores = scores.masked_fill(~attention_mask, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_weights = self.dropout(attn_weights)
            attn_out = torch.matmul(attn_weights, V)
            out = rearrange(attn_out, "b h t d -> b t (h d)")

        out = self.out_proj(out)
        return out, new_kv_cache


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------
class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al. 2023).

    Uses n_kv_heads K/V heads shared across n_q_heads query heads.
    n_q_heads must be divisible by n_kv_heads.
    """

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        max_seq_len: int = 4096,
        bias: bool = False,
    ):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_q_heads // n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, n_q_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len)
        self.use_rope = use_rope

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, "b t (h d) -> b h t d", h=self.n_q_heads)
        K = rearrange(K, "b t (h d) -> b h t d", h=self.n_kv_heads)
        V = rearrange(V, "b t (h d) -> b h t d", h=self.n_kv_heads)

        if self.use_rope:
            # Apply RoPE to Q (n_q_heads) and K (n_kv_heads) separately
            Q_flat = Q.reshape(B * self.n_q_heads, 1, T, self.head_dim)
            K_flat = K.reshape(B * self.n_kv_heads, 1, T, self.head_dim)
            Q_flat, _ = self.rope(Q_flat, Q_flat, seq_len=T)
            K_flat, _ = self.rope(K_flat, K_flat, seq_len=T)
            Q = Q_flat.reshape(B, self.n_q_heads, T, self.head_dim)
            K = K_flat.reshape(B, self.n_kv_heads, T, self.head_dim)

        if kv_cache is not None:
            K_past, V_past = kv_cache
            K = torch.cat([K_past, K], dim=2)
            V = torch.cat([V_past, V], dim=2)

        new_kv_cache = (K, V)
        T_k = K.shape[2]

        K_exp = K.repeat_interleave(self.n_groups, dim=1)
        V_exp = V.repeat_interleave(self.n_groups, dim=1)

        scores = torch.matmul(Q, K_exp.transpose(-2, -1)) * self.scale

        if causal:
            cm = torch.tril(torch.ones(T, T_k, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~cm.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                scores = scores.masked_fill(
                    ~attention_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, V_exp)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.out_proj(out)
        return out, new_kv_cache


# ---------------------------------------------------------------------------
# Mixture of Experts Layer
# ---------------------------------------------------------------------------
class ExpertFFN(nn.Module):
    """A single expert: SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))


class MixtureOfExpertsLayer(nn.Module):
    """
    Sparse MoE Layer with top-2 gating and load-balancing loss.

    For each token, the gating network selects top-2 experts.
    The output is a weighted sum of the two selected expert outputs.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        top_k: int = 2,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
        lb_coeff: float = 1e-2,
    ):
        super().__init__()
        if d_ffn is None:
            d_ffn = int(8 / 3 * d_model)
            d_ffn = ((d_ffn + 63) // 64) * 64

        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.lb_coeff = lb_coeff

        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ffn, dropout) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.register_buffer("expert_counts", torch.zeros(n_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_probs, topk_ids = gate_probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        output = torch.zeros_like(x_flat)
        expert_load = gate_probs.mean(dim=0)

        for k_idx in range(self.top_k):
            expert_ids = topk_ids[:, k_idx]
            weights = topk_probs[:, k_idx]

            for e_idx in range(self.n_experts):
                token_mask = (expert_ids == e_idx)
                if token_mask.sum() == 0:
                    continue
                tokens = x_flat[token_mask]
                expert_out = self.experts[e_idx](tokens)
                output[token_mask] += weights[token_mask].unsqueeze(-1) * expert_out

        with torch.no_grad():
            counts = torch.zeros(self.n_experts, device=x.device)
            for e in range(self.n_experts):
                counts[e] = (topk_ids == e).sum().float()
            self.expert_counts = 0.99 * self.expert_counts + 0.01 * counts

        expert_frac = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            expert_frac[e] = (topk_ids == e).float().sum() / (B * T * self.top_k)

        aux_loss = self.lb_coeff * self.n_experts * (expert_frac * expert_load).sum()

        return output.view(B, T, D), aux_loss


# ---------------------------------------------------------------------------
# TransformerBlock (Pre-Norm)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer Block: RMSNorm before each sub-layer.
    x → RMSNorm → Attention → residual → RMSNorm → FFN → residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
        use_moe: bool = False,
        n_experts: int = 8,
        use_rope: bool = True,
        use_alibi: bool = False,
        use_flash: bool = False,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.use_moe = use_moe

        if n_kv_heads is not None and n_kv_heads != n_heads:
            self.attn = GroupedQueryAttention(
                d_model, n_heads, n_kv_heads,
                dropout=dropout, use_rope=use_rope, max_seq_len=max_seq_len,
            )
        else:
            self.attn = MultiHeadSelfAttention(
                d_model, n_heads,
                dropout=dropout, use_rope=use_rope, use_alibi=use_alibi,
                use_flash=use_flash, max_seq_len=max_seq_len,
            )

        if use_moe:
            self.ffn = MixtureOfExpertsLayer(
                d_model, n_experts=n_experts, d_ffn=d_ffn, dropout=dropout
            )
        else:
            self.ffn = SwiGLUFFN(d_model, d_ffn, dropout=dropout)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        residual = x
        x_norm = self.norm1(x)
        attn_out, new_kv_cache = self.attn(
            x_norm,
            attention_mask=attention_mask,
            causal=causal,
            kv_cache=kv_cache,
            position_ids=position_ids,
        )
        x = residual + self.dropout(attn_out)

        residual = x
        x_norm = self.norm2(x)

        aux_loss = None
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(x_norm)
        else:
            ffn_out = self.ffn(x_norm)

        x = residual + self.dropout(ffn_out)
        return x, aux_loss, new_kv_cache


# ---------------------------------------------------------------------------
# Causal Transformer (Decoder)
# ---------------------------------------------------------------------------
class CausalTransformer(nn.Module):
    """Decoder-only causal transformer for autoregressive generation."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
        moe_every_n: int = 4,
        n_experts: int = 8,
        use_rope: bool = True,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_ffn=d_ffn,
                dropout=dropout,
                use_moe=(moe_every_n > 0) and (i % moe_every_n == moe_every_n - 1),
                n_experts=n_experts,
                use_rope=use_rope,
                max_seq_len=max_seq_len,
            )
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple]]:
        if kv_caches is None:
            kv_caches = [None] * self.n_layers

        total_aux_loss = torch.tensor(0.0, device=x.device)
        new_kv_caches = []

        for i, block in enumerate(self.blocks):
            x, aux_loss, new_kv = block(
                x,
                attention_mask=attention_mask,
                causal=True,
                kv_cache=kv_caches[i],
                position_ids=position_ids,
            )
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
            new_kv_caches.append(new_kv)

        x = self.norm(x)
        return x, total_aux_loss, new_kv_caches


# ---------------------------------------------------------------------------
# Bidirectional Transformer (Encoder)
# ---------------------------------------------------------------------------
class BidirectionalTransformer(nn.Module):
    """Encoder-only bidirectional transformer (full attention)."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
        moe_every_n: int = 0,
        n_experts: int = 8,
        use_rope: bool = True,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_ffn=d_ffn,
                dropout=dropout,
                use_moe=(moe_every_n > 0 and i % moe_every_n == moe_every_n - 1),
                n_experts=n_experts,
                use_rope=use_rope,
                max_seq_len=max_seq_len,
            )
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_aux_loss = torch.tensor(0.0, device=x.device)

        for block in self.blocks:
            x, aux_loss, _ = block(x, attention_mask=attention_mask, causal=False)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)
        return x, total_aux_loss


# ---------------------------------------------------------------------------
# LuminaModel — Full Stack
# ---------------------------------------------------------------------------
@dataclass
class LuminaConfig:
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: Optional[int] = 2
    d_ffn: Optional[int] = None
    dropout: float = 0.1
    max_seq_len: int = 512

    use_rope: bool = True
    use_alibi: bool = False
    use_temporal: bool = True

    use_moe: bool = True
    moe_every_n: int = 4
    n_experts: int = 8
    moe_lb_coeff: float = 1e-2

    use_flash: bool = False

    vocab_size: int = 50000
    unified_token_dim: int = 256

    arch: str = "causal"

    lm_head: bool = True
    pool_head: bool = True
    n_classes: int = 8


class LuminaModel(nn.Module):
    """
    Full Lumina transformer stack.

    Accepts pre-tokenized embeddings (from MultiModalTokenizer) and
    outputs hidden states for downstream tasks.
    """

    def __init__(self, config: LuminaConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.unified_token_dim, config.d_model)
        self.input_norm = RMSNorm(config.d_model)

        if config.use_temporal:
            self.temporal_enc = TemporalEncoding(config.d_model)

        if config.arch == "causal":
            self.transformer = CausalTransformer(
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_ffn=config.d_ffn,
                dropout=config.dropout,
                moe_every_n=config.moe_every_n if config.use_moe else 0,
                n_experts=config.n_experts,
                use_rope=config.use_rope,
                max_seq_len=config.max_seq_len,
            )
        else:
            self.transformer = BidirectionalTransformer(
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_ffn=config.d_ffn,
                dropout=config.dropout,
                moe_every_n=config.moe_every_n if config.use_moe else 0,
                n_experts=config.n_experts,
                use_rope=config.use_rope,
                max_seq_len=config.max_seq_len,
            )

        if config.lm_head:
            self.lm_head = nn.Sequential(
                RMSNorm(config.d_model),
                nn.Linear(config.d_model, config.unified_token_dim),
            )

        if config.pool_head:
            self.pool_norm = RMSNorm(config.d_model)
            self.cls_head = nn.Linear(config.d_model, config.n_classes)
            self.reg_head = nn.Linear(config.d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_caches: Optional[List] = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            token_embeddings: (B, T, unified_token_dim)
            attention_mask:   (B, T) bool
            timestamps:       (B, T) float unix seconds (for temporal encoding)
            position_ids:     (B, T) long
            kv_caches:        list of (K, V) per layer

        Returns:
            dict with keys: 'hidden', 'lm_output', 'cls_logits', 'reg_output',
                            'aux_loss', 'kv_caches', 'pooled'
        """
        B, T, _ = token_embeddings.shape

        x = self.input_norm(self.input_proj(token_embeddings))

        if self.config.use_temporal and timestamps is not None:
            x = x + self.temporal_enc(timestamps)

        if self.config.arch == "causal":
            hidden, aux_loss, new_kv_caches = self.transformer(
                x,
                attention_mask=attention_mask,
                kv_caches=kv_caches,
                position_ids=position_ids,
            )
        else:
            hidden, aux_loss = self.transformer(x, attention_mask=attention_mask)
            new_kv_caches = None

        results: Dict[str, torch.Tensor] = {
            "hidden": hidden,
            "aux_loss": aux_loss,
            "kv_caches": new_kv_caches,
        }

        if self.config.lm_head and hasattr(self, "lm_head"):
            results["lm_output"] = self.lm_head(hidden)

        if self.config.pool_head and hasattr(self, "cls_head"):
            if attention_mask is not None:
                mask_f = attention_mask.float().unsqueeze(-1)
                pooled = (hidden * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)
            else:
                pooled = hidden.mean(dim=1)
            pooled = self.pool_norm(pooled)
            results["pooled"] = pooled
            results["cls_logits"] = self.cls_head(pooled)
            results["reg_output"] = self.reg_head(pooled)

        return results

    def save_pretrained(self, path: str):
        import os, json
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        config_dict = self.config.__dict__.copy()
        # Make sure n_kv_heads can serialize (might be None)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "LuminaModel":
        import json
        with open(f"{path}/config.json") as f:
            config_dict = json.load(f)
        config = LuminaConfig(**config_dict)
        model = cls(config)
        state = torch.load(f"{path}/model.pt", map_location=device)
        model.load_state_dict(state)
        return model
