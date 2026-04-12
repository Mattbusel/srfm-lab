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


# ---------------------------------------------------------------------------
# QKNorm (Query-Key Normalization)
# ---------------------------------------------------------------------------
class QKNorm(nn.Module):
    """Normalize queries and keys before attention.

    Prevents attention entropy collapse in long sequences. Each head gets
    a separate learnable scale parameter.

    Reference: Henry et al. 2020, "Query-Key Normalization for Transformers"

    Args:
        head_dim:  dimension of each head
        n_heads:   number of query heads
        n_kv_heads: number of key-value heads (may differ from n_heads in GQA)
        eps:       numerical stability epsilon

    Example:
        >>> qk_norm = QKNorm(head_dim=64, n_heads=8, n_kv_heads=2)
        >>> q = torch.randn(2, 8, 64, 64)
        >>> k = torch.randn(2, 2, 64, 64)
        >>> q_n, k_n = qk_norm(q, k)
    """

    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = n_heads
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.eps = eps

        self.q_scale = nn.Parameter(torch.ones(n_heads, 1, head_dim))
        self.k_scale = nn.Parameter(torch.ones(n_kv_heads, 1, head_dim))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize q and k.

        Args:
            q: (B, n_heads, T, head_dim)
            k: (B, n_kv_heads, T, head_dim)

        Returns:
            q_norm, k_norm: normalized tensors, same shapes
        """
        q_norm = F.normalize(q, p=2, dim=-1, eps=self.eps) * self.q_scale
        k_norm = F.normalize(k, p=2, dim=-1, eps=self.eps) * self.k_scale
        return q_norm, k_norm


# ---------------------------------------------------------------------------
# GatedFFN (alternative to SwiGLU — uses GELU + gate)
# ---------------------------------------------------------------------------
class GatedFFN(nn.Module):
    """Gated Feed-Forward Network using GELU activation.

    GatedFFN(x) = (GELU(W1(x)) ⊙ W2(x)) @ W3

    Similar to SwiGLU but uses GELU instead of SiLU.
    Slightly different properties in practice.

    Args:
        d_model:   input/output dimension
        d_ffn:     intermediate dimension (defaults to 4 * d_model)
        dropout:   dropout probability
        use_bias:  whether to use bias in linear layers

    Example:
        >>> ffn = GatedFFN(d_model=512)
        >>> x = torch.randn(2, 64, 512)
        >>> out = ffn(x)  # (2, 64, 512)
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        super().__init__()
        if d_ffn is None:
            d_ffn = 4 * d_model

        self.d_model = d_model
        self.d_ffn = d_ffn

        self.gate = nn.Linear(d_model, d_ffn, bias=use_bias)
        self.up = nn.Linear(d_model, d_ffn, bias=use_bias)
        self.down = nn.Linear(d_ffn, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.dropout(F.gelu(self.gate(x)) * self.up(x)))


# ---------------------------------------------------------------------------
# ReGLU and GeGLU variants
# ---------------------------------------------------------------------------
class ReGLU(nn.Module):
    """ReGLU: Rectified-Gated Linear Unit.

    FFN(x) = (ReLU(W1(x)) ⊙ W2(x)) @ W3

    One of the original gated FFN variants from "GLU Variants Improve
    Transformer" (Noam Shazeer, 2020).

    Args:
        d_model:  input/output dimension
        d_ffn:    intermediate dimension (defaults to 4*d_model)
    """

    def __init__(self, d_model: int, d_ffn: Optional[int] = None):
        super().__init__()
        if d_ffn is None:
            d_ffn = 4 * d_model
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_model, d_ffn, bias=False)
        self.w3 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.relu(self.w1(x)) * self.w2(x))


class GeGLU(nn.Module):
    """GeGLU: GELU-Gated Linear Unit.

    FFN(x) = (GELU(W1(x)) ⊙ W2(x)) @ W3

    Args:
        d_model:  input/output dimension
        d_ffn:    intermediate dimension
    """

    def __init__(self, d_model: int, d_ffn: Optional[int] = None):
        super().__init__()
        if d_ffn is None:
            d_ffn = 4 * d_model
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_model, d_ffn, bias=False)
        self.w3 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.gelu(self.w1(x)) * self.w2(x))


# ---------------------------------------------------------------------------
# MoE Transformer Block (with separate routing)
# ---------------------------------------------------------------------------
class MoETransformerBlock(nn.Module):
    """Transformer block where FFN is replaced by a MoE layer.

    Every layer uses MoE (unlike TransformerBlock which optionally uses MoE).
    Designed for MoE-specialized model configurations.

    Args:
        d_model:    model dimension
        n_heads:    number of attention heads
        n_experts:  number of MoE experts
        top_k:      number of experts per token
        d_ffn:      expert FFN intermediate dimension
        dropout:    dropout probability
        lb_coeff:   load balancing auxiliary loss coefficient
        use_rope:   use rotary position encoding
        max_seq_len:maximum sequence length
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int = 8,
        top_k: int = 2,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
        lb_coeff: float = 1e-2,
        use_rope: bool = True,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )
        self.moe = MixtureOfExpertsLayer(
            d_model=d_model,
            n_experts=n_experts,
            top_k=top_k,
            d_ffn=d_ffn,
            dropout=dropout,
            lb_coeff=lb_coeff,
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        kv_cache: Optional[Tuple] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        residual = x
        attn_out, new_kv = self.attn(
            self.norm1(x),
            attention_mask=attention_mask,
            causal=causal,
            kv_cache=kv_cache,
            position_ids=position_ids,
        )
        x = residual + self.dropout(attn_out)

        residual = x
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = residual + self.dropout(moe_out)
        return x, aux_loss, new_kv


# ---------------------------------------------------------------------------
# GradientCheckpointTransformerBlock
# ---------------------------------------------------------------------------
class GradientCheckpointTransformerBlock(nn.Module):
    """Transformer block with gradient checkpointing.

    Wraps TransformerBlock to apply gradient checkpointing during training,
    which trades compute for memory by recomputing activations in the
    backward pass.

    Memory savings: ~√n_layers (roughly, for uniform layers).

    Args:
        block: underlying TransformerBlock module

    Example:
        >>> block = TransformerBlock(d_model=512, n_heads=8)
        >>> ckpt_block = GradientCheckpointTransformerBlock(block)
        >>> out, aux, kv = ckpt_block(x)
    """

    def __init__(self, block: TransformerBlock):
        super().__init__()
        self.block = block

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        kv_cache: Optional[Tuple] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        if self.training and x.requires_grad:
            # Gradient checkpointing: only use when training and gradients are needed
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.block),
                x, attention_mask, causal, kv_cache, position_ids,
                use_reentrant=False,
            )
        else:
            return self.block(x, attention_mask, causal, kv_cache, position_ids)


# ---------------------------------------------------------------------------
# Multi-Scale Transformer Block
# ---------------------------------------------------------------------------
class MultiScaleTransformerBlock(nn.Module):
    """Transformer block with multi-scale attention.

    Runs two attention sub-layers at different resolutions:
    1. Local attention (limited window)
    2. Global attention (strided/pooled representation)

    Useful for financial data with both high-frequency and trend signals.

    Args:
        d_model:       model dimension
        n_heads:       total number of heads (split between local and global)
        local_window:  local attention window size
        global_stride: stride for global attention pooling
        d_ffn:         FFN intermediate size
        dropout:       dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        local_window: int = 32,
        global_stride: int = 8,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_heads % 2 == 0, "n_heads must be even for multi-scale"
        half_heads = n_heads // 2
        self.d_model = d_model
        self.n_heads = n_heads
        self.local_window = local_window
        self.global_stride = global_stride

        # Local attention heads
        self.local_attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=half_heads,
            dropout=dropout,
            use_rope=True,
        )

        # Global attention heads (on strided representation)
        self.global_attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=half_heads,
            dropout=dropout,
            use_rope=True,
        )

        # Merge projections
        self.merge_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.ffn = SwiGLUFFN(d_model, d_ffn, dropout)

        self.norm1_local = RMSNorm(d_model)
        self.norm1_global = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _local_window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Create local attention window mask."""
        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        dist = (i.unsqueeze(1) - j.unsqueeze(0)).abs()
        return dist <= self.local_window

    def _global_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Stride-pool x for global attention."""
        return x[:, ::self.global_stride, :]

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, T, d_model)
            attention_mask: (B, T) or None

        Returns:
            out: (B, T, d_model)
        """
        B, T, D = x.shape

        # Local attention with window mask
        local_mask = self._local_window_mask(T, x.device)
        local_out, _ = self.local_attn(self.norm1_local(x), attention_mask=None)

        # Global attention on strided representation
        x_global = self._global_pool(x)
        global_hidden, _ = self.global_attn(self.norm1_global(x_global), attention_mask=None)
        # Upsample back to T
        T_g = x_global.shape[1]
        global_hidden = global_hidden.unsqueeze(2).expand(-1, -1, self.global_stride, -1)
        global_hidden = global_hidden.reshape(B, T_g * self.global_stride, D)
        global_hidden = global_hidden[:, :T, :]  # clip to T

        # Merge local + global
        merged = self.merge_proj(torch.cat([local_out, global_hidden], dim=-1))
        x = x + self.dropout(merged)

        # FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# LayerDrop (DropBlock at layer level)
# ---------------------------------------------------------------------------
class LayerDrop(nn.Module):
    """LayerDrop: stochastic depth at the layer level.

    During training, each layer is randomly dropped (bypassed) with
    probability p_drop. During eval, all layers are used.

    Reference: Fan et al. 2019, "Reducing Transformer Depth on Demand
    with Structured Dropout"

    Args:
        layers:  nn.ModuleList of transformer layers
        p_drop:  probability of dropping each layer during training

    Example:
        >>> blocks = nn.ModuleList([TransformerBlock(512, 8) for _ in range(12)])
        >>> ld = LayerDrop(blocks, p_drop=0.1)
        >>> out, aux_loss = ld(x)
    """

    def __init__(self, layers: nn.ModuleList, p_drop: float = 0.1):
        super().__init__()
        self.layers = layers
        self.p_drop = p_drop

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_aux = torch.tensor(0.0, device=x.device)

        for layer in self.layers:
            if self.training and torch.rand(1).item() < self.p_drop:
                continue  # Skip this layer

            x, aux, _ = layer(x, attention_mask=attention_mask, causal=causal)
            if aux is not None:
                total_aux = total_aux + aux

        return x, total_aux


# ---------------------------------------------------------------------------
# Cross-Attention Block
# ---------------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    """Cross-attention block for encoder-decoder or multi-modal fusion.

    Attends from query sequence (x) to context sequence (context).

    x → norm → cross_attention(Q=x, K=context, V=context) → residual → norm → FFN → residual

    Args:
        d_model:       model dimension
        n_heads:       number of attention heads
        d_ffn:         FFN intermediate dimension
        dropout:       dropout probability
        use_rope:      apply RoPE to Q and K

    Example:
        >>> xattn = CrossAttentionBlock(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 32, 512)
        >>> context = torch.randn(2, 128, 512)
        >>> out = xattn(x, context)  # (2, 32, 512)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
        use_rope: bool = False,  # usually False for cross-attention
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn = SwiGLUFFN(d_model, d_ffn, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm_ctx = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RotaryPositionalEncoding(self.head_dim)
        self.use_rope = use_rope

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            (B, T_q, d_model) query sequence
            context:      (B, T_k, d_model) key-value context
            context_mask: (B, T_k) bool mask for context (True = valid)

        Returns:
            output: (B, T_q, d_model)
        """
        B, T_q, D = x.shape
        T_k = context.shape[1]

        # Cross-attention
        residual = x
        x_norm = self.norm1(x)
        ctx_norm = self.norm_ctx(context)

        Q = self.q_proj(x_norm)
        K = self.k_proj(ctx_norm)
        V = self.v_proj(ctx_norm)

        Q = rearrange(Q, "b t (h d) -> b h t d", h=self.n_heads)
        K = rearrange(K, "b t (h d) -> b h t d", h=self.n_heads)
        V = rearrange(V, "b t (h d) -> b h t d", h=self.n_heads)

        if self.use_rope:
            Q, K = self.rope(Q, K)

        attn_out = self._attn(Q, K, V, context_mask)
        attn_out = rearrange(attn_out, "b h t d -> b t (h d)")
        attn_out = self.out_proj(attn_out)
        x = residual + self.dropout(attn_out)

        # FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Dual-Stream Fusion Block (e.g., for multi-modal features)
# ---------------------------------------------------------------------------
class DualStreamFusionBlock(nn.Module):
    """Dual-stream transformer block for fusing two input modalities.

    Each stream processes its own input, then uses cross-attention to
    fuse information with the other stream.

    Architecture:
        Stream A: self-attn(A) → cross-attn(A, B) → FFN
        Stream B: self-attn(B) → cross-attn(B, A) → FFN

    Args:
        d_model:   model dimension
        n_heads:   attention heads per attention layer
        d_ffn:     FFN intermediate size
        dropout:   dropout probability

    Example:
        >>> fusion = DualStreamFusionBlock(d_model=512, n_heads=8)
        >>> a = torch.randn(2, 64, 512)
        >>> b = torch.randn(2, 32, 512)
        >>> a_out, b_out = fusion(a, b)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn_a = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.self_attn_b = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.cross_a_from_b = CrossAttentionBlock(d_model, n_heads, d_ffn, dropout)
        self.cross_b_from_a = CrossAttentionBlock(d_model, n_heads, d_ffn, dropout)
        self.ffn_a = SwiGLUFFN(d_model, d_ffn, dropout)
        self.ffn_b = SwiGLUFFN(d_model, d_ffn, dropout)
        self.norm_a = RMSNorm(d_model)
        self.norm_b = RMSNorm(d_model)
        self.norm_fa = RMSNorm(d_model)
        self.norm_fb = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        mask_a: Optional[torch.Tensor] = None,
        mask_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            a: (B, T_a, d_model)
            b: (B, T_b, d_model)
            mask_a, mask_b: (B, T) bool masks

        Returns:
            a_out: (B, T_a, d_model)
            b_out: (B, T_b, d_model)
        """
        # Self-attention
        a_sa, _ = self.self_attn_a(self.norm_a(a), attention_mask=mask_a)
        a = a + self.dropout(a_sa)
        b_sa, _ = self.self_attn_b(self.norm_b(b), attention_mask=mask_b)
        b = b + self.dropout(b_sa)

        # Cross-attention: A from B
        a = self.cross_a_from_b(a, b, context_mask=mask_b)
        b = self.cross_b_from_a(b, a, context_mask=mask_a)

        # FFN
        a = a + self.dropout(self.ffn_a(self.norm_fa(a)))
        b = b + self.dropout(self.ffn_b(self.norm_fb(b)))

        return a, b


# ---------------------------------------------------------------------------
# Perceiver-style Resampler
# ---------------------------------------------------------------------------
class PerceiverResampler(nn.Module):
    """Perceiver-style resampler that maps variable-length inputs to fixed-length.

    Uses a set of learnable query vectors that cross-attend to the input,
    producing a fixed number of output tokens regardless of input length.

    Particularly useful for:
    - Compressing long financial sequences to fixed-length representations
    - Fusing multi-modal inputs to a standard sequence length
    - Efficient memory representation in streaming applications

    Args:
        d_model:    model dimension
        n_latents:  number of output latent tokens (fixed output length)
        n_heads:    attention heads
        n_layers:   number of latent self-attention + cross-attention layers
        d_ffn:      FFN intermediate size
        dropout:    dropout probability

    Example:
        >>> resampler = PerceiverResampler(d_model=512, n_latents=64)
        >>> x = torch.randn(2, 512, 512)  # variable-length input
        >>> out = resampler(x)             # (2, 64, 512) fixed-length output
    """

    def __init__(
        self,
        d_model: int,
        n_latents: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * d_model ** -0.5)

        # Alternating cross-attention and self-attention layers
        self.cross_attns = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        self.self_attns = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ffn=d_ffn, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resample input to fixed n_latents tokens.

        Args:
            x:            (B, T, d_model) variable-length input
            context_mask: (B, T) bool mask for input

        Returns:
            latent: (B, n_latents, d_model)
        """
        B = x.shape[0]
        latent = self.latents.expand(B, -1, -1)

        for cross_attn, self_attn in zip(self.cross_attns, self.self_attns):
            latent = cross_attn(latent, x, context_mask=context_mask)
            latent, _, _ = self_attn(latent)

        return self.norm(latent)


# ---------------------------------------------------------------------------
# Stacked Transformer (generic multi-layer)
# ---------------------------------------------------------------------------
@dataclass
class TransformerConfig:
    """Configuration for a stacked transformer."""
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    d_ffn: Optional[int] = None
    dropout: float = 0.1
    use_rope: bool = True
    use_alibi: bool = False
    use_flash: bool = False
    use_moe: bool = False
    moe_every_n: int = 4
    n_experts: int = 8
    lb_coeff: float = 1e-2
    max_seq_len: int = 4096
    use_gradient_checkpoint: bool = False
    layer_drop_p: float = 0.0
    causal: bool = False


class StackedTransformer(nn.Module):
    """Generic stacked transformer with configurable options.

    Supports:
    - Causal or bidirectional attention
    - Gradient checkpointing
    - LayerDrop stochastic depth
    - Mixed MoE and dense layers
    - GQA

    Args:
        config: TransformerConfig dataclass

    Example:
        >>> cfg = TransformerConfig(d_model=512, n_layers=12, n_heads=8)
        >>> model = StackedTransformer(cfg)
        >>> x = torch.randn(2, 64, 512)
        >>> out, aux = model(x)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.causal = config.causal

        blocks = []
        for i in range(config.n_layers):
            use_moe_layer = (
                config.use_moe
                and config.moe_every_n > 0
                and i % config.moe_every_n == config.moe_every_n - 1
            )
            block = TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_ffn=config.d_ffn,
                dropout=config.dropout,
                use_moe=use_moe_layer,
                n_experts=config.n_experts,
                use_rope=config.use_rope,
                use_alibi=config.use_alibi,
                use_flash=config.use_flash,
                max_seq_len=config.max_seq_len,
            )
            if config.use_gradient_checkpoint:
                block = GradientCheckpointTransformerBlock(block)
            blocks.append(block)

        if config.layer_drop_p > 0:
            self.backbone = LayerDrop(nn.ModuleList(blocks), p_drop=config.layer_drop_p)
            self._uses_layer_drop = True
        else:
            self.blocks = nn.ModuleList(blocks)
            self._uses_layer_drop = False

        self.norm = RMSNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List]]:
        """
        Args:
            x:              (B, T, d_model)
            attention_mask: (B, T) bool
            kv_caches:      list of (K, V) per layer
            position_ids:   (B, T) long

        Returns:
            hidden:    (B, T, d_model) final hidden states
            aux_loss:  scalar auxiliary loss (e.g., MoE load balance)
            kv_caches: updated KV caches
        """
        if kv_caches is None:
            kv_caches = [None] * self.config.n_layers

        total_aux = torch.tensor(0.0, device=x.device)
        new_kv_caches = []

        if self._uses_layer_drop:
            x, total_aux = self.backbone(x, attention_mask=attention_mask, causal=self.causal)
            new_kv_caches = None
        else:
            for i, block in enumerate(self.blocks):
                x, aux, new_kv = block(
                    x,
                    attention_mask=attention_mask,
                    causal=self.causal,
                    kv_cache=kv_caches[i],
                    position_ids=position_ids,
                )
                if aux is not None:
                    total_aux = total_aux + aux
                new_kv_caches.append(new_kv)

        x = self.norm(x)
        return x, total_aux, new_kv_caches if not self._uses_layer_drop else None


# ---------------------------------------------------------------------------
# Builder functions for standard configs
# ---------------------------------------------------------------------------

def build_lumina_base_config() -> LuminaConfig:
    """Build Lumina-Base config: 12 layers, 512 dim, 8 heads."""
    return LuminaConfig(
        d_model=512,
        n_layers=12,
        n_heads=8,
        n_kv_heads=2,
        d_ffn=2048,
        dropout=0.1,
        max_seq_len=512,
        use_rope=True,
        use_moe=True,
        moe_every_n=4,
        n_experts=8,
        arch="causal",
    )


def build_lumina_large_config() -> LuminaConfig:
    """Build Lumina-Large config: 24 layers, 1024 dim, 16 heads."""
    return LuminaConfig(
        d_model=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        d_ffn=4096,
        dropout=0.1,
        max_seq_len=1024,
        use_rope=True,
        use_moe=True,
        moe_every_n=4,
        n_experts=16,
        arch="causal",
    )


def build_lumina_deep_config() -> LuminaConfig:
    """Build Lumina-Deep config: 48 layers, 512 dim, 8 heads (depth-focused)."""
    return LuminaConfig(
        d_model=512,
        n_layers=48,
        n_heads=8,
        n_kv_heads=2,
        d_ffn=1536,
        dropout=0.05,
        max_seq_len=512,
        use_rope=True,
        use_moe=True,
        moe_every_n=6,
        n_experts=8,
        arch="causal",
    )


def build_lumina_xl_config() -> LuminaConfig:
    """Build Lumina-XL config: 36 layers, 2048 dim, 32 heads."""
    return LuminaConfig(
        d_model=2048,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        d_ffn=8192,
        dropout=0.0,
        max_seq_len=2048,
        use_rope=True,
        use_moe=True,
        moe_every_n=4,
        n_experts=64,
        arch="causal",
    )


def build_lumina_tiny_config() -> LuminaConfig:
    """Build Lumina-Tiny config for debugging/testing."""
    return LuminaConfig(
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=None,
        d_ffn=256,
        dropout=0.0,
        max_seq_len=128,
        use_rope=True,
        use_moe=False,
        arch="causal",
        unified_token_dim=64,
        n_classes=3,
    )


# ---------------------------------------------------------------------------
# Attention pool (for sequence-to-scalar tasks)
# ---------------------------------------------------------------------------
class AttentionPool(nn.Module):
    """Attention-based pooling of sequence to fixed-size representation.

    Learns a query vector that attends over all positions to produce
    a weighted average of the sequence.

    Args:
        d_model: input/output dimension
        n_heads: number of pooling heads

    Example:
        >>> pool = AttentionPool(d_model=512, n_heads=4)
        >>> x = torch.randn(2, 64, 512)
        >>> pooled = pool(x)  # (2, 512)
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Parameter(torch.randn(1, 1, d_model) * d_model ** -0.5)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, d_model)
            mask: (B, T) bool (True = valid)

        Returns:
            pooled: (B, d_model)
        """
        B = x.shape[0]
        Q = self.query.expand(B, -1, -1)  # (B, 1, d_model)
        K = self.k_proj(x)                 # (B, T, d_model)
        V = self.v_proj(x)

        Q = rearrange(Q, "b t (h d) -> b h t d", h=self.n_heads)
        K = rearrange(K, "b t (h d) -> b h t d", h=self.n_heads)
        V = rearrange(V, "b t (h d) -> b h t d", h=self.n_heads)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, 1, T)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, V)  # (B, H, 1, head_dim)
        out = rearrange(out, "b h 1 d -> b (h d)")
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Causal Transformer (decoder) — alias to StackedTransformer for compatibility
# ---------------------------------------------------------------------------
class CausalTransformerV2(nn.Module):
    """Improved causal transformer using StackedTransformer backend.

    Provides the same interface as CausalTransformer but uses the more
    configurable StackedTransformer.

    Args:
        config: TransformerConfig with causal=True
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        config.causal = True
        self.backbone = StackedTransformer(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List]]:
        return self.backbone(x, attention_mask, kv_caches, position_ids)


# ---------------------------------------------------------------------------
# Mixture of Depths (dynamic depth allocation)
# ---------------------------------------------------------------------------
class MixtureOfDepths(nn.Module):
    """Mixture of Depths: route tokens through varying numbers of layers.

    Each token is assigned a depth (number of layers to process through)
    based on a learned routing decision. Tokens routed to depth 0 are
    passed through unchanged.

    Reference: Raposo et al. 2024, "Mixture of Depths"

    Args:
        blocks:       list of transformer blocks
        capacity_factor: fraction of tokens to process at each layer
        router_type:  routing mechanism: "top_k" | "threshold"

    Example:
        >>> blocks = nn.ModuleList([TransformerBlock(512, 8) for _ in range(8)])
        >>> mod = MixtureOfDepths(blocks, capacity_factor=0.125)
        >>> out, aux = mod(x)  # out: (B, T, 512)
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        capacity_factor: float = 0.125,
        router_type: str = "top_k",
    ):
        super().__init__()
        self.blocks = blocks
        self.capacity_factor = capacity_factor
        self.router_type = router_type
        d_model = list(blocks.parameters())[0].shape[-1]

        self.routers = nn.ModuleList([
            nn.Linear(d_model, 1, bias=False) for _ in blocks
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:              (B, T, d_model)
            attention_mask: (B, T) optional

        Returns:
            x:        (B, T, d_model)
            aux_loss: scalar
        """
        B, T, D = x.shape
        total_aux = torch.tensor(0.0, device=x.device)
        k = max(1, int(T * self.capacity_factor))

        for block, router in zip(self.blocks, self.routers):
            scores = router(x).squeeze(-1)  # (B, T)
            _, topk_idx = scores.topk(k, dim=-1)  # (B, k)

            # Route selected tokens through block
            selected = x.gather(
                1, topk_idx.unsqueeze(-1).expand(-1, -1, D)
            )  # (B, k, D)
            processed, aux, _ = block(selected, causal=False)
            if aux is not None:
                total_aux = total_aux + aux

            # Scatter back
            x = x.scatter(
                1,
                topk_idx.unsqueeze(-1).expand(-1, -1, D),
                processed,
            )

            # Load balance auxiliary
            router_probs = torch.sigmoid(scores)
            total_aux = total_aux + 1e-2 * (router_probs.mean() - self.capacity_factor).pow(2)

        return x, total_aux


# ---------------------------------------------------------------------------
# Pretrained Weight Loading Utilities
# ---------------------------------------------------------------------------

def count_transformer_params(config: LuminaConfig) -> Dict[str, int]:
    """Estimate parameter counts for a LuminaConfig.

    Args:
        config: model configuration

    Returns:
        dict with keys: 'embedding', 'attention', 'ffn', 'total'
    """
    d = config.d_model
    h = config.n_heads
    kv_h = config.n_kv_heads or config.n_heads
    d_ffn = config.d_ffn or int(8/3 * d + 63) // 64 * 64
    L = config.n_layers

    # Per layer
    attn_params = d * d + (d // h * kv_h) * d * 2 + d * d  # Q, K, V, O
    ffn_params = 3 * d * d_ffn  # gate, up, down (SwiGLU)

    if config.use_moe:
        moe_layers = L // config.moe_every_n
        dense_layers = L - moe_layers
        ffn_total = dense_layers * ffn_params + moe_layers * config.n_experts * ffn_params
    else:
        ffn_total = L * ffn_params

    norm_params = 2 * L * d  # 2 RMSNorm per block
    head_params = config.unified_token_dim * d + d * config.n_classes + d * 1

    total = (
        config.unified_token_dim * d  # input projection
        + L * attn_params
        + ffn_total
        + norm_params
        + head_params
    )

    return {
        "input_proj": config.unified_token_dim * d,
        "attention_total": L * attn_params,
        "ffn_total": ffn_total,
        "norm": norm_params,
        "heads": head_params,
        "total": total,
        "total_M": total // 1_000_000,
    }


def init_transformer_weights(module: nn.Module, std: float = 0.02) -> None:
    """Initialize transformer weights.

    Args:
        module: nn.Module to initialize
        std:    standard deviation for normal initialization
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, (RMSNorm, nn.LayerNorm)):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


def scale_init_for_residual(module: nn.Module, n_layers: int) -> None:
    """Scale output projections for residual connections (GPT-2 style).

    Scales weights of attention output projections and FFN down projections
    by 1/sqrt(2 * n_layers) to ensure stable training.

    Args:
        module:   nn.Module to scale (TransformerBlock or LuminaModel)
        n_layers: total number of layers
    """
    scale = (2 * n_layers) ** -0.5
    for name, param in module.named_parameters():
        if "out_proj" in name or "w_down" in name:
            param.data.mul_(scale)


__all__ = [
    "RMSNorm",
    "SwiGLUFFN",
    "GatedFFN",
    "ReGLU",
    "GeGLU",
    "QKNorm",
    "MultiHeadSelfAttention",
    "GroupedQueryAttention",
    "ExpertFFN",
    "MixtureOfExpertsLayer",
    "TransformerBlock",
    "MoETransformerBlock",
    "GradientCheckpointTransformerBlock",
    "MultiScaleTransformerBlock",
    "LayerDrop",
    "CrossAttentionBlock",
    "DualStreamFusionBlock",
    "PerceiverResampler",
    "TransformerConfig",
    "StackedTransformer",
    "CausalTransformer",
    "CausalTransformerV2",
    "BidirectionalTransformer",
    "MixtureOfDepths",
    "AttentionPool",
    "LuminaConfig",
    "LuminaModel",
    "build_lumina_base_config",
    "build_lumina_large_config",
    "build_lumina_deep_config",
    "build_lumina_xl_config",
    "build_lumina_tiny_config",
    "count_transformer_params",
    "init_transformer_weights",
    "scale_init_for_residual",
]


# =============================================================================
# SECTION: Advanced Feed-Forward Network Variants
# =============================================================================

class MixedActivationFFN(nn.Module):
    """FFN using a mixture of activation functions learned per neuron.

    Each hidden neuron uses a learnable convex combination of multiple
    activation functions (ReLU, GELU, SiLU, Tanh), allowing the network
    to select the best activation per context.

    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension
        num_activations: Number of activation functions to mix
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_activations: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Learnable mixing weights per hidden unit
        self.act_weights = nn.Parameter(torch.ones(d_ff, num_activations) / num_activations)
        self._activations = [F.relu, F.gelu, F.silu, torch.tanh][:num_activations]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w1(x)  # (B, T, d_ff)
        # Compute per-activation outputs
        act_outs = torch.stack([act(h) for act in self._activations], dim=-1)  # (B, T, d_ff, num_act)
        w = torch.softmax(self.act_weights, dim=-1)  # (d_ff, num_act)
        h = (act_outs * w).sum(dim=-1)  # (B, T, d_ff)
        return self.w2(self.dropout(h))


class ExpertFFN(nn.Module):
    """Single expert FFN with SwiGLU activation for MoE blocks.

    Designed as a drop-in expert for Mixture of Experts architectures.

    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class ConvFFN(nn.Module):
    """Convolutional feed-forward network for local feature extraction.

    Replaces linear projection in FFN with 1D depthwise separable
    convolution to capture local temporal patterns.

    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        p = (kernel_size - 1) // 2
        self.dw_conv = nn.Conv1d(d_ff, d_ff, kernel_size, padding=p, groups=d_ff)
        self.pw_conv = nn.Conv1d(d_ff, d_ff, 1)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.norm = nn.LayerNorm(d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.up(x))  # (B, T, d_ff)
        h = h.transpose(1, 2)   # (B, d_ff, T)
        h = self.pw_conv(self.dw_conv(h)).transpose(1, 2)  # (B, T, d_ff)
        h = self.norm(h)
        return self.down(self.dropout(h))


class HyperNetwork(nn.Module):
    """Hypernetwork that generates FFN weights conditioned on input.

    The hypernetwork generates (small) weight matrices based on a
    compressed representation of the input. The main network then
    uses these dynamic weights for its computation.

    Reference: Ha et al., "HyperNetworks" (2017)

    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension of main network
        hyper_dim: Dimension of hypernetwork hidden layer
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        hyper_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # Compress input to hyper representation
        self.hyper_enc = nn.Linear(d_model, hyper_dim)
        # Generate weights for main FFN
        self.hyper_w1 = nn.Linear(hyper_dim, d_model * d_ff // 4)  # Smaller for efficiency
        self.hyper_w2 = nn.Linear(hyper_dim, d_ff // 4 * d_model)
        self.scale = d_ff ** -0.5
        self.dropout = nn.Dropout(dropout)
        # Static bias terms
        self.b1 = nn.Parameter(torch.zeros(d_ff // 4))
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        ff_small = self.d_ff // 4
        # Generate dynamic weights from pooled representation
        x_pool = x.mean(dim=1)  # (B, D) global context
        h = F.relu(self.hyper_enc(x_pool))  # (B, hyper_dim)
        w1 = self.hyper_w1(h).view(B, D, ff_small) * self.scale  # (B, D, ff_small)
        w2 = self.hyper_w2(h).view(B, ff_small, D) * self.scale  # (B, ff_small, D)
        # Apply dynamic FFN: x -> w1 -> gelu -> w2
        h1 = torch.einsum('btd,bdf->btf', x, w1) + self.b1  # (B, T, ff_small)
        h1 = F.gelu(h1)
        h1 = self.dropout(h1)
        out = torch.einsum('btf,bfd->btd', h1, w2) + self.b2  # (B, T, D)
        return out


# =============================================================================
# SECTION: Advanced Normalization Techniques
# =============================================================================

class ScaleNorm(nn.Module):
    """Scale normalization: normalize by L2 norm, then scale.

    Simpler than LayerNorm: x_out = g * x / ||x||_2

    Reference: Nguyen & Salazar, "Transformers without Tears" (2019)

    Args:
        d_model: Feature dimension
        eps: Numerical stability
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * (d_model ** 0.5))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return self.scale * x / norm


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm with context-dependent scale and shift.

    Conditions the normalization parameters on an auxiliary input
    (e.g., timestep embedding, regime embedding), enabling the
    model to adapt its normalization to different contexts.

    Reference: Inspired by DiT and AdaNorm papers.

    Args:
        d_model: Feature dimension
        context_dim: Dimension of conditioning context
        eps: Numerical stability
    """

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.gamma = nn.Linear(context_dim, d_model)
        self.beta = nn.Linear(context_dim, d_model)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, T, D)
            context: Conditioning context (B, context_dim) or (B, T, context_dim)
        """
        x = self.norm(x)
        if context.dim() == 2:
            context = context.unsqueeze(1)  # (B, 1, context_dim)
        gamma = self.gamma(context)  # (B, 1 or T, D)
        beta = self.beta(context)
        return x * gamma + beta


class CRMSNorm(nn.Module):
    """Conditional RMSNorm: RMS normalization with context-adaptive scale.

    Combines the simplicity of RMSNorm with the flexibility of
    adaptive normalization.

    Args:
        d_model: Feature dimension
        context_dim: Conditioning dimension
        eps: Numerical stability
    """

    def __init__(self, d_model: int, context_dim: int = 0, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        if context_dim > 0:
            self.gamma_proj = nn.Linear(context_dim, d_model)
            nn.init.ones_(self.gamma_proj.bias)
            nn.init.zeros_(self.gamma_proj.weight)
        else:
            self.gamma = nn.Parameter(torch.ones(d_model))
        self.context_dim = context_dim

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        if self.context_dim > 0 and context is not None:
            if context.dim() == 2:
                context = context.unsqueeze(1)
            gamma = self.gamma_proj(context)
        else:
            gamma = self.gamma
        return x * gamma


class GroupNorm1D(nn.Module):
    """Group normalization adapted for 1D sequence data (B, T, D).

    Args:
        d_model: Feature dimension
        num_groups: Number of groups
        eps: Numerical stability
    """

    def __init__(self, d_model: int, num_groups: int = 8, eps: float = 1e-5) -> None:
        super().__init__()
        assert d_model % num_groups == 0
        self.gn = nn.GroupNorm(num_groups, d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.gn(x)
        return x.permute(0, 2, 1)  # (B, T, D)


# =============================================================================
# SECTION: Advanced Transformer Block Variants
# =============================================================================

class MacaronTransformerBlock(nn.Module):
    """Macaron-style transformer block with FFN-Attn-FFN structure.

    Uses half-step FFN before and after attention, inspired by the
    Macaron architecture from speech processing.

    Reference: Lu et al., "Understanding and Improving Transformer
    From a Multi-Particle Dynamic System Point of View" (2020)

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        d_ff: FFN hidden dimension
        dropout: Dropout probability
        pre_norm: Use pre-norm (True) or post-norm (False)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        # First half-FFN
        self.ffn1 = SwiGLU(d_model, d_ff // 2 if d_ff > d_model else d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # Attention
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # Second half-FFN
        self.ffn2 = SwiGLU(d_model, d_ff // 2 if d_ff > d_model else d_ff, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Half-step FFN
        if self.pre_norm:
            x = x + 0.5 * self.dropout(self.ffn1(self.norm1(x)))
            x = x + self.dropout(self.attn(self.norm2(x), mask=mask))
            x = x + 0.5 * self.dropout(self.ffn2(self.norm3(x)))
        else:
            x = self.norm1(x + 0.5 * self.dropout(self.ffn1(x)))
            x = self.norm2(x + self.dropout(self.attn(x, mask=mask)))
            x = self.norm3(x + 0.5 * self.dropout(self.ffn2(x)))
        return x


class SandwichTransformerBlock(nn.Module):
    """Sandwich transformer: LayerNorm both before and after sublayers.

    Uses a combination of pre-norm and post-norm to improve optimization.

    Reference: Press et al., "Improving Transformer Models by Reordering
    Their Sublayers" (ACL 2020)

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        d_ff: FFN hidden dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.pre_attn_norm = nn.LayerNorm(d_model)
        self.post_attn_norm = nn.LayerNorm(d_model)
        self.pre_ffn_norm = nn.LayerNorm(d_model)
        self.post_ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Sandwich norm for attention
        h = self.pre_attn_norm(x)
        h = self.attn(h, mask=mask)
        h = self.dropout(h)
        x = self.post_attn_norm(x + h)
        # Sandwich norm for FFN
        h = self.pre_ffn_norm(x)
        h = self.ffn(h)
        h = self.dropout(h)
        x = self.post_ffn_norm(x + h)
        return x


class ParallelTransformerBlock(nn.Module):
    """Parallel (GPT-J style) transformer: attention and FFN run in parallel.

    Computes attention and FFN simultaneously on the same normalized input,
    then adds both outputs to the residual. This reduces communication
    overhead in model parallel settings.

    Reference: Wang et al., "Language Modeling with Gated Convolutional
    Networks" + GPT-J parallel transformer design.

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        d_ff: FFN hidden dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm(x)
        # Parallel computation
        attn_out = self.dropout(self.attn(h, mask=mask))
        ffn_out = self.dropout(self.ffn(h))
        return x + attn_out + ffn_out


class UniversalTransformerBlock(nn.Module):
    """Universal Transformer block with adaptive computation time.

    Applies the same transformer block recurrently with a halting
    mechanism. Each token can halt at a different step.

    Reference: Dehghani et al., "Universal Transformers" (ICLR 2019)

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        d_ff: FFN hidden dimension
        max_steps: Maximum recurrence steps
        threshold: Halting threshold for ACT
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_steps: int = 8,
        threshold: float = 0.99,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.threshold = threshold
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.halt = nn.Linear(d_model, 1)
        self.pos_emb = nn.Embedding(max_steps, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: (B, T, D) refined representation
            ponder_cost: scalar ACT computation cost for regularization
        """
        B, T, D = x.shape
        halting_prob = torch.zeros(B, T, device=x.device)
        remainder = torch.ones(B, T, device=x.device)
        output = torch.zeros_like(x)
        ponder_cost = torch.zeros(B, device=x.device)

        for step in range(self.max_steps):
            # Add step-specific positional embedding
            step_emb = self.pos_emb(torch.full((1,), step, device=x.device, dtype=torch.long))
            h = x + step_emb

            # Standard transformer step
            h = self.norm(h)
            h = x + self.dropout(self.attn(h, mask=mask))
            h = h + self.dropout(self.ffn(self.norm(h)))

            # Compute halting probabilities
            p = torch.sigmoid(self.halt(h).squeeze(-1))  # (B, T)

            # ACT update
            still_running = (halting_prob < self.threshold).float()
            new_halted = (halting_prob + p * still_running >= self.threshold).float() * still_running
            still_running_after = (halting_prob + p * still_running < self.threshold).float() * still_running

            halting_prob = halting_prob + p * still_running
            remainder_new = remainder - new_halted * remainder
            update_weights = p * still_running_after + new_halted * remainder

            output = output + update_weights.unsqueeze(-1) * h
            ponder_cost = ponder_cost + still_running.mean(dim=-1)

            remainder = remainder_new
            x = h  # Update state

            if still_running_after.sum() == 0:
                break

        return output, ponder_cost.mean()


class HopfieldTransformerBlock(nn.Module):
    """Hopfield Network-inspired transformer block for pattern completion.

    Uses modern Hopfield networks as the attention mechanism, which
    achieves exponential storage capacity compared to classical Hopfield.
    Useful for retrieving stored market patterns.

    Reference: Ramsauer et al., "Hopfield Networks is All You Need" (ICLR 2021)

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads (here: number of Hopfield heads)
        beta: Hopfield network inverse temperature
        d_ff: FFN hidden dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        beta: float = 1.0,
        d_ff: int = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.beta = beta
        # Query, stored patterns (keys), value projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(h).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(h).view(B, T, H, d).transpose(1, 2)
        # Hopfield attention: softmax(beta * Q @ K^T / sqrt(d)) @ V
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.beta * d ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(self.dropout(attn), v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.dropout(self.out_proj(out))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class RetNetBlock(nn.Module):
    """Retention Network block (simplified) for O(1) inference.

    RetNet replaces attention with a retention mechanism that has:
    - Training: O(n) parallel mode (efficient GPU utilization)
    - Inference: O(1) recurrent mode (constant memory)

    Reference: Sun et al., "Retentive Network: A Successor to Transformer
    for Large Language Models" (2023)

    Args:
        d_model: Embedding dimension
        num_heads: Number of retention heads
        d_ff: FFN hidden dimension
        gamma: Decay factor for retention (per head)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        gamma: float = 0.9,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.gamma = gamma
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)  # Gating
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.sub_norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # Per-head decay rates
        self.gammas = nn.Parameter(
            torch.log(torch.tensor([gamma ** (i + 1) for i in range(num_heads)]))
        )

    def _retention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Parallel retention computation."""
        B, H, T, d = q.shape
        # Retention matrix D[m,n] = gamma^(m-n) if m >= n else 0
        pos = torch.arange(T, device=q.device, dtype=torch.float32)
        decay = torch.exp(self.gammas).view(H, 1, 1)  # (H, 1, 1)
        # D[i, j] = decay^(i-j) for i >= j
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (T, T)
        D = torch.where(
            diff >= 0,
            torch.pow(decay, diff.unsqueeze(0)),  # (H, T, T)
            torch.zeros(1, device=q.device),
        )
        # Retention: (Q @ K^T) * D @ V
        attn = torch.matmul(q, k.transpose(-2, -1)) * (d ** -0.5)  # (B, H, T, T)
        attn = attn * D.unsqueeze(0)
        return torch.matmul(attn, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(h).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(h).view(B, T, H, d).transpose(1, 2)
        ret = self._retention(q, k, v).transpose(1, 2).contiguous().view(B, T, D)
        ret = self.sub_norm(ret)
        g = F.silu(self.g_proj(h))
        ret = ret * g
        x = x + self.dropout(self.out_proj(ret))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class MambaBlock(nn.Module):
    """Simplified Mamba-style selective state space model block.

    Implements a simplified version of the S4/Mamba SSM for efficient
    sequence modeling with selective state updates. The selectivity
    mechanism allows the model to focus on relevant input features.

    Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces" (2023)

    Args:
        d_model: Embedding dimension
        d_state: SSM state dimension
        d_conv: Convolution width for depthwise conv
        expand: Expansion factor for inner dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        # Depthwise conv for causal local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner
        )
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # (B, T, d_inner) each
        # Conv1d (causal)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        # SSM parameters from input (selective)
        ssm_in = self.x_proj(x_conv)  # (B, T, d_state*2 + d_inner)
        B_proj, C_proj, dt = ssm_in.split([self.d_state, self.d_state, self.d_inner], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, T, d_inner) positive
        A = -torch.exp(self.A_log)  # (d_inner, d_state) negative
        # Simplified: discretize and apply SSM via scan (sequential for correctness)
        # In practice use parallel scan; here use loop for simplicity
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t_idx in range(T):
            dA = torch.exp(dt[:, t_idx].unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
            dB = dt[:, t_idx].unsqueeze(-1) * B_proj[:, t_idx].unsqueeze(1)  # (B, d_inner, d_state)
            h = h * dA + dB * x_conv[:, t_idx].unsqueeze(-1)
            y_t = (h * C_proj[:, t_idx].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, T, d_inner)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return residual + self.dropout(self.out_proj(y))


class TransformerWithCrossAttention(nn.Module):
    """Transformer encoder block with optional cross-attention to context.

    Supports both self-attention (standard encoder) and cross-attention
    to an external memory/context sequence. Useful for conditioning
    the financial model on news, fundamentals, or macro context.

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        d_ff: FFN hidden dimension
        dropout: Dropout probability
        cross_attend: Whether to include cross-attention sublayer
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        cross_attend: bool = True,
    ) -> None:
        super().__init__()
        self.cross_attend = cross_attend
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        if cross_attend:
            self.cross_attn = CrossAttention(d_model, num_heads, dropout=dropout)
            self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=self_mask))
        # Cross-attention (if enabled and context provided)
        if self.cross_attend and context is not None:
            x = x + self.dropout(self.cross_attn(self.norm2(x), context, mask=cross_mask))
        # FFN
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


# =============================================================================
# SECTION: Transformer Architectures for Financial Time Series
# =============================================================================

class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer (TFT) for multi-horizon forecasting.

    Implements key TFT components:
    - Gated Residual Networks (GRN) for non-linear processing
    - Variable Selection Networks (VSN) for input importance
    - Temporal self-attention with interpretable attention weights
    - Quantile output for uncertainty estimation

    Reference: Lim et al., "Temporal Fusion Transformers for Interpretable
    Multi-horizon Time Series Forecasting" (2021)

    Args:
        num_features: Number of input features
        d_model: Model dimension
        num_heads: Attention heads
        num_layers: Number of transformer layers
        forecast_horizon: Number of future timesteps to predict
        quantiles: Quantile levels for probabilistic output
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_features: int,
        d_model: int,
        num_heads: int,
        num_layers: int = 3,
        forecast_horizon: int = 5,
        quantiles: Optional[List[float]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]

        # Input projection
        self.input_proj = nn.Linear(num_features, d_model)

        # Variable Selection Network
        self.vsn = nn.Sequential(
            nn.Linear(num_features, num_features * 2),
            nn.ELU(),
            nn.Linear(num_features * 2, num_features),
            nn.Softmax(dim=-1),
        )
        self.vsn_proj = nn.Linear(num_features, d_model)

        # Gated Residual Networks
        self.grn_layers = nn.ModuleList([
            GatedResidualNetwork(d_model, d_model, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Temporal self-attention
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.grn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # Output head: quantile regression
        self.output_head = nn.Linear(d_model, forecast_horizon * len(self.quantiles))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input features (B, T, num_features)
        Returns:
            Dict with 'predictions' (B, forecast_horizon, num_quantiles)
            and 'variable_importance' (B, num_features)
        """
        B, T, F = x.shape

        # Variable selection
        var_weights = self.vsn(x.mean(dim=1))  # (B, F) — global importance
        x_weighted = x * var_weights.unsqueeze(1)  # (B, T, F)
        x_proj = self.vsn_proj(x_weighted) + self.input_proj(x)  # (B, T, D)

        h = self.dropout(x_proj)

        # Stacked GRN + attention
        for grn, attn, anorm, gnorm in zip(
            self.grn_layers, self.attention_layers, self.attn_norms, self.grn_norms
        ):
            h = h + self.dropout(attn(anorm(h)))
            h = h + self.dropout(grn(gnorm(h)))

        # Decode: use last forecast_horizon positions or pool
        out = self.output_head(h[:, -self.forecast_horizon:, :])  # (B, H, Q*F_H)
        predictions = out.view(B, self.forecast_horizon, len(self.quantiles))

        return {
            "predictions": predictions,
            "variable_importance": var_weights,
            "encoded": h,
        }


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network component from TFT.

    Provides non-linear processing with gating and residual connection.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (defaults to input_dim)
        dropout: Dropout probability
        context_dim: Optional context dimension for conditioning
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim * 2)  # For GLU gating
        self.context_proj = nn.Linear(context_dim, hidden_dim) if context_dim else None
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.residual_proj = None
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x if self.residual_proj is None else self.residual_proj(x)
        h = F.elu(self.fc1(x))
        if context is not None and self.context_proj is not None:
            if context.dim() == 2:
                context = context.unsqueeze(1)
            h = h + self.context_proj(context)
        h = self.dropout(h)
        h = self.fc2(h)
        # GLU gating
        h, gate = h.chunk(2, dim=-1)
        h = h * torch.sigmoid(gate)
        return self.norm(residual + h)


class N_BEATSBlock(nn.Module):
    """N-BEATS building block for interpretable time series decomposition.

    Each block produces backcast (reconstruction of input) and
    forecast (prediction of future) by learning basis expansion
    coefficients on learned or fixed basis functions.

    Reference: Oreshkin et al., "N-BEATS: Neural basis expansion analysis
    for interpretable time series forecasting" (ICLR 2020)

    Args:
        backcast_len: Length of the input sequence
        forecast_len: Length of the forecast horizon
        num_layers: Number of fully connected layers
        hidden_dim: Hidden layer width
        basis_type: Type of basis ('generic', 'trend', 'seasonality')
        num_harmonics: Number of harmonics for seasonality basis
        poly_degree: Polynomial degree for trend basis
    """

    def __init__(
        self,
        backcast_len: int,
        forecast_len: int,
        num_layers: int = 4,
        hidden_dim: int = 256,
        basis_type: str = "generic",
        num_harmonics: int = 4,
        poly_degree: int = 3,
    ) -> None:
        super().__init__()
        self.backcast_len = backcast_len
        self.forecast_len = forecast_len
        self.basis_type = basis_type

        # Fully connected stack
        layers = [nn.Linear(backcast_len, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.fc_stack = nn.Sequential(*layers)

        # Basis-specific expansion
        if basis_type == "generic":
            self.theta_b = nn.Linear(hidden_dim, backcast_len, bias=False)
            self.theta_f = nn.Linear(hidden_dim, forecast_len, bias=False)
        elif basis_type == "trend":
            self.p = poly_degree
            # Polynomial basis matrices
            t_back = torch.linspace(0, 1, backcast_len)
            t_fore = torch.linspace(1, 2, forecast_len)
            T_back = torch.stack([t_back ** i for i in range(poly_degree + 1)], dim=1)
            T_fore = torch.stack([t_fore ** i for i in range(poly_degree + 1)], dim=1)
            self.register_buffer("T_back", T_back)
            self.register_buffer("T_fore", T_fore)
            self.theta = nn.Linear(hidden_dim, poly_degree + 1, bias=False)
        elif basis_type == "seasonality":
            self.H = num_harmonics
            t_back = torch.linspace(0, 1, backcast_len)
            t_fore = torch.linspace(1, 2, forecast_len)
            cos_back = torch.cat([torch.cos(2 * 3.14159 * h * t_back).unsqueeze(1)
                                   for h in range(1, num_harmonics + 1)], dim=1)
            sin_back = torch.cat([torch.sin(2 * 3.14159 * h * t_back).unsqueeze(1)
                                   for h in range(1, num_harmonics + 1)], dim=1)
            cos_fore = torch.cat([torch.cos(2 * 3.14159 * h * t_fore).unsqueeze(1)
                                   for h in range(1, num_harmonics + 1)], dim=1)
            sin_fore = torch.cat([torch.sin(2 * 3.14159 * h * t_fore).unsqueeze(1)
                                   for h in range(1, num_harmonics + 1)], dim=1)
            S_back = torch.cat([cos_back, sin_back], dim=1)  # (T_back, 2H)
            S_fore = torch.cat([cos_fore, sin_fore], dim=1)  # (T_fore, 2H)
            self.register_buffer("S_back", S_back)
            self.register_buffer("S_fore", S_fore)
            self.theta = nn.Linear(hidden_dim, 2 * num_harmonics, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input (B, backcast_len)
        Returns:
            backcast: (B, backcast_len)
            forecast: (B, forecast_len)
        """
        h = self.fc_stack(x)  # (B, hidden_dim)
        if self.basis_type == "generic":
            backcast = self.theta_b(h)
            forecast = self.theta_f(h)
        elif self.basis_type == "trend":
            theta = self.theta(h)  # (B, p+1)
            backcast = torch.matmul(theta, self.T_back.T)  # (B, T_back)
            forecast = torch.matmul(theta, self.T_fore.T)  # (B, T_fore)
        elif self.basis_type == "seasonality":
            theta = self.theta(h)  # (B, 2H)
            backcast = torch.matmul(theta, self.S_back.T)  # (B, T_back)
            forecast = torch.matmul(theta, self.S_fore.T)  # (B, T_fore)
        return backcast, forecast


class PatchTSTBlock(nn.Module):
    """PatchTST patch-based transformer for time series classification/regression.

    Divides time series into patches, projects to d_model, then applies
    standard transformer blocks. Achieves strong performance on time
    series benchmarks with channel-independence.

    Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term
    Forecasting with Transformers" (ICLR 2023)

    Args:
        seq_len: Total sequence length
        patch_size: Size of each patch
        stride: Stride between patches
        d_model: Model dimension
        num_heads: Attention heads
        num_layers: Transformer depth
        d_ff: FFN dimension
        num_features: Input feature dimension
        forecast_len: Forecast horizon
        dropout: Dropout probability
    """

    def __init__(
        self,
        seq_len: int,
        patch_size: int = 16,
        stride: int = 8,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 256,
        num_features: int = 5,
        forecast_len: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_features = num_features
        self.forecast_len = forecast_len
        # Number of patches
        self.num_patches = (seq_len - patch_size) // stride + 1
        # Patch embedding: linear projection
        self.patch_embed = nn.Linear(patch_size, d_model, bias=False)
        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        # Transformer encoder
        self.layers = nn.ModuleList([
            SandwichTransformerBlock(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Channel-wise prediction head
        self.head = nn.Linear(d_model * self.num_patches, forecast_len)
        self.dropout = nn.Dropout(dropout)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert sequence to patches.
        Args:
            x: (B, T) single-channel
        Returns:
            (B, num_patches, patch_size)
        """
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            patches.append(x[:, start:start + self.patch_size])
        return torch.stack(patches, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) multi-channel time series
        Returns:
            predictions: (B, forecast_len, C)
        """
        B, T, C = x.shape
        outputs = []
        for c in range(C):
            xc = x[:, :, c]  # (B, T)
            patches = self._patchify(xc)  # (B, P, patch_size)
            h = self.patch_embed(patches) + self.pos_embed  # (B, P, d_model)
            h = self.dropout(h)
            for layer in self.layers:
                h = layer(h)
            h_flat = h.reshape(B, -1)  # (B, P * d_model)
            pred_c = self.head(h_flat)  # (B, forecast_len)
            outputs.append(pred_c)
        return torch.stack(outputs, dim=-1)  # (B, forecast_len, C)


# =============================================================================
# SECTION: Transformer Utilities
# =============================================================================

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal (autoregressive) attention mask.

    Args:
        seq_len: Sequence length
        device: Target device
    Returns:
        Additive mask (T, T) with -inf for future positions
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    return mask


def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Create padding mask from sequence lengths.

    Args:
        lengths: (B,) tensor of actual sequence lengths
        max_len: Maximum sequence length
    Returns:
        Additive mask (B, 1, 1, max_len) with 0 for valid, -inf for padded
    """
    B = lengths.size(0)
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # (1, max_len)
    mask = (positions >= lengths.unsqueeze(1)).float()  # (B, max_len)
    mask = mask * float("-inf")
    mask = mask.view(B, 1, 1, max_len)
    return mask


def sinusoidal_init(embedding: nn.Embedding, max_len: int, d_model: int) -> None:
    """Initialize an embedding with sinusoidal position encodings.

    Args:
        embedding: nn.Embedding to initialize
        max_len: Maximum sequence length
        d_model: Embedding dimension
    """
    import math
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
    with torch.no_grad():
        embedding.weight.copy_(pe)


def get_parameter_groups_for_optimizer(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_patterns: Optional[List[str]] = None,
) -> List[Dict]:
    """Create parameter groups with selective weight decay.

    Excludes bias and LayerNorm parameters from weight decay, which
    is standard practice for transformer training.

    Args:
        model: PyTorch model
        weight_decay: Weight decay for decay group
        no_decay_patterns: Substring patterns that skip weight decay
    Returns:
        List of parameter group dicts for optimizer
    """
    if no_decay_patterns is None:
        no_decay_patterns = ["bias", "norm", "LayerNorm", "layer_norm"]

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(pat in name for pat in no_decay_patterns):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def freeze_layers(model: nn.Module, num_layers_to_freeze: int) -> int:
    """Freeze the first N transformer layers for fine-tuning.

    Args:
        model: Model with attribute 'layers' (ModuleList)
        num_layers_to_freeze: Number of layers to freeze from the bottom
    Returns:
        Number of frozen parameters
    """
    frozen = 0
    if not hasattr(model, "layers"):
        return frozen
    for i, layer in enumerate(model.layers):
        if i < num_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
                frozen += param.numel()
    return frozen


def get_attention_patterns(
    model: nn.Module,
    x: torch.Tensor,
    layer_indices: Optional[List[int]] = None,
) -> List[torch.Tensor]:
    """Extract attention weight patterns from specified layers.

    Registers forward hooks to capture attention weights during inference.

    Args:
        model: Transformer model with attention sub-modules
        x: Input tensor
        layer_indices: Which layers to extract (None = all)
    Returns:
        List of attention weight tensors
    """
    attention_weights = []
    hooks = []

    def make_hook(idx):
        def hook(module, inputs, output):
            # Assumes output is (attn_out, attn_weights) or just attn_out
            if isinstance(output, tuple) and len(output) == 2:
                attention_weights.append(output[1].detach())
        return hook

    # Find attention modules
    attn_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (MultiHeadSelfAttention, GroupedQueryAttention)):
            attn_modules.append((name, module))

    if layer_indices is not None:
        attn_modules = [attn_modules[i] for i in layer_indices if i < len(attn_modules)]

    for idx, (name, module) in enumerate(attn_modules):
        hooks.append(module.register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return attention_weights


class TransformerForSequenceClassification(nn.Module):
    """Transformer encoder for sequence-level classification.

    Applies a transformer backbone and pools the output to produce
    sequence-level predictions (e.g., market regime, direction).

    Args:
        num_classes: Number of output classes
        d_model: Model dimension
        num_heads: Attention heads
        num_layers: Transformer depth
        d_ff: FFN dimension
        max_seq_len: Maximum input sequence length
        num_features: Input feature count
        pooling: Pooling strategy ('cls', 'mean', 'max', 'last')
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int,
        num_heads: int,
        num_layers: int = 4,
        d_ff: int = None,
        max_seq_len: int = 512,
        num_features: int = 5,
        pooling: str = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.pooling = pooling
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, F = x.shape
        h = self.input_proj(x)
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            h = torch.cat([cls, h], dim=1)
            cls_pos = torch.zeros(B, 1, dtype=torch.long, device=x.device)
            pos_ids = torch.cat([cls_pos, pos_ids], dim=1)
        h = self.dropout(h + self.pos_embed(pos_ids))
        for layer in self.layers:
            h = layer(h, mask=mask)
        h = self.norm(h)
        if self.pooling == "cls":
            pooled = h[:, 0, :]
        elif self.pooling == "mean":
            pooled = h.mean(dim=1)
        elif self.pooling == "max":
            pooled = h.max(dim=1).values
        elif self.pooling == "last":
            pooled = h[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.head(pooled)


_NEW_TRANSFORMER_EXPORTS = [
    "MixedActivationFFN", "ExpertFFN", "ConvFFN", "HyperNetwork",
    "ScaleNorm", "AdaptiveLayerNorm", "CRMSNorm", "GroupNorm1D",
    "MacaronTransformerBlock", "SandwichTransformerBlock", "ParallelTransformerBlock",
    "UniversalTransformerBlock", "HopfieldTransformerBlock", "RetNetBlock",
    "MambaBlock", "TransformerWithCrossAttention",
    "TemporalFusionTransformer", "GatedResidualNetwork", "N_BEATSBlock", "PatchTSTBlock",
    "make_causal_mask", "make_padding_mask", "sinusoidal_init",
    "get_parameter_groups_for_optimizer", "freeze_layers",
    "get_attention_patterns", "TransformerForSequenceClassification",
]
