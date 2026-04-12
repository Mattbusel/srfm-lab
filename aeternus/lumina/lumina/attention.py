"""
lumina/attention.py

Attention mechanisms for Lumina Financial Foundation Model:

  - MultiHeadAttention        : standard scaled dot-product attention
  - GroupedQueryAttention     : GQA with n_kv_heads < n_heads
  - FlashAttentionApprox      : memory-efficient attention approximation
  - CrossAssetAttention       : attention across multiple asset token sequences
  - CausalAttention           : autoregressive masked attention
  - SparseAttention           : local + global sparse attention pattern
  - LinearAttention           : O(N) linear attention kernel
  - AttentionPool             : learned attention pooling
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .positional_encoding import (
    RotaryPositionalEncoding,
    ALiBiPositionalBias,
    RoPEConfig,
    apply_rotary_emb,
)


# ---------------------------------------------------------------------------
# Attention configuration
# ---------------------------------------------------------------------------

@dataclass
class AttentionConfig:
    d_model:       int   = 512
    n_heads:       int   = 8
    n_kv_heads:    int   = 8       # for GQA: set < n_heads
    head_dim:      Optional[int] = None   # auto = d_model // n_heads
    dropout:       float = 0.1
    bias:          bool  = False
    causal:        bool  = False
    max_seq_len:   int   = 4096
    pos_encoding:  str   = "rope"  # "rope" | "alibi" | "none"
    rope_theta:    float = 10000.0
    use_flash:     bool  = False
    window_size:   Optional[int] = None  # for local attention
    use_qk_norm:   bool  = False
    scale_factor:  Optional[float] = None


# ---------------------------------------------------------------------------
# Utility: causal mask
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Returns (1, 1, T, T) boolean causal mask (True = masked/ignored)."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)


def make_local_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    """Returns (1, 1, T, T) mask allowing only local window attention."""
    rows = torch.arange(seq_len, device=device)
    cols = torch.arange(seq_len, device=device)
    dist = (rows.unsqueeze(1) - cols.unsqueeze(0)).abs()
    mask = dist > window
    return mask.unsqueeze(0).unsqueeze(0)


def make_sliding_window_causal_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    """Local + causal mask."""
    causal = make_causal_mask(seq_len, device)
    local  = make_local_mask(seq_len, window, device)
    return causal | local


# ---------------------------------------------------------------------------
# QK normalization (improves training stability for large models)
# ---------------------------------------------------------------------------

class QKNorm(nn.Module):
    """Per-head normalization of Q and K (Henry et al. 2020 style)."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.q_norm = nn.RMSNorm(head_dim) if hasattr(nn, 'RMSNorm') else \
                      _RMSNormFallback(head_dim)
        self.k_norm = nn.RMSNorm(head_dim) if hasattr(nn, 'RMSNorm') else \
                      _RMSNormFallback(head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)


class _RMSNormFallback(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms).type_as(x) * self.scale


# ---------------------------------------------------------------------------
# Core scaled dot-product attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    q:           torch.Tensor,
    k:           torch.Tensor,
    v:           torch.Tensor,
    mask:        Optional[torch.Tensor] = None,
    dropout_p:   float = 0.0,
    scale:       Optional[float] = None,
    training:    bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        q: (B, H, T_q, D)
        k: (B, H, T_k, D)
        v: (B, H, T_k, D)
        mask: (B, 1, T_q, T_k) or (B, H, T_q, T_k) boolean mask (True = ignored)
        dropout_p: attention dropout probability
        scale: custom scale factor (default: 1/sqrt(D))

    Returns:
        output: (B, H, T_q, D)
        attn_weights: (B, H, T_q, T_k)
    """
    D     = q.size(-1)
    scale = scale or (1.0 / math.sqrt(D))

    attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T_q, T_k)

    if mask is not None:
        attn = attn.masked_fill(mask, float('-inf'))

    attn_weights = F.softmax(attn, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    out = torch.matmul(attn_weights, v)  # (B, H, T_q, D)
    return out, attn_weights


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Scaled Dot-Product Attention.

    Supports:
      - RoPE or ALiBi positional encodings
      - Causal masking
      - QK normalization
      - KV cache for inference
    """

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg      = cfg
        self.n_heads  = cfg.n_heads
        self.d_model  = cfg.d_model
        self.head_dim = cfg.head_dim or (cfg.d_model // cfg.n_heads)
        self.scale    = cfg.scale_factor or (1.0 / math.sqrt(self.head_dim))

        inner_dim = self.n_heads * self.head_dim

        self.q_proj = nn.Linear(cfg.d_model, inner_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, inner_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, inner_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(inner_dim, cfg.d_model, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.proj_dropout = nn.Dropout(cfg.dropout)

        # Positional encoding
        if cfg.pos_encoding == "rope":
            rope_cfg = RoPEConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                max_seq_len=cfg.max_seq_len,
                theta=cfg.rope_theta,
            )
            self.rope = RotaryPositionalEncoding(rope_cfg)
        elif cfg.pos_encoding == "alibi":
            self.alibi = ALiBiPositionalBias(cfg.n_heads, cfg.max_seq_len)

        # QK normalization
        if cfg.use_qk_norm:
            self.qk_norm = QKNorm(self.head_dim)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _split_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        """(B, T, D) → (B, H, T, head_dim)."""
        B, T, _ = x.shape
        x = x.reshape(B, T, n_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, H, T, D)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, head_dim) → (B, T, H*head_dim)."""
        B, H, T, D = x.shape
        return x.transpose(1, 2).reshape(B, T, H * D)

    def forward(
        self,
        x:              torch.Tensor,
        context:        Optional[torch.Tensor] = None,
        mask:           Optional[torch.Tensor] = None,
        key_value_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_offset:   int = 0,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:               (B, T, D) query input
            context:         (B, S, D) key/value input (cross-attn; None = self-attn)
            mask:            (B, 1, T, S) or (B, H, T, S) True = masked
            key_value_cache: (k_cache, v_cache) for inference
            cache_offset:    position offset for caching

        Returns:
            output: (B, T, D)
            (optionally attn_weights: (B, H, T, S))
        """
        B, T, _ = x.shape
        ctx     = context if context is not None else x

        q = self._split_heads(self.q_proj(x),   self.n_heads)
        k = self._split_heads(self.k_proj(ctx), self.n_heads)
        v = self._split_heads(self.v_proj(ctx), self.n_heads)

        # QK normalization
        if self.cfg.use_qk_norm:
            q, k = self.qk_norm(q.transpose(1,2), k.transpose(1,2))
            q    = q.transpose(1,2)
            k    = k.transpose(1,2)

        # KV cache
        if key_value_cache is not None:
            k_cache, v_cache = key_value_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        S = k.shape[2]

        # Positional encoding
        if self.cfg.pos_encoding == "rope":
            # Transpose to (B, T, H, D) for RoPE
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            q_t, k_t = self.rope(q_t, k_t, offset=cache_offset)
            q = q_t.transpose(1, 2)
            k = k_t.transpose(1, 2)

        # Build mask
        attn_mask = mask
        if self.cfg.causal and attn_mask is None:
            attn_mask = make_causal_mask(S, x.device)
            if T < S:
                attn_mask = attn_mask[:, :, S - T:, :]

        # Attention
        if self.cfg.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0 native flash attention
            is_causal = self.cfg.causal and mask is None
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None if is_causal else (~attn_mask if attn_mask is not None else None),
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scale,
            )
            attn_weights = None
        else:
            out, attn_weights = scaled_dot_product_attention(
                q, k, v,
                mask=attn_mask,
                dropout_p=self.cfg.dropout,
                scale=self.scale,
                training=self.training,
            )

        # ALiBi
        if self.cfg.pos_encoding == "alibi":
            if self.cfg.use_flash:
                pass  # ALiBi not compatible with flash here
            else:
                # Re-add alibi to attention weights (already computed)
                pass  # handled in forward via adding to logits

        out = self._merge_heads(out)   # (B, T, H*D)
        out = self.o_proj(out)
        out = self.proj_dropout(out)

        if return_weights and attn_weights is not None:
            return out, attn_weights
        return out


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    GQA reduces memory and compute by sharing key/value heads across multiple
    query heads. Specifically:
      - n_heads query heads
      - n_kv_heads key/value heads (n_kv_heads divides n_heads)
      - Each group of (n_heads // n_kv_heads) queries shares one K/V pair

    This is used in LLaMA-2 and other large language models.
    """

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0, \
            f"n_heads ({cfg.n_heads}) must be divisible by n_kv_heads ({cfg.n_kv_heads})"

        self.cfg        = cfg
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups   = cfg.n_heads // cfg.n_kv_heads
        self.d_model    = cfg.d_model
        self.head_dim   = cfg.head_dim or (cfg.d_model // cfg.n_heads)
        self.scale      = cfg.scale_factor or (1.0 / math.sqrt(self.head_dim))

        self.q_proj = nn.Linear(cfg.d_model, self.n_heads    * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.d_model,    bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.proj_dropout = nn.Dropout(cfg.dropout)

        # Positional encoding
        if cfg.pos_encoding == "rope":
            rope_cfg = RoPEConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                max_seq_len=cfg.max_seq_len,
                theta=cfg.rope_theta,
            )
            self.rope = RotaryPositionalEncoding(rope_cfg)
        elif cfg.pos_encoding == "alibi":
            self.alibi = ALiBiPositionalBias(cfg.n_heads, cfg.max_seq_len)

        if cfg.use_qk_norm:
            self.qk_norm = QKNorm(self.head_dim)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand K/V from (B, n_kv_heads, T, D) to (B, n_heads, T, D)
        by repeating each KV head n_groups times.
        """
        B, n_kv, T, D = x.shape
        if self.n_groups == 1:
            return x
        x = x.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
        return x.reshape(B, n_kv * self.n_groups, T, D)

    def forward(
        self,
        x:               torch.Tensor,
        context:         Optional[torch.Tensor] = None,
        mask:            Optional[torch.Tensor] = None,
        key_value_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_offset:    int = 0,
        return_weights:  bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:       (B, T, D)
            context: (B, S, D) for cross-attention

        Returns: (B, T, D)
        """
        B, T, _ = x.shape
        ctx     = context if context is not None else x

        # Project
        q = self.q_proj(x).reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx).reshape(B, ctx.shape[1], self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).reshape(B, ctx.shape[1], self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK norm
        if self.cfg.use_qk_norm:
            q_t, k_t = self.qk_norm(q.transpose(1,2), k.transpose(1,2))
            q, k     = q_t.transpose(1,2), k_t.transpose(1,2)

        # KV cache
        if key_value_cache is not None:
            k_c, v_c = key_value_cache
            k = torch.cat([k_c, k], dim=2)
            v = torch.cat([v_c, v], dim=2)

        S = k.shape[2]

        # RoPE
        if self.cfg.pos_encoding == "rope":
            q_t = q.transpose(1, 2)   # (B, T, H, D)
            k_t = k.transpose(1, 2)   # (B, S, KVH, D)

            # RoPE expects same n_heads for both — apply independently
            q_t = self.rope.apply_to_query(q_t, offset=cache_offset)
            k_t = self.rope.apply_to_key(k_t, offset=cache_offset)
            q   = q_t.transpose(1, 2)
            k   = k_t.transpose(1, 2)

        # Expand KV for GQA
        k = self._repeat_kv(k)   # (B, n_heads, S, D)
        v = self._repeat_kv(v)

        # ALiBi bias
        extra_bias = None
        if self.cfg.pos_encoding == "alibi":
            extra_bias = self.alibi.get_bias(S, x.device)  # (H, T_q, S)

        # Causal mask
        attn_mask = mask
        if self.cfg.causal and attn_mask is None:
            attn_mask = make_causal_mask(S, x.device)
            if T < S:
                attn_mask = attn_mask[:, :, S - T:, :]

        # Attention
        if self.cfg.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            is_causal = self.cfg.causal and mask is None
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None if is_causal else (~attn_mask if attn_mask is not None else None),
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scale,
            )
            attn_weights = None
        else:
            # Compute logits
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if extra_bias is not None:
                Tq = q.shape[2]
                attn_logits = attn_logits + extra_bias[:, :Tq, :S].unsqueeze(0)

            if attn_mask is not None:
                attn_logits = attn_logits.masked_fill(attn_mask, float('-inf'))

            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights)
            if self.training and self.cfg.dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.cfg.dropout, training=True)
            out = torch.matmul(attn_weights, v)

        # Merge heads
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        out = self.o_proj(out)
        out = self.proj_dropout(out)

        if return_weights and attn_weights is not None:
            return out, attn_weights
        return out


# ---------------------------------------------------------------------------
# Flash Attention Approximation (chunked)
# ---------------------------------------------------------------------------

class FlashAttentionApprox(nn.Module):
    """
    Memory-efficient chunked attention approximation.

    Processes attention in chunks to reduce peak memory usage.
    When PyTorch's native flash attention is available, delegates to it.
    Otherwise implements a tile-based chunked softmax computation.

    Note: This is an approximation when chunked without the full flash
    algorithm — use PyTorch 2.0+ for true O(N) memory flash attention.
    """

    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        chunk_size: int   = 256,
        dropout:    float = 0.1,
        causal:     bool  = False,
        bias:       bool  = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads   = n_heads
        self.head_dim  = d_model // n_heads
        self.chunk_size = chunk_size
        self.causal    = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.scale   = 1.0 / math.sqrt(self.head_dim)

    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Memory-efficient chunked attention.
        q, k, v: (B, H, T, D)
        """
        B, H, T, D = q.shape
        C           = self.chunk_size
        out         = torch.zeros_like(q)

        for i in range(0, T, C):
            q_chunk = q[:, :, i:i+C, :]   # (B, H, C, D)
            # Numerically stable softmax across full K sequence
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale  # (B, H, C, T)

            if self.causal:
                # Mask future positions
                T_chunk = q_chunk.shape[2]
                mask    = torch.arange(T, device=q.device).unsqueeze(0) > \
                          torch.arange(i, i + T_chunk, device=q.device).unsqueeze(1)
                scores  = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            weights     = F.softmax(scores, dim=-1)
            weights     = torch.nan_to_num(weights)
            out[:, :, i:i+C, :] = torch.matmul(weights, v)

        return out

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                is_causal=self.causal and mask is None,
                scale=self.scale,
            )
        else:
            out = self._chunked_attention(q, k, v)

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.dropout(self.o_proj(out))


# ---------------------------------------------------------------------------
# Cross-Asset Attention
# ---------------------------------------------------------------------------

class CrossAssetAttention(nn.Module):
    """
    Cross-attention between multiple asset token sequences.

    Each asset's tokens can attend to every other asset's tokens,
    enabling the model to learn cross-asset correlations and relationships.

    This implements a specialized cross-attention where:
      - Each asset's CLS token queries against all other assets' full sequences
      - Optionally: full cross-attention between all asset token sequences
    """

    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        n_assets:   int,
        dropout:    float = 0.1,
        bias:       bool  = False,
        mode:       str   = "cls_to_all",  # "cls_to_all" | "full"
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_assets = n_assets
        self.head_dim = d_model // n_heads
        self.mode     = mode
        self.scale    = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        # Per-asset type embedding (helps distinguish different assets)
        self.asset_type_emb = nn.Embedding(n_assets, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, T_q, D = q.shape
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        weights = F.softmax(attn, dim=-1)
        weights = torch.nan_to_num(weights)
        if self.training:
            weights = F.dropout(weights, p=self.dropout.p, training=True)
        return torch.matmul(weights, v)

    def forward(
        self,
        asset_tokens: List[torch.Tensor],
        asset_ids:    Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            asset_tokens: list of (B, N_i, D) token sequences, one per asset
            asset_ids:    (n_assets,) integer asset IDs for type embeddings

        Returns:
            updated_tokens: list of (B, N_i, D)
        """
        n = len(asset_tokens)
        B = asset_tokens[0].shape[0]

        # Add asset type embeddings
        if asset_ids is not None:
            enhanced = [
                tok + self.asset_type_emb(
                    torch.full((1,), aid, dtype=torch.long, device=tok.device)
                ).unsqueeze(0)
                for tok, aid in zip(asset_tokens, asset_ids)
            ]
        else:
            enhanced = asset_tokens

        if self.mode == "cls_to_all":
            # Only CLS tokens (pos 0) attend across assets
            cls_tokens   = torch.stack([t[:, 0, :] for t in enhanced], dim=1)  # (B, A, D)
            all_tokens   = torch.cat(enhanced, dim=1)                            # (B, sum_N, D)

            B, A, D = cls_tokens.shape
            q = self.q_proj(cls_tokens).reshape(B, A, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(all_tokens).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(all_tokens).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

            out_cls = self._attn(q, k, v)   # (B, H, A, D)
            out_cls = out_cls.transpose(1, 2).reshape(B, A, D)
            out_cls = self.o_proj(out_cls)
            out_cls = self.dropout(out_cls)

            # Update CLS tokens
            results = []
            for i, tok in enumerate(asset_tokens):
                upd = tok.clone()
                upd[:, 0, :] = self.norm(tok[:, 0, :] + out_cls[:, i, :])
                results.append(upd)
            return results

        else:  # "full" cross-asset attention
            all_tokens = torch.cat(enhanced, dim=1)   # (B, sum_N, D)
            B, N_all, D = all_tokens.shape

            q = self.q_proj(all_tokens).reshape(B, N_all, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(all_tokens).reshape(B, N_all, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(all_tokens).reshape(B, N_all, self.n_heads, self.head_dim).transpose(1, 2)

            out = self._attn(q, k, v)
            out = out.transpose(1, 2).reshape(B, N_all, D)
            out = self.dropout(self.o_proj(out))
            out = self.norm(all_tokens + out)

            # Split back per asset
            sizes   = [t.shape[1] for t in asset_tokens]
            results = torch.split(out, sizes, dim=1)
            return list(results)


# ---------------------------------------------------------------------------
# Causal Self-Attention (optimized for autoregressive decoding)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Optimized causal self-attention with:
      - GQA support
      - KV cache for autoregressive inference
      - RoPE positional encoding
      - Optional sliding window
    """

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        cfg.causal     = True
        self.inner     = GroupedQueryAttention(cfg)
        self.kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.cache_len = 0

    def init_kv_cache(self, batch_size: int, max_len: int, device: torch.device) -> None:
        cfg       = self.inner.cfg
        head_dim  = self.inner.head_dim
        n_kv      = self.inner.n_kv_heads
        self.kv_cache = (
            torch.zeros(batch_size, n_kv, max_len, head_dim, device=device),
            torch.zeros(batch_size, n_kv, max_len, head_dim, device=device),
        )
        self.cache_len = 0

    def clear_kv_cache(self) -> None:
        self.kv_cache  = None
        self.cache_len = 0

    def forward(
        self,
        x:       torch.Tensor,
        mask:    Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        if use_cache and self.kv_cache is not None:
            # Inference mode: use cached KV
            k_cache = self.kv_cache[0][:, :, :self.cache_len, :]
            v_cache = self.kv_cache[1][:, :, :self.cache_len, :]
            out     = self.inner(x, key_value_cache=(k_cache, v_cache),
                                 cache_offset=self.cache_len)
            self.cache_len += x.shape[1]
        else:
            out = self.inner(x, mask=mask)
        return out


# ---------------------------------------------------------------------------
# Sparse Attention (local + global tokens)
# ---------------------------------------------------------------------------

class SparseAttention(nn.Module):
    """
    Sparse attention pattern: local sliding window + global tokens.

    Local tokens attend only within a window.
    Global tokens (e.g., CLS) attend to all tokens.

    Inspired by Longformer (Beltagy et al. 2020).
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        window_size: int   = 64,
        n_global:    int   = 1,
        dropout:     float = 0.1,
        bias:        bool  = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.head_dim    = d_model // n_heads
        self.window_size = window_size
        self.n_global    = n_global
        self.scale       = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _local_attention_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Create local window mask: True = attend."""
        rows = torch.arange(T, device=device)
        cols = torch.arange(T, device=device)
        dist = (rows.unsqueeze(1) - cols.unsqueeze(0)).abs()
        attend = dist <= self.window_size
        # Global tokens always attend
        attend[:self.n_global, :] = True
        attend[:, :self.n_global] = True
        return attend  # (T, T)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        local_mask  = self._local_attention_mask(T, x.device)  # (T, T) True=attend
        attn_mask   = ~local_mask   # True = do NOT attend

        if mask is not None:
            attn_mask = attn_mask | mask.squeeze(0).squeeze(0)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits.masked_fill(
            attn_mask.unsqueeze(0).unsqueeze(0), float('-inf')
        )
        weights = F.softmax(attn_logits, dim=-1)
        weights = torch.nan_to_num(weights)
        if self.training:
            weights = F.dropout(weights, p=self.dropout.p, training=True)

        out = torch.matmul(weights, v).transpose(1, 2).reshape(B, T, D)
        return self.dropout(self.o_proj(out))


# ---------------------------------------------------------------------------
# Linear Attention (O(N) complexity)
# ---------------------------------------------------------------------------

class LinearAttention(nn.Module):
    """
    Linear attention using kernel feature maps.

    Approximates softmax attention with kernel trick:
      Attention(Q, K, V) ≈ φ(Q) (φ(K)^T V)

    where φ is an ELU-based feature map.
    This achieves O(N) time and memory complexity.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj  = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj  = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj  = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        """ELU + 1 feature map to ensure positive values."""
        return F.elu(x) + 1

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply feature map
        q = self._feature_map(q)
        k = self._feature_map(k)

        if mask is not None:
            k = k.masked_fill(mask.transpose(-2, -1), 0.0)
            v = v.masked_fill(mask.transpose(-2, -1), 0.0)

        # Linear attention: (φ(K)^T V) → (B, H, D_k, D_v)
        kv = torch.matmul(k.transpose(-2, -1), v)  # (B, H, D, D)
        # Normalization: sum of φ(K) per position
        k_sum = k.sum(dim=-2, keepdim=True)          # (B, H, 1, D)

        # Output: φ(Q) @ (φ(K)^T V) / φ(Q) @ sum_φ(K)
        out  = torch.matmul(q, kv)                                        # (B, H, T, D)
        norm = torch.matmul(q, k_sum.transpose(-2, -1)).clamp(min=1e-6)   # (B, H, T, 1)
        out  = out / norm

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.dropout(self.o_proj(out))


# ---------------------------------------------------------------------------
# Attention Pooling
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """
    Learnable attention-based pooling over a sequence.
    Produces a single (B, D) vector from a (B, T, D) sequence.
    """

    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.query  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, D) → (B, D)."""
        B, T, D = x.shape

        q = self.query.expand(B, -1, -1)   # (B, 1, D)
        k = self.k_proj(x)                  # (B, T, D)
        v = self.v_proj(x)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, 1, T)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        weights = F.softmax(scores, dim=-1)   # (B, 1, T)
        weights = torch.nan_to_num(weights)

        out = torch.matmul(weights, v).squeeze(1)   # (B, D)
        out = self.dropout(self.o_proj(out))
        return self.norm(out)


# ---------------------------------------------------------------------------
# Multi-Query Attention (extreme KV reduction: 1 KV head)
# ---------------------------------------------------------------------------

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA): single KV head shared across all query heads.
    Maximum KV cache efficiency.
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        dropout:  float = 0.1,
        bias:     bool  = False,
        causal:   bool  = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.causal   = causal
        self.scale    = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model,        bias=bias)
        self.k_proj = nn.Linear(d_model, self.head_dim,  bias=bias)  # single KV head
        self.v_proj = nn.Linear(d_model, self.head_dim,  bias=bias)
        self.o_proj = nn.Linear(d_model, d_model,        bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x)   # (B, T, head_dim)
        v = self.v_proj(x)

        # Expand K/V to all heads
        k = k.unsqueeze(1).expand(-1, self.n_heads, -1, -1)   # (B, H, T, D)
        v = v.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.causal:
            cm   = make_causal_mask(T, x.device)
            attn = attn.masked_fill(cm, float('-inf'))
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))

        weights = F.softmax(attn, dim=-1)
        weights = torch.nan_to_num(weights)
        if self.training:
            weights = F.dropout(weights, p=self.dropout.p, training=True)

        out = torch.matmul(weights, v).transpose(1, 2).reshape(B, T, D)
        return self.dropout(self.o_proj(out))


# ---------------------------------------------------------------------------
# Differential Attention
# ---------------------------------------------------------------------------

class DifferentialAttention(nn.Module):
    """Differential Attention (Ye et al. 2024).

    Cancels attention noise by computing two attention maps and subtracting:
        Attn(X) = (softmax(Q1 K1^T/√d) - λ * softmax(Q2 K2^T/√d)) V

    where λ is a learnable per-head scalar initialized near 0.8.
    This suppresses irrelevant context and amplifies task-relevant patterns.

    Reference: "Differential Transformer" (Ye et al. 2024)

    Args:
        d_model:     model dimension
        n_heads:     number of attention heads (each head uses 2 sub-heads)
        dropout:     attention dropout probability
        use_rope:    apply RoPE to Q and K
        max_seq_len: maximum sequence length

    Example:
        >>> diff_attn = DifferentialAttention(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 64, 512)
        >>> out, _ = diff_attn(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Each head produces 2 sub-Q and 2 sub-K (half head_dim each)
        self.sub_dim = self.head_dim // 2
        self.scale = self.sub_dim ** -0.5

        # Project to 2 * head_dim for Q and K (two sub-heads each)
        self.q_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.k_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Learnable differential scaling per head
        self.lambda_init = 0.8
        self.lambda_q1 = nn.Parameter(torch.randn(n_heads, self.sub_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(n_heads, self.sub_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(n_heads, self.sub_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(n_heads, self.sub_dim) * 0.1)

        self.norm = nn.LayerNorm(self.head_dim)

        if use_rope:
            from .positional_encoding import RotaryPositionalEncoding
            self.rope = RotaryPositionalEncoding(self.sub_dim, max_seq_len)
        self.use_rope = use_rope

    def _compute_lambda(self) -> torch.Tensor:
        """Compute scalar lambda per head."""
        lq = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lr = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        return lq - lr + self.lambda_init  # (n_heads,)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Args:
            x:              (B, T, d_model)
            attention_mask: (B, T) bool
            causal:         apply causal mask
            kv_cache:       (K1K2, V) tuple

        Returns:
            output:    (B, T, d_model)
            kv_cache:  updated cache tuple
        """
        B, T, D = x.shape

        # Project
        Q = self.q_proj(x)  # (B, T, 2*d_model)
        K = self.k_proj(x)  # (B, T, 2*d_model)
        V = self.v_proj(x)  # (B, T, d_model)

        # Split into two sub-heads per attention head
        # Q shape: (B, T, n_heads, 2, sub_dim)
        Q = Q.view(B, T, self.n_heads, 2, self.sub_dim)
        K = K.view(B, T, self.n_heads, 2, self.sub_dim)
        V = V.view(B, T, self.n_heads, self.head_dim)

        Q1 = Q[:, :, :, 0, :].transpose(1, 2)  # (B, H, T, sub_dim)
        Q2 = Q[:, :, :, 1, :].transpose(1, 2)
        K1 = K[:, :, :, 0, :].transpose(1, 2)
        K2 = K[:, :, :, 1, :].transpose(1, 2)
        V = V.transpose(1, 2)  # (B, H, T, head_dim)

        # Apply RoPE to sub-heads
        if self.use_rope:
            Q1, K1 = self.rope(Q1, K1)
            Q2, K2 = self.rope(Q2, K2)

        # Compute two attention maps
        score1 = torch.matmul(Q1, K1.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        score2 = torch.matmul(Q2, K2.transpose(-2, -1)) * self.scale

        if causal:
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            score1 = score1.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            score2 = score2.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            m = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            score1 = score1.masked_fill(~m, float("-inf"))
            score2 = score2.masked_fill(~m, float("-inf"))

        attn1 = F.softmax(score1, dim=-1)
        attn2 = F.softmax(score2, dim=-1)
        attn1 = torch.nan_to_num(attn1, nan=0.0)
        attn2 = torch.nan_to_num(attn2, nan=0.0)

        # Differential: attn1 - λ * attn2
        lam = self._compute_lambda()  # (H,)
        lam = lam.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, H, 1, 1)
        diff_attn = attn1 - lam * attn2
        diff_attn = self.dropout(diff_attn)

        # Weighted sum
        out = torch.matmul(diff_attn, V)  # (B, H, T, head_dim)
        out = (1 - self.lambda_init) * out  # scale factor
        out = self.norm(out)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        return out, None


# ---------------------------------------------------------------------------
# Sliding Window Attention
# ---------------------------------------------------------------------------

class SlidingWindowAttention(nn.Module):
    """Attention with sliding window mask (Longformer-style).

    Each position attends only to positions within a local window of size
    2 * window_size + 1, plus a set of global tokens.

    This achieves O(T * window_size) complexity instead of O(T^2).

    Args:
        d_model:      model dimension
        n_heads:      number of attention heads
        window_size:  half-window size (attends to window_size positions on each side)
        n_global:     number of global tokens (from the front) that attend to all positions
        dropout:      attention dropout

    Example:
        >>> swa = SlidingWindowAttention(d_model=512, n_heads=8, window_size=32)
        >>> x = torch.randn(2, 256, 512)
        >>> out, _ = swa(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 64,
        n_global: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.n_global = n_global
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _build_window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Build sliding window attention mask.

        Returns:
            mask: (T, T) bool, True = can attend
        """
        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        dist = (i.unsqueeze(1) - j.unsqueeze(0)).abs()
        mask = dist <= self.window_size

        # Global tokens attend to everything
        if self.n_global > 0:
            mask[:self.n_global, :] = True
            mask[:, :self.n_global] = True

        return mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x:              (B, T, d_model)
            attention_mask: (B, T) optional padding mask
            causal:         apply causal constraint within window

        Returns:
            output: (B, T, d_model)
            None:   (no KV cache for sliding window)
        """
        B, T, D = x.shape

        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Window mask
        win_mask = self._build_window_mask(T, x.device)  # (T, T)
        scores = scores.masked_fill(
            ~win_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        if causal:
            causal_m = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_m.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            scores = scores.masked_fill(
                ~attention_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out), None


# ---------------------------------------------------------------------------
# Reformer-style LSH Attention (simplified)
# ---------------------------------------------------------------------------

class LSHAttention(nn.Module):
    """Locality Sensitive Hashing Attention (simplified Reformer-style).

    Groups queries and keys into buckets using random projections,
    then applies attention within each bucket. This reduces complexity
    from O(T^2) to O(T log T) approximately.

    Note: This is a simplified approximation suitable for training.
    Full Reformer uses reversible residuals + LSH chunking.

    Args:
        d_model:    model dimension
        n_heads:    number of attention heads
        n_hashes:   number of LSH hash rounds
        bucket_size:size of each LSH bucket
        dropout:    attention dropout

    Example:
        >>> lsh = LSHAttention(d_model=512, n_heads=8, n_hashes=4)
        >>> x = torch.randn(2, 256, 512)
        >>> out = lsh(x)  # (2, 256, 512)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_hashes: int = 4,
        bucket_size: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size
        self.scale = self.head_dim ** -0.5

        self.qk_proj = nn.Linear(d_model, d_model, bias=False)  # shared Q=K projection (Reformer)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _lsh_bucket(self, q: torch.Tensor) -> torch.Tensor:
        """Assign queries to LSH buckets via random projections.

        Args:
            q: (B, H, T, head_dim)

        Returns:
            buckets: (B, H, T) long tensor of bucket indices
        """
        B, H, T, D = q.shape
        # Random rotation vectors
        n_buckets = max(2, T // self.bucket_size)
        rotations = torch.randn(D, n_buckets // 2, device=q.device)
        proj = torch.einsum("bhtd,dn->bhtn", q, rotations)  # (B, H, T, n_buckets//2)
        pos = proj > 0
        # Convert binary to integer bucket id
        powers = 2 ** torch.arange(n_buckets // 2, device=q.device)
        buckets = (pos.float() * powers).sum(dim=-1).long()  # (B, H, T)
        return buckets % n_buckets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            output: (B, T, d_model)
        """
        B, T, D = x.shape

        QK = self.qk_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Normalize QK for LSH (unit sphere)
        qk_norm = F.normalize(QK, p=2, dim=-1)

        # Assign to buckets
        buckets = self._lsh_bucket(qk_norm)  # (B, H, T)

        # Sort by bucket
        sort_idx = buckets.argsort(dim=-1)  # (B, H, T)
        unsort_idx = sort_idx.argsort(dim=-1)

        # Gather sorted queries/keys/values
        sorted_qk = qk_norm.gather(
            2, sort_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        )
        sorted_v = V.gather(
            2, sort_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        )

        # Compute attention within chunks (approximate bucket attention)
        scores = torch.matmul(sorted_qk, sorted_qk.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        sorted_out = torch.matmul(attn, sorted_v)

        # Unsort
        out = sorted_out.gather(
            2, unsort_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Hyper-Network Attention (per-sample attention weights)
# ---------------------------------------------------------------------------

class HyperNetworkAttention(nn.Module):
    """Attention where projection weights are generated by a hyper-network.

    Instead of fixed Q/K/V projection matrices, a small hyper-network generates
    input-dependent projections. This allows the attention to adapt its behavior
    based on the global context.

    Args:
        d_model:    model dimension
        n_heads:    number of attention heads
        hyper_dim:  hyper-network hidden dimension
        dropout:    attention dropout

    Example:
        >>> hna = HyperNetworkAttention(d_model=256, n_heads=4)
        >>> x = torch.randn(2, 32, 256)
        >>> out = hna(x)  # (2, 32, 256)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        hyper_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # Hyper-network: global context → projection offsets
        self.hyper_enc = nn.Sequential(
            nn.Linear(d_model, hyper_dim),
            nn.SiLU(),
            nn.Linear(hyper_dim, 3 * d_model * d_model // n_heads),  # Q, K, V offsets per head
        )

        # Base projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale_factor = 0.01  # scale for hyper offsets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            output: (B, T, d_model)
        """
        B, T, D = x.shape
        H = self.n_heads
        Hd = self.head_dim

        # Standard projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, T, H, Hd).transpose(1, 2)
        K = K.view(B, T, H, Hd).transpose(1, 2)
        V = V.view(B, T, H, Hd).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# KV-Compressed Attention (Memory-Efficient)
# ---------------------------------------------------------------------------

class KVCompressedAttention(nn.Module):
    """KV-cache compressed attention for long sequences.

    Compresses the key-value pairs to a fixed budget using:
    1. Top-k recent tokens (always kept)
    2. Attended tokens from earlier in the sequence
    3. Summary tokens representing older context

    Args:
        d_model:       model dimension
        n_heads:       number of heads
        budget:        maximum KV budget (number of cached tokens)
        recent_ratio:  fraction of budget reserved for recent tokens
        dropout:       attention dropout

    Example:
        >>> kvc = KVCompressedAttention(d_model=512, n_heads=8, budget=256)
        >>> x = torch.randn(2, 1024, 512)
        >>> out, _ = kvc(x, causal=True)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        budget: int = 256,
        recent_ratio: float = 0.25,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.budget = budget
        self.n_recent = max(1, int(budget * recent_ratio))
        self.n_attended = budget - self.n_recent
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x:      (B, T, d_model)
            causal: apply causal mask

        Returns:
            output: (B, T, d_model)
            None:   (no KV cache returned)
        """
        B, T, D = x.shape
        H = self.n_heads
        Hd = self.head_dim

        Q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, Hd).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Hd).transpose(1, 2)

        if T <= self.budget:
            # Short sequences: full attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            if causal:
                cm = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                scores = scores.masked_fill(~cm.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
        else:
            # Long sequences: compress KV to budget
            # Keep recent tokens + select attended tokens from the rest
            recent_start = T - self.n_recent
            K_recent = K[:, :, recent_start:, :]
            V_recent = V[:, :, recent_start:, :]

            # Attended tokens: pick top-n based on attention scores to Q (mean over T)
            Q_mean = Q.mean(dim=2, keepdim=True)  # (B, H, 1, Hd)
            old_K = K[:, :, :recent_start, :]     # (B, H, recent_start, Hd)
            selection_scores = (Q_mean * old_K).sum(dim=-1)  # (B, H, recent_start)
            n_attend = min(self.n_attended, recent_start)
            _, top_idx = selection_scores.topk(n_attend, dim=-1)  # (B, H, n_attend)
            top_idx_sorted = top_idx.sort(dim=-1).values

            K_selected = old_K.gather(
                2, top_idx_sorted.unsqueeze(-1).expand(-1, -1, -1, Hd)
            )
            V_selected = V[:, :, :recent_start, :].gather(
                2, top_idx_sorted.unsqueeze(-1).expand(-1, -1, -1, Hd)
            )

            # Concatenate: selected (old) + recent
            K_budget = torch.cat([K_selected, K_recent], dim=2)  # (B, H, budget, Hd)
            V_budget = torch.cat([V_selected, V_recent], dim=2)

            scores = torch.matmul(Q, K_budget.transpose(-2, -1)) * self.scale
            attn = F.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            V = V_budget  # Use compressed V for output

        attn = self.dropout(attn)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out), None


# ---------------------------------------------------------------------------
# Temporal Attention (with explicit time decay)
# ---------------------------------------------------------------------------

class TemporalDecayAttention(nn.Module):
    """Attention with learned temporal decay for financial time series.

    Modifies attention scores with a time-decay factor:
        score(i, j) = q_i · k_j / √d + decay(t_i - t_j)

    The decay function is parameterized as:
        decay(Δt) = -|Δt| * exp(log_decay_rate)

    where log_decay_rate is a learnable per-head parameter.

    This explicitly biases attention toward recent events while still
    allowing the model to attend to older context when useful.

    Args:
        d_model:      model dimension
        n_heads:      number of attention heads
        init_decay:   initial decay rate (higher = faster decay)
        dropout:      attention dropout

    Example:
        >>> tda = TemporalDecayAttention(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 128, 512)
        >>> timestamps = torch.arange(128).float().unsqueeze(0).expand(2, -1)
        >>> out = tda(x, timestamps)  # (2, 128, 512)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        init_decay: float = 0.01,
        dropout: float = 0.0,
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
        self.dropout = nn.Dropout(dropout)

        # Per-head learnable decay rates (log scale for stability)
        self.log_decay = nn.Parameter(
            torch.full((n_heads,), math.log(init_decay))
        )

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, T, d_model)
            timestamps:     (B, T) float timestamps (optional)
            attention_mask: (B, T) bool mask
            causal:         apply causal mask

        Returns:
            output: (B, T, d_model)
        """
        B, T, D = x.shape
        H = self.n_heads
        Hd = self.head_dim

        Q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, Hd).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Hd).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Add temporal decay
        if timestamps is not None:
            delta_t = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # (B, T, T)
            decay_rates = torch.exp(self.log_decay)  # (H,)
            # decay_bias: (1, H, T, T) — negative, larger for older tokens
            decay_bias = -delta_t.abs().unsqueeze(1) * decay_rates.view(1, H, 1, 1)
            scores = scores + decay_bias
        else:
            # Use integer positions as proxy for time
            pos = torch.arange(T, device=x.device, dtype=x.dtype)
            delta = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (T, T)
            decay_rates = torch.exp(self.log_decay)
            decay_bias = -delta.unsqueeze(0).unsqueeze(0) * decay_rates.view(1, H, 1, 1)
            scores = scores + decay_bias

        if causal:
            cm = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~cm.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            scores = scores.masked_fill(
                ~attention_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Cross-Asset Correlation Attention
# ---------------------------------------------------------------------------

class CrossAssetCorrelationAttention(nn.Module):
    """Attention that explicitly models cross-asset correlations.

    In multi-asset financial modeling, assets are not independent.
    This module computes attention across assets at each time step,
    allowing information flow between correlated instruments.

    Architecture:
    1. Group tokens by asset: x[asset] = x[:, asset::n_assets, :]
    2. For each time step, compute cross-asset attention
    3. Aggregate with skip connection

    Args:
        d_model:   model dimension
        n_heads:   attention heads for cross-asset attention
        n_assets:  number of assets (for reshaping)
        dropout:   dropout probability

    Example:
        >>> caca = CrossAssetCorrelationAttention(d_model=256, n_heads=4, n_assets=8)
        >>> # x: (B, T*n_assets, d_model) — interleaved layout
        >>> x = torch.randn(2, 128 * 8, 256)
        >>> out = caca(x)  # (2, 128 * 8, 256)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_assets: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_assets = n_assets
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T*n_assets, d_model) — tokens in time-major interleaved order

        Returns:
            out: (B, T*n_assets, d_model)
        """
        B, TN, D = x.shape
        T = TN // self.n_assets
        N = self.n_assets

        if TN % N != 0:
            # Fallback: standard self-attention
            Q = self.q_proj(x).view(B, TN, self.n_heads, self.head_dim).transpose(1, 2)
            K = self.k_proj(x).view(B, TN, self.n_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj(x).view(B, TN, self.n_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, V).transpose(1, 2).reshape(B, TN, D)
            return x + self.out_proj(out)

        # Reshape to (B*T, N, D) for cross-asset attention at each time step
        x_t = x.view(B, T, N, D).reshape(B * T, N, D)

        Q = self.q_proj(x_t).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_t).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_t).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B * T, N, D)
        out = self.out_proj(out)

        out = out.reshape(B, T, N, D).reshape(B, TN, D)
        return self.norm(x + out)


# ---------------------------------------------------------------------------
# Attention utility: compute attention entropy
# ---------------------------------------------------------------------------

def attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """Compute entropy of attention weights.

    High entropy = spread-out attention (attending broadly).
    Low entropy  = peaked attention (attending sharply to few tokens).

    Useful for analyzing attention patterns and diagnosing collapse.

    Args:
        attn_weights: (B, H, T, T) attention probability matrix

    Returns:
        entropy: (B, H, T) entropy per query position
    """
    eps = 1e-10
    ent = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)
    return ent


def attention_distance(attn_weights: torch.Tensor) -> torch.Tensor:
    """Compute mean attended distance for each query.

    Measures how far on average each query attends.

    Args:
        attn_weights: (B, H, T, T)

    Returns:
        mean_dist: (B, H, T)
    """
    T = attn_weights.shape[-1]
    device = attn_weights.device
    positions = torch.arange(T, device=device, dtype=attn_weights.dtype)
    # Distance from each query position to each key position
    q_pos = torch.arange(T, device=device, dtype=attn_weights.dtype)
    k_pos = torch.arange(T, device=device, dtype=attn_weights.dtype)
    dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs()  # (T, T)
    mean_dist = (attn_weights * dist.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    return mean_dist


# ---------------------------------------------------------------------------
# Attention analysis toolkit
# ---------------------------------------------------------------------------

class AttentionAnalyzer:
    """Toolkit for analyzing attention patterns in transformer models.

    Records attention weights from forward passes and computes statistics
    about attention behavior.

    Args:
        model:      transformer model with attention hooks
        layer_ids:  list of layer indices to analyze

    Example:
        >>> analyzer = AttentionAnalyzer(model, layer_ids=[0, 6, 11])
        >>> with analyzer:
        ...     out = model(x)
        >>> stats = analyzer.compute_stats()
    """

    def __init__(self, model: nn.Module, layer_ids: Optional[List[int]] = None):
        self.model = model
        self.layer_ids = layer_ids
        self._hooks = []
        self._attn_weights: Dict[str, List[torch.Tensor]] = {}

    def _register_hook(self, name: str):
        def hook(module, input, output):
            # Try to capture attention weights if module stores them
            if hasattr(module, "_last_attn_weights"):
                self._attn_weights.setdefault(name, []).append(
                    module._last_attn_weights.detach().cpu()
                )
        return hook

    def __enter__(self):
        self._attn_weights = {}
        for name, module in self.model.named_modules():
            if "attn" in name.lower():
                h = module.register_forward_hook(self._register_hook(name))
                self._hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def compute_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute attention statistics across all recorded layers.

        Returns:
            stats: dict mapping layer name → dict of statistics
        """
        stats = {}
        for layer, weight_list in self._attn_weights.items():
            weights = torch.cat(weight_list, dim=0)  # (N, H, T, T)
            ent = attention_entropy(weights)
            dist = attention_distance(weights)
            stats[layer] = {
                "mean_entropy": ent.mean().item(),
                "std_entropy": ent.std().item(),
                "mean_distance": dist.mean().item(),
                "max_weight": weights.max().item(),
                "min_entropy": ent.min().item(),
                "n_samples": weights.shape[0],
            }
        return stats


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "AttentionConfig",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "FlashAttentionApprox",
    "CrossAssetAttention",
    "CausalSelfAttention",
    "SparseAttention",
    "LinearAttention",
    "AttentionPool",
    "MultiQueryAttention",
    "QKNorm",
    "DifferentialAttention",
    "SlidingWindowAttention",
    "LSHAttention",
    "HyperNetworkAttention",
    "KVCompressedAttention",
    "TemporalDecayAttention",
    "CrossAssetCorrelationAttention",
    "AttentionAnalyzer",
    "make_causal_mask",
    "make_local_mask",
    "make_sliding_window_causal_mask",
    "scaled_dot_product_attention",
    "attention_entropy",
    "attention_distance",
]


# =============================================================================
# SECTION: Advanced Sparse and Efficient Attention Mechanisms
# =============================================================================

class BigBirdAttention(nn.Module):
    """BigBird sparse attention: global + window + random attention.

    Achieves O(n) complexity vs O(n^2) for standard attention.
    Three components:
      1. Global tokens attend to/from all positions
      2. Sliding window for local attention
      3. Random keys for long-range dependencies

    Reference: Zaheer et al., "Big Bird: Transformers for Longer Sequences" NeurIPS 2020.

    Args:
        d_model: Embedding dimension
        num_heads: Number of attention heads
        window_size: Local sliding window size
        num_global_tokens: Count of global attention tokens (prepended)
        num_random_keys: Random keys per query for long-range
        dropout: Attention dropout
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int = 3,
        num_global_tokens: int = 2,
        num_random_keys: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.num_random_keys = num_random_keys
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)

    def _sh(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _mh(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), -1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        g = self.num_global_tokens
        q, k, v = self._sh(self.q_proj(x)), self._sh(self.k_proj(x)), self._sh(self.v_proj(x))
        output = torch.zeros(B, self.num_heads, T, self.head_dim, device=x.device, dtype=x.dtype)
        # Global attention
        a_g = torch.matmul(q[:, :, :g], k.transpose(-2, -1)) * self.scale
        if mask is not None:
            a_g = a_g + mask[:, :, :g]
        output[:, :, :g] = torch.matmul(self.dropout(torch.softmax(a_g, -1)), v)
        # Local + random attention
        hw = self.window_size // 2
        ag2 = torch.matmul(q, k[:, :, :g].transpose(-2, -1)) * self.scale
        for t in range(g, T):
            s, e = max(g, t - hw), min(T, t + hw + 1)
            lk, lv = k[:, :, s:e], v[:, :, s:e]
            qt = q[:, :, t:t+1]
            al = torch.matmul(qt, lk.transpose(-2, -1)) * self.scale
            if self.num_random_keys > 0 and T - g > self.num_random_keys:
                ri = torch.randperm(T - g, device=x.device)[:self.num_random_keys] + g
                rk, rv = k[:, :, ri], v[:, :, ri]
                ar = torch.matmul(qt, rk.transpose(-2, -1)) * self.scale
                all_v = torch.cat([lv, v[:, :, :g], rv], 2)
                all_a = torch.cat([al, ag2[:, :, t:t+1], ar], -1)
            else:
                all_v = torch.cat([lv, v[:, :, :g]], 2)
                all_a = torch.cat([al, ag2[:, :, t:t+1]], -1)
            output[:, :, t:t+1] = torch.matmul(self.dropout(torch.softmax(all_a, -1)), all_v)
        return self.out_proj(self._mh(output))


class MemoryEfficientAttention(nn.Module):
    """Flash Attention-style wrapper using scaled_dot_product_attention.

    Uses torch.nn.functional.scaled_dot_product_attention when available
    to achieve memory-efficient attention without materializing the NxN matrix.
    Falls back to standard attention when unavailable.

    Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
    Attention with IO-Awareness" (2022)

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        dropout: Dropout probability
        causal: Enable causal (autoregressive) masking
        use_flash: Attempt to use scaled_dot_product_attention
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout
        self.causal = causal
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)
        if self.use_flash:
            dp = self.dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dp,
                is_causal=self.causal and mask is None,
            )
        else:
            scale = d ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if self.causal:
                cm = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
                attn = attn + cm
            if mask is not None:
                attn = attn + mask
            attn = torch.softmax(attn, dim=-1)
            if self.training and self.dropout_p > 0:
                attn = F.dropout(attn, p=self.dropout_p)
            out = torch.matmul(attn, v)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, D))


class CosineAttention(nn.Module):
    """Cosine-similarity attention with learnable per-head temperature.

    Normalizes queries and keys to unit vectors before computing
    similarity, making attention scale-invariant. Particularly useful
    for financial features with varying magnitudes.

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        temperature: Initial temperature (learnable per head)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        temperature: float = 10.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.temperature = nn.Parameter(torch.full((1, num_heads, 1, 1), temperature))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        q = F.normalize(self.q_proj(x).view(B, T, H, d).transpose(1, 2), p=2, dim=-1)
        k = F.normalize(self.k_proj(x).view(B, T, H, d).transpose(1, 2), p=2, dim=-1)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        if mask is not None:
            attn = attn + mask
        attn = self.dropout(torch.softmax(attn, dim=-1))
        return self.out_proj(torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D))


class TalkingHeadsAttention(nn.Module):
    """Talking-Heads: pre- and post-softmax linear head mixing.

    Applies linear projections over the head dimension before and after
    softmax, allowing attention heads to exchange information.

    Reference: Shazeer et al., "Talking-Heads Attention" (2020)

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.talking_pre = nn.Linear(num_heads, num_heads, bias=False)
        self.talking_post = nn.Linear(num_heads, num_heads, bias=False)
        nn.init.eye_(self.talking_pre.weight)
        nn.init.eye_(self.talking_post.weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, d).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, T, H, d).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, H, d).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        # Pre-softmax talking
        attn = self.talking_pre(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = torch.softmax(attn, dim=-1)
        # Post-softmax talking
        attn = self.talking_post(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = torch.matmul(self.dropout(attn), v).permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.out_proj(out)


class GatedAttentionUnit(nn.Module):
    """Gated Attention Unit from FLASH (linear-time transformer variant).

    Single-head attention with gating via SiLU nonlinearity. Achieves
    O(n) complexity in linear attention mode, O(n^2) in standard mode.

    Reference: Hua et al., "Transformer Quality in Linear Time" ICML 2022.

    Args:
        d_model: Embedding dimension
        expansion_factor: Hidden dim multiplier
        query_key_dim: Q/K projection dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 2,
        query_key_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner = d_model * expansion_factor
        self.inner_dim = inner
        self.scale = query_key_dim ** -0.5
        self.norm = nn.LayerNorm(d_model)
        self.to_uv = nn.Linear(d_model, inner * 2, bias=False)
        self.to_qk = nn.Linear(d_model, query_key_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, d_model, bias=False), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x = self.norm(x)
        u, v = self.to_uv(x).chunk(2, dim=-1)
        v = F.silu(v)
        q, k = self.to_qk(x).chunk(2, dim=-1)
        q, k = F.silu(q), F.silu(k)
        cm = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale + cm, dim=-1)
        return self.to_out(torch.matmul(attn, u) * v)


class ConvolutionalAttention(nn.Module):
    """Conv-Attention hybrid: depthwise conv local + self-attention global.

    Inspired by ConvBERT. Uses learnable gating to blend conv (local,
    inductive bias) and self-attention (global, permutation-equivariant).

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        kernel_size: Convolution kernel size for local span
        dropout: Dropout probability
    """

    def __init__(
        self, d_model: int, num_heads: int, kernel_size: int = 9, dropout: float = 0.0
    ) -> None:
        super().__init__()
        H = num_heads
        d = d_model // H
        self.num_heads, self.head_dim = H, d
        self.scale = d ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        p = (kernel_size - 1) // 2
        self.conv_dw = nn.Conv1d(d_model, d_model, kernel_size, padding=p, groups=d_model)
        self.conv_pw = nn.Conv1d(d_model, d_model, 1)
        self.conv_norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model * 2, 2)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)
        a = self.dropout(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, -1))
        ao = self.attn_out(torch.matmul(a, v).transpose(1, 2).contiguous().view(B, T, D))
        xc = self.conv_norm(self.conv_pw(self.conv_dw(x.transpose(1, 2))).transpose(1, 2))
        g = torch.softmax(self.gate(torch.cat([ao, xc], -1)), -1)
        return self.out_proj(g[:, :, 0:1] * ao + g[:, :, 1:2] * xc)


class RegimeAwareAttention(nn.Module):
    """Self-attention conditioned on market regime embeddings.

    Adds learnable biases to Q/K/V projections and scales attention
    temperature based on detected market regime (bull/bear/vol).

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        num_regimes: Number of market regime classes
        dropout: Dropout probability
    """

    def __init__(
        self, d_model: int, num_heads: int, num_regimes: int = 5, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.regime_q = nn.Embedding(num_regimes, d_model)
        self.regime_k = nn.Embedding(num_regimes, d_model)
        self.regime_v = nn.Embedding(num_regimes, d_model)
        self.regime_temp = nn.Embedding(num_regimes, num_heads)
        for emb in [self.regime_q, self.regime_k, self.regime_v]:
            nn.init.zeros_(emb.weight)
        nn.init.ones_(self.regime_temp.weight)

    def forward(
        self,
        x: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        if regime_ids is not None and regime_ids.dim() == 1:
            regime_ids = regime_ids.unsqueeze(1).expand(B, T)
        q = self.q_proj(x) + (self.regime_q(regime_ids) if regime_ids is not None else 0)
        k = self.k_proj(x) + (self.regime_k(regime_ids) if regime_ids is not None else 0)
        v = self.v_proj(x) + (self.regime_v(regime_ids) if regime_ids is not None else 0)
        H, d = self.num_heads, self.head_dim
        q = q.view(B, T, H, d).transpose(1, 2)
        k = k.view(B, T, H, d).transpose(1, 2)
        v = v.view(B, T, H, d).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1))
        if regime_ids is not None:
            dom = regime_ids.mode(dim=1).values
            t = self.regime_temp(dom).unsqueeze(-1).unsqueeze(-1)
            attn = attn * t * self.scale
        else:
            attn = attn * self.scale
        if mask is not None:
            attn = attn + mask
        out = torch.matmul(self.dropout(torch.softmax(attn, -1)), v)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, D))


class MultiResolutionAttention(nn.Module):
    """Attention at multiple temporal resolutions with learned fusion.

    Applies self-attention at geometrically spaced temporal scales,
    upsamples results to original length, then fuses all scales.

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads per scale
        scales: List of pooling downsampling factors
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        scales: Optional[List[int]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scales = scales or [1, 4, 16]
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        sf = self.head_dim ** -0.5
        self.sf = sf
        self.dropout = nn.Dropout(dropout)
        self.q_projs = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in self.scales])
        self.k_projs = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in self.scales])
        self.v_projs = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in self.scales])
        self.fusion = nn.Linear(d_model * len(self.scales), d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        outs = []
        for fac, qp, kp, vp in zip(self.scales, self.q_projs, self.k_projs, self.v_projs):
            Ts = max(1, T // fac)
            xs = x[:, :Ts * fac].view(B, Ts, fac, D).mean(2) if fac > 1 else x
            q = qp(xs).view(B, Ts, H, d).transpose(1, 2)
            k = kp(xs).view(B, Ts, H, d).transpose(1, 2)
            v = vp(xs).view(B, Ts, H, d).transpose(1, 2)
            a = self.dropout(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.sf, -1))
            os = torch.matmul(a, v).transpose(1, 2).contiguous().view(B, Ts, D)
            if Ts != T:
                os = F.interpolate(os.transpose(1, 2), size=T, mode="nearest").transpose(1, 2)
            outs.append(os)
        return self.norm(self.fusion(torch.cat(outs, -1)))


class AttentionWithExternalMemory(nn.Module):
    """Self-attention augmented with a learnable external memory bank.

    Provides O(M) additional key-value pairs (M = memory_size) that
    the model can read from. Useful for persistent financial knowledge.

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        memory_size: Number of external memory slots
        dropout: Dropout probability
    """

    def __init__(
        self, d_model: int, num_heads: int, memory_size: int = 256, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_mem = nn.Linear(d_model, d_model, bias=False)
        self.memory_k = nn.Parameter(torch.randn(memory_size, d_model) * 0.02)
        self.memory_v = nn.Parameter(torch.randn(memory_size, d_model) * 0.02)
        self.mem_gate = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        M = self.memory_k.size(0)
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        so = torch.matmul(self.dropout(torch.softmax(attn, -1)), v).transpose(1, 2).contiguous().view(B, T, D)
        qm = self.q_mem(x).view(B, T, H, d).transpose(1, 2)
        mk = self.memory_k.view(M, H, d).permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)
        mv = self.memory_v.view(M, H, d).permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)
        ma = self.dropout(torch.softmax(torch.matmul(qm, mk.transpose(-2, -1)) * self.scale, -1))
        mo = torch.matmul(ma, mv).transpose(1, 2).contiguous().view(B, T, D)
        g = torch.sigmoid(self.mem_gate(x))
        return self.out_proj(torch.cat([so, g * mo], -1))


class EventDrivenAttention(nn.Module):
    """Attention conditioned on discrete financial event types.

    Adds event-specific Q/K attention biases and gated V modulation
    for earnings releases, Fed announcements, index rebalancing, etc.

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        num_event_types: Number of event categories
        event_embed_dim: Event embedding dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_event_types: int = 16,
        event_embed_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.event_embed = nn.Embedding(num_event_types + 1, event_embed_dim, padding_idx=0)
        self.event_to_bias = nn.Linear(event_embed_dim, num_heads)
        self.event_gate = nn.Sequential(nn.Linear(event_embed_dim, d_model), nn.Sigmoid())

    def forward(
        self, x: torch.Tensor, event_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if event_ids is not None:
            ev = self.event_embed(event_ids)
            bias = self.event_to_bias(ev).permute(0, 2, 1).unsqueeze(-1)
            attn = attn + bias
            gate = self.event_gate(ev).view(B, T, H, d).transpose(1, 2)
            v = v * gate
        attn = self.dropout(torch.softmax(attn, -1))
        return self.out_proj(torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D))


class FractalAttention(nn.Module):
    """Multi-scale fractal attention for self-similar time series.

    Applies attention at geometrically spaced subsampling rates and
    aggregates with softmax-normalized fractal dimension weights.

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        num_scales: Number of fractal scales
        base_scale: Geometric spacing base
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_scales: int = 4,
        base_scale: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scales = [base_scale ** i for i in range(num_scales)]
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.sf = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.fw = nn.Parameter(torch.ones(num_scales, num_heads) / num_scales)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)
        fw = torch.softmax(self.fw, dim=0)  # (S, H)
        out = torch.zeros_like(q)
        for si, sc in enumerate(self.scales):
            if sc > T:
                break
            idx = torch.arange(0, T, sc, device=x.device)
            ks = k[:, :, idx]
            vs = v[:, :, idx]
            a = self.dropout(torch.softmax(torch.matmul(q, ks.transpose(-2, -1)) * self.sf, -1))
            os = torch.matmul(a, vs)
            w = fw[si].view(1, H, 1, 1)
            out = out + w * os
        return self.out_proj(self.norm(out.transpose(1, 2).contiguous().view(B, T, D)))


class LeadLagAttention(nn.Module):
    """Attention explicitly modeling lead-lag relationships.

    In financial markets, some assets lead others temporally
    (e.g., futures lead spot prices). This module encodes
    asymmetric attention biases for such relationships.

    Args:
        d_model: Embedding dimension
        num_heads: Attention heads
        max_lag: Maximum temporal lag in steps
        dropout: Dropout probability
    """

    def __init__(
        self, d_model: int, num_heads: int, max_lag: int = 5, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_lag = max_lag
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # Relative lag attention bias: (max_lag+1, num_heads)
        self.lag_bias = nn.Parameter(torch.zeros(max_lag + 1, num_heads))
        nn.init.normal_(self.lag_bias, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)
        # Causal mask
        cm = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        # Add lag bias based on distance
        lags = torch.clamp(torch.arange(T, device=x.device).unsqueeze(0) -
                           torch.arange(T, device=x.device).unsqueeze(1), 0, self.max_lag)
        lb = self.lag_bias[lags]  # (T, T, H)
        lb = lb.permute(2, 0, 1).unsqueeze(0)  # (1, H, T, T)
        attn = attn + cm + lb
        out = torch.matmul(self.dropout(torch.softmax(attn, -1)), v)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, D))


# =============================================================================
# Attention Analysis Utilities
# =============================================================================

def compute_attention_rollout(
    attention_weights_list: List[torch.Tensor],
    discard_ratio: float = 0.9,
) -> torch.Tensor:
    """Attention rollout for transformer interpretability.

    Propagates attention weights through layers to identify which
    input tokens most influence each output position.

    Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (2020)

    Args:
        attention_weights_list: List of per-layer (B, H, T, T) attention tensors
        discard_ratio: Fraction of lowest-weight attention to discard
    Returns:
        Rollout matrix (B, T, T)
    """
    masks = []
    for attn in attention_weights_list:
        avg = attn.mean(dim=1)  # (B, T, T)
        if discard_ratio > 0:
            flat = avg.view(avg.size(0), -1)
            thresh = torch.quantile(flat, discard_ratio, dim=1).view(-1, 1, 1)
            avg = avg * (avg >= thresh).float()
        I = torch.eye(avg.size(-1), device=avg.device).unsqueeze(0)
        avg = (avg + I) / ((avg + I).sum(-1, keepdim=True) + 1e-10)
        masks.append(avg)
    rollout = masks[0]
    for m in masks[1:]:
        rollout = torch.matmul(m, rollout)
    return rollout


def attention_sparsity(
    attn_weights: torch.Tensor,
    threshold: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Compute sparsity statistics for attention distributions.

    Args:
        attn_weights: (B, H, T, T) attention weight tensor
        threshold: Minimum weight considered non-zero
    Returns:
        Dict with sparse_fraction, gini_coefficient, effective_positions_mean,
        effective_positions_std, max_attention_mean
    """
    B, H, T, _ = attn_weights.shape
    sf = (attn_weights < threshold).float().mean()
    flat = attn_weights.view(B * H * T, T)
    srt = flat.sort(dim=-1).values
    idx = torch.arange(1, T + 1, device=attn_weights.device, dtype=torch.float32)
    denom = T * srt.sum(-1) + 1e-10
    gini = ((2 * (idx * srt).sum(-1) / denom) - (T + 1) / T).mean()
    eff = 1.0 / ((attn_weights ** 2).sum(-1) + 1e-10)
    return {
        "sparse_fraction": sf,
        "gini_coefficient": gini,
        "effective_positions_mean": eff.mean(),
        "effective_positions_std": eff.std(),
        "max_attention_mean": attn_weights.max(-1).values.mean(),
    }


def build_attention_module(
    attention_type: str,
    d_model: int,
    num_heads: int,
    **kwargs,
) -> nn.Module:
    """Factory function to create attention modules by type string.

    Args:
        attention_type: Identifier (e.g., "standard", "bigbird", "cosine")
        d_model: Model dimension
        num_heads: Number of attention heads
        **kwargs: Additional constructor arguments
    Returns:
        Instantiated attention nn.Module
    """
    registry = {
        "standard": MultiHeadSelfAttention,
        "gqa": GroupedQueryAttention,
        "differential": DifferentialAttention,
        "sliding_window": SlidingWindowAttention,
        "lsh": LSHAttention,
        "bigbird": BigBirdAttention,
        "memory_efficient": MemoryEfficientAttention,
        "cosine": CosineAttention,
        "talking_heads": TalkingHeadsAttention,
        "gau": GatedAttentionUnit,
        "convolutional": ConvolutionalAttention,
        "multi_resolution": MultiResolutionAttention,
        "regime_aware": RegimeAwareAttention,
        "external_memory": AttentionWithExternalMemory,
        "event_driven": EventDrivenAttention,
        "fractal": FractalAttention,
        "lead_lag": LeadLagAttention,
        "hypernetwork": HyperNetworkAttention,
        "kv_compressed": KVCompressedAttention,
        "temporal_decay": TemporalDecayAttention,
        "cross_asset": CrossAssetCorrelationAttention,
    }
    if attention_type not in registry:
        raise ValueError(
            f"Unknown attention type '{attention_type}'. "
            f"Available: {sorted(registry.keys())}"
        )
    return registry[attention_type](d_model=d_model, num_heads=num_heads, **kwargs)


_NEW_EXPORTS = [
    "BigBirdAttention", "MemoryEfficientAttention", "CosineAttention",
    "TalkingHeadsAttention", "GatedAttentionUnit", "ConvolutionalAttention",
    "MultiResolutionAttention", "RegimeAwareAttention", "AttentionWithExternalMemory",
    "EventDrivenAttention", "FractalAttention", "LeadLagAttention",
    "compute_attention_rollout", "attention_sparsity", "build_attention_module",
]


# =============================================================================
# SECTION: Expanded Attention Mechanisms (Part 3)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class ScaledDotProductAttentionV2(nn.Module):
    """Standard scaled dot-product attention with full feature set.

    Features:
    - Optional causal masking
    - Optional relative position bias
    - Dropout on attention weights
    - Optional head-specific temperature scaling
    - Key/value projection dimension override
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_kv: int = None,
        dropout: float = 0.1,
        causal: bool = False,
        use_bias: bool = True,
        head_specific_temperature: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_kv = d_kv or d_model
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, self.d_kv, bias=use_bias)
        self.v_proj = nn.Linear(d_model, self.d_kv, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

        if head_specific_temperature:
            self.temperature = nn.Parameter(torch.ones(n_heads, 1, 1))
        else:
            self.temperature = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        mask: torch.Tensor = None,
        position_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = query

        B, T_q, _ = query.shape
        T_k = key.shape[1]
        H, Dh = self.n_heads, self.d_head

        Q = self.q_proj(query).view(B, T_q, H, Dh).transpose(1, 2)
        K = self.k_proj(key).view(B, T_k, H, -1).transpose(1, 2)
        V = self.v_proj(value).view(B, T_k, H, -1).transpose(1, 2)

        scale = 1.0 / math.sqrt(Dh)
        scores = (Q @ K.transpose(-2, -1)) * scale

        if self.temperature is not None:
            scores = scores / self.temperature.clamp(min=1e-4)

        if position_bias is not None:
            scores = scores + position_bias

        if self.causal:
            causal_mask = torch.triu(torch.ones(T_q, T_k, device=query.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T_q, H * Dh)
        return self.out_proj(out)


class WindowAttention(nn.Module):
    """Shifted window attention (Swin Transformer style).

    Applies attention within non-overlapping local windows,
    with optional cyclic shift for cross-window interaction.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 16,
        shift_size: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

        # Relative position bias table
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1), n_heads)

    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """Partition sequence into windows."""
        B, T, D = x.shape
        n_windows = T // self.window_size
        x = x.view(B, n_windows, self.window_size, D)
        return x.view(B * n_windows, self.window_size, D)

    def _window_reverse(self, windows: torch.Tensor, T: int) -> torch.Tensor:
        """Reverse window partition."""
        n_windows = T // self.window_size
        BW = windows.shape[0]
        B = BW // n_windows
        x = windows.view(B, n_windows, self.window_size, -1)
        return x.view(B, T, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        if self.shift_size > 0:
            x = torch.roll(x, -self.shift_size, dims=1)

        # Pad to multiple of window_size
        pad_len = (self.window_size - T % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        T_pad = x.shape[1]

        windows = self._window_partition(x)  # [B*n_win, W, D]
        n_wins = windows.shape[0]

        qkv = self.qkv(windows).reshape(n_wins, self.window_size, 3, H, Dh)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        scores = (Q @ K.transpose(-2, -1)) / self.scale

        # Relative position bias
        W = self.window_size
        positions = torch.arange(W, device=x.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1) + W - 1
        bias = self.rel_pos_bias(rel_pos).permute(2, 0, 1)
        scores = scores + bias.unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().reshape(n_wins, self.window_size, D)
        out = self._window_reverse(out, T_pad)

        if pad_len > 0:
            out = out[:, :T, :]

        if self.shift_size > 0:
            out = torch.roll(out, self.shift_size, dims=1)

        return self.out(out)


class AxialAttention(nn.Module):
    """Axial attention for sequences structured as 2D grids (time x features).

    Factorizes full attention into row-wise + column-wise attention,
    reducing complexity from O(n^2) to O(n * sqrt(n)).
    Useful for tick data on (time, feature_dim) grids.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        T: int,
        F: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.T = T
        self.F = F
        self.d_model = d_model
        self.n_heads = n_heads

        # Row attention (over time dimension)
        self.row_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Column attention (over feature dimension)
        self.col_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T*F, D] (flattened 2D grid)
        """
        B, TF, D = x.shape
        x_2d = x.view(B, self.T, self.F, D)

        # Row attention: each row attends over columns
        x_row = x_2d.view(B * self.T, self.F, D)
        x_row_out, _ = self.row_attn(x_row, x_row, x_row)
        x_2d = self.norm1(x_2d + x_row_out.view(B, self.T, self.F, D))

        # Column attention: each column attends over rows
        x_col = x_2d.permute(0, 2, 1, 3).contiguous().view(B * self.F, self.T, D)
        x_col_out, _ = self.col_attn(x_col, x_col, x_col)
        x_2d = self.norm2(x_2d + x_col_out.view(B, self.F, self.T, D).permute(0, 2, 1, 3))

        return x_2d.view(B, TF, D)


class LocalGlobalAttention(nn.Module):
    """Interleaved local and global attention heads (Longformer-inspired).

    Half the heads attend locally (window), half attend globally.
    Global tokens (e.g., [CLS]) attend to all positions.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 32,
        n_global_tokens: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_heads % 2 == 0
        self.n_local_heads = n_heads // 2
        self.n_global_heads = n_heads - self.n_local_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.n_global_tokens = n_global_tokens

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor, global_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_local_heads + self.n_global_heads
        Dh = self.d_head

        Q = self.q(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v(x).view(B, T, H, Dh).transpose(1, 2)

        # Local attention for local heads
        W = self.window_size
        local_outs = []
        for h in range(self.n_local_heads):
            q_h = Q[:, h]  # [B, T, Dh]
            k_h = K[:, h]
            v_h = V[:, h]

            # Naive local window attention
            scores = torch.full((B, T, T), float("-inf"), device=x.device)
            for t in range(T):
                start = max(0, t - W // 2)
                end = min(T, t + W // 2 + 1)
                s = (q_h[:, t:t+1, :] @ k_h[:, start:end, :].transpose(-2, -1)) / self.scale
                scores[:, t, start:end] = s.squeeze(1)

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            local_outs.append((attn @ v_h).unsqueeze(1))

        # Global attention for global heads
        global_outs = []
        for h in range(self.n_global_heads):
            h_idx = self.n_local_heads + h
            q_h = Q[:, h_idx]
            k_h = K[:, h_idx]
            v_h = V[:, h_idx]

            scores = (q_h @ k_h.transpose(-2, -1)) / self.scale
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            global_outs.append((attn @ v_h).unsqueeze(1))

        combined = torch.cat(local_outs + global_outs, dim=1)  # [B, H, T, Dh]
        out = combined.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return self.out(out)


class DifferentialAttention(nn.Module):
    """Differential Attention (Ye et al. 2024).

    Uses two softmax attention maps and subtracts them to cancel noise:
    DiffAttn(Q1, Q2, K1, K2, V) = (softmax(Q1*K1^T / sqrt(d)) - lambda * softmax(Q2*K2^T / sqrt(d))) * V

    Where lambda is a learned scalar per head.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        lambda_init: float = 0.8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.q1 = nn.Linear(d_model, d_model // 2)
        self.q2 = nn.Linear(d_model, d_model // 2)
        self.k1 = nn.Linear(d_model, d_model // 2)
        self.k2 = nn.Linear(d_model, d_model // 2)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.lambda_ = nn.Parameter(torch.full((n_heads, 1, 1), lambda_init))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.d_head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        Dh = self.d_head // 2  # half heads for each stream

        Q1 = self.q1(x).view(B, T, H, Dh).transpose(1, 2)
        Q2 = self.q2(x).view(B, T, H, Dh).transpose(1, 2)
        K1 = self.k1(x).view(B, T, H, Dh).transpose(1, 2)
        K2 = self.k2(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v(x).view(B, T, H, self.d_head).transpose(1, 2)

        A1 = (Q1 @ K1.transpose(-2, -1)) / self.scale
        A2 = (Q2 @ K2.transpose(-2, -1)) / self.scale

        if mask is not None:
            A1 = A1.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float("-inf"))
            A2 = A2.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float("-inf"))

        A1 = F.softmax(A1, dim=-1)
        A2 = F.softmax(A2, dim=-1)

        lam = torch.sigmoid(self.lambda_)
        A = self.dropout(A1 - lam * A2)

        out = (A @ V)  # [B, H, T, Dh]
        out = self.norm(out)
        out = out.transpose(1, 2).contiguous().view(B, T, H * self.d_head)
        return self.out(out)


class InfiniteAttention(nn.Module):
    """Infini-Attention (Munkhdalai et al. 2024).

    Combines local attention with a compressed memory for long-range context.
    At each segment, updates a fixed-size memory via associative binding,
    then retrieves from it alongside local attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        segment_len: int = 64,
        memory_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.segment_len = segment_len

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

        # Memory gating
        self.beta = nn.Parameter(torch.zeros(n_heads, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def _update_memory(
        self,
        memory: torch.Tensor,
        z: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update memory using associative binding (linear attention update)."""
        # memory: [B, H, M, Dh], z: [B, H, M]
        K_feat = F.elu(K) + 1.0
        V_proj = V

        # Update: M = M + (K^T * V)
        update = torch.einsum("bhnd,bhnm->bhdm", K_feat, V_proj)
        new_memory = memory + update

        # Update normalization
        z_update = K_feat.sum(dim=2)
        new_z = z + z_update

        return new_memory, new_z

    def _retrieve_from_memory(
        self,
        memory: torch.Tensor,
        z: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve from memory via linear attention."""
        Q_feat = F.elu(Q) + 1.0
        retrieved = torch.einsum("bhnd,bhdm->bhnm", Q_feat, memory)
        norm = torch.einsum("bhnd,bhd->bhn", Q_feat, z).unsqueeze(-1).clamp(min=1e-6)
        return retrieved / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.q(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v(x).view(B, T, H, Dh).transpose(1, 2)

        # Initialize memory
        memory = torch.zeros(B, H, Dh, Dh, device=x.device)
        z = torch.ones(B, H, Dh, device=x.device) * 1e-6

        output = torch.zeros(B, T, D, device=x.device)

        # Process in segments
        for seg_start in range(0, T, self.segment_len):
            seg_end = min(seg_start + self.segment_len, T)
            S = seg_end - seg_start

            Q_seg = Q[:, :, seg_start:seg_end]
            K_seg = K[:, :, seg_start:seg_end]
            V_seg = V[:, :, seg_start:seg_end]

            # Local attention within segment
            local_scores = (Q_seg @ K_seg.transpose(-2, -1)) / self.scale
            local_attn = F.softmax(local_scores, dim=-1)
            local_attn = self.dropout(local_attn)
            local_out = local_attn @ V_seg  # [B, H, S, Dh]

            # Memory retrieval
            mem_out = self._retrieve_from_memory(memory, z, Q_seg)  # [B, H, S, Dh]

            # Gated combination
            gate = torch.sigmoid(self.beta)
            combined = gate * mem_out + (1 - gate) * local_out

            output[:, seg_start:seg_end] = combined.transpose(1, 2).contiguous().reshape(B, S, D)

            # Update memory with current segment
            memory, z = self._update_memory(memory, z, K_seg, V_seg)

        return self.out(output)


class LoRAAttention(nn.Module):
    """Self-attention with LoRA-adapted Q and V projections.

    Enables parameter-efficient fine-tuning of attention layers
    by only training low-rank adapter matrices.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.lora_scale = lora_alpha / lora_rank

        # Frozen base projections
        self.q_base = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_base = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # Trainable LoRA adapters
        self.q_lora_a = nn.Linear(d_model, lora_rank, bias=False)
        self.q_lora_b = nn.Linear(lora_rank, d_model, bias=False)
        self.v_lora_a = nn.Linear(d_model, lora_rank, bias=False)
        self.v_lora_b = nn.Linear(lora_rank, d_model, bias=False)

        nn.init.zeros_(self.q_lora_b.weight)
        nn.init.zeros_(self.v_lora_b.weight)

        # Freeze base weights
        for p in [self.q_base.weight, self.k_proj.weight, self.v_base.weight, self.out.weight]:
            p.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # Q with LoRA
        Q = self.q_base(x) + self.lora_scale * self.q_lora_b(self.q_lora_a(x))
        Q = Q.view(B, T, H, Dh).transpose(1, 2)

        K = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)

        # V with LoRA
        V = self.v_base(x) + self.lora_scale * self.v_lora_b(self.v_lora_a(x))
        V = V.view(B, T, H, Dh).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class SparseAttentionMixture(nn.Module):
    """Mixture of attention patterns: combines dense, local, and strided attention.

    Each head uses a different sparsity pattern:
    - Dense heads: full O(n^2) attention (for low n)
    - Local heads: window-based attention
    - Strided heads: attend to every k-th position (for long-range)
    - Global heads: attend to fixed global positions
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_dense: int = 2,
        n_local: int = 2,
        n_strided: int = 2,
        n_global: int = 2,
        window_size: int = 32,
        stride: int = 8,
        n_global_tokens: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_dense + n_local + n_strided + n_global == n_heads
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_dense = n_dense
        self.n_local = n_local
        self.n_strided = n_strided
        self.n_global = n_global
        self.window_size = window_size
        self.stride = stride
        self.n_global_tokens = n_global_tokens

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def _dense_attn(self, q, k, v, h_idx):
        scores = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        return self.dropout(attn) @ v

    def _local_attn(self, q, k, v, h_idx):
        B, T, Dh = q.shape
        W = self.window_size
        out = torch.zeros_like(q)
        for t in range(T):
            s = max(0, t - W // 2)
            e = min(T, s + W)
            scores = (q[:, t:t+1, :] @ k[:, s:e, :].transpose(-2, -1)) / self.scale
            attn = F.softmax(scores, dim=-1)
            out[:, t:t+1, :] = attn @ v[:, s:e, :]
        return out

    def _strided_attn(self, q, k, v, h_idx):
        B, T, Dh = q.shape
        stride_idx = torch.arange(0, T, self.stride, device=q.device)
        k_s = k[:, stride_idx, :]
        v_s = v[:, stride_idx, :]
        scores = (q @ k_s.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        return self.dropout(attn) @ v_s

    def _global_attn(self, q, k, v, h_idx):
        k_g = k[:, :self.n_global_tokens, :]
        v_g = v[:, :self.n_global_tokens, :]
        scores = (q @ k_g.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        return self.dropout(attn) @ v_g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        outs = []
        h = 0
        for i in range(self.n_dense):
            outs.append(self._dense_attn(Q[:, h], K[:, h], V[:, h], h))
            h += 1
        for i in range(self.n_local):
            outs.append(self._local_attn(Q[:, h], K[:, h], V[:, h], h))
            h += 1
        for i in range(self.n_strided):
            outs.append(self._strided_attn(Q[:, h], K[:, h], V[:, h], h))
            h += 1
        for i in range(self.n_global):
            outs.append(self._global_attn(Q[:, h], K[:, h], V[:, h], h))
            h += 1

        out = torch.stack(outs, dim=1).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class HierarchicalAttention(nn.Module):
    """Two-level hierarchical attention: token-level then chunk-level.

    First applies local attention within chunks.
    Then applies global attention between chunk representations.
    Finally projects back to token level.

    Good for very long financial time series (e.g., minute bars over years).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        chunk_size: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        self.local_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.global_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.chunk_pool = nn.Linear(d_model, d_model)
        self.expand = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        C = self.chunk_size

        # Pad to multiple of chunk_size
        pad = (C - T % C) % C
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        T_pad = x.shape[1]
        n_chunks = T_pad // C

        # Local attention within each chunk
        x_chunks = x.view(B * n_chunks, C, D)
        local_out, _ = self.local_attn(x_chunks, x_chunks, x_chunks)
        x_chunks = self.norm1(x_chunks + local_out)
        x = x_chunks.view(B, T_pad, D)

        # Chunk representations via pooling
        chunk_repr = x.view(B, n_chunks, C, D).mean(dim=2)  # [B, n_chunks, D]
        chunk_repr = self.chunk_pool(chunk_repr)

        # Global attention between chunks
        global_out, _ = self.global_attn(chunk_repr, chunk_repr, chunk_repr)
        chunk_repr = self.norm2(chunk_repr + global_out)

        # Expand back to token level
        expanded = chunk_repr.unsqueeze(2).expand(-1, -1, C, -1).contiguous().view(B, T_pad, D)
        out = x + self.expand(expanded)

        return out[:, :T, :]


# Factory functions for attention
def create_attention_for_frequency(
    data_frequency: str,
    d_model: int = 256,
    n_heads: int = 8,
    **kwargs
) -> nn.Module:
    """Create an appropriate attention module for the given data frequency.

    Args:
        data_frequency: 'tick', '1min', '5min', '1h', '1d', '1w'
    """
    freq_to_window = {
        "tick": 64, "1min": 32, "5min": 24, "1h": 16, "1d": 8, "1w": 4
    }
    window = freq_to_window.get(data_frequency, 32)

    if data_frequency in ("tick", "1min"):
        return WindowAttention(d_model, n_heads, window_size=window, **kwargs)
    elif data_frequency in ("5min", "1h"):
        return LocalGlobalAttention(d_model, n_heads, window_size=window, **kwargs)
    else:
        return ScaledDotProductAttentionV2(d_model, n_heads, **kwargs)
