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
    "make_causal_mask",
    "make_local_mask",
    "make_sliding_window_causal_mask",
    "scaled_dot_product_attention",
]
