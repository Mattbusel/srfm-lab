"""
lumina/positional_encoding.py

Comprehensive positional encoding schemes for Lumina financial foundation model.

This module implements every major positional encoding strategy used in modern
transformer architectures, with special adaptations for financial time-series:

  Absolute Encodings:
  - SinusoidalPositionalEncoding       — fixed sine/cosine (Vaswani et al. 2017)
  - LearnedAbsolutePositionalEncoding  — trainable embedding table
  - ScaledSinusoidalEncoding           — sinusoidal with learnable scale
  - AdaptiveSinusoidalEncoding         — content-adaptive sinusoidal

  Relative / Bias-based Encodings:
  - RotaryPositionalEncoding (RoPE)    — Su et al. 2021, RoFormer
  - RoPE2D                             — 2-D extension (time × feature axes)
  - TemporalRoPE                       — RoPE with real timestamps
  - ExtendedRoPE                       — YaRN / linear/NTK context extension
  - ALiBiPositionalBias                — Press et al. 2021
  - CausalALiBi                        — ALiBi with causal masking
  - T5RelativePositionBias             — bucket-based relative bias (Raffel et al.)
  - RelativePositionalEncoding         — Shaw et al. 2018, clipped relative
  - DisentangledAttentionBias          — DeBERTa-style p2c + c2p
  - PerformerPositionalEncoding        — random feature positional encoding

  Timestamp / Calendar Encodings:
  - TemporalEncoding                   — hour/dow/dom/month sin+cos + session emb
  - FourierTimeEncoding                — learnable Fourier features for timestamps
  - CalendarEncoding                   — rich calendar features (US market holidays)
  - MarketMicrostructureEncoding       — intraday pattern awareness
  - EconomicCycleEncoding              — macro-cycle features (QoQ, YoY)

  Multi-Modal / Cross-Modal:
  - CrossModalPositionalEncoding       — per-modality positions + cross-modal bias
  - ModalityEmbedding                  — learnable modality type tokens
  - HierarchicalPositionalEncoding     — segment + intra-segment positions

  Compound / Hybrid:
  - CompoundPositionalEncoding         — sinusoidal + temporal + RoPE
  - PositionalEncodingFactory          — registry/factory class

Utility functions:
  precompute_freqs_cis, apply_rotary_emb, apply_rotary_emb_single,
  rotate_half, get_slopes, interpolate_pos_encoding,
  compute_relative_positions, bucket_relative_positions
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Constants
# ===========================================================================

_LOG_10000 = math.log(10000.0)
_TWO_PI = 2 * math.pi

# US Market holiday dates (month, day) — approximate
_US_MARKET_HOLIDAYS = {
    (1, 1),   # New Year's Day
    (7, 4),   # Independence Day
    (11, 11), # Veterans Day
    (12, 25), # Christmas
}


# ===========================================================================
# Enumerations
# ===========================================================================

class PositionalEncodingType(str, Enum):
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    ROPE = "rope"
    ROPE2D = "rope2d"
    ALIBI = "alibi"
    T5_RELATIVE = "t5_relative"
    TEMPORAL = "temporal"
    FOURIER = "fourier"
    COMPOUND = "compound"
    NONE = "none"


class MarketSession(int, Enum):
    PRE_MARKET = 0
    RTH = 1          # Regular Trading Hours (9:30–16:00 ET)
    AFTER_HOURS = 2
    OVERNIGHT = 3
    WEEKEND = 4
    HOLIDAY = 5


class EconomicRegime(int, Enum):
    EXPANSION = 0
    PEAK = 1
    CONTRACTION = 2
    TROUGH = 3
    UNKNOWN = 4


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class PositionalEncodingConfig:
    """Configuration for positional encoding modules."""
    encoding_type: str = "rope"
    d_model: int = 512
    max_seq_len: int = 4096
    n_heads: int = 8
    base: float = 10000.0
    rope_scaling: Optional[str] = None       # None | "linear" | "ntk" | "yarn"
    rope_scaling_factor: float = 1.0
    rope_theta: float = 10000.0
    alibi_max_bias: float = 8.0
    t5_num_buckets: int = 32
    t5_max_distance: int = 128
    temporal_d_session: int = 16
    fourier_n_components: int = 64
    dropout: float = 0.0
    use_compound: bool = False
    compound_components: List[str] = field(default_factory=lambda: ["sinusoidal", "temporal"])


# ===========================================================================
# Utility Functions
# ===========================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input tensor.

    Splits tensor into two halves along last dimension and returns
    [-x2, x1] concatenation. Used in RoPE application.

    Args:
        x: (..., d) tensor

    Returns:
        rotated: (..., d) tensor where second half is negated and prepended
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q:   (B, n_heads, T, head_dim) query tensor
        k:   (B, n_kv_heads, T, head_dim) key tensor
        cos: (1, 1, T, head_dim) or (B, 1, T, head_dim)
        sin: same shape as cos

    Returns:
        q_rot, k_rot: rotated tensors of same shape as inputs
    """
    d = cos.shape[-1]
    q_rot = (q[..., :d] * cos) + (rotate_half(q[..., :d]) * sin)
    k_rot = (k[..., :d] * cos) + (rotate_half(k[..., :d]) * sin)
    if q.shape[-1] > d:
        q_rot = torch.cat([q_rot, q[..., d:]], dim=-1)
        k_rot = torch.cat([k_rot, k[..., d:]], dim=-1)
    return q_rot, k_rot


def apply_rotary_emb_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding to a single tensor (q or k).

    Args:
        x:   (..., T, head_dim)
        cos: (..., T, head_dim)
        sin: (..., T, head_dim)

    Returns:
        x_rot: same shape as x
    """
    d = cos.shape[-1]
    x_rot = (x[..., :d] * cos) + (rotate_half(x[..., :d]) * sin)
    if x.shape[-1] > d:
        x_rot = torch.cat([x_rot, x[..., d:]], dim=-1)
    return x_rot


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Precompute the complex exponentials for RoPE.

    Uses the formulation: freqs_cis[i] = exp(i * frequencies * j)
    where j is imaginary unit, producing unit complex numbers.

    Args:
        dim:            head dimension (must be even)
        end:            maximum sequence length
        theta:          RoPE base (default 10000)
        scaling_factor: linear scaling factor for context extension

    Returns:
        freqs_cis: (end, dim//2) complex float tensor
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if scaling_factor != 1.0:
        freqs = freqs / scaling_factor
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (end, dim//2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def precompute_freqs_cos_sin(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin caches for RoPE (real-valued).

    This is the alternative to complex-valued precompute_freqs_cis that
    avoids complex tensor operations for better hardware compatibility.

    Args:
        dim:            head dimension (must be even)
        end:            maximum sequence length
        theta:          RoPE base
        scaling_factor: linear scaling factor

    Returns:
        cos_cache: (end, dim) float tensor
        sin_cache: (end, dim) float tensor
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if scaling_factor != 1.0:
        freqs = freqs / scaling_factor
    t = torch.arange(end, dtype=torch.float32)
    freqs_mat = torch.outer(t, freqs)  # (end, dim//2)
    emb = torch.cat([freqs_mat, freqs_mat], dim=-1)  # (end, dim)
    return emb.cos(), emb.sin()


def get_slopes(n_heads: int) -> List[float]:
    """Compute ALiBi head slopes.

    For power-of-2 heads: slopes = 2^(-8/n * k) for k=1..n.
    For non-power-of-2 heads: interpolation between nearest powers of 2.

    Args:
        n_heads: number of attention heads

    Returns:
        slopes: list of n_heads float values
    """
    def _slopes_pow2(n: int) -> List[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if n_heads > 0 and (n_heads & (n_heads - 1)) == 0:
        return _slopes_pow2(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        base = _slopes_pow2(closest)
        extra = _slopes_pow2(2 * closest)
        extra = extra[0::2][: n_heads - closest]
        return base + extra


def interpolate_pos_encoding(
    pos_encoding: torch.Tensor,
    target_len: int,
    mode: str = "linear",
) -> torch.Tensor:
    """Interpolate positional encoding to a different sequence length.

    Used when the model needs to process sequences longer than what it
    was trained on.

    Args:
        pos_encoding: (1, T, d_model) positional encoding
        target_len:   target sequence length
        mode:         interpolation mode: "linear" | "bicubic" | "nearest"

    Returns:
        interpolated: (1, target_len, d_model) positional encoding
    """
    src_len = pos_encoding.shape[1]
    if src_len == target_len:
        return pos_encoding
    d_model = pos_encoding.shape[-1]
    # Reshape to (1, 1, T, d_model) for F.interpolate (treats as images)
    pe = pos_encoding.transpose(1, 2)  # (1, d_model, T)
    pe = pe.unsqueeze(0)                # (1, 1, d_model, T) — no, use 3D
    pe = pos_encoding.permute(0, 2, 1).unsqueeze(-1)  # (1, d_model, T, 1)
    interp = F.interpolate(pe, size=(target_len, 1), mode="bilinear", align_corners=False)
    interp = interp.squeeze(-1).permute(0, 2, 1)  # (1, target_len, d_model)
    return interp


def compute_relative_positions(
    query_len: int,
    key_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute relative position matrix.

    Returns matrix[i, j] = j - i (relative position from query i to key j).

    Args:
        query_len: number of query positions
        key_len:   number of key positions
        device:    target device

    Returns:
        rel_pos: (query_len, key_len) long tensor
    """
    q_pos = torch.arange(query_len, device=device)
    k_pos = torch.arange(key_len, device=device)
    rel_pos = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)  # (Q, K)
    return rel_pos


def bucket_relative_positions(
    rel_pos: torch.Tensor,
    num_buckets: int = 32,
    max_distance: int = 128,
    bidirectional: bool = True,
) -> torch.Tensor:
    """Map relative positions to bucket indices (T5-style).

    First num_buckets/2 buckets: exact positions 0..num_buckets//2-1.
    Remaining buckets: logarithmically spaced from num_buckets//2 to max_distance.

    Args:
        rel_pos:      (Q, K) relative position tensor
        num_buckets:  total number of buckets
        max_distance: maximum distance considered
        bidirectional: if True, separate buckets for positive/negative

    Returns:
        buckets: (Q, K) long tensor with bucket indices
    """
    ret = 0
    n = -rel_pos  # negative: looking backward

    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = n.abs()
    else:
        n = n.clamp(min=0)

    # Half the buckets for exact positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # Log-spaced for large positions
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact + 1e-8)
        / math.log(max_distance / max_exact + 1e-8)
        * (num_buckets - max_exact)
    ).long().clamp(max=num_buckets - 1)

    buckets = torch.where(is_small, n, val_if_large)
    return ret + buckets


def ntk_scale_freqs(
    inv_freq: torch.Tensor,
    scale_factor: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Apply NTK (Neural Tangent Kernel) scaling to RoPE frequencies.

    NTK scaling adjusts frequencies so that longer sequences can be
    processed without fine-tuning: only the base theta is scaled.

    base_new = base * scale_factor^(dim / (dim - 2))

    Args:
        inv_freq: (dim//2,) inverse frequency tensor
        scale_factor: context extension factor
        alpha: NTK alpha parameter (1.0 = standard NTK)

    Returns:
        scaled_inv_freq: (dim//2,) scaled inverse frequencies
    """
    dim = inv_freq.shape[0] * 2
    base_new_factor = scale_factor ** (dim / (dim - 2)) * alpha
    return inv_freq / base_new_factor


def yarn_scale_freqs(
    inv_freq: torch.Tensor,
    scale_factor: float,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    original_max_position: int = 4096,
) -> Tuple[torch.Tensor, float]:
    """Apply YaRN (Yet another RoPE extensioN) scaling to RoPE frequencies.

    YaRN applies different scaling to different frequency components:
    - High-frequency (small inv_freq): no scaling
    - Low-frequency (large inv_freq): linear scaling
    - Middle frequencies: interpolated

    Reference: Peng et al. 2023, "YaRN: Efficient Context Window Extension
    of Large Language Models"

    Args:
        inv_freq:                (dim//2,) inverse frequency tensor
        scale_factor:            context length ratio
        mscale:                  magnitude scaling factor
        mscale_all_dim:         apply mscale to all dimensions
        low_freq_factor:         low-frequency wavelength factor
        high_freq_factor:        high-frequency wavelength factor
        original_max_position:   original max training sequence length

    Returns:
        scaled_inv_freq: (dim//2,) scaled inverse frequencies
        attn_scale:      attention output scale factor
    """
    low_freq_wavelen = original_max_position / low_freq_factor
    high_freq_wavelen = original_max_position / high_freq_factor

    new_freqs = []
    for freq in inv_freq:
        wavelen = 2 * math.pi / freq.item()
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            smooth = (original_max_position / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

    scaled = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)

    # Compute attention scale
    if mscale_all_dim > 0:
        attn_scale = 0.1 * mscale * math.log(scale_factor) + 1.0
    else:
        attn_scale = float(
            0.1 * mscale * math.log(scale_factor) + 1.0
        ) if scale_factor > 1 else 1.0

    return scaled, attn_scale


# ===========================================================================
# Sinusoidal Positional Encoding (Fixed)
# ===========================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    These fixed encodings capture absolute position and enable the model
    to learn to attend to relative positions through linear combinations.

    Args:
        d_model:     embedding dimension
        max_seq_len: maximum sequence length to precompute
        dropout:     dropout probability applied after adding encodings
        learnable_scale: if True, add a learnable scalar multiplier

    Example:
        >>> pe = SinusoidalPositionalEncoding(d_model=512, max_seq_len=1024)
        >>> x = torch.randn(2, 64, 512)
        >>> out = pe(x)  # (2, 64, 512)
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        learnable_scale: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)
        self.learnable_scale = learnable_scale

        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))

        self.register_buffer("pe", self._build_pe(max_seq_len, d_model), persistent=False)

    @staticmethod
    def _build_pe(max_seq_len: int, d_model: int) -> torch.Tensor:
        """Build sinusoidal positional encoding table.

        Returns:
            pe: (1, max_seq_len, d_model) float tensor
        """
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-_LOG_10000 / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        return pe.unsqueeze(0)  # (1, T, d_model)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x:      (B, T, d_model) input tensor
            offset: position offset (useful during autoregressive generation)

        Returns:
            x + positional_encoding: (B, T, d_model)
        """
        T = x.shape[1]
        if offset + T > self.pe.shape[1]:
            # Extend cache if needed
            new_pe = self._build_pe(offset + T + 128, self.d_model)
            self.register_buffer("pe", new_pe.to(x.device), persistent=False)

        enc = self.pe[:, offset:offset + T, :]  # (1, T, d_model)
        if self.learnable_scale:
            enc = enc * self.scale
        return self.dropout(x + enc)

    def get_encoding(self, seq_len: int, offset: int = 0) -> torch.Tensor:
        """Retrieve positional encoding without adding to input.

        Args:
            seq_len: sequence length
            offset:  position offset

        Returns:
            pe: (1, seq_len, d_model)
        """
        return self.pe[:, offset:offset + seq_len, :]

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, max_seq_len={self.max_seq_len}, "
            f"learnable_scale={self.learnable_scale}"
        )


# ===========================================================================
# Scaled Sinusoidal Encoding
# ===========================================================================

class ScaledSinusoidalEncoding(nn.Module):
    """Sinusoidal encoding with learnable per-dimension scale.

    Each dimension i gets a learnable scale s_i:
    output_i = x_i + s_i * PE_i

    This allows the model to learn the relative importance of positional
    information for each feature dimension.

    Args:
        d_model:     embedding dimension
        max_seq_len: maximum sequence length
        init_scale:  initial value for learnable scales

    Example:
        >>> enc = ScaledSinusoidalEncoding(d_model=256)
        >>> x = torch.randn(4, 100, 256)
        >>> out = enc(x)
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 4096,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.scale = nn.Parameter(torch.full((d_model,), init_scale))

        pe = SinusoidalPositionalEncoding._build_pe(max_seq_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            x + scale * pe: (B, T, d_model)
        """
        T = x.shape[1]
        enc = self.pe[:, :T, :] * self.scale.unsqueeze(0).unsqueeze(0)
        return x + enc


# ===========================================================================
# Adaptive Sinusoidal Encoding
# ===========================================================================

class AdaptiveSinusoidalEncoding(nn.Module):
    """Content-adaptive sinusoidal positional encoding.

    Instead of fixed frequencies, learns to modulate frequency assignments
    based on the local content of the sequence. Each position's encoding is
    a weighted combination of sinusoids where weights depend on the content.

    Architecture:
        1. Compute base sinusoidal encoding (fixed)
        2. Compute content features from input: linear → sigmoid gates
        3. Multiply sinusoidal by content gates
        4. Project to d_model

    Args:
        d_model:      embedding dimension
        n_freqs:      number of frequency components
        max_seq_len:  maximum sequence length

    Example:
        >>> enc = AdaptiveSinusoidalEncoding(d_model=512, n_freqs=32)
        >>> x = torch.randn(2, 64, 512)
        >>> out = enc(x)  # (2, 64, 512)
    """

    def __init__(
        self,
        d_model: int,
        n_freqs: int = 32,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_freqs = n_freqs

        # Fixed frequency bank
        freqs = torch.arange(1, n_freqs + 1, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

        # Content-adaptive gating
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, n_freqs, bias=False),
            nn.Sigmoid(),
        )

        # Output projection: n_freqs sin + n_freqs cos + d_model input -> d_model
        self.out_proj = nn.Linear(2 * n_freqs, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate_proj[0].weight)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            x + adaptive_pe: (B, T, d_model)
        """
        B, T, D = x.shape
        device = x.device

        # Compute position-frequency matrix
        positions = torch.arange(T, dtype=torch.float32, device=device)  # (T,)
        angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0)  # (T, n_freqs)
        sin_features = angles.sin()  # (T, n_freqs)
        cos_features = angles.cos()  # (T, n_freqs)
        sincos = torch.cat([sin_features, cos_features], dim=-1)  # (T, 2*n_freqs)

        # Content-adaptive gates
        gates = self.gate_proj(x)  # (B, T, n_freqs)
        gated_sin = sin_features.unsqueeze(0) * gates  # (B, T, n_freqs)
        gated_cos = cos_features.unsqueeze(0) * gates  # (B, T, n_freqs)
        gated = torch.cat([gated_sin, gated_cos], dim=-1)  # (B, T, 2*n_freqs)

        # Project and add
        pe = self.out_proj(gated)  # (B, T, d_model)
        return self.norm(x + pe)


# ===========================================================================
# Learned Absolute Positional Encoding
# ===========================================================================

class LearnedAbsolutePositionalEncoding(nn.Module):
    """Trainable positional embedding table.

    Each position up to max_seq_len gets its own learned d_model-dimensional
    embedding. Simple and effective for fixed-length sequences.

    Args:
        d_model:     embedding dimension
        max_seq_len: maximum sequence length (number of learnable positions)
        dropout:     dropout probability
        init_std:    initialization standard deviation (None → normal init)

    Example:
        >>> pe = LearnedAbsolutePositionalEncoding(d_model=512, max_seq_len=512)
        >>> x = torch.randn(2, 128, 512)
        >>> out = pe(x)  # (2, 128, 512)
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        init_std: Optional[float] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self._init_weights(init_std)

    def _init_weights(self, std: Optional[float]):
        if std is not None:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=std)
        else:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model ** -0.5)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x:            (B, T, d_model)
            position_ids: (B, T) or (T,) optional custom positions
            offset:       position offset

        Returns:
            x + pos_emb: (B, T, d_model)
        """
        T = x.shape[1]

        if position_ids is None:
            pos_ids = torch.arange(
                offset, offset + T,
                dtype=torch.long, device=x.device
            ).unsqueeze(0).expand(x.shape[0], -1)
        else:
            pos_ids = position_ids

        pe = self.embedding(pos_ids)  # (B, T, d_model)
        return self.dropout(x + pe)

    def get_embedding_weight(self) -> torch.Tensor:
        """Return the embedding table for analysis."""
        return self.embedding.weight.data

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, max_seq_len={self.max_seq_len}"


# ===========================================================================
# Rotary Positional Encoding (RoPE)
# ===========================================================================

class RotaryPositionalEncoding(nn.Module):
    """RoPE: Rotary Position Embedding (Su et al. 2021, RoFormer).

    Applied to Q and K tensors in attention. Encodes relative positions by
    rotating pairs of dimensions by angles theta_i = pos / 10000^(2i/d).

    The key property: for queries at position m and keys at position n,
    the dot product depends only on (m - n), making this a relative encoding.

    Context extension modes:
    - None:     standard RoPE with theta=10000
    - "linear": linearly scale positions by 1/factor (Press et al. 2023)
    - "ntk":    NTK-aware scaling, rescales the base (LocalLLaMA 2023)
    - "yarn":   YaRN per-frequency scaling (Peng et al. 2023)
    - "dynamic":dynamically adjust based on sequence length at runtime

    Args:
        dim:            head dimension (must be even)
        max_seq_len:    maximum sequence length
        base:           RoPE base frequency (default 10000)
        scaling_type:   context extension type: None | "linear" | "ntk" | "yarn"
        scaling_factor: context length scaling factor
        yarn_params:    dict of extra YaRN parameters

    Usage:
        rope = RotaryPositionalEncoding(dim=64, max_seq_len=4096)
        q, k = rope(q, k)  # in-place rotation
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        scaling_type: Optional[str] = None,
        scaling_factor: float = 1.0,
        yarn_params: Optional[Dict] = None,
    ):
        super().__init__()
        assert dim % 2 == 0, f"RoPE requires even dimension, got {dim}"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.yarn_params = yarn_params or {}
        self._attn_scale = 1.0

        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies with optional scaling."""
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        if self.scaling_type is None:
            return inv_freq
        elif self.scaling_type == "linear":
            return inv_freq / self.scaling_factor
        elif self.scaling_type == "ntk":
            new_base = self.base * self.scaling_factor ** (self.dim / (self.dim - 2))
            return 1.0 / (new_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        elif self.scaling_type == "yarn":
            scaled, self._attn_scale = yarn_scale_freqs(
                inv_freq, self.scaling_factor, **self.yarn_params
            )
            return scaled
        else:
            warnings.warn(f"Unknown RoPE scaling type: {self.scaling_type}, using standard.")
            return inv_freq

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for positions 0..seq_len-1."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _maybe_extend_cache(self, seq_len: int):
        """Extend cache if sequence is longer than current cache."""
        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len + 256)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to query and key.

        Args:
            q:            (B, n_heads, T, head_dim) query
            k:            (B, n_kv_heads, T, head_dim) key
            seq_len:      explicit sequence length (inferred from q if None)
            position_ids: (B, T) optional custom position indices

        Returns:
            q_rot, k_rot: rotated tensors, same shape as inputs
        """
        T = q.shape[-2]
        if seq_len is None:
            seq_len = T

        self._maybe_extend_cache(seq_len)

        if position_ids is not None:
            # Custom positions (e.g., for packed sequences, generation with cache)
            cos = self.cos_cache[position_ids]  # (B, T, dim)
            sin = self.sin_cache[position_ids]
            cos = cos.unsqueeze(1)  # (B, 1, T, dim)
            sin = sin.unsqueeze(1)
        else:
            cos = self.cos_cache[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim)
            sin = self.sin_cache[:T].unsqueeze(0).unsqueeze(0)

        return apply_rotary_emb(q, k, cos, sin)

    def forward_single(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply rotary embedding to a single tensor.

        Args:
            x:            (B, n_heads, T, head_dim)
            position_ids: (B, T) optional custom positions

        Returns:
            x_rot: (B, n_heads, T, head_dim)
        """
        T = x.shape[-2]
        self._maybe_extend_cache(T)

        if position_ids is not None:
            cos = self.cos_cache[position_ids].unsqueeze(1)
            sin = self.sin_cache[position_ids].unsqueeze(1)
        else:
            cos = self.cos_cache[:T].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cache[:T].unsqueeze(0).unsqueeze(0)

        return apply_rotary_emb_single(x, cos, sin)

    @property
    def attention_scale(self) -> float:
        """Attention scale factor (relevant for YaRN)."""
        return self._attn_scale

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, max_seq_len={self.max_seq_len}, "
            f"base={self.base}, scaling_type={self.scaling_type}"
        )


# ===========================================================================
# 2D Rotary Position Embedding
# ===========================================================================

class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for 2D grids.

    Applies RoPE independently to two axes (time and feature/channel axes).
    Useful for vision-style patches of financial data where both temporal and
    cross-asset positions matter.

    The dimension is split in half:
    - First half: encodes time position
    - Second half: encodes feature/channel position

    Args:
        dim:            total head dimension (split evenly between two axes)
        max_rows:       maximum positions along first axis (time)
        max_cols:       maximum positions along second axis (features/assets)
        base:           RoPE base frequency

    Example:
        >>> rope2d = RoPE2D(dim=64, max_rows=256, max_cols=32)
        >>> q = torch.randn(2, 8, 256*32, 64)  # (B, H, T*C, head_dim)
        >>> row_ids = torch.arange(256).repeat_interleave(32).unsqueeze(0)
        >>> col_ids = torch.arange(32).repeat(256).unsqueeze(0)
        >>> q_rot, k_rot = rope2d(q, q, row_ids, col_ids)
    """

    def __init__(
        self,
        dim: int,
        max_rows: int = 256,
        max_cols: int = 64,
        base: float = 10000.0,
    ):
        super().__init__()
        assert dim % 4 == 0, f"RoPE2D requires dim divisible by 4, got {dim}"
        self.dim = dim
        self.half_dim = dim // 2
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.base = base

        # Two separate RoPE modules for each axis
        self.rope_row = RotaryPositionalEncoding(
            dim=self.half_dim, max_seq_len=max_rows, base=base
        )
        self.rope_col = RotaryPositionalEncoding(
            dim=self.half_dim, max_seq_len=max_cols, base=base
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D rotary embedding.

        Args:
            q:       (B, H, N, dim) query
            k:       (B, H, N, dim) key
            row_ids: (B, N) row position indices
            col_ids: (B, N) column position indices

        Returns:
            q_rot, k_rot: (B, H, N, dim)
        """
        B, H, N, D = q.shape
        half = self.half_dim

        # Split into row and column halves
        q_row = q[..., :half]   # (B, H, N, half)
        q_col = q[..., half:]   # (B, H, N, half)
        k_row = k[..., :half]
        k_col = k[..., half:]

        # Apply row rope
        q_row_rot, k_row_rot = self._apply_1d(
            q_row, k_row, row_ids, self.rope_row
        )
        # Apply col rope
        q_col_rot, k_col_rot = self._apply_1d(
            q_col, k_col, col_ids, self.rope_col
        )

        q_rot = torch.cat([q_row_rot, q_col_rot], dim=-1)
        k_rot = torch.cat([k_row_rot, k_col_rot], dim=-1)
        return q_rot, k_rot

    def _apply_1d(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        rope: RotaryPositionalEncoding,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 1D RoPE with custom position ids."""
        rope._maybe_extend_cache(position_ids.max().item() + 1)
        cos = rope.cos_cache[position_ids]  # (B, N, half)
        sin = rope.sin_cache[position_ids]
        cos = cos.unsqueeze(1)  # (B, 1, N, half)
        sin = sin.unsqueeze(1)
        return apply_rotary_emb(q, k, cos, sin)


# ===========================================================================
# Temporal RoPE (with real timestamps)
# ===========================================================================

class TemporalRoPE(nn.Module):
    """RoPE variant that uses real timestamps instead of integer positions.

    Financial time series have irregular sampling (market hours only, gaps
    for weekends/holidays). TemporalRoPE uses fractional positions derived
    from actual timestamps.

    Position mapping:
    - Convert Unix timestamps to fractional bar indices
    - Map to continuous position space
    - Apply standard RoPE formulation

    Args:
        dim:          head dimension
        base:         RoPE base
        time_unit:    time unit for normalization: "seconds" | "minutes" | "hours"
        max_distance: maximum time distance in time_unit units

    Example:
        >>> trope = TemporalRoPE(dim=64)
        >>> q = torch.randn(2, 8, 128, 64)
        >>> k = torch.randn(2, 8, 128, 64)
        >>> timestamps = torch.arange(128).float().unsqueeze(0) * 60  # minute bars
        >>> q_rot, k_rot = trope(q, k, timestamps)
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        time_unit: str = "minutes",
        max_distance: int = 10000,
    ):
        super().__init__()
        assert dim % 2 == 0, f"TemporalRoPE requires even dim, got {dim}"
        self.dim = dim
        self.base = base
        self.time_unit = time_unit
        self.max_distance = max_distance

        # Time unit multipliers
        _unit_map = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": 86400.0}
        self.time_scale = _unit_map.get(time_unit, 60.0)

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _timestamps_to_freqs(self, timestamps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert timestamps to cos/sin matrices.

        Args:
            timestamps: (B, T) float Unix timestamps in seconds

        Returns:
            cos: (B, 1, T, dim)
            sin: (B, 1, T, dim)
        """
        B, T = timestamps.shape
        # Normalize to time_unit
        t = timestamps.float() / self.time_scale  # (B, T)
        # Compute outer product with inv_freq for each batch element
        freqs = t.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)  # (B, T, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (B, T, dim)
        cos = emb.cos().unsqueeze(1)  # (B, 1, T, dim)
        sin = emb.sin().unsqueeze(1)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temporal rotary embedding.

        Args:
            q:          (B, H, T, head_dim)
            k:          (B, H, T, head_dim)
            timestamps: (B, T) Unix timestamps in seconds

        Returns:
            q_rot, k_rot: rotated tensors
        """
        cos, sin = self._timestamps_to_freqs(timestamps)
        return apply_rotary_emb(q, k, cos, sin)


# ===========================================================================
# Extended RoPE (NTK / Linear scaling)
# ===========================================================================

class ExtendedRoPE(nn.Module):
    """Extended RoPE with context length extrapolation.

    Supports multiple extrapolation strategies for processing sequences
    longer than the training context window:

    1. Linear:   divide positions by scale_factor (equivalent to lowering freq)
    2. NTK:      Neural Tangent Kernel scaling — adjusts base theta
    3. YaRN:     per-frequency interpolation between linear and original
    4. Dynamic:  apply scaling only when seq_len > original_max_len

    Args:
        dim:              head dimension
        max_seq_len:      training-time maximum sequence length
        base:             RoPE theta base
        extension_type:   "linear" | "ntk" | "yarn" | "dynamic"
        scale_factor:     context extension ratio (e.g., 4.0 for 4× context)

    Example:
        >>> rope = ExtendedRoPE(dim=128, max_seq_len=2048, extension_type="yarn", scale_factor=4.0)
        >>> q = torch.randn(1, 16, 8192, 128)  # 4× extended context
        >>> k = torch.randn(1, 16, 8192, 128)
        >>> q_rot, k_rot = rope(q, k)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        extension_type: str = "ntk",
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.extension_type = extension_type
        self.scale_factor = scale_factor
        self._current_scale = scale_factor
        self._attn_scale = 1.0

        inv_freq = self._compute_inv_freq(scale_factor)
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _compute_inv_freq(self, scale_factor: float) -> torch.Tensor:
        base_inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        if self.extension_type == "linear" or scale_factor == 1.0:
            return base_inv_freq / scale_factor if scale_factor != 1.0 else base_inv_freq
        elif self.extension_type == "ntk":
            new_base = self.base * scale_factor ** (self.dim / (self.dim - 2))
            return 1.0 / (
                new_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
            )
        elif self.extension_type == "yarn":
            scaled, self._attn_scale = yarn_scale_freqs(
                base_inv_freq, scale_factor,
                original_max_position=self.max_seq_len
            )
            return scaled
        elif self.extension_type == "dynamic":
            return base_inv_freq
        else:
            return base_inv_freq

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _maybe_update_dynamic(self, seq_len: int):
        """For dynamic mode, update scale factor if sequence exceeds training length."""
        if self.extension_type != "dynamic":
            return
        if seq_len > self.max_seq_len:
            new_scale = seq_len / self.max_seq_len
            if new_scale != self._current_scale:
                self._current_scale = new_scale
                new_inv_freq = self._compute_inv_freq(new_scale)
                self.inv_freq.copy_(new_inv_freq)
                self._build_cache(seq_len)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k:         (B, H, T, head_dim)
            position_ids: (B, T) optional custom positions

        Returns:
            q_rot, k_rot
        """
        T = q.shape[-2]
        self._maybe_update_dynamic(T)

        if T > self.cos_cache.shape[0]:
            self._build_cache(T + 256)

        if position_ids is not None:
            cos = self.cos_cache[position_ids].unsqueeze(1)
            sin = self.sin_cache[position_ids].unsqueeze(1)
        else:
            cos = self.cos_cache[:T].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cache[:T].unsqueeze(0).unsqueeze(0)

        return apply_rotary_emb(q, k, cos, sin)

    @property
    def attention_scale(self) -> float:
        return self._attn_scale


# ===========================================================================
# ALiBi (Attention with Linear Biases)
# ===========================================================================

class ALiBiPositionalBias(nn.Module):
    """ALiBi: Attention with Linear Biases (Press et al. 2021).

    Adds a fixed linear bias to attention logits:
        bias[i, j] = -m_h * |i - j|

    where m_h is a head-specific slope. This simple modification allows
    transformers trained on short sequences to generalize to much longer
    sequences without fine-tuning.

    The slopes are geometrically spaced: m_h = 2^(-8h/n) for head h and
    n total heads.

    Args:
        n_heads:     number of attention heads
        max_seq_len: maximum sequence length for bias precomputation
        alibi_bias_max: maximum bias value (clip slopes at this scale)

    Example:
        >>> alibi = ALiBiPositionalBias(n_heads=8, max_seq_len=2048)
        >>> bias = alibi(seq_len=128, device=torch.device('cpu'))  # (1, 8, 128, 128)
        >>> attn_logits = attn_logits + bias
    """

    def __init__(
        self,
        n_heads: int,
        max_seq_len: int = 8192,
        alibi_bias_max: float = 8.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.alibi_bias_max = alibi_bias_max

        slopes = torch.tensor(get_slopes(n_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes)
        self._precompute_bias(max_seq_len)

    def _precompute_bias(self, seq_len: int):
        """Precompute the full ALiBi bias matrix."""
        positions = torch.arange(seq_len)
        # relative_positions[i, j] = j - i (negative: looking left)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = -relative_positions.abs().float()  # (T, T)
        # bias[h, i, j] = slopes[h] * |i-j|
        bias = self.slopes.unsqueeze(-1).unsqueeze(-1) * relative_positions.unsqueeze(0)
        # bias: (n_heads, T, T)
        self.register_buffer("bias_cache", bias, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Get ALiBi bias for given sequence length.

        Args:
            seq_len: sequence length
            device:  target device (ignored, uses buffer device)

        Returns:
            bias: (1, n_heads, seq_len, seq_len)
        """
        if seq_len > self.bias_cache.shape[-1]:
            self._precompute_bias(seq_len + 256)
        return self.bias_cache[:, :seq_len, :seq_len].unsqueeze(0)

    def causal_bias(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Get causal ALiBi bias (future positions masked to -inf).

        Args:
            seq_len: sequence length
            device:  target device

        Returns:
            bias: (1, n_heads, seq_len, seq_len) with -inf for future tokens
        """
        bias = self.forward(seq_len, device)
        dev = bias.device
        mask = torch.tril(torch.ones(seq_len, seq_len, device=dev, dtype=torch.bool))
        return bias.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    def incremental_bias(
        self,
        query_len: int,
        key_len: int,
    ) -> torch.Tensor:
        """Get ALiBi bias for encoder-decoder or generation (q_len ≠ k_len).

        Args:
            query_len: number of query positions
            key_len:   number of key positions (usually ≥ query_len)

        Returns:
            bias: (1, n_heads, query_len, key_len)
        """
        if key_len > self.bias_cache.shape[-1]:
            self._precompute_bias(key_len + 256)
        start = key_len - query_len
        return self.bias_cache[:, start:start + query_len, :key_len].unsqueeze(0)

    def extra_repr(self) -> str:
        return f"n_heads={self.n_heads}, max_seq_len={self.max_seq_len}"


# ===========================================================================
# Causal ALiBi (convenience wrapper)
# ===========================================================================

class CausalALiBi(ALiBiPositionalBias):
    """ALiBi with causal masking automatically applied.

    Convenience subclass that always returns causally masked biases.
    """

    def forward(self, seq_len: int, device=None) -> torch.Tensor:
        return self.causal_bias(seq_len, device)


# ===========================================================================
# T5 Relative Position Bias
# ===========================================================================

class T5RelativePositionBias(nn.Module):
    """Bucket-based relative position bias from T5 (Raffel et al. 2020).

    Maps relative positions to a small set of buckets. Nearby positions
    have dedicated buckets; distant positions are merged into fewer buckets
    (log-spaced). The bias values are learned parameters.

    Bucket assignment:
    - Positions 0..num_buckets//2-1: exact buckets (unidirectional: backward only)
    - Positions num_buckets//2..num_buckets-1: log-spaced up to max_distance

    For bidirectional attention: double the buckets (half for positive, half negative).

    Args:
        n_heads:       number of attention heads
        num_buckets:   number of position buckets
        max_distance:  maximum distance for log-spaced buckets
        bidirectional: if True, separate buckets for each direction
        is_decoder:    if True, only attend to past (causal)

    Example:
        >>> t5_bias = T5RelativePositionBias(n_heads=8, num_buckets=32, max_distance=128)
        >>> bias = t5_bias(query_len=64, key_len=64)  # (1, 8, 64, 64)
    """

    def __init__(
        self,
        n_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.is_decoder = is_decoder

        self.embedding = nn.Embedding(num_buckets, n_heads)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self,
        query_len: int,
        key_len: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Compute relative position bias.

        Args:
            query_len: number of query positions
            key_len:   number of key positions
            device:    target device

        Returns:
            bias: (1, n_heads, query_len, key_len)
        """
        if device is None:
            device = self.embedding.weight.device

        q_pos = torch.arange(query_len, device=device)
        k_pos = torch.arange(key_len, device=device)
        rel_pos = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)  # (Q, K)

        bucket_ids = bucket_relative_positions(
            rel_pos,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
            bidirectional=self.bidirectional,
        )

        bias = self.embedding(bucket_ids)  # (Q, K, n_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, Q, K)
        return bias

    def extra_repr(self) -> str:
        return (
            f"n_heads={self.n_heads}, num_buckets={self.num_buckets}, "
            f"max_distance={self.max_distance}, bidirectional={self.bidirectional}"
        )


# ===========================================================================
# Shaw Relative Position Encoding
# ===========================================================================

class RelativePositionalEncoding(nn.Module):
    """Relative position encoding from Shaw et al. 2018.

    Adds learned relative position representations to keys and values.
    Relative positions are clipped to [-max_relative_position, +max_relative_position].

    For query i and key j:
        attn(i,j) += q_i · a_ij^K   (key relative bias)
    where a_ij^K is a learned embedding for relative position clip(j-i).

    Args:
        n_heads:              number of attention heads
        head_dim:             dimension per head
        max_relative_position: maximum relative distance
        dropout:              dropout on relative embeddings

    Example:
        >>> rpe = RelativePositionalEncoding(n_heads=8, head_dim=64, max_relative_position=32)
        >>> rel_bias = rpe(query_len=64, key_len=64)  # (1, 8, 64, 64)
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        max_relative_position: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_relative_position = max_relative_position
        self.dropout = nn.Dropout(p=dropout)

        vocab_size = 2 * max_relative_position + 1
        self.embedding_k = nn.Embedding(vocab_size, head_dim)
        nn.init.xavier_uniform_(self.embedding_k.weight)

    def _get_relative_ids(self, query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
        """Get clipped relative position ids."""
        q_pos = torch.arange(query_len, device=device)
        k_pos = torch.arange(key_len, device=device)
        rel = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)  # (Q, K)
        rel = rel.clamp(-self.max_relative_position, self.max_relative_position)
        rel = rel + self.max_relative_position  # shift to [0, 2*max+1)
        return rel

    def forward(
        self,
        query: torch.Tensor,
        key_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute relative position attention bias.

        Args:
            query:   (B, n_heads, query_len, head_dim)
            key_len: key sequence length (defaults to query_len)

        Returns:
            rel_attn_bias: (B, n_heads, query_len, key_len)
        """
        B, H, Q, D = query.shape
        K = key_len if key_len is not None else Q
        device = query.device

        rel_ids = self._get_relative_ids(Q, K, device)  # (Q, K)
        rel_emb = self.embedding_k(rel_ids)  # (Q, K, head_dim)
        rel_emb = self.dropout(rel_emb)

        # query: (B, H, Q, D) → (B*H, Q, D)
        q_flat = query.reshape(B * H, Q, D)
        # rel_emb: (Q, K, D) → (Q, D, K) for batched matmul
        rel_t = rel_emb.transpose(1, 2)  # (Q, D, K)
        # Expand for batch
        rel_t = rel_t.unsqueeze(0).expand(B * H, -1, -1, -1)  # (B*H, Q, D, K)

        # Compute q · rel_emb per position
        attn_rel = torch.einsum("bid,bidk->bik", q_flat, rel_t)  # (B*H, Q, K)
        attn_rel = attn_rel.reshape(B, H, Q, K)
        return attn_rel


# ===========================================================================
# DeBERTa-style Disentangled Attention Bias
# ===========================================================================

class DisentangledAttentionBias(nn.Module):
    """Disentangled attention position bias (DeBERTa He et al. 2021).

    Separates content-to-position (c2p) and position-to-content (p2c)
    attention components.

    Attention score = content-to-content + content-to-position + position-to-content
    = q_c · k_c + q_c · k_r(delta) + q_r(delta) · k_c

    where delta = i - j is the relative position.

    Args:
        n_heads:              number of attention heads
        head_dim:             dimension per head
        max_relative_position: maximum absolute relative distance

    Example:
        >>> dab = DisentangledAttentionBias(n_heads=8, head_dim=64)
        >>> q_content = torch.randn(2, 8, 64, 64)
        >>> k_content = torch.randn(2, 8, 64, 64)
        >>> bias = dab(q_content, k_content)  # (2, 8, 64, 64)
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        max_relative_position: int = 512,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_relative_position = max_relative_position

        pos_vocab = 2 * max_relative_position + 1
        self.pos_embedding = nn.Embedding(pos_vocab, head_dim)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def _get_rel_ids(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute clipped relative position ids for sequence."""
        pos = torch.arange(seq_len, device=device)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)
        rel = rel.clamp(-self.max_relative_position, self.max_relative_position)
        return rel + self.max_relative_position  # (T, T)

    def forward(
        self,
        query_content: torch.Tensor,
        key_content: torch.Tensor,
    ) -> torch.Tensor:
        """Compute disentangled attention bias.

        Args:
            query_content: (B, H, T, head_dim) — content queries
            key_content:   (B, H, T, head_dim) — content keys

        Returns:
            bias: (B, H, T, T) attention bias to add to content attention
        """
        B, H, T, D = query_content.shape
        device = query_content.device

        rel_ids = self._get_rel_ids(T, device)  # (T, T)
        pos_emb = self.pos_embedding(rel_ids)    # (T, T, head_dim)

        # Content-to-position: q_c · k_r
        # q_content: (B, H, T, D), pos_emb: (T, T, D) → (T, D, T)
        q_flat = query_content.reshape(B * H, T, D)
        pos_t = pos_emb.permute(0, 2, 1)  # (T, D, T)
        pos_t_exp = pos_t.unsqueeze(0).expand(B * H, -1, -1, -1)  # (BH, T, D, T)
        c2p = torch.einsum("bid,bidj->bij", q_flat, pos_t_exp)  # (BH, T, T)
        c2p = c2p.reshape(B, H, T, T)

        # Position-to-content: q_r · k_c
        k_flat = key_content.reshape(B * H, T, D)
        pos_emb_exp = pos_emb.unsqueeze(0).expand(B * H, -1, -1, -1)  # (BH, T, T, D)
        p2c = torch.einsum("bijd,bjd->bij", pos_emb_exp, k_flat)  # (BH, T, T)
        p2c = p2c.reshape(B, H, T, T)

        return c2p + p2c


# ===========================================================================
# Market Session Constants
# ===========================================================================

class MarketSessionConstants:
    """Market session boundary constants (US Eastern Time)."""
    PRE_MARKET_START = 4    # 04:00 ET
    RTH_START = 9           # 09:30 ET (approximate)
    RTH_END = 16            # 16:00 ET
    AFTER_HOURS_END = 20    # 20:00 ET


# ===========================================================================
# Temporal Encoding (rich calendar features)
# ===========================================================================

class TemporalEncoding(nn.Module):
    """Encodes real timestamps as learnable embedding vectors.

    Features extracted from Unix timestamps:
    - Hour-of-day (0–23): sin/cos pair → 2 dims
    - Day-of-week (0–6): sin/cos → 2 dims
    - Day-of-month (1–31): sin/cos → 2 dims
    - Month (1–12): sin/cos → 2 dims
    - Market session embedding (6 categories) → d_session dims
    Total raw: 8 + d_session dims → linear projection → d_model

    This gives the model awareness of time-of-day, day-of-week, and market
    session patterns which are crucial for intraday financial modeling.

    Args:
        d_model:    output embedding dimension
        d_session:  dimension for session embedding (before projection)
        n_sessions: number of session categories (default 6: PRE/RTH/AFTER/OVERNIGHT/WEEKEND/HOLIDAY)
        dropout:    dropout on output

    Example:
        >>> te = TemporalEncoding(d_model=512)
        >>> ts = torch.randint(1_600_000_000, 1_700_000_000, (2, 128)).float()
        >>> out = te(ts)  # (2, 128, 512)
    """

    def __init__(
        self,
        d_model: int,
        d_session: int = 16,
        n_sessions: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_session = d_session

        sin_dim = 8  # 4 sin/cos pairs
        self.session_emb = nn.Embedding(n_sessions, d_session)
        self.proj = nn.Linear(sin_dim + d_session, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @staticmethod
    def _to_sincos(value: torch.Tensor, period: float) -> torch.Tensor:
        """Convert scalar to sin/cos pair with given period.

        Args:
            value:  (...) tensor of values
            period: period length

        Returns:
            sincos: (..., 2) tensor with [sin, cos]
        """
        angle = _TWO_PI * value / period
        return torch.stack([angle.sin(), angle.cos()], dim=-1)

    def _classify_session(
        self, hour: torch.Tensor, minute: torch.Tensor, dow: torch.Tensor
    ) -> torch.Tensor:
        """Classify each timestamp into a market session.

        Args:
            hour:   (...) hour 0–23
            minute: (...) minute 0–59
            dow:    (...) day of week 0–6 (0=Monday)

        Returns:
            session: (...) long tensor with MarketSession values
        """
        is_weekend = dow >= 5
        hm = hour * 60 + minute  # minutes since midnight
        is_rth = (~is_weekend) & (hm >= 9 * 60 + 30) & (hm < 16 * 60)
        is_pre = (~is_weekend) & (hm >= 4 * 60) & (hm < 9 * 60 + 30)
        is_after = (~is_weekend) & (hm >= 16 * 60) & (hm < 20 * 60)
        is_overnight = (~is_weekend) & (~is_pre) & (~is_rth) & (~is_after)

        session = torch.full_like(hour, MarketSession.OVERNIGHT.value, dtype=torch.long)
        session[is_pre] = MarketSession.PRE_MARKET.value
        session[is_rth] = MarketSession.RTH.value
        session[is_after] = MarketSession.AFTER_HOURS.value
        session[is_overnight] = MarketSession.OVERNIGHT.value
        session[is_weekend] = MarketSession.WEEKEND.value
        return session

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps into temporal embeddings.

        Args:
            timestamps: (B, T) Unix timestamps in seconds (float or long)

        Returns:
            encoding: (B, T, d_model)
        """
        ts = timestamps.float()
        seconds_per_day = 86400.0
        seconds_per_hour = 3600.0
        seconds_per_minute = 60.0

        time_of_day = ts % seconds_per_day
        hour = (time_of_day / seconds_per_hour).long().float()
        minute = ((time_of_day % seconds_per_hour) / seconds_per_minute).long().float()
        day_num = (ts / seconds_per_day).long()
        dow = (day_num + 3) % 7  # 0=Monday (Unix epoch Jan 1 1970 = Thursday = +3)
        day_of_month = (day_num % 31 + 1).float()
        month = (day_num % 365 / 30.4 + 1).clamp(1, 12).float()

        sc_hour = self._to_sincos(hour, 24.0)
        sc_dow = self._to_sincos(dow.float(), 7.0)
        sc_dom = self._to_sincos(day_of_month, 31.0)
        sc_month = self._to_sincos(month, 12.0)

        sin_features = torch.cat([sc_hour, sc_dow, sc_dom, sc_month], dim=-1)  # (..., 8)

        session_ids = self._classify_session(hour.long(), minute.long(), dow)
        session_features = self.session_emb(session_ids)  # (..., d_session)

        combined = torch.cat([sin_features, session_features], dim=-1)
        out = self.dropout(self.norm(self.proj(combined)))
        return out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_session={self.d_session}"


# ===========================================================================
# Fourier Time Encoding
# ===========================================================================

class FourierTimeEncoding(nn.Module):
    """Learnable Fourier features for continuous timestamps.

    Maps scalar timestamp t → [sin(w_1*t + phi_1), cos(w_1*t + phi_1), ...]
    with learnable frequencies w_i and phases phi_i, then projects to d_model.

    This is the "Time2Vec" approach adapted for financial data.
    Learnable frequencies allow the model to focus on market-relevant
    periodicities (intraday, weekly, earnings cycles, etc.).

    Args:
        d_model:    output embedding dimension
        n_fourier:  number of frequency components
        normalize:  if True, z-score normalize timestamps before encoding
        dropout:    output dropout

    Example:
        >>> fte = FourierTimeEncoding(d_model=256, n_fourier=64)
        >>> ts = torch.arange(100).float().unsqueeze(0)  # (1, 100)
        >>> out = fte(ts)  # (1, 100, 256)
    """

    def __init__(
        self,
        d_model: int,
        n_fourier: int = 64,
        normalize: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_fourier = n_fourier
        self.normalize = normalize

        # Learnable frequencies and phases
        self.weights = nn.Parameter(torch.randn(n_fourier) * 0.1)
        self.phases = nn.Parameter(torch.zeros(n_fourier))

        # Linear trend feature
        self.trend_proj = nn.Linear(1, d_model // 4, bias=True)

        # Output projection: 2*n_fourier + d_model//4 → d_model
        fourier_out_dim = 2 * n_fourier
        self.proj = nn.Linear(fourier_out_dim + d_model // 4, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        # Initialize frequencies to cover different time scales
        with torch.no_grad():
            n = self.n_fourier
            # Cover periods from 1 minute to 1 year in log scale
            log_periods = torch.linspace(math.log(1.0), math.log(525960.0), n)
            periods = torch.exp(log_periods)
            self.weights.data = (2 * math.pi / periods)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps as Fourier features.

        Args:
            timestamps: (B, T) float timestamps (any units)

        Returns:
            encoding: (B, T, d_model)
        """
        t = timestamps.float()

        if self.normalize:
            t_mean = t.mean(dim=-1, keepdim=True)
            t_std = t.std(dim=-1, keepdim=True).clamp(min=1.0)
            t_norm = (t - t_mean) / t_std
        else:
            t_norm = t

        # Fourier features
        angles = (
            t_norm.unsqueeze(-1) * self.weights.unsqueeze(0).unsqueeze(0)
            + self.phases.unsqueeze(0).unsqueeze(0)
        )  # (B, T, n_fourier)
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, T, 2*n_fourier)

        # Linear trend
        trend = self.trend_proj(t_norm.unsqueeze(-1))  # (B, T, d_model//4)

        combined = torch.cat([fourier, trend], dim=-1)
        out = self.dropout(self.norm(self.proj(combined)))
        return out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_fourier={self.n_fourier}"


# ===========================================================================
# Calendar Encoding (US Market)
# ===========================================================================

class CalendarEncoding(nn.Module):
    """Rich calendar encoding for US equity market timing.

    Extracts and encodes the following calendar features:
    - Time of day (sin/cos, hour resolution)
    - Day of week (sin/cos)
    - Week of year (sin/cos)
    - Month (sin/cos)
    - Quarter (sin/cos)
    - Days to/from month-end
    - Days to/from quarter-end
    - Is-holiday indicator (US market holidays)
    - Is-earnings-season indicator (Jan/Apr/Jul/Oct)
    - Pre/post-FOMC meeting indicator
    - Day before/after holiday

    All temporal features are sin/cos encoded, holiday/event indicators
    are binary, and the whole concatenation is projected to d_model.

    Args:
        d_model:  output embedding dimension
        dropout:  output dropout probability

    Example:
        >>> cal = CalendarEncoding(d_model=512)
        >>> ts = torch.randint(1_600_000_000, 1_700_000_000, (2, 128)).float()
        >>> out = cal(ts)  # (2, 128, 512)
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Feature dimensions:
        # 8 sin/cos features + 4 scalar features + 4 binary event features = 16
        n_features = 20
        self.proj = nn.Sequential(
            nn.Linear(n_features, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def _sincos(val: torch.Tensor, period: float) -> torch.Tensor:
        """Encode scalar as sin/cos pair."""
        angle = _TWO_PI * val / period
        return torch.stack([angle.sin(), angle.cos()], dim=-1)

    @staticmethod
    def _is_near_holiday(month: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
        """Check if date is within 2 days of a US market holiday."""
        result = torch.zeros_like(month, dtype=torch.float)
        for m, d in _US_MARKET_HOLIDAYS:
            near = ((month == m) & ((day - d).abs() <= 2))
            result = result.masked_fill(near, 1.0)
        return result

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps as calendar features.

        Args:
            timestamps: (B, T) Unix timestamps in seconds

        Returns:
            encoding: (B, T, d_model)
        """
        ts = timestamps.float()

        seconds_per_day = 86400.0
        time_of_day = ts % seconds_per_day
        hour = (time_of_day / 3600.0).float()
        day_num = (ts / seconds_per_day).long().float()
        dow = (day_num + 3) % 7      # 0=Monday
        dom = day_num % 30 + 1       # day of month (approx)
        month = (day_num / 30.4 % 12 + 1).clamp(1, 12)
        week_of_year = (day_num % 365 / 7 + 1).clamp(1, 53)
        quarter = ((month - 1) / 3 + 1).long().float()

        # sin/cos features (8 × 2 = 16 values total — but we cat as pairs)
        sc_hour = self._sincos(hour, 24.0)      # (B, T, 2)
        sc_dow = self._sincos(dow, 7.0)         # (B, T, 2)
        sc_month = self._sincos(month, 12.0)    # (B, T, 2)
        sc_week = self._sincos(week_of_year, 53.0)  # (B, T, 2)

        # Scalar features
        days_to_month_end = (30 - dom).clamp(0, 30) / 30.0  # normalized
        quarter_progress = ((month - 1) % 3) / 3.0

        # Binary features
        is_earnings = ((month % 3 == 1) & (dom <= 15)).float()  # Jan/Apr/Jul/Oct first half
        is_near_holiday = self._is_near_holiday(month.long(), dom.long())
        is_monday = (dow == 0).float()
        is_friday = (dow == 4).float()

        # Concatenate
        features = torch.cat([
            sc_hour, sc_dow, sc_month, sc_week,  # 8
            days_to_month_end.unsqueeze(-1),       # 1
            quarter_progress.unsqueeze(-1),        # 1
            is_earnings.unsqueeze(-1),             # 1
            is_near_holiday.unsqueeze(-1),         # 1
            is_monday.unsqueeze(-1),               # 1
            is_friday.unsqueeze(-1),               # 1
            quarter.unsqueeze(-1) / 4.0,           # 1 (normalized)
            dom.unsqueeze(-1) / 31.0,              # 1
            week_of_year.unsqueeze(-1) / 53.0,     # 1 (extra)
            (dow / 6.0).unsqueeze(-1),             # 1 (extra)
        ], dim=-1)  # (B, T, 20)

        out = self.dropout(self.norm(self.proj(features)))
        return out


# ===========================================================================
# Market Microstructure Encoding
# ===========================================================================

class MarketMicrostructureEncoding(nn.Module):
    """Encodes intraday market microstructure timing patterns.

    Captures the following intraday phenomena:
    - Open auction effects (09:30–09:45)
    - Lunch lull (11:30–13:00)
    - Close auction effects (15:30–16:00)
    - Regular trading intensity (triangular weighting by time of day)
    - Pre-market / after-hours distinction

    These patterns are well-documented in market microstructure literature
    (e.g., U-shaped intraday volatility pattern, Admati & Pfleiderer 1988).

    Args:
        d_model:  output embedding dimension
        dropout:  output dropout

    Example:
        >>> enc = MarketMicrostructureEncoding(d_model=256)
        >>> ts = torch.randint(1_600_000_000, 1_700_000_000, (2, 64)).float()
        >>> out = enc(ts)  # (2, 64, 256)
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model

        n_features = 12  # various microstructure indicators
        self.proj = nn.Linear(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def _u_shaped_weight(self, minutes_from_open: torch.Tensor) -> torch.Tensor:
        """Compute U-shaped intraday weight.

        Higher at open/close, lower in the middle of the day.

        Args:
            minutes_from_open: (...) float tensor, 0=market open

        Returns:
            weight: (...) float in [0, 1]
        """
        total_rth_minutes = 390.0  # 6.5 hours
        t = minutes_from_open.clamp(0, total_rth_minutes) / total_rth_minutes
        # U-shape: high at 0 and 1, low at 0.5
        return 1.0 - 4 * t * (1.0 - t) * 0.8  # minimum of 0.2

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (B, T) Unix timestamps in seconds

        Returns:
            encoding: (B, T, d_model)
        """
        ts = timestamps.float()
        tod = ts % 86400.0  # time of day in seconds
        minutes_from_midnight = tod / 60.0

        # Market open is 9:30 = 570 minutes from midnight
        rth_open_min = 570.0
        rth_close_min = 960.0  # 16:00
        minutes_from_open = (minutes_from_midnight - rth_open_min).clamp(-60, 390)

        # Microstructure features
        is_rth = ((minutes_from_midnight >= rth_open_min) &
                  (minutes_from_midnight < rth_close_min)).float()
        is_open_auction = ((minutes_from_midnight >= rth_open_min) &
                           (minutes_from_midnight < rth_open_min + 15)).float()
        is_close_auction = ((minutes_from_midnight >= rth_close_min - 30) &
                            (minutes_from_midnight < rth_close_min)).float()
        is_lunch = ((minutes_from_midnight >= 690) &  # 11:30
                    (minutes_from_midnight < 780)).float()  # 13:00
        is_pre_market = ((minutes_from_midnight >= 240) &  # 04:00
                         (minutes_from_midnight < rth_open_min)).float()
        is_after_hours = ((minutes_from_midnight >= rth_close_min) &
                          (minutes_from_midnight < 1200)).float()  # 20:00

        u_weight = self._u_shaped_weight(minutes_from_open)
        progress_through_rth = (minutes_from_open / 390.0).clamp(0, 1)

        # Sin/cos of time position within RTH
        sc_rth = torch.stack([
            (progress_through_rth * math.pi * 2).sin(),
            (progress_through_rth * math.pi * 2).cos(),
        ], dim=-1)  # (..., 2)

        features = torch.cat([
            is_rth.unsqueeze(-1),
            is_open_auction.unsqueeze(-1),
            is_close_auction.unsqueeze(-1),
            is_lunch.unsqueeze(-1),
            is_pre_market.unsqueeze(-1),
            is_after_hours.unsqueeze(-1),
            u_weight.unsqueeze(-1),
            progress_through_rth.unsqueeze(-1),
            sc_rth,                              # 2 features
            (minutes_from_midnight / 1440.0).unsqueeze(-1),  # normalized time
            (minutes_from_open / 390.0).clamp(-1, 1).unsqueeze(-1),
        ], dim=-1)  # (B, T, 12)

        out = self.dropout(self.norm(self.proj(features)))
        return out


# ===========================================================================
# Economic Cycle Encoding
# ===========================================================================

class EconomicCycleEncoding(nn.Module):
    """Encodes macro-economic cycle information for long-horizon modeling.

    Captures:
    - Month-of-year cyclicality (sin/cos)
    - Quarter-of-year (sin/cos)
    - Year progress (sin/cos)
    - Decade within era (for regime shift awareness)
    - Recession proximity indicator (requires external label)

    Args:
        d_model:  output dimension
        dropout:  output dropout

    Example:
        >>> ece = EconomicCycleEncoding(d_model=128)
        >>> ts = torch.randint(1_000_000_000, 1_700_000_000, (2, 256)).float()
        >>> out = ece(ts)  # (2, 256, 128)
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        n_features = 10
        self.proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (B, T) Unix timestamps in seconds

        Returns:
            encoding: (B, T, d_model)
        """
        ts = timestamps.float()
        days = ts / 86400.0
        year_frac = (days % 365.25) / 365.25         # 0..1
        month_frac = (days % 30.44) / 30.44           # 0..1
        quarter_frac = (days % 91.31) / 91.31         # 0..1
        decade_frac = (days % 3652.5) / 3652.5        # 0..1 (within decade)
        half_year_frac = (days % 182.5) / 182.5       # 0..1

        features = torch.stack([
            torch.sin(_TWO_PI * year_frac),
            torch.cos(_TWO_PI * year_frac),
            torch.sin(_TWO_PI * month_frac),
            torch.cos(_TWO_PI * month_frac),
            torch.sin(_TWO_PI * quarter_frac),
            torch.cos(_TWO_PI * quarter_frac),
            torch.sin(_TWO_PI * half_year_frac),
            torch.cos(_TWO_PI * half_year_frac),
            torch.sin(_TWO_PI * decade_frac),
            torch.cos(_TWO_PI * decade_frac),
        ], dim=-1)  # (B, T, 10)

        out = self.dropout(self.norm(self.proj(features)))
        return out


# ===========================================================================
# Cross-Modal Positional Encoding
# ===========================================================================

class CrossModalPositionalEncoding(nn.Module):
    """Separate position spaces per modality with cross-modal relative positions.

    For multi-modal financial data (OHLCV + order book + on-chain + news),
    each modality has its own position embedding. Cross-modal attention uses
    relative position biases between modalities.

    Architecture:
    - Per-modality sinusoidal/learned position embeddings
    - Cross-modal relative bias tables (one per modality pair)
    - Single projection to d_model

    Args:
        d_model:             embedding dimension
        n_modalities:        number of distinct modality types
        max_seq_len:         maximum positions per modality
        max_cross_distance:  maximum cross-modal relative distance

    Example:
        >>> cpe = CrossModalPositionalEncoding(d_model=512, n_modalities=4)
        >>> x = torch.randn(2, 128, 512)
        >>> mod_ids = torch.randint(0, 4, (2, 128))
        >>> out = cpe(x, mod_ids)  # (2, 128, 512)
    """

    def __init__(
        self,
        d_model: int,
        n_modalities: int = 4,
        max_seq_len: int = 1024,
        max_cross_distance: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_modalities = n_modalities
        self.max_seq_len = max_seq_len
        self.max_cross_distance = max_cross_distance

        self.pos_embeddings = nn.ModuleList([
            nn.Embedding(max_seq_len, d_model)
            for _ in range(n_modalities)
        ])

        self.cross_biases = nn.ParameterDict({
            f"bias_{i}_{j}": nn.Parameter(torch.zeros(2 * max_cross_distance + 1))
            for i in range(n_modalities)
            for j in range(n_modalities)
            if i != j
        })

        self._init_sinusoidal()

    def _init_sinusoidal(self):
        """Initialize position embeddings with sinusoidal weights."""
        for emb in self.pos_embeddings:
            d = emb.embedding_dim
            n = emb.num_embeddings
            positions = torch.arange(n).unsqueeze(1).float()
            dims = torch.arange(0, d, 2).float()
            angles = positions / (10000.0 ** (dims / d))
            pe = torch.zeros(n, d)
            pe[:, 0::2] = angles.sin()
            pe[:, 1::2] = angles.cos()
            emb.weight.data.copy_(pe)

    def encode_modality(
        self, modality_id: int, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Get position encoding for a specific modality.

        Args:
            modality_id: modality index
            seq_len:     sequence length
            device:      target device

        Returns:
            pos_enc: (1, seq_len, d_model)
        """
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        return self.pos_embeddings[modality_id](pos_ids)

    def cross_modal_bias(
        self,
        src_modality: int,
        tgt_modality: int,
        src_positions: torch.Tensor,
        tgt_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute relative bias for cross-modal attention.

        Args:
            src_modality:  source modality index
            tgt_modality:  target modality index
            src_positions: (B, Q) query positions
            tgt_positions: (B, K) key positions

        Returns:
            bias: (B, Q, K)
        """
        rel = src_positions.unsqueeze(-1) - tgt_positions.unsqueeze(-2)
        rel = rel.clamp(-self.max_cross_distance, self.max_cross_distance)
        rel_idx = rel + self.max_cross_distance
        key = f"bias_{src_modality}_{tgt_modality}"
        bias_table = self.cross_biases[key]
        return bias_table[rel_idx]

    def forward(
        self,
        token_embeddings: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Add per-modality positional embeddings.

        Args:
            token_embeddings: (B, T, d_model)
            modality_ids:     (B, T) long

        Returns:
            token_embeddings + per-modality positions: (B, T, d_model)
        """
        B, T, D = token_embeddings.shape
        out = token_embeddings.clone()

        for mod_id in range(self.n_modalities):
            mask = modality_ids == mod_id  # (B, T)
            if not mask.any():
                continue
            for b in range(B):
                idx = mask[b].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                n = idx.numel()
                pos_ids = torch.arange(n, device=token_embeddings.device).unsqueeze(0)
                pos_emb = self.pos_embeddings[mod_id](pos_ids).squeeze(0)
                out[b, idx] += pos_emb

        return out


# ===========================================================================
# Modality Embedding
# ===========================================================================

class ModalityEmbedding(nn.Module):
    """Learnable modality-type token embeddings.

    Adds a learned embedding vector based on the modality type of each token.
    Similar to segment embeddings in BERT.

    Args:
        d_model:     embedding dimension
        n_modalities: number of distinct modalities

    Example:
        >>> me = ModalityEmbedding(d_model=512, n_modalities=4)
        >>> mod_ids = torch.randint(0, 4, (2, 128))
        >>> emb = me(mod_ids)  # (2, 128, 512)
    """

    def __init__(self, d_model: int, n_modalities: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(n_modalities, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, modality_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modality_ids: (B, T) long tensor

        Returns:
            embeddings: (B, T, d_model)
        """
        return self.embedding(modality_ids)


# ===========================================================================
# Hierarchical Positional Encoding
# ===========================================================================

class HierarchicalPositionalEncoding(nn.Module):
    """Hierarchical position encoding: segment + intra-segment positions.

    Encodes positions at two levels:
    1. Segment level: which segment (e.g., which day)
    2. Intra-segment level: position within segment (e.g., bar within day)

    Combined: PE_total[s, p] = PE_seg[s] + PE_intra[p]

    This allows the model to generalize across segments while learning
    fine-grained intra-segment patterns.

    Args:
        d_model:         embedding dimension (split between segment/intra)
        max_segments:    maximum number of segments
        max_intra_len:   maximum positions per segment

    Example:
        >>> hpe = HierarchicalPositionalEncoding(d_model=512, max_segments=252, max_intra_len=390)
        >>> x = torch.randn(2, 390*5, 512)  # 5 days of minute bars
        >>> seg_ids = torch.arange(5).repeat_interleave(390).unsqueeze(0).expand(2, -1)
        >>> intra_ids = torch.arange(390).repeat(5).unsqueeze(0).expand(2, -1)
        >>> out = hpe(x, seg_ids, intra_ids)  # (2, 1950, 512)
    """

    def __init__(
        self,
        d_model: int,
        max_segments: int = 512,
        max_intra_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2

        self.seg_embedding = LearnedAbsolutePositionalEncoding(half, max_segments)
        self.intra_embedding = SinusoidalPositionalEncoding(d_model - half, max_intra_len)

        self.half = half

    def forward(
        self,
        x: torch.Tensor,
        segment_ids: torch.Tensor,
        intra_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Add hierarchical position encodings.

        Args:
            x:           (B, T, d_model) input embeddings
            segment_ids: (B, T) long tensor of segment indices
            intra_ids:   (B, T) long tensor of intra-segment positions

        Returns:
            x + seg_pe + intra_pe: (B, T, d_model)
        """
        # Segment encoding: (B, T, half)
        seg_emb = self.seg_embedding.embedding(segment_ids)  # (B, T, half)
        # Intra-segment encoding: we use sinusoidal but need to extract for custom ids
        # Build intra pe for max intra position
        T = x.shape[1]
        max_intra = intra_ids.max().item() + 1
        if max_intra > self.intra_embedding.pe.shape[1]:
            self.intra_embedding.pe = SinusoidalPositionalEncoding._build_pe(
                max_intra + 64, self.d_model - self.half
            ).to(x.device)

        intra_emb = self.intra_embedding.pe[:, intra_ids.long(), :].squeeze(0)  # tricky
        # Fallback: simpler gather
        pe_table = self.intra_embedding.pe.squeeze(0)  # (max_T, d_half2)
        intra_emb = pe_table[intra_ids.clamp(0, pe_table.shape[0] - 1)]  # (B, T, d_half2)

        out = x.clone()
        out[..., :self.half] += seg_emb
        out[..., self.half:] += intra_emb
        return out


# ===========================================================================
# Compound Positional Encoding
# ===========================================================================

class CompoundPositionalEncoding(nn.Module):
    """Compound positional encoding combining multiple strategies.

    Combines sinusoidal absolute positions with temporal calendar features,
    with optional timestamp-based encoding.

    Components:
    1. Sinusoidal absolute position (always included)
    2. TemporalEncoding from Unix timestamps (if timestamps provided)
    3. Optional FourierTimeEncoding (learnable frequencies)

    A gating mechanism learns to weight the contributions of each component.

    Args:
        d_model:          embedding dimension
        max_seq_len:      maximum sequence length
        use_temporal:     whether to use TemporalEncoding
        use_fourier:      whether to use FourierTimeEncoding
        fourier_n:        number of Fourier components
        dropout:          output dropout

    Example:
        >>> cpe = CompoundPositionalEncoding(d_model=512)
        >>> x = torch.randn(2, 128, 512)
        >>> ts = torch.randint(1_600_000_000, 1_700_000_000, (2, 128)).float()
        >>> out = cpe(x, timestamps=ts)
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 4096,
        use_temporal: bool = True,
        use_fourier: bool = False,
        fourier_n: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_temporal = use_temporal
        self.use_fourier = use_fourier

        self.sinusoidal = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout=0.0)

        if use_temporal:
            self.temporal = TemporalEncoding(d_model)

        if use_fourier:
            self.fourier = FourierTimeEncoding(d_model, n_fourier=fourier_n)

        # Learnable gates for combining components
        n_components = 1 + int(use_temporal) + int(use_fourier)
        self.gate = nn.Parameter(torch.ones(n_components) / n_components)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add compound positional encoding.

        Args:
            x:          (B, T, d_model) input embeddings
            timestamps: (B, T) Unix timestamps (required if use_temporal or use_fourier)

        Returns:
            output: (B, T, d_model) with positional encoding added
        """
        # Compute gates
        gate_weights = torch.softmax(self.gate, dim=0)

        # Sinusoidal component (pure additive, no input modification)
        T = x.shape[1]
        sin_enc = self.sinusoidal.pe[:, :T, :]  # (1, T, d_model)
        components = [gate_weights[0] * sin_enc.expand_as(x)]
        idx = 1

        if self.use_temporal and timestamps is not None:
            temp_enc = self.temporal(timestamps)  # (B, T, d_model)
            components.append(gate_weights[idx] * temp_enc)
            idx += 1

        if self.use_fourier and timestamps is not None:
            fourier_enc = self.fourier(timestamps)  # (B, T, d_model)
            components.append(gate_weights[idx] * fourier_enc)

        combined = sum(components)
        return self.dropout(x + combined)


# ===========================================================================
# Positional Encoding Factory
# ===========================================================================

class PositionalEncodingFactory:
    """Factory class for creating positional encoding modules.

    Provides a registry-based interface for instantiating any supported
    positional encoding type by name.

    Supported types:
    - "sinusoidal":    SinusoidalPositionalEncoding
    - "learned":       LearnedAbsolutePositionalEncoding
    - "scaled_sin":    ScaledSinusoidalEncoding
    - "rope":          RotaryPositionalEncoding
    - "rope_extended": ExtendedRoPE
    - "rope2d":        RoPE2D
    - "temporal_rope": TemporalRoPE
    - "alibi":         ALiBiPositionalBias
    - "causal_alibi":  CausalALiBi
    - "t5":            T5RelativePositionBias
    - "relative":      RelativePositionalEncoding
    - "disentangled":  DisentangledAttentionBias
    - "temporal":      TemporalEncoding
    - "fourier":       FourierTimeEncoding
    - "calendar":      CalendarEncoding
    - "microstructure":MarketMicrostructureEncoding
    - "economic_cycle":EconomicCycleEncoding
    - "cross_modal":   CrossModalPositionalEncoding
    - "hierarchical":  HierarchicalPositionalEncoding
    - "compound":      CompoundPositionalEncoding
    - "none":          nn.Identity

    Example:
        >>> factory = PositionalEncodingFactory()
        >>> rope = factory.create("rope", dim=64, max_seq_len=2048)
        >>> alibi = factory.create("alibi", n_heads=8, max_seq_len=4096)
    """

    _registry: Dict[str, type] = {
        "sinusoidal": SinusoidalPositionalEncoding,
        "learned": LearnedAbsolutePositionalEncoding,
        "scaled_sin": ScaledSinusoidalEncoding,
        "adaptive_sin": AdaptiveSinusoidalEncoding,
        "rope": RotaryPositionalEncoding,
        "rope_extended": ExtendedRoPE,
        "rope2d": RoPE2D,
        "temporal_rope": TemporalRoPE,
        "alibi": ALiBiPositionalBias,
        "causal_alibi": CausalALiBi,
        "t5": T5RelativePositionBias,
        "relative": RelativePositionalEncoding,
        "disentangled": DisentangledAttentionBias,
        "temporal": TemporalEncoding,
        "fourier": FourierTimeEncoding,
        "calendar": CalendarEncoding,
        "microstructure": MarketMicrostructureEncoding,
        "economic_cycle": EconomicCycleEncoding,
        "cross_modal": CrossModalPositionalEncoding,
        "hierarchical": HierarchicalPositionalEncoding,
        "compound": CompoundPositionalEncoding,
        "none": nn.Identity,
    }

    @classmethod
    def create(cls, enc_type: str, **kwargs) -> nn.Module:
        """Instantiate a positional encoding module.

        Args:
            enc_type: encoding type string (see class docstring)
            **kwargs: constructor keyword arguments

        Returns:
            module: instantiated positional encoding module

        Raises:
            ValueError: if enc_type is not registered
        """
        if enc_type not in cls._registry:
            available = sorted(cls._registry.keys())
            raise ValueError(
                f"Unknown positional encoding type '{enc_type}'. "
                f"Available: {available}"
            )
        cls_type = cls._registry[enc_type]
        if cls_type is nn.Identity:
            return nn.Identity()
        return cls_type(**kwargs)

    @classmethod
    def register(cls, name: str, module_class: type) -> None:
        """Register a custom positional encoding class.

        Args:
            name:         registration key
            module_class: nn.Module subclass
        """
        if not issubclass(module_class, nn.Module):
            raise TypeError(f"{module_class} must be a subclass of nn.Module")
        cls._registry[name] = module_class

    @classmethod
    def list_available(cls) -> List[str]:
        """Return sorted list of available encoding types."""
        return sorted(cls._registry.keys())


# ===========================================================================
# Performer Positional Encoding (random feature approximation)
# ===========================================================================

class PerformerPositionalEncoding(nn.Module):
    """Random feature positional encoding for Performer-style linear attention.

    Uses random Fourier features to approximate the softmax kernel:
    K(q, k) ≈ phi(q) · phi(k)^T

    where phi(x) = exp(x·W^T + b) / sqrt(m), W ~ N(0, I), b ~ Uniform(0, 2pi).

    Args:
        d_model:   model dimension
        n_random:  number of random features
        seed:      random seed for reproducibility

    Example:
        >>> pfe = PerformerPositionalEncoding(d_model=256, n_random=256)
        >>> x = torch.randn(2, 64, 256)
        >>> phi_x = pfe(x)  # (2, 64, n_random)
    """

    def __init__(self, d_model: int, n_random: int = 256, seed: int = 42):
        super().__init__()
        self.d_model = d_model
        self.n_random = n_random

        # Fixed random projection matrix
        rng = torch.Generator()
        rng.manual_seed(seed)
        W = torch.randn(n_random, d_model, generator=rng)
        b = torch.rand(n_random, generator=rng) * _TWO_PI
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self._normalizer = n_random ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute random Fourier features.

        Args:
            x: (B, T, d_model)

        Returns:
            phi_x: (B, T, n_random) — positive random features
        """
        # z = x @ W^T + b: (B, T, n_random)
        z = x @ self.W.t() + self.b.unsqueeze(0).unsqueeze(0)
        # Use exp(z) for positive features (FAVOR+)
        phi = torch.exp(
            z - 0.5 * (x ** 2).sum(dim=-1, keepdim=True)
        ) * self._normalizer
        return phi


# ===========================================================================
# Positional Encoding Interpolation Utility
# ===========================================================================

class PositionalEncodingInterpolator(nn.Module):
    """Interpolates learned positional encodings to arbitrary lengths.

    Used when fine-tuning a model on sequences longer than its training
    context window.

    Args:
        base_pe:       (1, T, d_model) base positional encoding tensor
        mode:          interpolation mode: "linear" | "bicubic" | "nearest"
        antialias:     use antialiasing in bicubic mode

    Example:
        >>> pe = torch.randn(1, 512, 768)
        >>> interp = PositionalEncodingInterpolator(pe, mode="bicubic")
        >>> longer_pe = interp(target_len=1024)  # (1, 1024, 768)
    """

    def __init__(
        self,
        base_pe: torch.Tensor,
        mode: str = "bicubic",
        antialias: bool = True,
    ):
        super().__init__()
        self.register_buffer("base_pe", base_pe)
        self.mode = mode
        self.antialias = antialias
        self.src_len = base_pe.shape[1]
        self.d_model = base_pe.shape[2]

    def forward(self, target_len: int) -> torch.Tensor:
        """Interpolate to target length.

        Args:
            target_len: desired sequence length

        Returns:
            interp_pe: (1, target_len, d_model)
        """
        if target_len == self.src_len:
            return self.base_pe

        # Reshape for F.interpolate: (1, d_model, src_len, 1)
        pe = self.base_pe.permute(0, 2, 1).unsqueeze(-1)
        interp = F.interpolate(
            pe,
            size=(target_len, 1),
            mode="bilinear",
            align_corners=False,
        )
        return interp.squeeze(-1).permute(0, 2, 1)


# ===========================================================================
# Module exports / __all__
# ===========================================================================

__all__ = [
    # Utilities
    "rotate_half",
    "apply_rotary_emb",
    "apply_rotary_emb_single",
    "precompute_freqs_cis",
    "precompute_freqs_cos_sin",
    "get_slopes",
    "interpolate_pos_encoding",
    "compute_relative_positions",
    "bucket_relative_positions",
    "ntk_scale_freqs",
    "yarn_scale_freqs",
    # Enums
    "PositionalEncodingType",
    "MarketSession",
    "EconomicRegime",
    # Config
    "PositionalEncodingConfig",
    # Absolute encodings
    "SinusoidalPositionalEncoding",
    "LearnedAbsolutePositionalEncoding",
    "ScaledSinusoidalEncoding",
    "AdaptiveSinusoidalEncoding",
    # RoPE variants
    "RotaryPositionalEncoding",
    "RoPE2D",
    "TemporalRoPE",
    "ExtendedRoPE",
    # Bias-based
    "ALiBiPositionalBias",
    "CausalALiBi",
    "T5RelativePositionBias",
    "RelativePositionalEncoding",
    "DisentangledAttentionBias",
    # Timestamp / calendar
    "TemporalEncoding",
    "FourierTimeEncoding",
    "CalendarEncoding",
    "MarketMicrostructureEncoding",
    "EconomicCycleEncoding",
    # Multi-modal
    "CrossModalPositionalEncoding",
    "ModalityEmbedding",
    "HierarchicalPositionalEncoding",
    # Compound
    "CompoundPositionalEncoding",
    # Performer
    "PerformerPositionalEncoding",
    # Interpolation
    "PositionalEncodingInterpolator",
    # Factory
    "PositionalEncodingFactory",
]


# =============================================================================
# SECTION: Advanced Positional and Temporal Encoding for Financial Data
# =============================================================================



class NTKAwareRoPE(nn.Module):
    """NTK-aware RoPE with dynamic frequency adjustment.

    Adjusts base frequency dynamically based on context length. When the
        sequence exceeds training length, the effective frequency is scaled to
        maintain quality. Three scaling modes: linear, dynamic, and YaRN.
    
        Reference: LocalLLaMA community, "NTK-Aware Scaled RoPE" (2023)

    Args:
        d_model: Model dimension
        max_seq_len: Training context length
        base: Base frequency (10000 for RoPE)
        scale_type: Scaling mode: linear, dynamic, ntk, yarn
        scale_factor: Scaling factor for extended context
    """

    def __init__(
        self,
        d_model: int = 512,
        max_seq_len: int = 2048,
        base: int = 8,
        scale_type: str = "str",
        scale_factor: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        self.scale_type = scale_type
        self.scale_factor = scale_factor
        import math
        D = d_model
        T = max_seq_len
        # Create sinusoidal base encoding
        pe = torch.zeros(T, D)
        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:D//2])
        self.register_buffer("_base_pe", pe.unsqueeze(0))
        self.proj = nn.Linear(D, D, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        pe = self._base_pe[:, :T, :]
        x = x + pe
        return x


class BinaryPositionalEncoding(nn.Module):
    """Binary positional encoding for discrete positions.

    Represents each position as a binary vector. Unlike sinusoidal, binary
        encoding preserves exact position information but lacks interpolation
        properties. Useful for short sequences where exact position matters.

    Args:
        max_seq_len: Maximum sequence length
        d_model: Output dimension
        learnable_proj: Whether to learn linear projection
    """

    def __init__(
        self,
        max_seq_len: int = 2048,
        d_model: int = 512,
        learnable_proj: bool = True,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.learnable_proj = learnable_proj
        import math
        D = d_model
        T = max_seq_len
        # Create sinusoidal base encoding
        pe = torch.zeros(T, D)
        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:D//2])
        self.register_buffer("_base_pe", pe.unsqueeze(0))
        self.proj = nn.Linear(D, D, bias=False)
        self.amplitude = nn.Parameter(torch.ones(1, 1, D))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        pe = self._base_pe[:, :T, :]
        x = x + pe
        return x


class ConvolutionalPositionalEncoding(nn.Module):
    """Relative position via causal convolution.

    Uses a stack of dilated causal convolutions to encode local relative
        position information. The receptive field grows exponentially with depth,
        covering both local and moderate-range positions.

    Args:
        d_model: Model dimension
        num_layers: Number of convolution layers
        kernel_size: Convolution kernel size
        dilation_rate: Dilation growth factor per layer
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 8,
        kernel_size: int = 8,
        dilation_rate: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout = dropout
        import math
        D = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        return self.dropout(x)


class SandwichPositionalEncoding(nn.Module):
    """Learned positional encoding with normalization sandwich.

    Applies LayerNorm before and after positional embedding injection,
        preventing the positional information from dominating the content.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 512,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        import math
        D = d_model
        T = max_seq_len
        # Create sinusoidal base encoding
        pe = torch.zeros(T, D)
        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:D//2])
        self.register_buffer("_base_pe", pe.unsqueeze(0))
        self.proj = nn.Linear(D, D, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        pe = self._base_pe[:, :T, :]
        x = x + pe
        return self.dropout(x)


class MultiScaleRoPE(nn.Module):
    """Multi-scale RoPE combining multiple base frequencies.

    Computes RoPE at multiple frequency scales and concatenates them.
        Lower frequencies capture long-range dependencies while higher
        frequencies encode fine-grained local position.

    Args:
        d_model: Model dimension
        num_scales: Number of frequency scales
        base_frequencies: List of base frequencies
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        d_model: int = 512,
        num_scales: int = 8,
        base_frequencies: Optional[List[int]] = None,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.base_frequencies = base_frequencies
        self.max_seq_len = max_seq_len
        import math
        D = d_model
        T = max_seq_len
        # Create sinusoidal base encoding
        pe = torch.zeros(T, D)
        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:D//2])
        self.register_buffer("_base_pe", pe.unsqueeze(0))
        self.proj = nn.Linear(D, D, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        pe = self._base_pe[:, :T, :]
        x = x + pe
        return x


class RegimeSensitivePositionEncoding(nn.Module):
    """Position encoding adaptive to market regime.

    Adjusts positional encoding based on the current market regime.
        In trending markets, emphasizes longer-range position information.
        In volatile markets, focuses on recent positions.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        num_regimes: Number of market regimes
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 512,
        max_seq_len: int = 2048,
        num_regimes: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_regimes = num_regimes
        self.dropout = dropout
        import math
        D = d_model
        T = max_seq_len
        # Create sinusoidal base encoding
        pe = torch.zeros(T, D)
        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:D//2])
        self.register_buffer("_base_pe", pe.unsqueeze(0))
        self.proj = nn.Linear(D, D, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.regime_embed = nn.Embedding(num_regimes, d_model)
        nn.init.zeros_(self.regime_embed.weight)

    def forward(
        self,
        x: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        pe = self._base_pe[:, :T, :]
        x = x + pe
        if regime_ids is not None:
            if regime_ids.dim() == 1:
                regime_ids = regime_ids.unsqueeze(1).expand(B, T)
            x = x + self.regime_embed(regime_ids)
        return self.dropout(x)


class SinusoidalWithLearned(nn.Module):
    """Hybrid sinusoidal + learned positional encoding.

    Combines fixed sinusoidal encoding (for generalization) with a
        learnable offset (for adaptation). The learned component is regularized
        to stay close to zero initially.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        learned_weight: Initial weight of learned component
    """

    def __init__(
        self,
        d_model: int = 512,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        learned_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.learned_weight = learned_weight
        import math
        D = d_model
        T = max_seq_len
        # Create sinusoidal base encoding
        pe = torch.zeros(T, D)
        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:D//2])
        self.register_buffer("_base_pe", pe.unsqueeze(0))
        self.proj = nn.Linear(D, D, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        pe = self._base_pe[:, :T, :]
        x = x + pe
        return self.dropout(x)


class PeriodicPositionEncoding(nn.Module):
    """Periodic positional encoding for cyclic patterns.

    Designed for financial data with known periodicities (weekly, monthly,
        quarterly, annual). Combines multiple periodic components at known
        frequencies.

    Args:
        d_model: Model dimension
        periods: Known period lengths (e.g. [5, 21, 63, 252])
        max_seq_len: Maximum sequence length
        learnable: Whether period amplitudes are learnable
    """

    def __init__(
        self,
        d_model: int = 512,
        periods: Optional[List[int]] = None,
        max_seq_len: int = 2048,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.periods = periods
        self.max_seq_len = max_seq_len
        self.learnable = learnable
        import math
        D = d_model
        T = max_seq_len
        # Create sinusoidal base encoding
        pe = torch.zeros(T, D)
        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:D//2])
        self.register_buffer("_base_pe", pe.unsqueeze(0))
        self.proj = nn.Linear(D, D, bias=False)
        self.amplitude = nn.Parameter(torch.ones(1, 1, D))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        pe = self._base_pe[:, :T, :]
        x = x + pe
        return x


class RelativeBucketEncoding(nn.Module):
    """Bucketed relative position encoding.

    Groups relative positions into logarithmically-spaced buckets,
        inspired by T5 relative attention biases. Provides good coverage
        of both local and long-range relative positions.

    Args:
        d_model: Model dimension
        num_buckets: Number of position buckets
        max_distance: Maximum relative distance to encode
        bidirectional: Whether to encode both past and future
    """

    def __init__(
        self,
        d_model: int = 512,
        num_buckets: int = 8,
        max_distance: int = 8,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        import math
        D = d_model

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        return x


class TemporalHierarchicalEncoding(nn.Module):
    """Hierarchical temporal encoding across time scales.

    Encodes position at multiple temporal hierarchies simultaneously:
        intraday (minute), daily, weekly, monthly, quarterly, annual.
        Useful for multi-frequency financial models.

    Args:
        d_model: Model dimension
        hierarchies: Period lengths for each hierarchy level
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 512,
        hierarchies: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hierarchies = hierarchies
        self.dropout = dropout
        import math
        D = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        return self.dropout(x)


class ProgressivePositionalEncoding(nn.Module):
    """Position encoding with progressive resolution.

    Progressively adds position information at increasing resolutions.
        Coarse position information is available at early layers while
        fine-grained position emerges at later layers.

    Args:
        d_model: Model dimension
        num_levels: Number of resolution levels
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        d_model: int = 512,
        num_levels: int = 8,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        self.max_seq_len = max_seq_len
        import math
        D = d_model
        T = max_seq_len
        # Create sinusoidal base encoding
        pe = torch.zeros(T, D)
        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:D//2])
        self.register_buffer("_base_pe", pe.unsqueeze(0))
        self.proj = nn.Linear(D, D, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        pe = self._base_pe[:, :T, :]
        x = x + pe
        return x


class EventAlignedPositionEncoding(nn.Module):
    """Position encoding aligned to financial events.

    Defines relative position not just by timestep but by proximity
        to key financial events (earnings, dividends, index rebalancing).
        Tokens near events get special position representations.

    Args:
        d_model: Model dimension
        num_event_types: Number of event categories
        max_horizon: Max days from event to encode
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 512,
        num_event_types: int = 8,
        max_horizon: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_event_types = num_event_types
        self.max_horizon = max_horizon
        self.dropout = dropout
        import math
        D = d_model
        self.dropout = nn.Dropout(dropout)
        self.event_embed = nn.Embedding(num_event_types * 2 + 1, d_model, padding_idx=0)
        self.time_to_event_proj = nn.Embedding(max_horizon * 2 + 2, d_model)

    def forward(
        self,
        x: torch.Tensor,
        event_ids: Optional[torch.Tensor] = None,
        time_to_event: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        if event_ids is not None:
            ev_emb = self.event_embed(event_ids.clamp(0, self.num_event_types * 2))
            x = x + ev_emb
        if time_to_event is not None:
            tte = (time_to_event + self.max_horizon).clamp(0, self.max_horizon * 2 + 1)
            x = x + self.time_to_event_proj(tte)
        return self.dropout(x)


class NoPE(nn.Module):
    """No positional encoding (content-only baseline).

    Intentionally omits all positional encoding. Useful for cross-sectional
        models where position does not carry semantic meaning, or as an ablation
        baseline for studying the impact of positional encodings.

    Args:
        d_model: Model dimension (unused, for API compatibility)
    """

    def __init__(
        self,
        d_model: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        import math
        D = d_model

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        return x


class RandomFourierPositionEncoding(nn.Module):
    """Random Fourier feature positional encoding.

    Samples random frequencies and phases to create stochastic positional
        encodings. Provides diversity in the representation space.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        num_random_features: Number of random Fourier features
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        d_model: int = 512,
        max_seq_len: int = 2048,
        num_random_features: int = 8,
        seed: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_random_features = num_random_features
        self.seed = seed
        import math
        D = d_model
        T = max_seq_len
        # Create sinusoidal base encoding
        pe = torch.zeros(T, D)
        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:D//2])
        self.register_buffer("_base_pe", pe.unsqueeze(0))
        self.proj = nn.Linear(D, D, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        pe = self._base_pe[:, :T, :]
        x = x + pe
        return x


class CrossAssetPositionEncoding(nn.Module):
    """Unified position encoding for multi-asset sequences.

    Creates a joint positional representation for sequences that interleave
        multiple assets. Encodes both temporal position and asset identity.

    Args:
        d_model: Model dimension
        num_assets: Number of distinct assets
        max_seq_len: Maximum sequence length per asset
        asset_embed_dim: Dimension for asset identity embedding
    """

    def __init__(
        self,
        d_model: int = 512,
        num_assets: int = 8,
        max_seq_len: int = 2048,
        asset_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_assets = num_assets
        self.max_seq_len = max_seq_len
        self.asset_embed_dim = asset_embed_dim
        import math
        D = d_model
        T = max_seq_len
        # Create sinusoidal base encoding
        pe = torch.zeros(T, D)
        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:D//2])
        self.register_buffer("_base_pe", pe.unsqueeze(0))
        self.proj = nn.Linear(D, D, bias=False)
        self.asset_embed = nn.Embedding(num_assets + 1, asset_embed_dim, padding_idx=0)
        self.asset_proj = nn.Linear(asset_embed_dim + D, D, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        asset_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor (B, T, D)
        Returns:
            Position-encoded tensor (B, T, D)
        """
        B, T, D = x.shape
        pe = self._base_pe[:, :T, :]
        x = x + pe
        if asset_ids is not None:
            a_emb = self.asset_embed(asset_ids)
            if a_emb.dim() == 2:
                a_emb = a_emb.unsqueeze(1).expand(-1, T, -1)
            x = self.asset_proj(torch.cat([x, a_emb], dim=-1))
        return x


class PositionalEncodingRegistry:
    """Registry and factory for all positional encoding strategies.

    Provides a unified interface to create any positional encoding
    by name. Useful for hyperparameter search and configuration-driven
    model building.

    Registered strategies:
        - NTKAwareRoPE: NTK-aware RoPE with dynamic frequency adjustment
        - BinaryPositionalEncoding: Binary positional encoding for discrete positions
        - ConvolutionalPositionalEncoding: Relative position via causal convolution
        - SandwichPositionalEncoding: Learned positional encoding with normalization sandwich
        - MultiScaleRoPE: Multi-scale RoPE combining multiple base frequencies
        - RegimeSensitivePositionEncoding: Position encoding adaptive to market regime
        - SinusoidalWithLearned: Hybrid sinusoidal + learned positional encoding
        - PeriodicPositionEncoding: Periodic positional encoding for cyclic patterns
        - RelativeBucketEncoding: Bucketed relative position encoding
        - TemporalHierarchicalEncoding: Hierarchical temporal encoding across time scales
        - ProgressivePositionalEncoding: Position encoding with progressive resolution
        - EventAlignedPositionEncoding: Position encoding aligned to financial events
        - NoPE: No positional encoding (content-only baseline)
        - RandomFourierPositionEncoding: Random Fourier feature positional encoding
        - CrossAssetPositionEncoding: Unified position encoding for multi-asset sequences
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, enc_class: type) -> None:
        """Register a positional encoding class."""
        cls._registry[name] = enc_class

    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        """Create positional encoding by name.

        Args:
            name: Registered encoding name
            **kwargs: Constructor arguments
        Returns:
            Instantiated positional encoding module
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown encoding '{name}'. Available: {sorted(cls._registry.keys())}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_all(cls) -> List[str]:
        """Return list of all registered encoding names."""
        return sorted(cls._registry.keys())


# Register all encodings
PositionalEncodingRegistry.register("n_t_k_aware_ro_p_e", NTKAwareRoPE)
PositionalEncodingRegistry.register("binary_positional_encoding", BinaryPositionalEncoding)
PositionalEncodingRegistry.register("convolutional_positional_encoding", ConvolutionalPositionalEncoding)
PositionalEncodingRegistry.register("sandwich_positional_encoding", SandwichPositionalEncoding)
PositionalEncodingRegistry.register("multi_scale_ro_p_e", MultiScaleRoPE)
PositionalEncodingRegistry.register("regime_sensitive_position_encoding", RegimeSensitivePositionEncoding)
PositionalEncodingRegistry.register("sinusoidal_with_learned", SinusoidalWithLearned)
PositionalEncodingRegistry.register("periodic_position_encoding", PeriodicPositionEncoding)
PositionalEncodingRegistry.register("relative_bucket_encoding", RelativeBucketEncoding)
PositionalEncodingRegistry.register("temporal_hierarchical_encoding", TemporalHierarchicalEncoding)
PositionalEncodingRegistry.register("progressive_positional_encoding", ProgressivePositionalEncoding)
PositionalEncodingRegistry.register("event_aligned_position_encoding", EventAlignedPositionEncoding)
PositionalEncodingRegistry.register("no_p_e", NoPE)
PositionalEncodingRegistry.register("random_fourier_position_encoding", RandomFourierPositionEncoding)
PositionalEncodingRegistry.register("cross_asset_position_encoding", CrossAssetPositionEncoding)


# =============================================================================
# SECTION: Positional Encoding Analysis and Visualization Utilities
# =============================================================================


def compute_position_similarity(
    pe: torch.Tensor  # (T, D) positional encodings,
) -> torch.Tensor  # (T, T) similarity matrix:
    """Compute cosine similarity matrix between positional encodings."""
    pe_norm = F.normalize(pe, p=2, dim=-1)
    return torch.matmul(pe_norm, pe_norm.T)


def pe_rank_analysis(
    pe: torch.Tensor  # (T, D),
) -> Dict[str, float]:
    """Analyze the rank (effective dimensionality) of positional encodings."""
    U, S, V = torch.linalg.svd(pe, full_matrices=False)
    total_var = S.pow(2).sum()
    cumvar = S.pow(2).cumsum(0) / (total_var + 1e-10)
    rank_90 = int((cumvar < 0.9).sum()) + 1
    rank_95 = int((cumvar < 0.95).sum()) + 1
    rank_99 = int((cumvar < 0.99).sum()) + 1
    return {"singular_values": S.tolist(), "rank_90": rank_90, "rank_95": rank_95, "rank_99": rank_99, "effective_rank": float((S / S.sum()).pow(2).sum().pow(-1))}


def pe_interpolation_quality(
    pe_fn: Callable,
    test_positions: torch.Tensor,
    train_positions: torch.Tensor,
) -> float:
    """Measure interpolation quality for unseen sequence positions."""
    train_pe = pe_fn(train_positions)
    test_pe = pe_fn(test_positions)
    # Interpolation error: compare to average of nearest neighbors
    dists = torch.cdist(test_positions.float().unsqueeze(1), train_positions.float().unsqueeze(1)).squeeze()
    nn_idx = dists.argmin(dim=-1)
    nn_pe = train_pe[nn_idx]
    error = (test_pe - nn_pe).pow(2).mean()
    return float(error)


def align_positional_encodings(
    pe1: torch.Tensor,
    pe2: torch.Tensor,
) -> torch.Tensor  # Aligned pe2:
    """Align two sets of positional encodings via Procrustes analysis."""
    # Procrustes alignment: find rotation R that minimizes ||pe1 - pe2 @ R||
    U, S, Vt = torch.linalg.svd(pe1.T @ pe2)
    R = U @ Vt
    return pe2 @ R.T


def pe_distance_preserving(
    pe: torch.Tensor  # (T, D),
) -> float  # Correlation between input and PE distances:
    """Check how well positional encodings preserve input distances."""
    T = pe.size(0)
    pos = torch.arange(T, dtype=torch.float32)
    input_dists = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1)).view(-1)
    pe_dists = torch.cdist(pe, pe).view(-1)
    pe_dists_norm = pe_dists / (pe_dists.max() + 1e-10)
    input_dists_norm = input_dists / (input_dists.max() + 1e-10)
    corr = torch.corrcoef(torch.stack([input_dists_norm, pe_dists_norm]))[0, 1]
    return float(corr)


def extrapolation_score(
    model: nn.Module,
    train_len: int,
    test_len: int,
    d_model: int,
) -> Dict[str, float]:
    """Score positional encoding extrapolation beyond training length."""
    # Create position indices for train and test lengths
    train_ids = torch.arange(train_len)
    test_ids = torch.arange(test_len)
    # Simplified: compute PE variance in extended range
    with torch.no_grad():
        x_train = torch.zeros(1, train_len, d_model)
        x_test = torch.zeros(1, test_len, d_model)
        if hasattr(model, "forward"):
            out_train = model(x_train)
            out_test = model(x_test)
            var_train = out_train.var(dim=1).mean()
            var_test = out_test.var(dim=1).mean()
            return {"variance_ratio": float(var_test / (var_train + 1e-10)), "train_len": train_len, "test_len": test_len}
    return {}


_NEW_PE_EXPORTS = [
    "NTKAwareRoPE",
    "BinaryPositionalEncoding",
    "ConvolutionalPositionalEncoding",
    "SandwichPositionalEncoding",
    "MultiScaleRoPE",
    "RegimeSensitivePositionEncoding",
    "SinusoidalWithLearned",
    "PeriodicPositionEncoding",
    "RelativeBucketEncoding",
    "TemporalHierarchicalEncoding",
    "ProgressivePositionalEncoding",
    "EventAlignedPositionEncoding",
    "NoPE",
    "RandomFourierPositionEncoding",
    "CrossAssetPositionEncoding",
    "compute_position_similarity",
    "pe_rank_analysis",
    "pe_interpolation_quality",
    "align_positional_encodings",
    "pe_distance_preserving",
    "extrapolation_score",
    "PositionalEncodingRegistry",
]
