"""
lumina/positional_encoding.py

Positional encoding schemes for Lumina:

  - RotaryPositionalEncoding (RoPE)
  - ALiBiPositionalBias
  - TemporalEncoding (real timestamps, market sessions)
  - FourierTimeEncoding (learnable Fourier features)
  - CrossModalPositionalEncoding
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary Positional Encoding (RoPE)
# ---------------------------------------------------------------------------
class RotaryPositionalEncoding(nn.Module):
    """
    RoPE: Rotary Position Embedding (Su et al. 2021, RoFormer).

    Applied to Q and K tensors in attention.
    Encodes relative positions by rotating pairs of dimensions by angles
    theta_i = pos / 10000^(2i/d).

    Usage:
        rope = RotaryPositionalEncoding(dim=64, max_seq_len=4096)
        q, k = rope(q, k, seq_len=T)
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires even dimension"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)   # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (B, n_heads, T, head_dim)
            k: (B, n_heads, T, head_dim)
            seq_len: length of sequence
            position_ids: (B, T) optional custom positions

        Returns:
            q_rot, k_rot: rotated tensors, same shape
        """
        T = q.shape[-2]
        if seq_len is None:
            seq_len = T

        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len)

        if position_ids is not None:
            cos = self.cos_cache[position_ids]   # (B, T, dim)
            sin = self.sin_cache[position_ids]
            cos = cos.unsqueeze(1)  # (B, 1, T, dim)
            sin = sin.unsqueeze(1)
        else:
            cos = self.cos_cache[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim)
            sin = self.sin_cache[:T].unsqueeze(0).unsqueeze(0)

        d = self.dim
        q_rot = (q[..., :d] * cos) + (self._rotate_half(q[..., :d]) * sin)
        k_rot = (k[..., :d] * cos) + (self._rotate_half(k[..., :d]) * sin)

        if q.shape[-1] > d:
            q_rot = torch.cat([q_rot, q[..., d:]], dim=-1)
            k_rot = torch.cat([k_rot, k[..., d:]], dim=-1)

        return q_rot, k_rot


# ---------------------------------------------------------------------------
# ALiBi (Attention with Linear Biases)
# ---------------------------------------------------------------------------
class ALiBiPositionalBias(nn.Module):
    """
    ALiBi: Attention with Linear Biases (Press et al. 2021).

    Adds a fixed linear bias to attention logits:
    bias[i,j] = -m * |i - j| where m is a head-specific slope.
    """

    def __init__(self, n_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        slopes = self._get_slopes(n_heads)
        self.register_buffer("slopes", slopes)
        self._precompute_bias(max_seq_len)

    @staticmethod
    def _get_slopes(n: int) -> torch.Tensor:
        def _slopes_power_of_2(n: int):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            slopes = _slopes_power_of_2(n)
        else:
            closest_power = 2 ** math.floor(math.log2(n))
            base = _slopes_power_of_2(closest_power)
            extra = _slopes_power_of_2(2 * closest_power)
            extra = extra[0::2][: n - closest_power]
            slopes = base + extra

        return torch.tensor(slopes, dtype=torch.float32)

    def _precompute_bias(self, seq_len: int):
        positions = torch.arange(seq_len)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = -relative_positions.abs().float()
        bias = self.slopes.unsqueeze(-1).unsqueeze(-1) * relative_positions.unsqueeze(0)
        self.register_buffer("bias_cache", bias, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Returns: (1, n_heads, T, T)"""
        if seq_len > self.bias_cache.shape[-1]:
            self._precompute_bias(seq_len)
        bias = self.bias_cache[:, :seq_len, :seq_len]
        return bias.unsqueeze(0)

    def causal_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        bias = self.forward(seq_len, device)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        bias = bias.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        return bias


# ---------------------------------------------------------------------------
# Market Session constants
# ---------------------------------------------------------------------------
class MarketSession:
    PRE_MARKET = 0
    RTH = 1
    AFTER_HOURS = 2
    WEEKEND = 3


# ---------------------------------------------------------------------------
# Temporal Encoding
# ---------------------------------------------------------------------------
class TemporalEncoding(nn.Module):
    """
    Encodes real timestamps as embedding vectors.

    Features:
      - Hour-of-day (0-23): sin/cos pair → 2 dims
      - Day-of-week (0-6): sin/cos → 2 dims
      - Day-of-month (1-31): sin/cos → 2 dims
      - Month (1-12): sin/cos → 2 dims
      - Market session embedding (4 categories) → d_session dims
      - Linear projection to d_model
    """

    def __init__(self, d_model: int, d_session: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_session = d_session

        sin_dim = 8
        self.session_emb = nn.Embedding(4, d_session)
        self.proj = nn.Linear(sin_dim + d_session, d_model)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _to_sincos(value: torch.Tensor, period: float) -> torch.Tensor:
        angle = 2 * math.pi * value / period
        return torch.stack([angle.sin(), angle.cos()], dim=-1)

    def _classify_session(self, hour: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        is_weekend = (dow >= 5)
        is_rth = (~is_weekend) & (hour >= 9) & (hour < 16)
        is_pre = (~is_weekend) & (hour < 9)
        is_after = (~is_weekend) & (hour >= 16)
        session = torch.zeros_like(hour, dtype=torch.long)
        session[is_pre] = MarketSession.PRE_MARKET
        session[is_rth] = MarketSession.RTH
        session[is_after] = MarketSession.AFTER_HOURS
        session[is_weekend] = MarketSession.WEEKEND
        return session

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (B, T) Unix timestamps in seconds

        Returns:
            encoding: (B, T, d_model)
        """
        seconds_per_day = 86400
        seconds_per_hour = 3600

        ts = timestamps.float()
        hour = (ts % seconds_per_day / seconds_per_hour).long().clamp(0, 23).float()
        dow = ((ts / seconds_per_day).long() + 3) % 7
        day_of_month = ((ts / seconds_per_day).long() % 31 + 1).float()
        month = ((ts / (seconds_per_day * 30.4)).long() % 12 + 1).float()

        sc_hour = self._to_sincos(hour, 24.0)
        sc_dow = self._to_sincos(dow.float(), 7.0)
        sc_dom = self._to_sincos(day_of_month, 31.0)
        sc_month = self._to_sincos(month, 12.0)

        sin_features = torch.cat([sc_hour, sc_dow, sc_dom, sc_month], dim=-1)

        session_ids = self._classify_session(hour.long(), dow)
        session_features = self.session_emb(session_ids)

        combined = torch.cat([sin_features, session_features], dim=-1)
        out = self.norm(self.proj(combined))
        return out


# ---------------------------------------------------------------------------
# Fourier Time Encoding
# ---------------------------------------------------------------------------
class FourierTimeEncoding(nn.Module):
    """
    Learnable Fourier features for timestamps.

    Maps scalar timestamp t → [sin(w_1 t + b_1), cos(w_1 t + b_1), ...]
    projected to d_model.
    """

    def __init__(self, d_model: int, n_fourier: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_fourier = n_fourier

        self.weights = nn.Parameter(torch.randn(n_fourier) * 0.1)
        self.phases = nn.Parameter(torch.zeros(n_fourier))
        self.proj = nn.Linear(2 * n_fourier, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (B, T) float

        Returns:
            encoding: (B, T, d_model)
        """
        t = timestamps.float()
        t_mean = t.mean(dim=-1, keepdim=True)
        t_std = t.std(dim=-1, keepdim=True).clamp(min=1.0)
        t_norm = (t - t_mean) / t_std

        angles = (t_norm.unsqueeze(-1) * self.weights.unsqueeze(0).unsqueeze(0)
                  + self.phases.unsqueeze(0).unsqueeze(0))
        features = torch.cat([angles.sin(), angles.cos()], dim=-1)

        out = self.norm(self.proj(features))
        return out


# ---------------------------------------------------------------------------
# CrossModalPositionalEncoding
# ---------------------------------------------------------------------------
class CrossModalPositionalEncoding(nn.Module):
    """
    Separate position spaces per modality, with cross-modal relative positions.
    """

    def __init__(
        self,
        d_model: int,
        n_modalities: int = 4,
        max_seq_len: int = 1024,
        max_cross_distance: int = 512,
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
            f"{i}_{j}": nn.Parameter(torch.zeros(2 * max_cross_distance + 1))
            for i in range(n_modalities)
            for j in range(n_modalities)
            if i != j
        })

        self._init_sinusoidal()

    def _init_sinusoidal(self):
        for emb in self.pos_embeddings:
            d = emb.embedding_dim
            positions = torch.arange(emb.num_embeddings).unsqueeze(1).float()
            dims = torch.arange(0, d, 2).float()
            angles = positions / (10000.0 ** (dims / d))
            pe = torch.zeros(emb.num_embeddings, d)
            pe[:, 0::2] = angles.sin()
            pe[:, 1::2] = angles.cos()
            emb.weight.data.copy_(pe)

    def encode_modality(self, modality_id: int, seq_len: int, device: torch.device) -> torch.Tensor:
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        return self.pos_embeddings[modality_id](pos_ids)

    def cross_modal_bias(
        self,
        src_modality: int,
        tgt_modality: int,
        src_positions: torch.Tensor,
        tgt_positions: torch.Tensor,
    ) -> torch.Tensor:
        rel = src_positions.unsqueeze(-1) - tgt_positions.unsqueeze(-2)
        rel = rel.clamp(-self.max_cross_distance, self.max_cross_distance)
        rel_idx = rel + self.max_cross_distance
        key = f"{src_modality}_{tgt_modality}"
        bias_table = self.cross_biases[key]
        return bias_table[rel_idx]

    def forward(
        self,
        token_embeddings: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add per-modality positional embeddings to token embeddings.

        Args:
            token_embeddings: (B, T, d_model)
            modality_ids:     (B, T) long

        Returns:
            token_embeddings + positional encodings: (B, T, d_model)
        """
        B, T, D = token_embeddings.shape
        out = token_embeddings.clone()

        for mod_id in range(self.n_modalities):
            mask = (modality_ids == mod_id)
            if not mask.any():
                continue

            for b in range(B):
                idx = mask[b].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                n = idx.numel()
                pos_ids = torch.arange(n, device=token_embeddings.device).unsqueeze(0)
                pos_emb = self.pos_embeddings[mod_id](pos_ids).squeeze(0)
                out[b, idx] = out[b, idx] + pos_emb

        return out
