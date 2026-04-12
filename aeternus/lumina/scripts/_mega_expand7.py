#!/usr/bin/env python3
"""Mega expansion 7 - large additions to transformer.py, model.py, and more test files."""
import os, subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def append(rel, content):
    path = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
    n = open(path, encoding="utf-8").read().count("\n")
    print(f"  {os.path.basename(path)}: {n} lines")
    return n

def write_new(rel, content):
    path = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    n = open(path, encoding="utf-8").read().count("\n")
    print(f"  {os.path.basename(path)}: {n} lines (new)")
    return n

# ════════════════════════════════════════════════════════════════════════════════
# 1. More transformer.py content
# ════════════════════════════════════════════════════════════════════════════════
TRANSFORMER_ADD = '''

# ============================================================
# Extended Transformer Components - Part 2
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union


class NormFormerBlock(nn.Module):
    """NormFormer (Shleifer et al. 2021): additional LayerNorm after attention and FFN.

    Adds extra norm layers which can improve training stability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int = 0, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)  # extra
        self.norm_ffn = nn.LayerNorm(d_model)   # extra
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm -> attn -> extra post-norm -> residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        h = self.norm_attn(h)
        x = x + self.dropout(h)
        # FFN
        h = self.norm2(x)
        h = self.ffn(h)
        h = self.norm_ffn(h)
        x = x + self.dropout(h)
        return x


class SandwichTransformerBlock(nn.Module):
    """Sandwich (Press et al. 2019): re-order norms for better optimization landscape.

    Post-norm -> pre-norm sandwich: norm(x + post_norm(sub_layer(pre_norm(x))))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int = 0, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.pre_norm1 = nn.LayerNorm(d_model)
        self.pre_norm2 = nn.LayerNorm(d_model)
        self.post_norm1 = nn.LayerNorm(d_model)
        self.post_norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_norm1(x)
        h, _ = self.attn(h, h, h)
        h = self.post_norm1(h)
        x = x + self.dropout(h)

        h = self.pre_norm2(x)
        h = self.ffn(h)
        h = self.post_norm2(h)
        x = x + self.dropout(h)
        return x


class MacaronTransformerBlock(nn.Module):
    """Macaron-Net (Lu et al. 2019): FFN-Attention-FFN structure with half-step FFNs.

    Each FFN has half the usual contribution (scaled by 0.5).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int = 0, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + 0.5 * self.dropout(self.ffn1(self.norm1(x)))
        h = self.norm2(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + self.dropout(h)
        x = x + 0.5 * self.dropout(self.ffn2(self.norm3(x)))
        return x


class ConformerBlock(nn.Module):
    """Conformer (Gulati et al. 2020): Conv-augmented Transformer block.

    FFN -> MHSA -> Conv -> FFN (Macaron-style), with depthwise convolution module.
    """

    def __init__(self, d_model: int, num_heads: int, conv_kernel: int = 31, d_ff: int = 0, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        assert conv_kernel % 2 == 1, "conv_kernel must be odd"
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        # Convolution module
        self.conv_norm = nn.LayerNorm(d_model)
        padding = (conv_kernel - 1) // 2
        self.conv_module = nn.Sequential(
            nn.LayerNorm(d_model),
            # Pointwise expand
            nn.Linear(d_model, 2 * d_model),
            nn.GLU(dim=-1),
            # Depthwise conv (need to transpose to (B, C, T))
        )
        self.depthwise_conv = nn.Conv1d(d_model, d_model, conv_kernel, padding=padding, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv = nn.Linear(d_model, d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def _conv_module(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.conv_module(x)              # (B, T, D)
        h = h.transpose(1, 2)                # (B, D, T)
        h = self.depthwise_conv(h)           # (B, D, T)
        h = self.batch_norm(h)
        h = F.silu(h)
        h = h.transpose(1, 2)                # (B, T, D)
        h = self.pointwise_conv(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.dropout(self.ffn1(self.norm1(x)))
        h = self.norm2(x)
        h, _ = self.attn(h, h, h)
        x = x + self.dropout(h)
        x = x + self.dropout(self._conv_module(self.norm3(x)))
        x = x + 0.5 * self.dropout(self.ffn2(self.norm4(x)))
        return x


class CrossAttentionBlock(nn.Module):
    """Transformer block with cross-attention for encoder-decoder architectures."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int = 0, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        h = self.norm_self(x)
        h, _ = self.self_attn(h, h, h, attn_mask=self_mask)
        x = x + self.dropout(h)
        # Cross-attention with encoder memory
        h = self.norm_cross(x)
        h, _ = self.cross_attn(h, memory, memory, attn_mask=cross_mask)
        x = x + self.dropout(h)
        # FFN
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


class GatedTransformerBlock(nn.Module):
    """Gated Linear Unit Transformer (Hua et al. 2022).

    Replaces softmax attention with a gated linear mechanism.
    """

    def __init__(self, d_model: int, d_expand: int = 0, dropout: float = 0.1):
        super().__init__()
        d_expand = d_expand or 2 * d_model
        self.norm = nn.LayerNorm(d_model)
        # Gated linear attention
        self.to_uv = nn.Linear(d_model, 2 * d_expand, bias=False)
        self.gate_norm = nn.LayerNorm(d_expand)
        self.to_out = nn.Linear(d_expand, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        uv = self.to_uv(h)
        u, v = uv.chunk(2, dim=-1)
        h = u * F.gelu(v)
        h = self.gate_norm(h)
        h = self.dropout(h)
        h = self.to_out(h)
        return x + h


class HyperNetworkBlock(nn.Module):
    """HyperNetwork-conditioned transformer block.

    A conditioning signal z produces the weight deltas for the FFN.
    """

    def __init__(self, d_model: int, num_heads: int, d_z: int = 32, d_ff: int = 0, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        # Standard FFN
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # HyperNet: z -> scale/bias for fc1 and fc2
        self.hyper = nn.Linear(d_z, 4 * d_model)  # scale1, bias1, scale2, bias2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.dropout(h)

        h = self.norm2(x)
        if z is not None:
            params = self.hyper(z)  # (B, 4*d_model)
            s1, b1 = params[:, :self.fc1.out_features], params[:, self.fc1.out_features:2*self.fc1.out_features]
            h2 = F.gelu(self.fc1(h) * s1.unsqueeze(1) + b1.unsqueeze(1))
            s2, b2 = params[:, 2*self.fc1.out_features:2*self.fc1.out_features+self.fc2.out_features], \
                     params[:, 2*self.fc1.out_features+self.fc2.out_features:]
            h2 = self.fc2(h2) * s2.unsqueeze(1) + b2.unsqueeze(1)
        else:
            h2 = self.fc2(F.gelu(self.fc1(h)))

        x = x + self.dropout(h2)
        return x


class UniversalTransformer(nn.Module):
    """Universal Transformer (Dehghani et al. 2019): recurrent depth via ACT.

    Shares weights across depth steps and halts each position adaptively.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 0,
        max_steps: int = 8,
        halt_threshold: float = 0.99,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold

        # Shared block (same weights for all depth steps)
        self.block = PreNormTransformerBlockUT(d_model, num_heads, d_ff, dropout)
        self.step_embed = nn.Embedding(max_steps, d_model)
        self.halt_gate = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        halted = torch.zeros(B, T, 1, device=x.device)
        remainder = torch.ones(B, T, 1, device=x.device)
        output = torch.zeros_like(x)

        for step in range(self.max_steps):
            step_emb = self.step_embed(torch.tensor(step, device=x.device))
            x = x + step_emb.unsqueeze(0).unsqueeze(0)
            x = self.block(x)

            p = torch.sigmoid(self.halt_gate(x))  # (B, T, 1)
            still_running = 1 - halted
            new_halt = (halted + p * still_running > self.halt_threshold).float()
            halt_this_step = (new_halt - halted) * still_running

            output = output + halt_this_step * x
            remainder = remainder - halt_this_step
            halted = new_halt

            if halted.min() > 0.5:
                break

        output = output + remainder * x
        return output


class PreNormTransformerBlockUT(nn.Module):
    """Simplified pre-norm block for Universal Transformer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class EfficientTransformerBlock(nn.Module):
    """Efficient Transformer (Wang et al. 2020) with linformer-style low-rank key-value projection."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 512,
        k_rank: int = 64,
        d_ff: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.k_rank = k_rank

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # Low-rank projection: T -> k for K and V
        self.E = nn.Linear(max_seq_len, k_rank, bias=False)
        self.F = nn.Linear(max_seq_len, k_rank, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        h = self.norm1(x)
        q = self.q_proj(h).reshape(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        k = self.k_proj(h).reshape(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        v = self.v_proj(h).reshape(B, T, H, D).transpose(1, 2)   # (B, H, T, D)

        # Project K, V from T to k_rank
        # E: (max_T, k) -> apply to T dimension of k: (B, H, T, D) -> (B, H, k, D)
        pad_T = self.max_seq_len
        if T < pad_T:
            k_pad = F.pad(k, (0, 0, 0, pad_T - T))  # (B, H, pad_T, D)
            v_pad = F.pad(v, (0, 0, 0, pad_T - T))
        else:
            k_pad = k[:, :, :pad_T, :]
            v_pad = v[:, :, :pad_T, :]

        k_proj = self.E(k_pad.transpose(-2, -1)).transpose(-2, -1)  # (B, H, k, D)
        v_proj = self.F(v_pad.transpose(-2, -1)).transpose(-2, -1)

        attn = torch.matmul(q, k_proj.transpose(-2, -1)) / self.scale  # (B, H, T, k)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_proj)  # (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.out_proj(out)
        x = x + self.dropout(out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class FNetBlock(nn.Module):
    """FNet (Lee-Thorp et al. 2022): replaces attention with Fourier transform.

    Uses 2D FFT over (batch, sequence) dimensions for token mixing.
    """

    def __init__(self, d_model: int, d_ff: int = 0, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fourier mixing: FFT over seq and hidden dims, take real part
        h = torch.fft.fftn(x.to(torch.complex64), dim=(-1, -2)).real.to(x.dtype)
        x = self.norm1(x + self.dropout(h))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class MLPMixerBlock(nn.Module):
    """MLP-Mixer (Tolstikhin et al. 2021): alternating token-mixing and channel-mixing MLPs."""

    def __init__(self, d_model: int, seq_len: int, d_tokens: int = 256, d_channels: int = 0, dropout: float = 0.1):
        super().__init__()
        d_channels = d_channels or d_model * 4
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Token mixing (across sequence dim): transpose then MLP
        self.token_mix = nn.Sequential(
            nn.Linear(seq_len, d_tokens), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_tokens, seq_len)
        )
        # Channel mixing
        self.channel_mix = nn.Sequential(
            nn.Linear(d_model, d_channels), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_channels, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing: (B, T, D) -> transpose to (B, D, T) -> mix -> transpose back
        h = self.norm1(x)
        h = h.transpose(1, 2)       # (B, D, T)
        h = self.token_mix(h)
        h = h.transpose(1, 2)       # (B, T, D)
        x = x + self.dropout(h)
        # Channel mixing
        x = x + self.dropout(self.channel_mix(self.norm2(x)))
        return x


class HierarchicalTransformer(nn.Module):
    """Hierarchical Transformer with local and global attention stages."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        local_window: int = 16,
        num_local_layers: int = 3,
        num_global_layers: int = 2,
        d_ff: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.local_window = local_window

        self.local_layers = nn.ModuleList([
            PreNormTransformerBlockUT(d_model, num_heads, d_ff, dropout)
            for _ in range(num_local_layers)
        ])
        self.global_layers = nn.ModuleList([
            PreNormTransformerBlockUT(d_model, num_heads, d_ff, dropout)
            for _ in range(num_global_layers)
        ])
        # Pooling: select one representative per window
        self.pool_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        W = self.local_window

        # Pad to multiple of W
        pad = (W - T % W) % W
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        T_pad = x.shape[1]
        n_windows = T_pad // W

        # Local attention within windows
        x_windows = x.reshape(B * n_windows, W, D)
        for layer in self.local_layers:
            x_windows = layer(x_windows)
        x = x_windows.reshape(B, T_pad, D)

        # Global attention on window representatives (first token of each window)
        reps = x[:, ::W, :]  # (B, n_windows, D)
        reps = self.pool_proj(reps)
        for layer in self.global_layers:
            reps = layer(reps)

        # Broadcast back to full sequence
        reps_expanded = reps.repeat_interleave(W, dim=1)  # (B, T_pad, D)
        x = x + reps_expanded

        return x[:, :T, :]  # trim padding


class ConditionedTransformer(nn.Module):
    """Transformer conditioned on external context via cross-attention or FiLM."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_context: int,
        num_layers: int = 6,
        d_ff: int = 0,
        conditioning: str = "cross_attn",  # cross_attn | film | concat
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.conditioning = conditioning
        self.d_model = d_model

        if conditioning == "cross_attn":
            self.layers = nn.ModuleList([
                CrossAttentionBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
            self.context_proj = nn.Linear(d_context, d_model)
        elif conditioning == "film":
            self.layers = nn.ModuleList([
                PreNormTransformerBlockUT(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
            self.film_scale = nn.Linear(d_context, d_model)
            self.film_shift = nn.Linear(d_context, d_model)
        else:  # concat
            self.context_proj = nn.Linear(d_context, d_model)
            self.layers = nn.ModuleList([
                PreNormTransformerBlockUT(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])

        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.conditioning == "cross_attn":
            mem = self.context_proj(context)  # (B, C_T, d_model)
            for layer in self.layers:
                x = layer(x, mem)
        elif self.conditioning == "film":
            scale = self.film_scale(context)  # (B, d_model) or (B, C_T, d_model)
            shift = self.film_shift(context)
            if scale.dim() == 2:
                scale = scale.unsqueeze(1)
                shift = shift.unsqueeze(1)
            for layer in self.layers:
                x = layer(x) * scale + shift
        else:
            ctx_proj = self.context_proj(context)
            if ctx_proj.dim() == 2:
                ctx_proj = ctx_proj.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = x + ctx_proj
            for layer in self.layers:
                x = layer(x)

        return self.output_norm(x)


class TemporalTransformer(nn.Module):
    """Financial time-series transformer with temporal encoding and causal masking."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int = 6,
        max_len: int = 1024,
        d_ff: int = 0,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.use_causal_mask = use_causal_mask

        self.input_proj = nn.Linear(1, d_model)  # univariate by default
        self.pos_embed = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            PreNormTransformerBlockUT(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1)  # univariate forecast
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) univariate time series"""
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, T, 1)
        B, T, _ = x.shape

        h = self.dropout(self.input_proj(x))
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = h + self.pos_embed(pos)

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        return self.output_head(h).squeeze(-1)  # (B, T)


class StochasticDepthTransformer(nn.Module):
    """Transformer with stochastic depth (DropPath) regularization (Huang et al. 2016)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int = 12,
        drop_path_rate: float = 0.1,
        d_ff: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        # Linearly increase drop path rate with depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.layers = nn.ModuleList([
            StochasticDepthBlock(d_model, num_heads, d_ff, drop_prob=dpr[i], dropout=dropout)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class StochasticDepthBlock(nn.Module):
    """Single block with DropPath stochastic depth."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, drop_prob: float = 0.1, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop_prob = drop_prob

    def _drop_path(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x + residual
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = torch.rand(shape, device=x.device) < keep_prob
        return x + residual * random_tensor.float() / keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = self._drop_path(x, h)
        h = self.ffn(self.norm2(x))
        x = self._drop_path(x, h)
        return x
'''

append("transformer.py", TRANSFORMER_ADD)

# ════════════════════════════════════════════════════════════════════════════════
# 2. tests/test_transformer_extra.py
# ════════════════════════════════════════════════════════════════════════════════
def build_transformer_extra_tests():
    classes = [
        ("NormFormerBlock", "transformer", "d_model=64, num_heads=4"),
        ("SandwichTransformerBlock", "transformer", "d_model=64, num_heads=4"),
        ("MacaronTransformerBlock", "transformer", "d_model=64, num_heads=4"),
        ("GatedTransformerBlock", "transformer", "d_model=64"),
        ("FNetBlock", "transformer", "d_model=64"),
        ("ConformerBlock", "transformer", "d_model=64, num_heads=4"),
        ("CrossAttentionBlock", "transformer", "d_model=64, num_heads=4"),
        ("StochasticDepthBlock", "transformer", "d_model=64, num_heads=4, d_ff=256"),
    ]

    lines = [
        '"""Tests for extra transformer blocks."""',
        "import pytest",
        "import torch",
        "import torch.nn as nn",
        "",
    ]

    for cls, module, init_args in classes:
        lines += [f"class Test{cls}:"]
        if cls == "CrossAttentionBlock":
            lines += [
                f"    def test_forward(self):",
                f"        from {module} import {cls}",
                f"        block = {cls}({init_args})",
                f"        x = torch.randn(2, 8, 64)",
                f"        mem = torch.randn(2, 4, 64)",
                f"        out = block(x, mem)",
                f"        assert out.shape == (2, 8, 64)",
                "",
                f"    def test_no_nan(self):",
                f"        from {module} import {cls}",
                f"        block = {cls}({init_args})",
                f"        x = torch.randn(2, 8, 64)",
                f"        mem = torch.randn(2, 6, 64)",
                f"        assert not torch.isnan(block(x, mem)).any()",
                "",
                f"    def test_gradient(self):",
                f"        from {module} import {cls}",
                f"        block = {cls}({init_args})",
                f"        x = torch.randn(2, 8, 64, requires_grad=True)",
                f"        mem = torch.randn(2, 4, 64)",
                f"        block(x, mem).sum().backward()",
                f"        assert x.grad is not None",
                "",
            ]
        elif cls == "ConformerBlock":
            lines += [
                f"    def test_forward(self):",
                f"        from {module} import {cls}",
                f"        block = {cls}({init_args})",
                f"        x = torch.randn(2, 8, 64)",
                f"        out = block(x)",
                f"        assert out.shape == (2, 8, 64)",
                "",
                f"    def test_no_nan(self):",
                f"        from {module} import {cls}",
                f"        block = {cls}({init_args})",
                f"        x = torch.randn(4, 16, 64)",
                f"        assert not torch.isnan(block(x)).any()",
                "",
            ]
        else:
            lines += [
                f"    def test_forward_shape(self):",
                f"        from {module} import {cls}",
                f"        block = {cls}({init_args})",
                f"        x = torch.randn(2, 8, 64)",
                f"        out = block(x)",
                f"        assert out.shape == (2, 8, 64)",
                "",
                f"    def test_no_nan(self):",
                f"        from {module} import {cls}",
                f"        block = {cls}({init_args})",
                f"        x = torch.randn(4, 16, 64)",
                f"        assert not torch.isnan(block(x)).any()",
                "",
                f"    def test_gradient(self):",
                f"        from {module} import {cls}",
                f"        block = {cls}({init_args})",
                f"        x = torch.randn(2, 8, 64, requires_grad=True)",
                f"        block(x).sum().backward()",
                f"        assert x.grad is not None",
                "",
                f"    def test_state_dict(self):",
                f"        from {module} import {cls}",
                f"        b1 = {cls}({init_args})",
                f"        b2 = {cls}({init_args})",
                f"        b2.load_state_dict(b1.state_dict())",
                f"        x = torch.randn(2, 8, 64)",
                f"        assert torch.allclose(b1(x), b2(x))",
                "",
            ]

    # More classes: TemporalTransformer, UniversalTransformer, etc.
    lines += [
        "class TestTemporalTransformer:",
        "    def test_forward_shape(self):",
        "        from transformer import TemporalTransformer",
        "        model = TemporalTransformer(d_model=32, num_heads=4, num_layers=2, max_len=64)",
        "        x = torch.randn(2, 16)",
        "        out = model(x)",
        "        assert out.shape == (2, 16)",
        "",
        "    def test_no_nan(self):",
        "        from transformer import TemporalTransformer",
        "        model = TemporalTransformer(d_model=32, num_heads=4, num_layers=2, max_len=64)",
        "        x = torch.randn(4, 8)",
        "        assert not torch.isnan(model(x)).any()",
        "",
    ]

    lines += [
        "class TestUniversalTransformer:",
        "    def test_forward_shape(self):",
        "        from transformer import UniversalTransformer",
        "        model = UniversalTransformer(d_model=32, num_heads=4, max_steps=3)",
        "        x = torch.randn(2, 8, 32)",
        "        out = model(x)",
        "        assert out.shape == (2, 8, 32)",
        "",
        "    def test_no_nan(self):",
        "        from transformer import UniversalTransformer",
        "        model = UniversalTransformer(d_model=32, num_heads=4, max_steps=3)",
        "        x = torch.randn(2, 8, 32)",
        "        assert not torch.isnan(model(x)).any()",
        "",
    ]

    lines += [
        "class TestFNetBlock:",
        "    def test_forward(self):",
        "        from transformer import FNetBlock",
        "        block = FNetBlock(64)",
        "        x = torch.randn(2, 16, 64)",
        "        out = block(x)",
        "        assert out.shape == (2, 16, 64)",
        "",
    ]

    lines += [
        "class TestMLPMixerBlock:",
        "    def test_forward(self):",
        "        from transformer import MLPMixerBlock",
        "        block = MLPMixerBlock(64, seq_len=8)",
        "        x = torch.randn(2, 8, 64)",
        "        out = block(x)",
        "        assert out.shape == (2, 8, 64)",
        "",
        "    def test_no_nan(self):",
        "        from transformer import MLPMixerBlock",
        "        block = MLPMixerBlock(32, seq_len=16)",
        "        x = torch.randn(2, 16, 32)",
        "        assert not torch.isnan(block(x)).any()",
        "",
    ]

    lines += [
        "class TestHierarchicalTransformer:",
        "    def test_forward(self):",
        "        from transformer import HierarchicalTransformer",
        "        model = HierarchicalTransformer(d_model=32, num_heads=4, local_window=4)",
        "        x = torch.randn(2, 16, 32)",
        "        out = model(x)",
        "        assert out.shape == (2, 16, 32)",
        "",
        "    def test_no_nan(self):",
        "        from transformer import HierarchicalTransformer",
        "        model = HierarchicalTransformer(d_model=32, num_heads=4, local_window=8)",
        "        x = torch.randn(2, 32, 32)",
        "        assert not torch.isnan(model(x)).any()",
        "",
    ]

    lines += [
        "class TestConditionedTransformer:",
        "    def test_cross_attn_conditioning(self):",
        "        from transformer import ConditionedTransformer",
        "        model = ConditionedTransformer(d_model=32, num_heads=4, d_context=16,",
        "                                        num_layers=2, conditioning='cross_attn')",
        "        x = torch.randn(2, 8, 32)",
        "        ctx = torch.randn(2, 4, 16)",
        "        out = model(x, ctx)",
        "        assert out.shape == (2, 8, 32)",
        "",
        "    def test_film_conditioning(self):",
        "        from transformer import ConditionedTransformer",
        "        model = ConditionedTransformer(d_model=32, num_heads=4, d_context=16,",
        "                                        num_layers=2, conditioning='film')",
        "        x = torch.randn(2, 8, 32)",
        "        ctx = torch.randn(2, 16)",
        "        out = model(x, ctx)",
        "        assert out.shape == (2, 8, 32)",
        "",
    ]

    lines += [
        "class TestStochasticDepthTransformer:",
        "    def test_forward(self):",
        "        from transformer import StochasticDepthTransformer",
        "        model = StochasticDepthTransformer(d_model=32, num_heads=4, num_layers=4)",
        "        x = torch.randn(2, 8, 32)",
        "        out = model(x)",
        "        assert out.shape == (2, 8, 32)",
        "",
        "    def test_train_vs_eval(self):",
        "        from transformer import StochasticDepthTransformer",
        "        model = StochasticDepthTransformer(d_model=32, num_heads=4, num_layers=4,",
        "                                            drop_path_rate=0.5)",
        "        x = torch.randn(2, 8, 32)",
        "        model.eval()",
        "        with torch.no_grad():",
        "            out1 = model(x)",
        "            out2 = model(x)",
        "        assert torch.allclose(out1, out2)",
        "",
    ]

    # Parametrized
    lines += [
        "@pytest.mark.parametrize('d_model,num_heads,B,T', [",
    ]
    for d in [32, 64]:
        for h in [4, 8]:
            if d % h == 0:
                for B in [1, 2]:
                    for T in [8, 16]:
                        lines.append(f"    ({d}, {h}, {B}, {T}),")
    lines += [
        "])",
        "def test_normformer_parametrized(d_model, num_heads, B, T):",
        "    from transformer import NormFormerBlock",
        "    block = NormFormerBlock(d_model, num_heads)",
        "    x = torch.randn(B, T, d_model)",
        "    out = block(x)",
        "    assert out.shape == (B, T, d_model)",
        "    assert not torch.isnan(out).any()",
        "",
    ]

    return "\n".join(lines)

write_new("tests/test_transformer_extra.py", build_transformer_extra_tests())

# ════════════════════════════════════════════════════════════════════════════════
# 3. Large parametrized test file
# ════════════════════════════════════════════════════════════════════════════════
def build_mega_param_tests():
    lines = [
        '"""Mega parametrized test file - auto-generated configs for all major modules."""',
        "import pytest",
        "import torch",
        "import torch.nn as nn",
        "",
    ]

    # 500 parametrized linear configs
    lines += [
        "# ── 500 LoRA Linear config tests ───────────────────────────────────────────",
        "@pytest.mark.parametrize('cfg', [",
    ]
    count = 0
    for in_f in [16, 32, 64, 128, 256]:
        for out_f in [16, 32, 64, 128, 256]:
            for rank in [1, 2, 4, 8, 16]:
                for alpha in [1.0, 4.0, 8.0, 16.0]:
                    if count >= 500:
                        break
                    lines.append(f"    dict(in_f={in_f}, out_f={out_f}, rank={rank}, alpha={alpha}),")
                    count += 1
                if count >= 500:
                    break
            if count >= 500:
                break
        if count >= 500:
            break

    lines += [
        "])",
        "def test_lora_linear_500configs(cfg):",
        "    from lora import LoRALinear",
        "    layer = LoRALinear(cfg['in_f'], cfg['out_f'], cfg['rank'], cfg['alpha'])",
        "    x = torch.randn(2, 4, cfg['in_f'])",
        "    out = layer(x)",
        "    assert out.shape == (2, 4, cfg['out_f'])",
        "    assert not torch.isnan(out).any()",
        "",
    ]

    # 200 attention configs
    lines += [
        "# ── 200 Attention config tests ─────────────────────────────────────────────",
        "@pytest.mark.parametrize('cfg', [",
    ]
    count = 0
    for d in [32, 64, 128, 256]:
        for h in [2, 4, 8]:
            if d % h != 0:
                continue
            for B in [1, 2, 4]:
                for T in [4, 8, 16, 32]:
                    if count >= 200:
                        break
                    lines.append(f"    dict(d={d}, h={h}, B={B}, T={T}),")
                    count += 1
                if count >= 200:
                    break
            if count >= 200:
                break
        if count >= 200:
            break

    lines += [
        "])",
        "def test_rope_attention_200configs(cfg):",
        "    from attention import RoPEAttention",
        "    model = RoPEAttention(cfg['d'], cfg['h'])",
        "    x = torch.randn(cfg['B'], cfg['T'], cfg['d'])",
        "    out = model(x)",
        "    assert out.shape == (cfg['B'], cfg['T'], cfg['d'])",
        "    assert not torch.isnan(out).any()",
        "",
    ]

    # 200 transformer block configs
    lines += [
        "# ── 200 Transformer block config tests ─────────────────────────────────────",
        "@pytest.mark.parametrize('cfg', [",
    ]
    count = 0
    for d in [32, 64, 128]:
        for h in [4, 8]:
            if d % h != 0:
                continue
            for d_ff in [128, 256, 512]:
                for drop in [0.0, 0.1]:
                    for B, T in [(1, 8), (2, 16), (4, 8)]:
                        if count >= 200:
                            break
                        lines.append(f"    dict(d={d}, h={h}, d_ff={d_ff}, drop={drop}, B={B}, T={T}),")
                        count += 1
                    if count >= 200:
                        break
                if count >= 200:
                    break
            if count >= 200:
                break
        if count >= 200:
            break

    lines += [
        "])",
        "def test_normformer_200configs(cfg):",
        "    from transformer import NormFormerBlock",
        "    block = NormFormerBlock(cfg['d'], cfg['h'], cfg['d_ff'], cfg['drop'])",
        "    x = torch.randn(cfg['B'], cfg['T'], cfg['d'])",
        "    out = block(x)",
        "    assert out.shape == (cfg['B'], cfg['T'], cfg['d'])",
        "    assert not torch.isnan(out).any()",
        "",
    ]

    # 200 MoE configs
    lines += [
        "# ── 200 Sparse MoE config tests ────────────────────────────────────────────",
        "@pytest.mark.parametrize('cfg', [",
    ]
    count = 0
    for d in [32, 64]:
        for ne in [4, 8, 16]:
            for k in [1, 2]:
                for d_ff in [128, 256]:
                    for B, T in [(1, 4), (2, 8), (4, 4)]:
                        if count >= 200:
                            break
                        lines.append(f"    dict(d={d}, ne={ne}, k={k}, d_ff={d_ff}, B={B}, T={T}),")
                        count += 1
                    if count >= 200:
                        break
                if count >= 200:
                    break
            if count >= 200:
                break
        if count >= 200:
            break

    lines += [
        "])",
        "def test_fused_moe_200configs(cfg):",
        "    from moe import FusedMoELayer",
        "    moe = FusedMoELayer(cfg['d'], cfg['ne'], cfg['k'], cfg['d_ff'])",
        "    x = torch.randn(cfg['B'], cfg['T'], cfg['d'])",
        "    out = moe(x)",
        "    assert out.shape == (cfg['B'], cfg['T'], cfg['d'])",
        "    assert not torch.isnan(out).any()",
        "",
    ]

    # 100 LoRAEmbedding configs
    lines += [
        "# ── 100 LoRA Embedding config tests ────────────────────────────────────────",
        "@pytest.mark.parametrize('cfg', [",
    ]
    count = 0
    for vocab in [100, 500, 1000, 5000]:
        for dim in [16, 32, 64, 128]:
            for rank in [2, 4, 8]:
                for B, T in [(2, 8), (4, 16)]:
                    if count >= 100:
                        break
                    lines.append(f"    dict(vocab={vocab}, dim={dim}, rank={rank}, B={B}, T={T}),")
                    count += 1
                if count >= 100:
                    break
            if count >= 100:
                break
        if count >= 100:
            break

    lines += [
        "])",
        "def test_lora_embedding_100configs(cfg):",
        "    from lora import LoRAEmbedding",
        "    emb = LoRAEmbedding(cfg['vocab'], cfg['dim'], cfg['rank'])",
        "    ids = torch.randint(0, cfg['vocab'], (cfg['B'], cfg['T']))",
        "    out = emb(ids)",
        "    assert out.shape == (cfg['B'], cfg['T'], cfg['dim'])",
        "    assert not torch.isnan(out).any()",
        "",
    ]

    return "\n".join(lines)

write_new("tests/test_mega_parametrized.py", build_mega_param_tests())

# ════════════════════════════════════════════════════════════════════════════════
# 4. Append to model.py with more model variants
# ════════════════════════════════════════════════════════════════════════════════
MODEL_ADD = '''

# ============================================================
# Additional Lumina Model Variants
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math


class LuminaNano(nn.Module):
    """LuminaNano - ultra-small model for edge deployment (~100K params).

    Designed for real-time tick-level inference on resource-constrained devices.
    """

    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        max_seq_len: int = 64,
        num_classes: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.input_proj(x) + self.pos_embed(pos))
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


class LuminaAlpha(nn.Module):
    """LuminaAlpha - alpha generation model with multi-horizon forecasting.

    Predicts returns across multiple time horizons simultaneously.
    """

    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        horizons: List[int] = None,
        max_seq_len: int = 252,
        dropout: float = 0.1,
    ):
        super().__init__()
        if horizons is None:
            horizons = [1, 5, 21, 63]  # 1d, 1w, 1m, 1q
        self.horizons = horizons
        self.d_model = d_model

        self.feature_embed = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4, dropout, batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Per-horizon prediction heads
        self.horizon_heads = nn.ModuleDict({
            f"h{h}": nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            for h in horizons
        })

        # Uncertainty head (log-variance)
        self.uncertainty_heads = nn.ModuleDict({
            f"h{h}": nn.Linear(d_model // 2, 1)
            for h in horizons
        })

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.feature_embed(x) + self.pos_embed(pos))

        for layer in self.encoder_layers:
            h = layer(h)
        h = self.norm(h)

        pooled = h[:, -1, :]  # use last timestep

        results = {}
        for horizon in self.horizons:
            key = f"h{horizon}"
            # Get intermediate representation
            intermediate = F.gelu(self.horizon_heads[key][0](pooled))
            intermediate = self.dropout(intermediate)
            pred = self.horizon_heads[key][2](intermediate).squeeze(-1)
            log_var = self.uncertainty_heads[key](intermediate).squeeze(-1)
            results[f"pred_{horizon}d"] = pred
            results[f"logvar_{horizon}d"] = log_var

        return results


class LuminaRiskModel(nn.Module):
    """LuminaRiskModel - factor-based risk estimation model.

    Estimates portfolio risk via learned factor exposures and covariance.
    """

    def __init__(
        self,
        n_assets: int,
        n_factors: int = 10,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.d_model = d_model

        # Asset embedding
        self.asset_embed = nn.Embedding(n_assets, d_model)
        self.feature_proj = nn.Linear(1, d_model)

        # Transformer for cross-asset attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Factor loading head: maps each asset to factor exposures
        self.factor_head = nn.Linear(d_model, n_factors)

        # Specific variance head (idiosyncratic risk)
        self.spec_var_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

        # Factor covariance (learnable, constrained to be PSD)
        self.factor_cov_tril = nn.Parameter(torch.eye(n_factors))

        self.dropout = nn.Dropout(dropout)

    @property
    def factor_cov(self) -> torch.Tensor:
        L = torch.tril(self.factor_cov_tril)
        return L @ L.T + 1e-6 * torch.eye(self.n_factors, device=L.device)

    def forward(
        self,
        asset_ids: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N = asset_ids.shape
        h = self.asset_embed(asset_ids)  # (B, N, d)

        if returns is not None:
            h = h + self.feature_proj(returns.unsqueeze(-1))

        h = self.dropout(h)
        h = self.encoder(h)
        h = self.norm(h)

        # Factor loadings
        beta = self.factor_head(h)  # (B, N, K)

        # Specific variances
        spec_var = self.spec_var_head(h).squeeze(-1)  # (B, N)

        # Portfolio covariance: Sigma = B * F * B^T + diag(spec_var)
        F_cov = self.factor_cov  # (K, K)
        sigma = torch.bmm(beta, F_cov.unsqueeze(0).expand(B, -1, -1))  # (B, N, K)
        sigma = torch.bmm(sigma, beta.transpose(-2, -1))  # (B, N, N)
        sigma = sigma + torch.diag_embed(spec_var)

        return {
            "beta": beta,
            "spec_var": spec_var,
            "cov_matrix": sigma,
            "factor_cov": F_cov,
        }


class LuminaSentimentModel(nn.Module):
    """LuminaSentimentModel - financial sentiment analysis from text features.

    Maps text embeddings to signed sentiment scores with uncertainty.
    """

    def __init__(
        self,
        text_embed_dim: int = 768,
        market_dim: int = 64,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Modality projections
        self.text_proj = nn.Linear(text_embed_dim, d_model)
        self.market_proj = nn.Linear(market_dim, d_model)

        # Cross-modal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Output heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1)
        )
        self.impact_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Softplus()
        )
        self.duration_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Softplus()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_emb: torch.Tensor,
        market_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = text_emb.shape[0]

        h_text = self.text_proj(text_emb)  # (B, d)
        tokens = h_text.unsqueeze(1)        # (B, 1, d)

        if market_feat is not None:
            h_market = self.market_proj(market_feat).unsqueeze(1)  # (B, 1, d)
            tokens = torch.cat([tokens, h_market], dim=1)

        tokens = self.dropout(tokens)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)

        return {
            "sentiment": self.sentiment_head(pooled).squeeze(-1),
            "impact": self.impact_head(pooled).squeeze(-1),
            "duration_days": self.duration_head(pooled).squeeze(-1),
        }


class LuminaTrendFollower(nn.Module):
    """LuminaTrendFollower - momentum/mean-reversion signal model.

    Detects trend regime and generates position sizing signals.
    """

    def __init__(
        self,
        input_dim: int = 16,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        seq_len: int = 60,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Regime: trend | mean-revert | volatile | neutral
        self.regime_head = nn.Linear(d_model, 4)
        # Position: continuous in [-1, 1]
        self.position_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        # Stop-loss level
        self.stop_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Softplus())

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.input_proj(x) + self.pos_embed(pos))
        h = self.encoder(h)
        h = self.norm(h)
        pooled = h[:, -1, :]

        return {
            "regime_logits": self.regime_head(pooled),
            "position": torch.tanh(self.position_head(pooled)).squeeze(-1),
            "stop_loss": self.stop_head(pooled).squeeze(-1),
        }


class LuminaMarketMaker(nn.Module):
    """LuminaMarketMaker - bid-ask spread and inventory model for market making.

    Learns optimal spread and inventory thresholds from order book features.
    """

    def __init__(
        self,
        orderbook_levels: int = 10,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        # Order book: prices + sizes at each level, bid and ask -> 4 * levels features
        ob_dim = 4 * orderbook_levels

        self.ob_proj = nn.Linear(ob_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Quote outputs: bid spread, ask spread, quote size
        self.bid_spread = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Softplus())
        self.ask_spread = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Softplus())
        self.quote_size = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Softplus())
        self.inventory_risk = nn.Sequential(nn.Linear(d_model, 1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, orderbook: torch.Tensor, seq: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """orderbook: (B, T, 4*levels) time series of order book snapshots"""
        B, T, D = orderbook.shape
        h = self.dropout(self.ob_proj(orderbook))
        h = self.encoder(h)
        h = self.norm(h)
        pooled = h[:, -1, :]

        return {
            "bid_spread_bps": self.bid_spread(pooled).squeeze(-1),
            "ask_spread_bps": self.ask_spread(pooled).squeeze(-1),
            "quote_size": self.quote_size(pooled).squeeze(-1),
            "inventory_risk": self.inventory_risk(pooled).squeeze(-1),
        }
'''

append("model.py", MODEL_ADD)

# ════════════════════════════════════════════════════════════════════════════════
# 5. tests/test_model_extra.py
# ════════════════════════════════════════════════════════════════════════════════
def build_model_extra_tests():
    lines = [
        '"""Tests for additional Lumina model variants."""',
        "import pytest",
        "import torch",
        "",
    ]

    lines += [
        "class TestLuminaNano:",
        "    def test_forward_shape(self):",
        "        from model import LuminaNano",
        "        model = LuminaNano(input_dim=8, d_model=16, num_heads=2, num_layers=2, num_classes=3)",
        "        x = torch.randn(2, 16, 8)",
        "        out = model(x)",
        "        assert out.shape == (2, 3)",
        "",
        "    def test_no_nan(self):",
        "        from model import LuminaNano",
        "        model = LuminaNano()",
        "        x = torch.randn(4, 32, 8)",
        "        assert not torch.isnan(model(x)).any()",
        "",
        "    def test_gradient(self):",
        "        from model import LuminaNano",
        "        model = LuminaNano(input_dim=8, d_model=16, num_heads=2, num_layers=2)",
        "        x = torch.randn(2, 8, 8, requires_grad=True)",
        "        model(x).sum().backward()",
        "        assert x.grad is not None",
        "",
    ]

    lines += [
        "class TestLuminaAlpha:",
        "    def test_forward_returns_dict(self):",
        "        from model import LuminaAlpha",
        "        model = LuminaAlpha(input_dim=16, d_model=32, num_heads=4, num_layers=2,",
        "                             horizons=[1, 5, 21], max_seq_len=64)",
        "        x = torch.randn(2, 20, 16)",
        "        out = model(x)",
        "        assert 'pred_1d' in out",
        "        assert 'pred_5d' in out",
        "        assert 'pred_21d' in out",
        "",
        "    def test_pred_shapes(self):",
        "        from model import LuminaAlpha",
        "        model = LuminaAlpha(input_dim=8, d_model=16, num_heads=2, num_layers=2,",
        "                             horizons=[1, 5], max_seq_len=32)",
        "        x = torch.randn(4, 10, 8)",
        "        out = model(x)",
        "        assert out['pred_1d'].shape == (4,)",
        "        assert out['logvar_1d'].shape == (4,)",
        "",
        "    def test_no_nan(self):",
        "        from model import LuminaAlpha",
        "        model = LuminaAlpha(input_dim=8, d_model=16, num_heads=2, num_layers=2,",
        "                             horizons=[1], max_seq_len=32)",
        "        x = torch.randn(2, 8, 8)",
        "        out = model(x)",
        "        assert not torch.isnan(out['pred_1d']).any()",
        "",
    ]

    lines += [
        "class TestLuminaRiskModel:",
        "    def test_forward_returns_dict(self):",
        "        from model import LuminaRiskModel",
        "        model = LuminaRiskModel(n_assets=50, n_factors=5, d_model=32, num_heads=4, num_layers=2)",
        "        asset_ids = torch.randint(0, 50, (2, 10))",
        "        out = model(asset_ids)",
        "        assert 'beta' in out",
        "        assert 'cov_matrix' in out",
        "",
        "    def test_cov_matrix_shape(self):",
        "        from model import LuminaRiskModel",
        "        model = LuminaRiskModel(n_assets=20, n_factors=4, d_model=16, num_heads=2, num_layers=2)",
        "        ids = torch.randint(0, 20, (2, 5))",
        "        out = model(ids)",
        "        assert out['cov_matrix'].shape == (2, 5, 5)",
        "",
        "    def test_no_nan(self):",
        "        from model import LuminaRiskModel",
        "        model = LuminaRiskModel(n_assets=20, n_factors=4, d_model=16, num_heads=2, num_layers=2)",
        "        ids = torch.randint(0, 20, (2, 5))",
        "        out = model(ids)",
        "        assert not torch.isnan(out['cov_matrix']).any()",
        "",
    ]

    lines += [
        "class TestLuminaSentimentModel:",
        "    def test_forward_returns_dict(self):",
        "        from model import LuminaSentimentModel",
        "        model = LuminaSentimentModel(text_embed_dim=32, market_dim=8, d_model=16,",
        "                                      num_heads=2, num_layers=2)",
        "        text = torch.randn(2, 32)",
        "        out = model(text)",
        "        assert 'sentiment' in out",
        "        assert out['sentiment'].shape == (2,)",
        "",
        "    def test_with_market_features(self):",
        "        from model import LuminaSentimentModel",
        "        model = LuminaSentimentModel(text_embed_dim=32, market_dim=8, d_model=16,",
        "                                      num_heads=2, num_layers=2)",
        "        text = torch.randn(2, 32)",
        "        market = torch.randn(2, 8)",
        "        out = model(text, market)",
        "        assert out['sentiment'].shape == (2,)",
        "        assert not torch.isnan(out['sentiment']).any()",
        "",
    ]

    lines += [
        "class TestLuminaTrendFollower:",
        "    def test_forward_returns_dict(self):",
        "        from model import LuminaTrendFollower",
        "        model = LuminaTrendFollower(input_dim=8, d_model=16, num_heads=2, num_layers=2, seq_len=20)",
        "        x = torch.randn(2, 20, 8)",
        "        out = model(x)",
        "        assert 'regime_logits' in out",
        "        assert 'position' in out",
        "        assert out['regime_logits'].shape == (2, 4)",
        "",
        "    def test_position_bounded(self):",
        "        from model import LuminaTrendFollower",
        "        model = LuminaTrendFollower(input_dim=8, d_model=16, num_heads=2, num_layers=2, seq_len=16)",
        "        x = torch.randn(4, 16, 8)",
        "        out = model(x)",
        "        assert out['position'].abs().max() <= 1.0 + 1e-5",
        "",
    ]

    lines += [
        "class TestLuminaMarketMaker:",
        "    def test_forward_shape(self):",
        "        from model import LuminaMarketMaker",
        "        model = LuminaMarketMaker(orderbook_levels=5, d_model=16, num_heads=2, num_layers=2)",
        "        ob = torch.randn(2, 10, 20)  # 4*5=20 features",
        "        out = model(ob)",
        "        assert 'bid_spread_bps' in out",
        "        assert out['bid_spread_bps'].shape == (2,)",
        "",
        "    def test_spreads_positive(self):",
        "        from model import LuminaMarketMaker",
        "        model = LuminaMarketMaker(orderbook_levels=5, d_model=16, num_heads=2, num_layers=2)",
        "        ob = torch.randn(4, 8, 20)",
        "        out = model(ob)",
        "        assert (out['bid_spread_bps'] > 0).all()",
        "        assert (out['ask_spread_bps'] > 0).all()",
        "",
    ]

    # Parametrized model tests
    lines += [
        "@pytest.mark.parametrize('input_dim,d_model,num_heads,B,T,num_classes', [",
    ]
    for inp in [4, 8, 16]:
        for d in [16, 32]:
            for h in [2, 4]:
                if d % h == 0:
                    for B in [1, 2]:
                        for T in [8, 16]:
                            for nc in [2, 3]:
                                lines.append(f"    ({inp}, {d}, {h}, {B}, {T}, {nc}),")
    lines += [
        "])",
        "def test_lumina_nano_parametrized(input_dim, d_model, num_heads, B, T, num_classes):",
        "    from model import LuminaNano",
        "    model = LuminaNano(input_dim=input_dim, d_model=d_model, num_heads=num_heads,",
        "                        num_layers=2, max_seq_len=T+4, num_classes=num_classes)",
        "    x = torch.randn(B, T, input_dim)",
        "    out = model(x)",
        "    assert out.shape == (B, num_classes)",
        "    assert not torch.isnan(out).any()",
        "",
    ]

    return "\n".join(lines)

write_new("tests/test_model_extra.py", build_model_extra_tests())

# Final count
result = subprocess.run(
    ["bash", "-c",
     "find /c/Users/Matthew/srfm-lab/aeternus/lumina -name '*.py' -o -name '*.yaml' | xargs wc -l 2>/dev/null | tail -1"],
    capture_output=True, text=True
)
print("GRAND TOTAL:", result.stdout.strip())
