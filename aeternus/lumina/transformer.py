

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
            s2, b2 = params[:, 2*self.fc1.out_features:2*self.fc1.out_features+self.fc2.out_features],                      params[:, 2*self.fc1.out_features+self.fc2.out_features:]
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
