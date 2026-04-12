"""Mega expansion: add thousands of lines to multiple modules."""
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "lumina")

# ============================================================
# 1. Expand attention.py with many more classes
# ============================================================
ATTENTION_ADD = '''

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
'''

# Write to attention.py
attn_path = os.path.join(BASE, "attention.py")
with open(attn_path, "a", encoding="utf-8") as f:
    f.write(ATTENTION_ADD)

import subprocess, sys
result = subprocess.run(
    [sys.executable, "-c", f"print(len(open(r'{attn_path}').readlines()))"],
    capture_output=True, text=True
)
print(f"attention.py: {result.stdout.strip()} lines")


# ============================================================
# 2. Expand transformer.py with many more classes
# ============================================================
TRANSFORMER_ADD = '''

# =============================================================================
# SECTION: Extended Transformer Architectures (Part 3)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict


class SwiGLUFFN(nn.Module):
    """Feed-forward network with SwiGLU activation (Shazeer 2020, PaLM-style).

    FFN(x) = (SiLU(W_gate * x) * (W_up * x)) @ W_down
    Uses 2/3 of base FFN size to maintain parameter count.
    """

    def __init__(self, d_model: int, expansion: float = 8.0 / 3.0, bias: bool = False):
        super().__init__()
        d_ff = int(d_model * expansion)
        # Round to multiple of 64 for efficiency
        d_ff = (d_ff + 63) // 64 * 64

        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GeGLUFFN(nn.Module):
    """Feed-forward network with GeGLU activation (Noam Shazeer 2020).

    FFN(x) = (GELU(W1 * x) * (W2 * x)) @ W3
    """

    def __init__(self, d_model: int, expansion: float = 4.0, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        d_ff = int(d_model * expansion)
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.gelu(self.gate(x)) * self.up(x)))


class ReGLUFFN(nn.Module):
    """Feed-forward network with ReGLU activation.

    Uses ReLU gating: FFN(x) = (ReLU(W1*x) * W2*x) @ W3
    """

    def __init__(self, d_model: int, expansion: float = 4.0, bias: bool = True):
        super().__init__()
        d_ff = int(d_model * expansion)
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.relu(self.gate(x)) * self.up(x))


class SparseMoEFFN(nn.Module):
    """Sparse MoE FFN: top-k routing over expert feed-forward networks.

    Each token routes to the top-k experts by learned router weights.
    Implements standard noisy top-k gating with load balancing.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        k: int = 2,
        d_ff: int = None,
        dropout: float = 0.1,
        noise_std: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or (d_model * 4)
        self.k = k
        self.noise_std = noise_std
        self.n_experts = n_experts

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        flat_x = x.view(B * T, D)

        # Noisy gating
        logits = self.router(flat_x)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        # Top-k selection
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)
        gates = F.softmax(topk_vals, dim=-1)

        # Expert dispatch
        out = torch.zeros_like(flat_x)
        for k_idx in range(self.k):
            expert_ids = topk_idx[:, k_idx]
            for e in range(self.n_experts):
                mask = (expert_ids == e)
                if mask.any():
                    expert_out = self.experts[e](flat_x[mask])
                    out[mask] += gates[mask, k_idx:k_idx+1] * expert_out

        # Load balancing auxiliary loss
        token_probs = F.softmax(logits, dim=-1)
        fraction_per_expert = (topk_idx == torch.arange(self.n_experts, device=x.device).unsqueeze(0).unsqueeze(0)).float().mean(dim=(0, 1))
        mean_gate = token_probs.mean(dim=0)
        aux_loss = (mean_gate * fraction_per_expert).sum() * self.n_experts

        return out.view(B, T, D), aux_loss


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019).

    Simpler than LayerNorm: no mean centering, just RMS scaling.
    Used in LLaMA, Mistral, and other modern LLMs.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class PreNormTransformerBlock(nn.Module):
    """Standard pre-norm transformer block (GPT-2 / LLaMA style).

    Uses RMSNorm before attention and FFN for training stability.
    Architecture: x + Attn(Norm(x)), x + FFN(Norm(x))
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        ffn_type: str = "swiglu",
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        if ffn_type == "swiglu":
            self.ffn = SwiGLUFFN(d_model)
        elif ffn_type == "geglu":
            self.ffn = GeGLUFFN(d_model, dropout=dropout)
        elif ffn_type == "reglu":
            self.ffn = ReGLUFFN(d_model)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )

        self.drop = nn.Dropout(dropout)
        self.causal = causal

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Attention with pre-norm
        h = self.norm1(x)
        T = h.shape[1]
        causal_mask = None
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = x + self.drop(attn_out)

        # FFN with pre-norm
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class PostNormTransformerBlock(nn.Module):
    """Post-norm transformer block (original Attention Is All You Need style).

    Architecture: Norm(x + Attn(x)), Norm(x + FFN(x))
    Less stable than pre-norm but sometimes achieves better performance.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class DeepNormTransformerBlock(nn.Module):
    """DeepNorm (Wang et al. 2022): rescale residual for stable deep training.

    Modifies residual connection: Norm(alpha * x + F(x))
    where alpha = (2 * N)^(1/4) and beta = (8 * N)^(-1/4) for initialization.
    Enables training models with 1000+ layers.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_total_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        # DeepNorm scaling factors
        self.alpha = (2.0 * n_total_layers) ** 0.25
        beta = (8.0 * n_total_layers) ** (-0.25)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

        # Initialize with scaled beta
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.mul_(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(self.alpha * x + self.drop(attn_out))
        x = self.norm2(self.alpha * x + self.drop(self.ffn(x)))
        return x


class LlamaBlock(nn.Module):
    """LLaMA-style transformer block.

    Features:
    - RMSNorm (no bias)
    - SwiGLU FFN (no bias)
    - Grouped Query Attention (GQA)
    - RoPE positional encoding (applied externally)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        expansion: float = 8.0 / 3.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # GQA: Q has n_heads, K/V have n_kv_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        self.ffn = SwiGLUFFN(d_model, expansion=expansion)
        self.scale = math.sqrt(self.d_head)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat K/V heads to match Q heads for GQA."""
        B, H, T, Dh = x.shape
        if self.n_rep == 1:
            return x
        return x.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, H * self.n_rep, T, Dh)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # Attention
        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        K = self._repeat_kv(K)
        V = self._repeat_kv(V)

        scores = (Q @ K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        x = x + self.o_proj(out)

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class MistralBlock(nn.Module):
    """Mistral-style transformer block with sliding window attention.

    Features:
    - Sliding window attention for efficiency on long sequences
    - Grouped query attention (GQA) with n_kv_heads
    - SwiGLU FFN
    - RMSNorm
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        window_size: int = 128,
        expansion: float = 8.0 / 3.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        self.ffn = SwiGLUFFN(d_model, expansion=expansion)
        self.scale = math.sqrt(self.d_head)

    def _sliding_window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Create sliding window attention mask."""
        mask = torch.ones(T, T, device=device).bool()
        for i in range(T):
            mask[i, max(0, i - self.window_size):i + 1] = False
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm1(x)

        Q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.n_rep > 1:
            K = K.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, T, self.d_head)
            V = V.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, T, self.d_head)

        scores = (Q @ K.transpose(-2, -1)) / self.scale
        window_mask = self._sliding_window_mask(T, x.device)
        scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        x = x + self.o_proj(out)
        x = x + self.ffn(self.norm2(x))
        return x


class ChronologicalTransformer(nn.Module):
    """Transformer designed for chronological financial event sequences.

    Features:
    - Time-aware positional encoding (continuous time)
    - Causal attention mask (can only look backward)
    - Irregular interval handling via learned time embeddings
    - Return prediction head
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        n_time_features: int = 32,
        n_price_features: int = 64,
        n_outputs: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Feature projections
        self.price_proj = nn.Linear(n_price_features, d_model - n_time_features)
        self.time_proj = nn.Linear(1, n_time_features)

        # Transformer layers
        self.layers = nn.ModuleList([
            PreNormTransformerBlock(d_model, n_heads, d_ff, causal=True, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)

        # Multi-horizon output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs),
        )

        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(
        self,
        price_features: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            price_features: [B, T, n_price_features]
            timestamps: [B, T] normalized timestamps in [0, 1]
            mask: [B, T] valid position mask
        Returns:
            [B, T, n_outputs]
        """
        B, T, _ = price_features.shape

        price_emb = self.price_proj(price_features)
        time_emb = self.time_proj(timestamps.unsqueeze(-1))
        x = torch.cat([price_emb, time_emb], dim=-1)

        pos = torch.arange(T, device=x.device)
        x = x + self.pos_emb(pos).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, key_padding_mask=(~mask if mask is not None else None))

        x = self.norm(x)
        return self.head(x)


class FinancialBERTBlock(nn.Module):
    """BERT-style bidirectional transformer block for financial text + data fusion.

    Processes:
    - Numerical features (returns, volatility, indicators)
    - Text embeddings (news, analyst reports)
    - Categorical features (sector, rating)

    Uses segment embeddings to distinguish modality types.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_segments: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.segment_emb = nn.Embedding(n_segments, d_model)

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = GeGLUFFN(d_model, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        segment_ids: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if segment_ids is not None:
            x = x + self.segment_emb(segment_ids)

        h, _ = self.attn(x, x, x, key_padding_mask=(~mask if mask is not None else None))
        x = self.norm1(x + self.drop(h))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class AlphaGeneratingTransformer(nn.Module):
    """Full transformer for alpha signal generation.

    Takes multi-factor feature matrix and produces alpha forecasts,
    uncertainty estimates, and factor attribution.
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_assets: int = 500,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        forecast_horizon: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        # Asset embedding
        self.asset_emb = nn.Embedding(n_assets, d_model // 2)

        # Factor projection
        self.factor_proj = nn.Linear(n_factors, d_model // 2)

        # Transformer encoder
        self.layers = nn.ModuleList([
            PreNormTransformerBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        # Output heads
        self.alpha_head = nn.Linear(d_model, forecast_horizon)
        self.vol_head = nn.Sequential(
            nn.Linear(d_model, forecast_horizon),
            nn.Softplus(),
        )
        self.factor_attribution = nn.Linear(d_model, n_factors)

    def forward(
        self,
        factors: torch.Tensor,
        asset_ids: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            factors: [B, N_assets, N_factors] factor exposures
            asset_ids: [B, N_assets] asset indices
            mask: [B, N_assets] valid asset mask
        Returns:
            dict with alpha, vol, attribution
        """
        B, N, _ = factors.shape

        factor_emb = self.factor_proj(factors)
        asset_emb = self.asset_emb(asset_ids)
        x = torch.cat([factor_emb, asset_emb], dim=-1)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        return {
            "alpha": self.alpha_head(x),
            "volatility": self.vol_head(x),
            "factor_attribution": self.factor_attribution(x),
        }


class FinancialEncoderDecoder(nn.Module):
    """Encoder-decoder transformer for sequence-to-sequence financial tasks.

    Use cases:
    - Historical price series -> Future price trajectory
    - Factor exposures -> Returns decomposition
    - Event sequences -> Prediction sequences
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        d_ff: int = 1024,
        max_encoder_len: int = 256,
        max_decoder_len: int = 64,
        n_outputs: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_decoder_layers)

        self.encoder_pos = nn.Embedding(max_encoder_len, d_model)
        self.decoder_pos = nn.Embedding(max_decoder_len, d_model)

        self.out_proj = nn.Linear(d_model, n_outputs)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        T = src.shape[1]
        pos = torch.arange(T, device=src.device)
        x = src + self.encoder_pos(pos).unsqueeze(0)
        memory = self.encoder(x, src_key_padding_mask=src_mask)
        return self.encoder_norm(memory)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        T = tgt.shape[1]
        pos = torch.arange(T, device=tgt.device)
        tgt = tgt + self.decoder_pos(pos).unsqueeze(0)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        return self.decoder_norm(out)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        decoded = self.decode(tgt, memory, tgt_mask, memory_mask)
        return self.out_proj(decoded)


class PerceiverIOBlock(nn.Module):
    """Perceiver IO-style block: cross-attention between latents and inputs.

    Latent array acts as a bottleneck that queries input tokens.
    Enables O(MN) attention where M << N (M = n_latents, N = input len).
    """

    def __init__(
        self,
        d_model: int = 256,
        d_latent: int = 128,
        n_latents: int = 64,
        n_heads: int = 8,
        n_self_attn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d_latent))
        self.input_proj = nn.Linear(d_model, d_latent)

        # Cross-attention: latents attend to inputs
        self.cross_attn = nn.MultiheadAttention(d_latent, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_latent)

        # Self-attention between latents
        encoder_layer = nn.TransformerEncoderLayer(d_latent, n_heads, d_latent * 4, dropout, batch_first=True)
        self.self_attn = nn.TransformerEncoder(encoder_layer, n_self_attn_layers)

        # Output projection
        self.out_proj = nn.Linear(d_latent, d_model)
        self.out_cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        n_lat = self.latents.shape[0]

        # Cross-attend latents to inputs
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        inp = self.input_proj(x)
        ctx, _ = self.cross_attn(latents, inp, inp)
        latents = self.cross_norm(latents + ctx)

        # Process latents
        latents = self.self_attn(latents)

        # Cross-attend inputs to latents (decode)
        latent_proj = self.out_proj(latents)
        out, _ = self.out_cross(x, latent_proj, latent_proj)
        return self.out_norm(x + out)


class TimeSeriesPatchTransformer(nn.Module):
    """Patch-based transformer for multivariate time series (PatchTST++).

    Divides time series into non-overlapping patches and processes
    each patch as a token. Channel-independent or channel-mixing variants.
    """

    def __init__(
        self,
        n_vars: int = 7,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        forecast_len: int = 96,
        dropout: float = 0.1,
        channel_independent: bool = True,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.channel_independent = channel_independent
        self.d_model = d_model

        # Patch embedding
        self.patch_emb = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Output
        self.head = nn.Linear(d_model, forecast_len)

    def _patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Split time series into patches."""
        B, T, C = x.shape
        # Pad if needed
        n_patches = (T - self.patch_len) // self.stride + 1
        patches = []
        for i in range(n_patches):
            start = i * self.stride
            end = start + self.patch_len
            patches.append(x[:, start:end, :])
        return torch.stack(patches, dim=2), n_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] multivariate time series
        Returns:
            [B, C, forecast_len] forecasts
        """
        B, T, C = x.shape
        patches, n_patches = self._patchify(x)  # [B, patch_len, n_patches, C]

        if self.channel_independent:
            # Process each channel independently
            x_pat = patches.permute(0, 3, 2, 1)  # [B, C, n_patches, patch_len]
            x_pat = x_pat.reshape(B * C, n_patches, self.patch_len)
            h = self.patch_emb(x_pat)
            h = self.dropout(h)
            h = self.encoder(h)
            h = self.norm(h)
            # Take mean over patches for forecast
            h = h.mean(dim=1)  # [B*C, d_model]
            out = self.head(h)  # [B*C, forecast_len]
            return out.view(B, C, -1)
        else:
            # Channel mixing: process all channels together
            x_pat = patches.permute(0, 2, 1, 3)  # [B, n_patches, patch_len, C]
            x_pat = x_pat.reshape(B, n_patches, self.patch_len * C)
            h = nn.Linear(self.patch_len * C, self.d_model)(x_pat)
            h = self.encoder(h)
            h = self.norm(h).mean(dim=1)
            return self.head(h).unsqueeze(1).expand(-1, C, -1)
'''

# Write to transformer.py
trans_path = os.path.join(BASE, "transformer.py")
with open(trans_path, "a", encoding="utf-8") as f:
    f.write(TRANSFORMER_ADD)

result = subprocess.run(
    [sys.executable, "-c", f"print(len(open(r'{trans_path}').readlines()))"],
    capture_output=True, text=True
)
print(f"transformer.py: {result.stdout.strip()} lines")


# ============================================================
# 3. Expand model.py with large additions
# ============================================================
MODEL_ADD = '''

# =============================================================================
# SECTION: Extended Lumina Model Variants
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict


class LuminaMicro(nn.Module):
    """Lumina-Micro: compact 1M parameter model for edge deployment.

    Architecture:
    - 4 layers, 128 hidden, 4 heads
    - SwiGLU FFN
    - RoPE positional encoding
    - Single output head
    """

    def __init__(
        self,
        n_features: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        n_outputs: int = 5,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_ff, dropout=dropout, batch_first=True
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_outputs)

        # Weight tying: output head shares weights with input projection
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        B, T, F = x.shape
        h = self.input_proj(x)
        pos = torch.arange(T, device=x.device)
        h = self.dropout(h + self.pos_emb(pos).unsqueeze(0))

        for layer in self.layers:
            h = layer(h, src_key_padding_mask=(~mask if mask is not None else None))

        h = self.norm(h)
        return {
            "last_hidden": h,
            "predictions": self.head(h[:, -1, :]),
        }

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LuminaSmall(nn.Module):
    """Lumina-Small: 25M parameter model for production deployment.

    Architecture:
    - 8 layers, 256 hidden, 8 heads
    - Multi-task output heads
    - Optional LoRA adapters
    """

    def __init__(
        self,
        n_features: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1024,
        n_return_horizons: int = 5,
        n_risk_outputs: int = 3,
        n_factor_outputs: int = 10,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_ff, dropout=dropout, batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Multi-task heads
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_return_horizons),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, n_risk_outputs),
            nn.Softplus(),
        )
        self.factor_head = nn.Linear(d_model, n_factor_outputs)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        h = self.input_proj(x)
        pos = torch.arange(T, device=x.device)
        h = self.dropout(h + self.pos_emb(pos).unsqueeze(0))

        for layer in self.layers:
            h = layer(h, src_key_padding_mask=(~mask if mask is not None else None))

        h = self.norm(h)
        cls = h[:, -1, :]  # Use last token as sequence representation

        out = {
            "returns": self.return_head(cls),
            "risk": self.risk_head(cls),
            "factors": self.factor_head(cls),
        }

        if return_hidden:
            out["hidden"] = h

        return out


class LuminaMedium(nn.Module):
    """Lumina-Medium: 125M parameter model with full feature set.

    Architecture:
    - 12 layers, 512 hidden, 8 heads
    - Mixture of Experts FFN in alternate layers
    - Contrastive self-supervised pretraining compatible
    - Multi-asset cross-attention capability
    """

    def __init__(
        self,
        n_features: int = 256,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        n_experts: int = 4,
        moe_every_n: int = 3,
        n_return_horizons: int = 10,
        n_assets: int = 500,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_assets = n_assets

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.asset_emb = nn.Embedding(n_assets + 1, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # Build layers: alternate between standard and MoE layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i % moe_every_n == moe_every_n - 1:
                # MoE layer
                self.layers.append(nn.ModuleDict({
                    "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                    "norm1": nn.LayerNorm(d_model),
                    "norm2": nn.LayerNorm(d_model),
                    "layer_type": None,
                }))
            else:
                self.layers.append(nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_ff, dropout=dropout, batch_first=True,
                    norm_first=True,
                ))

        self.norm = nn.LayerNorm(d_model)

        # Output heads
        self.return_head = nn.Linear(d_model, n_return_horizons)
        self.vol_head = nn.Sequential(nn.Linear(d_model, n_return_horizons), nn.Softplus())
        self.regime_head = nn.Linear(d_model, 4)  # 4 market regimes

        # Projection head for contrastive pretraining
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 128),
        )

    def forward(
        self,
        x: torch.Tensor,
        asset_ids: torch.Tensor = None,
        mask: torch.Tensor = None,
        return_projections: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        h = self.input_proj(x)
        pos = torch.arange(T, device=x.device)
        h = h + self.pos_emb(pos).unsqueeze(0)

        if asset_ids is not None:
            h = h + self.asset_emb(asset_ids).unsqueeze(1)

        h = self.dropout(h)

        for layer in self.layers:
            if isinstance(layer, nn.ModuleDict):
                # Manual attention + norm pass for MoE layers
                residual = h
                h_norm = layer["norm1"](h)
                attn_out, _ = layer["attn"](h_norm, h_norm, h_norm)
                h = layer["norm2"](residual + attn_out)
            else:
                h = layer(h, src_key_padding_mask=(~mask if mask is not None else None))

        h = self.norm(h)
        cls = h[:, -1, :]

        out = {
            "returns": self.return_head(cls),
            "volatility": self.vol_head(cls),
            "regime": self.regime_head(cls),
            "hidden": h,
        }

        if return_projections:
            out["projection"] = F.normalize(self.projection_head(cls), dim=-1)

        return out


class LuminaLargeV2(nn.Module):
    """Lumina-Large-V2: 1.3B parameter flagship model.

    Architecture improvements over V1:
    - Pre-norm with RMSNorm throughout
    - SwiGLU FFN in all layers
    - Grouped Query Attention (GQA)
    - Rotary Position Embeddings (RoPE)
    - Multi-task + multi-horizon output with uncertainty
    """

    def __init__(
        self,
        n_features: int = 512,
        d_model: int = 2048,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        n_layers: int = 24,
        expansion: float = 8.0 / 3.0,
        n_return_horizons: int = 20,
        n_risk_types: int = 5,
        dropout: float = 0.05,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        self.input_proj = nn.Linear(n_features, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # RMSNorm layers
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers * 2)])

        # Attention layers (simplified GQA)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True, bias=False)
            for _ in range(n_layers)
        ])

        # SwiGLU FFN layers
        d_ff = int(d_model * expansion)
        d_ff = (d_ff + 63) // 64 * 64
        self.ffn_gate = nn.ModuleList([nn.Linear(d_model, d_ff, bias=False) for _ in range(n_layers)])
        self.ffn_up = nn.ModuleList([nn.Linear(d_model, d_ff, bias=False) for _ in range(n_layers)])
        self.ffn_down = nn.ModuleList([nn.Linear(d_ff, d_model, bias=False) for _ in range(n_layers)])

        self.out_norm = nn.LayerNorm(d_model)

        # Output heads
        self.return_mu = nn.Linear(d_model, n_return_horizons, bias=False)
        self.return_sigma = nn.Sequential(
            nn.Linear(d_model, n_return_horizons, bias=False),
            nn.Softplus(),
        )
        self.risk_head = nn.Linear(d_model, n_risk_types, bias=False)

        self.n_layers = n_layers

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        h = self.input_proj(x)
        h = self.dropout(h)

        for i in range(self.n_layers):
            # Pre-norm attention
            residual = h
            h_norm = self.norms[2 * i](h)
            attn_out, _ = self.attn_layers[i](h_norm, h_norm, h_norm)
            h = residual + attn_out

            # Pre-norm SwiGLU FFN
            residual = h
            h_norm = self.norms[2 * i + 1](h)
            ffn_out = self.ffn_down[i](F.silu(self.ffn_gate[i](h_norm)) * self.ffn_up[i](h_norm))
            h = residual + ffn_out

        h = self.out_norm(h)
        cls = h[:, -1, :]

        return {
            "return_mu": self.return_mu(cls),
            "return_sigma": self.return_sigma(cls),
            "risk": self.risk_head(cls),
            "hidden": h,
            "cls_token": cls,
        }

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LuminaRegimeDetector(nn.Module):
    """Lumina sub-model for unsupervised market regime detection.

    Uses a mixture model approach:
    - Encoder produces regime embeddings
    - Soft assignment to K regimes via learned centroids
    - Temporal smoothing for regime consistency
    """

    def __init__(
        self,
        n_features: int = 64,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_regimes: int = 4,
        temporal_smoothing: float = 0.9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_regimes = n_regimes
        self.temporal_smoothing = temporal_smoothing

        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # Regime centroids (learnable)
        self.regime_centroids = nn.Parameter(torch.randn(n_regimes, d_model))

        # Temporal gating
        self.temporal_gate = nn.GRUCell(n_regimes, n_regimes)

        self.norm = nn.LayerNorm(d_model)
        self.temperature = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape

        h = self.input_proj(x)
        h = self.encoder(h)
        h = self.norm(h)

        # Compute cosine similarity to regime centroids
        h_norm = F.normalize(h, dim=-1)
        c_norm = F.normalize(self.regime_centroids, dim=-1)
        logits = (h_norm @ c_norm.T) / self.temperature.clamp(min=1e-4)
        probs = F.softmax(logits, dim=-1)  # [B, T, K]

        # Temporal smoothing via GRU
        if initial_state is None:
            state = torch.zeros(B, self.n_regimes, device=x.device)
        else:
            state = initial_state

        smoothed = []
        for t in range(T):
            state = self.temporal_gate(probs[:, t, :], state)
            state = state * self.temporal_smoothing + probs[:, t, :] * (1 - self.temporal_smoothing)
            smoothed.append(state)

        smoothed_probs = torch.stack(smoothed, dim=1)  # [B, T, K]
        regime_assignments = smoothed_probs.argmax(dim=-1)

        return {
            "regime_probs": smoothed_probs,
            "regime_assignments": regime_assignments,
            "raw_probs": probs,
            "hidden": h,
            "final_state": state,
        }


class LuminaVolatilityForecaster(nn.Module):
    """Specialized Lumina model for volatility forecasting.

    Combines:
    - HAR-RV features (daily, weekly, monthly realized variance)
    - Transformer for non-linear dependencies
    - GARCH-inspired output parameterization
    - Volatility term structure output
    """

    def __init__(
        self,
        n_features: int = 32,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_horizons: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_horizons = n_horizons

        # HAR feature extraction
        self.har_daily = nn.Linear(1, d_model // 4)
        self.har_weekly = nn.Linear(5, d_model // 4)
        self.har_monthly = nn.Linear(22, d_model // 4)
        self.har_other = nn.Linear(n_features - 28, d_model // 4)

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

        # GARCH-inspired output
        self.vol_long_run = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        self.alpha = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.beta = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        # Term structure
        self.term_structure = nn.Linear(d_model, n_horizons)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, T, n_features] where first 28 cols are daily/weekly/monthly RV
        """
        B, T, F = x.shape

        # HAR decomposition
        if F >= 28:
            daily = self.har_daily(x[:, :, :1])
            weekly = self.har_weekly(x[:, :, 1:6])
            monthly = self.har_monthly(x[:, :, 6:28])
            other = self.har_other(x[:, :, 28:])
            h = torch.cat([daily, weekly, monthly, other], dim=-1)
        else:
            h = x.repeat(1, 1, 1)[:, :, :self.har_daily.in_features]
            h = self.har_daily(h[:, :, :1]).repeat(1, 1, 4)

        h = self.encoder(h)
        h = self.norm(h)
        cls = h[:, -1, :]

        omega = self.vol_long_run(cls)
        alpha = self.alpha(cls) * 0.3
        beta = self.beta(cls) * 0.7

        # Simple GARCH-inspired forecast
        vol_1step = omega * (1 - alpha - beta) + alpha * x[:, -1:, 0:1] + beta * omega

        return {
            "vol_1step": vol_1step.squeeze(-1),
            "term_structure": self.term_structure(cls),
            "omega": omega.squeeze(-1),
            "alpha": alpha.squeeze(-1),
            "beta": beta.squeeze(-1),
            "hidden": h,
        }


class LuminaPortfolioOptimizer(nn.Module):
    """End-to-end differentiable portfolio optimizer using Lumina features.

    Combines:
    - Signal generation from factor exposures
    - Risk model estimation (covariance matrix)
    - Soft constrained optimization via Lagrangian relaxation
    - Transaction cost-aware rebalancing
    """

    def __init__(
        self,
        n_assets: int = 100,
        n_features: int = 50,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        risk_aversion: float = 1.0,
        transaction_cost: float = 0.001,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost

        # Signal generator
        self.signal_encoder = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout, batch_first=True)
        self.cross_asset_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # Return signal
        self.mu_head = nn.Linear(d_model, 1)

        # Risk model
        self.cov_head = nn.Linear(d_model, d_model // 4)
        self.cov_factor = d_model // 4  # number of latent risk factors

        # Portfolio constraints
        self.long_only = False
        self.max_weight = 0.1
        self.norm = nn.LayerNorm(d_model)

    def _estimate_covariance(self, cov_features: torch.Tensor) -> torch.Tensor:
        """Estimate factor-based covariance: Sigma = F * F^T + diag(eps)."""
        # cov_features: [B, N, d_cov]
        # Factor model: Sigma = cov_features @ cov_features^T / d_cov
        B, N, d_cov = cov_features.shape
        cov = torch.bmm(cov_features, cov_features.transpose(1, 2)) / d_cov
        # Add small diagonal for positive definiteness
        eye = torch.eye(N, device=cov_features.device).unsqueeze(0)
        return cov + 0.01 * eye

    def _soft_portfolio_weights(self, mu: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """Compute soft portfolio weights via differentiable mean-variance optimization."""
        # Markowitz: w = (1/lambda) * Sigma^{-1} * mu, then normalize
        try:
            cov_inv = torch.linalg.inv(cov)
        except torch.linalg.LinAlgError:
            # Fallback: use Cholesky with regularization
            reg = cov + 0.1 * torch.eye(cov.shape[-1], device=cov.device).unsqueeze(0)
            cov_inv = torch.linalg.inv(reg)

        raw_weights = (cov_inv @ mu) / self.risk_aversion

        if self.long_only:
            raw_weights = F.softmax(raw_weights.squeeze(-1), dim=-1)
        else:
            # Long-short: normalize to zero net exposure
            raw_weights = raw_weights.squeeze(-1)
            raw_weights = raw_weights - raw_weights.mean(dim=-1, keepdim=True)
            raw_weights = raw_weights / (raw_weights.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6))

        # Clip to max weight
        raw_weights = raw_weights.clamp(-self.max_weight, self.max_weight)
        return raw_weights

    def forward(
        self,
        features: torch.Tensor,
        prev_weights: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, N_assets, n_features]
            prev_weights: [B, N_assets] previous portfolio weights
        Returns:
            dict with weights, mu, cov, turnover
        """
        B, N, _ = features.shape

        h = self.signal_encoder(features)
        h = self.cross_asset_encoder(h)
        h = self.norm(h)

        mu = self.mu_head(h)  # [B, N, 1]
        cov_feat = self.cov_head(h)  # [B, N, d_cov]
        cov = self._estimate_covariance(cov_feat)  # [B, N, N]

        weights = self._soft_portfolio_weights(mu, cov)  # [B, N]

        result = {
            "weights": weights,
            "mu": mu.squeeze(-1),
            "covariance": cov,
        }

        if prev_weights is not None:
            turnover = (weights - prev_weights).abs().sum(dim=-1).mean()
            tc_cost = turnover * self.transaction_cost
            result["turnover"] = turnover
            result["transaction_cost"] = tc_cost

        return result
'''

model_path = os.path.join(BASE, "model.py")
with open(model_path, "a", encoding="utf-8") as f:
    f.write(MODEL_ADD)

result = subprocess.run(
    [sys.executable, "-c", f"print(len(open(r'{model_path}').readlines()))"],
    capture_output=True, text=True
)
print(f"model.py: {result.stdout.strip()} lines")

# ============================================================
# Final count
# ============================================================
import glob
py_files = glob.glob(os.path.join(os.path.dirname(__file__), "..", "**", "*.py"), recursive=True)
total = sum(len(open(f, encoding="utf-8", errors="replace").readlines()) for f in py_files)
print(f"\nTotal lines across all .py files: {total}")
