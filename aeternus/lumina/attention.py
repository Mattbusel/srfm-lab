

# ============================================================
# Additional Attention Mechanisms (Extended)
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CosineAttention(nn.Module):
    """Cosine attention (Chen et al. 2021): uses cosine similarity instead of dot product.

    Normalizes queries and keys, making attention scale-invariant.
    """

    def __init__(self, d_model: int, num_heads: int, tau_init: float = 20.0, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable temperature per head
        self.tau = nn.Parameter(torch.full((num_heads, 1, 1), tau_init))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        # L2 normalize along head_dim
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Cosine similarity scaled by tau
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.tau  # (B, H, T, T)

        if mask is not None:
            attn = attn + mask

        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class RetentiveAttention(nn.Module):
    """RetNet retention mechanism (Sun et al. 2023) - parallel chunk mode.

    Replaces softmax attention with a decaying exponential retention matrix.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.group_norm = nn.GroupNorm(num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

        # Retention decay rates per head
        gamma = 1 - 2 ** (-5 - torch.arange(num_heads).float())
        self.register_buffer("gamma", gamma)

    def _build_decay_matrix(self, T: int, device) -> torch.Tensor:
        """Build T×T decay matrix D where D[i,j] = gamma^(i-j) if i>=j else 0."""
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        # (H, T, T)
        exponent = (i - j).clamp(min=0).unsqueeze(0).float()  # (1, T, T)
        gamma = self.gamma.unsqueeze(-1).unsqueeze(-1)          # (H, 1, 1)
        D = (gamma ** exponent) * (i >= j).float().unsqueeze(0)
        return D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).permute(0, 2, 1, 3)  # (B,H,T,D)
        k = self.k_proj(x).reshape(B, T, H, D).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, H, D).permute(0, 2, 1, 3)

        decay = self._build_decay_matrix(T, x.device)  # (H, T, T)

        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B,H,T,T)
        retention = qk * decay.unsqueeze(0)  # (B,H,T,T)
        retention = self.dropout(retention)
        out = torch.matmul(retention, v)  # (B,H,T,D)

        out = out.permute(0, 2, 1, 3).reshape(B, T, self.d_model)
        # Group norm for stabilization
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(out)


class KVCacheMultiHeadAttention(nn.Module):
    """Multi-head attention with explicit KV-cache for autoregressive decoding."""

    def __init__(self, d_model: int, num_heads: int, max_cache_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_cache_len = max_cache_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.head_dim)

        # KV cache (not a parameter)
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None
        self._cache_pos = 0

    def reset_cache(self):
        self._cache_k = None
        self._cache_v = None
        self._cache_pos = 0

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        if use_cache:
            if self._cache_k is None:
                self._cache_k = k
                self._cache_v = v
            else:
                self._cache_k = torch.cat([self._cache_k, k], dim=2)
                self._cache_v = torch.cat([self._cache_v, v], dim=2)
                # Trim to max_cache_len
                if self._cache_k.shape[2] > self.max_cache_len:
                    self._cache_k = self._cache_k[:, :, -self.max_cache_len:]
                    self._cache_v = self._cache_v[:, :, -self.max_cache_len:]
            k_eff = self._cache_k
            v_eff = self._cache_v
        else:
            k_eff = k
            v_eff = v

        attn = torch.matmul(q, k_eff.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_eff).transpose(1, 2).reshape(B, T, self.d_model)
        out = self.out_proj(out)

        past_kv = (self._cache_k, self._cache_v) if use_cache else None
        return out, past_kv


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (Shazeer 2019): single shared K and V, multiple Q heads.

    Reduces KV-cache memory by (num_heads) factor during inference.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        # Single K and V heads (not per-query-head)
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).unsqueeze(1)  # (B, 1, T, D) shared across heads
        v = self.v_proj(x).unsqueeze(1)  # (B, 1, T, D) shared across heads

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        if mask is not None:
            attn = attn + mask
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class ALiBiAttention(nn.Module):
    """ALiBi (Press et al. 2022): attention with linear biases instead of position embeddings.

    Adds a fixed (non-learned) bias proportional to query-key distance.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # ALiBi slopes: 2^(-8/H * h) for h=1..H
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes)

    @staticmethod
    def _get_slopes(n_heads: int) -> torch.Tensor:
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(n_heads))
        else:
            closest = 2 ** math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest)
            slopes += get_slopes_power_of_2(2 * closest)[0::2][:n_heads - closest]
            return torch.tensor(slopes[:n_heads])

    def _build_alibi_bias(self, T: int, device) -> torch.Tensor:
        """Returns (H, T, T) ALiBi bias matrix."""
        positions = torch.arange(T, device=device)
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        dist = -dist.abs().float()
        bias = self.slopes.unsqueeze(-1).unsqueeze(-1) * dist.unsqueeze(0)  # (H, T, T)
        return bias

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        alibi = self._build_alibi_bias(T, x.device).unsqueeze(0)    # (1, H, T, T)
        attn = attn + alibi

        if mask is not None:
            attn = attn + mask
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class RotaryPositionEmbedding(nn.Module):
    """RoPE (Su et al. 2021): rotary position embeddings applied to Q and K."""

    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _get_sin_cos(self, T: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(T, device=device).float()
        freqs = torch.outer(t, self.inv_freq)  # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
        return emb.sin(), emb.cos()

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        return x * cos + self.rotate_half(x) * sin

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[-2]
        sin, cos = self._get_sin_cos(T, q.device)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
        cos = cos.unsqueeze(0).unsqueeze(0)
        return self.apply_rotary(q, sin, cos), self.apply_rotary(k, sin, cos)


class RoPEAttention(nn.Module):
    """Multi-head attention with rotary position embeddings (RoPE)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, rope_base: int = 10000):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        self.rope = RotaryPositionEmbedding(self.head_dim, base=rope_base)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn + mask
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class FlashAttentionSimulator(nn.Module):
    """Simulates FlashAttention (Dao et al. 2022) interface - tiled computation.

    For testing purposes only (not the actual CUDA kernel).
    Computes exact attention but in tiles to simulate the memory access pattern.
    """

    def __init__(self, d_model: int, num_heads: int, block_size: int = 32, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.head_dim)

    def _tiled_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes attention in tiles (equivalent to standard attention)."""
        B, H, T, D = q.shape
        Bc = self.block_size
        out = torch.zeros_like(q)
        row_max = torch.full((B, H, T, 1), float("-inf"), device=q.device)
        row_sum = torch.zeros(B, H, T, 1, device=q.device)

        for j_start in range(0, T, Bc):
            j_end = min(j_start + Bc, T)
            k_block = k[:, :, j_start:j_end, :]  # (B, H, Bc, D)
            v_block = v[:, :, j_start:j_end, :]

            s = torch.matmul(q, k_block.transpose(-2, -1)) / self.scale  # (B, H, T, Bc)
            m_new = torch.maximum(row_max, s.max(dim=-1, keepdim=True).values)
            exp_s = torch.exp(s - m_new)

            out = out * torch.exp(row_max - m_new) + torch.matmul(exp_s, v_block)
            row_sum = row_sum * torch.exp(row_max - m_new) + exp_s.sum(dim=-1, keepdim=True)
            row_max = m_new

        out = out / (row_sum + 1e-8)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        out = self._tiled_attention(q, k, v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class XFormersStyleAttention(nn.Module):
    """xFormers-style efficient attention with memory_efficient_attention fallback."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, D)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Try scaled_dot_product_attention if available (PyTorch 2.0+)
        try:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        except AttributeError:
            attn = (q @ k.transpose(-2, -1)) / self.scale
            if mask is not None:
                attn = attn + mask
            attn = F.softmax(attn, dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)
