"""Script to append large attention module expansions."""
import os

ATTENTION_PATH = r"C:\Users\Matthew\srfm-lab\aeternus\lumina\lumina\attention.py"

CONTENT = r'''

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
'''

with open(ATTENTION_PATH, "a", encoding="utf-8") as f:
    f.write(CONTENT)

import subprocess
r = subprocess.run(["wc", "-l", ATTENTION_PATH], capture_output=True, text=True, shell=True)
print(r.stdout.strip())
