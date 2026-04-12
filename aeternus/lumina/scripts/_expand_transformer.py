"""Expand transformer.py with advanced components."""

PATH = r"C:\Users\Matthew\srfm-lab\aeternus\lumina\lumina\transformer.py"

CONTENT = r'''

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
'''

with open(PATH, "a", encoding="utf-8") as f:
    f.write(CONTENT)

import subprocess
r = subprocess.run(["wc", "-l", PATH], capture_output=True, text=True, shell=True)
print(r.stdout.strip())
