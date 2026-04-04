"""
Transformer-based policy for RL trading.

Architecture:
- Multi-head self-attention over price history (causal masking)
- Cross-asset attention for correlation modeling
- Positional encoding for time series
- Feed-forward policy and value heads
- Supports both sequence and single-step inference
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Independent
except ImportError:
    raise ImportError("PyTorch is required for Transformer policy.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    obs_dim: int = 64            # per-asset observation dim
    act_dim: int = 4             # total action dim (n_assets)
    n_assets: int = 1            # number of assets (for cross-asset attention)
    seq_len: int = 60            # price history length

    # Temporal transformer (self-attention over time)
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # Cross-asset transformer
    use_cross_asset: bool = True
    cross_heads: int = 4
    cross_layers: int = 2

    # Policy head
    policy_hidden: List[int] = field(default_factory=lambda: [256, 128])
    value_hidden: List[int] = field(default_factory=lambda: [256, 128])

    # Output
    log_std_init: float = -0.5
    log_std_min: float = -3.0
    log_std_max: float = 0.5

    lr: float = 1e-4
    max_grad_norm: float = 1.0
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embedding."""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(positions))


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Applies rotation to query and key in attention, enabling relative positional awareness.
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) for applying rotation."""
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, :, :], emb.sin()[None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


# ---------------------------------------------------------------------------
# Causal multi-head attention
# ---------------------------------------------------------------------------

class CausalMultiHeadAttention(nn.Module):
    """
    Causal (autoregressive) multi-head self-attention.
    Future positions are masked out.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self._scale = math.sqrt(self.d_head)

        # Initialize
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.constant_(proj.bias, 0.0)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, H, T, D/H)"""
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, D/H) -> (B, T, D)"""
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: causal mask (seq_len, seq_len) - True where attention is BLOCKED
        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        B, T, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self._scale  # (B, H, T, T)

        # Apply causal mask
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float("-inf"))

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Softmax may produce NaN if all values masked; handle gracefully
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        out = torch.matmul(attn_weights, v)  # (B, H, T, D/H)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        return out, attn_weights


# ---------------------------------------------------------------------------
# Cross-asset attention
# ---------------------------------------------------------------------------

class CrossAssetAttention(nn.Module):
    """
    Multi-head attention across assets at each time step.
    Models inter-asset dependencies and correlations.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch * seq_len, n_assets, d_model)  — treats assets as sequence
        Returns:
            (batch * seq_len, n_assets, d_model)
        """
        residual = x
        out, _ = self.attn(x, x, x)
        out = self.dropout(out)
        return self.norm(residual + out)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single pre-norm transformer block with causal attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = CausalMultiHeadAttention(d_model, n_heads, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        # Init FF
        for m in self.ff.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention (pre-norm)
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, mask=mask)
        x = x + attn_out

        # Feed-forward (pre-norm)
        x = x + self.ff(self.norm2(x))
        return x, attn_weights


# ---------------------------------------------------------------------------
# Cross-asset transformer block
# ---------------------------------------------------------------------------

class CrossAssetTransformerBlock(nn.Module):
    """Combined temporal + cross-asset transformer block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        cross_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal = TransformerBlock(d_model, n_heads, d_ff, dropout)
        self.cross    = CrossAssetAttention(d_model, cross_heads, dropout)
        self.norm     = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,        # (B, N, T, D) — batch, assets, time, features
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, T, D = x.shape

        # Temporal attention: merge batch and asset dims
        x_flat = x.view(B * N, T, D)
        x_flat, _ = self.temporal(x_flat, mask=mask)
        x = x_flat.view(B, N, T, D)

        # Cross-asset attention: merge batch and time dims
        x_cross = x.permute(0, 2, 1, 3).contiguous().view(B * T, N, D)
        x_cross = self.cross(x_cross)
        x = x_cross.view(B, T, N, D).permute(0, 2, 1, 3)

        return x


# ---------------------------------------------------------------------------
# Input embedding
# ---------------------------------------------------------------------------

class AssetInputEmbedding(nn.Module):
    """Projects per-asset observation features into d_model space."""

    def __init__(self, obs_dim: int, d_model: int, use_layer_norm: bool = True):
        super().__init__()
        self.proj = nn.Linear(obs_dim, d_model)
        self.norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.gelu(self.proj(x)))


# ---------------------------------------------------------------------------
# Main Transformer Policy
# ---------------------------------------------------------------------------

class TransformerPolicy(nn.Module):
    """
    Full Transformer-based actor-critic policy.

    Input: sequence of observations (B, seq_len, obs_dim) per asset
    or (B, seq_len, n_assets * obs_dim) flattened

    Output: action distribution mean/std + value estimate
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Per-asset input embedding
        self.input_embed = AssetInputEmbedding(config.obs_dim, config.d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(config.d_model, config.seq_len + 10, config.dropout)

        # Asset embedding (learned token per asset)
        if config.n_assets > 1:
            self.asset_embed = nn.Embedding(config.n_assets, config.d_model)
            nn.init.normal_(self.asset_embed.weight, std=0.02)
        else:
            self.asset_embed = None

        # Temporal transformer blocks (with optional cross-asset)
        if config.use_cross_asset and config.n_assets > 1:
            self.blocks = nn.ModuleList([
                CrossAssetTransformerBlock(
                    config.d_model, config.n_heads, config.cross_heads,
                    config.d_ff, config.dropout
                )
                for _ in range(config.n_layers)
            ])
            self._use_cross = True
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ])
            self._use_cross = False

        self.final_norm = nn.LayerNorm(config.d_model)

        # Policy (actor) head
        policy_layers = []
        prev = config.d_model * config.n_assets  # pool over assets
        for h in config.policy_hidden:
            policy_layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        self.policy_mlp = nn.Sequential(*policy_layers)
        self.actor_mean = nn.Linear(prev, config.act_dim)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

        self.log_std = nn.Parameter(torch.full((config.act_dim,), config.log_std_init))

        # Value (critic) head
        value_layers = []
        prev_v = config.d_model * config.n_assets
        for h in config.value_hidden:
            value_layers += [nn.Linear(prev_v, h), nn.GELU()]
            prev_v = h
        self.value_mlp = nn.Sequential(*value_layers)
        self.value_head = nn.Linear(prev_v, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

        # Causal mask cache
        self._causal_mask: Dict[int, torch.Tensor] = {}

        self.to(self.device)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, str(device))
        if key not in self._causal_mask:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            self._causal_mask[key] = mask
        return self._causal_mask[key]

    def _encode(
        self,
        obs_seq: torch.Tensor,   # (B, T, n_assets * obs_dim) or (B, T, obs_dim) if n_assets==1
    ) -> torch.Tensor:
        """
        Encode observation sequence into context vector.
        Returns (B, d_model * n_assets)
        """
        B, T, _ = obs_seq.shape
        N = self.config.n_assets

        if N > 1:
            # Reshape to (B, T, N, obs_dim)
            obs_per_asset = obs_seq.view(B, T, N, self.config.obs_dim)
        else:
            obs_per_asset = obs_seq.unsqueeze(2)  # (B, T, 1, obs_dim)

        # Embed each asset
        embedded = self.input_embed(obs_per_asset)  # (B, T, N, d_model)

        # Add asset embeddings
        if self.asset_embed is not None:
            asset_ids = torch.arange(N, device=obs_seq.device)
            asset_emb = self.asset_embed(asset_ids)  # (N, d_model)
            embedded = embedded + asset_emb[None, None, :, :]

        mask = self._get_causal_mask(T, obs_seq.device)

        if self._use_cross:
            # (B, N, T, d_model)
            x = embedded.permute(0, 2, 1, 3)
            for block in self.blocks:
                x = block(x, mask=mask)
            # Pool last time step, concat assets
            last = x[:, :, -1, :]     # (B, N, d_model)
            context = last.reshape(B, N * self.config.d_model)
        else:
            # Merge asset and feature dims
            x = embedded.view(B * N, T, self.config.d_model)
            x = self.pos_enc(x)
            for block in self.blocks:
                x, _ = block(x, mask=mask)
            x = self.final_norm(x)
            # Take last time step, reshape back
            last = x[:, -1, :]                     # (B*N, d_model)
            last = last.view(B, N * self.config.d_model)
            context = last

        return context

    def get_action_and_value(
        self,
        obs_seq: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_seq: (B, T, obs_dim) or (B, T, n_assets * obs_dim)
            action: optional action for computing log_prob of existing action
        Returns:
            (action, log_prob, entropy, value)
        """
        context = self._encode(obs_seq)

        # Policy head
        policy_feat = self.policy_mlp(context)
        mean = self.actor_mean(policy_feat)
        log_std = torch.clamp(self.log_std.expand_as(mean), self.config.log_std_min, self.config.log_std_max)
        std = log_std.exp()

        dist = Independent(Normal(mean, std), 1)

        if action is None:
            raw_action = dist.rsample()
        else:
            raw_action = action

        action_out = torch.tanh(raw_action)

        # Tanh-corrected log prob
        log_prob = dist.log_prob(raw_action)
        tanh_corr = (2.0 * (math.log(2) - raw_action - F.softplus(-2.0 * raw_action))).sum(-1)
        log_prob = log_prob - tanh_corr

        entropy = dist.entropy().mean()

        # Value head
        value_feat = self.value_mlp(context)
        value = self.value_head(value_feat).squeeze(-1)

        return action_out, log_prob, entropy, value

    @torch.no_grad()
    def get_action(
        self,
        obs_seq: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self._encode(obs_seq)
        policy_feat = self.policy_mlp(context)
        mean = self.actor_mean(policy_feat)

        if deterministic:
            return torch.tanh(mean), torch.zeros(mean.shape[0], device=mean.device)

        log_std = torch.clamp(self.log_std.expand_as(mean), self.config.log_std_min, self.config.log_std_max)
        std = log_std.exp()
        dist = Independent(Normal(mean, std), 1)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw) - (2.0 * (math.log(2) - raw - F.softplus(-2.0 * raw))).sum(-1)
        return action, log_prob

    @torch.no_grad()
    def get_value(self, obs_seq: torch.Tensor) -> torch.Tensor:
        context = self._encode(obs_seq)
        return self.value_head(self.value_mlp(context)).squeeze(-1)

    def get_attention_weights(self, obs_seq: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention weights for visualization."""
        B, T, _ = obs_seq.shape
        N = self.config.n_assets
        obs_per_asset = obs_seq.view(B, T, N, self.config.obs_dim)
        embedded = self.input_embed(obs_per_asset)

        if self.asset_embed is not None:
            asset_ids = torch.arange(N, device=obs_seq.device)
            embedded = embedded + self.asset_embed(asset_ids)[None, None, :, :]

        mask = self._get_causal_mask(T, obs_seq.device)
        x = embedded.view(B * N, T, self.config.d_model)
        x = self.pos_enc(x)

        all_weights = []
        for block in self.blocks:
            if isinstance(block, TransformerBlock):
                normed = block.norm1(x)
                _, weights = block.attn(normed, mask=mask)
                all_weights.append(weights.view(B, N, self.config.n_heads, T, T))
                x, _ = block(x, mask=mask)

        return all_weights


# ---------------------------------------------------------------------------
# Transformer PPO Agent
# ---------------------------------------------------------------------------

class TransformerPPOAgent:
    """
    PPO agent using Transformer policy.
    Maintains a rolling observation history (sequence buffer).
    """

    def __init__(self, config: TransformerConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.policy = TransformerPolicy(config)

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.lr,
            weight_decay=1e-4,
        )

        self._seq_buf: Optional[np.ndarray] = None   # (seq_len, obs_dim)
        self._train_step = 0

    def reset_sequence(self) -> None:
        self._seq_buf = None

    def _update_sequence(self, obs: np.ndarray) -> np.ndarray:
        """Maintain rolling window of observations."""
        seq_len = self.config.seq_len
        obs_dim = self.config.obs_dim * self.config.n_assets

        if self._seq_buf is None:
            self._seq_buf = np.zeros((seq_len, obs_dim), dtype=np.float32)
        self._seq_buf = np.roll(self._seq_buf, -1, axis=0)
        self._seq_buf[-1] = obs
        return self._seq_buf.copy()

    @torch.no_grad()
    def collect_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Sample action given current obs. Returns (action, log_prob, value)."""
        seq = self._update_sequence(obs)
        seq_t = torch.FloatTensor(seq).unsqueeze(0).to(self.device)  # (1, T, obs_dim)

        action, log_prob = self.policy.get_action(seq_t, deterministic=deterministic)
        value = self.policy.get_value(seq_t)

        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def compute_loss(
        self,
        obs_seqs: torch.Tensor,    # (B, T, obs_dim)
        actions: torch.Tensor,      # (B, act_dim) - raw (pre-tanh)
        old_log_probs: torch.Tensor,  # (B,)
        returns: torch.Tensor,         # (B,)
        advantages: torch.Tensor,      # (B,)
        old_values: torch.Tensor,      # (B,)
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        value_clip: float = 0.2,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss for transformer policy."""
        action_out, log_probs, entropy, values = self.policy.get_action_and_value(obs_seqs, actions)

        # Policy loss
        ratio = (log_probs - old_log_probs).exp()
        pg1 = -advantages * ratio
        pg2 = -advantages * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        policy_loss = torch.max(pg1, pg2).mean()

        # Value loss (clipped)
        v_clip = old_values + torch.clamp(values - old_values, -value_clip, value_clip)
        vf1 = F.mse_loss(values, returns)
        vf2 = F.mse_loss(v_clip, returns)
        value_loss = torch.max(vf1, vf2)

        ent_loss = -entropy_coef * entropy
        total = policy_loss + value_coef * value_loss + ent_loss

        info = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(((ratio - 1) - (log_probs - old_log_probs)).mean().item()),
        }
        return total, info

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self._train_step,
            "config": self.config,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._train_step = ckpt.get("train_step", 0)


# ---------------------------------------------------------------------------
# Temporal fusion transformer (additional architecture)
# ---------------------------------------------------------------------------

class TemporalFusionTransformer(nn.Module):
    """
    Simplified Temporal Fusion Transformer for financial time series.
    Incorporates static context (asset metadata), known future inputs,
    and observed inputs.
    """

    def __init__(
        self,
        obs_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1,
        seq_len: int = 30,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.seq_len = seq_len

        # Variable selection networks
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.static_context = nn.Linear(obs_dim, d_model)

        # LSTM encoder (for local processing)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)

        # Static enrichment gate
        self.static_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # Temporal self-attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        # Output
        self.output_proj = nn.Linear(d_model, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, obs_dim)
        Returns: (B, output_dim) — prediction for next step
        """
        B, T, _ = x.shape

        # Project input
        h = F.gelu(self.input_proj(x))  # (B, T, d_model)

        # Static context from mean of sequence
        static = self.static_context(x.mean(dim=1))  # (B, d_model)

        # LSTM encoding
        lstm_out, _ = self.lstm(h)  # (B, T, d_model)

        # Static enrichment
        static_expanded = static.unsqueeze(1).expand(-1, T, -1)
        gate = self.static_gate(torch.cat([lstm_out, static_expanded], dim=-1))
        h = lstm_out * gate + lstm_out  # residual gated connection

        # Self-attention
        residual = h
        attn_out, _ = self.attn(h, h, h)
        h = self.norm1(residual + self.dropout(attn_out))

        # FFN
        h = self.norm2(h + self.ff(h))

        # Output from last step
        out = self.output_proj(h[:, -1, :])  # (B, output_dim)
        return out


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("Testing Transformer policy...")
    config = TransformerConfig(
        obs_dim=32,
        act_dim=3,
        n_assets=2,
        seq_len=20,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        use_cross_asset=True,
        cross_heads=2,
        cross_layers=1,
        device="cpu",
    )
    agent = TransformerPPOAgent(config)
    agent.reset_sequence()

    t0 = time.time()
    obs_dim_total = config.obs_dim * config.n_assets
    for step in range(50):
        obs = np.random.randn(obs_dim_total).astype(np.float32)
        action, log_prob, value = agent.collect_action(obs)

    print(f"Action shape: {action.shape}, value: {value:.4f}, log_prob: {log_prob:.4f}")

    # Test forward pass
    B, T, D = 4, 20, obs_dim_total
    obs_batch = torch.randn(B, T, D)
    policy = agent.policy
    action_out, log_prob_out, entropy, value_out = policy.get_action_and_value(obs_batch)
    print(f"Batch action: {action_out.shape}, value: {value_out.shape}")

    # Attention weights
    weights = policy.get_attention_weights(obs_batch)
    print(f"Attention layers: {len(weights)}")

    # TFT test
    tft = TemporalFusionTransformer(obs_dim=32, seq_len=20)
    x = torch.randn(4, 20, 32)
    out = tft(x)
    print(f"TFT output: {out.shape}")

    print(f"Time: {time.time() - t0:.2f}s")
    print("Transformer policy self-test passed.")
