"""
networks.py — Shared neural network architectures for Hyper-Agent MARL.

Modules:
- GRUActor: recurrent policy network
- TransformerCritic: self-attention value function
- GraphAttentionNetwork: agent relation modeling
- DuelingDQN: dueling network architecture
- NoisyNet extensions
- Positional encodings
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agents.base_agent import layer_init, NoisyLinear, EPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# GRU Actor
# ---------------------------------------------------------------------------

class GRUActor(nn.Module):
    """
    Recurrent actor with GRU for partial observability.
    obs -> GRU -> Gaussian policy
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gru_layers: int = 1,
        pre_layers: int = 2,
        log_std_min: float = -5,
        log_std_max: float = 2,
        squash: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.squash = squash

        # Pre-GRU MLP
        pre = []
        in_d = obs_dim
        for _ in range(pre_layers):
            pre.append(nn.Linear(in_d, hidden_dim))
            pre.append(nn.LayerNorm(hidden_dim))
            pre.append(nn.ReLU())
            in_d = hidden_dim
        self.pre_net = nn.Sequential(*pre)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )

        self.post_norm = nn.LayerNorm(hidden_dim)

        self.mean_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

        self._init_gru()

    def _init_gru(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, T, obs_dim) or (B, obs_dim) for single step
            hidden: (num_layers, B, hidden_dim)
        Returns:
            mean, log_std, new_hidden
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        x = self.pre_net(obs)
        out, new_h = self.gru(x, hidden)
        out = self.post_norm(out)

        if squeeze:
            out = out.squeeze(1)

        mean = self.mean_head(out)
        log_std = torch.clamp(self.log_std_head(out), self.log_std_min, self.log_std_max)
        return mean, log_std, new_h

    def get_action(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, new_h = self.forward(obs, hidden)
        std = log_std.exp()

        if deterministic:
            action = mean
            if self.squash:
                action = torch.tanh(action)
            log_prob = torch.zeros(action.shape[:-1] + (1,), device=obs.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            x = dist.rsample()
            log_prob = dist.log_prob(x).sum(dim=-1, keepdim=True)
            if self.squash:
                action = torch.tanh(x)
                log_prob -= torch.sum(
                    torch.log(torch.clamp(1 - action ** 2, min=EPS)), dim=-1, keepdim=True
                )
            else:
                action = x

        return action, log_prob, new_h


# ---------------------------------------------------------------------------
# Transformer critic
# ---------------------------------------------------------------------------

class TransformerCritic(nn.Module):
    """
    Transformer-based value function.
    Takes a sequence of observations (history) and outputs V(s).
    """

    def __init__(
        self,
        obs_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 500,
        output_dim: int = 1,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(obs_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.value_head = layer_init(nn.Linear(d_model, output_dim), std=1.0)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.0)

    def forward(
        self,
        obs_seq: torch.Tensor,          # (B, T, obs_dim)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) True=masked
    ) -> torch.Tensor:
        """Returns value: (B, output_dim) using last timestep."""
        x = self.input_proj(obs_seq)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.out_norm(x)
        # Use last timestep
        x = x[:, -1, :]
        return self.value_head(x)

    def forward_all(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Returns value for every timestep: (B, T, output_dim)."""
        x = self.input_proj(obs_seq)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.out_norm(x)
        return self.value_head(x)


# ---------------------------------------------------------------------------
# Dueling DQN network
# ---------------------------------------------------------------------------

class DuelingNetwork(nn.Module):
    """
    Dueling Network architecture.
    Separates value V(s) and advantage A(s, a) streams.
    Q(s, a) = V(s) + A(s, a) - mean(A(s, *))
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        use_noisy: bool = False,
    ):
        super().__init__()

        # Shared base
        base_layers: List[nn.Module] = []
        in_d = input_dim
        for _ in range(num_hidden):
            base_layers.append(nn.Linear(in_d, hidden_dim))
            base_layers.append(nn.ReLU())
            in_d = hidden_dim
        self.base = nn.Sequential(*base_layers)

        # Value stream
        if use_noisy:
            self.value_stream = nn.Sequential(
                NoisyLinear(hidden_dim, hidden_dim),
                nn.ReLU(),
                NoisyLinear(hidden_dim, 1),
            )
            self.advantage_stream = nn.Sequential(
                NoisyLinear(hidden_dim, hidden_dim),
                nn.ReLU(),
                NoisyLinear(hidden_dim, action_dim),
            )
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, 1), std=1.0),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, action_dim), std=1.0),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base(x)
        value = self.value_stream(h)
        advantage = self.advantage_stream(h)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ---------------------------------------------------------------------------
# Multi-head output network
# ---------------------------------------------------------------------------

class MultiHeadOutputNetwork(nn.Module):
    """
    Shared backbone with multiple output heads.
    Used for multi-task or multi-agent parameter sharing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dims: Dict[str, int],
        num_shared_layers: int = 3,
    ):
        super().__init__()

        shared = []
        in_d = input_dim
        for _ in range(num_shared_layers):
            shared.append(nn.Linear(in_d, hidden_dim))
            shared.append(nn.LayerNorm(hidden_dim))
            shared.append(nn.ReLU())
            in_d = hidden_dim
        self.shared = nn.Sequential(*shared)

        self.heads = nn.ModuleDict({
            name: layer_init(nn.Linear(hidden_dim, dim), std=0.01)
            for name, dim in output_dims.items()
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(x)
        return {name: head(h) for name, head in self.heads.items()}


# ---------------------------------------------------------------------------
# World model network (for model-based RL)
# ---------------------------------------------------------------------------

class WorldModelNetwork(nn.Module):
    """
    Dynamics model: predicts next state given current state and action.
    Used for model-based planning / imagination rollouts.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        ensemble_size: int = 5,
        predict_reward: bool = True,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.predict_reward = predict_reward
        self.obs_dim = obs_dim

        # Ensemble of models for uncertainty estimation
        self.models = nn.ModuleList([
            self._build_model(obs_dim, action_dim, hidden_dim, num_layers, predict_reward)
            for _ in range(ensemble_size)
        ])

    def _build_model(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        predict_reward: bool,
    ) -> nn.Module:
        layers: List[nn.Module] = []
        in_d = obs_dim + action_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.SiLU())
            in_d = hidden_dim
        output_dim = obs_dim * 2  # mean + log_var
        if predict_reward:
            output_dim += 2  # reward mean + log_var
        layers.append(layer_init(nn.Linear(in_d, output_dim), std=0.01))
        return nn.Sequential(*layers)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        model_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with 'next_obs_mean', 'next_obs_logvar',
        optionally 'reward_mean', 'reward_logvar'.
        """
        x = torch.cat([obs, action], dim=-1)

        if model_idx is not None:
            out = self.models[model_idx](x)
        else:
            # Use random model from ensemble
            idx = np.random.randint(0, self.ensemble_size)
            out = self.models[idx](x)

        next_obs_mean = out[..., :self.obs_dim]
        next_obs_logvar = torch.clamp(out[..., self.obs_dim:2 * self.obs_dim], -10, 2)
        result = {
            "next_obs_mean": next_obs_mean + obs,  # predict delta
            "next_obs_logvar": next_obs_logvar,
        }
        if self.predict_reward:
            result["reward_mean"] = out[..., 2 * self.obs_dim:2 * self.obs_dim + 1]
            result["reward_logvar"] = out[..., 2 * self.obs_dim + 1:2 * self.obs_dim + 2]
        return result

    def ensemble_uncertainty(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute variance across ensemble as uncertainty estimate."""
        preds = []
        x = torch.cat([obs, action], dim=-1)
        for model in self.models:
            out = model(x)
            preds.append(out[..., :self.obs_dim])
        preds_stack = torch.stack(preds, dim=0)  # (E, B, obs_dim)
        return preds_stack.var(dim=0).mean(dim=-1, keepdim=True)  # (B, 1)

    def predict(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """Sample next observation from world model."""
        out = self.forward(obs, action)
        mean = out["next_obs_mean"]
        logvar = out["next_obs_logvar"]
        if sample:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(mean)
            return mean + std * eps
        return mean


# ---------------------------------------------------------------------------
# Ensemble value function
# ---------------------------------------------------------------------------

class EnsembleValueFunction(nn.Module):
    """
    Ensemble of value functions for uncertainty-aware value estimation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        ensemble_size: int = 5,
        num_layers: int = 3,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size

        self.nets = nn.ModuleList([
            self._build_net(input_dim, hidden_dim, num_layers)
            for _ in range(ensemble_size)
        ])

    def _build_net(self, input_dim: int, hidden_dim: int, num_layers: int) -> nn.Module:
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(layer_init(nn.Linear(hidden_dim, 1), std=1.0))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean_value, std_value) across ensemble."""
        values = torch.stack([net(x) for net in self.nets], dim=0)  # (E, B, 1)
        return values.mean(dim=0), values.std(dim=0)

    def min_value(self, x: torch.Tensor) -> torch.Tensor:
        """Pessimistic value: minimum across ensemble."""
        values = torch.stack([net(x) for net in self.nets], dim=0)
        return values.min(dim=0).values


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "PositionalEncoding",
    "GRUActor",
    "TransformerCritic",
    "DuelingNetwork",
    "MultiHeadOutputNetwork",
    "WorldModelNetwork",
    "EnsembleValueFunction",
]
