"""
Proximal Policy Optimization (PPO) with Actor-Critic network.

Architecture:
  - LSTM encoder over recent observations
  - Shared MLP trunk
  - Separate Actor (policy) and Critic (value) heads
  - GAE advantage estimation
  - Clipped surrogate objective
  - Entropy bonus for exploration
  - Value function clipping
  - Mini-batch updates with multiple epochs
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Independent
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch is required for PPO. Install with: pip install torch")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    # Network architecture
    obs_dim: int = 256
    act_dim: int = 4
    lstm_hidden: int = 256
    lstm_layers: int = 2
    mlp_hidden: List[int] = field(default_factory=lambda: [512, 256, 128])
    use_lstm: bool = True
    use_layer_norm: bool = True
    activation: str = "tanh"  # "tanh" | "relu" | "elu"

    # PPO hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    value_clip_epsilon: float = 0.2

    # Training
    n_epochs: int = 10
    minibatch_size: int = 64
    rollout_steps: int = 2048
    normalize_advantages: bool = True

    # Output
    log_std_init: float = -0.5
    log_std_min: float = -3.0
    log_std_max: float = 0.5

    # Devices
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Network components
# ---------------------------------------------------------------------------

def _get_activation(name: str) -> nn.Module:
    return {"tanh": nn.Tanh(), "relu": nn.ReLU(), "elu": nn.ELU(), "gelu": nn.GELU()}.get(name, nn.Tanh())


def _init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal initialization."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LSTM):
        for name, p in module.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)


class ResidualBlock(nn.Module):
    """Pre-norm residual block for stability."""

    def __init__(self, dim: int, activation: str = "tanh", use_layer_norm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.fc1  = nn.Linear(dim, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.act  = _get_activation(activation)
        _init_weights(self.fc1)
        _init_weights(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x + residual


class MLPEncoder(nn.Module):
    """Multi-layer perceptron with optional residual connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = "tanh",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(_get_activation(activation))
            prev = h
        self.net = nn.Sequential(*layers)
        self.output_dim = prev
        self.net.apply(lambda m: _init_weights(m) if isinstance(m, nn.Linear) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMEncoder(nn.Module):
    """LSTM-based sequence encoder for time series observations."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim
        _init_weights(self.lstm)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, input_dim) OR (batch, input_dim) for single step
        Returns:
            out: (batch, hidden_dim)
            (h_n, c_n)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        out, (h_n, c_n) = self.lstm(x, hidden)
        return out[:, -1, :], (h_n, c_n)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return h, c


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    PPO Actor-Critic with:
    - Optional LSTM encoder for temporal dependencies
    - Shared MLP trunk
    - Gaussian policy head (continuous actions)
    - Value head
    """

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        self.act_dim = config.act_dim
        device_str = config.device
        self.device = torch.device(device_str)

        # Encoder
        if config.use_lstm:
            self.encoder = LSTMEncoder(config.obs_dim, config.lstm_hidden, config.lstm_layers)
            trunk_input  = config.lstm_hidden
        else:
            self.encoder = None
            trunk_input  = config.obs_dim

        # Shared trunk
        self.trunk = MLPEncoder(
            trunk_input, config.mlp_hidden,
            config.activation, config.use_layer_norm
        )
        trunk_out = self.trunk.output_dim

        # Actor head
        self.actor_mean = nn.Linear(trunk_out, config.act_dim)
        _init_weights(self.actor_mean, gain=0.01)

        self.log_std = nn.Parameter(
            torch.full((config.act_dim,), config.log_std_init)
        )

        # Critic head
        self.critic = nn.Linear(trunk_out, 1)
        _init_weights(self.critic, gain=1.0)

        self.to(self.device)

    def _encode(
        self,
        obs: torch.Tensor,
        lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self.encoder is not None:
            encoded, new_hidden = self.encoder(obs, lstm_hidden)
        else:
            encoded = obs
            new_hidden = None
        features = self.trunk(encoded)
        return features, new_hidden

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Returns:
            action, log_prob, entropy, value, new_lstm_hidden
        """
        features, new_hidden = self._encode(obs, lstm_hidden)

        # Policy
        mean = self.actor_mean(features)
        log_std = torch.clamp(
            self.log_std.expand_as(mean),
            self.config.log_std_min,
            self.config.log_std_max,
        )
        std = log_std.exp()
        dist = Independent(Normal(mean, std), 1)

        if action is None:
            action = dist.rsample()
        action_clipped = torch.tanh(action)  # squash to [-1, 1]

        # Log prob with tanh correction
        log_prob = dist.log_prob(action)
        tanh_correction = (2.0 * (math.log(2) - action - F.softplus(-2.0 * action))).sum(dim=-1)
        log_prob = log_prob - tanh_correction

        entropy = dist.entropy().mean()

        # Value
        value = self.critic(features).squeeze(-1)

        return action_clipped, log_prob, entropy, value, new_hidden

    def get_value(
        self,
        obs: torch.Tensor,
        lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        features, _ = self._encode(obs, lstm_hidden)
        return self.critic(features).squeeze(-1)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        features, new_hidden = self._encode(obs, lstm_hidden)
        mean = self.actor_mean(features)
        log_std = torch.clamp(self.log_std.expand_as(mean),
                              self.config.log_std_min, self.config.log_std_max)
        std = log_std.exp()
        dist = Independent(Normal(mean, std), 1)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()

        action_clipped = torch.tanh(action)
        log_prob = dist.log_prob(action)
        tanh_correction = (2.0 * (math.log(2) - action - F.softplus(-2.0 * action))).sum(dim=-1)
        log_prob = log_prob - tanh_correction

        return action_clipped, log_prob, new_hidden


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores and processes rollout data for PPO updates."""

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        act_dim: int,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self._ptr = 0
        self._full = False

        self.obs       = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions   = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards   = np.zeros(buffer_size, dtype=np.float32)
        self.dones     = np.zeros(buffer_size, dtype=np.float32)
        self.values    = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.returns   = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        idx = self._ptr % self.buffer_size
        self.obs[idx]       = obs
        self.actions[idx]   = action
        self.rewards[idx]   = reward
        self.dones[idx]     = float(done)
        self.values[idx]    = value
        self.log_probs[idx] = log_prob
        self._ptr += 1
        if self._ptr >= self.buffer_size:
            self._full = True

    def compute_gae(self, last_value: float, last_done: bool) -> None:
        """Compute GAE advantages and returns in-place."""
        size = self.buffer_size
        last_gae = 0.0
        last_val = float(last_value) * (1.0 - float(last_done))

        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_val
                next_done  = float(last_done)
            else:
                next_value = self.values[t + 1]
                next_done  = self.dones[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * (1.0 - next_done)
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * last_gae
            self.advantages[t] = last_gae

        self.returns[:] = self.advantages + self.values

    def get_minibatches(
        self, minibatch_size: int, normalize_adv: bool = True
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Yield shuffled mini-batches."""
        size = self.buffer_size
        indices = np.random.permutation(size)

        adv = self.advantages.copy()
        if normalize_adv and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, size, minibatch_size):
            end = min(start + minibatch_size, size)
            idx = indices[start:end]
            yield {
                "obs":       torch.FloatTensor(self.obs[idx]).to(self.device),
                "actions":   torch.FloatTensor(self.actions[idx]).to(self.device),
                "log_probs": torch.FloatTensor(self.log_probs[idx]).to(self.device),
                "returns":   torch.FloatTensor(self.returns[idx]).to(self.device),
                "advantages": torch.FloatTensor(adv[idx]).to(self.device),
                "values":    torch.FloatTensor(self.values[idx]).to(self.device),
            }

    def reset(self) -> None:
        self._ptr = 0
        self._full = False


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    Full PPO agent with:
    - Actor-Critic network (LSTM + MLP)
    - GAE advantage estimation
    - Clipped surrogate objective
    - Entropy bonus
    - Value function clipping
    - Mini-batch SGD updates
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.network = ActorCritic(config)

        # Separate optimizers for actor and critic
        actor_params  = list(self.network.actor_mean.parameters()) + [self.network.log_std]
        if self.network.encoder:
            actor_params += list(self.network.encoder.parameters())
        actor_params += list(self.network.trunk.parameters())

        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.lr_actor, eps=1e-5
        )

        self.buffer = RolloutBuffer(
            buffer_size=config.rollout_steps,
            obs_dim=config.obs_dim,
            act_dim=config.act_dim,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=self.device,
        )

        self._lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._train_step = 0
        self._episode_rewards: List[float] = []

    def reset_lstm(self, batch_size: int = 1) -> None:
        if self.config.use_lstm and self.network.encoder is not None:
            self._lstm_hidden = self.network.encoder.init_hidden(batch_size, self.device)
        else:
            self._lstm_hidden = None

    @torch.no_grad()
    def collect_action(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Sample action from policy.
        Returns: (action_np, log_prob_float, value_float)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, new_hidden = self.network.get_action(obs_t, lstm_hidden=self._lstm_hidden)
        value = self.network.get_value(obs_t, lstm_hidden=self._lstm_hidden)
        self._lstm_hidden = new_hidden
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        self.buffer.add(obs, action, reward, done, value, log_prob)
        if done:
            self.reset_lstm()

    def update(self, last_obs: np.ndarray, last_done: bool) -> Dict[str, float]:
        """
        Run PPO update on collected rollout.
        Returns dict of training metrics.
        """
        self.network.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            last_value = float(self.network.get_value(obs_t).item())
        self.buffer.compute_gae(last_value, last_done)

        # Training
        self.network.train()
        metrics: Dict[str, List[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [],
            "approx_kl": [], "clip_fraction": [], "total_loss": [],
        }

        for epoch in range(self.config.n_epochs):
            for batch in self.buffer.get_minibatches(
                self.config.minibatch_size, self.config.normalize_advantages
            ):
                obs_b      = batch["obs"]
                actions_b  = batch["actions"]
                old_lp_b   = batch["log_probs"]
                returns_b  = batch["returns"]
                adv_b      = batch["advantages"]
                old_vals_b = batch["values"]

                # Forward pass
                _, new_lp, entropy, new_vals, _ = self.network.get_action_and_value(
                    obs_b, actions_b
                )

                # Policy loss (clipped surrogate)
                log_ratio = new_lp - old_lp_b
                ratio = log_ratio.exp()

                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(
                    ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                vals_clipped = old_vals_b + torch.clamp(
                    new_vals - old_vals_b,
                    -self.config.value_clip_epsilon,
                    self.config.value_clip_epsilon,
                )
                vf_loss1 = F.mse_loss(new_vals, returns_b)
                vf_loss2 = F.mse_loss(vals_clipped, returns_b)
                value_loss = torch.max(vf_loss1, vf_loss2)

                # Entropy bonus
                ent_loss = -self.config.entropy_coef * entropy

                # Total loss
                total_loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    + ent_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_fraction"].append(clip_frac)
                metrics["total_loss"].append(total_loss.item())

        self.buffer.reset()
        self._train_step += 1

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def save(self, path: str) -> None:
        """Save agent state."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "train_step": self._train_step,
            "config": self.config,
        }, path)

    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._train_step = checkpoint.get("train_step", 0)

    def set_learning_rate(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    @property
    def train_step(self) -> int:
        return self._train_step


# ---------------------------------------------------------------------------
# Learning rate schedules
# ---------------------------------------------------------------------------

class LinearLRSchedule:
    """Linearly decay learning rate from lr_start to lr_end."""

    def __init__(self, agent: PPOAgent, lr_start: float, lr_end: float, total_steps: int):
        self.agent = agent
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.total_steps = total_steps

    def step(self, current_step: int) -> float:
        frac = max(0.0, 1.0 - current_step / self.total_steps)
        lr = self.lr_end + frac * (self.lr_start - self.lr_end)
        self.agent.set_learning_rate(lr)
        return lr


class CosineAnnealingSchedule:
    """Cosine annealing with warm restarts."""

    def __init__(self, agent: PPOAgent, lr_max: float, lr_min: float, T_0: int, T_mult: int = 2):
        self.agent = agent
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_0 = T_0
        self.T_mult = T_mult
        self._t = 0
        self._T_cur = T_0

    def step(self) -> float:
        self._t += 1
        if self._t >= self._T_cur:
            self._t = 0
            self._T_cur *= self._T_mult
        frac = self._t / self._T_cur
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * frac))
        self.agent.set_learning_rate(lr)
        return lr


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------

def compute_explained_variance(returns: np.ndarray, values: np.ndarray) -> float:
    """Compute explained variance of value estimates."""
    var_returns = float(np.var(returns))
    if var_returns < 1e-10:
        return float("nan")
    residuals = returns - values
    return float(1.0 - np.var(residuals) / var_returns)


def compute_policy_entropy(log_stds: np.ndarray) -> float:
    """Compute mean entropy of Gaussian policy given log stds."""
    # H = 0.5 * log(2*pi*e*sigma^2)
    return float(0.5 * np.log(2 * math.pi * math.e) + log_stds.mean())


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("Testing PPO agent...")
    config = PPOConfig(
        obs_dim=64,
        act_dim=3,
        lstm_hidden=128,
        lstm_layers=1,
        mlp_hidden=[256, 128],
        n_epochs=3,
        minibatch_size=32,
        rollout_steps=128,
        device="cpu",
    )
    agent = PPOAgent(config)
    agent.reset_lstm()

    # Simulate rollout collection
    obs = np.random.randn(64).astype(np.float32)
    t0 = time.time()
    for step in range(128):
        action, log_prob, value = agent.collect_action(obs)
        reward = float(np.random.randn())
        done = (step == 127)
        agent.store_transition(obs, action, reward, done, value, log_prob)
        obs = np.random.randn(64).astype(np.float32)

    metrics = agent.update(obs, last_done=False)
    elapsed = time.time() - t0
    print(f"Update metrics: {metrics}")
    print(f"Time: {elapsed:.2f}s | train_step: {agent.train_step}")
    print("PPO self-test passed.")
