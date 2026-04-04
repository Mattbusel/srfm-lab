"""
Dueling Double DQN with:
- Dueling architecture (value + advantage streams)
- Double Q-learning (target network for action selection)
- Per-step TD error / prioritized experience replay
- NoisyNet layers for parameter-space exploration
- N-step returns
- Epsilon-greedy fallback
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
except ImportError:
    raise ImportError("PyTorch is required for DQN.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DQNConfig:
    obs_dim: int = 256
    n_actions: int = 9              # e.g. {-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1} × n_assets
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "relu"
    use_layer_norm: bool = True

    # Dueling
    value_hidden: int = 128
    advantage_hidden: int = 128

    # Noisy nets
    use_noisy_nets: bool = True
    noisy_sigma_init: float = 0.5

    # Double DQN
    use_double_dqn: bool = True

    # Training
    lr: float = 2.5e-4
    gamma: float = 0.99
    batch_size: int = 32
    buffer_capacity: int = 100_000
    warmup_steps: int = 1000
    target_update_freq: int = 500    # hard update every N steps
    target_polyak_tau: Optional[float] = None  # None=hard update, else soft

    # Epsilon greedy (used if noisy nets disabled)
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 50_000

    # N-step returns
    n_steps: int = 3

    # PER
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 100_000

    max_grad_norm: float = 10.0
    device: str = "cpu"


# ---------------------------------------------------------------------------
# NoisyLinear layer
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """
    Factorised NoisyNet linear layer.
    Adds parametric noise to weights for exploration.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_init   = sigma_init

        # Mean weights
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # Fixed noise buffers (not trained)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        """Sample new epsilon noise (factorized)."""
        eps_p = self._f(torch.randn(self.in_features,  device=self.weight_mu.device))
        eps_q = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_epsilon.copy_(eps_q.outer(eps_p))
        self.bias_epsilon.copy_(eps_q)

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# Shared encoder
# ---------------------------------------------------------------------------

def _make_layer(in_dim: int, out_dim: int, activation: str, use_ln: bool, use_noisy: bool) -> nn.Sequential:
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}.get(activation, nn.ReLU)
    linear = NoisyLinear(in_dim, out_dim) if use_noisy else nn.Linear(in_dim, out_dim)
    layers: List[nn.Module] = [linear]
    if use_ln:
        layers.append(nn.LayerNorm(out_dim))
    layers.append(act())
    return nn.Sequential(*layers)


class SharedEncoder(nn.Module):
    """Shared MLP encoder for the DQN."""

    def __init__(self, config: DQNConfig):
        super().__init__()
        self.config = config
        layers = []
        prev = config.obs_dim
        for h in config.hidden_dims:
            layers.append(_make_layer(prev, h, config.activation, config.use_layer_norm, config.use_noisy_nets))
            prev = h
        self.net = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ---------------------------------------------------------------------------
# Dueling DQN network
# ---------------------------------------------------------------------------

class DuelingDQN(nn.Module):
    """
    Dueling network architecture:
        Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]
    """

    def __init__(self, config: DQNConfig):
        super().__init__()
        self.config = config
        self.encoder = SharedEncoder(config)
        enc_out = self.encoder.output_dim

        # Value stream
        if config.use_noisy_nets:
            self.value_head = nn.Sequential(
                NoisyLinear(enc_out, config.value_hidden),
                nn.ReLU(),
                NoisyLinear(config.value_hidden, 1),
            )
            # Advantage stream
            self.advantage_head = nn.Sequential(
                NoisyLinear(enc_out, config.advantage_hidden),
                nn.ReLU(),
                NoisyLinear(config.advantage_hidden, config.n_actions),
            )
        else:
            self.value_head = nn.Sequential(
                nn.Linear(enc_out, config.value_hidden),
                nn.ReLU(),
                nn.Linear(config.value_hidden, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(enc_out, config.advantage_hidden),
                nn.ReLU(),
                nn.Linear(config.advantage_hidden, config.n_actions),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, NoisyLinear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Q-values of shape (batch, n_actions)."""
        features = self.encoder(x)
        value     = self.value_head(features)          # (batch, 1)
        advantage = self.advantage_head(features)       # (batch, n_actions)
        # Dueling combination: subtract mean advantage for identifiability
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self) -> None:
        self.encoder.reset_noise()
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ---------------------------------------------------------------------------
# PER Replay buffer (reuse segment tree logic)
# ---------------------------------------------------------------------------

class DQNReplayBuffer:
    """Replay buffer with optional PER for DQN."""

    def __init__(self, config: DQNConfig, device: torch.device):
        self.config = config
        self.device = device
        self.capacity = config.buffer_capacity
        self._ptr = 0
        self._size = 0

        self.obs      = np.zeros((self.capacity, config.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, config.obs_dim), dtype=np.float32)
        self.actions  = np.zeros(self.capacity, dtype=np.int64)
        self.rewards  = np.zeros(self.capacity, dtype=np.float32)
        self.dones    = np.zeros(self.capacity, dtype=np.float32)

        if config.use_per:
            # Sum tree for priorities
            self._tree_size = 1
            while self._tree_size < self.capacity:
                self._tree_size *= 2
            self._sum_tree = np.zeros(2 * self._tree_size, dtype=np.float64)
            self._min_tree = np.full(2 * self._tree_size, float("inf"), dtype=np.float64)
            self._max_priority = 1.0
            self.beta = config.per_beta_start
            self._beta_increment = (config.per_beta_end - config.per_beta_start) / max(config.per_beta_steps, 1)

    def _tree_update(self, idx: int, priority: float) -> None:
        tree_idx = idx + self._tree_size
        self._sum_tree[tree_idx] = priority
        self._min_tree[tree_idx] = priority
        # Propagate up
        i = tree_idx // 2
        while i >= 1:
            self._sum_tree[i] = self._sum_tree[2*i] + self._sum_tree[2*i+1]
            self._min_tree[i] = min(self._min_tree[2*i], self._min_tree[2*i+1])
            i //= 2

    def _tree_find(self, val: float) -> int:
        idx = 1
        while idx < self._tree_size:
            left = 2 * idx
            if self._sum_tree[left] >= val:
                idx = left
            else:
                val -= self._sum_tree[left]
                idx = left + 1
        return idx - self._tree_size

    def add(self, obs, action, reward, next_obs, done) -> None:
        idx = self._ptr % self.capacity
        self.obs[idx]      = obs
        self.next_obs[idx] = next_obs
        self.actions[idx]  = action
        self.rewards[idx]  = reward
        self.dones[idx]    = float(done)

        if self.config.use_per:
            p = self._max_priority ** self.config.per_alpha
            self._tree_update(idx, p)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        if self.config.use_per:
            return self._sample_per(batch_size)
        else:
            return self._sample_uniform(batch_size)

    def _sample_per(self, batch_size: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        total = float(self._sum_tree[1])
        segment = total / batch_size

        for i in range(batch_size):
            val = np.random.uniform(segment * i, segment * (i + 1))
            idx = self._tree_find(val)
            idx = max(0, min(idx, self._size - 1))
            indices[i] = idx
            priorities[i] = self._sum_tree[idx + self._tree_size]

        min_p = float(self._min_tree[1]) / total
        max_w = (min_p * self._size) ** (-self.beta)
        weights = ((priorities / total) * self._size) ** (-self.beta) / (max_w + 1e-12)
        weights = weights.astype(np.float32)
        self.beta = min(self.config.per_beta_end, self.beta + self._beta_increment)

        return self._to_tensors(indices, weights), indices, weights

    def _sample_uniform(self, batch_size: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        indices = np.random.randint(0, self._size, size=batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        return self._to_tensors(indices, weights), indices, weights

    def _to_tensors(self, indices: np.ndarray, weights: np.ndarray) -> Dict[str, torch.Tensor]:
        return {
            "obs":      torch.FloatTensor(self.obs[indices]).to(self.device),
            "next_obs": torch.FloatTensor(self.next_obs[indices]).to(self.device),
            "actions":  torch.LongTensor(self.actions[indices]).to(self.device),
            "rewards":  torch.FloatTensor(self.rewards[indices]).to(self.device),
            "dones":    torch.FloatTensor(self.dones[indices]).to(self.device),
            "weights":  torch.FloatTensor(weights).to(self.device),
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        if not self.config.use_per:
            return
        priorities = (np.abs(td_errors) + 1e-6) ** self.config.per_alpha
        for idx, p in zip(indices, priorities):
            self._tree_update(int(idx), float(p))
            self._max_priority = max(self._max_priority, float(p))

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# N-step return accumulator
# ---------------------------------------------------------------------------

class NStepAccumulator:
    """Compute n-step bootstrapped returns before storing transitions."""

    def __init__(self, n_steps: int, gamma: float):
        self.n = n_steps
        self.gamma = gamma
        self._buf: List[Tuple] = []

    def push(self, obs, action, reward, next_obs, done) -> Optional[Tuple]:
        self._buf.append((obs, action, reward, next_obs, done))
        if done:
            out = self._calc_nstep()
            self._buf.clear()
            return out
        if len(self._buf) >= self.n:
            return self._calc_nstep()
        return None

    def _calc_nstep(self) -> Tuple:
        first_obs, first_act = self._buf[0][0], self._buf[0][1]
        last_next_obs = self._buf[-1][3]
        last_done     = self._buf[-1][4]

        n_return = 0.0
        actual_done = False
        for i, (_, _, r, _, d) in enumerate(self._buf):
            n_return += (self.gamma ** i) * r
            if d:
                actual_done = True
                last_next_obs = self._buf[i][3]
                break

        if len(self._buf) > 0:
            self._buf.pop(0)
        return first_obs, first_act, n_return, last_next_obs, actual_done


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Dueling Double DQN agent.

    Action space: discrete, mapped from position sizing levels.
    """

    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.online_net = DuelingDQN(config).to(self.device)
        self.target_net = DuelingDQN(config).to(self.device)
        self._hard_update()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=config.lr, eps=1.5e-4
        )

        self.buffer = DQNReplayBuffer(config, self.device)
        self.nstep  = NStepAccumulator(config.n_steps, config.gamma)

        self._step = 0
        self._update_count = 0
        self._epsilon = config.eps_start

    def _hard_update(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _soft_update(self) -> None:
        tau = self.config.target_polyak_tau or 1.0
        for t, s in zip(self.target_net.parameters(), self.online_net.parameters()):
            t.data.copy_(tau * s.data + (1 - tau) * t.data)

    def _update_epsilon(self) -> None:
        if self._step < self.config.eps_decay_steps:
            self._epsilon = (
                self.config.eps_start
                + (self.config.eps_end - self.config.eps_start)
                * self._step / self.config.eps_decay_steps
            )
        else:
            self._epsilon = self.config.eps_end

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select action using noisy net or epsilon-greedy."""
        if self.config.use_noisy_nets:
            # NoisyNet: train mode uses noise
            self.online_net.eval()
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_vals = self.online_net(obs_t)
            action = int(q_vals.argmax(dim=-1).item())
            self.online_net.train()
        else:
            if not deterministic and np.random.random() < self._epsilon:
                action = np.random.randint(0, self.config.n_actions)
            else:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q_vals = self.online_net(obs_t)
                action = int(q_vals.argmax(dim=-1).item())
        return action

    def observe(self, obs, action, reward, next_obs, done) -> None:
        """Store n-step transition."""
        result = self.nstep.push(obs, action, reward, next_obs, done)
        if result is not None:
            self.buffer.add(*result)
        self._step += 1
        if not self.config.use_noisy_nets:
            self._update_epsilon()

    def should_update(self) -> bool:
        return len(self.buffer) >= self.config.warmup_steps

    def update(self) -> Dict[str, float]:
        """One gradient update step."""
        batch, indices, is_weights = self.buffer.sample(self.config.batch_size)

        obs      = batch["obs"]
        next_obs = batch["next_obs"]
        actions  = batch["actions"]
        rewards  = batch["rewards"]
        dones    = batch["dones"]
        weights  = batch["weights"]

        # Reset noise
        if self.config.use_noisy_nets:
            self.online_net.reset_noise()
            self.target_net.reset_noise()

        # Current Q
        q_values = self.online_net(obs)
        current_q = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Target Q (Double DQN)
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Use online net to select action, target net to evaluate
                next_q_online = self.online_net(next_obs)
                next_actions  = next_q_online.argmax(dim=-1, keepdim=True)
                next_q_target = self.target_net(next_obs)
                next_q = next_q_target.gather(1, next_actions).squeeze(-1)
            else:
                next_q = self.target_net(next_obs).max(dim=-1).values

            # N-step bootstrapped target
            gamma_n = self.config.gamma ** self.config.n_steps
            target_q = rewards + gamma_n * next_q * (1.0 - dones)

        td_errors = (current_q - target_q).detach().cpu().numpy()
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        # Update priorities
        self.buffer.update_priorities(indices, td_errors)

        # Target network update
        self._update_count += 1
        if self.config.target_polyak_tau is not None:
            self._soft_update()
        elif self._update_count % self.config.target_update_freq == 0:
            self._hard_update()

        return {
            "loss": float(loss.item()),
            "mean_q": float(current_q.mean().item()),
            "max_q": float(current_q.max().item()),
            "td_error_mean": float(np.abs(td_errors).mean()),
            "epsilon": float(self._epsilon),
            "buffer_size": len(self.buffer),
            "update_count": self._update_count,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "online_net":   self.online_net.state_dict(),
            "target_net":   self.target_net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "step":         self._step,
            "update_count": self._update_count,
            "epsilon":      self._epsilon,
            "config":       self.config,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step         = ckpt.get("step", 0)
        self._update_count = ckpt.get("update_count", 0)
        self._epsilon      = ckpt.get("epsilon", self.config.eps_end)


# ---------------------------------------------------------------------------
# Discretization helpers
# ---------------------------------------------------------------------------

class ActionDiscretizer:
    """
    Convert continuous position targets to discrete action indices and back.

    For N instruments and K levels each, total actions = K^N (full combinatorial)
    or N*K (independent per instrument).
    """

    def __init__(
        self,
        n_instruments: int,
        levels: Optional[List[float]] = None,
        mode: str = "independent",   # "independent" | "joint"
    ):
        self.n_instruments = n_instruments
        self.levels = levels or [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.n_levels = len(self.levels)
        self.mode = mode

        if mode == "independent":
            self.n_actions = n_instruments * self.n_levels
            self._action_map = self._build_independent_map()
        else:
            import itertools
            combos = list(itertools.product(range(self.n_levels), repeat=n_instruments))
            self._joint_combos = combos
            self.n_actions = len(combos)
            self._action_map = {
                i: [self.levels[c] for c in combo]
                for i, combo in enumerate(combos)
            }

    def _build_independent_map(self) -> Dict[int, List[float]]:
        """For each action index, return full position vector."""
        mapping = {}
        for i in range(self.n_instruments):
            for j, level in enumerate(self.levels):
                action_idx = i * self.n_levels + j
                pos = [0.0] * self.n_instruments
                pos[i] = level
                mapping[action_idx] = pos
        return mapping

    def decode(self, action_idx: int) -> np.ndarray:
        """Convert discrete action index to position vector."""
        return np.array(self._action_map[action_idx], dtype=np.float32)

    def encode(self, positions: np.ndarray) -> int:
        """Convert position vector to nearest discrete action."""
        if self.mode == "joint":
            # Find closest combination
            best_idx = 0
            best_dist = float("inf")
            for idx, pos in self._action_map.items():
                dist = np.sum((positions - np.array(pos)) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            return best_idx
        else:
            # Independent: pick best level per instrument
            action_idx = 0
            for i, p in enumerate(positions):
                best_j = int(np.argmin([abs(p - l) for l in self.levels]))
                action_idx = i * self.n_levels + best_j
            return action_idx


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("Testing DQN agent...")
    config = DQNConfig(
        obs_dim=64,
        n_actions=9,
        hidden_dims=[128, 128],
        use_noisy_nets=True,
        use_double_dqn=True,
        buffer_capacity=5_000,
        batch_size=32,
        warmup_steps=64,
        use_per=True,
        device="cpu",
    )
    agent = DQNAgent(config)

    obs = np.random.randn(64).astype(np.float32)
    t0 = time.time()
    metrics = {}
    for step in range(200):
        action = agent.select_action(obs)
        next_obs = np.random.randn(64).astype(np.float32)
        reward = float(np.random.randn())
        done = (step % 30 == 29)
        agent.observe(obs, action, reward, next_obs, done)
        obs = next_obs if not done else np.random.randn(64).astype(np.float32)

        if agent.should_update():
            metrics = agent.update()

    print(f"Last metrics: {metrics}")
    print(f"Time: {time.time() - t0:.2f}s")

    # Test discretizer
    disc = ActionDiscretizer(3, [-1, -0.5, 0, 0.5, 1], mode="independent")
    pos = disc.decode(2)
    print(f"Decoded action 2: {pos}")
    print("DQN self-test passed.")
