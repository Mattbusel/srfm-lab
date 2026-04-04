"""
Soft Actor-Critic (SAC) with:
- Twin Q-networks (clipped double-Q)
- Target networks with Polyak averaging
- Automatic entropy temperature tuning (alpha)
- Replay buffer (1M transitions)
- Prioritized Experience Replay (PER) with importance sampling
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
    TORCH_AVAILABLE = True
except ImportError:
    raise ImportError("PyTorch is required for SAC.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SACConfig:
    obs_dim: int = 256
    act_dim: int = 4

    # Network
    actor_hidden: List[int] = field(default_factory=lambda: [512, 256, 256])
    critic_hidden: List[int] = field(default_factory=lambda: [512, 256, 256])
    activation: str = "relu"
    use_layer_norm: bool = True

    # SAC hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    polyak_tau: float = 0.005
    batch_size: int = 256
    reward_scale: float = 1.0

    # Entropy target
    target_entropy: Optional[float] = None  # set to -act_dim if None
    init_alpha: float = 0.2
    auto_alpha: bool = True

    # Replay buffer
    buffer_capacity: int = 1_000_000
    use_per: bool = True
    per_alpha: float = 0.6        # priority exponent
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 200_000

    # Training
    warmup_steps: int = 1000
    update_freq: int = 1
    updates_per_step: int = 1
    gradient_steps: int = 1

    # Output bounds
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    device: str = "cpu"


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer
# ---------------------------------------------------------------------------

class SegmentTree:
    """Sum/Min segment tree for O(log n) PER operations."""

    def __init__(self, capacity: int, operation, neutral_element: float):
        self.capacity = capacity
        self.op = operation
        self.neutral = neutral_element
        self._tree = [neutral_element] * (2 * capacity)

    def _propagate(self, idx: int) -> None:
        parent = idx // 2
        while parent >= 1:
            self._tree[parent] = self.op(self._tree[2 * parent], self._tree[2 * parent + 1])
            parent //= 2

    def update(self, idx: int, val: float) -> None:
        tree_idx = idx + self.capacity
        self._tree[tree_idx] = val
        self._propagate(tree_idx)

    def query(self, start: int, end: int) -> float:
        """Query [start, end) in the original array."""
        result = self.neutral
        start += self.capacity
        end   += self.capacity
        while start < end:
            if start & 1:
                result = self.op(result, self._tree[start])
                start += 1
            if end & 1:
                end -= 1
                result = self.op(result, self._tree[end])
            start //= 2
            end   //= 2
        return result

    def total(self) -> float:
        return self._tree[1]

    def __getitem__(self, idx: int) -> float:
        return self._tree[idx + self.capacity]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Supports O(log n) sample and update operations via segment tree.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 200_000,
        device: Optional[torch.device] = None,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_increment = (beta_end - beta_start) / max(beta_steps, 1)
        self.device = device or torch.device("cpu")

        self._ptr = 0
        self._size = 0
        self._max_priority = 1.0

        # Storage
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)

        # Segment trees for sum and min
        self._sum_tree = SegmentTree(capacity, lambda a, b: a + b, 0.0)
        self._min_tree = SegmentTree(capacity, min, float("inf"))

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        idx = self._ptr % self.capacity
        self.obs[idx]      = obs
        self.next_obs[idx] = next_obs
        self.actions[idx]  = action
        self.rewards[idx]  = reward
        self.dones[idx]    = float(done)

        p = (priority if priority is not None else self._max_priority) ** self.alpha
        self._sum_tree.update(idx, p)
        self._min_tree.update(idx, p)
        self._max_priority = max(self._max_priority, p)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample a batch with priority-proportional probability.
        Returns: (batch_dict, indices, importance_weights)
        """
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        total = self._sum_tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            val = np.random.uniform(lo, hi)
            idx = self._find_prefix_sum(val)
            indices[i] = idx
            priorities[i] = self._sum_tree[idx]

        # Importance sampling weights
        min_p = self._min_tree.total() / (self._size + 1e-12)
        max_w = (min_p * self._size) ** (-self.beta)
        weights = ((priorities / total) * self._size) ** (-self.beta) / (max_w + 1e-12)
        weights = weights.astype(np.float32)

        batch = {
            "obs":      torch.FloatTensor(self.obs[indices]).to(self.device),
            "next_obs": torch.FloatTensor(self.next_obs[indices]).to(self.device),
            "actions":  torch.FloatTensor(self.actions[indices]).to(self.device),
            "rewards":  torch.FloatTensor(self.rewards[indices]).to(self.device),
            "dones":    torch.FloatTensor(self.dones[indices]).to(self.device),
            "weights":  torch.FloatTensor(weights).to(self.device),
        }

        self.beta = min(self.beta_end, self.beta + self.beta_increment)

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on new TD errors."""
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        for idx, p in zip(indices, priorities):
            self._sum_tree.update(idx, float(p))
            self._min_tree.update(idx, float(p))
        self._max_priority = max(self._max_priority, float(priorities.max()))

    def _find_prefix_sum(self, val: float) -> int:
        """Binary search for index where prefix sum >= val."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if self._sum_tree._tree[left] >= val:
                idx = left
            else:
                val -= self._sum_tree._tree[left]
                idx = right
        return idx - self.capacity

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size > 0


class UniformReplayBuffer:
    """Standard uniform replay buffer (fallback if PER not used)."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: Optional[torch.device] = None):
        self.capacity = capacity
        self.device = device or torch.device("cpu")
        self._ptr = 0
        self._size = 0

        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done, **kwargs) -> None:
        idx = self._ptr % self.capacity
        self.obs[idx]      = obs
        self.next_obs[idx] = next_obs
        self.actions[idx]  = action
        self.rewards[idx]  = reward
        self.dones[idx]    = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self._size, size=batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        batch = {
            "obs":      torch.FloatTensor(self.obs[idx]).to(self.device),
            "next_obs": torch.FloatTensor(self.next_obs[idx]).to(self.device),
            "actions":  torch.FloatTensor(self.actions[idx]).to(self.device),
            "rewards":  torch.FloatTensor(self.rewards[idx]).to(self.device),
            "dones":    torch.FloatTensor(self.dones[idx]).to(self.device),
            "weights":  torch.FloatTensor(weights).to(self.device),
        }
        return batch, idx, weights

    def update_priorities(self, indices, td_errors) -> None:
        pass  # no-op for uniform buffer

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size > 0


# ---------------------------------------------------------------------------
# Network components
# ---------------------------------------------------------------------------

def _mlp(input_dim: int, hidden_dims: List[int], output_dim: int,
         activation: str = "relu", use_layer_norm: bool = True) -> nn.Sequential:
    act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "gelu": nn.GELU}.get(activation, nn.ReLU)
    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act_fn())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    net = nn.Sequential(*layers)
    # Orthogonal init
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    # Small gain for last layer
    if isinstance(net[-1], nn.Linear):
        nn.init.orthogonal_(net[-1].weight, gain=0.01)
    return net


class SACActor(nn.Module):
    """Squashed Gaussian actor for SAC."""

    def __init__(self, config: SACConfig):
        super().__init__()
        self.config = config
        self.net = _mlp(
            config.obs_dim, config.actor_hidden,
            config.act_dim * 2,  # mean + log_std
            config.activation, config.use_layer_norm,
        )

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (action, log_prob), both (batch, act_dim) and (batch,)
        """
        out = self.net(obs)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)
        std = log_std.exp()

        dist = Independent(Normal(mean, std), 1)
        x = dist.rsample()
        y = torch.tanh(x)

        # Log prob with tanh correction
        log_prob = dist.log_prob(x)
        tanh_corr = (2.0 * (math.log(2) - x - F.softplus(-2.0 * x))).sum(dim=-1)
        log_prob = log_prob - tanh_corr

        return y, log_prob

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(obs)
        mean, _ = out.chunk(2, dim=-1)
        return torch.tanh(mean)

    def get_action_with_entropy(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob, entropy)."""
        out = self.net(obs)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)
        std = log_std.exp()
        dist = Independent(Normal(mean, std), 1)
        x = dist.rsample()
        y = torch.tanh(x)
        log_prob = dist.log_prob(x) - (2.0 * (math.log(2) - x - F.softplus(-2.0 * x))).sum(-1)
        entropy = dist.entropy()
        return y, log_prob, entropy


class SACCritic(nn.Module):
    """Twin Q-networks for SAC."""

    def __init__(self, config: SACConfig):
        super().__init__()
        input_dim = config.obs_dim + config.act_dim
        self.q1 = _mlp(input_dim, config.critic_hidden, 1, config.activation, config.use_layer_norm)
        self.q2 = _mlp(input_dim, config.critic_hidden, 1, config.activation, config.use_layer_norm)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (q1, q2) both shape (batch, 1)."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SACAgent:
    """
    Full Soft Actor-Critic agent.

    Features:
    - Twin Q-networks with target networks (Polyak averaging)
    - Automatic entropy temperature tuning
    - Prioritized Experience Replay
    - Gradient clipping
    """

    def __init__(self, config: SACConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Networks
        self.actor  = SACActor(config).to(self.device)
        self.critic = SACCritic(config).to(self.device)
        self.critic_target = SACCritic(config).to(self.device)
        self._hard_update(self.critic_target, self.critic)
        self.critic_target.eval()

        # Entropy temperature (alpha)
        self.log_alpha = torch.tensor(
            math.log(config.init_alpha), dtype=torch.float32,
            device=self.device, requires_grad=True
        )
        if config.target_entropy is None:
            self.target_entropy = -float(config.act_dim)
        else:
            self.target_entropy = config.target_entropy

        # Optimizers
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        self.alpha_optimizer  = torch.optim.Adam([self.log_alpha], lr=config.lr_alpha)

        # Replay buffer
        if config.use_per:
            self.buffer: PrioritizedReplayBuffer | UniformReplayBuffer = PrioritizedReplayBuffer(
                capacity=config.buffer_capacity,
                obs_dim=config.obs_dim,
                act_dim=config.act_dim,
                alpha=config.per_alpha,
                beta_start=config.per_beta_start,
                beta_end=config.per_beta_end,
                beta_steps=config.per_beta_steps,
                device=self.device,
            )
        else:
            self.buffer = UniformReplayBuffer(
                capacity=config.buffer_capacity,
                obs_dim=config.obs_dim,
                act_dim=config.act_dim,
                device=self.device,
            )

        self._step = 0
        self._update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _hard_update(self, target: nn.Module, source: nn.Module) -> None:
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(tau * s.data + (1.0 - tau) * t.data)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action for environment step."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if deterministic:
            action = self.actor.get_deterministic_action(obs_t)
        else:
            action, _ = self.actor(obs_t)
        return action.squeeze(0).cpu().numpy()

    def observe(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition to replay buffer."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self._step += 1

    def should_update(self) -> bool:
        return (
            len(self.buffer) >= self.config.warmup_steps
            and self._step % self.config.update_freq == 0
        )

    def update(self) -> Dict[str, float]:
        """Perform one gradient update step."""
        batch, indices, is_weights = self.buffer.sample(self.config.batch_size)

        obs      = batch["obs"]
        next_obs = batch["next_obs"]
        actions  = batch["actions"]
        rewards  = batch["rewards"]
        dones    = batch["dones"]
        weights  = batch["weights"]

        # ---- Critic update ----
        with torch.no_grad():
            next_actions, next_log_pis = self.actor(next_obs)
            next_q1, next_q2 = self.critic_target(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_pis.unsqueeze(-1)
            target_q = (
                self.config.reward_scale * rewards.unsqueeze(-1)
                + (1.0 - dones.unsqueeze(-1)) * self.config.gamma * next_q
            )

        current_q1, current_q2 = self.critic(obs, actions)
        td_errors1 = (current_q1.squeeze(-1) - target_q.squeeze(-1)).detach().cpu().numpy()
        td_errors2 = (current_q2.squeeze(-1) - target_q.squeeze(-1)).detach().cpu().numpy()
        td_errors  = 0.5 * (np.abs(td_errors1) + np.abs(td_errors2))

        critic_loss1 = (weights * F.mse_loss(current_q1.squeeze(-1), target_q.squeeze(-1), reduction="none")).mean()
        critic_loss2 = (weights * F.mse_loss(current_q2.squeeze(-1), target_q.squeeze(-1), reduction="none")).mean()
        critic_loss  = critic_loss1 + critic_loss2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update priorities
        self.buffer.update_priorities(indices, td_errors)

        # ---- Actor update ----
        new_actions, log_pis = self.actor(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha.detach() * log_pis - q_new.squeeze(-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ---- Alpha update ----
        alpha_loss = torch.tensor(0.0)
        if self.config.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # ---- Polyak update of target ----
        self._soft_update(self.critic_target, self.critic, self.config.polyak_tau)

        self._update_count += 1

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss":  float(actor_loss.item()),
            "alpha_loss":  float(alpha_loss.item()),
            "alpha":       float(self.alpha.item()),
            "mean_q":      float(q_new.mean().item()),
            "log_pi":      float(log_pis.mean().item()),
            "td_error":    float(td_errors.mean()),
            "buffer_size": len(self.buffer),
        }

    def update_n_times(self, n: int) -> Dict[str, float]:
        """Run n gradient updates, return averaged metrics."""
        all_metrics: Dict[str, List[float]] = {}
        for _ in range(n):
            m = self.update()
            for k, v in m.items():
                all_metrics.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in all_metrics.items()}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "actor":          self.actor.state_dict(),
            "critic":         self.critic.state_dict(),
            "critic_target":  self.critic_target.state_dict(),
            "log_alpha":      self.log_alpha.data,
            "actor_opt":      self.actor_optimizer.state_dict(),
            "critic_opt":     self.critic_optimizer.state_dict(),
            "alpha_opt":      self.alpha_optimizer.state_dict(),
            "step":           self._step,
            "update_count":   self._update_count,
            "config":         self.config,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_opt"])
        self._step = ckpt.get("step", 0)
        self._update_count = ckpt.get("update_count", 0)


# ---------------------------------------------------------------------------
# Multi-step returns wrapper
# ---------------------------------------------------------------------------

class NStepBuffer:
    """Accumulate n-step returns before storing in replay buffer."""

    def __init__(self, n_steps: int, gamma: float):
        self.n_steps = n_steps
        self.gamma = gamma
        self._buf: List[Tuple] = []

    def add(self, obs, action, reward, next_obs, done) -> Optional[Tuple]:
        """Add transition. Returns completed n-step transition if ready."""
        self._buf.append((obs, action, reward, next_obs, done))
        if done:
            # Flush remaining buffer
            return self._drain_all()
        if len(self._buf) >= self.n_steps:
            return self._compute_nstep()
        return None

    def _compute_nstep(self) -> Tuple:
        """Compute n-step return for oldest transition."""
        obs, action = self._buf[0][0], self._buf[0][1]
        next_obs_n = self._buf[-1][3]
        done_n = self._buf[-1][4]

        n_return = 0.0
        for i, (_, _, r, _, d) in enumerate(self._buf):
            n_return += (self.gamma ** i) * r
            if d:
                done_n = True
                break

        self._buf.pop(0)
        return obs, action, n_return, next_obs_n, done_n

    def _drain_all(self) -> Optional[Tuple]:
        """Return first n-step transition and clear buffer."""
        if not self._buf:
            return None
        return self._compute_nstep()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("Testing SAC agent...")
    config = SACConfig(
        obs_dim=64,
        act_dim=3,
        actor_hidden=[128, 128],
        critic_hidden=[128, 128],
        buffer_capacity=10_000,
        batch_size=32,
        warmup_steps=64,
        use_per=True,
        device="cpu",
    )
    agent = SACAgent(config)

    # Fill buffer
    obs = np.random.randn(64).astype(np.float32)
    t0 = time.time()
    for step in range(200):
        action = agent.select_action(obs)
        next_obs = np.random.randn(64).astype(np.float32)
        reward = float(np.random.randn())
        done = (step % 50 == 49)
        agent.observe(obs, action, reward, next_obs, done)
        obs = next_obs if not done else np.random.randn(64).astype(np.float32)

        if agent.should_update():
            metrics = agent.update()

    print(f"Last metrics: {metrics}")
    print(f"Buffer size: {len(agent.buffer)}")
    print(f"Alpha: {agent.alpha.item():.4f}")
    print(f"Time: {time.time() - t0:.2f}s")
    print("SAC self-test passed.")
