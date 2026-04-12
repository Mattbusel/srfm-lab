"""
_base_compat.py — Self-contained BaseAgent for our hand-written agent classes.

Independent from the linter-upgraded base_agent.py to avoid interface conflicts.
Our agents (MarketMaker, Momentum, Arb, Noise) inherit from THIS BaseAgent.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Running normalization (standalone, not from base_agent)
# ============================================================

class StandardNorm:
    """Online Welford running mean/variance normalization."""

    def __init__(self, shape, eps: float = 1e-8, clip: float = 10.0) -> None:
        self.shape  = shape
        self.eps    = eps
        self.clip   = clip
        self.mean   = np.zeros(shape, dtype=np.float64)
        self.var    = np.ones(shape, dtype=np.float64)
        self.count  = 0.0

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis]
        bm  = x.mean(axis=0)
        bv  = x.var(axis=0)
        bn  = x.shape[0]
        tot = self.count + bn
        d   = bm - self.mean
        self.mean  = self.mean + d * bn / tot
        M2  = self.var * self.count + bv * bn + d**2 * self.count * bn / tot
        self.var   = M2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n = (x - self.mean.astype(np.float32)) / (
            np.sqrt(self.var.astype(np.float32)) + self.eps
        )
        return np.clip(n, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> Dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: Dict) -> None:
        self.mean  = d["mean"].copy()
        self.var   = d["var"].copy()
        self.count = d["count"]


# ============================================================
# Memory / replay buffer
# ============================================================

class Memory:
    """Rolling replay buffer."""

    FIELDS = ("obs", "action_dir", "action_size", "log_prob",
              "reward", "next_obs", "done", "value")

    def __init__(self, capacity: int = 10_000) -> None:
        from collections import deque
        self.capacity = capacity
        self._buf = {f: deque(maxlen=capacity) for f in self.FIELDS}
        self._size = 0

    def push(self, obs, action_dir, action_size, log_prob,
             reward, next_obs, done, value=0.0) -> None:
        self._buf["obs"].append(obs)
        self._buf["action_dir"].append(action_dir)
        self._buf["action_size"].append(action_size)
        self._buf["log_prob"].append(log_prob)
        self._buf["reward"].append(reward)
        self._buf["next_obs"].append(next_obs)
        self._buf["done"].append(done)
        self._buf["value"].append(value)
        self._size = min(self._size + 1, self.capacity)

    def sample(self, n: int) -> Dict:
        idx = np.random.choice(self._size, size=min(n, self._size), replace=False)
        return {f: np.array([self._buf[f][i] for i in idx]) for f in self.FIELDS}

    def get_all(self) -> Dict:
        return {f: np.array(list(self._buf[f])) for f in self.FIELDS}

    def clear(self) -> None:
        for f in self.FIELDS:
            self._buf[f].clear()
        self._size = 0

    def __len__(self) -> int:
        return self._size


# ============================================================
# ObservationEncoder (standalone)
# ============================================================

class ObservationEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 64, out_dim: int = 64,
                 n_layers: int = 2, use_gru: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.out_dim = out_dim
        self.use_gru = use_gru
        layers: List[nn.Module] = []
        in_d = obs_dim
        for i in range(n_layers):
            is_last = i == n_layers - 1
            out = out_dim if is_last else hidden_dim
            layers += [nn.Linear(in_d, out), nn.LayerNorm(out), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        self.mlp = nn.Sequential(*layers)
        if use_gru:
            self.gru = nn.GRUCell(out_dim, out_dim)

    def forward(self, obs, hidden=None):
        flat   = obs.view(-1, self.obs_dim) if obs.dim() > 1 else obs.unsqueeze(0)
        latent = self.mlp(flat)
        if self.use_gru:
            h      = hidden if hidden is not None else torch.zeros_like(latent)
            latent = self.gru(latent, h)
            return latent, latent
        return latent, None


# ============================================================
# GAE
# ============================================================

def compute_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
    last_value: float = 0.0, gamma: float = 0.99, lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae        = 0.0
    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        mask     = 1.0 - float(dones[t])
        delta    = rewards[t] + gamma * next_val * mask - values[t]
        gae      = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = (advantages + values).astype(np.float32)
    return advantages.astype(np.float32), returns


# ============================================================
# BaseAgent
# ============================================================

class BaseAgent(abc.ABC):
    """
    Standalone abstract base for all hand-written Hyper-Agent agents.
    Does NOT inherit from the linter-upgraded base_agent.BaseAgent.
    """

    def __init__(
        self,
        agent_id:   str,
        obs_dim:    int,
        hidden_dim: int   = 64,
        lr:         float = 3e-4,
        gamma:      float = 0.99,
        lam:        float = 0.95,
        memory_cap: int   = 10_000,
        use_gru:    bool  = False,
        device:     str   = "cpu",
    ) -> None:
        self.agent_id   = agent_id
        self.obs_dim    = obs_dim
        self.hidden_dim = hidden_dim
        self.lr         = lr
        self.gamma      = gamma
        self.lam        = lam
        self.device     = torch.device(device)
        self.use_gru    = use_gru
        self.memory     = Memory(memory_cap)
        self.obs_norm   = StandardNorm((obs_dim,))
        self._gru_hidden = None
        self.total_steps    = 0
        self.total_episodes = 0
        self._episode_rewards: List[float] = []

    @property
    @abc.abstractmethod
    def agent_type(self) -> str: ...

    @abc.abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False
            ) -> Tuple[np.ndarray, float, float]: ...

    @abc.abstractmethod
    def update(self, batch=None) -> Dict[str, float]: ...

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(self.obs_norm.normalize(obs)).to(self.device)

    def store_transition(self, obs, action_dir, action_size, log_prob,
                         reward, next_obs, done, value=0.0) -> None:
        self.obs_norm.update(obs)
        self.memory.push(obs, action_dir, action_size, log_prob,
                         reward, next_obs, done, value)
        self.total_steps += 1
        self._episode_rewards.append(reward)
        if done:
            self.total_episodes += 1
            self._episode_rewards = []

    def receive_reward(self, reward: float, done: bool) -> float:
        self._episode_rewards.append(reward)
        return reward

    def reset_episode(self) -> None:
        self._gru_hidden = None
        self._episode_rewards = []

    def parameters(self):
        params = []
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, nn.Module):
                params.extend(list(obj.parameters()))
        return params

    def policy_kl(self, other: "BaseAgent", obs_batch: np.ndarray) -> float:
        return 0.0

    def clone(self) -> "BaseAgent":
        import copy
        return copy.deepcopy(self)

    def mutate(self, noise_std: float = 0.01) -> None:
        with torch.no_grad():
            for p in self.parameters():
                p.add_(torch.randn_like(p) * noise_std)

    def save(self, path: str) -> None:
        state = {"agent_id": self.agent_id}
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, nn.Module):
                state[name] = obj.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        for name, sd in ckpt.items():
            if hasattr(self, name) and isinstance(getattr(self, name), nn.Module):
                getattr(self, name).load_state_dict(sd)

    def get_policy_params(self) -> np.ndarray:
        params = []
        for p in self.parameters():
            params.append(p.detach().cpu().numpy().ravel())
        return np.concatenate(params) if params else np.zeros(1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id!r}, steps={self.total_steps})"


__all__ = ["BaseAgent", "ObservationEncoder", "StandardNorm", "Memory", "compute_gae"]
