"""
replay_buffer.py — Experience replay buffers for Hyper-Agent MARL.

Implements:
- UniformReplayBuffer: standard random-sample buffer
- PrioritizedReplayBuffer: proportional priority sampling (PER)
- EpisodeReplayBuffer: full episode storage with trajectory sampling
- MultiAgentReplayBuffer: per-agent or joint storage
- HindsightReplayBuffer: HER for goal-conditioned RL
"""

from __future__ import annotations

import math
import logging
import collections
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

EPS = 1e-8


# ---------------------------------------------------------------------------
# Segment tree for PER
# ---------------------------------------------------------------------------

class SumSegmentTree:
    """Segment tree for O(log N) sum queries and updates."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    def update(self, idx: int, value: float) -> None:
        idx += self.capacity
        self.tree[idx] = value
        while idx > 1:
            idx //= 2
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def query(self, left: int, right: int) -> float:
        """Sum in [left, right)."""
        result = 0.0
        left += self.capacity
        right += self.capacity
        while left < right:
            if left % 2 == 1:
                result += self.tree[left]
                left += 1
            if right % 2 == 1:
                right -= 1
                result += self.tree[right]
            left //= 2
            right //= 2
        return result

    def total(self) -> float:
        return float(self.tree[1])

    def find(self, value: float) -> int:
        """Find idx such that prefix_sum(0..idx) >= value."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] >= value:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - self.capacity


class MinSegmentTree:
    """Segment tree for O(log N) min queries."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.full(2 * capacity, float("inf"), dtype=np.float64)

    def update(self, idx: int, value: float) -> None:
        idx += self.capacity
        self.tree[idx] = value
        while idx > 1:
            idx //= 2
            self.tree[idx] = min(self.tree[2 * idx], self.tree[2 * idx + 1])

    def min(self) -> float:
        return float(self.tree[1])


# ---------------------------------------------------------------------------
# Uniform replay buffer
# ---------------------------------------------------------------------------

class UniformReplayBuffer:
    """
    Standard circular replay buffer.
    Stores (obs, action, reward, next_obs, done) tuples.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        extra_keys: Optional[List[str]] = None,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.extra_keys = extra_keys or []

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

        self._extras: Dict[str, np.ndarray] = {}
        for key in self.extra_keys:
            self._extras[key] = np.zeros(capacity, dtype=np.float32)

        self._pos = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs,
    ) -> None:
        self._obs[self._pos] = obs
        self._actions[self._pos] = action
        self._rewards[self._pos] = float(reward)
        self._next_obs[self._pos] = next_obs
        self._dones[self._pos] = float(done)
        for key in self.extra_keys:
            if key in kwargs:
                self._extras[key][self._pos] = kwargs[key]
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        assert self._size >= batch_size, f"Buffer too small: {self._size} < {batch_size}"
        indices = np.random.randint(0, self._size, batch_size)
        return self._get_batch(indices, device)

    def _get_batch(
        self, indices: np.ndarray, device: Optional[torch.device]
    ) -> Dict[str, torch.Tensor]:
        def t(x):
            return torch.tensor(x, device=device)

        batch = {
            "obs": t(self._obs[indices]),
            "actions": t(self._actions[indices]),
            "rewards": t(self._rewards[indices]),
            "next_obs": t(self._next_obs[indices]),
            "dones": t(self._dones[indices]),
        }
        for key in self.extra_keys:
            batch[key] = t(self._extras[key][indices])
        return batch

    def __len__(self) -> int:
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def clear(self) -> None:
        self._pos = 0
        self._size = 0


# ---------------------------------------------------------------------------
# Prioritized experience replay
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer(UniformReplayBuffer):
    """
    Prioritized Experience Replay (PER).
    Samples transitions proportional to their TD error priority.

    Priority: p_i = |delta_i| + epsilon
    Probability: P(i) = p_i^alpha / sum_j(p_j^alpha)
    Importance weight: w_i = (N * P(i))^{-beta} / max_j(w_j)
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100_000,
        priority_eps: float = 1e-6,
        extra_keys: Optional[List[str]] = None,
    ):
        # Round capacity to power of 2
        cap = 1
        while cap < capacity:
            cap *= 2
        super().__init__(cap, obs_dim, action_dim, extra_keys)

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.priority_eps = priority_eps
        self._beta_step = 0

        self._sum_tree = SumSegmentTree(cap)
        self._min_tree = MinSegmentTree(cap)
        self._max_priority = 1.0

    @property
    def beta(self) -> float:
        frac = min(self._beta_step / max(self.beta_steps, 1), 1.0)
        return self.beta_start + (self.beta_end - self.beta_start) * frac

    def add(self, obs, action, reward, next_obs, done, **kwargs) -> None:
        idx = self._pos
        super().add(obs, action, reward, next_obs, done, **kwargs)
        priority = self._max_priority ** self.alpha
        self._sum_tree.update(idx, priority)
        self._min_tree.update(idx, priority)

    def sample(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        assert self._size >= batch_size

        indices = []
        total = self._sum_tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            val = np.random.uniform(lo, hi)
            idx = self._sum_tree.find(val)
            idx = min(idx, self._size - 1)
            indices.append(idx)

        # Importance weights
        min_p = self._min_tree.min() / (total + EPS)
        max_w = (self._size * min_p) ** (-self.beta)

        weights = []
        for idx in indices:
            p = self._sum_tree.query(idx, idx + 1) / (total + EPS)
            w = (self._size * p) ** (-self.beta)
            weights.append(w / max_w)

        self._beta_step += 1
        indices_arr = np.array(indices)
        batch = self._get_batch(indices_arr, device)
        batch["weights"] = torch.tensor(
            np.array(weights, dtype=np.float32), device=device
        )
        batch["indices"] = torch.tensor(indices_arr, device=device)
        return batch

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priority for sampled transitions based on new TD errors."""
        for idx, pri in zip(indices, priorities):
            idx = int(idx)
            pri = max(abs(float(pri)), self.priority_eps)
            self._max_priority = max(self._max_priority, pri)
            p = (pri + self.priority_eps) ** self.alpha
            self._sum_tree.update(idx, p)
            self._min_tree.update(idx, p)


# ---------------------------------------------------------------------------
# Episode replay buffer
# ---------------------------------------------------------------------------

class EpisodeReplayBuffer:
    """
    Episode-level replay buffer.
    Stores complete episodes for sequence-based algorithms (QMIX, R2D2).
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int = 0,
        max_episode_len: int = 500,
        num_agents: int = 1,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_episode_len = max_episode_len
        self.num_agents = num_agents

        self._episodes: List[Dict] = []
        self._pos = 0

        # Current episode being built
        self._current_ep: Dict[str, List] = self._new_episode()

    def _new_episode(self) -> Dict[str, List]:
        d: Dict[str, List] = {
            "obs": [[] for _ in range(self.num_agents)],
            "actions": [[] for _ in range(self.num_agents)],
            "rewards": [[] for _ in range(self.num_agents)],
            "dones": [[] for _ in range(self.num_agents)],
            "next_obs": [[] for _ in range(self.num_agents)],
        }
        if self.state_dim > 0:
            d["global_states"] = []
        return d

    def add_step(
        self,
        obs_list: List[np.ndarray],
        action_list: List[np.ndarray],
        reward_list: List[float],
        done_list: List[bool],
        next_obs_list: List[np.ndarray],
        global_state: Optional[np.ndarray] = None,
    ) -> None:
        for i in range(self.num_agents):
            self._current_ep["obs"][i].append(obs_list[i].copy())
            self._current_ep["actions"][i].append(action_list[i].copy())
            self._current_ep["rewards"][i].append(float(reward_list[i]))
            self._current_ep["dones"][i].append(bool(done_list[i]))
            self._current_ep["next_obs"][i].append(next_obs_list[i].copy())
        if global_state is not None and self.state_dim > 0:
            self._current_ep["global_states"].append(global_state.copy())

    def end_episode(self) -> None:
        """Finalize and store the current episode."""
        ep = self._current_ep
        ep_len = len(ep["rewards"][0])
        if ep_len == 0:
            self._current_ep = self._new_episode()
            return

        if len(self._episodes) < self.capacity:
            self._episodes.append(ep)
        else:
            self._episodes[self._pos] = ep
        self._pos = (self._pos + 1) % self.capacity
        self._current_ep = self._new_episode()

    def sample_episodes(self, batch_size: int) -> List[Dict]:
        n = len(self._episodes)
        if n == 0:
            return []
        indices = np.random.randint(0, n, batch_size)
        return [self._episodes[i] for i in indices]

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Sample a batch of fixed-length subsequences from episodes.
        """
        n = len(self._episodes)
        if n == 0:
            return None

        batch_obs = np.zeros((batch_size, seq_len, self.num_agents, self.obs_dim), dtype=np.float32)
        batch_acts = np.zeros((batch_size, seq_len, self.num_agents, self.action_dim), dtype=np.float32)
        batch_rews = np.zeros((batch_size, seq_len, self.num_agents), dtype=np.float32)
        batch_dones = np.zeros((batch_size, seq_len, self.num_agents), dtype=np.float32)
        batch_masks = np.zeros((batch_size, seq_len), dtype=np.float32)
        batch_states = np.zeros((batch_size, seq_len, max(self.state_dim, 1)), dtype=np.float32)

        for b in range(batch_size):
            ep = self._episodes[np.random.randint(n)]
            ep_len = len(ep["rewards"][0])
            if ep_len <= seq_len:
                start = 0
                actual_len = ep_len
            else:
                start = np.random.randint(0, ep_len - seq_len + 1)
                actual_len = seq_len

            batch_masks[b, :actual_len] = 1.0

            for t in range(actual_len):
                ts = start + t
                for i in range(self.num_agents):
                    if ts < len(ep["obs"][i]):
                        batch_obs[b, t, i] = ep["obs"][i][ts]
                        batch_acts[b, t, i] = ep["actions"][i][ts]
                        batch_rews[b, t, i] = ep["rewards"][i][ts]
                        batch_dones[b, t, i] = float(ep["dones"][i][ts])
                if self.state_dim > 0 and "global_states" in ep and ts < len(ep["global_states"]):
                    batch_states[b, t] = ep["global_states"][ts]

        def t(x):
            return torch.tensor(x, device=device)

        return {
            "obs": t(batch_obs),
            "actions": t(batch_acts),
            "rewards": t(batch_rews),
            "dones": t(batch_dones),
            "masks": t(batch_masks),
            "global_states": t(batch_states) if self.state_dim > 0 else None,
        }

    def __len__(self) -> int:
        return len(self._episodes)

    def current_episode_len(self) -> int:
        return len(self._current_ep["rewards"][0]) if self._current_ep["rewards"] else 0


# ---------------------------------------------------------------------------
# Multi-agent replay buffer
# ---------------------------------------------------------------------------

class MultiAgentReplayBuffer:
    """
    Separate replay buffers for each agent, with shared global state.
    Also supports joint sampling for centralized training.
    """

    def __init__(
        self,
        num_agents: int,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int = 0,
        use_per: bool = False,
        per_alpha: float = 0.6,
    ):
        self.num_agents = num_agents
        self.capacity = capacity
        self.state_dim = state_dim
        self.use_per = use_per

        if use_per:
            self._buffers = [
                PrioritizedReplayBuffer(capacity, obs_dim, action_dim, alpha=per_alpha)
                for _ in range(num_agents)
            ]
        else:
            self._buffers = [
                UniformReplayBuffer(capacity, obs_dim, action_dim)
                for _ in range(num_agents)
            ]

        # Shared global state buffer
        if state_dim > 0:
            self._state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
            self._next_state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
            self._state_pos = 0

    def add(
        self,
        obs_list: List[np.ndarray],
        action_list: List[np.ndarray],
        reward_list: List[float],
        next_obs_list: List[np.ndarray],
        done_list: List[bool],
        global_state: Optional[np.ndarray] = None,
        next_global_state: Optional[np.ndarray] = None,
    ) -> None:
        for i in range(self.num_agents):
            self._buffers[i].add(
                obs_list[i], action_list[i], reward_list[i],
                next_obs_list[i], done_list[i],
            )
        if self.state_dim > 0 and global_state is not None:
            self._state_buf[self._state_pos] = global_state
            if next_global_state is not None:
                self._next_state_buf[self._state_pos] = next_global_state
            self._state_pos = (self._state_pos + 1) % self.capacity

    def sample_agent(
        self, agent_id: int, batch_size: int, device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        return self._buffers[agent_id].sample(batch_size, device)

    def sample_all(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> List[Dict[str, torch.Tensor]]:
        return [self.sample_agent(i, batch_size, device) for i in range(self.num_agents)]

    def is_ready(self, batch_size: int) -> bool:
        return all(len(b) >= batch_size for b in self._buffers)

    def update_priorities(
        self, agent_id: int, indices: np.ndarray, priorities: np.ndarray
    ) -> None:
        if self.use_per:
            self._buffers[agent_id].update_priorities(indices, priorities)

    def __len__(self) -> int:
        return len(self._buffers[0]) if self._buffers else 0


# ---------------------------------------------------------------------------
# Hindsight experience replay
# ---------------------------------------------------------------------------

class HindsightReplayBuffer(UniformReplayBuffer):
    """
    Hindsight Experience Replay (HER).
    Relabels episodes with achieved goals to create additional learning signal.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        goal_dim: int,
        strategy: str = "future",
        k: int = 4,
    ):
        super().__init__(capacity, obs_dim + goal_dim, action_dim)
        self.goal_dim = goal_dim
        self.obs_only_dim = obs_dim
        self.strategy = strategy
        self.k = k  # Number of HER transitions per real transition

        # Episode storage for relabeling
        self._ep_obs: List[np.ndarray] = []
        self._ep_actions: List[np.ndarray] = []
        self._ep_achieved_goals: List[np.ndarray] = []
        self._ep_desired_goals: List[np.ndarray] = []
        self._ep_rewards: List[float] = []
        self._ep_dones: List[bool] = []

    def add_episode_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self._ep_obs.append(obs)
        self._ep_actions.append(action)
        self._ep_achieved_goals.append(achieved_goal)
        self._ep_desired_goals.append(desired_goal)
        self._ep_rewards.append(reward)
        self._ep_dones.append(done)

    def end_episode(
        self, compute_reward_fn=None
    ) -> None:
        """
        Finalize episode and store real + HER transitions.
        compute_reward_fn: (achieved_goal, desired_goal) -> reward
        """
        ep_len = len(self._ep_obs)

        for t in range(ep_len):
            obs_goal = np.concatenate([self._ep_obs[t], self._ep_desired_goals[t]])
            next_t = min(t + 1, ep_len - 1)
            next_obs_goal = np.concatenate([self._ep_obs[next_t], self._ep_desired_goals[t]])
            super().add(
                obs_goal, self._ep_actions[t], self._ep_rewards[t],
                next_obs_goal, self._ep_dones[t]
            )

            # HER relabeling
            if self.strategy == "future":
                future_indices = list(range(t + 1, ep_len))
                if not future_indices:
                    continue
                her_indices = np.random.choice(
                    future_indices,
                    size=min(self.k, len(future_indices)),
                    replace=False,
                )
            elif self.strategy == "episode":
                her_indices = np.random.choice(ep_len, size=self.k, replace=True)
            elif self.strategy == "final":
                her_indices = [ep_len - 1] * self.k
            else:
                her_indices = []

            for her_t in her_indices:
                her_goal = self._ep_achieved_goals[her_t]
                if compute_reward_fn is not None:
                    her_reward = compute_reward_fn(self._ep_achieved_goals[next_t], her_goal)
                else:
                    her_reward = float(
                        np.linalg.norm(self._ep_achieved_goals[next_t] - her_goal) < 0.05
                    )
                her_obs_goal = np.concatenate([self._ep_obs[t], her_goal])
                her_next_obs_goal = np.concatenate([self._ep_obs[next_t], her_goal])
                super().add(
                    her_obs_goal, self._ep_actions[t], her_reward,
                    her_next_obs_goal, bool(her_t == ep_len - 1)
                )

        # Clear episode storage
        self._ep_obs.clear()
        self._ep_actions.clear()
        self._ep_achieved_goals.clear()
        self._ep_desired_goals.clear()
        self._ep_rewards.clear()
        self._ep_dones.clear()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "SumSegmentTree",
    "MinSegmentTree",
    "UniformReplayBuffer",
    "PrioritizedReplayBuffer",
    "EpisodeReplayBuffer",
    "MultiAgentReplayBuffer",
    "HindsightReplayBuffer",
]
