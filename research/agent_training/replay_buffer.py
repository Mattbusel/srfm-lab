"""
research/agent_training/replay_buffer.py

Replay buffers for off-policy and on-policy RL agents.

Implementations:
    - ReplayBuffer        : standard uniform experience replay
    - PrioritizedReplayBuffer : Proportional PER via sum-tree (O(log n) ops)
    - EpisodeBuffer       : on-policy rollout buffer for PPO / A2C
    - HindsightReplayBuffer : HER for goal-conditioned learning
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Batch dataclass
# ---------------------------------------------------------------------------


@dataclass
class Batch:
    """
    Unified batch container returned by all replay buffers.

    Attributes:
        obs         : (B, obs_dim) float64
        actions     : (B, action_dim) or (B,) float64
        rewards     : (B,) float64
        next_obs    : (B, obs_dim) float64
        dones       : (B,) bool
        weights     : (B,) float64  — IS weights (1.0 for uniform buffers)
        indices     : (B,) int64    — buffer indices (for PER priority update)
        log_probs   : optional (B,) float64 — for PPO
        advantages  : optional (B,) float64 — for PPO
        returns     : optional (B,) float64 — discounted returns for PPO
    """

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    weights: np.ndarray
    indices: np.ndarray
    log_probs: Optional[np.ndarray] = None
    advantages: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.rewards)

    def to_dict(self) -> dict:
        return {
            "obs": self.obs,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_obs": self.next_obs,
            "dones": self.dones,
            "weights": self.weights,
            "indices": self.indices,
        }


# ---------------------------------------------------------------------------
# ReplayBuffer — uniform experience replay
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """
    Standard circular replay buffer with uniform random sampling.

    Stores (obs, action, reward, next_obs, done) tuples up to `capacity`.
    When full, oldest experiences are overwritten.

    Args:
        capacity   : Maximum number of transitions to store.
        obs_dim    : Dimensionality of the observation space.
        action_dim : Dimensionality of the action space (use 1 for discrete).
        seed       : Optional RNG seed.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(seed)

        # Pre-allocated arrays
        self._obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.bool_)

        self._ptr: int = 0
        self._size: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray | float,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a single transition.

        Args:
            obs      : Current observation, shape (obs_dim,).
            action   : Action taken. Scalar or array of shape (action_dim,).
            reward   : Scalar reward signal.
            next_obs : Next observation, shape (obs_dim,).
            done     : Episode termination flag.
        """
        self._obs[self._ptr] = np.asarray(obs, dtype=np.float32).reshape(-1)[: self.obs_dim]
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size == 1 and self.action_dim > 1:
            action_arr = np.full(self.action_dim, action_arr[0], dtype=np.float32)
        self._actions[self._ptr] = action_arr[: self.action_dim]
        self._rewards[self._ptr] = float(reward)
        self._next_obs[self._ptr] = np.asarray(next_obs, dtype=np.float32).reshape(-1)[: self.obs_dim]
        self._dones[self._ptr] = bool(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """
        Sample a random mini-batch of transitions.

        Args:
            batch_size : Number of transitions to sample.

        Returns:
            Batch with uniform importance weights (all 1.0).

        Raises:
            ValueError: If fewer transitions are stored than batch_size.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer contains {self._size} transitions, "
                f"but batch_size={batch_size} was requested."
            )
        indices = self._rng.integers(0, self._size, size=batch_size)
        return self._gather(indices, weights=np.ones(batch_size, dtype=np.float32))

    def _gather(self, indices: np.ndarray, weights: np.ndarray) -> Batch:
        return Batch(
            obs=self._obs[indices].astype(np.float64),
            actions=self._actions[indices].astype(np.float64),
            rewards=self._rewards[indices].astype(np.float64),
            next_obs=self._next_obs[indices].astype(np.float64),
            dones=self._dones[indices],
            weights=weights.astype(np.float64),
            indices=indices.astype(np.int64),
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """True if the buffer has at least one full capacity worth of data (or 1000 samples)."""
        return self._size >= min(1000, self.capacity)

    def clear(self) -> None:
        """Reset the buffer to empty state."""
        self._ptr = 0
        self._size = 0

    def save(self, path: str) -> None:
        """Persist buffer contents to a .npz file."""
        np.savez_compressed(
            path,
            obs=self._obs[: self._size],
            actions=self._actions[: self._size],
            rewards=self._rewards[: self._size],
            next_obs=self._next_obs[: self._size],
            dones=self._dones[: self._size],
            size=np.array([self._size]),
            ptr=np.array([self._ptr]),
        )

    def load(self, path: str) -> None:
        """Load buffer contents from a .npz file."""
        data = np.load(path)
        size = int(data["size"][0])
        self._size = min(size, self.capacity)
        self._ptr = int(data["ptr"][0]) % self.capacity
        self._obs[: self._size] = data["obs"][: self._size]
        self._actions[: self._size] = data["actions"][: self._size]
        self._rewards[: self._size] = data["rewards"][: self._size]
        self._next_obs[: self._size] = data["next_obs"][: self._size]
        self._dones[: self._size] = data["dones"][: self._size]


# ---------------------------------------------------------------------------
# Sum-tree for PER
# ---------------------------------------------------------------------------


class _SumTree:
    """
    Binary sum-tree for O(log n) priority sampling and updates.

    Internally stores priorities in positions [capacity:2*capacity] (leaves)
    and aggregates sums up the tree. The total sum is at index 1.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._data_idx = 0  # next write position in the leaf layer

    def _propagate(self, leaf_idx: int, delta: float) -> None:
        parent = (leaf_idx + self.capacity) >> 1
        while parent >= 1:
            self._tree[parent] += delta
            parent >>= 1

    def update(self, data_idx: int, priority: float) -> None:
        """Set leaf at data_idx to priority, propagate delta to root."""
        leaf = data_idx + self.capacity
        delta = priority - self._tree[leaf]
        self._tree[leaf] = priority
        self._propagate(data_idx, delta)

    def total(self) -> float:
        return float(self._tree[1])

    def sample(self, value: float) -> Tuple[int, float]:
        """
        Find the leaf corresponding to cumulative priority `value`.

        Returns (data_idx, priority).
        """
        idx = 1
        while idx < self.capacity:
            left = idx << 1
            if self._tree[left] >= value:
                idx = left
            else:
                value -= self._tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return data_idx, float(self._tree[idx])

    def min_priority(self) -> float:
        """Minimum leaf priority (only among written leaves)."""
        return float(np.min(self._tree[self.capacity : self.capacity + self._data_idx + 1]))

    def max_priority(self) -> float:
        """Maximum leaf priority."""
        return float(np.max(self._tree[self.capacity : self.capacity + self._data_idx + 1]))


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.

    Priorities are set as |TD error|^alpha, and importance-sampling (IS)
    weights w_i = (1/N * 1/P(i))^beta are returned with each batch.

    The beta parameter should be annealed from beta_start to 1.0 over training.

    Args:
        capacity   : Maximum stored transitions.
        obs_dim    : Observation dimensionality.
        action_dim : Action dimensionality.
        alpha      : Priority exponent (0 = uniform, 1 = full priority).
        beta_start : IS weight exponent at start of training.
        beta_end   : IS weight exponent at end of training.
        eps        : Small constant added to all priorities to ensure non-zero sampling.
        seed       : Optional RNG seed.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        eps: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.eps = float(eps)
        self._rng = np.random.default_rng(seed)

        self._tree = _SumTree(self.capacity)

        self._obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.bool_)

        self._ptr: int = 0
        self._size: int = 0
        self._max_priority: float = 1.0
        self._step: int = 0  # training step counter for beta annealing

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray | float,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """
        Store a transition. New transitions receive max priority.

        Args:
            priority : Override priority (defaults to current max).
        """
        p = (priority + self.eps) ** self.alpha if priority is not None else self._max_priority
        self._tree.update(self._ptr, p)
        self._tree._data_idx = self._ptr

        self._obs[self._ptr] = np.asarray(obs, dtype=np.float32).reshape(-1)[: self.obs_dim]
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size == 1 and self.action_dim > 1:
            action_arr = np.full(self.action_dim, action_arr[0], dtype=np.float32)
        self._actions[self._ptr] = action_arr[: self.action_dim]
        self._rewards[self._ptr] = float(reward)
        self._next_obs[self._ptr] = np.asarray(next_obs, dtype=np.float32).reshape(-1)[: self.obs_dim]
        self._dones[self._ptr] = bool(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, beta: Optional[float] = None) -> Batch:
        """
        Sample a prioritised mini-batch.

        Args:
            batch_size : Number of transitions.
            beta       : IS weight exponent. If None, uses annealed value.

        Returns:
            Batch with IS importance weights and buffer indices.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has {self._size} transitions but batch_size={batch_size}."
            )

        if beta is None:
            beta = self._annealed_beta()

        total = self._tree.total()
        segment = total / batch_size
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = float(self._rng.uniform(low, high))
            data_idx, priority = self._tree.sample(min(value, total - 1e-10))
            data_idx = int(np.clip(data_idx, 0, self._size - 1))
            indices[i] = data_idx
            priorities[i] = priority + self.eps

        # Compute IS weights
        min_prob = priorities.min() / (total + 1e-12)
        max_weight = (1.0 / (self._size * min_prob + 1e-12)) ** beta
        weights = ((1.0 / (self._size * priorities / (total + 1e-12))) ** beta) / max_weight
        weights = np.clip(weights, 0.0, 1.0).astype(np.float32)

        return Batch(
            obs=self._obs[indices].astype(np.float64),
            actions=self._actions[indices].astype(np.float64),
            rewards=self._rewards[indices].astype(np.float64),
            next_obs=self._next_obs[indices].astype(np.float64),
            dones=self._dones[indices],
            weights=weights.astype(np.float64),
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities after a training step.

        Args:
            indices   : Buffer indices from the sampled Batch.
            td_errors : Absolute TD errors, shape (B,).
        """
        for idx, err in zip(indices, td_errors):
            p = (float(abs(err)) + self.eps) ** self.alpha
            self._tree.update(int(idx), p)
            self._max_priority = max(self._max_priority, p)
        self._step += 1

    def _annealed_beta(self) -> float:
        """Linearly anneal beta from beta_start to beta_end over training."""
        frac = min(1.0, self._step / max(1, 50_000))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size >= min(1000, self.capacity)


# ---------------------------------------------------------------------------
# EpisodeBuffer — on-policy rollout buffer for PPO / A2C
# ---------------------------------------------------------------------------


@dataclass
class _Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    log_prob: float
    value: float


class EpisodeBuffer:
    """
    On-policy rollout buffer for collecting full episodes (PPO, A2C).

    Stores transitions for a fixed number of steps or episodes, then
    computes GAE advantages and discounted returns before yielding batches.

    Args:
        obs_dim    : Observation dimensionality.
        action_dim : Action dimensionality.
        gamma      : Discount factor.
        gae_lambda : GAE lambda for advantage estimation.
        max_size   : Maximum stored transitions before flush.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_size: int = 4096,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_size = max_size

        self._transitions: list[_Transition] = []
        self._episode_ends: list[int] = []  # step indices where episodes end

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray | float,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Store a single on-policy transition."""
        self._transitions.append(
            _Transition(
                obs=np.asarray(obs, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32).reshape(self.action_dim),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                done=bool(done),
                log_prob=float(log_prob),
                value=float(value),
            )
        )
        if done:
            self._episode_ends.append(len(self._transitions) - 1)

    def compute_advantages(
        self, last_value: float = 0.0
    ) -> Batch:
        """
        Compute GAE advantages and discounted returns.

        Args:
            last_value : Bootstrap value for the last unfinished step.

        Returns:
            Batch with advantages and returns filled in.
        """
        n = len(self._transitions)
        if n == 0:
            raise ValueError("EpisodeBuffer is empty.")

        obs = np.stack([t.obs for t in self._transitions])
        actions = np.stack([t.action for t in self._transitions])
        rewards = np.array([t.reward for t in self._transitions], dtype=np.float64)
        next_obs = np.stack([t.next_obs for t in self._transitions])
        dones = np.array([t.done for t in self._transitions], dtype=np.bool_)
        log_probs = np.array([t.log_prob for t in self._transitions], dtype=np.float64)
        values = np.array([t.value for t in self._transitions], dtype=np.float64)

        advantages = np.zeros(n, dtype=np.float64)
        returns = np.zeros(n, dtype=np.float64)

        gae = 0.0
        next_val = last_value
        for t in reversed(range(n)):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_val = values[t]

        returns = advantages + values

        # Normalise advantages
        adv_mean = float(np.mean(advantages))
        adv_std = float(np.std(advantages)) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return Batch(
            obs=obs.astype(np.float64),
            actions=actions.astype(np.float64),
            rewards=rewards,
            next_obs=next_obs.astype(np.float64),
            dones=dones,
            weights=np.ones(n, dtype=np.float64),
            indices=np.arange(n, dtype=np.int64),
            log_probs=log_probs,
            advantages=advantages,
            returns=returns,
        )

    def sample_batches(
        self, batch_size: int, last_value: float = 0.0, seed: Optional[int] = None
    ) -> list[Batch]:
        """
        Compute advantages, then yield shuffled mini-batches.

        Args:
            batch_size : Size of each mini-batch.
            last_value : Bootstrap value for GAE.
            seed       : RNG seed for shuffling.

        Returns:
            List of Batch objects.
        """
        full_batch = self.compute_advantages(last_value)
        n = len(full_batch)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        batches = []
        for start in range(0, n, batch_size):
            idxs = perm[start : start + batch_size]
            if len(idxs) == 0:
                continue
            batches.append(
                Batch(
                    obs=full_batch.obs[idxs],
                    actions=full_batch.actions[idxs],
                    rewards=full_batch.rewards[idxs],
                    next_obs=full_batch.next_obs[idxs],
                    dones=full_batch.dones[idxs],
                    weights=full_batch.weights[idxs],
                    indices=full_batch.indices[idxs],
                    log_probs=full_batch.log_probs[idxs] if full_batch.log_probs is not None else None,
                    advantages=full_batch.advantages[idxs] if full_batch.advantages is not None else None,
                    returns=full_batch.returns[idxs] if full_batch.returns is not None else None,
                )
            )
        return batches

    def clear(self) -> None:
        self._transitions.clear()
        self._episode_ends.clear()

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def is_full(self) -> bool:
        return len(self._transitions) >= self.max_size


# ---------------------------------------------------------------------------
# HindsightReplayBuffer (HER)
# ---------------------------------------------------------------------------


@dataclass
class GoalTransition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    goal: np.ndarray
    achieved_goal: np.ndarray


class HindsightReplayBuffer:
    """
    Hindsight Experience Replay (HER) buffer for goal-conditioned learning.

    Stores goal-augmented transitions and retrospectively relabels goals
    using the 'future' strategy: for each real transition, k additional
    transitions are generated with future achieved-goals substituted as goals.

    In the trading context:
        goal          : target return or target equity level
        achieved_goal : actual return or equity achieved at step

    Args:
        capacity        : Max transitions (including hindsight).
        obs_dim         : Observation dimensionality.
        action_dim      : Action dimensionality.
        goal_dim        : Goal dimensionality.
        k_future        : Number of hindsight goals per real transition.
        reward_fn       : Function(achieved_goal, goal) -> float reward.
        seed            : Optional RNG seed.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        goal_dim: int,
        k_future: int = 4,
        reward_fn=None,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.k_future = int(k_future)
        self._rng = np.random.default_rng(seed)

        if reward_fn is None:
            # Default: reward 1.0 if achieved_goal >= goal, else -0.1
            self.reward_fn = lambda ag, g: 1.0 if float(np.mean(ag >= g)) > 0.5 else -0.1
        else:
            self.reward_fn = reward_fn

        # Store episodes as lists of GoalTransitions
        self._episodes: list[list[GoalTransition]] = []
        self._current_episode: list[GoalTransition] = []

        # Flat replay buffer for actual sampling
        _aug_obs_dim = obs_dim + goal_dim
        self._buffer = ReplayBuffer(capacity, _aug_obs_dim, action_dim, seed=seed)
        self._aug_obs_dim = _aug_obs_dim

    def push_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray | float,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        goal: np.ndarray,
        achieved_goal: np.ndarray,
    ) -> None:
        """Store a goal-augmented transition. Call push_episode_end() after each episode."""
        self._current_episode.append(
            GoalTransition(
                obs=np.asarray(obs, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32).reshape(self.action_dim),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                done=bool(done),
                goal=np.asarray(goal, dtype=np.float32),
                achieved_goal=np.asarray(achieved_goal, dtype=np.float32),
            )
        )
        # Store original transition with original goal
        aug_obs = np.concatenate([obs, goal]).astype(np.float32)
        aug_next = np.concatenate([next_obs, goal]).astype(np.float32)
        self._buffer.push(aug_obs, action, reward, aug_next, done)

    def push_episode_end(self) -> None:
        """
        Finalise the current episode and generate hindsight relabellings.
        """
        ep = self._current_episode
        n = len(ep)
        if n == 0:
            return

        # For each step, sample k future achieved goals as hindsight goals
        for t in range(n):
            future_indices = self._rng.integers(t, n, size=self.k_future)
            for fi in future_indices:
                hindsight_goal = ep[fi].achieved_goal
                r_hs = self.reward_fn(ep[t].achieved_goal, hindsight_goal)
                aug_obs = np.concatenate([ep[t].obs, hindsight_goal])
                aug_next = np.concatenate([ep[t].next_obs, hindsight_goal])
                self._buffer.push(aug_obs, ep[t].action, r_hs, aug_next, ep[t].done)

        self._episodes.append(ep)
        self._current_episode = []

        # Trim episodes list if memory is getting large
        if len(self._episodes) > 500:
            self._episodes = self._episodes[-500:]

    def sample(self, batch_size: int) -> Batch:
        """Sample a mini-batch from the augmented replay buffer."""
        return self._buffer.sample(batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        return self._buffer.is_ready
