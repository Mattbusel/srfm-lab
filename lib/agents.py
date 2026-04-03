"""
agents.py — Reinforcement learning agent ensemble for SRFM strategies.

Contains D3QN, DDQN, and TD3QN agents.  Each agent is self-contained so it
can be unit-tested independently of LEAN.  The ensemble combines their signals
via a configurable voting / weighting scheme.

Note: LEAN QCAlgorithm imports are intentionally avoided here so this module
is importable outside of LEAN (e.g., in notebooks or unit tests).
"""

from __future__ import annotations
import math
import random
from collections import deque
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer (shared by all agents)
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 2000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Base Agent
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    """
    Lightweight tabular / linear approximation agent that works without
    PyTorch or TensorFlow (LEAN's Docker image may not have them).

    State: tuple of discretised feature values.
    Actions: 0 = flat, 1 = long, 2 = short.
    """

    NUM_ACTIONS = 3

    def __init__(
        self,
        state_bins: int = 5,
        learning_rate: float = 0.01,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 2000,
        batch_size: int = 32,
    ):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.state_bins = state_bins

        # Q-table: dict mapping state → [Q(s,0), Q(s,1), Q(s,2)]
        self._q: dict = {}
        self.replay = ReplayBuffer(buffer_capacity)
        self.steps = 0

    # ------------------------------------------------------------------
    def _q_values(self, state) -> List[float]:
        key = tuple(state)
        if key not in self._q:
            self._q[key] = [0.0] * self.NUM_ACTIONS
        return self._q[key]

    def act(self, state, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.NUM_ACTIONS)
        q = self._q_values(state)
        return int(q.index(max(q)))

    def remember(self, state, action: int, reward: float, next_state, done: bool):
        self.replay.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        batch = self.replay.sample(self.batch_size)
        for state, action, reward, next_state, done in batch:
            q = self._q_values(state)
            q_next = self._q_values(next_state)
            target = reward if done else reward + self.gamma * max(q_next)
            q[action] += self.lr * (target - q[action])
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1

    def q_signal(self, state) -> float:
        """Return a [-1, +1] signal: argmax_q mapped to {short, flat, long}."""
        action = self.act(state, deterministic=True)
        return {0: 0.0, 1: 1.0, 2: -1.0}[action]


# ─────────────────────────────────────────────────────────────────────────────
# Specific Agent Variants
# ─────────────────────────────────────────────────────────────────────────────

class D3QNAgent(BaseAgent):
    """
    Dueling Double DQN (tabular approximation).

    Dueling: Q(s,a) = V(s) + A(s,a) − mean(A)
    Here approximated by maintaining separate value and advantage tables.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._v: dict = {}   # State-value table
        self._a: dict = {}   # Advantage table

    def _q_values(self, state) -> List[float]:
        key = tuple(state)
        if key not in self._v:
            self._v[key] = 0.0
            self._a[key] = [0.0] * self.NUM_ACTIONS
        v = self._v[key]
        a = self._a[key]
        mean_a = sum(a) / len(a)
        return [v + ai - mean_a for ai in a]

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        batch = self.replay.sample(self.batch_size)
        for state, action, reward, next_state, done in batch:
            key = tuple(state)
            next_key = tuple(next_state)
            if key not in self._v:
                self._v[key] = 0.0
                self._a[key] = [0.0] * self.NUM_ACTIONS
            if next_key not in self._v:
                self._v[next_key] = 0.0
                self._a[next_key] = [0.0] * self.NUM_ACTIONS

            q_next = max(self._q_values(next_state))
            td_target = reward if done else reward + self.gamma * q_next
            td_error = td_target - self._q_values(state)[action]

            self._v[key] += self.lr * td_error
            self._a[key][action] += self.lr * td_error

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1


class DDQNAgent(BaseAgent):
    """
    Double DQN — uses online network to select action, target network to evaluate.
    Approximated here with two Q-tables and periodic target sync.
    """

    def __init__(self, target_update_freq: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.target_update_freq = target_update_freq
        self._q_target: dict = {}

    def _q_target_values(self, state) -> List[float]:
        key = tuple(state)
        if key not in self._q_target:
            self._q_target[key] = [0.0] * self.NUM_ACTIONS
        return self._q_target[key]

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        batch = self.replay.sample(self.batch_size)
        for state, action, reward, next_state, done in batch:
            # Online selects action; target evaluates
            online_next = self._q_values(next_state)
            best_next_action = online_next.index(max(online_next))
            q_target_next = self._q_target_values(next_state)[best_next_action]

            target = reward if done else reward + self.gamma * q_target_next
            q = self._q_values(state)
            q[action] += self.lr * (target - q[action])

        # Periodically sync online → target
        if self.steps % self.target_update_freq == 0:
            import copy
            self._q_target = copy.deepcopy(self._q)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1


class TD3QNAgent(BaseAgent):
    """
    Triple-network DQN: maintains three Q-tables and uses the minimum to
    reduce overestimation (analogous to TD3's twin critics).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._q2: dict = {}
        self._q3: dict = {}

    def _all_q_values(self, state) -> Tuple[List[float], List[float], List[float]]:
        key = tuple(state)
        if key not in self._q:
            self._q[key] = [0.0] * self.NUM_ACTIONS
        if key not in self._q2:
            self._q2[key] = [0.0] * self.NUM_ACTIONS
        if key not in self._q3:
            self._q3[key] = [0.0] * self.NUM_ACTIONS
        return self._q[key], self._q2[key], self._q3[key]

    def _q_values(self, state) -> List[float]:
        q1, q2, q3 = self._all_q_values(state)
        return [min(a, b, c) for a, b, c in zip(q1, q2, q3)]

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        batch = self.replay.sample(self.batch_size)
        for state, action, reward, next_state, done in batch:
            q_next = max(self._q_values(next_state))
            target = reward if done else reward + self.gamma * q_next
            for q_table in self._all_q_values(state):
                q_table[action] += self.lr * (target - q_table[action])

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────

class AgentEnsemble:
    """
    Combines D3QN, DDQN, and TD3QN signals via weighted voting.

    Signal: +1 (long), 0 (flat), -1 (short).
    Consensus threshold: majority of weighted votes must agree.
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        consensus_threshold: float = 0.6,
        **agent_kwargs,
    ):
        self.agents = [
            D3QNAgent(**agent_kwargs),
            DDQNAgent(**agent_kwargs),
            TD3QNAgent(**agent_kwargs),
        ]
        self.weights = weights or [1.0, 1.0, 1.0]
        self.consensus_threshold = consensus_threshold

    def act(self, state, deterministic: bool = False) -> int:
        """Return the consensus action (0/1/2)."""
        votes = [0.0, 0.0, 0.0]
        total_weight = sum(self.weights)
        for agent, w in zip(self.agents, self.weights):
            a = agent.act(state, deterministic)
            votes[a] += w / total_weight
        best = votes.index(max(votes))
        if votes[best] >= self.consensus_threshold:
            return best
        return 0  # No consensus → flat

    def signal(self, state) -> float:
        """[-1, 0, +1] signal from consensus action."""
        return {0: 0.0, 1: 1.0, 2: -1.0}[self.act(state, deterministic=True)]

    def remember(self, state, action: int, reward: float, next_state, done: bool):
        for agent in self.agents:
            agent.remember(state, action, reward, next_state, done)

    def learn(self):
        for agent in self.agents:
            agent.learn()

    @property
    def epsilon(self) -> float:
        return self.agents[0].epsilon
