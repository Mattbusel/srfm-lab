"""
research/agent_training/agents.py

RL agent implementations for training and evaluation.

All agents use the pure-numpy network layer from networks.py.
They are designed to interface directly with TradingEnvironment.

Agents:
    DQNAgent      — discrete DQN with epsilon-greedy exploration
    DDQNAgent     — Double DQN (extends DQNAgent)
    D3QNAgent     — Dueling + Double DQN + Prioritized Replay
    TD3Agent      — Twin Delayed DDPG for continuous actions
    PPOAgent      — Proximal Policy Optimisation
    EnsembleAgent — Combines D3QN + DDQN + TD3 with regime-adaptive weights
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from research.agent_training.networks import (
    Linear,
    ReLU,
    Tanh,
    Sequential,
    DuelingHead,
    ActorNetwork,
    CriticNetwork,
    mlp,
    clip_grad_norm,
    save_all_weights,
    load_all_weights,
)
from research.agent_training.replay_buffer import Batch, ReplayBuffer, PrioritizedReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy_params(src: Sequential | ActorNetwork | CriticNetwork | DuelingHead,
                 dst: Sequential | ActorNetwork | CriticNetwork | DuelingHead) -> None:
    """Hard copy parameters from src to dst."""
    src_p = src.parameters()
    dst_p = dst.parameters()
    for k in dst_p:
        if k in src_p:
            dst_p[k][:] = src_p[k]


def _soft_update_params(
    src: Any,
    dst: Any,
    tau: float,
) -> None:
    """Polyak soft update: dst = tau*src + (1-tau)*dst."""
    src_p = src.parameters()
    dst_p = dst.parameters()
    for k in dst_p:
        if k in src_p:
            dst_p[k][:] = tau * src_p[k] + (1.0 - tau) * dst_p[k]


def _mse_loss(pred: np.ndarray, target: np.ndarray, weights: Optional[np.ndarray] = None) -> tuple[float, np.ndarray]:
    """Compute weighted MSE loss and gradient w.r.t. pred."""
    diff = pred.reshape(-1) - target.reshape(-1)
    if weights is not None:
        w = weights.reshape(-1)
        loss = float(np.mean(w * diff ** 2))
        grad = 2.0 * w * diff / len(diff)
    else:
        loss = float(np.mean(diff ** 2))
        grad = 2.0 * diff / len(diff)
    return loss, grad.reshape(pred.shape)


# ---------------------------------------------------------------------------
# DQNAgent
# ---------------------------------------------------------------------------


@dataclass
class DQNConfig:
    obs_dim: int = 14
    n_actions: int = 21          # discretised [-1,1] into 21 bins
    hidden_dims: list = field(default_factory=lambda: [256, 256])
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    target_update_freq: int = 100
    grad_clip: float = 10.0
    l2_reg: float = 1e-4
    dropout: float = 0.0


class DQNAgent:
    """
    Vanilla Deep Q-Network agent with epsilon-greedy exploration.

    Discretises the continuous action space [-1, 1] into n_actions bins.
    Uses ReplayBuffer for experience replay.

    Args:
        obs_dim  : Observation dimensionality.
        n_actions: Number of discrete action bins.
        lr       : Adam learning rate.
        gamma    : Discount factor.
        epsilon  : Initial exploration rate.
    """

    _action_type: str = "discrete"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 21,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        hidden_dims: list[int] = (256, 256),
        grad_clip: float = 10.0,
        l2_reg: float = 1e-4,
        seed: Optional[int] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.grad_clip = grad_clip
        self._rng = np.random.default_rng(seed)
        self._train_steps = 0

        # Q-network and target network
        self.q_net = mlp(
            obs_dim, list(hidden_dims), n_actions,
            activation="relu", lr=lr, l2_reg=l2_reg
        )
        self.target_net = mlp(
            obs_dim, list(hidden_dims), n_actions,
            activation="relu", lr=lr, l2_reg=l2_reg
        )
        _copy_params(self.q_net, self.target_net)
        self.target_net.set_training(False)

        # Discrete action bins
        self._action_bins = np.linspace(-1.0, 1.0, n_actions)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray) -> float:
        """
        Epsilon-greedy action selection.

        Returns:
            Continuous action in [-1, 1].
        """
        if float(self._rng.random()) < self.epsilon:
            action_idx = int(self._rng.integers(0, self.n_actions))
        else:
            self.q_net.set_training(False)
            q_values = self.q_net.forward(obs)
            self.q_net.set_training(True)
            action_idx = int(np.argmax(q_values))
        return float(self._action_bins[action_idx])

    def act_greedy(self, obs: np.ndarray) -> float:
        """Pure greedy action (no exploration)."""
        self.q_net.set_training(False)
        q_values = self.q_net.forward(obs)
        self.q_net.set_training(True)
        return float(self._action_bins[int(np.argmax(q_values))])

    def _action_to_idx(self, action_value: float) -> int:
        """Convert continuous action to nearest bin index."""
        diffs = np.abs(self._action_bins - action_value)
        return int(np.argmin(diffs))

    def train_step(self, batch: Batch) -> float:
        """
        Perform one gradient update on the Q-network.

        Args:
            batch : Mini-batch from replay buffer.

        Returns:
            Scalar loss value.
        """
        obs = batch.obs                           # (B, obs_dim)
        rewards = batch.rewards                    # (B,)
        next_obs = batch.next_obs                  # (B, obs_dim)
        dones = batch.dones.astype(np.float64)     # (B,)
        weights = batch.weights                    # (B,)

        # Convert action values to indices
        action_indices = np.array(
            [self._action_to_idx(float(a)) for a in batch.actions.reshape(-1)],
            dtype=np.int64,
        )

        # Compute target Q-values
        self.target_net.set_training(False)
        q_next = self.target_net.forward(next_obs)   # (B, n_actions)
        q_next_max = q_next.max(axis=1)              # (B,)
        targets = rewards + self.gamma * q_next_max * (1.0 - dones)  # (B,)

        # Forward pass through Q-network
        self.q_net.set_training(True)
        q_pred_all = self.q_net.forward(obs)         # (B, n_actions)

        # Extract Q-values for taken actions
        B = len(obs)
        q_pred = q_pred_all[np.arange(B), action_indices]  # (B,)

        # MSE loss
        loss, grad_q = _mse_loss(q_pred, targets, weights)

        # Backprop: gradient only flows through chosen action's Q-value
        full_grad = np.zeros_like(q_pred_all)  # (B, n_actions)
        full_grad[np.arange(B), action_indices] = grad_q

        self.q_net.zero_grad()
        self.q_net.backward(full_grad)
        clip_grad_norm(self.q_net.layers, self.grad_clip)
        self.q_net.update(self.lr)

        self._train_steps += 1
        self._decay_epsilon()
        return loss

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def soft_update_target(self, tau: float = 0.005) -> None:
        """Polyak soft update of target network."""
        _soft_update_params(self.q_net, self.target_net, tau)

    def hard_update_target(self) -> None:
        """Hard copy Q-net to target."""
        _copy_params(self.q_net, self.target_net)

    def save(self, path: str) -> None:
        save_all_weights(self.q_net, path)

    def load(self, path: str) -> None:
        load_all_weights(self.q_net, path)
        _copy_params(self.q_net, self.target_net)


# ---------------------------------------------------------------------------
# DDQNAgent — Double DQN
# ---------------------------------------------------------------------------


class DDQNAgent(DQNAgent):
    """
    Double DQN: decouples action selection from action evaluation.

    Action selection uses the online network;
    action evaluation uses the target network.
    """

    _action_type: str = "discrete"

    def train_step(self, batch: Batch) -> float:
        obs = batch.obs
        rewards = batch.rewards
        next_obs = batch.next_obs
        dones = batch.dones.astype(np.float64)
        weights = batch.weights

        action_indices = np.array(
            [self._action_to_idx(float(a)) for a in batch.actions.reshape(-1)],
            dtype=np.int64,
        )

        # Double DQN: use online net to SELECT, target net to EVALUATE
        self.q_net.set_training(False)
        q_next_online = self.q_net.forward(next_obs)            # (B, n_actions)
        best_actions = q_next_online.argmax(axis=1)             # (B,)

        self.target_net.set_training(False)
        q_next_target = self.target_net.forward(next_obs)       # (B, n_actions)
        B = len(obs)
        q_next_best = q_next_target[np.arange(B), best_actions] # (B,)

        targets = rewards + self.gamma * q_next_best * (1.0 - dones)

        self.q_net.set_training(True)
        q_pred_all = self.q_net.forward(obs)
        q_pred = q_pred_all[np.arange(B), action_indices]

        loss, grad_q = _mse_loss(q_pred, targets, weights)

        full_grad = np.zeros_like(q_pred_all)
        full_grad[np.arange(B), action_indices] = grad_q

        self.q_net.zero_grad()
        self.q_net.backward(full_grad)
        clip_grad_norm(self.q_net.layers, self.grad_clip)
        self.q_net.update(self.lr)

        self._train_steps += 1
        self._decay_epsilon()
        return loss


# ---------------------------------------------------------------------------
# D3QNAgent — Dueling Double DQN with Prioritized Experience Replay
# ---------------------------------------------------------------------------


class D3QNAgent:
    """
    D3QN = Dueling architecture + Double Q-learning + Prioritized Replay.

    This is the primary discrete agent mirroring the live D3QN signal function.

    Args:
        obs_dim    : Observation dimensionality.
        n_actions  : Number of discrete action bins.
        hidden_dim : Hidden layer size for value/advantage streams.
        lr         : Adam learning rate.
        gamma      : Discount factor.
        epsilon    : Initial exploration rate.
    """

    _action_type: str = "discrete"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 21,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        grad_clip: float = 10.0,
        seed: Optional[int] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.grad_clip = grad_clip
        self._rng = np.random.default_rng(seed)
        self._train_steps = 0

        # Shared feature extractor
        self._feat_net = mlp(obs_dim, [hidden_dim], hidden_dim, activation="relu", lr=lr)
        self._feat_target = mlp(obs_dim, [hidden_dim], hidden_dim, activation="relu", lr=lr)

        # Dueling head on top
        self.dueling = DuelingHead(hidden_dim, n_actions, hidden_dim=hidden_dim // 2, lr=lr)
        self.dueling_target = DuelingHead(hidden_dim, n_actions, hidden_dim=hidden_dim // 2, lr=lr)

        _copy_params(self._feat_net, self._feat_target)
        _copy_params(self.dueling, self.dueling_target)

        self._action_bins = np.linspace(-1.0, 1.0, n_actions)

    def _q_values(self, obs: np.ndarray, use_target: bool = False) -> np.ndarray:
        feat_net = self._feat_target if use_target else self._feat_net
        dueling = self.dueling_target if use_target else self.dueling
        features = feat_net.forward(obs)
        return dueling.forward(features)

    def act(self, obs: np.ndarray) -> float:
        if float(self._rng.random()) < self.epsilon:
            idx = int(self._rng.integers(0, self.n_actions))
        else:
            q = self._q_values(obs, use_target=False)
            idx = int(np.argmax(q))
        return float(self._action_bins[idx])

    def act_greedy(self, obs: np.ndarray) -> float:
        q = self._q_values(obs, use_target=False)
        return float(self._action_bins[int(np.argmax(q))])

    def _action_to_idx(self, v: float) -> int:
        return int(np.argmin(np.abs(self._action_bins - v)))

    def train_step(self, batch: Batch) -> float:
        """
        D3QN training step with PER importance weights.

        Returns:
            loss (float) and updates batch.indices priorities externally
            (caller should update PER with |td_errors|).
        """
        obs = batch.obs
        rewards = batch.rewards
        next_obs = batch.next_obs
        dones = batch.dones.astype(np.float64)
        weights = batch.weights
        B = len(obs)

        action_indices = np.array(
            [self._action_to_idx(float(a)) for a in batch.actions.reshape(-1)]
        )

        # Double DQN target
        q_next_online = self._q_values(next_obs, use_target=False)
        best_next_actions = q_next_online.argmax(axis=1)
        q_next_target = self._q_values(next_obs, use_target=True)
        q_next_best = q_next_target[np.arange(B), best_next_actions]
        targets = rewards + self.gamma * q_next_best * (1.0 - dones)

        # Online forward
        feats = self._feat_net.forward(obs)
        q_all = self.dueling.forward(feats)
        q_pred = q_all[np.arange(B), action_indices]

        td_errors = q_pred - targets
        loss = float(np.mean(weights * td_errors ** 2))

        full_grad = np.zeros_like(q_all)
        full_grad[np.arange(B), action_indices] = 2.0 * weights * td_errors / B

        self.dueling.zero_grad()
        self._feat_net.zero_grad()

        feat_grad = self.dueling.backward(full_grad)
        self._feat_net.backward(feat_grad)

        clip_grad_norm(self._feat_net.layers, self.grad_clip)
        self._feat_net.update(self.lr)
        self.dueling.update(self.lr)

        self._train_steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Expose TD errors for PER update
        self._last_td_errors = np.abs(td_errors)
        return loss

    def soft_update_target(self, tau: float = 0.005) -> None:
        _soft_update_params(self._feat_net, self._feat_target, tau)
        _soft_update_params(self.dueling, self.dueling_target, tau)

    def hard_update_target(self) -> None:
        _copy_params(self._feat_net, self._feat_target)
        _copy_params(self.dueling, self.dueling_target)

    def save(self, path: str) -> None:
        params = {}
        for k, v in self._feat_net.parameters().items():
            params[f"feat_{k}"] = v
        for k, v in self.dueling.parameters().items():
            params[f"duel_{k}"] = v
        np.savez_compressed(path, **params)

    def load(self, path: str) -> None:
        data = np.load(path)
        for k, v in self._feat_net.parameters().items():
            if f"feat_{k}" in data:
                v[:] = data[f"feat_{k}"]
        for k, v in self.dueling.parameters().items():
            if f"duel_{k}" in data:
                v[:] = data[f"duel_{k}"]
        self.hard_update_target()


# ---------------------------------------------------------------------------
# TD3Agent — Twin Delayed DDPG
# ---------------------------------------------------------------------------


class TD3Agent:
    """
    Twin Delayed DDPG (TD3) for continuous action spaces.

    Key features:
    - Two Q-critics (Q1, Q2) — take min to reduce overestimation
    - Delayed actor updates (every policy_delay critic updates)
    - Target policy smoothing: add clipped noise to target action

    Args:
        obs_dim      : Observation dimensionality.
        action_dim   : Action dimensionality (1 for single instrument).
        hidden_dims  : Hidden layer sizes.
        lr_actor     : Actor Adam learning rate.
        lr_critic    : Critic Adam learning rate.
        gamma        : Discount factor.
        tau          : Polyak averaging coefficient.
        policy_delay : Number of critic updates per actor update.
        noise_std    : Standard deviation of target policy smoothing noise.
        noise_clip   : Clip range for smoothing noise.
    """

    _action_type: str = "continuous"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 1,
        hidden_dims: list[int] = (256, 256),
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        noise_std: float = 0.2,
        noise_clip: float = 0.5,
        expl_noise: float = 0.1,
        grad_clip: float = 10.0,
        seed: Optional[int] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise
        self.grad_clip = grad_clip
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self._rng = np.random.default_rng(seed)
        self._train_steps = 0

        # Actor and target actor
        self.actor = ActorNetwork(obs_dim, action_dim, list(hidden_dims), lr=lr_actor)
        self.actor_target = ActorNetwork(obs_dim, action_dim, list(hidden_dims), lr=lr_actor)
        _copy_params(self.actor, self.actor_target)

        # Two critics and their targets
        self.critic1 = CriticNetwork(obs_dim, action_dim, list(hidden_dims), lr=lr_critic)
        self.critic2 = CriticNetwork(obs_dim, action_dim, list(hidden_dims), lr=lr_critic)
        self.critic1_target = CriticNetwork(obs_dim, action_dim, list(hidden_dims), lr=lr_critic)
        self.critic2_target = CriticNetwork(obs_dim, action_dim, list(hidden_dims), lr=lr_critic)
        _copy_params(self.critic1, self.critic1_target)
        _copy_params(self.critic2, self.critic2_target)

        self.actor_target.set_training(False)
        self.critic1_target.set_training(False)
        self.critic2_target.set_training(False)

    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray) -> float:
        """Action with Gaussian exploration noise."""
        self.actor.set_training(False)
        action = float(self.actor.forward(obs).reshape(-1)[0])
        self.actor.set_training(True)
        noise = float(self._rng.normal(0, self.expl_noise))
        return float(np.clip(action + noise, -1.0, 1.0))

    def act_greedy(self, obs: np.ndarray) -> float:
        """Deterministic action, no noise."""
        self.actor.set_training(False)
        action = float(self.actor.forward(obs).reshape(-1)[0])
        self.actor.set_training(True)
        return float(np.clip(action, -1.0, 1.0))

    def train_step(self, batch: Batch) -> float:
        """
        One TD3 training step.

        Updates both critics every step, and updates actor + target networks
        only every `policy_delay` critic steps.

        Returns:
            Total critic loss.
        """
        obs = batch.obs
        actions = batch.actions
        rewards = batch.rewards
        next_obs = batch.next_obs
        dones = batch.dones.astype(np.float64)
        weights = batch.weights
        B = len(obs)

        # Target action with smoothing noise
        self.actor_target.set_training(False)
        target_actions_raw = self.actor_target.forward(next_obs)  # (B, action_dim)
        noise = np.clip(
            self._rng.normal(0, self.noise_std, target_actions_raw.shape),
            -self.noise_clip, self.noise_clip,
        )
        target_actions = np.clip(target_actions_raw + noise, -1.0, 1.0)

        # Target Q-values: min of Q1, Q2
        q1_next = self.critic1_target.forward(next_obs, target_actions).reshape(-1)
        q2_next = self.critic2_target.forward(next_obs, target_actions).reshape(-1)
        q_next = np.minimum(q1_next, q2_next)
        q_target = rewards + self.gamma * q_next * (1.0 - dones)

        # Concatenated input for critics
        sa = np.concatenate([obs, actions.reshape(B, -1)], axis=1)

        # Critic 1 update
        self.critic1.set_training(True)
        q1_pred = self.critic1.forward_concat(sa).reshape(-1)
        loss1, grad1 = _mse_loss(q1_pred, q_target, weights)
        self.critic1.zero_grad()
        self.critic1.backward(grad1.reshape(-1, 1))
        clip_grad_norm(self.critic1._net.layers, self.grad_clip)
        self.critic1.update(self.lr_critic)

        # Critic 2 update
        self.critic2.set_training(True)
        q2_pred = self.critic2.forward_concat(sa).reshape(-1)
        loss2, grad2 = _mse_loss(q2_pred, q_target, weights)
        self.critic2.zero_grad()
        self.critic2.backward(grad2.reshape(-1, 1))
        clip_grad_norm(self.critic2._net.layers, self.grad_clip)
        self.critic2.update(self.lr_critic)

        total_loss = float(loss1 + loss2)

        # Delayed actor update
        if self._train_steps % self.policy_delay == 0:
            self.actor.set_training(True)
            actor_actions = self.actor.forward(obs)              # (B, action_dim)
            sa_actor = np.concatenate([obs, actor_actions.reshape(B, -1)], axis=1)
            q1_actor = self.critic1.forward_concat(sa_actor).reshape(-1)

            # Policy gradient: maximise Q1 -> minimise -Q1
            actor_loss = -float(np.mean(q1_actor))
            grad_q1 = -np.ones(B) / B                           # (B,)

            # Backprop through critic1 to get gradient w.r.t. actions
            grad_sa = self.critic1.backward(grad_q1.reshape(-1, 1))
            grad_actor_actions = grad_sa[:, self.obs_dim:]      # (B, action_dim)

            self.actor.zero_grad()
            self.actor.backward(grad_actor_actions)
            clip_grad_norm(self.actor._net.layers, self.grad_clip)
            self.actor.update(self.lr_actor)

            # Soft update targets
            _soft_update_params(self.actor, self.actor_target, self.tau)
            _soft_update_params(self.critic1, self.critic1_target, self.tau)
            _soft_update_params(self.critic2, self.critic2_target, self.tau)

        self._train_steps += 1
        return total_loss

    def save(self, path: str) -> None:
        params = {}
        for prefix, model in [
            ("actor", self.actor),
            ("c1", self.critic1),
            ("c2", self.critic2),
        ]:
            for k, v in model.parameters().items():
                params[f"{prefix}_{k}"] = v
        np.savez_compressed(path, **params)

    def load(self, path: str) -> None:
        data = np.load(path)
        for prefix, model in [("actor", self.actor), ("c1", self.critic1), ("c2", self.critic2)]:
            for k, v in model.parameters().items():
                key = f"{prefix}_{k}"
                if key in data:
                    v[:] = data[key]
        _copy_params(self.actor, self.actor_target)
        _copy_params(self.critic1, self.critic1_target)
        _copy_params(self.critic2, self.critic2_target)


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------


class PPOAgent:
    """
    Proximal Policy Optimisation (PPO-Clip) for continuous actions.

    Architecture:
        Actor:  obs -> Gaussian policy (mean, log_std)
        Critic: obs -> V(s)

    Args:
        obs_dim        : Observation dimensionality.
        action_dim     : Action dimensionality.
        lr_actor       : Actor Adam learning rate.
        lr_critic      : Critic Adam learning rate.
        clip_epsilon   : PPO clipping epsilon.
        gamma          : Discount factor.
        gae_lambda     : GAE lambda.
        entropy_coef   : Entropy bonus coefficient.
        value_coef     : Value loss coefficient.
        grad_clip      : Gradient clipping norm.
        hidden_dims    : Hidden layer sizes.
    """

    _action_type: str = "continuous"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 1,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        grad_clip: float = 0.5,
        hidden_dims: list[int] = (256, 256),
        seed: Optional[int] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.grad_clip = grad_clip
        self._rng = np.random.default_rng(seed)
        self._train_steps = 0

        # Actor: outputs mean of Gaussian
        self.actor = ActorNetwork(obs_dim, action_dim, list(hidden_dims), lr=lr_actor)
        # Log std as separate parameter (not input-dependent for simplicity)
        self.log_std = np.zeros(action_dim, dtype=np.float64) - 0.5
        self._log_std_adam = _AdamState((action_dim,), lr=lr_actor)

        # Critic: outputs V(s)
        self.critic = mlp(obs_dim, list(hidden_dims), 1, activation="relu", lr=lr_critic)

    def _gaussian_log_prob(self, mean: np.ndarray, log_std: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Compute Gaussian log probability: log N(action | mean, std)."""
        std = np.exp(log_std)
        log_prob = -0.5 * ((action - mean) / (std + 1e-8)) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        return log_prob.sum(axis=-1)

    def act(self, obs: np.ndarray) -> tuple[float, float]:
        """
        Sample action from current policy.

        Returns:
            (action, log_prob)
        """
        self.actor.set_training(False)
        mean = self.actor.forward(obs).reshape(-1)  # (action_dim,)
        std = np.exp(self.log_std)
        noise = self._rng.normal(0, 1, size=self.action_dim)
        action = np.clip(mean + std * noise, -1.0, 1.0)
        log_prob = float(self._gaussian_log_prob(mean, self.log_std, action))
        self.actor.set_training(True)
        return float(action[0]) if self.action_dim == 1 else action, log_prob

    def act_greedy(self, obs: np.ndarray) -> float:
        """Return mean (deterministic) action."""
        self.actor.set_training(False)
        mean = self.actor.forward(obs).reshape(-1)
        self.actor.set_training(True)
        return float(np.clip(mean[0], -1.0, 1.0))

    def get_value(self, obs: np.ndarray) -> float:
        """Return V(s) estimate."""
        self.critic.set_training(False)
        v = float(self.critic.forward(obs).reshape(-1)[0])
        self.critic.set_training(True)
        return v

    def collect_episode(self, env) -> list[dict]:
        """
        Collect one episode of experience using the current policy.

        Args:
            env : TradingEnvironment (or compatible).

        Returns:
            List of transition dicts with keys: obs, action, reward,
            next_obs, done, log_prob, value.
        """
        obs = env.reset()
        transitions = []
        done = False
        while not done:
            action, log_prob = self.act(obs)
            value = self.get_value(obs)
            next_obs, reward, done, info = env.step(action)
            transitions.append({
                "obs": obs.copy(),
                "action": np.array([action], dtype=np.float64),
                "reward": float(reward),
                "next_obs": next_obs.copy(),
                "done": bool(done),
                "log_prob": float(log_prob),
                "value": float(value),
                "equity": info.get("equity", 0.0),
                "info": info,
            })
            obs = next_obs
        return transitions

    def train_epoch(
        self,
        batch: Batch,
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """
        PPO training: multiple epochs over the collected rollout.

        Args:
            batch     : Batch with advantages and returns (from EpisodeBuffer).
            n_epochs  : Number of epochs over the rollout.
            batch_size: Mini-batch size within each epoch.

        Returns:
            Dict of mean losses: {'policy_loss', 'value_loss', 'entropy', 'total'}.
        """
        n = len(batch)
        all_policy_loss = []
        all_value_loss = []
        all_entropy = []

        for _ in range(n_epochs):
            perm = self._rng.permutation(n)
            for start in range(0, n, batch_size):
                idxs = perm[start : start + batch_size]
                if len(idxs) == 0:
                    continue

                obs_b = batch.obs[idxs]
                act_b = batch.actions[idxs]
                old_log_probs_b = batch.log_probs[idxs] if batch.log_probs is not None else np.zeros(len(idxs))
                advantages_b = batch.advantages[idxs] if batch.advantages is not None else np.zeros(len(idxs))
                returns_b = batch.returns[idxs] if batch.returns is not None else np.zeros(len(idxs))

                # New log probs
                self.actor.set_training(True)
                means = self.actor.forward(obs_b)                # (B, action_dim)
                new_log_probs = self._gaussian_log_prob(means, self.log_std, act_b)  # (B,)

                # PPO ratio
                log_ratio = new_log_probs - old_log_probs_b
                ratio = np.exp(np.clip(log_ratio, -10.0, 10.0))  # (B,)

                # Clipped policy loss
                surr1 = ratio * advantages_b
                surr2 = np.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_b
                policy_loss = float(-np.mean(np.minimum(surr1, surr2)))

                # Value loss
                self.critic.set_training(True)
                values_pred = self.critic.forward(obs_b).reshape(-1)
                value_loss, grad_vl = _mse_loss(values_pred, returns_b)

                # Entropy bonus (Gaussian)
                std = np.exp(self.log_std)
                entropy = float(np.sum(self.log_std + 0.5 * math.log(2 * math.pi * math.e)))

                # Backprop value loss
                self.critic.zero_grad()
                self.critic.backward(grad_vl.reshape(-1, 1))
                clip_grad_norm(self.critic.layers, self.grad_clip)
                self.critic.update(self.lr_critic)

                # Backprop policy loss
                B_b = len(idxs)
                # Gradient of clipped surrogate w.r.t. ratio
                clip_mask = (ratio > 1.0 + self.clip_epsilon) | (ratio < 1.0 - self.clip_epsilon)
                grad_ratio = np.where(clip_mask, 0.0, -advantages_b / B_b)
                # grad log_prob = grad_ratio * ratio (product rule)
                grad_log_prob = grad_ratio * ratio  # (B,)

                # Gradient w.r.t. means: d(log_prob)/d(mean) = (a-mu)/std^2
                std_b = np.exp(self.log_std)
                grad_mean = grad_log_prob[:, np.newaxis] * (act_b - means) / (std_b ** 2 + 1e-8)

                # Entropy gradient on log_std: d entropy / d log_std = 1 (per dim)
                grad_log_std_entropy = -self.entropy_coef * np.ones_like(self.log_std)

                self.actor.zero_grad()
                self.actor.backward(grad_mean)
                clip_grad_norm(self.actor._net.layers, self.grad_clip)
                self.actor.update(self.lr_actor)

                # Update log_std
                from research.agent_training.networks import _AdamState as _AS
                delta_ls = self._log_std_adam.step(grad_log_std_entropy)
                self.log_std -= delta_ls

                all_policy_loss.append(policy_loss)
                all_value_loss.append(value_loss)
                all_entropy.append(entropy)

        self._train_steps += n_epochs
        return {
            "policy_loss": float(np.mean(all_policy_loss)),
            "value_loss": float(np.mean(all_value_loss)),
            "entropy": float(np.mean(all_entropy)),
            "total": float(np.mean(all_policy_loss) + self.value_coef * np.mean(all_value_loss)),
        }

    def save(self, path: str) -> None:
        params = {}
        for k, v in self.actor.parameters().items():
            params[f"actor_{k}"] = v
        for k, v in self.critic.parameters().items():
            params[f"critic_{k}"] = v
        params["log_std"] = self.log_std
        np.savez_compressed(path, **params)

    def load(self, path: str) -> None:
        data = np.load(path)
        for k, v in self.actor.parameters().items():
            if f"actor_{k}" in data:
                v[:] = data[f"actor_{k}"]
        for k, v in self.critic.parameters().items():
            if f"critic_{k}" in data:
                v[:] = data[f"critic_{k}"]
        if "log_std" in data:
            self.log_std[:] = data["log_std"]


# ---------------------------------------------------------------------------
# Private Adam import for PPOAgent (already defined in networks but needed here)
# ---------------------------------------------------------------------------

from research.agent_training.networks import _AdamState  # noqa: E402


# ---------------------------------------------------------------------------
# EnsembleAgent
# ---------------------------------------------------------------------------


REGIME_WEIGHTS_ENSEMBLE: dict[str, np.ndarray] = {
    "BULL":            np.array([0.50, 0.30, 0.20]),   # D3QN, DDQN, TD3
    "BEAR":            np.array([0.40, 0.35, 0.25]),
    "SIDEWAYS":        np.array([0.20, 0.25, 0.55]),
    "HIGH_VOLATILITY": np.array([0.25, 0.25, 0.50]),
    "UNKNOWN":         np.array([0.33, 0.33, 0.34]),
}


class EnsembleAgent:
    """
    Ensemble of D3QN + DDQN + TD3 agents with regime-adaptive weighting.

    Mirrors the live trader's ensemble logic but operates on trained agents.

    The act() method selects actions from all three sub-agents, then computes
    a weighted sum using regime-specific weights (same scheme as agents.py
    REGIME_WEIGHTS in the live system).

    Args:
        obs_dim    : Observation dimensionality.
        n_actions  : Discrete action bins for D3QN / DDQN.
        hidden_dim : Hidden layer size.
        lr         : Learning rate for all sub-agents.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 21,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.d3qn = D3QNAgent(obs_dim, n_actions, hidden_dim, lr=lr, seed=seed)
        self.ddqn = DDQNAgent(obs_dim, n_actions, lr=lr, hidden_dims=(hidden_dim, hidden_dim), seed=seed)
        self.td3 = TD3Agent(obs_dim, action_dim=1, hidden_dims=[hidden_dim, hidden_dim], lr_actor=lr / 3, seed=seed)

        self._regime: str = "UNKNOWN"

    def set_regime(self, regime: str) -> None:
        """Set the current market regime for weight selection."""
        self._regime = regime.upper()

    def act(self, obs: np.ndarray) -> float:
        """
        Weighted ensemble action.

        Returns:
            Continuous action in [-1, 1].
        """
        a1 = self.d3qn.act_greedy(obs)
        a2 = self.ddqn.act_greedy(obs)
        a3 = self.td3.act_greedy(obs)

        regime = self._regime if self._regime in REGIME_WEIGHTS_ENSEMBLE else "UNKNOWN"
        w = REGIME_WEIGHTS_ENSEMBLE[regime]
        action = float(w[0] * a1 + w[1] * a2 + w[2] * a3)
        return float(np.clip(action, -1.0, 1.0))

    def act_explore(self, obs: np.ndarray) -> float:
        """Ensemble action with individual agent exploration."""
        a1 = self.d3qn.act(obs)
        a2 = self.ddqn.act(obs)
        a3 = self.td3.act(obs)
        regime = self._regime if self._regime in REGIME_WEIGHTS_ENSEMBLE else "UNKNOWN"
        w = REGIME_WEIGHTS_ENSEMBLE[regime]
        return float(np.clip(w[0] * a1 + w[1] * a2 + w[2] * a3, -1.0, 1.0))

    def train_all(self, batch: Batch) -> dict[str, float]:
        """
        Train all sub-agents on the same batch.

        Returns:
            Dict mapping agent name to loss.
        """
        loss_d3qn = self.d3qn.train_step(batch)
        loss_ddqn = self.ddqn.train_step(batch)
        loss_td3 = self.td3.train_step(batch)
        return {
            "d3qn": loss_d3qn,
            "ddqn": loss_ddqn,
            "td3": loss_td3,
        }

    def soft_update_targets(self, tau: float = 0.005) -> None:
        self.d3qn.soft_update_target(tau)
        self.ddqn.soft_update_target(tau)
        # TD3 updates inside train_step

    def save(self, base_path: str) -> None:
        self.d3qn.save(f"{base_path}_d3qn.npz")
        self.ddqn.save(f"{base_path}_ddqn.npz")
        self.td3.save(f"{base_path}_td3.npz")

    def load(self, base_path: str) -> None:
        self.d3qn.load(f"{base_path}_d3qn.npz")
        self.ddqn.load(f"{base_path}_ddqn.npz")
        self.td3.load(f"{base_path}_td3.npz")
