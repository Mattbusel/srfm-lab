"""
position_sizer.py — Reinforcement Learning position sizer for idea-engine.

Implements:
  - State space: regime, z-score, volatility, current_position, unrealized_pnl, time_in_trade
  - Action space: continuous [-1.0, 1.0] position target
  - Q-learning with linear + RBF function approximation
  - Actor-Critic agent (policy gradient with baseline)
  - Reward: risk-adjusted PnL, drawdown penalty, turnover penalty
  - Experience replay buffer
  - Episode-based training on historical data
  - Kelly criterion integration for position sizing
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# State and Action definitions
# ---------------------------------------------------------------------------

@dataclass
class State:
    """
    Market / portfolio state at a single timestep.
    All fields should be normalised before feeding into the agent.
    """
    regime: float           # -1=bear, 0=neutral, 1=bull (or fractional)
    z_score: float          # signal z-score
    volatility: float       # annualised vol, normalised (e.g. /0.40)
    current_position: float # current position [-1, 1]
    unrealized_pnl: float   # PnL in units of daily vol (normalised)
    time_in_trade: float    # fraction of max holding period [0, 1]

    def to_array(self) -> np.ndarray:
        return np.array([
            self.regime,
            self.z_score,
            self.volatility,
            self.current_position,
            self.unrealized_pnl,
            self.time_in_trade,
        ], dtype=np.float32)

    @staticmethod
    def dim() -> int:
        return 6


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size circular replay buffer for experience replay."""

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# RBF Feature Map for function approximation
# ---------------------------------------------------------------------------

class RBFFeatureMap:
    """
    Random Fourier Feature (RBF kernel approximation) feature map.
    Maps state vectors to higher-dimensional feature space for linear Q approx.
    """

    def __init__(self, input_dim: int, n_features: int = 128, gamma: float = 1.0, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W = rng.normal(0, np.sqrt(2 * gamma), (n_features, input_dim)).astype(np.float32)
        self.b = rng.uniform(0, 2 * np.pi, n_features).astype(np.float32)
        self.scale = np.sqrt(2.0 / n_features)
        self.n_features = n_features

    def transform(self, x: np.ndarray) -> np.ndarray:
        """x: (d,) or (N, d) -> (n_features,) or (N, n_features)"""
        z = x @ self.W.T + self.b
        return self.scale * np.cos(z)


# ---------------------------------------------------------------------------
# Q-Learning with Function Approximation
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Continuous-action Q-learning using linear function approximation over RBF features.
    We discretise actions into a fine grid for the Q-function lookup.
    """

    N_ACTIONS = 21  # [-1.0, -0.9, ..., 0.9, 1.0]

    def __init__(
        self,
        state_dim: int = State.dim(),
        n_rbf_features: int = 256,
        gamma: float = 0.99,
        alpha: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.actions = np.linspace(-1.0, 1.0, self.N_ACTIONS)
        self.rbf = RBFFeatureMap(state_dim, n_rbf_features)

        # Weight matrix: (n_actions, n_rbf_features)
        self.W = np.zeros((self.N_ACTIONS, n_rbf_features), dtype=np.float32)
        self.replay = ReplayBuffer()

        self._train_steps = 0
        self._loss_history: List[float] = []

    def _features(self, state: np.ndarray) -> np.ndarray:
        return self.rbf.transform(state.astype(np.float32))

    def q_values(self, state: np.ndarray) -> np.ndarray:
        phi = self._features(state)
        return self.W @ phi  # (n_actions,)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> float:
        if not greedy and random.random() < self.epsilon:
            return float(np.random.choice(self.actions))
        q = self.q_values(state)
        return float(self.actions[int(np.argmax(q))])

    def push(self, state, action, reward, next_state, done) -> None:
        self.replay.push(Transition(state, action, reward, next_state, done))

    def _action_idx(self, action: float) -> int:
        return int(np.argmin(np.abs(self.actions - action)))

    def train_step(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None

        batch = self.replay.sample(self.batch_size)
        total_loss = 0.0

        for t in batch:
            phi = self._features(t.state)
            phi_next = self._features(t.next_state)

            ai = self._action_idx(t.action)
            q_pred = float(self.W[ai] @ phi)

            if t.done:
                target = t.reward
            else:
                q_next = self.W @ phi_next  # (n_actions,)
                target = t.reward + self.gamma * float(q_next.max())

            td_error = target - q_pred
            # Gradient descent on (1/2)(td_error)^2
            self.W[ai] += self.alpha * td_error * phi
            total_loss += td_error ** 2

        self._train_steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        loss = total_loss / self.batch_size
        self._loss_history.append(loss)
        return loss


# ---------------------------------------------------------------------------
# Actor-Critic Agent (Policy Gradient with Baseline)
# ---------------------------------------------------------------------------

class ActorCriticAgent:
    """
    Simple Actor-Critic with linear function approximation on RBF features.
    Actor: Gaussian policy parameterised by mean = w_actor . phi(s)
    Critic: V(s) = w_critic . phi(s)
    """

    def __init__(
        self,
        state_dim: int = State.dim(),
        n_rbf_features: int = 256,
        gamma: float = 0.99,
        actor_lr: float = 1e-4,
        critic_lr: float = 5e-4,
        entropy_coef: float = 0.01,
        action_std: float = 0.3,
    ):
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.entropy_coef = entropy_coef
        self.action_std = action_std

        self.rbf = RBFFeatureMap(state_dim, n_rbf_features)
        self.w_actor = np.zeros(n_rbf_features, dtype=np.float64)
        self.w_critic = np.zeros(n_rbf_features, dtype=np.float64)

        # Eligibility traces for TD(lambda)
        self.e_actor = np.zeros_like(self.w_actor)
        self.e_critic = np.zeros_like(self.w_critic)
        self.lam = 0.9

        self.replay = ReplayBuffer(capacity=20_000)
        self._episode_returns: List[float] = []

    def _features(self, state: np.ndarray) -> np.ndarray:
        return self.rbf.transform(state.astype(np.float32)).astype(np.float64)

    def _value(self, state: np.ndarray) -> float:
        return float(self.w_critic @ self._features(state))

    def _mean_action(self, state: np.ndarray) -> float:
        mu = float(self.w_actor @ self._features(state))
        return float(np.clip(mu, -1.0, 1.0))

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[float, float]:
        """Returns (action, log_prob)."""
        mu = self._mean_action(state)
        if deterministic:
            return mu, 0.0
        noise = np.random.normal(0, self.action_std)
        action = float(np.clip(mu + noise, -1.0, 1.0))
        log_prob = -0.5 * ((action - mu) / self.action_std) ** 2 - np.log(self.action_std * np.sqrt(2 * np.pi))
        return action, float(log_prob)

    def update_online(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
    ) -> Dict[str, float]:
        """Online TD(λ) Actor-Critic update."""
        phi = self._features(state)
        phi_next = self._features(next_state)

        v = float(self.w_critic @ phi)
        v_next = 0.0 if done else float(self.w_critic @ phi_next)
        td_error = reward + self.gamma * v_next - v

        # Critic update with eligibility trace
        self.e_critic = self.gamma * self.lam * self.e_critic + phi
        self.w_critic += self.critic_lr * td_error * self.e_critic

        # Actor update: gradient of log π * advantage
        mu = float(self.w_actor @ phi)
        grad_log_pi = ((action - mu) / (self.action_std ** 2)) * phi
        # Entropy gradient for exploration bonus
        grad_entropy = (1.0 / self.action_std) * phi
        self.e_actor = self.gamma * self.lam * self.e_actor + grad_log_pi
        self.w_actor += self.actor_lr * (td_error * self.e_actor + self.entropy_coef * grad_entropy)

        if done:
            self.e_actor[:] = 0.0
            self.e_critic[:] = 0.0

        return {"td_error": td_error, "v": v}

    def train_from_buffer(self, batch_size: int = 128) -> Optional[float]:
        """Offline batch update from replay buffer (Monte Carlo returns)."""
        if len(self.replay) < batch_size:
            return None

        batch = self.replay.sample(batch_size)
        total_actor_loss = 0.0

        for t in batch:
            phi = self._features(t.state)
            v = float(self.w_critic @ phi)
            v_next = 0.0 if t.done else float(self.w_critic @ self._features(t.next_state))
            td = t.reward + self.gamma * v_next - v

            # Critic
            self.w_critic += self.critic_lr * td * phi

            # Actor
            mu = float(self.w_actor @ phi)
            grad = ((t.action - mu) / (self.action_std ** 2)) * phi
            self.w_actor += self.actor_lr * td * grad

            total_actor_loss += abs(td)

        return total_actor_loss / batch_size


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

class RewardFunction:
    """
    Composite reward for position sizing.
    Encourages risk-adjusted returns while penalising drawdowns and turnover.
    """

    def __init__(
        self,
        sharpe_weight: float = 1.0,
        drawdown_penalty: float = 2.0,
        turnover_penalty: float = 0.1,
        vol_scale: float = 0.02,
    ):
        self.sharpe_weight = sharpe_weight
        self.drawdown_penalty = drawdown_penalty
        self.turnover_penalty = turnover_penalty
        self.vol_scale = vol_scale

        self._pnl_history: deque = deque(maxlen=252)
        self._peak_pnl: float = 0.0
        self._cumulative_pnl: float = 0.0

    def compute(
        self,
        position: float,
        next_return: float,
        prev_position: float,
        volatility: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward given position, realised return, and context.
        Returns (reward, components).
        """
        raw_pnl = position * next_return
        self._cumulative_pnl += raw_pnl
        self._pnl_history.append(raw_pnl)

        # Risk-adjusted PnL (Sharpe-like)
        if len(self._pnl_history) >= 20:
            pnl_arr = np.array(self._pnl_history)
            sharpe_reward = pnl_arr.mean() / (pnl_arr.std() + 1e-8)
        else:
            sharpe_reward = raw_pnl / (volatility * self.vol_scale + 1e-8)

        # Drawdown penalty
        self._peak_pnl = max(self._peak_pnl, self._cumulative_pnl)
        drawdown = max(0.0, self._peak_pnl - self._cumulative_pnl)
        dd_penalty = self.drawdown_penalty * drawdown / (self.vol_scale + 1e-8) if drawdown > 0 else 0.0

        # Turnover penalty
        turnover = abs(position - prev_position)
        to_penalty = self.turnover_penalty * turnover

        reward = self.sharpe_weight * sharpe_reward - dd_penalty - to_penalty

        components = {
            "raw_pnl": raw_pnl,
            "sharpe_reward": sharpe_reward,
            "dd_penalty": dd_penalty,
            "turnover_penalty": to_penalty,
            "total": reward,
        }
        return float(reward), components

    def reset(self) -> None:
        self._pnl_history.clear()
        self._peak_pnl = 0.0
        self._cumulative_pnl = 0.0


# ---------------------------------------------------------------------------
# Kelly criterion integration
# ---------------------------------------------------------------------------

def kelly_position(
    win_prob: float,
    win_return: float,
    loss_return: float,
    max_leverage: float = 1.0,
    fractional: float = 0.5,
) -> float:
    """
    Compute Kelly fraction. fractional < 1 applies half/quarter Kelly.
    win_prob in [0, 1], win_return > 0, loss_return > 0 (magnitude of loss).
    """
    if loss_return <= 0 or win_prob <= 0:
        return 0.0
    edge = win_prob * win_return - (1.0 - win_prob) * loss_return
    kelly = edge / win_return if win_return > 0 else 0.0
    return float(np.clip(fractional * kelly, 0.0, max_leverage))


def kelly_continuous(
    expected_return: float,
    volatility: float,
    risk_free: float = 0.0,
    max_leverage: float = 2.0,
    fractional: float = 0.5,
) -> float:
    """
    Continuous Kelly for lognormal returns: f* = (mu - r) / sigma^2
    """
    if volatility <= 0:
        return 0.0
    f = (expected_return - risk_free) / (volatility ** 2 + 1e-9)
    return float(np.clip(fractional * f, -max_leverage, max_leverage))


# ---------------------------------------------------------------------------
# PositionSizer: main interface
# ---------------------------------------------------------------------------

class PositionSizer:
    """
    High-level position sizer that wraps RL agents and Kelly sizing.
    Supports two modes:
      - 'qlearn': Q-learning with RBF approximation
      - 'ac'    : Actor-Critic policy gradient
    """

    def __init__(
        self,
        mode: str = "ac",
        kelly_blend: float = 0.4,
        max_position: float = 1.0,
        **agent_kwargs,
    ):
        assert mode in ("qlearn", "ac"), f"Unknown mode: {mode}"
        self.mode = mode
        self.kelly_blend = kelly_blend
        self.max_position = max_position

        if mode == "qlearn":
            self.agent: QLearningAgent | ActorCriticAgent = QLearningAgent(**agent_kwargs)
        else:
            self.agent = ActorCriticAgent(**agent_kwargs)

        self.reward_fn = RewardFunction()
        self._current_position: float = 0.0
        self._episode_pnl: List[float] = []
        self._step_count: int = 0

    def act(self, state: State, expected_return: float = 0.0, volatility: float = 0.15) -> float:
        """
        Compute the target position given current state.
        Blends RL output with Kelly sizing.
        """
        s = state.to_array()

        if self.mode == "qlearn":
            rl_position = self.agent.select_action(s)
        else:
            rl_position, _ = self.agent.select_action(s)

        kelly = kelly_continuous(expected_return, volatility, fractional=0.5)
        kelly = float(np.clip(kelly, -self.max_position, self.max_position))

        # Blend: (1 - blend) * RL + blend * Kelly
        blended = (1.0 - self.kelly_blend) * rl_position + self.kelly_blend * kelly
        return float(np.clip(blended, -self.max_position, self.max_position))

    def observe(
        self,
        state: State,
        action: float,
        next_return: float,
        next_state: State,
        done: bool,
        volatility: float = 0.15,
        log_prob: float = 0.0,
    ) -> Dict:
        """Record transition, compute reward, update agent."""
        reward, reward_components = self.reward_fn.compute(
            action, next_return, self._current_position, volatility
        )
        self._current_position = action
        self._step_count += 1

        s = state.to_array()
        ns = next_state.to_array()

        if self.mode == "qlearn":
            self.agent.push(s, action, reward, ns, done)
            loss = self.agent.train_step()
            info = {"loss": loss, **reward_components}
        else:
            info_ac = self.agent.update_online(s, action, reward, ns, done, log_prob)
            self.agent.replay.push(Transition(s, action, reward, ns, done))
            loss = self.agent.train_from_buffer(64)
            info = {**info_ac, "batch_loss": loss, **reward_components}

        if done:
            self.reward_fn.reset()
            self._current_position = 0.0

        return info

    def train_episode(self, price_series: np.ndarray, signal_series: np.ndarray, regime_series: np.ndarray) -> Dict:
        """
        Train on one historical episode.
        price_series: (T,) asset prices
        signal_series: (T,) signal z-scores
        regime_series: (T,) regime labels in {-1, 0, 1}
        """
        T = len(price_series)
        returns = np.diff(np.log(price_series + 1e-9))
        vol_series = np.array([
            np.std(returns[max(0, i-21):i+1]) * np.sqrt(252)
            for i in range(len(returns))
        ])

        episode_reward = 0.0
        position = 0.0
        cumulative_pnl = 0.0

        for t in range(T - 1):
            vol = float(np.clip(vol_series[t], 0.01, 1.0))
            state = State(
                regime=float(regime_series[t]),
                z_score=float(np.clip(signal_series[t], -4, 4)) / 4.0,
                volatility=vol / 0.4,
                current_position=position,
                unrealized_pnl=float(np.clip(cumulative_pnl, -3, 3)) / 3.0,
                time_in_trade=t / (T - 1),
            )

            if self.mode == "qlearn":
                action = self.agent.select_action(state.to_array())
                log_prob = 0.0
            else:
                action, log_prob = self.agent.select_action(state.to_array())

            action = float(np.clip(action, -self.max_position, self.max_position))
            ret = float(returns[t])
            raw_pnl = action * ret
            cumulative_pnl += raw_pnl

            done = (t == T - 2)
            next_vol = float(np.clip(vol_series[min(t+1, len(vol_series)-1)], 0.01, 1.0))
            next_state = State(
                regime=float(regime_series[min(t+1, T-1)]),
                z_score=float(np.clip(signal_series[min(t+1, T-1)], -4, 4)) / 4.0,
                volatility=next_vol / 0.4,
                current_position=action,
                unrealized_pnl=float(np.clip(cumulative_pnl, -3, 3)) / 3.0,
                time_in_trade=(t+1) / (T - 1),
            )

            info = self.observe(state, action, ret, next_state, done, vol, log_prob)
            episode_reward += info.get("total", 0.0)
            position = action

        sharpe = 0.0
        if len(self.reward_fn._pnl_history) > 1:
            arr = np.array(list(self.reward_fn._pnl_history))
            sharpe = float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(252))

        return {
            "episode_reward": episode_reward,
            "cumulative_pnl": cumulative_pnl,
            "sharpe": sharpe,
            "steps": T - 1,
        }

    def get_stats(self) -> Dict:
        return {
            "mode": self.mode,
            "step_count": self._step_count,
            "replay_size": len(self.agent.replay),
            "current_position": self._current_position,
            "epsilon": getattr(self.agent, "epsilon", None),
        }

    def save(self, path: str) -> None:
        data = {
            "mode": self.mode,
            "kelly_blend": self.kelly_blend,
            "w_actor": getattr(self.agent, "w_actor", np.array([])).tolist(),
            "w_critic": getattr(self.agent, "w_critic", np.array([])).tolist(),
            "W": getattr(self.agent, "W", np.array([])).tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        if hasattr(self.agent, "w_actor"):
            self.agent.w_actor = np.array(data["w_actor"])
            self.agent.w_critic = np.array(data["w_critic"])
        elif hasattr(self.agent, "W"):
            self.agent.W = np.array(data["W"])


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    T = 500

    prices = np.cumprod(1 + np.random.normal(0.0002, 0.012, T)) * 100
    signals = np.random.randn(T)
    regimes = np.sign(np.random.randn(T))

    sizer = PositionSizer(mode="ac", kelly_blend=0.3)

    for ep in range(3):
        result = sizer.train_episode(prices, signals, regimes)
        print(f"Episode {ep+1}: reward={result['episode_reward']:.3f}, "
              f"pnl={result['cumulative_pnl']:.4f}, sharpe={result['sharpe']:.3f}")

    # Single step inference
    state = State(regime=0.5, z_score=0.8, volatility=0.5, current_position=0.2,
                  unrealized_pnl=0.01, time_in_trade=0.3)
    pos = sizer.act(state, expected_return=0.001, volatility=0.15)
    print(f"Target position: {pos:.4f}")
    print("Stats:", sizer.get_stats())
