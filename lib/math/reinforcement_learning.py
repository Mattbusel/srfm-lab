"""
reinforcement_learning.py — Tabular and function approximation RL for portfolio.

Covers:
  - TD(lambda) with eligibility traces
  - SARSA on-policy control
  - Expected SARSA
  - Double Q-learning
  - Prioritized Experience Replay buffer
  - Natural Policy Gradient (Fisher information matrix)
  - REINFORCE with baseline
  - TD3-lite (Twin Delayed DDPG with linear FA)
  - Multi-armed bandit: UCB1, Thompson sampling, Exp3
  - Contextual bandit: LinUCB
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

State = np.ndarray      # feature vector
Action = int
Reward = float


# ---------------------------------------------------------------------------
# 1. TD(lambda) with Eligibility Traces
# ---------------------------------------------------------------------------

@dataclass
class TDLambdaConfig:
    alpha: float = 0.01         # learning rate
    gamma: float = 0.99         # discount factor
    lam: float = 0.9            # lambda for eligibility traces
    n_features: int = 10        # dimension of feature vector
    replace_traces: bool = True


class TDLambda:
    """
    TD(lambda) value function approximation with linear function approximation.
    V(s) = w^T phi(s).
    """
    def __init__(self, cfg: TDLambdaConfig):
        self.cfg = cfg
        self.w = np.zeros(cfg.n_features)
        self.e = np.zeros(cfg.n_features)  # eligibility trace

    def value(self, phi: np.ndarray) -> float:
        return float(self.w @ phi)

    def update(self, phi: np.ndarray, reward: float, phi_next: Optional[np.ndarray], done: bool) -> float:
        """Single-step TD(lambda) update. Returns TD error."""
        cfg = self.cfg
        v = self.value(phi)
        v_next = 0.0 if done or phi_next is None else self.value(phi_next)
        delta = reward + cfg.gamma * v_next - v

        if cfg.replace_traces:
            self.e = cfg.gamma * cfg.lam * self.e
            self.e += phi  # replacing traces: set to phi (additive here for simplicity)
        else:
            self.e = cfg.gamma * cfg.lam * self.e + phi

        self.w += cfg.alpha * delta * self.e

        if done:
            self.e[:] = 0.0

        return delta

    def reset_traces(self):
        self.e[:] = 0.0


# ---------------------------------------------------------------------------
# 2. SARSA On-Policy Control
# ---------------------------------------------------------------------------

@dataclass
class SARSAConfig:
    alpha: float = 0.05
    gamma: float = 0.99
    epsilon: float = 0.10
    epsilon_decay: float = 0.999
    epsilon_min: float = 0.01
    n_features: int = 10
    n_actions: int = 3


class SARSA:
    """
    SARSA with linear function approximation.
    Q(s,a) = w_a^T phi(s).
    """
    def __init__(self, cfg: SARSAConfig):
        self.cfg = cfg
        self.W = np.zeros((cfg.n_actions, cfg.n_features))
        self.step = 0

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        return self.W @ phi

    def select_action(self, phi: np.ndarray, rng: np.random.Generator) -> int:
        eps = max(self.cfg.epsilon * (self.cfg.epsilon_decay ** self.step), self.cfg.epsilon_min)
        if rng.random() < eps:
            return int(rng.integers(self.cfg.n_actions))
        return int(np.argmax(self.q_values(phi)))

    def update(
        self,
        phi: np.ndarray,
        action: int,
        reward: float,
        phi_next: np.ndarray,
        next_action: int,
        done: bool,
    ) -> float:
        cfg = self.cfg
        q_sa = float(self.W[action] @ phi)
        q_next = 0.0 if done else float(self.W[next_action] @ phi_next)
        delta = reward + cfg.gamma * q_next - q_sa
        self.W[action] += cfg.alpha * delta * phi
        self.step += 1
        return delta


# ---------------------------------------------------------------------------
# 3. Expected SARSA
# ---------------------------------------------------------------------------

class ExpectedSARSA:
    """
    Expected SARSA: Q(s,a) update uses E_pi[Q(s',a')] under epsilon-greedy.
    """
    def __init__(self, cfg: SARSAConfig):
        self.cfg = cfg
        self.W = np.zeros((cfg.n_actions, cfg.n_features))
        self.step = 0

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        return self.W @ phi

    def _epsilon(self) -> float:
        return max(self.cfg.epsilon * (self.cfg.epsilon_decay ** self.step), self.cfg.epsilon_min)

    def _policy_probs(self, phi: np.ndarray) -> np.ndarray:
        n = self.cfg.n_actions
        eps = self._epsilon()
        q = self.q_values(phi)
        probs = np.full(n, eps / n)
        probs[int(np.argmax(q))] += 1.0 - eps
        return probs

    def select_action(self, phi: np.ndarray, rng: np.random.Generator) -> int:
        probs = self._policy_probs(phi)
        return int(rng.choice(self.cfg.n_actions, p=probs))

    def update(
        self,
        phi: np.ndarray,
        action: int,
        reward: float,
        phi_next: np.ndarray,
        done: bool,
    ) -> float:
        cfg = self.cfg
        q_sa = float(self.W[action] @ phi)
        if done:
            expected_q = 0.0
        else:
            probs = self._policy_probs(phi_next)
            expected_q = float(probs @ self.q_values(phi_next))
        delta = reward + cfg.gamma * expected_q - q_sa
        self.W[action] += cfg.alpha * delta * phi
        self.step += 1
        return delta


# ---------------------------------------------------------------------------
# 4. Double Q-Learning
# ---------------------------------------------------------------------------

class DoubleQLearning:
    """
    Double Q-learning with linear FA.
    Maintains two weight matrices W1 and W2 to reduce overestimation.
    """
    def __init__(self, cfg: SARSAConfig):
        self.cfg = cfg
        self.W1 = np.zeros((cfg.n_actions, cfg.n_features))
        self.W2 = np.zeros((cfg.n_actions, cfg.n_features))
        self.step = 0

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        return (self.W1 @ phi + self.W2 @ phi) / 2.0

    def select_action(self, phi: np.ndarray, rng: np.random.Generator) -> int:
        eps = max(self.cfg.epsilon * (self.cfg.epsilon_decay ** self.step), self.cfg.epsilon_min)
        if rng.random() < eps:
            return int(rng.integers(self.cfg.n_actions))
        return int(np.argmax(self.q_values(phi)))

    def update(
        self,
        phi: np.ndarray,
        action: int,
        reward: float,
        phi_next: np.ndarray,
        done: bool,
        rng: np.random.Generator,
    ) -> float:
        cfg = self.cfg
        if rng.random() < 0.5:
            # Update W1 using W2 for next-state evaluation
            best_a = int(np.argmax(self.W1 @ phi_next))
            q_next = 0.0 if done else float(self.W2[best_a] @ phi_next)
            q_sa = float(self.W1[action] @ phi)
            delta = reward + cfg.gamma * q_next - q_sa
            self.W1[action] += cfg.alpha * delta * phi
        else:
            # Update W2 using W1
            best_a = int(np.argmax(self.W2 @ phi_next))
            q_next = 0.0 if done else float(self.W1[best_a] @ phi_next)
            q_sa = float(self.W2[action] @ phi)
            delta = reward + cfg.gamma * q_next - q_sa
            self.W2[action] += cfg.alpha * delta * phi

        self.step += 1
        return delta


# ---------------------------------------------------------------------------
# 5. Prioritized Experience Replay Buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    priority: float = 1.0


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (Schaul et al. 2016).
    Priority = |TD error|^alpha.
    Importance-sampling weights correct for non-uniform sampling.
    """
    def __init__(self, capacity: int = 10_000, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 1e-4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer: List[Transition] = []
        self.pos = 0
        self._priorities: np.ndarray = np.zeros(capacity, dtype=float)

    def push(self, t: Transition):
        max_prio = float(self._priorities[:len(self.buffer)].max()) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(t)
        else:
            self.buffer[self.pos] = t
        self._priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        n = len(self.buffer)
        prios = self._priorities[:n] ** self.alpha
        prios_sum = prios.sum()
        probs = prios / prios_sum
        indices = rng.choice(n, size=batch_size, p=probs, replace=False if batch_size <= n else True)
        samples = [self.buffer[i] for i in indices]

        # IS weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, eps: float = 1e-6):
        for i, err in zip(indices, td_errors):
            self._priorities[i] = abs(float(err)) + eps

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 6. Natural Policy Gradient
# ---------------------------------------------------------------------------

@dataclass
class NPGConfig:
    alpha: float = 0.01
    gamma: float = 0.99
    n_features: int = 10
    n_actions: int = 3
    cg_iters: int = 10          # conjugate gradient iterations for F^{-1} g
    damping: float = 1e-3


class NaturalPolicyGradient:
    """
    Natural Policy Gradient using the Fisher information matrix.
    Policy: softmax over linear scores theta^T phi.
    Update: theta += alpha * F^{-1} * grad_J.
    Uses conjugate gradient to avoid explicit inversion of F.
    """
    def __init__(self, cfg: NPGConfig):
        self.cfg = cfg
        self.theta = np.zeros((cfg.n_actions, cfg.n_features))
        self.baseline_w = np.zeros(cfg.n_features)

    def policy_probs(self, phi: np.ndarray) -> np.ndarray:
        scores = self.theta @ phi
        scores -= scores.max()
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum()

    def select_action(self, phi: np.ndarray, rng: np.random.Generator) -> int:
        probs = self.policy_probs(phi)
        return int(rng.choice(self.cfg.n_actions, p=probs))

    def _log_grad(self, phi: np.ndarray, action: int) -> np.ndarray:
        """Gradient of log pi(a|s) w.r.t. theta, returns (n_actions, n_features) flat vector."""
        probs = self.policy_probs(phi)
        grad = np.zeros_like(self.theta)
        grad[action] += phi
        for a in range(self.cfg.n_actions):
            grad[a] -= probs[a] * phi
        return grad.ravel()

    def _fisher_vector_product(self, v: np.ndarray, states: List[np.ndarray]) -> np.ndarray:
        """Approximates F*v by averaging over states."""
        Fv = np.zeros_like(v)
        for phi in states:
            g = self._log_grad(phi, 0)  # simplified: use action 0 as representative
            Fv += g * (g @ v)
        Fv /= max(len(states), 1)
        return Fv + self.cfg.damping * v

    def _conjugate_gradient(self, b: np.ndarray, states: List[np.ndarray]) -> np.ndarray:
        x = np.zeros_like(b)
        r = b.copy()
        p = b.copy()
        rr = float(r @ r)
        for _ in range(self.cfg.cg_iters):
            Ap = self._fisher_vector_product(p, states)
            alpha_cg = rr / max(float(p @ Ap), 1e-10)
            x += alpha_cg * p
            r -= alpha_cg * Ap
            rr_new = float(r @ r)
            if rr_new < 1e-10:
                break
            p = r + (rr_new / rr) * p
            rr = rr_new
        return x

    def update(
        self,
        trajectory: List[Tuple[np.ndarray, int, float]],  # (phi, action, advantage)
    ):
        """Update policy using natural gradient. Trajectory = list of (phi, a, advantage)."""
        if not trajectory:
            return

        grad = np.zeros(self.cfg.n_actions * self.cfg.n_features)
        states = [phi for phi, _, _ in trajectory]
        for phi, action, advantage in trajectory:
            lg = self._log_grad(phi, action)
            grad += lg * advantage
        grad /= len(trajectory)

        nat_grad = self._conjugate_gradient(grad, states)
        self.theta += self.cfg.alpha * nat_grad.reshape(self.cfg.n_actions, self.cfg.n_features)


# ---------------------------------------------------------------------------
# 7. REINFORCE with Baseline
# ---------------------------------------------------------------------------

@dataclass
class REINFORCEConfig:
    alpha_policy: float = 0.001
    alpha_baseline: float = 0.01
    gamma: float = 0.99
    n_features: int = 10
    n_actions: int = 3


class REINFORCE:
    """
    REINFORCE (Williams 1992) with a linear baseline to reduce variance.
    Policy: softmax over linear scores.
    Baseline: V(s) = w^T phi(s).
    """
    def __init__(self, cfg: REINFORCEConfig):
        self.cfg = cfg
        self.theta = np.zeros((cfg.n_actions, cfg.n_features))
        self.w = np.zeros(cfg.n_features)

    def policy_probs(self, phi: np.ndarray) -> np.ndarray:
        scores = self.theta @ phi
        scores -= scores.max()
        e = np.exp(scores)
        return e / e.sum()

    def select_action(self, phi: np.ndarray, rng: np.random.Generator) -> int:
        return int(rng.choice(self.cfg.n_actions, p=self.policy_probs(phi)))

    def update_episode(self, episode: List[Tuple[np.ndarray, int, float]]):
        """
        episode: list of (phi, action, reward).
        Computes returns and updates policy + baseline.
        """
        cfg = self.cfg
        T = len(episode)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = episode[t][2] + cfg.gamma * G
            returns[t] = G

        for t, (phi, action, _) in enumerate(episode):
            G_t = returns[t]
            baseline = float(self.w @ phi)
            advantage = G_t - baseline

            # Update baseline
            self.w += cfg.alpha_baseline * advantage * phi

            # Update policy
            probs = self.policy_probs(phi)
            for a in range(cfg.n_actions):
                if a == action:
                    self.theta[a] += cfg.alpha_policy * cfg.gamma ** t * advantage * (1.0 - probs[a]) * phi
                else:
                    self.theta[a] -= cfg.alpha_policy * cfg.gamma ** t * advantage * probs[a] * phi


# ---------------------------------------------------------------------------
# 8. TD3-lite (Twin Delayed DDPG with Linear FA)
# ---------------------------------------------------------------------------

@dataclass
class TD3LiteConfig:
    alpha_actor: float = 0.001
    alpha_critic: float = 0.005
    gamma: float = 0.99
    tau: float = 0.005          # soft update rate
    policy_noise: float = 0.1
    noise_clip: float = 0.2
    policy_delay: int = 2
    n_features: int = 10
    action_dim: int = 1
    action_scale: float = 1.0


class TD3Lite:
    """
    TD3-lite: Twin Delayed DDPG with linear function approximation.
    Actor: mu(s) = W_actor @ phi(s), clipped to [-action_scale, action_scale].
    Twin critics: Q1(s,a) = w1 @ [phi, a], Q2(s,a) = w2 @ [phi, a].
    """
    def __init__(self, cfg: TD3LiteConfig):
        self.cfg = cfg
        feat = cfg.n_features + cfg.action_dim
        self.W_actor = np.zeros((cfg.action_dim, cfg.n_features))
        self.W_actor_tgt = self.W_actor.copy()
        self.w_c1 = np.zeros(feat)
        self.w_c2 = np.zeros(feat)
        self.w_c1_tgt = self.w_c1.copy()
        self.w_c2_tgt = self.w_c2.copy()
        self._update_count = 0

    def _phi_sa(self, phi: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.concatenate([phi, action.ravel()])

    def select_action(self, phi: np.ndarray, noise: float = 0.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        a = self.W_actor @ phi
        if noise > 0 and rng is not None:
            a += rng.standard_normal(self.cfg.action_dim) * noise
        return np.clip(a, -self.cfg.action_scale, self.cfg.action_scale)

    def update(self, batch: List[Transition], rng: np.random.Generator):
        cfg = self.cfg
        self._update_count += 1

        for t in batch:
            phi = t.state
            a = np.array([t.action], dtype=float) if isinstance(t.action, int) else t.action
            phi_next = t.next_state

            # Target action with noise
            a_next = self.W_actor_tgt @ phi_next
            noise = np.clip(rng.standard_normal(cfg.action_dim) * cfg.policy_noise,
                            -cfg.noise_clip, cfg.noise_clip)
            a_next = np.clip(a_next + noise, -cfg.action_scale, cfg.action_scale)

            # Target Q
            sa_next = self._phi_sa(phi_next, a_next)
            q1_tgt = float(self.w_c1_tgt @ sa_next)
            q2_tgt = float(self.w_c2_tgt @ sa_next)
            y = t.reward + (0.0 if t.done else cfg.gamma * min(q1_tgt, q2_tgt))

            # Critic update
            sa = self._phi_sa(phi, a)
            for w_c in [self.w_c1, self.w_c2]:
                err = y - float(w_c @ sa)
                w_c += cfg.alpha_critic * err * sa

            # Delayed actor update
            if self._update_count % cfg.policy_delay == 0:
                a_curr = self.W_actor @ phi
                sa_curr = self._phi_sa(phi, a_curr)
                q_grad_a = self.w_c1[cfg.n_features:]  # dQ/da from critic weights
                self.W_actor += cfg.alpha_actor * np.outer(q_grad_a, phi)

                # Soft target updates
                self.W_actor_tgt = cfg.tau * self.W_actor + (1 - cfg.tau) * self.W_actor_tgt
                self.w_c1_tgt = cfg.tau * self.w_c1 + (1 - cfg.tau) * self.w_c1_tgt
                self.w_c2_tgt = cfg.tau * self.w_c2 + (1 - cfg.tau) * self.w_c2_tgt


# ---------------------------------------------------------------------------
# 9. Multi-Armed Bandit: UCB1, Thompson Sampling, Exp3
# ---------------------------------------------------------------------------

class UCB1Bandit:
    """UCB1 algorithm (Auer et al. 2002)."""
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select(self) -> int:
        self.t += 1
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a
        ucb = self.values + np.sqrt(2.0 * math.log(self.t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class ThompsonSamplingBandit:
    """Thompson Sampling with Beta posterior for Bernoulli rewards."""
    def __init__(self, n_arms: int, seed: int = 0):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)   # successes + 1
        self.beta_ = np.ones(n_arms)   # failures + 1
        self.rng = np.random.default_rng(seed)

    def select(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta_)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        """reward should be in [0,1]; treated as Bernoulli sample."""
        self.alpha[arm] += reward
        self.beta_[arm] += 1.0 - reward


class Exp3Bandit:
    """Exp3 algorithm for adversarial bandits (Auer et al. 1995)."""
    def __init__(self, n_arms: int, gamma: float = 0.1):
        self.n_arms = n_arms
        self.gamma = gamma
        self.weights = np.ones(n_arms)
        self.rng = np.random.default_rng(0)

    def _probs(self) -> np.ndarray:
        w = self.weights
        uniform = np.ones(self.n_arms) / self.n_arms
        p = (1.0 - self.gamma) * (w / w.sum()) + self.gamma * uniform
        return p

    def select(self) -> int:
        p = self._probs()
        return int(self.rng.choice(self.n_arms, p=p))

    def update(self, arm: int, reward: float):
        p = self._probs()
        x_hat = reward / (p[arm] + 1e-10)
        self.weights[arm] *= math.exp(self.gamma * x_hat / self.n_arms)
        # Normalize to prevent overflow
        self.weights /= self.weights.max()


# ---------------------------------------------------------------------------
# 10. Contextual Bandit: LinUCB
# ---------------------------------------------------------------------------

@dataclass
class LinUCBConfig:
    n_arms: int = 5
    n_features: int = 10
    alpha: float = 1.0      # exploration parameter


class LinUCB:
    """
    LinUCB (Li et al. 2010) contextual bandit.
    For each arm a: UCB = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x).
    """
    def __init__(self, cfg: LinUCBConfig):
        self.cfg = cfg
        d = cfg.n_features
        self.A = [np.eye(d) for _ in range(cfg.n_arms)]
        self.b = [np.zeros(d) for _ in range(cfg.n_arms)]
        self._theta = [np.zeros(d) for _ in range(cfg.n_arms)]

    def _update_theta(self, arm: int):
        self._theta[arm] = np.linalg.solve(self.A[arm], self.b[arm])

    def select(self, context: np.ndarray) -> int:
        ucb_scores = np.zeros(self.cfg.n_arms)
        for a in range(self.cfg.n_arms):
            self._update_theta(a)
            theta = self._theta[a]
            A_inv = np.linalg.inv(self.A[a])
            ucb_scores[a] = float(theta @ context) + self.cfg.alpha * math.sqrt(float(context @ A_inv @ context))
        return int(np.argmax(ucb_scores))

    def update(self, arm: int, context: np.ndarray, reward: float):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context


# ---------------------------------------------------------------------------
# Portfolio RL Environment (thin interface)
# ---------------------------------------------------------------------------

@dataclass
class PortfolioRLState:
    features: np.ndarray        # current feature vector
    portfolio_weights: np.ndarray
    cash: float
    step: int


class PortfolioRLEnv:
    """
    Minimal portfolio RL environment wrapper.
    State: feature vector of shape (n_features,).
    Action: discrete allocation bucket (0=all_cash, 1=50/50, 2=all_risky).
    Reward: realized log return of portfolio - transaction_cost * |delta_weights|.
    """
    def __init__(
        self,
        returns: np.ndarray,        # shape (T, N) asset returns
        features: np.ndarray,       # shape (T, n_features)
        transaction_cost: float = 0.001,
        n_action_buckets: int = 3,
    ):
        self.returns = returns
        self.features = features
        self.tc = transaction_cost
        self.n_actions = n_action_buckets
        self.T, self.N = returns.shape
        self._allocations = np.linspace(0.0, 1.0, n_action_buckets)
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.weights = np.zeros(self.N)
        self.cash = 1.0
        return self.features[0].copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        alloc = self._allocations[action]
        new_weights = np.full(self.N, alloc / self.N)
        delta = np.abs(new_weights - self.weights).sum()
        tc_cost = self.tc * delta

        port_return = float(new_weights @ self.returns[self.t])
        reward = math.log(1.0 + port_return) - tc_cost

        self.weights = new_weights
        self.t += 1
        done = self.t >= self.T - 1
        next_state = self.features[min(self.t, self.T - 1)].copy()
        return next_state, reward, done
