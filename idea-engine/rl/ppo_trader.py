"""
PPO (Proximal Policy Optimization) trading agent — pure numpy, no ML libs.

Components:
  ActorNetwork    — policy π(a|s), discrete or continuous Gaussian
  CriticNetwork   — value function V(s)
  PPOBuffer       — trajectory storage with GAE advantage estimation
  PPOAgent        — full PPO training loop with entropy bonus, LR scheduling,
                    gradient clipping
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Activation functions and helpers
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)

def tanh_act(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-15)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def _clip_grad_norm(grads: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
    total_norm = math.sqrt(sum(float(np.sum(g ** 2)) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        return [g * scale for g in grads]
    return grads


# ---------------------------------------------------------------------------
# Adam optimizer (per parameter group)
# ---------------------------------------------------------------------------

class AdamOptimizer:
    def __init__(self, lr: float = 3e-4, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m: Dict[int, np.ndarray] = {}
        self._v: Dict[int, np.ndarray] = {}

    def step(self, param_id: int, param: np.ndarray,
             grad: np.ndarray) -> np.ndarray:
        if param_id not in self._m:
            self._m[param_id] = np.zeros_like(param)
            self._v[param_id] = np.zeros_like(param)
        self.t += 1
        self._m[param_id] = self.beta1 * self._m[param_id] + (1 - self.beta1) * grad
        self._v[param_id] = self.beta2 * self._v[param_id] + (1 - self.beta2) * grad ** 2
        m_hat = self._m[param_id] / (1 - self.beta1 ** self.t)
        v_hat = self._v[param_id] / (1 - self.beta2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def set_lr(self, lr: float):
        self.lr = lr


# ---------------------------------------------------------------------------
# MLP core (shared by actor and critic)
# ---------------------------------------------------------------------------

class MLP:
    """
    Simple multi-layer perceptron with tanh activations (good for RL).
    Supports forward, backward, and parameter update via external optimizer.
    """

    def __init__(self, layer_dims: List[int], activation: str = "tanh",
                 output_activation: str = "linear"):
        assert len(layer_dims) >= 2
        self.layers_W: List[np.ndarray] = []
        self.layers_b: List[np.ndarray] = []
        self.activation = activation
        self.output_activation = output_activation
        self._cache: List[Dict] = []

        act_fn = tanh_act if activation == "tanh" else relu
        scale_fn = (lambda n_in: math.sqrt(1.0 / n_in)) if activation == "tanh" else (lambda n_in: math.sqrt(2.0 / n_in))

        for i in range(len(layer_dims) - 1):
            n_in, n_out = layer_dims[i], layer_dims[i + 1]
            s = scale_fn(n_in) if i < len(layer_dims) - 2 else math.sqrt(0.01 / n_in)
            self.layers_W.append(np.random.randn(n_in, n_out) * s)
            self.layers_b.append(np.zeros(n_out))

        self.n_layers = len(self.layers_W)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = []
        h = x
        act = tanh_act if self.activation == "tanh" else relu

        for i, (W, b) in enumerate(zip(self.layers_W, self.layers_b)):
            z = h @ W + b
            is_last = (i == self.n_layers - 1)
            if is_last:
                if self.output_activation == "softmax":
                    a = softmax(z)
                elif self.output_activation == "sigmoid":
                    a = sigmoid(z)
                else:
                    a = z  # linear
            else:
                a = act(z)
            self._cache.append({"h_in": h, "z": z, "a": a, "is_last": is_last})
            h = a
        return h

    def backward(self, dout: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Returns (dW_list, db_list, dx)."""
        dW_list = [None] * self.n_layers
        db_list = [None] * self.n_layers
        act_g = tanh_grad if self.activation == "tanh" else relu_grad

        d = dout
        for i in reversed(range(self.n_layers)):
            cache = self._cache[i]
            h_in = cache["h_in"]
            z = cache["z"]
            is_last = cache["is_last"]

            if is_last:
                if self.output_activation in ("softmax", "linear"):
                    dz = d  # assume pre-softmax gradient passed in
                elif self.output_activation == "sigmoid":
                    dz = d * sigmoid(z) * (1 - sigmoid(z))
                else:
                    dz = d
            else:
                dz = d * act_g(z)

            dW_list[i] = h_in.T @ dz
            db_list[i] = dz.sum(axis=0)
            d = dz @ self.layers_W[i].T

        return dW_list, db_list, d

    def param_count(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.layers_W, self.layers_b))


# ---------------------------------------------------------------------------
# Actor Network
# ---------------------------------------------------------------------------

class ActorNetwork:
    """
    Policy network.
    - Discrete: outputs logits → softmax → categorical distribution.
    - Continuous: outputs (mean, log_std) → Gaussian distribution.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int], discrete: bool = True,
                 log_std_init: float = -0.5, log_std_min: float = -3.0,
                 log_std_max: float = 0.5):
        self.discrete = discrete
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        if discrete:
            dims = [state_dim] + hidden_dims + [action_dim]
            self.mlp = MLP(dims, activation="tanh", output_activation="linear")
        else:
            dims = [state_dim] + hidden_dims + [action_dim]
            self.mlp = MLP(dims, activation="tanh", output_activation="linear")
            self.log_std = np.full(action_dim, log_std_init)

        self.optimizer = AdamOptimizer(lr=3e-4)
        self._last_logits: Optional[np.ndarray] = None

    def forward(self, states: np.ndarray) -> np.ndarray:
        return self.mlp.forward(states)

    def get_distribution(self, states: np.ndarray
                         ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Returns (probs/means, stds)."""
        logits_or_means = self.forward(states)
        if self.discrete:
            probs = softmax(logits_or_means)
            self._last_logits = logits_or_means
            return probs, None
        else:
            means = logits_or_means
            log_std = np.clip(self.log_std, self.log_std_min, self.log_std_max)
            stds = np.exp(log_std)
            return means, stds

    def sample_action(self, state: np.ndarray,
                      deterministic: bool = False
                      ) -> Tuple[Any, float]:
        """
        Returns (action, log_prob).
        """
        state = np.atleast_2d(state)
        if self.discrete:
            probs, _ = self.get_distribution(state)
            probs = probs[0]
            if deterministic:
                action = int(np.argmax(probs))
            else:
                action = int(np.random.choice(len(probs), p=probs))
            log_prob = float(np.log(probs[action] + 1e-8))
            return action, log_prob
        else:
            means, stds = self.get_distribution(state)
            means, stds = means[0], stds
            if deterministic:
                action = means
            else:
                action = means + stds * np.random.randn(*means.shape)
            action = np.clip(action, -1.0, 1.0)
            log_prob = float(self._gaussian_log_prob(action, means, stds).sum())
            return action, log_prob

    def log_prob(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Returns log π(a|s) for each (s,a) pair.
        states: (B, state_dim), actions: (B,) or (B, action_dim)
        """
        if self.discrete:
            probs, _ = self.get_distribution(states)
            actions_int = actions.astype(int).ravel()
            lp = np.log(probs[np.arange(len(actions_int)), actions_int] + 1e-8)
            return lp
        else:
            means, stds = self.get_distribution(states)
            lp = self._gaussian_log_prob(actions, means, stds).sum(axis=-1)
            return lp

    def entropy(self, states: np.ndarray) -> np.ndarray:
        """Entropy of the policy distribution."""
        if self.discrete:
            probs, _ = self.get_distribution(states)
            return -np.sum(probs * np.log(probs + 1e-8), axis=-1)
        else:
            log_std = np.clip(self.log_std, self.log_std_min, self.log_std_max)
            return np.full(len(states),
                           float(np.sum(log_std + 0.5 * np.log(2 * math.pi * math.e))))

    @staticmethod
    def _gaussian_log_prob(x: np.ndarray, mu: np.ndarray,
                           std: np.ndarray) -> np.ndarray:
        var = std ** 2 + 1e-8
        return -0.5 * (np.log(2 * math.pi * var) + (x - mu) ** 2 / var)

    def compute_ppo_gradient(self, states: np.ndarray, actions: np.ndarray,
                              advantages: np.ndarray, old_log_probs: np.ndarray,
                              clip_eps: float, entropy_coef: float
                              ) -> Tuple[float, float]:
        """
        Compute PPO clip objective gradient and update weights.
        Returns (policy_loss, entropy).
        """
        B = len(states)
        new_log_probs = self.log_prob(states, actions)
        entropy = float(self.entropy(states).mean())
        ratio = np.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -float(np.minimum(surr1, surr2).mean()) - entropy_coef * entropy

        # Gradient through log_prob wrt network parameters
        # dL/d(log_prob) for each sample
        surr_select = np.where(
            (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps),
            np.zeros(B),
            advantages
        )
        d_log_prob = -surr_select / B - entropy_coef * (-1.0 - np.log(
            softmax(self.mlp.forward(states)).max(axis=-1) + 1e-8
        )) / B if self.discrete else -surr_select / B

        if self.discrete:
            probs = softmax(self.mlp.forward(states))
            # Gradient of log_prob wrt logits: d log p_a / d logits = one_hot - probs
            actions_int = actions.astype(int).ravel()
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(B), actions_int] = 1.0
            d_logits = (one_hot - probs) * d_log_prob[:, None]
            # Entropy gradient
            d_logits += entropy_coef * probs * (np.log(probs + 1e-8) + entropy) / B
            dW, db, _ = self.mlp.backward(d_logits)
        else:
            means, stds = self.get_distribution(states)
            d_mean = -(actions - means) / (stds ** 2 + 1e-8) * d_log_prob[:, None]
            dW, db, _ = self.mlp.backward(d_mean)
            # Update log_std
            d_log_std_sample = (-(actions - means) ** 2 / (stds ** 2 + 1e-8) + 1.0) * d_log_prob[:, None]
            d_log_std = d_log_std_sample.sum(axis=0)
            d_log_std -= entropy_coef * np.ones(self.action_dim)  # entropy wrt log_std = +1
            grad_clipped = _clip_grad_norm([d_log_std], 0.5)[0]
            self.log_std -= 3e-4 * grad_clipped  # simple SGD for log_std

        # Clip and apply gradients
        all_grads = dW + db
        all_grads = _clip_grad_norm(all_grads, max_norm=0.5)
        dW_clipped = all_grads[:self.mlp.n_layers]
        db_clipped = all_grads[self.mlp.n_layers:]

        for i in range(self.mlp.n_layers):
            self.mlp.layers_W[i] = self.optimizer.step(
                i * 2, self.mlp.layers_W[i], dW_clipped[i])
            self.mlp.layers_b[i] = self.optimizer.step(
                i * 2 + 1, self.mlp.layers_b[i], db_clipped[i])

        return policy_loss, entropy


# ---------------------------------------------------------------------------
# Critic Network
# ---------------------------------------------------------------------------

class CriticNetwork:
    """
    Value function V(s) → scalar estimate.
    """

    def __init__(self, state_dim: int, hidden_dims: List[int]):
        dims = [state_dim] + hidden_dims + [1]
        self.mlp = MLP(dims, activation="tanh", output_activation="linear")
        self.optimizer = AdamOptimizer(lr=1e-3)

    def forward(self, states: np.ndarray) -> np.ndarray:
        return self.mlp.forward(states).ravel()

    def compute_loss_and_update(self, states: np.ndarray,
                                 returns: np.ndarray,
                                 clip_val: Optional[float] = None,
                                 old_values: Optional[np.ndarray] = None
                                 ) -> float:
        """
        MSE value loss with optional clipping (PPO-style).
        """
        values = self.mlp.forward(states)  # (B, 1)
        values_flat = values.ravel()
        targets = returns

        if clip_val is not None and old_values is not None:
            v_clipped = old_values + np.clip(values_flat - old_values, -clip_val, clip_val)
            loss1 = (values_flat - targets) ** 2
            loss2 = (v_clipped - targets) ** 2
            loss = float(np.maximum(loss1, loss2).mean())
            dv = 2.0 * np.where(loss1 >= loss2,
                                  values_flat - targets,
                                  v_clipped - targets) / len(targets)
        else:
            loss = float(np.mean((values_flat - targets) ** 2))
            dv = 2.0 * (values_flat - targets) / len(targets)

        dout = dv[:, None]
        dW, db, _ = self.mlp.backward(dout)
        all_grads = _clip_grad_norm(dW + db, max_norm=0.5)
        dW_c = all_grads[:self.mlp.n_layers]
        db_c = all_grads[self.mlp.n_layers:]

        for i in range(self.mlp.n_layers):
            self.mlp.layers_W[i] = self.optimizer.step(
                i * 2, self.mlp.layers_W[i], dW_c[i])
            self.mlp.layers_b[i] = self.optimizer.step(
                i * 2 + 1, self.mlp.layers_b[i], db_c[i])

        return loss


# ---------------------------------------------------------------------------
# PPO Buffer with GAE
# ---------------------------------------------------------------------------

class PPOBuffer:
    """
    Trajectory buffer for PPO. Stores one rollout of length n_steps.
    Computes GAE advantages after rollout completes.
    """

    def __init__(self, state_dim: int, action_dim: int, n_steps: int,
                 gamma: float = 0.99, lam: float = 0.95):
        self.n = n_steps
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.full = False

        self.states = np.zeros((n_steps, state_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, max(action_dim, 1)), dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

    def push(self, state: np.ndarray, action: Any, reward: float,
             value: float, log_prob: float, done: bool):
        i = self.ptr % self.n
        self.states[i] = state
        self.actions[i] = np.atleast_1d(action).astype(np.float32)
        self.rewards[i] = reward
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.dones[i] = float(done)
        self.ptr += 1
        if self.ptr >= self.n:
            self.full = True

    def compute_advantages(self, last_value: float = 0.0):
        """
        GAE-Lambda advantage estimation.
        last_value: V(s_{T+1}), 0 if terminal.
        """
        n = self.n
        gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else self.values[t + 1]
            next_done = self.dones[t]
            delta = self.rewards[t] + self.gamma * next_val * (1.0 - next_done) - self.values[t]
            gae = delta + self.gamma * self.lam * (1.0 - next_done) * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values
        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

    def get_batches(self, batch_size: int) -> List[Dict[str, np.ndarray]]:
        """Yield shuffled mini-batches from buffer."""
        idx = np.random.permutation(self.n)
        batches = []
        for start in range(0, self.n, batch_size):
            b = idx[start:start + batch_size]
            batches.append({
                "states": self.states[b],
                "actions": self.actions[b],
                "old_log_probs": self.log_probs[b],
                "advantages": self.advantages[b],
                "returns": self.returns[b],
                "old_values": self.values[b],
            })
        return batches

    def reset(self):
        self.ptr = 0
        self.full = False


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    state_dim: int
    action_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    discrete: bool = True
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    target_kl: float = 0.01
    lr_schedule: str = "linear"  # "linear" or "constant"
    clip_value_loss: bool = True


class PPOAgent:
    """
    Proximal Policy Optimization agent.

    Usage:
        agent = PPOAgent(config)
        agent.train(env, n_iterations=100)
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.actor = ActorNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            discrete=config.discrete,
        )
        self.actor.optimizer = AdamOptimizer(lr=config.actor_lr)
        self.critic = CriticNetwork(
            state_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
        )
        self.critic.optimizer = AdamOptimizer(lr=config.critic_lr)
        self.buffer = PPOBuffer(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            n_steps=config.n_steps,
            gamma=config.gamma,
            lam=config.lam,
        )
        self.total_steps = 0
        self.n_updates = 0
        self._train_history: List[Dict] = []

    def collect_rollout(self, env: Any, n_steps: int) -> Dict[str, float]:
        """
        Collect n_steps of experience from env, filling the buffer.
        Returns rollout stats: mean_reward, episode_returns.
        """
        self.buffer.reset()
        state = env.reset()
        episode_return = 0.0
        episode_returns = []
        n_episodes = 0

        for _ in range(n_steps):
            state_t = np.asarray(state, dtype=np.float32)
            value = float(self.critic.forward(state_t[None])[0])
            action, log_prob = self.actor.sample_action(state_t)

            next_state, reward, done, info = env.step(action)
            episode_return += reward

            self.buffer.push(state_t, action, reward, value, log_prob, done)
            self.total_steps += 1

            if done:
                episode_returns.append(episode_return)
                episode_return = 0.0
                n_episodes += 1
                state = env.reset()
            else:
                state = next_state

        # Compute last value for GAE bootstrap
        state_t = np.asarray(state, dtype=np.float32)
        last_value = float(self.critic.forward(state_t[None])[0])
        self.buffer.compute_advantages(last_value=last_value if not done else 0.0)

        mean_ret = float(np.mean(episode_returns)) if episode_returns else episode_return
        return {
            "mean_episode_return": mean_ret,
            "n_episodes": n_episodes,
            "total_steps": self.total_steps,
        }

    def update_policy(self, n_epochs: int, clip_eps: float,
                      batch_size: int) -> Tuple[float, float, float]:
        """
        PPO policy update. Returns (mean_policy_loss, mean_entropy, mean_kl).
        """
        total_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_batches = 0

        for epoch in range(n_epochs):
            batches = self.buffer.get_batches(batch_size)
            for batch in batches:
                states = batch["states"]
                actions = batch["actions"]
                old_lp = batch["old_log_probs"]
                advantages = batch["advantages"]

                # Compute approximate KL before update
                new_lp = self.actor.log_prob(states, actions.ravel() if self.config.discrete else actions)
                approx_kl = float(np.mean(old_lp - new_lp))
                if approx_kl > 1.5 * self.config.target_kl:
                    break

                loss, ent = self.actor.compute_ppo_gradient(
                    states=states,
                    actions=actions.ravel() if self.config.discrete else actions,
                    advantages=advantages,
                    old_log_probs=old_lp,
                    clip_eps=clip_eps,
                    entropy_coef=self.config.entropy_coef,
                )
                total_loss += loss
                total_entropy += ent
                total_kl += approx_kl
                n_batches += 1

        if n_batches == 0:
            return 0.0, 0.0, 0.0
        return total_loss / n_batches, total_entropy / n_batches, total_kl / n_batches

    def update_value(self, n_epochs: int, batch_size: int) -> float:
        """Value function update. Returns mean value loss."""
        total_loss = 0.0
        n_batches = 0
        for epoch in range(n_epochs):
            batches = self.buffer.get_batches(batch_size)
            for batch in batches:
                v_loss = self.critic.compute_loss_and_update(
                    states=batch["states"],
                    returns=batch["returns"],
                    clip_val=self.config.clip_eps if self.config.clip_value_loss else None,
                    old_values=batch["old_values"],
                )
                total_loss += v_loss
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def _schedule_lr(self, iteration: int, total_iterations: int):
        """Linearly decay learning rate to zero."""
        if self.config.lr_schedule == "linear":
            frac = 1.0 - iteration / total_iterations
            self.actor.optimizer.set_lr(self.config.actor_lr * frac)
            self.critic.optimizer.set_lr(self.config.critic_lr * frac)

    def train(self, env: Any, n_iterations: int,
              verbose: bool = True, eval_env: Any = None,
              eval_freq: int = 10) -> List[Dict]:
        """
        Full PPO training loop.

        Parameters
        ----------
        env          : training environment (MarketEnv or MultiAssetEnv)
        n_iterations : number of collect→update cycles
        verbose      : print progress
        eval_env     : optional separate eval environment
        eval_freq    : evaluate every N iterations

        Returns list of per-iteration stats dicts.
        """
        history = []

        for iteration in range(n_iterations):
            self._schedule_lr(iteration, n_iterations)

            # Collect rollout
            rollout_stats = self.collect_rollout(env, self.config.n_steps)

            # Update networks
            p_loss, entropy, kl = self.update_policy(
                n_epochs=self.config.n_epochs,
                clip_eps=self.config.clip_eps,
                batch_size=self.config.batch_size,
            )
            v_loss = self.update_value(
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
            )
            self.n_updates += 1

            stats = {
                "iteration": iteration,
                "total_steps": self.total_steps,
                "mean_episode_return": rollout_stats["mean_episode_return"],
                "n_episodes": rollout_stats["n_episodes"],
                "policy_loss": p_loss,
                "value_loss": v_loss,
                "entropy": entropy,
                "kl": kl,
                "actor_lr": self.actor.optimizer.lr,
            }

            # Evaluation
            if eval_env is not None and (iteration + 1) % eval_freq == 0:
                eval_return = self._evaluate(eval_env, n_episodes=3)
                stats["eval_return"] = eval_return

            history.append(stats)
            self._train_history.append(stats)

            if verbose:
                msg = (
                    f"[{iteration+1:4d}/{n_iterations}] "
                    f"ret={stats['mean_episode_return']:+.4f} | "
                    f"p_loss={p_loss:.4f} | v_loss={v_loss:.4f} | "
                    f"ent={entropy:.4f} | kl={kl:.5f} | "
                    f"steps={self.total_steps}"
                )
                if "eval_return" in stats:
                    msg += f" | eval={stats['eval_return']:+.4f}"
                print(msg)

        return history

    def _evaluate(self, env: Any, n_episodes: int = 5) -> float:
        """Run greedy policy for n_episodes, return mean return."""
        returns = []
        for _ in range(n_episodes):
            state = env.reset()
            ep_return = 0.0
            done = False
            while not done:
                action, _ = self.actor.sample_action(
                    np.asarray(state, dtype=np.float32), deterministic=True
                )
                state, reward, done, _ = env.step(action)
                ep_return += reward
            returns.append(ep_return)
        return float(np.mean(returns))

    def act(self, state: np.ndarray, deterministic: bool = False) -> Any:
        """Select action for deployment (no gradient tracking)."""
        action, _ = self.actor.sample_action(
            np.asarray(state, dtype=np.float32), deterministic=deterministic
        )
        return action

    def save(self, path: str):
        """Save model weights to .npz file."""
        data = {}
        for i, (W, b) in enumerate(zip(self.actor.mlp.layers_W, self.actor.mlp.layers_b)):
            data[f"actor_W_{i}"] = W
            data[f"actor_b_{i}"] = b
        for i, (W, b) in enumerate(zip(self.critic.mlp.layers_W, self.critic.mlp.layers_b)):
            data[f"critic_W_{i}"] = W
            data[f"critic_b_{i}"] = b
        if not self.config.discrete:
            data["actor_log_std"] = self.actor.log_std
        np.savez(path, **data)

    def load(self, path: str):
        """Load model weights from .npz file."""
        d = np.load(path)
        for i in range(self.actor.mlp.n_layers):
            self.actor.mlp.layers_W[i] = d[f"actor_W_{i}"]
            self.actor.mlp.layers_b[i] = d[f"actor_b_{i}"]
        for i in range(self.critic.mlp.n_layers):
            self.critic.mlp.layers_W[i] = d[f"critic_W_{i}"]
            self.critic.mlp.layers_b[i] = d[f"critic_b_{i}"]
        if not self.config.discrete and "actor_log_std" in d:
            self.actor.log_std = d["actor_log_std"]

    def summary(self) -> Dict:
        if not self._train_history:
            return {}
        returns = [h["mean_episode_return"] for h in self._train_history]
        return {
            "total_iterations": len(self._train_history),
            "total_steps": self.total_steps,
            "best_return": float(max(returns)),
            "final_return": float(returns[-1]),
            "mean_return": float(np.mean(returns[-20:])),
            "final_entropy": float(self._train_history[-1].get("entropy", 0.0)),
            "final_kl": float(self._train_history[-1].get("kl", 0.0)),
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_ppo_agent(state_dim: int, n_actions: int,
                   discrete: bool = True,
                   hidden_dims: Optional[List[int]] = None) -> PPOAgent:
    """Convenience factory for default PPO configuration."""
    config = PPOConfig(
        state_dim=state_dim,
        action_dim=n_actions,
        hidden_dims=hidden_dims or [64, 64],
        discrete=discrete,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        target_kl=0.015,
        lr_schedule="linear",
    )
    return PPOAgent(config)
