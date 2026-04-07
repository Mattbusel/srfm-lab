"""
ml/rl_agent/ppo_agent.py -- Proximal Policy Optimization agent using numpy only.

Implements the PPO-Clip algorithm (Schulman et al. 2017) with:
  - Actor-critic shared backbone network
  - Generalized Advantage Estimation (GAE-lambda)
  - PPO clip objective with configurable clip ratio
  - Value loss with coefficient
  - Entropy bonus for exploration

No external ML frameworks. All network operations and gradient computations
are performed manually with numpy.

Architecture:
  Shared backbone: 10 -> 64 -> 64
  Actor head:      64 -> 3   (logits -> softmax -> categorical policy)
  Critic head:     64 -> 1   (value estimate V(s))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ml.rl_agent.environment import TradingEnvironment, N_FEATURES, N_ACTIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKBONE_H1 = 64
BACKBONE_H2 = 64
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
CLIP_RATIO = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
PPO_EPOCHS = 4           # number of update epochs per batch of episodes
GAMMA = 0.99
LAM = 0.95               # GAE lambda
MAX_GRAD_NORM = 0.5      # gradient clipping threshold


# ---------------------------------------------------------------------------
# Xavier initialization
# ---------------------------------------------------------------------------

def _xavier(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax."""
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _clip_grad(grad: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip gradient by global norm."""
    norm = float(np.linalg.norm(grad))
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad


# ---------------------------------------------------------------------------
# Episode data container
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """
    Stores the trajectory collected by running one episode.

    Attributes
    ----------
    states : np.ndarray, shape (T, 10)
    actions : np.ndarray, shape (T,) int
    rewards : np.ndarray, shape (T,) float
    values : np.ndarray, shape (T,) float -- V(s) estimates from critic
    log_probs : np.ndarray, shape (T,) float -- log pi(a|s) at collection time
    dones : np.ndarray, shape (T,) bool
    advantages : np.ndarray, shape (T,) float -- filled by compute_gae()
    returns : np.ndarray, shape (T,) float -- discounted returns / GAE targets
    """

    states: np.ndarray = field(default_factory=lambda: np.zeros((0, N_FEATURES), dtype=np.float32))
    actions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    rewards: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    log_probs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    dones: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.bool_))
    advantages: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    returns: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PolicyNetwork -- shared backbone + actor + critic heads
# ---------------------------------------------------------------------------

class PolicyNetwork:
    """
    Actor-critic network with shared feature backbone.

    Backbone: 10 -> 64 -> 64 (ReLU activations)
    Actor head: 64 -> 3 (softmax policy -- outputs action probabilities)
    Critic head: 64 -> 1 (linear value estimate)

    Parameters
    ----------
    learning_rate : float
        Adam learning rate for all parameters.
    seed : int, optional
    """

    def __init__(
        self,
        learning_rate: float = ACTOR_LR,
        seed: Optional[int] = None,
    ) -> None:
        self.lr = learning_rate
        rng = np.random.default_rng(seed)

        # Shared backbone: 10 -> 64 -> 64
        self.W1 = _xavier(N_FEATURES, BACKBONE_H1, rng)
        self.b1 = np.zeros(BACKBONE_H1, dtype=np.float32)
        self.W2 = _xavier(BACKBONE_H1, BACKBONE_H2, rng)
        self.b2 = np.zeros(BACKBONE_H2, dtype=np.float32)

        # Actor head: 64 -> 3 (logits)
        self.W_actor = _xavier(BACKBONE_H2, N_ACTIONS, rng)
        self.b_actor = np.zeros(N_ACTIONS, dtype=np.float32)

        # Critic head: 64 -> 1
        self.W_critic = _xavier(BACKBONE_H2, 1, rng)
        self.b_critic = np.zeros(1, dtype=np.float32)

        # Adam state
        self._t = 0
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._adam_eps = 1e-8
        self._m: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self._params()}
        self._v_adam: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self._params()}

    def _params(self):
        """Yield (name, array) for all parameters."""
        yield "W1", self.W1
        yield "b1", self.b1
        yield "W2", self.W2
        yield "b2", self.b2
        yield "W_actor", self.W_actor
        yield "b_actor", self.b_actor
        yield "W_critic", self.W_critic
        yield "b_critic", self.b_critic

    def _set_param(self, name: str, val: np.ndarray) -> None:
        setattr(self, name, val.astype(np.float32))

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_backbone(self, states: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Run the shared backbone.

        Returns
        -------
        h2 : np.ndarray, shape (batch, 64) -- backbone output
        cache : dict -- intermediate values for backprop
        """
        states = states.astype(np.float32)
        z1 = states @ self.W1 + self.b1   # (batch, 64)
        h1 = _relu(z1)
        z2 = h1 @ self.W2 + self.b2       # (batch, 64)
        h2 = _relu(z2)
        cache = {"states": states, "z1": z1, "h1": h1, "z2": z2, "h2": h2}
        return h2, cache

    def predict_action_probs(self, states: np.ndarray) -> np.ndarray:
        """
        Compute action probability distribution.

        Returns
        -------
        np.ndarray, shape (batch, 3) or (3,) for single state
        """
        single = states.ndim == 1
        if single:
            states = states[np.newaxis, :]
        h2, _ = self.forward_backbone(states)
        logits = h2 @ self.W_actor + self.b_actor   # (batch, 3)
        probs = _softmax(logits)
        return probs[0] if single else probs

    def predict_value(self, states: np.ndarray) -> np.ndarray:
        """
        Compute value estimates V(s).

        Returns
        -------
        np.ndarray, shape (batch,) or scalar for single state
        """
        single = states.ndim == 1
        if single:
            states = states[np.newaxis, :]
        h2, _ = self.forward_backbone(states)
        values = (h2 @ self.W_critic + self.b_critic).squeeze(-1)  # (batch,)
        return float(values[0]) if single else values

    def predict_both(
        self, states: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both action probs and value in one pass.

        Returns
        -------
        probs : (batch, 3) or (3,)
        values : (batch,) or scalar
        """
        single = states.ndim == 1
        if single:
            states = states[np.newaxis, :]
        h2, _ = self.forward_backbone(states)
        logits = h2 @ self.W_actor + self.b_actor
        probs = _softmax(logits)
        values = (h2 @ self.W_critic + self.b_critic).squeeze(-1)
        if single:
            return probs[0], float(values[0])
        return probs, values

    # ------------------------------------------------------------------
    # PPO update step (manual backprop)
    # ------------------------------------------------------------------

    def ppo_update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        old_log_probs: np.ndarray,
    ) -> Dict[str, float]:
        """
        One PPO gradient update step.

        Parameters
        ----------
        states : (T, 10)
        actions : (T,) int
        advantages : (T,) -- normalized GAE advantages
        returns : (T,) -- discounted returns for value target
        old_log_probs : (T,) -- log pi_old(a|s) from collection time

        Returns
        -------
        dict with "policy_loss", "value_loss", "entropy", "total_loss"
        """
        T = len(states)
        states = states.astype(np.float32)
        advantages = advantages.astype(np.float32)
        returns = returns.astype(np.float32)
        old_log_probs = old_log_probs.astype(np.float32)

        # Forward pass
        h2, cache = self.forward_backbone(states)
        logits = h2 @ self.W_actor + self.b_actor     # (T, 3)
        log_probs_all = _log_softmax(logits)           # (T, 3)
        probs = np.exp(log_probs_all)                  # (T, 3)
        values = (h2 @ self.W_critic + self.b_critic).squeeze(-1)  # (T,)

        # Per-step log probs for chosen actions
        log_probs = log_probs_all[np.arange(T), actions]  # (T,)

        # PPO clip ratio
        ratio = np.exp(log_probs - old_log_probs)      # (T,)
        clip_ratio = np.clip(ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO)

        # PPO policy loss: -min(r*A, clip(r)*A)
        policy_obj = np.minimum(ratio * advantages, clip_ratio * advantages)
        policy_loss = -float(np.mean(policy_obj))

        # Value loss: MSE between predicted value and return
        value_loss = float(np.mean((values - returns) ** 2))

        # Entropy bonus: H[pi] = -sum(pi * log(pi))
        entropy = float(-np.mean(np.sum(probs * log_probs_all, axis=-1)))

        # Total loss
        total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        # ------------------------------------------------------------------
        # Backward pass -- compute all gradients
        # ------------------------------------------------------------------

        # dL/d(policy_obj) = -1/T
        d_policy_obj = -np.ones(T, dtype=np.float32) / T

        # Gradient through min(r*A, clip(r)*A)
        # dL/d(ratio) = d_policy_obj * A if r*A <= clip*A (unclipped region)
        unclipped = (ratio * advantages <= clip_ratio * advantages)
        d_ratio = d_policy_obj * advantages * unclipped.astype(np.float32)  # (T,)

        # ratio = exp(log_prob - old_log_prob) => d_ratio/d(log_prob) = ratio
        d_log_probs_action = d_ratio * ratio  # (T,)

        # Gradient from value loss: dL/d(values) = 2 * (values - returns) / T * VALUE_COEF
        d_values = 2.0 * (values - returns) / T * VALUE_COEF  # (T,)

        # Gradient from entropy: dH/d(log_probs_all) = -(probs + 1) approximately
        # More precisely: d(-entropy)/d(logits_k) = pi_k * (log_pi_k - H)
        H_per_step = -np.sum(probs * log_probs_all, axis=-1)  # (T,)
        d_entropy_logits = probs * (log_probs_all + H_per_step[:, np.newaxis])
        d_logits_entropy = -ENTROPY_COEF * d_entropy_logits  # (T, 3)

        # Gradient of log_probs[t, a] w.r.t. logits[t, :] via log-softmax
        # d(log_softmax(x)_a)/d(x_k) = indicator(k==a) - softmax(x)_k
        d_logits_policy = np.zeros((T, N_ACTIONS), dtype=np.float32)
        for t in range(T):
            a = actions[t]
            d_logits_policy[t] = -d_log_probs_action[t] * probs[t]
            d_logits_policy[t, a] += d_log_probs_action[t]

        # Total gradient w.r.t. actor logits
        d_logits = d_logits_policy + d_logits_entropy  # (T, 3)

        # Actor head gradients
        dW_actor = cache["h2"].T @ d_logits  # (64, 3)
        db_actor = d_logits.sum(axis=0)       # (3,)
        d_h2_actor = d_logits @ self.W_actor.T  # (T, 64)

        # Critic head gradients
        d_values_col = d_values[:, np.newaxis]  # (T, 1)
        dW_critic = cache["h2"].T @ d_values_col  # (64, 1)
        db_critic = d_values_col.sum(axis=0)       # (1,)
        d_h2_critic = d_values_col @ self.W_critic.T  # (T, 64)

        # Backbone gradients
        d_h2 = d_h2_actor + d_h2_critic  # (T, 64)

        # Backbone layer 2 backprop
        d_z2 = d_h2 * _relu_grad(cache["z2"])  # (T, 64)
        dW2 = cache["h1"].T @ d_z2             # (64, 64)
        db2 = d_z2.sum(axis=0)                  # (64,)
        d_h1 = d_z2 @ self.W2.T                 # (T, 64)

        # Backbone layer 1 backprop
        d_z1 = d_h1 * _relu_grad(cache["z1"])   # (T, 64)
        dW1 = cache["states"].T @ d_z1           # (10, 64)
        db1 = d_z1.sum(axis=0)                   # (64,)

        grads = {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W_actor": dW_actor,
            "b_actor": db_actor,
            "W_critic": dW_critic,
            "b_critic": db_critic,
        }

        # Clip gradients
        for k in grads:
            grads[k] = _clip_grad(grads[k], MAX_GRAD_NORM)

        self._apply_adam(grads)

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
        }

    def _apply_adam(self, grads: Dict[str, np.ndarray]) -> None:
        self._t += 1
        t = self._t
        for name, param in self._params():
            g = grads[name]
            self._m[name] = self._beta1 * self._m[name] + (1 - self._beta1) * g
            self._v_adam[name] = self._beta2 * self._v_adam[name] + (1 - self._beta2) * g ** 2
            m_hat = self._m[name] / (1 - self._beta1 ** t)
            v_hat = self._v_adam[name] / (1 - self._beta2 ** t)
            new_val = param - self.lr * m_hat / (np.sqrt(v_hat) + self._adam_eps)
            self._set_param(name, new_val)

    def save(self, path: str) -> None:
        np.savez(path, **{k: v for k, v in self._params()})

    @classmethod
    def load(cls, path: str, **kwargs) -> "PolicyNetwork":
        net = cls(**kwargs)
        data = np.load(path)
        for name, _ in net._params():
            setattr(net, name, data[name].astype(np.float32))
        return net


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    PPO agent that collects episodes and performs batched policy updates.

    Parameters
    ----------
    learning_rate : float
    gamma : float
        Discount factor.
    lam : float
        GAE lambda.
    ppo_epochs : int
        Number of gradient update passes per episode batch.
    seed : int, optional
    """

    def __init__(
        self,
        learning_rate: float = ACTOR_LR,
        gamma: float = GAMMA,
        lam: float = LAM,
        ppo_epochs: int = PPO_EPOCHS,
        seed: Optional[int] = None,
    ) -> None:
        self.gamma = gamma
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        self.rng = np.random.default_rng(seed)
        self.policy = PolicyNetwork(learning_rate=learning_rate, seed=seed)

        # Metrics
        self.update_infos: List[Dict[str, float]] = []
        self.episode_returns: List[float] = []

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def collect_episode(self, env: TradingEnvironment) -> Episode:
        """
        Run one full episode using the current policy.

        Parameters
        ----------
        env : TradingEnvironment

        Returns
        -------
        Episode
            Collected trajectory with states, actions, rewards, values, log_probs.
        """
        states_list: List[np.ndarray] = []
        actions_list: List[int] = []
        rewards_list: List[float] = []
        values_list: List[float] = []
        log_probs_list: List[float] = []
        dones_list: List[bool] = []

        obs = env.reset()
        done = False

        while not done:
            probs, value = self.policy.predict_both(obs)
            # Sample action from distribution
            action = int(self.rng.choice(N_ACTIONS, p=probs))
            log_prob = float(np.log(probs[action] + 1e-8))

            states_list.append(obs.copy())
            actions_list.append(action)
            values_list.append(float(value))
            log_probs_list.append(log_prob)

            obs, reward, done, _info = env.step(action)
            rewards_list.append(float(reward))
            dones_list.append(bool(done))

        T = len(rewards_list)
        ep = Episode(
            states=np.array(states_list, dtype=np.float32),
            actions=np.array(actions_list, dtype=np.int32),
            rewards=np.array(rewards_list, dtype=np.float32),
            values=np.array(values_list, dtype=np.float32),
            log_probs=np.array(log_probs_list, dtype=np.float32),
            dones=np.array(dones_list, dtype=np.bool_),
        )
        self.episode_returns.append(float(np.sum(ep.rewards)))
        return ep

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        gamma: Optional[float] = None,
        lam: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE-lambda).

        GAE(t) = sum_{k=0}^{T-t-1} (gamma*lambda)^k * delta_{t+k}
        where delta_t = r_t + gamma * V(s_{t+1}) * (1-done_t) - V(s_t)

        Parameters
        ----------
        rewards : np.ndarray, shape (T,)
        values : np.ndarray, shape (T,)
            V(s_t) estimates.
        dones : np.ndarray, shape (T,) bool
        gamma : float, optional -- defaults to self.gamma
        lam : float, optional -- defaults to self.lam

        Returns
        -------
        advantages : np.ndarray, shape (T,)
        returns : np.ndarray, shape (T,)
            advantages + values (value target for critic loss)
        """
        if gamma is None:
            gamma = self.gamma
        if lam is None:
            lam = self.lam

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        # Bootstrap value for the terminal state is 0 (episode ended)
        next_value = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            next_val = next_value * mask
            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, episodes: List[Episode]) -> Dict[str, float]:
        """
        Perform PPO policy update from a list of collected episodes.

        Computes GAE for each episode, concatenates all transitions,
        normalizes advantages, and runs ppo_epochs gradient update passes.

        Parameters
        ----------
        episodes : list of Episode

        Returns
        -------
        dict with aggregated training metrics:
          "policy_loss", "value_loss", "entropy", "total_loss", "n_transitions"
        """
        # Compute GAE for each episode and attach to episode object
        all_states: List[np.ndarray] = []
        all_actions: List[np.ndarray] = []
        all_advantages: List[np.ndarray] = []
        all_returns: List[np.ndarray] = []
        all_log_probs: List[np.ndarray] = []

        for ep in episodes:
            adv, ret = self.compute_gae(ep.rewards, ep.values, ep.dones)
            ep.advantages = adv
            ep.returns = ret
            all_states.append(ep.states)
            all_actions.append(ep.actions)
            all_advantages.append(adv)
            all_returns.append(ret)
            all_log_probs.append(ep.log_probs)

        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions)
        advantages = np.concatenate(all_advantages)
        returns = np.concatenate(all_returns)
        old_log_probs = np.concatenate(all_log_probs)

        # Normalize advantages
        adv_mean = float(np.mean(advantages))
        adv_std = float(np.std(advantages)) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Multiple PPO epochs over the same data
        epoch_infos: List[Dict[str, float]] = []
        n = len(states)
        indices = np.arange(n)

        for _ in range(self.ppo_epochs):
            self.rng.shuffle(indices)
            info = self.policy.ppo_update(
                states[indices],
                actions[indices],
                advantages[indices],
                returns[indices],
                old_log_probs[indices],
            )
            epoch_infos.append(info)

        # Average metrics over epochs
        aggregated: Dict[str, float] = {}
        for key in epoch_infos[0]:
            aggregated[key] = float(np.mean([e[key] for e in epoch_infos]))
        aggregated["n_transitions"] = float(n)

        self.update_infos.append(aggregated)
        return aggregated

    # ------------------------------------------------------------------
    # Action selection (evaluation mode)
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select an action.

        Parameters
        ----------
        state : np.ndarray, shape (10,)
        deterministic : bool
            If True, select argmax action (no sampling).

        Returns
        -------
        int -- action index
        """
        probs = self.policy.predict_action_probs(state)
        if deterministic:
            return int(np.argmax(probs))
        return int(self.rng.choice(N_ACTIONS, p=probs))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        recent = self.episode_returns[-50:] if self.episode_returns else []
        recent_info = self.update_infos[-20:] if self.update_infos else []
        result: Dict[str, Any] = {
            "n_episodes": len(self.episode_returns),
            "mean_return_50": round(float(np.mean(recent)), 4) if recent else None,
        }
        if recent_info:
            for k in ["policy_loss", "value_loss", "entropy", "total_loss"]:
                result[f"mean_{k}_20"] = round(float(np.mean([i[k] for i in recent_info])), 6)
        return result

    def save(self, path: str) -> None:
        self.policy.save(path)

    def load(self, path: str) -> None:
        self.policy = PolicyNetwork.load(path, learning_rate=self.policy.lr)
