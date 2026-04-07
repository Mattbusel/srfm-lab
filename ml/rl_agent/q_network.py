"""
ml/rl_agent/q_network.py -- Deep Q-Network implementation using numpy only.

Implements a 3-layer MLP with manual backpropagation, a circular replay buffer
with Prioritized Experience Replay (PER), and a full DQN agent with:
  - Epsilon-greedy exploration (1.0 -> 0.05 over 50000 steps)
  - Separate online and target networks
  - Soft (Polyak) target network updates

No external ML frameworks (PyTorch, TensorFlow, JAX) are used. All gradient
computations are performed manually via chain rule.

Architecture: 10 -> 64 -> 64 -> 3
Activation: ReLU on hidden layers, linear output
Loss: MSE between predicted Q-values and Bellman targets
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_DIM = 10
HIDDEN_DIM = 64
OUTPUT_DIM = 3   # HOLD, PARTIAL_EXIT, FULL_EXIT

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 50_000
GAMMA = 0.99
LEARNING_RATE = 1e-3
TARGET_UPDATE_FREQ = 100   # hard update every N steps
BATCH_SIZE = 64
BUFFER_CAPACITY = 100_000
PER_ALPHA = 0.6            # prioritization exponent
PER_BETA_START = 0.4       # IS correction annealing start
PER_BETA_END = 1.0
PER_BETA_STEPS = 100_000
PER_EPSILON = 1e-6         # small constant to avoid zero priority


# ---------------------------------------------------------------------------
# Xavier initialization
# ---------------------------------------------------------------------------

def _xavier_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier/Glorot uniform initialization."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)


def _zeros(shape: Tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# QNetwork -- 3-layer MLP with manual backprop
# ---------------------------------------------------------------------------

class QNetwork:
    """
    3-layer fully-connected neural network: 10 -> 64 -> 64 -> 3.

    All parameters are stored as numpy arrays. Forward pass computes
    Q-values for each action. Backward pass computes MSE loss gradients
    manually via chain rule.

    Parameters
    ----------
    learning_rate : float
        SGD/Adam step size.
    seed : int, optional
        Random seed for weight initialization.
    use_adam : bool
        If True, use Adam optimizer; else vanilla SGD.
    """

    def __init__(
        self,
        learning_rate: float = LEARNING_RATE,
        seed: Optional[int] = None,
        use_adam: bool = True,
    ) -> None:
        self.lr = learning_rate
        self.use_adam = use_adam
        rng = np.random.default_rng(seed)

        # Layer 1: 10 -> 64
        self.W1 = _xavier_uniform(INPUT_DIM, HIDDEN_DIM, rng)
        self.b1 = _zeros((HIDDEN_DIM,))

        # Layer 2: 64 -> 64
        self.W2 = _xavier_uniform(HIDDEN_DIM, HIDDEN_DIM, rng)
        self.b2 = _zeros((HIDDEN_DIM,))

        # Layer 3: 64 -> 3 (linear output)
        self.W3 = _xavier_uniform(HIDDEN_DIM, OUTPUT_DIM, rng)
        self.b3 = _zeros((OUTPUT_DIM,))

        # Adam optimizer state
        self._t = 0
        self._adam_beta1 = 0.9
        self._adam_beta2 = 0.999
        self._adam_eps = 1e-8
        self._m: Dict[str, np.ndarray] = {}
        self._v: Dict[str, np.ndarray] = {}
        for name, param in self._named_params():
            self._m[name] = np.zeros_like(param)
            self._v[name] = np.zeros_like(param)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Q-values for all 3 actions given a state.

        Parameters
        ----------
        state : np.ndarray
            Shape (10,) or (batch, 10).

        Returns
        -------
        np.ndarray
            Q-values, shape (3,) or (batch, 3).
        """
        single = state.ndim == 1
        if single:
            state = state[np.newaxis, :]  # (1, 10)
        state = state.astype(np.float32)

        z1 = state @ self.W1 + self.b1    # (batch, 64)
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2       # (batch, 64)
        a2 = _relu(z2)
        out = a2 @ self.W3 + self.b3      # (batch, 3)

        if single:
            return out[0]  # (3,)
        return out         # (batch, 3)

    def _forward_with_cache(
        self, states: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass that caches intermediate activations for backprop."""
        states = states.astype(np.float32)
        z1 = states @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = _relu(z2)
        out = a2 @ self.W3 + self.b3
        cache = {"states": states, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "out": out}
        return out, cache

    # ------------------------------------------------------------------
    # Backward pass + parameter update
    # ------------------------------------------------------------------

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute MSE loss and update network parameters via backprop.

        Parameters
        ----------
        states : np.ndarray, shape (batch, 10)
        actions : np.ndarray, shape (batch,) int
            Action indices (0, 1, or 2) taken in each transition.
        targets : np.ndarray, shape (batch,)
            TD target values (Bellman targets).
        weights : np.ndarray, shape (batch,), optional
            Importance-sampling weights for PER. If None, uniform weights.

        Returns
        -------
        loss : float
            Mean squared error loss.
        td_errors : np.ndarray, shape (batch,)
            Per-sample TD errors for PER priority update.
        """
        batch_size = states.shape[0]
        if weights is None:
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            weights = weights.astype(np.float32)

        out, cache = self._forward_with_cache(states)

        # Gather Q-values for chosen actions
        q_pred = out[np.arange(batch_size), actions]  # (batch,)
        td_errors = targets.astype(np.float32) - q_pred  # (batch,)

        # Weighted MSE loss
        loss = float(np.mean(weights * td_errors ** 2))

        # ------------------------------------------------------------------
        # Backprop -- chain rule through linear layers + ReLU
        # ------------------------------------------------------------------

        # dL/d(out) -- only at the action indices
        # dL/dq_pred = -2 * weights * td_errors / batch (from MSE gradient)
        d_out = np.zeros_like(out)  # (batch, 3)
        d_out[np.arange(batch_size), actions] = -2.0 * weights * td_errors / batch_size

        # Layer 3 gradients: out = a2 @ W3 + b3
        dW3 = cache["a2"].T @ d_out                # (64, 3)
        db3 = d_out.sum(axis=0)                    # (3,)
        d_a2 = d_out @ self.W3.T                   # (batch, 64)

        # ReLU backward for layer 2
        d_z2 = d_a2 * _relu_grad(cache["z2"])      # (batch, 64)

        # Layer 2 gradients: z2 = a1 @ W2 + b2
        dW2 = cache["a1"].T @ d_z2                 # (64, 64)
        db2 = d_z2.sum(axis=0)                     # (64,)
        d_a1 = d_z2 @ self.W2.T                    # (batch, 64)

        # ReLU backward for layer 1
        d_z1 = d_a1 * _relu_grad(cache["z1"])      # (batch, 64)

        # Layer 1 gradients: z1 = states @ W1 + b1
        dW1 = cache["states"].T @ d_z1             # (10, 64)
        db1 = d_z1.sum(axis=0)                     # (64,)

        grads = {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3,
        }

        self._apply_gradients(grads)
        return loss, td_errors

    def _apply_gradients(self, grads: Dict[str, np.ndarray]) -> None:
        """Apply computed gradients using Adam or SGD."""
        if self.use_adam:
            self._t += 1
            t = self._t
            for name, param in self._named_params():
                g = grads[name]
                self._m[name] = self._adam_beta1 * self._m[name] + (1 - self._adam_beta1) * g
                self._v[name] = self._adam_beta2 * self._v[name] + (1 - self._adam_beta2) * g ** 2
                m_hat = self._m[name] / (1 - self._adam_beta1 ** t)
                v_hat = self._v[name] / (1 - self._adam_beta2 ** t)
                update = self.lr * m_hat / (np.sqrt(v_hat) + self._adam_eps)
                # Apply in-place
                self._set_param(name, param - update)
        else:
            # Vanilla SGD
            for name, param in self._named_params():
                self._set_param(name, param - self.lr * grads[name])

    def _named_params(self):
        """Yield (name, param_array) for all trainable parameters."""
        yield "W1", self.W1
        yield "b1", self.b1
        yield "W2", self.W2
        yield "b2", self.b2
        yield "W3", self.W3
        yield "b3", self.b3

    def _set_param(self, name: str, value: np.ndarray) -> None:
        setattr(self, name, value.astype(np.float32))

    # ------------------------------------------------------------------
    # Weight copy (for target network initialization)
    # ------------------------------------------------------------------

    def copy_weights_from(self, other: "QNetwork") -> None:
        """Copy all weights from another QNetwork (for target network sync)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()

    def soft_update_from(self, online: "QNetwork", tau: float = 0.005) -> None:
        """
        Polyak averaging: target = tau * online + (1-tau) * target.

        Parameters
        ----------
        online : QNetwork
            The online (learning) network.
        tau : float
            Blend factor -- 0.005 is standard for soft target updates.
        """
        self.W1 = tau * online.W1 + (1 - tau) * self.W1
        self.b1 = tau * online.b1 + (1 - tau) * self.b1
        self.W2 = tau * online.W2 + (1 - tau) * self.W2
        self.b2 = tau * online.b2 + (1 - tau) * self.b2
        self.W3 = tau * online.W3 + (1 - tau) * self.W3
        self.b3 = tau * online.b3 + (1 - tau) * self.b3

    def save(self, path: str) -> None:
        """Save network weights to a .npz file."""
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
        )

    @classmethod
    def load(cls, path: str, **kwargs) -> "QNetwork":
        """Load network weights from a .npz file."""
        net = cls(**kwargs)
        data = np.load(path)
        net.W1 = data["W1"]
        net.b1 = data["b1"]
        net.W2 = data["W2"]
        net.b2 = data["b2"]
        net.W3 = data["W3"]
        net.b3 = data["b3"]
        return net


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 where x > 0, else 0."""
    return (x > 0).astype(np.float32)


# ---------------------------------------------------------------------------
# ReplayBuffer with Prioritized Experience Replay (PER)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Circular replay buffer with optional Prioritized Experience Replay.

    Stores transitions (state, action, reward, next_state, done) and
    supports both uniform and priority-weighted sampling.

    PER Reference: Schaul et al. 2015 -- "Prioritized Experience Replay".
    Priority p_i = |delta_i| + epsilon, where delta_i is the TD error.
    Sampling probability: P(i) = p_i^alpha / sum(p_j^alpha).

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    state_dim : int
        Dimensionality of state vectors.
    alpha : float
        PER prioritization exponent. 0 = uniform, 1 = full prioritization.
    beta_start : float
        Initial IS correction exponent.
    beta_end : float
        Final IS correction exponent (after annealing).
    beta_steps : int
        Number of steps to anneal beta from beta_start to beta_end.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        capacity: int = BUFFER_CAPACITY,
        state_dim: int = INPUT_DIM,
        alpha: float = PER_ALPHA,
        beta_start: float = PER_BETA_START,
        beta_end: float = PER_BETA_END,
        beta_steps: int = PER_BETA_STEPS,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.rng = np.random.default_rng(seed)

        # Pre-allocated arrays for efficiency
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)
        self._priorities = np.zeros(capacity, dtype=np.float64)

        self._pos = 0        # next write position
        self._size = 0       # current number of stored transitions
        self._step = 0       # total number of add() calls (for beta annealing)
        self._max_priority = 1.0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: Optional[float] = None,
    ) -> None:
        """
        Add a transition to the buffer.

        New transitions receive the maximum current priority so they are
        sampled at least once before their TD error is known.

        Parameters
        ----------
        td_error : float, optional
            If provided, use |td_error| + epsilon as priority.
            Otherwise use current max priority.
        """
        if td_error is not None:
            priority = (abs(td_error) + PER_EPSILON) ** self.alpha
        else:
            priority = self._max_priority

        idx = self._pos
        self._states[idx] = state
        self._actions[idx] = int(action)
        self._rewards[idx] = float(reward)
        self._next_states[idx] = next_state
        self._dones[idx] = bool(done)
        self._priorities[idx] = priority

        self._max_priority = max(self._max_priority, priority)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._step += 1

    def sample(
        self, batch_size: int = BATCH_SIZE
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions.

        Returns
        -------
        states : (batch, state_dim)
        actions : (batch,) int32
        rewards : (batch,) float32
        next_states : (batch, state_dim)
        dones : (batch,) bool
        weights : (batch,) float32 -- IS correction weights
        indices : (batch,) int32 -- buffer indices (for priority update)
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has only {self._size} transitions but batch_size={batch_size} requested."
            )

        # Compute sampling probabilities from priorities
        priorities = self._priorities[: self._size]
        probs = priorities / priorities.sum()

        indices = self.rng.choice(self._size, size=batch_size, replace=False, p=probs)

        # Importance-sampling weights
        beta = self._compute_beta()
        weights = (self._size * probs[indices]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)  # normalize to [0, 1]

        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
            weights,
            indices.astype(np.int32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities for sampled transitions based on new TD errors.

        Parameters
        ----------
        indices : np.ndarray, shape (batch,)
            Buffer indices returned by sample().
        td_errors : np.ndarray, shape (batch,)
            New TD error magnitudes.
        """
        for idx, err in zip(indices, td_errors):
            priority = (abs(float(err)) + PER_EPSILON) ** self.alpha
            self._priorities[int(idx)] = priority
            self._max_priority = max(self._max_priority, priority)

    def _compute_beta(self) -> float:
        """Linearly anneal beta from beta_start to beta_end."""
        frac = min(1.0, self._step / max(1, self.beta_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def __len__(self) -> int:
        return self._size

    def is_ready(self, batch_size: int = BATCH_SIZE) -> bool:
        return self._size >= batch_size


# ---------------------------------------------------------------------------
# DQNAgent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Network agent with:
      - Separate online and target networks
      - Epsilon-greedy exploration (annealing)
      - Prioritized Experience Replay
      - Soft (Polyak) target network updates

    Parameters
    ----------
    learning_rate : float
    gamma : float
        Discount factor for future rewards.
    epsilon_start : float
        Initial exploration probability.
    epsilon_end : float
        Minimum exploration probability.
    epsilon_decay_steps : int
        Number of steps to decay epsilon from start to end.
    batch_size : int
    buffer_capacity : int
    target_update_freq : int
        How often to hard-copy online weights to target network.
    tau : float
        Soft update blend coefficient.
    seed : int, optional
    """

    def __init__(
        self,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        epsilon_start: float = EPS_START,
        epsilon_end: float = EPS_END,
        epsilon_decay_steps: int = EPS_DECAY_STEPS,
        batch_size: int = BATCH_SIZE,
        buffer_capacity: int = BUFFER_CAPACITY,
        target_update_freq: int = TARGET_UPDATE_FREQ,
        tau: float = 0.005,
        seed: Optional[int] = None,
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self._step = 0
        self.rng = np.random.default_rng(seed)

        # Online network (trained every step)
        self.online = QNetwork(learning_rate=learning_rate, seed=seed)
        # Target network (updated slowly)
        self.target = QNetwork(learning_rate=learning_rate, seed=(seed or 0) + 1)
        self.target.copy_weights_from(self.online)

        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=INPUT_DIM,
            seed=seed,
        )

        # Training metrics
        self.losses: List[float] = []
        self.episode_rewards: List[float] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.

        Parameters
        ----------
        state : np.ndarray, shape (10,)
        explore : bool
            If False, always act greedily (no exploration). Use for evaluation.

        Returns
        -------
        int -- action index (0, 1, or 2)
        """
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, OUTPUT_DIM))
        q_values = self.online.predict(state)
        return int(np.argmax(q_values))

    def _decay_epsilon(self) -> None:
        """Linearly decay epsilon over epsilon_decay_steps."""
        frac = min(1.0, self._step / max(1, self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start - frac * (self.epsilon_start - self.epsilon_end)

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        """
        Sample a batch from the replay buffer and update the online network.

        Returns
        -------
        float or None
            MSE loss for this step, or None if buffer not ready.
        """
        if not self.buffer.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(
            self.batch_size
        )

        # Compute Bellman targets using target network (Double DQN variant)
        # Online network selects action, target network evaluates it
        online_next_q = self.online.predict(next_states)         # (batch, 3)
        best_next_actions = np.argmax(online_next_q, axis=1)     # (batch,)
        target_next_q = self.target.predict(next_states)          # (batch, 3)
        next_q_values = target_next_q[np.arange(self.batch_size), best_next_actions]

        # Bellman equation
        not_done = (~dones).astype(np.float32)
        td_targets = rewards + self.gamma * next_q_values * not_done  # (batch,)

        # Update online network
        loss, td_errors = self.online.update(states, actions, td_targets, weights=weights)

        # Update PER priorities
        self.buffer.update_priorities(indices, td_errors)

        self.losses.append(loss)
        self._step += 1
        self._decay_epsilon()

        # Soft update target network every step
        self.target.soft_update_from(self.online, tau=self.tau)

        # Hard update every target_update_freq steps
        if self._step % self.target_update_freq == 0:
            self.target.copy_weights_from(self.online)

        return loss

    def soft_update_target(self, tau: float = 0.005) -> None:
        """Explicitly trigger a Polyak soft update of the target network."""
        self.target.soft_update_from(self.online, tau=tau)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, prefix: str) -> None:
        """Save online and target network weights."""
        self.online.save(f"{prefix}_online.npz")
        self.target.save(f"{prefix}_target.npz")

    def load(self, prefix: str) -> None:
        """Load online and target network weights."""
        self.online = QNetwork.load(f"{prefix}_online.npz", learning_rate=self.online.lr)
        self.target = QNetwork.load(f"{prefix}_target.npz", learning_rate=self.target.lr)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a dict of current training diagnostics."""
        recent_losses = self.losses[-100:] if self.losses else []
        return {
            "step": self._step,
            "epsilon": round(self.epsilon, 4),
            "buffer_size": len(self.buffer),
            "mean_loss_100": round(float(np.mean(recent_losses)), 6) if recent_losses else None,
            "n_episodes": len(self.episode_rewards),
            "mean_episode_reward": (
                round(float(np.mean(self.episode_rewards[-50:])), 4)
                if self.episode_rewards else None
            ),
        }
