"""
base_agent.py — Base agent for Hyper-Agent MARL.

Provides:
- ObservationEncoder: MLP mapping obs -> hidden state
- Actor: policy network (Gaussian or discrete)
- Critic: value network
- BaseAgent: abstract class with experience replay, update step
- GRU-based recurrent variants for partial observability
- Exploration strategies: epsilon-greedy, OU noise, parameter noise
"""

from __future__ import annotations

import abc
import math
import copy
import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MultivariateNormal

logger = logging.getLogger(__name__)

EPS = 1e-8
LOG_STD_MIN = -5
LOG_STD_MAX = 2


# ---------------------------------------------------------------------------
# Utility modules
# ---------------------------------------------------------------------------

def layer_init(layer: nn.Module, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal initialization."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: target = tau*source + (1-tau)*target."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration (NoisyNet).
    Adds learned noise to weights and biases.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        eps_in = self._f(torch.randn(self.in_features))
        eps_out = self._f(torch.randn(self.out_features))
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RunningMeanStd:
    """Welford online algorithm for running mean and variance."""

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + EPS)


# ---------------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------------

class ObservationEncoder(nn.Module):
    """
    MLP encoder mapping raw observations to a hidden state vector.
    Supports layer normalization and residual connections.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: str = "relu",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        act_fn = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }.get(activation, nn.ReLU)

        layers: List[nn.Module] = []
        in_dim = obs_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (..., obs_dim)
        Returns:
            hidden: (..., hidden_dim)
        """
        return self.net(obs)


class RecurrentObservationEncoder(nn.Module):
    """
    GRU-based recurrent encoder for partial observability.
    Maintains hidden state across time steps.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        gru_layers: int = 1,
        pre_gru_layers: int = 1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers

        # Pre-GRU MLP
        pre_layers: List[nn.Module] = []
        in_d = obs_dim
        for _ in range(pre_gru_layers):
            pre_layers.append(nn.Linear(in_d, hidden_dim))
            if use_layer_norm:
                pre_layers.append(nn.LayerNorm(hidden_dim))
            pre_layers.append(nn.ReLU())
            in_d = hidden_dim
        self.pre_net = nn.Sequential(*pre_layers)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )

        if use_layer_norm:
            self.post_norm = nn.LayerNorm(hidden_dim)
        else:
            self.post_norm = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, seq_len, obs_dim) or (batch, obs_dim)
            hidden: (num_layers, batch, hidden_dim) or None
        Returns:
            output: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            hidden: (num_layers, batch, hidden_dim)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (B, 1, obs_dim)
            squeeze = True
        else:
            squeeze = False

        x = self.pre_net(obs)  # (B, T, hidden_dim)
        output, new_hidden = self.gru(x, hidden)
        output = self.post_norm(output)

        if squeeze:
            output = output.squeeze(1)
        return output, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch_size, self.hidden_dim, device=device)


# ---------------------------------------------------------------------------
# Actor networks
# ---------------------------------------------------------------------------

class GaussianActor(nn.Module):
    """
    Continuous Gaussian policy.
    Outputs mean and log_std of action distribution.
    Supports squashed (SAC-style) or raw (PPO-style) actions.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        squash_output: bool = True,
        use_state_dependent_std: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.squash_output = squash_output

        self.mean_layer = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

        if use_state_dependent_std:
            self.log_std_layer = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
            self.log_std_is_param = False
        else:
            self.log_std_param = nn.Parameter(torch.zeros(action_dim))
            self.log_std_layer = None
            self.log_std_is_param = True

    def forward(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log_std)."""
        mean = self.mean_layer(hidden)
        if self.log_std_is_param:
            log_std = self.log_std_param.expand_as(mean)
        else:
            log_std = self.log_std_layer(hidden)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_distribution(self, hidden: torch.Tensor) -> Normal:
        mean, log_std = self.forward(hidden)
        return Normal(mean, log_std.exp())

    def get_action(
        self, hidden: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.
        Returns (action, log_prob).
        """
        mean, log_std = self.forward(hidden)
        std = log_std.exp()
        dist = Normal(mean, std)

        if deterministic:
            x = mean
        else:
            x = dist.rsample()

        log_prob = dist.log_prob(x).sum(dim=-1, keepdim=True)

        if self.squash_output:
            action = torch.tanh(x)
            # Correction for tanh squashing
            log_prob -= torch.sum(
                torch.log(torch.clamp(1.0 - action ** 2, min=EPS)), dim=-1, keepdim=True
            )
        else:
            action = x

        return action, log_prob

    def evaluate_actions(
        self, hidden: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy of given actions.
        Returns (log_prob, entropy).
        """
        mean, log_std = self.forward(hidden)
        std = log_std.exp()
        dist = Normal(mean, std)

        if self.squash_output:
            # Undo squashing for log prob calculation
            raw_actions = torch.atanh(torch.clamp(actions, -1 + EPS, 1 - EPS))
            log_prob = dist.log_prob(raw_actions).sum(dim=-1, keepdim=True)
            log_prob -= torch.sum(
                torch.log(torch.clamp(1.0 - actions ** 2, min=EPS)),
                dim=-1, keepdim=True
            )
        else:
            log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class DiscreteActor(nn.Module):
    """
    Discrete categorical policy.
    """

    def __init__(self, hidden_dim: int, num_actions: int):
        super().__init__()
        self.logits_layer = layer_init(nn.Linear(hidden_dim, num_actions), std=0.01)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.logits_layer(hidden)

    def get_distribution(self, hidden: torch.Tensor) -> Categorical:
        logits = self.forward(hidden)
        return Categorical(logits=logits)

    def get_action(
        self, hidden: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(hidden)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob

    def evaluate_actions(
        self, hidden: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(hidden)
        log_prob = dist.log_prob(actions).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return log_prob, entropy


# ---------------------------------------------------------------------------
# Critic networks
# ---------------------------------------------------------------------------

class ValueCritic(nn.Module):
    """
    State-value function V(s).
    """

    def __init__(self, hidden_dim: int, num_hidden: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(num_hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(layer_init(nn.Linear(hidden_dim, 1), std=1.0))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class QValueCritic(nn.Module):
    """
    Action-value function Q(s, a).
    """

    def __init__(self, hidden_dim: int, action_dim: int, num_hidden: int = 2):
        super().__init__()
        self.input_layer = nn.Linear(hidden_dim + action_dim, hidden_dim)
        layers: List[nn.Module] = [nn.ReLU()]
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(layer_init(nn.Linear(hidden_dim, 1), std=1.0))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden, action], dim=-1)
        x = F.relu(self.input_layer(x))
        return self.net(x)


class DoubleQValueCritic(nn.Module):
    """Double Q-value critic (two independent networks to reduce overestimation)."""

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.q1 = QValueCritic(hidden_dim, action_dim)
        self.q2 = QValueCritic(hidden_dim, action_dim)

    def forward(
        self, hidden: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(hidden, action), self.q2(hidden, action)

    def min(self, hidden: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(hidden, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# Exploration strategies
# ---------------------------------------------------------------------------

class OUNoise:
    """
    Ornstein-Uhlenbeck noise for continuous action exploration.
    """

    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
        seed: Optional[int] = None,
    ):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(action_dim)

    def reset(self) -> None:
        self.state = np.zeros(self.action_dim)

    def sample(self) -> np.ndarray:
        dx = (
            self.theta * (self.mu - self.state) * self.dt
            + self.sigma * math.sqrt(self.dt) * self.rng.standard_normal(self.action_dim)
        )
        self.state += dx
        return self.state.copy()


class ParameterNoise:
    """
    Adaptive parameter space noise for exploration.
    Perturbs network weights directly.
    """

    def __init__(self, initial_std: float = 0.1, desired_distance: float = 0.2):
        self.std = initial_std
        self.desired_distance = desired_distance

    def adapt(self, distance: float) -> None:
        if distance > self.desired_distance:
            self.std *= 0.9
        else:
            self.std *= 1.1

    def perturb_network(self, network: nn.Module) -> nn.Module:
        perturbed = copy.deepcopy(network)
        for param in perturbed.parameters():
            noise = torch.randn_like(param) * self.std
            param.data.add_(noise)
        return perturbed


# ---------------------------------------------------------------------------
# Experience buffer
# ---------------------------------------------------------------------------

class Transition:
    """Single environment transition."""
    __slots__ = [
        "obs", "action", "reward", "next_obs", "done",
        "log_prob", "value", "hidden", "global_state",
    ]

    def __init__(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        log_prob: float = 0.0,
        value: float = 0.0,
        hidden: Optional[np.ndarray] = None,
        global_state: Optional[np.ndarray] = None,
    ):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.done = done
        self.log_prob = log_prob
        self.value = value
        self.hidden = hidden
        self.global_state = global_state


class EpisodeBuffer:
    """Buffer for a single episode trajectory."""

    def __init__(self):
        self.transitions: List[Transition] = []

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)

    def compute_returns(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and discounted returns.
        Returns (advantages, returns).
        """
        n = len(self.transitions)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        next_value = last_value
        for i in reversed(range(n)):
            t = self.transitions[i]
            mask = 0.0 if t.done else 1.0
            delta = t.reward + gamma * next_value * mask - t.value
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[i] = gae
            next_value = t.value

        returns = advantages + np.array([t.value for t in self.transitions], dtype=np.float32)
        return advantages, returns

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "obs": np.array([t.obs for t in self.transitions], dtype=np.float32),
            "actions": np.array([t.action for t in self.transitions], dtype=np.float32),
            "rewards": np.array([t.reward for t in self.transitions], dtype=np.float32),
            "next_obs": np.array([t.next_obs for t in self.transitions], dtype=np.float32),
            "dones": np.array([t.done for t in self.transitions], dtype=np.float32),
            "log_probs": np.array([t.log_prob for t in self.transitions], dtype=np.float32),
            "values": np.array([t.value for t in self.transitions], dtype=np.float32),
        }

    def clear(self) -> None:
        self.transitions.clear()


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """
    Abstract base class for all Hyper-Agent MARL agents.

    Subclasses must implement:
    - select_action(obs, deterministic) -> (action, log_prob, value)
    - update(batch) -> Dict[str, float]
    - save(path) / load(path)
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Build networks (subclasses call this)
        self.encoder: Optional[nn.Module] = None
        self.actor: Optional[nn.Module] = None
        self.critic: Optional[nn.Module] = None

        self.actor_optimizer: Optional[torch.optim.Optimizer] = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None

        # Episode buffer
        self.buffer = EpisodeBuffer()

        # Running stats
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))

        # Tracking
        self._update_count = 0
        self._step_count = 0
        self._train_mode = True

        self._metrics: Dict[str, List[float]] = collections.defaultdict(list)

    # ---- Abstract interface ----------------------------------------------

    @abc.abstractmethod
    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Returns (action, log_prob, value_estimate)."""
        ...

    @abc.abstractmethod
    def update(self, batch: Optional[Dict] = None) -> Dict[str, float]:
        """Perform one gradient update step. Returns metric dict."""
        ...

    # ---- Common utilities ------------------------------------------------

    def observe(self, transition: Transition) -> None:
        """Store a transition in the episode buffer."""
        self.buffer.add(transition)
        self._step_count += 1
        self.obs_rms.update(transition.obs[None])

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return self.obs_rms.normalize(obs).astype(np.float32)

    def to_tensor(self, x: np.ndarray, dtype=None) -> torch.Tensor:
        if dtype is None:
            dtype = torch.float32
        return torch.tensor(x, dtype=dtype, device=self.device)

    def train(self) -> None:
        self._train_mode = True
        if self.encoder:
            self.encoder.train()
        if self.actor:
            self.actor.train()
        if self.critic:
            self.critic.train()

    def eval(self) -> None:
        self._train_mode = False
        if self.encoder:
            self.encoder.eval()
        if self.actor:
            self.actor.eval()
        if self.critic:
            self.critic.eval()

    def save(self, path: str) -> None:
        """Save all network weights and optimizer states."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "agent_id": self.agent_id,
            "update_count": self._update_count,
            "step_count": self._step_count,
            "obs_rms_mean": self.obs_rms.mean,
            "obs_rms_var": self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
        }
        if self.encoder:
            state["encoder"] = self.encoder.state_dict()
        if self.actor:
            state["actor"] = self.actor.state_dict()
        if self.critic:
            state["critic"] = self.critic.state_dict()
        if self.actor_optimizer:
            state["actor_optimizer"] = self.actor_optimizer.state_dict()
        if self.critic_optimizer:
            state["critic_optimizer"] = self.critic_optimizer.state_dict()
        torch.save(state, path)
        logger.info(f"Agent {self.agent_id} saved to {path}")

    def load(self, path: str) -> None:
        """Load network weights and optimizer states."""
        state = torch.load(path, map_location=self.device)
        if "encoder" in state and self.encoder:
            self.encoder.load_state_dict(state["encoder"])
        if "actor" in state and self.actor:
            self.actor.load_state_dict(state["actor"])
        if "critic" in state and self.critic:
            self.critic.load_state_dict(state["critic"])
        if "actor_optimizer" in state and self.actor_optimizer:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state and self.critic_optimizer:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self._update_count = state.get("update_count", 0)
        self._step_count = state.get("step_count", 0)
        self.obs_rms.mean = state.get("obs_rms_mean", self.obs_rms.mean)
        self.obs_rms.var = state.get("obs_rms_var", self.obs_rms.var)
        self.obs_rms.count = state.get("obs_rms_count", self.obs_rms.count)
        logger.info(f"Agent {self.agent_id} loaded from {path}")

    def log_metric(self, key: str, value: float) -> None:
        self._metrics[key].append(value)

    def get_metrics(self, last_n: int = 100) -> Dict[str, float]:
        return {
            k: float(np.mean(v[-last_n:])) if v else 0.0
            for k, v in self._metrics.items()
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id}, obs={self.obs_dim}, act={self.action_dim}, "
            f"steps={self._step_count}, updates={self._update_count})"
        )


# ---------------------------------------------------------------------------
# Concrete base: ActorCriticAgent
# ---------------------------------------------------------------------------

class ActorCriticAgent(BaseAgent):
    """
    Actor-Critic agent with:
    - MLP observation encoder
    - Gaussian actor (squashed)
    - Value critic
    - PPO-style update with GAE
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        normalize_advantages: bool = True,
        use_recurrent: bool = False,
        encoder_layers: int = 3,
        squash_output: bool = True,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device,
            seed=seed,
        )

        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.normalize_advantages = normalize_advantages
        self.use_recurrent = use_recurrent

        # Build networks
        if use_recurrent:
            self.encoder = RecurrentObservationEncoder(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
            ).to(self.device)
        else:
            self.encoder = ObservationEncoder(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
                num_layers=encoder_layers,
            ).to(self.device)

        self.actor = GaussianActor(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            squash_output=squash_output,
        ).to(self.device)

        self.critic = ValueCritic(hidden_dim=hidden_dim).to(self.device)

        # Optimizers
        actor_params = (
            list(self.encoder.parameters()) + list(self.actor.parameters())
        )
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Hidden state for recurrent mode
        self._hidden: Optional[torch.Tensor] = None

    def reset_hidden(self, batch_size: int = 1) -> None:
        if self.use_recurrent:
            self._hidden = self.encoder.init_hidden(batch_size, self.device)

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        obs_tensor = self.to_tensor(obs).unsqueeze(0)

        with torch.no_grad():
            if self.use_recurrent:
                if self._hidden is None:
                    self.reset_hidden(1)
                hidden_out, self._hidden = self.encoder(obs_tensor, self._hidden)
                enc = hidden_out.squeeze(0)
            else:
                enc = self.encoder(obs_tensor).squeeze(0)

            action, log_prob = self.actor.get_action(enc, deterministic=deterministic)
            value = self.critic(enc)

        action_np = action.cpu().numpy()
        log_prob_f = float(log_prob.cpu().item())
        value_f = float(value.cpu().item())
        return action_np, log_prob_f, value_f

    def update(self, batch: Optional[Dict] = None) -> Dict[str, float]:
        """
        PPO update on stored buffer.
        Returns dict of training metrics.
        """
        if len(self.buffer) == 0:
            return {}

        # Compute last value for bootstrapping
        last_value = 0.0

        adv, rets = self.buffer.compute_returns(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            last_value=last_value,
        )

        data = self.buffer.to_arrays()
        obs_t = self.to_tensor(data["obs"])
        acts_t = self.to_tensor(data["actions"])
        old_lp_t = self.to_tensor(data["log_probs"])
        adv_t = self.to_tensor(adv)
        ret_t = self.to_tensor(rets)

        if self.normalize_advantages:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + EPS)

        n = obs_t.shape[0]
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            # Mini-batches
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.mini_batch_size):
                idx = indices[start:start + self.mini_batch_size]
                mb_obs = obs_t[idx]
                mb_acts = acts_t[idx]
                mb_old_lp = old_lp_t[idx]
                mb_adv = adv_t[idx]
                mb_ret = ret_t[idx]

                enc = self.encoder(mb_obs)
                new_lp, entropy = self.actor.evaluate_actions(enc, mb_acts)
                value_pred = self.critic(enc)

                # Policy loss (PPO clip)
                ratio = torch.exp(new_lp.squeeze(-1) - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(value_pred.squeeze(-1), mb_ret)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = (
                    actor_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.actor.parameters()),
                    self.max_grad_norm,
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_grad_norm,
                )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_actor_loss += float(actor_loss.item())
                total_critic_loss += float(value_loss.item())
                total_entropy += float(-entropy_loss.item())
                num_updates += 1

        self.buffer.clear()
        self._update_count += 1

        if num_updates == 0:
            return {}

        metrics = {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }
        for k, v in metrics.items():
            self.log_metric(k, v)
        return metrics


# ---------------------------------------------------------------------------
# SAC-style agent
# ---------------------------------------------------------------------------

class SACAgent(BaseAgent):
    """
    Soft Actor-Critic agent.
    - Entropy-regularized off-policy RL
    - Twin Q-networks for critic
    - Automatic entropy tuning
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        target_entropy: Optional[float] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr_actor=lr,
            lr_critic=lr,
            gamma=gamma,
            device=device,
            seed=seed,
        )
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy = auto_entropy
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

        # Networks
        self.encoder = ObservationEncoder(obs_dim, hidden_dim).to(self.device)
        self.actor = GaussianActor(hidden_dim, action_dim, squash_output=True).to(self.device)

        self.critic = DoubleQValueCritic(hidden_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()), lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Auto entropy
        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy

        if auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # Replay buffer
        from collections import deque
        self._replay: deque = deque(maxlen=buffer_size)

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        obs_t = self.to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            enc = self.encoder(obs_t)
            action, log_prob = self.actor.get_action(enc, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy(), float(log_prob.item()), 0.0

    def observe(self, transition: Transition) -> None:
        self._replay.append(transition)
        self._step_count += 1

    def update(self, batch: Optional[Dict] = None) -> Dict[str, float]:
        if len(self._replay) < max(self.warmup_steps, self.batch_size):
            return {}

        # Sample mini-batch
        indices = np.random.randint(0, len(self._replay), self.batch_size)
        transitions = [self._replay[i] for i in indices]

        obs = self.to_tensor(np.array([t.obs for t in transitions]))
        acts = self.to_tensor(np.array([t.action for t in transitions]))
        rews = self.to_tensor(np.array([t.reward for t in transitions]))
        next_obs = self.to_tensor(np.array([t.next_obs for t in transitions]))
        dones = self.to_tensor(np.array([t.done for t in transitions]))

        alpha = self.log_alpha.exp().detach() if self.auto_entropy else self.alpha

        # Critic update
        with torch.no_grad():
            enc_next = self.encoder(next_obs)
            next_act, next_lp = self.actor.get_action(enc_next)
            q1_t, q2_t = self.critic_target(enc_next, next_act)
            q_target = torch.min(q1_t, q2_t) - alpha * next_lp
            target_q = rews.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * q_target

        enc = self.encoder(obs)
        q1, q2 = self.critic(enc.detach(), acts)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        enc2 = self.encoder(obs)
        new_act, log_p = self.actor.get_action(enc2)
        q1_pi, q2_pi = self.critic(enc2.detach(), new_act)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * log_p - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = 0.0
        if self.auto_entropy:
            alpha_loss_t = -(self.log_alpha * (log_p + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss_t.backward()
            self.alpha_optimizer.step()
            alpha_loss = float(alpha_loss_t.item())

        # Soft update target
        soft_update(self.critic_target, self.critic, self.tau)

        self._update_count += 1
        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(alpha.item() if hasattr(alpha, "item") else alpha),
            "alpha_loss": alpha_loss,
        }
        for k, v in metrics.items():
            self.log_metric(k, v)
        return metrics


# ---------------------------------------------------------------------------
# TD3 agent
# ---------------------------------------------------------------------------

class TD3Agent(BaseAgent):
    """
    Twin Delayed DDPG (TD3) agent.
    - Deterministic policy
    - Delayed policy updates
    - Target policy smoothing
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        exploration_noise: float = 0.1,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            agent_id=agent_id, obs_dim=obs_dim, action_dim=action_dim,
            hidden_dim=hidden_dim, lr_actor=lr, lr_critic=lr,
            gamma=gamma, device=device, seed=seed,
        )
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

        # Networks
        self.encoder = ObservationEncoder(obs_dim, hidden_dim).to(self.device)
        # Deterministic actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
            nn.Tanh(),
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        self.critic = DoubleQValueCritic(hidden_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()), lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self._ou_noise = OUNoise(action_dim, sigma=exploration_noise, seed=seed)

        from collections import deque
        self._replay: collections.deque = collections.deque(maxlen=buffer_size)

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        obs_t = self.to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            enc = self.encoder(obs_t)
            action = self.actor(enc).squeeze(0).cpu().numpy()
        if not deterministic:
            noise = self._ou_noise.sample()
            action = np.clip(action + noise, -1, 1)
        return action, 0.0, 0.0

    def observe(self, transition: Transition) -> None:
        self._replay.append(transition)
        self._step_count += 1

    def update(self, batch: Optional[Dict] = None) -> Dict[str, float]:
        if len(self._replay) < max(self.warmup_steps, self.batch_size):
            return {}

        indices = np.random.randint(0, len(self._replay), self.batch_size)
        transitions = [self._replay[i] for i in indices]

        obs = self.to_tensor(np.array([t.obs for t in transitions]))
        acts = self.to_tensor(np.array([t.action for t in transitions]))
        rews = self.to_tensor(np.array([t.reward for t in transitions]))
        next_obs = self.to_tensor(np.array([t.next_obs for t in transitions]))
        dones = self.to_tensor(np.array([t.done for t in transitions]))

        with torch.no_grad():
            enc_next = self.encoder(next_obs)
            noise = (
                torch.randn_like(acts) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.actor_target(enc_next) + noise).clamp(-1, 1)
            q1_t, q2_t = self.critic_target(enc_next, next_act)
            target_q = rews.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * torch.min(q1_t, q2_t)

        enc = self.encoder(obs)
        q1, q2 = self.critic(enc.detach(), acts)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_v = 0.0
        if self._update_count % self.policy_delay == 0:
            enc2 = self.encoder(obs)
            pi = self.actor(enc2)
            actor_loss = -self.critic.q1(enc2.detach(), pi).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            soft_update(self.actor_target, self.actor, self.tau)
            actor_loss_v = float(actor_loss.item())

        soft_update(self.critic_target, self.critic, self.tau)
        self._update_count += 1

        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_v,
        }
        for k, v in metrics.items():
            self.log_metric(k, v)
        return metrics


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "layer_init", "soft_update", "hard_update",
    "NoisyLinear", "RunningMeanStd",
    "ObservationEncoder", "RecurrentObservationEncoder",
    "GaussianActor", "DiscreteActor",
    "ValueCritic", "QValueCritic", "DoubleQValueCritic",
    "OUNoise", "ParameterNoise",
    "Transition", "EpisodeBuffer",
    "BaseAgent", "ActorCriticAgent", "SACAgent", "TD3Agent",
    "EPS", "LOG_STD_MIN", "LOG_STD_MAX",
]
