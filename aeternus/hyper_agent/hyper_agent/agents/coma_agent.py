"""
coma_agent.py — Counterfactual Multi-Agent (COMA) policy gradient.

Architecture:
- Centralized critic: conditions on global state + all agents' actions
- Decentralized actor: local observations -> action distribution
- Counterfactual baseline: isolates individual agent contribution
- Advantage = Q(s, a) - sum_a' pi(a'|tau) Q(s, (a_{-i}, a'))
"""

from __future__ import annotations

import math
import copy
import logging
import collections
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agent import (
    BaseAgent, ObservationEncoder, GaussianActor, ValueCritic,
    RunningMeanStd, Transition, EpisodeBuffer,
    layer_init, soft_update, EPS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Centralized COMA critic
# ---------------------------------------------------------------------------

class COMACounterfactualCritic(nn.Module):
    """
    Centralized critic for COMA.

    Computes Q(s, a_1, ..., a_N) for all agents.
    The counterfactual baseline isolates each agent's contribution by
    computing the expected Q over the agent's own action while keeping
    others fixed.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        # Input: state + all agents' actions
        input_dim = state_dim + num_agents * action_dim

        layers: List[nn.Module] = []
        in_d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_d = hidden_dim
        self.base = nn.Sequential(*layers)

        # One Q-value head per agent
        self.q_heads = nn.ModuleList([
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
            for _ in range(num_agents)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        state: torch.Tensor,
        all_actions: torch.Tensor,
        agent_id: int,
    ) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim)
            all_actions: (B, N * action_dim) flattened
            agent_id: which agent's Q-value head to use
        Returns:
            q_value: (B, 1)
        """
        x = torch.cat([state, all_actions], dim=-1)
        h = self.base(x)
        return self.q_heads[agent_id](h)

    def forward_all(
        self,
        state: torch.Tensor,
        all_actions: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Compute Q-values for all agents simultaneously."""
        x = torch.cat([state, all_actions], dim=-1)
        h = self.base(x)
        return [head(h) for head in self.q_heads]

    def counterfactual_advantage(
        self,
        state: torch.Tensor,
        all_actions: torch.Tensor,
        actor: "COMADecentralizedActor",
        obs: torch.Tensor,
        agent_id: int,
        num_samples: int = 10,
    ) -> torch.Tensor:
        """
        Compute counterfactual advantage for agent_id:
        A_i = Q(s, a) - E_{a_i ~ pi_i}[Q(s, (a_{-i}, a_i))]

        Uses Monte Carlo sampling for expectation.
        """
        B = state.shape[0]
        N = self.num_agents
        ad = self.action_dim

        # Q under actual joint action
        q_actual = self.forward(state, all_actions, agent_id)  # (B, 1)

        # Compute baseline: E_{a_i ~ pi_i}[Q]
        baselines = torch.zeros(B, 1, device=state.device)

        with torch.no_grad():
            # Get policy distribution for agent_id
            mean, log_std = actor.forward(obs)
            std = log_std.exp()

            for _ in range(num_samples):
                # Sample counterfactual action
                eps = torch.randn_like(mean)
                cf_action = torch.tanh(mean + std * eps)

                # Replace agent_id's action in joint action
                cf_joint = all_actions.clone()
                start = agent_id * ad
                cf_joint[:, start:start + ad] = cf_action

                q_cf = self.forward(state, cf_joint, agent_id)
                baselines += q_cf / num_samples

        advantage = q_actual - baselines
        return advantage


# ---------------------------------------------------------------------------
# Decentralized actor for COMA
# ---------------------------------------------------------------------------

class COMADecentralizedActor(nn.Module):
    """
    Decentralized actor network.
    Maps local observation -> Gaussian action distribution.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_d = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
            in_d = hidden_dim
        self.base = nn.Sequential(*layers)
        self.mean_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.base(obs)
        mean = torch.tanh(self.mean_head(h))
        log_std = torch.clamp(self.log_std, -5, 2)
        return mean, log_std

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        if deterministic:
            return mean, torch.zeros(mean.shape[0], 1, device=mean.device)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x).sum(dim=-1, keepdim=True)
        log_prob -= torch.sum(
            torch.log(torch.clamp(1 - action ** 2, min=EPS)), dim=-1, keepdim=True
        )
        return action, log_prob

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        # Undo squashing
        raw = torch.atanh(torch.clamp(actions, -1 + EPS, 1 - EPS))
        log_prob = dist.log_prob(raw).sum(dim=-1, keepdim=True)
        log_prob -= torch.sum(
            torch.log(torch.clamp(1 - actions ** 2, min=EPS)), dim=-1, keepdim=True
        )
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


# ---------------------------------------------------------------------------
# COMA replay buffer
# ---------------------------------------------------------------------------

class COMABuffer:
    """Trajectory buffer for COMA with global state storage."""

    def __init__(self, num_agents: int, capacity: int = 50000):
        self.num_agents = num_agents
        self.capacity = capacity

        self.obs: List[List[np.ndarray]] = [[] for _ in range(num_agents)]
        self.actions: List[List[np.ndarray]] = [[] for _ in range(num_agents)]
        self.rewards: List[List[float]] = [[] for _ in range(num_agents)]
        self.dones: List[List[bool]] = [[] for _ in range(num_agents)]
        self.log_probs: List[List[float]] = [[] for _ in range(num_agents)]
        self.global_states: List[np.ndarray] = []
        self._size = 0

    def add_step(
        self,
        obs_list: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        dones: List[bool],
        log_probs: List[float],
        global_state: np.ndarray,
    ) -> None:
        for i in range(self.num_agents):
            self.obs[i].append(obs_list[i].copy())
            self.actions[i].append(actions[i].copy())
            self.rewards[i].append(float(rewards[i]))
            self.dones[i].append(bool(dones[i]))
            self.log_probs[i].append(float(log_probs[i]))
        self.global_states.append(global_state.copy())
        self._size += 1

    def __len__(self) -> int:
        return self._size

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self._size < batch_size:
            return None
        indices = np.random.randint(0, self._size, batch_size)

        data = {}
        for i in range(self.num_agents):
            data[f"obs_{i}"] = torch.tensor(
                np.array([self.obs[i][j] for j in indices]), dtype=torch.float32, device=device
            )
            data[f"actions_{i}"] = torch.tensor(
                np.array([self.actions[i][j] for j in indices]), dtype=torch.float32, device=device
            )
            data[f"rewards_{i}"] = torch.tensor(
                np.array([self.rewards[i][j] for j in indices]), dtype=torch.float32, device=device
            )
            data[f"dones_{i}"] = torch.tensor(
                np.array([self.dones[i][j] for j in indices]), dtype=torch.float32, device=device
            )
            data[f"log_probs_{i}"] = torch.tensor(
                np.array([self.log_probs[i][j] for j in indices]), dtype=torch.float32, device=device
            )

        data["global_states"] = torch.tensor(
            np.array([self.global_states[j] for j in indices]), dtype=torch.float32, device=device
        )
        return data

    def clear(self) -> None:
        self.obs = [[] for _ in range(self.num_agents)]
        self.actions = [[] for _ in range(self.num_agents)]
        self.rewards = [[] for _ in range(self.num_agents)]
        self.dones = [[] for _ in range(self.num_agents)]
        self.log_probs = [[] for _ in range(self.num_agents)]
        self.global_states = []
        self._size = 0


# ---------------------------------------------------------------------------
# COMA agent
# ---------------------------------------------------------------------------

class COMAAgent(BaseAgent):
    """
    Counterfactual Multi-Agent Policy Gradient (COMA) agent.

    The counterfactual baseline removes the impact of agent i's action from
    Q(s, u) by computing:
        b_i(s, a_{-i}) = sum_{a_i'} pi_i(a_i'|o_i) * Q(s, (a_{-i}, a_i'))

    This isolates each agent's marginal contribution.
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        num_agents: int,
        hidden_dim: int = 128,
        critic_hidden_dim: int = 256,
        lr_actor: float = 1e-4,
        lr_critic: float = 5e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 10.0,
        batch_size: int = 128,
        buffer_capacity: int = 50_000,
        num_cf_samples: int = 10,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            agent_id=agent_id, obs_dim=obs_dim, action_dim=action_dim,
            hidden_dim=hidden_dim, lr_actor=lr_actor, lr_critic=lr_critic,
            gamma=gamma, device=device, seed=seed,
        )

        self.state_dim = state_dim
        self.num_agents = num_agents
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.num_cf_samples = num_cf_samples

        # Networks
        self.actor = COMADecentralizedActor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic = COMACounterfactualCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=critic_hidden_dim,
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Buffer
        self.coma_buffer = COMABuffer(num_agents=num_agents, capacity=buffer_capacity)

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        obs_t = self.to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy(), float(log_prob.item()), 0.0

    def update(self, batch: Optional[Dict] = None) -> Dict[str, float]:
        """
        Update actor using counterfactual advantage.
        Update critic using TD error.
        """
        data = self.coma_buffer.sample_batch(self.batch_size, self.device)
        if data is None:
            return {}

        i = self.agent_id
        obs_i = data[f"obs_{i}"]
        acts_i = data[f"actions_{i}"]
        rews_i = data[f"rewards_{i}"]
        dones_i = data[f"dones_{i}"]
        log_probs_i = data[f"log_probs_{i}"]
        states = data["global_states"]

        # Collect all agents' actions for joint action
        all_acts = torch.cat([data[f"actions_{j}"] for j in range(self.num_agents)], dim=-1)

        # Critic update
        q_pred = self.critic.forward(states, all_acts, i)
        # TD target (no bootstrapping for simplicity; use episode returns)
        with torch.no_grad():
            td_target = rews_i.unsqueeze(-1)  # (B, 1)
        critic_loss = F.mse_loss(q_pred, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Actor update with counterfactual advantage
        with torch.no_grad():
            advantage = self.critic.counterfactual_advantage(
                state=states,
                all_actions=all_acts,
                actor=self.actor,
                obs=obs_i,
                agent_id=i,
                num_samples=self.num_cf_samples,
            )
            # Normalize advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + EPS)

        # Policy gradient
        new_log_prob, entropy = self.actor.evaluate(obs_i, acts_i)
        actor_loss = -(new_log_prob * advantage.detach()).mean()
        entropy_loss = -entropy.mean()
        total_actor_loss = actor_loss + self.entropy_coef * entropy_loss

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        self._update_count += 1
        metrics = {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(-entropy_loss.item()),
            "advantage_mean": float(advantage.mean().item()),
        }
        for k, v in metrics.items():
            self.log_metric(k, v)
        return metrics


# ---------------------------------------------------------------------------
# COMA population coordinator
# ---------------------------------------------------------------------------

class COMACoordinator:
    """
    Coordinates COMA agents: manages shared critic and joint training.
    """

    def __init__(
        self,
        agents: List[COMAAgent],
        shared_critic: bool = True,
    ):
        self.agents = agents
        self.shared_critic = shared_critic
        self.num_agents = len(agents)

        if shared_critic and agents:
            # All agents share the same critic
            ref_critic = agents[0].critic
            for agent in agents[1:]:
                agent.critic = ref_critic
                agent.critic_optimizer = torch.optim.Adam(
                    ref_critic.parameters(), lr=5e-4
                )

        self.buffer = COMABuffer(num_agents=self.num_agents)

    def store_step(
        self,
        obs_list: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        dones: List[bool],
        log_probs: List[float],
        global_state: np.ndarray,
    ) -> None:
        self.buffer.add_step(obs_list, actions, rewards, dones, log_probs, global_state)
        # Also store in each agent's buffer
        for agent in self.agents:
            agent.coma_buffer.add_step(
                obs_list, actions, rewards, dones, log_probs, global_state
            )

    def update_all(self) -> List[Dict[str, float]]:
        return [agent.update() for agent in self.agents]

    def clear_buffers(self) -> None:
        self.buffer.clear()
        for agent in self.agents:
            agent.coma_buffer.clear()


__all__ = [
    "COMACounterfactualCritic",
    "COMADecentralizedActor",
    "COMABuffer",
    "COMAAgent",
    "COMACoordinator",
]
