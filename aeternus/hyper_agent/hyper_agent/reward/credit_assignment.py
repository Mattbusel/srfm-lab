"""
Credit Assignment for Multi-Agent RL.

Implementations:
  - COMA (Counterfactual Multi-Agent Policy Gradients)
  - QMIX (Monotonic Q-Value Mixing)
  - VDN  (Value Decomposition Networks)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ============================================================
# COMA: Counterfactual Multi-Agent
# ============================================================

class COMAQNet(nn.Module):
    """
    Centralized Q-network for COMA.

    Q(s, a_1, ..., a_n) for all action combinations.
    In practice, we compute Q for each agent with other agents' actions fixed.
    """

    def __init__(
        self,
        global_state_dim: int,
        n_agents:         int,
        n_actions:        int = 3,
        hidden_dim:       int = 128,
    ) -> None:
        super().__init__()
        self.n_agents  = n_agents
        self.n_actions = n_actions

        # Input: global state + all agents' actions (one-hot)
        input_dim = global_state_dim + n_agents * n_actions
        self.net  = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),  # Q-values for one agent's actions
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        global_state: torch.Tensor,   # (batch, global_dim)
        joint_actions: torch.Tensor,  # (batch, n_agents * n_actions) one-hot
    ) -> torch.Tensor:
        """Returns Q-values (batch, n_actions) for a single focal agent."""
        x = torch.cat([global_state, joint_actions], dim=-1)
        return self.net(x)


class COMAAdvantage:
    """
    COMA counterfactual advantage estimation.

    A_i(s, a) = Q(s, a_i, a_{-i}) - Σ_{a'} π_i(a'|o_i) * Q(s, a', a_{-i})

    This isolates agent i's marginal contribution to the team Q-value.

    The "counterfactual baseline" Σ π Q marginalizes over agent i's actions
    while keeping all other agents' actions fixed.
    """

    def __init__(
        self,
        global_state_dim: int,
        n_agents:         int,
        n_actions:        int = 3,
        hidden_dim:       int = 128,
        lr:               float = 1e-3,
        device:           str   = "cpu",
    ) -> None:
        self.n_agents  = n_agents
        self.n_actions = n_actions
        self.device    = torch.device(device)

        self.q_nets = nn.ModuleList([
            COMAQNet(global_state_dim, n_agents, n_actions, hidden_dim)
            for _ in range(n_agents)
        ]).to(self.device)

        self.optimizer = Adam(self.q_nets.parameters(), lr=lr)

    def compute_advantage(
        self,
        agent_idx:     int,
        global_states: torch.Tensor,     # (batch, global_dim)
        joint_actions: torch.Tensor,     # (batch, n_agents * n_actions)
        agent_actions: torch.Tensor,     # (batch,) int action of focal agent
        agent_probs:   torch.Tensor,     # (batch, n_actions) focal agent's policy
    ) -> torch.Tensor:
        """
        Returns counterfactual advantage (batch,) for agent_idx.
        """
        q_all  = self.q_nets[agent_idx](global_states, joint_actions)  # (batch, n_actions)
        q_taken = q_all.gather(1, agent_actions.unsqueeze(1)).squeeze(1)  # (batch,)
        # Counterfactual baseline: Σ_a' π(a') Q(a')
        baseline = (agent_probs * q_all).sum(dim=-1)  # (batch,)
        return q_taken - baseline

    def update(
        self,
        agent_idx:     int,
        global_states: torch.Tensor,
        joint_actions: torch.Tensor,
        agent_actions: torch.Tensor,
        targets:       torch.Tensor,   # TD targets
    ) -> float:
        """Update Q-network for agent_idx."""
        q_all   = self.q_nets[agent_idx](global_states, joint_actions)
        q_taken = q_all.gather(1, agent_actions.unsqueeze(1)).squeeze(1)
        loss    = F.mse_loss(q_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_nets[agent_idx].parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())


# ============================================================
# QMIX: Monotonic Q-Value Mixing
# ============================================================

class QMIXAgent(nn.Module):
    """
    Per-agent Q-network for QMIX.
    Takes local observation and returns Q-values over actions.
    """

    def __init__(
        self,
        obs_dim:    int,
        n_actions:  int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QMIXHyperNet(nn.Module):
    """
    Hypernetwork that generates mixing weights conditioned on global state.

    Monotonicity constraint: weights are always positive (via abs or softplus).
    """

    def __init__(
        self,
        global_state_dim: int,
        n_agents:         int,
        mixing_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_agents        = n_agents
        self.mixing_embed_dim = mixing_embed_dim

        # Hypernetworks for W1, b1, W2, b2
        self.hyper_w1 = nn.Sequential(
            nn.Linear(global_state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, n_agents * mixing_embed_dim),
        )
        self.hyper_b1 = nn.Linear(global_state_dim, mixing_embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(global_state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(global_state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )

    def forward(
        self,
        agent_q_vals:  torch.Tensor,  # (batch, n_agents)
        global_states: torch.Tensor,  # (batch, global_dim)
    ) -> torch.Tensor:
        """
        Returns mixed team Q-value (batch, 1).

        Monotonicity: W1, W2 are abs-constrained to be positive.
        """
        B = agent_q_vals.shape[0]

        # Layer 1
        w1 = torch.abs(self.hyper_w1(global_states))  # (batch, n_agents * embed)
        b1 = self.hyper_b1(global_states)              # (batch, embed)
        w1 = w1.view(B, self.n_agents, self.mixing_embed_dim)

        h  = F.elu(
            torch.bmm(agent_q_vals.unsqueeze(1), w1).squeeze(1) + b1
        )  # (batch, embed)

        # Layer 2
        w2 = torch.abs(self.hyper_w2(global_states))  # (batch, embed)
        b2 = self.hyper_b2(global_states)              # (batch, 1)

        q_tot = torch.sum(h * w2, dim=-1, keepdim=True) + b2  # (batch, 1)
        return q_tot


class QMIXMixer:
    """
    QMIX: Monotonic mixing of individual Q-values into team Q-value.

    Training:
      1. Compute individual agent Q-values for taken actions
      2. Mix via hypernetwork into team Q-value Q_tot
      3. Minimize TD loss on Q_tot
      4. Gradients flow through mixer back to individual agents

    The monotonicity constraint ensures that:
      argmax(Q_tot) = argmax(Q_individual) for each agent independently.
    This allows decentralized execution with centralized training.
    """

    def __init__(
        self,
        obs_dim:          int,
        global_state_dim: int,
        n_agents:         int,
        n_actions:        int   = 3,
        hidden_dim:       int   = 64,
        mixing_embed_dim: int   = 32,
        lr:               float = 5e-4,
        gamma:            float = 0.99,
        target_update:    int   = 200,
        device:           str   = "cpu",
    ) -> None:
        self.n_agents  = n_agents
        self.n_actions = n_actions
        self.gamma     = gamma
        self.device    = torch.device(device)
        self._update_count = 0
        self.target_update = target_update

        # Individual agent networks
        self.agent_nets = nn.ModuleList([
            QMIXAgent(obs_dim, n_actions, hidden_dim)
            for _ in range(n_agents)
        ]).to(self.device)

        self.agent_targets = nn.ModuleList([
            QMIXAgent(obs_dim, n_actions, hidden_dim)
            for _ in range(n_agents)
        ]).to(self.device)

        # Mixer
        self.mixer        = QMIXHyperNet(global_state_dim, n_agents, mixing_embed_dim).to(self.device)
        self.mixer_target = QMIXHyperNet(global_state_dim, n_agents, mixing_embed_dim).to(self.device)

        # Copy weights to targets
        self._sync_targets()

        self.optimizer = Adam(
            list(self.agent_nets.parameters()) + list(self.mixer.parameters()),
            lr=lr,
        )

    def _sync_targets(self) -> None:
        for net, target in zip(self.agent_nets, self.agent_targets):
            target.load_state_dict(net.state_dict())
        self.mixer_target.load_state_dict(self.mixer.state_dict())

    @torch.no_grad()
    def act(self, agent_idx: int, obs: torch.Tensor, eps: float = 0.0) -> int:
        """Epsilon-greedy action selection for agent_idx."""
        if np.random.random() < eps:
            return int(np.random.randint(self.n_actions))
        q_vals = self.agent_nets[agent_idx](obs.unsqueeze(0))
        return int(q_vals.argmax(dim=-1).item())

    def compute_team_q(
        self,
        obs_list:      List[torch.Tensor],  # list of (batch, obs_dim) per agent
        global_states: torch.Tensor,
        actions:       torch.Tensor,        # (batch, n_agents)
        use_target:    bool = False,
    ) -> torch.Tensor:
        """Compute Q_tot for given observations, global states, and actions."""
        nets = self.agent_targets if use_target else self.agent_nets
        mixer = self.mixer_target if use_target else self.mixer

        q_vals = []
        for i, obs in enumerate(obs_list):
            q_all  = nets[i](obs)              # (batch, n_actions)
            q_i    = q_all.gather(1, actions[:, i:i+1]).squeeze(-1)  # (batch,)
            q_vals.append(q_i)

        q_agents = torch.stack(q_vals, dim=-1)   # (batch, n_agents)
        q_tot    = mixer(q_agents, global_states)  # (batch, 1)
        return q_tot

    def update(
        self,
        obs_list:       List[np.ndarray],        # (batch, obs_dim) per agent
        global_states:  np.ndarray,              # (batch, global_dim)
        next_obs_list:  List[np.ndarray],
        next_global_states: np.ndarray,
        actions:        np.ndarray,              # (batch, n_agents) int
        rewards:        np.ndarray,              # (batch,) team reward
        dones:          np.ndarray,              # (batch,) bool
    ) -> Dict[str, float]:
        obs_tensors      = [torch.FloatTensor(o).to(self.device) for o in obs_list]
        next_obs_tensors = [torch.FloatTensor(o).to(self.device) for o in next_obs_list]
        gs_t     = torch.FloatTensor(global_states).to(self.device)
        ngs_t    = torch.FloatTensor(next_global_states).to(self.device)
        acts_t   = torch.LongTensor(actions).to(self.device)
        rews_t   = torch.FloatTensor(rewards).to(self.device)
        dones_t  = torch.FloatTensor(dones).to(self.device)

        # Current Q_tot
        q_tot = self.compute_team_q(obs_tensors, gs_t, acts_t, use_target=False)

        # Target Q_tot (greedy actions from online nets, values from target)
        with torch.no_grad():
            # Get greedy actions from online nets
            greedy_acts = []
            for i, obs in enumerate(next_obs_tensors):
                q = self.agent_nets[i](obs)
                greedy_acts.append(q.argmax(dim=-1))
            greedy_acts_t = torch.stack(greedy_acts, dim=-1)  # (batch, n_agents)

            next_q_tot = self.compute_team_q(
                next_obs_tensors, ngs_t, greedy_acts_t, use_target=True
            )
            targets = rews_t.unsqueeze(-1) + self.gamma * next_q_tot * (1 - dones_t.unsqueeze(-1))

        loss = F.mse_loss(q_tot, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent_nets.parameters()) + list(self.mixer.parameters()), 10.0
        )
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.target_update == 0:
            self._sync_targets()

        return {"qmix_loss": float(loss.item()), "mean_q_tot": float(q_tot.mean().item())}


# ============================================================
# VDN: Value Decomposition Networks
# ============================================================

class VDNMixer:
    """
    Value Decomposition Networks: additive decomposition of team value.

    Q_tot(s, a) = Σ_i Q_i(o_i, a_i)

    Simpler than QMIX (no hypernetwork), but less expressive.
    Works well when agent rewards are largely independent.

    Training: TD loss on Q_tot with gradient flowing to each Q_i.
    """

    def __init__(
        self,
        obs_dim:    int,
        n_agents:   int,
        n_actions:  int   = 3,
        hidden_dim: int   = 64,
        lr:         float = 5e-4,
        gamma:      float = 0.99,
        device:     str   = "cpu",
    ) -> None:
        self.n_agents  = n_agents
        self.n_actions = n_actions
        self.gamma     = gamma
        self.device    = torch.device(device)

        # Per-agent Q-networks
        self.q_nets = nn.ModuleList([
            QMIXAgent(obs_dim, n_actions, hidden_dim)
            for _ in range(n_agents)
        ]).to(self.device)

        self.q_targets = nn.ModuleList([
            QMIXAgent(obs_dim, n_actions, hidden_dim)
            for _ in range(n_agents)
        ]).to(self.device)

        self._sync_targets()
        self.optimizer = Adam(self.q_nets.parameters(), lr=lr)
        self._step     = 0

    def _sync_targets(self) -> None:
        for net, target in zip(self.q_nets, self.q_targets):
            target.load_state_dict(net.state_dict())

    @torch.no_grad()
    def act(self, agent_idx: int, obs: np.ndarray, eps: float = 0.0) -> int:
        if np.random.random() < eps:
            return int(np.random.randint(self.n_actions))
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        q_vals = self.q_nets[agent_idx](obs_t)
        return int(q_vals.argmax(dim=-1).item())

    def update(
        self,
        obs_list:      List[np.ndarray],  # per-agent (batch, obs_dim)
        next_obs_list: List[np.ndarray],
        actions:       np.ndarray,        # (batch, n_agents)
        team_rewards:  np.ndarray,        # (batch,)
        dones:         np.ndarray,
        target_freq:   int = 100,
    ) -> Dict[str, float]:
        obs_t    = [torch.FloatTensor(o).to(self.device) for o in obs_list]
        nobs_t   = [torch.FloatTensor(o).to(self.device) for o in next_obs_list]
        acts_t   = torch.LongTensor(actions).to(self.device)
        rews_t   = torch.FloatTensor(team_rewards).to(self.device)
        dones_t  = torch.FloatTensor(dones).to(self.device)

        # Q_tot = sum of Q_i for taken actions
        q_sum = torch.zeros(obs_t[0].shape[0], device=self.device)
        for i, obs in enumerate(obs_t):
            q_all  = self.q_nets[i](obs)
            q_i    = q_all.gather(1, acts_t[:, i:i+1]).squeeze(-1)
            q_sum += q_i

        # Target
        with torch.no_grad():
            next_q_sum = torch.zeros_like(q_sum)
            for i, obs in enumerate(nobs_t):
                q_all  = self.q_targets[i](obs)
                best_q = q_all.max(dim=-1).values
                next_q_sum += best_q
            targets = rews_t + self.gamma * next_q_sum * (1.0 - dones_t)

        loss = F.mse_loss(q_sum, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_nets.parameters(), 10.0)
        self.optimizer.step()

        self._step += 1
        if self._step % target_freq == 0:
            self._sync_targets()

        return {"vdn_loss": float(loss.item()), "mean_q_tot": float(q_sum.mean().item())}

    def individual_q_values(
        self, agent_idx: int, obs: np.ndarray
    ) -> np.ndarray:
        """Return Q-values for all actions for agent_idx."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_nets[agent_idx](obs_t)
        return q.squeeze(0).cpu().numpy()
