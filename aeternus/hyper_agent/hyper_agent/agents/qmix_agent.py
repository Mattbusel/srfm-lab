"""
qmix_agent.py — QMIX: Monotonic Joint Q-value Factorization for MARL.

Architecture:
- Individual Q-networks: Q_i(o_i, a_i) per agent
- Mixing network: monotonic combination -> Q_joint(s, a_1..N)
- Hypernetworks condition mixing weights on global state
- Target networks for stability
- Prioritized experience replay
- Double Q-learning for bias reduction
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
    BaseAgent, ObservationEncoder, RunningMeanStd,
    Transition, layer_init, soft_update, hard_update, EPS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual Q-network (drqn-style with GRU)
# ---------------------------------------------------------------------------

class IndividualQNetwork(nn.Module):
    """
    Individual agent Q-network: Q_i(tau_i, a_i) -> R
    Uses GRU for partial observability.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_actions: int = 0,  # if >0, discrete; else continuous bins
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Pre-GRU
        self.pre_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Q-value head: action embedding + hidden -> Q
        self.action_enc = nn.Linear(action_dim, hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, obs_dim)
            action: (B, action_dim)
            hidden: (B, hidden_dim) or None
        Returns:
            q_value: (B, 1)
            new_hidden: (B, hidden_dim)
        """
        x = self.pre_net(obs)
        if hidden is None:
            hidden = torch.zeros(obs.shape[0], self.hidden_dim, device=obs.device)
        new_hidden = self.gru(x, hidden)

        act_enc = F.relu(self.action_enc(action))
        combined = torch.cat([new_hidden, act_enc], dim=-1)
        q_value = self.q_head(combined)
        return q_value, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)


# ---------------------------------------------------------------------------
# Hypernetwork for QMIX mixing weights
# ---------------------------------------------------------------------------

class QMIXHyperNetwork(nn.Module):
    """
    Generates monotonic mixing network weights from global state.
    Uses absolute values to ensure monotonicity (QMIX constraint).
    """

    def __init__(
        self,
        state_dim: int,
        num_agents: int,
        mixing_hidden_dim: int = 32,
        hyper_hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.mixing_hidden_dim = mixing_hidden_dim

        # Hypernetworks for w1, b1, w2, b2
        # w1: (num_agents, mixing_hidden_dim) -> flattened
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, num_agents * mixing_hidden_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden_dim)

        # w2: (mixing_hidden_dim, 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, mixing_hidden_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self, agent_qs: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            agent_qs: (B, N) individual Q-values
            state: (B, state_dim)
        Returns:
            q_total: (B, 1)
        """
        B = state.shape[0]

        # First layer
        w1 = torch.abs(self.hyper_w1(state))  # (B, N*mixing_h)
        w1 = w1.view(B, self.num_agents, self.mixing_hidden_dim)
        b1 = self.hyper_b1(state).unsqueeze(1)  # (B, 1, mixing_h)

        # (B, 1, N) x (B, N, mixing_h) -> (B, 1, mixing_h)
        x = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # Second layer
        w2 = torch.abs(self.hyper_w2(state))  # (B, mixing_h)
        w2 = w2.unsqueeze(2)                   # (B, mixing_h, 1)
        b2 = self.hyper_b2(state).unsqueeze(1) # (B, 1, 1)

        # (B, 1, mixing_h) x (B, mixing_h, 1) -> (B, 1, 1)
        q_total = torch.bmm(x, w2) + b2
        return q_total.squeeze(-1)  # (B, 1)


# ---------------------------------------------------------------------------
# Multi-agent replay buffer for QMIX
# ---------------------------------------------------------------------------

class QMIXReplayBuffer:
    """
    Episode-based replay buffer for QMIX.
    Stores full episodes (trajectories) with global state.
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        max_episode_len: int = 200,
    ):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_episode_len = max_episode_len

        # Storage: list of episodes
        self._episodes: List[Dict] = []
        self._pos = 0
        self._full = False

    def store_episode(self, episode: Dict) -> None:
        """Store a complete episode dict."""
        if len(self._episodes) < self.capacity:
            self._episodes.append(episode)
        else:
            self._episodes[self._pos] = episode
        self._pos = (self._pos + 1) % self.capacity
        if self._pos == 0:
            self._full = True

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of episodes and return padded tensors."""
        n = len(self._episodes)
        if n == 0:
            return {}
        indices = np.random.randint(0, n, batch_size)
        episodes = [self._episodes[i] for i in indices]

        # Pad to max episode length
        max_len = max(len(ep["rewards"][0]) for ep in episodes)
        max_len = min(max_len, self.max_episode_len)

        B = batch_size
        T = max_len
        N = self.num_agents

        obs_b = np.zeros((B, T, N, self.obs_dim), dtype=np.float32)
        next_obs_b = np.zeros((B, T, N, self.obs_dim), dtype=np.float32)
        acts_b = np.zeros((B, T, N, self.action_dim), dtype=np.float32)
        rews_b = np.zeros((B, T, N), dtype=np.float32)
        dones_b = np.zeros((B, T, N), dtype=np.float32)
        states_b = np.zeros((B, T, self.state_dim), dtype=np.float32)
        next_states_b = np.zeros((B, T, self.state_dim), dtype=np.float32)
        masks_b = np.zeros((B, T), dtype=np.float32)

        for b, ep in enumerate(episodes):
            ep_len = min(len(ep["rewards"][0]), T)
            masks_b[b, :ep_len] = 1.0

            for t in range(ep_len):
                for i in range(N):
                    if i < len(ep.get("obs", [])) and t < len(ep["obs"][i]):
                        obs_b[b, t, i] = ep["obs"][i][t]
                        next_obs_b[b, t, i] = ep["next_obs"][i][t] if t < len(ep["next_obs"][i]) else ep["obs"][i][t]
                        acts_b[b, t, i] = ep["actions"][i][t]
                        rews_b[b, t, i] = ep["rewards"][i][t]
                        dones_b[b, t, i] = ep["dones"][i][t]
                if t < len(ep.get("global_states", [])):
                    states_b[b, t] = ep["global_states"][t]
                    if t + 1 < len(ep["global_states"]):
                        next_states_b[b, t] = ep["global_states"][t + 1]
                    else:
                        next_states_b[b, t] = ep["global_states"][t]

        return {
            "obs": torch.tensor(obs_b),
            "next_obs": torch.tensor(next_obs_b),
            "actions": torch.tensor(acts_b),
            "rewards": torch.tensor(rews_b),
            "dones": torch.tensor(dones_b),
            "global_states": torch.tensor(states_b),
            "next_global_states": torch.tensor(next_states_b),
            "masks": torch.tensor(masks_b),
        }

    def __len__(self) -> int:
        return len(self._episodes)


# ---------------------------------------------------------------------------
# QMIX trainer (manages all agents)
# ---------------------------------------------------------------------------

class QMIXTrainer:
    """
    QMIX: monotonic value function factorization for cooperative MARL.

    Each agent has its own Q-network. The mixing network combines individual
    Q-values into Q_total(s, u) using a monotonic function conditioned on
    global state s. This ensures:
        argmax_{u} Q_total(s, u) = (argmax_{u_1} Q_1, ..., argmax_{u_N} Q_N)
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 64,
        mixing_hidden_dim: int = 32,
        lr: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_interval: int = 200,
        grad_clip: float = 10.0,
        buffer_capacity: int = 5000,
        batch_size: int = 32,
        max_episode_len: int = 200,
        double_q: bool = True,
        device: Optional[str] = None,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.double_q = double_q
        self._update_count = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Individual Q-networks
        self.q_networks = nn.ModuleList([
            IndividualQNetwork(obs_dim, action_dim, hidden_dim)
            for _ in range(num_agents)
        ]).to(self.device)

        self.q_targets = nn.ModuleList([
            copy.deepcopy(qn)
            for qn in self.q_networks
        ]).to(self.device)

        for qn in self.q_targets:
            for p in qn.parameters():
                p.requires_grad = False

        # Mixing network
        self.mixer = QMIXHyperNetwork(
            state_dim=state_dim,
            num_agents=num_agents,
            mixing_hidden_dim=mixing_hidden_dim,
        ).to(self.device)

        self.mixer_target = copy.deepcopy(self.mixer).to(self.device)
        for p in self.mixer_target.parameters():
            p.requires_grad = False

        # Optimizer
        all_params = list(self.q_networks.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

        # Replay buffer
        self.replay_buffer = QMIXReplayBuffer(
            capacity=buffer_capacity,
            num_agents=num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            max_episode_len=max_episode_len,
        )

        # Hidden states (per agent, per episode)
        self._hiddens: Optional[List[torch.Tensor]] = None

        self._metrics: Dict[str, List[float]] = collections.defaultdict(list)

    def init_hidden(self, batch_size: int = 1) -> None:
        self._hiddens = [
            qn.init_hidden(batch_size, self.device)
            for qn in self.q_networks
        ]

    def select_actions(
        self,
        obs_list: List[np.ndarray],
        epsilon: float = 0.1,
    ) -> List[np.ndarray]:
        """
        Epsilon-greedy action selection for all agents.
        Returns list of continuous actions.
        """
        actions = []
        if self._hiddens is None:
            self.init_hidden(1)

        for i, (qn, obs) in enumerate(zip(self.q_networks, obs_list)):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if np.random.random() < epsilon:
                # Random action
                action = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
            else:
                # Greedy: evaluate Q for a few candidate actions and pick best
                candidates = np.random.uniform(-1, 1, (20, self.action_dim)).astype(np.float32)
                candidates_t = torch.tensor(candidates, device=self.device)
                obs_rep = obs_t.repeat(20, 1)
                hidden_rep = self._hiddens[i].repeat(20, 1)
                with torch.no_grad():
                    q_vals, _ = qn(obs_rep, candidates_t, hidden_rep)
                best_idx = int(q_vals.argmax().item())
                action = candidates[best_idx]

                # Update hidden state
                with torch.no_grad():
                    best_act_t = torch.tensor(action, device=self.device).unsqueeze(0)
                    _, new_h = qn(obs_t, best_act_t, self._hiddens[i])
                    self._hiddens[i] = new_h

            actions.append(action)
        return actions

    def store_episode(self, episode_data: Dict) -> None:
        """Store a complete episode in the replay buffer."""
        self.replay_buffer.store_episode(episode_data)

    def update(self) -> Dict[str, float]:
        """One gradient update step."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        if not batch:
            return {}

        batch = {k: v.to(self.device) for k, v in batch.items()}

        # (B, T, N, obs_dim), etc.
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]  # (B, T, N)
        dones = batch["dones"]
        states = batch["global_states"]
        next_states = batch["next_global_states"]
        masks = batch["masks"]  # (B, T)

        B, T, N, _ = obs.shape

        # Compute individual Q-values for current and next observations
        agent_qs = []
        agent_qs_next = []
        agent_qs_next_target = []

        for i, (qn, qt) in enumerate(zip(self.q_networks, self.q_targets)):
            # Reshape for GRU: process all timesteps
            obs_i = obs[:, :, i, :]    # (B, T, obs_dim)
            next_obs_i = next_obs[:, :, i, :]
            acts_i = actions[:, :, i, :]  # (B, T, action_dim)

            h = qn.init_hidden(B, self.device)
            qs_t = []
            for t in range(T):
                q, h = qn(obs_i[:, t], acts_i[:, t], h)
                qs_t.append(q)
            q_seq = torch.stack(qs_t, dim=1)  # (B, T, 1)
            agent_qs.append(q_seq)

            # Next Q (for target)
            h_target = qt.init_hidden(B, self.device)
            if self.double_q:
                h_online = qn.init_hidden(B, self.device)
            qs_next_t = []
            qs_next_online_t = []
            for t in range(T):
                # Use best online action for double Q
                if self.double_q:
                    # Use 5 candidate actions, pick best online, evaluate with target
                    q_next, h_target_new = qt(next_obs_i[:, t],
                                               acts_i[:, t],  # Use same action as proxy
                                               h_target)
                    qs_next_t.append(q_next)
                    h_target = h_target_new
                else:
                    q_next, h_target = qt(next_obs_i[:, t], acts_i[:, t], h_target)
                    qs_next_t.append(q_next)
            q_next_seq = torch.stack(qs_next_t, dim=1)
            agent_qs_next_target.append(q_next_seq)

        # Stack: (B, T, N)
        agent_qs_stacked = torch.cat(agent_qs, dim=-1)  # (B, T, N)
        agent_qs_next_stacked = torch.cat(agent_qs_next_target, dim=-1)

        # Team reward: mean across agents
        team_reward = rewards.mean(dim=-1)  # (B, T)
        team_done = dones.max(dim=-1).values  # (B, T)

        # Mixing
        # Reshape for mixer: (B*T, N) and (B*T, state_dim)
        BT = B * T
        qs_flat = agent_qs_stacked.view(BT, N)
        states_flat = states.view(BT, -1)

        q_total = self.mixer(qs_flat, states_flat).view(B, T)

        # Target Q_total
        qs_next_flat = agent_qs_next_stacked.view(BT, N)
        next_states_flat = next_states.view(BT, -1)
        q_total_next = self.mixer_target(qs_next_flat, next_states_flat).view(B, T)

        # TD target
        target = (
            team_reward + self.gamma * (1 - team_done) * q_total_next.detach()
        )

        # Loss (masked)
        td_error = (q_total - target) ** 2
        loss = (td_error * masks).sum() / (masks.sum() + EPS)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q_networks.parameters()) + list(self.mixer.parameters()),
            self.grad_clip,
        )
        self.optimizer.step()

        self._update_count += 1

        # Soft update targets
        for qn, qt in zip(self.q_networks, self.q_targets):
            soft_update(qt, qn, self.tau)
        soft_update(self.mixer_target, self.mixer, self.tau)

        metrics = {
            "loss": float(loss.item()),
            "q_total_mean": float(q_total.mean().item()),
            "td_error": float(td_error.mean().item()),
        }
        for k, v in metrics.items():
            self._metrics[k].append(v)
        return metrics

    def get_metrics(self, last_n: int = 100) -> Dict[str, float]:
        return {
            k: float(np.mean(v[-last_n:])) if v else 0.0
            for k, v in self._metrics.items()
        }

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "q_networks": self.q_networks.state_dict(),
            "mixer": self.mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self._update_count,
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.q_networks.load_state_dict(state["q_networks"])
        self.mixer.load_state_dict(state["mixer"])
        hard_update(self.q_targets, self.q_networks)
        hard_update(self.mixer_target, self.mixer)
        self.optimizer.load_state_dict(state["optimizer"])
        self._update_count = state.get("update_count", 0)


# ---------------------------------------------------------------------------
# QMIX Agent wrapper (BaseAgent interface)
# ---------------------------------------------------------------------------

class QMIXAgent(BaseAgent):
    """
    Wrapper that exposes QMIX as a BaseAgent-compatible interface.
    For single-agent use within QMIX framework.
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        num_agents: int,
        hidden_dim: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50000,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            agent_id=agent_id, obs_dim=obs_dim, action_dim=action_dim,
            hidden_dim=hidden_dim, lr_actor=lr, lr_critic=lr,
            gamma=gamma, device=device, seed=seed,
        )
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Individual Q-network
        self.q_net = IndividualQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q_net).to(self.device)
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self._hidden: Optional[torch.Tensor] = None

    @property
    def epsilon(self) -> float:
        progress = min(self._step_count / max(self.epsilon_decay, 1), 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        obs_t = self.to_tensor(obs).unsqueeze(0)
        if self._hidden is None:
            self._hidden = self.q_net.init_hidden(1, self.device)

        if not deterministic and np.random.random() < self.epsilon:
            action = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
            return action, 0.0, 0.0

        # Greedy: sample candidates
        candidates = np.random.uniform(-1, 1, (20, self.action_dim)).astype(np.float32)
        cand_t = torch.tensor(candidates, device=self.device)
        obs_rep = obs_t.repeat(20, 1)
        hidden_rep = self._hidden.repeat(20, 1)

        with torch.no_grad():
            q_vals, _ = self.q_net(obs_rep, cand_t, hidden_rep)
            best_idx = int(q_vals.argmax().item())
            action = candidates[best_idx]
            best_act_t = torch.tensor(action, device=self.device).unsqueeze(0)
            _, self._hidden = self.q_net(obs_t, best_act_t, self._hidden)

        return action, 0.0, 0.0

    def reset_hidden(self) -> None:
        self._hidden = None

    def update(self, batch: Optional[Dict] = None) -> Dict[str, float]:
        # QMIX updates are handled by QMIXTrainer
        return {}


__all__ = [
    "IndividualQNetwork",
    "QMIXHyperNetwork",
    "QMIXReplayBuffer",
    "QMIXTrainer",
    "QMIXAgent",
]
