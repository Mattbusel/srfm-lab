"""
mean_field_agent.py — Mean Field Game (MFG) agent for financial markets.

Implements:
- Mean field approximation of other agents' actions/states
- MFG equilibrium computation via fixed-point iteration
- Best-response dynamics
- Fictitious play for equilibrium convergence
- Mean field Q-learning (MF-Q and MF-AC)
- Population state tracking
"""

from __future__ import annotations

import math
import logging
import collections
import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agent import (
    BaseAgent, ObservationEncoder, GaussianActor, ValueCritic,
    QValueCritic, RunningMeanStd, Transition, EpisodeBuffer,
    layer_init, soft_update, EPS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mean field representation
# ---------------------------------------------------------------------------

class MeanFieldRepresentation:
    """
    Tracks and updates the mean field (distribution of agent states/actions).

    Maintains EMA of mean action, variance, and histogram.
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        num_bins: int = 20,
        ema_alpha: float = 0.1,
    ):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.num_bins = num_bins
        self.alpha = ema_alpha

        self.mean_action = np.zeros(action_dim, dtype=np.float32)
        self.var_action = np.ones(action_dim, dtype=np.float32)
        self.mean_obs = np.zeros(obs_dim, dtype=np.float32)

        self.action_hist = np.ones((action_dim, num_bins), dtype=np.float32) / num_bins
        self._bins = np.linspace(-1, 1, num_bins + 1)
        self._history: collections.deque = collections.deque(maxlen=200)

    def update(self, actions: List[np.ndarray], obs: Optional[List[np.ndarray]] = None) -> None:
        if not actions:
            return
        batch = np.array(actions, dtype=np.float32)
        new_mean = batch.mean(axis=0)
        new_var = batch.var(axis=0) + EPS
        self.mean_action = (1 - self.alpha) * self.mean_action + self.alpha * new_mean
        self.var_action = (1 - self.alpha) * self.var_action + self.alpha * new_var

        if obs is not None:
            obs_batch = np.array(obs, dtype=np.float32)
            self.mean_obs = (1 - self.alpha) * self.mean_obs + self.alpha * obs_batch.mean(axis=0)

        for d in range(self.action_dim):
            hist, _ = np.histogram(batch[:, d], bins=self._bins, density=True)
            hist_norm = hist / (hist.sum() + EPS)
            self.action_hist[d] = (1 - self.alpha) * self.action_hist[d] + self.alpha * hist_norm

        self._history.append(self.mean_action.copy())

    def get_feature_vector(self) -> np.ndarray:
        return np.concatenate([
            self.mean_action,
            self.var_action,
            np.sqrt(self.var_action),
            self.action_hist.flatten(),
        ]).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        return self.action_dim * 3 + self.action_dim * self.num_bins

    def is_converged(self, tol: float = 1e-3) -> bool:
        if len(self._history) < 20:
            return False
        recent = np.array(list(self._history)[-20:])
        return float(np.max(np.std(recent, axis=0))) < tol

    def reset(self) -> None:
        self.mean_action = np.zeros(self.action_dim, dtype=np.float32)
        self.var_action = np.ones(self.action_dim, dtype=np.float32)
        self.mean_obs = np.zeros(self.obs_dim, dtype=np.float32)
        self.action_hist = np.ones((self.action_dim, self.num_bins), dtype=np.float32) / self.num_bins
        self._history.clear()


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class MeanFieldEncoder(nn.Module):
    """Encodes mean field features into a compact embedding."""

    def __init__(self, mf_feature_dim: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(mf_feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MFGQNetwork(nn.Module):
    """Q(obs_enc, action, mf_embed) -> value."""

    def __init__(self, obs_enc_dim: int, action_dim: int, mf_embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = obs_enc_dim + action_dim + mf_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, obs_enc: torch.Tensor, action: torch.Tensor, mf_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_enc, action, mf_embed], dim=-1)
        return self.net(x)


class BestResponsePolicy(nn.Module):
    """Policy conditioned on (obs, mean_field) for best-response computation."""

    def __init__(self, obs_enc_dim: int, mf_embed_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = obs_enc_dim + mf_embed_dim
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs_enc: torch.Tensor, mf_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs_enc, mf_embed], dim=-1)
        h = self.base(x)
        mean = torch.tanh(self.mean_head(h))
        log_std = torch.clamp(self.log_std, -5, 2)
        return mean, log_std

    def get_action(
        self, obs_enc: torch.Tensor, mf_embed: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs_enc, mf_embed)
        if deterministic:
            return mean, torch.zeros(mean.shape[0], 1, device=mean.device)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x).sum(dim=-1, keepdim=True)
        log_prob -= torch.sum(torch.log(torch.clamp(1 - action ** 2, min=EPS)), dim=-1, keepdim=True)
        return action, log_prob


# ---------------------------------------------------------------------------
# Main MFG agent
# ---------------------------------------------------------------------------

class MeanFieldAgent(BaseAgent):
    """
    Mean Field Game agent.

    Approximates the influence of all other agents via mean field mu_t.
    Computes best response to mu_t and updates toward Nash equilibrium
    via fictitious play.
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 256,
        mf_embed_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        fictitious_play: bool = True,
        fp_learning_rate: float = 0.1,
        equilibrium_tol: float = 1e-3,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        mf_num_bins: int = 20,
        mf_ema_alpha: float = 0.1,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            agent_id=agent_id, obs_dim=obs_dim, action_dim=action_dim,
            hidden_dim=hidden_dim, lr_actor=lr, lr_critic=lr,
            gamma=gamma, device=device, seed=seed,
        )

        self.num_agents = num_agents
        self.mf_embed_dim = mf_embed_dim
        self.tau = tau
        self.fictitious_play = fictitious_play
        self.fp_lr = fp_learning_rate
        self.equilibrium_tol = equilibrium_tol
        self.batch_size = batch_size

        # Mean field
        self.mean_field = MeanFieldRepresentation(
            action_dim=action_dim, obs_dim=obs_dim,
            num_bins=mf_num_bins, ema_alpha=mf_ema_alpha,
        )
        mf_feature_dim = self.mean_field.feature_dim

        # Networks
        self.obs_encoder = ObservationEncoder(obs_dim, hidden_dim).to(self.device)
        self.mf_encoder = MeanFieldEncoder(mf_feature_dim, mf_embed_dim).to(self.device)
        self.policy = BestResponsePolicy(hidden_dim, mf_embed_dim, action_dim, hidden_dim).to(self.device)

        self.q1 = MFGQNetwork(hidden_dim, action_dim, mf_embed_dim).to(self.device)
        self.q2 = MFGQNetwork(hidden_dim, action_dim, mf_embed_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad = False

        if fictitious_play:
            self.fp_policy = copy.deepcopy(self.policy).to(self.device)
            self._fp_count = 0

        # Optimizers
        policy_params = (
            list(self.obs_encoder.parameters())
            + list(self.mf_encoder.parameters())
            + list(self.policy.parameters())
        )
        self.actor_optimizer = torch.optim.Adam(policy_params, lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(action_dim)

        self._replay: collections.deque = collections.deque(maxlen=buffer_size)
        self._mf_replay: collections.deque = collections.deque(maxlen=buffer_size)

    def _get_mf_embedding(self) -> torch.Tensor:
        feat = self.mean_field.get_feature_vector()
        t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.mf_encoder(t)

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        obs_t = self.to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            enc = self.obs_encoder(obs_t)
            mf_embed = self._get_mf_embedding()
            if self.fictitious_play and self._fp_count > 0:
                a_br, lp_br = self.policy.get_action(enc, mf_embed, deterministic)
                a_fp, _ = self.fp_policy.get_action(enc, mf_embed, deterministic)
                action = self.fp_lr * a_br + (1 - self.fp_lr) * a_fp
                action = torch.clamp(action, -1, 1)
                log_prob = float(lp_br.item())
            else:
                action, lp_t = self.policy.get_action(enc, mf_embed, deterministic)
                log_prob = float(lp_t.item())
        return action.squeeze(0).cpu().numpy(), log_prob, 0.0

    def update_mean_field(self, all_actions: List[np.ndarray], all_obs: Optional[List[np.ndarray]] = None) -> None:
        self.mean_field.update(all_actions, all_obs)

    def observe(self, transition: Transition) -> None:
        self._replay.append(transition)
        self._mf_replay.append(self.mean_field.get_feature_vector().copy())
        self._step_count += 1

    def update(self, batch: Optional[Dict] = None) -> Dict[str, float]:
        if len(self._replay) < self.batch_size:
            return {}

        indices = np.random.randint(0, len(self._replay), self.batch_size)
        transitions = [self._replay[i] for i in indices]
        mf_feats = [self._mf_replay[min(i, len(self._mf_replay) - 1)] for i in indices]

        obs = self.to_tensor(np.array([t.obs for t in transitions]))
        acts = self.to_tensor(np.array([t.action for t in transitions]))
        rews = self.to_tensor(np.array([t.reward for t in transitions]))
        next_obs = self.to_tensor(np.array([t.next_obs for t in transitions]))
        dones = self.to_tensor(np.array([t.done for t in transitions]))
        mf_t = self.to_tensor(np.array(mf_feats))

        alpha = self.log_alpha.exp().detach()

        enc = self.obs_encoder(obs)
        mf_embed = self.mf_encoder(mf_t)
        enc_next = self.obs_encoder(next_obs)

        # Critic update
        with torch.no_grad():
            next_act, next_lp = self.policy.get_action(enc_next, mf_embed)
            q1_t = self.q1_target(enc_next, next_act, mf_embed)
            q2_t = self.q2_target(enc_next, next_act, mf_embed)
            target_q = (
                rews.unsqueeze(-1)
                + self.gamma * (1 - dones.unsqueeze(-1))
                * (torch.min(q1_t, q2_t) - alpha * next_lp)
            )

        q1_pred = self.q1(enc.detach(), acts, mf_embed.detach())
        q2_pred = self.q2(enc.detach(), acts, mf_embed.detach())
        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.critic_optimizer.step()

        # Actor update
        enc2 = self.obs_encoder(obs)
        mf2 = self.mf_encoder(mf_t)
        new_act, new_lp = self.policy.get_action(enc2, mf2)
        q1_pi = self.q1(enc2.detach(), new_act, mf2.detach())
        q2_pi = self.q2(enc2.detach(), new_act, mf2.detach())
        actor_loss = (alpha * new_lp - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.obs_encoder.parameters()) + list(self.mf_encoder.parameters()) + list(self.policy.parameters()),
            10.0,
        )
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (new_lp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)

        if self.fictitious_play:
            lr = min(1.0 / (self._fp_count + 1), self.fp_lr)
            soft_update(self.fp_policy, self.policy, lr)
            self._fp_count += 1

        self._update_count += 1
        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
        }
        for k, v in metrics.items():
            self.log_metric(k, v)
        return metrics

    def compute_mfg_equilibrium(self, num_iterations: int = 100, tol: float = 1e-3) -> Tuple[bool, int]:
        for i in range(num_iterations):
            if self.mean_field.is_converged(tol=tol):
                return True, i
        return False, num_iterations

    def get_equilibrium_gap(self) -> float:
        if len(self._replay) < 10:
            return float("nan")
        indices = np.random.randint(0, min(len(self._replay), 100), 10)
        gaps = []
        for i in indices:
            t = self._replay[i]
            obs_t = self.to_tensor(t.obs).unsqueeze(0)
            mf_t = self.to_tensor(self.mean_field.get_feature_vector()).unsqueeze(0)
            with torch.no_grad():
                enc = self.obs_encoder(obs_t)
                mf_embed = self.mf_encoder(mf_t)
                pi_action, _ = self.policy.get_action(enc, mf_embed, deterministic=True)
                q_pi = torch.min(self.q1(enc, pi_action, mf_embed), self.q2(enc, pi_action, mf_embed))
                rand_action = torch.tanh(torch.randn_like(pi_action))
                q_rand = torch.min(self.q1(enc, rand_action, mf_embed), self.q2(enc, rand_action, mf_embed))
                gaps.append(float((q_pi - q_rand).item()))
        return float(np.mean(gaps))


# ---------------------------------------------------------------------------
# Population tracker
# ---------------------------------------------------------------------------

class PopulationMeanFieldTracker:
    """Tracks and broadcasts mean field across the agent population."""

    def __init__(self, num_agents: int, action_dim: int, obs_dim: int, history_len: int = 500):
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self._action_history: collections.deque = collections.deque(maxlen=history_len)
        self._obs_history: collections.deque = collections.deque(maxlen=history_len)
        self.mean_action = np.zeros(action_dim, dtype=np.float32)
        self.std_action = np.ones(action_dim, dtype=np.float32)

    def record(self, actions: List[np.ndarray], observations: Optional[List[np.ndarray]] = None) -> None:
        batch = np.array(actions, dtype=np.float32)
        self._action_history.append(batch)
        self.mean_action = batch.mean(axis=0)
        self.std_action = batch.std(axis=0) + EPS
        if observations is not None:
            self._obs_history.append(np.array(observations, dtype=np.float32))

    def broadcast_to_agents(self, agents: List[MeanFieldAgent]) -> None:
        if not self._action_history:
            return
        last = self._action_history[-1]
        all_actions = [last[i].tolist() for i in range(len(last))]
        obs_list = None
        if self._obs_history:
            last_obs = self._obs_history[-1]
            obs_list = [last_obs[i].tolist() for i in range(len(last_obs))]
        for agent in agents:
            agent.update_mean_field(all_actions, obs_list)

    def get_current_mf(self) -> Dict[str, np.ndarray]:
        return {"mean_action": self.mean_action.copy(), "std_action": self.std_action.copy()}


# ---------------------------------------------------------------------------
# Equilibrium solver
# ---------------------------------------------------------------------------

class MFGEquilibriumSolver:
    """Iterative MFG equilibrium solver."""

    def __init__(
        self,
        agents: List[MeanFieldAgent],
        tracker: PopulationMeanFieldTracker,
        convergence_tol: float = 1e-3,
        max_iterations: int = 200,
    ):
        self.agents = agents
        self.tracker = tracker
        self.convergence_tol = convergence_tol
        self.max_iterations = max_iterations
        self._iteration = 0
        self._convergence_history: List[float] = []

    def step(
        self, current_obs: List[np.ndarray], current_actions: List[np.ndarray]
    ) -> Tuple[bool, float]:
        old_mf = self.agents[0].mean_field.mean_action.copy()
        self.tracker.record(current_actions, current_obs)
        self.tracker.broadcast_to_agents(self.agents)
        new_mf = self.agents[0].mean_field.mean_action.copy()
        delta = float(np.max(np.abs(new_mf - old_mf)))
        self._convergence_history.append(delta)
        self._iteration += 1
        converged = delta < self.convergence_tol and self._iteration > 10
        return converged, delta

    def get_convergence_curve(self) -> List[float]:
        return self._convergence_history.copy()

    def reset(self) -> None:
        self._iteration = 0
        self._convergence_history.clear()
        for agent in self.agents:
            agent.mean_field.reset()


__all__ = [
    "MeanFieldRepresentation", "MeanFieldEncoder", "MFGQNetwork",
    "BestResponsePolicy", "MeanFieldAgent",
    "PopulationMeanFieldTracker", "MFGEquilibriumSolver",
]
