"""
mappo_agent.py — Multi-Agent Proximal Policy Optimization (MAPPO).

Architecture:
- Decentralized actor: local observation -> action
- Centralized critic: global state -> value (CTDE paradigm)
- GAE advantage estimation
- PPO clipping with entropy bonus
- Value normalization (running mean/std)
- Support for recurrent (GRU) actor
- Multi-epoch mini-batch updates
- Optional dual-clip PPO for improved stability
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
from torch.distributions import Normal

from .base_agent import (
    BaseAgent, ObservationEncoder, RecurrentObservationEncoder,
    GaussianActor, ValueCritic, RunningMeanStd,
    Transition, EpisodeBuffer, layer_init, soft_update, EPS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Centralized critic for MAPPO
# ---------------------------------------------------------------------------

class CentralizedCritic(nn.Module):
    """
    Centralized value function V(s_global).
    Takes global state (concatenation of all agent observations + market state).
    Uses a larger network than the decentralized actor.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        use_layer_norm: bool = True,
        use_attention: bool = False,
        num_agents: int = 1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.num_agents = num_agents

        layers: List[nn.Module] = []
        in_d = state_dim
        for i in range(num_layers):
            out_d = hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_d))
            layers.append(nn.GELU())
            in_d = out_d

        self.net = nn.Sequential(*layers)
        self.value_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        if use_attention:
            # Cross-agent attention for global state processing
            per_agent_dim = state_dim // max(num_agents, 1)
            self.attention = nn.MultiheadAttention(
                embed_dim=per_agent_dim,
                num_heads=4,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(per_agent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_state: (..., state_dim)
        Returns:
            value: (..., 1)
        """
        x = self.net(global_state)
        return self.value_head(x)


# ---------------------------------------------------------------------------
# Value normalizer
# ---------------------------------------------------------------------------

class ValueNormalizer:
    """
    Running normalization of value targets.
    Maintains running mean and variance, normalizes inputs for stable training.
    """

    def __init__(self, clip_range: float = 10.0, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        self.clip_range = clip_range
        self._alpha = 0.001

    def update(self, x: np.ndarray) -> None:
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        n = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + n
        self.mean = self.mean + delta * n / tot
        m_a = self.var * self.count
        m_b = batch_var * n
        M2 = m_a + m_b + delta ** 2 * self.count * n / tot
        self.var = M2 / tot
        self.count = tot

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean_t = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std_t = torch.tensor(math.sqrt(max(self.var, EPS)), dtype=x.dtype, device=x.device)
        return torch.clamp((x - mean_t) / std_t, -self.clip_range, self.clip_range)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean_t = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std_t = torch.tensor(math.sqrt(max(self.var, EPS)), dtype=x.dtype, device=x.device)
        return x * std_t + mean_t

    def reset(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = EPS


# ---------------------------------------------------------------------------
# Multi-agent episode buffer
# ---------------------------------------------------------------------------

class MARolloutBuffer:
    """
    Rollout buffer for multi-agent PPO.
    Stores trajectories for all agents with shared global state.
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        max_size: int = 4096,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_size = max_size

        self.obs: List[List[np.ndarray]] = [[] for _ in range(num_agents)]
        self.actions: List[List[np.ndarray]] = [[] for _ in range(num_agents)]
        self.rewards: List[List[float]] = [[] for _ in range(num_agents)]
        self.dones: List[List[bool]] = [[] for _ in range(num_agents)]
        self.log_probs: List[List[float]] = [[] for _ in range(num_agents)]
        self.values: List[List[float]] = [[] for _ in range(num_agents)]
        self.global_states: List[np.ndarray] = []
        self._size: int = 0

    def add(
        self,
        agent_id: int,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        global_state: Optional[np.ndarray] = None,
    ) -> None:
        self.obs[agent_id].append(obs.copy())
        self.actions[agent_id].append(action.copy())
        self.rewards[agent_id].append(float(reward))
        self.dones[agent_id].append(bool(done))
        self.log_probs[agent_id].append(float(log_prob))
        self.values[agent_id].append(float(value))
        if agent_id == 0 and global_state is not None:
            self.global_states.append(global_state.copy())
        self._size += 1

    def add_step(
        self,
        obs_list: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        dones: List[bool],
        log_probs: List[float],
        values: List[float],
        global_state: Optional[np.ndarray] = None,
    ) -> None:
        for i in range(self.num_agents):
            self.add(
                i, obs_list[i], actions[i], rewards[i],
                dones[i], log_probs[i], values[i],
                global_state if i == 0 else None,
            )

    def compute_advantages_and_returns(
        self,
        last_values: List[float],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute GAE advantages and returns for all agents."""
        all_adv = []
        all_ret = []

        for i in range(self.num_agents):
            n = len(self.rewards[i])
            adv = np.zeros(n, dtype=np.float32)
            gae = 0.0
            next_val = last_values[i]

            for t in reversed(range(n)):
                mask = 0.0 if self.dones[i][t] else 1.0
                delta = (
                    self.rewards[i][t]
                    + gamma * next_val * mask
                    - self.values[i][t]
                )
                gae = delta + gamma * gae_lambda * mask * gae
                adv[t] = gae
                next_val = self.values[i][t]

            returns = adv + np.array(self.values[i], dtype=np.float32)
            all_adv.append(adv)
            all_ret.append(returns)

        return all_adv, all_ret

    def __len__(self) -> int:
        return len(self.rewards[0]) if self.rewards[0] else 0

    def clear(self) -> None:
        self.obs = [[] for _ in range(self.num_agents)]
        self.actions = [[] for _ in range(self.num_agents)]
        self.rewards = [[] for _ in range(self.num_agents)]
        self.dones = [[] for _ in range(self.num_agents)]
        self.log_probs = [[] for _ in range(self.num_agents)]
        self.values = [[] for _ in range(self.num_agents)]
        self.global_states = []
        self._size = 0


# ---------------------------------------------------------------------------
# MAPPO agent
# ---------------------------------------------------------------------------

class MAPPOAgent(BaseAgent):
    """
    Multi-Agent PPO with centralized training, decentralized execution (CTDE).

    Each agent instance represents one agent in the population.
    During training, agents share a centralized critic that conditions on the
    global state. At execution time, only the decentralized actor is used.

    Features:
    - Decentralized GRU or MLP actor
    - Centralized value critic on global state
    - GAE advantage estimation
    - PPO clipping (clip_eps or dual-clip)
    - Entropy bonus
    - Value function normalization
    - Gradient clipping
    - Optional parameter sharing across agents
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        num_agents: int,
        hidden_dim: int = 256,
        critic_hidden_dim: int = 512,
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        dual_clip: Optional[float] = None,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 1.0,
        max_grad_norm: float = 10.0,
        ppo_epochs: int = 10,
        mini_batch_size: int = 256,
        normalize_advantages: bool = True,
        normalize_values: bool = True,
        use_recurrent: bool = False,
        use_attention_critic: bool = False,
        value_clip: bool = True,
        value_clip_eps: float = 0.2,
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

        self.state_dim = state_dim
        self.num_agents = num_agents
        self.clip_eps = clip_eps
        self.dual_clip = dual_clip
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.normalize_advantages = normalize_advantages
        self.normalize_values = normalize_values
        self.use_recurrent = use_recurrent
        self.value_clip = value_clip
        self.value_clip_eps = value_clip_eps

        # Decentralized actor
        if use_recurrent:
            self.encoder = RecurrentObservationEncoder(
                obs_dim=obs_dim, hidden_dim=hidden_dim
            ).to(self.device)
        else:
            self.encoder = ObservationEncoder(
                obs_dim=obs_dim, hidden_dim=hidden_dim, num_layers=3
            ).to(self.device)

        self.actor = GaussianActor(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            squash_output=False,  # PPO typically uses raw Gaussian
        ).to(self.device)

        # Centralized critic
        self.centralized_critic = CentralizedCritic(
            state_dim=state_dim,
            hidden_dim=critic_hidden_dim,
            num_layers=4,
            use_attention=use_attention_critic,
            num_agents=num_agents,
        ).to(self.device)

        # Optimizers
        actor_params = (
            list(self.encoder.parameters()) + list(self.actor.parameters())
        )
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=lr_actor, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(
            self.centralized_critic.parameters(), lr=lr_critic, eps=1e-5
        )

        # Learning rate schedulers
        self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.actor_optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.critic_optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000
        )

        # Value normalizer
        self.value_normalizer = ValueNormalizer()

        # Rollout buffer
        self.rollout_buffer = MARolloutBuffer(
            num_agents=num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
        )

        # Hidden state for recurrent mode
        self._hidden: Optional[torch.Tensor] = None

    def reset_hidden(self, batch_size: int = 1) -> None:
        if self.use_recurrent and isinstance(self.encoder, RecurrentObservationEncoder):
            self._hidden = self.encoder.init_hidden(batch_size, self.device)

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Decentralized action selection using local observation."""
        obs_t = self.to_tensor(obs).unsqueeze(0)

        with torch.no_grad():
            if self.use_recurrent and isinstance(self.encoder, RecurrentObservationEncoder):
                if self.hidden is None:
                    self.reset_hidden(1)
                enc, self._hidden = self.encoder(obs_t, self._hidden)
                enc = enc.squeeze(0)
            else:
                enc = self.encoder(obs_t).squeeze(0)

            action, log_prob = self.actor.get_action(enc, deterministic=deterministic)

        return (
            action.cpu().numpy(),
            float(log_prob.cpu().item()),
            0.0,  # value computed separately via get_value
        )

    def get_value(self, global_state: np.ndarray) -> float:
        """Compute value estimate from global state (centralized critic)."""
        state_t = self.to_tensor(global_state).unsqueeze(0)
        with torch.no_grad():
            value = self.centralized_critic(state_t)
        return float(value.cpu().item())

    def select_action_with_value(
        self,
        obs: np.ndarray,
        global_state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """Combined actor + critic forward pass."""
        obs_t = self.to_tensor(obs).unsqueeze(0)
        state_t = self.to_tensor(global_state).unsqueeze(0)

        with torch.no_grad():
            if self.use_recurrent and isinstance(self.encoder, RecurrentObservationEncoder):
                if self._hidden is None:
                    self.reset_hidden(1)
                enc, self._hidden = self.encoder(obs_t, self._hidden)
                enc = enc.squeeze(0)
            else:
                enc = self.encoder(obs_t).squeeze(0)

            action, log_prob = self.actor.get_action(enc, deterministic=deterministic)
            value = self.centralized_critic(state_t).squeeze()

        return (
            action.cpu().numpy(),
            float(log_prob.cpu().item()),
            float(value.cpu().item()),
        )

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        global_state: Optional[np.ndarray] = None,
    ) -> None:
        """Store single agent's transition in rollout buffer."""
        self.rollout_buffer.add(
            self.agent_id, obs, action, reward, done, log_prob, value, global_state
        )

    def update(self, batch: Optional[Dict] = None) -> Dict[str, float]:
        """
        MAPPO update.
        If batch is provided, uses that; otherwise uses rollout buffer.
        """
        if batch is not None:
            return self._update_from_batch(batch)
        return self._update_from_buffer()

    def _update_from_buffer(self) -> Dict[str, float]:
        if len(self.rollout_buffer) == 0:
            return {}

        # Compute last values for bootstrapping
        last_values = [0.0] * self.num_agents
        adv_list, ret_list = self.rollout_buffer.compute_advantages_and_returns(
            last_values=last_values,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Update value normalizer
        for ret in ret_list:
            self.value_normalizer.update(ret)

        # Get agent's data
        i = self.agent_id
        obs_np = np.array(self.rollout_buffer.obs[i], dtype=np.float32)
        acts_np = np.array(self.rollout_buffer.actions[i], dtype=np.float32)
        old_lp_np = np.array(self.rollout_buffer.log_probs[i], dtype=np.float32)
        adv_np = adv_list[i]
        ret_np = ret_list[i]
        old_val_np = np.array(self.rollout_buffer.values[i], dtype=np.float32)
        states_np = (
            np.array(self.rollout_buffer.global_states, dtype=np.float32)
            if self.rollout_buffer.global_states
            else None
        )

        obs_t = self.to_tensor(obs_np)
        acts_t = self.to_tensor(acts_np)
        old_lp_t = self.to_tensor(old_lp_np)
        adv_t = self.to_tensor(adv_np)
        ret_t = self.to_tensor(ret_np)
        old_val_t = self.to_tensor(old_val_np)

        if states_np is not None:
            states_t = self.to_tensor(states_np)
        else:
            states_t = obs_t  # fallback

        # Normalize advantages
        if self.normalize_advantages:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + EPS)

        # Normalize returns for critic
        ret_norm_t = self.value_normalizer.normalize(ret_t)

        n = obs_t.shape[0]
        total_a_loss = 0.0
        total_c_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0
        early_stop = False

        for epoch in range(self.ppo_epochs):
            if early_stop:
                break
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.mini_batch_size):
                idx = indices[start:start + self.mini_batch_size]
                mb_obs = obs_t[idx]
                mb_acts = acts_t[idx]
                mb_old_lp = old_lp_t[idx]
                mb_adv = adv_t[idx]
                mb_ret = ret_norm_t[idx]
                mb_old_val = old_val_t[idx]
                mb_states = states_t[idx] if states_t.shape[0] > idx.max() else states_t

                # Actor forward
                enc = self.encoder(mb_obs)
                new_lp, entropy = self.actor.evaluate_actions(enc, mb_acts)
                new_lp = new_lp.squeeze(-1)
                entropy = entropy.squeeze(-1)

                # PPO ratio
                ratio = torch.exp(new_lp - mb_old_lp)

                # Approximate KL for early stopping
                approx_kl = float(((ratio - 1) - torch.log(ratio)).mean().item())
                if approx_kl > 0.015 * 1.5:
                    early_stop = True
                    break

                # Policy loss (PPO clip or dual-clip)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                if self.dual_clip is not None:
                    # Dual clip: clip3 = c * advantage when advantage < 0
                    surr3 = self.dual_clip * mb_adv
                    actor_loss = -torch.mean(
                        torch.max(torch.min(surr1, surr2), surr3)
                    )
                else:
                    actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Critic forward
                val_pred = self.centralized_critic(mb_states)
                val_pred = val_pred.squeeze(-1)

                # Value clipping
                if self.value_clip:
                    val_pred_clipped = mb_old_val + torch.clamp(
                        val_pred - mb_old_val, -self.value_clip_eps, self.value_clip_eps
                    )
                    v_loss1 = F.mse_loss(val_pred, mb_ret, reduction="none")
                    v_loss2 = F.mse_loss(val_pred_clipped, mb_ret, reduction="none")
                    value_loss = torch.mean(torch.max(v_loss1, v_loss2))
                else:
                    value_loss = F.mse_loss(val_pred, mb_ret)

                # Total loss
                total_loss = (
                    actor_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.actor.parameters()),
                    self.max_grad_norm,
                )
                nn.utils.clip_grad_norm_(
                    self.centralized_critic.parameters(),
                    self.max_grad_norm,
                )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_a_loss += float(actor_loss.item())
                total_c_loss += float(value_loss.item())
                total_entropy += float(-entropy_loss.item())
                total_kl += approx_kl
                n_updates += 1

        self.rollout_buffer.clear()
        self._update_count += 1
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        if n_updates == 0:
            return {}

        metrics = {
            "actor_loss": total_a_loss / n_updates,
            "critic_loss": total_c_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": total_kl / n_updates,
            "early_stop": float(early_stop),
        }
        for k, v in metrics.items():
            self.log_metric(k, v)
        return metrics

    def _update_from_batch(self, batch: Dict) -> Dict[str, float]:
        """Update from externally provided batch dict."""
        obs_t = self.to_tensor(batch["obs"])
        acts_t = self.to_tensor(batch["actions"])
        old_lp_t = self.to_tensor(batch["log_probs"])
        adv_t = self.to_tensor(batch["advantages"])
        ret_t = self.to_tensor(batch["returns"])
        states_t = self.to_tensor(batch.get("global_states", batch["obs"]))
        old_val_t = self.to_tensor(batch.get("old_values", np.zeros(obs_t.shape[0])))

        if self.normalize_advantages:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + EPS)

        ret_norm_t = self.value_normalizer.normalize(ret_t)

        n = obs_t.shape[0]
        metrics_acc: Dict[str, float] = collections.defaultdict(float)
        n_updates = 0

        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.mini_batch_size):
                idx = indices[start:start + self.mini_batch_size]

                enc = self.encoder(obs_t[idx])
                new_lp, entropy = self.actor.evaluate_actions(enc, acts_t[idx])
                new_lp = new_lp.squeeze(-1)
                entropy_mean = entropy.squeeze(-1).mean()

                ratio = torch.exp(new_lp - old_lp_t[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                val_pred = self.centralized_critic(states_t[idx]).squeeze(-1)
                value_loss = F.mse_loss(val_pred, ret_norm_t[idx])

                total_loss = (
                    actor_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_mean
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.actor.parameters()),
                    self.max_grad_norm,
                )
                nn.utils.clip_grad_norm_(
                    self.centralized_critic.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                metrics_acc["actor_loss"] += float(actor_loss.item())
                metrics_acc["critic_loss"] += float(value_loss.item())
                metrics_acc["entropy"] += float(entropy_mean.item())
                n_updates += 1

        self._update_count += 1
        if n_updates == 0:
            return {}
        metrics = {k: v / n_updates for k, v in metrics_acc.items()}
        for k, v in metrics.items():
            self.log_metric(k, v)
        return metrics

    def save(self, path: str) -> None:
        """Save MAPPO-specific state."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "agent_id": self.agent_id,
            "update_count": self._update_count,
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "centralized_critic": self.centralized_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "value_normalizer_mean": self.value_normalizer.mean,
            "value_normalizer_var": self.value_normalizer.var,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(state["encoder"])
        self.actor.load_state_dict(state["actor"])
        self.centralized_critic.load_state_dict(state["centralized_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.value_normalizer.mean = state.get("value_normalizer_mean", 0.0)
        self.value_normalizer.var = state.get("value_normalizer_var", 1.0)
        self._update_count = state.get("update_count", 0)
        logger.info(f"MAPPO agent {self.agent_id} loaded from {path}")


# ---------------------------------------------------------------------------
# MAPPO population: shared parameters
# ---------------------------------------------------------------------------

class MAPPOPopulation:
    """
    Manages a population of MAPPO agents with optional parameter sharing.

    In parameter-sharing mode, all agents share the same actor and encoder weights,
    but receive their own agent_id as part of the observation.
    The centralized critic is always shared.
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        share_parameters: bool = True,
        agent_kwargs: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        self.num_agents = num_agents
        self.share_parameters = share_parameters
        kwargs = agent_kwargs or {}

        self.agents = [
            MAPPOAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                state_dim=state_dim,
                num_agents=num_agents,
                device=device,
                **kwargs,
            )
            for i in range(num_agents)
        ]

        if share_parameters and num_agents > 1:
            self._share_parameters()

    def _share_parameters(self) -> None:
        """Make all agents share encoder and actor parameters."""
        ref = self.agents[0]
        for agent in self.agents[1:]:
            # Share encoder
            agent.encoder.load_state_dict(ref.encoder.state_dict())
            for p_ref, p_ag in zip(ref.encoder.parameters(), agent.encoder.parameters()):
                p_ag.data = p_ref.data

            # Share actor
            agent.actor.load_state_dict(ref.actor.state_dict())
            for p_ref, p_ag in zip(ref.actor.parameters(), agent.actor.parameters()):
                p_ag.data = p_ref.data

            # Share centralized critic
            agent.centralized_critic.load_state_dict(ref.centralized_critic.state_dict())
            for p_ref, p_ag in zip(
                ref.centralized_critic.parameters(),
                agent.centralized_critic.parameters()
            ):
                p_ag.data = p_ref.data

    def select_actions(
        self,
        obs_list: List[np.ndarray],
        global_state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[List[np.ndarray], List[float], List[float]]:
        actions, log_probs, values = [], [], []
        for i, agent in enumerate(self.agents):
            a, lp, v = agent.select_action_with_value(
                obs_list[i], global_state, deterministic=deterministic
            )
            actions.append(a)
            log_probs.append(lp)
            values.append(v)
        return actions, log_probs, values

    def update_all(self) -> List[Dict[str, float]]:
        """Update all agents and return list of metric dicts."""
        return [agent.update() for agent in self.agents]

    def save_all(self, directory: str) -> None:
        import os
        os.makedirs(directory, exist_ok=True)
        for agent in self.agents:
            agent.save(os.path.join(directory, f"agent_{agent.agent_id}.pt"))

    def load_all(self, directory: str) -> None:
        import os
        for agent in self.agents:
            path = os.path.join(directory, f"agent_{agent.agent_id}.pt")
            if os.path.exists(path):
                agent.load(path)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "CentralizedCritic",
    "ValueNormalizer",
    "MARolloutBuffer",
    "MAPPOAgent",
    "MAPPOPopulation",
]
