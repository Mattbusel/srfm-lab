"""
MomentumAgent — Trend-following RL agent.

Features:
  - EWMA signal with learned decay parameter
  - Risk-adjusted PnL reward (Sharpe-like)
  - Position limits enforced by environment
  - Continuous action: target position in [-1, +1]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

from hyper_agent.agents._base_compat import BaseAgent, compute_gae, StandardNorm


# ============================================================
# Learned EWMA module
# ============================================================

class LearnedEWMA(nn.Module):
    """
    Differentiable EWMA signal with a learnable decay parameter per window.

    Maintains multiple EWMAs at different speeds, outputs them concatenated.
    Useful as a feature extractor for momentum agents.
    """

    def __init__(self, n_windows: int = 4, init_halflife: float = 10.0) -> None:
        super().__init__()
        self.n_windows = n_windows
        # Log-half-lives (so decay ∈ (0, 1) always)
        log_hl = math.log(init_halflife) * torch.ones(n_windows)
        # Stagger initial half-lives: 5, 10, 20, 40 bars
        for i in range(n_windows):
            log_hl[i] = math.log(5.0 * (2**i))
        self.log_halflife = nn.Parameter(log_hl)

    def decays(self) -> torch.Tensor:
        """Returns decay factors α_i = 2^{-1/hl_i}."""
        hl = self.log_halflife.exp().clamp(1.0, 200.0)
        return torch.pow(torch.tensor(2.0), -1.0 / hl)

    def forward(self, price_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            price_history: (T,) or (batch, T) price sequence

        Returns:
            ewma_signals: (n_windows,) or (batch, n_windows)
        """
        decays = self.decays()
        if price_history.dim() == 1:
            price_history = price_history.unsqueeze(0)

        T = price_history.shape[-1]
        result = []
        for i in range(self.n_windows):
            a = decays[i].item()
            ewma = price_history[:, 0]
            for t in range(1, T):
                ewma = a * ewma + (1.0 - a) * price_history[:, t]
            # Signal: EWMA relative deviation from last price
            signal = (ewma - price_history[:, -1]) / (price_history[:, -1].abs() + 1e-8)
            result.append(signal)
        return torch.stack(result, dim=-1)


import math


# ============================================================
# Momentum Actor network
# ============================================================

class MomentumActorNet(nn.Module):
    """
    Actor for momentum agent.

    Takes price history features + position → target position in [-1, +1].
    Uses Normal distribution for exploration.
    """

    def __init__(
        self,
        obs_dim:    int,
        hidden_dim: int = 64,
        n_ewma:     int = 4,
    ) -> None:
        super().__init__()
        self.n_ewma = n_ewma
        # EWMA feature extractor
        self.ewma = LearnedEWMA(n_ewma)

        input_dim = obs_dim + n_ewma
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Position target head (tanh → [-1, 1])
        self.mean_head   = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))  # learnable std

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.fc1, self.fc2, self.mean_head]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)

    def forward(
        self, obs: torch.Tensor, price_hist: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs:        (batch, obs_dim)
            price_hist: (batch, T) optional price history for EWMA signals

        Returns:
            (mean, std) of Normal distribution over target position
        """
        if price_hist is not None and price_hist.shape[-1] > 1:
            ewma_feats = self.ewma(price_hist)
            x = torch.cat([obs, ewma_feats], dim=-1)
        else:
            pad = torch.zeros(obs.shape[0], self.n_ewma, device=obs.device)
            x   = torch.cat([obs, pad], dim=-1)

        x   = F.relu(self.ln1(self.fc1(x)))
        x   = F.relu(self.ln2(self.fc2(x)))
        mu  = torch.tanh(self.mean_head(x)).squeeze(-1)  # in (-1, 1)
        std = self.log_std.exp().expand_as(mu).clamp(0.01, 0.5)
        return mu, std

    def get_dist(
        self, obs: torch.Tensor, price_hist: Optional[torch.Tensor] = None
    ) -> Normal:
        mu, std = self.forward(obs, price_hist)
        return Normal(mu, std)

    def sample(
        self, obs: torch.Tensor, price_hist: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs, price_hist)
        action = dist.rsample().clamp(-1.0, 1.0)
        return action, dist.log_prob(action)


# ============================================================
# Running Sharpe tracker
# ============================================================

class SharpeTracker:
    """
    Online computation of annualized Sharpe ratio.

    Used to compute risk-adjusted reward for momentum agent.
    """

    def __init__(self, window: int = 100, min_periods: int = 10) -> None:
        self.window      = window
        self.min_periods = min_periods
        self._returns: List[float] = []

    def update(self, ret: float) -> None:
        self._returns.append(ret)
        if len(self._returns) > self.window:
            self._returns.pop(0)

    def sharpe(self) -> float:
        if len(self._returns) < self.min_periods:
            return 0.0
        arr = np.array(self._returns)
        std = arr.std()
        if std < 1e-8:
            return 0.0
        return float(arr.mean() / std) * math.sqrt(252)  # annualize

    def sortino(self) -> float:
        """Sortino ratio: penalizes only downside volatility."""
        if len(self._returns) < self.min_periods:
            return 0.0
        arr = np.array(self._returns)
        downside = arr[arr < 0.0]
        if len(downside) < 2:
            return float(arr.mean()) / 1e-8
        down_std = downside.std()
        if down_std < 1e-8:
            return 0.0
        return float(arr.mean() / down_std) * math.sqrt(252)


# ============================================================
# MomentumAgent
# ============================================================

class MomentumAgent(BaseAgent):
    """
    Trend-following agent that learns to size positions based on momentum signals.

    Reward: risk-adjusted PnL using running Sharpe-like metric.
    """

    AGENT_TYPE = "momentum"

    def __init__(
        self,
        agent_id:       str,
        obs_dim:        int,
        hidden_dim:     int   = 64,
        lr:             float = 3e-4,
        gamma:          float = 0.99,
        lam:            float = 0.95,
        clip_eps:       float = 0.2,
        entropy_coef:   float = 0.005,
        vf_coef:        float = 0.5,
        max_grad_norm:  float = 0.5,
        n_ewma:         int   = 4,
        sharpe_window:  int   = 100,
        sharpe_weight:  float = 0.3,
        n_epochs:       int   = 4,
        minibatch_size: int   = 64,
        rollout_len:    int   = 256,
        device:         str   = "cpu",
    ) -> None:
        super().__init__(
            agent_id   = agent_id,
            obs_dim    = obs_dim,
            hidden_dim = hidden_dim,
            lr         = lr,
            gamma      = gamma,
            lam        = lam,
            device     = device,
        )
        self.clip_eps      = clip_eps
        self.entropy_coef  = entropy_coef
        self.vf_coef       = vf_coef
        self.max_grad_norm = max_grad_norm
        self.sharpe_weight = sharpe_weight
        self.n_epochs      = n_epochs
        self.minibatch_size = minibatch_size
        self.rollout_len    = rollout_len

        # Networks
        self.actor  = MomentumActorNet(obs_dim, hidden_dim, n_ewma).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        ).to(self.device)

        self.optimizer = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr, eps=1e-5,
        )

        self.sharpe_tracker = SharpeTracker(sharpe_window)

        # Rollout storage
        self._rollout: Dict[str, List] = {
            k: [] for k in ["obs", "actions", "log_probs", "rewards", "values", "dones"]
        }
        self._price_hist_buf: List[float] = []
        self.position_history: List[float] = []

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> str:
        return self.AGENT_TYPE

    @torch.no_grad()
    def act(
        self,
        obs:           np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        self.obs_norm.update(obs)
        norm_obs = self.obs_norm.normalize(obs)
        obs_t    = torch.FloatTensor(norm_obs).unsqueeze(0).to(self.device)

        # Build price history tensor if available
        if len(self._price_hist_buf) > 1:
            ph_arr = np.array(self._price_hist_buf[-32:], dtype=np.float32)
            ph_t   = torch.FloatTensor(ph_arr).unsqueeze(0).to(self.device)
        else:
            ph_t = None

        dist = self.actor.get_dist(obs_t, ph_t)
        if deterministic:
            action = dist.mean.clamp(-1.0, 1.0)
        else:
            action = dist.rsample().clamp(-1.0, 1.0)

        log_prob = dist.log_prob(action).item()
        value    = self.critic(obs_t).item()
        act_val  = action.item()

        # Map target position [-1, 1] to action array format
        # Positive → long (index 2), Negative → short (index 0)
        if act_val > 0.1:
            logits = np.array([-5.0, -1.0, 5.0 * act_val], dtype=np.float32)
        elif act_val < -0.1:
            logits = np.array([5.0 * abs(act_val), -1.0, -5.0], dtype=np.float32)
        else:
            logits = np.array([-1.0, 5.0, -1.0], dtype=np.float32)
        size = float(min(abs(act_val), 1.0))
        action_arr = np.array([*logits, size], dtype=np.float32)

        # Store for rollout
        self._rollout["obs"].append(norm_obs)
        self._rollout["actions"].append(act_val)
        self._rollout["log_probs"].append(log_prob)
        self._rollout["values"].append(value)
        self.position_history.append(act_val)

        return action_arr, log_prob, value

    def observe_price(self, price: float) -> None:
        """Call each step to update internal price history for EWMA."""
        self._price_hist_buf.append(price)
        if len(self._price_hist_buf) > 200:
            self._price_hist_buf.pop(0)

    def receive_reward(self, raw_reward: float, done: bool) -> float:
        """
        Shape reward with running Sharpe component.

        Shaped = (1 - w) * raw + w * sharpe_signal
        """
        self.sharpe_tracker.update(raw_reward)
        sharpe    = self.sharpe_tracker.sharpe()
        shaped    = (1.0 - self.sharpe_weight) * raw_reward + self.sharpe_weight * sharpe * 0.01

        self._rollout["rewards"].append(shaped)
        self._rollout["dones"].append(float(done))
        return shaped

    def update(self, batch: Optional[Dict] = None) -> Dict[str, float]:
        n = len(self._rollout["rewards"])
        if n < self.minibatch_size:
            return {}

        obs_arr   = np.array(self._rollout["obs"][:n],       dtype=np.float32)
        acts      = np.array(self._rollout["actions"][:n],   dtype=np.float32)
        log_probs = np.array(self._rollout["log_probs"][:n], dtype=np.float32)
        rewards   = np.array(self._rollout["rewards"][:n],   dtype=np.float32)
        values    = np.array(self._rollout["values"][:n],    dtype=np.float32)
        dones     = np.array(self._rollout["dones"][:n],     dtype=np.float32)

        advantages, returns = compute_gae(rewards, values, dones, gamma=self.gamma, lam=self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats_list: Dict[str, List[float]] = {"pol_loss": [], "val_loss": [], "entropy": []}

        for _ in range(self.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                b    = idx[start: start + self.minibatch_size]
                obs_t  = torch.FloatTensor(obs_arr[b]).to(self.device)
                act_t  = torch.FloatTensor(acts[b]).to(self.device)
                olp_t  = torch.FloatTensor(log_probs[b]).to(self.device)
                adv_t  = torch.FloatTensor(advantages[b]).to(self.device)
                ret_t  = torch.FloatTensor(returns[b]).to(self.device)

                dist     = self.actor.get_dist(obs_t)
                new_lp   = dist.log_prob(act_t.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
                entropy  = dist.entropy()

                ratio    = (new_lp - olp_t).exp()
                surr1    = ratio * adv_t
                surr2    = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t
                pol_loss = -torch.min(surr1, surr2).mean()

                val      = self.critic(obs_t).squeeze(-1)
                val_loss = F.mse_loss(val, ret_t)
                loss     = pol_loss + self.vf_coef * val_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                stats_list["pol_loss"].append(pol_loss.item())
                stats_list["val_loss"].append(val_loss.item())
                stats_list["entropy"].append(entropy.mean().item())

        for k in self._rollout:
            self._rollout[k].clear()

        return {k: float(np.mean(v)) for k, v in stats_list.items()}

    def get_ewma_signals(self) -> Optional[np.ndarray]:
        """Return current EWMA momentum signals."""
        if len(self._price_hist_buf) < 2:
            return None
        ph  = np.array(self._price_hist_buf[-32:], dtype=np.float32)
        ph_t = torch.FloatTensor(ph).unsqueeze(0)
        with torch.no_grad():
            sigs = self.actor.ewma(ph_t).squeeze(0).numpy()
        return sigs

    def running_sharpe(self) -> float:
        return self.sharpe_tracker.sharpe()
