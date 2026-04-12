"""
MarketMakerAgent — RL liquidity provider.

Specialization:
  - Action: quote spread (half-spread in ticks) + quantity to quote
  - Reward: spread captured on fills - inventory risk - adverse selection cost
  - Baseline: Avellaneda-Stoikov analytical solution
  - RL improves on AS by learning from actual fill patterns

Avellaneda-Stoikov model:
  Optimal mid-quote reservation price:
    r(s, q, t) = s - q * γ * σ² * (T - t)
  Optimal spread:
    δ* = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/κ)
  where:
    s = current mid price
    q = inventory
    γ = risk aversion
    σ = volatility
    T = time horizon
    κ = order arrival intensity
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from hyper_agent.agents._base_compat import BaseAgent, Memory, StandardNorm, compute_gae


# ============================================================
# Avellaneda-Stoikov baseline policy
# ============================================================

class AvellanedaStoikov:
    """
    Analytical Avellaneda-Stoikov market making policy.

    Provides baseline spread and reservation price that RL can improve upon.
    """

    def __init__(
        self,
        risk_aversion:   float = 0.1,
        order_intensity: float = 1.5,
        vol_window:      int   = 20,
    ) -> None:
        self.gamma   = risk_aversion
        self.kappa   = order_intensity
        self.vol_window = vol_window
        self._price_history: List[float] = []

    def update_price(self, price: float) -> None:
        self._price_history.append(price)
        if len(self._price_history) > self.vol_window * 2:
            self._price_history.pop(0)

    def estimate_vol(self) -> float:
        if len(self._price_history) < 2:
            return 0.01
        returns = np.diff(np.log(np.array(self._price_history[-self.vol_window:]) + 1e-8))
        return float(np.std(returns)) + 1e-6

    def compute_spread_and_reservation(
        self,
        mid_price: float,
        inventory: float,
        time_remaining: float,  # fraction [0, 1]
    ) -> Tuple[float, float]:
        """
        Returns (optimal_half_spread, reservation_price).
        """
        sigma = self.estimate_vol()
        T     = max(time_remaining, 0.01)

        # Reservation price (skew mid away from inventory)
        reservation = mid_price - inventory * self.gamma * sigma**2 * T

        # Optimal half-spread
        part1 = self.gamma * sigma**2 * T
        part2 = (2.0 / self.gamma) * math.log(1.0 + self.gamma / max(self.kappa, 1e-6))
        half_spread = (part1 + part2) / 2.0

        # Clip to reasonable range
        half_spread = np.clip(half_spread, 0.01, 0.5)
        return float(half_spread), float(reservation)


# ============================================================
# Market Maker Actor-Critic Network
# ============================================================

class MMActorNet(nn.Module):
    """
    Actor for market maker: outputs (half-spread, quote-size).

    Both outputs are continuous and passed through appropriate activations.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        # Half-spread head: output in (0, 1] ticks
        self.spread_mu    = nn.Linear(hidden_dim // 2, 1)
        self.spread_log_std = nn.Parameter(torch.zeros(1))

        # Quote size head: Beta(α, β) on [0, 1]
        self.size_alpha = nn.Linear(hidden_dim // 2, 1)
        self.size_beta  = nn.Linear(hidden_dim // 2, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (spread_mu, spread_std, size_alpha, size_beta).
        """
        h       = self.backbone(obs)
        s_mu    = F.softplus(self.spread_mu(h)) + 0.001
        s_std   = self.spread_log_std.exp().expand_as(s_mu).clamp(0.001, 0.5)
        s_alpha = F.softplus(self.size_alpha(h)) + 1e-3
        s_beta  = F.softplus(self.size_beta(h))  + 1e-3
        return s_mu, s_std, s_alpha, s_beta

    def sample(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (half_spread, quote_size, log_prob).
        """
        from torch.distributions import Normal, Beta
        s_mu, s_std, s_alpha, s_beta = self.forward(obs)
        spread_dist = Normal(s_mu.squeeze(-1), s_std.squeeze(-1))
        size_dist   = Beta(s_alpha.squeeze(-1), s_beta.squeeze(-1))

        spread   = spread_dist.sample().clamp(0.001, 2.0)
        size     = size_dist.sample()
        log_prob = spread_dist.log_prob(spread) + size_dist.log_prob(
            size.clamp(1e-6, 1 - 1e-6)
        )
        return spread, size, log_prob

    def evaluate(
        self,
        obs:     torch.Tensor,
        spreads: torch.Tensor,
        sizes:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from torch.distributions import Normal, Beta
        s_mu, s_std, s_alpha, s_beta = self.forward(obs)
        spread_dist = Normal(s_mu.squeeze(-1), s_std.squeeze(-1))
        size_dist   = Beta(s_alpha.squeeze(-1), s_beta.squeeze(-1))
        log_prob = spread_dist.log_prob(spreads) + size_dist.log_prob(
            sizes.clamp(1e-6, 1 - 1e-6)
        )
        entropy  = spread_dist.entropy() + size_dist.entropy()
        return log_prob, entropy


class MMCriticNet(nn.Module):
    """Value network for market maker critic."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ============================================================
# Reward decomposition tracker
# ============================================================

class MMRewardTracker:
    """
    Tracks and decomposes market maker reward components.

    Components:
      - spread_income:      half-spread × filled_quantity
      - inventory_penalty:  -γ_inv × position²
      - adverse_selection:  -AS cost when fills move against position
    """

    def __init__(
        self,
        inv_penalty_coef: float = 0.01,
        as_window:        int   = 5,
    ) -> None:
        self.inv_penalty_coef = inv_penalty_coef
        self.as_window        = as_window
        self._fill_history:   List[Tuple[int, float]] = []  # (direction, fill_price)
        self._price_at_fill:  List[float] = []

    def compute_reward(
        self,
        half_spread:    float,
        filled_qty:     float,
        position:       float,
        current_price:  float,
        fill_direction: int,   # +1 if we sold (received long flow), -1 if we bought
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute decomposed MM reward.

        Args:
            half_spread:    quoted half-spread
            filled_qty:     volume filled this step
            position:       current inventory after fill
            current_price:  current mid price
            fill_direction: direction of incoming order flow

        Returns:
            (total_reward, components_dict)
        """
        # Spread income: we capture half-spread on each fill
        spread_income = half_spread * abs(filled_qty)

        # Inventory penalty: quadratic cost of holding inventory
        inv_penalty = -self.inv_penalty_coef * position**2

        # Adverse selection: track fills and check if price moved adversely
        as_cost = self._compute_adverse_selection(
            fill_direction, filled_qty, current_price
        )

        total = spread_income + inv_penalty - as_cost

        components = {
            "spread_income":  spread_income,
            "inv_penalty":    inv_penalty,
            "adverse_selection": -as_cost,
            "total":          total,
        }
        return total, components

    def _compute_adverse_selection(
        self, direction: int, qty: float, current_price: float
    ) -> float:
        """
        Estimate adverse selection cost.

        When we fill a buy order at price p, if price subsequently rises,
        the order was informed and we lost money. Measure as:
        AS_cost = Σ_{fills in window} max(0, direction_k × (current - fill_price_k)) × qty_k
        """
        if qty > 0:
            self._fill_history.append((direction, current_price))
            self._price_at_fill.append(current_price)

        if len(self._fill_history) > self.as_window:
            self._fill_history.pop(0)
            self._price_at_fill.pop(0)

        as_cost = 0.0
        for i, (d, fill_p) in enumerate(self._fill_history[:-1]):
            price_move = (current_price - fill_p) * d
            as_cost   += max(0.0, price_move) * abs(qty)
        return as_cost * 0.1  # scale factor


# ============================================================
# MarketMakerAgent
# ============================================================

class MarketMakerAgent(BaseAgent):
    """
    RL Market Maker agent that learns to provide liquidity.

    Uses PPO with continuous action space (spread, size).
    Reward = spread income - inventory penalty - adverse selection.
    Compares against Avellaneda-Stoikov analytical baseline.
    """

    AGENT_TYPE = "market_maker"

    def __init__(
        self,
        agent_id:         str,
        obs_dim:          int,
        hidden_dim:       int   = 128,
        lr:               float = 1e-4,
        gamma:            float = 0.99,
        lam:              float = 0.95,
        clip_eps:         float = 0.2,
        entropy_coef:     float = 0.005,
        vf_coef:          float = 0.5,
        max_grad_norm:    float = 1.0,
        inv_penalty_coef: float = 0.01,
        n_epochs:         int   = 4,
        minibatch_size:   int   = 64,
        rollout_len:      int   = 256,
        use_as_baseline:  bool  = True,
        device:           str   = "cpu",
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
        self.clip_eps       = clip_eps
        self.entropy_coef   = entropy_coef
        self.vf_coef        = vf_coef
        self.max_grad_norm  = max_grad_norm
        self.n_epochs       = n_epochs
        self.minibatch_size = minibatch_size
        self.rollout_len    = rollout_len
        self.use_as_baseline = use_as_baseline

        # Networks
        self.actor  = MMActorNet(obs_dim, hidden_dim).to(self.device)
        self.critic = MMCriticNet(obs_dim, hidden_dim).to(self.device)

        # Optimizers
        self.optimizer = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr, eps=1e-5,
        )

        # Avellaneda-Stoikov baseline
        self.as_baseline    = AvellanedaStoikov()
        self.reward_tracker = MMRewardTracker(inv_penalty_coef)

        # Rollout storage
        self._rollout: Dict[str, List] = {
            k: [] for k in [
                "obs", "spreads", "sizes", "log_probs",
                "rewards", "values", "dones",
            ]
        }
        self._rollout_count = 0

        # Performance tracking
        self.fill_rate_history: List[float] = []
        self.spread_history:    List[float] = []
        self.inv_history:       List[float] = []
        self.reward_components: List[Dict]  = []

        # AS baseline tracking
        self.as_spread_history: List[float] = []
        self.rl_spread_history: List[float] = []

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
        """
        Returns action as (logit_neg, logit_zero, logit_pos, size).

        For market makers, the "direction" encodes the spread-side:
          - logit_neg ~ quoted half-spread (used as proxy for bid side)
          - logit_pos ~ quote size
          - size      ~ normalized quote size

        The trainer interprets the market maker action differently from
        directional traders.
        """
        self.obs_norm.update(obs)
        norm_obs = self.obs_norm.normalize(obs)
        obs_t    = torch.FloatTensor(norm_obs).unsqueeze(0).to(self.device)

        spread, size, log_prob = self.actor.sample(obs_t)
        value = self.critic(obs_t).item()

        hs  = spread.item()
        sz  = size.item()
        lp  = log_prob.item()

        # Store for rollout
        self._rollout["obs"].append(norm_obs)
        self._rollout["spreads"].append(hs)
        self._rollout["sizes"].append(sz)
        self._rollout["log_probs"].append(lp)
        self._rollout["values"].append(value)
        self._rollout_count += 1

        # Log spread
        self.spread_history.append(hs)
        if self.use_as_baseline:
            # Get AS reference for comparison
            self.as_spread_history.append(hs)
            self.rl_spread_history.append(hs)

        # Build action array (direction logits encode spread info)
        action = np.array([
            -hs,      # bid side deviation
             0.0,     # flat
             hs,      # ask side deviation
             sz,      # size
        ], dtype=np.float32)
        return action, lp, value

    def receive_reward(
        self,
        reward:         float,
        done:           bool,
        half_spread:    float,
        filled_qty:     float,
        position:       float,
        current_price:  float,
        fill_direction: int = 0,
    ) -> float:
        """
        Compute shaped reward and store in rollout.

        Returns the shaped reward (may differ from raw env reward).
        """
        shaped_reward, components = self.reward_tracker.compute_reward(
            half_spread, filled_qty, position, current_price, fill_direction
        )
        # Blend environment reward with shaped reward
        total = 0.5 * reward + 0.5 * shaped_reward

        self._rollout["rewards"].append(total)
        self._rollout["dones"].append(float(done))
        self.reward_components.append(components)
        self.inv_history.append(abs(position))

        return total

    def update(self, batch: Optional[Dict] = None) -> Dict[str, float]:
        """PPO update on collected rollout."""
        if len(self._rollout["rewards"]) < self.minibatch_size:
            return {}

        n = len(self._rollout["rewards"])
        obs_arr    = np.array(self._rollout["obs"][:n],      dtype=np.float32)
        spreads    = np.array(self._rollout["spreads"][:n],  dtype=np.float32)
        sizes      = np.array(self._rollout["sizes"][:n],    dtype=np.float32)
        log_probs  = np.array(self._rollout["log_probs"][:n],dtype=np.float32)
        rewards    = np.array(self._rollout["rewards"][:n],  dtype=np.float32)
        values     = np.array(self._rollout["values"][:n],   dtype=np.float32)
        dones      = np.array(self._rollout["dones"][:n],    dtype=np.float32)

        # GAE
        advantages, returns = compute_gae(rewards, values, dones, gamma=self.gamma, lam=self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        stats_list: Dict[str, List[float]] = {
            "pol_loss": [], "val_loss": [], "entropy": []
        }
        for _ in range(self.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                b = idx[start: start + self.minibatch_size]
                obs_t  = torch.FloatTensor(obs_arr[b]).to(self.device)
                sp_t   = torch.FloatTensor(spreads[b]).to(self.device)
                sz_t   = torch.FloatTensor(sizes[b]).to(self.device)
                olp_t  = torch.FloatTensor(log_probs[b]).to(self.device)
                adv_t  = torch.FloatTensor(advantages[b]).to(self.device)
                ret_t  = torch.FloatTensor(returns[b]).to(self.device)

                new_lp, entropy = self.actor.evaluate(obs_t, sp_t, sz_t)
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

        # Clear rollout
        for k in self._rollout:
            self._rollout[k].clear()
        self._rollout_count = 0

        return {k: float(np.mean(v)) for k, v in stats_list.items()}

    # ------------------------------------------------------------------
    # AS Baseline Comparison
    # ------------------------------------------------------------------

    def get_as_spread(
        self, mid_price: float, inventory: float, time_remaining: float
    ) -> Tuple[float, float]:
        self.as_baseline.update_price(mid_price)
        return self.as_baseline.compute_spread_and_reservation(
            mid_price, inventory, time_remaining
        )

    def rl_vs_as_performance(self) -> Dict[str, float]:
        """Compare RL agent's spread with AS baseline."""
        if not self.rl_spread_history or not self.as_spread_history:
            return {}
        return {
            "rl_mean_spread":   float(np.mean(self.rl_spread_history[-100:])),
            "as_mean_spread":   float(np.mean(self.as_spread_history[-100:])),
            "mean_inventory":   float(np.mean(self.inv_history[-100:])),
            "fill_rate":        float(np.mean(self.fill_rate_history[-100:]))
                                if self.fill_rate_history else 0.0,
        }
