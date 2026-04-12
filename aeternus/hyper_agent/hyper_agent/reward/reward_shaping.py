"""
Reward shaping functions for multi-agent financial market.

Components:
  - IndividualReward:     PnL-based per-agent reward
  - TeamReward:           shared reward for same-type agent teams
  - MarketQualityReward:  bonus for improving bid-ask spread / reducing volatility
  - AdversarialPenalty:   penalize market manipulation
  - PotentialBasedShaping: potential-function shaping that preserves Nash eq
  - CuriosityBonus:       RND-based intrinsic exploration reward
  - CompositeReward:      weighted combination of all components
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ============================================================
# Individual Reward
# ============================================================

class IndividualReward:
    """
    PnL-based reward for individual agents.

    Options:
      - raw PnL
      - log PnL (dampens large swings)
      - clipped PnL (prevents reward hacking)
      - risk-adjusted PnL (divide by rolling vol)
    """

    def __init__(
        self,
        mode:          str   = "raw",      # raw | log | clipped | risk_adj
        clip_range:    float = 5.0,
        vol_window:    int   = 50,
        min_vol:       float = 1e-4,
    ) -> None:
        self.mode       = mode
        self.clip_range = clip_range
        self.vol_window = vol_window
        self.min_vol    = min_vol
        self._pnl_history: deque = deque(maxlen=vol_window)

    def compute(self, pnl: float) -> float:
        self._pnl_history.append(pnl)

        if self.mode == "raw":
            return pnl

        elif self.mode == "log":
            sign = 1.0 if pnl >= 0 else -1.0
            return sign * math.log1p(abs(pnl))

        elif self.mode == "clipped":
            return float(np.clip(pnl, -self.clip_range, self.clip_range))

        elif self.mode == "risk_adj":
            if len(self._pnl_history) < 5:
                return pnl
            arr = np.array(list(self._pnl_history))
            vol = float(arr.std()) + self.min_vol
            return pnl / vol

        return pnl

    def reset(self) -> None:
        self._pnl_history.clear()


# ============================================================
# Team Reward
# ============================================================

class TeamReward:
    """
    Shared reward for a group of agents.

    Blends individual and team reward:
      R_i = (1 - alpha) * R_individual_i + alpha * R_team

    Teams are defined by agent type prefix (e.g., all "mm_*" agents).
    """

    def __init__(
        self,
        team_definitions: Dict[str, List[str]],  # team_name → list of agent_ids
        blend_alpha:      float = 0.3,
    ) -> None:
        self.teams       = team_definitions
        self.blend_alpha = blend_alpha
        # Reverse map: agent_id → team_name
        self._agent_team: Dict[str, str] = {}
        for team_name, members in team_definitions.items():
            for aid in members:
                self._agent_team[aid] = team_name

    def compute(
        self,
        individual_rewards: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute blended individual + team rewards.
        """
        # Compute team averages
        team_rewards: Dict[str, float] = {}
        for team_name, members in self.teams.items():
            rewards = [individual_rewards.get(m, 0.0) for m in members if m in individual_rewards]
            team_rewards[team_name] = float(np.mean(rewards)) if rewards else 0.0

        # Blend
        blended = {}
        for aid, ind_r in individual_rewards.items():
            team_name = self._agent_team.get(aid)
            if team_name is not None:
                team_r   = team_rewards.get(team_name, 0.0)
                blended[aid] = (
                    (1.0 - self.blend_alpha) * ind_r
                    + self.blend_alpha * team_r
                )
            else:
                blended[aid] = ind_r

        return blended

    @staticmethod
    def build_from_agent_ids(
        agent_ids: List[str],
        blend_alpha: float = 0.3,
    ) -> "TeamReward":
        """
        Auto-build teams from agent ID prefixes.
        e.g., "mm_0", "mm_1" → team "mm"
        """
        teams: Dict[str, List[str]] = {}
        for aid in agent_ids:
            prefix = aid.rsplit("_", 1)[0] if "_" in aid else aid
            teams.setdefault(prefix, []).append(aid)
        return TeamReward(teams, blend_alpha)


# ============================================================
# Market Quality Reward
# ============================================================

class MarketQualityReward:
    """
    Bonus reward for agents that improve market quality.

    Metrics that define "market quality":
      - Bid-ask spread (lower = better)
      - Price volatility (lower = better)
      - Market depth (higher = better)
      - Price discovery speed (higher = better)

    Used to align market maker incentives with market health.
    """

    def __init__(
        self,
        spread_weight:   float = 0.5,
        vol_weight:      float = 0.3,
        depth_weight:    float = 0.2,
        target_spread:   float = 0.02,  # bps of mid price
        spread_window:   int   = 20,
        vol_window:      int   = 50,
        bonus_scale:     float = 0.1,
    ) -> None:
        self.spread_weight = spread_weight
        self.vol_weight    = vol_weight
        self.depth_weight  = depth_weight
        self.target_spread = target_spread
        self.bonus_scale   = bonus_scale

        self._spread_hist: deque = deque(maxlen=spread_window)
        self._vol_hist:    deque = deque(maxlen=vol_window)
        self._depth_hist:  deque = deque(maxlen=spread_window)

    def update_market(
        self,
        spread:    float,
        mid_price: float,
        returns:   Optional[float] = None,
        depth:     Optional[float] = None,
    ) -> None:
        """Call each step to update market quality history."""
        if mid_price > 0:
            self._spread_hist.append(spread / mid_price)
        if returns is not None:
            self._vol_hist.append(returns**2)
        if depth is not None:
            self._depth_hist.append(depth)

    def compute_bonus(self, agent_type: str) -> float:
        """
        Compute market quality bonus for an agent.

        Market makers get the full bonus; others get 0.
        """
        if not agent_type.startswith("mm"):
            return 0.0

        bonus = 0.0

        # Spread bonus: reward tight spreads
        if self._spread_hist:
            avg_spread = float(np.mean(list(self._spread_hist)))
            spread_bonus = max(0.0, (self.target_spread - avg_spread) / self.target_spread)
            bonus += self.spread_weight * spread_bonus

        # Volatility penalty: penalize high vol
        if len(self._vol_hist) >= 5:
            avg_var = float(np.mean(list(self._vol_hist)))
            vol_bonus = max(0.0, 1.0 / (1.0 + 100 * avg_var) - 0.5)
            bonus += self.vol_weight * vol_bonus

        # Depth bonus: reward providing depth
        if self._depth_hist:
            avg_depth = float(np.mean(list(self._depth_hist)))
            depth_bonus = min(1.0, avg_depth / 100.0)
            bonus += self.depth_weight * depth_bonus

        return float(bonus * self.bonus_scale)

    def shape_rewards(
        self,
        rewards:     Dict[str, float],
        agent_types: Dict[str, str],
    ) -> Dict[str, float]:
        """Add market quality bonus to each agent's reward."""
        shaped = {}
        for aid, r in rewards.items():
            atype = agent_types.get(aid, "unknown")
            bonus = self.compute_bonus(atype)
            shaped[aid] = r + bonus
        return shaped


# ============================================================
# Adversarial Penalty
# ============================================================

class AdversarialPenalty:
    """
    Penalize agents for market manipulation behaviors.

    Detected behaviors:
      1. Spoofing:   rapid consecutive same-direction large orders cancelled
      2. Ramping:    sustained one-directional pressure
      3. Wash trading: buying and selling in same direction in same tick
    """

    def __init__(
        self,
        spoof_penalty:  float = 0.5,
        ramp_penalty:   float = 0.3,
        wash_penalty:   float = 0.2,
        window:         int   = 20,
    ) -> None:
        self.spoof_penalty = spoof_penalty
        self.ramp_penalty  = ramp_penalty
        self.wash_penalty  = wash_penalty
        self.window        = window
        self._action_hist: Dict[str, deque] = {}

    def observe_action(self, agent_id: str, direction: int, size: float) -> None:
        if agent_id not in self._action_hist:
            self._action_hist[agent_id] = deque(maxlen=self.window)
        self._action_hist[agent_id].append((direction, size))

    def compute_penalty(self, agent_id: str) -> float:
        hist = self._action_hist.get(agent_id)
        if hist is None or len(hist) < 5:
            return 0.0

        actions = list(hist)
        penalty = 0.0

        # Ramping: more than 80% of recent actions in same direction
        dirs   = [a[0] for a in actions if a[0] != 0]
        if dirs:
            dominant = max(abs(sum(dirs)), 0)
            ramp_frac = dominant / len(dirs)
            if ramp_frac > 0.85:
                penalty += self.ramp_penalty

        # Spoofing: large orders followed immediately by reversal
        sizes = [a[1] for a in actions]
        mean_size = float(np.mean(sizes)) + 1e-8
        for i in range(len(actions) - 1):
            if sizes[i] > 2 * mean_size and actions[i][0] != 0:
                # Large order followed by opposite direction
                if actions[i + 1][0] == -actions[i][0]:
                    penalty += self.spoof_penalty
                    break

        return float(min(penalty, 1.0))

    def shape_rewards(
        self,
        rewards:  Dict[str, float],
        actions:  Dict[str, Tuple[int, float]],
    ) -> Dict[str, float]:
        shaped = {}
        for aid, r in rewards.items():
            if aid in actions:
                d, s = actions[aid]
                self.observe_action(aid, d, s)
            penalty    = self.compute_penalty(aid)
            shaped[aid] = r - penalty
        return shaped


# ============================================================
# Potential-Based Shaping
# ============================================================

class PotentialBasedShaping:
    """
    Reward shaping via potential function.

    F(s, a, s') = γ * Φ(s') - Φ(s)

    By Ng et al. (1999), any potential-based shaping preserves
    the set of Nash equilibria of the original game.

    Potential function Φ(s) = f(market_quality) + g(agent_balance)
    """

    def __init__(
        self,
        gamma:          float = 0.99,
        quality_weight: float = 1.0,
        balance_weight: float = 0.5,
    ) -> None:
        self.gamma          = gamma
        self.quality_weight = quality_weight
        self.balance_weight = balance_weight

    def potential(
        self,
        mid_price:    float,
        spread:       float,
        volume:       float,
        agent_pnls:   Dict[str, float],
    ) -> float:
        """
        Φ(market_state) = quality_weight * market_quality
                        + balance_weight * wealth_gini_complement
        """
        # Market quality potential: tight spread, good volume
        if mid_price > 0:
            rel_spread = spread / mid_price
        else:
            rel_spread = 1.0
        quality = max(0.0, 1.0 - rel_spread * 100) * min(1.0, math.log1p(volume) / 5.0)

        # Wealth balance: 1 - Gini coefficient of agent PnLs
        pnls = np.array(list(agent_pnls.values()), dtype=np.float64)
        if len(pnls) > 1 and pnls.max() > pnls.min():
            gini   = self._gini(pnls - pnls.min())
            balance = 1.0 - gini
        else:
            balance = 1.0

        return float(
            self.quality_weight * quality
            + self.balance_weight * balance
        )

    @staticmethod
    def _gini(arr: np.ndarray) -> float:
        arr = np.sort(arr)
        n   = len(arr)
        if arr.sum() < 1e-8:
            return 0.0
        idx   = np.arange(1, n + 1)
        return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)

    def shape(
        self,
        rewards:          Dict[str, float],
        prev_state:       Dict[str, float],  # mid_price, spread, volume, pnls
        curr_state:       Dict[str, float],
    ) -> Dict[str, float]:
        """Add shaping bonus F = γΦ(s') - Φ(s) to each agent's reward."""
        prev_pnls = {k: v for k, v in prev_state.items() if k.startswith("pnl_")}
        curr_pnls = {k: v for k, v in curr_state.items() if k.startswith("pnl_")}

        phi_prev = self.potential(
            prev_state.get("mid_price", 100.0),
            prev_state.get("spread", 0.02),
            prev_state.get("volume", 1.0),
            prev_pnls,
        )
        phi_curr = self.potential(
            curr_state.get("mid_price", 100.0),
            curr_state.get("spread", 0.02),
            curr_state.get("volume", 1.0),
            curr_pnls,
        )
        shaping = self.gamma * phi_curr - phi_prev

        return {aid: r + shaping for aid, r in rewards.items()}


# ============================================================
# Curiosity Bonus (Random Network Distillation)
# ============================================================

class RNDNetwork(nn.Module):
    """Fixed random target network for RND exploration bonus."""

    def __init__(self, obs_dim: int, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )
        # Fix weights: target is never trained
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDPredictor(nn.Module):
    """Trained predictor network for RND (learns to match target)."""

    def __init__(self, obs_dim: int, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CuriosityBonus:
    """
    Intrinsic curiosity reward via Random Network Distillation.

    Bonus = MSE(predictor(obs), target(obs))
    High error → novel state → high intrinsic reward.
    Predictor is trained to reduce error over time.
    """

    def __init__(
        self,
        obs_dim:    int,
        out_dim:    int   = 32,
        lr:         float = 1e-3,
        scale:      float = 0.1,
        norm_clip:  float = 5.0,
        device:     str   = "cpu",
    ) -> None:
        self.scale     = scale
        self.norm_clip = norm_clip
        self.device    = torch.device(device)

        self.target    = RNDNetwork(obs_dim, out_dim).to(self.device)
        self.predictor = RNDPredictor(obs_dim, out_dim).to(self.device)
        self.optimizer = Adam(self.predictor.parameters(), lr=lr)

        # Running normalization of intrinsic rewards
        self._bonus_mean = 0.0
        self._bonus_var  = 1.0
        self._bonus_n    = 0

    def compute_bonus(self, obs: np.ndarray) -> float:
        """Compute intrinsic reward for a single observation."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            target_feat = self.target(obs_t)
            pred_feat   = self.predictor(obs_t)
            error       = F.mse_loss(pred_feat, target_feat).item()

        # Normalize bonus
        self._bonus_n    += 1
        delta             = error - self._bonus_mean
        self._bonus_mean += delta / self._bonus_n
        self._bonus_var   = (
            self._bonus_var * (self._bonus_n - 1) + delta * (error - self._bonus_mean)
        ) / max(self._bonus_n, 1)
        bonus_std = math.sqrt(max(self._bonus_var, 1e-8))

        norm_bonus = (error - self._bonus_mean) / bonus_std
        norm_bonus = float(np.clip(norm_bonus, -self.norm_clip, self.norm_clip))
        return float(self.scale * norm_bonus)

    def update_predictor(self, obs_batch: np.ndarray) -> float:
        """Train predictor to reduce error on seen observations."""
        obs_t = torch.FloatTensor(obs_batch).to(self.device)
        with torch.no_grad():
            target_feat = self.target(obs_t)
        pred_feat = self.predictor(obs_t)
        loss      = F.mse_loss(pred_feat, target_feat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


# ============================================================
# Composite Reward
# ============================================================

class CompositeReward:
    """
    Weighted combination of all reward components.

    Weights:
      individual:     PnL-based individual reward
      team:           team-based blended reward
      market_quality: market health bonus
      adversarial:    manipulation penalty
      potential:      potential-based shaping bonus
      curiosity:      RND intrinsic exploration

    All weights default to reasonable values that can be tuned.
    """

    def __init__(
        self,
        agent_ids:       List[str],
        agent_types:     Dict[str, str],
        ind_weight:      float = 1.0,
        team_weight:     float = 0.3,
        mq_weight:       float = 0.2,
        adv_weight:      float = 0.5,
        pot_weight:      float = 0.1,
        cur_weight:      float = 0.05,
        obs_dim:         int   = 23,
        ind_mode:        str   = "raw",
        device:          str   = "cpu",
    ) -> None:
        self.agent_ids   = agent_ids
        self.agent_types = agent_types
        self.weights     = {
            "individual":     ind_weight,
            "team":           team_weight,
            "market_quality": mq_weight,
            "adversarial":    adv_weight,
            "potential":      pot_weight,
            "curiosity":      cur_weight,
        }

        self.individual    = IndividualReward(mode=ind_mode)
        self.team          = TeamReward.build_from_agent_ids(agent_ids)
        self.mq            = MarketQualityReward()
        self.adversarial   = AdversarialPenalty()
        self.potential     = PotentialBasedShaping()
        self.curiosity     = {
            aid: CuriosityBonus(obs_dim, device=device)
            for aid in agent_ids
        }

        self._prev_state: Optional[Dict] = None

    def compute(
        self,
        raw_rewards:  Dict[str, float],
        observations: Dict[str, np.ndarray],
        actions:      Dict[str, Tuple[int, float]],
        market_info:  Dict[str, Any],
        update_pred:  bool = False,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Compute composite reward for all agents.

        Returns:
            (shaped_rewards, component_breakdown)
        """
        breakdowns: Dict[str, Dict[str, float]] = {a: {} for a in self.agent_ids}

        # 1. Individual reward
        ind_rewards = {
            aid: self.individual.compute(r) * self.weights["individual"]
            for aid, r in raw_rewards.items()
        }

        # 2. Team reward
        team_rewards = self.team.compute(ind_rewards)
        for aid in self.agent_ids:
            tr = team_rewards.get(aid, ind_rewards.get(aid, 0.0))
            breakdowns[aid]["team"] = (tr - ind_rewards.get(aid, 0.0)) * self.weights["team"]

        # 3. Market quality
        self.mq.update_market(
            spread    = market_info.get("spread", 0.02),
            mid_price = market_info.get("mid_price", 100.0),
            returns   = market_info.get("last_return"),
            depth     = market_info.get("depth"),
        )
        mq_bonuses = self.mq.shape_rewards(
            {a: 0.0 for a in self.agent_ids}, self.agent_types
        )
        for aid in self.agent_ids:
            breakdowns[aid]["market_quality"] = mq_bonuses.get(aid, 0.0)

        # 4. Adversarial penalty
        adv_shaped = self.adversarial.shape_rewards(ind_rewards, actions)
        for aid in self.agent_ids:
            penalty = ind_rewards.get(aid, 0.0) - adv_shaped.get(aid, ind_rewards.get(aid, 0.0))
            breakdowns[aid]["adversarial"] = -penalty * self.weights["adversarial"]

        # 5. Potential-based shaping
        curr_state: Dict[str, Any] = {
            "mid_price": market_info.get("mid_price", 100.0),
            "spread":    market_info.get("spread", 0.02),
            "volume":    market_info.get("volume", 1.0),
        }
        for aid in self.agent_ids:
            curr_state[f"pnl_{aid}"] = raw_rewards.get(aid, 0.0)

        if self._prev_state is not None:
            shaped_pot = self.potential.shape(ind_rewards, self._prev_state, curr_state)
            for aid in self.agent_ids:
                pot_bonus = shaped_pot.get(aid, 0.0) - ind_rewards.get(aid, 0.0)
                breakdowns[aid]["potential"] = pot_bonus * self.weights["potential"]
        self._prev_state = curr_state

        # 6. Curiosity
        for aid in self.agent_ids:
            if aid in observations:
                obs  = observations[aid]
                cur  = self.curiosity[aid].compute_bonus(obs)
                if update_pred:
                    self.curiosity[aid].update_predictor(obs[np.newaxis])
                breakdowns[aid]["curiosity"] = cur * self.weights["curiosity"]

        # Combine
        final_rewards: Dict[str, float] = {}
        for aid in self.agent_ids:
            total = ind_rewards.get(aid, 0.0)
            for k, v in breakdowns[aid].items():
                total += v
            final_rewards[aid] = total

        return final_rewards, breakdowns
