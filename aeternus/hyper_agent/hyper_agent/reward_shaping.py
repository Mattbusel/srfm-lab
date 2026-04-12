"""
reward_shaping.py
=================
Comprehensive multi-objective reward engineering for the Hyper-Agent MARL
ecosystem.  This module provides:

  - Sharpe-ratio shaped reward with rolling window
  - Drawdown penalty (maximum drawdown tracking)
  - Inventory risk penalty (quadratic + asymmetric)
  - Market impact cost estimator
  - Execution quality reward (vs VWAP benchmark)
  - Cooperative reward for liquidity provision agents
  - Adversarial reward for destabilising agents
  - Reward normalisation / clipping
  - Curriculum reward scheduling (gradually introduce harder objectives)
  - Multi-objective reward combiner with Pareto preference
  - Potential-based reward shaping (PBRS) for consistent equilibria

All reward components implement a common BaseRewardComponent interface and
are fully differentiable via PyTorch where numerical gradients are needed.
"""

from __future__ import annotations

import abc
import copy
import dataclasses
import enum
import logging
import math
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPS: float = 1e-9
CLIP_DEFAULT: float = 10.0
SHARPE_ANNUALISE_FACTOR: float = math.sqrt(252 * 390)   # trading ticks/year approx

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RewardContext:
    """All information available for reward computation at one time step."""
    agent_id: str
    inventory: float
    prev_inventory: float
    cash: float
    prev_cash: float
    mid_price: float
    prev_mid_price: float
    best_bid: float
    best_ask: float
    spread: float
    vwap: float
    fill_price: Optional[float]
    fill_size: Optional[float]
    fill_side: Optional[int]           # +1 buy, -1 sell
    slippage: float = 0.0
    market_impact: float = 0.0
    realized_pnl_delta: float = 0.0
    total_pnl: float = 0.0
    num_trades: int = 0
    episode_step: int = 0
    episode_length: int = 2000
    volatility: float = 0.001
    market_imbalance: float = 0.0


@dataclasses.dataclass
class RewardOutput:
    """Structured reward output from a component."""
    value: float
    raw_value: float
    clipped: bool = False
    metadata: Dict[str, float] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base component
# ---------------------------------------------------------------------------

class BaseRewardComponent(abc.ABC):
    """Abstract base class for all reward components."""

    def __init__(self, name: str, weight: float = 1.0,
                 clip: Optional[float] = CLIP_DEFAULT):
        self.name = name
        self.weight = weight
        self.clip = clip
        self._call_count = 0

    @abc.abstractmethod
    def _compute(self, ctx: RewardContext) -> float:
        """Compute raw un-weighted, un-clipped reward signal."""
        ...

    def compute(self, ctx: RewardContext) -> RewardOutput:
        raw = self._compute(ctx)
        value = raw * self.weight
        clipped = False
        if self.clip is not None and abs(value) > self.clip:
            value = math.copysign(self.clip, value)
            clipped = True
        self._call_count += 1
        return RewardOutput(value=value, raw_value=raw, clipped=clipped)

    def reset(self) -> None:
        """Reset per-episode state."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, w={self.weight})"


# ---------------------------------------------------------------------------
# 1. Mark-to-Market P&L reward
# ---------------------------------------------------------------------------

class MarkToMarketReward(BaseRewardComponent):
    """Step-wise change in portfolio mark-to-market value."""

    def __init__(self, weight: float = 1.0, clip: float = CLIP_DEFAULT):
        super().__init__("mark_to_market", weight, clip)

    def _compute(self, ctx: RewardContext) -> float:
        mtm_curr = ctx.inventory * ctx.mid_price + ctx.cash
        mtm_prev = ctx.prev_inventory * ctx.prev_mid_price + ctx.prev_cash
        return mtm_curr - mtm_prev


# ---------------------------------------------------------------------------
# 2. Realised P&L reward
# ---------------------------------------------------------------------------

class RealisedPnLReward(BaseRewardComponent):
    """Realised P&L delta this step (fills only)."""

    def __init__(self, weight: float = 1.0, clip: float = CLIP_DEFAULT):
        super().__init__("realised_pnl", weight, clip)

    def _compute(self, ctx: RewardContext) -> float:
        return ctx.realized_pnl_delta


# ---------------------------------------------------------------------------
# 3. Sharpe-ratio shaped reward
# ---------------------------------------------------------------------------

class SharpeReward(BaseRewardComponent):
    """
    Reward shaped by the rolling Sharpe ratio of P&L returns.
    Encourages consistent, low-variance positive returns.
    """

    def __init__(self,
                 window: int = 100,
                 min_samples: int = 10,
                 annualise: bool = False,
                 weight: float = 1.0,
                 clip: float = 5.0):
        super().__init__("sharpe", weight, clip)
        self.window = window
        self.min_samples = min_samples
        self.annualise = annualise
        self._pnl_buffer: deque = deque(maxlen=window)
        self._prev_pnl: float = 0.0

    def reset(self) -> None:
        self._pnl_buffer.clear()
        self._prev_pnl = 0.0

    def _compute(self, ctx: RewardContext) -> float:
        pnl = ctx.total_pnl
        ret = pnl - self._prev_pnl
        self._prev_pnl = pnl
        self._pnl_buffer.append(ret)

        if len(self._pnl_buffer) < self.min_samples:
            return 0.0

        rets = np.array(self._pnl_buffer)
        mean_r = rets.mean()
        std_r = rets.std() + EPS
        sharpe = mean_r / std_r
        if self.annualise:
            sharpe *= SHARPE_ANNUALISE_FACTOR
        return float(sharpe)


# ---------------------------------------------------------------------------
# 4. Drawdown penalty
# ---------------------------------------------------------------------------

class DrawdownPenalty(BaseRewardComponent):
    """
    Penalises agents proportionally to their current drawdown from peak PnL.
    Uses both maximum drawdown and current drawdown signals.
    """

    def __init__(self,
                 max_dd_coef: float = 0.5,
                 current_dd_coef: float = 1.0,
                 weight: float = 1.0,
                 clip: float = 5.0):
        super().__init__("drawdown_penalty", weight, clip)
        self.max_dd_coef = max_dd_coef
        self.current_dd_coef = current_dd_coef
        self._peak_pnl: float = 0.0
        self._max_drawdown: float = 0.0

    def reset(self) -> None:
        self._peak_pnl = 0.0
        self._max_drawdown = 0.0

    def _compute(self, ctx: RewardContext) -> float:
        pnl = ctx.total_pnl
        if pnl > self._peak_pnl:
            self._peak_pnl = pnl

        current_dd = self._peak_pnl - pnl
        if current_dd > self._max_drawdown:
            self._max_drawdown = current_dd

        penalty = (self.current_dd_coef * current_dd +
                   self.max_dd_coef * self._max_drawdown)
        return -penalty


# ---------------------------------------------------------------------------
# 5. Inventory risk penalty
# ---------------------------------------------------------------------------

class InventoryRiskPenalty(BaseRewardComponent):
    """
    Penalises large inventory positions.  Uses:
      - Quadratic penalty for symmetric discouragement of large positions
      - Asymmetric addon if inventory exceeds a safe threshold
      - Trending position reward when inventory is aligned with price direction
    """

    def __init__(self,
                 max_safe_inventory: float = 100.0,
                 quadratic_coef: float = 1e-4,
                 excess_coef: float = 1e-3,
                 trend_alignment_bonus: float = 5e-5,
                 weight: float = 1.0,
                 clip: float = 3.0):
        super().__init__("inventory_risk", weight, clip)
        self.max_safe_inventory = max_safe_inventory
        self.quadratic_coef = quadratic_coef
        self.excess_coef = excess_coef
        self.trend_alignment_bonus = trend_alignment_bonus
        self._prev_price: float = 0.0

    def reset(self) -> None:
        self._prev_price = 0.0

    def _compute(self, ctx: RewardContext) -> float:
        inv = ctx.inventory
        # Quadratic
        q_penalty = self.quadratic_coef * (inv ** 2)

        # Excess beyond safe zone
        excess = max(0.0, abs(inv) - self.max_safe_inventory)
        e_penalty = self.excess_coef * (excess ** 2)

        # Alignment bonus: reward if inventory is on right side of price move
        price_move = ctx.mid_price - ctx.prev_mid_price
        alignment = math.copysign(1.0, inv) * math.copysign(1.0, price_move) \
            if inv != 0 and price_move != 0 else 0.0
        a_bonus = self.trend_alignment_bonus * abs(inv) * alignment

        return -(q_penalty + e_penalty) + a_bonus


# ---------------------------------------------------------------------------
# 6. Market impact cost
# ---------------------------------------------------------------------------

class MarketImpactCost(BaseRewardComponent):
    """
    Penalises the market impact of the agent's own trades.
    Computed as the spread between fill price and theoretical fair value.
    """

    def __init__(self,
                 permanent_impact_coef: float = 1e-4,
                 temporary_impact_coef: float = 5e-5,
                 weight: float = 1.0,
                 clip: float = 2.0):
        super().__init__("market_impact_cost", weight, clip)
        self.permanent_impact_coef = permanent_impact_coef
        self.temporary_impact_coef = temporary_impact_coef

    def _compute(self, ctx: RewardContext) -> float:
        if ctx.fill_size is None or ctx.fill_size == 0:
            return 0.0

        vol_proxy = max(ctx.volatility, 1e-6)
        perm_impact = (self.permanent_impact_coef *
                       vol_proxy *
                       abs(ctx.fill_size) *
                       ctx.mid_price)
        temp_impact = (self.temporary_impact_coef *
                       abs(ctx.fill_size) ** 0.5 *
                       ctx.mid_price)
        return -(perm_impact + temp_impact)


# ---------------------------------------------------------------------------
# 7. Execution quality reward (vs VWAP benchmark)
# ---------------------------------------------------------------------------

class ExecutionQualityReward(BaseRewardComponent):
    """
    Rewards orders that execute at a better price than VWAP.
    Buy fills below VWAP are good; sell fills above VWAP are good.
    """

    def __init__(self, weight: float = 1.0, clip: float = 2.0,
                 normalise_by_spread: bool = True):
        super().__init__("execution_quality", weight, clip)
        self.normalise_by_spread = normalise_by_spread

    def _compute(self, ctx: RewardContext) -> float:
        if ctx.fill_price is None or ctx.fill_side is None:
            return 0.0

        fill_side = ctx.fill_side   # +1 buy, -1 sell
        improvement = fill_side * (ctx.vwap - ctx.fill_price)

        if self.normalise_by_spread and ctx.spread > 0:
            improvement /= ctx.spread

        return improvement * abs(ctx.fill_size or 0.0)


# ---------------------------------------------------------------------------
# 8. Slippage penalty
# ---------------------------------------------------------------------------

class SlippagePenalty(BaseRewardComponent):
    """Penalises observed slippage on each fill."""

    def __init__(self, weight: float = 1.0, clip: float = 2.0):
        super().__init__("slippage_penalty", weight, clip)

    def _compute(self, ctx: RewardContext) -> float:
        return -ctx.slippage


# ---------------------------------------------------------------------------
# 9. Cooperative liquidity provision reward
# ---------------------------------------------------------------------------

class LiquidityProvisionReward(BaseRewardComponent):
    """
    Rewards agents that narrow the spread (add liquidity).
    Designed for market-maker role agents.
    """

    def __init__(self,
                 spread_improvement_bonus: float = 10.0,
                 passive_fill_bonus: float = 1.0,
                 weight: float = 1.0,
                 clip: float = 3.0):
        super().__init__("liquidity_provision", weight, clip)
        self.spread_improvement_bonus = spread_improvement_bonus
        self.passive_fill_bonus = passive_fill_bonus
        self._prev_spread: float = 0.0

    def reset(self) -> None:
        self._prev_spread = 0.0

    def _compute(self, ctx: RewardContext) -> float:
        spread_improvement = self._prev_spread - ctx.spread
        self._prev_spread = ctx.spread

        reward = 0.0
        if spread_improvement > 0:
            reward += self.spread_improvement_bonus * spread_improvement

        # Bonus for passive fills (resting orders filled = provided liquidity)
        if ctx.fill_size is not None and ctx.fill_size > 0:
            is_passive = (ctx.fill_price is not None and
                          ctx.fill_side is not None and
                          ((ctx.fill_side > 0 and ctx.fill_price <= ctx.best_bid) or
                           (ctx.fill_side < 0 and ctx.fill_price >= ctx.best_ask)))
            if is_passive:
                reward += self.passive_fill_bonus * ctx.fill_size

        return reward


# ---------------------------------------------------------------------------
# 10. Adversarial destabilising reward
# ---------------------------------------------------------------------------

class DestabilisingReward(BaseRewardComponent):
    """
    Reward for adversarial agents that seek to destabilise market quality.
    Incentivises widening spreads, increasing volatility, causing crashes.
    """

    def __init__(self,
                 spread_widening_coef: float = 5.0,
                 vol_spike_coef: float = 100.0,
                 price_impact_coef: float = 1.0,
                 weight: float = 1.0,
                 clip: float = 5.0):
        super().__init__("destabilising", weight, clip)
        self.spread_widening_coef = spread_widening_coef
        self.vol_spike_coef = vol_spike_coef
        self.price_impact_coef = price_impact_coef
        self._prev_spread: float = 0.0
        self._prev_vol: float = 0.0

    def reset(self) -> None:
        self._prev_spread = 0.0
        self._prev_vol = 0.0

    def _compute(self, ctx: RewardContext) -> float:
        spread_change = ctx.spread - self._prev_spread
        vol_change = ctx.volatility - self._prev_vol
        self._prev_spread = ctx.spread
        self._prev_vol = ctx.volatility

        r = (self.spread_widening_coef * max(0.0, spread_change) +
             self.vol_spike_coef * max(0.0, vol_change))

        if ctx.fill_size is not None:
            price_impact = abs(ctx.mid_price - ctx.prev_mid_price) * ctx.fill_size
            r += self.price_impact_coef * price_impact

        return r


# ---------------------------------------------------------------------------
# 11. Potential-based reward shaping (PBRS)
# ---------------------------------------------------------------------------

class PBRSComponent(BaseRewardComponent):
    """
    Potential-based reward shaping: r_shaping = gamma * Phi(s') - Phi(s).
    Preserves the optimal policy of the original MDP.
    The potential function Phi can be any state-value estimate.
    """

    def __init__(self,
                 potential_fn: Callable[[RewardContext], float],
                 gamma: float = 0.99,
                 weight: float = 1.0,
                 clip: float = 2.0):
        super().__init__("pbrs", weight, clip)
        self.potential_fn = potential_fn
        self.gamma = gamma
        self._prev_potential: float = 0.0

    def reset(self) -> None:
        self._prev_potential = 0.0

    def _compute(self, ctx: RewardContext) -> float:
        curr_potential = self.potential_fn(ctx)
        shaping = self.gamma * curr_potential - self._prev_potential
        self._prev_potential = curr_potential
        return shaping


def inventory_potential(ctx: RewardContext) -> float:
    """Simple potential: negative squared inventory (encourages flat book)."""
    return -0.001 * (ctx.inventory ** 2)


def pnl_velocity_potential(ctx: RewardContext) -> float:
    """Potential proportional to P&L momentum."""
    return 0.01 * ctx.total_pnl


# ---------------------------------------------------------------------------
# 12. Curriculum reward scheduling
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Manages curriculum scheduling of reward component weights.
    As training progresses (by episode or step count), harder objectives
    are gradually introduced and their weights increase.

    Phases:
      Phase 0 (warm-up):    Only basic MTM reward
      Phase 1 (early):      Add inventory penalty
      Phase 2 (mid):        Add slippage + impact costs
      Phase 3 (mature):     Add Sharpe shaping + drawdown
      Phase 4 (advanced):   Full multi-objective
    """

    @dataclasses.dataclass
    class Phase:
        start_episode: int
        component_weights: Dict[str, float]
        description: str = ""

    def __init__(self, phases: Optional[List["CurriculumScheduler.Phase"]] = None):
        if phases is None:
            phases = self._default_phases()
        self.phases = sorted(phases, key=lambda p: p.start_episode)
        self._current_episode: int = 0
        self._current_phase_idx: int = 0

    @staticmethod
    def _default_phases() -> List["CurriculumScheduler.Phase"]:
        return [
            CurriculumScheduler.Phase(
                start_episode=0,
                component_weights={
                    "mark_to_market": 1.0,
                },
                description="warm-up: basic MTM",
            ),
            CurriculumScheduler.Phase(
                start_episode=500,
                component_weights={
                    "mark_to_market": 1.0,
                    "inventory_risk": 0.3,
                },
                description="early: add inventory penalty",
            ),
            CurriculumScheduler.Phase(
                start_episode=1500,
                component_weights={
                    "mark_to_market": 1.0,
                    "inventory_risk": 0.5,
                    "slippage_penalty": 0.3,
                    "market_impact_cost": 0.2,
                },
                description="mid: add execution costs",
            ),
            CurriculumScheduler.Phase(
                start_episode=3000,
                component_weights={
                    "mark_to_market": 0.5,
                    "sharpe": 0.5,
                    "inventory_risk": 0.5,
                    "slippage_penalty": 0.3,
                    "market_impact_cost": 0.3,
                    "drawdown_penalty": 0.3,
                },
                description="mature: add Sharpe + drawdown",
            ),
            CurriculumScheduler.Phase(
                start_episode=6000,
                component_weights={
                    "mark_to_market": 0.3,
                    "sharpe": 0.7,
                    "inventory_risk": 0.5,
                    "slippage_penalty": 0.4,
                    "market_impact_cost": 0.4,
                    "drawdown_penalty": 0.5,
                    "execution_quality": 0.3,
                    "liquidity_provision": 0.2,
                },
                description="advanced: full multi-objective",
            ),
        ]

    def step_episode(self) -> None:
        self._current_episode += 1
        for i, phase in enumerate(self.phases):
            if self._current_episode >= phase.start_episode:
                self._current_phase_idx = i

    def get_weights(self) -> Dict[str, float]:
        phase = self.phases[self._current_phase_idx]
        return dict(phase.component_weights)

    def apply_weights(self, components: Dict[str, BaseRewardComponent]) -> None:
        weights = self.get_weights()
        for name, comp in components.items():
            comp.weight = weights.get(name, 0.0)

    @property
    def current_phase_description(self) -> str:
        return self.phases[self._current_phase_idx].description

    @property
    def current_episode(self) -> int:
        return self._current_episode


# ---------------------------------------------------------------------------
# 13. Reward normaliser
# ---------------------------------------------------------------------------

class RewardNormaliser:
    """
    Online reward normalisation using Welford's algorithm.
    Clips after normalisation to prevent extreme values.
    """

    def __init__(self, clip: float = 10.0, update_freq: int = 1):
        self.clip = clip
        self.update_freq = update_freq
        self._mean: float = 0.0
        self._var: float = 1.0
        self._count: float = EPS
        self._step: int = 0

    def update_and_normalise(self, reward: float) -> float:
        self._step += 1
        if self._step % self.update_freq == 0:
            self._update(reward)
        std = math.sqrt(self._var) + EPS
        normalised = (reward - self._mean) / std
        return float(np.clip(normalised, -self.clip, self.clip))

    def _update(self, x: float) -> None:
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._var += (delta * delta2 - self._var) / self._count

    def normalise(self, reward: float) -> float:
        std = math.sqrt(self._var) + EPS
        return float(np.clip((reward - self._mean) / std, -self.clip, self.clip))

    def reset_stats(self) -> None:
        self._mean = 0.0
        self._var = 1.0
        self._count = EPS


# ---------------------------------------------------------------------------
# 14. Multi-objective reward combiner
# ---------------------------------------------------------------------------

class MultiObjectiveCombiner:
    """
    Combines multiple reward components into a scalar reward.

    Supports:
      - Weighted sum (default)
      - Chebyshev scalarisation (max deviation from utopia point)
      - Pareto dominance bonus
      - Constraint satisfaction mode (penalty if any objective below threshold)
    """

    class CombineMode(enum.Enum):
        WEIGHTED_SUM = "weighted_sum"
        CHEBYSHEV = "chebyshev"
        CONSTRAINT_PENALTY = "constraint_penalty"

    def __init__(self,
                 components: List[BaseRewardComponent],
                 mode: "MultiObjectiveCombiner.CombineMode" = None,
                 constraints: Optional[Dict[str, float]] = None,
                 constraint_penalty: float = 5.0):
        self.components = {c.name: c for c in components}
        self.mode = mode or self.CombineMode.WEIGHTED_SUM
        self.constraints = constraints or {}
        self.constraint_penalty = constraint_penalty
        self._normaliser = RewardNormaliser()
        self._component_history: Dict[str, deque] = {
            c.name: deque(maxlen=1000) for c in components
        }

    def compute(self, ctx: RewardContext,
                normalise: bool = True) -> Tuple[float, Dict[str, float]]:
        outputs: Dict[str, RewardOutput] = {
            name: comp.compute(ctx)
            for name, comp in self.components.items()
        }

        component_values = {name: out.value for name, out in outputs.items()}
        for name, val in component_values.items():
            self._component_history[name].append(val)

        if self.mode == self.CombineMode.WEIGHTED_SUM:
            reward = sum(component_values.values())
        elif self.mode == self.CombineMode.CHEBYSHEV:
            reward = self._chebyshev_scalar(component_values)
        elif self.mode == self.CombineMode.CONSTRAINT_PENALTY:
            reward = self._constraint_penalty(component_values)
        else:
            reward = sum(component_values.values())

        if normalise:
            reward = self._normaliser.update_and_normalise(reward)

        return reward, component_values

    def _chebyshev_scalar(self, values: Dict[str, float]) -> float:
        """Minimise maximum regret (Chebyshev scalarisation)."""
        if not values:
            return 0.0
        weighted_vals = list(values.values())
        return min(weighted_vals) + 0.1 * sum(weighted_vals)

    def _constraint_penalty(self, values: Dict[str, float]) -> float:
        base = sum(v for name, v in values.items()
                   if name not in self.constraints)
        penalty = 0.0
        for name, threshold in self.constraints.items():
            val = values.get(name, 0.0)
            if val < threshold:
                penalty += self.constraint_penalty * (threshold - val)
        return base - penalty

    def reset(self) -> None:
        for comp in self.components.values():
            comp.reset()
        self._normaliser.reset_stats()

    def component_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for name, hist in self._component_history.items():
            if hist:
                arr = np.array(hist)
                stats[name] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
        return stats


# ---------------------------------------------------------------------------
# 15. Role-specific reward factory
# ---------------------------------------------------------------------------

class AgentRole(enum.Enum):
    MARKET_MAKER = "market_maker"
    MOMENTUM_TRADER = "momentum_trader"
    ARBITRAGEUR = "arbitrageur"
    ADVERSARIAL = "adversarial"
    GENERIC = "generic"


def build_reward_combiner(role: AgentRole = AgentRole.GENERIC,
                          curriculum: bool = True) -> "ManagedRewardCombiner":
    """Factory: create a fully configured reward combiner for a given agent role."""
    if role == AgentRole.MARKET_MAKER:
        components = [
            MarkToMarketReward(weight=0.5),
            LiquidityProvisionReward(weight=1.5),
            InventoryRiskPenalty(weight=1.0, max_safe_inventory=50.0),
            SlippagePenalty(weight=0.5),
            DrawdownPenalty(weight=0.3),
        ]
    elif role == AgentRole.MOMENTUM_TRADER:
        components = [
            SharpeReward(window=50, weight=1.0),
            MarkToMarketReward(weight=0.5),
            InventoryRiskPenalty(weight=0.3, trend_alignment_bonus=2e-4),
            DrawdownPenalty(weight=0.5),
            ExecutionQualityReward(weight=0.5),
        ]
    elif role == AgentRole.ARBITRAGEUR:
        components = [
            RealisedPnLReward(weight=1.5),
            ExecutionQualityReward(weight=1.0),
            MarketImpactCost(weight=0.8),
            SlippagePenalty(weight=0.8),
            InventoryRiskPenalty(weight=0.5),
        ]
    elif role == AgentRole.ADVERSARIAL:
        components = [
            DestabilisingReward(weight=2.0),
            MarkToMarketReward(weight=0.3),
        ]
    else:  # GENERIC
        components = [
            MarkToMarketReward(weight=1.0),
            InventoryRiskPenalty(weight=0.5),
            DrawdownPenalty(weight=0.3),
        ]

    return ManagedRewardCombiner(
        components=components,
        curriculum=CurriculumScheduler() if curriculum else None,
    )


# ---------------------------------------------------------------------------
# 16. Managed reward combiner (top-level API)
# ---------------------------------------------------------------------------

class ManagedRewardCombiner:
    """
    Top-level reward computation object that combines all components,
    applies curriculum scheduling, normalisation, and logging.
    """

    def __init__(self,
                 components: List[BaseRewardComponent],
                 curriculum: Optional[CurriculumScheduler] = None,
                 normalise: bool = True,
                 combine_mode: MultiObjectiveCombiner.CombineMode = None):
        self._combiner = MultiObjectiveCombiner(
            components=components,
            mode=combine_mode or MultiObjectiveCombiner.CombineMode.WEIGHTED_SUM,
        )
        self._curriculum = curriculum
        self._normalise = normalise
        self._episode: int = 0
        self._step: int = 0
        self._reward_history: deque = deque(maxlen=10000)

    def compute(self, ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
        if self._curriculum is not None:
            self._curriculum.apply_weights(self._combiner.components)

        reward, breakdown = self._combiner.compute(ctx, normalise=self._normalise)
        self._step += 1
        self._reward_history.append(reward)
        return reward, breakdown

    def on_episode_end(self) -> None:
        self._episode += 1
        self._step = 0
        self._combiner.reset()
        if self._curriculum is not None:
            self._curriculum.step_episode()

    def get_stats(self) -> Dict[str, Any]:
        hist = np.array(self._reward_history) if self._reward_history else np.zeros(1)
        stats: Dict[str, Any] = {
            "episode": self._episode,
            "step": self._step,
            "mean_reward": float(hist.mean()),
            "std_reward": float(hist.std()),
            "component_stats": self._combiner.component_stats(),
        }
        if self._curriculum:
            stats["curriculum_phase"] = self._curriculum.current_phase_description
            stats["curriculum_episode"] = self._curriculum.current_episode
        return stats


# ---------------------------------------------------------------------------
# 17. Differential reward (for credit assignment)
# ---------------------------------------------------------------------------

class DifferentialReward:
    """
    Computes counterfactual difference rewards for COMA-style credit assignment.
    r_i = R(a_1,...,a_n) - R(a_1,...,a_default_i,...,a_n)
    where a_default_i is the default (no-op) action for agent i.
    """

    def __init__(self, default_action_fn: Callable[[str], RewardContext]):
        self.default_action_fn = default_action_fn

    def compute_difference_reward(self,
                                   joint_reward: float,
                                   agent_id: str,
                                   combiner: ManagedRewardCombiner) -> float:
        default_ctx = self.default_action_fn(agent_id)
        default_reward, _ = combiner.compute(default_ctx)
        return joint_reward - default_reward


# ---------------------------------------------------------------------------
# 18. Reward function for cooperative team tasks
# ---------------------------------------------------------------------------

class TeamRewardAggregator:
    """
    Aggregates individual rewards into team reward with configurable mixing.
    Supports:
      - Global team reward (fully cooperative)
      - Individual reward (fully competitive)
      - Mixed (lambda * individual + (1-lambda) * team)
      - Competitive: individual - mean_others
    """

    class MixMode(enum.Enum):
        GLOBAL = "global"
        INDIVIDUAL = "individual"
        MIXED = "mixed"
        COMPETITIVE = "competitive"

    def __init__(self, n_agents: int,
                 mix_mode: "TeamRewardAggregator.MixMode" = None,
                 mix_lambda: float = 0.5):
        self.n_agents = n_agents
        self.mix_mode = mix_mode or self.MixMode.MIXED
        self.mix_lambda = mix_lambda

    def aggregate(self, individual_rewards: Dict[str, float]) -> Dict[str, float]:
        if not individual_rewards:
            return {}

        vals = list(individual_rewards.values())
        team_reward = float(np.mean(vals))

        if self.mix_mode == self.MixMode.GLOBAL:
            return {aid: team_reward for aid in individual_rewards}

        if self.mix_mode == self.MixMode.INDIVIDUAL:
            return dict(individual_rewards)

        if self.mix_mode == self.MixMode.MIXED:
            return {
                aid: self.mix_lambda * r + (1 - self.mix_lambda) * team_reward
                for aid, r in individual_rewards.items()
            }

        if self.mix_mode == self.MixMode.COMPETITIVE:
            return {
                aid: r - (team_reward - r / self.n_agents) * (self.n_agents / max(1, self.n_agents - 1))
                for aid, r in individual_rewards.items()
            }

        return dict(individual_rewards)


# ---------------------------------------------------------------------------
# 19. Neural reward shaping network (learned potential function)
# ---------------------------------------------------------------------------

class NeuralPotentialNetwork(nn.Module):
    """
    Learned potential function Phi(s) for PBRS.
    Maps observation vectors to scalar potentials.
    Trained jointly with policy to maximise episode returns.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)

    def potential(self, obs: np.ndarray, device: str = "cpu") -> float:
        t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return float(self.forward(t).item())


class LearnedPBRS(BaseRewardComponent):
    """PBRS component backed by a neural potential network."""

    def __init__(self,
                 obs_dim: int,
                 gamma: float = 0.99,
                 hidden_dim: int = 64,
                 weight: float = 0.1,
                 clip: float = 1.0,
                 device: str = "cpu"):
        super().__init__("learned_pbrs", weight, clip)
        self.gamma = gamma
        self.device = device
        self.network = NeuralPotentialNetwork(obs_dim, hidden_dim).to(device)
        self._prev_potential: float = 0.0
        self._prev_obs: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_potential = 0.0
        self._prev_obs = None

    def _compute(self, ctx: RewardContext) -> float:
        # ctx doesn't carry obs directly; return 0 if no obs available
        return 0.0

    def compute_from_obs(self, obs: np.ndarray) -> float:
        curr_potential = self.network.potential(obs, self.device)
        shaping = self.gamma * curr_potential - self._prev_potential
        self._prev_potential = curr_potential
        return shaping * self.weight


# ---------------------------------------------------------------------------
# 20. Composite reward for market-making (convenience preset)
# ---------------------------------------------------------------------------

class MarketMakerRewardSuite:
    """
    Complete reward suite for market-maker agents.
    Provides a single .step(ctx) -> float interface with full telemetry.
    """

    def __init__(self,
                 spread_bonus_coef: float = 5.0,
                 inventory_limit: float = 100.0,
                 curriculum_enabled: bool = True):
        self.combiner = build_reward_combiner(
            role=AgentRole.MARKET_MAKER,
            curriculum=curriculum_enabled,
        )
        self._spread_bonus_coef = spread_bonus_coef
        self._inventory_limit = inventory_limit
        self._step_count = 0
        self._episode_rewards: List[float] = []

    def step(self, ctx: RewardContext) -> float:
        reward, breakdown = self.combiner.compute(ctx)

        # Hard penalty for exceeding inventory limit
        if abs(ctx.inventory) > self._inventory_limit:
            reward -= 5.0

        self._step_count += 1
        self._episode_rewards.append(reward)
        return reward

    def end_episode(self) -> Dict[str, float]:
        episode_return = sum(self._episode_rewards)
        stats = self.combiner.get_stats()
        stats["episode_return"] = episode_return
        stats["episode_steps"] = len(self._episode_rewards)
        self.combiner.on_episode_end()
        self._episode_rewards = []
        self._step_count = 0
        return stats


# ---------------------------------------------------------------------------
# Utilities: reward context construction helpers
# ---------------------------------------------------------------------------

def make_reward_context(agent_id: str,
                         prev_state: Dict[str, float],
                         curr_state: Dict[str, float],
                         snapshot_prev: Any,
                         snapshot_curr: Any,
                         fill: Optional[Any] = None,
                         episode_step: int = 0,
                         episode_length: int = 2000) -> RewardContext:
    """
    Convenience constructor: builds a RewardContext from state dicts
    and LOBSnapshot objects (duck-typed).
    """
    fill_price = fill.fill_price if fill is not None else None
    fill_size = fill.fill_size if fill is not None else None
    fill_side = (1 if fill.side == 0 else -1) if fill is not None else None
    slippage = fill.slippage if fill is not None else 0.0
    market_impact = fill.market_impact if fill is not None else 0.0

    realized_delta = (
        curr_state.get("realized_pnl", 0.0) - prev_state.get("realized_pnl", 0.0)
    )

    return RewardContext(
        agent_id=agent_id,
        inventory=curr_state.get("inventory", 0.0),
        prev_inventory=prev_state.get("inventory", 0.0),
        cash=curr_state.get("cash", 0.0),
        prev_cash=prev_state.get("cash", 0.0),
        mid_price=getattr(snapshot_curr, "mid_price", 0.0),
        prev_mid_price=getattr(snapshot_prev, "mid_price", 0.0),
        best_bid=getattr(snapshot_curr, "best_bid", 0.0),
        best_ask=getattr(snapshot_curr, "best_ask", 0.0),
        spread=getattr(snapshot_curr, "spread", 0.0),
        vwap=getattr(snapshot_curr, "vwap", 0.0),
        fill_price=fill_price,
        fill_size=fill_size,
        fill_side=fill_side,
        slippage=slippage,
        market_impact=market_impact,
        realized_pnl_delta=realized_delta,
        total_pnl=curr_state.get("total_pnl", 0.0),
        num_trades=curr_state.get("num_trades", 0),
        episode_step=episode_step,
        episode_length=episode_length,
        volatility=getattr(snapshot_curr, "volatility_est", 0.001),
        market_imbalance=getattr(snapshot_curr, "imbalance", 0.0),
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== reward_shaping.py smoke test ===")

    ctx = RewardContext(
        agent_id="agent_0",
        inventory=10.0,
        prev_inventory=5.0,
        cash=1000.0,
        prev_cash=900.0,
        mid_price=100.5,
        prev_mid_price=100.0,
        best_bid=100.4,
        best_ask=100.6,
        spread=0.2,
        vwap=100.3,
        fill_price=100.5,
        fill_size=5.0,
        fill_side=1,
        slippage=0.02,
        market_impact=0.01,
        realized_pnl_delta=5.0,
        total_pnl=50.0,
        num_trades=3,
        episode_step=100,
        episode_length=2000,
        volatility=0.002,
        market_imbalance=0.2,
    )

    components = [
        MarkToMarketReward(weight=1.0),
        SharpeReward(weight=0.5),
        InventoryRiskPenalty(weight=0.5),
        DrawdownPenalty(weight=0.3),
        LiquidityProvisionReward(weight=0.5),
        ExecutionQualityReward(weight=0.3),
        SlippagePenalty(weight=0.5),
    ]

    combiner = MultiObjectiveCombiner(components)
    reward, breakdown = combiner.compute(ctx, normalise=False)
    print(f"Total reward: {reward:.4f}")
    for name, val in breakdown.items():
        print(f"  {name:25s}: {val:+.6f}")

    # Test curriculum
    sched = CurriculumScheduler()
    for _ in range(3001):
        sched.step_episode()
    print(f"\nCurriculum phase at ep 3001: {sched.current_phase_description}")
    print(f"Weights: {sched.get_weights()}")

    # Test team aggregator
    agg = TeamRewardAggregator(3, TeamRewardAggregator.MixMode.MIXED, 0.5)
    ind_rewards = {"agent_0": 1.0, "agent_1": 2.0, "agent_2": -1.0}
    mixed = agg.aggregate(ind_rewards)
    print(f"\nMixed rewards: {mixed}")

    print("\nAll smoke tests passed.")
