"""
population.py — Heterogeneous agent population management.

Implements:
- Market maker agents (inventory management, spread quoting)
- Momentum agents (trend following)
- Arbitrageur agents (cross-asset price discrepancy)
- Noise traders (random background activity)
- Fitness-based selection (evolutionary dynamics)
- Population genetics: mutation, crossover, selection
- Replicator dynamics for strategy evolution
"""

from __future__ import annotations

import math
import copy
import logging
import collections
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

EPS = 1e-8


# ---------------------------------------------------------------------------
# Agent type enumerations
# ---------------------------------------------------------------------------

AGENT_TYPE_MARKET_MAKER = "market_maker"
AGENT_TYPE_MOMENTUM = "momentum"
AGENT_TYPE_ARBITRAGEUR = "arbitrageur"
AGENT_TYPE_NOISE = "noise_trader"
AGENT_TYPE_MARL = "marl"
AGENT_TYPE_FUNDAMENTAL = "fundamental"


# ---------------------------------------------------------------------------
# Scripted agents (non-RL, for background population)
# ---------------------------------------------------------------------------

class ScriptedAgent:
    """Base class for scripted (non-learning) agents."""

    def __init__(self, agent_id: int, agent_type: str, num_assets: int, seed: Optional[int] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.num_assets = num_assets
        self.rng = np.random.default_rng(seed)
        self.fitness: float = 0.0
        self._step_count: int = 0
        self._pnl_history: List[float] = []

    @abstractmethod
    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        ...

    def update_fitness(self, pnl_delta: float) -> None:
        self._pnl_history.append(pnl_delta)
        if len(self._pnl_history) >= 50:
            pnl_array = np.array(self._pnl_history[-50:])
            mean_ret = np.mean(pnl_array)
            std_ret = np.std(pnl_array) + EPS
            self.fitness = float(mean_ret / std_ret * math.sqrt(252))
        else:
            self.fitness = float(np.sum(self._pnl_history))

    def step(self) -> None:
        self._step_count += 1


class MarketMakerAgent(ScriptedAgent):
    """
    Market maker: provides liquidity by quoting tight bid-ask spreads.

    Strategy:
    - Quote at mid ± spread/2
    - Skew quotes to reduce inventory risk
    - Cancel and requote when position exceeds threshold
    """

    def __init__(
        self,
        agent_id: int,
        num_assets: int,
        base_spread: float = 0.002,
        inventory_target: float = 0.0,
        inventory_limit: float = 200.0,
        inventory_skew: float = 0.001,
        quote_size: float = 20.0,
        seed: Optional[int] = None,
    ):
        super().__init__(agent_id, AGENT_TYPE_MARKET_MAKER, num_assets, seed)
        self.base_spread = base_spread
        self.inventory_target = inventory_target
        self.inventory_limit = inventory_limit
        self.inventory_skew = inventory_skew
        self.quote_size = quote_size

    def select_action(self, obs: np.ndarray, inventory: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Returns action array: [bid_offset, ask_offset, bid_size_frac, ask_size_frac] per asset.
        """
        action = np.zeros(self.num_assets * 4, dtype=np.float32)
        inv = inventory if inventory is not None else np.zeros(self.num_assets)

        for i in range(self.num_assets):
            # Inventory skew: if long, lower ask to sell; if short, raise bid to buy
            inv_skew = -self.inventory_skew * inv[i] / (self.inventory_limit + EPS)
            inv_skew = np.clip(inv_skew, -0.5, 0.5)

            # Spread component (scaled to [-1, 1] offset range)
            half_spread = self.base_spread / 0.02  # normalize by max_offset
            bid_offset = -half_spread + inv_skew
            ask_offset = half_spread + inv_skew

            # Size: reduce when approaching inventory limit
            inv_frac = abs(inv[i]) / (self.inventory_limit + EPS)
            size_frac = max(0.1, 1.0 - inv_frac)

            action[i * 4 + 0] = float(np.clip(bid_offset, -1, 1))
            action[i * 4 + 1] = float(np.clip(ask_offset, -1, 1))
            action[i * 4 + 2] = float(size_frac)
            action[i * 4 + 3] = float(size_frac)

        return action


class MomentumAgent(ScriptedAgent):
    """
    Momentum trader: follows recent price trends.

    Strategy:
    - Buy when short-term MA > long-term MA
    - Sell when short-term MA < long-term MA
    - Position size proportional to signal strength
    """

    def __init__(
        self,
        agent_id: int,
        num_assets: int,
        short_window: int = 5,
        long_window: int = 20,
        signal_threshold: float = 0.001,
        max_position_frac: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(agent_id, AGENT_TYPE_MOMENTUM, num_assets, seed)
        self.short_window = short_window
        self.long_window = long_window
        self.signal_threshold = signal_threshold
        self.max_position_frac = max_position_frac
        self._price_history: Dict[int, collections.deque] = {
            i: collections.deque(maxlen=long_window + 1)
            for i in range(num_assets)
        }

    def update_prices(self, prices: np.ndarray) -> None:
        for i, p in enumerate(prices):
            self._price_history[i].append(float(p))

    def select_action(self, obs: np.ndarray, prices: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        action = np.zeros(self.num_assets * 4, dtype=np.float32)

        if prices is not None:
            self.update_prices(prices)

        for i in range(self.num_assets):
            hist = list(self._price_history[i])
            if len(hist) < self.long_window:
                continue

            short_ma = float(np.mean(hist[-self.short_window:]))
            long_ma = float(np.mean(hist[-self.long_window:]))
            signal = (short_ma - long_ma) / (long_ma + EPS)

            if abs(signal) < self.signal_threshold:
                continue

            # Directional bet: buy on uptrend, sell on downtrend
            signal_strength = min(abs(signal) / (3 * self.signal_threshold), 1.0)
            size_frac = self.max_position_frac * signal_strength

            if signal > 0:
                # Buy: aggressive bid
                action[i * 4 + 0] = 0.1  # bid slightly above mid
                action[i * 4 + 1] = -0.9  # ask far away (don't sell)
                action[i * 4 + 2] = size_frac
                action[i * 4 + 3] = 0.0
            else:
                # Sell: aggressive ask
                action[i * 4 + 0] = -0.9  # bid far away
                action[i * 4 + 1] = -0.1  # ask slightly below mid
                action[i * 4 + 2] = 0.0
                action[i * 4 + 3] = size_frac

        return action


class ArbitrageurAgent(ScriptedAgent):
    """
    Arbitrageur: exploits price discrepancies across assets.

    Strategy:
    - Monitor spread between correlated assets
    - Buy cheap asset, sell expensive asset when spread exceeds threshold
    """

    def __init__(
        self,
        agent_id: int,
        num_assets: int,
        spread_threshold: float = 0.005,
        lookback: int = 30,
        position_limit: float = 0.3,
        seed: Optional[int] = None,
    ):
        super().__init__(agent_id, AGENT_TYPE_ARBITRAGEUR, num_assets, seed)
        self.spread_threshold = spread_threshold
        self.lookback = lookback
        self.position_limit = position_limit
        self._spread_history: Dict[Tuple[int, int], collections.deque] = {}
        self._pairs: List[Tuple[int, int]] = [
            (i, j) for i in range(num_assets) for j in range(i + 1, num_assets)
        ]
        for pair in self._pairs:
            self._spread_history[pair] = collections.deque(maxlen=lookback)

    def update_prices(self, prices: np.ndarray) -> None:
        for i, j in self._pairs:
            spread = float(prices[i] / (prices[j] + EPS) - 1.0)
            self._spread_history[(i, j)].append(spread)

    def select_action(self, obs: np.ndarray, prices: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        action = np.zeros(self.num_assets * 4, dtype=np.float32)

        if prices is not None:
            self.update_prices(prices)

        for i, j in self._pairs:
            hist = list(self._spread_history[(i, j)])
            if len(hist) < 10:
                continue

            current_spread = hist[-1]
            mean_spread = float(np.mean(hist))
            std_spread = float(np.std(hist)) + EPS
            z_score = (current_spread - mean_spread) / std_spread

            if abs(z_score) < 1.0:
                continue

            size = min(self.position_limit, abs(z_score) / 3.0)

            if z_score > 1.0:
                # Asset i is expensive relative to j: sell i, buy j
                action[i * 4 + 1] = -0.05   # sell i aggressively
                action[i * 4 + 3] = size
                action[j * 4 + 0] = 0.05    # buy j aggressively
                action[j * 4 + 2] = size
            elif z_score < -1.0:
                # Asset i is cheap: buy i, sell j
                action[i * 4 + 0] = 0.05
                action[i * 4 + 2] = size
                action[j * 4 + 1] = -0.05
                action[j * 4 + 3] = size

        return action


class NoiseTraderAgent(ScriptedAgent):
    """
    Noise trader: submits random orders to provide background liquidity.
    Behavior based on Ornstein-Uhlenbeck random process.
    """

    def __init__(
        self,
        agent_id: int,
        num_assets: int,
        mean_size: float = 0.1,
        order_frequency: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(agent_id, AGENT_TYPE_NOISE, num_assets, seed)
        self.mean_size = mean_size
        self.order_frequency = order_frequency
        self._ou_state = np.zeros(num_assets)

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        action = np.zeros(self.num_assets * 4, dtype=np.float32)

        if self.rng.random() > self.order_frequency:
            return action

        # Random direction
        for i in range(self.num_assets):
            self._ou_state[i] = (
                0.9 * self._ou_state[i]
                + 0.1 * float(self.rng.standard_normal())
            )

            size = abs(float(self.rng.normal(self.mean_size, self.mean_size * 0.5)))
            offset = float(self.rng.normal(0, 0.1))

            if self._ou_state[i] > 0:
                # Random buy
                action[i * 4 + 0] = float(np.clip(offset, -1, 1))
                action[i * 4 + 2] = min(size, 0.5)
            else:
                # Random sell
                action[i * 4 + 1] = float(np.clip(offset, -1, 1))
                action[i * 4 + 3] = min(size, 0.5)

        return action


class FundamentalValueAgent(ScriptedAgent):
    """
    Fundamental value agent: trades based on estimated fundamental value.
    Buys when price < fundamental, sells when price > fundamental.
    """

    def __init__(
        self,
        agent_id: int,
        num_assets: int,
        fundamental_values: Optional[np.ndarray] = None,
        signal_noise: float = 0.02,
        aggressiveness: float = 0.3,
        seed: Optional[int] = None,
    ):
        super().__init__(agent_id, AGENT_TYPE_FUNDAMENTAL, num_assets, seed)
        if fundamental_values is not None:
            self.fundamental_values = np.asarray(fundamental_values)
        else:
            self.fundamental_values = np.ones(num_assets) * 100.0
        self.signal_noise = signal_noise
        self.aggressiveness = aggressiveness

    def select_action(self, obs: np.ndarray, prices: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        action = np.zeros(self.num_assets * 4, dtype=np.float32)

        if prices is None:
            return action

        for i in range(self.num_assets):
            # Noisy estimate of fundamental value
            fv = self.fundamental_values[i] * (1 + self.rng.normal(0, self.signal_noise))
            mispricing = (fv - prices[i]) / (prices[i] + EPS)

            if abs(mispricing) < 0.005:
                continue

            size = min(abs(mispricing) * self.aggressiveness * 10, 0.5)

            if mispricing > 0:
                # Price below fundamental: buy
                action[i * 4 + 0] = float(np.clip(mispricing * 5, -1, 1))
                action[i * 4 + 2] = float(size)
            else:
                # Price above fundamental: sell
                action[i * 4 + 1] = float(np.clip(mispricing * 5, -1, 1))
                action[i * 4 + 3] = float(size)

        return action


# ---------------------------------------------------------------------------
# Evolutionary dynamics
# ---------------------------------------------------------------------------

class FitnessEvaluator:
    """
    Evaluates agent fitness based on trading performance metrics.
    """

    def __init__(
        self,
        sharpe_weight: float = 0.5,
        profit_weight: float = 0.3,
        drawdown_weight: float = 0.2,
        evaluation_window: int = 100,
    ):
        self.sharpe_weight = sharpe_weight
        self.profit_weight = profit_weight
        self.drawdown_weight = drawdown_weight
        self.evaluation_window = evaluation_window

    def compute_fitness(
        self,
        returns: List[float],
        equity_history: List[float],
    ) -> float:
        if len(returns) < 10:
            return 0.0

        rets = np.array(returns[-self.evaluation_window:])

        # Sharpe ratio component
        mean_ret = float(np.mean(rets))
        std_ret = float(np.std(rets)) + EPS
        sharpe = mean_ret / std_ret * math.sqrt(252)

        # Profit component
        total_profit = float(np.sum(rets))

        # Drawdown component
        if len(equity_history) >= 2:
            eq = np.array(equity_history[-self.evaluation_window:])
            peak = np.maximum.accumulate(eq)
            drawdown = float(np.max((peak - eq) / (peak + EPS)))
        else:
            drawdown = 0.0

        fitness = (
            self.sharpe_weight * np.tanh(sharpe / 2.0)
            + self.profit_weight * np.tanh(total_profit * 10)
            - self.drawdown_weight * drawdown
        )
        return float(fitness)


class EvolutionaryDynamics:
    """
    Fitness-proportionate selection and reproduction for agent population.

    Implements:
    - Tournament selection
    - Fitness-proportionate (roulette wheel) selection
    - Replicator dynamics
    - Parameter mutation
    """

    def __init__(
        self,
        mutation_rate: float = 0.05,
        mutation_std: float = 0.1,
        tournament_size: int = 3,
        elitism: float = 0.1,  # fraction of top agents preserved
        selection_method: str = "tournament",  # "tournament", "roulette", "replicator"
    ):
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.selection_method = selection_method

    def tournament_select(
        self, fitnesses: np.ndarray, n_select: int
    ) -> List[int]:
        """Tournament selection: pick n_select parents."""
        selected = []
        N = len(fitnesses)
        for _ in range(n_select):
            competitors = np.random.choice(N, self.tournament_size, replace=False)
            winner = int(competitors[fitnesses[competitors].argmax()])
            selected.append(winner)
        return selected

    def roulette_select(
        self, fitnesses: np.ndarray, n_select: int
    ) -> List[int]:
        """Fitness-proportionate selection."""
        # Shift to ensure non-negative
        shifted = fitnesses - fitnesses.min() + EPS
        probs = shifted / shifted.sum()
        return list(np.random.choice(len(fitnesses), n_select, p=probs, replace=True))

    def replicator_select(
        self, fitnesses: np.ndarray, population_sizes: np.ndarray
    ) -> np.ndarray:
        """
        Replicator dynamics: update population fractions.
        dp_i/dt = p_i * (f_i - f_avg) / f_avg
        """
        f_avg = float(np.dot(population_sizes / population_sizes.sum(), fitnesses))
        new_sizes = population_sizes.copy()
        for i in range(len(fitnesses)):
            growth = (fitnesses[i] - f_avg) / (abs(f_avg) + EPS)
            new_sizes[i] *= max(0.1, 1 + 0.1 * growth)
        # Normalize
        new_sizes = new_sizes / new_sizes.sum()
        return new_sizes

    def select_parents(
        self, fitnesses: np.ndarray, n_select: int
    ) -> List[int]:
        if self.selection_method == "tournament":
            return self.tournament_select(fitnesses, n_select)
        elif self.selection_method == "roulette":
            return self.roulette_select(fitnesses, n_select)
        else:
            # Default to tournament
            return self.tournament_select(fitnesses, n_select)

    def mutate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Gaussian mutation to scalar parameters."""
        mutated = {}
        for k, v in params.items():
            if isinstance(v, float) and np.random.random() < self.mutation_rate:
                noise = np.random.normal(0, self.mutation_std * abs(v + EPS))
                mutated[k] = v + noise
            elif isinstance(v, np.ndarray) and np.random.random() < self.mutation_rate:
                noise = np.random.normal(0, self.mutation_std, v.shape) * v
                mutated[k] = v + noise
            else:
                mutated[k] = v
        return mutated

    def mutate_network(self, network: nn.Module, device: torch.device) -> nn.Module:
        """Apply parameter noise to neural network weights."""
        mutated = copy.deepcopy(network)
        with torch.no_grad():
            for param in mutated.parameters():
                if np.random.random() < self.mutation_rate:
                    noise = torch.randn_like(param) * self.mutation_std
                    param.add_(noise)
        return mutated


# ---------------------------------------------------------------------------
# Population manager
# ---------------------------------------------------------------------------

class AgentPopulation:
    """
    Manages a heterogeneous population of trading agents.

    Supports:
    - Mixed population (scripted + RL agents)
    - Fitness evaluation and tracking
    - Evolutionary replacement
    - Population statistics
    """

    def __init__(
        self,
        num_market_makers: int = 2,
        num_momentum: int = 2,
        num_arbitrageurs: int = 1,
        num_noise_traders: int = 3,
        num_fundamental: int = 1,
        num_marl_agents: int = 4,
        num_assets: int = 4,
        evolutionary_dynamics: Optional[EvolutionaryDynamics] = None,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        evolution_interval: int = 100,
        seed: Optional[int] = None,
    ):
        self.num_assets = num_assets
        self.evolution_interval = evolution_interval
        self.rng = np.random.default_rng(seed)

        self.evo_dynamics = evolutionary_dynamics or EvolutionaryDynamics()
        self.fitness_evaluator = fitness_evaluator or FitnessEvaluator()

        # Create scripted agents
        self.scripted_agents: List[ScriptedAgent] = []
        agent_id = 0

        for _ in range(num_market_makers):
            self.scripted_agents.append(
                MarketMakerAgent(agent_id, num_assets, seed=seed + agent_id if seed else None)
            )
            agent_id += 1

        for _ in range(num_momentum):
            self.scripted_agents.append(
                MomentumAgent(agent_id, num_assets, seed=seed + agent_id if seed else None)
            )
            agent_id += 1

        for _ in range(num_arbitrageurs):
            self.scripted_agents.append(
                ArbitrageurAgent(agent_id, num_assets, seed=seed + agent_id if seed else None)
            )
            agent_id += 1

        for _ in range(num_noise_traders):
            self.scripted_agents.append(
                NoiseTraderAgent(agent_id, num_assets, seed=seed + agent_id if seed else None)
            )
            agent_id += 1

        for _ in range(num_fundamental):
            self.scripted_agents.append(
                FundamentalValueAgent(agent_id, num_assets, seed=seed + agent_id if seed else None)
            )
            agent_id += 1

        # MARL agent slots (filled externally)
        self.marl_agent_ids: List[int] = list(range(agent_id, agent_id + num_marl_agents))
        self.marl_agents: List = []  # to be filled by training code

        # Total agents
        self.total_agents = len(self.scripted_agents) + num_marl_agents
        self.num_marl_agents = num_marl_agents

        # Fitness tracking
        self._fitness_history: Dict[int, List[float]] = {
            a.agent_id: [] for a in self.scripted_agents
        }
        self._population_fractions: np.ndarray = np.ones(len(self.scripted_agents)) / max(len(self.scripted_agents), 1)

        self._step_count = 0

    def register_marl_agents(self, agents: List) -> None:
        """Register RL agents with the population."""
        self.marl_agents = agents
        for agent in agents:
            self._fitness_history[agent.agent_id] = []

    def get_scripted_actions(
        self,
        obs_list: Optional[List[np.ndarray]] = None,
        prices: Optional[np.ndarray] = None,
        inventories: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Get actions from all scripted agents."""
        actions = []
        for i, agent in enumerate(self.scripted_agents):
            obs = obs_list[i] if obs_list is not None else np.zeros(10)
            inv = inventories[i] if inventories is not None else None
            action = agent.select_action(obs, prices=prices, inventory=inv)
            actions.append(action)
            agent.step()
        return actions

    def update_fitness(
        self,
        agent_id: int,
        pnl_delta: float,
        equity_history: Optional[List[float]] = None,
    ) -> None:
        """Update fitness for an agent."""
        if agent_id in self._fitness_history:
            self._fitness_history[agent_id].append(pnl_delta)

        # Find scripted agent
        for agent in self.scripted_agents:
            if agent.agent_id == agent_id:
                agent.update_fitness(pnl_delta)
                break

    def evolutionary_step(self, force: bool = False) -> bool:
        """
        Run evolutionary dynamics step.
        Returns True if evolution occurred.
        """
        self._step_count += 1
        if not force and self._step_count % self.evolution_interval != 0:
            return False

        fitnesses = np.array([a.fitness for a in self.scripted_agents])

        if len(fitnesses) == 0:
            return False

        # Update population fractions via replicator dynamics
        self._population_fractions = self.evo_dynamics.replicator_select(
            fitnesses, self._population_fractions
        )

        # Replace worst performers
        n = len(self.scripted_agents)
        if n < 2:
            return True

        sorted_by_fitness = np.argsort(fitnesses)
        n_replace = max(1, int(n * 0.2))  # replace 20%
        worst = sorted_by_fitness[:n_replace]
        parents = self.evo_dynamics.select_parents(fitnesses, n_replace)

        for w_idx, p_idx in zip(worst, parents):
            agent_to_replace = self.scripted_agents[w_idx]
            parent_agent = self.scripted_agents[p_idx]

            # Inherit type with mutation
            agent_to_replace.agent_type = parent_agent.agent_type
            agent_to_replace._pnl_history.clear()
            agent_to_replace.fitness = 0.0

        logger.debug(
            f"Evolutionary step {self._step_count}: "
            f"fractions={self._population_fractions.round(3)}"
        )
        return True

    def get_population_stats(self) -> Dict[str, Any]:
        """Return population-level statistics."""
        type_counts: Dict[str, int] = collections.defaultdict(int)
        for agent in self.scripted_agents:
            type_counts[agent.agent_type] += 1

        fitnesses = np.array([a.fitness for a in self.scripted_agents])

        return {
            "total_agents": self.total_agents,
            "type_distribution": dict(type_counts),
            "mean_fitness": float(fitnesses.mean()) if len(fitnesses) > 0 else 0.0,
            "max_fitness": float(fitnesses.max()) if len(fitnesses) > 0 else 0.0,
            "min_fitness": float(fitnesses.min()) if len(fitnesses) > 0 else 0.0,
            "population_fractions": self._population_fractions.tolist(),
            "step": self._step_count,
        }

    def reset(self) -> None:
        """Reset all agents to initial state."""
        for agent in self.scripted_agents:
            agent._pnl_history.clear()
            agent.fitness = 0.0
            agent._step_count = 0
        self._step_count = 0


# ---------------------------------------------------------------------------
# Population evolution logger
# ---------------------------------------------------------------------------

class PopulationEvolutionLogger:
    """Logs and visualizes population evolution over time."""

    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
        self._snapshots: List[Dict] = []
        self._step = 0

    def record(self, population: AgentPopulation) -> None:
        self._step += 1
        if self._step % self.log_interval != 0:
            return
        snapshot = population.get_population_stats()
        snapshot["timestamp"] = self._step
        self._snapshots.append(snapshot)

    def get_history(self) -> List[Dict]:
        return self._snapshots.copy()

    def get_fitness_evolution(self) -> Dict[str, List[float]]:
        result: Dict[str, List[float]] = collections.defaultdict(list)
        for s in self._snapshots:
            result["mean_fitness"].append(s.get("mean_fitness", 0.0))
            result["max_fitness"].append(s.get("max_fitness", 0.0))
            result["timestamps"].append(s.get("timestamp", 0))
        return dict(result)


__all__ = [
    "AGENT_TYPE_MARKET_MAKER", "AGENT_TYPE_MOMENTUM", "AGENT_TYPE_ARBITRAGEUR",
    "AGENT_TYPE_NOISE", "AGENT_TYPE_MARL", "AGENT_TYPE_FUNDAMENTAL",
    "ScriptedAgent", "MarketMakerAgent", "MomentumAgent", "ArbitrageurAgent",
    "NoiseTraderAgent", "FundamentalValueAgent",
    "FitnessEvaluator", "EvolutionaryDynamics",
    "AgentPopulation", "PopulationEvolutionLogger",
]
