"""
Meta-Reward Co-Evolution: evolve reward functions for RL trading agents.

Instead of hand-tuning the RL reward function (Sharpe? Sortino? Calmar?),
this module evolves the *weights* of a multi-component reward function
using a genetic algorithm. Each reward genome is evaluated by:
  1. Training a PPO agent on the evolved reward function
  2. Testing that agent on a HOLDOUT market regime it has never seen
  3. Using actual PnL / Sortino on the holdout as the GA fitness

This prevents reward hacking: the GA cannot game the fitness metric because
the fitness is computed on unseen data using a fixed gold-standard metric.

Architecture:
  GA (evolves reward weights) -> PPO (trains on evolved reward) -> Holdout (evaluates real PnL)

Integration points:
  - ml.genetic.genome.StrategyGenome: base genome class
  - ml.genetic.genome.GenomeFactory: genome creation
  - ml.genetic.genome.ParamRange / ParamType: parameter definitions
  - ml.genetic.fitness.compute_sharpe/sortino/max_drawdown/calmar: fitness metrics
  - ml.genetic.evolution.EvolutionConfig: evolution parameters
  - ml.genetic.population.Population, HallOfFame, SelectionOperator
  - ml.genetic.operators.AdaptiveMutationRate, ReproductionOperator
  - idea_engine.rl.ppo_trader: PPO agent
  - idea_engine.rl.market_env: MarketEnv
"""

from __future__ import annotations

import copy
import math
import random
import time
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .genome import StrategyGenome, GenomeFactory, ParamRange, ParamType
from .fitness import (
    compute_sharpe, compute_sortino, compute_max_drawdown,
    compute_calmar, BacktestResult,
)
from .population import Population, PopulationConfig, HallOfFame, SelectionOperator


# ---------------------------------------------------------------------------
# Reward Genome: encodes the weights for a multi-component reward function
# ---------------------------------------------------------------------------

# The reward components the GA can weight
REWARD_COMPONENTS = [
    ParamRange("w_pnl", ParamType.CONTINUOUS, 0.0, 5.0, default=1.0,
               description="Weight on raw PnL per step"),
    ParamRange("w_sharpe_local", ParamType.CONTINUOUS, 0.0, 3.0, default=0.5,
               description="Weight on rolling Sharpe contribution"),
    ParamRange("w_sortino_local", ParamType.CONTINUOUS, 0.0, 3.0, default=0.0,
               description="Weight on rolling Sortino contribution"),
    ParamRange("w_drawdown_penalty", ParamType.CONTINUOUS, 0.0, 5.0, default=1.0,
               description="Penalty weight on drawdown depth"),
    ParamRange("w_volatility_penalty", ParamType.CONTINUOUS, 0.0, 3.0, default=0.5,
               description="Penalty weight on realized volatility"),
    ParamRange("w_turnover_penalty", ParamType.CONTINUOUS, 0.0, 2.0, default=0.3,
               description="Penalty on excessive position changes"),
    ParamRange("w_holding_bonus", ParamType.CONTINUOUS, 0.0, 1.0, default=0.1,
               description="Bonus for holding profitable positions"),
    ParamRange("w_regime_alignment", ParamType.CONTINUOUS, 0.0, 2.0, default=0.0,
               description="Bonus for aligning with detected regime"),
    ParamRange("risk_aversion", ParamType.CONTINUOUS, 0.1, 10.0, default=2.0,
               description="Overall risk aversion scaling factor"),
    ParamRange("reward_clipping", ParamType.CONTINUOUS, 1.0, 20.0, default=5.0,
               description="Max absolute reward per step (clips outliers)"),
]


@dataclass
class RewardGenome:
    """
    Genome encoding the weights for a dynamic reward function.
    The GA evolves these weights; a PPO agent is trained on the resulting reward.
    """
    genome_id: str = ""
    weights: Dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0
    holdout_pnl: float = 0.0
    holdout_sharpe: float = 0.0
    holdout_sortino: float = 0.0
    holdout_max_dd: float = 0.0
    training_episodes: int = 0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.weights:
            for param in REWARD_COMPONENTS:
                self.weights[param.name] = param.default

    def compute_reward(
        self,
        pnl_step: float,
        position: float,
        prev_position: float,
        equity: float,
        peak_equity: float,
        returns_window: List[float],
        regime_signal: float = 0.0,
    ) -> float:
        """
        Compute the shaped reward for this genome's weight configuration.
        Called by the RL environment at each step.
        """
        w = self.weights
        reward = 0.0

        # 1. Raw PnL component
        reward += w.get("w_pnl", 1.0) * pnl_step

        # 2. Rolling Sharpe contribution (uses recent returns window)
        # Clamp local metrics to [-3, 3] to prevent gradient spikes
        if len(returns_window) >= 5:
            mean_r = sum(returns_window) / len(returns_window)
            var_r = sum((r - mean_r)**2 for r in returns_window) / len(returns_window)
            std_r = max(math.sqrt(var_r), 1e-6)
            sharpe_local = max(-3.0, min(3.0, mean_r / std_r))
            reward += w.get("w_sharpe_local", 0.0) * sharpe_local

            # 3. Rolling Sortino contribution
            downside = [min(r, 0) for r in returns_window]
            downside_var = sum(d**2 for d in downside) / len(returns_window)
            downside_std = max(math.sqrt(downside_var), 1e-6)
            sortino_local = max(-3.0, min(3.0, mean_r / downside_std))
            reward += w.get("w_sortino_local", 0.0) * sortino_local

        # 4. Drawdown penalty
        if peak_equity > 0:
            drawdown = (peak_equity - equity) / peak_equity
            reward -= w.get("w_drawdown_penalty", 1.0) * drawdown

        # 5. Volatility penalty (penalize erratic PnL, capped to prevent explosion)
        if len(returns_window) >= 3:
            recent = returns_window[-min(len(returns_window), 10):]
            vol = math.sqrt(sum(r**2 for r in recent) / len(recent)) if recent else 0.0
            vol = min(vol, 1.0)  # cap to prevent domination
            reward -= w.get("w_volatility_penalty", 0.5) * vol

        # 6. Turnover penalty (penalize excessive trading)
        position_change = abs(position - prev_position)
        reward -= w.get("w_turnover_penalty", 0.3) * position_change

        # 7. Holding bonus (reward holding profitable positions)
        if pnl_step > 0 and abs(position) > 0.1 and abs(position - prev_position) < 0.1:
            reward += w.get("w_holding_bonus", 0.1) * pnl_step

        # 8. Regime alignment bonus
        if abs(regime_signal) > 0.3:
            alignment = position * regime_signal  # positive if aligned
            reward += w.get("w_regime_alignment", 0.0) * max(alignment, 0)

        # 9. Risk aversion scaling (smooth, no discontinuity at 0)
        risk_aversion = w.get("risk_aversion", 2.0)
        if reward < 0:
            # Smooth scaling: use sqrt to dampen extreme penalty spikes
            reward *= (1.0 + (risk_aversion - 1.0) * min(abs(reward), 1.0))

        # 10. Clip to prevent extreme reward signals
        clip_val = w.get("reward_clipping", 5.0)
        reward = max(-clip_val, min(clip_val, reward))

        return reward

    def to_dict(self) -> dict:
        return {
            "genome_id": self.genome_id,
            "weights": self.weights,
            "fitness": self.fitness,
            "holdout_pnl": self.holdout_pnl,
            "holdout_sharpe": self.holdout_sharpe,
            "holdout_sortino": self.holdout_sortino,
            "holdout_max_dd": self.holdout_max_dd,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
        }


# ---------------------------------------------------------------------------
# Reward Genome Factory
# ---------------------------------------------------------------------------

class RewardGenomeFactory:
    """Create and manipulate RewardGenome instances."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"rg_{self._counter:05d}"

    def create_random(self) -> RewardGenome:
        """Create a genome with random reward weights."""
        weights = {}
        for param in REWARD_COMPONENTS:
            if param.param_type == ParamType.CONTINUOUS:
                weights[param.name] = self.rng.uniform(param.low, param.high)
            else:
                weights[param.name] = param.default
        return RewardGenome(genome_id=self._next_id(), weights=weights)

    def create_default(self) -> RewardGenome:
        """Create a genome with default (hand-tuned) weights."""
        return RewardGenome(genome_id=self._next_id())

    def mutate(self, genome: RewardGenome, mutation_rate: float = 0.2,
               sigma: float = 0.15) -> RewardGenome:
        """Gaussian mutation on reward weights."""
        new_weights = {}
        for param in REWARD_COMPONENTS:
            val = genome.weights.get(param.name, param.default)
            if self.rng.random() < mutation_rate:
                noise = self.rng.gauss(0, sigma * (param.high - param.low))
                val = param.clip(val + noise)
            new_weights[param.name] = val

        child = RewardGenome(
            genome_id=self._next_id(),
            weights=new_weights,
            parent_ids=[genome.genome_id],
        )
        return child

    def crossover(self, parent_a: RewardGenome, parent_b: RewardGenome) -> RewardGenome:
        """Uniform crossover: each weight randomly from parent A or B."""
        new_weights = {}
        for param in REWARD_COMPONENTS:
            if self.rng.random() < 0.5:
                new_weights[param.name] = parent_a.weights.get(param.name, param.default)
            else:
                new_weights[param.name] = parent_b.weights.get(param.name, param.default)

        child = RewardGenome(
            genome_id=self._next_id(),
            weights=new_weights,
            parent_ids=[parent_a.genome_id, parent_b.genome_id],
        )
        return child

    def blx_alpha_crossover(self, parent_a: RewardGenome, parent_b: RewardGenome,
                             alpha: float = 0.3) -> RewardGenome:
        """BLX-alpha crossover: sample from expanded range between parents."""
        new_weights = {}
        for param in REWARD_COMPONENTS:
            va = parent_a.weights.get(param.name, param.default)
            vb = parent_b.weights.get(param.name, param.default)
            lo = min(va, vb)
            hi = max(va, vb)
            span = hi - lo
            new_val = self.rng.uniform(lo - alpha * span, hi + alpha * span)
            new_weights[param.name] = param.clip(new_val)

        return RewardGenome(
            genome_id=self._next_id(),
            weights=new_weights,
            parent_ids=[parent_a.genome_id, parent_b.genome_id],
        )


# ---------------------------------------------------------------------------
# Lightweight PPO agent (inline, no heavy imports)
# ---------------------------------------------------------------------------

class _MiniMLP:
    """Tiny MLP for the inline PPO. 2 hidden layers, tanh activation."""

    def __init__(self, input_dim: int, hidden: int, output_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.w1 = rng.normal(0, scale1, (input_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, scale2, (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.w3 = rng.normal(0, scale2, (hidden, output_dim))
        self.b3 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.w1 + self.b1)
        h = np.tanh(h @ self.w2 + self.b2)
        return h @ self.w3 + self.b3

    def get_params(self) -> List[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def set_params(self, params: List[np.ndarray]) -> None:
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = params


class _InlinePPOAgent:
    """
    Lightweight PPO agent for meta-reward evaluation.
    Discrete actions: FLAT=0, LONG=1, SHORT=2.
    Trains quickly (no backprop -- uses REINFORCE with baseline for speed).
    """

    def __init__(self, obs_dim: int, n_actions: int = 3, lr: float = 1e-3,
                 gamma: float = 0.99, seed: int = 42):
        self.policy = _MiniMLP(obs_dim, 32, n_actions, seed)
        self.value = _MiniMLP(obs_dim, 32, 1, seed + 1)
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        logits = self.policy.forward(obs.reshape(1, -1))[0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum() + 1e-10
        return int(self.rng.choice(self.n_actions, p=probs))

    def train_episode(self, observations: List[np.ndarray], actions: List[int],
                      rewards: List[float]) -> float:
        """Train on a single episode using REINFORCE with baseline."""
        T = len(rewards)
        if T == 0:
            return 0.0

        # Compute returns
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        # Baseline: value function predictions
        obs_arr = np.array(observations[:T])
        baselines = self.value.forward(obs_arr).flatten()
        advantages = returns - baselines

        # Normalize advantages
        adv_std = max(float(np.std(advantages)), 1e-8)
        advantages = (advantages - advantages.mean()) / adv_std

        # Policy gradient update (REINFORCE)
        for t in range(T):
            obs = obs_arr[t:t+1]
            logits = self.policy.forward(obs)[0]
            probs = np.exp(logits - logits.max())
            probs /= probs.sum() + 1e-10

            # Gradient of log pi(a|s) * advantage
            grad = -probs.copy()
            grad[actions[t]] += 1.0
            grad *= advantages[t] * self.lr

            # Simple weight update on output layer only (fast)
            h = np.tanh(np.tanh(obs @ self.policy.w1 + self.policy.b1) @ self.policy.w2 + self.policy.b2)
            self.policy.w3 += h.T @ grad.reshape(1, -1)
            self.policy.b3 += grad

        # Value function update (MSE gradient)
        for t in range(T):
            obs = obs_arr[t:t+1]
            v_pred = self.value.forward(obs)[0, 0]
            v_error = returns[t] - v_pred
            h = np.tanh(np.tanh(obs @ self.value.w1 + self.value.b1) @ self.value.w2 + self.value.b2)
            self.value.w3 += self.lr * v_error * h.T
            self.value.b3[0] += self.lr * v_error

        return float(returns[0])


# ---------------------------------------------------------------------------
# Market Environment Wrapper (uses price data directly)
# ---------------------------------------------------------------------------

@dataclass
class MarketSlice:
    """A slice of market data for training or evaluation."""
    prices: np.ndarray        # (T,) close prices
    returns: np.ndarray       # (T-1,) log returns
    volumes: np.ndarray       # (T,) volumes (optional, zeros ok)
    regime_labels: np.ndarray  # (T,) regime signals (-1 to +1)
    name: str = "unknown"


class MetaRewardEnv:
    """
    Simple market environment that uses a RewardGenome to shape rewards.
    """

    def __init__(self, data: MarketSlice, transaction_cost_bps: float = 10.0):
        self.data = data
        self.tc = transaction_cost_bps / 10000.0
        self.T = len(data.returns)
        self.obs_dim = 6  # [return, vol_5, vol_21, position, drawdown, regime]
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 21  # skip warmup
        self.position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.returns_history: List[float] = []
        return self._obs()

    def _obs(self) -> np.ndarray:
        t = self.t
        ret = float(self.data.returns[min(t, self.T - 1)])
        vol_5 = float(np.std(self.data.returns[max(0, t-5):t]) * math.sqrt(252)) if t >= 5 else 0.15
        vol_21 = float(np.std(self.data.returns[max(0, t-21):t]) * math.sqrt(252)) if t >= 21 else 0.15
        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-10)
        regime = float(self.data.regime_labels[min(t, len(self.data.regime_labels) - 1)])
        return np.array([ret, vol_5, vol_21, self.position, dd, regime])

    def step(self, action: int, reward_genome: RewardGenome) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action and compute reward from the genome's reward function.
        action: 0=FLAT, 1=LONG, 2=SHORT
        """
        # Map action to position
        target = {0: 0.0, 1: 1.0, 2: -1.0}.get(action, 0.0)
        prev_position = self.position

        # Transaction cost on position change
        cost = abs(target - self.position) * self.tc
        self.position = target

        # PnL from market return
        if self.t < self.T:
            market_return = float(self.data.returns[self.t])
        else:
            market_return = 0.0

        pnl_step = self.position * market_return - cost
        self.equity *= (1 + pnl_step)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.returns_history.append(pnl_step)

        # Compute shaped reward using the genome
        reward = reward_genome.compute_reward(
            pnl_step=pnl_step,
            position=self.position,
            prev_position=prev_position,
            equity=self.equity,
            peak_equity=self.peak_equity,
            returns_window=self.returns_history[-21:],
            regime_signal=float(self.data.regime_labels[min(self.t, len(self.data.regime_labels) - 1)]),
        )

        self.t += 1
        done = self.t >= self.T
        obs = self._obs()

        return obs, reward, done


# ---------------------------------------------------------------------------
# Holdout Evaluator: gold-standard fitness on unseen data
# ---------------------------------------------------------------------------

def evaluate_agent_on_holdout(
    agent: _InlinePPOAgent,
    holdout_data: MarketSlice,
    transaction_cost_bps: float = 10.0,
) -> Dict[str, float]:
    """
    Evaluate a trained agent on holdout data using REAL metrics.
    No reward genome involved -- this is the ground truth.
    """
    tc = transaction_cost_bps / 10000.0
    T = len(holdout_data.returns)
    position = 0.0
    equity = 1.0
    peak = 1.0
    returns_list = []

    env = MetaRewardEnv(holdout_data, transaction_cost_bps)
    obs = env.reset()

    for t in range(21, T):
        action = agent.act(obs)
        target = {0: 0.0, 1: 1.0, 2: -1.0}.get(action, 0.0)
        cost = abs(target - position) * tc
        position = target

        market_ret = float(holdout_data.returns[t])
        pnl = position * market_ret - cost
        equity *= (1 + pnl)
        peak = max(peak, equity)
        returns_list.append(pnl)

        # Update obs manually for agent
        vol_5 = float(np.std(holdout_data.returns[max(0, t-5):t]) * math.sqrt(252)) if t >= 5 else 0.15
        vol_21 = float(np.std(holdout_data.returns[max(0, t-21):t]) * math.sqrt(252)) if t >= 21 else 0.15
        dd = (peak - equity) / max(peak, 1e-10)
        regime = float(holdout_data.regime_labels[min(t, len(holdout_data.regime_labels) - 1)])
        obs = np.array([market_ret, vol_5, vol_21, position, dd, regime])

    total_return = equity - 1.0
    sharpe = compute_sharpe(returns_list) if returns_list else 0.0
    sortino = compute_sortino(returns_list) if returns_list else 0.0
    max_dd = compute_max_drawdown(returns_list) if returns_list else 0.0
    calmar = compute_calmar(returns_list) if returns_list else 0.0

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "n_trades": sum(1 for i in range(1, len(returns_list)) if returns_list[i] != returns_list[i-1]),
        "equity_final": equity,
    }


# ---------------------------------------------------------------------------
# Meta-Reward Co-Evolver
# ---------------------------------------------------------------------------

@dataclass
class MetaRewardConfig:
    """Configuration for meta-reward evolution."""
    # GA parameters
    population_size: int = 30
    n_generations: int = 50
    elite_count: int = 3
    mutation_rate: float = 0.25
    mutation_sigma: float = 0.15
    crossover_rate: float = 0.7
    tournament_size: int = 3

    # RL training per genome
    training_episodes: int = 50
    agent_lr: float = 1e-3
    agent_gamma: float = 0.99

    # Data split
    train_fraction: float = 0.6
    holdout_fraction: float = 0.2  # remaining 0.2 is validation

    # Fitness
    fitness_metric: str = "sortino"  # which holdout metric is the GA fitness
    min_trades: int = 10  # genomes producing < this many trades get penalized

    # Misc
    seed: int = 42
    verbose: bool = True
    checkpoint_dir: str = ""


class MetaRewardEvolver:
    """
    Orchestrates the co-evolution of reward functions and RL agents.

    The loop:
      1. GA maintains a population of RewardGenome (reward function weights)
      2. For each genome: train a PPO agent using that genome's reward
      3. Evaluate the trained agent on HOLDOUT data using real PnL metrics
      4. Use holdout performance as the GA fitness
      5. Evolve: select, crossover, mutate the reward genomes
      6. Repeat

    The key insight: the GA fitness is NOT the reward the agent received.
    It is the agent's actual trading performance on unseen data.
    This prevents reward hacking.
    """

    def __init__(self, config: MetaRewardConfig, price_data: np.ndarray,
                 regime_data: Optional[np.ndarray] = None,
                 volume_data: Optional[np.ndarray] = None):
        self.config = config
        self.rng = random.Random(config.seed)
        self.factory = RewardGenomeFactory(config.seed)

        # Split data into train / holdout / validation
        T = len(price_data)
        train_end = int(T * config.train_fraction)
        holdout_end = int(T * (config.train_fraction + config.holdout_fraction))

        returns = np.diff(np.log(price_data + 1e-10))
        if regime_data is None:
            regime_data = np.zeros(T)
        if volume_data is None:
            volume_data = np.ones(T)

        self.train_data = MarketSlice(
            prices=price_data[:train_end],
            returns=returns[:train_end - 1],
            volumes=volume_data[:train_end],
            regime_labels=regime_data[:train_end],
            name="train",
        )
        self.holdout_data = MarketSlice(
            prices=price_data[train_end:holdout_end],
            returns=returns[train_end - 1:holdout_end - 1],
            volumes=volume_data[train_end:holdout_end],
            regime_labels=regime_data[train_end:holdout_end],
            name="holdout",
        )
        self.validation_data = MarketSlice(
            prices=price_data[holdout_end:],
            returns=returns[holdout_end - 1:],
            volumes=volume_data[holdout_end:],
            regime_labels=regime_data[holdout_end:],
            name="validation",
        )

        # Population
        self.population: List[RewardGenome] = []
        self.hall_of_fame: List[RewardGenome] = []
        self.generation_history: List[Dict] = []

    def initialize_population(self) -> None:
        """Create initial population: mostly random + one default."""
        self.population = [self.factory.create_default()]
        for _ in range(self.config.population_size - 1):
            self.population.append(self.factory.create_random())

        if self.config.verbose:
            print(f"Initialized population: {len(self.population)} reward genomes")
            print(f"  Train data: {len(self.train_data.returns)} bars")
            print(f"  Holdout data: {len(self.holdout_data.returns)} bars")
            print(f"  Validation data: {len(self.validation_data.returns)} bars")

    def evaluate_genome(self, genome: RewardGenome) -> float:
        """
        The core fitness function:
        1. Train a PPO agent using this genome's reward function
        2. Evaluate on holdout using real metrics
        3. Return holdout fitness

        This is the function that prevents reward hacking:
        the GA cannot game the holdout metric because it never sees
        the holdout data during training.
        """
        # 1. Create fresh agent
        agent = _InlinePPOAgent(
            obs_dim=6,
            n_actions=3,
            lr=self.config.agent_lr,
            gamma=self.config.agent_gamma,
            seed=self.config.seed + hash(genome.genome_id) % 10000,
        )

        # 2. Train on training data with evolved reward
        env = MetaRewardEnv(self.train_data, transaction_cost_bps=10.0)

        for episode in range(self.config.training_episodes):
            obs = env.reset()
            observations, actions, rewards = [], [], []

            done = False
            while not done:
                action = agent.act(obs)
                observations.append(obs.copy())
                actions.append(action)

                obs, reward, done = env.step(action, genome)
                rewards.append(reward)

            agent.train_episode(observations, actions, rewards)

        genome.training_episodes = self.config.training_episodes

        # 3. Evaluate on HOLDOUT with REAL metrics (no reward genome involved)
        holdout_result = evaluate_agent_on_holdout(agent, self.holdout_data)

        # 4. Record results
        genome.holdout_pnl = holdout_result["total_return"]
        genome.holdout_sharpe = holdout_result["sharpe"]
        genome.holdout_sortino = holdout_result["sortino"]
        genome.holdout_max_dd = holdout_result["max_drawdown"]

        # 5. Fitness: the holdout metric the GA optimizes
        fitness = holdout_result.get(self.config.fitness_metric, holdout_result["sortino"])

        # Penalty for too few trades (reward function that encourages inaction)
        if holdout_result["n_trades"] < self.config.min_trades:
            fitness *= 0.1

        # Penalty for excessive drawdown
        if holdout_result["max_drawdown"] > 0.5:
            fitness *= 0.5

        genome.fitness = fitness
        return fitness

    def _tournament_select(self) -> RewardGenome:
        """Tournament selection."""
        candidates = self.rng.sample(self.population, min(self.config.tournament_size, len(self.population)))
        return max(candidates, key=lambda g: g.fitness)

    def evolve_generation(self) -> Dict:
        """Run one generation of evolution."""
        gen_start = time.time()

        # Evaluate all genomes
        for genome in self.population:
            self.evaluate_genome(genome)

        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Stats
        fitnesses = [g.fitness for g in self.population]
        best = self.population[0]
        gen_stats = {
            "best_fitness": best.fitness,
            "best_holdout_pnl": best.holdout_pnl,
            "best_holdout_sharpe": best.holdout_sharpe,
            "best_holdout_sortino": best.holdout_sortino,
            "best_holdout_max_dd": best.holdout_max_dd,
            "mean_fitness": sum(fitnesses) / len(fitnesses),
            "std_fitness": float(np.std(fitnesses)),
            "best_weights": best.weights.copy(),
            "elapsed_seconds": time.time() - gen_start,
        }

        # Update hall of fame
        if not self.hall_of_fame or best.fitness > self.hall_of_fame[0].fitness:
            self.hall_of_fame.insert(0, copy.deepcopy(best))
            self.hall_of_fame = self.hall_of_fame[:10]

        # Create next generation
        next_pop = []

        # Elitism: keep top N unchanged
        for i in range(self.config.elite_count):
            elite = copy.deepcopy(self.population[i])
            elite.genome_id = self.factory._next_id()
            next_pop.append(elite)

        # Fill rest with crossover + mutation
        while len(next_pop) < self.config.population_size:
            if self.rng.random() < self.config.crossover_rate:
                parent_a = self._tournament_select()
                parent_b = self._tournament_select()
                child = self.factory.blx_alpha_crossover(parent_a, parent_b)
            else:
                parent = self._tournament_select()
                child = copy.deepcopy(parent)
                child.genome_id = self.factory._next_id()

            # Mutate
            child = self.factory.mutate(child, self.config.mutation_rate, self.config.mutation_sigma)
            child.generation = len(self.generation_history) + 1
            next_pop.append(child)

        self.population = next_pop
        self.generation_history.append(gen_stats)

        return gen_stats

    def run(self) -> Dict:
        """
        Run the full meta-reward evolution.
        Returns the best reward genome and validation results.
        """
        self.initialize_population()

        if self.config.verbose:
            print(f"\nStarting meta-reward evolution: {self.config.n_generations} generations")
            print(f"  Population: {self.config.population_size}")
            print(f"  Training episodes per genome: {self.config.training_episodes}")
            print(f"  Fitness metric: {self.config.fitness_metric}")
            print("-" * 60)

        for gen in range(self.config.n_generations):
            stats = self.evolve_generation()

            if self.config.verbose:
                print(
                    f"  Gen {gen+1:3d} | "
                    f"Best {self.config.fitness_metric}: {stats['best_fitness']:+.3f} | "
                    f"Holdout PnL: {stats['best_holdout_pnl']:+.1%} | "
                    f"Sharpe: {stats['best_holdout_sharpe']:.2f} | "
                    f"MaxDD: {stats['best_holdout_max_dd']:.1%} | "
                    f"Mean: {stats['mean_fitness']:+.3f} | "
                    f"{stats['elapsed_seconds']:.1f}s"
                )

        # Final: validate best genome on validation data
        best = self.hall_of_fame[0] if self.hall_of_fame else self.population[0]

        if self.config.verbose:
            print("\n" + "=" * 60)
            print("EVOLUTION COMPLETE")
            print(f"  Best genome: {best.genome_id}")
            print(f"  Best weights:")
            for k, v in best.weights.items():
                print(f"    {k:25s} = {v:.4f}")

            # Validate on unseen data
            print(f"\n  Validating on unseen data ({len(self.validation_data.returns)} bars)...")

        agent = _InlinePPOAgent(obs_dim=6, n_actions=3, lr=self.config.agent_lr,
                                 gamma=self.config.agent_gamma, seed=self.config.seed)
        env = MetaRewardEnv(self.train_data)
        for _ in range(self.config.training_episodes):
            obs = env.reset()
            observations, actions, rewards = [], [], []
            done = False
            while not done:
                action = agent.act(obs)
                observations.append(obs.copy())
                actions.append(action)
                obs, reward, done = env.step(action, best)
                rewards.append(reward)
            agent.train_episode(observations, actions, rewards)

        validation_result = evaluate_agent_on_holdout(agent, self.validation_data)

        if self.config.verbose:
            print(f"  Validation PnL: {validation_result['total_return']:+.1%}")
            print(f"  Validation Sharpe: {validation_result['sharpe']:.2f}")
            print(f"  Validation Sortino: {validation_result['sortino']:.2f}")
            print(f"  Validation MaxDD: {validation_result['max_drawdown']:.1%}")

        return {
            "best_genome": best.to_dict(),
            "holdout_results": {
                "pnl": best.holdout_pnl,
                "sharpe": best.holdout_sharpe,
                "sortino": best.holdout_sortino,
                "max_dd": best.holdout_max_dd,
            },
            "validation_results": validation_result,
            "generation_history": self.generation_history,
            "hall_of_fame": [g.to_dict() for g in self.hall_of_fame],
        }

    def save(self, filepath: str) -> None:
        """Save evolution results to JSON."""
        result = {
            "config": {
                "population_size": self.config.population_size,
                "n_generations": self.config.n_generations,
                "training_episodes": self.config.training_episodes,
                "fitness_metric": self.config.fitness_metric,
            },
            "hall_of_fame": [g.to_dict() for g in self.hall_of_fame],
            "generation_history": self.generation_history,
        }
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)
