"""
agent_zoo.py
============
Pre-defined agent zoo for the Hyper-Agent MARL ecosystem.

Contains:
  - Hand-crafted benchmark agents (momentum, mean-reversion, random, VWAP executor)
  - Agent performance leaderboard
  - Agent behaviour fingerprinting
  - Zoo manager for registering/retrieving agents
  - Evaluation framework for comparing zoo agents to trained MARL policies
"""

from __future__ import annotations

import abc
import dataclasses
import enum
import logging
import math
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base zoo agent
# ---------------------------------------------------------------------------

class BaseZooAgent(abc.ABC):
    """Abstract base for all pre-defined zoo agents."""

    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self._n_steps = 0
        self._episode_return = 0.0
        self._performance_history: deque = deque(maxlen=100)

    @abc.abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return action vector given observation."""
        ...

    def reset(self) -> None:
        """Reset per-episode state."""
        self._n_steps = 0

    def record_return(self, total_return: float) -> None:
        self._performance_history.append(total_return)

    @property
    def mean_return(self) -> float:
        if not self._performance_history:
            return 0.0
        return float(np.mean(self._performance_history))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, name={self.name})"


# ---------------------------------------------------------------------------
# Benchmark agents
# ---------------------------------------------------------------------------

class RandomAgent(BaseZooAgent):
    """Uniformly random action agent."""

    def __init__(self, act_dim: int = 10, seed: int = 0):
        super().__init__("random", "Random Agent")
        self.act_dim = act_dim
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._n_steps += 1
        return self._rng.uniform(-1.0, 1.0, self.act_dim).astype(np.float32)


class NopAgent(BaseZooAgent):
    """Agent that always takes the no-op action."""

    def __init__(self, act_dim: int = 10):
        super().__init__("nop", "No-Op Agent")
        self.act_dim = act_dim

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = np.zeros(self.act_dim, dtype=np.float32)
        action[2] = 1.0   # NOP order type
        return action


class MomentumAgent(BaseZooAgent):
    """
    Pure momentum agent: buys when price is rising, sells when falling.
    Uses exponential moving average crossover.
    """

    def __init__(self, act_dim: int = 10,
                 fast_window: int = 5, slow_window: int = 20,
                 position_size: float = 0.5):
        super().__init__("momentum", "Momentum Agent")
        self.act_dim = act_dim
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.position_size = position_size
        self._price_history: deque = deque(maxlen=slow_window + 1)

    def reset(self) -> None:
        super().reset()
        self._price_history.clear()

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._n_steps += 1
        # Extract mid price from obs (assume obs[0] is normalised mid price)
        mid_price = float(obs[0]) * 1000.0   # denormalise
        self._price_history.append(mid_price)

        action = np.zeros(self.act_dim, dtype=np.float32)

        if len(self._price_history) < self.slow_window:
            action[2] = 1.0   # NOP while warming up
            return action

        prices = np.array(list(self._price_history))
        fast_ma = prices[-self.fast_window:].mean()
        slow_ma = prices[-self.slow_window:].mean()

        if fast_ma > slow_ma * 1.001:   # Buy signal
            action[0] = 1.0   # order type: limit
            action[4] = 1.0   # side: buy
            action[5] = 0.0   # price offset: at market
            action[6] = self.position_size
        elif fast_ma < slow_ma * 0.999:  # Sell signal
            action[0] = 1.0   # order type: limit
            action[4] = -1.0  # side: sell
            action[5] = 0.0
            action[6] = self.position_size
        else:
            action[2] = 1.0   # NOP

        return action


class MeanReversionAgent(BaseZooAgent):
    """
    Mean reversion agent: buys when price is below VWAP, sells above.
    Uses Bollinger Band signals.
    """

    def __init__(self, act_dim: int = 10,
                 window: int = 20, n_std: float = 2.0,
                 position_size: float = 0.4):
        super().__init__("mean_reversion", "Mean Reversion Agent")
        self.act_dim = act_dim
        self.window = window
        self.n_std = n_std
        self.position_size = position_size
        self._price_history: deque = deque(maxlen=window + 1)

    def reset(self) -> None:
        super().reset()
        self._price_history.clear()

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._n_steps += 1
        mid_price = float(obs[0]) * 1000.0
        self._price_history.append(mid_price)

        action = np.zeros(self.act_dim, dtype=np.float32)

        if len(self._price_history) < self.window:
            action[2] = 1.0
            return action

        prices = np.array(list(self._price_history))
        mean = prices.mean()
        std = prices.std() + 1e-9

        upper_band = mean + self.n_std * std
        lower_band = mean - self.n_std * std

        if mid_price < lower_band:   # Oversold: buy
            action[0] = 1.0
            action[4] = 1.0
            action[6] = self.position_size * (1 + (lower_band - mid_price) / std)
        elif mid_price > upper_band:  # Overbought: sell
            action[0] = 1.0
            action[4] = -1.0
            action[6] = self.position_size * (1 + (mid_price - upper_band) / std)
        else:
            action[2] = 1.0

        action[6] = float(np.clip(action[6], 0, 1))
        return action


class VWAPExecutorAgent(BaseZooAgent):
    """
    VWAP execution agent: distributes a target volume uniformly over the episode,
    executing proportional to historical volume patterns.
    """

    def __init__(self, act_dim: int = 10,
                 target_volume: float = 100.0,
                 episode_len: int = 2000,
                 side: int = 1,   # +1 buy, -1 sell
                 urgency: float = 0.5):
        super().__init__("vwap_executor", "VWAP Executor Agent")
        self.act_dim = act_dim
        self.target_volume = target_volume
        self.episode_len = episode_len
        self.side = side
        self.urgency = urgency
        self._executed_volume = 0.0
        self._step = 0

    def reset(self) -> None:
        super().reset()
        self._executed_volume = 0.0
        self._step = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._n_steps += 1
        self._step += 1
        action = np.zeros(self.act_dim, dtype=np.float32)

        remaining_volume = self.target_volume - self._executed_volume
        remaining_steps = max(1, self.episode_len - self._step)

        if remaining_volume <= 0:
            action[2] = 1.0   # Done
            return action

        # Execute proportional to time remaining
        volume_this_step = remaining_volume / remaining_steps
        volume_this_step *= (1.0 + self.urgency * (1 - self._step / self.episode_len))
        volume_this_step = float(np.clip(volume_this_step / self.target_volume, 0, 1))

        action[1] = 1.0   # market order
        action[4] = float(self.side)
        action[6] = volume_this_step
        self._executed_volume += volume_this_step * self.target_volume

        return action


class TwapExecutorAgent(BaseZooAgent):
    """
    TWAP execution agent: executes equal slices at uniform time intervals.
    """

    def __init__(self, act_dim: int = 10,
                 target_volume: float = 100.0,
                 episode_len: int = 2000,
                 side: int = 1,
                 n_slices: int = 20):
        super().__init__("twap_executor", "TWAP Executor Agent")
        self.act_dim = act_dim
        self.target_volume = target_volume
        self.episode_len = episode_len
        self.side = side
        self.n_slices = n_slices
        self.slice_volume = target_volume / n_slices
        self.slice_interval = episode_len // n_slices
        self._step = 0

    def reset(self) -> None:
        super().reset()
        self._step = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._n_steps += 1
        self._step += 1
        action = np.zeros(self.act_dim, dtype=np.float32)

        if self._step % self.slice_interval == 0:
            action[1] = 1.0   # market order
            action[4] = float(self.side)
            action[6] = min(1.0, self.slice_volume / self.target_volume)
        else:
            action[2] = 1.0   # NOP

        return action


class SimpleMarketMakerAgent(BaseZooAgent):
    """
    Hand-crafted market maker: posts limit orders on both sides of the spread.
    Adjusts quote aggressiveness based on inventory.
    """

    def __init__(self, act_dim: int = 10,
                 base_spread: float = 0.01,
                 max_inventory: float = 100.0,
                 inventory_skew_factor: float = 0.5):
        super().__init__("simple_mm", "Simple Market Maker")
        self.act_dim = act_dim
        self.base_spread = base_spread
        self.max_inventory = max_inventory
        self.inventory_skew_factor = inventory_skew_factor
        self._inventory = 0.0
        self._tick = 0
        self._post_bid = True

    def reset(self) -> None:
        super().reset()
        self._inventory = 0.0
        self._tick = 0
        self._post_bid = True

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._n_steps += 1
        self._tick += 1
        action = np.zeros(self.act_dim, dtype=np.float32)

        # Extract inventory from obs (assume obs[-4] is inventory)
        if len(obs) > 4:
            self._inventory = float(obs[-4]) * 1000.0   # rough denorm

        inv_ratio = self._inventory / self.max_inventory
        skew = -inv_ratio * self.inventory_skew_factor

        # Alternate between bid and ask to avoid one-sided accumulation
        if self._tick % 2 == 0:
            side = 1 + (1 if skew < 0 else 0)   # prefer buying if short
            action[0] = 1.0
            action[4] = 1.0 if self._tick % 4 < 2 else -1.0
            action[5] = -self.base_spread / 2 + skew  # bid slightly below mid
            action[6] = 0.3
        else:
            action[2] = 1.0   # NOP on alternate ticks

        return action


class AdaptiveMomentumAgent(BaseZooAgent):
    """
    Momentum agent with adaptive position sizing based on volatility.
    """

    def __init__(self, act_dim: int = 10,
                 fast_window: int = 5, slow_window: int = 20,
                 vol_window: int = 10):
        super().__init__("adaptive_momentum", "Adaptive Momentum Agent")
        self.act_dim = act_dim
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.vol_window = vol_window
        self._price_history: deque = deque(maxlen=slow_window + 1)

    def reset(self) -> None:
        super().reset()
        self._price_history.clear()

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._n_steps += 1
        mid_price = float(obs[0]) * 1000.0
        self._price_history.append(mid_price)

        action = np.zeros(self.act_dim, dtype=np.float32)
        if len(self._price_history) < self.slow_window:
            action[2] = 1.0
            return action

        prices = np.array(list(self._price_history))
        fast_ma = prices[-self.fast_window:].mean()
        slow_ma = prices.mean()
        recent_vol = prices[-self.vol_window:].std() / (prices.mean() + 1e-9)

        signal = (fast_ma - slow_ma) / (slow_ma + 1e-9)
        # Reduce size in high-vol regime
        vol_scale = max(0.1, 1.0 - recent_vol * 100)
        size = float(np.clip(abs(signal) * vol_scale * 5, 0, 1))

        if abs(signal) > 0.001:
            action[0] = 1.0
            action[4] = 1.0 if signal > 0 else -1.0
            action[6] = size
        else:
            action[2] = 1.0

        return action


# ---------------------------------------------------------------------------
# Neural zoo agent (wraps trained policy)
# ---------------------------------------------------------------------------

class NeuralZooAgent(BaseZooAgent):
    """Wraps a trained PyTorch policy network as a zoo agent."""

    def __init__(self, policy: nn.Module, agent_id: str = "neural",
                 name: str = "Neural Agent", device: str = "cpu"):
        super().__init__(agent_id, name)
        self.policy = policy.to(device)
        self.device = device
        self.policy.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._n_steps += 1
        obs_t = torch.tensor(obs, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy.act(obs_t, deterministic=False)
        return action.cpu().numpy()[0]

    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy.act(obs_t, deterministic=True)
        return action.cpu().numpy()[0]


# ---------------------------------------------------------------------------
# Performance leaderboard
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LeaderboardEntry:
    agent_id: str
    name: str
    mean_return: float
    std_return: float
    sharpe: float
    max_drawdown: float
    n_episodes: int
    win_rate_vs_random: float = 0.5
    tags: List[str] = dataclasses.field(default_factory=list)

    def __lt__(self, other: "LeaderboardEntry") -> bool:
        return self.sharpe < other.sharpe


class AgentLeaderboard:
    """Tracks and ranks agent performance."""

    def __init__(self):
        self._entries: Dict[str, LeaderboardEntry] = {}
        self._head_to_head: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "draws": 0}
        )

    def update(self, agent: BaseZooAgent,
                episode_returns: List[float],
                tags: Optional[List[str]] = None) -> LeaderboardEntry:
        if not episode_returns:
            return LeaderboardEntry(agent.agent_id, agent.name, 0, 0, 0, 0, 0)

        arr = np.array(episode_returns)
        sharpe = float(arr.mean() / (arr.std() + 1e-9))

        # Max drawdown
        cum = np.cumsum(arr)
        max_dd = float((np.maximum.accumulate(cum) - cum).max())

        entry = LeaderboardEntry(
            agent_id=agent.agent_id,
            name=agent.name,
            mean_return=float(arr.mean()),
            std_return=float(arr.std()),
            sharpe=sharpe,
            max_drawdown=max_dd,
            n_episodes=len(arr),
            tags=tags or [],
        )
        self._entries[agent.agent_id] = entry
        return entry

    def record_h2h(self, agent_a_id: str, agent_b_id: str,
                    a_wins: bool) -> None:
        key = (min(agent_a_id, agent_b_id), max(agent_a_id, agent_b_id))
        if a_wins:
            self._head_to_head[key]["wins"] += 1
        else:
            self._head_to_head[key]["losses"] += 1

    def ranked(self, by: str = "sharpe") -> List[LeaderboardEntry]:
        entries = list(self._entries.values())
        if by == "sharpe":
            entries.sort(key=lambda e: e.sharpe, reverse=True)
        elif by == "mean_return":
            entries.sort(key=lambda e: e.mean_return, reverse=True)
        elif by == "max_drawdown":
            entries.sort(key=lambda e: e.max_drawdown)
        return entries

    def print_leaderboard(self, top_k: int = 10) -> None:
        entries = self.ranked()[:top_k]
        print(f"{'Rank':4} {'Agent':25} {'Mean Ret':10} {'Sharpe':8} {'MaxDD':8} {'N_eps':6}")
        print("-" * 70)
        for i, e in enumerate(entries):
            print(f"{i+1:4} {e.name:25} {e.mean_return:+10.3f} {e.sharpe:8.3f} "
                  f"{e.max_drawdown:8.3f} {e.n_episodes:6}")


# ---------------------------------------------------------------------------
# Behaviour fingerprinter
# ---------------------------------------------------------------------------

class BehaviourFingerprinter:
    """
    Characterises agent behaviour via statistics on:
      - Action distribution (mean, std per action dim)
      - Trade frequency and size
      - Position-holding patterns
    """

    def __init__(self, n_canonical_obs: int = 200, obs_dim: int = 64, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._canonical_obs = self._rng.standard_normal((n_canonical_obs, obs_dim)).astype(np.float32)
        self._fingerprints: Dict[str, np.ndarray] = {}

    def fingerprint(self, agent: BaseZooAgent) -> np.ndarray:
        actions = np.array([agent.act(obs) for obs in self._canonical_obs])
        fp = np.concatenate([
            actions.mean(axis=0),
            actions.std(axis=0),
            actions.min(axis=0),
            actions.max(axis=0),
        ])
        self._fingerprints[agent.agent_id] = fp
        return fp

    def similarity(self, agent_a_id: str, agent_b_id: str) -> float:
        fp_a = self._fingerprints.get(agent_a_id)
        fp_b = self._fingerprints.get(agent_b_id)
        if fp_a is None or fp_b is None:
            return 0.0
        denom = (np.linalg.norm(fp_a) * np.linalg.norm(fp_b)) + 1e-9
        return float(np.dot(fp_a, fp_b) / denom)

    def cluster(self, n_clusters: int = 3) -> Dict[str, int]:
        """Simple K-means clustering of agent fingerprints."""
        ids = list(self._fingerprints.keys())
        if len(ids) < n_clusters:
            return {aid: i for i, aid in enumerate(ids)}

        fp_matrix = np.stack([self._fingerprints[aid] for aid in ids])
        # Random initialisation
        rng = np.random.default_rng(0)
        centres = fp_matrix[rng.choice(len(ids), n_clusters, replace=False)]

        for _ in range(100):
            dists = np.linalg.norm(fp_matrix[:, None, :] - centres[None, :, :], axis=-1)
            assignments = dists.argmin(axis=1)
            new_centres = np.array([
                fp_matrix[assignments == k].mean(axis=0) if (assignments == k).any()
                else centres[k]
                for k in range(n_clusters)
            ])
            if np.allclose(centres, new_centres, atol=1e-6):
                break
            centres = new_centres

        return {aid: int(assignments[i]) for i, aid in enumerate(ids)}


# ---------------------------------------------------------------------------
# Zoo manager
# ---------------------------------------------------------------------------

class AgentZoo:
    """
    Registry and manager for all zoo agents.
    Provides standardised evaluation and comparison.
    """

    def __init__(self):
        self._agents: Dict[str, BaseZooAgent] = {}
        self.leaderboard = AgentLeaderboard()
        self.fingerprinter = BehaviourFingerprinter()

    def register(self, agent: BaseZooAgent) -> None:
        self._agents[agent.agent_id] = agent
        logger.info("Registered zoo agent: %s (%s)", agent.agent_id, agent.name)

    def get(self, agent_id: str) -> Optional[BaseZooAgent]:
        return self._agents.get(agent_id)

    def all_agents(self) -> List[BaseZooAgent]:
        return list(self._agents.values())

    def evaluate(self, agent: BaseZooAgent, env_fn: Callable,
                  n_episodes: int = 100) -> LeaderboardEntry:
        """Run agent for n_episodes and record performance."""
        episode_returns = []
        for ep in range(n_episodes):
            env = env_fn()
            obs, _ = env.reset(seed=ep)
            agent.reset()
            total_return = 0.0
            done = False
            while not done:
                action = agent.act(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_return += reward
                done = terminated or truncated
            episode_returns.append(total_return)
            agent.record_return(total_return)

        return self.leaderboard.update(agent, episode_returns)

    def evaluate_all(self, env_fn: Callable, n_episodes: int = 50) -> None:
        for agent in self._agents.values():
            logger.info("Evaluating %s...", agent.name)
            entry = self.evaluate(agent, env_fn, n_episodes)
            logger.info("%s: mean_return=%.2f, sharpe=%.3f",
                        agent.name, entry.mean_return, entry.sharpe)

    def fingerprint_all(self) -> Dict[str, np.ndarray]:
        fps = {}
        for agent in self._agents.values():
            try:
                fp = self.fingerprinter.fingerprint(agent)
                fps[agent.agent_id] = fp
            except Exception as exc:
                logger.debug("Fingerprinting failed for %s: %s", agent.agent_id, exc)
        return fps

    def head_to_head(self, agent_a_id: str, agent_b_id: str,
                      env_fn: Callable, n_trials: int = 50) -> Dict[str, int]:
        """Run head-to-head evaluation between two agents."""
        agent_a = self._agents.get(agent_a_id)
        agent_b = self._agents.get(agent_b_id)
        if agent_a is None or agent_b is None:
            return {"error": "agent not found"}

        results = {"a_wins": 0, "b_wins": 0, "draws": 0}
        for trial in range(n_trials):
            env_a = env_fn()
            env_b = env_fn()
            obs_a, _ = env_a.reset(seed=trial)
            obs_b, _ = env_b.reset(seed=trial)
            agent_a.reset()
            agent_b.reset()

            ret_a = ret_b = 0.0
            done_a = done_b = False
            for _ in range(2000):
                if not done_a:
                    action_a = agent_a.act(obs_a)
                    obs_a, r_a, term_a, trunc_a, _ = env_a.step(action_a)
                    ret_a += r_a
                    done_a = term_a or trunc_a
                if not done_b:
                    action_b = agent_b.act(obs_b)
                    obs_b, r_b, term_b, trunc_b, _ = env_b.step(action_b)
                    ret_b += r_b
                    done_b = term_b or trunc_b
                if done_a and done_b:
                    break

            if ret_a > ret_b + 0.1:
                results["a_wins"] += 1
            elif ret_b > ret_a + 0.1:
                results["b_wins"] += 1
            else:
                results["draws"] += 1

        return results

    def print_leaderboard(self, top_k: int = 10) -> None:
        self.leaderboard.print_leaderboard(top_k)


# ---------------------------------------------------------------------------
# Default zoo factory
# ---------------------------------------------------------------------------

def build_default_zoo(obs_dim: int = 64, act_dim: int = 10) -> AgentZoo:
    """Build the default zoo with all benchmark agents."""
    zoo = AgentZoo()
    zoo.register(RandomAgent(act_dim=act_dim, seed=42))
    zoo.register(NopAgent(act_dim=act_dim))
    zoo.register(MomentumAgent(act_dim=act_dim))
    zoo.register(MeanReversionAgent(act_dim=act_dim))
    zoo.register(VWAPExecutorAgent(act_dim=act_dim, side=1))
    zoo.register(TwapExecutorAgent(act_dim=act_dim, side=1))
    zoo.register(SimpleMarketMakerAgent(act_dim=act_dim))
    zoo.register(AdaptiveMomentumAgent(act_dim=act_dim))
    return zoo


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== agent_zoo.py smoke test ===")

    obs_dim, act_dim = 64, 10
    zoo = build_default_zoo(obs_dim, act_dim)
    print(f"Zoo agents: {[a.name for a in zoo.all_agents()]}")

    # Test individual agents
    rng = np.random.default_rng(0)
    obs = rng.standard_normal(obs_dim).astype(np.float32)
    obs[0] = 0.1   # normalised mid price

    for agent in zoo.all_agents():
        agent.reset()
        action = agent.act(obs)
        assert action.shape == (act_dim,), f"Wrong action shape for {agent.name}"
        print(f"  {agent.name:30s}: action={action.round(3)[:4]}")

    # Test fingerprinting
    fp = zoo.fingerprint_all()
    print(f"\nFingerprints computed for: {list(fp.keys())}")

    # Test leaderboard (manual)
    for agent in zoo.all_agents():
        dummy_returns = rng.normal(5, 10, 50).tolist()
        zoo.leaderboard.update(agent, dummy_returns)

    zoo.print_leaderboard(top_k=5)

    print("\nAll smoke tests passed.")
