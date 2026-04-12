"""
PopulationTrainer — Co-evolutionary training of heterogeneous agent population.

Features:
  - Heterogeneous agent pool (MM, momentum, arb, noise)
  - Periodic agent replacement (worst → copy of best + mutation)
  - Co-evolutionary dynamics tracking
  - Population diversity metric (mean pairwise KL divergence)
  - Fitness landscape: which agent types thrive in which regimes
"""

from __future__ import annotations

import time
from collections import deque, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from hyper_agent.env_compat import MultiAgentTradingEnv
from hyper_agent.agents.base_agent import BaseAgent
from hyper_agent.agents.mappo_agent import MAPPOAgent
from hyper_agent.agents.market_maker_agent import MarketMakerAgent
from hyper_agent.agents.momentum_agent import MomentumAgent
from hyper_agent.agents.arbitrage_agent import ArbitrageAgent
from hyper_agent.agents.noise_trader import NoiseTrader
from hyper_agent.agents.mean_field_agent import MeanFieldAgent


# ============================================================
# Population fitness tracker
# ============================================================

class FitnessTracker:
    """
    Tracks per-agent fitness (cumulative PnL) over time.

    Used to identify weak agents for replacement and strong agents
    for cloning.
    """

    def __init__(self, agent_ids: List[str], window: int = 100) -> None:
        self.agent_ids = agent_ids
        self._rewards: Dict[str, deque] = {
            a: deque(maxlen=window) for a in agent_ids
        }
        self._total: Dict[str, float] = {a: 0.0 for a in agent_ids}

    def update(self, agent_id: str, reward: float) -> None:
        self._rewards[agent_id].append(reward)
        self._total[agent_id] += reward

    def fitness(self, agent_id: str) -> float:
        """Rolling mean reward."""
        rewards = self._rewards[agent_id]
        if not rewards:
            return 0.0
        return float(np.mean(rewards))

    def ranking(self) -> List[Tuple[str, float]]:
        """Return (agent_id, fitness) sorted best-first."""
        ranked = [(a, self.fitness(a)) for a in self.agent_ids]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def worst_k(self, k: int) -> List[str]:
        ranked = self.ranking()
        return [aid for aid, _ in ranked[-k:]]

    def best_k(self, k: int) -> List[str]:
        ranked = self.ranking()
        return [aid for aid, _ in ranked[:k]]

    def add_agent(self, agent_id: str) -> None:
        self._rewards[agent_id] = deque(maxlen=len(next(iter(self._rewards.values()))))
        self._total[agent_id]   = 0.0
        self.agent_ids.append(agent_id)

    def remove_agent(self, agent_id: str) -> None:
        self._rewards.pop(agent_id, None)
        self._total.pop(agent_id, None)
        if agent_id in self.agent_ids:
            self.agent_ids.remove(agent_id)


# ============================================================
# Population Diversity Metric
# ============================================================

class DiversityMetric:
    """
    Measures population diversity as mean pairwise KL divergence
    between agent policies.

    Used to prevent population collapse (all agents converging
    to the same strategy).
    """

    def __init__(self, sample_obs: np.ndarray) -> None:
        """
        Args:
            sample_obs: (N, obs_dim) batch of observations to evaluate on
        """
        self.sample_obs = sample_obs

    def compute(self, agents: List[BaseAgent]) -> float:
        """
        Compute mean pairwise KL divergence between all agent pairs.
        Returns 0 if fewer than 2 agents.
        """
        if len(agents) < 2:
            return 0.0

        # Only compute for agents with proper policy networks
        learnable = [a for a in agents if hasattr(a, "actor")]
        if len(learnable) < 2:
            return 0.0

        # Sample subset for efficiency
        obs_sample = self.sample_obs[:20]  # limit to 20 obs for speed
        kl_pairs   = []

        for i in range(len(learnable)):
            for j in range(i + 1, min(i + 5, len(learnable))):  # limit comparisons
                try:
                    kl = learnable[i].policy_kl(learnable[j], obs_sample)
                    kl_pairs.append(kl)
                except Exception:
                    continue

        return float(np.mean(kl_pairs)) if kl_pairs else 0.0


# ============================================================
# Fitness Landscape Tracker
# ============================================================

class FitnessLandscape:
    """
    Tracks which agent types thrive in which market regimes.

    Records (regime, agent_type, fitness) tuples and computes
    conditional fitness E[fitness | regime, agent_type].
    """

    def __init__(self) -> None:
        self._records: List[Tuple[str, str, float]] = []  # (regime, type, fitness)

    def record(self, regime: str, agent_type: str, fitness: float) -> None:
        self._records.append((regime, agent_type, fitness))

    def conditional_fitness(self) -> Dict[Tuple[str, str], float]:
        """
        Returns E[fitness | (regime, agent_type)] for all observed combos.
        """
        groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for regime, atype, fitness in self._records:
            groups[(regime, atype)].append(fitness)
        return {k: float(np.mean(v)) for k, v in groups.items()}

    def best_type_per_regime(self) -> Dict[str, str]:
        """Return best agent type for each regime."""
        cf = self.conditional_fitness()
        regime_best: Dict[str, Tuple[str, float]] = {}
        for (regime, atype), fit in cf.items():
            if regime not in regime_best or fit > regime_best[regime][1]:
                regime_best[regime] = (atype, fit)
        return {r: t for r, (t, _) in regime_best.items()}


# ============================================================
# PopulationTrainer
# ============================================================

class PopulationTrainer:
    """
    Trains a heterogeneous population of RL agents via co-evolution.

    Population composition:
      - N_mm:    market maker agents
      - N_mom:   momentum agents
      - N_arb:   arbitrage agents
      - N_noise: noise traders (random, no learning)

    Evolution:
      Every `replace_every` episodes:
        1. Evaluate all agents' fitness over the window
        2. Replace bottom K agents with copies of top K agents + noise mutation
        3. Track diversity; if it drops below threshold, inject random agents

    Co-evolution dynamics:
      - Agents' strategies co-evolve in response to each other
      - E.g., if many momentum agents → market makers adapt by widening spreads
      - Arb agents detect and exploit new pricing inefficiencies
    """

    def __init__(
        self,
        env:               MultiAgentTradingEnv,
        obs_dim:           int,
        agents:            Dict[str, BaseAgent],
        replace_every:     int   = 50,
        replace_fraction:  float = 0.2,
        mutation_std:      float = 0.02,
        min_diversity:     float = 0.1,
        diversity_obs:     Optional[np.ndarray] = None,
        n_total_episodes:  int   = 5_000,
        checkpoint_dir:    str   = "./pop_checkpoints",
        device:            str   = "cpu",
    ) -> None:
        self.env               = env
        self.obs_dim           = obs_dim
        self.agents            = agents
        self.agent_ids         = list(agents.keys())
        self.replace_every     = replace_every
        self.replace_fraction  = replace_fraction
        self.mutation_std      = mutation_std
        self.min_diversity     = min_diversity
        self.n_total_episodes  = n_total_episodes
        self.checkpoint_dir    = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device            = device

        self.fitness_tracker = FitnessTracker(self.agent_ids)

        # Diversity metric: needs sample observations
        if diversity_obs is not None:
            self._div_metric = DiversityMetric(diversity_obs)
        else:
            dummy_obs = np.zeros((20, obs_dim), dtype=np.float32)
            self._div_metric = DiversityMetric(dummy_obs)

        self.landscape = FitnessLandscape()

        # Tracking
        self.episode_count  = 0
        self.global_step    = 0
        self._diversity_hist: deque = deque(maxlen=200)
        self._regime_hist:    deque = deque(maxlen=200)
        self._replacement_log: List[Dict] = []

        # Stats
        self.stats: Dict[str, List] = defaultdict(list)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Run full population training."""
        start = time.time()
        obs, _ = self.env.reset()

        while self.episode_count < self.n_total_episodes:
            # Run one episode
            ep_rewards, ep_len, regime = self._run_episode(obs)
            obs, _    = self.env.reset()
            self.episode_count += 1

            # Update fitness
            for aid, r in ep_rewards.items():
                self.fitness_tracker.update(aid, r)

            # Record landscape
            for aid, r in ep_rewards.items():
                atype = self.agents[aid].agent_type
                self.landscape.record(regime, atype, r)

            # Diversity measurement
            diversity = self._measure_diversity()
            self._diversity_hist.append(diversity)
            self._regime_hist.append(regime)

            # Log
            self.stats["mean_reward"].append(float(np.mean(list(ep_rewards.values()))))
            self.stats["diversity"].append(diversity)
            self.stats["episode_len"].append(ep_len)

            # Agent replacement
            if self.episode_count % self.replace_every == 0:
                self._replace_agents(diversity)

            # Periodic checkpoint
            if self.episode_count % 200 == 0:
                self._save_population()
                elapsed = time.time() - start
                print(
                    f"Episode {self.episode_count} | "
                    f"Mean Reward: {self.stats['mean_reward'][-1]:.4f} | "
                    f"Diversity: {diversity:.4f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

        return self._final_report()

    def _run_episode(
        self, initial_obs: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[Dict[str, float], int, str]:
        """
        Run a single episode.

        Returns (episode_rewards_per_agent, episode_length, regime).
        """
        if initial_obs is None:
            obs, _ = self.env.reset()
        else:
            obs = initial_obs

        ep_rewards = {a: 0.0 for a in self.agent_ids}
        step        = 0
        regime      = "normal"

        max_steps = self.env.env.max_steps
        while step < max_steps:
            actions = {}
            for aid, agent in self.agents.items():
                if aid in obs:
                    act, _, _ = agent.act(obs[aid])
                    actions[aid] = act

            next_obs, rewards, terminated, truncated, info = self.env.step(actions)

            # Update per-agent rollout buffers
            for aid, agent in self.agents.items():
                r    = rewards.get(aid, 0.0)
                done = terminated.get(aid, False) or truncated.get(aid, False)
                ep_rewards[aid] += r

                # Type-specific reward storage
                if hasattr(agent, "receive_reward"):
                    agent.receive_reward(r, done)
                elif hasattr(agent, "store_transition"):
                    if aid in obs and aid in next_obs:
                        logits = actions[aid][:3]
                        probs  = np.exp(logits - logits.max())
                        probs /= probs.sum()
                        agent.store_transition(
                            obs[aid], int(np.argmax(probs)), float(actions[aid][3]),
                            0.0, r, next_obs[aid], done,
                        )

            # Get regime from info
            if info:
                first_info = next(iter(info.values()))
                regime     = "crisis" if first_info.get("in_crisis", False) else "normal"

            obs   = next_obs
            step += 1
            self.global_step += 1

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        # Update agents
        for aid, agent in self.agents.items():
            try:
                agent.update()
            except Exception:
                pass  # Some agents (noise) return empty dict

        return ep_rewards, step, regime

    # ------------------------------------------------------------------
    # Agent replacement (evolutionary selection)
    # ------------------------------------------------------------------

    def _replace_agents(self, diversity: float) -> None:
        """
        Replace worst-performing agents with copies of best agents + mutation.

        If diversity drops below threshold, also inject fresh random agents.
        """
        n_replace = max(1, int(len(self.agent_ids) * self.replace_fraction))

        worst   = self.fitness_tracker.worst_k(n_replace)
        best    = self.fitness_tracker.best_k(max(1, n_replace // 2))

        replaced_ids = []
        for i, dead_id in enumerate(worst):
            # Skip noise traders (they don't learn)
            if self.agents[dead_id].agent_type == "noise":
                continue

            # Clone a top agent
            parent_id = best[i % len(best)]
            parent    = self.agents[parent_id]

            # Can only clone same type
            child = parent.clone()
            child.agent_id = dead_id
            child.mutate(self.mutation_std)

            # Reset the child's episode history
            if hasattr(child, "reset_episode"):
                child.reset_episode()

            self.agents[dead_id]           = child
            self.fitness_tracker._rewards[dead_id].clear()
            replaced_ids.append((dead_id, parent_id))

        # Diversity injection: if too homogeneous, add noise
        if diversity < self.min_diversity:
            self._inject_diversity()

        if replaced_ids:
            self._replacement_log.append({
                "episode":   self.episode_count,
                "replaced":  replaced_ids,
                "diversity": diversity,
            })

    def _inject_diversity(self) -> None:
        """
        Inject diversity by mutating a random subset of agents strongly.
        """
        n_inject = max(1, len(self.agent_ids) // 10)
        targets  = np.random.choice(self.agent_ids, size=n_inject, replace=False)

        for aid in targets:
            agent = self.agents[aid]
            if agent.agent_type != "noise":
                agent.mutate(self.mutation_std * 5.0)  # stronger mutation

    def _measure_diversity(self) -> float:
        """Compute population diversity (mean pairwise KL)."""
        all_agents = list(self.agents.values())
        return self._div_metric.compute(all_agents)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def population_composition(self) -> Dict[str, int]:
        """Count agents by type."""
        counts: Dict[str, int] = defaultdict(int)
        for agent in self.agents.values():
            counts[agent.agent_type] += 1
        return dict(counts)

    def type_fitness(self) -> Dict[str, float]:
        """Mean fitness by agent type."""
        by_type: Dict[str, List[float]] = defaultdict(list)
        for aid in self.agent_ids:
            atype = self.agents[aid].agent_type
            by_type[atype].append(self.fitness_tracker.fitness(aid))
        return {t: float(np.mean(v)) for t, v in by_type.items()}

    def diversity_trend(self) -> np.ndarray:
        return np.array(list(self._diversity_hist))

    def fitness_landscape_summary(self) -> Dict[str, Any]:
        return {
            "best_type_per_regime": self.landscape.best_type_per_regime(),
            "conditional_fitness":  self.landscape.conditional_fitness(),
        }

    def _save_population(self) -> None:
        for aid, agent in self.agents.items():
            if agent.agent_type != "noise":
                path = self.checkpoint_dir / f"{aid}_pop.pt"
                try:
                    agent.save(str(path))
                except Exception:
                    pass

    def _final_report(self) -> Dict[str, Any]:
        return {
            "total_episodes":     self.episode_count,
            "global_steps":       self.global_step,
            "final_composition":  self.population_composition(),
            "final_type_fitness": self.type_fitness(),
            "mean_diversity":     float(np.mean(self.diversity_trend())) if len(self.diversity_trend()) > 0 else 0.0,
            "n_replacements":     len(self._replacement_log),
            "fitness_landscape":  self.fitness_landscape_summary(),
        }
