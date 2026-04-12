"""
curriculum_learning.py
=======================
Automated curriculum learning for the Hyper-Agent MARL ecosystem.

Provides:
  - Task difficulty scoring and tracking
  - Agent performance tracking per task type
  - Adaptive scenario selection near competency boundary (ZPD)
  - Population-based training (PBT) for hyperparameter evolution
  - Domain randomisation manager
  - Automatic curriculum progression
  - Multi-agent curriculum coordination
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import json
import logging
import math
import pathlib
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task descriptor
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Task:
    """A training task with associated difficulty and configuration."""
    task_id: str
    difficulty: float           # [0, 1]
    env_kwargs: Dict[str, Any]
    scenario_type: str = "normal"
    prerequisite_tasks: List[str] = dataclasses.field(default_factory=list)
    mastery_threshold: float = 0.7   # success_rate needed to advance
    description: str = ""

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class TaskPerformance:
    """Tracks an agent's performance on a specific task."""
    task_id: str
    agent_id: str
    attempts: int = 0
    successes: int = 0
    recent_returns: deque = dataclasses.field(
        default_factory=lambda: deque(maxlen=100)
    )
    recent_successes: deque = dataclasses.field(
        default_factory=lambda: deque(maxlen=50)
    )

    @property
    def success_rate(self) -> float:
        if not self.recent_successes:
            return 0.0
        return float(np.mean(self.recent_successes))

    @property
    def mean_return(self) -> float:
        if not self.recent_returns:
            return float("-inf")
        return float(np.mean(self.recent_returns))

    def record(self, episode_return: float, success: bool) -> None:
        self.attempts += 1
        if success:
            self.successes += 1
        self.recent_returns.append(episode_return)
        self.recent_successes.append(float(success))


# ---------------------------------------------------------------------------
# Difficulty scorer
# ---------------------------------------------------------------------------

class TaskDifficultyScorer:
    """
    Assigns numerical difficulty scores to tasks based on empirical agent
    performance across a population.
    """

    def __init__(self, population_size: int = 8):
        self.population_size = population_size
        self._task_returns: Dict[str, List[float]] = defaultdict(list)

    def record_return(self, task_id: str, agent_return: float) -> None:
        self._task_returns[task_id].append(agent_return)
        # Keep rolling window
        if len(self._task_returns[task_id]) > 1000:
            self._task_returns[task_id] = self._task_returns[task_id][-500:]

    def score(self, task_id: str) -> float:
        """
        Returns empirical difficulty in [0,1].
        Difficulty = 1 - normalised success rate across population.
        """
        returns = self._task_returns.get(task_id, [])
        if not returns:
            return 0.5   # unknown difficulty

        arr = np.array(returns)
        # Normalise: positive returns = success
        success_rate = float((arr > 0).mean())
        return 1.0 - success_rate

    def relative_difficulty(self, task_id_a: str, task_id_b: str) -> float:
        """Returns positive if A is harder than B."""
        return self.score(task_id_a) - self.score(task_id_b)


# ---------------------------------------------------------------------------
# Curriculum progression model
# ---------------------------------------------------------------------------

class ProgressionModel(enum.Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    ADAPTIVE = "adaptive"


class CurriculumProgressionEngine:
    """
    Manages curriculum progression: tracks mastery of each task
    and decides when to advance to the next difficulty level.
    """

    def __init__(self,
                 tasks: List[Task],
                 progression_model: ProgressionModel = ProgressionModel.ADAPTIVE,
                 min_attempts_before_advance: int = 20,
                 mastery_window: int = 50):
        # Sort tasks by difficulty
        self.tasks = sorted(tasks, key=lambda t: t.difficulty)
        self.progression_model = progression_model
        self.min_attempts = min_attempts_before_advance
        self.mastery_window = mastery_window

        self._current_level: int = 0
        self._task_perf: Dict[str, TaskPerformance] = {
            t.task_id: TaskPerformance(task_id=t.task_id, agent_id="default")
            for t in self.tasks
        }

    @property
    def current_task(self) -> Task:
        return self.tasks[min(self._current_level, len(self.tasks) - 1)]

    def record(self, task_id: str, episode_return: float, success: bool) -> None:
        if task_id in self._task_perf:
            self._task_perf[task_id].record(episode_return, success)
        self._maybe_advance(task_id)

    def _maybe_advance(self, task_id: str) -> None:
        perf = self._task_perf.get(task_id)
        if perf is None:
            return
        task_idx = next((i for i, t in enumerate(self.tasks) if t.task_id == task_id), -1)
        if task_idx != self._current_level:
            return

        task = self.tasks[task_idx]
        if (perf.attempts >= self.min_attempts and
                perf.success_rate >= task.mastery_threshold):
            if self._current_level < len(self.tasks) - 1:
                self._current_level += 1
                logger.info(
                    "Curriculum advanced to level %d: %s",
                    self._current_level,
                    self.current_task.task_id,
                )

    def get_sampling_weights(self) -> Dict[str, float]:
        """Sample near-current-level tasks, with some exploration of harder tasks."""
        weights = {}
        for i, task in enumerate(self.tasks):
            dist_from_current = abs(i - self._current_level)
            weight = math.exp(-0.5 * dist_from_current)
            # Give extra weight to current and one above
            if i == self._current_level:
                weight *= 2.0
            elif i == self._current_level + 1:
                weight *= 1.2
            weights[task.task_id] = weight
        return weights

    def sample_task(self, rng: Optional[np.random.Generator] = None) -> Task:
        if rng is None:
            rng = np.random.default_rng()
        weights = self.get_sampling_weights()
        task_ids = list(weights.keys())
        probs = np.array([weights[tid] for tid in task_ids])
        probs /= probs.sum()
        chosen_id = rng.choice(task_ids, p=probs)
        return next(t for t in self.tasks if t.task_id == chosen_id)

    def summary(self) -> Dict[str, Any]:
        return {
            "current_level": self._current_level,
            "current_task": self.current_task.task_id,
            "n_tasks": len(self.tasks),
            "task_performances": {
                tid: {"success_rate": p.success_rate,
                      "mean_return": p.mean_return,
                      "attempts": p.attempts}
                for tid, p in self._task_perf.items()
            },
        }


# ---------------------------------------------------------------------------
# Zone of proximal development (ZPD) selector
# ---------------------------------------------------------------------------

class ZPDSelector:
    """
    Selects scenarios in the agent's Zone of Proximal Development:
    tasks that are challenging but not impossible — just above current competency.
    """

    def __init__(self,
                 target_success_rate: float = 0.6,
                 window: int = 50,
                 tolerance: float = 0.15):
        self.target_success_rate = target_success_rate
        self.window = window
        self.tolerance = tolerance
        self._perf_by_difficulty: Dict[float, deque] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._rng = np.random.default_rng()

    def record(self, difficulty: float, success: bool) -> None:
        # Bin to nearest 0.1
        key = round(difficulty, 1)
        self._perf_by_difficulty[key].append(float(success))

    def select_difficulty(self,
                           difficulty_range: Tuple[float, float] = (0.1, 0.9)) -> float:
        """Select difficulty level in the ZPD."""
        if not self._perf_by_difficulty:
            return float(self._rng.uniform(*difficulty_range))

        # Find difficulty closest to target success rate
        best_diff = None
        best_score = float("inf")
        for diff, buf in sorted(self._perf_by_difficulty.items()):
            if not buf:
                continue
            sr = float(np.mean(buf))
            score = abs(sr - self.target_success_rate)
            if score < best_score:
                best_score = score
                best_diff = diff

        if best_diff is None:
            return float(self._rng.uniform(*difficulty_range))

        # Add small exploration noise
        noise = float(self._rng.normal(0, 0.05))
        return float(np.clip(best_diff + noise, *difficulty_range))

    def competency_level(self) -> float:
        """Estimated current competency (difficulty where success ~ 0.7)."""
        competency = 0.0
        for diff, buf in sorted(self._perf_by_difficulty.items()):
            if buf and float(np.mean(buf)) >= 0.7:
                competency = diff
        return competency


# ---------------------------------------------------------------------------
# Population-based training (PBT)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HyperparamConfig:
    """A hyperparameter configuration for one member of the PBT population."""
    agent_id: str
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    clip_eps: float = 0.2
    n_epochs: int = 4
    batch_size: int = 64
    gae_lambda: float = 0.95
    inventory_penalty: float = 0.01
    reward_scale: float = 1.0
    comm_msg_dim: int = 64
    # Fitness tracking
    fitness: float = 0.0
    generation: int = 0
    parent_id: Optional[str] = None

    def mutate(self, rng: np.random.Generator,
                mutation_factor: float = 0.2) -> "HyperparamConfig":
        """Perturb hyperparameters by up to mutation_factor in log space."""
        child = copy.copy(self)
        for field in dataclasses.fields(self):
            if field.name in ("agent_id", "fitness", "generation", "parent_id"):
                continue
            val = getattr(self, field.name)
            if isinstance(val, float) and val > 0:
                new_val = val * float(rng.lognormal(0, mutation_factor))
                setattr(child, field.name, max(new_val, 1e-9))
            elif isinstance(val, int) and val > 0:
                new_val = max(1, int(round(val * float(rng.lognormal(0, 0.3)))))
                setattr(child, field.name, new_val)
        child.parent_id = self.agent_id
        return child

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def random(cls, agent_id: str,
                rng: Optional[np.random.Generator] = None) -> "HyperparamConfig":
        if rng is None:
            rng = np.random.default_rng()
        return cls(
            agent_id=agent_id,
            lr=float(rng.loguniform(1e-5, 1e-3)),
            gamma=float(rng.uniform(0.95, 0.999)),
            entropy_coef=float(rng.loguniform(1e-4, 0.05)),
            clip_eps=float(rng.uniform(0.1, 0.3)),
            n_epochs=int(rng.integers(2, 10)),
            batch_size=int(rng.choice([32, 64, 128, 256])),
            gae_lambda=float(rng.uniform(0.9, 0.99)),
            inventory_penalty=float(rng.loguniform(1e-4, 0.1)),
            reward_scale=float(rng.uniform(0.5, 2.0)),
        )


class PopulationBasedTraining:
    """
    PBT manager: maintains a population of agents with different hyperparameters,
    periodically exploits top performers and explores new configurations.
    """

    def __init__(self,
                 population_size: int = 8,
                 exploit_fraction: float = 0.2,
                 explore_probability: float = 0.8,
                 eval_interval: int = 100,
                 mutation_factor: float = 0.2,
                 seed: int = 0):
        self.population_size = population_size
        self.exploit_fraction = exploit_fraction
        self.explore_probability = explore_probability
        self.eval_interval = eval_interval
        self.mutation_factor = mutation_factor
        self._rng = np.random.default_rng(seed)

        self._population: List[HyperparamConfig] = [
            HyperparamConfig.random(f"agent_{i}", self._rng)
            for i in range(population_size)
        ]
        self._fitness_history: Dict[str, List[float]] = {
            p.agent_id: [] for p in self._population
        }
        self._step: int = 0
        self._generation: int = 0

    def update_fitness(self, agent_id: str, fitness: float) -> None:
        for p in self._population:
            if p.agent_id == agent_id:
                p.fitness = fitness
                self._fitness_history[agent_id].append(fitness)
                break

    def get_config(self, agent_id: str) -> Optional[HyperparamConfig]:
        return next((p for p in self._population if p.agent_id == agent_id), None)

    def step(self) -> List[str]:
        """
        Perform one PBT step. Returns list of agent IDs whose configs were replaced.
        """
        self._step += 1
        if self._step % self.eval_interval != 0:
            return []

        return self._exploit_and_explore()

    def _exploit_and_explore(self) -> List[str]:
        """Bottom fraction copies top fraction, then perturbs."""
        n_exploit = max(1, int(self.population_size * self.exploit_fraction))
        sorted_pop = sorted(self._population, key=lambda p: p.fitness, reverse=True)

        top = sorted_pop[:n_exploit]
        bottom = sorted_pop[-n_exploit:]
        replaced_ids = []

        for loser in bottom:
            parent = top[int(self._rng.integers(0, len(top)))]
            if self._rng.random() < self.explore_probability:
                new_config = parent.mutate(self._rng, self.mutation_factor)
            else:
                new_config = copy.copy(parent)

            new_config.agent_id = loser.agent_id
            new_config.fitness = 0.0
            new_config.generation = self._generation + 1
            # Replace loser in population
            for i, p in enumerate(self._population):
                if p.agent_id == loser.agent_id:
                    self._population[i] = new_config
                    break
            self._fitness_history[loser.agent_id].append(new_config.fitness)
            replaced_ids.append(loser.agent_id)

        self._generation += 1
        logger.info("PBT generation %d: replaced %d agents", self._generation, len(replaced_ids))
        return replaced_ids

    def leaderboard(self) -> List[Dict[str, Any]]:
        sorted_pop = sorted(self._population, key=lambda p: p.fitness, reverse=True)
        return [
            {"rank": i + 1, "agent_id": p.agent_id,
             "fitness": p.fitness, "generation": p.generation,
             "lr": p.lr, "gamma": p.gamma}
            for i, p in enumerate(sorted_pop)
        ]

    def best_config(self) -> HyperparamConfig:
        return max(self._population, key=lambda p: p.fitness)

    @property
    def population(self) -> List[HyperparamConfig]:
        return list(self._population)


# ---------------------------------------------------------------------------
# Domain randomisation
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DomainParams:
    """Randomisable environment parameters for domain randomisation."""
    tick_size: float = 0.01
    max_order_size: int = 50
    episode_len: int = 2000
    base_spread: float = 0.02
    base_volatility: float = 0.0002
    initial_price: float = 100.0
    inventory_penalty_coef: float = 0.01
    slippage_factor: float = 5e-5
    market_impact_factor: float = 1e-4
    history_len: int = 20
    lob_depth: int = 10


class DomainRandomiser:
    """
    Randomises environment parameters each episode to improve generalisation.
    Supports uniform, log-uniform, and categorical distributions per parameter.
    """

    def __init__(self,
                 ranges: Optional[Dict[str, Any]] = None,
                 seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._ranges = ranges or self._default_ranges()

    @staticmethod
    def _default_ranges() -> Dict[str, Any]:
        return {
            "base_spread": ("uniform", 0.01, 0.1),
            "base_volatility": ("loguniform", 5e-5, 1e-3),
            "episode_len": ("choice", [1000, 1500, 2000, 2500]),
            "inventory_penalty_coef": ("loguniform", 1e-4, 0.05),
            "slippage_factor": ("loguniform", 1e-5, 1e-4),
            "market_impact_factor": ("loguniform", 1e-5, 1e-3),
            "history_len": ("choice", [10, 20, 30, 50]),
        }

    def sample(self) -> DomainParams:
        params = DomainParams()
        for field_name, spec in self._ranges.items():
            dist = spec[0]
            if dist == "uniform":
                val = float(self._rng.uniform(spec[1], spec[2]))
            elif dist == "loguniform":
                val = float(self._rng.loguniform(spec[1], spec[2]))
            elif dist == "choice":
                val = self._rng.choice(spec[1])
            else:
                continue
            if hasattr(params, field_name):
                setattr(params, field_name, type(getattr(params, field_name))(val))
        return params

    def set_range(self, field_name: str, spec: Any) -> None:
        self._ranges[field_name] = spec


# ---------------------------------------------------------------------------
# Multi-agent curriculum coordinator
# ---------------------------------------------------------------------------

class MultiAgentCurriculumCoordinator:
    """
    Coordinates curriculum across multiple agents to ensure the overall
    training remains balanced and cooperative objectives are met.
    """

    def __init__(self, agent_ids: List[str],
                 tasks: List[Task],
                 synchronise_levels: bool = False):
        self.agent_ids = agent_ids
        self.tasks = tasks
        self.synchronise_levels = synchronise_levels

        self._progressions: Dict[str, CurriculumProgressionEngine] = {
            aid: CurriculumProgressionEngine(copy.deepcopy(tasks))
            for aid in agent_ids
        }
        self._zpd_selectors: Dict[str, ZPDSelector] = {
            aid: ZPDSelector() for aid in agent_ids
        }

    def record_episode(self, agent_id: str, task_id: str,
                        episode_return: float, success: bool) -> None:
        if agent_id in self._progressions:
            self._progressions[agent_id].record(task_id, episode_return, success)
        if agent_id in self._zpd_selectors:
            task = next((t for t in self.tasks if t.task_id == task_id), None)
            if task:
                self._zpd_selectors[agent_id].record(task.difficulty, success)

        if self.synchronise_levels:
            self._sync_levels()

    def _sync_levels(self) -> None:
        """Set all agents to median level."""
        levels = [p._current_level for p in self._progressions.values()]
        median_level = int(np.median(levels))
        for prog in self._progressions.values():
            prog._current_level = max(prog._current_level,
                                       min(median_level, len(prog.tasks) - 1))

    def sample_task_for_agent(self, agent_id: str) -> Task:
        prog = self._progressions[agent_id]
        return prog.sample_task()

    def sample_difficulty_for_agent(self, agent_id: str) -> float:
        zpd = self._zpd_selectors[agent_id]
        return zpd.select_difficulty()

    def global_competency(self) -> float:
        """Average competency across all agents."""
        competencies = [
            self._zpd_selectors[aid].competency_level()
            for aid in self.agent_ids
        ]
        return float(np.mean(competencies)) if competencies else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "global_competency": self.global_competency(),
            "agent_levels": {
                aid: self._progressions[aid]._current_level
                for aid in self.agent_ids
            },
            "agent_tasks": {
                aid: self._progressions[aid].current_task.task_id
                for aid in self.agent_ids
            },
        }


# ---------------------------------------------------------------------------
# Curriculum task library
# ---------------------------------------------------------------------------

def build_default_curriculum(n_ticks: int = 2000) -> List[Task]:
    """Build the default task ladder for financial MARL training."""
    tasks = []

    # Level 0: Trivial flat market
    tasks.append(Task(
        task_id="flat_market_easy",
        difficulty=0.1,
        env_kwargs={
            "scenario": "normal",
            "n_synthetic": 1000,
            "episode_len": 500,
            "inventory_penalty_coef": 0.001,
        },
        scenario_type="normal",
        mastery_threshold=0.65,
        description="Flat market, short episodes",
    ))

    # Level 1: Normal market
    tasks.append(Task(
        task_id="normal_market",
        difficulty=0.3,
        env_kwargs={
            "scenario": "normal",
            "n_synthetic": 2000,
            "episode_len": 1000,
            "inventory_penalty_coef": 0.005,
        },
        prerequisite_tasks=["flat_market_easy"],
        scenario_type="normal",
        mastery_threshold=0.6,
        description="Standard market conditions",
    ))

    # Level 2: Trending market
    tasks.append(Task(
        task_id="trending_market",
        difficulty=0.45,
        env_kwargs={
            "scenario": "trending",
            "n_synthetic": 2000,
            "episode_len": 1500,
            "inventory_penalty_coef": 0.01,
        },
        prerequisite_tasks=["normal_market"],
        scenario_type="trending_up",
        mastery_threshold=0.6,
        description="Trending market with price drift",
    ))

    # Level 3: High volatility
    tasks.append(Task(
        task_id="high_vol",
        difficulty=0.6,
        env_kwargs={
            "scenario": "high_vol",
            "n_synthetic": 2000,
            "episode_len": 2000,
        },
        prerequisite_tasks=["normal_market"],
        scenario_type="high_vol",
        mastery_threshold=0.55,
        description="High volatility regime",
    ))

    # Level 4: Liquidity crisis
    tasks.append(Task(
        task_id="liquidity_crisis",
        difficulty=0.75,
        env_kwargs={
            "scenario": "liquidity_crisis",
            "n_synthetic": 2000,
            "episode_len": 2000,
        },
        prerequisite_tasks=["high_vol"],
        scenario_type="liquidity_crisis",
        mastery_threshold=0.5,
        description="Liquidity crisis with wide spreads",
    ))

    # Level 5: Flash crash
    tasks.append(Task(
        task_id="flash_crash",
        difficulty=0.85,
        env_kwargs={
            "scenario": "flash_crash",
            "n_synthetic": 2000,
            "episode_len": 2000,
        },
        prerequisite_tasks=["liquidity_crisis"],
        scenario_type="flash_crash",
        mastery_threshold=0.45,
        description="Flash crash scenario",
    ))

    # Level 6: Mixed adversarial
    tasks.append(Task(
        task_id="adversarial_mix",
        difficulty=0.95,
        env_kwargs={
            "scenario": None,
            "n_synthetic": 5000,
            "episode_len": 2000,
        },
        prerequisite_tasks=["flash_crash"],
        scenario_type="mixed",
        mastery_threshold=0.4,
        description="Mixed adversarial scenarios",
    ))

    return tasks


# ---------------------------------------------------------------------------
# Training manager that integrates all curriculum components
# ---------------------------------------------------------------------------

class CurriculumTrainingManager:
    """
    High-level training manager that ties together:
      - CurriculumProgressionEngine
      - ZPDSelector
      - PopulationBasedTraining
      - DomainRandomiser
    """

    def __init__(self,
                 agent_ids: List[str],
                 tasks: Optional[List[Task]] = None,
                 pbt_population_size: int = 8,
                 domain_randomise: bool = True,
                 seed: int = 0):
        self.agent_ids = agent_ids
        self.tasks = tasks or build_default_curriculum()
        self._rng = np.random.default_rng(seed)

        self.coordinator = MultiAgentCurriculumCoordinator(
            agent_ids, self.tasks
        )
        self.pbt = PopulationBasedTraining(
            population_size=pbt_population_size, seed=seed
        )
        self.domain_randomiser = DomainRandomiser(seed=seed) if domain_randomise else None
        self._episode_count: int = 0
        self._step_count: int = 0

    def on_episode_end(self,
                        agent_id: str,
                        task_id: str,
                        episode_return: float,
                        success: bool) -> Dict[str, Any]:
        self._episode_count += 1
        self.coordinator.record_episode(agent_id, task_id, episode_return, success)
        self.pbt.update_fitness(agent_id, episode_return)
        replaced = self.pbt.step()
        return {
            "episode": self._episode_count,
            "replaced_agents": replaced,
            "curriculum_summary": self.coordinator.summary(),
        }

    def get_next_task(self, agent_id: str) -> Tuple[Task, DomainParams]:
        task = self.coordinator.sample_task_for_agent(agent_id)
        domain = (self.domain_randomiser.sample()
                  if self.domain_randomiser else DomainParams())
        return task, domain

    def get_hyperparam_config(self, agent_id: str) -> HyperparamConfig:
        cfg = self.pbt.get_config(agent_id)
        return cfg or HyperparamConfig(agent_id=agent_id)

    def global_progress(self) -> float:
        return self.coordinator.global_competency()

    def full_summary(self) -> Dict[str, Any]:
        return {
            "episode_count": self._episode_count,
            "global_competency": self.global_progress(),
            "pbt_leaderboard": self.pbt.leaderboard()[:5],
            "curriculum": self.coordinator.summary(),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== curriculum_learning.py smoke test ===")

    tasks = build_default_curriculum()
    print(f"Built {len(tasks)} tasks: {[t.task_id for t in tasks]}")

    # Test progression engine
    engine = CurriculumProgressionEngine(tasks)
    for i in range(30):
        engine.record("flat_market_easy", 10.0, success=True)
    print(f"Level after 30 successes: {engine._current_level} ({engine.current_task.task_id})")

    # Test ZPD selector
    zpd = ZPDSelector(target_success_rate=0.6)
    for i in range(50):
        zpd.record(0.3, i % 5 != 0)
    for i in range(30):
        zpd.record(0.5, i % 3 != 0)
    chosen_diff = zpd.select_difficulty()
    print(f"ZPD selected difficulty: {chosen_diff:.2f}")
    print(f"Competency estimate: {zpd.competency_level():.2f}")

    # Test PBT
    pbt = PopulationBasedTraining(population_size=4, eval_interval=2)
    for step in range(10):
        for p in pbt.population:
            pbt.update_fitness(p.agent_id, float(np.random.randn()))
        replaced = pbt.step()
        if replaced:
            print(f"PBT step {step}: replaced {replaced}")
    print("PBT leaderboard:", pbt.leaderboard()[:2])

    # Test domain randomiser
    dr = DomainRandomiser(seed=42)
    domain = dr.sample()
    print(f"Domain sample: spread={domain.base_spread:.4f}, vol={domain.base_volatility:.6f}")

    # Test curriculum training manager
    mgr = CurriculumTrainingManager(
        agent_ids=["agent_0", "agent_1"],
        pbt_population_size=4,
    )
    for ep in range(20):
        for aid in ["agent_0", "agent_1"]:
            result = mgr.on_episode_end(aid, "flat_market_easy",
                                         float(np.random.randn()), True)
    print("Manager summary:", mgr.full_summary())

    print("\nAll smoke tests passed.")
