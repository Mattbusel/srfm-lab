"""
Genetic hypothesis evolver — evolves trading hypotheses through selection and mutation.

Applies genetic algorithm principles:
  - Population of hypothesis parameter sets
  - Fitness = risk-adjusted backtest performance
  - Selection: tournament selection
  - Crossover: uniform and arithmetic crossover
  - Mutation: Gaussian perturbation
  - Elitism: preserve top performers
  - Diversity maintenance: crowding/niching
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Chromosome:
    """A single hypothesis parameter set."""
    params: dict
    fitness: float = 0.0
    generation: int = 0
    id: str = ""
    parent_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = _random_id()


@dataclass
class EvolverConfig:
    population_size: int = 50
    n_generations: int = 100
    elite_frac: float = 0.10
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.20
    crossover_rate: float = 0.70
    tournament_size: int = 5
    diversity_weight: float = 0.10
    min_fitness: float = -np.inf   # prune below this
    seed: int = 42


class GeneticHypothesisEvolver:
    """
    Evolves a population of hypothesis parameter sets toward higher fitness.
    """

    def __init__(
        self,
        param_schema: dict,       # {"param_name": (min, max, type)}
        fitness_fn: Callable,     # fitness_fn(params: dict) -> float
        config: Optional[EvolverConfig] = None,
    ):
        self.schema = param_schema
        self.fitness_fn = fitness_fn
        self.config = config or EvolverConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.population: list[Chromosome] = []
        self.history: list[dict] = []
        self.best_ever: Optional[Chromosome] = None

    def initialize(self) -> None:
        """Create initial random population."""
        self.population = [
            Chromosome(params=self._random_params(), generation=0)
            for _ in range(self.config.population_size)
        ]

    def _random_params(self) -> dict:
        """Sample random parameter set within schema bounds."""
        params = {}
        for name, (lo, hi, dtype) in self.schema.items():
            val = self.rng.uniform(lo, hi)
            params[name] = dtype(val)
        return params

    def evaluate(self) -> None:
        """Evaluate fitness for all unevaluated chromosomes."""
        for chrom in self.population:
            if chrom.fitness == 0.0:
                try:
                    chrom.fitness = float(self.fitness_fn(chrom.params))
                except Exception:
                    chrom.fitness = -1e6

    def select(self, n: int) -> list[Chromosome]:
        """Tournament selection."""
        selected = []
        for _ in range(n):
            tournament = self.rng.choice(len(self.population), self.config.tournament_size, replace=False)
            winner = max(tournament, key=lambda i: self.population[i].fitness)
            selected.append(self.population[winner])
        return selected

    def crossover(self, parent_a: Chromosome, parent_b: Chromosome) -> tuple[Chromosome, Chromosome]:
        """Uniform + arithmetic crossover."""
        params_c, params_d = {}, {}
        for name in self.schema:
            if self.rng.random() < 0.5:
                # Arithmetic crossover
                alpha = self.rng.random()
                lo, hi, dtype = self.schema[name]
                v_c = alpha * parent_a.params[name] + (1 - alpha) * parent_b.params[name]
                v_d = (1 - alpha) * parent_a.params[name] + alpha * parent_b.params[name]
                params_c[name] = dtype(np.clip(v_c, lo, hi))
                params_d[name] = dtype(np.clip(v_d, lo, hi))
            else:
                # Uniform swap
                params_c[name] = parent_a.params[name]
                params_d[name] = parent_b.params[name]

        return (
            Chromosome(params=params_c, parent_ids=[parent_a.id, parent_b.id]),
            Chromosome(params=params_d, parent_ids=[parent_a.id, parent_b.id]),
        )

    def mutate(self, chrom: Chromosome) -> Chromosome:
        """Gaussian mutation with probability mutation_rate per gene."""
        params = chrom.params.copy()
        for name, (lo, hi, dtype) in self.schema.items():
            if self.rng.random() < self.config.mutation_rate:
                sigma = (hi - lo) * self.config.mutation_sigma
                val = params[name] + self.rng.normal(0, sigma)
                params[name] = dtype(np.clip(val, lo, hi))
        return Chromosome(params=params, parent_ids=[chrom.id])

    def step(self) -> dict:
        """One generation of evolution."""
        self.evaluate()

        # Sort by fitness
        self.population.sort(key=lambda c: c.fitness, reverse=True)

        # Update best ever
        if self.best_ever is None or self.population[0].fitness > self.best_ever.fitness:
            self.best_ever = self.population[0]

        # Elites
        n_elite = max(1, int(self.config.elite_frac * self.config.population_size))
        new_pop = self.population[:n_elite]

        # Fill rest via selection + crossover + mutation
        while len(new_pop) < self.config.population_size:
            parents = self.select(2)
            if self.rng.random() < self.config.crossover_rate:
                c1, c2 = self.crossover(parents[0], parents[1])
            else:
                c1 = Chromosome(params=parents[0].params.copy(), parent_ids=[parents[0].id])
                c2 = Chromosome(params=parents[1].params.copy(), parent_ids=[parents[1].id])

            c1 = self.mutate(c1)
            c2 = self.mutate(c2)
            new_pop.extend([c1, c2])

        self.population = new_pop[:self.config.population_size]

        # Generation stats
        fitnesses = [c.fitness for c in self.population[:n_elite + 10]]
        stats = {
            "generation": self.population[0].generation,
            "best_fitness": float(self.population[0].fitness),
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "best_params": self.population[0].params,
        }
        self.history.append(stats)

        # Increment generation counter
        for c in new_pop:
            c.generation = len(self.history)

        return stats

    def run(self, verbose: bool = False) -> Chromosome:
        """Run full evolution. Returns best chromosome found."""
        self.initialize()
        for gen in range(self.config.n_generations):
            stats = self.step()
            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: best={stats['best_fitness']:.4f}, mean={stats['mean_fitness']:.4f}")

        return self.best_ever or self.population[0]

    def pareto_front(self) -> list[Chromosome]:
        """
        Return approximate Pareto-optimal set (for multi-objective problems).
        Requires fitness_fn to return tuple of objectives.
        """
        # Re-evaluate for multi-objective
        front = []
        for c in self.population:
            dominated = False
            for other in self.population:
                if other is c:
                    continue
                # Assume fitness is scalar here; extend for multi-objective
                if other.fitness > c.fitness:
                    dominated = True
                    break
            if not dominated:
                front.append(c)
        return front


def _random_id() -> str:
    """Generate random chromosome ID."""
    import random
    import string
    return "chrom_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


# ── Example schemas ───────────────────────────────────────────────────────────

OU_PAIRS_SCHEMA = {
    "entry_z": (1.0, 4.0, float),
    "exit_z": (0.1, 1.5, float),
    "max_half_life": (5.0, 100.0, float),
    "min_half_life": (1.0, 15.0, float),
    "position_size_factor": (0.1, 2.0, float),
    "stop_loss_z": (3.0, 8.0, float),
}

AS_MM_SCHEMA = {
    "gamma": (0.01, 1.0, float),
    "k": (0.5, 5.0, float),
    "max_inventory": (1.0, 20.0, float),
    "quote_refresh_seconds": (1.0, 60.0, float),
    "vpin_threshold": (0.2, 0.8, float),
}

MOMENTUM_SCHEMA = {
    "lookback_fast": (3, 20, int),
    "lookback_slow": (15, 60, int),
    "entry_threshold": (0.01, 0.10, float),
    "stop_loss": (0.02, 0.15, float),
    "vol_lookback": (10, 60, int),
    "vol_scalar": (0.5, 3.0, float),
}
