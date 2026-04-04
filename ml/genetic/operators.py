"""
Genetic algorithm operators: adaptive mutation rate control, crowding distance,
niche preservation, and island model primitives.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .genome import StrategyGenome, GenomeFactory, MutationOperator, CrossoverOperator
from .population import Population, PopulationConfig, SelectionOperator, DiversityMetrics


# ---------------------------------------------------------------------------
# Adaptive mutation rate control
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveMutationConfig:
    """Configuration for adaptive mutation rate."""
    initial_rate: float = 0.1
    min_rate: float = 0.001
    max_rate: float = 0.5
    # 1/5 success rule parameters
    success_window: int = 20          # rolling window for measuring success
    success_threshold: float = 0.20   # target success ratio
    increase_factor: float = 1.22     # factor to increase rate on failure
    decrease_factor: float = 0.82     # factor to decrease rate on success
    # Diversity-based adaptation
    use_diversity_signal: bool = True
    low_diversity_threshold: float = 0.10
    high_diversity_threshold: float = 0.40
    # Stagnation-based adaptation
    use_stagnation_signal: bool = True
    stagnation_window: int = 15
    stagnation_boost_factor: float = 2.0


class AdaptiveMutationRate:
    """
    Adapts the mutation rate during evolution based on:
    1. 1/5 success rule (Rechenberg's rule)
    2. Population diversity
    3. Fitness stagnation
    """

    def __init__(self, config: AdaptiveMutationConfig) -> None:
        self.config = config
        self.rate = config.initial_rate
        self._success_history: deque = deque(maxlen=config.success_window)
        self._fitness_history: deque = deque(maxlen=config.stagnation_window)

    def record_offspring(self, parent_fitness: float,
                          offspring_fitness: float) -> None:
        """Record whether an offspring improved on its parent."""
        success = 1 if offspring_fitness > parent_fitness else 0
        self._success_history.append(success)

    def record_generation_best(self, best_fitness: float) -> None:
        self._fitness_history.append(best_fitness)

    def update(self, diversity: float = 0.25) -> float:
        """
        Update mutation rate based on success ratio, diversity, and stagnation.
        Returns new mutation rate.
        """
        # 1/5 success rule
        if len(self._success_history) >= self.config.success_window // 2:
            success_ratio = sum(self._success_history) / len(self._success_history)
            if success_ratio > self.config.success_threshold:
                self.rate *= self.config.decrease_factor
            else:
                self.rate *= self.config.increase_factor

        # Diversity-based adjustment
        if self.config.use_diversity_signal:
            if diversity < self.config.low_diversity_threshold:
                # Low diversity: increase mutation to escape local optima
                self.rate = min(self.rate * 1.5, self.config.max_rate)
            elif diversity > self.config.high_diversity_threshold:
                # High diversity: decrease mutation to focus search
                self.rate = max(self.rate * 0.9, self.config.min_rate)

        # Stagnation-based adjustment
        if (self.config.use_stagnation_signal and
                len(self._fitness_history) >= self.config.stagnation_window):
            window = list(self._fitness_history)
            recent_half = window[len(window) // 2:]
            early_half = window[:len(window) // 2]
            recent_mean = sum(recent_half) / len(recent_half)
            early_mean = sum(early_half) / len(early_half)
            improvement = abs(recent_mean - early_mean)
            if improvement < 1e-6:  # Stagnated
                self.rate = min(
                    self.rate * self.config.stagnation_boost_factor,
                    self.config.max_rate
                )

        # Clip to [min_rate, max_rate]
        self.rate = max(self.config.min_rate, min(self.config.max_rate, self.rate))
        return self.rate

    @property
    def current_rate(self) -> float:
        return self.rate


# ---------------------------------------------------------------------------
# Crossover rate scheduling
# ---------------------------------------------------------------------------

class CrossoverRateScheduler:
    """Schedules crossover probability over generations."""

    def __init__(self, initial_rate: float = 0.8, final_rate: float = 0.6,
                 n_generations: int = 100) -> None:
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.n_generations = n_generations

    def get_rate(self, generation: int) -> float:
        """Linear decay from initial to final rate."""
        progress = min(1.0, generation / max(self.n_generations, 1))
        return self.initial_rate + progress * (self.final_rate - self.initial_rate)

    def cosine_rate(self, generation: int) -> float:
        """Cosine annealing schedule."""
        progress = min(1.0, generation / max(self.n_generations, 1))
        cosine_factor = (1 + math.cos(math.pi * progress)) / 2
        return self.final_rate + cosine_factor * (self.initial_rate - self.final_rate)


# ---------------------------------------------------------------------------
# Adaptive operator selection
# ---------------------------------------------------------------------------

@dataclass
class OperatorRecord:
    """Tracks performance of a single operator."""
    name: str
    n_applications: int = 0
    total_improvement: float = 0.0
    n_successes: int = 0
    weight: float = 1.0

    @property
    def success_rate(self) -> float:
        return self.n_successes / max(self.n_applications, 1)

    @property
    def mean_improvement(self) -> float:
        return self.total_improvement / max(self.n_applications, 1)


class AdaptiveOperatorSelection:
    """
    Probability Matching (PM) / Adaptive Pursuit for selecting
    among multiple mutation and crossover operators based on their
    historical performance.
    """

    def __init__(self, mutation_methods: Optional[List[str]] = None,
                 crossover_methods: Optional[List[str]] = None,
                 learning_rate: float = 0.1,
                 min_weight: float = 0.05) -> None:
        if mutation_methods is None:
            mutation_methods = ["gaussian", "uniform", "polynomial", "creep", "boundary"]
        if crossover_methods is None:
            crossover_methods = ["uniform", "two_point", "arithmetic", "sbx"]

        self.mutation_records = {m: OperatorRecord(m) for m in mutation_methods}
        self.crossover_records = {m: OperatorRecord(m) for m in crossover_methods}
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self._rng = random.Random()

    def select_mutation_method(self) -> str:
        """Sample a mutation method proportional to its weight."""
        return self._weighted_choice(self.mutation_records)

    def select_crossover_method(self) -> str:
        """Sample a crossover method proportional to its weight."""
        return self._weighted_choice(self.crossover_records)

    def _weighted_choice(self, records: Dict[str, OperatorRecord]) -> str:
        total = sum(r.weight for r in records.values())
        pick = self._rng.uniform(0, total)
        cumulative = 0.0
        for name, record in records.items():
            cumulative += record.weight
            if cumulative >= pick:
                return name
        return list(records.keys())[-1]

    def record_outcome(self, operator_type: str, method: str,
                       parent_fitness: float, child_fitness: float) -> None:
        """Record the outcome of applying an operator."""
        if operator_type == "mutation":
            records = self.mutation_records
        else:
            records = self.crossover_records

        if method not in records:
            return

        record = records[method]
        record.n_applications += 1
        improvement = child_fitness - parent_fitness
        record.total_improvement += improvement
        if improvement > 0:
            record.n_successes += 1

        # Update weights using Probability Matching
        self._update_weights(records)

    def _update_weights(self, records: Dict[str, OperatorRecord]) -> None:
        """Update operator weights based on relative performance."""
        # Compute quality scores (mean improvement)
        qualities = {name: max(0.0, r.mean_improvement + r.success_rate)
                     for name, r in records.items()}
        total_quality = sum(qualities.values())

        if total_quality > 0:
            # Target probabilities proportional to quality
            target_probs = {name: q / total_quality for name, q in qualities.items()}
        else:
            # Uniform if no quality info yet
            n = len(records)
            target_probs = {name: 1.0 / n for name in records}

        # Move current weights toward target probabilities
        for name, record in records.items():
            target = target_probs[name]
            current = record.weight / sum(r.weight for r in records.values())
            new = current + self.learning_rate * (target - current)
            record.weight = max(self.min_weight, new)

    def get_operator_stats(self) -> Dict[str, Dict[str, float]]:
        """Return statistics for all operators."""
        stats = {}
        for name, record in {**self.mutation_records, **self.crossover_records}.items():
            stats[name] = {
                "n_applications": record.n_applications,
                "success_rate": record.success_rate,
                "mean_improvement": record.mean_improvement,
                "weight": record.weight,
            }
        return stats


# ---------------------------------------------------------------------------
# Niching operators
# ---------------------------------------------------------------------------

class NichePreservation:
    """
    Niche preservation techniques for maintaining population diversity.
    Implements deterministic crowding, clearing, and restricted mating.
    """

    @staticmethod
    def deterministic_crowding(
        population: List[StrategyGenome],
        offspring: List[StrategyGenome],
    ) -> List[StrategyGenome]:
        """
        Deterministic crowding: each offspring competes with the most similar
        individual in the population. Winner replaces loser.
        Works best when offspring are paired (pairs from crossover).
        """
        if not offspring:
            return population

        new_population = list(population)

        for child in offspring:
            # Find most similar individual in current population
            most_similar_idx = 0
            min_distance = float("inf")
            child_vec = child.chromosome.to_float_vector()

            for i, parent in enumerate(new_population):
                parent_vec = parent.chromosome.to_float_vector()
                if len(parent_vec) == len(child_vec):
                    d = math.sqrt(sum((a - b) ** 2
                                      for a, b in zip(child_vec, parent_vec)))
                    if d < min_distance:
                        min_distance = d
                        most_similar_idx = i

            # Compare fitness and keep better
            incumbent = new_population[most_similar_idx]
            child_fit = child.fitness if child.fitness is not None else float("-inf")
            incumbent_fit = incumbent.fitness if incumbent.fitness is not None else float("-inf")

            if child_fit > incumbent_fit:
                new_population[most_similar_idx] = child

        return new_population

    @staticmethod
    def clearing(population: List[StrategyGenome],
                 sigma_clear: float = 0.1,
                 capacity: int = 2) -> List[StrategyGenome]:
        """
        Clearing: only the 'capacity' best individuals in each niche
        retain their fitness; others are zeroed out.
        This encourages exploration of multiple niches.
        """
        if not population:
            return population

        result = [g.clone() for g in population]
        n = len(result)
        cleared = [False] * n

        # Sort by fitness descending
        order = sorted(range(n),
                       key=lambda i: result[i].fitness if result[i].fitness is not None else float("-inf"),
                       reverse=True)

        for i_rank, i in enumerate(order):
            if cleared[i]:
                continue
            # This individual is a niche winner (not cleared)
            # Clear all similar individuals within sigma_clear
            niche_count = 1
            vec_i = result[i].chromosome.to_float_vector()

            for j in order[i_rank + 1:]:
                if cleared[j]:
                    continue
                vec_j = result[j].chromosome.to_float_vector()
                if len(vec_i) == len(vec_j):
                    d = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_i, vec_j)) / max(len(vec_i), 1))
                    if d < sigma_clear:
                        niche_count += 1
                        if niche_count > capacity:
                            result[j].fitness = None  # Clear fitness
                            cleared[j] = True

        return result

    @staticmethod
    def restricted_mating(population: List[StrategyGenome],
                          sigma_mate: float = 0.3,
                          rng: Optional[random.Random] = None) -> Tuple[StrategyGenome, StrategyGenome]:
        """
        Restricted mating: only mate individuals that are within
        sigma_mate distance of each other in parameter space.
        Falls back to tournament selection if no close mate found.
        """
        _rng = rng or random.Random()
        if len(population) < 2:
            return population[0], population[0]

        parent1 = SelectionOperator.tournament(population, rng=_rng)
        vec1 = parent1.chromosome.to_float_vector()

        # Find potential mates within sigma_mate distance
        candidates = []
        for g in population:
            if g.metadata.genome_id == parent1.metadata.genome_id:
                continue
            vec2 = g.chromosome.to_float_vector()
            if len(vec1) == len(vec2):
                d = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)) / max(len(vec1), 1))
                if d <= sigma_mate:
                    candidates.append(g)

        if candidates:
            parent2 = SelectionOperator.tournament(candidates, rng=_rng)
        else:
            parent2 = SelectionOperator.tournament(population, rng=_rng)

        return parent1, parent2


# ---------------------------------------------------------------------------
# Island model
# ---------------------------------------------------------------------------

@dataclass
class IslandConfig:
    """Configuration for a single island."""
    island_id: int
    population_size: int = 50
    mutation_method: str = "gaussian"
    crossover_method: str = "uniform"
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_method: str = "tournament"
    # Each island can specialize in different search behavior
    specialize: str = "default"   # default, explorer, exploiter, random


@dataclass
class MigrationConfig:
    """Configuration for island migration."""
    migration_interval: int = 20    # migrate every N generations
    migration_fraction: float = 0.10
    migration_topology: str = "ring"  # ring, fully_connected, random, star
    n_islands: int = 4
    migration_selection: str = "best"  # best, random, tournament


class IslandModel:
    """
    Island model GA: multiple sub-populations evolve independently
    with periodic migration of individuals between islands.
    """

    def __init__(self, migration_config: MigrationConfig,
                 island_configs: Optional[List[IslandConfig]] = None,
                 factory: Optional[GenomeFactory] = None,
                 seed: Optional[int] = None) -> None:
        self.migration_config = migration_config
        self._rng = random.Random(seed)
        self.factory = factory
        self.generation = 0
        self.n_islands = migration_config.n_islands

        # Create island configs if not provided
        if island_configs is None:
            island_configs = self._default_island_configs()
        self.island_configs = island_configs

        # Initialize island populations
        self.islands: List[List[StrategyGenome]] = []
        self.island_stats: List[Dict[str, Any]] = [{}] * self.n_islands

    def _default_island_configs(self) -> List[IslandConfig]:
        """Create diverse island configurations."""
        specializations = [
            ("gaussian", "uniform", 0.05, 0.9, "exploiter"),
            ("polynomial", "sbx", 0.15, 0.8, "default"),
            ("gaussian", "arithmetic", 0.20, 0.7, "explorer"),
            ("uniform", "two_point", 0.30, 0.6, "random"),
        ]
        configs = []
        for i in range(self.n_islands):
            spec = specializations[i % len(specializations)]
            configs.append(IslandConfig(
                island_id=i,
                population_size=50,
                mutation_method=spec[0],
                crossover_method=spec[1],
                mutation_rate=spec[2],
                crossover_rate=spec[3],
                specialize=spec[4],
            ))
        return configs

    def initialize(self, base_population: Optional[List[StrategyGenome]] = None) -> None:
        """Initialize all islands."""
        if self.factory is None:
            raise ValueError("factory required to initialize islands")

        if base_population:
            # Distribute base population across islands
            self.islands = []
            n = len(base_population)
            for i in range(self.n_islands):
                cfg = self.island_configs[i]
                # Start with slice of base population
                start = (i * n) // self.n_islands
                end = ((i + 1) * n) // self.n_islands
                island = [g.clone() for g in base_population[start:end]]
                # Fill remainder with random
                while len(island) < cfg.population_size:
                    island.append(self.factory.create_random())
                self.islands.append(island[:cfg.population_size])
        else:
            self.islands = []
            for cfg in self.island_configs:
                island = self.factory.create_population(cfg.population_size)
                self.islands.append(island)

    def should_migrate(self, generation: int) -> bool:
        return generation > 0 and generation % self.migration_config.migration_interval == 0

    def migrate(self) -> int:
        """
        Perform migration between islands according to the topology.
        Returns total number of individuals migrated.
        """
        n_migrate = max(1, int(self.migration_config.migration_fraction *
                               min(len(island) for island in self.islands)))
        topology = self.migration_config.migration_topology
        total_migrated = 0

        # Build migration pairs based on topology
        pairs = self._get_migration_pairs(topology)

        for src_idx, dst_idx in pairs:
            migrants = self._select_migrants(self.islands[src_idx], n_migrate)
            self._insert_migrants(self.islands[dst_idx], migrants)
            total_migrated += len(migrants)

        return total_migrated

    def _get_migration_pairs(self, topology: str) -> List[Tuple[int, int]]:
        """Get (source, destination) island pairs for migration."""
        n = self.n_islands
        if topology == "ring":
            return [(i, (i + 1) % n) for i in range(n)]
        elif topology == "fully_connected":
            return [(i, j) for i in range(n) for j in range(n) if i != j]
        elif topology == "random":
            pairs = []
            for i in range(n):
                j = self._rng.randint(0, n - 1)
                if j != i:
                    pairs.append((i, j))
            return pairs
        elif topology == "star":
            # Hub = island 0; all others send to and receive from hub
            pairs = [(0, i) for i in range(1, n)]
            pairs += [(i, 0) for i in range(1, n)]
            return pairs
        else:
            return [(i, (i + 1) % n) for i in range(n)]

    def _select_migrants(self, island: List[StrategyGenome],
                         n: int) -> List[StrategyGenome]:
        """Select individuals to migrate from an island."""
        strategy = self.migration_config.migration_selection
        evaluated = [g for g in island if g.fitness is not None]
        if not evaluated:
            return [g.clone() for g in self._rng.sample(island, min(n, len(island)))]

        if strategy == "best":
            sorted_island = sorted(evaluated,
                                   key=lambda g: g.fitness,  # type: ignore
                                   reverse=True)
            return [g.clone() for g in sorted_island[:n]]
        elif strategy == "random":
            return [g.clone() for g in self._rng.sample(evaluated, min(n, len(evaluated)))]
        elif strategy == "tournament":
            migrants = []
            for _ in range(n):
                winner = SelectionOperator.tournament(evaluated, rng=self._rng)
                migrants.append(winner.clone())
            return migrants
        else:
            return [g.clone() for g in evaluated[:n]]

    def _insert_migrants(self, island: List[StrategyGenome],
                          migrants: List[StrategyGenome]) -> None:
        """Insert migrants into an island, replacing worst individuals."""
        if not island:
            island.extend(migrants)
            return

        # Replace worst individuals
        evaluated = [(i, g) for i, g in enumerate(island) if g.fitness is not None]
        if evaluated:
            worst_pairs = sorted(evaluated,
                                 key=lambda x: x[1].fitness)  # type: ignore
            for k, migrant in enumerate(migrants):
                if k < len(worst_pairs):
                    replace_idx = worst_pairs[k][0]
                    island[replace_idx] = migrant
                else:
                    island.append(migrant)
        else:
            island.extend(migrants)

    def get_best_individual(self) -> Optional[StrategyGenome]:
        """Return best individual across all islands."""
        all_genomes = [g for island in self.islands for g in island
                       if g.fitness is not None]
        if not all_genomes:
            return None
        return max(all_genomes, key=lambda g: g.fitness)  # type: ignore

    def get_all_individuals(self) -> List[StrategyGenome]:
        """Return all individuals from all islands combined."""
        return [g for island in self.islands for g in island]

    def island_diversity(self, island_idx: int) -> float:
        """Compute diversity of a specific island."""
        return DiversityMetrics.mean_pairwise_distance(self.islands[island_idx], sample_size=20)

    def inter_island_diversity(self) -> float:
        """Compute diversity between island bests."""
        bests = []
        for island in self.islands:
            evaluated = [g for g in island if g.fitness is not None]
            if evaluated:
                bests.append(max(evaluated, key=lambda g: g.fitness))  # type: ignore
        return DiversityMetrics.mean_pairwise_distance(bests)


# ---------------------------------------------------------------------------
# Evolutionary strategy (sigma adaptation, derandomized)
# ---------------------------------------------------------------------------

class CovarianceMatrixAdaptation:
    """
    Simplified CMA-ES step size adaptation.
    Tracks cumulative path lengths to adapt mutation magnitude.
    Not a full CMA-ES, but uses the 1/5 rule with cumulative adaptation.
    """

    def __init__(self, n_dims: int, initial_sigma: float = 0.3) -> None:
        self.n_dims = n_dims
        self.sigma = initial_sigma
        self._evolution_path = [0.0] * n_dims
        self._c_sigma = 4.0 / (n_dims + 4.0)    # cumulation constant
        self._d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((1 - 1.0 / n_dims) - 1.0)) + self._c_sigma
        self._chi_n = math.sqrt(n_dims) * (1 - 1.0 / (4 * n_dims) + 1.0 / (21 * n_dims ** 2))

    def update(self, step_direction: List[float]) -> None:
        """Update cumulative path and adjust sigma."""
        c = self._c_sigma
        # Update evolution path (cumulative step size control)
        self._evolution_path = [
            (1 - c) * ep + math.sqrt(c * (2 - c)) * s
            for ep, s in zip(self._evolution_path, step_direction)
        ]
        # Path length
        path_len = math.sqrt(sum(ep ** 2 for ep in self._evolution_path))
        # Adapt sigma
        self.sigma *= math.exp(
            (c / self._d_sigma) * (path_len / self._chi_n - 1)
        )
        # Clip sigma to reasonable range
        self.sigma = max(1e-6, min(1.0, self.sigma))

    def sample_step(self, rng: random.Random) -> List[float]:
        """Sample a random step scaled by sigma."""
        return [self.sigma * rng.gauss(0.0, 1.0) for _ in range(self.n_dims)]


# ---------------------------------------------------------------------------
# Restart strategies
# ---------------------------------------------------------------------------

class RestartStrategy:
    """
    Handles population restarts when evolution stagnates.
    """

    def __init__(self, stagnation_patience: int = 30,
                 min_improvement: float = 1e-5,
                 restart_method: str = "partial") -> None:
        self.stagnation_patience = stagnation_patience
        self.min_improvement = min_improvement
        self.restart_method = restart_method
        self._stagnation_counter = 0
        self._best_fitness_seen = float("-inf")
        self._n_restarts = 0

    def check_stagnation(self, current_best_fitness: float) -> bool:
        """
        Return True if a restart should be triggered.
        """
        improvement = current_best_fitness - self._best_fitness_seen
        if improvement > self.min_improvement:
            self._best_fitness_seen = current_best_fitness
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1
        return self._stagnation_counter >= self.stagnation_patience

    def apply_restart(self, population: List[StrategyGenome],
                      factory: GenomeFactory,
                      elite_fraction: float = 0.10,
                      rng: Optional[random.Random] = None) -> List[StrategyGenome]:
        """
        Apply restart: replace most of the population with new individuals,
        preserving elites.
        """
        _rng = rng or random.Random()
        self._stagnation_counter = 0
        self._n_restarts += 1

        if self.restart_method == "full":
            # Keep only best individual
            evaluated = [g for g in population if g.fitness is not None]
            if evaluated:
                best = max(evaluated, key=lambda g: g.fitness)  # type: ignore
                new_pop = [best.clone()]
            else:
                new_pop = []
            while len(new_pop) < len(population):
                new_pop.append(factory.create_random())
            return new_pop[:len(population)]

        elif self.restart_method == "partial":
            # Keep top elite_fraction
            n_elite = max(1, int(elite_fraction * len(population)))
            evaluated = [g for g in population if g.fitness is not None]
            if evaluated:
                sorted_pop = sorted(evaluated, key=lambda g: g.fitness,  # type: ignore
                                    reverse=True)
                elites = [g.clone() for g in sorted_pop[:n_elite]]
            else:
                elites = []
            new_pop = elites
            while len(new_pop) < len(population):
                new_pop.append(factory.create_random())
            return new_pop[:len(population)]

        elif self.restart_method == "perturbed":
            # Mutate all non-elite individuals heavily
            n_elite = max(1, int(elite_fraction * len(population)))
            evaluated = [g for g in population if g.fitness is not None]
            if evaluated:
                sorted_pop = sorted(evaluated, key=lambda g: g.fitness,  # type: ignore
                                    reverse=True)
                new_pop = [g.clone() for g in sorted_pop[:n_elite]]
                for g in sorted_pop[n_elite:]:
                    mutated = g.mutate(0.5, method="gaussian", sigma_scale=0.3, rng=_rng)
                    mutated.fitness = None  # Force re-evaluation
                    new_pop.append(mutated)
            else:
                new_pop = [factory.create_random() for _ in range(len(population))]
            return new_pop[:len(population)]

        else:
            return population

    @property
    def n_restarts(self) -> int:
        return self._n_restarts

    @property
    def stagnation_counter(self) -> int:
        return self._stagnation_counter


# ---------------------------------------------------------------------------
# Reproductive operator dispatcher
# ---------------------------------------------------------------------------

@dataclass
class OperatorConfig:
    """Full operator configuration for one evolution step."""
    mutation_method: str = "gaussian"
    crossover_method: str = "uniform"
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    sigma_scale: float = 0.1
    polynomial_eta: float = 20.0
    sbx_eta: float = 2.0
    use_adaptive_operators: bool = True
    use_elitism: bool = True
    elite_fraction: float = 0.10


class ReproductionOperator:
    """
    Applies crossover and mutation to produce offspring from a population.
    Supports adaptive operator selection.
    """

    def __init__(self, config: OperatorConfig,
                 adaptive_operators: Optional[AdaptiveOperatorSelection] = None,
                 rng: Optional[random.Random] = None) -> None:
        self.config = config
        self.adaptive_ops = adaptive_operators
        self._rng = rng or random.Random()

    def produce_offspring(self,
                          population: List[StrategyGenome],
                          n_offspring: int,
                          fitness_evaluator: Optional[Callable] = None) -> List[StrategyGenome]:
        """
        Produce n_offspring from population using crossover and mutation.
        """
        offspring = []

        while len(offspring) < n_offspring:
            # Select parents
            p1 = SelectionOperator.tournament(population, rng=self._rng)
            p2 = SelectionOperator.tournament(population, rng=self._rng)

            # Choose operators
            if self.config.use_adaptive_operators and self.adaptive_ops:
                xover_method = self.adaptive_ops.select_crossover_method()
                mut_method = self.adaptive_ops.select_mutation_method()
            else:
                xover_method = self.config.crossover_method
                mut_method = self.config.mutation_method

            # Crossover
            if self._rng.random() < self.config.crossover_rate:
                c1, c2 = StrategyGenome.crossover(p1, p2, method=xover_method, rng=self._rng)
            else:
                c1, c2 = p1.clone(), p2.clone()
                c1.fitness = None
                c2.fitness = None

            # Mutation
            c1 = c1.mutate(self.config.mutation_rate, method=mut_method,
                           sigma_scale=self.config.sigma_scale,
                           eta=self.config.polynomial_eta, rng=self._rng)
            c2 = c2.mutate(self.config.mutation_rate, method=mut_method,
                           sigma_scale=self.config.sigma_scale,
                           eta=self.config.polynomial_eta, rng=self._rng)

            # Evaluate if evaluator provided
            if fitness_evaluator:
                fitness_evaluator(c1)
                fitness_evaluator(c2)
                # Record outcomes for adaptive operator selection
                if self.adaptive_ops:
                    p1_fit = p1.fitness or 0.0
                    p2_fit = p2.fitness or 0.0
                    c1_fit = c1.fitness or 0.0
                    c2_fit = c2.fitness or 0.0
                    self.adaptive_ops.record_outcome("crossover", xover_method,
                                                     (p1_fit + p2_fit) / 2, c1_fit)
                    self.adaptive_ops.record_outcome("mutation", mut_method,
                                                     p1_fit, c1_fit)

            offspring.extend([c1, c2])

        return offspring[:n_offspring]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Operators self-test ===")
    from .genome import GenomeFactory
    from .fitness import FitnessEvaluator, FitnessConfig

    factory = GenomeFactory("momentum", seed=42)
    pop = factory.create_population(30)

    # Create fitness evaluator
    prices = FitnessEvaluator._generate_synthetic_prices(300, seed=42)
    evaluator = FitnessEvaluator(FitnessConfig(), price_data=prices, strategy_type="momentum")
    evaluator.evaluate_population(pop)

    print(f"Initial population: {len(pop)} genomes evaluated")

    # Test adaptive mutation rate
    amc = AdaptiveMutationConfig(initial_rate=0.1)
    adaptive_mut = AdaptiveMutationRate(amc)
    for g in pop[:10]:
        parent_fit = g.fitness or 0.0
        child = g.mutate(adaptive_mut.current_rate, method="gaussian")
        evaluator.evaluate(child)
        adaptive_mut.record_offspring(parent_fit, child.fitness or 0.0)
    adaptive_mut.record_generation_best(max(g.fitness for g in pop if g.fitness))
    new_rate = adaptive_mut.update(diversity=0.2)
    print(f"Adaptive mutation rate: {new_rate:.4f}")

    # Test adaptive operator selection
    aos = AdaptiveOperatorSelection()
    for g in pop[:10]:
        method = aos.select_mutation_method()
        child = g.mutate(0.15, method=method)
        evaluator.evaluate(child)
        aos.record_outcome("mutation", method, g.fitness or 0.0, child.fitness or 0.0)
    stats = aos.get_operator_stats()
    print(f"\nAdaptive operator stats (sample): {list(stats.items())[:2]}")

    # Test niche preservation
    print("\nTesting niche preservation...")
    offspring = factory.create_population(10)
    evaluator.evaluate_population(offspring)
    dc_result = NichePreservation.deterministic_crowding(pop[:10], offspring[:5])
    print(f"Deterministic crowding: {len(dc_result)} individuals")

    clearing_result = NichePreservation.clearing(pop[:15], sigma_clear=0.15, capacity=2)
    cleared = sum(1 for g in clearing_result if g.fitness is None)
    print(f"Clearing: {cleared} individuals cleared")

    p1, p2 = NichePreservation.restricted_mating(pop, sigma_mate=0.3)
    print(f"Restricted mating: selected {p1.fitness:.4f}, {p2.fitness:.4f}")

    # Test island model
    print("\nTesting island model...")
    mc = MigrationConfig(migration_interval=5, n_islands=4, migration_fraction=0.1)
    islands = IslandModel(mc, factory=factory, seed=42)
    islands.initialize()
    for island in islands.islands:
        evaluator.evaluate_population(island)
    print(f"Islands initialized: {[len(i) for i in islands.islands]}")

    n_migrated = islands.migrate()
    print(f"Migrated {n_migrated} individuals")
    best = islands.get_best_individual()
    print(f"Best across islands: {best.fitness:.4f}")

    # Test restart strategy
    print("\nTesting restart strategy...")
    restart = RestartStrategy(stagnation_patience=5)
    for _ in range(6):
        should = restart.check_stagnation(0.5)  # same fitness = stagnation
    print(f"Stagnation triggered: {should}, restarts: {restart.n_restarts}")
    if should:
        new_pop = restart.apply_restart(pop, factory, elite_fraction=0.1)
        print(f"Restart result: {len(new_pop)} individuals")

    # Test reproduction operator
    print("\nTesting reproduction operator...")
    op_config = OperatorConfig(mutation_rate=0.1, crossover_rate=0.8,
                                use_adaptive_operators=True)
    repro = ReproductionOperator(op_config, adaptive_operators=aos)
    offspring = repro.produce_offspring(pop, n_offspring=10)
    print(f"Produced {len(offspring)} offspring")

    # Test CMA step size adaptation
    cma = CovarianceMatrixAdaptation(n_dims=10, initial_sigma=0.2)
    step = cma.sample_step(random.Random(42))
    cma.update(step)
    print(f"\nCMA sigma after update: {cma.sigma:.4f}")

    print("\nAll operator tests passed.")
