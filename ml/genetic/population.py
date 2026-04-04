"""
Population management for genetic algorithm optimization.

Provides tournament selection, elitism, fitness sharing, hall of fame,
and population diversity tracking.
"""

from __future__ import annotations

import heapq
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
)

from .genome import StrategyGenome, GenomeFactory, ParamRange


# ---------------------------------------------------------------------------
# Hall of Fame: tracks all-time best individuals
# ---------------------------------------------------------------------------

class HallOfFame:
    """
    Stores the best N unique individuals ever seen during evolution.
    Uses fingerprinting to avoid duplicate entries.
    """

    def __init__(self, maxsize: int = 50) -> None:
        self.maxsize = maxsize
        self._entries: List[StrategyGenome] = []
        self._fingerprints: Set[str] = set()

    def update(self, genomes: Iterable[StrategyGenome]) -> int:
        """Add all eligible genomes. Returns number of new entries added."""
        added = 0
        for genome in genomes:
            if genome.fitness is None:
                continue
            fp = genome.fingerprint()
            if fp in self._fingerprints:
                continue
            self._fingerprints.add(fp)
            self._entries.append(genome)
            added += 1

        # Sort descending by fitness and keep top maxsize
        self._entries.sort(key=lambda g: g.fitness if g.fitness is not None else float("-inf"),
                           reverse=True)
        if len(self._entries) > self.maxsize:
            removed = self._entries[self.maxsize:]
            self._entries = self._entries[:self.maxsize]
            for g in removed:
                self._fingerprints.discard(g.fingerprint())

        return added

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, idx: int) -> StrategyGenome:
        return self._entries[idx]

    @property
    def best(self) -> Optional[StrategyGenome]:
        return self._entries[0] if self._entries else None

    @property
    def top_k(self) -> List[StrategyGenome]:
        return list(self._entries)

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [
            {"rank": i + 1, "fitness": g.fitness, "params": g.chromosome.to_dict(),
             "genome_id": g.metadata.genome_id, "generation": g.metadata.generation}
            for i, g in enumerate(self._entries)
        ]


# ---------------------------------------------------------------------------
# Selection operators
# ---------------------------------------------------------------------------

class SelectionOperator:
    """Collection of parent selection methods."""

    @staticmethod
    def tournament(population: List[StrategyGenome], k: int = 3,
                   rng: Optional[random.Random] = None) -> StrategyGenome:
        """
        Tournament selection: sample k individuals without replacement,
        return the one with highest fitness.
        """
        _rng = rng or random.Random()
        contestants = _rng.sample(population, min(k, len(population)))
        winner = max(contestants,
                     key=lambda g: g.fitness if g.fitness is not None else float("-inf"))
        return winner

    @staticmethod
    def tournament_pair(population: List[StrategyGenome], k: int = 3,
                        rng: Optional[random.Random] = None) -> Tuple[StrategyGenome, StrategyGenome]:
        """Select two parents via separate tournaments."""
        _rng = rng or random.Random()
        p1 = SelectionOperator.tournament(population, k, _rng)
        # Ensure we get a different individual for p2 if possible
        for _ in range(10):
            p2 = SelectionOperator.tournament(population, k, _rng)
            if p2.metadata.genome_id != p1.metadata.genome_id or len(population) == 1:
                break
        return p1, p2

    @staticmethod
    def roulette_wheel(population: List[StrategyGenome],
                       rng: Optional[random.Random] = None) -> StrategyGenome:
        """
        Fitness-proportionate (roulette wheel) selection.
        Shifts all fitnesses so minimum is 0 to handle negatives.
        """
        _rng = rng or random.Random()
        fitnesses = [g.fitness if g.fitness is not None else 0.0 for g in population]
        min_f = min(fitnesses)
        # Shift so all are non-negative
        shifted = [f - min_f + 1e-10 for f in fitnesses]
        total = sum(shifted)
        pick = _rng.uniform(0, total)
        cumulative = 0.0
        for genome, s in zip(population, shifted):
            cumulative += s
            if cumulative >= pick:
                return genome
        return population[-1]

    @staticmethod
    def rank_based(population: List[StrategyGenome],
                   selection_pressure: float = 2.0,
                   rng: Optional[random.Random] = None) -> StrategyGenome:
        """
        Rank-based selection: assign selection probability based on rank.
        selection_pressure in [1, 2]: 1=uniform, 2=strongly biased toward best.
        """
        _rng = rng or random.Random()
        n = len(population)
        sorted_pop = sorted(population,
                            key=lambda g: g.fitness if g.fitness is not None else float("-inf"))
        # Rank 1 = worst, rank n = best
        probs = []
        for rank in range(1, n + 1):
            p = (2 - selection_pressure) / n + 2 * rank * (selection_pressure - 1) / (n * (n - 1))
            probs.append(max(0.0, p))
        total = sum(probs)
        probs = [p / total for p in probs]
        pick = _rng.random()
        cumulative = 0.0
        for genome, p in zip(sorted_pop, probs):
            cumulative += p
            if cumulative >= pick:
                return genome
        return sorted_pop[-1]

    @staticmethod
    def stochastic_universal_sampling(population: List[StrategyGenome],
                                      n_select: int,
                                      rng: Optional[random.Random] = None) -> List[StrategyGenome]:
        """
        Stochastic Universal Sampling (SUS): selects n individuals with equal spacing
        on the roulette wheel. Reduces sampling variance compared to roulette wheel.
        """
        _rng = rng or random.Random()
        fitnesses = [g.fitness if g.fitness is not None else 0.0 for g in population]
        min_f = min(fitnesses)
        shifted = [f - min_f + 1e-10 for f in fitnesses]
        total = sum(shifted)

        step = total / n_select
        start = _rng.uniform(0, step)
        pointers = [start + i * step for i in range(n_select)]

        selected = []
        cumulative = 0.0
        ptr_idx = 0
        for genome, s in zip(population, shifted):
            cumulative += s
            while ptr_idx < len(pointers) and cumulative >= pointers[ptr_idx]:
                selected.append(genome)
                ptr_idx += 1
        # Pad if needed (floating point edge cases)
        while len(selected) < n_select:
            selected.append(population[-1])

        _rng.shuffle(selected)
        return selected

    @staticmethod
    def lexicase(population: List[StrategyGenome],
                 case_fitnesses: Optional[List[List[float]]] = None,
                 rng: Optional[random.Random] = None) -> StrategyGenome:
        """
        Lexicase selection: filter population by test cases in random order.
        case_fitnesses[i][j] = fitness of individual i on test case j.
        Falls back to tournament if case_fitnesses not provided.
        """
        _rng = rng or random.Random()
        if case_fitnesses is None or not case_fitnesses:
            return SelectionOperator.tournament(population, rng=_rng)

        n_cases = len(case_fitnesses[0])
        case_order = list(range(n_cases))
        _rng.shuffle(case_order)

        candidates = list(range(len(population)))
        for case_idx in case_order:
            if len(candidates) == 1:
                break
            case_scores = [case_fitnesses[i][case_idx] for i in candidates]
            best_score = max(case_scores)
            # epsilon-lexicase: accept within mad of best
            abs_deviations = [abs(s - best_score) for s in case_scores]
            mad = statistics.median(abs_deviations) if abs_deviations else 0.0
            threshold = best_score - mad
            candidates = [i for i, s in zip(candidates, case_scores) if s >= threshold]

        winner_idx = _rng.choice(candidates)
        return population[winner_idx]


# ---------------------------------------------------------------------------
# Fitness sharing / niche preservation
# ---------------------------------------------------------------------------

class FitnessSharing:
    """
    Fitness sharing: reduce fitness of individuals in dense regions of
    parameter space to promote population diversity.
    """

    def __init__(self, sigma_share: float = 0.1, alpha: float = 1.0) -> None:
        """
        sigma_share: niche radius in normalized parameter space [0,1].
        alpha: shape parameter (1.0 = linear, >1 = sharper boundary).
        """
        self.sigma_share = sigma_share
        self.alpha = alpha

    def sharing_function(self, distance: float) -> float:
        """Standard triangular sharing function."""
        if distance >= self.sigma_share:
            return 0.0
        return 1.0 - (distance / self.sigma_share) ** self.alpha

    @staticmethod
    def euclidean_distance_normalized(g1: StrategyGenome, g2: StrategyGenome) -> float:
        """Euclidean distance in normalized [0,1]^n parameter space."""
        v1 = g1.chromosome.to_float_vector()
        v2 = g2.chromosome.to_float_vector()
        if len(v1) != len(v2) or not v1:
            return 0.0
        sq_sum = sum((a - b) ** 2 for a, b in zip(v1, v2))
        return math.sqrt(sq_sum / len(v1))

    def compute_niche_counts(self, population: List[StrategyGenome]) -> List[float]:
        """Compute niche count m_i for each individual."""
        n = len(population)
        niche_counts = [0.0] * n
        for i in range(n):
            for j in range(n):
                d = self.euclidean_distance_normalized(population[i], population[j])
                niche_counts[i] += self.sharing_function(d)
        return niche_counts

    def apply(self, population: List[StrategyGenome]) -> None:
        """Modify genome fitness in-place by dividing by niche count."""
        niche_counts = self.compute_niche_counts(population)
        for genome, nc in zip(population, niche_counts):
            if genome.fitness is not None and nc > 0:
                genome.niche_count = nc
                genome.fitness = genome.fitness / nc


# ---------------------------------------------------------------------------
# Population statistics and diversity metrics
# ---------------------------------------------------------------------------

@dataclass
class PopulationStats:
    """Summary statistics of a population at one generation."""
    generation: int
    size: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    worst_fitness: float
    median_fitness: float
    fitness_range: float
    param_diversity: float          # mean pairwise distance
    unique_fingerprints: int
    elite_count: int
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"Gen {self.generation:4d} | "
            f"Best: {self.best_fitness:8.4f} | "
            f"Mean: {self.mean_fitness:8.4f} ± {self.std_fitness:6.4f} | "
            f"Diversity: {self.param_diversity:.4f} | "
            f"Unique: {self.unique_fingerprints}/{self.size}"
        )


class DiversityMetrics:
    """Compute diversity metrics for a population."""

    @staticmethod
    def mean_pairwise_distance(population: List[StrategyGenome],
                               sample_size: int = 50) -> float:
        """
        Estimate mean pairwise Euclidean distance in normalized parameter space.
        Uses sampling for large populations to keep cost O(sample_size^2).
        """
        if len(population) < 2:
            return 0.0
        _rng = random.Random()
        sample = _rng.sample(population, min(sample_size, len(population)))
        total = 0.0
        count = 0
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                v1 = sample[i].chromosome.to_float_vector()
                v2 = sample[j].chromosome.to_float_vector()
                if len(v1) == len(v2) and v1:
                    d = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)) / len(v1))
                    total += d
                    count += 1
        return total / count if count > 0 else 0.0

    @staticmethod
    def entropy(population: List[StrategyGenome], n_bins: int = 10) -> float:
        """
        Approximate entropy of the population distribution using binned histograms
        over each parameter dimension, then average across dimensions.
        """
        if len(population) < 2:
            return 0.0
        n_params = len(population[0].chromosome.genes)
        total_entropy = 0.0
        for dim in range(n_params):
            vals = [g.chromosome.to_float_vector()[dim] for g in population]
            counts = [0] * n_bins
            for v in vals:
                bin_idx = min(int(v * n_bins), n_bins - 1)
                counts[bin_idx] += 1
            total = len(vals)
            entropy = 0.0
            for c in counts:
                if c > 0:
                    p = c / total
                    entropy -= p * math.log2(p)
            total_entropy += entropy
        return total_entropy / n_params

    @staticmethod
    def phenotypic_diversity(population: List[StrategyGenome]) -> float:
        """Std dev of fitness values normalized by mean."""
        fitnesses = [g.fitness for g in population if g.fitness is not None]
        if len(fitnesses) < 2:
            return 0.0
        mean_f = sum(fitnesses) / len(fitnesses)
        if abs(mean_f) < 1e-10:
            return 0.0
        variance = sum((f - mean_f) ** 2 for f in fitnesses) / len(fitnesses)
        return math.sqrt(variance) / abs(mean_f)

    @staticmethod
    def compute_stats(population: List[StrategyGenome],
                      generation: int) -> PopulationStats:
        """Compute full population statistics."""
        evaluated = [g for g in population if g.fitness is not None]
        if not evaluated:
            return PopulationStats(
                generation=generation, size=len(population),
                best_fitness=0.0, mean_fitness=0.0, std_fitness=0.0,
                worst_fitness=0.0, median_fitness=0.0, fitness_range=0.0,
                param_diversity=0.0, unique_fingerprints=0, elite_count=0,
            )
        fitnesses = [g.fitness for g in evaluated]
        mean_f = sum(fitnesses) / len(fitnesses)
        std_f = math.sqrt(sum((f - mean_f) ** 2 for f in fitnesses) / len(fitnesses))
        best_f = max(fitnesses)
        # Elite count: top 10%
        n_elite = max(1, int(0.1 * len(evaluated)))
        elite_threshold = sorted(fitnesses, reverse=True)[n_elite - 1]
        elite_count = sum(1 for f in fitnesses if f >= elite_threshold)

        diversity = DiversityMetrics.mean_pairwise_distance(population, sample_size=30)
        fingerprints = set(g.fingerprint() for g in population)

        return PopulationStats(
            generation=generation,
            size=len(population),
            best_fitness=best_f,
            mean_fitness=mean_f,
            std_fitness=std_f,
            worst_fitness=min(fitnesses),
            median_fitness=sorted(fitnesses)[len(fitnesses) // 2],
            fitness_range=best_f - min(fitnesses),
            param_diversity=diversity,
            unique_fingerprints=len(fingerprints),
            elite_count=elite_count,
        )


# ---------------------------------------------------------------------------
# Population manager
# ---------------------------------------------------------------------------

@dataclass
class PopulationConfig:
    size: int = 100
    elite_fraction: float = 0.10       # top fraction preserved each generation
    tournament_size: int = 5
    selection_method: str = "tournament"  # tournament, roulette, rank, sus
    fitness_sharing: bool = False
    sharing_sigma: float = 0.15
    diversity_injection_interval: int = 20  # inject random individuals every N gens
    diversity_injection_fraction: float = 0.05
    deduplication: bool = True
    max_duplicate_fraction: float = 0.30
    min_unique_fraction: float = 0.70
    seed: Optional[int] = None


class Population:
    """
    Manages a population of StrategyGenomes through selection, reproduction,
    and replacement.
    """

    def __init__(self, config: PopulationConfig,
                 factory: GenomeFactory) -> None:
        self.config = config
        self.factory = factory
        self._rng = random.Random(config.seed)
        self.individuals: List[StrategyGenome] = []
        self.hall_of_fame = HallOfFame(maxsize=100)
        self.generation = 0
        self.stats_history: List[PopulationStats] = []
        self._fitness_sharing = FitnessSharing(
            sigma_share=config.sharing_sigma) if config.fitness_sharing else None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, seeds: Optional[List[Dict[str, Any]]] = None) -> None:
        """Create the initial population."""
        self.individuals = self.factory.create_seeded_population(
            self.config.size, seeds)
        self.generation = 0

    def reset(self) -> None:
        self.individuals = []
        self.generation = 0
        self.stats_history = []
        self.hall_of_fame = HallOfFame(maxsize=100)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_parents(self) -> Tuple[StrategyGenome, StrategyGenome]:
        """Select two parents using the configured selection method."""
        evaluated = [g for g in self.individuals if g.fitness is not None]
        if not evaluated:
            evaluated = self.individuals
        method = self.config.selection_method
        if method == "tournament":
            return SelectionOperator.tournament_pair(
                evaluated, self.config.tournament_size, self._rng)
        elif method == "roulette":
            p1 = SelectionOperator.roulette_wheel(evaluated, self._rng)
            p2 = SelectionOperator.roulette_wheel(evaluated, self._rng)
            return p1, p2
        elif method == "rank":
            p1 = SelectionOperator.rank_based(evaluated, rng=self._rng)
            p2 = SelectionOperator.rank_based(evaluated, rng=self._rng)
            return p1, p2
        elif method == "sus":
            selected = SelectionOperator.stochastic_universal_sampling(
                evaluated, 2, self._rng)
            return selected[0], selected[1]
        else:
            return SelectionOperator.tournament_pair(
                evaluated, self.config.tournament_size, self._rng)

    def select_elite(self) -> List[StrategyGenome]:
        """Return the top elite_fraction individuals."""
        evaluated = [g for g in self.individuals if g.fitness is not None]
        if not evaluated:
            return []
        n_elite = max(1, int(self.config.elite_fraction * len(self.individuals)))
        sorted_pop = sorted(evaluated,
                            key=lambda g: g.fitness,  # type: ignore
                            reverse=True)
        return [g.clone() for g in sorted_pop[:n_elite]]

    # ------------------------------------------------------------------
    # Replacement
    # ------------------------------------------------------------------

    def replace(self, offspring: List[StrategyGenome],
                preserve_elite: bool = True) -> None:
        """
        Replace the current population with offspring.
        If preserve_elite, keep the top elite_fraction from the old population.
        """
        new_population: List[StrategyGenome] = []

        if preserve_elite:
            elite = self.select_elite()
            new_population.extend(elite)

        # Update hall of fame with current population
        self.hall_of_fame.update(self.individuals)

        # Fill remaining slots with offspring
        remaining = self.config.size - len(new_population)
        if len(offspring) >= remaining:
            new_population.extend(offspring[:remaining])
        else:
            new_population.extend(offspring)
            # Fill gap with random individuals if needed
            while len(new_population) < self.config.size:
                new_population.append(self.factory.create_random())

        # Apply fitness sharing if configured
        if self._fitness_sharing is not None:
            # Only apply to evaluated individuals
            evaluated = [g for g in new_population if g.fitness is not None]
            if evaluated:
                self._fitness_sharing.apply(evaluated)

        # Deduplicate if configured
        if self.config.deduplication:
            new_population = self._deduplicate(new_population)

        # Diversity injection
        if (self.generation > 0 and
                self.generation % self.config.diversity_injection_interval == 0):
            n_inject = max(1, int(self.config.diversity_injection_fraction * self.config.size))
            new_population = self._inject_diversity(new_population, n_inject)

        self.individuals = new_population[:self.config.size]
        # Increment age for all individuals
        for g in self.individuals:
            g.metadata.age += 1
        self.generation += 1

    def _deduplicate(self, population: List[StrategyGenome]) -> List[StrategyGenome]:
        """Remove duplicate genomes (by fingerprint) and replace with random ones."""
        seen: Set[str] = set()
        unique: List[StrategyGenome] = []
        duplicates: List[int] = []

        for i, g in enumerate(population):
            fp = g.fingerprint()
            if fp not in seen:
                seen.add(fp)
                unique.append(g)
            else:
                duplicates.append(i)

        # If too many duplicates, check if we need to inject
        dup_fraction = len(duplicates) / max(len(population), 1)
        if dup_fraction > self.config.max_duplicate_fraction:
            n_replace = len(duplicates)
            for _ in range(n_replace):
                unique.append(self.factory.create_random())

        return unique

    def _inject_diversity(self, population: List[StrategyGenome],
                          n_inject: int) -> List[StrategyGenome]:
        """Replace the weakest n_inject individuals with random ones."""
        evaluated = [(i, g) for i, g in enumerate(population) if g.fitness is not None]
        if not evaluated:
            return population

        # Sort by fitness ascending (worst first)
        evaluated.sort(key=lambda x: x[1].fitness)  # type: ignore
        indices_to_replace = [idx for idx, _ in evaluated[:n_inject]]

        result = list(population)
        for idx in indices_to_replace:
            result[idx] = self.factory.create_random()
        return result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_stats(self) -> PopulationStats:
        stats = DiversityMetrics.compute_stats(self.individuals, self.generation)
        self.stats_history.append(stats)
        return stats

    def best_individual(self) -> Optional[StrategyGenome]:
        evaluated = [g for g in self.individuals if g.fitness is not None]
        if not evaluated:
            return None
        return max(evaluated, key=lambda g: g.fitness)  # type: ignore

    def worst_individual(self) -> Optional[StrategyGenome]:
        evaluated = [g for g in self.individuals if g.fitness is not None]
        if not evaluated:
            return None
        return min(evaluated, key=lambda g: g.fitness)  # type: ignore

    # ------------------------------------------------------------------
    # Crowding distance (NSGA-II)
    # ------------------------------------------------------------------

    def assign_crowding_distances(self, front: List[StrategyGenome]) -> None:
        """
        Assign crowding distances to individuals in a Pareto front.
        Used by NSGA-II for density estimation.
        """
        n = len(front)
        if n == 0:
            return
        for g in front:
            g.crowding_distance = 0.0

        if front[0].objectives is None:
            return

        n_objectives = len(front[0].objectives)
        for obj_idx in range(n_objectives):
            sorted_front = sorted(front,
                                  key=lambda g: g.objectives[obj_idx]  # type: ignore
                                  if g.objectives else 0.0)
            # Boundary individuals get infinite distance
            sorted_front[0].crowding_distance = float("inf")
            sorted_front[-1].crowding_distance = float("inf")

            obj_range = (sorted_front[-1].objectives[obj_idx] -  # type: ignore
                         sorted_front[0].objectives[obj_idx])  # type: ignore
            if obj_range == 0:
                continue

            for i in range(1, n - 1):
                prev_obj = sorted_front[i - 1].objectives[obj_idx]  # type: ignore
                next_obj = sorted_front[i + 1].objectives[obj_idx]  # type: ignore
                sorted_front[i].crowding_distance += (next_obj - prev_obj) / obj_range

    # ------------------------------------------------------------------
    # Pareto front computation (NSGA-II non-dominated sorting)
    # ------------------------------------------------------------------

    def non_dominated_sort(self,
                           genomes: Optional[List[StrategyGenome]] = None
                           ) -> List[List[StrategyGenome]]:
        """
        Compute Pareto fronts via non-dominated sorting.
        Returns list of fronts, front[0] = Pareto-optimal front.
        """
        pop = genomes or self.individuals
        n = len(pop)
        dominates: Dict[int, List[int]] = defaultdict(list)
        dominated_count: Dict[int, int] = defaultdict(int)
        fronts: List[List[StrategyGenome]] = [[]]
        front_indices: List[List[int]] = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if pop[i].dominates(pop[j]):
                    dominates[i].append(j)
                elif pop[j].dominates(pop[i]):
                    dominated_count[i] += 1

            if dominated_count[i] == 0:
                pop[i].rank = 1
                front_indices[0].append(i)

        fronts[0] = [pop[i] for i in front_indices[0]]
        current_front = 0

        while front_indices[current_front]:
            next_front_indices: List[int] = []
            for i in front_indices[current_front]:
                for j in dominates[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        pop[j].rank = current_front + 2
                        next_front_indices.append(j)
            current_front += 1
            front_indices.append(next_front_indices)
            fronts.append([pop[i] for i in next_front_indices])

        # Assign crowding distances per front
        for front in fronts:
            if front:
                self.assign_crowding_distances(front)

        return [f for f in fronts if f]

    # ------------------------------------------------------------------
    # NSGA-II selection for multi-objective optimization
    # ------------------------------------------------------------------

    def nsga2_select(self, combined: List[StrategyGenome],
                     n_select: int) -> List[StrategyGenome]:
        """
        NSGA-II selection: fill new population with Pareto fronts,
        using crowding distance to break ties in the last front.
        """
        fronts = self.non_dominated_sort(combined)
        selected: List[StrategyGenome] = []

        for front in fronts:
            if len(selected) + len(front) <= n_select:
                selected.extend(front)
            else:
                # Sort by crowding distance descending (prefer less crowded)
                remaining = n_select - len(selected)
                front_sorted = sorted(front,
                                      key=lambda g: g.crowding_distance,
                                      reverse=True)
                selected.extend(front_sorted[:remaining])
                break

        return selected

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_list(self) -> List[Dict[str, Any]]:
        return [self.factory.serialize_genome(g) for g in self.individuals]

    def from_list(self, data: List[Dict[str, Any]]) -> None:
        self.individuals = [self.factory.deserialize_genome(d) for d in data]

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    def __getitem__(self, idx: int) -> StrategyGenome:
        return self.individuals[idx]

    def __repr__(self) -> str:
        best = self.best_individual()
        return (f"Population(size={len(self)}, gen={self.generation}, "
                f"best={best.fitness:.4f if best and best.fitness else 'N/A'})")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Population self-test ===")
    from .genome import GenomeFactory

    factory = GenomeFactory("momentum", seed=42)

    cfg = PopulationConfig(
        size=50,
        elite_fraction=0.10,
        tournament_size=5,
        selection_method="tournament",
        fitness_sharing=True,
        sharing_sigma=0.2,
        deduplication=True,
        seed=42,
    )
    pop = Population(cfg, factory)
    pop.initialize()
    print(f"Initialized: {pop}")

    # Assign random fitnesses for testing
    for g in pop.individuals:
        g.fitness = random.gauss(0.5, 0.2)

    stats = pop.compute_stats()
    print(f"Stats: {stats}")

    # Test selection
    p1, p2 = pop.select_parents()
    print(f"\nSelected parents: {p1.fitness:.4f}, {p2.fitness:.4f}")

    # Test elite selection
    elite = pop.select_elite()
    print(f"Elite count: {len(elite)}, best elite fitness: {max(g.fitness for g in elite):.4f}")

    # Test hall of fame
    pop.hall_of_fame.update(pop.individuals)
    print(f"Hall of fame size: {len(pop.hall_of_fame)}")
    print(f"HOF best: {pop.hall_of_fame.best.fitness:.4f}")

    # Test replacement
    offspring = [factory.create_random() for _ in range(40)]
    for g in offspring:
        g.fitness = random.gauss(0.6, 0.2)  # Slightly better offspring
    pop.replace(offspring, preserve_elite=True)
    print(f"\nAfter replacement: {pop}")

    # Test multi-objective
    for g in pop.individuals:
        g.objectives = [random.gauss(0.5, 0.2), random.gauss(0.3, 0.1)]

    fronts = pop.non_dominated_sort()
    print(f"\nPareto fronts: {len(fronts)}, front 0 size: {len(fronts[0])}")

    # Test diversity metrics
    diversity = DiversityMetrics.mean_pairwise_distance(pop.individuals)
    entropy = DiversityMetrics.entropy(pop.individuals)
    print(f"Diversity: {diversity:.4f}, Entropy: {entropy:.4f}")

    # Test selection methods
    for method in ["roulette", "rank", "sus"]:
        cfg2 = PopulationConfig(size=20, selection_method=method, seed=42)
        pop2 = Population(cfg2, factory)
        pop2.initialize()
        for g in pop2.individuals:
            g.fitness = random.gauss(0.5, 0.2)
        p1, p2 = pop2.select_parents()
        print(f"  {method} selection: {p1.fitness:.4f}, {p2.fitness:.4f}")

    print("\nAll population tests passed.")
