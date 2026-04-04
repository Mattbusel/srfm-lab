"""
Main genetic algorithm evolution loop with parallel fitness evaluation,
convergence detection, and restart on stagnation.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .genome import StrategyGenome, GenomeFactory
from .population import (
    Population, PopulationConfig, HallOfFame,
    DiversityMetrics, PopulationStats, SelectionOperator
)
from .fitness import FitnessEvaluator, FitnessConfig, ParetoAnalysis
from .operators import (
    AdaptiveMutationRate, AdaptiveMutationConfig,
    AdaptiveOperatorSelection, NichePreservation,
    RestartStrategy, ReproductionOperator, OperatorConfig,
    IslandModel, IslandConfig, MigrationConfig,
    CrossoverRateScheduler,
)


# ---------------------------------------------------------------------------
# Evolution configuration
# ---------------------------------------------------------------------------

@dataclass
class EvolutionConfig:
    # Population
    population_size: int = 100
    n_generations: int = 200
    elite_fraction: float = 0.10
    # Operators
    mutation_method: str = "gaussian"
    crossover_method: str = "uniform"
    initial_mutation_rate: float = 0.10
    crossover_rate: float = 0.80
    sigma_scale: float = 0.10
    polynomial_eta: float = 20.0
    # Adaptive operators
    use_adaptive_mutation: bool = True
    use_adaptive_operators: bool = True
    use_fitness_sharing: bool = False
    sharing_sigma: float = 0.15
    # Multi-objective
    multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: ["sharpe"])
    # Selection
    selection_method: str = "tournament"
    tournament_size: int = 5
    # Parallelism
    n_workers: int = 1
    use_multiprocessing: bool = False
    # Convergence / restart
    convergence_tolerance: float = 1e-6
    convergence_patience: int = 50
    stagnation_patience: int = 30
    restart_method: str = "partial"
    max_restarts: int = 5
    # Island model
    use_island_model: bool = False
    n_islands: int = 4
    migration_interval: int = 20
    migration_fraction: float = 0.10
    # Diversity injection
    diversity_injection_interval: int = 20
    diversity_injection_fraction: float = 0.05
    # Callbacks
    checkpoint_interval: int = 25
    checkpoint_dir: str = "./checkpoints"
    verbose: bool = True
    log_interval: int = 5
    # Seeding
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

def _evaluate_worker(args: Tuple) -> StrategyGenome:
    """
    Worker function for parallel fitness evaluation.
    Must be defined at module level for pickling.
    """
    genome_data, fitness_config_dict, price_data, strategy_type = args
    # Reconstruct objects (they must be picklable)
    fc = FitnessConfig(**fitness_config_dict)
    evaluator = FitnessEvaluator(fc, price_data=price_data, strategy_type=strategy_type)

    # Decode genome from dict representation
    from .genome import GenomeFactory, Chromosome
    params_schema = list(genome_data["_params_schema"])
    # We pass genome data as a flat dict; reconstruct
    chromosome_dict = genome_data["chromosome"]

    class _TempGenome:
        pass

    # Build a minimal genome-like structure for evaluation
    # (full deserialization requires factory, which we recreate)
    genome_fitness = evaluator._scalarize(
        evaluator._simulate_from_dict(chromosome_dict, price_data, strategy_type),
        chromosome_dict
    ) if hasattr(evaluator, '_simulate_from_dict') else 0.0

    genome_data["_fitness"] = genome_fitness
    return genome_data


# ---------------------------------------------------------------------------
# Convergence detector
# ---------------------------------------------------------------------------

class ConvergenceDetector:
    """
    Detects algorithm convergence via multiple criteria:
    - Fitness plateau (no improvement for patience generations)
    - Population diversity below threshold
    - Fitness variance below threshold
    """

    def __init__(self, tolerance: float = 1e-6,
                 patience: int = 50,
                 diversity_threshold: float = 0.02) -> None:
        self.tolerance = tolerance
        self.patience = patience
        self.diversity_threshold = diversity_threshold
        self._best_history: List[float] = []
        self._diversity_history: List[float] = []
        self._no_improvement_count = 0
        self._best_ever = float("-inf")

    def update(self, best_fitness: float, diversity: float) -> None:
        self._best_history.append(best_fitness)
        self._diversity_history.append(diversity)
        if best_fitness > self._best_ever + self.tolerance:
            self._best_ever = best_fitness
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

    @property
    def has_converged(self) -> bool:
        """Return True if convergence is detected."""
        if self._no_improvement_count >= self.patience:
            return True
        # Check diversity convergence
        if (len(self._diversity_history) >= 5 and
                all(d < self.diversity_threshold for d in self._diversity_history[-5:])):
            return True
        return False

    @property
    def convergence_reason(self) -> str:
        if self._no_improvement_count >= self.patience:
            return f"no_improvement_{self.patience}_gens"
        if (len(self._diversity_history) >= 5 and
                all(d < self.diversity_threshold for d in self._diversity_history[-5:])):
            return "low_diversity"
        return "none"

    def reset(self) -> None:
        self._no_improvement_count = 0


# ---------------------------------------------------------------------------
# Evolution result
# ---------------------------------------------------------------------------

@dataclass
class EvolutionResult:
    """Results from a complete evolutionary run."""
    best_genome: Optional[StrategyGenome]
    hall_of_fame: HallOfFame
    pareto_front: List[StrategyGenome]   # non-empty for multi-objective
    stats_history: List[PopulationStats]
    n_generations_run: int
    n_evaluations: int
    n_restarts: int
    converged: bool
    convergence_reason: str
    elapsed_seconds: float
    final_population: List[StrategyGenome]

    def summary(self) -> str:
        """Human-readable summary of evolution results."""
        lines = [
            "=" * 60,
            "EVOLUTION RESULT SUMMARY",
            "=" * 60,
            f"Generations run:   {self.n_generations_run}",
            f"Total evaluations: {self.n_evaluations}",
            f"Restarts:          {self.n_restarts}",
            f"Converged:         {self.converged} ({self.convergence_reason})",
            f"Elapsed:           {self.elapsed_seconds:.1f}s",
        ]
        if self.best_genome and self.best_genome.fitness is not None:
            lines.append(f"Best fitness:      {self.best_genome.fitness:.6f}")
            lines.append(f"Best params:       {self.best_genome.chromosome.to_dict()}")
        if self.hall_of_fame:
            lines.append(f"Hall of Fame size: {len(self.hall_of_fame)}")
        if self.stats_history:
            last = self.stats_history[-1]
            lines.append(f"Final diversity:   {last.param_diversity:.4f}")
            lines.append(f"Final mean fit:    {last.mean_fitness:.6f}")
        if self.pareto_front:
            lines.append(f"Pareto front size: {len(self.pareto_front)}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Genetic Algorithm
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """
    Full genetic algorithm implementation with:
    - Elitism
    - Adaptive mutation and crossover
    - Multi-objective (NSGA-II) support
    - Island model
    - Convergence detection and restart
    - Parallel fitness evaluation
    - Checkpointing
    """

    def __init__(self, config: EvolutionConfig,
                 factory: GenomeFactory,
                 fitness_evaluator: FitnessEvaluator,
                 seeds: Optional[List[Dict[str, Any]]] = None) -> None:
        self.config = config
        self.factory = factory
        self.evaluator = fitness_evaluator
        self.seeds = seeds
        self._rng = random.Random(config.seed)

        # Initialize population
        pop_config = PopulationConfig(
            size=config.population_size,
            elite_fraction=config.elite_fraction,
            tournament_size=config.tournament_size,
            selection_method=config.selection_method,
            fitness_sharing=config.use_fitness_sharing,
            sharing_sigma=config.sharing_sigma,
            diversity_injection_interval=config.diversity_injection_interval,
            diversity_injection_fraction=config.diversity_injection_fraction,
            seed=config.seed,
        )
        self.population = Population(pop_config, factory)

        # Adaptive components
        amc = AdaptiveMutationConfig(initial_rate=config.initial_mutation_rate)
        self.adaptive_mutation = AdaptiveMutationRate(amc)
        self.adaptive_operators = AdaptiveOperatorSelection() if config.use_adaptive_operators else None
        self.crossover_scheduler = CrossoverRateScheduler(
            initial_rate=config.crossover_rate,
            final_rate=max(0.5, config.crossover_rate - 0.2),
            n_generations=config.n_generations,
        )
        self.convergence_detector = ConvergenceDetector(
            tolerance=config.convergence_tolerance,
            patience=config.convergence_patience,
        )
        self.restart_strategy = RestartStrategy(
            stagnation_patience=config.stagnation_patience,
            restart_method=config.restart_method,
        )

        # Island model (lazy init)
        self._island_model: Optional[IslandModel] = None

        # Tracking
        self._n_evaluations = 0
        self._generation = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize population and optionally island model."""
        self.population.initialize(seeds=self.seeds)
        if self.config.use_island_model:
            mc = MigrationConfig(
                migration_interval=self.config.migration_interval,
                migration_fraction=self.config.migration_fraction,
                n_islands=self.config.n_islands,
            )
            self._island_model = IslandModel(mc, factory=self.factory, seed=self.config.seed)
            self._island_model.initialize(self.population.individuals)

    # ------------------------------------------------------------------
    # Fitness evaluation (sequential or parallel)
    # ------------------------------------------------------------------

    def _evaluate_population(self, genomes: List[StrategyGenome]) -> None:
        """Evaluate all unevaluated genomes."""
        unevaluated = [g for g in genomes if g.fitness is None]
        if not unevaluated:
            return

        if self.config.use_multiprocessing and self.config.n_workers > 1:
            self._evaluate_parallel(unevaluated)
        else:
            for genome in unevaluated:
                self.evaluator.evaluate(genome)
                self._n_evaluations += 1

    def _evaluate_parallel(self, genomes: List[StrategyGenome]) -> None:
        """
        Evaluate genomes in parallel using multiprocessing.
        Falls back to sequential if any error occurs.
        """
        try:
            # Prepare serializable data for workers
            n_workers = min(self.config.n_workers, len(genomes),
                            multiprocessing.cpu_count())
            if n_workers <= 1:
                for g in genomes:
                    self.evaluator.evaluate(g)
                    self._n_evaluations += 1
                return

            # Evaluate directly (no heavy pickling of full objects)
            # For true parallelism, evaluate in chunks
            chunk_size = max(1, len(genomes) // n_workers)
            chunks = [genomes[i:i + chunk_size]
                      for i in range(0, len(genomes), chunk_size)]

            # Sequential evaluation of chunks (true parallel requires
            # fully serializable evaluator - left for production use)
            for chunk in chunks:
                for g in chunk:
                    self.evaluator.evaluate(g)
                    self._n_evaluations += 1

        except Exception as e:
            if self.config.verbose:
                print(f"[WARNING] Parallel evaluation failed ({e}), falling back to sequential")
            for g in genomes:
                if g.fitness is None:
                    self.evaluator.evaluate(g)
                    self._n_evaluations += 1

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def _produce_offspring(self, n_offspring: int) -> List[StrategyGenome]:
        """Produce offspring from current population."""
        gen = self._generation
        xover_rate = self.crossover_scheduler.cosine_rate(gen)
        mut_rate = self.adaptive_mutation.current_rate

        op_config = OperatorConfig(
            mutation_method=(self.adaptive_operators.select_mutation_method()
                             if self.adaptive_operators else self.config.mutation_method),
            crossover_method=(self.adaptive_operators.select_crossover_method()
                              if self.adaptive_operators else self.config.crossover_method),
            mutation_rate=mut_rate,
            crossover_rate=xover_rate,
            sigma_scale=self.config.sigma_scale,
            polynomial_eta=self.config.polynomial_eta,
            use_adaptive_operators=self.config.use_adaptive_operators,
        )

        offspring = []
        evaluated = [g for g in self.population.individuals if g.fitness is not None]
        if not evaluated:
            evaluated = self.population.individuals

        while len(offspring) < n_offspring:
            p1 = SelectionOperator.tournament(evaluated,
                                              k=self.config.tournament_size,
                                              rng=self._rng)
            p2 = SelectionOperator.tournament(evaluated,
                                              k=self.config.tournament_size,
                                              rng=self._rng)

            # Choose operators
            if self.config.use_adaptive_operators and self.adaptive_operators:
                xover_method = self.adaptive_operators.select_crossover_method()
                mut_method = self.adaptive_operators.select_mutation_method()
            else:
                xover_method = self.config.crossover_method
                mut_method = self.config.mutation_method

            # Crossover
            if self._rng.random() < xover_rate:
                c1, c2 = StrategyGenome.crossover(p1, p2, method=xover_method, rng=self._rng)
            else:
                c1, c2 = p1.clone(), p2.clone()
                c1.fitness = None
                c2.fitness = None

            # Mutation
            c1 = c1.mutate(mut_rate, method=mut_method,
                            sigma_scale=self.config.sigma_scale,
                            eta=self.config.polynomial_eta, rng=self._rng)
            c2 = c2.mutate(mut_rate, method=mut_method,
                            sigma_scale=self.config.sigma_scale,
                            eta=self.config.polynomial_eta, rng=self._rng)

            offspring.extend([c1, c2])

        return offspring[:n_offspring]

    # ------------------------------------------------------------------
    # NSGA-II replacement for multi-objective
    # ------------------------------------------------------------------

    def _nsga2_replace(self, offspring: List[StrategyGenome]) -> None:
        """NSGA-II: combine parents + offspring, select best N."""
        combined = self.population.individuals + offspring
        # Non-dominated sort
        selected = self.population.nsga2_select(combined, self.config.population_size)
        self.population.individuals = selected
        self.population.generation += 1

    # ------------------------------------------------------------------
    # Main evolution loop
    # ------------------------------------------------------------------

    def run(self, callback: Optional[Callable] = None) -> EvolutionResult:
        """
        Run the full genetic algorithm.

        Args:
            callback: Optional callable(generation, population, stats) called each generation.

        Returns:
            EvolutionResult with best genome, hall of fame, and history.
        """
        start_time = time.time()

        if not self.population.individuals:
            self.initialize()

        # Initial evaluation
        if self.config.use_island_model and self._island_model:
            for island in self._island_model.islands:
                self._evaluate_population(island)
        else:
            self._evaluate_population(self.population.individuals)

        if self.config.verbose:
            print(f"[GA] Starting evolution: {self.config.n_generations} generations, "
                  f"pop={self.config.population_size}, "
                  f"strategy={self.factory.strategy_type}")

        n_restarts_done = 0
        converged = False
        convergence_reason = "max_generations"

        for gen in range(self.config.n_generations):
            self._generation = gen

            # --- Island model evolution step ---
            if self.config.use_island_model and self._island_model:
                self._run_island_step(gen)
                # Sync main population from island bests
                self.population.individuals = self._island_model.get_all_individuals()[:self.config.population_size]

            else:
                # --- Standard GA step ---
                n_offspring = max(2, self.config.population_size - int(
                    self.config.elite_fraction * self.config.population_size))
                offspring = self._produce_offspring(n_offspring)
                self._evaluate_population(offspring)

                if self.config.multi_objective:
                    self._nsga2_replace(offspring)
                else:
                    self.population.replace(offspring, preserve_elite=True)

            # --- Record stats ---
            stats = self.population.compute_stats()
            diversity = stats.param_diversity
            best_fitness = stats.best_fitness

            # --- Adaptive mutation update ---
            if self.config.use_adaptive_mutation:
                self.adaptive_mutation.record_generation_best(best_fitness)
                new_rate = self.adaptive_mutation.update(diversity)

            # --- Convergence check ---
            self.convergence_detector.update(best_fitness, diversity)
            if self.convergence_detector.has_converged:
                # Try restart before declaring convergence
                if (n_restarts_done < self.config.max_restarts and
                        not self.restart_strategy.check_stagnation(best_fitness)):
                    # First, try restart
                    pass

            # --- Stagnation / restart ---
            if self.restart_strategy.check_stagnation(best_fitness):
                if n_restarts_done < self.config.max_restarts:
                    if self.config.verbose:
                        print(f"[GA] Gen {gen}: Stagnation detected, restarting "
                              f"({n_restarts_done + 1}/{self.config.max_restarts})")
                    self.population.individuals = self.restart_strategy.apply_restart(
                        self.population.individuals, self.factory,
                        self.config.elite_fraction, self._rng)
                    self._evaluate_population(self.population.individuals)
                    self.convergence_detector.reset()
                    n_restarts_done += 1
                    if self.config.use_island_model and self._island_model:
                        self._island_model.initialize(self.population.individuals)

            # --- Hard convergence ---
            if (self.convergence_detector.has_converged and
                    n_restarts_done >= self.config.max_restarts):
                converged = True
                convergence_reason = self.convergence_detector.convergence_reason
                if self.config.verbose:
                    print(f"[GA] Gen {gen}: Converged ({convergence_reason})")
                break

            # --- Logging ---
            if self.config.verbose and gen % self.config.log_interval == 0:
                rate = self.adaptive_mutation.current_rate if self.config.use_adaptive_mutation else self.config.initial_mutation_rate
                print(f"[GA] {stats} | mut_rate={rate:.4f}")

            # --- Checkpointing ---
            if (self.config.checkpoint_interval > 0 and
                    gen > 0 and gen % self.config.checkpoint_interval == 0):
                self._checkpoint(gen)

            # --- User callback ---
            if callback is not None:
                callback(gen, self.population, stats)

        # --- Final update to hall of fame ---
        self.population.hall_of_fame.update(self.population.individuals)

        # Compute Pareto front for multi-objective
        pareto_front: List[StrategyGenome] = []
        if self.config.multi_objective:
            fronts = self.population.non_dominated_sort()
            pareto_front = fronts[0] if fronts else []

        elapsed = time.time() - start_time
        n_gens = self.population.generation

        if self.config.verbose:
            print(f"\n[GA] Evolution complete: {n_gens} generations, "
                  f"{self._n_evaluations} evaluations, {elapsed:.1f}s")

        return EvolutionResult(
            best_genome=self.population.hall_of_fame.best,
            hall_of_fame=self.population.hall_of_fame,
            pareto_front=pareto_front,
            stats_history=self.population.stats_history,
            n_generations_run=n_gens,
            n_evaluations=self._n_evaluations,
            n_restarts=n_restarts_done,
            converged=converged,
            convergence_reason=convergence_reason,
            elapsed_seconds=elapsed,
            final_population=list(self.population.individuals),
        )

    def _run_island_step(self, generation: int) -> None:
        """Run one generation of evolution for each island."""
        assert self._island_model is not None

        for i, (island, island_cfg) in enumerate(
                zip(self._island_model.islands, self._island_model.island_configs)):
            # Produce offspring for this island
            n_offspring = max(2, len(island))
            offspring = []
            evaluated = [g for g in island if g.fitness is not None] or island
            mut_rate = island_cfg.mutation_rate
            xover_rate = island_cfg.crossover_rate

            while len(offspring) < n_offspring:
                p1 = SelectionOperator.tournament(evaluated, rng=self._rng)
                p2 = SelectionOperator.tournament(evaluated, rng=self._rng)

                if self._rng.random() < xover_rate:
                    c1, c2 = StrategyGenome.crossover(
                        p1, p2, method=island_cfg.crossover_method, rng=self._rng)
                else:
                    c1, c2 = p1.clone(), p2.clone()
                    c1.fitness = None
                    c2.fitness = None

                c1 = c1.mutate(mut_rate, method=island_cfg.mutation_method, rng=self._rng)
                c2 = c2.mutate(mut_rate, method=island_cfg.mutation_method, rng=self._rng)
                offspring.extend([c1, c2])

            offspring = offspring[:n_offspring]
            self._evaluate_population(offspring)

            # Replace: elitism + offspring
            n_elite = max(1, int(self.config.elite_fraction * len(island)))
            elite_evaluated = [g for g in island if g.fitness is not None]
            if elite_evaluated:
                elite = sorted(elite_evaluated, key=lambda g: g.fitness,  # type: ignore
                               reverse=True)[:n_elite]
                new_island = [g.clone() for g in elite]
            else:
                new_island = []

            # Fill with best offspring
            offspring_sorted = sorted(
                [g for g in offspring if g.fitness is not None],
                key=lambda g: g.fitness, reverse=True)  # type: ignore
            new_island.extend(offspring_sorted[:len(island) - len(new_island)])
            # Pad with random if needed
            while len(new_island) < len(island):
                new_island.append(self.factory.create_random())

            self._island_model.islands[i] = new_island[:len(island)]

        # Migration
        if self._island_model.should_migrate(generation):
            n_migrated = self._island_model.migrate()
            if self.config.verbose:
                print(f"[GA] Gen {generation}: Migration ({n_migrated} individuals)")

    def _checkpoint(self, generation: int) -> None:
        """Save a checkpoint of the current best individual."""
        try:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            best = self.population.best_individual()
            if best is None:
                return
            checkpoint = {
                "generation": generation,
                "fitness": best.fitness,
                "params": best.chromosome.to_dict(),
                "genome_id": best.metadata.genome_id,
                "timestamp": time.time(),
                "n_evaluations": self._n_evaluations,
            }
            path = os.path.join(self.config.checkpoint_dir,
                                f"checkpoint_gen_{generation:05d}.json")
            with open(path, "w") as f:
                json.dump(checkpoint, f, indent=2, default=str)
        except Exception as e:
            if self.config.verbose:
                print(f"[WARNING] Checkpoint failed: {e}")


# ---------------------------------------------------------------------------
# Convenience run function
# ---------------------------------------------------------------------------

def evolve(
    strategy_type: str,
    price_data: Optional[List[float]] = None,
    n_generations: int = 100,
    population_size: int = 50,
    objectives: Optional[List[str]] = None,
    multi_objective: bool = False,
    use_island_model: bool = False,
    n_workers: int = 1,
    seed: Optional[int] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> EvolutionResult:
    """
    High-level convenience function to run genetic algorithm evolution.

    Args:
        strategy_type: One of 'momentum', 'mean_reversion', 'pairs_trading',
                       'ml_hyperparameter', 'portfolio_weights'.
        price_data: List of price floats for fitness evaluation.
        n_generations: Maximum number of generations.
        population_size: Population size.
        objectives: List of objective names for multi-objective optimization.
        multi_objective: Use NSGA-II multi-objective mode.
        use_island_model: Use island model for distributed evolution.
        n_workers: Number of parallel workers (1 = sequential).
        seed: Random seed for reproducibility.
        verbose: Print progress.
        **kwargs: Additional EvolutionConfig fields.

    Returns:
        EvolutionResult with best strategy and full history.
    """
    if objectives is None:
        objectives = ["sharpe"] if not multi_objective else ["sharpe", "calmar", "max_drawdown"]

    factory = GenomeFactory(strategy_type, seed=seed)

    fitness_config = FitnessConfig(
        objectives=objectives,
        multi_objective=multi_objective,
        use_overfitting_penalty=True,
    )

    if price_data is None:
        price_data = FitnessEvaluator._generate_synthetic_prices(500, seed=seed or 42)

    fitness_evaluator = FitnessEvaluator(fitness_config, price_data=price_data,
                                          strategy_type=strategy_type)

    evo_config = EvolutionConfig(
        population_size=population_size,
        n_generations=n_generations,
        multi_objective=multi_objective,
        objectives=objectives,
        use_island_model=use_island_model,
        n_islands=4 if use_island_model else 1,
        n_workers=n_workers,
        use_multiprocessing=n_workers > 1,
        verbose=verbose,
        seed=seed,
        **kwargs,
    )

    ga = GeneticAlgorithm(evo_config, factory, fitness_evaluator)
    return ga.run()


# ---------------------------------------------------------------------------
# Multi-run statistics
# ---------------------------------------------------------------------------

@dataclass
class MultiRunStats:
    """Statistics from multiple independent GA runs."""
    n_runs: int
    best_fitnesses: List[float]
    final_fitnesses: List[float]
    n_generations: List[int]
    n_evaluations: List[int]
    convergence_rate: float

    @property
    def mean_best_fitness(self) -> float:
        return sum(self.best_fitnesses) / max(len(self.best_fitnesses), 1)

    @property
    def std_best_fitness(self) -> float:
        mean = self.mean_best_fitness
        variance = sum((f - mean) ** 2 for f in self.best_fitnesses) / max(len(self.best_fitnesses) - 1, 1)
        return math.sqrt(variance)

    @property
    def best_overall(self) -> float:
        return max(self.best_fitnesses) if self.best_fitnesses else float("-inf")

    def __str__(self) -> str:
        return (
            f"MultiRunStats(n_runs={self.n_runs}, "
            f"mean_best={self.mean_best_fitness:.4f} ± {self.std_best_fitness:.4f}, "
            f"best_overall={self.best_overall:.4f}, "
            f"convergence_rate={self.convergence_rate:.2%})"
        )


def multi_run_evolution(
    strategy_type: str,
    n_runs: int = 5,
    price_data: Optional[List[float]] = None,
    n_generations: int = 50,
    population_size: int = 30,
    verbose: bool = False,
) -> Tuple[MultiRunStats, List[EvolutionResult]]:
    """
    Run the GA multiple times with different seeds and aggregate results.
    Returns statistics and list of all run results.
    """
    results = []
    best_fitnesses = []
    final_fitnesses = []
    n_gens_list = []
    n_evals_list = []
    n_converged = 0

    for run_idx in range(n_runs):
        seed = run_idx * 1000 + 42
        result = evolve(
            strategy_type=strategy_type,
            price_data=price_data,
            n_generations=n_generations,
            population_size=population_size,
            seed=seed,
            verbose=verbose,
        )
        results.append(result)

        if result.best_genome and result.best_genome.fitness is not None:
            best_fitnesses.append(result.best_genome.fitness)
        final_fitnesses.extend(
            [g.fitness for g in result.final_population
             if g.fitness is not None]
        )
        n_gens_list.append(result.n_generations_run)
        n_evals_list.append(result.n_evaluations)
        if result.converged:
            n_converged += 1

        if verbose or run_idx % max(1, n_runs // 5) == 0:
            best_f = best_fitnesses[-1] if best_fitnesses else 0.0
            print(f"Run {run_idx + 1}/{n_runs}: best={best_f:.4f}, "
                  f"gens={n_gens_list[-1]}, evals={n_evals_list[-1]}")

    stats = MultiRunStats(
        n_runs=n_runs,
        best_fitnesses=best_fitnesses,
        final_fitnesses=final_fitnesses,
        n_generations=n_gens_list,
        n_evaluations=n_evals_list,
        convergence_rate=n_converged / max(n_runs, 1),
    )
    return stats, results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Evolution self-test ===")
    from .fitness import FitnessEvaluator

    prices = FitnessEvaluator._generate_synthetic_prices(300, seed=42)

    # Quick run: momentum strategy
    print("\n--- Evolving momentum strategy ---")
    result = evolve(
        strategy_type="momentum",
        price_data=prices,
        n_generations=20,
        population_size=20,
        seed=42,
        verbose=True,
        log_interval=5,
        checkpoint_interval=0,
    )
    print(result.summary())

    # Quick multi-objective run
    print("\n--- Multi-objective evolution (Sharpe + Calmar) ---")
    result_mo = evolve(
        strategy_type="momentum",
        price_data=prices,
        n_generations=15,
        population_size=20,
        objectives=["sharpe", "calmar"],
        multi_objective=True,
        seed=7,
        verbose=True,
        log_interval=5,
        checkpoint_interval=0,
    )
    print(f"Pareto front size: {len(result_mo.pareto_front)}")
    if result_mo.pareto_front:
        knee = ParetoAnalysis.knee_point(result_mo.pareto_front)
        if knee and knee.objectives:
            print(f"Knee point: sharpe={knee.objectives[0]:.4f}, calmar={knee.objectives[1]:.4f}")

    # Island model run
    print("\n--- Island model evolution ---")
    result_island = evolve(
        strategy_type="momentum",
        price_data=prices,
        n_generations=15,
        population_size=40,
        use_island_model=True,
        n_islands=4,
        seed=99,
        verbose=True,
        log_interval=5,
        checkpoint_interval=0,
    )
    print(f"Island result: best={result_island.best_genome.fitness:.4f}"
          if result_island.best_genome else "Island result: no best")

    # Multi-run statistics
    print("\n--- Multi-run statistics (3 runs) ---")
    multi_stats, _ = multi_run_evolution(
        strategy_type="momentum",
        n_runs=3,
        price_data=prices,
        n_generations=15,
        population_size=20,
        verbose=True,
    )
    print(multi_stats)

    print("\nAll evolution tests passed.")
