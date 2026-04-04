"""
Command-line interface for the genetic algorithm strategy optimizer.

Supports:
- Evolving buy-and-hold (BH) parameters
- Evolving ML hyperparameters
- Evolving portfolio weights
- Running coevolution (maker vs taker)
- Viewing results and hall of fame
- Comparing multiple runs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .genome import GenomeFactory, StrategyGenome, ParamRange, ParamType
from .population import PopulationConfig, Population, HallOfFame
from .fitness import FitnessEvaluator, FitnessConfig, simulate_strategy
from .operators import OperatorConfig
from .evolution import (
    EvolutionConfig, GeneticAlgorithm, evolve,
    multi_run_evolution, EvolutionResult,
)
from .coevolution import (
    CompetitiveCoevolution, CoevolutionConfig,
    CooperativeCoevolution, StrategyEnsembleCoevolution,
)
from .visualization import (
    PlotConfig, EvolutionDashboard, ParetoFrontVisualizer,
    ParameterConvergenceVisualizer, FitnessLandscapeVisualizer,
    GenealogyTracker, plot_multi_run_comparison, ASCIIPlotter,
)


# ---------------------------------------------------------------------------
# Color output
# ---------------------------------------------------------------------------

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    @staticmethod
    def disable() -> None:
        Colors.RESET = Colors.BOLD = Colors.RED = Colors.GREEN = ""
        Colors.YELLOW = Colors.BLUE = Colors.CYAN = Colors.WHITE = ""


def _print_banner() -> None:
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════╗
║     Genetic Algorithm Strategy Optimizer (GASO)      ║
║     Multi-objective trading strategy evolution       ║
╚══════════════════════════════════════════════════════╝
{Colors.RESET}"""
    print(banner)


def _print_section(title: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'=' * 55}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'=' * 55}{Colors.RESET}")


def _print_result(key: str, value: Any) -> None:
    print(f"  {Colors.CYAN}{key:30s}{Colors.RESET}: {Colors.WHITE}{value}{Colors.RESET}")


# ---------------------------------------------------------------------------
# Price data loading
# ---------------------------------------------------------------------------

def load_price_data(path: Optional[str] = None,
                    n_synthetic: int = 500,
                    seed: int = 42,
                    regime: str = "mixed") -> List[float]:
    """
    Load price data from a CSV file or generate synthetic data.
    CSV format: one price per line, or header + date + price columns.
    """
    if path and os.path.isfile(path):
        prices = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Try parsing last column as float
                parts = line.split(",")
                for part in reversed(parts):
                    try:
                        prices.append(float(part.strip()))
                        break
                    except ValueError:
                        continue
        if prices:
            print(f"[Data] Loaded {len(prices)} prices from {path}")
            return prices
        else:
            print(f"[WARNING] Could not parse {path}, using synthetic data")

    # Generate synthetic prices
    print(f"[Data] Generating {n_synthetic} synthetic prices "
          f"(regime={regime}, seed={seed})")
    return _generate_regime_prices(n_synthetic, seed=seed, regime=regime)


def _generate_regime_prices(n: int, seed: int = 42,
                              regime: str = "mixed") -> List[float]:
    """Generate synthetic price data with different regime characteristics."""
    rng = random.Random(seed)
    prices = [100.0]

    if regime == "trending":
        mu, sigma = 0.0008, 0.012
        for i in range(n):
            r = rng.gauss(mu, sigma)
            prices.append(prices[-1] * math.exp(r))

    elif regime == "mean_reverting":
        # Ornstein-Uhlenbeck process
        mean_level = 100.0
        kappa = 0.05
        sigma = 0.015
        for i in range(n):
            current = prices[-1]
            drift = kappa * (mean_level - current) / current
            shock = rng.gauss(0.0, sigma)
            prices.append(current * math.exp(drift + shock))

    elif regime == "volatile":
        # GARCH-like volatility clustering
        vol = 0.015
        for i in range(n):
            vol = 0.9 * vol + 0.1 * abs(rng.gauss(0, 0.020))
            vol = max(0.005, min(0.05, vol))
            r = rng.gauss(0.0, vol)
            prices.append(prices[-1] * math.exp(r))

    elif regime == "crash":
        # Gradual trend + crash
        mu, sigma = 0.0005, 0.010
        crash_start = int(n * 0.7)
        crash_end = int(n * 0.8)
        for i in range(n):
            if crash_start <= i < crash_end:
                r = rng.gauss(-0.020, 0.025)  # crash
            elif i >= crash_end:
                r = rng.gauss(0.001, 0.012)   # recovery
            else:
                r = rng.gauss(mu, sigma)
            prices.append(prices[-1] * math.exp(r))

    else:  # mixed
        # Alternate between regimes
        regime_len = n // 4
        regimes = [
            (0.0006, 0.010),   # trending up
            (0.0, 0.008),      # quiet
            (-0.0003, 0.018),  # weak downtrend
            (0.001, 0.020),    # strong uptrend + vol
        ]
        for i in range(n):
            regime_idx = min(i // regime_len, len(regimes) - 1)
            mu_r, sigma_r = regimes[regime_idx]
            r = rng.gauss(mu_r, sigma_r)
            prices.append(prices[-1] * math.exp(r))

    return prices


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_evolve(args: argparse.Namespace) -> int:
    """Run strategy evolution."""
    _print_section(f"Evolving '{args.strategy}' strategy")

    # Load data
    prices = load_price_data(
        path=getattr(args, "data", None),
        n_synthetic=getattr(args, "n_prices", 500),
        seed=getattr(args, "seed", 42),
        regime=getattr(args, "regime", "mixed"),
    )

    # Configure objectives
    objectives = args.objectives.split(",") if args.objectives else ["sharpe"]
    multi_objective = len(objectives) > 1 or args.multi_objective

    # Run evolution
    print(f"\n[Config] Strategy: {args.strategy}")
    print(f"[Config] Generations: {args.generations}")
    print(f"[Config] Population: {args.population}")
    print(f"[Config] Objectives: {objectives}")
    print(f"[Config] Multi-objective: {multi_objective}")
    print(f"[Config] Island model: {args.islands}")
    print(f"[Config] Seed: {args.seed}")
    print()

    # Load seeds if provided
    seeds = None
    if getattr(args, "seeds_file", None) and os.path.isfile(args.seeds_file):
        with open(args.seeds_file) as f:
            seeds_data = json.load(f)
        seeds = seeds_data if isinstance(seeds_data, list) else [seeds_data]
        print(f"[Config] Seeded with {len(seeds)} known solutions")

    start_time = time.time()
    result = evolve(
        strategy_type=args.strategy,
        price_data=prices,
        n_generations=args.generations,
        population_size=args.population,
        objectives=objectives,
        multi_objective=multi_objective,
        use_island_model=args.islands,
        n_islands=getattr(args, "n_islands", 4),
        seed=args.seed,
        verbose=not args.quiet,
        log_interval=getattr(args, "log_interval", 10),
        checkpoint_interval=getattr(args, "checkpoint_interval", 25),
        checkpoint_dir=getattr(args, "checkpoint_dir", "./checkpoints"),
    )
    elapsed = time.time() - start_time

    # Display results
    _print_section("Evolution Results")
    print(result.summary())

    if result.best_genome:
        _print_section("Best Strategy Parameters")
        for key, value in result.best_genome.chromosome.to_dict().items():
            _print_result(key, value)

    # Save results
    output_path = getattr(args, "output", None)
    if output_path:
        _save_result(result, output_path)
        print(f"\n[Output] Results saved to: {output_path}")

    # Generate plots
    if getattr(args, "plot", False):
        _generate_plots(result, args)

    return 0


def cmd_multi_run(args: argparse.Namespace) -> int:
    """Run multiple independent evolutions and compare results."""
    _print_section(f"Multi-run evolution: {args.n_runs} runs")

    prices = load_price_data(
        path=getattr(args, "data", None),
        n_synthetic=getattr(args, "n_prices", 500),
        regime=getattr(args, "regime", "mixed"),
    )

    print(f"\n[Config] Strategy: {args.strategy}")
    print(f"[Config] Runs: {args.n_runs}")
    print(f"[Config] Generations: {args.generations}")
    print(f"[Config] Population: {args.population}")
    print()

    multi_stats, results = multi_run_evolution(
        strategy_type=args.strategy,
        n_runs=args.n_runs,
        price_data=prices,
        n_generations=args.generations,
        population_size=args.population,
        verbose=not args.quiet,
    )

    _print_section("Multi-Run Statistics")
    print(f"\n{multi_stats}")
    print(f"\n  Best overall fitness: {multi_stats.best_overall:.6f}")
    print(f"  Mean best fitness:    {multi_stats.mean_best_fitness:.6f} ± {multi_stats.std_best_fitness:.6f}")
    print(f"  Convergence rate:     {multi_stats.convergence_rate:.1%}")

    # Find absolute best across all runs
    all_best = [r.best_genome for r in results if r.best_genome]
    if all_best:
        abs_best = max(all_best, key=lambda g: g.fitness or float("-inf"))
        _print_section("Best Genome (Across All Runs)")
        for key, value in abs_best.chromosome.to_dict().items():
            _print_result(key, value)

    # Plot comparison
    if getattr(args, "plot", False):
        config = PlotConfig(output_dir=getattr(args, "output_dir", "./plots"))
        path = plot_multi_run_comparison(results, config)
        if path:
            print(f"\n[Plot] Comparison saved: {path}")

    return 0


def cmd_coevolve(args: argparse.Namespace) -> int:
    """Run competitive coevolution (maker vs taker)."""
    _print_section("Competitive Coevolution: Maker vs Taker")

    prices = load_price_data(
        path=getattr(args, "data", None),
        n_synthetic=getattr(args, "n_prices", 500),
        regime=getattr(args, "regime", "mixed"),
    )

    config = CoevolutionConfig(
        population_size_per_species=getattr(args, "population", 30),
        n_generations=getattr(args, "generations", 50),
        mutation_rate=getattr(args, "mutation_rate", 0.1),
        verbose=not args.quiet,
        log_interval=getattr(args, "log_interval", 5),
        seed=getattr(args, "seed", 42),
    )

    print(f"\n[Config] Population per species: {config.population_size_per_species}")
    print(f"[Config] Generations: {config.n_generations}")
    print(f"[Config] Seed: {config.seed}")
    print()

    coevo = CompetitiveCoevolution(config, price_data=prices, seed=args.seed)
    result = coevo.run()

    _print_section("Coevolution Results")
    print(result.summary())

    if result.best_maker:
        _print_section("Best Market Maker Parameters")
        for key, value in result.best_maker.chromosome.to_dict().items():
            _print_result(key, value)

    if result.best_taker:
        _print_section("Best Market Taker Parameters")
        for key, value in result.best_taker.chromosome.to_dict().items():
            _print_result(key, value)

    output_path = getattr(args, "output", None)
    if output_path:
        output_data = {
            "best_maker": result.best_maker.chromosome.to_dict() if result.best_maker else None,
            "best_taker": result.best_taker.chromosome.to_dict() if result.best_taker else None,
            "maker_fitness": result.best_maker.fitness if result.best_maker else None,
            "taker_fitness": result.best_taker.fitness if result.best_taker else None,
            "n_generations": result.n_generations_run,
            "elapsed_seconds": result.elapsed_seconds,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n[Output] Results saved to: {output_path}")

    return 0


def cmd_ensemble(args: argparse.Namespace) -> int:
    """Run ensemble coevolution across market regimes."""
    _print_section("Ensemble Coevolution Across Market Regimes")

    prices = load_price_data(
        path=getattr(args, "data", None),
        n_synthetic=getattr(args, "n_prices", 500),
        regime="mixed",
    )

    ensemble = StrategyEnsembleCoevolution(
        price_data=prices,
        n_strategies_per_regime=getattr(args, "population", 10),
        n_generations=getattr(args, "generations", 30),
        seed=getattr(args, "seed", 42),
    )

    result = ensemble.run(verbose=not args.quiet)

    _print_section("Ensemble Results")
    for regime, params in result["best_per_regime"].items():
        fitness = result["best_fitnesses"][regime]
        print(f"\n  {Colors.CYAN}Regime: {regime}{Colors.RESET}")
        if params:
            for key, value in list(params.items())[:5]:
                _print_result(f"    {key}", value)
        fitness_str = f"{fitness:.4f}" if fitness is not None else "N/A"
        print(f"    {'Fitness':30s}: {fitness_str}")

    return 0


def cmd_optimize_bh(args: argparse.Namespace) -> int:
    """
    Evolve buy-and-hold parameters.
    Optimizes portfolio weights for a buy-and-hold strategy across assets.
    """
    n_assets = getattr(args, "n_assets", 10)
    _print_section(f"Optimizing Buy-and-Hold Portfolio ({n_assets} assets)")

    prices = load_price_data(
        path=getattr(args, "data", None),
        n_synthetic=getattr(args, "n_prices", 500),
        regime=getattr(args, "regime", "mixed"),
    )

    # For BH optimization, we simulate multiple assets
    n_assets = min(n_assets, 20)

    print(f"\n[Config] Assets: {n_assets}")
    print(f"[Config] Generations: {args.generations}")
    print(f"[Config] Population: {args.population}")

    result = evolve(
        strategy_type="portfolio_weights",
        price_data=prices,
        n_generations=args.generations,
        population_size=args.population,
        objectives=["sharpe", "calmar"] if args.multi_objective else ["sharpe"],
        multi_objective=args.multi_objective,
        seed=args.seed,
        verbose=not args.quiet,
        log_interval=getattr(args, "log_interval", 5),
        checkpoint_interval=0,
    )

    _print_section("BH Portfolio Optimization Results")
    print(result.summary())

    if result.best_genome:
        weights = {k: v for k, v in result.best_genome.chromosome.to_dict().items()
                   if k.startswith("weight_")}
        _print_section("Optimal Portfolio Weights")
        total_weight = sum(weights.values())
        for asset_key, w in sorted(weights.items()):
            normalized_w = w / max(total_weight, 1e-10)
            bar = "#" * int(normalized_w * 30)
            print(f"  {asset_key:15s}: {normalized_w:6.4f}  [{bar:<30}]")

    return 0


def cmd_optimize_ml(args: argparse.Namespace) -> int:
    """Evolve ML hyperparameters."""
    _print_section("Optimizing ML Hyperparameters")

    prices = load_price_data(
        path=getattr(args, "data", None),
        n_synthetic=getattr(args, "n_prices", 500),
        regime=getattr(args, "regime", "mixed"),
    )

    print(f"\n[Config] Generations: {args.generations}")
    print(f"[Config] Population: {args.population}")
    print(f"[Config] Objectives: {args.objectives}")
    print()

    objectives = args.objectives.split(",") if args.objectives else ["sharpe"]

    result = evolve(
        strategy_type="ml_hyperparameter",
        price_data=prices,
        n_generations=args.generations,
        population_size=args.population,
        objectives=objectives,
        multi_objective=len(objectives) > 1 or args.multi_objective,
        seed=args.seed,
        verbose=not args.quiet,
        log_interval=getattr(args, "log_interval", 5),
        checkpoint_interval=0,
    )

    _print_section("ML Hyperparameter Optimization Results")
    print(result.summary())

    if result.best_genome:
        _print_section("Best Hyperparameter Configuration")
        params = result.best_genome.chromosome.to_dict()
        for key, value in params.items():
            _print_result(key, value)

        if getattr(args, "output", None):
            _save_result(result, args.output)
            print(f"\n[Output] Saved to: {args.output}")

    return 0


def cmd_view_hof(args: argparse.Namespace) -> int:
    """View the Hall of Fame from a saved checkpoint."""
    checkpoint_path = getattr(args, "checkpoint", None)
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
        return 1

    with open(checkpoint_path) as f:
        data = json.load(f)

    _print_section("Hall of Fame")
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "hall_of_fame" in data:
        entries = data["hall_of_fame"]
    else:
        entries = [data]

    print(f"\n  {'Rank':5s} {'Genome ID':12s} {'Fitness':10s} {'Generation':10s}")
    print("  " + "-" * 45)
    for i, entry in enumerate(entries[:args.top_k]):
        gid = str(entry.get("genome_id", "unknown"))[:10]
        fitness = entry.get("fitness", "N/A")
        gen = entry.get("generation", "N/A")
        fitness_str = f"{fitness:.6f}" if isinstance(fitness, float) else str(fitness)
        print(f"  {i + 1:5d} {gid:12s} {fitness_str:10s} {gen!s:10s}")

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """
    Benchmark the GA on standard test problems.
    Compares different operator combinations.
    """
    _print_section("Operator Benchmark")

    prices = load_price_data(n_synthetic=300, seed=42, regime="mixed")

    configs = [
        ("gaussian+uniform", "gaussian", "uniform"),
        ("polynomial+sbx", "polynomial", "sbx"),
        ("adaptive+arithmetic", "gaussian", "arithmetic"),
        ("uniform+two_point", "uniform", "two_point"),
    ]

    results_list = []
    labels = []

    for label, mut_method, xover_method in configs:
        print(f"\n  Testing: {label} ...")
        factory = GenomeFactory(args.strategy, seed=42)
        fitness_config = FitnessConfig()
        fitness_evaluator = FitnessEvaluator(fitness_config, price_data=prices,
                                              strategy_type=args.strategy)

        evo_config = EvolutionConfig(
            population_size=20,
            n_generations=20,
            mutation_method=mut_method,
            crossover_method=xover_method,
            use_adaptive_mutation=False,
            use_adaptive_operators=False,
            verbose=False,
            checkpoint_interval=0,
            seed=42,
        )
        ga = GeneticAlgorithm(evo_config, factory, fitness_evaluator)
        result = ga.run()
        results_list.append(result)
        labels.append(label)

        best_f = result.best_genome.fitness if result.best_genome else 0.0
        print(f"    Best: {best_f:.4f}, Evals: {result.n_evaluations}")

    _print_section("Benchmark Results")
    print(f"\n  {'Configuration':35s} {'Best Fitness':12s} {'Evaluations':12s}")
    print("  " + "-" * 60)
    for label, result in zip(labels, results_list):
        best_f = result.best_genome.fitness if result.best_genome else 0.0
        print(f"  {label:35s} {best_f:12.6f} {result.n_evaluations:12d}")

    if getattr(args, "plot", False):
        config = PlotConfig(output_dir=getattr(args, "output_dir", "./plots"))
        path = plot_multi_run_comparison(results_list, config, labels=labels)
        if path:
            print(f"\n[Plot] Benchmark comparison saved: {path}")

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze and visualize a saved evolution result."""
    result_path = getattr(args, "result_file", None)
    if not result_path or not os.path.isfile(result_path):
        print(f"[ERROR] Result file not found: {result_path}")
        return 1

    with open(result_path) as f:
        data = json.load(f)

    _print_section("Result Analysis")
    print(f"\n  File: {result_path}")

    # Display basic info
    for key in ["best_fitness", "n_generations", "n_evaluations",
                 "converged", "elapsed_seconds"]:
        if key in data:
            _print_result(key, data[key])

    if "best_params" in data:
        _print_section("Best Parameters")
        for key, value in data["best_params"].items():
            _print_result(key, value)

    # ASCII visualization of fitness history
    if "fitness_history" in data:
        history = data["fitness_history"]
        if isinstance(history, list) and history:
            if isinstance(history[0], dict):
                bests = [h.get("best_fitness", 0) for h in history]
            else:
                bests = history
            chart = ASCIIPlotter.line_chart(bests, title="Fitness History")
            print("\n" + chart)

    return 0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_result(result: EvolutionResult, path: str) -> None:
    """Save evolution result to JSON file."""
    data: Dict[str, Any] = {
        "n_generations": result.n_generations_run,
        "n_evaluations": result.n_evaluations,
        "n_restarts": result.n_restarts,
        "converged": result.converged,
        "convergence_reason": result.convergence_reason,
        "elapsed_seconds": result.elapsed_seconds,
    }

    if result.best_genome:
        data["best_fitness"] = result.best_genome.fitness
        data["best_params"] = result.best_genome.chromosome.to_dict()
        data["best_genome_id"] = result.best_genome.metadata.genome_id
        data["best_generation"] = result.best_genome.metadata.generation

    data["hall_of_fame"] = result.hall_of_fame.to_dicts()

    if result.stats_history:
        data["fitness_history"] = [
            {
                "generation": s.generation,
                "best_fitness": s.best_fitness,
                "mean_fitness": s.mean_fitness,
                "std_fitness": s.std_fitness,
                "param_diversity": s.param_diversity,
                "unique_fingerprints": s.unique_fingerprints,
            }
            for s in result.stats_history
        ]

    if result.pareto_front:
        data["pareto_front"] = [
            {
                "objectives": g.objectives,
                "params": g.chromosome.to_dict(),
                "genome_id": g.metadata.genome_id,
            }
            for g in result.pareto_front[:50]
        ]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _generate_plots(result: EvolutionResult, args: argparse.Namespace) -> None:
    """Generate and save plots for an evolution result."""
    output_dir = getattr(args, "output_dir", "./plots")
    config = PlotConfig(output_dir=output_dir)
    dashboard = EvolutionDashboard(config)

    param_tracker = ParameterConvergenceVisualizer(config)
    genealogy = GenealogyTracker()

    # Record final population
    for g in result.final_population:
        genealogy.record(g)

    param_tracker.record_generation(result.final_population)

    paths = dashboard.generate_full_report(result, population=result.final_population)

    print(f"\n{Colors.CYAN}Generated plots:{Colors.RESET}")
    for name, path in paths.items():
        if path:
            print(f"  {name:30s}: {path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaso",
        description="Genetic Algorithm Strategy Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evolve momentum strategy
  python -m ml.genetic.cli evolve --strategy momentum --generations 100 --population 50

  # Multi-objective optimization
  python -m ml.genetic.cli evolve --strategy momentum --objectives sharpe,calmar --multi-objective

  # Portfolio weights optimization
  python -m ml.genetic.cli optimize-bh --n-assets 10 --generations 50

  # ML hyperparameter optimization
  python -m ml.genetic.cli optimize-ml --objectives sharpe,calmar --multi-objective

  # Competitive coevolution
  python -m ml.genetic.cli coevolve --generations 50 --population 30

  # Multiple runs
  python -m ml.genetic.cli multi-run --strategy momentum --n-runs 5

  # Benchmark operators
  python -m ml.genetic.cli benchmark --strategy momentum
        """,
    )

    # Global options
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ---- evolve ----
    p_evolve = subparsers.add_parser("evolve", help="Run strategy evolution")
    p_evolve.add_argument("--strategy", default="momentum",
                           choices=["momentum", "mean_reversion", "pairs_trading",
                                    "ml_hyperparameter", "portfolio_weights"],
                           help="Strategy type to evolve")
    p_evolve.add_argument("--generations", "-g", type=int, default=100)
    p_evolve.add_argument("--population", "-p", type=int, default=50)
    p_evolve.add_argument("--objectives", default="sharpe",
                           help="Comma-separated objectives (sharpe,calmar,profit_factor,...)")
    p_evolve.add_argument("--multi-objective", action="store_true")
    p_evolve.add_argument("--islands", action="store_true", help="Use island model")
    p_evolve.add_argument("--n-islands", type=int, default=4)
    p_evolve.add_argument("--data", help="Path to price data CSV")
    p_evolve.add_argument("--n-prices", type=int, default=500)
    p_evolve.add_argument("--regime", default="mixed",
                           choices=["mixed", "trending", "mean_reverting", "volatile", "crash"])
    p_evolve.add_argument("--output", "-o", help="Output JSON file path")
    p_evolve.add_argument("--output-dir", default="./plots")
    p_evolve.add_argument("--plot", action="store_true", help="Generate plots")
    p_evolve.add_argument("--log-interval", type=int, default=10)
    p_evolve.add_argument("--checkpoint-interval", type=int, default=25)
    p_evolve.add_argument("--checkpoint-dir", default="./checkpoints")
    p_evolve.add_argument("--seeds-file", help="JSON file with seed solutions")

    # ---- multi-run ----
    p_multi = subparsers.add_parser("multi-run", help="Multiple evolution runs")
    p_multi.add_argument("--strategy", default="momentum")
    p_multi.add_argument("--n-runs", type=int, default=5)
    p_multi.add_argument("--generations", "-g", type=int, default=50)
    p_multi.add_argument("--population", "-p", type=int, default=30)
    p_multi.add_argument("--data", help="Path to price data CSV")
    p_multi.add_argument("--n-prices", type=int, default=500)
    p_multi.add_argument("--regime", default="mixed")
    p_multi.add_argument("--plot", action="store_true")
    p_multi.add_argument("--output-dir", default="./plots")

    # ---- coevolve ----
    p_coe = subparsers.add_parser("coevolve", help="Competitive coevolution")
    p_coe.add_argument("--generations", "-g", type=int, default=50)
    p_coe.add_argument("--population", "-p", type=int, default=30)
    p_coe.add_argument("--mutation-rate", type=float, default=0.1)
    p_coe.add_argument("--data", help="Path to price data CSV")
    p_coe.add_argument("--n-prices", type=int, default=500)
    p_coe.add_argument("--regime", default="mixed")
    p_coe.add_argument("--output", "-o", help="Output JSON file")
    p_coe.add_argument("--log-interval", type=int, default=5)

    # ---- ensemble ----
    p_ens = subparsers.add_parser("ensemble", help="Ensemble regime coevolution")
    p_ens.add_argument("--generations", "-g", type=int, default=30)
    p_ens.add_argument("--population", "-p", type=int, default=10)
    p_ens.add_argument("--data", help="Path to price data CSV")
    p_ens.add_argument("--n-prices", type=int, default=500)

    # ---- optimize-bh ----
    p_bh = subparsers.add_parser("optimize-bh", help="Optimize buy-and-hold portfolio weights")
    p_bh.add_argument("--n-assets", type=int, default=10)
    p_bh.add_argument("--generations", "-g", type=int, default=50)
    p_bh.add_argument("--population", "-p", type=int, default=40)
    p_bh.add_argument("--multi-objective", action="store_true")
    p_bh.add_argument("--data", help="Path to price data CSV")
    p_bh.add_argument("--n-prices", type=int, default=500)
    p_bh.add_argument("--regime", default="mixed")
    p_bh.add_argument("--output", "-o", help="Output JSON file")
    p_bh.add_argument("--log-interval", type=int, default=5)

    # ---- optimize-ml ----
    p_ml = subparsers.add_parser("optimize-ml", help="Optimize ML hyperparameters")
    p_ml.add_argument("--generations", "-g", type=int, default=50)
    p_ml.add_argument("--population", "-p", type=int, default=40)
    p_ml.add_argument("--objectives", default="sharpe")
    p_ml.add_argument("--multi-objective", action="store_true")
    p_ml.add_argument("--data", help="Path to price data CSV")
    p_ml.add_argument("--n-prices", type=int, default=500)
    p_ml.add_argument("--regime", default="mixed")
    p_ml.add_argument("--output", "-o", help="Output JSON file")
    p_ml.add_argument("--log-interval", type=int, default=5)

    # ---- view-hof ----
    p_hof = subparsers.add_parser("view-hof", help="View Hall of Fame from checkpoint")
    p_hof.add_argument("checkpoint", help="Path to checkpoint JSON file")
    p_hof.add_argument("--top-k", type=int, default=20)

    # ---- benchmark ----
    p_bench = subparsers.add_parser("benchmark", help="Benchmark operator combinations")
    p_bench.add_argument("--strategy", default="momentum")
    p_bench.add_argument("--plot", action="store_true")
    p_bench.add_argument("--output-dir", default="./plots")

    # ---- analyze ----
    p_analyze = subparsers.add_parser("analyze", help="Analyze saved evolution result")
    p_analyze.add_argument("result_file", help="Path to result JSON file")

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "no_color", False):
        Colors.disable()

    _print_banner()

    if args.command is None:
        parser.print_help()
        return 0

    command_map = {
        "evolve": cmd_evolve,
        "multi-run": cmd_multi_run,
        "coevolve": cmd_coevolve,
        "ensemble": cmd_ensemble,
        "optimize-bh": cmd_optimize_bh,
        "optimize-ml": cmd_optimize_ml,
        "view-hof": cmd_view_hof,
        "benchmark": cmd_benchmark,
        "analyze": cmd_analyze,
    }

    handler = command_map.get(args.command)
    if handler is None:
        print(f"[ERROR] Unknown command: {args.command}")
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}[INTERRUPTED] Evolution stopped by user.{Colors.RESET}")
        return 130
    except Exception as e:
        print(f"\n{Colors.RED}[ERROR] {e}{Colors.RESET}")
        if not getattr(args, "quiet", False):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
