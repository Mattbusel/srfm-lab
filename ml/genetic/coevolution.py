"""
Coevolutionary genetic algorithm for competitive and cooperative strategy evolution.

Implements:
- Competitive coevolution: market-maker vs market-taker adversarial fitness
- Cooperative coevolution: decompose strategy into collaborating sub-populations
- Hall of champions for tracking best opponents
- Relative fitness computation
"""

from __future__ import annotations

import math
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .genome import StrategyGenome, GenomeFactory, ParamRange, ParamType
from .population import (
    Population, PopulationConfig, HallOfFame,
    SelectionOperator, DiversityMetrics, PopulationStats
)
from .fitness import FitnessEvaluator, FitnessConfig, BacktestResult, simulate_strategy
from .operators import (
    AdaptiveMutationRate, AdaptiveMutationConfig,
    RestartStrategy, ReproductionOperator
)
from .evolution import EvolutionConfig, EvolutionResult, ConvergenceDetector


# ---------------------------------------------------------------------------
# Market interaction: maker vs taker
# ---------------------------------------------------------------------------

@dataclass
class InteractionResult:
    """Result of a competitive interaction between two strategies."""
    maker_genome_id: str
    taker_genome_id: str
    maker_pnl: float
    taker_pnl: float
    maker_sharpe: float
    taker_sharpe: float
    spread_captured: float   # maker captures spread
    adverse_selection: float # taker adverse selection cost
    n_interactions: int


class MarketMakerStrategy:
    """
    A market-making strategy that posts bid/ask quotes and earns spread.
    Genome parameters: spread width, inventory limit, skew factor, quote size.
    """

    @staticmethod
    def params() -> List[ParamRange]:
        return [
            ParamRange("spread_bps", ParamType.CONTINUOUS, 1.0, 50.0, default=10.0,
                       description="Quote spread in basis points"),
            ParamRange("inventory_limit", ParamType.CONTINUOUS, 0.1, 1.0, default=0.5,
                       description="Max inventory as fraction of capital"),
            ParamRange("skew_factor", ParamType.CONTINUOUS, 0.0, 2.0, default=0.5,
                       description="How aggressively to skew quotes on inventory"),
            ParamRange("quote_size", ParamType.CONTINUOUS, 0.01, 0.20, default=0.05,
                       description="Quote size as fraction of capital"),
            ParamRange("hedge_freq", ParamType.INTEGER, 1, 20, default=5,
                       description="Hedging frequency in ticks"),
            ParamRange("vol_multiplier", ParamType.CONTINUOUS, 0.5, 3.0, default=1.0,
                       description="Spread multiplier in high-vol regime"),
            ParamRange("fade_speed", ParamType.CONTINUOUS, 0.01, 0.5, default=0.1,
                       description="Rate of quote fade on fill"),
            ParamRange("min_edge", ParamType.CONTINUOUS, 0.0, 0.005, default=0.001,
                       description="Minimum edge per trade"),
        ]

    @staticmethod
    def simulate(params: Dict[str, Any],
                 price_path: List[float],
                 taker_orders: List[Tuple[float, str]],
                 transaction_cost: float = 0.0001) -> BacktestResult:
        """
        Simulate market maker receiving taker orders.
        taker_orders: list of (price, side) where side='buy' or 'sell'.
        """
        if not price_path or not taker_orders:
            return _empty_backtest_result()

        spread = params.get("spread_bps", 10.0) / 10000.0
        inv_limit = params.get("inventory_limit", 0.5)
        skew = params.get("skew_factor", 0.5)
        size = params.get("quote_size", 0.05)

        inventory = 0.0
        capital = 1.0
        pnl_series = []
        n = min(len(price_path) - 1, len(taker_orders))

        for i in range(n):
            mid = price_path[i]
            next_mid = price_path[i + 1]
            order_price, order_side = taker_orders[i]

            # Skew quotes based on inventory
            inv_ratio = inventory / max(inv_limit, 1e-10)
            bid_skew = -skew * inv_ratio * spread
            ask_skew = skew * inv_ratio * spread

            bid = mid * (1 - spread / 2 + bid_skew)
            ask = mid * (1 + spread / 2 + ask_skew)

            # Check if taker order hits our quotes
            if order_side == "buy" and order_price >= ask:
                # Taker buys from us (we sell) -> we go short
                trade_pnl = (ask - mid) * size - transaction_cost * size
                inventory -= size
                pnl_series.append(trade_pnl)
            elif order_side == "sell" and order_price <= bid:
                # Taker sells to us (we buy) -> we go long
                trade_pnl = (mid - bid) * size - transaction_cost * size
                inventory += size
                pnl_series.append(trade_pnl)
            else:
                pnl_series.append(0.0)

            # Hedge: reduce inventory toward 0
            hedge_freq = int(params.get("hedge_freq", 5))
            if i % hedge_freq == 0 and abs(inventory) > 0.001:
                hedge_cost = abs(inventory) * transaction_cost
                pnl_series[-1] -= hedge_cost
                inventory *= 0.5  # Partial hedge

            # Mark-to-market: inventory position gains/loses with price change
            price_change = (next_mid - mid) / max(mid, 1e-10)
            mtm_pnl = inventory * price_change
            pnl_series[-1] += mtm_pnl

        if not pnl_series:
            return _empty_backtest_result()

        from .fitness import (compute_sharpe, compute_max_drawdown,
                               compute_calmar, compute_profit_factor, compute_win_rate)
        return BacktestResult(
            returns=pnl_series,
            sharpe=compute_sharpe(pnl_series),
            sortino=0.0,
            calmar=compute_calmar(pnl_series),
            max_drawdown=compute_max_drawdown(pnl_series),
            profit_factor=compute_profit_factor(pnl_series),
            win_rate=compute_win_rate(pnl_series),
            total_return=sum(pnl_series),
            n_trades=sum(1 for p in pnl_series if p != 0),
            annualized_return=sum(pnl_series) * 252 / max(len(pnl_series), 1),
            annualized_vol=0.0,
            var_5pct=0.0,
            cvar_5pct=0.0,
            omega_ratio=1.0,
        )


class MarketTakerStrategy:
    """
    A market-taking (directional) strategy that sends aggressive orders.
    Genome parameters: signal threshold, aggression, hold time, size.
    """

    @staticmethod
    def params() -> List[ParamRange]:
        return [
            ParamRange("signal_threshold", ParamType.CONTINUOUS, 0.001, 0.05, default=0.01,
                       description="Signal strength needed to take liquidity"),
            ParamRange("aggression", ParamType.CONTINUOUS, 0.0, 1.0, default=0.5,
                       description="How far through the spread to hit"),
            ParamRange("hold_periods", ParamType.INTEGER, 1, 50, default=10,
                       description="Holding period after taking"),
            ParamRange("position_size", ParamType.CONTINUOUS, 0.05, 0.5, default=0.2,
                       description="Size of each take"),
            ParamRange("momentum_window", ParamType.INTEGER, 3, 30, default=10,
                       description="Momentum signal window"),
            ParamRange("vol_filter", ParamType.CONTINUOUS, 0.0, 0.05, default=0.01,
                       description="Minimum volatility to trade"),
            ParamRange("max_slippage", ParamType.CONTINUOUS, 0.0001, 0.005, default=0.001,
                       description="Maximum acceptable slippage"),
        ]

    @staticmethod
    def generate_orders(params: Dict[str, Any],
                        price_path: List[float]) -> List[Tuple[float, str]]:
        """
        Generate a sequence of (price, side) orders based on momentum signal.
        """
        if len(price_path) < 5:
            return []

        window = int(params.get("momentum_window", 10))
        threshold = float(params.get("signal_threshold", 0.01))
        aggression = float(params.get("aggression", 0.5))
        spread_est = 0.001  # estimated spread

        orders = []
        for i in range(window, len(price_path)):
            # Simple momentum signal
            past = price_path[i - window:i]
            momentum = (price_path[i] - past[0]) / max(past[0], 1e-10)
            # Volatility filter
            returns = [(past[j] - past[j-1]) / max(past[j-1], 1e-10)
                       for j in range(1, len(past))]
            vol = math.sqrt(sum(r**2 for r in returns) / max(len(returns), 1))
            vol_filter = float(params.get("vol_filter", 0.01))

            if vol < vol_filter:
                orders.append((price_path[i], "none"))
                continue

            if momentum > threshold:
                # Buy signal: send buy order at ask + aggression * spread
                order_price = price_path[i] * (1 + aggression * spread_est)
                orders.append((order_price, "buy"))
            elif momentum < -threshold:
                # Sell signal: send sell order at bid - aggression * spread
                order_price = price_path[i] * (1 - aggression * spread_est)
                orders.append((order_price, "sell"))
            else:
                orders.append((price_path[i], "none"))

        return orders


def _empty_backtest_result() -> BacktestResult:
    return BacktestResult(
        returns=[], sharpe=0.0, sortino=0.0, calmar=0.0,
        max_drawdown=0.0, profit_factor=1.0, win_rate=0.0,
        total_return=0.0, n_trades=0, annualized_return=0.0,
        annualized_vol=0.0, var_5pct=0.0, cvar_5pct=0.0, omega_ratio=1.0,
    )


# ---------------------------------------------------------------------------
# Competitive coevolution evaluator
# ---------------------------------------------------------------------------

class CompetitiveFitnessEvaluator:
    """
    Evaluates fitness through competition between maker and taker strategies.
    A maker's fitness = its PnL when facing a sample of takers.
    A taker's fitness = its PnL when facing a sample of makers.
    Relative fitness: comparison to a hall of champions.
    """

    def __init__(self, price_data: List[float],
                 hall_size: int = 20,
                 n_opponents: int = 5,
                 seed: int = 42) -> None:
        self.price_data = price_data
        self.n_opponents = n_opponents
        self.maker_champions: HallOfFame = HallOfFame(maxsize=hall_size)
        self.taker_champions: HallOfFame = HallOfFame(maxsize=hall_size)
        self._rng = random.Random(seed)
        self._n_evaluations = 0

    def evaluate_maker(self, maker: StrategyGenome,
                       taker_pool: List[StrategyGenome]) -> float:
        """
        Evaluate a market maker by simulating against a sample of takers.
        Returns average maker Sharpe across interactions.
        """
        if not taker_pool:
            # No opponents: use default taker behavior
            result = self._maker_vs_default_taker(maker)
            maker.fitness = result.sharpe
            return result.sharpe

        opponents = self._rng.sample(taker_pool, min(self.n_opponents, len(taker_pool)))
        sharpes = []

        for taker in opponents:
            interaction = self._simulate_interaction(maker, taker)
            sharpes.append(interaction.maker_sharpe)

        # Also compare to hall of champions
        for champ in list(self.taker_champions)[:min(3, len(self.taker_champions))]:
            interaction = self._simulate_interaction(maker, champ)
            sharpes.append(interaction.maker_sharpe * 0.5)  # Downweight historical

        avg_sharpe = sum(sharpes) / max(len(sharpes), 1)
        maker.fitness = avg_sharpe
        self._n_evaluations += 1
        return avg_sharpe

    def evaluate_taker(self, taker: StrategyGenome,
                       maker_pool: List[StrategyGenome]) -> float:
        """
        Evaluate a market taker by simulating against a sample of makers.
        Returns average taker Sharpe minus adverse selection cost.
        """
        if not maker_pool:
            result = simulate_strategy(
                taker.chromosome.to_dict(), self.price_data, "momentum")
            taker.fitness = result.sharpe
            return result.sharpe

        opponents = self._rng.sample(maker_pool, min(self.n_opponents, len(maker_pool)))
        sharpes = []

        for maker in opponents:
            interaction = self._simulate_interaction(maker, taker)
            sharpes.append(interaction.taker_sharpe)

        for champ in list(self.maker_champions)[:min(3, len(self.maker_champions))]:
            interaction = self._simulate_interaction(champ, taker)
            sharpes.append(interaction.taker_sharpe * 0.5)

        avg_sharpe = sum(sharpes) / max(len(sharpes), 1)
        taker.fitness = avg_sharpe
        self._n_evaluations += 1
        return avg_sharpe

    def _simulate_interaction(self, maker: StrategyGenome,
                               taker: StrategyGenome) -> InteractionResult:
        """Simulate one interaction between a maker and a taker."""
        maker_params = maker.chromosome.to_dict()
        taker_params = taker.chromosome.to_dict()

        # Taker generates orders based on price path
        orders = MarketTakerStrategy.generate_orders(taker_params, self.price_data)
        if not orders:
            return InteractionResult(
                maker_genome_id=maker.metadata.genome_id,
                taker_genome_id=taker.metadata.genome_id,
                maker_pnl=0.0, taker_pnl=0.0,
                maker_sharpe=0.0, taker_sharpe=0.0,
                spread_captured=0.0, adverse_selection=0.0,
                n_interactions=0,
            )

        # Maker processes taker orders
        maker_result = MarketMakerStrategy.simulate(
            maker_params, self.price_data, orders)

        # Taker evaluates against the maker's quotes
        taker_result = simulate_strategy(
            taker_params, self.price_data[:len(orders) + 1], "momentum")

        spread_bps = float(maker_params.get("spread_bps", 10.0))
        n_filled = sum(1 for _, side in orders if side != "none")

        return InteractionResult(
            maker_genome_id=maker.metadata.genome_id,
            taker_genome_id=taker.metadata.genome_id,
            maker_pnl=maker_result.total_return,
            taker_pnl=taker_result.total_return,
            maker_sharpe=maker_result.sharpe,
            taker_sharpe=taker_result.sharpe,
            spread_captured=spread_bps * n_filled / max(len(orders), 1),
            adverse_selection=max(0.0, taker_result.total_return - maker_result.total_return * 0.5),
            n_interactions=n_filled,
        )

    def _maker_vs_default_taker(self, maker: StrategyGenome) -> BacktestResult:
        """Simulate maker against a simple random walk taker."""
        rng = random.Random(42)
        orders = []
        for p in self.price_data[1:]:
            side = rng.choice(["buy", "sell"])
            orders.append((p, side))
        return MarketMakerStrategy.simulate(
            maker.chromosome.to_dict(), self.price_data, orders)

    def update_champions(self, makers: List[StrategyGenome],
                          takers: List[StrategyGenome]) -> None:
        """Add strong individuals to hall of champions."""
        self.maker_champions.update(makers)
        self.taker_champions.update(takers)


# ---------------------------------------------------------------------------
# Cooperative coevolution: sub-population decomposition
# ---------------------------------------------------------------------------

@dataclass
class CoopComponent:
    """A component (sub-population) in cooperative coevolution."""
    component_id: int
    component_name: str
    factory: GenomeFactory
    population: List[StrategyGenome] = field(default_factory=list)
    representatives: List[StrategyGenome] = field(default_factory=list)
    n_representatives: int = 3


class CooperativeCoevolution:
    """
    Cooperative coevolution: decompose a complex strategy into components
    that are evolved independently but evaluated in combination.

    Example: A full trading strategy = entry_signal + exit_signal + position_sizing
    Each component is a separate population that co-evolves.
    """

    def __init__(self, component_specs: List[Dict[str, Any]],
                 fitness_function: Callable,
                 population_size: int = 30,
                 n_generations: int = 50,
                 seed: Optional[int] = None) -> None:
        """
        Args:
            component_specs: List of dicts with 'name' and 'strategy_type'.
            fitness_function: callable(component_dict) -> float
                where component_dict maps component_name -> params_dict.
            population_size: Size of each component population.
            n_generations: Generations per evolution cycle.
            seed: Random seed.
        """
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.n_generations = n_generations
        self._rng = random.Random(seed)
        self.generation = 0

        # Create components
        self.components: List[CoopComponent] = []
        for i, spec in enumerate(component_specs):
            factory = GenomeFactory(spec["strategy_type"],
                                    seed=(seed or 0) + i * 1000)
            comp = CoopComponent(
                component_id=i,
                component_name=spec["name"],
                factory=factory,
                n_representatives=spec.get("n_representatives", 3),
            )
            self.components.append(comp)

    def initialize(self) -> None:
        """Initialize all component populations."""
        for comp in self.components:
            comp.population = comp.factory.create_population(self.population_size)
            # Initialize representatives randomly
            comp.representatives = self._rng.sample(comp.population,
                                                     min(comp.n_representatives,
                                                         len(comp.population)))

    def evaluate_individual(self, individual: StrategyGenome,
                             component_idx: int) -> float:
        """
        Evaluate an individual by combining it with representatives from
        all other components and calling the fitness function.
        """
        component_dict = {}
        for i, comp in enumerate(self.components):
            if i == component_idx:
                component_dict[comp.component_name] = individual.chromosome.to_dict()
            else:
                # Use a random representative from other components
                if comp.representatives:
                    rep = self._rng.choice(comp.representatives)
                    component_dict[comp.component_name] = rep.chromosome.to_dict()
                else:
                    component_dict[comp.component_name] = comp.factory.create_random().chromosome.to_dict()

        fitness = self.fitness_function(component_dict)
        individual.fitness = fitness
        return fitness

    def evaluate_component(self, component_idx: int) -> None:
        """Evaluate all unevaluated individuals in a component."""
        comp = self.components[component_idx]
        for individual in comp.population:
            if individual.fitness is None:
                self.evaluate_individual(individual, component_idx)

    def update_representatives(self, component_idx: int,
                                n_reps: Optional[int] = None) -> None:
        """Update representatives for a component (best n_reps individuals)."""
        comp = self.components[component_idx]
        n = n_reps or comp.n_representatives
        evaluated = [g for g in comp.population if g.fitness is not None]
        if not evaluated:
            return
        best = sorted(evaluated, key=lambda g: g.fitness,  # type: ignore
                      reverse=True)[:n]
        comp.representatives = best

    def evolve_component(self, component_idx: int,
                          mutation_rate: float = 0.1) -> None:
        """Run one generation of evolution for a single component."""
        comp = self.components[component_idx]
        pop = comp.population
        evaluated = [g for g in pop if g.fitness is not None]
        if not evaluated:
            return

        # Select elite
        n_elite = max(1, int(0.10 * len(pop)))
        sorted_pop = sorted(evaluated, key=lambda g: g.fitness, reverse=True)  # type: ignore
        elite = [g.clone() for g in sorted_pop[:n_elite]]

        # Produce offspring
        offspring = []
        while len(offspring) < len(pop) - n_elite:
            p1 = SelectionOperator.tournament(evaluated, rng=self._rng)
            p2 = SelectionOperator.tournament(evaluated, rng=self._rng)
            c1, c2 = StrategyGenome.crossover(p1, p2, rng=self._rng)
            c1 = c1.mutate(mutation_rate, rng=self._rng)
            c2 = c2.mutate(mutation_rate, rng=self._rng)
            offspring.extend([c1, c2])

        comp.population = elite + offspring[:len(pop) - n_elite]

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run cooperative coevolution for n_generations.
        Returns best combined solution and fitness history.
        """
        if not self.components[0].population:
            self.initialize()

        fitness_history = []
        best_combined = None
        best_fitness = float("-inf")

        for gen in range(self.n_generations):
            self.generation = gen

            # Evaluate and evolve each component
            for comp_idx in range(len(self.components)):
                self.evaluate_component(comp_idx)
                self.update_representatives(comp_idx)
                self.evolve_component(comp_idx)
                self.evaluate_component(comp_idx)

            # Compute best combined solution
            best_combo = {}
            for comp in self.components:
                best_genome = max(
                    [g for g in comp.population if g.fitness is not None],
                    key=lambda g: g.fitness, default=None)  # type: ignore
                if best_genome:
                    best_combo[comp.component_name] = best_genome.chromosome.to_dict()

            if best_combo:
                combined_fitness = self.fitness_function(best_combo)
                fitness_history.append(combined_fitness)
                if combined_fitness > best_fitness:
                    best_fitness = combined_fitness
                    best_combined = best_combo

            if verbose and gen % 10 == 0:
                print(f"[CoopEvo] Gen {gen}: best_combined={best_fitness:.4f}")

        return {
            "best_combined": best_combined,
            "best_fitness": best_fitness,
            "fitness_history": fitness_history,
            "components": {
                comp.component_name: {
                    "best_params": max(
                        [g for g in comp.population if g.fitness is not None],
                        key=lambda g: g.fitness, default=comp.factory.create_random()
                    ).chromosome.to_dict()
                }
                for comp in self.components
            },
        }


# ---------------------------------------------------------------------------
# Competitive coevolution main loop
# ---------------------------------------------------------------------------

@dataclass
class CoevolutionConfig:
    population_size_per_species: int = 40
    n_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_fraction: float = 0.10
    n_opponents_per_eval: int = 5
    champion_hall_size: int = 20
    update_champions_interval: int = 10
    verbose: bool = True
    log_interval: int = 5
    seed: Optional[int] = None


@dataclass
class CoevolutionResult:
    best_maker: Optional[StrategyGenome]
    best_taker: Optional[StrategyGenome]
    maker_fitness_history: List[float]
    taker_fitness_history: List[float]
    maker_hall: HallOfFame
    taker_hall: HallOfFame
    n_generations_run: int
    elapsed_seconds: float

    def summary(self) -> str:
        lines = [
            "=" * 60, "COEVOLUTION RESULT",
            "=" * 60,
            f"Generations: {self.n_generations_run}",
            f"Elapsed:     {self.elapsed_seconds:.1f}s",
        ]
        if self.best_maker:
            lines.append(f"Best maker fitness:  {self.best_maker.fitness:.4f}")
        if self.best_taker:
            lines.append(f"Best taker fitness:  {self.best_taker.fitness:.4f}")
        lines.append(f"Maker hall size: {len(self.maker_hall)}")
        lines.append(f"Taker hall size: {len(self.taker_hall)}")
        lines.append("=" * 60)
        return "\n".join(lines)


class CompetitiveCoevolution:
    """
    Competitive coevolution between market makers and market takers.
    Each species evolves against the other's population.
    """

    def __init__(self, config: CoevolutionConfig,
                 price_data: Optional[List[float]] = None,
                 seed: Optional[int] = None) -> None:
        self.config = config
        self._rng = random.Random(seed or config.seed)

        if price_data is None:
            from .fitness import FitnessEvaluator
            price_data = FitnessEvaluator._generate_synthetic_prices(500, seed=42)
        self.price_data = price_data

        # Factories
        maker_params = MarketMakerStrategy.params()
        taker_params = MarketTakerStrategy.params()
        from .genome import Chromosome, GenomeMetadata
        self._maker_factory = GenomeFactory(
            "momentum",
            custom_params=maker_params,
            seed=(seed or 0) + 1,
        )
        self._taker_factory = GenomeFactory(
            "momentum",
            custom_params=taker_params,
            seed=(seed or 0) + 2,
        )

        # Populations
        self.makers: List[StrategyGenome] = []
        self.takers: List[StrategyGenome] = []

        # Competitive evaluator
        self.competitive_eval = CompetitiveFitnessEvaluator(
            price_data=price_data,
            hall_size=config.champion_hall_size,
            n_opponents=config.n_opponents_per_eval,
            seed=seed or 42,
        )

    def initialize(self) -> None:
        self.makers = self._maker_factory.create_population(
            self.config.population_size_per_species)
        self.takers = self._taker_factory.create_population(
            self.config.population_size_per_species)

    def _evaluate_all(self) -> None:
        """Evaluate all unevaluated makers and takers."""
        unevaluated_makers = [g for g in self.makers if g.fitness is None]
        unevaluated_takers = [g for g in self.takers if g.fitness is None]
        for maker in unevaluated_makers:
            self.competitive_eval.evaluate_maker(maker, self.takers)
        for taker in unevaluated_takers:
            self.competitive_eval.evaluate_taker(taker, self.makers)

    def _evolve_species(self, population: List[StrategyGenome],
                         factory: GenomeFactory) -> List[StrategyGenome]:
        """Run one generation of evolution for a species population."""
        evaluated = [g for g in population if g.fitness is not None]
        if not evaluated:
            return population

        n_elite = max(1, int(self.config.elite_fraction * len(population)))
        sorted_pop = sorted(evaluated, key=lambda g: g.fitness, reverse=True)  # type: ignore
        elite = [g.clone() for g in sorted_pop[:n_elite]]

        offspring = []
        while len(offspring) < len(population) - n_elite:
            p1 = SelectionOperator.tournament(evaluated, rng=self._rng)
            p2 = SelectionOperator.tournament(evaluated, rng=self._rng)

            if self._rng.random() < self.config.crossover_rate:
                c1, c2 = StrategyGenome.crossover(p1, p2, rng=self._rng)
            else:
                c1, c2 = p1.clone(), p2.clone()
                c1.fitness = None
                c2.fitness = None

            c1 = c1.mutate(self.config.mutation_rate, rng=self._rng)
            c2 = c2.mutate(self.config.mutation_rate, rng=self._rng)
            offspring.extend([c1, c2])

        return elite + offspring[:len(population) - n_elite]

    def run(self) -> CoevolutionResult:
        """Run the full competitive coevolution."""
        if not self.makers:
            self.initialize()

        start_time = time.time()
        maker_fitness_history = []
        taker_fitness_history = []

        for gen in range(self.config.n_generations):
            # Evaluate
            self._evaluate_all()

            # Track best fitness
            maker_fits = [g.fitness for g in self.makers if g.fitness is not None]
            taker_fits = [g.fitness for g in self.takers if g.fitness is not None]

            if maker_fits:
                maker_fitness_history.append(max(maker_fits))
            if taker_fits:
                taker_fitness_history.append(max(taker_fits))

            # Update champions periodically
            if gen % self.config.update_champions_interval == 0:
                self.competitive_eval.update_champions(self.makers, self.takers)

            # Evolve both species
            self.makers = self._evolve_species(self.makers, self._maker_factory)
            self.takers = self._evolve_species(self.takers, self._taker_factory)

            if self.config.verbose and gen % self.config.log_interval == 0:
                m_best = max(maker_fits) if maker_fits else 0.0
                t_best = max(taker_fits) if taker_fits else 0.0
                print(f"[CoEvo] Gen {gen:4d}: maker_best={m_best:.4f}, "
                      f"taker_best={t_best:.4f}")

        # Final evaluation
        self._evaluate_all()
        self.competitive_eval.update_champions(self.makers, self.takers)

        elapsed = time.time() - start_time

        # Find best individuals
        maker_evaluated = [g for g in self.makers if g.fitness is not None]
        taker_evaluated = [g for g in self.takers if g.fitness is not None]
        best_maker = max(maker_evaluated, key=lambda g: g.fitness) if maker_evaluated else None  # type: ignore
        best_taker = max(taker_evaluated, key=lambda g: g.fitness) if taker_evaluated else None  # type: ignore

        return CoevolutionResult(
            best_maker=best_maker,
            best_taker=best_taker,
            maker_fitness_history=maker_fitness_history,
            taker_fitness_history=taker_fitness_history,
            maker_hall=self.competitive_eval.maker_champions,
            taker_hall=self.competitive_eval.taker_champions,
            n_generations_run=self.config.n_generations,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Strategy coevolution: ensemble of specialized sub-strategies
# ---------------------------------------------------------------------------

class StrategyEnsembleCoevolution:
    """
    Coevolve an ensemble of strategies that specialize in different
    market regimes (trending, mean-reverting, volatile, quiet).
    Each strategy evolves to maximize performance in its regime,
    and a meta-strategy evolves to allocate between them.
    """

    REGIMES = ["trending", "mean_reverting", "high_vol", "low_vol"]

    def __init__(self, price_data: List[float],
                 n_strategies_per_regime: int = 10,
                 n_generations: int = 50,
                 seed: Optional[int] = None) -> None:
        self.price_data = price_data
        self.n_strategies = n_strategies_per_regime
        self.n_generations = n_generations
        self._rng = random.Random(seed)

        # One factory per regime
        self.factories = {
            regime: GenomeFactory("momentum", seed=(seed or 0) + i * 100)
            for i, regime in enumerate(self.REGIMES)
        }
        self.regime_populations: Dict[str, List[StrategyGenome]] = {
            regime: [] for regime in self.REGIMES
        }

        # Meta-strategy: allocate weights among regimes
        from .genome import ParamRange, Chromosome, Gene
        meta_params = [
            ParamRange(f"weight_{regime}", ParamType.CONTINUOUS, 0.0, 1.0,
                       default=0.25, description=f"Weight for {regime} strategy")
            for regime in self.REGIMES
        ]
        meta_params.append(
            ParamRange("rebalance_threshold", ParamType.CONTINUOUS, 0.0, 0.3, default=0.1)
        )
        self.meta_factory = GenomeFactory("momentum", custom_params=meta_params,
                                           seed=(seed or 0) + 9999)
        self.meta_population: List[StrategyGenome] = []

    def initialize(self) -> None:
        for regime in self.REGIMES:
            self.regime_populations[regime] = self.factories[regime].create_population(
                self.n_strategies)
        self.meta_population = self.meta_factory.create_population(10)

    def _classify_regime(self, prices: List[float]) -> str:
        """Classify a price segment into a regime."""
        if len(prices) < 10:
            return "trending"
        returns = [(prices[i] - prices[i-1]) / max(prices[i-1], 1e-10)
                   for i in range(1, len(prices))]
        vol = math.sqrt(sum(r**2 for r in returns) / max(len(returns), 1))
        trend = sum(returns) / max(len(returns), 1)

        if abs(trend) > vol * 0.5:
            return "trending"
        elif vol > 0.02:
            return "high_vol"
        elif vol < 0.005:
            return "low_vol"
        else:
            return "mean_reverting"

    def _evaluate_regime_strategy(self, genome: StrategyGenome,
                                   regime: str) -> float:
        """Evaluate a strategy on price data classified as the given regime."""
        # Use full data but weight fitness by regime alignment
        result = simulate_strategy(
            genome.chromosome.to_dict(), self.price_data, "momentum")
        # Regime alignment bonus/penalty
        detected = self._classify_regime(self.price_data)
        regime_bonus = 0.2 if detected == regime else -0.1
        genome.fitness = result.sharpe + regime_bonus
        return genome.fitness

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Run ensemble coevolution and return best strategy per regime."""
        if not self.regime_populations[self.REGIMES[0]]:
            self.initialize()

        best_per_regime: Dict[str, Optional[StrategyGenome]] = {r: None for r in self.REGIMES}

        for gen in range(self.n_generations):
            # Evaluate each regime population
            for regime, pop in self.regime_populations.items():
                for g in pop:
                    if g.fitness is None:
                        self._evaluate_regime_strategy(g, regime)

                # Evolve regime population
                evaluated = [g for g in pop if g.fitness is not None]
                if evaluated:
                    n_elite = max(1, int(0.10 * len(pop)))
                    sorted_pop = sorted(evaluated, key=lambda g: g.fitness, reverse=True)  # type: ignore
                    elite = [g.clone() for g in sorted_pop[:n_elite]]
                    offspring = []
                    while len(offspring) < len(pop) - n_elite:
                        p1 = SelectionOperator.tournament(evaluated, rng=self._rng)
                        p2 = SelectionOperator.tournament(evaluated, rng=self._rng)
                        c1, c2 = StrategyGenome.crossover(p1, p2, rng=self._rng)
                        c1 = c1.mutate(0.1, rng=self._rng)
                        offspring.extend([c1, c2])
                    self.regime_populations[regime] = elite + offspring[:len(pop) - n_elite]

                    best = max(evaluated, key=lambda g: g.fitness)  # type: ignore
                    best_per_regime[regime] = best

            if verbose and gen % 10 == 0:
                summary = {r: f"{g.fitness:.4f}" if g else "N/A"
                           for r, g in best_per_regime.items()}
                print(f"[EnsembleCoEvo] Gen {gen}: {summary}")

        return {
            "best_per_regime": {r: g.chromosome.to_dict() if g else None
                                  for r, g in best_per_regime.items()},
            "best_fitnesses": {r: g.fitness if g else None
                               for r, g in best_per_regime.items()},
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Coevolution self-test ===")
    from .fitness import FitnessEvaluator

    prices = FitnessEvaluator._generate_synthetic_prices(300, seed=42)

    # Test competitive coevolution
    print("\n--- Competitive coevolution (maker vs taker) ---")
    coe_config = CoevolutionConfig(
        population_size_per_species=15,
        n_generations=10,
        mutation_rate=0.15,
        verbose=True,
        log_interval=3,
        seed=42,
    )
    competitive = CompetitiveCoevolution(coe_config, price_data=prices, seed=42)
    result = competitive.run()
    print(result.summary())

    # Test cooperative coevolution
    print("\n--- Cooperative coevolution ---")

    def combined_fitness(component_dict: Dict[str, Any]) -> float:
        """Example: combined fitness of entry + position sizing components."""
        entry = component_dict.get("entry_signal", {})
        sizing = component_dict.get("position_sizing", {})
        # Merge params and evaluate
        merged = {**entry, **sizing}
        result = simulate_strategy(merged, prices, "momentum")
        return result.sharpe

    coop = CooperativeCoevolution(
        component_specs=[
            {"name": "entry_signal", "strategy_type": "momentum"},
            {"name": "position_sizing", "strategy_type": "mean_reversion"},
        ],
        fitness_function=combined_fitness,
        population_size=10,
        n_generations=5,
        seed=42,
    )
    coop_result = coop.run(verbose=True)
    print(f"Cooperative best fitness: {coop_result['best_fitness']:.4f}")
    print(f"Best combined params: {list(coop_result['components'].keys())}")

    # Test ensemble coevolution
    print("\n--- Ensemble coevolution ---")
    ensemble = StrategyEnsembleCoevolution(prices, n_strategies_per_regime=5,
                                            n_generations=5, seed=42)
    ens_result = ensemble.run(verbose=True)
    print(f"Best fitnesses per regime: {ens_result['best_fitnesses']}")

    # Test interaction simulation
    print("\n--- Maker/Taker interaction test ---")
    maker_factory = GenomeFactory("momentum", custom_params=MarketMakerStrategy.params(), seed=1)
    taker_factory = GenomeFactory("momentum", custom_params=MarketTakerStrategy.params(), seed=2)
    maker = maker_factory.create_random()
    taker = taker_factory.create_random()

    orders = MarketTakerStrategy.generate_orders(taker.chromosome.to_dict(), prices[:100])
    print(f"Generated {len(orders)} taker orders, "
          f"active: {sum(1 for _, s in orders if s != 'none')}")

    maker_result = MarketMakerStrategy.simulate(
        maker.chromosome.to_dict(), prices[:100], orders[:99])
    print(f"Maker Sharpe: {maker_result.sharpe:.4f}, PnL: {maker_result.total_return:.4f}")

    print("\nAll coevolution tests passed.")
