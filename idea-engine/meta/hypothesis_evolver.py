"""
hypothesis_evolver.py — Evolutionary hypothesis optimizer for idea-engine.

Improves trading hypotheses over generations via:
  - HypothesisGene: parameter vector (entry threshold, lookback, sizing)
  - EvolutionState: population with fitness scores
  - Fitness: Sharpe ratio on held-out test window
  - Selection: tournament
  - Crossover: blend crossover (BLX-alpha)
  - Mutation: Gaussian perturbation
  - Elitism: always keep top N
  - Novelty search: penalize near-duplicate hypotheses
  - Island model: sub-populations with periodic migration
  - Multi-objective: Pareto frontier (Sharpe / Calmar / MaxDD)
  - HypothesisEvolver.evolve(population, n_generations) → evolved population
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Gene and Hypothesis Definitions
# ---------------------------------------------------------------------------

GENE_NAMES = [
    "entry_threshold",      # signal z-score to enter (0.5 - 3.0)
    "exit_threshold",       # signal z-score to exit (0.0 - 2.0)
    "lookback",             # lookback period in days (5 - 252)
    "vol_lookback",         # volatility lookback (5 - 63)
    "sizing_scale",         # position sizing multiplier (0.1 - 2.0)
    "stop_loss",            # stop loss as fraction of entry price (0.005 - 0.10)
    "take_profit",          # take profit fraction (0.005 - 0.20)
    "signal_smoothing",     # EMA smoothing factor for signal (0.0 - 0.5)
    "min_holding_days",     # minimum holding period (1 - 21)
    "regime_filter",        # threshold for regime filter (0.0 - 1.0)
]

GENE_BOUNDS: List[Tuple[float, float]] = [
    (0.5,  3.0),
    (0.0,  2.0),
    (5.0,  252.0),
    (5.0,  63.0),
    (0.1,  2.0),
    (0.005, 0.10),
    (0.005, 0.20),
    (0.0,  0.5),
    (1.0,  21.0),
    (0.0,  1.0),
]
N_GENES = len(GENE_NAMES)


@dataclass
class HypothesisGene:
    """Parameter vector encoding a trading hypothesis."""
    params: np.ndarray                  # shape (N_GENES,)
    gene_id: int = 0
    parent_ids: Tuple[int, int] = (0, 0)
    generation: int = 0

    def __post_init__(self):
        self.params = np.clip(self.params, [b[0] for b in GENE_BOUNDS], [b[1] for b in GENE_BOUNDS])

    def entry_threshold(self) -> float:
        return float(self.params[0])

    def exit_threshold(self) -> float:
        return float(self.params[1])

    def lookback(self) -> int:
        return max(5, int(round(self.params[2])))

    def vol_lookback(self) -> int:
        return max(5, int(round(self.params[3])))

    def sizing_scale(self) -> float:
        return float(self.params[4])

    def stop_loss(self) -> float:
        return float(self.params[5])

    def take_profit(self) -> float:
        return float(self.params[6])

    def signal_smoothing(self) -> float:
        return float(self.params[7])

    def min_holding_days(self) -> int:
        return max(1, int(round(self.params[8])))

    def regime_filter(self) -> float:
        return float(self.params[9])

    def distance(self, other: "HypothesisGene") -> float:
        """Normalized Euclidean distance in gene space."""
        ranges = np.array([b[1] - b[0] for b in GENE_BOUNDS])
        diff = (self.params - other.params) / ranges
        return float(np.linalg.norm(diff))

    @staticmethod
    def random(rng: np.random.Generator, gene_id: int = 0, generation: int = 0) -> "HypothesisGene":
        lo = np.array([b[0] for b in GENE_BOUNDS])
        hi = np.array([b[1] for b in GENE_BOUNDS])
        params = rng.uniform(lo, hi)
        return HypothesisGene(params=params, gene_id=gene_id, generation=generation)

    def to_dict(self) -> Dict:
        return {name: float(self.params[i]) for i, name in enumerate(GENE_NAMES)}


# ---------------------------------------------------------------------------
# Fitness Evaluation
# ---------------------------------------------------------------------------

@dataclass
class FitnessResult:
    sharpe: float
    calmar: float
    max_drawdown: float
    total_return: float
    n_trades: int
    win_rate: float


def _simulate_hypothesis(
    gene: HypothesisGene,
    prices: np.ndarray,         # shape (T,) close prices
    signal: np.ndarray,         # shape (T,) raw signal (e.g. z-score of some indicator)
) -> FitnessResult:
    """
    Simulate a simple long/short strategy defined by gene on price series.
    Returns FitnessResult with Sharpe, Calmar, MaxDD etc.
    """
    T = len(prices)
    lb = gene.lookback()
    vol_lb = gene.vol_lookback()
    entry_thr = gene.entry_threshold()
    exit_thr = gene.exit_threshold()
    size = gene.sizing_scale()
    stop = gene.stop_loss()
    tp = gene.take_profit()
    smooth = gene.signal_smoothing()
    min_hold = gene.min_holding_days()

    # Smooth signal
    if smooth > 0:
        smoothed = np.zeros(T)
        alpha_s = 2.0 / (1.0 + max(1, int(1.0 / smooth)))
        smoothed[0] = signal[0]
        for t in range(1, T):
            smoothed[t] = alpha_s * signal[t] + (1 - alpha_s) * smoothed[t - 1]
        sig = smoothed
    else:
        sig = signal.copy()

    # Realized vol for position sizing
    log_rets = np.diff(np.log(np.maximum(prices, 1e-10)))

    position = 0.0
    entry_price = 0.0
    holding_days = 0
    daily_rets: List[float] = []
    trades: List[float] = []

    for t in range(lb, T):
        price = prices[t]
        ret = log_rets[t - 1] if t > 0 else 0.0

        if position != 0.0:
            trade_ret = position * ret
            daily_rets.append(trade_ret)
            holding_days += 1

            pnl_pct = (price - entry_price) / entry_price * position
            if pnl_pct <= -stop or pnl_pct >= tp or (abs(sig[t]) < exit_thr and holding_days >= min_hold):
                trades.append(pnl_pct)
                position = 0.0
                entry_price = 0.0
                holding_days = 0
        else:
            daily_rets.append(0.0)
            # Local vol for sizing
            local_vol = float(np.std(log_rets[max(0, t - vol_lb):t])) * math.sqrt(252)
            vol_adj = 0.10 / max(local_vol, 0.01)  # target 10% vol

            if sig[t] > entry_thr:
                position = +size * min(vol_adj, 3.0)
                entry_price = price
                holding_days = 0
            elif sig[t] < -entry_thr:
                position = -size * min(vol_adj, 3.0)
                entry_price = price
                holding_days = 0

    rets = np.array(daily_rets)
    if len(rets) < 10:
        return FitnessResult(0.0, 0.0, 0.0, 0.0, 0, 0.0)

    ann_ret = float(np.mean(rets)) * 252
    ann_vol = float(np.std(rets)) * math.sqrt(252)
    sharpe = ann_ret / max(ann_vol, 1e-8)

    cum = np.exp(np.cumsum(rets))
    running_max = np.maximum.accumulate(cum)
    drawdowns = (running_max - cum) / running_max
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
    calmar = ann_ret / max(max_dd, 1e-8)

    n_trades = len(trades)
    win_rate = float(np.mean([t > 0 for t in trades])) if trades else 0.0

    return FitnessResult(
        sharpe=sharpe,
        calmar=calmar,
        max_drawdown=max_dd,
        total_return=float(np.sum(rets)),
        n_trades=n_trades,
        win_rate=win_rate,
    )


# ---------------------------------------------------------------------------
# Population and Evolution State
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    gene: HypothesisGene
    fitness: FitnessResult = field(default_factory=lambda: FitnessResult(0, 0, 0, 0, 0, 0))
    novelty_score: float = 0.0
    island_id: int = 0
    is_elite: bool = False

    @property
    def scalar_fitness(self) -> float:
        """Weighted single objective for comparison."""
        return self.fitness.sharpe

    @property
    def multi_objective(self) -> Tuple[float, float, float]:
        return (self.fitness.sharpe, self.fitness.calmar, -self.fitness.max_drawdown)


@dataclass
class EvolutionState:
    population: List[Individual]
    generation: int = 0
    best_fitness: float = -999.0
    pareto_front: List[Individual] = field(default_factory=list)
    diversity_metric: float = 0.0
    island_populations: List[List[Individual]] = field(default_factory=list)

    def best(self) -> Individual:
        return max(self.population, key=lambda x: x.scalar_fitness)

    def mean_fitness(self) -> float:
        return float(np.mean([ind.scalar_fitness for ind in self.population]))


# ---------------------------------------------------------------------------
# Selection Operators
# ---------------------------------------------------------------------------

def tournament_selection(
    population: List[Individual],
    rng: np.random.Generator,
    tournament_size: int = 4,
    use_novelty: bool = False,
) -> Individual:
    """Select one individual via tournament selection."""
    contestants = rng.choice(len(population), size=min(tournament_size, len(population)), replace=False)
    contestants = [population[i] for i in contestants]
    if use_novelty:
        key = lambda x: x.scalar_fitness + 0.3 * x.novelty_score
    else:
        key = lambda x: x.scalar_fitness
    return max(contestants, key=key)


# ---------------------------------------------------------------------------
# Crossover Operators
# ---------------------------------------------------------------------------

def blend_crossover(
    parent1: HypothesisGene,
    parent2: HypothesisGene,
    rng: np.random.Generator,
    alpha: float = 0.5,
    gene_id: int = 0,
    generation: int = 0,
) -> HypothesisGene:
    """
    BLX-alpha crossover: each gene is sampled uniformly from
    [min(p1,p2) - alpha*d, max(p1,p2) + alpha*d] where d = |p1 - p2|.
    """
    p1, p2 = parent1.params, parent2.params
    lo_b = np.array([b[0] for b in GENE_BOUNDS])
    hi_b = np.array([b[1] for b in GENE_BOUNDS])
    d = np.abs(p1 - p2)
    lo = np.maximum(np.minimum(p1, p2) - alpha * d, lo_b)
    hi = np.minimum(np.maximum(p1, p2) + alpha * d, hi_b)
    child_params = rng.uniform(lo, hi)
    return HypothesisGene(
        params=child_params, gene_id=gene_id,
        parent_ids=(parent1.gene_id, parent2.gene_id),
        generation=generation,
    )


def uniform_crossover(
    parent1: HypothesisGene,
    parent2: HypothesisGene,
    rng: np.random.Generator,
    gene_id: int = 0,
    generation: int = 0,
) -> HypothesisGene:
    """Each gene independently chosen from either parent with p=0.5."""
    mask = rng.random(N_GENES) < 0.5
    child_params = np.where(mask, parent1.params, parent2.params)
    return HypothesisGene(
        params=child_params, gene_id=gene_id,
        parent_ids=(parent1.gene_id, parent2.gene_id),
        generation=generation,
    )


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def gaussian_mutation(
    gene: HypothesisGene,
    rng: np.random.Generator,
    mutation_rate: float = 0.15,
    sigma_fraction: float = 0.1,
) -> HypothesisGene:
    """
    Gaussian perturbation. Each gene mutated independently with probability
    mutation_rate. Sigma = sigma_fraction * (hi - lo) for each gene.
    """
    lo_b = np.array([b[0] for b in GENE_BOUNDS])
    hi_b = np.array([b[1] for b in GENE_BOUNDS])
    ranges = hi_b - lo_b
    sigma = sigma_fraction * ranges

    new_params = gene.params.copy()
    for i in range(N_GENES):
        if rng.random() < mutation_rate:
            new_params[i] += rng.standard_normal() * sigma[i]
    new_params = np.clip(new_params, lo_b, hi_b)
    return HypothesisGene(params=new_params, gene_id=gene.gene_id, generation=gene.generation)


# ---------------------------------------------------------------------------
# Novelty Search
# ---------------------------------------------------------------------------

def compute_novelty_scores(
    population: List[Individual],
    k_nearest: int = 5,
) -> np.ndarray:
    """
    Novelty = mean distance to k nearest neighbors.
    Penalizes redundant hypotheses.
    """
    n = len(population)
    genes = [ind.gene for ind in population]
    novelty = np.zeros(n)
    for i in range(n):
        dists = sorted([genes[i].distance(genes[j]) for j in range(n) if j != i])
        k = min(k_nearest, len(dists))
        novelty[i] = float(np.mean(dists[:k])) if k > 0 else 0.0
    return novelty


# ---------------------------------------------------------------------------
# Pareto Dominance
# ---------------------------------------------------------------------------

def _dominates(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
    """Returns True if a dominates b (all objectives >=, at least one >)."""
    return all(ai >= bi for ai, bi in zip(a, b)) and any(ai > bi for ai, bi in zip(a, b))


def compute_pareto_front(population: List[Individual]) -> List[Individual]:
    """Return the non-dominated set (Pareto front)."""
    front = []
    for i, ind_i in enumerate(population):
        dominated = False
        for j, ind_j in enumerate(population):
            if i != j and _dominates(ind_j.multi_objective, ind_i.multi_objective):
                dominated = True
                break
        if not dominated:
            front.append(ind_i)
    return front


# ---------------------------------------------------------------------------
# Island Model
# ---------------------------------------------------------------------------

@dataclass
class IslandModelConfig:
    n_islands: int = 4
    migration_interval: int = 10    # generations between migrations
    migration_rate: float = 0.1     # fraction of population that migrates
    migration_topology: str = "ring"  # "ring" or "fully_connected"


def migrate(
    islands: List[List[Individual]],
    cfg: IslandModelConfig,
    rng: np.random.Generator,
) -> List[List[Individual]]:
    """
    Migrate top individuals between islands.
    Ring topology: island i sends to island (i+1) % n_islands.
    """
    n = len(islands)
    n_migrate = max(1, int(len(islands[0]) * cfg.migration_rate))
    new_islands = [list(isl) for isl in islands]

    if cfg.migration_topology == "ring":
        connections = [(i, (i + 1) % n) for i in range(n)]
    else:
        connections = [(i, j) for i in range(n) for j in range(n) if i != j]

    for src, dst in connections:
        # Send top n_migrate from src to dst (replace worst in dst)
        sorted_src = sorted(new_islands[src], key=lambda x: x.scalar_fitness, reverse=True)
        migrants = copy.deepcopy(sorted_src[:n_migrate])
        for m in migrants:
            m.island_id = dst

        sorted_dst = sorted(new_islands[dst], key=lambda x: x.scalar_fitness)
        for i, m in enumerate(migrants):
            if i < len(sorted_dst):
                idx = new_islands[dst].index(sorted_dst[i])
                new_islands[dst][idx] = m

    return new_islands


# ---------------------------------------------------------------------------
# Main Evolver
# ---------------------------------------------------------------------------

@dataclass
class EvolverConfig:
    population_size: int = 100
    n_elite: int = 5
    tournament_size: int = 5
    crossover_prob: float = 0.7
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.10
    novelty_weight: float = 0.2
    use_islands: bool = True
    island_cfg: IslandModelConfig = field(default_factory=IslandModelConfig)
    seed: int = 42


class HypothesisEvolver:
    """
    Evolutionary optimizer for trading hypothesis genes.

    Usage:
        evolver = HypothesisEvolver(cfg, prices=..., signal=..., test_split=0.7)
        state = evolver.initialize()
        state = evolver.evolve(state, n_generations=50)
        best = state.best()
    """
    def __init__(
        self,
        cfg: EvolverConfig,
        prices: np.ndarray,
        signal: np.ndarray,
        test_split: float = 0.70,
        fitness_fn: Optional[Callable] = None,
    ):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        split = int(len(prices) * test_split)
        self.train_prices = prices[:split]
        self.test_prices = prices[split:]
        self.train_signal = signal[:split]
        self.test_signal = signal[split:]
        self._gene_counter = 0
        self._fitness_fn = fitness_fn or _simulate_hypothesis

    def _next_id(self) -> int:
        self._gene_counter += 1
        return self._gene_counter

    def _evaluate(self, gene: HypothesisGene, on_test: bool = False) -> FitnessResult:
        prices = self.test_prices if on_test else self.train_prices
        signal = self.test_signal if on_test else self.train_signal
        if len(prices) < gene.lookback() + 10:
            return FitnessResult(0, 0, 1.0, 0, 0, 0)
        return self._fitness_fn(gene, prices, signal)

    def initialize(self) -> EvolutionState:
        population = []
        for _ in range(self.cfg.population_size):
            gene = HypothesisGene.random(self.rng, gene_id=self._next_id(), generation=0)
            fitness = self._evaluate(gene)
            individual = Individual(gene=gene, fitness=fitness)
            population.append(individual)

        state = EvolutionState(population=population, generation=0)
        state.best_fitness = state.best().scalar_fitness

        if self.cfg.use_islands:
            n_isl = self.cfg.island_cfg.n_islands
            island_size = self.cfg.population_size // n_isl
            islands = []
            for i in range(n_isl):
                isl = population[i * island_size:(i + 1) * island_size]
                for ind in isl:
                    ind.island_id = i
                islands.append(isl)
            state.island_populations = islands

        return state

    def _evolve_island(
        self,
        island: List[Individual],
        generation: int,
    ) -> List[Individual]:
        cfg = self.cfg
        n = len(island)
        if n == 0:
            return island

        # Compute novelty
        novelty = compute_novelty_scores(island)
        for i, ind in enumerate(island):
            ind.novelty_score = float(novelty[i])

        # Elitism: keep top N
        sorted_pop = sorted(island, key=lambda x: x.scalar_fitness, reverse=True)
        n_elite = min(cfg.n_elite, n)
        elites = copy.deepcopy(sorted_pop[:n_elite])
        for e in elites:
            e.is_elite = True

        new_pop: List[Individual] = elites[:]

        while len(new_pop) < n:
            # Selection
            p1 = tournament_selection(island, self.rng, cfg.tournament_size, use_novelty=True)
            p2 = tournament_selection(island, self.rng, cfg.tournament_size, use_novelty=True)

            # Crossover
            if self.rng.random() < cfg.crossover_prob:
                child_gene = blend_crossover(p1.gene, p2.gene, self.rng,
                                              gene_id=self._next_id(), generation=generation)
            else:
                child_gene = copy.deepcopy(p1.gene)
                child_gene.gene_id = self._next_id()
                child_gene.generation = generation

            # Mutation
            child_gene = gaussian_mutation(child_gene, self.rng,
                                            cfg.mutation_rate, cfg.mutation_sigma)

            # Evaluate
            fitness = self._evaluate(child_gene)
            new_pop.append(Individual(gene=child_gene, fitness=fitness))

        return new_pop[:n]

    def evolve(self, state: EvolutionState, n_generations: int) -> EvolutionState:
        """
        Run n_generations of evolution.
        Returns the evolved EvolutionState with updated population and Pareto front.
        """
        cfg = self.cfg
        island_cfg = cfg.island_cfg

        for gen in range(1, n_generations + 1):
            state.generation += 1

            if cfg.use_islands and state.island_populations:
                # Evolve each island independently
                new_islands = []
                for isl in state.island_populations:
                    new_isl = self._evolve_island(isl, state.generation)
                    new_islands.append(new_isl)

                # Migration
                if gen % island_cfg.migration_interval == 0:
                    new_islands = migrate(new_islands, island_cfg, self.rng)

                state.island_populations = new_islands
                state.population = [ind for isl in new_islands for ind in isl]
            else:
                state.population = self._evolve_island(state.population, state.generation)

            # Update state metadata
            current_best = state.best().scalar_fitness
            if current_best > state.best_fitness:
                state.best_fitness = current_best

            # Diversity metric: mean pairwise distance in sample
            sample_size = min(20, len(state.population))
            sample = state.population[:sample_size]
            if len(sample) > 1:
                dists = []
                for i in range(len(sample)):
                    for j in range(i + 1, len(sample)):
                        dists.append(sample[i].gene.distance(sample[j].gene))
                state.diversity_metric = float(np.mean(dists))

        # Final Pareto front computation
        state.pareto_front = compute_pareto_front(state.population)

        # Re-evaluate top individuals on test set
        top_n = min(20, len(state.population))
        sorted_pop = sorted(state.population, key=lambda x: x.scalar_fitness, reverse=True)
        for ind in sorted_pop[:top_n]:
            ind.fitness = self._evaluate(ind.gene, on_test=True)

        return state

    def get_pareto_summary(self, state: EvolutionState) -> List[Dict]:
        """Return the Pareto front as a list of dicts for analysis."""
        summary = []
        for ind in state.pareto_front:
            d = ind.gene.to_dict()
            d["sharpe"] = ind.fitness.sharpe
            d["calmar"] = ind.fitness.calmar
            d["max_drawdown"] = ind.fitness.max_drawdown
            d["n_trades"] = ind.fitness.n_trades
            d["win_rate"] = ind.fitness.win_rate
            d["novelty"] = ind.novelty_score
            d["generation"] = ind.gene.generation
            summary.append(d)
        return summary
