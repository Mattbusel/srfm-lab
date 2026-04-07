"""
gp_engine.py -- Genetic programming engine for alpha signal discovery.

Evolves symbolic expression trees using a generational GP loop with:
  - Ramped half-and-half initialization
  - Tournament + lexicase hybrid parent selection
  - Subtree crossover and mixed mutation
  - Elite preservation
  - Parsimony pressure (complexity penalty)
  - Multi-objective fitness: IC, ICIR, regime-conditioned IC, Sharpe
  - Pareto front extraction at termination
  - Automatic bloat control via hoist mutation

Dependencies: numpy, scipy (for Spearman rank correlation only).
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

from .expression_tree import ExpressionTree, TreeGenerator
from .gp_operators import (
    subtree_crossover,
    point_mutation,
    subtree_mutation,
    hoist_mutation,
    tournament_select,
    lexicase_select,
    apply_random_mutation,
)


# ---------------------------------------------------------------------------
# Spearman IC (no scipy fallback)
# ---------------------------------------------------------------------------

def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Spearman rank correlation between x and y, ignoring NaN pairs.
    Falls back to a pure-numpy rank-based implementation if scipy is absent.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 4:
        return 0.0
    xv, yv = x[mask], y[mask]
    if _HAVE_SCIPY:
        rho, _ = spearmanr(xv, yv)
        return float(rho) if np.isfinite(rho) else 0.0
    # pure numpy: convert to ranks
    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        return ranks
    rx, ry = _rank(xv), _rank(yv)
    n = len(rx)
    if n < 2:
        return 0.0
    rx_m = rx - rx.mean()
    ry_m = ry - ry.mean()
    denom = np.sqrt((rx_m ** 2).sum() * (ry_m ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((rx_m * ry_m).sum() / denom)


# ---------------------------------------------------------------------------
# Rolling IC over time windows
# ---------------------------------------------------------------------------

def _rolling_ic(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """
    Compute rolling Spearman IC with a sliding window.
    Returns array of length len(signal) - window + 1.
    """
    n = len(signal)
    if n < window:
        return np.array([_spearman_ic(signal, forward_returns)])
    ic_vals = []
    for i in range(window, n + 1):
        ic = _spearman_ic(signal[i - window: i], forward_returns[i - window: i])
        ic_vals.append(ic)
    return np.array(ic_vals)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """A GP individual: expression tree + fitness scores."""
    tree: ExpressionTree
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    id: int = field(default_factory=lambda: random.randint(0, 10 ** 9))

    @property
    def combined_fitness(self) -> float:
        return self.fitness_scores.get("combined", -999.0)

    def __lt__(self, other: "Individual") -> bool:
        return self.combined_fitness < other.combined_fitness


@dataclass
class GPConfig:
    """Configuration for the GP engine."""
    population_size: int     = 500
    max_generations: int     = 100
    crossover_rate: float    = 0.80
    mutation_rate: float     = 0.15
    max_depth: int           = 8
    tournament_k: int        = 7
    elite_n: int             = 10
    min_ic_threshold: float  = 0.05

    # fitness weights
    w_ic: float              = 0.40
    w_icir: float            = 0.30
    w_regime: float          = 0.20
    # complexity penalty coefficient
    complexity_coef: float   = 0.001

    # rolling IC window for ICIR
    ic_window: int           = 30

    # minimum non-NaN fraction required to score an individual
    min_valid_frac: float    = 0.10

    # verbosity: 0=silent, 1=per-generation summary, 2=full
    verbosity: int           = 1

    # early stopping: stop if best combined fitness does not improve by this
    # amount over stagnation_patience generations
    stagnation_patience: int  = 20
    stagnation_delta: float   = 1e-4

    # lexicase weight in hybrid selection (rest uses tournament)
    lexicase_prob: float      = 0.30


# ---------------------------------------------------------------------------
# GPEngine
# ---------------------------------------------------------------------------

class GPEngine:
    """
    Genetic programming engine for discovering alpha signal expressions.

    Usage
    -----
    engine = GPEngine()
    config = GPConfig(population_size=200, max_generations=50)
    pareto = engine.run(data, forward_returns, config)
    best   = pareto[0]
    export = engine.export_best(best)
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.generator = TreeGenerator()
        self._eval_counter = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_population(
        self, config: GPConfig
    ) -> List[Individual]:
        """
        Create initial population using ramped half-and-half.
        Depth ranges from 2 to min(config.max_depth, 6).
        """
        population: List[Individual] = []
        for _ in range(config.population_size):
            tree = self.generator.random_tree_ramped(
                max_depth=min(config.max_depth, 6)
            )
            ind = Individual(tree=tree, generation=0)
            population.append(ind)
        return population

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def evaluate_fitness(
        self,
        individual: Individual,
        data: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        regime_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single individual.

        Metrics computed:
          ic          -- Spearman correlation with 1-bar forward return
          icir        -- IC / std(IC) over rolling windows
          ic_regime_on  -- IC when regime is active (regime_mask == 1)
          ic_regime_off -- IC when regime is inactive (regime_mask == 0)
          sharpe      -- Sharpe of a +1/-1 equal-weight signal portfolio
          complexity  -- number of nodes (used in penalty)
          combined    -- weighted combination (primary selection criterion)

        Returns dict of metric -> float.
        """
        scores: Dict[str, float] = {}
        self._eval_counter += 1

        n = len(forward_returns)

        # -- evaluate expression tree
        try:
            signal = individual.tree.evaluate(data)
        except Exception:
            return self._zero_scores(individual.tree)

        if len(signal) != n:
            return self._zero_scores(individual.tree)

        # -- validity check
        valid_mask = ~(np.isnan(signal) | np.isinf(signal))
        if valid_mask.sum() < n * 0.10:
            return self._zero_scores(individual.tree)

        # clip to reasonable range to prevent numerical issues
        signal = np.clip(signal, -1e6, 1e6)

        # -- IC (overall)
        ic = _spearman_ic(signal, forward_returns)
        scores["ic"] = float(ic)

        # -- rolling IC for ICIR
        ic_series = _rolling_ic(signal, forward_returns, window=30)
        valid_ic = ic_series[np.isfinite(ic_series)]
        if len(valid_ic) >= 3:
            ic_std = float(np.std(valid_ic, ddof=1))
            icir = float(np.mean(valid_ic)) / max(ic_std, 1e-8)
        else:
            icir = 0.0
        scores["icir"] = float(icir)

        # -- regime-conditioned IC
        if regime_mask is not None and len(regime_mask) == n:
            on_mask  = (regime_mask == 1) & valid_mask
            off_mask = (regime_mask == 0) & valid_mask
            ic_on  = _spearman_ic(signal[on_mask],  forward_returns[on_mask])  if on_mask.sum()  > 4 else 0.0
            ic_off = _spearman_ic(signal[off_mask], forward_returns[off_mask]) if off_mask.sum() > 4 else 0.0
        else:
            # use bh_mass as proxy if available (BH active = large mass)
            if "bh_mass" in data and len(data["bh_mass"]) == n:
                bh = data["bh_mass"]
                median_bh = float(np.nanmedian(bh))
                on_mask  = (bh >= median_bh) & valid_mask
                off_mask = (bh <  median_bh) & valid_mask
                ic_on  = _spearman_ic(signal[on_mask],  forward_returns[on_mask])  if on_mask.sum()  > 4 else 0.0
                ic_off = _spearman_ic(signal[off_mask], forward_returns[off_mask]) if off_mask.sum() > 4 else 0.0
            else:
                ic_on, ic_off = ic, ic

        scores["ic_regime_on"]  = float(ic_on)
        scores["ic_regime_off"] = float(ic_off)
        regime_bonus = float(ic_on - ic_off)  # reward regime differentiation

        # -- Sharpe of signal portfolio
        sharpe = self._signal_sharpe(signal, forward_returns, valid_mask)
        scores["sharpe"] = float(sharpe)

        # -- complexity
        n_nodes = individual.tree.node_count()
        scores["complexity"] = float(n_nodes)
        complexity_penalty = 0.001 * n_nodes

        # -- combined fitness
        combined = (
            0.40 * ic
            + 0.30 * icir
            + 0.20 * regime_bonus
            - complexity_penalty
        )
        scores["combined"] = float(combined)

        return scores

    @staticmethod
    def _signal_sharpe(
        signal: np.ndarray,
        forward_returns: np.ndarray,
        valid_mask: np.ndarray,
    ) -> float:
        """
        Sharpe ratio of a binary long/short portfolio based on the signal.
        Position = sign(signal), scaled to +/-1.
        """
        s = signal[valid_mask]
        r = forward_returns[valid_mask]
        if len(s) < 5:
            return 0.0
        positions = np.sign(s)
        pnl = positions * r
        mean_pnl = float(np.mean(pnl))
        std_pnl  = float(np.std(pnl, ddof=1))
        if std_pnl < 1e-12:
            return 0.0
        return float(mean_pnl / std_pnl * np.sqrt(252))

    @staticmethod
    def _zero_scores(tree: ExpressionTree) -> Dict[str, float]:
        return {
            "ic": 0.0,
            "icir": 0.0,
            "ic_regime_on": 0.0,
            "ic_regime_off": 0.0,
            "sharpe": 0.0,
            "complexity": float(tree.node_count()),
            "combined": -999.0,
        }

    def evaluate_population(
        self,
        population: List[Individual],
        data: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        regime_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Evaluate all individuals in-place."""
        for ind in population:
            if not ind.fitness_scores:  # skip already-evaluated
                ind.fitness_scores = self.evaluate_fitness(
                    ind, data, forward_returns, regime_mask
                )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_parents(
        self,
        population: List[Individual],
        config: GPConfig,
    ) -> Tuple[ExpressionTree, ExpressionTree]:
        """
        Hybrid tournament + lexicase parent selection.

        With probability lexicase_prob: lexicase on (IC, ICIR, regime_bonus).
        Otherwise: tournament.

        Returns two parent trees.
        """
        trees    = [ind.tree for ind in population]
        combined = [ind.combined_fitness for ind in population]

        if random.random() < config.lexicase_prob:
            # build fitnesses_matrix: columns = IC, ICIR, regime differentiation
            ic_col     = np.array([ind.fitness_scores.get("ic",   0.0) for ind in population])
            icir_col   = np.array([ind.fitness_scores.get("icir", 0.0) for ind in population])
            regime_col = np.array([
                ind.fitness_scores.get("ic_regime_on",  0.0)
                - ind.fitness_scores.get("ic_regime_off", 0.0)
                for ind in population
            ])
            matrix = np.column_stack([ic_col, icir_col, regime_col])
            parent1 = lexicase_select(trees, matrix)
            parent2 = lexicase_select(trees, matrix)
        else:
            parent1 = tournament_select(trees, combined, k=config.tournament_k)
            parent2 = tournament_select(trees, combined, k=config.tournament_k)

        return parent1, parent2

    # ------------------------------------------------------------------
    # Generation step
    # ------------------------------------------------------------------

    def evolve_generation(
        self,
        population: List[Individual],
        data: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        config: GPConfig,
        generation: int,
        regime_mask: Optional[np.ndarray] = None,
    ) -> List[Individual]:
        """
        Produce the next generation from the current population.

        Steps:
          1. Sort by combined fitness descending.
          2. Copy top elite_n directly.
          3. Fill remainder with offspring from crossover / mutation.
          4. Evaluate new offspring.
          5. Apply automatic bloat control (hoist if depth > max_depth).
        """
        # sort
        population.sort(key=lambda ind: ind.combined_fitness, reverse=True)

        new_population: List[Individual] = []

        # -- elitism
        for ind in population[: config.elite_n]:
            elite_copy = Individual(
                tree=ind.tree.copy(),
                fitness_scores=copy.deepcopy(ind.fitness_scores),
                generation=generation,
                parent_ids=[ind.id],
            )
            new_population.append(elite_copy)

        # -- fill rest with offspring
        while len(new_population) < config.population_size:
            parent1, parent2 = self.select_parents(population, config)
            r = random.random()

            if r < config.crossover_rate:
                child1_tree, child2_tree = subtree_crossover(
                    parent1, parent2, max_depth=config.max_depth
                )
                children_trees = [child1_tree, child2_tree]
            elif r < config.crossover_rate + config.mutation_rate:
                child_tree = apply_random_mutation(
                    parent1,
                    max_tree_depth=config.max_depth,
                    generator=self.generator,
                )
                children_trees = [child_tree]
            else:
                # reproduction (clone)
                children_trees = [parent1.copy()]

            for ct in children_trees:
                if len(new_population) >= config.population_size:
                    break
                # bloat control
                if ct.depth() > config.max_depth:
                    ct = hoist_mutation(ct)
                new_ind = Individual(tree=ct, generation=generation)
                new_population.append(new_ind)

        # -- evaluate new individuals (skips already-scored elites)
        self.evaluate_population(
            new_population, data, forward_returns, regime_mask
        )
        return new_population

    # ------------------------------------------------------------------
    # Full evolution run
    # ------------------------------------------------------------------

    def run(
        self,
        data: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        config: Optional[GPConfig] = None,
        regime_mask: Optional[np.ndarray] = None,
    ) -> List[Individual]:
        """
        Full GP evolution.

        Returns the Pareto front of non-dominated individuals ranked by
        (IC, ICIR) objectives. At minimum returns the top-elite_n by combined
        fitness even if Pareto computation is degenerate.

        Parameters
        ----------
        data : dict of terminal_name -> 1D numpy array (equal length)
        forward_returns : 1D numpy array of forward bar returns
        config : GPConfig (defaults created if None)
        regime_mask : optional 1D array of 0/1 regime indicator

        Returns
        -------
        List[Individual] sorted by combined fitness descending.
        """
        if config is None:
            config = GPConfig()

        if config.verbosity >= 1:
            print(f"[GP] Initializing population: {config.population_size} individuals")

        population = self.initialize_population(config)
        self.evaluate_population(population, data, forward_returns, regime_mask)

        best_combined = max(ind.combined_fitness for ind in population)
        stagnation_counter = 0
        t0 = time.time()

        for gen in range(1, config.max_generations + 1):
            population = self.evolve_generation(
                population, data, forward_returns, config, gen, regime_mask
            )
            gen_best = max(ind.combined_fitness for ind in population)
            gen_mean = float(np.mean([ind.combined_fitness for ind in population]))

            if config.verbosity >= 1:
                elapsed = time.time() - t0
                print(
                    f"[GP] Gen {gen:4d}/{config.max_generations} | "
                    f"best={gen_best:.4f} | mean={gen_mean:.4f} | "
                    f"evals={self._eval_counter} | t={elapsed:.1f}s"
                )

            # early stopping
            if gen_best > best_combined + config.stagnation_delta:
                best_combined = gen_best
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            if stagnation_counter >= config.stagnation_patience:
                if config.verbosity >= 1:
                    print(f"[GP] Early stop -- no improvement for {config.stagnation_patience} generations.")
                break

        # -- extract Pareto front (IC vs ICIR)
        pareto = self._extract_pareto_front(population)
        pareto.sort(key=lambda ind: ind.combined_fitness, reverse=True)

        if config.verbosity >= 1:
            print(f"[GP] Done. Pareto front size: {len(pareto)}")
        return pareto

    @staticmethod
    def _extract_pareto_front(population: List[Individual]) -> List[Individual]:
        """
        Extract non-dominated individuals (Pareto front) by (IC, ICIR).
        Dominated = another individual has both higher IC AND higher ICIR.
        """
        front: List[Individual] = []
        for candidate in population:
            cic   = candidate.fitness_scores.get("ic",   0.0)
            cicir = candidate.fitness_scores.get("icir", 0.0)
            dominated = False
            for other in population:
                if other is candidate:
                    continue
                oic   = other.fitness_scores.get("ic",   0.0)
                oicir = other.fitness_scores.get("icir", 0.0)
                if oic >= cic and oicir >= cicir and (oic > cic or oicir > cicir):
                    dominated = True
                    break
            if not dominated:
                front.append(candidate)
        if not front:
            # fallback: return top by combined
            return sorted(population, key=lambda i: i.combined_fitness, reverse=True)[:10]
        return front

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_best(self, individual: Individual) -> Dict[str, Any]:
        """
        Export an individual for production use.

        Returns a dict with:
          expression    -- human-readable infix string
          fitness_scores -- dict of all fitness metrics
          generation    -- generation at which this individual was created
          node_count    -- tree complexity
          python_lambda -- string representation of a Python lambda for live use
                           (evaluates the signal given a data dict)
        """
        tree = individual.tree
        expr = tree.to_string()

        # build a python lambda string (evaluated via eval in live system)
        lambda_str = f"lambda data: tree.evaluate(data)"

        return {
            "expression":     expr,
            "fitness_scores": dict(individual.fitness_scores),
            "generation":     individual.generation,
            "node_count":     tree.node_count(),
            "depth":          tree.depth(),
            "python_lambda":  lambda_str,
            "tree_repr":      repr(tree),
        }
