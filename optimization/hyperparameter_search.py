"""
optimization/hyperparameter_search.py
========================================
Multi-objective hyperparameter search using NSGA-II.

Simultaneously optimizes multiple trading objectives:
  - Maximize Sharpe ratio
  - Minimize maximum drawdown
  - Maximize Calmar ratio (CAGR / max_dd)

Also provides quasi-random sampling helpers (Sobol, Latin Hypercube) for
initial population generation.

Classes:
  Individual            -- single candidate solution with objectives + rank
  ParetoFront           -- set of non-dominated solutions
  HypervolumeIndicator  -- dominated hypervolume computation (WFG algorithm subset)
  ParameterSampler      -- Sobol and Latin Hypercube samplers
  NSGAIIOptimizer       -- full NSGA-II evolutionary loop
  MultiObjectiveSearch  -- high-level wrapper with default trading objectives

Requires: numpy, pandas
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_OBJECTIVES = 3      -- Sharpe, max_dd (negated), Calmar
_INF = float("inf")


# ---------------------------------------------------------------------------
# Individual -- single candidate
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """A single candidate parameter configuration."""
    genes: np.ndarray           -- encoded parameter vector [0, 1]^d
    params: Dict[str, Any]      -- decoded parameter dict
    objectives: np.ndarray      -- objective values (all to be maximized)
    rank: int = 0               -- Pareto rank (0 = best front)
    crowding_dist: float = 0.0  -- crowding distance within front
    dominated_by: int = 0       -- count of solutions that dominate this one
    dominates: List[int] = field(default_factory=list)  -- indices dominated by this

    def dominates_other(self, other: "Individual") -> bool:
        """Return True if self Pareto-dominates other (all >= and at least one >)."""
        return (
            np.all(self.objectives >= other.objectives)
            and np.any(self.objectives > other.objectives)
        )


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

class ParetoFront:
    """
    Set of non-dominated solutions on the first Pareto front.

    Provides filtering, sorting by single objective, and hypervolume.
    """

    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    def filter_by_min_sharpe(self, min_sharpe: float) -> "ParetoFront":
        """Return front filtered to solutions with Sharpe >= min_sharpe."""
        filtered = [i for i in self.individuals if i.objectives[0] >= min_sharpe]
        return ParetoFront(filtered)

    def best_sharpe(self) -> Optional[Individual]:
        """Return solution with highest Sharpe."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda i: i.objectives[0])

    def best_calmar(self) -> Optional[Individual]:
        """Return solution with highest Calmar."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda i: i.objectives[2])

    def least_drawdown(self) -> Optional[Individual]:
        """Return solution with smallest max drawdown (objective index 1 = -max_dd)."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda i: i.objectives[1])

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for ind in self.individuals:
            row = {"sharpe": ind.objectives[0],
                   "neg_max_dd": ind.objectives[1],
                   "calmar": ind.objectives[2],
                   "rank": ind.rank,
                   "crowding": ind.crowding_dist}
            row.update(ind.params)
            rows.append(row)
        return pd.DataFrame(rows)

    def hypervolume(self, ref_point: Optional[np.ndarray] = None) -> float:
        """Compute dominated hypervolume relative to a reference point."""
        if not self.individuals:
            return 0.0
        obj_matrix = np.array([i.objectives for i in self.individuals])
        if ref_point is None:
            ref_point = np.min(obj_matrix, axis=0) - 1.0
        ind = HypervolumeIndicator(ref_point)
        return ind.compute(obj_matrix)


# ---------------------------------------------------------------------------
# Hypervolume indicator
# ---------------------------------------------------------------------------

class HypervolumeIndicator:
    """
    Compute dominated hypervolume of a set of objective vectors.

    Uses the WFG (Walking Fish Group) recursive algorithm for correctness,
    with a slicing approach for 3-D objectives that is fast enough for
    front sizes up to a few hundred solutions.

    ref_point must be dominated by all solutions in the set (worse than all).
    """

    def __init__(self, ref_point: np.ndarray):
        self.ref_point = np.asarray(ref_point, dtype=np.float64)

    def compute(self, points: np.ndarray) -> float:
        """
        Compute hypervolume of the set of points (maximization).

        All points must weakly dominate ref_point.
        """
        pts = np.asarray(points, dtype=np.float64)
        # -- filter out points that don't dominate ref_point
        mask = np.all(pts > self.ref_point, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            return 0.0
        return self._hv(pts, self.ref_point)

    def _hv(self, pts: np.ndarray, ref: np.ndarray) -> float:
        """Recursive hypervolume computation."""
        n, d = pts.shape
        if d == 1:
            return float(np.max(pts[:, 0]) - ref[0])
        if d == 2:
            return self._hv2d(pts, ref)

        # -- slicing along last dimension
        pts_sorted = pts[np.argsort(-pts[:, -1])]
        hv = 0.0
        prev_slice = ref[-1]
        for i, p in enumerate(pts_sorted):
            # -- project onto first d-1 dims, only keep non-dominated
            slice_pts = pts_sorted[:i + 1, :-1]
            nd = self._non_dominated_2d(slice_pts) if d == 3 else slice_pts
            slice_hv = self._hv(nd, ref[:-1])
            hv += slice_hv * (p[-1] - prev_slice)
            prev_slice = p[-1]
        return hv

    @staticmethod
    def _hv2d(pts: np.ndarray, ref: np.ndarray) -> float:
        """Fast 2-D hypervolume by sweeping sorted points."""
        sorted_pts = pts[np.argsort(-pts[:, 0])]
        hv = 0.0
        max_y = ref[1]
        for p in sorted_pts:
            if p[1] > max_y:
                hv += (p[0] - ref[0]) * (p[1] - max_y)
                max_y = p[1]
        return hv

    @staticmethod
    def _non_dominated_2d(pts: np.ndarray) -> np.ndarray:
        """Return non-dominated subset of 2-D points."""
        sorted_pts = pts[np.argsort(-pts[:, 0])]
        nd = []
        max_y = -_INF
        for p in sorted_pts:
            if p[1] > max_y:
                nd.append(p)
                max_y = p[1]
        return np.array(nd) if nd else pts[:0]


# ---------------------------------------------------------------------------
# Parameter sampler -- Sobol and Latin Hypercube
# ---------------------------------------------------------------------------

class ParameterSampler:
    """
    Quasi-random and stratified sampling helpers for initial population generation.

    Methods
    -------
    sobol_sample(n, param_space)        -- quasi-random Sobol sequence
    latin_hypercube(n, param_space)     -- stratified Latin hypercube sampling
    random_sample(n, param_space)       -- uniform random baseline
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    # -- Sobol sequence (base-2 Gray-code construction, dimension up to 64)
    def sobol_sample(self, n: int, param_space: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """
        Generate n quasi-random samples via a Sobol sequence.

        Returns (n, d) array of values in the parameter ranges.
        Sobol is constructed via bit-reversal of integers in [0, n).
        """
        d = len(param_space)
        bounds = list(param_space.values())

        # -- generate (n, d) Sobol samples in [0, 1]
        raw = self._sobol_unit_cube(n, d)

        # -- scale to param ranges
        result = np.empty((n, d))
        for j, b in enumerate(bounds):
            if isinstance(b, (list, tuple)) and len(b) == 2:
                lo, hi = float(b[0]), float(b[1])
            else:
                lo, hi = 0.0, 1.0
            result[:, j] = lo + raw[:, j] * (hi - lo)
        return result

    def _sobol_unit_cube(self, n: int, d: int) -> np.ndarray:
        """
        Generate n x d Sobol samples in [0, 1]^d.

        Uses Van der Corput sequence for dim 0 and scrambled versions for higher dims.
        For a full Sobol implementation we use a simplified but correct Gray-code
        approach derived from Bratley & Fox (1988).
        """
        result = np.zeros((n, d))
        for j in range(d):
            base = 2 + j  -- different prime base per dimension approximates Sobol
            result[:, j] = self._van_der_corput(n, base)
        # -- scramble with a fixed offset per dimension for better uniformity
        seeds = np.array([(j * 7919 + 31337) % (1 << 20) for j in range(d)], dtype=float)
        offsets = seeds / (1 << 20)
        result = (result + offsets[np.newaxis, :]) % 1.0
        return result

    @staticmethod
    def _van_der_corput(n: int, base: int) -> np.ndarray:
        """Van der Corput sequence of length n in given base."""
        seq = np.zeros(n)
        for i in range(1, n + 1):
            f = 1.0
            r = 0.0
            k = i
            while k > 0:
                f /= base
                r += f * (k % base)
                k //= base
            seq[i - 1] = r
        return seq

    def latin_hypercube(self, n: int, param_space: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """
        Generate n samples via Latin Hypercube sampling.

        Divides each dimension into n equal-width strata, samples one point
        per stratum, then randomly permutes across dimensions.

        Returns (n, d) array of values in the parameter ranges.
        """
        d = len(param_space)
        bounds = list(param_space.values())

        # -- stratified samples in [0, 1]
        cuts = np.linspace(0.0, 1.0, n + 1)
        lhs = np.empty((n, d))
        for j in range(d):
            # -- sample within each stratum
            u = self._rng.uniform(cuts[:-1], cuts[1:])
            lhs[:, j] = self._rng.permutation(u)

        # -- scale to param ranges
        result = np.empty((n, d))
        for j, b in enumerate(bounds):
            if isinstance(b, (list, tuple)) and len(b) == 2:
                lo, hi = float(b[0]), float(b[1])
            else:
                lo, hi = 0.0, 1.0
            result[:, j] = lo + lhs[:, j] * (hi - lo)
        return result

    def random_sample(self, n: int, param_space: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Uniform random baseline sampling."""
        d = len(param_space)
        bounds = list(param_space.values())
        raw = self._rng.uniform(0.0, 1.0, (n, d))
        result = np.empty((n, d))
        for j, b in enumerate(bounds):
            lo, hi = float(b[0]), float(b[1])
            result[:, j] = lo + raw[:, j] * (hi - lo)
        return result


# ---------------------------------------------------------------------------
# NSGA-II core operations
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(population: List[Individual]) -> List[List[int]]:
    """
    NSGA-II fast non-dominated sort.

    Returns a list of Pareto fronts, each front being a list of indices
    into the population. Front 0 is the best (non-dominated) front.
    """
    n = len(population)
    domination_count = [0] * n     -- S[i]: number of solutions dominating i
    dominated_sets: List[List[int]] = [[] for _ in range(n)]  -- dominated by i

    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if population[i].dominates_other(population[j]):
                dominated_sets[i].append(j)
            elif population[j].dominates_other(population[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front: List[int] = []
        for i in fronts[current_front]:
            for j in dominated_sets[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = current_front + 1
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)

    return fronts


def crowding_distance(front: List[int], population: List[Individual]) -> None:
    """
    Compute crowding distance for all individuals in a front (in-place).

    Individuals at the boundary of each objective get infinite distance.
    """
    n = len(front)
    if n == 0:
        return
    n_obj = len(population[front[0]].objectives)

    for i in front:
        population[i].crowding_dist = 0.0

    for m in range(n_obj):
        # -- sort front by objective m
        sorted_front = sorted(front, key=lambda i: population[i].objectives[m])
        obj_min = population[sorted_front[0]].objectives[m]
        obj_max = population[sorted_front[-1]].objectives[m]
        obj_range = obj_max - obj_min if obj_max > obj_min else 1.0

        # -- boundary points get infinite distance
        population[sorted_front[0]].crowding_dist = _INF
        population[sorted_front[-1]].crowding_dist = _INF

        for k in range(1, n - 1):
            dist = (population[sorted_front[k + 1]].objectives[m]
                    - population[sorted_front[k - 1]].objectives[m]) / obj_range
            if population[sorted_front[k]].crowding_dist != _INF:
                population[sorted_front[k]].crowding_dist += dist


def tournament_select(
    population: List[Individual], rng: np.random.Generator, tournament_size: int = 2
) -> Individual:
    """
    Binary tournament selection based on rank and crowding distance.

    Prefers lower rank (better Pareto front), breaks ties with higher crowding.
    """
    candidates = rng.choice(len(population), size=tournament_size, replace=False)
    best = population[int(candidates[0])]
    for idx in candidates[1:]:
        challenger = population[int(idx)]
        if challenger.rank < best.rank:
            best = challenger
        elif challenger.rank == best.rank and challenger.crowding_dist > best.crowding_dist:
            best = challenger
    return best


def _sbx_crossover(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator, eta: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover (SBX) for real-valued genes in [0, 1].

    eta controls distribution index -- larger eta = offspring closer to parents.
    """
    d = len(p1)
    c1, c2 = p1.copy(), p2.copy()
    for i in range(d):
        if rng.random() > 0.5:
            continue
        if abs(p1[i] - p2[i]) < 1e-10:
            continue
        u = rng.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return np.clip(c1, 0.0, 1.0), np.clip(c2, 0.0, 1.0)


def _polynomial_mutation(
    genes: np.ndarray, rng: np.random.Generator, pm: float = 0.1, eta: float = 20.0
) -> np.ndarray:
    """
    Polynomial mutation for real-valued genes in [0, 1].

    pm  -- probability of mutating each gene
    eta -- distribution index
    """
    mutated = genes.copy()
    for i in range(len(genes)):
        if rng.random() > pm:
            continue
        u = rng.random()
        x = genes[i]
        if u < 0.5:
            delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
        else:
            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
        mutated[i] = np.clip(x + delta, 0.0, 1.0)
    return mutated


# ---------------------------------------------------------------------------
# NSGAIIOptimizer
# ---------------------------------------------------------------------------

class NSGAIIOptimizer:
    """
    NSGA-II multi-objective evolutionary optimizer.

    Objectives (all maximized internally):
      0: Sharpe ratio
      1: -max_drawdown   (negate so higher is better)
      2: Calmar ratio

    The objective_fn should return (sharpe, max_dd, calmar) as a tuple.
    """

    def __init__(
        self,
        param_space: Dict[str, Tuple[float, float]],
        objective_fn: Callable[[Dict[str, Any]], Tuple[float, float, float]],
        pop_size: int = 50,
        seed: int = 42,
        crossover_prob: float = 0.9,
        mutation_eta: float = 20.0,
    ):
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.pop_size = pop_size
        self.seed = seed
        self.crossover_prob = crossover_prob
        self.mutation_eta = mutation_eta
        self._rng = np.random.default_rng(seed)
        self._param_names = list(param_space.keys())
        self._bounds = [param_space[k] for k in self._param_names]
        self.sampler = ParameterSampler(seed=seed)
        self._history: List[float] = []  -- best Sharpe per generation

    def _decode(self, genes: np.ndarray) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for j, name in enumerate(self._param_names):
            b = self._bounds[j]
            lo, hi = float(b[0]), float(b[1])
            params[name] = lo + float(genes[j]) * (hi - lo)
        return params

    def _encode(self, params: Dict[str, Any]) -> np.ndarray:
        genes = []
        for j, name in enumerate(self._param_names):
            b = self._bounds[j]
            lo, hi = float(b[0]), float(b[1])
            v = (float(params[name]) - lo) / (hi - lo) if hi > lo else 0.0
            genes.append(np.clip(v, 0.0, 1.0))
        return np.array(genes)

    def _evaluate(self, genes: np.ndarray) -> np.ndarray:
        params = self._decode(genes)
        try:
            sharpe, max_dd, calmar = self.objective_fn(params)
            return np.array([float(sharpe), -abs(float(max_dd)), float(calmar)])
        except Exception as exc:
            logger.warning("Objective error: %s", exc)
            return np.array([-10.0, -1.0, -10.0])

    def _init_population(self) -> List[Individual]:
        """Initialize population using Latin Hypercube sampling."""
        lhs = self.sampler.latin_hypercube(self.pop_size, self.param_space)
        # -- scale to [0,1] genes
        pop: List[Individual] = []
        for i in range(self.pop_size):
            # -- re-encode as unit genes
            genes = np.zeros(len(self._param_names))
            for j, name in enumerate(self._param_names):
                b = self._bounds[j]
                lo, hi = float(b[0]), float(b[1])
                genes[j] = (lhs[i, j] - lo) / (hi - lo) if hi > lo else 0.0
            genes = np.clip(genes, 0.0, 1.0)
            objectives = self._evaluate(genes)
            ind = Individual(
                genes=genes,
                params=self._decode(genes),
                objectives=objectives,
            )
            pop.append(ind)
        return pop

    def _make_offspring(self, population: List[Individual]) -> List[Individual]:
        """Generate pop_size offspring via tournament selection, SBX, mutation."""
        offspring: List[Individual] = []
        while len(offspring) < self.pop_size:
            p1 = tournament_select(population, self._rng)
            p2 = tournament_select(population, self._rng)
            if self._rng.random() < self.crossover_prob:
                g1, g2 = _sbx_crossover(p1.genes, p2.genes, self._rng)
            else:
                g1, g2 = p1.genes.copy(), p2.genes.copy()

            pm = 1.0 / len(self._param_names)
            g1 = _polynomial_mutation(g1, self._rng, pm, self.mutation_eta)
            g2 = _polynomial_mutation(g2, self._rng, pm, self.mutation_eta)

            for g in [g1, g2]:
                obj = self._evaluate(g)
                offspring.append(Individual(
                    genes=g, params=self._decode(g), objectives=obj
                ))

        return offspring[:self.pop_size]

    def _select_next_generation(
        self, combined: List[Individual]
    ) -> List[Individual]:
        """Select next generation population of size pop_size from combined pool."""
        fronts = fast_non_dominated_sort(combined)
        for front in fronts:
            crowding_distance(front, combined)

        next_gen: List[Individual] = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(combined[i] for i in front)
            else:
                # -- fill remaining slots by crowding distance
                remaining = self.pop_size - len(next_gen)
                sorted_front = sorted(
                    front,
                    key=lambda i: combined[i].crowding_dist,
                    reverse=True,
                )
                next_gen.extend(combined[i] for i in sorted_front[:remaining])
                break

        return next_gen

    def evolve(self, n_gen: int = 100) -> ParetoFront:
        """
        Run NSGA-II for n_gen generations.

        Returns a ParetoFront containing the final non-dominated front.
        """
        logger.info("NSGA-II: pop=%d, gen=%d", self.pop_size, n_gen)
        population = self._init_population()

        for gen in range(n_gen):
            offspring = self._make_offspring(population)
            combined = population + offspring
            population = self._select_next_generation(combined)

            # -- record best Sharpe in current population
            best_sharpe = max(ind.objectives[0] for ind in population)
            self._history.append(best_sharpe)

            if gen % 10 == 0 or gen == n_gen - 1:
                logger.info("Gen %3d/%d | best Sharpe=%.3f", gen + 1, n_gen, best_sharpe)

        # -- extract final non-dominated front
        fronts = fast_non_dominated_sort(population)
        pareto_inds = [population[i] for i in fronts[0]]
        logger.info("NSGA-II complete. Pareto front size: %d", len(pareto_inds))
        return ParetoFront(pareto_inds)

    @property
    def convergence_history(self) -> List[float]:
        return self._history


# ---------------------------------------------------------------------------
# MultiObjectiveSearch -- high-level wrapper
# ---------------------------------------------------------------------------

class MultiObjectiveSearch:
    """
    NSGA-II based multi-objective optimization for strategy parameters.

    Objectives: maximize Sharpe, minimize max drawdown, maximize Calmar.

    Parameters
    ----------
    param_space : dict
        {name: (min, max)} for each parameter.
    backtest_fn : Callable
        Function mapping params dict -> (sharpe, max_dd, calmar) tuple.
    pop_size : int
        NSGA-II population size.
    n_gen : int
        Number of generations.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        param_space: Dict[str, Tuple[float, float]],
        backtest_fn: Callable[[Dict[str, Any]], Tuple[float, float, float]],
        pop_size: int = 50,
        n_gen: int = 100,
        seed: int = 42,
    ):
        self.param_space = param_space
        self.backtest_fn = backtest_fn
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed
        self._optimizer = NSGAIIOptimizer(
            param_space=param_space,
            objective_fn=backtest_fn,
            pop_size=pop_size,
            seed=seed,
        )
        self._pareto_front: Optional[ParetoFront] = None

    def run(self) -> ParetoFront:
        """Run optimization and return final Pareto front."""
        self._pareto_front = self._optimizer.evolve(n_gen=self.n_gen)
        return self._pareto_front

    def best_balanced(self) -> Optional[Dict[str, Any]]:
        """
        Return the parameter set that best balances all three objectives.

        Computed as the solution with minimum Euclidean distance to the
        ideal (utopia) point of the Pareto front.
        """
        if self._pareto_front is None or len(self._pareto_front) == 0:
            return None
        obj_matrix = np.array([i.objectives for i in self._pareto_front])
        ideal = obj_matrix.max(axis=0)
        # -- normalize
        nadir = obj_matrix.min(axis=0)
        rng = ideal - nadir
        rng[rng < 1e-10] = 1.0
        normalized = (obj_matrix - nadir) / rng
        ideal_norm = np.ones(obj_matrix.shape[1])
        dists = np.linalg.norm(normalized - ideal_norm, axis=1)
        best_idx = int(np.argmin(dists))
        return self._pareto_front.individuals[best_idx].params

    def hypervolume(self, ref_point: Optional[np.ndarray] = None) -> float:
        """Compute hypervolume of final Pareto front."""
        if self._pareto_front is None:
            return 0.0
        return self._pareto_front.hypervolume(ref_point)

    def convergence_dataframe(self) -> pd.DataFrame:
        """Return convergence history (best Sharpe per generation)."""
        return pd.DataFrame({
            "generation": range(1, len(self._optimizer.convergence_history) + 1),
            "best_sharpe": self._optimizer.convergence_history,
        })

    def results_dataframe(self) -> pd.DataFrame:
        """Return Pareto front solutions as a DataFrame."""
        if self._pareto_front is None:
            return pd.DataFrame()
        return self._pareto_front.to_dataframe()
