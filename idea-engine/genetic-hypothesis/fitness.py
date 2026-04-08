"""
Multi-objective fitness evaluation for genetic hypothesis evolution.

Capabilities:
  - FitnessFunction dataclass with standard trading metrics
  - Pareto front computation (non-dominated sorting)
  - NSGA-II: fast non-dominated sort + crowding distance
  - Weighted fitness aggregation with regime-specific weights
  - Backtest-based fitness evaluation
  - Robustness fitness via bootstrap resampling
  - Penalized fitness (overfitting, turnover, parameter sensitivity)
  - Fitness landscape analysis (ruggedness, autocorrelation)
  - Reference-point hypervolume indicator
  - Tournament selection using Pareto rank + crowding
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Fitness value representation
# ---------------------------------------------------------------------------

@dataclass
class FitnessFunction:
    """Multi-objective fitness vector for a trading hypothesis."""
    sharpe: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0              # as positive fraction, e.g. 0.15
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0        # days
    total_return: float = 0.0
    n_trades: int = 0
    sortino: float = 0.0
    tail_ratio: float = 0.0               # P95 / |P5|

    def to_dict(self) -> Dict[str, float]:
        return {
            "sharpe": self.sharpe, "calmar": self.calmar,
            "max_drawdown": self.max_drawdown, "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_duration": self.avg_trade_duration,
            "total_return": self.total_return, "n_trades": self.n_trades,
            "sortino": self.sortino, "tail_ratio": self.tail_ratio,
        }

    def objective_vector(self, objectives: Optional[List[str]] = None) -> np.ndarray:
        """Extract an objective vector (default: sharpe, calmar, win_rate)."""
        objs = objectives or ["sharpe", "calmar", "win_rate"]
        d = self.to_dict()
        return np.array([d[o] for o in objs])


# ---------------------------------------------------------------------------
# Dominance and Pareto front
# ---------------------------------------------------------------------------

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a dominates b (all objectives >= and at least one >)."""
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_sort(fitness_vectors: List[np.ndarray]) -> List[List[int]]:
    """
    Fast non-dominated sort (NSGA-II style).

    Returns list of fronts, where each front is a list of indices.
    Front 0 = Pareto front, Front 1 = next layer, etc.
    """
    n = len(fitness_vectors)
    domination_count = np.zeros(n, dtype=int)
    dominated_set: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(fitness_vectors[p], fitness_vectors[q]):
                dominated_set[p].append(q)
            elif dominates(fitness_vectors[q], fitness_vectors[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    current_front = 0
    while fronts[current_front]:
        next_front: List[int] = []
        for p in fronts[current_front]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        current_front += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


def pareto_front_indices(fitness_vectors: List[np.ndarray]) -> List[int]:
    """Return indices of non-dominated (Pareto optimal) solutions."""
    fronts = non_dominated_sort(fitness_vectors)
    return fronts[0] if fronts else []


def pareto_ranks(fitness_vectors: List[np.ndarray]) -> List[int]:
    """Assign Pareto rank to each individual (0 = best front)."""
    fronts = non_dominated_sort(fitness_vectors)
    n = len(fitness_vectors)
    ranks = [0] * n
    for rank, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank
    return ranks


# ---------------------------------------------------------------------------
# Crowding distance (used in NSGA-II)
# ---------------------------------------------------------------------------

def crowding_distance(fitness_vectors: List[np.ndarray]) -> np.ndarray:
    """Compute NSGA-II crowding distance for a set of solutions."""
    n = len(fitness_vectors)
    if n <= 2:
        return np.full(n, float("inf"))

    n_obj = fitness_vectors[0].shape[0]
    distances = np.zeros(n)

    for m in range(n_obj):
        values = np.array([fv[m] for fv in fitness_vectors])
        sorted_idx = np.argsort(values)
        obj_min = values[sorted_idx[0]]
        obj_max = values[sorted_idx[-1]]
        spread = obj_max - obj_min

        distances[sorted_idx[0]] = float("inf")
        distances[sorted_idx[-1]] = float("inf")

        if spread < 1e-12:
            continue

        for i in range(1, n - 1):
            idx = sorted_idx[i]
            prev_val = values[sorted_idx[i - 1]]
            next_val = values[sorted_idx[i + 1]]
            distances[idx] += (next_val - prev_val) / spread

    return distances


# ---------------------------------------------------------------------------
# NSGA-II selection
# ---------------------------------------------------------------------------

class NSGA2Selector:
    """NSGA-II selection: non-dominated sort + crowding distance."""

    def __init__(self, objectives: List[str], seed: int = 42):
        self.objectives = objectives
        self.rng = np.random.default_rng(seed)

    def select(self, fitness_list: List[FitnessFunction],
               n_select: int) -> List[int]:
        """Select n_select individuals using NSGA-II."""
        vectors = [f.objective_vector(self.objectives) for f in fitness_list]
        fronts = non_dominated_sort(vectors)
        selected: List[int] = []

        for front in fronts:
            if len(selected) + len(front) <= n_select:
                selected.extend(front)
            else:
                # Need partial front — use crowding distance
                front_vectors = [vectors[i] for i in front]
                cd = crowding_distance(front_vectors)
                sorted_by_cd = np.argsort(-cd)
                remaining = n_select - len(selected)
                for k in range(remaining):
                    selected.append(front[sorted_by_cd[k]])
                break

        return selected

    def tournament(self, fitness_list: List[FitnessFunction],
                   tournament_size: int = 2) -> int:
        """Binary tournament using Pareto rank + crowding."""
        vectors = [f.objective_vector(self.objectives) for f in fitness_list]
        ranks = pareto_ranks(vectors)
        cd = crowding_distance(vectors)
        n = len(fitness_list)

        candidates = self.rng.choice(n, size=min(tournament_size, n),
                                     replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best] and cd[c] > cd[best]:
                best = c
        return int(best)


# ---------------------------------------------------------------------------
# Weighted fitness aggregation
# ---------------------------------------------------------------------------

@dataclass
class FitnessWeights:
    """Weights for scalarizing the multi-objective fitness."""
    sharpe_w: float = 0.30
    calmar_w: float = 0.20
    win_rate_w: float = 0.15
    profit_factor_w: float = 0.15
    max_drawdown_w: float = 0.10      # penalty — higher DD → lower score
    sortino_w: float = 0.10


DEFAULT_WEIGHTS = FitnessWeights()

REGIME_WEIGHTS: Dict[str, FitnessWeights] = {
    "risk_on": FitnessWeights(sharpe_w=0.25, calmar_w=0.15, win_rate_w=0.10,
                               profit_factor_w=0.20, max_drawdown_w=0.10,
                               sortino_w=0.20),
    "risk_off": FitnessWeights(sharpe_w=0.20, calmar_w=0.25, win_rate_w=0.15,
                                profit_factor_w=0.10, max_drawdown_w=0.20,
                                sortino_w=0.10),
    "crisis": FitnessWeights(sharpe_w=0.10, calmar_w=0.30, win_rate_w=0.10,
                              profit_factor_w=0.10, max_drawdown_w=0.30,
                              sortino_w=0.10),
    "recovery": FitnessWeights(sharpe_w=0.30, calmar_w=0.15, win_rate_w=0.15,
                                profit_factor_w=0.20, max_drawdown_w=0.05,
                                sortino_w=0.15),
    "low_vol_grind": FitnessWeights(sharpe_w=0.35, calmar_w=0.10,
                                     win_rate_w=0.20, profit_factor_w=0.15,
                                     max_drawdown_w=0.05, sortino_w=0.15),
}


def weighted_fitness(ff: FitnessFunction,
                     weights: Optional[FitnessWeights] = None) -> float:
    """Scalarize multi-objective fitness into single value."""
    w = weights or DEFAULT_WEIGHTS
    score = (w.sharpe_w * ff.sharpe
             + w.calmar_w * ff.calmar
             + w.win_rate_w * ff.win_rate
             + w.profit_factor_w * min(ff.profit_factor, 5.0)
             - w.max_drawdown_w * ff.max_drawdown * 10
             + w.sortino_w * ff.sortino)
    return score


def regime_weighted_fitness(ff: FitnessFunction, regime: str) -> float:
    """Compute fitness weighted for a specific market regime."""
    w = REGIME_WEIGHTS.get(regime, DEFAULT_WEIGHTS)
    return weighted_fitness(ff, w)


# ---------------------------------------------------------------------------
# Backtest-based fitness evaluation
# ---------------------------------------------------------------------------

class BacktestFitnessEvaluator:
    """
    Evaluate hypothesis fitness by simulating its signal on price data.

    The evaluator takes a signal function and price series, runs the
    backtest, and computes all fitness metrics.
    """

    def __init__(self, prices: np.ndarray, risk_free_rate: float = 0.04,
                 transaction_cost_bps: float = 5.0):
        self.prices = prices
        self.returns = np.diff(np.log(prices))
        self.risk_free_rate = risk_free_rate
        self.tc_bps = transaction_cost_bps

    def evaluate(self, signal: np.ndarray) -> FitnessFunction:
        """
        Evaluate a signal array against the price series.

        signal: array of positions [-1, 0, 1] aligned to self.returns.
        """
        n = min(len(signal), len(self.returns))
        sig = signal[:n]
        ret = self.returns[:n]

        # Transaction costs
        position_changes = np.abs(np.diff(np.concatenate([[0], sig])))
        tc = position_changes * self.tc_bps / 10_000

        strategy_returns = sig * ret - tc
        cumulative = np.cumsum(strategy_returns)
        total_return = float(np.exp(cumulative[-1]) - 1) if len(cumulative) > 0 else 0.0

        # Sharpe
        if len(strategy_returns) > 1 and np.std(strategy_returns) > 1e-12:
            daily_rf = self.risk_free_rate / 252
            sharpe = float((np.mean(strategy_returns) - daily_rf)
                           / np.std(strategy_returns) * math.sqrt(252))
        else:
            sharpe = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Calmar
        calmar = total_return / max_dd if max_dd > 1e-9 else 0.0

        # Win rate
        trades = strategy_returns[sig != 0]
        wins = np.sum(trades > 0)
        n_trades = len(trades)
        win_rate = float(wins / n_trades) if n_trades > 0 else 0.0

        # Profit factor
        gross_profit = float(np.sum(trades[trades > 0]))
        gross_loss = float(np.abs(np.sum(trades[trades < 0])))
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else 0.0

        # Sortino
        downside = strategy_returns[strategy_returns < 0]
        if len(downside) > 0:
            downside_std = float(np.std(downside))
            sortino = float((np.mean(strategy_returns) - self.risk_free_rate / 252)
                            / downside_std * math.sqrt(252)) if downside_std > 1e-12 else 0.0
        else:
            sortino = sharpe * 1.5  # no downside

        # Tail ratio
        if len(strategy_returns) > 20:
            p95 = float(np.percentile(strategy_returns, 95))
            p5 = float(np.abs(np.percentile(strategy_returns, 5)))
            tail_ratio = p95 / p5 if p5 > 1e-12 else 0.0
        else:
            tail_ratio = 0.0

        # Average trade duration (approximate)
        in_trade = sig != 0
        trade_blocks = np.diff(np.concatenate([[0], in_trade.astype(int), [0]]))
        entries = np.where(trade_blocks == 1)[0]
        exits = np.where(trade_blocks == -1)[0]
        if len(entries) > 0 and len(exits) > 0:
            durations = exits[:len(entries)] - entries[:len(exits)]
            avg_duration = float(np.mean(durations)) if len(durations) > 0 else 0.0
        else:
            avg_duration = 0.0

        return FitnessFunction(
            sharpe=sharpe, calmar=calmar, max_drawdown=max_dd,
            win_rate=win_rate, profit_factor=profit_factor,
            avg_trade_duration=avg_duration, total_return=total_return,
            n_trades=n_trades, sortino=sortino, tail_ratio=tail_ratio,
        )


# ---------------------------------------------------------------------------
# Robustness fitness — bootstrap resampling
# ---------------------------------------------------------------------------

class RobustnessFitnessEvaluator:
    """Average fitness across bootstrap samples for robustness check."""

    def __init__(self, evaluator: BacktestFitnessEvaluator,
                 n_bootstrap: int = 50, block_size: int = 20,
                 seed: int = 42):
        self.evaluator = evaluator
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)

    def evaluate(self, signal: np.ndarray) -> Tuple[FitnessFunction, float]:
        """
        Evaluate fitness across bootstrapped return series.

        Returns (mean_fitness, std_of_sharpe_across_samples).
        """
        n = len(self.evaluator.returns)
        sharpe_samples: List[float] = []
        all_ff: List[FitnessFunction] = []

        for _ in range(self.n_bootstrap):
            # Block bootstrap
            n_blocks = math.ceil(n / self.block_size)
            block_starts = self.rng.integers(0, n - self.block_size + 1,
                                             size=n_blocks)
            bootstrap_idx = np.concatenate([
                np.arange(s, min(s + self.block_size, n)) for s in block_starts
            ])[:n]

            # Reorder returns and signal
            boot_returns = self.evaluator.returns[bootstrap_idx]
            boot_signal = signal[bootstrap_idx] if len(signal) >= n else signal

            # Create temporary evaluator with bootstrapped data
            temp_prices = np.exp(np.concatenate([[0], np.cumsum(boot_returns)]))
            temp_eval = BacktestFitnessEvaluator(
                temp_prices, self.evaluator.risk_free_rate, self.evaluator.tc_bps)
            ff = temp_eval.evaluate(boot_signal)
            all_ff.append(ff)
            sharpe_samples.append(ff.sharpe)

        # Average across bootstrap samples
        mean_ff = FitnessFunction(
            sharpe=float(np.mean([f.sharpe for f in all_ff])),
            calmar=float(np.mean([f.calmar for f in all_ff])),
            max_drawdown=float(np.mean([f.max_drawdown for f in all_ff])),
            win_rate=float(np.mean([f.win_rate for f in all_ff])),
            profit_factor=float(np.mean([f.profit_factor for f in all_ff])),
            avg_trade_duration=float(np.mean([f.avg_trade_duration for f in all_ff])),
            total_return=float(np.mean([f.total_return for f in all_ff])),
            n_trades=int(np.mean([f.n_trades for f in all_ff])),
            sortino=float(np.mean([f.sortino for f in all_ff])),
            tail_ratio=float(np.mean([f.tail_ratio for f in all_ff])),
        )
        sharpe_std = float(np.std(sharpe_samples))
        return mean_ff, sharpe_std


# ---------------------------------------------------------------------------
# Penalized fitness
# ---------------------------------------------------------------------------

class PenalizedFitness:
    """
    Subtract penalties for overfitting, high turnover, and parameter sensitivity.
    """

    def __init__(self, overfit_penalty_weight: float = 0.5,
                 turnover_penalty_weight: float = 0.1,
                 sensitivity_penalty_weight: float = 0.3):
        self.overfit_w = overfit_penalty_weight
        self.turnover_w = turnover_penalty_weight
        self.sensitivity_w = sensitivity_penalty_weight

    def compute(self, is_fitness: FitnessFunction, oos_fitness: FitnessFunction,
                turnover: float, sensitivity: float) -> float:
        """
        Compute penalized scalar fitness.

        is_fitness: in-sample fitness
        oos_fitness: out-of-sample fitness
        turnover: daily turnover as fraction
        sensitivity: parameter sensitivity (higher = less robust)
        """
        base = weighted_fitness(oos_fitness)

        # Overfitting penalty: IS-OOS gap
        is_score = weighted_fitness(is_fitness)
        oos_score = base
        overfit_gap = max(is_score - oos_score, 0)
        overfit_penalty = self.overfit_w * overfit_gap

        # Turnover penalty
        turnover_penalty = self.turnover_w * turnover * 100

        # Sensitivity penalty
        sensitivity_penalty = self.sensitivity_w * sensitivity

        penalized = base - overfit_penalty - turnover_penalty - sensitivity_penalty
        return penalized

    def decompose(self, is_fitness: FitnessFunction, oos_fitness: FitnessFunction,
                  turnover: float, sensitivity: float) -> Dict[str, float]:
        """Return breakdown of penalties."""
        base = weighted_fitness(oos_fitness)
        is_score = weighted_fitness(is_fitness)
        overfit_gap = max(is_score - base, 0)

        return {
            "base_score": base,
            "overfit_penalty": self.overfit_w * overfit_gap,
            "turnover_penalty": self.turnover_w * turnover * 100,
            "sensitivity_penalty": self.sensitivity_w * sensitivity,
            "final_score": self.compute(is_fitness, oos_fitness, turnover, sensitivity),
        }


# ---------------------------------------------------------------------------
# Fitness landscape analysis
# ---------------------------------------------------------------------------

class FitnessLandscape:
    """Analyze properties of the fitness landscape."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def ruggedness(self, fitness_sequence: List[float]) -> float:
        """
        Compute ruggedness of the fitness landscape.

        Ruggedness = number of sign changes in consecutive fitness differences.
        Higher = more rugged = harder to optimize.
        """
        if len(fitness_sequence) < 3:
            return 0.0
        diffs = np.diff(fitness_sequence)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        return float(sign_changes / (len(diffs) - 1))

    def autocorrelation(self, fitness_sequence: List[float],
                        lag: int = 1) -> float:
        """
        Autocorrelation of fitness values at given lag.

        High autocorrelation → smooth landscape → easy to optimize.
        """
        f = np.array(fitness_sequence)
        if len(f) < lag + 2:
            return 0.0
        mean = np.mean(f)
        var = np.var(f)
        if var < 1e-12:
            return 0.0
        autocov = np.mean((f[:-lag] - mean) * (f[lag:] - mean))
        return float(autocov / var)

    def correlation_length(self, fitness_sequence: List[float],
                           max_lag: int = 20) -> float:
        """
        Correlation length: lag at which autocorrelation drops below 1/e.

        Longer correlation length → smoother landscape.
        """
        threshold = 1.0 / math.e
        for lag in range(1, min(max_lag, len(fitness_sequence) // 2)):
            ac = self.autocorrelation(fitness_sequence, lag)
            if ac < threshold:
                return float(lag)
        return float(max_lag)

    def fitness_distance_correlation(self, fitness_values: List[float],
                                     genotype_distances: List[float]) -> float:
        """
        Correlation between fitness and distance to optimum.

        Strong negative correlation → easy to optimize (gradient points to optimum).
        """
        f = np.array(fitness_values)
        d = np.array(genotype_distances)
        if len(f) < 3 or np.std(f) < 1e-12 or np.std(d) < 1e-12:
            return 0.0
        return float(np.corrcoef(f, d)[0, 1])

    def neutrality(self, fitness_sequence: List[float],
                   tolerance: float = 0.001) -> float:
        """Fraction of consecutive pairs with near-identical fitness."""
        if len(fitness_sequence) < 2:
            return 0.0
        diffs = np.abs(np.diff(fitness_sequence))
        neutral = np.sum(diffs < tolerance)
        return float(neutral / len(diffs))


# ---------------------------------------------------------------------------
# Hypervolume indicator
# ---------------------------------------------------------------------------

class HypervolumeIndicator:
    """
    Reference-point hypervolume indicator for comparing Pareto fronts.

    Uses a simple 2D/3D exact algorithm for small objective counts
    and a Monte Carlo approximation for higher dimensions.
    """

    def __init__(self, reference_point: np.ndarray):
        self.reference_point = reference_point

    def compute_2d(self, front: List[np.ndarray]) -> float:
        """Exact 2D hypervolume."""
        if not front:
            return 0.0
        ref = self.reference_point
        # Sort by first objective descending
        sorted_front = sorted(front, key=lambda x: x[0], reverse=True)
        hv = 0.0
        prev_y = ref[1]
        for point in sorted_front:
            if point[0] > ref[0] or point[1] > ref[1]:
                continue  # dominated by reference
            hv += (ref[0] - point[0]) * (prev_y - point[1])
            prev_y = min(prev_y, point[1])
        return hv

    def compute_mc(self, front: List[np.ndarray],
                   n_samples: int = 100_000, seed: int = 42) -> float:
        """Monte Carlo hypervolume approximation for any dimension."""
        if not front:
            return 0.0
        rng = np.random.default_rng(seed)
        ref = self.reference_point
        n_obj = len(ref)

        # Ideal point (best in each objective)
        ideal = np.min(np.array(front), axis=0)
        # Volume of bounding box
        box_vol = float(np.prod(ref - ideal))
        if box_vol <= 0:
            return 0.0

        # Sample random points in bounding box
        samples = rng.uniform(ideal, ref, size=(n_samples, n_obj))
        # Count points dominated by at least one front member
        dominated = np.zeros(n_samples, dtype=bool)
        for point in front:
            dominated |= np.all(samples >= point, axis=1)

        hv = box_vol * float(np.mean(dominated))
        return hv

    def compute(self, front: List[np.ndarray]) -> float:
        """Auto-select algorithm based on dimensionality."""
        if not front:
            return 0.0
        n_obj = len(front[0])
        if n_obj == 2:
            return self.compute_2d(front)
        return self.compute_mc(front)


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------

def tournament_selection(fitness_list: List[FitnessFunction],
                         objectives: List[str],
                         tournament_size: int = 2,
                         n_select: int = 1,
                         seed: int = 42) -> List[int]:
    """
    Tournament selection using Pareto rank + crowding distance.

    Returns list of selected indices.
    """
    rng = np.random.default_rng(seed)
    vectors = [f.objective_vector(objectives) for f in fitness_list]
    ranks = pareto_ranks(vectors)
    cd = crowding_distance(vectors)
    n = len(fitness_list)

    selected: List[int] = []
    for _ in range(n_select):
        candidates = rng.choice(n, size=min(tournament_size, n), replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best] and cd[c] > cd[best]:
                best = c
        selected.append(int(best))
    return selected


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def evaluate_population(evaluator: BacktestFitnessEvaluator,
                        signals: List[np.ndarray]) -> List[FitnessFunction]:
    """Evaluate a list of signals and return fitness for each."""
    return [evaluator.evaluate(sig) for sig in signals]


def population_fitness_summary(fitness_list: List[FitnessFunction]
                               ) -> Dict[str, float]:
    """Summary statistics of population fitness."""
    sharpes = [f.sharpe for f in fitness_list]
    return {
        "mean_sharpe": float(np.mean(sharpes)),
        "max_sharpe": float(np.max(sharpes)),
        "min_sharpe": float(np.min(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "mean_win_rate": float(np.mean([f.win_rate for f in fitness_list])),
        "mean_max_dd": float(np.mean([f.max_drawdown for f in fitness_list])),
        "mean_profit_factor": float(np.mean([f.profit_factor for f in fitness_list])),
        "n_positive_sharpe": int(np.sum(np.array(sharpes) > 0)),
    }
