"""
Multi-objective fitness evaluation for genetic algorithm strategy optimization.

Provides Sharpe ratio, Calmar ratio, profit factor, drawdown, trade frequency
fitness functions, Pareto front computation (NSGA-II), and overfitting penalties.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .genome import StrategyGenome


# ---------------------------------------------------------------------------
# Return series utilities
# ---------------------------------------------------------------------------

def compute_sharpe(returns: List[float], risk_free_rate: float = 0.0,
                   annualize: bool = True, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio from daily returns."""
    if len(returns) < 2:
        return 0.0
    excess = [r - risk_free_rate / periods_per_year for r in returns]
    mean_excess = sum(excess) / len(excess)
    variance = sum((r - mean_excess) ** 2 for r in excess) / max(len(excess) - 1, 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    ratio = mean_excess / std
    if annualize:
        ratio *= math.sqrt(periods_per_year)
    return ratio


def compute_sortino(returns: List[float], risk_free_rate: float = 0.0,
                    annualize: bool = True, periods_per_year: int = 252) -> float:
    """Compute annualized Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    threshold = risk_free_rate / periods_per_year
    excess = [r - threshold for r in returns]
    mean_excess = sum(excess) / len(excess)
    downside = [min(r, 0) for r in excess]
    downside_variance = sum(d ** 2 for d in downside) / max(len(downside) - 1, 1)
    downside_std = math.sqrt(downside_variance)
    if downside_std == 0:
        return 0.0 if mean_excess <= 0 else 10.0
    ratio = mean_excess / downside_std
    if annualize:
        ratio *= math.sqrt(periods_per_year)
    return ratio


def compute_max_drawdown(returns: List[float]) -> float:
    """Compute maximum drawdown from a series of returns (as fractions)."""
    if not returns:
        return 0.0
    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        cumulative *= (1.0 + r)
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_calmar(returns: List[float], periods_per_year: int = 252) -> float:
    """Compute annualized Calmar ratio = CAGR / max drawdown."""
    if not returns:
        return 0.0
    max_dd = compute_max_drawdown(returns)
    if max_dd == 0:
        return 10.0 if sum(returns) > 0 else 0.0
    total_return = 1.0
    for r in returns:
        total_return *= (1.0 + r)
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    cagr = total_return ** (1.0 / n_years) - 1.0
    return cagr / max_dd


def compute_profit_factor(returns: List[float]) -> float:
    """Gross profit / gross loss. Returns inf if no losses."""
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    if gross_loss == 0:
        return 10.0 if gross_profit > 0 else 1.0
    return gross_profit / gross_loss


def compute_win_rate(returns: List[float]) -> float:
    """Fraction of positive returns."""
    if not returns:
        return 0.0
    return sum(1 for r in returns if r > 0) / len(returns)


def compute_omega_ratio(returns: List[float], threshold: float = 0.0) -> float:
    """Omega ratio: probability-weighted gains above threshold / losses below."""
    above = sum(r - threshold for r in returns if r > threshold)
    below = sum(threshold - r for r in returns if r < threshold)
    if below == 0:
        return 10.0 if above > 0 else 1.0
    return above / below


def compute_var(returns: List[float], confidence: float = 0.05) -> float:
    """Value at Risk at given confidence level (lower tail)."""
    if not returns:
        return 0.0
    sorted_r = sorted(returns)
    idx = max(0, int(math.floor(confidence * len(returns))) - 1)
    return -sorted_r[idx]


def compute_cvar(returns: List[float], confidence: float = 0.05) -> float:
    """Conditional VaR (Expected Shortfall) at given confidence level."""
    if not returns:
        return 0.0
    sorted_r = sorted(returns)
    n_tail = max(1, int(math.floor(confidence * len(returns))))
    tail = sorted_r[:n_tail]
    return -sum(tail) / len(tail)


def compute_information_ratio(strategy_returns: List[float],
                               benchmark_returns: List[float],
                               annualize: bool = True,
                               periods_per_year: int = 252) -> float:
    """Information ratio = mean active return / tracking error."""
    if len(strategy_returns) != len(benchmark_returns) or not strategy_returns:
        return 0.0
    active = [s - b for s, b in zip(strategy_returns, benchmark_returns)]
    mean_active = sum(active) / len(active)
    variance = sum((a - mean_active) ** 2 for a in active) / max(len(active) - 1, 1)
    tracking_error = math.sqrt(variance)
    if tracking_error == 0:
        return 0.0
    ir = mean_active / tracking_error
    if annualize:
        ir *= math.sqrt(periods_per_year)
    return ir


# ---------------------------------------------------------------------------
# Backtest simulator (light-weight, used inside fitness evaluation)
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Result of a quick in-sample backtest."""
    returns: List[float]
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    total_return: float
    n_trades: int
    annualized_return: float
    annualized_vol: float
    var_5pct: float
    cvar_5pct: float
    omega_ratio: float


def simulate_strategy(params: Dict[str, Any],
                      price_data: List[float],
                      strategy_type: str = "momentum",
                      transaction_cost: float = 0.001) -> BacktestResult:
    """
    Lightweight strategy simulation for fitness evaluation.
    Uses params dict to configure a simple strategy and runs it
    on price_data (list of close prices).
    """
    if len(price_data) < 10:
        empty = BacktestResult(
            returns=[], sharpe=0.0, sortino=0.0, calmar=0.0,
            max_drawdown=0.0, profit_factor=1.0, win_rate=0.0,
            total_return=0.0, n_trades=0, annualized_return=0.0,
            annualized_vol=0.0, var_5pct=0.0, cvar_5pct=0.0, omega_ratio=1.0,
        )
        return empty

    # Compute log returns
    log_returns = [math.log(price_data[i] / price_data[i - 1])
                   for i in range(1, len(price_data))]

    strategy_returns = []
    position = 0.0  # current position: +1, -1, or 0
    n_trades = 0

    if strategy_type == "momentum":
        lookback = int(params.get("lookback_fast", 10))
        slow = int(params.get("lookback_slow", 30))
        threshold = float(params.get("signal_threshold", 0.0))
        max_pos = float(params.get("max_position", 1.0))

        for i in range(max(lookback, slow), len(log_returns)):
            fast_ret = sum(log_returns[i - lookback:i]) / lookback
            slow_ret = sum(log_returns[i - slow:i]) / slow
            signal = fast_ret - slow_ret

            vol_w = min(i, 20)
            recent_vol = math.sqrt(
                sum(r ** 2 for r in log_returns[i - vol_w:i]) / vol_w + 1e-10
            )
            normalized_signal = signal / (vol_w * recent_vol + 1e-10)

            new_position = max_pos if normalized_signal > threshold else (
                -max_pos if normalized_signal < -threshold else 0.0
            )

            if new_position != position:
                n_trades += 1
                cost = abs(new_position - position) * transaction_cost
            else:
                cost = 0.0

            strategy_returns.append(position * log_returns[i] - cost)
            position = new_position

    elif strategy_type == "mean_reversion":
        window = int(params.get("mean_window", 20))
        entry_z = float(params.get("entry_zscore", 2.0))
        exit_z = float(params.get("exit_zscore", 0.2))
        max_hold = int(params.get("max_holding_days", 10))
        pos_size = float(params.get("position_size", 0.5))

        hold_days = 0
        for i in range(window, len(log_returns)):
            window_rets = log_returns[i - window:i]
            mean_r = sum(window_rets) / len(window_rets)
            var_r = sum((r - mean_r) ** 2 for r in window_rets) / max(len(window_rets) - 1, 1)
            std_r = math.sqrt(var_r + 1e-10)
            z_score = (log_returns[i] - mean_r) / std_r

            if position != 0:
                hold_days += 1

            if position == 0:
                if z_score > entry_z:
                    position = -pos_size  # short on extreme up
                    n_trades += 1
                    hold_days = 0
                elif z_score < -entry_z:
                    position = pos_size   # long on extreme down
                    n_trades += 1
                    hold_days = 0
            else:
                should_exit = (abs(z_score) < exit_z) or (hold_days >= max_hold)
                if should_exit:
                    position = 0.0
                    n_trades += 1
                    hold_days = 0

            cost = abs(position) * transaction_cost if n_trades > 0 else 0.0
            strategy_returns.append(position * log_returns[i] - cost * 0.01)

    else:
        # Generic: random walk for testing
        strategy_returns = [0.0] * len(log_returns)

    if not strategy_returns:
        strategy_returns = [0.0]

    periods_per_year = 252
    n_years = len(strategy_returns) / periods_per_year

    sharpe = compute_sharpe(strategy_returns)
    sortino = compute_sortino(strategy_returns)
    max_dd = compute_max_drawdown(strategy_returns)
    calmar = compute_calmar(strategy_returns)
    pf = compute_profit_factor(strategy_returns)
    win_rate = compute_win_rate(strategy_returns)

    total_r = sum(strategy_returns)
    ann_r = total_r / max(n_years, 1e-10)
    mean_r = sum(strategy_returns) / len(strategy_returns)
    variance = sum((r - mean_r) ** 2 for r in strategy_returns) / max(len(strategy_returns) - 1, 1)
    ann_vol = math.sqrt(variance * periods_per_year)

    return BacktestResult(
        returns=strategy_returns,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        profit_factor=pf,
        win_rate=win_rate,
        total_return=total_r,
        n_trades=n_trades,
        annualized_return=ann_r,
        annualized_vol=ann_vol,
        var_5pct=compute_var(strategy_returns),
        cvar_5pct=compute_cvar(strategy_returns),
        omega_ratio=compute_omega_ratio(strategy_returns),
    )


# ---------------------------------------------------------------------------
# Fitness functions
# ---------------------------------------------------------------------------

@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation."""
    # Objectives (used in multi-objective mode)
    objectives: List[str] = field(default_factory=lambda: ["sharpe"])
    # Objective weights for single-objective scalarization
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe": 1.0,
        "calmar": 0.5,
        "profit_factor": 0.3,
        "win_rate": 0.2,
        "max_drawdown": -1.0,  # negative weight = penalize
        "trade_frequency": -0.1,
    })
    # Overfitting penalty parameters
    use_overfitting_penalty: bool = True
    in_sample_fraction: float = 0.60
    oos_sharpe_weight: float = 0.4   # weight of out-of-sample Sharpe
    in_sample_weight: float = 0.6
    is_oos_correlation_target: float = 0.8
    # Trade frequency bounds
    min_trades_per_year: int = 5
    max_trades_per_year: int = 500
    # Drawdown constraint
    max_acceptable_drawdown: float = 0.30
    # Risk-free rate
    risk_free_rate: float = 0.02
    # Transaction cost
    transaction_cost_bps: float = 10.0
    # Periods per year
    periods_per_year: int = 252
    # Multi-objective mode
    multi_objective: bool = False


class FitnessEvaluator:
    """
    Evaluates trading strategy genomes and assigns fitness scores.
    Supports single-objective (weighted sum) and multi-objective modes.
    """

    def __init__(self, config: FitnessConfig,
                 price_data: Optional[List[float]] = None,
                 strategy_type: str = "momentum") -> None:
        self.config = config
        self._price_data = price_data or self._generate_synthetic_prices(1000)
        self.strategy_type = strategy_type
        self._n_evaluations = 0

    @staticmethod
    def _generate_synthetic_prices(n: int, seed: int = 42) -> List[float]:
        """Generate synthetic GBM price series for testing."""
        rng = random.Random(seed)
        prices = [100.0]
        mu = 0.0001
        sigma = 0.015
        for _ in range(n):
            r = rng.gauss(mu, sigma)
            prices.append(prices[-1] * math.exp(r))
        return prices

    def evaluate(self, genome: StrategyGenome) -> StrategyGenome:
        """Evaluate a genome and assign fitness (and objectives if multi-objective)."""
        params = genome.chromosome.to_dict()
        transaction_cost = self.config.transaction_cost_bps / 10000.0

        if self.config.use_overfitting_penalty:
            # Split data into in-sample and out-of-sample
            split = int(len(self._price_data) * self.config.in_sample_fraction)
            is_data = self._price_data[:split + 1]
            oos_data = self._price_data[split:]

            is_result = simulate_strategy(
                params, is_data, self.strategy_type, transaction_cost)
            oos_result = simulate_strategy(
                params, oos_data, self.strategy_type, transaction_cost)

            # Compute combined fitness with overfitting penalty
            if self.config.multi_objective:
                is_objs = self._compute_objectives(is_result, params)
                oos_objs = self._compute_objectives(oos_result, params)
                # Weighted average of in-sample and out-of-sample objectives
                objectives = [
                    self.config.in_sample_weight * iso +
                    self.config.oos_sharpe_weight * ooso
                    for iso, ooso in zip(is_objs, oos_objs)
                ]
                genome.objectives = objectives
                genome.fitness = sum(objectives) / max(len(objectives), 1)
            else:
                is_fitness = self._scalarize(is_result, params)
                oos_fitness = self._scalarize(oos_result, params)
                overfitting_penalty = self._compute_overfitting_penalty(
                    is_result, oos_result)
                combined = (
                    self.config.in_sample_weight * is_fitness +
                    self.config.oos_sharpe_weight * oos_fitness -
                    overfitting_penalty
                )
                genome.fitness = combined
        else:
            result = simulate_strategy(
                params, self._price_data, self.strategy_type, transaction_cost)
            if self.config.multi_objective:
                genome.objectives = self._compute_objectives(result, params)
                genome.fitness = sum(genome.objectives) / max(len(genome.objectives), 1)
            else:
                genome.fitness = self._scalarize(result, params)

        # Constraint violations
        genome.constraint_violations = self._check_constraints(genome, params)
        if genome.constraint_violations > 0:
            penalty = genome.constraint_violations * 0.5
            genome.fitness = (genome.fitness or 0.0) - penalty

        genome.metadata.n_evaluations += 1
        self._n_evaluations += 1
        return genome

    def _compute_objectives(self, result: BacktestResult,
                            params: Dict[str, Any]) -> List[float]:
        """Compute list of objectives for multi-objective optimization."""
        obj_map = {
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "calmar": result.calmar,
            "profit_factor": min(result.profit_factor, 10.0),
            "win_rate": result.win_rate,
            "max_drawdown": -result.max_drawdown,  # negated (maximize = minimize drawdown)
            "trade_frequency": self._trade_frequency_score(result.n_trades),
            "omega_ratio": min(result.omega_ratio, 10.0),
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
        }
        return [obj_map.get(obj, 0.0) for obj in self.config.objectives]

    def _scalarize(self, result: BacktestResult,
                   params: Dict[str, Any]) -> float:
        """Weighted sum scalarization of multiple objectives."""
        weights = self.config.objective_weights
        score = 0.0

        if "sharpe" in weights:
            score += weights["sharpe"] * result.sharpe
        if "sortino" in weights:
            score += weights.get("sortino", 0.0) * result.sortino
        if "calmar" in weights:
            score += weights["calmar"] * result.calmar
        if "profit_factor" in weights:
            score += weights["profit_factor"] * min(result.profit_factor, 10.0)
        if "win_rate" in weights:
            score += weights["win_rate"] * result.win_rate
        if "max_drawdown" in weights:
            score += weights["max_drawdown"] * result.max_drawdown  # weight is negative
        if "trade_frequency" in weights:
            score += weights["trade_frequency"] * self._trade_frequency_score(result.n_trades)
        if "omega_ratio" in weights:
            score += weights.get("omega_ratio", 0.0) * min(result.omega_ratio, 10.0)

        return score

    def _trade_frequency_score(self, n_trades: int) -> float:
        """
        Penalize strategies with too few or too many trades.
        Returns 0 for out-of-range, 1 for ideal range.
        """
        n_years = len(self._price_data) / self.config.periods_per_year
        trades_per_year = n_trades / max(n_years, 1e-10)
        min_t = self.config.min_trades_per_year
        max_t = self.config.max_trades_per_year

        if trades_per_year < min_t:
            return trades_per_year / min_t - 1.0  # negative below min
        if trades_per_year > max_t:
            return -(trades_per_year - max_t) / max_t
        # Gaussian bell around midpoint of [min_t, max_t]
        mid = (min_t + max_t) / 2.0
        width = (max_t - min_t) / 4.0
        return math.exp(-((trades_per_year - mid) ** 2) / (2 * width ** 2))

    def _compute_overfitting_penalty(self, is_result: BacktestResult,
                                      oos_result: BacktestResult) -> float:
        """
        Penalty for large degradation from in-sample to out-of-sample performance.
        """
        sharpe_degradation = max(0.0, is_result.sharpe - oos_result.sharpe)
        dd_increase = max(0.0, oos_result.max_drawdown - is_result.max_drawdown)

        # Correlation of IS vs OOS return series (if sufficient data)
        is_r = is_result.returns
        oos_r = oos_result.returns
        corr_penalty = 0.0
        if len(is_r) > 20 and len(oos_r) > 20:
            is_sample = is_r[-min(100, len(is_r)):]
            oos_sample = oos_r[:min(100, len(oos_r))]
            # Pattern correlation: do high-return periods match?
            n = min(len(is_sample), len(oos_sample))
            if n > 1:
                is_n = is_sample[:n]
                oos_n = oos_sample[:n]
                mean_is = sum(is_n) / n
                mean_oos = sum(oos_n) / n
                cov = sum((a - mean_is) * (b - mean_oos)
                          for a, b in zip(is_n, oos_n)) / n
                std_is = math.sqrt(sum((a - mean_is) ** 2 for a in is_n) / n + 1e-10)
                std_oos = math.sqrt(sum((b - mean_oos) ** 2 for b in oos_n) / n + 1e-10)
                correlation = cov / (std_is * std_oos)
                target = self.config.is_oos_correlation_target
                if correlation < target:
                    corr_penalty = (target - correlation) * 0.5

        penalty = 0.3 * sharpe_degradation + 0.2 * dd_increase * 5 + corr_penalty
        return max(0.0, penalty)

    def _check_constraints(self, genome: StrategyGenome,
                           params: Dict[str, Any]) -> float:
        """Check constraint violations. Returns total violation magnitude."""
        violations = 0.0
        # Max drawdown constraint
        # (Actual constraint checking happens after simulation, not here)
        # Parameter-level constraints
        if "lookback_fast" in params and "lookback_slow" in params:
            fast = int(params["lookback_fast"])
            slow = int(params["lookback_slow"])
            if fast >= slow:
                violations += (fast - slow + 1) / slow
        if "rsi_oversold" in params and "rsi_overbought" in params:
            if params["rsi_oversold"] >= params["rsi_overbought"]:
                violations += 1.0
        if "entry_zscore" in params and "exit_zscore" in params:
            if params.get("exit_zscore", 0) >= params.get("entry_zscore", 1):
                violations += 1.0
        return violations

    def evaluate_population(self, genomes: List[StrategyGenome]) -> List[StrategyGenome]:
        """Evaluate all genomes in a population."""
        for genome in genomes:
            if genome.fitness is None:
                self.evaluate(genome)
        return genomes

    @property
    def n_evaluations(self) -> int:
        return self._n_evaluations


# ---------------------------------------------------------------------------
# Multi-objective fitness: NSGA-II Pareto front utilities
# ---------------------------------------------------------------------------

class ParetoAnalysis:
    """Analysis tools for multi-objective Pareto-optimal solutions."""

    @staticmethod
    def hypervolume_indicator(front: List[StrategyGenome],
                              reference_point: Optional[List[float]] = None) -> float:
        """
        Compute hypervolume indicator for a 2D Pareto front.
        Only supports 2 objectives (for higher dimensions, use Monte Carlo approximation).
        """
        if not front or front[0].objectives is None:
            return 0.0
        n_obj = len(front[0].objectives)
        if n_obj != 2:
            return ParetoAnalysis._hypervolume_mc(front, reference_point)

        objs = [(g.objectives[0], g.objectives[1]) for g in front
                if g.objectives is not None]
        if not objs:
            return 0.0

        if reference_point is None:
            ref = [min(o[i] for o in objs) - 1.0 for i in range(2)]
        else:
            ref = reference_point

        # Sort by first objective descending
        objs_sorted = sorted(objs, key=lambda x: x[0], reverse=True)
        hv = 0.0
        current_y = ref[1]
        for x, y in objs_sorted:
            if y > current_y:
                hv += (x - ref[0]) * (y - current_y)
                current_y = y
        return max(0.0, hv)

    @staticmethod
    def _hypervolume_mc(front: List[StrategyGenome],
                        reference_point: Optional[List[float]] = None,
                        n_samples: int = 10000) -> float:
        """Monte Carlo hypervolume estimate for n > 2 objectives."""
        if not front or front[0].objectives is None:
            return 0.0
        n_obj = len(front[0].objectives)
        rng = random.Random(42)

        all_objs = [g.objectives for g in front if g.objectives is not None]
        if not all_objs:
            return 0.0

        if reference_point is None:
            ref = [min(o[i] for o in all_objs) - 1.0 for i in range(n_obj)]
        else:
            ref = reference_point

        max_vals = [max(o[i] for o in all_objs) for i in range(n_obj)]

        dominated_count = 0
        for _ in range(n_samples):
            point = [rng.uniform(ref[i], max_vals[i]) for i in range(n_obj)]
            # Check if point is dominated by any member of front
            for o in all_objs:
                if all(o[i] >= point[i] for i in range(n_obj)):
                    dominated_count += 1
                    break

        volume = 1.0
        for i in range(n_obj):
            volume *= (max_vals[i] - ref[i])
        return volume * dominated_count / n_samples

    @staticmethod
    def spacing_metric(front: List[StrategyGenome]) -> float:
        """
        Compute spacing metric: std dev of distances to nearest neighbor.
        Lower = more uniformly spread front.
        """
        if len(front) < 2 or front[0].objectives is None:
            return 0.0

        objs = [g.objectives for g in front if g.objectives is not None]
        n = len(objs)
        n_obj = len(objs[0])

        min_dists = []
        for i in range(n):
            min_d = float("inf")
            for j in range(n):
                if i == j:
                    continue
                d = math.sqrt(sum((objs[i][k] - objs[j][k]) ** 2 for k in range(n_obj)))
                min_d = min(min_d, d)
            if min_d < float("inf"):
                min_dists.append(min_d)

        if not min_dists:
            return 0.0
        mean_d = sum(min_dists) / len(min_dists)
        variance = sum((d - mean_d) ** 2 for d in min_dists) / len(min_dists)
        return math.sqrt(variance)

    @staticmethod
    def extract_best_by_objective(front: List[StrategyGenome],
                                  obj_index: int = 0) -> Optional[StrategyGenome]:
        """Return the genome with the best value for a specific objective."""
        candidates = [g for g in front if g.objectives is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda g: g.objectives[obj_index])  # type: ignore

    @staticmethod
    def knee_point(front: List[StrategyGenome]) -> Optional[StrategyGenome]:
        """
        Find the knee point (best trade-off) in a 2D Pareto front.
        Uses minimum distance to normalized ideal-nadir line.
        """
        if not front or front[0].objectives is None or len(front[0].objectives) != 2:
            return front[0] if front else None

        objs = [(g, g.objectives) for g in front if g.objectives is not None]
        if not objs:
            return None

        min_obj = [min(o[i] for _, o in objs) for i in range(2)]
        max_obj = [max(o[i] for _, o in objs) for i in range(2)]
        ranges = [max_obj[i] - min_obj[i] for i in range(2)]

        # Normalize to [0, 1]
        norm = [
            (g, [(o[i] - min_obj[i]) / max(ranges[i], 1e-10) for i in range(2)])
            for g, o in objs
        ]

        # Line from (0, 1) to (1, 0) in normalized space
        # Distance from point (x, y) to this line (x + y - 1 = 0) is |x + y - 1| / sqrt(2)
        best_genome = min(norm, key=lambda x: abs(x[1][0] + x[1][1] - 1.0))
        return best_genome[0]


# ---------------------------------------------------------------------------
# Composite fitness evaluator with multiple objectives
# ---------------------------------------------------------------------------

class MultiObjectiveFitnessEvaluator(FitnessEvaluator):
    """
    Fitness evaluator specifically for multi-objective optimization.
    Computes Sharpe, Calmar, and other objectives independently.
    """

    def __init__(self, objectives: Optional[List[str]] = None,
                 price_data: Optional[List[float]] = None,
                 strategy_type: str = "momentum",
                 transaction_cost_bps: float = 10.0) -> None:
        if objectives is None:
            objectives = ["sharpe", "calmar", "profit_factor", "max_drawdown"]
        config = FitnessConfig(
            objectives=objectives,
            multi_objective=True,
            use_overfitting_penalty=True,
            transaction_cost_bps=transaction_cost_bps,
        )
        super().__init__(config, price_data, strategy_type)


# ---------------------------------------------------------------------------
# Overfitting penalty analysis
# ---------------------------------------------------------------------------

class OverfittingAnalyzer:
    """
    Analyzes and quantifies overfitting in evolved strategies.
    """

    def __init__(self, n_folds: int = 5, periods_per_year: int = 252) -> None:
        self.n_folds = n_folds
        self.periods_per_year = periods_per_year

    def walk_forward_score(self, genome: StrategyGenome,
                           price_data: List[float],
                           strategy_type: str = "momentum") -> Dict[str, float]:
        """
        Compute walk-forward scores for a genome across multiple time folds.
        Returns summary statistics.
        """
        params = genome.chromosome.to_dict()
        n = len(price_data)
        fold_size = n // self.n_folds
        if fold_size < 50:
            return {"mean_sharpe": 0.0, "std_sharpe": 1.0,
                    "degradation": 1.0, "consistency": 0.0}

        is_sharpes = []
        oos_sharpes = []

        # Walk-forward: train on fold 0..k, test on fold k+1
        for k in range(self.n_folds - 1):
            train_end = (k + 1) * fold_size
            test_end = (k + 2) * fold_size

            is_data = price_data[:train_end + 1]
            oos_data = price_data[train_end:test_end + 1]

            if len(is_data) < 20 or len(oos_data) < 10:
                continue

            is_r = simulate_strategy(params, is_data, strategy_type)
            oos_r = simulate_strategy(params, oos_data, strategy_type)

            is_sharpes.append(is_r.sharpe)
            oos_sharpes.append(oos_r.sharpe)

        if not is_sharpes:
            return {"mean_sharpe": 0.0, "std_sharpe": 1.0,
                    "degradation": 1.0, "consistency": 0.0}

        mean_is = sum(is_sharpes) / len(is_sharpes)
        mean_oos = sum(oos_sharpes) / len(oos_sharpes)
        degradation = max(0.0, mean_is - mean_oos)
        consistency = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)
        std_oos = math.sqrt(
            sum((s - mean_oos) ** 2 for s in oos_sharpes) / max(len(oos_sharpes) - 1, 1)
        ) if len(oos_sharpes) > 1 else 0.0

        return {
            "mean_is_sharpe": mean_is,
            "mean_oos_sharpe": mean_oos,
            "std_oos_sharpe": std_oos,
            "degradation": degradation,
            "consistency": consistency,
            "overfitting_score": degradation / (abs(mean_is) + 1e-10),
        }

    def compute_complexity_penalty(self, genome: StrategyGenome) -> float:
        """
        Penalize overly complex strategies.
        Uses number of parameters and interaction terms as proxy for complexity.
        """
        n_params = len(genome.chromosome.genes)
        # Very rough proxy: more parameters = more overfitting risk
        base_penalty = 0.0
        if n_params > 20:
            base_penalty = (n_params - 20) * 0.01
        return base_penalty


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Fitness self-test ===")
    from .genome import GenomeFactory

    # Generate synthetic price data
    evaluator = FitnessEvaluator(FitnessConfig())
    prices = evaluator._generate_synthetic_prices(500, seed=42)
    print(f"Generated {len(prices)} synthetic prices, range: {min(prices):.2f}-{max(prices):.2f}")

    # Test single-strategy evaluation
    factory = GenomeFactory("momentum", seed=42)
    genome = factory.create_random()
    print(f"\nPre-eval: {genome}")

    evaluator = FitnessEvaluator(
        FitnessConfig(use_overfitting_penalty=True),
        price_data=prices,
        strategy_type="momentum",
    )
    evaluator.evaluate(genome)
    print(f"Post-eval: {genome}")

    # Test population evaluation
    pop = factory.create_population(10)
    evaluator.evaluate_population(pop)
    fitnesses = [g.fitness for g in pop if g.fitness is not None]
    print(f"\nPopulation fitness: min={min(fitnesses):.4f}, max={max(fitnesses):.4f}")

    # Test multi-objective
    mo_evaluator = MultiObjectiveFitnessEvaluator(
        objectives=["sharpe", "calmar", "max_drawdown"],
        price_data=prices,
    )
    mo_pop = factory.create_population(10)
    mo_evaluator.evaluate_population(mo_pop)
    objs_example = mo_pop[0].objectives
    print(f"\nMulti-obj example: sharpe={objs_example[0]:.4f}, calmar={objs_example[1]:.4f}, "
          f"dd={objs_example[2]:.4f}")

    # Test Pareto analysis
    for g in mo_pop:
        g.objectives = [random.gauss(0.5, 0.3), random.gauss(0.3, 0.2)]
    hv = ParetoAnalysis.hypervolume_indicator(mo_pop)
    print(f"\nHypervolume indicator: {hv:.4f}")
    spacing = ParetoAnalysis.spacing_metric(mo_pop)
    print(f"Spacing metric: {spacing:.4f}")
    knee = ParetoAnalysis.knee_point(mo_pop)
    print(f"Knee point objectives: {knee.objectives}")

    # Test constraint checking
    constrained_genome = factory.create_from_dict(
        {"lookback_fast": 30, "lookback_slow": 10})  # Invalid: fast > slow
    evaluator.evaluate(constrained_genome)
    print(f"\nConstrained genome violations: {constrained_genome.constraint_violations:.4f}")
    print(f"Constrained genome fitness: {constrained_genome.fitness:.4f}")

    # Test walk-forward analysis
    analyzer = OverfittingAnalyzer(n_folds=3)
    wf_result = analyzer.walk_forward_score(pop[0], prices)
    print(f"\nWalk-forward: {wf_result}")

    # Test metrics
    sample_returns = [random.gauss(0.001, 0.01) for _ in range(252)]
    print(f"\nSample metrics:")
    print(f"  Sharpe: {compute_sharpe(sample_returns):.4f}")
    print(f"  Sortino: {compute_sortino(sample_returns):.4f}")
    print(f"  MaxDD: {compute_max_drawdown(sample_returns):.4f}")
    print(f"  Calmar: {compute_calmar(sample_returns):.4f}")
    print(f"  ProfitFactor: {compute_profit_factor(sample_returns):.4f}")
    print(f"  WinRate: {compute_win_rate(sample_returns):.4f}")
    print(f"  VaR 5%: {compute_var(sample_returns):.4f}")
    print(f"  CVaR 5%: {compute_cvar(sample_returns):.4f}")

    print("\nAll fitness tests passed.")
