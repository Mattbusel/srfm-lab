"""
self_improving_engine.py — Self-improving idea engine.

Wraps IdeaBank and a scoring history to adaptively evolve signal weights,
prune/promote templates, learn regime-signal associations, and auto-tune
pipeline thresholds via gradient-free optimization.
"""

from __future__ import annotations

import math
import random
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ScoredRun:
    """Record of one evaluation of a parameter configuration."""
    run_id: str
    params: Dict[str, float]
    signal_weights: Dict[str, float]
    regime: str
    sharpe: float
    pnl: float
    max_drawdown: float
    n_trades: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class TemplateRecord:
    """Track lifetime stats for a signal template."""
    name: str
    n_activations: int = 0
    total_pnl: float = 0.0
    total_sharpe: float = 0.0
    wins: int = 0
    losses: int = 0
    promoted: bool = False
    pruned: bool = False
    regime_affinity: Dict[str, float] = field(default_factory=dict)  # regime → avg_sharpe

    @property
    def mean_sharpe(self) -> float:
        if self.n_activations == 0:
            return 0.0
        return self.total_sharpe / self.n_activations

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5


@dataclass
class FitnessPoint:
    """A point on the parameter fitness landscape."""
    params: Dict[str, float]
    fitness: float            # higher is better (e.g. Sharpe)
    n_evals: int = 1


@dataclass
class EvolutionSummary:
    """What changed during a self-improvement cycle."""
    cycle_id: int
    pruned_templates: List[str]
    promoted_templates: List[str]
    weight_changes: Dict[str, Tuple[float, float]]   # {signal: (old, new)}
    threshold_changes: Dict[str, Tuple[float, float]]
    best_params: Dict[str, float]
    best_sharpe: float
    regime_discoveries: Dict[str, str]   # {regime: best_signal}
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Fitness Landscape Tracker
# ---------------------------------------------------------------------------

class FitnessLandscape:
    """
    Tracks evaluated parameter configurations and their fitness.
    Supports retrieval of best / neighbourhood sampling.
    """

    def __init__(self, max_points: int = 500):
        self.max_points = max_points
        self._points: List[FitnessPoint] = []
        self._param_keys: Optional[List[str]] = None

    def add(self, params: Dict[str, float], fitness: float) -> None:
        # Check if we've seen these params before (within tolerance)
        for pt in self._points:
            if self._params_close(pt.params, params):
                pt.fitness = (pt.fitness * pt.n_evals + fitness) / (pt.n_evals + 1)
                pt.n_evals += 1
                return
        self._points.append(FitnessPoint(params=dict(params), fitness=fitness))
        if len(self._points) > self.max_points:
            # Drop the worst performer
            self._points.sort(key=lambda p: p.fitness, reverse=True)
            self._points = self._points[:self.max_points]
        if self._param_keys is None and params:
            self._param_keys = sorted(params.keys())

    def best(self) -> Optional[FitnessPoint]:
        if not self._points:
            return None
        return max(self._points, key=lambda p: p.fitness)

    def top_k(self, k: int = 5) -> List[FitnessPoint]:
        return sorted(self._points, key=lambda p: p.fitness, reverse=True)[:k]

    def _params_close(self, a: Dict[str, float], b: Dict[str, float],
                       tol: float = 1e-4) -> bool:
        if set(a) != set(b):
            return False
        return all(abs(a[k] - b[k]) < tol for k in a)

    def neighbourhood_sample(self, centre: Dict[str, float],
                               sigma: float = 0.05,
                               rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """Return a parameter dict near `centre` with Gaussian perturbations."""
        if rng is None:
            rng = np.random.default_rng()
        return {k: float(v + rng.normal(0, sigma * max(abs(v), 0.01)))
                for k, v in centre.items()}

    def param_importance(self) -> Dict[str, float]:
        """
        Approximate importance of each parameter via variance of fitness
        conditioned on parameter value (ANOVA-style, very crude).
        """
        if len(self._points) < 5 or self._param_keys is None:
            return {}
        importance: Dict[str, float] = {}
        fitness_arr = np.array([pt.fitness for pt in self._points])
        global_var = float(np.var(fitness_arr)) + 1e-15
        for key in self._param_keys:
            vals = np.array([pt.params.get(key, 0.0) for pt in self._points])
            # Correlation of param value with fitness
            corr = float(np.corrcoef(vals, fitness_arr)[0, 1])
            importance[key] = abs(corr)
        return importance


# ---------------------------------------------------------------------------
# Rolling Performance Attribution
# ---------------------------------------------------------------------------

class PerformanceAttributor:
    """
    Attributes recent PnL/Sharpe to active parameter configurations
    and signal weights using a sliding window.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._runs: Deque[ScoredRun] = deque(maxlen=window)

    def record(self, run: ScoredRun) -> None:
        self._runs.append(run)

    def attribute_to_weights(self) -> Dict[str, float]:
        """
        Returns dict: {signal_name: avg_sharpe_when_active}.
        Proxy: for each run, attribute the Sharpe proportionally to weight.
        """
        signal_sharpe: Dict[str, List[float]] = defaultdict(list)
        for run in self._runs:
            for sig, w in run.signal_weights.items():
                if w > 0:
                    signal_sharpe[sig].append(run.sharpe * w)
        return {k: float(np.mean(v)) for k, v in signal_sharpe.items() if v}

    def attribute_to_params(self) -> Dict[str, float]:
        """
        Returns dict: {param_name: correlation_with_sharpe}.
        """
        if len(self._runs) < 5:
            return {}
        sharpes = np.array([r.sharpe for r in self._runs])
        all_keys = set()
        for r in self._runs:
            all_keys.update(r.params.keys())
        result: Dict[str, float] = {}
        for key in all_keys:
            vals = np.array([r.params.get(key, 0.0) for r in self._runs])
            if np.std(vals) < 1e-10:
                result[key] = 0.0
                continue
            corr = float(np.corrcoef(vals, sharpes)[0, 1])
            result[key] = corr
        return result

    def regime_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Returns {regime: {sharpe_mean, pnl_mean, n}}."""
        by_regime: Dict[str, List[ScoredRun]] = defaultdict(list)
        for r in self._runs:
            by_regime[r.regime].append(r)
        result: Dict[str, Dict[str, float]] = {}
        for regime, runs in by_regime.items():
            sharpes = [r.sharpe for r in runs]
            pnls = [r.pnl for r in runs]
            result[regime] = {
                "sharpe_mean": float(np.mean(sharpes)),
                "pnl_mean": float(np.mean(pnls)),
                "n": float(len(runs)),
            }
        return result


# ---------------------------------------------------------------------------
# Template Manager: pruning and promotion
# ---------------------------------------------------------------------------

class TemplateManager:
    """
    Maintains TemplateRecord for each signal template.
    Prunes consistently poor templates and promotes strong ones.
    """

    def __init__(self,
                 prune_threshold_sharpe: float = -0.5,
                 promote_threshold_sharpe: float = 1.0,
                 min_activations: int = 20,
                 prune_cooldown: int = 50):
        self.prune_thresh = prune_threshold_sharpe
        self.promote_thresh = promote_threshold_sharpe
        self.min_activations = min_activations
        self.prune_cooldown = prune_cooldown
        self._templates: Dict[str, TemplateRecord] = {}
        self._last_action: Dict[str, int] = {}   # {name: bar_count}
        self._bar_count: int = 0

    def ensure(self, name: str) -> TemplateRecord:
        if name not in self._templates:
            self._templates[name] = TemplateRecord(name=name)
        return self._templates[name]

    def record_activation(self, name: str, sharpe: float, pnl: float,
                           regime: str) -> None:
        t = self.ensure(name)
        t.n_activations += 1
        t.total_sharpe += sharpe
        t.total_pnl += pnl
        if pnl > 0:
            t.wins += 1
        else:
            t.losses += 1
        # Regime affinity
        if regime not in t.regime_affinity:
            t.regime_affinity[regime] = sharpe
        else:
            # Exponential moving average
            t.regime_affinity[regime] = 0.9 * t.regime_affinity[regime] + 0.1 * sharpe
        self._bar_count += 1

    def review(self) -> Tuple[List[str], List[str]]:
        """
        Run pruning and promotion checks.
        Returns (pruned_names, promoted_names).
        """
        pruned: List[str] = []
        promoted: List[str] = []
        for name, rec in self._templates.items():
            if rec.pruned or rec.n_activations < self.min_activations:
                continue
            cooldown_ok = (self._bar_count - self._last_action.get(name, 0)) > self.prune_cooldown
            if not cooldown_ok:
                continue
            if rec.mean_sharpe < self.prune_thresh:
                rec.pruned = True
                self._last_action[name] = self._bar_count
                pruned.append(name)
            elif rec.mean_sharpe > self.promote_thresh and not rec.promoted:
                rec.promoted = True
                self._last_action[name] = self._bar_count
                promoted.append(name)
        return pruned, promoted

    def best_template_per_regime(self) -> Dict[str, str]:
        """Returns {regime: best_template_name} by regime affinity."""
        regime_best: Dict[str, Tuple[str, float]] = {}
        for name, rec in self._templates.items():
            if rec.pruned:
                continue
            for regime, avg_sharpe in rec.regime_affinity.items():
                if regime not in regime_best or avg_sharpe > regime_best[regime][1]:
                    regime_best[regime] = (name, avg_sharpe)
        return {r: v[0] for r, v in regime_best.items()}

    def all_active(self) -> List[str]:
        return [n for n, t in self._templates.items() if not t.pruned]

    def summary(self) -> Dict[str, Dict]:
        return {n: {"mean_sharpe": t.mean_sharpe, "win_rate": t.win_rate,
                    "n": t.n_activations, "promoted": t.promoted, "pruned": t.pruned}
                for n, t in self._templates.items()}


# ---------------------------------------------------------------------------
# Gradient-Free Parameter Optimizer (CMA-ES lite / pattern search)
# ---------------------------------------------------------------------------

class GradientFreeOptimizer:
    """
    Lightweight gradient-free optimizer using a mix of:
    - Nelder-Mead simplex moves (exploratory)
    - Gaussian mutation around current best (exploitation)
    - Random restarts when stagnant
    """

    def __init__(self,
                 param_bounds: Dict[str, Tuple[float, float]],
                 mutation_sigma: float = 0.05,
                 population: int = 12,
                 stagnation_patience: int = 20,
                 seed: int = 0):
        self.bounds = param_bounds
        self.mutation_sigma = mutation_sigma
        self.population = population
        self.stagnation_patience = stagnation_patience
        self._rng = np.random.default_rng(seed)
        self._best_params: Optional[Dict[str, float]] = None
        self._best_fitness: float = -math.inf
        self._stagnation_count: int = 0
        self._landscape = FitnessLandscape(max_points=300)

    def _random_params(self) -> Dict[str, float]:
        return {k: float(self._rng.uniform(lo, hi))
                for k, (lo, hi) in self.bounds.items()}

    def _clip(self, params: Dict[str, float]) -> Dict[str, float]:
        return {k: float(np.clip(v, self.bounds[k][0], self.bounds[k][1]))
                for k, v in params.items()}

    def suggest(self) -> List[Dict[str, float]]:
        """
        Generate `population` candidate parameter dicts to evaluate.
        """
        candidates: List[Dict[str, float]] = []
        if self._best_params is None:
            # Cold start: Latin hypercube-ish random
            for _ in range(self.population):
                candidates.append(self._random_params())
        else:
            # Mostly exploit best + small mutations
            for _ in range(self.population - 2):
                mutated = self._landscape.neighbourhood_sample(
                    self._best_params, sigma=self.mutation_sigma, rng=self._rng)
                candidates.append(self._clip(mutated))
            # 2 random restarts for diversity
            candidates.append(self._random_params())
            candidates.append(self._random_params())
        return candidates

    def update(self, evaluations: List[Tuple[Dict[str, float], float]]) -> bool:
        """
        Record evaluated (params, fitness) pairs.
        Returns True if a new best was found.
        """
        improved = False
        for params, fitness in evaluations:
            self._landscape.add(params, fitness)
            if fitness > self._best_fitness:
                self._best_fitness = fitness
                self._best_params = dict(params)
                self._stagnation_count = 0
                improved = True

        if not improved:
            self._stagnation_count += 1
            if self._stagnation_count >= self.stagnation_patience:
                # Widen search: increase sigma temporarily
                self.mutation_sigma = min(self.mutation_sigma * 1.5, 0.5)
                self._stagnation_count = 0
        else:
            # Tighten search on improvement
            self.mutation_sigma = max(self.mutation_sigma * 0.9, 0.005)
        return improved

    @property
    def best(self) -> Optional[Dict[str, float]]:
        return self._best_params

    @property
    def best_fitness(self) -> float:
        return self._best_fitness

    def param_importance(self) -> Dict[str, float]:
        return self._landscape.param_importance()


# ---------------------------------------------------------------------------
# Main: SelfImprovingEngine
# ---------------------------------------------------------------------------

class SelfImprovingEngine:
    """
    Self-improving idea engine that learns from its own history.

    Orchestrates:
    - Template pruning and promotion (TemplateManager)
    - Signal weight evolution (PerformanceAttributor + gradient-free search)
    - Meta-learning: which signals work in which regime
    - Fitness landscape tracking (FitnessLandscape)
    - Auto-tuning of pipeline thresholds
    - Evolution summaries

    Usage
    -----
    engine = SelfImprovingEngine(signal_names=["momentum", "mean_reversion"],
                                  param_bounds={"entry_threshold": (0.01, 0.1), ...})
    # After each backtest run:
    engine.record_run(params, signal_weights, regime, sharpe, pnl, drawdown, n_trades)
    # Periodically:
    summary = engine.evolve_cycle()
    next_params = engine.suggest_next_params()
    """

    def __init__(self,
                 signal_names: Optional[List[str]] = None,
                 param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 prune_sharpe_threshold: float = -0.5,
                 promote_sharpe_threshold: float = 1.0,
                 min_activations_before_review: int = 20,
                 attribution_window: int = 100,
                 mutation_sigma: float = 0.05,
                 optimizer_population: int = 12,
                 evolution_interval: int = 50,
                 idea_bank: Optional[Any] = None):
        self.signal_names: List[str] = signal_names or []
        self.param_bounds: Dict[str, Tuple[float, float]] = param_bounds or {}
        self._template_mgr = TemplateManager(
            prune_threshold_sharpe=prune_sharpe_threshold,
            promote_threshold_sharpe=promote_sharpe_threshold,
            min_activations=min_activations_before_review)
        self._attributor = PerformanceAttributor(window=attribution_window)
        self._landscape = FitnessLandscape(max_points=500)
        self._optimizer = GradientFreeOptimizer(
            param_bounds=self.param_bounds,
            mutation_sigma=mutation_sigma,
            population=optimizer_population)
        self.idea_bank = idea_bank
        self.evolution_interval = evolution_interval
        self._run_count: int = 0
        self._cycle_count: int = 0
        self._evolution_history: List[EvolutionSummary] = []
        self._current_weights: Dict[str, float] = {
            s: 1.0 / max(len(self.signal_names), 1) for s in self.signal_names}
        self._current_params: Dict[str, float] = {
            k: (lo + hi) / 2.0 for k, (lo, hi) in self.param_bounds.items()}
        # Meta-learning: {regime: {signal: [sharpe_history]}}
        self._regime_signal_perf: Dict[str, Dict[str, Deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=200)))
        self._pending_evaluations: List[Tuple[Dict[str, float], float]] = []

    # ------------------------------------------------------------------
    # Core: record a completed run
    # ------------------------------------------------------------------

    def record_run(self,
                    params: Dict[str, float],
                    signal_weights: Dict[str, float],
                    regime: str,
                    sharpe: float,
                    pnl: float,
                    max_drawdown: float,
                    n_trades: int) -> ScoredRun:
        """
        Record the outcome of one strategy evaluation / backtest run.
        """
        run = ScoredRun(
            run_id=str(uuid.uuid4())[:8],
            params=dict(params),
            signal_weights=dict(signal_weights),
            regime=regime,
            sharpe=sharpe,
            pnl=pnl,
            max_drawdown=max_drawdown,
            n_trades=n_trades)
        self._attributor.record(run)
        # Update template records
        for sig, w in signal_weights.items():
            if w > 0.01:
                self._template_mgr.record_activation(sig, sharpe * w, pnl * w, regime)
        # Meta-learning update
        for sig, w in signal_weights.items():
            if w > 0.01:
                self._regime_signal_perf[regime][sig].append(sharpe)
        # Fitness landscape for composite score
        fitness = self._composite_fitness(sharpe, pnl, max_drawdown, n_trades)
        self._landscape.add(params, fitness)
        self._pending_evaluations.append((params, fitness))
        self._run_count += 1
        # Auto-trigger evolution
        if self._run_count % self.evolution_interval == 0:
            self.evolve_cycle()
        return run

    def _composite_fitness(self, sharpe: float, pnl: float,
                            drawdown: float, n_trades: int) -> float:
        """Composite fitness: Sharpe - penalty for high drawdown and low trade count."""
        drawdown_penalty = max(0.0, drawdown - 0.1) * 2.0
        activity_bonus = math.log1p(n_trades) * 0.05
        return sharpe - drawdown_penalty + activity_bonus

    # ------------------------------------------------------------------
    # Signal Weight Evolution
    # ------------------------------------------------------------------

    def _evolve_weights(self) -> Dict[str, Tuple[float, float]]:
        """
        Adjust signal weights based on attribution.
        Returns {signal: (old_weight, new_weight)}.
        """
        attribution = self._attributor.attribute_to_weights()
        if not attribution:
            return {}
        # Shift weights towards signals with higher attributed Sharpe
        old_weights = dict(self._current_weights)
        raw_new = {}
        for sig in self.signal_names:
            score = attribution.get(sig, 0.0)
            # Gradient-like update: move weight in direction of attribution
            current = self._current_weights.get(sig, 0.0)
            step = 0.05 * np.sign(score)
            raw_new[sig] = max(0.0, current + step)
        total = sum(raw_new.values()) or 1.0
        new_weights = {k: v / total for k, v in raw_new.items()}
        changes = {}
        for sig in self.signal_names:
            if abs(new_weights.get(sig, 0) - old_weights.get(sig, 0)) > 1e-4:
                changes[sig] = (old_weights.get(sig, 0.0), new_weights.get(sig, 0.0))
        self._current_weights = new_weights
        return changes

    # ------------------------------------------------------------------
    # Threshold Auto-Tuning
    # ------------------------------------------------------------------

    def _autotune_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """
        Use GradientFreeOptimizer to improve pipeline thresholds.
        Returns {param: (old_val, new_val)} for changed params.
        """
        if not self.param_bounds:
            return {}
        # Feed pending evaluations into optimizer
        if self._pending_evaluations:
            self._optimizer.update(self._pending_evaluations)
            self._pending_evaluations.clear()
        old_params = dict(self._current_params)
        # Get optimizer's best suggestion
        best = self._optimizer.best
        if best is None:
            return {}
        changes: Dict[str, Tuple[float, float]] = {}
        for k, new_v in best.items():
            old_v = old_params.get(k, new_v)
            if abs(new_v - old_v) > 1e-5:
                changes[k] = (old_v, new_v)
        self._current_params = dict(best)
        return changes

    # ------------------------------------------------------------------
    # Meta-Learning
    # ------------------------------------------------------------------

    def _regime_signal_learning(self) -> Dict[str, str]:
        """
        Identify which signal performs best in each regime.
        Returns {regime: best_signal}.
        """
        discoveries: Dict[str, str] = {}
        for regime, sig_dict in self._regime_signal_perf.items():
            best_sig = None
            best_sharpe = -math.inf
            for sig, history in sig_dict.items():
                if len(history) < 5:
                    continue
                avg = float(np.mean(list(history)))
                if avg > best_sharpe:
                    best_sharpe = avg
                    best_sig = sig
            if best_sig is not None:
                discoveries[regime] = best_sig
        return discoveries

    # ------------------------------------------------------------------
    # Evolution Cycle
    # ------------------------------------------------------------------

    def evolve_cycle(self) -> EvolutionSummary:
        """
        Run one full self-improvement cycle:
        1. Review templates (prune/promote)
        2. Evolve signal weights
        3. Auto-tune thresholds
        4. Capture meta-learning discoveries
        5. Record evolution summary

        Returns an EvolutionSummary describing what changed.
        """
        self._cycle_count += 1
        pruned, promoted = self._template_mgr.review()
        weight_changes = self._evolve_weights()
        threshold_changes = self._autotune_thresholds()
        discoveries = self._regime_signal_learning()
        best_pt = self._landscape.best()
        best_params = best_pt.params if best_pt else dict(self._current_params)
        best_sharpe = best_pt.fitness if best_pt else 0.0
        summary = EvolutionSummary(
            cycle_id=self._cycle_count,
            pruned_templates=pruned,
            promoted_templates=promoted,
            weight_changes=weight_changes,
            threshold_changes=threshold_changes,
            best_params=best_params,
            best_sharpe=best_sharpe,
            regime_discoveries=discoveries)
        self._evolution_history.append(summary)
        return summary

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate_params(self, params: Optional[Dict[str, float]] = None,
                       sigma: float = 0.05) -> Dict[str, float]:
        """
        Apply small random mutations to params.
        Uses current best if params not provided.
        """
        rng = np.random.default_rng()
        base = params or self._current_params
        if not self.param_bounds:
            return {k: float(v + rng.normal(0, sigma * max(abs(v), 0.01)))
                    for k, v in base.items()}
        mutated = {}
        for k, v in base.items():
            lo, hi = self.param_bounds.get(k, (-math.inf, math.inf))
            delta = rng.normal(0, sigma * max(abs(v), 0.01))
            mutated[k] = float(np.clip(v + delta, lo, hi))
        return mutated

    def mutate_weights(self, sigma: float = 0.05) -> Dict[str, float]:
        """Generate a mutated version of current signal weights (normalised)."""
        rng = np.random.default_rng()
        raw = {k: max(0.0, v + rng.normal(0, sigma))
               for k, v in self._current_weights.items()}
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Suggest next parameters
    # ------------------------------------------------------------------

    def suggest_next_params(self) -> List[Dict[str, float]]:
        """
        Return a list of candidate parameter dicts for the next evaluation batch.
        Combines optimizer suggestions with mutations of current best.
        """
        suggestions = self._optimizer.suggest()
        # Add a mutation of current best for exploitation
        best = self._optimizer.best
        if best is not None:
            suggestions.append(self.mutate_params(best, sigma=0.03))
        return suggestions

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def evolution_report(self) -> Dict[str, Any]:
        """
        Full evolution report: weight history, parameter importance,
        template stats, regime meta-learning.
        """
        return {
            "run_count": self._run_count,
            "cycle_count": self._cycle_count,
            "current_weights": dict(self._current_weights),
            "current_params": dict(self._current_params),
            "best_fitness": self._optimizer.best_fitness,
            "best_params": self._optimizer.best,
            "template_summary": self._template_mgr.summary(),
            "param_importance": self._optimizer.param_importance(),
            "regime_best_signals": self._regime_signal_learning(),
            "regime_breakdown": self._attributor.regime_breakdown(),
            "weight_attribution": self._attributor.attribute_to_weights(),
            "recent_evolutions": [
                {"cycle": s.cycle_id, "pruned": s.pruned_templates,
                 "promoted": s.promoted_templates,
                 "best_sharpe": s.best_sharpe,
                 "regime_discoveries": s.regime_discoveries}
                for s in self._evolution_history[-5:]
            ],
        }

    def latest_evolution(self) -> Optional[EvolutionSummary]:
        return self._evolution_history[-1] if self._evolution_history else None

    def regime_affinity_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Return {regime: {signal: avg_sharpe}} as a meta-learning summary.
        """
        result: Dict[str, Dict[str, float]] = {}
        for regime, sig_dict in self._regime_signal_perf.items():
            result[regime] = {}
            for sig, history in sig_dict.items():
                if history:
                    result[regime][sig] = float(np.mean(list(history)))
        return result

    def pruned_signals(self) -> List[str]:
        return [n for n, t in self._template_mgr._templates.items() if t.pruned]

    def promoted_signals(self) -> List[str]:
        return [n for n, t in self._template_mgr._templates.items() if t.promoted]

    def fitness_top_k(self, k: int = 5) -> List[FitnessPoint]:
        return self._landscape.top_k(k)
