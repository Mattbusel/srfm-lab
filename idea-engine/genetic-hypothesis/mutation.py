"""
Mutation operators for genetic hypothesis evolution — applies controlled
perturbations to hypothesis genotypes to explore the search space.

Operators:
  - ParameterMutation: Gaussian perturbation of numeric parameters
  - StructuralMutation: swap signal components (e.g., RSI ↔ MACD)
  - RegimeMutation: change which regimes activate the hypothesis
  - TimescaleMutation: shift lookback windows
  - EntryExitMutation: modify entry/exit thresholds
  - InversionMutation: flip long/short direction
  - HybridMutation: combine two mutation types
  - AdaptiveMutationRate: increase rate when fitness stagnates
  - MutationHistory: track which mutations improved fitness
  - CrowdingDistance: maintain diversity via NSGA-II-style crowding
"""

from __future__ import annotations
import math
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Hypothesis gene representation
# ---------------------------------------------------------------------------

@dataclass
class HypothesisGene:
    """A single hypothesis genotype — encodes a complete trading strategy."""
    gene_id: str
    parameters: Dict[str, float]          # numeric params (e.g. rsi_period=14)
    signal_components: List[str]          # e.g. ["rsi", "macd", "bbands"]
    active_regimes: List[str]             # e.g. ["risk_on", "recovery"]
    lookback_windows: Dict[str, int]      # signal -> lookback days
    entry_threshold: float = 0.6
    exit_threshold: float = 0.4
    direction: int = 1                    # 1 = long, -1 = short
    fitness: float = 0.0
    generation: int = 0
    mutation_history: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)

    def clone(self) -> HypothesisGene:
        """Deep copy this gene."""
        g = copy.deepcopy(self)
        return g


# ---------------------------------------------------------------------------
# Base mutation operator
# ---------------------------------------------------------------------------

class MutationOperator:
    """Abstract base for mutation operators."""

    def __init__(self, name: str, probability: float = 0.2, seed: int = 42):
        self.name = name
        self.probability = probability
        self.rng = np.random.default_rng(seed)

    def should_mutate(self) -> bool:
        return self.rng.random() < self.probability

    def mutate(self, gene: HypothesisGene) -> HypothesisGene:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(prob={self.probability:.2f})"


# ---------------------------------------------------------------------------
# Parameter mutation — Gaussian perturbation
# ---------------------------------------------------------------------------

class ParameterMutation(MutationOperator):
    """Perturb numeric parameters with Gaussian noise."""

    def __init__(self, sigma_frac: float = 0.1, probability: float = 0.3,
                 param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 seed: int = 42):
        super().__init__("parameter_mutation", probability, seed)
        self.sigma_frac = sigma_frac
        self.param_bounds = param_bounds or {}

    def mutate(self, gene: HypothesisGene) -> HypothesisGene:
        child = gene.clone()
        if not child.parameters:
            return child

        # Pick a random subset of parameters to mutate
        params = list(child.parameters.keys())
        n_mutate = max(1, int(len(params) * 0.3))
        chosen = self.rng.choice(params, size=min(n_mutate, len(params)), replace=False)

        for pname in chosen:
            old_val = child.parameters[pname]
            sigma = abs(old_val) * self.sigma_frac if abs(old_val) > 1e-9 else self.sigma_frac
            new_val = old_val + self.rng.normal(0, sigma)
            # Clip to bounds if available
            if pname in self.param_bounds:
                lo, hi = self.param_bounds[pname]
                new_val = max(lo, min(hi, new_val))
            child.parameters[pname] = new_val

        child.mutation_history.append(f"param_mut({list(chosen)})")
        return child


# ---------------------------------------------------------------------------
# Structural mutation — swap signal components
# ---------------------------------------------------------------------------

AVAILABLE_SIGNALS = [
    "rsi", "macd", "bbands", "stochastic", "cci", "williams_r",
    "adx", "atr", "obv", "vwap", "ema_cross", "sma_cross",
    "ichimoku", "parabolic_sar", "keltner", "donchian",
    "momentum", "roc", "trix", "aroon",
]


class StructuralMutation(MutationOperator):
    """Swap or add/remove signal components in the hypothesis."""

    def __init__(self, available_signals: Optional[List[str]] = None,
                 probability: float = 0.15, max_components: int = 5,
                 seed: int = 42):
        super().__init__("structural_mutation", probability, seed)
        self.available_signals = available_signals or list(AVAILABLE_SIGNALS)
        self.max_components = max_components

    def mutate(self, gene: HypothesisGene) -> HypothesisGene:
        child = gene.clone()
        action = self.rng.choice(["swap", "add", "remove"])

        if action == "swap" and child.signal_components:
            idx = int(self.rng.integers(0, len(child.signal_components)))
            candidates = [s for s in self.available_signals
                          if s not in child.signal_components]
            if candidates:
                old = child.signal_components[idx]
                new = str(self.rng.choice(candidates))
                child.signal_components[idx] = new
                child.mutation_history.append(f"struct_swap({old}->{new})")
                # Add default lookback for new signal
                if new not in child.lookback_windows:
                    child.lookback_windows[new] = int(self.rng.integers(5, 50))
                # Remove old lookback
                child.lookback_windows.pop(old, None)

        elif action == "add" and len(child.signal_components) < self.max_components:
            candidates = [s for s in self.available_signals
                          if s not in child.signal_components]
            if candidates:
                new = str(self.rng.choice(candidates))
                child.signal_components.append(new)
                child.lookback_windows[new] = int(self.rng.integers(5, 50))
                child.mutation_history.append(f"struct_add({new})")

        elif action == "remove" and len(child.signal_components) > 1:
            idx = int(self.rng.integers(0, len(child.signal_components)))
            removed = child.signal_components.pop(idx)
            child.lookback_windows.pop(removed, None)
            child.mutation_history.append(f"struct_remove({removed})")

        return child


# ---------------------------------------------------------------------------
# Regime mutation — change active regimes
# ---------------------------------------------------------------------------

ALL_REGIMES = ["risk_on", "risk_off", "crisis", "recovery", "low_vol_grind"]


class RegimeMutation(MutationOperator):
    """Toggle which regimes activate the hypothesis."""

    def __init__(self, all_regimes: Optional[List[str]] = None,
                 probability: float = 0.15, seed: int = 42):
        super().__init__("regime_mutation", probability, seed)
        self.all_regimes = all_regimes or list(ALL_REGIMES)

    def mutate(self, gene: HypothesisGene) -> HypothesisGene:
        child = gene.clone()
        action = self.rng.choice(["toggle", "replace_all"])

        if action == "toggle":
            regime = str(self.rng.choice(self.all_regimes))
            if regime in child.active_regimes:
                if len(child.active_regimes) > 1:
                    child.active_regimes.remove(regime)
                    child.mutation_history.append(f"regime_off({regime})")
            else:
                child.active_regimes.append(regime)
                child.mutation_history.append(f"regime_on({regime})")

        elif action == "replace_all":
            n = int(self.rng.integers(1, len(self.all_regimes) + 1))
            child.active_regimes = list(self.rng.choice(
                self.all_regimes, size=n, replace=False))
            child.mutation_history.append(f"regime_replace({child.active_regimes})")

        return child


# ---------------------------------------------------------------------------
# Timescale mutation — shift lookback windows
# ---------------------------------------------------------------------------

class TimescaleMutation(MutationOperator):
    """Shift lookback windows up or down."""

    def __init__(self, min_lookback: int = 2, max_lookback: int = 200,
                 shift_factor: float = 0.2, probability: float = 0.2,
                 seed: int = 42):
        super().__init__("timescale_mutation", probability, seed)
        self.min_lookback = min_lookback
        self.max_lookback = max_lookback
        self.shift_factor = shift_factor

    def mutate(self, gene: HypothesisGene) -> HypothesisGene:
        child = gene.clone()
        if not child.lookback_windows:
            return child

        signals = list(child.lookback_windows.keys())
        target = str(self.rng.choice(signals))
        old_val = child.lookback_windows[target]

        # Multiplicative shift
        multiplier = 1.0 + self.rng.normal(0, self.shift_factor)
        multiplier = max(0.3, min(3.0, multiplier))
        new_val = int(round(old_val * multiplier))
        new_val = max(self.min_lookback, min(self.max_lookback, new_val))

        child.lookback_windows[target] = new_val
        child.mutation_history.append(f"timescale({target}:{old_val}->{new_val})")
        return child


# ---------------------------------------------------------------------------
# Entry/Exit mutation — modify thresholds
# ---------------------------------------------------------------------------

class EntryExitMutation(MutationOperator):
    """Modify entry and exit thresholds."""

    def __init__(self, sigma: float = 0.05, probability: float = 0.2,
                 seed: int = 42):
        super().__init__("entry_exit_mutation", probability, seed)
        self.sigma = sigma

    def mutate(self, gene: HypothesisGene) -> HypothesisGene:
        child = gene.clone()
        which = self.rng.choice(["entry", "exit", "both"])

        if which in ("entry", "both"):
            old = child.entry_threshold
            new = old + self.rng.normal(0, self.sigma)
            child.entry_threshold = max(0.0, min(1.0, new))

        if which in ("exit", "both"):
            old = child.exit_threshold
            new = old + self.rng.normal(0, self.sigma)
            child.exit_threshold = max(0.0, min(1.0, new))

        # Ensure entry > exit
        if child.entry_threshold < child.exit_threshold:
            child.entry_threshold, child.exit_threshold = (
                child.exit_threshold, child.entry_threshold)

        child.mutation_history.append(
            f"threshold(entry={child.entry_threshold:.3f},"
            f"exit={child.exit_threshold:.3f})")
        return child


# ---------------------------------------------------------------------------
# Inversion mutation — flip long/short
# ---------------------------------------------------------------------------

class InversionMutation(MutationOperator):
    """Flip the direction of the hypothesis (long ↔ short)."""

    def __init__(self, probability: float = 0.05, seed: int = 42):
        super().__init__("inversion_mutation", probability, seed)

    def mutate(self, gene: HypothesisGene) -> HypothesisGene:
        child = gene.clone()
        child.direction *= -1
        child.mutation_history.append(
            f"inversion({'long' if child.direction == 1 else 'short'})")
        return child


# ---------------------------------------------------------------------------
# Hybrid mutation — combine two operators
# ---------------------------------------------------------------------------

class HybridMutation(MutationOperator):
    """Apply two mutation operators sequentially."""

    def __init__(self, op_a: MutationOperator, op_b: MutationOperator,
                 probability: float = 0.1, seed: int = 42):
        super().__init__("hybrid_mutation", probability, seed)
        self.op_a = op_a
        self.op_b = op_b

    def mutate(self, gene: HypothesisGene) -> HypothesisGene:
        child = self.op_a.mutate(gene)
        child = self.op_b.mutate(child)
        child.mutation_history.append(
            f"hybrid({self.op_a.name}+{self.op_b.name})")
        return child


# ---------------------------------------------------------------------------
# Composite mutation pipeline
# ---------------------------------------------------------------------------

class MutationPipeline:
    """Apply a sequence of mutation operators, each with its own probability."""

    def __init__(self, operators: Optional[List[MutationOperator]] = None,
                 seed: int = 42):
        self.operators = operators or self._default_operators(seed)
        self.rng = np.random.default_rng(seed)
        self._call_count = 0
        self._mutation_counts: Dict[str, int] = {}

    @staticmethod
    def _default_operators(seed: int) -> List[MutationOperator]:
        return [
            ParameterMutation(probability=0.30, seed=seed),
            StructuralMutation(probability=0.15, seed=seed + 1),
            RegimeMutation(probability=0.15, seed=seed + 2),
            TimescaleMutation(probability=0.20, seed=seed + 3),
            EntryExitMutation(probability=0.20, seed=seed + 4),
            InversionMutation(probability=0.05, seed=seed + 5),
        ]

    def mutate(self, gene: HypothesisGene) -> HypothesisGene:
        """Apply all operators that fire based on their probability."""
        child = gene.clone()
        any_mutated = False
        for op in self.operators:
            if op.should_mutate():
                child = op.mutate(child)
                self._mutation_counts[op.name] = (
                    self._mutation_counts.get(op.name, 0) + 1)
                any_mutated = True
        self._call_count += 1

        # If nothing fired, force at least one mutation
        if not any_mutated and self.operators:
            op = self.operators[int(self.rng.integers(0, len(self.operators)))]
            child = op.mutate(child)
            self._mutation_counts[op.name] = (
                self._mutation_counts.get(op.name, 0) + 1)

        return child

    def stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self._call_count,
            "mutation_counts": dict(self._mutation_counts),
        }


# ---------------------------------------------------------------------------
# Adaptive mutation rate
# ---------------------------------------------------------------------------

class AdaptiveMutationRate:
    """
    Increase mutation rate when fitness stagnates, decrease when improving.

    Uses a sliding window of best fitness values to detect stagnation.
    """

    def __init__(self, base_rate: float = 0.15, min_rate: float = 0.05,
                 max_rate: float = 0.60, window_size: int = 10,
                 stagnation_threshold: float = 0.001,
                 increase_factor: float = 1.3, decrease_factor: float = 0.85):
        self.base_rate = base_rate
        self.current_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.window_size = window_size
        self.stagnation_threshold = stagnation_threshold
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.fitness_history: List[float] = []
        self.rate_history: List[float] = [base_rate]

    def update(self, best_fitness: float) -> float:
        """Update the mutation rate based on latest best fitness."""
        self.fitness_history.append(best_fitness)

        if len(self.fitness_history) < self.window_size:
            return self.current_rate

        window = self.fitness_history[-self.window_size:]
        improvement = abs(window[-1] - window[0])

        if improvement < self.stagnation_threshold:
            # Stagnating → increase mutation rate
            self.current_rate = min(
                self.current_rate * self.increase_factor, self.max_rate)
        else:
            # Improving → decrease mutation rate (exploit more)
            self.current_rate = max(
                self.current_rate * self.decrease_factor, self.min_rate)

        self.rate_history.append(self.current_rate)
        return self.current_rate

    def apply_to_pipeline(self, pipeline: MutationPipeline) -> None:
        """Scale all operator probabilities by the current rate ratio."""
        ratio = self.current_rate / self.base_rate
        for op in pipeline.operators:
            op.probability = min(op.probability * ratio, 0.95)

    def is_stagnating(self) -> bool:
        """Check if we're currently in a stagnation phase."""
        if len(self.fitness_history) < self.window_size:
            return False
        window = self.fitness_history[-self.window_size:]
        return abs(window[-1] - window[0]) < self.stagnation_threshold

    def reset(self) -> None:
        """Reset to base rate."""
        self.current_rate = self.base_rate
        self.fitness_history.clear()
        self.rate_history = [self.base_rate]


# ---------------------------------------------------------------------------
# Mutation history tracker
# ---------------------------------------------------------------------------

class MutationHistory:
    """Track which mutations improved fitness and which didn't."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self._by_operator: Dict[str, List[bool]] = {}

    def record(self, operator_name: str, parent_fitness: float,
               child_fitness: float) -> None:
        """Record a mutation outcome."""
        improved = child_fitness > parent_fitness
        self.records.append({
            "operator": operator_name,
            "parent_fitness": parent_fitness,
            "child_fitness": child_fitness,
            "delta": child_fitness - parent_fitness,
            "improved": improved,
        })
        if operator_name not in self._by_operator:
            self._by_operator[operator_name] = []
        self._by_operator[operator_name].append(improved)

    def success_rate(self, operator_name: str) -> float:
        """Fraction of times this operator improved fitness."""
        outcomes = self._by_operator.get(operator_name, [])
        if not outcomes:
            return 0.0
        return sum(outcomes) / len(outcomes)

    def all_success_rates(self) -> Dict[str, float]:
        return {name: self.success_rate(name) for name in self._by_operator}

    def best_operators(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """Return operators ranked by success rate."""
        rates = self.all_success_rates()
        ranked = sorted(rates.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def average_delta(self, operator_name: str) -> float:
        """Average fitness change from this operator."""
        deltas = [r["delta"] for r in self.records if r["operator"] == operator_name]
        return sum(deltas) / len(deltas) if deltas else 0.0

    def total_records(self) -> int:
        return len(self.records)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_mutations": self.total_records(),
            "success_rates": self.all_success_rates(),
            "avg_deltas": {name: self.average_delta(name)
                           for name in self._by_operator},
        }


# ---------------------------------------------------------------------------
# Crowding distance — NSGA-II-style diversity maintenance
# ---------------------------------------------------------------------------

class CrowdingDistance:
    """
    Compute NSGA-II crowding distance for a set of individuals.

    Individuals on the boundary of the Pareto front get infinite distance.
    Others get distance proportional to the "cuboid" they occupy in
    objective space.
    """

    def __init__(self, objectives: List[str]):
        """objectives: list of fitness attribute names."""
        self.objectives = objectives

    def compute(self, population: List[HypothesisGene],
                fitness_values: Dict[str, List[float]]) -> np.ndarray:
        """
        Compute crowding distances for the population.

        fitness_values: dict mapping objective name -> list of values
                        (one per individual, same order as population).

        Returns array of crowding distances (same order as population).
        """
        n = len(population)
        if n <= 2:
            return np.full(n, float("inf"))

        distances = np.zeros(n)

        for obj in self.objectives:
            values = np.array(fitness_values[obj])
            sorted_indices = np.argsort(values)
            obj_min = values[sorted_indices[0]]
            obj_max = values[sorted_indices[-1]]
            spread = obj_max - obj_min

            # Boundary individuals get infinite distance
            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            if spread < 1e-12:
                continue

            for i in range(1, n - 1):
                idx = sorted_indices[i]
                prev_val = values[sorted_indices[i - 1]]
                next_val = values[sorted_indices[i + 1]]
                distances[idx] += (next_val - prev_val) / spread

        return distances

    def select_by_crowding(self, population: List[HypothesisGene],
                           fitness_values: Dict[str, List[float]],
                           n_select: int) -> List[int]:
        """Select n_select indices by highest crowding distance."""
        distances = self.compute(population, fitness_values)
        ranked = np.argsort(-distances)  # descending
        return ranked[:n_select].tolist()

    def tournament_with_crowding(self, population: List[HypothesisGene],
                                 fitness_values: Dict[str, List[float]],
                                 pareto_ranks: List[int],
                                 tournament_size: int = 2,
                                 rng: Optional[np.random.Generator] = None
                                 ) -> int:
        """
        Tournament selection using Pareto rank + crowding distance.

        Lower rank wins. If tied, higher crowding distance wins.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        distances = self.compute(population, fitness_values)
        n = len(population)
        candidates = rng.choice(n, size=min(tournament_size, n), replace=False)

        best = candidates[0]
        for c in candidates[1:]:
            if pareto_ranks[c] < pareto_ranks[best]:
                best = c
            elif pareto_ranks[c] == pareto_ranks[best]:
                if distances[c] > distances[best]:
                    best = c
        return int(best)


# ---------------------------------------------------------------------------
# Convenience: create a full default mutation suite
# ---------------------------------------------------------------------------

def default_mutation_suite(seed: int = 42) -> Dict[str, Any]:
    """Create all mutation components with sensible defaults."""
    pipeline = MutationPipeline(seed=seed)
    adaptive = AdaptiveMutationRate()
    history = MutationHistory()
    crowding = CrowdingDistance(objectives=["sharpe", "calmar", "win_rate"])
    return {
        "pipeline": pipeline,
        "adaptive_rate": adaptive,
        "history": history,
        "crowding": crowding,
    }
