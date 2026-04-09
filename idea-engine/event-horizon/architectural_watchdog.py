"""
Architectural Watchdog: meta-meta monitor for the RMEA system.

Watches the recursive meta-evolutionary loop itself and detects when
the ENTIRE system is stagnating -- not just parameters, but the
architecture of discovery.

When stagnation is detected, triggers ARCHITECTURAL MUTATIONS:
  - Swap out the signal evaluation method
  - Change which agents participate in debates
  - Alter the fitness landscape topology
  - Inject entirely new physics domains
  - Reset the search space dimensionality

This is the immune system of the autonomous research system.
"""

from __future__ import annotations
import math
import time
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import numpy as np


@dataclass
class SystemHealth:
    """Health metrics for the entire RMEA system."""
    innovation_yield: float       # promoted / tested (0-1)
    search_entropy: float         # variance of fitness across population
    idea_diversity: float         # unique template types in active hypotheses
    alpha_decay_rate: float       # average IC decay across live signals
    debate_consensus_trend: float # is consensus getting stronger or weaker?
    meta_fitness_trend: float     # is the hyper-genome getting fitter?
    stagnation_score: float       # composite (0=healthy, 1=completely stuck)
    cycles_since_improvement: int
    total_cycles: int


@dataclass
class ArchitecturalMutation:
    """A structural change to the system (not just parameter tweaks)."""
    mutation_id: str
    name: str
    description: str
    target_component: str         # which subsystem to mutate
    mutation_type: str            # swap / inject / reset / prune / expand
    parameters: Dict = field(default_factory=dict)
    applied_at: float = 0.0
    outcome: str = "pending"      # pending / improved / neutral / degraded


# Architectural mutation library
ARCHITECTURAL_MUTATIONS = [
    ArchitecturalMutation(
        mutation_id="arch_001",
        name="Domain Injection: Topology",
        description="Inject topological data analysis (Betti numbers, persistence diagrams) "
                     "as a new signal domain. Forces the system to explore non-metric features.",
        target_component="signal_domains",
        mutation_type="inject",
        parameters={"new_domain": "topology", "weight": 0.3},
    ),
    ArchitecturalMutation(
        mutation_id="arch_002",
        name="Debate Agent Rotation",
        description="Remove the worst-performing debate agent and replace with a fresh "
                     "agent initialized with random weights. Prevents groupthink.",
        target_component="debate_system",
        mutation_type="swap",
        parameters={"remove_worst": True, "add_random": True},
    ),
    ArchitecturalMutation(
        mutation_id="arch_003",
        name="Fitness Landscape Inversion",
        description="Temporarily invert the fitness function: reward signals that LOSE money "
                     "in the current regime. Forces discovery of contrarian/hedging signals.",
        target_component="fitness_function",
        mutation_type="swap",
        parameters={"invert_fitness": True, "duration_generations": 5},
    ),
    ArchitecturalMutation(
        mutation_id="arch_004",
        name="Search Space Collapse",
        description="Reduce the parameter search space to the neighborhood of the current "
                     "best solution. Intensify exploitation when exploration is exhausted.",
        target_component="search_space",
        mutation_type="prune",
        parameters={"collapse_factor": 0.3, "center_on_best": True},
    ),
    ArchitecturalMutation(
        mutation_id="arch_005",
        name="Search Space Explosion",
        description="10x the parameter search space in all dimensions. Cambrian explosion "
                     "of signal variants. Use when the system has converged prematurely.",
        target_component="search_space",
        mutation_type="expand",
        parameters={"expansion_factor": 10.0},
    ),
    ArchitecturalMutation(
        mutation_id="arch_006",
        name="Evaluation Method Swap: IS to OOS",
        description="Switch from in-sample fitness to pure out-of-sample fitness. "
                     "If the system has been overfitting IS data, this breaks the cycle.",
        target_component="evaluation",
        mutation_type="swap",
        parameters={"eval_mode": "pure_oos", "oos_fraction": 0.4},
    ),
    ArchitecturalMutation(
        mutation_id="arch_007",
        name="Timeframe Shift",
        description="Shift the entire system from 15m bars to 4h bars (or vice versa). "
                     "Different timeframes reveal different physics.",
        target_component="data_pipeline",
        mutation_type="swap",
        parameters={"target_timeframe": "4h"},
    ),
    ArchitecturalMutation(
        mutation_id="arch_008",
        name="Cross-Asset Expansion",
        description="Add a new asset class to the universe (e.g., if only crypto, add equities). "
                     "New assets bring new correlation structures and physics.",
        target_component="universe",
        mutation_type="expand",
        parameters={"add_asset_class": "equities", "symbols": ["SPY", "QQQ", "GLD"]},
    ),
    ArchitecturalMutation(
        mutation_id="arch_009",
        name="Red Queen Intensification",
        description="Double the adversarial population and mutation rate. The Red Queen "
                     "becomes much harder to beat, forcing truly robust signals.",
        target_component="red_queen",
        mutation_type="expand",
        parameters={"population_multiplier": 2.0, "mutation_rate_boost": 1.5},
    ),
    ArchitecturalMutation(
        mutation_id="arch_010",
        name="Full System Reset (Nuclear Option)",
        description="Reset everything except the Hall of Fame. Re-initialize populations, "
                     "clear caches, start fresh. Only use when deeply stuck.",
        target_component="all",
        mutation_type="reset",
        parameters={"preserve_hall_of_fame": True, "preserve_best_n": 3},
    ),
]


class ArchitecturalWatchdog:
    """
    Monitors the health of the entire RMEA system and triggers
    architectural mutations when systemic stagnation is detected.

    This is NOT a parameter optimizer. It changes the STRUCTURE of
    the optimization process itself.
    """

    def __init__(self, patience: int = 10, critical_patience: int = 25):
        self.patience = patience              # generations before minor intervention
        self.critical_patience = critical_patience  # generations before major intervention
        self.rng = random.Random(42)

        self._fitness_history: deque = deque(maxlen=100)
        self._innovation_history: deque = deque(maxlen=100)
        self._entropy_history: deque = deque(maxlen=100)
        self._best_fitness: float = float("-inf")
        self._cycles_since_improvement: int = 0
        self._mutations_applied: List[ArchitecturalMutation] = []

    def update(
        self,
        generation: int,
        best_fitness: float,
        mean_fitness: float,
        innovation_yield: float,
        search_entropy: float,
        n_live_signals: int,
        n_retired_signals: int,
    ) -> Optional[ArchitecturalMutation]:
        """
        Update health metrics and decide if an architectural mutation is needed.

        Returns an ArchitecturalMutation if intervention is warranted, else None.
        """
        self._fitness_history.append(best_fitness)
        self._innovation_history.append(innovation_yield)
        self._entropy_history.append(search_entropy)

        # Track improvement
        if best_fitness > self._best_fitness + 0.001:
            self._best_fitness = best_fitness
            self._cycles_since_improvement = 0
        else:
            self._cycles_since_improvement += 1

        # Compute health
        health = self._compute_health(n_live_signals, n_retired_signals)

        # Decision: what level of intervention?
        if health.stagnation_score < 0.3:
            return None  # healthy, no intervention

        if self._cycles_since_improvement >= self.critical_patience:
            # Critical stagnation: major architectural mutation
            mutation = self._select_mutation(severity="critical")
        elif self._cycles_since_improvement >= self.patience:
            # Moderate stagnation: minor architectural mutation
            mutation = self._select_mutation(severity="moderate")
        elif health.stagnation_score > 0.7:
            # High stagnation score even if recent improvement
            mutation = self._select_mutation(severity="moderate")
        else:
            return None

        mutation.applied_at = time.time()
        self._mutations_applied.append(mutation)
        return mutation

    def _compute_health(self, n_live: int, n_retired: int) -> SystemHealth:
        """Compute system health from accumulated metrics."""
        n = len(self._fitness_history)

        # Innovation yield
        recent_innovation = list(self._innovation_history)[-10:] if self._innovation_history else [0]
        avg_innovation = float(np.mean(recent_innovation))

        # Search entropy (population diversity)
        recent_entropy = list(self._entropy_history)[-10:] if self._entropy_history else [0]
        avg_entropy = float(np.mean(recent_entropy))

        # Fitness trend
        if n >= 5:
            recent_fit = list(self._fitness_history)[-10:]
            fit_trend = float(np.polyfit(range(len(recent_fit)), recent_fit, 1)[0])
        else:
            fit_trend = 0.0

        # Stagnation composite
        stagnation = 0.0
        stagnation += max(0, 0.3 - avg_innovation) * 2   # low innovation
        stagnation += max(0, 0.1 - avg_entropy) * 3       # low diversity
        stagnation += max(0, -fit_trend) * 10              # declining fitness
        stagnation += min(self._cycles_since_improvement / max(self.critical_patience, 1), 1.0) * 0.3
        stagnation = min(1.0, stagnation)

        return SystemHealth(
            innovation_yield=avg_innovation,
            search_entropy=avg_entropy,
            idea_diversity=float(n_live),
            alpha_decay_rate=float(n_retired / max(n_live + n_retired, 1)),
            debate_consensus_trend=0.0,
            meta_fitness_trend=fit_trend,
            stagnation_score=stagnation,
            cycles_since_improvement=self._cycles_since_improvement,
            total_cycles=n,
        )

    def _select_mutation(self, severity: str) -> ArchitecturalMutation:
        """Select an appropriate architectural mutation based on severity."""
        if severity == "critical":
            # Heavy interventions
            candidates = [m for m in ARCHITECTURAL_MUTATIONS
                          if m.mutation_type in ("reset", "expand", "swap")]
        else:
            # Lighter interventions
            candidates = [m for m in ARCHITECTURAL_MUTATIONS
                          if m.mutation_type in ("inject", "swap", "prune")]

        # Avoid repeating recent mutations
        recent_ids = {m.mutation_id for m in self._mutations_applied[-5:]}
        candidates = [m for m in candidates if m.mutation_id not in recent_ids]

        if not candidates:
            candidates = ARCHITECTURAL_MUTATIONS  # fallback to any

        import copy
        return copy.deepcopy(self.rng.choice(candidates))

    def record_outcome(self, mutation_id: str, outcome: str) -> None:
        """Record whether a mutation helped, was neutral, or degraded performance."""
        for m in self._mutations_applied:
            if m.mutation_id == mutation_id:
                m.outcome = outcome

    def get_report(self) -> Dict:
        """System health report."""
        health = self._compute_health(0, 0)
        return {
            "stagnation_score": health.stagnation_score,
            "cycles_since_improvement": self._cycles_since_improvement,
            "innovation_yield": health.innovation_yield,
            "search_entropy": health.search_entropy,
            "fitness_trend": health.meta_fitness_trend,
            "mutations_applied": len(self._mutations_applied),
            "recent_mutations": [
                {"name": m.name, "type": m.mutation_type, "outcome": m.outcome}
                for m in self._mutations_applied[-5:]
            ],
        }
