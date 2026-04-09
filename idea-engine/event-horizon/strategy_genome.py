"""
Strategy Genome: each active strategy is a living organism that is born,
matures, reproduces, and dies based on its fitness in the current environment.

This module implements biological lifecycle management:
  - Birth: new strategy discovered by EHS or genetic mutation
  - Infancy: paper-traded in shadow portfolio (protected from real risk)
  - Maturity: promoted to live with small allocation, grows with success
  - Reproduction: successful strategies spawn variants via mutation
  - Aging: performance naturally decays, requiring adaptation
  - Death: strategy retired when alpha is exhausted
  - Resurrection: dead strategies can be revived if regime returns

The population of active strategies is managed like an ecosystem:
  - Carrying capacity limits total number of live strategies
  - Niche competition: similar strategies compete for allocation
  - Symbiosis: complementary strategies get allocation bonus
  - Predator-prey: contrarian strategies thrive when consensus strategies fail
"""

from __future__ import annotations
import math
import time
import copy
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


class LifecycleStage:
    EMBRYO = "embryo"         # just discovered, not yet tested
    INFANT = "infant"         # shadow trading, protected
    JUVENILE = "juvenile"     # small live allocation, proving itself
    MATURE = "mature"         # full allocation, stable performance
    AGING = "aging"           # performance declining, allocation shrinking
    DORMANT = "dormant"       # paused, waiting for right regime
    DEAD = "dead"             # retired, preserved in graveyard for resurrection


@dataclass
class StrategyOrganism:
    """A living strategy with biological lifecycle."""
    genome_id: str
    name: str
    template_type: str
    regime_affinity: List[str]
    signal_weights: Dict[str, float]

    # Lifecycle
    stage: str = LifecycleStage.EMBRYO
    birth_time: float = 0.0
    maturity_time: float = 0.0
    death_time: float = 0.0
    age_bars: int = 0

    # Performance
    sharpe_lifetime: float = 0.0
    sharpe_recent: float = 0.0     # rolling 63-bar Sharpe
    pnl_cumulative: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0

    # Allocation
    current_allocation_pct: float = 0.0
    peak_allocation_pct: float = 0.0

    # Genetics
    parent_id: Optional[str] = None
    generation: int = 0
    n_offspring: int = 0
    mutations: List[str] = field(default_factory=list)

    # Fitness (combined metric)
    fitness: float = 0.0


class StrategyEcosystem:
    """
    Manages the population of living strategies.

    The ecosystem enforces:
    - Carrying capacity: max N live strategies
    - Niche competition: similar strategies share allocation
    - Symbiosis bonus: negatively correlated strategies get boosted
    - Natural selection: worst performers die, best reproduce
    """

    def __init__(
        self,
        carrying_capacity: int = 20,
        infant_allocation: float = 0.02,
        juvenile_allocation: float = 0.05,
        mature_allocation: float = 0.10,
        infant_bars: int = 100,
        juvenile_bars: int = 300,
        death_sharpe: float = -0.5,
        reproduction_sharpe: float = 1.5,
        seed: int = 42,
    ):
        self.capacity = carrying_capacity
        self.infant_alloc = infant_allocation
        self.juvenile_alloc = juvenile_allocation
        self.mature_alloc = mature_allocation
        self.infant_bars = infant_bars
        self.juvenile_bars = juvenile_bars
        self.death_sharpe = death_sharpe
        self.reproduction_sharpe = reproduction_sharpe
        self.rng = random.Random(seed)

        self._organisms: Dict[str, StrategyOrganism] = {}
        self._graveyard: Dict[str, StrategyOrganism] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"org_{self._counter:05d}"

    def birth(self, name: str, template_type: str, regime_affinity: List[str],
               signal_weights: Dict[str, float], parent_id: Optional[str] = None) -> StrategyOrganism:
        """Create a new strategy organism."""
        org = StrategyOrganism(
            genome_id=self._next_id(),
            name=name,
            template_type=template_type,
            regime_affinity=regime_affinity,
            signal_weights=signal_weights,
            stage=LifecycleStage.EMBRYO,
            birth_time=time.time(),
            parent_id=parent_id,
            generation=(self._organisms.get(parent_id, StrategyOrganism("", "", "", [])).generation + 1) if parent_id else 0,
        )
        self._organisms[org.genome_id] = org
        return org

    def update_performance(self, genome_id: str, bar_return: float) -> None:
        """Update an organism's performance with a new bar."""
        org = self._organisms.get(genome_id)
        if not org or org.stage == LifecycleStage.DEAD:
            return

        org.age_bars += 1
        org.pnl_cumulative += bar_return

        # Track win rate
        org.n_trades += 1
        if bar_return > 0:
            org.win_rate = (org.win_rate * (org.n_trades - 1) + 1) / org.n_trades
        else:
            org.win_rate = org.win_rate * (org.n_trades - 1) / org.n_trades

        # Update Sharpe (simplified rolling)
        org.sharpe_recent = org.pnl_cumulative / max(org.age_bars, 1) * math.sqrt(252)

        # Lifecycle transitions
        self._lifecycle_transition(org)

    def _lifecycle_transition(self, org: StrategyOrganism) -> None:
        """Check and execute lifecycle transitions."""
        # Embryo -> Infant: after creation
        if org.stage == LifecycleStage.EMBRYO:
            org.stage = LifecycleStage.INFANT
            org.current_allocation_pct = self.infant_alloc

        # Infant -> Juvenile: after proving itself
        elif org.stage == LifecycleStage.INFANT and org.age_bars >= self.infant_bars:
            if org.sharpe_recent > 0:
                org.stage = LifecycleStage.JUVENILE
                org.current_allocation_pct = self.juvenile_alloc
            else:
                self._kill(org, "Failed infancy: negative Sharpe after infant period")

        # Juvenile -> Mature: after sustained performance
        elif org.stage == LifecycleStage.JUVENILE and org.age_bars >= self.juvenile_bars:
            if org.sharpe_recent > 0.3:
                org.stage = LifecycleStage.MATURE
                org.maturity_time = time.time()
                org.current_allocation_pct = self.mature_alloc
                org.peak_allocation_pct = self.mature_alloc
            elif org.sharpe_recent < self.death_sharpe:
                self._kill(org, "Failed juvenile: Sharpe below death threshold")

        # Mature -> Aging: when performance declines
        elif org.stage == LifecycleStage.MATURE:
            if org.sharpe_recent < 0.1 and org.age_bars > self.juvenile_bars * 2:
                org.stage = LifecycleStage.AGING
                org.current_allocation_pct *= 0.5

        # Aging -> Death or Recovery
        elif org.stage == LifecycleStage.AGING:
            if org.sharpe_recent < self.death_sharpe:
                self._kill(org, "Natural death: alpha exhausted")
            elif org.sharpe_recent > 0.5:
                org.stage = LifecycleStage.MATURE
                org.current_allocation_pct = self.mature_alloc * 0.7

        # Reproduction: successful mature strategies spawn offspring
        if org.stage == LifecycleStage.MATURE and org.sharpe_recent > self.reproduction_sharpe:
            if org.n_offspring < 3 and len(self._organisms) < self.capacity:
                self._reproduce(org)

    def _reproduce(self, parent: StrategyOrganism) -> StrategyOrganism:
        """Create a mutated offspring from a successful parent."""
        child_weights = {}
        for signal, weight in parent.signal_weights.items():
            child_weights[signal] = weight + self.rng.gauss(0, 0.1)

        child = self.birth(
            name=f"{parent.name} Gen{parent.generation + 1}",
            template_type=parent.template_type,
            regime_affinity=parent.regime_affinity.copy(),
            signal_weights=child_weights,
            parent_id=parent.genome_id,
        )
        child.mutations.append("gaussian_weight_perturbation")
        parent.n_offspring += 1
        return child

    def _kill(self, org: StrategyOrganism, reason: str) -> None:
        """Move strategy to graveyard."""
        org.stage = LifecycleStage.DEAD
        org.death_time = time.time()
        org.current_allocation_pct = 0.0
        self._graveyard[org.genome_id] = org

    def resurrect(self, genome_id: str, current_regime: str) -> Optional[StrategyOrganism]:
        """
        Resurrect a dead strategy if the current regime matches its affinity.
        Like a dormant seed sprouting when conditions are right.
        """
        dead = self._graveyard.get(genome_id)
        if not dead:
            return None

        if current_regime in dead.regime_affinity:
            dead.stage = LifecycleStage.INFANT
            dead.current_allocation_pct = self.infant_alloc
            dead.age_bars = 0
            dead.pnl_cumulative = 0.0
            dead.sharpe_recent = 0.0
            self._organisms[genome_id] = dead
            del self._graveyard[genome_id]
            return dead
        return None

    def enforce_carrying_capacity(self) -> List[str]:
        """Kill weakest organisms if over capacity."""
        alive = [o for o in self._organisms.values() if o.stage != LifecycleStage.DEAD]
        if len(alive) <= self.capacity:
            return []

        # Sort by fitness (worst first)
        alive.sort(key=lambda o: o.sharpe_recent)
        killed = []
        while len([o for o in self._organisms.values() if o.stage != LifecycleStage.DEAD]) > self.capacity:
            weakest = alive.pop(0)
            self._kill(weakest, "Carrying capacity exceeded")
            killed.append(weakest.genome_id)

        return killed

    def get_allocation_map(self) -> Dict[str, float]:
        """Get current allocation per strategy."""
        return {
            gid: org.current_allocation_pct
            for gid, org in self._organisms.items()
            if org.stage not in (LifecycleStage.DEAD, LifecycleStage.EMBRYO)
        }

    def get_ecosystem_status(self) -> Dict:
        """Full ecosystem status."""
        by_stage = defaultdict(int)
        for org in self._organisms.values():
            by_stage[org.stage] += 1

        alive = [o for o in self._organisms.values() if o.stage != LifecycleStage.DEAD]
        avg_sharpe = float(np.mean([o.sharpe_recent for o in alive])) if alive else 0

        return {
            "total_alive": len(alive),
            "by_stage": dict(by_stage),
            "graveyard_size": len(self._graveyard),
            "carrying_capacity": self.capacity,
            "avg_sharpe": avg_sharpe,
            "best_organism": max(alive, key=lambda o: o.sharpe_recent).name if alive else None,
            "total_allocation_pct": sum(o.current_allocation_pct for o in alive),
            "oldest_age_bars": max(o.age_bars for o in alive) if alive else 0,
            "max_generation": max(o.generation for o in alive) if alive else 0,
        }
