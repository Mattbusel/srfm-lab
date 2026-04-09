"""
Alpha Allocator: dynamically allocate capital to alpha sources based on
their real-time quality, decay rate, and capacity.

Instead of fixed signal weights, this module:
1. Tracks the IC (information coefficient) of each signal in real-time
2. Estimates each signal's capacity (how much can you trade before impact erodes alpha)
3. Allocates capital to signals using a Kelly-like criterion
4. Automatically reduces allocation to decaying signals
5. Redirects capital from dying signals to newly discovered ones
6. Maintains a "bench" of signals waiting for allocation

Think of this as a venture capital fund for trading signals:
  - Each signal is a "startup" competing for capital
  - Successful signals get more capital (Series B, C)
  - Failing signals get cut (bridge to nowhere)
  - New signals start with seed allocation
"""

from __future__ import annotations
import math
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class AlphaSource:
    """A single source of alpha (a signal or strategy)."""
    source_id: str
    name: str
    source_type: str              # "physics" / "microstructure" / "macro" / "ml" / "alternative"

    # Real-time quality
    ic_rolling_21d: float = 0.0   # rolling 21-day information coefficient
    ic_rolling_63d: float = 0.0   # rolling 63-day IC
    ic_trend: float = 0.0         # is IC improving or declining?
    sharpe_contribution: float = 0.0  # marginal Sharpe contribution to portfolio

    # Capacity
    estimated_capacity_usd: float = 1e6  # how much can trade on this signal
    current_allocation_usd: float = 0.0
    utilization_pct: float = 0.0  # current / capacity

    # Lifecycle
    status: str = "bench"         # "bench" / "seed" / "growth" / "mature" / "declining" / "retired"
    allocation_pct: float = 0.0   # fraction of total capital
    days_active: int = 0

    # History
    ic_history: List[float] = field(default_factory=list)
    pnl_history: List[float] = field(default_factory=list)


class AlphaAllocator:
    """
    Dynamic capital allocation across alpha sources.

    Like a VC fund: signals compete for capital based on quality.
    """

    def __init__(self, total_capital: float = 1_000_000,
                  max_per_source_pct: float = 0.20,
                  seed_allocation_pct: float = 0.02,
                  min_ic_for_growth: float = 0.03,
                  min_ic_for_maintain: float = 0.01):
        self.total_capital = total_capital
        self.max_per_source = max_per_source_pct
        self.seed_alloc = seed_allocation_pct
        self.min_ic_growth = min_ic_for_growth
        self.min_ic_maintain = min_ic_for_maintain

        self._sources: Dict[str, AlphaSource] = {}
        self._total_allocated: float = 0.0

    def register_source(self, source_id: str, name: str,
                         source_type: str) -> AlphaSource:
        """Register a new alpha source (starts on the bench)."""
        source = AlphaSource(
            source_id=source_id,
            name=name,
            source_type=source_type,
            status="bench",
        )
        self._sources[source_id] = source
        return source

    def update_quality(self, source_id: str, ic: float, pnl: float) -> None:
        """Update a source's quality metrics."""
        source = self._sources.get(source_id)
        if not source:
            return

        source.ic_history.append(ic)
        source.pnl_history.append(pnl)
        source.days_active += 1

        # Rolling IC
        if len(source.ic_history) >= 21:
            source.ic_rolling_21d = float(np.mean(source.ic_history[-21:]))
        if len(source.ic_history) >= 63:
            source.ic_rolling_63d = float(np.mean(source.ic_history[-63:]))

        # IC trend
        if len(source.ic_history) >= 10:
            recent = source.ic_history[-10:]
            source.ic_trend = float(np.polyfit(range(len(recent)), recent, 1)[0])

    def reallocate(self) -> Dict[str, float]:
        """
        Run the full reallocation.

        Returns: dict of source_id -> allocation_pct
        """
        # Score all sources
        scored = []
        for source in self._sources.values():
            score = self._score_source(source)
            scored.append((source, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Allocate: Kelly-like proportional to score
        total_score = sum(max(s, 0) for _, s in scored)
        allocations = {}

        for source, score in scored:
            if score <= 0 or source.ic_rolling_21d < self.min_ic_maintain:
                # No allocation
                source.allocation_pct = 0.0
                source.status = "retired" if source.days_active > 63 else "bench"
                allocations[source.source_id] = 0.0
                continue

            # Proportional allocation capped at max
            if total_score > 0:
                alloc = (score / total_score) * 0.8  # use 80% of capital
            else:
                alloc = 0.0

            alloc = min(alloc, self.max_per_source)

            # Lifecycle transitions
            if source.status == "bench" and alloc > 0:
                source.status = "seed"
                alloc = self.seed_alloc
            elif source.status == "seed" and source.ic_rolling_21d > self.min_ic_growth:
                source.status = "growth"
            elif source.status == "growth" and source.ic_trend < -0.001:
                source.status = "declining"
                alloc *= 0.5
            elif source.status == "declining" and source.ic_rolling_21d < self.min_ic_maintain:
                source.status = "retired"
                alloc = 0.0

            source.allocation_pct = alloc
            source.current_allocation_usd = alloc * self.total_capital
            source.utilization_pct = source.current_allocation_usd / max(source.estimated_capacity_usd, 1e-10)
            allocations[source.source_id] = alloc

        self._total_allocated = sum(allocations.values())
        return allocations

    def _score_source(self, source: AlphaSource) -> float:
        """Score a source for allocation priority."""
        if not source.ic_history:
            return 0.0

        # Components
        ic_quality = source.ic_rolling_21d * 10  # scale up IC
        ic_stability = max(0, 1 - abs(source.ic_trend) * 100)  # penalize instability
        capacity_room = max(0, 1 - source.utilization_pct)  # prefer under-utilized
        pnl_positive = float(np.mean(source.pnl_history[-21:]) > 0) if len(source.pnl_history) >= 21 else 0.5

        score = (
            ic_quality * 0.40 +
            ic_stability * 0.20 +
            capacity_room * 0.15 +
            pnl_positive * 0.25
        )

        return max(0, score)

    def get_allocation_table(self) -> List[Dict]:
        """Pretty allocation table for display."""
        rows = []
        for source in sorted(self._sources.values(), key=lambda s: s.allocation_pct, reverse=True):
            rows.append({
                "name": source.name,
                "type": source.source_type,
                "status": source.status,
                "allocation_pct": f"{source.allocation_pct:.1%}",
                "ic_21d": f"{source.ic_rolling_21d:.4f}",
                "ic_trend": "up" if source.ic_trend > 0.001 else "down" if source.ic_trend < -0.001 else "stable",
                "utilization": f"{source.utilization_pct:.0%}",
                "days_active": source.days_active,
            })
        return rows

    def get_summary(self) -> Dict:
        by_status = defaultdict(int)
        by_type = defaultdict(float)
        for s in self._sources.values():
            by_status[s.status] += 1
            by_type[s.source_type] += s.allocation_pct

        return {
            "total_sources": len(self._sources),
            "by_status": dict(by_status),
            "by_type": dict(by_type),
            "total_allocated_pct": self._total_allocated,
            "unallocated_pct": 1 - self._total_allocated,
        }
