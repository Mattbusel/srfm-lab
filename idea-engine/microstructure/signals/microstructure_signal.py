"""
microstructure/signals/microstructure_signal.py

MicrostructureSignal: composite health score for one symbol at one moment.

Purpose
-------
Aggregates outputs from all microstructure models into a single actionable
signal that the IAE can use to:
1. Scale position size (recommended_size_multiplier)
2. Gate new entries (composite_health < 0.3 → skip)
3. Feed into the live_monitor loop for real-time decisions

Scoring model
-------------
composite_health = weighted average of component scores:
    amihud_score      (0-1): 1 = liquid, 0 = extremely illiquid
    roll_score        (0-1): 1 = tight spread, 0 = extremely wide
    adverse_score     (0-1): 1 = low informed trading, 0 = high PIN
    kyle_score        (0-1): 1 = low price impact, 0 = high impact

Weights: amihud=0.35, roll=0.30, adverse=0.20, kyle=0.15
(Amihud and Roll are most directly traded-against; they get highest weight.)

Size multiplier schedule
------------------------
composite_health >= 0.70 → 1.00 (full size, healthy market)
composite_health >= 0.50 → 0.75 (slightly reduced)
composite_health >= 0.30 → 0.50 (thin market, half size)
composite_health <  0.30 → 0.00 (broken microstructure, skip entry)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from microstructure.models.adverse_selection import AdverseSelectionRisk


class MicrostructureHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    THIN = "thin"
    BROKEN = "broken"


@dataclass
class MicrostructureSignal:
    """
    Composite microstructure health signal for one symbol.

    Consumed by the IAE live monitor, position sizer, and hypothesis generator.
    """
    symbol: str
    timestamp: str

    # --- Component scores (0-1) ----------------------------------------
    amihud_score: float             # 1 = liquid, 0 = illiquid
    roll_score: float               # 1 = tight spread, 0 = wide
    adverse_score: float            # 1 = low PIN, 0 = high PIN
    kyle_score: float               # 1 = low impact, 0 = high impact

    # --- Raw values for context ----------------------------------------
    amihud_percentile: float        # 0-1, where current ILLIQ sits in history
    estimated_spread: float         # Roll spread in price units
    adverse_selection_risk: AdverseSelectionRisk
    kyle_size_multiplier: float     # from Kyle lambda module

    # --- Composite output ----------------------------------------------
    composite_health: float         # 0-1 weighted aggregate
    health_state: MicrostructureHealth
    recommended_size_multiplier: float   # 0.0, 0.5, 0.75, or 1.0
    entry_blocked: bool             # True if composite_health < 0.3

    # --- Metadata -------------------------------------------------------
    model_inputs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        symbol: str,
        amihud_percentile: float,
        amihud_is_thin: bool,
        roll_spread: float,
        roll_baseline: float,
        adverse_risk: AdverseSelectionRisk,
        adverse_pin: float,
        kyle_percentile: float,
        kyle_size_multiplier: float,
        timestamp: str | None = None,
    ) -> "MicrostructureSignal":
        """
        Construct a MicrostructureSignal from raw model outputs.

        All inputs are expected to already be computed by the respective
        model modules.  This class only aggregates them.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        # --- Component scores ------------------------------------------
        # Amihud: higher percentile = less liquid = worse score
        amihud_score = max(0.0, 1.0 - amihud_percentile)
        if amihud_is_thin:
            amihud_score = min(amihud_score, 0.3)

        # Roll: spread_ratio = roll_spread / baseline
        spread_ratio = roll_spread / roll_baseline if roll_baseline > 1e-12 else 1.0
        roll_score = max(0.0, 1.0 - max(0.0, spread_ratio - 1.0) / 3.0)

        # Adverse selection: map risk level to score
        adverse_score = {
            AdverseSelectionRisk.LOW: 0.9,
            AdverseSelectionRisk.MEDIUM: 0.5,
            AdverseSelectionRisk.HIGH: 0.1,
        }[adverse_risk]
        # Also penalise by raw PIN
        adverse_score = min(adverse_score, max(0.0, 1.0 - adverse_pin * 2))

        # Kyle: already a 0-1 percentile; higher = worse
        kyle_score = max(0.0, 1.0 - kyle_percentile)

        # --- Composite -------------------------------------------------
        composite = (
            0.35 * amihud_score
            + 0.30 * roll_score
            + 0.20 * adverse_score
            + 0.15 * kyle_score
        )
        composite = max(0.0, min(1.0, composite))

        # --- Health state and size multiplier --------------------------
        if composite >= 0.70:
            health_state = MicrostructureHealth.HEALTHY
            size_mult = 1.00
        elif composite >= 0.50:
            health_state = MicrostructureHealth.DEGRADED
            size_mult = 0.75
        elif composite >= 0.30:
            health_state = MicrostructureHealth.THIN
            size_mult = 0.50
        else:
            health_state = MicrostructureHealth.BROKEN
            size_mult = 0.00

        # Kyle's own multiplier provides an additional floor
        size_mult = min(size_mult, kyle_size_multiplier)

        return cls(
            symbol=symbol,
            timestamp=ts,
            amihud_score=round(amihud_score, 4),
            roll_score=round(roll_score, 4),
            adverse_score=round(adverse_score, 4),
            kyle_score=round(kyle_score, 4),
            amihud_percentile=round(amihud_percentile, 4),
            estimated_spread=round(roll_spread, 6),
            adverse_selection_risk=adverse_risk,
            kyle_size_multiplier=round(kyle_size_multiplier, 3),
            composite_health=round(composite, 4),
            health_state=health_state,
            recommended_size_multiplier=round(size_mult, 3),
            entry_blocked=composite < 0.30,
            model_inputs={
                "amihud_percentile": amihud_percentile,
                "amihud_is_thin": amihud_is_thin,
                "roll_spread": roll_spread,
                "roll_baseline": roll_baseline,
                "spread_ratio": round(spread_ratio, 4),
                "adverse_pin": adverse_pin,
                "kyle_percentile": kyle_percentile,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "composite_health": self.composite_health,
            "health_state": self.health_state.value,
            "recommended_size_multiplier": self.recommended_size_multiplier,
            "entry_blocked": self.entry_blocked,
            "amihud_score": self.amihud_score,
            "amihud_percentile": self.amihud_percentile,
            "roll_score": self.roll_score,
            "estimated_spread": self.estimated_spread,
            "adverse_score": self.adverse_score,
            "adverse_selection_risk": self.adverse_selection_risk.value,
            "kyle_score": self.kyle_score,
            "kyle_size_multiplier": self.kyle_size_multiplier,
        }

    def __str__(self) -> str:
        return (
            f"MicrostructureSignal({self.symbol} @ {self.timestamp[:16]}: "
            f"health={self.composite_health:.3f} [{self.health_state.value}], "
            f"size_mult={self.recommended_size_multiplier}, "
            f"entry_blocked={self.entry_blocked})"
        )
