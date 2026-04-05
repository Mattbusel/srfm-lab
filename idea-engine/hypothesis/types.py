"""
hypothesis/types.py

Core data types for the Idea Automation Engine hypothesis system.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class HypothesisType(str, Enum):
    ENTRY_TIMING = "entry_timing"
    EXIT_RULE = "exit_rule"
    REGIME_FILTER = "regime_filter"
    CROSS_ASSET = "cross_asset"
    PARAMETER_TWEAK = "parameter_tweak"
    COMPOUND = "compound"


class HypothesisStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    TESTING = "testing"
    VALIDATED = "validated"
    REJECTED = "rejected"
    PROMOTED = "promoted"


@dataclass
class MinedPattern:
    """
    Represents a statistically significant pattern discovered from trade/backtest data.
    Produced by the pattern miners and consumed by hypothesis generators.
    """

    pattern_id: str
    pattern_type: str  # time_of_day | regime_cluster | cross_asset | anomaly | mass_physics | drawdown
    instruments: list[str]
    regime_context: dict[str, Any]
    p_value: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    evidence: dict[str, Any]  # raw stats / counts
    discovered_at: str  # ISO 8601 datetime

    @classmethod
    def create(
        cls,
        pattern_type: str,
        instruments: list[str],
        p_value: float,
        effect_size: float,
        ci_lower: float,
        ci_upper: float,
        evidence: dict[str, Any],
        regime_context: dict[str, Any] | None = None,
    ) -> "MinedPattern":
        return cls(
            pattern_id=str(uuid.uuid4()),
            pattern_type=pattern_type,
            instruments=instruments,
            regime_context=regime_context or {},
            p_value=p_value,
            effect_size=effect_size,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            evidence=evidence,
            discovered_at=datetime.now(timezone.utc).isoformat(),
        )

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower

    def to_dict(self) -> dict[str, Any]:
        import json

        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "instruments": json.dumps(self.instruments),
            "regime_context": json.dumps(self.regime_context),
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "evidence": json.dumps(self.evidence),
            "discovered_at": self.discovered_at,
        }


@dataclass
class Hypothesis:
    """
    A testable trading hypothesis derived from a MinedPattern.
    Written to idea_engine.db for scheduling and tracking.
    """

    hypothesis_id: str
    type: HypothesisType
    parent_pattern_id: str
    parameters: dict[str, Any]          # the actual parameter changes to test
    predicted_sharpe_delta: float
    predicted_dd_delta: float
    novelty_score: float
    priority_rank: int
    status: HypothesisStatus
    created_at: str                      # ISO 8601
    description: str = ""
    compound_child_ids: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        hypothesis_type: HypothesisType,
        parent_pattern_id: str,
        parameters: dict[str, Any],
        predicted_sharpe_delta: float = 0.0,
        predicted_dd_delta: float = 0.0,
        novelty_score: float = 0.5,
        description: str = "",
    ) -> "Hypothesis":
        return cls(
            hypothesis_id=str(uuid.uuid4()),
            type=hypothesis_type,
            parent_pattern_id=parent_pattern_id,
            parameters=parameters,
            predicted_sharpe_delta=predicted_sharpe_delta,
            predicted_dd_delta=predicted_dd_delta,
            novelty_score=novelty_score,
            priority_rank=0,
            status=HypothesisStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
            description=description,
        )

    def to_dict(self) -> dict[str, Any]:
        import json

        return {
            "hypothesis_id": self.hypothesis_id,
            "type": self.type.value,
            "parent_pattern_id": self.parent_pattern_id,
            "parameters": json.dumps(self.parameters),
            "predicted_sharpe_delta": self.predicted_sharpe_delta,
            "predicted_dd_delta": self.predicted_dd_delta,
            "novelty_score": self.novelty_score,
            "priority_rank": self.priority_rank,
            "status": self.status.value,
            "created_at": self.created_at,
            "description": self.description,
            "compound_child_ids": json.dumps(self.compound_child_ids),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Hypothesis":
        import json

        compound_raw = row.get("compound_child_ids", "[]")
        if isinstance(compound_raw, str):
            compound_raw = json.loads(compound_raw)

        params_raw = row.get("parameters", "{}")
        if isinstance(params_raw, str):
            params_raw = json.loads(params_raw)

        return cls(
            hypothesis_id=row["hypothesis_id"],
            type=HypothesisType(row["type"]),
            parent_pattern_id=row["parent_pattern_id"],
            parameters=params_raw,
            predicted_sharpe_delta=float(row["predicted_sharpe_delta"]),
            predicted_dd_delta=float(row["predicted_dd_delta"]),
            novelty_score=float(row["novelty_score"]),
            priority_rank=int(row["priority_rank"]),
            status=HypothesisStatus(row["status"]),
            created_at=row["created_at"],
            description=row.get("description", ""),
            compound_child_ids=compound_raw,
        )


@dataclass
class ScoredHypothesis:
    """Hypothesis with scoring metadata attached (not persisted directly)."""

    hypothesis: Hypothesis
    impact_score: float           # 0-1, estimated expected value
    testability_score: float      # 0-1, how directly this maps to existing params
    novelty_score: float          # 0-1, distance from already-known hypotheses
    composite_priority: float     # impact * novelty * testability

    def apply_to_hypothesis(self) -> Hypothesis:
        """Write scores back to the underlying Hypothesis object."""
        h = self.hypothesis
        h.predicted_sharpe_delta = self.hypothesis.predicted_sharpe_delta  # already set
        h.novelty_score = self.novelty_score
        h.priority_rank = 0  # will be set by prioritizer
        return h
