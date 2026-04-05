"""
Knowledge graph edge definitions.

Edges encode typed, weighted relationships between nodes.  Edge types
capture causality, correlation, improvement, conflict, co-occurrence, and
lead/lag relationships.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class EdgeType(str, Enum):
    CAUSES           = "CAUSES"            # causal relationship
    CORRELATES_WITH  = "CORRELATES_WITH"   # statistical correlation
    IMPROVES         = "IMPROVES"          # hypothesis improves a metric
    CONFLICTS_WITH   = "CONFLICTS_WITH"    # hypotheses contradict each other
    SUPPORTS         = "SUPPORTS"          # hypothesis supports another
    OCCURS_DURING    = "OCCURS_DURING"     # pattern observed in a regime
    LEADS            = "LEADS"             # instrument A leads instrument B
    BELONGS_TO       = "BELONGS_TO"        # node belongs to a group/family
    AFFECTS          = "AFFECTS"           # event affects instrument
    PARAMETERISES    = "PARAMETERISES"     # parameter controls a signal


@dataclass
class Edge:
    """
    A directed, typed, weighted relationship between two graph nodes.

    Attributes
    ----------
    source_id:      node_id of the source node
    target_id:      node_id of the target node
    edge_type:      one of EdgeType values
    weight:         numeric relationship strength (-1 to 1 for correlations,
                    0-1 for causal strength, otherwise free)
    evidence_count: number of observations supporting this edge
    last_updated:   when this edge was last updated
    properties:     additional metadata
    """

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    evidence_count: int = 1
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    properties: Dict[str, Any] = field(default_factory=dict)
    edge_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.edge_id:
            self.edge_id = f"{self.source_id}__{self.edge_type.value}__{self.target_id}"

    @property
    def is_causal(self) -> bool:
        return self.edge_type == EdgeType.CAUSES

    @property
    def is_conflict(self) -> bool:
        return self.edge_type == EdgeType.CONFLICTS_WITH

    def reinforce(self, delta_evidence: int = 1, weight_bump: float = 0.05) -> None:
        """Add evidence and strengthen the edge weight (capped at 1.0)."""
        self.evidence_count += delta_evidence
        self.weight = min(1.0, self.weight + weight_bump)
        self.last_updated = datetime.now(timezone.utc)

    def decay(self, factor: float = 0.95) -> None:
        """Apply evidence decay — edges grow stale without reinforcement."""
        self.weight *= factor
        self.last_updated = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "evidence_count": self.evidence_count,
            "last_updated": self.last_updated.isoformat(),
            "properties": self.properties,
        }

    @classmethod
    def causes(
        cls,
        source_id: str,
        target_id: str,
        weight: float = 0.7,
        evidence_count: int = 1,
        **kwargs: Any,
    ) -> "Edge":
        """Convenience constructor for causal edges."""
        return cls(source_id, target_id, EdgeType.CAUSES, weight, evidence_count, **kwargs)

    @classmethod
    def correlates_with(
        cls,
        source_id: str,
        target_id: str,
        weight: float = 0.5,   # -1 to 1
        evidence_count: int = 1,
        **kwargs: Any,
    ) -> "Edge":
        return cls(source_id, target_id, EdgeType.CORRELATES_WITH, weight, evidence_count, **kwargs)

    @classmethod
    def improves(
        cls,
        hypothesis_id: str,
        metric: str,
        weight: float = 0.6,
        evidence_count: int = 1,
        **kwargs: Any,
    ) -> "Edge":
        return cls(hypothesis_id, metric, EdgeType.IMPROVES, weight, evidence_count, **kwargs)

    @classmethod
    def conflicts_with(
        cls,
        h1_id: str,
        h2_id: str,
        weight: float = 1.0,
        evidence_count: int = 1,
        **kwargs: Any,
    ) -> "Edge":
        return cls(h1_id, h2_id, EdgeType.CONFLICTS_WITH, weight, evidence_count, **kwargs)

    @classmethod
    def supports(
        cls,
        h1_id: str,
        h2_id: str,
        weight: float = 0.7,
        evidence_count: int = 1,
        **kwargs: Any,
    ) -> "Edge":
        return cls(h1_id, h2_id, EdgeType.SUPPORTS, weight, evidence_count, **kwargs)

    @classmethod
    def occurs_during(
        cls,
        pattern_id: str,
        regime_id: str,
        weight: float = 0.7,
        evidence_count: int = 1,
        **kwargs: Any,
    ) -> "Edge":
        return cls(pattern_id, regime_id, EdgeType.OCCURS_DURING, weight, evidence_count, **kwargs)

    @classmethod
    def leads(
        cls,
        leader_id: str,
        follower_id: str,
        lag_bars: int = 1,
        weight: float = 0.6,
        evidence_count: int = 1,
        **kwargs: Any,
    ) -> "Edge":
        props = {"lag_bars": lag_bars}
        return cls(
            leader_id, follower_id, EdgeType.LEADS, weight, evidence_count,
            properties=props, **kwargs
        )
