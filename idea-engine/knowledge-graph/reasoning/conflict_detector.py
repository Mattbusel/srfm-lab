"""
ConflictDetector: detects and resolves conflicting hypotheses in the graph.

Conflict types:
  - Direct parameter conflict: H1 says "block hour 13", H2 says "boost hour 13"
  - Asset conflict: H1 says "remove SOL", H2 says "SOL is best momentum asset"
  - Direction conflict: two hypotheses modify the same parameter in opposite ways

Resolution: keep the higher-confidence hypothesis, archive the other.
Conflicts are recorded in the graph with CONFLICTS_WITH edges.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..graph.knowledge_graph import KnowledgeGraph
from ..graph.node import BaseNode, NodeType, HypothesisNode
from ..graph.edge import Edge, EdgeType

logger = logging.getLogger(__name__)


@dataclass
class ConflictRecord:
    """A detected conflict between two hypothesis nodes."""

    conflict_id: str
    h1_id: str
    h2_id: str
    conflict_type: str    # parameter_direction | asset_conflict | generic
    description: str
    h1_confidence: float
    h2_confidence: float
    resolution: str       # keep_h1 | keep_h2 | unresolved
    resolved_at: Optional[datetime] = None

    @property
    def winner_id(self) -> Optional[str]:
        if self.resolution == "keep_h1":
            return self.h1_id
        if self.resolution == "keep_h2":
            return self.h2_id
        return None

    @property
    def loser_id(self) -> Optional[str]:
        if self.resolution == "keep_h1":
            return self.h2_id
        if self.resolution == "keep_h2":
            return self.h1_id
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "h1_id": self.h1_id,
            "h2_id": self.h2_id,
            "conflict_type": self.conflict_type,
            "description": self.description,
            "h1_confidence": self.h1_confidence,
            "h2_confidence": self.h2_confidence,
            "resolution": self.resolution,
            "winner_id": self.winner_id,
            "loser_id": self.loser_id,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


# Opposite direction pairs for parameter modifications
OPPOSITE_DIRECTIONS: List[Tuple[str, str]] = [
    ("block", "boost"),
    ("reduce", "increase"),
    ("remove", "add"),
    ("decrease", "increase"),
    ("lower", "raise"),
    ("disable", "enable"),
    ("exclude", "include"),
    ("avoid", "prefer"),
    ("bearish", "bullish"),
]


class ConflictDetector:
    """
    Scans the knowledge graph for conflicting hypotheses and resolves them.

    Usage::

        detector  = ConflictDetector(kg)
        conflicts = detector.detect_all_conflicts()
        resolved  = detector.auto_resolve(conflicts)
    """

    def __init__(self, kg: KnowledgeGraph) -> None:
        self._kg = kg
        self._conflicts: Dict[str, ConflictRecord] = {}

    # ── public API ──────────────────────────────────────────────────────────────

    def detect_all_conflicts(self) -> List[ConflictRecord]:
        """Scan all hypothesis pairs for conflicts."""
        hypotheses = self._kg.get_nodes_by_type(NodeType.HYPOTHESIS)
        conflicts: List[ConflictRecord] = []

        # O(n²) pairwise check — acceptable for typical graph sizes (<10k hypotheses)
        for i in range(len(hypotheses)):
            for j in range(i + 1, len(hypotheses)):
                h1 = hypotheses[i]
                h2 = hypotheses[j]
                conflict = self._check_pair(h1, h2)
                if conflict:
                    self._conflicts[conflict.conflict_id] = conflict
                    conflicts.append(conflict)

        return conflicts

    def auto_resolve(
        self, conflicts: Optional[List[ConflictRecord]] = None
    ) -> List[ConflictRecord]:
        """
        Resolve conflicts by keeping the higher-confidence hypothesis and
        archiving the loser.  Adds CONFLICTS_WITH edges to the graph.
        """
        if conflicts is None:
            conflicts = self.detect_all_conflicts()

        resolved: List[ConflictRecord] = []
        for conflict in conflicts:
            self._resolve_conflict(conflict)
            resolved.append(conflict)

        return resolved

    def get_conflicting_hypotheses(
        self, hypothesis_id: str
    ) -> List[Tuple[BaseNode, ConflictRecord]]:
        """Return all hypotheses that conflict with *hypothesis_id*."""
        results = []
        for cr in self._conflicts.values():
            if cr.h1_id == hypothesis_id:
                other = self._kg.get_node(cr.h2_id)
                if other:
                    results.append((other, cr))
            elif cr.h2_id == hypothesis_id:
                other = self._kg.get_node(cr.h1_id)
                if other:
                    results.append((other, cr))
        return results

    def is_conflicted(self, hypothesis_id: str) -> bool:
        """Return True if this hypothesis has any unresolved conflict."""
        for cr in self._conflicts.values():
            if (cr.h1_id == hypothesis_id or cr.h2_id == hypothesis_id) \
                    and cr.resolution == "unresolved":
                return True
        return False

    # ── detection logic ───────────────────────────────────────────────────────────

    def _check_pair(self, h1: BaseNode, h2: BaseNode) -> Optional[ConflictRecord]:
        text1 = h1.properties.get("hypothesis_text", h1.label).lower()
        text2 = h2.properties.get("hypothesis_text", h2.label).lower()
        param1 = h1.properties.get("parameter", "").lower()
        param2 = h2.properties.get("parameter", "").lower()

        # Already have a CONFLICTS_WITH edge → already known
        existing = self._kg.get_edges_between(h1.node_id, h2.node_id)
        if any(e.edge_type == EdgeType.CONFLICTS_WITH for e in existing):
            return None

        # Rule 1: same parameter, opposite direction
        if param1 and param1 == param2:
            dir1 = h1.properties.get("direction", "")
            dir2 = h2.properties.get("direction", "")
            if self._are_opposite_directions(dir1, dir2):
                return self._make_conflict(
                    h1, h2,
                    "parameter_direction",
                    f"Both modify '{param1}' but in opposite directions: '{dir1}' vs '{dir2}'",
                )

        # Rule 2: keyword-level direction conflict in text
        conflict_type = self._text_direction_conflict(text1, text2)
        if conflict_type:
            shared_subject = self._shared_subject(text1, text2)
            if shared_subject:
                return self._make_conflict(
                    h1, h2,
                    "text_direction",
                    f"Opposing directions about '{shared_subject}': "
                    f"'{conflict_type[0]}' vs '{conflict_type[1]}'",
                )

        return None

    @staticmethod
    def _are_opposite_directions(d1: str, d2: str) -> bool:
        if not d1 or not d2:
            return False
        d1, d2 = d1.lower(), d2.lower()
        for a, b in OPPOSITE_DIRECTIONS:
            if (a in d1 and b in d2) or (b in d1 and a in d2):
                return True
        return False

    @staticmethod
    def _text_direction_conflict(t1: str, t2: str) -> Optional[Tuple[str, str]]:
        for a, b in OPPOSITE_DIRECTIONS:
            if a in t1 and b in t2:
                return (a, b)
            if b in t1 and a in t2:
                return (b, a)
        return None

    @staticmethod
    def _shared_subject(t1: str, t2: str) -> Optional[str]:
        """Find shared tokens (potential subjects) between two hypothesis texts."""
        # Extract capitalised tokens (instruments, parameters, etc.)
        words1 = set(re.findall(r"[A-Z]{2,}", t1.upper()))
        words2 = set(re.findall(r"[A-Z]{2,}", t2.upper()))
        shared = words1 & words2
        if shared:
            return ", ".join(sorted(shared))
        # Fall back to common lowercase words (ignoring stopwords)
        STOP = {"the", "a", "is", "in", "of", "and", "or", "for", "at", "to", "be"}
        w1 = set(t1.split()) - STOP
        w2 = set(t2.split()) - STOP
        common = w1 & w2
        return ", ".join(sorted(common)[:3]) if common else None

    def _make_conflict(
        self, h1: BaseNode, h2: BaseNode, conflict_type: str, description: str
    ) -> ConflictRecord:
        cid = f"conflict_{h1.node_id}__{h2.node_id}"
        resolution = "keep_h1" if h1.confidence >= h2.confidence else "keep_h2"
        return ConflictRecord(
            conflict_id=cid,
            h1_id=h1.node_id,
            h2_id=h2.node_id,
            conflict_type=conflict_type,
            description=description,
            h1_confidence=h1.confidence,
            h2_confidence=h2.confidence,
            resolution=resolution,
        )

    # ── resolution ────────────────────────────────────────────────────────────────

    def _resolve_conflict(self, conflict: ConflictRecord) -> None:
        """Add CONFLICTS_WITH edge and archive the lower-confidence hypothesis."""
        # Add bidirectional CONFLICTS_WITH edges
        self._kg.add_edge(Edge.conflicts_with(conflict.h1_id, conflict.h2_id))
        self._kg.add_edge(Edge.conflicts_with(conflict.h2_id, conflict.h1_id))

        loser_id = conflict.loser_id
        if loser_id:
            loser = self._kg.get_node(loser_id)
            if loser:
                loser.set_property("archived", True)
                loser.set_property("archived_reason", f"Conflict: {conflict.description}")
                loser.update_confidence(loser.confidence * 0.5)  # halve confidence

        conflict.resolved_at = datetime.now(timezone.utc)
        logger.info("Resolved conflict %s: %s wins", conflict.conflict_id, conflict.winner_id)
