"""
CausalReasoner: answer causal queries using the knowledge graph.

Supports:
  - Why did a hypothesis improve performance? (trace CAUSES / IMPROVES back)
  - What instruments are affected if BTC leads? (follow LEADS edges forward)
  - Which regime should a signal run in?  (find OCCURS_DURING edges)
  - Confidence propagation: multiply confidence along a path (evidence decay)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..graph.knowledge_graph import KnowledgeGraph
from ..graph.node import BaseNode, NodeType
from ..graph.edge import Edge, EdgeType

logger = logging.getLogger(__name__)


@dataclass
class CausalChain:
    """A causal path through the graph with aggregated confidence."""

    path: List[str]                  # node_id sequence
    edges: List[Edge]                # edges along the path
    aggregate_confidence: float      # product of confidences along path
    explanation: str = ""

    @property
    def length(self) -> int:
        return len(self.path) - 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "edges": [e.to_dict() for e in self.edges],
            "aggregate_confidence": self.aggregate_confidence,
            "explanation": self.explanation,
        }


class CausalReasoner:
    """
    Answer causal queries over the knowledge graph.

    Usage::

        reasoner = CausalReasoner(kg)
        chains   = reasoner.why_did_hypothesis_improve("hyp_reduce_hour_13")
        affected = reasoner.what_instruments_affected_by_leader("instr_btc")
        regimes  = reasoner.best_regimes_for_signal("sig_bh_signal")
    """

    def __init__(self, kg: KnowledgeGraph) -> None:
        self._kg = kg

    # ── public queries ───────────────────────────────────────────────────────────

    def why_did_hypothesis_improve(
        self, hypothesis_id: str, max_depth: int = 5
    ) -> List[CausalChain]:
        """
        Trace backwards through CAUSES/IMPROVES edges from a hypothesis node to
        find what caused its observed improvement.
        """
        node = self._kg.get_node(hypothesis_id)
        if not node:
            logger.warning("Node not found: %s", hypothesis_id)
            return []

        chains: List[CausalChain] = []
        # Walk in-edges of CAUSES and IMPROVES type
        for edge_type in (EdgeType.CAUSES, EdgeType.IMPROVES):
            for edge in self._kg._in_edges.get(hypothesis_id, []):
                if edge.edge_type != edge_type:
                    continue
                path = self._kg.find_path(
                    edge.source_id, hypothesis_id, max_depth=max_depth, edge_type=edge_type
                )
                if path:
                    chain = self._build_chain(path)
                    chain.explanation = (
                        f"Node '{edge.source_id}' {edge_type.value} '{hypothesis_id}' "
                        f"(weight={edge.weight:.2f}, evidence={edge.evidence_count})"
                    )
                    chains.append(chain)

        return sorted(chains, key=lambda c: c.aggregate_confidence, reverse=True)

    def what_instruments_affected_by_leader(
        self, leader_id: str, max_depth: int = 3
    ) -> List[BaseNode]:
        """
        Follow LEADS edges forward from *leader_id* to find all follower
        instruments.
        """
        visited: set[str] = set()
        frontier = [leader_id]
        followers: List[BaseNode] = []

        for _ in range(max_depth):
            next_frontier = []
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                for edge in self._kg._out_edges.get(nid, []):
                    if edge.edge_type == EdgeType.LEADS and edge.target_id not in visited:
                        target = self._kg.get_node(edge.target_id)
                        if target and target.node_type == NodeType.INSTRUMENT:
                            followers.append(target)
                        next_frontier.append(edge.target_id)
            frontier = next_frontier

        return followers

    def best_regimes_for_signal(self, signal_id: str) -> List[Dict[str, Any]]:
        """
        Find regimes where *signal_id* OCCURS_DURING, sorted by edge weight.
        """
        results = []
        for edge in self._kg._out_edges.get(signal_id, []):
            if edge.edge_type == EdgeType.OCCURS_DURING:
                regime = self._kg.get_node(edge.target_id)
                if regime:
                    results.append({
                        "regime": regime.to_dict(),
                        "edge_weight": edge.weight,
                        "evidence_count": edge.evidence_count,
                    })
        return sorted(results, key=lambda r: r["edge_weight"], reverse=True)

    def propagate_confidence(self, path: List[str]) -> float:
        """
        Compute aggregate confidence along a node path by multiplying
        edge weights and node confidences (evidence decay model).
        """
        if len(path) < 2:
            node = self._kg.get_node(path[0]) if path else None
            return node.confidence if node else 0.0

        confidence = 1.0
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            src_node = self._kg.get_node(src)
            edges = self._kg.get_edges_between(src, tgt)
            if not edges or not src_node:
                return 0.0
            best_edge = max(edges, key=lambda e: e.weight)
            confidence *= src_node.confidence * best_edge.weight

        tgt_node = self._kg.get_node(path[-1])
        if tgt_node:
            confidence *= tgt_node.confidence

        return min(max(confidence, 0.0), 1.0)

    def find_all_causes_of(self, target_id: str, max_depth: int = 4) -> List[BaseNode]:
        """
        Return all nodes that CAUSE (directly or transitively) *target_id*.
        """
        causes: List[BaseNode] = []
        visited: set[str] = set()
        frontier = [target_id]

        for _ in range(max_depth):
            next_frontier = []
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                for edge in self._kg._in_edges.get(nid, []):
                    if edge.edge_type == EdgeType.CAUSES and edge.source_id not in visited:
                        cause_node = self._kg.get_node(edge.source_id)
                        if cause_node:
                            causes.append(cause_node)
                        next_frontier.append(edge.source_id)
            frontier = next_frontier

        return causes

    def explain_path(self, path: List[str]) -> str:
        """Return a human-readable explanation for a chain of nodes."""
        if not path:
            return "Empty path."
        parts = []
        for i in range(len(path) - 1):
            src_node = self._kg.get_node(path[i])
            tgt_node = self._kg.get_node(path[i + 1])
            edges = self._kg.get_edges_between(path[i], path[i + 1])
            src_label = src_node.label if src_node else path[i]
            tgt_label = tgt_node.label if tgt_node else path[i + 1]
            if edges:
                e = max(edges, key=lambda x: x.weight)
                parts.append(f"'{src_label}' --[{e.edge_type.value} w={e.weight:.2f}]--> '{tgt_label}'")
            else:
                parts.append(f"'{src_label}' --> '{tgt_label}'")
        return " | ".join(parts)

    # ── internal helpers ──────────────────────────────────────────────────────────

    def _build_chain(self, path: List[str]) -> CausalChain:
        edges: List[Edge] = []
        for i in range(len(path) - 1):
            es = self._kg.get_edges_between(path[i], path[i + 1])
            if es:
                edges.append(max(es, key=lambda e: e.weight))
        conf = self.propagate_confidence(path)
        return CausalChain(path=path, edges=edges, aggregate_confidence=conf)
