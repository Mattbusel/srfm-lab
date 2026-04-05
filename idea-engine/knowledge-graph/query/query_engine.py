"""
QueryEngine: natural-language-style query interface for the knowledge graph.

Available query methods:
  - find_all_causes_of(effect)            -> List[Node]
  - find_bullish_signals_for(instrument)  -> List[Node]
  - what_regime_is_best_for(signal)       -> List[Node]
  - get_hypothesis_family(parameter)      -> List[Node]
  - find_conflicts_for(hypothesis)        -> List[Node]
  - get_top_confidence_hypotheses(n)      -> List[Node]
  - search(text_query)                    -> List[Node]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..graph.knowledge_graph import KnowledgeGraph
from ..graph.node import BaseNode, NodeType
from ..graph.edge import Edge, EdgeType
from ..reasoning.causal_reasoner import CausalReasoner, CausalChain

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Container for query results with metadata."""

    query: str
    nodes: List[BaseNode]
    chains: List[CausalChain] = field(default_factory=list)
    explanation: str = ""
    total_found: int = 0

    def __post_init__(self) -> None:
        self.total_found = len(self.nodes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_found": self.total_found,
            "explanation": self.explanation,
            "nodes": [n.to_dict() for n in self.nodes],
            "chains": [c.to_dict() for c in self.chains],
        }

    def top(self, n: int = 5) -> "QueryResult":
        """Return a copy with only the top n results by confidence."""
        sorted_nodes = sorted(self.nodes, key=lambda x: x.confidence, reverse=True)
        return QueryResult(
            query=self.query,
            nodes=sorted_nodes[:n],
            chains=self.chains[:n],
            explanation=self.explanation,
        )


class QueryEngine:
    """
    High-level query API over the knowledge graph.

    Usage::

        qe = QueryEngine(kg)
        causes = qe.find_all_causes_of("poor_win_rate_hour_1")
        regimes = qe.what_regime_is_best_for("sig_bh_signal")
        family = qe.get_hypothesis_family("min_hold_bars")
    """

    def __init__(self, kg: KnowledgeGraph) -> None:
        self._kg = kg
        self._reasoner = CausalReasoner(kg)

    # ── causal queries ────────────────────────────────────────────────────────────

    def find_all_causes_of(self, effect: str, max_depth: int = 4) -> QueryResult:
        """
        Find all nodes that causally lead to *effect* (a node label or node_id).
        """
        target = self._resolve_node(effect)
        if not target:
            return QueryResult(query=f"find_all_causes_of({effect})", nodes=[],
                               explanation=f"Node '{effect}' not found in graph.")

        causes = self._reasoner.find_all_causes_of(target.node_id, max_depth=max_depth)
        chains = []
        for cause in causes:
            path = self._kg.find_path(cause.node_id, target.node_id, max_depth=max_depth)
            if path:
                chain = self._build_chain(path)
                chain.explanation = self._reasoner.explain_path(path)
                chains.append(chain)

        return QueryResult(
            query=f"find_all_causes_of({effect})",
            nodes=causes,
            chains=chains,
            explanation=f"Found {len(causes)} causal predecessors of '{target.label}'.",
        )

    # ── signal queries ─────────────────────────────────────────────────────────────

    def find_bullish_signals_for(self, instrument: str) -> QueryResult:
        """
        Return signal nodes that are associated with *instrument* and have a
        bullish direction in their properties or label.
        """
        instr_node = self._resolve_node(instrument)
        if not instr_node:
            return QueryResult(query=f"find_bullish_signals_for({instrument})", nodes=[],
                               explanation=f"Instrument '{instrument}' not found.")

        signals = []
        for edge in self._kg._in_edges.get(instr_node.node_id, []):
            if edge.edge_type in (EdgeType.AFFECTS, EdgeType.CORRELATES_WITH):
                src = self._kg.get_node(edge.source_id)
                if src and src.node_type == NodeType.SIGNAL:
                    direction = src.properties.get("direction", "").upper()
                    if direction in ("BULLISH", "LONG", "") or "bull" in src.label.lower():
                        signals.append(src)

        # Also find hypotheses that suggest increasing this instrument's weight
        for hyp in self._kg.get_nodes_by_type(NodeType.HYPOTHESIS):
            instr_upper = instrument.upper()
            text = hyp.properties.get("hypothesis_text", hyp.label)
            direction = hyp.properties.get("direction", "")
            if instr_upper in text.upper() and any(
                kw in direction.lower() for kw in ("increase", "add", "boost", "bullish")
            ):
                signals.append(hyp)

        return QueryResult(
            query=f"find_bullish_signals_for({instrument})",
            nodes=signals,
            explanation=f"Found {len(signals)} bullish signals/hypotheses for {instrument}.",
        )

    def find_bearish_signals_for(self, instrument: str) -> QueryResult:
        """Return bearish signal nodes for *instrument*."""
        instr_node = self._resolve_node(instrument)
        if not instr_node:
            return QueryResult(query=f"find_bearish_signals_for({instrument})", nodes=[])

        signals = []
        for hyp in self._kg.get_nodes_by_type(NodeType.HYPOTHESIS):
            instr_upper = instrument.upper()
            text = hyp.properties.get("hypothesis_text", hyp.label)
            direction = hyp.properties.get("direction", "")
            if instr_upper in text.upper() and any(
                kw in direction.lower() for kw in ("decrease", "remove", "bearish", "reduce")
            ):
                signals.append(hyp)

        return QueryResult(
            query=f"find_bearish_signals_for({instrument})",
            nodes=signals,
            explanation=f"Found {len(signals)} bearish signals for {instrument}.",
        )

    # ── regime queries ────────────────────────────────────────────────────────────

    def what_regime_is_best_for(self, signal: str) -> QueryResult:
        """
        Find which regimes are associated with the best performance of *signal*.
        """
        sig_node = self._resolve_node(signal)
        if not sig_node:
            return QueryResult(query=f"what_regime_is_best_for({signal})", nodes=[])

        regime_data = self._reasoner.best_regimes_for_signal(sig_node.node_id)
        regime_nodes = []
        for rd in regime_data:
            node = self._kg.get_node(rd["regime"]["node_id"])
            if node:
                regime_nodes.append(node)

        return QueryResult(
            query=f"what_regime_is_best_for({signal})",
            nodes=regime_nodes,
            explanation=f"Found {len(regime_nodes)} regimes associated with '{signal}'.",
        )

    # ── hypothesis family queries ──────────────────────────────────────────────────

    def get_hypothesis_family(self, parameter: str) -> QueryResult:
        """
        Return all hypotheses that reference *parameter*.
        """
        matches: List[BaseNode] = []
        param_lower = parameter.lower()
        for hyp in self._kg.get_nodes_by_type(NodeType.HYPOTHESIS):
            hyp_param = hyp.properties.get("parameter", "").lower()
            hyp_text = hyp.properties.get("hypothesis_text", "").lower()
            if param_lower in hyp_param or param_lower in hyp_text:
                matches.append(hyp)

        # Also include directly connected parameter node
        param_node = self._kg.get_node(f"param_{param_lower}")
        if param_node:
            for edge in self._kg._in_edges.get(param_node.node_id, []):
                if edge.edge_type == EdgeType.PARAMETERISES:
                    src = self._kg.get_node(edge.source_id)
                    if src and src not in matches:
                        matches.append(src)

        return QueryResult(
            query=f"get_hypothesis_family({parameter})",
            nodes=matches,
            explanation=f"Found {len(matches)} hypotheses in the '{parameter}' family.",
        )

    # ── conflict queries ──────────────────────────────────────────────────────────

    def find_conflicts_for(self, hypothesis: str) -> QueryResult:
        """Return all nodes that CONFLICTS_WITH *hypothesis*."""
        hyp_node = self._resolve_node(hypothesis)
        if not hyp_node:
            return QueryResult(query=f"find_conflicts_for({hypothesis})", nodes=[])

        conflicts = self._kg.get_neighbors(
            hyp_node.node_id, edge_type=EdgeType.CONFLICTS_WITH
        )
        return QueryResult(
            query=f"find_conflicts_for({hypothesis})",
            nodes=conflicts,
            explanation=f"Found {len(conflicts)} conflicting nodes for '{hyp_node.label}'.",
        )

    # ── top-confidence queries ────────────────────────────────────────────────────

    def get_top_confidence_hypotheses(self, n: int = 10) -> QueryResult:
        hypotheses = sorted(
            self._kg.get_nodes_by_type(NodeType.HYPOTHESIS),
            key=lambda x: x.confidence,
            reverse=True,
        )
        return QueryResult(
            query=f"get_top_confidence_hypotheses({n})",
            nodes=hypotheses[:n],
            explanation=f"Top {n} hypotheses by confidence.",
        )

    # ── text search ───────────────────────────────────────────────────────────────

    def search(self, text_query: str, node_types: Optional[List[NodeType]] = None) -> QueryResult:
        """
        Search all nodes whose label or properties contain *text_query*.
        """
        tq = text_query.lower()
        results: List[BaseNode] = []
        for node in self._kg._nodes.values():
            if node_types and node.node_type not in node_types:
                continue
            if tq in node.label.lower():
                results.append(node)
                continue
            # Search in properties
            for v in node.properties.values():
                if isinstance(v, str) and tq in v.lower():
                    results.append(node)
                    break

        results.sort(key=lambda x: x.confidence, reverse=True)
        return QueryResult(
            query=f"search({text_query!r})",
            nodes=results,
            explanation=f"Found {len(results)} nodes matching '{text_query}'.",
        )

    # ── graph summary ─────────────────────────────────────────────────────────────

    def graph_summary(self) -> Dict[str, Any]:
        return self._kg.summary()

    # ── helpers ───────────────────────────────────────────────────────────────────

    def _resolve_node(self, identifier: str) -> Optional[BaseNode]:
        """Resolve a node by direct ID or by label search."""
        node = self._kg.get_node(identifier)
        if node:
            return node
        # Try common ID patterns
        for prefix in ("instr_", "sig_", "hyp_", "regime_", "pat_", "param_", "metric_"):
            node = self._kg.get_node(f"{prefix}{identifier.lower()}")
            if node:
                return node
        # Search by label
        for n in self._kg._nodes.values():
            if n.label.lower() == identifier.lower():
                return n
        return None

    def _build_chain(self, path: List[str]) -> CausalChain:
        edges: List[Edge] = []
        for i in range(len(path) - 1):
            es = self._kg.get_edges_between(path[i], path[i + 1])
            if es:
                edges.append(max(es, key=lambda e: e.weight))
        conf = self._reasoner.propagate_confidence(path)
        return CausalChain(path=path, edges=edges, aggregate_confidence=conf)
