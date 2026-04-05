"""
HypothesisLinker: automatically links new IAE hypotheses to existing graph nodes.

For each incoming hypothesis:
  1. Parse text to extract mentioned instruments, signal types, regimes, parameters
  2. Create graph nodes for the hypothesis
  3. Find existing hypotheses that support or contradict this one
  4. Add SUPPORTS and CONFLICTS_WITH edges automatically
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..graph.knowledge_graph import KnowledgeGraph
from ..graph.node import (
    BaseNode, NodeType, HypothesisNode, InstrumentNode, SignalNode,
    RegimeNode, ParameterNode,
)
from ..graph.edge import Edge, EdgeType

logger = logging.getLogger(__name__)

# Known instruments (extended as needed)
KNOWN_INSTRUMENTS: Set[str] = {
    "BTC", "ETH", "SOL", "AVAX", "ARB", "OP", "APT", "BNB", "MATIC",
    "LINK", "UNI", "AAVE", "DOT", "ADA", "NEAR", "FTM", "INJ",
}

# Known regimes
KNOWN_REGIMES: Set[str] = {
    "bull", "bear", "high_vol", "low_vol", "trending", "ranging",
    "risk_on", "risk_off", "consolidation",
}

# Known signal families
KNOWN_SIGNALS: Set[str] = {
    "BH", "momentum", "mean_reversion", "breakout", "trend_following",
    "volume", "on_chain", "sentiment", "macro", "sopr", "mvrv", "funding",
}

# Parameter name hints
PARAMETER_HINTS: List[str] = [
    "min_hold_bars", "max_hold_bars", "entry_hour", "exit_hour",
    "stop_loss", "take_profit", "position_size", "leverage",
    "lookback", "threshold", "window", "filter", "multiplier",
]


@dataclass
class LinkingResult:
    """Result of linking a hypothesis to the graph."""

    hypothesis_id: str
    nodes_created: List[str] = field(default_factory=list)
    edges_added: List[str] = field(default_factory=list)
    supporting_hypotheses: List[str] = field(default_factory=list)
    conflicting_hypotheses: List[str] = field(default_factory=list)
    extracted_instruments: List[str] = field(default_factory=list)
    extracted_signals: List[str] = field(default_factory=list)
    extracted_regimes: List[str] = field(default_factory=list)
    extracted_parameters: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "nodes_created": self.nodes_created,
            "edges_added": self.edges_added,
            "supporting_hypotheses": self.supporting_hypotheses,
            "conflicting_hypotheses": self.conflicting_hypotheses,
            "extracted": {
                "instruments": self.extracted_instruments,
                "signals": self.extracted_signals,
                "regimes": self.extracted_regimes,
                "parameters": self.extracted_parameters,
            },
        }


class HypothesisLinker:
    """
    Automatically integrates new IAE hypotheses into the knowledge graph.

    Usage::

        linker = HypothesisLinker(kg)
        result = linker.link_hypothesis(
            hypothesis_id="hyp_block_hour_13",
            hypothesis_text="Block new entries at hour 13 UTC due to low liquidity",
            parameter="entry_hour_filter",
            direction="block",
            confidence=0.7,
        )
    """

    def __init__(self, kg: KnowledgeGraph) -> None:
        self._kg = kg

    # ── public API ──────────────────────────────────────────────────────────────

    def link_hypothesis(
        self,
        hypothesis_id: str,
        hypothesis_text: str,
        parameter: str = "",
        direction: str = "",
        confidence: float = 0.5,
        extra_properties: Optional[Dict[str, Any]] = None,
    ) -> LinkingResult:
        """
        Parse a hypothesis, create its node, link to related graph entities.
        """
        result = LinkingResult(hypothesis_id=hypothesis_id)

        # 1. Extract entities from text
        instruments = self._extract_instruments(hypothesis_text)
        signals = self._extract_signals(hypothesis_text)
        regimes = self._extract_regimes(hypothesis_text)
        parameters = self._extract_parameters(hypothesis_text, parameter)

        result.extracted_instruments = instruments
        result.extracted_signals = signals
        result.extracted_regimes = regimes
        result.extracted_parameters = parameters

        # 2. Create hypothesis node
        props = extra_properties or {}
        hyp_node = HypothesisNode(
            hypothesis_id=hypothesis_id,
            hypothesis_text=hypothesis_text,
            parameter=parameter,
            direction=direction,
            confidence=confidence,
            **props,
        )
        node_id = self._kg.add_node(hyp_node)
        result.nodes_created.append(node_id)

        # 3. Ensure instrument nodes exist and link
        for sym in instruments:
            instr_id = self._ensure_instrument_node(sym)
            edge_id = self._kg.add_edge(Edge(
                source_id=node_id,
                target_id=instr_id,
                edge_type=EdgeType.AFFECTS,
                weight=0.7,
            ))
            result.edges_added.append(edge_id)

        # 4. Ensure signal nodes exist and link
        for sig in signals:
            sig_id = self._ensure_signal_node(sig)
            edge_id = self._kg.add_edge(Edge(
                source_id=node_id,
                target_id=sig_id,
                edge_type=EdgeType.AFFECTS,
                weight=0.6,
            ))
            result.edges_added.append(edge_id)

        # 5. Ensure regime nodes exist and link
        for regime in regimes:
            regime_id = self._ensure_regime_node(regime)
            edge_id = self._kg.add_edge(Edge(
                source_id=node_id,
                target_id=regime_id,
                edge_type=EdgeType.OCCURS_DURING,
                weight=0.5,
            ))
            result.edges_added.append(edge_id)

        # 6. Ensure parameter nodes exist and link
        for param in parameters:
            param_id = self._ensure_parameter_node(param)
            edge_id = self._kg.add_edge(Edge(
                source_id=node_id,
                target_id=param_id,
                edge_type=EdgeType.PARAMETERISES,
                weight=0.8,
            ))
            result.edges_added.append(edge_id)

        # 7. Find supporting and conflicting hypotheses
        supporters, conflicts = self._find_related_hypotheses(
            hyp_node, hypothesis_text, parameter, direction
        )

        for sup_id in supporters:
            self._kg.add_edge(Edge.supports(node_id, sup_id, weight=0.6))
            self._kg.add_edge(Edge.supports(sup_id, node_id, weight=0.6))
            result.supporting_hypotheses.append(sup_id)

        for conf_id in conflicts:
            self._kg.add_edge(Edge.conflicts_with(node_id, conf_id))
            self._kg.add_edge(Edge.conflicts_with(conf_id, node_id))
            result.conflicting_hypotheses.append(conf_id)

        logger.info(
            "Linked hypothesis '%s': %d nodes, %d edges, %d supporters, %d conflicts",
            hypothesis_id, len(result.nodes_created), len(result.edges_added),
            len(result.supporting_hypotheses), len(result.conflicting_hypotheses),
        )
        return result

    def link_batch(
        self, hypotheses: List[Dict[str, Any]]
    ) -> List[LinkingResult]:
        """Link multiple hypotheses from a list of dicts."""
        results = []
        for h in hypotheses:
            results.append(self.link_hypothesis(**h))
        return results

    # ── entity extraction ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_instruments(text: str) -> List[str]:
        found = []
        for sym in KNOWN_INSTRUMENTS:
            if re.search(rf"\b{sym}\b", text, re.IGNORECASE):
                found.append(sym.upper())
        return found

    @staticmethod
    def _extract_signals(text: str) -> List[str]:
        found = []
        for sig in KNOWN_SIGNALS:
            if re.search(rf"\b{re.escape(sig)}\b", text, re.IGNORECASE):
                found.append(sig)
        return found

    @staticmethod
    def _extract_regimes(text: str) -> List[str]:
        found = []
        for regime in KNOWN_REGIMES:
            if re.search(rf"\b{re.escape(regime)}\b", text, re.IGNORECASE):
                found.append(regime)
        return found

    @staticmethod
    def _extract_parameters(text: str, explicit_param: str = "") -> List[str]:
        found = set()
        if explicit_param:
            found.add(explicit_param)
        for hint in PARAMETER_HINTS:
            if hint in text.lower():
                found.add(hint)
        return list(found)

    # ── node creation helpers ─────────────────────────────────────────────────────

    def _ensure_instrument_node(self, symbol: str) -> str:
        node_id = f"instr_{symbol.lower()}"
        if not self._kg.get_node(node_id):
            self._kg.add_node(InstrumentNode(symbol=symbol))
        return node_id

    def _ensure_signal_node(self, signal_name: str) -> str:
        node_id = f"sig_{signal_name.lower()}"
        if not self._kg.get_node(node_id):
            self._kg.add_node(SignalNode(signal_name=signal_name))
        return node_id

    def _ensure_regime_node(self, regime_name: str) -> str:
        node_id = f"regime_{regime_name.lower()}"
        if not self._kg.get_node(node_id):
            self._kg.add_node(RegimeNode(regime_name=regime_name))
        return node_id

    def _ensure_parameter_node(self, param_name: str) -> str:
        node_id = f"param_{param_name.lower()}"
        if not self._kg.get_node(node_id):
            self._kg.add_node(ParameterNode(param_name=param_name))
        return node_id

    # ── related hypothesis search ─────────────────────────────────────────────────

    def _find_related_hypotheses(
        self,
        new_node: HypothesisNode,
        text: str,
        parameter: str,
        direction: str,
    ) -> Tuple[List[str], List[str]]:
        """Return (supporters, conflicters) node_id lists."""
        supporters: List[str] = []
        conflicters: List[str] = []

        existing = self._kg.get_nodes_by_type(NodeType.HYPOTHESIS)
        for node in existing:
            if node.node_id == new_node.node_id:
                continue
            other_text = node.properties.get("hypothesis_text", node.label).lower()
            other_param = node.properties.get("parameter", "").lower()
            other_dir = node.properties.get("direction", "").lower()

            # Support: same parameter, same direction
            if parameter and parameter == other_param:
                if direction and direction == other_dir:
                    supporters.append(node.node_id)
                    continue

            # Conflict: same parameter, opposite direction
            if parameter and parameter == other_param and direction:
                if self._are_opposite(direction, other_dir):
                    conflicters.append(node.node_id)
                    continue

            # Semantic similarity via shared key terms
            key_terms = self._key_terms(text)
            other_terms = self._key_terms(other_text)
            overlap = len(key_terms & other_terms) / max(len(key_terms | other_terms), 1)
            if overlap > 0.5:
                supporters.append(node.node_id)

        return supporters, conflicters

    @staticmethod
    def _are_opposite(d1: str, d2: str) -> bool:
        pairs = [("block", "boost"), ("reduce", "increase"), ("remove", "add"),
                 ("decrease", "increase"), ("disable", "enable"), ("bearish", "bullish")]
        d1, d2 = d1.lower(), d2.lower()
        for a, b in pairs:
            if (a in d1 and b in d2) or (b in d1 and a in d2):
                return True
        return False

    @staticmethod
    def _key_terms(text: str) -> set[str]:
        STOP = {"the", "a", "is", "in", "of", "and", "or", "for", "at", "to",
                "be", "that", "this", "with", "due", "by", "on", "new"}
        return {w for w in text.lower().split() if len(w) > 3 and w not in STOP}
