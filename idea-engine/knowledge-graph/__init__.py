"""
knowledge-graph: Semantic knowledge graph of the trading domain.

Connects instruments, signals, hypotheses, causal relationships, market
regimes, and IAE ideas into a queryable research database.
"""

from .graph.knowledge_graph import KnowledgeGraph
from .graph.node import (
    InstrumentNode, SignalNode, HypothesisNode, RegimeNode,
    PatternNode, EventNode, ParameterNode,
)
from .graph.edge import Edge, EdgeType
from .query.query_engine import QueryEngine

__all__ = [
    "KnowledgeGraph",
    "InstrumentNode", "SignalNode", "HypothesisNode", "RegimeNode",
    "PatternNode", "EventNode", "ParameterNode",
    "Edge", "EdgeType",
    "QueryEngine",
]
