"""
KnowledgeGraph: in-memory adjacency list representation of the trading
knowledge graph.

Supports add/query/path-finding operations and centrality computation.
Backed by graph_store.py for persistence.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from .node import BaseNode, NodeType
from .edge import Edge, EdgeType

logger = logging.getLogger(__name__)


@dataclass
class Centrality:
    """Centrality scores for a node."""

    node_id: str
    degree: int
    in_degree: int
    out_degree: int
    betweenness_approx: float = 0.0


class KnowledgeGraph:
    """
    In-memory knowledge graph with adjacency list storage.

    Nodes and edges are stored in dicts keyed by ID.  Adjacency lists
    index outgoing edges per source node.

    Usage::

        kg = KnowledgeGraph()
        kg.add_node(InstrumentNode("BTC"))
        kg.add_node(SignalNode("BH_signal"))
        kg.add_edge(Edge.leads("instr_btc", "instr_eth", lag_bars=2))
        path = kg.find_path("instr_btc", "instr_sol")
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, BaseNode] = {}
        # outgoing: source_id -> list of Edge
        self._out_edges: Dict[str, List[Edge]] = defaultdict(list)
        # incoming: target_id -> list of Edge
        self._in_edges: Dict[str, List[Edge]] = defaultdict(list)
        # edge index: edge_id -> Edge
        self._edges: Dict[str, Edge] = {}

    # ── node operations ──────────────────────────────────────────────────────────

    def add_node(self, node: BaseNode) -> str:
        """Add or replace a node. Returns node_id."""
        self._nodes[node.node_id] = node
        return node.node_id

    def get_node(self, node_id: str) -> Optional[BaseNode]:
        return self._nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[BaseNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def get_nodes_by_confidence(self, min_confidence: float) -> List[BaseNode]:
        return [n for n in self._nodes.values() if n.confidence >= min_confidence]

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        # remove associated edges
        for edge in list(self._out_edges.pop(node_id, [])):
            self._edges.pop(edge.edge_id, None)
            self._in_edges[edge.target_id] = [
                e for e in self._in_edges[edge.target_id] if e.edge_id != edge.edge_id
            ]
        for edge in list(self._in_edges.pop(node_id, [])):
            self._edges.pop(edge.edge_id, None)
            self._out_edges[edge.source_id] = [
                e for e in self._out_edges[edge.source_id] if e.edge_id != edge.edge_id
            ]
        return True

    def node_count(self) -> int:
        return len(self._nodes)

    # ── edge operations ──────────────────────────────────────────────────────────

    def add_edge(self, edge: Edge) -> str:
        """Add or reinforce an edge. Returns edge_id."""
        if edge.edge_id in self._edges:
            self._edges[edge.edge_id].reinforce()
            return edge.edge_id
        self._edges[edge.edge_id] = edge
        self._out_edges[edge.source_id].append(edge)
        self._in_edges[edge.target_id].append(edge)
        return edge.edge_id

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        return self._edges.get(edge_id)

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "out",
    ) -> List[BaseNode]:
        """
        Return neighbour nodes connected via *edge_type* in *direction*.

        direction: 'out' (following), 'in' (preceding), 'both'
        """
        result_ids: Set[str] = set()
        if direction in ("out", "both"):
            for e in self._out_edges.get(node_id, []):
                if edge_type is None or e.edge_type == edge_type:
                    result_ids.add(e.target_id)
        if direction in ("in", "both"):
            for e in self._in_edges.get(node_id, []):
                if edge_type is None or e.edge_type == edge_type:
                    result_ids.add(e.source_id)
        return [self._nodes[nid] for nid in result_ids if nid in self._nodes]

    def get_edges_between(self, source_id: str, target_id: str) -> List[Edge]:
        return [e for e in self._out_edges.get(source_id, []) if e.target_id == target_id]

    def get_edges_of_type(self, edge_type: EdgeType) -> List[Edge]:
        return [e for e in self._edges.values() if e.edge_type == edge_type]

    def edge_count(self) -> int:
        return len(self._edges)

    # ── path finding ──────────────────────────────────────────────────────────────

    def find_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 6,
        edge_type: Optional[EdgeType] = None,
    ) -> Optional[List[str]]:
        """
        BFS shortest path from *from_id* to *to_id*.

        Returns list of node_ids (including endpoints) or None if unreachable.
        """
        if from_id not in self._nodes or to_id not in self._nodes:
            return None
        queue: deque[Tuple[str, List[str]]] = deque([(from_id, [from_id])])
        visited: Set[str] = {from_id}
        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth:
                continue
            for edge in self._out_edges.get(current, []):
                if edge_type and edge.edge_type != edge_type:
                    continue
                nxt = edge.target_id
                if nxt == to_id:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return None

    def find_all_paths(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 4,
        edge_type: Optional[EdgeType] = None,
    ) -> List[List[str]]:
        """DFS to find all simple paths up to max_depth."""
        results: List[List[str]] = []
        self._dfs_paths(from_id, to_id, [], set(), max_depth, edge_type, results)
        return results

    def _dfs_paths(
        self,
        current: str,
        target: str,
        path: List[str],
        visited: Set[str],
        max_depth: int,
        edge_type: Optional[EdgeType],
        results: List[List[str]],
    ) -> None:
        path = path + [current]
        visited = visited | {current}
        if current == target:
            results.append(path)
            return
        if len(path) > max_depth:
            return
        for edge in self._out_edges.get(current, []):
            if edge_type and edge.edge_type != edge_type:
                continue
            if edge.target_id not in visited:
                self._dfs_paths(edge.target_id, target, path, visited, max_depth, edge_type, results)

    # ── subgraph ──────────────────────────────────────────────────────────────────

    def get_subgraph(self, node_ids: Iterable[str]) -> "KnowledgeGraph":
        """Return a new KnowledgeGraph containing only the specified nodes and edges between them."""
        id_set = set(node_ids)
        sub = KnowledgeGraph()
        for nid in id_set:
            if nid in self._nodes:
                sub.add_node(self._nodes[nid])
        for edge in self._edges.values():
            if edge.source_id in id_set and edge.target_id in id_set:
                sub.add_edge(edge)
        return sub

    # ── centrality ────────────────────────────────────────────────────────────────

    def compute_centrality(self) -> Dict[str, Centrality]:
        """
        Compute degree and approximate betweenness centrality for all nodes.

        Betweenness is approximated via random-sample BFS (k=min(100, n) sources)
        to keep it tractable for large graphs.
        """
        import random
        centralities: Dict[str, Centrality] = {}
        for nid in self._nodes:
            out_deg = len(self._out_edges.get(nid, []))
            in_deg = len(self._in_edges.get(nid, []))
            centralities[nid] = Centrality(
                node_id=nid,
                degree=out_deg + in_deg,
                in_degree=in_deg,
                out_degree=out_deg,
            )

        # Approximate betweenness: count how often each node appears on shortest paths
        all_ids = list(self._nodes.keys())
        sample_size = min(100, len(all_ids))
        sources = random.sample(all_ids, sample_size)
        betweenness: Dict[str, float] = defaultdict(float)

        for src in sources:
            # BFS from src to all reachable nodes
            prev: Dict[str, List[str]] = defaultdict(list)
            dist: Dict[str, int] = {src: 0}
            sigma: Dict[str, float] = defaultdict(float)
            sigma[src] = 1.0
            queue: deque[str] = deque([src])
            order: List[str] = []
            while queue:
                v = queue.popleft()
                order.append(v)
                for e in self._out_edges.get(v, []):
                    w = e.target_id
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        prev[w].append(v)
            delta: Dict[str, float] = defaultdict(float)
            for w in reversed(order):
                for v in prev[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != src:
                    betweenness[w] += delta[w]

        norm = max(1, (len(all_ids) - 1) * (len(all_ids) - 2))
        for nid, cent in centralities.items():
            cent.betweenness_approx = betweenness.get(nid, 0.0) / norm

        return centralities

    # ── export ────────────────────────────────────────────────────────────────────

    def export_json(self) -> str:
        """Serialize entire graph to JSON string."""
        data = {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
            "stats": {
                "node_count": self.node_count(),
                "edge_count": self.edge_count(),
            },
        }
        return json.dumps(data, indent=2, default=str)

    def summary(self) -> Dict[str, Any]:
        type_counts: Dict[str, int] = defaultdict(int)
        edge_type_counts: Dict[str, int] = defaultdict(int)
        for n in self._nodes.values():
            type_counts[n.node_type.value] += 1
        for e in self._edges.values():
            edge_type_counts[e.edge_type.value] += 1
        return {
            "nodes": self.node_count(),
            "edges": self.edge_count(),
            "node_types": dict(type_counts),
            "edge_types": dict(edge_type_counts),
        }
