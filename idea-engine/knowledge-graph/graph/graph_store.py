"""
SQLite persistence for the knowledge graph.

Schema: nodes table + edges table.  Node/edge properties are stored as JSON
text columns.  Supports efficient queries by type, confidence, and edge type.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .node import BaseNode, NodeType
from .edge import Edge, EdgeType
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

DEFAULT_DB = Path(__file__).parent.parent / "knowledge_graph.db"

DDL = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id      TEXT PRIMARY KEY,
    node_type    TEXT NOT NULL,
    label        TEXT NOT NULL,
    properties   TEXT,       -- JSON
    confidence   REAL NOT NULL DEFAULT 0.5,
    created_at   TEXT NOT NULL,
    updated_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_nodes_type       ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON nodes(confidence);
CREATE INDEX IF NOT EXISTS idx_nodes_label      ON nodes(label);

CREATE TABLE IF NOT EXISTS edges (
    edge_id        TEXT PRIMARY KEY,
    source_id      TEXT NOT NULL,
    target_id      TEXT NOT NULL,
    edge_type      TEXT NOT NULL,
    weight         REAL NOT NULL DEFAULT 1.0,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    last_updated   TEXT NOT NULL,
    properties     TEXT   -- JSON
);

CREATE INDEX IF NOT EXISTS idx_edges_source    ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target    ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type      ON edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_src_type  ON edges(source_id, edge_type);
"""


class GraphStore:
    """
    Persists and loads a KnowledgeGraph to/from SQLite.

    Usage::

        store = GraphStore()
        store.save_node(node)
        store.save_edge(edge)
        graph = store.load_graph()
    """

    def __init__(self, db_path: Path = DEFAULT_DB) -> None:
        self._db_path = db_path
        self._init_db()

    # ── lifecycle ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(DDL)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── node persistence ─────────────────────────────────────────────────────────

    def save_node(self, node: BaseNode) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO nodes
                   (node_id, node_type, label, properties, confidence, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    node.node_id,
                    node.node_type.value,
                    node.label,
                    json.dumps(node.properties, default=str),
                    node.confidence,
                    node.created_at.isoformat(),
                    node.updated_at.isoformat() if node.updated_at else None,
                ),
            )

    def save_nodes(self, nodes: List[BaseNode]) -> int:
        for n in nodes:
            self.save_node(n)
        return len(nodes)

    def load_node_row(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM nodes WHERE node_type = ? ORDER BY confidence DESC",
                (node_type,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_nodes_with_min_confidence(self, min_conf: float) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM nodes WHERE confidence >= ? ORDER BY confidence DESC",
                (min_conf,),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_nodes_by_label(self, pattern: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM nodes WHERE label LIKE ?", (f"%{pattern}%",)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── edge persistence ─────────────────────────────────────────────────────────

    def save_edge(self, edge: Edge) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO edges
                   (edge_id, source_id, target_id, edge_type, weight,
                    evidence_count, last_updated, properties)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    edge.edge_id,
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type.value,
                    edge.weight,
                    edge.evidence_count,
                    edge.last_updated.isoformat(),
                    json.dumps(edge.properties, default=str),
                ),
            )

    def save_edges(self, edges: List[Edge]) -> int:
        for e in edges:
            self.save_edge(e)
        return len(edges)

    def get_edges_by_type(self, edge_type: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM edges WHERE edge_type = ? ORDER BY weight DESC",
                (edge_type,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_edges_from_node(self, node_id: str, edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            if edge_type:
                rows = conn.execute(
                    "SELECT * FROM edges WHERE source_id = ? AND edge_type = ?",
                    (node_id, edge_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM edges WHERE source_id = ?", (node_id,)
                ).fetchall()
        return [dict(r) for r in rows]

    def get_subgraph_around(self, node_id: str, hops: int = 2) -> Dict[str, Any]:
        """Return all nodes and edges within *hops* of *node_id*."""
        visited_nodes: set[str] = set()
        frontier = {node_id}
        all_edges: List[Dict[str, Any]] = []

        for _ in range(hops):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited_nodes:
                    continue
                visited_nodes.add(nid)
                edges = self.get_edges_from_node(nid)
                all_edges.extend(edges)
                for e in edges:
                    next_frontier.add(e["target_id"])
            frontier = next_frontier - visited_nodes

        node_rows: List[Dict[str, Any]] = []
        for nid in visited_nodes:
            row = self.load_node_row(nid)
            if row:
                node_rows.append(row)

        return {"nodes": node_rows, "edges": all_edges}

    # ── full graph load ──────────────────────────────────────────────────────────

    def load_graph(self) -> KnowledgeGraph:
        """Reconstruct an in-memory KnowledgeGraph from the database."""
        from .node import (
            InstrumentNode, SignalNode, HypothesisNode, RegimeNode,
            PatternNode, EventNode, ParameterNode,
        )

        kg = KnowledgeGraph()

        # Load all nodes as generic BaseNode (reconstructing exact subtype is optional)
        with self._conn() as conn:
            node_rows = conn.execute("SELECT * FROM nodes").fetchall()
            edge_rows = conn.execute("SELECT * FROM edges").fetchall()

        for row in node_rows:
            props = json.loads(row["properties"] or "{}")
            node = BaseNode(
                node_id=row["node_id"],
                node_type=NodeType(row["node_type"]),
                label=row["label"],
                properties=props,
                confidence=row["confidence"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            kg.add_node(node)

        for row in edge_rows:
            props = json.loads(row["properties"] or "{}")
            try:
                et = EdgeType(row["edge_type"])
            except ValueError:
                et = EdgeType.CORRELATES_WITH
            edge = Edge(
                source_id=row["source_id"],
                target_id=row["target_id"],
                edge_type=et,
                weight=row["weight"],
                evidence_count=row["evidence_count"],
                last_updated=datetime.fromisoformat(row["last_updated"]),
                properties=props,
                edge_id=row["edge_id"],
            )
            kg.add_edge(edge)

        logger.info(
            "Loaded graph: %d nodes, %d edges from %s",
            kg.node_count(), kg.edge_count(), self._db_path,
        )
        return kg

    def save_graph(self, kg: KnowledgeGraph) -> None:
        """Persist entire in-memory graph to database."""
        for node in kg._nodes.values():
            self.save_node(node)
        for edge in kg._edges.values():
            self.save_edge(edge)
        logger.info("Saved graph: %d nodes, %d edges", kg.node_count(), kg.edge_count())
