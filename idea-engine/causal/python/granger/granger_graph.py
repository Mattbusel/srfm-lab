"""
causal/python/granger/granger_graph.py

CausalGraph: builds a networkx DiGraph from GrangerResult edges.

Features:
    - Filters by significance threshold
    - get_causal_parents(feature) → list of features that Granger-cause it
    - get_causal_children(feature) → features it Granger-causes
    - Writes edges to the `causal_edges` table in idea_engine.db
    - Provides graph-level statistics (hub nodes, isolated nodes, etc.)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from causal.python.granger.granger_tests import GrangerEdge, GrangerResult

log = logging.getLogger(__name__)

DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")

# DDL for causal_edges table
_CREATE_CAUSAL_EDGES = """
CREATE TABLE IF NOT EXISTS causal_edges (
    edge_id         TEXT PRIMARY KEY,
    cause           TEXT NOT NULL,
    effect          TEXT NOT NULL,
    optimal_lag     INTEGER NOT NULL,
    p_value         REAL NOT NULL,
    raw_p_value     REAL NOT NULL,
    f_statistic     REAL NOT NULL,
    effect_size     REAL NOT NULL,
    significant     INTEGER NOT NULL DEFAULT 1,
    aic_at_lag      REAL,
    bic_at_lag      REAL,
    discovered_at   TEXT NOT NULL,
    instrument      TEXT,
    UNIQUE(cause, effect, optimal_lag)
);
"""

_CREATE_CAUSAL_EDGES_IDX = """
CREATE INDEX IF NOT EXISTS idx_causal_edges_effect ON causal_edges (effect);
"""


# ---------------------------------------------------------------------------
# Causal Graph
# ---------------------------------------------------------------------------

class CausalGraph:
    """
    Directed graph of Granger causal relationships.

    Parameters
    ----------
    significance_threshold : p-value threshold for edge inclusion
    db_path                : path to idea_engine.db for persistence
    instrument             : instrument tag for DB records (optional)
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,
        db_path: Path | str = DB_PATH,
        instrument: str | None = None,
    ) -> None:
        self.threshold = significance_threshold
        self.db_path = Path(db_path)
        self.instrument = instrument
        self.graph: nx.DiGraph = nx.DiGraph()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Build from GrangerResult
    # ------------------------------------------------------------------

    def build_from_result(self, result: GrangerResult) -> "CausalGraph":
        """
        Populate the internal DiGraph from a GrangerResult.
        Only adds edges where edge.p_value < significance_threshold.
        """
        self.graph.clear()

        # Add all features as nodes
        for feature in result.features_tested:
            self.graph.add_node(feature)

        # Add significant edges
        added = 0
        for edge in result.edges:
            if edge.p_value < self.threshold:
                self.graph.add_edge(
                    edge.cause,
                    edge.effect,
                    lag=edge.optimal_lag,
                    p_value=edge.p_value,
                    raw_p_value=edge.raw_p_value,
                    f_statistic=edge.f_statistic,
                    effect_size=edge.effect_size,
                    aic=edge.aic_at_lag,
                    bic=edge.bic_at_lag,
                    weight=edge.effect_size,
                )
                added += 1

        log.info(
            "CausalGraph built: %d nodes, %d edges (threshold=%.3f)",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self.threshold,
        )
        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_causal_parents(self, feature: str) -> list[str]:
        """
        Return all features that Granger-cause the given feature.
        i.e. the set of direct parents in the DAG.
        """
        if feature not in self.graph:
            return []
        return list(self.graph.predecessors(feature))

    def get_causal_children(self, feature: str) -> list[str]:
        """Return all features that the given feature Granger-causes."""
        if feature not in self.graph:
            return []
        return list(self.graph.successors(feature))

    def get_edge_data(self, cause: str, effect: str) -> dict[str, Any] | None:
        """Return edge attributes for cause → effect, or None if edge doesn't exist."""
        if self.graph.has_edge(cause, effect):
            return dict(self.graph[cause][effect])
        return None

    def get_all_edges(self) -> list[dict[str, Any]]:
        """Return all edges as a list of dicts."""
        edges = []
        for cause, effect, data in self.graph.edges(data=True):
            edges.append({
                "cause": cause,
                "effect": effect,
                **data,
            })
        return sorted(edges, key=lambda x: x.get("p_value", 1.0))

    def ancestors_of(self, feature: str) -> set[str]:
        """All ancestors (transitive parents) of feature."""
        if feature not in self.graph:
            return set()
        return nx.ancestors(self.graph, feature)

    def descendants_of(self, feature: str) -> set[str]:
        """All descendants (transitive children) of feature."""
        if feature not in self.graph:
            return set()
        return nx.descendants(self.graph, feature)

    # ------------------------------------------------------------------
    # Graph statistics
    # ------------------------------------------------------------------

    def hub_nodes(self, top_n: int = 5) -> list[tuple[str, int]]:
        """Return top-N nodes by out-degree (most causal influence)."""
        out_degrees = [(n, self.graph.out_degree(n)) for n in self.graph.nodes]
        return sorted(out_degrees, key=lambda x: x[1], reverse=True)[:top_n]

    def isolated_nodes(self) -> list[str]:
        """Nodes with no edges at all."""
        return [n for n in self.graph.nodes if self.graph.degree(n) == 0]

    def summary(self) -> dict[str, Any]:
        g = self.graph
        return {
            "n_nodes": g.number_of_nodes(),
            "n_edges": g.number_of_edges(),
            "density": round(nx.density(g), 6),
            "is_dag": nx.is_directed_acyclic_graph(g),
            "hub_nodes": self.hub_nodes(5),
            "isolated_nodes": self.isolated_nodes(),
            "weakly_connected_components": nx.number_weakly_connected_components(g),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_db(self) -> int:
        """
        Write all significant edges to the causal_edges table.
        Uses INSERT OR REPLACE to update existing edges.
        Returns number of rows written.
        """
        import uuid as _uuid

        edges = self.get_all_edges()
        if not edges:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for e in edges:
            rows.append((
                str(_uuid.uuid4()),
                e["cause"],
                e["effect"],
                int(e.get("lag", 1)),
                float(e.get("p_value", 1.0)),
                float(e.get("raw_p_value", 1.0)),
                float(e.get("f_statistic", 0.0)),
                float(e.get("effect_size", 0.0)),
                1,
                e.get("aic"),
                e.get("bic"),
                now,
                self.instrument,
            ))

        sql = """
            INSERT OR REPLACE INTO causal_edges
            (edge_id, cause, effect, optimal_lag, p_value, raw_p_value,
             f_statistic, effect_size, significant, aic_at_lag, bic_at_lag,
             discovered_at, instrument)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.executemany(sql, rows)
            conn.commit()

        log.info("Saved %d causal edges to DB", len(rows))
        return len(rows)

    def load_from_db(
        self, instrument: str | None = None, min_effect_size: float = 0.0
    ) -> "CausalGraph":
        """
        Load edges from the causal_edges table and rebuild the graph.
        """
        self.graph.clear()
        sql = "SELECT * FROM causal_edges WHERE significant = 1"
        params: list[Any] = []
        if instrument is not None:
            sql += " AND instrument = ?"
            params.append(instrument)
        if min_effect_size > 0:
            sql += " AND effect_size >= ?"
            params.append(min_effect_size)

        try:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql, params).fetchall()
        except Exception as exc:
            log.error("Failed to load causal edges from DB: %s", exc)
            return self

        for row in rows:
            self.graph.add_edge(
                row["cause"],
                row["effect"],
                lag=row["optimal_lag"],
                p_value=row["p_value"],
                raw_p_value=row["raw_p_value"],
                f_statistic=row["f_statistic"],
                effect_size=row["effect_size"],
                weight=row["effect_size"],
                aic=row["aic_at_lag"],
                bic=row["bic_at_lag"],
            )

        log.info("Loaded %d causal edges from DB", self.graph.number_of_edges())
        return self

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(_CREATE_CAUSAL_EDGES)
                conn.execute(_CREATE_CAUSAL_EDGES_IDX)
                conn.commit()
        except Exception as exc:
            log.warning("Could not initialise causal_edges schema: %s", exc)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))
