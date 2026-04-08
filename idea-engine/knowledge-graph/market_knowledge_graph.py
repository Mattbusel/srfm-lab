"""
Market Knowledge Graph — structured knowledge representation for trading.

Implements:
  - Entity types: Asset, Sector, Factor, Macro_Indicator, Strategy, Regime
  - Relationship types: correlated_with, causes, hedges, member_of, exposed_to
  - Graph construction from correlation data, sector classifications, factor loadings
  - Reasoning: causal chain inference, pathway search
  - Query: "what is correlated with X?", "what hedges Y?"
  - Knowledge propagation: if A causes B and B causes C, infer A -> C
  - Temporal knowledge: relationships that vary over time/regime
  - Conflict detection: contradictory relationships
  - Embedding: node2vec-style embedding for graph queries
  - Integration: connect to regime oracle, signals, debate system
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict, deque


# ── Entities ──────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """Node in the knowledge graph."""
    id: str
    name: str
    entity_type: str   # asset / sector / factor / indicator / strategy / regime
    attributes: dict = field(default_factory=dict)


@dataclass
class Relationship:
    """Edge in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: str   # correlated_with / causes / hedges / member_of / exposed_to / leads / lags
    weight: float = 1.0  # strength of relationship
    confidence: float = 0.5
    regime: str = "all"  # which regime this relationship holds in
    temporal: bool = False
    lag_days: int = 0
    metadata: dict = field(default_factory=dict)


# ── Knowledge Graph ───────────────────────────────────────────────────────────

class MarketKnowledgeGraph:
    """
    Graph database of market knowledge: entities, relationships, reasoning.
    """

    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self._adj: dict[str, list[Relationship]] = defaultdict(list)
        self._rev_adj: dict[str, list[Relationship]] = defaultdict(list)

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity

    def add_relationship(self, rel: Relationship) -> None:
        self.relationships.append(rel)
        self._adj[rel.source_id].append(rel)
        self._rev_adj[rel.target_id].append(rel)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def get_neighbors(self, entity_id: str, relation_type: Optional[str] = None) -> list[tuple[Entity, Relationship]]:
        """Get outgoing neighbors, optionally filtered by relation type."""
        results = []
        for rel in self._adj.get(entity_id, []):
            if relation_type is None or rel.relation_type == relation_type:
                target = self.entities.get(rel.target_id)
                if target:
                    results.append((target, rel))
        return results

    def get_incoming(self, entity_id: str, relation_type: Optional[str] = None) -> list[tuple[Entity, Relationship]]:
        """Get incoming neighbors."""
        results = []
        for rel in self._rev_adj.get(entity_id, []):
            if relation_type is None or rel.relation_type == relation_type:
                source = self.entities.get(rel.source_id)
                if source:
                    results.append((source, rel))
        return results

    # ── Construction ──────────────────────────────────────────────────────

    def build_from_correlations(
        self,
        asset_names: list[str],
        corr_matrix: np.ndarray,
        threshold: float = 0.5,
        sector_labels: Optional[list[str]] = None,
    ) -> None:
        """Build graph from correlation matrix."""
        n = len(asset_names)
        for i in range(n):
            self.add_entity(Entity(
                id=f"asset_{asset_names[i]}",
                name=asset_names[i],
                entity_type="asset",
                attributes={"sector": sector_labels[i] if sector_labels else "unknown"},
            ))

        # Sector entities
        if sector_labels:
            for sec in set(sector_labels):
                self.add_entity(Entity(id=f"sector_{sec}", name=sec, entity_type="sector"))
            for i, name in enumerate(asset_names):
                self.add_relationship(Relationship(
                    source_id=f"asset_{name}",
                    target_id=f"sector_{sector_labels[i]}",
                    relation_type="member_of",
                    weight=1.0,
                    confidence=1.0,
                ))

        # Correlation edges
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) > threshold:
                    self.add_relationship(Relationship(
                        source_id=f"asset_{asset_names[i]}",
                        target_id=f"asset_{asset_names[j]}",
                        relation_type="correlated_with",
                        weight=float(corr_matrix[i, j]),
                        confidence=min(abs(corr_matrix[i, j]), 1.0),
                    ))
                    # Bidirectional
                    self.add_relationship(Relationship(
                        source_id=f"asset_{asset_names[j]}",
                        target_id=f"asset_{asset_names[i]}",
                        relation_type="correlated_with",
                        weight=float(corr_matrix[i, j]),
                        confidence=min(abs(corr_matrix[i, j]), 1.0),
                    ))

    def add_factor_structure(
        self,
        asset_names: list[str],
        factor_names: list[str],
        factor_loadings: np.ndarray,  # (n_assets, n_factors)
        loading_threshold: float = 0.3,
    ) -> None:
        """Add factor exposure relationships."""
        for fname in factor_names:
            self.add_entity(Entity(id=f"factor_{fname}", name=fname, entity_type="factor"))

        for i, aname in enumerate(asset_names):
            for j, fname in enumerate(factor_names):
                if abs(factor_loadings[i, j]) > loading_threshold:
                    self.add_relationship(Relationship(
                        source_id=f"asset_{aname}",
                        target_id=f"factor_{fname}",
                        relation_type="exposed_to",
                        weight=float(factor_loadings[i, j]),
                        confidence=0.8,
                    ))

    def add_causal_link(
        self,
        source_name: str,
        target_name: str,
        weight: float = 1.0,
        confidence: float = 0.5,
        lag_days: int = 0,
        regime: str = "all",
    ) -> None:
        """Add a causal relationship."""
        self.add_relationship(Relationship(
            source_id=source_name,
            target_id=target_name,
            relation_type="causes",
            weight=weight,
            confidence=confidence,
            lag_days=lag_days,
            regime=regime,
            temporal=lag_days > 0,
        ))

    def add_hedge_relationship(
        self,
        asset_name: str,
        hedge_name: str,
        hedge_ratio: float,
        effectiveness: float,
    ) -> None:
        """Add a hedging relationship."""
        self.add_relationship(Relationship(
            source_id=f"asset_{hedge_name}",
            target_id=f"asset_{asset_name}",
            relation_type="hedges",
            weight=hedge_ratio,
            confidence=effectiveness,
            metadata={"hedge_ratio": hedge_ratio, "effectiveness": effectiveness},
        ))

    # ── Queries ───────────────────────────────────────────────────────────

    def what_correlates_with(self, entity_id: str, min_strength: float = 0.3) -> list[dict]:
        """Find entities correlated with given entity."""
        results = []
        for target, rel in self.get_neighbors(entity_id, "correlated_with"):
            if abs(rel.weight) >= min_strength:
                results.append({
                    "entity": target.name,
                    "correlation": rel.weight,
                    "confidence": rel.confidence,
                })
        results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return results

    def what_hedges(self, entity_id: str) -> list[dict]:
        """Find hedges for given entity."""
        results = []
        for source, rel in self.get_incoming(entity_id, "hedges"):
            results.append({
                "hedge": source.name,
                "hedge_ratio": rel.metadata.get("hedge_ratio", rel.weight),
                "effectiveness": rel.confidence,
            })
        results.sort(key=lambda x: x["effectiveness"], reverse=True)
        return results

    def what_causes(self, entity_id: str) -> list[dict]:
        """Find causal drivers of given entity."""
        results = []
        for source, rel in self.get_incoming(entity_id, "causes"):
            results.append({
                "cause": source.name,
                "strength": rel.weight,
                "confidence": rel.confidence,
                "lag_days": rel.lag_days,
                "regime": rel.regime,
            })
        return results

    def what_is_affected_by(self, entity_id: str) -> list[dict]:
        """Find what is affected by changes in given entity."""
        results = []
        for target, rel in self.get_neighbors(entity_id, "causes"):
            results.append({
                "affected": target.name,
                "strength": rel.weight,
                "confidence": rel.confidence,
                "lag_days": rel.lag_days,
            })
        return results

    def factor_exposure(self, entity_id: str) -> list[dict]:
        """Get factor exposures for an entity."""
        results = []
        for target, rel in self.get_neighbors(entity_id, "exposed_to"):
            results.append({
                "factor": target.name,
                "loading": rel.weight,
                "confidence": rel.confidence,
            })
        return results

    # ── Reasoning ─────────────────────────────────────────────────────────

    def causal_chain(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 4,
    ) -> list[list[str]]:
        """Find causal pathways from source to target (BFS)."""
        if source_id not in self.entities or target_id not in self.entities:
            return []

        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        paths = []

        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth:
                continue

            for target, rel in self.get_neighbors(current, "causes"):
                if target.id == target_id:
                    paths.append(path + [target.id])
                elif target.id not in visited:
                    visited.add(target.id)
                    queue.append((target.id, path + [target.id]))

        return paths

    def transitive_impact(
        self,
        source_id: str,
        shock: float = 1.0,
        max_depth: int = 3,
        decay: float = 0.5,
    ) -> dict[str, float]:
        """
        Propagate a shock through causal graph.
        Impact decays with distance from source.
        """
        impacts = {source_id: shock}
        frontier = [(source_id, shock, 0)]

        while frontier:
            next_frontier = []
            for eid, impact, depth in frontier:
                if depth >= max_depth:
                    continue
                for target, rel in self.get_neighbors(eid, "causes"):
                    propagated = impact * rel.weight * rel.confidence * decay
                    if abs(propagated) > 0.01:
                        current = impacts.get(target.id, 0)
                        impacts[target.id] = current + propagated
                        next_frontier.append((target.id, propagated, depth + 1))
            frontier = next_frontier

        return {k: float(v) for k, v in sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)}

    def conflict_detection(self) -> list[dict]:
        """Find contradictory relationships in the graph."""
        conflicts = []
        # Check: if A causes B positively AND A hedges B, that's contradictory
        for entity_id in self.entities:
            causes = {r.target_id: r.weight for r in self._adj.get(entity_id, []) if r.relation_type == "causes"}
            hedges = {r.target_id: r.weight for r in self._adj.get(entity_id, []) if r.relation_type == "hedges"}

            for tid in set(causes) & set(hedges):
                if causes[tid] > 0:  # positive causation + hedging = conflict
                    conflicts.append({
                        "source": entity_id,
                        "target": tid,
                        "conflict": "positive_cause_and_hedge",
                        "cause_weight": causes[tid],
                        "hedge_weight": hedges[tid],
                    })

        return conflicts

    # ── Embedding ─────────────────────────────────────────────────────────

    def compute_embeddings(self, dim: int = 16, n_walks: int = 50, walk_length: int = 10, seed: int = 42) -> dict[str, np.ndarray]:
        """
        Simple node2vec-style embeddings via random walks + skip-gram proxy.
        """
        rng = np.random.default_rng(seed)
        entity_ids = list(self.entities.keys())
        if not entity_ids:
            return {}

        n = len(entity_ids)
        id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

        # Co-occurrence matrix from random walks
        cooccur = np.zeros((n, n))

        for _ in range(n_walks):
            start = rng.choice(entity_ids)
            walk = [start]
            current = start

            for _ in range(walk_length):
                neighbors = self._adj.get(current, [])
                if not neighbors:
                    break
                rel = neighbors[rng.integers(len(neighbors))]
                current = rel.target_id
                walk.append(current)

            # Update co-occurrence within window
            for i, w1 in enumerate(walk):
                for j in range(max(0, i - 3), min(len(walk), i + 4)):
                    if i != j and w1 in id_to_idx and walk[j] in id_to_idx:
                        cooccur[id_to_idx[w1], id_to_idx[walk[j]]] += 1

        # SVD on co-occurrence for embeddings
        cooccur_log = np.log(cooccur + 1)
        U, S, Vt = np.linalg.svd(cooccur_log, full_matrices=False)
        k = min(dim, len(S))
        embeddings = U[:, :k] * np.sqrt(S[:k])

        return {eid: embeddings[idx] for eid, idx in id_to_idx.items()}

    def similar_entities(self, entity_id: str, embeddings: dict[str, np.ndarray], top_k: int = 5) -> list[dict]:
        """Find similar entities by embedding distance."""
        if entity_id not in embeddings:
            return []

        target = embeddings[entity_id]
        similarities = []
        for eid, emb in embeddings.items():
            if eid != entity_id:
                cos_sim = float(np.dot(target, emb) / (np.linalg.norm(target) * np.linalg.norm(emb) + 1e-10))
                similarities.append({"entity": eid, "name": self.entities[eid].name, "similarity": cos_sim})

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

    # ── Analytics ─────────────────────────────────────────────────────────

    def graph_summary(self) -> dict:
        """Summary statistics of the knowledge graph."""
        entity_types = defaultdict(int)
        for e in self.entities.values():
            entity_types[e.entity_type] += 1

        relation_types = defaultdict(int)
        for r in self.relationships:
            relation_types[r.relation_type] += 1

        return {
            "n_entities": len(self.entities),
            "n_relationships": len(self.relationships),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
            "avg_degree": float(len(self.relationships) * 2 / max(len(self.entities), 1)),
        }

    def regime_subgraph(self, regime: str) -> list[Relationship]:
        """Get relationships active in a specific regime."""
        return [r for r in self.relationships if r.regime in (regime, "all")]

    def most_connected(self, top_k: int = 10) -> list[dict]:
        """Find most connected entities."""
        degree = defaultdict(int)
        for rel in self.relationships:
            degree[rel.source_id] += 1
            degree[rel.target_id] += 1

        sorted_deg = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"entity": eid, "name": self.entities[eid].name if eid in self.entities else eid, "degree": d}
                for eid, d in sorted_deg]


# ── Pre-built Knowledge Templates ─────────────────────────────────────────────

def build_default_market_knowledge() -> MarketKnowledgeGraph:
    """Build a default knowledge graph with common market relationships."""
    kg = MarketKnowledgeGraph()

    # Macro indicators
    for ind in ["fed_funds_rate", "cpi_yoy", "gdp_growth", "unemployment",
                "pmi_manufacturing", "consumer_confidence", "yield_curve_2s10s",
                "vix", "credit_ig_oas", "credit_hy_oas", "dxy", "oil_price"]:
        kg.add_entity(Entity(id=ind, name=ind, entity_type="indicator"))

    # Regimes
    for r in REGIME_NAMES if 'REGIME_NAMES' in dir() else ["risk_on", "risk_off", "crisis", "recovery"]:
        kg.add_entity(Entity(id=f"regime_{r}", name=r, entity_type="regime"))

    # Causal relationships
    causal_links = [
        ("fed_funds_rate", "yield_curve_2s10s", -0.6, 0.8, 0),
        ("fed_funds_rate", "dxy", 0.4, 0.6, 5),
        ("cpi_yoy", "fed_funds_rate", 0.5, 0.7, 30),
        ("oil_price", "cpi_yoy", 0.3, 0.5, 20),
        ("pmi_manufacturing", "gdp_growth", 0.6, 0.8, 60),
        ("vix", "credit_hy_oas", 0.7, 0.8, 0),
        ("credit_hy_oas", "regime_crisis", 0.5, 0.6, 5),
        ("yield_curve_2s10s", "gdp_growth", 0.4, 0.6, 180),
        ("unemployment", "consumer_confidence", -0.5, 0.7, 15),
        ("dxy", "oil_price", -0.3, 0.5, 0),
    ]

    for src, tgt, weight, conf, lag in causal_links:
        kg.add_causal_link(src, tgt, weight, conf, lag)

    return kg
