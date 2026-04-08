"""
Hypothesis genealogy tracker — tracks lineage, mutation, and evolution of trading ideas.

Maintains a directed acyclic graph of hypothesis relationships:
  - Parent → child via mutation or refinement
  - Sibling → sibling via crossover
  - Convergent: multiple independent hypotheses pointing at same trade
  - Divergent: one hypothesis spawns specialized variants

Enables:
  - "This idea is a refinement of X which died in 2023 regime"
  - "Ideas from this parent lineage historically over-fit"
  - "Three independent miners found the same trade — high conviction"
  - Automatic hypothesis merging when variants converge
"""

from __future__ import annotations
import json
import math
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


# ── Hypothesis Node ───────────────────────────────────────────────────────────

@dataclass
class HypothesisNode:
    id: str
    name: str
    description: str
    template: str
    params: dict
    tags: list[str]
    created_at: float = field(default_factory=time.time)
    parent_ids: list[str] = field(default_factory=list)
    children_ids: list[str] = field(default_factory=list)
    generation: int = 0

    # Performance history
    backtest_sharpe: Optional[float] = None
    backtest_calmar: Optional[float] = None
    live_pnl: float = 0.0
    live_trades: int = 0
    win_rate: Optional[float] = None

    # Lifecycle
    status: str = "active"   # active, retired, merged, superseded
    retired_reason: Optional[str] = None
    superseded_by: Optional[str] = None
    confidence: float = 0.5
    conviction: float = 0.5

    # Discovery metadata
    miner_source: Optional[str] = None
    regime_at_creation: Optional[str] = None
    discovery_method: str = "manual"   # manual, miner, genetic, serendipity, debate

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "params": self.params,
            "tags": self.tags,
            "created_at": self.created_at,
            "parent_ids": self.parent_ids,
            "children_ids": self.children_ids,
            "generation": self.generation,
            "backtest_sharpe": self.backtest_sharpe,
            "backtest_calmar": self.backtest_calmar,
            "live_pnl": self.live_pnl,
            "live_trades": self.live_trades,
            "win_rate": self.win_rate,
            "status": self.status,
            "retired_reason": self.retired_reason,
            "superseded_by": self.superseded_by,
            "confidence": self.confidence,
            "conviction": self.conviction,
            "miner_source": self.miner_source,
            "regime_at_creation": self.regime_at_creation,
            "discovery_method": self.discovery_method,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HypothesisNode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Genealogy Graph ───────────────────────────────────────────────────────────

class HypothesisGenealogy:
    """
    DAG of hypothesis relationships. Tracks lineage, performance inheritance,
    and convergent/divergent idea patterns.
    """

    def __init__(self):
        self.nodes: dict[str, HypothesisNode] = {}
        self.edges: dict[str, list[str]] = defaultdict(list)   # parent → children
        self.reverse_edges: dict[str, list[str]] = defaultdict(list)  # child → parents
        self.convergence_groups: dict[str, list[str]] = defaultdict(list)  # trade → [hypotheses]
        self._lineage_performance_cache: dict[str, float] = {}

    # ── Node management ───────────────────────────────────────────────────────

    def add(self, node: HypothesisNode) -> None:
        """Register a new hypothesis node."""
        self.nodes[node.id] = node
        for pid in node.parent_ids:
            self.edges[pid].append(node.id)
            self.reverse_edges[node.id].append(pid)
            if pid in self.nodes:
                self.nodes[pid].children_ids.append(node.id)

    def retire(self, node_id: str, reason: str) -> None:
        """Retire a hypothesis with a reason."""
        if node_id in self.nodes:
            self.nodes[node_id].status = "retired"
            self.nodes[node_id].retired_reason = reason

    def supersede(self, old_id: str, new_id: str) -> None:
        """Mark old hypothesis as superseded by new one."""
        if old_id in self.nodes:
            self.nodes[old_id].status = "superseded"
            self.nodes[old_id].superseded_by = new_id

    def merge(self, ids: list[str], merged_node: HypothesisNode) -> None:
        """Merge multiple hypotheses into one (convergent discovery)."""
        merged_node.parent_ids = ids
        merged_node.generation = max(self.nodes[i].generation for i in ids if i in self.nodes) + 1
        for old_id in ids:
            if old_id in self.nodes:
                self.nodes[old_id].status = "merged"
                self.nodes[old_id].superseded_by = merged_node.id
        self.add(merged_node)

    # ── Lineage analysis ──────────────────────────────────────────────────────

    def ancestors(self, node_id: str, depth: int = 10) -> list[str]:
        """Return all ancestor IDs up to given depth."""
        visited = set()
        frontier = [node_id]
        result = []
        for _ in range(depth):
            next_frontier = []
            for nid in frontier:
                for pid in self.reverse_edges.get(nid, []):
                    if pid not in visited:
                        visited.add(pid)
                        result.append(pid)
                        next_frontier.append(pid)
            frontier = next_frontier
            if not frontier:
                break
        return result

    def descendants(self, node_id: str, depth: int = 10) -> list[str]:
        """Return all descendant IDs up to given depth."""
        visited = set()
        frontier = [node_id]
        result = []
        for _ in range(depth):
            next_frontier = []
            for nid in frontier:
                for cid in self.edges.get(nid, []):
                    if cid not in visited:
                        visited.add(cid)
                        result.append(cid)
                        next_frontier.append(cid)
            frontier = next_frontier
            if not frontier:
                break
        return result

    def lineage_performance(self, node_id: str) -> dict:
        """
        Aggregate performance statistics across a lineage.
        Used to identify whether a hypothesis family has edge.
        """
        all_ids = [node_id] + self.ancestors(node_id)
        nodes = [self.nodes[i] for i in all_ids if i in self.nodes]

        sharpes = [n.backtest_sharpe for n in nodes if n.backtest_sharpe is not None]
        pnls = [n.live_pnl for n in nodes]
        win_rates = [n.win_rate for n in nodes if n.win_rate is not None]

        return {
            "lineage_size": len(nodes),
            "avg_sharpe": float(sum(sharpes) / len(sharpes)) if sharpes else None,
            "best_sharpe": float(max(sharpes)) if sharpes else None,
            "total_live_pnl": float(sum(pnls)),
            "avg_win_rate": float(sum(win_rates) / len(win_rates)) if win_rates else None,
            "active_count": sum(1 for n in nodes if n.status == "active"),
            "retired_count": sum(1 for n in nodes if n.status == "retired"),
            "lineage_health": self._lineage_health(nodes),
        }

    def _lineage_health(self, nodes: list[HypothesisNode]) -> str:
        """Classify lineage health based on performance history."""
        sharpes = [n.backtest_sharpe for n in nodes if n.backtest_sharpe is not None]
        if not sharpes:
            return "unproven"
        avg = sum(sharpes) / len(sharpes)
        if avg > 1.5:
            return "excellent"
        elif avg > 0.8:
            return "good"
        elif avg > 0.0:
            return "marginal"
        else:
            return "negative_edge"

    # ── Convergence detection ─────────────────────────────────────────────────

    def register_convergent(self, trade_key: str, hypothesis_id: str) -> dict:
        """
        Register that a hypothesis points at a specific trade.
        Returns convergence info: how many independent hypotheses agree.
        """
        self.convergence_groups[trade_key].append(hypothesis_id)
        ids = self.convergence_groups[trade_key]

        # Check independence: different parents = more independent
        parent_sets = []
        for hid in ids:
            if hid in self.nodes:
                parent_sets.append(set(self.ancestors(hid)))

        # Independence score: fraction of pairs with no shared ancestors
        independent_pairs = 0
        total_pairs = 0
        for i in range(len(parent_sets)):
            for j in range(i + 1, len(parent_sets)):
                total_pairs += 1
                if not parent_sets[i] & parent_sets[j]:
                    independent_pairs += 1

        independence = float(independent_pairs / max(total_pairs, 1))
        conviction_boost = 1.0 + independence * (len(ids) - 1) * 0.2

        return {
            "trade_key": trade_key,
            "n_hypotheses": len(ids),
            "hypothesis_ids": ids,
            "independence_score": independence,
            "conviction_multiplier": conviction_boost,
            "high_conviction": bool(len(ids) >= 3 and independence > 0.5),
        }

    # ── Mutation and spawning ─────────────────────────────────────────────────

    def spawn_variant(
        self,
        parent_id: str,
        param_overrides: dict,
        description_suffix: str = "",
        method: str = "refinement",
    ) -> HypothesisNode:
        """Spawn a child hypothesis by modifying parent parameters."""
        if parent_id not in self.nodes:
            raise ValueError(f"Parent {parent_id} not found")

        parent = self.nodes[parent_id]
        new_params = {**parent.params, **param_overrides}
        child_id = _make_id(parent.name + str(param_overrides) + str(time.time()))

        child = HypothesisNode(
            id=child_id,
            name=f"{parent.name}_v{parent.generation + 1}",
            description=parent.description + (f" — {description_suffix}" if description_suffix else ""),
            template=parent.template,
            params=new_params,
            tags=parent.tags.copy(),
            parent_ids=[parent_id],
            generation=parent.generation + 1,
            regime_at_creation=parent.regime_at_creation,
            discovery_method=method,
        )
        self.add(child)
        return child

    def crossover_spawn(
        self,
        parent_a_id: str,
        parent_b_id: str,
        param_blend: float = 0.5,
    ) -> HypothesisNode:
        """
        Crossover two hypotheses to create a hybrid.
        param_blend: fraction from parent_a (rest from parent_b).
        """
        a = self.nodes[parent_a_id]
        b = self.nodes[parent_b_id]

        # Blend numeric params, take tags union
        blended_params = {}
        all_keys = set(a.params) | set(b.params)
        for k in all_keys:
            va = a.params.get(k, b.params.get(k))
            vb = b.params.get(k, a.params.get(k))
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                blended_params[k] = type(va)(param_blend * va + (1 - param_blend) * vb)
            else:
                blended_params[k] = va if param_blend >= 0.5 else vb

        child_id = _make_id(parent_a_id + parent_b_id + str(time.time()))
        child = HypothesisNode(
            id=child_id,
            name=f"hybrid_{a.name[:12]}_{b.name[:12]}",
            description=f"Crossover: {a.name} × {b.name}",
            template=a.template if param_blend >= 0.5 else b.template,
            params=blended_params,
            tags=list(set(a.tags + b.tags)),
            parent_ids=[parent_a_id, parent_b_id],
            generation=max(a.generation, b.generation) + 1,
            discovery_method="genetic_crossover",
        )
        self.add(child)
        return child

    # ── Search and retrieval ──────────────────────────────────────────────────

    def search(
        self,
        tags: Optional[list[str]] = None,
        status: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        regime: Optional[str] = None,
        discovery_method: Optional[str] = None,
        top_k: int = 20,
    ) -> list[HypothesisNode]:
        """Search hypothesis bank with filters."""
        results = list(self.nodes.values())
        if tags:
            results = [n for n in results if any(t in n.tags for t in tags)]
        if status:
            results = [n for n in results if n.status == status]
        if min_sharpe is not None:
            results = [n for n in results if n.backtest_sharpe is not None and n.backtest_sharpe >= min_sharpe]
        if regime:
            results = [n for n in results if n.regime_at_creation == regime]
        if discovery_method:
            results = [n for n in results if n.discovery_method == discovery_method]

        # Sort by conviction * confidence
        results.sort(key=lambda n: n.conviction * n.confidence, reverse=True)
        return results[:top_k]

    def similar_hypotheses(self, node_id: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Find similar hypotheses by tag overlap and template."""
        if node_id not in self.nodes:
            return []
        ref = self.nodes[node_id]
        ref_tags = set(ref.tags)
        scores = []
        for nid, node in self.nodes.items():
            if nid == node_id:
                continue
            tag_sim = len(ref_tags & set(node.tags)) / max(len(ref_tags | set(node.tags)), 1)
            template_match = 1.0 if node.template == ref.template else 0.0
            score = 0.7 * tag_sim + 0.3 * template_match
            scores.append((nid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps({
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": dict(self.edges),
            "convergence_groups": dict(self.convergence_groups),
        }, indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> "HypothesisGenealogy":
        data = json.loads(s)
        g = cls()
        for node_data in data.get("nodes", {}).values():
            try:
                g.nodes[node_data["id"]] = HypothesisNode.from_dict(node_data)
            except Exception:
                pass
        g.edges = defaultdict(list, data.get("edges", {}))
        g.convergence_groups = defaultdict(list, data.get("convergence_groups", {}))
        # Rebuild reverse edges
        for pid, children in g.edges.items():
            for cid in children:
                g.reverse_edges[cid].append(pid)
        return g

    # ── Statistics ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """High-level genealogy statistics."""
        nodes = list(self.nodes.values())
        statuses = {}
        for n in nodes:
            statuses[n.status] = statuses.get(n.status, 0) + 1

        methods = {}
        for n in nodes:
            methods[n.discovery_method] = methods.get(n.discovery_method, 0) + 1

        generations = [n.generation for n in nodes]
        sharpes = [n.backtest_sharpe for n in nodes if n.backtest_sharpe is not None]

        return {
            "total_hypotheses": len(nodes),
            "by_status": statuses,
            "by_discovery_method": methods,
            "max_generation": max(generations) if generations else 0,
            "avg_generation": float(sum(generations) / len(generations)) if generations else 0,
            "avg_backtest_sharpe": float(sum(sharpes) / len(sharpes)) if sharpes else None,
            "high_conviction_trades": sum(
                1 for grp in self.convergence_groups.values() if len(grp) >= 3
            ),
            "total_lineages": len([n for n in nodes if not n.parent_ids]),
        }


def _make_id(seed: str) -> str:
    return "hyp_" + hashlib.md5(seed.encode()).hexdigest()[:12]
