"""
AutoPopulator: populates the knowledge graph from live IAE data sources.

Sources:
  1. IAE hypotheses database (SQLite — idea_engine.db or hypothesis store)
  2. Causal discovery results (Julia causal_discovery.jl JSON output)
  3. Historical backtest results (which parameter changes improved what)
  4. On-chain signal correlations (pre-computed correlation matrix)

Runs on startup, then can be called incrementally as new data arrives.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..graph.knowledge_graph import KnowledgeGraph
from ..graph.node import (
    HypothesisNode, InstrumentNode, SignalNode, RegimeNode,
    PatternNode, ParameterNode, EventNode,
)
from ..graph.edge import Edge, EdgeType
from ..graph.graph_store import GraphStore
from ..reasoning.pattern_library import PatternLibrary
from ..reasoning.hypothesis_linker import HypothesisLinker

logger = logging.getLogger(__name__)

# Default path for the main IAE database
DEFAULT_IAE_DB = Path(__file__).parents[3] / "idea_engine.db"
DEFAULT_CAUSAL_JSON = Path(__file__).parents[3] / "causal" / "causal_results.json"


class AutoPopulator:
    """
    Orchestrates population of the knowledge graph from all available sources.

    Usage::

        pop = AutoPopulator(kg)
        stats = pop.run_full_population()
        print(stats)  # {'hypotheses': 142, 'instruments': 18, ...}
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        iae_db_path: Path = DEFAULT_IAE_DB,
        causal_json_path: Path = DEFAULT_CAUSAL_JSON,
    ) -> None:
        self._kg = kg
        self._iae_db = iae_db_path
        self._causal_json = causal_json_path
        self._linker = HypothesisLinker(kg)
        self._pattern_lib = PatternLibrary(kg)

    # ── main entry point ─────────────────────────────────────────────────────────

    def run_full_population(self) -> Dict[str, int]:
        """Run all population tasks. Returns counts per source."""
        stats: Dict[str, int] = {}

        stats["patterns"] = self._pattern_lib.populate_graph()

        n = self._populate_from_iae_db()
        stats["hypotheses_from_db"] = n

        n = self._populate_from_causal_json()
        stats["causal_edges"] = n

        n = self._populate_instruments()
        stats["instruments"] = n

        n = self._populate_regimes()
        stats["regimes"] = n

        logger.info("AutoPopulator complete: %s", stats)
        return stats

    def run_incremental(self, since: Optional[datetime] = None) -> Dict[str, int]:
        """
        Re-populate only data newer than *since*.
        Falls back to full population if since is None.
        """
        if since is None:
            return self.run_full_population()
        return {"hypotheses_from_db": self._populate_from_iae_db(since=since)}

    # ── population tasks ──────────────────────────────────────────────────────────

    def _populate_instruments(self) -> int:
        """Seed the graph with all known instruments."""
        INSTRUMENTS = [
            ("BTC", "bitcoin"),  ("ETH", "ethereum"), ("SOL", "solana"),
            ("AVAX", "avalanche"), ("ARB", "arbitrum"), ("OP", "optimism"),
            ("APT", "aptos"),    ("BNB", "binance_coin"), ("MATIC", "polygon"),
            ("LINK", "chainlink"), ("UNI", "uniswap"),  ("AAVE", "aave"),
            ("DOT", "polkadot"), ("ADA", "cardano"),   ("NEAR", "near"),
            ("INJ", "injective"), ("FTM", "fantom"),   ("JUP", "jupiter"),
        ]
        added = 0
        for symbol, name in INSTRUMENTS:
            node = InstrumentNode(symbol=symbol, confidence=0.99)
            node.set_property("full_name", name)
            nid = self._kg.add_node(node)
            added += 1
        return added

    def _populate_regimes(self) -> int:
        """Seed the graph with standard market regimes."""
        REGIMES = [
            ("bull", "volatility", "Sustained uptrend, positive sentiment"),
            ("bear", "volatility", "Sustained downtrend, negative sentiment"),
            ("high_vol", "volatility", "BVIV or realised vol above 80th pctile"),
            ("low_vol", "volatility", "BVIV below 30th pctile, tight range"),
            ("trending", "trend", "Directional price movement, ADX > 25"),
            ("ranging", "trend", "Sideways price action, ADX < 20"),
            ("risk_on", "sentiment", "Equities rising, crypto correlating"),
            ("risk_off", "sentiment", "Flight to safety, crypto selling off"),
        ]
        added = 0
        for name, rtype, definition in REGIMES:
            node = RegimeNode(regime_name=name, regime_type=rtype, definition=definition)
            self._kg.add_node(node)
            added += 1
        return added

    def _populate_from_iae_db(
        self, since: Optional[datetime] = None
    ) -> int:
        """Parse IAE hypothesis tables and link each hypothesis into the graph."""
        if not self._iae_db.exists():
            logger.info("IAE database not found at %s — skipping", self._iae_db)
            return 0

        count = 0
        try:
            conn = sqlite3.connect(self._iae_db)
            conn.row_factory = sqlite3.Row

            # Try several common table/column names that may exist
            tables_to_try = ["hypotheses", "hypothesis", "ideas", "iae_hypotheses"]
            for table in tables_to_try:
                try:
                    if since:
                        rows = conn.execute(
                            f"SELECT * FROM {table} WHERE created_at >= ?",
                            (since.isoformat(),),
                        ).fetchall()
                    else:
                        rows = conn.execute(f"SELECT * FROM {table}").fetchall()

                    for row in rows:
                        d = dict(row)
                        self._ingest_hypothesis_row(d)
                        count += 1
                    break  # found a valid table
                except sqlite3.OperationalError:
                    continue

            conn.close()
        except Exception as exc:
            logger.warning("Could not read IAE database: %s", exc)

        return count

    def _ingest_hypothesis_row(self, row: Dict[str, Any]) -> None:
        """Convert a DB row into a graph node."""
        hyp_id = str(row.get("id", row.get("hypothesis_id", row.get("idea_id", ""))))
        text = str(row.get("hypothesis", row.get("text", row.get("description", ""))))
        param = str(row.get("parameter", row.get("param", "")))
        direction = str(row.get("direction", row.get("change_type", "")))
        confidence_raw = row.get("confidence", row.get("score", 0.5))
        try:
            confidence = float(confidence_raw)
        except (ValueError, TypeError):
            confidence = 0.5

        if not hyp_id or not text:
            return

        self._linker.link_hypothesis(
            hypothesis_id=f"hyp_{hyp_id}",
            hypothesis_text=text,
            parameter=param,
            direction=direction,
            confidence=confidence,
        )

    def _populate_from_causal_json(self) -> int:
        """
        Load causal discovery results from Julia output JSON.

        Expected format::

            {
              "edges": [
                {"from": "node_id_a", "to": "node_id_b", "strength": 0.7,
                 "lag": 2, "evidence": 150},
                ...
              ]
            }
        """
        if not self._causal_json.exists():
            logger.info("Causal JSON not found at %s — skipping", self._causal_json)
            return 0

        count = 0
        try:
            with open(self._causal_json) as f:
                data = json.load(f)
            for item in data.get("edges", []):
                src = str(item.get("from", ""))
                tgt = str(item.get("to", ""))
                strength = float(item.get("strength", 0.5))
                lag = int(item.get("lag", 0))
                evidence = int(item.get("evidence", 1))
                if not src or not tgt:
                    continue
                # Ensure nodes exist
                for nid in (src, tgt):
                    if not self._kg.get_node(nid):
                        from ..graph.node import BaseNode, NodeType
                        self._kg.add_node(BaseNode(
                            node_id=nid, node_type=NodeType.SIGNAL,
                            label=nid, confidence=0.5,
                        ))
                if lag > 0:
                    edge = Edge.leads(src, tgt, lag_bars=lag, weight=strength, evidence_count=evidence)
                else:
                    edge = Edge.causes(src, tgt, weight=strength, evidence_count=evidence)
                self._kg.add_edge(edge)
                count += 1
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Could not parse causal JSON: %s", exc)

        return count

    def _populate_from_backtest_results(self, results: List[Dict[str, Any]]) -> int:
        """
        Ingest backtest results that show which parameter changes improved metrics.

        results format::

            [{"hypothesis_id": "hyp_123", "metric": "sharpe_ratio",
              "delta": +0.15, "significant": True}, ...]
        """
        count = 0
        for result in results:
            hyp_id = result.get("hypothesis_id", "")
            metric = result.get("metric", "")
            delta = float(result.get("delta", 0.0))
            significant = bool(result.get("significant", False))

            if not hyp_id or not metric or not significant:
                continue

            hyp_node = self._kg.get_node(f"hyp_{hyp_id}") or self._kg.get_node(hyp_id)
            if not hyp_node:
                hyp_node = HypothesisNode(
                    hypothesis_id=hyp_id,
                    hypothesis_text=f"Hypothesis {hyp_id}",
                    confidence=0.5,
                )
                self._kg.add_node(hyp_node)

            metric_node_id = f"metric_{metric}"
            if not self._kg.get_node(metric_node_id):
                from ..graph.node import BaseNode, NodeType
                self._kg.add_node(BaseNode(
                    node_id=metric_node_id, node_type=NodeType.PARAMETER,
                    label=metric, confidence=0.9,
                ))

            weight = min(abs(delta) * 2, 1.0)
            self._kg.add_edge(Edge.improves(
                hyp_node.node_id, metric_node_id, weight=weight
            ))
            # Reinforce hypothesis confidence if it improved things
            hyp_node.update_confidence(min(hyp_node.confidence + 0.05, 0.95))
            count += 1

        return count
