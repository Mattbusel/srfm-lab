"""
PatternLibrary: library of known market patterns with causal explanations.

Each pattern includes:
  - Description of the pattern
  - Causal explanation (why it occurs)
  - Historical accuracy / confidence
  - Which regime it manifests in

Patterns are stored as PatternNode + causal chain edges in the graph.
Supports similarity queries to find patterns matching current market state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..graph.knowledge_graph import KnowledgeGraph
from ..graph.node import PatternNode, RegimeNode, NodeType
from ..graph.edge import Edge, EdgeType

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """A documented market pattern with full causal context."""

    pattern_id: str
    name: str
    description: str
    causal_explanation: str
    pattern_type: str          # price | on_chain | macro | liquidity
    regime: str                # regime where this manifests
    confidence: float
    lead_time_days: Optional[tuple[int, int]] = None  # (min, max) days of lead
    tags: List[str] = field(default_factory=list)
    historical_accuracy: Optional[float] = None
    source_references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "causal_explanation": self.causal_explanation,
            "pattern_type": self.pattern_type,
            "regime": self.regime,
            "confidence": self.confidence,
            "lead_time_days": self.lead_time_days,
            "tags": self.tags,
            "historical_accuracy": self.historical_accuracy,
        }


# ── Built-in pattern catalogue ────────────────────────────────────────────────

BUILTIN_PATTERNS: List[Pattern] = [
    Pattern(
        pattern_id="pat_hour1_utc_liquidity_gap",
        name="Hour 1 UTC Liquidity Gap",
        description=(
            "Win rate and P&L are significantly below average for trades opened "
            "in the 01:00-02:00 UTC window."
        ),
        causal_explanation=(
            "01:00 UTC falls in the dead zone between Asian close and European open. "
            "Market makers pull back liquidity, spreads widen, and mean-reversion "
            "signals fire on noise rather than real momentum."
        ),
        pattern_type="liquidity",
        regime="low_vol",
        confidence=0.75,
        tags=["hour_filter", "liquidity", "spread"],
        historical_accuracy=0.72,
    ),
    Pattern(
        pattern_id="pat_sopr_below_1_capitulation",
        name="SOPR < 1 Capitulation Bottom Signal",
        description=(
            "When Spent Output Profit Ratio (SOPR) drops below 1.0 on a 7-day MA "
            "basis, the market is near or at a bottom."
        ),
        causal_explanation=(
            "SOPR < 1 means on average coins are being moved at a loss. "
            "Sustained below-1 readings indicate capitulation sellers are exhausted — "
            "the marginal seller has already sold and new demand absorbs remaining "
            "supply at lower prices, leading to price recovery."
        ),
        pattern_type="on_chain",
        regime="bear",
        confidence=0.68,
        lead_time_days=(0, 14),
        tags=["sopr", "on_chain", "capitulation", "bottom"],
        historical_accuracy=0.65,
        source_references=["Glassnode SOPR analysis", "Bitcoin on-chain fundamentals"],
    ),
    Pattern(
        pattern_id="pat_hash_rate_ribbon_bull",
        name="Hash Rate Ribbon Crossover Bull Signal",
        description=(
            "When the 30-day MA of Bitcoin hash rate crosses above the 60-day MA "
            "(after a compression / death cross period), it precedes bull market "
            "initiation by 30-90 days."
        ),
        causal_explanation=(
            "Hash rate ribbon death crosses occur when miners shut off unprofitable "
            "hardware during bear markets. The subsequent recovery cross signals "
            "that miner capitulation has ended — they have either upgraded equipment "
            "or the price has risen enough to restore profitability. "
            "This miner confidence often precedes broader market recovery."
        ),
        pattern_type="on_chain",
        regime="bear",
        confidence=0.70,
        lead_time_days=(30, 90),
        tags=["hash_rate", "miner", "ribbon", "bull_signal"],
        historical_accuracy=0.67,
        source_references=["Capriole hash rate ribbon", "Charles Edwards research"],
    ),
    Pattern(
        pattern_id="pat_funding_rate_extreme_reversal",
        name="Extreme Funding Rate Mean Reversion",
        description=(
            "When perpetual swap funding rates exceed +0.10% (8h) or go below "
            "-0.05% for 3+ consecutive periods, price tends to reverse."
        ),
        causal_explanation=(
            "Extreme positive funding = everyone is long and paying longs. "
            "This imbalance becomes self-correcting: shorts are paid to exist, "
            "attracting more short sellers until the long squeeze occurs. "
            "Extreme negative funding triggers the opposite dynamic."
        ),
        pattern_type="price",
        regime="high_vol",
        confidence=0.65,
        lead_time_days=(0, 3),
        tags=["funding_rate", "perpetuals", "mean_reversion", "long_squeeze"],
        historical_accuracy=0.61,
    ),
    Pattern(
        pattern_id="pat_mvrv_z_overvalued",
        name="MVRV Z-Score Overvaluation Signal",
        description=(
            "When Bitcoin MVRV Z-score exceeds +7, it has historically marked "
            "macro cycle tops within 1-3 months."
        ),
        causal_explanation=(
            "MVRV Z-score compares market cap to realised cap normalised by "
            "standard deviation. Z > 7 means the average holder is sitting on "
            "extreme unrealised profit — the incentive to sell is at its peak. "
            "Distribution pressure from long-term holders suppresses further gains."
        ),
        pattern_type="on_chain",
        regime="bull",
        confidence=0.72,
        lead_time_days=(14, 90),
        tags=["mvrv", "overvaluation", "cycle_top", "on_chain"],
        historical_accuracy=0.70,
        source_references=["Glassnode MVRV Z-Score"],
    ),
    Pattern(
        pattern_id="pat_binance_listing_premium",
        name="Binance Listing 24h Premium",
        description=(
            "Assets listed on Binance gain +25-35% on average in the first 24 hours, "
            "followed by a partial reversion in the next 48-72h."
        ),
        causal_explanation=(
            "Binance listing dramatically increases addressable liquidity and retail "
            "access. FOMO buyers front-run the listing and buy immediately after "
            "announcement. The premium fades as initial excitement normalises and "
            "sellers who held the asset pre-listing take profits."
        ),
        pattern_type="price",
        regime="any",
        confidence=0.78,
        lead_time_days=(0, 1),
        tags=["listing", "binance", "premium", "reversion"],
        historical_accuracy=0.74,
    ),
    Pattern(
        pattern_id="pat_high_vol_entry_drag",
        name="High-Vol Entry Quality Drag",
        description=(
            "During high-volatility regimes (BVIV > 80th percentile), "
            "entry signals have lower IC and worse risk-adjusted returns."
        ),
        causal_explanation=(
            "High volatility increases slippage, makes trend signals noisy, "
            "and causes wider bid/ask spreads. Mean-reversion setups that work "
            "in ranging markets get stopped out by vol spikes. "
            "Reducing position sizes or pausing during high-vol improves Sharpe."
        ),
        pattern_type="price",
        regime="high_vol",
        confidence=0.80,
        tags=["volatility", "position_sizing", "regime_filter"],
        historical_accuracy=0.76,
    ),
]


class PatternLibrary:
    """
    Library of known market patterns with causal graph integration.

    Usage::

        lib = PatternLibrary(kg)
        lib.populate_graph()
        matches = lib.similar_patterns({"type": "on_chain", "regime": "bear"})
    """

    def __init__(self, kg: KnowledgeGraph) -> None:
        self._kg = kg
        self._patterns: Dict[str, Pattern] = {p.pattern_id: p for p in BUILTIN_PATTERNS}

    # ── public API ──────────────────────────────────────────────────────────────

    def populate_graph(self) -> int:
        """Add all built-in patterns to the knowledge graph. Returns count added."""
        added = 0
        for pattern in self._patterns.values():
            self._add_pattern_to_graph(pattern)
            added += 1
        logger.info("PatternLibrary: populated %d patterns into graph", added)
        return added

    def add_pattern(self, pattern: Pattern) -> None:
        """Add a new pattern to the library and graph."""
        self._patterns[pattern.pattern_id] = pattern
        self._add_pattern_to_graph(pattern)

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        return self._patterns.get(pattern_id)

    def all_patterns(self) -> List[Pattern]:
        return list(self._patterns.values())

    def similar_patterns(self, current_state: Dict[str, Any]) -> List[Pattern]:
        """
        Return patterns that match the current market state.

        current_state keys (all optional):
          - type: str         (price | on_chain | macro | liquidity)
          - regime: str       (high_vol | bear | bull | ranging …)
          - tags: List[str]   (free-form tags to match)
          - min_confidence: float
        """
        results: List[Pattern] = []
        ptype = current_state.get("type", "")
        regime = current_state.get("regime", "").lower()
        tags = set(current_state.get("tags", []))
        min_conf = float(current_state.get("min_confidence", 0.0))

        for p in self._patterns.values():
            if p.confidence < min_conf:
                continue
            score = 0.0
            if ptype and p.pattern_type == ptype:
                score += 2.0
            if regime and (p.regime == regime or p.regime == "any"):
                score += 2.0
            if tags:
                overlap = tags & set(p.tags)
                score += len(overlap)
            if score > 0:
                # Attach score as temporary attribute for sorting
                p_copy = Pattern(**{k: getattr(p, k) for k in p.__dataclass_fields__})  # type: ignore[attr-defined]
                results.append((score, p_copy))

        results.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in results]

    def patterns_for_regime(self, regime: str) -> List[Pattern]:
        return [p for p in self._patterns.values()
                if p.regime == regime or p.regime == "any"]

    def patterns_by_type(self, pattern_type: str) -> List[Pattern]:
        return [p for p in self._patterns.values() if p.pattern_type == pattern_type]

    # ── graph integration ─────────────────────────────────────────────────────────

    def _add_pattern_to_graph(self, pattern: Pattern) -> None:
        # Create pattern node
        pat_node = PatternNode(
            pattern_name=pattern.pattern_id,
            pattern_type=pattern.pattern_type,
            description=pattern.description,
            causal_explanation=pattern.causal_explanation,
            confidence=pattern.confidence,
        )
        pat_node.set_property("historical_accuracy", pattern.historical_accuracy)
        pat_node.set_property("tags", pattern.tags)
        pat_node.set_property("lead_time_days", pattern.lead_time_days)
        self._kg.add_node(pat_node)

        # Create or connect to regime node
        if pattern.regime and pattern.regime != "any":
            regime_id = f"regime_{pattern.regime.lower()}"
            if not self._kg.get_node(regime_id):
                self._kg.add_node(RegimeNode(regime_name=pattern.regime))
            self._kg.add_edge(Edge.occurs_during(
                pat_node.node_id, regime_id,
                weight=pattern.confidence,
            ))
