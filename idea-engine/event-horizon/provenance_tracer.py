"""
Provenance Tracer: trace any trading decision back through the full chain.

For any trade, reconstructs the complete decision DAG:
  Trade -> Signal -> Debate -> Hypothesis -> Physics Concept -> EHS -> RMEA

Generates human-readable explanations with confidence intervals at each step.

This is the XAI (Explainable AI) layer for the autonomous trading system.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


# ---------------------------------------------------------------------------
# Trace Node: one step in the decision chain
# ---------------------------------------------------------------------------

@dataclass
class TraceNode:
    """One node in the decision provenance DAG."""
    node_id: str
    layer: str                    # rmea / ehs / debate / signal / execution / trade
    component: str                # specific module name
    action: str                   # what happened at this node
    input_summary: str            # what went in
    output_summary: str           # what came out
    confidence: float             # 0-1 confidence in this step
    timestamp: float = 0.0
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProvenanceTrace:
    """Complete provenance trace for one trading decision."""
    trace_id: str
    trade_symbol: str
    trade_direction: str          # "long" / "short"
    trade_size: float
    trade_timestamp: float
    nodes: List[TraceNode] = field(default_factory=list)
    explanation: str = ""
    overall_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Explanation Templates
# ---------------------------------------------------------------------------

def _confidence_word(conf: float) -> str:
    if conf > 0.8:
        return "high confidence"
    elif conf > 0.6:
        return "moderate confidence"
    elif conf > 0.4:
        return "low confidence"
    else:
        return "speculative"


def _format_pct(val: float) -> str:
    return f"{val:+.1%}" if abs(val) < 1 else f"{val:+.1f}x"


# ---------------------------------------------------------------------------
# Provenance Tracer
# ---------------------------------------------------------------------------

class ProvenanceTracer:
    """
    Traces trading decisions back through the full autonomous pipeline.

    Usage:
      tracer = ProvenanceTracer()
      trace = tracer.create_trace(trade_info)
      trace = tracer.add_signal_node(trace, signal_info)
      trace = tracer.add_debate_node(trace, debate_info)
      trace = tracer.add_hypothesis_node(trace, hypothesis_info)
      trace = tracer.add_physics_node(trace, physics_info)
      explanation = tracer.generate_explanation(trace)
    """

    def __init__(self):
        self._counter = 0
        self._traces: Dict[str, ProvenanceTrace] = {}

    def _next_id(self, prefix: str = "node") -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:06d}"

    def create_trace(
        self,
        symbol: str,
        direction: str,
        size: float,
        price: float,
        timestamp: float = 0.0,
    ) -> ProvenanceTrace:
        """Create a new provenance trace for a trade."""
        trace_id = self._next_id("trace")

        trade_node = TraceNode(
            node_id=self._next_id("trade"),
            layer="trade",
            component="execution_engine",
            action=f"{'Bought' if direction == 'long' else 'Sold'} {size:.4f} {symbol} at {price:.2f}",
            input_summary=f"Signal: {direction}, Size: {size}",
            output_summary=f"Order submitted to broker",
            confidence=1.0,
            timestamp=timestamp or time.time(),
        )

        trace = ProvenanceTrace(
            trace_id=trace_id,
            trade_symbol=symbol,
            trade_direction=direction,
            trade_size=size,
            trade_timestamp=timestamp or time.time(),
            nodes=[trade_node],
        )

        self._traces[trace_id] = trace
        return trace

    def add_signal_node(
        self,
        trace: ProvenanceTrace,
        signal_name: str,
        signal_value: float,
        signal_confidence: float,
        signal_source: str,
        contributing_features: List[str] = None,
    ) -> ProvenanceTrace:
        """Add a signal generation node to the trace."""
        node = TraceNode(
            node_id=self._next_id("signal"),
            layer="signal",
            component=signal_source,
            action=f"Generated {signal_name} signal: {signal_value:+.3f}",
            input_summary=f"Features: {', '.join(contributing_features[:5]) if contributing_features else 'market data'}",
            output_summary=f"Signal value {signal_value:+.3f} ({_confidence_word(signal_confidence)})",
            confidence=signal_confidence,
            parent_ids=[trace.nodes[-1].node_id] if trace.nodes else [],
            timestamp=time.time(),
        )
        trace.nodes.append(node)
        return trace

    def add_debate_node(
        self,
        trace: ProvenanceTrace,
        consensus_score: float,
        n_supporters: int,
        n_opponents: int,
        key_arguments: List[str] = None,
        risk_veto: bool = False,
    ) -> ProvenanceTrace:
        """Add a debate evaluation node."""
        total = n_supporters + n_opponents
        node = TraceNode(
            node_id=self._next_id("debate"),
            layer="debate",
            component="debate_tournament",
            action=f"Debate: {n_supporters}/{total} agents support ({consensus_score:.0%} consensus)",
            input_summary=f"Hypothesis evaluated by {total} agents",
            output_summary=f"{'APPROVED' if not risk_veto else 'VETOED'}: {', '.join(key_arguments[:3]) if key_arguments else 'no arguments recorded'}",
            confidence=consensus_score,
            parent_ids=[trace.nodes[-1].node_id] if trace.nodes else [],
            timestamp=time.time(),
            metadata={"risk_veto": risk_veto},
        )
        trace.nodes.append(node)
        return trace

    def add_hypothesis_node(
        self,
        trace: ProvenanceTrace,
        hypothesis_name: str,
        template_type: str,
        regime: str,
        backtest_sharpe: float,
        validation_score: float,
    ) -> ProvenanceTrace:
        """Add a hypothesis origin node."""
        node = TraceNode(
            node_id=self._next_id("hyp"),
            layer="hypothesis",
            component="hypothesis_validator",
            action=f"Hypothesis: {hypothesis_name} ({template_type})",
            input_summary=f"Template: {template_type}, Regime: {regime}",
            output_summary=f"Backtest Sharpe: {backtest_sharpe:.2f}, Validation: {validation_score:.0f}/100",
            confidence=min(1.0, validation_score / 100),
            parent_ids=[trace.nodes[-1].node_id] if trace.nodes else [],
            timestamp=time.time(),
        )
        trace.nodes.append(node)
        return trace

    def add_physics_node(
        self,
        trace: ProvenanceTrace,
        physics_concept: str,
        domain: str,
        primitive_id: str,
        ehs_generation: int,
    ) -> ProvenanceTrace:
        """Add a physics concept origin node (from the EHS)."""
        node = TraceNode(
            node_id=self._next_id("physics"),
            layer="ehs",
            component="event_horizon_synthesizer",
            action=f"Physics: {physics_concept} ({domain})",
            input_summary=f"Concept from EHS generation {ehs_generation}",
            output_summary=f"Primitive {primitive_id} synthesized and validated",
            confidence=0.7,
            parent_ids=[trace.nodes[-1].node_id] if trace.nodes else [],
            timestamp=time.time(),
        )
        trace.nodes.append(node)
        return trace

    def add_consciousness_node(
        self,
        trace: ProvenanceTrace,
        emergent_belief: str,
        collective_activation: float,
        agreement_level: float,
        surprising_agreements: List[str] = None,
    ) -> ProvenanceTrace:
        """Add a market consciousness node."""
        node = TraceNode(
            node_id=self._next_id("consciousness"),
            layer="consciousness",
            component="market_consciousness",
            action=f"Emergent belief: {emergent_belief}",
            input_summary=f"Collective activation: {collective_activation:+.2f}, Agreement: {agreement_level:.0%}",
            output_summary=f"Surprising agreements: {', '.join(surprising_agreements[:3]) if surprising_agreements else 'none'}",
            confidence=agreement_level,
            parent_ids=[trace.nodes[-1].node_id] if trace.nodes else [],
            timestamp=time.time(),
        )
        trace.nodes.append(node)
        return trace

    def add_rmea_node(
        self,
        trace: ProvenanceTrace,
        hyper_genome_id: str,
        meta_fitness: float,
        adversarial_intensity: float,
        serendipity_event: Optional[str] = None,
    ) -> ProvenanceTrace:
        """Add an RMEA meta-evolution node."""
        node = TraceNode(
            node_id=self._next_id("rmea"),
            layer="rmea",
            component="recursive_meta_evolver",
            action=f"RMEA: genome {hyper_genome_id} (fitness {meta_fitness:+.3f})",
            input_summary=f"Adversarial intensity: {adversarial_intensity:.2f}",
            output_summary=f"Serendipity: {serendipity_event}" if serendipity_event else "No serendipity event",
            confidence=min(1.0, max(0, meta_fitness)),
            parent_ids=[trace.nodes[-1].node_id] if trace.nodes else [],
            timestamp=time.time(),
        )
        trace.nodes.append(node)
        return trace

    def generate_explanation(self, trace: ProvenanceTrace) -> str:
        """
        Generate a human-readable natural language explanation of the
        complete decision chain.
        """
        lines = []
        lines.append(f"## Trade Explanation: {trace.trade_direction.upper()} {trace.trade_symbol}")
        lines.append(f"*Trace ID: {trace.trace_id}*\n")

        # Overall confidence: geometric mean of all node confidences
        confidences = [n.confidence for n in trace.nodes if n.confidence > 0]
        if confidences:
            overall = math.exp(sum(math.log(c) for c in confidences) / len(confidences))
        else:
            overall = 0.0
        trace.overall_confidence = overall

        lines.append(f"**Overall confidence: {overall:.0%}** ({_confidence_word(overall)})\n")

        # Walk the chain in reverse (from deepest origin to trade)
        for node in reversed(trace.nodes):
            layer_emoji = {
                "rmea": "**[META]**",
                "ehs": "**[PHYSICS]**",
                "consciousness": "**[CONSCIOUSNESS]**",
                "hypothesis": "**[HYPOTHESIS]**",
                "debate": "**[DEBATE]**",
                "signal": "**[SIGNAL]**",
                "trade": "**[TRADE]**",
            }.get(node.layer, f"[{node.layer.upper()}]")

            lines.append(f"{layer_emoji} {node.action}")
            lines.append(f"  Input: {node.input_summary}")
            lines.append(f"  Output: {node.output_summary}")
            lines.append(f"  Confidence: {node.confidence:.0%}\n")

        # Summary
        lines.append("---")
        lines.append(f"**Decision chain depth:** {len(trace.nodes)} layers")
        lines.append(f"**Weakest link:** {min(trace.nodes, key=lambda n: n.confidence).component} "
                     f"({min(n.confidence for n in trace.nodes):.0%})")
        lines.append(f"**Strongest link:** {max(trace.nodes, key=lambda n: n.confidence).component} "
                     f"({max(n.confidence for n in trace.nodes):.0%})")

        explanation = "\n".join(lines)
        trace.explanation = explanation
        return explanation

    def get_trace(self, trace_id: str) -> Optional[ProvenanceTrace]:
        return self._traces.get(trace_id)

    def get_all_traces(self) -> List[ProvenanceTrace]:
        return list(self._traces.values())
