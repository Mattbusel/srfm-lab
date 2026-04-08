"""
Idea synthesizer — combines multiple hypothesis components into coherent trade ideas.

Takes signals from different domains (physics, ML, microstructure, macro)
and synthesizes them into a unified trade thesis with narrative.

Features:
  - Multi-domain signal fusion
  - Narrative generation: translates signals into human-readable thesis
  - Conflict resolution: handles conflicting signals from different domains
  - Confidence calibration across domains
  - Hypothesis template library expansion
  - Cross-domain insight: physics insight applied to micro-structure, etc.
  - Synthesis quality scoring
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Signal Domain ─────────────────────────────────────────────────────────────

@dataclass
class DomainSignal:
    """Signal from a specific analytical domain."""
    domain: str          # physics/ml/microstructure/macro/technical/alternative
    signal_name: str
    value: float         # -1 to +1
    confidence: float    # 0 to 1
    regime: str
    key_observations: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)


# ── Synthesized Idea ──────────────────────────────────────────────────────────

@dataclass
class SynthesizedIdea:
    """A complete synthesized trade idea."""
    id: str
    name: str
    direction: float          # +1 long, -1 short
    conviction: float         # 0-1
    composite_signal: float
    synthesis_confidence: float

    # Domain breakdown
    domain_signals: list[DomainSignal]
    dominant_domain: str
    domain_alignment: float   # how well domains agree

    # Trade narrative
    thesis: str
    key_factors: list[str]
    risk_factors: list[str]
    invalidation_conditions: list[str]

    # Metadata
    template_type: str
    regime: str
    synthesis_quality: float  # 0-1 (how complete/consistent)


# ── Conflict Detector ─────────────────────────────────────────────────────────

class ConflictDetector:
    """Detects and resolves conflicts between domain signals."""

    def detect(self, signals: list[DomainSignal]) -> dict:
        if len(signals) < 2:
            return {"has_conflict": False, "conflict_pairs": []}

        conflict_pairs = []
        for i in range(len(signals)):
            for j in range(i + 1, len(signals)):
                s1, s2 = signals[i], signals[j]
                # Conflict: significant signals pointing in opposite directions
                if (abs(s1.value) > 0.2 and abs(s2.value) > 0.2
                        and s1.value * s2.value < 0
                        and abs(s1.value - s2.value) > 0.4):
                    conflict_pairs.append((s1.domain, s2.domain, s1.value, s2.value))

        return {
            "has_conflict": len(conflict_pairs) > 0,
            "conflict_pairs": conflict_pairs,
            "n_conflicts": len(conflict_pairs),
        }

    def resolve(
        self,
        signals: list[DomainSignal],
        resolution: str = "confidence_weighted",
    ) -> float:
        """
        Resolve conflicting signals into a single direction.
        resolution: 'confidence_weighted', 'dominant', 'conservative'
        """
        if not signals:
            return 0.0

        if resolution == "confidence_weighted":
            total_conf = sum(abs(s.value) * s.confidence for s in signals)
            if total_conf < 1e-10:
                return 0.0
            return float(sum(s.value * s.confidence for s in signals) / total_conf)

        elif resolution == "dominant":
            # Use signal from highest confidence domain
            best = max(signals, key=lambda s: s.confidence)
            return float(best.value)

        elif resolution == "conservative":
            # Only trade if majority align
            positive = sum(1 for s in signals if s.value > 0.15)
            negative = sum(1 for s in signals if s.value < -0.15)
            total = len(signals)
            if positive / max(total, 1) > 0.6:
                return float(np.mean([s.value for s in signals if s.value > 0]))
            elif negative / max(total, 1) > 0.6:
                return float(np.mean([s.value for s in signals if s.value < 0]))
            return 0.0

        return 0.0


# ── Narrative Generator ───────────────────────────────────────────────────────

class NarrativeGenerator:
    """Generates human-readable trade narratives from signals."""

    DOMAIN_DESCRIPTIONS = {
        "physics": "physics-inspired market dynamics",
        "ml": "machine learning pattern recognition",
        "microstructure": "order flow and market microstructure",
        "macro": "macroeconomic regime analysis",
        "technical": "technical price analysis",
        "alternative": "alternative data signals",
        "regime": "regime detection models",
    }

    SIGNAL_DESCRIPTIONS = {
        "trend": ("upward trending", "downward trending"),
        "momentum": ("strong positive momentum", "strong negative momentum"),
        "mean_reversion": ("oversold reversal potential", "overbought reversal potential"),
        "volatility": ("elevated volatility environment", "low volatility environment"),
        "entropy": ("low entropy ordered market", "high entropy chaotic market"),
        "bh_gravitational": ("high gravitational pull upward", "gravitational pull downward"),
    }

    def generate_thesis(
        self,
        direction: float,
        dominant_domain: str,
        key_factors: list[str],
        regime: str,
        conviction: float,
    ) -> str:
        """Generate a trade thesis string."""
        dir_str = "long" if direction > 0 else "short"
        domain_desc = self.DOMAIN_DESCRIPTIONS.get(dominant_domain, dominant_domain)
        conviction_str = ("high" if conviction > 0.7 else "moderate" if conviction > 0.4 else "low")

        thesis_parts = [
            f"We see a {conviction_str}-conviction {dir_str} opportunity",
            f"driven by {domain_desc}",
            f"in a {regime} market regime.",
        ]

        if key_factors:
            thesis_parts.append(f"Key supporting factors: {', '.join(key_factors[:3])}.")

        return " ".join(thesis_parts)

    def generate_factors(
        self,
        signals: list[DomainSignal],
        direction: float,
    ) -> tuple[list[str], list[str]]:
        """Generate key factors and risk factors."""
        key_factors = []
        risk_factors = []

        for sig in signals:
            sign_match = float(np.sign(sig.value)) == float(np.sign(direction))

            obs = sig.key_observations[:2] if sig.key_observations else [
                f"{sig.domain} signal at {sig.value:.2f}"
            ]

            if sign_match and abs(sig.value) > 0.2:
                key_factors.extend(obs)
            elif not sign_match and abs(sig.value) > 0.2:
                risk_factors.extend([f"[{sig.domain}] {o}" for o in obs])

        # Add regime risk
        risk_factors.append("Regime change could invalidate thesis")

        return key_factors[:5], risk_factors[:5]

    def invalidation_conditions(
        self,
        direction: float,
        template_type: str,
    ) -> list[str]:
        """Generate invalidation conditions based on template."""
        conditions = []
        if direction > 0:
            conditions.append("Price breaks below key support level")
            conditions.append("Volume divergence: price rising but volume falling")
        else:
            conditions.append("Price breaks above key resistance level")
            conditions.append("Short squeeze: forced covering drives price up")

        if template_type in ("mean_reversion", "pairs_trade"):
            conditions.append("Spread widens beyond 3-sigma instead of reverting")
        elif template_type in ("momentum", "trend_following"):
            conditions.append("Momentum reversal: price fails to make new high/low")
        elif template_type in ("volatility_breakout",):
            conditions.append("False breakout: volatility expansion reverses quickly")

        conditions.append("Macro regime shift invalidates signal assumptions")
        return conditions[:4]


# ── Domain Weights ────────────────────────────────────────────────────────────

DOMAIN_WEIGHTS_BY_REGIME = {
    "trending_bull": {
        "technical": 0.30, "ml": 0.25, "macro": 0.20,
        "physics": 0.10, "microstructure": 0.10, "alternative": 0.05,
    },
    "trending_bear": {
        "technical": 0.25, "ml": 0.20, "macro": 0.25,
        "physics": 0.10, "microstructure": 0.15, "alternative": 0.05,
    },
    "mean_reverting": {
        "technical": 0.20, "ml": 0.20, "microstructure": 0.25,
        "physics": 0.15, "macro": 0.10, "alternative": 0.10,
    },
    "high_volatility": {
        "microstructure": 0.30, "technical": 0.20, "ml": 0.15,
        "physics": 0.20, "macro": 0.10, "alternative": 0.05,
    },
    "crisis": {
        "macro": 0.35, "microstructure": 0.25, "ml": 0.15,
        "physics": 0.15, "technical": 0.05, "alternative": 0.05,
    },
    "unknown": {
        "technical": 0.20, "ml": 0.20, "macro": 0.20,
        "microstructure": 0.20, "physics": 0.10, "alternative": 0.10,
    },
}


# ── Idea Synthesizer ──────────────────────────────────────────────────────────

class IdeaSynthesizer:
    """
    Synthesizes signals from multiple domains into coherent trade ideas.
    """

    def __init__(self):
        self.conflict_detector = ConflictDetector()
        self.narrative_gen = NarrativeGenerator()
        self._id_counter = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"idea_{self._id_counter:05d}"

    def synthesize(
        self,
        domain_signals: list[DomainSignal],
        regime: str = "unknown",
        template_type: str = "composite",
    ) -> SynthesizedIdea:
        """Synthesize multiple domain signals into a trade idea."""
        if not domain_signals:
            return self._empty_idea(regime)

        # Get regime-specific domain weights
        weights = DOMAIN_WEIGHTS_BY_REGIME.get(regime, DOMAIN_WEIGHTS_BY_REGIME["unknown"])

        # Weighted composite signal
        total_weight = 0.0
        weighted_signal = 0.0
        for sig in domain_signals:
            w = weights.get(sig.domain, 0.1) * sig.confidence
            weighted_signal += sig.value * w
            total_weight += w

        composite = float(weighted_signal / max(total_weight, 1e-10))

        # Detect conflicts
        conflicts = self.conflict_detector.detect(domain_signals)
        if conflicts["has_conflict"]:
            # Use conflict resolution
            resolved = self.conflict_detector.resolve(domain_signals, "confidence_weighted")
            composite = 0.7 * composite + 0.3 * resolved

        # Direction and conviction
        direction = float(np.sign(composite)) if abs(composite) > 0.1 else 0.0

        # Domain alignment score
        signs = [np.sign(s.value) for s in domain_signals if abs(s.value) > 0.1]
        if signs:
            agreement = sum(1 for s in signs if s == np.sign(composite)) / len(signs)
        else:
            agreement = 0.5
        domain_alignment = float(agreement)

        # Dominant domain
        best_sig = max(domain_signals, key=lambda s: abs(s.value) * s.confidence)
        dominant_domain = best_sig.domain

        # Conviction
        conviction = float(abs(composite) * domain_alignment * (1 - 0.2 * conflicts["n_conflicts"]))
        conviction = float(np.clip(conviction, 0, 1))

        # Synthesis quality
        coverage = len(domain_signals) / 6  # 6 domains total
        quality = float(min(conviction * coverage * domain_alignment, 1))

        # Narrative
        key_factors, risk_factors = self.narrative_gen.generate_factors(domain_signals, direction)
        thesis = self.narrative_gen.generate_thesis(
            direction, dominant_domain, key_factors, regime, conviction
        )
        invalidation = self.narrative_gen.invalidation_conditions(direction, template_type)

        # Name
        dir_word = "Long" if direction > 0 else "Short" if direction < 0 else "Neutral"
        name = f"{dir_word} {template_type.replace('_', ' ').title()} [{regime}]"

        return SynthesizedIdea(
            id=self._next_id(),
            name=name,
            direction=direction,
            conviction=conviction,
            composite_signal=composite,
            synthesis_confidence=quality,
            domain_signals=domain_signals,
            dominant_domain=dominant_domain,
            domain_alignment=domain_alignment,
            thesis=thesis,
            key_factors=key_factors,
            risk_factors=risk_factors,
            invalidation_conditions=invalidation,
            template_type=template_type,
            regime=regime,
            synthesis_quality=quality,
        )

    def _empty_idea(self, regime: str) -> SynthesizedIdea:
        return SynthesizedIdea(
            id=self._next_id(),
            name="No Signal",
            direction=0.0,
            conviction=0.0,
            composite_signal=0.0,
            synthesis_confidence=0.0,
            domain_signals=[],
            dominant_domain="none",
            domain_alignment=0.5,
            thesis="No signal detected.",
            key_factors=[],
            risk_factors=[],
            invalidation_conditions=[],
            template_type="none",
            regime=regime,
            synthesis_quality=0.0,
        )

    def batch_synthesize(
        self,
        all_domain_signals: list[list[DomainSignal]],
        regime: str = "unknown",
        min_conviction: float = 0.25,
    ) -> list[SynthesizedIdea]:
        """Synthesize multiple groups of signals, return sorted by conviction."""
        ideas = [
            self.synthesize(sigs, regime)
            for sigs in all_domain_signals
        ]
        ideas = [i for i in ideas if i.conviction >= min_conviction and i.direction != 0]
        ideas.sort(key=lambda i: i.conviction, reverse=True)
        return ideas

    def synthesis_report(self, ideas: list[SynthesizedIdea]) -> dict:
        """Summary of synthesis results."""
        if not ideas:
            return {"n_ideas": 0}
        n_long = sum(1 for i in ideas if i.direction > 0)
        n_short = sum(1 for i in ideas if i.direction < 0)
        avg_conviction = float(np.mean([i.conviction for i in ideas]))
        top_domains = {}
        for idea in ideas:
            top_domains[idea.dominant_domain] = top_domains.get(idea.dominant_domain, 0) + 1

        return {
            "n_ideas": len(ideas),
            "n_long": n_long,
            "n_short": n_short,
            "avg_conviction": avg_conviction,
            "top_idea": ideas[0].name if ideas else None,
            "top_conviction": float(ideas[0].conviction) if ideas else 0.0,
            "dominant_domains": top_domains,
        }
