"""
Debate orchestrator — runs multi-agent debate on hypothesis candidates.

Coordinates all debate agents:
  - Quant Researcher (statistical validity)
  - Macro Analyst (macro context)
  - Risk Manager (risk assessment)
  - Regime Expert (regime alignment)
  - Devil's Advocate (challenge assumptions)
  - Microstructure Specialist (execution feasibility)

Applies Dempster-Shafer evidence combination for final verdict.
"""

from __future__ import annotations
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Agent Opinion ─────────────────────────────────────────────────────────────

@dataclass
class AgentOpinion:
    agent_name: str
    hypothesis_id: str
    support_score: float        # 0-1 (how much agent supports hypothesis)
    reject_score: float         # 0-1 (how much agent rejects)
    uncertainty: float          # 0-1 (how uncertain is agent)
    reasoning: list[str]        # key reasons
    warnings: list[str]
    weight: float = 1.0         # agent weight in final aggregation


@dataclass
class DebateVerdict:
    hypothesis_id: str
    combined_support: float     # Dempster-Shafer combined belief
    combined_reject: float
    uncertainty_mass: float
    consensus_score: float      # simplified final score 0-1
    direction: float            # +1 long, -1 short, 0 neutral
    agent_opinions: list[AgentOpinion]
    debate_warnings: list[str]
    should_proceed: bool
    confidence: float


# ── Mock Agent Implementations ────────────────────────────────────────────────
# These call into the real agent modules if available, else use simple heuristics

class QuantResearcherOpinion:
    """Evaluates statistical rigor of the hypothesis."""
    name = "quant_researcher"
    weight = 1.0

    def evaluate(
        self,
        hypothesis: dict,
        returns: np.ndarray,
        context: dict,
    ) -> AgentOpinion:
        reasoning = []
        warnings = []
        support = 0.5
        reject = 0.2
        uncertainty = 0.3

        signal = float(hypothesis.get("entry_signal", 0.0))
        direction = float(hypothesis.get("direction", 0.0))

        # Check signal strength
        if abs(signal) > 0.5:
            support += 0.2
            reasoning.append(f"Strong signal strength: {signal:.2f}")
        elif abs(signal) < 0.15:
            reject += 0.2
            warnings.append("Weak signal — low statistical confidence")

        # Check if signal aligns with recent returns
        if len(returns) >= 20:
            recent_mean = float(returns[-20:].mean())
            if math.copysign(1, recent_mean) == direction and abs(recent_mean) > 0:
                support += 0.15
                reasoning.append("Signal direction matches recent momentum")
            else:
                reject += 0.1
                warnings.append("Signal direction contradicts recent returns")

        # Sharpe sanity check
        if len(returns) >= 63:
            r = returns[-63:]
            sharpe = float(r.mean() / (r.std() + 1e-10) * math.sqrt(252))
            if abs(sharpe) > 1.0 and math.copysign(1, sharpe) == direction:
                support += 0.1
                reasoning.append(f"Favorable Sharpe: {sharpe:.2f}")

        support = float(min(support, 0.95))
        reject = float(min(reject, 0.95))
        uncertainty = float(max(1 - support - reject, 0.05))

        return AgentOpinion(
            agent_name=self.name,
            hypothesis_id=str(hypothesis.get("id", "")),
            support_score=support,
            reject_score=reject,
            uncertainty=uncertainty,
            reasoning=reasoning,
            warnings=warnings,
            weight=self.weight,
        )


class MacroAnalystOpinion:
    """Evaluates macro context."""
    name = "macro_analyst"
    weight = 0.8

    def evaluate(
        self,
        hypothesis: dict,
        returns: np.ndarray,
        context: dict,
    ) -> AgentOpinion:
        reasoning = []
        warnings = []
        regime = context.get("regime", "unknown")
        direction = float(hypothesis.get("direction", 0.0))

        support = 0.4
        reject = 0.2
        uncertainty = 0.4

        # Regime alignment
        if regime in ("trending_bull",) and direction > 0:
            support += 0.25
            reasoning.append("Macro regime supports long direction")
        elif regime in ("trending_bear",) and direction < 0:
            support += 0.25
            reasoning.append("Macro regime supports short direction")
        elif regime == "high_volatility":
            reject += 0.15
            warnings.append("High volatility regime increases macro risk")
            uncertainty += 0.1
        elif regime == "mean_reverting":
            if hypothesis.get("template_type", "") in ("mean_reversion", "pairs_trade"):
                support += 0.2
                reasoning.append("Regime supports mean-reversion thesis")
            else:
                reject += 0.1
                warnings.append("Regime is mean-reverting but hypothesis is directional")

        support = float(min(support, 0.9))
        reject = float(min(reject, 0.9))
        uncertainty = float(max(1 - support - reject, 0.05))

        return AgentOpinion(
            agent_name=self.name,
            hypothesis_id=str(hypothesis.get("id", "")),
            support_score=support,
            reject_score=reject,
            uncertainty=uncertainty,
            reasoning=reasoning,
            warnings=warnings,
            weight=self.weight,
        )


class RiskManagerOpinion:
    """Evaluates risk and sizing."""
    name = "risk_manager"
    weight = 1.2  # risk manager gets extra weight

    def evaluate(
        self,
        hypothesis: dict,
        returns: np.ndarray,
        context: dict,
    ) -> AgentOpinion:
        reasoning = []
        warnings = []

        support = 0.5
        reject = 0.15
        uncertainty = 0.35

        if len(returns) >= 21:
            vol_ann = float(returns[-21:].std() * math.sqrt(252))

            if vol_ann > 0.8:
                reject += 0.25
                warnings.append(f"Very high volatility: {vol_ann:.1%} annualized")
            elif vol_ann > 0.4:
                reject += 0.10
                warnings.append(f"Elevated volatility: {vol_ann:.1%}")
            else:
                support += 0.10
                reasoning.append(f"Manageable volatility: {vol_ann:.1%}")

        # Check drawdown environment
        if len(returns) >= 63:
            from_peak = float(np.min(np.cumsum(returns[-63:])))
            if from_peak < -0.15:
                reject += 0.15
                warnings.append(f"Asset in drawdown: {from_peak:.1%}")

        liquidity = float(context.get("liquidity_score", 0.5))
        if liquidity < 0.3:
            reject += 0.1
            warnings.append("Low liquidity score — execution risk")
        elif liquidity > 0.7:
            support += 0.05
            reasoning.append("Good liquidity conditions")

        support = float(min(support, 0.9))
        reject = float(min(reject, 0.9))
        uncertainty = float(max(1 - support - reject, 0.05))

        return AgentOpinion(
            agent_name=self.name,
            hypothesis_id=str(hypothesis.get("id", "")),
            support_score=support,
            reject_score=reject,
            uncertainty=uncertainty,
            reasoning=reasoning,
            warnings=warnings,
            weight=self.weight,
        )


class DevilsAdvocateOpinion:
    """Always challenges the thesis."""
    name = "devils_advocate"
    weight = 0.7

    def evaluate(
        self,
        hypothesis: dict,
        returns: np.ndarray,
        context: dict,
    ) -> AgentOpinion:
        reasoning = []
        warnings = []

        # Devil's advocate starts skeptical
        support = 0.2
        reject = 0.4
        uncertainty = 0.4

        template = hypothesis.get("template_type", "")
        complex_templates = {"regime_adaptive", "chameleon", "structural_break", "crisis_alpha"}

        if template in complex_templates:
            reject += 0.15
            warnings.append(f"Template '{template}' is complex — curve-fitting risk")

        # Challenge: is this just looking at noise?
        if len(returns) >= 30:
            t_stat = float(
                returns[-30:].mean() / (returns[-30:].std() / math.sqrt(30) + 1e-10)
            )
            if abs(t_stat) < 1.65:
                reject += 0.10
                warnings.append(f"Signal not statistically significant (t={t_stat:.2f})")
            else:
                support += 0.05

        # Counter: what if regime flips?
        regime = context.get("regime", "unknown")
        if regime not in ("trending_bull", "trending_bear"):
            warnings.append("Regime unclear — hypothesis validity contingent on regime stability")
            uncertainty += 0.1

        support = float(min(support, 0.8))
        reject = float(min(reject, 0.9))
        uncertainty = float(max(1 - support - reject, 0.05))

        return AgentOpinion(
            agent_name=self.name,
            hypothesis_id=str(hypothesis.get("id", "")),
            support_score=support,
            reject_score=reject,
            uncertainty=uncertainty,
            reasoning=reasoning,
            warnings=warnings,
            weight=self.weight,
        )


# ── Dempster-Shafer Combination ───────────────────────────────────────────────

def dempster_shafer_combine(opinions: list[AgentOpinion]) -> tuple[float, float, float]:
    """
    Combine agent opinions using Dempster's rule of combination.
    Each agent has a Basic Probability Assignment over {support, reject, uncertain}.
    Returns (combined_support, combined_reject, combined_uncertainty).
    """
    if not opinions:
        return 0.5, 0.3, 0.2

    # Initialize with first opinion (weighted)
    total_weight = sum(o.weight for o in opinions)

    # Weighted combination
    # bpa[0] = support mass, bpa[1] = reject mass, bpa[2] = uncertainty mass
    bpa = np.array([
        sum(o.support_score * o.weight / total_weight for o in opinions),
        sum(o.reject_score * o.weight / total_weight for o in opinions),
        sum(o.uncertainty * o.weight / total_weight for o in opinions),
    ])

    # Normalize
    bpa = np.maximum(bpa, 0)
    bpa /= bpa.sum() + 1e-10

    return float(bpa[0]), float(bpa[1]), float(bpa[2])


# ── Orchestrator ──────────────────────────────────────────────────────────────

class DebateOrchestrator:
    """
    Runs multi-agent debate on hypothesis candidates.
    Applies Dempster-Shafer for final verdict.
    """

    def __init__(self, min_support_to_proceed: float = 0.4):
        self.agents = [
            QuantResearcherOpinion(),
            MacroAnalystOpinion(),
            RiskManagerOpinion(),
            DevilsAdvocateOpinion(),
        ]
        self.min_support = min_support_to_proceed

    def debate_hypothesis(
        self,
        hypothesis: dict,
        returns: np.ndarray,
        context: Optional[dict] = None,
    ) -> DebateVerdict:
        """Run full debate on a single hypothesis."""
        if context is None:
            context = {}

        opinions = []
        all_warnings = []

        for agent in self.agents:
            opinion = agent.evaluate(hypothesis, returns, context)
            opinions.append(opinion)
            all_warnings.extend([f"[{agent.name}] {w}" for w in opinion.warnings])

        # Dempster-Shafer combination
        combined_support, combined_reject, uncertainty_mass = dempster_shafer_combine(opinions)

        # Consensus score
        consensus = float(combined_support - 0.5 * combined_reject)
        consensus = float(np.clip(consensus, 0, 1))

        # Direction
        direction = float(hypothesis.get("direction", 0.0))

        # Confidence: high support, low rejection, low uncertainty
        confidence = float(
            combined_support * (1 - combined_reject) * (1 - 0.5 * uncertainty_mass)
        )

        should_proceed = (
            combined_support >= self.min_support
            and combined_support > combined_reject
            and direction != 0
        )

        return DebateVerdict(
            hypothesis_id=str(hypothesis.get("id", "")),
            combined_support=combined_support,
            combined_reject=combined_reject,
            uncertainty_mass=uncertainty_mass,
            consensus_score=consensus,
            direction=direction,
            agent_opinions=opinions,
            debate_warnings=all_warnings,
            should_proceed=should_proceed,
            confidence=confidence,
        )

    def batch_debate(
        self,
        hypotheses: list[dict],
        returns: np.ndarray,
        context: Optional[dict] = None,
        top_k: int = 5,
    ) -> list[DebateVerdict]:
        """Run debate on all hypothesis candidates, return top-k verdicts."""
        verdicts = []
        for hyp in hypotheses:
            verdict = self.debate_hypothesis(hyp, returns, context)
            verdicts.append(verdict)

        # Sort by consensus score
        verdicts.sort(key=lambda v: v.consensus_score, reverse=True)
        return verdicts[:top_k]

    def debate_summary(self, verdicts: list[DebateVerdict]) -> dict:
        """Summarize debate results across all verdicts."""
        if not verdicts:
            return {"n_debated": 0, "n_proceeding": 0}

        n_proceeding = sum(1 for v in verdicts if v.should_proceed)
        avg_confidence = float(np.mean([v.confidence for v in verdicts]))
        avg_consensus = float(np.mean([v.consensus_score for v in verdicts]))

        # Which agent was most often the swing vote?
        agent_supports = {}
        for v in verdicts:
            for op in v.agent_opinions:
                if op.agent_name not in agent_supports:
                    agent_supports[op.agent_name] = []
                agent_supports[op.agent_name].append(op.support_score)

        agent_avg = {k: float(np.mean(v)) for k, v in agent_supports.items()}
        most_bullish_agent = max(agent_avg, key=lambda k: agent_avg[k]) if agent_avg else None
        most_bearish_agent = min(agent_avg, key=lambda k: agent_avg[k]) if agent_avg else None

        return {
            "n_debated": len(verdicts),
            "n_proceeding": n_proceeding,
            "proceed_rate": float(n_proceeding / max(len(verdicts), 1)),
            "avg_confidence": avg_confidence,
            "avg_consensus": avg_consensus,
            "most_bullish_agent": most_bullish_agent,
            "most_bearish_agent": most_bearish_agent,
            "agent_avg_support": agent_avg,
        }
