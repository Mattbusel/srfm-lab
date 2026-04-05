"""
debate-system/debate/chamber.py

DebateChamber: orchestrates the multi-agent debate over a hypothesis.

Process
-------
1. Load all 6 analyst agents.
2. For each hypothesis, run all agents in parallel (ThreadPoolExecutor).
3. Collect AnalystVerdicts.
4. Compute weighted vote: weight = agent.credibility_score * verdict.confidence.
5. Tally: FOR weight / total weight.
6. Classify:
     APPROVED        if weighted_for > 0.65
     REJECTED        if weighted_for < 0.35
     NEEDS_MORE_DATA otherwise
7. Persist full transcript to SQLite.
8. Return DebateResult.

Special case: Risk veto
-----------------------
If the RiskManagementAnalyst votes AGAINST with confidence >= 0.99
(i.e., hard veto), the result is immediately REJECTED regardless of
other votes. This is the non-negotiable drawdown gate.

Weighted vote rationale
-----------------------
Not all agents are equally reliable.  Early on, all credibility scores
are 0.5 (equal weight), but over time agents that make correct calls
accumulate higher alpha in their Beta prior and earn higher weights.
A consistently-right StatisticalAnalyst should outweigh a persistently-
wrong DevilsAdvocate (though in practice the DA should also be right
most of the time, just vote FOR less often).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from hypothesis.types import Hypothesis

from debate_system.agents.base_agent import AnalystVerdict, Vote
from debate_system.agents.devil_advocate import DevilsAdvocate
from debate_system.agents.market_structure import MarketStructureAnalyst
from debate_system.agents.quant_researcher import QuantResearcher
from debate_system.agents.regime_specialist import RegimeSpecialist
from debate_system.agents.risk_manager import RiskManagementAnalyst
from debate_system.agents.statistician import StatisticalAnalyst
from debate_system.debate.transcript import (
    DebateTranscript,
    TranscriptStore,
    VoteSummary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class DebateOutcome(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_MORE_DATA = "NEEDS_MORE_DATA"


@dataclass
class DebateResult:
    """Outcome of a complete debate session."""
    hypothesis_id: str
    debate_id: str
    outcome: DebateOutcome
    weighted_for: float
    weighted_against: float
    consensus_score: float          # how unanimous the vote was (0=split, 1=unanimous)
    veto_issued: bool
    transcript: DebateTranscript
    verdicts: list[AnalystVerdict]

    @property
    def is_approved(self) -> bool:
        return self.outcome == DebateOutcome.APPROVED

    @property
    def is_rejected(self) -> bool:
        return self.outcome == DebateOutcome.REJECTED

    @property
    def needs_more_data(self) -> bool:
        return self.outcome == DebateOutcome.NEEDS_MORE_DATA

    def summary(self) -> str:
        return (
            f"Debate [{self.debate_id[:8]}] on hypothesis {self.hypothesis_id[:8]}: "
            f"{self.outcome.value} "
            f"(FOR={self.weighted_for:.2f}, "
            f"AGAINST={self.weighted_against:.2f}, "
            f"consensus={self.consensus_score:.2f}"
            + (" | VETO" if self.veto_issued else "")
            + ")"
        )


# ---------------------------------------------------------------------------
# DebateChamber
# ---------------------------------------------------------------------------

class DebateChamber:
    """
    Orchestrates the multi-agent debate over trading hypotheses.

    Usage
    -----
    chamber = DebateChamber()
    result = chamber.debate(hypothesis, market_data)
    if result.is_approved:
        promoter.promote(hypothesis)
    """

    APPROVE_THRESHOLD = 0.65
    REJECT_THRESHOLD = 0.35
    RISK_VETO_CONFIDENCE = 0.98      # RiskManager confidence >= this = hard veto

    def __init__(self, db_path: str | None = None) -> None:
        self.agents = [
            StatisticalAnalyst(),
            DevilsAdvocate(),
            MarketStructureAnalyst(),
            RegimeSpecialist(),
            RiskManagementAnalyst(),
            QuantResearcher(),
        ]
        store_kwargs: dict[str, Any] = {}
        if db_path:
            from pathlib import Path
            store_kwargs["db_path"] = Path(db_path)
        self.transcript_store = TranscriptStore(**store_kwargs)
        logger.info(
            "DebateChamber initialised with %d agents: %s",
            len(self.agents),
            [a.name for a in self.agents],
        )

    def debate(
        self,
        hypothesis: Hypothesis,
        market_data: dict[str, Any],
    ) -> DebateResult:
        """
        Run the full debate for a single hypothesis.

        All agents analyse in parallel; results are collected and tallied.

        Parameters
        ----------
        hypothesis  : Hypothesis to evaluate.
        market_data : Shared data dict passed to all agents.  Should contain
                      all keys expected by each agent.  Missing keys default
                      to safe values inside each agent's analyze() method.
        """
        transcript = DebateTranscript.new(
            hypothesis_id=hypothesis.hypothesis_id,
            description=hypothesis.description,
        )
        initial_round = transcript.add_round("initial_analysis")

        # --- Parallel agent execution -----------------------------------
        verdicts: list[AnalystVerdict] = []
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=len(self.agents), thread_name_prefix="agent") as pool:
            future_to_agent = {
                pool.submit(agent.analyze, hypothesis, market_data): agent
                for agent in self.agents
            }
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    verdict = future.result(timeout=30)
                    verdicts.append(verdict)
                    initial_round.add_argument(verdict)
                    logger.debug(
                        "%s voted %s (conf=%.2f)",
                        agent.name,
                        verdict.vote.value,
                        verdict.confidence,
                    )
                except Exception as exc:
                    errors.append(f"{agent.name}: {exc}")
                    logger.exception("Agent %s raised an exception", agent.name)

        if not verdicts:
            raise RuntimeError(
                f"All agents failed for hypothesis {hypothesis.hypothesis_id}. "
                f"Errors: {errors}"
            )

        # --- Risk veto check -------------------------------------------
        veto_issued = self._check_risk_veto(verdicts)

        # --- Weighted vote calculation ----------------------------------
        agent_by_name = {a.name: a for a in self.agents}
        total_for_weight = 0.0
        total_against_weight = 0.0
        total_weight = 0.0
        agent_vote_records: list[dict[str, Any]] = []

        for verdict in verdicts:
            agent = agent_by_name.get(verdict.agent_name)
            cred = agent.credibility_score if agent else 0.5
            weight = verdict.effective_weight(cred)
            agent_vote_records.append({
                "agent": verdict.agent_name,
                "vote": verdict.vote.value,
                "confidence": verdict.confidence,
                "credibility": round(cred, 4),
                "effective_weight": round(weight, 4),
            })
            if verdict.vote == Vote.FOR:
                total_for_weight += weight
            elif verdict.vote == Vote.AGAINST:
                total_against_weight += weight
            total_weight += weight

        if total_weight > 0:
            weighted_for = total_for_weight / total_weight
            weighted_against = total_against_weight / total_weight
        else:
            weighted_for = 0.0
            weighted_against = 0.0

        # Consensus score: how far from 50/50 split (0=split, 1=unanimous)
        consensus_score = abs(weighted_for - 0.5) * 2.0

        # --- Outcome determination -------------------------------------
        if veto_issued:
            outcome = DebateOutcome.REJECTED
        elif weighted_for > self.APPROVE_THRESHOLD:
            outcome = DebateOutcome.APPROVED
        elif weighted_for < self.REJECT_THRESHOLD:
            outcome = DebateOutcome.REJECTED
        else:
            outcome = DebateOutcome.NEEDS_MORE_DATA

        # --- Finalise transcript ----------------------------------------
        transcript.set_final_positions(verdicts)
        vote_summary = VoteSummary(
            weighted_for=round(weighted_for, 4),
            weighted_against=round(weighted_against, 4),
            total_weight=round(total_weight, 4),
            result=outcome.value,
            agent_votes=agent_vote_records,
        )
        transcript.close(vote_summary)

        try:
            self.transcript_store.save(transcript)
        except Exception as exc:
            logger.error("Failed to persist transcript: %s", exc)

        result = DebateResult(
            hypothesis_id=hypothesis.hypothesis_id,
            debate_id=transcript.debate_id,
            outcome=outcome,
            weighted_for=weighted_for,
            weighted_against=weighted_against,
            consensus_score=consensus_score,
            veto_issued=veto_issued,
            transcript=transcript,
            verdicts=verdicts,
        )
        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_risk_veto(self, verdicts: list[AnalystVerdict]) -> bool:
        """
        Check if the RiskManagementAnalyst issued a hard veto.
        Hard veto = AGAINST vote with confidence >= RISK_VETO_CONFIDENCE.
        """
        for v in verdicts:
            if (
                v.agent_name == "RiskManagementAnalyst"
                and v.vote == Vote.AGAINST
                and v.confidence >= self.RISK_VETO_CONFIDENCE
            ):
                logger.warning(
                    "RISK VETO issued by RiskManagementAnalyst "
                    "(confidence=%.3f). Hypothesis will be REJECTED.",
                    v.confidence,
                )
                return True
        return False

    def get_agent(self, name: str) -> Any:
        """Retrieve an agent by name (for credibility updates)."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def agent_credibility_summary(self) -> dict[str, float]:
        """Return current credibility scores for all agents."""
        return {a.name: round(a.credibility_score, 4) for a in self.agents}
