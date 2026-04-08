"""
debate-system/debate_tournament.py

Tournament-style debate engine for hypothesis evaluation.

Runs a structured multi-round tournament where every registered agent
evaluates every candidate hypothesis.  Hypotheses that fail hard veto
conditions (e.g., risk-manager reject) are eliminated.  Surviving
hypotheses are scored via a weighted vote matrix, with optional
iterative consensus-building rounds where agents can update their views
after seeing other agents' arguments.

Tournament phases:
  1. Round-Robin  -- every agent evaluates every hypothesis independently
  2. Elimination  -- hypotheses that fail veto criteria are removed
  3. Devil's Advocate -- one agent randomly argues against each survivor
  4. Consensus Building -- iterative update rounds (agents revise)
  5. Final Verdict -- weighted vote aggregation + veto enforcement
  6. Calibration -- compare predictions to outcomes, update agent weights

Key concepts:
  - ScoringMatrix: agents x hypotheses matrix of scores
  - DebateTranscript: full log of every argument and counter-argument
  - CalibrationTracker: historical accuracy per agent for weight adjustment
  - Quality metrics: agreement level, argument diversity, time to consensus
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols & types
# ---------------------------------------------------------------------------

class DebateAgent(Protocol):
    """Minimal protocol that tournament agents must satisfy."""

    name: str
    specialization: str

    @property
    def credibility_score(self) -> float: ...

    def analyze(
        self, hypothesis: Any, market_data: dict[str, Any],
    ) -> Any: ...

    def update_credibility(self, was_right: bool) -> None: ...


class HypothesisLike(Protocol):
    """Minimal hypothesis protocol."""
    id: str | int
    name: str


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TournamentPhase(str, Enum):
    ROUND_ROBIN     = "round_robin"
    ELIMINATION     = "elimination"
    DEVILS_ADVOCATE = "devils_advocate"
    CONSENSUS       = "consensus"
    FINAL_VERDICT   = "final_verdict"


class HypothesisStatus(str, Enum):
    ACTIVE     = "active"
    ELIMINATED = "eliminated"
    APPROVED   = "approved"
    REJECTED   = "rejected"


class VetoReason(str, Enum):
    RISK_LIMIT       = "risk_limit_breach"
    EXECUTION_COST   = "execution_cost_excessive"
    STATISTICAL       = "statistical_invalidity"
    REGIME_MISMATCH  = "regime_mismatch"
    CUSTOM           = "custom_veto"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentScore:
    """One agent's evaluation of one hypothesis."""
    agent_name: str
    hypothesis_id: str
    vote: str              # "FOR" | "AGAINST" | "ABSTAIN"
    confidence: float      # 0-1
    raw_score: float       # agent's internal score
    reasoning: str
    key_concerns: list[str] = field(default_factory=list)
    round_number: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    @property
    def signed_score(self) -> float:
        """Positive for FOR, negative for AGAINST, zero for ABSTAIN."""
        if self.vote == "FOR":
            return self.confidence
        elif self.vote == "AGAINST":
            return -self.confidence
        return 0.0


@dataclass
class VetoRecord:
    """Record of a veto applied to a hypothesis."""
    hypothesis_id: str
    agent_name: str
    reason: VetoReason
    detail: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


@dataclass
class TranscriptEntry:
    """One line in the debate transcript."""
    phase: TournamentPhase
    round_number: int
    agent_name: str
    hypothesis_id: str
    action: str            # "evaluate" | "veto" | "devils_advocate" | "revise"
    content: str
    score_delta: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


@dataclass
class TournamentResult:
    """Final output of a tournament run."""
    tournament_id: str
    started_at: str
    finished_at: str
    duration_seconds: float
    n_hypotheses_entered: int
    n_hypotheses_survived: int
    n_agents: int
    n_rounds: int
    scoring_matrix: dict[str, dict[str, float]]   # agent -> hyp -> score
    final_rankings: list[dict[str, Any]]           # sorted list of hypotheses
    vetoes: list[VetoRecord]
    transcript: list[TranscriptEntry]
    quality_metrics: dict[str, float]
    calibration_snapshot: dict[str, float]         # agent -> current accuracy

    def to_dict(self) -> dict[str, Any]:
        return {
            "tournament_id": self.tournament_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "n_hypotheses_entered": self.n_hypotheses_entered,
            "n_hypotheses_survived": self.n_hypotheses_survived,
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
            "final_rankings": self.final_rankings,
            "quality_metrics": {
                k: round(v, 4) for k, v in self.quality_metrics.items()
            },
        }


# ---------------------------------------------------------------------------
# Calibration tracker
# ---------------------------------------------------------------------------

class CalibrationTracker:
    """
    Tracks per-agent prediction accuracy over time to calibrate vote weights.

    Agents with better track records get higher weight in the final vote.
    Weight = base_weight * (0.5 + accuracy).  An agent at 50% accuracy
    gets 1.0x weight; at 80% accuracy, 1.3x.
    """

    def __init__(self, decay: float = 0.99) -> None:
        self._decay = decay
        self._records: dict[str, list[bool]] = defaultdict(list)
        self._ema_accuracy: dict[str, float] = {}

    def record(self, agent_name: str, was_correct: bool) -> None:
        self._records[agent_name].append(was_correct)
        # EMA update
        prev = self._ema_accuracy.get(agent_name, 0.5)
        val = 1.0 if was_correct else 0.0
        self._ema_accuracy[agent_name] = prev * self._decay + val * (1 - self._decay)

    def accuracy(self, agent_name: str) -> float:
        return self._ema_accuracy.get(agent_name, 0.5)

    def weight_multiplier(self, agent_name: str) -> float:
        acc = self.accuracy(agent_name)
        return 0.5 + acc  # range [0.5, 1.5]

    def n_observations(self, agent_name: str) -> int:
        return len(self._records.get(agent_name, []))

    def snapshot(self) -> dict[str, float]:
        return dict(self._ema_accuracy)


# ---------------------------------------------------------------------------
# Scoring matrix
# ---------------------------------------------------------------------------

class ScoringMatrix:
    """
    Agents x Hypotheses matrix of evaluation scores.

    Supports querying by agent, by hypothesis, and computing weighted
    aggregates.
    """

    def __init__(self) -> None:
        self._scores: dict[str, dict[str, list[AgentScore]]] = defaultdict(
            lambda: defaultdict(list),
        )

    def add(self, score: AgentScore) -> None:
        self._scores[score.agent_name][score.hypothesis_id].append(score)

    def get_agent_scores(self, agent_name: str) -> dict[str, list[AgentScore]]:
        return dict(self._scores.get(agent_name, {}))

    def get_hypothesis_scores(self, hyp_id: str) -> list[AgentScore]:
        result = []
        for agent_scores in self._scores.values():
            for s in agent_scores.get(hyp_id, []):
                result.append(s)
        return result

    def latest_score(self, agent_name: str, hyp_id: str) -> AgentScore | None:
        scores = self._scores.get(agent_name, {}).get(hyp_id, [])
        return scores[-1] if scores else None

    def weighted_hypothesis_score(
        self,
        hyp_id: str,
        weight_fn: Callable[[str], float],
    ) -> float:
        """
        Compute weighted aggregate score for a hypothesis.
        weight_fn maps agent_name -> weight.
        """
        scores = self.get_hypothesis_scores(hyp_id)
        if not scores:
            return 0.0
        # Use latest score per agent
        latest: dict[str, AgentScore] = {}
        for s in scores:
            latest[s.agent_name] = s

        total_weight = 0.0
        weighted_sum = 0.0
        for agent_name, s in latest.items():
            w = weight_fn(agent_name)
            weighted_sum += s.signed_score * w
            total_weight += w
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Latest signed score per agent per hypothesis."""
        result: dict[str, dict[str, float]] = {}
        for agent_name, hyp_map in self._scores.items():
            result[agent_name] = {}
            for hyp_id, score_list in hyp_map.items():
                if score_list:
                    result[agent_name][hyp_id] = score_list[-1].signed_score
        return result

    @property
    def agent_names(self) -> list[str]:
        return list(self._scores.keys())

    @property
    def hypothesis_ids(self) -> set[str]:
        ids: set[str] = set()
        for hyp_map in self._scores.values():
            ids.update(hyp_map.keys())
        return ids


# ---------------------------------------------------------------------------
# Tournament configuration
# ---------------------------------------------------------------------------

@dataclass
class TournamentConfig:
    """Configuration for a tournament run."""
    max_consensus_rounds: int = 3         # max iterative update rounds
    convergence_threshold: float = 0.05   # stop consensus if max change < this
    veto_agents: list[str] = field(default_factory=lambda: ["RiskManager"])
    elimination_threshold: float = -0.5   # raw score below this = eliminated
    devils_advocate_enabled: bool = True
    seed: int | None = None               # for reproducible DA assignment
    min_agents_for_consensus: int = 3
    approval_threshold: float = 0.2       # weighted score above this = approved


# ---------------------------------------------------------------------------
# DebateTournament
# ---------------------------------------------------------------------------

class DebateTournament:
    """
    Orchestrates a tournament-style debate among multiple agents
    evaluating multiple hypotheses.
    """

    def __init__(
        self,
        agents: Sequence[DebateAgent],
        config: TournamentConfig | None = None,
        calibration: CalibrationTracker | None = None,
    ) -> None:
        self._agents = {a.name: a for a in agents}
        self._config = config or TournamentConfig()
        self._calibration = calibration or CalibrationTracker()
        self._rng = random.Random(self._config.seed)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        hypotheses: Sequence[Any],
        market_data: dict[str, Any],
    ) -> TournamentResult:
        """
        Execute the full tournament pipeline.

        Parameters
        ----------
        hypotheses  : Sequence of hypothesis objects (must have .id, .name).
        market_data : Shared market context passed to every agent.
        """
        t0 = time.monotonic()
        started = datetime.now(timezone.utc).isoformat()
        tid = self._generate_id(hypotheses)

        matrix = ScoringMatrix()
        transcript: list[TranscriptEntry] = []
        vetoes: list[VetoRecord] = []
        statuses: dict[str, HypothesisStatus] = {
            str(h.id): HypothesisStatus.ACTIVE for h in hypotheses
        }

        # Phase 1: Round-robin
        logger.info("Tournament %s: round-robin (%d agents x %d hypotheses)",
                     tid, len(self._agents), len(hypotheses))
        self._round_robin(hypotheses, market_data, matrix, transcript)

        # Phase 2: Elimination
        logger.info("Tournament %s: elimination phase", tid)
        self._elimination(
            hypotheses, matrix, statuses, vetoes, transcript, market_data,
        )

        active_hyps = [
            h for h in hypotheses
            if statuses[str(h.id)] == HypothesisStatus.ACTIVE
        ]

        # Phase 3: Devil's advocate
        if self._config.devils_advocate_enabled and active_hyps:
            logger.info("Tournament %s: devil's advocate phase", tid)
            self._devils_advocate(active_hyps, market_data, matrix, transcript)

        # Phase 4: Consensus building
        n_rounds = 0
        if len(active_hyps) > 0 and len(self._agents) >= self._config.min_agents_for_consensus:
            logger.info("Tournament %s: consensus building", tid)
            n_rounds = self._consensus_rounds(
                active_hyps, market_data, matrix, transcript,
            )

        # Phase 5: Final verdict
        logger.info("Tournament %s: final verdict", tid)
        rankings = self._final_verdict(
            hypotheses, matrix, statuses, transcript,
        )

        # Quality metrics
        quality = self._compute_quality_metrics(matrix, active_hyps, n_rounds)

        elapsed = time.monotonic() - t0
        finished = datetime.now(timezone.utc).isoformat()

        return TournamentResult(
            tournament_id=tid,
            started_at=started,
            finished_at=finished,
            duration_seconds=elapsed,
            n_hypotheses_entered=len(hypotheses),
            n_hypotheses_survived=len(active_hyps),
            n_agents=len(self._agents),
            n_rounds=1 + n_rounds,  # round-robin + consensus rounds
            scoring_matrix=matrix.to_dict(),
            final_rankings=rankings,
            vetoes=[v.__dict__ for v in vetoes],  # type: ignore[arg-type]
            transcript=transcript,
            quality_metrics=quality,
            calibration_snapshot=self._calibration.snapshot(),
        )

    # ------------------------------------------------------------------
    # Phase 1: Round-Robin
    # ------------------------------------------------------------------

    def _round_robin(
        self,
        hypotheses: Sequence[Any],
        market_data: dict[str, Any],
        matrix: ScoringMatrix,
        transcript: list[TranscriptEntry],
    ) -> None:
        """Every agent evaluates every hypothesis independently."""
        for agent in self._agents.values():
            for hyp in hypotheses:
                try:
                    verdict = agent.analyze(hyp, market_data)
                    score = AgentScore(
                        agent_name=agent.name,
                        hypothesis_id=str(hyp.id),
                        vote=str(getattr(verdict, "vote", "ABSTAIN")),
                        confidence=float(getattr(verdict, "confidence", 0.5)),
                        raw_score=float(getattr(verdict, "confidence", 0.5)),
                        reasoning=str(getattr(verdict, "reasoning", "")),
                        key_concerns=list(getattr(verdict, "key_concerns", [])),
                        round_number=0,
                    )
                    matrix.add(score)
                    transcript.append(TranscriptEntry(
                        phase=TournamentPhase.ROUND_ROBIN,
                        round_number=0,
                        agent_name=agent.name,
                        hypothesis_id=str(hyp.id),
                        action="evaluate",
                        content=score.reasoning,
                    ))
                except Exception as exc:
                    logger.warning(
                        "Agent %s failed on hypothesis %s: %s",
                        agent.name, hyp.id, exc,
                    )
                    transcript.append(TranscriptEntry(
                        phase=TournamentPhase.ROUND_ROBIN,
                        round_number=0,
                        agent_name=agent.name,
                        hypothesis_id=str(hyp.id),
                        action="error",
                        content=f"Agent evaluation failed: {exc}",
                    ))

    # ------------------------------------------------------------------
    # Phase 2: Elimination
    # ------------------------------------------------------------------

    def _elimination(
        self,
        hypotheses: Sequence[Any],
        matrix: ScoringMatrix,
        statuses: dict[str, HypothesisStatus],
        vetoes: list[VetoRecord],
        transcript: list[TranscriptEntry],
        market_data: dict[str, Any],
    ) -> None:
        """Eliminate hypotheses that fail veto or score too low."""
        for hyp in hypotheses:
            hid = str(hyp.id)
            if statuses[hid] != HypothesisStatus.ACTIVE:
                continue

            # Check veto agents
            for veto_name in self._config.veto_agents:
                score = matrix.latest_score(veto_name, hid)
                if score is not None and score.vote == "AGAINST" and score.confidence > 0.7:
                    veto = VetoRecord(
                        hypothesis_id=hid,
                        agent_name=veto_name,
                        reason=VetoReason.RISK_LIMIT,
                        detail=score.reasoning,
                    )
                    vetoes.append(veto)
                    statuses[hid] = HypothesisStatus.ELIMINATED
                    transcript.append(TranscriptEntry(
                        phase=TournamentPhase.ELIMINATION,
                        round_number=0,
                        agent_name=veto_name,
                        hypothesis_id=hid,
                        action="veto",
                        content=f"VETOED: {score.reasoning}",
                    ))
                    logger.info("Hypothesis %s vetoed by %s", hid, veto_name)
                    break

            if statuses[hid] != HypothesisStatus.ACTIVE:
                continue

            # Check aggregate score threshold
            def _weight_fn(name: str) -> float:
                agent = self._agents.get(name)
                if agent is None:
                    return 1.0
                return agent.credibility_score * self._calibration.weight_multiplier(name)

            agg = matrix.weighted_hypothesis_score(hid, _weight_fn)
            if agg < self._config.elimination_threshold:
                statuses[hid] = HypothesisStatus.ELIMINATED
                transcript.append(TranscriptEntry(
                    phase=TournamentPhase.ELIMINATION,
                    round_number=0,
                    agent_name="tournament",
                    hypothesis_id=hid,
                    action="eliminate",
                    content=(
                        f"Eliminated: aggregate score {agg:.3f} < "
                        f"threshold {self._config.elimination_threshold}"
                    ),
                ))

    # ------------------------------------------------------------------
    # Phase 3: Devil's Advocate
    # ------------------------------------------------------------------

    def _devils_advocate(
        self,
        active_hypotheses: Sequence[Any],
        market_data: dict[str, Any],
        matrix: ScoringMatrix,
        transcript: list[TranscriptEntry],
    ) -> None:
        """
        For each surviving hypothesis, randomly assign one agent to
        argue against it.  The DA agent re-evaluates with a bearish bias
        and its counter-arguments are logged.
        """
        agent_names = list(self._agents.keys())
        if len(agent_names) < 2:
            return

        for hyp in active_hypotheses:
            hid = str(hyp.id)
            # Pick an agent that voted FOR this hypothesis (if any)
            for_agents = [
                s.agent_name
                for s in matrix.get_hypothesis_scores(hid)
                if s.vote == "FOR"
            ]
            if for_agents:
                da_name = self._rng.choice(for_agents)
            else:
                da_name = self._rng.choice(agent_names)

            da_agent = self._agents.get(da_name)
            if da_agent is None:
                continue

            # Create a biased market_data to encourage contrarian view
            biased_data = dict(market_data)
            biased_data["devils_advocate_mode"] = True
            biased_data["force_contrarian"] = True

            try:
                verdict = da_agent.analyze(hyp, biased_data)
                da_score = AgentScore(
                    agent_name=f"{da_name}_DA",
                    hypothesis_id=hid,
                    vote="AGAINST",  # DA always argues against
                    confidence=float(getattr(verdict, "confidence", 0.5)) * 0.8,
                    raw_score=-float(getattr(verdict, "confidence", 0.5)),
                    reasoning=f"[DEVIL'S ADVOCATE] {getattr(verdict, 'reasoning', '')}",
                    key_concerns=list(getattr(verdict, "key_concerns", [])),
                    round_number=0,
                )
                matrix.add(da_score)
                transcript.append(TranscriptEntry(
                    phase=TournamentPhase.DEVILS_ADVOCATE,
                    round_number=0,
                    agent_name=f"{da_name}_DA",
                    hypothesis_id=hid,
                    action="devils_advocate",
                    content=da_score.reasoning,
                ))
            except Exception as exc:
                logger.warning("DA evaluation failed for %s: %s", da_name, exc)

    # ------------------------------------------------------------------
    # Phase 4: Consensus Building
    # ------------------------------------------------------------------

    def _consensus_rounds(
        self,
        active_hypotheses: Sequence[Any],
        market_data: dict[str, Any],
        matrix: ScoringMatrix,
        transcript: list[TranscriptEntry],
    ) -> int:
        """
        Iterative consensus rounds.  Each agent sees the current scoring
        matrix summary and can revise its scores.  Stops when scores
        converge or max rounds reached.
        """
        rounds_done = 0

        for round_num in range(1, self._config.max_consensus_rounds + 1):
            max_delta = 0.0
            round_context = self._build_consensus_context(matrix, active_hypotheses)

            for agent in self._agents.values():
                for hyp in active_hypotheses:
                    hid = str(hyp.id)
                    prev = matrix.latest_score(agent.name, hid)
                    if prev is None:
                        continue

                    # Provide consensus context in market_data
                    enriched = dict(market_data)
                    enriched["consensus_context"] = round_context.get(hid, {})
                    enriched["consensus_round"] = round_num

                    try:
                        verdict = agent.analyze(hyp, enriched)
                        new_conf = float(getattr(verdict, "confidence", prev.confidence))
                        new_vote = str(getattr(verdict, "vote", prev.vote))

                        new_score = AgentScore(
                            agent_name=agent.name,
                            hypothesis_id=hid,
                            vote=new_vote,
                            confidence=new_conf,
                            raw_score=new_conf,
                            reasoning=str(getattr(verdict, "reasoning", "")),
                            key_concerns=list(getattr(verdict, "key_concerns", [])),
                            round_number=round_num,
                        )

                        delta = abs(new_score.signed_score - prev.signed_score)
                        max_delta = max(max_delta, delta)

                        matrix.add(new_score)
                        transcript.append(TranscriptEntry(
                            phase=TournamentPhase.CONSENSUS,
                            round_number=round_num,
                            agent_name=agent.name,
                            hypothesis_id=hid,
                            action="revise",
                            content=new_score.reasoning,
                            score_delta=delta,
                        ))
                    except Exception:
                        pass  # keep previous score

            rounds_done = round_num
            logger.info(
                "Consensus round %d: max_delta=%.4f (threshold=%.4f)",
                round_num, max_delta, self._config.convergence_threshold,
            )
            if max_delta < self._config.convergence_threshold:
                logger.info("Consensus reached after %d rounds", round_num)
                break

        return rounds_done

    def _build_consensus_context(
        self,
        matrix: ScoringMatrix,
        hypotheses: Sequence[Any],
    ) -> dict[str, dict[str, Any]]:
        """Build a summary of current scores for each hypothesis."""
        context: dict[str, dict[str, Any]] = {}
        for hyp in hypotheses:
            hid = str(hyp.id)
            scores = matrix.get_hypothesis_scores(hid)
            if not scores:
                continue
            # Latest per agent
            latest: dict[str, AgentScore] = {}
            for s in scores:
                latest[s.agent_name] = s

            for_count = sum(1 for s in latest.values() if s.vote == "FOR")
            against_count = sum(1 for s in latest.values() if s.vote == "AGAINST")
            avg_conf = float(np.mean([s.confidence for s in latest.values()]))

            context[hid] = {
                "for_count": for_count,
                "against_count": against_count,
                "abstain_count": len(latest) - for_count - against_count,
                "avg_confidence": avg_conf,
                "key_concerns": [
                    c for s in latest.values() for c in s.key_concerns
                ][:10],
            }
        return context

    # ------------------------------------------------------------------
    # Phase 5: Final Verdict
    # ------------------------------------------------------------------

    def _final_verdict(
        self,
        hypotheses: Sequence[Any],
        matrix: ScoringMatrix,
        statuses: dict[str, HypothesisStatus],
        transcript: list[TranscriptEntry],
    ) -> list[dict[str, Any]]:
        """Compute final weighted rankings and approve/reject."""

        def _weight_fn(name: str) -> float:
            agent = self._agents.get(name)
            base = agent.credibility_score if agent else 0.5
            cal = self._calibration.weight_multiplier(name)
            return base * cal

        rankings: list[dict[str, Any]] = []

        for hyp in hypotheses:
            hid = str(hyp.id)
            status = statuses[hid]

            if status == HypothesisStatus.ELIMINATED:
                rankings.append({
                    "hypothesis_id": hid,
                    "name": getattr(hyp, "name", hid),
                    "status": HypothesisStatus.ELIMINATED.value,
                    "final_score": float("-inf"),
                    "vote_breakdown": {},
                })
                continue

            final_score = matrix.weighted_hypothesis_score(hid, _weight_fn)

            # Vote breakdown
            scores = matrix.get_hypothesis_scores(hid)
            latest: dict[str, AgentScore] = {}
            for s in scores:
                if not s.agent_name.endswith("_DA"):
                    latest[s.agent_name] = s

            breakdown = {
                name: {"vote": s.vote, "confidence": round(s.confidence, 3)}
                for name, s in latest.items()
            }

            if final_score >= self._config.approval_threshold:
                statuses[hid] = HypothesisStatus.APPROVED
            else:
                statuses[hid] = HypothesisStatus.REJECTED

            rankings.append({
                "hypothesis_id": hid,
                "name": getattr(hyp, "name", hid),
                "status": statuses[hid].value,
                "final_score": round(final_score, 4),
                "vote_breakdown": breakdown,
            })

            transcript.append(TranscriptEntry(
                phase=TournamentPhase.FINAL_VERDICT,
                round_number=0,
                agent_name="tournament",
                hypothesis_id=hid,
                action="verdict",
                content=(
                    f"Final score={final_score:.4f}, "
                    f"status={statuses[hid].value}"
                ),
            ))

        # Sort by final_score descending
        rankings.sort(key=lambda r: r["final_score"], reverse=True)
        return rankings

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    def _compute_quality_metrics(
        self,
        matrix: ScoringMatrix,
        active_hypotheses: Sequence[Any],
        n_consensus_rounds: int,
    ) -> dict[str, float]:
        """
        Compute debate quality metrics:
        - agreement_level: how much agents agree (1 = unanimous, 0 = split)
        - argument_diversity: how many unique concerns were raised
        - time_to_consensus: how many rounds it took
        - average_confidence: mean confidence across all evaluations
        """
        all_scores: list[AgentScore] = []
        for hyp in active_hypotheses:
            all_scores.extend(matrix.get_hypothesis_scores(str(hyp.id)))

        if not all_scores:
            return {
                "agreement_level": 0.0,
                "argument_diversity": 0.0,
                "time_to_consensus": float(n_consensus_rounds),
                "average_confidence": 0.0,
            }

        # Agreement: 1 - std of signed scores (lower variance = more agreement)
        signed = [s.signed_score for s in all_scores]
        agreement = max(0.0, 1.0 - float(np.std(signed)))

        # Diversity: number of unique key concerns
        all_concerns: set[str] = set()
        for s in all_scores:
            all_concerns.update(s.key_concerns)
        diversity = min(1.0, len(all_concerns) / max(1, len(all_scores)))

        avg_conf = float(np.mean([s.confidence for s in all_scores]))

        return {
            "agreement_level": round(agreement, 4),
            "argument_diversity": round(diversity, 4),
            "time_to_consensus": float(n_consensus_rounds),
            "average_confidence": round(avg_conf, 4),
            "n_unique_concerns": float(len(all_concerns)),
        }

    # ------------------------------------------------------------------
    # Calibration: record outcomes
    # ------------------------------------------------------------------

    def record_outcomes(
        self,
        outcomes: dict[str, bool],
    ) -> None:
        """
        After hypotheses resolve, record which agents were correct.

        Parameters
        ----------
        outcomes : hypothesis_id -> True if hypothesis was profitable.
        """
        for agent in self._agents.values():
            for hid, was_profitable in outcomes.items():
                # The agent was "right" if it voted FOR a profitable hyp
                # or AGAINST an unprofitable one
                # Use the scoring matrix from the last tournament if available
                pass  # In practice, integrate with stored matrix

    def record_agent_outcome(
        self, agent_name: str, was_correct: bool,
    ) -> None:
        """Record a single outcome for calibration."""
        self._calibration.record(agent_name, was_correct)
        agent = self._agents.get(agent_name)
        if agent is not None:
            agent.update_credibility(was_correct)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_id(hypotheses: Sequence[Any]) -> str:
        raw = f"{time.time()}_{len(hypotheses)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
