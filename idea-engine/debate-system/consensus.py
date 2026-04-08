"""
consensus.py
------------
Multi-agent consensus mechanism for hypothesis evaluation.

Aggregates votes from debate agents, tracks agent credibility, detects
disagreement, and auto-triggers a devil's advocate when consensus is too strong.
"""

from __future__ import annotations

import math
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations & constants
# ---------------------------------------------------------------------------

class Vote(Enum):
    STRONG_BUY   = 2
    BUY          = 1
    NEUTRAL      = 0
    SELL         = -1
    STRONG_SELL  = -2


VOTE_SCORES: Dict[Vote, float] = {
    Vote.STRONG_BUY:  1.0,
    Vote.BUY:         0.75,
    Vote.NEUTRAL:     0.5,
    Vote.SELL:        0.25,
    Vote.STRONG_SELL: 0.0,
}

DEVILS_ADVOCATE_THRESHOLD = 0.85    # consensus strength above this triggers DA
DISAGREEMENT_THRESHOLD    = 0.25    # std dev of vote scores above this = flag


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentVote:
    agent_id: str
    vote: Vote
    confidence: float          # agent's self-reported confidence [0, 1]
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class AccuracyRecord:
    """Track how accurate an agent has been historically."""
    n_correct: int = 0
    n_total: int = 0
    rolling_accuracy: float = 0.5   # initialised at prior
    decay: float = 0.95             # exponential moving average decay

    def update(self, correct: bool) -> None:
        self.n_total += 1
        outcome = 1.0 if correct else 0.0
        self.rolling_accuracy = (
            self.decay * self.rolling_accuracy + (1 - self.decay) * outcome
        )
        if correct:
            self.n_correct += 1

    @property
    def accuracy(self) -> float:
        if self.n_total == 0:
            return 0.5
        return self.rolling_accuracy


@dataclass
class ConsensusResult:
    hypothesis_id: str
    method: str
    consensus_vote: Vote
    consensus_score: float         # 0=strong sell, 1=strong buy
    consensus_strength: float      # 0=no agreement, 1=perfect agreement
    disagreement_flagged: bool
    devils_advocate_triggered: bool
    participating_agents: List[str]
    vote_breakdown: Dict[str, int] = field(default_factory=dict)
    weighted_scores: Dict[str, float] = field(default_factory=dict)
    reasoning_summary: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class DebateSession:
    session_id: str
    hypothesis_id: str
    hypothesis_description: str
    started_at: float
    closed_at: Optional[float] = None
    votes: List[AgentVote] = field(default_factory=list)
    consensus_results: List[ConsensusResult] = field(default_factory=list)
    minutes: List[str] = field(default_factory=list)   # structured log
    status: str = "open"   # open | closed | flagged


# ---------------------------------------------------------------------------
# Dempster-Shafer utilities
# ---------------------------------------------------------------------------

def _ds_combine(m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
    """
    Dempster-Shafer combination of two basic probability assignments.

    Both dicts map hypothesis labels to masses; must sum to ≤ 1.
    The remainder is mass on the universal set (Omega).
    """
    labels = set(m1.keys()) | set(m2.keys()) - {"_omega"}
    omega1 = 1.0 - sum(v for k, v in m1.items() if k != "_omega")
    omega2 = 1.0 - sum(v for k, v in m2.items() if k != "_omega")

    combined: Dict[str, float] = defaultdict(float)
    conflict = 0.0

    # A ∩ B masses
    all_keys = list(labels) + ["_omega"]
    for k1 in all_keys:
        mass1 = m1.get(k1, omega1 if k1 == "_omega" else 0.0)
        for k2 in all_keys:
            mass2 = m2.get(k2, omega2 if k2 == "_omega" else 0.0)
            product = mass1 * mass2
            if k1 == "_omega":
                intersection = k2
            elif k2 == "_omega":
                intersection = k1
            elif k1 == k2:
                intersection = k1
            else:
                intersection = None   # conflict

            if intersection is None:
                conflict += product
            else:
                combined[intersection] += product

    # Normalise by (1 - conflict)
    denom = 1.0 - conflict
    if denom < 1e-9:
        # Full conflict — return vacuous belief
        return {"_omega": 1.0}
    return {k: v / denom for k, v in combined.items()}


def _votes_to_ds_mass(votes: List[AgentVote]) -> Dict[str, float]:
    """Convert a list of agent votes to a DS mass function over {buy, sell, neutral}."""
    mass: Dict[str, float] = defaultdict(float)
    for v in votes:
        conf = v.confidence
        if v.vote in (Vote.BUY, Vote.STRONG_BUY):
            label = "buy"
        elif v.vote in (Vote.SELL, Vote.STRONG_SELL):
            label = "sell"
        else:
            label = "neutral"
        mass[label] += conf / len(votes)
    return dict(mass)


# ---------------------------------------------------------------------------
# Borda count
# ---------------------------------------------------------------------------

def _borda_scores(votes: List[AgentVote], agent_weights: Dict[str, float]) -> Dict[Vote, float]:
    """
    Weighted Borda count.

    Votes are ranked STRONG_BUY > BUY > NEUTRAL > SELL > STRONG_SELL.
    Each agent distributes points from n-1 down to 0.
    """
    ordered = [Vote.STRONG_BUY, Vote.BUY, Vote.NEUTRAL, Vote.SELL, Vote.STRONG_SELL]
    n = len(ordered)
    rank_points = {v: n - 1 - i for i, v in enumerate(ordered)}
    borda: Dict[Vote, float] = defaultdict(float)
    for av in votes:
        w = agent_weights.get(av.agent_id, 1.0)
        borda[av.vote] += rank_points[av.vote] * w
    return dict(borda)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class ConsensusEngine:
    """
    Aggregates agent votes on trading hypotheses using multiple voting methods.

    Features
    --------
    - Simple majority vote
    - Credibility-weighted vote
    - Dempster-Shafer belief combination
    - Borda count
    - Disagreement detection
    - Devil's advocate auto-trigger
    - Structured meeting minutes
    - Agent credibility tracking
    """

    def __init__(
        self,
        default_method: str = "weighted",
        devils_advocate_agent_id: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        default_method : str
            One of 'majority', 'weighted', 'dempster_shafer', 'borda'.
        devils_advocate_agent_id : str, optional
            ID of the agent to auto-invite when consensus is too strong.
        """
        self.default_method = default_method
        self.devils_advocate_id = devils_advocate_agent_id
        self._sessions: Dict[str, DebateSession] = {}
        self._agent_credibility: Dict[str, AccuracyRecord] = {}
        self._agent_weights_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def open_session(
        self,
        hypothesis_id: str,
        hypothesis_description: str = "",
        session_id: Optional[str] = None,
    ) -> str:
        sid = session_id or str(uuid.uuid4())[:8]
        session = DebateSession(
            session_id=sid,
            hypothesis_id=hypothesis_id,
            hypothesis_description=hypothesis_description,
            started_at=time.time(),
        )
        self._sessions[sid] = session
        self._log(session, f"Session {sid} opened for hypothesis '{hypothesis_id}'.")
        return sid

    def close_session(self, session_id: str) -> DebateSession:
        session = self._get_session(session_id)
        session.closed_at = time.time()
        session.status = "closed"
        self._log(session, f"Session {session_id} closed.")
        return session

    def get_session(self, session_id: str) -> DebateSession:
        return self._get_session(session_id)

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def cast_vote(self, session_id: str, vote: AgentVote) -> None:
        """Add an agent's vote to the session."""
        session = self._get_session(session_id)
        if session.status != "open":
            raise RuntimeError(f"Session {session_id} is not open.")
        session.votes.append(vote)
        self._log(
            session,
            f"Agent '{vote.agent_id}' voted {vote.vote.name} "
            f"(conf={vote.confidence:.2f}): {vote.reasoning[:80]}",
        )

    def tally(
        self,
        session_id: str,
        method: Optional[str] = None,
    ) -> ConsensusResult:
        """
        Compute consensus from current votes in a session.

        Parameters
        ----------
        method : str
            Override default method for this tally.
        """
        session = self._get_session(session_id)
        m = method or self.default_method
        votes = session.votes
        if not votes:
            raise ValueError("No votes cast in session.")

        agent_weights = self._compute_agent_weights(votes)

        if m == "majority":
            result = self._majority_vote(votes, session.hypothesis_id)
        elif m == "weighted":
            result = self._weighted_vote(votes, agent_weights, session.hypothesis_id)
        elif m == "dempster_shafer":
            result = self._ds_vote(votes, session.hypothesis_id)
        elif m == "borda":
            result = self._borda_vote(votes, agent_weights, session.hypothesis_id)
        else:
            raise ValueError(f"Unknown method: {m}. Choose majority/weighted/dempster_shafer/borda.")

        # Check disagreement
        scores = [VOTE_SCORES[av.vote] for av in votes]
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        result.disagreement_flagged = std > DISAGREEMENT_THRESHOLD

        # Devil's advocate trigger
        da_triggered = (
            result.consensus_strength > DEVILS_ADVOCATE_THRESHOLD
            and self.devils_advocate_id is not None
        )
        result.devils_advocate_triggered = da_triggered
        if da_triggered:
            self._log(
                session,
                f"DEVIL'S ADVOCATE TRIGGERED (consensus_strength={result.consensus_strength:.3f}). "
                f"Agent '{self.devils_advocate_id}' should provide contrarian analysis.",
            )
            session.status = "flagged"
        if result.disagreement_flagged:
            self._log(
                session,
                f"DISAGREEMENT FLAGGED (vote std={std:.3f}). Deeper analysis recommended.",
            )

        session.consensus_results.append(result)
        self._log(
            session,
            f"Tally ({m}): {result.consensus_vote.name} "
            f"score={result.consensus_score:.3f} strength={result.consensus_strength:.3f}",
        )
        return result

    # ------------------------------------------------------------------
    # Voting method implementations
    # ------------------------------------------------------------------

    def _majority_vote(
        self, votes: List[AgentVote], hypothesis_id: str
    ) -> ConsensusResult:
        """Simple majority: pick the most frequent vote."""
        vote_counts: Dict[Vote, int] = defaultdict(int)
        for av in votes:
            vote_counts[av.vote] += 1
        winner = max(vote_counts, key=lambda v: vote_counts[v])
        n = len(votes)
        strength = vote_counts[winner] / n

        breakdown = {v.name: c for v, c in vote_counts.items()}
        return ConsensusResult(
            hypothesis_id=hypothesis_id,
            method="majority",
            consensus_vote=winner,
            consensus_score=VOTE_SCORES[winner],
            consensus_strength=strength,
            disagreement_flagged=False,
            devils_advocate_triggered=False,
            participating_agents=[av.agent_id for av in votes],
            vote_breakdown=breakdown,
        )

    def _weighted_vote(
        self,
        votes: List[AgentVote],
        weights: Dict[str, float],
        hypothesis_id: str,
    ) -> ConsensusResult:
        """
        Credibility-weighted vote: continuous score in [0, 1].
        Also weighs by agent's self-reported confidence.
        """
        total_w = 0.0
        weighted_sum = 0.0
        w_scores: Dict[str, float] = {}

        for av in votes:
            cred = weights.get(av.agent_id, 1.0)
            combined_w = cred * av.confidence
            score = VOTE_SCORES[av.vote]
            weighted_sum += combined_w * score
            total_w += combined_w
            w_scores[av.agent_id] = round(combined_w, 4)

        if total_w < 1e-9:
            consensus_score = 0.5
        else:
            consensus_score = weighted_sum / total_w

        consensus_vote = self._score_to_vote(consensus_score)
        # Strength: 1 - normalised variance of scores
        scores = [VOTE_SCORES[av.vote] for av in votes]
        var = statistics.variance(scores) if len(scores) > 1 else 0.0
        strength = max(0.0, 1.0 - 4.0 * var)   # 4x because max var of 5-point scale is 0.25

        vote_counts: Dict[Vote, int] = defaultdict(int)
        for av in votes:
            vote_counts[av.vote] += 1

        return ConsensusResult(
            hypothesis_id=hypothesis_id,
            method="weighted",
            consensus_vote=consensus_vote,
            consensus_score=round(consensus_score, 4),
            consensus_strength=round(strength, 4),
            disagreement_flagged=False,
            devils_advocate_triggered=False,
            participating_agents=[av.agent_id for av in votes],
            vote_breakdown={v.name: c for v, c in vote_counts.items()},
            weighted_scores=w_scores,
        )

    def _ds_vote(
        self, votes: List[AgentVote], hypothesis_id: str
    ) -> ConsensusResult:
        """Dempster-Shafer belief combination across agents."""
        if not votes:
            raise ValueError("No votes.")

        # Start with first agent's mass
        combined = _votes_to_ds_mass([votes[0]])
        for av in votes[1:]:
            m2 = _votes_to_ds_mass([av])
            combined = _ds_combine(combined, m2)

        # Map to consensus
        buy_belief = combined.get("buy", 0.0)
        sell_belief = combined.get("sell", 0.0)
        neutral_belief = combined.get("neutral", 0.0)
        omega_belief = combined.get("_omega", 0.0)

        # Weighted score
        consensus_score = (
            buy_belief * 1.0
            + neutral_belief * 0.5
            + sell_belief * 0.0
            + omega_belief * 0.5   # ignorance → neutral
        )
        consensus_vote = self._score_to_vote(consensus_score)
        # Strength = 1 - omega (uncertainty)
        strength = max(0.0, 1.0 - omega_belief)

        return ConsensusResult(
            hypothesis_id=hypothesis_id,
            method="dempster_shafer",
            consensus_vote=consensus_vote,
            consensus_score=round(consensus_score, 4),
            consensus_strength=round(strength, 4),
            disagreement_flagged=False,
            devils_advocate_triggered=False,
            participating_agents=[av.agent_id for av in votes],
            vote_breakdown={
                "buy_belief": round(buy_belief, 4),
                "sell_belief": round(sell_belief, 4),
                "neutral_belief": round(neutral_belief, 4),
                "omega_belief": round(omega_belief, 4),
            },
        )

    def _borda_vote(
        self,
        votes: List[AgentVote],
        weights: Dict[str, float],
        hypothesis_id: str,
    ) -> ConsensusResult:
        """Weighted Borda count."""
        borda = _borda_scores(votes, weights)
        winner = max(borda, key=lambda v: borda[v])
        total_points = sum(borda.values())
        strength = borda.get(winner, 0.0) / max(total_points, 1.0)

        return ConsensusResult(
            hypothesis_id=hypothesis_id,
            method="borda",
            consensus_vote=winner,
            consensus_score=VOTE_SCORES[winner],
            consensus_strength=round(strength, 4),
            disagreement_flagged=False,
            devils_advocate_triggered=False,
            participating_agents=[av.agent_id for av in votes],
            vote_breakdown={v.name: round(s, 2) for v, s in borda.items()},
        )

    # ------------------------------------------------------------------
    # Agent credibility
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        agent_id: str,
        predicted_vote: Vote,
        actual_outcome: Vote,
    ) -> None:
        """Update agent accuracy after outcome is known."""
        if agent_id not in self._agent_credibility:
            self._agent_credibility[agent_id] = AccuracyRecord()
        correct = (
            (predicted_vote in (Vote.BUY, Vote.STRONG_BUY) and
             actual_outcome in (Vote.BUY, Vote.STRONG_BUY))
            or
            (predicted_vote in (Vote.SELL, Vote.STRONG_SELL) and
             actual_outcome in (Vote.SELL, Vote.STRONG_SELL))
            or
            (predicted_vote == Vote.NEUTRAL and actual_outcome == Vote.NEUTRAL)
        )
        self._agent_credibility[agent_id].update(correct)
        # Invalidate cache
        self._agent_weights_cache = {}

    def get_agent_credibility(self, agent_id: str) -> float:
        """Return agent's rolling accuracy score [0, 1]."""
        if agent_id not in self._agent_credibility:
            return 0.5
        return self._agent_credibility[agent_id].accuracy

    def all_agent_credibilities(self) -> Dict[str, float]:
        return {
            aid: rec.accuracy
            for aid, rec in self._agent_credibility.items()
        }

    def _compute_agent_weights(
        self, votes: List[AgentVote]
    ) -> Dict[str, float]:
        """Return normalised credibility weights for agents in this vote set."""
        raw: Dict[str, float] = {}
        for av in votes:
            raw[av.agent_id] = self.get_agent_credibility(av.agent_id)
        total = sum(raw.values())
        if total < 1e-9:
            return {k: 1.0 / len(raw) for k in raw}
        return {k: v / total for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Meeting minutes
    # ------------------------------------------------------------------

    def get_minutes(self, session_id: str) -> List[str]:
        return self._get_session(session_id).minutes

    def print_minutes(self, session_id: str) -> None:
        session = self._get_session(session_id)
        print(f"\n{'='*60}")
        print(f"MEETING MINUTES — Session {session_id}")
        print(f"Hypothesis: {session.hypothesis_description}")
        print(f"{'='*60}")
        for line in session.minutes:
            print(f"  {line}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_vote(score: float) -> Vote:
        if score >= 0.875:
            return Vote.STRONG_BUY
        elif score >= 0.625:
            return Vote.BUY
        elif score >= 0.375:
            return Vote.NEUTRAL
        elif score >= 0.125:
            return Vote.SELL
        else:
            return Vote.STRONG_SELL

    def _log(self, session: DebateSession, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        session.minutes.append(f"[{ts}] {message}")

    def _get_session(self, session_id: str) -> DebateSession:
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self._sessions[session_id]

    def __repr__(self) -> str:
        return (
            f"ConsensusEngine("
            f"sessions={len(self._sessions)}, "
            f"agents_tracked={len(self._agent_credibility)})"
        )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = ConsensusEngine(
        default_method="weighted",
        devils_advocate_agent_id="contrarian_agent",
    )

    # Simulate agent history so they have credibility scores
    for agent_id in ["quant_alpha", "macro_desk", "ml_model", "fundamental"]:
        for _ in range(20):
            import random
            pred = random.choice(list(Vote))
            actual = random.choice(list(Vote))
            engine.record_outcome(agent_id, pred, actual)

    sid = engine.open_session(
        hypothesis_id="H001",
        hypothesis_description="SPY momentum breakout above 200d MA",
    )

    # Cast votes
    votes = [
        AgentVote("quant_alpha", Vote.BUY,        0.8, "Momentum confirmed by volume"),
        AgentVote("macro_desk",  Vote.BUY,        0.7, "Risk-on regime, USD weak"),
        AgentVote("ml_model",    Vote.STRONG_BUY, 0.9, "Ensemble model score 0.82"),
        AgentVote("fundamental", Vote.NEUTRAL,    0.5, "Valuation stretched"),
    ]
    for v in votes:
        engine.cast_vote(sid, v)

    print("=== Weighted consensus ===")
    r = engine.tally(sid, method="weighted")
    print(vars(r))

    print("\n=== Dempster-Shafer consensus ===")
    r_ds = engine.tally(sid, method="dempster_shafer")
    print(vars(r_ds))

    print("\n=== Borda count consensus ===")
    r_borda = engine.tally(sid, method="borda")
    print(vars(r_borda))

    engine.close_session(sid)
    engine.print_minutes(sid)

    print("Agent credibilities:", engine.all_agent_credibilities())
