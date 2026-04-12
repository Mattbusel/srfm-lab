"""
ConsensusProtocol — Agents reach consensus on market direction via
iterative voting rounds with Byzantine fault tolerance.

Architecture is analogous to Bayesian Debate from Event Horizon but
applied to MARL price consensus.

Components:
  - CredibilityTracker:  Beta(α, β) posterior over each agent's accuracy
  - WeightedVoting:      credibility-weighted majority vote
  - ByzantineFilter:     detect and exclude lying agents (3f+1 rule)
  - ConsensusProtocol:   iterative convergence rounds
  - EmergentConsensus:   track convergence speed and reliability
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats


# ============================================================
# Direction enum
# ============================================================

DIRECTION_BEARISH  = 0
DIRECTION_NEUTRAL  = 1
DIRECTION_BULLISH  = 2
DIRECTION_NAMES    = ["bearish", "neutral", "bullish"]


# ============================================================
# Agent Vote
# ============================================================

@dataclass
class Vote:
    """A single agent's vote on market direction."""
    agent_id:    str
    direction:   int        # 0=bearish, 1=neutral, 2=bullish
    confidence:  float      # [0, 1]
    round_num:   int
    timestamp:   int


# ============================================================
# Credibility Tracker
# ============================================================

class CredibilityTracker:
    """
    Bayesian credibility model for each agent using Beta distribution.

    Beta(α, β) posterior where:
      α = prior_alpha + correct_predictions
      β = prior_beta  + incorrect_predictions

    Credibility = E[Beta] = α / (α + β)
    """

    def __init__(
        self,
        agent_ids:    List[str],
        prior_alpha:  float = 2.0,
        prior_beta:   float = 2.0,
        decay:        float = 0.99,  # decay old evidence
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta  = prior_beta
        self.decay       = decay

        self._alpha: Dict[str, float] = {a: prior_alpha for a in agent_ids}
        self._beta:  Dict[str, float] = {a: prior_beta  for a in agent_ids}

        self.history: Dict[str, deque] = {
            a: deque(maxlen=100) for a in agent_ids
        }

    def update(self, agent_id: str, was_correct: bool) -> None:
        """
        Update agent's credibility given outcome.

        Args:
            agent_id:    agent to update
            was_correct: True if agent's prediction was correct
        """
        # Decay toward prior
        self._alpha[agent_id] = (
            self._alpha[agent_id] * self.decay
            + self.prior_alpha * (1 - self.decay)
        )
        self._beta[agent_id] = (
            self._beta[agent_id] * self.decay
            + self.prior_beta * (1 - self.decay)
        )

        # Update
        if was_correct:
            self._alpha[agent_id] += 1.0
        else:
            self._beta[agent_id] += 1.0

        self.history[agent_id].append(int(was_correct))

    def credibility(self, agent_id: str) -> float:
        """Return posterior mean credibility ∈ (0, 1)."""
        a = self._alpha[agent_id]
        b = self._beta[agent_id]
        return float(a / (a + b))

    def credibility_interval(
        self, agent_id: str, ci: float = 0.95
    ) -> Tuple[float, float]:
        """Return credibility posterior interval (lower, upper)."""
        a = self._alpha[agent_id]
        b = self._beta[agent_id]
        return stats.beta.interval(ci, a, b)  # type: ignore

    def all_credibilities(self) -> Dict[str, float]:
        return {a: self.credibility(a) for a in self._alpha}

    def top_k(self, k: int) -> List[str]:
        """Return k most credible agent IDs."""
        creds = self.all_credibilities()
        return sorted(creds, key=creds.get, reverse=True)[:k]  # type: ignore

    def add_agent(self, agent_id: str) -> None:
        self._alpha[agent_id] = self.prior_alpha
        self._beta[agent_id]  = self.prior_beta
        self.history[agent_id] = deque(maxlen=100)

    def remove_agent(self, agent_id: str) -> None:
        self._alpha.pop(agent_id, None)
        self._beta.pop(agent_id, None)
        self.history.pop(agent_id, None)


# ============================================================
# Weighted Voting
# ============================================================

class WeightedVoting:
    """
    Credibility-weighted majority vote over market directions.

    Computes a probability distribution over [bearish, neutral, bullish]
    where each agent's vote is weighted by their credibility × confidence.
    """

    def __init__(self, credibility_tracker: CredibilityTracker) -> None:
        self.cred = credibility_tracker

    def tally(
        self,
        votes:           List[Vote],
        use_credibility: bool = True,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Aggregate votes into a direction distribution.

        Args:
            votes:           list of agent votes
            use_credibility: weight by credibility (else uniform)

        Returns:
            (probs, consensus_direction, consensus_confidence)
        """
        weighted_votes = np.zeros(3, dtype=np.float64)

        for v in votes:
            if use_credibility:
                weight = self.cred.credibility(v.agent_id) * v.confidence
            else:
                weight = v.confidence
            weighted_votes[v.direction] += weight

        total = weighted_votes.sum()
        if total < 1e-12:
            probs = np.ones(3) / 3.0
        else:
            probs = weighted_votes / total

        consensus_dir  = int(np.argmax(probs))
        consensus_conf = float(probs[consensus_dir])
        return probs.astype(np.float32), consensus_dir, consensus_conf

    def entropy(self, probs: np.ndarray) -> float:
        """Shannon entropy of vote distribution (in nats)."""
        probs = probs + 1e-12
        return float(-np.sum(probs * np.log(probs)))

    def agreement(self, probs: np.ndarray) -> float:
        """
        Agreement index: 1 if unanimous, 0 if uniform.
        agreement = 1 - H / H_max
        """
        h     = self.entropy(probs)
        h_max = math.log(3)  # max entropy for 3 directions
        return float(max(0.0, 1.0 - h / h_max))


# ============================================================
# Byzantine Fault Tolerance
# ============================================================

class ByzantineFilter:
    """
    Filter out potential Byzantine (lying) agents.

    Uses the 3f+1 principle: with n agents, up to f = (n-1)//3 can be
    Byzantine. Identifies outliers by comparing each agent's vote to
    the majority.

    Detection heuristic:
      - Track agents whose votes consistently oppose consensus
      - Flag agents with credibility < threshold
      - Apply trimmed mean voting ignoring outliers
    """

    def __init__(
        self,
        n_agents:         int,
        credibility_threshold: float = 0.2,
        suspect_window:   int   = 20,
    ) -> None:
        self.n_agents    = n_agents
        self.max_faulty  = max(0, (n_agents - 1) // 3)
        self.cred_thresh = credibility_threshold
        self.window      = suspect_window

        self._vote_history:    Dict[str, deque] = defaultdict(lambda: deque(maxlen=suspect_window))
        self._consensus_hist:  deque = deque(maxlen=suspect_window)
        self._suspect_counts:  Dict[str, int] = defaultdict(int)

    def record_round(self, votes: List[Vote], consensus_dir: int) -> None:
        """Record votes and consensus to build detection history."""
        self._consensus_hist.append(consensus_dir)
        for v in votes:
            self._vote_history[v.agent_id].append(v.direction)
            # Flag if vote consistently opposes consensus
            if v.direction != consensus_dir and consensus_dir != DIRECTION_NEUTRAL:
                self._suspect_counts[v.agent_id] += 1

    def get_suspects(
        self,
        creds: Dict[str, float],
        votes: List[Vote],
    ) -> List[str]:
        """
        Return list of agent IDs that are likely Byzantine.

        Identifies up to f agents based on:
          1. Credibility below threshold
          2. High disagreement with consensus
        """
        suspects = []

        # Low credibility
        for aid, cred in creds.items():
            if cred < self.cred_thresh:
                suspects.append(aid)

        # High disagreement rate
        consensus_hist = list(self._consensus_hist)
        if len(consensus_hist) >= 5:
            for v in votes:
                hist = list(self._vote_history[v.agent_id])
                if len(hist) >= 5:
                    disagree_rate = sum(
                        1 for vh, ch in zip(hist[-10:], consensus_hist[-10:])
                        if vh != ch and ch != DIRECTION_NEUTRAL
                    ) / min(10, len(hist))
                    if disagree_rate > 0.8:
                        suspects.append(v.agent_id)

        # Deduplicate and limit to max_faulty
        suspects = list(set(suspects))
        return suspects[:self.max_faulty]

    def filter_votes(
        self,
        votes:    List[Vote],
        suspects: List[str],
    ) -> List[Vote]:
        """Remove suspect agents' votes."""
        suspect_set = set(suspects)
        return [v for v in votes if v.agent_id not in suspect_set]


# ============================================================
# Consensus Protocol
# ============================================================

@dataclass
class ConsensusResult:
    direction:         int
    confidence:        float
    vote_probs:        np.ndarray
    n_rounds:          int
    converged:         bool
    n_agents_voted:    int
    n_filtered:        int
    agreement_score:   float


class ConsensusProtocol:
    """
    Multi-round consensus protocol for agents to agree on market direction.

    Algorithm:
      Round 1: All agents submit initial votes
      Round k: Agents observe weighted vote tally; update their vote
               (Bayesian update of own signal with social information)
      Terminate: when KL(votes_k || votes_{k-1}) < epsilon
                 or max_rounds reached
    """

    def __init__(
        self,
        agent_ids:        List[str],
        max_rounds:       int   = 5,
        convergence_kl:   float = 0.01,
        prior_alpha:      float = 2.0,
        prior_beta:       float = 2.0,
        credibility_decay: float = 0.99,
        byzantine_tolerance: bool = True,
    ) -> None:
        self.agent_ids    = list(agent_ids)
        self.max_rounds   = max_rounds
        self.convergence_kl = convergence_kl
        self.byzantine    = byzantine_tolerance

        self.cred_tracker = CredibilityTracker(
            agent_ids, prior_alpha, prior_beta, credibility_decay
        )
        self.voter = WeightedVoting(self.cred_tracker)
        self.byz_filter = ByzantineFilter(len(agent_ids))

        # History
        self.consensus_history: deque = deque(maxlen=500)
        self.convergence_history: List[int] = []
        self.round_stats: List[Dict] = []

    def run_consensus(
        self,
        initial_votes: Dict[str, Vote],
        agent_vote_update_fn: Optional[Any] = None,
    ) -> ConsensusResult:
        """
        Run consensus protocol from initial votes.

        Args:
            initial_votes:         agent_id → Vote for round 0
            agent_vote_update_fn:  optional callable(agent_id, social_probs) → new_direction
                                   if None, agents don't update (simple vote)

        Returns:
            ConsensusResult
        """
        votes         = list(initial_votes.values())
        prev_probs    = np.ones(3) / 3.0
        n_filtered    = 0
        converged     = False

        for rnd in range(self.max_rounds):
            # Byzantine filtering
            if self.byzantine:
                creds    = self.cred_tracker.all_credibilities()
                suspects = self.byz_filter.get_suspects(creds, votes)
                filtered = self.byz_filter.filter_votes(votes, suspects)
                n_filtered = len(suspects)
            else:
                filtered = votes
                suspects = []

            # Tally
            probs, consensus_dir, conf = self.voter.tally(filtered)

            # Record round for byzantine detection
            self.byz_filter.record_round(votes, consensus_dir)

            # Check convergence
            kl = self._kl_div(probs, prev_probs)
            self.round_stats.append({
                "round":       rnd,
                "kl":          kl,
                "consensus":   consensus_dir,
                "confidence":  conf,
                "n_filtered":  n_filtered,
            })

            if kl < self.convergence_kl and rnd > 0:
                converged = True
                break

            prev_probs = probs.copy()

            # Allow agents to update votes given social information
            if agent_vote_update_fn is not None:
                for v in votes:
                    new_dir = agent_vote_update_fn(v.agent_id, probs)
                    v.direction = new_dir
                    v.round_num = rnd + 1

        agreement = self.voter.agreement(probs)
        result = ConsensusResult(
            direction       = consensus_dir,
            confidence      = conf,
            vote_probs      = probs,
            n_rounds        = rnd + 1,
            converged       = converged,
            n_agents_voted  = len(votes),
            n_filtered      = n_filtered,
            agreement_score = agreement,
        )
        self.consensus_history.append(result)
        self.convergence_history.append(rnd + 1)
        return result

    def update_credibilities(
        self,
        agent_votes:   Dict[str, Vote],
        actual_outcome: int,  # DIRECTION_BEARISH / _NEUTRAL / _BULLISH
    ) -> None:
        """
        Update credibility posteriors based on whether agents were correct.
        Call after price move resolves.
        """
        for aid, vote in agent_votes.items():
            correct = (vote.direction == actual_outcome)
            self.cred_tracker.update(aid, correct)

    def simple_vote(
        self,
        votes: Dict[str, Vote],
    ) -> ConsensusResult:
        """
        Single-round weighted vote (no iteration).
        Fast alternative for high-frequency use.
        """
        vote_list                   = list(votes.values())
        probs, consensus_dir, conf  = self.voter.tally(vote_list)
        agreement                   = self.voter.agreement(probs)
        return ConsensusResult(
            direction       = consensus_dir,
            confidence      = conf,
            vote_probs      = probs,
            n_rounds        = 1,
            converged       = True,
            n_agents_voted  = len(vote_list),
            n_filtered      = 0,
            agreement_score = agreement,
        )

    def mean_convergence_rounds(self) -> float:
        if not self.convergence_history:
            return 0.0
        return float(np.mean(self.convergence_history))

    def consensus_stability(self, window: int = 20) -> float:
        """
        Fraction of recent consensus rounds where all voted for the same direction.
        """
        recent = list(self.consensus_history)[-window:]
        if not recent:
            return 0.0
        unanimous = sum(1 for r in recent if r.agreement_score > 0.8)
        return float(unanimous) / len(recent)

    @staticmethod
    def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
        p = np.asarray(p, dtype=np.float64) + eps
        q = np.asarray(q, dtype=np.float64) + eps
        p /= p.sum()
        q /= q.sum()
        return float(np.sum(p * np.log(p / q)))


# ============================================================
# Emergent Consensus Tracker
# ============================================================

class EmergentConsensusTracker:
    """
    Tracks emergent properties of agent consensus over time.

    Metrics:
      - Speed of convergence (rounds to agreement)
      - Reliability (fraction correct vs future price)
      - Predictive accuracy by agent type
      - Cascade detection: did a single influential agent seed consensus?
    """

    def __init__(self, agent_ids: List[str], agent_types: Dict[str, str]) -> None:
        self.agent_ids   = agent_ids
        self.agent_types = agent_types

        self._results: deque = deque(maxlen=1000)
        self._correct: deque = deque(maxlen=1000)
        self._by_type: Dict[str, List[bool]] = defaultdict(list)

    def record(self, result: ConsensusResult, outcome: Optional[int] = None) -> None:
        """Record a consensus result and optional true outcome."""
        self._results.append(result)
        if outcome is not None:
            correct = (result.direction == outcome)
            self._correct.append(correct)

    def record_agent_accuracy(
        self, agent_id: str, direction: int, outcome: int
    ) -> None:
        correct = (direction == outcome)
        atype   = self.agent_types.get(agent_id, "unknown")
        self._by_type[atype].append(correct)

    def consensus_accuracy(self) -> float:
        if not self._correct:
            return 0.0
        return float(np.mean(list(self._correct)))

    def mean_rounds(self) -> float:
        if not self._results:
            return 0.0
        return float(np.mean([r.n_rounds for r in self._results]))

    def mean_agreement(self) -> float:
        if not self._results:
            return 0.0
        return float(np.mean([r.agreement_score for r in self._results]))

    def accuracy_by_type(self) -> Dict[str, float]:
        return {
            t: float(np.mean(v)) if v else 0.0
            for t, v in self._by_type.items()
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "consensus_accuracy":  self.consensus_accuracy(),
            "mean_rounds":         self.mean_rounds(),
            "mean_agreement":      self.mean_agreement(),
            "accuracy_by_type":    self.accuracy_by_type(),
            "n_consensus_rounds":  len(self._results),
        }
