"""
debate-system/agents/base_agent.py

BaseAnalyst abstract class for the multi-agent debate system.

Each analyst specializes in a different lens of hypothesis evaluation.
Credibility scores are updated via Bayesian track-record tracking so that
agents who are consistently right earn higher vote weight over time.

Bayesian credibility update model:
    Prior: Beta(alpha=5, beta=5)  →  credibility_score = alpha / (alpha + beta) = 0.5
    On correct prediction:  alpha += 1
    On wrong prediction:    beta  += 1
    credibility_score = alpha / (alpha + beta)   ∈ (0, 1)

This gives a principled starting point of 0.5 and pulls toward
the agent's empirical accuracy over time.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from hypothesis.types import Hypothesis


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Vote(str, Enum):
    FOR = "FOR"
    AGAINST = "AGAINST"
    ABSTAIN = "ABSTAIN"


# ---------------------------------------------------------------------------
# AnalystVerdict
# ---------------------------------------------------------------------------

@dataclass
class AnalystVerdict:
    """
    The structured output of one analyst reviewing one hypothesis.

    Fields
    ------
    vote          : FOR / AGAINST / ABSTAIN
    confidence    : 0-1, how certain the analyst is about this vote.
                    Used to scale the agent's weighted contribution.
    reasoning     : Human-readable explanation of the vote.
    key_concerns  : Specific technical concerns the chamber should surface
                    to human reviewers or downstream agents.
    """

    agent_name: str
    vote: Vote
    confidence: float          # 0.0 – 1.0
    reasoning: str
    key_concerns: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def effective_weight(self, credibility_score: float) -> float:
        """
        Weighted vote strength = credibility * confidence.
        ABSTAIN always contributes 0.
        """
        if self.vote == Vote.ABSTAIN:
            return 0.0
        return credibility_score * self.confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "vote": self.vote.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_concerns": self.key_concerns,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# BaseAnalyst
# ---------------------------------------------------------------------------

class BaseAnalyst(ABC):
    """
    Abstract base for all debate-system analysts.

    Subclasses implement `analyze()` with their specialised logic.
    The base class handles Bayesian credibility bookkeeping so the
    DebateChamber can weight each agent's vote by track record.
    """

    def __init__(
        self,
        name: str,
        specialization: str,
        initial_credibility: float = 0.5,
        # Bayesian Beta prior hyper-parameters
        alpha: float = 5.0,
        beta: float = 5.0,
    ) -> None:
        self.name = name
        self.specialization = specialization
        # These track the running Beta distribution parameters.
        # credibility_score is always recomputed from them.
        self._alpha = alpha
        self._beta = beta
        # Allow injecting a specific starting credibility by adjusting priors.
        if initial_credibility != 0.5:
            # Solve:  alpha / (alpha + beta) = initial_credibility
            # while keeping alpha + beta = 10 (same precision as default).
            total = alpha + beta
            self._alpha = initial_credibility * total
            self._beta = (1.0 - initial_credibility) * total

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def credibility_score(self) -> float:
        """Posterior mean of the Beta distribution = E[θ] = α/(α+β)."""
        return self._alpha / (self._alpha + self._beta)

    @property
    def credibility_uncertainty(self) -> float:
        """
        Posterior variance of the Beta(α,β).
        High early on, shrinks as evidence accumulates.
        """
        a, b = self._alpha, self._beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def analyze(
        self,
        hypothesis: Hypothesis,
        market_data: dict[str, Any],
    ) -> AnalystVerdict:
        """
        Evaluate a hypothesis and return a structured verdict.

        Parameters
        ----------
        hypothesis  : The Hypothesis object to evaluate.
        market_data : Dict with OHLCV arrays, pre-computed statistics,
                      p-values, effect sizes, sub-period breakdowns, etc.
                      Keys vary by analyst but include at minimum:
                        'n_samples', 'p_value', 'effect_size',
                        'sub_period_results', 'regime_breakdown'
        """
        ...

    # ------------------------------------------------------------------
    # Credibility update
    # ------------------------------------------------------------------

    def update_credibility(self, was_right: bool) -> None:
        """
        Bayesian update of credibility score after a prediction is resolved.

        Called by track_record.py once a backtested hypothesis result is known.
        Correct vote (was_right=True)  → increment alpha (success count).
        Wrong vote   (was_right=False) → increment beta  (failure count).

        This is the conjugate Beta-Bernoulli update:
            posterior = Beta(alpha + correct, beta + (1 - correct))
        """
        if was_right:
            self._alpha += 1.0
        else:
            self._beta += 1.0

    # ------------------------------------------------------------------
    # Utility helpers for subclasses
    # ------------------------------------------------------------------

    def _make_verdict(
        self,
        vote: Vote,
        confidence: float,
        reasoning: str,
        key_concerns: list[str] | None = None,
    ) -> AnalystVerdict:
        """Convenience factory so subclasses don't repeat boilerplate."""
        return AnalystVerdict(
            agent_name=self.name,
            vote=vote,
            confidence=max(0.0, min(1.0, confidence)),
            reasoning=reasoning,
            key_concerns=key_concerns or [],
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"credibility={self.credibility_score:.3f}, "
            f"α={self._alpha:.1f}, β={self._beta:.1f})"
        )
