"""
Meta-Cognitive Agent Competency Tracker.

Moves beyond binary Right/Wrong accuracy tracking to multi-dimensional
competency mapping. Each debate agent has a per-regime, per-template
accuracy profile, enabling context-sensitive vote weighting.

Instead of one Beta(alpha, beta) per agent, maintains a CompetencyMap:
  agent -> {(regime, template_type) -> Beta(alpha, beta)}

When the debate system needs to weight an agent's vote, it looks up
the agent's specific credibility for the current hypothesis's context.
"""

from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BetaPosterior:
    """Beta distribution posterior for a single competency dimension."""
    alpha: float = 1.0   # prior successes + 1
    beta: float = 1.0    # prior failures + 1

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def count(self) -> int:
        return int(self.alpha + self.beta - 2)

    @property
    def confidence(self) -> float:
        """How confident are we in this estimate? Based on sample size."""
        n = self.count
        if n < 5:
            return 0.1
        elif n < 20:
            return 0.5
        else:
            return min(0.9, 0.5 + 0.4 * math.log(n / 20 + 1))

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    def credibility_interval(self, width: float = 0.9) -> Tuple[float, float]:
        """Approximate credibility interval using normal approximation."""
        n = self.alpha + self.beta
        mean = self.mean
        std = math.sqrt(self.alpha * self.beta / (n * n * (n + 1)))
        z = 1.645 if width == 0.9 else 1.96
        return (max(0, mean - z * std), min(1, mean + z * std))


@dataclass
class ErrorRecord:
    """Record of a specific prediction error."""
    agent_name: str
    regime: str
    template_type: str
    error_type: str   # REGIME_MISMATCH / VOL_OVERESTIMATE / DIRECTION_WRONG / TIMING_ERROR
    predicted: float
    actual: float
    timestamp: float = 0.0


ERROR_TYPES = [
    "REGIME_MISMATCH",        # agent predicted wrong regime applicability
    "VOL_OVERESTIMATE",       # agent overestimated opportunity (too bullish)
    "VOL_UNDERESTIMATE",      # agent underestimated risk
    "DIRECTION_WRONG",        # agent predicted wrong trade direction
    "TIMING_ERROR",           # right idea, wrong timing (too early/late)
    "COST_UNDERESTIMATE",     # agent underestimated transaction costs
]


class CompetencyTracker:
    """
    Multi-dimensional competency tracking for debate agents.

    For each agent, tracks accuracy across:
      - Market regimes (trending_bull, trending_bear, mean_reverting, high_vol, crisis)
      - Template types (momentum, mean_reversion, breakout, etc.)
      - Error types (regime mismatch, direction wrong, etc.)

    The debate system uses this to contextually weight votes:
    if an agent is historically bad at crisis-regime hypotheses,
    their vote weight is suppressed for that specific debate.
    """

    def __init__(self, agent_names: List[str]):
        self.agents = agent_names

        # Per-agent, per-context Beta distributions
        # Key: (agent_name, context_type, context_value)
        self._competency: Dict[Tuple[str, str, str], BetaPosterior] = {}

        # Global accuracy (fallback)
        self._global: Dict[str, BetaPosterior] = {
            name: BetaPosterior() for name in agent_names
        }

        # Error log
        self._errors: List[ErrorRecord] = []

    def _get_posterior(self, agent: str, context_type: str, context_value: str) -> BetaPosterior:
        key = (agent, context_type, context_value)
        if key not in self._competency:
            self._competency[key] = BetaPosterior()
        return self._competency[key]

    def record_outcome(
        self,
        agent_name: str,
        was_correct: bool,
        regime: str,
        template_type: str,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record the outcome of an agent's prediction.

        Updates:
          - Global accuracy
          - Regime-specific accuracy
          - Template-specific accuracy
          - Regime x template accuracy (most specific)
        """
        # Global
        if agent_name in self._global:
            self._global[agent_name].update(was_correct)

        # Regime-specific
        self._get_posterior(agent_name, "regime", regime).update(was_correct)

        # Template-specific
        self._get_posterior(agent_name, "template", template_type).update(was_correct)

        # Regime x template (most granular)
        self._get_posterior(agent_name, "regime_template", f"{regime}:{template_type}").update(was_correct)

        # Error type tracking (only on failures)
        if not was_correct and error_type:
            self._get_posterior(agent_name, "error_type", error_type).update(True)  # counts error frequency
            self._errors.append(ErrorRecord(
                agent_name=agent_name,
                regime=regime,
                template_type=template_type,
                error_type=error_type,
                predicted=0.0,
                actual=0.0,
            ))

    def get_contextual_weight(
        self,
        agent_name: str,
        regime: str,
        template_type: str,
    ) -> float:
        """
        Get context-sensitive vote weight for an agent.

        Uses the most specific context available:
          1. If we have regime x template data (>5 observations): use that
          2. Else if regime data (>5 obs): use regime accuracy
          3. Else if template data (>5 obs): use template accuracy
          4. Else: use global accuracy

        Weight = accuracy * confidence (confidence increases with sample size)
        """
        # Most specific: regime x template
        key_rt = (agent_name, "regime_template", f"{regime}:{template_type}")
        if key_rt in self._competency and self._competency[key_rt].count >= 5:
            post = self._competency[key_rt]
            return post.mean * post.confidence

        # Regime-specific
        key_r = (agent_name, "regime", regime)
        if key_r in self._competency and self._competency[key_r].count >= 5:
            post = self._competency[key_r]
            return post.mean * post.confidence

        # Template-specific
        key_t = (agent_name, "template", template_type)
        if key_t in self._competency and self._competency[key_t].count >= 5:
            post = self._competency[key_t]
            return post.mean * post.confidence

        # Global fallback
        if agent_name in self._global:
            post = self._global[agent_name]
            return post.mean * post.confidence

        return 0.5  # neutral weight for unknown agents

    def get_agent_profile(self, agent_name: str) -> Dict:
        """Full competency profile for an agent."""
        profile = {
            "global_accuracy": self._global.get(agent_name, BetaPosterior()).mean,
            "global_count": self._global.get(agent_name, BetaPosterior()).count,
            "regime_accuracy": {},
            "template_accuracy": {},
            "error_distribution": {},
        }

        for (agent, ctx_type, ctx_val), post in self._competency.items():
            if agent != agent_name:
                continue
            if ctx_type == "regime":
                profile["regime_accuracy"][ctx_val] = {"accuracy": post.mean, "count": post.count}
            elif ctx_type == "template":
                profile["template_accuracy"][ctx_val] = {"accuracy": post.mean, "count": post.count}
            elif ctx_type == "error_type":
                profile["error_distribution"][ctx_val] = post.count

        return profile

    def agent_weakness_report(self, agent_name: str) -> List[str]:
        """Identify where this agent consistently underperforms."""
        weaknesses = []
        for (agent, ctx_type, ctx_val), post in self._competency.items():
            if agent != agent_name:
                continue
            if ctx_type in ("regime", "template") and post.count >= 10 and post.mean < 0.4:
                weaknesses.append(
                    f"{ctx_type}={ctx_val}: {post.mean:.0%} accuracy ({post.count} observations)"
                )
        return weaknesses

    def leaderboard(self, regime: Optional[str] = None, template: Optional[str] = None) -> List[Dict]:
        """Rank agents by accuracy for a given context."""
        entries = []
        for name in self.agents:
            weight = self.get_contextual_weight(name, regime or "unknown", template or "unknown")
            entries.append({"agent": name, "weight": weight})
        entries.sort(key=lambda x: x["weight"], reverse=True)
        return entries
