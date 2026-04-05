"""
hypothesis/scorer.py

HypothesisScorer: estimates predicted_sharpe_delta and predicted_dd_delta
using a simple Bayesian prior updated from past hypothesis outcomes stored
in idea_engine.db.

Scoring model:
    impact_score = f(predicted_sharpe_delta, predicted_dd_delta)
    testability_score = f(parameter_space_coverage)
    composite = impact * novelty * testability

The Bayesian component:
    Prior: Beta(alpha=2, beta=2) for "does this hypothesis type improve Sharpe?"
    Update: each validated/rejected past hypothesis of the same type updates (alpha, beta).
    Posterior mean = alpha / (alpha + beta) = base_rate of success.
    This modulates our confidence in the predicted_sharpe_delta.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hypothesis.types import (
    Hypothesis,
    HypothesisStatus,
    HypothesisType,
    MinedPattern,
    ScoredHypothesis,
)

DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")

# Known backtest parameter names — if all hypothesis params map to these,
# testability = 1.0, otherwise 0.5 (needs new code).
KNOWN_BACKTEST_PARAMS = {
    "entry_hour_start", "entry_hour_end", "days_of_week",
    "exit_mass_threshold", "max_loss_bars", "atr_stop_mult",
    "blocked_regime_clusters", "required_tf_score_min",
    "lead_instrument", "lead_mass_threshold", "lag_bars",
    "bh_form", "cf", "stale_threshold", "atr_period",
    "vol_lookback", "mass_decay",
    "anomaly_zscore_exit",
}


# ---------------------------------------------------------------------------
# Beta distribution helpers
# ---------------------------------------------------------------------------

@dataclass
class BetaPrior:
    alpha: float = 2.0
    beta: float = 2.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def update(self, successes: int, failures: int) -> "BetaPrior":
        return BetaPrior(
            alpha=self.alpha + successes,
            beta=self.beta + failures,
        )

    def credible_interval(self, level: float = 0.9) -> tuple[float, float]:
        """Approximate HPD interval using normal approximation."""
        from scipy.stats import beta as beta_dist
        half_alpha = (1 - level) / 2
        lo = float(beta_dist.ppf(half_alpha, self.alpha, self.beta))
        hi = float(beta_dist.ppf(1 - half_alpha, self.alpha, self.beta))
        return lo, hi


# ---------------------------------------------------------------------------
# History loader
# ---------------------------------------------------------------------------

class OutcomeHistory:
    """
    Loads historical hypothesis outcomes from idea_engine.db.
    Used to update the Bayesian prior per hypothesis type.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._cache: dict[str, tuple[int, int]] | None = None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def get_outcomes(self) -> dict[str, tuple[int, int]]:
        """
        Returns dict: hypothesis_type_str → (successes, failures)
        successes = validated count, failures = rejected count
        """
        if self._cache is not None:
            return self._cache
        try:
            sql = """
                SELECT type,
                       SUM(CASE WHEN status = 'validated' THEN 1 ELSE 0 END) AS successes,
                       SUM(CASE WHEN status = 'rejected'  THEN 1 ELSE 0 END) AS failures
                FROM hypotheses
                GROUP BY type
            """
            with self._connect() as conn:
                rows = conn.execute(sql).fetchall()
            result = {}
            for r in rows:
                result[r["type"]] = (int(r["successes"]), int(r["failures"]))
            self._cache = result
            return result
        except Exception:
            return {}

    def get_prior_for_type(self, hypothesis_type: HypothesisType) -> BetaPrior:
        outcomes = self.get_outcomes()
        successes, failures = outcomes.get(hypothesis_type.value, (0, 0))
        prior = BetaPrior()
        return prior.update(successes, failures)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class HypothesisScorer:
    """
    Scores a Hypothesis against a MinedPattern.

    Returns a ScoredHypothesis with:
        impact_score     — Bayesian-adjusted estimate of Sharpe improvement
        testability_score — 1.0 if params map to known backtest params, 0.5 otherwise
        novelty_score    — passed through from the hypothesis (set by template)
        composite_priority — impact * novelty * testability
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.history = OutcomeHistory(db_path)

    def score(self, hypothesis: Hypothesis, pattern: MinedPattern) -> ScoredHypothesis:
        """Compute all scores and return a ScoredHypothesis."""
        impact = self._compute_impact(hypothesis, pattern)
        testability = self._compute_testability(hypothesis)
        novelty = hypothesis.novelty_score
        composite = impact * novelty * testability

        return ScoredHypothesis(
            hypothesis=hypothesis,
            impact_score=impact,
            testability_score=testability,
            novelty_score=novelty,
            composite_priority=composite,
        )

    # ------------------------------------------------------------------
    # Impact score
    # ------------------------------------------------------------------

    def _compute_impact(self, hypothesis: Hypothesis, pattern: MinedPattern) -> float:
        """
        Bayesian-modulated impact score.

        1. Start with raw predicted_sharpe_delta from template.
        2. Get posterior mean (base_rate) for this hypothesis type.
        3. Blend: impact = raw_delta * base_rate_confidence * p_value_weight
        4. Add drawdown component: dd_benefit = max(predicted_dd_delta, 0) * 0.5
        5. Normalise to [0, 1].
        """
        prior = self.history.get_prior_for_type(hypothesis.type)
        base_rate = prior.mean  # posterior probability this type succeeds

        raw_sharpe = hypothesis.predicted_sharpe_delta
        raw_dd = hypothesis.predicted_dd_delta

        # p_value weight: lower p → more confidence
        p = max(min(pattern.p_value, 1.0 - 1e-10), 1e-10)
        p_weight = min(-math.log10(p) / 5.0, 1.0)  # p=0.05→0.26, p=0.001→0.6

        # CI width penalty
        ci_width = max(pattern.ci_upper - pattern.ci_lower, 1e-6)
        ci_factor = 1.0 / (1.0 + ci_width * 0.3)

        # Blended sharpe component
        sharpe_component = max(raw_sharpe, 0.0) * base_rate * p_weight * ci_factor

        # Drawdown bonus (cutting drawdown is always good)
        dd_component = max(raw_dd, 0.0) * 0.5

        # Combined
        raw_impact = sharpe_component + dd_component

        # Normalise: cap at 1.0, ensure 0 lower bound
        impact = float(min(max(raw_impact, 0.0), 1.0))
        return impact

    # ------------------------------------------------------------------
    # Testability score
    # ------------------------------------------------------------------

    def _compute_testability(self, hypothesis: Hypothesis) -> float:
        """
        1.0 if all non-meta parameters map to known backtest params.
        0.75 if most do.
        0.5 if few do (needs new code to test).
        """
        params = hypothesis.parameters
        # Filter out meta-keys
        meta_keys = {"instruments", "filter_mode", "variant", "description",
                     "cluster_labels", "param_description", "exit_trigger",
                     "signal_direction", "trail_from_entry", "require_loss",
                     "session_label", "target_instruments", "lead_feature",
                     "current_value", "direction"}
        testable_keys = {k for k in params if k not in meta_keys}

        if not testable_keys:
            return 0.5

        known = testable_keys & KNOWN_BACKTEST_PARAMS
        ratio = len(known) / len(testable_keys)

        if ratio >= 0.9:
            return 1.0
        if ratio >= 0.5:
            return 0.75
        return 0.5

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def score_many(
        self,
        hypotheses: list[Hypothesis],
        pattern: MinedPattern,
    ) -> list[ScoredHypothesis]:
        return [self.score(h, pattern) for h in hypotheses]

    def score_and_apply(
        self,
        hypothesis: Hypothesis,
        pattern: MinedPattern,
    ) -> Hypothesis:
        """
        Score the hypothesis and write scores back into the hypothesis object.
        Updates predicted_sharpe_delta with impact-adjusted value and novelty_score.
        """
        scored = self.score(hypothesis, pattern)
        # Adjust stored predicted_sharpe_delta by impact confidence
        prior = self.history.get_prior_for_type(hypothesis.type)
        hypothesis.predicted_sharpe_delta = round(
            hypothesis.predicted_sharpe_delta * prior.mean, 4
        )
        hypothesis.novelty_score = round(scored.novelty_score, 4)
        return hypothesis

    # ------------------------------------------------------------------
    # Utility: describe scoring for a hypothesis
    # ------------------------------------------------------------------

    def explain(self, hypothesis: Hypothesis, pattern: MinedPattern) -> dict[str, Any]:  # noqa: D102
        scored = self.score(hypothesis, pattern)
        prior = self.history.get_prior_for_type(hypothesis.type)
        lo, hi = prior.credible_interval()
        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "type": hypothesis.type.value,
            "impact_score": round(scored.impact_score, 4),
            "testability_score": round(scored.testability_score, 4),
            "novelty_score": round(scored.novelty_score, 4),
            "composite_priority": round(scored.composite_priority, 4),
            "bayesian_base_rate": round(prior.mean, 4),
            "base_rate_ci_90": (round(lo, 4), round(hi, 4)),
            "prior_alpha": prior.alpha,
            "prior_beta": prior.beta,
            "pattern_p_value": pattern.p_value,
            "pattern_effect_size": pattern.effect_size,
        }
