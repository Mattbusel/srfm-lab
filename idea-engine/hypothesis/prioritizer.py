"""
hypothesis/prioritizer.py

HypothesisPrioritizer: assigns priority_rank to all pending hypotheses.

Priority formula:
    priority_score = impact_score * novelty_score * testability_score

    impact_score     = f(predicted_sharpe_delta, predicted_dd_delta)
    novelty_score    = stored on hypothesis (set by template + scorer)
    testability_score = 1.0 if all params map to existing backtest params, 0.5 otherwise

priority_rank is the ordinal rank (1 = highest priority).
Ranks are updated in idea_engine.db for all PENDING hypotheses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hypothesis.hypothesis_store import HypothesisStore
from hypothesis.scorer import KNOWN_BACKTEST_PARAMS
from hypothesis.types import Hypothesis, HypothesisStatus, HypothesisType

DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")

# Weights for the three scoring axes
IMPACT_WEIGHT = 0.50
NOVELTY_WEIGHT = 0.25
TESTABILITY_WEIGHT = 0.25

# Parameters that need new code (not directly testable via existing backtest params)
NEEDS_NEW_CODE_PARAMS = {
    "anomaly_zscore_exit", "lead_mass_threshold", "lag_bars",
    "blocked_regime_clusters", "required_tf_score_min",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class RankedHypothesis:
    hypothesis: Hypothesis
    impact_score: float
    testability_score: float
    novelty_score: float
    composite_score: float
    rank: int


def _impact_score(h: Hypothesis) -> float:
    """
    Convert predicted_sharpe_delta and predicted_dd_delta to a 0-1 impact score.
    sharpe_delta > 0 is good, dd_delta > 0 (less drawdown) is good.
    """
    sharpe_contrib = math.tanh(max(h.predicted_sharpe_delta, 0.0) * 2.0) * 0.7
    dd_contrib = math.tanh(max(h.predicted_dd_delta, 0.0) * 5.0) * 0.3
    return float(min(sharpe_contrib + dd_contrib, 1.0))


def _testability_score(h: Hypothesis) -> float:
    """
    1.0  — all non-meta params are directly in existing backtest parameter set
    0.75 — most params known, a few need minor code changes
    0.5  — significant new code needed
    """
    params = h.parameters
    meta_keys = {
        "instruments", "filter_mode", "variant", "description",
        "cluster_labels", "param_description", "exit_trigger",
        "signal_direction", "trail_from_entry", "require_loss",
        "session_label", "target_instruments", "lead_feature",
        "current_value", "direction",
    }
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


def _bonus_modifiers(h: Hypothesis) -> float:
    """
    Small bonus/penalty modifiers based on hypothesis metadata:
    - COMPOUND hypotheses get a slight bonus (harder to discover)
    - Very low p_value patterns get a small bonus (encoded in novelty)
    - Entry/exit hypotheses are slightly preferred over parameter tweaks
      because they're orthogonal to each other (different parts of the system)
    """
    bonus = 0.0
    if h.type == HypothesisType.COMPOUND:
        bonus += 0.05
    if h.type == HypothesisType.PARAMETER_TWEAK:
        bonus -= 0.02
    return bonus


def _composite_score(h: Hypothesis) -> float:
    impact = _impact_score(h)
    testability = _testability_score(h)
    novelty = max(min(h.novelty_score, 1.0), 0.0)
    base = (
        IMPACT_WEIGHT * impact
        + NOVELTY_WEIGHT * novelty
        + TESTABILITY_WEIGHT * testability
    )
    return float(min(max(base + _bonus_modifiers(h), 0.0), 1.0))


# ---------------------------------------------------------------------------
# Prioritizer
# ---------------------------------------------------------------------------

class HypothesisPrioritizer:
    """
    Computes and writes priority_rank for all PENDING hypotheses in the store.

    Usage:
        prioritizer = HypothesisPrioritizer(store)
        ranked = prioritizer.rank_pending()
        # priority_rank is now written to idea_engine.db
    """

    def __init__(self, store: HypothesisStore) -> None:
        self.store = store

    def rank_pending(self) -> list[RankedHypothesis]:
        """
        Load all pending hypotheses, compute scores, assign ranks, persist.
        Returns list of RankedHypothesis sorted by rank ascending (1 = best).
        """
        pending = self.store.get_pending()
        if not pending:
            return []

        return self._rank_and_persist(pending)

    def rank_all(self) -> list[RankedHypothesis]:
        """
        Rank all hypotheses regardless of status.
        Useful for re-prioritisation after bulk imports.
        """
        all_hypotheses = self.store.get_all()
        if not all_hypotheses:
            return []
        return self._rank_and_persist(all_hypotheses)

    def rank_subset(self, hypotheses: list[Hypothesis]) -> list[RankedHypothesis]:
        """
        Rank a given list in-memory without touching the DB.
        Useful for previewing before committing.
        """
        return self._rank_only(hypotheses)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rank_only(self, hypotheses: list[Hypothesis]) -> list[RankedHypothesis]:
        """Score and sort; do not write to DB."""
        scored: list[tuple[float, Hypothesis]] = []
        for h in hypotheses:
            score = _composite_score(h)
            scored.append((score, h))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        ranked: list[RankedHypothesis] = []
        for rank, (score, h) in enumerate(scored, start=1):
            ranked.append(
                RankedHypothesis(
                    hypothesis=h,
                    impact_score=round(_impact_score(h), 4),
                    testability_score=round(_testability_score(h), 4),
                    novelty_score=round(h.novelty_score, 4),
                    composite_score=round(score, 4),
                    rank=rank,
                )
            )
        return ranked

    def _rank_and_persist(self, hypotheses: list[Hypothesis]) -> list[RankedHypothesis]:
        ranked = self._rank_only(hypotheses)

        # Build bulk update list
        updates: list[tuple[str, int]] = [
            (rh.hypothesis.hypothesis_id, rh.rank) for rh in ranked
        ]
        self.store.bulk_update_ranks(updates)

        return ranked

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def top_n(self, n: int = 10) -> list[RankedHypothesis]:
        """Return top-N ranked pending hypotheses."""
        ranked = self.rank_pending()
        return ranked[:n]

    def summary(self) -> dict[str, Any]:
        """Return a brief summary of the current priority landscape."""
        ranked = self.rank_pending()
        if not ranked:
            return {"count": 0, "top_type": None, "avg_composite": 0.0}

        avg_score = sum(r.composite_score for r in ranked) / len(ranked)
        top = ranked[0]
        type_counts: dict[str, int] = {}
        for r in ranked:
            t = r.hypothesis.type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "count": len(ranked),
            "avg_composite_score": round(avg_score, 4),
            "top_hypothesis_id": top.hypothesis.hypothesis_id,
            "top_composite_score": top.composite_score,
            "top_type": top.hypothesis.type.value,
            "top_description": top.hypothesis.description[:120],
            "type_distribution": type_counts,
        }
