"""
hypothesis/generator.py

HypothesisGenerator: the main entry point for converting a MinedPattern
into a list of Hypothesis objects.

Pipeline per pattern:
    1. Route to one or more templates → raw hypotheses
    2. Score each hypothesis (Bayesian-adjusted impact + testability)
    3. Deduplicate against existing hypotheses in idea_engine.db
    4. Generate compound hypotheses by combining pairs of novel simple ones
    5. Write all novel hypotheses to idea_engine.db
    6. Re-run prioritizer over all pending hypotheses
"""

from __future__ import annotations

import itertools
import logging
import uuid
from pathlib import Path
from typing import Any

from hypothesis.deduplicator import Deduplicator
from hypothesis.hypothesis_store import HypothesisStore
from hypothesis.prioritizer import HypothesisPrioritizer
from hypothesis.scorer import HypothesisScorer
from hypothesis.templates import get_all_applicable_templates
from hypothesis.types import Hypothesis, HypothesisStatus, HypothesisType, MinedPattern

log = logging.getLogger(__name__)

DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")

# Maximum compound hypotheses to generate per pattern (combinatorial explosion guard)
MAX_COMPOUND_PER_PATTERN = 5
# Only attempt compounding if both simple hypotheses have composite_score > this threshold
COMPOUND_SCORE_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Compound hypothesis builder
# ---------------------------------------------------------------------------

def _build_compound(h1: Hypothesis, h2: Hypothesis, pattern: MinedPattern) -> Hypothesis:
    """
    Merge two hypotheses into a COMPOUND hypothesis.
    Parameters are merged (h2 wins on conflict).
    Predicted scores are the average of the two, with a 1.1× synergy bonus capped at 1.5.
    """
    merged_params: dict[str, Any] = {**h1.parameters}
    # Instruments: union
    instr1 = set(h1.parameters.get("instruments", []))
    instr2 = set(h2.parameters.get("instruments", []))
    merged_params["instruments"] = sorted(instr1 | instr2)

    for k, v in h2.parameters.items():
        if k == "instruments":
            continue
        if k in merged_params and merged_params[k] != v:
            # Prefix with component types to avoid silent clobber
            merged_params[f"{h2.type.value}__{k}"] = v
        else:
            merged_params[k] = v

    merged_params["compound_components"] = [h1.hypothesis_id, h2.hypothesis_id]
    merged_params["component_types"] = [h1.type.value, h2.type.value]

    synergy = 1.1
    sharpe_delta = min(
        (h1.predicted_sharpe_delta + h2.predicted_sharpe_delta) / 2 * synergy,
        1.5,
    )
    dd_delta = min(
        (h1.predicted_dd_delta + h2.predicted_dd_delta) / 2 * synergy,
        0.25,
    )
    novelty = min((h1.novelty_score + h2.novelty_score) / 2 * 1.05, 1.0)

    desc = (
        f"Compound: [{h1.type.value}] {h1.description[:80].rstrip('.')} "
        f"+ [{h2.type.value}] {h2.description[:80].rstrip('.')}."
    )

    h = Hypothesis.create(
        hypothesis_type=HypothesisType.COMPOUND,
        parent_pattern_id=pattern.pattern_id,
        parameters=merged_params,
        predicted_sharpe_delta=round(sharpe_delta, 4),
        predicted_dd_delta=round(dd_delta, 4),
        novelty_score=round(novelty, 4),
        description=desc,
    )
    h.compound_child_ids = [h1.hypothesis_id, h2.hypothesis_id]
    return h


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class HypothesisGenerator:
    """
    Main pipeline orchestrator.

    Parameters:
        db_path: path to idea_engine.db
        enable_compounding: if True, generate compound hypotheses
        max_compound: max compound hypotheses per call
        score_before_dedup: if True, apply scorer before dedup (recommended)
        auto_prioritize: if True, re-run prioritizer after inserting
    """

    def __init__(
        self,
        db_path: Path | str = DB_PATH,
        enable_compounding: bool = True,
        max_compound: int = MAX_COMPOUND_PER_PATTERN,
        score_before_dedup: bool = True,
        auto_prioritize: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.enable_compounding = enable_compounding
        self.max_compound = max_compound
        self.score_before_dedup = score_before_dedup
        self.auto_prioritize = auto_prioritize

        self.store = HypothesisStore(self.db_path)
        self.scorer = HypothesisScorer(self.db_path)
        self.prioritizer = HypothesisPrioritizer(self.store)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        """
        Full pipeline: template → score → dedup → compound → persist → prioritize.
        Returns the list of novel hypotheses actually written to the DB.
        """
        log.info(
            "Generating hypotheses for pattern %s (type=%s)",
            pattern.pattern_id[:8],
            pattern.pattern_type,
        )

        # Step 1: Route to templates
        raw = self._run_templates(pattern)
        if not raw:
            log.info("No hypotheses generated for pattern %s", pattern.pattern_id[:8])
            return []

        log.debug("Templates produced %d raw hypotheses", len(raw))

        # Step 2: Score
        if self.score_before_dedup:
            raw = [self.scorer.score_and_apply(h, pattern) for h in raw]

        # Step 3: Load existing + deduplicate
        existing = self.store.get_all()
        dedup = Deduplicator(existing)
        novel, dupes = dedup.filter_duplicates(raw)
        log.debug("Deduplication: %d novel, %d duplicates", len(novel), len(dupes))

        # Step 4: Compound hypotheses
        compound: list[Hypothesis] = []
        if self.enable_compounding and len(novel) >= 2:
            compound = self._generate_compounds(novel, pattern, dedup)
            log.debug("Generated %d compound hypotheses", len(compound))

        all_novel = novel + compound

        # Step 5: Persist
        inserted = self.store.insert_many(all_novel)
        log.info(
            "Inserted %d/%d hypotheses for pattern %s",
            inserted, len(all_novel), pattern.pattern_id[:8],
        )

        # Step 6: Re-prioritize all pending
        if self.auto_prioritize and all_novel:
            self.prioritizer.rank_pending()

        return all_novel

    def generate_many(self, patterns: list[MinedPattern]) -> list[Hypothesis]:
        """Process a batch of MinedPatterns; returns all novel hypotheses."""
        all_novel: list[Hypothesis] = []
        for pattern in patterns:
            all_novel.extend(self.generate(pattern))
        return all_novel

    # ------------------------------------------------------------------
    # Template routing
    # ------------------------------------------------------------------

    def _run_templates(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Run all applicable templates and collect results."""
        templates = get_all_applicable_templates(pattern)
        if not templates:
            log.warning(
                "No templates found for pattern type '%s'", pattern.pattern_type
            )
            return []

        results: list[Hypothesis] = []
        for tmpl in templates:
            try:
                generated = tmpl.generate(pattern)
                results.extend(generated)
            except Exception as exc:
                log.error(
                    "Template %s failed for pattern %s: %s",
                    tmpl.__class__.__name__,
                    pattern.pattern_id[:8],
                    exc,
                    exc_info=True,
                )
        return results

    # ------------------------------------------------------------------
    # Compound generation
    # ------------------------------------------------------------------

    def _generate_compounds(
        self,
        novel: list[Hypothesis],
        pattern: MinedPattern,
        dedup: Deduplicator,
    ) -> list[Hypothesis]:
        """
        Attempt to combine pairs of novel hypotheses into compound hypotheses.
        Only combines hypotheses of *different* types.
        Limits total compounds to self.max_compound.
        """
        compounds: list[Hypothesis] = []
        # Sort by novelty * predicted_sharpe to pick best pairs first
        scored = sorted(
            novel,
            key=lambda h: h.novelty_score * max(h.predicted_sharpe_delta, 0),
            reverse=True,
        )
        # Only use top-K candidates to avoid O(n^2) explosion
        candidates = scored[:min(len(scored), 8)]

        for h1, h2 in itertools.combinations(candidates, 2):
            if len(compounds) >= self.max_compound:
                break

            # Only combine different hypothesis types for maximal orthogonality
            if h1.type == h2.type:
                continue

            # Skip if either has low predicted benefit
            if (
                h1.predicted_sharpe_delta < COMPOUND_SCORE_THRESHOLD
                and h2.predicted_sharpe_delta < COMPOUND_SCORE_THRESHOLD
            ):
                continue

            compound = _build_compound(h1, h2, pattern)

            # Score the compound
            compound = self.scorer.score_and_apply(compound, pattern)

            # Dedup check
            if dedup.is_duplicate(compound):
                continue

            compounds.append(compound)
            # Add to dedup index incrementally
            dedup._by_type.setdefault(compound.type.value, []).append(compound)

        return compounds

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_pending_count(self) -> int:
        return len(self.store.get_pending())

    def get_stats(self) -> dict[str, Any]:
        counts = self.store.count_by_status()
        priority_summary = self.prioritizer.summary()
        return {
            "status_counts": counts,
            "priority_summary": priority_summary,
        }
