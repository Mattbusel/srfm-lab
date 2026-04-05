"""
hypothesis/templates/regime_filter_template.py

Converts regime_cluster MinedPatterns into regime-filter hypotheses.
Hypothesis: "skip trades when in regime cluster K"
           "require tf_score >= N before entering"
"""

from __future__ import annotations

import math
from typing import Any

from hypothesis.types import Hypothesis, HypothesisType, MinedPattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sharpe_delta_regime(
    effect_size: float,
    p_value: float,
    ci_lower: float,
    ci_upper: float,
) -> float:
    if p_value <= 0:
        p_value = 1e-10
    if p_value >= 1:
        p_value = 1 - 1e-10
    confidence = min(-math.log10(p_value) / 7.0, 1.0)
    ci_width = max(ci_upper - ci_lower, 1e-6)
    ci_penalty = 1.0 / (1.0 + ci_width * 0.4)
    raw = effect_size * confidence * ci_penalty
    return float(max(min(raw, 1.5), -0.5))


def _dd_delta_regime(effect_size: float, regime_context: dict) -> float:
    base = max(effect_size * 0.06, 0.0)
    # Blocking volatile regimes often cuts tail drawdowns more
    if regime_context.get("high_vol", False):
        base += 0.04
    if regime_context.get("adverse_regime", False):
        base += 0.03
    return float(min(base, 0.20))


def _extract_bad_clusters(evidence: dict[str, Any]) -> list[int]:
    """
    Extract regime cluster IDs that showed negative or poor performance.
    Evidence may contain:
        bad_cluster_ids: list[int]
        cluster_sharpe: dict[str, float]   — cluster_id_str → sharpe
        cluster_win_rate: dict[str, float]
    """
    if "bad_cluster_ids" in evidence:
        raw = evidence["bad_cluster_ids"]
        if isinstance(raw, list):
            return [int(x) for x in raw]

    # Derive from cluster_sharpe
    if "cluster_sharpe" in evidence:
        cs: dict = evidence["cluster_sharpe"]
        return [int(k) for k, v in cs.items() if float(v) < 0]

    # Derive from cluster_win_rate < 0.45
    if "cluster_win_rate" in evidence:
        cw: dict = evidence["cluster_win_rate"]
        return [int(k) for k, v in cw.items() if float(v) < 0.45]

    return []


def _extract_tf_score_threshold(evidence: dict[str, Any], regime_context: dict) -> int:
    """
    Derive minimum tf_score required to enter.
    tf_score is an integer score (typically 0–10) summing timeframe alignments.
    """
    for key in ("min_tf_score", "tf_score_threshold", "required_tf_score"):
        if key in evidence:
            return int(evidence[key])
    if "mean_tf_score_good_trades" in evidence:
        return max(int(float(evidence["mean_tf_score_good_trades"]) * 0.75), 1)
    # Regime context hint
    if regime_context.get("strong_trend", False):
        return 6
    if regime_context.get("choppy", False):
        return 8
    return 5


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

class RegimeFilterTemplate:
    """
    Generates regime-filter hypotheses from regime_cluster MinedPatterns.

    Expected evidence fields (any subset):
        bad_cluster_ids: list[int]       — cluster IDs to block
        cluster_sharpe: dict             — cluster_id_str → float
        cluster_win_rate: dict           — cluster_id_str → float
        cluster_labels: dict             — cluster_id_str → human label
        min_tf_score: int                — minimum timeframe alignment score
        mean_tf_score_good_trades: float
    """

    PATTERN_TYPE = "regime_cluster"
    MIN_EFFECT_SIZE = 0.04

    def can_handle(self, pattern: MinedPattern) -> bool:
        return pattern.pattern_type == self.PATTERN_TYPE

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        if not self.can_handle(pattern):
            raise ValueError(
                f"RegimeFilterTemplate cannot handle pattern type '{pattern.pattern_type}'"
            )
        if abs(pattern.effect_size) < self.MIN_EFFECT_SIZE:
            return []

        hypotheses: list[Hypothesis] = []

        bad_clusters = _extract_bad_clusters(pattern.evidence)
        tf_threshold = _extract_tf_score_threshold(pattern.evidence, pattern.regime_context)
        cluster_labels: dict = pattern.evidence.get("cluster_labels", {})

        # 1. Block-bad-clusters hypothesis
        if bad_clusters:
            hypotheses.extend(
                self._blocked_clusters_hypotheses(pattern, bad_clusters, cluster_labels)
            )

        # 2. TF-score filter hypothesis
        hypotheses.extend(
            self._tf_score_hypotheses(pattern, tf_threshold)
        )

        # 3. Combined block + tf_score (compound-lite)
        if bad_clusters:
            hypotheses.append(
                self._combined_hypothesis(pattern, bad_clusters, tf_threshold, cluster_labels)
            )

        return hypotheses

    # ------------------------------------------------------------------
    # Sub-generators
    # ------------------------------------------------------------------

    def _blocked_clusters_hypotheses(
        self,
        pattern: MinedPattern,
        bad_clusters: list[int],
        cluster_labels: dict,
    ) -> list[Hypothesis]:
        results = []
        sharpe_delta = _sharpe_delta_regime(
            pattern.effect_size, pattern.p_value, pattern.ci_lower, pattern.ci_upper
        )
        dd_delta = _dd_delta_regime(pattern.effect_size, pattern.regime_context)

        # Full block of all bad clusters
        label_str = ", ".join(
            cluster_labels.get(str(c), f"cluster-{c}") for c in bad_clusters
        )
        params: dict[str, Any] = {
            "blocked_regime_clusters": bad_clusters,
            "instruments": pattern.instruments,
            "filter_mode": "block_regime",
            "cluster_labels": {str(c): cluster_labels.get(str(c), f"cluster-{c}") for c in bad_clusters},
        }
        desc = (
            f"Skip trades in regime clusters {bad_clusters} ({label_str}). "
            f"Effect size: {pattern.effect_size:.3f}, p={pattern.p_value:.4f}."
        )
        results.append(
            Hypothesis.create(
                hypothesis_type=HypothesisType.REGIME_FILTER,
                parent_pattern_id=pattern.pattern_id,
                parameters=params,
                predicted_sharpe_delta=sharpe_delta,
                predicted_dd_delta=dd_delta,
                novelty_score=self._novelty(pattern),
                description=desc,
            )
        )

        # Also generate per-cluster individual blocks (for granular testing)
        if len(bad_clusters) > 1:
            for c in bad_clusters:
                c_label = cluster_labels.get(str(c), f"cluster-{c}")
                p_single: dict[str, Any] = {
                    "blocked_regime_clusters": [c],
                    "instruments": pattern.instruments,
                    "filter_mode": "block_regime",
                    "cluster_labels": {str(c): c_label},
                }
                d_single = (
                    f"Skip trades specifically in regime cluster {c} ({c_label})."
                )
                results.append(
                    Hypothesis.create(
                        hypothesis_type=HypothesisType.REGIME_FILTER,
                        parent_pattern_id=pattern.pattern_id,
                        parameters=p_single,
                        predicted_sharpe_delta=sharpe_delta * 0.6,
                        predicted_dd_delta=dd_delta * 0.6,
                        novelty_score=self._novelty(pattern) * 0.85,
                        description=d_single,
                    )
                )

        return results

    def _tf_score_hypotheses(
        self,
        pattern: MinedPattern,
        tf_threshold: int,
    ) -> list[Hypothesis]:
        results = []
        sharpe_delta = _sharpe_delta_regime(
            pattern.effect_size, pattern.p_value, pattern.ci_lower, pattern.ci_upper
        )
        dd_delta = _dd_delta_regime(pattern.effect_size, pattern.regime_context)

        for offset, label in [(-1, "loose"), (0, "base"), (1, "strict"), (2, "very_strict")]:
            t = max(tf_threshold + offset, 1)
            params: dict[str, Any] = {
                "required_tf_score_min": t,
                "instruments": pattern.instruments,
                "filter_mode": "tf_score_min",
                "variant": label,
            }
            scale = {"loose": 0.7, "base": 1.0, "strict": 0.9, "very_strict": 0.75}[label]
            desc = (
                f"Only enter when tf_score >= {t} ({label} variant). "
                f"Instruments: {pattern.instruments}."
            )
            results.append(
                Hypothesis.create(
                    hypothesis_type=HypothesisType.REGIME_FILTER,
                    parent_pattern_id=pattern.pattern_id,
                    parameters=params,
                    predicted_sharpe_delta=sharpe_delta * scale,
                    predicted_dd_delta=dd_delta * scale,
                    novelty_score=self._novelty(pattern) * (1.0 if label == "base" else 0.85),
                    description=desc,
                )
            )
        return results

    def _combined_hypothesis(
        self,
        pattern: MinedPattern,
        bad_clusters: list[int],
        tf_threshold: int,
        cluster_labels: dict,
    ) -> Hypothesis:
        sharpe_delta = _sharpe_delta_regime(
            pattern.effect_size * 1.1,  # combo often better than either alone
            pattern.p_value,
            pattern.ci_lower,
            pattern.ci_upper,
        )
        dd_delta = _dd_delta_regime(pattern.effect_size * 1.1, pattern.regime_context)

        params: dict[str, Any] = {
            "blocked_regime_clusters": bad_clusters,
            "required_tf_score_min": tf_threshold,
            "instruments": pattern.instruments,
            "filter_mode": "combined_regime_tf",
            "cluster_labels": {str(c): cluster_labels.get(str(c), f"cluster-{c}") for c in bad_clusters},
        }
        desc = (
            f"Combined filter: block clusters {bad_clusters} AND require tf_score >= {tf_threshold}. "
            f"Instruments: {pattern.instruments}."
        )
        return Hypothesis.create(
            hypothesis_type=HypothesisType.REGIME_FILTER,
            parent_pattern_id=pattern.pattern_id,
            parameters=params,
            predicted_sharpe_delta=sharpe_delta,
            predicted_dd_delta=dd_delta,
            novelty_score=self._novelty(pattern) * 1.05,
            description=desc,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _novelty(pattern: MinedPattern) -> float:
        base = 0.55
        if abs(pattern.effect_size) > 0.6:
            base += 0.1
        if pattern.regime_context:
            base += min(len(pattern.regime_context) * 0.04, 0.2)
        return min(base, 1.0)
