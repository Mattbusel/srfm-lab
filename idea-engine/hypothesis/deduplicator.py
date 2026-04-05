"""
hypothesis/deduplicator.py

Deduplicator: blocks hypotheses that are too similar to existing ones.

Algorithm:
    1. Encode each hypothesis's parameters as a numeric feature vector.
    2. Compute Euclidean distance between the candidate and each existing hypothesis
       of the SAME type.
    3. If min(distances) < DISTANCE_THRESHOLD → duplicate.

Feature extraction rules:
    - Numeric values are used directly (normalised by known ranges).
    - Lists of ints (cluster IDs, days_of_week) → set-Jaccard similarity converted to distance.
    - String values → ignored (non-numeric).
    - Missing features → imputed with 0.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from hypothesis.types import Hypothesis, HypothesisType

DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")

# Per-type distance threshold below which we consider two hypotheses duplicates.
# Tuned conservatively: we prefer to allow near-duplicates than block novel ones.
DISTANCE_THRESHOLDS: dict[str, float] = {
    HypothesisType.ENTRY_TIMING.value:     0.08,
    HypothesisType.EXIT_RULE.value:        0.10,
    HypothesisType.REGIME_FILTER.value:    0.12,
    HypothesisType.CROSS_ASSET.value:      0.10,
    HypothesisType.PARAMETER_TWEAK.value:  0.06,
    HypothesisType.COMPOUND.value:         0.12,
}
DEFAULT_THRESHOLD = 0.10

# Normalisation ranges for known parameters
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "entry_hour_start":     (0, 24),
    "entry_hour_end":       (0, 24),
    "exit_mass_threshold":  (0, 2),
    "max_loss_bars":        (1, 50),
    "atr_stop_mult":        (0.5, 5),
    "required_tf_score_min": (0, 10),
    "lead_mass_threshold":  (0, 2),
    "lag_bars":             (1, 20),
    "bh_form":              (0.05, 2.0),
    "cf":                   (0.0, 1.0),
    "stale_threshold":      (1, 50),
    "atr_period":           (5, 100),
    "vol_lookback":         (10, 200),
    "mass_decay":           (0.01, 0.99),
    "anomaly_zscore_exit":  (0.5, 5),
    "new_value":            (0.0, 10.0),
}


# ---------------------------------------------------------------------------
# Feature encoder
# ---------------------------------------------------------------------------

def _jaccard_distance(a: list, b: list) -> float:
    """Convert Jaccard similarity → distance in [0, 1]."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    return 1.0 - intersection / union


def _encode_params(params: dict[str, Any], hypothesis_type: str) -> dict[str, float]:
    """
    Encode hypothesis parameters into a flat dict of normalised floats.
    Returns feature_name → normalised_value in [0, 1].
    """
    features: dict[str, float] = {}

    for key, val in params.items():
        if key in ("instruments", "filter_mode", "variant", "description",
                   "cluster_labels", "param_description", "exit_trigger",
                   "signal_direction", "session_label", "param_name"):
            continue  # skip non-numeric / meta fields

        if isinstance(val, bool):
            features[key] = 1.0 if val else 0.0

        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            lo, hi = PARAM_RANGES.get(key, (0.0, 1.0))
            span = hi - lo
            if span > 0:
                features[key] = float(max(0.0, min((val - lo) / span, 1.0)))
            else:
                features[key] = 0.0

        elif isinstance(val, list):
            # Encode as list size + mean value if numeric
            if all(isinstance(x, (int, float)) for x in val):
                features[f"{key}__len"] = min(len(val) / 10.0, 1.0)
                if val:
                    lo, hi = PARAM_RANGES.get(key, (0.0, 100.0))
                    span = hi - lo or 1.0
                    features[f"{key}__mean"] = float(
                        max(0.0, min((sum(val) / len(val) - lo) / span, 1.0))
                    )

    return features


def _feature_vector(features: dict[str, float], all_keys: list[str]) -> np.ndarray:
    """Align features dict to a fixed-length vector using all_keys ordering."""
    return np.array([features.get(k, 0.0) for k in all_keys], dtype=np.float64)


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 1.0  # no features → treat as maximally different
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def _jaccard_list_distance(h1: Hypothesis, h2: Hypothesis, list_keys: list[str]) -> float:
    """
    Additional distance contribution from list-valued parameters
    (e.g. blocked_regime_clusters, days_of_week, target_instruments).
    Returns mean Jaccard distance over shared list keys.
    """
    distances = []
    for key in list_keys:
        v1 = h1.parameters.get(key)
        v2 = h2.parameters.get(key)
        if isinstance(v1, list) and isinstance(v2, list):
            distances.append(_jaccard_distance(v1, v2))
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------

class Deduplicator:
    """
    Checks whether a candidate Hypothesis is a near-duplicate of any
    hypothesis already in the store.

    Usage:
        dedup = Deduplicator(existing_hypotheses)
        if not dedup.is_duplicate(new_hypothesis):
            store.insert(new_hypothesis)
    """

    LIST_KEYS = ["blocked_regime_clusters", "days_of_week", "target_instruments"]

    def __init__(self, existing: list[Hypothesis]) -> None:
        self.existing = existing
        # Pre-group by type for efficiency
        self._by_type: dict[str, list[Hypothesis]] = {}
        for h in existing:
            self._by_type.setdefault(h.type.value, []).append(h)

    @classmethod
    def from_store(cls, store: Any) -> "Deduplicator":
        """Construct from a HypothesisStore instance."""
        return cls(store.get_all())

    def is_duplicate(self, candidate: Hypothesis, threshold: float | None = None) -> bool:
        """
        Returns True if the candidate is too similar to an existing hypothesis.
        """
        same_type = self._by_type.get(candidate.type.value, [])
        if not same_type:
            return False

        thr = threshold or DISTANCE_THRESHOLDS.get(candidate.type.value, DEFAULT_THRESHOLD)

        candidate_features = _encode_params(candidate.parameters, candidate.type.value)
        all_keys = sorted(set(candidate_features.keys()))

        if not all_keys:
            # No numeric features: use instrument overlap as proxy
            for existing in same_type:
                inst_dist = _jaccard_distance(
                    candidate.parameters.get("instruments", []),
                    existing.parameters.get("instruments", []),
                )
                if inst_dist < 0.1:
                    return True
            return False

        cand_vec = _feature_vector(candidate_features, all_keys)

        min_dist = math.inf
        for existing in same_type:
            ex_features = _encode_params(existing.parameters, existing.type.value)
            ex_vec = _feature_vector(ex_features, all_keys)
            dist = _euclidean_distance(cand_vec, ex_vec)

            # Blend in Jaccard distance for list-valued params
            jaccard = _jaccard_list_distance(candidate, existing, self.LIST_KEYS)
            blended = 0.7 * dist + 0.3 * jaccard

            if blended < min_dist:
                min_dist = blended

        return min_dist < thr

    def find_nearest(self, candidate: Hypothesis) -> tuple[Hypothesis | None, float]:
        """
        Return the most similar existing hypothesis and its blended distance.
        Useful for debugging / reporting.
        """
        same_type = self._by_type.get(candidate.type.value, [])
        if not same_type:
            return None, math.inf

        candidate_features = _encode_params(candidate.parameters, candidate.type.value)
        all_keys = sorted(set(candidate_features.keys()))
        cand_vec = _feature_vector(candidate_features, all_keys)

        best_h: Hypothesis | None = None
        best_dist = math.inf

        for existing in same_type:
            ex_features = _encode_params(existing.parameters, existing.type.value)
            ex_vec = _feature_vector(ex_features, all_keys)
            dist = _euclidean_distance(cand_vec, ex_vec)
            jaccard = _jaccard_list_distance(candidate, existing, self.LIST_KEYS)
            blended = 0.7 * dist + 0.3 * jaccard
            if blended < best_dist:
                best_dist = blended
                best_h = existing

        return best_h, best_dist

    def filter_duplicates(
        self, candidates: list[Hypothesis]
    ) -> tuple[list[Hypothesis], list[Hypothesis]]:
        """
        Split candidates into (novel, duplicate) lists.
        Novel candidates are added to the internal index as they're accepted,
        so each call is incremental (no intra-batch duplicates).
        """
        novel: list[Hypothesis] = []
        duplicates: list[Hypothesis] = []

        for h in candidates:
            if self.is_duplicate(h):
                duplicates.append(h)
            else:
                novel.append(h)
                # Add to index so subsequent candidates in this batch are checked against it
                self._by_type.setdefault(h.type.value, []).append(h)

        return novel, duplicates
