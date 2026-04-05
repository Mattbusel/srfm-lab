"""
causal/python/hypothesis_converter.py

CausalHypothesisConverter: converts causal DAG edges into Hypothesis objects.

Logic:
    - Iterate over significant edges where the TARGET (effect) is a trade outcome.
    - Trade outcome features: trade_pnl, trade_win, (any feature matching OUTCOME_PATTERNS)
    - For each qualifying edge X → Y (outcome):
        * Generate a CROSS_ASSET or ENTRY_TIMING hypothesis:
          "Feature X Granger-causes outcome Y with lag L →
           use X as a leading indicator for Y"
    - Only emit hypotheses where effect_size > MIN_EFFECT_SIZE and lag <= MAX_LAG.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from causal.python.granger.granger_graph import CausalGraph
from causal.python.granger.granger_tests import GrangerEdge, GrangerResult
from hypothesis.types import Hypothesis, HypothesisType, MinedPattern

log = logging.getLogger(__name__)

# Features whose name matches one of these patterns are considered trade outcomes
OUTCOME_PATTERNS = {
    "trade_pnl", "trade_win", "pnl", "win", "loss",
    "return", "profit", "drawdown",
}

# Minimum effect size to generate a hypothesis from a causal edge
MIN_EFFECT_SIZE = 0.02
MAX_LAG = 10

# Mapping from causal feature name → hypothesis type
FEATURE_TYPE_MAP: dict[str, HypothesisType] = {
    "hour_of_day":          HypothesisType.ENTRY_TIMING,
    "day_of_week":          HypothesisType.ENTRY_TIMING,
    "bh_mass_d":            HypothesisType.CROSS_ASSET,
    "bh_mass_h":            HypothesisType.CROSS_ASSET,
    "bh_mass_15m":          HypothesisType.CROSS_ASSET,
    "btc_dominance_proxy":  HypothesisType.CROSS_ASSET,
    "cross_asset_momentum": HypothesisType.CROSS_ASSET,
    "tf_score":             HypothesisType.REGIME_FILTER,
    "atr":                  HypothesisType.PARAMETER_TWEAK,
    "garch_vol":            HypothesisType.PARAMETER_TWEAK,
    "ou_zscore":            HypothesisType.ENTRY_TIMING,
}

DEFAULT_HYPOTHESIS_TYPE = HypothesisType.CROSS_ASSET


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_outcome(feature: str) -> bool:
    """Return True if this feature represents a trade outcome."""
    f_lower = feature.lower()
    return any(pat in f_lower for pat in OUTCOME_PATTERNS)


def _sharpe_delta_from_edge(edge: GrangerEdge) -> float:
    """Estimate Sharpe delta from Granger edge effect size and p-value."""
    p = max(min(edge.p_value, 1 - 1e-10), 1e-10)
    confidence = min(-math.log10(p) / 6.0, 1.0)
    lag_penalty = 1.0 / (1.0 + (edge.optimal_lag - 1) * 0.08)
    raw = edge.effect_size * confidence * lag_penalty
    return float(max(min(raw, 1.5), 0.0))


def _dd_delta_from_edge(edge: GrangerEdge) -> float:
    return float(min(max(edge.effect_size * 0.06, 0.0), 0.15))


def _novelty(edge: GrangerEdge) -> float:
    """Causal edges are inherently more informative than correlations."""
    base = 0.65
    if edge.effect_size > 0.5:
        base += 0.1
    if edge.optimal_lag <= 2:
        base += 0.05  # short-lag causal relationships are more actionable
    return min(base, 0.95)


def _infer_hypothesis_type(cause_feature: str) -> HypothesisType:
    return FEATURE_TYPE_MAP.get(cause_feature, DEFAULT_HYPOTHESIS_TYPE)


def _build_parameters(
    edge: GrangerEdge,
    instruments: list[str],
) -> dict[str, Any]:
    """Build the parameters dict for the hypothesis."""
    params: dict[str, Any] = {
        "causal_feature": edge.cause,
        "causal_effect_target": edge.effect,
        "lag_bars": edge.optimal_lag,
        "granger_p_value": round(edge.p_value, 6),
        "granger_effect_size": round(edge.effect_size, 6),
        "instruments": instruments,
    }

    # Add type-specific parameters
    hyp_type = _infer_hypothesis_type(edge.cause)

    if hyp_type == HypothesisType.CROSS_ASSET:
        params["lead_instrument"] = instruments[0] if instruments else "unknown"
        params["lead_feature"] = edge.cause
        params["lead_mass_threshold"] = 0.5  # default; will be refined in testing
        params["signal_direction"] = "above"

    elif hyp_type == HypothesisType.ENTRY_TIMING:
        if "hour" in edge.cause:
            params["entry_hour_start"] = 0
            params["entry_hour_end"] = 24
            params["days_of_week"] = list(range(7))
        elif "day" in edge.cause:
            params["days_of_week"] = list(range(7))
            params["entry_hour_start"] = 0
            params["entry_hour_end"] = 24

    elif hyp_type == HypothesisType.REGIME_FILTER:
        params["required_tf_score_min"] = 5
        params["blocked_regime_clusters"] = []

    elif hyp_type == HypothesisType.PARAMETER_TWEAK:
        if "atr" in edge.cause:
            params["param_name"] = "atr_period"
        elif "vol" in edge.cause or "garch" in edge.cause:
            params["param_name"] = "vol_lookback"
        else:
            params["param_name"] = edge.cause
        params["new_value"] = 0.0  # to be calibrated in testing

    return params


def _build_synthetic_pattern(edge: GrangerEdge, instruments: list[str]) -> MinedPattern:
    """
    Create a synthetic MinedPattern representing the causal edge.
    Used as the parent_pattern for hypotheses generated from DAG edges.
    """
    return MinedPattern.create(
        pattern_type="cross_asset",
        instruments=instruments,
        p_value=edge.p_value,
        effect_size=edge.effect_size,
        ci_lower=max(edge.effect_size - 0.1, 0.0),
        ci_upper=edge.effect_size + 0.1,
        evidence={
            "cause": edge.cause,
            "effect": edge.effect,
            "lag_bars": edge.optimal_lag,
            "f_statistic": edge.f_statistic,
            "source": "granger_causal_dag",
        },
        regime_context={},
    )


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class CausalHypothesisConverter:
    """
    Converts significant causal DAG edges into Hypothesis objects.

    Only converts edges where:
      1. The target (effect) is a trade outcome feature.
      2. effect_size > MIN_EFFECT_SIZE.
      3. optimal_lag <= MAX_LAG.

    Parameters
    ----------
    instruments    : list of instrument symbols these hypotheses apply to
    min_effect_size: minimum Granger effect size to generate a hypothesis
    max_lag        : maximum lag to accept
    """

    def __init__(
        self,
        instruments: list[str] | None = None,
        min_effect_size: float = MIN_EFFECT_SIZE,
        max_lag: int = MAX_LAG,
    ) -> None:
        self.instruments = instruments or []
        self.min_effect_size = min_effect_size
        self.max_lag = max_lag

    def convert_result(self, result: GrangerResult) -> list[Hypothesis]:
        """
        Convert all qualifying edges from a GrangerResult into hypotheses.
        """
        qualifying = [
            e for e in result.significant_edges
            if _is_outcome(e.effect)
            and e.effect_size >= self.min_effect_size
            and e.optimal_lag <= self.max_lag
        ]

        log.info(
            "CausalHypothesisConverter: %d/%d significant edges qualify",
            len(qualifying), result.n_significant,
        )

        hypotheses: list[Hypothesis] = []
        for edge in qualifying:
            h = self._edge_to_hypothesis(edge)
            if h is not None:
                hypotheses.append(h)

        return hypotheses

    def convert_graph(
        self, graph: CausalGraph, outcome_features: list[str] | None = None
    ) -> list[Hypothesis]:
        """
        Convert a CausalGraph into hypotheses by inspecting all edges.
        outcome_features: if provided, only convert edges targeting these features.
        """
        all_edges = graph.get_all_edges()

        qualifying = []
        for edge_dict in all_edges:
            cause = edge_dict["cause"]
            effect = edge_dict["effect"]
            eff_size = float(edge_dict.get("effect_size", 0.0))
            lag = int(edge_dict.get("lag", 1))

            if outcome_features is not None:
                if effect not in outcome_features:
                    continue
            else:
                if not _is_outcome(effect):
                    continue

            if eff_size < self.min_effect_size:
                continue
            if lag > self.max_lag:
                continue

            qualifying.append(
                GrangerEdge(
                    cause=cause,
                    effect=effect,
                    optimal_lag=lag,
                    p_value=float(edge_dict.get("p_value", 0.05)),
                    raw_p_value=float(edge_dict.get("raw_p_value", 0.05)),
                    f_statistic=float(edge_dict.get("f_statistic", 0.0)),
                    effect_size=eff_size,
                    significant=True,
                    aic_at_lag=float(edge_dict.get("aic", 0.0)),
                    bic_at_lag=float(edge_dict.get("bic", 0.0)),
                )
            )

        log.info(
            "convert_graph: %d qualifying edges → hypotheses", len(qualifying)
        )

        hypotheses: list[Hypothesis] = []
        for edge in qualifying:
            h = self._edge_to_hypothesis(edge)
            if h is not None:
                hypotheses.append(h)

        return hypotheses

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _edge_to_hypothesis(self, edge: GrangerEdge) -> Hypothesis | None:
        """Convert a single GrangerEdge to a Hypothesis."""
        try:
            hyp_type = _infer_hypothesis_type(edge.cause)
            params = _build_parameters(edge, self.instruments)
            sharpe_delta = _sharpe_delta_from_edge(edge)
            dd_delta = _dd_delta_from_edge(edge)
            novelty = _novelty(edge)

            # Build a synthetic parent pattern for traceability
            synthetic_pattern = _build_synthetic_pattern(edge, self.instruments)

            desc = (
                f"[Causal DAG] Feature '{edge.cause}' Granger-causes '{edge.effect}' "
                f"with lag={edge.optimal_lag} bars "
                f"(F={edge.f_statistic:.3f}, p={edge.p_value:.4f}, "
                f"effect_size={edge.effect_size:.4f}). "
                f"→ Use '{edge.cause}' as leading indicator for '{edge.effect}'. "
                f"Instruments: {self.instruments}."
            )

            return Hypothesis.create(
                hypothesis_type=hyp_type,
                parent_pattern_id=synthetic_pattern.pattern_id,
                parameters=params,
                predicted_sharpe_delta=round(sharpe_delta, 4),
                predicted_dd_delta=round(dd_delta, 4),
                novelty_score=round(novelty, 4),
                description=desc,
            )

        except Exception as exc:
            log.error(
                "Failed to convert edge %s → %s: %s",
                edge.cause, edge.effect, exc, exc_info=True,
            )
            return None
