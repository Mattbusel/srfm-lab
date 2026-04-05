"""
hypothesis/templates/cross_asset_template.py

Converts cross_asset MinedPatterns into cross-asset lead-signal hypotheses.
Primary hypothesis: "Use BTC hourly BH mass as lead signal for all alts"
Generalised: "Use <lead_instrument> <feature> as lead signal for <targets> with lag L"
"""

from __future__ import annotations

import math
from typing import Any

from hypothesis.types import Hypothesis, HypothesisType, MinedPattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sharpe_delta_cross(
    effect_size: float,
    p_value: float,
    ci_lower: float,
    ci_upper: float,
    lag_bars: int,
) -> float:
    if p_value <= 0:
        p_value = 1e-10
    if p_value >= 1:
        p_value = 1 - 1e-10
    confidence = min(-math.log10(p_value) / 8.0, 1.0)
    ci_penalty = 1.0 / (1.0 + max(ci_upper - ci_lower, 1e-6) * 0.4)
    # Longer lags are harder to exploit → slight penalty
    lag_penalty = 1.0 / (1.0 + (lag_bars - 1) * 0.1)
    raw = effect_size * confidence * ci_penalty * lag_penalty
    return float(max(min(raw, 1.3), -0.3))


def _dd_delta_cross(effect_size: float) -> float:
    return float(min(max(effect_size * 0.05, 0.0), 0.15))


def _infer_lead_instrument(instruments: list[str], evidence: dict[str, Any]) -> str:
    """
    The lead instrument is the one whose move precedes the others.
    Preference order: explicit evidence → BTC → largest cap proxy → first in list.
    """
    if "lead_instrument" in evidence:
        return str(evidence["lead_instrument"])
    # Prefer BTC as the canonical crypto lead
    for sym in instruments:
        if "BTC" in sym.upper():
            return sym
    # Prefer ETH next
    for sym in instruments:
        if "ETH" in sym.upper():
            return sym
    return instruments[0] if instruments else "BTC"


def _infer_lag_bars(evidence: dict[str, Any]) -> int:
    """
    Number of bars the lead signal precedes target move.
    """
    for key in ("optimal_lag_bars", "lag_bars", "best_lag", "median_lag"):
        if key in evidence:
            return max(int(evidence[key]), 1)
    return 1


def _infer_mass_threshold(evidence: dict[str, Any], effect_size: float) -> float:
    """
    BH mass threshold on the lead instrument that must be exceeded to enter lags.
    """
    for key in ("lead_mass_threshold", "bh_mass_threshold", "mass_trigger"):
        if key in evidence:
            return round(float(evidence[key]), 4)
    # Heuristic from effect size
    if effect_size > 0.7:
        return 0.65
    if effect_size > 0.4:
        return 0.55
    return 0.45


def _infer_lead_feature(evidence: dict[str, Any]) -> str:
    """Name of the feature on the lead instrument used as signal."""
    return str(evidence.get("lead_feature", "bh_mass_hourly"))


def _infer_signal_direction(evidence: dict[str, Any]) -> str:
    """'above' → enter when mass > threshold; 'below' → enter when mass < threshold."""
    return str(evidence.get("signal_direction", "above"))


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

class CrossAssetTemplate:
    """
    Generates cross-asset lead-signal hypotheses from cross_asset MinedPatterns.

    Expected evidence fields (any subset):
        lead_instrument: str           — e.g. "BTCUSDT"
        lead_feature: str              — e.g. "bh_mass_hourly"
        lead_mass_threshold: float     — trigger threshold
        signal_direction: str          — "above" | "below"
        optimal_lag_bars: int          — lead-lag in bars
        correlation: float             — Pearson r between lead and target
        granger_p: float               — Granger causality p-value (optional)
        target_instruments: list[str]  — instruments driven by the lead
    """

    PATTERN_TYPE = "cross_asset"
    MIN_EFFECT_SIZE = 0.04

    def can_handle(self, pattern: MinedPattern) -> bool:
        return pattern.pattern_type == self.PATTERN_TYPE

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        if not self.can_handle(pattern):
            raise ValueError(
                f"CrossAssetTemplate cannot handle pattern type '{pattern.pattern_type}'"
            )
        if abs(pattern.effect_size) < self.MIN_EFFECT_SIZE:
            return []

        hypotheses: list[Hypothesis] = []

        lead = _infer_lead_instrument(pattern.instruments, pattern.evidence)
        targets = pattern.evidence.get(
            "target_instruments",
            [i for i in pattern.instruments if i != lead],
        )
        if not targets:
            targets = [i for i in pattern.instruments if i != lead]
        if not targets:
            return []

        lag_bars = _infer_lag_bars(pattern.evidence)
        mass_threshold = _infer_mass_threshold(pattern.evidence, pattern.effect_size)
        lead_feature = _infer_lead_feature(pattern.evidence)
        signal_dir = _infer_signal_direction(pattern.evidence)

        # 1. Primary: lead signal for all targets combined
        hypotheses.append(
            self._primary_hypothesis(
                pattern, lead, targets, lag_bars, mass_threshold, lead_feature, signal_dir
            )
        )

        # 2. Per-target individual hypotheses (allows testing each alt independently)
        for target in targets:
            hypotheses.append(
                self._per_target_hypothesis(
                    pattern, lead, target, lag_bars, mass_threshold, lead_feature, signal_dir
                )
            )

        # 3. Lag variations (+1, -1 bar)
        for lag_offset, label in [(-1, "faster"), (1, "slower")]:
            adjusted_lag = max(lag_bars + lag_offset, 1)
            if adjusted_lag != lag_bars:
                hypotheses.append(
                    self._lag_variation_hypothesis(
                        pattern, lead, targets, adjusted_lag,
                        mass_threshold, lead_feature, signal_dir, label
                    )
                )

        # 4. Threshold variation (tighter / looser mass threshold)
        for mult, label in [(0.85, "loose"), (1.15, "tight")]:
            t = round(mass_threshold * mult, 4)
            hypotheses.append(
                self._threshold_variation_hypothesis(
                    pattern, lead, targets, lag_bars, t, lead_feature, signal_dir, label
                )
            )

        return hypotheses

    # ------------------------------------------------------------------
    # Sub-builders
    # ------------------------------------------------------------------

    def _primary_hypothesis(
        self,
        pattern: MinedPattern,
        lead: str,
        targets: list[str],
        lag_bars: int,
        mass_threshold: float,
        lead_feature: str,
        signal_dir: str,
    ) -> Hypothesis:
        sharpe_delta = _sharpe_delta_cross(
            pattern.effect_size, pattern.p_value,
            pattern.ci_lower, pattern.ci_upper, lag_bars
        )
        dd_delta = _dd_delta_cross(pattern.effect_size)

        params: dict[str, Any] = {
            "lead_instrument": lead,
            "lead_feature": lead_feature,
            "lead_mass_threshold": mass_threshold,
            "signal_direction": signal_dir,
            "lag_bars": lag_bars,
            "target_instruments": targets,
            "instruments": pattern.instruments,
        }
        desc = (
            f"Use {lead} {lead_feature} {signal_dir} {mass_threshold:.4f} "
            f"as lead signal for {targets} with lag={lag_bars} bars. "
            f"Effect size: {pattern.effect_size:.3f}, p={pattern.p_value:.4f}."
        )
        return Hypothesis.create(
            hypothesis_type=HypothesisType.CROSS_ASSET,
            parent_pattern_id=pattern.pattern_id,
            parameters=params,
            predicted_sharpe_delta=sharpe_delta,
            predicted_dd_delta=dd_delta,
            novelty_score=self._novelty(pattern),
            description=desc,
        )

    def _per_target_hypothesis(
        self,
        pattern: MinedPattern,
        lead: str,
        target: str,
        lag_bars: int,
        mass_threshold: float,
        lead_feature: str,
        signal_dir: str,
    ) -> Hypothesis:
        sharpe_delta = _sharpe_delta_cross(
            pattern.effect_size * 0.9, pattern.p_value,
            pattern.ci_lower, pattern.ci_upper, lag_bars
        )
        dd_delta = _dd_delta_cross(pattern.effect_size * 0.9)

        params: dict[str, Any] = {
            "lead_instrument": lead,
            "lead_feature": lead_feature,
            "lead_mass_threshold": mass_threshold,
            "signal_direction": signal_dir,
            "lag_bars": lag_bars,
            "target_instruments": [target],
            "instruments": [lead, target],
        }
        desc = (
            f"[{target}] Use {lead} {lead_feature} {signal_dir} {mass_threshold:.4f} "
            f"with lag={lag_bars} bars as entry filter for {target}."
        )
        return Hypothesis.create(
            hypothesis_type=HypothesisType.CROSS_ASSET,
            parent_pattern_id=pattern.pattern_id,
            parameters=params,
            predicted_sharpe_delta=sharpe_delta,
            predicted_dd_delta=dd_delta,
            novelty_score=self._novelty(pattern) * 0.9,
            description=desc,
        )

    def _lag_variation_hypothesis(
        self,
        pattern: MinedPattern,
        lead: str,
        targets: list[str],
        lag_bars: int,
        mass_threshold: float,
        lead_feature: str,
        signal_dir: str,
        label: str,
    ) -> Hypothesis:
        sharpe_delta = _sharpe_delta_cross(
            pattern.effect_size * 0.85, pattern.p_value,
            pattern.ci_lower, pattern.ci_upper, lag_bars
        )
        params: dict[str, Any] = {
            "lead_instrument": lead,
            "lead_feature": lead_feature,
            "lead_mass_threshold": mass_threshold,
            "signal_direction": signal_dir,
            "lag_bars": lag_bars,
            "target_instruments": targets,
            "instruments": pattern.instruments,
            "variant": label,
        }
        desc = f"Lag variation ({label}): {lead} → {targets} with lag={lag_bars}."
        return Hypothesis.create(
            hypothesis_type=HypothesisType.CROSS_ASSET,
            parent_pattern_id=pattern.pattern_id,
            parameters=params,
            predicted_sharpe_delta=sharpe_delta,
            predicted_dd_delta=_dd_delta_cross(pattern.effect_size * 0.85),
            novelty_score=self._novelty(pattern) * 0.8,
            description=desc,
        )

    def _threshold_variation_hypothesis(
        self,
        pattern: MinedPattern,
        lead: str,
        targets: list[str],
        lag_bars: int,
        mass_threshold: float,
        lead_feature: str,
        signal_dir: str,
        label: str,
    ) -> Hypothesis:
        sharpe_delta = _sharpe_delta_cross(
            pattern.effect_size * 0.88, pattern.p_value,
            pattern.ci_lower, pattern.ci_upper, lag_bars
        )
        params: dict[str, Any] = {
            "lead_instrument": lead,
            "lead_feature": lead_feature,
            "lead_mass_threshold": mass_threshold,
            "signal_direction": signal_dir,
            "lag_bars": lag_bars,
            "target_instruments": targets,
            "instruments": pattern.instruments,
            "variant": label,
        }
        desc = (
            f"Threshold variation ({label}): {lead} {lead_feature} {signal_dir} {mass_threshold:.4f} "
            f"→ {targets} lag={lag_bars}."
        )
        return Hypothesis.create(
            hypothesis_type=HypothesisType.CROSS_ASSET,
            parent_pattern_id=pattern.pattern_id,
            parameters=params,
            predicted_sharpe_delta=sharpe_delta,
            predicted_dd_delta=_dd_delta_cross(pattern.effect_size * 0.88),
            novelty_score=self._novelty(pattern) * 0.82,
            description=desc,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _novelty(pattern: MinedPattern) -> float:
        base = 0.6   # cross-asset hypotheses are inherently interesting
        if abs(pattern.effect_size) > 0.5:
            base += 0.1
        if pattern.regime_context:
            base += min(len(pattern.regime_context) * 0.03, 0.15)
        return min(base, 1.0)
