"""
hypothesis/templates/exit_rule_template.py

Converts anomaly / mass_physics MinedPatterns into exit-rule hypotheses.

Generated hypotheses:
  1. "Exit when BH mass drops below threshold T"
  2. "Exit after N bars in a losing position"
  3. "ATR-multiple trailing stop"
  4. "Exit on anomaly score spike"
"""

from __future__ import annotations

import math
from typing import Any

from hypothesis.types import Hypothesis, HypothesisType, MinedPattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sharpe_delta_from_exit(
    effect_size: float,
    p_value: float,
    ci_lower: float,
    ci_upper: float,
    exit_type: str,
) -> float:
    """
    Estimate Sharpe improvement from an exit rule.
    Exit rules typically have a different calibration than entry filters:
    they can cut losers faster (positive Sharpe effect) but may also cut winners
    (negative if too tight). We use effect_size direction to decide.
    """
    if p_value <= 0:
        p_value = 1e-10
    if p_value >= 1:
        p_value = 1 - 1e-10

    confidence = min(-math.log10(p_value) / 8.0, 1.0)
    ci_width = max(ci_upper - ci_lower, 1e-6)
    ci_penalty = 1.0 / (1.0 + ci_width * 0.5)

    # For mass_physics exits, effect sizes tend to be larger but noisier
    scale = {"mass_threshold": 0.9, "time_stop": 0.7, "atr_stop": 0.8, "anomaly_spike": 0.75}.get(
        exit_type, 0.8
    )

    raw = effect_size * confidence * ci_penalty * scale
    return float(max(min(raw, 1.2), -0.5))


def _dd_delta_from_exit(effect_size: float, pattern_type: str) -> float:
    """Exit rules generally improve drawdown (positive = less drawdown)."""
    base = max(effect_size * 0.08, 0.0)
    if pattern_type == "mass_physics":
        base += 0.03  # mass-based exits are particularly good at limiting tail loss
    return float(min(base, 0.20))


def _compute_mass_threshold(evidence: dict[str, Any]) -> float:
    """
    Derive the BH-mass exit threshold from pattern evidence.
    Looks for: mass_at_exit, mass_percentile, mean_exit_mass.
    Falls back to 0.3 (conservative default).
    """
    if "mass_at_exit" in evidence:
        v = float(evidence["mass_at_exit"])
        return round(v, 4)
    if "mass_percentile_10" in evidence:
        return round(float(evidence["mass_percentile_10"]), 4)
    if "mean_exit_mass" in evidence:
        return round(float(evidence["mean_exit_mass"]) * 0.8, 4)
    # Fallback
    return 0.30


def _compute_max_loss_bars(evidence: dict[str, Any]) -> int:
    """
    Derive max-loss-bars from evidence.
    Looks for: median_loss_duration, mean_loss_bars, loss_bar_p90.
    Falls back to 8.
    """
    for key in ("median_loss_duration", "mean_loss_bars", "loss_bar_p75"):
        if key in evidence:
            v = int(round(float(evidence[key])))
            return max(v, 2)
    return 8


def _compute_atr_mult(evidence: dict[str, Any], effect_size: float) -> float:
    """
    Derive ATR stop multiplier. Higher effect size → tighter stop.
    Looks for: optimal_atr_mult, atr_mult_at_best_sharpe.
    """
    for key in ("optimal_atr_mult", "atr_mult_at_best_sharpe"):
        if key in evidence:
            return round(float(evidence[key]), 2)
    # Heuristic: large effect → tighter stop, small effect → looser stop
    if effect_size > 0.8:
        return 1.5
    if effect_size > 0.4:
        return 2.0
    return 2.5


def _compute_anomaly_threshold(evidence: dict[str, Any]) -> float:
    """Z-score threshold above which an anomaly triggers exit."""
    for key in ("anomaly_zscore_threshold", "spike_zscore", "zscore_p95"):
        if key in evidence:
            return round(float(evidence[key]), 2)
    return 2.5


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

class ExitRuleTemplate:
    """
    Generates exit-rule hypotheses from anomaly and mass_physics MinedPatterns.

    Expected evidence fields (any subset):
        mass_at_exit: float          — BH mass when exit triggered
        mean_exit_mass: float        — average BH mass across exits
        mass_percentile_10: float    — 10th percentile of exit mass
        median_loss_duration: int    — median bars in losing trade
        mean_loss_bars: int          — mean bars in losing trade
        loss_bar_p75: int            — 75th pctile bars in loss
        optimal_atr_mult: float      — ATR stop multiple
        anomaly_zscore_threshold: float
        spike_zscore: float
    """

    HANDLED_TYPES = {"anomaly", "mass_physics"}
    MIN_EFFECT_SIZE = 0.05

    def can_handle(self, pattern: MinedPattern) -> bool:
        return pattern.pattern_type in self.HANDLED_TYPES

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        if not self.can_handle(pattern):
            raise ValueError(
                f"ExitRuleTemplate cannot handle pattern type '{pattern.pattern_type}'"
            )
        if abs(pattern.effect_size) < self.MIN_EFFECT_SIZE:
            return []

        hypotheses: list[Hypothesis] = []

        if pattern.pattern_type == "mass_physics":
            hypotheses.extend(self._mass_threshold_hypotheses(pattern))
            hypotheses.extend(self._time_stop_hypotheses(pattern))
            hypotheses.extend(self._atr_stop_hypotheses(pattern))
        elif pattern.pattern_type == "anomaly":
            hypotheses.extend(self._anomaly_spike_hypotheses(pattern))
            hypotheses.extend(self._time_stop_hypotheses(pattern))

        return hypotheses

    # ------------------------------------------------------------------
    # Sub-generators
    # ------------------------------------------------------------------

    def _mass_threshold_hypotheses(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Exit when BH mass drops below a threshold."""
        threshold = _compute_mass_threshold(pattern.evidence)
        results = []

        for instrument in pattern.instruments:
            sharpe_delta = _sharpe_delta_from_exit(
                pattern.effect_size, pattern.p_value,
                pattern.ci_lower, pattern.ci_upper, "mass_threshold"
            )
            dd_delta = _dd_delta_from_exit(pattern.effect_size, pattern.pattern_type)

            # Also generate a slightly tighter and looser variant
            for mult, label in [(1.0, "base"), (1.2, "tight"), (0.8, "loose")]:
                t = round(threshold * mult, 4)
                params: dict[str, Any] = {
                    "exit_mass_threshold": t,
                    "instruments": [instrument],
                    "exit_trigger": "bh_mass_drop",
                    "variant": label,
                }
                desc = (
                    f"[{instrument}] Exit when BH mass drops below {t:.4f} ({label} variant). "
                    f"Pattern effect_size={pattern.effect_size:.3f}, p={pattern.p_value:.4f}."
                )
                h = Hypothesis.create(
                    hypothesis_type=HypothesisType.EXIT_RULE,
                    parent_pattern_id=pattern.pattern_id,
                    parameters=params,
                    predicted_sharpe_delta=sharpe_delta * (1.0 if label == "base" else 0.85),
                    predicted_dd_delta=dd_delta,
                    novelty_score=self._novelty(pattern, label),
                    description=desc,
                )
                results.append(h)

        return results

    def _time_stop_hypotheses(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Exit after N bars in a losing position."""
        max_bars = _compute_max_loss_bars(pattern.evidence)
        results = []

        sharpe_delta = _sharpe_delta_from_exit(
            pattern.effect_size, pattern.p_value,
            pattern.ci_lower, pattern.ci_upper, "time_stop"
        )
        dd_delta = _dd_delta_from_exit(pattern.effect_size, pattern.pattern_type)

        for n_bars in sorted({max(max_bars - 2, 2), max_bars, max_bars + 4}):
            params: dict[str, Any] = {
                "max_loss_bars": n_bars,
                "instruments": pattern.instruments,
                "exit_trigger": "time_stop",
                "require_loss": True,   # only exits if position is in loss
            }
            desc = (
                f"Exit after {n_bars} bars in a losing position. "
                f"Instruments: {pattern.instruments}."
            )
            h = Hypothesis.create(
                hypothesis_type=HypothesisType.EXIT_RULE,
                parent_pattern_id=pattern.pattern_id,
                parameters=params,
                predicted_sharpe_delta=sharpe_delta * 0.9,
                predicted_dd_delta=dd_delta * 0.9,
                novelty_score=self._novelty(pattern, f"time_{n_bars}"),
                description=desc,
            )
            results.append(h)

        return results

    def _atr_stop_hypotheses(self, pattern: MinedPattern) -> list[Hypothesis]:
        """ATR-multiple trailing stop derived from mass_physics pattern."""
        atr_mult = _compute_atr_mult(pattern.evidence, pattern.effect_size)
        results = []

        sharpe_delta = _sharpe_delta_from_exit(
            pattern.effect_size, pattern.p_value,
            pattern.ci_lower, pattern.ci_upper, "atr_stop"
        )
        dd_delta = _dd_delta_from_exit(pattern.effect_size, pattern.pattern_type)

        for mult_offset in [-0.5, 0.0, 0.5]:
            m = round(atr_mult + mult_offset, 1)
            if m <= 0:
                continue
            params: dict[str, Any] = {
                "atr_stop_mult": m,
                "instruments": pattern.instruments,
                "exit_trigger": "atr_trailing",
                "trail_from_entry": True,
            }
            desc = (
                f"ATR trailing stop at {m}× ATR. "
                f"Instruments: {pattern.instruments}. "
                f"From mass_physics pattern {pattern.pattern_id[:8]}."
            )
            h = Hypothesis.create(
                hypothesis_type=HypothesisType.EXIT_RULE,
                parent_pattern_id=pattern.pattern_id,
                parameters=params,
                predicted_sharpe_delta=sharpe_delta * 0.95,
                predicted_dd_delta=dd_delta,
                novelty_score=self._novelty(pattern, f"atr_{m}"),
                description=desc,
            )
            results.append(h)

        return results

    def _anomaly_spike_hypotheses(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Exit when anomaly z-score exceeds threshold."""
        threshold = _compute_anomaly_threshold(pattern.evidence)
        results = []

        sharpe_delta = _sharpe_delta_from_exit(
            pattern.effect_size, pattern.p_value,
            pattern.ci_lower, pattern.ci_upper, "anomaly_spike"
        )
        dd_delta = _dd_delta_from_exit(pattern.effect_size, pattern.pattern_type)

        for mult, label in [(0.8, "aggressive"), (1.0, "base"), (1.3, "conservative")]:
            t = round(threshold * mult, 2)
            params: dict[str, Any] = {
                "anomaly_zscore_exit": t,
                "instruments": pattern.instruments,
                "exit_trigger": "anomaly_spike",
                "variant": label,
            }
            desc = (
                f"Exit when anomaly z-score exceeds {t:.2f} ({label}). "
                f"Instruments: {pattern.instruments}."
            )
            h = Hypothesis.create(
                hypothesis_type=HypothesisType.EXIT_RULE,
                parent_pattern_id=pattern.pattern_id,
                parameters=params,
                predicted_sharpe_delta=sharpe_delta * (1.0 if label == "base" else 0.88),
                predicted_dd_delta=dd_delta,
                novelty_score=self._novelty(pattern, label),
                description=desc,
            )
            results.append(h)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _novelty(pattern: MinedPattern, variant: str = "base") -> float:
        base = 0.5
        if pattern.pattern_type == "mass_physics":
            base += 0.15   # mass-physics exits are domain-specific → higher novelty
        if abs(pattern.effect_size) > 0.5:
            base += 0.1
        if variant != "base":
            base -= 0.05   # variants are less novel than the base idea
        if pattern.regime_context:
            base += min(len(pattern.regime_context) * 0.03, 0.15)
        return min(max(base, 0.1), 1.0)
