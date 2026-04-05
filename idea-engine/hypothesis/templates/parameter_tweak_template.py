"""
hypothesis/templates/parameter_tweak_template.py

Converts any MinedPattern (any type) that suggests a parameter change into
PARAMETER_TWEAK hypotheses.

Supported parameters:
    bh_form           — BH formation threshold (float, typically 0.1–1.0)
    cf                — confirmation factor (float, 0.0–1.0)
    stale_threshold   — bars before a signal is considered stale (int cast to float)
    atr_period        — ATR lookback period
    vol_lookback      — volatility lookback window
    mass_decay        — BH mass decay rate

For each parameter, we generate: raise, lower, and a calibrated "optimal" variant.
"""

from __future__ import annotations

import math
from typing import Any

from hypothesis.types import Hypothesis, HypothesisType, MinedPattern


# ---------------------------------------------------------------------------
# Parameter metadata: (min_val, max_val, step, description)
# ---------------------------------------------------------------------------

PARAM_REGISTRY: dict[str, dict[str, Any]] = {
    "bh_form": {
        "min": 0.05,
        "max": 2.0,
        "step": 0.05,
        "description": "BH formation threshold — controls how easily mass concentrates",
        "default": 0.3,
    },
    "cf": {
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "description": "Confirmation factor — fraction of confirmation required before entry",
        "default": 0.5,
    },
    "stale_threshold": {
        "min": 1.0,
        "max": 50.0,
        "step": 1.0,
        "description": "Bars before a signal is considered stale",
        "default": 10.0,
    },
    "atr_period": {
        "min": 5.0,
        "max": 100.0,
        "step": 5.0,
        "description": "ATR lookback period in bars",
        "default": 14.0,
    },
    "vol_lookback": {
        "min": 10.0,
        "max": 200.0,
        "step": 10.0,
        "description": "Volatility estimation lookback window",
        "default": 30.0,
    },
    "mass_decay": {
        "min": 0.01,
        "max": 0.99,
        "step": 0.01,
        "description": "BH mass exponential decay rate per bar",
        "default": 0.1,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sharpe_delta_param(
    effect_size: float,
    p_value: float,
    ci_lower: float,
    ci_upper: float,
    direction: str,  # "raise" | "lower" | "calibrate"
) -> float:
    if p_value <= 0:
        p_value = 1e-10
    if p_value >= 1:
        p_value = 1 - 1e-10
    confidence = min(-math.log10(p_value) / 8.0, 1.0)
    ci_penalty = 1.0 / (1.0 + max(ci_upper - ci_lower, 1e-6) * 0.3)
    scale = {"raise": 0.85, "lower": 0.85, "calibrate": 1.0}.get(direction, 0.85)
    raw = effect_size * confidence * ci_penalty * scale
    return float(max(min(raw, 1.2), -0.3))


def _dd_delta_param(effect_size: float) -> float:
    return float(min(max(effect_size * 0.04, 0.0), 0.12))


def _infer_param_name(pattern: MinedPattern) -> str | None:
    """
    Extract the parameter name the pattern is suggesting we change.
    Checks evidence first, then regime_context, then pattern_type hints.
    """
    # Explicit evidence key
    for key in ("param_name", "suggested_param", "parameter"):
        if key in pattern.evidence:
            name = str(pattern.evidence[key])
            if name in PARAM_REGISTRY:
                return name

    # Pattern-type heuristics
    if pattern.pattern_type == "mass_physics":
        if pattern.effect_size > 0.5:
            return "bh_form"
        return "mass_decay"
    if pattern.pattern_type == "anomaly":
        return "cf"
    if pattern.pattern_type == "drawdown":
        return "stale_threshold"
    if pattern.pattern_type == "time_of_day":
        return "atr_period"

    return None


def _infer_direction(pattern: MinedPattern, param_name: str) -> str:
    """
    Decide whether to raise or lower the parameter.
    effect_size > 0 → raise (more of what's working)
    effect_size < 0 → lower (less of what's hurting)
    Can be overridden by evidence.
    """
    if "param_direction" in pattern.evidence:
        d = str(pattern.evidence["param_direction"])
        if d in ("raise", "lower"):
            return d

    meta = PARAM_REGISTRY.get(param_name, {})
    current = float(pattern.evidence.get("current_value", meta.get("default", 0.5)))

    if pattern.effect_size >= 0:
        return "raise"
    return "lower"


def _compute_new_value(
    param_name: str,
    direction: str,
    effect_size: float,
    evidence: dict[str, Any],
) -> float:
    """
    Compute the suggested new parameter value.
    Uses evidence['suggested_value'] if available, otherwise steps by
    the parameter's natural step size scaled by effect_size.
    """
    if "suggested_value" in evidence:
        return float(evidence["suggested_value"])

    meta = PARAM_REGISTRY.get(param_name, {})
    current = float(evidence.get("current_value", meta.get("default", 0.5)))
    step = float(meta.get("step", 0.1))
    # Scale step by effect_size: bigger effect → bigger change
    n_steps = max(1, round(abs(effect_size) * 3))
    delta = step * n_steps

    if direction == "raise":
        new_val = current + delta
    else:
        new_val = current - delta

    # Clamp to registry bounds
    new_val = max(float(meta.get("min", -1e9)), min(new_val, float(meta.get("max", 1e9))))
    return round(new_val, 4)


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

class ParameterTweakTemplate:
    """
    Generates PARAMETER_TWEAK hypotheses from any MinedPattern that implies
    a change to a system parameter (bh_form, cf, stale_threshold, etc.).

    Works across all pattern types.
    """

    MIN_EFFECT_SIZE = 0.03

    def can_handle(self, pattern: MinedPattern) -> bool:
        """Can handle any pattern that has a detectable parameter target."""
        return _infer_param_name(pattern) is not None or "param_name" in pattern.evidence

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        if abs(pattern.effect_size) < self.MIN_EFFECT_SIZE:
            return []

        param_name = _infer_param_name(pattern)
        if param_name is None:
            return []

        direction = _infer_direction(pattern, param_name)
        meta = PARAM_REGISTRY.get(param_name, {})
        hypotheses: list[Hypothesis] = []

        current = float(pattern.evidence.get("current_value", meta.get("default", 0.5)))
        step = float(meta.get("step", 0.1))

        # --- Three variants: calibrated optimal, directional +1 step, directional +2 steps ---
        variants: list[tuple[str, float]] = []

        # Calibrated: exact suggested value
        calibrated = _compute_new_value(param_name, direction, pattern.effect_size, pattern.evidence)
        variants.append(("calibrate", calibrated))

        # Directional step variants
        for n_steps in [1, 2]:
            delta = step * n_steps
            if direction == "raise":
                v = round(
                    min(current + delta, float(meta.get("max", 1e9))), 4
                )
            else:
                v = round(
                    max(current - delta, float(meta.get("min", -1e9))), 4
                )
            label = f"{direction}_{n_steps}step"
            variants.append((label, v))

        for label, new_value in variants:
            h = self._build_hypothesis(
                pattern=pattern,
                param_name=param_name,
                new_value=new_value,
                current_value=current,
                direction=direction if label != "calibrate" else "calibrate",
                label=label,
                meta=meta,
            )
            hypotheses.append(h)

        # Per-instrument variants if pattern covers multiple instruments
        if len(pattern.instruments) > 1:
            for instrument in pattern.instruments:
                h_inst = self._build_hypothesis(
                    pattern=pattern,
                    param_name=param_name,
                    new_value=calibrated,
                    current_value=current,
                    direction="calibrate",
                    label=f"per_instrument_{instrument}",
                    meta=meta,
                    instruments=[instrument],
                )
                hypotheses.append(h_inst)

        return hypotheses

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    def _build_hypothesis(
        self,
        pattern: MinedPattern,
        param_name: str,
        new_value: float,
        current_value: float,
        direction: str,
        label: str,
        meta: dict[str, Any],
        instruments: list[str] | None = None,
    ) -> Hypothesis:
        instr = instruments or pattern.instruments

        sharpe_delta = _sharpe_delta_param(
            pattern.effect_size, pattern.p_value,
            pattern.ci_lower, pattern.ci_upper, direction
        )
        dd_delta = _dd_delta_param(pattern.effect_size)

        action = "raise" if new_value > current_value else "lower"
        change_pct = (
            (new_value - current_value) / max(abs(current_value), 1e-8) * 100.0
        )

        params: dict[str, Any] = {
            "param_name": param_name,
            "new_value": new_value,
            "current_value": current_value,
            "instruments": instr,
            "direction": action,
            "variant": label,
            "param_description": meta.get("description", ""),
        }

        desc = (
            f"{action.capitalize()} {param_name} from {current_value} → {new_value} "
            f"({change_pct:+.1f}%). Instruments: {instr}. "
            f"Variant: {label}. Pattern: {pattern.pattern_type}, "
            f"effect_size={pattern.effect_size:.3f}, p={pattern.p_value:.4f}."
        )

        return Hypothesis.create(
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            parent_pattern_id=pattern.pattern_id,
            parameters=params,
            predicted_sharpe_delta=sharpe_delta * (1.0 if label == "calibrate" else 0.88),
            predicted_dd_delta=dd_delta,
            novelty_score=self._novelty(pattern, label),
            description=desc,
        )

    @staticmethod
    def _novelty(pattern: MinedPattern, label: str) -> float:
        base = 0.4   # parameter tweaks are less novel than new rule types
        if abs(pattern.effect_size) > 0.6:
            base += 0.1
        if label == "calibrate":
            base += 0.05
        if pattern.regime_context:
            base += min(len(pattern.regime_context) * 0.03, 0.15)
        return min(base, 0.9)
