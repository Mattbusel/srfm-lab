"""
Volatility Expert debate agent.

Specializes in:
  - Vol regime classification and transitions
  - Variance risk premium (VRP) dynamics
  - GARCH/rough vol model implications
  - Options market vol signals
  - VoV and vol clustering effects on strategy performance
  - Tail risk from fat tails and jump diffusion
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .base_agent import BaseDebateAgent, DebateArgument, ArgumentStrength


@dataclass
class VolatilityExpert(BaseDebateAgent):
    name: str = "volatility_expert"
    expertise: str = "Volatility regimes, GARCH, rough vol, VRP, jump risk"
    weight: float = 1.3  # high weight for vol-sensitive hypotheses

    def evaluate(self, hypothesis: Any, context: dict) -> DebateArgument:
        """Evaluate hypothesis from vol expert perspective."""
        h = hypothesis
        args_for = []
        args_against = []
        confidence = 0.5

        # Check vol regime compatibility
        current_vol_regime = context.get("vol_regime", "medium")
        hypothesis_type = getattr(h, "hypothesis_type", "")

        params = getattr(h, "parameters", {})
        tags = getattr(h, "tags", [])

        # Assess entry timing in context of vol
        if "entry" in str(hypothesis_type).lower():
            if current_vol_regime == "spike":
                args_against.append(
                    "Entering in vol spike regime is high risk — fat tails make stops unreliable. "
                    "Expected drawdown 2-3x normal. Suggest waiting for vol stabilization."
                )
                confidence -= 0.2
            elif current_vol_regime == "low":
                args_for.append(
                    "Low vol regime favors entry — tight stops more effective, "
                    "vol expansion potential if thesis correct increases expected return."
                )
                confidence += 0.1

        # VRP check
        vrp = context.get("variance_risk_premium", 0.0)
        if vrp > 0.01 and "long" in str(tags):
            args_for.append(
                f"Positive VRP of {vrp:.1%} suggests realized vol likely to compress — "
                f"favorable for long momentum trades."
            )
            confidence += 0.08
        elif vrp < -0.005 and "long" in str(tags):
            args_against.append(
                f"Negative VRP ({vrp:.1%}) — market implying vol expansion. "
                f"Consider reducing size or widening stops."
            )
            confidence -= 0.1

        # Jump risk assessment
        jump_intensity = context.get("jump_intensity", 0.0)
        if jump_intensity > 5.0:  # >5 jumps/year
            args_against.append(
                f"High jump intensity ({jump_intensity:.0f}/year) detected. "
                f"Jump processes invalidate Gaussian stop-loss assumptions. "
                f"Gap risk is elevated — reduce position or use options for protection."
            )
            confidence -= 0.15

        # Vol of vol
        vov = context.get("vol_of_vol", 0.0)
        if vov > 1.5:
            args_against.append(
                f"Vol-of-vol ({vov:.1f}) exceeds 1.5 — regime instability. "
                f"Strategy parameters calibrated in different vol regime may underperform."
            )
            confidence -= 0.1

        # Rough vol Hurst
        hurst_vol = context.get("vol_hurst", 0.5)
        if hurst_vol < 0.2:  # very rough — vol extremely mean-reverting
            args_for.append(
                f"Vol Hurst H={hurst_vol:.2f} (rough vol regime) — vol spikes very short-lived, "
                f"entry after spike is especially favorable."
            )
            confidence += 0.08

        # Term structure
        term_slope = context.get("vol_term_structure_slope", 0.0)
        if term_slope < -0.2:
            args_against.append(
                "Inverted vol term structure — market pricing imminent stress event. "
                "Reduce directional exposure until term structure normalizes."
            )
            confidence -= 0.12

        if not args_for:
            args_for.append("Vol conditions are neutral — hypothesis is viable on vol basis.")

        strength = ArgumentStrength.STRONG if abs(confidence - 0.5) > 0.15 else ArgumentStrength.MODERATE

        return DebateArgument(
            agent_name=self.name,
            for_hypothesis=confidence > 0.5,
            confidence=float(min(max(confidence, 0.1), 0.95)),
            arguments_for=args_for,
            arguments_against=args_against,
            key_concern=(
                args_against[0] if args_against and confidence < 0.5
                else (args_for[0] if args_for else "Vol conditions neutral.")
            ),
            strength=strength,
            suggested_modifications=self._suggest_vol_modifications(params, current_vol_regime, vov),
        )

    def _suggest_vol_modifications(
        self,
        params: dict,
        vol_regime: str,
        vov: float,
    ) -> list[str]:
        suggestions = []
        if vol_regime == "high" or vol_regime == "spike":
            suggestions.append("Widen stop-loss by 1.5x in current vol regime.")
        if vov > 1.5:
            suggestions.append("Add VoV filter: only enter when VoV < 1.2.")
        if "stop_loss_pct" in params:
            sl = params["stop_loss_pct"]
            if sl < 0.03:
                suggestions.append(
                    f"Stop loss {sl:.1%} may be too tight given current vol. "
                    f"Suggest minimum {max(sl * 1.5, 0.04):.1%}."
                )
        return suggestions
