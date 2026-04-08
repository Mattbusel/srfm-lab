"""
Macro Analyst debate agent.

Specializes in:
  - DXY / Fed policy impact on crypto
  - Risk-on / risk-off regime detection
  - Equity-crypto correlation regime
  - Liquidity conditions (M2, SOFR, credit spreads)
  - Global macro cycle phase
  - BTC dominance and altcoin cycle
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .base_agent import BaseDebateAgent, DebateArgument, ArgumentStrength


@dataclass
class MacroAnalyst(BaseDebateAgent):
    name: str = "macro_analyst"
    expertise: str = "Global macro, DXY, liquidity cycles, risk-on/off, BTC dominance"
    weight: float = 1.2

    def evaluate(self, hypothesis: Any, context: dict) -> DebateArgument:
        h = hypothesis
        args_for = []
        args_against = []
        confidence = 0.5

        tags = getattr(h, "tags", [])
        params = getattr(h, "parameters", {})
        symbol = params.get("symbol", "")

        # Risk-on / risk-off regime
        risk_regime = context.get("risk_regime", "neutral")
        if risk_regime == "risk_off":
            if any(t in str(tags) for t in ["long", "momentum"]):
                args_against.append(
                    "Current macro environment is risk-off (rising DXY, falling equities). "
                    "Long crypto momentum strategies historically lose 35% more in risk-off. "
                    "Recommend halving position size or waiting for regime reversal."
                )
                confidence -= 0.2
        elif risk_regime == "risk_on":
            if any(t in str(tags) for t in ["long", "momentum"]):
                args_for.append(
                    "Risk-on macro regime: DXY weakening, equity markets strong. "
                    "Crypto typically outperforms in this environment. "
                    "Favorable macro tailwind for this hypothesis."
                )
                confidence += 0.12

        # DXY direction
        dxy_trend = context.get("dxy_trend", 0.0)  # % change over 20d
        if dxy_trend > 2.0:
            args_against.append(
                f"DXY up {dxy_trend:.1f}% over 20 days — strong dollar headwind for crypto. "
                f"Historically each 1% DXY gain = ~3-5% crypto underperformance."
            )
            confidence -= 0.1
        elif dxy_trend < -1.5:
            args_for.append(
                f"DXY weakening ({dxy_trend:.1f}% over 20d) — dollar weakness supports crypto. "
                f"Positive macro backdrop."
            )
            confidence += 0.08

        # Liquidity conditions
        liquidity_score = context.get("liquidity_score", 0.0)  # -1 tight, +1 loose
        if liquidity_score < -0.5:
            args_against.append(
                f"Liquidity conditions tight (score={liquidity_score:.2f}). "
                f"Higher SOFR, credit spread widening. Risk assets under pressure. "
                f"Reduce leverage and avoid high-beta altcoins."
            )
            confidence -= 0.15
        elif liquidity_score > 0.3:
            args_for.append(
                f"Loose liquidity environment (score={liquidity_score:.2f}). "
                f"Excess capital seeking yield supports crypto momentum."
            )
            confidence += 0.1

        # BTC dominance
        btc_dominance = context.get("btc_dominance", 0.5)
        if btc_dominance > 0.55 and symbol not in ("BTC", ""):
            args_against.append(
                f"BTC dominance {btc_dominance:.0%} and rising — capital rotating to BTC, "
                f"away from alts. Altcoin {symbol} likely underperforms. "
                f"Consider BTC exposure over altcoin."
            )
            confidence -= 0.1
        elif btc_dominance < 0.40 and symbol not in ("BTC", ""):
            args_for.append(
                f"Altcoin season conditions: BTC dominance at {btc_dominance:.0%}. "
                f"Risk capital flowing into alts — favorable for {symbol}."
            )
            confidence += 0.1

        # Fed policy
        fed_direction = context.get("fed_direction", "neutral")
        if fed_direction == "hawkish":
            args_against.append(
                "Fed in hawkish cycle — higher rates increase discount rate, "
                "particularly damaging for high-beta speculative assets."
            )
            confidence -= 0.08
        elif fed_direction == "dovish":
            args_for.append(
                "Fed pivoting dovish — rate cuts or QE expected. "
                "Historically crypto rallies aggressively in early easing cycles."
            )
            confidence += 0.1

        # Crypto-specific: altcoin cycle phase
        altcoin_cycle_phase = context.get("altcoin_cycle_phase", "unknown")
        if altcoin_cycle_phase == "late" and symbol not in ("BTC", "ETH"):
            args_against.append(
                "Late-stage altcoin cycle detected — speculative excess in small caps. "
                "High reversal risk in next 30-60 days."
            )
            confidence -= 0.12

        if not args_for:
            args_for.append("Macro conditions are neutral — no strong directional bias.")

        strength = ArgumentStrength.STRONG if abs(confidence - 0.5) > 0.15 else ArgumentStrength.MODERATE

        return DebateArgument(
            agent_name=self.name,
            for_hypothesis=confidence > 0.5,
            confidence=float(min(max(confidence, 0.1), 0.95)),
            arguments_for=args_for,
            arguments_against=args_against,
            key_concern=(
                args_against[0] if args_against and confidence < 0.5
                else (args_for[0] if args_for else "Macro neutral.")
            ),
            strength=strength,
            suggested_modifications=self._macro_modifications(risk_regime, dxy_trend),
        )

    def _macro_modifications(self, risk_regime: str, dxy_trend: float) -> list[str]:
        mods = []
        if risk_regime == "risk_off":
            mods.append("Add macro filter: only enter when SPX 20d return > -3% and DXY 20d < +2%.")
        if dxy_trend > 3.0:
            mods.append("Consider halving altcoin exposure; increase BTC relative weight.")
        return mods
