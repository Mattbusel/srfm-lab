"""
Microstructure Specialist debate agent.

Specializes in:
  - Order flow toxicity (VPIN, adverse selection)
  - Bid-ask spread dynamics
  - Market impact and execution cost estimation
  - Slippage in different liquidity regimes
  - Order book resilience
  - Trade timing relative to microstructure cycles
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .base_agent import BaseDebateAgent, DebateArgument, ArgumentStrength


@dataclass
class MicrostructureSpecialist(BaseDebateAgent):
    name: str = "microstructure_specialist"
    expertise: str = "VPIN, adverse selection, spread, market impact, execution quality"
    weight: float = 1.1

    def evaluate(self, hypothesis: Any, context: dict) -> DebateArgument:
        h = hypothesis
        args_for = []
        args_against = []
        confidence = 0.5

        params = getattr(h, "parameters", {})
        tags = getattr(h, "tags", [])
        symbol = params.get("symbol", "")

        # VPIN check
        vpin = context.get("vpin", 0.5)
        if vpin > 0.75:
            args_against.append(
                f"VPIN {vpin:.2f} elevated — high probability of informed trading. "
                f"Adverse selection cost is high. Market makers widening spreads. "
                f"Entry/exit execution quality will suffer — wait for VPIN < 0.65."
            )
            confidence -= 0.2
        elif vpin < 0.4:
            args_for.append(
                f"VPIN {vpin:.2f} low — uninformed flow dominant, tight spreads, "
                f"good execution quality expected."
            )
            confidence += 0.08

        # Spread regime
        spread_bps = context.get("spread_bps", 5)
        typical_spread = context.get("typical_spread_bps", 5)
        if spread_bps > typical_spread * 2:
            args_against.append(
                f"Bid-ask spread {spread_bps:.0f}bps vs typical {typical_spread:.0f}bps — "
                f"2x widening reduces net alpha. Expected entry+exit round-trip cost: "
                f"{spread_bps * 2:.0f}bps."
            )
            confidence -= 0.1
            # Check if hypothesis edge exceeds spread cost
            edge = getattr(h, "expected_impact", 0.0)
            if edge < spread_bps / 10000 * 2:
                args_against.append(
                    f"WARNING: Expected edge {edge:.2%} < round-trip spread cost "
                    f"{spread_bps / 10000 * 2:.2%}. Hypothesis is NOT viable at current spreads."
                )
                confidence -= 0.25

        # Position size vs daily volume
        position_size_usd = context.get("position_size_usd", 0)
        daily_volume_usd = context.get("daily_volume_usd", 1e9)
        participation_rate = position_size_usd / max(daily_volume_usd, 1) if daily_volume_usd > 0 else 0.0
        if participation_rate > 0.02:  # >2% of daily volume
            args_against.append(
                f"Position size {position_size_usd:,.0f} = {participation_rate:.1%} of daily volume. "
                f"Market impact will be significant. Estimate impact cost: "
                f"{participation_rate * 50:.0f}bps via square-root impact model."
            )
            confidence -= 0.1
        elif participation_rate < 0.001:
            args_for.append(
                f"Position size negligible ({participation_rate:.3%} of daily volume) — "
                f"zero market impact. Execution risk minimal."
            )
            confidence += 0.05

        # Amihud illiquidity
        amihud = context.get("amihud_illiquidity", 0.0)
        amihud_historical = context.get("amihud_historical_avg", 0.0)
        if amihud_historical > 0 and amihud > amihud_historical * 2:
            args_against.append(
                f"Amihud illiquidity {amihud:.2e} — 2x above historical average. "
                f"Liquidity crisis or thin order book. Increase slippage assumption by 50%."
            )
            confidence -= 0.1

        # Time-of-day microstructure
        hour_utc = context.get("hour_utc", 12)
        if hour_utc in (0, 1, 2, 3):  # UTC 0-3 = low liquidity for crypto
            args_against.append(
                "Entry during UTC 0-3 low-liquidity window. "
                "Bid-ask spreads 30-50% wider, higher adverse selection. "
                "Consider delaying entry to UTC 8-16 window."
            )
            confidence -= 0.08

        # Kyle lambda
        kyle_lambda = context.get("kyle_lambda", 0.0)
        if kyle_lambda > 0:
            impact_est = kyle_lambda * position_size_usd if position_size_usd > 0 else 0
            if impact_est > 0.002:  # >20bps impact
                args_against.append(
                    f"Kyle's lambda = {kyle_lambda:.2e}. "
                    f"Estimated price impact = {impact_est:.2%}. "
                    f"Consider splitting order over multiple bars."
                )
                confidence -= 0.05

        if not args_for:
            args_for.append(
                "Microstructure conditions are acceptable — spreads and liquidity within normal range."
            )

        strength = ArgumentStrength.STRONG if abs(confidence - 0.5) > 0.15 else ArgumentStrength.MODERATE

        return DebateArgument(
            agent_name=self.name,
            for_hypothesis=confidence > 0.5,
            confidence=float(min(max(confidence, 0.1), 0.95)),
            arguments_for=args_for,
            arguments_against=args_against,
            key_concern=(
                args_against[0] if args_against and confidence < 0.5
                else (args_for[0] if args_for else "Microstructure neutral.")
            ),
            strength=strength,
            suggested_modifications=self._micro_modifications(vpin, spread_bps, participation_rate),
        )

    def _micro_modifications(
        self,
        vpin: float,
        spread_bps: float,
        participation_rate: float,
    ) -> list[str]:
        mods = []
        if vpin > 0.7:
            mods.append("Add VPIN gate: only enter when VPIN < 0.65.")
        if spread_bps > 20:
            mods.append("Use limit orders only — market orders too expensive at current spreads.")
        if participation_rate > 0.01:
            mods.append("Split entry into 3-5 child orders over 15-30 min to reduce impact.")
        return mods
