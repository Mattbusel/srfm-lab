"""
Liquidity trap hypothesis template.

Generates hypotheses around liquidity events:
  - Low liquidity periods (avoid entry)
  - Liquidity drain before large moves
  - Spread widening as exit signal
  - Volume drought before breakouts
  - Amihud illiquidity regime shifts
"""

from __future__ import annotations
from dataclasses import dataclass
from ..types import Hypothesis, MinedPattern, HypothesisType, HypothesisStatus


@dataclass
class LiquidityTrapTemplate:
    name: str = "liquidity_trap"

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        hypotheses = []

        if pattern.pattern_type == "low_liquidity_entry":
            hypotheses += self._low_liquidity_avoidance(pattern)
        elif pattern.pattern_type == "liquidity_drain_precursor":
            hypotheses += self._liquidity_drain_exit(pattern)
        elif pattern.pattern_type == "volume_drought_breakout":
            hypotheses += self._volume_drought_breakout(pattern)
        elif pattern.pattern_type == "amihud_regime_shift":
            hypotheses += self._amihud_regime_adjustment(pattern)

        return hypotheses

    def _low_liquidity_avoidance(self, pattern: MinedPattern) -> list[Hypothesis]:
        loss_in_low_liq = pattern.metadata.get("avg_loss_low_liquidity", -0.015)
        return [Hypothesis(
            id=f"low_liq_avoid_{pattern.symbol}",
            name="Low Liquidity Period — Avoid Entry",
            description=(
                f"Entries in {pattern.symbol} during low liquidity (Amihud > 90th pct, "
                f"volume < 30th pct) historically lose avg {loss_in_low_liq:.1%}. "
                f"Skip entries during low liquidity windows."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "amihud_pct_threshold": 90,
                "volume_pct_threshold": 30,
                "liquidity_lookback": 30,
                "skip_entry_in_low_liq": True,
            },
            expected_impact=abs(loss_in_low_liq),
            confidence=0.65,
            source_pattern=pattern,
            tags=["liquidity", "filter", "amihud"],
        )]

    def _liquidity_drain_exit(self, pattern: MinedPattern) -> list[Hypothesis]:
        return [Hypothesis(
            id=f"liq_drain_exit_{pattern.symbol}",
            name="Liquidity Drain Precedes Large Move — Exit",
            description=(
                f"When bid-ask spread in {pattern.symbol} widens >50% of 20d average "
                f"while holding a position, it often precedes a large adverse move. "
                f"Exit 40% of position when spread widens significantly."
            ),
            hypothesis_type=HypothesisType.EXIT_RULE,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "spread_widening_threshold": 1.5,
                "spread_lookback": 20,
                "exit_fraction": 0.4,
            },
            expected_impact=pattern.metadata.get("drain_loss_avoided", 0.015),
            confidence=0.60,
            source_pattern=pattern,
            tags=["liquidity", "spread", "exit"],
        )]

    def _volume_drought_breakout(self, pattern: MinedPattern) -> list[Hypothesis]:
        return [Hypothesis(
            id=f"vol_drought_breakout_{pattern.symbol}",
            name="Volume Drought Breakout Entry",
            description=(
                f"After {pattern.symbol} volume drops to <20th pct for 10+ days, "
                f"the next volume surge (>2x 20d avg) on a price move is a high-quality breakout. "
                f"Enter on the surge bar with wider stop."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "volume_drought_pct": 20,
                "drought_min_bars": 10,
                "surge_threshold_x": 2.0,
                "surge_volume_lookback": 20,
                "stop_loss_pct": 0.04,
            },
            expected_impact=pattern.metadata.get("drought_breakout_return", 0.05),
            confidence=0.58,
            source_pattern=pattern,
            tags=["liquidity", "volume", "breakout", "drought"],
        )]

    def _amihud_regime_adjustment(self, pattern: MinedPattern) -> list[Hypothesis]:
        new_amihud = pattern.metadata.get("new_amihud", 0.0)
        old_amihud = pattern.metadata.get("old_amihud", 0.0)
        ratio = new_amihud / max(old_amihud, 1e-10)

        return [Hypothesis(
            id=f"amihud_regime_{pattern.symbol}",
            name="Amihud Regime Shift — Resize Positions",
            description=(
                f"Amihud illiquidity in {pattern.symbol} shifted {ratio:.1f}x. "
                f"Position sizes must be recalibrated to avoid excessive market impact. "
                f"Reduce max position by {min(ratio - 1.0, 1.0):.0%}."
            ),
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "amihud_size_scale": float(1.0 / max(ratio, 1.0)),
                "amihud_lookback": 30,
                "dynamic_sizing": True,
            },
            expected_impact=0.01,
            confidence=0.65,
            source_pattern=pattern,
            tags=["liquidity", "amihud", "sizing", "market_impact"],
        )]
