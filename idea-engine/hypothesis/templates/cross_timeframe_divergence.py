"""
Cross-timeframe divergence hypothesis template.

Generates hypotheses when signal direction diverges across timeframes:
  - 4h BH active but 1h BH fading → early exit signal
  - 15m overbought but 4h trending → hold through noise
  - Multi-TF alignment confirmation for higher confidence entries
  - Divergence as mean-reversion opportunity
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from ..types import Hypothesis, MinedPattern, HypothesisType, HypothesisStatus


@dataclass
class CrossTimeframeDivergenceTemplate:
    name: str = "cross_timeframe_divergence"

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        hypotheses = []

        if pattern.pattern_type == "tf_divergence_bh":
            hypotheses += self._bh_timeframe_divergence(pattern)
        elif pattern.pattern_type == "tf_alignment":
            hypotheses += self._multi_tf_alignment_entry(pattern)
        elif pattern.pattern_type == "tf_divergence_momentum":
            hypotheses += self._momentum_tf_divergence(pattern)
        elif pattern.pattern_type == "higher_tf_override":
            hypotheses += self._higher_tf_regime_override(pattern)

        return hypotheses

    def _bh_timeframe_divergence(self, pattern: MinedPattern) -> list[Hypothesis]:
        """BH active on 4h but dead on 1h → reduce position."""
        divergence_loss = pattern.metadata.get("avg_loss_when_divergent", -0.03)
        if abs(divergence_loss) < 0.01:
            return []

        return [Hypothesis(
            id=f"bh_tf_divergence_{pattern.symbol}",
            name="BH Cross-TF Divergence Exit",
            description=(
                f"When BH is active on 4h but fading on 1h in {pattern.symbol}, "
                f"exit 50% of position. Divergence has historically preceded "
                f"avg {divergence_loss:.1%} moves. Reduces whipsaw losses."
            ),
            hypothesis_type=HypothesisType.EXIT_RULE,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "fast_tf": "1h",
                "slow_tf": "4h",
                "divergence_threshold": 0.3,  # BH mass ratio
                "exit_fraction": 0.5,
                "reentry_on_realignment": True,
                "realignment_threshold": 0.7,
            },
            expected_impact=abs(divergence_loss) * 0.5,
            confidence=0.62,
            source_pattern=pattern,
            tags=["cross_timeframe", "black_hole", "exit"],
        )]

    def _multi_tf_alignment_entry(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Only enter when all timeframes aligned — higher win rate."""
        aligned_win_rate = pattern.metadata.get("aligned_win_rate", 0.55)
        unaligned_win_rate = pattern.metadata.get("unaligned_win_rate", 0.48)

        if aligned_win_rate - unaligned_win_rate < 0.05:
            return []

        return [Hypothesis(
            id=f"multi_tf_alignment_{pattern.symbol}",
            name="Multi-TF Alignment Entry Filter",
            description=(
                f"In {pattern.symbol}, only enter positions when 15m, 1h, AND 4h "
                f"BH signals agree in direction. Aligned entries show "
                f"{aligned_win_rate:.0%} win rate vs {unaligned_win_rate:.0%} unaligned."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "timeframes": ["15m", "1h", "4h"],
                "required_alignment": "all",  # or 'majority'
                "min_bh_mass_all_tf": 1.5,
                "direction_agreement": True,
                "size_multiplier_aligned": 1.2,
                "size_multiplier_unaligned": 0.0,
            },
            expected_impact=float(aligned_win_rate - 0.5) * 0.1,
            confidence=float(aligned_win_rate),
            source_pattern=pattern,
            tags=["cross_timeframe", "entry_filter", "multi_tf"],
        )]

    def _momentum_tf_divergence(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Price overbought on 15m but uptrending on 4h → hold, don't exit."""
        return [Hypothesis(
            id=f"momentum_tf_noise_{pattern.symbol}",
            name="Higher-TF Momentum Override Short-Term Noise",
            description=(
                f"When 15m shows overbought RSI but 4h trend is strong bullish, "
                f"override 15m exit signal in {pattern.symbol}. Short-term oscillations "
                f"are noise within the larger trend."
            ),
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "fast_tf_rsi_threshold": 75,
                "slow_tf_trend_min_strength": 0.6,
                "override_exit_within_bars": 8,
                "rsi_override_factor": 1.5,
            },
            expected_impact=0.015,
            confidence=0.58,
            source_pattern=pattern,
            tags=["cross_timeframe", "noise_filter", "momentum"],
        )]

    def _higher_tf_regime_override(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Higher-TF regime overrides lower-TF signal direction."""
        return [Hypothesis(
            id=f"htf_regime_override_{pattern.symbol}",
            name="Higher-TF Regime Overrides Entry Direction",
            description=(
                f"If 4h regime is BEAR in {pattern.symbol}, suppress all 15m long entries. "
                f"Trading against higher-TF regime has shown "
                f"{pattern.metadata.get('against_regime_win_rate', 0.40):.0%} win rate — below breakeven."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "regime_timeframe": "4h",
                "suppress_long_in_bear": True,
                "suppress_short_in_bull": True,
                "regime_lookback_bars": 48,
                "regime_ema_fast": 12,
                "regime_ema_slow": 48,
            },
            expected_impact=pattern.metadata.get("regime_filter_improvement", 0.02),
            confidence=0.70,
            source_pattern=pattern,
            tags=["cross_timeframe", "regime_filter", "direction"],
        )]
