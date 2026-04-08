"""
Momentum exhaustion hypothesis template.

Detects when momentum is running out of steam:
  - Price acceleration declining (jerk < 0)
  - Volume-price divergence (price up, volume declining)
  - Breadth exhaustion (fewer coins participating)
  - BH mass plateau (mass growth slowing)
  - RSI overbought + vol spike → reversal
"""

from __future__ import annotations
from dataclasses import dataclass
from ..types import Hypothesis, MinedPattern, HypothesisType, HypothesisStatus


@dataclass
class MomentumExhaustionTemplate:
    name: str = "momentum_exhaustion"

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        hypotheses = []

        if pattern.pattern_type == "price_acceleration_decline":
            hypotheses += self._jerk_reversal(pattern)
        elif pattern.pattern_type == "vol_price_divergence":
            hypotheses += self._volume_price_divergence(pattern)
        elif pattern.pattern_type == "bh_mass_plateau":
            hypotheses += self._bh_mass_plateau_exit(pattern)
        elif pattern.pattern_type == "breadth_exhaustion":
            hypotheses += self._breadth_exhaustion(pattern)
        elif pattern.pattern_type == "rsi_divergence":
            hypotheses += self._rsi_divergence_exit(pattern)

        return hypotheses

    def _jerk_reversal(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Negative price jerk (acceleration turning down) → exit signal."""
        jerk_reversal_rate = pattern.metadata.get("reversal_rate", 0.6)
        avg_subsequent = pattern.metadata.get("avg_subsequent_return", -0.02)

        if jerk_reversal_rate < 0.55:
            return []

        return [Hypothesis(
            id=f"jerk_reversal_{pattern.symbol}",
            name="Price Jerk Reversal Exit",
            description=(
                f"When price acceleration (d²p/dt²) turns negative for 3 consecutive bars "
                f"in {pattern.symbol} while in profit, exit {pattern.metadata.get('exit_fraction', 0.5):.0%} "
                f"of position. Seen reversal {jerk_reversal_rate:.0%} of the time, "
                f"avg subsequent move {avg_subsequent:.1%}."
            ),
            hypothesis_type=HypothesisType.EXIT_RULE,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "jerk_negative_bars": 3,
                "min_profit_to_apply": 0.02,
                "exit_fraction": pattern.metadata.get("exit_fraction", 0.5),
                "jerk_smoothing_window": 3,
                "price_jerk_z_threshold": -1.0,
            },
            expected_impact=abs(avg_subsequent) * pattern.metadata.get("exit_fraction", 0.5),
            confidence=float(jerk_reversal_rate),
            source_pattern=pattern,
            tags=["momentum_exhaustion", "jerk", "exit"],
        )]

    def _volume_price_divergence(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Price rising but volume declining → exhaustion."""
        divergence_predictability = pattern.metadata.get("divergence_predictability", 0.58)

        return [Hypothesis(
            id=f"vol_price_diverge_{pattern.symbol}",
            name="Volume-Price Divergence Exit Signal",
            description=(
                f"When {pattern.symbol} price rises >3% over 10 bars but volume "
                f"is in lower 30th percentile of 30-day range, momentum exhaustion likely. "
                f"Exit partial position ({divergence_predictability:.0%} historical accuracy)."
            ),
            hypothesis_type=HypothesisType.EXIT_RULE,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "price_move_threshold": 0.03,
                "price_lookback_bars": 10,
                "volume_pct_threshold": 30,
                "volume_lookback_bars": 30,
                "exit_fraction_on_divergence": 0.4,
            },
            expected_impact=float(divergence_predictability - 0.5) * 0.03,
            confidence=float(divergence_predictability),
            source_pattern=pattern,
            tags=["momentum_exhaustion", "volume", "divergence"],
        )]

    def _bh_mass_plateau_exit(self, pattern: MinedPattern) -> list[Hypothesis]:
        """BH mass stops growing → momentum engine stalling."""
        plateau_loss = pattern.metadata.get("avg_loss_after_plateau", -0.025)

        return [Hypothesis(
            id=f"bh_plateau_exit_{pattern.symbol}",
            name="BH Mass Plateau Exit Trigger",
            description=(
                f"When BH mass in {pattern.symbol} stops growing for 8+ consecutive bars "
                f"(<0.1% change per bar), reduce position by 30%. Mass plateau signals "
                f"momentum stall, often precedes {plateau_loss:.1%} avg drawdown."
            ),
            hypothesis_type=HypothesisType.EXIT_RULE,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "plateau_bars": 8,
                "mass_growth_threshold": 0.001,
                "exit_fraction": 0.3,
                "full_exit_if_mass_declining": True,
                "mass_decline_threshold": -0.01,
            },
            expected_impact=abs(plateau_loss) * 0.3,
            confidence=0.62,
            source_pattern=pattern,
            tags=["momentum_exhaustion", "black_hole", "mass_plateau"],
        )]

    def _breadth_exhaustion(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Fewer assets in uptrend → narrow market → fragile momentum."""
        breadth_threshold = pattern.metadata.get("breadth_reversal_threshold", 0.3)

        return [Hypothesis(
            id=f"breadth_exhaustion_{pattern.symbol}",
            name="Market Breadth Exhaustion Filter",
            description=(
                f"When market breadth (fraction of crypto top-20 in uptrend) "
                f"drops below {breadth_threshold:.0%} while holding {pattern.symbol}, "
                f"apply 50% size reduction. Narrow leadership = fragile rally."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "reference_assets": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX"],
                "breadth_threshold": float(breadth_threshold),
                "breadth_lookback_bars": 10,
                "size_multiplier_low_breadth": 0.5,
                "breadth_calculation": "pct_above_20ema",
            },
            expected_impact=0.015,
            confidence=0.6,
            source_pattern=pattern,
            tags=["breadth", "momentum_exhaustion", "portfolio"],
        )]

    def _rsi_divergence_exit(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Price making new highs but RSI declining → bearish divergence."""
        div_success_rate = pattern.metadata.get("divergence_accuracy", 0.62)

        return [Hypothesis(
            id=f"rsi_bearish_divergence_{pattern.symbol}",
            name="Bearish RSI Divergence Exit",
            description=(
                f"When {pattern.symbol} price makes new 20-bar high but RSI "
                f"is below its own 20-bar high, bearish divergence detected. "
                f"Exit 40% of position. Accuracy: {div_success_rate:.0%}."
            ),
            hypothesis_type=HypothesisType.EXIT_RULE,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "rsi_period": 14,
                "divergence_lookback": 20,
                "min_rsi_divergence": 3.0,
                "exit_fraction": 0.4,
                "min_profit_to_apply": 0.01,
            },
            expected_impact=float(div_success_rate - 0.5) * 0.03,
            confidence=float(div_success_rate),
            source_pattern=pattern,
            tags=["rsi", "divergence", "momentum_exhaustion", "exit"],
        )]
