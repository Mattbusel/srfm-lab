"""
Funding rate hypothesis template for crypto perpetuals.

Generates hypotheses based on perpetual futures funding rates:
  - Extreme positive funding → short squeeze setup
  - Extreme negative funding → long squeeze setup
  - Funding rate divergence across exchanges → arbitrage
  - Funding rate trend as sentiment indicator
  - Decay after funding rate spike → mean reversion
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from ..types import Hypothesis, MinedPattern, HypothesisType, HypothesisStatus


@dataclass
class FundingRateSignalTemplate:
    name: str = "funding_rate_signal"

    EXTREME_POSITIVE_THRESHOLD: float = 0.001    # 0.1% per 8h = ~110% annualized
    EXTREME_NEGATIVE_THRESHOLD: float = -0.0005

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        hypotheses = []

        if pattern.pattern_type == "funding_extreme_positive":
            hypotheses += self._extreme_positive_funding(pattern)
        elif pattern.pattern_type == "funding_extreme_negative":
            hypotheses += self._extreme_negative_funding(pattern)
        elif pattern.pattern_type == "funding_rate_divergence":
            hypotheses += self._funding_divergence(pattern)
        elif pattern.pattern_type == "funding_spike_decay":
            hypotheses += self._funding_spike_decay(pattern)
        elif pattern.pattern_type == "funding_trend":
            hypotheses += self._funding_trend_momentum(pattern)

        return hypotheses

    def _extreme_positive_funding(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Extreme positive funding → longs paying heavily → mean reversion risk."""
        freq = pattern.metadata.get("reversion_frequency", 0.6)
        avg_reversion = pattern.metadata.get("avg_reversion_magnitude", 0.05)

        return [Hypothesis(
            id=f"funding_extreme_pos_{pattern.symbol}",
            name="Extreme Positive Funding Mean Reversion",
            description=(
                f"When {pattern.symbol} 8h funding rate exceeds {self.EXTREME_POSITIVE_THRESHOLD:.3%}, "
                f"longs are paying unsustainably high rates. Historical {freq:.0%} of such events "
                f"see price pullback within 24h (avg {avg_reversion:.1%}). "
                f"Consider exit or short entry with tight stop."
            ),
            hypothesis_type=HypothesisType.EXIT_RULE,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "funding_long_exit_threshold": self.EXTREME_POSITIVE_THRESHOLD,
                "funding_short_entry_threshold": self.EXTREME_POSITIVE_THRESHOLD * 1.5,
                "stop_loss_pct": 0.03,
                "target_reversion_pct": avg_reversion * 0.6,
                "max_hold_funding_periods": 3,
                "position_cap_in_extreme_funding": 0.5,
            },
            expected_impact=float(freq - 0.5) * avg_reversion,
            confidence=float(freq),
            source_pattern=pattern,
            tags=["funding_rate", "mean_reversion", "crypto", "sentiment"],
        )]

    def _extreme_negative_funding(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Extreme negative funding → shorts squeezed → bounce setup."""
        freq = pattern.metadata.get("bounce_frequency", 0.58)
        avg_bounce = pattern.metadata.get("avg_bounce_magnitude", 0.04)

        return [Hypothesis(
            id=f"funding_extreme_neg_{pattern.symbol}",
            name="Extreme Negative Funding Bounce Entry",
            description=(
                f"When {pattern.symbol} funding rate drops below {self.EXTREME_NEGATIVE_THRESHOLD:.3%}, "
                f"shorts paying maximum pain. Potential long entry if BH still forming. "
                f"Avg bounce {avg_bounce:.1%} seen {freq:.0%} of the time."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "funding_rate_threshold": self.EXTREME_NEGATIVE_THRESHOLD,
                "require_bh_mass_min": 1.5,
                "entry_size_multiplier": 0.8,
                "stop_loss_pct": 0.04,
                "target_pct": avg_bounce * 0.5,
            },
            expected_impact=float(freq - 0.5) * avg_bounce,
            confidence=float(freq),
            source_pattern=pattern,
            tags=["funding_rate", "contrarian", "long", "crypto"],
        )]

    def _funding_divergence(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Cross-exchange funding divergence → basis arbitrage."""
        div_magnitude = pattern.metadata.get("avg_divergence", 0.0002)

        return [Hypothesis(
            id=f"funding_divergence_{pattern.symbol}",
            name="Cross-Exchange Funding Rate Divergence",
            description=(
                f"When funding rates diverge across exchanges for {pattern.symbol} "
                f"by >{div_magnitude:.3%}, basis arbitrage opportunity exists. "
                f"Long lower-funding exchange, short higher-funding exchange."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "min_divergence_threshold": div_magnitude,
                "exchanges": ["binance", "bybit", "okx"],
                "hedge_ratio": 1.0,
                "max_basis_risk": 0.005,
                "unwind_on_convergence": True,
            },
            expected_impact=float(div_magnitude * 3 * 365 / 3),  # annualized
            confidence=0.7,
            source_pattern=pattern,
            tags=["funding_rate", "arbitrage", "basis", "crypto"],
        )]

    def _funding_spike_decay(self, pattern: MinedPattern) -> list[Hypothesis]:
        """After funding spike, hold shorter — decay accelerates deleveraging."""
        return [Hypothesis(
            id=f"funding_spike_hold_reduce_{pattern.symbol}",
            name="Reduce Hold Time After Funding Spike",
            description=(
                f"When funding rate spikes during a {pattern.symbol} position, "
                f"reduce max hold time by 30%. Funding spikes often precede rapid "
                f"deleveraging events that compress gains."
            ),
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "funding_spike_z": 2.0,
                "hold_time_multiplier": 0.7,
                "target_pct_reduction": 0.8,
            },
            expected_impact=0.01,
            confidence=0.6,
            source_pattern=pattern,
            tags=["funding_rate", "hold_time", "risk"],
        )]

    def _funding_trend_momentum(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Rising funding rate trend confirms bullish momentum."""
        trend_corr = pattern.metadata.get("funding_price_correlation", 0.3)

        if abs(trend_corr) < 0.2:
            return []

        return [Hypothesis(
            id=f"funding_momentum_{pattern.symbol}",
            name="Funding Rate Trend as Momentum Confirmation",
            description=(
                f"Rising funding rate in {pattern.symbol} correlates "
                f"r={trend_corr:.2f} with subsequent price momentum. "
                f"Use as secondary entry confirmation alongside BH signal."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "funding_trend_lookback_periods": 6,
                "min_funding_trend_z": 1.0,
                "confirmation_weight": 0.2,
                "direction_confirmation_only": True,
            },
            expected_impact=float(trend_corr * 0.02),
            confidence=float(min(abs(trend_corr) + 0.3, 0.75)),
            source_pattern=pattern,
            tags=["funding_rate", "momentum", "confirmation"],
        )]
