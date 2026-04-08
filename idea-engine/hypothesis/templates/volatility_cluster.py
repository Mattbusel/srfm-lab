"""
Volatility clustering hypothesis template.

Generates hypotheses around volatility regime transitions:
  - Enter after vol spike subsides (mean-revert vol)
  - Exit before predicted vol expansion
  - Size down during high-VoV periods
  - Use vol term structure slope as entry filter
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from ..types import Hypothesis, MinedPattern, HypothesisType, HypothesisStatus


@dataclass
class VolatilityClusterTemplate:
    """Generates volatility cluster hypotheses from mined patterns."""

    name: str = "volatility_cluster"
    min_pattern_count: int = 20
    min_edge_ratio: float = 1.1

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Generate vol cluster hypotheses from a mined pattern."""
        hypotheses = []

        if pattern.pattern_type == "vol_spike":
            hypotheses += self._vol_spike_fade(pattern)
        elif pattern.pattern_type == "vol_compression":
            hypotheses += self._vol_compression_entry(pattern)
        elif pattern.pattern_type == "vol_regime_transition":
            hypotheses += self._vol_regime_transition(pattern)
        elif pattern.pattern_type == "vov_spike":
            hypotheses += self._vov_spike_caution(pattern)

        return hypotheses

    def _vol_spike_fade(self, pattern: MinedPattern) -> list[Hypothesis]:
        """After vol spike, fade back toward mean."""
        if pattern.count < self.min_pattern_count:
            return []

        win_rate = pattern.metadata.get("post_spike_win_rate", 0.5)
        avg_return = pattern.metadata.get("post_spike_avg_return", 0.0)

        if win_rate < 0.55:
            return []

        return [Hypothesis(
            id=f"vol_spike_fade_{pattern.symbol}_{pattern.timeframe}",
            name="Vol Spike Fade Entry",
            description=(
                f"After a vol spike (>2σ above rolling mean) in {pattern.symbol} "
                f"on {pattern.timeframe}, enter long as vol mean-reverts. "
                f"Observed {win_rate:.0%} win rate over {pattern.count} events."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "timeframe": pattern.timeframe,
                "vol_spike_threshold_z": 2.0,
                "entry_bars_after_spike": 2,
                "vol_mean_revert_threshold": 0.8,  # enter when vol drops to 80% of spike
                "stop_loss_pct": 0.05,
                "target_return_pct": avg_return * 2,
            },
            expected_impact=float(win_rate - 0.5) * float(avg_return),
            confidence=float(win_rate),
            source_pattern=pattern,
            tags=["volatility", "mean_reversion", "regime"],
        )]

    def _vol_compression_entry(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Vol compression (Bollinger squeeze) → breakout entry."""
        compression_ratio = pattern.metadata.get("vol_compression_ratio", 1.0)
        if compression_ratio > 0.5:  # need significant compression
            return []

        return [Hypothesis(
            id=f"vol_compression_{pattern.symbol}_{pattern.timeframe}",
            name="Vol Compression Breakout",
            description=(
                f"When realized vol in {pattern.symbol} compresses to bottom "
                f"{compression_ratio:.0%} of 1-year range, prepare for breakout entry. "
                f"Direction determined by BH mass direction."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "timeframe": pattern.timeframe,
                "vol_pct_threshold": 15,
                "vol_lookback_days": 252,
                "direction_from_bh": True,
                "entry_on_vol_expansion": True,
                "expansion_threshold_z": 1.5,
                "target_hold_bars": 20,
            },
            expected_impact=pattern.metadata.get("avg_breakout_return", 0.05),
            confidence=0.6,
            source_pattern=pattern,
            tags=["volatility", "breakout", "compression"],
        )]

    def _vol_regime_transition(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Reduce position during vol regime transition to high."""
        transition_prob = pattern.metadata.get("transition_probability", 0.5)
        if transition_prob < 0.6:
            return []

        return [Hypothesis(
            id=f"vol_regime_exit_{pattern.symbol}",
            name="Exit Before Vol Regime Shift",
            description=(
                f"When vol term structure inverts and VoV rises above 1.5σ, "
                f"reduce position by 50% in {pattern.symbol} to avoid vol regime transition losses."
            ),
            hypothesis_type=HypothesisType.EXIT_RULE,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "vov_threshold_z": 1.5,
                "term_structure_slope_threshold": -0.2,
                "position_reduction": 0.5,
                "reentry_vol_pct": 50,
            },
            expected_impact=pattern.metadata.get("avoided_loss", 0.02),
            confidence=float(transition_prob),
            source_pattern=pattern,
            tags=["volatility", "risk_management", "exit"],
        )]

    def _vov_spike_caution(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Reduce sizing when vol-of-vol is elevated."""
        return [Hypothesis(
            id=f"vov_size_down_{pattern.symbol}",
            name="VoV Spike Size Reduction",
            description=(
                f"When vol-of-vol (30-day std of daily vol) spikes above 2σ in "
                f"{pattern.symbol}, reduce position size by 30% to limit exposure "
                f"to unpredictable vol regime changes."
            ),
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "vov_z_threshold": 2.0,
                "size_multiplier": 0.7,
                "vol_window": 10,
                "vov_window": 30,
            },
            expected_impact=0.01,
            confidence=0.65,
            source_pattern=pattern,
            tags=["volatility", "sizing", "risk"],
        )]
