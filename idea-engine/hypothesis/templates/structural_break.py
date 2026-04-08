"""
Structural break hypothesis template.

Detects and generates hypotheses around structural breaks in:
  - Price autocorrelation structure
  - Vol clustering regime
  - Correlation between assets
  - BH mass equilibrium level
  - CF threshold recalibration needs
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from ..types import Hypothesis, MinedPattern, HypothesisType, HypothesisStatus


@dataclass
class StructuralBreakTemplate:
    name: str = "structural_break"

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        hypotheses = []

        if pattern.pattern_type == "autocorr_break":
            hypotheses += self._autocorr_regime_break(pattern)
        elif pattern.pattern_type == "corr_break":
            hypotheses += self._correlation_break(pattern)
        elif pattern.pattern_type == "vol_level_break":
            hypotheses += self._vol_level_break(pattern)
        elif pattern.pattern_type == "cf_drift":
            hypotheses += self._cf_threshold_recalibration(pattern)
        elif pattern.pattern_type == "mean_break":
            hypotheses += self._price_mean_break(pattern)

        return hypotheses

    def _autocorr_regime_break(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Return autocorrelation structure changed — recalibrate strategy."""
        old_ac = pattern.metadata.get("old_ac1", 0.0)
        new_ac = pattern.metadata.get("new_ac1", 0.0)
        break_date = pattern.metadata.get("break_date", "unknown")

        return [Hypothesis(
            id=f"autocorr_break_{pattern.symbol}_{break_date}",
            name="Autocorrelation Structural Break — Recalibrate",
            description=(
                f"Return autocorrelation in {pattern.symbol} shifted from "
                f"{old_ac:.3f} to {new_ac:.3f} around {break_date}. "
                f"This changes the optimal momentum lookback and exit timing."
            ),
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "old_optimal_lookback": pattern.metadata.get("old_lookback", 20),
                "new_optimal_lookback": pattern.metadata.get("new_lookback", 20),
                "ac_break_threshold": 0.1,
                "recalibrate_cf_threshold": True,
                "recalibrate_hold_time": True,
                "break_detection_window": 60,
            },
            expected_impact=pattern.metadata.get("recalibration_improvement", 0.02),
            confidence=0.65,
            source_pattern=pattern,
            tags=["structural_break", "autocorrelation", "recalibration"],
        )]

    def _correlation_break(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Cross-asset correlation changed — update portfolio construction."""
        asset_pair = pattern.metadata.get("asset_pair", [])
        old_corr = pattern.metadata.get("old_corr", 0.5)
        new_corr = pattern.metadata.get("new_corr", 0.5)

        if abs(new_corr - old_corr) < 0.2:
            return []

        return [Hypothesis(
            id=f"corr_break_{'_'.join(asset_pair)}",
            name=f"Correlation Break: {' / '.join(asset_pair)}",
            description=(
                f"Correlation between {asset_pair} broke from {old_corr:.2f} to {new_corr:.2f}. "
                f"Portfolio construction must update: "
                f"{'reduce combined exposure' if new_corr > old_corr else 'can increase combined exposure'}."
            ),
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            status=HypothesisStatus.PENDING,
            parameters={
                "asset_pair": asset_pair,
                "new_correlation_estimate": float(new_corr),
                "max_combined_exposure": float(1.0 / max(abs(new_corr), 0.3)),
                "correlation_update_halflife": 30,
                "use_rmt_cleaning": True,
            },
            expected_impact=abs(new_corr - old_corr) * 0.01,
            confidence=0.6,
            source_pattern=pattern,
            tags=["structural_break", "correlation", "portfolio"],
        )]

    def _vol_level_break(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Volatility level permanently shifted — recalibrate CF thresholds."""
        old_vol = pattern.metadata.get("old_vol", 0.02)
        new_vol = pattern.metadata.get("new_vol", 0.02)
        ratio = new_vol / max(old_vol, 1e-10)

        return [Hypothesis(
            id=f"vol_level_break_{pattern.symbol}",
            name="Vol Level Break — Recalibrate CF Thresholds",
            description=(
                f"Realized vol in {pattern.symbol} permanently shifted by {ratio:.1f}x. "
                f"CF thresholds need recalibration to maintain target timelike rate."
            ),
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "cf_15m_adjustment_factor": float(ratio),
                "cf_1h_adjustment_factor": float(ratio),
                "cf_4h_adjustment_factor": float(ratio),
                "recalibration_lookback_days": 90,
                "target_timelike_rate": 0.97,
                "trigger_recal_if_vol_shift": 0.3,
            },
            expected_impact=0.02,
            confidence=0.7,
            source_pattern=pattern,
            tags=["structural_break", "cf_threshold", "calibration", "black_hole"],
        )]

    def _cf_threshold_recalibration(self, pattern: MinedPattern) -> list[Hypothesis]:
        """CF threshold drift detected — price distribution shifted."""
        current_timelike_rate = pattern.metadata.get("current_timelike_rate", 0.90)
        target = 0.97

        if abs(current_timelike_rate - target) < 0.02:
            return []

        direction = "increase" if current_timelike_rate < target else "decrease"

        return [Hypothesis(
            id=f"cf_recal_{pattern.symbol}_{pattern.timeframe}",
            name=f"CF Threshold {direction.title()} — Timelike Rate Drift",
            description=(
                f"Current timelike rate for {pattern.symbol}/{pattern.timeframe} is "
                f"{current_timelike_rate:.1%}, target is {target:.1%}. "
                f"CF threshold must be {direction}d to maintain BH formation capability."
            ),
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "timeframe": pattern.timeframe,
                "current_cf": pattern.metadata.get("current_cf", 0.02),
                "required_cf": pattern.metadata.get("required_cf", 0.02),
                "target_timelike_rate": target,
                "recalibration_percentile": 97,
                "lookback_days": 90,
            },
            expected_impact=abs(current_timelike_rate - target) * 0.1,
            confidence=0.75,
            source_pattern=pattern,
            tags=["cf_threshold", "calibration", "black_hole", "structural_break"],
        )]

    def _price_mean_break(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Price mean shifted — update OU mean reversion targets."""
        old_mean = pattern.metadata.get("old_mean", 0.0)
        new_mean = pattern.metadata.get("new_mean", 0.0)

        return [Hypothesis(
            id=f"price_mean_break_{pattern.symbol}",
            name="Price Mean Shift — Update MR Targets",
            description=(
                f"Price mean in {pattern.symbol} shifted from {old_mean:.4f} to {new_mean:.4f}. "
                f"Mean reversion strategies must update their equilibrium targets."
            ),
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "new_equilibrium": float(new_mean),
                "mean_estimation_halflife": 60,
                "ou_kappa_update": True,
            },
            expected_impact=0.01,
            confidence=0.55,
            source_pattern=pattern,
            tags=["structural_break", "mean_reversion", "equilibrium"],
        )]
