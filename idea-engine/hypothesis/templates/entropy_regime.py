"""
Entropy-based regime hypothesis template.

Uses information-theoretic measures to generate hypotheses:
  - Low permutation entropy → predictable → increase size
  - High entropy → chaotic → reduce exposure
  - Transfer entropy → causal leading indicator
  - LZ complexity regime → market efficiency cycles
  - Complexity-entropy plane positioning → strategy selection
"""

from __future__ import annotations
from dataclasses import dataclass
from ..types import Hypothesis, MinedPattern, HypothesisType, HypothesisStatus


@dataclass
class EntropyRegimeTemplate:
    name: str = "entropy_regime"

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        hypotheses = []

        if pattern.pattern_type == "low_entropy_period":
            hypotheses += self._low_entropy_size_up(pattern)
        elif pattern.pattern_type == "high_entropy_period":
            hypotheses += self._high_entropy_risk_off(pattern)
        elif pattern.pattern_type == "transfer_entropy_signal":
            hypotheses += self._transfer_entropy_lead(pattern)
        elif pattern.pattern_type == "entropy_regime_transition":
            hypotheses += self._entropy_regime_transition(pattern)
        elif pattern.pattern_type == "complexity_plane":
            hypotheses += self._complexity_plane_strategy(pattern)

        return hypotheses

    def _low_entropy_size_up(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Low permutation entropy → predictable regime → size up."""
        low_pe_win_rate = pattern.metadata.get("win_rate_low_pe", 0.65)
        pe_threshold = pattern.metadata.get("pe_threshold", 0.4)

        if low_pe_win_rate < 0.55:
            return []

        return [Hypothesis(
            id=f"low_entropy_size_up_{pattern.symbol}",
            name="Low Entropy Regime — Increase Position Size",
            description=(
                f"When rolling 60-bar permutation entropy in {pattern.symbol} drops "
                f"below {pe_threshold:.2f} (normalized), market is in predictable regime. "
                f"Win rate increases to {low_pe_win_rate:.0%}. Size up by 1.3x."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "pe_window": 60,
                "pe_order": 4,
                "pe_low_threshold": float(pe_threshold),
                "size_multiplier_low_entropy": 1.3,
                "size_multiplier_high_entropy": 0.5,
                "pe_ema_smoothing": 10,
            },
            expected_impact=float(low_pe_win_rate - 0.5) * 0.05,
            confidence=float(low_pe_win_rate),
            source_pattern=pattern,
            tags=["entropy", "regime", "sizing", "information_theory"],
        )]

    def _high_entropy_risk_off(self, pattern: MinedPattern) -> list[Hypothesis]:
        """High entropy → chaotic/random → risk off."""
        high_pe_loss = pattern.metadata.get("avg_loss_high_pe", -0.03)
        pe_threshold = pattern.metadata.get("high_pe_threshold", 0.8)

        return [Hypothesis(
            id=f"high_entropy_risk_off_{pattern.symbol}",
            name="High Entropy Regime — Risk Off / Reduce Exposure",
            description=(
                f"When permutation entropy in {pattern.symbol} exceeds {pe_threshold:.2f}, "
                f"market enters chaotic regime. Strategies based on pattern recognition fail. "
                f"Reduce all positions by 60% until entropy normalizes."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "pe_high_threshold": float(pe_threshold),
                "position_scale_in_chaos": 0.4,
                "entropy_window": 60,
                "normalize_exit_threshold": 0.65,
                "spectral_entropy_confirm": True,
            },
            expected_impact=abs(high_pe_loss) * 0.6,
            confidence=0.65,
            source_pattern=pattern,
            tags=["entropy", "risk_off", "chaos", "regime"],
        )]

    def _transfer_entropy_lead(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Asset A has positive TE → asset B → use A as leading indicator for B."""
        te_value = pattern.metadata.get("transfer_entropy", 0.0)
        lead_lag = pattern.metadata.get("optimal_lag", 1)
        source = pattern.metadata.get("source_asset", "BTC")
        target = pattern.metadata.get("target_asset", pattern.symbol)
        predictability = pattern.metadata.get("predictability_r2", 0.0)

        if te_value < 0.01 or predictability < 0.05:
            return []

        return [Hypothesis(
            id=f"te_lead_{source}_to_{target}",
            name=f"Transfer Entropy Lead: {source} → {target}",
            description=(
                f"{source} has transfer entropy {te_value:.3f} bits toward {target} "
                f"at lag {lead_lag} bars (R²={predictability:.2f}). "
                f"Use {source} signal {lead_lag} bars early as entry confirmation for {target}."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "source_asset": source,
                "target_asset": target,
                "lead_lag_bars": int(lead_lag),
                "te_threshold": float(te_value * 0.5),
                "confirmation_weight": float(min(predictability * 2, 0.4)),
                "te_window": 100,
                "te_n_bins": 8,
            },
            expected_impact=float(predictability * 0.02),
            confidence=float(min(te_value * 10, 0.7)),
            source_pattern=pattern,
            tags=["transfer_entropy", "lead_lag", "information_theory", "cross_asset"],
        )]

    def _entropy_regime_transition(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Entropy transitioning from low to high → exit before chaos."""
        return [Hypothesis(
            id=f"entropy_transition_exit_{pattern.symbol}",
            name="Entropy Rising — Pre-emptive Position Reduction",
            description=(
                f"When permutation entropy in {pattern.symbol} rises by >0.15 over "
                f"10 bars (entropy acceleration), exit 30% of position proactively. "
                f"Entropy acceleration predicts regime change within 5-20 bars."
            ),
            hypothesis_type=HypothesisType.EXIT_RULE,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "pe_acceleration_threshold": 0.15,
                "pe_acceleration_window": 10,
                "preemptive_exit_fraction": 0.3,
                "full_exit_if_entropy_continues": True,
                "entropy_continuation_bars": 5,
            },
            expected_impact=0.015,
            confidence=0.58,
            source_pattern=pattern,
            tags=["entropy", "regime_transition", "proactive_exit"],
        )]

    def _complexity_plane_strategy(self, pattern: MinedPattern) -> list[Hypothesis]:
        """Complexity-entropy plane positioning determines strategy type."""
        H = pattern.metadata.get("entropy_H", 0.5)
        C = pattern.metadata.get("complexity_C", 0.1)

        if H > 0.7 and C < 0.1:
            # Near random walk → low predictability
            strategy = "reduce"
            description = "Near random walk region — reduce directional exposure, consider volatility plays."
            size_mult = 0.4
        elif H < 0.4 and C > 0.2:
            # Deterministic chaos → complex but structured
            strategy = "momentum"
            description = "Deterministic structure detected — momentum strategies likely to work."
            size_mult = 1.2
        elif H < 0.3 and C < 0.15:
            # Highly regular → periodic
            strategy = "cycle"
            description = "Highly regular regime — cyclic/mean-reversion strategies optimal."
            size_mult = 1.3
        else:
            return []

        return [Hypothesis(
            id=f"complexity_plane_{pattern.symbol}_{strategy}",
            name=f"Complexity-Entropy Plane: {strategy.title()} Strategy",
            description=description + f" (H={H:.2f}, C={C:.2f})",
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": pattern.symbol,
                "H_value": float(H),
                "C_value": float(C),
                "strategy_type": strategy,
                "size_multiplier": float(size_mult),
                "pe_order": 4,
                "update_frequency_bars": 20,
            },
            expected_impact=abs(size_mult - 1.0) * 0.02,
            confidence=0.60,
            source_pattern=pattern,
            tags=["entropy", "complexity", "strategy_selection", "information_theory"],
        )]
