"""
Correlation regime hypothesis template.

Detects and generates hypotheses around correlation structure changes:
  - Correlation spike (crisis = all correlations → 1)
  - Decorrelation (altseason = idio-alpha rises)
  - Broken correlation pairs (regime change in relationships)
  - RMT-cleaned signal eigenvalue shifts
  - BTC-alt correlation divergence
"""

from __future__ import annotations
from dataclasses import dataclass
from ..types import Hypothesis, MinedPattern, HypothesisType, HypothesisStatus


@dataclass
class CorrelationRegimeTemplate:
    name: str = "correlation_regime"

    def generate(self, pattern: MinedPattern) -> list[Hypothesis]:
        hypotheses = []

        if pattern.pattern_type == "correlation_spike":
            hypotheses += self._correlation_crisis(pattern)
        elif pattern.pattern_type == "decorrelation":
            hypotheses += self._decorrelation_opportunity(pattern)
        elif pattern.pattern_type == "correlation_break_pair":
            hypotheses += self._correlation_pair_break(pattern)
        elif pattern.pattern_type == "rmt_signal_collapse":
            hypotheses += self._rmt_signal_collapse(pattern)

        return hypotheses

    def _correlation_crisis(self, pattern: MinedPattern) -> list[Hypothesis]:
        avg_corr = pattern.metadata.get("avg_correlation", 0.8)
        return [Hypothesis(
            id=f"correlation_crisis_{pattern.symbol}",
            name="Correlation Crisis — Reduce Portfolio to Single Position",
            description=(
                f"Average pairwise crypto correlation reached {avg_corr:.2f} "
                f"(crisis level > 0.75). Diversification illusory — holding multiple "
                f"positions is equivalent to concentrated BTC exposure. "
                f"Reduce to 1-2 highest-conviction positions during correlation crisis."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "correlation_crisis_threshold": 0.75,
                "correlation_window": 20,
                "max_positions_in_crisis": 2,
                "preferred_crisis_asset": "BTC",
                "correlation_recovery_threshold": 0.55,
            },
            expected_impact=pattern.metadata.get("crisis_loss_avoided", 0.04),
            confidence=0.70,
            source_pattern=pattern,
            tags=["correlation", "crisis", "portfolio", "concentration"],
        )]

    def _decorrelation_opportunity(self, pattern: MinedPattern) -> list[Hypothesis]:
        avg_corr = pattern.metadata.get("avg_correlation", 0.3)
        idio_alpha = pattern.metadata.get("avg_idio_alpha", 0.02)
        return [Hypothesis(
            id=f"decorrelation_opp_{pattern.symbol}",
            name="Decorrelation Regime — Expand Portfolio Diversity",
            description=(
                f"Average pairwise crypto correlation dropped to {avg_corr:.2f}. "
                f"Idiosyncratic alpha {idio_alpha:.1%} per trade available. "
                f"Expand to 6-8 positions — genuine diversification benefit. "
                f"RMT analysis shows {pattern.metadata.get('n_signal_eigenvalues', 3)} "
                f"independent signal dimensions."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "decorrelation_threshold": 0.40,
                "max_positions_in_decorrelated": 8,
                "use_rmt_portfolio": True,
                "rmt_n_signal_eigs": pattern.metadata.get("n_signal_eigenvalues", 3),
                "correlation_window": 20,
            },
            expected_impact=float(idio_alpha * 4),
            confidence=0.65,
            source_pattern=pattern,
            tags=["correlation", "decorrelation", "portfolio", "diversification"],
        )]

    def _correlation_pair_break(self, pattern: MinedPattern) -> list[Hypothesis]:
        pair = pattern.metadata.get("asset_pair", ["BTC", "ETH"])
        old_corr = pattern.metadata.get("old_corr", 0.8)
        new_corr = pattern.metadata.get("new_corr", 0.3)
        return [Hypothesis(
            id=f"corr_pair_break_{'_'.join(pair)}",
            name=f"Correlation Break: {' / '.join(pair)} Now Independent",
            description=(
                f"Correlation between {pair} dropped from {old_corr:.2f} to {new_corr:.2f}. "
                f"These assets are no longer moving together. "
                f"Pairs trading / stat arb opportunity: bet on eventual reconvergence, "
                f"or treat as independent signals (don't skip one because of the other)."
            ),
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            status=HypothesisStatus.PENDING,
            parameters={
                "asset_pair": pair,
                "reconvergence_half_life_days": 30,
                "mean_revert_z_entry": 2.0,
                "pairs_trade_enabled": True if abs(new_corr - old_corr) > 0.3 else False,
                "treat_as_independent_signals": True,
            },
            expected_impact=abs(old_corr - new_corr) * 0.02,
            confidence=0.58,
            source_pattern=pattern,
            tags=["correlation", "pairs_trading", "stat_arb", "structural_break"],
        )]

    def _rmt_signal_collapse(self, pattern: MinedPattern) -> list[Hypothesis]:
        old_n_signal = pattern.metadata.get("old_n_signal", 5)
        new_n_signal = pattern.metadata.get("new_n_signal", 2)
        return [Hypothesis(
            id=f"rmt_signal_collapse_{pattern.symbol}",
            name="RMT Signal Eigenvalue Collapse — Reduce Complexity",
            description=(
                f"Random matrix theory analysis shows number of signal eigenvalues "
                f"dropped from {old_n_signal} to {new_n_signal}. "
                f"Portfolio is becoming more concentrated — genuine diversification limited. "
                f"Simplify portfolio to {new_n_signal} uncorrelated positions."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "max_positions": int(new_n_signal),
                "rmt_recalculation_window": 60,
                "rmt_alpha": 1.0,
                "rmt_method": "clip",
                "reduce_on_signal_collapse": True,
            },
            expected_impact=0.02,
            confidence=0.65,
            source_pattern=pattern,
            tags=["rmt", "correlation", "portfolio", "concentration"],
        )]
