"""
onchain/hypothesis_generator.py
─────────────────────────────────
Convert on-chain signals to IAE Hypotheses.

Hypothesis generation logic
────────────────────────────
Extreme on-chain readings are among the most reliable macro signals in crypto.
When the composite score exceeds a threshold, we generate a high-confidence
IAE hypothesis with specific parameter delta suggestions:

  Composite > +0.7  → STRONG ACCUMULATION
    → Hypothesis: increase position sizing, loosen stop, extend hold period
    → Predicted Sharpe delta: +0.15 to +0.25

  Composite < -0.7  → STRONG DISTRIBUTION
    → Hypothesis: reduce position sizing, tighten stop, reduce max_loss_bars
    → Predicted Sharpe delta: +0.10 to +0.20 (by avoiding the drawdown)

  Individual metric extremes also trigger targeted hypotheses:
    MVRV > 7 or < 0.8 → cycle-level conviction signals
    SOPR capitulation  → short-term reversal hypothesis
    Hash rate ribbon golden cross → strong accumulation signal
    Exchange reserves 30d outflow > 5% → supply shock hypothesis
    Whale flow 30d > 5000 BTC accumulation → institutional buying

Confidence is calibrated to the z-score distance from historical mean:
  |composite| > 0.9 → confidence = 0.90
  |composite| > 0.7 → confidence = 0.75
  |composite| > 0.5 → confidence = 0.60
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import List

from .composite_signal import OnChainResult

# Import IAE types from the shared hypothesis module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from hypothesis.types import Hypothesis, HypothesisType, HypothesisStatus

logger = logging.getLogger(__name__)

# Thresholds for hypothesis generation
_COMPOSITE_HIGH_THRESHOLD  =  0.70
_COMPOSITE_LOW_THRESHOLD   = -0.70
_MVRV_OVERHEATED           =  7.0
_MVRV_UNDERVALUED          =  0.8
_SOPR_CAPITULATION_SMOOTH  =  0.93
_HASH_RATE_GOLDEN_CROSS    = "golden"
_EXCHANGE_OUTFLOW_ALERT    = -0.05   # 30d RoC < -5%
_WHALE_ACCUM_ALERT_30D     =  4000   # BTC
_WHALE_DISTRIB_ALERT_30D   = -4000   # BTC


class OnChainHypothesisGenerator:
    """Generates IAE hypotheses from on-chain signal extremes.

    Parameters
    ----------
    composite_threshold:
        Minimum |composite_score| to trigger a composite-level hypothesis.
    """

    def __init__(self, composite_threshold: float = 0.70) -> None:
        self.composite_threshold = composite_threshold

    def generate(self, result: OnChainResult) -> List[Hypothesis]:
        """Generate hypotheses from an OnChainResult.

        Returns a (potentially empty) list of Hypothesis objects ready for
        insertion into the IAE hypothesis queue.
        """
        hypotheses: List[Hypothesis] = []

        # 1. Composite-level hypothesis
        if abs(result.composite_score) >= self.composite_threshold:
            h = self._composite_hypothesis(result)
            if h:
                hypotheses.append(h)

        # 2. MVRV extreme
        if result.mvrv is not None:
            h = self._mvrv_hypothesis(result)
            if h:
                hypotheses.append(h)

        # 3. SOPR capitulation
        if result.sopr is not None and result.sopr.is_capitulation:
            hypotheses.append(self._sopr_capitulation_hypothesis(result))

        # 4. Hash rate ribbon golden cross (BTC only)
        if result.hash_rate is not None and result.hash_rate.ribbon_crossover == _HASH_RATE_GOLDEN_CROSS:
            hypotheses.append(self._hash_rate_golden_cross_hypothesis(result))

        # 5. Exchange reserve supply shock
        if result.exchange_reserves is not None:
            h = self._exchange_reserve_hypothesis(result)
            if h:
                hypotheses.append(h)

        # 6. Whale flow extreme
        if result.whale is not None:
            h = self._whale_flow_hypothesis(result)
            if h:
                hypotheses.append(h)

        logger.info(
            "OnChainHypothesisGenerator: %d hypotheses generated for %s (score=%.3f)",
            len(hypotheses), result.symbol, result.composite_score,
        )
        return hypotheses

    # ── Individual generators ───────────────────────────────────────────────

    def _composite_hypothesis(self, result: OnChainResult) -> Hypothesis | None:
        score = result.composite_score
        if abs(score) < self.composite_threshold:
            return None

        bullish = score > 0
        confidence = _score_to_confidence(abs(score))
        sharpe_delta = round(confidence * 0.25, 3)
        dd_delta     = round(-confidence * 0.05, 3) if bullish else round(-confidence * 0.08, 3)

        if bullish:
            params = {
                "position_size_multiplier": round(1.0 + confidence * 0.40, 2),  # e.g. 1.30
                "atr_stop_mult":            round(2.0 + confidence * 0.5,  2),  # wider stop
                "max_loss_bars":            int(20 + confidence * 20),           # hold longer
                "onchain_score":            result.composite_score,
                "regime":                   result.regime_label,
                "rationale":                "On-chain accumulation composite extreme — historically precedes price appreciation",
            }
            description = (
                f"ON-CHAIN ACCUMULATION: composite={result.composite_score:+.3f} [{result.regime_label}]. "
                f"Signals: {_format_component_signals(result.component_signals)}. "
                f"Suggested: increase position sizing to {params['position_size_multiplier']:.2f}x, "
                f"widen ATR stop to {params['atr_stop_mult']:.1f}."
            )
        else:
            params = {
                "position_size_multiplier": round(1.0 - confidence * 0.50, 2),  # e.g. 0.65
                "atr_stop_mult":            round(1.5 - confidence * 0.3,  2),  # tighter stop
                "max_loss_bars":            int(10 + confidence * 5),            # exit faster
                "onchain_score":            result.composite_score,
                "regime":                   result.regime_label,
                "rationale":                "On-chain distribution composite extreme — historically precedes drawdowns",
            }
            description = (
                f"ON-CHAIN DISTRIBUTION: composite={result.composite_score:+.3f} [{result.regime_label}]. "
                f"Signals: {_format_component_signals(result.component_signals)}. "
                f"Suggested: reduce position sizing to {params['position_size_multiplier']:.2f}x, "
                f"tighten ATR stop to {params['atr_stop_mult']:.1f}."
            )

        return Hypothesis.create(
            hypothesis_type=HypothesisType.REGIME_FILTER,
            parent_pattern_id=f"onchain_composite_{result.symbol}",
            parameters=params,
            predicted_sharpe_delta=sharpe_delta,
            predicted_dd_delta=dd_delta,
            novelty_score=min(1.0, abs(score)),
            description=description,
        )

    def _mvrv_hypothesis(self, result: OnChainResult) -> Hypothesis | None:
        if result.mvrv is None:
            return None
        ratio = result.mvrv.mvrv_ratio
        if ratio > _MVRV_OVERHEATED:
            params = {
                "position_size_multiplier": 0.40,
                "mvrv_ratio":               ratio,
                "rationale":                f"MVRV={ratio:.2f} > 7 — historically marks cycle tops",
            }
            return Hypothesis.create(
                hypothesis_type=HypothesisType.REGIME_FILTER,
                parent_pattern_id=f"onchain_mvrv_{result.symbol}",
                parameters=params,
                predicted_sharpe_delta=0.18,
                predicted_dd_delta=-0.12,
                novelty_score=0.85,
                description=f"MVRV OVERHEATED: ratio={ratio:.2f} > 7. Reduce sizing to 40% — cycle top zone.",
            )
        if ratio < _MVRV_UNDERVALUED:
            params = {
                "position_size_multiplier": 1.50,
                "mvrv_ratio":               ratio,
                "rationale":                f"MVRV={ratio:.2f} < 0.8 — historically marks cycle bottoms",
            }
            return Hypothesis.create(
                hypothesis_type=HypothesisType.REGIME_FILTER,
                parent_pattern_id=f"onchain_mvrv_{result.symbol}",
                parameters=params,
                predicted_sharpe_delta=0.30,
                predicted_dd_delta=-0.05,
                novelty_score=0.90,
                description=f"MVRV UNDERVALUED: ratio={ratio:.2f} < 0.8. Increase sizing to 150% — cycle bottom zone.",
            )
        return None

    def _sopr_capitulation_hypothesis(self, result: OnChainResult) -> Hypothesis:
        sopr_val = result.sopr.sopr_smooth if result.sopr else 0.95
        params = {
            "position_size_multiplier": 1.30,
            "max_loss_bars":            15,
            "atr_stop_mult":            1.8,
            "sopr_smooth":              sopr_val,
            "rationale":                "SOPR capitulation: sellers realising losses = exhaustion signal",
        }
        return Hypothesis.create(
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            parent_pattern_id=f"onchain_sopr_cap_{result.symbol}",
            parameters=params,
            predicted_sharpe_delta=0.20,
            predicted_dd_delta=-0.03,
            novelty_score=0.80,
            description=(
                f"SOPR CAPITULATION: smoothed SOPR={sopr_val:.3f} < 1 for 3+ days. "
                f"Weak hands exhausted. Increase sizing 30%, hold for reversal."
            ),
        )

    def _hash_rate_golden_cross_hypothesis(self, result: OnChainResult) -> Hypothesis:
        ribbon = result.hash_rate.ribbon_ratio if result.hash_rate else 1.0
        params = {
            "position_size_multiplier": 1.40,
            "atr_stop_mult":            2.2,
            "max_loss_bars":            25,
            "hash_rate_ribbon":         ribbon,
            "rationale":                "Hash rate ribbon golden cross after miner capitulation — historically very strong BTC buy signal",
        }
        return Hypothesis.create(
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            parent_pattern_id=f"onchain_hashribbon_{result.symbol}",
            parameters=params,
            predicted_sharpe_delta=0.28,
            predicted_dd_delta=-0.04,
            novelty_score=0.88,
            description=(
                f"HASH RATE RIBBON GOLDEN CROSS: ribbon={ribbon:.3f}. "
                f"Miner capitulation resolved. Historically strong accumulation signal. "
                f"Increase sizing to 1.40x, hold min 25 bars."
            ),
        )

    def _exchange_reserve_hypothesis(self, result: OnChainResult) -> Hypothesis | None:
        if result.exchange_reserves is None:
            return None
        roc = result.exchange_reserves.roc_30d
        if roc > 0.05:
            params = {
                "position_size_multiplier": 0.70,
                "exchange_roc_30d":         roc,
                "rationale":                f"Exchange reserves +{roc:.1%} in 30d — distribution pressure",
            }
            return Hypothesis.create(
                hypothesis_type=HypothesisType.REGIME_FILTER,
                parent_pattern_id=f"onchain_exr_{result.symbol}",
                parameters=params,
                predicted_sharpe_delta=0.12,
                predicted_dd_delta=-0.07,
                novelty_score=0.70,
                description=f"EXCHANGE INFLOW SURGE: +{roc:.1%} 30d. Coins entering exchanges. Reduce sizing to 70%.",
            )
        if roc < _EXCHANGE_OUTFLOW_ALERT:
            params = {
                "position_size_multiplier": 1.25,
                "exchange_roc_30d":         roc,
                "rationale":                f"Exchange reserves {roc:.1%} in 30d — supply shock accumulation",
            }
            return Hypothesis.create(
                hypothesis_type=HypothesisType.REGIME_FILTER,
                parent_pattern_id=f"onchain_exr_{result.symbol}",
                parameters=params,
                predicted_sharpe_delta=0.16,
                predicted_dd_delta=-0.03,
                novelty_score=0.72,
                description=f"EXCHANGE OUTFLOW SURGE: {roc:.1%} 30d. Coins leaving exchanges. Supply shock bullish. Size up 25%.",
            )
        return None

    def _whale_flow_hypothesis(self, result: OnChainResult) -> Hypothesis | None:
        if result.whale is None:
            return None
        flow = result.whale.net_flow_30d
        if flow > _WHALE_ACCUM_ALERT_30D:
            params = {
                "position_size_multiplier": 1.20,
                "whale_flow_30d_btc":       flow,
                "rationale":                f"Net whale accumulation {flow:+,.0f} BTC in 30d",
            }
            return Hypothesis.create(
                hypothesis_type=HypothesisType.CROSS_ASSET,
                parent_pattern_id=f"onchain_whale_{result.symbol}",
                parameters=params,
                predicted_sharpe_delta=0.12,
                predicted_dd_delta=-0.02,
                novelty_score=0.65,
                description=f"WHALE ACCUMULATION: +{flow:,.0f} BTC net flow over 30d. Institutional buying. Size up 20%.",
            )
        if flow < _WHALE_DISTRIB_ALERT_30D:
            params = {
                "position_size_multiplier": 0.80,
                "whale_flow_30d_btc":       flow,
                "rationale":                f"Net whale distribution {flow:+,.0f} BTC in 30d",
            }
            return Hypothesis.create(
                hypothesis_type=HypothesisType.CROSS_ASSET,
                parent_pattern_id=f"onchain_whale_{result.symbol}",
                parameters=params,
                predicted_sharpe_delta=0.10,
                predicted_dd_delta=-0.06,
                novelty_score=0.65,
                description=f"WHALE DISTRIBUTION: {flow:,.0f} BTC net flow over 30d. Large holders selling. Reduce to 80%.",
            )
        return None


# ── Helpers ─────────────────────────────────────────────────────────────────

def _score_to_confidence(abs_score: float) -> float:
    """Map absolute composite score to [0, 1] confidence."""
    if abs_score >= 0.90:
        return 0.90
    if abs_score >= 0.80:
        return 0.80
    if abs_score >= 0.70:
        return 0.70
    if abs_score >= 0.60:
        return 0.60
    return 0.50


def _format_component_signals(signals: dict) -> str:
    parts = []
    labels = {"mvrv": "MVRV", "nvt": "NVT", "sopr": "SOPR",
              "exchange_reserves": "ExR", "whale": "Whale"}
    for k, label in labels.items():
        if k in signals:
            parts.append(f"{label}={signals[k]:+.2f}")
    return " | ".join(parts)
