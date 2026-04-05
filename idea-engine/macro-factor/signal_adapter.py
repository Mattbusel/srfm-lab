"""
macro-factor/signal_adapter.py
────────────────────────────────
Convert macro regime to position-size multipliers and IAE hypotheses.

Position multipliers by regime
───────────────────────────────
  RISK_ON      → 1.20x  (add 20% to all position sizes)
  RISK_NEUTRAL → 1.00x  (no adjustment)
  RISK_OFF     → 0.60x  (cut position sizes 40%)
  CRISIS       → 0.25x  (minimal exposure — survival mode)

Hypothesis generation logic
────────────────────────────
A regime change (or extreme composite score) generates a PARAMETER_TWEAK
hypothesis with specific parameter delta suggestions for the active genome set.

Crisis override hypotheses are highest priority — they recommend immediate
risk reduction regardless of other signals.

Component extreme hypotheses:
  VIX spike → immediate stop-tightening hypothesis
  DXY surge → reduce max position hypothesis
  Equity 200d MA break → early warning hypothesis
  Liquidity surge (lagged) → preemptive sizing increase hypothesis
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from .regime_classifier import RegimeClassification, MacroRegime
from .factors.dxy import DXYResult
from .factors.vix import VIXResult
from .factors.equity_momentum import EquityMomentumResult
from .factors.liquidity import LiquidityResult

sys.path.insert(0, str(Path(__file__).parent.parent))
from hypothesis.types import Hypothesis, HypothesisType

logger = logging.getLogger(__name__)

# Regime → position multiplier
REGIME_MULTIPLIERS = {
    MacroRegime.RISK_ON:      1.20,
    MacroRegime.RISK_NEUTRAL: 1.00,
    MacroRegime.RISK_OFF:     0.60,
    MacroRegime.CRISIS:       0.25,
}


class SignalAdapter:
    """Convert macro regime and factor signals to IAE hypotheses.

    Parameters
    ----------
    composite_threshold:
        Minimum |composite_score| departure from ±0 to trigger a hypothesis.
    """

    def __init__(self, composite_threshold: float = 0.25) -> None:
        self.composite_threshold = composite_threshold

    def adapt(
        self,
        classification: RegimeClassification,
        dxy:       Optional[DXYResult]           = None,
        vix:       Optional[VIXResult]           = None,
        equity:    Optional[EquityMomentumResult] = None,
        liquidity: Optional[LiquidityResult]     = None,
    ) -> List[Hypothesis]:
        """Generate IAE hypotheses from the macro regime classification.

        Returns a list of Hypothesis objects ready for the hypothesis queue.
        """
        hypotheses: List[Hypothesis] = []

        # 1. Primary regime hypothesis
        h = self._regime_hypothesis(classification)
        if h:
            hypotheses.append(h)

        # 2. Crisis override (highest priority)
        if classification.crisis_override:
            hypotheses.append(self._crisis_hypothesis(classification))

        # 3. Component-specific hypotheses
        if vix is not None and (vix.is_spike or vix.vix_current > 35):
            hypotheses.append(self._vix_spike_hypothesis(vix))

        if dxy is not None and dxy.momentum_20d > 0.04:
            hypotheses.append(self._dxy_surge_hypothesis(dxy))

        if equity is not None and "just_crossed_below" in (
            equity.spy_200d_crossover, equity.qqq_200d_crossover
        ):
            hypotheses.append(self._equity_ma_break_hypothesis(equity))

        if liquidity is not None and liquidity.m2_growth_lagged > 0.06:
            hypotheses.append(self._liquidity_surge_hypothesis(liquidity))

        logger.info(
            "SignalAdapter: %d hypotheses generated for %s regime",
            len(hypotheses), classification.regime.value,
        )
        return hypotheses

    def get_position_multiplier(self, classification: RegimeClassification) -> float:
        """Return the position-size multiplier for the given regime."""
        return REGIME_MULTIPLIERS[classification.regime]

    # ── Individual hypothesis generators ───────────────────────────────────

    def _regime_hypothesis(self, c: RegimeClassification) -> Optional[Hypothesis]:
        """Generate a PARAMETER_TWEAK hypothesis for any non-neutral regime."""
        mult = REGIME_MULTIPLIERS[c.regime]
        if abs(mult - 1.0) < 0.05 and not c.crisis_override:
            return None  # RISK_NEUTRAL with no crisis — no action needed

        # Sharpe delta: regime filter adds value proportional to departure from neutral
        sharpe_delta  = round(abs(c.composite_score) * 0.20, 3)
        dd_delta      = round(-abs(c.composite_score) * 0.05, 3) if c.composite_score > 0 else round(-abs(c.composite_score) * 0.10, 3)

        params = {
            "position_size_multiplier": mult,
            "macro_regime":             c.regime.value,
            "macro_composite_score":    c.composite_score,
            "confidence":               c.confidence,
            "component_signals":        c.component_signals,
            "rationale":                f"Macro regime {c.regime.value}: composite={c.composite_score:+.3f}",
        }

        if c.regime == MacroRegime.RISK_ON:
            params["atr_stop_mult"]  = 2.2   # wider stop — let winners run
            params["max_loss_bars"]  = 22
            description = (
                f"MACRO RISK_ON: composite={c.composite_score:+.3f} (conf={c.confidence:.2f}). "
                f"Position sizing x{mult}. Widen stops. "
                f"Signals: {_fmt_signals(c.component_signals)}"
            )
        elif c.regime == MacroRegime.RISK_OFF:
            params["atr_stop_mult"]  = 1.4   # tighter stop
            params["max_loss_bars"]  = 8
            description = (
                f"MACRO RISK_OFF: composite={c.composite_score:+.3f}. "
                f"Reduce position sizing to {mult}x. Tighten stops. "
                f"Signals: {_fmt_signals(c.component_signals)}"
            )
        else:  # CRISIS
            params["atr_stop_mult"]  = 1.2
            params["max_loss_bars"]  = 5
            description = (
                f"MACRO CRISIS: composite={c.composite_score:+.3f}. "
                f"Emergency position reduction to {mult}x. "
                f"{c.crisis_reason if c.crisis_override else ''}"
            )

        return Hypothesis.create(
            hypothesis_type=HypothesisType.REGIME_FILTER,
            parent_pattern_id=f"macro_regime_{c.regime.value}",
            parameters=params,
            predicted_sharpe_delta=sharpe_delta,
            predicted_dd_delta=dd_delta,
            novelty_score=min(0.95, abs(c.composite_score) + 0.1),
            description=description,
        )

    def _crisis_hypothesis(self, c: RegimeClassification) -> Hypothesis:
        """High-priority crisis override hypothesis — immediate risk reduction."""
        params = {
            "position_size_multiplier": 0.25,
            "atr_stop_mult":            1.0,
            "max_loss_bars":            3,
            "crisis_reason":            c.crisis_reason,
            "macro_composite_score":    c.composite_score,
        }
        return Hypothesis.create(
            hypothesis_type=HypothesisType.REGIME_FILTER,
            parent_pattern_id="macro_crisis_override",
            parameters=params,
            predicted_sharpe_delta=0.25,   # high value — protects from large drawdown
            predicted_dd_delta=-0.20,
            novelty_score=0.95,
            description=f"CRISIS OVERRIDE: {c.crisis_reason}. Immediate 75% position reduction.",
        )

    def _vix_spike_hypothesis(self, vix: VIXResult) -> Hypothesis:
        """Tight stop hypothesis triggered by VIX spike."""
        params = {
            "atr_stop_mult":            1.2,
            "max_loss_bars":            5,
            "position_size_multiplier": 0.50,
            "vix_current":              vix.vix_current,
            "vix_5d_change_pct":        vix.vix_5d_change_pct,
            "rationale":                f"VIX spike {vix.vix_5d_change_pct:+.0%} — volatility regime change",
        }
        return Hypothesis.create(
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            parent_pattern_id="macro_vix_spike",
            parameters=params,
            predicted_sharpe_delta=0.15,
            predicted_dd_delta=-0.12,
            novelty_score=0.80,
            description=(
                f"VIX SPIKE: {vix.vix_current:.1f} ({vix.vix_5d_change_pct:+.0%} in 5d). "
                f"Tighten stops to 1.2x ATR, halve position size."
            ),
        )

    def _dxy_surge_hypothesis(self, dxy: DXYResult) -> Hypothesis:
        """Reduce size hypothesis triggered by DXY surge."""
        params = {
            "position_size_multiplier": 0.70,
            "dxy_momentum_20d":         dxy.momentum_20d,
            "dxy_level":                dxy.dxy_level,
            "rationale":                f"DXY surging +{dxy.momentum_20d:.1%} — crypto headwind",
        }
        return Hypothesis.create(
            hypothesis_type=HypothesisType.REGIME_FILTER,
            parent_pattern_id="macro_dxy_surge",
            parameters=params,
            predicted_sharpe_delta=0.10,
            predicted_dd_delta=-0.06,
            novelty_score=0.65,
            description=f"DXY SURGE: +{dxy.momentum_20d:.1%} 20d. Strong dollar headwind for crypto. Reduce sizing to 70%.",
        )

    def _equity_ma_break_hypothesis(self, equity: EquityMomentumResult) -> Hypothesis:
        """Early warning hypothesis for equity 200d MA break."""
        cross = equity.spy_200d_crossover if "crossed" in equity.spy_200d_crossover else equity.qqq_200d_crossover
        params = {
            "position_size_multiplier": 0.65,
            "atr_stop_mult":            1.5,
            "max_loss_bars":            8,
            "spy_crossover":            equity.spy_200d_crossover,
            "qqq_crossover":            equity.qqq_200d_crossover,
            "rationale":                f"Equity 200d MA break ({cross}) — crypto historically follows in 2-5 days",
        }
        return Hypothesis.create(
            hypothesis_type=HypothesisType.REGIME_FILTER,
            parent_pattern_id="macro_equity_ma_break",
            parameters=params,
            predicted_sharpe_delta=0.18,
            predicted_dd_delta=-0.10,
            novelty_score=0.82,
            description=(
                f"EQUITY 200d MA BREAK: SPY={equity.spy_200d_crossover}, QQQ={equity.qqq_200d_crossover}. "
                f"Crypto historically follows within 2–5 days. Reduce sizing to 65%, tighten stops."
            ),
        )

    def _liquidity_surge_hypothesis(self, liquidity: LiquidityResult) -> Hypothesis:
        """Preemptive size increase for lagged M2 surge."""
        params = {
            "position_size_multiplier": 1.25,
            "m2_growth_lagged":         liquidity.m2_growth_lagged,
            "rationale":                f"M2 growth {liquidity.m2_growth_lagged:+.1%} 3 months ago — crypto tailwind now",
        }
        return Hypothesis.create(
            hypothesis_type=HypothesisType.REGIME_FILTER,
            parent_pattern_id="macro_liquidity_surge",
            parameters=params,
            predicted_sharpe_delta=0.14,
            predicted_dd_delta=-0.02,
            novelty_score=0.70,
            description=(
                f"M2 LIQUIDITY SURGE: lagged M2 growth={liquidity.m2_growth_lagged:+.1%}. "
                f"Historically precedes crypto rally by ~3 months. Increase sizing to 1.25x."
            ),
        )


def _fmt_signals(signals: dict) -> str:
    return " | ".join(f"{k.upper()[:4]}={v:+.2f}" for k, v in signals.items())
