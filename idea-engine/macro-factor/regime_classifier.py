"""
macro-factor/regime_classifier.py
───────────────────────────────────
Macro Regime Classifier.

Classifies the current macro environment into one of four regimes:
  RISK_ON      → all systems go — buy crypto aggressively
  RISK_NEUTRAL → normal conditions — standard position sizing
  RISK_OFF     → headwinds building — reduce sizing 40%
  CRISIS       → severe risk-off — minimal exposure

Scoring model
─────────────
Each factor module returns a signal in [-1, +1].
We compute a weighted composite "macro score":

  DXY          20% — inverse: falling dollar = bullish for crypto
  Rates        20% — rising rates = bearish
  VIX          25% — fear gauge (most reactive to sudden risk-off)
  Gold         10% — safe-haven vs liquidity regime
  Equity Mom.  15% — correlated with crypto since 2020
  Liquidity    10% — 3-month lagged M2 growth

Regime thresholds (calibrated to historical BTC drawdowns):
  composite > +0.30  → RISK_ON
  composite > -0.15  → RISK_NEUTRAL
  composite > -0.55  → RISK_OFF
  composite ≤ -0.55  → CRISIS

An emergency CRISIS override fires when any of:
  - VIX > 40 (crisis absolute level)
  - VIX 5-day spike > 70%
  - SPY or QQQ just_crossed_below 200d MA AND VIX > 28
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

import numpy as np

from .factors.dxy import DXYResult
from .factors.rates import RatesResult
from .factors.vix import VIXResult
from .factors.gold import GoldResult
from .factors.equity_momentum import EquityMomentumResult
from .factors.liquidity import LiquidityResult

logger = logging.getLogger(__name__)

# Factor weights (must sum to 1.0)
_WEIGHTS: Dict[str, float] = {
    "dxy":              0.20,
    "rates":            0.20,
    "vix":              0.25,
    "gold":             0.10,
    "equity_momentum":  0.15,
    "liquidity":        0.10,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9

# Regime thresholds
_RISK_ON_THRESHOLD      =  0.30
_RISK_NEUTRAL_THRESHOLD = -0.15
_RISK_OFF_THRESHOLD     = -0.55

# Crisis overrides
_VIX_CRISIS_LEVEL       = 40.0
_VIX_SPIKE_CRISIS       = 0.70   # 70% 5d spike
_VIX_MA_BREAK_THRESHOLD = 28.0


class MacroRegime(str, Enum):
    RISK_ON      = "RISK_ON"
    RISK_NEUTRAL = "RISK_NEUTRAL"
    RISK_OFF     = "RISK_OFF"
    CRISIS       = "CRISIS"


@dataclass
class RegimeClassification:
    regime: MacroRegime
    composite_score: float
    component_signals: Dict[str, float]
    crisis_override: bool
    crisis_reason: str
    confidence: float           # [0, 1] — how close to the threshold boundary
    computed_at: str

    @property
    def position_multiplier(self) -> float:
        """Return the default position-size multiplier for this regime."""
        return {
            MacroRegime.RISK_ON:      1.20,
            MacroRegime.RISK_NEUTRAL: 1.00,
            MacroRegime.RISK_OFF:     0.60,
            MacroRegime.CRISIS:       0.25,
        }[self.regime]


class RegimeClassifier:
    """Classify macro regime from factor results.

    Parameters
    ----------
    weights:
        Override the default factor weights.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or _WEIGHTS

    def classify(
        self,
        dxy:       Optional[DXYResult]             = None,
        rates:     Optional[RatesResult]            = None,
        vix:       Optional[VIXResult]              = None,
        gold:      Optional[GoldResult]             = None,
        equity:    Optional[EquityMomentumResult]   = None,
        liquidity: Optional[LiquidityResult]        = None,
    ) -> RegimeClassification:
        """Classify the current macro regime from factor results.

        Any factor passed as None is excluded from the composite (its weight
        is redistributed proportionally to present factors).

        Returns
        -------
        RegimeClassification with regime label, score, and crisis flags.
        """
        signals: Dict[str, float] = {}
        if dxy       is not None:   signals["dxy"]             = dxy.signal
        if rates     is not None:   signals["rates"]           = rates.signal
        if vix       is not None:   signals["vix"]             = vix.signal
        if gold      is not None:   signals["gold"]            = gold.signal
        if equity    is not None:   signals["equity_momentum"] = equity.signal
        if liquidity is not None:   signals["liquidity"]       = liquidity.signal

        # Weighted composite (renormalise if factors are missing)
        total_weight = sum(self.weights[k] for k in signals)
        if total_weight == 0:
            composite = 0.0
        else:
            composite = float(sum(
                signals[k] * self.weights[k] for k in signals
            ) / total_weight)
        composite = float(np.clip(composite, -1.0, 1.0))

        # --- Crisis override checks ---
        crisis_override = False
        crisis_reason   = ""

        if vix is not None:
            if vix.vix_current >= _VIX_CRISIS_LEVEL:
                crisis_override = True
                crisis_reason   = f"VIX={vix.vix_current:.1f} >= {_VIX_CRISIS_LEVEL} (crisis level)"

            if vix.vix_5d_change_pct >= _VIX_SPIKE_CRISIS:
                crisis_override = True
                crisis_reason   = f"VIX spike {vix.vix_5d_change_pct:+.0%} in 5 days"

        if equity is not None and vix is not None:
            ma_break = "just_crossed_below" in (
                equity.spy_200d_crossover, equity.qqq_200d_crossover
            )
            if ma_break and vix.vix_current > _VIX_MA_BREAK_THRESHOLD:
                crisis_override = True
                crisis_reason   = (
                    f"200d MA break (SPY:{equity.spy_200d_crossover}/QQQ:{equity.qqq_200d_crossover}) "
                    f"+ VIX={vix.vix_current:.1f}"
                )

        # --- Regime classification ---
        if crisis_override:
            regime = MacroRegime.CRISIS
        elif composite >= _RISK_ON_THRESHOLD:
            regime = MacroRegime.RISK_ON
        elif composite >= _RISK_NEUTRAL_THRESHOLD:
            regime = MacroRegime.RISK_NEUTRAL
        elif composite >= _RISK_OFF_THRESHOLD:
            regime = MacroRegime.RISK_OFF
        else:
            regime = MacroRegime.CRISIS

        # Confidence: inverse of distance to nearest threshold
        thresholds = [_RISK_ON_THRESHOLD, _RISK_NEUTRAL_THRESHOLD, _RISK_OFF_THRESHOLD]
        min_dist   = min(abs(composite - t) for t in thresholds)
        confidence = float(np.clip(min_dist / 0.20, 0.0, 1.0))  # 0.20 away = full confidence

        logger.info(
            "RegimeClassifier: composite=%.3f → %s (conf=%.2f, crisis=%s)",
            composite, regime.value, confidence, crisis_override,
        )

        return RegimeClassification(
            regime=regime,
            composite_score=round(composite, 4),
            component_signals={k: round(v, 4) for k, v in signals.items()},
            crisis_override=crisis_override,
            crisis_reason=crisis_reason,
            confidence=round(confidence, 4),
            computed_at=datetime.now(timezone.utc).isoformat(),
        )
