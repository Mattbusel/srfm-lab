"""
regime.py — Market regime detection for SRFM strategies.

Detects four regimes: TRENDING, RANGING, CRISIS, RECOVERY.
Uses a combination of SRFM geodesic deviation, causal fraction, and
classical volatility/momentum signals.
"""

from __future__ import annotations
import math
from collections import deque
from typing import Optional

from srfm_core import Causal, GeodesicAnalyzer, MarketRegime


class RegimeDetector:
    """
    Combine SRFM and classical signals to classify the current market regime.

    Regime definitions:
    - TRENDING  : high causal fraction + directional momentum
    - RANGING   : low deviation + no momentum
    - CRISIS    : high geodesic deviation + spacelike dominance
    - RECOVERY  : transitioning from CRISIS with increasing causal fraction
    """

    def __init__(
        self,
        window: int = 30,
        crisis_deviation_threshold: float = 0.015,
        crisis_causal_threshold: float = 0.35,
        trending_causal_threshold: float = 0.65,
        trending_momentum_threshold: float = 0.005,
        recovery_causal_min: float = 0.45,
    ):
        self.crisis_dev_thresh = crisis_deviation_threshold
        self.crisis_causal_thresh = crisis_causal_threshold
        self.trending_causal_thresh = trending_causal_threshold
        self.trending_mom_thresh = trending_momentum_threshold
        self.recovery_causal_min = recovery_causal_min

        self._geodesic = GeodesicAnalyzer(window=window)
        self._returns: deque = deque(maxlen=window)
        self._regimes: deque = deque(maxlen=window)

        self.current: MarketRegime = MarketRegime.RANGING
        self._prior_crisis: bool = False

    def update(self, price_return: float, causal: Causal) -> MarketRegime:
        self._geodesic.update(price_return, causal)
        self._returns.append(price_return)

        dev = self._geodesic.geodesic_deviation
        cf = self._geodesic.causal_fraction
        mom = self._momentum()

        # Crisis: high curvature + spacelike dominance
        if dev > self.crisis_dev_thresh and cf < self.crisis_causal_thresh:
            regime = MarketRegime.CRISIS
            self._prior_crisis = True

        # Recovery: coming out of crisis
        elif self._prior_crisis and cf >= self.recovery_causal_min:
            regime = MarketRegime.RECOVERY
            if cf >= self.trending_causal_thresh:
                self._prior_crisis = False

        # Trending: high causal fraction + clear momentum
        elif cf >= self.trending_causal_thresh and abs(mom) > self.trending_mom_thresh:
            regime = MarketRegime.TRENDING
            self._prior_crisis = False

        # Ranging: everything else
        else:
            regime = MarketRegime.RANGING
            self._prior_crisis = False

        self.current = regime
        self._regimes.append(regime)
        return regime

    def _momentum(self) -> float:
        if len(self._returns) < 5:
            return 0.0
        recent = list(self._returns)[-5:]
        return sum(recent) / len(recent)

    @property
    def is_crisis(self) -> bool:
        return self.current == MarketRegime.CRISIS

    @property
    def is_trending(self) -> bool:
        return self.current == MarketRegime.TRENDING

    @property
    def regime_stability(self) -> float:
        """Fraction of recent bars in the same regime as current."""
        if not self._regimes:
            return 0.0
        return sum(1 for r in self._regimes if r == self.current) / len(self._regimes)
