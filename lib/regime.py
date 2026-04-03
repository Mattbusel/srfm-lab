"""
regime.py — Market regime detection extracted from LARSA detect_regime().

Inputs: price, EMA12/26/50/200, ADX, ATR (current + rolling history).
Output: MarketRegime enum + confidence float.

Decision tree (exact LARSA logic):
  1. atr_ratio >= 1.5      → HIGH_VOLATILITY
  2. price > e200, e12>e26 → BULL if ADX threshold else SIDEWAYS
  3. price < e200, e12<e26 → BEAR if ADX threshold else SIDEWAYS
  4. else                  → SIDEWAYS

ADX threshold:
  - Full EMA stack aligned: 14
  - Partial alignment:      18
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Tuple
from srfm_core import MarketRegime


class RegimeDetector:
    """
    Stateful regime detector.  Feed one bar at a time via update().

    Parameters
    ----------
    atr_window : int
        Rolling window of ATR values used to compute atr_ratio.
        LARSA uses 50 (same as aw RollingWindow size).
    """

    def __init__(self, atr_window: int = 50):
        self._atr_hist: deque = deque(maxlen=atr_window)
        self.regime:    MarketRegime = MarketRegime.SIDEWAYS
        self.confidence: float = 0.5
        self.bars_in_regime: int = 0  # rhb counter

    # ------------------------------------------------------------------
    def update(
        self,
        price: float,
        ema12: float,
        ema26: float,
        ema50: float,
        ema200: float,
        adx:   float,
        atr:   float,
    ) -> Tuple[MarketRegime, float]:
        """
        Returns (regime, confidence).
        Exact reproduction of LARSA detect_regime().
        """
        self._atr_hist.append(atr)
        self.bars_in_regime += 1

        # ATR ratio vs rolling mean
        atr_ratio = 1.0
        if len(self._atr_hist) >= 2:
            arr      = np.array(list(self._atr_hist))
            mean_atr = arr.mean()
            atr_ratio = atr / mean_atr if mean_atr > 0 else 1.0

        prev_regime = self.regime

        if atr_ratio >= 1.5:
            nr = MarketRegime.HIGH_VOLATILITY
            nc = min(0.9, 0.5 + (atr_ratio - 1.5) * 0.4)

        elif price > ema200 and ema12 > ema26:
            full_stack = ema12 > ema26 > ema50 > ema200
            adx_thresh = 14 if full_stack else 18
            if adx > adx_thresh:
                nr = MarketRegime.BULL
                nc = min(0.95, 0.5 + (adx - 14) / 60)
            else:
                nr = MarketRegime.SIDEWAYS
                nc = max(0.3, 0.7 - adx / 80)

        elif price < ema200 and ema12 < ema26:
            full_stack = ema200 > ema50 > ema26 > ema12
            adx_thresh = 14 if full_stack else 18
            if adx > adx_thresh:
                nr = MarketRegime.BEAR
                nc = min(0.95, 0.5 + (adx - 14) / 60)
            else:
                nr = MarketRegime.SIDEWAYS
                nc = max(0.3, 0.7 - adx / 80)

        else:
            nr = MarketRegime.SIDEWAYS
            nc = max(0.3, 0.7 - adx / 80)

        if nr != self.regime:
            self.bars_in_regime = 0
            self.regime         = nr

        self.confidence = nc
        return self.regime, self.confidence

    @property
    def is_trending(self) -> bool:
        return self.regime in (MarketRegime.BULL, MarketRegime.BEAR)

    @property
    def is_crisis(self) -> bool:
        return self.regime == MarketRegime.HIGH_VOLATILITY
