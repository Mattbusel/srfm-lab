"""
alternative_data/derivatives_signal.py
=========================================
Composite DerivativesSignal combining open interest, funding rates,
and liquidation data into a single actionable signal per symbol.

Composite signal rationale
---------------------------
No single derivatives metric is reliable alone:
  - Funding rate alone can stay extreme for days before reverting
  - OI change alone is ambiguous without price direction
  - Liquidations alone can be noise (small cascades happen regularly)

The power comes from confluence:

  SQUEEZE SETUP (bullish):
    funding < -0.05%  AND  OI rising  AND  recent large SHORT liquidations
    → Shorts are paying, more shorts being opened, but shorts are getting
      flushed out.  Classic coiled-spring setup.

  FADE SETUP (bearish):
    funding > +0.10%  AND  OI rising  AND  recent large LONG liquidations
    → Longs paying enormous carry, new longs still opening, but longs
      already being liquidated.  Capitulation of overcrowded trade.

  TREND CONFIRMATION (neutral to mildly bullish):
    funding 0.02-0.05%  AND  OI rising  AND  no significant liquidations
    → Normal healthy bull trend.  Not a fade signal yet.

  CAPITULATION CONFIRMATION (bearish bottom?):
    funding negative  AND  OI falling  AND  large LONG liquidations
    → Forced selling of longs; historically ~70% of these end within 48h
      and price recovers.  Contrarian long setup with tight risk.

Score construction:
  Each signal component contributes a sub-score in [-1, +1].
  Weights: funding=0.35, OI=0.35, liquidations=0.30
  Positive composite = bullish, negative = bearish.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .futures_oi     import OISnapshot
from .funding_rates  import FundingRateSnapshot
from .liquidations   import LiquidationSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

FUNDING_WEIGHT:       float = 0.35
OI_WEIGHT:            float = 0.35
LIQUIDATION_WEIGHT:   float = 0.30

# Confluence bonus: applied when all 3 signals agree directionally
CONFLUENCE_BONUS:     float = 0.15


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DerivativesSignal:
    """
    Composite derivatives-market signal for a single symbol.

    Attributes
    ----------
    symbol              : Canonical ticker
    composite_score     : Weighted composite score [-1, +1]
                          Positive = bullish pressure, Negative = bearish
    confidence          : 0-1 based on data quality and signal confluence
    signal_label        : 'squeeze_setup' | 'fade_setup' | 'trend_confirmation' |
                          'capitulation_bottom' | 'neutral'
    funding_score       : Individual funding sub-score [-1, +1]
    oi_score            : Individual OI sub-score [-1, +1]
    liq_score           : Individual liquidation sub-score [-1, +1]
    confluence          : True if all 3 components agree directionally
    funding_rate        : Current funding rate
    oi_regime           : 'rising' | 'falling' | 'stable'
    liq_cascade         : True if a liquidation cascade is in progress
    dominant_liq_side   : 'long' | 'short' | 'balanced'
    timestamp           : UTC ISO string
    """
    symbol:            str
    composite_score:   float
    confidence:        float
    signal_label:      str
    funding_score:     float
    oi_score:          float
    liq_score:         float
    confluence:        bool
    funding_rate:      float
    oi_regime:         str
    liq_cascade:       bool
    dominant_liq_side: str
    timestamp:         str

    @property
    def is_bullish(self) -> bool:
        return self.composite_score > 0.15

    @property
    def is_bearish(self) -> bool:
        return self.composite_score < -0.15

    @property
    def is_squeeze_setup(self) -> bool:
        return self.signal_label == "squeeze_setup"

    @property
    def is_fade_setup(self) -> bool:
        return self.signal_label == "fade_setup"

    def to_dict(self) -> dict:
        return {
            "symbol":            self.symbol,
            "composite_score":   self.composite_score,
            "confidence":        self.confidence,
            "signal_label":      self.signal_label,
            "funding_rate":      self.funding_rate,
            "oi_regime":         self.oi_regime,
            "liq_cascade":       self.liq_cascade,
            "dominant_liq_side": self.dominant_liq_side,
            "confluence":        self.confluence,
            "timestamp":         self.timestamp,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class DerivativesSignalBuilder:
    """
    Combines OISnapshot, FundingRateSnapshot, and LiquidationSnapshot into
    a single DerivativesSignal per symbol.

    Usage::

        builder  = DerivativesSignalBuilder()
        signals  = builder.build_all(oi_snaps, funding_snaps, liq_snaps)
        for sig in signals:
            print(sig.symbol, sig.signal_label, sig.composite_score)
    """

    def build_all(
        self,
        oi_snapshots:      list[OISnapshot],
        funding_snapshots: list[FundingRateSnapshot],
        liq_snapshots:     list[LiquidationSnapshot],
    ) -> list[DerivativesSignal]:
        """
        Build composite signals for all symbols that have data in all three sources.

        Symbols with missing data from any source receive a neutral signal with
        low confidence rather than being dropped entirely.

        Returns
        -------
        List of DerivativesSignal sorted by abs(composite_score) descending.
        """
        # Build lookup dicts by ticker
        oi_map:  dict[str, OISnapshot]            = {s.ticker: s for s in oi_snapshots}
        fr_map:  dict[str, FundingRateSnapshot]   = {s.ticker: s for s in funding_snapshots}
        liq_map: dict[str, LiquidationSnapshot]   = {s.symbol: s for s in liq_snapshots}

        all_symbols = set(oi_map) | set(fr_map) | set(liq_map)
        ts_now      = datetime.now(timezone.utc).isoformat()
        results:    list[DerivativesSignal] = []

        for sym in all_symbols:
            oi  = oi_map.get(sym)
            fr  = fr_map.get(sym)
            liq = liq_map.get(sym)

            sig = self.build_one(sym, oi, fr, liq, ts_now)
            results.append(sig)

        results.sort(key=lambda s: abs(s.composite_score), reverse=True)
        logger.info(
            "DerivativesSignalBuilder: built %d composite signals.", len(results)
        )
        return results

    def build_one(
        self,
        symbol:    str,
        oi:        Optional[OISnapshot],
        fr:        Optional[FundingRateSnapshot],
        liq:       Optional[LiquidationSnapshot],
        ts_now:    str = "",
    ) -> DerivativesSignal:
        """
        Build a single composite signal, handling any missing components gracefully.
        """
        if not ts_now:
            ts_now = datetime.now(timezone.utc).isoformat()

        components_available = sum(x is not None for x in (oi, fr, liq))

        # --- Funding sub-score ----------------------------------------
        funding_score, funding_rate = self._score_funding(fr)

        # --- OI sub-score ---------------------------------------------
        oi_score, oi_regime = self._score_oi(oi)

        # --- Liquidation sub-score ------------------------------------
        liq_score, liq_cascade, dominant_liq = self._score_liquidations(liq)

        # --- Composite score ------------------------------------------
        raw_composite = (
            FUNDING_WEIGHT     * funding_score
            + OI_WEIGHT        * oi_score
            + LIQUIDATION_WEIGHT * liq_score
        )

        # Confluence bonus: if all three directionally agree
        direction_signs = [
            math.copysign(1, funding_score),
            math.copysign(1, oi_score),
            math.copysign(1, liq_score),
        ]
        confluence = len(set(direction_signs)) == 1 and all(
            abs(s) > 0.1 for s in (funding_score, oi_score, liq_score)
        )
        if confluence:
            raw_composite *= (1 + CONFLUENCE_BONUS)

        composite = max(-1.0, min(1.0, raw_composite))

        # --- Confidence -----------------------------------------------
        base_conf = components_available / 3.0
        # Increase confidence if signals are strong
        magnitude = abs(composite)
        confidence = min(1.0, base_conf * (0.5 + magnitude))

        # --- Signal label ----------------------------------------------
        label = self._classify_label(composite, fr, oi, liq)

        return DerivativesSignal(
            symbol=symbol,
            composite_score=round(composite, 4),
            confidence=round(confidence, 3),
            signal_label=label,
            funding_score=round(funding_score, 4),
            oi_score=round(oi_score, 4),
            liq_score=round(liq_score, 4),
            confluence=confluence,
            funding_rate=funding_rate,
            oi_regime=oi_regime,
            liq_cascade=liq_cascade,
            dominant_liq_side=dominant_liq,
            timestamp=ts_now,
        )

    # ------------------------------------------------------------------ #
    # Scoring sub-methods                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_funding(fr: Optional[FundingRateSnapshot]) -> tuple[float, float]:
        """
        Convert funding rate to a sub-score.
        Extreme positive funding → bearish fade signal → negative score.
        Extreme negative funding → bullish squeeze setup → positive score.
        """
        if fr is None:
            return 0.0, 0.0
        rate = fr.current_rate
        # Normalise: ±0.001 (0.1%) maps to ±1.0
        score = -math.tanh(rate / 0.001)  # invert: high funding = bearish
        return float(score), float(rate)

    @staticmethod
    def _score_oi(oi: Optional[OISnapshot]) -> tuple[float, str]:
        """
        Convert OI market signal to a sub-score.
        """
        if oi is None:
            return 0.0, "unknown"
        signal_map = {
            "trend_confirmation": +0.5,
            "short_squeeze":      +0.8,
            "bearish_expansion":  -0.6,
            "capitulation":       -0.4,   # can be contrarian positive but label negative
            "neutral":             0.0,
        }
        score = signal_map.get(oi.market_signal, 0.0)
        # Amplify by OI change magnitude
        mag    = min(1.0, abs(oi.oi_change_pct) / 10.0)  # normalise at 10%
        score *= (0.7 + 0.3 * mag)
        return float(score), oi.oi_regime

    @staticmethod
    def _score_liquidations(liq: Optional[LiquidationSnapshot]) -> tuple[float, bool, str]:
        """
        Convert liquidation snapshot to a sub-score.
        Dominant long liquidations → bearish.
        Dominant short liquidations → bullish (squeeze indicator).
        """
        if liq is None:
            return 0.0, False, "balanced"
        score = 0.0
        if liq.dominant_side == "short":   # shorts getting squeezed → bullish
            score = min(1.0, liq.z_score / 3.0) * 0.8
        elif liq.dominant_side == "long":  # longs getting crushed → bearish
            score = -min(1.0, liq.z_score / 3.0) * 0.8
        return float(score), liq.is_cascade, liq.dominant_side

    @staticmethod
    def _classify_label(
        composite: float,
        fr:        Optional[FundingRateSnapshot],
        oi:        Optional[OISnapshot],
        liq:       Optional[LiquidationSnapshot],
    ) -> str:
        """Assign a human-readable label based on the composite signal and component states."""
        if fr and fr.extreme_negative and oi and oi.oi_regime == "rising" and liq and liq.dominant_side == "short":
            return "squeeze_setup"
        if fr and fr.extreme_positive and oi and oi.oi_regime == "rising" and liq and liq.dominant_side == "long":
            return "fade_setup"
        if oi and oi.market_signal == "capitulation" and liq and liq.dominant_side == "long":
            return "capitulation_bottom"
        if composite > 0.15 and oi and oi.market_signal == "trend_confirmation":
            return "trend_confirmation"
        return "neutral"
