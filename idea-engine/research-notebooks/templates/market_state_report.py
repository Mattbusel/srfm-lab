"""
MarketStateReportTemplate: current market state snapshot.

Data sources aggregated:
  - Macro: FOMC schedule, CPI status (via EconomicCalendar)
  - On-chain: simulated SOPR, MVRV-Z, hash-rate ribbon (placeholder values
    when live API unavailable)
  - Sentiment: fear & greed index (api.alternative.me)
  - Derivatives: funding rate proxy (simulated when unavailable)

Produces a composite signal score (0=full bear, 1=full bull) and flags
extreme readings.  Generates a plain-English trading recommendation.

Output: MarketStateReport
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Historical percentile buckets for composite scoring
# Score < 0.3  => bear / risk-off
# Score > 0.7  => bull / risk-on
SCORE_THRESHOLDS = {"strong_bear": 0.2, "bear": 0.35, "neutral": 0.65, "bull": 0.8}


@dataclass
class SignalReading:
    """One component signal reading."""

    name: str
    value: float
    normalised: float    # 0-1 (1=most bullish)
    regime_label: str    # bearish | neutral | bullish
    source: str
    stale: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "normalised": round(self.normalised, 4),
            "regime_label": self.regime_label,
            "source": self.source,
            "stale": self.stale,
        }


@dataclass
class MarketStateReport:
    """Full market state snapshot."""

    generated_at: datetime
    composite_score: float         # 0-1
    composite_label: str           # strong_bear | bear | neutral | bull | strong_bull
    signals: List[SignalReading]
    macro_buffer_active: bool
    macro_buffer_reason: str
    recommendation: str
    entry_size_multiplier: float   # 0-1 position size modifier
    flagged_extremes: List[str]    # list of extreme readings
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "composite_score": round(self.composite_score, 4),
            "composite_label": self.composite_label,
            "signals": [s.to_dict() for s in self.signals],
            "macro_buffer_active": self.macro_buffer_active,
            "macro_buffer_reason": self.macro_buffer_reason,
            "recommendation": self.recommendation,
            "entry_size_multiplier": round(self.entry_size_multiplier, 2),
            "flagged_extremes": self.flagged_extremes,
        }


class MarketStateReportTemplate:
    """
    Generates a current market state report.

    Usage::

        template = MarketStateReportTemplate()
        report   = template.run()
        print(report.recommendation)
    """

    def __init__(self, fred_api_key: str = "") -> None:
        self._fred_key = fred_api_key
        self._timeout = 8

    def run(self) -> MarketStateReport:
        """Collect all signals and generate the market state report."""
        now = datetime.now(timezone.utc)
        signals: List[SignalReading] = []
        flagged: List[str] = []

        # --- Fear & Greed ---
        fg = self._fetch_fear_greed()
        signals.append(fg)
        if fg.normalised < 0.15:
            flagged.append(f"Extreme Fear: Fear & Greed = {fg.value:.0f}")
        elif fg.normalised > 0.85:
            flagged.append(f"Extreme Greed: Fear & Greed = {fg.value:.0f}")

        # --- On-chain signals (simulated if no API) ---
        sopr = self._sopr_signal()
        mvrv = self._mvrv_signal()
        hr = self._hash_rate_ribbon_signal()
        for sig in (sopr, mvrv, hr):
            signals.append(sig)
            if sig.regime_label == "bearish" and sig.normalised < 0.2:
                flagged.append(f"Extreme bearish reading: {sig.name} = {sig.value:.3f}")
            elif sig.regime_label == "bullish" and sig.normalised > 0.85:
                flagged.append(f"Extreme bullish reading: {sig.name} = {sig.value:.3f}")

        # --- Funding rate ---
        funding = self._funding_rate_signal()
        signals.append(funding)
        if abs(funding.value) > 0.08:
            flagged.append(f"Extreme funding rate: {funding.value:.4f}")

        # --- Macro buffer ---
        macro_buffer, macro_reason = self._check_macro_buffer()

        # --- Composite score ---
        weights = [1.5, 1.0, 1.0, 0.8, 0.7]  # F&G, SOPR, MVRV, HR, Funding
        if len(signals) < len(weights):
            weights = weights[:len(signals)]
        composite = float(np.average(
            [s.normalised for s in signals[:len(weights)]],
            weights=weights,
        ))

        label = self._label_from_score(composite)

        # --- Position size multiplier ---
        mult = self._compute_multiplier(composite, macro_buffer, len(flagged))

        # --- Recommendation ---
        recommendation = self._generate_recommendation(
            composite, label, signals, macro_buffer, macro_reason, flagged, mult
        )

        return MarketStateReport(
            generated_at=now,
            composite_score=composite,
            composite_label=label,
            signals=signals,
            macro_buffer_active=macro_buffer,
            macro_buffer_reason=macro_reason,
            recommendation=recommendation,
            entry_size_multiplier=mult,
            flagged_extremes=flagged,
        )

    # ── signal fetchers ───────────────────────────────────────────────────────────

    def _fetch_fear_greed(self) -> SignalReading:
        try:
            url = "https://api.alternative.me/fng/?limit=1&format=json"
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "SRFM-IdeaEngine/1.0")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
            value = float(data["data"][0]["value"])
            normalised = value / 100.0
            label = "bullish" if value > 60 else ("bearish" if value < 30 else "neutral")
            return SignalReading("fear_greed_index", value, normalised, label, "alternative.me")
        except Exception as exc:
            logger.debug("Fear & Greed fetch failed: %s", exc)
            # Simulate neutral
            return SignalReading("fear_greed_index", 50.0, 0.50, "neutral", "simulated", stale=True)

    def _sopr_signal(self) -> SignalReading:
        """
        Simulate SOPR reading.  SOPR > 1 = holders selling at profit (neutral/bull).
        SOPR < 1 = capitulation (bearish but potential bottom).
        """
        # Placeholder — replace with Glassnode API call if available
        value = 1.02  # slightly above 1 = mild bullish
        normalised = min(max((value - 0.95) / 0.15, 0.0), 1.0)  # 0.95-1.10 range maps to 0-1
        label = "bullish" if value > 1.01 else ("bearish" if value < 0.99 else "neutral")
        return SignalReading("sopr_7d_ma", value, normalised, label, "simulated", stale=True)

    def _mvrv_signal(self) -> SignalReading:
        """
        Simulate MVRV Z-score.  Z > 7 = overvalued, Z < 0 = undervalued.
        """
        value = 1.5  # moderate bull
        normalised = min(max((value + 1) / 8.0, 0.0), 1.0)  # -1 to 7 maps to 0-1
        label = "bullish" if value > 3 else ("bearish" if value < 0 else "neutral")
        return SignalReading("mvrv_z_score", value, normalised, label, "simulated", stale=True)

    def _hash_rate_ribbon_signal(self) -> SignalReading:
        """
        Simulate hash rate ribbon.  1 = ribbon expanded (bullish), 0 = compressed (bearish).
        """
        value = 0.7  # ribbon is above compressed zone
        label = "bullish" if value > 0.6 else ("bearish" if value < 0.3 else "neutral")
        return SignalReading("hash_rate_ribbon", value, value, label, "simulated", stale=True)

    def _funding_rate_signal(self) -> SignalReading:
        """
        Simulate aggregate funding rate (8h).
        Extreme positive = crowded longs (bearish risk), extreme negative = bearish.
        Normalise: neutral at 0.01% (8h)
        """
        value = 0.012  # slightly positive (mild bull market longs)
        # Convert: -0.10 to +0.10 range, neutral = 0 normalised to 0.5
        normalised = min(max(0.5 + value * 5, 0.0), 1.0)
        label = "bullish" if value > 0.05 else ("bearish" if value < -0.02 else "neutral")
        return SignalReading("btc_funding_rate_8h", value, normalised, label, "simulated", stale=True)

    def _check_macro_buffer(self) -> Tuple[bool, str]:
        """Check if a macro event buffer is active."""
        try:
            from ...event_calendar.sources.economic_calendar import EconomicCalendar
            cal = EconomicCalendar(fred_api_key=self._fred_key)
            if cal.is_macro_buffer_active():
                active = cal.active_buffer_events()
                reason = f"Macro buffer: {', '.join(e.event_name for e in active)}"
                return True, reason
        except ImportError:
            pass
        return False, ""

    # ── scoring helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score <= SCORE_THRESHOLDS["strong_bear"]:
            return "strong_bear"
        if score <= SCORE_THRESHOLDS["bear"]:
            return "bear"
        if score <= SCORE_THRESHOLDS["neutral"]:
            return "neutral"
        if score <= SCORE_THRESHOLDS["bull"]:
            return "bull"
        return "strong_bull"

    @staticmethod
    def _compute_multiplier(
        composite: float, macro_buffer: bool, n_extremes: int
    ) -> float:
        mult = 1.0
        if composite < 0.25:
            mult = 0.5   # strong bear — reduce all entries
        elif composite < 0.40:
            mult = 0.75
        if macro_buffer:
            mult *= 0.5
        if n_extremes >= 2:
            mult *= 0.8
        return round(max(mult, 0.1), 2)

    @staticmethod
    def _generate_recommendation(
        composite: float,
        label: str,
        signals: List[SignalReading],
        macro_buffer: bool,
        macro_reason: str,
        flagged: List[str],
        mult: float,
    ) -> str:
        lines = [f"Market State: {label.upper()} (composite={composite:.2f})"]

        if flagged:
            lines.append("Extreme readings: " + "; ".join(flagged))

        if macro_buffer:
            lines.append(f"MACRO BUFFER ACTIVE: {macro_reason} — reduce all entries by 50%.")

        if composite < 0.35:
            lines.append("BEARISH BIAS: Prefer short setups, reduce long exposure, tighten stops.")
        elif composite > 0.65:
            lines.append("BULLISH BIAS: Favour long setups, standard or slightly elevated sizing.")
        else:
            lines.append("NEUTRAL: No strong directional edge — trade setups on merit, standard sizing.")

        lines.append(f"Position size multiplier: {mult:.0%} of normal.")

        stale = [s.name for s in signals if s.stale]
        if stale:
            lines.append(f"NOTE: {len(stale)} signal(s) using simulated data: {', '.join(stale)}.")

        return " | ".join(lines)
