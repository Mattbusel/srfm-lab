"""
Event detection for financial news.

Detects:
- Earnings surprises (beat/miss vs consensus)
- Guidance changes (raise/cut/initiate)
- M&A signals (merger, acquisition, rumor)
- Analyst upgrades/downgrades
- Macro surprises (Fed, GDP, employment)
- Regulatory events (SEC, FDA, antitrust)
- Insider buying/selling clusters
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Event data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedEvent:
    """A financial event detected in news/data."""
    event_type: str              # See EventTypes below
    subtype: str                 # More specific classification
    ticker: str                  # Primary affected ticker (empty = macro event)
    confidence: float            # 0-1 detection confidence
    direction: float             # -1 (negative) to +1 (positive) expected impact
    magnitude: float             # 0-1 expected magnitude
    detected_at: datetime
    source_text: str             # Original text snippet
    extracted_values: Dict[str, Any] = field(default_factory=dict)
    related_tickers: List[str] = field(default_factory=list)
    horizon: str = "1d"          # Expected impact horizon: "1h" | "4h" | "1d" | "1w"
    metadata: Dict = field(default_factory=dict)

    @property
    def alpha_signal(self) -> float:
        """Composite alpha signal: direction * magnitude * confidence."""
        return float(self.direction * self.magnitude * self.confidence)

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["detected_at"] = self.detected_at.isoformat()
        return d


class EventTypes:
    EARNINGS_BEAT     = "earnings_beat"
    EARNINGS_MISS     = "earnings_miss"
    GUIDANCE_RAISE    = "guidance_raise"
    GUIDANCE_CUT      = "guidance_cut"
    GUIDANCE_INITIATE = "guidance_initiate"
    MA_ANNOUNCE       = "ma_announce"
    MA_RUMOR          = "ma_rumor"
    MA_TERMINATION    = "ma_termination"
    ANALYST_UPGRADE   = "analyst_upgrade"
    ANALYST_DOWNGRADE = "analyst_downgrade"
    ANALYST_INITIATE  = "analyst_initiate"
    MACRO_SURPRISE    = "macro_surprise"
    FED_DECISION      = "fed_decision"
    REGULATORY_FINE   = "regulatory_fine"
    REGULATORY_APPROVAL = "regulatory_approval"
    INSIDER_BUY       = "insider_buy"
    INSIDER_SELL      = "insider_sell"
    PRODUCT_LAUNCH    = "product_launch"
    DIVIDEND_CHANGE   = "dividend_change"
    SHARE_REPURCHASE  = "share_repurchase"
    LITIGATION        = "litigation"
    CREDIT_RATING     = "credit_rating"
    SHORT_INTEREST    = "short_interest"

    ALL = [
        EARNINGS_BEAT, EARNINGS_MISS, GUIDANCE_RAISE, GUIDANCE_CUT,
        GUIDANCE_INITIATE, MA_ANNOUNCE, MA_RUMOR, MA_TERMINATION,
        ANALYST_UPGRADE, ANALYST_DOWNGRADE, ANALYST_INITIATE,
        MACRO_SURPRISE, FED_DECISION, REGULATORY_FINE, REGULATORY_APPROVAL,
        INSIDER_BUY, INSIDER_SELL, PRODUCT_LAUNCH, DIVIDEND_CHANGE,
        SHARE_REPURCHASE, LITIGATION, CREDIT_RATING, SHORT_INTEREST,
    ]

    # Expected direction for each event type
    DIRECTION_MAP = {
        EARNINGS_BEAT:        +1.0,
        EARNINGS_MISS:        -1.0,
        GUIDANCE_RAISE:       +1.0,
        GUIDANCE_CUT:         -1.0,
        GUIDANCE_INITIATE:    +0.3,
        MA_ANNOUNCE:          +0.8,
        MA_RUMOR:             +0.5,
        MA_TERMINATION:       -0.5,
        ANALYST_UPGRADE:      +0.6,
        ANALYST_DOWNGRADE:    -0.6,
        ANALYST_INITIATE:     +0.2,
        MACRO_SURPRISE:       0.0,   # depends on direction
        FED_DECISION:         -0.3,  # rate hike default
        REGULATORY_FINE:      -0.7,
        REGULATORY_APPROVAL:  +0.8,
        INSIDER_BUY:          +0.5,
        INSIDER_SELL:         -0.3,
        PRODUCT_LAUNCH:       +0.4,
        DIVIDEND_CHANGE:      +0.5,
        SHARE_REPURCHASE:     +0.5,
        LITIGATION:           -0.6,
        CREDIT_RATING:        0.0,   # depends on direction
        SHORT_INTEREST:       -0.3,
    }

    # Typical impact magnitude by event type
    MAGNITUDE_MAP = {
        EARNINGS_BEAT:        0.75,
        EARNINGS_MISS:        0.75,
        GUIDANCE_RAISE:       0.65,
        GUIDANCE_CUT:         0.70,
        MA_ANNOUNCE:          0.90,
        ANALYST_UPGRADE:      0.45,
        ANALYST_DOWNGRADE:    0.45,
        FED_DECISION:         0.85,
        REGULATORY_FINE:      0.60,
        REGULATORY_APPROVAL:  0.80,
        INSIDER_BUY:          0.30,
        INSIDER_SELL:         0.25,
        LITIGATION:           0.55,
    }

    @classmethod
    def get_magnitude(cls, event_type: str) -> float:
        return cls.MAGNITUDE_MAP.get(event_type, 0.3)

    @classmethod
    def get_direction(cls, event_type: str) -> float:
        return cls.DIRECTION_MAP.get(event_type, 0.0)


# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------

EARNINGS_BEAT_PATTERNS = [
    re.compile(r'(?:eps|earnings?|profit)\s+(?:of\s+)?(?:\$[\d.]+)\s+(?:beat|exceeded?|surpassed?|topped?|vs\.?\s+(?:estimate|consensus|expected)\s+of)\s+(?:\$[\d.]+)', re.I),
    re.compile(r'(?:beat|exceeded?|surpassed?)\s+(?:analyst|street|consensus|eps|earnings?)\s+(?:estimates?|expectations?|forecasts?|consensus)', re.I),
    re.compile(r'(?:eps|earnings?\s+per\s+share)\s+(?:of\s+)?(?:\$[\d.]+)\s+(?:vs\.?\s+)?(?:\$[\d.]+)\s+(?:estimate|expected|consensus)', re.I),
    re.compile(r'(?:quarterly\s+)?(?:earnings?|profits?)\s+(?:beat|exceeded?|topped?)\s+(?:expectations?|forecasts?|estimates?)', re.I),
]

EARNINGS_MISS_PATTERNS = [
    re.compile(r'(?:eps|earnings?|profit)\s+(?:of\s+)?(?:\$[\d.]+)\s+(?:miss|fell\s+short|below|under)\s+(?:\$[\d.]+)\s+(?:estimate|consensus|expected)', re.I),
    re.compile(r'(?:miss|fell\s+short|below)\s+(?:analyst|street|consensus|eps|earnings?)\s+(?:estimates?|expectations?|forecasts?)', re.I),
    re.compile(r'(?:quarterly\s+)?(?:earnings?|profits?)\s+(?:miss|fell\s+short|disappointed)', re.I),
]

GUIDANCE_RAISE_PATTERNS = [
    re.compile(r'(?:raised?|increased?|upped?|boosted?|lifted?)\s+(?:full.year\s+|annual\s+|fiscal\s+|Q\d\s+)?(?:guidance|outlook|forecast|estimate)', re.I),
    re.compile(r'(?:guidance|outlook|forecast)\s+(?:raised?|increased?|above)', re.I),
    re.compile(r'(?:now|currently)\s+expects?\s+(?:revenue|earnings?|eps)\s+(?:of\s+)?\$[\d.]+(?:\s*(?:to|-)\s*\$[\d.]+)?\s*(?:billion|million)', re.I),
]

GUIDANCE_CUT_PATTERNS = [
    re.compile(r'(?:lowered?|cut|reduced?|trimmed?|decreased?)\s+(?:full.year\s+|annual\s+|fiscal\s+|Q\d\s+)?(?:guidance|outlook|forecast|estimate)', re.I),
    re.compile(r'(?:guidance|outlook|forecast)\s+(?:lowered?|cut|reduced?|below)', re.I),
    re.compile(r'expect(?:s?|ed)\s+(?:revenue|earnings?|eps)\s+to\s+be\s+(?:lower|less|below)\s+than', re.I),
]

MA_PATTERNS = [
    re.compile(r'(?:agrees?\s+to|announced?|plans?\s+to)\s+(?:acquire|buy|purchase|merge\s+with|take\s+over)', re.I),
    re.compile(r'(?:acquisition|merger|takeover|buyout)\s+(?:of|for|at)', re.I),
    re.compile(r'(?:to\s+be\s+)?acquired\s+by\s+\w+\s+for', re.I),
    re.compile(r'(?:\$[\d.]+(?:\s*billion|\s*million)?)\s+(?:cash\s+)?(?:deal|transaction|offer|bid)', re.I),
]

MA_RUMOR_PATTERNS = [
    re.compile(r'(?:report(?:ed|edly)|said\s+to|exploring|considering|weighing)\s+(?:a\s+)?(?:sale|acquisition|merger|takeover)', re.I),
    re.compile(r'(?:M&A|merger)\s+(?:rumors?|speculation|reports?)', re.I),
    re.compile(r'(?:approaches?|in\s+talks?|in\s+discussions?)\s+(?:about|regarding|for)\s+(?:a\s+)?(?:potential\s+)?(?:deal|merger|acquisition)', re.I),
]

ANALYST_UPGRADE_PATTERNS = [
    re.compile(r'(?:upgraded?|raises?|initiated?\s+coverage)\s+(?:to\s+)?(?:buy|outperform|overweight|strong\s+buy|add)', re.I),
    re.compile(r'(?:raised?|increased?)\s+(?:target|price\s+target)\s+(?:to\s+)?\$[\d.]+', re.I),
    re.compile(r'(?:initiated?|started?)\s+at\s+(?:buy|outperform|overweight|strong\s+buy)', re.I),
]

ANALYST_DOWNGRADE_PATTERNS = [
    re.compile(r'(?:downgraded?|lowers?)\s+(?:to\s+)?(?:sell|underperform|underweight|neutral|hold)', re.I),
    re.compile(r'(?:lowered?|cut|reduced?)\s+(?:target|price\s+target)\s+(?:to\s+)?\$[\d.]+', re.I),
    re.compile(r'(?:initiated?|started?)\s+at\s+(?:sell|underperform|underweight)', re.I),
]

FED_PATTERNS = [
    re.compile(r'(?:federal\s+reserve|fed)\s+(?:raises?|increases?|hikes?|cuts?|lowers?|holds?|pauses?)\s+(?:interest\s+)?rates?', re.I),
    re.compile(r'fomc\s+(?:decision|meeting|statement|minutes)', re.I),
    re.compile(r'(?:interest\s+)?rates?\s+(?:raised?|increased?|cut|lowered?|unchanged)\s+(?:by\s+)?(?:\d+\s*basis\s+points|\d+%)', re.I),
]

REGULATORY_FINE_PATTERNS = [
    re.compile(r'(?:fined?|penalized?|charged?|sanctioned?)\s+\$[\d.]+(?:\s*billion|\s*million)?', re.I),
    re.compile(r'(?:settlement|fine|penalty|sanction)\s+of\s+\$[\d.]+(?:\s*billion|\s*million)?', re.I),
    re.compile(r'(?:sec|doj|ftc|cftc|occ)\s+(?:charges?|fines?|sues?|investigates?)', re.I),
]

REGULATORY_APPROVAL_PATTERNS = [
    re.compile(r'(?:fda|sec|ftc|eu|doj)\s+(?:approved?|cleared?|granted?)', re.I),
    re.compile(r'(?:received?|gets?|obtained?)\s+(?:regulatory\s+)?(?:approval|clearance|authorization)', re.I),
]

DIVIDEND_PATTERNS = [
    re.compile(r'(?:increases?|raises?|boosts?|initiates?)\s+(?:quarterly\s+)?(?:dividend|payout)', re.I),
    re.compile(r'declared?\s+(?:a\s+)?(?:quarterly\s+)?(?:special\s+)?dividend\s+of\s+\$[\d.]+', re.I),
]

BUYBACK_PATTERNS = [
    re.compile(r'(?:announces?|authorized?|approved?)\s+\$[\d.]+(?:\s*billion|\s*million)?\s+(?:share\s+)?(?:repurchase|buyback)', re.I),
]

LITIGATION_PATTERNS = [
    re.compile(r'(?:sued|lawsuit|class\s+action|litigation|legal\s+action)\s+(?:filed?|against|over)', re.I),
    re.compile(r'(?:faces?|subject\s+to)\s+(?:a\s+)?(?:lawsuit|class\s+action|investigation|probe)', re.I),
]


# ---------------------------------------------------------------------------
# Pattern-based event detector
# ---------------------------------------------------------------------------

class PatternEventDetector:
    """Detects financial events using regex pattern matching."""

    def detect(
        self,
        text: str,
        ticker: str = "",
        timestamp: Optional[datetime] = None,
    ) -> List[DetectedEvent]:
        """
        Scan text for financial events.
        Returns list of detected events.
        """
        ts = timestamp or datetime.now(timezone.utc)
        events = []

        checks = [
            (EARNINGS_BEAT_PATTERNS,      EventTypes.EARNINGS_BEAT,       +1.0, 0.80, "1d"),
            (EARNINGS_MISS_PATTERNS,      EventTypes.EARNINGS_MISS,       -1.0, 0.80, "1d"),
            (GUIDANCE_RAISE_PATTERNS,     EventTypes.GUIDANCE_RAISE,      +1.0, 0.65, "1d"),
            (GUIDANCE_CUT_PATTERNS,       EventTypes.GUIDANCE_CUT,        -1.0, 0.70, "1d"),
            (MA_PATTERNS,                 EventTypes.MA_ANNOUNCE,         +0.9, 0.85, "1d"),
            (MA_RUMOR_PATTERNS,           EventTypes.MA_RUMOR,            +0.5, 0.50, "1d"),
            (ANALYST_UPGRADE_PATTERNS,    EventTypes.ANALYST_UPGRADE,     +0.6, 0.50, "1d"),
            (ANALYST_DOWNGRADE_PATTERNS,  EventTypes.ANALYST_DOWNGRADE,   -0.6, 0.50, "1d"),
            (FED_PATTERNS,                EventTypes.FED_DECISION,        -0.4, 0.85, "4h"),
            (REGULATORY_FINE_PATTERNS,    EventTypes.REGULATORY_FINE,     -0.7, 0.55, "1d"),
            (REGULATORY_APPROVAL_PATTERNS, EventTypes.REGULATORY_APPROVAL, +0.8, 0.80, "1d"),
            (DIVIDEND_PATTERNS,           EventTypes.DIVIDEND_CHANGE,     +0.5, 0.50, "1w"),
            (BUYBACK_PATTERNS,            EventTypes.SHARE_REPURCHASE,    +0.5, 0.50, "1w"),
            (LITIGATION_PATTERNS,         EventTypes.LITIGATION,          -0.6, 0.50, "1d"),
        ]

        text_lower = text.lower()

        for patterns, event_type, default_direction, base_conf, horizon in checks:
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    # Refine direction based on specific words
                    direction = self._refine_direction(text_lower, event_type, default_direction)
                    magnitude = EventTypes.get_magnitude(event_type)
                    confidence = self._compute_confidence(text, match, base_conf)

                    event = DetectedEvent(
                        event_type=event_type,
                        subtype=self._classify_subtype(text_lower, event_type),
                        ticker=ticker,
                        confidence=confidence,
                        direction=direction,
                        magnitude=magnitude,
                        detected_at=ts,
                        source_text=match.group()[:200],
                        extracted_values=self._extract_values(text, event_type),
                        horizon=horizon,
                    )
                    events.append(event)
                    break  # Only detect each type once

        # Deduplicate
        return self._deduplicate(events)

    def _refine_direction(self, text: str, event_type: str, default: float) -> float:
        """Refine direction based on additional context."""
        if event_type == EventTypes.FED_DECISION:
            if re.search(r'cuts?\s+rates?|lowers?\s+rates?|easing', text, re.I):
                return +0.4  # rate cut = equity bullish
            elif re.search(r'raises?\s+rates?|hikes?\s+rates?|tightening', text, re.I):
                return -0.5  # rate hike = equity bearish
            elif re.search(r'holds?\s+rates?|pauses?|unchanged', text, re.I):
                return +0.1  # hold = neutral to slight positive

        if event_type == EventTypes.CREDIT_RATING:
            if re.search(r'upgraded?|raised?|improved?', text, re.I):
                return +0.6
            elif re.search(r'downgraded?|lowered?|cut|negative\s+outlook', text, re.I):
                return -0.6

        return default

    def _classify_subtype(self, text: str, event_type: str) -> str:
        """Get more specific subtype."""
        if event_type == EventTypes.EARNINGS_BEAT:
            if re.search(r'eps\s+beat|earnings\s+per\s+share', text):
                return "eps_beat"
            elif re.search(r'revenue\s+beat|sales\s+beat', text):
                return "revenue_beat"
            return "general_beat"

        if event_type == EventTypes.ANALYST_UPGRADE:
            if re.search(r'strong\s+buy|overweight', text):
                return "strong_upgrade"
            elif re.search(r'initiated?|coverage', text):
                return "initiation"
            return "upgrade"

        if event_type == EventTypes.MA_ANNOUNCE:
            if re.search(r'billion|bn', text):
                return "large_ma"
            return "ma"

        return event_type

    def _compute_confidence(self, text: str, match: re.Match, base: float) -> float:
        """Estimate detection confidence."""
        conf = base

        # Boost if multiple evidence in text
        if len(text) < 100:
            conf *= 0.8  # short text, less context
        elif len(text) > 500:
            conf = min(conf + 0.05, 1.0)

        # Boost if exact monetary values found
        if re.search(r'\$[\d.]+\s*(billion|million|B|M)', text, re.I):
            conf = min(conf + 0.05, 1.0)

        # Boost if ticker mentioned
        if re.search(r'\b[A-Z]{1,5}\b', text):
            conf = min(conf + 0.03, 1.0)

        return float(conf)

    def _extract_values(self, text: str, event_type: str) -> Dict[str, Any]:
        """Extract specific numeric values from text."""
        values = {}

        # Extract dollar amounts
        dollar_matches = re.findall(r'\$(\d+(?:\.\d+)?)\s*(billion|million|B|M)?', text, re.I)
        if dollar_matches:
            amounts = []
            for val_str, mult in dollar_matches:
                try:
                    val = float(val_str)
                    if mult and mult.upper() in ("BILLION", "B"):
                        val *= 1000
                    amounts.append(val)
                except ValueError:
                    pass
            if amounts:
                values["dollar_amounts_millions"] = amounts

        # Extract percentages
        pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent|basis\s+points?)', text, re.I)
        if pct_matches:
            values["percentages"] = [float(p) for p in pct_matches[:5]]

        # EPS specific
        if event_type in (EventTypes.EARNINGS_BEAT, EventTypes.EARNINGS_MISS):
            eps_match = re.search(r'\$(\d+\.\d{2})\s+(?:per|a)\s+(?:diluted\s+)?share', text, re.I)
            if eps_match:
                values["eps"] = float(eps_match.group(1))

        return values

    def _deduplicate(self, events: List[DetectedEvent]) -> List[DetectedEvent]:
        """Remove duplicate event types."""
        seen = set()
        unique = []
        for ev in events:
            if ev.event_type not in seen:
                seen.add(ev.event_type)
                unique.append(ev)
        return unique


# ---------------------------------------------------------------------------
# Earnings surprise detector
# ---------------------------------------------------------------------------

class EarningsSurpriseDetector:
    """
    Detects and quantifies earnings surprises.
    Requires consensus estimates for precise surprise calculation.
    """

    def __init__(self):
        self._pattern_detector = PatternEventDetector()

    def detect_from_text(
        self,
        text: str,
        ticker: str,
        consensus_eps: Optional[float] = None,
        consensus_revenue_millions: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[DetectedEvent]:
        """
        Detect earnings surprise and compute magnitude.
        """
        from ..scrapers.sec_edgar import EDGARFetcher, SECFiling
        ts = timestamp or datetime.now(timezone.utc)

        # Use pattern detection as base
        events = self._pattern_detector.detect(text, ticker, ts)
        earnings_events = [e for e in events if e.event_type in (EventTypes.EARNINGS_BEAT, EventTypes.EARNINGS_MISS)]

        if not earnings_events:
            return None

        event = earnings_events[0]

        # Try to extract actual EPS/revenue for precise surprise calculation
        values = event.extracted_values
        eps_actual = values.get("eps")
        amounts    = values.get("dollar_amounts_millions", [])

        if consensus_eps is not None and eps_actual is not None:
            eps_surprise = (eps_actual - consensus_eps) / abs(consensus_eps + 1e-8)
            event.extracted_values["eps_surprise_pct"] = float(eps_surprise * 100)

            # Refine magnitude based on actual surprise size
            surprise_mag = min(abs(eps_surprise) * 3, 1.0)  # 33% surprise = max magnitude
            event.magnitude = float(max(event.magnitude, surprise_mag))

            # Correct direction
            event.direction = +1.0 if eps_surprise > 0 else -1.0
            event.event_type = EventTypes.EARNINGS_BEAT if eps_surprise > 0 else EventTypes.EARNINGS_MISS

        if consensus_revenue_millions is not None and amounts:
            rev_actual = amounts[0]  # assume first large amount is revenue
            rev_surprise = (rev_actual - consensus_revenue_millions) / abs(consensus_revenue_millions + 1e-8)
            event.extracted_values["revenue_surprise_pct"] = float(rev_surprise * 100)

        return event

    def detect_from_filing(
        self,
        filing_data: Dict[str, Any],
        ticker: str,
        consensus: Optional[Dict[str, float]] = None,
    ) -> Optional[DetectedEvent]:
        """
        Detect earnings surprise from parsed SEC filing data.
        """
        ts = datetime.now(timezone.utc)
        eps_reported = filing_data.get("eps_reported")
        revenue = filing_data.get("revenue")

        if eps_reported is None and revenue is None:
            return None

        consensus = consensus or {}
        consensus_eps = consensus.get("eps")
        consensus_rev = consensus.get("revenue")

        # Determine beat/miss
        event_type = EventTypes.EARNINGS_BEAT  # default
        direction = +1.0
        magnitude = 0.6

        if consensus_eps is not None and eps_reported is not None:
            eps_surprise_pct = (eps_reported - consensus_eps) / abs(consensus_eps + 1e-8)
            if eps_surprise_pct < 0:
                event_type = EventTypes.EARNINGS_MISS
                direction = -1.0
            magnitude = min(abs(eps_surprise_pct) * 5, 1.0)

        extracted = {
            "eps_reported": eps_reported,
            "revenue_millions": revenue,
        }
        if consensus_eps is not None and eps_reported is not None:
            extracted["eps_surprise_pct"] = float((eps_reported - consensus_eps) / abs(consensus_eps + 1e-8) * 100)

        return DetectedEvent(
            event_type=event_type,
            subtype="filing_based",
            ticker=ticker,
            confidence=0.90,
            direction=direction,
            magnitude=magnitude,
            detected_at=ts,
            source_text=f"Earnings filing: EPS={eps_reported}, Revenue={revenue}M",
            extracted_values=extracted,
            horizon="1d",
        )


# ---------------------------------------------------------------------------
# M&A detector
# ---------------------------------------------------------------------------

class MADetector:
    """Specialized M&A event detector with premium/deal size extraction."""

    def detect(
        self,
        text: str,
        ticker: str = "",
        timestamp: Optional[datetime] = None,
    ) -> List[DetectedEvent]:
        ts = timestamp or datetime.now(timezone.utc)
        events = []

        # Check for announcement
        for pattern in MA_PATTERNS:
            if pattern.search(text):
                deal_size = self._extract_deal_size(text)
                premium = self._extract_premium(text)
                acquirer, target = self._extract_parties(text)

                event = DetectedEvent(
                    event_type=EventTypes.MA_ANNOUNCE,
                    subtype="acquisition" if acquirer else "merger",
                    ticker=target or ticker,
                    confidence=0.85,
                    direction=+0.9,  # target usually pops
                    magnitude=0.90,
                    detected_at=ts,
                    source_text=pattern.search(text).group()[:200],
                    extracted_values={
                        "deal_size_millions": deal_size,
                        "premium_pct": premium,
                        "acquirer": acquirer,
                        "target": target,
                    },
                    related_tickers=[acquirer, target] if acquirer and target else [],
                    horizon="1d",
                )
                events.append(event)
                break

        # Check for rumor
        for pattern in MA_RUMOR_PATTERNS:
            if pattern.search(text) and not events:
                event = DetectedEvent(
                    event_type=EventTypes.MA_RUMOR,
                    subtype="rumor",
                    ticker=ticker,
                    confidence=0.55,
                    direction=+0.5,
                    magnitude=0.50,
                    detected_at=ts,
                    source_text=pattern.search(text).group()[:200],
                    horizon="1d",
                )
                events.append(event)
                break

        return events

    def _extract_deal_size(self, text: str) -> Optional[float]:
        """Extract deal size in millions."""
        matches = re.findall(r'\$(\d+(?:\.\d+)?)\s*(billion|million|B|M)\b', text, re.I)
        sizes = []
        for val_str, mult in matches:
            val = float(val_str)
            if mult.upper() in ("BILLION", "B"):
                val *= 1000
            sizes.append(val)
        return max(sizes) if sizes else None

    def _extract_premium(self, text: str) -> Optional[float]:
        """Extract acquisition premium percentage."""
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s+(?:premium)',
            r'premium\s+of\s+(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s+(?:above|over)\s+(?:current|yesterday)',
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return None

    def _extract_parties(self, text: str) -> Tuple[str, str]:
        """Try to extract acquirer and target company names."""
        # Look for "X to acquire Y" or "X to buy Y"
        acq_match = re.search(
            r'([A-Z][a-zA-Z\s]+(?:Inc|Corp|Ltd|LLC|Company|Co)?)\.?\s+'
            r'(?:to\s+)?(?:acquire|buy|purchase|merge\s+with)\s+'
            r'([A-Z][a-zA-Z\s]+(?:Inc|Corp|Ltd|LLC|Company|Co)?)',
            text, re.I
        )
        if acq_match:
            return acq_match.group(1).strip(), acq_match.group(2).strip()
        return "", ""


# ---------------------------------------------------------------------------
# Analyst action detector
# ---------------------------------------------------------------------------

class AnalystActionDetector:
    """Detects analyst rating changes and price target updates."""

    def detect(
        self,
        text: str,
        ticker: str = "",
        timestamp: Optional[datetime] = None,
    ) -> List[DetectedEvent]:
        ts = timestamp or datetime.now(timezone.utc)
        events = []

        # Check upgrade
        for pattern in ANALYST_UPGRADE_PATTERNS:
            if pattern.search(text):
                firm, analyst = self._extract_analyst(text)
                old_rating, new_rating = self._extract_rating_change(text)
                target = self._extract_price_target(text)

                event = DetectedEvent(
                    event_type=EventTypes.ANALYST_UPGRADE,
                    subtype="upgrade",
                    ticker=ticker,
                    confidence=0.80,
                    direction=+0.60,
                    magnitude=0.45,
                    detected_at=ts,
                    source_text=pattern.search(text).group()[:200],
                    extracted_values={
                        "firm": firm,
                        "old_rating": old_rating,
                        "new_rating": new_rating,
                        "price_target": target,
                    },
                    horizon="1d",
                )
                events.append(event)
                break

        # Check downgrade
        for pattern in ANALYST_DOWNGRADE_PATTERNS:
            if pattern.search(text):
                firm, analyst = self._extract_analyst(text)
                old_rating, new_rating = self._extract_rating_change(text)
                target = self._extract_price_target(text)

                event = DetectedEvent(
                    event_type=EventTypes.ANALYST_DOWNGRADE,
                    subtype="downgrade",
                    ticker=ticker,
                    confidence=0.80,
                    direction=-0.60,
                    magnitude=0.45,
                    detected_at=ts,
                    source_text=pattern.search(text).group()[:200],
                    extracted_values={
                        "firm": firm,
                        "old_rating": old_rating,
                        "new_rating": new_rating,
                        "price_target": target,
                    },
                    horizon="1d",
                )
                events.append(event)
                break

        return events

    def _extract_analyst(self, text: str) -> Tuple[str, str]:
        known_firms = [
            "Goldman Sachs", "Morgan Stanley", "JPMorgan", "Bank of America",
            "Citi", "Citigroup", "Wells Fargo", "Deutsche Bank", "UBS",
            "Barclays", "Credit Suisse", "Raymond James", "Piper Sandler",
            "RBC Capital", "Oppenheimer", "Needham", "Jefferies", "Cowen",
            "KeyBanc", "Truist", "BofA Securities", "Mizuho",
        ]
        for firm in known_firms:
            if firm.lower() in text.lower():
                return firm, ""
        return "", ""

    def _extract_rating_change(self, text: str) -> Tuple[str, str]:
        ratings = ["buy", "sell", "hold", "neutral", "outperform", "underperform",
                   "overweight", "underweight", "market perform", "strong buy"]
        found = []
        for r in ratings:
            if r.lower() in text.lower():
                found.append(r)
        if len(found) >= 2:
            return found[0], found[1]
        elif len(found) == 1:
            return "", found[0]
        return "", ""

    def _extract_price_target(self, text: str) -> Optional[float]:
        patterns = [
            r'(?:price\s+)?target\s+(?:of\s+|to\s+)?\$(\d+(?:\.\d+)?)',
            r'PT\s+(?:to\s+|of\s+)?\$(\d+(?:\.\d+)?)',
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return None


# ---------------------------------------------------------------------------
# Composite event detector
# ---------------------------------------------------------------------------

class EventDetector:
    """
    Composite event detector that combines all sub-detectors.
    Main entry point for the NLP alpha pipeline.
    """

    def __init__(self):
        self._pattern = PatternEventDetector()
        self._ma = MADetector()
        self._analyst = AnalystActionDetector()
        self._earnings = EarningsSurpriseDetector()

    def detect_all(
        self,
        text: str,
        ticker: str = "",
        timestamp: Optional[datetime] = None,
        consensus: Optional[Dict[str, float]] = None,
    ) -> List[DetectedEvent]:
        """
        Run all detectors on text and return merged event list.
        """
        ts = timestamp or datetime.now(timezone.utc)
        all_events = []

        # Pattern-based (covers most event types)
        all_events.extend(self._pattern.detect(text, ticker, ts))

        # Specialized detectors (override/enhance pattern results)
        ma_events = self._ma.detect(text, ticker, ts)
        analyst_events = self._analyst.detect(text, ticker, ts)

        # Merge: specialized takes priority over pattern
        for ev in ma_events + analyst_events:
            existing_types = {e.event_type for e in all_events}
            if ev.event_type not in existing_types:
                all_events.append(ev)
            else:
                # Replace pattern result with specialized
                all_events = [e for e in all_events if e.event_type != ev.event_type]
                all_events.append(ev)

        # Sort by confidence descending
        all_events.sort(key=lambda e: -e.confidence)

        return all_events

    def detect_from_articles(
        self,
        articles: List[Dict[str, str]],  # list of {title, text, ticker}
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, List[DetectedEvent]]:
        """
        Detect events from a list of articles.
        Returns dict: ticker -> events.
        """
        from ..utils.text_processing import extract_tickers

        ticker_events: Dict[str, List[DetectedEvent]] = {}
        ts = timestamp or datetime.now(timezone.utc)

        for article in articles:
            text   = (article.get("title", "") + " " + article.get("text", ""))[:1000]
            ticker = article.get("ticker", "")

            # Auto-detect tickers if not provided
            if not ticker:
                tickers = extract_tickers(text)
                ticker = tickers[0] if tickers else ""

            events = self.detect_all(text, ticker, ts)
            if ticker:
                ticker_events.setdefault(ticker, []).extend(events)

        return ticker_events


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing event detector...")

    detector = EventDetector()

    test_cases = [
        ("AAPL beats Q3 earnings estimates: EPS $1.52 vs $1.45 expected. Revenue $89.5B up 8% YoY. Raises Q4 guidance.", "AAPL"),
        ("Microsoft to acquire Activision Blizzard for $68.7 billion in all-cash deal at $95 per share, 45% premium.", "MSFT"),
        ("Goldman Sachs downgrades Tesla to Sell from Neutral, cuts price target to $180 from $250.", "TSLA"),
        ("Federal Reserve raises interest rates by 25 basis points. FOMC signals two more hikes possible.", ""),
        ("Apple (AAPL) fined $1.8 billion by EU for violating antitrust regulations.", "AAPL"),
    ]

    for text, ticker in test_cases:
        events = detector.detect_all(text, ticker)
        print(f"\nText: {text[:80]}...")
        for ev in events:
            print(f"  [{ev.event_type:20s}] conf={ev.confidence:.2f} dir={ev.direction:+.1f} "
                  f"mag={ev.magnitude:.2f} | {ev.source_text[:50]}")

    print("\nEvent detector self-test passed.")
