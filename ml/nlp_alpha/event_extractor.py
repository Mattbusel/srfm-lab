"""
event_extractor.py -- Structured event extraction from financial text.

Pure rule-based / regex extraction. No ML models required.
Extracts typed events with confidence scores and historical impact estimates.

Event types covered:
    EARNINGS_BEAT, EARNINGS_MISS, PARTNERSHIP, HACK_EXPLOIT, REGULATORY_ACTION,
    UPGRADE_DOWNGRADE, TOKEN_LAUNCH, LISTING, DELISTING, WHALE_MOVE, FORK
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# EventType enum
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    EARNINGS_BEAT = "EARNINGS_BEAT"
    EARNINGS_MISS = "EARNINGS_MISS"
    PARTNERSHIP = "PARTNERSHIP"
    HACK_EXPLOIT = "HACK_EXPLOIT"
    REGULATORY_ACTION = "REGULATORY_ACTION"
    UPGRADE_DOWNGRADE = "UPGRADE_DOWNGRADE"
    TOKEN_LAUNCH = "TOKEN_LAUNCH"
    LISTING = "LISTING"
    DELISTING = "DELISTING"
    WHALE_MOVE = "WHALE_MOVE"
    FORK = "FORK"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Historical impact estimates (median 24h price move from research)
# Signs: positive = price goes up, negative = price goes down
# ---------------------------------------------------------------------------

HISTORICAL_IMPACT: Dict[EventType, float] = {
    EventType.EARNINGS_BEAT: +0.04,
    EventType.EARNINGS_MISS: -0.05,
    EventType.PARTNERSHIP: +0.03,
    EventType.HACK_EXPLOIT: -0.15,
    EventType.REGULATORY_ACTION: -0.12,
    EventType.UPGRADE_DOWNGRADE: +0.02,   # net of upgrade vs downgrade (see below)
    EventType.TOKEN_LAUNCH: +0.05,
    EventType.LISTING: +0.08,
    EventType.DELISTING: -0.10,
    EventType.WHALE_MOVE: -0.02,          # usually mild negative (sell pressure signal)
    EventType.FORK: +0.03,
    EventType.UNKNOWN: 0.00,
}

# Upgrade vs downgrade split -- used internally
_UPGRADE_IMPACT: float = +0.04
_DOWNGRADE_IMPACT: float = -0.06


# ---------------------------------------------------------------------------
# ExtractedEvent dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEvent:
    """A structured event extracted from raw financial text."""
    event_type: EventType
    symbol: str
    confidence: float
    impact_estimate: float          # from historical calibration
    raw_text: str
    subtype: str = ""               # e.g. "upgrade" vs "downgrade"
    extracted_at: Optional[datetime] = None
    magnitude_hint: Optional[float] = None   # dollar amount or % if extracted
    exchange_name: str = ""         # for LISTING / DELISTING
    amount_usd: Optional[float] = None       # for WHALE_MOVE / HACK_EXPLOIT

    def __post_init__(self) -> None:
        if self.extracted_at is None:
            self.extracted_at = datetime.now(timezone.utc)

    def decayed_impact(self, age_minutes: float, half_life_minutes: float = 120.0) -> float:
        """
        Return impact estimate decayed by age.
        Default half-life = 2 hours (120 minutes).
        """
        lam = math.log(2.0) / half_life_minutes
        return self.impact_estimate * math.exp(-lam * age_minutes)

    def weighted_signal(self) -> float:
        """Confidence-weighted impact estimate."""
        return self.impact_estimate * self.confidence


# ---------------------------------------------------------------------------
# CryptoEventPatterns
# ---------------------------------------------------------------------------

class CryptoEventPatterns:
    """
    Compiled regex patterns for crypto-specific event detection.
    Each pattern set returns (match, subtype, confidence_boost).
    """

    # -- Hack / Exploit patterns --
    HACK_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        (re.compile(r"\bhack(?:ed|er|ing|s)?\b", re.I), "hack", 0.85),
        (re.compile(r"\bexploit(?:ed|s|ing)?\b", re.I), "exploit", 0.85),
        (re.compile(r"\bbreach(?:ed|ing)?\b", re.I), "breach", 0.75),
        (re.compile(r"\bstolen\b", re.I), "stolen", 0.80),
        (re.compile(r"\bfunds?\s+drain(?:ed|ing|s)?\b", re.I), "drained", 0.85),
        (re.compile(r"\bvulnerabilit(?:y|ies)\b", re.I), "vulnerability", 0.65),
        (re.compile(r"\battack(?:ed|s|ing)?\b", re.I), "attack", 0.60),
        (re.compile(r"\bcompromis(?:ed|ing)\b", re.I), "compromised", 0.70),
        (re.compile(r"\bflash\s+loan\s+attack\b", re.I), "flash_loan", 0.90),
        (re.compile(r"\brug\s*pull\b", re.I), "rugpull", 0.95),
        (re.compile(r"\b(?:\$[\d,\.]+[MmBbKk]?)\s+(?:stolen|drained|lost|missing)\b", re.I), "dollar_stolen", 0.90),
        (re.compile(r"\bsecurity\s+incident\b", re.I), "security_incident", 0.70),
    ]

    # -- Listing patterns --
    LISTING_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        (re.compile(r"\blisted\s+on\b", re.I), "listing", 0.90),
        (re.compile(r"\bnow\s+(?:available|trading)\s+on\b", re.I), "listing", 0.85),
        (re.compile(r"\btrading\s+(?:live|now|starts?)\s+on\b", re.I), "listing", 0.85),
        (re.compile(r"\badds?\s+(?:support\s+for\s+)?(?:trading\s+of\s+)?[A-Z]{2,8}\b", re.I), "listing", 0.70),
        (re.compile(r"\b(?:binance|coinbase|kraken|okx|bybit|kucoin|ftx|huobi|gate\.io|gemini)\s+(?:lists?|adds?|supports?)\b", re.I), "major_exchange_listing", 0.92),
        (re.compile(r"\blisting\s+(?:announcement|confirmed|date)\b", re.I), "listing_announcement", 0.80),
        (re.compile(r"\bwill\s+(?:list|trade)\s+on\b", re.I), "upcoming_listing", 0.75),
    ]

    # -- Delisting patterns --
    DELISTING_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        (re.compile(r"\bdelist(?:ed|ing|s)?\b", re.I), "delisting", 0.90),
        (re.compile(r"\bremov(?:ed?|ing)\s+from\s+(?:trading|exchange|platform)\b", re.I), "removal", 0.85),
        (re.compile(r"\btrading\s+suspended\b", re.I), "suspension", 0.80),
        (re.compile(r"\bwithdrawal\s+(?:disabled|suspended|halted)\b", re.I), "withdrawal_suspended", 0.75),
    ]

    # -- Whale move patterns --
    WHALE_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        (re.compile(r"\bwhale\s+alert\b", re.I), "whale_alert", 0.90),
        (re.compile(r"\b(?:\d[\d,\.]*)\s+(?:BTC|ETH|SOL|ADA|BNB)\s+(?:moved?|transferred?|sent)\b", re.I), "large_transfer", 0.85),
        (re.compile(r"\b\$[\d,\.]+[MmBb]\s+(?:worth\s+of\s+)?(?:BTC|ETH|crypto|coins?)\s+(?:moved?|transferred?|sent)\b", re.I), "large_dollar_transfer", 0.85),
        (re.compile(r"\bwhale\s+(?:wallet|address|accumulating|selling|bought|sold)\b", re.I), "whale_activity", 0.80),
        (re.compile(r"\blarge\s+(?:transfer|withdrawal|deposit|transaction)\b", re.I), "large_transaction", 0.65),
        (re.compile(r"\bdormant\s+wallet\s+(?:moves?|transferred?|active)\b", re.I), "dormant_wallet", 0.80),
    ]

    # -- Partnership patterns --
    PARTNERSHIP_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        (re.compile(r"\bpartner(?:ship|s|ed|ing)?\b", re.I), "partnership", 0.75),
        (re.compile(r"\bcollaborat(?:ion|ing|ed|e)\b", re.I), "collaboration", 0.70),
        (re.compile(r"\bintegrat(?:ion|ed|ing|es?)\b", re.I), "integration", 0.65),
        (re.compile(r"\bjoint\s+venture\b", re.I), "joint_venture", 0.80),
        (re.compile(r"\b(?:signs?|signed|inking|ink)\s+(?:deal|agreement|mou|contract)\b", re.I), "signed_deal", 0.80),
        (re.compile(r"\bstrategic\s+alliance\b", re.I), "strategic_alliance", 0.75),
        (re.compile(r"\bteams?\s+up\s+with\b", re.I), "team_up", 0.75),
    ]

    # -- Fork patterns --
    FORK_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        (re.compile(r"\bhard\s+fork\b", re.I), "hard_fork", 0.92),
        (re.compile(r"\bsoft\s+fork\b", re.I), "soft_fork", 0.85),
        (re.compile(r"\bnetwork\s+upgrade\b", re.I), "network_upgrade", 0.75),
        (re.compile(r"\bprotocol\s+upgrade\b", re.I), "protocol_upgrade", 0.70),
        (re.compile(r"\bchain\s+split\b", re.I), "chain_split", 0.90),
        (re.compile(r"\bhalving\b", re.I), "halving", 0.95),
    ]

    # -- Token launch patterns --
    TOKEN_LAUNCH_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        (re.compile(r"\btoken\s+launch(?:ed|ing|es?)?\b", re.I), "token_launch", 0.85),
        (re.compile(r"\binitial\s+(?:coin|token|dex|exchange)\s+offering\b", re.I), "ico_ido", 0.85),
        (re.compile(r"\bICO\b"), "ico", 0.80),
        (re.compile(r"\bIDO\b"), "ido", 0.80),
        (re.compile(r"\bIEO\b"), "ieo", 0.80),
        (re.compile(r"\bairdrop\b", re.I), "airdrop", 0.75),
        (re.compile(r"\btoken\s+generation\s+event\b", re.I), "tge", 0.85),
        (re.compile(r"\bnew\s+token\s+(?:released?|deployed?|launched?)\b", re.I), "new_token", 0.80),
        (re.compile(r"\bmainnet\s+launch(?:ed)?\b", re.I), "mainnet_launch", 0.90),
    ]

    # -- Regulatory patterns --
    REGULATORY_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        (re.compile(r"\bsec\s+(?:charges?|sues?|files?|investigat|subpoena|enforcement)\b", re.I), "sec_action", 0.90),
        (re.compile(r"\bregulatory\s+(?:action|enforcement|crackdown|ban|restriction)\b", re.I), "regulatory_action", 0.85),
        (re.compile(r"\bban(?:ned|ning|s)?\s+(?:crypto|bitcoin|trading|exchange)\b", re.I), "ban", 0.88),
        (re.compile(r"\bcftc\s+(?:charges?|files?|action|sues?)\b", re.I), "cftc_action", 0.88),
        (re.compile(r"\bfinra\s+(?:charges?|fines?|action)\b", re.I), "finra_action", 0.85),
        (re.compile(r"\bcourt\s+order(?:ed|s)?\b", re.I), "court_order", 0.80),
        (re.compile(r"\banti-money\s+laundering\b", re.I), "aml", 0.70),
        (re.compile(r"\bknow\s+your\s+customer\b", re.I), "kyc", 0.60),
        (re.compile(r"\bsanction(?:ed|s|ing)?\b", re.I), "sanctions", 0.85),
        (re.compile(r"\bfine(?:d|s)?\s+\$[\d,\.]+\b", re.I), "fined", 0.85),
        (re.compile(r"\blegal\s+action\b", re.I), "legal_action", 0.75),
        (re.compile(r"\barrest(?:ed|s|ing)?\b", re.I), "arrest", 0.85),
    ]

    # -- Earnings patterns --
    EARNINGS_BEAT_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        (re.compile(r"\beat(?:s|ing)?\s+(?:estimate|consensus|expectation|forecast)\b", re.I), "beat_estimate", 0.88),
        (re.compile(r"\bsurpass(?:ed?|ing|es?)?\s+(?:estimate|consensus|expectation)\b", re.I), "surpassed", 0.85),
        (re.compile(r"\bexceed(?:ed?|ing|s?)?\s+(?:estimate|consensus|expectation|forecast)\b", re.I), "exceeded", 0.85),
        (re.compile(r"\bbetter\s+than\s+expected\b", re.I), "better_than_expected", 0.80),
        (re.compile(r"\bupside\s+surprise\b", re.I), "upside_surprise", 0.85),
        (re.compile(r"\brecord\s+(?:revenue|earnings|profit|eps)\b", re.I), "record_earnings", 0.80),
        (re.compile(r"\braises?\s+(?:guidance|outlook|forecast)\b", re.I), "raises_guidance", 0.82),
    ]

    EARNINGS_MISS_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        (re.compile(r"\bmiss(?:es?|ed|ing)?\s+(?:estimate|consensus|expectation|forecast)\b", re.I), "missed_estimate", 0.88),
        (re.compile(r"\bfell?\s+short\s+of\b", re.I), "fell_short", 0.82),
        (re.compile(r"\bworse\s+than\s+expected\b", re.I), "worse_than_expected", 0.80),
        (re.compile(r"\bdisappointing\s+(?:results?|earnings?|revenue|eps)\b", re.I), "disappointing", 0.82),
        (re.compile(r"\bdownside\s+surprise\b", re.I), "downside_surprise", 0.85),
        (re.compile(r"\bcuts?\s+(?:guidance|outlook|forecast)\b", re.I), "cuts_guidance", 0.82),
        (re.compile(r"\bbelow\s+(?:estimate|consensus|expectation)\b", re.I), "below_estimate", 0.80),
    ]

    # -- Upgrade / Downgrade patterns --
    UPGRADE_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        (re.compile(r"\bupgrad(?:ed?|ing|es?)\s+(?:to\s+)?(?:buy|outperform|overweight|strong buy|accumulate)\b", re.I), "upgrade_buy", 0.88),
        (re.compile(r"\binitiates?\s+(?:with\s+)?(?:buy|outperform|overweight|strong buy)\b", re.I), "initiate_buy", 0.85),
        (re.compile(r"\brais(?:ed?|ing|es?)\s+(?:price\s+)?target\b", re.I), "raised_target", 0.80),
        (re.compile(r"\bbullish\s+(?:note|call|thesis|outlook)\b", re.I), "bullish_call", 0.78),
    ]

    DOWNGRADE_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        (re.compile(r"\bdowngrad(?:ed?|ing|es?)\s+(?:to\s+)?(?:sell|underperform|underweight|neutral|hold|reduce)\b", re.I), "downgrade_sell", 0.88),
        (re.compile(r"\binitiates?\s+(?:with\s+)?(?:sell|underperform|underweight)\b", re.I), "initiate_sell", 0.85),
        (re.compile(r"\bcut(?:s|ting)?\s+(?:price\s+)?target\b", re.I), "cut_target", 0.80),
        (re.compile(r"\bbearish\s+(?:note|call|thesis|outlook)\b", re.I), "bearish_call", 0.78),
    ]

    # -- Dollar amount extractor (for quantifying hack / whale events) --
    DOLLAR_AMOUNT_PATTERN: re.Pattern = re.compile(
        r"\$\s*([\d,\.]+)\s*([MmBbKkTt]?)", re.I
    )

    @classmethod
    def extract_dollar_amount(cls, text: str) -> Optional[float]:
        """
        Extract largest dollar amount mentioned in text.
        Returns amount in USD (float) or None if not found.
        """
        matches = cls.DOLLAR_AMOUNT_PATTERN.findall(text)
        if not matches:
            return None
        amounts: List[float] = []
        suffix_mult = {"m": 1e6, "b": 1e9, "k": 1e3, "t": 1e12}
        for num_str, suffix in matches:
            try:
                num = float(num_str.replace(",", ""))
                mult = suffix_mult.get(suffix.lower(), 1.0)
                amounts.append(num * mult)
            except ValueError:
                continue
        return max(amounts) if amounts else None

    @classmethod
    def extract_exchange_name(cls, text: str) -> str:
        """Extract the name of an exchange mentioned in text."""
        exchanges = [
            "Binance", "Coinbase", "Kraken", "OKX", "Bybit",
            "KuCoin", "FTX", "Huobi", "Gate.io", "Gemini",
            "Bitfinex", "Bitstamp", "Bittrex", "Poloniex",
            "Uniswap", "dYdX", "Curve",
        ]
        text_lower = text.lower()
        for ex in exchanges:
            if ex.lower() in text_lower:
                return ex
        return ""


# ---------------------------------------------------------------------------
# EventExtractor
# ---------------------------------------------------------------------------

class EventExtractor:
    """
    Rule-based event extractor for financial text.

    Applies ordered pattern sets for each EventType and returns a list
    of ExtractedEvent objects with confidence scores and impact estimates.
    """

    def __init__(self) -> None:
        self._patterns = CryptoEventPatterns()

    def _run_patterns(
        self,
        text: str,
        patterns: List[Tuple[re.Pattern, str, float]],
        event_type: EventType,
        symbol: str,
        base_impact: float,
    ) -> Optional[ExtractedEvent]:
        """
        Try all patterns for a given event type.
        Returns the highest-confidence match or None.
        """
        best_confidence = 0.0
        best_subtype = ""

        for pattern, subtype, confidence in patterns:
            if pattern.search(text):
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_subtype = subtype

        if best_confidence == 0.0:
            return None

        # Adjust impact for upgrade vs downgrade
        actual_impact = base_impact
        if event_type == EventType.UPGRADE_DOWNGRADE:
            if "downgrad" in best_subtype or "sell" in best_subtype or "cut" in best_subtype:
                actual_impact = _DOWNGRADE_IMPACT
            else:
                actual_impact = _UPGRADE_IMPACT

        return ExtractedEvent(
            event_type=event_type,
            symbol=symbol,
            confidence=best_confidence,
            impact_estimate=actual_impact,
            raw_text=text[:300],
            subtype=best_subtype,
            extracted_at=datetime.now(timezone.utc),
            amount_usd=self._patterns.extract_dollar_amount(text)
            if event_type in (EventType.HACK_EXPLOIT, EventType.WHALE_MOVE)
            else None,
            exchange_name=self._patterns.extract_exchange_name(text)
            if event_type in (EventType.LISTING, EventType.DELISTING)
            else "",
        )

    def extract(self, text: str, symbol: str = "") -> List[ExtractedEvent]:
        """
        Extract all events from a piece of text.

        Parameters
        ----------
        text : str
            Raw headline or article text.
        symbol : str
            Primary trading symbol context.

        Returns
        -------
        list of ExtractedEvent, sorted by confidence descending.
        """
        events: List[ExtractedEvent] = []
        p = self._patterns

        # Ordered by impact severity for priority
        candidates = [
            (p.HACK_PATTERNS, EventType.HACK_EXPLOIT, HISTORICAL_IMPACT[EventType.HACK_EXPLOIT]),
            (p.REGULATORY_PATTERNS, EventType.REGULATORY_ACTION, HISTORICAL_IMPACT[EventType.REGULATORY_ACTION]),
            (p.DELISTING_PATTERNS, EventType.DELISTING, HISTORICAL_IMPACT[EventType.DELISTING]),
            (p.WHALE_PATTERNS, EventType.WHALE_MOVE, HISTORICAL_IMPACT[EventType.WHALE_MOVE]),
            (p.EARNINGS_MISS_PATTERNS, EventType.EARNINGS_MISS, HISTORICAL_IMPACT[EventType.EARNINGS_MISS]),
            (p.DOWNGRADE_PATTERNS, EventType.UPGRADE_DOWNGRADE, _DOWNGRADE_IMPACT),
            (p.FORK_PATTERNS, EventType.FORK, HISTORICAL_IMPACT[EventType.FORK]),
            (p.TOKEN_LAUNCH_PATTERNS, EventType.TOKEN_LAUNCH, HISTORICAL_IMPACT[EventType.TOKEN_LAUNCH]),
            (p.LISTING_PATTERNS, EventType.LISTING, HISTORICAL_IMPACT[EventType.LISTING]),
            (p.EARNINGS_BEAT_PATTERNS, EventType.EARNINGS_BEAT, HISTORICAL_IMPACT[EventType.EARNINGS_BEAT]),
            (p.UPGRADE_PATTERNS, EventType.UPGRADE_DOWNGRADE, _UPGRADE_IMPACT),
            (p.PARTNERSHIP_PATTERNS, EventType.PARTNERSHIP, HISTORICAL_IMPACT[EventType.PARTNERSHIP]),
        ]

        for patterns, event_type, base_impact in candidates:
            event = self._run_patterns(text, patterns, event_type, symbol, base_impact)
            if event:
                events.append(event)

        # Sort by confidence descending
        events.sort(key=lambda e: -e.confidence)

        # Deduplicate -- remove lower-confidence event if same type
        seen_types: set = set()
        deduped: List[ExtractedEvent] = []
        for ev in events:
            if ev.event_type not in seen_types:
                deduped.append(ev)
                seen_types.add(ev.event_type)

        return deduped

    def extract_batch(
        self, texts: List[str], symbol: str = ""
    ) -> List[List[ExtractedEvent]]:
        """Extract events from a list of texts."""
        return [self.extract(t, symbol) for t in texts]

    def extract_with_context(
        self,
        text: str,
        symbol: str = "",
        timestamp: Optional[datetime] = None,
    ) -> List[ExtractedEvent]:
        """
        Extract events and stamp with a provided timestamp.
        Useful when processing historical data with known publish times.
        """
        events = self.extract(text, symbol)
        if timestamp:
            for ev in events:
                ev.extracted_at = timestamp
        return events


# ---------------------------------------------------------------------------
# EventSignalGenerator
# ---------------------------------------------------------------------------

class EventSignalGenerator:
    """
    Converts a list of extracted events into a single alpha signal.

    Signal = weighted sum of (impact_estimate * confidence * decay_factor)
    Decay: exponential with configurable half-life (default 2 hours).
    Output is clamped to [-1, 1].
    """

    DEFAULT_HALF_LIFE_MINUTES: float = 120.0  # 2-hour half-life

    def __init__(self, half_life_minutes: float = DEFAULT_HALF_LIFE_MINUTES) -> None:
        self.half_life = half_life_minutes
        self._lambda = math.log(2.0) / half_life_minutes

    def _decay(self, age_minutes: float) -> float:
        """Exponential decay factor."""
        return math.exp(-self._lambda * age_minutes)

    def _age_minutes(
        self, event: ExtractedEvent, now: Optional[datetime] = None
    ) -> float:
        """Age of event in minutes."""
        now = now or datetime.now(timezone.utc)
        if event.extracted_at is None:
            return 0.0
        ts = event.extracted_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return max(0.0, (now - ts).total_seconds() / 60.0)

    def generate_signal(
        self,
        events: List[ExtractedEvent],
        symbol: str,
        now: Optional[datetime] = None,
    ) -> float:
        """
        Compute alpha signal for a given symbol from extracted events.

        Filters to events matching the symbol, computes confidence-weighted,
        time-decayed impact sum, and normalizes to [-1, 1].

        Returns 0.0 if no relevant events.
        """
        symbol_upper = symbol.upper()
        relevant = [
            ev for ev in events
            if not ev.symbol or ev.symbol.upper() == symbol_upper
        ]

        if not relevant:
            return 0.0

        total_weight = 0.0
        weighted_impact = 0.0

        for ev in relevant:
            age = self._age_minutes(ev, now)
            decay = self._decay(age)
            w = ev.confidence * decay
            weighted_impact += ev.impact_estimate * w
            total_weight += w

        if total_weight == 0.0:
            return 0.0

        # Normalize: impact estimates are typically small (~0.05-0.15)
        # Scale up before tanh to make signal responsive
        raw = weighted_impact / total_weight
        signal = math.tanh(raw * 8.0)   # scale factor tuned to impact range
        return max(-1.0, min(1.0, signal))

    def get_active_events(
        self,
        events: List[ExtractedEvent],
        symbol: str,
        max_age_minutes: float = 240.0,   # 4 hours
        now: Optional[datetime] = None,
    ) -> List[ExtractedEvent]:
        """Return events for symbol that are still within the active window."""
        symbol_upper = symbol.upper()
        now = now or datetime.now(timezone.utc)
        active = []
        for ev in events:
            if ev.symbol and ev.symbol.upper() != symbol_upper:
                continue
            if self._age_minutes(ev, now) <= max_age_minutes:
                active.append(ev)
        return active

    def summarize(
        self,
        events: List[ExtractedEvent],
        symbol: str,
        now: Optional[datetime] = None,
    ) -> Dict:
        """
        Return a human-readable summary of recent events for a symbol.
        """
        active = self.get_active_events(events, symbol, now=now)
        signal = self.generate_signal(events, symbol, now)

        return {
            "symbol": symbol,
            "signal": signal,
            "n_active_events": len(active),
            "events": [
                {
                    "type": ev.event_type.value,
                    "subtype": ev.subtype,
                    "confidence": ev.confidence,
                    "impact": ev.impact_estimate,
                    "age_minutes": self._age_minutes(ev, now),
                }
                for ev in active
            ],
        }


# ---------------------------------------------------------------------------
# Multi-symbol batch extraction
# ---------------------------------------------------------------------------

def extract_events_for_symbols(
    texts: List[str],
    symbols: List[str],
    timestamps: Optional[List[Optional[datetime]]] = None,
) -> Dict[str, List[ExtractedEvent]]:
    """
    Batch-extract events from a list of texts and assign to matching symbols.

    Returns dict mapping symbol -> list of ExtractedEvent.
    Texts are matched to all symbols (simple substring check).
    """
    extractor = EventExtractor()
    results: Dict[str, List[ExtractedEvent]] = {s: [] for s in symbols}

    for i, text in enumerate(texts):
        ts = timestamps[i] if timestamps else None
        text_upper = text.upper()

        # Determine which symbols this text mentions
        matched_symbols = []
        for sym in symbols:
            if sym.upper() in text_upper:
                matched_symbols.append(sym)

        # If none matched, try all (broadcast)
        if not matched_symbols:
            matched_symbols = symbols

        for sym in matched_symbols:
            events = extractor.extract_with_context(text, sym, ts)
            results[sym].extend(events)

    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    extractor = EventExtractor()
    gen = EventSignalGenerator()

    test_inputs = [
        ("Binance exchange hacked, $150M in funds drained overnight", "BNB"),
        ("Bitcoin now listed on new Nasdaq ETF", "BTC"),
        ("Ethereum hard fork scheduled for next month", "ETH"),
        ("SEC charges Ripple with unregistered securities offering", "XRP"),
        ("Apple beats Q3 EPS estimates by 12%, raises full-year guidance", "AAPL"),
        ("Tesla misses revenue targets, cuts 2024 delivery forecast", "TSLA"),
        ("Goldman Sachs upgrades Nvidia to Buy, raises price target to $900", "NVDA"),
        ("Whale alert: 10,000 BTC moved from dormant wallet", "BTC"),
        ("Solana mainnet launch confirmed for December", "SOL"),
        ("New partnership signed between Chainlink and major bank", "LINK"),
    ]

    print("Event Extractor Test\n" + "=" * 50)
    for text, symbol in test_inputs:
        events = extractor.extract(text, symbol)
        sig = gen.generate_signal(events, symbol)
        print(f"\n[{symbol}] {text[:70]}")
        for ev in events:
            print(f"  -> {ev.event_type.value:25s} | conf={ev.confidence:.2f} | impact={ev.impact_estimate:+.3f}")
        print(f"  Signal: {sig:+.4f}")
