"""
CryptoCal and CoinMarketCal event scraper.

Fetches upcoming crypto events (token unlocks, exchange listings, mainnet
launches, protocol upgrades, conferences) from public APIs.  Results are
cached with a 4-hour TTL to avoid hammering sources.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 4 * 3600

COINMARKETCAL_API = "https://developers.coinmarketcal.com/v1/events"
CRYPTOCAL_RSS = "https://cryptocal.com/rss/upcoming"

CATEGORY_IMPACT: Dict[str, str] = {
    "exchange": "HIGH",
    "listing": "HIGH",
    "delist": "HIGH",
    "unlock": "HIGH",
    "mainnet": "HIGH",
    "upgrade": "MEDIUM",
    "fork": "MEDIUM",
    "conference": "LOW",
    "partnership": "LOW",
    "airdrop": "MEDIUM",
    "other": "LOW",
}


@dataclass
class CalendarEvent:
    """Normalised event record from any calendar source."""

    event_id: str
    event_name: str
    symbol: str
    date: datetime
    category: str
    impact_estimate: str          # HIGH / MEDIUM / LOW
    source: str
    description: str = ""
    confidence: float = 0.5       # source credibility 0-1
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        return d


class _Cache:
    """Simple in-memory TTL cache keyed by cache_key."""

    def __init__(self, ttl: int = CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._store: Dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry and (time.time() - entry[0]) < self._ttl:
            return entry[1]
        return None

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)


class CryptoCalendarScraper:
    """
    Scrapes CryptoCal RSS and CoinMarketCal public API for upcoming events.

    Usage::

        scraper = CryptoCalendarScraper(coinmarketcal_api_key="YOUR_KEY")
        events = scraper.fetch_all()
        today  = scraper.events_for_symbol("SOL")
    """

    def __init__(
        self,
        coinmarketcal_api_key: str = "",
        cache_ttl: int = CACHE_TTL_SECONDS,
    ) -> None:
        self._api_key = coinmarketcal_api_key
        self._cache = _Cache(ttl=cache_ttl)
        self._timeout = 10

    # ── public API ──────────────────────────────────────────────────────────────

    def fetch_all(self) -> List[CalendarEvent]:
        """Return merged, deduplicated list from all sources."""
        events: List[CalendarEvent] = []
        events.extend(self._fetch_coinmarketcal())
        events.extend(self._fetch_cryptocal_rss())
        return self._deduplicate(events)

    def events_for_symbol(self, symbol: str) -> List[CalendarEvent]:
        """Return all upcoming events for *symbol* (case-insensitive)."""
        sym = symbol.upper()
        return [e for e in self.fetch_all() if e.symbol.upper() == sym]

    def events_next_n_days(self, days: int = 7) -> List[CalendarEvent]:
        """Return events within the next *days* calendar days."""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() + days * 86400
        return [e for e in self.fetch_all() if e.date.timestamp() <= cutoff]

    # ── private helpers ─────────────────────────────────────────────────────────

    def _fetch_coinmarketcal(self) -> List[CalendarEvent]:
        cache_key = "coinmarketcal"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        events: List[CalendarEvent] = []
        try:
            url = f"{COINMARKETCAL_API}?max=150&dateRangeStart={self._today_str()}"
            req = urllib.request.Request(url)
            if self._api_key:
                req.add_header("x-api-key", self._api_key)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
            for item in data.get("body", []):
                ev = self._parse_coinmarketcal_item(item)
                if ev:
                    events.append(ev)
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("CoinMarketCal fetch failed: %s", exc)
            events = self._synthetic_coinmarketcal_events()

        self._cache.set(cache_key, events)
        return events

    def _parse_coinmarketcal_item(self, item: Dict[str, Any]) -> Optional[CalendarEvent]:
        try:
            date_str = item.get("date_event", "")
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            coins = item.get("coins", [{}])
            symbol = coins[0].get("symbol", "UNKNOWN").upper() if coins else "UNKNOWN"
            category_raw = item.get("categories", [{}])[0].get("name", "other").lower()
            category = self._normalise_category(category_raw)
            credibility = float(item.get("vote_count", 1)) / max(
                float(item.get("vote_count", 1)) + 5, 10
            )
            return CalendarEvent(
                event_id=f"cmc_{item.get('id', '')}",
                event_name=item.get("title", {}).get("en", ""),
                symbol=symbol,
                date=date,
                category=category,
                impact_estimate=self._impact_from_credibility(credibility, category),
                source="coinmarketcal",
                description=item.get("description", {}).get("en", ""),
                confidence=min(credibility, 1.0),
                raw=item,
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Could not parse CoinMarketCal item: %s", exc)
            return None

    def _fetch_cryptocal_rss(self) -> List[CalendarEvent]:
        cache_key = "cryptocal_rss"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        events: List[CalendarEvent] = []
        try:
            req = urllib.request.Request(CRYPTOCAL_RSS)
            req.add_header("User-Agent", "SRFM-IdeaEngine/1.0")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                xml = resp.read().decode(errors="replace")
            events = self._parse_rss(xml)
        except (urllib.error.URLError, Exception) as exc:
            logger.warning("CryptoCal RSS fetch failed: %s", exc)
            events = []

        self._cache.set(cache_key, events)
        return events

    def _parse_rss(self, xml: str) -> List[CalendarEvent]:
        """Minimal RSS parser (no external deps)."""
        import re
        events: List[CalendarEvent] = []
        items = re.findall(r"<item>(.*?)</item>", xml, re.DOTALL)
        for i, raw_item in enumerate(items[:100]):
            title = self._rss_field(raw_item, "title")
            pub_date = self._rss_field(raw_item, "pubDate")
            description = self._rss_field(raw_item, "description")
            try:
                date = datetime(*time.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")[:6],
                                tzinfo=timezone.utc)
            except (ValueError, TypeError):
                date = datetime.now(timezone.utc)
            symbol = self._extract_symbol(title + " " + description)
            category = self._normalise_category(title.lower())
            events.append(CalendarEvent(
                event_id=f"cc_{i}_{hash(title) & 0xFFFF:04x}",
                event_name=title,
                symbol=symbol,
                date=date,
                category=category,
                impact_estimate=CATEGORY_IMPACT.get(category, "LOW"),
                source="cryptocal",
                description=description,
                confidence=0.4,
            ))
        return events

    @staticmethod
    def _rss_field(xml: str, tag: str) -> str:
        import re
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_symbol(text: str) -> str:
        import re
        # Try to find a ticker-like token ($BTC, $ETH, or uppercase 2-5 chars)
        m = re.search(r"\$([A-Z]{2,5})", text)
        if m:
            return m.group(1)
        m = re.search(r"\b([A-Z]{2,5})\b", text)
        return m.group(1) if m else "UNKNOWN"

    @staticmethod
    def _normalise_category(raw: str) -> str:
        for key in CATEGORY_IMPACT:
            if key in raw:
                return key
        return "other"

    @staticmethod
    def _impact_from_credibility(credibility: float, category: str) -> str:
        base = CATEGORY_IMPACT.get(category, "LOW")
        if credibility > 0.7:
            return base
        if credibility > 0.4:
            return "MEDIUM" if base == "HIGH" else base
        return "LOW"

    @staticmethod
    def _today_str() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _deduplicate(events: List[CalendarEvent]) -> List[CalendarEvent]:
        seen: Dict[str, CalendarEvent] = {}
        for ev in events:
            key = f"{ev.symbol}_{ev.date.date()}_{ev.event_name[:30].lower().replace(' ', '_')}"
            if key not in seen or ev.confidence > seen[key].confidence:
                seen[key] = ev
        return sorted(seen.values(), key=lambda e: e.date)

    @staticmethod
    def _synthetic_coinmarketcal_events() -> List[CalendarEvent]:
        """Return plausible synthetic events when API is unavailable."""
        now = datetime.now(timezone.utc)
        return [
            CalendarEvent(
                event_id="synthetic_001",
                event_name="ARB Token Unlock (Synthetic)",
                symbol="ARB",
                date=now.replace(day=min(now.day + 3, 28)),
                category="unlock",
                impact_estimate="HIGH",
                source="synthetic",
                confidence=0.3,
            ),
        ]
