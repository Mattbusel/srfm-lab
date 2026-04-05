"""
Exchange listing and delisting monitor.

Scrapes Binance announcement RSS and Coinbase blog for listing / delist events.
Applies historical average impact estimates:
  - Binance new listing  -> +30% avg in 24h
  - Coinbase new listing -> +15% avg in 24h
  - Any delist           -> -40% avg in 24h

Results are cached 4h.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BINANCE_RSS = "https://www.binance.com/en/support/announcement/rss/c-48"
COINBASE_RSS = "https://www.coinbase.com/blog/rss"
CACHE_TTL = 4 * 3600

LISTING_KEYWORDS = [
    "will list", "has listed", "listing of", "new listing",
    "adds", "supports", "trading pair",
]
DELIST_KEYWORDS = [
    "delist", "will remove", "removal of", "discontinue trading",
    "suspend trading",
]

IMPACT_TABLE: Dict[str, Dict[str, Any]] = {
    "binance": {
        "listing": {"price_change_pct": 30.0,  "direction": "BULLISH", "impact": "HIGH"},
        "delist":  {"price_change_pct": -40.0, "direction": "BEARISH", "impact": "HIGH"},
    },
    "coinbase": {
        "listing": {"price_change_pct": 15.0,  "direction": "BULLISH", "impact": "HIGH"},
        "delist":  {"price_change_pct": -40.0, "direction": "BEARISH", "impact": "HIGH"},
    },
}


@dataclass
class ListingEvent:
    """An exchange listing or delisting event."""

    event_id: str
    exchange: str           # binance | coinbase
    event_subtype: str      # listing | delist
    symbol: str
    detected_at: datetime
    title: str
    url: str
    expected_price_change_pct: float
    direction: str          # BULLISH | BEARISH
    impact: str             # HIGH | MEDIUM | LOW
    source_url: str
    raw_text: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["detected_at"] = self.detected_at.isoformat()
        return d


class ExchangeListingMonitor:
    """
    Monitors Binance and Coinbase announcement feeds for listing events.

    Usage::

        monitor = ExchangeListingMonitor()
        events  = monitor.fetch_all()
        for ev in events:
            print(ev.symbol, ev.direction, ev.expected_price_change_pct)
    """

    def __init__(self, cache_ttl: int = CACHE_TTL) -> None:
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._timeout = 10

    # ── public API ──────────────────────────────────────────────────────────────

    def fetch_all(self) -> List[ListingEvent]:
        """Return all listing/delist events detected from all sources."""
        events: List[ListingEvent] = []
        events.extend(self._fetch_binance())
        events.extend(self._fetch_coinbase())
        return events

    def get_recent_listings(self, hours: int = 48) -> List[ListingEvent]:
        """Return listing events detected in the last *hours* hours."""
        cutoff = time.time() - hours * 3600
        return [
            e for e in self.fetch_all()
            if e.detected_at.timestamp() >= cutoff
        ]

    def is_listing_active(self, symbol: str, hours: int = 24) -> Optional[ListingEvent]:
        """
        Return the most recent listing event for *symbol* within *hours* if one
        exists, otherwise None.  Used to decide whether a listing premium is
        still in play.
        """
        sym = symbol.upper()
        recent = [
            e for e in self.get_recent_listings(hours)
            if e.symbol.upper() == sym and e.event_subtype == "listing"
        ]
        return max(recent, key=lambda e: e.detected_at) if recent else None

    # ── Binance ─────────────────────────────────────────────────────────────────

    def _fetch_binance(self) -> List[ListingEvent]:
        key = "binance"
        cached = self._get_cache(key)
        if cached is not None:
            return cached

        events: List[ListingEvent] = []
        try:
            xml = self._http_get(BINANCE_RSS)
            items = self._parse_rss_items(xml)
            for title, pub_date, link, description in items:
                subtype = self._detect_subtype(title + " " + description)
                if not subtype:
                    continue
                symbol = self._extract_symbol(title + " " + description)
                meta = IMPACT_TABLE["binance"][subtype]
                events.append(ListingEvent(
                    event_id=f"bnb_{hash(title) & 0xFFFFFFFF:08x}",
                    exchange="binance",
                    event_subtype=subtype,
                    symbol=symbol,
                    detected_at=self._parse_rss_date(pub_date),
                    title=title,
                    url=link,
                    expected_price_change_pct=meta["price_change_pct"],
                    direction=meta["direction"],
                    impact=meta["impact"],
                    source_url=BINANCE_RSS,
                    raw_text=description[:500],
                ))
        except Exception as exc:
            logger.warning("Binance RSS fetch failed: %s", exc)

        self._set_cache(key, events)
        return events

    # ── Coinbase ─────────────────────────────────────────────────────────────────

    def _fetch_coinbase(self) -> List[ListingEvent]:
        key = "coinbase"
        cached = self._get_cache(key)
        if cached is not None:
            return cached

        events: List[ListingEvent] = []
        try:
            xml = self._http_get(COINBASE_RSS)
            items = self._parse_rss_items(xml)
            for title, pub_date, link, description in items:
                subtype = self._detect_subtype(title + " " + description)
                if not subtype:
                    continue
                symbol = self._extract_symbol(title + " " + description)
                meta = IMPACT_TABLE["coinbase"][subtype]
                events.append(ListingEvent(
                    event_id=f"cb_{hash(title) & 0xFFFFFFFF:08x}",
                    exchange="coinbase",
                    event_subtype=subtype,
                    symbol=symbol,
                    detected_at=self._parse_rss_date(pub_date),
                    title=title,
                    url=link,
                    expected_price_change_pct=meta["price_change_pct"],
                    direction=meta["direction"],
                    impact=meta["impact"],
                    source_url=COINBASE_RSS,
                    raw_text=description[:500],
                ))
        except Exception as exc:
            logger.warning("Coinbase RSS fetch failed: %s", exc)

        self._set_cache(key, events)
        return events

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _http_get(self, url: str) -> str:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "SRFM-IdeaEngine/1.0")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return resp.read().decode(errors="replace")

    @staticmethod
    def _parse_rss_items(xml: str) -> List[Tuple[str, str, str, str]]:
        items = re.findall(r"<item>(.*?)</item>", xml, re.DOTALL)
        result = []
        for raw in items[:50]:
            title = ExchangeListingMonitor._rss_tag(raw, "title")
            pub_date = ExchangeListingMonitor._rss_tag(raw, "pubDate")
            link = ExchangeListingMonitor._rss_tag(raw, "link")
            description = re.sub(r"<[^>]+>", " ",
                                  ExchangeListingMonitor._rss_tag(raw, "description"))
            result.append((title, pub_date, link, description))
        return result

    @staticmethod
    def _rss_tag(xml: str, tag: str) -> str:
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _detect_subtype(text: str) -> Optional[str]:
        tl = text.lower()
        if any(kw in tl for kw in DELIST_KEYWORDS):
            return "delist"
        if any(kw in tl for kw in LISTING_KEYWORDS):
            return "listing"
        return None

    @staticmethod
    def _extract_symbol(text: str) -> str:
        # Prefer $TICKER pattern then ALL_CAPS 2-6 chars
        m = re.search(r"\$([A-Z]{2,6})", text)
        if m:
            return m.group(1)
        m = re.search(r"\b([A-Z]{2,6})\b", text)
        return m.group(1) if m else "UNKNOWN"

    @staticmethod
    def _parse_rss_date(date_str: str) -> datetime:
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT"):
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return datetime.now(timezone.utc)

    def _get_cache(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry and (time.time() - entry[0]) < self._cache_ttl:
            return entry[1]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = (time.time(), value)
