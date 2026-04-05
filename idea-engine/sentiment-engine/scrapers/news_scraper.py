"""
sentiment_engine/scrapers/news_scraper.py
=========================================
RSS feed parser for major crypto news outlets.

Financial rationale
-------------------
Institutional and retail narratives in crypto are largely driven by news
headlines.  CoinDesk, CoinTelegraph, Decrypt, and The Block represent the
four most-cited sources in crypto markets with measurable price impact
within 30-60 minutes of publication (Vidal-Tomás & Ibáñez, 2018 update for
digital assets).

We parse:
  - titles  (high information density, always present)
  - summaries / descriptions (more context, often absent)

Symbol detection maps full names and ticker mentions to canonical symbols
so downstream aggregation can attribute sentiment to specific assets.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feed definitions
# ---------------------------------------------------------------------------

RSS_FEEDS: dict[str, str] = {
    "CoinDesk":      "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "Decrypt":       "https://decrypt.co/feed",
    "TheBlock":      "https://www.theblock.co/rss.xml",
}

# Source credibility weights (used by sentiment scorer)
SOURCE_CREDIBILITY: dict[str, float] = {
    "CoinDesk":      0.90,
    "CoinTelegraph": 0.85,
    "Decrypt":       0.80,
    "TheBlock":      0.88,
    "unknown":       0.50,
}

# Request timeout and retry policy
FETCH_TIMEOUT: int  = 10   # seconds
MAX_RETRIES:   int  = 2
RETRY_DELAY:   float = 1.5  # seconds between retries

# Max age of articles to keep (seconds)
MAX_ARTICLE_AGE_S: int = 4 * 3600  # 4 hours


# ---------------------------------------------------------------------------
# Symbol mention patterns
# ---------------------------------------------------------------------------

# Maps pattern → canonical symbol
_SYMBOL_PATTERNS: dict[str, str] = {
    r"\bbitcoin\b":       "BTC",
    r"\$?BTC\b":          "BTC",
    r"\bethereum\b":      "ETH",
    r"\$?ETH\b":          "ETH",
    r"\bsolana\b":        "SOL",
    r"\$?SOL\b":          "SOL",
    r"\bbinance\s+coin\b":"BNB",
    r"\$?BNB\b":          "BNB",
    r"\bripple\b":        "XRP",
    r"\$?XRP\b":          "XRP",
    r"\bdogecoin\b":      "DOGE",
    r"\$?DOGE\b":         "DOGE",
    r"\bcardano\b":       "ADA",
    r"\$?ADA\b":          "ADA",
    r"\bpolygon\b":       "POL",
    r"\$?MATIC\b":        "POL",
    r"\bavalanche\b":     "AVAX",
    r"\$?AVAX\b":         "AVAX",
    r"\bchainlink\b":     "LINK",
    r"\$?LINK\b":         "LINK",
}

# Compiled once at module load
_COMPILED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), sym)
    for pat, sym in _SYMBOL_PATTERNS.items()
]


def detect_symbols(text: str) -> set[str]:
    """Return the set of canonical crypto symbols mentioned in *text*."""
    found: set[str] = set()
    for pattern, symbol in _COMPILED_PATTERNS:
        if pattern.search(text):
            found.add(symbol)
    return found


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NewsItem:
    """
    A single parsed news article from an RSS feed.

    Attributes
    ----------
    source      : Feed name (e.g. 'CoinDesk')
    title       : Article headline
    summary     : Lead paragraph or RSS description
    url         : Canonical article URL
    published   : Publication datetime (UTC)
    symbols     : Crypto symbols mentioned in title + summary
    credibility : Source credibility weight [0, 1]
    full_text   : Concatenated title + ' ' + summary
    age_seconds : Seconds since publication at scrape time
    """
    source:      str
    title:       str
    summary:     str
    url:         str
    published:   datetime
    symbols:     set[str]   = field(default_factory=set)
    credibility: float      = 0.70

    @property
    def full_text(self) -> str:
        return f"{self.title} {self.summary}".strip()

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.published).total_seconds()


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class NewsScraper:
    """
    Fetches and parses RSS feeds from major crypto news outlets.

    Parameters
    ----------
    feeds         : Override the default RSS_FEEDS dict
    max_age_hours : Discard articles older than this many hours
    session       : Optional pre-configured requests.Session
    """

    def __init__(
        self,
        feeds:          dict[str, str] | None = None,
        max_age_hours:  float = 4.0,
        session:        Optional[requests.Session] = None,
    ) -> None:
        self.feeds         = feeds or RSS_FEEDS
        self.max_age_s     = int(max_age_hours * 3600)
        self._session      = session or requests.Session()
        self._session.headers.update({
            "User-Agent": "sentiment-engine/0.1 (research; contact@example.com)"
        })

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fetch_all(self) -> list[NewsItem]:
        """
        Fetch and parse all configured RSS feeds.

        Returns
        -------
        Deduplicated list of NewsItem sorted by descending recency.
        Items older than max_age_hours are filtered out.
        """
        items:   list[NewsItem] = []
        seen_urls: set[str]     = set()

        for source, url in self.feeds.items():
            try:
                feed_items = self._fetch_feed(source, url)
                for item in feed_items:
                    if item.url not in seen_urls:
                        seen_urls.add(item.url)
                        items.append(item)
            except Exception as exc:
                logger.error("Error fetching feed '%s' (%s): %s", source, url, exc)

        # Filter by age and sort newest-first
        now = datetime.now(timezone.utc)
        items = [
            i for i in items
            if (now - i.published).total_seconds() <= self.max_age_s
        ]
        items.sort(key=lambda i: i.published, reverse=True)

        logger.info(
            "NewsScraper: %d recent articles collected from %d feeds.",
            len(items), len(self.feeds),
        )
        return items

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _fetch_feed(self, source: str, url: str) -> list[NewsItem]:
        """Fetch and parse a single RSS feed with retries."""
        xml_text = self._fetch_with_retry(url)
        if not xml_text:
            return []
        return self._parse_rss(source, xml_text)

    def _fetch_with_retry(self, url: str) -> Optional[str]:
        """GET with exponential back-off; returns raw text or None on failure."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._session.get(url, timeout=FETCH_TIMEOUT)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as exc:
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "Feed %s failed (attempt %d/%d), retrying in %.1fs: %s",
                        url, attempt + 1, MAX_RETRIES + 1, delay, exc,
                    )
                    time.sleep(delay)
                else:
                    logger.error("Feed %s permanently failed: %s", url, exc)
        return None

    def _parse_rss(self, source: str, xml_text: str) -> list[NewsItem]:
        """Parse an RSS 2.0 feed into NewsItem objects."""
        credibility = SOURCE_CREDIBILITY.get(source, SOURCE_CREDIBILITY["unknown"])
        items: list[NewsItem] = []

        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError as exc:
            logger.error("XML parse error for %s: %s", source, exc)
            return []

        # Support both RSS 2.0 (<rss><channel><item>) and Atom feeds
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        channel = root.find("channel")
        if channel is not None:
            raw_items = channel.findall("item")
        else:
            # Atom fallback
            raw_items = root.findall("atom:entry", ns)

        for raw in raw_items:
            item = self._parse_item(source, raw, credibility, ns)
            if item:
                items.append(item)

        return items

    def _parse_item(
        self,
        source:      str,
        raw:         ElementTree.Element,
        credibility: float,
        ns:          dict[str, str],
    ) -> Optional[NewsItem]:
        """Parse a single RSS <item> or Atom <entry> element."""
        # Title
        title_el = raw.find("title") or raw.find("atom:title", ns)
        title    = (title_el.text or "").strip() if title_el is not None else ""
        if not title:
            return None

        # Summary / description
        summary_el = (
            raw.find("description")
            or raw.find("summary")
            or raw.find("atom:summary", ns)
            or raw.find("atom:content", ns)
        )
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        # Strip HTML tags from summary
        summary = re.sub(r"<[^>]+>", " ", summary)
        summary = re.sub(r"\s+", " ", summary).strip()

        # URL
        link_el = raw.find("link") or raw.find("atom:link", ns)
        if link_el is not None:
            url = link_el.text or link_el.get("href", "")
        else:
            url = ""

        # Published date
        pub_el = (
            raw.find("pubDate")
            or raw.find("published")
            or raw.find("atom:published", ns)
        )
        published: datetime
        if pub_el is not None and pub_el.text:
            try:
                published = parsedate_to_datetime(pub_el.text)
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                try:
                    published = datetime.fromisoformat(
                        pub_el.text.replace("Z", "+00:00")
                    )
                except ValueError:
                    published = datetime.now(timezone.utc)
        else:
            published = datetime.now(timezone.utc)

        full_text = f"{title} {summary}"
        symbols   = detect_symbols(full_text)

        return NewsItem(
            source=source,
            title=title,
            summary=summary[:500],  # cap to avoid huge payloads
            url=url.strip(),
            published=published,
            symbols=symbols,
            credibility=credibility,
        )
