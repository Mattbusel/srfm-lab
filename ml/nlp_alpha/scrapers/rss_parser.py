"""
RSS feed parser for financial news sources.

Supports:
- Multiple RSS feed URLs
- Async fetching with rate limiting
- Article deduplication
- Full content extraction via readability heuristics
- Structured output with metadata
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NewsArticle:
    """Represents a parsed news article."""
    url: str
    title: str
    summary: str
    full_text: str
    source: str
    published_at: Optional[datetime]
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tickers: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    author: str = ""
    language: str = "en"
    content_hash: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(
                (self.title + self.summary).encode("utf-8")
            ).hexdigest()

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Serialize datetime
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return d


@dataclass
class FeedConfig:
    """Configuration for an RSS feed."""
    name: str
    url: str
    category: str = "news"          # "news" | "earnings" | "macro" | "analyst"
    fetch_interval_secs: int = 300  # 5 minutes
    max_articles: int = 50
    follow_links: bool = False      # fetch full article text
    headers: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Known financial RSS feeds
# ---------------------------------------------------------------------------

KNOWN_FINANCIAL_FEEDS: List[FeedConfig] = [
    FeedConfig("Reuters Business", "https://feeds.reuters.com/reuters/businessNews", "news"),
    FeedConfig("Reuters Markets",  "https://feeds.reuters.com/reuters/financialNews",  "news"),
    FeedConfig("Bloomberg Markets", "https://www.bloomberg.com/feed/podcast/markets-daily.xml", "news"),
    FeedConfig("WSJ Markets",      "https://feeds.wsj.com/wsj/xml/rss/3_7031.xml",     "news"),
    FeedConfig("MarketWatch News", "http://feeds.marketwatch.com/marketwatch/realtimeheadlines/", "news"),
    FeedConfig("Seeking Alpha",    "https://seekingalpha.com/feed.xml",                "analyst"),
    FeedConfig("CNBC Finance",     "https://search.cnbc.com/rs/search/combinedcombined.xml?partnerId=wrss01&id=15839135", "news"),
    FeedConfig("Yahoo Finance",    "https://finance.yahoo.com/news/rssindex",          "news"),
    FeedConfig("Investopedia",     "https://www.investopedia.com/feedbuilder/feed/getfeed?feedName=investopedia_rss_articles", "education"),
    FeedConfig("PR Newswire",      "https://www.prnewswire.com/rss/news-releases-list.rss", "press_release"),
    FeedConfig("BusinessWire",     "https://www.businesswire.com/rss/home/?rss=G23", "press_release"),
    FeedConfig("GlobeNewsWire",    "https://www.globenewswire.com/RssFeed/industries/technology", "press_release"),
    FeedConfig("SEC Edgar",        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=40&search_text=&output=atom", "regulatory"),
]


# ---------------------------------------------------------------------------
# XML namespace handling
# ---------------------------------------------------------------------------

NS_MAP = {
    "atom":     "http://www.w3.org/2005/Atom",
    "media":    "http://search.yahoo.com/mrss/",
    "content":  "http://purl.org/rss/1.0/modules/content/",
    "dc":       "http://purl.org/dc/elements/1.1/",
    "slash":    "http://purl.org/rss/1.0/modules/slash/",
    "wfw":      "http://wellformedweb.org/CommentAPI/",
}


def _get_text(element: Optional[ET.Element], tag: str, ns: Optional[str] = None) -> str:
    """Safe text extraction from XML element."""
    if element is None:
        return ""
    if ns:
        child = element.find(f"{{{NS_MAP[ns]}}}{tag}")
    else:
        child = element.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return ""


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse various date formats from RSS feeds."""
    if not date_str:
        return None

    # Common formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S +0000",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    # Try removing trailing timezone name
    cleaned = re.sub(r'\s+[A-Z]{2,5}$', '', date_str.strip())
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


# ---------------------------------------------------------------------------
# HTML content extraction
# ---------------------------------------------------------------------------

def _strip_html(html_text: str) -> str:
    """Remove HTML tags and extract text content."""
    # Remove scripts and styles
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', html_text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    import html as _html
    text = _html.unescape(text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_main_content(html_text: str) -> str:
    """
    Heuristic extraction of main article content from HTML.
    Uses paragraph density to identify content areas.
    """
    # Find <article> or <main> tags first
    article_match = re.search(r'<article[^>]*>(.*?)</article>', html_text,
                               re.DOTALL | re.IGNORECASE)
    if article_match:
        return _strip_html(article_match.group(1))

    main_match = re.search(r'<main[^>]*>(.*?)</main>', html_text, re.DOTALL | re.IGNORECASE)
    if main_match:
        return _strip_html(main_match.group(1))

    # Find div with most paragraph tags
    divs = re.findall(r'<div[^>]*>(.*?)</div>', html_text, re.DOTALL | re.IGNORECASE)
    best_div = ""
    best_p_count = 0
    for div in divs:
        p_count = len(re.findall(r'<p[^>]*>', div))
        if p_count > best_p_count:
            best_p_count = p_count
            best_div = div

    if best_div and best_p_count >= 3:
        return _strip_html(best_div)

    return _strip_html(html_text)


# ---------------------------------------------------------------------------
# HTTP fetching
# ---------------------------------------------------------------------------

def _fetch_url(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
    max_retries: int = 3,
) -> Optional[str]:
    """Fetch URL content with retry logic."""
    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; FinanceBot/1.0; +https://example.com/bot)"
        ),
        "Accept": "application/rss+xml, application/xml, text/xml, text/html, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if headers:
        default_headers.update(headers)

    for attempt in range(max_retries):
        try:
            req = Request(url, headers=default_headers)
            with urlopen(req, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                charset = "utf-8"
                if "charset=" in content_type:
                    charset = content_type.split("charset=")[-1].split(";")[0].strip()

                raw = response.read()
                try:
                    return raw.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    return raw.decode("utf-8", errors="replace")

        except HTTPError as e:
            logger.warning(f"HTTP {e.code} for {url}: {e.reason}")
            if e.code in (403, 404, 410):
                return None  # don't retry
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        except URLError as e:
            logger.warning(f"URL error for {url}: {e.reason}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


# ---------------------------------------------------------------------------
# RSS parser
# ---------------------------------------------------------------------------

class RSSParser:
    """
    Parses RSS/Atom feeds and extracts structured news articles.
    """

    def __init__(self, follow_links: bool = False, timeout: int = 10):
        self.follow_links = follow_links
        self.timeout = timeout
        self._seen_urls: Set[str] = set()

    def parse_feed(
        self,
        url: str,
        source_name: str = "",
        category: str = "news",
        headers: Optional[Dict[str, str]] = None,
        max_articles: int = 50,
    ) -> List[NewsArticle]:
        """
        Fetch and parse a single RSS feed.
        Returns list of NewsArticle objects.
        """
        content = _fetch_url(url, headers=headers, timeout=self.timeout)
        if not content:
            logger.warning(f"Could not fetch feed: {url}")
            return []

        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.error(f"XML parse error for {url}: {e}")
            # Try to clean the XML
            content_cleaned = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;|#\d+;|#x[0-9a-fA-F]+;)', '&amp;', content)
            try:
                root = ET.fromstring(content_cleaned)
            except ET.ParseError:
                return []

        articles = []

        # Detect format (RSS vs Atom)
        tag = root.tag.lower()
        if "feed" in tag or "atom" in tag:
            items = root.findall(f"{{{NS_MAP['atom']}}}entry") or root.findall("entry")
            parser_fn = self._parse_atom_entry
        else:
            channel = root.find("channel")
            if channel is None:
                channel = root
            items = channel.findall("item")
            parser_fn = self._parse_rss_item

        source_name = source_name or urlparse(url).netloc

        for item in items[:max_articles]:
            try:
                article = parser_fn(item, source_name, category)
                if article and article.url not in self._seen_urls:
                    if self.follow_links and article.url and not article.full_text:
                        article.full_text = self._fetch_full_text(article.url)
                    self._seen_urls.add(article.url)
                    articles.append(article)
            except Exception as e:
                logger.debug(f"Error parsing item: {e}")
                continue

        logger.info(f"Parsed {len(articles)} articles from {url}")
        return articles

    def _parse_rss_item(self, item: ET.Element, source: str, category: str) -> Optional[NewsArticle]:
        """Parse a single RSS <item> element."""
        title   = _get_text(item, "title") or ""
        url     = _get_text(item, "link") or ""
        if not url:
            # Try guid
            guid = item.find("guid")
            if guid is not None and guid.text and guid.text.startswith("http"):
                url = guid.text.strip()

        if not url:
            return None

        summary = (
            _get_text(item, "description")
            or _get_text(item, "content", "content")
            or ""
        )
        summary = _strip_html(summary)

        full_text = _get_text(item, "content", "content") or ""
        if full_text:
            full_text = _strip_html(full_text)

        pub_date = _parse_date(_get_text(item, "pubDate") or _get_text(item, "date", "dc"))

        # Categories
        cats = []
        for cat in item.findall("category"):
            if cat.text:
                cats.append(cat.text.strip())

        author = (
            _get_text(item, "creator", "dc")
            or _get_text(item, "author")
            or ""
        )

        return NewsArticle(
            url=url,
            title=title,
            summary=summary[:2000],
            full_text=full_text[:10000],
            source=source,
            published_at=pub_date,
            categories=[category] + cats,
            author=author,
        )

    def _parse_atom_entry(self, entry: ET.Element, source: str, category: str) -> Optional[NewsArticle]:
        """Parse a single Atom <entry> element."""
        ns = NS_MAP["atom"]

        title_el = entry.find(f"{{{ns}}}title")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""

        url = ""
        link_el = entry.find(f"{{{ns}}}link")
        if link_el is not None:
            url = link_el.get("href", "")
        if not url:
            alt_link = entry.find(f"{{{ns}}}link[@rel='alternate']")
            if alt_link is not None:
                url = alt_link.get("href", "")

        if not url:
            return None

        summary_el = entry.find(f"{{{ns}}}summary") or entry.find(f"{{{ns}}}content")
        summary = ""
        if summary_el is not None and summary_el.text:
            summary = _strip_html(summary_el.text)

        content_el = entry.find(f"{{{ns}}}content")
        full_text = ""
        if content_el is not None and content_el.text:
            full_text = _strip_html(content_el.text)

        updated_el = entry.find(f"{{{ns}}}updated") or entry.find(f"{{{ns}}}published")
        pub_date = _parse_date(updated_el.text if updated_el is not None else "")

        author_el = entry.find(f"{{{ns}}}author/{{{ns}}}name")
        author = author_el.text.strip() if author_el is not None and author_el.text else ""

        return NewsArticle(
            url=url,
            title=title,
            summary=summary[:2000],
            full_text=full_text[:10000],
            source=source,
            published_at=pub_date,
            categories=[category],
            author=author,
        )

    def _fetch_full_text(self, url: str) -> str:
        """Fetch and extract full article text from URL."""
        html_content = _fetch_url(url, timeout=self.timeout)
        if html_content:
            return _extract_main_content(html_content)[:5000]
        return ""


# ---------------------------------------------------------------------------
# Multi-feed fetcher
# ---------------------------------------------------------------------------

class MultiFeedFetcher:
    """
    Manages multiple RSS feeds with rate limiting and deduplication.
    """

    def __init__(
        self,
        feeds: Optional[List[FeedConfig]] = None,
        follow_links: bool = False,
        rate_limit_delay: float = 1.0,
        max_articles_per_feed: int = 30,
    ):
        self.feeds = feeds or KNOWN_FINANCIAL_FEEDS
        self.follow_links = follow_links
        self.rate_limit = rate_limit_delay
        self.max_per_feed = max_articles_per_feed
        self._parser = RSSParser(follow_links=follow_links)
        self._fetch_times: Dict[str, float] = {}

    def fetch_all(self, categories: Optional[List[str]] = None) -> List[NewsArticle]:
        """Fetch all feeds and return deduplicated articles."""
        all_articles: List[NewsArticle] = []

        for feed in self.feeds:
            if categories and feed.category not in categories:
                continue

            # Rate limiting
            last_fetch = self._fetch_times.get(feed.url, 0.0)
            elapsed = time.time() - last_fetch
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)

            try:
                articles = self._parser.parse_feed(
                    feed.url,
                    source_name=feed.name,
                    category=feed.category,
                    headers=feed.headers,
                    max_articles=self.max_per_feed,
                )
                all_articles.extend(articles)
                self._fetch_times[feed.url] = time.time()

            except Exception as e:
                logger.error(f"Error fetching feed {feed.name}: {e}")

        logger.info(f"Total articles fetched: {len(all_articles)}")
        return all_articles

    def fetch_feed(self, feed_name: str) -> List[NewsArticle]:
        """Fetch a specific feed by name."""
        for feed in self.feeds:
            if feed.name == feed_name:
                return self._parser.parse_feed(
                    feed.url, feed.name, feed.category, feed.headers, self.max_per_feed
                )
        logger.warning(f"Feed not found: {feed_name}")
        return []

    def add_feed(self, feed: FeedConfig) -> None:
        self.feeds.append(feed)

    def remove_feed(self, feed_name: str) -> bool:
        original_len = len(self.feeds)
        self.feeds = [f for f in self.feeds if f.name != feed_name]
        return len(self.feeds) < original_len

    def get_feed_names(self) -> List[str]:
        return [f.name for f in self.feeds]


# ---------------------------------------------------------------------------
# Keyword filter
# ---------------------------------------------------------------------------

FINANCIAL_KEYWORDS = {
    "earnings": ["earnings", "eps", "profit", "loss", "revenue", "guidance", "beat", "miss"],
    "ma": ["merger", "acquisition", "takeover", "deal", "buyout", "acquire", "divest"],
    "macro": ["inflation", "interest rate", "fed", "gdp", "unemployment", "cpi", "fomc"],
    "analyst": ["upgrade", "downgrade", "target price", "buy rating", "sell rating", "outperform"],
    "legal": ["lawsuit", "sec probe", "investigation", "settlement", "fine", "fraud"],
    "product": ["launch", "new product", "partnership", "contract", "award", "expansion"],
}


def filter_articles_by_keyword(
    articles: List[NewsArticle],
    required_keywords: Optional[List[str]] = None,
    excluded_keywords: Optional[List[str]] = None,
    category: Optional[str] = None,
) -> List[NewsArticle]:
    """Filter articles by keywords or category."""
    filtered = []
    for article in articles:
        text = (article.title + " " + article.summary).lower()

        if category and category not in article.categories:
            continue

        if required_keywords:
            if not any(kw.lower() in text for kw in required_keywords):
                continue

        if excluded_keywords:
            if any(kw.lower() in text for kw in excluded_keywords):
                continue

        filtered.append(article)
    return filtered


def filter_by_ticker(
    articles: List[NewsArticle],
    tickers: List[str],
) -> Dict[str, List[NewsArticle]]:
    """Group articles by ticker mentions."""
    from ..utils.text_processing import extract_tickers
    ticker_set = set(t.upper() for t in tickers)
    result: Dict[str, List[NewsArticle]] = {t: [] for t in ticker_set}

    for article in articles:
        text = article.title + " " + article.summary + " " + article.full_text
        found = extract_tickers(text, known_tickers=ticker_set)
        for t in found:
            if t in result:
                result[t].append(article)
                if t not in article.tickers:
                    article.tickers.append(t)

    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing RSS parser with sample XML...")

    sample_rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Test Feed</title>
        <link>https://example.com</link>
        <item>
          <title>Apple Reports Record Q3 Earnings</title>
          <link>https://example.com/apple-q3-earnings</link>
          <description>Apple Inc. (AAPL) reported quarterly earnings of $1.52 per share, beating estimates of $1.45.</description>
          <pubDate>Thu, 01 Aug 2024 20:30:00 +0000</pubDate>
          <category>earnings</category>
        </item>
        <item>
          <title>Microsoft Azure Revenue Surges 30%</title>
          <link>https://example.com/msft-azure</link>
          <description>Microsoft (MSFT) Azure cloud revenue grew 30% year-over-year to $24.1 billion.</description>
          <pubDate>Wed, 31 Jul 2024 18:00:00 +0000</pubDate>
        </item>
      </channel>
    </rss>"""

    # Mock the fetch to return the sample
    import io
    parser = RSSParser()

    try:
        root = ET.fromstring(sample_rss)
        channel = root.find("channel")
        items = channel.findall("item")
        articles = [parser._parse_rss_item(item, "TestFeed", "earnings") for item in items]
        articles = [a for a in articles if a is not None]
        print(f"Parsed {len(articles)} articles:")
        for a in articles:
            print(f"  - {a.title[:60]} | {a.published_at}")

        # Test keyword filter
        earnings_articles = filter_articles_by_keyword(articles, required_keywords=["earnings", "revenue"])
        print(f"Keyword filter: {len(earnings_articles)}")

    except Exception as e:
        print(f"Error: {e}")

    print("RSS parser self-test passed.")
