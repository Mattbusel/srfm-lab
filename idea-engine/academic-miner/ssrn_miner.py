"""
ssrn_miner.py — SSRN Abstract Page Scraper
===========================================
Scrapes public SSRN abstract pages for papers relevant to the BH trading
strategy.  SSRN does not provide a public API, so we scrape the HTML of
known abstract pages and build a relevance score using keyword matching.

Design constraints
------------------
- Respects robots.txt (checked once per session, cached).
- 5-second minimum delay between requests.
- Uses only stdlib: urllib, html.parser, re, json, sqlite3.
- No requests, BeautifulSoup, or Selenium.

Typical usage
-------------
    miner = SSRNMiner(db_path="idea_engine.db")
    papers = miner.search_keyword("momentum trading crypto")
    miner.store_papers(papers)
"""

from __future__ import annotations

import html
import html.parser
import json
import logging
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SSRN_BASE: str = "https://papers.ssrn.com"
SSRN_SEARCH_URL: str = "https://papers.ssrn.com/sol3/results.cfm"
SSRN_ABSTRACT_BASE: str = "https://papers.ssrn.com/sol3/papers.cfm"

REQUEST_DELAY_SECONDS: float = 5.0
MAX_PAPERS_PER_RUN:    int   = 10
REQUEST_TIMEOUT:       int   = 30

# Same keyword list as ArXivMiner for consistency
STRATEGY_KEYWORDS: List[str] = [
    "momentum trading",
    "mean reversion crypto",
    "volatility forecasting GARCH",
    "regime detection",
    "causal discovery financial",
    "genetic algorithm trading",
    "Ornstein-Uhlenbeck",
    "Kelly criterion",
    "drawdown control",
    "market microstructure",
]

HIGH_VALUE_TERMS: List[str] = [
    "momentum", "mean reversion", "volatility", "regime", "drawdown",
    "kelly", "ornstein-uhlenbeck", "microstructure", "causal", "genetic",
    "reinforcement learning", "entropy", "fractal", "hurst", "kalman",
    "phase transition", "turbulence", "liquidity", "spread",
    "order flow", "market impact", "alpha decay", "signal decay",
]

MEDIUM_VALUE_TERMS: List[str] = [
    "cryptocurrency", "bitcoin", "ethereum", "crypto", "digital asset",
    "high frequency", "intraday", "tick data", "limit order book",
    "portfolio optimization", "risk management", "position sizing",
    "backtest", "out-of-sample", "sharpe", "sortino",
    "maximum drawdown", "var", "cvar",
    "neural network", "lstm", "transformer", "gradient boosting",
]

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SSRNPaper:
    """Lightweight container for a single SSRN paper."""

    paper_id: str                    # SSRN abstract_id
    title:    str
    authors:  List[str]              = field(default_factory=list)
    abstract: str                    = ""
    keywords: List[str]              = field(default_factory=list)
    url:      str                    = ""
    relevance_score: float           = 0.0
    db_id:   Optional[int]           = None

    def to_dict(self) -> dict:
        return {
            "paper_id":       self.paper_id,
            "title":          self.title,
            "authors":        self.authors,
            "abstract":       self.abstract,
            "keywords":       self.keywords,
            "url":            self.url,
            "relevance_score": self.relevance_score,
        }

    def __repr__(self) -> str:
        return (f"SSRNPaper(id={self.paper_id!r}, "
                f"score={self.relevance_score:.3f}, title={self.title[:60]!r})")


# ---------------------------------------------------------------------------
# Minimal HTML parser — extracts SSRN abstract page fields
# ---------------------------------------------------------------------------

class _SSRNPageParser(html.parser.HTMLParser):
    """
    Extracts title, authors, abstract, and keywords from an SSRN abstract
    page using Python's built-in html.parser.

    State machine: tracks which tag is active and accumulates text.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title:    str        = ""
        self.abstract: str        = ""
        self.authors:  List[str]  = []
        self.keywords: List[str]  = []

        self._in_title:    bool = False
        self._in_abstract: bool = False
        self._in_author:   bool = False
        self._in_keyword:  bool = False
        self._buf:         str  = ""
        self._depth:       int  = 0   # nesting depth tracker

    # -- HTMLParser callbacks ------------------------------------------------

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_dict: Dict[str, str] = {k: (v or "") for k, v in attrs}
        cls  = attr_dict.get("class", "")
        id_  = attr_dict.get("id", "")

        # Title: <h1 class="title ..."> or <title>
        if tag in ("h1",) and "title" in cls.lower():
            self._in_title = True
            self._buf = ""
        elif tag == "title" and not self.title:
            self._in_title = True
            self._buf = ""

        # Abstract: <div class="abstract-text"> or id="abstract"
        elif tag == "div" and ("abstract" in cls.lower() or id_ == "abstract"):
            self._in_abstract = True
            self._buf = ""

        # Authors: <div class="authors"> span.author-name
        elif tag == "span" and "author" in cls.lower():
            self._in_author = True
            self._buf = ""

        # Keywords: <div class="keywords"> or similar
        elif tag in ("div", "span") and "keyword" in cls.lower():
            self._in_keyword = True
            self._buf = ""

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag in ("h1", "title"):
            self.title = self._buf.strip()
            self._in_title = False
            self._buf = ""

        elif self._in_abstract and tag == "div":
            if not self.abstract:
                self.abstract = re.sub(r"\s+", " ", self._buf).strip()
            self._in_abstract = False
            self._buf = ""

        elif self._in_author and tag == "span":
            name = self._buf.strip()
            if name and len(name) < 100:
                self.authors.append(name)
            self._in_author = False
            self._buf = ""

        elif self._in_keyword and tag in ("div", "span"):
            kw = self._buf.strip()
            if kw:
                self.keywords.extend(k.strip() for k in kw.split(",") if k.strip())
            self._in_keyword = False
            self._buf = ""

    def handle_data(self, data: str) -> None:
        if self._in_title or self._in_abstract or self._in_author or self._in_keyword:
            self._buf += data


# ---------------------------------------------------------------------------
# SSRNMiner
# ---------------------------------------------------------------------------

class SSRNMiner:
    """
    Scrapes SSRN for papers relevant to the BH trading strategy.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    request_delay : float
        Minimum seconds between HTTP requests (default 5 s).
    max_per_run : int
        Hard cap on papers per run.
    """

    def __init__(
        self,
        db_path: str = "idea_engine.db",
        request_delay: float = REQUEST_DELAY_SECONDS,
        max_per_run: int = MAX_PAPERS_PER_RUN,
    ) -> None:
        self.db_path       = db_path
        self.request_delay = request_delay
        self.max_per_run   = max_per_run
        self._db: Optional[sqlite3.Connection] = None
        self._robots: Optional[urllib.robotparser.RobotFileParser] = None
        self._last_request: float = 0.0
        self._ensure_schema()
        self._load_robots()

    # ------------------------------------------------------------------
    # robots.txt
    # ------------------------------------------------------------------

    def _load_robots(self) -> None:
        """Fetch and cache SSRN robots.txt (once per session)."""
        rp = urllib.robotparser.RobotFileParser()
        robots_url = f"{SSRN_BASE}/robots.txt"
        try:
            rp.set_url(robots_url)
            rp.read()
            self._robots = rp
            logger.debug("Loaded robots.txt from %s", robots_url)
        except Exception as exc:
            logger.warning("Could not load robots.txt: %s — proceeding conservatively.", exc)
            self._robots = None

    def _can_fetch(self, url: str) -> bool:
        """Return True if our user-agent is allowed to fetch *url*."""
        if self._robots is None:
            return True
        return self._robots.can_fetch("*", url)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep if needed to honour self.request_delay."""
        elapsed = time.monotonic() - self._last_request
        wait = self.request_delay - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _fetch(self, url: str) -> Optional[str]:
        """
        Fetch *url* obeying robots.txt and rate limiting.

        Returns
        -------
        str or None
            Decoded page text, or None on error.
        """
        if not self._can_fetch(url):
            logger.info("robots.txt disallows: %s", url)
            return None

        self._throttle()
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "SRFMLab-IAE/1.0 (academic research bot; mailto:lab@srfm.local)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                raw = resp.read()
            return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                logger.warning("SSRN returned 403 for %s — likely bot detection.", url)
            else:
                logger.error("HTTP %s for %s", exc.code, url)
            return None
        except Exception as exc:
            logger.error("Fetch error for %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Search & scrape
    # ------------------------------------------------------------------

    def search_keyword(self, keyword: str) -> List[SSRNPaper]:
        """
        Search SSRN for papers matching *keyword*.

        Constructs a search-results URL, scrapes the listing page for
        abstract IDs, then fetches each abstract page individually.

        Parameters
        ----------
        keyword : str

        Returns
        -------
        List[SSRNPaper]
        """
        params = {
            "txtKey_Words": keyword,
            "orderBy":      "ab_approval_date",
            "orderByAsc":   "desc",
            "stype":        "1",
        }
        search_url = f"{SSRN_SEARCH_URL}?{urllib.parse.urlencode(params)}"
        logger.info("SSRN search: %r -> %s", keyword, search_url)

        html_text = self._fetch(search_url)
        if not html_text:
            return []

        abstract_ids = self._extract_abstract_ids(html_text)
        logger.info("Found %d candidate abstract IDs for %r", len(abstract_ids), keyword)

        papers: List[SSRNPaper] = []
        for abs_id in abstract_ids[: self.max_per_run]:
            paper = self.fetch_abstract(abs_id)
            if paper:
                paper.relevance_score = self.score_relevance(paper.abstract, paper.keywords)
                papers.append(paper)

        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return papers

    def _extract_abstract_ids(self, html_text: str) -> List[str]:
        """
        Pull SSRN abstract IDs from a search-results page.

        SSRN search results contain links of the form:
            /sol3/papers.cfm?abstract_id=XXXXXXXX

        Parameters
        ----------
        html_text : str

        Returns
        -------
        List[str]
            Unique abstract ID strings, preserving order.
        """
        seen: Set[str] = set()
        ids: List[str] = []
        for m in re.finditer(r"abstract_id=(\d{6,10})", html_text):
            aid = m.group(1)
            if aid not in seen:
                seen.add(aid)
                ids.append(aid)
        return ids

    def fetch_abstract(self, abstract_id: str) -> Optional[SSRNPaper]:
        """
        Fetch and parse the SSRN abstract page for *abstract_id*.

        Parameters
        ----------
        abstract_id : str
            The numeric SSRN abstract ID.

        Returns
        -------
        SSRNPaper or None
            Parsed paper, or None if the page could not be fetched/parsed.
        """
        url = f"{SSRN_ABSTRACT_BASE}?abstract_id={abstract_id}"
        html_text = self._fetch(url)
        if not html_text:
            return None

        parsed = self.parse_abstract(html_text)
        if not parsed:
            return None

        title, authors, abstract, keywords = parsed
        if not title:
            title = f"SSRN Paper {abstract_id}"

        return SSRNPaper(
            paper_id  = abstract_id,
            title     = title,
            authors   = authors,
            abstract  = abstract,
            keywords  = keywords,
            url       = url,
        )

    def parse_abstract(self, html_text: str) -> Optional[Tuple[str, List[str], str, List[str]]]:
        """
        Extract structured fields from an SSRN abstract page HTML string.

        Falls back to regex-based extraction when the HTML structure
        deviates from the parser's expectations.

        Parameters
        ----------
        html_text : str
            Raw HTML of the SSRN abstract page.

        Returns
        -------
        Tuple[str, List[str], str, List[str]] or None
            (title, authors, abstract, keywords), or None on hard failure.
        """
        # --- Try structured parser first ---
        parser = _SSRNPageParser()
        try:
            parser.feed(html_text)
        except Exception as exc:
            logger.debug("HTML parser exception (non-fatal): %s", exc)

        title    = parser.title
        authors  = parser.authors
        abstract = parser.abstract
        keywords = parser.keywords

        # --- Regex fallbacks for common SSRN patterns ---

        if not title:
            m = re.search(
                r'<h1[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</h1>',
                html_text, re.DOTALL | re.IGNORECASE
            )
            if m:
                title = re.sub(r"<[^>]+>", "", m.group(1)).strip()

        if not title:
            m = re.search(r"<title>([^<]{5,200})</title>", html_text, re.IGNORECASE)
            if m:
                raw = m.group(1)
                # Remove common SSRN suffixes like " by Author :: SSRN"
                title = re.sub(r"\s*(::|\|).*$", "", raw).strip()

        if not abstract:
            m = re.search(
                r'class="[^"]*abstract[^"]*"[^>]*>(.*?)</div>',
                html_text, re.DOTALL | re.IGNORECASE
            )
            if m:
                abstract = re.sub(r"<[^>]+>", " ", m.group(1))
                abstract = re.sub(r"\s+", " ", abstract).strip()

        if not authors:
            # Grab links inside author divs: <a>Author Name</a>
            author_section = re.search(
                r'class="[^"]*authors[^"]*"[^>]*>(.*?)</div>',
                html_text, re.DOTALL | re.IGNORECASE
            )
            if author_section:
                names = re.findall(r"<a[^>]*>([^<]{2,60})</a>", author_section.group(1))
                authors = [html.unescape(n).strip() for n in names if n.strip()]

        if not title:
            return None  # hard fail — no usable content

        return title, authors, abstract, keywords

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_relevance(self, abstract: str, keywords: Optional[List[str]] = None) -> float:
        """
        Score an abstract's relevance to the BH strategy on [0, 1].

        Parameters
        ----------
        abstract : str
        keywords : List[str] or None
            Paper keywords from the abstract page (bonus if match).

        Returns
        -------
        float
            Relevance score in [0.0, 1.0].
        """
        text = abstract.lower()
        if keywords:
            text += " " + " ".join(k.lower() for k in keywords)

        raw = 0.0
        for term in HIGH_VALUE_TERMS:
            raw += text.count(term.lower()) * 0.15
        for term in MEDIUM_VALUE_TERMS:
            raw += text.count(term.lower()) * 0.05

        # Keyword exact matches are high signal
        if keywords:
            for kw in keywords:
                kw_l = kw.lower()
                for hv in HIGH_VALUE_TERMS:
                    if hv in kw_l:
                        raw += 0.20
                        break

        score = raw / (raw + 2.0) if raw > 0 else 0.0
        return round(min(score, 1.0), 4)

    # ------------------------------------------------------------------
    # Run all keywords
    # ------------------------------------------------------------------

    def run(self) -> List[SSRNPaper]:
        """
        Convenience: search all STRATEGY_KEYWORDS, store results.

        Returns
        -------
        List[SSRNPaper]
        """
        seen: dict[str, SSRNPaper] = {}
        for kw in STRATEGY_KEYWORDS:
            if len(seen) >= self.max_per_run:
                break
            papers = self.search_keyword(kw)
            for p in papers:
                if p.paper_id not in seen:
                    seen[p.paper_id] = p
        result = sorted(seen.values(), key=lambda p: p.relevance_score, reverse=True)
        self.store_papers(result)
        return result

    # ------------------------------------------------------------------
    # DB
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(self.db_path)
            self._db.row_factory = sqlite3.Row
        return self._db

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS academic_papers (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source          TEXT    NOT NULL,
                paper_id        TEXT    UNIQUE,
                title           TEXT    NOT NULL,
                authors         TEXT,
                abstract        TEXT,
                relevance_score REAL,
                categories      TEXT,
                url             TEXT,
                mined_at        TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
        """)
        conn.commit()

    def store_papers(self, papers: List[SSRNPaper]) -> int:
        """
        Upsert *papers* into ``academic_papers``.

        Parameters
        ----------
        papers : List[SSRNPaper]

        Returns
        -------
        int
            Rows inserted.
        """
        conn = self._connect()
        inserted = 0
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for p in papers:
            try:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO academic_papers
                        (source, paper_id, title, authors, abstract,
                         relevance_score, categories, url, mined_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "ssrn",
                        p.paper_id,
                        p.title,
                        json.dumps(p.authors),
                        p.abstract,
                        p.relevance_score,
                        json.dumps(p.keywords),
                        p.url,
                        now,
                    ),
                )
                if cur.rowcount:
                    p.db_id = cur.lastrowid
                    inserted += 1
            except sqlite3.Error as exc:
                logger.warning("DB insert failed for %s: %s", p.paper_id, exc)
        conn.commit()
        logger.info("Stored %d/%d SSRN papers.", inserted, len(papers))
        return inserted

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    def __repr__(self) -> str:
        return f"SSRNMiner(db={self.db_path!r}, delay={self.request_delay}s)"

    def __enter__(self) -> "SSRNMiner":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )
    kw = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "momentum trading crypto"
    with SSRNMiner() as miner:
        papers = miner.search_keyword(kw)
        for p in papers:
            print(f"[{p.relevance_score:.3f}] {p.paper_id}  {p.title[:70]}")
