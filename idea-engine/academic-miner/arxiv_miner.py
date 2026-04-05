"""
arxiv_miner.py — ArXiv Academic Paper Miner
============================================
Queries the public arXiv API (https://export.arxiv.org/api/query) for papers
relevant to the BH trading strategy and extracts actionable ideas from abstracts.

The miner enforces polite rate-limiting (3 s between requests) and caps each
run at MAX_PAPERS_PER_RUN to avoid hammering the API.

Classes
-------
    ArXivPaper      — Lightweight data container for a single arXiv result.
    ArXivMiner      — Main miner: search → score → extract → store.

Database
--------
    Requires the `academic_papers` table defined in schema_extension.sql.
    Pass a path to an existing SQLite database or let the class create it.

Usage
-----
    miner  = ArXivMiner(db_path="idea_engine.db")
    papers = miner.search("momentum trading", max_results=20)
    miner.store_papers(papers)
    for p in papers:
        print(p.title, "—", round(p.relevance_score, 3))
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARXIV_API_BASE: str = "https://export.arxiv.org/api/query"

# arXiv categories we care about for the BH strategy
TARGET_CATEGORIES: List[str] = [
    "q-fin.TR",   # Trading and Market Microstructure
    "q-fin.PM",   # Portfolio Management
    "q-fin.ST",   # Statistical Finance
    "cs.LG",      # Machine Learning
    "stat.ML",    # Machine Learning (stats)
]

# Primary search keywords aligned with BH strategy components
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

# Scoring keyword banks — higher weights signal closer alignment to BH strategy
HIGH_VALUE_TERMS: List[str] = [
    "momentum", "mean reversion", "volatility", "regime", "drawdown",
    "kelly", "ornstein-uhlenbeck", "microstructure", "causal", "genetic",
    "reinforcement learning", "entropy", "fractal", "hurst", "kalman",
    "black hole", "phase transition", "turbulence", "liquidity", "spread",
    "order flow", "market impact", "alpha decay", "signal decay",
]

MEDIUM_VALUE_TERMS: List[str] = [
    "cryptocurrency", "bitcoin", "ethereum", "crypto", "digital asset",
    "high frequency", "intraday", "tick data", "limit order book",
    "portfolio optimization", "risk management", "position sizing",
    "backtest", "out-of-sample", "walk-forward", "sharpe", "sortino",
    "information ratio", "maximum drawdown", "var", "cvar",
    "neural network", "lstm", "transformer", "gradient boosting",
    "random forest", "feature importance", "dimensionality reduction",
]

LOW_VALUE_TERMS: List[str] = [
    "stock", "equity", "bond", "futures", "option", "derivative",
    "market", "trading", "financial", "return", "price", "volume",
    "correlation", "covariance", "factor", "model", "estimate",
    "empirical", "analysis", "study", "evidence", "test",
]

# Per-run cap to avoid rate-limit bans
MAX_PAPERS_PER_RUN: int = 10

# Polite delay between HTTP requests (seconds)
REQUEST_DELAY_SECONDS: float = 3.0

# arXiv XML namespace
NS: dict = {"atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom"}

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ArXivPaper:
    """Represents a single arXiv paper with metadata and scoring fields."""

    paper_id: str                    # e.g. "2301.12345"
    title: str
    abstract: str
    authors: List[str]               = field(default_factory=list)
    categories: List[str]            = field(default_factory=list)
    url: str                         = ""
    published: str                   = ""
    relevance_score: float           = 0.0
    ideas: List[str]                 = field(default_factory=list)  # extracted ideas
    db_id: Optional[int]             = None                          # row id after insert

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "paper_id":       self.paper_id,
            "title":          self.title,
            "abstract":       self.abstract,
            "authors":        self.authors,
            "categories":     self.categories,
            "url":            self.url,
            "published":      self.published,
            "relevance_score": self.relevance_score,
            "ideas":          self.ideas,
        }

    def __repr__(self) -> str:
        return (f"ArXivPaper(id={self.paper_id!r}, "
                f"score={self.relevance_score:.3f}, title={self.title[:60]!r})")


# ---------------------------------------------------------------------------
# ArXivMiner
# ---------------------------------------------------------------------------

class ArXivMiner:
    """
    Mines arXiv for papers relevant to the BH trading strategy.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database (will be created if absent).
    request_delay : float
        Seconds to sleep between consecutive HTTP requests.
    max_per_run : int
        Hard cap on papers fetched per single run() call.
    """

    def __init__(
        self,
        db_path: str = "idea_engine.db",
        request_delay: float = REQUEST_DELAY_SECONDS,
        max_per_run: int = MAX_PAPERS_PER_RUN,
    ) -> None:
        self.db_path        = db_path
        self.request_delay  = request_delay
        self.max_per_run    = max_per_run
        self._db: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(self.db_path)
            self._db.row_factory = sqlite3.Row
        return self._db

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
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
            CREATE TABLE IF NOT EXISTS hypothesis_candidates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source_paper_id INTEGER REFERENCES academic_papers(id),
                hypothesis_text TEXT    NOT NULL,
                mapped_component TEXT,
                param_suggestions TEXT,
                confidence      REAL,
                status          TEXT    NOT NULL DEFAULT 'pending',
                created_at      TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
        """)
        conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _fetch_url(self, url: str, timeout: int = 30) -> str:
        """Fetch URL content with a polite delay, returning decoded text."""
        logger.debug("GET %s", url)
        time.sleep(self.request_delay)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SRFMLab-IAE/1.0 (research; mailto:lab@srfm.local)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            logger.error("HTTP %s for %s", exc.code, url)
            raise
        except urllib.error.URLError as exc:
            logger.error("URL error for %s: %s", url, exc.reason)
            raise

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 50) -> List[ArXivPaper]:
        """
        Query the arXiv API for papers matching *query*.

        The search is restricted to TARGET_CATEGORIES and results are
        ranked by the arXiv default (relevance). Each paper is scored
        with :meth:`relevance_score` before being returned.

        Parameters
        ----------
        query : str
            Free-text search query.
        max_results : int
            Maximum papers to retrieve from arXiv (before local cap).

        Returns
        -------
        List[ArXivPaper]
            Papers sorted by descending relevance score.
        """
        effective_max = min(max_results, self.max_per_run)
        category_filter = " OR ".join(f"cat:{c}" for c in TARGET_CATEGORIES)
        full_query = f"({query}) AND ({category_filter})"
        params = {
            "search_query": full_query,
            "start":        0,
            "max_results":  effective_max,
            "sortBy":       "relevance",
            "sortOrder":    "descending",
        }
        url = f"{ARXIV_API_BASE}?{urllib.parse.urlencode(params)}"
        logger.info("Searching arXiv: %r (cap=%d)", query, effective_max)

        try:
            xml_text = self._fetch_url(url)
        except Exception as exc:
            logger.error("arXiv search failed: %s", exc)
            return []

        papers = self._parse_atom(xml_text)
        for p in papers:
            p.relevance_score = self.relevance_score(p)
            p.ideas = self.extract_ideas(p)

        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        logger.info("Found %d papers for query %r", len(papers), query)
        return papers

    def search_all_keywords(self) -> List[ArXivPaper]:
        """
        Run :meth:`search` for every keyword in STRATEGY_KEYWORDS.

        Deduplicates by paper_id and honours the per-run cap across the
        full keyword sweep.

        Returns
        -------
        List[ArXivPaper]
            Deduplicated list sorted by relevance score.
        """
        seen: dict[str, ArXivPaper] = {}
        for kw in STRATEGY_KEYWORDS:
            if len(seen) >= self.max_per_run:
                logger.info("Per-run cap of %d reached; stopping keyword sweep.", self.max_per_run)
                break
            remaining = self.max_per_run - len(seen)
            papers = self.search(kw, max_results=remaining)
            for p in papers:
                if p.paper_id not in seen:
                    seen[p.paper_id] = p
        result = sorted(seen.values(), key=lambda p: p.relevance_score, reverse=True)
        return result

    # ------------------------------------------------------------------
    # XML parsing
    # ------------------------------------------------------------------

    def _parse_atom(self, xml_text: str) -> List[ArXivPaper]:
        """Parse arXiv Atom feed XML into ArXivPaper objects."""
        papers: List[ArXivPaper] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.error("XML parse error: %s", exc)
            return papers

        for entry in root.findall("atom:entry", NS):
            paper = self._entry_to_paper(entry)
            if paper:
                papers.append(paper)
        return papers

    def _entry_to_paper(self, entry: ET.Element) -> Optional[ArXivPaper]:
        """Convert a single <entry> element to an ArXivPaper."""
        try:
            raw_id = entry.findtext("atom:id", default="", namespaces=NS)
            # Strip base URL: http://arxiv.org/abs/2301.12345v1 → 2301.12345
            paper_id = re.sub(r"^.*abs/", "", raw_id).rstrip("v0123456789")
            if not paper_id:
                return None

            title    = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip()
            abstract = (entry.findtext("atom:summary", default="", namespaces=NS) or "").strip()
            abstract = re.sub(r"\s+", " ", abstract)  # collapse whitespace

            authors = [
                a.findtext("atom:name", default="", namespaces=NS).strip()
                for a in entry.findall("atom:author", NS)
            ]

            categories = [
                t.get("term", "")
                for t in entry.findall("atom:category", NS)
            ]

            url = ""
            for link in entry.findall("atom:link", NS):
                if link.get("rel") == "alternate":
                    url = link.get("href", "")

            published = entry.findtext("atom:published", default="", namespaces=NS)

            return ArXivPaper(
                paper_id   = paper_id,
                title      = title,
                abstract   = abstract,
                authors    = authors,
                categories = categories,
                url        = url,
                published  = published,
            )
        except Exception as exc:
            logger.warning("Failed to parse entry: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def relevance_score(self, paper: ArXivPaper) -> float:
        """
        Score a paper's relevance to the BH strategy on [0, 1].

        Algorithm
        ---------
        1. Combine title + abstract into a single lower-case search text.
        2. Award weighted hits for HIGH / MEDIUM / LOW term matches.
        3. Add a category bonus for exact TARGET_CATEGORIES membership.
        4. Normalise to [0, 1] using a soft sigmoid-like cap.

        Parameters
        ----------
        paper : ArXivPaper

        Returns
        -------
        float
            Relevance score in [0.0, 1.0].
        """
        text = (paper.title + " " + paper.abstract).lower()

        raw = 0.0
        # Term matching
        for term in HIGH_VALUE_TERMS:
            count = text.count(term.lower())
            raw += count * 0.15
        for term in MEDIUM_VALUE_TERMS:
            count = text.count(term.lower())
            raw += count * 0.05
        for term in LOW_VALUE_TERMS:
            count = text.count(term.lower())
            raw += count * 0.01

        # Category bonus
        for cat in paper.categories:
            if cat in TARGET_CATEGORIES:
                raw += 0.20

        # Soft normalise: score / (score + 2) maps [0,∞) → [0,1)
        score = raw / (raw + 2.0) if raw > 0 else 0.0
        return round(min(score, 1.0), 4)

    # ------------------------------------------------------------------
    # Idea extraction (lightweight delegation)
    # ------------------------------------------------------------------

    def extract_ideas(self, paper: ArXivPaper) -> List[str]:
        """
        Extract a list of brief idea strings from a paper's abstract.

        This is a fast keyword-pattern pass; for deep NLP extraction use
        :class:`IdeaExtractor` directly.

        Parameters
        ----------
        paper : ArXivPaper

        Returns
        -------
        List[str]
            Up to 5 short idea strings.
        """
        ideas: List[str] = []
        text = paper.abstract

        # Pattern: "we propose/present/introduce X"
        propose_matches = re.findall(
            r"we (?:propose|present|introduce|develop|demonstrate)\s+([^.]{10,80})\.",
            text, re.IGNORECASE
        )
        ideas.extend(m.strip() for m in propose_matches[:2])

        # Pattern: "X outperforms/improves Y"
        perf_matches = re.findall(
            r"([A-Z][^.]{5,60})\s+(?:outperforms|improves|beats|achieves)\s+([^.]{5,60})\.",
            text
        )
        for m in perf_matches[:2]:
            ideas.append(f"{m[0].strip()} outperforms {m[1].strip()}")

        # Pattern: "X is a significant predictor of Y"
        pred_matches = re.findall(
            r"([A-Z][^.]{5,50})\s+(?:is a|are)\s+significant\s+predictor[s]?\s+of\s+([^.]{5,50})\.",
            text
        )
        for m in pred_matches[:1]:
            ideas.append(f"{m[0].strip()} predicts {m[1].strip()}")

        return ideas[:5]

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_papers(self, papers: List[ArXivPaper]) -> int:
        """
        Upsert *papers* into the ``academic_papers`` table.

        Duplicate paper_id rows are skipped (INSERT OR IGNORE).

        Parameters
        ----------
        papers : List[ArXivPaper]

        Returns
        -------
        int
            Number of rows actually inserted (not counting skips).
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
                        "arxiv",
                        p.paper_id,
                        p.title,
                        json.dumps(p.authors),
                        p.abstract,
                        p.relevance_score,
                        json.dumps(p.categories),
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
        logger.info("Stored %d/%d papers to DB.", inserted, len(papers))
        return inserted

    # ------------------------------------------------------------------
    # Convenience: full run
    # ------------------------------------------------------------------

    def run(self) -> List[ArXivPaper]:
        """
        Convenience method: sweep all STRATEGY_KEYWORDS, store results.

        Returns
        -------
        List[ArXivPaper]
            All newly stored papers.
        """
        papers = self.search_all_keywords()
        self.store_papers(papers)
        return papers

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def iter_stored(self, min_score: float = 0.0) -> Iterator[dict]:
        """
        Yield stored arXiv papers from the DB as plain dicts.

        Parameters
        ----------
        min_score : float
            Only yield papers with relevance_score >= min_score.
        """
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT * FROM academic_papers
            WHERE source = 'arxiv' AND relevance_score >= ?
            ORDER BY relevance_score DESC
            """,
            (min_score,),
        )
        for row in cur:
            yield dict(row)

    def top_papers(self, n: int = 10, min_score: float = 0.5) -> List[dict]:
        """
        Return the top-*n* stored papers by relevance score.

        Parameters
        ----------
        n : int
        min_score : float

        Returns
        -------
        List[dict]
        """
        return list(self.iter_stored(min_score=min_score))[:n]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ArXivMiner(db={self.db_path!r}, delay={self.request_delay}s)"

    def __enter__(self) -> "ArXivMiner":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CLI entry-point (python -m academic_miner.arxiv_miner)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "momentum trading crypto"
    with ArXivMiner() as miner:
        papers = miner.search(query, max_results=5)
        for p in papers:
            print(f"[{p.relevance_score:.3f}] {p.paper_id}  {p.title[:70]}")
            if p.ideas:
                for idea in p.ideas:
                    print(f"   -> {idea}")
