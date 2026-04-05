"""
sentiment_engine/aggregator.py
================================
Aggregates all sentiment sources into per-symbol SentimentSignal objects,
stored in SQLite via the idea_engine.db connection.

Architecture
------------
  TwitterScraper   ─┐
  RedditScraper    ─┤─→ raw text + engagement weights ─→ NLP pipeline
  NewsScraper      ─┘                                      (tokenize → score → extract symbols)
                                                               │
  FearGreedClient ──────────────────────────────────────────→ │
                                                               ↓
                                                     SentimentSignal per symbol
                                                               │
                                                               ↓
                                                     idea_engine.db  (sentiment_signals table)

SentimentSignal schema
----------------------
  symbol          : canonical ticker (BTC, ETH, ...)
  score           : weighted mean sentiment [-1, +1]
  confidence      : 0-1, based on volume + consistency
  volume_mentions : total cross-source mention count
  fear_greed_index: current Fear & Greed value (0-100)
  timestamp       : ISO 8601 UTC of this aggregation run
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .scrapers.twitter_scraper import TwitterScraper
from .scrapers.reddit_scraper  import RedditScraper
from .scrapers.news_scraper    import NewsScraper
from .scrapers.fear_greed      import FearGreedClient
from .nlp.tokenizer            import CryptoTokenizer
from .nlp.sentiment_scorer     import SentimentScorer
from .nlp.symbol_extractor     import SymbolExtractor, SymbolSentiment

logger = logging.getLogger(__name__)

# Default DB path — mirrors idea_engine.db init.py convention
_HERE       = Path(__file__).parent.parent
DEFAULT_DB  = _HERE / "idea_engine.db"

# DDL for the sentiment_signals table (created if not exists)
_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS sentiment_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    symbol          TEXT    NOT NULL,
    score           REAL    NOT NULL,
    confidence      REAL    NOT NULL,
    volume_mentions INTEGER NOT NULL DEFAULT 0,
    fear_greed_index INTEGER,
    source_breakdown TEXT,    -- JSON: {twitter: n, reddit: n, news: n}
    timestamp       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sent_sym ON sentiment_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_sent_ts  ON sentiment_signals(ts);
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SentimentSignal:
    """
    Aggregated per-symbol sentiment signal ready for the signal bridge.

    Attributes
    ----------
    symbol          : Canonical ticker (e.g. 'BTC')
    score           : Ensemble weighted sentiment score [-1, +1]
                      +1 = maximally bullish, -1 = maximally bearish
    confidence      : How reliable this score is [0, 1]
                      Driven by mention volume and cross-source agreement
    volume_mentions : Total number of mentions contributing to this signal
    fear_greed_index: Current Alternative.me F&G value (0-100)
    timestamp       : When this signal was generated (UTC ISO string)
    source_breakdown: Dict with per-source mention counts
    db_id           : Row ID after persistence (None before)
    """
    symbol:           str
    score:            float
    confidence:       float
    volume_mentions:  int
    fear_greed_index: int
    timestamp:        str
    source_breakdown: dict   = None  # type: ignore[assignment]
    db_id:            Optional[int] = None

    def __post_init__(self) -> None:
        if self.source_breakdown is None:
            self.source_breakdown = {}

    @property
    def is_bullish(self) -> bool:
        return self.score > 0.0

    @property
    def is_bearish(self) -> bool:
        return self.score < 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_breakdown"] = json.dumps(d["source_breakdown"])
        return d


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class SentimentAggregator:
    """
    Orchestrates a full scrape-score-aggregate cycle.

    Parameters
    ----------
    db_path        : Path to idea_engine.db
    twitter_mock   : Force mock mode for Twitter scraper
    reddit_mock    : Force mock mode for Reddit scraper
    min_mentions   : Minimum cross-source mentions to emit a signal
    """

    def __init__(
        self,
        db_path:       Path | str = DEFAULT_DB,
        twitter_mock:  bool = True,
        reddit_mock:   bool = True,
        min_mentions:  int  = 2,
    ) -> None:
        self.db_path      = Path(db_path)
        self.min_mentions = min_mentions

        # Scrapers
        self._twitter = TwitterScraper(mock_mode=twitter_mock)
        self._reddit  = RedditScraper(mock_mode=reddit_mock)
        self._news    = NewsScraper()
        self._fg      = FearGreedClient()

        # NLP
        self._tokenizer = CryptoTokenizer()
        self._scorer    = SentimentScorer(tokenizer=self._tokenizer)
        self._extractor = SymbolExtractor(
            scorer_fn=lambda text: self._scorer.score(text).final_score
        )

        # Ensure table exists
        self._ensure_schema()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run_cycle(self) -> list[SentimentSignal]:
        """
        Execute a full scrape-score-aggregate cycle.

        Returns
        -------
        List of SentimentSignal (one per symbol with enough mentions),
        sorted by descending |score|.
        """
        logger.info("SentimentAggregator: starting scrape cycle …")
        ts_now = datetime.now(timezone.utc).isoformat()

        # ── Step 1: Scrape ────────────────────────────────────────────
        fear_greed_val = self._safe_fear_greed()

        twitter_texts: list[tuple[str, float, str]] = []   # (text, engagement, source)
        try:
            tweets = self._twitter.fetch_recent()
            for t in tweets:
                twitter_texts.append((t.text, t.engagement_weight, "twitter"))
        except Exception as exc:
            logger.error("Twitter scrape failed: %s", exc)

        reddit_texts: list[tuple[str, float, str]] = []
        try:
            posts = self._reddit.fetch_all()
            for p in posts:
                reddit_texts.append((p.all_text(), p.combined_weight(), "reddit"))
        except Exception as exc:
            logger.error("Reddit scrape failed: %s", exc)

        news_texts: list[tuple[str, float, str]] = []
        try:
            articles = self._news.fetch_all()
            for a in articles:
                news_texts.append((a.full_text, 1.0, a.source))
        except Exception as exc:
            logger.error("News scrape failed: %s", exc)

        all_texts = twitter_texts + reddit_texts + news_texts
        if not all_texts:
            logger.warning("SentimentAggregator: no texts collected this cycle.")
            return []

        # ── Step 2: Score + Extract symbols ──────────────────────────
        symbol_accum: dict[str, list[tuple[float, float]]] = {}  # symbol → [(score, weight)]
        source_counts: dict[str, dict[str, int]] = {}            # symbol → {source: count}

        for text, engagement, source in all_texts:
            if not text.strip():
                continue

            # Score the full text
            scored = self._scorer.score(text, source=source, engagement=engagement)

            # Extract per-symbol attributions
            per_sym = self._extractor.extract(self._tokenizer.normalize(text))
            for sym, ss in per_sym.items():
                if sym not in symbol_accum:
                    symbol_accum[sym] = []
                    source_counts[sym] = {"twitter": 0, "reddit": 0, "news": 0}

                # Weight = engagement × recency (already in weighted_score) × mention_count
                contrib_weight = scored.engagement_weight * scored.recency_decay + 1e-6
                symbol_accum[sym].append((ss.weighted_score, contrib_weight))

                src_key = source if source in ("twitter", "reddit") else "news"
                source_counts[sym][src_key] += ss.mention_count

        # ── Step 3: Aggregate per symbol ─────────────────────────────
        signals: list[SentimentSignal] = []
        for sym, score_weight_pairs in symbol_accum.items():
            total_mentions = sum(
                source_counts[sym][k] for k in source_counts[sym]
            )
            if total_mentions < self.min_mentions:
                continue

            total_weight = sum(w for _, w in score_weight_pairs)
            if total_weight == 0:
                agg_score = 0.0
            else:
                agg_score = sum(s * w for s, w in score_weight_pairs) / total_weight

            agg_score = max(-1.0, min(1.0, agg_score))

            # Confidence: volume × consistency
            import math
            vol_factor = min(1.0, math.log1p(total_mentions) / math.log1p(30))
            raw_scores = [s for s, _ in score_weight_pairs]
            mean_s     = sum(raw_scores) / len(raw_scores)
            variance   = sum((x - mean_s) ** 2 for x in raw_scores) / len(raw_scores)
            consistency = max(0.0, 1.0 - math.sqrt(variance))
            confidence  = vol_factor * consistency

            sig = SentimentSignal(
                symbol=sym,
                score=round(agg_score, 4),
                confidence=round(confidence, 4),
                volume_mentions=total_mentions,
                fear_greed_index=fear_greed_val,
                timestamp=ts_now,
                source_breakdown=dict(source_counts[sym]),
            )
            signals.append(sig)

        # Sort by signal strength
        signals.sort(key=lambda s: abs(s.score), reverse=True)

        # ── Step 4: Persist ──────────────────────────────────────────
        self._persist(signals)

        logger.info(
            "SentimentAggregator: cycle complete — %d signals generated, F&G=%d.",
            len(signals), fear_greed_val,
        )
        return signals

    def query_latest(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Query the most recent sentiment signals from the DB.

        Parameters
        ----------
        symbol : filter by symbol (optional)
        limit  : max rows to return

        Returns
        -------
        List of row dicts sorted by ts descending.
        """
        conn = self._get_conn()
        try:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM sentiment_signals WHERE symbol=? ORDER BY ts DESC LIMIT ?",
                    (symbol.upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sentiment_signals ORDER BY ts DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _safe_fear_greed(self) -> int:
        """Fetch F&G index, returning 50 (neutral) on any failure."""
        try:
            return self._fg.get_current().value
        except Exception as exc:
            logger.warning("Fear & Greed fetch failed: %s", exc)
            return 50

    def _persist(self, signals: list[SentimentSignal]) -> None:
        """Write signals to sentiment_signals table."""
        if not signals:
            return
        conn = self._get_conn()
        try:
            for sig in signals:
                cur = conn.execute(
                    """
                    INSERT INTO sentiment_signals
                        (symbol, score, confidence, volume_mentions,
                         fear_greed_index, source_breakdown, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sig.symbol,
                        sig.score,
                        sig.confidence,
                        sig.volume_mentions,
                        sig.fear_greed_index,
                        json.dumps(sig.source_breakdown),
                        sig.timestamp,
                    ),
                )
                sig.db_id = cur.lastrowid
            conn.commit()
            logger.debug("Persisted %d sentiment signals.", len(signals))
        except Exception as exc:
            logger.error("Persist failed: %s", exc)
            conn.rollback()
        finally:
            conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        """Create sentiment_signals table if it does not exist."""
        conn = self._get_conn()
        try:
            conn.executescript(_CREATE_SQL)
            conn.commit()
        finally:
            conn.close()
