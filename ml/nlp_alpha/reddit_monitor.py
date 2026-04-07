"""
reddit_monitor.py -- Reddit sentiment monitoring for crypto alpha signals.

Monitors crypto-focused subreddits and generates rolling sentiment signals.
Uses SQLite for a lightweight rolling 24h cache of processed posts.
No Reddit API required for offline testing -- designed to work with injected data.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .sentiment_analyzer import FinancialLexicon, HeadlineScorer, SentimentScore


# ---------------------------------------------------------------------------
# SubredditConfig
# ---------------------------------------------------------------------------

@dataclass
class SubredditConfig:
    """Configuration for a monitored subreddit."""
    name: str
    symbols_covered: List[str]
    credibility_weight: float       # [0, 1] -- higher = more trusted
    post_threshold: int             # minimum upvotes to consider a post
    all_caps_penalty: float = 0.5   # weight multiplier for all-caps titles
    exclamation_limit: int = 3      # posts with > N exclamation marks get penalized
    exclamation_penalty: float = 0.7


# ---------------------------------------------------------------------------
# Preconfigured subreddit registry
# ---------------------------------------------------------------------------

SUBREDDIT_CONFIGS: Dict[str, SubredditConfig] = {
    "r/CryptoCurrency": SubredditConfig(
        name="r/CryptoCurrency",
        symbols_covered=["BTC", "ETH", "SOL", "ADA", "DOT", "AVAX", "LINK", "UNI", "MATIC"],
        credibility_weight=0.8,
        post_threshold=50,
    ),
    "r/Bitcoin": SubredditConfig(
        name="r/Bitcoin",
        symbols_covered=["BTC"],
        credibility_weight=0.9,
        post_threshold=25,
    ),
    "r/ethereum": SubredditConfig(
        name="r/ethereum",
        symbols_covered=["ETH"],
        credibility_weight=0.85,
        post_threshold=25,
    ),
    "r/solana": SubredditConfig(
        name="r/solana",
        symbols_covered=["SOL"],
        credibility_weight=0.8,
        post_threshold=20,
    ),
    "r/wallstreetbets": SubredditConfig(
        name="r/wallstreetbets",
        symbols_covered=["AAPL", "TSLA", "NVDA", "AMZN", "MSFT", "META", "AMD"],
        credibility_weight=0.6,
        post_threshold=100,
        all_caps_penalty=0.4,
        exclamation_penalty=0.5,
    ),
    "r/investing": SubredditConfig(
        name="r/investing",
        symbols_covered=["AAPL", "TSLA", "NVDA", "AMZN", "MSFT", "BTC", "ETH"],
        credibility_weight=0.85,
        post_threshold=30,
    ),
    "r/options": SubredditConfig(
        name="r/options",
        symbols_covered=["AAPL", "TSLA", "SPY", "QQQ", "NVDA", "AMZN"],
        credibility_weight=0.75,
        post_threshold=40,
    ),
    "r/ethfinance": SubredditConfig(
        name="r/ethfinance",
        symbols_covered=["ETH"],
        credibility_weight=0.88,
        post_threshold=15,
    ),
    "r/CryptoMarkets": SubredditConfig(
        name="r/CryptoMarkets",
        symbols_covered=["BTC", "ETH", "SOL", "ADA", "XRP"],
        credibility_weight=0.75,
        post_threshold=30,
    ),
}


# ---------------------------------------------------------------------------
# PostSignal dataclass
# ---------------------------------------------------------------------------

@dataclass
class PostSignal:
    """Sentiment signal extracted from a single Reddit post."""
    symbol: str
    sentiment: float                # [-1, 1] base sentiment
    confidence: float               # [0, 1]
    virality_score: float           # log(upvotes + 1)
    subreddit: str = ""
    post_id: str = ""
    title: str = ""
    timestamp: Optional[datetime] = None
    upvotes: int = 0
    n_comments: int = 0
    weight_adjusted: float = 0.0   # final weight after all adjustments

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        # Compute weight_adjusted if not explicitly provided
        if self.weight_adjusted == 0.0:
            self.weight_adjusted = self.virality_score * self.confidence

    def effective_signal(self) -> float:
        """Weight-adjusted sentiment signal."""
        return self.sentiment * self.weight_adjusted


# ---------------------------------------------------------------------------
# PostAnalyzer
# ---------------------------------------------------------------------------

class PostAnalyzer:
    """
    Analyzes individual Reddit posts for sentiment signals.

    Applies:
    - Upvote threshold filtering
    - All-caps / excessive exclamation mark detection
    - Subreddit credibility weighting
    - Virality scoring via log(upvotes + 1)
    """

    # Regex for detecting crypto symbols in text
    _TICKER_PATTERN = re.compile(r"\b([A-Z]{2,8})\b")

    def __init__(
        self,
        lexicon: Optional[FinancialLexicon] = None,
        scorer: Optional[HeadlineScorer] = None,
    ) -> None:
        self._lexicon = lexicon or FinancialLexicon()
        self._scorer = scorer or HeadlineScorer(self._lexicon)

    def _is_all_caps(self, title: str) -> bool:
        """Returns True if the majority of alphabetic chars are uppercase."""
        alpha_chars = [c for c in title if c.isalpha()]
        if len(alpha_chars) < 5:
            return False
        upper_count = sum(1 for c in alpha_chars if c.isupper())
        return upper_count / len(alpha_chars) > 0.75

    def _count_exclamations(self, text: str) -> int:
        return text.count("!")

    def _compute_virality(self, upvotes: int, n_comments: int = 0) -> float:
        """
        Virality score = log(upvotes + 1) + 0.3 * log(n_comments + 1).
        Higher is more viral.
        """
        return math.log(upvotes + 1) + 0.3 * math.log(n_comments + 1)

    def _extract_symbols_from_text(self, text: str, known_symbols: Optional[List[str]] = None) -> List[str]:
        """Extract mentioned crypto/equity symbols from text."""
        found: List[str] = []
        matches = self._TICKER_PATTERN.findall(text.upper())

        # Common crypto symbols to look for
        crypto_symbols = {
            "BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "AVAX", "LINK",
            "UNI", "MATIC", "DOGE", "SHIB", "LTC", "BCH", "ATOM", "NEAR",
        }
        equity_symbols = {
            "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NVDA",
            "AMD", "INTC", "SPY", "QQQ",
        }
        all_known = crypto_symbols | equity_symbols
        if known_symbols:
            all_known |= set(s.upper() for s in known_symbols)

        for m in matches:
            if m in all_known and m not in found:
                found.append(m)

        return found

    def analyze_post(
        self,
        title: str,
        selftext: str,
        score: int,
        n_comments: int,
        subreddit: str = "",
        post_id: str = "",
        timestamp: Optional[datetime] = None,
        target_symbol: str = "",
    ) -> Optional[PostSignal]:
        """
        Analyze a single Reddit post.

        Parameters
        ----------
        title : str
        selftext : str
        score : int
            Reddit post score (upvotes - downvotes).
        n_comments : int
        subreddit : str
        post_id : str
        timestamp : datetime, optional
        target_symbol : str
            If provided, only analyze if this symbol is mentioned.

        Returns
        -------
        PostSignal or None if post is filtered out.
        """
        config = SUBREDDIT_CONFIGS.get(subreddit)
        threshold = config.post_threshold if config else 10

        # -- Filter: score too low --
        if score < threshold:
            return None

        # Combine title and body (truncated)
        full_text = title
        if selftext:
            full_text += " " + selftext[:500]

        # -- Filter by symbol relevance --
        if target_symbol:
            text_upper = full_text.upper()
            if target_symbol.upper() not in text_upper:
                # Check full name
                from .sentiment_analyzer import SYMBOL_NAME_MAP
                found = False
                for name, ticker in SYMBOL_NAME_MAP.items():
                    if ticker == target_symbol.upper() and name.lower() in full_text.lower():
                        found = True
                        break
                if not found:
                    return None

        # -- Compute base sentiment --
        sentiment_score = self._scorer.score_headline(
            full_text,
            symbol=target_symbol,
            source=subreddit,
            timestamp=timestamp,
        )

        # -- Suspicion adjustments --
        weight_mult = 1.0
        if self._is_all_caps(title):
            penalty = config.all_caps_penalty if config else 0.5
            weight_mult *= penalty

        n_exclamations = self._count_exclamations(title)
        exclamation_limit = config.exclamation_limit if config else 3
        if n_exclamations > exclamation_limit:
            penalty = config.exclamation_penalty if config else 0.7
            weight_mult *= penalty

        # -- Virality score --
        virality = self._compute_virality(max(0, score), n_comments)

        # -- Subreddit credibility --
        credibility = config.credibility_weight if config else 0.7

        final_weight = virality * credibility * weight_mult * sentiment_score.confidence

        return PostSignal(
            symbol=target_symbol,
            sentiment=sentiment_score.adjusted_score,
            confidence=sentiment_score.confidence,
            virality_score=virality,
            subreddit=subreddit,
            post_id=post_id,
            title=title[:200],
            timestamp=timestamp or datetime.now(timezone.utc),
            upvotes=score,
            n_comments=n_comments,
            weight_adjusted=max(0.0, final_weight),
        )

    def analyze_batch(
        self,
        posts: List[Dict],
        target_symbol: str = "",
    ) -> List[PostSignal]:
        """
        Analyze a batch of posts.

        Each dict may contain:
        - title (str, required)
        - selftext (str)
        - score (int)
        - num_comments (int)
        - subreddit (str)
        - id (str)
        - created_utc (float or datetime)
        """
        results: List[PostSignal] = []
        for post in posts:
            title = post.get("title", "")
            selftext = post.get("selftext", "")
            score = post.get("score", 0)
            n_comments = post.get("num_comments", 0)
            subreddit = post.get("subreddit", "")
            post_id = post.get("id", "")

            # Parse timestamp
            ts = post.get("timestamp") or post.get("created_utc")
            if isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts, tz=timezone.utc)

            signal = self.analyze_post(
                title, selftext, score, n_comments,
                subreddit, post_id, ts, target_symbol,
            )
            if signal is not None:
                results.append(signal)

        return results


# ---------------------------------------------------------------------------
# RedditSentimentCache (SQLite-backed)
# ---------------------------------------------------------------------------

class RedditSentimentCache:
    """
    SQLite-backed rolling cache for processed Reddit post signals.

    Stores PostSignal data with timestamps, supporting:
    - Batch insert of new signals
    - Rolling window sentiment aggregation
    - Automatic pruning of old entries (>24h)
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS post_signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id     TEXT NOT NULL DEFAULT '',
            symbol      TEXT NOT NULL,
            sentiment   REAL NOT NULL,
            confidence  REAL NOT NULL,
            virality    REAL NOT NULL,
            weight_adj  REAL NOT NULL,
            subreddit   TEXT NOT NULL DEFAULT '',
            title       TEXT NOT NULL DEFAULT '',
            upvotes     INTEGER NOT NULL DEFAULT 0,
            n_comments  INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL
        );
    """

    _CREATE_IDX_SYMBOL = """
        CREATE INDEX IF NOT EXISTS idx_symbol ON post_signals(symbol);
    """

    _CREATE_IDX_CREATED = """
        CREATE INDEX IF NOT EXISTS idx_created ON post_signals(created_at);
    """

    DEFAULT_DB_PATH = ":memory:"
    MAX_AGE_HOURS = 24.0
    HALF_LIFE_HOURS = 2.0   # EW decay half-life for sentiment aggregation

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        max_age_hours: float = MAX_AGE_HOURS,
        half_life_hours: float = HALF_LIFE_HOURS,
    ) -> None:
        self.db_path = db_path
        self.max_age_hours = max_age_hours
        self.half_life_hours = half_life_hours
        self._lambda = math.log(2.0) / (half_life_hours * 60.0)   # per-minute decay
        self._lock = threading.Lock()
        self._conn = self._init_db()

    def _init_db(self) -> sqlite3.Connection:
        """Initialize SQLite connection and schema."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(self._CREATE_TABLE)
        conn.execute(self._CREATE_IDX_SYMBOL)
        conn.execute(self._CREATE_IDX_CREATED)
        conn.commit()
        return conn

    def _ts_str(self, dt: datetime) -> str:
        """Serialize datetime to ISO string."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    def _parse_ts(self, ts_str: str) -> datetime:
        """Deserialize ISO string to datetime."""
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _age_minutes(self, ts: datetime, now: Optional[datetime] = None) -> float:
        """Age in minutes."""
        now = now or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return max(0.0, (now - ts).total_seconds() / 60.0)

    def _decay_weight(self, age_minutes: float) -> float:
        """Exponential decay weight."""
        return math.exp(-self._lambda * age_minutes)

    def update(self, posts: List[PostSignal]) -> int:
        """
        Insert a batch of PostSignals into the cache.
        Returns number of rows inserted.
        """
        if not posts:
            return 0

        rows = []
        for p in posts:
            ts = p.timestamp or datetime.now(timezone.utc)
            rows.append((
                p.post_id,
                p.symbol.upper(),
                p.sentiment,
                p.confidence,
                p.virality_score,
                p.weight_adjusted,
                p.subreddit,
                p.title[:200],
                p.upvotes,
                p.n_comments,
                self._ts_str(ts),
            ))

        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO post_signals
                    (post_id, symbol, sentiment, confidence, virality, weight_adj,
                     subreddit, title, upvotes, n_comments, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()

        return len(rows)

    def get_rolling_sentiment(
        self,
        symbol: str,
        window_hours: float = 4.0,
        now: Optional[datetime] = None,
    ) -> float:
        """
        Compute exponentially-weighted rolling sentiment for a symbol.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g. "BTC", "ETH").
        window_hours : float
            Time window to look back.
        now : datetime, optional

        Returns
        -------
        float in [-1, 1], or 0.0 if no data.
        """
        now = now or datetime.now(timezone.utc)
        cutoff = now.timestamp() - window_hours * 3600.0

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT sentiment, confidence, weight_adj, created_at
                FROM post_signals
                WHERE symbol = ?
                  AND datetime(created_at) >= datetime(?, 'unixepoch')
                ORDER BY created_at DESC
                """,
                (symbol.upper(), cutoff),
            ).fetchall()

        if not rows:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for sentiment, confidence, weight_adj, ts_str in rows:
            ts = self._parse_ts(ts_str)
            age_minutes = self._age_minutes(ts, now)
            decay = self._decay_weight(age_minutes)
            w = decay * max(0.0, weight_adj)
            weighted_sum += sentiment * w
            total_weight += w

        if total_weight == 0.0:
            return 0.0

        return max(-1.0, min(1.0, weighted_sum / total_weight))

    def prune_old(self, max_age_hours: Optional[float] = None) -> int:
        """
        Delete entries older than max_age_hours.
        Returns number of rows deleted.
        """
        max_age = max_age_hours or self.max_age_hours
        cutoff_ts = datetime.now(timezone.utc).timestamp() - max_age * 3600.0

        with self._lock:
            cursor = self._conn.execute(
                """
                DELETE FROM post_signals
                WHERE datetime(created_at) < datetime(?, 'unixepoch')
                """,
                (cutoff_ts,),
            )
            self._conn.commit()
            return cursor.rowcount

    def get_symbol_stats(
        self, symbol: str, window_hours: float = 24.0
    ) -> Dict:
        """
        Summary stats for a symbol over the past window_hours.
        """
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - window_hours * 3600.0

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT sentiment, confidence, virality, upvotes, subreddit, created_at
                FROM post_signals
                WHERE symbol = ?
                  AND datetime(created_at) >= datetime(?, 'unixepoch')
                """,
                (symbol.upper(), cutoff),
            ).fetchall()

        if not rows:
            return {
                "symbol": symbol,
                "n_posts": 0,
                "avg_sentiment": 0.0,
                "rolling_signal": 0.0,
                "avg_confidence": 0.0,
                "total_virality": 0.0,
                "subreddits": [],
            }

        sentiments = [r[0] for r in rows]
        confidences = [r[1] for r in rows]
        viralities = [r[2] for r in rows]
        subreddits = list(set(r[4] for r in rows))

        return {
            "symbol": symbol,
            "n_posts": len(rows),
            "avg_sentiment": sum(sentiments) / len(sentiments),
            "rolling_signal": self.get_rolling_sentiment(symbol, window_hours=4.0),
            "avg_confidence": sum(confidences) / len(confidences),
            "total_virality": sum(viralities),
            "subreddits": subreddits,
        }

    def get_top_posts(
        self, symbol: str, window_hours: float = 4.0, limit: int = 10
    ) -> List[Dict]:
        """Return top posts by virality for a symbol."""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - window_hours * 3600.0

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT post_id, title, sentiment, confidence, virality,
                       upvotes, n_comments, subreddit, created_at
                FROM post_signals
                WHERE symbol = ?
                  AND datetime(created_at) >= datetime(?, 'unixepoch')
                ORDER BY virality DESC
                LIMIT ?
                """,
                (symbol.upper(), cutoff, limit),
            ).fetchall()

        return [
            {
                "post_id": r[0],
                "title": r[1],
                "sentiment": r[2],
                "confidence": r[3],
                "virality": r[4],
                "upvotes": r[5],
                "n_comments": r[6],
                "subreddit": r[7],
                "created_at": r[8],
            }
            for r in rows
        ]

    def count(self, symbol: Optional[str] = None) -> int:
        """Count rows in cache, optionally filtered by symbol."""
        with self._lock:
            if symbol:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM post_signals WHERE symbol = ?",
                    (symbol.upper(),),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM post_signals"
                ).fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# RedditSignalAggregator -- High-level interface
# ---------------------------------------------------------------------------

class RedditSignalAggregator:
    """
    Top-level interface combining PostAnalyzer and RedditSentimentCache.

    Accepts raw post data, processes it, stores in cache, and
    provides rolling sentiment signals per symbol.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        half_life_hours: float = 2.0,
    ) -> None:
        self._analyzer = PostAnalyzer()
        self._cache = RedditSentimentCache(
            db_path=db_path,
            half_life_hours=half_life_hours,
        )

    def ingest_posts(
        self,
        posts: List[Dict],
        symbol: str = "",
    ) -> int:
        """
        Ingest a batch of raw Reddit post dicts, analyze, and cache.
        Returns number of signals stored.
        """
        signals = self._analyzer.analyze_batch(posts, target_symbol=symbol)
        return self._cache.update(signals)

    def get_signal(
        self,
        symbol: str,
        window_hours: float = 4.0,
    ) -> float:
        """Get rolling EW sentiment signal for symbol."""
        return self._cache.get_rolling_sentiment(symbol, window_hours)

    def get_stats(self, symbol: str) -> Dict:
        """Get summary stats for symbol."""
        return self._cache.get_symbol_stats(symbol)

    def prune(self) -> int:
        """Prune old cache entries."""
        return self._cache.prune_old()

    def close(self) -> None:
        self._cache.close()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyzer = PostAnalyzer()
    cache = RedditSentimentCache()

    test_posts_raw = [
        {
            "title": "Bitcoin just hit 100k -- we're all gonna make it! Diamond hands paying off!",
            "selftext": "Been holding since 2020. Never selling. WAGMI",
            "score": 5000,
            "num_comments": 1200,
            "subreddit": "r/Bitcoin",
            "id": "post_001",
            "created_utc": datetime.now(timezone.utc).timestamp() - 1800,
        },
        {
            "title": "BTC GOING TO ZERO!! CRASH INCOMING!! SELL SELL SELL!!!",
            "selftext": "",
            "score": 100,
            "num_comments": 50,
            "subreddit": "r/CryptoCurrency",
            "id": "post_002",
            "created_utc": datetime.now(timezone.utc).timestamp() - 3600,
        },
        {
            "title": "Ethereum network upgrade scheduled -- major improvements to gas fees",
            "selftext": "The upcoming Dencun upgrade will dramatically reduce L2 fees",
            "score": 800,
            "num_comments": 320,
            "subreddit": "r/ethereum",
            "id": "post_003",
            "created_utc": datetime.now(timezone.utc).timestamp() - 900,
        },
        {
            "title": "Cautious on ETH -- regulatory uncertainty and competition from SOL",
            "selftext": "Not bearish per se, but the risk/reward isn't there right now",
            "score": 400,
            "num_comments": 180,
            "subreddit": "r/investing",
            "id": "post_004",
            "created_utc": datetime.now(timezone.utc).timestamp() - 7200,
        },
    ]

    signals_btc = analyzer.analyze_batch(test_posts_raw, target_symbol="BTC")
    signals_eth = analyzer.analyze_batch(test_posts_raw, target_symbol="ETH")

    cache.update(signals_btc + signals_eth)

    print("Reddit Monitor Test\n" + "=" * 50)
    for signal in signals_btc:
        print(f"\n[BTC] {signal.title[:60]}")
        print(f"  Sentiment: {signal.sentiment:+.3f} | Conf: {signal.confidence:.2f} | Virality: {signal.virality_score:.2f}")

    for symbol in ["BTC", "ETH"]:
        rolling = cache.get_rolling_sentiment(symbol, window_hours=4.0)
        stats = cache.get_symbol_stats(symbol)
        print(f"\nRolling sentiment [{symbol}, 4h]: {rolling:+.4f}")
        print(f"Stats: {stats}")

    print(f"\nCache row count: {cache.count()}")
    cache.close()
