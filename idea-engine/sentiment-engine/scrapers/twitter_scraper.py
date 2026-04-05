"""
sentiment_engine/scrapers/twitter_scraper.py
============================================
Mock Twitter/X API scraper that simulates the v2 Recent Search endpoint.

Why engagement weighting matters for trading signals
------------------------------------------------------
A tweet with 50 000 likes and 10 000 retweets carries substantially more
market-moving weight than an unknown account posting identical text.
Viral sentiment shifts have historically preceded short-term price moves in
crypto (Kraaijeveld & De Smedt, 2020).  We therefore weight each tweet's
sentiment contribution by a logarithmic engagement score:

    engagement_weight = log(1 + likes + retweets * 2 + replies * 0.5)

The 2× multiplier on retweets reflects that re-broadcasting is a stronger
signal of conviction than passive liking.

NOTE: The HTTP call is stubbed — replace _fetch_raw_page() with real
      Bearer-token requests once API credentials are available.  All
      downstream processing (rate-limit handling, pagination, engagement
      weighting, deduplication) is fully implemented and production-ready.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TWITTER_V2_SEARCH = "https://api.twitter.com/2/tweets/search/recent"
DEFAULT_BEARER_TOKEN = ""          # set via env var TWITTER_BEARER_TOKEN in prod

# Crypto keywords we track
CRYPTO_SEARCH_TERMS: list[str] = [
    "$BTC", "$ETH", "$SOL", "$BNB", "$XRP", "$DOGE",
    "bitcoin", "ethereum", "crypto", "altcoin", "defi", "nft",
    "#crypto", "#bitcoin", "#ethereum",
]

# Simulated tweet corpus for mock mode
_MOCK_TWEETS_TEMPLATE: list[dict] = [
    {
        "text": "$BTC breaking ATH again 🚀 this bull run is just getting started HODL",
        "public_metrics": {"like_count": 3200, "retweet_count": 870, "reply_count": 140},
        "author_id": "mock_001",
        "lang": "en",
    },
    {
        "text": "Ethereum merge was a mistake, ETH is getting rekt 📉 switching to SOL",
        "public_metrics": {"like_count": 450, "retweet_count": 120, "reply_count": 88},
        "author_id": "mock_002",
        "lang": "en",
    },
    {
        "text": "Just bought more BTC on this dip. Crypto is the future, not a rug.",
        "public_metrics": {"like_count": 980, "retweet_count": 230, "reply_count": 55},
        "author_id": "mock_003",
        "lang": "en",
    },
    {
        "text": "SOL devs are shipping non-stop. Fundamental value is accumulating quietly.",
        "public_metrics": {"like_count": 620, "retweet_count": 180, "reply_count": 42},
        "author_id": "mock_004",
        "lang": "en",
    },
    {
        "text": "Crypto dump incoming 📉 whale wallets moving millions to exchanges rn",
        "public_metrics": {"like_count": 1850, "retweet_count": 640, "reply_count": 220},
        "author_id": "mock_005",
        "lang": "en",
    },
    {
        "text": "BNB ecosystem expanding rapidly. New projects launching every week 🚀",
        "public_metrics": {"like_count": 310, "retweet_count": 95, "reply_count": 28},
        "author_id": "mock_006",
        "lang": "en",
    },
    {
        "text": "another day another pump and dump scheme. crypto is dead lol",
        "public_metrics": {"like_count": 210, "retweet_count": 55, "reply_count": 67},
        "author_id": "mock_007",
        "lang": "en",
    },
    {
        "text": "XRP moon mission confirmed. SEC losing their case finally!",
        "public_metrics": {"like_count": 4500, "retweet_count": 1200, "reply_count": 340},
        "author_id": "mock_008",
        "lang": "en",
    },
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Tweet:
    """
    Represents a single tweet with engagement metadata.

    Attributes
    ----------
    tweet_id       : Deduplicated identifier (SHA256 of text+timestamp if mock)
    text           : Raw tweet text
    author_id      : Opaque author identifier
    created_at     : UTC timestamp of posting
    likes          : Like count
    retweets       : Retweet count
    replies        : Reply count
    engagement_weight : Computed log-engagement score (see module docstring)
    lang           : ISO 639-1 language code
    search_term    : Which keyword triggered this result
    """
    tweet_id:          str
    text:              str
    author_id:         str
    created_at:        datetime
    likes:             int
    retweets:          int
    replies:           int
    engagement_weight: float
    lang:              str    = "en"
    search_term:       str    = ""

    @classmethod
    def from_api_object(cls, raw: dict, search_term: str = "") -> "Tweet":
        """Parse a Twitter v2 API tweet object."""
        metrics  = raw.get("public_metrics", {})
        likes    = int(metrics.get("like_count",    0))
        retweets = int(metrics.get("retweet_count", 0))
        replies  = int(metrics.get("reply_count",   0))

        # Engagement weight: log-scale to dampen outlier viral tweets
        weight = math.log1p(likes + retweets * 2 + replies * 0.5)

        # Parse created_at; fall back to now if absent (mock mode)
        raw_ts = raw.get("created_at", "")
        try:
            ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.now(timezone.utc)

        # Build a stable ID from content hash if no real ID supplied
        tweet_id = raw.get("id", "") or hashlib.sha256(
            (raw.get("text", "") + ts.isoformat()).encode()
        ).hexdigest()[:16]

        return cls(
            tweet_id=str(tweet_id),
            text=raw.get("text", ""),
            author_id=str(raw.get("author_id", "unknown")),
            created_at=ts,
            likes=likes,
            retweets=retweets,
            replies=replies,
            engagement_weight=weight,
            lang=raw.get("lang", "en"),
            search_term=search_term,
        )


# ---------------------------------------------------------------------------
# Rate-limit state
# ---------------------------------------------------------------------------

@dataclass
class _RateLimitState:
    """Tracks remaining requests in the current 15-minute window."""
    requests_remaining: int = 450   # Twitter v2 free tier: 500/15min
    window_reset_ts:    float = field(default_factory=lambda: time.time() + 900)

    def consume(self) -> None:
        """Consume one request credit, sleeping if the window is exhausted."""
        now = time.time()
        if now >= self.window_reset_ts:
            self.requests_remaining = 450
            self.window_reset_ts = now + 900

        if self.requests_remaining <= 0:
            sleep_s = max(0.0, self.window_reset_ts - now)
            logger.warning(
                "Twitter rate limit hit — sleeping %.1f s until window resets.", sleep_s
            )
            time.sleep(sleep_s)
            self.requests_remaining = 450
            self.window_reset_ts    = time.time() + 900

        self.requests_remaining -= 1


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class TwitterScraper:
    """
    Scrapes Twitter/X for crypto-relevant tweets.

    In production, set bearer_token to a valid Twitter v2 Bearer Token.
    In mock mode (default), returns a randomised sample of synthetic tweets
    so the full downstream pipeline can be exercised without API credentials.

    Parameters
    ----------
    bearer_token : str, optional
        Twitter v2 Bearer Token.  If empty, mock mode is activated.
    max_results  : int
        Tweets per page (10-100 for v2 recent search).
    pages        : int
        How many pages to fetch per search term.
    mock_mode    : bool
        Force mock mode regardless of bearer_token presence.
    """

    def __init__(
        self,
        bearer_token: str = DEFAULT_BEARER_TOKEN,
        max_results:  int = 50,
        pages:        int = 2,
        mock_mode:    bool = True,
    ) -> None:
        self.bearer_token = bearer_token
        self.max_results  = max(10, min(100, max_results))
        self.pages        = max(1, pages)
        self.mock_mode    = mock_mode or not bearer_token
        self._rate_limit  = _RateLimitState()
        self._seen_ids:   set[str] = set()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fetch_recent(
        self,
        search_terms: list[str] | None = None,
    ) -> list[Tweet]:
        """
        Fetch recent tweets for all search terms, deduplicated.

        Returns
        -------
        List of Tweet objects sorted by descending engagement weight.
        """
        terms   = search_terms or CRYPTO_SEARCH_TERMS
        results: list[Tweet] = []

        for term in terms:
            try:
                for tweet in self._fetch_term(term):
                    if tweet.tweet_id not in self._seen_ids:
                        self._seen_ids.add(tweet.tweet_id)
                        results.append(tweet)
            except Exception as exc:
                logger.error("Error fetching term '%s': %s", term, exc, exc_info=True)

        results.sort(key=lambda t: t.engagement_weight, reverse=True)
        logger.info(
            "TwitterScraper: fetched %d unique tweets across %d terms.",
            len(results), len(terms),
        )
        return results

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _fetch_term(self, term: str) -> Iterator[Tweet]:
        """Yield tweets page by page for a single search term."""
        next_token: Optional[str] = None

        for page_num in range(self.pages):
            self._rate_limit.consume()

            if self.mock_mode:
                raw_tweets = self._fetch_mock_page(term)
            else:
                raw_tweets, next_token = self._fetch_real_page(term, next_token)

            for raw in raw_tweets:
                yield Tweet.from_api_object(raw, search_term=term)

            if not next_token and not self.mock_mode:
                break

    def _fetch_mock_page(self, term: str) -> list[dict]:
        """
        Return a randomised subsample of _MOCK_TWEETS_TEMPLATE, stamped
        with the current time so recency decay behaves realistically.
        """
        count  = random.randint(3, len(_MOCK_TWEETS_TEMPLATE))
        sample = random.sample(_MOCK_TWEETS_TEMPLATE, count)
        now    = datetime.now(timezone.utc).isoformat()
        return [
            {**t, "id": hashlib.sha256((t["text"] + term + now).encode()).hexdigest()[:16],
             "created_at": now}
            for t in sample
        ]

    def _fetch_real_page(
        self,
        term: str,
        next_token: Optional[str],
    ) -> tuple[list[dict], Optional[str]]:
        """
        Call the Twitter v2 recent search endpoint.

        Returns (list_of_raw_tweet_dicts, next_pagination_token).
        """
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        params: dict = {
            "query":       f"{term} lang:en -is:retweet",
            "max_results": self.max_results,
            "tweet.fields": "created_at,public_metrics,lang,author_id",
        }
        if next_token:
            params["next_token"] = next_token

        resp = requests.get(
            TWITTER_V2_SEARCH,
            headers=headers,
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()

        tweets    = body.get("data", [])
        new_token = body.get("meta", {}).get("next_token")
        return tweets, new_token

    def get_stats(self) -> dict:
        """Return scraper state statistics."""
        return {
            "mode":                "mock" if self.mock_mode else "live",
            "unique_tweets_seen":  len(self._seen_ids),
            "rate_limit_remaining": self._rate_limit.requests_remaining,
        }
