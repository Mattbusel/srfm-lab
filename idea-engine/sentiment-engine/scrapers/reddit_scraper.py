"""
sentiment_engine/scrapers/reddit_scraper.py
===========================================
PRAW-based Reddit scraper targeting crypto subreddits.

Financial rationale
-------------------
Reddit retail sentiment has documented predictive power for short-term crypto
price moves (Caporale et al., 2021).  The r/CryptoCurrency daily thread in
particular concentrates high-signal discussion that frequently precedes 24-48h
price action.  We weight posts and comments by upvote score to differentiate
community-validated opinions from noise.

Scoring formula
---------------
    post_weight   = log(1 + max(0, score))          # upvotes minus downvotes
    comment_weight = log(1 + max(0, comment.score)) * depth_discount
    depth_discount = 0.7 ^ depth                    # top-level comments >  nested

If PRAW credentials are not configured, the scraper falls back to a rich mock
corpus so the NLP pipeline can run end-to-end.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target subreddits
# ---------------------------------------------------------------------------

TARGET_SUBREDDITS: list[str] = [
    "CryptoCurrency",
    "Bitcoin",
    "ethereum",
    "altcoin",
    "CryptoMarkets",
    "solana",
]

FETCH_LIMIT: int = 50   # posts per subreddit per cycle
COMMENT_DEPTH: int = 2  # levels of comments to traverse

# ---------------------------------------------------------------------------
# Mock corpus
# ---------------------------------------------------------------------------

_MOCK_POSTS: list[dict] = [
    {"title": "BTC just broke 100k resistance for the 3rd time — this time it holds?",
     "selftext": "Chart looks like a clear cup and handle. RSI not yet overbought. I'm long.",
     "score": 4200, "num_comments": 380, "subreddit": "CryptoCurrency",
     "comments": [
         {"body": "Cup and handle is textbook. Target is 115k based on the depth.", "score": 890},
         {"body": "Every time people say 'this time it holds' it dumps lol", "score": 420},
     ]},
    {"title": "Ethereum gas fees are down 90% since the upgrade — adoption is coming",
     "selftext": "L2 rollups plus reduced base fees = mass adoption finally viable.",
     "score": 3100, "num_comments": 210, "subreddit": "ethereum",
     "comments": [
         {"body": "ETH is becoming deflationary which is extremely bullish long-term.", "score": 750},
         {"body": "Still too complex for normal users. UX needs work.", "score": 310},
     ]},
    {"title": "Why I sold everything and moved to stablecoins",
     "selftext": "Too much uncertainty. Macro environment is bad. Risk off until clarity.",
     "score": 1800, "num_comments": 440, "subreddit": "CryptoCurrency",
     "comments": [
         {"body": "Smart. Dry powder is a position.", "score": 620},
         {"body": "This is literally the bottom signal we needed 😂", "score": 1100},
     ]},
    {"title": "SOL validators have 99.9% uptime for 6 months — narrative shift incoming",
     "selftext": "The outage FUD is dead. Solana is now more reliable than Ethereum mainnet.",
     "score": 2600, "num_comments": 185, "subreddit": "solana",
     "comments": [
         {"body": "Been bullish on SOL since $20. Not stopping now.", "score": 540},
     ]},
    {"title": "Altseason incoming? BTC dominance dropping fast",
     "selftext": "Historically when BTC.D drops below 50%, alts 2-5x in 60 days.",
     "score": 3800, "num_comments": 320, "subreddit": "altcoin",
     "comments": [
         {"body": "History doesn't repeat but it rhymes. Watch ETH/BTC ratio.", "score": 880},
         {"body": "Rug pull warning: most alts are scams.", "score": 290},
     ]},
    {"title": "Lost 80% of my portfolio in this bear market — lessons learned",
     "selftext": "Over-leveraged, ignored risk management, FOMO'd into shitcoins. Don't repeat my mistakes.",
     "score": 5100, "num_comments": 510, "subreddit": "CryptoCurrency",
     "comments": [
         {"body": "Sorry for your loss. This is the way — learn from it.", "score": 1200},
         {"body": "Never invest more than you can afford to lose. Written in blood.", "score": 980},
     ]},
    {"title": "Bitcoin mining difficulty just hit ATH — network health is strong",
     "selftext": "Hash rate at 600 EH/s. Miners are not capitulating. Bullish fundamentals.",
     "score": 2900, "num_comments": 165, "subreddit": "Bitcoin",
     "comments": [
         {"body": "Hash rate is one of the strongest on-chain signals. Very bullish.", "score": 670},
     ]},
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RedditPost:
    """
    Normalised Reddit post with engagement-weighted score.

    Attributes
    ----------
    post_id         : Reddit post ID (t3_xxxxx) or mock hash
    subreddit       : Subreddit name
    title           : Post title
    body            : Selftext body (may be empty for link posts)
    score           : Net upvote score
    num_comments    : Total comment count
    post_weight     : log(1 + max(0, score))
    comment_texts   : Flat list of (text, weight) tuples from scraped comments
    created_utc     : Post creation timestamp
    url             : Canonical URL
    """
    post_id:       str
    subreddit:     str
    title:         str
    body:          str
    score:         int
    num_comments:  int
    post_weight:   float
    comment_texts: list[tuple[str, float]]
    created_utc:   datetime
    url:           str = ""

    def all_text(self) -> str:
        """Concatenate title + body + all comment texts for NLP processing."""
        parts = [self.title, self.body]
        parts.extend(text for text, _ in self.comment_texts)
        return " ".join(p for p in parts if p)

    def combined_weight(self) -> float:
        """Sum of post_weight and all comment weights."""
        return self.post_weight + sum(w for _, w in self.comment_texts)


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class RedditScraper:
    """
    Fetches hot and new posts + top-level comments from crypto subreddits.

    Parameters
    ----------
    client_id     : PRAW OAuth2 client ID
    client_secret : PRAW OAuth2 client secret
    user_agent    : Identifies your app to Reddit's API
    fetch_limit   : Posts per subreddit per call
    comment_depth : How many comment nesting levels to retrieve
    mock_mode     : Skip live API; use mock corpus instead
    """

    def __init__(
        self,
        client_id:     str = os.environ.get("REDDIT_CLIENT_ID",     ""),
        client_secret: str = os.environ.get("REDDIT_CLIENT_SECRET", ""),
        user_agent:    str = "sentiment-engine/0.1 (IAE research bot)",
        fetch_limit:   int = FETCH_LIMIT,
        comment_depth: int = COMMENT_DEPTH,
        mock_mode:     bool = True,
    ) -> None:
        self.client_id     = client_id
        self.client_secret = client_secret
        self.user_agent    = user_agent
        self.fetch_limit   = fetch_limit
        self.comment_depth = comment_depth
        self.mock_mode     = mock_mode or not (client_id and client_secret)

        self._reddit: Optional[object] = None
        if not self.mock_mode:
            self._init_praw()

    def _init_praw(self) -> None:
        """Lazy-initialise PRAW client."""
        try:
            import praw  # type: ignore
            self._reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )
            logger.info("PRAW client initialised (read-only mode).")
        except ImportError:
            logger.warning("praw not installed; falling back to mock mode.")
            self.mock_mode = True

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fetch_all(
        self,
        subreddits: list[str] | None = None,
        sort: str = "hot",          # "hot" | "new" | "rising"
    ) -> list[RedditPost]:
        """
        Fetch posts from all target subreddits.

        Parameters
        ----------
        subreddits : override default subreddit list
        sort       : PRAW listing sort order

        Returns
        -------
        List of RedditPost sorted by descending combined_weight.
        """
        subs    = subreddits or TARGET_SUBREDDITS
        results: list[RedditPost] = []

        for sub in subs:
            try:
                posts = (
                    self._fetch_mock_subreddit(sub)
                    if self.mock_mode
                    else self._fetch_live_subreddit(sub, sort)
                )
                results.extend(posts)
            except Exception as exc:
                logger.error("Error fetching r/%s: %s", sub, exc, exc_info=True)

        results.sort(key=lambda p: p.combined_weight(), reverse=True)
        logger.info(
            "RedditScraper: fetched %d posts from %d subreddits.", len(results), len(subs)
        )
        return results

    # ------------------------------------------------------------------ #
    # Internal — mock                                                      #
    # ------------------------------------------------------------------ #

    def _fetch_mock_subreddit(self, subreddit: str) -> list[RedditPost]:
        """Return a filtered + perturbed sample from _MOCK_POSTS."""
        import hashlib

        posts_for_sub = [
            p for p in _MOCK_POSTS
            if p["subreddit"].lower() == subreddit.lower()
        ]
        # If no exact match, include a generic sample
        if not posts_for_sub:
            posts_for_sub = random.sample(_MOCK_POSTS, min(3, len(_MOCK_POSTS)))

        result = []
        now    = datetime.now(timezone.utc)
        for raw in posts_for_sub:
            score    = raw["score"] + random.randint(-200, 200)
            weight   = math.log1p(max(0, score))
            comments: list[tuple[str, float]] = []
            for depth, c in enumerate(raw.get("comments", [])):
                cw = math.log1p(max(0, c["score"])) * (0.7 ** depth)
                comments.append((c["body"], cw))

            pid = hashlib.sha256(
                (raw["title"] + subreddit + now.isoformat()).encode()
            ).hexdigest()[:12]

            result.append(RedditPost(
                post_id=f"t3_{pid}",
                subreddit=subreddit,
                title=raw["title"],
                body=raw.get("selftext", ""),
                score=score,
                num_comments=raw["num_comments"],
                post_weight=weight,
                comment_texts=comments,
                created_utc=now,
                url=f"https://www.reddit.com/r/{subreddit}/comments/{pid}/",
            ))
        return result

    # ------------------------------------------------------------------ #
    # Internal — live PRAW                                                 #
    # ------------------------------------------------------------------ #

    def _fetch_live_subreddit(self, subreddit: str, sort: str) -> list[RedditPost]:
        """Fetch live posts via PRAW."""
        import praw  # type: ignore

        reddit = self._reddit
        sub    = reddit.subreddit(subreddit)

        listing = {
            "hot":    sub.hot,
            "new":    sub.new,
            "rising": sub.rising,
        }.get(sort, sub.hot)

        result: list[RedditPost] = []
        for submission in listing(limit=self.fetch_limit):
            weight = math.log1p(max(0, submission.score))

            # Expand top-level comments
            submission.comments.replace_more(limit=0)
            comment_texts: list[tuple[str, float]] = []
            self._collect_comments(
                submission.comments.list(), comment_texts, depth=0, max_depth=self.comment_depth
            )

            ts = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
            result.append(RedditPost(
                post_id=f"t3_{submission.id}",
                subreddit=subreddit,
                title=submission.title,
                body=submission.selftext or "",
                score=submission.score,
                num_comments=submission.num_comments,
                post_weight=weight,
                comment_texts=comment_texts,
                created_utc=ts,
                url=f"https://www.reddit.com{submission.permalink}",
            ))
        return result

    def _collect_comments(
        self,
        comments: list,
        output:   list[tuple[str, float]],
        depth:    int,
        max_depth: int,
    ) -> None:
        """Recursively collect comment text + depth-discounted weights."""
        if depth > max_depth:
            return
        discount = 0.7 ** depth
        for comment in comments:
            # Skip MoreComments objects
            if not hasattr(comment, "body"):
                continue
            w = math.log1p(max(0, comment.score)) * discount
            output.append((comment.body, w))
            if hasattr(comment, "replies") and depth < max_depth:
                self._collect_comments(comment.replies, output, depth + 1, max_depth)
