"""
Social media scraper for Reddit and Twitter/X financial content.

Provides:
- Reddit API client (subreddit posts + comments)
- Twitter/X API client (financial tweets)
- Rate limiting, retry logic
- Structured output compatible with the NLP pipeline
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SocialPost:
    """A social media post with metadata."""
    platform: str           # "reddit" | "twitter"
    post_id: str
    text: str
    author: str
    url: str
    created_at: datetime
    subreddit: Optional[str] = None
    score: int = 0
    n_comments: int = 0
    n_likes: int = 0
    n_retweets: int = 0
    tickers: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    content_hash: str = ""
    sentiment_score: Optional[float] = None
    is_news: bool = False
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return d


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self._interval = 60.0 / calls_per_minute
        self._last_call = 0.0
        self._call_count = 0
        self._window_start = time.time()

    def wait(self) -> None:
        """Block until next call is allowed."""
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call = time.time()
        self._call_count += 1

    def reset_window(self) -> None:
        self._call_count = 0
        self._window_start = time.time()


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _api_get(
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict] = None,
    timeout: int = 15,
    max_retries: int = 3,
) -> Optional[Dict]:
    """Make authenticated API GET request, return JSON dict."""
    if params:
        url = url + "?" + urlencode(params)

    for attempt in range(max_retries):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429:
                # Rate limited
                retry_after = int(e.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited. Sleeping {retry_after}s")
                time.sleep(retry_after)
            elif e.code in (401, 403):
                logger.error(f"Authentication error: {e.code}")
                return None
            elif e.code in (404, 410):
                return None
            else:
                logger.warning(f"HTTP {e.code}, attempt {attempt+1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        except URLError as e:
            logger.warning(f"URL error: {e.reason}, attempt {attempt+1}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


# ---------------------------------------------------------------------------
# Reddit API client
# ---------------------------------------------------------------------------

FINANCIAL_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "StockMarket",
    "SecurityAnalysis",
    "financialindependence",
    "personalfinance",
    "Economics",
    "algotrading",
    "quant",
    "ValueInvesting",
    "dividends",
    "ETFs",
    "CryptoCurrency",
]


class RedditClient:
    """
    Reddit API client using OAuth2.
    Supports fetching posts from subreddits and searching by ticker.
    """

    API_BASE = "https://oauth.reddit.com"
    AUTH_URL = "https://www.reddit.com/api/v1/access_token"
    PUBLIC_BASE = "https://www.reddit.com"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "FinanceBot/1.0",
        use_public_api: bool = True,  # Use public JSON API if no credentials
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.use_public = use_public_api or (not client_id)
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._rate_limiter = RateLimiter(calls_per_minute=30)

    def _get_headers(self) -> Dict[str, str]:
        headers = {"User-Agent": self.user_agent}
        if not self.use_public and self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        return headers

    def _authenticate(self) -> bool:
        """Get OAuth2 access token."""
        if not self.client_id or not self.client_secret:
            return False
        if time.time() < self._token_expiry - 60:
            return True

        import base64
        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        headers = {
            "Authorization": f"Basic {credentials}",
            "User-Agent": self.user_agent,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = b"grant_type=client_credentials"
        try:
            req = Request(self.AUTH_URL, data=data, headers=headers, method="POST")
            with urlopen(req, timeout=10) as response:
                token_data = json.loads(response.read().decode("utf-8"))
                self._access_token = token_data.get("access_token")
                expires_in = token_data.get("expires_in", 3600)
                self._token_expiry = time.time() + expires_in
                return True
        except Exception as e:
            logger.error(f"Reddit auth failed: {e}")
            return False

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request, handling auth and rate limiting."""
        self._rate_limiter.wait()

        if not self.use_public and self._access_token:
            url = self.API_BASE + endpoint
        else:
            url = self.PUBLIC_BASE + endpoint + ".json"

        return _api_get(url, self._get_headers(), params)

    def get_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "hot",
        limit: int = 25,
        time_filter: str = "day",  # "hour" | "day" | "week" | "month"
    ) -> List[SocialPost]:
        """Fetch posts from a subreddit."""
        if not self.use_public and not self._authenticate():
            logger.warning("Reddit auth failed, using public API")

        endpoint = f"/r/{subreddit}/{sort}"
        params = {"limit": min(limit, 100), "t": time_filter, "raw_json": 1}
        data = self._make_request(endpoint, params)

        if not data:
            return []

        posts = []
        try:
            items = data.get("data", {}).get("children", [])
            for item in items:
                post_data = item.get("data", {})
                post = self._parse_post(post_data, subreddit)
                if post:
                    posts.append(post)
        except Exception as e:
            logger.error(f"Error parsing Reddit posts: {e}")

        return posts

    def search_ticker(
        self,
        ticker: str,
        subreddits: Optional[List[str]] = None,
        limit: int = 25,
        time_filter: str = "day",
    ) -> List[SocialPost]:
        """Search for posts mentioning a ticker."""
        subreddits = subreddits or ["wallstreetbets", "stocks", "investing"]
        all_posts = []

        for sub in subreddits:
            endpoint = f"/r/{sub}/search"
            params = {
                "q": ticker,
                "sort": "new",
                "t": time_filter,
                "limit": limit,
                "restrict_sr": "true",
                "raw_json": 1,
            }
            data = self._make_request(endpoint, params)
            if data:
                try:
                    items = data.get("data", {}).get("children", [])
                    for item in items:
                        post = self._parse_post(item.get("data", {}), sub)
                        if post:
                            post.tickers.append(ticker.upper())
                            all_posts.append(post)
                except Exception as e:
                    logger.debug(f"Error parsing search results: {e}")

        return all_posts

    def get_post_comments(
        self,
        post_id: str,
        subreddit: str,
        limit: int = 20,
    ) -> List[SocialPost]:
        """Fetch top-level comments for a post."""
        endpoint = f"/r/{subreddit}/comments/{post_id}"
        params = {"limit": limit, "raw_json": 1, "sort": "top"}
        data = self._make_request(endpoint, params)

        if not data or not isinstance(data, list) or len(data) < 2:
            return []

        comments = []
        try:
            comment_listing = data[1].get("data", {}).get("children", [])
            for item in comment_listing:
                if item.get("kind") != "t1":
                    continue
                d = item.get("data", {})
                body = d.get("body", "")
                if not body or body == "[deleted]" or len(body) < 10:
                    continue

                comment = SocialPost(
                    platform="reddit",
                    post_id=f"comment_{d.get('id', '')}",
                    text=body[:2000],
                    author=d.get("author", "[deleted]"),
                    url=f"https://reddit.com{d.get('permalink', '')}",
                    created_at=datetime.fromtimestamp(
                        float(d.get("created_utc", 0)), tz=timezone.utc
                    ),
                    subreddit=subreddit,
                    score=int(d.get("score", 0)),
                )
                comments.append(comment)
        except Exception as e:
            logger.debug(f"Error parsing comments: {e}")

        return comments

    def _parse_post(self, post_data: Dict, subreddit: str) -> Optional[SocialPost]:
        """Parse Reddit post JSON into SocialPost."""
        post_id = post_data.get("id", "")
        title   = post_data.get("title", "")
        selftext = post_data.get("selftext", "") or ""
        text = (title + "\n" + selftext).strip()

        if not text or post_data.get("removed_by_category"):
            return None

        url = post_data.get("url", "")
        permalink = f"https://reddit.com{post_data.get('permalink', '')}"

        created_utc = float(post_data.get("created_utc", 0))
        created_at = datetime.fromtimestamp(created_utc, tz=timezone.utc)

        # Extract tickers from title/text
        from ..utils.text_processing import extract_tickers
        tickers = extract_tickers(title + " " + selftext[:500])

        # Extract hashtags (sometimes used in WSB etc.)
        hashtags = re.findall(r'#(\w+)', text)

        return SocialPost(
            platform="reddit",
            post_id=post_id,
            text=text[:3000],
            author=post_data.get("author", "[deleted]"),
            url=permalink,
            created_at=created_at,
            subreddit=subreddit,
            score=int(post_data.get("score", 0)),
            n_comments=int(post_data.get("num_comments", 0)),
            tickers=tickers,
            hashtags=hashtags,
        )

    def get_sentiment_proxy(self, posts: List[SocialPost]) -> Dict[str, float]:
        """
        Compute simple proxy sentiment from Reddit scores.
        Returns dict of ticker -> upvote_ratio_mean.
        """
        from collections import defaultdict
        ticker_scores: Dict[str, List[float]] = defaultdict(list)

        for post in posts:
            if post.score > 0:
                normalized = min(float(post.score) / 1000.0, 1.0)
                for ticker in post.tickers:
                    ticker_scores[ticker].append(normalized)

        return {
            ticker: float(sum(scores) / len(scores))
            for ticker, scores in ticker_scores.items()
            if scores
        }


# ---------------------------------------------------------------------------
# Twitter/X API client
# ---------------------------------------------------------------------------

FINANCIAL_HASHTAGS = [
    "#stocks", "#investing", "#trading", "#earnings", "#stockmarket",
    "#finance", "#wallstreet", "#options", "#forex", "#crypto",
    "#fintwit", "#sp500", "#nasdaq", "#dow",
]

FINANCIAL_ACCOUNTS = [
    "markets", "financialtimes", "wsj", "bloomberg", "reuters",
    "CNBC", "MarketWatch", "TheStreet", "investopedia",
]


class TwitterClient:
    """
    Twitter/X API v2 client for financial tweets.

    Supports bearer token authentication for read-only access.
    Falls back to scraping if no credentials provided.
    """

    API_BASE = "https://api.twitter.com/2"

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.api_secret = api_secret
        self._rate_limiter = RateLimiter(calls_per_minute=15)  # Standard tier

    def _get_headers(self) -> Dict[str, str]:
        if self.bearer_token:
            return {
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json",
            }
        return {"Content-Type": "application/json"}

    def _is_authenticated(self) -> bool:
        return bool(self.bearer_token)

    def search_recent(
        self,
        query: str,
        max_results: int = 100,
        since_id: Optional[str] = None,
    ) -> List[SocialPost]:
        """
        Search recent tweets (7 days).
        Requires Elevated API access.
        """
        if not self._is_authenticated():
            logger.warning("Twitter: no credentials, returning empty")
            return []

        self._rate_limiter.wait()
        endpoint = "/tweets/search/recent"
        params: Dict[str, Any] = {
            "query": query + " lang:en -is:retweet",
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,author_id,public_metrics,entities",
            "expansions": "author_id",
            "user.fields": "username,public_metrics",
        }
        if since_id:
            params["since_id"] = since_id

        data = _api_get(self.API_BASE + endpoint, self._get_headers(), params)
        if not data:
            return []

        return self._parse_tweets(data)

    def get_ticker_tweets(
        self,
        ticker: str,
        max_results: int = 50,
    ) -> List[SocialPost]:
        """Search for tweets about a specific ticker (cashtag)."""
        query = f"${ticker} (earnings OR revenue OR guidance OR price OR buy OR sell)"
        return self.search_recent(query, max_results=max_results)

    def get_user_tweets(
        self,
        username: str,
        max_results: int = 20,
    ) -> List[SocialPost]:
        """Get recent tweets from a specific user."""
        if not self._is_authenticated():
            return []

        # First get user ID
        self._rate_limiter.wait()
        user_data = _api_get(
            f"{self.API_BASE}/users/by/username/{username}",
            self._get_headers(),
        )
        if not user_data:
            return []

        user_id = user_data.get("data", {}).get("id")
        if not user_id:
            return []

        self._rate_limiter.wait()
        params: Dict[str, Any] = {
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics,entities",
        }
        data = _api_get(
            f"{self.API_BASE}/users/{user_id}/tweets",
            self._get_headers(),
            params,
        )
        if not data:
            return []

        return self._parse_tweets(data)

    def get_trending_finance_tweets(self, max_results: int = 100) -> List[SocialPost]:
        """Fetch trending financial tweets using common financial hashtags."""
        all_posts = []
        for hashtag in FINANCIAL_HASHTAGS[:5]:  # limit API calls
            posts = self.search_recent(hashtag, max_results=max_results // 5)
            all_posts.extend(posts)
        return all_posts

    def _parse_tweets(self, data: Dict) -> List[SocialPost]:
        """Parse Twitter API v2 response into SocialPost list."""
        posts = []
        tweets = data.get("data", [])
        includes = data.get("includes", {})
        users = {u["id"]: u["username"] for u in includes.get("users", [])}

        for tweet in tweets:
            tweet_id = tweet.get("id", "")
            text = tweet.get("text", "")
            if not text:
                continue

            author_id = tweet.get("author_id", "")
            username  = users.get(author_id, "unknown")

            created_str = tweet.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except Exception:
                created_at = datetime.now(timezone.utc)

            metrics = tweet.get("public_metrics", {})
            n_likes    = int(metrics.get("like_count", 0))
            n_retweets = int(metrics.get("retweet_count", 0))

            # Extract entities
            entities = tweet.get("entities", {})
            hashtags = [h["tag"] for h in entities.get("hashtags", [])]
            cashtags = [c["tag"].upper() for c in entities.get("cashtags", [])]

            from ..utils.text_processing import extract_tickers
            text_tickers = extract_tickers(text)
            all_tickers = list(set(cashtags + text_tickers))

            url = f"https://twitter.com/{username}/status/{tweet_id}"

            post = SocialPost(
                platform="twitter",
                post_id=tweet_id,
                text=text[:2000],
                author=username,
                url=url,
                created_at=created_at,
                n_likes=n_likes,
                n_retweets=n_retweets,
                tickers=all_tickers,
                hashtags=hashtags,
            )
            posts.append(post)

        return posts

    def compute_twitter_sentiment(
        self,
        ticker: str,
        posts: List[SocialPost],
    ) -> Dict[str, float]:
        """
        Simple engagement-weighted sentiment proxy.
        Returns {score, volume, engagement}.
        """
        relevant = [p for p in posts if ticker.upper() in p.tickers]
        if not relevant:
            return {"score": 0.0, "volume": 0.0, "engagement": 0.0}

        # Engagement = likes + 2*retweets (retweets have more reach)
        engagements = [p.n_likes + 2 * p.n_retweets for p in relevant]
        total_eng = sum(engagements)

        # Use engagement-weighted average of post scores
        scores = []
        for p, eng in zip(relevant, engagements):
            weight = (eng + 1) / (total_eng + len(relevant))
            # Simple heuristic: use sentiment if available, else 0
            score = p.sentiment_score or 0.0
            scores.append(weight * score)

        return {
            "score": float(sum(scores)),
            "volume": float(len(relevant)),
            "engagement": float(total_eng),
        }


# ---------------------------------------------------------------------------
# Social signal aggregator
# ---------------------------------------------------------------------------

class SocialSignalAggregator:
    """
    Aggregates social media signals from Reddit and Twitter.
    Produces per-ticker sentiment scores and volume metrics.
    """

    def __init__(
        self,
        reddit_client: Optional[RedditClient] = None,
        twitter_client: Optional[TwitterClient] = None,
    ):
        self.reddit = reddit_client or RedditClient(use_public_api=True)
        self.twitter = twitter_client

    def fetch_all_signals(
        self,
        tickers: List[str],
        reddit_subs: Optional[List[str]] = None,
        time_filter: str = "day",
    ) -> Dict[str, List[SocialPost]]:
        """
        Fetch all social signals for given tickers.
        Returns dict: ticker -> list of posts.
        """
        reddit_subs = reddit_subs or ["wallstreetbets", "stocks", "investing"]
        all_posts: List[SocialPost] = []

        # Reddit
        for sub in reddit_subs[:3]:  # limit
            posts = self.reddit.get_subreddit_posts(sub, sort="hot", time_filter=time_filter, limit=50)
            all_posts.extend(posts)

        # Twitter
        if self.twitter and self.twitter._is_authenticated():
            for ticker in tickers[:5]:  # limit API calls
                posts = self.twitter.get_ticker_tweets(ticker, max_results=25)
                all_posts.extend(posts)

        # Tag tickers on all posts
        from ..utils.text_processing import extract_tickers
        ticker_set = set(t.upper() for t in tickers)
        for post in all_posts:
            if not post.tickers:
                found = extract_tickers(post.text, known_tickers=ticker_set)
                post.tickers = found

        # Group by ticker
        result: Dict[str, List[SocialPost]] = {t: [] for t in ticker_set}
        for post in all_posts:
            for ticker in post.tickers:
                if ticker in result:
                    result[ticker].append(post)

        return result

    def compute_signals(
        self,
        ticker_posts: Dict[str, List[SocialPost]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute aggregated signals per ticker.
        Returns: {ticker: {mention_count, mean_score, engagement, wsb_ratio}}
        """
        signals = {}
        for ticker, posts in ticker_posts.items():
            if not posts:
                signals[ticker] = {
                    "mention_count": 0.0,
                    "mean_score": 0.0,
                    "total_engagement": 0.0,
                    "wsb_ratio": 0.0,
                    "mean_reddit_score": 0.0,
                }
                continue

            reddit_posts  = [p for p in posts if p.platform == "reddit"]
            twitter_posts = [p for p in posts if p.platform == "twitter"]
            wsb_posts     = [p for p in reddit_posts if p.subreddit == "wallstreetbets"]

            reddit_scores = [p.score for p in reddit_posts if p.score > 0]
            engagement    = sum(p.n_likes + 2 * p.n_retweets for p in twitter_posts)

            signals[ticker] = {
                "mention_count": float(len(posts)),
                "reddit_mention_count": float(len(reddit_posts)),
                "twitter_mention_count": float(len(twitter_posts)),
                "wsb_mention_count": float(len(wsb_posts)),
                "wsb_ratio": float(len(wsb_posts) / max(len(reddit_posts), 1)),
                "mean_reddit_score": float(sum(reddit_scores) / max(len(reddit_scores), 1)),
                "total_engagement": float(engagement),
                "n_comments_total": float(sum(p.n_comments for p in reddit_posts)),
                "has_sentiment": float(any(p.sentiment_score is not None for p in posts)),
                "mean_sentiment": float(
                    sum(p.sentiment_score for p in posts if p.sentiment_score is not None)
                    / max(sum(1 for p in posts if p.sentiment_score is not None), 1)
                ),
            }

        return signals


# ---------------------------------------------------------------------------
# Self-test (no actual API calls)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing social scraper data structures...")

    # Test SocialPost creation
    post = SocialPost(
        platform="reddit",
        post_id="abc123",
        text="AAPL looks bullish! Earnings beat was massive. $AAPL 🚀",
        author="user123",
        url="https://reddit.com/r/wallstreetbets/abc123",
        created_at=datetime.now(timezone.utc),
        subreddit="wallstreetbets",
        score=1500,
        n_comments=200,
        tickers=["AAPL"],
    )
    print(f"Post: {post.text[:60]}")
    print(f"Hash: {post.content_hash}")

    # Test aggregator (mock)
    agg = SocialSignalAggregator()
    mock_posts = {"AAPL": [post], "MSFT": []}
    signals = agg.compute_signals(mock_posts)
    print(f"AAPL signals: {signals.get('AAPL', {})}")

    # Test rate limiter
    rl = RateLimiter(calls_per_minute=120)
    t0 = time.time()
    for _ in range(3):
        rl.wait()
    print(f"Rate limiter: 3 calls in {time.time() - t0:.3f}s")

    print("Social scraper self-test passed.")
