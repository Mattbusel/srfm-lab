"""scrapers — data collection layer for the sentiment engine."""

from .twitter_scraper import TwitterScraper, Tweet
from .reddit_scraper  import RedditScraper, RedditPost
from .news_scraper    import NewsScraper, NewsItem
from .fear_greed      import FearGreedClient, FearGreedReading

__all__ = [
    "TwitterScraper", "Tweet",
    "RedditScraper",  "RedditPost",
    "NewsScraper",    "NewsItem",
    "FearGreedClient","FearGreedReading",
]
