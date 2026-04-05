"""
sentiment-engine — Component of the Idea Automation Engine (IAE)
=================================================================
Full NLP sentiment pipeline that ingests social media, news, and crowd-fear
data to produce per-symbol sentiment signals for the crypto trading strategy.

Architecture
------------
1. Scrapers  — collect raw text from Twitter/X, Reddit, RSS news feeds, and
               the Alternative.me Fear & Greed Index.
2. NLP layer — tokenise crypto-domain text, score sentiment via VADER + TextBlob
               ensemble, extract which symbols are being discussed.
3. Aggregator — roll up raw scores into a SentimentSignal per symbol, stored in
                SQLite via idea_engine.db.
4. Signal bridge — convert SentimentSignal thresholds into IAE hypothesis rows.
5. Scheduler — async 15-minute scrape loop.

Public API
----------
    from sentiment_engine import SentimentAggregator, SignalBridge, SentimentScheduler
    from sentiment_engine.scrapers import TwitterScraper, RedditScraper, NewsScraper, FearGreedClient
    from sentiment_engine.nlp import CryptoTokenizer, SentimentScorer, SymbolExtractor

Typical usage::

    agg = SentimentAggregator(db_path="idea_engine.db")
    signals = agg.run_cycle()
    bridge  = SignalBridge(db_path="idea_engine.db")
    hypotheses = bridge.convert(signals)
"""

from .aggregator    import SentimentAggregator, SentimentSignal
from .signal_bridge import SignalBridge
from .scheduler     import SentimentScheduler

__all__ = [
    "SentimentAggregator",
    "SentimentSignal",
    "SignalBridge",
    "SentimentScheduler",
]

__version__ = "0.1.0"
