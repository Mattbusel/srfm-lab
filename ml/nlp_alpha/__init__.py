"""
NLP Alpha package -- statistical NLP signal generation for crypto and equities.

Modules:
    sentiment_analyzer -- Financial lexicon, headline scoring, EW aggregation
    event_extractor    -- Rule-based event extraction (hack, listing, earnings, etc.)
    reddit_monitor     -- Reddit post analysis and SQLite-backed sentiment cache
    fear_greed         -- Crypto Fear & Greed Index computation
    alpha_combiner     -- Composite NLP alpha signal combining all subsystems
    pipeline           -- Full NLP alpha pipeline (existing, FinBERT-based)
"""

from .pipeline import NLPAlphaPipeline, PipelineConfig, PipelineResult
from .sentiment_analyzer import (
    FinancialLexicon,
    HeadlineScorer,
    NewsFeedAggregator,
    SentimentScore,
)
from .event_extractor import (
    EventType,
    EventExtractor,
    EventSignalGenerator,
    ExtractedEvent,
)
from .reddit_monitor import (
    PostAnalyzer,
    PostSignal,
    RedditSentimentCache,
    SUBREDDIT_CONFIGS,
)
from .fear_greed import FearGreedIndex, FearGreedResult
from .alpha_combiner import NLPAlphaModule, NLPSignal, make_nlp_module
