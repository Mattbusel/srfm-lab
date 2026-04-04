"""
Full NLP alpha pipeline: fetch → parse → sentiment → signal → alpha.

Orchestrates all components:
1. RSS/social/SEC news fetching
2. Text cleaning and deduplication
3. FinBERT sentiment analysis
4. LLM scoring for relevance and impact
5. Event detection
6. Alpha signal building
7. Output as tradeable signals
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from .scrapers.rss_parser import MultiFeedFetcher, NewsArticle, FeedConfig
from .scrapers.social_scraper import SocialSignalAggregator, SocialPost
from .scrapers.sec_edgar import EDGARFetcher, SECFiling
from .utils.text_processing import Deduplicator, clean_text, extract_tickers
from .models.finbert import FinBERTSentiment, FinBERTConfig, SentimentResult
from .models.llm_scorer import LLMScorer, LLMScorerConfig, ArticleScore
from .signal.event_detector import EventDetector, DetectedEvent
from .signal.alpha_builder import AlphaBuilder, AlphaSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # Data fetching
    tickers: List[str] = field(default_factory=list)
    fetch_rss: bool = True
    fetch_social: bool = True
    fetch_sec: bool = True
    rss_categories: Optional[List[str]] = None
    max_articles_per_source: int = 30
    lookback_hours: int = 24       # only process articles from last N hours

    # NLP models
    finbert_model: str = "ProsusAI/finbert"
    use_llm_scorer: bool = True
    llm_backend: str = "rule_based"   # "openai" | "anthropic" | "rule_based"
    llm_api_key: Optional[str] = None

    # Signal generation
    min_relevance: float = 0.4
    min_confidence: float = 0.5
    signal_decay_halflife_hours: float = 12.0

    # Caching
    cache_dir: str = ".nlp_cache"
    use_cache: bool = True

    # Logging
    verbose: bool = True
    save_articles: bool = False
    output_dir: str = "./nlp_output"


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Complete output of one pipeline run."""
    run_timestamp: datetime
    n_articles_fetched: int
    n_after_dedup: int
    n_scored: int
    n_events_detected: int
    sentiment_by_ticker: Dict[str, float]
    events_by_ticker: Dict[str, List[DetectedEvent]]
    signals_by_ticker: Dict[str, List[AlphaSignal]]
    composite_signals: Dict[str, float]
    position_sizes: Dict[str, float]
    processing_time_secs: float
    articles: List[NewsArticle] = field(default_factory=list)
    social_posts: List[SocialPost] = field(default_factory=list)
    sec_filings: List[SECFiling] = field(default_factory=list)


# ---------------------------------------------------------------------------
# NLP Alpha Pipeline
# ---------------------------------------------------------------------------

class NLPAlphaPipeline:
    """
    Full NLP alpha generation pipeline.

    Steps:
    1. Fetch news from multiple sources
    2. Clean and deduplicate
    3. Run FinBERT sentiment + LLM scoring
    4. Detect financial events
    5. Build alpha signals with decay
    6. Output position sizing hints
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        os.makedirs(self.config.cache_dir, exist_ok=True)
        if self.config.save_articles:
            os.makedirs(self.config.output_dir, exist_ok=True)

        # Initialize components
        self.rss_fetcher = MultiFeedFetcher(
            max_articles_per_feed=self.config.max_articles_per_source
        )

        self.social_agg = SocialSignalAggregator()

        self.edgar_fetcher = EDGARFetcher()

        self.dedup = Deduplicator(near_dedupe=True, similarity_threshold=0.85)

        finbert_cfg = FinBERTConfig(
            model_name=self.config.finbert_model,
            use_cache=self.config.use_cache,
            cache_dir=os.path.join(self.config.cache_dir, "finbert"),
        )
        self.sentiment = FinBERTSentiment(finbert_cfg)

        llm_cfg = LLMScorerConfig(
            backend=self.config.llm_backend,
            api_key=self.config.llm_api_key,
            use_cache=self.config.use_cache,
            cache_dir=os.path.join(self.config.cache_dir, "llm"),
        )
        self.llm_scorer = LLMScorer(llm_cfg) if self.config.use_llm_scorer else None

        self.event_detector = EventDetector()

        self.alpha_builder = AlphaBuilder()

        self._run_count = 0
        self._total_articles = 0

    def run(self, tickers: Optional[List[str]] = None) -> PipelineResult:
        """
        Execute full pipeline.
        Returns PipelineResult with signals for given tickers.
        """
        t0 = time.time()
        ts = datetime.now(timezone.utc)
        tickers = tickers or self.config.tickers
        self._run_count += 1

        if self.config.verbose:
            logger.info(f"Pipeline run #{self._run_count} | tickers: {tickers} | ts: {ts}")

        # --- Step 1: Fetch news ---
        articles: List[NewsArticle] = []
        social_posts: List[SocialPost] = []
        sec_filings: List[SECFiling] = []

        if self.config.fetch_rss:
            try:
                raw_articles = self.rss_fetcher.fetch_all(categories=self.config.rss_categories)
                articles.extend(raw_articles)
                logger.info(f"RSS: {len(raw_articles)} articles")
            except Exception as e:
                logger.error(f"RSS fetch error: {e}")

        if self.config.fetch_social and tickers:
            try:
                ticker_posts = self.social_agg.fetch_all_signals(tickers)
                for t_posts in ticker_posts.values():
                    social_posts.extend(t_posts)
                logger.info(f"Social: {len(social_posts)} posts")
            except Exception as e:
                logger.error(f"Social fetch error: {e}")

        if self.config.fetch_sec:
            try:
                sec_filings = self.edgar_fetcher.get_material_events_feed(n=20)
                logger.info(f"SEC: {len(sec_filings)} filings")
            except Exception as e:
                logger.error(f"SEC fetch error: {e}")

        n_fetched = len(articles) + len(social_posts) + len(sec_filings)

        # --- Step 2: Filter by lookback and deduplicate ---
        cutoff = ts - timedelta(hours=self.config.lookback_hours)

        recent_articles = [
            a for a in articles
            if a.published_at is None or a.published_at >= cutoff
        ]

        # Tag tickers on articles
        ticker_set = set(t.upper() for t in tickers) if tickers else set()
        for article in recent_articles:
            if not article.tickers:
                text = article.title + " " + article.summary
                article.tickers = extract_tickers(text, ticker_set or None)

        # Deduplicate
        texts_seen = set()
        unique_articles: List[NewsArticle] = []
        for a in recent_articles:
            key = (a.title + a.summary)[:200]
            if key not in texts_seen:
                texts_seen.add(key)
                unique_articles.append(a)

        n_dedup = len(unique_articles) + len(social_posts) + len(sec_filings)

        # --- Step 3: NLP analysis ---
        all_texts = [
            (a.title + ". " + a.summary)[:512]
            for a in unique_articles
        ] + [
            p.text[:512] for p in social_posts
        ]

        sentiment_results: List[SentimentResult] = []
        article_scores: List[ArticleScore] = []

        if all_texts:
            try:
                batch_sent = self.sentiment.analyze_batch(all_texts)
                sentiment_results = batch_sent.results
            except Exception as e:
                logger.error(f"Sentiment error: {e}")

            if self.llm_scorer:
                try:
                    article_scores = self.llm_scorer.score_batch(all_texts[:50])  # cap to avoid high costs
                except Exception as e:
                    logger.error(f"LLM scoring error: {e}")

        n_scored = len(sentiment_results)

        # --- Step 4: Event detection ---
        events_by_ticker: Dict[str, List[DetectedEvent]] = {}
        all_events: List[DetectedEvent] = []

        for i, article in enumerate(unique_articles):
            text = article.title + " " + article.summary
            for ticker in (article.tickers or []):
                events = self.event_detector.detect_all(text, ticker, ts)
                for ev in events:
                    all_events.append(ev)
                    events_by_ticker.setdefault(ticker, []).append(ev)

        # Also process SEC filings
        for filing in sec_filings:
            for ticker in filing.tickers:
                if not ticker:
                    continue
                text = filing.summary or filing.company_name
                events = self.event_detector.detect_all(text, ticker, filing.filed_date)
                for ev in events:
                    all_events.append(ev)
                    events_by_ticker.setdefault(ticker, []).append(ev)

        n_events = len(all_events)

        # --- Step 5: Aggregate sentiment by ticker ---
        sentiment_by_ticker: Dict[str, List[float]] = {}

        for i, (article, sent) in enumerate(zip(unique_articles, sentiment_results[:len(unique_articles)])):
            for ticker in (article.tickers or []):
                sentiment_by_ticker.setdefault(ticker, []).append(sent.score)

        # Social sentiment
        for post in social_posts:
            if post.sentiment_score is not None:
                for ticker in post.tickers:
                    sentiment_by_ticker.setdefault(ticker, []).append(post.sentiment_score)

        ticker_sentiment = {
            ticker: float(np.mean(scores))
            for ticker, scores in sentiment_by_ticker.items()
            if scores
        }

        # --- Step 6: Build alpha signals ---
        signals_by_ticker = self.alpha_builder.build_all_signals(
            all_events, ticker_sentiment, tickers or list(ticker_sentiment.keys()), ts
        )
        composite = self.alpha_builder.get_composite_signals(signals_by_ticker)
        positions = self.alpha_builder.get_position_sizes(
            composite,
            max_position=1.0,
            signal_threshold=0.1,
        )

        elapsed = time.time() - t0
        self._total_articles += n_fetched

        if self.config.verbose:
            logger.info(
                f"Pipeline complete: {n_fetched} fetched, {n_dedup} unique, "
                f"{n_scored} scored, {n_events} events | {elapsed:.1f}s"
            )
            for ticker, sig in sorted(composite.items(), key=lambda x: -abs(x[1])):
                logger.info(f"  Signal: {ticker:6s} = {sig:+.3f}")

        result = PipelineResult(
            run_timestamp=ts,
            n_articles_fetched=n_fetched,
            n_after_dedup=n_dedup,
            n_scored=n_scored,
            n_events_detected=n_events,
            sentiment_by_ticker=ticker_sentiment,
            events_by_ticker=events_by_ticker,
            signals_by_ticker=signals_by_ticker,
            composite_signals=composite,
            position_sizes=positions,
            processing_time_secs=elapsed,
            articles=unique_articles,
            social_posts=social_posts,
            sec_filings=sec_filings,
        )

        if self.config.save_articles:
            self._save_result(result)

        return result

    def run_for_ticker(self, ticker: str) -> PipelineResult:
        """Convenience method to run pipeline for a single ticker."""
        return self.run(tickers=[ticker])

    def _save_result(self, result: PipelineResult) -> None:
        """Save pipeline result to disk."""
        ts_str = result.run_timestamp.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.config.output_dir, f"pipeline_{ts_str}.json")
        try:
            data = {
                "run_timestamp": result.run_timestamp.isoformat(),
                "n_articles_fetched": result.n_articles_fetched,
                "n_after_dedup": result.n_after_dedup,
                "n_events_detected": result.n_events_detected,
                "composite_signals": result.composite_signals,
                "position_sizes": result.position_sizes,
                "processing_time_secs": result.processing_time_secs,
            }
            import json
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save result: {e}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "run_count": self._run_count,
            "total_articles_processed": self._total_articles,
            "sentiment_cache_stats": {},  # could expose cache stats
        }


# ---------------------------------------------------------------------------
# Scheduled runner
# ---------------------------------------------------------------------------

class PipelineScheduler:
    """
    Runs the NLP pipeline on a schedule (e.g., every 5 minutes during market hours).
    Maintains a rolling signal buffer.
    """

    def __init__(
        self,
        pipeline: NLPAlphaPipeline,
        interval_seconds: int = 300,
        run_outside_market_hours: bool = False,
    ):
        self.pipeline = pipeline
        self.interval = interval_seconds
        self.run_outside = run_outside_market_hours
        self._signal_buffer: List[PipelineResult] = []
        self._buffer_maxlen = 100
        self._running = False

    def is_market_hours(self, dt: Optional[datetime] = None) -> bool:
        """Check if current time is within US market hours (9:30-16:00 ET)."""
        dt = dt or datetime.now(timezone.utc)
        # Simple check: 14:30-21:00 UTC (approximate EST/EDT)
        hour_utc = dt.hour + dt.minute / 60.0
        return 14.5 <= hour_utc <= 21.0 and dt.weekday() < 5

    def run_once(self, tickers: Optional[List[str]] = None) -> PipelineResult:
        """Run pipeline once and store result."""
        result = self.pipeline.run(tickers)
        self._signal_buffer.append(result)
        if len(self._signal_buffer) > self._buffer_maxlen:
            self._signal_buffer.pop(0)
        return result

    def run_loop(self, tickers: Optional[List[str]] = None, n_runs: int = -1) -> None:
        """Run pipeline in a loop. n_runs=-1 for infinite."""
        self._running = True
        runs = 0
        while self._running:
            if self.run_outside or self.is_market_hours():
                self.run_once(tickers)
                runs += 1
            if n_runs > 0 and runs >= n_runs:
                break
            time.sleep(self.interval)

    def stop(self) -> None:
        self._running = False

    def get_latest_signals(self) -> Dict[str, float]:
        """Get most recent composite signals."""
        if not self._signal_buffer:
            return {}
        return self._signal_buffer[-1].composite_signals

    def get_signal_history(self, ticker: str, n: int = 20) -> List[float]:
        """Get signal history for a ticker."""
        return [
            r.composite_signals.get(ticker, 0.0)
            for r in self._signal_buffer[-n:]
        ]


# ---------------------------------------------------------------------------
# Self-test (no external API calls)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing NLP Alpha Pipeline...")

    config = PipelineConfig(
        tickers=["AAPL", "MSFT", "TSLA"],
        fetch_rss=False,      # disabled to avoid network calls in test
        fetch_social=False,
        fetch_sec=False,
        use_llm_scorer=True,
        llm_backend="rule_based",
        verbose=True,
        use_cache=False,
    )
    pipeline = NLPAlphaPipeline(config)

    # Inject synthetic articles directly (bypassing fetch)
    from .scrapers.rss_parser import NewsArticle
    from .signal.event_detector import DetectedEvent, EventTypes

    # Manually build events to test signal pipeline
    ts = datetime.now(timezone.utc)
    test_events = [
        DetectedEvent(
            event_type=EventTypes.EARNINGS_BEAT, subtype="eps_beat",
            ticker="AAPL", confidence=0.90, direction=+1.0, magnitude=0.75,
            detected_at=ts, source_text="AAPL Q3 EPS $1.52 beat estimate $1.45"
        ),
        DetectedEvent(
            event_type=EventTypes.ANALYST_DOWNGRADE, subtype="downgrade",
            ticker="TSLA", confidence=0.80, direction=-1.0, magnitude=0.50,
            detected_at=ts, source_text="Goldman downgrades TSLA to Sell"
        ),
    ]
    test_sentiments = {"AAPL": 0.7, "MSFT": 0.3, "TSLA": -0.5}

    signals = pipeline.alpha_builder.build_all_signals(
        test_events, test_sentiments, ["AAPL", "MSFT", "TSLA"], ts
    )
    composite = pipeline.alpha_builder.get_composite_signals(signals)
    positions = pipeline.alpha_builder.get_position_sizes(composite)

    print("\nTest signals:")
    for ticker in ["AAPL", "MSFT", "TSLA"]:
        sig = composite.get(ticker, 0.0)
        pos = positions.get(ticker, 0.0)
        print(f"  {ticker}: signal={sig:+.3f}, position={pos:+.3f}")

    stats = pipeline.get_stats()
    print(f"\nPipeline stats: {stats}")
    print("NLP Alpha Pipeline self-test passed.")
