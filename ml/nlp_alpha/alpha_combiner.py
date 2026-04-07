"""
alpha_combiner.py -- Combines NLP signals into a composite alpha signal.

Integrates:
    - HeadlineScorer (news sentiment)
    - EventSignalGenerator (structured events)
    - RedditSentimentCache (social sentiment)
    - FearGreedIndex (macro crypto sentiment)

Weighting scheme:
    - Event component:     0.40 if any relevant event in last 2h, else 0
    - Headline component:  0.30
    - Sentiment component: 0.20
    - Fear/Greed component: 0.10
    (Remaining weight redistributed if event component is zero)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .sentiment_analyzer import (
    FinancialLexicon,
    HeadlineScorer,
    NewsFeedAggregator,
    SentimentScore,
)
from .event_extractor import (
    EventExtractor,
    EventSignalGenerator,
    ExtractedEvent,
    EventType,
)
from .reddit_monitor import (
    PostAnalyzer,
    PostSignal,
    RedditSentimentCache,
)
from .fear_greed import FearGreedIndex, FearGreedResult


# ---------------------------------------------------------------------------
# NLPSignal dataclass
# ---------------------------------------------------------------------------

@dataclass
class NLPSignal:
    """Composite NLP alpha signal for a single symbol at a given bar time."""
    symbol: str
    composite_score: float          # final combined signal [-1, 1]
    headline_component: float       # news sentiment component [-1, 1]
    event_component: float          # event signal component [-1, 1]
    sentiment_component: float      # reddit/social sentiment component [-1, 1]
    fear_greed_component: float     # contrarian fear/greed signal [-1, 1]
    confidence: float               # overall confidence [0, 1]
    bar_time: Optional[datetime] = None
    event_weight_used: float = 0.0  # actual weight applied to event component
    n_active_events: int = 0
    n_headlines: int = 0
    fear_greed_score: float = 50.0  # raw [0, 100] fear/greed score

    def is_bullish(self) -> bool:
        return self.composite_score > 0.05

    def is_bearish(self) -> bool:
        return self.composite_score < -0.05

    def is_neutral(self) -> bool:
        return abs(self.composite_score) <= 0.05

    def position_hint(self) -> float:
        """
        Translate signal to a position size hint in [-1, 1].
        Applies a dead-band around zero and confidence scaling.
        """
        if abs(self.composite_score) < 0.05:
            return 0.0
        return self.composite_score * self.confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "composite_score": round(self.composite_score, 4),
            "headline_component": round(self.headline_component, 4),
            "event_component": round(self.event_component, 4),
            "sentiment_component": round(self.sentiment_component, 4),
            "fear_greed_component": round(self.fear_greed_component, 4),
            "confidence": round(self.confidence, 4),
            "bar_time": self.bar_time.isoformat() if self.bar_time else None,
            "event_weight_used": round(self.event_weight_used, 4),
            "n_active_events": self.n_active_events,
            "n_headlines": self.n_headlines,
            "fear_greed_score": round(self.fear_greed_score, 1),
        }


# ---------------------------------------------------------------------------
# Weight configuration
# ---------------------------------------------------------------------------

@dataclass
class CombinerWeights:
    """
    Signal component weights.

    event_base is used when there are active events in the last 2 hours.
    When no events, event weight is redistributed proportionally to others.
    """
    event_base: float = 0.40
    headline: float = 0.30
    sentiment: float = 0.20
    fear_greed: float = 0.10

    def no_event_weights(self) -> Dict[str, float]:
        """
        Redistribute event weight when no active events.
        Scales headline, sentiment, fear_greed proportionally to sum to 1.0.
        """
        remaining = self.headline + self.sentiment + self.fear_greed
        if remaining == 0.0:
            return {"headline": 0.5, "sentiment": 0.3, "fear_greed": 0.2}
        scale = 1.0 / remaining
        return {
            "headline": self.headline * scale,
            "sentiment": self.sentiment * scale,
            "fear_greed": self.fear_greed * scale,
        }

    def event_weights(self) -> Dict[str, float]:
        """Weights when active events are present."""
        return {
            "event": self.event_base,
            "headline": self.headline,
            "sentiment": self.sentiment,
            "fear_greed": self.fear_greed,
        }


# ---------------------------------------------------------------------------
# NLPAlphaModule
# ---------------------------------------------------------------------------

class NLPAlphaModule:
    """
    Top-level NLP alpha signal generator.

    Integrates all NLP subsystems into a unified get_composite_signal() API.

    Lifecycle:
        1. Create module (instantiates all subsystems)
        2. Feed live data via update_live() as headlines arrive
        3. Call get_composite_signal(symbol, bar_time) at each bar
        4. Use NLPSignal.composite_score or position_hint() for trading

    Fear & Greed is computed lazily from a provided components dict.
    Update via update_fear_greed_components() between bars.
    """

    DEFAULT_HEADLINE_WINDOW_MINUTES: float = 60.0
    DEFAULT_REDDIT_WINDOW_HOURS: float = 4.0
    EVENT_ACTIVE_WINDOW_MINUTES: float = 120.0   # 2 hours

    def __init__(
        self,
        weights: Optional[CombinerWeights] = None,
        reddit_db_path: str = ":memory:",
        headline_half_life_minutes: float = 30.0,
        event_half_life_minutes: float = 120.0,
    ) -> None:
        self.weights = weights or CombinerWeights()

        # Subsystems
        lexicon = FinancialLexicon()
        self._headline_scorer = HeadlineScorer(lexicon)
        self._news_aggregator = NewsFeedAggregator(
            half_life_minutes=headline_half_life_minutes,
            scorer=self._headline_scorer,
        )
        self._event_extractor = EventExtractor()
        self._event_gen = EventSignalGenerator(
            half_life_minutes=event_half_life_minutes
        )
        self._reddit_cache = RedditSentimentCache(db_path=reddit_db_path)
        self._fear_greed_index = FearGreedIndex()

        # Internal event store (rolling buffer)
        self._events: List[ExtractedEvent] = []
        self._max_event_age_hours: float = 6.0

        # Current Fear & Greed components (updated externally)
        self._fear_greed_components: Dict = {
            "price_momentum": 0.0,
            "volatility": 1.0,
            "social_volume": 1.0,
            "btc_dominance": 55.0,
            "google_trends": 50.0,
            "reddit_sentiment": 0.0,
        }
        self._last_fear_greed: Optional[FearGreedResult] = None

    # -----------------------------------------------------------------------
    # Live update methods
    # -----------------------------------------------------------------------

    def update_live(self, headline: Dict) -> Optional[SentimentScore]:
        """
        Process an incoming headline dict in real-time.

        Expected dict fields:
            - text / title: str (required)
            - summary: str (optional)
            - symbol: str (optional)
            - source: str (optional)
            - timestamp: datetime (optional)

        Simultaneously:
        1. Scores the headline and adds to news buffer
        2. Extracts events and appends to event store
        """
        text = headline.get("text") or headline.get("title") or ""
        summary = headline.get("summary", "")
        if summary:
            text = text + ". " + summary
        text = text[:512]

        if not text.strip():
            return None

        symbol = headline.get("symbol", "")
        source = headline.get("source", "default")
        timestamp = headline.get("timestamp")

        # Score headline
        score = self._news_aggregator.add_headline(text, symbol, source, timestamp)

        # Extract events
        events = self._event_extractor.extract_with_context(text, symbol, timestamp)
        self._events.extend(events)

        # Prune old events
        self._prune_events()

        return score

    def update_live_batch(self, headlines: List[Dict]) -> List[Optional[SentimentScore]]:
        """Process a batch of incoming headlines."""
        return [self.update_live(h) for h in headlines]

    def ingest_reddit_posts(self, posts: List[Dict], symbol: str = "") -> int:
        """
        Ingest Reddit post data dict list.
        Returns number of posts stored in cache.
        """
        analyzer = PostAnalyzer()
        signals = analyzer.analyze_batch(posts, target_symbol=symbol)
        return self._reddit_cache.update(signals)

    def update_fear_greed_components(self, components: Dict) -> FearGreedResult:
        """
        Update Fear & Greed components and recompute the index.

        components: dict of raw component values
            (see FearGreedIndex.compute docstring for format)

        Returns the new FearGreedResult.
        """
        self._fear_greed_components.update(components)
        self._last_fear_greed = self._fear_greed_index.compute(
            self._fear_greed_components
        )
        return self._last_fear_greed

    # -----------------------------------------------------------------------
    # Signal computation
    # -----------------------------------------------------------------------

    def get_composite_signal(
        self,
        symbol: str,
        bar_time: Optional[datetime] = None,
        headline_window_minutes: float = DEFAULT_HEADLINE_WINDOW_MINUTES,
        reddit_window_hours: float = DEFAULT_REDDIT_WINDOW_HOURS,
    ) -> NLPSignal:
        """
        Compute the composite NLP alpha signal for a symbol at bar_time.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g. "BTC", "ETH", "AAPL").
        bar_time : datetime, optional
            The current bar timestamp. Defaults to now.
        headline_window_minutes : float
            Lookback window for news headline aggregation.
        reddit_window_hours : float
            Lookback window for Reddit sentiment.

        Returns
        -------
        NLPSignal
        """
        now = bar_time or datetime.now(timezone.utc)

        # -- Headline sentiment component --
        headline_signal = self._news_aggregator.compute_signal(
            symbol, headline_window_minutes, now
        )
        n_headlines = len(self._news_aggregator.get_symbol_scores(
            symbol, headline_window_minutes
        ))

        # -- Event signal component --
        active_events = self._event_gen.get_active_events(
            self._events, symbol,
            max_age_minutes=self.EVENT_ACTIVE_WINDOW_MINUTES,
            now=now,
        )
        event_signal = self._event_gen.generate_signal(self._events, symbol, now)
        n_active_events = len(active_events)

        # -- Reddit sentiment component --
        reddit_signal = self._reddit_cache.get_rolling_sentiment(
            symbol, window_hours=reddit_window_hours, now=now
        )

        # -- Fear & Greed component --
        if self._last_fear_greed is None:
            self._last_fear_greed = self._fear_greed_index.compute(
                self._fear_greed_components
            )
        fg_result = self._last_fear_greed
        # Use contrarian signal as component
        fg_signal = fg_result.to_signal()
        fg_score = fg_result.score

        # -- Combine with weighting logic --
        composite, event_weight_used = self._combine(
            headline_signal=headline_signal,
            event_signal=event_signal,
            reddit_signal=reddit_signal,
            fg_signal=fg_signal,
            has_active_events=n_active_events > 0,
        )

        # -- Confidence: average of available sub-confidences --
        confidence = self._compute_confidence(
            n_headlines=n_headlines,
            n_events=n_active_events,
            reddit_signal=reddit_signal,
            fg_result=fg_result,
        )

        return NLPSignal(
            symbol=symbol,
            composite_score=composite,
            headline_component=headline_signal,
            event_component=event_signal,
            sentiment_component=reddit_signal,
            fear_greed_component=fg_signal,
            confidence=confidence,
            bar_time=now,
            event_weight_used=event_weight_used,
            n_active_events=n_active_events,
            n_headlines=n_headlines,
            fear_greed_score=fg_score,
        )

    def _combine(
        self,
        headline_signal: float,
        event_signal: float,
        reddit_signal: float,
        fg_signal: float,
        has_active_events: bool,
    ) -> tuple[float, float]:
        """
        Compute weighted composite signal.

        Returns (composite_score, event_weight_used).
        """
        if has_active_events:
            w = self.weights.event_weights()
            composite = (
                event_signal * w["event"]
                + headline_signal * w["headline"]
                + reddit_signal * w["sentiment"]
                + fg_signal * w["fear_greed"]
            )
            event_weight_used = w["event"]
        else:
            w = self.weights.no_event_weights()
            composite = (
                headline_signal * w["headline"]
                + reddit_signal * w["sentiment"]
                + fg_signal * w["fear_greed"]
            )
            event_weight_used = 0.0

        return max(-1.0, min(1.0, composite)), event_weight_used

    def _compute_confidence(
        self,
        n_headlines: int,
        n_events: int,
        reddit_signal: float,
        fg_result: FearGreedResult,
    ) -> float:
        """
        Estimate overall signal confidence.

        More data sources active -> higher confidence.
        """
        sources_active = 0
        total_conf = 0.0

        if n_headlines > 0:
            sources_active += 1
            # More headlines -> higher confidence, capped
            total_conf += min(0.95, 1.0 - math.exp(-0.1 * n_headlines))

        if n_events > 0:
            sources_active += 1
            total_conf += min(0.95, 1.0 - math.exp(-0.5 * n_events)) * 1.2  # events are high-quality

        if abs(reddit_signal) > 0.05:
            sources_active += 1
            total_conf += 0.5 + abs(reddit_signal) * 0.3

        # Fear & Greed is always available
        sources_active += 1
        # Extreme values are higher confidence signals
        fg_distance = abs(fg_result.score - 50.0) / 50.0
        total_conf += 0.3 + fg_distance * 0.4

        if sources_active == 0:
            return 0.1

        avg_conf = total_conf / sources_active
        # Bonus for multiple confirming sources
        multi_source_bonus = min(0.1, (sources_active - 1) * 0.03)
        return min(0.95, avg_conf + multi_source_bonus)

    def get_signals_for_symbols(
        self,
        symbols: List[str],
        bar_time: Optional[datetime] = None,
        **kwargs,
    ) -> Dict[str, NLPSignal]:
        """Compute composite signals for multiple symbols."""
        return {
            sym: self.get_composite_signal(sym, bar_time, **kwargs)
            for sym in symbols
        }

    # -----------------------------------------------------------------------
    # Event buffer management
    # -----------------------------------------------------------------------

    def _prune_events(self) -> None:
        """Remove events older than max_event_age_hours."""
        now = datetime.now(timezone.utc)
        cutoff_seconds = self._max_event_age_hours * 3600.0
        self._events = [
            ev for ev in self._events
            if ev.extracted_at is not None
            and (now - ev.extracted_at).total_seconds() < cutoff_seconds
        ]

    def add_event(self, event: ExtractedEvent) -> None:
        """Directly inject a pre-extracted event."""
        self._events.append(event)
        self._prune_events()

    def clear_events(self) -> None:
        """Clear all stored events."""
        self._events.clear()

    def get_active_events(
        self, symbol: str, max_age_minutes: float = 120.0
    ) -> List[ExtractedEvent]:
        """Return active events for a symbol."""
        return self._event_gen.get_active_events(
            self._events, symbol, max_age_minutes
        )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def diagnostics(self, symbol: str) -> Dict[str, Any]:
        """
        Return a full diagnostic dict for debugging signal composition.
        """
        now = datetime.now(timezone.utc)

        headline_summary = self._news_aggregator.get_summary(
            symbol, self.DEFAULT_HEADLINE_WINDOW_MINUTES
        )
        event_summary = self._event_gen.summarize(self._events, symbol, now)
        reddit_stats = self._reddit_cache.get_symbol_stats(symbol)

        fg = self._last_fear_greed
        fg_info = fg.to_dict() if fg else {"score": 50.0, "label": "Neutral"}

        return {
            "symbol": symbol,
            "headline": headline_summary,
            "events": event_summary,
            "reddit": reddit_stats,
            "fear_greed": fg_info,
            "weights": self.weights.__dict__,
        }

    def close(self) -> None:
        """Release resources (SQLite connection)."""
        self._reddit_cache.close()


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def make_nlp_module(
    reddit_db_path: str = ":memory:",
    event_weight: float = 0.40,
    headline_weight: float = 0.30,
    sentiment_weight: float = 0.20,
    fear_greed_weight: float = 0.10,
) -> NLPAlphaModule:
    """
    Create an NLPAlphaModule with custom weights.

    Weights must sum to 1.0 (event_weight is only active when events present).
    """
    weights = CombinerWeights(
        event_base=event_weight,
        headline=headline_weight,
        sentiment=sentiment_weight,
        fear_greed=fear_greed_weight,
    )
    return NLPAlphaModule(weights=weights, reddit_db_path=reddit_db_path)


def quick_signal(
    symbol: str,
    headlines: List[Dict],
    fear_greed_components: Optional[Dict] = None,
) -> NLPSignal:
    """
    One-shot signal computation from a list of headlines.

    Creates a fresh module, ingests headlines, returns signal.
    Useful for backtesting and one-off analyses.
    """
    module = NLPAlphaModule()
    module.update_live_batch(headlines)
    if fear_greed_components:
        module.update_fear_greed_components(fear_greed_components)
    signal = module.get_composite_signal(symbol)
    module.close()
    return signal


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import timedelta

    module = NLPAlphaModule()

    now = datetime.now(timezone.utc)

    # Feed some test headlines
    test_headlines = [
        {
            "title": "Bitcoin surges past $100k on ETF inflow record",
            "symbol": "BTC",
            "source": "bloomberg",
            "timestamp": now - timedelta(minutes=15),
        },
        {
            "title": "Ethereum network upgrade boosts DeFi optimism",
            "symbol": "ETH",
            "source": "coindesk",
            "timestamp": now - timedelta(minutes=30),
        },
        {
            "title": "Major exchange hack: $200M in ETH stolen",
            "symbol": "ETH",
            "source": "cointelegraph",
            "timestamp": now - timedelta(minutes=10),
        },
        {
            "title": "Solana rallies as Firedancer mainnet launch confirmed",
            "symbol": "SOL",
            "source": "theblock",
            "timestamp": now - timedelta(minutes=45),
        },
        {
            "title": "Bitcoin not showing bearish signals despite market correction",
            "symbol": "BTC",
            "source": "reddit",
            "timestamp": now - timedelta(minutes=55),
        },
    ]

    for h in test_headlines:
        module.update_live(h)

    # Update Fear & Greed
    module.update_fear_greed_components({
        "price_momentum": {"pct_change_7d": 0.15, "pct_change_30d": 0.35},
        "volatility": {"current_vol": 0.025, "avg_vol_30d": 0.020},
        "social_volume": 2.0,
        "btc_dominance": 52.0,
        "google_trends": 70.0,
        "reddit_sentiment": 0.4,
    })

    # Inject Reddit posts
    reddit_posts = [
        {
            "title": "ETH is going to get rekt -- too many hacks, FUD is real",
            "selftext": "Seriously considering selling my ETH bags",
            "score": 800,
            "num_comments": 350,
            "subreddit": "r/ethereum",
            "id": "test_001",
            "created_utc": (now - timedelta(minutes=20)).timestamp(),
        },
        {
            "title": "BTC to the moon -- diamond hands only!",
            "selftext": "HODL forever. WAGMI.",
            "score": 2000,
            "num_comments": 600,
            "subreddit": "r/Bitcoin",
            "id": "test_002",
            "created_utc": (now - timedelta(minutes=30)).timestamp(),
        },
    ]
    module.ingest_reddit_posts(reddit_posts, symbol="BTC")
    module.ingest_reddit_posts(reddit_posts, symbol="ETH")

    # Get signals
    symbols = ["BTC", "ETH", "SOL"]
    print("NLP Alpha Combiner Test\n" + "=" * 50)
    for symbol in symbols:
        sig = module.get_composite_signal(symbol, now)
        print(f"\n[{symbol}]")
        print(f"  Composite: {sig.composite_score:+.4f} | Confidence: {sig.confidence:.3f}")
        print(f"  Headline:  {sig.headline_component:+.4f}")
        print(f"  Events:    {sig.event_component:+.4f} (weight={sig.event_weight_used:.2f}, n={sig.n_active_events})")
        print(f"  Reddit:    {sig.sentiment_component:+.4f}")
        print(f"  Fear/Greed:{sig.fear_greed_component:+.4f} (score={sig.fear_greed_score:.1f})")
        print(f"  Position hint: {sig.position_hint():+.4f}")

    module.close()
    print("\nNLP Alpha Combiner self-test passed.")
