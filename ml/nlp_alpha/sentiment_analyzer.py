"""
sentiment_analyzer.py -- Lightweight statistical NLP sentiment for financial text.

No ML model downloads. Pure rule-based / lexicon-driven scoring with:
- Domain-specific crypto/trading lexicon
- Intensity modifier and negation handling
- Crypto slang normalization (HODL, rekt, wen moon, FUD)
- Source credibility weighting
- Symbol relevance boosting
- Exponentially-weighted window aggregation
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Symbol name map -- maps full names to tickers for relevance detection
# ---------------------------------------------------------------------------

SYMBOL_NAME_MAP: Dict[str, str] = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "cardano": "ADA",
    "ripple": "XRP",
    "dogecoin": "DOGE",
    "polkadot": "DOT",
    "avalanche": "AVAX",
    "chainlink": "LINK",
    "uniswap": "UNI",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "nvidia": "NVDA",
    "amd": "AMD",
    "intel": "INTC",
}

# Source credibility weights -- higher weight = more trusted source
SOURCE_WEIGHTS: Dict[str, float] = {
    "coindesk": 1.2,
    "cointelegraph": 1.1,
    "bloomberg": 1.3,
    "reuters": 1.3,
    "wsj": 1.25,
    "ft": 1.2,
    "cnbc": 1.1,
    "theblock": 1.15,
    "decrypt": 1.0,
    "reddit": 0.7,
    "twitter": 0.6,
    "x.com": 0.6,
    "4chan": 0.3,
    "telegram": 0.5,
    "default": 0.9,
}


# ---------------------------------------------------------------------------
# FinancialLexicon
# ---------------------------------------------------------------------------

class FinancialLexicon:
    """
    Domain-specific sentiment lexicon for crypto and equity trading.

    Scoring model:
        base_score = sum of word scores
        Apply intensity multipliers (stacking: "very strongly bullish" = 1.5 * 1.4)
        Apply negation flip: preceding negation within 3 tokens flips sign
        Normalize: tanh(base_score / normalization_factor) -> [-1, 1]
    """

    # --- Bullish words with base scores [0.2, 1.0] ---
    BULLISH: Dict[str, float] = {
        # Strong signals
        "moon": 0.9,
        "moonshot": 1.0,
        "mooning": 0.9,
        "breakout": 0.8,
        "surge": 0.75,
        "surges": 0.75,
        "surging": 0.75,
        "surged": 0.75,
        "accumulate": 0.65,
        "accumulating": 0.65,
        "accumulation": 0.65,
        "bullish": 0.85,
        "bull": 0.7,
        "long": 0.6,
        "buy": 0.65,
        "buying": 0.65,
        "bought": 0.6,
        "rally": 0.75,
        "rallies": 0.75,
        "rallying": 0.75,
        "rallied": 0.7,
        "outperform": 0.7,
        "outperforming": 0.7,
        "outperformed": 0.65,
        "beat": 0.6,
        "beats": 0.6,
        "exceeded": 0.55,
        "exceeds": 0.55,
        "exceed": 0.55,
        "uptrend": 0.65,
        "upgrade": 0.65,
        "upgraded": 0.65,
        "overweight": 0.6,
        "recovery": 0.55,
        "recovering": 0.55,
        "recovered": 0.5,
        "rebound": 0.6,
        "rebounding": 0.6,
        "pump": 0.55,
        "pumping": 0.55,
        "pumped": 0.5,
        "ath": 0.9,          # all-time high
        "high": 0.3,
        "higher": 0.35,
        "gain": 0.5,
        "gains": 0.5,
        "gained": 0.45,
        "profit": 0.5,
        "profitable": 0.55,
        "profitability": 0.5,
        "growth": 0.5,
        "growing": 0.45,
        "grew": 0.45,
        "strong": 0.5,
        "strength": 0.5,
        "strengthening": 0.5,
        "positive": 0.45,
        "optimistic": 0.6,
        "optimism": 0.55,
        "confidence": 0.45,
        "confident": 0.45,
        "adoption": 0.5,
        "listing": 0.55,
        "listed": 0.5,
        "partnership": 0.5,
        "integration": 0.4,
        "launch": 0.45,
        "launched": 0.4,
        "milestone": 0.45,
        "record": 0.4,
        "breakthrough": 0.65,
        "innovation": 0.4,
        "approve": 0.55,
        "approved": 0.6,
        "approval": 0.6,
        "support": 0.35,
        "institutional": 0.45,
    }

    # --- Bearish words with base scores [-1.0, -0.2] ---
    BEARISH: Dict[str, float] = {
        # Strong signals
        "crash": -0.9,
        "crashes": -0.9,
        "crashing": -0.9,
        "crashed": -0.85,
        "dump": -0.75,
        "dumping": -0.75,
        "dumped": -0.7,
        "bearish": -0.85,
        "bear": -0.7,
        "short": -0.6,
        "sell": -0.65,
        "selling": -0.65,
        "sold": -0.55,
        "liquidate": -0.75,
        "liquidated": -0.75,
        "liquidation": -0.8,
        "capitulation": -0.9,
        "capitulate": -0.85,
        "breakdown": -0.8,
        "downtrend": -0.65,
        "downgrade": -0.65,
        "downgraded": -0.65,
        "underperform": -0.65,
        "underperforming": -0.65,
        "underperformed": -0.6,
        "miss": -0.6,
        "missed": -0.6,
        "misses": -0.6,
        "below": -0.4,
        "below-estimate": -0.6,
        "disappointing": -0.6,
        "disappointed": -0.55,
        "disappoint": -0.55,
        "decline": -0.55,
        "declining": -0.55,
        "declined": -0.5,
        "drop": -0.55,
        "dropping": -0.55,
        "dropped": -0.5,
        "fall": -0.5,
        "falling": -0.5,
        "fell": -0.5,
        "loss": -0.55,
        "losses": -0.55,
        "losing": -0.5,
        "lost": -0.45,
        "hack": -0.85,
        "hacked": -0.9,
        "exploit": -0.85,
        "exploited": -0.85,
        "breach": -0.8,
        "breached": -0.8,
        "stolen": -0.85,
        "drained": -0.8,
        "ban": -0.7,
        "banned": -0.75,
        "banning": -0.7,
        "crackdown": -0.7,
        "regulation": -0.35,
        "lawsuit": -0.65,
        "sued": -0.65,
        "fraud": -0.85,
        "scam": -0.9,
        "rug": -0.9,
        "rugpull": -0.95,
        "ponzi": -0.9,
        "bankrupt": -0.9,
        "bankruptcy": -0.9,
        "insolvent": -0.85,
        "insolvency": -0.85,
        "weak": -0.45,
        "weakness": -0.45,
        "concern": -0.4,
        "concerns": -0.4,
        "risk": -0.3,
        "risks": -0.3,
        "risky": -0.35,
        "volatile": -0.2,
        "uncertainty": -0.35,
        "uncertain": -0.35,
        "warning": -0.5,
        "warn": -0.45,
        "fears": -0.45,
        "fear": -0.4,
        "panic": -0.7,
        "panicking": -0.7,
        "correction": -0.45,
        "correcting": -0.4,
        "overvalued": -0.55,
        "overbought": -0.5,
        "bubble": -0.6,
        "resistance": -0.25,
        "rejection": -0.45,
        "reject": -0.4,
        "rejected": -0.45,
    }

    # --- Intensity modifiers ---
    INTENSIFIERS: Dict[str, float] = {
        "very": 1.4,
        "extremely": 1.6,
        "strongly": 1.5,
        "hugely": 1.5,
        "massively": 1.55,
        "significantly": 1.35,
        "substantially": 1.3,
        "considerably": 1.25,
        "highly": 1.35,
        "absolutely": 1.45,
        "totally": 1.3,
        "completely": 1.35,
        "incredibly": 1.5,
        "insanely": 1.5,
        "wildly": 1.45,
        "super": 1.4,
        "ultra": 1.45,
        "major": 1.35,
        "massive": 1.4,
        "huge": 1.35,
        "enormous": 1.4,
        "dramatic": 1.3,
        "drastically": 1.4,
        "sharply": 1.3,
        "slightly": 0.7,
        "somewhat": 0.75,
        "mildly": 0.7,
        "barely": 0.6,
        "little": 0.65,
    }

    # --- Negation words (flip sentiment sign within 3-token window) ---
    NEGATIONS = frozenset([
        "not", "no", "never", "rarely", "hardly",
        "doesn't", "don't", "didn't", "won't", "wouldn't",
        "isn't", "aren't", "wasn't", "weren't", "cannot",
        "can't", "couldn't", "shouldn't", "neither", "nor",
        "without", "lacks", "lack", "lacking",
    ])

    # --- Crypto-specific slang (pre-normalized to standard tokens) ---
    CRYPTO_SLANG: Dict[str, str] = {
        r"\bhodl\b": "bullish accumulate",
        r"\bhodling\b": "bullish accumulate",
        r"\brekt\b": "liquidated crash",
        r"\bget rekt\b": "liquidated crash",
        r"\bwen moon\b": "moon bullish",
        r"\bto the moon\b": "moon rally",
        r"\bfud\b": "fear uncertainty bearish",
        r"\bfudding\b": "fear uncertainty bearish",
        r"\bbuidl\b": "building growth bullish",
        r"\bngmi\b": "loss bearish capitulation",
        r"\bwagmi\b": "bullish optimism gain",
        r"\bdegen\b": "risky volatile",
        r"\bapeing in\b": "buy bullish",
        r"\bape in\b": "buy bullish",
        r"\bpaper hands\b": "sell capitulation",
        r"\bdiamond hands\b": "hodl accumulate",
        r"\bbullrun\b": "bull rally bullish",
        r"\balt season\b": "rally bullish",
        r"\bsatoshi\b": "",          # neutral -- just a unit
        r"\bdefi\b": "",             # neutral DeFi reference
        r"\bnft\b": "",              # neutral
        r"\bgm\b": "optimistic",     # "good morning" == positive community signal
        r"\bgn\b": "",               # "good night" neutral
        r"\bser\b": "",              # "sir" neutral
        r"\bbear trap\b": "bullish breakout",
        r"\bbull trap\b": "bearish breakdown",
        r"\bshort squeeze\b": "rally bullish",
        r"\bdead cat bounce\b": "bearish correction",
    }

    # --- Normalization factor for tanh compression ---
    NORMALIZATION_FACTOR: float = 3.0

    def __init__(self) -> None:
        # Merge all scored words into a single lookup
        self._word_scores: Dict[str, float] = {}
        self._word_scores.update(self.BULLISH)
        self._word_scores.update(self.BEARISH)
        # Compile slang patterns
        self._slang_patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(pat, re.IGNORECASE), repl)
            for pat, repl in self.CRYPTO_SLANG.items()
        ]

    def normalize_text(self, text: str) -> str:
        """Apply crypto slang normalization."""
        for pattern, replacement in self._slang_patterns:
            text = pattern.sub(replacement, text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove @mentions and #hashtags but keep the word
        text = re.sub(r"[@#]", "", text)
        # Normalize dollar amounts -- "$50M hack" -> "50m hack"
        text = re.sub(r"\$[\d,\.]+[MmBbKkTt]?", "dollar_amount", text)
        # Lower
        text = text.lower()
        # Tokenize on non-alpha (keep apostrophes for contractions)
        tokens = re.findall(r"[a-z']+", text)
        return tokens

    def score_tokens(self, tokens: List[str]) -> Tuple[float, List[str]]:
        """
        Score a list of tokens.

        Returns (raw_score, matched_terms).
        Handles negation (3-token lookback) and intensity multipliers.
        """
        raw_score = 0.0
        matched_terms: List[str] = []
        n = len(tokens)

        for i, token in enumerate(tokens):
            if token not in self._word_scores:
                continue

            word_score = self._word_scores[token]

            # --- Negation check: look back up to 3 tokens ---
            is_negated = False
            for j in range(max(0, i - 3), i):
                if tokens[j] in self.NEGATIONS:
                    is_negated = True
                    break

            # --- Intensity check: look back up to 2 tokens ---
            intensity = 1.0
            for j in range(max(0, i - 2), i):
                if tokens[j] in self.INTENSIFIERS:
                    intensity *= self.INTENSIFIERS[tokens[j]]

            final_score = word_score * intensity
            if is_negated:
                final_score = -final_score

            raw_score += final_score
            matched_terms.append(
                f"{'NOT ' if is_negated else ''}{token}({final_score:+.2f})"
            )

        return raw_score, matched_terms

    def score_text(self, text: str) -> Tuple[float, float, List[str]]:
        """
        Full scoring pipeline.

        Returns (raw_score, normalized_score, matched_terms).
        normalized_score is in [-1, 1] via tanh.
        """
        normalized = self.normalize_text(text)
        tokens = self.tokenize(normalized)
        raw_score, matched_terms = self.score_tokens(tokens)
        # Normalize with tanh for bounded [-1, 1] output
        norm_score = math.tanh(raw_score / self.NORMALIZATION_FACTOR)
        return raw_score, norm_score, matched_terms

    def compute_confidence(self, matched_terms: List[str], token_count: int) -> float:
        """
        Confidence based on signal density (matched terms per token).
        Returns [0, 1].
        """
        if token_count == 0:
            return 0.0
        density = len(matched_terms) / token_count
        # Sigmoid-like scaling: more matches -> higher confidence, capped at 0.95
        confidence = min(0.95, 1.0 - math.exp(-5.0 * density))
        # Minimum confidence when at least 1 match
        if matched_terms and confidence < 0.1:
            confidence = 0.1
        return confidence


# ---------------------------------------------------------------------------
# SentimentScore dataclass
# ---------------------------------------------------------------------------

@dataclass
class SentimentScore:
    """Output of scoring a single piece of financial text."""
    raw_score: float
    adjusted_score: float         # symbol-relevance-boosted, source-weighted
    confidence: float
    matched_terms: List[str]
    symbol: str = ""
    source: str = "default"
    headline: str = ""
    timestamp: Optional[datetime] = None

    def is_bullish(self) -> bool:
        return self.adjusted_score > 0.1

    def is_bearish(self) -> bool:
        return self.adjusted_score < -0.1

    def is_neutral(self) -> bool:
        return abs(self.adjusted_score) <= 0.1

    def signal_strength(self) -> float:
        """Confidence-weighted adjusted score."""
        return self.adjusted_score * self.confidence


# ---------------------------------------------------------------------------
# HeadlineScorer
# ---------------------------------------------------------------------------

class HeadlineScorer:
    """
    Scores individual financial headlines for sentiment.

    Applies:
    - Base lexicon scoring
    - Symbol relevance boost (headline mentions the target ticker/name)
    - Source credibility weighting
    """

    # Symbol relevance boost multiplier when ticker appears in headline
    SYMBOL_RELEVANCE_BOOST: float = 1.25

    def __init__(self, lexicon: Optional[FinancialLexicon] = None) -> None:
        self.lexicon = lexicon or FinancialLexicon()

    def _detect_symbol_relevance(self, text: str, symbol: str) -> bool:
        """
        Returns True if the headline explicitly mentions the symbol
        (by ticker or full name).
        """
        text_upper = text.upper()
        symbol_upper = symbol.upper()

        # Direct ticker match (word boundary)
        if re.search(r"\b" + re.escape(symbol_upper) + r"\b", text_upper):
            return True

        # Full name match
        for name, ticker in SYMBOL_NAME_MAP.items():
            if ticker == symbol_upper:
                if name.lower() in text.lower():
                    return True

        return False

    def _get_source_weight(self, source: str) -> float:
        """Return credibility weight for a news source."""
        source_lower = source.lower()
        for key, weight in SOURCE_WEIGHTS.items():
            if key in source_lower:
                return weight
        return SOURCE_WEIGHTS["default"]

    def score_headline(
        self,
        text: str,
        symbol: str = "",
        source: str = "default",
        timestamp: Optional[datetime] = None,
    ) -> SentimentScore:
        """
        Score a single headline.

        Parameters
        ----------
        text : str
            Headline text (or title + summary concatenated).
        symbol : str
            Target trading symbol (e.g. "BTC", "AAPL"). Empty = no relevance boost.
        source : str
            News source identifier for credibility weighting.
        timestamp : datetime, optional
            Publication time of the headline.

        Returns
        -------
        SentimentScore
        """
        raw_score, norm_score, matched_terms = self.lexicon.score_text(text)

        # Confidence from signal density
        tokens = self.lexicon.tokenize(self.lexicon.normalize_text(text))
        confidence = self.lexicon.compute_confidence(matched_terms, len(tokens))

        # Symbol relevance boost
        adjusted = norm_score
        if symbol and self._detect_symbol_relevance(text, symbol):
            # Boost magnitude, preserve sign
            adjusted = math.tanh(
                math.atanh(max(-0.9999, min(0.9999, adjusted)))
                * self.SYMBOL_RELEVANCE_BOOST
            )

        # Source credibility weighting
        source_weight = self._get_source_weight(source)
        adjusted = adjusted * source_weight
        # Clamp back to [-1, 1]
        adjusted = max(-1.0, min(1.0, adjusted))

        return SentimentScore(
            raw_score=raw_score,
            adjusted_score=adjusted,
            confidence=confidence,
            matched_terms=matched_terms,
            symbol=symbol,
            source=source,
            headline=text[:200],
            timestamp=timestamp or datetime.now(timezone.utc),
        )

    def batch_score(
        self,
        headlines: List[Dict],
        default_symbol: str = "",
    ) -> List[SentimentScore]:
        """
        Vectorized batch scoring.

        Each dict in headlines may contain:
          - "text" or "title": str (required)
          - "symbol": str (optional)
          - "source": str (optional)
          - "timestamp": datetime (optional)
          - "summary": str (optional, appended to text)

        Returns list of SentimentScore in same order as input.
        """
        results: List[SentimentScore] = []
        for item in headlines:
            text = item.get("text") or item.get("title") or ""
            summary = item.get("summary", "")
            if summary:
                text = text + ". " + summary
            text = text[:512]  # cap to avoid huge inputs

            symbol = item.get("symbol", default_symbol)
            source = item.get("source", "default")
            timestamp = item.get("timestamp")

            score = self.score_headline(text, symbol, source, timestamp)
            results.append(score)

        return results


# ---------------------------------------------------------------------------
# NewsFeedAggregator
# ---------------------------------------------------------------------------

class NewsFeedAggregator:
    """
    Aggregates a stream of SentimentScore objects over time windows
    using exponential decay weighting.

    Decay model:
        weight(age) = exp(-lambda * age_minutes)
        lambda = ln(2) / half_life_minutes  (default half-life = 30 min)
    """

    DEFAULT_HALF_LIFE_MINUTES: float = 30.0

    def __init__(
        self,
        half_life_minutes: float = DEFAULT_HALF_LIFE_MINUTES,
        scorer: Optional[HeadlineScorer] = None,
    ) -> None:
        self.half_life = half_life_minutes
        self._lambda = math.log(2.0) / half_life_minutes
        self.scorer = scorer or HeadlineScorer()
        # Internal buffer: list of (timestamp, SentimentScore)
        self._buffer: List[Tuple[datetime, SentimentScore]] = []

    def _age_minutes(self, ts: datetime, now: Optional[datetime] = None) -> float:
        """Compute age of a timestamp in minutes relative to now."""
        now = now or datetime.now(timezone.utc)
        # Ensure both are timezone-aware
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        delta = now - ts
        return max(0.0, delta.total_seconds() / 60.0)

    def _decay_weight(self, age_minutes: float) -> float:
        """Exponential decay weight."""
        return math.exp(-self._lambda * age_minutes)

    def add_score(self, score: SentimentScore) -> None:
        """Add a pre-scored headline to the internal buffer."""
        ts = score.timestamp or datetime.now(timezone.utc)
        self._buffer.append((ts, score))

    def add_headline(
        self,
        text: str,
        symbol: str = "",
        source: str = "default",
        timestamp: Optional[datetime] = None,
    ) -> SentimentScore:
        """Score a raw headline and add to buffer. Returns the score."""
        score = self.scorer.score_headline(text, symbol, source, timestamp)
        self.add_score(score)
        return score

    def aggregate_window(
        self,
        headlines: List[SentimentScore],
        window_minutes: float = 60.0,
        now: Optional[datetime] = None,
    ) -> float:
        """
        Compute exponentially-weighted sentiment for a list of scores
        within a time window.

        Returns weighted average in [-1, 1], or 0.0 if no scores present.
        """
        now = now or datetime.now(timezone.utc)
        total_weight = 0.0
        weighted_sum = 0.0

        for score in headlines:
            ts = score.timestamp
            if ts is None:
                continue
            age = self._age_minutes(ts, now)
            if age > window_minutes:
                continue  # outside window

            w = self._decay_weight(age) * score.confidence
            # Use signal_strength (confidence-adjusted) for robustness
            weighted_sum += w * score.adjusted_score
            total_weight += w

        if total_weight == 0.0:
            return 0.0

        return max(-1.0, min(1.0, weighted_sum / total_weight))

    def compute_signal(
        self,
        symbol: str,
        window_minutes: float = 60.0,
        now: Optional[datetime] = None,
    ) -> float:
        """
        Compute sentiment signal for a symbol from the internal buffer.

        Returns [-1, 1] sentiment signal.
        """
        now = now or datetime.now(timezone.utc)
        relevant: List[SentimentScore] = []

        for ts, score in self._buffer:
            # Filter by symbol (empty symbol = applies to all)
            if score.symbol and score.symbol.upper() != symbol.upper():
                continue
            relevant.append(score)

        return self.aggregate_window(relevant, window_minutes, now)

    def prune_old(self, max_age_minutes: float = 1440.0) -> int:
        """
        Remove entries older than max_age_minutes (default 24h).
        Returns number of entries removed.
        """
        now = datetime.now(timezone.utc)
        before = len(self._buffer)
        self._buffer = [
            (ts, s) for ts, s in self._buffer
            if self._age_minutes(ts, now) <= max_age_minutes
        ]
        return before - len(self._buffer)

    def get_symbol_scores(
        self, symbol: str, window_minutes: float = 60.0
    ) -> List[SentimentScore]:
        """Return all scores for a symbol within the window."""
        now = datetime.now(timezone.utc)
        return [
            s for ts, s in self._buffer
            if (not s.symbol or s.symbol.upper() == symbol.upper())
            and self._age_minutes(ts, now) <= window_minutes
        ]

    def get_summary(self, symbol: str, window_minutes: float = 60.0) -> Dict:
        """
        Summary statistics for a symbol's recent sentiment.

        Returns dict with: signal, n_headlines, n_bullish, n_bearish,
        avg_confidence, top_terms.
        """
        scores = self.get_symbol_scores(symbol, window_minutes)
        if not scores:
            return {
                "signal": 0.0,
                "n_headlines": 0,
                "n_bullish": 0,
                "n_bearish": 0,
                "avg_confidence": 0.0,
                "top_terms": [],
            }

        signal = self.compute_signal(symbol, window_minutes)
        n_bullish = sum(1 for s in scores if s.is_bullish())
        n_bearish = sum(1 for s in scores if s.is_bearish())
        avg_conf = sum(s.confidence for s in scores) / len(scores)

        # Aggregate top matched terms by frequency
        term_freq: Dict[str, int] = {}
        for s in scores:
            for term in s.matched_terms:
                base = term.split("(")[0].strip()
                term_freq[base] = term_freq.get(base, 0) + 1
        top_terms = sorted(term_freq, key=lambda t: -term_freq[t])[:10]

        return {
            "signal": signal,
            "n_headlines": len(scores),
            "n_bullish": n_bullish,
            "n_bearish": n_bearish,
            "avg_confidence": avg_conf,
            "top_terms": top_terms,
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_scorer(source: str = "default") -> HeadlineScorer:
    """Create a HeadlineScorer with a fresh lexicon."""
    return HeadlineScorer(FinancialLexicon())


def score_text_quick(text: str, symbol: str = "", source: str = "default") -> float:
    """
    One-liner quick scoring. Returns adjusted_score in [-1, 1].
    Useful for prototyping and notebooks.
    """
    scorer = HeadlineScorer()
    result = scorer.score_headline(text, symbol, source)
    return result.adjusted_score


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lexicon = FinancialLexicon()
    scorer = HeadlineScorer(lexicon)

    test_cases = [
        ("Bitcoin surges to new ATH as institutional buying accelerates", "BTC", "bloomberg"),
        ("Ethereum crashes 20% amid regulatory crackdown fears", "ETH", "coindesk"),
        ("Solana is not bearish despite recent correction", "SOL", "reddit"),
        ("HODL! BTC to the moon, FUD is spreading but diamond hands prevail", "BTC", "twitter"),
        ("Exchange rekt by exploit, funds drained in $50M hack", "ETH", "cointelegraph"),
        ("Tesla beats Q3 earnings estimates, strong guidance", "TSLA", "bloomberg"),
        ("Apple misses revenue target, below consensus", "AAPL", "wsj"),
    ]

    print("Headline Sentiment Scorer Test\n" + "=" * 50)
    for text, symbol, source in test_cases:
        s = scorer.score_headline(text, symbol, source)
        print(f"\n[{source}] {text[:60]}...")
        print(f"  Symbol: {symbol} | Raw: {s.raw_score:+.2f} | Adj: {s.adjusted_score:+.3f} | Conf: {s.confidence:.2f}")
        print(f"  Matched: {s.matched_terms[:5]}")
