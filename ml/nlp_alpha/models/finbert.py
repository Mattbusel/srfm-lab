"""
FinBERT sentiment wrapper.

Provides:
- Batch inference over financial text
- Ticker entity extraction and per-ticker sentiment
- Aspect-based sentiment (management tone, guidance, risk factors)
- Caching and performance optimization
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional heavy dependencies
_transformers_available = False
_torch_available = False

try:
    import torch
    _torch_available = True
except ImportError:
    pass

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline as hf_pipeline
    _transformers_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FinBERTConfig:
    model_name: str = "ProsusAI/finbert"   # HuggingFace model ID
    max_length: int = 512
    batch_size: int = 16
    device: str = "cpu"                    # "cpu" | "cuda" | "mps"
    use_cache: bool = True
    cache_dir: str = ".finbert_cache"
    truncation_strategy: str = "only_first"
    return_all_scores: bool = True
    softmax: bool = True


# Sentiment labels for FinBERT
FINBERT_LABELS = {0: "positive", 1: "negative", 2: "neutral"}
LABEL_TO_IDX   = {"positive": 0, "negative": 1, "neutral": 2}
LABEL_SCORE    = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


# ---------------------------------------------------------------------------
# Sentiment data structures
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    """Sentiment analysis result for a single text."""
    text: str
    positive: float
    negative: float
    neutral: float
    label: str
    score: float           # composite: positive - negative
    confidence: float      # max probability
    tickers: List[str] = field(default_factory=list)
    aspects: Dict[str, float] = field(default_factory=dict)  # aspect-based scores
    cache_hit: bool = False

    @property
    def is_positive(self) -> bool:
        return self.label == "positive"

    @property
    def is_negative(self) -> bool:
        return self.label == "negative"

    @property
    def is_neutral(self) -> bool:
        return self.label == "neutral"


@dataclass
class BatchSentimentResult:
    """Results for a batch of texts."""
    results: List[SentimentResult]
    processing_time: float
    model_name: str
    n_cache_hits: int = 0

    @property
    def mean_score(self) -> float:
        return float(np.mean([r.score for r in self.results])) if self.results else 0.0

    @property
    def positive_ratio(self) -> float:
        if not self.results:
            return 0.0
        return float(sum(1 for r in self.results if r.is_positive) / len(self.results))

    def get_ticker_sentiments(self) -> Dict[str, float]:
        """Aggregate sentiment by ticker across all results."""
        ticker_scores: Dict[str, List[float]] = {}
        for r in self.results:
            for ticker in r.tickers:
                ticker_scores.setdefault(ticker, []).append(r.score)
        return {t: float(np.mean(scores)) for t, scores in ticker_scores.items()}


# ---------------------------------------------------------------------------
# Rule-based fallback sentiment
# ---------------------------------------------------------------------------

POSITIVE_WORDS = frozenset([
    "beat", "exceeded", "record", "growth", "strong", "momentum", "upgrade",
    "outperform", "raised", "guidance", "above", "better", "profit", "gain",
    "surge", "rally", "bullish", "buy", "overweight", "positive", "improve",
    "expand", "increase", "revenue", "earnings", "beat", "boost", "rise",
    "accelerate", "opportunity", "confident", "optimistic", "robust",
])

NEGATIVE_WORDS = frozenset([
    "miss", "declined", "loss", "weak", "downgrade", "underperform", "cut",
    "below", "worse", "bearish", "sell", "underweight", "negative", "decrease",
    "fall", "drop", "slowdown", "concern", "warning", "risk", "challenging",
    "disappointing", "miss", "shortfall", "layoff", "restructure", "headwind",
    "lower", "reduce", "pressure", "cautious", "uncertain",
])

INTENSIFIERS = frozenset(["significantly", "substantially", "dramatically", "sharply", "materially"])
NEGATIONS = frozenset(["not", "no", "never", "neither", "nor", "barely", "hardly"])


def rule_based_sentiment(text: str) -> SentimentResult:
    """
    Fast rule-based sentiment as fallback when model unavailable.
    Uses keyword counting with negation handling.
    """
    from ..utils.text_processing import tokenize_words

    words = tokenize_words(text, lowercase=True, min_length=2)
    n = len(words)
    if n == 0:
        return SentimentResult(text=text[:200], positive=0.33, negative=0.33, neutral=0.34,
                               label="neutral", score=0.0, confidence=0.34)

    pos_count = 0
    neg_count = 0
    negate = False
    intensify = 1.0

    for i, word in enumerate(words):
        if word in NEGATIONS:
            negate = True
            continue
        if word in INTENSIFIERS:
            intensify = 2.0
            continue

        if word in POSITIVE_WORDS:
            if negate:
                neg_count += intensify
            else:
                pos_count += intensify
            negate = False
            intensify = 1.0
        elif word in NEGATIVE_WORDS:
            if negate:
                pos_count += intensify
            else:
                neg_count += intensify
            negate = False
            intensify = 1.0
        else:
            negate = False
            intensify = 1.0

    total = pos_count + neg_count + 1.0  # +1 neutral
    pos_prob = pos_count / total
    neg_prob = neg_count / total
    neu_prob = 1.0 / total

    # Normalize to sum to 1
    total_prob = pos_prob + neg_prob + neu_prob
    pos_prob /= total_prob
    neg_prob /= total_prob
    neu_prob /= total_prob

    if pos_prob > neg_prob and pos_prob > neu_prob:
        label = "positive"
        confidence = pos_prob
    elif neg_prob > pos_prob and neg_prob > neu_prob:
        label = "negative"
        confidence = neg_prob
    else:
        label = "neutral"
        confidence = neu_prob

    score = float(pos_prob - neg_prob)

    return SentimentResult(
        text=text[:200],
        positive=float(pos_prob),
        negative=float(neg_prob),
        neutral=float(neu_prob),
        label=label,
        score=score,
        confidence=float(confidence),
    )


# ---------------------------------------------------------------------------
# Sentiment cache
# ---------------------------------------------------------------------------

class SentimentCache:
    """Simple disk-backed sentiment result cache."""

    def __init__(self, cache_dir: str = ".finbert_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._mem_cache: Dict[str, SentimentResult] = {}

    def _cache_key(self, text: str, model: str) -> str:
        combined = f"{model}:{text[:500]}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[SentimentResult]:
        key = self._cache_key(text, model)
        if key in self._mem_cache:
            result = self._mem_cache[key]
            result.cache_hit = True
            return result

        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                result = SentimentResult(**{
                    k: v for k, v in data.items()
                    if k in SentimentResult.__dataclass_fields__
                })
                result.cache_hit = True
                self._mem_cache[key] = result
                return result
            except Exception:
                pass
        return None

    def set(self, text: str, model: str, result: SentimentResult) -> None:
        key = self._cache_key(text, model)
        self._mem_cache[key] = result
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "text": result.text,
                    "positive": result.positive,
                    "negative": result.negative,
                    "neutral": result.neutral,
                    "label": result.label,
                    "score": result.score,
                    "confidence": result.confidence,
                    "tickers": result.tickers,
                    "aspects": result.aspects,
                }, f)
        except Exception:
            pass

    def clear(self) -> None:
        self._mem_cache.clear()


# ---------------------------------------------------------------------------
# FinBERT wrapper
# ---------------------------------------------------------------------------

class FinBERTSentiment:
    """
    FinBERT-based financial sentiment analysis.

    Falls back to rule-based sentiment if transformers not available.
    Supports batch inference, caching, entity extraction.
    """

    def __init__(self, config: Optional[FinBERTConfig] = None):
        self.config = config or FinBERTConfig()
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._model_loaded = False
        self._cache = SentimentCache(self.config.cache_dir) if self.config.use_cache else None
        self._load_model()

    def _load_model(self) -> None:
        """Load FinBERT model and tokenizer."""
        if not _transformers_available or not _torch_available:
            logger.warning("Transformers/PyTorch not available. Using rule-based sentiment.")
            return

        try:
            logger.info(f"Loading FinBERT: {self.config.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name
            )
            device = -1 if self.config.device == "cpu" else (0 if self.config.device == "cuda" else -1)
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device,
                return_all_scores=True,
            )
            self._model_loaded = True
            logger.info("FinBERT loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load FinBERT: {e}. Using rule-based fallback.")

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text."""
        return self.analyze_batch([text]).results[0]

    def analyze_batch(self, texts: List[str]) -> BatchSentimentResult:
        """
        Analyze a batch of texts.
        Returns BatchSentimentResult with per-text results.
        """
        from ..utils.text_processing import extract_tickers

        t0 = time.time()
        results = []
        n_cache = 0

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i: i + self.config.batch_size]
            batch_results = []

            for text in batch:
                # Check cache
                cached = None
                if self._cache:
                    cached = self._cache.get(text, self.config.model_name)

                if cached:
                    n_cache += 1
                    batch_results.append(cached)
                else:
                    result = self._infer_single(text)
                    result.tickers = extract_tickers(text)
                    if self._cache:
                        self._cache.set(text, self.config.model_name, result)
                    batch_results.append(result)

            results.extend(batch_results)

        elapsed = time.time() - t0
        return BatchSentimentResult(
            results=results,
            processing_time=elapsed,
            model_name=self.config.model_name,
            n_cache_hits=n_cache,
        )

    def _infer_single(self, text: str) -> SentimentResult:
        """Run inference on a single text."""
        if not self._model_loaded:
            return rule_based_sentiment(text)

        try:
            # Truncate to max_length characters
            text_trunc = text[:self.config.max_length * 4]  # rough char estimate

            output = self._pipeline(text_trunc, truncation=True, max_length=self.config.max_length)

            if isinstance(output, list) and output:
                if isinstance(output[0], list):
                    scores_list = output[0]
                else:
                    scores_list = output

                scores = {item["label"].lower(): float(item["score"]) for item in scores_list}

                # Map to standard labels
                pos  = scores.get("positive", 0.33)
                neg  = scores.get("negative", 0.33)
                neu  = scores.get("neutral",  0.34)

                label = max(scores, key=scores.get)
                score = pos - neg

                return SentimentResult(
                    text=text[:200],
                    positive=pos,
                    negative=neg,
                    neutral=neu,
                    label=label,
                    score=score,
                    confidence=max(scores.values()),
                )
        except Exception as e:
            logger.debug(f"FinBERT inference error: {e}")

        return rule_based_sentiment(text)

    def analyze_with_aspects(self, text: str) -> SentimentResult:
        """
        Aspect-based sentiment analysis.
        Segments text into: management_tone, guidance, risk_factors, financials
        and scores each independently.
        """
        result = self.analyze(text)

        # Segment by aspect
        aspects = self._segment_aspects(text)
        aspect_scores = {}

        for aspect, aspect_text in aspects.items():
            if aspect_text:
                asp_result = self._infer_single(aspect_text)
                aspect_scores[aspect] = asp_result.score

        result.aspects = aspect_scores
        return result

    def _segment_aspects(self, text: str) -> Dict[str, str]:
        """Segment text into financial aspects."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        aspects = {
            "financials": [],
            "guidance": [],
            "risk": [],
            "management_tone": [],
            "market": [],
        }

        financial_kws = ["revenue", "earnings", "profit", "margin", "eps", "cash flow"]
        guidance_kws  = ["expect", "guidance", "outlook", "forecast", "target"]
        risk_kws      = ["risk", "challenge", "uncertainty", "concern", "headwind"]
        market_kws    = ["market", "demand", "competition", "industry", "share"]

        for sent in sentences:
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in financial_kws):
                aspects["financials"].append(sent)
            elif any(kw in sent_lower for kw in guidance_kws):
                aspects["guidance"].append(sent)
            elif any(kw in sent_lower for kw in risk_kws):
                aspects["risk"].append(sent)
            elif any(kw in sent_lower for kw in market_kws):
                aspects["market"].append(sent)
            else:
                aspects["management_tone"].append(sent)

        return {k: " ".join(v)[:1000] for k, v in aspects.items()}

    def score_ticker_sentiment(
        self,
        articles: List[Dict[str, str]],  # list of {text, title} dicts
        ticker: str,
    ) -> Dict[str, float]:
        """
        Compute aggregated sentiment score for a specific ticker
        from a list of articles.
        """
        texts = [
            (a.get("title", "") + " " + a.get("text", ""))[:512]
            for a in articles
        ]

        batch = self.analyze_batch(texts)

        # Filter to articles mentioning the ticker
        relevant = [
            r for r in batch.results
            if ticker.upper() in r.tickers or ticker.upper() in r.text.upper()
        ]

        if not relevant:
            return {"score": 0.0, "n_articles": 0, "positive_pct": 0.0}

        scores = [r.score for r in relevant]
        return {
            "score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "n_articles": len(relevant),
            "positive_pct": float(sum(1 for r in relevant if r.is_positive) / len(relevant)),
            "negative_pct": float(sum(1 for r in relevant if r.is_negative) / len(relevant)),
            "max_score": float(max(scores)),
            "min_score": float(min(scores)),
        }

    def get_entity_sentiments(self, text: str) -> Dict[str, SentimentResult]:
        """
        For each ticker mention in text, extract surrounding context
        and compute ticker-specific sentiment.
        """
        from ..utils.text_processing import extract_tickers

        tickers = extract_tickers(text)
        results = {}

        for ticker in tickers:
            # Find sentences mentioning this ticker
            sentences = re.split(r'(?<=[.!?])\s+', text)
            relevant_sents = [s for s in sentences if ticker.upper() in s.upper()]

            if relevant_sents:
                context = " ".join(relevant_sents[:5])
                result = self._infer_single(context)
                result.tickers = [ticker]
                results[ticker] = result

        return results


# ---------------------------------------------------------------------------
# VADER-style financial sentiment (faster fallback)
# ---------------------------------------------------------------------------

class VaderFinancial:
    """
    VADER-inspired lexicon-based sentiment for financial text.
    Uses an extended financial lexicon.
    """

    FINANCIAL_LEXICON: Dict[str, float] = {
        # Strong positive
        "beat": 2.0, "exceeded": 2.0, "record": 1.8, "surpassed": 1.8,
        "outperformed": 1.8, "raised": 1.5, "upgrade": 1.8, "buy": 1.5,
        "overweight": 1.5, "bullish": 1.8, "strong": 1.5, "robust": 1.5,
        "momentum": 1.3, "growth": 1.3, "accelerating": 1.5, "expansion": 1.3,
        "dividend": 1.2, "buyback": 1.2, "increased": 1.2, "profitable": 1.5,
        # Mild positive
        "improving": 0.8, "stable": 0.5, "inline": 0.3, "met": 0.3,
        "consistent": 0.5, "solid": 0.8, "positive": 0.8,
        # Strong negative
        "miss": -2.0, "missed": -2.0, "loss": -1.8, "shortfall": -2.0,
        "declined": -1.5, "downgrade": -2.0, "sell": -1.5, "bearish": -1.8,
        "underperform": -1.8, "weak": -1.5, "disappointing": -1.8,
        "warning": -2.0, "fraud": -3.0, "lawsuit": -2.0, "recall": -2.5,
        "layoff": -2.0, "restructuring": -1.5, "bankruptcy": -3.0,
        "cut": -1.5, "reduced": -1.2, "pressure": -1.2, "headwind": -1.3,
        # Mild negative
        "mixed": -0.3, "concerns": -0.8, "uncertainty": -0.8, "challenging": -1.0,
        "slow": -0.8, "cautious": -0.6, "modest": -0.3,
    }

    def score(self, text: str) -> SentimentResult:
        words = text.lower().split()
        compound = 0.0
        n_found = 0

        for i, w in enumerate(words):
            clean_w = re.sub(r'[^\w]', '', w)
            if clean_w in self.FINANCIAL_LEXICON:
                lex_score = self.FINANCIAL_LEXICON[clean_w]

                # Check for negation
                window_start = max(0, i - 3)
                context = words[window_start:i]
                if any(neg in context for neg in ["not", "no", "never", "n't"]):
                    lex_score = -lex_score * 0.5

                # Check for intensifier
                if any(itn in context for itn in ["very", "highly", "extremely", "significantly"]):
                    lex_score *= 1.5

                compound += lex_score
                n_found += 1

        if n_found == 0:
            return SentimentResult(text=text[:200], positive=0.33, negative=0.33,
                                   neutral=0.34, label="neutral", score=0.0, confidence=0.34)

        # Normalize to [-1, 1]
        score = np.tanh(compound / (n_found * 2))

        pos = max(0.0, float(score))
        neg = max(0.0, float(-score))
        neu = 1.0 - pos - neg

        if score > 0.1:
            label, conf = "positive", pos
        elif score < -0.1:
            label, conf = "negative", neg
        else:
            label, conf = "neutral", neu

        return SentimentResult(
            text=text[:200], positive=pos, negative=neg, neutral=neu,
            label=label, score=float(score), confidence=float(conf)
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing FinBERT wrapper...")

    # Test rule-based fallback
    texts = [
        "Apple beat earnings expectations with record revenue of $89B, up 8% YoY.",
        "The company missed revenue estimates and cut full-year guidance significantly.",
        "Quarterly results were in line with expectations, with stable margins.",
    ]

    # Without model (use rule-based)
    config = FinBERTConfig(use_cache=False)
    analyzer = FinBERTSentiment(config)

    batch = analyzer.analyze_batch(texts)
    print(f"Processing time: {batch.processing_time:.3f}s | Cache hits: {batch.n_cache_hits}")
    for t, r in zip(texts, batch.results):
        print(f"  [{r.label:8s} {r.score:+.3f}] {t[:60]}...")

    print(f"\nMean score: {batch.mean_score:.3f}")
    print(f"Positive ratio: {batch.positive_ratio:.1%}")

    # Test VADER variant
    vader = VaderFinancial()
    for t in texts:
        r = vader.score(t)
        print(f"VADER [{r.label:8s} {r.score:+.3f}]: {t[:60]}...")

    # Test aspect-based
    result = analyzer.analyze_with_aspects(texts[0])
    print(f"\nAspects: {result.aspects}")

    print("FinBERT self-test passed.")
