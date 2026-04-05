"""
sentiment_engine/nlp/sentiment_scorer.py
=========================================
Ensemble sentiment scorer: VADER + TextBlob, with source/recency/engagement weighting.

Ensemble design rationale
-------------------------
VADER (Valence Aware Dictionary and sEntiment Reasoner) was specifically
designed for social media short-form text.  It handles capitalisation, punctuation
emphasis (!!!!), and degree modifiers (very, extremely) natively.

TextBlob uses a different lexicon (Pattern, derived from subjective adjectives
in movie reviews) and performs better on longer-form prose like news articles.

We blend them by text length:
  - len < 80 chars  → 0.85 VADER + 0.15 TextBlob  (tweet-length, VADER dominates)
  - len < 280 chars → 0.70 VADER + 0.30 TextBlob
  - len >= 280      → 0.45 VADER + 0.55 TextBlob  (article-length, TextBlob dominates)

Final score adjustments:
  1. Source credibility multiplier (news > reddit > twitter)
  2. Recency decay: exp(-ln(2) * age_hours / half_life_hours), half_life = 4h
     At 4h the contribution is halved; at 12h it's ~16% of original weight.
  3. Engagement weight: log(1 + engagement)
  4. Emoji sentiment bonus from tokenizer

All three modifiers are multiplicative; the raw compound score is additive
w.r.t. the emoji bonus.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VADER
    _vader_available = True
except ImportError:
    _vader_available = False
    logger.warning("vaderSentiment not installed; using fallback scorer.")

try:
    from textblob import TextBlob as _TextBlob
    _textblob_available = True
except ImportError:
    _textblob_available = False
    logger.warning("textblob not installed; using fallback scorer.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECENCY_HALF_LIFE_HOURS: float = 4.0   # contribution halves every 4 hours

# Source credibility weights (must match values in news_scraper.py)
SOURCE_CREDIBILITY: dict[str, float] = {
    "CoinDesk":      0.90,
    "CoinTelegraph": 0.85,
    "Decrypt":       0.80,
    "TheBlock":      0.88,
    "reddit":        0.65,
    "twitter":       0.50,
    "unknown":       0.45,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScoredText:
    """
    Sentiment analysis result for a single piece of text.

    Attributes
    ----------
    raw_compound    : Unweighted VADER/TextBlob ensemble score [-1, +1]
    vader_compound  : VADER compound score (or 0 if unavailable)
    textblob_polarity : TextBlob polarity (or 0 if unavailable)
    emoji_adjustment: Additive emoji sentiment bonus
    final_score     : raw_compound + emoji_adjustment, clipped to [-1, +1]
    weighted_score  : final_score × credibility × recency_decay × engagement_weight
    credibility     : Source credibility multiplier used
    recency_decay   : Temporal decay factor applied
    engagement_weight: Log-engagement multiplier applied
    text_length     : Character count of scored text
    """
    raw_compound:       float
    vader_compound:     float
    textblob_polarity:  float
    emoji_adjustment:   float
    final_score:        float
    weighted_score:     float
    credibility:        float
    recency_decay:      float
    engagement_weight:  float
    text_length:        int


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class SentimentScorer:
    """
    Ensemble sentiment scorer for crypto text.

    Parameters
    ----------
    tokenizer : CryptoTokenizer instance for pre-processing
    """

    def __init__(self, tokenizer=None) -> None:
        self._tokenizer = tokenizer
        self._vader: Optional[object] = _VADER() if _vader_available else None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def score(
        self,
        text:             str,
        source:           str    = "unknown",
        created_at:       Optional[datetime] = None,
        engagement:       float  = 0.0,
        apply_emoji_bonus: bool  = True,
    ) -> ScoredText:
        """
        Score a single text string.

        Parameters
        ----------
        text             : Raw (un-normalised) text to score
        source           : Source identifier for credibility lookup
        created_at       : Publication/creation datetime (UTC); used for recency
        engagement       : Raw engagement count (likes + retweets, etc.)
        apply_emoji_bonus: Whether to add emoji sentiment adjustment

        Returns
        -------
        ScoredText with all intermediate and final scores populated.
        """
        if not text or not text.strip():
            return self._zero_score()

        # --- Pre-process ---
        emoji_adj  = 0.0
        if self._tokenizer and apply_emoji_bonus:
            emoji_adj = self._tokenizer.extract_emoji_score(text)
            norm_text = self._tokenizer.normalize(text)
        else:
            norm_text = text

        # --- VADER score ---
        vader_c = self._vader_score(norm_text)

        # --- TextBlob score ---
        tb_pol = self._textblob_score(norm_text)

        # --- Ensemble blend by length ---
        raw_compound = self._blend(vader_c, tb_pol, len(norm_text))

        # --- Emoji adjustment (capped to prevent emoji spam exploits) ---
        emoji_adj_capped = max(-0.3, min(0.3, emoji_adj * 0.1))
        final_score = max(-1.0, min(1.0, raw_compound + emoji_adj_capped))

        # --- Multipliers ---
        credibility     = SOURCE_CREDIBILITY.get(source, SOURCE_CREDIBILITY["unknown"])
        recency_decay   = self._recency_decay(created_at)
        eng_weight      = math.log1p(max(0.0, engagement))

        weighted_score  = final_score * credibility * recency_decay * (1.0 + eng_weight * 0.1)

        return ScoredText(
            raw_compound=raw_compound,
            vader_compound=vader_c,
            textblob_polarity=tb_pol,
            emoji_adjustment=emoji_adj_capped,
            final_score=final_score,
            weighted_score=weighted_score,
            credibility=credibility,
            recency_decay=recency_decay,
            engagement_weight=eng_weight,
            text_length=len(norm_text),
        )

    def score_batch(
        self,
        texts:      list[str],
        source:     str = "unknown",
        created_at: Optional[datetime] = None,
        engagement: float = 0.0,
    ) -> list[ScoredText]:
        """Score a list of texts with shared metadata."""
        return [
            self.score(t, source=source, created_at=created_at, engagement=engagement)
            for t in texts
        ]

    def aggregate_scores(
        self,
        scored: list[ScoredText],
        min_texts: int = 1,
    ) -> tuple[float, float]:
        """
        Aggregate a list of ScoredText into (mean_weighted_score, confidence).

        Confidence is based on volume of scored texts (more = higher confidence)
        and variance (lower variance = higher confidence).

        Returns
        -------
        (aggregate_score, confidence) both in [0, 1] range for confidence
        and [-1, 1] for score.
        """
        if not scored or len(scored) < min_texts:
            return 0.0, 0.0

        weights = [abs(s.weighted_score) + 1e-6 for s in scored]
        total_w = sum(weights)

        # Weighted mean score
        agg_score = sum(s.weighted_score * w for s, w in zip(scored, weights)) / total_w

        # Confidence: volume factor × consistency factor
        volume_factor = min(1.0, math.log1p(len(scored)) / math.log1p(50))  # saturates at ~50 texts

        # Consistency: 1 - normalised std dev of final scores
        scores_arr  = [s.final_score for s in scored]
        mean_s      = sum(scores_arr) / len(scores_arr)
        variance    = sum((x - mean_s) ** 2 for x in scores_arr) / len(scores_arr)
        std_dev     = math.sqrt(variance)
        consistency = max(0.0, 1.0 - std_dev)  # std in [0,1] since scores in [-1,1]

        confidence = volume_factor * consistency

        return float(max(-1.0, min(1.0, agg_score))), float(confidence)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _vader_score(self, text: str) -> float:
        """Return VADER compound score [-1, +1] or 0.0 if unavailable."""
        if self._vader is None:
            return self._simple_lexicon_score(text)
        try:
            scores = self._vader.polarity_scores(text)  # type: ignore[union-attr]
            return float(scores.get("compound", 0.0))
        except Exception as exc:
            logger.debug("VADER error: %s", exc)
            return 0.0

    def _textblob_score(self, text: str) -> float:
        """Return TextBlob polarity [-1, +1] or 0.0 if unavailable."""
        if not _textblob_available:
            return self._simple_lexicon_score(text)
        try:
            blob = _TextBlob(text)  # type: ignore[misc]
            return float(blob.sentiment.polarity)
        except Exception as exc:
            logger.debug("TextBlob error: %s", exc)
            return 0.0

    @staticmethod
    def _blend(vader: float, textblob: float, text_len: int) -> float:
        """Blend VADER and TextBlob scores based on text length."""
        if text_len < 80:
            v_w, t_w = 0.85, 0.15
        elif text_len < 280:
            v_w, t_w = 0.70, 0.30
        else:
            v_w, t_w = 0.45, 0.55
        return v_w * vader + t_w * textblob

    @staticmethod
    def _recency_decay(created_at: Optional[datetime]) -> float:
        """
        Exponential decay: exp(-ln(2) * age_hours / HALF_LIFE).
        Returns 1.0 if no timestamp provided (full weight).
        """
        if created_at is None:
            return 1.0
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600.0
        age_hours = max(0.0, age_hours)
        decay = math.exp(-math.log(2) * age_hours / RECENCY_HALF_LIFE_HOURS)
        return float(decay)

    @staticmethod
    def _simple_lexicon_score(text: str) -> float:
        """
        Minimal fallback scorer when neither VADER nor TextBlob is available.
        Uses a small hand-crafted positive/negative word list.
        """
        positive_words = {
            "bullish", "bull", "up", "rise", "rising", "gain", "gains", "profit",
            "buy", "buying", "long", "breakout", "recovery", "strong", "growth",
            "positive", "good", "great", "excellent", "moon", "rocket", "higher",
            "support", "accumulate", "all_time_high", "bullish", "confident",
        }
        negative_words = {
            "bearish", "bear", "down", "fall", "falling", "loss", "losses", "sell",
            "selling", "short", "breakdown", "crash", "weak", "drop", "decline",
            "negative", "bad", "terrible", "dump", "rekt", "rug", "fear", "lower",
            "resistance", "distribution", "all_time_low", "pessimistic",
        }
        tokens = text.lower().split()
        pos = sum(1 for t in tokens if t in positive_words)
        neg = sum(1 for t in tokens if t in negative_words)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    @staticmethod
    def _zero_score() -> ScoredText:
        return ScoredText(
            raw_compound=0.0,
            vader_compound=0.0,
            textblob_polarity=0.0,
            emoji_adjustment=0.0,
            final_score=0.0,
            weighted_score=0.0,
            credibility=0.0,
            recency_decay=0.0,
            engagement_weight=0.0,
            text_length=0,
        )
