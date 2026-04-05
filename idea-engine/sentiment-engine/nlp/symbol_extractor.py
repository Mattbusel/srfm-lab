"""
sentiment_engine/nlp/symbol_extractor.py
=========================================
Extracts crypto symbol mentions and attributes per-symbol sentiment scores.

Design rationale
----------------
A single Reddit post may mention BTC, ETH, and SOL in different contexts:
  "BTC is mooning while ETH lags, I'm dumping my SOL bags"

The per-symbol sentiment here is:
  BTC → positive  (mooning)
  ETH → mildly negative (lagging)
  SOL → negative (dumping bags)

We implement this via a sentence-splitting approach:
  1. Split the normalised text into sentences (punctuation + newline boundaries).
  2. For each sentence, detect which symbols appear.
  3. Score that sentence independently.
  4. Accumulate weighted scores per symbol.

Sentences mentioning multiple symbols split the score equally among them
(a simple heuristic; more accurate would require dependency parsing, but that
adds heavy NLP dependencies we want to avoid).

Mapping
-------
We map both ticker forms ($BTC, BTC) and full names (bitcoin, Bitcoin, BITCOIN)
to the canonical uppercase ticker symbol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Symbol registry
# ---------------------------------------------------------------------------

# name/alias → canonical ticker
_NAME_TO_TICKER: dict[str, str] = {
    "bitcoin":         "BTC",
    "btc":             "BTC",
    "ethereum":        "ETH",
    "eth":             "ETH",
    "ether":           "ETH",
    "solana":          "SOL",
    "sol":             "SOL",
    "binancecoin":     "BNB",
    "bnb":             "BNB",
    "ripple":          "XRP",
    "xrp":             "XRP",
    "dogecoin":        "DOGE",
    "doge":            "DOGE",
    "cardano":         "ADA",
    "ada":             "ADA",
    "avalanche":       "AVAX",
    "avax":            "AVAX",
    "chainlink":       "LINK",
    "link":            "LINK",
    "polygon":         "POL",
    "matic":           "POL",
    "polkadot":        "DOT",
    "dot":             "DOT",
    "uniswap":         "UNI",
    "uni":             "UNI",
    "litecoin":        "LTC",
    "ltc":             "LTC",
    "shiba":           "SHIB",
    "shib":            "SHIB",
    "pepe":            "PEPE",
    "arbitrum":        "ARB",
    "arb":             "ARB",
    "optimism":        "OP",
    "aptos":           "APT",
    "apt":             "APT",
    "sui":             "SUI",
    "celestia":        "TIA",
    "tia":             "TIA",
    "injective":       "INJ",
    "inj":             "INJ",
    "render":          "RNDR",
    "rndr":            "RNDR",
    "near":            "NEAR",
    "cosmos":          "ATOM",
    "atom":            "ATOM",
    "stellar":         "XLM",
    "xlm":             "XLM",
    "monero":          "XMR",
    "xmr":             "XMR",
    "tron":            "TRX",
    "trx":             "TRX",
    "ticker_btc":      "BTC",  # normalised form from tokenizer
    "ticker_eth":      "ETH",
    "ticker_sol":      "SOL",
    "ticker_bnb":      "BNB",
    "ticker_xrp":      "XRP",
    "ticker_doge":     "DOGE",
    "ticker_ada":      "ADA",
    "ticker_avax":     "AVAX",
    "ticker_link":     "LINK",
    "ticker_matic":    "POL",
    "ticker_dot":      "DOT",
    "ticker_uni":      "UNI",
    "ticker_ltc":      "LTC",
    "ticker_shib":     "SHIB",
    "ticker_arb":      "ARB",
    "ticker_op":       "OP",
    "ticker_apt":      "APT",
    "ticker_sui":      "SUI",
    "ticker_near":     "NEAR",
    "ticker_atom":     "ATOM",
}

# Build a single compiled pattern for fast detection
_ALL_TERMS = sorted(_NAME_TO_TICKER.keys(), key=len, reverse=True)  # longest first
_DETECT_RE = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in _ALL_TERMS) + r')\b',
    re.IGNORECASE,
)

# Sentence splitter — split on sentence-ending punctuation + newlines
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|\n+')


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SymbolSentiment:
    """
    Aggregated sentiment for a single crypto symbol from one text source.

    Attributes
    ----------
    symbol          : Canonical uppercase ticker (e.g. 'BTC')
    mention_count   : How many times the symbol was mentioned
    raw_scores      : List of per-sentence compound scores where symbol appeared
    weighted_score  : Engagement-weighted mean of raw_scores
    confidence      : 0-1, based on mention volume and score consistency
    """
    symbol:         str
    mention_count:  int                  = 0
    raw_scores:     list[float]          = field(default_factory=list)
    weighted_score: float                = 0.0
    confidence:     float                = 0.0

    def aggregate(self) -> None:
        """Compute weighted_score and confidence from raw_scores."""
        import math
        if not self.raw_scores:
            self.weighted_score = 0.0
            self.confidence     = 0.0
            return
        n     = len(self.raw_scores)
        mean  = sum(self.raw_scores) / n
        # Confidence: volume factor × consistency
        vol   = min(1.0, math.log1p(n) / math.log1p(20))
        if n > 1:
            var  = sum((x - mean) ** 2 for x in self.raw_scores) / n
            std  = math.sqrt(var)
            cons = max(0.0, 1.0 - std)
        else:
            cons = 0.5  # single data point — moderate uncertainty
        self.weighted_score = float(max(-1.0, min(1.0, mean)))
        self.confidence     = float(vol * cons)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class SymbolExtractor:
    """
    Extracts per-symbol sentiment from text by:
      1. Splitting into sentences
      2. Detecting symbol mentions per sentence
      3. Scoring each sentence (via a provided scorer callable)
      4. Attributing the score to all symbols in that sentence

    Parameters
    ----------
    scorer_fn : callable(text: str) -> float
        A function that accepts a text string and returns a compound score
        in [-1, +1].  Typically wraps SentimentScorer.score().
    """

    def __init__(self, scorer_fn=None) -> None:
        self._scorer = scorer_fn or (lambda text: 0.0)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def extract(self, text: str) -> dict[str, SymbolSentiment]:
        """
        Extract per-symbol sentiment from *text*.

        Returns
        -------
        Dict mapping canonical ticker → SymbolSentiment.
        Only symbols that are actually mentioned are returned.
        """
        if not text or not text.strip():
            return {}

        accumulator: dict[str, SymbolSentiment] = {}

        sentences = self._split_sentences(text)
        for sent in sentences:
            if not sent.strip():
                continue

            symbols_in_sent = self._detect_symbols(sent)
            if not symbols_in_sent:
                continue

            # Score this sentence once, share among all symbols in it
            try:
                score = float(self._scorer(sent))
            except Exception:
                score = 0.0

            per_symbol_score = score / max(1, len(symbols_in_sent))

            for sym in symbols_in_sent:
                if sym not in accumulator:
                    accumulator[sym] = SymbolSentiment(symbol=sym)
                ss = accumulator[sym]
                ss.mention_count += 1
                ss.raw_scores.append(per_symbol_score)

        # Finalise aggregation
        for ss in accumulator.values():
            ss.aggregate()

        return accumulator

    def extract_multi(
        self,
        texts:   list[str],
        weights: Optional[list[float]] = None,
    ) -> dict[str, SymbolSentiment]:
        """
        Extract and merge per-symbol sentiment from multiple texts.

        Parameters
        ----------
        texts   : List of raw texts
        weights : Optional per-text engagement weights; defaults to 1.0 each

        Returns
        -------
        Merged dict of symbol → SymbolSentiment, aggregated across all texts.
        """
        if weights is None:
            weights = [1.0] * len(texts)

        merged: dict[str, SymbolSentiment] = {}

        for text, weight in zip(texts, weights):
            per_text = self.extract(text)
            for sym, ss in per_text.items():
                if sym not in merged:
                    merged[sym] = SymbolSentiment(symbol=sym)
                m = merged[sym]
                m.mention_count += ss.mention_count
                # Apply weight to each score contribution
                m.raw_scores.extend(s * weight for s in ss.raw_scores)

        for ss in merged.values():
            ss.aggregate()

        return merged

    @staticmethod
    def symbols_mentioned(text: str) -> set[str]:
        """
        Convenience: return only the set of canonical tickers mentioned in *text*
        without scoring.
        """
        return SymbolExtractor._detect_symbols(text)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences on punctuation and newlines."""
        parts = _SENTENCE_RE.split(text)
        # Also split on semicolons for compound sentences common in crypto posts
        result: list[str] = []
        for part in parts:
            result.extend(part.split(";"))
        return result

    @staticmethod
    def _detect_symbols(text: str) -> set[str]:
        """Return canonical tickers mentioned in *text* (case-insensitive)."""
        found: set[str] = set()
        for m in _DETECT_RE.finditer(text):
            key = m.group(0).lower()
            sym = _NAME_TO_TICKER.get(key)
            if sym:
                found.add(sym)
        return found
