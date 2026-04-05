"""
sentiment_engine/nlp/tokenizer.py
==================================
Crypto-domain text tokenizer and normaliser.

Design notes
------------
Standard NLP tokenizers treat "$BTC", "BTC", "Bitcoin", and "bitcoin" as
four distinct tokens, fragmenting signal.  This tokenizer:

  1. Normalises ticker mentions — "$BTC", "$btc", "BTC" → "TICKER_BTC"
  2. Translates crypto slang to sentiment-bearing tokens:
       "moon" → "very_bullish_movement"
       "rekt" → "large_loss_suffered"
       "rug"  → "exit_scam_occurred"
       etc.
  3. Maps emoji to sentiment markers:
       🚀 → "_emoji_rocket_"   (bullish)
       📉 → "_emoji_chart_down_" (bearish)
       💎 → "_emoji_diamond_"  (long-term hold, mildly bullish)
       🐻 → "_emoji_bear_"     (bearish)
       🐂 → "_emoji_bull_"     (bullish)
  4. Expands abbreviations:
       ATH → "all_time_high"
       ATL → "all_time_low"
       HODL → "hold_long_term"
       etc.
  5. Strips URLs, usernames, and non-alphanumeric noise.

The transformed text is then fed to VADER / TextBlob which can score the
expanded, normalised form far more accurately than raw crypto slang.
"""

from __future__ import annotations

import re
import unicodedata
from typing import ClassVar


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# Emoji → sentiment-bearing text replacement
EMOJI_MAP: dict[str, str] = {
    "🚀": " _emoji_rocket_ ",
    "📈": " _emoji_chart_up_ ",
    "📉": " _emoji_chart_down_ ",
    "💎": " _emoji_diamond_hands_ ",
    "🐻": " _emoji_bear_ ",
    "🐂": " _emoji_bull_ ",
    "🐋": " _emoji_whale_ ",
    "🔥": " _emoji_fire_ ",
    "💀": " _emoji_skull_ ",
    "🤑": " _emoji_money_face_ ",
    "😱": " _emoji_shocked_ ",
    "😂": " _emoji_laughing_ ",
    "🎯": " _emoji_target_ ",
    "⚠️": " _emoji_warning_ ",
    "✅": " _emoji_checkmark_ ",
    "❌": " _emoji_cross_ ",
    "🧨": " _emoji_dynamite_ ",
    "💸": " _emoji_money_flying_ ",
    "🩸": " _emoji_blood_ ",
    "🌙": " _emoji_moon_ ",
    "☀️": " _emoji_sun_ ",
}

# Sentiment contribution of NLP-normalised emoji tokens
# Used by SentimentScorer to apply an additive adjustment
EMOJI_SENTIMENT: dict[str, float] = {
    "_emoji_rocket_":        +0.5,
    "_emoji_chart_up_":      +0.4,
    "_emoji_chart_down_":    -0.5,
    "_emoji_diamond_hands_": +0.2,
    "_emoji_bear_":          -0.35,
    "_emoji_bull_":          +0.35,
    "_emoji_whale_":          0.0,   # neutral — could be buy or sell
    "_emoji_fire_":          +0.15,
    "_emoji_skull_":         -0.45,
    "_emoji_money_face_":    +0.3,
    "_emoji_shocked_":       -0.1,
    "_emoji_laughing_":      -0.05,  # often sarcasm in crypto context
    "_emoji_target_":        +0.1,
    "_emoji_warning_":       -0.2,
    "_emoji_checkmark_":     +0.2,
    "_emoji_cross_":         -0.2,
    "_emoji_dynamite_":      -0.3,
    "_emoji_money_flying_":  +0.25,
    "_emoji_blood_":         -0.4,
    "_emoji_moon_":          +0.4,
    "_emoji_sun_":           +0.2,
}

# Crypto slang → expanded phrase (sentiment-neutral expansion; VADER scores the phrase)
SLANG_MAP: dict[str, str] = {
    r"\bhodl\b":          "hold for long term investment",
    r"\bhodling\b":       "holding for long term investment",
    r"\bmoon\b":          "very bullish price increase expected",
    r"\bmooning\b":       "price increasing sharply upward",
    r"\brekt\b":          "suffered large financial loss",
    r"\brug\b":           "exit scam fraud occurred",
    r"\brugged\b":        "exit scam fraud occurred",
    r"\bpump\b":          "rapid price increase possibly manipulated",
    r"\bdump\b":          "rapid price decrease sharp drop",
    r"\bpumping\b":       "price rapidly increasing",
    r"\bdumping\b":       "price rapidly decreasing sharply",
    r"\bfud\b":           "fear uncertainty and doubt negative",
    r"\bfomo\b":          "fear of missing out anxious buying",
    r"\bbullish\b":       "positive upward trend expected",
    r"\bbearish\b":       "negative downward trend expected",
    r"\bwagmi\b":         "we are all going to make it optimistic",
    r"\bngmi\b":          "not going to make it pessimistic failure",
    r"\bsers\b":          "attention everyone",
    r"\bdegen\b":         "high risk speculative trader",
    r"\baltseason\b":     "alternative coins price increase season",
    r"\bcapitulation\b":  "forced selling bottom market",
    r"\baccumulation\b":  "buying and holding strategic position",
    r"\bbreakout\b":      "price breaking above resistance level",
    r"\bbreakdown\b":     "price breaking below support level",
    r"\bsqueeze\b":       "short squeeze forced covering upward price",
    r"\bcascade\b":       "cascading liquidations downward price",
    r"\bwhale\b":         "large investor significant market impact",
    r"\bbag\b":           "holding losing position",
    r"\bbagholder\b":     "investor holding losing depreciating asset",
    r"\bflippening\b":    "ethereum surpassing bitcoin market cap event",
    r"\bhalving\b":       "bitcoin supply reduction event bullish",
    r"\bdefi\b":          "decentralized finance protocol",
    r"\bnft\b":           "non fungible token digital asset",
    r"\bstaking\b":       "earning yield through network participation",
    r"\byield\b":         "earning return on investment",
    r"\bliquidity\b":     "available capital for trading market depth",
    r"\bslippage\b":      "price impact from large trade execution",
    r"\bgasless\b":       "zero transaction fee positive adoption",
    r"\bgas\b":           "transaction fee cost network usage",
}

# Abbreviation expansions
ABBREVIATION_MAP: dict[str, str] = {
    r"\bATH\b":   "all time high price record",
    r"\bATL\b":   "all time low price record",
    r"\bROI\b":   "return on investment profit",
    r"\bDCA\b":   "dollar cost averaging buying strategy",
    r"\bP2E\b":   "play to earn game rewards",
    r"\bL2\b":    "layer two scaling solution",
    r"\bL1\b":    "layer one base blockchain",
    r"\bTVL\b":   "total value locked protocol metric",
    r"\bAMM\b":   "automated market maker protocol",
    r"\bDAO\b":   "decentralized autonomous organization",
    r"\bEVM\b":   "ethereum virtual machine compatible",
    r"\bPOS\b":   "proof of stake consensus",
    r"\bPOW\b":   "proof of work mining consensus",
    r"\bAPY\b":   "annual percentage yield return",
    r"\bAPR\b":   "annual percentage rate return",
    r"\bCEX\b":   "centralized exchange platform",
    r"\bDEX\b":   "decentralized exchange protocol",
    r"\bTX\b":    "transaction on blockchain",
    r"\bFNG\b":   "fear and greed index sentiment",
    r"\bOI\b":    "open interest derivatives market",
}

# Compiled substitution list: (compiled_pattern, replacement)
_SUBSTITUTIONS: list[tuple[re.Pattern, str]] = []
for _slang_pat, _slang_rep in SLANG_MAP.items():
    _SUBSTITUTIONS.append((re.compile(_slang_pat, re.IGNORECASE), _slang_rep))
for _abbr_pat, _abbr_rep in ABBREVIATION_MAP.items():
    _SUBSTITUTIONS.append((re.compile(_abbr_pat), _abbr_rep))  # case-sensitive

# Ticker normalisation: $BTC → TICKER_BTC, $ETH → TICKER_ETH, etc.
_TICKER_RE = re.compile(r'\$([A-Za-z]{2,6})\b')

# URL pattern
_URL_RE = re.compile(r'https?://\S+|www\.\S+')

# @mention and #hashtag stripping (keep the word, drop the @/#)
_MENTION_RE = re.compile(r'@\w+')
_HASHTAG_RE = re.compile(r'#(\w+)')


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class CryptoTokenizer:
    """
    Normalises raw crypto social-media text for NLP sentiment analysis.

    Usage::

        tok  = CryptoTokenizer()
        text = tok.normalize("$BTC mooning 🚀 ATH soon! HODL rn, ngmi if you sell")
        # → "TICKER_BTC very bullish price increase expected ... hold for long term..."

    Methods
    -------
    normalize(text)      : Full pipeline; returns clean string.
    extract_emoji_score  : Sum of sentiment values for emoji found in text.
    tokenize(text)       : normalize → split into list of tokens.
    """

    # Class-level cache of compiled substitutions (shared across instances)
    _subs: ClassVar[list[tuple[re.Pattern, str]]] = _SUBSTITUTIONS

    def normalize(self, text: str) -> str:
        """
        Apply the full normalisation pipeline to *text*.

        Steps (in order):
          1. Unicode normalise to NFC
          2. Replace emoji with sentiment tokens
          3. Strip URLs
          4. Strip @mentions (noise)
          5. Keep #hashtag words (strip the #)
          6. Normalise $TICKER → TICKER_XXX
          7. Expand abbreviations
          8. Expand slang
          9. Collapse whitespace
        """
        if not text:
            return ""

        # 1. Unicode normalise
        text = unicodedata.normalize("NFC", text)

        # 2. Emoji → text tokens
        for emoji_char, replacement in EMOJI_MAP.items():
            text = text.replace(emoji_char, replacement)

        # 3. Strip URLs
        text = _URL_RE.sub(" ", text)

        # 4. Strip @mentions
        text = _MENTION_RE.sub(" ", text)

        # 5. Hashtags: keep word part
        text = _HASHTAG_RE.sub(r'\1', text)

        # 6. Ticker normalisation
        text = _TICKER_RE.sub(lambda m: f"TICKER_{m.group(1).upper()}", text)

        # 7 & 8. Abbreviations + slang (order: abbreviations first, then slang)
        for pattern, replacement in self._subs:
            text = pattern.sub(replacement, text)

        # 9. Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_emoji_score(self, raw_text: str) -> float:
        """
        Return the sum of EMOJI_SENTIMENT scores for all emoji found in *raw_text*
        (before normalisation, since normalisation replaces them).

        Returns a float in approximately [-2, +2], depending on emoji density.
        """
        total = 0.0
        for emoji_char, token in EMOJI_MAP.items():
            count   = raw_text.count(emoji_char)
            token_k = token.strip()
            total  += count * EMOJI_SENTIMENT.get(token_k, 0.0)
        return total

    def tokenize(self, text: str) -> list[str]:
        """
        Normalize then split on whitespace, filtering empty tokens.
        """
        return [t for t in self.normalize(text).split() if t]

    def batch_normalize(self, texts: list[str]) -> list[str]:
        """Normalize a list of texts."""
        return [self.normalize(t) for t in texts]
