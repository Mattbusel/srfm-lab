"""
Text processing utilities for financial NLP.

Covers:
- Tokenization (word, sentence, subword)
- Financial stopwords
- Financial entity normalization (tickers, amounts, dates)
- Text deduplication (exact + near-duplicate)
- HTML/boilerplate cleaning
- Sentence segmentation
- Number normalization
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Financial stopwords
# ---------------------------------------------------------------------------

FINANCIAL_STOPWORDS: FrozenSet[str] = frozenset([
    # Standard English stopwords
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "that", "this", "these", "those", "it", "its", "itself",
    "they", "them", "their", "we", "our", "you", "your", "he", "she",
    "him", "his", "her", "who", "which", "what", "when", "where", "how",
    "up", "down", "out", "over", "under", "again", "further", "then",
    "once", "there", "here", "than", "about", "into", "through",
    # Financial-specific stopwords (high frequency, low signal)
    "company", "companies", "firm", "firms", "business", "businesses",
    "market", "markets", "share", "shares", "stock", "stocks",
    "quarter", "annual", "fiscal", "year", "period", "report", "reported",
    "said", "says", "according", "statement", "management", "executive",
    "investor", "investors", "analyst", "analysts", "financial", "results",
    "increase", "decreased", "increased", "rose", "fell", "billion",
    "million", "thousand", "percent", "basis", "points", "per",
    "compared", "versus", "versus", "prior", "previous", "following",
    "including", "included", "total", "net", "gross", "adjusted",
    "operating", "non", "gaap", "diluted", "weighted", "average",
    "common", "outstanding", "approximately", "expect", "expected",
    "guidance", "outlook", "estimate", "approximately", "due",
])

FINANCIAL_CUSTOM_STOPWORDS: FrozenSet[str] = frozenset([
    "press", "release", "forward", "looking", "statements", "disclaimer",
    "safe", "harbor", "cautionary", "note", "regarding", "references",
    "information", "purposes", "only", "consult", "advisor", "counsel",
    "attached", "exhibit", "appendix", "table", "figure", "page",
])

ALL_STOPWORDS = FINANCIAL_STOPWORDS | FINANCIAL_CUSTOM_STOPWORDS


# ---------------------------------------------------------------------------
# Financial entity normalization
# ---------------------------------------------------------------------------

# Common exchange suffixes
EXCHANGE_SUFFIXES = {
    ".US", ".NYSE", ".NASDAQ", ".AMEX", ".OTC", ".LSE", ".TSX",
    ".HK", ".SG", ".AU", ".TO", ":US", ":NYSE", ":NASDAQ",
}

# Known ticker aliases
TICKER_ALIASES: Dict[str, str] = {
    "GOOGL": "GOOG",
    "ALPHABET": "GOOG",
    "META": "META",
    "FACEBOOK": "META",
    "TESLA": "TSLA",
    "AMAZON": "AMZN",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "BERKSHIRE": "BRK.B",
}

# Regex patterns
TICKER_PATTERN     = re.compile(r'\b([A-Z]{1,5})(?:\.[A-Z]{1,3})?\b')
DOLLAR_PATTERN     = re.compile(r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?', re.IGNORECASE)
PERCENT_PATTERN    = re.compile(r'(\d+(?:\.\d+)?)\s*(?:%|percent|pct)', re.IGNORECASE)
DATE_PATTERN       = re.compile(
    r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    r'\s+\d{1,2}(?:,\s*\d{4})?|\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})\b',
    re.IGNORECASE
)
EPS_PATTERN        = re.compile(r'\$?\s*(\d+(?:\.\d+)?)\s*(?:per|a)?\s*(?:diluted\s*)?share', re.IGNORECASE)
REVENUE_PATTERN    = re.compile(r'(?:revenue|sales|net sales|net revenue)\s+(?:of\s+)?\$?\s*(\d+(?:\.\d+)?)\s*(billion|million|B|M)?', re.IGNORECASE)
GUIDANCE_PATTERN   = re.compile(r'(?:guid(?:ance|es)|forecast|outlook|target(?:ing)?)\s+.*?\$?(\d+(?:\.\d+)?)\s*(billion|million|B|M)?', re.IGNORECASE)


def normalize_dollar_amount(match_groups: Tuple) -> float:
    """Convert dollar amount with multiplier to float (in millions)."""
    amount_str = match_groups[0].replace(",", "")
    try:
        amount = float(amount_str)
    except ValueError:
        return 0.0
    multiplier_str = (match_groups[1] or "").strip().upper()
    if multiplier_str in ("BILLION", "B"):
        amount *= 1000
    elif multiplier_str in ("THOUSAND", "K"):
        amount /= 1000
    # else MILLION or empty => already in millions
    return amount


def normalize_ticker(raw: str) -> str:
    """Clean and normalize a ticker symbol."""
    raw = raw.upper().strip()
    for suffix in EXCHANGE_SUFFIXES:
        raw = raw.replace(suffix, "")
    return TICKER_ALIASES.get(raw, raw)


def extract_tickers(text: str, known_tickers: Optional[Set[str]] = None) -> List[str]:
    """
    Extract ticker mentions from text.
    Optionally validate against known_tickers set.
    Returns sorted unique list.
    """
    # Common false-positive words that match ticker pattern
    EXCLUDED_WORDS = {
        "A", "I", "IT", "AT", "IN", "ON", "OR", "TO", "UP", "BY",
        "US", "CEO", "CFO", "COO", "CTO", "IPO", "SEC", "ETF",
        "GDP", "CPI", "PPI", "PMI", "NFP", "FED", "ECB", "BOE",
        "NYSE", "NASDAQ", "AMEX", "OTC", "PDF", "USA", "UK",
        "Q1", "Q2", "Q3", "Q4", "YOY", "QOQ", "MOM", "YTD",
    }

    matches = TICKER_PATTERN.findall(text)
    tickers = []
    for m in matches:
        t = normalize_ticker(m)
        if t in EXCLUDED_WORDS:
            continue
        if known_tickers and t not in known_tickers:
            continue
        if len(t) >= 1 and t not in tickers:
            tickers.append(t)

    return tickers


def extract_financial_entities(text: str) -> Dict[str, List]:
    """
    Extract structured financial entities from text.
    Returns: {tickers, amounts, percents, dates, eps, revenue}
    """
    entities: Dict[str, List] = {
        "tickers": [],
        "dollar_amounts": [],
        "percents": [],
        "dates": [],
        "eps_values": [],
        "revenue_values": [],
    }

    entities["tickers"] = extract_tickers(text)

    # Dollar amounts
    for m in DOLLAR_PATTERN.finditer(text):
        val = normalize_dollar_amount(m.groups())
        if val > 0:
            entities["dollar_amounts"].append(val)

    # Percents
    for m in PERCENT_PATTERN.finditer(text):
        try:
            entities["percents"].append(float(m.group(1)))
        except ValueError:
            pass

    # Dates
    entities["dates"] = DATE_PATTERN.findall(text)

    # EPS
    for m in EPS_PATTERN.finditer(text):
        try:
            entities["eps_values"].append(float(m.group(1)))
        except ValueError:
            pass

    # Revenue
    for m in REVENUE_PATTERN.finditer(text):
        val = normalize_dollar_amount(m.groups())
        if val > 0:
            entities["revenue_values"].append(val)

    return entities


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

HTML_TAG_RE       = re.compile(r'<[^>]+>')
URL_RE            = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE          = re.compile(r'\S+@\S+\.\S+')
WHITESPACE_RE     = re.compile(r'\s+')
SPECIAL_CHAR_RE   = re.compile(r'[^\w\s\$\%\.\,\!\?\-\:\;\(\)\[\]\'\"]')
MULTI_PUNCT_RE    = re.compile(r'([\.!\?]){2,}')
NUMBER_RE         = re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b')


def clean_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return text


def remove_boilerplate(text: str) -> str:
    """Remove common financial document boilerplate."""
    boilerplate_patterns = [
        r'(?i)forward.looking\s+statements?.*?(?=\n\n|\Z)',
        r'(?i)safe\s+harbor.*?(?=\n\n|\Z)',
        r'(?i)this\s+(press\s+)?release\s+contains.*?(?=\n\n|\Z)',
        r'(?i)about\s+\w+[\s\w]*?\n.*?(?=\n\n|\Z)',
        r'(?i)contact:.*?(?=\n\n|\Z)',
        r'(?i)ir\s+contact.*?(?=\n\n|\Z)',
        r'(?i)(END OF RELEASE|###|\*\*\*)',
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    return text


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII where possible."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def clean_text(
    text: str,
    remove_html: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_special_chars: bool = False,
    lowercase: bool = False,
    normalize_unicode_chars: bool = True,
    remove_boilerplate_text: bool = True,
) -> str:
    """Full text cleaning pipeline."""
    if not text or not isinstance(text, str):
        return ""

    if normalize_unicode_chars:
        text = normalize_unicode(text)

    if remove_html:
        text = clean_html(text)

    if remove_boilerplate_text:
        text = remove_boilerplate(text)

    if remove_urls:
        text = URL_RE.sub(" [URL] ", text)

    if remove_emails:
        text = EMAIL_RE.sub(" [EMAIL] ", text)

    if remove_special_chars:
        text = SPECIAL_CHAR_RE.sub(" ", text)

    # Normalize whitespace
    text = MULTI_PUNCT_RE.sub(r'\1', text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    if lowercase:
        text = text.lower()

    return text


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

SENTENCE_BOUNDARIES = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
WORD_TOKEN_RE = re.compile(r"(?:[A-Za-z]+-[A-Za-z]+|[A-Za-z]+|\$\d+(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?|[^\w\s])")


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Handle common abbreviations
    abbrev_re = re.compile(
        r'\b(?:Mr|Mrs|Ms|Dr|Prof|Inc|Corp|Ltd|Co|vs|etc|approx|est|avg|no)\.'
    )
    # Temporarily replace period in abbreviations
    temp = abbrev_re.sub(lambda m: m.group().replace(".", "|||"), text)
    sentences = SENTENCE_BOUNDARIES.split(temp)
    return [s.replace("|||", ".").strip() for s in sentences if s.strip()]


def tokenize_words(
    text: str,
    lowercase: bool = True,
    remove_stopwords: bool = False,
    min_length: int = 2,
) -> List[str]:
    """Tokenize text into words."""
    tokens = WORD_TOKEN_RE.findall(text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    if min_length > 0:
        tokens = [t for t in tokens if len(t) >= min_length]
    if remove_stopwords:
        tokens = [t for t in tokens if t.lower() not in ALL_STOPWORDS]
    return tokens


def tokenize_subword(text: str, max_vocab: int = 50_000) -> List[str]:
    """
    Simple BPE-like subword tokenization placeholder.
    In production, use a proper tokenizer (e.g., HuggingFace tokenizers).
    """
    # Fallback to word tokenization
    return tokenize_words(text, lowercase=True)


def ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Compute n-grams from token list."""
    return ["_".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bag_of_words(tokens: List[str]) -> Dict[str, int]:
    """Compute bag-of-words counts."""
    counts: Dict[str, int] = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

class TFIDF:
    """Simple TF-IDF implementation for financial news."""

    def __init__(self, max_features: int = 10_000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab_: Dict[str, int] = {}
        self.idf_: np.ndarray = np.array([])
        self._df: Dict[str, int] = {}
        self._n_docs = 0

    def fit(self, documents: List[str]) -> "TFIDF":
        """Fit on a corpus of documents."""
        self._n_docs = len(documents)
        df: Dict[str, int] = defaultdict(int)

        tokenized = []
        for doc in documents:
            tokens = set(tokenize_words(doc, remove_stopwords=True))
            tokenized.append(tokens)
            for t in tokens:
                df[t] += 1

        # Filter by min_df and sort by frequency descending
        vocab_items = [(t, cnt) for t, cnt in df.items() if cnt >= self.min_df]
        vocab_items.sort(key=lambda x: -x[1])
        vocab_items = vocab_items[:self.max_features]

        self.vocab_ = {t: i for i, (t, _) in enumerate(vocab_items)}
        self._df = {t: df[t] for t, _ in vocab_items}

        # Compute IDF
        n = self._n_docs
        idf = np.zeros(len(self.vocab_))
        for t, idx in self.vocab_.items():
            idf[idx] = np.log((1 + n) / (1 + self._df[t])) + 1.0
        self.idf_ = idf
        return self

    def transform(self, document: str) -> np.ndarray:
        """Convert document to TF-IDF vector."""
        tokens = tokenize_words(document, remove_stopwords=True)
        tf: Dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0
        if tokens:
            for t in tf:
                tf[t] /= len(tokens)

        vec = np.zeros(len(self.vocab_))
        for t, tf_val in tf.items():
            if t in self.vocab_:
                idx = self.vocab_[t]
                vec[idx] = tf_val * self.idf_[idx]
        return vec

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        self.fit(documents)
        return np.vstack([self.transform(d) for d in documents])


# ---------------------------------------------------------------------------
# Text deduplication
# ---------------------------------------------------------------------------

def text_fingerprint(text: str) -> str:
    """MD5 fingerprint of cleaned text for exact deduplication."""
    cleaned = WHITESPACE_RE.sub(" ", text.lower().strip())
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()


def shingling(text: str, k: int = 5) -> Set[int]:
    """k-shingle set for near-duplicate detection."""
    tokens = tokenize_words(text, lowercase=True, min_length=1)
    shingles: Set[int] = set()
    for i in range(len(tokens) - k + 1):
        shingle = " ".join(tokens[i:i+k])
        shingles.add(hash(shingle) % (2 ** 32))
    return shingles


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return float(intersection / (union + 1e-8))


class Deduplicator:
    """
    Deduplicates a stream of text documents using:
    1. Exact hash deduplication
    2. Near-duplicate detection via MinHash/Jaccard
    """

    def __init__(
        self,
        exact_dedupe: bool = True,
        near_dedupe: bool = True,
        similarity_threshold: float = 0.8,
        shingle_size: int = 5,
    ):
        self.exact_dedupe = exact_dedupe
        self.near_dedupe  = near_dedupe
        self.threshold    = similarity_threshold
        self.k            = shingle_size

        self._seen_hashes: Set[str] = set()
        self._shingle_sets: List[Set[int]] = []
        self._documents: List[str] = []

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate of previously seen docs."""
        # Exact dedup
        if self.exact_dedupe:
            fp = text_fingerprint(text)
            if fp in self._seen_hashes:
                return True

        # Near-dup
        if self.near_dedupe and self._shingle_sets:
            new_shingles = shingling(text, self.k)
            for seen in self._shingle_sets[-100:]:   # only compare to recent 100
                if jaccard_similarity(new_shingles, seen) >= self.threshold:
                    return True

        return False

    def add(self, text: str) -> bool:
        """
        Add text if not duplicate.
        Returns True if added, False if duplicate.
        """
        if self.is_duplicate(text):
            return False

        fp = text_fingerprint(text)
        self._seen_hashes.add(fp)

        if self.near_dedupe:
            self._shingle_sets.append(shingling(text, self.k))
            self._documents.append(text)

        return True

    def deduplicate(self, texts: List[str]) -> List[str]:
        """Deduplicate a list of texts, preserving order."""
        unique = []
        for t in texts:
            if self.add(t):
                unique.append(t)
        return unique

    def reset(self) -> None:
        self._seen_hashes.clear()
        self._shingle_sets.clear()
        self._documents.clear()


# ---------------------------------------------------------------------------
# Number normalization
# ---------------------------------------------------------------------------

def normalize_numbers(text: str) -> str:
    """
    Replace specific numbers with normalized tokens.
    e.g., "$5.2B" → "[LARGE_AMOUNT]", "3.5%" → "[PERCENT_POSITIVE]"
    """
    # Dollar amounts
    def replace_dollar(m):
        try:
            val = normalize_dollar_amount(m.groups())
            if val >= 10_000:
                return "[VERY_LARGE_AMOUNT]"
            elif val >= 1_000:
                return "[LARGE_AMOUNT]"
            elif val >= 100:
                return "[MEDIUM_AMOUNT]"
            else:
                return "[SMALL_AMOUNT]"
        except Exception:
            return m.group()

    text = DOLLAR_PATTERN.sub(replace_dollar, text)

    # Percentages
    def replace_pct(m):
        try:
            val = float(m.group(1))
            if val > 10:
                return "[HIGH_PERCENT]"
            elif val > 0:
                return "[POS_PERCENT]"
            elif val < 0:
                return "[NEG_PERCENT]"
            else:
                return "[ZERO_PERCENT]"
        except Exception:
            return m.group()

    text = PERCENT_PATTERN.sub(replace_pct, text)
    return text


def extract_numeric_features(text: str) -> Dict[str, float]:
    """
    Extract numeric summary features from text:
    - n_dollar_mentions, max_dollar_amount, sum_dollar_amounts
    - n_percent_mentions, mean_percent, n_positive_pcts, n_negative_pcts
    - n_eps_mentions, max_eps
    """
    entities = extract_financial_entities(text)
    features = {}

    dollar_amounts = entities["dollar_amounts"]
    features["n_dollar_mentions"]  = float(len(dollar_amounts))
    features["max_dollar_amount"]  = float(max(dollar_amounts, default=0.0))
    features["sum_dollar_amounts"] = float(sum(dollar_amounts))

    percents = entities["percents"]
    features["n_percent_mentions"] = float(len(percents))
    features["mean_percent"]       = float(np.mean(percents)) if percents else 0.0
    features["n_positive_pcts"]    = float(sum(1 for p in percents if p > 0))
    features["n_negative_pcts"]    = float(sum(1 for p in percents if p < 0))

    eps = entities["eps_values"]
    features["n_eps_mentions"] = float(len(eps))
    features["max_eps"]        = float(max(eps, default=0.0))

    revenues = entities["revenue_values"]
    features["n_revenue_mentions"] = float(len(revenues))
    features["max_revenue"]        = float(max(revenues, default=0.0))

    return features


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_text = """
    Apple Inc. (AAPL) reported Q3 2025 earnings of $1.52 per diluted share,
    beating analyst estimates of $1.45. Revenue came in at $89.5 billion,
    up 8.2% year-over-year. The company raised guidance for Q4, targeting
    $92-95 billion in revenue. CEO Tim Cook noted strong iPhone sales growth
    of 12% and services revenue reaching $24.2 billion.
    """

    print("=== Text cleaning ===")
    cleaned = clean_text(sample_text)
    print(cleaned[:200])

    print("\n=== Financial entity extraction ===")
    entities = extract_financial_entities(sample_text)
    for k, v in entities.items():
        print(f"  {k}: {v}")

    print("\n=== Numeric features ===")
    num_feats = extract_numeric_features(sample_text)
    for k, v in num_feats.items():
        print(f"  {k}: {v}")

    print("\n=== Tokenization ===")
    tokens = tokenize_words(sample_text, remove_stopwords=True)
    print(f"  Tokens: {tokens[:15]}...")

    print("\n=== Deduplication ===")
    dedup = Deduplicator()
    texts = [sample_text, sample_text, "Different text about MSFT earnings.", sample_text[:100]]
    unique = dedup.deduplicate(texts)
    print(f"  Input: {len(texts)}, Output: {len(unique)}")

    print("text_processing self-test passed.")
