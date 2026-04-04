"""
LLM-based news scorer using prompt engineering.

Scores articles for:
- Financial relevance (0-1)
- Urgency/time-sensitivity (0-1)
- Market impact direction (-1 to +1)
- Market impact magnitude (0-1)
- Affected asset classes
- Confidence in scoring
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMScorerConfig:
    # LLM backend
    backend: str = "openai"           # "openai" | "anthropic" | "local" | "rule_based"
    model: str = "gpt-4o-mini"        # model name
    api_key: Optional[str] = None     # falls back to env variable
    max_tokens: int = 256
    temperature: float = 0.1          # low temperature for consistency
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

    # Scoring
    batch_size: int = 5               # articles per LLM call
    use_few_shot: bool = True
    confidence_threshold: float = 0.7

    # Caching
    use_cache: bool = True
    cache_dir: str = ".llm_scorer_cache"


# ---------------------------------------------------------------------------
# Scoring result
# ---------------------------------------------------------------------------

@dataclass
class ArticleScore:
    """LLM-generated scores for a news article."""
    text: str
    relevance: float          # 0-1: how relevant is this to financial markets
    urgency: float            # 0-1: how time-sensitive
    direction: float          # -1 to +1: expected market impact direction
    magnitude: float          # 0-1: expected magnitude of impact
    confidence: float         # 0-1: model confidence in scores
    affected_tickers: List[str] = field(default_factory=list)
    affected_sectors: List[str] = field(default_factory=list)
    event_type: str = "unknown"   # "earnings" | "ma" | "macro" | "analyst" | "other"
    key_facts: List[str] = field(default_factory=list)
    summary: str = ""
    model_used: str = ""
    from_cache: bool = False
    raw_response: str = ""

    @property
    def impact_score(self) -> float:
        """Composite impact score: direction * magnitude * relevance."""
        return float(self.direction * self.magnitude * self.relevance)

    @property
    def is_actionable(self) -> bool:
        return self.relevance >= 0.6 and self.magnitude >= 0.3 and self.confidence >= 0.6


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a quantitative financial analyst specializing in news impact assessment.
Your task is to score news articles for their potential market impact.
Always respond with valid JSON only. Be precise and objective."""

FEW_SHOT_EXAMPLES = [
    {
        "text": "Apple reported Q3 EPS of $1.52, beating consensus of $1.45 by 4.8%. Revenue of $89.5B exceeded estimates of $87.9B. Company raised Q4 guidance.",
        "scores": {
            "relevance": 0.95,
            "urgency": 0.90,
            "direction": 0.80,
            "magnitude": 0.75,
            "confidence": 0.92,
            "event_type": "earnings",
            "affected_tickers": ["AAPL"],
            "affected_sectors": ["technology"],
            "summary": "Apple earnings beat with guidance raise - strongly bullish for AAPL"
        }
    },
    {
        "text": "Federal Reserve raises interest rates by 25 basis points, signals two more hikes possible.",
        "scores": {
            "relevance": 0.98,
            "urgency": 0.95,
            "direction": -0.60,
            "magnitude": 0.85,
            "confidence": 0.90,
            "event_type": "macro",
            "affected_tickers": [],
            "affected_sectors": ["financials", "real_estate", "technology", "utilities"],
            "summary": "Fed rate hike broadly bearish for equities, mixed for financials"
        }
    },
    {
        "text": "Microsoft Azure cloud revenue grew 30% YoY, beating estimates of 27% growth.",
        "scores": {
            "relevance": 0.90,
            "urgency": 0.85,
            "direction": 0.70,
            "magnitude": 0.65,
            "confidence": 0.88,
            "event_type": "earnings",
            "affected_tickers": ["MSFT", "AMZN", "GOOGL"],
            "affected_sectors": ["technology", "cloud"],
            "summary": "MSFT Azure beat positive for cloud sector"
        }
    },
]

SCORING_TEMPLATE = """Analyze the following financial news article and provide impact scores.

{few_shot_section}

Now analyze this article:
TEXT: {text}

Respond with ONLY a JSON object with these exact fields:
{{
  "relevance": <0.0-1.0>,
  "urgency": <0.0-1.0>,
  "direction": <-1.0 to 1.0>,
  "magnitude": <0.0-1.0>,
  "confidence": <0.0-1.0>,
  "event_type": "<earnings|macro|ma|analyst|regulatory|product|other>",
  "affected_tickers": [<list of ticker strings>],
  "affected_sectors": [<list of sector strings>],
  "summary": "<one sentence summary>",
  "key_facts": [<list of up to 3 key facts>]
}}"""

FEW_SHOT_SECTION = "\n".join([
    f"EXAMPLE {i+1}:\nTEXT: {ex['text']}\nSCORES: {json.dumps(ex['scores'])}"
    for i, ex in enumerate(FEW_SHOT_EXAMPLES)
])


# ---------------------------------------------------------------------------
# Rule-based scorer (fallback)
# ---------------------------------------------------------------------------

RELEVANCE_KEYWORDS = {
    "high": ["earnings", "revenue", "eps", "guidance", "merger", "acquisition", "buyout",
             "fed", "rate", "inflation", "gdp", "upgrade", "downgrade", "dividend", "buyback"],
    "medium": ["sales", "profit", "market", "quarterly", "annual", "analyst", "forecast"],
    "low": ["management", "employee", "partnership", "award", "contract", "milestone"],
}

URGENCY_KEYWORDS = {
    "high": ["today", "now", "just", "breaking", "immediate", "alert", "flash",
             "surprise", "unexpected", "suddenly", "shock", "warning"],
    "medium": ["quarterly", "monthly", "weekly", "update", "scheduled"],
    "low": ["annual", "long-term", "strategic", "eventually", "planning"],
}

DIRECTION_POSITIVE = ["beat", "exceeded", "raised", "upgrade", "buy", "overweight",
                       "strong", "record", "growth", "profit", "outperformed", "bullish",
                       "increased", "positive", "approval", "win", "deal", "partnership"]

DIRECTION_NEGATIVE = ["miss", "missed", "cut", "downgrade", "sell", "underperform",
                       "weak", "loss", "declined", "bearish", "reduced", "warning",
                       "concern", "risk", "probe", "lawsuit", "recall", "layoff"]

EVENT_TYPE_KEYWORDS = {
    "earnings": ["earnings", "eps", "revenue", "quarterly", "guidance", "outlook"],
    "ma": ["merger", "acquisition", "takeover", "buyout", "deal", "divest"],
    "macro": ["federal reserve", "fed", "inflation", "gdp", "unemployment", "cpi", "rate"],
    "analyst": ["upgrade", "downgrade", "target price", "rating", "coverage"],
    "regulatory": ["sec", "investigation", "probe", "fine", "settlement", "regulation"],
    "product": ["launch", "product", "approval", "fda", "patent", "innovation"],
}

SECTOR_KEYWORDS = {
    "technology": ["software", "cloud", "ai", "semiconductor", "tech", "digital"],
    "financials": ["bank", "insurance", "financial", "credit", "lending", "payment"],
    "healthcare": ["drug", "pharmaceutical", "biotech", "medical", "clinical", "fda"],
    "energy": ["oil", "gas", "energy", "renewable", "solar", "wind", "coal"],
    "consumer": ["retail", "consumer", "restaurant", "hotel", "travel", "e-commerce"],
    "industrials": ["manufacturing", "aerospace", "defense", "industrial", "logistics"],
    "real_estate": ["reit", "property", "real estate", "housing", "mortgage"],
    "utilities": ["utility", "electric", "water", "gas distribution"],
}


class RuleBasedScorer:
    """Fast rule-based news scorer using keyword heuristics."""

    def score(self, text: str) -> ArticleScore:
        from ..utils.text_processing import extract_tickers

        text_lower = text.lower()

        # Relevance
        relevance = 0.3  # base
        for kw in RELEVANCE_KEYWORDS["high"]:
            if kw in text_lower:
                relevance = min(relevance + 0.1, 1.0)
        for kw in RELEVANCE_KEYWORDS["medium"]:
            if kw in text_lower:
                relevance = min(relevance + 0.05, 1.0)

        # Urgency
        urgency = 0.3
        for kw in URGENCY_KEYWORDS["high"]:
            if kw in text_lower:
                urgency = min(urgency + 0.15, 1.0)
        for kw in URGENCY_KEYWORDS["medium"]:
            if kw in text_lower:
                urgency = min(urgency + 0.07, 1.0)

        # Direction
        pos_score = sum(1.0 for kw in DIRECTION_POSITIVE if kw in text_lower)
        neg_score = sum(1.0 for kw in DIRECTION_NEGATIVE if kw in text_lower)
        total_dir = pos_score + neg_score
        direction = float((pos_score - neg_score) / max(total_dir, 1.0))
        direction = float(np.clip(direction, -1.0, 1.0))

        # Magnitude (proxy: count of financial keywords + number mentions)
        n_numbers = len(re.findall(r'\d+(?:\.\d+)?%?', text))
        magnitude = min(0.2 + min(n_numbers, 10) * 0.05 + relevance * 0.3, 1.0)

        # Event type
        event_type = "other"
        best_count = 0
        for etype, keywords in EVENT_TYPE_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                event_type = etype

        # Affected sectors
        affected_sectors = []
        for sector, keywords in SECTOR_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                affected_sectors.append(sector)

        # Tickers
        tickers = extract_tickers(text)

        return ArticleScore(
            text=text[:200],
            relevance=float(relevance),
            urgency=float(urgency),
            direction=float(direction),
            magnitude=float(magnitude),
            confidence=0.5,  # lower confidence for rule-based
            affected_tickers=tickers,
            affected_sectors=affected_sectors[:3],
            event_type=event_type,
            model_used="rule_based",
        )


# ---------------------------------------------------------------------------
# LLM API clients
# ---------------------------------------------------------------------------

def _call_openai(
    prompt: str,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Optional[str]:
    """Call OpenAI API."""
    try:
        import urllib.request as request_mod
        import json as json_mod

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = json_mod.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")

        req = request_mod.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with request_mod.urlopen(req, timeout=timeout) as resp:
            data = json_mod.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None


def _call_anthropic(
    prompt: str,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Optional[str]:
    """Call Anthropic API."""
    try:
        import urllib.request as request_mod
        import json as json_mod

        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        body = json_mod.dumps({
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")

        req = request_mod.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers=headers,
            method="POST",
        )
        with request_mod.urlopen(req, timeout=timeout) as resp:
            data = json_mod.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]
    except Exception as e:
        logger.error(f"Anthropic API error: {e}")
        return None


# ---------------------------------------------------------------------------
# LLM Scorer
# ---------------------------------------------------------------------------

class LLMScorer:
    """
    LLM-based news scorer.

    Uses GPT-4/Claude to score articles for financial relevance and impact.
    Falls back to rule-based scoring when API unavailable.
    """

    def __init__(self, config: Optional[LLMScorerConfig] = None):
        self.config = config or LLMScorerConfig()
        self._rule_scorer = RuleBasedScorer()
        self._cache: Dict[str, ArticleScore] = {}
        self._cache_dir = self.config.cache_dir
        if self.config.use_cache:
            os.makedirs(self._cache_dir, exist_ok=True)
        self._n_api_calls = 0
        self._n_errors = 0

        # Get API key
        if not self.config.api_key:
            if self.config.backend == "openai":
                self.config.api_key = os.environ.get("OPENAI_API_KEY", "")
            elif self.config.backend == "anthropic":
                self.config.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def _cache_key(self, text: str) -> str:
        return f"{self.config.model}_{hashlib_md5(text[:300])}"

    def _load_cache(self, key: str) -> Optional[ArticleScore]:
        if not self.config.use_cache:
            return None
        if key in self._cache:
            score = self._cache[key]
            score.from_cache = True
            return score
        cache_file = os.path.join(self._cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                score = ArticleScore(**{
                    k: v for k, v in data.items()
                    if k in ArticleScore.__dataclass_fields__
                })
                score.from_cache = True
                self._cache[key] = score
                return score
            except Exception:
                pass
        return None

    def _save_cache(self, key: str, score: ArticleScore) -> None:
        if not self.config.use_cache:
            return
        self._cache[key] = score
        cache_file = os.path.join(self._cache_dir, f"{key}.json")
        try:
            with open(cache_file, "w") as f:
                json.dump({
                    "text": score.text,
                    "relevance": score.relevance,
                    "urgency": score.urgency,
                    "direction": score.direction,
                    "magnitude": score.magnitude,
                    "confidence": score.confidence,
                    "affected_tickers": score.affected_tickers,
                    "affected_sectors": score.affected_sectors,
                    "event_type": score.event_type,
                    "key_facts": score.key_facts,
                    "summary": score.summary,
                    "model_used": score.model_used,
                }, f)
        except Exception:
            pass

    def score_article(self, text: str) -> ArticleScore:
        """Score a single article."""
        key = self._cache_key(text)
        cached = self._load_cache(key)
        if cached:
            return cached

        if self.config.backend == "rule_based" or not self.config.api_key:
            score = self._rule_scorer.score(text)
        else:
            score = self._score_with_llm(text)

        self._save_cache(key, score)
        return score

    def score_batch(self, texts: List[str]) -> List[ArticleScore]:
        """Score a batch of articles."""
        results = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i: i + self.config.batch_size]

            if self.config.backend != "rule_based" and self.config.api_key:
                # Score batch in single LLM call
                batch_scores = self._score_batch_with_llm(batch)
            else:
                batch_scores = [self._rule_scorer.score(t) for t in batch]

            results.extend(batch_scores)

        return results

    def _score_with_llm(self, text: str) -> ArticleScore:
        """Score a single article with LLM."""
        few_shot = FEW_SHOT_SECTION if self.config.use_few_shot else ""
        prompt = SCORING_TEMPLATE.format(text=text[:800], few_shot_section=few_shot)

        response = self._call_llm(prompt)
        if not response:
            return self._rule_scorer.score(text)

        return self._parse_llm_response(text, response)

    def _score_batch_with_llm(self, texts: List[str]) -> List[ArticleScore]:
        """Score multiple articles in one LLM call."""
        articles_formatted = "\n".join([
            f"ARTICLE {i+1}: {t[:400]}" for i, t in enumerate(texts)
        ])

        prompt = f"""Score each of these {len(texts)} financial articles.
Respond with a JSON array of {len(texts)} scoring objects.
Each object must have: relevance, urgency, direction, magnitude, confidence, event_type, affected_tickers, affected_sectors, summary.

{articles_formatted}

JSON array response:"""

        response = self._call_llm(prompt)
        if not response:
            return [self._rule_scorer.score(t) for t in texts]

        try:
            # Find JSON array in response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                scores_data = json.loads(json_match.group())
                results = []
                for i, (text, sd) in enumerate(zip(texts, scores_data)):
                    results.append(self._dict_to_score(text, sd))
                return results
        except Exception as e:
            logger.debug(f"Batch LLM parse error: {e}")

        return [self._rule_scorer.score(t) for t in texts]

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the configured LLM backend."""
        for attempt in range(self.config.max_retries):
            try:
                if self.config.backend == "openai":
                    response = _call_openai(
                        prompt, self.config.model, self.config.api_key,
                        self.config.max_tokens, self.config.temperature, self.config.timeout
                    )
                elif self.config.backend == "anthropic":
                    response = _call_anthropic(
                        prompt, self.config.model, self.config.api_key,
                        self.config.max_tokens, self.config.temperature, self.config.timeout
                    )
                else:
                    return None

                if response:
                    self._n_api_calls += 1
                    return response

            except Exception as e:
                logger.warning(f"LLM call error (attempt {attempt+1}): {e}")
                self._n_errors += 1
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))

        return None

    def _parse_llm_response(self, text: str, response: str) -> ArticleScore:
        """Parse LLM JSON response into ArticleScore."""
        # Find JSON in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            return self._rule_scorer.score(text)

        try:
            data = json.loads(json_match.group())
            return self._dict_to_score(text, data)
        except json.JSONDecodeError:
            return self._rule_scorer.score(text)

    def _dict_to_score(self, text: str, data: Dict) -> ArticleScore:
        """Convert dict to ArticleScore with validation."""
        def clamp(v, lo, hi):
            try:
                return float(max(lo, min(hi, float(v))))
            except (TypeError, ValueError):
                return 0.5

        return ArticleScore(
            text=text[:200],
            relevance=clamp(data.get("relevance", 0.5), 0, 1),
            urgency=clamp(data.get("urgency", 0.5), 0, 1),
            direction=clamp(data.get("direction", 0.0), -1, 1),
            magnitude=clamp(data.get("magnitude", 0.3), 0, 1),
            confidence=clamp(data.get("confidence", 0.7), 0, 1),
            affected_tickers=data.get("affected_tickers", []),
            affected_sectors=data.get("affected_sectors", []),
            event_type=str(data.get("event_type", "other")),
            key_facts=data.get("key_facts", []),
            summary=str(data.get("summary", "")),
            model_used=self.config.model,
            raw_response=str(data),
        )

    def get_stats(self) -> Dict[str, int]:
        return {"api_calls": self._n_api_calls, "errors": self._n_errors}


def hashlib_md5(s: str) -> str:
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Prompt-based market impact predictor
# ---------------------------------------------------------------------------

class MarketImpactPredictor:
    """
    Extends LLMScorer with more detailed market impact predictions.
    Estimates: affected instruments, timeline, magnitude by sector.
    """

    IMPACT_PROMPT = """
You are a quantitative analyst. Given this financial news, predict the short-term market impact.

NEWS: {text}

Predict impacts in JSON format:
{{
  "equity_impact": <-1 to 1>,
  "bonds_impact": <-1 to 1>,
  "fx_impact": <-1 to 1>,
  "commodities_impact": <-1 to 1>,
  "vix_impact": <-1 to 1>,
  "impact_horizon": "<1h|4h|1d|1w>",
  "primary_movers": [<list of tickers most affected>],
  "sector_impacts": {{<sector>: <-1 to 1>}},
  "rationale": "<brief explanation>"
}}"""

    def __init__(self, scorer: Optional[LLMScorer] = None):
        self.scorer = scorer or LLMScorer()

    def predict_impact(self, text: str) -> Dict[str, Any]:
        """Predict detailed market impact for a news article."""
        base_score = self.scorer.score_article(text)

        # If rule-based, synthesize impact estimates
        if base_score.model_used == "rule_based":
            return self._synthesize_impact(base_score)

        # Try LLM for detailed impact
        if self.scorer.config.api_key:
            prompt = self.IMPACT_PROMPT.format(text=text[:600])
            response = self.scorer._call_llm(prompt)
            if response:
                try:
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                except Exception:
                    pass

        return self._synthesize_impact(base_score)

    def _synthesize_impact(self, score: ArticleScore) -> Dict[str, Any]:
        """Synthesize impact estimates from base score."""
        d = score.direction
        m = score.magnitude

        # Simple sector-based impact mapping
        event_impacts = {
            "earnings": {"equity_impact": d * m, "vix_impact": -d * 0.3},
            "macro":    {"equity_impact": d * m * 0.5, "bonds_impact": d * m * 0.8, "vix_impact": -d * 0.5},
            "ma":       {"equity_impact": d * m * 0.7},
            "analyst":  {"equity_impact": d * m * 0.6},
            "other":    {"equity_impact": d * m * 0.3},
        }

        impacts = event_impacts.get(score.event_type, event_impacts["other"])

        return {
            "equity_impact": float(np.clip(impacts.get("equity_impact", d * 0.3), -1, 1)),
            "bonds_impact": float(np.clip(impacts.get("bonds_impact", -d * 0.1), -1, 1)),
            "fx_impact": float(np.clip(impacts.get("fx_impact", 0.0), -1, 1)),
            "commodities_impact": float(np.clip(impacts.get("commodities_impact", 0.0), -1, 1)),
            "vix_impact": float(np.clip(impacts.get("vix_impact", -d * 0.2), -1, 1)),
            "impact_horizon": "1d" if score.urgency > 0.7 else "1w",
            "primary_movers": score.affected_tickers,
            "sector_impacts": {s: float(d * 0.5) for s in score.affected_sectors},
            "rationale": f"Rule-based synthesis from {score.event_type} event score",
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing LLM scorer (rule-based mode)...")

    config = LLMScorerConfig(backend="rule_based", use_cache=False)
    scorer = LLMScorer(config)
    predictor = MarketImpactPredictor(scorer)

    texts = [
        "Apple Inc. (AAPL) Q3 earnings beat expectations: EPS $1.52 vs $1.45 estimate. Revenue $89.5B, up 8% YoY. Company raises Q4 guidance to $92-95B.",
        "Federal Reserve raises interest rates 25bps. Chair Powell signals two additional hikes possible in 2025 if inflation persists.",
        "Microsoft to acquire cybersecurity firm for $13.5 billion in all-cash deal. Regulators expected to review.",
        "Goldman Sachs downgrades Tesla to Sell from Neutral. New target price $180, down from $250.",
    ]

    print("\nSingle article scoring:")
    for t in texts:
        score = scorer.score_article(t)
        print(f"  [{score.event_type:10s}] rel={score.relevance:.2f} dir={score.direction:+.2f} "
              f"mag={score.magnitude:.2f} | {t[:60]}...")

    print("\nBatch scoring:")
    batch = scorer.score_batch(texts)
    for score in batch:
        print(f"  Impact={score.impact_score:+.3f} actionable={score.is_actionable}")

    print("\nMarket impact prediction:")
    impact = predictor.predict_impact(texts[0])
    print(f"  {impact}")

    print(f"\nStats: {scorer.get_stats()}")
    print("LLM scorer self-test passed.")
