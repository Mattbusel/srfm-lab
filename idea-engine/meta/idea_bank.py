"""
idea_bank.py — Persistent idea storage, retrieval, and ranking system.

Features:
  - IdeaBank class backed by in-memory store with JSON persistence
  - Idea schema: id, hypothesis, signals, regime, confidence, tags, created_at, performance_log
  - Full-text search over idea descriptions
  - Tag-based filtering
  - Performance attribution: tracks which ideas generated PnL
  - Idea similarity via cosine similarity of feature vectors
  - Surfacing stale ideas when regime returns
  - Top-N ideas by recency, confidence, performance
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PerformanceEntry:
    timestamp: float
    pnl: float
    sharpe: float
    max_drawdown: float
    holding_days: float
    regime_at_entry: str
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "pnl": round(self.pnl, 6),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "holding_days": round(self.holding_days, 2),
            "regime_at_entry": self.regime_at_entry,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PerformanceEntry":
        return cls(**d)


@dataclass
class Idea:
    """Core idea schema."""
    id: str
    hypothesis: str                        # natural language hypothesis
    signals: List[str]                     # list of signal names / descriptors
    regime: str                            # intended regime for the idea
    confidence: float                      # initial confidence [0, 1]
    tags: List[str]                        # free-form tags
    created_at: float                      # Unix timestamp
    updated_at: float
    ticker: Optional[str] = None
    direction: str = "long"                # "long" | "short" | "neutral"
    feature_vector: Optional[List[float]] = None  # numeric embedding for similarity
    performance_log: List[PerformanceEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True                    # False = archived

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    def staleness_days(self) -> float:
        return (time.time() - self.updated_at) / 86400.0

    def cumulative_pnl(self) -> float:
        return sum(p.pnl for p in self.performance_log)

    def avg_sharpe(self) -> float:
        if not self.performance_log:
            return 0.0
        return float(np.mean([p.sharpe for p in self.performance_log]))

    def win_rate(self) -> float:
        if not self.performance_log:
            return 0.0
        return float(np.mean([1.0 if p.pnl > 0 else 0.0 for p in self.performance_log]))

    def times_used(self) -> int:
        return len(self.performance_log)

    def last_pnl(self) -> float:
        if not self.performance_log:
            return 0.0
        return self.performance_log[-1].pnl

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "signals": self.signals,
            "regime": self.regime,
            "confidence": round(self.confidence, 4),
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ticker": self.ticker,
            "direction": self.direction,
            "feature_vector": self.feature_vector,
            "performance_log": [p.to_dict() for p in self.performance_log],
            "metadata": self.metadata,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Idea":
        perf = [PerformanceEntry.from_dict(p) for p in d.get("performance_log", [])]
        return cls(
            id=d["id"],
            hypothesis=d["hypothesis"],
            signals=d.get("signals", []),
            regime=d.get("regime", "unknown"),
            confidence=d.get("confidence", 0.5),
            tags=d.get("tags", []),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            ticker=d.get("ticker"),
            direction=d.get("direction", "long"),
            feature_vector=d.get("feature_vector"),
            performance_log=perf,
            metadata=d.get("metadata", {}),
            active=d.get("active", True),
        )


# ---------------------------------------------------------------------------
# Text search utilities
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Lower-case tokenisation with basic stopword removal."""
    STOPWORDS = {"a", "an", "the", "is", "in", "on", "at", "to", "of", "and",
                 "or", "for", "this", "that", "with", "as", "by", "are", "it"}
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _tfidf_vector(tokens: List[str], vocab: Dict[str, int], idf: Dict[str, float]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    for t, cnt in freq.items():
        if t in vocab:
            tf = cnt / (len(tokens) + 1e-9)
            vec[vocab[t]] = tf * idf.get(t, 1.0)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------------------
# IdeaBank
# ---------------------------------------------------------------------------

class IdeaBank:
    """
    Persistent idea store with search, ranking, similarity, and performance tracking.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self._ideas: Dict[str, Idea] = {}

        # TF-IDF index
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_vectors: Dict[str, np.ndarray] = {}

        if persist_path and os.path.exists(persist_path):
            self._load(persist_path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        hypothesis: str,
        signals: List[str],
        regime: str,
        confidence: float,
        tags: Optional[List[str]] = None,
        ticker: Optional[str] = None,
        direction: str = "long",
        feature_vector: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
        idea_id: Optional[str] = None,
    ) -> Idea:
        """Add a new idea and return it."""
        now = time.time()
        idea = Idea(
            id=idea_id or str(uuid.uuid4()),
            hypothesis=hypothesis,
            signals=signals,
            regime=regime,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            tags=tags or [],
            created_at=now,
            updated_at=now,
            ticker=ticker,
            direction=direction,
            feature_vector=feature_vector,
            metadata=metadata or {},
        )
        self._ideas[idea.id] = idea
        self._rebuild_index()
        return idea

    def get(self, idea_id: str) -> Optional[Idea]:
        return self._ideas.get(idea_id)

    def update(self, idea_id: str, **kwargs) -> Optional[Idea]:
        idea = self._ideas.get(idea_id)
        if idea is None:
            return None
        for k, v in kwargs.items():
            if hasattr(idea, k):
                setattr(idea, k, v)
        idea.updated_at = time.time()
        self._rebuild_index()
        return idea

    def archive(self, idea_id: str) -> bool:
        idea = self._ideas.get(idea_id)
        if idea:
            idea.active = False
            idea.updated_at = time.time()
            return True
        return False

    def delete(self, idea_id: str) -> bool:
        if idea_id in self._ideas:
            del self._ideas[idea_id]
            self._rebuild_index()
            return True
        return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10, active_only: bool = True) -> List[Tuple[Idea, float]]:
        """Full-text search with TF-IDF scoring. Returns (idea, score) pairs."""
        if not self._vocab:
            return []

        q_tokens = _tokenise(query)
        if not q_tokens:
            return []

        q_vec = _tfidf_vector(q_tokens, self._vocab, self._idf)
        results: List[Tuple[Idea, float]] = []

        for idea_id, doc_vec in self._doc_vectors.items():
            idea = self._ideas.get(idea_id)
            if idea is None:
                continue
            if active_only and not idea.active:
                continue
            score = _cosine(q_vec, doc_vec)
            if score > 0:
                results.append((idea, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def filter_by_tags(self, tags: List[str], match_all: bool = False, active_only: bool = True) -> List[Idea]:
        """Filter ideas by tags. match_all=True requires all tags to be present."""
        tag_set = set(t.lower() for t in tags)
        result = []
        for idea in self._ideas.values():
            if active_only and not idea.active:
                continue
            idea_tags = set(t.lower() for t in idea.tags)
            if match_all:
                if tag_set.issubset(idea_tags):
                    result.append(idea)
            else:
                if tag_set & idea_tags:
                    result.append(idea)
        return result

    def filter_by_regime(self, regime: str, active_only: bool = True) -> List[Idea]:
        """Return ideas matching a regime."""
        return [
            i for i in self._ideas.values()
            if i.regime.lower() == regime.lower() and (not active_only or i.active)
        ]

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    def log_performance(
        self,
        idea_id: str,
        pnl: float,
        sharpe: float,
        max_drawdown: float,
        holding_days: float,
        regime_at_entry: str = "unknown",
        notes: str = "",
    ) -> Optional[PerformanceEntry]:
        idea = self._ideas.get(idea_id)
        if idea is None:
            return None
        entry = PerformanceEntry(
            timestamp=time.time(),
            pnl=pnl,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            holding_days=holding_days,
            regime_at_entry=regime_at_entry,
            notes=notes,
        )
        idea.performance_log.append(entry)
        idea.updated_at = time.time()
        # Update confidence based on recent performance
        self._update_confidence(idea)
        return entry

    def _update_confidence(self, idea: Idea, lookback: int = 5) -> None:
        """Bayesian-like confidence update from recent performance."""
        recent = idea.performance_log[-lookback:]
        if not recent:
            return
        win_rate = float(np.mean([1.0 if p.pnl > 0 else 0.0 for p in recent]))
        avg_sharpe = float(np.mean([p.sharpe for p in recent]))
        # Blend prior confidence with performance signal
        perf_signal = 0.5 * win_rate + 0.5 * float(np.clip(avg_sharpe / 3.0, 0.0, 1.0))
        alpha = min(0.3, len(recent) / 20.0)  # learning rate grows with data
        idea.confidence = float(np.clip((1 - alpha) * idea.confidence + alpha * perf_signal, 0.0, 1.0))

    def performance_attribution(self) -> List[Dict]:
        """Rank ideas by realised PnL contribution."""
        rows = []
        for idea in self._ideas.values():
            if idea.performance_log:
                rows.append({
                    "id": idea.id,
                    "hypothesis": idea.hypothesis[:60],
                    "ticker": idea.ticker,
                    "times_used": idea.times_used(),
                    "cumulative_pnl": round(idea.cumulative_pnl(), 6),
                    "avg_sharpe": round(idea.avg_sharpe(), 3),
                    "win_rate": round(idea.win_rate(), 3),
                    "last_pnl": round(idea.last_pnl(), 6),
                })
        rows.sort(key=lambda r: r["cumulative_pnl"], reverse=True)
        return rows

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def similar(self, idea_id: str, top_k: int = 5) -> List[Tuple[Idea, float]]:
        """Find ideas most similar to the given idea using cosine similarity of feature vectors."""
        source = self._ideas.get(idea_id)
        if source is None or source.feature_vector is None:
            # Fall back to text similarity
            return self.search(source.hypothesis if source else "", top_k=top_k + 1)

        src_vec = np.array(source.feature_vector, dtype=np.float32)
        src_norm = np.linalg.norm(src_vec)
        if src_norm < 1e-9:
            return []

        results: List[Tuple[Idea, float]] = []
        for other_id, other in self._ideas.items():
            if other_id == idea_id or other.feature_vector is None:
                continue
            ov = np.array(other.feature_vector, dtype=np.float32)
            sim = _cosine(src_vec, ov)
            results.append((other, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def deduplicate(self, threshold: float = 0.90) -> List[Tuple[str, str, float]]:
        """
        Find near-duplicate ideas (cosine similarity > threshold).
        Returns list of (id_a, id_b, similarity) pairs.
        """
        ids = [i for i in self._ideas if self._ideas[i].feature_vector is not None]
        dupes = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                va = np.array(self._ideas[ids[i]].feature_vector, dtype=np.float32)
                vb = np.array(self._ideas[ids[j]].feature_vector, dtype=np.float32)
                sim = _cosine(va, vb)
                if sim >= threshold:
                    dupes.append((ids[i], ids[j], round(sim, 4)))
        return sorted(dupes, key=lambda x: x[2], reverse=True)

    # ------------------------------------------------------------------
    # Regime-based surfacing
    # ------------------------------------------------------------------

    def surface_for_regime(self, current_regime: str, min_confidence: float = 0.4) -> List[Idea]:
        """
        Surface ideas that were created for this regime and have past performance,
        or are not yet tested. Useful for idea recycling when regime returns.
        """
        candidates = self.filter_by_regime(current_regime)
        # Also include ideas tagged with the regime
        tagged = self.filter_by_tags([current_regime])
        all_ideas = {i.id: i for i in candidates + tagged}

        result = []
        for idea in all_ideas.values():
            if idea.confidence < min_confidence:
                continue
            # Boost confidence for ideas with good prior performance in this regime
            prior_perf = [p for p in idea.performance_log if p.regime_at_entry == current_regime]
            if prior_perf:
                avg_pnl = float(np.mean([p.pnl for p in prior_perf]))
                if avg_pnl > 0:
                    result.append(idea)
            else:
                result.append(idea)  # untested but regime-matched

        result.sort(key=lambda i: i.confidence, reverse=True)
        return result

    # ------------------------------------------------------------------
    # Top-N rankings
    # ------------------------------------------------------------------

    def top_by_recency(self, n: int = 10, active_only: bool = True) -> List[Idea]:
        ideas = [i for i in self._ideas.values() if not active_only or i.active]
        return sorted(ideas, key=lambda i: i.created_at, reverse=True)[:n]

    def top_by_confidence(self, n: int = 10, active_only: bool = True) -> List[Idea]:
        ideas = [i for i in self._ideas.values() if not active_only or i.active]
        return sorted(ideas, key=lambda i: i.confidence, reverse=True)[:n]

    def top_by_performance(self, n: int = 10, metric: str = "pnl") -> List[Idea]:
        ideas = [i for i in self._ideas.values() if i.performance_log]
        if metric == "pnl":
            ideas.sort(key=lambda i: i.cumulative_pnl(), reverse=True)
        elif metric == "sharpe":
            ideas.sort(key=lambda i: i.avg_sharpe(), reverse=True)
        elif metric == "win_rate":
            ideas.sort(key=lambda i: i.win_rate(), reverse=True)
        return ideas[:n]

    def recently_stale(self, min_days: float = 7.0, max_days: float = 60.0) -> List[Idea]:
        """Ideas that are stale but not so old they are irrelevant."""
        return [
            i for i in self._ideas.values()
            if i.active and min_days <= i.staleness_days() <= max_days
        ]

    # ------------------------------------------------------------------
    # TF-IDF index
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Rebuild TF-IDF vocabulary and document vectors from all active ideas."""
        corpus: Dict[str, List[str]] = {}
        for idea in self._ideas.values():
            doc = " ".join([idea.hypothesis] + idea.signals + idea.tags)
            corpus[idea.id] = _tokenise(doc)

        # Build vocabulary from document frequency
        df: Dict[str, int] = {}
        for tokens in corpus.values():
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        N = max(len(corpus), 1)
        self._vocab = {t: i for i, t in enumerate(sorted(df.keys()))}
        self._idf = {t: math.log((N + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()}

        self._doc_vectors = {}
        for idea_id, tokens in corpus.items():
            self._doc_vectors[idea_id] = _tfidf_vector(tokens, self._vocab, self._idf)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        target = path or self.persist_path
        if not target:
            raise ValueError("No persist path specified.")
        data = {
            "version": 1,
            "saved_at": time.time(),
            "ideas": [i.to_dict() for i in self._ideas.values()],
        }
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for d in data.get("ideas", []):
            idea = Idea.from_dict(d)
            self._ideas[idea.id] = idea
        self._rebuild_index()

    def load(self, path: Optional[str] = None) -> None:
        target = path or self.persist_path
        if not target:
            raise ValueError("No persist path specified.")
        self._load(target)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        ideas = list(self._ideas.values())
        active = [i for i in ideas if i.active]
        with_perf = [i for i in ideas if i.performance_log]
        return {
            "total": len(ideas),
            "active": len(active),
            "with_performance": len(with_perf),
            "regimes": list(set(i.regime for i in active)),
            "tags": list(set(t for i in active for t in i.tags)),
            "avg_confidence": round(float(np.mean([i.confidence for i in active])) if active else 0.0, 4),
            "top_performers": [
                {"id": i.id, "ticker": i.ticker, "pnl": round(i.cumulative_pnl(), 4)}
                for i in self.top_by_performance(3)
            ],
        }

    def __len__(self) -> int:
        return len(self._ideas)

    def __repr__(self) -> str:
        return f"IdeaBank(ideas={len(self._ideas)}, path={self.persist_path!r})"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bank = IdeaBank()

    # Populate
    bank.add(
        hypothesis="Mean reversion on S&P 500 after vol spike — sell short when VIX >25 and SPY oversold",
        signals=["VIX_spike", "RSI_oversold", "put_call_ratio"],
        regime="volatile",
        confidence=0.72,
        tags=["mean_reversion", "equities", "vol_regime"],
        ticker="SPY",
        direction="short",
        feature_vector=list(np.random.randn(32)),
    )
    bank.add(
        hypothesis="Tech momentum: buy NASDAQ laggards with strong earnings revision in bull regime",
        signals=["earnings_revision", "price_momentum_12m", "analyst_upgrades"],
        regime="bull",
        confidence=0.65,
        tags=["momentum", "tech", "earnings"],
        ticker="QQQ",
        direction="long",
        feature_vector=list(np.random.randn(32)),
    )
    bank.add(
        hypothesis="Credit spread widening predicts equity volatility 2 weeks ahead in risk-off regime",
        signals=["HY_spread_change", "IG_spread_change", "VIX_term_structure"],
        regime="risk_off",
        confidence=0.58,
        tags=["cross_asset", "credit", "vol_prediction"],
        feature_vector=list(np.random.randn(32)),
    )

    # Log performance
    first_id = bank.top_by_confidence(1)[0].id
    bank.log_performance(first_id, pnl=0.023, sharpe=1.8, max_drawdown=0.04,
                         holding_days=5, regime_at_entry="volatile")

    # Search
    results = bank.search("vol spike mean reversion", top_k=3)
    print("Search results:")
    for idea, score in results:
        print(f"  [{score:.3f}] {idea.hypothesis[:60]}")

    # Top by confidence
    print("\nTop by confidence:")
    for i in bank.top_by_confidence(3):
        print(f"  {i.ticker} | {i.regime} | conf={i.confidence:.3f} | {i.hypothesis[:50]}")

    # Attribution
    print("\nPerformance attribution:", bank.performance_attribution())

    # Summary
    print("\nSummary:", json.dumps(bank.summary(), indent=2))
