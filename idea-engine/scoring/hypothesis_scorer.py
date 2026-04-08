"""
hypothesis_scorer.py
--------------------
Bayesian multi-factor hypothesis scorer for the idea-engine.

Maintains prior beliefs about trading hypotheses and updates them via evidence
(backtest results, live performance, regime alignment). Supports score
decomposition, confidence intervals, and JSON persistence.
"""

from __future__ import annotations

import json
import math
import statistics
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

FACTOR_NAMES = [
    "regime_alignment",
    "signal_quality",
    "risk_reward",
    "uniqueness",
    "timing",
]

DEFAULT_FACTOR_WEIGHTS: Dict[str, float] = {
    "regime_alignment": 0.25,
    "signal_quality":   0.30,
    "risk_reward":      0.20,
    "uniqueness":       0.15,
    "timing":           0.10,
}


@dataclass
class FactorScore:
    """Score for a single factor in [0, 1]."""
    name: str
    raw: float          # raw evidence value
    score: float        # normalised to [0, 1]
    confidence: float   # confidence in this factor score [0, 1]
    evidence_count: int = 0


@dataclass
class EvidenceRecord:
    """Single piece of evidence added to a hypothesis."""
    timestamp: float
    sharpe: Optional[float] = None
    ic: Optional[float] = None                     # information coefficient
    regime_match: Optional[float] = None           # 1 = perfect regime match
    max_drawdown: Optional[float] = None           # positive fraction
    win_rate: Optional[float] = None
    correlation_to_existing: Optional[float] = None  # 0 = unique, 1 = duplicate
    live_days: Optional[int] = None
    notes: str = ""


@dataclass
class HypothesisState:
    """Full state of one hypothesis."""
    hypothesis_id: str
    name: str
    description: str
    created_at: float
    factor_scores: Dict[str, FactorScore] = field(default_factory=dict)
    evidence_history: List[EvidenceRecord] = field(default_factory=list)
    composite_score: float = 0.5
    composite_confidence: float = 0.0
    prior_alpha: float = 1.0   # Beta distribution parameter
    prior_beta: float = 1.0
    posterior_alpha: float = 1.0
    posterior_beta: float = 1.0
    last_updated: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    archived: bool = False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float, k: float = 5.0) -> float:
    """Smooth mapping from real line to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-k * x))


def _beta_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def _beta_variance(alpha: float, beta: float) -> float:
    ab = alpha + beta
    return (alpha * beta) / (ab * ab * (ab + 1.0))


def _beta_ci(alpha: float, beta: float, z: float = 1.96) -> Tuple[float, float]:
    """Approximate CI for Beta distribution using normal approximation."""
    mu = _beta_mean(alpha, beta)
    var = _beta_variance(alpha, beta)
    sd = math.sqrt(max(var, 1e-9))
    lo = max(0.0, mu - z * sd)
    hi = min(1.0, mu + z * sd)
    return lo, hi


def _sharpe_to_score(sharpe: float) -> float:
    """Map Sharpe ratio to [0, 1] score."""
    # Sharpe ≥ 2 → ~0.95, Sharpe = 0 → 0.5, Sharpe ≤ -2 → ~0.05
    return _sigmoid(sharpe / 2.0)


def _drawdown_to_score(max_dd: float) -> float:
    """Map max drawdown fraction (0 to 1) to quality score [0,1]."""
    # 0 dd → 1.0, 0.5 dd → ~0.37
    return math.exp(-3.0 * max_dd)


def _ic_to_score(ic: float) -> float:
    """Map IC (typically -0.2 to 0.2) to [0, 1]."""
    return _sigmoid(ic / 0.1)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class BayesianHypothesisScorer:
    """
    Maintains and updates Bayesian scores for a collection of hypotheses.

    Each hypothesis has:
      - A Beta(alpha, beta) posterior over its "quality" probability.
      - Five factor sub-scores that are combined via weighted average.
      - Evidence history with exponential time decay.

    Usage
    -----
    scorer = BayesianHypothesisScorer()
    hid = scorer.add_hypothesis("Mean reversion on SPY", "...")
    scorer.add_evidence(hid, EvidenceRecord(timestamp=time.time(), sharpe=1.4, ...))
    print(scorer.rank_hypotheses())
    """

    def __init__(
        self,
        factor_weights: Optional[Dict[str, float]] = None,
        decay_halflife_days: float = 90.0,
        min_evidence_for_confidence: int = 3,
        persistence_path: Optional[str] = None,
    ):
        self.factor_weights: Dict[str, float] = (
            factor_weights or dict(DEFAULT_FACTOR_WEIGHTS)
        )
        self._normalize_weights()
        self.decay_halflife_days = decay_halflife_days
        self.decay_lambda = math.log(2.0) / (decay_halflife_days * 86400.0)
        self.min_evidence = min_evidence_for_confidence
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self._hypotheses: Dict[str, HypothesisState] = {}

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def _normalize_weights(self) -> None:
        total = sum(self.factor_weights.values())
        if total > 0:
            for k in self.factor_weights:
                self.factor_weights[k] /= total

    def set_factor_weight(self, factor: str, weight: float) -> None:
        if factor not in FACTOR_NAMES:
            raise ValueError(f"Unknown factor: {factor}. Valid: {FACTOR_NAMES}")
        self.factor_weights[factor] = max(0.0, weight)
        self._normalize_weights()

    # ------------------------------------------------------------------
    # Hypothesis lifecycle
    # ------------------------------------------------------------------

    def add_hypothesis(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        hypothesis_id: Optional[str] = None,
        prior_strength: float = 1.0,
    ) -> str:
        """Register a new hypothesis. Returns its ID."""
        hid = hypothesis_id or str(uuid.uuid4())[:8]
        state = HypothesisState(
            hypothesis_id=hid,
            name=name,
            description=description,
            created_at=time.time(),
            prior_alpha=prior_strength,
            prior_beta=prior_strength,
            posterior_alpha=prior_strength,
            posterior_beta=prior_strength,
            tags=tags or [],
        )
        # Initialise factor scores at neutral 0.5
        for factor in FACTOR_NAMES:
            state.factor_scores[factor] = FactorScore(
                name=factor, raw=0.0, score=0.5, confidence=0.0
            )
        self._hypotheses[hid] = state
        return hid

    def archive_hypothesis(self, hypothesis_id: str) -> None:
        self._get(hypothesis_id).archived = True

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisState:
        return self._get(hypothesis_id)

    def list_hypotheses(self, include_archived: bool = False) -> List[str]:
        return [
            hid for hid, h in self._hypotheses.items()
            if include_archived or not h.archived
        ]

    # ------------------------------------------------------------------
    # Evidence ingestion
    # ------------------------------------------------------------------

    def add_evidence(
        self, hypothesis_id: str, evidence: EvidenceRecord
    ) -> HypothesisState:
        """Add one evidence record and re-compute scores."""
        state = self._get(hypothesis_id)
        state.evidence_history.append(evidence)
        self._update_factor_scores(state)
        self._update_bayesian_posterior(state)
        self._update_composite_score(state)
        state.last_updated = time.time()
        return state

    def _decay_weight(self, now: float, timestamp: float) -> float:
        """Exponential time-decay weight for a piece of evidence."""
        age_sec = max(0.0, now - timestamp)
        return math.exp(-self.decay_lambda * age_sec)

    def _update_factor_scores(self, state: HypothesisState) -> None:
        """Recompute each factor score using decay-weighted evidence."""
        now = time.time()
        accumulators: Dict[str, List[Tuple[float, float]]] = {
            f: [] for f in FACTOR_NAMES
        }

        for ev in state.evidence_history:
            w = self._decay_weight(now, ev.timestamp)
            if ev.sharpe is not None:
                accumulators["signal_quality"].append(
                    (_sharpe_to_score(ev.sharpe), w)
                )
                accumulators["risk_reward"].append(
                    (_sharpe_to_score(ev.sharpe * 0.8), w)
                )
            if ev.ic is not None:
                accumulators["signal_quality"].append(
                    (_ic_to_score(ev.ic), w * 0.5)
                )
            if ev.max_drawdown is not None:
                accumulators["risk_reward"].append(
                    (_drawdown_to_score(ev.max_drawdown), w)
                )
            if ev.win_rate is not None:
                accumulators["signal_quality"].append((ev.win_rate, w * 0.3))
            if ev.regime_match is not None:
                accumulators["regime_alignment"].append(
                    (max(0.0, min(1.0, ev.regime_match)), w)
                )
            if ev.correlation_to_existing is not None:
                uniqueness = 1.0 - abs(ev.correlation_to_existing)
                accumulators["uniqueness"].append((uniqueness, w))
            if ev.live_days is not None:
                # More live days → better timing confirmation
                timing_score = _sigmoid((ev.live_days - 30) / 30.0)
                accumulators["timing"].append((timing_score, w))

        for factor in FACTOR_NAMES:
            entries = accumulators[factor]
            fs = state.factor_scores[factor]
            if not entries:
                continue
            total_w = sum(wt for _, wt in entries)
            if total_w < 1e-9:
                continue
            weighted_mean = sum(sc * wt for sc, wt in entries) / total_w
            fs.score = max(0.0, min(1.0, weighted_mean))
            fs.raw = weighted_mean
            fs.evidence_count = len(entries)
            # Confidence grows with evidence count, capped at 1
            fs.confidence = min(
                1.0, len(entries) / max(1, self.min_evidence)
            )

    def _update_bayesian_posterior(self, state: HypothesisState) -> None:
        """
        Rolling Bayesian update on Beta(alpha, beta).

        For each evidence record we treat the signal_quality score as a
        pseudo-Bernoulli outcome and update with decay-weighted pseudo-counts.
        """
        now = time.time()
        pseudo_successes = 0.0
        pseudo_failures = 0.0

        for ev in state.evidence_history:
            w = self._decay_weight(now, ev.timestamp)
            if ev.sharpe is not None:
                p = _sharpe_to_score(ev.sharpe)
                pseudo_successes += w * p
                pseudo_failures += w * (1.0 - p)
            if ev.ic is not None:
                p = _ic_to_score(ev.ic)
                pseudo_successes += w * p * 0.5
                pseudo_failures += w * (1.0 - p) * 0.5

        state.posterior_alpha = state.prior_alpha + pseudo_successes
        state.posterior_beta = state.prior_beta + pseudo_failures

    def _update_composite_score(self, state: HypothesisState) -> None:
        """Weighted combination of factor scores, modulated by posterior."""
        factor_composite = sum(
            self.factor_weights.get(f, 0.0) * state.factor_scores[f].score
            for f in FACTOR_NAMES
        )
        # Posterior belief (probability that hypothesis is "good")
        posterior_belief = _beta_mean(
            state.posterior_alpha, state.posterior_beta
        )
        n_evidence = len(state.evidence_history)
        # Blend factor composite with posterior; trust posterior more as evidence accumulates
        blend = min(1.0, n_evidence / max(1, self.min_evidence * 2))
        state.composite_score = (
            (1.0 - blend) * factor_composite + blend * posterior_belief
        )
        avg_conf = statistics.mean(
            fs.confidence for fs in state.factor_scores.values()
        )
        state.composite_confidence = avg_conf

    # ------------------------------------------------------------------
    # Scoring API
    # ------------------------------------------------------------------

    def score(self, hypothesis_id: str) -> float:
        """Return composite score in [0, 1]."""
        return self._get(hypothesis_id).composite_score

    def confidence_interval(
        self, hypothesis_id: str, level: float = 0.95
    ) -> Tuple[float, float]:
        """Return CI around composite score using Beta posterior."""
        state = self._get(hypothesis_id)
        z = 1.96 if abs(level - 0.95) < 0.01 else 2.576
        return _beta_ci(state.posterior_alpha, state.posterior_beta, z)

    def decompose(self, hypothesis_id: str) -> Dict[str, Dict]:
        """Return per-factor score breakdown."""
        state = self._get(hypothesis_id)
        result = {}
        for f in FACTOR_NAMES:
            fs = state.factor_scores[f]
            weight = self.factor_weights.get(f, 0.0)
            result[f] = {
                "score": round(fs.score, 4),
                "confidence": round(fs.confidence, 4),
                "weight": round(weight, 4),
                "weighted_contribution": round(fs.score * weight, 4),
                "evidence_count": fs.evidence_count,
            }
        ci_lo, ci_hi = self.confidence_interval(hypothesis_id)
        result["_composite"] = {
            "score": round(state.composite_score, 4),
            "confidence": round(state.composite_confidence, 4),
            "ci_95_lo": round(ci_lo, 4),
            "ci_95_hi": round(ci_hi, 4),
            "posterior_alpha": round(state.posterior_alpha, 4),
            "posterior_beta": round(state.posterior_beta, 4),
        }
        return result

    def rank_hypotheses(
        self, include_archived: bool = False, top_n: Optional[int] = None
    ) -> List[Dict]:
        """Return hypotheses sorted by composite score descending."""
        results = []
        for hid in self.list_hypotheses(include_archived):
            state = self._hypotheses[hid]
            ci_lo, ci_hi = self.confidence_interval(hid)
            results.append({
                "hypothesis_id": hid,
                "name": state.name,
                "composite_score": round(state.composite_score, 4),
                "composite_confidence": round(state.composite_confidence, 4),
                "ci_95_lo": round(ci_lo, 4),
                "ci_95_hi": round(ci_hi, 4),
                "evidence_count": len(state.evidence_history),
                "tags": state.tags,
                "archived": state.archived,
            })
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results

    def sharpe_fitness(
        self, hypothesis_id: str, risk_free: float = 0.0
    ) -> float:
        """
        Sharpe-based fitness: estimate annualised Sharpe from evidence.
        Returns NaN if insufficient data.
        """
        state = self._get(hypothesis_id)
        sharpes = [
            ev.sharpe for ev in state.evidence_history if ev.sharpe is not None
        ]
        if len(sharpes) < 2:
            return float("nan")
        now = time.time()
        weights = [
            self._decay_weight(now, ev.timestamp)
            for ev in state.evidence_history
            if ev.sharpe is not None
        ]
        w_sum = sum(weights)
        w_mean = sum(s * w for s, w in zip(sharpes, weights)) / w_sum
        w_var = sum(
            w * (s - w_mean) ** 2 for s, w in zip(sharpes, weights)
        ) / w_sum
        if w_var < 1e-12:
            return w_mean - risk_free
        return (w_mean - risk_free) / math.sqrt(w_var)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Serialise all hypothesis states to JSON."""
        target = Path(path) if path else self.persistence_path
        if target is None:
            raise ValueError("No persistence path set.")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict = {
            "metadata": {
                "saved_at": time.time(),
                "factor_weights": self.factor_weights,
                "decay_halflife_days": self.decay_halflife_days,
            },
            "hypotheses": {},
        }
        for hid, state in self._hypotheses.items():
            d = asdict(state)
            payload["hypotheses"][hid] = d
        with open(target, "w") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "BayesianHypothesisScorer":
        """Restore scorer from JSON file."""
        with open(path) as fh:
            payload = json.load(fh)
        meta = payload.get("metadata", {})
        scorer = cls(
            factor_weights=meta.get("factor_weights"),
            decay_halflife_days=meta.get("decay_halflife_days", 90.0),
            persistence_path=path,
        )
        for hid, d in payload.get("hypotheses", {}).items():
            state = HypothesisState(
                hypothesis_id=d["hypothesis_id"],
                name=d["name"],
                description=d.get("description", ""),
                created_at=d.get("created_at", 0.0),
                prior_alpha=d.get("prior_alpha", 1.0),
                prior_beta=d.get("prior_beta", 1.0),
                posterior_alpha=d.get("posterior_alpha", 1.0),
                posterior_beta=d.get("posterior_beta", 1.0),
                composite_score=d.get("composite_score", 0.5),
                composite_confidence=d.get("composite_confidence", 0.0),
                tags=d.get("tags", []),
                archived=d.get("archived", False),
                last_updated=d.get("last_updated", 0.0),
            )
            for fname, fd in d.get("factor_scores", {}).items():
                state.factor_scores[fname] = FactorScore(**fd)
            for ev_d in d.get("evidence_history", []):
                state.evidence_history.append(EvidenceRecord(**ev_d))
            scorer._hypotheses[hid] = state
        return scorer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, hypothesis_id: str) -> HypothesisState:
        if hypothesis_id not in self._hypotheses:
            raise KeyError(f"Unknown hypothesis: {hypothesis_id}")
        return self._hypotheses[hypothesis_id]

    def __repr__(self) -> str:
        n = len(self._hypotheses)
        return f"BayesianHypothesisScorer(n_hypotheses={n})"


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scorer = BayesianHypothesisScorer(decay_halflife_days=30.0)

    h1 = scorer.add_hypothesis(
        "SPY mean-reversion on RSI oversold",
        description="Buy when RSI < 30, exit at 50",
        tags=["equity", "mean_reversion"],
    )
    h2 = scorer.add_hypothesis(
        "VIX spike short vol",
        description="Short VIX futures after spike > 30",
        tags=["volatility", "contrarian"],
    )

    now = time.time()
    # Add evidence for h1
    for i, (sharpe, ic, dd) in enumerate(
        [(1.1, 0.05, 0.12), (1.4, 0.07, 0.09), (0.9, 0.04, 0.15)]
    ):
        scorer.add_evidence(
            h1,
            EvidenceRecord(
                timestamp=now - (30 - i * 10) * 86400,
                sharpe=sharpe,
                ic=ic,
                max_drawdown=dd,
                regime_match=0.8,
                correlation_to_existing=0.15,
                live_days=20 + i * 5,
            ),
        )

    # Add evidence for h2 (weaker)
    scorer.add_evidence(
        h2,
        EvidenceRecord(
            timestamp=now - 10 * 86400,
            sharpe=0.6,
            ic=0.02,
            max_drawdown=0.25,
            regime_match=0.5,
        ),
    )

    print("=== Rankings ===")
    for row in scorer.rank_hypotheses():
        print(row)

    print("\n=== Decomposition for h1 ===")
    for k, v in scorer.decompose(h1).items():
        print(f"  {k}: {v}")

    print(f"\n=== Sharpe fitness h1: {scorer.sharpe_fitness(h1):.3f} ===")
    print(f"=== CI h1: {scorer.confidence_interval(h1)} ===")
