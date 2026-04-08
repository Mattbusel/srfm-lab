"""
idea_scorer.py
--------------
Multi-dimensional idea scoring engine.
Scores each new trading idea across novelty, feasibility, risk-adjusted return,
timing, cross-domain confirmation, regime fit, and capacity. Tracks historical
accuracy of similar scores and enforces per-dimension minimum thresholds.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


# ---------------------------------------------------------------------------
# Idea input descriptor
# ---------------------------------------------------------------------------

@dataclass
class IdeaInput:
    """
    Raw description of a trading idea provided by the caller.

    Fields
    ------
    feature_vector : list[float]
        Embedding / feature representation used for novelty computation.
    signal_strength : float
        Expected per-trade edge (e.g. 0.55 for a 55% win-rate edge).
    win_rate : float
        Expected fraction of profitable trades [0, 1].
    max_loss_per_trade : float
        Expected maximum loss on a losing trade (positive fraction, e.g. 0.02).
    data_complexity : float
        Complexity of data requirements [0, 1]; 0 = simple price data.
    liquidity_score : float
        Liquidity of the instrument(s) [0, 1]; 1 = highly liquid.
    setup_frequency_now : float
        How often the setup is firing right now (count or fraction).
    setup_base_rate : float
        Historical base rate of the setup per period.
    domain_signals : dict[str, float]
        {domain_name: agreement_score [0,1]} from independent domains.
    regime_label : str
        Current market regime.
    idea_regime_fit : dict[str, float]
        {regime_label: fit_score [0,1]} expressing how well the idea works
        in each regime.
    capacity_adv_fraction : float
        Max position as fraction of average daily volume [0, 1];
        lower = better capacity.
    name : str
        Human-readable idea name.
    description : str
    tags : list[str]
    metadata : dict
    """
    feature_vector: list[float]
    signal_strength: float
    win_rate: float
    max_loss_per_trade: float
    data_complexity: float
    liquidity_score: float
    setup_frequency_now: float
    setup_base_rate: float
    domain_signals: dict[str, float] = field(default_factory=dict)
    regime_label: str = "unknown"
    idea_regime_fit: dict[str, float] = field(default_factory=dict)
    capacity_adv_fraction: float = 0.01
    name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Composite IdeaScore
# ---------------------------------------------------------------------------

@dataclass
class IdeaScore:
    """All scoring dimensions and the composite rank for one idea."""
    id: str
    name: str
    timestamp: str

    # Dimension scores [0, 1]
    novelty: float             # 1 = completely novel
    feasibility: float         # 1 = easy to implement
    risk_adj_return: float     # normalised risk-adjusted expected return
    timing: float              # 1 = setup firing right now at peak rate
    cross_domain: float        # 1 = all domains agree
    regime_fit: float          # 1 = perfect fit with current regime
    capacity: float            # 1 = unlimited capacity

    # Raw values (not normalised)
    raw_expected_return: float
    raw_risk_adj_return: float
    raw_timing_ratio: float
    n_confirming_domains: int
    n_total_domains: int

    # Composite
    overall_score: float       # weighted average of dimension scores
    rank: Optional[int] = None # populated by IdeaScorer.rank_ideas()
    passed_thresholds: bool = True
    failed_dimensions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IdeaScore":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Historical accuracy record
# ---------------------------------------------------------------------------

@dataclass
class AccuracyRecord:
    """Maps a score bucket to observed outcome statistics."""
    score_bucket: float        # e.g. 0.7 means scores in [0.65, 0.75)
    n_ideas: int = 0
    n_winners: int = 0
    avg_return: float = 0.0
    avg_hold_days: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.n_winners / self.n_ideas if self.n_ideas else 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Idea Book (existing idea embeddings)
# ---------------------------------------------------------------------------

class IdeaBook:
    """
    Stores feature vectors of all previously scored / accepted ideas.
    Used to compute novelty via cosine similarity.
    """

    def __init__(self) -> None:
        self._vectors: list[np.ndarray] = []
        self._ids: list[str] = []
        self._names: list[str] = []

    def add(self, idea_id: str, name: str, vector: list[float]) -> None:
        self._vectors.append(np.asarray(vector, dtype=float))
        self._ids.append(idea_id)
        self._names.append(name)

    def max_similarity(self, vector: list[float]) -> float:
        """Return max cosine similarity to any idea in the book."""
        if not self._vectors:
            return 0.0
        q = np.asarray(vector, dtype=float)
        sims = [_cosine_sim(q, v) for v in self._vectors]
        return float(max(sims))

    def most_similar(self, vector: list[float], top_k: int = 3) -> list[tuple[str, str, float]]:
        """Return list of (id, name, similarity) for the top-k closest ideas."""
        if not self._vectors:
            return []
        q = np.asarray(vector, dtype=float)
        sims = [(self._ids[i], self._names[i], _cosine_sim(q, v))
                for i, v in enumerate(self._vectors)]
        sims.sort(key=lambda x: -x[2])
        return sims[:top_k]

    def __len__(self) -> int:
        return len(self._vectors)


# ---------------------------------------------------------------------------
# Main Scorer
# ---------------------------------------------------------------------------

class IdeaScorer:
    """
    Multi-dimensional scoring engine for trading ideas.

    Parameters
    ----------
    dimension_weights : dict[str, float]
        Weight assigned to each scoring dimension in the composite score.
    thresholds : dict[str, float]
        Minimum acceptable score per dimension. Ideas below any threshold
        are flagged (passed_thresholds=False) but still scored.
    return_normaliser : float
        Expected return value that maps to a risk_adj_return score of 1.0.
    n_accuracy_buckets : int
        Number of buckets for historical accuracy tracking (default 10).
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "novelty": 0.12,
        "feasibility": 0.14,
        "risk_adj_return": 0.22,
        "timing": 0.14,
        "cross_domain": 0.14,
        "regime_fit": 0.12,
        "capacity": 0.12,
    }

    DEFAULT_THRESHOLDS: dict[str, float] = {
        "novelty": 0.20,
        "feasibility": 0.30,
        "risk_adj_return": 0.10,
        "timing": 0.10,
        "cross_domain": 0.20,
        "regime_fit": 0.20,
        "capacity": 0.10,
    }

    def __init__(
        self,
        idea_book: Optional[IdeaBook] = None,
        dimension_weights: Optional[dict[str, float]] = None,
        thresholds: Optional[dict[str, float]] = None,
        return_normaliser: float = 0.30,
        n_accuracy_buckets: int = 10,
    ) -> None:
        self.idea_book = idea_book or IdeaBook()
        self.weights = dimension_weights or dict(self.DEFAULT_WEIGHTS)
        self.thresholds = thresholds or dict(self.DEFAULT_THRESHOLDS)
        self.return_normaliser = return_normaliser
        self.n_accuracy_buckets = n_accuracy_buckets

        # Normalise weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        # Historical accuracy: bucket -> AccuracyRecord
        bucket_edges = np.linspace(0, 1, n_accuracy_buckets + 1)
        self._bucket_centres = 0.5 * (bucket_edges[:-1] + bucket_edges[1:])
        self._accuracy: dict[float, AccuracyRecord] = {
            round(float(c), 4): AccuracyRecord(score_bucket=round(float(c), 4))
            for c in self._bucket_centres
        }

        self._scored: list[IdeaScore] = []

    # ------------------------------------------------------------------
    # Scoring entry point
    # ------------------------------------------------------------------

    def score(self, idea: IdeaInput, add_to_book: bool = True) -> IdeaScore:
        """Score a single idea and (optionally) add it to the idea book."""
        idea_id = str(uuid.uuid4())
        notes: list[str] = []

        novelty = self._score_novelty(idea, notes)
        feasibility = self._score_feasibility(idea, notes)
        rar, raw_rar, raw_exp_ret = self._score_risk_adj_return(idea, notes)
        timing, raw_timing = self._score_timing(idea, notes)
        cross_domain, n_confirm, n_total = self._score_cross_domain(idea, notes)
        regime_fit = self._score_regime_fit(idea, notes)
        capacity = self._score_capacity(idea, notes)

        dims = {
            "novelty": novelty,
            "feasibility": feasibility,
            "risk_adj_return": rar,
            "timing": timing,
            "cross_domain": cross_domain,
            "regime_fit": regime_fit,
            "capacity": capacity,
        }
        overall = float(sum(self.weights[k] * v for k, v in dims.items()))

        failed = [
            k for k, v in dims.items()
            if v < self.thresholds.get(k, 0.0)
        ]
        if failed:
            notes.append(f"Below threshold: {', '.join(failed)}")

        score_obj = IdeaScore(
            id=idea_id,
            name=idea.name or idea_id[:8],
            timestamp=_utcnow(),
            novelty=round(novelty, 4),
            feasibility=round(feasibility, 4),
            risk_adj_return=round(rar, 4),
            timing=round(timing, 4),
            cross_domain=round(cross_domain, 4),
            regime_fit=round(regime_fit, 4),
            capacity=round(capacity, 4),
            raw_expected_return=round(raw_exp_ret, 6),
            raw_risk_adj_return=round(raw_rar, 6),
            raw_timing_ratio=round(raw_timing, 4),
            n_confirming_domains=n_confirm,
            n_total_domains=n_total,
            overall_score=round(overall, 4),
            passed_thresholds=len(failed) == 0,
            failed_dimensions=failed,
            notes=notes,
        )

        self._scored.append(score_obj)

        if add_to_book:
            self.idea_book.add(idea_id, score_obj.name, idea.feature_vector)

        return score_obj

    def score_batch(self, ideas: list[IdeaInput], add_to_book: bool = True) -> list[IdeaScore]:
        return [self.score(idea, add_to_book=add_to_book) for idea in ideas]

    # ------------------------------------------------------------------
    # Individual dimension scorers
    # ------------------------------------------------------------------

    def _score_novelty(self, idea: IdeaInput, notes: list[str]) -> float:
        """
        Novelty = 1 - max_cosine_similarity to existing ideas.
        A score of 1.0 means completely new; 0.0 means identical copy.
        """
        if len(self.idea_book) == 0:
            notes.append("Novelty: idea book empty, assuming fully novel.")
            return 1.0
        sim = self.idea_book.max_similarity(idea.feature_vector)
        novelty = _clamp(1.0 - sim)
        if novelty < 0.5:
            similar = self.idea_book.most_similar(idea.feature_vector, top_k=1)
            if similar:
                notes.append(
                    f"Novelty LOW: most similar to '{similar[0][1]}' "
                    f"(sim={similar[0][2]:.2f})"
                )
        return novelty

    def _score_feasibility(self, idea: IdeaInput, notes: list[str]) -> float:
        """
        Feasibility is penalised by data complexity and rewarded by liquidity.
        """
        complexity_penalty = idea.data_complexity          # [0,1]; 1 = very complex
        liquidity_bonus = idea.liquidity_score             # [0,1]; 1 = very liquid
        # Weighted: 40% complexity drag, 60% liquidity
        raw = 0.60 * liquidity_bonus + 0.40 * (1.0 - complexity_penalty)
        return _clamp(raw)

    def _score_risk_adj_return(
        self, idea: IdeaInput, notes: list[str]
    ) -> tuple[float, float, float]:
        """
        risk_adj_return = signal_strength * win_rate - max_loss * (1 - win_rate)
        Returns (normalised_score [0,1], raw_rar, raw_expected_return).
        """
        wr = _clamp(idea.win_rate)
        ss = _clamp(idea.signal_strength, 0.0, 10.0)
        ml = _clamp(idea.max_loss_per_trade, 0.0, 10.0)

        raw_rar = ss * wr - ml * (1.0 - wr)
        # Expected return ≈ signal_strength * win_rate (gross)
        raw_exp_ret = ss * wr

        # Normalise: raw_rar / return_normaliser, capped at 1
        score = _clamp(raw_rar / (self.return_normaliser + 1e-9))
        if raw_rar < 0:
            notes.append(f"Risk-adjusted return is NEGATIVE ({raw_rar:.4f}).")
        return score, raw_rar, raw_exp_ret

    def _score_timing(
        self, idea: IdeaInput, notes: list[str]
    ) -> tuple[float, float]:
        """
        Timing = min(frequency_now / base_rate, 2) / 2
        A ratio of 1 means the setup fires at exactly its historical rate.
        A ratio of 2 means it's firing twice as often (strong timing).
        """
        if idea.setup_base_rate <= 0:
            notes.append("Timing: base rate is zero, using 0.5.")
            return 0.5, 1.0
        ratio = idea.setup_frequency_now / idea.setup_base_rate
        score = _clamp(min(ratio, 2.0) / 2.0)
        return score, ratio

    def _score_cross_domain(
        self, idea: IdeaInput, notes: list[str]
    ) -> tuple[float, int, int]:
        """
        Cross-domain score = fraction of domains with agreement > 0.5,
        weighted by their agreement level.
        """
        if not idea.domain_signals:
            notes.append("Cross-domain: no domain signals provided.")
            return 0.5, 0, 0
        values = np.array(list(idea.domain_signals.values()), dtype=float)
        n_total = len(values)
        n_confirm = int(np.sum(values > 0.5))
        if n_total == 0:
            return 0.5, 0, 0
        score = _clamp(float(np.mean(values)))
        if n_confirm < n_total:
            notes.append(
                f"Cross-domain: {n_confirm}/{n_total} domains confirm "
                f"(avg={score:.2f})."
            )
        return score, n_confirm, n_total

    def _score_regime_fit(self, idea: IdeaInput, notes: list[str]) -> float:
        """
        Return the fit score for the current regime.
        Falls back to average fit if current regime is unknown.
        """
        if not idea.idea_regime_fit:
            notes.append("Regime fit: no regime map provided, using 0.5.")
            return 0.5
        regime = idea.regime_label
        if regime in idea.idea_regime_fit:
            return _clamp(idea.idea_regime_fit[regime])
        # Average over known regimes
        avg = float(np.mean(list(idea.idea_regime_fit.values())))
        notes.append(f"Regime '{regime}' not in fit map; using average {avg:.2f}.")
        return _clamp(avg)

    def _score_capacity(self, idea: IdeaInput, notes: list[str]) -> float:
        """
        Capacity score is higher when ADV fraction is small.
        Uses a simple exponential decay: exp(-k * adv_fraction).
        k = 50 means adv_fraction > 0.10 scores near 0.
        """
        k = 50.0
        score = _clamp(float(np.exp(-k * idea.capacity_adv_fraction)))
        if idea.capacity_adv_fraction > 0.05:
            notes.append(
                f"Capacity: large ADV fraction ({idea.capacity_adv_fraction:.2%}) "
                f"limits position size."
            )
        return score

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_ideas(
        self,
        ideas: Optional[list[IdeaScore]] = None,
        filter_passed: bool = False,
    ) -> list[IdeaScore]:
        """
        Return list of IdeaScore objects sorted by overall_score descending,
        with rank field populated.
        """
        pool = ideas if ideas is not None else list(self._scored)
        if filter_passed:
            pool = [s for s in pool if s.passed_thresholds]
        pool.sort(key=lambda x: x.overall_score, reverse=True)
        for i, s in enumerate(pool):
            s.rank = i + 1
        return pool

    def top_ideas(self, n: int = 5, filter_passed: bool = True) -> list[IdeaScore]:
        return self.rank_ideas(filter_passed=filter_passed)[:n]

    # ------------------------------------------------------------------
    # Historical accuracy
    # ------------------------------------------------------------------

    def _bucket_for_score(self, score: float) -> float:
        """Return the bucket centre closest to *score*."""
        diffs = np.abs(self._bucket_centres - score)
        return float(self._bucket_centres[int(np.argmin(diffs))])

    def record_outcome(self, idea_id: str, winner: bool, actual_return: float,
                       hold_days: float = 0.0) -> None:
        """
        After an idea resolves, record whether it was a winner and its return.
        Finds the idea in _scored by id and updates the accuracy bucket.
        """
        for s in self._scored:
            if s.id == idea_id:
                bucket = round(self._bucket_for_score(s.overall_score), 4)
                rec = self._accuracy.get(bucket)
                if rec is None:
                    rec = AccuracyRecord(score_bucket=bucket)
                    self._accuracy[bucket] = rec
                rec.n_ideas += 1
                if winner:
                    rec.n_winners += 1
                # Running mean
                rec.avg_return = (
                    (rec.avg_return * (rec.n_ideas - 1) + actual_return) / rec.n_ideas
                )
                rec.avg_hold_days = (
                    (rec.avg_hold_days * (rec.n_ideas - 1) + hold_days) / rec.n_ideas
                )
                return
        raise KeyError(f"No scored idea with id={idea_id!r}")

    def accuracy_report(self) -> list[dict]:
        rows = []
        for bucket in sorted(self._accuracy.keys()):
            rec = self._accuracy[bucket]
            if rec.n_ideas > 0:
                rows.append({
                    "score_bucket": rec.score_bucket,
                    "n_ideas": rec.n_ideas,
                    "win_rate": round(rec.win_rate, 3),
                    "avg_return": round(rec.avg_return, 4),
                    "avg_hold_days": round(rec.avg_hold_days, 1),
                })
        return rows

    def expected_win_rate_for_score(self, score: float) -> Optional[float]:
        """Interpolate historical win rate for a given overall score."""
        bucket = round(self._bucket_for_score(score), 4)
        rec = self._accuracy.get(bucket)
        if rec and rec.n_ideas > 0:
            return rec.win_rate
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            "version": "1.0",
            "exported_at": _utcnow(),
            "weights": self.weights,
            "thresholds": self.thresholds,
            "scored": [s.to_dict() for s in self._scored],
            "accuracy": {str(k): v.to_dict() for k, v in self._accuracy.items()},
            "idea_book": {
                "ids": self.idea_book._ids,
                "names": self.idea_book._names,
                "vectors": [v.tolist() for v in self.idea_book._vectors],
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.weights = data.get("weights", self.weights)
        self.thresholds = data.get("thresholds", self.thresholds)
        self._scored = [IdeaScore.from_dict(s) for s in data.get("scored", [])]
        for k_str, v in data.get("accuracy", {}).items():
            k = round(float(k_str), 4)
            self._accuracy[k] = AccuracyRecord(**v)
        book_data = data.get("idea_book", {})
        self.idea_book._ids = book_data.get("ids", [])
        self.idea_book._names = book_data.get("names", [])
        self.idea_book._vectors = [
            np.asarray(v, dtype=float) for v in book_data.get("vectors", [])
        ]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if not self._scored:
            return {"total": 0}
        scores = np.array([s.overall_score for s in self._scored])
        passed = sum(1 for s in self._scored if s.passed_thresholds)
        return {
            "total_scored": len(self._scored),
            "passed_thresholds": passed,
            "failed_thresholds": len(self._scored) - passed,
            "mean_score": round(float(np.mean(scores)), 4),
            "std_score": round(float(np.std(scores)), 4),
            "min_score": round(float(np.min(scores)), 4),
            "max_score": round(float(np.max(scores)), 4),
            "idea_book_size": len(self.idea_book),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"IdeaScorer(scored={s.get('total_scored', 0)}, "
            f"passed={s.get('passed_thresholds', 0)}, "
            f"book_size={s.get('idea_book_size', 0)})"
        )


# ---------------------------------------------------------------------------
# Score formatter
# ---------------------------------------------------------------------------

def format_score(score: IdeaScore) -> str:
    bar_width = 20

    def bar(val: float) -> str:
        filled = int(round(val * bar_width))
        return "[" + "#" * filled + "." * (bar_width - filled) + f"] {val:.2f}"

    dims = [
        ("Novelty        ", score.novelty),
        ("Feasibility    ", score.feasibility),
        ("Risk-Adj Return", score.risk_adj_return),
        ("Timing         ", score.timing),
        ("Cross-Domain   ", score.cross_domain),
        ("Regime Fit     ", score.regime_fit),
        ("Capacity       ", score.capacity),
    ]
    lines = [
        "=" * 55,
        f"  IDEA SCORE: {score.name}",
        f"  ID: {score.id[:12]}...   [{score.timestamp[:19]}]",
        "-" * 55,
    ]
    for label, val in dims:
        lines.append(f"  {label}: {bar(val)}")
    lines += [
        "-" * 55,
        f"  OVERALL SCORE  : {score.overall_score:.4f}",
        f"  Rank           : {score.rank}",
        f"  Passed         : {score.passed_thresholds}",
    ]
    if score.failed_dimensions:
        lines.append(f"  Failed dims    : {', '.join(score.failed_dimensions)}")
    if score.notes:
        lines.append("-" * 55)
        for note in score.notes:
            lines.append(f"  NOTE: {note}")
    lines.append("=" * 55)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    rng = np.random.default_rng(0)

    def rand_vec(n: int = 32) -> list[float]:
        return rng.standard_normal(n).tolist()

    book = IdeaBook()
    # Pre-populate with a few existing ideas
    for name in ["TrendFollowV1", "MeanRevV2", "PairsArb"]:
        book.add(str(uuid.uuid4()), name, rand_vec())

    scorer = IdeaScorer(idea_book=book)

    ideas = [
        IdeaInput(
            feature_vector=rand_vec(),
            signal_strength=0.58, win_rate=0.56, max_loss_per_trade=0.015,
            data_complexity=0.2, liquidity_score=0.9,
            setup_frequency_now=12, setup_base_rate=10,
            domain_signals={"technical": 0.8, "fundamental": 0.6, "macro": 0.7},
            regime_label="trending",
            idea_regime_fit={"trending": 0.9, "mean_reverting": 0.3, "high_vol": 0.5},
            capacity_adv_fraction=0.005,
            name="MomentumBreakout_v3",
        ),
        IdeaInput(
            feature_vector=rand_vec(),
            signal_strength=0.40, win_rate=0.48, max_loss_per_trade=0.03,
            data_complexity=0.7, liquidity_score=0.4,
            setup_frequency_now=2, setup_base_rate=10,
            domain_signals={"technical": 0.4, "sentiment": 0.3},
            regime_label="high_vol",
            idea_regime_fit={"trending": 0.2, "high_vol": 0.4},
            capacity_adv_fraction=0.12,
            name="LowFreqAlgo_v1",
        ),
        IdeaInput(
            feature_vector=rand_vec(),
            signal_strength=0.65, win_rate=0.60, max_loss_per_trade=0.01,
            data_complexity=0.3, liquidity_score=0.85,
            setup_frequency_now=8, setup_base_rate=6,
            domain_signals={"technical": 0.9, "fundamental": 0.75, "macro": 0.8, "quant": 0.7},
            regime_label="trending",
            idea_regime_fit={"trending": 0.85, "mean_reverting": 0.5},
            capacity_adv_fraction=0.008,
            name="MultiFactorLong",
        ),
    ]

    scores = scorer.score_batch(ideas)
    ranked = scorer.rank_ideas()

    for s in ranked:
        print(format_score(s))
        print()

    print(scorer.summary())

    # Simulate outcomes
    scorer.record_outcome(scores[0].id, winner=True, actual_return=0.042, hold_days=5)
    scorer.record_outcome(scores[1].id, winner=False, actual_return=-0.018, hold_days=3)
    scorer.record_outcome(scores[2].id, winner=True, actual_return=0.073, hold_days=8)

    print("\nAccuracy report:", scorer.accuracy_report())


if __name__ == "__main__":
    _smoke_test()
