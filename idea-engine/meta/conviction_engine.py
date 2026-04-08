"""
conviction_engine.py — Multi-dimensional conviction scoring for trading ideas.

Dimensions:
  - technical   : pattern quality, chart structure
  - fundamental : regime alignment, macro backdrop
  - quantitative: signal IC, factor exposure quality
  - timing      : entry quality, pullback, momentum
  - risk        : max drawdown risk, tail risk estimate
  - uniqueness  : alpha non-overlap with existing positions

Includes:
  - ConvictionEngine class with weighted scoring
  - Conviction decay (staleness)
  - High-conviction watchlist management
  - Calibration against realised performance
  - Conviction → position size mapping
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Conviction dimensions and scoring
# ---------------------------------------------------------------------------

class ConvictionDimension(str, Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    QUANTITATIVE = "quantitative"
    TIMING = "timing"
    RISK = "risk"
    UNIQUENESS = "uniqueness"


DEFAULT_WEIGHTS: Dict[ConvictionDimension, float] = {
    ConvictionDimension.TECHNICAL:    0.20,
    ConvictionDimension.FUNDAMENTAL:  0.20,
    ConvictionDimension.QUANTITATIVE: 0.25,
    ConvictionDimension.TIMING:       0.15,
    ConvictionDimension.RISK:         0.10,
    ConvictionDimension.UNIQUENESS:   0.10,
}

# How each dimension decays per day (multiplied to score)
DECAY_RATES: Dict[ConvictionDimension, float] = {
    ConvictionDimension.TECHNICAL:    0.97,   # fast decay — patterns fade
    ConvictionDimension.FUNDAMENTAL:  0.995,  # slow decay — regime persists
    ConvictionDimension.QUANTITATIVE: 0.99,
    ConvictionDimension.TIMING:       0.93,   # fastest — timing is very ephemeral
    ConvictionDimension.RISK:         0.99,
    ConvictionDimension.UNIQUENESS:   0.985,
}


@dataclass
class DimensionScore:
    dimension: ConvictionDimension
    raw_score: float          # 0.0 – 1.0
    confidence: float         # how confident are we in this score
    notes: str = ""
    scored_at: float = field(default_factory=time.time)

    def decayed_score(self, days_elapsed: float) -> float:
        rate = DECAY_RATES[self.dimension]
        return self.raw_score * (rate ** days_elapsed)


@dataclass
class ConvictionRecord:
    idea_id: str
    ticker: str
    direction: str           # "long" | "short" | "neutral"
    dimension_scores: Dict[ConvictionDimension, DimensionScore]
    composite_score: float   # weighted, decayed composite [0, 1]
    created_at: float        # Unix timestamp
    last_updated: float
    regime: str = "unknown"
    notes: str = ""

    # Realised performance tracking
    realised_pnl: float = 0.0
    realised_sharpe: float = 0.0
    outcome: Optional[str] = None  # "win" | "loss" | "scratch" | None

    def staleness_days(self) -> float:
        return (time.time() - self.last_updated) / 86400.0

    def to_dict(self) -> Dict:
        return {
            "idea_id": self.idea_id,
            "ticker": self.ticker,
            "direction": self.direction,
            "composite_score": round(self.composite_score, 4),
            "regime": self.regime,
            "staleness_days": round(self.staleness_days(), 2),
            "dimension_scores": {
                k.value: {
                    "raw": round(v.raw_score, 4),
                    "decayed": round(v.decayed_score(self.staleness_days()), 4),
                    "confidence": round(v.confidence, 4),
                    "notes": v.notes,
                }
                for k, v in self.dimension_scores.items()
            },
            "realised_pnl": self.realised_pnl,
            "realised_sharpe": self.realised_sharpe,
            "outcome": self.outcome,
        }


# ---------------------------------------------------------------------------
# Scoring helpers for each dimension
# ---------------------------------------------------------------------------

class TechnicalScorer:
    """Score technical quality of a setup."""

    def score(
        self,
        pattern_quality: float,       # 0-1: pattern completeness / clarity
        trend_alignment: float,       # 0-1: aligned with higher-timeframe trend
        support_proximity: float,     # 0-1: closeness to support/resistance
        momentum_confluence: float,   # 0-1: RSI, MACD alignment
    ) -> DimensionScore:
        weights = [0.35, 0.30, 0.20, 0.15]
        vals = [pattern_quality, trend_alignment, support_proximity, momentum_confluence]
        raw = float(np.dot(weights, [np.clip(v, 0, 1) for v in vals]))
        confidence = float(np.mean([v for v in vals if v > 0]))
        return DimensionScore(
            dimension=ConvictionDimension.TECHNICAL,
            raw_score=raw,
            confidence=confidence,
            notes=f"pattern={pattern_quality:.2f} trend={trend_alignment:.2f}",
        )


class FundamentalScorer:
    """Score fundamental / macro regime alignment."""

    REGIME_SCORES = {
        "bull":       {"long": 0.90, "short": 0.15, "neutral": 0.50},
        "bear":       {"long": 0.15, "short": 0.90, "neutral": 0.50},
        "neutral":    {"long": 0.55, "short": 0.55, "neutral": 0.70},
        "volatile":   {"long": 0.35, "short": 0.45, "neutral": 0.60},
        "risk_off":   {"long": 0.20, "short": 0.80, "neutral": 0.55},
        "risk_on":    {"long": 0.85, "short": 0.20, "neutral": 0.55},
        "unknown":    {"long": 0.50, "short": 0.50, "neutral": 0.50},
    }

    def score(
        self,
        regime: str,
        direction: str,
        macro_score: float = 0.5,      # 0-1: strength of macro backdrop
        sector_rotation: float = 0.5,  # 0-1: sector momentum alignment
    ) -> DimensionScore:
        regime_base = self.REGIME_SCORES.get(regime, self.REGIME_SCORES["unknown"]).get(direction, 0.5)
        raw = 0.5 * regime_base + 0.3 * macro_score + 0.2 * sector_rotation
        confidence = macro_score * 0.6 + 0.4
        return DimensionScore(
            dimension=ConvictionDimension.FUNDAMENTAL,
            raw_score=float(np.clip(raw, 0, 1)),
            confidence=float(confidence),
            notes=f"regime={regime} macro={macro_score:.2f}",
        )


class QuantitativeScorer:
    """Score quantitative signal quality: IC, factor exposure, backtesting."""

    def score(
        self,
        signal_ic: float,            # information coefficient [-1, 1]
        ic_stability: float,         # 0-1: rolling IC stability
        factor_loading: float,       # 0-1: exposure to quality factors
        backtest_sharpe: float,      # annualised Sharpe from backtest
    ) -> DimensionScore:
        # Normalise IC to [0, 1]
        ic_score = (signal_ic + 1.0) / 2.0
        # Cap backtest Sharpe: 3.0 is excellent
        sharpe_score = float(np.clip(backtest_sharpe / 3.0, 0.0, 1.0))
        weights = [0.30, 0.25, 0.20, 0.25]
        vals = [ic_score, ic_stability, factor_loading, sharpe_score]
        raw = float(np.dot(weights, [np.clip(v, 0, 1) for v in vals]))
        confidence = ic_stability * 0.7 + 0.3
        return DimensionScore(
            dimension=ConvictionDimension.QUANTITATIVE,
            raw_score=raw,
            confidence=confidence,
            notes=f"IC={signal_ic:.3f} sharpe={backtest_sharpe:.2f}",
        )


class TimingScorer:
    """Score entry timing quality."""

    def score(
        self,
        z_score: float,          # signal z-score — how extreme is the entry?
        vol_percentile: float,   # current vol vs history [0, 1]
        bid_ask_norm: float,     # spread normalised [0, 1], lower = better
        catalyst_proximity: float,  # 0-1: nearness to a catalyst event
    ) -> DimensionScore:
        # Penalise entering when vol is very high (bad timing)
        vol_penalty = max(0.0, vol_percentile - 0.8) * 2.0
        # z_score: 1-3 is ideal entry zone
        abs_z = abs(z_score)
        if abs_z < 1.0:
            z_timing = 0.3
        elif abs_z <= 2.5:
            z_timing = 0.6 + 0.2 * (abs_z - 1.0) / 1.5
        else:
            z_timing = max(0.2, 1.0 - (abs_z - 2.5) * 0.3)

        spread_score = 1.0 - bid_ask_norm
        raw = 0.35 * z_timing + 0.25 * spread_score + 0.25 * catalyst_proximity - 0.15 * vol_penalty
        confidence = 0.5 + 0.5 * catalyst_proximity
        return DimensionScore(
            dimension=ConvictionDimension.TIMING,
            raw_score=float(np.clip(raw, 0, 1)),
            confidence=float(confidence),
            notes=f"z={z_score:.2f} vol_pct={vol_percentile:.2f}",
        )


class RiskScorer:
    """Score risk-adjusted attractiveness of the trade."""

    def score(
        self,
        reward_to_risk: float,    # expected R:R ratio
        max_drawdown_est: float,  # estimated max drawdown [0, 1]
        liquidity_score: float,   # 0-1: how liquid is the instrument
        correlation_to_book: float,  # [-1, 1]: correlation with existing portfolio
    ) -> DimensionScore:
        # R:R score: 2.0 = good, 3.0+ = excellent
        rr_score = float(np.clip(reward_to_risk / 3.0, 0.0, 1.0))
        dd_score = 1.0 - max_drawdown_est
        # Penalise high correlation (increases concentration risk)
        corr_penalty = max(0.0, correlation_to_book - 0.5) * 0.5
        raw = 0.35 * rr_score + 0.30 * dd_score + 0.20 * liquidity_score - corr_penalty
        confidence = liquidity_score * 0.5 + 0.5
        return DimensionScore(
            dimension=ConvictionDimension.RISK,
            raw_score=float(np.clip(raw, 0, 1)),
            confidence=float(confidence),
            notes=f"R:R={reward_to_risk:.2f} dd_est={max_drawdown_est:.2f}",
        )


class UniquenessScorer:
    """Score alpha uniqueness — how different is this idea from existing positions?"""

    def score(
        self,
        existing_signals: List[np.ndarray],
        new_signal: np.ndarray,
        factor_overlap: float = 0.0,  # 0-1: overlap with known factors
    ) -> DimensionScore:
        if not existing_signals or len(new_signal) == 0:
            uniqueness = 1.0
        else:
            sims = []
            for sig in existing_signals:
                if len(sig) == len(new_signal) and np.linalg.norm(new_signal) > 0 and np.linalg.norm(sig) > 0:
                    cos_sim = float(np.dot(sig, new_signal) / (np.linalg.norm(sig) * np.linalg.norm(new_signal)))
                    sims.append(abs(cos_sim))
            max_sim = max(sims) if sims else 0.0
            uniqueness = 1.0 - max_sim

        raw = 0.7 * uniqueness + 0.3 * (1.0 - factor_overlap)
        confidence = 0.6 if existing_signals else 0.4
        return DimensionScore(
            dimension=ConvictionDimension.UNIQUENESS,
            raw_score=float(np.clip(raw, 0, 1)),
            confidence=float(confidence),
            notes=f"uniqueness={uniqueness:.3f} factor_overlap={factor_overlap:.2f}",
        )


# ---------------------------------------------------------------------------
# ConvictionEngine
# ---------------------------------------------------------------------------

class ConvictionEngine:
    """
    Computes, tracks, and manages conviction scores for trading ideas.
    """

    # Conviction thresholds for watchlist tiers
    TIER_THRESHOLDS = {
        "top_conviction": 0.75,
        "high":           0.60,
        "medium":         0.45,
        "low":            0.30,
    }

    # Position size mapping (fraction of max capital)
    POSITION_SIZE_MAP = {
        "top_conviction": 1.00,
        "high":           0.65,
        "medium":         0.35,
        "low":            0.10,
    }

    def __init__(self, weights: Optional[Dict[ConvictionDimension, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self._normalise_weights()

        self.technical_scorer = TechnicalScorer()
        self.fundamental_scorer = FundamentalScorer()
        self.quantitative_scorer = QuantitativeScorer()
        self.timing_scorer = TimingScorer()
        self.risk_scorer = RiskScorer()
        self.uniqueness_scorer = UniquenessScorer()

        self._records: Dict[str, ConvictionRecord] = {}
        self._calibration_data: List[Tuple[float, float]] = []  # (predicted, actual pnl)

    def _normalise_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def compute_conviction(
        self,
        idea_id: str,
        ticker: str,
        direction: str,
        regime: str,
        # Technical inputs
        pattern_quality: float = 0.5,
        trend_alignment: float = 0.5,
        support_proximity: float = 0.5,
        momentum_confluence: float = 0.5,
        # Fundamental inputs
        macro_score: float = 0.5,
        sector_rotation: float = 0.5,
        # Quantitative inputs
        signal_ic: float = 0.0,
        ic_stability: float = 0.5,
        factor_loading: float = 0.5,
        backtest_sharpe: float = 1.0,
        # Timing inputs
        z_score: float = 0.0,
        vol_percentile: float = 0.5,
        bid_ask_norm: float = 0.1,
        catalyst_proximity: float = 0.5,
        # Risk inputs
        reward_to_risk: float = 2.0,
        max_drawdown_est: float = 0.1,
        liquidity_score: float = 0.8,
        correlation_to_book: float = 0.0,
        # Uniqueness inputs
        existing_signals: Optional[List[np.ndarray]] = None,
        new_signal: Optional[np.ndarray] = None,
        factor_overlap: float = 0.0,
        notes: str = "",
    ) -> ConvictionRecord:
        """Compute a full conviction record for a trading idea."""
        dim_scores: Dict[ConvictionDimension, DimensionScore] = {}

        dim_scores[ConvictionDimension.TECHNICAL] = self.technical_scorer.score(
            pattern_quality, trend_alignment, support_proximity, momentum_confluence
        )
        dim_scores[ConvictionDimension.FUNDAMENTAL] = self.fundamental_scorer.score(
            regime, direction, macro_score, sector_rotation
        )
        dim_scores[ConvictionDimension.QUANTITATIVE] = self.quantitative_scorer.score(
            signal_ic, ic_stability, factor_loading, backtest_sharpe
        )
        dim_scores[ConvictionDimension.TIMING] = self.timing_scorer.score(
            z_score, vol_percentile, bid_ask_norm, catalyst_proximity
        )
        dim_scores[ConvictionDimension.RISK] = self.risk_scorer.score(
            reward_to_risk, max_drawdown_est, liquidity_score, correlation_to_book
        )
        dim_scores[ConvictionDimension.UNIQUENESS] = self.uniqueness_scorer.score(
            existing_signals or [], new_signal if new_signal is not None else np.array([]), factor_overlap
        )

        composite = self._weighted_composite(dim_scores)

        now = time.time()
        record = ConvictionRecord(
            idea_id=idea_id,
            ticker=ticker,
            direction=direction,
            dimension_scores=dim_scores,
            composite_score=composite,
            created_at=now,
            last_updated=now,
            regime=regime,
            notes=notes,
        )

        self._records[idea_id] = record
        return record

    def _weighted_composite(self, dim_scores: Dict[ConvictionDimension, DimensionScore]) -> float:
        total = 0.0
        for dim, ds in dim_scores.items():
            w = self.weights.get(dim, 0.0)
            total += w * ds.raw_score
        return float(np.clip(total, 0.0, 1.0))

    def get_decayed_score(self, idea_id: str) -> Optional[float]:
        """Return composite score adjusted for staleness."""
        rec = self._records.get(idea_id)
        if rec is None:
            return None
        days = rec.staleness_days()
        total = 0.0
        for dim, ds in rec.dimension_scores.items():
            w = self.weights.get(dim, 0.0)
            total += w * ds.decayed_score(days)
        return float(np.clip(total, 0.0, 1.0))

    def refresh_scores(self) -> None:
        """Update composite scores for all records based on decay."""
        for idea_id, rec in self._records.items():
            decayed = self.get_decayed_score(idea_id)
            if decayed is not None:
                rec.composite_score = decayed
                rec.last_updated = time.time()

    def tier(self, score: float) -> str:
        if score >= self.TIER_THRESHOLDS["top_conviction"]:
            return "top_conviction"
        elif score >= self.TIER_THRESHOLDS["high"]:
            return "high"
        elif score >= self.TIER_THRESHOLDS["medium"]:
            return "medium"
        elif score >= self.TIER_THRESHOLDS["low"]:
            return "low"
        return "discard"

    def position_size(self, idea_id: str, max_capital: float = 1.0) -> float:
        """Map conviction score to position size fraction."""
        score = self.get_decayed_score(idea_id)
        if score is None:
            return 0.0
        t = self.tier(score)
        if t == "discard":
            return 0.0
        fraction = self.POSITION_SIZE_MAP[t]
        # Smooth scaling within tier
        if t == "top_conviction":
            sub = (score - self.TIER_THRESHOLDS["top_conviction"]) / (1.0 - self.TIER_THRESHOLDS["top_conviction"])
        elif t == "high":
            sub = (score - self.TIER_THRESHOLDS["high"]) / (self.TIER_THRESHOLDS["top_conviction"] - self.TIER_THRESHOLDS["high"])
        elif t == "medium":
            sub = (score - self.TIER_THRESHOLDS["medium"]) / (self.TIER_THRESHOLDS["high"] - self.TIER_THRESHOLDS["medium"])
        else:
            sub = (score - self.TIER_THRESHOLDS["low"]) / (self.TIER_THRESHOLDS["medium"] - self.TIER_THRESHOLDS["low"])

        # Interpolate within tier band
        lower = self.POSITION_SIZE_MAP.get(t, 0.0)
        return float(np.clip(lower * (0.8 + 0.2 * sub) * max_capital, 0.0, max_capital))

    # ------------------------------------------------------------------
    # Watchlist management
    # ------------------------------------------------------------------

    def watchlist(self, tier_filter: Optional[str] = None) -> List[ConvictionRecord]:
        """Return watchlist sorted by decayed conviction score."""
        self.refresh_scores()
        items = list(self._records.values())
        if tier_filter:
            items = [r for r in items if self.tier(r.composite_score) == tier_filter]
        return sorted(items, key=lambda r: r.composite_score, reverse=True)

    def top_n(self, n: int = 5) -> List[ConvictionRecord]:
        return self.watchlist()[:n]

    def remove_stale(self, max_days: float = 14.0) -> List[str]:
        """Remove ideas that have decayed below threshold or are too old."""
        to_remove = []
        for idea_id, rec in list(self._records.items()):
            if rec.staleness_days() > max_days:
                to_remove.append(idea_id)
            elif (self.get_decayed_score(idea_id) or 0.0) < self.TIER_THRESHOLDS["low"]:
                to_remove.append(idea_id)
        for idea_id in to_remove:
            del self._records[idea_id]
        return to_remove

    # ------------------------------------------------------------------
    # Performance calibration
    # ------------------------------------------------------------------

    def record_outcome(self, idea_id: str, realised_pnl: float, realised_sharpe: float) -> None:
        rec = self._records.get(idea_id)
        if rec is None:
            return
        rec.realised_pnl = realised_pnl
        rec.realised_sharpe = realised_sharpe
        rec.outcome = "win" if realised_pnl > 0 else "loss" if realised_pnl < 0 else "scratch"
        self._calibration_data.append((rec.composite_score, realised_pnl))

    def calibration_stats(self) -> Dict:
        """Analyse how well conviction scores predicted outcomes."""
        if len(self._calibration_data) < 5:
            return {"error": "insufficient data"}

        scores = np.array([x[0] for x in self._calibration_data])
        pnls = np.array([x[1] for x in self._calibration_data])

        # Rank correlation
        rank_s = np.argsort(np.argsort(scores)).astype(float)
        rank_p = np.argsort(np.argsort(pnls)).astype(float)
        n = len(rank_s)
        spearman = 1.0 - 6.0 * ((rank_s - rank_p) ** 2).sum() / (n * (n**2 - 1) + 1e-9)

        # Win rate by tier
        bins = [0.0, self.TIER_THRESHOLDS["low"], self.TIER_THRESHOLDS["medium"],
                self.TIER_THRESHOLDS["high"], self.TIER_THRESHOLDS["top_conviction"], 1.01]
        tier_names = ["discard", "low", "medium", "high", "top_conviction"]
        tier_stats = {}
        for i, tname in enumerate(tier_names):
            mask = (scores >= bins[i]) & (scores < bins[i+1])
            if mask.sum() > 0:
                tier_pnls = pnls[mask]
                tier_stats[tname] = {
                    "n": int(mask.sum()),
                    "win_rate": float((tier_pnls > 0).mean()),
                    "avg_pnl": float(tier_pnls.mean()),
                }

        return {
            "spearman_rank_corr": round(float(spearman), 4),
            "n_observations": len(self._calibration_data),
            "overall_win_rate": float((pnls > 0).mean()),
            "tier_stats": tier_stats,
        }

    def adjust_weights_from_calibration(self) -> None:
        """
        Simple weight adjustment: boost dimensions with highest individual IC.
        Uses last 50 calibrated ideas.
        """
        if len(self._calibration_data) < 20:
            return

        recent_ids = [r.idea_id for r in sorted(self._records.values(),
                      key=lambda r: r.last_updated, reverse=True)[:50]
                      if r.outcome is not None]

        if not recent_ids:
            return

        dim_corrs: Dict[ConvictionDimension, float] = {}
        for dim in ConvictionDimension:
            xs, ys = [], []
            for rid in recent_ids:
                rec = self._records.get(rid)
                if rec and dim in rec.dimension_scores and rec.realised_pnl != 0.0:
                    xs.append(rec.dimension_scores[dim].raw_score)
                    ys.append(rec.realised_pnl)
            if len(xs) >= 5:
                corr = float(np.corrcoef(xs, ys)[0, 1])
                dim_corrs[dim] = max(0.01, abs(corr))

        if dim_corrs:
            total = sum(dim_corrs.values())
            for dim, c in dim_corrs.items():
                self.weights[dim] = c / total

    def export(self) -> List[Dict]:
        """Export all records as JSON-serialisable dicts."""
        return [r.to_dict() for r in self._records.values()]

    def summary(self) -> Dict:
        self.refresh_scores()
        by_tier: Dict[str, int] = {}
        for rec in self._records.values():
            t = self.tier(rec.composite_score)
            by_tier[t] = by_tier.get(t, 0) + 1
        return {
            "total_ideas": len(self._records),
            "by_tier": by_tier,
            "weights": {k.value: round(v, 4) for k, v in self.weights.items()},
            "calibration": self.calibration_stats(),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = ConvictionEngine()

    rec = engine.compute_conviction(
        idea_id="idea_001",
        ticker="AAPL",
        direction="long",
        regime="bull",
        pattern_quality=0.80,
        trend_alignment=0.75,
        support_proximity=0.65,
        momentum_confluence=0.70,
        macro_score=0.72,
        sector_rotation=0.65,
        signal_ic=0.12,
        ic_stability=0.70,
        factor_loading=0.60,
        backtest_sharpe=1.8,
        z_score=1.5,
        vol_percentile=0.45,
        bid_ask_norm=0.05,
        catalyst_proximity=0.60,
        reward_to_risk=2.5,
        max_drawdown_est=0.08,
        liquidity_score=0.95,
        correlation_to_book=0.20,
        new_signal=np.random.randn(20),
        notes="Breakout from 6-month consolidation",
    )

    print(f"Composite conviction: {rec.composite_score:.4f}")
    print(f"Tier: {engine.tier(rec.composite_score)}")
    print(f"Position size (max cap=1.0): {engine.position_size('idea_001'):.4f}")
    engine.record_outcome("idea_001", realised_pnl=0.032, realised_sharpe=1.9)
    print("Summary:", json.dumps(engine.summary(), indent=2))
