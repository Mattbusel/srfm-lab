"""
signal_combiner.py -- multi-signal combination framework.

Provides IC-weighted, rank-based, and Ridge-regression ensemble combiners.
Also detects and resolves conflicting signal directions using BH mass state.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import logging

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    import warnings
    warnings.warn("numpy not found; EnsembleCombiner disabled", ImportWarning)

logger = logging.getLogger(__name__)

# -- BH physics constants (from srfm_core)
BH_MASS_THRESH = 1.92
BH_DECAY       = 0.924
BH_COLLAPSE    = 0.992

# -- rolling IC window for ICWeightedCombiner
DEFAULT_IC_WINDOW = 20
# -- minimum absolute IC to carry nonzero weight
MIN_IC_WEIGHT    = 1e-6


@dataclass
class CompositeSignal:
    """Output of any combiner: a single score with attribution metadata."""
    score: float                          # final composite in [-1, 1]
    confidence: float                     # 0..1 magnitude of agreement
    components: dict[str, float]          # {signal_id: raw_value}
    weights: dict[str, float]             # {signal_id: weight_applied}
    regime: Optional[str] = None          # trending | reverting | neutral
    conflicts: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# IC-weighted combiner
# ------------------------------------------------------------------

class ICWeightedCombiner:
    """
    Weights each signal by its trailing IC (information coefficient).
    IC is estimated as the sign-agreement between signal and realized return.
    """

    def __init__(self, ic_window: int = DEFAULT_IC_WINDOW):
        self.ic_window = ic_window
        # -- signal_id -> deque of (signal_value, realized_return) pairs
        self._history: dict[str, deque[tuple[float, float]]] = {}
        # -- cached IC per signal
        self._ic_cache: dict[str, float] = {}

    def _compute_ic(self, signal_id: str) -> float:
        """
        Compute Pearson correlation between lagged signal and returns.
        Falls back to 0.0 when insufficient data.
        """
        hist = list(self._history.get(signal_id, []))
        if len(hist) < 2:
            return 0.0
        signals  = [h[0] for h in hist]
        returns  = [h[1] for h in hist]
        n = len(signals)
        mean_s = sum(signals) / n
        mean_r = sum(returns) / n
        num = sum((s - mean_s) * (r - mean_r) for s, r in zip(signals, returns))
        den_s = math.sqrt(sum((s - mean_s) ** 2 for s in signals))
        den_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns))
        denom = den_s * den_r
        if denom < 1e-12:
            return 0.0
        return num / denom

    def update_ic(self, signal_id: str, signal_value: float, realized_return: float) -> None:
        """Record a (signal, return) pair for IC estimation."""
        if signal_id not in self._history:
            self._history[signal_id] = deque(maxlen=self.ic_window)
        self._history[signal_id].append((signal_value, realized_return))
        self._ic_cache[signal_id] = self._compute_ic(signal_id)

    def get_ic(self, signal_id: str) -> float:
        return self._ic_cache.get(signal_id, 0.0)

    def combine(self, signals: dict[str, float]) -> CompositeSignal:
        """
        Weighted average: score = sum(w_i * s_i) / sum(w_i)
        where w_i = max(IC_i, 0) clipped to MIN_IC_WEIGHT.
        """
        weights: dict[str, float] = {}
        for sid in signals:
            ic = self.get_ic(sid)
            weights[sid] = max(ic, MIN_IC_WEIGHT)

        total_w = sum(weights.values())
        if total_w < 1e-12:
            # -- fallback: equal weights
            total_w = len(signals) or 1
            weights = {sid: 1.0 for sid in signals}

        score = sum(weights[sid] * v for sid, v in signals.items()) / total_w
        score = max(-1.0, min(1.0, score))

        norm_weights = {sid: w / total_w for sid, w in weights.items()}
        confidence = abs(score)

        return CompositeSignal(
            score=score,
            confidence=confidence,
            components=dict(signals),
            weights=norm_weights,
        )


# ------------------------------------------------------------------
# Rank combiner
# ------------------------------------------------------------------

class RankCombiner:
    """
    Convert each signal to cross-sectional rank, average ranks,
    then normalize the result to [-1, 1].
    """

    def combine(self, signals: dict[str, float]) -> CompositeSignal:
        if not signals:
            return CompositeSignal(score=0.0, confidence=0.0, components={}, weights={})

        items = list(signals.items())
        n = len(items)

        # -- rank in ascending order (1 = most negative signal)
        sorted_by_val = sorted(items, key=lambda x: x[1])
        ranks: dict[str, int] = {sid: i + 1 for i, (sid, _) in enumerate(sorted_by_val)}

        avg_rank = sum(ranks.values()) / n
        # -- normalize avg_rank from [1, n] to [-1, 1]
        if n > 1:
            score = 2.0 * (avg_rank - 1) / (n - 1) - 1.0
        else:
            score = 0.0

        score = max(-1.0, min(1.0, score))
        equal_w = 1.0 / n
        weights = {sid: equal_w for sid in signals}

        return CompositeSignal(
            score=score,
            confidence=abs(score),
            components=dict(signals),
            weights=weights,
        )


# ------------------------------------------------------------------
# Ensemble combiner (Ridge regression)
# ------------------------------------------------------------------

class EnsembleCombiner:
    """
    Learns optimal signal weights via Ridge regression on (signals, returns).
    Falls back to equal-weight average if numpy is unavailable.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha        # Ridge regularization
        self._coefs: Optional["np.ndarray"] = None
        self._signal_ids: list[str] = []
        self._fitted = False

    def fit(self, X: "np.ndarray", y: "np.ndarray", signal_ids: list[str]) -> None:
        """
        Fit Ridge: w = (X'X + alpha*I)^{-1} X'y
        X shape: (n_samples, n_signals); y shape: (n_samples,)
        """
        if not _HAS_NUMPY:
            logger.warning("numpy unavailable; EnsembleCombiner.fit skipped")
            return
        self._signal_ids = signal_ids
        n_features = X.shape[1]
        A = X.T @ X + self.alpha * np.eye(n_features)
        b = X.T @ y
        try:
            self._coefs = np.linalg.solve(A, b)
            self._fitted = True
        except np.linalg.LinAlgError as exc:
            logger.error("EnsembleCombiner fit failed: %s", exc)
            self._coefs = None
            self._fitted = False

    def predict(self, signals: dict[str, float]) -> CompositeSignal:
        """
        Predict composite score using fitted coefficients.
        Signals not seen during fit are ignored.
        """
        if not _HAS_NUMPY or not self._fitted or self._coefs is None:
            # -- equal-weight fallback
            if not signals:
                return CompositeSignal(score=0.0, confidence=0.0, components={}, weights={})
            avg = sum(signals.values()) / len(signals)
            eq_w = 1.0 / len(signals)
            return CompositeSignal(
                score=max(-1.0, min(1.0, avg)),
                confidence=abs(avg),
                components=dict(signals),
                weights={sid: eq_w for sid in signals},
            )

        x_vec = np.array([signals.get(sid, 0.0) for sid in self._signal_ids])
        raw_score = float(x_vec @ self._coefs)
        score = max(-1.0, min(1.0, raw_score))

        total_coef = sum(abs(c) for c in self._coefs) or 1.0
        weights = {
            sid: abs(float(self._coefs[i])) / total_coef
            for i, sid in enumerate(self._signal_ids)
        }

        return CompositeSignal(
            score=score,
            confidence=abs(score),
            components=dict(signals),
            weights=weights,
        )


# ------------------------------------------------------------------
# Signal conflict detector
# ------------------------------------------------------------------

class SignalConflictDetector:
    """
    Detects when signals disagree on direction and resolves via BH mass.
    """

    def detect_conflicts(self, signals: dict[str, float]) -> list[str]:
        """
        Return list of signal_ids whose sign conflicts with the majority sign.
        Empty if all signals agree or if there are fewer than 2 signals.
        """
        if len(signals) < 2:
            return []
        values = list(signals.values())
        pos_count = sum(1 for v in values if v > 0)
        neg_count = sum(1 for v in values if v < 0)
        majority_positive = pos_count >= neg_count
        conflicts = []
        for sid, v in signals.items():
            if v == 0.0:
                continue
            if majority_positive and v < 0:
                conflicts.append(sid)
            elif not majority_positive and v > 0:
                conflicts.append(sid)
        return conflicts

    def resolve_conflict(self, signals: dict[str, float], bh_mass: float = 0.0) -> float:
        """
        Use BH mass to determine trending vs. reverting regime and choose
        the appropriate directional bias when signals conflict.

        bh_mass > BH_MASS_THRESH: trending -- follow majority direction.
        bh_mass <= BH_MASS_THRESH: reverting -- down-weight extremes.
        """
        if not signals:
            return 0.0

        values = list(signals.values())
        mean_val = sum(values) / len(values)
        conflicts = self.detect_conflicts(signals)

        if not conflicts:
            return max(-1.0, min(1.0, mean_val))

        if bh_mass > BH_MASS_THRESH:
            # -- trending: use median to suppress outlier conflicting signals
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            if n % 2 == 1:
                resolved = sorted_vals[n // 2]
            else:
                resolved = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
        else:
            # -- reverting: use trimmed mean excluding extremes
            sorted_vals = sorted(values)
            trim = max(1, len(sorted_vals) // 4)
            trimmed = sorted_vals[trim:-trim] if len(sorted_vals) > 2 * trim else sorted_vals
            resolved = sum(trimmed) / len(trimmed) if trimmed else mean_val

        return max(-1.0, min(1.0, resolved))


# ------------------------------------------------------------------
# Top-level SignalCombiner facade
# ------------------------------------------------------------------

class SignalCombiner:
    """
    Combines multiple alpha signals into a single composite score.
    Supports IC-weighted, rank-weighted, and ensemble methods.
    """

    def __init__(
        self,
        method: str = "ic_weighted",     # ic_weighted | rank | ensemble
        ic_window: int = DEFAULT_IC_WINDOW,
        ridge_alpha: float = 0.1,
    ):
        self.method = method
        self._ic_combiner    = ICWeightedCombiner(ic_window)
        self._rank_combiner  = RankCombiner()
        self._ens_combiner   = EnsembleCombiner(ridge_alpha)
        self._conflict_det   = SignalConflictDetector()

    def combine(
        self,
        signals: dict[str, float],
        bh_mass: float = 0.0,
        regime: Optional[str] = None,
    ) -> CompositeSignal:
        """
        Combine signals using the configured method.
        Conflict detection always runs; BH mass used for resolution.
        """
        conflicts = self._conflict_det.detect_conflicts(signals)

        if self.method == "rank":
            result = self._rank_combiner.combine(signals)
        elif self.method == "ensemble":
            result = self._ens_combiner.predict(signals)
        else:
            result = self._ic_combiner.combine(signals)

        # -- if conflicts exist, blend in the BH-resolved score
        if conflicts:
            resolved = self._conflict_det.resolve_conflict(signals, bh_mass)
            # -- 70% combiner output, 30% BH-resolved
            blended = 0.7 * result.score + 0.3 * resolved
            result = CompositeSignal(
                score=max(-1.0, min(1.0, blended)),
                confidence=result.confidence * 0.85,   # -- reduce confidence on conflict
                components=result.components,
                weights=result.weights,
                regime=regime,
                conflicts=conflicts,
            )
        else:
            result.regime = regime

        return result

    def update_ic(self, signal_id: str, signal_value: float, realized_return: float) -> None:
        """Pass through to ICWeightedCombiner for continuous IC estimation."""
        self._ic_combiner.update_ic(signal_id, signal_value, realized_return)

    def fit_ensemble(
        self,
        X: "np.ndarray",
        y: "np.ndarray",
        signal_ids: list[str],
    ) -> None:
        self._ens_combiner.fit(X, y, signal_ids)
