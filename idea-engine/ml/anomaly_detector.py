"""
anomaly_detector.py — Market anomaly and regime break detector.

Implements from scratch (numpy only):
  - IsolationForest
  - Local Outlier Factor (LOF)
  - CUSUM-based change-point detector
  - Mahalanobis distance scoring
  - AnomalyDetector ensemble class
  - Rolling anomaly scoring with regime context
  - Anomaly taxonomy: vol spike, correlation break, volume anomaly, momentum gap
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Anomaly taxonomy
# ---------------------------------------------------------------------------

class AnomalyType(str, Enum):
    VOL_SPIKE = "vol_spike"
    CORRELATION_BREAK = "correlation_break"
    VOLUME_ANOMALY = "volume_anomaly"
    MOMENTUM_GAP = "momentum_gap"
    REGIME_BREAK = "regime_break"
    OUTLIER = "outlier"
    UNKNOWN = "unknown"


@dataclass
class AnomalyEvent:
    timestamp: float
    score: float                      # 0-1, higher = more anomalous
    anomaly_type: AnomalyType
    component_scores: Dict[str, float]
    description: str
    features: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Isolation Forest (numpy only)
# ---------------------------------------------------------------------------

class _IsolationTree:
    """Single isolation tree trained on a subsample of data."""

    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.split_feature: Optional[int] = None
        self.split_value: Optional[float] = None
        self.left: Optional[_IsolationTree] = None
        self.right: Optional[_IsolationTree] = None
        self.size: int = 0

    def fit(self, X: np.ndarray, depth: int = 0) -> None:
        n, d = X.shape
        self.size = n

        if depth >= self.max_depth or n <= 1:
            return

        # Random feature and random split within [min, max]
        feat = np.random.randint(0, d)
        x_min, x_max = X[:, feat].min(), X[:, feat].max()
        if x_min == x_max:
            return

        split = np.random.uniform(x_min, x_max)
        self.split_feature = feat
        self.split_value = split

        left_mask = X[:, feat] < split
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            self.split_feature = None
            return

        self.left = _IsolationTree(self.max_depth)
        self.right = _IsolationTree(self.max_depth)
        self.left.fit(X[left_mask], depth + 1)
        self.right.fit(X[right_mask], depth + 1)

    def path_length(self, x: np.ndarray, depth: int = 0) -> float:
        if self.split_feature is None:
            # Leaf node: add average path length correction
            return depth + _avg_path_length(self.size)

        if x[self.split_feature] < self.split_value:
            child = self.left
        else:
            child = self.right

        if child is None:
            return depth + _avg_path_length(self.size)

        return child.path_length(x, depth + 1)


def _avg_path_length(n: int) -> float:
    """Expected path length for BST with n nodes (harmonic approximation)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    H = np.log(n - 1) + 0.5772156649  # Euler-Mascheroni constant
    return 2.0 * H - (2.0 * (n - 1) / n)


class IsolationForest:
    """
    Isolation Forest anomaly detector implemented from scratch.
    Anomaly score: higher = more anomalous (0-1 range).
    """

    def __init__(self, n_estimators: int = 100, max_samples: int = 256, contamination: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.trees: List[_IsolationTree] = []
        self._n_fit: int = 0

    def fit(self, X: np.ndarray) -> "IsolationForest":
        np.random.seed(self.random_state)
        n = X.shape[0]
        sample_size = min(self.max_samples, n)
        max_depth = int(np.ceil(np.log2(sample_size))) + 1

        self.trees = []
        self._n_fit = n

        for _ in range(self.n_estimators):
            idx = np.random.choice(n, size=sample_size, replace=False)
            tree = _IsolationTree(max_depth)
            tree.fit(X[idx])
            self.trees.append(tree)

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly score per sample in [0, 1]. Higher = more anomalous."""
        if not self.trees:
            raise RuntimeError("IsolationForest not fitted.")

        avg_paths = np.array([
            np.mean([t.path_length(X[i]) for t in self.trees])
            for i in range(X.shape[0])
        ])
        c = _avg_path_length(min(self.max_samples, self._n_fit))
        if c == 0:
            return np.zeros(len(avg_paths))
        scores = 2.0 ** (-avg_paths / c)  # standard IF formula
        return scores  # higher = more anomalous


# ---------------------------------------------------------------------------
# Local Outlier Factor (LOF) from scratch
# ---------------------------------------------------------------------------

class LOF:
    """
    Local Outlier Factor — compares local density of a point to its neighbours.
    Score > 1 indicates an outlier; we map to [0, 1] for the ensemble.
    """

    def __init__(self, k: int = 20):
        self.k = k
        self._X_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "LOF":
        self._X_train = X.copy()
        return self

    def _knn(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        dists = np.linalg.norm(self._X_train - x, axis=1)
        idx = np.argsort(dists)[:k + 1]  # +1 because x itself might be in train
        # Exclude self if present
        self_mask = np.all(self._X_train[idx] == x, axis=1)
        if self_mask.any():
            idx = idx[~self_mask][:k]
        else:
            idx = idx[:k]
        return idx, dists[idx]

    def _reach_dist(self, x: np.ndarray, neighbor_idx: int) -> float:
        neighbor = self._X_train[neighbor_idx]
        k_dist_neighbor = np.sort(np.linalg.norm(self._X_train - neighbor, axis=1))[self.k]
        return max(k_dist_neighbor, np.linalg.norm(x - neighbor))

    def _lrd(self, x: np.ndarray) -> float:
        idx, _ = self._knn(x, self.k)
        reach_dists = [self._reach_dist(x, i) for i in idx]
        mean_rd = np.mean(reach_dists) if reach_dists else 1e-9
        return 1.0 / (mean_rd + 1e-12)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return LOF scores. We normalise to [0,1] via sigmoid-like transform."""
        scores = []
        for x in X:
            lrd_x = self._lrd(x)
            idx, _ = self._knn(x, self.k)
            lrd_neighbors = [self._lrd(self._X_train[i]) for i in idx]
            lof = np.mean(lrd_neighbors) / (lrd_x + 1e-12) if lrd_neighbors else 1.0
            scores.append(lof)
        scores = np.array(scores)
        # Map to [0, 1]: lof=1 → 0, lof→∞ → 1
        return 1.0 - 1.0 / (scores + 1e-9)


# ---------------------------------------------------------------------------
# CUSUM change-point detector
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) change-point detector for a 1-D signal.
    Flags when the cumulative deviation from a running mean exceeds threshold.
    """

    def __init__(self, k: float = 0.5, h: float = 5.0, warmup: int = 30):
        self.k = k      # allowance parameter (slack)
        self.h = h      # decision threshold
        self.warmup = warmup

        self._s_pos: float = 0.0
        self._s_neg: float = 0.0
        self._mu: float = 0.0
        self._sigma: float = 1.0
        self._history: deque = deque(maxlen=warmup * 2)
        self._alarm: bool = False

    def reset(self) -> None:
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._alarm = False

    def update(self, value: float) -> Tuple[bool, float]:
        """Update with new observation. Returns (alarm, score)."""
        self._history.append(value)

        if len(self._history) >= self.warmup:
            arr = np.array(self._history)
            self._mu = arr.mean()
            self._sigma = max(arr.std(), 1e-9)

        z = (value - self._mu) / self._sigma
        self._s_pos = max(0.0, self._s_pos + z - self.k)
        self._s_neg = max(0.0, self._s_neg - z - self.k)

        score = max(self._s_pos, self._s_neg) / self.h
        self._alarm = score >= 1.0

        if self._alarm:
            self.reset()  # reset after alarm

        return self._alarm, min(score, 1.0)

    def batch_score(self, series: np.ndarray) -> np.ndarray:
        """Score an entire time series, return per-step scores."""
        self.reset()
        scores = []
        for v in series:
            _, s = self.update(float(v))
            scores.append(s)
        return np.array(scores)


# ---------------------------------------------------------------------------
# Mahalanobis distance anomaly scoring
# ---------------------------------------------------------------------------

class MahalanobisDetector:
    """
    Anomaly scoring via Mahalanobis distance from the training distribution.
    Uses robust shrinkage covariance estimation (Ledoit-Wolf shrinkage).
    """

    def __init__(self, shrinkage: float = 0.1):
        self.shrinkage = shrinkage
        self._mu: Optional[np.ndarray] = None
        self._inv_cov: Optional[np.ndarray] = None
        self._threshold_95: float = 0.0

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        self._mu = X.mean(axis=0)
        n, d = X.shape
        emp_cov = np.cov(X.T) if d > 1 else np.var(X).reshape(1, 1)
        # Ledoit-Wolf shrinkage toward identity
        identity = np.eye(d) * np.trace(emp_cov) / d
        cov = (1 - self.shrinkage) * emp_cov + self.shrinkage * identity
        try:
            self._inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self._inv_cov = np.linalg.pinv(cov)

        # Calibrate threshold from training data
        dists = self._raw_distances(X)
        self._threshold_95 = np.percentile(dists, 95)
        return self

    def _raw_distances(self, X: np.ndarray) -> np.ndarray:
        diff = X - self._mu
        return np.array([
            np.sqrt(max(0.0, d @ self._inv_cov @ d))
            for d in diff
        ])

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1]."""
        dists = self._raw_distances(X)
        # Normalise relative to 95th percentile of training distribution
        if self._threshold_95 > 0:
            scores = dists / (self._threshold_95 + 1e-9)
        else:
            scores = dists
        return np.clip(scores / (scores.max() + 1e-9), 0.0, 1.0)


# ---------------------------------------------------------------------------
# AnomalyDetector ensemble
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Ensemble anomaly detector combining IsolationForest, LOF, CUSUM,
    and Mahalanobis distance. Returns a unified anomaly score in [0, 1].

    Also classifies anomalies by type (vol spike, correlation break, etc.)
    and maintains a rolling score buffer for regime context.
    """

    def __init__(
        self,
        n_estimators: int = 80,
        lof_k: int = 15,
        cusum_h: float = 4.5,
        window: int = 60,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.window = window
        self.weights = weights or {
            "iforest": 0.35,
            "lof": 0.25,
            "mahalanobis": 0.25,
            "cusum": 0.15,
        }

        self.iforest = IsolationForest(n_estimators=n_estimators)
        self.lof = LOF(k=lof_k)
        self.mahal = MahalanobisDetector()
        self.cusum = CUSUMDetector(h=cusum_h)

        self._fitted = False
        self._score_history: deque = deque(maxlen=window * 3)
        self._events: List[AnomalyEvent] = []

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """Fit all sub-detectors on historical feature matrix X."""
        self.iforest.fit(X)
        self.lof.fit(X)
        self.mahal.fit(X)
        # CUSUM is online — fit on the first feature dimension
        self.cusum.batch_score(X[:, 0])
        self._fitted = True
        return self

    def score(self, x: np.ndarray) -> float:
        """Score a single observation vector. Returns anomaly score in [0, 1]."""
        return self.score_batch(x.reshape(1, -1))[0]

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Score a batch of observations."""
        if not self._fitted:
            raise RuntimeError("AnomalyDetector not fitted. Call fit() first.")

        s_if = self.iforest.score_samples(X)
        s_lof = self.lof.score_samples(X)
        s_mah = self.mahal.score_samples(X)
        s_cusum = np.array([self.cusum.update(X[i, 0])[1] for i in range(len(X))])

        ensemble = (
            self.weights["iforest"] * s_if
            + self.weights["lof"] * s_lof
            + self.weights["mahalanobis"] * s_mah
            + self.weights["cusum"] * s_cusum
        )
        ensemble = np.clip(ensemble, 0.0, 1.0)

        for i, score in enumerate(ensemble):
            self._score_history.append(score)

        return ensemble

    def detect(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> List[AnomalyEvent]:
        """
        Run full detection pipeline. Returns list of AnomalyEvent for
        observations that exceed the dynamic threshold.
        """
        scores = self.score_batch(X)
        threshold = self._dynamic_threshold()
        events = []

        for i, score in enumerate(scores):
            if score >= threshold:
                ts = float(timestamps[i]) if timestamps is not None else time.time()
                atype = self._classify(X[i], feature_names)
                comp = self._component_scores(X[i:i+1])
                event = AnomalyEvent(
                    timestamp=ts,
                    score=float(score),
                    anomaly_type=atype,
                    component_scores=comp,
                    description=self._describe(atype, score),
                    features=X[i].copy(),
                )
                events.append(event)
                self._events.append(event)

        return events

    def _dynamic_threshold(self) -> float:
        """Adaptive threshold: mean + 2*std of recent scores, floored at 0.6."""
        if len(self._score_history) < 10:
            return 0.65
        arr = np.array(self._score_history)
        return float(np.clip(arr.mean() + 2.0 * arr.std(), 0.55, 0.90))

    def _component_scores(self, X: np.ndarray) -> Dict[str, float]:
        s_if = float(self.iforest.score_samples(X)[0])
        s_lof = float(self.lof.score_samples(X)[0])
        s_mah = float(self.mahal.score_samples(X)[0])
        _, s_cusum = self.cusum.update(float(X[0, 0]))
        return {
            "isolation_forest": round(s_if, 4),
            "lof": round(s_lof, 4),
            "mahalanobis": round(s_mah, 4),
            "cusum": round(s_cusum, 4),
        }

    def _classify(self, x: np.ndarray, feature_names: Optional[List[str]]) -> AnomalyType:
        """Heuristic taxonomy based on which feature dimension is most extreme."""
        if feature_names is None:
            return AnomalyType.OUTLIER

        fn = [f.lower() for f in feature_names]

        # Find the most deviated feature relative to training distribution
        if self.mahal._mu is not None:
            deviations = np.abs(x - self.mahal._mu) / (
                np.sqrt(np.diag(np.linalg.inv(self.mahal._inv_cov))) + 1e-9
            )
            top_feat_idx = int(np.argmax(deviations))
            if top_feat_idx < len(fn):
                name = fn[top_feat_idx]
                if any(k in name for k in ("vol", "atr", "vix", "variance")):
                    return AnomalyType.VOL_SPIKE
                if any(k in name for k in ("corr", "beta", "covar")):
                    return AnomalyType.CORRELATION_BREAK
                if any(k in name for k in ("vol_", "volume", "turnover", "flow")):
                    return AnomalyType.VOLUME_ANOMALY
                if any(k in name for k in ("mom", "ret", "gap", "price")):
                    return AnomalyType.MOMENTUM_GAP

        return AnomalyType.REGIME_BREAK

    def _describe(self, atype: AnomalyType, score: float) -> str:
        severity = "critical" if score > 0.85 else "elevated" if score > 0.70 else "moderate"
        labels = {
            AnomalyType.VOL_SPIKE: "Volatility spike detected",
            AnomalyType.CORRELATION_BREAK: "Correlation structure break",
            AnomalyType.VOLUME_ANOMALY: "Abnormal volume pattern",
            AnomalyType.MOMENTUM_GAP: "Momentum/price gap anomaly",
            AnomalyType.REGIME_BREAK: "Potential regime transition",
            AnomalyType.OUTLIER: "Multi-dimensional outlier",
            AnomalyType.UNKNOWN: "Unclassified anomaly",
        }
        return f"[{severity.upper()}] {labels.get(atype, 'Anomaly')} (score={score:.3f})"

    def rolling_scores(self) -> np.ndarray:
        """Return recent rolling anomaly scores."""
        return np.array(self._score_history)

    def regime_context(self) -> Dict[str, float]:
        """Summarise current anomaly regime."""
        arr = self.rolling_scores()
        if len(arr) == 0:
            return {}
        return {
            "mean_score": float(arr.mean()),
            "max_score": float(arr.max()),
            "pct_elevated": float((arr > 0.65).mean()),
            "current_regime": "stressed" if arr[-min(5, len(arr)):].mean() > 0.65 else "normal",
        }

    def recent_events(self, n: int = 10) -> List[AnomalyEvent]:
        return list(self._events[-n:])

    def summary(self) -> Dict:
        ctx = self.regime_context()
        return {
            "fitted": self._fitted,
            "total_events": len(self._events),
            "regime_context": ctx,
            "event_type_counts": {
                t.value: sum(1 for e in self._events if e.anomaly_type == t)
                for t in AnomalyType
            },
        }


# ---------------------------------------------------------------------------
# Taxonomy-specific detectors (market-aware wrappers)
# ---------------------------------------------------------------------------

class VolSpikeDetector:
    """Focused volatility spike detector using realised vol ratio."""

    def __init__(self, short_window: int = 5, long_window: int = 21, threshold: float = 2.5):
        self.short_w = short_window
        self.long_w = long_window
        self.threshold = threshold
        self._returns: deque = deque(maxlen=long_window * 3)

    def update(self, ret: float) -> Tuple[bool, float]:
        self._returns.append(ret)
        arr = np.array(self._returns)
        if len(arr) < self.long_w:
            return False, 0.0
        short_vol = np.std(arr[-self.short_w:]) * np.sqrt(252)
        long_vol = np.std(arr[-self.long_w:]) * np.sqrt(252)
        ratio = short_vol / (long_vol + 1e-9)
        score = np.clip((ratio - 1.0) / (self.threshold - 1.0), 0.0, 1.0)
        return bool(ratio >= self.threshold), float(score)


class CorrelationBreakDetector:
    """Detects breaks in cross-asset correlation structure."""

    def __init__(self, window: int = 30, threshold: float = 0.4):
        self.window = window
        self.threshold = threshold
        self._data: deque = deque(maxlen=window * 2)

    def update(self, multi_returns: np.ndarray) -> Tuple[bool, float]:
        self._data.append(multi_returns.copy())
        if len(self._data) < self.window:
            return False, 0.0

        data = np.array(self._data)
        corr_recent = np.corrcoef(data[-self.window // 2:].T)
        corr_baseline = np.corrcoef(data[:self.window].T)

        diff = np.abs(corr_recent - corr_baseline)
        # Frobenius norm of difference, normalised
        n = diff.shape[0]
        off_diag = (diff.sum() - np.trace(diff)) / max((n * n - n), 1)
        score = np.clip(off_diag / self.threshold, 0.0, 1.0)
        return bool(off_diag >= self.threshold), float(score)


# ---------------------------------------------------------------------------
# Quick demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    n_train, n_test, d = 500, 50, 6
    X_train = np.random.randn(n_train, d)
    X_test = np.random.randn(n_test, d)
    # Inject anomalies into last 5 rows
    X_test[-5:] += np.random.uniform(4, 8, (5, d))

    feature_names = ["vol_1d", "vol_5d", "corr_spy", "volume_zscore", "mom_5d", "atr_14"]

    detector = AnomalyDetector(n_estimators=50, lof_k=10)
    detector.fit(X_train)
    events = detector.detect(X_test, feature_names=feature_names)

    print(f"Detected {len(events)} anomaly events")
    for ev in events:
        print(f"  {ev.description}  components={ev.component_scores}")
    print("Regime context:", detector.regime_context())
