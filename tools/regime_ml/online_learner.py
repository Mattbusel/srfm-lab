"""
online_learner.py — Online / streaming ML for next-bar direction prediction.

Uses the `river` library for all incremental models.  Falls back gracefully
to a minimal pure-Python SGD classifier if river is not installed.

Features (supplied by caller or pulled from FeatureStore)
    bh_mass         float  [0, 1]   BH pressure mass
    ctl_count       float           recent control-loop count proxy
    garch_vol       float           GARCH(1,1) forecast volatility
    ou_zscore       float           OU-process z-score
    hurst           float  [0, 1]   ensemble Hurst exponent
    entropy         float  [0, 1]   permutation entropy
    hour_of_day     int    [0, 23]
    day_of_week     int    [0, 6]

Target: next-bar return direction  +1 (up) or 0 (down/flat).

Evaluation: PREQUENTIAL (test-then-train) with rolling accuracy.

Concept drift detection
    ADWIN        — adaptive windowing (detects mean shift)
    Page-Hinkley — sequential test (detects step change in loss)
    KSWIN        — Kolmogorov-Smirnov windowed test

Ensemble: majority vote across HoeffdingTree, AdaptiveRandomForest,
          and PassiveAggressiveClassifier.

Model persistence: pickled to online_model.pkl every 100 updates.
"""

from __future__ import annotations

import math
import os
import pickle
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional river import
# ---------------------------------------------------------------------------

try:
    import river  # noqa: F401
    from river import drift, ensemble, linear_model, metrics, preprocessing, tree  # type: ignore

    _RIVER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RIVER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "online_model.pkl"
_SAVE_EVERY = 100
_ROLLING_ACC_WINDOW = 500
_FEATURE_NAMES = [
    "bh_mass",
    "ctl_count",
    "garch_vol",
    "ou_zscore",
    "hurst",
    "entropy",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OnlinePrediction:
    prediction: int          # 0 or 1
    probability: float       # P(up), 0–1
    confidence: float        # |probability - 0.5| * 2, 0–1
    rolling_accuracy: float  # prequential rolling accuracy
    drift_detected: bool
    drift_method: Optional[str]  # "adwin" / "page_hinkley" / "kswin"
    model_updates: int
    feature_importances: Dict[str, float]


# ---------------------------------------------------------------------------
# Fallback pure-Python SGD classifier (no river)
# ---------------------------------------------------------------------------


class _FallbackSGD:
    """Minimal online logistic regression via SGD — no external deps."""

    def __init__(self, lr: float = 0.01, n_features: int = 10) -> None:
        self.lr = lr
        self.w = np.zeros(n_features)
        self.b = 0.0

    def _sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))

    def predict_proba_one(self, x: Dict[str, float]) -> Dict[int, float]:
        vec = np.array([x.get(f, 0.0) for f in _FEATURE_NAMES])
        p = self._sigmoid(float(np.dot(self.w, vec)) + self.b)
        return {1: p, 0: 1.0 - p}

    def learn_one(self, x: Dict[str, float], y: int) -> None:
        vec = np.array([x.get(f, 0.0) for f in _FEATURE_NAMES])
        p = self._sigmoid(float(np.dot(self.w, vec)) + self.b)
        err = float(y) - p
        self.w += self.lr * err * vec
        self.b += self.lr * err


# ---------------------------------------------------------------------------
# Drift detectors (pure Python fallback)
# ---------------------------------------------------------------------------


class _PageHinkley:
    """Page-Hinkley sequential change-detection test."""

    def __init__(self, delta: float = 0.005, lambda_: float = 50.0) -> None:
        self.delta = delta
        self.lambda_ = lambda_
        self._sum = 0.0
        self._min_sum = 0.0
        self._n = 0
        self._mean = 0.0

    def update(self, x: float) -> bool:
        self._n += 1
        self._mean += (x - self._mean) / self._n
        self._sum += x - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)
        return (self._sum - self._min_sum) > self.lambda_

    def reset(self) -> None:
        self._sum = 0.0
        self._min_sum = 0.0
        self._n = 0
        self._mean = 0.0


class _ADWIN:
    """
    Simplified ADWIN (Adaptive Windowing) drift detector.
    Uses two-half variance test on a sliding buffer.
    """

    def __init__(self, delta: float = 0.002, window: int = 300) -> None:
        self.delta = delta
        self._buf: Deque[float] = deque(maxlen=window)

    def update(self, x: float) -> bool:
        self._buf.append(x)
        n = len(self._buf)
        if n < 30:
            return False
        arr = np.array(self._buf)
        half = n // 2
        m1, m2 = arr[:half].mean(), arr[half:].mean()
        eps_cut = math.sqrt((1.0 / (2.0 * half)) * math.log(4.0 * n / self.delta))
        return abs(m1 - m2) > eps_cut


class _KSWIN:
    """
    Simplified Kolmogorov-Smirnov windowed drift detector.
    Compares the last `stat_size` samples vs the rest of the window.
    """

    def __init__(self, alpha: float = 0.005, window: int = 200, stat_size: int = 30) -> None:
        self.alpha = alpha
        self.stat_size = stat_size
        self._buf: Deque[float] = deque(maxlen=window)

    def update(self, x: float) -> bool:
        self._buf.append(x)
        n = len(self._buf)
        if n < self.stat_size * 2:
            return False
        arr = np.array(self._buf)
        recent = arr[-self.stat_size:]
        older  = arr[:-self.stat_size]
        # Approximate KS stat
        ks = float(np.max(np.abs(
            np.sort(recent) / len(recent) - np.sort(np.random.choice(older, len(recent), replace=False)) / len(recent)
        )))
        # Critical value for alpha=0.005: ~1.36 / sqrt(n)
        crit = 1.36 / math.sqrt(self.stat_size)
        return ks > crit


# ---------------------------------------------------------------------------
# Feature encoding helpers
# ---------------------------------------------------------------------------


def _encode_features(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Encode raw feature dict into model-ready features.
    Cyclically encodes hour_of_day and day_of_week.
    """
    out: Dict[str, float] = {}
    for k in ("bh_mass", "ctl_count", "garch_vol", "ou_zscore", "hurst", "entropy"):
        v = raw.get(k, 0.0)
        out[k] = float(v) if v is not None else 0.0

    h = int(raw.get("hour_of_day", 0)) % 24
    out["hour_of_day_sin"] = math.sin(2 * math.pi * h / 24)
    out["hour_of_day_cos"] = math.cos(2 * math.pi * h / 24)

    d = int(raw.get("day_of_week", 0)) % 7
    out["day_of_week_sin"] = math.sin(2 * math.pi * d / 7)
    out["day_of_week_cos"] = math.cos(2 * math.pi * d / 7)
    return out


# ---------------------------------------------------------------------------
# River model wrappers
# ---------------------------------------------------------------------------


def _build_river_pipeline():
    """Build three River models for the ensemble."""
    if not _RIVER_AVAILABLE:
        return None

    ht = preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier(
        grace_period=50,
        delta=1e-5,
        leaf_prediction="nb",
    )
    arf = preprocessing.StandardScaler() | ensemble.AdaptiveRandomForestClassifier(
        n_models=10,
        seed=42,
    )
    pa = preprocessing.StandardScaler() | linear_model.PAClassifier(C=0.01, mode=1)
    return [("ht", ht), ("arf", arf), ("pa", pa)]


# ---------------------------------------------------------------------------
# OnlineLearner
# ---------------------------------------------------------------------------


class OnlineLearner:
    """
    Streaming ML ensemble with concept-drift detection and prequential eval.

    Parameters
    ----------
    model_path : str | Path
        Where to pickle model state periodically.
    save_every : int
        Persist model every N updates.
    rolling_window : int
        Window for rolling prequential accuracy.
    """

    def __init__(
        self,
        model_path: str | Path = _MODEL_PATH,
        save_every: int = _SAVE_EVERY,
        rolling_window: int = _ROLLING_ACC_WINDOW,
    ) -> None:
        self.model_path = Path(model_path)
        self.save_every = save_every

        self._n_updates: int = 0
        self._rolling_correct: Deque[int] = deque(maxlen=rolling_window)

        # Drift detectors
        self._adwin = _ADWIN()
        self._ph = _PageHinkley()
        self._kswin = _KSWIN()
        self._drift_count: int = 0

        # Feature importance via permutation — track correlation of each feature with correct prediction
        self._feat_imp: Dict[str, float] = {f: 0.0 for f in _FEATURE_NAMES}
        self._feat_imp_alpha: float = 0.005

        # Build models
        if _RIVER_AVAILABLE:
            self._models = _build_river_pipeline()
            self._metric = metrics.Accuracy()
        else:
            self._models = None
            self._fallback = _FallbackSGD(n_features=len(_FEATURE_NAMES))

        # Try to load saved state
        self._load()

    # ------------------------------------------------------------------
    # Core update / predict
    # ------------------------------------------------------------------

    def update(
        self,
        raw_features: Dict[str, Any],
        label: int,
    ) -> OnlinePrediction:
        """
        PREQUENTIAL step: predict first, then train.

        Parameters
        ----------
        raw_features : dict with keys matching _FEATURE_NAMES (pre-encoding)
        label        : actual next-bar direction (0 or 1)

        Returns
        -------
        OnlinePrediction with prediction made BEFORE learning this label.
        """
        x = _encode_features(raw_features)

        # --- Predict (test) ---
        prob, pred = self._predict_ensemble(x)
        correct = int(pred == label)
        self._rolling_correct.append(correct)

        # --- Concept drift on prediction error ---
        loss = 1.0 - correct
        adwin_drift = self._adwin.update(loss)
        ph_drift = self._ph.update(loss)
        kswin_drift = self._kswin.update(loss)
        drift_detected = adwin_drift or ph_drift or kswin_drift
        drift_method: Optional[str] = None
        if adwin_drift:
            drift_method = "adwin"
        elif ph_drift:
            drift_method = "page_hinkley"
        elif kswin_drift:
            drift_method = "kswin"
        if drift_detected:
            self._drift_count += 1
            self._on_drift()

        # --- Update feature importance (EMA of |feature| * correct) ---
        for fname in _FEATURE_NAMES:
            fval = abs(x.get(fname, 0.0))
            contrib = fval * correct
            self._feat_imp[fname] = (
                (1 - self._feat_imp_alpha) * self._feat_imp[fname]
                + self._feat_imp_alpha * contrib
            )

        # --- Train ---
        self._train_ensemble(x, label)
        self._n_updates += 1

        # --- Persist ---
        if self._n_updates % self.save_every == 0:
            self._save()

        rolling_acc = (
            sum(self._rolling_correct) / len(self._rolling_correct)
            if self._rolling_correct
            else 0.5
        )

        return OnlinePrediction(
            prediction=pred,
            probability=prob,
            confidence=abs(prob - 0.5) * 2.0,
            rolling_accuracy=rolling_acc,
            drift_detected=drift_detected,
            drift_method=drift_method,
            model_updates=self._n_updates,
            feature_importances=dict(self._feat_imp),
        )

    def predict_only(self, raw_features: Dict[str, Any]) -> OnlinePrediction:
        """Predict without updating the model."""
        x = _encode_features(raw_features)
        prob, pred = self._predict_ensemble(x)
        rolling_acc = (
            sum(self._rolling_correct) / len(self._rolling_correct)
            if self._rolling_correct
            else 0.5
        )
        return OnlinePrediction(
            prediction=pred,
            probability=prob,
            confidence=abs(prob - 0.5) * 2.0,
            rolling_accuracy=rolling_acc,
            drift_detected=False,
            drift_method=None,
            model_updates=self._n_updates,
            feature_importances=dict(self._feat_imp),
        )

    # ------------------------------------------------------------------
    # Ensemble prediction
    # ------------------------------------------------------------------

    def _predict_ensemble(self, x: Dict[str, float]) -> Tuple[float, int]:
        if _RIVER_AVAILABLE and self._models:
            probs = []
            for _, model in self._models:
                try:
                    p = model.predict_proba_one(x)
                    probs.append(p.get(1, 0.5))
                except Exception:
                    probs.append(0.5)
            prob = float(np.mean(probs))
        else:
            prob = self._fallback.predict_proba_one(x).get(1, 0.5)
        pred = 1 if prob >= 0.5 else 0
        return prob, pred

    def _train_ensemble(self, x: Dict[str, float], y: int) -> None:
        if _RIVER_AVAILABLE and self._models:
            for _, model in self._models:
                try:
                    model.learn_one(x, y)
                except Exception:
                    pass
        else:
            self._fallback.learn_one(x, y)

    # ------------------------------------------------------------------
    # Drift handling
    # ------------------------------------------------------------------

    def _on_drift(self) -> None:
        """Reset drift detectors; optionally retrain from scratch."""
        self._adwin = _ADWIN()
        self._ph = _PageHinkley()
        self._kswin = _KSWIN()
        # Don't reset the models — let ARF/HT handle concept drift internally

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "n_updates": self._n_updates,
            "feat_imp": self._feat_imp,
            "rolling_correct": list(self._rolling_correct),
            "drift_count": self._drift_count,
        }
        if _RIVER_AVAILABLE and self._models:
            state["models"] = self._models
        else:
            state["fallback_w"] = self._fallback.w.tolist()
            state["fallback_b"] = self._fallback.b
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(state, f)
        except Exception:
            pass

    def _load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            with open(self.model_path, "rb") as f:
                state = pickle.load(f)
            self._n_updates = state.get("n_updates", 0)
            self._feat_imp = state.get("feat_imp", self._feat_imp)
            self._rolling_correct = deque(
                state.get("rolling_correct", []), maxlen=_ROLLING_ACC_WINDOW
            )
            self._drift_count = state.get("drift_count", 0)
            if _RIVER_AVAILABLE and "models" in state:
                self._models = state["models"]
            elif not _RIVER_AVAILABLE and "fallback_w" in state:
                self._fallback.w = np.array(state["fallback_w"])
                self._fallback.b = state["fallback_b"]
        except Exception:
            pass  # corrupt pickle → fresh start

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        acc = (
            sum(self._rolling_correct) / len(self._rolling_correct)
            if self._rolling_correct
            else 0.0
        )
        sorted_fi = sorted(self._feat_imp.items(), key=lambda kv: kv[1], reverse=True)
        fi_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted_fi[:5])
        return (
            f"OnlineLearner: updates={self._n_updates}  "
            f"rolling_acc={acc:.3f}  drift_events={self._drift_count}\n"
            f"  Top features: {fi_str}"
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import csv
    from pathlib import Path

    csv_path = Path(__file__).parent.parent.parent / "data" / "NDX_hourly_poly.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(open(csv_path)))
        closes = [float(r.get("close", r.get("Close", 0))) for r in rows[:1000]]
    else:
        rng = np.random.default_rng(3)
        closes = (15000.0 + np.cumsum(rng.normal(0.1, 20, 1000))).tolist()

    learner = OnlineLearner(model_path=Path("/tmp/online_model_demo.pkl"))
    rng2 = np.random.default_rng(99)
    n_correct = 0
    n_total = 0

    for i in range(10, len(closes) - 1):
        ret = math.log(closes[i] / closes[i - 1])
        next_ret = math.log(closes[i + 1] / closes[i])
        label = 1 if next_ret > 0 else 0
        raw = {
            "bh_mass":    float(rng2.uniform(0, 1)),
            "ctl_count":  float(rng2.integers(0, 10)),
            "garch_vol":  abs(ret),
            "ou_zscore":  ret / 0.01 if abs(ret) < 1 else 0.0,
            "hurst":      0.5 + rng2.uniform(-0.2, 0.2),
            "entropy":    rng2.uniform(0.4, 0.9),
            "hour_of_day": (i % 24),
            "day_of_week": (i // 24) % 7,
        }
        result = learner.update(raw, label)
        n_correct += int(result.prediction == label)
        n_total += 1

    print(learner.summary())
    print(f"Naive accuracy over demo: {n_correct / max(1, n_total):.3f}")


if __name__ == "__main__":
    _demo()
