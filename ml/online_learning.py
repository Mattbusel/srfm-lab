"""
ml/online_learning.py
Online learning algorithms for live trading signals.

Each algorithm supports incremental updates (fit_one), prediction,
serialization (to_dict / from_dict), and optional feature importance.

No em dashes. Uses numpy and scipy only.
"""

from __future__ import annotations

import math
import hashlib
import struct
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import expit as _scipy_sigmoid


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _clip(x: float, lo: float = -30.0, hi: float = 30.0) -> float:
    return max(lo, min(hi, x))


def _soft_threshold(x: float, threshold: float) -> float:
    """Proximal operator for L1 regularization."""
    if x > threshold:
        return x - threshold
    elif x < -threshold:
        return x + threshold
    return 0.0


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


# ---------------------------------------------------------------------------
# ADWIN: adaptive windowing for concept drift detection
# ---------------------------------------------------------------------------

class ADWIN:
    """
    Adaptive Windowing algorithm (Bifet & Gavalda, 2007).
    Detects concept drift by monitoring mean shifts in a sliding window.
    """

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        self._window: deque[float] = deque()
        self._n = 0
        self._total = 0.0
        self.drift_detected = False

    def update(self, value: float) -> bool:
        """Add a value. Returns True if drift is detected."""
        self._window.append(value)
        self._n += 1
        self._total += value
        self.drift_detected = self._detect_change()
        return self.drift_detected

    def _detect_change(self) -> bool:
        n = len(self._window)
        if n < 2:
            return False
        arr = np.array(self._window)
        total = arr.sum()
        for i in range(1, n):
            n0 = i
            n1 = n - i
            mean0 = arr[:i].sum() / n0
            mean1 = arr[i:].sum() / n1
            mean_all = total / n
            delta_term = math.log(4.0 * math.log(n) / self.delta)
            eps_cut = math.sqrt(
                (1.0 / (2.0 * n0) + 1.0 / (2.0 * n1)) * delta_term
            )
            if abs(mean0 - mean1) >= eps_cut:
                # trim the older half
                trim = n // 2
                for _ in range(trim):
                    old = self._window.popleft()
                    self._total -= old
                    self._n -= 1
                return True
        return False

    @property
    def mean(self) -> float:
        if not self._window:
            return 0.0
        return self._total / len(self._window)

    def reset(self) -> None:
        self._window.clear()
        self._n = 0
        self._total = 0.0
        self.drift_detected = False


# ---------------------------------------------------------------------------
# 1. OnlineLogistic
# ---------------------------------------------------------------------------

class OnlineLogistic:
    """
    Online logistic regression with AdaGrad adaptive learning rates,
    L1 + L2 regularization, and optional feature hashing for high-dimensional
    sparse inputs.

    Parameters
    ----------
    n_features : int
        Number of input features (before hashing if use_hashing=True).
    learning_rate : float
        Base learning rate (eta0).
    l1 : float
        L1 regularization coefficient.
    l2 : float
        L2 regularization coefficient.
    use_hashing : bool
        If True, apply feature hashing to map inputs into hash_dim dimensions.
    hash_dim : int
        Target dimensionality when use_hashing=True.
    eps_adagrad : float
        AdaGrad epsilon for numerical stability.
    """

    def __init__(
        self,
        n_features: int = 20,
        learning_rate: float = 0.1,
        l1: float = 1e-4,
        l2: float = 1e-4,
        use_hashing: bool = False,
        hash_dim: int = 512,
        eps_adagrad: float = 1e-8,
    ) -> None:
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.use_hashing = use_hashing
        self.hash_dim = hash_dim
        self.eps_adagrad = eps_adagrad

        dim = hash_dim if use_hashing else n_features
        self._dim = dim
        self._w = np.zeros(dim)
        self._b = 0.0
        self._G = np.full(dim, eps_adagrad)  # accumulated squared gradients
        self._G_b = eps_adagrad
        self._n_updates = 0

    def _hash_features(self, x: np.ndarray) -> np.ndarray:
        """Feature hashing (Weinberger et al., 2009)."""
        result = np.zeros(self.hash_dim)
        for i, val in enumerate(x):
            h = int(hashlib.md5(struct.pack("i", i)).hexdigest(), 16)
            idx = h % self.hash_dim
            sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
            result[idx] += sign * val
        return result

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.use_hashing:
            x = self._hash_features(x)
        elif len(x) < self._dim:
            x = np.pad(x, (0, self._dim - len(x)))
        elif len(x) > self._dim:
            x = x[: self._dim]
        return x

    def predict_proba(self, x: np.ndarray) -> float:
        """Return P(y=1|x) in (0, 1)."""
        xp = self._prepare(x)
        return _sigmoid(_clip(float(self._w @ xp) + self._b))

    def predict(self, x: np.ndarray) -> float:
        """Return signal in [-1, 1] via tanh transform of log-odds."""
        xp = self._prepare(x)
        z = _clip(float(self._w @ xp) + self._b)
        return math.tanh(z * 0.5)

    def fit_one(self, x: np.ndarray, y: float) -> float:
        """
        Update model on single (x, y) pair.
        y should be 0.0 or 1.0 for classification.
        Returns the prediction error.
        """
        xp = self._prepare(x)
        z = float(self._w @ xp) + self._b
        p = _sigmoid(_clip(z))
        err = p - y

        # Gradient
        grad_w = err * xp + self.l2 * self._w
        grad_b = err

        # AdaGrad update
        self._G += grad_w ** 2
        self._G_b += grad_b ** 2

        eta_w = self.learning_rate / np.sqrt(self._G)
        eta_b = self.learning_rate / math.sqrt(self._G_b)

        self._w -= eta_w * grad_w
        self._b -= eta_b * grad_b

        # Proximal L1
        if self.l1 > 0.0:
            threshold = self.l1 * eta_w
            self._w = np.sign(self._w) * np.maximum(np.abs(self._w) - threshold, 0.0)

        self._n_updates += 1
        return abs(err)

    def feature_importance(self) -> np.ndarray:
        """Return absolute weight magnitudes as feature importance."""
        return np.abs(self._w).copy()

    def reset(self) -> None:
        self._w = np.zeros(self._dim)
        self._b = 0.0
        self._G = np.full(self._dim, self.eps_adagrad)
        self._G_b = self.eps_adagrad
        self._n_updates = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "OnlineLogistic",
            "n_features": self.n_features,
            "learning_rate": self.learning_rate,
            "l1": self.l1,
            "l2": self.l2,
            "use_hashing": self.use_hashing,
            "hash_dim": self.hash_dim,
            "eps_adagrad": self.eps_adagrad,
            "_dim": self._dim,
            "_w": self._w.tolist(),
            "_b": self._b,
            "_G": self._G.tolist(),
            "_G_b": self._G_b,
            "_n_updates": self._n_updates,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OnlineLogistic":
        obj = cls(
            n_features=d["n_features"],
            learning_rate=d["learning_rate"],
            l1=d["l1"],
            l2=d["l2"],
            use_hashing=d["use_hashing"],
            hash_dim=d["hash_dim"],
            eps_adagrad=d["eps_adagrad"],
        )
        obj._dim = d["_dim"]
        obj._w = np.array(d["_w"])
        obj._b = d["_b"]
        obj._G = np.array(d["_G"])
        obj._G_b = d["_G_b"]
        obj._n_updates = d["_n_updates"]
        return obj


# ---------------------------------------------------------------------------
# 2. OnlineRidge
# ---------------------------------------------------------------------------

class OnlineRidge:
    """
    Online ridge regression using the Woodbury matrix identity for O(d^2)
    rank-1 updates instead of O(d^3) matrix inversions.

    Used for continuous return forecasting. Maintains the inverse of
    (X^T X + lambda * I) incrementally.
    """

    def __init__(self, n_features: int = 20, lam: float = 1.0) -> None:
        self.n_features = n_features
        self.lam = lam
        self._w = np.zeros(n_features)
        # Initialize with ridge prior: precision = lam * I
        self._P = np.eye(n_features) / lam  # precision matrix inverse = (X'X + lam*I)^{-1}
        self._n_updates = 0

    def fit_one(self, x: np.ndarray, y: float) -> float:
        """
        Rank-1 update using Woodbury identity:
        P_new = P - (P x x^T P) / (1 + x^T P x)
        w_new = P_new * (X^T y) = P_new * (old_x'y + x*y)
        """
        x = np.asarray(x, dtype=float)
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))
        elif len(x) > self.n_features:
            x = x[: self.n_features]

        Px = self._P @ x
        denom = 1.0 + float(x @ Px)
        # Woodbury update
        self._P -= np.outer(Px, Px) / denom

        # Update weights: w += P_new * x * (y - x^T w)
        residual = y - float(self._w @ x)
        self._w += self._P @ x * residual

        self._n_updates += 1
        return abs(residual)

    def predict(self, x: np.ndarray) -> float:
        """Return continuous prediction (return forecast)."""
        x = np.asarray(x, dtype=float)
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))
        elif len(x) > self.n_features:
            x = x[: self.n_features]
        return float(self._w @ x)

    def predict_proba(self, x: np.ndarray) -> float:
        """Map continuous prediction to (0,1) via sigmoid."""
        return _sigmoid(_clip(self.predict(x)))

    def feature_importance(self) -> np.ndarray:
        return np.abs(self._w).copy()

    def reset(self) -> None:
        self._w = np.zeros(self.n_features)
        self._P = np.eye(self.n_features) / self.lam
        self._n_updates = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "OnlineRidge",
            "n_features": self.n_features,
            "lam": self.lam,
            "_w": self._w.tolist(),
            "_P": self._P.tolist(),
            "_n_updates": self._n_updates,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OnlineRidge":
        obj = cls(n_features=d["n_features"], lam=d["lam"])
        obj._w = np.array(d["_w"])
        obj._P = np.array(d["_P"])
        obj._n_updates = d["_n_updates"]
        return obj


# ---------------------------------------------------------------------------
# 3. OnlinePassiveAggressive
# ---------------------------------------------------------------------------

class OnlinePassiveAggressive:
    """
    Passive-Aggressive algorithm (Crammer et al., 2006).
    Supports PA-I and PA-II variants for classification and regression.

    PA-I:  tau = loss / ||x||^2
    PA-II: tau = loss / (||x||^2 + 1/(2*C))

    Handles non-stationary data via adaptive margin.
    """

    def __init__(
        self,
        n_features: int = 20,
        C: float = 1.0,
        variant: str = "PA-II",
        task: str = "classification",
        epsilon: float = 0.1,
    ) -> None:
        if variant not in ("PA-I", "PA-II"):
            raise ValueError("variant must be 'PA-I' or 'PA-II'")
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.n_features = n_features
        self.C = C
        self.variant = variant
        self.task = task
        self.epsilon = epsilon  # insensitive zone for regression
        self._w = np.zeros(n_features)
        self._b = 0.0
        self._n_updates = 0

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))
        elif len(x) > self.n_features:
            x = x[: self.n_features]
        return x

    def fit_one(self, x: np.ndarray, y: float) -> float:
        """
        Update on one (x, y) pair.
        For classification: y in {-1, +1}
        For regression: y is a continuous target
        """
        x = self._prepare(x)
        score = float(self._w @ x) + self._b

        if self.task == "classification":
            # Hinge loss
            loss = max(0.0, 1.0 - y * score)
            if loss == 0.0:
                return 0.0
            norm_sq = float(x @ x) + 1.0  # +1 for bias
            if self.variant == "PA-I":
                tau = min(self.C, loss / norm_sq)
            else:
                tau = loss / (norm_sq + 1.0 / (2.0 * self.C))
            self._w += tau * y * x
            self._b += tau * y
        else:
            # Epsilon-insensitive loss
            loss = max(0.0, abs(score - y) - self.epsilon)
            if loss == 0.0:
                return 0.0
            sign = 1.0 if score < y else -1.0
            norm_sq = float(x @ x) + 1.0
            if self.variant == "PA-I":
                tau = min(self.C, loss / norm_sq)
            else:
                tau = loss / (norm_sq + 1.0 / (2.0 * self.C))
            self._w += tau * sign * x
            self._b += tau * sign

        self._n_updates += 1
        return loss

    def predict(self, x: np.ndarray) -> float:
        x = self._prepare(x)
        raw = float(self._w @ x) + self._b
        if self.task == "classification":
            return math.tanh(raw * 0.5)
        return raw

    def predict_proba(self, x: np.ndarray) -> float:
        x = self._prepare(x)
        raw = float(self._w @ x) + self._b
        return _sigmoid(_clip(raw))

    def feature_importance(self) -> np.ndarray:
        return np.abs(self._w).copy()

    def reset(self) -> None:
        self._w = np.zeros(self.n_features)
        self._b = 0.0
        self._n_updates = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "OnlinePassiveAggressive",
            "n_features": self.n_features,
            "C": self.C,
            "variant": self.variant,
            "task": self.task,
            "epsilon": self.epsilon,
            "_w": self._w.tolist(),
            "_b": self._b,
            "_n_updates": self._n_updates,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OnlinePassiveAggressive":
        obj = cls(
            n_features=d["n_features"],
            C=d["C"],
            variant=d["variant"],
            task=d["task"],
            epsilon=d["epsilon"],
        )
        obj._w = np.array(d["_w"])
        obj._b = d["_b"]
        obj._n_updates = d["_n_updates"]
        return obj


# ---------------------------------------------------------------------------
# 4. FTRL
# ---------------------------------------------------------------------------

class FTRL:
    """
    Follow The Regularized Leader - Proximal (McMahan et al., 2013).
    Standard for sparse online learning (Google ads system).

    Per-feature adaptive learning rates. L1 encourages sparsity.
    L2 provides smoothing.

    Update rule:
        z_i += g_i - (sigma_i) * w_i
        w_i = -z_i / (lambda2 + sigma_sum_i) if |z_i| > lambda1 else 0
    where sigma_i = (sqrt(n_i) - sqrt(n_i - g_i^2)) / alpha
    """

    def __init__(
        self,
        n_features: int = 20,
        alpha: float = 0.1,
        beta: float = 1.0,
        l1: float = 0.1,
        l2: float = 1.0,
    ) -> None:
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2

        self._z = np.zeros(n_features)
        self._n = np.zeros(n_features)  # accumulated squared gradients
        self._n_updates = 0

    def _get_w(self) -> np.ndarray:
        """Compute current weight vector from z and n."""
        sigma_sum = (np.sqrt(self._n) + self.beta) / self.alpha
        mask = np.abs(self._z) > self.l1
        w = np.where(
            mask,
            -(self._z - np.sign(self._z) * self.l1) / (sigma_sum + self.l2),
            0.0,
        )
        return w

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))
        elif len(x) > self.n_features:
            x = x[: self.n_features]
        return x

    def predict_proba(self, x: np.ndarray) -> float:
        x = self._prepare(x)
        w = self._get_w()
        return _sigmoid(_clip(float(w @ x)))

    def predict(self, x: np.ndarray) -> float:
        x = self._prepare(x)
        w = self._get_w()
        return math.tanh(float(w @ x) * 0.5)

    def fit_one(self, x: np.ndarray, y: float) -> float:
        """
        Update on one (x, y) pair where y in {0, 1} (classification).
        """
        x = self._prepare(x)
        w = self._get_w()
        p = _sigmoid(_clip(float(w @ x)))
        g = (p - y) * x

        sigma = (np.sqrt(self._n + g ** 2) - np.sqrt(self._n)) / self.alpha
        self._z += g - sigma * w
        self._n += g ** 2

        self._n_updates += 1
        return abs(p - y)

    def feature_importance(self) -> np.ndarray:
        return np.abs(self._get_w())

    @property
    def sparsity(self) -> float:
        """Fraction of zero weights."""
        w = self._get_w()
        return float(np.sum(w == 0.0)) / len(w)

    def reset(self) -> None:
        self._z = np.zeros(self.n_features)
        self._n = np.zeros(self.n_features)
        self._n_updates = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "FTRL",
            "n_features": self.n_features,
            "alpha": self.alpha,
            "beta": self.beta,
            "l1": self.l1,
            "l2": self.l2,
            "_z": self._z.tolist(),
            "_n": self._n.tolist(),
            "_n_updates": self._n_updates,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FTRL":
        obj = cls(
            n_features=d["n_features"],
            alpha=d["alpha"],
            beta=d["beta"],
            l1=d["l1"],
            l2=d["l2"],
        )
        obj._z = np.array(d["_z"])
        obj._n = np.array(d["_n"])
        obj._n_updates = d["_n_updates"]
        return obj


# ---------------------------------------------------------------------------
# 5. OnlineGradientBoosting
# ---------------------------------------------------------------------------

@dataclass
class _StumpNode:
    """Depth-1 decision stump."""
    feature_idx: int = 0
    threshold: float = 0.0
    left_val: float = 0.0
    right_val: float = 0.0

    def predict(self, x: np.ndarray) -> float:
        return self.left_val if x[self.feature_idx] <= self.threshold else self.right_val


class _ShallowTree:
    """Decision tree of depth 1-3 fitted on residuals."""

    def __init__(self, max_depth: int = 2, n_features: int = 20) -> None:
        self.max_depth = max_depth
        self.n_features = n_features
        self._stumps: List[_StumpNode] = []
        self._fitted = False

    def _fit_stump(
        self, X: np.ndarray, residuals: np.ndarray
    ) -> _StumpNode:
        """Fit a single stump to minimize MSE on residuals."""
        n, d = X.shape
        best_loss = float("inf")
        best_stump = _StumpNode(0, 0.0, 0.0, 0.0)

        for fi in range(min(d, self.n_features)):
            vals = X[:, fi]
            thresholds = np.percentile(vals, [25, 50, 75])
            for thr in thresholds:
                left_mask = vals <= thr
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                lv = residuals[left_mask].mean()
                rv = residuals[right_mask].mean()
                loss = (
                    ((residuals[left_mask] - lv) ** 2).sum()
                    + ((residuals[right_mask] - rv) ** 2).sum()
                )
                if loss < best_loss:
                    best_loss = loss
                    best_stump = _StumpNode(fi, thr, lv, rv)
        return best_stump

    def fit(self, X: np.ndarray, residuals: np.ndarray) -> None:
        self._stumps = []
        current_residuals = residuals.copy()
        for _ in range(self.max_depth):
            stump = self._fit_stump(X, current_residuals)
            self._stumps.append(stump)
            preds = np.array([stump.predict(X[i]) for i in range(len(X))])
            current_residuals -= preds
        self._fitted = True

    def predict_one(self, x: np.ndarray) -> float:
        if not self._fitted:
            return 0.0
        return sum(s.predict(x) for s in self._stumps)


class OnlineGradientBoosting:
    """
    Online gradient boosting ensemble of shallow trees (depth 1-3).
    Each tree is fitted on residuals of the previous ensemble.
    Forgetting factor lambda controls how quickly old data is discarded,
    enabling adaptation to concept drift.

    Maintains a buffer of recent samples; refits entire ensemble
    every refitting_interval new samples.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 2,
        learning_rate: float = 0.1,
        forgetting_factor: float = 0.99,
        buffer_size: int = 200,
        refitting_interval: int = 20,
        n_features: int = 20,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.forgetting_factor = forgetting_factor
        self.buffer_size = buffer_size
        self.refitting_interval = refitting_interval
        self.n_features = n_features

        self._X_buf: deque = deque(maxlen=buffer_size)
        self._y_buf: deque = deque(maxlen=buffer_size)
        self._w_buf: deque = deque(maxlen=buffer_size)
        self._trees: List[_ShallowTree] = []
        self._n_since_refit = 0
        self._n_updates = 0

    def _build_trees(self) -> None:
        if len(self._X_buf) < 10:
            return
        X = np.array(list(self._X_buf))
        y = np.array(list(self._y_buf))
        w = np.array(list(self._w_buf))
        w = w / w.sum()

        self._trees = []
        residuals = y.copy()
        for _ in range(self.n_estimators):
            tree = _ShallowTree(max_depth=self.max_depth, n_features=self.n_features)
            # Weight residuals by sample weights
            tree.fit(X, residuals * w * len(w))
            preds = np.array([tree.predict_one(X[i]) for i in range(len(X))])
            residuals -= self.learning_rate * preds
            self._trees.append(tree)

    def fit_one(self, x: np.ndarray, y: float) -> float:
        x = np.asarray(x, dtype=float)
        if len(x) > self.n_features:
            x = x[: self.n_features]
        elif len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))

        # Forgetting: weight of new sample = 1, old samples multiplied by factor
        self._X_buf.append(x.copy())
        self._y_buf.append(float(y))

        # Compute weight for this sample
        n = len(self._X_buf)
        weights = [self.forgetting_factor ** (n - 1 - i) for i in range(n)]
        # Update buffer weights (lazy: recompute on rebuild)
        self._w_buf.append(1.0)

        self._n_since_refit += 1
        self._n_updates += 1

        if self._n_since_refit >= self.refitting_interval:
            # Recompute weights with forgetting factor
            buf_len = len(self._X_buf)
            new_weights = deque(
                [self.forgetting_factor ** (buf_len - 1 - i) for i in range(buf_len)],
                maxlen=self.buffer_size,
            )
            self._w_buf = new_weights
            self._build_trees()
            self._n_since_refit = 0

        pred = self.predict(x)
        return abs(pred - y)

    def predict(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if len(x) > self.n_features:
            x = x[: self.n_features]
        elif len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))

        if not self._trees:
            return 0.0
        raw = sum(
            self.learning_rate * t.predict_one(x) for t in self._trees
        )
        return float(np.tanh(raw))

    def predict_proba(self, x: np.ndarray) -> float:
        return _sigmoid(_clip(self.predict(x) * 2.0))

    def reset(self) -> None:
        self._X_buf.clear()
        self._y_buf.clear()
        self._w_buf.clear()
        self._trees = []
        self._n_since_refit = 0
        self._n_updates = 0

    def to_dict(self) -> Dict[str, Any]:
        # Trees are not serialized (refit on load from buffer)
        return {
            "type": "OnlineGradientBoosting",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "forgetting_factor": self.forgetting_factor,
            "buffer_size": self.buffer_size,
            "refitting_interval": self.refitting_interval,
            "n_features": self.n_features,
            "_X_buf": [x.tolist() for x in self._X_buf],
            "_y_buf": list(self._y_buf),
            "_w_buf": list(self._w_buf),
            "_n_updates": self._n_updates,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OnlineGradientBoosting":
        obj = cls(
            n_estimators=d["n_estimators"],
            max_depth=d["max_depth"],
            learning_rate=d["learning_rate"],
            forgetting_factor=d["forgetting_factor"],
            buffer_size=d["buffer_size"],
            refitting_interval=d["refitting_interval"],
            n_features=d["n_features"],
        )
        for xv in d["_X_buf"]:
            obj._X_buf.append(np.array(xv))
        for yv in d["_y_buf"]:
            obj._y_buf.append(yv)
        for wv in d["_w_buf"]:
            obj._w_buf.append(wv)
        obj._n_updates = d["_n_updates"]
        obj._build_trees()
        return obj


# ---------------------------------------------------------------------------
# 6. KernelOnlineLearning
# ---------------------------------------------------------------------------

class KernelOnlineLearning:
    """
    Nonlinear online learning via Random Kitchen Sinks (Rahimi & Recht, 2007).

    Maps inputs to a D-dimensional random Fourier feature space approximating
    the RBF kernel, then applies an online linear model in that space.
    This gives O(D) prediction and update cost.

    Parameters
    ----------
    n_features : int
        Dimensionality of input space.
    D : int
        Number of random Fourier features (budget). Default 256.
    gamma : float
        RBF kernel bandwidth parameter.
    inner_model : str
        Which linear model to use: 'logistic', 'ridge', or 'ftrl'.
    """

    def __init__(
        self,
        n_features: int = 20,
        D: int = 256,
        gamma: float = 1.0,
        inner_model: str = "logistic",
        learning_rate: float = 0.05,
        l1: float = 1e-4,
        l2: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.n_features = n_features
        self.D = D
        self.gamma = gamma
        self.inner_model_type = inner_model
        self.learning_rate = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.seed = seed

        rng = np.random.RandomState(seed)
        # Random Fourier features: omega ~ N(0, gamma * I)
        self._omega = rng.randn(D, n_features) * math.sqrt(2.0 * gamma)
        self._bias = rng.uniform(0.0, 2.0 * math.pi, D)

        if inner_model == "logistic":
            self._model: Any = OnlineLogistic(
                n_features=D, learning_rate=learning_rate, l1=l1, l2=l2
            )
        elif inner_model == "ridge":
            self._model = OnlineRidge(n_features=D, lam=l2)
        elif inner_model == "ftrl":
            self._model = FTRL(n_features=D, alpha=learning_rate, l1=l1, l2=l2)
        else:
            raise ValueError(f"Unknown inner_model: {inner_model}")

        self._n_updates = 0

    def _rff_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply random Fourier feature transform."""
        x = np.asarray(x, dtype=float)
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))
        elif len(x) > self.n_features:
            x = x[: self.n_features]
        z = math.sqrt(2.0 / self.D) * np.cos(self._omega @ x + self._bias)
        return z

    def fit_one(self, x: np.ndarray, y: float) -> float:
        z = self._rff_transform(x)
        self._n_updates += 1
        return self._model.fit_one(z, y)

    def predict(self, x: np.ndarray) -> float:
        z = self._rff_transform(x)
        return self._model.predict(z)

    def predict_proba(self, x: np.ndarray) -> float:
        z = self._rff_transform(x)
        return self._model.predict_proba(z)

    def reset(self) -> None:
        self._model.reset()
        self._n_updates = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "KernelOnlineLearning",
            "n_features": self.n_features,
            "D": self.D,
            "gamma": self.gamma,
            "inner_model_type": self.inner_model_type,
            "learning_rate": self.learning_rate,
            "l1": self.l1,
            "l2": self.l2,
            "seed": self.seed,
            "_omega": self._omega.tolist(),
            "_bias": self._bias.tolist(),
            "_model": self._model.to_dict(),
            "_n_updates": self._n_updates,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KernelOnlineLearning":
        obj = cls(
            n_features=d["n_features"],
            D=d["D"],
            gamma=d["gamma"],
            inner_model=d["inner_model_type"],
            learning_rate=d["learning_rate"],
            l1=d["l1"],
            l2=d["l2"],
            seed=d["seed"],
        )
        obj._omega = np.array(d["_omega"])
        obj._bias = np.array(d["_bias"])
        model_d = d["_model"]
        t = model_d.get("type", "")
        if t == "OnlineLogistic":
            obj._model = OnlineLogistic.from_dict(model_d)
        elif t == "OnlineRidge":
            obj._model = OnlineRidge.from_dict(model_d)
        elif t == "FTRL":
            obj._model = FTRL.from_dict(model_d)
        obj._n_updates = d["_n_updates"]
        return obj


# ---------------------------------------------------------------------------
# 7. ForgettingEnsemble
# ---------------------------------------------------------------------------

class ForgettingEnsemble:
    """
    Weighted ensemble of online models where weights decay with recent error.

    Maintains a softmax weight over N component models. After each prediction,
    the model's weight is updated based on its realized error using an
    exponential moving average. Models with higher recent errors get lower weight.

    Concept drift is detected per-model via ADWIN on prediction errors.
    On drift detection, the drifting model's weight is halved and it is reset.

    Parameters
    ----------
    models : list
        List of online model instances (each with fit_one, predict, predict_proba).
    learning_rate : float
        Weight update rate (eta in multiplicative weights update).
    adwin_delta : float
        ADWIN delta parameter for drift detection.
    min_weight : float
        Minimum weight floor per model.
    """

    def __init__(
        self,
        models: Optional[List[Any]] = None,
        learning_rate: float = 0.1,
        adwin_delta: float = 0.002,
        min_weight: float = 0.01,
    ) -> None:
        if models is None:
            models = [
                OnlineLogistic(n_features=20),
                FTRL(n_features=20),
                OnlinePassiveAggressive(n_features=20),
            ]
        self.models = models
        self.learning_rate = learning_rate
        self.adwin_delta = adwin_delta
        self.min_weight = min_weight

        n = len(models)
        self._weights = np.ones(n) / n
        self._error_ema = np.zeros(n)  # exponential moving avg of error per model
        self._adwins = [ADWIN(delta=adwin_delta) for _ in range(n)]
        self._n_updates = 0
        self._drift_counts = np.zeros(n, dtype=int)
        self._weight_history: List[np.ndarray] = []

    @property
    def n_models(self) -> int:
        return len(self.models)

    def predict(self, x: np.ndarray) -> float:
        """Weighted average of model signals."""
        preds = np.array([m.predict(x) for m in self.models])
        return float(self._weights @ preds)

    def predict_proba(self, x: np.ndarray) -> float:
        """Weighted average of model probabilities."""
        probas = np.array([m.predict_proba(x) for m in self.models])
        return float(self._weights @ probas)

    def fit_one(self, x: np.ndarray, y: float) -> float:
        """
        Get predictions from all models, then update all models and weights.
        y should be 0/1 for classification.
        Returns the ensemble's prediction error.
        """
        # 1. Collect pre-update predictions
        preds = np.array([m.predict_proba(x) for m in self.models])
        errors = np.abs(preds - y)

        # 2. Update individual models
        for m in self.models:
            m.fit_one(x, y)

        # 3. Multiplicative weights update (Hedge algorithm)
        losses = errors
        self._weights *= np.exp(-self.learning_rate * losses)

        # 4. Enforce minimum weight
        self._weights = np.maximum(self._weights, self.min_weight)
        self._weights /= self._weights.sum()

        # 5. ADWIN drift detection
        for i, (err, adwin) in enumerate(zip(errors, self._adwins)):
            if adwin.update(err):
                # Drift detected: halve weight and reset model
                self._weights[i] *= 0.5
                self.models[i].reset()
                self._drift_counts[i] += 1
                self._adwins[i] = ADWIN(delta=self.adwin_delta)

        # 6. Re-normalize
        self._weights /= self._weights.sum()

        # 7. Update error EMA
        alpha = 0.1
        self._error_ema = (1 - alpha) * self._error_ema + alpha * errors

        self._n_updates += 1
        self._weight_history.append(self._weights.copy())
        if len(self._weight_history) > 1000:
            self._weight_history = self._weight_history[-500:]

        ensemble_pred = float(self._weights @ preds)
        return abs(ensemble_pred - y)

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def get_drift_counts(self) -> np.ndarray:
        return self._drift_counts.copy()

    def reset(self) -> None:
        for m in self.models:
            m.reset()
        n = len(self.models)
        self._weights = np.ones(n) / n
        self._error_ema = np.zeros(n)
        self._adwins = [ADWIN(delta=self.adwin_delta) for _ in range(n)]
        self._n_updates = 0
        self._drift_counts = np.zeros(n, dtype=int)
        self._weight_history = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ForgettingEnsemble",
            "learning_rate": self.learning_rate,
            "adwin_delta": self.adwin_delta,
            "min_weight": self.min_weight,
            "_weights": self._weights.tolist(),
            "_error_ema": self._error_ema.tolist(),
            "_drift_counts": self._drift_counts.tolist(),
            "_n_updates": self._n_updates,
            "models": [m.to_dict() for m in self.models],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ForgettingEnsemble":
        model_list = []
        _dispatch = {
            "OnlineLogistic": OnlineLogistic,
            "OnlineRidge": OnlineRidge,
            "OnlinePassiveAggressive": OnlinePassiveAggressive,
            "FTRL": FTRL,
            "OnlineGradientBoosting": OnlineGradientBoosting,
            "KernelOnlineLearning": KernelOnlineLearning,
        }
        for md in d["models"]:
            t = md.get("type", "")
            if t in _dispatch:
                model_list.append(_dispatch[t].from_dict(md))
        obj = cls(
            models=model_list if model_list else None,
            learning_rate=d["learning_rate"],
            adwin_delta=d["adwin_delta"],
            min_weight=d["min_weight"],
        )
        obj._weights = np.array(d["_weights"])
        obj._error_ema = np.array(d["_error_ema"])
        obj._drift_counts = np.array(d["_drift_counts"], dtype=int)
        obj._n_updates = d["_n_updates"]
        return obj


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_default_ensemble(n_features: int = 20) -> ForgettingEnsemble:
    """
    Build a default ForgettingEnsemble with diverse component models.
    Suitable for live trading signal generation.
    """
    models = [
        OnlineLogistic(n_features=n_features, learning_rate=0.05, l1=1e-4, l2=1e-4),
        FTRL(n_features=n_features, alpha=0.1, beta=1.0, l1=0.05, l2=1.0),
        OnlinePassiveAggressive(n_features=n_features, C=1.0, variant="PA-II"),
        OnlineRidge(n_features=n_features, lam=1.0),
        KernelOnlineLearning(n_features=n_features, D=256, gamma=0.5, inner_model="logistic"),
    ]
    return ForgettingEnsemble(
        models=models, learning_rate=0.05, adwin_delta=0.002, min_weight=0.05
    )
