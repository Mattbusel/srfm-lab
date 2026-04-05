"""
models/xgboost_signal.py
========================
Gradient Boosted Decision Trees (GBDT) for tabular signal prediction.
Implemented from scratch using NumPy.

Financial rationale
-------------------
Tabular ML is the default strong baseline for financial signal research
because most alpha factors are naturally structured as cross-sectional
feature vectors (not sequences).  GBDT is the empirical champion on
tabular data: it handles missing values, captures non-linear interactions
between features (e.g. high momentum AND low volatility), and provides
built-in feature importance via gain, which is invaluable for alpha
decomposition.

Two prediction heads capture complementary views of alpha:
    direction  : P(next-bar return > 0) in [0, 1]  – via sigmoid
    magnitude  : expected |return| via softplus      – strictly positive

Final signal: direction * 2 - 1 (mapping [0,1] → [-1,+1])

Architecture
------------
Weak learner:  decision stump (depth-1 tree on one feature)
               selected by best gain = Σ(grad²/hess) improvement
Ensemble:      200 trees, learning_rate=0.1, row_subsample=0.8
               feature_subsample=0.8 per tree
Gradients:     logistic regression (direction head)
               MSE for magnitude head
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import MLSignal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_TREES     = 200
LR          = 0.1
ROW_SUB     = 0.8
FEAT_SUB    = 0.8
REG_LAMBDA  = 1.0     # L2 regularisation on leaf values


# ---------------------------------------------------------------------------
# Decision stump
# ---------------------------------------------------------------------------

@dataclass
class _Stump:
    """Depth-1 decision tree on a single feature."""
    feature_idx: int
    threshold:   float
    left_value:  float   # prediction for x[feature] <= threshold
    right_value: float   # prediction for x[feature] >  threshold
    gain:        float   # information gain achieved

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X : (N, F) → preds : (N,)"""
        mask = X[:, self.feature_idx] <= self.threshold
        out  = np.where(mask, self.left_value, self.right_value)
        return out.astype(np.float64)


def _fit_stump(
    X: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    feature_indices: np.ndarray,
    reg_lambda: float = REG_LAMBDA,
) -> _Stump:
    """Fit a single decision stump by maximising XGBoost gain.

    Gain formula (per split):
        G = (G_L²/(H_L + λ)) + (G_R²/(H_R + λ)) - (G²/(H + λ))
    where G = sum(gradients), H = sum(hessians) in each node.
    """
    best_gain  = -np.inf
    best_fi    = feature_indices[0]
    best_thr   = 0.0
    best_lv    = 0.0
    best_rv    = 0.0

    G_tot = grad.sum()
    H_tot = hess.sum()

    for fi in feature_indices:
        vals = X[:, fi]
        sorted_idx = np.argsort(vals)
        g_sorted   = grad[sorted_idx]
        h_sorted   = hess[sorted_idx]
        vals_sorted = vals[sorted_idx]

        G_L, H_L = 0.0, 0.0
        for k in range(1, len(sorted_idx)):
            G_L += g_sorted[k - 1]
            H_L += h_sorted[k - 1]
            G_R  = G_tot - G_L
            H_R  = H_tot - H_L
            if H_L < 1.0 or H_R < 1.0:
                continue
            gain = (G_L**2 / (H_L + reg_lambda)
                    + G_R**2 / (H_R + reg_lambda)
                    - G_tot**2 / (H_tot + reg_lambda))
            if gain > best_gain:
                best_gain = gain
                best_fi   = fi
                # Use midpoint of adjacent unique values as threshold
                best_thr  = float((vals_sorted[k - 1] + vals_sorted[k]) / 2)
                best_lv   = float(-G_L / (H_L + reg_lambda))
                best_rv   = float(-G_R / (H_R + reg_lambda))

    return _Stump(best_fi, best_thr, best_lv, best_rv, float(best_gain))


# ---------------------------------------------------------------------------
# GBDT head
# ---------------------------------------------------------------------------

class _GBDTHead:
    """One GBDT regression head.

    loss : 'mse' or 'logistic'
    """

    def __init__(
        self,
        n_trees: int,
        lr: float,
        row_sub: float,
        feat_sub: float,
        loss: str,
        rng: np.random.Generator,
    ) -> None:
        self.n_trees  = n_trees
        self.lr       = lr
        self.row_sub  = row_sub
        self.feat_sub = feat_sub
        self.loss     = loss
        self._rng     = rng
        self._stumps: List[_Stump] = []
        self._base    = 0.0
        self._gain_by_feature: Dict[int, float] = {}

    # ---- gradient / hessian ----------------------------------------

    def _grads_hess(self, y: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.loss == "mse":
            grad = F - y
            hess = np.ones_like(y)
        else:  # logistic
            p    = 1.0 / (1.0 + np.exp(-np.clip(F, -30, 30)))
            grad = p - y
            hess = p * (1.0 - p) + 1e-6
        return grad, hess

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        N, F = X.shape
        n_feat_sub  = max(1, int(F * self.feat_sub))
        n_row_sub   = max(1, int(N * self.row_sub))

        self._base = float(np.mean(y))
        preds = np.full(N, self._base)

        self._stumps.clear()
        self._gain_by_feature.clear()

        for _ in range(self.n_trees):
            row_idx  = self._rng.choice(N, n_row_sub, replace=False)
            feat_idx = self._rng.choice(F, n_feat_sub, replace=False)

            grad, hess = self._grads_hess(y[row_idx], preds[row_idx])
            stump = _fit_stump(X[row_idx], grad, hess, feat_idx)
            self._stumps.append(stump)

            # Accumulate gain by feature
            fi = stump.feature_idx
            self._gain_by_feature[fi] = self._gain_by_feature.get(fi, 0.0) + stump.gain

            preds += self.lr * stump.predict(X)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        out = np.full(len(X), self._base)
        for stump in self._stumps:
            out += self.lr * stump.predict(X)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.loss == "logistic":
            return 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
        else:
            # softplus for magnitude
            return np.log1p(np.exp(np.clip(raw, -20, 20)))

    def feature_gain(self) -> Dict[int, float]:
        return dict(self._gain_by_feature)

    def to_dict(self) -> List[dict]:
        return [
            {"fi": s.feature_idx, "thr": s.threshold,
             "lv": s.left_value, "rv": s.right_value, "gain": s.gain}
            for s in self._stumps
        ]

    def from_dict(self, data: List[dict]) -> None:
        self._stumps = [
            _Stump(d["fi"], d["thr"], d["lv"], d["rv"], d["gain"])
            for d in data
        ]


# ---------------------------------------------------------------------------
# Public XGBoost signal
# ---------------------------------------------------------------------------

FEATURE_COLS_DEFAULT = [
    "returns_1d", "returns_5d", "returns_10d", "returns_20d", "returns_60d",
    "log_return_1d", "vol_5d", "vol_20d", "vol_60d", "vol_of_vol",
    "rsi_14", "macd", "bb_pct_b", "atr_ratio", "vwap_deviation",
    "bh_mass", "bh_active", "bh_ctl", "ou_zscore",
    "btc_return_1d", "btc_eth_corr_20d",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "volume_ratio", "high_low_ratio", "close_position",
    "mayer_multiple", "ema_ratio",
]


class XGBoostSignal(MLSignal):
    """GBDT-based tabular signal with direction and magnitude heads.

    See module docstring for full description.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        n_trees: int = N_TREES,
        lr: float = LR,
        row_sub: float = ROW_SUB,
        feat_sub: float = FEAT_SUB,
        seed: int = 42,
    ) -> None:
        super().__init__(name="XGBoostSignal")
        self.feature_cols = feature_cols or FEATURE_COLS_DEFAULT
        self.n_trees  = n_trees
        self.lr       = lr
        self.row_sub  = row_sub
        self.feat_sub = feat_sub
        self._rng     = np.random.default_rng(seed)
        self._dir_head: Optional[_GBDTHead]  = None
        self._mag_head: Optional[_GBDTHead]  = None
        self._active_cols: List[str] = []

    # ------------------------------------------------------------------
    def _get_X(self, df: pd.DataFrame) -> np.ndarray:
        self._active_cols = [c for c in self.feature_cols if c in df.columns]
        X = df[self._active_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def fit(self, df: pd.DataFrame) -> "XGBoostSignal":
        """Train direction and magnitude heads on the feature DataFrame."""
        X = self._get_X(df)
        if "target" not in df.columns:
            raise ValueError("DataFrame must contain a 'target' column (forward return).")
        y = df["target"].values.astype(np.float64)

        # Direction target: 1 if return > 0, else 0
        y_dir = (y > 0).astype(np.float64)
        # Magnitude target: absolute return (we use softplus, so raw is fine)
        y_mag = np.abs(y)

        self._dir_head = _GBDTHead(
            self.n_trees, self.lr, self.row_sub, self.feat_sub, "logistic", self._rng)
        self._mag_head = _GBDTHead(
            self.n_trees, self.lr, self.row_sub, self.feat_sub, "mse", self._rng)

        self._dir_head.fit(X, y_dir)
        self._mag_head.fit(X, y_mag)

        self._is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> float:
        """Return signal in [-1, +1] for the most recent bar."""
        self._check_fitted()
        X = self._get_X(df)
        if len(X) == 0:
            return 0.0
        x = X[[-1]]
        p_up = float(self._dir_head.predict(x)[0])   # [0, 1]
        # Map: p_up=1 → +1, p_up=0 → -1
        score = 2.0 * p_up - 1.0
        return float(np.clip(score, -1.0, 1.0))

    def predict_proba(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (direction_prob, magnitude) for all rows."""
        self._check_fitted()
        X = self._get_X(df)
        return self._dir_head.predict(X), self._mag_head.predict(X)

    def save(self, path: pathlib.Path) -> None:
        self._check_fitted()
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        import json
        payload = {
            "feature_cols": self._active_cols,
            "dir_base":  self._dir_head._base,
            "mag_base":  self._mag_head._base,
            "dir_trees": self._dir_head.to_dict(),
            "mag_trees": self._mag_head.to_dict(),
        }
        with open(path / "xgboost_signal.json", "w") as f:
            json.dump(payload, f)

    def load(self, path: pathlib.Path) -> "XGBoostSignal":
        import json
        path = pathlib.Path(path)
        with open(path / "xgboost_signal.json") as f:
            payload = json.load(f)
        self._active_cols = payload["feature_cols"]

        self._dir_head = _GBDTHead(
            self.n_trees, self.lr, self.row_sub, self.feat_sub, "logistic", self._rng)
        self._dir_head._base = payload["dir_base"]
        self._dir_head.from_dict(payload["dir_trees"])

        self._mag_head = _GBDTHead(
            self.n_trees, self.lr, self.row_sub, self.feat_sub, "mse", self._rng)
        self._mag_head._base = payload["mag_base"]
        self._mag_head.from_dict(payload["mag_trees"])

        self._is_fitted = True
        return self

    def feature_importance(self) -> Dict[str, float]:
        """Gain-based feature importance, normalised to sum=1."""
        self._check_fitted()
        gain = self._dir_head.feature_gain()
        total = sum(gain.values()) + 1e-9
        result = {}
        for fi, g in gain.items():
            if fi < len(self._active_cols):
                result[self._active_cols[fi]] = g / total
        # Fill missing features with 0
        for col in self._active_cols:
            result.setdefault(col, 0.0)
        return result
