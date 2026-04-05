"""
models/ensemble.py
==================
EnsembleSignal: stacking meta-learner + dynamic IC-based weighting.

Financial rationale
-------------------
No single model dominates across all market regimes.  The LSTM tends to
do best in trending markets (it carries momentum implicitly in its cell
state), the Transformer excels at regime switches (global attention
captures structural breaks), and the XGBoost is most reliable in
range-bound markets (non-linear interactions between technical indicators
matter more than sequence structure).

Stacking (2-level ensemble):
    Level 0: LSTM, Transformer, XGBoost, BH signal → out-of-fold preds
    Level 1: Ridge regression on level-0 preds

Dynamic weighting:
    Each day compute rolling 30-day Spearman IC for each base model.
    Scale weight by max(0, IC / sum(max(IC_i, 0))).
    If IC_i < 0.02 for model i, zero its weight entirely.

Output:
    ensemble_score : float in [-1, +1]
    confidence     : float in [ 0,  1]  – based on cross-model agreement
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .base import MLSignal, SignalMetrics
from .lstm_signal import LSTMSignal
from .transformer_signal import TransformerSignal
from .xgboost_signal import XGBoostSignal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IC_MIN_THRESHOLD   = 0.02    # below this IC → zero weight
IC_WINDOW          = 30      # rolling window for IC estimation
RIDGE_ALPHA        = 1.0     # L2 regularisation for meta-learner
N_FOLDS_OOF        = 5       # out-of-fold folds for stacking


# ---------------------------------------------------------------------------
# Ridge regression (NumPy)
# ---------------------------------------------------------------------------

class _RidgeMeta:
    """Closed-form ridge regression: β = (XᵀX + αI)⁻¹ Xᵀy."""

    def __init__(self, alpha: float = RIDGE_ALPHA) -> None:
        self.alpha = alpha
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RidgeMeta":
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        N, F = Xc.shape
        A  = Xc.T @ Xc + self.alpha * np.eye(F)
        b  = Xc.T @ yc
        self.coef_      = np.linalg.solve(A, b)
        self.intercept_ = float(y.mean() - X.mean(axis=0) @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    def to_dict(self) -> dict:
        return {"coef": self.coef_.tolist(), "intercept": self.intercept_,
                "alpha": self.alpha}

    def from_dict(self, d: dict) -> None:
        self.coef_ = np.array(d["coef"])
        self.intercept_ = float(d["intercept"])
        self.alpha = float(d["alpha"])


# ---------------------------------------------------------------------------
# Ensemble signal
# ---------------------------------------------------------------------------

class EnsembleSignal(MLSignal):
    """Combines LSTM + Transformer + XGBoost + BH signal via stacking.

    Parameters
    ----------
    lstm        : fitted LSTMSignal
    transformer : fitted TransformerSignal
    xgboost     : fitted XGBoostSignal
    bh_col      : column name in the feature DataFrame that holds the
                  raw BH physics signal (default: ``'bh_signal'``)
    ridge_alpha : regularisation for the meta-learner
    ic_window   : rolling window (bars) for dynamic weight estimation
    ic_min      : minimum IC below which a model receives zero weight
    """

    def __init__(
        self,
        lstm: Optional[LSTMSignal] = None,
        transformer: Optional[TransformerSignal] = None,
        xgboost: Optional[XGBoostSignal] = None,
        bh_col: str = "bh_signal",
        ridge_alpha: float = RIDGE_ALPHA,
        ic_window: int = IC_WINDOW,
        ic_min: float = IC_MIN_THRESHOLD,
    ) -> None:
        super().__init__(name="EnsembleSignal")
        self.lstm        = lstm
        self.transformer = transformer
        self.xgboost     = xgboost
        self.bh_col      = bh_col
        self.ridge_alpha = ridge_alpha
        self.ic_window   = ic_window
        self.ic_min      = ic_min

        self._meta: Optional[_RidgeMeta] = None
        self._static_weights: np.ndarray = np.array([0.25, 0.25, 0.25, 0.25])
        self._rolling_ics: Dict[str, List[float]] = {
            "lstm": [], "transformer": [], "xgboost": [], "bh": []}
        self._model_names = ["lstm", "transformer", "xgboost", "bh"]

    # ------------------------------------------------------------------
    # Out-of-fold prediction builder
    # ------------------------------------------------------------------

    def _collect_base_preds(
        self, df: pd.DataFrame, n_folds: int = N_FOLDS_OOF
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate out-of-fold stacking features (N, 4) and targets (N,)."""
        N = len(df)
        fold_size = N // n_folds
        oof_preds = np.full((N, 4), np.nan)
        targets   = df["target"].values.astype(np.float64) if "target" in df.columns else np.zeros(N)

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end   = val_start + fold_size if fold < n_folds - 1 else N
            train_df  = pd.concat([df.iloc[:val_start], df.iloc[val_end:]])
            val_df    = df.iloc[val_start:val_end]

            if len(train_df) < 50:
                continue

            # Re-fit base models on train slice
            if self.lstm is not None:
                self.lstm.fit(train_df)
            if self.transformer is not None:
                self.transformer.fit(train_df)
            if self.xgboost is not None:
                self.xgboost.fit(train_df)

            # Predict on each val row using expanding context
            for i, idx in enumerate(val_df.index):
                pos = df.index.get_loc(idx)
                context = df.iloc[max(0, pos - 100) : pos + 1]

                lstm_pred = self.lstm.predict(context) if self.lstm and self.lstm.is_fitted else 0.0
                trans_pred = self.transformer.predict(context) if self.transformer and self.transformer.is_fitted else 0.0
                xgb_pred  = self.xgboost.predict(context) if self.xgboost and self.xgboost.is_fitted else 0.0
                bh_pred   = float(context[self.bh_col].iloc[-1]) if self.bh_col in context.columns else 0.0

                oof_preds[val_start + i] = [lstm_pred, trans_pred, xgb_pred, bh_pred]

        # Fill any remaining NaN rows with 0
        oof_preds = np.nan_to_num(oof_preds, nan=0.0)
        return oof_preds, targets

    # ------------------------------------------------------------------
    # MLSignal interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "EnsembleSignal":
        """Fit the meta-learner using out-of-fold predictions from base models."""
        oof_preds, targets = self._collect_base_preds(df)

        # Retrain all base models on full data before final meta fit
        if self.lstm is not None:
            self.lstm.fit(df)
        if self.transformer is not None:
            self.transformer.fit(df)
        if self.xgboost is not None:
            self.xgboost.fit(df)

        self._meta = _RidgeMeta(alpha=self.ridge_alpha)
        self._meta.fit(oof_preds, targets)

        # Initialise static weights from meta coefficients
        raw = np.abs(self._meta.coef_)
        self._static_weights = raw / (raw.sum() + 1e-9)

        self._is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> float:
        """Return ensemble_score in [-1, +1]."""
        self._check_fitted()
        preds = self._get_base_predictions(df)
        weights = self._dynamic_weights()
        raw = float(np.dot(weights, preds))
        return float(np.clip(np.tanh(raw * 2.0), -1.0, 1.0))

    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Return (ensemble_score, confidence).

        Confidence is the inter-model agreement: 1 - std(preds) / max_possible_std.
        """
        self._check_fitted()
        preds   = self._get_base_predictions(df)
        weights = self._dynamic_weights()
        score   = float(np.clip(np.tanh(np.dot(weights, preds) * 2.0), -1.0, 1.0))
        # Agreement: all preds in same direction → high confidence
        active_preds = preds[weights > 0]
        if len(active_preds) < 2:
            confidence = 0.5
        else:
            std  = float(np.std(active_preds))
            confidence = float(np.clip(1.0 - std, 0.0, 1.0))
        return score, confidence

    # ------------------------------------------------------------------
    # Dynamic IC weighting
    # ------------------------------------------------------------------

    def update_rolling_ic(self, preds_dict: Dict[str, np.ndarray], returns: np.ndarray) -> None:
        """Update rolling IC history for each base model.

        Parameters
        ----------
        preds_dict : dict mapping model name to array of predictions
        returns    : realised forward returns aligned with predictions
        """
        for name in self._model_names:
            if name not in preds_dict:
                continue
            p = preds_dict[name]
            if len(p) < 5 or len(returns) < 5:
                continue
            ic, _ = spearmanr(p[-self.ic_window:], returns[-self.ic_window:])
            if not np.isnan(ic):
                self._rolling_ics[name].append(float(ic))
                # Keep only the most recent window
                if len(self._rolling_ics[name]) > self.ic_window * 3:
                    self._rolling_ics[name] = self._rolling_ics[name][-self.ic_window * 3:]

    def _dynamic_weights(self) -> np.ndarray:
        """Compute dynamic weights based on rolling IC."""
        recent_ics = np.array([
            float(np.mean(self._rolling_ics[n][-self.ic_window:]))
            if len(self._rolling_ics[n]) >= 5 else self._static_weights[i]
            for i, n in enumerate(self._model_names)
        ])
        # Zero out models below IC threshold
        recent_ics = np.where(recent_ics < self.ic_min, 0.0, recent_ics)
        total = recent_ics.sum()
        if total < 1e-9:
            return self._static_weights.copy()
        return recent_ics / total

    def _get_base_predictions(self, df: pd.DataFrame) -> np.ndarray:
        lstm_p  = self.lstm.predict(df) if self.lstm and self.lstm.is_fitted else 0.0
        trans_p = self.transformer.predict(df) if self.transformer and self.transformer.is_fitted else 0.0
        xgb_p   = self.xgboost.predict(df) if self.xgboost and self.xgboost.is_fitted else 0.0
        bh_p    = float(df[self.bh_col].iloc[-1]) if self.bh_col in df.columns else 0.0
        return np.array([lstm_p, trans_p, xgb_p, bh_p])

    # ------------------------------------------------------------------
    def save(self, path: pathlib.Path) -> None:
        self._check_fitted()
        import json
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self._meta.to_dict(),
            "static_weights": self._static_weights.tolist(),
            "rolling_ics": self._rolling_ics,
            "config": {
                "ridge_alpha": self.ridge_alpha,
                "ic_window": self.ic_window,
                "ic_min": self.ic_min,
                "bh_col": self.bh_col,
            },
        }
        with open(path / "ensemble.json", "w") as f:
            json.dump(payload, f)
        if self.lstm:
            self.lstm.save(path / "lstm")
        if self.transformer:
            self.transformer.save(path / "transformer")
        if self.xgboost:
            self.xgboost.save(path / "xgboost")

    def load(self, path: pathlib.Path) -> "EnsembleSignal":
        import json
        path = pathlib.Path(path)
        with open(path / "ensemble.json") as f:
            payload = json.load(f)
        self._meta = _RidgeMeta()
        self._meta.from_dict(payload["meta"])
        self._static_weights = np.array(payload["static_weights"])
        self._rolling_ics    = payload["rolling_ics"]
        cfg = payload["config"]
        self.ridge_alpha = cfg["ridge_alpha"]
        self.ic_window   = cfg["ic_window"]
        self.ic_min      = cfg["ic_min"]
        self.bh_col      = cfg["bh_col"]
        if self.lstm and (path / "lstm").exists():
            self.lstm.load(path / "lstm")
        if self.transformer and (path / "transformer").exists():
            self.transformer.load(path / "transformer")
        if self.xgboost and (path / "xgboost").exists():
            self.xgboost.load(path / "xgboost")
        self._is_fitted = True
        return self

    def feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance across base models, weighted by static weights."""
        self._check_fitted()
        combined: Dict[str, float] = {}
        sources = [
            (self.lstm, self._static_weights[0]),
            (self.transformer, self._static_weights[1]),
            (self.xgboost, self._static_weights[2]),
        ]
        for model, w in sources:
            if model and model.is_fitted:
                for feat, imp in model.feature_importance().items():
                    combined[feat] = combined.get(feat, 0.0) + w * imp
        total = sum(combined.values()) + 1e-9
        return {k: v / total for k, v in sorted(combined.items(), key=lambda x: -x[1])}
