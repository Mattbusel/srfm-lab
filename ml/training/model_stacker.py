"""
ml/training/model_stacker.py

Model stacking, blending, and calibration for SRFM ensemble signals.

Stacking uses purged CV to generate out-of-fold (OOF) predictions so the
meta-learner never sees in-sample fitted values.  This prevents look-ahead
leakage in the meta-learning step.

Classes
-------
ModelStacker      -- full stacking pipeline with a meta-learner
ModelBlender      -- weighted average blend using validation IC
EnsembleCalibrator -- Platt scaling for probability calibration
StackingReport    -- structured summary of the stacking run
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit  -- logistic sigmoid

from .cross_validator import (
    PurgedKFoldCV,
    compute_information_coefficient,
    CrossValidationReport,
)


# ---------------------------------------------------------------------------
# StackingReport
# ---------------------------------------------------------------------------

@dataclass
class StackingReport:
    """
    Structured summary of a stacking run.

    Attributes
    ----------
    oof_scores : dict mapping model name -> per-fold IC list
    meta_coefs : dict of meta-learner coefficients (if linear meta)
    feature_importances : dict mapping model name -> importance array
    blend_weights : dict mapping model name -> blend weight
    oof_ic_mean : float  -- mean IC across all models and folds
    """

    oof_scores: Dict[str, List[float]] = field(default_factory=dict)
    meta_coefs: Dict[str, float] = field(default_factory=dict)
    feature_importances: Dict[str, np.ndarray] = field(default_factory=dict)
    blend_weights: Dict[str, float] = field(default_factory=dict)
    oof_ic_mean: float = float("nan")

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, scores in self.oof_scores.items():
            rows.append(
                {
                    "model": name,
                    "oof_ic_mean": float(np.nanmean(scores)),
                    "oof_ic_std": float(np.nanstd(scores)),
                    "n_folds": len(scores),
                    "blend_weight": self.blend_weights.get(name, float("nan")),
                }
            )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        n = len(self.oof_scores)
        return f"StackingReport(n_models={n}, oof_ic_mean={self.oof_ic_mean:.4f})"


# ---------------------------------------------------------------------------
# ModelStacker
# ---------------------------------------------------------------------------

class ModelStacker:
    """
    Stacks multiple base models using a meta-learner.

    Uses purged CV to generate out-of-fold predictions so the meta-learner
    is trained only on OOF predictions -- never on fitted training data.

    Parameters
    ----------
    base_models : list of (name, estimator) tuples
        Each estimator must implement fit(X, y) and predict(X).
    meta_learner : estimator
        Must implement fit(X, y) and predict(X).  A Ridge regression or
        small linear model is recommended.
    cv : PurgedKFoldCV
        Cross-validator used to generate OOF predictions.
    passthrough : bool
        If True, include original X features as additional meta-features.
    event_times : array or None
        Optional event end times passed to the purged CV.
    """

    def __init__(
        self,
        base_models: List[Tuple[str, Any]],
        meta_learner: Any,
        cv: PurgedKFoldCV,
        passthrough: bool = False,
        event_times: Optional[np.ndarray] = None,
    ):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.cv = cv
        self.passthrough = passthrough
        self.event_times = event_times

        -- fitted state (set after fit())
        self._fitted_base: List[Tuple[str, Any]] = []
        self._fitted_meta: Any = None
        self._report: Optional[StackingReport] = None
        self._n_features: Optional[int] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModelStacker":
        """
        1. Generate OOF predictions from each base model via purged CV.
        2. Stack OOF predictions as meta-features.
        3. Fit the meta-learner on stacked OOF features.
        4. Refit each base model on the full training set.

        Returns self.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape
        self._n_features = p

        n_base = len(self.base_models)
        oof_matrix = np.full((n, n_base), np.nan)
        oof_scores: Dict[str, List[float]] = {name: [] for name, _ in self.base_models}

        -- step 1: generate OOF predictions for each base model
        for col_idx, (name, model) in enumerate(self.base_models):
            for fold_idx, (train_idx, test_idx) in enumerate(
                self.cv.split(X, y, event_times=self.event_times)
            ):
                if len(train_idx) == 0 or len(test_idx) == 0:
                    warnings.warn(
                        f"Stacker fold {fold_idx} for model '{name}' is empty, skipping."
                    )
                    continue

                -- clone-like behavior: use a fresh copy per fold via fit
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
                oof_matrix[test_idx, col_idx] = pred

                -- compute IC for this fold
                ic = compute_information_coefficient(pred, y[test_idx])
                oof_scores[name].append(ic)

        -- step 2: build meta-feature matrix
        -- rows with any NaN OOF prediction are dropped from meta-training
        meta_X = self._build_meta_features(oof_matrix, X)
        valid_mask = ~np.isnan(meta_X).any(axis=1)

        if valid_mask.sum() < 10:
            raise RuntimeError(
                "Too few valid OOF samples to train meta-learner. "
                "Increase training set size or reduce n_splits."
            )

        -- step 3: fit meta-learner
        self._fitted_meta = self.meta_learner
        self._fitted_meta.fit(meta_X[valid_mask], y[valid_mask])

        -- step 4: refit each base model on the full training set
        self._fitted_base = []
        for name, model in self.base_models:
            model.fit(X, y)
            self._fitted_base.append((name, model))

        -- build report
        all_ics = [ic for scores in oof_scores.values() for ic in scores if not np.isnan(ic)]
        self._report = StackingReport(
            oof_scores=oof_scores,
            meta_coefs=self._extract_meta_coefs(),
            feature_importances=self._extract_feature_importances(),
            blend_weights={},
            oof_ic_mean=float(np.mean(all_ics)) if all_ics else float("nan"),
        )

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions by passing base model outputs through meta-learner.

        Each base model predicts; the stacked predictions (+ optional
        passthrough features) are fed to the meta-learner.
        """
        if self._fitted_meta is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X)
        n = len(X)
        n_base = len(self._fitted_base)
        base_preds = np.zeros((n, n_base))

        for col_idx, (name, model) in enumerate(self._fitted_base):
            base_preds[:, col_idx] = model.predict(X)

        meta_X = self._build_meta_features(base_preds, X)
        return self._fitted_meta.predict(meta_X)

    # ------------------------------------------------------------------
    # Report access
    # ------------------------------------------------------------------

    @property
    def report(self) -> Optional[StackingReport]:
        return self._report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_meta_features(
        self, oof_matrix: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        if self.passthrough:
            return np.hstack([oof_matrix, X])
        return oof_matrix

    def _extract_meta_coefs(self) -> Dict[str, float]:
        meta = self._fitted_meta
        coefs: Dict[str, float] = {}
        if hasattr(meta, "coef_"):
            for idx, (name, _) in enumerate(self.base_models):
                if idx < len(np.atleast_1d(meta.coef_)):
                    coefs[name] = float(np.atleast_1d(meta.coef_)[idx])
        return coefs

    def _extract_feature_importances(self) -> Dict[str, np.ndarray]:
        imps: Dict[str, np.ndarray] = {}
        for name, model in self._fitted_base:
            if hasattr(model, "feature_importances_"):
                imps[name] = np.array(model.feature_importances_)
            elif hasattr(model, "coef_"):
                imps[name] = np.abs(np.atleast_1d(model.coef_))
        return imps

    def __repr__(self) -> str:
        names = [n for n, _ in self.base_models]
        return (
            f"ModelStacker(base_models={names}, "
            f"meta_learner={type(self.meta_learner).__name__}, "
            f"cv={self.cv})"
        )


# ---------------------------------------------------------------------------
# ModelBlender
# ---------------------------------------------------------------------------

class ModelBlender:
    """
    Simple weighted average ensemble.

    Weights are determined by validation IC: models with higher IC
    receive proportionally higher weight.  Negative-IC models are
    optionally excluded.

    Parameters
    ----------
    base_models : list of (name, estimator) tuples
    cv : PurgedKFoldCV
    weight_floor : float
        Minimum weight (after normalisation) before a model is excluded.
    event_times : array or None
    """

    def __init__(
        self,
        base_models: List[Tuple[str, Any]],
        cv: PurgedKFoldCV,
        weight_floor: float = 0.0,
        event_times: Optional[np.ndarray] = None,
    ):
        self.base_models = base_models
        self.cv = cv
        self.weight_floor = weight_floor
        self.event_times = event_times

        self._weights: Optional[np.ndarray] = None
        self._fitted_models: List[Tuple[str, Any]] = []
        self._oof_ics: Dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModelBlender":
        X = np.asarray(X)
        y = np.asarray(y)

        oof_ics: Dict[str, float] = {}

        for name, model in self.base_models:
            fold_ics = []
            for train_idx, test_idx in self.cv.split(
                X, y, event_times=self.event_times
            ):
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
                ic = compute_information_coefficient(pred, y[test_idx])
                fold_ics.append(ic)
            oof_ics[name] = float(np.nanmean(fold_ics)) if fold_ics else 0.0

        -- build weights from IC; clip negative ICs to zero
        raw_weights = np.array(
            [max(0.0, oof_ics[name]) for name, _ in self.base_models]
        )
        total = raw_weights.sum()
        if total < 1e-10:
            -- uniform weights as fallback
            raw_weights = np.ones(len(self.base_models))
            total = float(len(self.base_models))

        normalised = raw_weights / total

        -- apply weight floor: zero out models below threshold
        normalised[normalised < self.weight_floor] = 0.0
        floor_total = normalised.sum()
        if floor_total > 0:
            normalised /= floor_total

        self._weights = normalised
        self._oof_ics = oof_ics

        -- refit all models on full data
        self._fitted_models = []
        for name, model in self.base_models:
            model.fit(X, y)
            self._fitted_models.append((name, model))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X)
        result = np.zeros(len(X))
        for (name, model), w in zip(self._fitted_models, self._weights):
            if w > 0:
                result += w * model.predict(X)
        return result

    @property
    def weights(self) -> Dict[str, float]:
        if self._weights is None:
            return {}
        return {name: float(self._weights[i]) for i, (name, _) in enumerate(self.base_models)}

    @property
    def oof_ics(self) -> Dict[str, float]:
        return self._oof_ics

    def __repr__(self) -> str:
        return (
            f"ModelBlender(n_models={len(self.base_models)}, "
            f"weights={self.weights})"
        )


# ---------------------------------------------------------------------------
# EnsembleCalibrator  -- Platt scaling
# ---------------------------------------------------------------------------

class EnsembleCalibrator:
    """
    Platt scaling to calibrate raw probability predictions.

    Fits a logistic function  P(y=1 | f) = 1 / (1 + exp(A*f + B))
    where f is the raw model score and A, B are fitted via MLE.

    Parameters
    ----------
    eps : float
        Label smoothing epsilon to avoid log(0) in the loss.
    """

    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self._A: float = 1.0
        self._B: float = 0.0
        self._fitted: bool = False

    def fit(
        self, y_pred_proba: np.ndarray, y_true: np.ndarray
    ) -> "EnsembleCalibrator":
        """
        Fit sigmoid calibration curve using Platt's method (BFGS optimisation).

        Parameters
        ----------
        y_pred_proba : array of shape (n_samples,)
            Raw model scores or uncalibrated probabilities.
        y_true : array of shape (n_samples,)
            Binary labels {0, 1}.
        """
        y_pred_proba = np.asarray(y_pred_proba, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        n = len(y_true)
        n_pos = y_true.sum()
        n_neg = n - n_pos

        -- Platt's target labels (smoothed)
        t = np.where(
            y_true > 0.5,
            (n_pos + 1.0) / (n_pos + 2.0),
            1.0 / (n_neg + 2.0),
        )

        def neg_log_likelihood(params: np.ndarray) -> float:
            A, B = params
            fA = y_pred_proba * A + B
            p = expit(-fA)
            p = np.clip(p, self.eps, 1.0 - self.eps)
            return -np.sum(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))

        result = optimize.minimize(
            neg_log_likelihood,
            x0=np.array([1.0, 0.0]),
            method="BFGS",
            options={"maxiter": 200, "gtol": 1e-5},
        )

        if not result.success:
            warnings.warn(
                f"Platt scaling optimisation did not converge: {result.message}"
            )

        self._A, self._B = float(result.x[0]), float(result.x[1])
        self._fitted = True
        return self

    def calibrate(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Apply the calibration transform.

        Parameters
        ----------
        y_pred_proba : array of shape (n_samples,)

        Returns
        -------
        np.ndarray of calibrated probabilities in (0, 1).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before calibrate().")
        y_pred_proba = np.asarray(y_pred_proba, dtype=float)
        fA = y_pred_proba * self._A + self._B
        return expit(-fA)

    @property
    def params(self) -> Tuple[float, float]:
        """Return (A, B) Platt scaling parameters."""
        return self._A, self._B

    def reliability_diagram(
        self, y_pred_proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Compute reliability diagram data: binned mean predicted prob vs
        binned actual frequency.
        """
        calibrated = self.calibrate(y_pred_proba)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (calibrated >= lo) & (calibrated < hi)
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "bin_center": float((lo + hi) / 2.0),
                    "mean_pred": float(calibrated[mask].mean()),
                    "actual_freq": float(y_true[mask].mean()),
                    "n_samples": int(mask.sum()),
                }
            )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        if self._fitted:
            return f"EnsembleCalibrator(A={self._A:.4f}, B={self._B:.4f})"
        return "EnsembleCalibrator(unfitted)"


# ---------------------------------------------------------------------------
# Convenience: OOF prediction generator
# ---------------------------------------------------------------------------

def generate_oof_predictions(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: PurgedKFoldCV,
    event_times: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate out-of-fold predictions from a single model using purged CV.

    Parameters
    ----------
    model : sklearn-compatible estimator
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    cv : PurgedKFoldCV
    event_times : optional event end bars for purging

    Returns
    -------
    np.ndarray of OOF predictions (NaN where not predicted).
    """
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in cv.split(X, y, event_times=event_times):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict(X[test_idx])
    return oof
