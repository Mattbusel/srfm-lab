"""
ensemble_forecaster.py
Ensemble return/price forecaster built from scratch using numpy and scipy.
Models: Linear (Ridge), Kernel Ridge (RBF), Decision Tree (CART), SVM (QP).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Base Model Interface
# ---------------------------------------------------------------------------

class BaseModel:
    """Abstract base for all ensemble members."""

    name: str = "base"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Linear Ridge Regression
# ---------------------------------------------------------------------------

class LinearModel(BaseModel):
    """Ridge regression: minimizes ||y - Xw||^2 + alpha * ||w||^2."""

    name = "ridge_linear"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearModel":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n, p = X.shape
        # w = (X^T X + alpha * I)^{-1} X^T y
        A = X.T @ X + self.alpha * np.eye(p)
        b = X.T @ y
        self._weights = np.linalg.solve(A, b)
        self._bias = float(np.mean(y) - X.mean(axis=0) @ self._weights)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._weights is None:
            raise RuntimeError("Model not fitted.")
        return X @ self._weights + self._bias


# ---------------------------------------------------------------------------
# 2. Kernel Ridge Regression (RBF)
# ---------------------------------------------------------------------------

class KernelModel(BaseModel):
    """Kernel ridge regression with RBF kernel: K(x,z) = exp(-gamma * ||x-z||^2)."""

    name = "kernel_ridge_rbf"

    def __init__(self, alpha: float = 1.0, gamma: float = 0.1) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self._X_train: Optional[np.ndarray] = None
        self._dual: Optional[np.ndarray] = None

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix between X and Y."""
        sq_dists = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-self.gamma * sq_dists)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelModel":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self._X_train = X
        n = X.shape[0]
        K = self._rbf_kernel(X, X)
        # dual coefficients: alpha_vec = (K + alpha * I)^{-1} y
        A = K + self.alpha * np.eye(n)
        self._dual = np.linalg.solve(A, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._X_train is None or self._dual is None:
            raise RuntimeError("Model not fitted.")
        K_test = self._rbf_kernel(X, self._X_train)
        return K_test @ self._dual


# ---------------------------------------------------------------------------
# 3. Decision Tree (CART-like recursive splits)
# ---------------------------------------------------------------------------

@dataclass
class _TreeNode:
    is_leaf: bool
    value: float = 0.0
    feature_idx: int = 0
    threshold: float = 0.0
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None


class TreeModel(BaseModel):
    """
    CART-style regression tree using variance reduction (MSE) splitting criterion.
    Built via recursive partitioning with max_depth and min_samples_leaf constraints.
    """

    name = "cart_tree"

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 5,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._root: Optional[_TreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TreeModel":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self._root = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        n, p = X.shape
        # Leaf conditions
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return _TreeNode(is_leaf=True, value=float(np.mean(y)))

        best_gain = 0.0
        best_feat, best_thresh = 0, 0.0
        parent_var = float(np.var(y)) * n

        for feat in range(p):
            col = X[:, feat]
            thresholds = np.percentile(col, np.linspace(10, 90, 15))
            for t in np.unique(thresholds):
                left_mask = col <= t
                right_mask = ~left_mask
                nl, nr = int(np.sum(left_mask)), int(np.sum(right_mask))
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                yl, yr = y[left_mask], y[right_mask]
                gain = parent_var - (float(np.var(yl)) * nl + float(np.var(yr)) * nr)
                if gain > best_gain:
                    best_gain = gain
                    best_feat, best_thresh = feat, float(t)

        if best_gain <= 0:
            return _TreeNode(is_leaf=True, value=float(np.mean(y)))

        left_mask = X[:, best_feat] <= best_thresh
        node = _TreeNode(
            is_leaf=False,
            feature_idx=best_feat,
            threshold=best_thresh,
        )
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _predict_row(self, row: np.ndarray, node: _TreeNode) -> float:
        if node.is_leaf:
            return node.value
        if row[node.feature_idx] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._root is None:
            raise RuntimeError("Model not fitted.")
        return np.array([self._predict_row(row, self._root) for row in X])


# ---------------------------------------------------------------------------
# 4. Soft-Margin SVM via Quadratic Programming (simplified)
# ---------------------------------------------------------------------------

class SVMModel(BaseModel):
    """
    Support Vector Regression (epsilon-SVR) via scipy minimize.
    Minimizes: 0.5 * w^T w + C * sum(max(0, |y - Xw - b| - epsilon))
    Gradient descent via L-BFGS-B for efficiency.
    """

    name = "svm_svr"

    def __init__(self, C: float = 1.0, epsilon: float = 0.01) -> None:
        self.C = C
        self.epsilon = epsilon
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self._p: int = 0

    def _objective(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        w, b = params[:-1], params[-1]
        residuals = y - (X @ w + b)
        hinge = np.maximum(0.0, np.abs(residuals) - self.epsilon)
        return 0.5 * float(w @ w) + self.C * float(np.sum(hinge))

    def _gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        w, b = params[:-1], params[-1]
        residuals = y - (X @ w + b)
        abs_r = np.abs(residuals)
        support = abs_r > self.epsilon
        signs = np.sign(residuals)  # sign of (y - pred)

        grad_w = w.copy()
        grad_b = 0.0
        if np.any(support):
            # d/dw [C * sum(|r| - eps)] = -C * sign(r) * x for support vectors
            grad_w -= self.C * (X[support].T @ signs[support])
            grad_b -= self.C * float(np.sum(signs[support]))

        return np.append(grad_w, grad_b)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMModel":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n, p = X.shape
        self._p = p
        params0 = np.zeros(p + 1)

        result = minimize(
            fun=self._objective,
            x0=params0,
            jac=self._gradient,
            args=(X, y),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9},
        )
        self._weights = result.x[:-1]
        self._bias = float(result.x[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._weights is None:
            raise RuntimeError("Model not fitted.")
        return X @ self._weights + self._bias


# ---------------------------------------------------------------------------
# Walk-Forward Validation Metrics
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardMetrics:
    ic_mean: float
    ic_std: float
    mse_mean: float
    directional_accuracy: float
    sharpe_of_predictions: float
    n_windows: int


# ---------------------------------------------------------------------------
# Ensemble Forecaster
# ---------------------------------------------------------------------------

class EnsembleForecast:
    """
    Ensemble of LinearModel, KernelModel, TreeModel, SVMModel.
    Supports stacking (meta-learner) or simple average / exponential reweighting.
    """

    def __init__(
        self,
        use_stacking: bool = False,
        stacking_alpha: float = 1.0,
        reweight_halflife: int = 20,
    ) -> None:
        self.use_stacking = use_stacking
        self.stacking_alpha = stacking_alpha
        self.reweight_halflife = reweight_halflife

        self._models: list[BaseModel] = [
            LinearModel(alpha=1.0),
            KernelModel(alpha=1.0, gamma=0.1),
            TreeModel(max_depth=4, min_samples_leaf=5),
            SVMModel(C=1.0, epsilon=0.01),
        ]
        self._weights: np.ndarray = np.ones(4) / 4.0
        self._meta: Optional[LinearModel] = None
        self._is_fitted: bool = False
        self._train_X: Optional[np.ndarray] = None
        self._train_y: Optional[np.ndarray] = None
        self._oof_predictions: Optional[np.ndarray] = None  # out-of-fold

    # -----------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleForecast":
        """Train all base models. If stacking, fit a meta-learner on OOF predictions."""
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n = X.shape[0]
        self._train_X = X
        self._train_y = y

        for model in self._models:
            model.fit(X, y)

        if self.use_stacking:
            # 5-fold OOF predictions for meta-learner
            folds = 5
            fold_size = n // folds
            oof = np.zeros((n, len(self._models)))

            for fold in range(folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < folds - 1 else n
                train_idx = np.concatenate([
                    np.arange(0, val_start),
                    np.arange(val_end, n),
                ])
                val_idx = np.arange(val_start, val_end)

                if len(train_idx) < 10:
                    continue

                for i, m in enumerate(self._models):
                    m_clone = m.__class__()
                    try:
                        m_clone.fit(X[train_idx], y[train_idx])
                        oof[val_idx, i] = m_clone.predict(X[val_idx])
                    except Exception:
                        oof[val_idx, i] = np.mean(y[train_idx])

            self._oof_predictions = oof
            self._meta = LinearModel(alpha=self.stacking_alpha)
            self._meta.fit(oof, y)

        self._is_fitted = True
        return self

    # -----------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble prediction."""
        X = np.asarray(X, dtype=float)
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted.")

        preds = np.column_stack([m.predict(X) for m in self._models])

        if self.use_stacking and self._meta is not None:
            return self._meta.predict(preds)

        return preds @ self._weights

    # -----------------------------------------------------------------------
    def update_weights(
        self,
        y_true: np.ndarray,
        predictions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Online weight update via exponential reweighting.
        Models with lower recent MSE get higher weight.
        predictions: (T, n_models) array; if None, re-predicts on stored train data.
        """
        y_true = np.asarray(y_true, dtype=float)

        if predictions is None:
            if self._train_X is None:
                raise RuntimeError("No training data stored — pass predictions array.")
            predictions = np.column_stack([m.predict(self._train_X) for m in self._models])

        predictions = np.asarray(predictions, dtype=float)
        n_models = len(self._models)

        # Use last min(T, reweight_halflife * 3) observations
        T = len(y_true)
        lookback = min(T, self.reweight_halflife * 3)
        y_recent = y_true[-lookback:]
        p_recent = predictions[-lookback:]

        # Exponential decay weights on time
        decay = np.exp(
            -np.arange(lookback - 1, -1, -1) / self.reweight_halflife
        )
        decay /= decay.sum()

        # Weighted MSE per model
        mse_per_model = np.array([
            float(np.sum(decay * (y_recent - p_recent[:, i]) ** 2))
            for i in range(n_models)
        ])

        # Convert MSE to weights: lower MSE → higher weight (softmax on neg MSE)
        neg_mse = -mse_per_model
        neg_mse -= neg_mse.max()
        exp_w = np.exp(neg_mse)
        self._weights = exp_w / exp_w.sum()

        return self._weights.copy()

    # -----------------------------------------------------------------------
    def feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Permutation importance: drop in ensemble MSE when feature j is shuffled.
        Returns array of shape (n_features,) with importance scores.
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        rng = np.random.default_rng(seed)

        base_preds = self.predict(X)
        base_mse = float(np.mean((y - base_preds) ** 2))
        n_features = X.shape[1]
        importances = np.zeros(n_features)

        for j in range(n_features):
            drop_sum = 0.0
            for _ in range(n_repeats):
                X_perm = X.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                perm_preds = self.predict(X_perm)
                perm_mse = float(np.mean((y - perm_preds) ** 2))
                drop_sum += perm_mse - base_mse
            importances[j] = drop_sum / n_repeats

        return importances

    # -----------------------------------------------------------------------
    def confidence_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap: int = 100,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Bootstrap confidence interval on ensemble predictions.
        Returns (lower_bound, upper_bound) arrays of shape (n_samples,).
        """
        X = np.asarray(X, dtype=float)
        if self._train_X is None or self._train_y is None:
            raise RuntimeError("No training data stored for bootstrap.")

        rng = np.random.default_rng(seed)
        n_train = self._train_X.shape[0]
        boot_preds = np.zeros((n_bootstrap, X.shape[0]))

        for b in range(n_bootstrap):
            idx = rng.integers(0, n_train, size=n_train)
            X_b = self._train_X[idx]
            y_b = self._train_y[idx]

            boot_ensemble = EnsembleForecast(
                use_stacking=False,
                reweight_halflife=self.reweight_halflife,
            )
            try:
                boot_ensemble.fit(X_b, y_b)
                boot_preds[b] = boot_ensemble.predict(X)
            except Exception:
                boot_preds[b] = np.mean(y_b)

        lower = np.percentile(boot_preds, 100 * (alpha / 2), axis=0)
        upper = np.percentile(boot_preds, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

    # -----------------------------------------------------------------------
    def walk_forward_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        window: int = 252,
    ) -> WalkForwardMetrics:
        """
        Rolling walk-forward cross-validation.
        Train on [t - window : t], predict [t : t + step].
        Returns aggregate metrics.
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n = len(y)
        step = max(1, window // 20)

        all_preds: list[float] = []
        all_actuals: list[float] = []

        for train_end in range(window, n - step + 1, step):
            train_X = X[train_end - window: train_end]
            train_y = y[train_end - window: train_end]
            val_X = X[train_end: train_end + step]
            val_y = y[train_end: train_end + step]

            wf_ens = EnsembleForecast(use_stacking=False)
            try:
                wf_ens.fit(train_X, train_y)
                preds = wf_ens.predict(val_X)
                all_preds.extend(preds.tolist())
                all_actuals.extend(val_y.tolist())
            except Exception:
                pass

        if not all_preds:
            return WalkForwardMetrics(0, 0, 0, 0, 0, 0)

        preds_arr = np.array(all_preds)
        actuals_arr = np.array(all_actuals)

        from scipy.stats import spearmanr
        ic_series = []
        cs = max(1, len(actuals_arr) // 20)
        for i in range(0, len(actuals_arr) - cs, cs):
            rho, _ = spearmanr(preds_arr[i:i+cs], actuals_arr[i:i+cs])
            if math.isfinite(rho):
                ic_series.append(rho)

        ic_arr = np.array(ic_series) if ic_series else np.array([0.0])
        mse = float(np.mean((preds_arr - actuals_arr) ** 2))
        dir_acc = float(np.mean(np.sign(preds_arr) == np.sign(actuals_arr)))

        pred_sharpe_ret = preds_arr * actuals_arr  # simplified: sign*return
        sr = (float(np.mean(pred_sharpe_ret)) / float(np.std(pred_sharpe_ret))
              if float(np.std(pred_sharpe_ret)) > 1e-10 else 0.0)

        return WalkForwardMetrics(
            ic_mean=float(np.mean(ic_arr)),
            ic_std=float(np.std(ic_arr)),
            mse_mean=mse,
            directional_accuracy=dir_acc,
            sharpe_of_predictions=sr * math.sqrt(252),
            n_windows=len(ic_arr),
        )

    # -----------------------------------------------------------------------
    def detect_concept_drift(
        self,
        X_recent: np.ndarray,
        y_recent: np.ndarray,
        threshold: float = 0.3,
    ) -> dict[str, float | bool]:
        """
        Detect concept drift by measuring disagreement among ensemble members.
        High variance across model predictions on recent data = potential drift.
        Also checks if recent prediction errors are significantly higher than training errors.

        Returns dict with: drift_detected, model_agreement, prediction_variance,
        error_ratio, recommendation.
        """
        X_recent = np.asarray(X_recent, dtype=float)
        y_recent = np.asarray(y_recent, dtype=float)

        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted.")

        # Model-level predictions
        preds = np.column_stack([m.predict(X_recent) for m in self._models])

        # Variance across models (disagreement)
        pred_variance = float(np.mean(np.var(preds, axis=1)))
        ensemble_pred = preds @ self._weights
        recent_mse = float(np.mean((ensemble_pred - y_recent) ** 2))

        # Training MSE benchmark
        if self._train_X is not None and self._train_y is not None:
            train_pred = self.predict(self._train_X)
            train_mse = float(np.mean((train_pred - self._train_y) ** 2))
            error_ratio = recent_mse / max(train_mse, 1e-12)
        else:
            error_ratio = 1.0

        # Agreement score: 1 - (normalized variance)
        pred_scale = float(np.std(ensemble_pred)) + 1e-10
        model_agreement = max(0.0, 1.0 - pred_variance / pred_scale ** 2)

        drift_detected = bool(
            model_agreement < (1.0 - threshold) or error_ratio > (1.0 + threshold * 2)
        )

        recommendation = (
            "REFIT RECOMMENDED: model agreement is low or recent errors significantly elevated."
            if drift_detected
            else "No significant drift detected. Models remain consistent."
        )

        return {
            "drift_detected": drift_detected,
            "model_agreement": round(model_agreement, 4),
            "prediction_variance": round(pred_variance, 8),
            "error_ratio": round(error_ratio, 4),
            "recent_mse": round(recent_mse, 8),
            "recommendation": recommendation,
        }

    # -----------------------------------------------------------------------
    @property
    def model_weights(self) -> dict[str, float]:
        """Return current model weights by name."""
        return {m.name: round(float(w), 4) for m, w in zip(self._models, self._weights)}
