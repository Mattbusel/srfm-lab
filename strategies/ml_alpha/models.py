"""
ml_alpha/models.py — ML alpha models.

Implements:
- LightGBMAlpha
- RandomForestAlpha
- XGBoostAlpha
- NeuralNetAlpha (PyTorch LSTM)
- EnsembleAlpha
- LinearAlpha (Ridge/Lasso/ElasticNet)

All models share a common interface:
    model.train(X, y) -> None
    model.predict(X) -> np.ndarray
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _information_coefficient(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Spearman rank correlation (IC)."""
    from scipy.stats import spearmanr
    if len(y_pred) < 5:
        return 0.0
    valid = ~(np.isnan(y_pred) | np.isnan(y_true))
    if valid.sum() < 5:
        return 0.0
    ic, _ = spearmanr(y_pred[valid], y_true[valid])
    return float(ic)


def _check_fit(model_name: str, is_fitted: bool):
    if not is_fitted:
        raise RuntimeError(f"{model_name} must be fitted before calling predict()")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LightGBMAlpha
# ─────────────────────────────────────────────────────────────────────────────

class LightGBMAlpha:
    """
    LightGBM gradient boosting model for alpha prediction.

    Targets a forward return series. Uses early stopping on validation set.
    Provides SHAP values and feature importance.

    Parameters
    ----------
    n_estimators    : max number of trees (default 500)
    learning_rate   : boosting learning rate (default 0.05)
    num_leaves      : max leaves per tree (default 31)
    max_depth       : max tree depth (default -1 = no limit)
    min_child_samples: min samples per leaf (default 20)
    subsample       : row sampling fraction (default 0.8)
    colsample_bytree: column sampling fraction (default 0.8)
    reg_alpha       : L1 regularization (default 0.0)
    reg_lambda      : L2 regularization (default 1.0)
    early_stopping_rounds: patience for early stopping (default 50)
    objective       : LGB objective (default "regression")
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        objective: str = "regression",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.objective = objective
        self.random_state = random_state

        self._model = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        self._train_ic: float = 0.0
        self._val_ic: float = 0.0
        self._best_iteration: int = 0

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Train LightGBM model with optional early stopping on validation set.

        Parameters
        ----------
        X            : training features (n_samples, n_features)
        y            : training labels (n_samples,)
        val_X        : validation features (optional)
        val_y        : validation labels (optional)
        feature_names: optional list of feature names
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")

        # Remove NaN rows
        valid_train = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid_train]
        y_clean = y[valid_train]

        if feature_names is not None:
            self._feature_names = feature_names
        else:
            self._feature_names = [f"f{i}" for i in range(X.shape[1])]

        params = {
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

        train_set = lgb.Dataset(X_clean, label=y_clean, feature_name=self._feature_names)

        callbacks = []
        valid_sets = [train_set]
        valid_names = ["train"]

        if val_X is not None and val_y is not None:
            valid_val = ~(np.isnan(val_X).any(axis=1) | np.isnan(val_y))
            val_set = lgb.Dataset(val_X[valid_val], label=val_y[valid_val])
            valid_sets.append(val_set)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(period=-1))

        self._model = lgb.train(
            params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self._best_iteration = self._model.best_iteration or self.n_estimators
        self._is_fitted = True

        # Compute IC
        y_pred_train = self._model.predict(X_clean)
        self._train_ic = _information_coefficient(y_pred_train, y_clean)

        if val_X is not None and val_y is not None:
            val_pred = self._model.predict(val_X[~(np.isnan(val_X).any(axis=1) | np.isnan(val_y))])
            self._val_ic = _information_coefficient(val_pred, val_y[~(np.isnan(val_X).any(axis=1) | np.isnan(val_y))])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        _check_fit("LightGBMAlpha", self._is_fitted)
        valid = ~np.isnan(X).any(axis=1)
        preds = np.full(len(X), np.nan)
        if valid.any():
            preds[valid] = self._model.predict(X[valid], num_iteration=self._best_iteration)
        return preds

    def feature_importance(self) -> pd.Series:
        """Return feature importance (gain-based)."""
        _check_fit("LightGBMAlpha", self._is_fitted)
        imp = self._model.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self._feature_names).sort_values(ascending=False)

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values using LightGBM's built-in method.

        Returns array of shape (n_samples, n_features).
        Requires shap: pip install shap
        """
        _check_fit("LightGBMAlpha", self._is_fitted)
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            valid = ~np.isnan(X).any(axis=1)
            shap_vals = np.zeros((len(X), X.shape[1]))
            if valid.any():
                shap_vals[valid] = explainer.shap_values(X[valid])
            return shap_vals
        except ImportError:
            # Fallback: use LGB's predict with pred_contrib
            shap_raw = self._model.predict(X, pred_contrib=True)
            # Last column is bias — remove it
            return shap_raw[:, :-1]

    def training_summary(self) -> dict:
        return {
            "best_iteration": self._best_iteration,
            "train_ic": self._train_ic,
            "val_ic": self._val_ic,
            "n_features": len(self._feature_names),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. RandomForestAlpha
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestAlpha:
    """
    Random Forest model for alpha prediction.

    Uses sklearn's RandomForestRegressor with OOB score.

    Parameters
    ----------
    n_estimators  : number of trees (default 200)
    max_depth     : max tree depth (default None)
    max_features  : features per split (default "sqrt")
    min_samples_leaf: min samples per leaf (default 20)
    n_jobs        : parallel jobs (default -1)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        max_features: Any = "sqrt",
        min_samples_leaf: int = 20,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._model = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        self._oob_score: float = 0.0
        self._train_ic: float = 0.0

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Train Random Forest on X, y."""
        from sklearn.ensemble import RandomForestRegressor

        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid]
        y_clean = y[valid]

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            oob_score=True,
        )
        self._model.fit(X_clean, y_clean)
        self._is_fitted = True
        self._oob_score = float(self._model.oob_score_)
        self._train_ic = _information_coefficient(self._model.predict(X_clean), y_clean)

    def predict(self, X: np.ndarray) -> np.ndarray:
        _check_fit("RandomForestAlpha", self._is_fitted)
        valid = ~np.isnan(X).any(axis=1)
        preds = np.full(len(X), np.nan)
        if valid.any():
            preds[valid] = self._model.predict(X[valid])
        return preds

    def oob_score(self) -> float:
        """Return Out-of-Bag R² score."""
        _check_fit("RandomForestAlpha", self._is_fitted)
        return self._oob_score

    def feature_importance(self) -> pd.Series:
        _check_fit("RandomForestAlpha", self._is_fitted)
        return pd.Series(
            self._model.feature_importances_,
            index=self._feature_names
        ).sort_values(ascending=False)

    def training_summary(self) -> dict:
        return {
            "n_trees": self.n_estimators,
            "oob_score": self._oob_score,
            "train_ic": self._train_ic,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. XGBoostAlpha
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostAlpha:
    """
    XGBoost gradient boosting model for alpha prediction.

    Parameters
    ----------
    n_estimators        : max rounds (default 500)
    learning_rate       : step size (default 0.05)
    max_depth           : tree depth (default 6)
    subsample           : row subsampling (default 0.8)
    colsample_bytree    : column subsampling (default 0.8)
    min_child_weight    : min leaf weight (default 5)
    reg_alpha           : L1 regularization (default 0.0)
    reg_lambda          : L2 regularization (default 1.0)
    early_stopping_rounds: patience (default 50)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        self._model = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        self._best_iteration: int = 0
        self._train_ic: float = 0.0
        self._val_ic: float = 0.0

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Train XGBoost with optional early stopping."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid]; y_clean = y[valid]
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        params = {
            "objective": "reg:squarederror",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }

        dtrain = xgb.DMatrix(X_clean, label=y_clean, feature_names=self._feature_names)
        eval_list = [(dtrain, "train")]

        esr = None
        if val_X is not None and val_y is not None:
            valid_val = ~(np.isnan(val_X).any(axis=1) | np.isnan(val_y))
            dval = xgb.DMatrix(val_X[valid_val], label=val_y[valid_val])
            eval_list.append((dval, "valid"))
            esr = self.early_stopping_rounds

        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=eval_list,
            early_stopping_rounds=esr,
            verbose_eval=False,
        )

        self._best_iteration = getattr(self._model, "best_iteration", self.n_estimators)
        self._is_fitted = True

        import xgboost as xgb_imp
        y_pred_train = self._model.predict(xgb_imp.DMatrix(X_clean))
        self._train_ic = _information_coefficient(y_pred_train, y_clean)

        if val_X is not None and val_y is not None:
            vv = ~(np.isnan(val_X).any(axis=1) | np.isnan(val_y))
            val_pred = self._model.predict(xgb_imp.DMatrix(val_X[vv]))
            self._val_ic = _information_coefficient(val_pred, val_y[vv])

    def predict(self, X: np.ndarray) -> np.ndarray:
        _check_fit("XGBoostAlpha", self._is_fitted)
        try:
            import xgboost as xgb
            valid = ~np.isnan(X).any(axis=1)
            preds = np.full(len(X), np.nan)
            if valid.any():
                preds[valid] = self._model.predict(xgb.DMatrix(X[valid]))
            return preds
        except ImportError:
            return np.zeros(len(X))

    def feature_importance(self) -> pd.Series:
        _check_fit("XGBoostAlpha", self._is_fitted)
        imp = self._model.get_score(importance_type="gain")
        return pd.Series(imp).sort_values(ascending=False)

    def training_summary(self) -> dict:
        return {
            "best_iteration": self._best_iteration,
            "train_ic": self._train_ic,
            "val_ic": self._val_ic,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. NeuralNetAlpha (PyTorch LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class NeuralNetAlpha:
    """
    PyTorch LSTM neural network for time-series alpha prediction.

    Architecture:
    - LSTM layers (n_layers deep) with dropout
    - Linear output head
    - Input: (sequence_length, batch_size, input_dim)
    - Output: scalar prediction per sequence

    Parameters
    ----------
    input_dim    : number of input features
    hidden_dim   : LSTM hidden dimension (default 128)
    n_layers     : number of LSTM layers (default 2)
    dropout      : dropout between LSTM layers (default 0.2)
    sequence_len : input sequence length (default 20)
    """

    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        sequence_len: int = 20,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.sequence_len = sequence_len
        self._model = None
        self._is_fitted = False
        self._loss_history: List[float] = []

    def _build_model(self):
        """Build the LSTM model."""
        try:
            import torch
            import torch.nn as nn

            class LSTMModel(nn.Module):
                def __init__(self, input_dim, hidden_dim, n_layers, dropout):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_dim,
                        hidden_size=hidden_dim,
                        num_layers=n_layers,
                        dropout=dropout if n_layers > 1 else 0.0,
                        batch_first=True,
                    )
                    self.dropout_layer = nn.Dropout(dropout)
                    self.fc = nn.Linear(hidden_dim, 1)
                    self.bn = nn.BatchNorm1d(hidden_dim)

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    # Take last time step
                    last_out = lstm_out[:, -1, :]
                    last_out = self.dropout_layer(last_out)
                    out = self.fc(last_out)
                    return out.squeeze(-1)

            return LSTMModel(self.input_dim, self.hidden_dim, self.n_layers, self.dropout)
        except ImportError:
            return None

    def _prepare_sequences(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple:
        """Convert flat feature matrix to sequences for LSTM."""
        n = len(X)
        sequences = []
        targets = []

        for i in range(self.sequence_len, n + 1):
            seq = X[i - self.sequence_len:i]
            if not np.isnan(seq).any():
                sequences.append(seq)
                if y is not None:
                    targets.append(y[i - 1])

        if len(sequences) == 0:
            return np.array([]), np.array([]) if y is not None else np.array([])

        X_seq = np.array(sequences, dtype=np.float32)
        if y is not None:
            y_seq = np.array(targets, dtype=np.float32)
            return X_seq, y_seq
        return X_seq

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        val_fraction: float = 0.1,
    ) -> List[float]:
        """
        Train the LSTM.

        Parameters
        ----------
        X         : features (n_samples, n_features)
        y         : targets (n_samples,)
        epochs    : training epochs
        batch_size: mini-batch size
        lr        : learning rate
        val_fraction: fraction for validation

        Returns
        -------
        List of training loss per epoch.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            raise ImportError("PyTorch not installed. Run: pip install torch")

        # Adjust input_dim to match actual features
        self.input_dim = X.shape[1]
        model = self._build_model()
        if model is None:
            raise RuntimeError("Could not build PyTorch model")

        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(X, y)
        if len(X_seq) == 0:
            return []

        # Train/val split
        n_val = max(1, int(len(X_seq) * val_fraction))
        X_train, X_val = X_seq[:-n_val], X_seq[-n_val:]
        y_train, y_val = y_seq[:-n_val], y_seq[-n_val:]

        # Datasets
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = nn.MSELoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        best_val_loss = float("inf")
        best_state = None
        loss_history = []

        for epoch in range(epochs):
            model.train()
            train_losses = []
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(Xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_dl:
                    Xb, yb = Xb.to(device), yb.to(device)
                    pred = model(Xb)
                    val_losses.append(float(loss_fn(pred, yb).item()))

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses) if val_losses else train_loss
            loss_history.append(train_loss)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        self._model = model.cpu()
        self._is_fitted = True
        self._loss_history = loss_history
        return loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions. Returns array aligned with input (NaN in warmup)."""
        _check_fit("NeuralNetAlpha", self._is_fitted)
        try:
            import torch
        except ImportError:
            return np.zeros(len(X))

        self._model.eval()
        n = len(X)
        preds = np.full(n, np.nan)

        for i in range(self.sequence_len, n + 1):
            seq = X[i - self.sequence_len:i]
            if np.isnan(seq).any():
                continue
            x_tensor = torch.tensor(seq[np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                pred = self._model(x_tensor)
            preds[i - 1] = float(pred.item())

        return preds

    def loss_history(self) -> List[float]:
        """Return training loss history."""
        return self._loss_history


# ─────────────────────────────────────────────────────────────────────────────
# 5. EnsembleAlpha
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleAlpha:
    """
    Ensemble of alpha models with weighted average predictions.

    Supports:
    - Simple average (equal weights)
    - Weighted average (by IC or custom weights)
    - Stacking (use meta-model on model predictions)

    Parameters
    ----------
    models  : list of fitted model instances (must have .predict() method)
    weights : list of weights (must sum to 1). If None: equal weights.
    """

    def __init__(
        self,
        models: List,
        weights: Optional[List[float]] = None,
    ):
        self.models = models
        if weights is not None:
            w = np.array(weights, dtype=float)
            self.weights = (w / w.sum()).tolist()
        else:
            self.weights = [1.0 / len(models)] * len(models)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted average of model predictions.

        Returns array of shape (n_samples,).
        NaN where all models predict NaN.
        """
        all_preds = []
        for model in self.models:
            try:
                p = model.predict(X)
                all_preds.append(p)
            except Exception:
                all_preds.append(np.full(len(X), np.nan))

        # Weighted average (ignore NaN for each sample)
        pred_stack = np.array(all_preds)  # (n_models, n_samples)
        weights = np.array(self.weights)

        result = np.full(pred_stack.shape[1], np.nan)
        for j in range(pred_stack.shape[1]):
            col = pred_stack[:, j]
            valid = ~np.isnan(col)
            if valid.any():
                w = weights[valid] / weights[valid].sum()
                result[j] = float(np.dot(w, col[valid]))

        return result

    def ic_weights(
        self,
        val_X: np.ndarray,
        val_y: np.ndarray,
    ) -> List[float]:
        """
        Compute IC-weighted weights on a validation set.
        Returns new weights proportional to IC of each model.
        """
        ics = []
        for model in self.models:
            try:
                preds = model.predict(val_X)
                ic = _information_coefficient(preds, val_y)
                ics.append(max(0.0, ic))  # negative IC → 0 weight
            except Exception:
                ics.append(0.0)

        total_ic = sum(ics)
        if total_ic <= 0:
            return [1.0 / len(self.models)] * len(self.models)
        return [ic / total_ic for ic in ics]

    def set_ic_weights(self, val_X: np.ndarray, val_y: np.ndarray) -> None:
        """Update weights based on IC performance on validation set."""
        self.weights = self.ic_weights(val_X, val_y)

    def individual_predictions(self, X: np.ndarray) -> pd.DataFrame:
        """Return individual model predictions as DataFrame."""
        preds = {}
        for i, model in enumerate(self.models):
            name = getattr(model, "__class__", type(model)).__name__
            try:
                preds[f"{name}_{i}"] = model.predict(X)
            except Exception:
                preds[f"{name}_{i}"] = np.full(len(X), np.nan)
        return pd.DataFrame(preds)


# ─────────────────────────────────────────────────────────────────────────────
# 6. LinearAlpha (Ridge, Lasso, ElasticNet)
# ─────────────────────────────────────────────────────────────────────────────

class LinearAlpha:
    """
    Regularized linear models for alpha prediction.

    Supports Ridge, Lasso, and ElasticNet via sklearn.
    Optionally applies PCA dimensionality reduction.

    Parameters
    ----------
    method      : "ridge", "lasso", or "elasticnet" (default "ridge")
    alpha       : regularization strength (default 1.0)
    l1_ratio    : ElasticNet mixing (0=Ridge, 1=Lasso, default 0.5)
    fit_intercept: fit intercept (default True)
    n_pca_components: PCA components (default None = no PCA)
    """

    def __init__(
        self,
        method: str = "ridge",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        n_pca_components: Optional[int] = None,
        normalize_features: bool = True,
    ):
        self.method = method.lower()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.n_pca_components = n_pca_components
        self.normalize_features = normalize_features

        self._model = None
        self._scaler = None
        self._pca = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        self._train_ic: float = 0.0
        self._coef: np.ndarray = np.array([])

    def _get_model(self):
        """Instantiate the sklearn model."""
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        if self.method == "ridge":
            return Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        elif self.method == "lasso":
            return Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept, max_iter=5000)
        else:
            return ElasticNet(
                alpha=self.alpha, l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept, max_iter=5000
            )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Train the linear model.

        Parameters
        ----------
        X            : features (n_samples, n_features)
        y            : targets (n_samples,)
        feature_names: optional feature names for interpretability
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid]; y_clean = y[valid]
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # Normalize
        if self.normalize_features:
            self._scaler = StandardScaler()
            X_clean = self._scaler.fit_transform(X_clean)

        # PCA
        if self.n_pca_components is not None:
            n_comp = min(self.n_pca_components, X_clean.shape[1], X_clean.shape[0] - 1)
            self._pca = PCA(n_components=n_comp)
            X_clean = self._pca.fit_transform(X_clean)

        self._model = self._get_model()
        self._model.fit(X_clean, y_clean)
        self._is_fitted = True
        self._coef = self._model.coef_

        y_pred = self._model.predict(X_clean)
        self._train_ic = _information_coefficient(y_pred, y_clean)

    def predict(self, X: np.ndarray) -> np.ndarray:
        _check_fit("LinearAlpha", self._is_fitted)
        valid = ~np.isnan(X).any(axis=1)
        preds = np.full(len(X), np.nan)

        X_proc = X.copy()
        if valid.any():
            X_proc_valid = X_proc[valid]
            if self._scaler is not None:
                X_proc_valid = self._scaler.transform(X_proc_valid)
            if self._pca is not None:
                X_proc_valid = self._pca.transform(X_proc_valid)
            preds[valid] = self._model.predict(X_proc_valid)

        return preds

    def coefficients(self) -> pd.Series:
        """Return feature coefficients (not available when PCA is used)."""
        _check_fit("LinearAlpha", self._is_fitted)
        if self._pca is not None:
            return pd.Series(self._coef, name="pca_component_coef")
        return pd.Series(self._coef, index=self._feature_names).sort_values(key=abs, ascending=False)

    def training_summary(self) -> dict:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "train_ic": self._train_ic,
            "n_features": len(self._feature_names),
            "n_nonzero": int(np.sum(np.abs(self._coef) > 1e-8)) if len(self._coef) > 0 else 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1000
    n_feat = 30

    # Synthetic features and targets
    X = rng.normal(0, 1, (n, n_feat)).astype(np.float32)
    # Target: weak linear combination of first 5 features + noise
    true_alpha = X[:, :5] @ rng.normal(0, 0.1, 5)
    y = true_alpha + rng.normal(0, 0.5, n)

    n_train = 700
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    # Linear model
    lin = LinearAlpha(method="ridge", alpha=1.0)
    lin.train(X_train, y_train)
    pred = lin.predict(X_val)
    ic = _information_coefficient(pred[~np.isnan(pred)], y_val[~np.isnan(pred)])
    print(f"LinearAlpha Ridge: train_IC={lin._train_ic:.3f}, val_IC={ic:.3f}")
    print(lin.training_summary())

    # Lasso
    lasso = LinearAlpha(method="lasso", alpha=0.01)
    lasso.train(X_train, y_train)
    pred_l = lasso.predict(X_val)
    ic_l = _information_coefficient(pred_l[~np.isnan(pred_l)], y_val[~np.isnan(pred_l)])
    print(f"LinearAlpha Lasso: val_IC={ic_l:.3f}")

    # Random Forest
    rf = RandomForestAlpha(n_estimators=50, min_samples_leaf=5)
    rf.train(X_train, y_train)
    pred_rf = rf.predict(X_val)
    ic_rf = _information_coefficient(pred_rf[~np.isnan(pred_rf)], y_val[~np.isnan(pred_rf)])
    print(f"RandomForest: OOB={rf.oob_score():.3f}, val_IC={ic_rf:.3f}")

    # Ensemble
    ens = EnsembleAlpha([lin, lasso, rf])
    ens.set_ic_weights(X_val, y_val)
    pred_ens = ens.predict(X_val)
    ic_ens = _information_coefficient(pred_ens[~np.isnan(pred_ens)], y_val[~np.isnan(pred_ens)])
    print(f"Ensemble (IC-weighted): val_IC={ic_ens:.3f}, weights={ens.weights}")
