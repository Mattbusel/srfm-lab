# ml/training/online_trainer.py -- online/incremental model training for SRFM live signals
from __future__ import annotations

import logging
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _rank_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman rank IC between predictions and labels."""
    from scipy import stats

    if len(y_pred) < 3:
        return 0.0
    corr, _ = stats.spearmanr(y_pred, y_true)
    return float(corr) if not np.isnan(corr) else 0.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


# ---------------------------------------------------------------------------
# ConceptDriftDetector
# ---------------------------------------------------------------------------


class ConceptDriftDetector:
    """
    CUSUM-based concept drift detector.

    Monitors the sequence of (prediction - actual) errors and raises a drift
    flag when the cumulative sum of (|error| - target_error) exceeds a
    control limit h.

    Parameters
    ----------
    target_error:
        Expected mean absolute prediction error under no-drift conditions.
    h:
        Control limit (CUSUM alarm threshold).  Typical values: 4 * sigma.
    slack:
        Allowable slack (k) in CUSUM, typically 0.5 * sigma.
    window:
        Rolling window used to compute recent error statistics.
    """

    def __init__(
        self,
        target_error: float = 0.1,
        h: float = 5.0,
        slack: float = 0.5,
        window: int = 200,
    ) -> None:
        self.target_error = target_error
        self.h = h
        self.slack = slack
        self.window = window

        self._cusum_pos: float = 0.0  # upper CUSUM accumulator
        self._cusum_neg: float = 0.0  # lower CUSUM accumulator
        self._errors: Deque[float] = deque(maxlen=window)
        self._n_updates: int = 0
        self._drift_detected: bool = False
        self._last_reset: int = 0

    def update(self, prediction: float, actual: float) -> None:
        """
        Feed one new (prediction, actual) pair.

        Parameters
        ----------
        prediction:
            Model's predicted value for this bar.
        actual:
            Realized label / return for this bar.
        """
        error = abs(float(prediction) - float(actual))
        self._errors.append(error)
        self._n_updates += 1

        # CUSUM update
        deviation = error - (self.target_error + self.slack)
        self._cusum_pos = max(0.0, self._cusum_pos + deviation)
        self._cusum_neg = max(0.0, self._cusum_neg - deviation - 2 * self.slack)

        if self._cusum_pos > self.h or self._cusum_neg > self.h:
            self._drift_detected = True
        else:
            self._drift_detected = False

    def is_drifting(self) -> bool:
        """Return True if CUSUM has exceeded the control limit."""
        return self._drift_detected

    def drift_score(self) -> float:
        """
        Return a normalized drift score in [0, 1].

        0 = completely stable, 1 = definite drift.
        """
        max_cusum = max(self._cusum_pos, self._cusum_neg)
        return float(min(1.0, max_cusum / (self.h + 1e-12)))

    def reset(self) -> None:
        """Reset CUSUM accumulators after a retrain event."""
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._drift_detected = False
        self._last_reset = self._n_updates
        logger.debug("ConceptDriftDetector reset after %d updates", self._n_updates)

    @property
    def mean_recent_error(self) -> float:
        if not self._errors:
            return 0.0
        return float(np.mean(list(self._errors)))

    @property
    def n_updates(self) -> int:
        return self._n_updates


# ---------------------------------------------------------------------------
# IncrementalSGDSignal
# ---------------------------------------------------------------------------


class IncrementalSGDSignal:
    """
    Logistic regression trained incrementally via mini-batch SGD.

    Features expected (in order):
        momentum, vol_ratio, BH_mass, Hurst, OFI, funding_rate

    Outputs a probability in [0, 1] where > 0.5 corresponds to a long signal.

    Parameters
    ----------
    n_features:
        Number of input features.
    learning_rate:
        Initial SGD learning rate.
    lr_decay:
        Multiplicative decay applied every decay_every steps.
    decay_every:
        Number of updates between learning rate decay steps.
    l2:
        L2 regularization coefficient.
    checkpoint_dir:
        Directory for automatic checkpoints.
    checkpoint_every:
        Save checkpoint every N updates.
    """

    DEFAULT_FEATURES: List[str] = [
        "momentum", "vol_ratio", "BH_mass", "Hurst", "OFI", "funding_rate"
    ]

    def __init__(
        self,
        n_features: int = 6,
        learning_rate: float = 0.01,
        lr_decay: float = 0.995,
        decay_every: int = 50,
        l2: float = 1e-4,
        checkpoint_dir: str = "checkpoints",
        checkpoint_every: int = 100,
    ) -> None:
        self.n_features = n_features
        self.lr0 = learning_rate
        self.lr_decay = lr_decay
        self.decay_every = decay_every
        self.l2 = l2
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every

        # Parameters
        self.weights: np.ndarray = np.zeros(n_features, dtype=float)
        self.bias: float = 0.0
        self.lr: float = learning_rate

        self._n_updates: int = 0

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _forward(self, features: np.ndarray) -> float:
        """Logistic forward pass, returns probability."""
        x = np.asarray(features, dtype=float).ravel()
        logit = float(np.dot(self.weights, x) + self.bias)
        return float(_sigmoid(np.array([logit]))[0])

    def update(self, features: np.ndarray, label: float) -> float:
        """
        Perform one SGD update step.

        Parameters
        ----------
        features:
            1-D feature vector of length n_features.
        label:
            Binary label in {0, 1} or continuous in [0, 1].

        Returns
        -------
        float
            Binary cross-entropy loss for this sample.
        """
        x = np.asarray(features, dtype=float).ravel()[: self.n_features]
        y = float(label)

        prob = self._forward(x)
        error = prob - y  # gradient of BCE w.r.t. logit

        # BCE loss
        eps = 1e-12
        loss = -(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))

        # Gradient update with L2 regularization
        self.weights -= self.lr * (error * x + self.l2 * self.weights)
        self.bias -= self.lr * error

        self._n_updates += 1

        # Learning rate decay
        if self._n_updates % self.decay_every == 0:
            self.lr = max(1e-6, self.lr * self.lr_decay)

        # Auto-checkpoint
        if self._n_updates % self.checkpoint_every == 0:
            self._save_checkpoint()

        return float(loss)

    def predict(self, features: np.ndarray) -> float:
        """Return probability in [0, 1]."""
        x = np.asarray(features, dtype=float).ravel()[: self.n_features]
        return self._forward(x)

    def _save_checkpoint(self) -> None:
        path = self.checkpoint_dir / f"sgd_signal_{self._n_updates}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "weights": self.weights.copy(),
                    "bias": self.bias,
                    "lr": self.lr,
                    "n_updates": self._n_updates,
                },
                fh,
            )
        logger.debug("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load weights from a checkpoint file."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        self.weights = state["weights"]
        self.bias = state["bias"]
        self.lr = state["lr"]
        self._n_updates = state["n_updates"]
        logger.info("Loaded checkpoint from %s (n_updates=%d)", path, self._n_updates)

    def latest_checkpoint(self) -> Optional[str]:
        """Return the path to the most recent checkpoint, or None."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("sgd_signal_*.pkl"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if not checkpoints:
            return None
        return str(checkpoints[-1])

    @property
    def n_updates(self) -> int:
        return self._n_updates


# ---------------------------------------------------------------------------
# WarmStartTrainer
# ---------------------------------------------------------------------------


class WarmStartTrainer:
    """
    Initializes from a pre-trained model and continues training on new data.

    Designed for daily or weekly warm-start updates where the bulk of the
    model parameters are loaded from a checkpoint and only fine-tuning
    updates are applied.

    Parameters
    ----------
    model:
        Pre-trained model object with a fit(X, y) or update(x, y) method.
    feature_names:
        Names of expected input features.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names or []
        self._checkpoint_path: Optional[str] = None

    def load_checkpoint(self, path: str) -> None:
        """Load model state from a pickle checkpoint."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)

        if isinstance(state, dict):
            self.model = state.get("model", state)
            self.feature_names = state.get("feature_names", self.feature_names)
        else:
            self.model = state

        self._checkpoint_path = path
        logger.info("WarmStartTrainer loaded checkpoint from %s", path)

    def save_checkpoint(self, path: str) -> None:
        """Save current model state to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {"model": self.model, "feature_names": self.feature_names},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("WarmStartTrainer saved checkpoint to %s", path)

    def warm_start_update(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 1,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Update the model on new data using mini-batch SGD.

        Works when the underlying model is an IncrementalSGDSignal.
        For other model types, falls back to calling model.fit(features, labels).

        Parameters
        ----------
        features:
            Array of shape (n_samples, n_features).
        labels:
            Array of shape (n_samples,).
        n_epochs:
            Number of passes over the data.
        batch_size:
            Mini-batch size for SGD-compatible models.

        Returns
        -------
        Dict with training statistics: {"mean_loss": float, "n_samples": int}
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")

        X = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float)
        n_samples = len(X)

        if isinstance(self.model, IncrementalSGDSignal):
            losses: List[float] = []
            for _ in range(n_epochs):
                idx = np.random.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    batch_idx = idx[start: start + batch_size]
                    for i in batch_idx:
                        loss = self.model.update(X[i], float(y[i]))
                        losses.append(loss)
            return {
                "mean_loss": float(np.mean(losses)) if losses else float("nan"),
                "n_samples": n_samples,
            }

        # Generic fallback: call fit()
        if hasattr(self.model, "fit"):
            for _ in range(n_epochs):
                self.model.fit(X, y)
            return {"mean_loss": float("nan"), "n_samples": n_samples}

        raise TypeError(f"Model of type {type(self.model)} has no fit() or update() method")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predictions from the underlying model."""
        if self.model is None:
            raise RuntimeError("No model loaded.")
        X = np.asarray(features, dtype=float)
        if isinstance(self.model, IncrementalSGDSignal):
            return np.array([self.model.predict(X[i]) for i in range(len(X))])
        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(X))
        raise TypeError(f"Model of type {type(self.model)} has no predict() method")


# ---------------------------------------------------------------------------
# OnlineTrainer
# ---------------------------------------------------------------------------


class OnlineTrainer:
    """
    Online/incremental training manager for SRFM live signal models.

    Wraps an IncrementalSGDSignal with drift detection and automatic
    retrain scheduling.  Keeps a rolling window of recent predictions
    and labels for performance monitoring.

    Parameters
    ----------
    model_name:
        Logical name for this signal (used in checkpoint naming).
    feature_names:
        List of feature names (for logging and validation).
    n_features:
        Number of input features (inferred from feature_names if given).
    learning_rate:
        Initial SGD learning rate.
    drift_h:
        CUSUM control limit for drift detection.
    retrain_threshold_ic:
        Trigger retrain if rolling IC drops below this value.
    monitor_window:
        Number of recent samples used for performance monitoring.
    checkpoint_dir:
        Directory for model checkpoints.
    """

    def __init__(
        self,
        model_name: str,
        feature_names: Optional[List[str]] = None,
        n_features: int = 6,
        learning_rate: float = 0.01,
        drift_h: float = 5.0,
        retrain_threshold_ic: float = 0.02,
        monitor_window: int = 500,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.model_name = model_name
        self.feature_names = feature_names or IncrementalSGDSignal.DEFAULT_FEATURES
        if feature_names is not None:
            n_features = len(feature_names)

        self.retrain_threshold_ic = retrain_threshold_ic
        self.monitor_window = monitor_window

        self.model = IncrementalSGDSignal(
            n_features=n_features,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
        )

        self.drift_detector = ConceptDriftDetector(h=drift_h)

        # Rolling buffers for performance monitoring
        self._pred_buffer: Deque[float] = deque(maxlen=monitor_window)
        self._label_buffer: Deque[float] = deque(maxlen=monitor_window)
        self._loss_buffer: Deque[float] = deque(maxlen=monitor_window)

        # Retrain tracking
        self._last_retrain_at: int = 0
        self._total_updates: int = 0

    def update(self, features: np.ndarray, label: float) -> None:
        """
        Perform an incremental update on one new sample.

        Parameters
        ----------
        features:
            1-D feature array.
        label:
            Realized label (e.g., next-bar return sign in {0, 1}).
        """
        x = np.asarray(features, dtype=float)
        pred = self.model.predict(x)
        loss = self.model.update(x, label)

        self._pred_buffer.append(pred)
        self._label_buffer.append(float(label))
        self._loss_buffer.append(loss)

        self.drift_detector.update(pred, float(label))
        self._total_updates += 1

    def predict(self, features: np.ndarray) -> float:
        """Return a signal probability in [0, 1]."""
        return self.model.predict(np.asarray(features, dtype=float))

    def performance_since_last_retrain(self) -> Dict[str, float]:
        """
        Compute IC and mean loss since the last retrain event.

        Returns
        -------
        Dict with keys: ic, mean_loss, n_samples, drift_score
        """
        preds = np.array(list(self._pred_buffer))
        labels = np.array(list(self._label_buffer))
        losses = np.array(list(self._loss_buffer))

        if len(preds) < 3:
            return {
                "ic": 0.0,
                "mean_loss": float(np.mean(losses)) if len(losses) else float("nan"),
                "n_samples": int(len(preds)),
                "drift_score": self.drift_detector.drift_score(),
            }

        ic = _rank_ic(preds, labels)
        return {
            "ic": ic,
            "mean_loss": float(np.mean(losses)),
            "n_samples": int(len(preds)),
            "drift_score": self.drift_detector.drift_score(),
        }

    def should_retrain(self) -> bool:
        """
        Determine if a full retrain is warranted.

        Returns True if:
          - Drift detector is signaling drift, or
          - Rolling IC has dropped below retrain_threshold_ic
        """
        if self.drift_detector.is_drifting():
            logger.info(
                "Retrain triggered: drift detected (score=%.3f)",
                self.drift_detector.drift_score(),
            )
            return True

        perf = self.performance_since_last_retrain()
        if perf["n_samples"] >= 50 and perf["ic"] < self.retrain_threshold_ic:
            logger.info(
                "Retrain triggered: IC=%.4f below threshold=%.4f",
                perf["ic"],
                self.retrain_threshold_ic,
            )
            return True

        return False

    def acknowledge_retrain(self) -> None:
        """
        Call this after completing a full retrain.

        Resets the drift detector and rolling buffers.
        """
        self.drift_detector.reset()
        self._pred_buffer.clear()
        self._label_buffer.clear()
        self._loss_buffer.clear()
        self._last_retrain_at = self._total_updates
        logger.info(
            "OnlineTrainer %s acknowledged retrain at update %d",
            self.model_name,
            self._total_updates,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model weights from a checkpoint file."""
        self.model.load_checkpoint(path)

    def latest_checkpoint(self) -> Optional[str]:
        """Return path to the most recent model checkpoint."""
        return self.model.latest_checkpoint()

    @property
    def total_updates(self) -> int:
        return self._total_updates

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict for logging/monitoring."""
        perf = self.performance_since_last_retrain()
        return {
            "model_name": self.model_name,
            "total_updates": self._total_updates,
            "sgd_updates": self.model.n_updates,
            "current_lr": self.model.lr,
            **perf,
            "should_retrain": self.should_retrain(),
        }
