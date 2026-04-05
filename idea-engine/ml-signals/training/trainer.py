"""
training/trainer.py
====================
MLTrainer: walk-forward training with early stopping and model persistence.

Financial rationale
-------------------
Standard k-fold cross-validation violates the temporal ordering of
financial data: training on a mix of past and future observations is
equivalent to using a crystal ball.  Walk-forward training addresses
this by always training on data that precedes the validation window.

Walk-forward schedule:
    Initial train window : first ``train_window`` bars
    Expanding window     : add ``retrain_every`` bars at each step
    Validation window    : next ``val_window`` bars after train end

Early stopping:
    Monitor validation MSE loss.  If no improvement for ``patience``
    steps, roll back to best weights and stop.

Model management:
    One model file per (instrument, timeframe, model_type).
    Saved under: ``model_dir / instrument / timeframe / model_type /``
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd

from ..models.base import MLSignal, SignalMetrics
from ..models.lstm_signal import LSTMSignal
from ..models.transformer_signal import TransformerSignal
from ..models.xgboost_signal import XGBoostSignal
from ..models.ensemble import EnsembleSignal


@dataclass
class TrainingResult:
    """Summary of one training run."""
    instrument:   str
    timeframe:    str
    model_name:   str
    train_start:  str
    train_end:    str
    val_start:    str
    val_end:      str
    val_ic:       float
    val_icir:     float
    best_epoch:   int
    train_sec:    float
    model_path:   str
    metrics:      Optional[SignalMetrics] = None
    extra:        Dict = field(default_factory=dict)


class MLTrainer:
    """Orchestrates walk-forward training of ML signal models.

    Parameters
    ----------
    model_dir : Path
        Root directory for saving model weights.
    train_window : int
        Minimum number of bars in the initial training window.
    val_window : int
        Size of the validation window (bars).
    retrain_every : int
        Number of new bars before the next retrain step.
    patience : int
        Early stopping patience (training epochs without val improvement).
    """

    def __init__(
        self,
        model_dir: str | pathlib.Path = "models",
        train_window: int = 504,    # ~2 years daily
        val_window: int   = 126,    # ~6 months daily
        retrain_every: int = 30,    # monthly retrain
        patience: int      = 10,
    ) -> None:
        self.model_dir    = pathlib.Path(model_dir)
        self.train_window = train_window
        self.val_window   = val_window
        self.retrain_every = retrain_every
        self.patience      = patience
        self._history: List[TrainingResult] = []

    # ------------------------------------------------------------------
    # Walk-forward schedule generator
    # ------------------------------------------------------------------

    def _wf_splits(self, n: int):
        """Yield (train_end, val_end) integer positions for walk-forward."""
        start = self.train_window
        while start + self.val_window <= n:
            yield 0, start, start, start + self.val_window
            start += self.retrain_every

    # ------------------------------------------------------------------
    # Core train method
    # ------------------------------------------------------------------

    def train_walkforward(
        self,
        df: pd.DataFrame,
        model_class,
        model_kwargs: dict,
        instrument: str = "BTC-USDT",
        timeframe: str  = "1d",
    ) -> List[TrainingResult]:
        """Run a full walk-forward training loop.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame with a ``target`` column.
        model_class : type
            One of LSTMSignal, TransformerSignal, XGBoostSignal, EnsembleSignal.
        model_kwargs : dict
            Constructor keyword arguments for model_class.
        instrument : str
        timeframe : str

        Returns
        -------
        List[TrainingResult]  one per walk-forward step.
        """
        results = []
        n = len(df)

        for tr_s, tr_e, val_s, val_e in self._wf_splits(n):
            train_df = df.iloc[tr_s:tr_e].copy()
            val_df   = df.iloc[val_s:val_e].copy()

            if len(train_df) < 50 or "target" not in train_df.columns:
                continue

            t0    = time.perf_counter()
            model = model_class(**model_kwargs)
            model = self._train_with_early_stopping(model, train_df, val_df)
            elapsed = time.perf_counter() - t0

            # Evaluate on validation
            val_preds  = self._batch_predict(model, val_df)
            val_rets   = val_df["target"].values
            metrics    = model.evaluate(val_preds, val_rets)

            # Persist
            save_path = self._model_path(instrument, timeframe, model.name, tr_e)
            model.save(save_path)

            result = TrainingResult(
                instrument  = instrument,
                timeframe   = timeframe,
                model_name  = model.name,
                train_start = str(df.index[tr_s]),
                train_end   = str(df.index[tr_e - 1]),
                val_start   = str(df.index[val_s]),
                val_end     = str(df.index[min(val_e - 1, n - 1)]),
                val_ic      = metrics.ic,
                val_icir    = metrics.icir,
                best_epoch  = 0,
                train_sec   = elapsed,
                model_path  = str(save_path),
                metrics     = metrics,
            )
            results.append(result)
            self._history.append(result)

        return results

    # ------------------------------------------------------------------

    def _train_with_early_stopping(
        self,
        model: MLSignal,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> MLSignal:
        """Train model with early stopping based on validation MSE.

        For LSTM/Transformer (epoch-based), we re-fit with increasing
        epochs and track val loss.  For XGBoost (tree-based), a single
        fit is sufficient since it has its own regularisation.
        """
        if isinstance(model, (XGBoostSignal, EnsembleSignal)):
            model.fit(train_df)
            return model

        # Epoch-based models: LSTM / Transformer
        best_val_loss = np.inf
        best_weights_path = self.model_dir / "_tmp_best"
        best_weights_path.mkdir(parents=True, exist_ok=True)

        orig_epochs = getattr(model, "epochs", 30)
        patience_counter = 0

        # Train in 1-epoch increments up to max epochs
        for ep in range(1, orig_epochs + 1):
            model.epochs = 1
            model.fit(train_df)

            val_preds = self._batch_predict(model, val_df)
            if "target" in val_df.columns:
                val_rets = val_df["target"].values[:len(val_preds)]
                val_loss = float(np.mean((val_preds - val_rets)**2))
            else:
                break

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                model.save(best_weights_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    # Restore best weights
                    model.load(best_weights_path)
                    break

        model.epochs = orig_epochs
        return model

    # ------------------------------------------------------------------

    def _batch_predict(self, model: MLSignal, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for every bar in df using expanding context."""
        preds = []
        for i in range(len(df)):
            context = df.iloc[: i + 1]
            try:
                p = model.predict(context)
            except Exception:
                p = 0.0
            preds.append(p)
        return np.array(preds, dtype=np.float64)

    # ------------------------------------------------------------------

    def _model_path(
        self, instrument: str, timeframe: str, model_name: str, step: int
    ) -> pathlib.Path:
        p = (
            self.model_dir
            / instrument.replace("/", "_")
            / timeframe
            / model_name
            / f"step_{step:05d}"
        )
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # Random hyperparameter search wrapper
    # ------------------------------------------------------------------

    def random_search(
        self,
        df: pd.DataFrame,
        model_class,
        param_grid: dict,
        n_trials: int = 50,
        instrument: str = "BTC-USDT",
        timeframe: str  = "1d",
    ) -> TrainingResult:
        """Run random hyperparameter search over ``param_grid``.

        Parameters
        ----------
        param_grid : dict
            Mapping of parameter name → list of candidate values.
        n_trials : int
            Number of random configurations to evaluate.

        Returns
        -------
        TrainingResult  for the best configuration found.
        """
        rng = np.random.default_rng(0)
        best_result: Optional[TrainingResult] = None

        n = len(df)
        train_df = df.iloc[: int(n * 0.7)]
        val_df   = df.iloc[int(n * 0.7) :]

        for trial in range(n_trials):
            kwargs = {k: rng.choice(v) for k, v in param_grid.items()}
            try:
                model = model_class(**kwargs)
                model = self._train_with_early_stopping(model, train_df, val_df)
                preds = self._batch_predict(model, val_df)
                rets  = val_df["target"].values
                m     = model.evaluate(preds, rets)

                if best_result is None or m.ic > best_result.val_ic:
                    save_path = self._model_path(instrument, timeframe,
                                                  model.name, trial)
                    model.save(save_path)
                    best_result = TrainingResult(
                        instrument=instrument, timeframe=timeframe,
                        model_name=model.name,
                        train_start=str(train_df.index[0]),
                        train_end=str(train_df.index[-1]),
                        val_start=str(val_df.index[0]),
                        val_end=str(val_df.index[-1]),
                        val_ic=m.ic, val_icir=m.icir,
                        best_epoch=0, train_sec=0.0,
                        model_path=str(save_path),
                        metrics=m,
                        extra={"trial": trial, "kwargs": {k: str(v) for k, v in kwargs.items()}},
                    )
            except Exception as exc:
                continue  # skip invalid configurations

        return best_result

    @property
    def history(self) -> List[TrainingResult]:
        return list(self._history)
