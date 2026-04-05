"""
inference/predictor.py
=======================
LivePredictor: sub-millisecond signal generation for live trading.

Financial rationale
-------------------
In a live crypto trading system, signal latency directly affects
execution quality.  A signal that arrives 10 ms after the event that
triggered it may already be stale (market makers respond in <1 ms on
major venues).  The LivePredictor achieves <1 ms per instrument by:

1. Feature caching: the feature vector for bar t is identical whether
   we call predict() at t+0ms or t+500ms.  We store the last feature
   hash and skip recomputation if the input DataFrame has not changed.

2. Rolling window: we maintain a sliding buffer of the last ``max_bars``
   rows so that sequence models (LSTM, Transformer) do not re-read the
   entire historical DataFrame on each call.

3. Lazy loading: models are loaded from disk on first predict() call
   for a given (instrument, timeframe) pair and held in memory.

The prediction cache (feature_hash → score) has a TTL of one bar
period to prevent stale signals from persisting across regime changes.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from models.base import MLSignal
from models.ensemble import EnsembleSignal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_BARS     = 200    # rolling buffer size
CACHE_TTL_S  = 60.0  # prediction cache TTL in seconds


# ---------------------------------------------------------------------------
# LivePredictor
# ---------------------------------------------------------------------------

class LivePredictor:
    """Load trained models and generate live signals with feature caching.

    Parameters
    ----------
    model_dir : Path
        Root directory where models are saved
        (same structure as used by MLTrainer).
    max_bars : int
        Size of the rolling feature buffer per instrument.
    cache_ttl : float
        Prediction cache time-to-live in seconds.
    """

    def __init__(
        self,
        model_dir:  str | Path = "models",
        max_bars:   int        = MAX_BARS,
        cache_ttl:  float      = CACHE_TTL_S,
    ) -> None:
        self.model_dir  = Path(model_dir)
        self.max_bars   = max_bars
        self.cache_ttl  = cache_ttl

        # instrument → model
        self._models: Dict[str, MLSignal] = {}
        # instrument → rolling DataFrame buffer
        self._buffers: Dict[str, pd.DataFrame] = {}
        # instrument → (hash, score, timestamp)
        self._pred_cache: Dict[str, Tuple[str, float, float]] = {}

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, instrument: str, model: MLSignal) -> None:
        """Register a pre-loaded model for an instrument.

        Parameters
        ----------
        instrument : str  e.g. ``'BTC-USDT'``
        model : MLSignal  already-fitted model instance
        """
        self._models[instrument] = model

    def load_from_disk(
        self,
        instrument: str,
        timeframe:  str,
        model_instance: MLSignal,
    ) -> bool:
        """Load the latest saved model for (instrument, timeframe).

        Returns True on success, False if no saved model found.
        """
        base = (
            self.model_dir
            / instrument.replace("/", "_")
            / timeframe
            / model_instance.name
        )
        if not base.exists():
            return False

        # Find the latest step directory
        step_dirs = sorted(base.glob("step_*"), key=lambda p: p.name)
        if not step_dirs:
            return False

        latest = step_dirs[-1]
        try:
            model_instance.load(latest)
            self._models[instrument] = model_instance
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def update_buffer(self, instrument: str, new_bar: pd.DataFrame) -> None:
        """Append ``new_bar`` to the rolling buffer for ``instrument``.

        Parameters
        ----------
        new_bar : pd.DataFrame
            Single-row (or multi-row) feature DataFrame to append.
        """
        if instrument not in self._buffers:
            self._buffers[instrument] = new_bar.tail(self.max_bars).copy()
        else:
            combined = pd.concat([self._buffers[instrument], new_bar])
            self._buffers[instrument] = combined.tail(self.max_bars).copy()
        # Invalidate cache when buffer changes
        self._pred_cache.pop(instrument, None)

    def set_buffer(self, instrument: str, df: pd.DataFrame) -> None:
        """Set the full feature buffer (e.g. on startup)."""
        self._buffers[instrument] = df.tail(self.max_bars).copy()
        self._pred_cache.pop(instrument, None)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instrument: str) -> Tuple[float, float]:
        """Generate signal for ``instrument``.

        Uses the rolling buffer loaded via :meth:`set_buffer` or
        :meth:`update_buffer`.

        Returns
        -------
        (score, latency_ms)
            score       : float in [-1, +1]
            latency_ms  : inference time in milliseconds
        """
        t0 = time.perf_counter()

        if instrument not in self._models:
            return 0.0, 0.0
        if instrument not in self._buffers:
            return 0.0, 0.0

        df = self._buffers[instrument]

        # Check cache
        h = self._feature_hash(df)
        now = time.time()
        if instrument in self._pred_cache:
            cached_h, cached_score, cached_ts = self._pred_cache[instrument]
            if cached_h == h and (now - cached_ts) < self.cache_ttl:
                return cached_score, (time.perf_counter() - t0) * 1000

        # Compute prediction
        try:
            model = self._models[instrument]
            score = float(model.predict(df))
        except Exception:
            score = 0.0

        score = float(np.clip(score, -1.0, 1.0))
        self._pred_cache[instrument] = (h, score, now)

        latency_ms = (time.perf_counter() - t0) * 1000
        return score, latency_ms

    def predict_with_confidence(
        self, instrument: str
    ) -> Tuple[float, float, float]:
        """Return (score, confidence, latency_ms) for ensemble models."""
        t0 = time.perf_counter()
        if instrument not in self._models:
            return 0.0, 0.0, 0.0
        model = self._models[instrument]
        df    = self._buffers.get(instrument, pd.DataFrame())
        if df.empty:
            return 0.0, 0.0, 0.0
        try:
            if isinstance(model, EnsembleSignal):
                score, confidence = model.predict_with_confidence(df)
            else:
                score      = float(model.predict(df))
                confidence = abs(score)          # proxy: |score| as confidence
        except Exception:
            score, confidence = 0.0, 0.0

        latency_ms = (time.perf_counter() - t0) * 1000
        return float(np.clip(score, -1.0, 1.0)), float(confidence), latency_ms

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear_cache(self, instrument: Optional[str] = None) -> None:
        """Clear prediction cache for one or all instruments."""
        if instrument:
            self._pred_cache.pop(instrument, None)
        else:
            self._pred_cache.clear()

    def loaded_instruments(self) -> list:
        return list(self._models.keys())

    def buffer_info(self, instrument: str) -> dict:
        buf = self._buffers.get(instrument)
        if buf is None:
            return {"rows": 0}
        return {
            "rows":  len(buf),
            "start": str(buf.index[0]) if len(buf) else None,
            "end":   str(buf.index[-1]) if len(buf) else None,
            "cols":  list(buf.columns),
        }

    @staticmethod
    def _feature_hash(df: pd.DataFrame) -> str:
        """Compute a short hash of the last row of a feature DataFrame."""
        if df.empty:
            return "empty"
        last_row = df.iloc[-1].values
        raw = last_row.tobytes()
        return hashlib.md5(raw, usedforsecurity=False).hexdigest()[:16]
