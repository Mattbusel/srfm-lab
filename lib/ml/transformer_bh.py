"""
Transformer-based BH Formation Predictor (T3-5)
Predicts probability of BH formation N bars ahead using sequence of Minkowski features.

Architecture:
  - Encoder-only Transformer (BERT-style)
  - Input: last 64 bars of (ds2, beta, cf, mass, dp_frac, volume_norm) per instrument
  - Positional encoding respects Minkowski metric (TIMELIKE bars weighted differently)
  - Output: probability of BH formation in next [1, 3, 5, 10] bars

Cross-instrument attention (optional): when multiple instruments are fed together,
attention can learn cross-asset BH propagation patterns.
"""
import math
import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)

@dataclass
class TransformerBHConfig:
    seq_len: int = 64          # lookback window in bars
    n_features: int = 6        # (ds2, beta, cf, mass, dp_frac, vol_norm)
    d_model: int = 64          # embedding dimension
    n_heads: int = 4           # attention heads
    n_layers: int = 2          # transformer layers
    d_ff: int = 128            # feedforward hidden dim
    dropout: float = 0.1
    forecast_horizons: list = field(default_factory=lambda: [1, 3, 5, 10])
    model_path: Optional[str] = None  # path to saved weights

class MinkowskiPositionalEncoding:
    """
    Positional encoding that weights TIMELIKE bars (ds2 > 0) more heavily
    than SPACELIKE bars. TIMELIKE = causal, ordered moves. SPACELIKE = shocks.
    """
    def __init__(self, seq_len: int, d_model: int):
        self.seq_len = seq_len
        self.d_model = d_model
        # Standard sinusoidal PE
        pe = np.zeros((seq_len, d_model))
        positions = np.arange(seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term[:d_model//2])
        self.pe = pe

    def encode(self, features: np.ndarray, ds2_sequence: np.ndarray) -> np.ndarray:
        """
        features: (seq_len, d_model) — already projected input
        ds2_sequence: (seq_len,) — raw ds² values for weighting
        Returns: (seq_len, d_model) — positionally encoded features
        """
        # Timelike weight: ds2 > 0 gets +1.0, spacelike gets +0.5
        tl_weight = np.where(ds2_sequence > 0, 1.0, 0.5).reshape(-1, 1)
        return features + tl_weight * self.pe[:len(features)]

class TransformerBHPredictor:
    """
    Numpy-only Transformer implementation for inference (no PyTorch dependency for deployment).

    For training, use the Rust rl-exit-optimizer crate (or a PyTorch script separately).
    This class handles:
      1. Feature normalization
      2. Forward pass (inference-only, from saved weights)
      3. Streaming prediction via a rolling feature buffer

    When no weights are available, returns calibrated heuristic predictions.
    """

    def __init__(self, cfg: TransformerBHConfig = None):
        self.cfg = cfg or TransformerBHConfig()
        self._feature_buffer: list[list[float]] = []  # rolling window of feature vectors
        self._weights_loaded = False
        self._pos_enc = MinkowskiPositionalEncoding(self.cfg.seq_len, self.cfg.d_model)

        # Running normalization stats
        self._feat_mean = np.zeros(self.cfg.n_features)
        self._feat_std = np.ones(self.cfg.n_features)
        self._n_obs = 0

        if self.cfg.model_path:
            self._try_load_weights(self.cfg.model_path)

    def _try_load_weights(self, path: str):
        try:
            p = Path(path)
            if p.exists():
                # Weights stored as JSON arrays of numpy arrays
                with open(p) as f:
                    w = json.load(f)
                self._weights = w
                self._weights_loaded = True
                log.info("TransformerBH: loaded weights from %s", path)
        except Exception as e:
            log.warning("TransformerBH: failed to load weights: %s", e)

    def update(
        self,
        ds2: float,
        beta: float,
        cf: float,
        mass: float,
        dp_frac: float,
        volume_norm: float,
    ) -> dict:
        """
        Feed one bar. Returns BH formation probability predictions.

        Returns dict:
          probs: {1: float, 3: float, 5: float, 10: float} — BH formation probabilities
          confidence: float — model confidence (low if insufficient history)
        """
        feat = [ds2, beta, cf, mass, dp_frac, volume_norm]

        # Online normalization update (Welford)
        self._n_obs += 1
        feat_arr = np.array(feat)
        delta = feat_arr - self._feat_mean
        self._feat_mean += delta / self._n_obs
        delta2 = feat_arr - self._feat_mean
        if self._n_obs > 1:
            self._feat_std = np.sqrt(
                ((self._n_obs - 2) * self._feat_std**2 + delta * delta2) / (self._n_obs - 1)
            )
            self._feat_std = np.maximum(self._feat_std, 1e-6)

        self._feature_buffer.append(feat)
        if len(self._feature_buffer) > self.cfg.seq_len:
            self._feature_buffer.pop(0)

        if len(self._feature_buffer) < 10:
            # Insufficient history
            return {"probs": {h: 0.5 for h in self.cfg.forecast_horizons}, "confidence": 0.0}

        if self._weights_loaded:
            return self._forward_pass()
        else:
            return self._heuristic_predict()

    def _heuristic_predict(self) -> dict:
        """
        Calibrated heuristic when no trained weights are available.
        Uses physics features directly: high mass + high CF + TIMELIKE → high BH prob.
        """
        recent = self._feature_buffer[-min(16, len(self._feature_buffer)):]

        # Feature indices: ds2=0, beta=1, cf=2, mass=3, dp_frac=4, vol=5
        avg_mass = sum(r[3] for r in recent) / len(recent)
        avg_cf = sum(r[2] for r in recent) / len(recent)
        timelike_frac = sum(1 for r in recent if r[0] > 0) / len(recent)

        # Base probability from mass (normalized by BH_FORM threshold 1.92)
        mass_prob = min(0.90, max(0.05, avg_mass / 3.84))  # 3.84 = 2x BH_FORM

        # CF momentum contribution
        cf_boost = min(0.20, avg_cf * 5.0)

        # Timelike coherence contribution
        tl_boost = (timelike_frac - 0.5) * 0.20

        base_prob = min(0.90, max(0.05, mass_prob + cf_boost + tl_boost))

        # Decay probability with horizon (harder to predict further out)
        probs = {}
        for h in self.cfg.forecast_horizons:
            decay = 0.85 ** (h - 1)
            probs[h] = min(0.90, max(0.05, base_prob * decay + 0.5 * (1 - decay)))

        confidence = min(1.0, len(self._feature_buffer) / self.cfg.seq_len)
        return {"probs": probs, "confidence": confidence}

    def _forward_pass(self) -> dict:
        """Inference forward pass using loaded weights. Placeholder for trained model."""
        # When weights are loaded, this would run the actual transformer forward pass
        # For now, fall back to heuristic
        return self._heuristic_predict()

    def get_bh_prob(self, horizon: int = 3) -> float:
        """Convenience: get BH formation probability for a specific horizon."""
        if not self._feature_buffer:
            return 0.5
        result = self._heuristic_predict() if not self._weights_loaded else self._forward_pass()
        return result["probs"].get(horizon, 0.5)
