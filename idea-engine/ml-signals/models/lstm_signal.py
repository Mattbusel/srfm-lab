"""
models/lstm_signal.py
=====================
2-layer LSTM signal model implemented entirely in NumPy.

Financial rationale
-------------------
Crypto returns exhibit short-term momentum / mean-reversion regimes that
change on a scale of 10–30 bars.  An LSTM can implicitly maintain a
"hidden state" representing the current regime context, allowing it to
weight recent price moves differently depending on the sequence of events
that led to them.  This is fundamentally different from a rolling-window
linear model: the cell state acts as a lossy memory that forgets old
information at a learned rate (forget gate).

Architecture
------------
Input:  (seq_len=20, n_features=8)
        [returns_1d, returns_5d, vol_20d, rsi_14,
         bh_mass, bh_active, mayer_multiple, ema_ratio]
Layer 1: LSTM, hidden_size=64
Layer 2: LSTM, hidden_size=64
Output:  linear projection → scalar in (-1, +1) via tanh

Training: Adam, MSE loss, gradient clipping ||g||₂ ≤ 1.0, BPTT.
Online:   fine-tune on latest 30 bars each day (warm-start).
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import MLSignal, SignalMetrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "returns_1d", "returns_5d", "vol_20d", "rsi_14",
    "bh_mass", "bh_active", "mayer_multiple", "ema_ratio",
]
SEQ_LEN      = 20
HIDDEN_SIZE  = 64
N_LAYERS     = 2
LR           = 1e-3
CLIP_NORM    = 1.0
ONLINE_BARS  = 30
EPOCHS       = 30
BATCH_SIZE   = 32


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -20, 20))


# ---------------------------------------------------------------------------
# LSTM layer (single layer, vectorised over batch)
# ---------------------------------------------------------------------------

class _LSTMLayer:
    """Single LSTM layer with full BPTT support.

    Weights layout follows standard convention:
        W_ih  (4H, I)  – input-hidden weights  [i, f, g, o]
        W_hh  (4H, H)  – hidden-hidden weights
        b     (4H,)    – bias
    """

    def __init__(self, input_size: int, hidden_size: int, rng: np.random.Generator) -> None:
        H, I = hidden_size, input_size
        scale_ih = np.sqrt(2.0 / (I + H))
        scale_hh = np.sqrt(1.0 / H)
        self.W_ih = rng.normal(0, scale_ih, (4 * H, I))
        self.W_hh = rng.normal(0, scale_hh, (4 * H, H))
        self.b    = np.zeros(4 * H)
        self.H    = H

        # Adam state
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0

    def _params(self) -> Dict[str, np.ndarray]:
        return {"W_ih": self.W_ih, "W_hh": self.W_hh, "b": self.b}

    def forward(
        self, X: np.ndarray, h0: Optional[np.ndarray] = None, c0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Forward pass over a sequence.

        Parameters
        ----------
        X  : (T, B, I)
        h0 : (B, H) initial hidden state, zeros if None
        c0 : (B, H) initial cell state, zeros if None

        Returns
        -------
        outputs : (T, B, H)
        h_n     : (B, H) final hidden state
        c_n     : (B, H) final cell state
        cache   : list of per-step cache dicts for BPTT
        """
        T, B, _ = X.shape
        H = self.H
        h = np.zeros((B, H)) if h0 is None else h0.copy()
        c = np.zeros((B, H)) if c0 is None else c0.copy()

        outputs = np.zeros((T, B, H))
        cache: list = []

        for t in range(T):
            x_t = X[t]                                      # (B, I)
            gates = x_t @ self.W_ih.T + h @ self.W_hh.T + self.b  # (B, 4H)
            i_g = _sigmoid(gates[:, :H])
            f_g = _sigmoid(gates[:, H:2*H])
            g   = _tanh(gates[:, 2*H:3*H])
            o_g = _sigmoid(gates[:, 3*H:])
            c_new = f_g * c + i_g * g
            h_new = o_g * _tanh(c_new)
            cache.append(dict(x=x_t, h=h, c=c, i=i_g, f=f_g, g=g, o=o_g,
                              c_new=c_new, h_new=h_new, gates=gates))
            h, c = h_new, c_new
            outputs[t] = h

        return outputs, h, c, cache

    def backward(
        self, doutputs: np.ndarray, cache: list
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """BPTT over the stored cache.

        Parameters
        ----------
        doutputs : (T, B, H) upstream gradient w.r.t. outputs

        Returns
        -------
        dX   : (T, B, I) gradient w.r.t. inputs
        grads: dict of parameter gradients
        """
        T, B, I_size = doutputs.shape[0], doutputs.shape[1], self.W_ih.shape[1]
        H = self.H

        dW_ih = np.zeros_like(self.W_ih)
        dW_hh = np.zeros_like(self.W_hh)
        db    = np.zeros_like(self.b)
        dX    = np.zeros((T, B, I_size))

        dh_next = np.zeros((B, H))
        dc_next = np.zeros((B, H))

        for t in reversed(range(T)):
            ca = cache[t]
            dh = doutputs[t] + dh_next                      # (B, H)
            # o gate
            dtanh_cn = dh * ca["o"]
            dc = dtanh_cn * (1.0 - _tanh(ca["c_new"])**2) + dc_next
            # gate gradients
            di = dc * ca["g"]
            df = dc * ca["c"]
            dg = dc * ca["i"]
            do = dh * _tanh(ca["c_new"])
            # pre-activation gradients
            di_pre = di * ca["i"] * (1.0 - ca["i"])
            df_pre = df * ca["f"] * (1.0 - ca["f"])
            dg_pre = dg * (1.0 - ca["g"]**2)
            do_pre = do * ca["o"] * (1.0 - ca["o"])
            dgates = np.concatenate([di_pre, df_pre, dg_pre, do_pre], axis=1)  # (B, 4H)
            dW_ih += dgates.T @ ca["x"]
            dW_hh += dgates.T @ ca["h"]
            db    += dgates.sum(axis=0)
            dX[t]  = dgates @ self.W_ih
            dh_next = dgates @ self.W_hh
            dc_next = dc * ca["f"]

        return dX, {"W_ih": dW_ih, "W_hh": dW_hh, "b": db}

    def adam_step(self, grads: Dict[str, np.ndarray], lr: float,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self._t += 1
        for k, p in self._params().items():
            g = grads[k]
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * g**2
            m_hat = self._m[k] / (1 - beta1**self._t)
            v_hat = self._v[k] / (1 - beta2**self._t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ---------------------------------------------------------------------------
# Linear output head
# ---------------------------------------------------------------------------

class _LinearHead:
    def __init__(self, in_size: int, rng: np.random.Generator) -> None:
        self.W = rng.normal(0, 0.01, (1, in_size))
        self.b = np.zeros(1)
        self._m_W = np.zeros_like(self.W)
        self._v_W = np.zeros_like(self.W)
        self._m_b = np.zeros_like(self.b)
        self._v_b = np.zeros_like(self.b)
        self._t   = 0

    def forward(self, h: np.ndarray) -> np.ndarray:
        return np.tanh(h @ self.W.T + self.b).squeeze(-1)   # (B,)

    def backward(self, h: np.ndarray, dy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_raw = h @ self.W.T + self.b                        # (B, 1)
        dtanh = (1.0 - np.tanh(y_raw)**2) * dy[:, None]
        dW = dtanh.T @ h
        db = dtanh.sum(axis=0)
        dh = dtanh @ self.W
        return dh, dW, db

    def adam_step(self, dW: np.ndarray, db: np.ndarray, lr: float,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self._t += 1
        for (m, v, p, g) in [
            (self._m_W, self._v_W, self.W, dW),
            (self._m_b, self._v_b, self.b, db),
        ]:
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**self._t)
            v_hat = v / (1 - beta2**self._t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ---------------------------------------------------------------------------
# Public LSTM signal
# ---------------------------------------------------------------------------

class LSTMSignal(MLSignal):
    """2-layer LSTM for crypto return prediction (NumPy-only).

    See module docstring for full description.
    """

    def __init__(
        self,
        feature_cols: List[str] = FEATURE_COLS,
        seq_len: int = SEQ_LEN,
        hidden_size: int = HIDDEN_SIZE,
        n_layers: int = N_LAYERS,
        lr: float = LR,
        clip_norm: float = CLIP_NORM,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        seed: int = 42,
    ) -> None:
        super().__init__(name="LSTMSignal")
        self.feature_cols = feature_cols
        self.seq_len      = seq_len
        self.hidden_size  = hidden_size
        self.n_layers     = n_layers
        self.lr           = lr
        self.clip_norm    = clip_norm
        self.epochs       = epochs
        self.batch_size   = batch_size
        self._rng         = np.random.default_rng(seed)
        self._layers: List[_LSTMLayer] = []
        self._head: Optional[_LinearHead] = None
        self._loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------

    def _make_sequences(
        self, df: pd.DataFrame, target_col: str = "target"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X, y) sequence arrays from a feature DataFrame."""
        cols = [c for c in self.feature_cols if c in df.columns]
        data = df[cols].values.astype(np.float32)
        y    = df[target_col].values.astype(np.float32) if target_col in df.columns else None

        X_list, y_list = [], []
        for i in range(self.seq_len, len(data)):
            X_list.append(data[i - self.seq_len : i])
            if y is not None:
                y_list.append(y[i])

        X_arr = np.stack(X_list, axis=0)                    # (N, seq_len, F)
        y_arr = np.array(y_list, dtype=np.float32) if y_list else np.zeros(len(X_arr))
        return X_arr, y_arr

    # ------------------------------------------------------------------
    # Gradient clipping
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_grads(grads_list: List[Dict[str, np.ndarray]], max_norm: float) -> None:
        total_norm = 0.0
        for grads in grads_list:
            for g in grads.values():
                total_norm += float(np.sum(g**2))
        total_norm = np.sqrt(total_norm)
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for grads in grads_list:
                for k in grads:
                    grads[k] *= scale

    # ------------------------------------------------------------------
    # MLSignal interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "LSTMSignal":
        """Train the LSTM on the full history in ``df``."""
        n_features = sum(1 for c in self.feature_cols if c in df.columns)
        self._layers = [
            _LSTMLayer(n_features if i == 0 else self.hidden_size,
                       self.hidden_size, self._rng)
            for i in range(self.n_layers)
        ]
        self._head = _LinearHead(self.hidden_size, self._rng)

        X, y = self._make_sequences(df)
        N = len(X)
        self._loss_history.clear()

        for epoch in range(self.epochs):
            idx = self._rng.permutation(N)
            epoch_losses = []
            for start in range(0, N, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                Xb = X[batch_idx].transpose(1, 0, 2)        # (T, B, F)
                yb = y[batch_idx]

                # ---- forward ----
                inp = Xb
                caches = []
                for layer in self._layers:
                    out, _, _, cache = layer.forward(inp)
                    caches.append(cache)
                    inp = out
                # Use last time-step hidden state for prediction
                h_last = inp[-1]                             # (B, H)
                y_pred = self._head.forward(h_last)          # (B,)

                # MSE loss
                diff   = y_pred - yb
                loss   = float(np.mean(diff**2))
                epoch_losses.append(loss)

                # ---- backward ----
                # Head gradient
                dy = 2.0 * diff / len(yb)
                dh_last, dW_head, db_head = self._head.backward(h_last, dy)

                # Only last time-step needs gradient from head
                doutputs = np.zeros_like(inp)
                doutputs[-1] = dh_last

                layer_grads_list = []
                for layer, cache in zip(reversed(self._layers), reversed(caches)):
                    dX, lg = layer.backward(doutputs, cache)
                    layer_grads_list.append(lg)
                    doutputs = dX

                # Gradient clipping
                all_grads = layer_grads_list + [{"W": dW_head, "b": db_head}]
                self._clip_grads(all_grads, self.clip_norm)

                # Adam updates
                for layer, lg in zip(reversed(self._layers), layer_grads_list):
                    layer.adam_step(lg, self.lr)
                self._head.adam_step(dW_head, db_head, self.lr)

            self._loss_history.append(float(np.mean(epoch_losses)))

        self._is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> float:
        """Return signal score for the most recent bar."""
        self._check_fitted()
        X, _ = self._make_sequences(df)
        if len(X) == 0:
            return 0.0
        # Use last sequence
        Xb = X[[-1]].transpose(1, 0, 2)                     # (T=20, B=1, F)
        inp = Xb
        for layer in self._layers:
            out, _, _, _ = layer.forward(inp)
            inp = out
        h_last = inp[-1]
        return float(self._head.forward(h_last)[0])

    def online_update(self, df: pd.DataFrame) -> None:
        """Fine-tune on the latest ``ONLINE_BARS`` rows."""
        self._check_fitted()
        tail = df.iloc[-ONLINE_BARS - self.seq_len :]
        orig_epochs = self.epochs
        self.epochs  = 5
        self.fit(tail)
        self.epochs  = orig_epochs

    def save(self, path: pathlib.Path) -> None:
        self._check_fitted()
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        for i, layer in enumerate(self._layers):
            arrays[f"l{i}_W_ih"] = layer.W_ih
            arrays[f"l{i}_W_hh"] = layer.W_hh
            arrays[f"l{i}_b"]    = layer.b
        arrays["head_W"] = self._head.W
        arrays["head_b"] = self._head.b
        np.savez(path / "lstm_weights.npz", **arrays)

    def load(self, path: pathlib.Path) -> "LSTMSignal":
        path = pathlib.Path(path)
        data = np.load(path / "lstm_weights.npz")
        n_features = data["l0_W_ih"].shape[1]
        self._layers = []
        for i in range(self.n_layers):
            layer = _LSTMLayer.__new__(_LSTMLayer)
            layer.H    = self.hidden_size
            layer.W_ih = data[f"l{i}_W_ih"]
            layer.W_hh = data[f"l{i}_W_hh"]
            layer.b    = data[f"l{i}_b"]
            layer._m   = {k: np.zeros_like(v) for k, v in layer._params().items()}
            layer._v   = {k: np.zeros_like(v) for k, v in layer._params().items()}
            layer._t   = 0
            self._layers.append(layer)
        self._head = _LinearHead.__new__(_LinearHead)
        self._head.W    = data["head_W"]
        self._head.b    = data["head_b"]
        self._head._m_W = np.zeros_like(self._head.W)
        self._head._v_W = np.zeros_like(self._head.W)
        self._head._m_b = np.zeros_like(self._head.b)
        self._head._v_b = np.zeros_like(self._head.b)
        self._head._t   = 0
        self._is_fitted = True
        return self

    def feature_importance(self) -> Dict[str, float]:
        """Approximate importance via input-weight L2 norms (layer 0)."""
        self._check_fitted()
        W = self._layers[0].W_ih                             # (4H, F)
        importances = np.linalg.norm(W, axis=0)             # (F,)
        importances /= importances.sum() + 1e-9
        cols = [c for c in self.feature_cols][:importances.shape[0]]
        return dict(zip(cols, importances.tolist()))
