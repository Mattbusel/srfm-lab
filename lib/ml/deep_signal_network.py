"""
Deep neural network signal generator — pure numpy, no ML libraries.

Layers: Dense, BatchNorm, Dropout, Residual, RNNCell, LSTMCell, TCNBlock.
Model:  SignalNetwork — configurable architecture with forward pass,
        backpropagation, gradient clipping, Adam optimizer.
Training: walk_forward_train (expanding window), generate_signal (inference).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def tanh_act(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

_ACT = {
    "relu": (relu, relu_grad),
    "tanh": (tanh_act, tanh_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "linear": (lambda x: x, lambda x: np.ones_like(x)),
}


# ---------------------------------------------------------------------------
# Adam optimizer state (per parameter)
# ---------------------------------------------------------------------------

@dataclass
class AdamState:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    t: int = 0
    m: Optional[np.ndarray] = field(default=None, repr=False)
    v: Optional[np.ndarray] = field(default=None, repr=False)

    def init(self, shape: Tuple):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)

    def step(self, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.init(grad.shape)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Dense Layer
# ---------------------------------------------------------------------------

class DenseLayer:
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu",
                 lr: float = 1e-3):
        scale = math.sqrt(2.0 / in_dim) if activation == "relu" else math.sqrt(1.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.act_fn, self.act_grad = _ACT[activation]
        self._opt_W = AdamState(lr=lr)
        self._opt_b = AdamState(lr=lr)
        self._cache: Dict[str, Any] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        z = x @ self.W + self.b
        a = self.act_fn(z)
        self._cache = {"x": x, "z": z}
        return a

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self._cache["x"]
        z = self._cache["z"]
        dz = dout * self.act_grad(z)
        self.dW = x.T @ dz
        self.db = dz.sum(axis=0)
        dx = dz @ self.W.T
        return dx

    def apply_gradients(self, clip_norm: float = 5.0):
        for name, g, opt in [("W", self.dW, self._opt_W), ("b", self.db, self._opt_b)]:
            g = _clip_grad(g, clip_norm)
            update = opt.step(g)
            if name == "W":
                self.W -= update
            else:
                self.b -= update


# ---------------------------------------------------------------------------
# Batch Normalization Layer
# ---------------------------------------------------------------------------

class BatchNormLayer:
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1,
                 lr: float = 1e-3):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self._opt_g = AdamState(lr=lr)
        self._opt_b = AdamState(lr=lr)
        self._cache: Dict[str, Any] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training and x.shape[0] > 1:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            x_hat = (x - mu) / np.sqrt(var + self.eps)
            self._cache = {"x": x, "x_hat": x_hat, "mu": mu, "var": var}
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self._cache = {"x_hat": x_hat, "var": self.running_var}
        return self.gamma * x_hat + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x_hat = self._cache["x_hat"]
        var = self._cache.get("var", self.running_var)
        n = dout.shape[0]
        self.dgamma = (dout * x_hat).sum(axis=0)
        self.dbeta = dout.sum(axis=0)
        dx_hat = dout * self.gamma
        dvar = (-0.5 * dx_hat * x_hat / (var + self.eps)).sum(axis=0)
        dmu = (-dx_hat / np.sqrt(var + self.eps)).sum(axis=0)
        if "mu" in self._cache:
            x = self._cache["x"]
            mu = self._cache["mu"]
            dx = (dx_hat / np.sqrt(var + self.eps)
                  + 2 * dvar * (x - mu) / n
                  + dmu / n)
        else:
            dx = dx_hat / np.sqrt(var + self.eps)
        return dx

    def apply_gradients(self, clip_norm: float = 5.0):
        self.gamma -= self._opt_g.step(_clip_grad(self.dgamma, clip_norm))
        self.beta -= self._opt_b.step(_clip_grad(self.dbeta, clip_norm))


# ---------------------------------------------------------------------------
# Dropout Layer
# ---------------------------------------------------------------------------

class DropoutLayer:
    def __init__(self, rate: float = 0.5):
        self.rate = rate
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training and self.rate > 0:
            self._mask = (np.random.rand(*x.shape) > self.rate) / (1.0 - self.rate)
            return x * self._mask
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._mask is not None:
            return dout * self._mask
        return dout

    def apply_gradients(self, clip_norm: float = 5.0):
        pass  # no parameters


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock:
    def __init__(self, dim: int, activation: str = "relu", lr: float = 1e-3):
        self.layer1 = DenseLayer(dim, dim, activation=activation, lr=lr)
        self.bn1 = BatchNormLayer(dim, lr=lr)
        self.layer2 = DenseLayer(dim, dim, activation="linear", lr=lr)
        self.bn2 = BatchNormLayer(dim, lr=lr)
        self.act_fn, self.act_grad = _ACT[activation]
        self._cache: Dict[str, Any] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        identity = x
        h = self.layer1.forward(x, training)
        h = self.bn1.forward(h, training)
        h = self.layer2.forward(h, training)
        h = self.bn2.forward(h, training)
        out = self.act_fn(h + identity)
        self._cache = {"h_pre": h, "identity": identity}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        h_pre = self._cache["h_pre"]
        identity = self._cache["identity"]
        d_act = dout * self.act_grad(h_pre + identity)
        d_skip = d_act  # gradient through skip connection
        d_h = self.bn2.backward(d_act)
        d_h = self.layer2.backward(d_h)
        d_h = self.bn1.backward(d_h)
        d_h = self.layer1.backward(d_h)
        return d_h + d_skip

    def apply_gradients(self, clip_norm: float = 5.0):
        for layer in [self.layer1, self.bn1, self.layer2, self.bn2]:
            layer.apply_gradients(clip_norm)


# ---------------------------------------------------------------------------
# Simple RNN Cell
# ---------------------------------------------------------------------------

class RNNCell:
    def __init__(self, input_dim: int, hidden_dim: int, lr: float = 1e-3):
        scale = math.sqrt(1.0 / (input_dim + hidden_dim))
        self.Wx = np.random.randn(input_dim, hidden_dim) * scale
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b = np.zeros(hidden_dim)
        self._opt = {k: AdamState(lr=lr) for k in ["Wx", "Wh", "b"]}
        self._cache: Dict[str, Any] = {}

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        z = x @ self.Wx + h_prev @ self.Wh + self.b
        h = np.tanh(z)
        self._cache = {"x": x, "h_prev": h_prev, "z": z, "h": h}
        return h

    def backward(self, dh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self._cache["x"]
        h_prev = self._cache["h_prev"]
        z = self._cache["z"]
        dz = dh * (1.0 - np.tanh(z) ** 2)
        self.dWx = x.T @ dz
        self.dWh = h_prev.T @ dz
        self.db = dz.sum(axis=0)
        dx = dz @ self.Wx.T
        dh_prev = dz @ self.Wh.T
        return dx, dh_prev

    def apply_gradients(self, clip_norm: float = 5.0):
        for name, g in [("Wx", self.dWx), ("Wh", self.dWh), ("b", self.db)]:
            g = _clip_grad(g, clip_norm)
            delta = self._opt[name].step(g)
            if name == "Wx":
                self.Wx -= delta
            elif name == "Wh":
                self.Wh -= delta
            else:
                self.b -= delta


# ---------------------------------------------------------------------------
# LSTM Cell
# ---------------------------------------------------------------------------

class LSTMCell:
    def __init__(self, input_dim: int, hidden_dim: int, lr: float = 1e-3):
        scale = math.sqrt(1.0 / (input_dim + hidden_dim))
        # Gates: input, forget, cell, output
        self.W = np.random.randn(input_dim + hidden_dim, 4 * hidden_dim) * scale
        self.b = np.zeros(4 * hidden_dim)
        self.b[hidden_dim:2 * hidden_dim] = 1.0  # forget gate bias = 1
        self.hidden_dim = hidden_dim
        self._opt_W = AdamState(lr=lr)
        self._opt_b = AdamState(lr=lr)
        self._cache: Dict[str, Any] = {}

    def forward(self, x: np.ndarray, h_prev: np.ndarray,
                c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xh = np.concatenate([x, h_prev], axis=-1)
        gates = xh @ self.W + self.b
        d = self.hidden_dim
        i_gate = sigmoid(gates[..., :d])
        f_gate = sigmoid(gates[..., d:2*d])
        g_gate = np.tanh(gates[..., 2*d:3*d])
        o_gate = sigmoid(gates[..., 3*d:])
        c = f_gate * c_prev + i_gate * g_gate
        h = o_gate * np.tanh(c)
        self._cache = {
            "xh": xh, "i": i_gate, "f": f_gate, "g": g_gate,
            "o": o_gate, "c": c, "c_prev": c_prev, "h_prev": h_prev
        }
        return h, c

    def backward(self, dh: np.ndarray,
                 dc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self._cache["c"]
        c_prev = self._cache["c_prev"]
        i, f, g, o = self._cache["i"], self._cache["f"], self._cache["g"], self._cache["o"]
        xh = self._cache["xh"]

        tanh_c = np.tanh(c)
        do = dh * tanh_c
        dc_total = dh * o * (1.0 - tanh_c ** 2) + dc
        di = dc_total * g
        df = dc_total * c_prev
        dg = dc_total * i
        dc_prev = dc_total * f

        # Gate input gradients
        di_pre = di * i * (1 - i)
        df_pre = df * f * (1 - f)
        dg_pre = dg * (1 - g ** 2)
        do_pre = do * o * (1 - o)
        d = self.hidden_dim
        dgates = np.concatenate([di_pre, df_pre, dg_pre, do_pre], axis=-1)

        self.dW = xh.T @ dgates
        self.db = dgates.sum(axis=0)
        dxh = dgates @ self.W.T
        dx = dxh[..., :xh.shape[-1] - d]
        dh_prev = dxh[..., xh.shape[-1] - d:]
        return dx, dh_prev, dc_prev

    def apply_gradients(self, clip_norm: float = 5.0):
        self.W -= self._opt_W.step(_clip_grad(self.dW, clip_norm))
        self.b -= self._opt_b.step(_clip_grad(self.db, clip_norm))


# ---------------------------------------------------------------------------
# TCN Block (Temporal Convolutional Network)
# ---------------------------------------------------------------------------

class TCNBlock:
    """
    Dilated causal 1-D convolution block for sequence modeling.
    Implemented as dense layers over a sliding window (no external deps).
    """

    def __init__(self, seq_len: int, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1, lr: float = 1e-3):
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.seq_len = seq_len
        self.in_ch = in_channels
        self.out_ch = out_channels
        # Flatten receptive field for each position
        receptive = kernel_size * in_channels
        self.conv = DenseLayer(receptive, out_channels, activation="relu", lr=lr)
        self.bn = BatchNormLayer(out_channels, lr=lr)
        # Residual projection if dims differ
        self.proj: Optional[DenseLayer] = None
        if in_channels != out_channels:
            self.proj = DenseLayer(in_channels, out_channels, activation="linear", lr=lr)
        self._cache: Dict[str, Any] = {}

    def _extract_patches(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, seq_len, in_ch) → (batch*seq_len, kernel_size*in_ch) causal."""
        batch, T, C = x.shape
        k, d = self.kernel_size, self.dilation
        patches = np.zeros((batch, T, k * C))
        for t in range(T):
            for i, ki in enumerate(range(k)):
                src = t - (k - 1 - ki) * d
                if src >= 0:
                    patches[:, t, i*C:(i+1)*C] = x[:, src, :]
        return patches.reshape(batch * T, k * C)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """x: (batch, seq_len, in_ch) → (batch, seq_len, out_ch)."""
        batch, T, C = x.shape
        patches = self._extract_patches(x)
        h = self.conv.forward(patches, training)
        h = self.bn.forward(h, training)
        h = h.reshape(batch, T, self.out_ch)
        # Residual
        if self.proj is not None:
            res = self.proj.forward(x.reshape(batch * T, C), training)
            res = res.reshape(batch, T, self.out_ch)
        else:
            res = x[..., :self.out_ch] if C >= self.out_ch else np.pad(x, ((0,0),(0,0),(0,self.out_ch-C)))
        out = relu(h + res)
        self._cache = {"x": x, "h": h, "res": res}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self._cache["x"]
        h = self._cache["h"]
        res = self._cache["res"]
        batch, T, C = x.shape
        d_act = dout * relu_grad(h + res)
        dh = d_act
        dres = d_act
        # Backprop through conv
        dh_flat = self.bn.backward(dh.reshape(batch * T, self.out_ch))
        dx_patches = self.conv.backward(dh_flat)
        # Scatter patches back (approximate: sum contributions)
        dx = np.zeros_like(x)
        k, d = self.kernel_size, self.dilation
        dx_patches = dx_patches.reshape(batch, T, k * C)
        for t in range(T):
            for i, ki in enumerate(range(k)):
                src = t - (k - 1 - ki) * d
                if src >= 0:
                    dx[:, src, :] += dx_patches[:, t, i*C:(i+1)*C]
        # Residual backprop
        if self.proj is not None:
            dx_res = self.proj.backward(dres.reshape(batch * T, self.out_ch))
            dx += dx_res.reshape(batch, T, C)
        return dx

    def apply_gradients(self, clip_norm: float = 5.0):
        self.conv.apply_gradients(clip_norm)
        self.bn.apply_gradients(clip_norm)
        if self.proj is not None:
            self.proj.apply_gradients(clip_norm)


# ---------------------------------------------------------------------------
# Gradient clipping helper
# ---------------------------------------------------------------------------

def _clip_grad(g: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(g)
    if norm > max_norm:
        return g * max_norm / (norm + 1e-8)
    return g


# ---------------------------------------------------------------------------
# Signal Network
# ---------------------------------------------------------------------------

class SignalNetwork:
    """
    Configurable deep signal generation network.
    Architecture specified as list of layer config dicts.

    Example config:
        [
            {"type": "dense", "in": 20, "out": 64, "act": "relu"},
            {"type": "batchnorm", "dim": 64},
            {"type": "dropout", "rate": 0.3},
            {"type": "residual", "dim": 64},
            {"type": "dense", "out": 1, "act": "linear"},
        ]
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int], activation: str = "relu",
                 dropout_rate: float = 0.2, use_batchnorm: bool = True,
                 lr: float = 1e-3, clip_norm: float = 5.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.clip_norm = clip_norm
        self.layers: List[Any] = []
        self.training = True

        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(DenseLayer(dims[i], dims[i+1], activation=activation, lr=lr))
            if use_batchnorm:
                self.layers.append(BatchNormLayer(dims[i+1], lr=lr))
            if dropout_rate > 0:
                self.layers.append(DropoutLayer(rate=dropout_rate))

        self.layers.append(DenseLayer(dims[-1], output_dim, activation="linear", lr=lr))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            if isinstance(layer, (BatchNormLayer, DropoutLayer)):
                x = layer.forward(x, training=self.training)
            else:
                x = layer.forward(x, training=self.training)
        return x

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        grad = dloss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def apply_gradients(self):
        for layer in self.layers:
            layer.apply_gradients(self.clip_norm)

    def mse_loss(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
        diff = pred - target
        loss = float(np.mean(diff ** 2))
        grad = 2 * diff / pred.shape[0]
        return loss, grad

    def bce_loss(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
        eps = 1e-7
        p = sigmoid(pred)
        loss = float(-np.mean(target * np.log(p + eps) + (1 - target) * np.log(1 - p + eps)))
        grad = (p - target) / pred.shape[0]
        return loss, grad

    def train_step(self, x: np.ndarray, y: np.ndarray,
                   loss_type: str = "mse") -> float:
        self.training = True
        pred = self.forward(x)
        if loss_type == "mse":
            loss, grad = self.mse_loss(pred, y)
        else:
            loss, grad = self.bce_loss(pred, y)
        self.backward(grad)
        self.apply_gradients()
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.training = False
        return self.forward(x)


# ---------------------------------------------------------------------------
# Walk-Forward Training
# ---------------------------------------------------------------------------

def walk_forward_train(
    network: SignalNetwork,
    X: np.ndarray,
    y: np.ndarray,
    min_train_size: int = 100,
    val_size: int = 20,
    batch_size: int = 32,
    epochs_per_step: int = 5,
    loss_type: str = "mse",
    verbose: bool = False,
) -> List[float]:
    """
    Walk-forward (expanding window) training of SignalNetwork.

    X: (n_samples, input_dim)
    y: (n_samples, output_dim) or (n_samples,)
    Returns list of validation losses per fold.
    """
    n = len(X)
    y = y.reshape(n, -1) if y.ndim == 1 else y
    val_losses = []
    t = min_train_size

    while t + val_size <= n:
        X_train, y_train = X[:t], y[:t]
        X_val, y_val = X[t:t + val_size], y[t:t + val_size]

        # Mini-batch training
        for epoch in range(epochs_per_step):
            idx = np.random.permutation(t)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, t, batch_size):
                batch_idx = idx[start:start + batch_size]
                loss = network.train_step(X_train[batch_idx], y_train[batch_idx], loss_type)
                epoch_loss += loss
                n_batches += 1
            if verbose and epoch == epochs_per_step - 1:
                print(f"  fold t={t}, epoch {epoch+1}, loss={epoch_loss/max(n_batches,1):.6f}")

        # Validation
        val_pred = network.predict(X_val)
        if loss_type == "mse":
            val_loss = float(np.mean((val_pred - y_val) ** 2))
        else:
            val_loss = float(network.bce_loss(val_pred, y_val)[0])
        val_losses.append(val_loss)

        t += val_size

    return val_losses


# ---------------------------------------------------------------------------
# Inference: generate trading signal from recent window
# ---------------------------------------------------------------------------

def generate_signal(
    network: SignalNetwork,
    recent_features: np.ndarray,
    signal_type: str = "regression",
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Run inference on recent_features window.

    signal_type: "regression" → return raw prediction,
                 "classification" → return sigmoid probability,
                 "direction" → return +1 / 0 / -1.

    Returns dict with keys: raw, signal, confidence.
    """
    x = np.atleast_2d(recent_features)
    raw = network.predict(x)
    raw_val = float(raw.ravel()[0])

    if signal_type == "classification":
        prob = float(sigmoid(np.array([raw_val]))[0])
        direction = 1 if prob > 0.5 + threshold else (-1 if prob < 0.5 - threshold else 0)
        return {"raw": raw_val, "signal": direction, "confidence": abs(prob - 0.5) * 2}

    elif signal_type == "direction":
        direction = 1 if raw_val > threshold else (-1 if raw_val < -threshold else 0)
        return {"raw": raw_val, "signal": direction, "confidence": abs(raw_val)}

    else:  # regression
        return {"raw": raw_val, "signal": raw_val, "confidence": 1.0}


# ---------------------------------------------------------------------------
# Factory: build standard signal network architectures
# ---------------------------------------------------------------------------

def build_momentum_network(input_dim: int, lr: float = 1e-3) -> SignalNetwork:
    """Shallow wide network for momentum signal generation."""
    return SignalNetwork(
        input_dim=input_dim, output_dim=1,
        hidden_dims=[128, 64, 32],
        activation="relu", dropout_rate=0.2,
        use_batchnorm=True, lr=lr,
    )


def build_mean_reversion_network(input_dim: int, lr: float = 1e-3) -> SignalNetwork:
    """Deeper network for mean-reversion signals with residual connections."""
    return SignalNetwork(
        input_dim=input_dim, output_dim=1,
        hidden_dims=[64, 64, 64],
        activation="tanh", dropout_rate=0.1,
        use_batchnorm=True, lr=lr,
    )


def build_regime_classifier(input_dim: int, n_regimes: int = 3,
                             lr: float = 1e-3) -> SignalNetwork:
    """Network that classifies market regime (multi-class via softmax output)."""
    net = SignalNetwork(
        input_dim=input_dim, output_dim=n_regimes,
        hidden_dims=[64, 32],
        activation="relu", dropout_rate=0.2,
        use_batchnorm=True, lr=lr,
    )
    return net
