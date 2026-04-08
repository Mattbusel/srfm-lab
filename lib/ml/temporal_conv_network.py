"""
Temporal Convolutional Network (TCN) for time series forecasting.

Numpy-only implementation with causal/dilated convolutions, residual
blocks, gated activations (WaveNet-style), weight normalization,
and full backpropagation.  Supports sequence-to-one and
sequence-to-sequence prediction for multi-horizon return forecasting.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

# ---------------------------------------------------------------------------
# Causal 1D convolution
# ---------------------------------------------------------------------------

class CausalConv1D:
    """
    Causal 1D convolution: output at time t depends only on inputs at t and earlier.

    Input shape:  (batch, in_channels, seq_len)
    Output shape: (batch, out_channels, seq_len)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, rng: Optional[np.random.RandomState] = None):
        rng = rng or np.random.RandomState(42)
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation  # causal padding (left only)

        std = np.sqrt(2.0 / (in_channels * kernel_size))
        self.W = rng.randn(out_channels, in_channels, kernel_size) * std
        self.b = np.zeros((out_channels, 1))

        # Cache for backprop
        self._input_padded = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch, C_in, L = x.shape
        # Left-pad for causality
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, 0)), mode="constant")
        else:
            x_padded = x
        self._input_padded = x_padded
        L_pad = x_padded.shape[2]

        out = np.zeros((batch, self.out_ch, L))
        for t in range(L):
            for k in range(self.kernel_size):
                idx = t + self.padding - k * self.dilation
                if 0 <= idx < L_pad:
                    # x_padded[:, :, idx] -> (batch, C_in)
                    # W[:, :, k] -> (out_ch, C_in)
                    out[:, :, t] += x_padded[:, :, idx] @ self.W[:, :, k].T
        out += self.b[None, :, :]  # broadcast bias
        return out

    def backward(self, d_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (d_input, dW, db)."""
        batch, _, L = d_out.shape
        L_pad = self._input_padded.shape[2]

        dW = np.zeros_like(self.W)
        d_input_padded = np.zeros_like(self._input_padded)

        for t in range(L):
            for k in range(self.kernel_size):
                idx = t + self.padding - k * self.dilation
                if 0 <= idx < L_pad:
                    # dW[:, :, k] += d_out[:, :, t]^T @ x_padded[:, :, idx]
                    dW[:, :, k] += d_out[:, :, t].T @ self._input_padded[:, :, idx] / batch
                    # d_input_padded[:, :, idx] += d_out[:, :, t] @ W[:, :, k]
                    d_input_padded[:, :, idx] += d_out[:, :, t] @ self.W[:, :, k]

        db = np.mean(np.sum(d_out, axis=2, keepdims=True), axis=0)

        if self.padding > 0:
            d_input = d_input_padded[:, :, self.padding:]
        else:
            d_input = d_input_padded
        return d_input, dW, db

# ---------------------------------------------------------------------------
# Batch normalization (1D temporal)
# ---------------------------------------------------------------------------

class BatchNorm1D:
    """Batch normalization over the channel dimension for (batch, C, L)."""

    def __init__(self, n_channels: int, momentum: float = 0.1, eps: float = 1e-5):
        self.gamma = np.ones((1, n_channels, 1))
        self.beta_param = np.zeros((1, n_channels, 1))
        self.running_mean = np.zeros((1, n_channels, 1))
        self.running_var = np.ones((1, n_channels, 1))
        self.momentum = momentum
        self.eps = eps
        self.training = True
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            mean = x.mean(axis=(0, 2), keepdims=True)
            var = x.var(axis=(0, 2), keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        self._cache = {"x": x, "mean": mean, "var": var, "x_norm": x_norm}
        return self.gamma * x_norm + self.beta_param

    def backward(self, d_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_norm = self._cache["x_norm"]
        var = self._cache["var"]
        x = self._cache["x"]
        mean = self._cache["mean"]
        N = x.shape[0] * x.shape[2]

        d_gamma = np.sum(d_out * x_norm, axis=(0, 2), keepdims=True)
        d_beta = np.sum(d_out, axis=(0, 2), keepdims=True)

        d_xnorm = d_out * self.gamma
        std_inv = 1.0 / np.sqrt(var + self.eps)
        d_var = np.sum(d_xnorm * (x - mean) * (-0.5) * (var + self.eps) ** (-1.5),
                       axis=(0, 2), keepdims=True)
        d_mean = np.sum(d_xnorm * (-std_inv), axis=(0, 2), keepdims=True)
        d_x = d_xnorm * std_inv + d_var * 2.0 * (x - mean) / N + d_mean / N

        return d_x, d_gamma, d_beta

# ---------------------------------------------------------------------------
# Dropout (inverted)
# ---------------------------------------------------------------------------

class Dropout1D:
    """Inverted dropout for (batch, C, L) tensors."""

    def __init__(self, p: float = 0.1, rng: Optional[np.random.RandomState] = None):
        self.p = p
        self.rng = rng or np.random.RandomState(42)
        self.training = True
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            return x
        self._mask = (self.rng.rand(*x.shape) > self.p).astype(x.dtype) / (1.0 - self.p)
        return x * self._mask

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return d_out
        return d_out * self._mask

# ---------------------------------------------------------------------------
# Weight normalization (simplified)
# ---------------------------------------------------------------------------

def weight_norm_init(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose W into direction v and magnitude g: W = g * v / ||v||."""
    norm = np.sqrt(np.sum(W ** 2, axis=(1, 2), keepdims=True) + 1e-12)
    g = norm.copy()
    v = W.copy()
    return g, v


def weight_norm_compute(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reconstruct W = g * v / ||v||."""
    norm = np.sqrt(np.sum(v ** 2, axis=(1, 2), keepdims=True) + 1e-12)
    return g * v / norm

# ---------------------------------------------------------------------------
# TCN residual block
# ---------------------------------------------------------------------------

class TCNBlock:
    """
    Single TCN residual block:
      dilated_conv -> batch_norm -> relu -> dropout ->
      dilated_conv -> batch_norm -> relu -> dropout + residual
    """

    def __init__(self, n_channels: int, kernel_size: int, dilation: int,
                 dropout: float = 0.1, use_weight_norm: bool = False,
                 rng: Optional[np.random.RandomState] = None):
        rng = rng or np.random.RandomState(42)
        self.conv1 = CausalConv1D(n_channels, n_channels, kernel_size, dilation, rng)
        self.bn1 = BatchNorm1D(n_channels)
        self.drop1 = Dropout1D(dropout, rng)
        self.conv2 = CausalConv1D(n_channels, n_channels, kernel_size, dilation, rng)
        self.bn2 = BatchNorm1D(n_channels)
        self.drop2 = Dropout1D(dropout, rng)
        self.use_wn = use_weight_norm
        if use_weight_norm:
            self.g1, self.v1 = weight_norm_init(self.conv1.W)
            self.g2, self.v2 = weight_norm_init(self.conv2.W)
        # Cache
        self._input = None
        self._h1_pre = None
        self._h1_post = None
        self._h2_pre = None
        self._h2_post = None

    def _set_training(self, training: bool):
        self.bn1.training = training
        self.bn2.training = training
        self.drop1.training = training
        self.drop2.training = training

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        if self.use_wn:
            self.conv1.W = weight_norm_compute(self.g1, self.v1)
            self.conv2.W = weight_norm_compute(self.g2, self.v2)

        h = self.conv1.forward(x)
        h = self.bn1.forward(h)
        self._h1_pre = h
        h = _relu(h)
        self._h1_post = h
        h = self.drop1.forward(h)

        h = self.conv2.forward(h)
        h = self.bn2.forward(h)
        self._h2_pre = h
        h = _relu(h)
        self._h2_post = h
        h = self.drop2.forward(h)

        return h + x  # residual

    def backward(self, d_out: np.ndarray):
        """Returns d_input and dict of gradients."""
        grads = {}
        d_residual = d_out  # residual path

        # Through drop2
        d_h = self.drop2.backward(d_out)
        # Through relu2
        d_h = d_h * _relu_grad(self._h2_pre)
        # Through bn2
        d_h, d_gamma2, d_beta2 = self.bn2.backward(d_h)
        grads["bn2_gamma"] = d_gamma2
        grads["bn2_beta"] = d_beta2
        # Through conv2
        d_h, dW2, db2 = self.conv2.backward(d_h)
        grads["conv2_W"] = dW2
        grads["conv2_b"] = db2

        # Through drop1
        d_h = self.drop1.backward(d_h)
        # Through relu1
        d_h = d_h * _relu_grad(self._h1_pre)
        # Through bn1
        d_h, d_gamma1, d_beta1 = self.bn1.backward(d_h)
        grads["bn1_gamma"] = d_gamma1
        grads["bn1_beta"] = d_beta1
        # Through conv1
        d_h, dW1, db1 = self.conv1.backward(d_h)
        grads["conv1_W"] = dW1
        grads["conv1_b"] = db1

        d_input = d_h + d_residual
        return d_input, grads

# ---------------------------------------------------------------------------
# Gated TCN Block (WaveNet-style)
# ---------------------------------------------------------------------------

class GatedTCNBlock:
    """
    WaveNet-style gated activation:
      tanh(conv_filter(x)) * sigmoid(conv_gate(x)) + residual
    """

    def __init__(self, n_channels: int, kernel_size: int, dilation: int,
                 dropout: float = 0.1,
                 rng: Optional[np.random.RandomState] = None):
        rng = rng or np.random.RandomState(42)
        self.conv_filter = CausalConv1D(n_channels, n_channels, kernel_size, dilation, rng)
        self.conv_gate = CausalConv1D(n_channels, n_channels, kernel_size, dilation, rng)
        self.drop = Dropout1D(dropout, rng)
        # 1x1 conv for mixing
        self.conv_out = CausalConv1D(n_channels, n_channels, 1, 1, rng)
        self._cache = {}

    def _set_training(self, training: bool):
        self.drop.training = training

    def forward(self, x: np.ndarray) -> np.ndarray:
        f = _tanh(self.conv_filter.forward(x))
        g = _sigmoid(self.conv_gate.forward(x))
        h = f * g
        h = self.drop.forward(h)
        h = self.conv_out.forward(h)
        self._cache = {"x": x, "f": f, "g": g, "fg": h}
        return h + x

    def backward(self, d_out: np.ndarray):
        grads = {}
        d_residual = d_out
        d_h, dW_out, db_out = self.conv_out.backward(d_out)
        grads["conv_out_W"] = dW_out
        grads["conv_out_b"] = db_out
        d_h = self.drop.backward(d_h)
        f = self._cache["f"]
        g = self._cache["g"]
        d_f = d_h * g
        d_g = d_h * f
        # tanh grad
        d_conv_f = d_f * (1 - f ** 2)
        d_input_f, dW_f, db_f = self.conv_filter.backward(d_conv_f)
        grads["conv_filter_W"] = dW_f
        grads["conv_filter_b"] = db_f
        # sigmoid grad
        d_conv_g = d_g * g * (1 - g)
        d_input_g, dW_g, db_g = self.conv_gate.backward(d_conv_g)
        grads["conv_gate_W"] = dW_g
        grads["conv_gate_b"] = db_g
        d_input = d_input_f + d_input_g + d_residual
        return d_input, grads

# ---------------------------------------------------------------------------
# Input projection and output head
# ---------------------------------------------------------------------------

class Conv1x1:
    """1x1 convolution for channel projection."""

    def __init__(self, in_ch: int, out_ch: int,
                 rng: Optional[np.random.RandomState] = None):
        rng = rng or np.random.RandomState(42)
        std = np.sqrt(2.0 / in_ch)
        self.W = rng.randn(out_ch, in_ch) * std
        self.b = np.zeros((out_ch, 1))
        self._input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (batch, in_ch, L)
        self._input = x
        # W: (out_ch, in_ch) -> (batch, out_ch, L)
        return np.einsum("oi,biL->boL", self.W, x) + self.b[None, :, :]

    def backward(self, d_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = d_out.shape[0]
        dW = np.einsum("boL,biL->oi", d_out, self._input) / batch
        db = np.mean(np.sum(d_out, axis=2, keepdims=True), axis=0)
        d_input = np.einsum("oi,boL->biL", self.W.T, d_out)
        return d_input, dW, db

# ---------------------------------------------------------------------------
# Multi-scale TCN
# ---------------------------------------------------------------------------

class MultiScaleTCN:
    """
    Parallel branches with different dilation factors, concatenated.
    """

    def __init__(self, in_channels: int, branch_channels: int,
                 kernel_size: int, dilation_sets: List[List[int]],
                 dropout: float = 0.1,
                 rng: Optional[np.random.RandomState] = None):
        rng = rng or np.random.RandomState(42)
        self.branches: List[List[TCNBlock]] = []
        self.input_projs: List[Conv1x1] = []
        for dilations in dilation_sets:
            self.input_projs.append(Conv1x1(in_channels, branch_channels, rng))
            blocks = []
            for d in dilations:
                blocks.append(TCNBlock(branch_channels, kernel_size, d, dropout, rng=rng))
            self.branches.append(blocks)
        self.out_channels = branch_channels * len(dilation_sets)

    def _set_training(self, training: bool):
        for branch in self.branches:
            for block in branch:
                block._set_training(training)

    def forward(self, x: np.ndarray) -> np.ndarray:
        outputs = []
        for proj, branch in zip(self.input_projs, self.branches):
            h = proj.forward(x)
            for block in branch:
                h = block.forward(h)
            outputs.append(h)
        return np.concatenate(outputs, axis=1)

# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------

class AdamOptimizer:
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m: Dict[str, np.ndarray] = {}
        self._v: Dict[str, np.ndarray] = {}

    def step(self, key: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if key not in self._m:
            self._m[key] = np.zeros_like(param)
            self._v[key] = np.zeros_like(param)
        self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * grad
        self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * grad ** 2
        m_hat = self._m[key] / (1 - self.beta1 ** self.t)
        v_hat = self._v[key] / (1 - self.beta2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ---------------------------------------------------------------------------
# TCN model
# ---------------------------------------------------------------------------

class TemporalConvNetwork:
    """
    Full TCN for time series prediction.

    Input:  (batch, n_features, seq_len)
    Output: (batch, n_outputs) for seq-to-one, or (batch, n_outputs, seq_len) for seq-to-seq.

    Parameters
    ----------
    n_features : int
        Number of input features per time step.
    n_outputs : int
        Number of output targets.
    n_channels : int
        Hidden channel width in TCN blocks.
    kernel_size : int
        Convolution kernel size.
    n_layers : int
        Number of residual blocks (dilation doubles each layer).
    dropout : float
        Dropout probability.
    mode : str
        "seq2one" for final-step prediction, "seq2seq" for every step.
    gated : bool
        Use WaveNet-style gated activations instead of ReLU blocks.
    lr : float
        Learning rate.
    seed : int
        Random seed.
    """

    def __init__(self, n_features: int, n_outputs: int, n_channels: int = 32,
                 kernel_size: int = 3, n_layers: int = 4, dropout: float = 0.1,
                 mode: str = "seq2one", gated: bool = False,
                 lr: float = 1e-3, seed: int = 42):
        self.mode = mode
        self.n_outputs = n_outputs
        rng = np.random.RandomState(seed)
        self.input_proj = Conv1x1(n_features, n_channels, rng)
        self.blocks: List = []
        for i in range(n_layers):
            dilation = 2 ** i
            if gated:
                self.blocks.append(GatedTCNBlock(n_channels, kernel_size, dilation, dropout, rng))
            else:
                self.blocks.append(TCNBlock(n_channels, kernel_size, dilation, dropout, rng=rng))
        self.output_proj = Conv1x1(n_channels, n_outputs, rng)
        self.adam = AdamOptimizer(lr=lr)
        self.history: List[float] = []
        self._cache_hidden = None

    def _set_training(self, training: bool):
        for block in self.blocks:
            block._set_training(training)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, n_features, seq_len)"""
        h = self.input_proj.forward(x)
        for block in self.blocks:
            h = block.forward(h)
        self._cache_hidden = h
        out = self.output_proj.forward(h)  # (batch, n_outputs, seq_len)
        if self.mode == "seq2one":
            return out[:, :, -1]  # (batch, n_outputs)
        return out

    def _loss_mse(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
        diff = pred - target
        loss = float(np.mean(diff ** 2))
        grad = 2.0 * diff / diff.size
        return loss, grad

    def _backward(self, d_out_proj: np.ndarray):
        grads = {}
        d_h, dW_out, db_out = self.output_proj.backward(d_out_proj)
        grads["out_W"] = dW_out
        grads["out_b"] = db_out

        for i in reversed(range(len(self.blocks))):
            d_h, block_grads = self.blocks[i].backward(d_h)
            for k, v in block_grads.items():
                grads[f"block{i}_{k}"] = v

        d_x, dW_in, db_in = self.input_proj.backward(d_h)
        grads["in_W"] = dW_in
        grads["in_b"] = db_in
        return grads

    def _update_params(self, grads: Dict[str, np.ndarray]):
        self.adam.t += 1
        # input proj
        self.input_proj.W = self.adam.step("in_W", self.input_proj.W, grads["in_W"])
        self.input_proj.b = self.adam.step("in_b", self.input_proj.b, grads["in_b"])
        # output proj
        self.output_proj.W = self.adam.step("out_W", self.output_proj.W, grads["out_W"])
        self.output_proj.b = self.adam.step("out_b", self.output_proj.b, grads["out_b"])
        # blocks
        for i, block in enumerate(self.blocks):
            prefix = f"block{i}_"
            for k, v in grads.items():
                if k.startswith(prefix):
                    short = k[len(prefix):]
                    if "conv1_W" == short:
                        block.conv1.W = self.adam.step(k, block.conv1.W, v) if hasattr(block, "conv1") else block.conv_filter.W
                    elif "conv1_b" == short:
                        block.conv1.b = self.adam.step(k, block.conv1.b, v) if hasattr(block, "conv1") else block.conv_filter.b
                    elif "conv2_W" == short:
                        block.conv2.W = self.adam.step(k, block.conv2.W, v) if hasattr(block, "conv2") else block.conv_gate.W
                    elif "conv2_b" == short:
                        block.conv2.b = self.adam.step(k, block.conv2.b, v) if hasattr(block, "conv2") else block.conv_gate.b
                    elif "bn1_gamma" == short and hasattr(block, "bn1"):
                        block.bn1.gamma = self.adam.step(k, block.bn1.gamma, v)
                    elif "bn1_beta" == short and hasattr(block, "bn1"):
                        block.bn1.beta_param = self.adam.step(k, block.bn1.beta_param, v)
                    elif "bn2_gamma" == short and hasattr(block, "bn2"):
                        block.bn2.gamma = self.adam.step(k, block.bn2.gamma, v)
                    elif "bn2_beta" == short and hasattr(block, "bn2"):
                        block.bn2.beta_param = self.adam.step(k, block.bn2.beta_param, v)
                    # Gated block params
                    elif "conv_filter_W" == short and hasattr(block, "conv_filter"):
                        block.conv_filter.W = self.adam.step(k, block.conv_filter.W, v)
                    elif "conv_filter_b" == short and hasattr(block, "conv_filter"):
                        block.conv_filter.b = self.adam.step(k, block.conv_filter.b, v)
                    elif "conv_gate_W" == short and hasattr(block, "conv_gate"):
                        block.conv_gate.W = self.adam.step(k, block.conv_gate.W, v)
                    elif "conv_gate_b" == short and hasattr(block, "conv_gate"):
                        block.conv_gate.b = self.adam.step(k, block.conv_gate.b, v)
                    elif "conv_out_W" == short and hasattr(block, "conv_out"):
                        block.conv_out.W = self.adam.step(k, block.conv_out.W, v)
                    elif "conv_out_b" == short and hasattr(block, "conv_out"):
                        block.conv_out.b = self.adam.step(k, block.conv_out.b, v)

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        self._set_training(True)
        pred = self.forward(x)
        if self.mode == "seq2one":
            loss, d_pred = self._loss_mse(pred, y)
            # Expand grad back to (batch, n_outputs, seq_len) with zeros except last
            d_full = np.zeros((x.shape[0], self.n_outputs, x.shape[2]))
            d_full[:, :, -1] = d_pred
        else:
            loss, d_full = self._loss_mse(pred, y)
        grads = self._backward(d_full)
        self._update_params(grads)
        return loss

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100,
            batch_size: int = 32, verbose: bool = False) -> List[float]:
        n = X.shape[0]
        rng = np.random.RandomState(0)
        self.history = []
        for epoch in range(epochs):
            idx = rng.permutation(n)
            epoch_loss = 0.0
            nb = 0
            for s in range(0, n, batch_size):
                bi = idx[s:s + batch_size]
                loss = self.train_step(X[bi], Y[bi])
                epoch_loss += loss
                nb += 1
            epoch_loss /= nb
            self.history.append(epoch_loss)
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch:4d} | MSE={epoch_loss:.6f}")
        return self.history

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._set_training(False)
        return self.forward(x)

    def receptive_field(self) -> int:
        """Compute the receptive field of the TCN."""
        rf = 1
        for block in self.blocks:
            if hasattr(block, "conv1"):
                k = block.conv1.kernel_size
                d = block.conv1.dilation
            else:
                k = block.conv_filter.kernel_size
                d = block.conv_filter.dilation
            rf += 2 * (k - 1) * d  # two conv layers per block
        return rf

# ---------------------------------------------------------------------------
# Multi-horizon return forecasting application
# ---------------------------------------------------------------------------

class MultiHorizonForecaster:
    """
    Multi-horizon return forecasting with TCN.

    Given a window of past returns and features, predict returns at
    multiple future horizons simultaneously.
    """

    def __init__(self, n_features: int, horizons: List[int],
                 n_channels: int = 32, kernel_size: int = 3,
                 n_layers: int = 4, dropout: float = 0.1,
                 gated: bool = False, lr: float = 1e-3, seed: int = 42):
        self.horizons = horizons
        self.n_outputs = len(horizons)
        self.tcn = TemporalConvNetwork(
            n_features, self.n_outputs, n_channels, kernel_size,
            n_layers, dropout, mode="seq2one", gated=gated, lr=lr, seed=seed)
        self._mean_x = None
        self._std_x = None
        self._mean_y = None
        self._std_y = None

    def _prepare_data(self, returns: np.ndarray, lookback: int,
                      fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Create supervised windows from a return series (T, n_features)."""
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        T, F = returns.shape
        max_h = max(self.horizons)
        n_samples = T - lookback - max_h + 1
        if n_samples <= 0:
            raise ValueError("Not enough data for given lookback and horizons.")
        X = np.zeros((n_samples, F, lookback))
        Y = np.zeros((n_samples, self.n_outputs))
        for i in range(n_samples):
            X[i] = returns[i:i + lookback].T
            for j, h in enumerate(self.horizons):
                Y[i, j] = returns[i + lookback + h - 1, 0]

        if fit:
            self._mean_x = X.mean(axis=(0, 2), keepdims=True)
            self._std_x = X.std(axis=(0, 2), keepdims=True) + 1e-8
            self._mean_y = Y.mean(axis=0, keepdims=True)
            self._std_y = Y.std(axis=0, keepdims=True) + 1e-8
        X = (X - self._mean_x) / self._std_x
        Y = (Y - self._mean_y) / self._std_y
        return X, Y

    def fit(self, returns: np.ndarray, lookback: int = 60,
            epochs: int = 100, batch_size: int = 32,
            verbose: bool = False) -> List[float]:
        X, Y = self._prepare_data(returns, lookback, fit=True)
        return self.tcn.fit(X, Y, epochs, batch_size, verbose)

    def predict(self, returns: np.ndarray, lookback: int = 60) -> np.ndarray:
        X, _ = self._prepare_data(returns, lookback)
        pred_norm = self.tcn.predict(X)
        return pred_norm * self._std_y + self._mean_y

    def evaluate(self, returns: np.ndarray, lookback: int = 60) -> Dict[str, float]:
        X, Y = self._prepare_data(returns, lookback)
        pred = self.tcn.predict(X)
        mse = float(np.mean((pred - Y) ** 2))
        # Directional accuracy per horizon
        results = {"mse": mse}
        for j, h in enumerate(self.horizons):
            da = float(np.mean(np.sign(pred[:, j]) == np.sign(Y[:, j])))
            results[f"dir_acc_h{h}"] = da
        return results


# ---------------------------------------------------------------------------
# Autoregressive sequence generation
# ---------------------------------------------------------------------------

def autoregressive_generate(tcn: TemporalConvNetwork,
                            seed_seq: np.ndarray,
                            n_steps: int) -> np.ndarray:
    """
    Generate future steps autoregressively using a seq2one TCN.

    seed_seq: (1, n_features, seq_len)
    Returns: (n_steps, n_outputs)
    """
    tcn._set_training(False)
    generated = []
    current = seed_seq.copy()
    for _ in range(n_steps):
        pred = tcn.forward(current)  # (1, n_outputs)
        generated.append(pred[0])
        # Shift window: drop first timestep, append prediction
        new_step = np.zeros((1, current.shape[1], 1))
        new_step[0, :pred.shape[1], 0] = pred[0]
        current = np.concatenate([current[:, :, 1:], new_step], axis=2)
    return np.array(generated)
