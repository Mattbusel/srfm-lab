"""
research/agent_training/networks.py

Pure-numpy neural network building blocks for RL agents.

No PyTorch or TensorFlow — all forward/backward passes are implemented
using NumPy operations with manual gradient computation.

Layers:
    Linear, ReLU, Tanh, Sigmoid, GELU, LayerNorm, Dropout
    Sequential (container)
    DuelingHead   — dueling DQN value/advantage split
    ActorNetwork  — continuous action actor (tanh output)
    CriticNetwork — (state, action) -> Q-value
    TransformerBlock — single-head self-attention (temporal)
    LSTMCell      — single LSTM cell for sequential processing

All parameter layers expose:
    forward(x)             -> y
    backward(grad_out)     -> grad_in, and accumulates param gradients
    update(lr)             -> Adam step on accumulated gradients
    zero_grad()            -> clear accumulated gradients
    save_weights(path)     -> .npz
    load_weights(path)     -> restores params
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Adam state helper
# ---------------------------------------------------------------------------


class _AdamState:
    """Per-parameter Adam first and second moment estimates."""

    def __init__(self, shape: tuple, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Layer:
    """Abstract base for all network layers."""

    training: bool = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, lr: float) -> None:
        pass

    def zero_grad(self) -> None:
        pass

    def save_weights(self, path: str) -> None:
        pass

    def load_weights(self, path: str) -> None:
        pass

    def set_training(self, mode: bool) -> None:
        self.training = mode

    def parameters(self) -> dict[str, np.ndarray]:
        return {}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------


class Linear(Layer):
    """
    Fully connected linear layer: y = x @ W.T + b

    Args:
        in_features  : Input dimensionality.
        out_features : Output dimensionality.
        use_bias     : Whether to include bias.
        lr           : Adam learning rate.
        l2_reg       : L2 weight decay coefficient.
        init         : Weight initialisation scheme ('he', 'xavier', 'normal').
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        lr: float = 1e-3,
        l2_reg: float = 0.0,
        init: str = "he",
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.l2_reg = l2_reg

        if init == "he":
            std = math.sqrt(2.0 / in_features)
        elif init == "xavier":
            std = math.sqrt(2.0 / (in_features + out_features))
        else:
            std = 0.01

        self.W = np.random.randn(out_features, in_features).astype(np.float64) * std
        self.b = np.zeros(out_features, dtype=np.float64) if use_bias else None

        self._adam_W = _AdamState((out_features, in_features), lr=lr)
        self._adam_b = _AdamState((out_features,), lr=lr) if use_bias else None

        self._grad_W = np.zeros_like(self.W)
        self._grad_b = np.zeros_like(self.b) if use_bias else None
        self._x_cache: Optional[np.ndarray] = None  # saved for backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x_cache = x
        out = x @ self.W.T
        if self.use_bias:
            out = out + self.b
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        grad_out: (batch, out_features) or (out_features,)
        Returns grad w.r.t. input x.
        """
        x = self._x_cache
        if x.ndim == 1:
            x_2d = x.reshape(1, -1)
            go_2d = grad_out.reshape(1, -1)
        else:
            x_2d = x
            go_2d = grad_out

        self._grad_W += go_2d.T @ x_2d + self.l2_reg * self.W
        if self.use_bias:
            self._grad_b += go_2d.sum(axis=0)

        grad_x = go_2d @ self.W
        if x.ndim == 1:
            return grad_x.reshape(-1)
        return grad_x

    def update(self, lr: float) -> None:
        dW = self._adam_W.step(self._grad_W)
        self.W -= dW
        if self.use_bias:
            db = self._adam_b.step(self._grad_b)  # type: ignore[union-attr]
            self.b -= db
        self.zero_grad()

    def zero_grad(self) -> None:
        self._grad_W[:] = 0.0
        if self.use_bias:
            self._grad_b[:] = 0.0  # type: ignore[index]

    def parameters(self) -> dict[str, np.ndarray]:
        d = {"W": self.W}
        if self.use_bias:
            d["b"] = self.b  # type: ignore[assignment]
        return d

    def save_weights(self, path: str) -> None:
        np.savez(path, W=self.W, b=self.b if self.use_bias else np.array([]))

    def load_weights(self, path: str) -> None:
        data = np.load(path)
        self.W[:] = data["W"]
        if self.use_bias and data["b"].size > 0:
            self.b[:] = data["b"]

    def clip_gradients(self, max_norm: float) -> None:
        """Clip accumulated gradients by global norm."""
        norm = float(np.linalg.norm(self._grad_W))
        if norm > max_norm:
            self._grad_W *= max_norm / norm
        if self.use_bias:
            nb = float(np.linalg.norm(self._grad_b))  # type: ignore[arg-type]
            if nb > max_norm:
                self._grad_b *= max_norm / nb  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Activation layers
# ---------------------------------------------------------------------------


class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask


class Tanh(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * (1.0 - self._out ** 2)


class Sigmoid(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
        return self._out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._out * (1.0 - self._out)


class GELU(Layer):
    """
    Gaussian Error Linear Unit activation.
    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x^3)))
    """

    _K = math.sqrt(2.0 / math.pi)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        inner = self._K * (x + 0.044715 * x ** 3)
        self._tanh_inner = np.tanh(inner)
        return 0.5 * x * (1.0 + self._tanh_inner)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._x
        t = self._tanh_inner
        inner = self._K * (x + 0.044715 * x ** 3)
        dt_dx = self._K * (1.0 + 3.0 * 0.044715 * x ** 2)
        dgelu_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * dt_dx
        return grad_out * dgelu_dx


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------


class LayerNorm(Layer):
    """
    Layer normalisation over the last dimension.

    Args:
        normalized_shape : Number of features.
        eps              : Numerical stability epsilon.
        lr               : Adam learning rate for gamma and beta.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, lr: float = 1e-3) -> None:
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = np.ones(normalized_shape, dtype=np.float64)
        self.beta = np.zeros(normalized_shape, dtype=np.float64)
        self._adam_g = _AdamState((normalized_shape,), lr=lr)
        self._adam_b = _AdamState((normalized_shape,), lr=lr)
        self._grad_g = np.zeros(normalized_shape, dtype=np.float64)
        self._grad_b = np.zeros(normalized_shape, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._mean = x.mean(axis=-1, keepdims=True)
        self._var = x.var(axis=-1, keepdims=True)
        self._x_norm = (x - self._mean) / np.sqrt(self._var + self.eps)
        return self.gamma * self._x_norm + self.beta

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        xn = self._x_norm
        n = self.normalized_shape
        std_inv = 1.0 / np.sqrt(self._var + self.eps)

        self._grad_g += (grad_out * xn).reshape(-1, n).sum(axis=0)
        self._grad_b += grad_out.reshape(-1, n).sum(axis=0)

        dxn = grad_out * self.gamma
        dvar = (-0.5 * dxn * (self._x - self._mean) * std_inv ** 3).sum(axis=-1, keepdims=True)
        dmean = (-dxn * std_inv).sum(axis=-1, keepdims=True) + dvar * (-2.0 * (self._x - self._mean)).mean(axis=-1, keepdims=True)
        dx = dxn * std_inv + dvar * 2.0 * (self._x - self._mean) / n + dmean / n
        return dx

    def update(self, lr: float) -> None:
        self.gamma -= self._adam_g.step(self._grad_g)
        self.beta -= self._adam_b.step(self._grad_b)
        self.zero_grad()

    def zero_grad(self) -> None:
        self._grad_g[:] = 0.0
        self._grad_b[:] = 0.0

    def parameters(self) -> dict[str, np.ndarray]:
        return {"gamma": self.gamma, "beta": self.beta}


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------


class Dropout(Layer):
    """
    Inverted dropout. Inactive during inference (training=False).

    Args:
        p : Dropout probability (fraction of units to zero out).
    """

    def __init__(self, p: float = 0.1) -> None:
        self.p = p
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            return x
        self._mask = (np.random.rand(*x.shape) > self.p).astype(np.float64) / (1.0 - self.p)
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if not self.training or self._mask is None:
            return grad_out
        return grad_out * self._mask


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------


class Sequential(Layer):
    """
    Chain of layers executed sequentially.

    Args:
        layers : Ordered list of Layer objects.
        lr     : Learning rate passed to update() of each layer.
    """

    def __init__(self, layers: list[Layer], lr: float = 1e-3) -> None:
        self.layers = layers
        self.lr = lr

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = grad_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, lr: Optional[float] = None) -> None:
        _lr = lr if lr is not None else self.lr
        for layer in self.layers:
            layer.update(_lr)

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def set_training(self, mode: bool) -> None:
        self.training = mode
        for layer in self.layers:
            layer.set_training(mode)

    def parameters(self) -> dict[str, np.ndarray]:
        params = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.parameters().items():
                params[f"layer{i}_{k}"] = v
        return params

    def save_weights(self, path: str) -> None:
        params = self.parameters()
        np.savez_compressed(path, **params)

    def load_weights(self, path: str) -> None:
        data = np.load(path)
        params = self.parameters()
        for key in params:
            if key in data:
                params[key][:] = data[key]

    def clip_all_gradients(self, max_norm: float) -> None:
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.clip_gradients(max_norm)

    def __repr__(self) -> str:
        parts = [f"  ({i}): {type(l).__name__}" for i, l in enumerate(self.layers)]
        return "Sequential(\n" + "\n".join(parts) + "\n)"


def mlp(
    in_features: int,
    hidden_dims: list[int],
    out_features: int,
    activation: str = "relu",
    dropout: float = 0.0,
    layer_norm: bool = False,
    lr: float = 1e-3,
    l2_reg: float = 0.0,
) -> Sequential:
    """
    Build a fully connected MLP.

    Args:
        in_features  : Input size.
        hidden_dims  : List of hidden layer sizes.
        out_features : Output size.
        activation   : 'relu', 'tanh', 'sigmoid', 'gelu'.
        dropout      : Dropout probability (0 = off).
        layer_norm   : Whether to apply LayerNorm after each hidden layer.
        lr           : Adam learning rate.
        l2_reg       : L2 regularisation on Linear weights.

    Returns:
        Sequential model.
    """
    act_map: dict[str, type] = {
        "relu": ReLU,
        "tanh": Tanh,
        "sigmoid": Sigmoid,
        "gelu": GELU,
    }
    act_cls = act_map.get(activation.lower(), ReLU)

    layers: list[Layer] = []
    prev = in_features
    for h in hidden_dims:
        layers.append(Linear(prev, h, lr=lr, l2_reg=l2_reg))
        if layer_norm:
            layers.append(LayerNorm(h, lr=lr))
        layers.append(act_cls())
        if dropout > 0.0:
            layers.append(Dropout(dropout))
        prev = h
    layers.append(Linear(prev, out_features, lr=lr, l2_reg=l2_reg))
    return Sequential(layers, lr=lr)


# ---------------------------------------------------------------------------
# DuelingHead
# ---------------------------------------------------------------------------


class DuelingHead(Layer):
    """
    Dueling network head: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))

    Args:
        in_features : Input feature dimension.
        n_actions   : Number of discrete actions.
        hidden_dim  : Size of value and advantage streams.
        lr          : Adam learning rate.
    """

    def __init__(
        self,
        in_features: int,
        n_actions: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
    ) -> None:
        self.n_actions = n_actions

        # Value stream: scalar V(s)
        self.value_stream = Sequential(
            [Linear(in_features, hidden_dim, lr=lr), ReLU(), Linear(hidden_dim, 1, lr=lr)],
            lr=lr,
        )
        # Advantage stream: A(s,a) for each action
        self.advantage_stream = Sequential(
            [Linear(in_features, hidden_dim, lr=lr), ReLU(), Linear(hidden_dim, n_actions, lr=lr)],
            lr=lr,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Returns Q-values of shape (..., n_actions).
        """
        V = self.value_stream.forward(x)           # (..., 1)
        A = self.advantage_stream.forward(x)       # (..., n_actions)
        A_mean = A.mean(axis=-1, keepdims=True)
        return V + (A - A_mean)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        n = self.n_actions
        # d(Q)/d(A) = I - 1/n * ones
        # d(Q)/d(V) = 1
        grad_A = grad_out - grad_out.mean(axis=-1, keepdims=True)
        grad_V = grad_out.sum(axis=-1, keepdims=True)
        self.value_stream.backward(grad_V)
        return self.advantage_stream.backward(grad_A)

    def update(self, lr: float) -> None:
        self.value_stream.update(lr)
        self.advantage_stream.update(lr)

    def zero_grad(self) -> None:
        self.value_stream.zero_grad()
        self.advantage_stream.zero_grad()

    def set_training(self, mode: bool) -> None:
        self.value_stream.set_training(mode)
        self.advantage_stream.set_training(mode)

    def save_weights(self, path: str) -> None:
        params_v = {f"V_{k}": v for k, v in self.value_stream.parameters().items()}
        params_a = {f"A_{k}": v for k, v in self.advantage_stream.parameters().items()}
        np.savez_compressed(path, **params_v, **params_a)

    def load_weights(self, path: str) -> None:
        data = np.load(path)
        for k, v in self.value_stream.parameters().items():
            key = f"V_{k}"
            if key in data:
                v[:] = data[key]
        for k, v in self.advantage_stream.parameters().items():
            key = f"A_{k}"
            if key in data:
                v[:] = data[key]


# ---------------------------------------------------------------------------
# ActorNetwork
# ---------------------------------------------------------------------------


class ActorNetwork(Layer):
    """
    Continuous action actor: maps state -> action in [-1, 1].

    Architecture: MLP + Tanh output.

    Args:
        obs_dim     : State dimensionality.
        action_dim  : Action dimensionality.
        hidden_dims : Hidden layer sizes.
        lr          : Adam learning rate.
        dropout     : Dropout probability.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = (256, 256),
        lr: float = 3e-4,
        dropout: float = 0.0,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        layers_: list[Layer] = []
        prev = obs_dim
        for h in hidden_dims:
            layers_.append(Linear(prev, h, lr=lr))
            layers_.append(ReLU())
            if dropout > 0:
                layers_.append(Dropout(dropout))
            prev = h
        layers_.append(Linear(prev, action_dim, lr=lr, init="xavier"))
        layers_.append(Tanh())
        self._net = Sequential(layers_, lr=lr)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._net.forward(x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return self._net.backward(grad_out)

    def update(self, lr: float) -> None:
        self._net.update(lr)

    def zero_grad(self) -> None:
        self._net.zero_grad()

    def set_training(self, mode: bool) -> None:
        self._net.set_training(mode)

    def parameters(self) -> dict[str, np.ndarray]:
        return self._net.parameters()

    def save_weights(self, path: str) -> None:
        self._net.save_weights(path)

    def load_weights(self, path: str) -> None:
        self._net.load_weights(path)


# ---------------------------------------------------------------------------
# CriticNetwork
# ---------------------------------------------------------------------------


class CriticNetwork(Layer):
    """
    (State, Action) -> Q-value critic.

    Concatenates [state, action] then passes through MLP.

    Args:
        obs_dim    : State dimensionality.
        action_dim : Action dimensionality.
        hidden_dims: Hidden layer sizes.
        lr         : Adam learning rate.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = (256, 256),
        lr: float = 1e-3,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        in_dim = obs_dim + action_dim

        layers_: list[Layer] = []
        prev = in_dim
        for h in hidden_dims:
            layers_.append(Linear(prev, h, lr=lr))
            layers_.append(ReLU())
            prev = h
        layers_.append(Linear(prev, 1, lr=lr))
        self._net = Sequential(layers_, lr=lr)

    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Args:
            state  : (batch, obs_dim) or (obs_dim,)
            action : (batch, action_dim) or (action_dim,)
        Returns:
            Q-values: (batch, 1) or (1,)
        """
        if state.ndim == 1:
            x = np.concatenate([state, np.asarray(action).reshape(-1)])
        else:
            x = np.concatenate([state, np.asarray(action).reshape(state.shape[0], -1)], axis=-1)
        return self._net.forward(x)

    def forward_concat(self, x: np.ndarray) -> np.ndarray:
        """Forward pass on pre-concatenated (state, action) input."""
        return self._net.forward(x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return self._net.backward(grad_out)

    def update(self, lr: float) -> None:
        self._net.update(lr)

    def zero_grad(self) -> None:
        self._net.zero_grad()

    def set_training(self, mode: bool) -> None:
        self._net.set_training(mode)

    def parameters(self) -> dict[str, np.ndarray]:
        return self._net.parameters()

    def save_weights(self, path: str) -> None:
        self._net.save_weights(path)

    def load_weights(self, path: str) -> None:
        self._net.load_weights(path)


# ---------------------------------------------------------------------------
# TransformerBlock (simplified single-head self-attention)
# ---------------------------------------------------------------------------


class TransformerBlock(Layer):
    """
    Single-head self-attention + feed-forward block for temporal sequences.

    Input shape: (seq_len, d_model)
    Output shape: (seq_len, d_model)

    Args:
        d_model   : Feature dimension.
        d_ff      : Feed-forward hidden dimension.
        dropout   : Dropout probability.
        lr        : Adam learning rate.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
    ) -> None:
        self.d_model = d_model
        self.scale = math.sqrt(d_model)

        # QKV projections
        self.W_q = Linear(d_model, d_model, lr=lr)
        self.W_k = Linear(d_model, d_model, lr=lr)
        self.W_v = Linear(d_model, d_model, lr=lr)
        self.W_o = Linear(d_model, d_model, lr=lr)

        # Feed-forward
        self.ff = Sequential(
            [Linear(d_model, d_ff, lr=lr), GELU(), Linear(d_ff, d_model, lr=lr)],
            lr=lr,
        )

        # Layer norms
        self.ln1 = LayerNorm(d_model, lr=lr)
        self.ln2 = LayerNorm(d_model, lr=lr)

        self.dropout = Dropout(dropout)
        self._attn_weights: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (seq_len, d_model)
        Returns:
            out: (seq_len, d_model)
        """
        # Self-attention
        Q = self.W_q.forward(x)   # (T, d)
        K = self.W_k.forward(x)
        V = self.W_v.forward(x)

        scores = Q @ K.T / self.scale   # (T, T)
        # Causal mask
        T = x.shape[0]
        mask = np.triu(np.full((T, T), -1e9), k=1)
        scores = scores + mask

        attn = _softmax(scores)           # (T, T)
        self._attn_weights = attn

        context = attn @ V               # (T, d)
        context = self.dropout.forward(context)
        attn_out = self.W_o.forward(context)

        # Residual + LayerNorm
        x1 = self.ln1.forward(x + attn_out)

        # Feed-forward
        ff_out = self.ff.forward(x1)
        out = self.ln2.forward(x1 + ff_out)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # Simplified: only pass gradient through FF and linear layers
        grad_ff = self.ln2.backward(grad_out)
        grad_ff2 = self.ff.backward(grad_ff)
        grad_attn = self.ln1.backward(grad_ff + grad_ff2)
        grad_attn2 = self.W_o.backward(grad_attn)
        return grad_attn2

    def update(self, lr: float) -> None:
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o, self.ff, self.ln1, self.ln2]:
            layer.update(lr)

    def zero_grad(self) -> None:
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o, self.ff, self.ln1, self.ln2]:
            layer.zero_grad()

    def set_training(self, mode: bool) -> None:
        self.training = mode
        self.dropout.set_training(mode)
        self.ff.set_training(mode)

    def parameters(self) -> dict[str, np.ndarray]:
        params = {}
        for name, mod in [("Wq", self.W_q), ("Wk", self.W_k), ("Wv", self.W_v), ("Wo", self.W_o)]:
            for k, v in mod.parameters().items():
                params[f"{name}_{k}"] = v
        return params

    def save_weights(self, path: str) -> None:
        np.savez_compressed(path, **self.parameters())

    def load_weights(self, path: str) -> None:
        data = np.load(path)
        params = self.parameters()
        for k, v in params.items():
            if k in data:
                v[:] = data[k]


# ---------------------------------------------------------------------------
# LSTM cell
# ---------------------------------------------------------------------------


class LSTMCell(Layer):
    """
    Single LSTM cell (pure numpy) for sequential BH signal processing.

    State: (h_t, c_t) each of shape (hidden_size,)

    Args:
        input_size  : Size of input vector x_t.
        hidden_size : Size of hidden/cell state.
        lr          : Adam learning rate.
    """

    def __init__(self, input_size: int, hidden_size: int, lr: float = 1e-3) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        n = hidden_size
        m = input_size

        # Gates: i, f, g, o — all in one stacked matrix
        # W: (4n, m+n), b: (4n,)
        k = math.sqrt(1.0 / n)
        self.W = np.random.uniform(-k, k, (4 * n, m + n)).astype(np.float64)
        self.b = np.zeros(4 * n, dtype=np.float64)

        self._adam_W = _AdamState((4 * n, m + n), lr=lr)
        self._adam_b = _AdamState((4 * n,), lr=lr)
        self._grad_W = np.zeros_like(self.W)
        self._grad_b = np.zeros_like(self.b)

        # Cache for backward
        self._cache: list = []

    def forward(
        self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Single step forward.

        Args:
            x      : Input at time t, shape (input_size,).
            h_prev : Previous hidden state, shape (hidden_size,).
            c_prev : Previous cell state, shape (hidden_size,).

        Returns:
            (h_t, c_t) each of shape (hidden_size,).
        """
        concat = np.concatenate([x, h_prev])  # (m+n,)
        gates_raw = self.W @ concat + self.b  # (4n,)
        n = self.hidden_size

        i_raw = gates_raw[:n]
        f_raw = gates_raw[n : 2 * n]
        g_raw = gates_raw[2 * n : 3 * n]
        o_raw = gates_raw[3 * n : 4 * n]

        i_gate = _sigmoid(i_raw)
        f_gate = _sigmoid(f_raw)
        g_gate = np.tanh(g_raw)
        o_gate = _sigmoid(o_raw)

        c_t = f_gate * c_prev + i_gate * g_gate
        h_t = o_gate * np.tanh(c_t)

        self._cache.append((concat, i_gate, f_gate, g_gate, o_gate, c_prev, c_t, h_t))
        return h_t, c_t

    def backward(
        self,
        grad_h: np.ndarray,
        grad_c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        BPTT for one step.

        Returns:
            (grad_x, grad_h_prev, grad_c_prev)
        """
        if not self._cache:
            n = self.hidden_size
            m = self.input_size
            return np.zeros(m), np.zeros(n), np.zeros(n)

        concat, i_g, f_g, g_g, o_g, c_prev, c_t, h_t = self._cache.pop()
        tanh_ct = np.tanh(c_t)

        d_o = grad_h * tanh_ct * o_g * (1.0 - o_g)
        d_c = grad_h * o_g * (1.0 - tanh_ct ** 2) + grad_c
        d_f = d_c * c_prev * f_g * (1.0 - f_g)
        d_i = d_c * g_g * i_g * (1.0 - i_g)
        d_g = d_c * i_g * (1.0 - g_g ** 2)
        d_c_prev = d_c * f_g

        d_gates = np.concatenate([d_i, d_f, d_g, d_o])  # (4n,)
        self._grad_W += np.outer(d_gates, concat)
        self._grad_b += d_gates

        d_concat = self.W.T @ d_gates
        m = self.input_size
        grad_x = d_concat[:m]
        grad_h_prev = d_concat[m:]

        return grad_x, grad_h_prev, d_c_prev

    def update(self, lr: float) -> None:
        self.W -= self._adam_W.step(self._grad_W)
        self.b -= self._adam_b.step(self._grad_b)
        self.zero_grad()

    def zero_grad(self) -> None:
        self._grad_W[:] = 0.0
        self._grad_b[:] = 0.0
        self._cache.clear()

    def init_hidden(self, batch_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return zero (h, c) states."""
        n = self.hidden_size
        return np.zeros((batch_size, n)), np.zeros((batch_size, n))

    def parameters(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}

    def save_weights(self, path: str) -> None:
        np.savez(path, W=self.W, b=self.b)

    def load_weights(self, path: str) -> None:
        data = np.load(path)
        self.W[:] = data["W"]
        self.b[:] = data["b"]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    x_max = x.max(axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def clip_grad_norm(layers: list[Layer], max_norm: float) -> float:
    """
    Clip gradients across all Linear layers by global L2 norm.

    Returns the pre-clipping gradient norm.
    """
    total_sq = 0.0
    linear_layers = [l for l in layers if isinstance(l, Linear)]
    for layer in linear_layers:
        total_sq += float(np.sum(layer._grad_W ** 2))
        if layer.use_bias:
            total_sq += float(np.sum(layer._grad_b ** 2))  # type: ignore
    norm = math.sqrt(total_sq) + 1e-12
    if norm > max_norm:
        scale = max_norm / norm
        for layer in linear_layers:
            layer._grad_W *= scale
            if layer.use_bias:
                layer._grad_b *= scale  # type: ignore
    return norm


def save_all_weights(model: Layer, path: str) -> None:
    """Save all parameters of a model to a .npz file."""
    params = model.parameters()
    np.savez_compressed(path, **{k: v for k, v in params.items()})


def load_all_weights(model: Layer, path: str) -> None:
    """Load all parameters of a model from a .npz file."""
    data = np.load(path)
    params = model.parameters()
    for k, v in params.items():
        if k in data:
            v[:] = data[k]
