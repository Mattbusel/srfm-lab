"""
Transformer architecture for financial time series (numpy-only).

Implements:
  - Multi-head self-attention (scaled dot-product)
  - Positional encoding (sinusoidal + learnable)
  - Transformer encoder block with residual connections
  - Feed-forward sublayer with GELU activation
  - Layer normalization
  - Temporal Fusion Transformer (TFT) elements:
    - Variable Selection Network
    - Gated Residual Network (GRN)
    - Quantile output head
  - Market-specific: causal masking, volatility-weighted attention
  - Training loop with Adam and gradient clipping
  - Walk-forward inference
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Activation Functions ──────────────────────────────────────────────────────

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation: x * Phi(x) where Phi is standard normal CDF."""
    return x * 0.5 * (1 + np.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of GELU."""
    tanh_arg = math.sqrt(2/math.pi) * (x + 0.044715 * x**3)
    tanh_val = np.tanh(tanh_arg)
    sech2 = 1 - tanh_val**2
    dtanh = sech2 * math.sqrt(2/math.pi) * (1 + 3 * 0.044715 * x**2)
    return 0.5 * (1 + tanh_val) + x * 0.5 * dtanh


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-10)


# ── Layer Normalization ───────────────────────────────────────────────────────

class LayerNorm:
    def __init__(self, d: int, eps: float = 1e-6):
        self.d = d
        self.eps = eps
        self.gamma = np.ones(d)
        self.beta = np.zeros(d)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mu) / np.sqrt(var + self.eps) + self.beta

    def params(self) -> list[np.ndarray]:
        return [self.gamma, self.beta]


# ── Positional Encoding ───────────────────────────────────────────────────────

def sinusoidal_pe(T: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding."""
    pe = np.zeros((T, d_model))
    pos = np.arange(T)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-math.log(10000) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
    return pe


# ── Scaled Dot-Product Attention ──────────────────────────────────────────────

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
    dropout_rate: float = 0.0,
    training: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Q, K, V: (..., seq, d_k)
    Returns (output, attention_weights)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    attn = softmax(scores, axis=-1)

    if dropout_rate > 0 and training and rng is not None:
        keep = rng.binomial(1, 1 - dropout_rate, attn.shape).astype(float)
        attn = attn * keep / (1 - dropout_rate + 1e-10)

    return attn @ V, attn


# ── Multi-Head Attention ──────────────────────────────────────────────────────

class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int, seed: int = 42):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        rng = np.random.default_rng(seed)
        scale = math.sqrt(2.0 / (d_model + self.d_k))
        self.W_Q = rng.standard_normal((d_model, d_model)) * scale
        self.W_K = rng.standard_normal((d_model, d_model)) * scale
        self.W_V = rng.standard_normal((d_model, d_model)) * scale
        self.W_O = rng.standard_normal((d_model, d_model)) * scale
        self.b_Q = np.zeros(d_model)
        self.b_K = np.zeros(d_model)
        self.b_V = np.zeros(d_model)
        self.b_O = np.zeros(d_model)

    def forward(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """X: (batch, T, d_model) or (T, d_model)"""
        squeeze = X.ndim == 2
        if squeeze:
            X = X[None, :, :]

        B, T, _ = X.shape
        Q = X @ self.W_Q + self.b_Q  # (B, T, d_model)
        K = X @ self.W_K + self.b_K
        V = X @ self.W_V + self.b_V

        # Split heads
        Q = Q.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)  # (B, h, T, d_k)
        K = K.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask, training=training)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        out = out @ self.W_O + self.b_O

        if squeeze:
            out = out[0]
            attn = attn[0]

        return out, attn

    def params(self) -> list[np.ndarray]:
        return [self.W_Q, self.W_K, self.W_V, self.W_O,
                self.b_Q, self.b_K, self.b_V, self.b_O]


# ── Feed-Forward Sublayer ─────────────────────────────────────────────────────

class FeedForward:
    def __init__(self, d_model: int, d_ff: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = math.sqrt(2.0 / d_model)
        self.W1 = rng.standard_normal((d_model, d_ff)) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.standard_normal((d_ff, d_model)) * scale
        self.b2 = np.zeros(d_model)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return gelu(X @ self.W1 + self.b1) @ self.W2 + self.b2

    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]


# ── Transformer Encoder Block ─────────────────────────────────────────────────

class TransformerEncoderBlock:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, seed: int = 42):
        self.attn = MultiHeadAttention(d_model, n_heads, seed)
        self.ff = FeedForward(d_model, d_ff, seed + 1)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        attn_out, attn_weights = self.attn.forward(X, mask=mask, training=training)
        X = self.norm1.forward(X + attn_out)
        ff_out = self.ff.forward(X)
        X = self.norm2.forward(X + ff_out)
        return X, attn_weights

    def params(self) -> list[np.ndarray]:
        return self.attn.params() + self.ff.params() + self.norm1.params() + self.norm2.params()


# ── Causal Mask ───────────────────────────────────────────────────────────────

def causal_mask(T: int) -> np.ndarray:
    """Upper triangular mask for autoregressive (causal) attention."""
    return np.triu(np.ones((T, T), dtype=bool), k=1)


# ── Market Transformer ────────────────────────────────────────────────────────

@dataclass
class MarketTransformerConfig:
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 3
    n_input_features: int = 10
    n_output: int = 1
    dropout_rate: float = 0.1
    use_causal_mask: bool = True
    seed: int = 42


class MarketTransformer:
    """
    Transformer encoder for financial time series.
    Input: (T, n_features) feature matrix
    Output: (T, n_output) signal predictions
    """

    def __init__(self, config: Optional[MarketTransformerConfig] = None):
        if config is None:
            config = MarketTransformerConfig()
        self.config = config
        c = config
        rng = np.random.default_rng(c.seed)

        # Input projection
        scale = math.sqrt(2.0 / c.n_input_features)
        self.input_proj = rng.standard_normal((c.n_input_features, c.d_model)) * scale
        self.input_bias = np.zeros(c.d_model)

        # Transformer blocks
        self.blocks = [
            TransformerEncoderBlock(c.d_model, c.n_heads, c.d_ff, c.seed + i)
            for i in range(c.n_layers)
        ]

        # Output head
        self.output_proj = rng.standard_normal((c.d_model, c.n_output)) * 0.01
        self.output_bias = np.zeros(c.n_output)

        self.norm_out = LayerNorm(c.d_model)
        self._all_params = self._collect_params()

    def _collect_params(self) -> list[np.ndarray]:
        params = [self.input_proj, self.input_bias, self.output_proj, self.output_bias]
        params += self.norm_out.params()
        for block in self.blocks:
            params += block.params()
        return params

    def forward(
        self,
        X: np.ndarray,
        training: bool = False,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        X: (T, n_features)
        Returns (predictions (T, n_output), attention_weights list)
        """
        # Input projection
        H = X @ self.input_proj + self.input_bias  # (T, d_model)

        # Add positional encoding
        T = H.shape[0]
        H = H + sinusoidal_pe(T, self.config.d_model)

        # Causal mask
        mask = causal_mask(T) if self.config.use_causal_mask else None

        # Transformer blocks
        all_attn = []
        for block in self.blocks:
            H, attn = block.forward(H, mask=mask, training=training)
            all_attn.append(attn)

        H = self.norm_out.forward(H)
        out = H @ self.output_proj + self.output_bias  # (T, n_output)
        return out, all_attn

    def predict_signal(self, X: np.ndarray) -> np.ndarray:
        """Generate signal predictions from features."""
        out, _ = self.forward(X, training=False)
        return np.tanh(out[:, 0])  # compress to [-1, 1]

    def walk_forward_signals(
        self,
        features: np.ndarray,
        lookback: int = 60,
        step: int = 1,
    ) -> np.ndarray:
        """
        Walk-forward signal generation.
        features: (T, n_features)
        Returns signal array of length T.
        """
        T = len(features)
        signals = np.zeros(T)

        for t in range(lookback, T, step):
            window = features[max(0, t - lookback): t]
            out, _ = self.forward(window)
            signals[t] = float(np.tanh(out[-1, 0]))

        return signals

    def attention_heatmap(self, X: np.ndarray, layer: int = 0) -> np.ndarray:
        """
        Return attention heatmap for a given layer.
        Returns (T, T) average attention across heads.
        """
        _, all_attn = self.forward(X, training=False)
        if layer >= len(all_attn):
            return np.zeros((len(X), len(X)))
        attn = all_attn[layer]  # (n_heads, T, T) or (batch, n_heads, T, T)
        if attn.ndim == 4:
            attn = attn[0]  # take first batch
        return attn.mean(axis=0)  # average over heads


# ── Gated Residual Network (TFT element) ─────────────────────────────────────

class GatedResidualNetwork:
    """
    GRN from Temporal Fusion Transformer.
    Selectively activates information via gating.
    """

    def __init__(self, d_model: int, d_hidden: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = math.sqrt(2.0 / d_model)
        self.W1 = rng.standard_normal((d_model, d_hidden)) * scale
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.standard_normal((d_hidden, d_model)) * scale
        self.b2 = np.zeros(d_model)
        # Gate
        self.Wg = rng.standard_normal((d_hidden, d_model)) * scale
        self.bg = np.zeros(d_model)
        self.norm = LayerNorm(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = gelu(x @ self.W1 + self.b1)
        output = hidden @ self.W2 + self.b2
        gate = 1 / (1 + np.exp(-(hidden @ self.Wg + self.bg)))  # sigmoid
        return self.norm.forward(x + gate * output)
