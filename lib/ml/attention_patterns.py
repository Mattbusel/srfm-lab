"""
attention_patterns.py
=====================
Attention-based temporal pattern recognition for financial time series.

Implements from scratch using NumPy only (no PyTorch, no TensorFlow):

- ScaledDotProductAttention
- MultiHeadAttention       (Q, K, V projections + scaled dot-product + concat)
- PositionalEncoding       (sinusoidal)
- TemporalPatternExtractor (self-attention over returns for regime detection)
- CrossAssetAttention      (asset A's past vs. asset B's present)
- AttentionAnomalyDetector (unusual attention pattern → anomaly score)
- Attention heatmap        (returns weight matrix for visualisation)

All computations use float64 NumPy arrays for numerical stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along a given axis."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-12)


def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Layer normalisation: normalise along the last axis."""
    mean = np.mean(x, axis=-1, keepdims=True)
    std  = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation approximation."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _xavier_init(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Xavier/Glorot uniform initialisation."""
    fan_in, fan_out = shape[0], shape[1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


# ──────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────

class PositionalEncoding:
    """
    Sinusoidal positional encoding (Vaswani et al. 2017).

    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

    Encodes the relative temporal position of each time step in a
    financial sequence so the model can distinguish lag-0 from lag-k.
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        self.d_model = d_model
        self.max_len = max_len
        self._pe = self._build_encoding(max_len, d_model)

    @staticmethod
    def _build_encoding(max_len: int, d_model: int) -> np.ndarray:
        pos = np.arange(max_len)[:, None]              # (max_len, 1)
        i   = np.arange(d_model)[None, :]              # (1, d_model)
        angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
        pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
        return pe.astype(np.float64)                   # (max_len, d_model)

    def encode(self, T: int) -> np.ndarray:
        """Return positional encoding for T time steps: shape (T, d_model)."""
        return self._pe[:T]

    def add_to(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding to input tensor x of shape (T, d_model)."""
        T = x.shape[0]
        return x + self._pe[:T]


# ──────────────────────────────────────────────────────────────────────────────
# Scaled Dot-Product Attention
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AttentionOutput:
    """Output of a single attention computation."""
    values: np.ndarray      # shape (T_q, d_v) — weighted value sum
    weights: np.ndarray     # shape (T_q, T_k) — attention weight matrix


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> AttentionOutput:
    """
    Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V

    Parameters
    ----------
    Q    : query  matrix, shape (T_q, d_k)
    K    : key    matrix, shape (T_k, d_k)
    V    : value  matrix, shape (T_k, d_v)
    mask : optional boolean mask, shape (T_q, T_k); True = mask out (set to -∞)

    Returns
    -------
    AttentionOutput with .values shape (T_q, d_v) and .weights shape (T_q, T_k)
    """
    d_k = Q.shape[-1]
    scale = math.sqrt(d_k)

    # Scores: (T_q, d_k) @ (d_k, T_k) = (T_q, T_k)
    scores = Q @ K.T / scale

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    weights = _softmax(scores, axis=-1)           # (T_q, T_k)
    values  = weights @ V                          # (T_q, d_v)

    return AttentionOutput(values=values, weights=weights)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Head Attention
# ──────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention:
    """
    Multi-head attention (Vaswani et al. 2017) — NumPy implementation.

    Each head projects Q, K, V to lower-dimensional subspaces,
    computes scaled dot-product attention, then concatenates and projects back.

    Parameters
    ----------
    d_model  : total dimensionality of the model
    n_heads  : number of attention heads (must divide d_model evenly)
    seed     : random seed for weight initialisation
    """

    def __init__(self, d_model: int, n_heads: int, seed: int = 0):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.d_v     = d_model // n_heads

        rng = np.random.default_rng(seed)
        # Projection weights: (d_model, d_k * n_heads) = (d_model, d_model)
        self.W_Q = _xavier_init((d_model, d_model), rng)
        self.W_K = _xavier_init((d_model, d_model), rng)
        self.W_V = _xavier_init((d_model, d_model), rng)
        self.W_O = _xavier_init((d_model, d_model), rng)

        self._last_weights: list[np.ndarray] = []   # stored per head

    def forward(
        self,
        query:  np.ndarray,
        key:    np.ndarray,
        value:  np.ndarray,
        mask:   np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Multi-head attention forward pass.

        Parameters
        ----------
        query  : (T_q, d_model)
        key    : (T_k, d_model)
        value  : (T_k, d_model)
        mask   : optional (T_q, T_k) boolean mask

        Returns
        -------
        output       : (T_q, d_model)
        head_weights : list of (T_q, T_k) weight matrices, one per head
        """
        T_q = query.shape[0]
        T_k = key.shape[0]

        # Linear projections
        Q_proj = query @ self.W_Q    # (T_q, d_model)
        K_proj = key   @ self.W_K    # (T_k, d_model)
        V_proj = value @ self.W_V    # (T_k, d_model)

        head_outputs: list[np.ndarray] = []
        head_weights: list[np.ndarray] = []

        for h in range(self.n_heads):
            start = h * self.d_k
            end   = start + self.d_k

            Q_h = Q_proj[:, start:end]   # (T_q, d_k)
            K_h = K_proj[:, start:end]   # (T_k, d_k)
            V_h = V_proj[:, start:end]   # (T_k, d_v)

            attn_out = scaled_dot_product_attention(Q_h, K_h, V_h, mask)
            head_outputs.append(attn_out.values)    # (T_q, d_v)
            head_weights.append(attn_out.weights)   # (T_q, T_k)

        # Concatenate all heads: (T_q, d_model)
        concat = np.concatenate(head_outputs, axis=-1)

        # Final linear projection
        output = concat @ self.W_O   # (T_q, d_model)

        self._last_weights = head_weights
        return output, head_weights

    def __call__(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        return self.forward(query, key, value, mask)

    def get_last_weights(self) -> list[np.ndarray]:
        """Return attention weight matrices from the last forward pass."""
        return self._last_weights

    def average_head_weights(self) -> np.ndarray | None:
        """Return mean attention weight matrix averaged over all heads."""
        if not self._last_weights:
            return None
        return np.mean(np.stack(self._last_weights, axis=0), axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Feed-forward sub-layer (for completeness / Transformer block)
# ──────────────────────────────────────────────────────────────────────────────

class FeedForward:
    """
    Position-wise feed-forward sub-layer:
        FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
    """

    def __init__(self, d_model: int, d_ff: int, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.W1 = _xavier_init((d_model, d_ff), rng)
        self.b1 = np.zeros(d_ff)
        self.W2 = _xavier_init((d_ff, d_model), rng)
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return _gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# ──────────────────────────────────────────────────────────────────────────────
# Input embedding for financial time series
# ──────────────────────────────────────────────────────────────────────────────

class FinancialEmbedding:
    """
    Projects a univariate or multivariate return series into d_model dimensions.

    Input  : (T, n_features)
    Output : (T, d_model)
    """

    def __init__(self, n_features: int, d_model: int, seed: int = 2):
        rng = np.random.default_rng(seed)
        self.W = _xavier_init((n_features, d_model), rng)
        self.b = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# ──────────────────────────────────────────────────────────────────────────────
# TemporalPatternExtractor
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimePattern:
    """Extracted temporal pattern for a window of returns."""
    t_start: int
    t_end: int
    attention_entropy: float    # how diffuse is the attention? (low = focused regime)
    dominant_lag: int           # lag with highest average attention
    concentration: float        # top-1 attention weight (how peaked is focus)
    feature_vector: np.ndarray  # compressed representation (d_model,)
    regime_label: str           # heuristic label: "trending", "volatile", "mean-reverting"


class TemporalPatternExtractor:
    """
    Applies self-attention to a return series to detect temporal regime patterns.

    The key insight: in a trending market, recent returns receive high attention
    from all query positions (attention concentrates on recent lags).
    In a mean-reverting market, attention focuses on distant past turning points.
    In a volatile/noisy market, attention is diffuse (high entropy).

    Architecture
    ------------
    x (T, n_features)
    → FinancialEmbedding → (T, d_model)
    → + PositionalEncoding
    → MultiHeadAttention (self-attention: Q=K=V=x)
    → + residual → LayerNorm
    → FeedForward → + residual → LayerNorm
    → pooled output + attention weights for analysis
    """

    def __init__(
        self,
        n_features: int = 1,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 64,
        seed: int = 0,
    ):
        self.d_model    = d_model
        self.n_features = n_features
        self.embedding  = FinancialEmbedding(n_features, d_model, seed)
        self.pos_enc    = PositionalEncoding(d_model)
        self.mha        = MultiHeadAttention(d_model, n_heads, seed)
        self.ff         = FeedForward(d_model, d_ff, seed + 1)

    def encode(
        self,
        returns: np.ndarray,
        causal_mask: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode a (T, n_features) or (T,) return series.

        Parameters
        ----------
        returns      : shape (T,) or (T, n_features)
        causal_mask  : if True, apply causal (look-ahead) mask

        Returns
        -------
        encoded      : (T, d_model) context representations
        avg_weights  : (T, T) averaged attention weight matrix
        """
        if returns.ndim == 1:
            returns = returns[:, None]

        T = returns.shape[0]

        # Embed + positional encoding
        x = self.embedding(returns)          # (T, d_model)
        x = self.pos_enc.add_to(x)           # (T, d_model)

        # Causal mask: upper-triangular = True (mask future positions)
        mask = None
        if causal_mask:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)

        # Self-attention: Q = K = V = x
        attn_out, weights_per_head = self.mha(x, x, x, mask)

        # Residual + layer norm
        x = _layer_norm(x + attn_out)

        # Feed-forward + residual + layer norm
        x = _layer_norm(x + self.ff(x))

        # Average attention weights over heads: (T, T)
        avg_w = np.mean(np.stack(weights_per_head, axis=0), axis=0)

        return x, avg_w

    def extract_patterns(
        self,
        returns: np.ndarray,
        window: int = 63,
        step: int = 21,
    ) -> list[RegimePattern]:
        """
        Slide a window over a return series and extract a RegimePattern per window.

        Parameters
        ----------
        returns : shape (T,)  — univariate return series
        window  : rolling window size
        step    : stride between windows
        """
        T = len(returns)
        patterns: list[RegimePattern] = []

        for t_start in range(0, T - window, step):
            t_end = t_start + window
            window_rets = returns[t_start:t_end]

            encoded, attn_weights = self.encode(window_rets, causal_mask=True)

            # Summarise attention pattern
            # Use the last query position (current time) attention weights
            last_query_weights = attn_weights[-1]        # (window,)
            entropy     = _attention_entropy(last_query_weights)
            dominant    = int(np.argmax(last_query_weights))
            dominant_lag = (window - 1) - dominant       # convert to lag
            conc        = float(np.max(last_query_weights))

            # Mean-pool the encoded representation
            feature_vec = np.mean(encoded, axis=0)       # (d_model,)

            regime = _classify_regime(entropy, dominant_lag, conc, window_rets)

            patterns.append(RegimePattern(
                t_start=t_start,
                t_end=t_end,
                attention_entropy=float(entropy),
                dominant_lag=dominant_lag,
                concentration=conc,
                feature_vector=feature_vec,
                regime_label=regime,
            ))

        return patterns

    def compute_self_relevance(
        self,
        returns: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-time-step self-relevance score:
        "which past periods are most relevant to the current period?"

        Returns a (T,) array where entry t is the mean attention weight
        that the last observation places on time t.
        """
        _, attn_weights = self.encode(returns, causal_mask=True)
        # attn_weights[-1]: the current step's attention over all past steps
        return attn_weights[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Asset Attention
# ──────────────────────────────────────────────────────────────────────────────

class CrossAssetAttention:
    """
    Cross-asset temporal attention:
    "How correlated is asset A's past with asset B's present?"

    Uses cross-attention: queries from asset B (target), keys/values from
    asset A (source).  High attention weight = past A strongly predicts
    current B at that lag.

    Architecture
    ------------
    returns_A (T, 1) → embed → (T, d_model)  [source: keys, values]
    returns_B (T, 1) → embed → (T, d_model)  [target: queries]
    Cross-attention: Q=B, K=A, V=A
    → attention weight matrix (T, T): B[t] ← A[τ] for τ < t
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        seed: int = 3,
    ):
        self.d_model   = d_model
        self.embed_A   = FinancialEmbedding(1, d_model, seed)
        self.embed_B   = FinancialEmbedding(1, d_model, seed + 1)
        self.pos_enc   = PositionalEncoding(d_model)
        self.mha       = MultiHeadAttention(d_model, n_heads, seed + 2)

    def compute_cross_attention(
        self,
        returns_A: np.ndarray,
        returns_B: np.ndarray,
        causal_mask: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute cross-attention between two return series.

        Parameters
        ----------
        returns_A : (T,) source series
        returns_B : (T,) target series
        causal_mask : mask future positions of A when querying from B

        Returns
        -------
        output   : (T, d_model)
        weights  : (T, T) cross-attention weight matrix [avg over heads]
        """
        T = min(len(returns_A), len(returns_B))
        rA = returns_A[:T, None].astype(np.float64)
        rB = returns_B[:T, None].astype(np.float64)

        emb_A = self.pos_enc.add_to(self.embed_A(rA))   # (T, d_model)
        emb_B = self.pos_enc.add_to(self.embed_B(rB))   # (T, d_model)

        mask = None
        if causal_mask:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)

        output, weights_per_head = self.mha(
            query=emb_B, key=emb_A, value=emb_A, mask=mask
        )
        avg_weights = np.mean(np.stack(weights_per_head, axis=0), axis=0)
        return output, avg_weights

    def leading_lag_profile(
        self,
        returns_A: np.ndarray,
        returns_B: np.ndarray,
        max_lag: int = 20,
    ) -> np.ndarray:
        """
        Extract a leading-lag profile: the average attention that B's present
        places on A's lag-k past, for k in [0, max_lag].

        Returns
        -------
        profile : (max_lag + 1,) — normalised average attention by lag
        """
        _, weights = self.compute_cross_attention(returns_A, returns_B)
        T = weights.shape[0]
        profile = np.zeros(max_lag + 1)

        # For each query position t, the weight weights[t, t-k] is the
        # attention on lag k
        for k in range(max_lag + 1):
            vals = []
            for t in range(k, T):
                vals.append(weights[t, t - k])
            if vals:
                profile[k] = float(np.mean(vals))

        # Normalise
        s = profile.sum()
        if s > 1e-12:
            profile /= s
        return profile


# ──────────────────────────────────────────────────────────────────────────────
# Attention-Based Anomaly Detector
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    """Result of attention-based anomaly detection."""
    t: int
    anomaly_score: float         # higher = more anomalous
    attention_entropy: float
    kl_from_baseline: float      # KL divergence from baseline attention pattern
    is_anomaly: bool
    context_window: tuple[int, int]


class AttentionAnomalyDetector:
    """
    Anomaly detection via unusual attention patterns.

    Intuition: in normal market conditions, the self-attention pattern has
    a characteristic shape (e.g., most weight on recent lags). An anomalous
    period produces an unusual attention weight distribution.

    Method
    ------
    1. Compute a 'baseline' attention distribution from a training window
       (first half of the series).
    2. For each new window, compute attention and score via:
       - KL divergence from the baseline attention distribution
       - Entropy change (unusually concentrated or diffuse)
       - Euclidean distance between encoded feature vectors
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        seed: int = 4,
    ):
        self.extractor = TemporalPatternExtractor(
            n_features=1, d_model=d_model, n_heads=n_heads, seed=seed
        )
        self._baseline_weights: np.ndarray | None = None
        self._baseline_entropy: float = 0.0
        self._baseline_features: np.ndarray | None = None

    def fit(self, returns: np.ndarray, train_fraction: float = 0.5) -> None:
        """
        Fit the baseline attention distribution from the training portion.

        Parameters
        ----------
        returns        : full return series (T,)
        train_fraction : fraction to use as training baseline
        """
        T_train = int(len(returns) * train_fraction)
        train_rets = returns[:T_train]

        _, avg_weights = self.extractor.encode(train_rets, causal_mask=True)

        # Baseline: mean attention distribution across all query positions
        # Shape: (T_train,) — the mean attention weight a typical query places
        # on each key position
        self._baseline_weights = np.mean(avg_weights, axis=0)
        self._baseline_weights = self._baseline_weights / (
            self._baseline_weights.sum() + 1e-12
        )
        self._baseline_entropy = float(_attention_entropy(self._baseline_weights))

        # Baseline feature
        encoded, _ = self.extractor.encode(train_rets, causal_mask=True)
        self._baseline_features = np.mean(encoded, axis=0)

    def score(
        self,
        returns: np.ndarray,
        window: int = 63,
        step: int = 1,
        threshold_percentile: float = 95.0,
    ) -> list[AnomalyResult]:
        """
        Score each window of the return series for anomalousness.

        Parameters
        ----------
        returns              : (T,) return series (full, including test portion)
        window               : rolling window for each score computation
        step                 : stride between scored windows
        threshold_percentile : percentile of scores to define anomaly threshold

        Returns
        -------
        List of AnomalyResult, one per window.
        """
        if self._baseline_weights is None:
            raise RuntimeError("Call .fit() before .score()")

        T = len(returns)
        raw_scores: list[float] = []
        results_raw: list[tuple[int, float, float, float]] = []

        for t in range(window, T, step):
            window_rets = returns[t - window:t]
            _, attn_w = self.extractor.encode(window_rets, causal_mask=True)

            # Current attention distribution (last query position)
            curr_dist = attn_w[-1]
            curr_dist = curr_dist / (curr_dist.sum() + 1e-12)

            # Align lengths for KL computation
            b = self._baseline_weights
            L = min(len(curr_dist), len(b))
            c_trimmed = curr_dist[:L] / (curr_dist[:L].sum() + 1e-12)
            b_trimmed = b[:L] / (b[:L].sum() + 1e-12)

            kl_div = float(np.sum(
                c_trimmed * np.log((c_trimmed + 1e-12) / (b_trimmed + 1e-12))
            ))

            curr_entropy = float(_attention_entropy(curr_dist))
            entropy_delta = abs(curr_entropy - self._baseline_entropy)

            # Composite score
            score = kl_div + 0.5 * entropy_delta
            raw_scores.append(score)
            results_raw.append((t, score, curr_entropy, kl_div))

        if not raw_scores:
            return []

        threshold = float(np.percentile(raw_scores, threshold_percentile))

        results: list[AnomalyResult] = []
        for t, score, entropy, kl in results_raw:
            results.append(AnomalyResult(
                t=t,
                anomaly_score=round(score, 6),
                attention_entropy=round(entropy, 6),
                kl_from_baseline=round(kl, 6),
                is_anomaly=score >= threshold,
                context_window=(t - window, t),
            ))

        return results

    def get_anomaly_dates(
        self, results: list[AnomalyResult]
    ) -> list[int]:
        """Return time indices of detected anomalies."""
        return [r.t for r in results if r.is_anomaly]


# ──────────────────────────────────────────────────────────────────────────────
# Attention visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def attention_heatmap_ascii(
    weights: np.ndarray,
    labels_q: list[str] | None = None,
    labels_k: list[str] | None = None,
    n_chars: int = 12,
) -> str:
    """
    Render an attention weight matrix as an ASCII heatmap.

    Characters used for weight levels (low → high):
        · ░ ▒ ▓ █

    Parameters
    ----------
    weights  : (T_q, T_k) attention weight matrix
    labels_q : optional query axis labels
    labels_k : optional key axis labels
    n_chars  : number of columns to display (slices to last n_chars)
    """
    chars = [" ", "·", "░", "▒", "▓", "█"]
    T_q, T_k = weights.shape
    T_q_show = min(T_q, n_chars)
    T_k_show = min(T_k, n_chars)
    w = weights[-T_q_show:, -T_k_show:]

    max_w = w.max() + 1e-12
    lines = []

    # Header
    if labels_k:
        header = "  " + " ".join(f"{lbl[:3]:>3}" for lbl in labels_k[-T_k_show:])
        lines.append(header)

    for i in range(T_q_show):
        row_label = labels_q[-T_q_show + i][:3] if labels_q else f"{T_q - T_q_show + i:3d}"
        row = f"{row_label} "
        for j in range(T_k_show):
            level = int(w[i, j] / max_w * (len(chars) - 1))
            row += chars[level] + " "
        lines.append(row)

    return "\n".join(lines)


def top_attention_lags(
    weights: np.ndarray, top_k: int = 5
) -> list[tuple[int, float]]:
    """
    For the final query position (current time), return top-k (lag, weight) pairs.

    Parameters
    ----------
    weights : (T, T) attention weight matrix (causal)

    Returns
    -------
    List of (lag, weight) sorted by weight descending.
    """
    last_row = weights[-1]
    T = len(last_row)
    # lag k means the key position was at index (T - 1 - k)
    lag_weight: list[tuple[int, float]] = [
        (T - 1 - i, float(last_row[i])) for i in range(T)
    ]
    lag_weight.sort(key=lambda x: x[1], reverse=True)
    return lag_weight[:top_k]


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _attention_entropy(weights: np.ndarray) -> float:
    """Shannon entropy of an attention distribution."""
    w = np.clip(weights, 1e-12, None)
    w = w / w.sum()
    return float(-np.sum(w * np.log(w)))


def _classify_regime(
    entropy: float,
    dominant_lag: int,
    concentration: float,
    returns: np.ndarray,
) -> str:
    """
    Heuristic regime classification from attention statistics.

    Rules
    -----
    - Low entropy + small dominant lag → trending (recent returns matter most)
    - Low entropy + large dominant lag → mean-reverting (distant past matters)
    - High entropy                     → volatile / noisy
    - High concentration               → regime-change focus
    """
    T = len(returns)
    max_entropy = math.log(max(T, 1))
    normalised_entropy = entropy / (max_entropy + 1e-12)

    if normalised_entropy > 0.7:
        return "volatile"
    if dominant_lag <= T // 5:
        return "trending"
    if dominant_lag >= T * 3 // 4:
        return "mean-reverting"
    return "transitional"


# ──────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ──────────────────────────────────────────────────────────────────────────────

def _demo():
    rng = np.random.default_rng(42)
    T = 500

    # Simulate a return series with 3 regimes: trend, noisy, mean-reverting
    trend_rets  = 0.001 + rng.standard_normal(150) * 0.01
    noisy_rets  = rng.standard_normal(200) * 0.03
    mr_rets     = -0.5 * np.roll(rng.standard_normal(150) * 0.02, 1)[:150]
    returns     = np.concatenate([trend_rets, noisy_rets, mr_rets])

    print("=== TemporalPatternExtractor ===")
    extractor = TemporalPatternExtractor(n_features=1, d_model=16, n_heads=4, seed=0)
    patterns  = extractor.extract_patterns(returns, window=50, step=25)
    print(f"Extracted {len(patterns)} regime patterns")
    for p in patterns[:5]:
        print(f"  [{p.t_start:3d}-{p.t_end:3d}] "
              f"regime={p.regime_label:<15} "
              f"entropy={p.attention_entropy:.3f}  "
              f"dominant_lag={p.dominant_lag:3d}  "
              f"concentration={p.concentration:.3f}")

    print("\n=== Self-relevance scores (last 10 positions) ===")
    relevance = extractor.compute_self_relevance(returns[-63:])
    top_lags  = sorted(enumerate(relevance), key=lambda x: x[1], reverse=True)[:5]
    for lag_idx, weight in top_lags:
        print(f"  Position {lag_idx:3d} (lag {62 - lag_idx}): weight={weight:.4f}")

    print("\n=== CrossAssetAttention ===")
    A = rng.standard_normal(T) * 0.01
    B = np.roll(A, 3) + rng.standard_normal(T) * 0.005   # B lags A by 3
    cross_attn = CrossAssetAttention(d_model=16, n_heads=4, seed=3)
    profile    = cross_attn.leading_lag_profile(A, B, max_lag=10)
    print("Leading-lag profile (A→B, normalised):")
    for lag, w in enumerate(profile):
        bar = "█" * int(w * 50)
        print(f"  lag {lag:2d}: {w:.4f} {bar}")

    print("\n=== AttentionAnomalyDetector ===")
    detector = AttentionAnomalyDetector(d_model=16, n_heads=4, seed=4)
    detector.fit(returns, train_fraction=0.5)
    anomaly_results = detector.score(returns, window=50, step=10)
    anomaly_times   = detector.get_anomaly_dates(anomaly_results)
    print(f"Detected {len(anomaly_times)} anomalies out of "
          f"{len(anomaly_results)} scored windows")
    print(f"Anomaly time indices: {anomaly_times[:10]}")

    # Attention heatmap
    print("\n=== Attention heatmap (last 12×12 of self-attention) ===")
    _, attn_weights = extractor.encode(returns[-80:], causal_mask=True)
    print(attention_heatmap_ascii(attn_weights, n_chars=12))


if __name__ == "__main__":
    _demo()
