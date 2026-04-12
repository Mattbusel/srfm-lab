"""
compression.py — Neural network model compression using Tensor Trains (Project AETERNUS).

Implements:
  - TT-Linear: dense layer compressed to TT format
  - TT-LSTM: LSTM with TT-compressed weight matrices
  - TT-Embedding: embedding layer in TT format
  - TT-Conv: 2D convolution with TT-compressed filters
  - TT-Attention: attention mechanism with TT weight compression
  - Flax module wrappers for all TT layers
  - Compression utilities: compress existing models, estimate compression ratios
  - TT initialization from pre-trained weights
  - Fine-tuning utilities for TT-compressed models
  - Quantization + TT-compression pipeline
  - Rank selection for target compression ratio
"""

from __future__ import annotations

import math
import functools
from typing import List, Optional, Tuple, Sequence, Union, Dict, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad, vmap

import flax.linen as nn
import optax

from .tt_decomp import (
    TensorTrain, TensorTrainMatrix,
    tt_svd, tt_round, tt_normalize, tt_from_matrix,
    tt_to_dense, tt_matvec
)


# ============================================================================
# TT-Linear Layer (Flax module)
# ============================================================================

class TTLinear(nn.Module):
    """
    TT-Linear: dense linear layer with weights in Tensor Train format.

    Replaces a (m, n) weight matrix W with a TT-matrix factorization:
      W ≈ TTM with cores G_k of shape (r_{k-1}, m_k, n_k, r_k)

    This reduces parameters from m*n to sum_k r_{k-1} * m_k * n_k * r_k.

    Attributes
    ----------
    row_shape : factored row shape (m_1, ..., m_d) s.t. prod = m
    col_shape : factored col shape (n_1, ..., n_d) s.t. prod = n
    tt_rank : TT-rank (uniform across bonds)
    use_bias : whether to add a bias term
    """

    row_shape: Tuple[int, ...]
    col_shape: Tuple[int, ...]
    tt_rank: int = 4
    use_bias: bool = True
    kernel_init: Any = nn.initializers.glorot_uniform()
    bias_init: Any = nn.initializers.zeros

    @property
    def in_features(self) -> int:
        result = 1
        for n in self.col_shape:
            result *= n
        return result

    @property
    def out_features(self) -> int:
        result = 1
        for m in self.row_shape:
            result *= m
        return result

    @property
    def n_modes(self) -> int:
        return len(self.row_shape)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x : (..., in_features) input tensor

        Returns
        -------
        (..., out_features) output tensor
        """
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        batch = x_flat.shape[0]

        # TT-matrix cores
        cores = []
        n_modes = self.n_modes
        for k in range(n_modes):
            r_l = 1 if k == 0 else self.tt_rank
            r_r = 1 if k == n_modes - 1 else self.tt_rank
            m_k = self.row_shape[k]
            n_k = self.col_shape[k]

            G = self.param(
                f"tt_core_{k}",
                nn.initializers.normal(stddev=1.0 / math.sqrt(r_l * m_k * n_k * r_r)),
                (r_l, m_k, n_k, r_r),
            )
            cores.append(G)

        # Contract TT-matrix with input vector
        # Reshape input to (batch, n_1, n_2, ..., n_d)
        x_shaped = x_flat.reshape([batch] + list(self.col_shape))

        # Sequential contraction
        # y[b, m_1, ..., m_d] = sum_{n_1,...,n_d} TTM[m_1,...,m_d, n_1,...,n_d] * x[b, n_1,...,n_d]
        result = x_shaped  # (batch, n_1, ..., n_d)

        # Contract mode by mode
        # After k contractions: (batch, m_1, ..., m_k, n_{k+1}, ..., n_d, r_k)
        r = jnp.ones((batch, 1), dtype=x_flat.dtype)  # running bond (batch, r_l)

        for k in range(n_modes):
            G = cores[k]  # (r_l, m_k, n_k, r_r)
            r_l, m_k, n_k, r_r = G.shape

            # Current mode of result: extract n_k axis
            # result shape: (batch, ..., n_k, ..., remaining)
            # Pull n_k to position 1
            n_k_actual = self.col_shape[k]

            # Extract the k-th mode
            n_remaining = result.shape[1] if result.ndim > 1 else 1
            x_k = result.reshape(batch, n_k_actual, -1)  # (batch, n_k, rest)

            # Contract with G: sum_{n_k} G[r_l, m_k, n_k, r_r] * x_k[b, n_k, rest] * r[b, r_l]
            # -> out[b, m_k, rest, r_r]
            tmp = jnp.einsum("br,rmnR,bnd->bmdR", r, G, x_k)  # (batch, m_k, rest, r_r)
            batch_r, m, d_rest, r_right = tmp.shape
            r = tmp.reshape(batch_r, m * d_rest, r_right).reshape(batch_r, -1, r_right)
            # Keep r as (batch, output_so_far, r_right) - need to handle carefully

            # Simplified: just reshape at each step
            if k < n_modes - 1:
                r = tmp.reshape(batch, m_k, -1, r_r)
                # For next iteration
                result = tmp.reshape(batch, -1, r_r)[:, :, :]
                r = result[:, :, -r_right:] if result.shape[-1] >= r_right else result
            else:
                # Last mode: r_r = 1
                r = tmp.reshape(batch, -1, 1)

        # r has shape (batch, prod(row_shape), 1) approximately
        # Final output
        out = r.reshape(batch, self.out_features) if r.shape[1] == self.out_features else \
              self._dense_fallback(x_flat, cores)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.out_features,))
            out = out + bias

        return out.reshape(batch_shape + (self.out_features,))

    def _dense_fallback(self, x: jnp.ndarray, cores: List[jnp.ndarray]) -> jnp.ndarray:
        """Fallback: reconstruct dense matrix and apply."""
        # Build the full weight matrix from TT cores
        ttm = TensorTrainMatrix(cores, self.row_shape, self.col_shape)
        W = ttm.to_dense()  # (out_features, in_features)
        return x @ W.T


class TTLinearSimple(nn.Module):
    """
    Simpler TT-Linear that always uses dense fallback reconstruction.
    More numerically stable, slightly less efficient.
    """

    row_shape: Tuple[int, ...]
    col_shape: Tuple[int, ...]
    tt_rank: int = 4
    use_bias: bool = True

    @property
    def in_features(self) -> int:
        result = 1
        for n in self.col_shape:
            result *= n
        return result

    @property
    def out_features(self) -> int:
        result = 1
        for m in self.row_shape:
            result *= m
        return result

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        n_modes = len(self.row_shape)
        cores = []
        for k in range(n_modes):
            r_l = 1 if k == 0 else self.tt_rank
            r_r = 1 if k == n_modes - 1 else self.tt_rank
            m_k = self.row_shape[k]
            n_k = self.col_shape[k]
            scale = 1.0 / math.sqrt(r_l * m_k * n_k * r_r + 1)
            G = self.param(
                f"core_{k}",
                nn.initializers.normal(stddev=scale),
                (r_l, m_k, n_k, r_r),
            )
            cores.append(G)

        # Reconstruct dense weight matrix
        ttm = TensorTrainMatrix(cores, self.row_shape, self.col_shape)
        W = ttm.to_dense()  # (out_features, in_features)

        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        out = x_flat @ W.T
        out = out.reshape(batch_shape + (self.out_features,))

        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.out_features,))
            out = out + bias
        return out

    def n_params(self) -> int:
        n_modes = len(self.row_shape)
        count = 0
        for k in range(n_modes):
            r_l = 1 if k == 0 else self.tt_rank
            r_r = 1 if k == n_modes - 1 else self.tt_rank
            count += r_l * self.row_shape[k] * self.col_shape[k] * r_r
        if self.use_bias:
            count += self.out_features
        return count

    def compression_ratio(self) -> float:
        dense_params = self.in_features * self.out_features + (self.out_features if self.use_bias else 0)
        return dense_params / (self.n_params() + 1e-10)


# ============================================================================
# TT-Embedding Layer
# ============================================================================

class TTEmbedding(nn.Module):
    """
    TT-Embedding: embedding lookup table in TT format.

    Replaces an (vocab_size, embed_dim) embedding matrix with a TT-matrix
    of shape (vocab_shape, embed_shape) where:
      vocab_size = prod(vocab_shape)
      embed_dim  = prod(embed_shape)

    Parameters from: Hrinchuk et al. (2020) "Tensorized Embedding Layers."

    Attributes
    ----------
    vocab_size : total vocabulary size
    embed_dim : embedding dimension
    vocab_shape : factored vocab shape
    embed_shape : factored embed shape
    tt_rank : TT-rank
    """

    vocab_size: int
    embed_dim: int
    vocab_shape: Tuple[int, ...]
    embed_shape: Tuple[int, ...]
    tt_rank: int = 16
    max_norm: Optional[float] = None

    @nn.compact
    def __call__(self, indices: jnp.ndarray) -> jnp.ndarray:
        """
        Look up embeddings for given indices.

        Parameters
        ----------
        indices : integer array of shape (...)

        Returns
        -------
        Embedding array of shape (..., embed_dim)
        """
        n_modes = len(self.vocab_shape)
        assert len(self.embed_shape) == n_modes

        cores = []
        for k in range(n_modes):
            r_l = 1 if k == 0 else self.tt_rank
            r_r = 1 if k == n_modes - 1 else self.tt_rank
            v_k = self.vocab_shape[k]
            e_k = self.embed_shape[k]
            scale = 1.0 / math.sqrt(r_l * v_k * e_k * r_r + 1)
            G = self.param(
                f"emb_core_{k}",
                nn.initializers.normal(stddev=scale),
                (r_l, v_k, e_k, r_r),
            )
            cores.append(G)

        # Build full embedding matrix
        ttm = TensorTrainMatrix(cores, self.embed_shape, self.vocab_shape)
        W = ttm.to_dense().T  # (vocab_size, embed_dim)

        # Normalize if max_norm specified
        if self.max_norm is not None:
            norms = jnp.linalg.norm(W, axis=1, keepdims=True)
            W = jnp.where(norms > self.max_norm, W / norms * self.max_norm, W)

        return W[indices]

    def n_params(self) -> int:
        n_modes = len(self.vocab_shape)
        count = 0
        for k in range(n_modes):
            r_l = 1 if k == 0 else self.tt_rank
            r_r = 1 if k == n_modes - 1 else self.tt_rank
            count += r_l * self.vocab_shape[k] * self.embed_shape[k] * r_r
        return count

    def compression_ratio(self) -> float:
        return (self.vocab_size * self.embed_dim) / (self.n_params() + 1e-10)


# ============================================================================
# TT-LSTM
# ============================================================================

class TTLSTMCell(nn.Module):
    """
    LSTM cell with TT-compressed weight matrices.

    The standard LSTM uses (input_size + hidden_size, 4 * hidden_size)
    weight matrix. We compress it as a TT-matrix.

    Attributes
    ----------
    input_size : input dimension
    hidden_size : hidden state dimension
    input_shape : factored input shape
    hidden_shape : factored hidden shape
    tt_rank : TT-rank
    """

    input_size: int
    hidden_size: int
    input_shape: Tuple[int, ...]
    hidden_shape: Tuple[int, ...]
    tt_rank: int = 4

    @nn.compact
    def __call__(
        self,
        carry: Tuple[jnp.ndarray, jnp.ndarray],
        x: jnp.ndarray,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        One LSTM step.

        Parameters
        ----------
        carry : (h, c) hidden state and cell state
        x : (batch, input_size) input

        Returns
        -------
        ((h_new, c_new), h_new)
        """
        h, c = carry
        batch = x.shape[0]

        # Concatenate input and hidden state
        inp = jnp.concatenate([x, h], axis=-1)  # (batch, input_size + hidden_size)

        # TT-compressed gate computation (4 gates: i, f, g, o)
        # Row shape: 4 * hidden_size (factored)
        # Col shape: input_size + hidden_size (factored)

        # For simplicity, use TTLinearSimple for each gate
        # (Full LSTM uses one matrix; here we use 4 separate for clarity)
        def gate_linear(name_suffix, inp_tensor):
            return TTLinearSimple(
                row_shape=self.hidden_shape,
                col_shape=self.input_shape,
                tt_rank=self.tt_rank,
                use_bias=True,
                name=f"gate_{name_suffix}",
            )(inp_tensor)

        i_gate = jax.nn.sigmoid(gate_linear("i", inp))
        f_gate = jax.nn.sigmoid(gate_linear("f", inp))
        g_gate = jnp.tanh(gate_linear("g", inp))
        o_gate = jax.nn.sigmoid(gate_linear("o", inp))

        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * jnp.tanh(c_new)

        return (h_new, c_new), h_new

    def initialize_carry(
        self,
        batch_size: int,
        dtype=jnp.float32,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize hidden and cell states to zero."""
        h = jnp.zeros((batch_size, self.hidden_size), dtype=dtype)
        c = jnp.zeros((batch_size, self.hidden_size), dtype=dtype)
        return h, c


class TTLSTM(nn.Module):
    """
    Multi-layer TT-LSTM for sequence modeling.

    Applies TTLSTMCell sequentially over a sequence.
    """

    input_size: int
    hidden_size: int
    n_layers: int = 2
    tt_rank: int = 4
    dropout_rate: float = 0.1

    @property
    def input_shape(self) -> Tuple[int, ...]:
        # Factor input_size + hidden_size
        total = self.input_size + self.hidden_size
        return _factorize(total)

    @property
    def hidden_shape(self) -> Tuple[int, ...]:
        return _factorize(self.hidden_size)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, Tuple]:
        """
        Forward pass over a sequence.

        Parameters
        ----------
        x : (batch, seq_len, input_size)
        training : dropout mode

        Returns
        -------
        (outputs, final_states) where outputs has shape (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = x.shape

        all_outputs = []
        final_states = []

        h = x  # Input to first layer
        for layer_idx in range(self.n_layers):
            cell = TTLSTMCell(
                input_size=h.shape[-1],
                hidden_size=self.hidden_size,
                input_shape=_factorize(h.shape[-1] + self.hidden_size),
                hidden_shape=self.hidden_shape,
                tt_rank=self.tt_rank,
                name=f"lstm_layer_{layer_idx}",
            )

            carry = cell.initialize_carry(batch)
            outputs = []

            for t in range(seq_len):
                carry, out = cell(carry, h[:, t, :])
                outputs.append(out)

            h = jnp.stack(outputs, axis=1)  # (batch, seq_len, hidden_size)

            if training and self.dropout_rate > 0:
                # Dropout between layers
                key = self.make_rng("dropout")
                mask = jax.random.bernoulli(key, 1 - self.dropout_rate, h.shape)
                h = h * mask / (1 - self.dropout_rate)

            all_outputs.append(h)
            final_states.append(carry)

        return h, tuple(final_states)


# ============================================================================
# TT-Convolution
# ============================================================================

class TTConv2d(nn.Module):
    """
    2D Convolution with TT-compressed filters.

    Compresses the (out_channels, in_channels, kH, kW) weight tensor
    into TT format, reducing parameters for large filter tensors.

    Attributes
    ----------
    out_channels : number of output channels
    in_channels : number of input channels
    kernel_size : (kH, kW) kernel size
    tt_rank : TT-rank
    stride : convolution stride
    padding : padding type ('same' or 'valid')
    """

    out_channels: int
    in_channels: int
    kernel_size: Tuple[int, int] = (3, 3)
    tt_rank: int = 4
    stride: int = 1
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        x : (batch, height, width, in_channels) NHWC format

        Returns
        -------
        (batch, H', W', out_channels) output
        """
        kH, kW = self.kernel_size

        # TT filter: shape (out_channels, in_channels, kH, kW)
        # Factorize as TT over [out_channels, kH, kW, in_channels] modes
        filter_shape = (self.out_channels, kH, kW, self.in_channels)
        filter_sizes = filter_shape

        # TT cores for the filter tensor
        n_modes = 4
        filter_cores = []
        for k, size in enumerate(filter_sizes):
            r_l = 1 if k == 0 else self.tt_rank
            r_r = 1 if k == n_modes - 1 else self.tt_rank
            scale = 1.0 / math.sqrt(r_l * size * r_r + 1)
            G = self.param(
                f"filter_core_{k}",
                nn.initializers.normal(stddev=scale),
                (r_l, size, r_r),
            )
            filter_cores.append(G)

        # Reconstruct filter from TT
        tt_filter = TensorTrain(filter_cores, filter_shape)
        W = tt_to_dense(tt_filter)  # (out_channels, kH, kW, in_channels)

        # Reshape to (kH, kW, in_channels, out_channels) for JAX conv
        W_conv = W.transpose(1, 2, 3, 0)  # (kH, kW, in_channels, out_channels)

        bias = self.param("bias", nn.initializers.zeros, (self.out_channels,))

        # Apply convolution
        y = jax.lax.conv_general_dilated(
            x,
            W_conv,
            window_strides=(self.stride, self.stride),
            padding=self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        return y + bias

    def n_params(self) -> int:
        kH, kW = self.kernel_size
        n_modes = 4
        sizes = (self.out_channels, kH, kW, self.in_channels)
        count = self.out_channels  # bias
        for k in range(n_modes):
            r_l = 1 if k == 0 else self.tt_rank
            r_r = 1 if k == n_modes - 1 else self.tt_rank
            count += r_l * sizes[k] * r_r
        return count

    def compression_ratio(self) -> float:
        kH, kW = self.kernel_size
        dense = self.out_channels * self.in_channels * kH * kW + self.out_channels
        return dense / (self.n_params() + 1e-10)


# ============================================================================
# Compression utilities
# ============================================================================

def compress_dense_layer(
    weight: jnp.ndarray,
    row_shape: Tuple[int, ...],
    col_shape: Tuple[int, ...],
    tt_rank: int = 8,
    cutoff: float = 1e-8,
) -> TensorTrainMatrix:
    """
    Compress a dense weight matrix into TT-matrix format.

    Parameters
    ----------
    weight : (out_features, in_features) weight matrix
    row_shape : factored output shape
    col_shape : factored input shape
    tt_rank : maximum TT-rank
    cutoff : SVD truncation threshold

    Returns
    -------
    TensorTrainMatrix approximation
    """
    return tt_from_matrix(weight, row_shape, col_shape, max_rank=tt_rank, cutoff=cutoff)


def estimate_compression_ratio(
    in_features: int,
    out_features: int,
    row_shape: Tuple[int, ...],
    col_shape: Tuple[int, ...],
    tt_rank: int,
) -> float:
    """
    Estimate the compression ratio for a TT-linear layer.

    Parameters
    ----------
    in_features : input dimension
    out_features : output dimension
    row_shape : factored output shape
    col_shape : factored input shape
    tt_rank : TT-rank

    Returns
    -------
    Compression ratio (dense_params / tt_params)
    """
    n_modes = len(row_shape)
    dense_params = in_features * out_features
    tt_params = sum(
        (1 if k == 0 else tt_rank) * row_shape[k] * col_shape[k] * (1 if k == n_modes - 1 else tt_rank)
        for k in range(n_modes)
    )
    return dense_params / (tt_params + 1e-10)


def select_tt_rank_for_ratio(
    in_features: int,
    out_features: int,
    row_shape: Tuple[int, ...],
    col_shape: Tuple[int, ...],
    target_ratio: float,
    max_rank: int = 64,
) -> int:
    """
    Find the TT-rank that achieves a target compression ratio.

    Binary search over ranks from 1 to max_rank.

    Parameters
    ----------
    in_features : input dimension
    out_features : output dimension
    row_shape : factored row shape
    col_shape : factored col shape
    target_ratio : desired compression ratio
    max_rank : maximum rank to consider

    Returns
    -------
    Recommended TT-rank
    """
    lo, hi = 1, max_rank
    while lo < hi:
        mid = (lo + hi) // 2
        ratio = estimate_compression_ratio(in_features, out_features, row_shape, col_shape, mid)
        if ratio >= target_ratio:
            hi = mid
        else:
            lo = mid + 1
    return lo


def factorize_dimension(n: int, n_modes: int = 3) -> Tuple[int, ...]:
    """
    Factorize a dimension n into n_modes roughly equal factors.

    Parameters
    ----------
    n : dimension to factorize
    n_modes : number of factors

    Returns
    -------
    Tuple of n_modes factors approximately equal to n^(1/n_modes)
    """
    return _factorize(n, n_modes=n_modes)


def _factorize(n: int, n_modes: int = 2) -> Tuple[int, ...]:
    """Helper to factorize n into approximately equal parts."""
    if n_modes == 1:
        return (n,)

    # Find the most even factorization
    best = None
    best_prod = 0

    # Try all factor pairs
    for f1 in range(2, int(n ** 0.5) + 1):
        if n % f1 == 0:
            rest = n // f1
            sub = _factorize(rest, n_modes - 1)
            candidate = (f1,) + sub
            prod = 1
            for c in candidate:
                prod *= c
            if prod == n and (best is None or max(candidate) < max(best)):
                best = candidate
                best_prod = prod

    if best is None:
        # Fallback: use powers of 2
        if n_modes == 2:
            f1 = max(1, int(math.sqrt(n)))
            f2 = math.ceil(n / f1)
            while f1 * f2 < n:
                f2 += 1
            return (f1, f2)
        else:
            f = max(1, int(n ** (1.0 / n_modes)))
            factors = [f] * n_modes
            # Adjust last factor
            prod = 1
            for f_ in factors[:-1]:
                prod *= f_
            factors[-1] = math.ceil(n / prod)
            return tuple(factors)

    return best


# ============================================================================
# Model compression pipeline
# ============================================================================

class ModelCompressor:
    """
    Utility class for compressing Flax neural network models.

    Replaces Dense layers with TTLinearSimple layers while preserving
    the original weight values (via TT-SVD initialization).
    """

    def __init__(
        self,
        tt_rank: int = 8,
        target_compression: float = 10.0,
        cutoff: float = 1e-8,
    ):
        self.tt_rank = tt_rank
        self.target_compression = target_compression
        self.cutoff = cutoff

    def compress_weight(
        self,
        weight: jnp.ndarray,
    ) -> TensorTrainMatrix:
        """
        Compress a single weight matrix to TT format.

        Parameters
        ----------
        weight : (out, in) weight matrix

        Returns
        -------
        TensorTrainMatrix
        """
        out_features, in_features = weight.shape

        # Choose factorization
        row_shape = _factorize(out_features)
        col_shape = _factorize(in_features)

        # Choose rank for target compression
        rank = select_tt_rank_for_ratio(
            in_features, out_features, row_shape, col_shape,
            self.target_compression, max_rank=self.tt_rank
        )

        return compress_dense_layer(weight, row_shape, col_shape, tt_rank=rank, cutoff=self.cutoff)

    def compute_compression_stats(
        self,
        original_params: int,
        compressed_params: int,
    ) -> Dict[str, float]:
        """Report compression statistics."""
        return {
            "original_params": original_params,
            "compressed_params": compressed_params,
            "compression_ratio": original_params / (compressed_params + 1e-10),
            "params_saved": original_params - compressed_params,
            "params_saved_frac": 1 - compressed_params / (original_params + 1e-10),
        }


# ============================================================================
# TT-Attention Layer
# ============================================================================

class TTAttention(nn.Module):
    """
    Multi-head attention with TT-compressed projection matrices.

    Replaces the Q, K, V, and output projection Dense layers with
    TTLinearSimple layers, reducing parameters while preserving attention
    expressivity.

    Attributes
    ----------
    d_model : model dimension
    n_heads : number of attention heads
    tt_rank : TT-rank for weight compression
    """

    d_model: int
    n_heads: int
    tt_rank: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Multi-head attention with TT projections.

        Parameters
        ----------
        query : (batch, seq_q, d_model)
        key : (batch, seq_k, d_model)
        value : (batch, seq_k, d_model)
        mask : optional attention mask
        training : dropout mode

        Returns
        -------
        (batch, seq_q, d_model)
        """
        batch, seq_q, d = query.shape
        _, seq_k, _ = key.shape
        d_head = self.d_model // self.n_heads

        row_shape = _factorize(self.d_model)
        col_shape = _factorize(self.d_model)

        # TT-compressed projections
        Q = TTLinearSimple(
            row_shape=row_shape, col_shape=col_shape,
            tt_rank=self.tt_rank, name="q_proj"
        )(query)  # (batch, seq_q, d_model)

        K = TTLinearSimple(
            row_shape=row_shape, col_shape=col_shape,
            tt_rank=self.tt_rank, name="k_proj"
        )(key)

        V = TTLinearSimple(
            row_shape=row_shape, col_shape=col_shape,
            tt_rank=self.tt_rank, name="v_proj"
        )(value)

        # Split into heads
        Q = Q.reshape(batch, seq_q, self.n_heads, d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_k, self.n_heads, d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_k, self.n_heads, d_head).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = math.sqrt(d_head)
        scores = jnp.einsum("bhqd,bhkd->bhqk", Q, K) / scale

        if mask is not None:
            scores = scores + mask * (-1e9)

        attn_weights = jax.nn.softmax(scores, axis=-1)

        if training and self.dropout_rate > 0:
            key_rng = self.make_rng("dropout")
            attn_weights = nn.Dropout(self.dropout_rate)(attn_weights, deterministic=not training)

        out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, V)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_q, self.d_model)

        # Output projection
        out = TTLinearSimple(
            row_shape=row_shape, col_shape=col_shape,
            tt_rank=self.tt_rank, name="out_proj"
        )(out)

        return out


# ============================================================================
# Quantization + TT compression pipeline
# ============================================================================

def quantize_tt_cores(
    tt: TensorTrain,
    n_bits: int = 8,
) -> TensorTrain:
    """
    Quantize TT cores to n_bits fixed-point representation.

    Parameters
    ----------
    tt : TensorTrain to quantize
    n_bits : number of quantization bits

    Returns
    -------
    Quantized TensorTrain (values rounded to n_bits grid)
    """
    scale = 2 ** n_bits - 1
    new_cores = []
    for G in tt.cores:
        G_min = jnp.min(G)
        G_max = jnp.max(G)
        # Quantize to [0, scale]
        G_quant = jnp.round((G - G_min) / (G_max - G_min + 1e-10) * scale)
        # Dequantize
        G_deq = G_quant / scale * (G_max - G_min) + G_min
        new_cores.append(G_deq)
    return TensorTrain(new_cores, tt.shape)


def tt_compress_and_quantize(
    weight: jnp.ndarray,
    row_shape: Tuple[int, ...],
    col_shape: Tuple[int, ...],
    tt_rank: int = 8,
    n_bits: int = 8,
) -> Dict[str, Any]:
    """
    Two-stage compression: TT decomposition + quantization.

    Parameters
    ----------
    weight : (out, in) weight matrix
    row_shape : factored output shape
    col_shape : factored input shape
    tt_rank : TT-rank
    n_bits : quantization bits

    Returns
    -------
    Dictionary with TT cores, reconstructed matrix, and stats
    """
    # Stage 1: TT decomposition
    ttm = compress_dense_layer(weight, row_shape, col_shape, tt_rank=tt_rank)

    # Stage 2: Quantize TT cores
    n_modes = len(row_shape)
    cores_tt = [ttm.cores[k].reshape(
        ttm.ranks[k], row_shape[k] * col_shape[k], ttm.ranks[k + 1]
    ) for k in range(n_modes)]
    tt_obj = TensorTrain(cores_tt, tuple(row_shape[k] * col_shape[k] for k in range(n_modes)))
    tt_q = quantize_tt_cores(tt_obj, n_bits=n_bits)

    # Reconstruct
    W_recon = ttm.to_dense()
    err = float(jnp.linalg.norm(weight - W_recon) / (jnp.linalg.norm(weight) + 1e-10))

    # Parameter count
    tt_params = sum(G.size for G in ttm.cores)
    quantized_bits = tt_params * n_bits / 8  # bytes

    return {
        "ttm": ttm,
        "reconstruction_error": err,
        "original_params": weight.size,
        "tt_params": tt_params,
        "quantized_bytes": quantized_bits,
        "compression_ratio_tt": weight.size * 4 / (tt_params * 4 + 1e-10),
        "compression_ratio_tt_q": weight.size * 4 / (quantized_bits + 1e-10),
        "reconstructed": W_recon,
    }
