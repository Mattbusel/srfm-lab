"""
quantum_ml.py — Quantum-inspired machine learning with tensor networks (Project AETERNUS).

Implements:
  - MPS Classifier: discriminative model using MPS as feature map
  - MERA-inspired feature maps: multi-scale entanglement renormalization ansatz
  - Tensor Network Attention mechanism
  - Entanglement entropy as a regularizer for neural networks
  - Quantum kernel approximation via MPS inner products
  - Born Machine: generative model with MPS
  - Quantum circuit simulation (basic)
  - Variational Quantum Eigensolver (VQE) for portfolio optimization
  - Variational Portfolio Optimizer (quantum-inspired)
  - MPS sampling (Born rule)
  - Quantum feature encoding (angle encoding, amplitude encoding)
  - Quantum-classical hybrid gradients
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

from .mps import (
    MatrixProductState, mps_random, mps_normalize,
    mps_inner_product, mps_norm, mps_from_dense,
    mps_compress, mps_left_canonicalize
)
from .tt_decomp import TensorTrain, tt_svd


# ============================================================================
# Quantum feature encoding
# ============================================================================

def angle_encoding(
    x: jnp.ndarray,
    n_qubits: int,
    encoding_type: str = "rx",
) -> List[jnp.ndarray]:
    """
    Encode classical data into quantum angles (angle encoding).

    Maps each feature x_i to a rotation angle theta_i = pi * tanh(x_i).
    Returns a list of single-qubit rotation matrices.

    Parameters
    ----------
    x : (n_features,) feature vector
    n_qubits : number of qubits
    encoding_type : 'rx', 'ry', or 'rz'

    Returns
    -------
    List of (2, 2) rotation matrices for each qubit
    """
    # Map features to angles
    n_feat = x.shape[0]
    angles = jnp.pi * jnp.tanh(x)

    # Repeat or truncate to n_qubits
    if n_feat < n_qubits:
        angles = jnp.concatenate([angles, jnp.zeros(n_qubits - n_feat)])
    else:
        angles = angles[:n_qubits]

    gates = []
    for i in range(n_qubits):
        theta = angles[i]
        if encoding_type == "rx":
            # Rx(theta) = [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]]
            c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
            R = jnp.array([[c, -1j * s], [-1j * s, c]], dtype=jnp.complex64)
        elif encoding_type == "ry":
            c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
            R = jnp.array([[c, -s], [s, c]], dtype=jnp.complex64)
        else:  # rz
            R = jnp.array([[jnp.exp(-1j * theta / 2), 0],
                           [0, jnp.exp(1j * theta / 2)]], dtype=jnp.complex64)
        gates.append(R)

    return gates


def amplitude_encoding(
    x: jnp.ndarray,
    n_qubits: int,
) -> MatrixProductState:
    """
    Encode a classical vector x as quantum amplitudes in an MPS.

    Normalizes x and encodes it as the amplitudes |psi> = sum_i x_i/||x|| |i>.

    Parameters
    ----------
    x : (d,) feature vector, d must equal 2^n_qubits (or gets padded)
    n_qubits : number of qubits

    Returns
    -------
    MatrixProductState with d=2 per site encoding the state
    """
    d_target = 2 ** n_qubits
    x_arr = jnp.array(x, dtype=jnp.float32)

    if x_arr.shape[0] < d_target:
        x_arr = jnp.concatenate([x_arr, jnp.zeros(d_target - x_arr.shape[0])])
    else:
        x_arr = x_arr[:d_target]

    # Normalize
    x_arr = x_arr / (jnp.linalg.norm(x_arr) + 1e-10)

    phys_dims = [2] * n_qubits
    return mps_from_dense(x_arr, phys_dims, max_bond=min(32, 2 ** (n_qubits // 2)))


def iqp_encoding(
    x: jnp.ndarray,
    n_qubits: int,
    n_layers: int = 2,
) -> MatrixProductState:
    """
    IQP (Instantaneous Quantum Polynomial) encoding for kernel methods.

    Encodes classical data via a diagonal unitary: e^{i phi(x)},
    where phi(x) is a polynomial function of x.

    Parameters
    ----------
    x : (n_features,) feature vector
    n_qubits : number of qubits
    n_layers : number of IQP layers

    Returns
    -------
    MatrixProductState representing the encoded quantum state
    """
    n_feat = x.shape[0]
    angles = jnp.tanh(x) * jnp.pi

    # Start with |+>^N state
    mps = mps_random(n_qubits, 2, 1, jax.random.PRNGKey(0))
    # Set to |+> = (|0> + |1>) / sqrt(2)
    plus_state = jnp.array([[1.0, 1.0]]) / math.sqrt(2)
    tensors = [plus_state.reshape(1, 2, 1) for _ in range(n_qubits)]
    mps = MatrixProductState(tensors, (2,) * n_qubits)

    # Apply diagonal gates (approximated in MPS)
    for layer in range(n_layers):
        for i in range(n_qubits):
            angle = float(angles[i % n_feat])
            # Rz gate (diagonal)
            rz = jnp.array([jnp.exp(-1j * angle / 2), jnp.exp(1j * angle / 2)],
                           dtype=jnp.complex64)
            new_t = mps.tensors[i].astype(jnp.complex64)
            new_t = jnp.einsum("abc,b->abc", new_t, rz)
            tensors = list(mps.tensors)
            tensors[i] = jnp.real(new_t).astype(jnp.float32)
            mps = MatrixProductState(tensors, mps.phys_dims)

    return mps


# ============================================================================
# MPS Classifier
# ============================================================================

class MPSClassifier:
    """
    Discriminative MPS-based classifier.

    Uses a set of class-specific MPS to classify inputs:
      argmax_c |<psi_c | phi(x)>|^2

    where |phi(x)> is the quantum feature encoding of x and |psi_c>
    is the learned class MPS.

    Training uses gradient descent on the negative log-likelihood.
    """

    def __init__(
        self,
        n_classes: int,
        n_qubits: int = 8,
        bond_dim: int = 8,
        encoding: str = "amplitude",
        n_epochs: int = 100,
        lr: float = 0.01,
    ):
        self.n_classes = n_classes
        self.n_qubits = n_qubits
        self.bond_dim = bond_dim
        self.encoding = encoding
        self.n_epochs = n_epochs
        self.lr = lr

        self.class_mps_: List[MatrixProductState] = []
        self.is_fitted = False

    def _encode_features(
        self,
        x: jnp.ndarray,
    ) -> MatrixProductState:
        """Encode feature vector as MPS."""
        if self.encoding == "amplitude":
            return amplitude_encoding(x, self.n_qubits)
        elif self.encoding == "iqp":
            return iqp_encoding(x, self.n_qubits)
        else:
            # Angle encoding -> product state
            gates = angle_encoding(x, self.n_qubits, encoding_type="ry")
            tensors = []
            for R in gates:
                # Take first column of R (|0> state)
                v = jnp.real(R[:, 0]).astype(jnp.float32)
                tensors.append(v.reshape(1, 2, 1))
            return MatrixProductState(tensors, (2,) * self.n_qubits)

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        key: Optional[jax.random.KeyArray] = None,
        verbose: bool = False,
    ) -> "MPSClassifier":
        """
        Fit MPS classifiers to training data.

        Parameters
        ----------
        X : (n_samples, n_features) feature matrix
        y : (n_samples,) integer class labels
        key : JAX random key
        verbose : print training progress

        Returns
        -------
        self
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        X = jnp.array(X, dtype=jnp.float32)
        y = jnp.array(y, dtype=jnp.int32)
        n_samples, n_feat = X.shape

        # Initialize class MPS randomly
        self.class_mps_ = []
        for c in range(self.n_classes):
            key, subkey = jax.random.split(key)
            mps_c = mps_random(self.n_qubits, 2, self.bond_dim, subkey)
            self.class_mps_.append(mps_c)

        # Train with projected gradient descent (simplified)
        optimizer = optax.adam(self.lr)

        # Use list of tensors as parameter tree
        params = [mps.tensors for mps in self.class_mps_]
        opt_state = optimizer.init(params)

        def loss_fn(params_inner, X_batch, y_batch):
            """Cross-entropy loss over batch."""
            total_loss = jnp.zeros(())
            for idx in range(X_batch.shape[0]):
                x_i = X_batch[idx]
                label = y_batch[idx]
                phi = self._encode_features(x_i)

                # Compute log-probabilities
                log_probs = []
                for c in range(self.n_classes):
                    mps_c = MatrixProductState(params_inner[c], (2,) * self.n_qubits)
                    mps_c = mps_normalize(mps_c)
                    ip = mps_inner_product(mps_c, phi)
                    prob = jnp.abs(ip) ** 2
                    log_probs.append(jnp.log(prob + 1e-10))

                log_probs_arr = jnp.stack(log_probs)
                # Softmax cross-entropy
                log_norm = jax.nn.log_softmax(log_probs_arr)
                total_loss = total_loss - log_norm[label]

            return total_loss / X_batch.shape[0]

        for epoch in range(self.n_epochs):
            # Mini-batch gradient step
            batch_size = min(32, n_samples)
            key, subkey = jax.random.split(key)
            idx = jax.random.permutation(subkey, n_samples)[:batch_size]
            X_batch = X[idx]
            y_batch = y[idx]

            loss, grads = jax.value_and_grad(loss_fn)(params, X_batch, y_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {float(loss):.4f}")

        # Update class MPS
        self.class_mps_ = [
            mps_normalize(MatrixProductState(p, (2,) * self.n_qubits))
            for p in params
        ]
        self.is_fitted = True
        return self

    def predict_proba(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute class probabilities.

        Parameters
        ----------
        X : (n_samples, n_features)

        Returns
        -------
        (n_samples, n_classes) probability matrix
        """
        assert self.is_fitted
        X = jnp.array(X, dtype=jnp.float32)
        probs = []
        for i in range(X.shape[0]):
            phi = self._encode_features(X[i])
            row = []
            for mps_c in self.class_mps_:
                ip = mps_inner_product(mps_c, phi)
                row.append(float(jnp.abs(ip) ** 2))
            total = sum(row) + 1e-10
            probs.append([p / total for p in row])
        return jnp.array(probs)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return jnp.argmax(proba, axis=1)


# ============================================================================
# Quantum Kernel
# ============================================================================

class QuantumKernel:
    """
    Quantum kernel function k(x, x') = |<phi(x)|phi(x')>|^2.

    Uses MPS inner products as the kernel evaluation. The MPS encoding
    is determined by the quantum feature map.

    This provides an expressive non-linear kernel that respects the
    entanglement structure of the data.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        bond_dim: int = 4,
        encoding: str = "amplitude",
    ):
        self.n_qubits = n_qubits
        self.bond_dim = bond_dim
        self.encoding = encoding
        self._feature_cache: Dict[int, MatrixProductState] = {}

    def _encode(self, x: jnp.ndarray) -> MatrixProductState:
        """Encode feature vector."""
        if self.encoding == "amplitude":
            mps = amplitude_encoding(x, self.n_qubits)
        elif self.encoding == "iqp":
            mps = iqp_encoding(x, self.n_qubits)
        else:
            mps = amplitude_encoding(x, self.n_qubits)
        return mps_compress(mps, max_bond=self.bond_dim)

    def __call__(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Evaluate quantum kernel k(x1, x2) = |<phi(x1)|phi(x2)>|^2.

        Parameters
        ----------
        x1, x2 : feature vectors

        Returns
        -------
        Scalar kernel value
        """
        phi1 = self._encode(x1)
        phi2 = self._encode(x2)
        ip = mps_inner_product(phi1, phi2)
        return jnp.abs(ip) ** 2

    def compute_kernel_matrix(
        self,
        X1: jnp.ndarray,
        X2: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute Gram matrix K[i, j] = k(X1[i], X2[j]).

        Parameters
        ----------
        X1 : (n1, n_features) matrix
        X2 : (n2, n_features) matrix (if None, use X1)

        Returns
        -------
        (n1, n2) kernel matrix
        """
        if X2 is None:
            X2 = X1

        X1 = jnp.array(X1, dtype=jnp.float32)
        X2 = jnp.array(X2, dtype=jnp.float32)
        n1, n2 = X1.shape[0], X2.shape[0]

        # Encode all points
        phi1 = [self._encode(X1[i]) for i in range(n1)]
        phi2 = [self._encode(X2[j]) for j in range(n2)]

        K = jnp.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                ip = mps_inner_product(phi1[i], phi2[j])
                K = K.at[i, j].set(float(jnp.abs(ip) ** 2))

        return K

    def quantum_kernel_svm(
        self,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        C: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Train a kernel SVM using the quantum kernel.

        Uses dual formulation with gradient-based optimization.

        Parameters
        ----------
        X_train : (n_train, n_features)
        y_train : (n_train,) binary labels {-1, +1}
        C : SVM regularization parameter

        Returns
        -------
        Dictionary with alphas, support vectors, bias
        """
        K = self.compute_kernel_matrix(X_train)
        n = K.shape[0]

        y = jnp.array(y_train, dtype=jnp.float32)

        # Solve dual QP via gradient ascent
        # max sum(alpha) - 0.5 alpha^T (y y^T * K) alpha
        # s.t. 0 <= alpha <= C, sum(alpha * y) = 0
        alpha = jnp.zeros(n) + 0.01
        lr = 0.01

        for _ in range(200):
            Q = jnp.outer(y, y) * K
            grad_alpha = jnp.ones(n) - Q @ alpha
            alpha = alpha + lr * grad_alpha
            alpha = jnp.clip(alpha, 0, C)
            # Project onto equality constraint
            alpha = alpha - jnp.mean(alpha * y) * y

        # Compute bias
        sv_mask = alpha > 1e-5 * C
        if jnp.sum(sv_mask) > 0:
            b = jnp.mean(y[sv_mask] - (K[sv_mask] @ (alpha * y)))
        else:
            b = jnp.zeros(())

        return {
            "alphas": alpha,
            "support_vector_mask": sv_mask,
            "bias": b,
            "kernel_matrix": K,
        }


# ============================================================================
# MERA-inspired feature map
# ============================================================================

class MERA(nn.Module):
    """
    MERA (Multi-scale Entanglement Renormalization Ansatz) inspired feature map.

    A MERA is a hierarchical tensor network that maps a physical state to a
    coarser representation through alternating disentangler and isometry layers.

    Here, we use a classical analog: a hierarchical neural network with
    structured weight matrices that mimic the MERA renormalization structure.

    Architecture:
    - L layers of (disentangler + isometry) blocks
    - Each block halves the feature dimension
    - Total: input_dim -> input_dim / 2^L features
    """

    n_layers: int = 3
    input_dim: int = 64
    bond_dim: int = 4
    output_dim: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass through the MERA feature map.

        Parameters
        ----------
        x : (batch, input_dim) input features
        training : whether in training mode

        Returns
        -------
        (batch, output_dim) MERA features
        """
        h = x
        current_dim = self.input_dim

        for layer in range(self.n_layers):
            # Disentangler: pairwise unitary rotations
            h = self._disentangler_layer(h, layer, current_dim)
            # Isometry: coarsen by factor 2
            half_dim = max(current_dim // 2, self.output_dim)
            h = nn.Dense(half_dim, name=f"isometry_{layer}")(h)
            h = nn.relu(h)
            current_dim = half_dim

        # Final projection
        h = nn.Dense(self.output_dim, name="output")(h)
        return h

    def _disentangler_layer(
        self,
        h: jnp.ndarray,
        layer: int,
        dim: int,
    ) -> jnp.ndarray:
        """Apply disentangler: local unitary-like transformations."""
        # Pair adjacent features and apply 2x2 rotation blocks
        if dim < 2:
            return h

        # Reshape to pairs
        n_pairs = dim // 2
        h_pairs = h[..., :2 * n_pairs].reshape(-1, n_pairs, 2)

        # Apply learned 2x2 rotation (parameterized as angle)
        theta = self.param(f"theta_{layer}", nn.initializers.zeros, (n_pairs,))
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        # Rotation matrix applied per pair
        h_rot = jnp.stack([
            h_pairs[..., 0] * c - h_pairs[..., 1] * s,
            h_pairs[..., 0] * s + h_pairs[..., 1] * c,
        ], axis=-1)

        h_out = h_rot.reshape(-1, 2 * n_pairs)
        if dim % 2 == 1:
            h_out = jnp.concatenate([h_out, h[..., -1:]], axis=-1)
        return h_out


# ============================================================================
# Tensor Network Attention
# ============================================================================

class TensorNetworkAttention(nn.Module):
    """
    Tensor Network Attention mechanism for financial time series.

    Replaces the standard dot-product attention with a tensor-contraction
    attention score, enabling higher-order interactions between queries
    and keys.

    The attention score is:
      A(q, k) = q^T W1 k + (q ⊗ q)^T W2 (k ⊗ k)  (rank-2 interaction)

    This captures pairwise and cross-feature interactions that standard
    attention misses.
    """

    d_model: int = 64
    n_heads: int = 4
    d_key: int = 16
    tt_rank: int = 4
    dropout_rate: float = 0.1

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
        Multi-head tensor network attention.

        Parameters
        ----------
        query : (batch, seq_q, d_model) query tensor
        key : (batch, seq_k, d_model) key tensor
        value : (batch, seq_k, d_model) value tensor
        mask : optional attention mask
        training : dropout mode

        Returns
        -------
        (batch, seq_q, d_model) attended output
        """
        batch, seq_q, _ = query.shape
        _, seq_k, _ = key.shape
        d_head = self.d_model // self.n_heads

        # Standard linear projections
        Q = nn.Dense(self.d_model, name="q_proj")(query)
        K = nn.Dense(self.d_model, name="k_proj")(key)
        V = nn.Dense(self.d_model, name="v_proj")(value)

        # Reshape for multi-head
        Q = Q.reshape(batch, seq_q, self.n_heads, d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_k, self.n_heads, d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_k, self.n_heads, d_head).transpose(0, 2, 1, 3)

        # Standard scaled dot-product attention scores
        scale = math.sqrt(d_head)
        scores = jnp.einsum("bhqd,bhkd->bhqk", Q, K) / scale

        # Second-order tensor interaction term
        # Pairwise feature product interaction
        W2 = self.param(
            "W2_interaction",
            nn.initializers.xavier_uniform(),
            (self.n_heads, d_head, d_head, self.tt_rank),
        )
        # Compute rank-1 TT interaction: Q_i W2 K_j
        Q_proj = jnp.einsum("bhqd,hddr->bhqr", Q, W2.reshape(self.n_heads, d_head, d_head * self.tt_rank))
        # (batch, heads, seq_q, d_head * tt_rank)
        # Dot with K via tt_rank contraction
        K_proj = jnp.einsum("bhkd,hddr->bhkr",
                             K,
                             W2.reshape(self.n_heads, d_head, d_head, self.tt_rank).transpose(0, 2, 1, 3).reshape(
                                 self.n_heads, d_head, d_head * self.tt_rank))
        # Interaction score: (batch, heads, seq_q, seq_k)
        Q_p = Q_proj.reshape(batch, self.n_heads, seq_q, d_head, self.tt_rank)
        K_p = K_proj.reshape(batch, self.n_heads, seq_k, d_head, self.tt_rank)
        interaction = jnp.einsum("bhqdr,bhkdr->bhqk", Q_p, K_p) / (d_head * scale)

        # Combine standard and tensor interaction scores
        alpha = self.param("alpha", nn.initializers.zeros, ())
        total_scores = scores + alpha * interaction

        if mask is not None:
            total_scores = total_scores + mask * (-1e9)

        attn_weights = jax.nn.softmax(total_scores, axis=-1)
        if training:
            attn_weights = nn.Dropout(rate=self.dropout_rate)(attn_weights, deterministic=not training)

        # Attend to values
        out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, V)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_q, self.d_model)

        # Output projection
        out = nn.Dense(self.d_model, name="out_proj")(out)
        return out


# ============================================================================
# Entanglement entropy regularizer
# ============================================================================

def entanglement_regularizer(
    params: Any,
    model: Any,
    x: jnp.ndarray,
    alpha: float = 0.01,
    n_qubits: int = 8,
) -> jnp.ndarray:
    """
    Entanglement entropy as a regularizer for neural networks.

    Encourages the hidden representations to have low entanglement entropy
    (i.e., to be factorable), promoting disentangled representations.

    The regularizer computes the entanglement entropy of the hidden activation
    when encoded as an MPS.

    Parameters
    ----------
    params : model parameters (Flax pytree)
    model : Flax model
    x : input batch
    alpha : regularization strength
    n_qubits : MPS encoding qubits

    Returns
    -------
    Scalar regularization term
    """
    # Get hidden activation
    h = model.apply(params, x)  # Assume model returns hidden features

    batch = h.shape[0]
    total_entropy = jnp.zeros(())

    for i in range(min(batch, 4)):  # Limit for tractability
        h_i = h[i]
        mps = amplitude_encoding(h_i, n_qubits)

        # Compute bond entropies
        from .mps import mps_bond_entropies
        entropies = mps_bond_entropies(mps)
        total_entropy = total_entropy + jnp.mean(entropies)

    return alpha * total_entropy / min(batch, 4)


def mps_regularizer(
    weight_matrix: jnp.ndarray,
    row_shape: Tuple[int, ...],
    col_shape: Tuple[int, ...],
    max_rank: int = 4,
    alpha: float = 0.001,
) -> jnp.ndarray:
    """
    TT-rank regularization for weight matrices.

    Encourages weight matrices to be low-rank in TT format by penalizing
    the Frobenius norm of the difference from its TT approximation.

    Parameters
    ----------
    weight_matrix : (m, n) weight matrix
    row_shape : factored row shape
    col_shape : factored column shape
    max_rank : target TT-rank
    alpha : regularization strength

    Returns
    -------
    Scalar regularization loss
    """
    from .tt_decomp import tt_from_matrix, tt_to_dense

    ttm = tt_from_matrix(weight_matrix, row_shape, col_shape, max_rank=max_rank)
    # Reconstruct
    recon = ttm.to_dense()
    diff = weight_matrix - recon
    return alpha * jnp.linalg.norm(diff) ** 2


# ============================================================================
# Born Machine (generative MPS model)
# ============================================================================

class BornMachine:
    """
    Born Machine: generative model using MPS as the probability amplitude.

    Models the probability distribution P(x) = |<x|psi(theta)>|^2
    where |psi(theta)> is a parameterized MPS.

    Applications:
    - Modeling return distributions
    - Generating synthetic financial scenarios
    - Density estimation

    References
    ----------
    Han, Z. Y., Wang, J., Fan, H., Wang, L., & Zhang, P. (2018). Unsupervised
    generative modeling using matrix product states. Physical Review X, 8(3), 031012.
    """

    def __init__(
        self,
        n_sites: int = 10,
        phys_dim: int = 2,
        bond_dim: int = 8,
        n_epochs: int = 200,
        lr: float = 0.005,
        batch_size: int = 64,
    ):
        self.n_sites = n_sites
        self.phys_dim = phys_dim
        self.bond_dim = bond_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.mps_: Optional[MatrixProductState] = None
        self.training_losses_: List[float] = []
        self.is_fitted = False

    def fit(
        self,
        data: jnp.ndarray,
        key: Optional[jax.random.KeyArray] = None,
        verbose: bool = False,
    ) -> "BornMachine":
        """
        Train the Born Machine on binary data.

        Parameters
        ----------
        data : (n_samples, n_sites) binary data matrix (values 0 or 1)
        key : JAX random key
        verbose : print training progress

        Returns
        -------
        self
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        data = jnp.array(data, dtype=jnp.int32)
        n_samples, n = data.shape
        assert n == self.n_sites

        # Initialize MPS
        key, subkey = jax.random.split(key)
        self.mps_ = mps_random(n, self.phys_dim, self.bond_dim, subkey)
        self.mps_ = mps_left_canonicalize(self.mps_)

        optimizer = optax.adam(self.lr)
        params = self.mps_.tensors
        opt_state = optimizer.init(params)

        def nll_loss(tensors, batch):
            """Negative log-likelihood on a batch."""
            mps = MatrixProductState(tensors, (self.phys_dim,) * self.n_sites)
            # Normalize
            mps = mps_normalize(mps)
            total_nll = jnp.zeros(())
            for i in range(batch.shape[0]):
                # Build the basis state for this sample
                config = batch[i]
                # Contract MPS at config: amplitude psi[config]
                v = mps.tensors[0][0, config[0], :]  # (chi,)
                for j in range(1, n):
                    v = v @ jnp.reshape(mps.tensors[j][:, config[j], :], (v.shape[0], -1))
                    if j < n - 1:
                        pass  # v now has shape matching next bond
                # v is scalar (shape (1,) after full contraction)
                amp = v[0] if len(v.shape) > 0 else v
                prob = amp ** 2
                total_nll = total_nll - jnp.log(jnp.maximum(prob, 1e-10))
            return total_nll / batch.shape[0]

        for epoch in range(self.n_epochs):
            key, subkey = jax.random.split(key)
            idx = jax.random.permutation(subkey, n_samples)[:self.batch_size]
            batch = data[idx]

            loss, grads = jax.value_and_grad(nll_loss)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            self.training_losses_.append(float(loss))

            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}: NLL = {float(loss):.4f}")

        self.mps_ = mps_normalize(
            MatrixProductState(params, (self.phys_dim,) * self.n_sites)
        )
        self.is_fitted = True
        return self

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log P(x) for a sample x.

        Parameters
        ----------
        x : (n_sites,) binary configuration

        Returns
        -------
        Log probability scalar
        """
        assert self.is_fitted
        mps = self.mps_
        v = mps.tensors[0][0, int(x[0]), :]
        for j in range(1, self.n_sites):
            s = int(x[j])
            v = v @ mps.tensors[j][:, s, :]
        prob = float(v[0]) ** 2
        return jnp.log(jnp.maximum(jnp.array(prob), 1e-10))

    def sample(
        self,
        n_samples: int,
        key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """
        Draw samples from Born distribution P(x) = |psi(x)|^2.

        Parameters
        ----------
        n_samples : number of samples
        key : random key

        Returns
        -------
        (n_samples, n_sites) integer array
        """
        assert self.is_fitted
        from .mps import mps_sample
        return mps_sample(self.mps_, n_samples, key)


# ============================================================================
# Quantum Circuit Simulation
# ============================================================================

class QuantumCircuitSim:
    """
    Basic quantum circuit simulator using MPS representation.

    Simulates a quantum circuit acting on n_qubits by:
    1. Starting from |0>^n
    2. Applying gates as tensor operations on MPS
    3. Computing expectation values via MPS inner products

    The MPS representation allows simulation up to ~50+ qubits for
    circuits with limited entanglement (bounded bond dimension).
    """

    def __init__(
        self,
        n_qubits: int,
        max_bond: int = 32,
    ):
        self.n_qubits = n_qubits
        self.max_bond = max_bond
        self._state = self._init_zero_state()

    def _init_zero_state(self) -> MatrixProductState:
        """Initialize |0>^n state as MPS."""
        zero_vec = jnp.array([1.0, 0.0])
        tensors = [zero_vec.reshape(1, 2, 1) for _ in range(self.n_qubits)]
        return MatrixProductState(tensors, (2,) * self.n_qubits)

    @property
    def state(self) -> MatrixProductState:
        """Current quantum state."""
        return self._state

    def reset(self):
        """Reset to |0>^n."""
        self._state = self._init_zero_state()

    def h(self, qubit: int) -> "QuantumCircuitSim":
        """Apply Hadamard gate on qubit i."""
        H = jnp.array([[1, 1], [1, -1]]) / math.sqrt(2)
        return self._apply_single_qubit_gate(H, qubit)

    def x(self, qubit: int) -> "QuantumCircuitSim":
        """Apply Pauli-X (NOT) gate."""
        X = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        return self._apply_single_qubit_gate(X, qubit)

    def y(self, qubit: int) -> "QuantumCircuitSim":
        """Apply Pauli-Y gate."""
        Y = jnp.array([[0.0, -1j], [1j, 0.0]], dtype=jnp.complex64)
        return self._apply_single_qubit_gate(Y, qubit)

    def z(self, qubit: int) -> "QuantumCircuitSim":
        """Apply Pauli-Z gate."""
        Z = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        return self._apply_single_qubit_gate(Z, qubit)

    def rx(self, qubit: int, theta: float) -> "QuantumCircuitSim":
        """Apply Rx rotation gate."""
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        Rx = jnp.array([[c, -1j * s], [-1j * s, c]], dtype=jnp.complex64)
        return self._apply_single_qubit_gate(Rx, qubit)

    def ry(self, qubit: int, theta: float) -> "QuantumCircuitSim":
        """Apply Ry rotation gate."""
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        Ry = jnp.array([[c, -s], [s, c]])
        return self._apply_single_qubit_gate(Ry, qubit)

    def rz(self, qubit: int, theta: float) -> "QuantumCircuitSim":
        """Apply Rz rotation gate."""
        Rz = jnp.array([[jnp.exp(-1j * theta / 2), 0],
                         [0, jnp.exp(1j * theta / 2)]], dtype=jnp.complex64)
        return self._apply_single_qubit_gate(Rz, qubit)

    def cnot(self, control: int, target: int) -> "QuantumCircuitSim":
        """Apply CNOT gate between control and target qubits."""
        # CNOT as a 2-site gate
        CNOT = jnp.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=jnp.float32)
        CNOT_tensor = CNOT.reshape(2, 2, 2, 2)
        return self._apply_two_qubit_gate(CNOT_tensor, control, target)

    def _apply_single_qubit_gate(
        self,
        gate: jnp.ndarray,
        qubit: int,
    ) -> "QuantumCircuitSim":
        """Apply a single-qubit gate at qubit site."""
        tensors = [jnp.array(t) for t in self._state.tensors]
        t = tensors[qubit].astype(jnp.complex64)  # (chi_l, d, chi_r)
        # Apply gate: new_t[chi_l, s', chi_r] = sum_s gate[s', s] t[chi_l, s, chi_r]
        new_t = jnp.einsum("sp,lpq->lsq", gate.astype(jnp.complex64), t)
        tensors[qubit] = jnp.real(new_t).astype(jnp.float32)
        self._state = MatrixProductState(tensors, self._state.phys_dims)
        return self

    def _apply_two_qubit_gate(
        self,
        gate: jnp.ndarray,
        site1: int,
        site2: int,
    ) -> "QuantumCircuitSim":
        """Apply a two-qubit gate between adjacent sites (site1 < site2)."""
        assert abs(site1 - site2) == 1, "Only adjacent two-qubit gates supported"
        i = min(site1, site2)
        j = i + 1

        tensors = [jnp.array(t) for t in self._state.tensors]
        A = tensors[i]  # (chi_l, 2, chi_m)
        B = tensors[j]  # (chi_m, 2, chi_r)

        # Contract into two-site tensor
        theta = jnp.einsum("asc,csd->ascd", A, B)  # (chi_l, 2, 2, chi_r)
        chi_l, _, _, chi_r = theta.shape

        # Apply gate: gate[s'1, s'2, s1, s2]
        gate_reshaped = gate  # (2, 2, 2, 2) = (s'1, s'2, s1, s2)
        theta_new = jnp.einsum("stpq,aptqb->asb", gate_reshaped, theta.reshape(chi_l, 2, 2, chi_r))
        # theta_new: (chi_l, 2, 2, chi_r)? Let me redo:
        theta_new = jnp.einsum("stpq,apqb->astb", gate_reshaped, theta.reshape(chi_l, 4, chi_r).reshape(chi_l, 2, 2, chi_r))

        # SVD to split back
        M = theta_new.reshape(chi_l * 2, 2 * chi_r)
        U, s, Vt = jnp.linalg.svd(M, full_matrices=False)
        chi_new = min(self.max_bond, U.shape[1])
        U = U[:, :chi_new]
        s_k = s[:chi_new]
        Vt = Vt[:chi_new, :]

        tensors[i] = (U * s_k[None, :]).reshape(chi_l, 2, chi_new)
        tensors[j] = Vt.reshape(chi_new, 2, chi_r)

        self._state = MatrixProductState(tensors, self._state.phys_dims)
        return self

    def measure_z(self, qubit: int) -> float:
        """Measure <Z> on a qubit."""
        from .mps import mps_expectation_single
        Z = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        return float(jnp.real(mps_expectation_single(self._state, Z, qubit)))

    def measure_all_z(self) -> jnp.ndarray:
        """Measure <Z_i> for all qubits."""
        from .mps import mps_magnetization
        return mps_magnetization(self._state)

    def entanglement_entropy(self, bond: int) -> float:
        """Von Neumann entanglement entropy at a given bond."""
        from .mps import mps_bond_entropies
        entropies = mps_bond_entropies(self._state)
        return float(entropies[bond])

    def run_vqc(
        self,
        params: jnp.ndarray,
        n_layers: int = 2,
    ) -> MatrixProductState:
        """
        Run a variational quantum circuit with given parameters.

        The circuit has n_layers of Ry rotations + CNOT entanglers.

        Parameters
        ----------
        params : (n_layers * n_qubits,) rotation angles
        n_layers : number of circuit layers

        Returns
        -------
        Final MPS state
        """
        self.reset()
        n = self.n_qubits
        param_idx = 0

        for layer in range(n_layers):
            # Single-qubit rotations
            for i in range(n):
                if param_idx < len(params):
                    self.ry(i, float(params[param_idx]))
                    param_idx += 1

            # Entangling CNOT layer
            for i in range(0, n - 1, 2):
                self.cnot(i, i + 1)
            for i in range(1, n - 1, 2):
                self.cnot(i, i + 1)

        return self.state


# ============================================================================
# Variational Portfolio Optimizer (VQE-inspired)
# ============================================================================

class VariationalPortfolioOptimizer:
    """
    Variational Quantum Eigensolver (VQE)-inspired portfolio optimizer.

    Frames portfolio optimization as a quantum eigenvalue problem:
    find the minimum energy state of a Hamiltonian encoding the
    Markowitz optimization:

      H = sum_{i,j} sigma_{ij} w_i w_j - mu sum_i w_i

    where sigma_{ij} is the covariance and mu_i are expected returns.

    Uses an MPS ansatz for the portfolio weights in the quantum amplitude basis.
    """

    def __init__(
        self,
        n_assets: int,
        n_qubits_per_asset: int = 3,
        bond_dim: int = 4,
        n_layers: int = 3,
        n_iter: int = 100,
        lr: float = 0.01,
        risk_aversion: float = 1.0,
    ):
        self.n_assets = n_assets
        self.n_qubits_per_asset = n_qubits_per_asset
        self.bond_dim = bond_dim
        self.n_layers = n_layers
        self.n_iter = n_iter
        self.lr = lr
        self.risk_aversion = risk_aversion
        self.n_qubits = n_assets * n_qubits_per_asset
        self.optimal_weights_: Optional[jnp.ndarray] = None
        self.training_energies_: List[float] = []
        self.is_fitted = False

    def fit(
        self,
        expected_returns: jnp.ndarray,
        covariance: jnp.ndarray,
        key: Optional[jax.random.KeyArray] = None,
    ) -> "VariationalPortfolioOptimizer":
        """
        Run VQE-inspired optimization for portfolio weights.

        Parameters
        ----------
        expected_returns : (n_assets,) mean return vector
        covariance : (n_assets, n_assets) covariance matrix
        key : random key

        Returns
        -------
        self
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        mu = jnp.array(expected_returns, dtype=jnp.float32)
        Sigma = jnp.array(covariance, dtype=jnp.float32)
        n = self.n_assets

        # Initialize variational parameters for the quantum circuit
        n_params = self.n_layers * self.n_qubits
        key, subkey = jax.random.split(key)
        params = jax.random.uniform(subkey, (n_params,), minval=0, maxval=2 * jnp.pi)

        # Set up quantum circuit
        circuit = QuantumCircuitSim(self.n_qubits, max_bond=self.bond_dim)

        def portfolio_energy(theta):
            """Energy = Markowitz objective in quantum basis."""
            # Run circuit
            state = circuit.run_vqc(theta, n_layers=self.n_layers)

            # Compute expectation of portfolio objective
            # Map qubit measurements to portfolio weights
            z_vals = jnp.array([
                float(jnp.real(mps_inner_product(
                    state,
                    _apply_z_operator(state, i)
                )))
                for i in range(self.n_qubits)
            ])

            # Aggregate to asset-level weights
            weights_raw = jnp.zeros(n)
            for a in range(n):
                qubits_a = z_vals[a * self.n_qubits_per_asset:(a + 1) * self.n_qubits_per_asset]
                weights_raw = weights_raw.at[a].set(jnp.mean((1 + qubits_a) / 2))

            # Normalize weights
            weights = weights_raw / (jnp.sum(jnp.abs(weights_raw)) + 1e-10)

            # Portfolio objective: minimize variance - risk_aversion * return
            variance = weights @ Sigma @ weights
            ret = weights @ mu
            energy = self.risk_aversion * variance - ret
            return energy, weights

        # Gradient descent on circuit parameters
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(params)

        for iteration in range(self.n_iter):
            # Compute gradient numerically (finite difference)
            eps = 1e-3
            energy_0, weights_0 = portfolio_energy(params)
            grad_params = jnp.zeros_like(params)
            for p_idx in range(min(len(params), 20)):  # Limit grad computation
                e_plus, _ = portfolio_energy(params.at[p_idx].set(params[p_idx] + eps))
                e_minus, _ = portfolio_energy(params.at[p_idx].set(params[p_idx] - eps))
                grad_params = grad_params.at[p_idx].set((e_plus - e_minus) / (2 * eps))

            updates, opt_state = optimizer.update(grad_params, opt_state)
            params = optax.apply_updates(params, updates)

            self.training_energies_.append(float(energy_0))

        _, final_weights = portfolio_energy(params)
        self.optimal_weights_ = final_weights
        self.is_fitted = True
        return self

    def get_weights(self) -> jnp.ndarray:
        """Return optimized portfolio weights."""
        assert self.is_fitted
        return self.optimal_weights_

    def sharpe_ratio(
        self,
        expected_returns: jnp.ndarray,
        covariance: jnp.ndarray,
        risk_free: float = 0.0,
    ) -> float:
        """
        Compute Sharpe ratio of the optimized portfolio.

        Parameters
        ----------
        expected_returns : (n_assets,) mean returns
        covariance : (n_assets, n_assets) covariance
        risk_free : risk-free rate

        Returns
        -------
        Sharpe ratio
        """
        assert self.is_fitted
        w = self.optimal_weights_
        ret = float(w @ expected_returns)
        vol = math.sqrt(float(w @ covariance @ w) + 1e-10)
        return (ret - risk_free) / vol


def _apply_z_operator(
    state: MatrixProductState,
    qubit: int,
) -> MatrixProductState:
    """Apply Z operator at a given qubit and return the resulting MPS."""
    tensors = [jnp.array(t) for t in state.tensors]
    Z = jnp.array([[1.0, 0.0], [0.0, -1.0]])
    t = tensors[qubit]
    tensors[qubit] = jnp.einsum("sp,lpq->lsq", Z, t)
    return MatrixProductState(tensors, state.phys_dims)


# ============================================================================
# MPS sampling (Born rule) -- module-level for import compatibility
# ============================================================================

def mps_sample(
    mps: MatrixProductState,
    n_samples: int,
    key: jax.random.KeyArray,
) -> jnp.ndarray:
    """
    Sample configurations from Born distribution P(x) = |psi(x)|^2.

    Parameters
    ----------
    mps : normalized MatrixProductState
    n_samples : number of samples
    key : random key

    Returns
    -------
    (n_samples, n_sites) integer array
    """
    from .mps import mps_sample as _mps_sample
    return _mps_sample(mps, n_samples, key)
