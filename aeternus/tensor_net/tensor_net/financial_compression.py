"""
financial_compression.py — Financial applications of MPS/TT for TensorNet.

Provides tools for compressing high-dimensional financial correlation structures,
detecting anomalies via reconstruction error, and regime-aware compression.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from tensor_net.mps import (
    MatrixProductState,
    mps_from_dense,
    mps_compress,
    mps_to_dense,
    mps_frobenius_error,
    mps_bond_entropies,
    mps_norm,
)
from tensor_net.tensor_train import (
    TensorTrain,
    tt_svd,
    tt_round,
    tt_to_dense,
    tt_norm,
    TensorTrainMatrix,
    ttm_from_matrix,
)


# ---------------------------------------------------------------------------
# CorrelationMPS
# ---------------------------------------------------------------------------

class CorrelationMPS:
    """
    Compress an N×N correlation matrix of financial assets as an MPS/TT.

    The correlation matrix is flattened to a vector of length N^2 or
    treated as a 2D TT-matrix. Compression via TT-SVD.

    Parameters
    ----------
    n_assets : number of assets
    max_bond : maximum bond dimension for TT/MPS compression
    window : rolling window size for estimating correlations
    """

    def __init__(
        self,
        n_assets: int,
        max_bond: int = 8,
        window: int = 252,
        cutoff: float = 1e-8,
    ):
        self.n_assets = n_assets
        self.max_bond = max_bond
        self.window = window
        self.cutoff = cutoff
        self.compressed_: Optional[TensorTrainMatrix] = None
        self.compression_error_: float = 0.0
        self.compression_ratio_: float = 1.0
        self.last_corr_matrix_: Optional[jnp.ndarray] = None

    def fit(self, returns: jnp.ndarray) -> "CorrelationMPS":
        """
        Fit MPS compression to a window of returns.

        Parameters
        ----------
        returns : array of shape (T, N) — T time steps, N assets

        Returns
        -------
        self
        """
        returns = jnp.array(returns, dtype=jnp.float32)
        T, N = returns.shape
        assert N == self.n_assets, f"Expected {self.n_assets} assets, got {N}"

        # Use last `window` bars
        w = min(self.window, T)
        ret_window = returns[-w:, :]

        # Compute empirical correlation matrix
        corr = self._compute_correlation(ret_window)
        self.last_corr_matrix_ = corr

        # Compress via TT-SVD (treating corr as 2D array factored by rows/cols)
        # Find factorization: N = d1 * d2 * ... dk where d_i ~ 2..5
        phys_dims = self._factorize_dim(N)
        self.phys_dims_ = phys_dims

        # Reshape N×N matrix to (d1*d2*...*dk, d1*d2*...*dk)
        # Then represent as TT-matrix
        # For simplicity, vectorize the upper triangle and compress as TT-vector
        # Alternative: treat corr as MPS via vectorization
        corr_vec = corr.reshape(-1)  # (N^2,)

        # Build physical dims for MPS: N^2 as product of small dims
        n_sq = N * N
        mps_dims = self._factorize_dim(n_sq)
        self.mps_dims_ = mps_dims

        # Compress via MPS
        mps = mps_from_dense(corr_vec, mps_dims, self.max_bond, self.cutoff)
        self.mps_ = mps

        # Also compress as TT-matrix
        try:
            self.ttm_ = ttm_from_matrix(
                corr,
                row_shape=tuple(phys_dims),
                col_shape=tuple(phys_dims),
                max_rank=self.max_bond,
                cutoff=self.cutoff,
            )
        except Exception:
            self.ttm_ = None

        # Compute compression error
        corr_reconstructed = self.decompress()
        diff = corr_reconstructed - corr
        self.compression_error_ = float(
            jnp.linalg.norm(diff) / (jnp.linalg.norm(corr) + 1e-12)
        )

        # Compression ratio
        n_dense = N * N
        n_compressed = mps.num_params()
        self.compression_ratio_ = n_dense / max(n_compressed, 1)

        return self

    def decompress(self) -> jnp.ndarray:
        """Reconstruct correlation matrix from compressed form."""
        N = self.n_assets
        dense_vec = mps_to_dense(self.mps_).reshape(-1)
        corr_reconstructed = dense_vec[:N * N].reshape(N, N)
        # Symmetrize
        corr_reconstructed = (corr_reconstructed + corr_reconstructed.T) / 2
        return corr_reconstructed

    def fit_rolling(
        self, returns: jnp.ndarray
    ) -> List[Dict]:
        """
        Fit MPS at each rolling window step.
        Returns list of dicts with time, error, ratio, bond_dims.
        """
        T, N = returns.shape
        results = []

        for t in range(self.window, T + 1):
            window_returns = returns[t - self.window:t, :]
            self.fit(window_returns)
            entropies = mps_bond_entropies(self.mps_)
            results.append({
                "t": t,
                "error": self.compression_error_,
                "ratio": self.compression_ratio_,
                "bond_dims": self.mps_.bond_dims,
                "max_bond_used": self.mps_.max_bond,
                "entropies": entropies.tolist(),
            })

        return results

    def _compute_correlation(self, returns: jnp.ndarray) -> jnp.ndarray:
        """Compute Pearson correlation matrix from returns."""
        returns_centered = returns - jnp.mean(returns, axis=0, keepdims=True)
        std = jnp.std(returns_centered, axis=0) + 1e-8
        returns_norm = returns_centered / std[None, :]
        corr = returns_norm.T @ returns_norm / returns.shape[0]
        return corr

    def _factorize_dim(self, N: int) -> List[int]:
        """
        Factorize N as a product of small integers for MPS physical dimensions.
        Tries to get approximately log2(N) sites with dim ~2-4.
        """
        dims = []
        remaining = N
        # Try powers of 2, then 3, then remainder
        for base in [4, 2, 3, 5]:
            while remaining % base == 0 and remaining > 1:
                dims.append(base)
                remaining //= base
        if remaining > 1:
            dims.append(remaining)
        if len(dims) == 0:
            dims = [N]
        return dims

    def variance_explained(self, k: int) -> float:
        """
        Fraction of variance explained by the top-k singular values
        at each bond of the MPS.
        """
        corr = self.last_corr_matrix_
        if corr is None:
            return 0.0
        # Use eigenvalues of correlation matrix
        eigvals = jnp.linalg.eigvalsh(corr)
        eigvals = jnp.sort(eigvals)[::-1]
        eigvals = jnp.maximum(eigvals, 0.0)
        total = jnp.sum(eigvals)
        top_k = jnp.sum(eigvals[:k])
        return float(top_k / (total + 1e-12))


# ---------------------------------------------------------------------------
# CausalityTensor
# ---------------------------------------------------------------------------

class CausalityTensor:
    """
    Compress the N×N×lag Granger causality tensor as a TT decomposition.

    The tensor C[i, j, tau] represents the Granger causality from asset j to asset i
    at lag tau. Compressing this reveals the dominant causal structure.
    """

    def __init__(
        self,
        n_assets: int,
        max_lags: int = 10,
        max_bond: int = 4,
        cutoff: float = 1e-8,
    ):
        self.n_assets = n_assets
        self.max_lags = max_lags
        self.max_bond = max_bond
        self.cutoff = cutoff
        self.tt_: Optional[TensorTrain] = None
        self.causality_tensor_: Optional[jnp.ndarray] = None

    def fit(self, returns: jnp.ndarray) -> "CausalityTensor":
        """
        Estimate and compress the Granger causality tensor.

        Parameters
        ----------
        returns : array of shape (T, N)
        """
        returns = jnp.array(returns, dtype=jnp.float32)
        T, N = returns.shape
        assert N == self.n_assets

        # Compute pairwise Granger causality (simplified: use cross-correlation as proxy)
        # Full Granger causality requires VAR model fitting
        # We use normalized cross-covariance as a proxy
        C = np.zeros((N, N, self.max_lags), dtype=np.float32)

        returns_np = np.array(returns)
        for i in range(N):
            for j in range(N):
                for lag in range(self.max_lags):
                    if lag == 0:
                        cov = float(np.cov(returns_np[:, i], returns_np[:, j])[0, 1])
                    else:
                        x = returns_np[lag:, i]
                        y = returns_np[:-lag, j]
                        cov = float(np.cov(x, y)[0, 1])
                    C[i, j, lag] = cov

        # Normalize
        max_val = np.abs(C).max()
        if max_val > 0:
            C = C / max_val

        self.causality_tensor_ = jnp.array(C)

        # Compress via TT-SVD
        self.tt_ = tt_svd(self.causality_tensor_, max_rank=self.max_bond, cutoff=self.cutoff)

        # Compute compression stats
        dense_reconstructed = tt_to_dense(self.tt_)
        diff = dense_reconstructed - self.causality_tensor_
        self.compression_error_ = float(
            jnp.linalg.norm(diff) / (jnp.linalg.norm(self.causality_tensor_) + 1e-12)
        )
        n_dense = N * N * self.max_lags
        self.compression_ratio_ = n_dense / max(self.tt_.n_params, 1)

        return self

    def dominant_causal_structure(self, threshold: float = 0.5) -> jnp.ndarray:
        """
        Return the dominant causal structure by summing over lags
        and thresholding.
        """
        if self.causality_tensor_ is None:
            raise RuntimeError("Call fit() first.")
        # Sum absolute values over lags
        C_sum = jnp.sum(jnp.abs(self.causality_tensor_), axis=2)
        # Threshold
        return (C_sum > threshold).astype(jnp.float32)

    def causal_graph(self) -> np.ndarray:
        """
        Return the causal adjacency matrix (i→j has edge if C[j,i,:] is large).
        """
        C = np.array(self.dominant_causal_structure(threshold=0.3))
        return C


# ---------------------------------------------------------------------------
# DependencyHypercube
# ---------------------------------------------------------------------------

class DependencyHypercube:
    """
    Compress the 15-signal Feature Hypercube from Event Horizon as MPS.

    The Feature Hypercube is a high-dimensional tensor T[s1, s2, ..., s15]
    where each s_i is a discretized signal value (e.g., buckets 0..B-1).
    The tensor captures joint dependencies between all 15 signals.

    Compression via MPS with bond dimension D reveals dominant dependencies.
    """

    def __init__(
        self,
        n_signals: int = 15,
        n_buckets: int = 4,
        max_bond: int = 8,
        cutoff: float = 1e-8,
    ):
        self.n_signals = n_signals
        self.n_buckets = n_buckets
        self.max_bond = max_bond
        self.cutoff = cutoff
        self.mps_: Optional[MatrixProductState] = None
        self.compression_error_: float = 0.0
        self.compression_ratio_: float = 1.0

    def fit(self, signal_data: jnp.ndarray) -> "DependencyHypercube":
        """
        Build and compress the Feature Hypercube from signal time series.

        Parameters
        ----------
        signal_data : array of shape (T, n_signals)
            Each column is a signal; values will be discretized into buckets.
        """
        signal_data = jnp.array(signal_data, dtype=jnp.float32)
        T, S = signal_data.shape
        assert S == self.n_signals, f"Expected {self.n_signals} signals, got {S}"

        # Discretize signals into n_buckets bins
        signal_np = np.array(signal_data)
        bucket_data = np.zeros_like(signal_np, dtype=int)
        for s in range(S):
            col = signal_np[:, s]
            percentiles = np.percentile(col, np.linspace(0, 100, self.n_buckets + 1))
            percentiles[-1] += 1e-6  # Ensure last bin is inclusive
            bucket_data[:, s] = np.digitize(col, percentiles[1:-1])

        # Build joint frequency tensor
        # Shape: (n_buckets,) * n_signals — too large to store fully for 15 signals
        # Instead, build low-rank approximation via random sampling
        B = self.n_buckets
        N = self.n_signals

        # Use marginal/pairwise approach for large N
        # For N≤10, can build exact tensor; for N>10, use TT-cross approximation
        if N <= 8:
            # Build full joint frequency tensor
            tensor_shape = (B,) * N
            freq_tensor = np.zeros(tensor_shape, dtype=np.float32)
            for t in range(T):
                idx = tuple(int(bucket_data[t, s]) for s in range(N))
                freq_tensor[idx] += 1
            freq_tensor /= T  # Normalize to probability

            # Compress via MPS
            self.mps_ = mps_from_dense(
                jnp.array(freq_tensor),
                phys_dims=[B] * N,
                max_bond=self.max_bond,
                cutoff=self.cutoff,
            )

            # Compute error
            reconstructed = mps_to_dense(self.mps_)
            self.compression_error_ = float(
                jnp.linalg.norm(reconstructed - freq_tensor) /
                (jnp.linalg.norm(freq_tensor) + 1e-12)
            )
            n_dense = B ** N
            self.compression_ratio_ = n_dense / max(self.mps_.num_params(), 1)

        else:
            # For large N, use pairwise tensor as proxy
            # Build pairwise joint distribution tensor: (N, N, B, B)
            pairwise = np.zeros((N, N, B, B), dtype=np.float32)
            for t in range(T):
                for i in range(N):
                    for j in range(N):
                        pairwise[i, j, int(bucket_data[t, i]), int(bucket_data[t, j])] += 1
            pairwise /= T

            # Compress each pairwise block
            self.pairwise_ = jnp.array(pairwise)

            # Build a chain MPS from the pairwise structure (approximate)
            # Each site has dim B; tensors initialized from pairwise marginals
            tensors = []
            for s in range(N):
                # marginal distribution at site s
                marginal = np.sum(pairwise[s, s, :, :], axis=1)
                marginal /= marginal.sum() + 1e-12
                # Shape: (1, B, 1) — product state initialization
                t_k = jnp.array(marginal).reshape(1, B, 1)
                tensors.append(t_k)

            self.mps_ = MatrixProductState(tensors, (B,) * N)
            self.compression_error_ = 0.0
            self.compression_ratio_ = float(B ** N) / max(self.mps_.num_params(), 1)

        return self

    def query_joint_prob(self, signal_values: List[int]) -> float:
        """
        Query the compressed joint probability distribution.
        signal_values: list of bucket indices (one per signal)
        """
        if self.mps_ is None:
            raise RuntimeError("Call fit() first.")
        # Contract MPS with specific index
        result = 1.0
        for i, (tensor, val) in enumerate(zip(self.mps_.tensors, signal_values)):
            # tensor: (chi_l, d, chi_r)
            result = result * float(tensor[0, val, 0]) if tensor.shape[0] == 1 else result
        dense = mps_to_dense(self.mps_)
        idx = tuple(signal_values)
        try:
            return float(dense[idx])
        except (IndexError, TypeError):
            return 0.0


# ---------------------------------------------------------------------------
# StreamingCompressor
# ---------------------------------------------------------------------------

class StreamingCompressor:
    """
    Online MPS update as new data arrives (streaming compression).

    Supports rank-1 updates: given a new data vector x,
    update the MPS to incorporate x with minimal recompression.
    """

    def __init__(
        self,
        n_features: int,
        max_bond: int = 8,
        phys_dims: Optional[List[int]] = None,
        forget_factor: float = 0.99,
        recompression_interval: int = 50,
    ):
        self.n_features = n_features
        self.max_bond = max_bond
        self.forget_factor = forget_factor
        self.recompression_interval = recompression_interval
        self.n_updates_ = 0
        self.mps_: Optional[MatrixProductState] = None

        if phys_dims is not None:
            self.phys_dims = phys_dims
        else:
            # Auto-factorize n_features
            self.phys_dims = self._factorize_dim(n_features)

    def initialize(self, x: jnp.ndarray) -> "StreamingCompressor":
        """Initialize MPS from first data point."""
        x = jnp.array(x, dtype=jnp.float32)
        x_padded = self._pad_to_shape(x)
        self.mps_ = mps_from_dense(x_padded, self.phys_dims, self.max_bond)
        self.n_updates_ = 1
        return self

    def update(self, x: jnp.ndarray) -> "StreamingCompressor":
        """
        Incorporate new data point x into the streaming MPS.

        Uses exponential forgetting: new_mps ≈ forget * old_mps + (1-forget) * rank1_mps(x)
        Then recompress to maintain bounded bond dimension.
        """
        x = jnp.array(x, dtype=jnp.float32)
        x_padded = self._pad_to_shape(x)

        if self.mps_ is None:
            return self.initialize(x)

        # Build rank-1 MPS from new data point
        new_mps = mps_from_dense(x_padded, self.phys_dims, max_bond=1)

        # Add with forgetting: mps ← forget * mps + (1-forget) * new_mps
        from tensor_net.mps import mps_add, mps_scale
        combined = mps_add(
            mps_scale(self.mps_, self.forget_factor),
            mps_scale(new_mps, 1.0 - self.forget_factor),
        )

        # Recompress periodically (or always if bond dims get too large)
        if combined.max_bond > 2 * self.max_bond or \
                self.n_updates_ % self.recompression_interval == 0:
            self.mps_, _ = mps_compress(combined, self.max_bond)
        else:
            self.mps_ = combined

        self.n_updates_ += 1
        return self

    def reconstruct(self, x: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct a data vector using the compressed MPS."""
        return mps_to_dense(self.mps_).reshape(-1)[:self.n_features]

    def reconstruction_error(self, x: jnp.ndarray) -> float:
        """Compute reconstruction error for a data vector."""
        x = jnp.array(x, dtype=jnp.float32)
        x_padded = self._pad_to_shape(x)
        reconstructed = mps_to_dense(self.mps_).reshape(-1)
        err = jnp.linalg.norm(reconstructed - x_padded)
        return float(err / (jnp.linalg.norm(x_padded) + 1e-12))

    def _pad_to_shape(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pad x to match the product of phys_dims."""
        target_size = 1
        for d in self.phys_dims:
            target_size *= d
        if x.size < target_size:
            x_padded = jnp.concatenate([
                x.reshape(-1),
                jnp.zeros(target_size - x.size)
            ])
        else:
            x_padded = x.reshape(-1)[:target_size]
        return x_padded

    def _factorize_dim(self, N: int) -> List[int]:
        dims = []
        remaining = N
        for base in [4, 2, 3, 5]:
            while remaining % base == 0 and remaining > 1:
                dims.append(base)
                remaining //= base
        if remaining > 1:
            dims.append(remaining)
        if len(dims) == 0:
            dims = [N]
        return dims


# ---------------------------------------------------------------------------
# RegimeCompression
# ---------------------------------------------------------------------------

class RegimeCompression:
    """
    Maintain a separate MPS per market regime.
    Switch between MPS models based on regime detector signal.

    Regimes are labeled 0, 1, ..., n_regimes-1.
    """

    def __init__(
        self,
        n_regimes: int,
        n_assets: int,
        max_bond: int = 8,
        window: int = 252,
    ):
        self.n_regimes = n_regimes
        self.n_assets = n_assets
        self.max_bond = max_bond
        self.window = window
        self.regime_compressors_: Dict[int, CorrelationMPS] = {}
        self.regime_counts_: Dict[int, int] = {r: 0 for r in range(n_regimes)}
        self.current_regime_: int = 0

    def fit_regime(
        self,
        regime_id: int,
        returns: jnp.ndarray,
    ) -> "RegimeCompression":
        """Fit compression model for a specific regime."""
        assert 0 <= regime_id < self.n_regimes
        comp = CorrelationMPS(
            n_assets=self.n_assets,
            max_bond=self.max_bond,
            window=min(self.window, returns.shape[0]),
        )
        comp.fit(returns)
        self.regime_compressors_[regime_id] = comp
        self.regime_counts_[regime_id] = returns.shape[0]
        return self

    def fit_all_regimes(
        self,
        returns: jnp.ndarray,
        regime_labels: jnp.ndarray,
    ) -> "RegimeCompression":
        """
        Fit separate MPS for each regime.

        Parameters
        ----------
        returns : array of shape (T, N)
        regime_labels : array of shape (T,) with integer regime labels
        """
        returns = jnp.array(returns, dtype=jnp.float32)
        regime_labels = np.array(regime_labels)

        for r in range(self.n_regimes):
            mask = regime_labels == r
            if mask.sum() > 20:  # Need minimum data
                regime_returns = returns[mask]
                self.fit_regime(r, regime_returns)

        return self

    def switch_regime(self, regime_id: int) -> "RegimeCompression":
        """Switch to a different regime model."""
        assert regime_id in self.regime_compressors_, (
            f"Regime {regime_id} not fitted."
        )
        self.current_regime_ = regime_id
        return self

    def get_current_correlation(self) -> Optional[jnp.ndarray]:
        """Return compressed correlation matrix for current regime."""
        if self.current_regime_ not in self.regime_compressors_:
            return None
        return self.regime_compressors_[self.current_regime_].decompress()

    def regime_similarity(self, r1: int, r2: int) -> float:
        """
        Compute similarity between two regime MPS via inner product
        of their compressed correlation representations.
        """
        if r1 not in self.regime_compressors_ or r2 not in self.regime_compressors_:
            return 0.0
        c1 = self.regime_compressors_[r1].last_corr_matrix_.reshape(-1)
        c2 = self.regime_compressors_[r2].last_corr_matrix_.reshape(-1)
        dot = float(jnp.dot(c1, c2))
        norm1 = float(jnp.linalg.norm(c1))
        norm2 = float(jnp.linalg.norm(c2))
        return dot / (norm1 * norm2 + 1e-12)

    def predict_regime(self, returns_window: jnp.ndarray) -> int:
        """
        Predict the most likely regime for new returns data.
        Uses minimum reconstruction error across all regime MPS.
        """
        returns_window = jnp.array(returns_window, dtype=jnp.float32)
        best_regime = 0
        best_error = float("inf")

        # Compute empirical correlation
        temp_comp = CorrelationMPS(self.n_assets, window=returns_window.shape[0])
        temp_comp.fit(returns_window)
        corr_new = temp_comp.last_corr_matrix_.reshape(-1)

        for r, comp in self.regime_compressors_.items():
            corr_r = comp.decompress().reshape(-1)
            err = float(jnp.linalg.norm(corr_new - corr_r))
            if err < best_error:
                best_error = err
                best_regime = r

        return best_regime


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Anomaly detection via MPS reconstruction error.

    The idea: fit MPS to "normal" market behavior.
    When new data arrives, compute reconstruction error under the compressed MPS.
    High reconstruction error = data is structurally unusual = potential crisis signal.

    This captures non-linear dependencies that PCA misses.
    """

    def __init__(
        self,
        n_assets: int,
        max_bond: int = 8,
        window: int = 252,
        detection_window: int = 20,
        z_score_threshold: float = 2.5,
    ):
        self.n_assets = n_assets
        self.max_bond = max_bond
        self.window = window
        self.detection_window = detection_window
        self.z_score_threshold = z_score_threshold
        self.baseline_mps_: Optional[MatrixProductState] = None
        self.baseline_corr_: Optional[jnp.ndarray] = None
        self.error_history_: List[float] = []
        self.anomaly_scores_: List[float] = []

    def fit_baseline(self, returns: jnp.ndarray) -> "AnomalyDetector":
        """
        Fit baseline MPS from normal-period returns.

        Parameters
        ----------
        returns : array of shape (T, N) of normal-period returns
        """
        returns = jnp.array(returns, dtype=jnp.float32)
        T, N = returns.shape
        assert N == self.n_assets

        # Compute baseline correlation
        comp = CorrelationMPS(N, self.max_bond, min(self.window, T))
        comp.fit(returns)
        self.baseline_mps_ = comp.mps_
        self.baseline_corr_ = comp.last_corr_matrix_
        self.baseline_mps_dims_ = comp.mps_dims_

        # Compute baseline error distribution
        baseline_errors = []
        step = max(1, T // 50)
        for t in range(self.window, T, step):
            win_ret = returns[t - self.detection_window:t, :]
            err = self._compute_reconstruction_error(win_ret)
            baseline_errors.append(err)

        self.baseline_error_mean_ = float(np.mean(baseline_errors)) if baseline_errors else 0.0
        self.baseline_error_std_ = float(np.std(baseline_errors)) if baseline_errors else 1.0

        return self

    def score(self, returns_window: jnp.ndarray) -> float:
        """
        Compute anomaly score for a window of returns.

        Returns a z-score: (error - baseline_mean) / baseline_std
        Positive values indicate unusual behavior.
        """
        err = self._compute_reconstruction_error(returns_window)
        self.error_history_.append(err)
        z_score = (err - self.baseline_error_mean_) / (self.baseline_error_std_ + 1e-8)
        self.anomaly_scores_.append(z_score)
        return z_score

    def score_sequence(
        self,
        returns: jnp.ndarray,
        step: int = 1,
    ) -> jnp.ndarray:
        """
        Compute anomaly scores over a rolling window on a returns time series.

        Parameters
        ----------
        returns : array of shape (T, N)
        step : step size for rolling window

        Returns
        -------
        scores : array of shape (n_windows,)
        timestamps : array of corresponding time indices
        """
        T, N = returns.shape
        scores = []
        timestamps = []

        for t in range(self.detection_window, T + 1, step):
            win = returns[t - self.detection_window:t, :]
            z = self.score(win)
            scores.append(z)
            timestamps.append(t)

        return jnp.array(scores), jnp.array(timestamps)

    def is_anomaly(self, returns_window: jnp.ndarray) -> bool:
        """Return True if the window exhibits anomalous behavior."""
        z = self.score(returns_window)
        return z > self.z_score_threshold

    def _compute_reconstruction_error(self, returns_window: jnp.ndarray) -> float:
        """
        Compute the Frobenius reconstruction error:
        ||corr(returns_window) - decompress(compress(corr))||_F
        """
        if self.baseline_mps_ is None:
            raise RuntimeError("Call fit_baseline() first.")

        returns_window = jnp.array(returns_window, dtype=jnp.float32)
        T, N = returns_window.shape

        if T < 2:
            return 0.0

        # Compute correlation of the new window
        ret_centered = returns_window - jnp.mean(returns_window, axis=0, keepdims=True)
        std = jnp.std(ret_centered, axis=0) + 1e-8
        ret_norm = ret_centered / std[None, :]
        corr_new = ret_norm.T @ ret_norm / T  # (N, N)

        # Reconstruct from baseline MPS
        corr_baseline = mps_to_dense(self.baseline_mps_).reshape(-1)
        N_sq = N * N
        corr_baseline_mat = corr_baseline[:N_sq].reshape(N, N)

        # Error
        diff = corr_new - corr_baseline_mat
        err = float(jnp.linalg.norm(diff) / (jnp.linalg.norm(corr_new) + 1e-12))
        return err

    def compare_pca_detector(
        self,
        returns: jnp.ndarray,
        n_components: int = 5,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compare MPS anomaly detector vs PCA-based reconstruction error.

        Parameters
        ----------
        returns : array of shape (T, N)
        n_components : number of PCA components for reconstruction

        Returns
        -------
        dict with 'mps_scores', 'pca_scores', 'timestamps'
        """
        T, N = returns.shape
        mps_scores, timestamps = self.score_sequence(returns)

        # PCA detector
        pca_scores = []
        for t in range(self.detection_window, T + 1):
            win = np.array(returns[t - self.detection_window:t, :])
            # Compute correlation
            win_c = win - win.mean(0)
            win_std = win_c.std(0) + 1e-8
            win_n = win_c / win_std
            corr = win_n.T @ win_n / len(win_n)

            # PCA reconstruction
            eigvals, eigvecs = np.linalg.eigh(corr)
            top_idx = np.argsort(eigvals)[::-1][:n_components]
            V = eigvecs[:, top_idx]
            corr_recon = V @ np.diag(eigvals[top_idx]) @ V.T

            err = np.linalg.norm(corr - corr_recon) / (np.linalg.norm(corr) + 1e-12)
            pca_scores.append(err)

        pca_scores = jnp.array(pca_scores)
        pca_mean = float(jnp.mean(pca_scores[:len(pca_scores)//2]))
        pca_std = float(jnp.std(pca_scores[:len(pca_scores)//2])) + 1e-8
        pca_z = (pca_scores - pca_mean) / pca_std

        return {
            "mps_scores": mps_scores,
            "pca_scores": pca_z,
            "timestamps": timestamps,
        }


# ---------------------------------------------------------------------------
# FinancialMPS experiment runner
# ---------------------------------------------------------------------------

def run_financial_mps_experiment(
    n_assets: int = 30,
    n_bars: int = 1000,
    max_bond: int = 8,
    window: int = 500,
    seed: int = 42,
) -> Dict:
    """
    Full financial MPS experiment:
    1. Generate synthetic returns for n_assets
    2. Compress rolling correlation matrices as MPS
    3. Track error vs time

    Returns
    -------
    dict with keys: returns, results, compression_errors, bond_dims, timestamps
    """
    np.random.seed(seed)

    # Generate correlated returns with regime changes
    # 3 regimes: low vol, high vol, crisis
    n1, n2, n3 = n_bars // 3, n_bars // 3, n_bars - 2 * (n_bars // 3)

    # Regime 1: Low vol, mild correlation
    cov1 = 0.01 * (0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets))
    L1 = np.linalg.cholesky(cov1 + 1e-6 * np.eye(n_assets))
    ret1 = np.random.randn(n1, n_assets) @ L1.T

    # Regime 2: Higher vol, stronger correlation
    cov2 = 0.04 * (0.6 * np.ones((n_assets, n_assets)) + 0.4 * np.eye(n_assets))
    L2 = np.linalg.cholesky(cov2 + 1e-6 * np.eye(n_assets))
    ret2 = np.random.randn(n2, n_assets) @ L2.T

    # Regime 3: Crisis — very high correlation, fat tails
    cov3 = 0.16 * (0.85 * np.ones((n_assets, n_assets)) + 0.15 * np.eye(n_assets))
    L3 = np.linalg.cholesky(cov3 + 1e-6 * np.eye(n_assets))
    ret3 = np.random.randn(n3, n_assets) @ L3.T
    # Add fat tail shocks
    shock_times = np.random.choice(n3, size=n3 // 20, replace=False)
    ret3[shock_times] *= 3.0

    returns = np.vstack([ret1, ret2, ret3])

    # Fit rolling compression
    comp = CorrelationMPS(n_assets, max_bond=max_bond, window=window)
    results = comp.fit_rolling(jnp.array(returns))

    timestamps = [r["t"] for r in results]
    errors = [r["error"] for r in results]
    ratios = [r["ratio"] for r in results]
    bond_dims_max = [r["max_bond_used"] for r in results]

    # Anomaly detection
    anomaly_det = AnomalyDetector(n_assets, max_bond=max_bond, window=window // 2,
                                  detection_window=20)
    anomaly_det.fit_baseline(jnp.array(returns[:n1 + n2]))
    mps_scores, score_times = anomaly_det.score_sequence(
        jnp.array(returns), step=5
    )

    return {
        "returns": returns,
        "results": results,
        "timestamps": timestamps,
        "compression_errors": errors,
        "compression_ratios": ratios,
        "bond_dims_max": bond_dims_max,
        "anomaly_scores": mps_scores,
        "score_times": score_times,
        "regime_boundaries": [n1, n1 + n2, n_bars],
        "n_assets": n_assets,
        "max_bond": max_bond,
    }
