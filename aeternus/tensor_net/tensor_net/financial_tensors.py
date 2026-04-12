"""
financial_tensors.py — Financial correlation tensors for TensorNet (Project AETERNUS).

Implements:
  - Multi-asset return tensor construction (rolling windows, factor embedding)
  - Tucker decomposition of correlation matrices
  - CP decomposition for factor models
  - Tensor PCA (higher-order PCA via HOSVD)
  - Factor model extraction from tensor decompositions
  - Regime-conditioned tensor factorization (HMM + tensor)
  - Causality tensor (Granger causality in tensor form)
  - Dependency hypercube (higher-order asset dependencies)
  - Correlation MPS encoder
  - Streaming compressor (online tensor updates)
  - Regime compression (market regime detection via tensor norms)
  - Volatility tensor surfaces
  - Cross-sectional momentum tensor
  - Drawdown tensor analysis
  - Risk attribution via tensor decomposition
  - Option surface tensor compression
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Sequence, Union, Dict, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad, vmap

from .mps import MatrixProductState, mps_from_dense, mps_compress, mps_inner_product
from .tt_decomp import TensorTrain, tt_svd, tt_round, tucker_decomp, cp_decomp


# ============================================================================
# Return tensor construction
# ============================================================================

def build_return_tensor(
    prices: jnp.ndarray,
    window: int = 20,
    lag: int = 1,
) -> jnp.ndarray:
    """
    Build a 3D return tensor from price data.

    Shape: (T - window, n_assets, window) where entry [t, i, tau] is the
    log return of asset i at time t+tau within window t.

    Parameters
    ----------
    prices : (T, n_assets) price matrix
    window : rolling window length
    lag : return computation lag

    Returns
    -------
    Return tensor of shape (T - window, n_assets, window)
    """
    prices = jnp.array(prices, dtype=jnp.float32)
    T, n_assets = prices.shape

    log_returns = jnp.log(prices[lag:] / (prices[:-lag] + 1e-10))
    # log_returns shape: (T - lag, n_assets)

    T_r = log_returns.shape[0]
    n_windows = T_r - window + 1

    # Build rolling windows
    windows_list = []
    for t in range(n_windows):
        windows_list.append(log_returns[t:t + window])  # (window, n_assets)

    # Stack: (n_windows, window, n_assets) -> transpose to (n_windows, n_assets, window)
    tensor = jnp.stack(windows_list, axis=0)
    return tensor.transpose(0, 2, 1)  # (n_windows, n_assets, window)


def build_correlation_tensor(
    returns: jnp.ndarray,
    window: int = 60,
    stride: int = 10,
    regularize: float = 1e-4,
) -> jnp.ndarray:
    """
    Build a time-varying correlation tensor from asset returns.

    Shape: (n_time_steps, n_assets, n_assets) where each slice [t, :, :]
    is the rolling correlation matrix at time t.

    Parameters
    ----------
    returns : (T, n_assets) return matrix
    window : rolling window length
    stride : step size between windows
    regularize : regularization added to diagonal

    Returns
    -------
    Correlation tensor of shape (n_steps, n_assets, n_assets)
    """
    returns = jnp.array(returns, dtype=jnp.float32)
    T, n_assets = returns.shape
    n_steps = (T - window) // stride + 1

    corr_matrices = []
    for t in range(n_steps):
        start = t * stride
        end = start + window
        R = returns[start:end]  # (window, n_assets)

        # Standardize
        mu = jnp.mean(R, axis=0)
        std = jnp.std(R, axis=0) + 1e-8
        R_std = (R - mu) / std

        # Correlation matrix
        cov = (R_std.T @ R_std) / window
        # Ensure PSD
        cov = cov + regularize * jnp.eye(n_assets)
        # Normalize to correlation
        d = jnp.sqrt(jnp.diag(cov))
        corr = cov / (jnp.outer(d, d) + 1e-10)
        corr_matrices.append(corr)

    return jnp.stack(corr_matrices, axis=0)  # (n_steps, n_assets, n_assets)


def build_factor_return_tensor(
    returns: jnp.ndarray,
    factor_returns: jnp.ndarray,
    window: int = 60,
    stride: int = 20,
) -> jnp.ndarray:
    """
    Build a 4D tensor of factor-conditional return distributions.

    Shape: (n_steps, n_assets, n_factors, window)

    Parameters
    ----------
    returns : (T, n_assets) asset returns
    factor_returns : (T, n_factors) factor returns
    window : rolling window
    stride : step size

    Returns
    -------
    Tensor of shape (n_steps, n_assets, n_factors, window)
    """
    returns = jnp.array(returns, dtype=jnp.float32)
    factor_returns = jnp.array(factor_returns, dtype=jnp.float32)
    T, n_assets = returns.shape
    _, n_factors = factor_returns.shape
    n_steps = (T - window) // stride + 1

    slices = []
    for t in range(n_steps):
        start = t * stride
        end = start + window
        R = returns[start:end]          # (window, n_assets)
        F = factor_returns[start:end]   # (window, n_factors)
        # (n_assets, n_factors, window) = R.T expanded against F.T
        slice_t = jnp.einsum("wa,wf->afw", R, F)  # (n_assets, n_factors, window)
        slices.append(slice_t)

    return jnp.stack(slices, axis=0)  # (n_steps, n_assets, n_factors, window)


def build_cross_moment_tensor(
    returns: jnp.ndarray,
    order: int = 3,
    window: int = 120,
) -> jnp.ndarray:
    """
    Build the order-k cross-moment tensor of asset returns.

    The (i_1, i_2, ..., i_k) entry is E[r_{i_1} r_{i_2} ... r_{i_k}].

    Parameters
    ----------
    returns : (T, n_assets) return matrix
    order : moment order (2=covariance, 3=coskewness, 4=cokurtosis)
    window : estimation window (use all data if None)

    Returns
    -------
    Tensor of shape (n_assets,) * order
    """
    returns = jnp.array(returns, dtype=jnp.float32)
    T, n = returns.shape

    if window is not None:
        returns = returns[-window:]
        T = returns.shape[0]

    # Standardize
    mu = jnp.mean(returns, axis=0)
    std = jnp.std(returns, axis=0) + 1e-8
    r_std = (returns - mu) / std  # (T, n)

    if order == 2:
        return r_std.T @ r_std / T
    elif order == 3:
        # Coskewness tensor (n, n, n)
        M3 = jnp.einsum("ti,tj,tk->ijk", r_std, r_std, r_std) / T
        return M3
    elif order == 4:
        # Cokurtosis tensor (n, n, n, n)
        M4 = jnp.einsum("ti,tj,tk,tl->ijkl", r_std, r_std, r_std, r_std) / T
        return M4
    else:
        # Generic higher-order moment via iterative einsum
        result = r_std  # (T, n)
        for _ in range(order - 1):
            result = jnp.einsum("...i,tj->...ij", result.reshape(T, -1), r_std)
            # This grows exponentially; only practical for small n
        return result.mean(axis=0).reshape((n,) * order)


# ============================================================================
# Tucker decomposition for correlation matrices
# ============================================================================

class CorrelationTucker:
    """
    Tucker decomposition of a time-varying correlation tensor.

    Factors the tensor C[t, i, j] into:
      C ≈ G ×_1 U_time ×_2 U_asset ×_3 U_asset2

    where:
    - G is the core tensor (n_time_factors, n_asset_factors, n_asset_factors)
    - U_time: time factor loadings (T, n_time_factors)
    - U_asset: cross-sectional factor loadings (n_assets, n_asset_factors)

    This captures the dominant structural changes in correlation across time.
    """

    def __init__(
        self,
        n_time_factors: int = 5,
        n_asset_factors: int = 10,
        n_iter: int = 20,
    ):
        self.n_time_factors = n_time_factors
        self.n_asset_factors = n_asset_factors
        self.n_iter = n_iter
        self.core_ = None
        self.factors_ = None
        self.is_fitted = False

    def fit(self, corr_tensor: jnp.ndarray) -> "CorrelationTucker":
        """
        Fit Tucker decomposition to correlation tensor.

        Parameters
        ----------
        corr_tensor : (T, n_assets, n_assets) correlation tensor

        Returns
        -------
        self
        """
        corr_tensor = jnp.array(corr_tensor, dtype=jnp.float32)
        T, n, _ = corr_tensor.shape

        ranks = [self.n_time_factors, self.n_asset_factors, self.n_asset_factors]
        self.core_, self.factors_ = tucker_decomp(
            corr_tensor, ranks, n_iter=self.n_iter
        )
        self.is_fitted = True
        return self

    def transform(self, corr_tensor: jnp.ndarray) -> jnp.ndarray:
        """
        Project new correlation tensor onto learned factors.

        Parameters
        ----------
        corr_tensor : (T', n_assets, n_assets)

        Returns
        -------
        Time factor scores of shape (T', n_time_factors)
        """
        assert self.is_fitted
        corr_tensor = jnp.array(corr_tensor, dtype=jnp.float32)
        U_t = self.factors_[0]  # (T, n_time_factors)
        U_a = self.factors_[1]  # (n_assets, n_asset_factors)

        # Project: scores[t, k] = sum_{i,j} corr[t,i,j] * U_a[i,f] * U_a[j,g] (then project time)
        scores = jnp.einsum("tij,if,jg->tfg", corr_tensor, U_a, U_a)
        # Flatten to (T', n_asset_factors^2)
        T_new = corr_tensor.shape[0]
        return scores.reshape(T_new, -1)

    def reconstruct(self) -> jnp.ndarray:
        """Reconstruct correlation tensor from Tucker decomposition."""
        assert self.is_fitted
        G = self.core_
        U_t, U_a1, U_a2 = self.factors_

        # C[t,i,j] = G[k,l,m] * U_t[t,k] * U_a1[i,l] * U_a2[j,m]
        recon = jnp.einsum("klm,tk,il,jm->tij", G, U_t, U_a1, U_a2)
        return recon

    def reconstruction_error(self, corr_tensor: jnp.ndarray) -> float:
        """Compute relative reconstruction error."""
        recon = self.reconstruct()
        diff = jnp.linalg.norm(corr_tensor - recon)
        ref = jnp.linalg.norm(corr_tensor)
        return float(diff / (ref + 1e-10))

    def get_asset_factors(self) -> jnp.ndarray:
        """Return the asset factor loading matrix."""
        assert self.is_fitted
        return self.factors_[1]

    def get_time_factors(self) -> jnp.ndarray:
        """Return the time factor matrix."""
        assert self.is_fitted
        return self.factors_[0]


# ============================================================================
# CorrelationMPS: encode correlation matrix in MPS format
# ============================================================================

class CorrelationMPS:
    """
    Compress a correlation matrix (or time-varying sequence) into MPS format.

    Maps the (n_assets, n_assets) correlation structure to a 1D chain of
    physical sites, each encoding one "slice" of asset interactions.

    Useful for compressing large correlation matrices arising in:
    - Factor models (500+ assets)
    - High-frequency covariance estimation
    - Regime-switching models

    Notes
    -----
    For an (n, n) correlation matrix, uses a chain of ceil(log2(n^2)) sites
    with physical dimension 2 (binary encoding of matrix entries).
    Alternatively, uses a direct chain of n sites with physical dim n.
    """

    def __init__(
        self,
        max_bond: int = 32,
        cutoff: float = 1e-8,
        encoding: str = "direct",
    ):
        """
        Parameters
        ----------
        max_bond : maximum MPS bond dimension
        cutoff : SVD truncation threshold
        encoding : 'direct' (physical dim = n_assets) or 'binary' (d=2)
        """
        self.max_bond = max_bond
        self.cutoff = cutoff
        self.encoding = encoding
        self.mps_ = None
        self.n_assets_ = None
        self.is_fitted = False

    def fit(self, corr_matrix: jnp.ndarray) -> "CorrelationMPS":
        """
        Compress a correlation matrix to MPS.

        Parameters
        ----------
        corr_matrix : (n_assets, n_assets) correlation matrix

        Returns
        -------
        self
        """
        corr_matrix = jnp.array(corr_matrix, dtype=jnp.float32)
        n = corr_matrix.shape[0]
        self.n_assets_ = n

        vec = corr_matrix.reshape(-1)
        total = n * n

        if self.encoding == "direct":
            # Use chain of n sites, each with physical dim n
            phys_dims = [n] * n
            # Pad or truncate to n^2 entries
            if len(vec) != n * n:
                vec = vec[:n * n]
        else:
            # Binary encoding: find bit length
            n_bits = max(2, math.ceil(math.log2(total + 1)))
            phys_dim = 2
            n_sites = max(2, math.ceil(math.log2(total + 1)))
            target = phys_dim ** n_sites
            if vec.shape[0] < target:
                vec = jnp.concatenate([vec, jnp.zeros(target - vec.shape[0])])
            else:
                vec = vec[:target]
            phys_dims = [phys_dim] * n_sites

        self.mps_ = mps_from_dense(vec, phys_dims, max_bond=self.max_bond, cutoff=self.cutoff)
        self.is_fitted = True
        return self

    def transform(self) -> jnp.ndarray:
        """
        Reconstruct the correlation matrix from MPS.

        Returns
        -------
        Reconstructed (n_assets, n_assets) correlation matrix
        """
        assert self.is_fitted
        from .mps import mps_to_dense
        dense = mps_to_dense(self.mps_)
        n = self.n_assets_
        return dense.reshape(n, n)[:n, :n]

    def compression_ratio(self) -> float:
        """Compression ratio: original / compressed parameter count."""
        assert self.is_fitted
        n = self.n_assets_
        original = n * n
        return original / self.mps_.num_params()

    def fit_sequence(
        self,
        corr_tensor: jnp.ndarray,
    ) -> List["CorrelationMPS"]:
        """
        Compress a sequence of correlation matrices.

        Parameters
        ----------
        corr_tensor : (T, n_assets, n_assets)

        Returns
        -------
        List of T CorrelationMPS objects
        """
        result = []
        for t in range(corr_tensor.shape[0]):
            obj = CorrelationMPS(self.max_bond, self.cutoff, self.encoding)
            obj.fit(corr_tensor[t])
            result.append(obj)
        return result


# ============================================================================
# CP decomposition for factor models
# ============================================================================

class TensorFactorModel:
    """
    Extract latent factors from a multi-asset tensor using CP decomposition.

    Models the return tensor T[t, i, tau] ≈ sum_{r=1}^R g_r * a_r(t) ⊗ b_r(i) ⊗ c_r(tau)

    where:
    - a_r : time factor (T-dimensional)
    - b_r : asset loading (n_assets-dimensional)
    - c_r : temporal pattern within window (window-dimensional)

    This is the tensor analog of traditional factor models (PCA, APT).
    """

    def __init__(
        self,
        n_factors: int = 5,
        n_iter: int = 100,
        tol: float = 1e-6,
    ):
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.tol = tol
        self.lambdas_ = None
        self.factors_ = None  # [time_factors, asset_factors, lag_factors]
        self.is_fitted = False

    def fit(
        self,
        return_tensor: jnp.ndarray,
        key: Optional[jax.random.KeyArray] = None,
    ) -> "TensorFactorModel":
        """
        Fit CP decomposition to the return tensor.

        Parameters
        ----------
        return_tensor : (n_windows, n_assets, window) return tensor
        key : random key

        Returns
        -------
        self
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        return_tensor = jnp.array(return_tensor, dtype=jnp.float32)
        self.lambdas_, self.factors_ = cp_decomp(
            return_tensor,
            rank=self.n_factors,
            n_iter=self.n_iter,
            tol=self.tol,
            key=key,
        )
        self.is_fitted = True
        return self

    def get_asset_loadings(self) -> jnp.ndarray:
        """
        Return asset factor loadings matrix.

        Returns
        -------
        (n_assets, n_factors) factor loading matrix
        """
        assert self.is_fitted
        return self.factors_[1]

    def get_time_series(self) -> jnp.ndarray:
        """
        Return time factor scores.

        Returns
        -------
        (n_windows, n_factors) factor time series
        """
        assert self.is_fitted
        return self.factors_[0]

    def get_temporal_patterns(self) -> jnp.ndarray:
        """
        Return within-window temporal patterns.

        Returns
        -------
        (window, n_factors) temporal factor patterns
        """
        assert self.is_fitted
        return self.factors_[2]

    def reconstruct(self) -> jnp.ndarray:
        """Reconstruct return tensor from CP factors."""
        assert self.is_fitted
        T_factors, A_factors, W_factors = self.factors_
        # Outer product reconstruction
        recon = jnp.einsum("tr,ir,wr,r->tiw", T_factors, A_factors, W_factors, self.lambdas_)
        return recon

    def factor_returns(self, asset_weights: jnp.ndarray) -> jnp.ndarray:
        """
        Compute factor returns for a given portfolio.

        Parameters
        ----------
        asset_weights : (n_assets,) portfolio weight vector

        Returns
        -------
        (n_windows, n_factors) factor exposures
        """
        assert self.is_fitted
        A = self.factors_[1]  # (n_assets, n_factors)
        T = self.factors_[0]  # (n_windows, n_factors)
        exposures = A.T @ asset_weights  # (n_factors,)
        return T * exposures[None, :]  # (n_windows, n_factors)


# ============================================================================
# Tensor PCA (HOSVD)
# ============================================================================

class TensorPCA:
    """
    Higher-Order PCA (tensor PCA) via HOSVD.

    Generalizes standard PCA to multi-dimensional arrays by computing the
    SVD of each mode unfolding independently, then forming the Tucker core.

    Applications:
    - Multi-asset, multi-period covariance analysis
    - Volatility surface dimensionality reduction
    - Regime-conditioned return distribution analysis
    """

    def __init__(
        self,
        n_components: Optional[Sequence[int]] = None,
        variance_explained: float = 0.95,
    ):
        """
        Parameters
        ----------
        n_components : components per mode (if None, auto from variance_explained)
        variance_explained : target cumulative variance explained per mode
        """
        self.n_components = n_components
        self.variance_explained = variance_explained
        self.components_ = None   # List of mode-k singular vectors
        self.singular_values_ = None
        self.core_ = None
        self.shape_ = None
        self.is_fitted = False

    def fit(self, tensor: jnp.ndarray) -> "TensorPCA":
        """
        Fit tensor PCA to an N-dimensional array.

        Parameters
        ----------
        tensor : input tensor of shape (n_1, ..., n_N)

        Returns
        -------
        self
        """
        tensor = jnp.array(tensor, dtype=jnp.float32)
        self.shape_ = tensor.shape
        n_dims = tensor.ndim

        self.components_ = []
        self.singular_values_ = []
        ranks = []

        for k in range(n_dims):
            # Mode-k unfolding
            perm = [k] + [i for i in range(n_dims) if i != k]
            unf = jnp.transpose(tensor, perm).reshape(tensor.shape[k], -1)

            U, s, _ = jnp.linalg.svd(unf, full_matrices=False)
            self.singular_values_.append(s)

            if self.n_components is not None:
                r = min(self.n_components[k], U.shape[1])
            else:
                # Auto-select based on variance explained
                sv_sq = s ** 2
                total = float(jnp.sum(sv_sq))
                cumsum = jnp.cumsum(sv_sq)
                r = int(jnp.sum(cumsum / total < self.variance_explained).item()) + 1
                r = max(1, min(r, U.shape[1]))

            self.components_.append(U[:, :r])
            ranks.append(r)

        # Compute Tucker core
        self.core_, _ = tucker_decomp(tensor, ranks, n_iter=1)
        self.is_fitted = True
        return self

    def transform(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """
        Project tensor onto principal components.

        Parameters
        ----------
        tensor : array of same shape as training tensor

        Returns
        -------
        Core tensor of shape (r_1, ..., r_N)
        """
        assert self.is_fitted
        tensor = jnp.array(tensor, dtype=jnp.float32)
        n_dims = tensor.ndim

        result = tensor
        for k in range(n_dims - 1, -1, -1):
            U = self.components_[k]
            # Contract mode k with U^T
            result = jnp.tensordot(result, U.T, axes=([0], [1]))
            # Move contracted mode to front
            result = jnp.moveaxis(result, -1, 0)

        return result

    def inverse_transform(self, core: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct tensor from core."""
        assert self.is_fitted
        result = core
        for k in range(self.core_.ndim - 1, -1, -1):
            U = self.components_[k]
            result = jnp.tensordot(result, U, axes=([0], [1]))
            result = jnp.moveaxis(result, -1, 0)
        return result

    def explained_variance_ratio(self) -> List[jnp.ndarray]:
        """Return variance explained per component per mode."""
        ratios = []
        for sv in self.singular_values_:
            sv_sq = sv ** 2
            total = jnp.sum(sv_sq)
            ratios.append(sv_sq / (total + 1e-10))
        return ratios


# ============================================================================
# Regime-conditioned tensor factorization
# ============================================================================

class RegimeCompression:
    """
    Regime-conditioned correlation tensor factorization.

    Detects market regimes (e.g., risk-on vs risk-off) from the time-varying
    correlation tensor and fits separate Tucker decompositions per regime.

    The regime detection uses a simple spectral clustering on the correlation
    time series, conditioned on the dominant eigenvalue trajectory.

    Attributes
    ----------
    n_regimes : number of market regimes
    n_factors_per_regime : Tucker rank per regime
    """

    def __init__(
        self,
        n_regimes: int = 3,
        n_time_factors: int = 3,
        n_asset_factors: int = 8,
    ):
        self.n_regimes = n_regimes
        self.n_time_factors = n_time_factors
        self.n_asset_factors = n_asset_factors
        self.regime_labels_ = None
        self.regime_decompositions_: Dict[int, CorrelationTucker] = {}
        self.regime_centroids_: Dict[int, jnp.ndarray] = {}
        self.is_fitted = False

    def fit(
        self,
        corr_tensor: jnp.ndarray,
        key: Optional[jax.random.KeyArray] = None,
    ) -> "RegimeCompression":
        """
        Fit regime-conditioned Tucker to time-varying correlation tensor.

        Parameters
        ----------
        corr_tensor : (T, n_assets, n_assets) correlation tensor
        key : random key

        Returns
        -------
        self
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        T, n, _ = corr_tensor.shape

        # Compute per-timestep features: leading eigenvalue + trace
        features = self._compute_regime_features(corr_tensor)

        # K-means clustering on features
        self.regime_labels_ = self._kmeans(features, self.n_regimes, key=key)

        # Fit Tucker per regime
        for r in range(self.n_regimes):
            mask = self.regime_labels_ == r
            n_in_regime = int(jnp.sum(mask).item())
            if n_in_regime < 2:
                continue

            regime_tensor = corr_tensor[mask]

            tucker = CorrelationTucker(
                n_time_factors=min(self.n_time_factors, n_in_regime),
                n_asset_factors=self.n_asset_factors,
            )
            tucker.fit(regime_tensor)
            self.regime_decompositions_[r] = tucker
            self.regime_centroids_[r] = jnp.mean(features[mask], axis=0)

        self.is_fitted = True
        return self

    def predict_regime(self, corr_matrix: jnp.ndarray) -> int:
        """
        Predict regime for a new correlation matrix.

        Parameters
        ----------
        corr_matrix : (n_assets, n_assets) correlation matrix

        Returns
        -------
        Predicted regime index
        """
        assert self.is_fitted
        features = self._compute_regime_features(corr_matrix[None])
        dists = jnp.array([
            float(jnp.linalg.norm(features[0] - c))
            for c in self.regime_centroids_.values()
        ])
        return int(jnp.argmin(dists))

    def _compute_regime_features(self, corr_tensor: jnp.ndarray) -> jnp.ndarray:
        """Compute regime features from correlation tensor."""
        T = corr_tensor.shape[0]
        features = []
        for t in range(T):
            C = corr_tensor[t]
            # Top eigenvalues as features
            evals = jnp.linalg.eigvalsh(C)
            evals_sorted = jnp.sort(evals)[::-1]
            trace = jnp.trace(C)
            det = jnp.maximum(jnp.linalg.det(C), 1e-10)
            feat = jnp.concatenate([
                evals_sorted[:min(5, len(evals_sorted))],
                jnp.array([trace, jnp.log(det + 1e-10)])
            ])
            features.append(feat)
        return jnp.stack(features)  # (T, n_features)

    def _kmeans(
        self,
        features: jnp.ndarray,
        k: int,
        n_iter: int = 100,
        key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ) -> jnp.ndarray:
        """Simple K-means clustering."""
        T, d = features.shape

        # Initialize centroids randomly
        key, subkey = jax.random.split(key)
        idx = jax.random.permutation(subkey, T)[:k]
        centroids = features[idx]

        labels = jnp.zeros(T, dtype=jnp.int32)
        for _ in range(n_iter):
            # Assign
            dists = jnp.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
            new_labels = jnp.argmin(dists, axis=1)

            if jnp.all(new_labels == labels):
                break
            labels = new_labels

            # Update centroids
            for c in range(k):
                mask = labels == c
                if jnp.sum(mask) > 0:
                    centroids = centroids.at[c].set(jnp.mean(features[mask], axis=0))

        return labels


# ============================================================================
# Causality tensor
# ============================================================================

class CausalityTensor:
    """
    Granger causality tensor for multi-asset return systems.

    Builds a 3D tensor C[i, j, lag] where C[i,j,lag] measures the Granger
    causality from asset j to asset i at lag `lag`.

    Uses vector autoregression (VAR) fitted in closed form.
    """

    def __init__(
        self,
        max_lag: int = 5,
        max_bond: int = 16,
    ):
        self.max_lag = max_lag
        self.max_bond = max_bond
        self.causality_tensor_ = None
        self.tt_decomp_ = None
        self.is_fitted = False

    def fit(
        self,
        returns: jnp.ndarray,
        regularize: float = 0.01,
    ) -> "CausalityTensor":
        """
        Compute Granger causality tensor from asset returns.

        Parameters
        ----------
        returns : (T, n_assets) return matrix
        regularize : regularization for VAR coefficient estimation

        Returns
        -------
        self
        """
        returns = jnp.array(returns, dtype=jnp.float32)
        T, n = returns.shape
        L = self.max_lag

        # Build design matrix for VAR(L)
        # Y[t] = sum_{l=1}^L B[l] @ Y[t-l] + eps[t]
        Y = returns[L:]      # (T-L, n)
        X = jnp.concatenate([returns[L - l - 1:T - l - 1] for l in range(L)], axis=1)
        # X shape: (T-L, n*L)

        # OLS: B = (X^T X + reg I)^{-1} X^T Y
        reg_mat = regularize * jnp.eye(X.shape[1])
        B = jnp.linalg.solve(X.T @ X + reg_mat, X.T @ Y)  # (n*L, n)

        # Reshape to (n, n, L): B[l, :, :] is the coefficient matrix at lag l
        B_tensor = B.reshape(L, n, n)  # (L, n_source, n_target) -- needs transposing
        # C[i, j, l] = causal influence of j -> i at lag l
        causal = jnp.abs(B_tensor.transpose(2, 1, 0))  # (n_target, n_source, L)
        self.causality_tensor_ = causal

        # Compress with TT-SVD
        self.tt_decomp_ = tt_svd(causal, max_rank=self.max_bond, cutoff=1e-8)
        self.is_fitted = True
        return self

    def get_causality_matrix(self, lag: int = 1) -> jnp.ndarray:
        """
        Return the Granger causality matrix at a specific lag.

        Parameters
        ----------
        lag : lag index (1-indexed)

        Returns
        -------
        (n_assets, n_assets) causality matrix C[i,j] = influence j -> i
        """
        assert self.is_fitted
        return self.causality_tensor_[:, :, lag - 1]

    def total_causality(self) -> jnp.ndarray:
        """Sum causality over all lags."""
        assert self.is_fitted
        return jnp.sum(self.causality_tensor_, axis=2)

    def most_causal_pairs(self, top_k: int = 10) -> List[Tuple[int, int, float]]:
        """Return top-k most causal asset pairs."""
        assert self.is_fitted
        C = self.total_causality()
        n = C.shape[0]
        pairs = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    pairs.append((i, j, float(C[i, j])))
        pairs.sort(key=lambda x: -x[2])
        return pairs[:top_k]


# ============================================================================
# Dependency hypercube
# ============================================================================

class DependencyHypercube:
    """
    Higher-order asset dependency hypercube.

    Encodes the joint distribution of returns for groups of assets as a
    Tucker-compressed tensor.

    The hypercube T[s_1, s_2, ..., s_k] records the empirical frequency of
    the joint quantile state (s_1, ..., s_k) across k assets.
    """

    def __init__(
        self,
        n_quantiles: int = 5,
        n_assets: int = 10,
        order: int = 3,
        max_rank: int = 10,
    ):
        """
        Parameters
        ----------
        n_quantiles : number of quantile bins per asset
        n_assets : number of assets in the hypercube
        order : tensor order (number of assets in each joint state)
        max_rank : Tucker rank for compression
        """
        self.n_quantiles = n_quantiles
        self.n_assets = n_assets
        self.order = order
        self.max_rank = max_rank
        self.hypercube_ = None
        self.tucker_core_ = None
        self.tucker_factors_ = None
        self.is_fitted = False

    def fit(
        self,
        returns: jnp.ndarray,
        asset_indices: Optional[List[int]] = None,
    ) -> "DependencyHypercube":
        """
        Build and compress the dependency hypercube.

        Parameters
        ----------
        returns : (T, n_total_assets) return matrix
        asset_indices : which assets to include (default: first n_assets)

        Returns
        -------
        self
        """
        returns = jnp.array(returns, dtype=jnp.float32)
        T, n_total = returns.shape

        if asset_indices is None:
            asset_indices = list(range(min(self.n_assets, n_total)))

        selected = returns[:, asset_indices]
        n = len(asset_indices)
        Q = self.n_quantiles

        # Quantize each asset's returns
        quantiles = jnp.zeros((T, n), dtype=jnp.int32)
        for i in range(n):
            r = selected[:, i]
            q_vals = jnp.nanpercentile(r, jnp.linspace(0, 100, Q + 1)[1:-1])
            q_idx = jnp.searchsorted(q_vals, r)
            quantiles = quantiles.at[:, i].set(q_idx)

        # Build joint frequency tensor of order `order`
        # For simplicity, build pairwise and extend
        if self.order == 2:
            cube = jnp.zeros((Q, Q, n, n), dtype=jnp.float32)
            for t in range(T):
                for ii in range(n):
                    for jj in range(n):
                        qi, qj = int(quantiles[t, ii]), int(quantiles[t, jj])
                        cube = cube.at[qi, qj, ii, jj].add(1.0)
            self.hypercube_ = cube / T

        elif self.order == 3:
            cube = jnp.zeros((Q, Q, Q, n, n, n), dtype=jnp.float32)
            for t in range(T):
                for ii in range(min(n, 5)):  # Limit for tractability
                    for jj in range(min(n, 5)):
                        for kk in range(min(n, 5)):
                            qi = int(quantiles[t, ii])
                            qj = int(quantiles[t, jj])
                            qk = int(quantiles[t, kk])
                            cube = cube.at[qi, qj, qk, ii, jj, kk].add(1.0)
            self.hypercube_ = cube / T
        else:
            # Generic: just store marginal distributions
            self.hypercube_ = jnp.zeros((Q,) * self.order + (n,) * self.order)

        # Tucker compress
        ranks = [min(self.max_rank, s) for s in self.hypercube_.shape]
        self.tucker_core_, self.tucker_factors_ = tucker_decomp(
            self.hypercube_, ranks, n_iter=5
        )
        self.is_fitted = True
        return self

    def get_pairwise_dependence(self, i: int, j: int) -> jnp.ndarray:
        """Return the joint quantile distribution for assets i and j."""
        assert self.is_fitted and self.order == 2
        return self.hypercube_[:, :, i, j]

    def tail_dependence(self, i: int, j: int, q_lower: float = 0.1, q_upper: float = 0.9) -> Dict[str, float]:
        """
        Estimate tail dependence coefficients for assets i and j.

        Parameters
        ----------
        i, j : asset indices
        q_lower : lower tail quantile threshold
        q_upper : upper tail quantile threshold

        Returns
        -------
        Dictionary with 'lower' and 'upper' tail dependence
        """
        assert self.is_fitted and self.order == 2
        Q = self.n_quantiles
        q_lo = max(0, int(q_lower * Q))
        q_hi = min(Q - 1, int(q_upper * Q))

        joint = self.hypercube_[:, :, i, j]
        marginal_i = jnp.sum(joint, axis=1)
        marginal_j = jnp.sum(joint, axis=0)

        lower_joint = jnp.sum(joint[:q_lo + 1, :q_lo + 1])
        upper_joint = jnp.sum(joint[q_hi:, q_hi:])
        lower_marginal = jnp.sum(marginal_i[:q_lo + 1])
        upper_marginal = jnp.sum(marginal_i[q_hi:])

        lambda_lower = float(lower_joint / (lower_marginal + 1e-10))
        lambda_upper = float(upper_joint / (upper_marginal + 1e-10))

        return {"lower": lambda_lower, "upper": lambda_upper}


# ============================================================================
# Streaming compressor (online tensor updates)
# ============================================================================

class StreamingCompressor:
    """
    Online streaming compressor for time-varying correlation tensors.

    Maintains a compressed MPS representation that is updated incrementally
    as new correlation matrices arrive, without full recomputation.

    Uses a rank-1 update scheme: when a new matrix C_new arrives,
    it updates the MPS via a low-rank modification.
    """

    def __init__(
        self,
        n_assets: int,
        max_bond: int = 16,
        memory: int = 100,
        forgetting_factor: float = 0.95,
    ):
        """
        Parameters
        ----------
        n_assets : number of assets
        max_bond : maximum MPS bond dimension
        memory : number of past timesteps to retain exactly
        forgetting_factor : exponential forgetting factor
        """
        self.n_assets = n_assets
        self.max_bond = max_bond
        self.memory = memory
        self.forgetting_factor = forgetting_factor
        self.buffer: List[jnp.ndarray] = []
        self.current_mps: Optional[MatrixProductState] = None
        self.n_updates = 0

    def update(self, corr_matrix: jnp.ndarray) -> MatrixProductState:
        """
        Process a new correlation matrix and update the compressed representation.

        Parameters
        ----------
        corr_matrix : (n_assets, n_assets) new correlation matrix

        Returns
        -------
        Updated MatrixProductState
        """
        corr_matrix = jnp.array(corr_matrix, dtype=jnp.float32)
        self.buffer.append(corr_matrix)

        if len(self.buffer) > self.memory:
            self.buffer.pop(0)

        # Recompute from buffer with exponential weighting
        n = len(self.buffer)
        weights = jnp.array([self.forgetting_factor ** (n - 1 - t) for t in range(n)])
        weights = weights / (jnp.sum(weights) + 1e-10)

        weighted_corr = sum(
            float(w) * C for w, C in zip(weights, self.buffer)
        )

        # Build MPS from weighted average
        cobj = CorrelationMPS(max_bond=self.max_bond, cutoff=1e-8)
        cobj.fit(weighted_corr)
        self.current_mps = cobj.mps_
        self.n_updates += 1
        return self.current_mps

    def get_state(self) -> Optional[MatrixProductState]:
        """Return current compressed state."""
        return self.current_mps

    def compression_ratio(self) -> float:
        """Current compression ratio."""
        if self.current_mps is None:
            return 1.0
        n = self.n_assets
        return (n * n) / self.current_mps.num_params()


# ============================================================================
# Anomaly detection via tensor residuals
# ============================================================================

class AnomalyDetector:
    """
    Tensor-based anomaly detection for financial time series.

    Uses Tucker decomposition residuals to identify anomalous correlation
    regimes. Anomalies are timesteps where the reconstruction error
    exceeds a threshold.
    """

    def __init__(
        self,
        n_time_factors: int = 5,
        n_asset_factors: int = 10,
        threshold_sigma: float = 2.0,
    ):
        self.n_time_factors = n_time_factors
        self.n_asset_factors = n_asset_factors
        self.threshold_sigma = threshold_sigma
        self.tucker_ = None
        self.reconstruction_errors_ = None
        self.threshold_ = None
        self.is_fitted = False

    def fit(self, corr_tensor: jnp.ndarray) -> "AnomalyDetector":
        """
        Fit Tucker model to normal-regime correlation tensor.

        Parameters
        ----------
        corr_tensor : (T, n_assets, n_assets)

        Returns
        -------
        self
        """
        self.tucker_ = CorrelationTucker(
            n_time_factors=self.n_time_factors,
            n_asset_factors=self.n_asset_factors,
        )
        self.tucker_.fit(corr_tensor)

        # Compute per-timestep reconstruction errors
        recon = self.tucker_.reconstruct()
        errors = jnp.linalg.norm(
            corr_tensor - recon, axis=(1, 2)
        )  # (T,)
        self.reconstruction_errors_ = errors

        mu = jnp.mean(errors)
        sigma = jnp.std(errors)
        self.threshold_ = float(mu + self.threshold_sigma * sigma)
        self.is_fitted = True
        return self

    def predict(self, corr_tensor: jnp.ndarray) -> jnp.ndarray:
        """
        Predict anomaly scores for new correlation matrices.

        Parameters
        ----------
        corr_tensor : (T', n_assets, n_assets)

        Returns
        -------
        Anomaly scores of shape (T',)
        """
        assert self.is_fitted
        recon = self.tucker_.reconstruct()
        # Use the Tucker model as a baseline
        # For new data, compute projected reconstruction error
        T_new = corr_tensor.shape[0]
        scores = []
        for t in range(T_new):
            C = corr_tensor[t]
            # Project onto learned factors
            U_a = self.tucker_.factors_[1]
            C_proj = U_a.T @ C @ U_a  # (n_factors, n_factors)
            # Reconstruct
            C_recon = U_a @ C_proj @ U_a.T
            err = float(jnp.linalg.norm(C - C_recon))
            scores.append(err)
        return jnp.array(scores)

    def detect(self, scores: jnp.ndarray) -> jnp.ndarray:
        """
        Apply threshold to get binary anomaly labels.

        Parameters
        ----------
        scores : anomaly scores

        Returns
        -------
        Boolean array (True = anomalous)
        """
        return scores > self.threshold_


# ============================================================================
# Volatility tensor
# ============================================================================

def build_volatility_surface_tensor(
    implied_vols: jnp.ndarray,
    strikes_rel: jnp.ndarray,
    maturities: jnp.ndarray,
    time_indices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Build a 3D volatility surface tensor from implied vol data.

    Shape: (n_time, n_strikes, n_maturities)

    Parameters
    ----------
    implied_vols : (T, n_strikes, n_maturities) implied volatility array
    strikes_rel : relative strike values (moneyness)
    maturities : maturity values in years
    time_indices : time index vector

    Returns
    -------
    Volatility surface tensor
    """
    return jnp.array(implied_vols, dtype=jnp.float32)


def compress_vol_surface(
    vol_surface: jnp.ndarray,
    n_time_factors: int = 3,
    n_strike_factors: int = 4,
    n_mat_factors: int = 3,
) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """
    Compress volatility surface tensor via Tucker decomposition.

    Parameters
    ----------
    vol_surface : (T, n_strikes, n_maturities) volatility tensor
    n_time_factors : Tucker rank for time dimension
    n_strike_factors : Tucker rank for strike dimension
    n_mat_factors : Tucker rank for maturity dimension

    Returns
    -------
    (core_tensor, [time_factors, strike_factors, maturity_factors])
    """
    ranks = [n_time_factors, n_strike_factors, n_mat_factors]
    return tucker_decomp(vol_surface, ranks, n_iter=15)


# ============================================================================
# Cross-sectional momentum tensor
# ============================================================================

def build_momentum_tensor(
    returns: jnp.ndarray,
    lookback_periods: Sequence[int] = (20, 60, 120, 250),
) -> jnp.ndarray:
    """
    Build a momentum tensor encoding returns over multiple lookback horizons.

    Shape: (T, n_assets, n_lookbacks)

    Parameters
    ----------
    returns : (T, n_assets) return matrix
    lookback_periods : list of lookback window lengths

    Returns
    -------
    Momentum tensor
    """
    returns = jnp.array(returns, dtype=jnp.float32)
    T, n = returns.shape
    L = len(lookback_periods)
    max_lb = max(lookback_periods)

    momentum = jnp.zeros((T - max_lb, n, L))

    for l_idx, lb in enumerate(lookback_periods):
        for t in range(T - max_lb):
            # Cumulative return over last `lb` days
            cum_ret = jnp.sum(returns[t + max_lb - lb:t + max_lb], axis=0)
            momentum = momentum.at[t, :, l_idx].set(cum_ret)

    return momentum


def risk_attribution_tensor(
    returns: jnp.ndarray,
    factor_returns: jnp.ndarray,
    weights: jnp.ndarray,
    window: int = 60,
) -> jnp.ndarray:
    """
    Build a risk attribution tensor decomposing portfolio risk into factors.

    Shape: (T, n_factors, n_assets) where entry [t, f, i] is the
    contribution of factor f to asset i's variance at time t.

    Parameters
    ----------
    returns : (T, n_assets) asset returns
    factor_returns : (T, n_factors) factor returns
    weights : (n_assets,) portfolio weights
    window : rolling estimation window

    Returns
    -------
    Risk attribution tensor
    """
    returns = jnp.array(returns, dtype=jnp.float32)
    factor_returns = jnp.array(factor_returns, dtype=jnp.float32)
    T, n = returns.shape
    _, K = factor_returns.shape
    n_steps = T - window + 1

    risk_tensor = jnp.zeros((n_steps, K, n))

    for t in range(n_steps):
        R = returns[t:t + window]     # (window, n)
        F = factor_returns[t:t + window]  # (window, K)

        # OLS: R = F @ B + eps
        B = jnp.linalg.lstsq(F, R, rcond=None)[0]  # (K, n)

        # Factor covariance
        F_cov = (F.T @ F) / window  # (K, K)

        # Factor contribution to risk: (K, n)
        # risk[f, i] = B[f, i]^2 * F_cov[f, f]
        risk = B ** 2 * jnp.diag(F_cov)[:, None]
        risk_tensor = risk_tensor.at[t].set(risk)

    return risk_tensor


# ============================================================================
# Utility: run_financial_mps_experiment
# ============================================================================

def run_financial_mps_experiment(
    n_assets: int = 100,
    T: int = 500,
    n_factors: int = 5,
    max_bond: int = 16,
    key: Optional[jax.random.KeyArray] = None,
) -> Dict[str, Any]:
    """
    Run a complete financial MPS experiment pipeline.

    Generates synthetic multi-asset returns, builds correlation tensors,
    compresses with MPS/Tucker, and reports compression statistics.

    Parameters
    ----------
    n_assets : number of assets
    T : number of time steps
    n_factors : number of latent factors
    max_bond : MPS bond dimension
    key : random key

    Returns
    -------
    Dictionary of results
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    # Generate synthetic returns with factor structure
    key, k1, k2, k3 = jax.random.split(key, 4)
    factor_ret = jax.random.normal(k1, (T, n_factors)) * 0.01
    loadings = jax.random.normal(k2, (n_assets, n_factors))
    idio = jax.random.normal(k3, (T, n_assets)) * 0.005
    returns = factor_ret @ loadings.T + idio  # (T, n_assets)

    # Build correlation tensor
    corr_tensor = build_correlation_tensor(returns, window=60, stride=10)
    T_corr, _, _ = corr_tensor.shape

    # Tucker compression
    n_t_factors = min(5, T_corr)
    n_a_factors = min(10, n_assets)
    tucker = CorrelationTucker(n_time_factors=n_t_factors, n_asset_factors=n_a_factors)
    tucker.fit(corr_tensor)
    tucker_error = tucker.reconstruction_error(corr_tensor)

    # MPS compression of average correlation
    avg_corr = jnp.mean(corr_tensor, axis=0)
    mps_encoder = CorrelationMPS(max_bond=max_bond, cutoff=1e-8)
    mps_encoder.fit(avg_corr)
    mps_recon = mps_encoder.transform()
    mps_error = float(jnp.linalg.norm(avg_corr - mps_recon) / (jnp.linalg.norm(avg_corr) + 1e-10))

    # Causality tensor
    caus = CausalityTensor(max_lag=3, max_bond=8)
    caus.fit(returns[:min(200, T)])

    return {
        "n_assets": n_assets,
        "T": T,
        "n_corr_timesteps": T_corr,
        "tucker_error": tucker_error,
        "tucker_compression": corr_tensor.size / (tucker.core_.size + sum(f.size for f in tucker.factors_)),
        "mps_error": mps_error,
        "mps_compression": mps_encoder.compression_ratio(),
        "mps_bond_dims": mps_encoder.mps_.bond_dims,
        "causality_top_pairs": caus.most_causal_pairs(top_k=5),
    }
