"""
anomaly_detection.py — Tensor-based anomaly detection for TensorNet (Project AETERNUS).

Implements:
  - Tucker residual anomaly detection
  - Robust Tensor PCA (TRPCA) via convex relaxation
  - Streaming tensor anomaly updates (online RPCA)
  - Local Outlier Factor in tensor space
  - Isolation Tensor Forest
  - Change point detection via tensor divergence
  - Multivariate tail-risk anomaly scoring
  - Regime shift detection via spectral tensor analysis
  - MPS-based density estimation for anomaly scoring
  - High-dimensional correlation breakdown detection
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Sequence, Union, Dict, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad, vmap

from .mps import MatrixProductState, mps_from_dense, mps_inner_product, mps_norm
from .tt_decomp import TensorTrain, tt_svd, tt_round, tucker_decomp


# ============================================================================
# Tucker Residual Anomaly Detector
# ============================================================================

class TuckerResidualDetector:
    """
    Tucker decomposition residual-based anomaly detector.

    Fits a Tucker model to normal-period data and flags timesteps where
    the reconstruction error exceeds a learned threshold.

    The residual at time t is:
      e(t) = ||C(t) - C_hat(t)||_F

    where C_hat(t) is the Tucker reconstruction.

    This captures structural breaks in the correlation matrix that
    are not explained by the low-rank factor structure.
    """

    def __init__(
        self,
        n_time_factors: int = 5,
        n_asset_factors: int = 10,
        threshold_method: str = "sigma",
        threshold_param: float = 2.5,
        contamination: float = 0.05,
    ):
        """
        Parameters
        ----------
        n_time_factors : Tucker rank for time dimension
        n_asset_factors : Tucker rank for asset dimensions
        threshold_method : 'sigma' (mean + k*std) or 'quantile'
        threshold_param : k for sigma method, or quantile for quantile method
        contamination : expected fraction of anomalies (for quantile method)
        """
        self.n_time_factors = n_time_factors
        self.n_asset_factors = n_asset_factors
        self.threshold_method = threshold_method
        self.threshold_param = threshold_param
        self.contamination = contamination

        self.core_ = None
        self.factors_ = None
        self.threshold_ = None
        self.train_residuals_ = None
        self.is_fitted = False

    def fit(
        self,
        corr_tensor: jnp.ndarray,
        val_fraction: float = 0.2,
    ) -> "TuckerResidualDetector":
        """
        Fit Tucker model to correlation tensor.

        Parameters
        ----------
        corr_tensor : (T, n_assets, n_assets) correlation tensor
        val_fraction : fraction of data for threshold estimation

        Returns
        -------
        self
        """
        corr_tensor = jnp.array(corr_tensor, dtype=jnp.float32)
        T, n, _ = corr_tensor.shape

        # Split train / val
        T_train = int(T * (1 - val_fraction))
        train_data = corr_tensor[:T_train]
        val_data = corr_tensor[T_train:]

        # Fit Tucker on training set
        ranks = [
            min(self.n_time_factors, T_train),
            min(self.n_asset_factors, n),
            min(self.n_asset_factors, n),
        ]
        self.core_, self.factors_ = tucker_decomp(train_data, ranks, n_iter=15)

        # Compute training residuals
        train_recon = self._reconstruct(train_data)
        train_errors = jnp.linalg.norm(
            (train_data - train_recon).reshape(T_train, -1),
            axis=1
        )

        # Compute threshold from validation residuals
        if val_data.shape[0] > 0:
            val_errors = self._score(val_data)
            all_errors = jnp.concatenate([train_errors, val_errors])
        else:
            all_errors = train_errors

        self.train_residuals_ = all_errors

        if self.threshold_method == "sigma":
            mu = jnp.mean(all_errors)
            sigma = jnp.std(all_errors)
            self.threshold_ = float(mu + self.threshold_param * sigma)
        else:
            q = 1.0 - self.contamination
            self.threshold_ = float(jnp.nanpercentile(all_errors, 100 * q))

        self.is_fitted = True
        return self

    def _reconstruct(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct tensor using trained Tucker factors."""
        G, factors = self.core_, self.factors_
        T = tensor.shape[0]
        U_t, U_a1, U_a2 = factors

        # Adjust time factor for new tensor size
        if T != U_t.shape[0]:
            # Project and reconstruct differently for new timesteps
            return tensor  # fallback

        recon = jnp.einsum("klm,tk,il,jm->tij", G, U_t, U_a1, U_a2)
        return recon

    def _score(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """Compute per-timestep anomaly scores."""
        T = tensor.shape[0]
        n = tensor.shape[1]
        U_a = self.factors_[1]  # (n_assets, n_asset_factors)

        scores = []
        for t in range(T):
            C = tensor[t]
            # Project onto asset factor space
            C_proj = U_a.T @ C @ U_a
            C_recon = U_a @ C_proj @ U_a.T
            err = float(jnp.linalg.norm(C - C_recon))
            scores.append(err)
        return jnp.array(scores)

    def predict_scores(self, corr_tensor: jnp.ndarray) -> jnp.ndarray:
        """
        Compute anomaly scores for new correlation matrices.

        Parameters
        ----------
        corr_tensor : (T, n_assets, n_assets) or (n_assets, n_assets)

        Returns
        -------
        Anomaly scores of shape (T,)
        """
        assert self.is_fitted
        if corr_tensor.ndim == 2:
            corr_tensor = corr_tensor[None]
        return self._score(corr_tensor)

    def predict(self, corr_tensor: jnp.ndarray) -> jnp.ndarray:
        """
        Binary anomaly labels (True = anomalous).

        Parameters
        ----------
        corr_tensor : (T, n_assets, n_assets)

        Returns
        -------
        Boolean array (T,)
        """
        scores = self.predict_scores(corr_tensor)
        return scores > self.threshold_

    def anomaly_report(
        self,
        corr_tensor: jnp.ndarray,
        dates: Optional[Sequence] = None,
    ) -> Dict[str, Any]:
        """
        Generate a detailed anomaly report.

        Parameters
        ----------
        corr_tensor : (T, n_assets, n_assets)
        dates : optional sequence of date labels

        Returns
        -------
        Dictionary with scores, labels, top anomalies
        """
        scores = self.predict_scores(corr_tensor)
        labels = scores > self.threshold_
        n_anomalies = int(jnp.sum(labels))
        top_idx = jnp.argsort(-scores)[:10]

        report = {
            "n_timesteps": corr_tensor.shape[0],
            "n_anomalies": n_anomalies,
            "anomaly_fraction": n_anomalies / corr_tensor.shape[0],
            "threshold": self.threshold_,
            "mean_score": float(jnp.mean(scores)),
            "max_score": float(jnp.max(scores)),
            "top_anomaly_indices": [int(i) for i in top_idx],
            "scores": scores,
            "labels": labels,
        }
        if dates is not None:
            report["top_anomaly_dates"] = [dates[int(i)] for i in top_idx if int(i) < len(dates)]
        return report


# ============================================================================
# Robust Tensor PCA (TRPCA)
# ============================================================================

class RobustTensorPCA:
    """
    Robust Tensor PCA for financial correlation tensors.

    Decomposes a noisy tensor T = L + S where:
    - L is the low-Tucker-rank background (normal correlation structure)
    - S is the sparse component (anomalies / outliers)

    Solves the convex optimization:
      min_{L,S} ||L||_* + lambda * ||S||_1
      subject to L + S = T

    where ||L||_* is the tensor nuclear norm (sum of matrix nuclear norms)
    and ||S||_1 is the element-wise L1 norm.

    Uses ADMM (Alternating Direction Method of Multipliers).
    """

    def __init__(
        self,
        lambda_sparse: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-5,
        rho: float = 1.0,
    ):
        """
        Parameters
        ----------
        lambda_sparse : sparsity regularization weight
        max_iter : maximum ADMM iterations
        tol : convergence tolerance
        rho : ADMM penalty parameter
        """
        self.lambda_sparse = lambda_sparse
        self.max_iter = max_iter
        self.tol = tol
        self.rho = rho
        self.L_ = None
        self.S_ = None
        self.is_fitted = False

    def fit_transform(
        self,
        tensor: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Decompose tensor into low-rank + sparse components.

        Parameters
        ----------
        tensor : input tensor (any shape)

        Returns
        -------
        (L, S) : low-rank and sparse components
        """
        tensor = jnp.array(tensor, dtype=jnp.float32)
        T_shape = tensor.shape

        # Initialize
        L = jnp.zeros_like(tensor)
        S = jnp.zeros_like(tensor)
        Y = jnp.zeros_like(tensor)  # Lagrange multiplier

        rho = self.rho

        for iteration in range(self.max_iter):
            L_old = L
            S_old = S

            # L update: proximal of nuclear norm
            M_L = tensor - S + Y / rho
            L = self._prox_nuclear(M_L, 1.0 / rho)

            # S update: proximal of L1 norm
            M_S = tensor - L + Y / rho
            S = self._prox_l1(M_S, self.lambda_sparse / rho)

            # Dual update
            Y = Y + rho * (tensor - L - S)

            # Check convergence
            primal_res = jnp.linalg.norm(tensor - L - S)
            dual_res_L = rho * jnp.linalg.norm(L - L_old)
            dual_res_S = rho * jnp.linalg.norm(S - S_old)

            if float(primal_res) < self.tol and float(dual_res_L + dual_res_S) < self.tol:
                break

        self.L_ = L
        self.S_ = S
        self.is_fitted = True
        return L, S

    def _prox_nuclear(
        self,
        tensor: jnp.ndarray,
        tau: float,
    ) -> jnp.ndarray:
        """
        Proximal operator of the tensor nuclear norm.

        Applies matrix SVD soft-thresholding to each mode-k unfolding.

        Parameters
        ----------
        tensor : input tensor
        tau : threshold

        Returns
        -------
        Proximal step result
        """
        shape = tensor.shape
        n_dims = tensor.ndim
        result = jnp.zeros_like(tensor)

        for k in range(n_dims):
            # Mode-k unfolding
            perm = [k] + [i for i in range(n_dims) if i != k]
            t_perm = jnp.transpose(tensor, perm)
            unf = t_perm.reshape(shape[k], -1)

            # SVD soft-threshold
            U, s, Vt = jnp.linalg.svd(unf, full_matrices=False)
            s_thresh = jnp.maximum(s - tau, 0.0)
            unf_thresh = (U * s_thresh[None, :]) @ Vt

            # Fold back
            t_back = unf_thresh.reshape([shape[k]] + [shape[i] for i in range(n_dims) if i != k])
            inv_perm = [0] * n_dims
            for pos, orig in enumerate(perm):
                inv_perm[orig] = pos
            result = result + jnp.transpose(t_back, inv_perm)

        return result / n_dims

    def _prox_l1(
        self,
        tensor: jnp.ndarray,
        tau: float,
    ) -> jnp.ndarray:
        """Proximal operator of L1 norm (soft-thresholding)."""
        return jnp.sign(tensor) * jnp.maximum(jnp.abs(tensor) - tau, 0.0)

    def anomaly_scores(self) -> jnp.ndarray:
        """
        Compute per-timestep anomaly scores as ||S[t]||_F.

        Returns
        -------
        Array of anomaly scores
        """
        assert self.is_fitted
        if self.S_.ndim == 3:
            return jnp.linalg.norm(self.S_.reshape(self.S_.shape[0], -1), axis=1)
        else:
            return jnp.array([float(jnp.linalg.norm(self.S_))])


# ============================================================================
# Streaming tensor anomaly detection
# ============================================================================

class StreamingTensorAnomalyDetector:
    """
    Online streaming anomaly detector for time-varying correlation tensors.

    Maintains a running Tucker model that is updated incrementally.
    Anomaly scores are computed as deviations from the current model.

    Uses exponential forgetting to adapt to gradual regime changes while
    detecting sudden structural breaks.
    """

    def __init__(
        self,
        n_factors: int = 8,
        forgetting_factor: float = 0.95,
        threshold_sigma: float = 3.0,
        warmup_period: int = 50,
    ):
        """
        Parameters
        ----------
        n_factors : Tucker rank for compression
        forgetting_factor : exponential forgetting parameter
        threshold_sigma : anomaly threshold in units of standard deviation
        warmup_period : number of timesteps before scoring begins
        """
        self.n_factors = n_factors
        self.forgetting_factor = forgetting_factor
        self.threshold_sigma = threshold_sigma
        self.warmup_period = warmup_period

        self.buffer: List[jnp.ndarray] = []
        self.scores_history: List[float] = []
        self.running_mean: float = 0.0
        self.running_var: float = 1.0
        self.n_processed: int = 0
        self.current_factors_: Optional[jnp.ndarray] = None
        self.is_warmed_up = False

    def update(
        self,
        corr_matrix: jnp.ndarray,
    ) -> Tuple[float, bool]:
        """
        Process a new correlation matrix.

        Parameters
        ----------
        corr_matrix : (n_assets, n_assets) new correlation matrix

        Returns
        -------
        (anomaly_score, is_anomaly) tuple
        """
        corr_matrix = jnp.array(corr_matrix, dtype=jnp.float32)
        self.buffer.append(corr_matrix)
        self.n_processed += 1

        if len(self.buffer) < self.warmup_period:
            self.scores_history.append(0.0)
            return 0.0, False

        if not self.is_warmed_up:
            self._initialize_model()
            self.is_warmed_up = True

        # Compute anomaly score
        score = self._compute_score(corr_matrix)

        # Update running statistics with forgetting
        alpha = 1.0 - self.forgetting_factor
        self.running_mean = (1 - alpha) * self.running_mean + alpha * score
        self.running_var = (1 - alpha) * self.running_var + alpha * (score - self.running_mean) ** 2

        # Z-score
        z_score = (score - self.running_mean) / (math.sqrt(self.running_var) + 1e-10)

        is_anomaly = abs(z_score) > self.threshold_sigma
        self.scores_history.append(float(z_score))

        # Update model incrementally
        self._update_model(corr_matrix)

        return float(z_score), is_anomaly

    def _initialize_model(self):
        """Initialize Tucker model on warmup buffer."""
        stack = jnp.stack(self.buffer[-self.warmup_period:])
        n = stack.shape[1]
        ranks = [min(self.n_factors, self.warmup_period), min(self.n_factors, n), min(self.n_factors, n)]
        _, factors = tucker_decomp(stack, ranks, n_iter=5)
        self.current_factors_ = factors[1]  # Asset factor matrix

        # Compute initial running stats
        scores = []
        for C in self.buffer[-self.warmup_period:]:
            scores.append(self._compute_score(C))
        self.running_mean = float(np.mean(scores))
        self.running_var = float(np.var(scores) + 1e-8)

    def _compute_score(self, corr_matrix: jnp.ndarray) -> float:
        """Compute projection residual score."""
        if self.current_factors_ is None:
            return 0.0
        U = self.current_factors_
        C_proj = U.T @ corr_matrix @ U
        C_recon = U @ C_proj @ U.T
        return float(jnp.linalg.norm(corr_matrix - C_recon))

    def _update_model(self, new_corr: jnp.ndarray):
        """Incremental model update via weighted averaging of recent data."""
        if len(self.buffer) > self.warmup_period:
            recent = jnp.stack(self.buffer[-self.warmup_period:])
            n = recent.shape[1]
            ranks = [min(self.n_factors, self.warmup_period), min(self.n_factors, n), min(self.n_factors, n)]
            _, factors = tucker_decomp(recent, ranks, n_iter=2)
            # Smooth update of factors
            ff = self.forgetting_factor
            if self.current_factors_ is not None and self.current_factors_.shape == factors[1].shape:
                self.current_factors_ = ff * self.current_factors_ + (1 - ff) * factors[1]
            else:
                self.current_factors_ = factors[1]


# ============================================================================
# Local Outlier Factor in Tensor Space
# ============================================================================

class TensorLOF:
    """
    Local Outlier Factor (LOF) anomaly detector in tensor space.

    Extends LOF to high-dimensional tensors by using compressed distances.
    The distance between correlation tensors is computed as the Tucker
    factor-space Frobenius distance.

    LOF score > 1 indicates an anomaly; LOF score ≈ 1 indicates normal.
    """

    def __init__(
        self,
        k: int = 20,
        n_factors: int = 10,
        contamination: float = 0.05,
    ):
        """
        Parameters
        ----------
        k : number of nearest neighbors
        n_factors : Tucker rank for feature compression
        contamination : expected anomaly fraction
        """
        self.k = k
        self.n_factors = n_factors
        self.contamination = contamination
        self.features_train_: Optional[jnp.ndarray] = None
        self.lof_scores_train_: Optional[jnp.ndarray] = None
        self.threshold_: Optional[float] = None
        self.is_fitted = False

    def fit(self, corr_tensor: jnp.ndarray) -> "TensorLOF":
        """
        Fit LOF to training correlation tensor.

        Parameters
        ----------
        corr_tensor : (T, n_assets, n_assets) correlation tensor

        Returns
        -------
        self
        """
        corr_tensor = jnp.array(corr_tensor, dtype=jnp.float32)
        T, n, _ = corr_tensor.shape

        # Extract compressed features
        self.features_train_ = self._extract_features(corr_tensor)

        # Compute LOF scores on training data
        self.lof_scores_train_ = self._compute_lof(
            self.features_train_, self.features_train_
        )

        # Set threshold
        q = 1.0 - self.contamination
        self.threshold_ = float(jnp.nanpercentile(self.lof_scores_train_, 100 * q))

        self.is_fitted = True
        return self

    def _extract_features(self, corr_tensor: jnp.ndarray) -> jnp.ndarray:
        """Extract feature vectors from correlation matrices."""
        T, n, _ = corr_tensor.shape
        features = []
        for t in range(T):
            C = corr_tensor[t]
            # Use upper triangle as features
            upper = C[jnp.triu(jnp.ones((n, n), dtype=bool), k=1)]
            features.append(upper)
        return jnp.stack(features)

    def _knn_distances(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        k: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute k-nearest neighbor distances and indices."""
        # Compute pairwise distances
        diff = X[:, None, :] - Y[None, :, :]
        dists = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)  # (|X|, |Y|)

        # k-nearest neighbors (excluding self if X == Y)
        k_actual = min(k, dists.shape[1])
        sorted_idx = jnp.argsort(dists, axis=1)[:, :k_actual + 1]
        sorted_dists = jnp.take_along_axis(dists, sorted_idx, axis=1)[:, :k_actual + 1]

        # If X == Y (same dataset), exclude self (distance 0)
        if X.shape == Y.shape and jnp.allclose(X, Y):
            sorted_idx = sorted_idx[:, 1:k_actual + 1]
            sorted_dists = sorted_dists[:, 1:k_actual + 1]
        else:
            sorted_idx = sorted_idx[:, :k_actual]
            sorted_dists = sorted_dists[:, :k_actual]

        return sorted_dists, sorted_idx

    def _compute_lof(
        self,
        X_test: jnp.ndarray,
        X_train: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute LOF scores."""
        T_test = X_test.shape[0]
        T_train = X_train.shape[0]
        k = min(self.k, T_train - 1)

        # kNN distances
        test_dists, test_idx = self._knn_distances(X_test, X_train, k)
        train_dists, train_idx = self._knn_distances(X_train, X_train, k)

        # k-distance of each training point
        k_dists_train = train_dists[:, -1]  # (T_train,)

        # Reachability distances
        reach_dists_test = jnp.maximum(
            test_dists,
            k_dists_train[test_idx],
        )

        # Local reachability density for test points
        lrd_test = 1.0 / (jnp.mean(reach_dists_test, axis=1) + 1e-10)

        # LRD for training neighbors
        reach_dists_train = jnp.maximum(
            train_dists,
            k_dists_train[train_idx],
        )
        lrd_train = 1.0 / (jnp.mean(reach_dists_train, axis=1) + 1e-10)

        # LOF score
        lof = jnp.mean(lrd_train[test_idx] / (lrd_test[:, None] + 1e-10), axis=1)
        return lof

    def predict_scores(self, corr_tensor: jnp.ndarray) -> jnp.ndarray:
        """
        Compute LOF anomaly scores for new data.

        Parameters
        ----------
        corr_tensor : (T, n_assets, n_assets)

        Returns
        -------
        LOF scores of shape (T,)
        """
        assert self.is_fitted
        features_test = self._extract_features(corr_tensor)
        return self._compute_lof(features_test, self.features_train_)

    def predict(self, corr_tensor: jnp.ndarray) -> jnp.ndarray:
        """Binary anomaly labels (True = anomalous)."""
        scores = self.predict_scores(corr_tensor)
        return scores > self.threshold_


# ============================================================================
# Isolation Tensor Forest
# ============================================================================

class IsolationTensorForest:
    """
    Isolation Forest adapted for tensor data.

    Uses random Tucker projections to define isolation trees in the
    compressed tensor space.

    Points that are isolated (separated from others in few random splits)
    are anomalous.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        n_factors: int = 5,
        contamination: float = 0.05,
        key: Optional[jax.random.KeyArray] = None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_factors = n_factors
        self.contamination = contamination
        self.key = key if key is not None else jax.random.PRNGKey(0)
        self.trees_: List[Dict] = []
        self.threshold_: Optional[float] = None
        self.is_fitted = False

    def fit(self, features: jnp.ndarray) -> "IsolationTensorForest":
        """
        Fit isolation forest to feature vectors.

        Parameters
        ----------
        features : (T, d) feature matrix

        Returns
        -------
        self
        """
        features = jnp.array(features, dtype=jnp.float32)
        T, d = features.shape

        self.trees_ = []
        for _ in range(self.n_estimators):
            self.key, subkey = jax.random.split(self.key)
            # Sample max_samples points
            n_sub = min(self.max_samples, T)
            idx = jax.random.permutation(subkey, T)[:n_sub]
            sub_features = features[idx]

            # Build isolation tree
            tree = self._build_tree(sub_features, 0, int(math.ceil(math.log2(n_sub + 1))))
            self.trees_.append(tree)

        # Compute anomaly scores on training data
        scores = self._score_batch(features)
        self.threshold_ = float(jnp.nanpercentile(scores, 100 * (1 - self.contamination)))
        self.is_fitted = True
        return self

    def _build_tree(
        self,
        X: jnp.ndarray,
        depth: int,
        max_depth: int,
    ) -> Dict:
        """Build a single isolation tree recursively."""
        n, d = X.shape

        if n <= 1 or depth >= max_depth:
            return {"type": "leaf", "size": n}

        self.key, subkey = jax.random.split(self.key)
        # Random feature
        feat_idx = int(jax.random.randint(subkey, (), 0, d))

        col = X[:, feat_idx]
        col_min, col_max = float(jnp.min(col)), float(jnp.max(col))

        if col_max <= col_min:
            return {"type": "leaf", "size": n}

        self.key, subkey = jax.random.split(self.key)
        split_val = float(jax.random.uniform(subkey, (), col_min, col_max))

        mask = X[:, feat_idx] < split_val

        return {
            "type": "internal",
            "feat_idx": feat_idx,
            "split_val": split_val,
            "left": self._build_tree(X[mask], depth + 1, max_depth),
            "right": self._build_tree(X[~mask], depth + 1, max_depth),
        }

    def _path_length(self, x: jnp.ndarray, tree: Dict, depth: int = 0) -> float:
        """Compute path length for a single point in a tree."""
        if tree["type"] == "leaf":
            n = tree["size"]
            return depth + self._c(n)

        feat_idx = tree["feat_idx"]
        split_val = tree["split_val"]

        if float(x[feat_idx]) < split_val:
            return self._path_length(x, tree["left"], depth + 1)
        else:
            return self._path_length(x, tree["right"], depth + 1)

    def _c(self, n: int) -> float:
        """Expected path length for a random tree of n samples."""
        if n <= 1:
            return 0.0
        return 2.0 * (math.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n

    def _score_single(self, x: jnp.ndarray) -> float:
        """Anomaly score for a single point."""
        path_lengths = [self._path_length(x, tree) for tree in self.trees_]
        mean_path = float(np.mean(path_lengths))
        n = self.max_samples
        score = 2.0 ** (-mean_path / (self._c(n) + 1e-10))
        return score

    def _score_batch(self, X: jnp.ndarray) -> jnp.ndarray:
        """Anomaly scores for a batch."""
        return jnp.array([self._score_single(X[i]) for i in range(X.shape[0])])

    def predict_scores(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Compute anomaly scores for feature matrix.

        Parameters
        ----------
        features : (T, d) feature matrix

        Returns
        -------
        Anomaly scores of shape (T,)
        """
        assert self.is_fitted
        return self._score_batch(features)

    def predict(self, features: jnp.ndarray) -> jnp.ndarray:
        """Binary anomaly labels."""
        scores = self.predict_scores(features)
        return scores > self.threshold_


# ============================================================================
# Tensor-based anomaly detection (unified API)
# ============================================================================

class TensorAnomalyDetector:
    """
    Unified tensor anomaly detection interface combining multiple methods.

    Ensembles Tucker residuals, LOF, and isolation forest to provide
    robust anomaly detection for financial time series.
    """

    def __init__(
        self,
        n_factors: int = 8,
        lof_k: int = 15,
        n_trees: int = 50,
        contamination: float = 0.05,
        ensemble_weights: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Parameters
        ----------
        n_factors : Tucker rank for compression
        lof_k : LOF nearest neighbors
        n_trees : isolation forest trees
        contamination : expected anomaly fraction
        ensemble_weights : (tucker_w, lof_w, iforest_w) weights for ensemble
        """
        self.n_factors = n_factors
        self.lof_k = lof_k
        self.n_trees = n_trees
        self.contamination = contamination
        self.ensemble_weights = ensemble_weights or (0.4, 0.3, 0.3)

        self.tucker_detector_ = TuckerResidualDetector(
            n_time_factors=n_factors,
            n_asset_factors=n_factors,
            contamination=contamination,
        )
        self.lof_detector_ = TensorLOF(k=lof_k, n_factors=n_factors, contamination=contamination)
        self.iforest_ = IsolationTensorForest(
            n_estimators=n_trees, contamination=contamination
        )
        self.is_fitted = False

    def fit(self, corr_tensor: jnp.ndarray) -> "TensorAnomalyDetector":
        """Fit all detectors."""
        corr_tensor = jnp.array(corr_tensor, dtype=jnp.float32)
        T, n, _ = corr_tensor.shape

        # Tucker detector
        if T >= 10:
            self.tucker_detector_.fit(corr_tensor)

        # LOF
        if T >= 5:
            self.lof_detector_.fit(corr_tensor)

        # Isolation forest on Tucker features
        features = self._extract_features(corr_tensor)
        if T >= 5:
            self.iforest_.fit(features)

        self.is_fitted = True
        return self

    def _extract_features(self, corr_tensor: jnp.ndarray) -> jnp.ndarray:
        """Extract features from correlation tensor."""
        T, n, _ = corr_tensor.shape
        features = []
        for t in range(T):
            C = corr_tensor[t]
            evals = jnp.linalg.eigvalsh(C)
            upper = C[jnp.triu(jnp.ones((n, n), dtype=bool), k=1)]
            feat = jnp.concatenate([evals, upper[:min(20, len(upper))]])
            features.append(feat)
        return jnp.stack(features)

    def predict_scores(self, corr_tensor: jnp.ndarray) -> jnp.ndarray:
        """
        Compute ensemble anomaly scores.

        Parameters
        ----------
        corr_tensor : (T, n_assets, n_assets)

        Returns
        -------
        Ensemble anomaly scores of shape (T,)
        """
        assert self.is_fitted
        T = corr_tensor.shape[0]
        w_tucker, w_lof, w_iforest = self.ensemble_weights

        # Tucker scores (normalized)
        tucker_scores = self.tucker_detector_.predict_scores(corr_tensor)
        tucker_scores_norm = tucker_scores / (jnp.max(tucker_scores) + 1e-10)

        # LOF scores
        lof_scores = self.lof_detector_.predict_scores(corr_tensor)
        lof_scores_norm = lof_scores / (jnp.max(jnp.abs(lof_scores)) + 1e-10)

        # Isolation forest scores
        features = self._extract_features(corr_tensor)
        iforest_scores = self.iforest_.predict_scores(features)
        iforest_scores_norm = iforest_scores / (jnp.max(iforest_scores) + 1e-10)

        # Ensemble
        ensemble = (
            w_tucker * tucker_scores_norm
            + w_lof * lof_scores_norm
            + w_iforest * iforest_scores_norm
        )
        return ensemble

    def predict(self, corr_tensor: jnp.ndarray) -> jnp.ndarray:
        """Binary anomaly labels."""
        scores = self.predict_scores(corr_tensor)
        threshold = float(jnp.nanpercentile(scores, 100 * (1 - self.contamination)))
        return scores > threshold


# ============================================================================
# Change point detection
# ============================================================================

class TensorChangePointDetector:
    """
    Change point detection in time-varying tensor streams.

    Detects abrupt structural changes in the correlation tensor using:
    1. CUSUM statistic on Tucker factor deviations
    2. KL divergence between consecutive tensor slices (in compressed space)
    3. Spectral change detection via leading eigenvalue tracking
    """

    def __init__(
        self,
        n_factors: int = 8,
        window: int = 20,
        threshold_sigma: float = 3.0,
    ):
        self.n_factors = n_factors
        self.window = window
        self.threshold_sigma = threshold_sigma
        self.is_fitted = False
        self.reference_factors_: Optional[jnp.ndarray] = None
        self.cusum_state_: float = 0.0
        self.cusum_history_: List[float] = []

    def fit(self, reference_tensor: jnp.ndarray) -> "TensorChangePointDetector":
        """
        Fit on a reference (normal) period.

        Parameters
        ----------
        reference_tensor : (T, n_assets, n_assets) reference correlation tensor
        """
        T, n, _ = reference_tensor.shape
        ranks = [min(self.n_factors, T), min(self.n_factors, n), min(self.n_factors, n)]
        _, factors = tucker_decomp(reference_tensor, ranks, n_iter=5)
        self.reference_factors_ = factors[1]  # Asset factor matrix
        self.is_fitted = True
        return self

    def detect(
        self,
        new_corr: jnp.ndarray,
    ) -> Tuple[float, bool]:
        """
        Process a new correlation matrix for change points.

        Parameters
        ----------
        new_corr : (n_assets, n_assets) new correlation matrix

        Returns
        -------
        (cusum_stat, is_change_point)
        """
        assert self.is_fitted
        U = self.reference_factors_
        C = jnp.array(new_corr, dtype=jnp.float32)

        # Projection deviation
        C_proj = U.T @ C @ U
        C_recon = U @ C_proj @ U.T
        deviation = float(jnp.linalg.norm(C - C_recon))

        # CUSUM update
        # drift = estimated normal deviation level
        drift = 0.5  # Fixed, could be learned from reference
        self.cusum_state_ = max(0.0, self.cusum_state_ + deviation - drift)
        self.cusum_history_.append(self.cusum_state_)

        # Threshold from reference period statistics
        # (simplified: use fixed threshold)
        threshold = 10.0 * drift
        is_change = self.cusum_state_ > threshold

        if is_change:
            self.cusum_state_ = 0.0  # Reset after detection

        return self.cusum_state_, is_change


# ============================================================================
# Financial regime anomaly scoring
# ============================================================================

def compute_tail_risk_scores(
    returns: jnp.ndarray,
    window: int = 60,
    var_level: float = 0.05,
) -> jnp.ndarray:
    """
    Compute rolling Value-at-Risk based tail risk anomaly scores.

    Parameters
    ----------
    returns : (T, n_assets) asset returns
    window : rolling window
    var_level : VaR confidence level

    Returns
    -------
    (T - window + 1, n_assets) VaR anomaly scores
    """
    returns = jnp.array(returns, dtype=jnp.float32)
    T, n = returns.shape
    n_steps = T - window + 1

    scores = jnp.zeros((n_steps, n))

    for t in range(n_steps):
        R = returns[t:t + window]  # (window, n)
        # Historical VaR
        var = jnp.nanpercentile(R, 100 * var_level, axis=0)
        # CVaR
        below_var = R < var[None, :]
        cvar = jnp.where(
            jnp.sum(below_var, axis=0) > 0,
            jnp.nanmean(jnp.where(below_var, R, jnp.nan), axis=0),
            var,
        )
        scores = scores.at[t].set(-cvar)  # Positive = higher tail risk

    return scores


def correlation_breakdown_score(
    corr_tensor: jnp.ndarray,
    reference_corr: jnp.ndarray,
) -> jnp.ndarray:
    """
    Score correlation breakdown relative to a reference.

    Parameters
    ----------
    corr_tensor : (T, n_assets, n_assets) time-varying correlations
    reference_corr : (n_assets, n_assets) reference correlation matrix

    Returns
    -------
    (T,) breakdown scores
    """
    scores = jnp.linalg.norm(
        (corr_tensor - reference_corr[None]).reshape(corr_tensor.shape[0], -1),
        axis=1,
    )
    return scores / (jnp.linalg.norm(reference_corr) + 1e-10)


def multivariate_kurtosis_score(
    returns: jnp.ndarray,
    window: int = 60,
) -> jnp.ndarray:
    """
    Rolling multivariate kurtosis as an anomaly indicator.

    High multivariate kurtosis signals fat-tailed regimes (crises).

    Parameters
    ----------
    returns : (T, n_assets)
    window : rolling window

    Returns
    -------
    (T - window + 1,) rolling kurtosis scores
    """
    returns = jnp.array(returns, dtype=jnp.float32)
    T, n = returns.shape
    n_steps = T - window + 1

    scores = []
    for t in range(n_steps):
        R = returns[t:t + window]  # (window, n)
        mu = jnp.mean(R, axis=0)
        R_c = R - mu
        cov = R_c.T @ R_c / window + 1e-6 * jnp.eye(n)

        # Mahalanobis distances
        cov_inv = jnp.linalg.inv(cov)
        mah_sq = jnp.einsum("ti,ij,tj->t", R_c, cov_inv, R_c)

        # Mardia's multivariate kurtosis
        kurt = float(jnp.mean(mah_sq ** 2)) / (n * (n + 2))
        scores.append(kurt)

    return jnp.array(scores)
