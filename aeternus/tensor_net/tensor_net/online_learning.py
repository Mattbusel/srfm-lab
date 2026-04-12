"""
online_learning.py — Online/streaming tensor updates for TensorNet (Project AETERNUS).

Provides:
  - Rank-1 ALS updates for online MPS/TT learning
  - Exponential forgetting (sliding window adaptation)
  - Concept drift detection in tensor space
  - Online TT-SVD approximation (streaming SVD)
  - Memory-bounded streaming algorithm with eviction
  - Incremental Tucker update
  - Online covariance tensor update
  - Drift-aware rank adjustment
  - Running statistics tracker for tensors
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp


# ============================================================================
# Running statistics
# ============================================================================

class RunningStats:
    """Welford's online algorithm for running mean and variance.

    Args:
        n_features: Number of features per observation.
        forget_factor: Exponential forgetting factor (1.0 = no forgetting).
    """

    def __init__(self, n_features: int, forget_factor: float = 1.0):
        self.n_features = n_features
        self.forget_factor = forget_factor
        self._n: float = 0.0
        self._mean: np.ndarray = np.zeros(n_features)
        self._M2: np.ndarray = np.zeros(n_features)

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a new observation.

        Args:
            x: Observation vector of shape (n_features,).
        """
        x = np.asarray(x, dtype=float).reshape(-1)[:self.n_features]
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))

        # Apply forgetting
        if self.forget_factor < 1.0:
            self._n *= self.forget_factor
            self._M2 *= self.forget_factor

        self._n += 1.0
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def variance(self) -> np.ndarray:
        if self._n < 2:
            return np.zeros(self.n_features)
        return self._M2 / (self._n - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance + 1e-15)

    @property
    def n(self) -> float:
        return self._n

    def reset(self) -> None:
        """Reset statistics."""
        self._n = 0.0
        self._mean = np.zeros(self.n_features)
        self._M2 = np.zeros(self.n_features)


# ============================================================================
# Online SVD (streaming PCA / TT-SVD)
# ============================================================================

class StreamingSVD:
    """Streaming / online SVD approximation.

    Maintains a rank-r approximation of a data matrix using
    incremental updates. Based on the Brand (2002) incremental SVD.

    Args:
        rank: Number of singular vectors to maintain.
        forget_factor: Exponential downweighting of old data.
        n_features: Feature dimension.
    """

    def __init__(
        self,
        rank: int,
        n_features: int,
        forget_factor: float = 0.99,
    ):
        self.rank = rank
        self.n_features = n_features
        self.forget_factor = forget_factor

        self.U: Optional[np.ndarray] = None   # (n_features, rank)
        self.s: Optional[np.ndarray] = None   # (rank,)
        self.Vt: Optional[np.ndarray] = None  # (rank, n_seen)
        self._n_seen: int = 0

    def update(self, x: np.ndarray) -> None:
        """Process a new data vector.

        Args:
            x: New observation (n_features,) or batch (batch, n_features).
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        batch, d = x.shape
        assert d == self.n_features

        if self.U is None:
            # First update: initialize via SVD of batch
            U, s, Vt = np.linalg.svd(x.T, full_matrices=False)
            r = min(self.rank, len(s))
            self.U = U[:, :r]
            self.s = s[:r]
            self.Vt = Vt[:r, :]
            self._n_seen = batch
            return

        # Apply forgetting to existing singular values
        self.s = self.s * self.forget_factor

        # Project new data onto existing basis
        m = x.T  # (n_features, batch)
        proj = self.U.T @ m          # (rank, batch)
        residual = m - self.U @ proj  # (n_features, batch): component orthogonal to U

        # QR decomposition of residual
        if np.any(np.abs(residual) > 1e-12):
            Q, R = np.linalg.qr(residual)
            n_new = Q.shape[1]
        else:
            Q = np.zeros((self.n_features, 0))
            R = np.zeros((0, batch))
            n_new = 0

        # Build augmented system
        if n_new > 0:
            U_aug = np.concatenate([self.U, Q], axis=1)  # (n_features, rank + n_new)
            K_top = np.concatenate([np.diag(self.s), proj], axis=1)
            K_bot = np.concatenate([R, np.zeros((n_new, self._n_seen + batch - batch))], axis=1) if self._n_seen > 0 else np.concatenate([R[:, :0], R], axis=1)
            K = np.concatenate([K_top, np.zeros((n_new, K_top.shape[1]))], axis=0)
            # Simplified: just truncate with SVD
            M = np.concatenate([np.diag(self.s), proj], axis=1)
        else:
            U_aug = self.U
            M = np.concatenate([np.diag(self.s), proj], axis=1)

        # SVD of augmented system
        U_m, s_m, Vt_m = np.linalg.svd(M, full_matrices=False)
        r = min(self.rank, len(s_m))

        self.U = (U_aug @ U_m[:U_aug.shape[1], :r]) if n_new > 0 else (self.U @ U_m[:, :r])
        self.s = s_m[:r]
        self._n_seen += batch

    @property
    def components(self) -> np.ndarray:
        """Current top-r left singular vectors, shape (n_features, rank)."""
        return self.U if self.U is not None else np.zeros((self.n_features, self.rank))

    @property
    def singular_values(self) -> np.ndarray:
        """Current singular values, shape (rank,)."""
        return self.s if self.s is not None else np.zeros(self.rank)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct x via the current low-rank approximation.

        Args:
            x: Data (n_features,) or (batch, n_features).

        Returns:
            Reconstructed data.
        """
        if self.U is None:
            return x
        x_arr = np.asarray(x)
        single = x_arr.ndim == 1
        if single:
            x_arr = x_arr.reshape(1, -1)
        proj = x_arr @ self.U  # (batch, rank)
        recon = proj @ self.U.T  # (batch, n_features)
        return recon[0] if single else recon


# ============================================================================
# Rank-1 ALS update
# ============================================================================

def rank1_als_update(
    cores: List[np.ndarray],
    new_data: np.ndarray,
    learning_rate: float = 0.01,
    regularization: float = 1e-4,
) -> List[np.ndarray]:
    """Rank-1 ALS (Alternating Least Squares) update for TT cores.

    Updates each core one at a time to minimize reconstruction error
    on the new data point, holding all others fixed.

    Args:
        cores: Current TT cores, each (r_l, d, r_r).
        new_data: New data vector (flattened).
        learning_rate: Step size for gradient descent.
        regularization: L2 regularization coefficient.

    Returns:
        Updated TT cores.
    """
    updated = [np.array(c) for c in cores]
    n_sites = len(updated)
    phys_dims = [c.shape[1] for c in updated]
    total_d = int(np.prod(phys_dims))

    x = np.asarray(new_data, dtype=np.float64).reshape(-1)[:total_d]
    if len(x) < total_d:
        x = np.pad(x, (0, total_d - len(x)))

    # Left environments: left_envs[i] = contraction of cores 0..i-1 with x[0..i-1]
    # Shape of left_envs[i]: (r_i,)
    left_envs = [np.ones(1)]
    x_reshaped = x.reshape(phys_dims)

    for i in range(n_sites - 1):
        core = updated[i]
        r_l, d, r_r = core.shape
        env = left_envs[-1]
        xi = x_reshaped[i].reshape(-1)[:d] if n_sites > 1 else x_reshaped.reshape(-1)[:d]
        r_l_actual = min(len(env), r_l)
        new_env = np.einsum("a,b,abr->r", env[:r_l_actual], xi[:d], core[:r_l_actual, :d, :])
        left_envs.append(new_env)

    # Right environments: right_envs[i] = contraction of cores i+1..N with x[i+1..]
    right_envs = [np.ones(1)]
    for i in range(n_sites - 1, 0, -1):
        core = updated[i]
        r_l, d, r_r = core.shape
        env = right_envs[0]
        xi = x_reshaped[i].reshape(-1)[:d] if n_sites > 1 else x_reshaped.reshape(-1)[:d]
        r_r_actual = min(len(env), r_r)
        new_env = np.einsum("a,b,rba->r", env[:r_r_actual], xi[:d], core[:, :d, :r_r_actual])
        right_envs.insert(0, new_env)

    # Update each core with gradient descent
    for i, core in enumerate(updated):
        r_l, d, r_r = core.shape
        L = left_envs[i]
        R = right_envs[i]
        xi = x_reshaped[i].reshape(-1)[:d] if n_sites > 1 else x_reshaped.reshape(-1)[:d]

        r_l_actual = min(len(L), r_l)
        r_r_actual = min(len(R), r_r)

        # Gradient: outer product of environments and physical index
        grad_c = np.einsum("a,b,r->abr", L[:r_l_actual], xi[:d], R[:r_r_actual])

        # Apply gradient descent
        core[:r_l_actual, :d, :r_r_actual] -= learning_rate * (
            -grad_c + regularization * core[:r_l_actual, :d, :r_r_actual]
        )

    return updated


# ============================================================================
# Exponential forgetting
# ============================================================================

class ExponentialForgettingTT:
    """TT model with exponential forgetting of old data.

    Maintains a TT representation with a sliding effective window.
    New data is weighted more strongly than old data.

    Args:
        cores: Initial TT cores.
        forget_factor: Per-step weight decay (e.g., 0.99 for ~100-step window).
        learning_rate: ALS update learning rate.
    """

    def __init__(
        self,
        cores: List[np.ndarray],
        forget_factor: float = 0.99,
        learning_rate: float = 0.01,
    ):
        self.cores = [np.array(c, dtype=np.float64) for c in cores]
        self.forget_factor = forget_factor
        self.learning_rate = learning_rate
        self._step = 0
        self._eff_n: float = 0.0

    def update(self, x: np.ndarray) -> None:
        """Update TT with a new observation.

        Args:
            x: New data vector.
        """
        self._step += 1
        self._eff_n = self._eff_n * self.forget_factor + 1.0

        # Effective learning rate accounts for forgetting window
        eff_lr = self.learning_rate / (self._eff_n + 1e-8)

        self.cores = rank1_als_update(
            self.cores, x, eff_lr, regularization=1e-5
        )

        # Scale cores by forget factor to downweight old contributions
        for i in range(len(self.cores)):
            self.cores[i] *= self.forget_factor

    @property
    def effective_window(self) -> float:
        """Effective number of samples in memory."""
        return self._eff_n


# ============================================================================
# Concept drift detection
# ============================================================================

@dataclass
class DriftDetectionState:
    """State for concept drift detection."""
    n_seen: int = 0
    error_history: List[float] = field(default_factory=list)
    drift_events: List[int] = field(default_factory=list)
    warning_events: List[int] = field(default_factory=list)
    reference_error_mean: float = 0.0
    reference_error_std: float = 1.0


class TensorDriftDetector:
    """Detect concept drift in tensor approximation quality.

    Monitors reconstruction error of an online TT model.
    Triggers warnings and drift alarms when error increases significantly.

    Implements a simplified CUSUM/DDM-like detector.

    Args:
        warning_threshold: Z-score threshold for warning (default 2.0).
        drift_threshold: Z-score threshold for drift alarm (default 3.0).
        min_samples: Minimum samples before drift detection activates.
        forget_factor: Forgetting factor for reference statistics.
    """

    def __init__(
        self,
        warning_threshold: float = 2.0,
        drift_threshold: float = 3.0,
        min_samples: int = 30,
        forget_factor: float = 0.99,
    ):
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self.forget_factor = forget_factor
        self._state = DriftDetectionState()
        self._stats = RunningStats(n_features=1, forget_factor=forget_factor)

    def update(self, reconstruction_error: float) -> Dict[str, Any]:
        """Process a new reconstruction error observation.

        Args:
            reconstruction_error: Current reconstruction error.

        Returns:
            Dict with 'drift' (bool), 'warning' (bool), 'z_score' (float).
        """
        self._state.n_seen += 1
        self._state.error_history.append(reconstruction_error)
        self._stats.update(np.array([reconstruction_error]))

        result = {"drift": False, "warning": False, "z_score": 0.0}

        if self._state.n_seen < self.min_samples:
            return result

        # Z-score of current error vs running statistics
        mu = float(self._stats.mean[0])
        sigma = float(self._stats.std[0])
        z = (reconstruction_error - mu) / (sigma + 1e-8)
        result["z_score"] = z

        if abs(z) > self.drift_threshold:
            result["drift"] = True
            self._state.drift_events.append(self._state.n_seen)
        elif abs(z) > self.warning_threshold:
            result["warning"] = True
            self._state.warning_events.append(self._state.n_seen)

        return result

    def reset_reference(self) -> None:
        """Reset reference statistics (after confirmed drift)."""
        self._stats.reset()
        self._state.reference_error_mean = 0.0
        self._state.reference_error_std = 1.0

    @property
    def n_drift_events(self) -> int:
        return len(self._state.drift_events)

    @property
    def n_warning_events(self) -> int:
        return len(self._state.warning_events)

    def summary(self) -> Dict[str, Any]:
        return {
            "n_seen": self._state.n_seen,
            "n_drift_events": self.n_drift_events,
            "n_warning_events": self.n_warning_events,
            "drift_events": self._state.drift_events,
            "warning_events": self._state.warning_events,
            "current_error_mean": float(self._stats.mean[0]),
            "current_error_std": float(self._stats.std[0]),
        }


# ============================================================================
# Memory-bounded streaming algorithm
# ============================================================================

class MemoryBoundedStreamTT:
    """Memory-bounded online TT learning.

    Maintains a fixed-size buffer of recent data. When the buffer is full,
    evicts the oldest entry (FIFO). Periodically recompresses the buffer
    to update the TT model.

    Args:
        max_buffer_size: Maximum number of data points in buffer.
        tt_rank: TT bond dimension.
        recompress_every: Recompress every N updates.
        forget_factor: Exponential forgetting factor.
    """

    def __init__(
        self,
        max_buffer_size: int = 500,
        tt_rank: int = 8,
        recompress_every: int = 50,
        forget_factor: float = 0.99,
    ):
        self.max_buffer_size = max_buffer_size
        self.tt_rank = tt_rank
        self.recompress_every = recompress_every
        self.forget_factor = forget_factor

        self._buffer: Deque[np.ndarray] = deque(maxlen=max_buffer_size)
        self._cores: Optional[List[np.ndarray]] = None
        self._step = 0
        self._svd: Optional[StreamingSVD] = None
        self._n_features: Optional[int] = None

    def update(
        self,
        x: np.ndarray,
    ) -> Optional[float]:
        """Add a new data point to the buffer.

        Args:
            x: New observation (n_features,).

        Returns:
            Reconstruction error if recompression occurred, else None.
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)

        if self._n_features is None:
            self._n_features = len(x)
            self._svd = StreamingSVD(self.tt_rank, self._n_features, self.forget_factor)

        self._buffer.append(x)
        self._svd.update(x)
        self._step += 1

        if self._step % self.recompress_every == 0:
            return self._recompress()

        return None

    def _recompress(self) -> float:
        """Recompress buffer contents.

        Returns:
            Reconstruction error.
        """
        if len(self._buffer) < 2:
            return float("inf")

        data = np.stack(list(self._buffer))  # (buffer_size, n_features)
        U, s, Vt = np.linalg.svd(data, full_matrices=False)
        r = min(self.tt_rank, len(s))

        core_left = (U[:, :r] * s[:r]).reshape(1, data.shape[0], r).astype(np.float32)
        core_right = Vt[:r].reshape(r, data.shape[1], 1).astype(np.float32)
        self._cores = [core_left, core_right]

        # Compute reconstruction error
        recon = (U[:, :r] * s[:r]) @ Vt[:r]
        err = float(np.linalg.norm(data - recon) / (np.linalg.norm(data) + 1e-15))
        return err

    @property
    def cores(self) -> Optional[List[np.ndarray]]:
        """Current TT cores."""
        return self._cores

    @property
    def buffer_size(self) -> int:
        """Current number of data points in buffer."""
        return len(self._buffer)

    def get_components(self) -> Optional[np.ndarray]:
        """Get current streaming SVD components.

        Returns:
            Left singular vectors (n_features, rank).
        """
        if self._svd is None:
            return None
        return self._svd.components

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct x using current model.

        Args:
            x: Input to reconstruct.

        Returns:
            Reconstructed x.
        """
        if self._svd is None:
            return x
        return self._svd.reconstruct(x)


# ============================================================================
# Online Tucker update
# ============================================================================

class OnlineTuckerUpdate:
    """Incremental Tucker decomposition update.

    Maintains Tucker factors and updates them with new data slices.

    Args:
        n_modes: Number of tensor modes.
        mode_ranks: Tucker rank per mode.
        forget_factor: Exponential forgetting.
    """

    def __init__(
        self,
        n_modes: int,
        mode_ranks: List[int],
        forget_factor: float = 0.99,
    ):
        assert len(mode_ranks) == n_modes
        self.n_modes = n_modes
        self.mode_ranks = mode_ranks
        self.forget_factor = forget_factor

        # Streaming SVD per mode
        self._mode_svds: List[Optional[StreamingSVD]] = [None] * n_modes
        self._step = 0

    def update(self, tensor_slice: np.ndarray, mode: int = 0) -> None:
        """Update the Tucker factor for a given mode.

        Args:
            tensor_slice: A slice along the given mode.
            mode: Mode index to update.
        """
        assert 0 <= mode < self.n_modes
        flat = tensor_slice.reshape(-1, tensor_slice.shape[-1]) if tensor_slice.ndim > 1 else tensor_slice.reshape(1, -1)
        n_features = flat.shape[1]

        if self._mode_svds[mode] is None:
            self._mode_svds[mode] = StreamingSVD(
                self.mode_ranks[mode], n_features, self.forget_factor
            )

        self._mode_svds[mode].update(flat)
        self._step += 1

    def get_factor(self, mode: int) -> Optional[np.ndarray]:
        """Get the current Tucker factor matrix for a mode.

        Args:
            mode: Mode index.

        Returns:
            Factor matrix (n_features, rank) or None if not yet initialized.
        """
        if self._mode_svds[mode] is None:
            return None
        return self._mode_svds[mode].components

    def get_all_factors(self) -> List[Optional[np.ndarray]]:
        """Get all Tucker factor matrices."""
        return [self.get_factor(m) for m in range(self.n_modes)]


# ============================================================================
# Online covariance tensor
# ============================================================================

class OnlineCovarianceTensor:
    """Online (streaming) covariance matrix estimator.

    Supports exponential forgetting and provides a low-rank approximation.

    Args:
        n_assets: Number of assets.
        rank: Rank for low-rank approximation.
        forget_factor: Exponential forgetting.
    """

    def __init__(
        self,
        n_assets: int,
        rank: int = 8,
        forget_factor: float = 0.99,
    ):
        self.n_assets = n_assets
        self.rank = rank
        self.forget_factor = forget_factor

        self._n: float = 0.0
        self._mean: np.ndarray = np.zeros(n_assets)
        self._cov: np.ndarray = np.eye(n_assets) * 0.01
        self._svd = StreamingSVD(rank, n_assets, forget_factor)

    def update(self, returns: np.ndarray) -> None:
        """Update covariance estimate with new return vector.

        Args:
            returns: Return vector (n_assets,).
        """
        x = np.asarray(returns, dtype=np.float64).reshape(-1)[:self.n_assets]
        if len(x) < self.n_assets:
            x = np.pad(x, (0, self.n_assets - len(x)))

        self._n = self._n * self.forget_factor + 1.0
        self._mean = self._mean * self.forget_factor + x / self._n

        # Sherman-Morrison rank-1 covariance update
        diff = x - self._mean
        self._cov = self._cov * self.forget_factor + np.outer(diff, diff) / self._n

        # Update streaming SVD with flattened covariance row
        self._svd.update(diff)

    @property
    def covariance(self) -> np.ndarray:
        """Current covariance matrix (n_assets, n_assets)."""
        return self._cov.copy()

    @property
    def correlation(self) -> np.ndarray:
        """Current correlation matrix."""
        std = np.sqrt(np.diag(self._cov) + 1e-15)
        return self._cov / np.outer(std, std)

    def low_rank_covariance(self) -> np.ndarray:
        """Low-rank approximation of covariance.

        Returns:
            Low-rank covariance (n_assets, n_assets).
        """
        U = self._svd.components  # (n_assets, rank)
        s = self._svd.singular_values  # (rank,)
        return (U * s ** 2) @ U.T


# ============================================================================
# Drift-aware rank adjustment
# ============================================================================

class DriftAwareOnlineTT:
    """Online TT model that adjusts rank upon drift detection.

    When drift is detected, increases the TT rank to capture new patterns.
    When stable, optionally prunes rank to improve efficiency.

    Args:
        initial_rank: Starting bond dimension.
        max_rank: Maximum allowed rank.
        n_features: Feature dimension.
        forget_factor: Forgetting factor.
        drift_threshold: Z-score for drift alarm.
    """

    def __init__(
        self,
        initial_rank: int = 4,
        max_rank: int = 32,
        n_features: int = 64,
        forget_factor: float = 0.99,
        drift_threshold: float = 3.0,
    ):
        self.max_rank = max_rank
        self.n_features = n_features
        self.forget_factor = forget_factor

        self._rank = initial_rank
        self._model = MemoryBoundedStreamTT(
            max_buffer_size=500,
            tt_rank=initial_rank,
            recompress_every=50,
            forget_factor=forget_factor,
        )
        self._drift_detector = TensorDriftDetector(
            drift_threshold=drift_threshold,
        )
        self._step = 0

    def update(self, x: np.ndarray) -> Dict[str, Any]:
        """Process a new data point.

        Args:
            x: New observation (n_features,).

        Returns:
            Dict with drift status, current rank, and reconstruction error.
        """
        x = np.asarray(x).reshape(-1)

        recon_err = self._model.update(x)
        result: Dict[str, Any] = {
            "rank": self._rank,
            "recon_error": recon_err,
            "drift": False,
            "warning": False,
        }

        if recon_err is not None:
            drift_result = self._drift_detector.update(recon_err)
            result.update(drift_result)

            if drift_result["drift"]:
                # Increase rank
                new_rank = min(self._rank * 2, self.max_rank)
                if new_rank != self._rank:
                    self._rank = new_rank
                    self._model = MemoryBoundedStreamTT(
                        max_buffer_size=500,
                        tt_rank=self._rank,
                        recompress_every=50,
                        forget_factor=self.forget_factor,
                    )
                    self._drift_detector.reset_reference()
                result["rank"] = self._rank

        self._step += 1
        return result

    @property
    def current_rank(self) -> int:
        return self._rank

    def drift_summary(self) -> Dict[str, Any]:
        return self._drift_detector.summary()


# ============================================================================
# Online factorization with momentum
# ============================================================================

class MomentumOnlineTT:
    """Online TT update with momentum (heavy-ball method).

    Uses a momentum buffer to smooth gradient updates, leading to
    faster convergence on smooth objective surfaces.

    Args:
        cores: Initial TT cores.
        learning_rate: Base learning rate.
        momentum: Momentum coefficient (0.9 typical).
        weight_decay: L2 regularization.
    """

    def __init__(
        self,
        cores: List[np.ndarray],
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-5,
    ):
        self.cores = [np.array(c, dtype=np.float64) for c in cores]
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = [np.zeros_like(c) for c in self.cores]
        self._step = 0
        self.loss_history: List[float] = []

    def update(self, x: np.ndarray, target: Optional[float] = None) -> float:
        """Process one data point and update cores with momentum.

        Args:
            x: Input vector.
            target: Optional target value. If None, uses reconstruction error.

        Returns:
            Current loss value.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        self._step += 1

        # Compute reconstruction error as loss
        loss = self._reconstruction_loss(x, target)
        self.loss_history.append(loss)

        # Compute gradients via rank-1 ALS update
        grads = self._compute_gradients(x)

        # Momentum update
        for i, (core, grad, vel) in enumerate(zip(self.cores, grads, self._velocity)):
            new_vel = self.momentum * vel - self.learning_rate * (
                grad + self.weight_decay * core
            )
            self.cores[i] = core + new_vel
            self._velocity[i] = new_vel

        return loss

    def _reconstruction_loss(self, x: np.ndarray, target: Optional[float]) -> float:
        """Compute reconstruction loss for current observation."""
        if target is not None:
            return float(target)

        # TT inner product with x as proxy loss
        n_sites = len(self.cores)
        phys_dims = [c.shape[1] for c in self.cores]
        total_d = int(np.prod(phys_dims))
        x_trunc = x[:total_d] if len(x) >= total_d else np.pad(x, (0, total_d - len(x)))
        x_reshaped = x_trunc.reshape(phys_dims)

        bond = np.ones(1)
        for i, core in enumerate(self.cores):
            r_l, d, r_r = core.shape
            xi = x_reshaped[i] if n_sites > 1 else x_reshaped.reshape(-1)
            r_l_actual = min(len(bond), r_l)
            bond = np.einsum("a,b,abr->r", bond[:r_l_actual], xi[:d], core[:r_l_actual, :d, :])

        return float(1.0 - bond[0] ** 2) if bond.shape == (1,) else float(1.0 - np.sum(bond ** 2))

    def _compute_gradients(self, x: np.ndarray) -> List[np.ndarray]:
        """Compute approximate gradients via ALS."""
        return rank1_als_update(
            self.cores, x, learning_rate=0.0, regularization=0.0
        )  # returns cores but we use as gradient proxy

    @property
    def effective_lr(self) -> float:
        """Effective learning rate accounting for momentum."""
        return self.learning_rate / (1.0 - self.momentum + 1e-8)


# ============================================================================
# Tensor sketching for streaming approximation
# ============================================================================

class TensorSketch:
    """Count sketch / tensor sketch for streaming low-rank approximation.

    Maintains a compressed sketch of incoming data without storing
    all data points, using random hashing.

    Args:
        n_features: Number of features per data point.
        sketch_size: Size of the sketch (compressed dimension).
        rng_seed: Random seed.
    """

    def __init__(
        self,
        n_features: int,
        sketch_size: int = 128,
        rng_seed: int = 42,
    ):
        self.n_features = n_features
        self.sketch_size = sketch_size
        rng = np.random.default_rng(rng_seed)

        # Random hashing for count sketch
        self._hash_idx = rng.integers(0, sketch_size, size=n_features)
        self._signs = rng.choice([-1.0, 1.0], size=n_features)
        self._sketch = np.zeros(sketch_size)
        self._n_updates = 0

    def update(self, x: np.ndarray) -> None:
        """Update sketch with a new data vector.

        Args:
            x: Data vector (n_features,).
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)[:self.n_features]
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))

        signed_x = x * self._signs
        np.add.at(self._sketch, self._hash_idx, signed_x)
        self._n_updates += 1

    def query(self, x: np.ndarray) -> float:
        """Estimate inner product <x, accumulated_data> from sketch.

        Args:
            x: Query vector.

        Returns:
            Estimated inner product.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)[:self.n_features]
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))

        signed_x = x * self._signs
        estimate = np.sum(signed_x * self._sketch[self._hash_idx])
        return float(estimate)

    def get_sketch(self) -> np.ndarray:
        """Get the current sketch vector."""
        return self._sketch.copy()

    def reset(self) -> None:
        """Reset the sketch."""
        self._sketch = np.zeros(self.sketch_size)
        self._n_updates = 0

    @property
    def n_updates(self) -> int:
        return self._n_updates


# ============================================================================
# Online anomaly detection in tensor space
# ============================================================================

class OnlineTensorAnomalyDetector:
    """Detect anomalies in streaming tensor data using reconstruction error.

    Maintains a streaming SVD of normal data and flags points where
    the reconstruction error is significantly above the baseline.

    Args:
        n_features: Feature dimension.
        rank: Low-rank approximation rank.
        threshold_sigma: Number of standard deviations for anomaly threshold.
        warmup_steps: Number of steps before anomaly detection activates.
        forget_factor: Exponential forgetting for background statistics.
    """

    def __init__(
        self,
        n_features: int,
        rank: int = 8,
        threshold_sigma: float = 3.0,
        warmup_steps: int = 50,
        forget_factor: float = 0.99,
    ):
        self.n_features = n_features
        self.rank = rank
        self.threshold_sigma = threshold_sigma
        self.warmup_steps = warmup_steps
        self._svd = StreamingSVD(rank, n_features, forget_factor)
        self._error_stats = RunningStats(1, forget_factor=forget_factor)
        self._step = 0
        self._anomaly_log: List[Tuple[int, float]] = []

    def update(self, x: np.ndarray) -> Dict[str, Any]:
        """Process a new data point and check for anomalies.

        Args:
            x: Data vector (n_features,).

        Returns:
            Dict with is_anomaly, reconstruction_error, threshold, z_score.
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        self._step += 1

        # Update SVD
        self._svd.update(x)

        # Compute reconstruction error
        recon = self._svd.reconstruct(x)
        error = float(np.linalg.norm(x - recon) / (np.linalg.norm(x) + 1e-8))

        # Update error statistics
        self._error_stats.update(np.array([error]))

        result: Dict[str, Any] = {
            "is_anomaly": False,
            "reconstruction_error": error,
            "threshold": None,
            "z_score": 0.0,
            "step": self._step,
        }

        if self._step < self.warmup_steps:
            return result

        mu = float(self._error_stats.mean[0])
        sigma = float(self._error_stats.std[0])
        z = (error - mu) / (sigma + 1e-8)
        threshold = mu + self.threshold_sigma * sigma

        result["z_score"] = z
        result["threshold"] = threshold

        if error > threshold:
            result["is_anomaly"] = True
            self._anomaly_log.append((self._step, error))

        return result

    @property
    def n_anomalies(self) -> int:
        return len(self._anomaly_log)

    def anomaly_rate(self) -> float:
        """Fraction of steps flagged as anomalies."""
        if self._step == 0:
            return 0.0
        return len(self._anomaly_log) / self._step

    def summary(self) -> Dict[str, Any]:
        return {
            "n_steps": self._step,
            "n_anomalies": self.n_anomalies,
            "anomaly_rate": self.anomaly_rate(),
            "mean_error": float(self._error_stats.mean[0]),
            "std_error": float(self._error_stats.std[0]),
        }


# ---------------------------------------------------------------------------
# Section: Advanced online tensor learning
# ---------------------------------------------------------------------------

import numpy as np
import warnings
from collections import deque as _dq


class OnlineKalmanTT:
    """
    Kalman filter for a low-rank state-space model.

    Models the state evolution as::

        x_t = A x_{t-1} + w_t,   w_t ~ N(0, Q)
        y_t = C x_t + v_t,       v_t ~ N(0, R)

    Parameters
    ----------
    state_dim : int
        Latent state dimension.
    obs_dim : int
        Observation dimension.
    process_noise : float
        Diagonal process noise variance.
    obs_noise : float
        Diagonal observation noise variance.
    """

    def __init__(
        self,
        state_dim: int = 8,
        obs_dim: int = 64,
        process_noise: float = 1e-3,
        obs_noise: float = 1e-2,
    ) -> None:
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self._A = np.eye(state_dim) * 0.95
        self._C = np.random.randn(obs_dim, state_dim) * 0.1
        self._Q = np.eye(state_dim) * process_noise
        self._R = np.eye(obs_dim) * obs_noise
        self._x = np.zeros(state_dim)
        self._P = np.eye(state_dim)
        self._n_updates = 0

    def predict(self) -> None:
        """Kalman predict step."""
        self._x = self._A @ self._x
        self._P = self._A @ self._P @ self._A.T + self._Q

    def update(self, y: np.ndarray) -> np.ndarray:
        """
        Kalman update step.

        Parameters
        ----------
        y : np.ndarray, shape (obs_dim,)

        Returns
        -------
        x_updated : np.ndarray, shape (state_dim,)
        """
        y_pred = self._C @ self._x
        innov = y - y_pred
        S = self._C @ self._P @ self._C.T + self._R
        K = self._P @ self._C.T @ np.linalg.inv(S)
        self._x = self._x + K @ innov
        self._P = (np.eye(self.state_dim) - K @ self._C) @ self._P
        self._n_updates += 1
        return self._x.copy()

    def filter(self, observations: np.ndarray) -> np.ndarray:
        """
        Run Kalman filter over a sequence.

        Parameters
        ----------
        observations : np.ndarray, shape (T, obs_dim)

        Returns
        -------
        states : np.ndarray, shape (T, state_dim)
        """
        T = observations.shape[0]
        states = np.zeros((T, self.state_dim), dtype=np.float32)
        for t in range(T):
            self.predict()
            x = self.update(observations[t])
            states[t] = x
        return states

    def reconstruct(self, states: np.ndarray) -> np.ndarray:
        """Reconstruct observations from latent states."""
        return (states @ self._C.T).astype(np.float32)

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    @property
    def n_updates(self) -> int:
        return self._n_updates


class BayesianOnlineTT:
    """
    Bayesian online learning for TT core parameters.

    Maintains a mean-field Gaussian posterior over TT core elements and
    updates it incrementally using variational inference.

    Parameters
    ----------
    n_features : int
    n_components : int
    prior_std : float
    lr : float
    """

    def __init__(
        self,
        n_features: int = 64,
        n_components: int = 8,
        prior_std: float = 1.0,
        lr: float = 0.01,
    ) -> None:
        self.n_features = n_features
        self.n_components = n_components
        self.prior_std = prior_std
        self.lr = lr
        self._mu = np.zeros((n_features, n_components), dtype=np.float64)
        self._log_sigma2 = np.full((n_features, n_components), np.log(prior_std ** 2))
        self._n_updates = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Posterior mean prediction, shape (n_components,)."""
        return (x @ self._mu).astype(np.float32)

    def predict_with_uncertainty(self, x: np.ndarray) -> tuple:
        """Returns (mean, std) each shape (n_components,)."""
        mean = x @ self._mu
        sigma2 = np.exp(self._log_sigma2)
        pred_var = x ** 2 @ sigma2
        return mean.astype(np.float32), np.sqrt(pred_var).astype(np.float32)

    def update(self, x: np.ndarray, y: np.ndarray, noise_var: float = 0.1) -> None:
        """Update posterior with a new observation (x, y)."""
        sigma2 = np.exp(self._log_sigma2)
        pred = x @ self._mu
        error = y - pred
        grad_mu = np.outer(x, error) / noise_var - self._mu / (self.prior_std ** 2)
        self._mu += self.lr * grad_mu
        precision_likelihood = np.outer(x ** 2, np.ones(self.n_components)) / noise_var
        precision_prior = 1.0 / (self.prior_std ** 2)
        grad_lv = 0.5 * (1.0 - sigma2 * (precision_likelihood + precision_prior))
        self._log_sigma2 += self.lr * grad_lv
        self._n_updates += 1

    def kl_divergence(self) -> float:
        """KL divergence from posterior to prior."""
        sigma2 = np.exp(self._log_sigma2)
        prior_var = self.prior_std ** 2
        kl = 0.5 * np.sum(
            sigma2 / prior_var + self._mu ** 2 / prior_var - 1
            - self._log_sigma2 + np.log(prior_var)
        )
        return float(kl)


class GradientTrackerTT:
    """
    Tracks gradient statistics for TT core updates.

    Parameters
    ----------
    window : int
        Rolling window for statistics.
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self._grad_norms: _dq = _dq(maxlen=window)
        self._grad_max: _dq = _dq(maxlen=window)
        self._n_recorded = 0

    def record(self, grad: np.ndarray) -> None:
        """Record a gradient array."""
        g = np.array(grad, dtype=np.float64)
        self._grad_norms.append(float(np.linalg.norm(g)))
        self._grad_max.append(float(np.abs(g).max()))
        self._n_recorded += 1

    def is_exploding(self, threshold: float = 100.0) -> bool:
        if not self._grad_norms:
            return False
        return self._grad_norms[-1] > threshold

    def is_vanishing(self, threshold: float = 1e-7) -> bool:
        if not self._grad_norms:
            return False
        return self._grad_norms[-1] < threshold

    def summary(self) -> dict:
        if not self._grad_norms:
            return {}
        norms = np.array(self._grad_norms)
        return {
            "n_recorded": self._n_recorded,
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
            "max_norm": float(norms.max()),
            "min_norm": float(norms.min()),
            "latest_norm": float(self._grad_norms[-1]),
            "is_exploding": self.is_exploding(),
            "is_vanishing": self.is_vanishing(),
        }


class AdaptiveLearningRateTT:
    """
    AdaGrad-style adaptive per-core learning rate for TT networks.

    Parameters
    ----------
    base_lr : float
    eps : float
    max_lr : float
    """

    def __init__(self, base_lr: float = 0.01, eps: float = 1e-8, max_lr: float = 0.1) -> None:
        self.base_lr = base_lr
        self.eps = eps
        self.max_lr = max_lr
        self._acc: dict = {}

    def step(self, core_name: str, grad: np.ndarray) -> float:
        """Compute adaptive learning rate for a core."""
        g_sq = float(np.sum(grad ** 2))
        if core_name not in self._acc:
            self._acc[core_name] = 0.0
        self._acc[core_name] += g_sq
        lr = self.base_lr / (np.sqrt(self._acc[core_name]) + self.eps)
        return float(min(lr, self.max_lr))

    def reset(self, core_name=None) -> None:
        if core_name is not None:
            self._acc.pop(core_name, None)
        else:
            self._acc.clear()

    def learning_rates(self) -> dict:
        return {
            k: float(self.base_lr / (np.sqrt(v) + self.eps))
            for k, v in self._acc.items()
        }


class OnlineModelSelector:
    """
    Online selection between multiple streaming TT models using
    exponentially-weighted prediction performance.

    Parameters
    ----------
    model_names : list of str
    window : int
    decay : float
    """

    def __init__(self, model_names: list, window: int = 50, decay: float = 0.99) -> None:
        self.model_names = model_names
        self.window = window
        self.decay = decay
        self._errors: dict = {name: _dq(maxlen=window) for name in model_names}
        self._weights = {name: 1.0 / max(1, len(model_names)) for name in model_names}

    def update(self, predictions: dict, actual: np.ndarray) -> None:
        """Update model performance weights given actual observation."""
        for name in self.model_names:
            if name in predictions:
                pred = np.array(predictions[name])
                err = float(np.mean((pred - actual) ** 2))
                self._errors[name].append(err)

        avg_errors = {}
        for name in self.model_names:
            if self._errors[name]:
                avg_errors[name] = float(np.mean(list(self._errors[name])))
            else:
                avg_errors[name] = float("inf")

        min_err = min(avg_errors.values())
        raw_w = {n: np.exp(-(avg_errors[n] - min_err)) for n in self.model_names}
        total = sum(raw_w.values()) + 1e-12
        for name in self.model_names:
            target_w = raw_w[name] / total
            self._weights[name] = (
                self.decay * self._weights[name] + (1 - self.decay) * target_w
            )

    def best_model(self) -> str:
        """Return name of currently best-performing model."""
        return max(self._weights, key=self._weights.get)

    def ensemble_predict(self, predictions: dict) -> np.ndarray:
        """Weighted ensemble prediction."""
        total_w = sum(self._weights.get(n, 0.0) for n in predictions)
        if total_w < 1e-12:
            return np.array(list(predictions.values())[0], dtype=np.float32)
        result = None
        for name, pred in predictions.items():
            w = self._weights.get(name, 0.0) / total_w
            arr = np.array(pred, dtype=np.float64) * w
            result = arr if result is None else result + arr
        return result.astype(np.float32)

    def weights(self) -> dict:
        return dict(self._weights)


class StreamingCorrelationEstimator:
    """
    Streaming exponentially-weighted correlation estimator.

    Maintains (N x N) correlation matrix estimate using a running
    scatter matrix accumulation with optional shrinkage.

    Parameters
    ----------
    n_assets : int
    half_life : float
        Exponential decay half-life in observations.
    shrinkage : float
        Ledoit-Wolf shrinkage intensity.
    """

    def __init__(self, n_assets: int = 64, half_life: float = 63.0, shrinkage: float = 0.1) -> None:
        self.n_assets = n_assets
        self.half_life = half_life
        self.shrinkage = shrinkage
        import math
        self._decay = math.pow(0.5, 1.0 / half_life)
        self._S = np.zeros((n_assets, n_assets), dtype=np.float64)
        self._mu = np.zeros(n_assets, dtype=np.float64)
        self._n = 0.0

    def update(self, x: np.ndarray) -> None:
        """Update with a new observation or batch."""
        x = np.atleast_2d(x).astype(np.float64)
        lam = self._decay
        for xi in x:
            self._S = lam * self._S + (1 - lam) * np.outer(xi, xi)
            self._mu = lam * self._mu + (1 - lam) * xi
            self._n = lam * self._n + 1.0

    def correlation(self) -> np.ndarray:
        """Return current correlation matrix estimate."""
        cov = self._S - np.outer(self._mu, self._mu)
        std = np.sqrt(np.diag(cov) + 1e-12)
        corr = cov / np.outer(std, std)
        # Apply shrinkage toward identity
        alpha = self.shrinkage
        shrunk = (1 - alpha) * corr + alpha * np.eye(self.n_assets)
        np.clip(shrunk, -1, 1, out=shrunk)
        np.fill_diagonal(shrunk, 1.0)
        return shrunk.astype(np.float32)

    def covariance(self) -> np.ndarray:
        """Return current covariance matrix estimate."""
        cov = self._S - np.outer(self._mu, self._mu)
        alpha = self.shrinkage
        mu_target = np.trace(cov) / self.n_assets
        shrunk = (1 - alpha) * cov + alpha * mu_target * np.eye(self.n_assets)
        return (shrunk + 1e-6 * np.eye(self.n_assets)).astype(np.float32)

    def reset(self) -> None:
        self._S = np.zeros((self.n_assets, self.n_assets), dtype=np.float64)
        self._mu = np.zeros(self.n_assets, dtype=np.float64)
        self._n = 0.0


class StreamingRegressionTT:
    """
    Online least-squares regression with TT-structured weight matrix.

    Uses the recursive least squares (RLS) algorithm for exact online updates.

    Parameters
    ----------
    n_features : int
        Input dimensionality.
    n_targets : int
        Output dimensionality.
    forgetting_factor : float
        Lambda in (0, 1]. 1.0 = no forgetting.
    ridge : float
        L2 regularisation for initial covariance.
    """

    def __init__(
        self,
        n_features: int = 64,
        n_targets: int = 1,
        forgetting_factor: float = 0.99,
        ridge: float = 1e-3,
    ) -> None:
        self.n_features = n_features
        self.n_targets = n_targets
        self.lam = forgetting_factor
        self._W = np.zeros((n_features, n_targets), dtype=np.float64)
        self._P = np.eye(n_features) / ridge
        self._n_updates = 0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        RLS update with new observation.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)
        y : np.ndarray, shape (n_targets,)
        """
        x = x.reshape(-1).astype(np.float64)
        y = y.reshape(-1).astype(np.float64)
        lam = self.lam
        Px = self._P @ x
        denom = lam + x @ Px
        K = Px / denom
        error = y - x @ self._W
        self._W += np.outer(K, error)
        self._P = (self._P - np.outer(Px, Px) / denom) / lam
        self._n_updates += 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict output for input x."""
        return (x @ self._W).astype(np.float32)

    @property
    def weights(self) -> np.ndarray:
        return self._W.astype(np.float32)

    @property
    def n_updates(self) -> int:
        return self._n_updates

