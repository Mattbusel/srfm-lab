"""
regime_tensor.py — Regime-conditioned tensor operations for TensorNet (Project AETERNUS).

Provides:
  - Separate TT decompositions per market regime
  - Regime detection via HMM on tensor residuals
  - Cross-regime interpolation of TT cores
  - Regime probability weighting of tensor operations
  - Viterbi decoding for regime sequences
  - Regime-conditioned covariance tensor
  - Soft switching via Bayesian mixture of TT models
  - Regime change point detection
  - Regime-aware rolling compression
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap


# ============================================================================
# HMM for regime detection
# ============================================================================

@dataclass
class HMMParams:
    """Parameters for a Gaussian HMM."""
    n_states: int
    transition_matrix: np.ndarray   # (n_states, n_states)
    means: np.ndarray               # (n_states, n_features)
    covariances: np.ndarray         # (n_states, n_features, n_features)
    initial_probs: np.ndarray       # (n_states,)

    def __post_init__(self):
        assert self.transition_matrix.shape == (self.n_states, self.n_states)
        assert self.means.shape[0] == self.n_states
        assert self.covariances.shape[0] == self.n_states
        assert self.initial_probs.shape == (self.n_states,)


def init_hmm_params(
    n_states: int,
    n_features: int,
    rng: Optional[np.random.Generator] = None,
) -> HMMParams:
    """Initialize random HMM parameters.

    Args:
        n_states: Number of hidden states.
        n_features: Number of observation features.
        rng: Random number generator.

    Returns:
        HMMParams with random initialization.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Random transition matrix (row-stochastic)
    A = rng.dirichlet(np.ones(n_states), size=n_states)
    # Diagonal-dominant: prefer staying in same state
    A = A * 0.1 + np.eye(n_states) * 0.9
    A /= A.sum(axis=1, keepdims=True)

    means = rng.normal(0, 1, (n_states, n_features))
    covs = np.stack([np.eye(n_features) * (0.5 + rng.exponential(0.5)) for _ in range(n_states)])
    pi = np.ones(n_states) / n_states

    return HMMParams(
        n_states=n_states,
        transition_matrix=A,
        means=means,
        covariances=covs,
        initial_probs=pi,
    )


def gaussian_log_likelihood(
    x: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
) -> float:
    """Log-likelihood of x under a multivariate Gaussian.

    Args:
        x: Observation vector (n_features,).
        mean: Mean vector (n_features,).
        cov: Covariance matrix (n_features, n_features).

    Returns:
        Log-likelihood scalar.
    """
    n = len(x)
    diff = x - mean
    try:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            logdet = -np.inf
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(n))
        mahal = float(diff @ cov_inv @ diff)
        return -0.5 * (n * math.log(2 * math.pi) + logdet + mahal)
    except np.linalg.LinAlgError:
        return -np.inf


def hmm_forward(
    observations: np.ndarray,
    params: HMMParams,
) -> Tuple[np.ndarray, float]:
    """Forward algorithm for HMM (scaled).

    Args:
        observations: Array of shape (T, n_features).
        params: HMM parameters.

    Returns:
        (alpha, log_likelihood) where alpha is (T, n_states).
    """
    T = len(observations)
    K = params.n_states

    alpha = np.zeros((T, K))
    scaling = np.zeros(T)

    # Initialization
    for k in range(K):
        ll = gaussian_log_likelihood(observations[0], params.means[k], params.covariances[k])
        alpha[0, k] = params.initial_probs[k] * math.exp(max(ll, -500))

    scale_t = alpha[0].sum()
    if scale_t > 0:
        alpha[0] /= scale_t
    scaling[0] = max(scale_t, 1e-300)

    # Recursion
    for t in range(1, T):
        for k in range(K):
            ll = gaussian_log_likelihood(observations[t], params.means[k], params.covariances[k])
            emit = math.exp(max(ll, -500))
            alpha[t, k] = emit * float(alpha[t - 1] @ params.transition_matrix[:, k])

        scale_t = alpha[t].sum()
        if scale_t > 0:
            alpha[t] /= scale_t
        scaling[t] = max(scale_t, 1e-300)

    log_likelihood = float(np.sum(np.log(scaling)))
    return alpha, log_likelihood


def hmm_backward(
    observations: np.ndarray,
    params: HMMParams,
    scaling: np.ndarray,
) -> np.ndarray:
    """Backward algorithm for HMM.

    Args:
        observations: (T, n_features).
        params: HMM parameters.
        scaling: Scaling factors from forward pass.

    Returns:
        Beta array of shape (T, n_states).
    """
    T = len(observations)
    K = params.n_states
    beta = np.zeros((T, K))
    beta[T - 1] = 1.0

    for t in range(T - 2, -1, -1):
        for j in range(K):
            total = 0.0
            for k in range(K):
                ll = gaussian_log_likelihood(observations[t + 1], params.means[k], params.covariances[k])
                emit = math.exp(max(ll, -500))
                total += params.transition_matrix[j, k] * emit * beta[t + 1, k]
            beta[t, j] = total

        scale = scaling[t + 1] if t + 1 < T else 1.0
        if scale > 0:
            beta[t] /= scale

    return beta


def hmm_viterbi(
    observations: np.ndarray,
    params: HMMParams,
) -> np.ndarray:
    """Viterbi algorithm: most likely state sequence.

    Args:
        observations: (T, n_features).
        params: HMM parameters.

    Returns:
        State sequence (T,) with integer state labels.
    """
    T = len(observations)
    K = params.n_states

    log_delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=int)

    # Initialization
    for k in range(K):
        ll = gaussian_log_likelihood(observations[0], params.means[k], params.covariances[k])
        log_delta[0, k] = math.log(max(params.initial_probs[k], 1e-300)) + ll

    # Recursion
    for t in range(1, T):
        for k in range(K):
            ll = gaussian_log_likelihood(observations[t], params.means[k], params.covariances[k])
            scores = log_delta[t - 1] + np.log(params.transition_matrix[:, k] + 1e-300)
            best = int(np.argmax(scores))
            log_delta[t, k] = scores[best] + ll
            psi[t, k] = best

    # Backtrack
    states = np.zeros(T, dtype=int)
    states[T - 1] = int(np.argmax(log_delta[T - 1]))
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


def hmm_em_step(
    observations: np.ndarray,
    params: HMMParams,
    min_cov_diag: float = 1e-4,
) -> HMMParams:
    """One EM step to update HMM parameters.

    Args:
        observations: (T, n_features).
        params: Current HMM parameters.
        min_cov_diag: Minimum diagonal value for covariances.

    Returns:
        Updated HMMParams.
    """
    T, d = observations.shape
    K = params.n_states

    alpha, log_ll = hmm_forward(observations, params)
    scaling = np.ones(T)  # simplified; use from forward properly
    _, log_ll2 = hmm_forward(observations, params)
    alpha2, _ = hmm_forward(observations, params)
    beta = hmm_backward(observations, params, scaling)

    # Posterior state probabilities gamma
    gamma = alpha2 * beta
    gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

    # Update initial probs
    new_pi = gamma[0]

    # Update transition matrix
    new_A = np.zeros((K, K))
    for t in range(T - 1):
        for j in range(K):
            for k in range(K):
                ll = gaussian_log_likelihood(observations[t + 1], params.means[k], params.covariances[k])
                emit = math.exp(max(ll, -500))
                new_A[j, k] += alpha2[t, j] * params.transition_matrix[j, k] * emit * beta[t + 1, k]

    row_sums = new_A.sum(axis=1, keepdims=True)
    new_A /= row_sums + 1e-300

    # Update means
    new_means = np.zeros((K, d))
    for k in range(K):
        w = gamma[:, k].sum()
        if w > 1e-10:
            new_means[k] = (gamma[:, k, np.newaxis] * observations).sum(axis=0) / w

    # Update covariances
    new_covs = np.zeros((K, d, d))
    for k in range(K):
        w = gamma[:, k].sum()
        if w > 1e-10:
            diff = observations - new_means[k]
            new_covs[k] = (gamma[:, k, np.newaxis, np.newaxis] * diff[:, :, np.newaxis] * diff[:, np.newaxis, :]).sum(axis=0) / w
        else:
            new_covs[k] = np.eye(d)
        new_covs[k] += min_cov_diag * np.eye(d)

    return HMMParams(
        n_states=K,
        transition_matrix=new_A,
        means=new_means,
        covariances=new_covs,
        initial_probs=new_pi,
    )


def fit_hmm(
    observations: np.ndarray,
    n_states: int,
    n_iter: int = 20,
    rng_seed: int = 42,
    verbose: bool = False,
) -> HMMParams:
    """Fit HMM via EM (Baum-Welch).

    Args:
        observations: (T, n_features).
        n_states: Number of regimes.
        n_iter: EM iterations.
        rng_seed: Random seed.
        verbose: Print per-iteration log-likelihood.

    Returns:
        Fitted HMMParams.
    """
    rng = np.random.default_rng(rng_seed)
    params = init_hmm_params(n_states, observations.shape[1], rng)

    prev_ll = -np.inf
    for i in range(n_iter):
        _, ll = hmm_forward(observations, params)
        if verbose:
            print(f"  HMM EM iter {i+1}/{n_iter}: log_likelihood={ll:.4f}")
        if abs(ll - prev_ll) < 1e-6 and i > 2:
            break
        prev_ll = ll
        params = hmm_em_step(observations, params)

    return params


# ============================================================================
# Regime-conditioned TT
# ============================================================================

@dataclass
class RegimeTTModel:
    """Separate TT decomposition per market regime.

    Stores a TT model (list of cores) for each detected regime,
    along with HMM regime parameters for regime detection.

    Args:
        n_regimes: Number of market regimes.
        cores_per_regime: Dict mapping regime index -> list of TT cores.
        hmm_params: Fitted HMM parameters.
    """
    n_regimes: int
    cores_per_regime: Dict[int, List[np.ndarray]]
    hmm_params: Optional[HMMParams] = None
    regime_labels: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.regime_labels is None:
            self.regime_labels = [f"regime_{i}" for i in range(self.n_regimes)]

    def get_cores(self, regime: int) -> List[np.ndarray]:
        """Get TT cores for a given regime."""
        return self.cores_per_regime[regime]

    def get_jax_cores(self, regime: int) -> List[jnp.ndarray]:
        """Get TT cores as JAX arrays for a given regime."""
        return [jnp.array(c) for c in self.cores_per_regime[regime]]


class RegimeTensorOps:
    """Operations for regime-conditioned tensor networks.

    Provides:
    - Fitting a RegimeTTModel from time-series data
    - Getting current regime probabilities
    - Interpolating between regime TT cores
    - Applying the correct regime's TT for a given time step
    - Regime change point detection

    Args:
        n_regimes: Number of market regimes.
        max_rank: Maximum TT bond dimension per regime.
        n_em_iter: HMM EM iterations.
        rng_seed: Random seed.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        max_rank: int = 16,
        n_em_iter: int = 20,
        rng_seed: int = 42,
    ):
        self.n_regimes = n_regimes
        self.max_rank = max_rank
        self.n_em_iter = n_em_iter
        self.rng_seed = rng_seed
        self._model: Optional[RegimeTTModel] = None
        self._regime_sequence: Optional[np.ndarray] = None
        self._regime_probs: Optional[np.ndarray] = None

    def fit(
        self,
        observations: np.ndarray,
        data_tensor: np.ndarray,
        verbose: bool = False,
    ) -> RegimeTTModel:
        """Fit regime-conditioned TT models.

        Args:
            observations: Features for HMM training, shape (T, n_features).
            data_tensor: Data tensor to decompose per regime, shape (T, ...).
            verbose: Print progress.

        Returns:
            Fitted RegimeTTModel.
        """
        if verbose:
            print(f"Fitting HMM with {self.n_regimes} regimes ...")

        hmm_params = fit_hmm(
            observations, self.n_regimes, self.n_em_iter, self.rng_seed, verbose
        )

        regime_seq = hmm_viterbi(observations, hmm_params)
        self._regime_sequence = regime_seq

        alpha, _ = hmm_forward(observations, hmm_params)
        self._regime_probs = alpha

        if verbose:
            print("Fitting per-regime TT decompositions ...")

        cores_per_regime: Dict[int, List[np.ndarray]] = {}
        for regime in range(self.n_regimes):
            mask = regime_seq == regime
            if mask.sum() == 0:
                # No data for this regime; use all data with small weight
                mask = np.ones(len(regime_seq), dtype=bool)

            regime_data = data_tensor[mask]
            flat = regime_data.reshape(regime_data.shape[0], -1)

            # SVD-based TT proxy
            U, s, Vt = np.linalg.svd(flat, full_matrices=False)
            r = min(self.max_rank, len(s))
            core_left = (U[:, :r] * s[:r]).reshape(1, flat.shape[0], r).astype(np.float32)
            core_right = Vt[:r].reshape(r, flat.shape[1], 1).astype(np.float32)
            cores_per_regime[regime] = [core_left, core_right]

            if verbose:
                n_samples = mask.sum()
                print(f"  Regime {regime}: {n_samples} samples, rank={r}")

        self._model = RegimeTTModel(
            n_regimes=self.n_regimes,
            cores_per_regime=cores_per_regime,
            hmm_params=hmm_params,
        )
        return self._model

    def current_regime(
        self,
        observation: np.ndarray,
        use_soft: bool = False,
    ) -> Union[int, np.ndarray]:
        """Detect the current regime for a new observation.

        Args:
            observation: Single observation vector (n_features,).
            use_soft: If True, return probability distribution over regimes.

        Returns:
            Hard regime index (int) or soft probabilities (np.ndarray).
        """
        if self._model is None or self._model.hmm_params is None:
            raise RuntimeError("Call fit() first.")

        params = self._model.hmm_params
        K = params.n_states

        # Forward pass for single step: use uniform initial
        lls = np.array([
            gaussian_log_likelihood(observation, params.means[k], params.covariances[k])
            for k in range(K)
        ])
        lls = np.exp(lls - lls.max())  # numerical stability
        probs = lls * params.initial_probs
        probs /= probs.sum() + 1e-300

        if use_soft:
            return probs
        return int(np.argmax(probs))

    def interpolate_regime_cores(
        self,
        regime_a: int,
        regime_b: int,
        weight_a: float,
    ) -> List[jnp.ndarray]:
        """Interpolate TT cores between two regimes.

        Args:
            regime_a: First regime index.
            regime_b: Second regime index.
            weight_a: Weight for regime_a (weight_b = 1 - weight_a).

        Returns:
            Interpolated TT cores as JAX arrays.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        cores_a = self._model.cores_per_regime[regime_a]
        cores_b = self._model.cores_per_regime[regime_b]

        n = min(len(cores_a), len(cores_b))
        interpolated = []
        for i in range(n):
            ca = np.asarray(cores_a[i], dtype=np.float32)
            cb = np.asarray(cores_b[i], dtype=np.float32)

            # Align shapes (broadcast to compatible dims)
            min_shape = tuple(min(a, b) for a, b in zip(ca.shape, cb.shape))
            ca_s = ca[tuple(slice(0, s) for s in min_shape)]
            cb_s = cb[tuple(slice(0, s) for s in min_shape)]
            blended = weight_a * ca_s + (1.0 - weight_a) * cb_s
            interpolated.append(jnp.array(blended))

        return interpolated

    def weighted_regime_cores(
        self,
        regime_weights: np.ndarray,
    ) -> List[jnp.ndarray]:
        """Compute weighted average of TT cores across all regimes.

        Args:
            regime_weights: Probability weights over regimes (sums to 1).

        Returns:
            Weighted TT cores.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        weights = np.asarray(regime_weights, dtype=np.float32)
        weights /= weights.sum() + 1e-300

        # Use minimum number of cores across regimes
        n_cores = min(len(self._model.cores_per_regime[r]) for r in range(self.n_regimes))

        weighted = []
        for i in range(n_cores):
            core_sum = None
            for r in range(self.n_regimes):
                c = np.asarray(self._model.cores_per_regime[r][i], dtype=np.float32)
                if core_sum is None:
                    core_sum = weights[r] * c
                else:
                    min_shape = tuple(min(a, b) for a, b in zip(core_sum.shape, c.shape))
                    cs = core_sum[tuple(slice(0, s) for s in min_shape)]
                    cc = c[tuple(slice(0, s) for s in min_shape)]
                    core_sum = cs + weights[r] * cc
            weighted.append(jnp.array(core_sum))

        return weighted

    def regime_sequence(self) -> Optional[np.ndarray]:
        """Return the most likely regime sequence from Viterbi."""
        return self._regime_sequence

    def regime_probs(self) -> Optional[np.ndarray]:
        """Return soft regime probabilities from HMM forward pass."""
        return self._regime_probs

    def detect_change_points(
        self,
        min_segment_length: int = 10,
    ) -> List[int]:
        """Detect regime change points in the Viterbi sequence.

        Args:
            min_segment_length: Minimum segment length to report.

        Returns:
            List of time indices where regime changes occur.
        """
        if self._regime_sequence is None:
            raise RuntimeError("Call fit() first.")

        seq = self._regime_sequence
        change_points = []
        for t in range(1, len(seq)):
            if seq[t] != seq[t - 1]:
                change_points.append(t)

        # Filter by minimum segment length
        if min_segment_length > 1:
            filtered = []
            prev = 0
            for cp in change_points:
                if cp - prev >= min_segment_length:
                    filtered.append(cp)
                    prev = cp
            return filtered

        return change_points


# ============================================================================
# Regime-aware compression
# ============================================================================

class RegimeAwareCompressor:
    """Compress data differently per market regime.

    Maintains separate compression pipelines per regime,
    selecting which to use based on real-time regime detection.

    Args:
        n_regimes: Number of regimes.
        max_rank_per_regime: Max TT rank per regime.
        hmm_n_iter: HMM EM iterations.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        max_rank_per_regime: int = 16,
        hmm_n_iter: int = 20,
    ):
        from tensor_net.compression_pipeline import (
            CompressionPipeline,
            CompressionPipelineConfig,
        )

        self.n_regimes = n_regimes
        self.ops = RegimeTensorOps(n_regimes, max_rank_per_regime, hmm_n_iter)
        self.pipelines = {
            r: CompressionPipeline(
                CompressionPipelineConfig(max_rank=max_rank_per_regime)
            )
            for r in range(n_regimes)
        }
        self._current_regime: int = 0

    def fit(
        self,
        returns: np.ndarray,
        verbose: bool = False,
    ) -> None:
        """Fit regime models on historical return data.

        Args:
            returns: (T, n_assets) return array.
            verbose: Print progress.
        """
        # Use rolling covariance as observation features
        win = 20
        n_t = len(returns)
        features = []
        for t in range(win, n_t):
            window = returns[t - win : t]
            cov = np.cov(window.T)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            # Use upper triangle as feature vector
            features.append(cov[np.triu_indices_from(cov)])
        features = np.array(features)

        self.ops.fit(features, features, verbose=verbose)

    def compress_with_regime(
        self,
        matrix: np.ndarray,
        observation: np.ndarray,
    ) -> Tuple[int, Any]:
        """Compress a matrix using the regime-appropriate pipeline.

        Args:
            matrix: Matrix to compress.
            observation: Feature vector for regime detection.

        Returns:
            (regime_id, CompressedMatrix).
        """
        if self.ops._model is not None:
            regime = self.ops.current_regime(observation)
        else:
            regime = 0

        self._current_regime = int(regime)
        compressed = self.pipelines[regime].compress(matrix, name="current")
        return self._current_regime, compressed

    @property
    def current_regime(self) -> int:
        return self._current_regime


# ============================================================================
# Regime-conditioned tensor prediction
# ============================================================================

def regime_weighted_prediction(
    regime_probs: np.ndarray,
    regime_predictions: List[np.ndarray],
) -> np.ndarray:
    """Combine regime-conditioned predictions using soft weights.

    Args:
        regime_probs: Probability vector over regimes, shape (n_regimes,).
        regime_predictions: List of prediction arrays, one per regime.

    Returns:
        Weighted mixture prediction.
    """
    n_regimes = len(regime_predictions)
    assert len(regime_probs) == n_regimes

    weights = np.asarray(regime_probs, dtype=np.float64)
    weights /= weights.sum() + 1e-300

    result = None
    for r in range(n_regimes):
        pred = np.asarray(regime_predictions[r], dtype=np.float64)
        if result is None:
            result = weights[r] * pred
        else:
            result = result + weights[r] * pred

    return result.astype(np.float32) if result is not None else np.zeros(1)


def cross_regime_correlation(
    cores_per_regime: Dict[int, List[np.ndarray]],
) -> np.ndarray:
    """Compute pairwise correlation between regime TT models.

    Measures similarity between regimes by comparing their TT core norms.

    Args:
        cores_per_regime: Dict mapping regime -> TT cores.

    Returns:
        Correlation matrix of shape (n_regimes, n_regimes).
    """
    n_regimes = len(cores_per_regime)
    regime_ids = sorted(cores_per_regime.keys())

    # Compute a feature vector per regime (concatenated core norms)
    features = []
    for r in regime_ids:
        cores = cores_per_regime[r]
        feat = np.array([float(np.linalg.norm(c)) for c in cores])
        features.append(feat)

    # Pad to equal length
    max_len = max(len(f) for f in features)
    padded = np.array([np.pad(f, (0, max_len - len(f))) for f in features])

    corr = np.corrcoef(padded)
    return corr


# ============================================================================
# Regime transition analysis
# ============================================================================

class RegimeTransitionAnalyzer:
    """Analyze regime transitions in financial tensor data.

    Computes transition statistics, dwell times, and probability
    of transitions between market regimes.

    Args:
        n_regimes: Number of regimes.
    """

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes

    def compute_transition_matrix(
        self,
        regime_sequence: np.ndarray,
    ) -> np.ndarray:
        """Compute empirical regime transition matrix.

        Args:
            regime_sequence: Integer array of regime labels.

        Returns:
            Row-stochastic transition matrix (n_regimes, n_regimes).
        """
        A = np.zeros((self.n_regimes, self.n_regimes))
        for t in range(len(regime_sequence) - 1):
            i = int(regime_sequence[t])
            j = int(regime_sequence[t + 1])
            if 0 <= i < self.n_regimes and 0 <= j < self.n_regimes:
                A[i, j] += 1

        row_sums = A.sum(axis=1, keepdims=True)
        A /= row_sums + 1e-300
        return A

    def dwell_times(
        self,
        regime_sequence: np.ndarray,
    ) -> Dict[int, List[int]]:
        """Compute dwell times (consecutive steps) in each regime.

        Args:
            regime_sequence: Integer regime sequence.

        Returns:
            Dict mapping regime_id -> list of dwell times.
        """
        dwell: Dict[int, List[int]] = {r: [] for r in range(self.n_regimes)}
        if len(regime_sequence) == 0:
            return dwell

        current = int(regime_sequence[0])
        count = 1
        for t in range(1, len(regime_sequence)):
            r = int(regime_sequence[t])
            if r == current:
                count += 1
            else:
                if 0 <= current < self.n_regimes:
                    dwell[current].append(count)
                current = r
                count = 1
        if 0 <= current < self.n_regimes:
            dwell[current].append(count)

        return dwell

    def mean_dwell_times(
        self,
        regime_sequence: np.ndarray,
    ) -> Dict[int, float]:
        """Compute mean dwell time per regime.

        Args:
            regime_sequence: Integer regime sequence.

        Returns:
            Dict mapping regime_id -> mean dwell time.
        """
        dwells = self.dwell_times(regime_sequence)
        return {
            r: float(np.mean(v)) if v else 0.0
            for r, v in dwells.items()
        }

    def regime_frequency(
        self,
        regime_sequence: np.ndarray,
    ) -> Dict[int, float]:
        """Compute fraction of time spent in each regime.

        Args:
            regime_sequence: Integer regime sequence.

        Returns:
            Dict mapping regime_id -> frequency.
        """
        total = len(regime_sequence)
        if total == 0:
            return {r: 0.0 for r in range(self.n_regimes)}
        freqs = {}
        for r in range(self.n_regimes):
            freqs[r] = float(np.sum(regime_sequence == r)) / total
        return freqs

    def stationary_distribution(
        self,
        regime_sequence: np.ndarray,
    ) -> np.ndarray:
        """Estimate stationary distribution from empirical transitions.

        Finds the leading left eigenvector of the transition matrix.

        Args:
            regime_sequence: Integer regime sequence.

        Returns:
            Stationary distribution array (n_regimes,).
        """
        A = self.compute_transition_matrix(regime_sequence)
        eigenvalues, eigenvectors = np.linalg.eig(A.T)
        # Stationary distribution is eigenvector for eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = np.abs(pi)
        pi /= pi.sum() + 1e-300
        return pi

    def full_analysis(
        self,
        regime_sequence: np.ndarray,
    ) -> Dict[str, Any]:
        """Run full transition analysis.

        Args:
            regime_sequence: Integer regime labels.

        Returns:
            Dict with transition_matrix, dwell_times, frequencies, stationary_dist.
        """
        return {
            "transition_matrix": self.compute_transition_matrix(regime_sequence).tolist(),
            "mean_dwell_times": self.mean_dwell_times(regime_sequence),
            "regime_frequencies": self.regime_frequency(regime_sequence),
            "stationary_distribution": self.stationary_distribution(regime_sequence).tolist(),
            "n_regimes": self.n_regimes,
            "sequence_length": len(regime_sequence),
            "n_transitions": int(np.sum(np.diff(regime_sequence) != 0)),
        }


# ============================================================================
# Regime-conditioned portfolio optimizer
# ============================================================================

class RegimeConditionedPortfolio:
    """Portfolio optimization conditioned on market regime.

    Uses separate covariance estimates per regime to compute
    regime-aware portfolio weights.

    Args:
        n_regimes: Number of regimes.
        risk_aversion: Portfolio risk aversion parameter.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        risk_aversion: float = 1.0,
    ):
        self.n_regimes = n_regimes
        self.risk_aversion = risk_aversion
        self._regime_covs: Dict[int, np.ndarray] = {}
        self._regime_means: Dict[int, np.ndarray] = {}

    def fit_regime_stats(
        self,
        returns: np.ndarray,
        regime_sequence: np.ndarray,
    ) -> None:
        """Fit regime-specific return statistics.

        Args:
            returns: Return matrix (T, n_assets).
            regime_sequence: Regime label per time step.
        """
        for r in range(self.n_regimes):
            mask = regime_sequence == r
            if mask.sum() < 2:
                # Fallback to full sample
                mask = np.ones(len(regime_sequence), dtype=bool)
            regime_rets = returns[mask]
            self._regime_means[r] = regime_rets.mean(axis=0)
            if regime_rets.shape[0] >= 2:
                self._regime_covs[r] = np.cov(regime_rets.T) + 1e-6 * np.eye(regime_rets.shape[1])
            else:
                self._regime_covs[r] = np.eye(returns.shape[1])

    def optimal_weights(
        self,
        regime_probs: np.ndarray,
    ) -> np.ndarray:
        """Compute optimal portfolio weights given regime probabilities.

        Uses mean-variance optimization with regime-blended covariance.

        Args:
            regime_probs: Probability over regimes (n_regimes,).

        Returns:
            Portfolio weights (n_assets,), sums to 1.
        """
        if not self._regime_covs:
            raise RuntimeError("Call fit_regime_stats() first.")

        weights = np.asarray(regime_probs, dtype=np.float64)
        weights /= weights.sum() + 1e-300

        # Blend covariances
        n_assets = list(self._regime_covs.values())[0].shape[0]
        blended_cov = np.zeros((n_assets, n_assets))
        blended_mean = np.zeros(n_assets)

        for r in range(self.n_regimes):
            if r in self._regime_covs:
                blended_cov += weights[r] * self._regime_covs[r]
                blended_mean += weights[r] * self._regime_means.get(r, np.zeros(n_assets))

        # Mean-variance weights: w* = (1/lambda) * Sigma^{-1} * mu
        try:
            cov_inv = np.linalg.inv(blended_cov + 1e-8 * np.eye(n_assets))
            raw_weights = cov_inv @ blended_mean / self.risk_aversion
            # Normalize to sum to 1 (long-only constraint)
            raw_weights = np.maximum(raw_weights, 0)
            total = raw_weights.sum()
            if total > 1e-10:
                raw_weights /= total
            else:
                raw_weights = np.ones(n_assets) / n_assets
        except np.linalg.LinAlgError:
            raw_weights = np.ones(n_assets) / n_assets

        return raw_weights.astype(np.float32)

    def regime_portfolio_summary(
        self,
        regime_probs: np.ndarray,
    ) -> Dict[str, Any]:
        """Get portfolio allocation summary for given regime distribution.

        Args:
            regime_probs: Probability over regimes.

        Returns:
            Dict with weights, expected_return, portfolio_variance, sharpe_proxy.
        """
        w = self.optimal_weights(regime_probs)
        weights_arr = np.asarray(regime_probs, dtype=np.float64)
        weights_arr /= weights_arr.sum() + 1e-300

        n_assets = w.shape[0]
        blended_cov = np.zeros((n_assets, n_assets))
        blended_mean = np.zeros(n_assets)
        for r in range(self.n_regimes):
            if r in self._regime_covs:
                blended_cov += weights_arr[r] * self._regime_covs[r]
                blended_mean += weights_arr[r] * self._regime_means.get(r, np.zeros(n_assets))

        port_return = float(w @ blended_mean)
        port_var = float(w @ blended_cov @ w)
        port_std = math.sqrt(max(port_var, 0))
        sharpe = port_return / (port_std + 1e-8) * math.sqrt(252)  # annualized

        return {
            "weights": w.tolist(),
            "expected_return": port_return,
            "portfolio_variance": port_var,
            "portfolio_std": port_std,
            "sharpe_proxy_annualized": sharpe,
            "dominant_regime": int(np.argmax(regime_probs)),
        }



# ---------------------------------------------------------------------------
# Section: Regime-aware signal utilities
# ---------------------------------------------------------------------------

import numpy as np


def compute_regime_conditional_sharpe(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    n_regimes: int = 3,
) -> dict:
    """
    Compute Sharpe ratio per regime.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    regime_labels : np.ndarray, shape (T,)
    n_regimes : int

    Returns
    -------
    dict mapping regime_id -> dict with mean, std, sharpe per asset.
    """
    result = {}
    for k in range(n_regimes):
        mask = regime_labels == k
        if mask.sum() < 5:
            continue
        r_k = returns[mask]
        mu = r_k.mean(axis=0)
        sigma = r_k.std(axis=0) + 1e-12
        sharpe = mu / sigma * np.sqrt(252)
        result[k] = {
            "mean": mu.astype(np.float32),
            "std": sigma.astype(np.float32),
            "sharpe": sharpe.astype(np.float32),
            "n_obs": int(mask.sum()),
        }
    return result


def regime_stability_score(regime_labels: np.ndarray) -> float:
    """
    Compute regime stability as average run length.

    Higher = more stable regimes.

    Parameters
    ----------
    regime_labels : np.ndarray, shape (T,)

    Returns
    -------
    score : float
    """
    if len(regime_labels) < 2:
        return 1.0
    transitions = np.diff(regime_labels) != 0
    n_transitions = transitions.sum()
    if n_transitions == 0:
        return float(len(regime_labels))
    return float(len(regime_labels)) / (n_transitions + 1.0)


def regime_persistence_matrix(
    regime_labels: np.ndarray,
    n_regimes: int = 3,
) -> np.ndarray:
    """
    Compute empirical regime transition matrix.

    Parameters
    ----------
    regime_labels : np.ndarray, shape (T,)
    n_regimes : int

    Returns
    -------
    P : np.ndarray, shape (n_regimes, n_regimes)
        Row-stochastic transition matrix.
    """
    T = len(regime_labels)
    counts = np.zeros((n_regimes, n_regimes), dtype=np.float64)
    for t in range(T - 1):
        i, j = int(regime_labels[t]), int(regime_labels[t + 1])
        if 0 <= i < n_regimes and 0 <= j < n_regimes:
            counts[i, j] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    P = counts / (row_sums + 1e-12)
    return P.astype(np.float32)


def expected_regime_duration(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Compute expected duration of each regime from transition matrix.

    Parameters
    ----------
    transition_matrix : np.ndarray, shape (K, K)

    Returns
    -------
    durations : np.ndarray, shape (K,)
        Expected number of steps in each regime.
    """
    self_transition = np.diag(transition_matrix)
    return 1.0 / (1.0 - self_transition + 1e-12)


def regime_adjusted_returns(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    risk_on_regimes: list | None = None,
    risk_off_scale: float = 0.5,
) -> np.ndarray:
    """
    Scale returns based on current regime.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    regime_labels : np.ndarray, shape (T,)
    risk_on_regimes : list of int
        Regime IDs where full exposure is taken.
    risk_off_scale : float
        Scale factor applied in non-risk-on regimes.

    Returns
    -------
    adjusted : np.ndarray, shape (T, N)
    """
    if risk_on_regimes is None:
        risk_on_regimes = [0]
    T, N = returns.shape
    scale = np.where(
        np.isin(regime_labels, risk_on_regimes),
        1.0,
        risk_off_scale,
    )
    return (returns * scale[:, None]).astype(np.float32)
