"""
rank_selection.py — Automated rank/bond-dimension discovery for TensorNet (Project AETERNUS).

Provides:
  - Cross-validation rank sweep for MPS/TT models
  - Rank-1 update Bayesian Information Criterion (BIC) scoring
  - Adaptive rank growth during training (rank ratchet)
  - Rank pruning via magnitude thresholding on singular values
  - Hierarchical rank selection for Tucker decomposition
  - Rank profile visualization utilities
  - Optimal rank estimation via elbow detection
  - Variance-explained rank selection
  - Minimum description length (MDL) rank criterion
  - Effective rank metrics
"""

from __future__ import annotations

import math
import functools
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tensor_net.mps import (
    MatrixProductState,
    mps_compress,
    mps_from_dense,
    mps_to_dense,
    mps_norm,
    mps_random,
    mps_bond_entropies,
)
from tensor_net.tt_decomp import (
    TensorTrain,
    tt_svd,
    tt_round,
    tucker_decomp,
    cp_decomp,
)
from tensor_net.tensor_train import tt_to_dense, tt_norm


# ============================================================================
# Constants and configuration
# ============================================================================

DEFAULT_MAX_RANK = 64
DEFAULT_MIN_RANK = 1
DEFAULT_RANK_STEP = 2
DEFAULT_CV_FOLDS = 5
BIC_PENALTY_COEFFICIENT = 1.0
MDL_BITS_PER_PARAM = math.log(2)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class RankSweepResult:
    """Result container for a rank sweep experiment."""
    ranks: List[int]
    train_errors: List[float]
    val_errors: List[float]
    bic_scores: List[float]
    mdl_scores: List[float]
    n_params: List[int]
    optimal_rank_cv: int
    optimal_rank_bic: int
    optimal_rank_mdl: int
    optimal_rank_elbow: int
    singular_value_profiles: Dict[int, np.ndarray]

    def summary(self) -> str:
        lines = [
            "RankSweepResult Summary",
            "=" * 40,
            f"  Ranks tested:       {self.ranks}",
            f"  Optimal (CV):       {self.optimal_rank_cv}",
            f"  Optimal (BIC):      {self.optimal_rank_bic}",
            f"  Optimal (MDL):      {self.optimal_rank_mdl}",
            f"  Optimal (Elbow):    {self.optimal_rank_elbow}",
            f"  Best val error:     {min(self.val_errors):.6f}",
        ]
        return "\n".join(lines)


@dataclass
class AdaptiveRankState:
    """State for adaptive rank growth during training."""
    current_ranks: List[int]
    step: int = 0
    growth_history: List[Tuple[int, List[int]]] = field(default_factory=list)
    prune_history: List[Tuple[int, List[int]]] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    plateau_count: int = 0


@dataclass
class TuckerRankProfile:
    """Hierarchical rank profile for Tucker decomposition."""
    mode_ranks: List[int]
    mode_singular_values: List[np.ndarray]
    mode_variances_explained: List[float]
    total_compression_ratio: float
    reconstruction_error: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_ranks": self.mode_ranks,
            "mode_variances_explained": self.mode_variances_explained,
            "total_compression_ratio": self.total_compression_ratio,
            "reconstruction_error": self.reconstruction_error,
        }


# ============================================================================
# Singular value analysis utilities
# ============================================================================

def compute_singular_value_profile(
    matrix: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute the singular value profile of a 2D matrix.

    Args:
        matrix: 2D array of shape (m, n).
        normalize: If True, normalize by the largest singular value.

    Returns:
        Array of singular values in descending order.
    """
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    if normalize and s[0] > 0:
        s = s / s[0]
    return s


def effective_rank(singular_values: np.ndarray, threshold: float = 0.01) -> int:
    """Compute effective rank as number of singular values above threshold.

    Args:
        singular_values: Array of singular values (should be normalized).
        threshold: Relative threshold below which singular values are ignored.

    Returns:
        Integer effective rank.
    """
    sv = np.asarray(singular_values)
    if sv.max() > 0:
        sv = sv / sv.max()
    return int(np.sum(sv > threshold))


def stable_rank(matrix: np.ndarray) -> float:
    """Compute the stable rank: ||A||_F^2 / ||A||_2^2.

    The stable rank is a continuous relaxation of matrix rank.

    Args:
        matrix: 2D array.

    Returns:
        Stable rank as a float.
    """
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    frob_sq = float(np.sum(matrix ** 2))
    spectral_sq = float(np.linalg.norm(matrix, ord=2) ** 2)
    if spectral_sq < 1e-15:
        return 0.0
    return frob_sq / spectral_sq


def nuclear_norm_rank_estimate(matrix: np.ndarray, penalty: float = 1.0) -> int:
    """Estimate rank via nuclear norm minimization proxy.

    Uses the soft-thresholding interpretation: the rank is the number of
    singular values that survive a penalty-level threshold.

    Args:
        matrix: 2D array.
        penalty: Threshold level.

    Returns:
        Estimated integer rank.
    """
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return int(np.sum(s > penalty))


def elbow_rank(singular_values: np.ndarray) -> int:
    """Detect elbow in singular value decay curve.

    Uses the maximum curvature / second-difference heuristic.

    Args:
        singular_values: 1D array of singular values.

    Returns:
        Rank at the elbow point (1-indexed count).
    """
    sv = np.asarray(singular_values, dtype=float)
    if len(sv) < 3:
        return len(sv)
    # Normalize to [0, 1]
    sv_norm = sv / (sv[0] + 1e-15)
    # Second difference
    d2 = np.diff(np.diff(sv_norm))
    elbow_idx = int(np.argmax(np.abs(d2))) + 2  # +2 because of two diffs
    return max(1, min(elbow_idx, len(sv)))


def variance_explained_rank(
    singular_values: np.ndarray,
    target_variance: float = 0.99,
) -> int:
    """Find minimum rank to explain a given fraction of variance.

    Args:
        singular_values: 1D array of singular values.
        target_variance: Target fraction of total variance to explain.

    Returns:
        Minimum rank to achieve target variance.
    """
    sv = np.asarray(singular_values, dtype=float)
    sv_sq = sv ** 2
    total = sv_sq.sum()
    if total < 1e-15:
        return 1
    cumulative = np.cumsum(sv_sq) / total
    rank_indices = np.where(cumulative >= target_variance)[0]
    if len(rank_indices) == 0:
        return len(sv)
    return int(rank_indices[0]) + 1


# ============================================================================
# BIC / MDL / AIC scoring
# ============================================================================

def count_tt_parameters(
    n_sites: int,
    phys_dims: Sequence[int],
    bond_dims: Sequence[int],
) -> int:
    """Count the number of parameters in a TT decomposition.

    Args:
        n_sites: Number of TT cores.
        phys_dims: Physical dimension per site.
        bond_dims: Bond dimensions; length should be n_sites - 1.

    Returns:
        Total parameter count.
    """
    assert len(phys_dims) == n_sites
    assert len(bond_dims) == n_sites - 1
    ranks = [1] + list(bond_dims) + [1]
    total = 0
    for i in range(n_sites):
        total += ranks[i] * phys_dims[i] * ranks[i + 1]
    return total


def bic_score(
    reconstruction_error: float,
    n_data_points: int,
    n_params: int,
    penalty: float = BIC_PENALTY_COEFFICIENT,
) -> float:
    """Compute Bayesian Information Criterion for a tensor model.

    BIC = n * log(MSE) + k * log(n)

    Args:
        reconstruction_error: Mean squared reconstruction error.
        n_data_points: Number of data points.
        n_params: Number of model parameters.
        penalty: Penalty multiplier (default 1.0 = standard BIC).

    Returns:
        BIC score (lower is better).
    """
    if reconstruction_error < 1e-15:
        reconstruction_error = 1e-15
    log_likelihood_proxy = n_data_points * math.log(reconstruction_error)
    complexity_penalty = penalty * n_params * math.log(max(n_data_points, 2))
    return log_likelihood_proxy + complexity_penalty


def aic_score(
    reconstruction_error: float,
    n_data_points: int,
    n_params: int,
) -> float:
    """Compute Akaike Information Criterion.

    AIC = n * log(MSE) + 2 * k

    Args:
        reconstruction_error: Mean squared reconstruction error.
        n_data_points: Number of data points.
        n_params: Number of model parameters.

    Returns:
        AIC score (lower is better).
    """
    if reconstruction_error < 1e-15:
        reconstruction_error = 1e-15
    return n_data_points * math.log(reconstruction_error) + 2 * n_params


def mdl_score(
    reconstruction_error: float,
    n_data_points: int,
    n_params: int,
) -> float:
    """Compute Minimum Description Length criterion.

    MDL = 0.5 * k * log2(n) + n * log2(e) * MSE

    Args:
        reconstruction_error: Mean squared reconstruction error.
        n_data_points: Number of data points.
        n_params: Number of model parameters.

    Returns:
        MDL score in bits (lower is better).
    """
    param_bits = 0.5 * n_params * math.log2(max(n_data_points, 2))
    data_bits = n_data_points * math.log2(math.e) * max(reconstruction_error, 1e-15)
    return param_bits + data_bits


# ============================================================================
# Cross-validation rank sweep
# ============================================================================

def _split_data_cv(
    data: np.ndarray,
    n_folds: int,
    fold_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split data into train/validation for k-fold CV.

    Args:
        data: Array of shape (n_samples, ...).
        n_folds: Total number of folds.
        fold_idx: Which fold to use as validation.

    Returns:
        (train_data, val_data) tuple.
    """
    n = data.shape[0]
    fold_size = n // n_folds
    start = fold_idx * fold_size
    end = start + fold_size if fold_idx < n_folds - 1 else n
    val_idx = np.arange(start, end)
    train_idx = np.concatenate([np.arange(0, start), np.arange(end, n)])
    return data[train_idx], data[val_idx]


def _fit_tt_rank_and_eval(
    train_data: np.ndarray,
    val_data: np.ndarray,
    target_rank: int,
    n_iter: int = 20,
    rng_seed: int = 0,
) -> Tuple[float, float, int]:
    """Fit a TT decomposition at a given rank and evaluate reconstruction error.

    Args:
        train_data: Training data tensor.
        val_data: Validation data tensor.
        target_rank: Bond dimension to use.
        n_iter: Number of ALS iterations.
        rng_seed: Random seed.

    Returns:
        (train_error, val_error, n_params) tuple.
    """
    rng = np.random.default_rng(rng_seed)

    # Flatten data to matrix for SVD-based rank compression
    orig_shape = train_data.shape
    flat_train = train_data.reshape(-1, train_data.shape[-1]) if train_data.ndim > 2 else train_data
    flat_val = val_data.reshape(-1, val_data.shape[-1]) if val_data.ndim > 2 else val_data

    # Use truncated SVD as proxy for TT compression at given rank
    try:
        U, s, Vt = np.linalg.svd(flat_train, full_matrices=False)
        r = min(target_rank, len(s))
        U_r, s_r, Vt_r = U[:, :r], s[:r], Vt[:r, :]
        recon_train = (U_r * s_r) @ Vt_r
        recon_val = flat_val @ Vt_r.T @ np.diag(1.0 / (s_r + 1e-10)) @ Vt_r

        train_err = float(np.mean((flat_train - recon_train) ** 2))
        val_err = float(np.mean((flat_val - recon_val) ** 2))
        n_params = U_r.size + s_r.size + Vt_r.size
    except Exception:
        train_err = float("inf")
        val_err = float("inf")
        n_params = target_rank * 2

    return train_err, val_err, n_params


def rank_sweep_cv(
    data: np.ndarray,
    ranks: Optional[List[int]] = None,
    n_folds: int = DEFAULT_CV_FOLDS,
    n_iter: int = 20,
    rng_seed: int = 42,
    verbose: bool = False,
) -> RankSweepResult:
    """Perform cross-validation rank sweep over a range of bond dimensions.

    For each candidate rank, fits a TT-like decomposition on training folds
    and evaluates reconstruction error on validation fold. Computes BIC, MDL,
    and elbow-based optimal ranks.

    Args:
        data: Data array of shape (n_samples, *dims). Will be flattened to 2D.
        ranks: List of ranks to sweep. Defaults to powers of 2 up to 64.
        n_folds: Number of CV folds.
        n_iter: ALS iterations per fit.
        rng_seed: Random seed for reproducibility.
        verbose: Whether to print progress.

    Returns:
        RankSweepResult with all metrics and optimal rank recommendations.
    """
    if ranks is None:
        ranks = [1, 2, 4, 8, 16, 32, 64]

    n_samples = data.shape[0]
    flat_data = data.reshape(n_samples, -1)

    train_errors: List[float] = []
    val_errors: List[float] = []
    bic_scores: List[float] = []
    mdl_scores: List[float] = []
    n_params_list: List[int] = []
    sv_profiles: Dict[int, np.ndarray] = {}

    for rank in ranks:
        if verbose:
            print(f"  Sweeping rank={rank} ...", flush=True)

        fold_train_errs = []
        fold_val_errs = []
        fold_n_params = []

        for fold_idx in range(n_folds):
            tr, va = _split_data_cv(flat_data, n_folds, fold_idx)
            te, ve, np_ = _fit_tt_rank_and_eval(tr, va, rank, n_iter, rng_seed + fold_idx)
            fold_train_errs.append(te)
            fold_val_errs.append(ve)
            fold_n_params.append(np_)

        mean_train = float(np.mean(fold_train_errs))
        mean_val = float(np.mean(fold_val_errs))
        mean_n_params = int(np.mean(fold_n_params))

        bic = bic_score(mean_val, n_samples, mean_n_params)
        mdl = mdl_score(mean_val, n_samples, mean_n_params)

        train_errors.append(mean_train)
        val_errors.append(mean_val)
        bic_scores.append(bic)
        mdl_scores.append(mdl)
        n_params_list.append(mean_n_params)

        # Compute singular value profile for this rank's training data
        tr_full, _ = _split_data_cv(flat_data, 2, 0)
        sv_profiles[rank] = compute_singular_value_profile(tr_full, normalize=True)

    # Determine optimal ranks
    optimal_cv = ranks[int(np.argmin(val_errors))]
    optimal_bic = ranks[int(np.argmin(bic_scores))]
    optimal_mdl = ranks[int(np.argmin(mdl_scores))]

    # Elbow from validation error
    val_err_arr = np.array(val_errors)
    elbow_idx = elbow_rank(val_err_arr)
    optimal_elbow = ranks[min(elbow_idx - 1, len(ranks) - 1)]

    return RankSweepResult(
        ranks=ranks,
        train_errors=train_errors,
        val_errors=val_errors,
        bic_scores=bic_scores,
        mdl_scores=mdl_scores,
        n_params=n_params_list,
        optimal_rank_cv=optimal_cv,
        optimal_rank_bic=optimal_bic,
        optimal_rank_mdl=optimal_mdl,
        optimal_rank_elbow=optimal_elbow,
        singular_value_profiles=sv_profiles,
    )


# ============================================================================
# Rank-1 update BIC for incremental rank discovery
# ============================================================================

class RankOneBICSelector:
    """Incremental rank discovery using rank-1 BIC updates.

    Starting from rank 1, greedily adds rank-1 components and accepts
    each addition only if the BIC score improves.

    Usage::

        selector = RankOneBICSelector(data, max_rank=32)
        result = selector.fit()
        print(f"Optimal rank: {result['optimal_rank']}")
    """

    def __init__(
        self,
        data: np.ndarray,
        max_rank: int = DEFAULT_MAX_RANK,
        bic_penalty: float = BIC_PENALTY_COEFFICIENT,
        convergence_tol: float = 1e-4,
        rng_seed: int = 42,
    ):
        self.data = data.reshape(data.shape[0], -1)
        self.max_rank = max_rank
        self.bic_penalty = bic_penalty
        self.convergence_tol = convergence_tol
        self.rng_seed = rng_seed
        self.n_samples, self.n_features = self.data.shape

    def _fit_rank_r(self, r: int) -> Tuple[float, int]:
        """Fit SVD at rank r and return (reconstruction_mse, n_params)."""
        U, s, Vt = np.linalg.svd(self.data, full_matrices=False)
        r = min(r, len(s))
        recon = (U[:, :r] * s[:r]) @ Vt[:r, :]
        mse = float(np.mean((self.data - recon) ** 2))
        n_params = self.n_samples * r + r + r * self.n_features
        return mse, n_params

    def fit(self) -> Dict[str, Any]:
        """Run incremental rank-1 BIC selection.

        Returns:
            Dict with keys: optimal_rank, bic_history, mse_history, accepted_ranks.
        """
        bic_history = []
        mse_history = []
        accepted_ranks = []

        best_bic = float("inf")
        best_rank = 1

        for rank in range(1, self.max_rank + 1):
            mse, n_params = self._fit_rank_r(rank)
            bic = bic_score(mse, self.n_samples, n_params, self.bic_penalty)

            bic_history.append(bic)
            mse_history.append(mse)

            if bic < best_bic - self.convergence_tol:
                best_bic = bic
                best_rank = rank
                accepted_ranks.append(rank)
            elif bic > best_bic + abs(best_bic) * 0.01:
                # BIC is increasing — stop early
                break

        return {
            "optimal_rank": best_rank,
            "best_bic": best_bic,
            "bic_history": bic_history,
            "mse_history": mse_history,
            "accepted_ranks": accepted_ranks,
        }


# ============================================================================
# Adaptive rank growth during training
# ============================================================================

class AdaptiveRankGrowth:
    """Controller for adaptive rank growth/pruning during TT training.

    Monitors training loss and bond-dimension singular value spectra.
    Grows ranks when loss plateaus; prunes ranks when singular values are small.

    Args:
        initial_ranks: Starting bond dimensions.
        max_rank: Maximum allowed bond dimension.
        plateau_patience: Steps before triggering rank growth.
        growth_factor: Multiplicative factor for rank growth (e.g., 1.5 → +50%).
        prune_threshold: Singular value magnitude below which rank is pruned.
        growth_loss_threshold: Minimum relative loss improvement to avoid plateau.
    """

    def __init__(
        self,
        initial_ranks: List[int],
        max_rank: int = DEFAULT_MAX_RANK,
        plateau_patience: int = 50,
        growth_factor: float = 1.5,
        prune_threshold: float = 1e-3,
        growth_loss_threshold: float = 1e-3,
    ):
        self.max_rank = max_rank
        self.plateau_patience = plateau_patience
        self.growth_factor = growth_factor
        self.prune_threshold = prune_threshold
        self.growth_loss_threshold = growth_loss_threshold
        self.state = AdaptiveRankState(current_ranks=list(initial_ranks))

    @property
    def current_ranks(self) -> List[int]:
        return self.state.current_ranks

    def step(
        self,
        loss: float,
        singular_values: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Process one training step and potentially update ranks.

        Args:
            loss: Current training loss.
            singular_values: Optional list of singular value arrays per bond.

        Returns:
            Dict with keys: grew (bool), pruned (bool), new_ranks (List[int]).
        """
        self.state.step += 1
        self.state.loss_history.append(loss)

        grew = False
        pruned = False

        # Check for plateau
        if len(self.state.loss_history) >= self.plateau_patience:
            recent = self.state.loss_history[-self.plateau_patience:]
            rel_improvement = (recent[0] - recent[-1]) / (abs(recent[0]) + 1e-15)
            if rel_improvement < self.growth_loss_threshold:
                self.state.plateau_count += 1
            else:
                self.state.plateau_count = 0

        # Grow ranks if plateau detected
        if self.state.plateau_count >= 3:  # 3 consecutive plateau periods
            new_ranks = [
                min(int(r * self.growth_factor) + 1, self.max_rank)
                for r in self.state.current_ranks
            ]
            if new_ranks != self.state.current_ranks:
                self.state.growth_history.append((self.state.step, new_ranks))
                self.state.current_ranks = new_ranks
                self.state.plateau_count = 0
                grew = True

        # Prune ranks based on singular value magnitudes
        if singular_values is not None:
            new_ranks = []
            for i, (rank, sv) in enumerate(
                zip(self.state.current_ranks, singular_values)
            ):
                sv_arr = np.asarray(sv)
                if len(sv_arr) > 0:
                    sv_norm = sv_arr / (sv_arr.max() + 1e-15)
                    new_rank = max(1, int(np.sum(sv_norm > self.prune_threshold)))
                    new_rank = min(new_rank, rank)
                else:
                    new_rank = rank
                new_ranks.append(new_rank)

            if new_ranks != self.state.current_ranks:
                self.state.prune_history.append((self.state.step, new_ranks))
                self.state.current_ranks = new_ranks
                pruned = True

        return {
            "grew": grew,
            "pruned": pruned,
            "new_ranks": list(self.state.current_ranks),
        }

    def reset_plateau(self) -> None:
        """Manually reset plateau counter."""
        self.state.plateau_count = 0

    def summary(self) -> str:
        lines = [
            "AdaptiveRankGrowth Summary",
            "=" * 40,
            f"  Current ranks:    {self.state.current_ranks}",
            f"  Total steps:      {self.state.step}",
            f"  Growth events:    {len(self.state.growth_history)}",
            f"  Prune events:     {len(self.state.prune_history)}",
            f"  Plateau count:    {self.state.plateau_count}",
        ]
        return "\n".join(lines)


# ============================================================================
# Rank pruning via magnitude thresholding
# ============================================================================

def prune_tt_ranks_magnitude(
    cores: List[np.ndarray],
    threshold: float = 1e-3,
    min_rank: int = 1,
    normalize_threshold: bool = True,
) -> Tuple[List[np.ndarray], List[int]]:
    """Prune TT ranks by removing small singular value components.

    Performs SVD at each bond, truncates singular values below threshold,
    and reabsorbs into neighboring cores.

    Args:
        cores: List of TT cores, each of shape (r_left, d, r_right).
        threshold: Threshold for singular value pruning.
        min_rank: Minimum bond dimension to keep.
        normalize_threshold: If True, threshold is relative to max singular value.

    Returns:
        (pruned_cores, new_ranks) where new_ranks is the list of bond dimensions.
    """
    if len(cores) == 0:
        return cores, []

    pruned = [np.array(c) for c in cores]
    new_ranks = []

    for bond_idx in range(len(pruned) - 1):
        left = pruned[bond_idx]         # (r_l, d_l, r_r)
        right = pruned[bond_idx + 1]    # (r_r, d_r, r_rr)

        r_l, d_l, r_r = left.shape
        r_rr = right.shape[2]
        d_r = right.shape[1]

        # Reshape left for SVD: (r_l * d_l, r_r)
        mat = left.reshape(r_l * d_l, r_r)
        U, s, Vt = np.linalg.svd(mat, full_matrices=False)

        # Determine threshold
        if normalize_threshold:
            thr = threshold * (s.max() + 1e-15)
        else:
            thr = threshold

        mask = s > thr
        n_keep = max(min_rank, int(mask.sum()))
        n_keep = min(n_keep, len(s))

        # Truncate
        U_t = U[:, :n_keep]
        s_t = s[:n_keep]
        Vt_t = Vt[:n_keep, :]

        # New left core
        pruned[bond_idx] = (U_t * s_t).reshape(r_l, d_l, n_keep)

        # Absorb Vt into right core
        right_mat = right.reshape(r_r, d_r * r_rr)
        new_right_mat = Vt_t @ right_mat
        pruned[bond_idx + 1] = new_right_mat.reshape(n_keep, d_r, r_rr)

        new_ranks.append(n_keep)

    return pruned, new_ranks


def prune_mps_bonds(
    mps: "MatrixProductState",
    threshold: float = 1e-3,
    min_rank: int = 1,
) -> "MatrixProductState":
    """Prune MPS bond dimensions by magnitude thresholding.

    Args:
        mps: Input MatrixProductState.
        threshold: Relative singular value threshold.
        min_rank: Minimum bond dimension.

    Returns:
        New MatrixProductState with pruned bonds.
    """
    cores = [np.array(t) for t in mps.tensors]
    pruned_cores, _ = prune_tt_ranks_magnitude(cores, threshold, min_rank)
    new_tensors = [jnp.array(c) for c in pruned_cores]
    return MatrixProductState(tensors=new_tensors, phys_dims=mps.phys_dims)


# ============================================================================
# Hierarchical rank selection for Tucker decomposition
# ============================================================================

def tucker_rank_per_mode(
    tensor: np.ndarray,
    target_variance: float = 0.99,
    max_rank_per_mode: Optional[int] = None,
) -> TuckerRankProfile:
    """Determine optimal Tucker rank for each mode via variance-explained criterion.

    For each mode, unfolds the tensor, computes SVD, and finds the minimum rank
    to explain the target fraction of variance.

    Args:
        tensor: N-dimensional array.
        target_variance: Fraction of variance to explain per mode.
        max_rank_per_mode: Optional cap on rank per mode.

    Returns:
        TuckerRankProfile with per-mode ranks and diagnostics.
    """
    n_modes = tensor.ndim
    mode_ranks = []
    mode_sv = []
    mode_var_explained = []

    full_frob = float(np.sum(tensor ** 2))

    for mode in range(n_modes):
        # Mode-n unfolding
        unfolded = _mode_unfold(tensor, mode)  # (d_mode, product_of_rest)
        _, s, _ = np.linalg.svd(unfolded, full_matrices=False)

        # Variance explained
        sv_sq = s ** 2
        total_mode = sv_sq.sum()
        if total_mode < 1e-15:
            r = 1
            var_exp = 0.0
        else:
            r = variance_explained_rank(s, target_variance)
            var_exp = float(sv_sq[:r].sum() / total_mode)

        if max_rank_per_mode is not None:
            r = min(r, max_rank_per_mode)
        r = max(1, r)

        mode_ranks.append(r)
        mode_sv.append(s)
        mode_var_explained.append(var_exp)

    # Estimate compression ratio
    orig_size = tensor.size
    d = tensor.shape
    core_size = int(np.prod(mode_ranks))
    factor_sizes = sum(d[i] * mode_ranks[i] for i in range(n_modes))
    compressed_size = core_size + factor_sizes
    compression_ratio = float(orig_size / max(1, compressed_size))

    # Estimate reconstruction error via mode-wise truncation
    recon_error = _estimate_tucker_error(tensor, mode_ranks)

    return TuckerRankProfile(
        mode_ranks=mode_ranks,
        mode_singular_values=mode_sv,
        mode_variances_explained=mode_var_explained,
        total_compression_ratio=compression_ratio,
        reconstruction_error=recon_error,
    )


def _mode_unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Unfold tensor along a given mode.

    Args:
        tensor: N-dimensional array.
        mode: Mode index to unfold along.

    Returns:
        2D array of shape (d_mode, prod_of_other_dims).
    """
    return np.reshape(
        np.moveaxis(tensor, mode, 0),
        (tensor.shape[mode], -1),
    )


def _estimate_tucker_error(
    tensor: np.ndarray,
    mode_ranks: List[int],
) -> float:
    """Estimate Tucker reconstruction error via sequential mode truncation.

    Args:
        tensor: Original N-dimensional array.
        mode_ranks: Rank per mode.

    Returns:
        Frobenius reconstruction error (normalized by original norm).
    """
    approx = np.array(tensor)
    for mode, r in enumerate(mode_ranks):
        unfolded = _mode_unfold(approx, mode)
        U, s, _ = np.linalg.svd(unfolded, full_matrices=False)
        r = min(r, U.shape[1])
        proj = U[:, :r] @ U[:, :r].T
        refolded = proj @ unfolded
        shape = list(approx.shape)
        approx = np.moveaxis(
            refolded.reshape([shape[mode]] + [shape[i] for i in range(len(shape)) if i != mode]),
            0,
            mode,
        )
    orig_norm = float(np.linalg.norm(tensor))
    err = float(np.linalg.norm(tensor - approx))
    return err / (orig_norm + 1e-15)


def hierarchical_tucker_rank_selection(
    tensor: np.ndarray,
    n_levels: int = 2,
    target_variance_per_level: float = 0.99,
) -> Dict[str, Any]:
    """Hierarchical Tucker rank selection across multiple compression levels.

    Applies Tucker rank selection iteratively, compressing the core tensor
    at each level with decreasing variance thresholds.

    Args:
        tensor: Original N-dimensional array.
        n_levels: Number of hierarchical compression levels.
        target_variance_per_level: Variance explained per level per mode.

    Returns:
        Dict with per-level TuckerRankProfile objects and cumulative metrics.
    """
    results = {}
    current_tensor = np.array(tensor)
    cumulative_compression = 1.0

    for level in range(n_levels):
        profile = tucker_rank_per_mode(
            current_tensor,
            target_variance=target_variance_per_level,
        )
        results[f"level_{level}"] = profile
        cumulative_compression *= profile.total_compression_ratio

        # Compress current tensor to its core for next level
        current_tensor = _tucker_core(current_tensor, profile.mode_ranks)

    results["cumulative_compression_ratio"] = cumulative_compression
    results["final_core_shape"] = current_tensor.shape
    return results


def _tucker_core(tensor: np.ndarray, mode_ranks: List[int]) -> np.ndarray:
    """Compute Tucker core tensor by projecting each mode onto top singular vectors.

    Args:
        tensor: N-dimensional array.
        mode_ranks: Rank per mode.

    Returns:
        Core tensor of shape mode_ranks.
    """
    core = np.array(tensor)
    for mode, r in enumerate(mode_ranks):
        unfolded = _mode_unfold(core, mode)
        U, _, _ = np.linalg.svd(unfolded, full_matrices=False)
        r = min(r, U.shape[1])
        projected = U[:, :r].T @ unfolded
        shape = list(core.shape)
        shape[mode] = r
        core = np.moveaxis(
            projected.reshape([r] + [core.shape[i] for i in range(core.ndim) if i != mode]),
            0,
            mode,
        )
    return core


# ============================================================================
# Rank profile visualization
# ============================================================================

def rank_profile_to_dict(
    result: RankSweepResult,
) -> Dict[str, Any]:
    """Convert a RankSweepResult to a plain dict for JSON serialization.

    Args:
        result: RankSweepResult from rank_sweep_cv.

    Returns:
        Dictionary suitable for JSON/YAML serialization.
    """
    return {
        "ranks": result.ranks,
        "train_errors": result.train_errors,
        "val_errors": result.val_errors,
        "bic_scores": result.bic_scores,
        "mdl_scores": result.mdl_scores,
        "n_params": result.n_params,
        "optimal_rank_cv": result.optimal_rank_cv,
        "optimal_rank_bic": result.optimal_rank_bic,
        "optimal_rank_mdl": result.optimal_rank_mdl,
        "optimal_rank_elbow": result.optimal_rank_elbow,
    }


def plot_rank_sweep(
    result: RankSweepResult,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Visualize rank sweep results.

    Produces a 2x2 figure:
    - Top-left: Train/val reconstruction error vs rank
    - Top-right: BIC and MDL vs rank
    - Bottom-left: Number of parameters vs rank
    - Bottom-right: Singular value profiles for selected ranks

    Args:
        result: RankSweepResult to visualize.
        save_path: If provided, save figure to this path.
        show: If True, call plt.show().
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available; skipping rank_sweep plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Rank Sweep Analysis", fontsize=14)

    # --- Train/Val errors ---
    ax = axes[0, 0]
    ax.semilogy(result.ranks, result.train_errors, "b-o", label="Train MSE")
    ax.semilogy(result.ranks, result.val_errors, "r-o", label="Val MSE")
    ax.axvline(result.optimal_rank_cv, color="r", linestyle="--", alpha=0.7, label=f"CV opt={result.optimal_rank_cv}")
    ax.axvline(result.optimal_rank_elbow, color="g", linestyle="--", alpha=0.7, label=f"Elbow={result.optimal_rank_elbow}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction Error vs Rank")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- BIC / MDL ---
    ax = axes[0, 1]
    ax.plot(result.ranks, result.bic_scores, "b-o", label="BIC")
    ax.plot(result.ranks, result.mdl_scores, "r-o", label="MDL")
    ax.axvline(result.optimal_rank_bic, color="b", linestyle="--", alpha=0.7, label=f"BIC opt={result.optimal_rank_bic}")
    ax.axvline(result.optimal_rank_mdl, color="r", linestyle="--", alpha=0.7, label=f"MDL opt={result.optimal_rank_mdl}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Score")
    ax.set_title("Information Criteria vs Rank")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- N params ---
    ax = axes[1, 0]
    ax.semilogy(result.ranks, result.n_params, "g-o")
    ax.set_xlabel("Rank")
    ax.set_ylabel("# Parameters")
    ax.set_title("Model Size vs Rank")
    ax.grid(True, alpha=0.3)

    # --- Singular value profiles ---
    ax = axes[1, 1]
    selected_ranks = result.ranks[::max(1, len(result.ranks) // 4)]
    for r in selected_ranks:
        if r in result.singular_value_profiles:
            sv = result.singular_value_profiles[r]
            n_show = min(50, len(sv))
            ax.semilogy(np.arange(1, n_show + 1), sv[:n_show], label=f"r={r}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular Value (normalized)")
    ax.set_title("Singular Value Profiles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_tucker_rank_profile(
    profile: TuckerRankProfile,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Visualize Tucker rank profile per mode.

    Args:
        profile: TuckerRankProfile from tucker_rank_per_mode.
        save_path: If provided, save figure to this path.
        show: If True, call plt.show().
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available; skipping Tucker profile plot.")
        return

    n_modes = len(profile.mode_ranks)
    fig, axes = plt.subplots(1, n_modes, figsize=(4 * n_modes, 4))
    if n_modes == 1:
        axes = [axes]

    for mode_idx, (ax, rank, sv, var_exp) in enumerate(
        zip(axes, profile.mode_ranks, profile.mode_singular_values, profile.mode_variances_explained)
    ):
        n_show = min(50, len(sv))
        ax.semilogy(np.arange(1, n_show + 1), sv[:n_show] / (sv[0] + 1e-15), "b-")
        ax.axvline(rank, color="r", linestyle="--", label=f"r={rank} ({var_exp:.1%})")
        ax.set_xlabel("Component index")
        ax.set_ylabel("Normalized singular value")
        ax.set_title(f"Mode {mode_idx}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Tucker Rank Profile — Compression {profile.total_compression_ratio:.1f}x, "
        f"Error {profile.reconstruction_error:.4f}",
        fontsize=12,
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================================
# Rank selection via information-theoretic methods
# ============================================================================

class InformationTheoreticRankSelector:
    """Comprehensive rank selector combining BIC, MDL, AIC, and variance criteria.

    Provides a unified interface for rank selection with multiple criteria
    and consensus voting.

    Args:
        data: Input data array.
        max_rank: Maximum rank to consider.
        bic_penalty: BIC penalty coefficient.
        variance_target: Target variance for variance-explained criterion.
        rng_seed: Random seed.
    """

    def __init__(
        self,
        data: np.ndarray,
        max_rank: int = DEFAULT_MAX_RANK,
        bic_penalty: float = BIC_PENALTY_COEFFICIENT,
        variance_target: float = 0.99,
        rng_seed: int = 42,
    ):
        self.data = data.reshape(data.shape[0], -1)
        self.max_rank = max_rank
        self.bic_penalty = bic_penalty
        self.variance_target = variance_target
        self.rng_seed = rng_seed
        self.n_samples, self.n_features = self.data.shape
        self._svd_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def _get_svd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._svd_cache is None:
            self._svd_cache = np.linalg.svd(self.data, full_matrices=False)
        return self._svd_cache

    def rank_by_bic(self) -> int:
        """Select rank by BIC criterion."""
        U, s, Vt = self._get_svd()
        best_bic = float("inf")
        best_r = 1
        for r in range(1, min(self.max_rank + 1, len(s) + 1)):
            recon = (U[:, :r] * s[:r]) @ Vt[:r, :]
            mse = float(np.mean((self.data - recon) ** 2))
            n_params = self.n_samples * r + r + r * self.n_features
            b = bic_score(mse, self.n_samples, n_params, self.bic_penalty)
            if b < best_bic:
                best_bic = b
                best_r = r
        return best_r

    def rank_by_aic(self) -> int:
        """Select rank by AIC criterion."""
        U, s, Vt = self._get_svd()
        best_aic = float("inf")
        best_r = 1
        for r in range(1, min(self.max_rank + 1, len(s) + 1)):
            recon = (U[:, :r] * s[:r]) @ Vt[:r, :]
            mse = float(np.mean((self.data - recon) ** 2))
            n_params = self.n_samples * r + r + r * self.n_features
            a = aic_score(mse, self.n_samples, n_params)
            if a < best_aic:
                best_aic = a
                best_r = r
        return best_r

    def rank_by_mdl(self) -> int:
        """Select rank by MDL criterion."""
        U, s, Vt = self._get_svd()
        best_mdl = float("inf")
        best_r = 1
        for r in range(1, min(self.max_rank + 1, len(s) + 1)):
            recon = (U[:, :r] * s[:r]) @ Vt[:r, :]
            mse = float(np.mean((self.data - recon) ** 2))
            n_params = self.n_samples * r + r + r * self.n_features
            m = mdl_score(mse, self.n_samples, n_params)
            if m < best_mdl:
                best_mdl = m
                best_r = r
        return best_r

    def rank_by_variance(self) -> int:
        """Select rank by variance-explained criterion."""
        _, s, _ = self._get_svd()
        return variance_explained_rank(s, self.variance_target)

    def rank_by_elbow(self) -> int:
        """Select rank by elbow method."""
        _, s, _ = self._get_svd()
        return elbow_rank(s)

    def rank_by_effective(self, threshold: float = 0.01) -> int:
        """Select rank by effective rank method."""
        _, s, _ = self._get_svd()
        return effective_rank(s, threshold)

    def consensus_rank(self, method: str = "median") -> int:
        """Get consensus rank across all criteria.

        Args:
            method: Aggregation method — "median", "min", "max", or "mean".

        Returns:
            Consensus rank.
        """
        ranks = [
            self.rank_by_bic(),
            self.rank_by_aic(),
            self.rank_by_mdl(),
            self.rank_by_variance(),
            self.rank_by_elbow(),
        ]
        if method == "median":
            return int(np.median(ranks))
        elif method == "min":
            return int(np.min(ranks))
        elif method == "max":
            return int(np.max(ranks))
        elif method == "mean":
            return int(np.mean(ranks))
        else:
            raise ValueError(f"Unknown method: {method}")

    def full_report(self) -> Dict[str, Any]:
        """Generate full rank selection report.

        Returns:
            Dict with all criterion results and consensus.
        """
        return {
            "rank_bic": self.rank_by_bic(),
            "rank_aic": self.rank_by_aic(),
            "rank_mdl": self.rank_by_mdl(),
            "rank_variance": self.rank_by_variance(),
            "rank_elbow": self.rank_by_elbow(),
            "rank_effective": self.rank_by_effective(),
            "consensus_median": self.consensus_rank("median"),
            "consensus_min": self.consensus_rank("min"),
            "consensus_max": self.consensus_rank("max"),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "max_rank": self.max_rank,
        }


# ============================================================================
# Automatic rank selection for TT format
# ============================================================================

def auto_rank_tt(
    tensor: np.ndarray,
    method: str = "bic",
    max_rank: int = DEFAULT_MAX_RANK,
    target_variance: float = 0.99,
    threshold: float = 1e-3,
    verbose: bool = False,
) -> List[int]:
    """Automatically select TT ranks for an N-dimensional tensor.

    Unfolds the tensor left-to-right and selects the optimal rank at each
    unfolding using the specified information criterion.

    Args:
        tensor: N-dimensional array to compress.
        method: Rank selection method — "bic", "aic", "mdl", "variance", "elbow", "threshold".
        max_rank: Maximum rank per bond.
        target_variance: Variance threshold (only for "variance" method).
        threshold: Singular value threshold (only for "threshold" method).
        verbose: Print per-bond rank selections.

    Returns:
        List of bond dimensions of length n_modes - 1.
    """
    n_modes = tensor.ndim
    if n_modes < 2:
        return []

    ranks = []
    current = np.array(tensor)

    for mode in range(n_modes - 1):
        d_left = current.shape[0]
        d_rest = int(np.prod(current.shape[1:]))
        mat = current.reshape(d_left, d_rest)

        _, s, Vt = np.linalg.svd(mat, full_matrices=False)

        if method == "threshold":
            r = max(1, int(np.sum(s / (s[0] + 1e-15) > threshold)))
        elif method == "variance":
            r = variance_explained_rank(s, target_variance)
        elif method == "elbow":
            r = elbow_rank(s)
        elif method == "bic":
            selector = InformationTheoreticRankSelector(mat, max_rank=max_rank)
            r = selector.rank_by_bic()
        elif method == "aic":
            selector = InformationTheoreticRankSelector(mat, max_rank=max_rank)
            r = selector.rank_by_aic()
        elif method == "mdl":
            selector = InformationTheoreticRankSelector(mat, max_rank=max_rank)
            r = selector.rank_by_mdl()
        else:
            raise ValueError(f"Unknown method: {method}")

        r = min(r, max_rank, d_left, d_rest)
        r = max(1, r)
        ranks.append(r)

        if verbose:
            print(f"  Bond {mode}: rank={r} (d_left={d_left}, d_rest={d_rest})")

        # Advance: compress left side into rank-r representation
        U_r = np.linalg.svd(mat, full_matrices=False)[0][:, :r]
        s_r = s[:r]
        Vt_r = Vt[:r, :]
        compressed_left = (U_r * s_r).reshape(d_left, r)
        # New current = Vt_r reshaped to (r, *remaining_dims)
        remaining_shape = (r,) + current.shape[1:]
        current = (Vt_r @ current.reshape(d_left, -1).T).T.reshape(remaining_shape)

    return ranks


# ============================================================================
# Rank selection integration with financial tensors
# ============================================================================

def select_correlation_tensor_rank(
    corr_matrix: np.ndarray,
    n_assets: int,
    n_time_steps: int,
    method: str = "bic",
    max_rank: int = 32,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Select optimal TT rank for a financial correlation tensor.

    Given a correlation matrix or tensor, determines the optimal compression
    rank for TT decomposition.

    Args:
        corr_matrix: Correlation matrix/tensor to compress.
        n_assets: Number of assets.
        n_time_steps: Number of time steps.
        method: Rank selection method.
        max_rank: Maximum rank.
        verbose: Print diagnostics.

    Returns:
        Dict with recommended_rank, estimated_compression_ratio, diagnostics.
    """
    # Reshape to (n_assets, -1) if needed
    if corr_matrix.ndim == 1:
        side = int(math.sqrt(len(corr_matrix)))
        corr_matrix = corr_matrix.reshape(side, side)

    flat = corr_matrix.reshape(n_assets, -1) if corr_matrix.ndim > 1 else corr_matrix

    selector = InformationTheoreticRankSelector(flat, max_rank=max_rank)
    report = selector.full_report()
    recommended = report["consensus_median"]

    # Estimate compression
    orig_params = corr_matrix.size
    compressed_params = n_assets * recommended + recommended + recommended * flat.shape[1]
    ratio = orig_params / max(1, compressed_params)

    if verbose:
        print(f"Correlation tensor rank selection:")
        for k, v in report.items():
            print(f"  {k}: {v}")
        print(f"  Recommended rank: {recommended}")
        print(f"  Compression ratio: {ratio:.2f}x")

    return {
        "recommended_rank": recommended,
        "estimated_compression_ratio": ratio,
        "diagnostics": report,
    }


# ============================================================================
# Rank stability analysis
# ============================================================================

def rank_stability_bootstrap(
    data: np.ndarray,
    n_bootstrap: int = 20,
    method: str = "bic",
    max_rank: int = DEFAULT_MAX_RANK,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """Assess stability of rank selection via bootstrap resampling.

    Repeatedly samples the data with replacement and estimates the optimal
    rank on each bootstrap sample, reporting the distribution.

    Args:
        data: Input data array of shape (n_samples, ...).
        n_bootstrap: Number of bootstrap iterations.
        method: Rank selection method.
        max_rank: Maximum rank.
        rng_seed: Random seed.

    Returns:
        Dict with mean_rank, std_rank, rank_histogram, confidence_interval.
    """
    rng = np.random.default_rng(rng_seed)
    n = data.shape[0]
    flat_data = data.reshape(n, -1)

    bootstrap_ranks = []

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = flat_data[idx]
        selector = InformationTheoreticRankSelector(sample, max_rank=max_rank)

        if method == "bic":
            r = selector.rank_by_bic()
        elif method == "aic":
            r = selector.rank_by_aic()
        elif method == "mdl":
            r = selector.rank_by_mdl()
        elif method == "variance":
            r = selector.rank_by_variance()
        elif method == "elbow":
            r = selector.rank_by_elbow()
        else:
            r = selector.consensus_rank()

        bootstrap_ranks.append(r)

    ranks_arr = np.array(bootstrap_ranks)
    unique, counts = np.unique(ranks_arr, return_counts=True)

    return {
        "mean_rank": float(ranks_arr.mean()),
        "std_rank": float(ranks_arr.std()),
        "median_rank": int(np.median(ranks_arr)),
        "min_rank": int(ranks_arr.min()),
        "max_rank": int(ranks_arr.max()),
        "ci_95": (float(np.percentile(ranks_arr, 2.5)), float(np.percentile(ranks_arr, 97.5))),
        "rank_histogram": {int(k): int(v) for k, v in zip(unique, counts)},
        "bootstrap_samples": bootstrap_ranks,
    }


# ============================================================================
# Rank selection for multi-asset time series
# ============================================================================

def select_window_ranks(
    returns: np.ndarray,
    window_size: int,
    stride: int = 1,
    method: str = "bic",
    max_rank: int = 16,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Select TT ranks for rolling-window financial tensor.

    Constructs a rolling window tensor from return data and determines
    the optimal rank at each window via the chosen criterion.

    Args:
        returns: Return array of shape (n_time, n_assets).
        window_size: Rolling window length.
        stride: Stride between windows.
        method: Rank selection method.
        max_rank: Maximum allowed rank.
        verbose: Print progress.

    Returns:
        Dict with per-window ranks, mean_rank, std_rank, rank_timeline.
    """
    n_time, n_assets = returns.shape
    n_windows = (n_time - window_size) // stride + 1

    window_ranks = []
    for w_idx in range(n_windows):
        start = w_idx * stride
        end = start + window_size
        window_data = returns[start:end, :]  # (window_size, n_assets)

        selector = InformationTheoreticRankSelector(window_data, max_rank=max_rank)

        if method == "bic":
            r = selector.rank_by_bic()
        elif method == "variance":
            r = selector.rank_by_variance()
        elif method == "elbow":
            r = selector.rank_by_elbow()
        else:
            r = selector.consensus_rank()

        window_ranks.append(r)

        if verbose and w_idx % 50 == 0:
            print(f"  Window {w_idx}/{n_windows}: rank={r}")

    window_ranks_arr = np.array(window_ranks)

    return {
        "per_window_ranks": window_ranks,
        "mean_rank": float(window_ranks_arr.mean()),
        "std_rank": float(window_ranks_arr.std()),
        "min_rank": int(window_ranks_arr.min()),
        "max_rank": int(window_ranks_arr.max()),
        "rank_timeline": window_ranks,
        "n_windows": n_windows,
        "window_size": window_size,
        "stride": stride,
    }


# ============================================================================
# JAX-compatible rank selection utilities
# ============================================================================

@jax.jit
def jax_effective_rank(
    matrix: jnp.ndarray,
    threshold: float = 0.01,
) -> jnp.ndarray:
    """JAX-JIT-compatible effective rank computation.

    Args:
        matrix: 2D JAX array.
        threshold: Relative threshold.

    Returns:
        Scalar effective rank.
    """
    _, s, _ = jnp.linalg.svd(matrix, full_matrices=False)
    s_norm = s / (s[0] + 1e-15)
    return jnp.sum(s_norm > threshold).astype(jnp.float32)


@jax.jit
def jax_stable_rank(matrix: jnp.ndarray) -> jnp.ndarray:
    """JAX-JIT-compatible stable rank.

    Args:
        matrix: 2D JAX array.

    Returns:
        Scalar stable rank.
    """
    frob_sq = jnp.sum(matrix ** 2)
    spectral = jnp.linalg.norm(matrix, ord=2)
    return frob_sq / (spectral ** 2 + 1e-15)


def jax_rank_regularizer(
    cores: List[jnp.ndarray],
    target_rank: float,
    weight: float = 0.01,
) -> jnp.ndarray:
    """Differentiable rank regularizer for TT training.

    Penalizes stable rank deviation from a target, encouraging
    the model to maintain a specific effective rank.

    Args:
        cores: List of TT core tensors.
        target_rank: Desired stable rank.
        weight: Regularization weight.

    Returns:
        Scalar regularization loss.
    """
    total_reg = jnp.zeros(())
    for core in cores:
        r_l, d, r_r = core.shape
        mat = core.reshape(r_l * d, r_r)
        sr = jax_stable_rank(mat)
        total_reg = total_reg + (sr - target_rank) ** 2
    return weight * total_reg


# ============================================================================
# Rank sweep over TT decomposition quality metrics
# ============================================================================

def tt_reconstruction_quality(
    tensor: np.ndarray,
    rank: int,
    n_iter: int = 10,
) -> Dict[str, float]:
    """Evaluate TT decomposition quality at a given rank.

    Computes reconstruction error, compression ratio, and related metrics.

    Args:
        tensor: N-dimensional array to decompose.
        rank: Bond dimension.
        n_iter: HOSVD iterations.

    Returns:
        Dict with reconstruction_error, compression_ratio, stable_rank, n_params.
    """
    orig_shape = tensor.shape
    orig_size = tensor.size

    # Flatten to matrix for SVD proxy
    mat = tensor.reshape(orig_shape[0], -1)
    n, m = mat.shape
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    r = min(rank, len(s))
    recon_mat = (U[:, :r] * s[:r]) @ Vt[:r, :]
    recon = recon_mat.reshape(orig_shape)

    orig_norm = float(np.linalg.norm(tensor))
    err = float(np.linalg.norm(tensor - recon)) / (orig_norm + 1e-15)
    n_params = n * r + r + r * m
    ratio = orig_size / max(1, n_params)
    sr = stable_rank(mat)

    return {
        "reconstruction_error": err,
        "compression_ratio": ratio,
        "stable_rank": sr,
        "n_params": n_params,
        "rank_used": r,
        "rank_requested": rank,
        "variance_explained": float(np.sum(s[:r] ** 2) / (np.sum(s ** 2) + 1e-15)),
    }


def full_rank_quality_sweep(
    tensor: np.ndarray,
    ranks: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[int, Dict[str, float]]:
    """Sweep over ranks and compute full quality metrics for each.

    Args:
        tensor: N-dimensional input array.
        ranks: List of ranks. Defaults to [1, 2, 4, 8, 16, 32, 64].
        verbose: Print progress.

    Returns:
        Dict mapping rank -> quality metrics dict.
    """
    if ranks is None:
        ranks = [1, 2, 4, 8, 16, 32, 64]

    results = {}
    for rank in ranks:
        if verbose:
            print(f"  Evaluating rank={rank} ...")
        results[rank] = tt_reconstruction_quality(tensor, rank)

    return results


# ============================================================================
# Rank selection utilities for training pipelines
# ============================================================================

class RankScheduler:
    """Learning-rate-style scheduler for rank growth during training.

    Supports step, cosine, and linear rank growth schedules, useful when
    a curriculum-style approach to rank is desired.

    Args:
        initial_rank: Starting rank.
        final_rank: Target rank at end of schedule.
        total_steps: Total training steps.
        schedule: One of "step", "linear", "cosine".
        step_milestones: Steps at which to increase rank (for "step" schedule).
    """

    def __init__(
        self,
        initial_rank: int,
        final_rank: int,
        total_steps: int,
        schedule: str = "linear",
        step_milestones: Optional[List[int]] = None,
    ):
        self.initial_rank = initial_rank
        self.final_rank = final_rank
        self.total_steps = total_steps
        self.schedule = schedule
        self.step_milestones = step_milestones or []
        self._step = 0

    def step(self) -> int:
        """Advance the scheduler by one step and return current rank.

        Returns:
            Current target rank.
        """
        self._step += 1
        return self.get_rank(self._step)

    def get_rank(self, step: int) -> int:
        """Get the target rank at a given step.

        Args:
            step: Training step.

        Returns:
            Target rank at this step.
        """
        progress = min(1.0, step / max(1, self.total_steps))

        if self.schedule == "linear":
            rank = self.initial_rank + (self.final_rank - self.initial_rank) * progress
        elif self.schedule == "cosine":
            rank = self.initial_rank + (self.final_rank - self.initial_rank) * (
                1 - math.cos(math.pi * progress)
            ) / 2
        elif self.schedule == "step":
            n_passed = sum(1 for m in self.step_milestones if step >= m)
            total_milestones = len(self.step_milestones)
            if total_milestones > 0:
                rank = self.initial_rank + (self.final_rank - self.initial_rank) * n_passed / total_milestones
            else:
                rank = self.final_rank if progress >= 1.0 else self.initial_rank
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return max(self.initial_rank, min(self.final_rank, round(rank)))

    def reset(self) -> None:
        """Reset the scheduler to step 0."""
        self._step = 0


# ---------------------------------------------------------------------------
# Section: Adaptive rank selection with cross-validation
# ---------------------------------------------------------------------------

import numpy as np
import warnings


def kfold_rank_selection(
    data: np.ndarray,
    rank_candidates: list,
    n_folds: int = 5,
    method: str = "tt",
    random_seed: int = 42,
) -> dict:
    """
    K-fold cross-validation based rank selection.

    For each rank candidate, fits a tensor decomposition on K-1 folds and
    evaluates reconstruction error on the held-out fold.

    Parameters
    ----------
    data : np.ndarray, shape (T, N) or (T, N, M)
        Data tensor for decomposition.
    rank_candidates : list of int
        Candidate ranks to evaluate.
    n_folds : int
        Number of CV folds.
    method : str
        Decomposition method: "tt" | "tucker" | "svd"
    random_seed : int

    Returns
    -------
    dict with keys:
    * ``"best_rank"`` : int
    * ``"cv_errors"`` : dict rank -> list of fold errors
    * ``"mean_cv_error"`` : dict rank -> float
    * ``"std_cv_error"`` : dict rank -> float
    """
    rng = np.random.default_rng(random_seed)
    T = data.shape[0]
    indices = np.arange(T)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    cv_errors = {r: [] for r in rank_candidates}

    for fold_idx in range(n_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])
        train_data = data[train_idx]
        val_data = data[val_idx]

        for rank in rank_candidates:
            try:
                if method == "svd" or data.ndim == 2:
                    # Truncated SVD reconstruction error
                    U, s, Vt = np.linalg.svd(
                        train_data.reshape(len(train_idx), -1), full_matrices=False
                    )
                    r = min(rank, len(s))
                    recon = U[:r, :r] @ np.diag(s[:r]) @ Vt[:r, :]
                    val_flat = val_data.reshape(len(val_idx), -1)
                    # Project val onto train basis
                    # Use Vt from training
                    val_proj = val_flat @ Vt[:r, :].T @ Vt[:r, :]
                    err = float(np.mean((val_flat - val_proj) ** 2))
                else:
                    # Flatten and use SVD as proxy
                    flat = train_data.reshape(len(train_idx), -1)
                    U, s, Vt = np.linalg.svd(flat, full_matrices=False)
                    r = min(rank, len(s))
                    val_flat = val_data.reshape(len(val_idx), -1)
                    val_proj = val_flat @ Vt[:r, :].T @ Vt[:r, :]
                    err = float(np.mean((val_flat - val_proj) ** 2))
                cv_errors[rank].append(err)
            except Exception as e:
                cv_errors[rank].append(float("inf"))
                warnings.warn(f"CV error for rank={rank}, fold={fold_idx}: {e}")

    mean_errs = {r: np.mean(errors) for r, errors in cv_errors.items()}
    std_errs = {r: np.std(errors) for r, errors in cv_errors.items()}
    best_rank = min(mean_errs, key=mean_errs.get)

    return {
        "best_rank": best_rank,
        "cv_errors": cv_errors,
        "mean_cv_error": mean_errs,
        "std_cv_error": std_errs,
    }


def one_standard_error_rule(
    cv_result: dict,
) -> int:
    """
    Apply the one-standard-error rule to CV results.

    Select the simplest model (lowest rank) whose CV error is within
    one standard error of the best model.

    Parameters
    ----------
    cv_result : dict from kfold_rank_selection

    Returns
    -------
    selected_rank : int
    """
    mean_errs = cv_result["mean_cv_error"]
    std_errs = cv_result["std_cv_error"]
    best_rank = cv_result["best_rank"]
    best_err = mean_errs[best_rank]
    best_std = std_errs[best_rank]
    threshold = best_err + best_std

    # Select simplest model within threshold
    sorted_ranks = sorted(mean_errs.keys())
    for r in sorted_ranks:
        if mean_errs[r] <= threshold:
            return r
    return best_rank


# ---------------------------------------------------------------------------
# Section: Tensor network rank distribution analysis
# ---------------------------------------------------------------------------


def analyse_tt_rank_distribution(
    tensor: np.ndarray,
    max_rank: int = 32,
    shape: tuple | None = None,
) -> dict:
    """
    Analyse the natural rank distribution of a tensor via sequential SVD.

    Parameters
    ----------
    tensor : np.ndarray, shape (N1, N2, ..., Nd)
    max_rank : int
        Maximum rank to consider.
    shape : tuple, optional
        If tensor is 2-D, treat it as unfolded tensor with this shape.

    Returns
    -------
    dict with:
    * ``"singular_values"`` : list of np.ndarray per mode
    * ``"effective_ranks"`` : np.ndarray, shape (d-1,) — suggested ranks
    * ``"cumulative_variance"`` : list of np.ndarray per mode
    """
    if tensor.ndim == 2:
        modes = [tensor]
    else:
        d = tensor.ndim
        modes = []
        for k in range(d - 1):
            dims = tensor.shape
            n_k = dims[k]
            rest = int(np.prod(dims[:k]) * np.prod(dims[k+1:]))
            unfolded = tensor.reshape(n_k, rest)
            modes.append(unfolded)

    sv_lists = []
    eff_ranks = []
    cumvar_lists = []

    for mode in modes:
        _, s, _ = np.linalg.svd(mode, full_matrices=False)
        s = s[:max_rank]
        sv_lists.append(s)
        total_energy = np.sum(s ** 2)
        cum_var = np.cumsum(s ** 2) / (total_energy + 1e-12)
        cumvar_lists.append(cum_var)
        # Effective rank: rank needed for 95% variance
        r95 = int(np.searchsorted(cum_var, 0.95)) + 1
        eff_ranks.append(r95)

    return {
        "singular_values": sv_lists,
        "effective_ranks": np.array(eff_ranks),
        "cumulative_variance": cumvar_lists,
    }


def suggest_tt_ranks(
    tensor: np.ndarray,
    target_variance: float = 0.95,
    max_rank: int = 64,
) -> list:
    """
    Suggest TT bond dimensions for a given tensor.

    Parameters
    ----------
    tensor : np.ndarray
    target_variance : float
        Fraction of variance to preserve at each bond.
    max_rank : int

    Returns
    -------
    ranks : list of int  — bond dimensions (length = n_modes - 1)
    """
    analysis = analyse_tt_rank_distribution(tensor, max_rank=max_rank)
    ranks = []
    for cum_var in analysis["cumulative_variance"]:
        r = int(np.searchsorted(cum_var, target_variance)) + 1
        r = min(r, max_rank)
        ranks.append(r)
    return ranks


# ---------------------------------------------------------------------------
# Section: Rank sensitivity analysis
# ---------------------------------------------------------------------------


def rank_sensitivity_analysis(
    data: np.ndarray,
    rank_range: tuple = (1, 32),
    n_samples: int = 10,
    metric: str = "frobenius",
) -> dict:
    """
    Analyse how reconstruction quality varies with rank.

    Parameters
    ----------
    data : np.ndarray, shape (T, N)
    rank_range : tuple (min_rank, max_rank)
    n_samples : int
        Number of rank values to evaluate (log-spaced).
    metric : str
        "frobenius" | "relative" | "max_abs"

    Returns
    -------
    dict with ``"ranks"``, ``"errors"``, ``"compression_ratios"``.
    """
    T, N = data.shape
    ranks = np.unique(np.linspace(rank_range[0], rank_range[1], n_samples).astype(int))
    U_full, s_full, Vt_full = np.linalg.svd(data, full_matrices=False)
    total_norm = float(np.linalg.norm(data, "fro") + 1e-12)
    n_params_full = T * N

    errors = []
    comp_ratios = []

    for r in ranks:
        r = min(r, len(s_full))
        recon = (U_full[:, :r] * s_full[:r]) @ Vt_full[:r, :]
        diff = data - recon
        if metric == "frobenius":
            err = float(np.linalg.norm(diff, "fro"))
        elif metric == "relative":
            err = float(np.linalg.norm(diff, "fro") / total_norm)
        elif metric == "max_abs":
            err = float(np.abs(diff).max())
        else:
            err = float(np.linalg.norm(diff, "fro") / total_norm)
        errors.append(err)

        # Compression: parameters stored in low-rank form
        n_params_lr = r * T + r + r * N
        comp_ratios.append(n_params_lr / n_params_full)

    return {
        "ranks": ranks.tolist(),
        "errors": errors,
        "compression_ratios": comp_ratios,
    }


def find_elbow_rank(errors: list, ranks: list) -> int:
    """
    Find the 'elbow' rank using the L-method (maximum curvature).

    Parameters
    ----------
    errors : list of float
    ranks : list of int

    Returns
    -------
    elbow_rank : int
    """
    if len(ranks) < 3:
        return ranks[0]
    errors_arr = np.array(errors, dtype=np.float64)
    ranks_arr = np.array(ranks, dtype=np.float64)
    # Normalise to [0, 1]
    r_norm = (ranks_arr - ranks_arr.min()) / (ranks_arr.ptp() + 1e-12)
    e_norm = (errors_arr - errors_arr.min()) / (errors_arr.ptp() + 1e-12)
    # Distance from each point to the line connecting first and last
    x0, y0 = r_norm[0], e_norm[0]
    x1, y1 = r_norm[-1], e_norm[-1]
    dx = x1 - x0
    dy = y1 - y0
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-12
    distances = np.abs(dy * r_norm - dx * e_norm + x1 * y0 - y1 * x0) / norm
    best_idx = int(np.argmax(distances))
    return int(ranks[best_idx])


# ---------------------------------------------------------------------------
# Section: Randomised rank selection
# ---------------------------------------------------------------------------


def randomised_svd_rank_selection(
    data: np.ndarray,
    target_fraction: float = 0.95,
    n_oversampling: int = 5,
    n_power_iter: int = 2,
    random_seed: int = 42,
) -> int:
    """
    Estimate minimum rank via randomised SVD.

    Uses a random sketch to estimate the top singular values efficiently,
    then finds the rank needed to explain ``target_fraction`` of variance.

    Parameters
    ----------
    data : np.ndarray, shape (T, N)
    target_fraction : float
    n_oversampling : int
    n_power_iter : int
    random_seed : int

    Returns
    -------
    rank : int
    """
    rng = np.random.default_rng(random_seed)
    T, N = data.shape
    max_rank = min(T, N)
    sketch_size = min(max_rank, 64) + n_oversampling

    # Random sketch
    Omega = rng.standard_normal((N, sketch_size))
    Y = data @ Omega
    # Power iteration for better approximation
    for _ in range(n_power_iter):
        Y = data @ (data.T @ Y)
    # QR factorisation
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ data          # (sketch_size, N)
    _, s, _ = np.linalg.svd(B, full_matrices=False)

    total_energy = np.sum(s ** 2)
    cum_var = np.cumsum(s ** 2) / (total_energy + 1e-12)
    rank = int(np.searchsorted(cum_var, target_fraction)) + 1
    return min(rank, max_rank)


# ---------------------------------------------------------------------------
# Section: Rank selection for streaming/online settings
# ---------------------------------------------------------------------------


class OnlineRankEstimator:
    """
    Online estimator of effective rank using streaming observations.

    Maintains a sketch of the data covariance and updates a rank estimate
    based on the running spectral decay.

    Parameters
    ----------
    n_features : int
        Dimensionality of incoming observations.
    sketch_size : int
        Number of sketch components.
    target_variance : float
        Fraction of variance to explain.
    decay : float
        Exponential forgetting factor in [0, 1].
    """

    def __init__(
        self,
        n_features: int,
        sketch_size: int = 32,
        target_variance: float = 0.95,
        decay: float = 0.99,
    ) -> None:
        self.n_features = n_features
        self.sketch_size = sketch_size
        self.target_variance = target_variance
        self.decay = decay
        self._C = np.zeros((n_features, n_features), dtype=np.float64)
        self._n = 0.0

    def update(self, x: np.ndarray) -> int:
        """
        Update with a new observation and return current rank estimate.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,) or (T, n_features)

        Returns
        -------
        rank : int
        """
        x = np.atleast_2d(x).astype(np.float64)
        lam = self.decay
        for xi in x:
            self._C = lam * self._C + (1 - lam) * np.outer(xi, xi)
            self._n = lam * self._n + 1.0
        return self.estimate_rank()

    def estimate_rank(self) -> int:
        """Return current rank estimate from the covariance sketch."""
        if self._n < 2:
            return 1
        eigvals = np.sort(np.linalg.eigvalsh(self._C))[::-1]
        eigvals = np.maximum(eigvals, 0)
        total = eigvals.sum() + 1e-12
        cum_var = np.cumsum(eigvals) / total
        rank = int(np.searchsorted(cum_var, self.target_variance)) + 1
        return max(1, min(rank, self.n_features))

    def reset(self) -> None:
        self._C = np.zeros((self.n_features, self.n_features), dtype=np.float64)
        self._n = 0.0


# ---------------------------------------------------------------------------
# Section: Hierarchical rank selection
# ---------------------------------------------------------------------------


def nested_rank_selection(
    tensor: np.ndarray,
    level_ranks: list | None = None,
    target_variance: float = 0.95,
) -> dict:
    """
    Select ranks for a hierarchical Tucker decomposition via nested SVD.

    Parameters
    ----------
    tensor : np.ndarray, shape (N1, N2, ..., Nd)
    level_ranks : list of int, optional
        Maximum rank per level. If None, auto-detected.
    target_variance : float

    Returns
    -------
    dict with ``"mode_ranks"`` and ``"level_ranks"``.
    """
    d = tensor.ndim
    mode_ranks = []

    for mode in range(d):
        dims = list(tensor.shape)
        n_mode = dims[mode]
        rest = int(np.prod(dims) // n_mode)
        unfolded = np.moveaxis(tensor, mode, 0).reshape(n_mode, rest)
        _, s, _ = np.linalg.svd(unfolded, full_matrices=False)
        total = np.sum(s ** 2) + 1e-12
        cum_var = np.cumsum(s ** 2) / total
        r = int(np.searchsorted(cum_var, target_variance)) + 1
        max_r = level_ranks[mode] if level_ranks and mode < len(level_ranks) else n_mode
        mode_ranks.append(min(r, max_r, n_mode))

    return {
        "mode_ranks": mode_ranks,
        "d": d,
        "tensor_shape": list(tensor.shape),
        "target_variance": target_variance,
    }


# ---------------------------------------------------------------------------
# Section: Rank regularisation schedules
# ---------------------------------------------------------------------------


def cosine_rank_schedule(
    step: int,
    total_steps: int,
    min_rank: int = 1,
    max_rank: int = 32,
) -> int:
    """
    Cosine annealing rank schedule.

    Parameters
    ----------
    step : int
        Current training step.
    total_steps : int
    min_rank : int
    max_rank : int

    Returns
    -------
    rank : int
    """
    import math
    frac = step / max(1, total_steps)
    cos_val = 0.5 * (1 + math.cos(math.pi * frac))
    rank = min_rank + cos_val * (max_rank - min_rank)
    return max(min_rank, min(max_rank, round(rank)))


def cyclic_rank_schedule(
    step: int,
    cycle_length: int,
    min_rank: int = 1,
    max_rank: int = 32,
    mode: str = "triangular",
) -> int:
    """
    Cyclic rank schedule.

    Parameters
    ----------
    step : int
    cycle_length : int
    min_rank : int
    max_rank : int
    mode : str
        "triangular" | "exp_range"

    Returns
    -------
    rank : int
    """
    cycle = step % cycle_length
    frac = cycle / max(1, cycle_length)
    if mode == "triangular":
        if frac < 0.5:
            rank = min_rank + 2 * frac * (max_rank - min_rank)
        else:
            rank = max_rank - 2 * (frac - 0.5) * (max_rank - min_rank)
    elif mode == "exp_range":
        import math
        rank = min_rank + (max_rank - min_rank) * math.exp(-frac * 3)
    else:
        rank = min_rank + frac * (max_rank - min_rank)
    return max(min_rank, min(max_rank, round(rank)))


def warmup_then_fixed_rank_schedule(
    step: int,
    warmup_steps: int,
    final_rank: int,
    min_rank: int = 1,
) -> int:
    """
    Linearly increase rank from min_rank to final_rank during warmup.

    Parameters
    ----------
    step : int
    warmup_steps : int
    final_rank : int
    min_rank : int
    """
    if step >= warmup_steps:
        return final_rank
    frac = step / max(1, warmup_steps)
    rank = min_rank + frac * (final_rank - min_rank)
    return max(min_rank, min(final_rank, round(rank)))


# ---------------------------------------------------------------------------
# Section: Rank compression diagnostics
# ---------------------------------------------------------------------------


def rank_compression_diagnostic(
    original: np.ndarray,
    compressed: np.ndarray,
    rank: int,
) -> dict:
    """
    Compute a battery of diagnostics comparing original and compressed tensors.

    Parameters
    ----------
    original : np.ndarray
    compressed : np.ndarray (same shape)
    rank : int
        Rank used in compression.

    Returns
    -------
    dict with error metrics.
    """
    assert original.shape == compressed.shape, "Shapes must match"
    diff = original - compressed
    orig_norm = float(np.linalg.norm(original.ravel()) + 1e-12)
    diff_norm = float(np.linalg.norm(diff.ravel()))

    return {
        "rank": rank,
        "frobenius_error": diff_norm,
        "relative_frobenius_error": diff_norm / orig_norm,
        "max_absolute_error": float(np.abs(diff).max()),
        "mean_absolute_error": float(np.abs(diff).mean()),
        "snr_db": float(20 * np.log10(orig_norm / (diff_norm + 1e-12))),
        "compression_ratio": float(
            np.prod(original.shape) / (rank * (sum(original.shape) + rank))
        ),
        "r_squared": float(1 - np.sum(diff ** 2) / (np.sum(original ** 2) + 1e-12)),
    }


# ---------------------------------------------------------------------------
# Section: Rank selection for attention mechanisms
# ---------------------------------------------------------------------------


def select_attention_rank(
    query_key_matrix: np.ndarray,
    target_approx_error: float = 0.01,
    max_rank: int = 64,
) -> int:
    """
    Select attention matrix rank for low-rank attention approximation.

    Finds the minimum rank such that the Frobenius norm of the reconstruction
    error divided by the original norm is below ``target_approx_error``.

    Parameters
    ----------
    query_key_matrix : np.ndarray, shape (T, T) or (B, T, T)
        Attention score matrix (before softmax) or a batch thereof.
    target_approx_error : float
    max_rank : int

    Returns
    -------
    rank : int
    """
    if query_key_matrix.ndim == 3:
        # Average over batch
        qk = query_key_matrix.mean(axis=0)
    else:
        qk = query_key_matrix

    _, s, _ = np.linalg.svd(qk, full_matrices=False)
    total_energy = np.sum(s ** 2) + 1e-12
    target_energy = (1 - target_approx_error ** 2) * total_energy

    cum_energy = np.cumsum(s ** 2)
    rank = int(np.searchsorted(cum_energy, target_energy)) + 1
    return min(rank, max_rank, len(s))


def select_head_ranks(
    attention_matrices: list,
    target_approx_error: float = 0.01,
    max_rank: int = 64,
) -> list:
    """
    Select per-head ranks for multi-head attention.

    Parameters
    ----------
    attention_matrices : list of np.ndarray
        One (T, T) attention matrix per head.
    target_approx_error : float
    max_rank : int

    Returns
    -------
    ranks : list of int
    """
    return [
        select_attention_rank(mat, target_approx_error, max_rank)
        for mat in attention_matrices
    ]

