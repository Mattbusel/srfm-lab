"""Extension for rank_selection.py — appended programmatically."""

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
