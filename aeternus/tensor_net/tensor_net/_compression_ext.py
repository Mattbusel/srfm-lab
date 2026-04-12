"""Extension for compression_pipeline.py — appended programmatically."""

# ---------------------------------------------------------------------------
# Section: Advanced compression utilities
# ---------------------------------------------------------------------------

import numpy as np
import warnings


def compute_compression_pareto_front(
    data: np.ndarray,
    rank_range: tuple = (1, 32),
    n_points: int = 15,
) -> dict:
    """
    Compute the Pareto front of compression ratio vs. reconstruction quality.

    Parameters
    ----------
    data : np.ndarray, shape (M, N)
    rank_range : tuple (min_rank, max_rank)
    n_points : int

    Returns
    -------
    dict with ``"ranks"``, ``"compression_ratios"``, ``"snr_db_list"``,
    ``"relative_errors"``.
    """
    M, N = data.shape
    ranks = np.unique(
        np.round(np.logspace(np.log10(rank_range[0]), np.log10(rank_range[1]), n_points)).astype(int)
    )
    U_full, s_full, Vt_full = np.linalg.svd(data, full_matrices=False)
    orig_norm = float(np.linalg.norm(data, "fro") + 1e-12)

    compression_ratios = []
    snr_db_list = []
    relative_errors = []

    for r in ranks:
        r = int(min(r, len(s_full)))
        recon = (U_full[:, :r] * s_full[:r]) @ Vt_full[:r, :]
        err = float(np.linalg.norm(data - recon, "fro"))
        snr_db = float(20 * np.log10(orig_norm / (err + 1e-12)))
        rel_err = err / orig_norm
        n_params_lr = r * M + r + r * N
        n_params_full = M * N
        comp_ratio = n_params_full / (n_params_lr + 1e-12)

        compression_ratios.append(comp_ratio)
        snr_db_list.append(snr_db)
        relative_errors.append(rel_err)

    return {
        "ranks": ranks.tolist(),
        "compression_ratios": compression_ratios,
        "snr_db_list": snr_db_list,
        "relative_errors": relative_errors,
    }


def adaptive_rank_from_snr_target(
    data: np.ndarray,
    target_snr_db: float = 30.0,
    max_rank: int | None = None,
) -> int:
    """
    Find minimum rank achieving ``target_snr_db`` SNR.

    Parameters
    ----------
    data : np.ndarray, shape (M, N)
    target_snr_db : float
    max_rank : int, optional

    Returns
    -------
    rank : int
    """
    M, N = data.shape
    max_r = min(M, N) if max_rank is None else min(max_rank, M, N)
    U, s, Vt = np.linalg.svd(data, full_matrices=False)
    orig_norm = float(np.linalg.norm(data, "fro"))
    target_err = orig_norm / (10 ** (target_snr_db / 20.0))

    for r in range(1, max_r + 1):
        recon = (U[:, :r] * s[:r]) @ Vt[:r, :]
        err = float(np.linalg.norm(data - recon, "fro"))
        if err <= target_err:
            return r

    return max_r


def compress_with_target_snr(
    data: np.ndarray,
    target_snr_db: float = 30.0,
) -> tuple:
    """
    Compress data to achieve a target SNR.

    Returns
    -------
    (rank, compressed_data) where compressed_data has same shape as data.
    """
    rank = adaptive_rank_from_snr_target(data, target_snr_db)
    U, s, Vt = np.linalg.svd(data, full_matrices=False)
    recon = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
    return rank, recon.astype(np.float32)


class TieredCompressionPipeline:
    """
    Multi-tier compression pipeline that applies different compression
    levels based on data freshness/importance.

    Tiers:
    * Tier 1 (hot): recent data, high quality (high rank)
    * Tier 2 (warm): medium-age data, balanced
    * Tier 3 (cold): old data, high compression

    Parameters
    ----------
    hot_rank : int
    warm_rank : int
    cold_rank : int
    hot_window : int
        Number of most-recent time steps in hot tier.
    warm_window : int
        Additional time steps in warm tier.
    """

    def __init__(
        self,
        hot_rank: int = 16,
        warm_rank: int = 8,
        cold_rank: int = 4,
        hot_window: int = 21,
        warm_window: int = 63,
    ) -> None:
        self.hot_rank = hot_rank
        self.warm_rank = warm_rank
        self.cold_rank = cold_rank
        self.hot_window = hot_window
        self.warm_window = warm_window

    def compress(self, data: np.ndarray) -> dict:
        """
        Apply tiered compression.

        Parameters
        ----------
        data : np.ndarray, shape (T, N)

        Returns
        -------
        dict with compressed arrays for each tier and metadata.
        """
        T, N = data.shape
        hot_end = T
        hot_start = max(0, T - self.hot_window)
        warm_start = max(0, hot_start - self.warm_window)

        def compress_segment(seg, rank):
            if seg.shape[0] < 1:
                return seg
            r = min(rank, seg.shape[0], seg.shape[1])
            U, s, Vt = np.linalg.svd(seg, full_matrices=False)
            return (U[:, :r] * s[:r]) @ Vt[:r, :]

        hot = compress_segment(data[hot_start:hot_end], self.hot_rank)
        warm = compress_segment(data[warm_start:hot_start], self.warm_rank)
        cold = compress_segment(data[:warm_start], self.cold_rank) if warm_start > 0 else np.zeros((0, N), dtype=np.float32)

        return {
            "hot": hot.astype(np.float32),
            "warm": warm.astype(np.float32),
            "cold": cold.astype(np.float32),
            "hot_range": (hot_start, hot_end),
            "warm_range": (warm_start, hot_start),
            "cold_range": (0, warm_start),
        }

    def decompress(self, compressed: dict) -> np.ndarray:
        """Reconstruct full tensor from tiered compressed form."""
        parts = [compressed["cold"], compressed["warm"], compressed["hot"]]
        parts = [p for p in parts if p.shape[0] > 0]
        if not parts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Section: Incremental / streaming compression
# ---------------------------------------------------------------------------


class IncrementalMatrixCompressor:
    """
    Incrementally updates a low-rank matrix approximation as new rows arrive.

    Uses Brand (2002) incremental SVD to maintain a rank-r approximation
    without recomputing the full SVD.

    Parameters
    ----------
    rank : int
        Target rank.
    n_features : int
        Number of columns (features).
    decay : float
        Exponential forgetting factor.
    """

    def __init__(self, rank: int, n_features: int, decay: float = 1.0) -> None:
        self.rank = rank
        self.n_features = n_features
        self.decay = decay
        self._U: np.ndarray | None = None   # (T, r)
        self._s: np.ndarray | None = None   # (r,)
        self._Vt: np.ndarray | None = None  # (r, N)
        self._n_updates = 0

    def update(self, new_row: np.ndarray) -> None:
        """
        Update the approximation with a new row vector.

        Parameters
        ----------
        new_row : np.ndarray, shape (n_features,)
        """
        x = new_row.reshape(1, self.n_features).astype(np.float64)
        r = self.rank
        decay = self.decay

        if self._U is None:
            # First update
            U, s, Vt = np.linalg.svd(x, full_matrices=False)
            self._U = U[:, :r]
            self._s = s[:r]
            self._Vt = Vt[:r, :]
        else:
            # Approximate rank-1 update
            # Scale existing components
            self._U *= decay
            self._s *= decay
            # Add new row
            proj = x @ self._Vt.T      # (1, r) projection onto existing basis
            perp = x - proj @ self._Vt  # (1, N) perpendicular component
            perp_norm = float(np.linalg.norm(perp))

            if perp_norm < 1e-10:
                # New vector in span of existing basis
                K = np.zeros((r + 1, r + 1))
                K[:r, :r] = np.diag(self._s)
                K[:r, r] = proj.ravel()
                K[r, r] = 0.0
            else:
                q = perp / perp_norm
                K = np.zeros((r + 1, r + 1))
                K[:r, :r] = np.diag(self._s)
                K[:r, r] = proj.ravel()
                K[r, r] = perp_norm

            # SVD of small K matrix
            Up, sp, Vpt = np.linalg.svd(K, full_matrices=False)
            r_new = min(r, len(sp))

            # Update factors
            Q_u = np.block([
                [self._U, np.zeros((self._U.shape[0], 1))],
                [np.zeros((1, r)), np.ones((1, 1))]
            ])
            Q_v = np.block([
                [self._Vt],
                [q if perp_norm > 1e-10 else np.zeros((1, self.n_features))]
            ])

            self._U = Q_u @ Up[:, :r_new]
            self._s = sp[:r_new]
            self._Vt = (Vpt[:r_new, :] @ Q_v)[:r_new, :]

        self._n_updates += 1

    def reconstruct(self) -> np.ndarray:
        """Return the current rank-r approximation as (n_updates, n_features)."""
        if self._U is None:
            return np.zeros((0, self.n_features), dtype=np.float32)
        return (self._U * self._s[None, :]) @ self._Vt

    @property
    def n_updates(self) -> int:
        return self._n_updates

    def compression_ratio(self) -> float:
        """Current compression ratio."""
        r = self._s.shape[0] if self._s is not None else 0
        n_stored = r * (self._n_updates + self.n_features + 1)
        n_original = self._n_updates * self.n_features
        return n_original / (n_stored + 1e-12)


# ---------------------------------------------------------------------------
# Section: Financial matrix compression — correlation and covariance
# ---------------------------------------------------------------------------


def compress_correlation_matrix(
    corr: np.ndarray,
    rank: int,
    ensure_valid: bool = True,
) -> tuple:
    """
    Compress a correlation matrix via low-rank approximation.

    Parameters
    ----------
    corr : np.ndarray, shape (N, N)  symmetric, diag = 1
    rank : int
    ensure_valid : bool
        If True, project compressed matrix to nearest valid correlation matrix.

    Returns
    -------
    (compressed_corr, error_metrics)
    """
    N = corr.shape[0]
    # Eigen-decomposition (symmetric matrix)
    eigvals, eigvecs = np.linalg.eigh(corr)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    r = min(rank, N)
    recon = (eigvecs[:, :r] * eigvals[:r]) @ eigvecs[:, :r].T

    if ensure_valid:
        # Project to valid correlation matrix
        # 1. Ensure diagonal = 1 by rescaling
        std = np.sqrt(np.diag(recon))
        recon = recon / (np.outer(std, std) + 1e-12)
        np.fill_diagonal(recon, 1.0)
        # 2. Ensure PSD by clipping negative eigenvalues
        ev, evec = np.linalg.eigh(recon)
        ev = np.maximum(ev, 0)
        recon = (evec * ev) @ evec.T
        std2 = np.sqrt(np.diag(recon))
        recon = recon / (np.outer(std2, std2) + 1e-12)
        np.fill_diagonal(recon, 1.0)

    diff = corr - recon
    metrics = {
        "rank": r,
        "frobenius_error": float(np.linalg.norm(diff, "fro")),
        "max_off_diag_error": float(np.abs(diff - np.diag(np.diag(diff))).max()),
        "min_eigenvalue": float(np.linalg.eigvalsh(recon).min()),
    }
    return recon.astype(np.float32), metrics


def compress_covariance_matrix(
    cov: np.ndarray,
    rank: int,
) -> tuple:
    """
    Compress a covariance matrix via factor model approximation.

    Represents Sigma ~ B Sigma_f B^T + D where D is diagonal.

    Parameters
    ----------
    cov : np.ndarray, shape (N, N)
    rank : int

    Returns
    -------
    (compressed_cov, B, Sigma_f, D, error_metrics)
    """
    N = cov.shape[0]
    std = np.sqrt(np.diag(cov) + 1e-12)
    corr = cov / np.outer(std, std)

    # Factor decomposition via eigen
    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    r = min(rank, N)

    # Factor loadings and covariance
    B = eigvecs[:, :r] * np.sqrt(np.maximum(eigvals[:r], 0))  # (N, r)
    Sigma_f = np.eye(r)   # by construction
    systematic = B @ Sigma_f @ B.T

    # Rescale to original covariance
    B_scaled = B * std[:, None]
    systematic_scaled = B_scaled @ Sigma_f @ B_scaled.T

    # Idiosyncratic diagonal
    D_diag = np.maximum(np.diag(cov) - np.diag(systematic_scaled), 0)
    D = np.diag(D_diag)

    compressed_cov = (systematic_scaled + D).astype(np.float32)
    diff = cov - compressed_cov
    metrics = {
        "rank": r,
        "frobenius_error": float(np.linalg.norm(diff, "fro")),
        "relative_error": float(np.linalg.norm(diff, "fro") / (np.linalg.norm(cov, "fro") + 1e-12)),
        "idiosyncratic_fraction": float(D_diag.sum() / (np.trace(cov) + 1e-12)),
    }
    return compressed_cov, B_scaled, Sigma_f, D, metrics


# ---------------------------------------------------------------------------
# Section: Compression format export
# ---------------------------------------------------------------------------


def save_compressed_factors(
    factors: dict,
    path: str,
    format: str = "npz",
) -> None:
    """
    Save compressed factor matrices to disk.

    Parameters
    ----------
    factors : dict mapping name -> np.ndarray
    path : str
        Output path (with or without extension).
    format : str
        "npz" | "json"
    """
    if format == "npz":
        np.savez_compressed(path, **factors)
    elif format == "json":
        import json
        serialisable = {}
        for k, v in factors.items():
            if isinstance(v, np.ndarray):
                serialisable[k] = {"data": v.tolist(), "shape": list(v.shape), "dtype": str(v.dtype)}
            else:
                serialisable[k] = v
        with open(path, "w") as fh:
            json.dump(serialisable, fh)
    else:
        raise ValueError(f"Unknown format: {format!r}")


def load_compressed_factors(path: str, format: str = "npz") -> dict:
    """
    Load compressed factor matrices from disk.

    Parameters
    ----------
    path : str
    format : str
        "npz" | "json"

    Returns
    -------
    dict
    """
    if format == "npz":
        data = np.load(path)
        return dict(data)
    elif format == "json":
        import json
        with open(path) as fh:
            raw = json.load(fh)
        result = {}
        for k, v in raw.items():
            if isinstance(v, dict) and "data" in v:
                result[k] = np.array(v["data"], dtype=v.get("dtype", "float32"))
            else:
                result[k] = v
        return result
    else:
        raise ValueError(f"Unknown format: {format!r}")


# ---------------------------------------------------------------------------
# Section: Compression quality monitoring
# ---------------------------------------------------------------------------


class CompressionQualityMonitor:
    """
    Monitors compression quality over time and alerts when quality degrades.

    Parameters
    ----------
    target_snr_db : float
        Minimum acceptable SNR in dB.
    window : int
        Rolling window for quality trend detection.
    alert_threshold_std : float
        Alert if quality drops by more than this many std devs.
    """

    def __init__(
        self,
        target_snr_db: float = 25.0,
        window: int = 20,
        alert_threshold_std: float = 2.0,
    ) -> None:
        self.target_snr_db = target_snr_db
        self.window = window
        self.alert_threshold_std = alert_threshold_std
        self._history: list = []

    def record(self, snr_db: float) -> bool:
        """
        Record a new SNR measurement.

        Returns True if an alert is triggered (quality degraded).
        """
        self._history.append(snr_db)
        return self.check_alert()

    def check_alert(self) -> bool:
        """Check if current quality is below threshold or degrading."""
        if len(self._history) < 2:
            return False
        latest = self._history[-1]
        # Alert 1: below absolute target
        if latest < self.target_snr_db:
            return True
        # Alert 2: sharp drop vs. rolling window
        if len(self._history) >= self.window:
            recent = np.array(self._history[-self.window:])
            mean = recent[:-1].mean()
            std = recent[:-1].std() + 1e-8
            if (mean - latest) > self.alert_threshold_std * std:
                return True
        return False

    def trend(self) -> float:
        """Return linear trend of SNR (positive = improving, negative = degrading)."""
        if len(self._history) < 2:
            return 0.0
        y = np.array(self._history, dtype=np.float64)
        x = np.arange(len(y), dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0])
        return slope

    def summary(self) -> dict:
        if not self._history:
            return {}
        arr = np.array(self._history)
        return {
            "n_measurements": len(arr),
            "mean_snr_db": float(arr.mean()),
            "min_snr_db": float(arr.min()),
            "max_snr_db": float(arr.max()),
            "std_snr_db": float(arr.std()),
            "trend": self.trend(),
            "alerts_would_trigger": int(self.check_alert()),
        }
