"""
compression_pipeline.py — End-to-end TT compression pipeline for TensorNet (Project AETERNUS).

Provides:
  - End-to-end: ingest correlation matrix → rank selection → TT compress → monitor → decompress
  - Streaming update support (rolling re-compression as new data arrives)
  - Compression ratio dashboard (console + matplotlib)
  - Fidelity alert system (webhook / logging) when error exceeds threshold
  - On-demand decompression with optional precision upgrade
  - Compression history tracking
  - Batch compression for multiple matrices
  - Differential update (update compressed form with a rank-1 correction)
  - Compression quality metrics (SNR, relative error, SSIM-proxy)
"""

from __future__ import annotations

import math
import time
import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp

from tensor_net.rank_selection import (
    InformationTheoreticRankSelector,
    tucker_rank_per_mode,
    prune_tt_ranks_magnitude,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class CompressionRecord:
    """Single compression event record."""
    timestamp: float
    input_shape: Tuple[int, ...]
    n_params_original: int
    n_params_compressed: int
    compression_ratio: float
    reconstruction_error: float
    rank_used: int
    method: str
    elapsed_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def snr_db(self) -> float:
        if self.reconstruction_error < 1e-15:
            return float("inf")
        return -20.0 * math.log10(self.reconstruction_error + 1e-15)


@dataclass
class CompressionPipelineConfig:
    """Configuration for the end-to-end compression pipeline."""
    max_rank: int = 32
    rank_method: str = "bic"      # rank selection method
    target_variance: float = 0.99
    prune_threshold: float = 1e-3
    fidelity_threshold: float = 0.05   # alert when error > this
    alert_on_fidelity: bool = True
    streaming_window: int = 500        # points to retain in streaming buffer
    streaming_overlap: int = 50
    decompress_dtype: str = "float32"
    store_history: bool = True
    max_history: int = 1000
    verbose: bool = False
    dtype: str = "float32"


@dataclass
class CompressedMatrix:
    """A compressed correlation/covariance matrix in TT format."""
    cores: List[np.ndarray]      # TT cores (stored as numpy for serialization)
    original_shape: Tuple[int, ...]
    ranks: List[int]
    compression_ratio: float
    reconstruction_error: float
    timestamp: float = field(default_factory=time.time)

    def to_dense(self) -> np.ndarray:
        """Decompress to dense matrix.

        Returns:
            Dense matrix of shape original_shape.
        """
        result = np.array(self.cores[0])
        for core in self.cores[1:]:
            r_l = result.shape[-1]
            r_r = core.shape[2]
            d = core.shape[1]
            result_flat = result.reshape(-1, r_l)
            core_flat = core.reshape(r_l, -1)
            result = (result_flat @ core_flat).reshape(
                result.shape[:-1] + (d, r_r)
            )

        return result.reshape(self.original_shape)

    @property
    def n_params(self) -> int:
        return sum(c.size for c in self.cores)


# ============================================================================
# Core compression functions
# ============================================================================

def compress_matrix_to_tt(
    matrix: np.ndarray,
    max_rank: int = 32,
    method: str = "bic",
    target_variance: float = 0.99,
    prune_threshold: float = 1e-3,
    verbose: bool = False,
) -> CompressedMatrix:
    """Compress a 2D matrix to Tensor Train format.

    Uses SVD-based rank selection followed by TT decomposition.

    Args:
        matrix: Input 2D array (n, m).
        max_rank: Maximum TT bond dimension.
        method: Rank selection method.
        target_variance: Variance threshold (for "variance" method).
        prune_threshold: Singular value pruning threshold.
        verbose: Print diagnostics.

    Returns:
        CompressedMatrix.
    """
    t0 = time.time()
    mat = np.asarray(matrix, dtype=np.float64)
    orig_shape = mat.shape
    n_orig = mat.size

    # Flatten to 2D if needed
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    flat = mat.reshape(mat.shape[0], -1)

    # Rank selection
    selector = InformationTheoreticRankSelector(flat, max_rank=max_rank)
    if method == "bic":
        rank = selector.rank_by_bic()
    elif method == "aic":
        rank = selector.rank_by_aic()
    elif method == "mdl":
        rank = selector.rank_by_mdl()
    elif method == "variance":
        rank = selector.rank_by_variance(target_variance)
    elif method == "elbow":
        rank = selector.rank_by_elbow()
    else:
        rank = selector.consensus_rank()

    if verbose:
        print(f"  Rank selected: {rank} (method={method})")

    # SVD truncation (proxy for TT)
    U, s, Vt = np.linalg.svd(flat, full_matrices=False)
    r = min(rank, len(s))
    U_r = (U[:, :r] * s[:r]).astype(np.float32)
    Vt_r = Vt[:r, :].astype(np.float32)

    # Store as two TT cores: (1, n, r) and (r, m, 1)
    cores = [
        U_r.reshape(1, mat.shape[0], r),
        Vt_r.reshape(r, flat.shape[1], 1),
    ]

    # Optionally prune
    if prune_threshold > 0:
        cores, _ = prune_tt_ranks_magnitude(cores, threshold=prune_threshold)

    n_compressed = sum(c.size for c in cores)
    compression_ratio = n_orig / max(1, n_compressed)

    # Compute reconstruction error
    recon = (U_r @ Vt_r).reshape(orig_shape)
    orig_norm = float(np.linalg.norm(matrix))
    err = float(np.linalg.norm(matrix.reshape(orig_shape) - recon)) / (orig_norm + 1e-15)

    if verbose:
        print(f"  Compression: {compression_ratio:.2f}x, Error: {err:.6f}")

    return CompressedMatrix(
        cores=cores,
        original_shape=orig_shape,
        ranks=[r],
        compression_ratio=compression_ratio,
        reconstruction_error=err,
        timestamp=time.time(),
    )


def decompress_matrix(
    compressed: CompressedMatrix,
    dtype: str = "float32",
) -> np.ndarray:
    """Decompress a CompressedMatrix to dense.

    Args:
        compressed: CompressedMatrix object.
        dtype: Output dtype.

    Returns:
        Dense matrix.
    """
    dense = compressed.to_dense()
    return dense.astype(dtype)


# ============================================================================
# Streaming compression
# ============================================================================

class StreamingCompressor:
    """Online streaming compressor that recompresses as new data arrives.

    Maintains a sliding window of the most recent data and periodically
    recompresses when enough new data has accumulated.

    Args:
        config: CompressionPipelineConfig.
    """

    def __init__(self, config: CompressionPipelineConfig):
        self.config = config
        self._buffer: List[np.ndarray] = []
        self._buffer_size = 0
        self._compressed: Optional[CompressedMatrix] = None
        self._history: List[CompressionRecord] = []
        self._step = 0

    def update(
        self,
        new_data: np.ndarray,
        force_recompress: bool = False,
    ) -> Optional[CompressedMatrix]:
        """Add new data to the buffer and optionally recompress.

        Args:
            new_data: New data to add (single matrix or array).
            force_recompress: Recompress immediately regardless of buffer size.

        Returns:
            Updated CompressedMatrix if recompression occurred, else None.
        """
        self._buffer.append(np.asarray(new_data))
        self._buffer_size += 1
        self._step += 1

        # Prune buffer to window size
        while self._buffer_size > self.config.streaming_window:
            self._buffer.pop(0)
            self._buffer_size -= 1

        should_compress = (
            force_recompress
            or self._buffer_size >= self.config.streaming_window
            or (self._step % max(1, self.config.streaming_overlap) == 0 and self._buffer_size >= 10)
        )

        if should_compress:
            return self._recompress()
        return None

    def _recompress(self) -> CompressedMatrix:
        """Recompress current buffer contents."""
        t0 = time.time()
        stacked = np.stack(self._buffer) if len(self._buffer) > 1 else self._buffer[0][np.newaxis]

        # Flatten to 2D for compression
        flat = stacked.reshape(len(stacked), -1)

        compressed = compress_matrix_to_tt(
            flat,
            max_rank=self.config.max_rank,
            method=self.config.rank_method,
            target_variance=self.config.target_variance,
            prune_threshold=self.config.prune_threshold,
            verbose=self.config.verbose,
        )

        elapsed = time.time() - t0

        record = CompressionRecord(
            timestamp=time.time(),
            input_shape=flat.shape,
            n_params_original=flat.size,
            n_params_compressed=compressed.n_params,
            compression_ratio=compressed.compression_ratio,
            reconstruction_error=compressed.reconstruction_error,
            rank_used=compressed.ranks[0] if compressed.ranks else 0,
            method=self.config.rank_method,
            elapsed_seconds=elapsed,
        )

        if self.config.store_history:
            self._history.append(record)
            if len(self._history) > self.config.max_history:
                self._history.pop(0)

        self._compressed = compressed

        if (
            self.config.alert_on_fidelity
            and compressed.reconstruction_error > self.config.fidelity_threshold
        ):
            self._trigger_fidelity_alert(compressed.reconstruction_error)

        return compressed

    def _trigger_fidelity_alert(self, error: float) -> None:
        """Trigger fidelity degradation alert."""
        msg = (
            f"[TensorNet] Compression fidelity alert: "
            f"error={error:.6f} exceeds threshold={self.config.fidelity_threshold:.4f}"
        )
        logger.warning(msg)
        if self.config.verbose:
            print(f"WARNING: {msg}")

    @property
    def compressed(self) -> Optional[CompressedMatrix]:
        """Current compressed representation."""
        return self._compressed

    @property
    def history(self) -> List[CompressionRecord]:
        """Full compression history."""
        return self._history

    def get_current_dense(self) -> Optional[np.ndarray]:
        """Get the current compressed matrix as dense."""
        if self._compressed is None:
            return None
        return decompress_matrix(self._compressed, self.config.decompress_dtype)


# ============================================================================
# End-to-end pipeline
# ============================================================================

class CompressionPipeline:
    """End-to-end correlation matrix compression pipeline.

    Steps:
      1. Ingest correlation/covariance matrix
      2. Determine optimal rank via information criterion
      3. Compress to TT format
      4. Monitor reconstruction fidelity
      5. Decompress on demand

    Args:
        config: CompressionPipelineConfig.
    """

    def __init__(self, config: Optional[CompressionPipelineConfig] = None):
        self.config = config or CompressionPipelineConfig()
        self._compressed_store: Dict[str, CompressedMatrix] = {}
        self._streaming: Dict[str, StreamingCompressor] = {}
        self._history: List[CompressionRecord] = []

    def compress(
        self,
        matrix: np.ndarray,
        name: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CompressedMatrix:
        """Compress a matrix and store it.

        Args:
            matrix: Input correlation/covariance matrix.
            name: Identifier for this matrix (for retrieval).
            metadata: Optional metadata to attach.

        Returns:
            CompressedMatrix.
        """
        t0 = time.time()
        compressed = compress_matrix_to_tt(
            matrix,
            max_rank=self.config.max_rank,
            method=self.config.rank_method,
            target_variance=self.config.target_variance,
            prune_threshold=self.config.prune_threshold,
            verbose=self.config.verbose,
        )
        elapsed = time.time() - t0

        record = CompressionRecord(
            timestamp=time.time(),
            input_shape=matrix.shape,
            n_params_original=matrix.size,
            n_params_compressed=compressed.n_params,
            compression_ratio=compressed.compression_ratio,
            reconstruction_error=compressed.reconstruction_error,
            rank_used=compressed.ranks[0] if compressed.ranks else 0,
            method=self.config.rank_method,
            elapsed_seconds=elapsed,
            metadata=metadata or {},
        )

        if self.config.store_history:
            self._history.append(record)

        self._compressed_store[name] = compressed

        if (
            self.config.alert_on_fidelity
            and compressed.reconstruction_error > self.config.fidelity_threshold
        ):
            self._fidelity_alert(name, compressed.reconstruction_error)

        if self.config.verbose:
            print(
                f"Compressed '{name}': "
                f"{compressed.compression_ratio:.2f}x, error={compressed.reconstruction_error:.6f}"
            )

        return compressed

    def decompress(
        self,
        name: str = "default",
        dtype: str = "float32",
    ) -> np.ndarray:
        """Decompress a stored matrix.

        Args:
            name: Identifier of the compressed matrix.
            dtype: Output dtype.

        Returns:
            Dense matrix.
        """
        if name not in self._compressed_store:
            raise KeyError(f"No compressed matrix '{name}' found.")
        return decompress_matrix(self._compressed_store[name], dtype=dtype)

    def update_streaming(
        self,
        new_matrix: np.ndarray,
        name: str = "default",
    ) -> Optional[CompressedMatrix]:
        """Feed a new matrix into the streaming compressor.

        Args:
            new_matrix: Latest correlation/covariance matrix.
            name: Stream identifier.

        Returns:
            Updated CompressedMatrix if recompression triggered, else None.
        """
        if name not in self._streaming:
            self._streaming[name] = StreamingCompressor(self.config)

        compressed = self._streaming[name].update(new_matrix)
        if compressed is not None:
            self._compressed_store[name] = compressed
        return compressed

    def _fidelity_alert(self, name: str, error: float) -> None:
        msg = (
            f"Fidelity alert for '{name}': error={error:.6f} "
            f"> threshold={self.config.fidelity_threshold:.4f}"
        )
        logger.warning(msg)
        if self.config.verbose:
            print(f"WARNING: {msg}")

    def batch_compress(
        self,
        matrices: Dict[str, np.ndarray],
    ) -> Dict[str, CompressedMatrix]:
        """Compress multiple named matrices.

        Args:
            matrices: Dict mapping name -> matrix.

        Returns:
            Dict mapping name -> CompressedMatrix.
        """
        results = {}
        for name, mat in matrices.items():
            results[name] = self.compress(mat, name=name)
        return results

    def monitor(self) -> Dict[str, Any]:
        """Return current monitoring dashboard data.

        Returns:
            Dict with compression ratios, errors, and history stats.
        """
        stored = {}
        for name, cm in self._compressed_store.items():
            stored[name] = {
                "compression_ratio": cm.compression_ratio,
                "reconstruction_error": cm.reconstruction_error,
                "ranks": cm.ranks,
                "n_params_compressed": cm.n_params,
                "timestamp": cm.timestamp,
            }

        history_stats: Dict[str, Any] = {}
        if self._history:
            ratios = [r.compression_ratio for r in self._history]
            errors = [r.reconstruction_error for r in self._history]
            history_stats = {
                "n_compressions": len(self._history),
                "mean_ratio": float(np.mean(ratios)),
                "mean_error": float(np.mean(errors)),
                "max_error": float(np.max(errors)),
                "min_error": float(np.min(errors)),
                "fidelity_alerts": sum(
                    1 for e in errors if e > self.config.fidelity_threshold
                ),
            }

        return {
            "stored_matrices": stored,
            "history_stats": history_stats,
            "config": {
                "max_rank": self.config.max_rank,
                "rank_method": self.config.rank_method,
                "fidelity_threshold": self.config.fidelity_threshold,
            },
        }

    def dashboard(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """Display compression ratio dashboard.

        Args:
            save_path: Optional file path to save figure.
            show: Whether to call plt.show().
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available; skipping dashboard.")
            return

        if not self._history:
            print("No compression history to display.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Compression Pipeline Dashboard", fontsize=14)

        times = [r.timestamp - self._history[0].timestamp for r in self._history]
        ratios = [r.compression_ratio for r in self._history]
        errors = [r.reconstruction_error for r in self._history]
        ranks = [r.rank_used for r in self._history]
        snrs = [r.snr_db for r in self._history]

        # Compression ratio over time
        ax = axes[0, 0]
        ax.plot(times, ratios, "b-o", markersize=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Compression Ratio")
        ax.set_title("Compression Ratio Over Time")
        ax.grid(True, alpha=0.3)

        # Reconstruction error over time
        ax = axes[0, 1]
        ax.semilogy(times, errors, "r-o", markersize=3)
        ax.axhline(self.config.fidelity_threshold, color="orange", linestyle="--",
                   label=f"Alert threshold={self.config.fidelity_threshold}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Reconstruction Error")
        ax.set_title("Fidelity Monitor")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # SNR
        ax = axes[1, 0]
        finite_snrs = [s for s in snrs if np.isfinite(s)]
        if finite_snrs:
            ax.plot(times[:len(finite_snrs)], finite_snrs, "g-o", markersize=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SNR (dB)")
        ax.set_title("Signal-to-Noise Ratio")
        ax.grid(True, alpha=0.3)

        # Rank over time
        ax = axes[1, 1]
        ax.step(times, ranks, "m-", where="post")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Rank Used")
        ax.set_title("Selected Rank Over Time")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def export_history_json(self, path: str) -> None:
        """Export compression history to a JSON file.

        Args:
            path: Output file path.
        """
        records = []
        for r in self._history:
            records.append({
                "timestamp": r.timestamp,
                "input_shape": list(r.input_shape),
                "n_params_original": r.n_params_original,
                "n_params_compressed": r.n_params_compressed,
                "compression_ratio": r.compression_ratio,
                "reconstruction_error": r.reconstruction_error,
                "rank_used": r.rank_used,
                "method": r.method,
                "elapsed_seconds": r.elapsed_seconds,
                "snr_db": r.snr_db,
            })
        with open(path, "w") as f:
            json.dump(records, f, indent=2)


# ============================================================================
# Differential rank-1 update
# ============================================================================

def rank1_update_compressed(
    compressed: CompressedMatrix,
    u: np.ndarray,
    v: np.ndarray,
    alpha: float = 1.0,
) -> CompressedMatrix:
    """Apply a rank-1 update to a compressed matrix.

    Computes A' = A + alpha * u @ v^T without full decompression.
    Decompresses, updates, and recompresses.

    Args:
        compressed: Existing CompressedMatrix.
        u: Left vector (n,).
        v: Right vector (m,).
        alpha: Scaling factor.

    Returns:
        Updated CompressedMatrix.
    """
    dense = decompress_matrix(compressed, dtype="float64")
    n, m = dense.shape[0], dense.shape[1] if dense.ndim > 1 else 1
    dense_2d = dense.reshape(n, -1)

    u_arr = np.asarray(u, dtype=np.float64).reshape(-1, 1)
    v_arr = np.asarray(v, dtype=np.float64).reshape(1, -1)

    if u_arr.shape[0] == dense_2d.shape[0] and v_arr.shape[1] == dense_2d.shape[1]:
        updated = dense_2d + alpha * (u_arr @ v_arr)
    else:
        warnings.warn("rank1_update dimension mismatch; skipping update.")
        updated = dense_2d

    return compress_matrix_to_tt(updated.reshape(compressed.original_shape))


# ============================================================================
# Quality metrics
# ============================================================================

def compression_snr(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """Signal-to-noise ratio of compressed reconstruction.

    SNR = 10 * log10(||original||^2 / ||original - reconstructed||^2)

    Args:
        original: Original matrix.
        reconstructed: Reconstructed matrix.

    Returns:
        SNR in dB.
    """
    signal_power = float(np.sum(original ** 2))
    noise_power = float(np.sum((original - reconstructed) ** 2))
    if noise_power < 1e-15:
        return float("inf")
    return 10.0 * math.log10(signal_power / noise_power)


def compression_ssim_proxy(
    original: np.ndarray,
    reconstructed: np.ndarray,
    c1: float = 0.01,
    c2: float = 0.03,
) -> float:
    """Proxy SSIM-like structural similarity for matrices.

    Computes a simplified SSIM between two matrices by treating them
    as flattened signals.

    Args:
        original: Original matrix.
        reconstructed: Reconstructed matrix.
        c1, c2: Stability constants.

    Returns:
        Structural similarity score in [0, 1].
    """
    x = original.reshape(-1).astype(float)
    y = reconstructed.reshape(-1).astype(float)

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.std()
    sigma_y = y.std()
    sigma_xy = float(np.mean((x - mu_x) * (y - mu_y)))

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2)
    return float(numerator / (denominator + 1e-15))


def full_quality_report(
    original: np.ndarray,
    compressed: CompressedMatrix,
) -> Dict[str, float]:
    """Compute full quality metrics for a compressed matrix.

    Args:
        original: Original matrix.
        compressed: CompressedMatrix.

    Returns:
        Dict with snr_db, relative_error, ssim, compression_ratio.
    """
    recon = decompress_matrix(compressed)
    orig_arr = original.reshape(recon.shape) if original.size == recon.size else original

    return {
        "snr_db": compression_snr(orig_arr, recon),
        "relative_error": float(
            np.linalg.norm(orig_arr - recon) / (np.linalg.norm(orig_arr) + 1e-15)
        ),
        "ssim_proxy": compression_ssim_proxy(orig_arr, recon),
        "compression_ratio": compressed.compression_ratio,
        "n_params_original": original.size,
        "n_params_compressed": compressed.n_params,
    }


# ============================================================================
# Adaptive compression with fidelity tracking
# ============================================================================

class AdaptiveCompressionController:
    """Controller that dynamically adjusts rank based on fidelity feedback.

    Monitors reconstruction quality over time and adjusts the compression
    rank to maintain a target fidelity level.

    Args:
        target_error: Target reconstruction error (lower = better fidelity).
        min_rank: Minimum compression rank.
        max_rank: Maximum compression rank.
        rank_step: How much to increase/decrease rank at each adjustment.
        patience: Steps to wait before adjusting rank.
        config: Optional CompressionPipelineConfig.
    """

    def __init__(
        self,
        target_error: float = 0.01,
        min_rank: int = 2,
        max_rank: int = 64,
        rank_step: int = 2,
        patience: int = 10,
        config: Optional[CompressionPipelineConfig] = None,
    ):
        self.target_error = target_error
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.rank_step = rank_step
        self.patience = patience
        self._current_rank = min_rank
        self._step = 0
        self._error_buffer: List[float] = []
        self._rank_history: List[Tuple[int, int]] = []  # (step, rank)
        self.config = config or CompressionPipelineConfig(max_rank=max_rank)

    def compress_and_adapt(
        self,
        matrix: np.ndarray,
        name: str = "adaptive",
    ) -> Tuple["CompressedMatrix", Dict[str, Any]]:
        """Compress a matrix and adapt the rank based on fidelity.

        Args:
            matrix: Input matrix to compress.
            name: Identifier for this matrix.

        Returns:
            (CompressedMatrix, adaptation_info) tuple.
        """
        compressed = compress_matrix_to_tt(
            matrix,
            max_rank=self._current_rank,
            method=self.config.rank_method,
        )
        error = compressed.reconstruction_error
        self._error_buffer.append(error)
        self._step += 1

        adaptation_info: Dict[str, Any] = {
            "step": self._step,
            "rank": self._current_rank,
            "error": error,
            "rank_changed": False,
        }

        if len(self._error_buffer) >= self.patience:
            mean_error = float(np.mean(self._error_buffer[-self.patience:]))

            if mean_error > self.target_error * 1.1:
                # Error too high — increase rank
                new_rank = min(self._current_rank + self.rank_step, self.max_rank)
                if new_rank != self._current_rank:
                    self._current_rank = new_rank
                    self._rank_history.append((self._step, new_rank))
                    adaptation_info["rank_changed"] = True
                    adaptation_info["rank"] = new_rank
                    if self.config.verbose:
                        print(f"  AdaptiveController: rank increased to {new_rank} (error={mean_error:.5f})")

            elif mean_error < self.target_error * 0.5:
                # Error much lower than target — decrease rank (save space)
                new_rank = max(self._current_rank - self.rank_step, self.min_rank)
                if new_rank != self._current_rank:
                    self._current_rank = new_rank
                    self._rank_history.append((self._step, new_rank))
                    adaptation_info["rank_changed"] = True
                    adaptation_info["rank"] = new_rank
                    if self.config.verbose:
                        print(f"  AdaptiveController: rank decreased to {new_rank} (error={mean_error:.5f})")

        return compressed, adaptation_info

    @property
    def current_rank(self) -> int:
        return self._current_rank

    @property
    def rank_history(self) -> List[Tuple[int, int]]:
        return self._rank_history

    def summary(self) -> Dict[str, Any]:
        errors = self._error_buffer
        return {
            "current_rank": self._current_rank,
            "n_steps": self._step,
            "n_rank_changes": len(self._rank_history),
            "mean_error": float(np.mean(errors)) if errors else 0.0,
            "min_error": float(np.min(errors)) if errors else 0.0,
            "max_error": float(np.max(errors)) if errors else 0.0,
        }


# ============================================================================
# Tucker compression pipeline
# ============================================================================

def compress_tensor_tucker(
    tensor: np.ndarray,
    target_variance: float = 0.99,
    max_rank_per_mode: Optional[int] = None,
) -> Dict[str, Any]:
    """Compress an N-dimensional tensor using Tucker decomposition.

    Args:
        tensor: N-dimensional input array.
        target_variance: Variance explained per mode.
        max_rank_per_mode: Optional cap on mode rank.

    Returns:
        Dict with core_tensor, factor_matrices, ranks, compression_ratio, error.
    """
    from tensor_net.rank_selection import tucker_rank_per_mode, _tucker_core, _mode_unfold

    profile = tucker_rank_per_mode(
        tensor, target_variance=target_variance, max_rank_per_mode=max_rank_per_mode
    )

    # Compute Tucker factors
    factor_matrices = []
    for mode in range(tensor.ndim):
        unfolded = _mode_unfold(tensor, mode)
        U, _, _ = np.linalg.svd(unfolded, full_matrices=False)
        r = profile.mode_ranks[mode]
        factor_matrices.append(U[:, :r])

    # Compute Tucker core
    core = _tucker_core(tensor, profile.mode_ranks)

    # Reconstruction
    recon = core
    for mode, F in enumerate(factor_matrices):
        recon_unfolded = _mode_unfold(recon, mode)
        projected = F @ recon_unfolded.reshape(F.shape[1], -1)
        recon = np.moveaxis(
            projected.reshape([F.shape[0]] + [recon.shape[i] for i in range(recon.ndim) if i != mode]),
            0, mode,
        )

    orig_norm = float(np.linalg.norm(tensor))
    recon_error = float(np.linalg.norm(tensor - recon)) / (orig_norm + 1e-15)
    orig_size = tensor.size
    compressed_size = core.size + sum(F.size for F in factor_matrices)
    ratio = orig_size / max(1, compressed_size)

    return {
        "core_tensor": core,
        "factor_matrices": factor_matrices,
        "ranks": profile.mode_ranks,
        "compression_ratio": ratio,
        "reconstruction_error": recon_error,
        "profile": profile,
    }


# ============================================================================
# Correlation matrix compression suite
# ============================================================================

class CorrelationCompressionSuite:
    """Comprehensive suite for compressing financial correlation matrices.

    Provides multiple compression strategies (TT, Tucker, Eigendecomp)
    and automatic selection of the best method per matrix.

    Args:
        max_rank: Maximum rank.
        target_variance: Target variance for Tucker.
        prefer_method: "auto", "tt", "tucker", or "eigen".
    """

    def __init__(
        self,
        max_rank: int = 32,
        target_variance: float = 0.99,
        prefer_method: str = "auto",
    ):
        self.max_rank = max_rank
        self.target_variance = target_variance
        self.prefer_method = prefer_method
        self._pipeline = CompressionPipeline(
            CompressionPipelineConfig(max_rank=max_rank)
        )

    def compress_correlation(
        self,
        corr_matrix: np.ndarray,
        name: str = "corr",
        return_all: bool = False,
    ) -> Dict[str, Any]:
        """Compress a correlation matrix using the best available method.

        Args:
            corr_matrix: Symmetric correlation matrix.
            name: Identifier.
            return_all: If True, return results from all methods.

        Returns:
            Dict with compressed representation and quality metrics.
        """
        corr = np.asarray(corr_matrix, dtype=np.float64)
        n = corr.shape[0]

        # Strategy: TT compression
        tt_result = self._pipeline.compress(corr, name=f"{name}_tt")
        tt_error = tt_result.reconstruction_error
        tt_ratio = tt_result.compression_ratio

        # Strategy: Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Find rank for variance explained
        ev_sq = np.maximum(eigenvalues, 0)
        total = ev_sq.sum()
        cumulative = np.cumsum(ev_sq) / (total + 1e-15)
        rank_eigen = int(np.searchsorted(cumulative, self.target_variance)) + 1
        rank_eigen = min(rank_eigen, min(self.max_rank, n))
        rank_eigen = max(1, rank_eigen)

        U_r = eigenvectors[:, :rank_eigen]
        s_r = eigenvalues[:rank_eigen]
        recon_eigen = (U_r * s_r) @ U_r.T
        eigen_error = float(np.linalg.norm(corr - recon_eigen) / (np.linalg.norm(corr) + 1e-15))
        eigen_ratio = n * n / max(1, n * rank_eigen + rank_eigen)

        # Select best method
        if self.prefer_method == "auto":
            if tt_error <= eigen_error:
                best_method = "tt"
            else:
                best_method = "eigen"
        else:
            best_method = self.prefer_method

        result: Dict[str, Any] = {
            "name": name,
            "n_assets": n,
            "best_method": best_method,
            "tt_error": tt_error,
            "tt_ratio": tt_ratio,
            "tt_rank": self.max_rank,
            "eigen_error": eigen_error,
            "eigen_ratio": eigen_ratio,
            "eigen_rank": rank_eigen,
        }

        if best_method == "tt":
            result["compressed"] = tt_result
        else:
            result["eigenvectors"] = U_r
            result["eigenvalues"] = s_r
            result["reconstruction_error"] = eigen_error
            result["compression_ratio"] = eigen_ratio

        if return_all:
            result["all_methods"] = {
                "tt": {"error": tt_error, "ratio": tt_ratio},
                "eigen": {"error": eigen_error, "ratio": eigen_ratio},
            }

        return result

    def batch_compress_correlations(
        self,
        matrices: List[np.ndarray],
        names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Compress a list of correlation matrices.

        Args:
            matrices: List of correlation matrices.
            names: Optional list of names.

        Returns:
            List of compression result dicts.
        """
        if names is None:
            names = [f"corr_{i}" for i in range(len(matrices))]

        return [
            self.compress_correlation(mat, name=n)
            for mat, n in zip(matrices, names)
        ]

    def monitor_drift(
        self,
        new_corr: np.ndarray,
        reference_name: str,
    ) -> Dict[str, float]:
        """Compare a new correlation matrix to the stored reference.

        Args:
            new_corr: New correlation matrix.
            reference_name: Name of the stored reference.

        Returns:
            Dict with frobenius_diff, spectral_diff, rank_correlation.
        """
        try:
            ref = self._pipeline.decompress(f"{reference_name}_tt")
        except KeyError:
            return {"error": "reference_not_found"}

        new = np.asarray(new_corr, dtype=np.float64)
        diff = new - ref
        frob_diff = float(np.linalg.norm(diff))
        spectral_diff = float(np.linalg.norm(diff, ord=2))

        # Rank correlation of off-diagonal elements
        mask = ~np.eye(new.shape[0], dtype=bool)
        new_off = new[mask]
        ref_off = ref[mask]
        if len(new_off) > 1:
            rho = float(np.corrcoef(new_off, ref_off)[0, 1])
        else:
            rho = 1.0

        return {
            "frobenius_diff": frob_diff,
            "spectral_diff": spectral_diff,
            "rank_correlation": rho,
            "n_assets": new.shape[0],
        }


# ============================================================================
# Rolling compression for time-series of correlation matrices
# ============================================================================

class RollingCorrelationCompressor:
    """Compress a rolling sequence of correlation matrices efficiently.

    Avoids re-running full rank selection for each window by reusing
    the previously determined rank unless fidelity degrades.

    Args:
        initial_rank: Starting rank.
        max_rank: Maximum rank.
        fidelity_threshold: Alert when error exceeds this.
        rerank_every: Re-run rank selection every N matrices.
    """

    def __init__(
        self,
        initial_rank: int = 8,
        max_rank: int = 32,
        fidelity_threshold: float = 0.05,
        rerank_every: int = 50,
    ):
        self.initial_rank = initial_rank
        self.max_rank = max_rank
        self.fidelity_threshold = fidelity_threshold
        self.rerank_every = rerank_every

        self._current_rank = initial_rank
        self._compressed_history: List[CompressedMatrix] = []
        self._error_history: List[float] = []
        self._step = 0

    def update(
        self,
        corr_matrix: np.ndarray,
    ) -> CompressedMatrix:
        """Compress the next correlation matrix in the sequence.

        Args:
            corr_matrix: Correlation matrix (n, n).

        Returns:
            CompressedMatrix.
        """
        self._step += 1

        # Periodically re-select rank
        if self._step % self.rerank_every == 1:
            from tensor_net.rank_selection import InformationTheoreticRankSelector
            mat = np.asarray(corr_matrix, dtype=np.float64)
            selector = InformationTheoreticRankSelector(
                mat.reshape(1, -1) if mat.ndim > 1 else mat,
                max_rank=self.max_rank,
            )
            self._current_rank = selector.rank_by_bic()

        compressed = compress_matrix_to_tt(
            corr_matrix,
            max_rank=self._current_rank,
        )

        self._compressed_history.append(compressed)
        self._error_history.append(compressed.reconstruction_error)

        if compressed.reconstruction_error > self.fidelity_threshold:
            # Increase rank
            self._current_rank = min(self._current_rank + 2, self.max_rank)
            logger.warning(
                f"RollingCompressor: fidelity degraded at step {self._step}, "
                f"increasing rank to {self._current_rank}"
            )

        # Keep history bounded
        if len(self._compressed_history) > 1000:
            self._compressed_history.pop(0)
            self._error_history.pop(0)

        return compressed

    @property
    def latest(self) -> Optional[CompressedMatrix]:
        if not self._compressed_history:
            return None
        return self._compressed_history[-1]

    def error_trend(self, window: int = 20) -> float:
        """Compute recent trend in reconstruction error.

        Positive = worsening, negative = improving.

        Args:
            window: Window size for trend computation.

        Returns:
            Trend slope (linear regression over recent errors).
        """
        if len(self._error_history) < window:
            return 0.0
        recent = np.array(self._error_history[-window:])
        x = np.arange(window, dtype=float)
        slope = float(np.polyfit(x, recent, 1)[0])
        return slope

    def summary(self) -> Dict[str, Any]:
        return {
            "current_rank": self._current_rank,
            "n_compressed": self._step,
            "mean_error": float(np.mean(self._error_history)) if self._error_history else 0.0,
            "latest_error": self._error_history[-1] if self._error_history else 0.0,
            "error_trend": self.error_trend(),
        }


# ============================================================================
# Wavelet-inspired multi-scale compression
# ============================================================================

def multi_scale_compress(
    matrix: np.ndarray,
    n_scales: int = 3,
    base_rank: int = 4,
    verbose: bool = False,
) -> List[CompressedMatrix]:
    """Multi-scale compression using decreasing rank at each scale.

    Compresses residuals at progressively lower ranks, analogous to
    a wavelet decomposition.

    Args:
        matrix: Input matrix.
        n_scales: Number of scales.
        base_rank: Rank at the finest scale.
        verbose: Print diagnostics.

    Returns:
        List of CompressedMatrix objects, one per scale (coarsest to finest).
    """
    mat = np.asarray(matrix, dtype=np.float64)
    residual = mat.copy()
    compressed_scales = []

    for scale in range(n_scales):
        rank = max(1, base_rank // (2 ** scale))
        c = compress_matrix_to_tt(residual, max_rank=rank, verbose=verbose)
        compressed_scales.insert(0, c)

        # Subtract reconstruction from residual
        recon = decompress_matrix(c, dtype="float64")
        residual = residual - recon.reshape(residual.shape)

        if verbose:
            print(f"  Scale {scale}: rank={rank}, error={c.reconstruction_error:.6f}")

    return compressed_scales


def multi_scale_decompress(
    compressed_scales: List[CompressedMatrix],
) -> np.ndarray:
    """Reconstruct matrix from multi-scale compressed representation.

    Args:
        compressed_scales: List from multi_scale_compress (coarsest to finest).

    Returns:
        Reconstructed matrix.
    """
    result = None
    for c in compressed_scales:
        recon = decompress_matrix(c, dtype="float64")
        if result is None:
            result = recon
        else:
            result = result + recon.reshape(result.shape)
    return result.astype(np.float32) if result is not None else np.zeros((1,))


# ============================================================================
# Compression benchmark utilities
# ============================================================================

def benchmark_compression_methods(
    matrix: np.ndarray,
    ranks: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Benchmark multiple compression configurations on a single matrix.

    Args:
        matrix: Input matrix.
        ranks: List of ranks to test. Defaults to [2, 4, 8, 16, 32].
        verbose: Print progress.

    Returns:
        Benchmark results dict.
    """
    import time
    if ranks is None:
        ranks = [2, 4, 8, 16, 32]

    results: Dict[str, Any] = {
        "matrix_shape": matrix.shape,
        "matrix_size": matrix.size,
        "by_rank": {},
    }

    for rank in ranks:
        if verbose:
            print(f"  Benchmarking rank={rank} ...")
        t0 = time.time()
        compressed = compress_matrix_to_tt(matrix, max_rank=rank)
        elapsed = time.time() - t0

        t1 = time.time()
        recon = decompress_matrix(compressed)
        decompress_time = time.time() - t1

        report = full_quality_report(matrix, compressed)

        results["by_rank"][rank] = {
            **report,
            "compress_time_ms": elapsed * 1000,
            "decompress_time_ms": decompress_time * 1000,
        }

    # Find best rank by SNR
    best_rank = max(ranks, key=lambda r: results["by_rank"][r]["snr_db"])
    results["best_rank_by_snr"] = best_rank
    results["best_rank_by_ratio"] = max(ranks, key=lambda r: results["by_rank"][r]["compression_ratio"])

    return results


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

