"""
AETERNUS Real-Time Execution Layer (RTEL)
tensor_utils.py — Tensor compression utilities for TensorNet integration

Implements Tensor-Train (TT) decomposition and reconstruction for
compressing multi-asset volatility surfaces and feature tensors.
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TT decomposition
# ---------------------------------------------------------------------------

def tt_svd(tensor: np.ndarray,
           max_rank: int = 8,
           rel_error: float = 1e-4) -> List[np.ndarray]:
    """
    Tensor-Train SVD decomposition (TT-SVD algorithm).
    Returns list of TT-cores [G1, G2, ..., Gd].
    Each core Gk has shape [r_{k-1}, n_k, r_k].
    """
    shape = tensor.shape
    d     = len(shape)
    if d == 0:
        return []

    delta   = rel_error / math.sqrt(d - 1) * np.linalg.norm(tensor)
    cores   = []
    C       = tensor.copy().astype(np.float64)
    r_prev  = 1

    for k in range(d - 1):
        C = C.reshape(r_prev * shape[k], -1)
        # Truncated SVD
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        # Determine truncation rank
        cumulative = np.cumsum(S[::-1] ** 2)[::-1]
        rank = np.searchsorted(cumulative[::-1], delta ** 2) + 1
        rank = min(max(1, rank), max_rank, len(S))
        # Truncate
        U = U[:, :rank]
        S = S[:rank]
        Vt= Vt[:rank, :]
        # Store core
        core = U.reshape(r_prev, shape[k], rank)
        cores.append(core.astype(np.float32))
        # Remainder
        C = np.diag(S) @ Vt
        r_prev = rank

    # Last core
    cores.append(C.reshape(r_prev, shape[-1], 1).astype(np.float32))
    return cores


def tt_reconstruct(cores: List[np.ndarray]) -> np.ndarray:
    """Reconstruct tensor from TT-cores."""
    if not cores:
        return np.array([])
    # Start with first core [1, n0, r0] → [n0, r0]
    result = cores[0][0]  # shape [n0, r0]
    for k in range(1, len(cores)):
        # cores[k]: [r_{k-1}, n_k, r_k]
        r_prev, n_k, r_k = cores[k].shape
        # result: [*dims, r_{k-1}]
        # contract over r_{k-1}
        shape_in = result.shape
        result = result.reshape(-1, r_prev)  # [*, r_{k-1}]
        # [*, r_{k-1}] @ [r_{k-1}, n_k * r_k] → [*, n_k * r_k]
        mat = cores[k].reshape(r_prev, n_k * r_k)
        result = result @ mat  # [*, n_k * r_k]
        result = result.reshape(*shape_in[:-1], n_k, r_k) if len(shape_in) > 1 \
            else result.reshape(n_k, r_k)
    # Remove last rank-1 dimension
    return result[..., 0] if result.ndim > 1 else result


def tt_ranks(cores: List[np.ndarray]) -> List[int]:
    """Return list of TT-ranks [r0, r1, ..., r_{d-1}]."""
    if not cores:
        return []
    return [int(c.shape[2]) for c in cores[:-1]]


def tt_compression_ratio(original_shape: Tuple[int, ...],
                          cores: List[np.ndarray]) -> float:
    """Compression ratio: original_size / tt_size."""
    original = math.prod(original_shape)
    tt_size  = sum(math.prod(c.shape) for c in cores)
    return original / max(1, tt_size)


def tt_reconstruction_error(original: np.ndarray,
                              cores: List[np.ndarray]) -> float:
    """Relative Frobenius reconstruction error."""
    recon = tt_reconstruct(cores)
    diff  = original.astype(np.float64) - recon.astype(np.float64)
    norm  = np.linalg.norm(original)
    return float(np.linalg.norm(diff) / (norm + 1e-10))


def tt_add(cores_a: List[np.ndarray],
           cores_b: List[np.ndarray]) -> List[np.ndarray]:
    """Sum of two TT tensors."""
    assert len(cores_a) == len(cores_b)
    result = []
    d = len(cores_a)
    for k in range(d):
        a, b = cores_a[k], cores_b[k]
        ra, na, ra_ = a.shape
        rb, nb, rb_ = b.shape
        assert na == nb
        if k == 0:
            core = np.concatenate([a, b], axis=2)  # [1, n, ra+rb]
        elif k == d - 1:
            core = np.concatenate([a, b], axis=0)  # [ra+rb, n, 1]
        else:
            # Block diagonal
            r_in  = ra + rb
            r_out = ra_ + rb_
            core = np.zeros((r_in, na, r_out), dtype=a.dtype)
            core[:ra, :, :ra_] = a
            core[ra:, :, ra_:] = b
        result.append(core)
    return result


def tt_rounding(cores: List[np.ndarray], max_rank: int = 8,
                tol: float = 1e-4) -> List[np.ndarray]:
    """TT rounding: recompress TT cores to lower ranks."""
    tensor = tt_reconstruct(cores)
    return tt_svd(tensor, max_rank=max_rank, rel_error=tol)


# ---------------------------------------------------------------------------
# Multi-asset vol surface compression
# ---------------------------------------------------------------------------

class VolSurfaceCompressor:
    """
    Compresses batches of volatility surfaces using TT-SVD.
    Operates on [n_assets × n_strikes × n_expiries] tensors.
    """

    def __init__(self, max_rank: int = 6, rel_error: float = 1e-3):
        self.max_rank  = max_rank
        self.rel_error = rel_error
        self._cores_cache: dict = {}

    def compress(self, vol_tensor: np.ndarray
                 ) -> Tuple[List[np.ndarray], float, float]:
        """
        Compress vol_tensor [n_assets × K × T].
        Returns (cores, compression_ratio, reconstruction_error).
        """
        cores = tt_svd(vol_tensor, self.max_rank, self.rel_error)
        ratio = tt_compression_ratio(vol_tensor.shape, cores)
        error = tt_reconstruction_error(vol_tensor, cores)
        return cores, ratio, error

    def decompress(self, cores: List[np.ndarray],
                   shape: Tuple[int, ...]) -> np.ndarray:
        """Reconstruct vol tensor from cores."""
        tensor = tt_reconstruct(cores)
        return tensor.reshape(shape).astype(np.float32)

    def compress_and_cache(self, asset_id: int,
                            vol_surface: np.ndarray) -> float:
        """Compress a single asset vol surface and cache cores. Returns ratio."""
        cores, ratio, err = self.compress(vol_surface)
        self._cores_cache[asset_id] = (cores, vol_surface.shape)
        logger.debug("Asset %d: vol compressed %.1fx, error=%.2e", asset_id, ratio, err)
        return ratio

    def get_cached(self, asset_id: int) -> Optional[np.ndarray]:
        if asset_id not in self._cores_cache:
            return None
        cores, shape = self._cores_cache[asset_id]
        return self.decompress(cores, shape)

    def batch_compress(self, vol_batch: np.ndarray) -> dict:
        """Compress batch [n_assets × K × T]. Returns stats dict."""
        n_assets = vol_batch.shape[0]
        total_ratio = 0.0
        total_error = 0.0
        for i in range(n_assets):
            ratio = self.compress_and_cache(i, vol_batch[i])
            total_ratio += ratio
        return {
            "n_assets":        n_assets,
            "mean_ratio":      total_ratio / max(1, n_assets),
            "mean_error":      total_error / max(1, n_assets),
            "cached_assets":   len(self._cores_cache),
        }


# ---------------------------------------------------------------------------
# Tucker decomposition (for cross-asset tensors)
# ---------------------------------------------------------------------------

def tucker_decompose(tensor: np.ndarray,
                     ranks: Tuple[int, ...]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Higher-order SVD (HOSVD) Tucker decomposition.
    Returns (core_tensor, factor_matrices).
    """
    d = tensor.ndim
    assert len(ranks) == d
    factors = []
    C = tensor.copy().astype(np.float64)

    for mode in range(d):
        # Unfold tensor along mode
        unfolded = np.moveaxis(C, mode, 0).reshape(C.shape[mode], -1)
        U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
        r = min(ranks[mode], U.shape[1])
        U = U[:, :r]
        factors.append(U.astype(np.float32))
        # Project
        C = np.tensordot(C, U.T, axes=([mode], [1]))
        # Move the new mode to position 'mode'
        C = np.moveaxis(C, -1, mode)

    return C.astype(np.float32), factors


def tucker_reconstruct(core: np.ndarray,
                        factors: List[np.ndarray]) -> np.ndarray:
    """Reconstruct tensor from Tucker decomposition."""
    result = core.copy().astype(np.float64)
    for mode, U in enumerate(factors):
        result = np.tensordot(result, U, axes=([0], [1]))
        result = np.moveaxis(result, -1, mode)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# LOB tensor packing
# ---------------------------------------------------------------------------

def pack_lob_tensor(lob_matrix: np.ndarray,
                    history_len: int) -> Optional[np.ndarray]:
    """
    Pack LOB feature matrix history into a 3D tensor.
    Input:  list of [n_assets × feat_dim] arrays of length history_len
    Output: [history_len × n_assets × feat_dim] tensor
    """
    if len(lob_matrix) == 0:
        return None
    if isinstance(lob_matrix, list):
        return np.stack(lob_matrix[-history_len:], axis=0).astype(np.float32)
    return lob_matrix.astype(np.float32)


def compute_tensor_statistics(tensor: np.ndarray) -> dict:
    """Compute statistics of a tensor."""
    return {
        "shape":     tensor.shape,
        "dtype":     str(tensor.dtype),
        "mean":      float(tensor.mean()),
        "std":       float(tensor.std()),
        "min":       float(tensor.min()),
        "max":       float(tensor.max()),
        "nonzero":   int(np.count_nonzero(tensor)),
        "sparsity":  float(1.0 - np.count_nonzero(tensor) / tensor.size),
        "norm_fro":  float(np.linalg.norm(tensor)),
    }
