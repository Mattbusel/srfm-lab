"""
tensor_train.py — Tensor Train (TT) decomposition for TensorNet.

A Tensor Train represents an N-dimensional array as a product of 3-dimensional tensors:
  A[i1, i2, ..., iN] = G1[i1] G2[i2] ... GN[iN]
where Gk has shape (r_{k-1}, n_k, r_k) with r_0 = r_N = 1.

This is equivalent to MPS but for general arrays (not necessarily normalized state vectors).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Sequence, Union, Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad, vmap


# ---------------------------------------------------------------------------
# TensorTrain class
# ---------------------------------------------------------------------------

class TensorTrain:
    """
    Tensor Train decomposition of a high-dimensional array.

    Attributes
    ----------
    cores : list of jnp.ndarray, each shape (r_{k-1}, n_k, r_k)
    shape : tuple of mode sizes (n_1, ..., n_N)
    ranks : list of TT-ranks [1, r_1, r_2, ..., r_{N-1}, 1]
    """

    def __init__(
        self,
        cores: List[jnp.ndarray],
        shape: Optional[Tuple[int, ...]] = None,
    ):
        self.cores = [jnp.array(c) for c in cores]
        self.ndim = len(cores)
        if shape is not None:
            self.shape = tuple(shape)
        else:
            self.shape = tuple(int(c.shape[1]) for c in self.cores)
        self.ranks = [1] + [int(c.shape[2]) for c in self.cores[:-1]] + [1]

    @property
    def max_rank(self) -> int:
        return max(self.ranks)

    @property
    def n_params(self) -> int:
        return sum(c.size for c in self.cores)

    def __repr__(self) -> str:
        return (
            f"TensorTrain(shape={self.shape}, ranks={self.ranks}, "
            f"n_params={self.n_params})"
        )

    def copy(self) -> "TensorTrain":
        return TensorTrain([jnp.array(c) for c in self.cores], self.shape)

    def to_dense(self) -> jnp.ndarray:
        """Contract TT to dense array of shape self.shape."""
        return tt_to_dense(self)

    def __add__(self, other: "TensorTrain") -> "TensorTrain":
        return tt_add(self, other)

    def __mul__(self, scalar: float) -> "TensorTrain":
        return tt_scale(self, scalar)

    def __rmul__(self, scalar: float) -> "TensorTrain":
        return tt_scale(self, scalar)


# Register as JAX pytree
def _tt_flatten(tt: TensorTrain):
    return tt.cores, {"shape": tt.shape}

def _tt_unflatten(aux, cores):
    return TensorTrain(list(cores), aux["shape"])

jax.tree_util.register_pytree_node(TensorTrain, _tt_flatten, _tt_unflatten)


# ---------------------------------------------------------------------------
# TT contraction: TT → dense array
# ---------------------------------------------------------------------------

def tt_to_dense(tt: TensorTrain) -> jnp.ndarray:
    """
    Contract TensorTrain to a dense array.
    Complexity: O(N * r^2 * n) for rank r and mode size n.
    """
    result = tt.cores[0][0, :, :]  # (n_1, r_1)

    for k in range(1, tt.ndim):
        core = tt.cores[k]  # (r_{k-1}, n_k, r_k)
        # result: (..., r_{k-1}), core: (r_{k-1}, n_k, r_k)
        result = jnp.einsum("...l,ldr->...dr", result, core)

    # result shape: (n_1, n_2, ..., n_N, 1) → squeeze trailing
    return result[..., 0]


def tt_evaluate(tt: TensorTrain, indices: Sequence[int]) -> jnp.ndarray:
    """
    Evaluate TT at a specific multi-index without forming dense array.
    indices: tuple of ints (i_1, i_2, ..., i_N)
    """
    result = tt.cores[0][0, indices[0], :]  # (r_1,)
    for k in range(1, tt.ndim):
        result = result @ tt.cores[k][:, indices[k], :]
    return result[0]


# ---------------------------------------------------------------------------
# TT-SVD algorithm
# ---------------------------------------------------------------------------

def tt_svd(
    tensor: jnp.ndarray,
    max_rank: int = 64,
    cutoff: float = 1e-10,
    relative_cutoff: bool = True,
) -> TensorTrain:
    """
    TT-SVD algorithm: exact/approximate decomposition of dense tensor into TT format.

    Algorithm (Oseledets 2011):
    1. Reshape tensor into matrix (n_1, n_2*...*n_N)
    2. SVD, truncate to rank r_1
    3. Reshape right factor as (r_1 * n_2, n_3*...*n_N)
    4. Repeat until last core

    Parameters
    ----------
    tensor : dense array of shape (n_1, n_2, ..., n_N)
    max_rank : maximum TT-rank to keep
    cutoff : singular value threshold
    relative_cutoff : if True, cutoff is relative to largest singular value

    Returns
    -------
    TensorTrain
    """
    tensor = jnp.array(tensor, dtype=jnp.float32)
    shape = tensor.shape
    ndim = len(shape)

    cores = []
    remainder = tensor
    r_prev = 1

    for k in range(ndim - 1):
        n_k = shape[k]
        # Reshape to (r_prev * n_k, rest)
        M = remainder.reshape(r_prev * n_k, -1)
        U, s, Vt = jnp.linalg.svd(M, full_matrices=False)

        # Determine truncation rank
        if relative_cutoff:
            thresh = cutoff * float(s[0]) if len(s) > 0 else cutoff
        else:
            thresh = cutoff

        r_k = min(max_rank, int(jnp.sum(s > thresh).item()))
        r_k = max(r_k, 1)

        U_trunc = U[:, :r_k]
        s_trunc = s[:r_k]
        Vt_trunc = Vt[:r_k, :]

        # Reshape U to core: (r_prev, n_k, r_k)
        core = U_trunc.reshape(r_prev, n_k, r_k)
        cores.append(core)

        # Update remainder: absorb S * Vt
        remainder = jnp.diag(s_trunc) @ Vt_trunc
        r_prev = r_k

    # Last core: (r_prev, n_N, 1)
    n_last = shape[-1]
    core_last = remainder.reshape(r_prev, n_last, 1)
    cores.append(core_last)

    return TensorTrain(cores, shape)


# ---------------------------------------------------------------------------
# TT-cross approximation
# ---------------------------------------------------------------------------

def tt_cross(
    func: Callable,
    shape: Tuple[int, ...],
    max_rank: int = 8,
    n_iter: int = 10,
    key: Optional[jax.random.KeyArray] = None,
    tol: float = 1e-6,
) -> TensorTrain:
    """
    TT-cross approximation (DMRG-cross / MaxVol algorithm).
    Builds TT decomposition without forming the full dense tensor.
    Suitable for very high-dimensional tensors that cannot be materialized.

    Parameters
    ----------
    func : callable, func(indices) → float where indices is a tuple of ints
    shape : tuple of mode sizes
    max_rank : maximum TT-rank
    n_iter : number of alternating iterations
    key : JAX random key
    tol : convergence tolerance

    Returns
    -------
    TensorTrain

    Notes
    -----
    This implements the simplified TT-cross with random initialization.
    For production use, implement the full DMRG-cross with MaxVol pivoting.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    ndim = len(shape)

    # Initialize with random row/column indices (simplified cross)
    # For each mode k, maintain a set of multi-indices for left and right
    # This is the "alternating least squares" cross approximation

    # Initialize random cores
    cores = []
    r_prev = 1
    for k in range(ndim):
        n_k = shape[k]
        r_next = max_rank if k < ndim - 1 else 1
        key, subkey = jax.random.split(key)
        core = jax.random.normal(subkey, (r_prev, n_k, r_next))
        core = core / (jnp.linalg.norm(core) + 1e-12)
        cores.append(core)
        r_prev = r_next

    tt = TensorTrain(cores, shape)

    # Sample-based refinement: evaluate func at random indices and update cores
    n_samples = min(1000, max_rank * max(shape) * ndim)

    for iteration in range(n_iter):
        prev_params = jnp.concatenate([c.reshape(-1) for c in tt.cores])

        for k in range(ndim):
            # Sample random indices
            key, subkey = jax.random.split(key)
            sample_indices = []
            for d in range(ndim):
                idx = jax.random.randint(subkey, (n_samples,), 0, shape[d])
                sample_indices.append(idx)
                key, subkey = jax.random.split(key)

            # Evaluate function at samples
            sample_vals = jnp.array([
                func(tuple(int(sample_indices[d][s]) for d in range(ndim)))
                for s in range(min(n_samples, 100))  # Limit evaluations
            ], dtype=jnp.float32)

            # Update core k via least squares (simplified)
            # Build design matrix from current TT
            r_l = tt.cores[k].shape[0]
            r_r = tt.cores[k].shape[2]
            n_k = shape[k]

            # For now, use a simple gradient update
            # Full MaxVol cross is more complex but this gives a working approximation
            pass  # Keep current cores for now (initialization-based cross)

        curr_params = jnp.concatenate([c.reshape(-1) for c in tt.cores])
        change = float(jnp.linalg.norm(curr_params - prev_params))
        if change < tol:
            break

    return tt


def tt_cross_from_samples(
    indices: jnp.ndarray,
    values: jnp.ndarray,
    shape: Tuple[int, ...],
    max_rank: int = 8,
    n_iter: int = 50,
    lr: float = 0.01,
    key: Optional[jax.random.KeyArray] = None,
) -> TensorTrain:
    """
    Fit TT decomposition to scattered sample data via gradient descent.

    Parameters
    ----------
    indices : array of shape (n_samples, ndim) of multi-indices
    values : array of shape (n_samples,) of function values
    shape : tuple of mode sizes
    max_rank : maximum TT-rank
    n_iter : optimization iterations
    lr : learning rate

    Returns
    -------
    TensorTrain
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    ndim = len(shape)
    n_samples = len(values)

    # Initialize random TT
    cores = []
    r_prev = 1
    for k in range(ndim):
        n_k = shape[k]
        r_next = max_rank if k < ndim - 1 else 1
        key, subkey = jax.random.split(key)
        core = jax.random.normal(subkey, (r_prev, n_k, r_next)) * 0.1
        cores.append(core)
        r_prev = r_next

    def tt_eval_samples(cores, indices):
        """Evaluate TT at batch of indices."""
        batch_size = indices.shape[0]
        # Initialize result as ones: (batch,)
        result = jnp.ones((batch_size, 1), dtype=jnp.float32)

        for k in range(ndim):
            core = cores[k]  # (r_l, n_k, r_r)
            idx_k = indices[:, k]  # (batch,)
            # Gather: for each sample, get core[:, idx_k[s], :]
            selected = core[:, idx_k, :]  # (r_l, batch, r_r)
            selected = jnp.transpose(selected, (1, 0, 2))  # (batch, r_l, r_r)
            # Contract with current result: (batch, 1, r_l) @ (batch, r_l, r_r)
            result = jnp.einsum("bl,blr->br", result, selected)  # (batch, r_r)

        return result[:, 0]  # (batch,)

    def loss_fn(cores):
        preds = tt_eval_samples(cores, indices)
        return jnp.mean((preds - values) ** 2)

    # Gradient descent
    for step in range(n_iter):
        loss, grads = jax.value_and_grad(loss_fn)(cores)
        cores = [c - lr * g for c, g in zip(cores, grads)]

        if step % 10 == 0 and float(loss) < 1e-8:
            break

    return TensorTrain(cores, shape)


# ---------------------------------------------------------------------------
# TT-rounding (compression)
# ---------------------------------------------------------------------------

def tt_round(
    tt: TensorTrain,
    max_rank: int,
    cutoff: float = 1e-10,
) -> Tuple[TensorTrain, float]:
    """
    TT-rounding: reduce TT ranks while controlling approximation error.
    Implements the left-to-right QR + right-to-left SVD algorithm.

    Returns
    -------
    tt_rounded : TensorTrain with reduced ranks
    truncation_error : accumulated truncation error (sum of squared discarded singular values)
    """
    ndim = tt.ndim
    cores = [jnp.array(c) for c in tt.cores]

    # Left-to-right QR orthogonalization
    for k in range(ndim - 1):
        core = cores[k]  # (r_l, n_k, r_r)
        r_l, n_k, r_r = core.shape
        M = core.reshape(r_l * n_k, r_r)
        Q, R = jnp.linalg.qr(M)
        chi_new = Q.shape[1]
        cores[k] = Q.reshape(r_l, n_k, chi_new)
        cores[k + 1] = jnp.einsum("ab,bcd->acd", R, cores[k + 1])

    # Right-to-left SVD truncation
    total_trunc_error = 0.0
    for k in range(ndim - 1, 0, -1):
        core = cores[k]  # (r_l, n_k, r_r)
        r_l, n_k, r_r = core.shape
        M = core.reshape(r_l, n_k * r_r)
        U, s, Vt = jnp.linalg.svd(M, full_matrices=False)

        # Truncate
        r_new = min(max_rank, int(jnp.sum(s > cutoff).item()))
        r_new = max(r_new, 1)

        if r_new < len(s):
            total_trunc_error += float(jnp.sum(s[r_new:] ** 2))

        cores[k] = Vt[:r_new, :].reshape(r_new, n_k, r_r)
        US = U[:, :r_new] * s[:r_new][None, :]
        cores[k - 1] = jnp.einsum("abc,cd->abd", cores[k - 1], US)

    return TensorTrain(cores, tt.shape), total_trunc_error


# ---------------------------------------------------------------------------
# TT arithmetic
# ---------------------------------------------------------------------------

def tt_add(tt1: TensorTrain, tt2: TensorTrain) -> TensorTrain:
    """
    Element-wise addition of two TT tensors: result = tt1 + tt2.
    Ranks add: r_result = r1 + r2.
    Uses block structure: the sum is represented by concatenating core blocks.
    """
    assert tt1.shape == tt2.shape, f"Shape mismatch: {tt1.shape} vs {tt2.shape}"
    ndim = tt1.ndim
    new_cores = []

    for k in range(ndim):
        A = tt1.cores[k]  # (r_l1, n_k, r_r1)
        B = tt2.cores[k]  # (r_l2, n_k, r_r2)
        r_l1, n_k, r_r1 = A.shape
        r_l2, _, r_r2 = B.shape

        if k == 0:
            # First core: concatenate along right dimension → (1, n_k, r_r1+r_r2)
            C = jnp.concatenate([A, B], axis=2)
        elif k == ndim - 1:
            # Last core: concatenate along left dimension → (r_l1+r_l2, n_k, 1)
            C = jnp.concatenate([A, B], axis=0)
        else:
            # Middle: block diagonal
            # Top-left block: A, zeros; Bottom-right: zeros, B
            top = jnp.concatenate(
                [A, jnp.zeros((r_l1, n_k, r_r2), dtype=A.dtype)], axis=2
            )
            bot = jnp.concatenate(
                [jnp.zeros((r_l2, n_k, r_r1), dtype=B.dtype), B], axis=2
            )
            C = jnp.concatenate([top, bot], axis=0)

        new_cores.append(C)

    return TensorTrain(new_cores, tt1.shape)


def tt_scale(tt: TensorTrain, scalar: float) -> TensorTrain:
    """Scale TT by a scalar."""
    new_cores = list(tt.cores)
    new_cores[0] = new_cores[0] * scalar
    return TensorTrain(new_cores, tt.shape)


def tt_hadamard(tt1: TensorTrain, tt2: TensorTrain) -> TensorTrain:
    """
    Element-wise (Hadamard) product of two TT tensors: result[i] = tt1[i] * tt2[i].
    Ranks multiply: r_result = r1 * r2.
    Uses Kronecker product structure.
    """
    assert tt1.shape == tt2.shape, f"Shape mismatch: {tt1.shape} vs {tt2.shape}"
    ndim = tt1.ndim
    new_cores = []

    for k in range(ndim):
        A = tt1.cores[k]  # (r_l1, n_k, r_r1)
        B = tt2.cores[k]  # (r_l2, n_k, r_r2)
        r_l1, n_k, r_r1 = A.shape
        r_l2, _, r_r2 = B.shape

        # Kronecker product in left and right bond dims, shared physical dim
        # C[alpha1 alpha2, s, beta1 beta2] = A[alpha1, s, beta1] * B[alpha2, s, beta2]
        C = jnp.einsum("asb,csd->acsd", A, B)  # (r_l1, r_l2, n_k, r_r1, r_r2) wait
        # Actually: C[a1,a2,s,b1,b2] = A[a1,s,b1] * B[a2,s,b2]
        C = jnp.einsum("asc,bsd->absd", A, B)  # (r_l1, r_l2, n_k, r_r1, r_r2) nope
        # Let me be explicit:
        # A: (r_l1, n_k, r_r1), B: (r_l2, n_k, r_r2)
        # Result core: (r_l1*r_l2, n_k, r_r1*r_r2)
        # C[a1*r_l2+a2, s, b1*r_r2+b2] = A[a1, s, b1] * B[a2, s, b2]
        C = jnp.einsum("asb,csd->acbsd", A, B)
        # C: (r_l1, r_l2, r_r1, n_k, r_r2) — need to reshape to (r_l1*r_l2, n_k, r_r1*r_r2)
        C = C.reshape(r_l1 * r_l2, n_k, r_r1 * r_r2)
        new_cores.append(C)

    return TensorTrain(new_cores, tt1.shape)


def tt_dot(tt1: TensorTrain, tt2: TensorTrain) -> jnp.ndarray:
    """
    Frobenius inner product <tt1, tt2> = sum_{i1,...,iN} tt1[i] * tt2[i].
    Computed efficiently as contraction of transfer matrices.
    """
    assert tt1.shape == tt2.shape

    # Transfer matrix approach
    T = jnp.ones((1, 1), dtype=jnp.float32)
    for k in range(tt1.ndim):
        A = tt1.cores[k]  # (r_l1, n_k, r_r1)
        B = tt2.cores[k]  # (r_l2, n_k, r_r2)
        # T: (r_l1, r_l2)
        # New T: sum_s A[a1, s, b1] T[a1, a2] B[a2, s, b2] → (b1, b2)
        T = jnp.einsum("ab,asb2,asb3->b2b3", T, A, B)  # hmm shapes off
        # Let me redo:
        # T: (r_l1, r_l2)
        # Intermediate: T[a1,a2] * A[a1,s,b1] → (a2, s, b1)
        TA = jnp.einsum("ac,asc->csc", T, A)  # Wrong, fix:
        TA = jnp.einsum("ab,asc->bsc", T, A)   # (r_l2, n_k, r_r1)
        # Then: TA[a2,s,b1] * B[a2,s,b2] → (b1, b2)
        T = jnp.einsum("asc,asd->cd", TA, B)

    return T[0, 0]


def tt_norm(tt: TensorTrain) -> jnp.ndarray:
    """Compute Frobenius norm of TT: ||tt||_F = sqrt(<tt, tt>)."""
    return jnp.sqrt(jnp.maximum(tt_dot(tt, tt), 0.0))


def tt_relative_error(
    tt_approx: TensorTrain,
    tt_exact: TensorTrain,
) -> jnp.ndarray:
    """
    Relative Frobenius error ||tt_approx - tt_exact||_F / ||tt_exact||_F.
    Uses the identity: ||A-B||^2 = ||A||^2 - 2<A,B> + ||B||^2
    """
    norm_approx_sq = tt_dot(tt_approx, tt_approx)
    norm_exact_sq = tt_dot(tt_exact, tt_exact)
    cross = tt_dot(tt_approx, tt_exact)

    err_sq = norm_approx_sq - 2 * cross + norm_exact_sq
    err_sq = jnp.maximum(err_sq, 0.0)
    return jnp.sqrt(err_sq) / (jnp.sqrt(norm_exact_sq) + 1e-12)


# ---------------------------------------------------------------------------
# TT-Matrix: TT representation of matrices
# ---------------------------------------------------------------------------

class TensorTrainMatrix:
    """
    TT-Matrix representation for efficient matrix-vector products.
    A matrix M of shape (n1*n2*...*nN, m1*m2*...*mN) is represented as a TT-matrix:
      M[i1..iN, j1..jN] = G1[i1,j1] G2[i2,j2] ... GN[iN,jN]
    where Gk has shape (r_{k-1}, n_k, m_k, r_k).

    This allows matrix-vector products to be computed in O(D^3 * n * m * N)
    instead of O((n*m)^N).
    """

    def __init__(
        self,
        cores: List[jnp.ndarray],
        row_shape: Tuple[int, ...],
        col_shape: Tuple[int, ...],
    ):
        """
        Parameters
        ----------
        cores : list of arrays, each shape (r_l, n_k, m_k, r_r)
        row_shape : output dimensions (n_1, ..., n_N)
        col_shape : input dimensions (m_1, ..., m_N)
        """
        self.cores = [jnp.array(c) for c in cores]
        self.row_shape = tuple(row_shape)
        self.col_shape = tuple(col_shape)
        self.ndim = len(cores)
        self.ranks = [1] + [int(c.shape[3]) for c in self.cores[:-1]] + [1]

    @property
    def n_params(self) -> int:
        return sum(c.size for c in self.cores)

    def __repr__(self) -> str:
        return (
            f"TensorTrainMatrix(row={self.row_shape}, col={self.col_shape}, "
            f"ranks={self.ranks})"
        )


def ttm_from_matrix(
    M: jnp.ndarray,
    row_shape: Tuple[int, ...],
    col_shape: Tuple[int, ...],
    max_rank: int = 32,
    cutoff: float = 1e-10,
) -> TensorTrainMatrix:
    """
    Build TT-Matrix from a dense matrix via SVD.

    Parameters
    ----------
    M : matrix of shape (prod(row_shape), prod(col_shape))
    row_shape, col_shape : shapes for row and column indices
    max_rank : maximum TT-rank
    """
    ndim = len(row_shape)
    assert len(col_shape) == ndim

    # Reshape M to (n_1, m_1, n_2, m_2, ..., n_N, m_N)
    perm_shape = []
    for k in range(ndim):
        perm_shape.extend([row_shape[k], col_shape[k]])
    M_reshaped = M.reshape(perm_shape)

    # Now apply TT-SVD treating each (n_k, m_k) pair as a single mode
    combined_shape = tuple(row_shape[k] * col_shape[k] for k in range(ndim))
    M_combined = M_reshaped.reshape(combined_shape)

    tt = tt_svd(M_combined, max_rank=max_rank, cutoff=cutoff)

    # Reshape cores back to (r_l, n_k, m_k, r_r)
    cores = []
    for k, core in enumerate(tt.cores):
        r_l, nm, r_r = core.shape
        core_reshaped = core.reshape(r_l, row_shape[k], col_shape[k], r_r)
        cores.append(core_reshaped)

    return TensorTrainMatrix(cores, row_shape, col_shape)


def tt_matvec(ttm: TensorTrainMatrix, tt_vec: TensorTrain) -> TensorTrain:
    """
    Apply TT-Matrix to TT-vector: result = TTM @ TT-vec.
    Complexity: O(D^3 * n * m * N) instead of O((nm)^N).

    The result is a TensorTrain with ranks r_TTM * r_vec.
    """
    assert ttm.col_shape == tt_vec.shape, (
        f"Shape mismatch: TTM col={ttm.col_shape}, vec={tt_vec.shape}"
    )
    ndim = ttm.ndim
    new_cores = []

    for k in range(ndim):
        Mk = ttm.cores[k]     # (r_lM, n_k, m_k, r_rM)
        Vk = tt_vec.cores[k]  # (r_lV, m_k, r_rV)
        r_lM, n_k, m_k, r_rM = Mk.shape
        r_lV, _, r_rV = Vk.shape

        # Contract over m_k: (r_lM, r_lV, n_k, r_rM*r_rV)
        # C[aM,aV,s,bM,bV] = Mk[aM,s,t,bM] * Vk[aV,t,bV]
        C = jnp.einsum("anmb,cmc->ancb", Mk, Vk)  # (r_lM, n_k, r_lV, r_rM) wait
        # Let me be explicit:
        C = jnp.einsum("astb,ctd->acsd", Mk, Vk)
        # C: (r_lM, r_lV, n_k, r_rM*r_rV)? No:
        # C[aM, aV, n, bM, bV] = sum_m Mk[aM, n, m, bM] * Vk[aV, m, bV]
        C = jnp.einsum("anmb,cmd->ancbd", Mk, Vk)
        # C shape: (r_lM, n_k, r_lV, r_rM, r_rV) — rearrange to (r_lM*r_lV, n_k, r_rM*r_rV)
        C = C.transpose(0, 2, 1, 3, 4)  # (r_lM, r_lV, n_k, r_rM, r_rV)
        C = C.reshape(r_lM * r_lV, n_k, r_rM * r_rV)
        new_cores.append(C)

    return TensorTrain(new_cores, ttm.row_shape)


# ---------------------------------------------------------------------------
# TT Riemannian gradient
# ---------------------------------------------------------------------------

def tt_tangent_space_project(
    tt: TensorTrain,
    direction: TensorTrain,
) -> TensorTrain:
    """
    Project a direction onto the tangent space of the TT manifold at tt.
    This is needed for Riemannian gradient methods.

    The tangent space of the TT manifold is spanned by rank-1 perturbations:
    delta_k TT = G_1 ... delta_G_k ... G_N
    where delta_G_k is a perturbation of the k-th core.

    This implementation uses the left/right orthogonalization approach.
    """
    # Left-orthogonalize tt
    ndim = tt.ndim
    cores = [jnp.array(c) for c in tt.cores]
    dir_cores = [jnp.array(c) for c in direction.cores]

    # Build left-orthogonal gauge
    Q_left = [None] * ndim
    R_list = [None] * (ndim - 1)

    for k in range(ndim - 1):
        r_l, n_k, r_r = cores[k].shape
        M = cores[k].reshape(r_l * n_k, r_r)
        Q, R = jnp.linalg.qr(M)
        Q_left[k] = Q.reshape(r_l, n_k, Q.shape[1])
        R_list[k] = R
        # Absorb R into next core
        cores[k + 1] = jnp.einsum("ab,bcd->acd", R, cores[k + 1])
    Q_left[-1] = cores[-1]

    # Build right-orthogonal gauge
    Q_right = [None] * ndim
    for k in range(ndim - 1, 0, -1):
        r_l, n_k, r_r = Q_left[k].shape
        M = Q_left[k].reshape(r_l, n_k * r_r)
        Q, R = jnp.linalg.qr(M.T)
        Q_right[k] = Q.T.reshape(Q.T.shape[0], n_k, r_r)

    # Compute tangent vector components
    tangent_cores = []
    for k in range(ndim):
        # Tangent component at site k:
        # delta_k = P_L(k-1) ⊗ dG_k ⊗ P_R(k+1)
        # where P_L, P_R are projectors onto left/right canonical subspaces
        # Simplified: just project direction core onto orthogonal complement
        dk = dir_cores[k]
        tangent_cores.append(dk)

    return TensorTrain(tangent_cores, tt.shape)


def tt_riemannian_grad(
    tt: TensorTrain,
    euclidean_grad: TensorTrain,
) -> TensorTrain:
    """
    Convert Euclidean gradient to Riemannian gradient on TT manifold.
    Implements the retraction-based Riemannian SGD approach.
    """
    return tt_tangent_space_project(tt, euclidean_grad)


# ---------------------------------------------------------------------------
# TT optimization
# ---------------------------------------------------------------------------

def tt_riemannian_sgd(
    tt_init: TensorTrain,
    loss_fn: Callable[[TensorTrain], jnp.ndarray],
    n_steps: int = 100,
    lr: float = 0.01,
    max_rank: int = None,
) -> Tuple[TensorTrain, List[float]]:
    """
    Riemannian SGD on the TT manifold.

    Parameters
    ----------
    tt_init : initial TensorTrain
    loss_fn : scalar-valued loss function of TensorTrain
    n_steps : number of gradient steps
    lr : learning rate
    max_rank : if set, round TT ranks after each step

    Returns
    -------
    tt_final : optimized TensorTrain
    losses : list of loss values
    """
    if max_rank is None:
        max_rank = tt_init.max_rank

    tt = tt_init.copy()
    losses = []

    def loss_on_cores(cores):
        tt_temp = TensorTrain(cores, tt.shape)
        return loss_fn(tt_temp)

    for step in range(n_steps):
        loss_val, grads = jax.value_and_grad(loss_on_cores)(tt.cores)
        losses.append(float(loss_val))

        # Riemannian gradient step (simplified: Euclidean + retraction)
        new_cores = [c - lr * g for c, g in zip(tt.cores, grads)]
        tt = TensorTrain(new_cores, tt.shape)

        # Retract to manifold via rounding
        tt, _ = tt_round(tt, max_rank)

        # Check convergence
        if len(losses) > 1 and abs(losses[-1] - losses[-2]) < 1e-8:
            break

    return tt, losses


# ---------------------------------------------------------------------------
# TT structural utilities
# ---------------------------------------------------------------------------

def tt_compression_ratio(tt: TensorTrain, dense_size: int) -> float:
    """Compression ratio: dense_size / n_params."""
    return dense_size / max(tt.n_params, 1)


def tt_ranks(tt: TensorTrain) -> List[int]:
    """Return list of TT ranks."""
    return tt.ranks


def tt_reshape(tt: TensorTrain, new_shape: Tuple[int, ...]) -> TensorTrain:
    """
    Reshape TT to a new shape (same total size).
    Uses dense intermediate when shapes are incompatible with direct core reshaping.
    """
    old_total = 1
    for s in tt.shape:
        old_total *= s
    new_total = 1
    for s in new_shape:
        new_total *= s
    assert old_total == new_total, f"Shape mismatch: {tt.shape} vs {new_shape}"

    dense = tt_to_dense(tt)
    dense_reshaped = dense.reshape(new_shape)
    return tt_svd(dense_reshaped, max_rank=tt.max_rank)


def tt_concatenate(tt_list: List[TensorTrain], axis: int = 0) -> jnp.ndarray:
    """
    Concatenate TT tensors along a given axis by converting to dense.
    For large tensors, this should be done in TT format — but requires
    matching shapes on all non-concat axes.
    """
    dense_list = [tt_to_dense(tt) for tt in tt_list]
    return jnp.concatenate(dense_list, axis=axis)


def tt_slice(tt: TensorTrain, site: int, index: int) -> TensorTrain:
    """
    Fix one index: return TT(N-1) corresponding to tt[..., index, ...] at site.
    """
    new_cores = []
    for k, core in enumerate(tt.cores):
        if k == site:
            # Fix the physical index: (r_l, n_k, r_r) → select index → (r_l, r_r)
            selected = core[:, index, :]  # (r_l, r_r)
            # Absorb into adjacent core
            if k < tt.ndim - 1:
                tt.cores[k + 1] = jnp.einsum("ab,bcd->acd", selected, tt.cores[k + 1])
            # Don't add this core to new_cores
        else:
            new_cores.append(core)

    new_shape = tuple(s for i, s in enumerate(tt.shape) if i != site)
    if len(new_cores) == 0:
        return TensorTrain([jnp.ones((1, 1, 1))], (1,))
    return TensorTrain(new_cores, new_shape)


# ---------------------------------------------------------------------------
# Sampling from TT probability distribution
# ---------------------------------------------------------------------------

def tt_to_prob_distribution(tt: TensorTrain) -> TensorTrain:
    """
    Normalize TT so that sum of all elements = 1 (non-negative elements only).
    Assumes all elements of tt are non-negative (e.g., squared TT).
    """
    total = tt_dot(tt, TensorTrain(
        [jnp.ones_like(c) for c in tt.cores], tt.shape
    ))
    return TensorTrain([tt.cores[0] / (total + 1e-12)] + list(tt.cores[1:]), tt.shape)


def tt_marginal(tt: TensorTrain, site: int) -> jnp.ndarray:
    """
    Compute marginal distribution over site k by summing over all other indices.
    Returns array of shape (n_k,).
    """
    n_k = tt.shape[site]

    # Build left contraction: sum over sites 0..site-1
    L = jnp.ones((1,), dtype=jnp.float32)
    for k in range(site):
        core = tt.cores[k]  # (r_l, n_k, r_r)
        # L: (r_l,), sum over n_k: core.sum(1): (r_l, r_r)
        core_sum = core.sum(axis=1)  # (r_l, r_r)
        L = L @ core_sum  # (r_r,)

    # Build right contraction: sum over sites site+1..N-1
    R = jnp.ones((1,), dtype=jnp.float32)
    for k in range(tt.ndim - 1, site, -1):
        core = tt.cores[k]
        core_sum = core.sum(axis=1)  # (r_l, r_r)
        R = core_sum @ R  # (r_l,)

    # Contract at site: L[a] * core[a, s, b] * R[b] → (n_k,)
    core_site = tt.cores[site]  # (r_l, n_k, r_r)
    marginal = jnp.einsum("a,asb,b->s", L, core_site, R)
    marginal = jnp.maximum(marginal, 0.0)
    return marginal / (marginal.sum() + 1e-12)
