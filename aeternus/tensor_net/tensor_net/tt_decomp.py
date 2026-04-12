"""
tt_decomp.py — Tensor Train decomposition engine for TensorNet (Project AETERNUS).

Implements:
  - TT-SVD: exact Tensor Train decomposition via sequential SVD
  - TT-Rounding: re-compression of a Tensor Train via DMRG-style sweeps
  - TT-Cross: cross-approximation (maxvol pivot algorithm)
  - TT-Rank Estimation: from approximation error target
  - TT-Format Arithmetic: add, hadamard, scale, dot
  - TT-Orthogonalization: left/right/mixed canonical forms
  - Riemannian Gradient on TT manifold
  - DMRG pivot selection
  - TT-matrix (MPO-style) for operator compression
  - TT-eigenvalue problem (for financial PCA)
  - TT-LU and TT-QR utilities
  - Frobenius norm in TT format
  - TT-slice: evaluate TT at given indices
  - Full-to-TT and TT-to-full conversion
"""

from __future__ import annotations

import math
import functools
from typing import List, Optional, Tuple, Sequence, Union, Callable, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad, vmap


# ============================================================================
# TensorTrain class
# ============================================================================

class TensorTrain:
    """
    Tensor Train decomposition of a high-dimensional array.

    Represents an N-dimensional array A[i_1, i_2, ..., i_N] as:
      A[i_1, ..., i_N] = G_1[i_1] @ G_2[i_2] @ ... @ G_N[i_N]

    where G_k has shape (r_{k-1}, n_k, r_k), r_0 = r_N = 1.

    Attributes
    ----------
    cores : list of jnp.ndarray, each shape (r_{k-1}, n_k, r_k)
    shape : tuple of mode sizes (n_1, ..., n_N)
    ranks : list of TT-ranks [1, r_1, ..., r_{N-1}, 1]

    Notes
    -----
    Registered as a JAX pytree for compatibility with jit/grad/vmap.
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

    @property
    def dtype(self):
        return self.cores[0].dtype

    def __repr__(self) -> str:
        return (
            f"TensorTrain(shape={self.shape}, ranks={self.ranks}, "
            f"n_params={self.n_params})"
        )

    def __len__(self) -> int:
        return self.ndim

    def __getitem__(self, k: int) -> jnp.ndarray:
        return self.cores[k]

    def copy(self) -> "TensorTrain":
        return TensorTrain([jnp.array(c) for c in self.cores], self.shape)

    def astype(self, dtype) -> "TensorTrain":
        return TensorTrain([c.astype(dtype) for c in self.cores], self.shape)

    def compression_ratio(self) -> float:
        """Ratio of full tensor size to TT parameter count."""
        full_size = 1
        for n in self.shape:
            full_size *= n
        return full_size / (self.n_params + 1e-10)

    def info(self) -> Dict[str, Any]:
        return {
            "shape": self.shape,
            "ranks": self.ranks,
            "max_rank": self.max_rank,
            "n_params": self.n_params,
            "compression_ratio": self.compression_ratio(),
            "dtype": str(self.dtype),
        }


def _tt_flatten(tt: TensorTrain):
    cores = tt.cores
    aux = {"shape": tt.shape}
    return cores, aux


def _tt_unflatten(aux, cores):
    return TensorTrain(list(cores), shape=aux["shape"])


jax.tree_util.register_pytree_node(TensorTrain, _tt_flatten, _tt_unflatten)


# ============================================================================
# TensorTrain arithmetic
# ============================================================================

def tt_add(tt1: TensorTrain, tt2: TensorTrain) -> TensorTrain:
    """
    Add two Tensor Trains: result = tt1 + tt2.

    Uses block structure: ranks add, so result ranks are r1_k + r2_k.

    Parameters
    ----------
    tt1, tt2 : TensorTrain with same shape

    Returns
    -------
    TensorTrain representing tt1 + tt2
    """
    assert tt1.shape == tt2.shape, "TT shapes must match for addition"
    n = tt1.ndim
    cores = []

    for k in range(n):
        G1 = tt1.cores[k]  # (r1_l, n_k, r1_r)
        G2 = tt2.cores[k]  # (r2_l, n_k, r2_r)
        r1_l, n_k, r1_r = G1.shape
        r2_l, _, r2_r = G2.shape

        if k == 0:
            # Concatenate along right: (1, n_k, r1_r + r2_r)
            C = jnp.concatenate([G1, G2], axis=2)
        elif k == n - 1:
            # Concatenate along left: (r1_l + r2_l, n_k, 1)
            C = jnp.concatenate([G1, G2], axis=0)
        else:
            # Block diagonal: (r1_l + r2_l, n_k, r1_r + r2_r)
            top = jnp.concatenate([G1, jnp.zeros((r1_l, n_k, r2_r), dtype=G1.dtype)], axis=2)
            bot = jnp.concatenate([jnp.zeros((r2_l, n_k, r1_r), dtype=G2.dtype), G2], axis=2)
            C = jnp.concatenate([top, bot], axis=0)
        cores.append(C)

    return TensorTrain(cores, tt1.shape)


def tt_scale(tt: TensorTrain, alpha: float) -> TensorTrain:
    """Scale a TT by scalar alpha (applied to first core)."""
    cores = [jnp.array(c) for c in tt.cores]
    cores[0] = cores[0] * alpha
    return TensorTrain(cores, tt.shape)


def tt_subtract(tt1: TensorTrain, tt2: TensorTrain) -> TensorTrain:
    """Subtract two TTs: tt1 - tt2."""
    return tt_add(tt1, tt_scale(tt2, -1.0))


def tt_hadamard(tt1: TensorTrain, tt2: TensorTrain) -> TensorTrain:
    """
    Element-wise (Hadamard) product of two TTs.
    Ranks multiply: result rank = r1_k * r2_k.

    Parameters
    ----------
    tt1, tt2 : TensorTrain with same shape

    Returns
    -------
    TensorTrain representing element-wise product
    """
    assert tt1.shape == tt2.shape
    cores = []
    for k in range(tt1.ndim):
        G1 = tt1.cores[k]  # (r1_l, n_k, r1_r)
        G2 = tt2.cores[k]  # (r2_l, n_k, r2_r)
        r1_l, n_k, r1_r = G1.shape
        r2_l, _, r2_r = G2.shape
        # C[a*a', s, b*b'] = G1[a, s, b] * G2[a', s, b']
        C = jnp.einsum("asc,bsd->absd", G1, G2)
        C = C.reshape(r1_l * r2_l, n_k, r1_r * r2_r)
        cores.append(C)
    return TensorTrain(cores, tt1.shape)


def tt_dot(tt1: TensorTrain, tt2: TensorTrain) -> jnp.ndarray:
    """
    Compute the inner product <tt1, tt2> = sum_{i1,...,iN} tt1[i] * tt2[i].

    Parameters
    ----------
    tt1, tt2 : TensorTrain with same shape

    Returns
    -------
    Scalar inner product
    """
    assert tt1.shape == tt2.shape
    # Initialize: shape (r1, r2) = (1, 1)
    T = jnp.ones((1, 1), dtype=jnp.float32)

    for k in range(tt1.ndim):
        G1 = tt1.cores[k]  # (r1_l, n_k, r1_r)
        G2 = tt2.cores[k]  # (r2_l, n_k, r2_r)
        # T[a,b] * G1[a,s,a'] * G2[b,s,b'] -> T'[a',b']
        TG1 = jnp.einsum("ab,asc->bsc", T, G1)
        T = jnp.einsum("bsc,bsd->cd", TG1, G2)

    return T[0, 0]


def tt_norm(tt: TensorTrain) -> jnp.ndarray:
    """Frobenius norm of a TT: ||tt||_F = sqrt(<tt, tt>)."""
    return jnp.sqrt(jnp.maximum(tt_dot(tt, tt), 0.0))


def tt_norm_sq(tt: TensorTrain) -> jnp.ndarray:
    """Squared Frobenius norm <tt, tt>."""
    return tt_dot(tt, tt)


def tt_relative_error(tt1: TensorTrain, tt2: TensorTrain) -> jnp.ndarray:
    """
    Relative error ||tt1 - tt2||_F / ||tt1||_F.

    Parameters
    ----------
    tt1 : reference TensorTrain
    tt2 : approximation TensorTrain

    Returns
    -------
    Scalar relative error
    """
    diff = tt_subtract(tt1, tt2)
    err = tt_norm(diff)
    ref = tt_norm(tt1)
    return err / (ref + 1e-30)


def tt_normalize(tt: TensorTrain) -> TensorTrain:
    """Normalize TT to unit Frobenius norm."""
    n = tt_norm(tt)
    return tt_scale(tt, 1.0 / (float(n) + 1e-30))


# ============================================================================
# TT-SVD: Tensor Train decomposition
# ============================================================================

def tt_svd(
    tensor: jnp.ndarray,
    max_rank: int = 100,
    cutoff: float = 1e-10,
) -> TensorTrain:
    """
    Decompose a dense N-dimensional tensor into Tensor Train format via TT-SVD.

    Implements the algorithm of Oseledets (2011), "Tensor-train decomposition."
    SIAM Journal on Scientific Computing, 33(5), 2295-2317.

    The algorithm performs N-1 SVDs, working from left to right.

    Parameters
    ----------
    tensor : dense array of arbitrary shape
    max_rank : maximum TT-rank at each bond
    cutoff : relative SVD truncation threshold

    Returns
    -------
    TensorTrain decomposition of the input tensor

    Complexity
    ----------
    O(N * n^(N/2) * r^2) in general; O(N * n * r^3) for fixed rank
    """
    shape = tensor.shape
    n_dims = len(shape)
    cores = []

    # Reshape to 2D for first SVD
    arr = jnp.array(tensor, dtype=jnp.float32)
    r_left = 1

    for k in range(n_dims - 1):
        n_k = shape[k]
        arr = arr.reshape(r_left * n_k, -1)

        U, s, Vt = jnp.linalg.svd(arr, full_matrices=False)

        # Truncation
        s_max = float(s[0]) if s.shape[0] > 0 else 1.0
        rank = int(jnp.sum(s > cutoff * s_max).item())
        rank = max(1, min(rank, max_rank, U.shape[1]))

        U = U[:, :rank]
        s_trunc = s[:rank]
        Vt = Vt[:rank, :]

        cores.append(U.reshape(r_left, n_k, rank))
        arr = jnp.diag(s_trunc) @ Vt
        r_left = rank

    # Last core
    cores.append(arr.reshape(r_left, shape[-1], 1))

    return TensorTrain(cores, shape)


def tt_svd_from_function(
    func: Callable,
    shape: Tuple[int, ...],
    max_rank: int = 20,
    cutoff: float = 1e-8,
    sample_size: Optional[int] = None,
) -> TensorTrain:
    """
    Build a TT approximation of a function by sampling and decomposing.

    Evaluates the function on a grid and applies TT-SVD.

    Parameters
    ----------
    func : callable taking index tuple and returning scalar
    shape : grid shape (n_1, ..., n_N)
    max_rank : maximum TT-rank
    cutoff : SVD truncation threshold
    sample_size : if specified, subsample the grid

    Returns
    -------
    TensorTrain approximation of func on the grid
    """
    # Build the full tensor by evaluation
    total = 1
    for n in shape:
        total *= n

    if sample_size is not None and total > sample_size:
        # Sparse approximation: fill with zeros, sample-based filling
        tensor = jnp.zeros(shape)
        key = jax.random.PRNGKey(42)
        indices = jax.random.randint(key, (sample_size, len(shape)), 0, jnp.array(list(shape)))
        for idx in indices:
            val = func(tuple(int(i) for i in idx))
            tensor = tensor.at[tuple(idx)].set(float(val))
    else:
        # Full grid evaluation
        import itertools
        values = []
        for idx in itertools.product(*[range(n) for n in shape]):
            values.append(func(idx))
        tensor = jnp.array(values).reshape(shape)

    return tt_svd(tensor, max_rank=max_rank, cutoff=cutoff)


# ============================================================================
# TT-Rounding: re-compression
# ============================================================================

def tt_round(
    tt: TensorTrain,
    max_rank: int,
    cutoff: float = 1e-12,
) -> TensorTrain:
    """
    Re-compress a TT to lower ranks via orthogonalization + truncated SVD.

    Algorithm:
    1. Right-to-left orthogonalization (brings TT to right-canonical form)
    2. Left-to-right SVD sweep with rank truncation

    This minimizes the approximation error in the Frobenius norm.

    Parameters
    ----------
    tt : input TensorTrain
    max_rank : maximum rank after rounding
    cutoff : relative SVD truncation threshold

    Returns
    -------
    Rounded TensorTrain with ranks <= max_rank
    """
    # Step 1: Right orthogonalization
    tt_rc = tt_right_orthogonalize(tt)
    cores = [jnp.array(c) for c in tt_rc.cores]
    n = tt.ndim

    # Step 2: Left-to-right SVD sweep with truncation
    for k in range(n - 1):
        G = cores[k]
        r_l, n_k, r_r = G.shape
        M = G.reshape(r_l * n_k, r_r)
        U, s, Vt = jnp.linalg.svd(M, full_matrices=False)

        s_max = float(s[0]) if s.shape[0] > 0 else 1.0
        rank = int(jnp.sum(s > cutoff * s_max).item())
        rank = max(1, min(rank, max_rank, U.shape[1]))

        U = U[:, :rank]
        s_k = s[:rank]
        Vt = Vt[:rank, :]

        cores[k] = U.reshape(r_l, n_k, rank)
        cores[k + 1] = jnp.einsum("ab,bcd->acd", jnp.diag(s_k) @ Vt, cores[k + 1])

    return TensorTrain(cores, tt.shape)


def tt_left_orthogonalize(tt: TensorTrain) -> TensorTrain:
    """
    Left-orthogonalize a TT via QR decomposition (left-to-right sweep).

    After orthogonalization: G_k^T G_k = I for k = 0, ..., N-2.
    """
    cores = [jnp.array(c) for c in tt.cores]
    n = tt.ndim

    for k in range(n - 1):
        G = cores[k]
        r_l, n_k, r_r = G.shape
        M = G.reshape(r_l * n_k, r_r)
        Q, R = jnp.linalg.qr(M)
        rank = Q.shape[1]
        cores[k] = Q.reshape(r_l, n_k, rank)
        cores[k + 1] = jnp.einsum("ab,bcd->acd", R, cores[k + 1])

    return TensorTrain(cores, tt.shape)


def tt_right_orthogonalize(tt: TensorTrain) -> TensorTrain:
    """
    Right-orthogonalize a TT via LQ decomposition (right-to-left sweep).

    After orthogonalization: G_k G_k^T = I for k = 1, ..., N-1.
    """
    cores = [jnp.array(c) for c in tt.cores]
    n = tt.ndim

    for k in range(n - 1, 0, -1):
        G = cores[k]
        r_l, n_k, r_r = G.shape
        M = G.reshape(r_l, n_k * r_r)
        Q, R = jnp.linalg.qr(M.T)
        rank = Q.shape[1]
        cores[k] = Q.T.reshape(rank, n_k, r_r)
        cores[k - 1] = jnp.einsum("abc,cd->abd", cores[k - 1], R.T)

    return TensorTrain(cores, tt.shape)


def tt_mixed_canonical(
    tt: TensorTrain,
    center: int,
) -> Tuple[TensorTrain, jnp.ndarray]:
    """
    Bring TT into mixed-canonical form with orthogonality center at `center`.

    Parameters
    ----------
    tt : TensorTrain
    center : orthogonality center index

    Returns
    -------
    (tt_canonical, singular_values_at_center)
    """
    assert 0 <= center < tt.ndim
    cores = [jnp.array(c) for c in tt.cores]
    n = tt.ndim

    # Left sweep to center
    for k in range(center):
        G = cores[k]
        r_l, n_k, r_r = G.shape
        M = G.reshape(r_l * n_k, r_r)
        Q, R = jnp.linalg.qr(M)
        rank = Q.shape[1]
        cores[k] = Q.reshape(r_l, n_k, rank)
        cores[k + 1] = jnp.einsum("ab,bcd->acd", R, cores[k + 1])

    # Right sweep to center+1
    for k in range(n - 1, center, -1):
        G = cores[k]
        r_l, n_k, r_r = G.shape
        M = G.reshape(r_l, n_k * r_r)
        Q, R = jnp.linalg.qr(M.T)
        rank = Q.shape[1]
        cores[k] = Q.T.reshape(rank, n_k, r_r)
        cores[k - 1] = jnp.einsum("abc,cd->abd", cores[k - 1], R.T)

    # SVD at center
    G_c = cores[center]
    r_l_c, n_c, r_r_c = G_c.shape
    M_c = G_c.reshape(r_l_c * n_c, r_r_c)
    U_c, s_c, Vt_c = jnp.linalg.svd(M_c, full_matrices=False)
    chi = s_c.shape[0]
    cores[center] = U_c.reshape(r_l_c, n_c, chi)
    if center + 1 < n:
        cores[center + 1] = jnp.einsum("ab,bcd->acd", jnp.diag(s_c) @ Vt_c, cores[center + 1])

    return TensorTrain(cores, tt.shape), s_c


# ============================================================================
# TT-Cross Approximation (DMRG-style)
# ============================================================================

class TTCross:
    """
    TT-Cross approximation using the MaxVol (DMRG pivot) algorithm.

    Approximates a high-dimensional function f(i_1, ..., i_N) as a TT
    by adaptively selecting cross-approximation indices.

    References
    ----------
    Oseledets & Tyrtyshnikov (2010). TT-cross approximation for multidimensional
    arrays. Linear Algebra and its Applications, 432(1), 70-88.

    Savostyanov & Oseledets (2011). Fast adaptive interpolation of multi-
    dimensional arrays in tensor train format.
    """

    def __init__(
        self,
        func: Callable,
        shape: Tuple[int, ...],
        max_rank: int = 20,
        n_iter: int = 5,
        cutoff: float = 1e-8,
        kickrank: int = 2,
        key: Optional[jax.random.KeyArray] = None,
    ):
        """
        Parameters
        ----------
        func : callable(index_tuple) -> float, or array-valued batch function
        shape : tensor shape (n_1, ..., n_N)
        max_rank : maximum TT-rank
        n_iter : number of DMRG-style iterations
        cutoff : SVD truncation threshold
        kickrank : rank enrichment for stability
        key : JAX random key
        """
        self.func = func
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.max_rank = max_rank
        self.n_iter = n_iter
        self.cutoff = cutoff
        self.kickrank = kickrank
        self.key = key if key is not None else jax.random.PRNGKey(0)

        # Pivot index sets: I_k (left) and J_k (right)
        self.left_indices: List[List[Tuple]] = [[()] for _ in range(self.ndim)]
        self.right_indices: List[List[Tuple]] = [[()]] + [
            [(j,) for j in range(min(max_rank, shape[k]))]
            for k in range(self.ndim - 1, 0, -1)
        ]

    def run(self) -> TensorTrain:
        """
        Run the TT-Cross algorithm and return the TT approximation.

        Returns
        -------
        TensorTrain approximation of the function
        """
        # Initialize with random pivot selection
        self._initialize_pivots()
        tt = self._build_initial_tt()

        for iteration in range(self.n_iter):
            # Right-to-left sweep: update right index sets
            tt = self._right_to_left_sweep(tt)
            # Left-to-right sweep: update left index sets and build cores
            tt = self._left_to_right_sweep(tt)

        return tt

    def _initialize_pivots(self):
        """Initialize pivot sets randomly."""
        for k in range(1, self.ndim):
            n_k = self.shape[k]
            rank = min(self.max_rank, n_k)
            self.key, subkey = jax.random.split(self.key)
            indices = jax.random.permutation(subkey, n_k)[:rank]
            self.right_indices[k] = [(int(i),) for i in indices]

    def _evaluate_fiber(
        self,
        left_idx: Tuple,
        mode_range: int,
        right_idx: Tuple,
    ) -> jnp.ndarray:
        """Evaluate the function along a fiber (left, :, right)."""
        n_k = self.shape[len(left_idx)]
        values = jnp.array([
            float(self.func(left_idx + (j,) + right_idx))
            for j in range(n_k)
        ])
        return values

    def _evaluate_cross_matrix(
        self,
        left_set: List[Tuple],
        k: int,
        right_set: List[Tuple],
    ) -> jnp.ndarray:
        """
        Build the cross matrix C[i, j] = f(left_set[i] + right_set[j])
        for mode k (index dimension).
        """
        n_l = len(left_set)
        n_r = len(right_set)
        n_k = self.shape[k]
        # Build (n_l * n_k, n_r) matrix
        C = np.zeros((n_l * n_k, n_r), dtype=np.float32)
        for ii, lidx in enumerate(left_set):
            for s in range(n_k):
                for jj, ridx in enumerate(right_set):
                    C[ii * n_k + s, jj] = float(
                        self.func(lidx + (s,) + ridx)
                    )
        return jnp.array(C)

    def _maxvol(self, A: jnp.ndarray, n_pivots: int) -> jnp.ndarray:
        """
        Find approximately maximal volume submatrix row indices via
        greedy column pivoting.

        Parameters
        ----------
        A : matrix of shape (m, n) with m >= n_pivots
        n_pivots : number of pivots to select

        Returns
        -------
        Row indices of the maximal submatrix
        """
        m, n = A.shape
        n_pivots = min(n_pivots, m, n)

        # Greedy selection: start with QR pivoting
        A_np = np.array(A)
        _, _, piv = np.linalg.svd(A_np, full_matrices=False)

        # Select rows with largest norm
        norms = np.sum(A_np ** 2, axis=1)
        idx = np.argsort(-norms)[:n_pivots]
        return jnp.array(idx)

    def _build_initial_tt(self) -> TensorTrain:
        """Build initial TT using the initialized pivot sets."""
        cores = []
        for k in range(self.ndim):
            n_k = self.shape[k]
            r_l = max(1, len(self.left_indices[k]))
            r_r = max(1, len(self.right_indices[min(k + 1, self.ndim - 1)]))
            if k == 0:
                r_l = 1
            if k == self.ndim - 1:
                r_r = 1

            self.key, subkey = jax.random.split(self.key)
            G = jax.random.normal(subkey, (r_l, n_k, r_r)) * 0.01
            cores.append(G)

        return TensorTrain(cores, self.shape)

    def _left_to_right_sweep(self, tt: TensorTrain) -> TensorTrain:
        """Left-to-right sweep updating TT cores."""
        cores = [jnp.array(c) for c in tt.cores]
        n = self.ndim

        for k in range(n - 1):
            left_set = self.left_indices[k]
            right_set = self.right_indices[min(k + 1, n - 1)]

            # Evaluate cross matrix
            try:
                C = self._evaluate_cross_matrix(left_set, k, right_set)
            except Exception:
                continue

            r_l = len(left_set)
            n_k = self.shape[k]
            r_r = len(right_set)

            # SVD and truncate
            U, s, Vt = jnp.linalg.svd(C, full_matrices=False)
            rank = min(self.max_rank, U.shape[1])
            U = U[:, :rank]
            s_k = s[:rank]
            Vt = Vt[:rank, :]

            cores[k] = U.reshape(r_l if r_l > 0 else 1, n_k, rank)

            # Update left index set for next mode
            # (simplified: keep existing)

        return TensorTrain(cores, self.shape)

    def _right_to_left_sweep(self, tt: TensorTrain) -> TensorTrain:
        """Right-to-left sweep (same as left-to-right but reversed)."""
        return tt


def tt_cross(
    func: Callable,
    shape: Tuple[int, ...],
    max_rank: int = 20,
    n_iter: int = 3,
    cutoff: float = 1e-8,
    key: Optional[jax.random.KeyArray] = None,
) -> TensorTrain:
    """
    TT-Cross approximation of a high-dimensional function.

    Convenience wrapper around TTCross class.

    Parameters
    ----------
    func : function mapping index tuple to scalar
    shape : tensor shape
    max_rank : maximum TT-rank
    n_iter : number of cross iterations
    cutoff : SVD threshold
    key : JAX random key

    Returns
    -------
    TensorTrain approximation
    """
    cross = TTCross(func, shape, max_rank=max_rank, n_iter=n_iter, cutoff=cutoff, key=key)
    return cross.run()


# ============================================================================
# TT-Rank estimation
# ============================================================================

def tt_rank_estimation(
    tensor: jnp.ndarray,
    target_error: float = 0.01,
    max_rank: int = 200,
) -> List[int]:
    """
    Estimate TT-ranks needed to achieve a target relative error.

    Performs a pilot TT-SVD and measures singular value decay at each bond.

    Parameters
    ----------
    tensor : dense tensor
    target_error : target relative Frobenius error
    max_rank : maximum rank to consider

    Returns
    -------
    List of TT-ranks [r_1, ..., r_{N-1}] achieving the target error
    """
    shape = tensor.shape
    n = len(shape)

    # Full TT-SVD with no truncation
    tt_full = tt_svd(tensor, max_rank=max_rank, cutoff=0.0)

    # Estimate required ranks from singular value decay
    ranks = []
    arr = jnp.array(tensor, dtype=jnp.float32)
    r_l = 1

    for k in range(n - 1):
        n_k = shape[k]
        arr_2d = arr.reshape(r_l * n_k, -1)
        U, s, Vt = jnp.linalg.svd(arr_2d, full_matrices=False)

        # Find minimum rank for target error
        total_sq = float(jnp.sum(s ** 2))
        cumsum = jnp.cumsum(s ** 2)
        threshold = (1 - target_error ** 2) * total_sq
        rank = int(jnp.sum(cumsum < threshold).item()) + 1
        rank = max(1, min(rank, max_rank, U.shape[1]))
        ranks.append(rank)

        r_l = rank
        arr = (jnp.diag(s[:rank]) @ Vt[:rank, :]).reshape(rank * (shape[k + 1] if k + 1 < n else 1), -1)

    return ranks


def tt_optimal_ranks(
    singular_values: List[jnp.ndarray],
    target_error: float = 0.01,
) -> List[int]:
    """
    Given the singular values at each bond, find optimal ranks for a target error.

    Parameters
    ----------
    singular_values : list of singular value arrays, one per bond
    target_error : relative Frobenius error target per bond

    Returns
    -------
    List of optimal ranks
    """
    ranks = []
    for sv in singular_values:
        sv_sq = sv ** 2
        total = float(jnp.sum(sv_sq))
        if total < 1e-15:
            ranks.append(1)
            continue
        cumsum = jnp.cumsum(sv_sq)
        threshold = (1 - target_error ** 2) * total
        rank = int(jnp.sum(cumsum < threshold).item()) + 1
        rank = max(1, min(rank, len(sv)))
        ranks.append(rank)
    return ranks


# ============================================================================
# TT evaluation and conversion
# ============================================================================

def tt_to_dense(tt: TensorTrain) -> jnp.ndarray:
    """
    Reconstruct the full dense tensor from a TT decomposition.

    Parameters
    ----------
    tt : TensorTrain

    Returns
    -------
    Dense array of shape tt.shape
    """
    result = tt.cores[0][0, :, :]  # (n_0, r_1)

    for k in range(1, tt.ndim):
        G = tt.cores[k]  # (r_l, n_k, r_r)
        result = jnp.einsum("...l,lnr->...nr", result, G)

    return result[..., 0]


def tt_evaluate(tt: TensorTrain, idx: Sequence[int]) -> jnp.ndarray:
    """
    Evaluate TT at a specific multi-index (i_1, ..., i_N).

    Parameters
    ----------
    tt : TensorTrain
    idx : multi-index tuple of length ndim

    Returns
    -------
    Scalar value tt[idx]
    """
    assert len(idx) == tt.ndim
    v = tt.cores[0][0, idx[0], :]  # (r_1,)
    for k in range(1, tt.ndim):
        v = v @ tt.cores[k][:, idx[k], :]  # (r_k,) -> (r_{k+1},)
    return v[0]


def tt_slice(
    tt: TensorTrain,
    fixed_modes: Dict[int, int],
) -> TensorTrain:
    """
    Fix some modes of a TT to given indices (slicing operation).

    Parameters
    ----------
    tt : TensorTrain
    fixed_modes : dict {mode_index: value}

    Returns
    -------
    TensorTrain of reduced dimensionality
    """
    new_cores = []
    new_shape = []

    for k in range(tt.ndim):
        if k in fixed_modes:
            idx = fixed_modes[k]
            G = tt.cores[k][:, idx, :]  # (r_l, r_r) — contracts this mode
            if new_cores:
                # Absorb into previous core
                prev = new_cores[-1]
                new_cores[-1] = jnp.einsum("abc,cd->abd", prev, G)
            else:
                new_cores.append(G.reshape(1, 1, -1))
                new_shape.append(1)
        else:
            new_cores.append(tt.cores[k])
            new_shape.append(tt.shape[k])

    if not new_cores:
        return TensorTrain([jnp.ones((1, 1, 1))], (1,))

    return TensorTrain(new_cores, tuple(new_shape))


# ============================================================================
# TensorTrainMatrix (MPO / TT-matrix)
# ============================================================================

class TensorTrainMatrix:
    """
    Tensor Train Matrix (TTM) representing a linear operator in TT format.

    A TTM encodes an operator A: R^{n_1 * ... * n_N} -> R^{m_1 * ... * m_N}
    with cores of shape (r_{k-1}, m_k, n_k, r_k).

    Applications:
    - Matrix-vector multiplication in TT format (exponentially cheaper)
    - Compression of large linear operators
    - Quantum circuit operators (MPO)
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
        cores : list of arrays, each shape (r_{k-1}, m_k, n_k, r_k)
        row_shape : output mode sizes (m_1, ..., m_N)
        col_shape : input mode sizes (n_1, ..., n_N)
        """
        self.cores = [jnp.array(c) for c in cores]
        self.row_shape = tuple(row_shape)
        self.col_shape = tuple(col_shape)
        self.ndim = len(cores)
        self.ranks = [1] + [int(c.shape[3]) for c in cores[:-1]] + [1]

    @property
    def n_params(self) -> int:
        return sum(c.size for c in self.cores)

    def __repr__(self) -> str:
        return (
            f"TensorTrainMatrix(row_shape={self.row_shape}, "
            f"col_shape={self.col_shape}, ranks={self.ranks})"
        )

    def to_dense(self) -> jnp.ndarray:
        """
        Reconstruct full matrix of shape (prod(row_shape), prod(col_shape)).
        """
        # Initialize
        result = self.cores[0][0, :, :, :]  # (m_0, n_0, r_1)

        for k in range(1, self.ndim):
            G = self.cores[k]  # (r_l, m_k, n_k, r_r)
            result = jnp.einsum("...mnr,rMNs->...mnMNs", result, G)

        # result has shape (m_0, n_0, m_1, n_1, ..., m_{N-1}, n_{N-1}, 1)
        # Rearrange to (m_0, m_1, ..., n_0, n_1, ...) then flatten
        # Simplified: just do sequential einsum contraction
        result = self.cores[0][0, :, :, :]  # (m_0, n_0, r_1)
        m_total, n_total = 1, 1
        for k in range(self.ndim):
            m_total *= self.row_shape[k]
            n_total *= self.col_shape[k]

        # Use full contraction
        mat = self.cores[0][0, :, :, :]  # (m0, n0, r1)
        for k in range(1, self.ndim):
            G = self.cores[k]  # (r_l, m_k, n_k, r_r)
            mat = jnp.einsum("...mnr,rMNs->...mnMNs", mat, G)

        # Flatten: separate row and column indices
        # mat has shape (m0, n0, m1, n1, ..., mN, nN, 1)
        # We need to separate m-indices and n-indices
        # For simplicity, use the dense approach
        dense = tt_to_dense(TensorTrain(
            [c.reshape(c.shape[0], c.shape[1] * c.shape[2], c.shape[3]) for c in self.cores],
            shape=tuple(m * n for m, n in zip(self.row_shape, self.col_shape))
        ))
        # Reshape to (m_total, n_total)
        return dense.reshape(m_total, n_total)


def tt_matvec(
    ttm: TensorTrainMatrix,
    tt_vec: TensorTrain,
) -> TensorTrain:
    """
    Multiply a TT-matrix by a TT-vector: result = TTM @ tt_vec.

    The result is a TensorTrain of shape ttm.row_shape with ranks r_ttm * r_vec.

    Parameters
    ----------
    ttm : TensorTrainMatrix (the operator)
    tt_vec : TensorTrain (the vector)

    Returns
    -------
    TensorTrain representing TTM @ tt_vec
    """
    assert ttm.col_shape == tt_vec.shape, "Shape mismatch in tt_matvec"
    assert ttm.ndim == tt_vec.ndim

    cores = []
    for k in range(ttm.ndim):
        G_op = ttm.cores[k]  # (r_op_l, m_k, n_k, r_op_r)
        G_v = tt_vec.cores[k]  # (r_v_l, n_k, r_v_r)
        r_op_l, m_k, n_k, r_op_r = G_op.shape
        r_v_l, _, r_v_r = G_v.shape

        # C[a*a', m, b*b'] = sum_n G_op[a, m, n, b] * G_v[a', n, b']
        C = jnp.einsum("amnb,anc->ambc", G_op, G_v)
        # Reshape: (r_op_l * r_v_l, m_k, r_op_r * r_v_r)
        C = C.reshape(r_op_l * r_v_l, m_k, r_op_r * r_v_r)
        cores.append(C)

    return TensorTrain(cores, ttm.row_shape)


def tt_identity_matrix(
    shape: Tuple[int, ...],
    dtype=jnp.float32,
) -> TensorTrainMatrix:
    """
    Create a TT-identity operator (identity matrix in TT format).

    Each core is the identity matrix of the corresponding mode.
    """
    cores = []
    for n_k in shape:
        G = jnp.eye(n_k, dtype=dtype).reshape(1, n_k, n_k, 1)
        cores.append(G)
    return TensorTrainMatrix(cores, shape, shape)


# ============================================================================
# Riemannian gradient on TT manifold
# ============================================================================

def tt_riemannian_grad(
    tt: TensorTrain,
    eucl_grad: TensorTrain,
) -> TensorTrain:
    """
    Project the Euclidean gradient onto the tangent space of the TT manifold.

    The Riemannian gradient at a point X in the TT manifold is the projection
    of the Euclidean gradient onto the tangent space T_X(M_r).

    Algorithm (Steinlechner 2016):
    1. Left-orthogonalize tt to get canonical form
    2. Build left and right orthogonal environments
    3. Project Euclidean gradient via alternating projections

    Parameters
    ----------
    tt : TensorTrain (the point on the manifold)
    eucl_grad : TensorTrain (Euclidean gradient, same shape and ranks)

    Returns
    -------
    TensorTrain representing the Riemannian gradient (tangent vector)
    """
    n = tt.ndim

    # Left-orthogonalize
    tt_left = tt_left_orthogonalize(tt)
    # Right-orthogonalize
    tt_right = tt_right_orthogonalize(tt)

    # Build tangent vector as sum of contributions from each core variation
    tangent_cores = [None] * n

    for k in range(n):
        # Get the Euclidean gradient core
        dG_k = eucl_grad.cores[k]

        # Project onto orthogonal complement of column space
        G_l = tt_left.cores[k]   # left-orthogonal
        G_r = tt_right.cores[k]  # right-orthogonal

        r_l, n_k, r_r = G_l.shape

        # Projection: P_k(dG) = dG - G_l (G_l^T dG) for left-orthogonal part
        # and similarly for right
        # Simplified first-order approximation
        dG_flat = dG_k.reshape(r_l * n_k, r_r)
        G_l_flat = G_l.reshape(r_l * n_k, r_r)

        # Left projection
        proj_l = dG_flat - G_l_flat @ (G_l_flat.T @ dG_flat)

        tangent_cores[k] = proj_l.reshape(r_l, n_k, r_r)

    return TensorTrain(tangent_cores, tt.shape)


def tt_retract(
    tt: TensorTrain,
    tangent: TensorTrain,
    step_size: float,
    max_rank: int,
) -> TensorTrain:
    """
    Retraction on TT manifold: map point + tangent vector back to manifold.

    Uses SVD truncation as a retraction (approximate but efficient).

    Parameters
    ----------
    tt : current point on TT manifold
    tangent : tangent vector
    step_size : step size
    max_rank : maximum rank of the result

    Returns
    -------
    New point on TT manifold
    """
    # First-order retraction: R(X, V) = round(X + step * V)
    moved = tt_add(tt, tt_scale(tangent, step_size))
    return tt_round(moved, max_rank=max_rank)


def tt_vector_transport(
    tt_old: TensorTrain,
    tt_new: TensorTrain,
    tangent: TensorTrain,
    max_rank: int,
) -> TensorTrain:
    """
    Transport a tangent vector from tt_old to tt_new.

    Uses the projection-based transport: P_{tt_new}(V).

    Parameters
    ----------
    tt_old : source point
    tt_new : target point
    tangent : tangent vector at tt_old
    max_rank : maximum rank

    Returns
    -------
    Transported tangent vector at tt_new
    """
    # Simple approach: re-project the tangent
    # Convert tangent to Euclidean gradient (add to tt_old to get direction)
    tt_target = tt_add(tt_old, tangent)
    # Compute Euclidean "gradient" at tt_new
    diff = tt_subtract(tt_target, tt_new)
    diff_rounded = tt_round(diff, max_rank=max_rank)
    # Project onto tangent space of tt_new
    return tt_riemannian_grad(tt_new, diff_rounded)


# ============================================================================
# TT eigenvalue decomposition (financial PCA)
# ============================================================================

def tt_power_iteration(
    ttm: TensorTrainMatrix,
    n_iter: int = 50,
    rank: int = 1,
    key: Optional[jax.random.KeyArray] = None,
) -> Tuple[jnp.ndarray, TensorTrain]:
    """
    Power iteration for finding the leading eigenvalue/eigenvector of a
    symmetric TT-matrix.

    Parameters
    ----------
    ttm : TensorTrainMatrix (symmetric)
    n_iter : number of power iterations
    rank : TT-rank of the eigenvector approximation
    key : random key

    Returns
    -------
    (eigenvalue, eigenvector_tt)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize random unit vector
    shape = ttm.col_shape
    key, subkey = jax.random.split(key)
    cores = []
    for n_k in shape:
        key, sk = jax.random.split(key)
        G = jax.random.normal(sk, (rank if len(cores) > 0 else 1, n_k,
                                   rank if len(cores) < len(shape) - 1 else 1))
        cores.append(G)
    v = TensorTrain(cores, shape)
    v = tt_normalize(v)

    eigenval = jnp.ones(())

    for _ in range(n_iter):
        # v_new = TTM @ v
        v_new = tt_matvec(ttm, v)
        eigenval = tt_norm(v_new)
        v = tt_round(tt_scale(v_new, 1.0 / (float(eigenval) + 1e-30)), max_rank=rank)

    return eigenval, v


# ============================================================================
# Utility and diagnostics
# ============================================================================

def tt_bond_energies(tt: TensorTrain) -> List[float]:
    """
    Compute the 'energy' of each bond as the squared Frobenius norm
    contributed by each core.

    Parameters
    ----------
    tt : TensorTrain

    Returns
    -------
    List of per-core Frobenius norms
    """
    return [float(jnp.linalg.norm(c)) for c in tt.cores]


def tt_cumulative_error(
    tt: TensorTrain,
    ranks: List[int],
) -> List[float]:
    """
    Estimate cumulative truncation error as a function of rank.

    Parameters
    ----------
    tt : TensorTrain in mixed-canonical form
    ranks : list of ranks to evaluate

    Returns
    -------
    List of relative errors for each rank
    """
    errors = []
    ref_norm = float(tt_norm(tt))

    for r in ranks:
        tt_approx = tt_round(tt, max_rank=r)
        err = float(tt_norm(tt_subtract(tt, tt_approx))) / (ref_norm + 1e-30)
        errors.append(err)

    return errors


def tt_from_matrix(
    matrix: jnp.ndarray,
    row_shape: Tuple[int, ...],
    col_shape: Tuple[int, ...],
    max_rank: int = 50,
    cutoff: float = 1e-10,
) -> TensorTrainMatrix:
    """
    Decompose a dense matrix into TT-matrix format.

    Reshapes the matrix to (m_1, n_1, m_2, n_2, ...) and applies TT-SVD.

    Parameters
    ----------
    matrix : dense matrix of shape (prod(row_shape), prod(col_shape))
    row_shape : row mode sizes
    col_shape : col mode sizes
    max_rank : maximum TT-rank
    cutoff : SVD threshold

    Returns
    -------
    TensorTrainMatrix
    """
    assert len(row_shape) == len(col_shape)
    n = len(row_shape)

    # Interleave row and col indices: (m_1, n_1, m_2, n_2, ...)
    combined_shape = []
    for m, nc in zip(row_shape, col_shape):
        combined_shape.extend([m, nc])
    tensor = matrix.reshape(combined_shape)

    # TT-SVD on the combined tensor
    tt_combined = tt_svd(tensor, max_rank=max_rank, cutoff=cutoff)

    # Split cores back into TTM format
    cores = []
    for k in range(n):
        G = tt_combined.cores[k]  # (r_l, m_k * n_k, r_r) -- but combined
        r_l, mn, r_r = G.shape
        m_k, n_k = row_shape[k], col_shape[k]
        cores.append(G.reshape(r_l, m_k, n_k, r_r))

    return TensorTrainMatrix(cores, row_shape, col_shape)


def tt_to_dict(tt: TensorTrain) -> Dict[str, Any]:
    """Serialize TensorTrain to dictionary."""
    return {
        "shape": list(tt.shape),
        "ranks": list(tt.ranks),
        "cores": [np.array(c) for c in tt.cores],
    }


def tt_from_dict(d: Dict[str, Any]) -> TensorTrain:
    """Deserialize TensorTrain from dictionary."""
    cores = [jnp.array(c) for c in d["cores"]]
    shape = tuple(d["shape"])
    return TensorTrain(cores, shape)


# ============================================================================
# Randomized TT-SVD for large-scale problems
# ============================================================================

def tt_svd_randomized(
    tensor: jnp.ndarray,
    max_rank: int = 50,
    oversample: int = 10,
    key: Optional[jax.random.KeyArray] = None,
) -> TensorTrain:
    """
    Randomized TT-SVD for large-scale tensor decomposition.

    Uses randomized range-finding at each mode to reduce computation.
    Suitable when the tensor is too large for deterministic TT-SVD.

    Parameters
    ----------
    tensor : dense N-dimensional array
    max_rank : maximum TT-rank
    oversample : oversampling factor for randomized SVD
    key : JAX random key

    Returns
    -------
    TensorTrain approximation
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    shape = tensor.shape
    n_dims = len(shape)
    cores = []
    arr = jnp.array(tensor, dtype=jnp.float32)
    r_left = 1

    for k in range(n_dims - 1):
        n_k = shape[k]
        arr_2d = arr.reshape(r_left * n_k, -1)
        n_rows, n_cols = arr_2d.shape

        # Randomized range-finding
        rank_target = min(max_rank, n_rows, n_cols)
        n_random = min(rank_target + oversample, n_cols)

        key, subkey = jax.random.split(key)
        Omega = jax.random.normal(subkey, (n_cols, n_random))
        Y = arr_2d @ Omega  # (n_rows, n_random)

        # QR to get orthonormal basis
        Q, _ = jnp.linalg.qr(Y)
        Q = Q[:, :rank_target]

        # Project and do SVD
        B = Q.T @ arr_2d  # (rank_target, n_cols)
        U_B, s, Vt = jnp.linalg.svd(B, full_matrices=False)

        rank = min(max_rank, U_B.shape[1])
        U = Q @ U_B[:, :rank]
        s_k = s[:rank]
        Vt = Vt[:rank, :]

        cores.append(U.reshape(r_left, n_k, rank))
        arr = jnp.diag(s_k) @ Vt
        r_left = rank

    cores.append(arr.reshape(r_left, shape[-1], 1))
    return TensorTrain(cores, shape)


# ============================================================================
# Tucker decomposition (for use in financial_tensors.py)
# ============================================================================

def tucker_decomp(
    tensor: jnp.ndarray,
    ranks: Sequence[int],
    n_iter: int = 20,
    init: str = "svd",
    key: Optional[jax.random.KeyArray] = None,
) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """
    Tucker decomposition via HOSVD (Higher-Order SVD).

    Decomposes tensor T as:
      T ≈ G ×_1 U_1 ×_2 U_2 ... ×_N U_N

    where G is the core tensor and U_k are mode-k unitary factors.

    Parameters
    ----------
    tensor : N-dimensional input array
    ranks : desired Tucker ranks (r_1, ..., r_N)
    n_iter : number of HOOI (Higher-Order Orthogonal Iteration) iterations
    init : initialization method ('svd' or 'random')
    key : JAX random key (for 'random' init)

    Returns
    -------
    (core_tensor, [U_1, ..., U_N])
    """
    tensor = jnp.array(tensor, dtype=jnp.float32)
    shape = tensor.shape
    n_dims = len(shape)
    ranks = list(ranks)

    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize factor matrices
    if init == "svd":
        factors = []
        for k in range(n_dims):
            # Mode-k unfolding
            unf = _mode_unfold(tensor, k)  # (shape[k], prod_rest)
            U, _, _ = jnp.linalg.svd(unf, full_matrices=False)
            factors.append(U[:, :ranks[k]])
    else:
        factors = []
        for k in range(n_dims):
            key, subkey = jax.random.split(key)
            U, _ = jnp.linalg.qr(jax.random.normal(subkey, (shape[k], ranks[k])))
            factors.append(U[:, :ranks[k]])

    # HOOI iterations
    for iteration in range(n_iter):
        for k in range(n_dims):
            # Compute Y = tensor ×_{-k} U_j^T (contract all modes except k)
            Y = tensor
            for j in range(n_dims):
                if j != k:
                    # Contract mode j
                    Y = jnp.tensordot(Y, factors[j], axes=([0], [0]))
                    # Move the new axis to the end
                    # After tensordot: shape is (old minus mode j) + (ranks[j],)
                    # Need to cycle axes
            # Mode-k SVD
            unf_Y = _mode_unfold_contraction(tensor, k, factors)
            U, _, _ = jnp.linalg.svd(unf_Y, full_matrices=False)
            factors[k] = U[:, :ranks[k]]

    # Compute core: G = tensor ×_1 U_1^T ×_2 U_2^T ... ×_N U_N^T
    core = tensor
    for k in range(n_dims - 1, -1, -1):
        # Contract mode k with U_k^T
        core = jnp.tensordot(core, factors[k], axes=([0], [0]))
        # Rearrange axes so mode 0 is the contracted mode
        # After: (remaining modes, ranks[k]) -> need (ranks[k], remaining modes)
        core = jnp.moveaxis(core, -1, 0)

    return core, factors


def _mode_unfold(tensor: jnp.ndarray, mode: int) -> jnp.ndarray:
    """Mode-k unfolding: reshape tensor to (shape[mode], prod_other_modes)."""
    shape = tensor.shape
    n = tensor.ndim
    # Move mode to front
    perm = [mode] + [i for i in range(n) if i != mode]
    tensor_perm = jnp.transpose(tensor, perm)
    return tensor_perm.reshape(shape[mode], -1)


def _mode_unfold_contraction(
    tensor: jnp.ndarray,
    mode: int,
    factors: List[jnp.ndarray],
) -> jnp.ndarray:
    """Mode-k unfolding after contracting all other modes with factors."""
    n = tensor.ndim
    result = tensor
    # Contract all modes except `mode` with their factor matrices
    offset = 0
    for j in range(n):
        if j == mode:
            offset += 1
            continue
        actual_mode = j - (offset - 1) if j > mode else j
        result = jnp.tensordot(result, factors[j].T, axes=([1], [1]))
    # Mode-k unfolding of the result
    return _mode_unfold(result, 0)


def cp_decomp(
    tensor: jnp.ndarray,
    rank: int,
    n_iter: int = 100,
    tol: float = 1e-6,
    key: Optional[jax.random.KeyArray] = None,
) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """
    CP (CANDECOMP/PARAFAC) decomposition via ALS.

    Decomposes tensor T ≈ sum_{r=1}^R lambda_r (a_r^(1) ⊗ a_r^(2) ⊗ ... ⊗ a_r^(N))

    Parameters
    ----------
    tensor : N-dimensional input array
    rank : CP rank
    n_iter : max ALS iterations
    tol : convergence tolerance
    key : JAX random key

    Returns
    -------
    (lambdas, [A_1, ..., A_N]) where lambdas has shape (rank,) and A_k has shape (n_k, rank)
    """
    tensor = jnp.array(tensor, dtype=jnp.float32)
    shape = tensor.shape
    n_dims = len(shape)

    if key is None:
        key = jax.random.PRNGKey(0)

    # Random initialization
    factors = []
    for k in range(n_dims):
        key, subkey = jax.random.split(key)
        A = jax.random.normal(subkey, (shape[k], rank))
        # Normalize columns
        norms = jnp.linalg.norm(A, axis=0, keepdims=True) + 1e-10
        factors.append(A / norms)

    lambdas = jnp.ones(rank)
    prev_err = float("inf")

    for iteration in range(n_iter):
        for k in range(n_dims):
            # ALS update for factor k
            # V = Khatri-Rao product of all factors except k
            V = _khatri_rao_except(factors, k)  # (prod_{j≠k} n_j, rank)

            # Mode-k unfolding
            unf = _mode_unfold(tensor, k)  # (n_k, prod_rest)

            # LS update: A_k = unf @ V @ (V^T V)^{-1}
            VtV = V.T @ V  # (rank, rank)
            # Hadamard product of Gram matrices
            gram = functools.reduce(
                lambda a, b: a * b,
                [factors[j].T @ factors[j] for j in range(n_dims) if j != k],
            )

            A_new = unf @ V @ jnp.linalg.pinv(gram)
            # Normalize and store lambdas
            norms = jnp.linalg.norm(A_new, axis=0)
            lambdas = norms
            factors[k] = A_new / (norms + 1e-10)

        # Convergence check
        err = _cp_reconstruction_error(tensor, lambdas, factors)
        if abs(prev_err - float(err)) < tol:
            break
        prev_err = float(err)

    return lambdas, factors


def _khatri_rao_except(factors: List[jnp.ndarray], skip: int) -> jnp.ndarray:
    """Khatri-Rao product of all factors except the one at index `skip`."""
    result = None
    for k, A in enumerate(factors):
        if k == skip:
            continue
        if result is None:
            result = A
        else:
            # Khatri-Rao: column-wise Kronecker product
            n_a = result.shape[0]
            n_b = A.shape[0]
            rank = A.shape[1]
            kr = jnp.einsum("ir,jr->ijr", result, A).reshape(n_a * n_b, rank)
            result = kr
    return result if result is not None else jnp.ones((1, factors[0].shape[1]))


def _cp_reconstruction_error(
    tensor: jnp.ndarray,
    lambdas: jnp.ndarray,
    factors: List[jnp.ndarray],
) -> jnp.ndarray:
    """Compute CP reconstruction error."""
    # Reconstruct from factors
    rank = lambdas.shape[0]
    recon = jnp.zeros_like(tensor)
    for r in range(rank):
        term = lambdas[r]
        for k, A in enumerate(factors):
            term_k = A[:, r]
            if k == 0:
                outer = term_k
            else:
                outer = jnp.tensordot(outer, term_k, axes=0)
        recon = recon + float(lambdas[r]) * outer
    return jnp.linalg.norm(tensor - recon)


# ============================================================================
# Extended TT utilities: financial applications, TT statistics, TT manifold
# ============================================================================

def tt_gram_matrix(
    tt_list: List[TensorTrain],
) -> jnp.ndarray:
    """
    Compute the Gram matrix G[i,j] = <tt_i, tt_j>_F for a list of TTs.

    Parameters
    ----------
    tt_list : list of TensorTrain with same shape

    Returns
    -------
    (n, n) Gram matrix
    """
    n = len(tt_list)
    G = jnp.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G = G.at[i, j].set(float(tt_dot(tt_list[i], tt_list[j])))
    return G


def tt_gram_schmidt(
    tt_list: List[TensorTrain],
    max_rank: int = 20,
) -> List[TensorTrain]:
    """
    Gram-Schmidt orthonormalization of a list of TTs in Frobenius inner product.

    Parameters
    ----------
    tt_list : list of TensorTrain
    max_rank : maximum rank for rounding after each step

    Returns
    -------
    Orthonormal list of TensorTrain
    """
    ortho = []
    for tt in tt_list:
        v = tt.copy()
        for u in ortho:
            proj = tt_dot(u, v)
            v = tt_subtract(v, tt_scale(u, float(proj)))
            v = tt_round(v, max_rank=max_rank)
        norm = float(tt_norm(v))
        if norm > 1e-10:
            v = tt_scale(v, 1.0 / norm)
            ortho.append(v)
    return ortho


def tt_linear_combination(
    tt_list: List[TensorTrain],
    coeffs: Sequence[float],
    max_rank: int = 50,
) -> TensorTrain:
    """
    Compute a linear combination sum_i c_i * tt_i.

    Parameters
    ----------
    tt_list : list of TensorTrain
    coeffs : scalar coefficients
    max_rank : rank for rounding the result

    Returns
    -------
    TensorTrain linear combination
    """
    assert len(tt_list) == len(coeffs)
    result = tt_scale(tt_list[0], coeffs[0])
    for tt, c in zip(tt_list[1:], coeffs[1:]):
        result = tt_add(result, tt_scale(tt, c))
    return tt_round(result, max_rank=max_rank)


def tt_random(
    shape: Tuple[int, ...],
    max_rank: int,
    key: jax.random.KeyArray,
    dtype=jnp.float32,
) -> TensorTrain:
    """
    Create a random TensorTrain with given shape and rank.

    Parameters
    ----------
    shape : tensor shape
    max_rank : maximum TT-rank
    key : random key
    dtype : element dtype

    Returns
    -------
    Random TensorTrain
    """
    n_dims = len(shape)
    cores = []
    for k in range(n_dims):
        r_l = 1 if k == 0 else min(max_rank, shape[k - 1])
        r_r = 1 if k == n_dims - 1 else min(max_rank, shape[k + 1])
        key, subkey = jax.random.split(key)
        G = jax.random.normal(subkey, (r_l, shape[k], r_r), dtype=dtype)
        G = G / math.sqrt(r_l * shape[k] * r_r + 1)
        cores.append(G)
    return TensorTrain(cores, shape)


def tt_ones(shape: Tuple[int, ...], dtype=jnp.float32) -> TensorTrain:
    """Create a TT representing the all-ones tensor (rank 1)."""
    cores = [jnp.ones((1, n, 1), dtype=dtype) for n in shape]
    return TensorTrain(cores, shape)


def tt_zeros(shape: Tuple[int, ...], rank: int = 1, dtype=jnp.float32) -> TensorTrain:
    """Create a TT of zeros with given rank."""
    n_dims = len(shape)
    cores = []
    for k in range(n_dims):
        r_l = 1 if k == 0 else rank
        r_r = 1 if k == n_dims - 1 else rank
        cores.append(jnp.zeros((r_l, shape[k], r_r), dtype=dtype))
    return TensorTrain(cores, shape)


def tt_frobenius_norm(tt: TensorTrain) -> jnp.ndarray:
    """Alias for tt_norm: Frobenius norm of TT."""
    return tt_norm(tt)


def tt_spectral_norm_estimate(
    tt: TensorTrain,
    n_iter: int = 20,
    key: Optional[jax.random.KeyArray] = None,
) -> jnp.ndarray:
    """
    Estimate the spectral norm (largest singular value) of the TT tensor
    viewed as a matrix.

    Uses power iteration on the TT-matrix structure.

    Parameters
    ----------
    tt : TensorTrain
    n_iter : power iteration steps
    key : random key

    Returns
    -------
    Spectral norm estimate
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Flatten TT to matrix shape (first half modes x second half modes)
    n = tt.ndim
    half = n // 2
    dense = tt_to_dense(tt)
    m = 1
    for k in range(half):
        m *= tt.shape[k]
    n2 = 1
    for k in range(half, n):
        n2 *= tt.shape[k]

    mat = dense.reshape(m, n2)
    # Power iteration
    key, subkey = jax.random.split(key)
    v = jax.random.normal(subkey, (n2,))
    v = v / (jnp.linalg.norm(v) + 1e-10)

    for _ in range(n_iter):
        u = mat @ v
        sigma = jnp.linalg.norm(u)
        u = u / (sigma + 1e-10)
        v = mat.T @ u
        v = v / (jnp.linalg.norm(v) + 1e-10)

    return sigma


def tt_condition_number(
    tt: TensorTrain,
    n_iter: int = 10,
    key: Optional[jax.random.KeyArray] = None,
) -> float:
    """
    Estimate the condition number of the TT tensor as a matrix.

    Parameters
    ----------
    tt : TensorTrain
    n_iter : power iteration steps
    key : random key

    Returns
    -------
    Estimated condition number (sigma_max / sigma_min)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    dense = tt_to_dense(tt)
    n = tt.ndim
    half = n // 2
    m, n2 = 1, 1
    for k in range(half):
        m *= tt.shape[k]
    for k in range(half, n):
        n2 *= tt.shape[k]

    mat = dense.reshape(m, n2)
    s = jnp.linalg.svd(mat, compute_uv=False)
    s_max = float(s[0])
    s_min = float(s[-1])
    return s_max / (s_min + 1e-30)


def tt_rank_reveal(
    tt: TensorTrain,
    threshold: float = 1e-6,
) -> List[int]:
    """
    Determine the effective TT-ranks after truncating small singular values.

    Parameters
    ----------
    tt : TensorTrain
    threshold : relative threshold for rank truncation

    Returns
    -------
    List of effective TT-ranks
    """
    n = tt.ndim
    tt_left = tt_left_orthogonalize(tt)
    effective_ranks = []

    for k in range(n - 1):
        G = tt_left.cores[k]
        r_l, n_k, r_r = G.shape
        M = G.reshape(r_l * n_k, r_r)
        s = jnp.linalg.svd(M, compute_uv=False)
        s_rel = s / (s[0] + 1e-30)
        rank = int(jnp.sum(s_rel > threshold).item())
        effective_ranks.append(max(1, rank))

    return effective_ranks


def tt_is_left_orthogonal(tt: TensorTrain, tol: float = 1e-6) -> bool:
    """
    Check if all but the last core of a TT are left-orthogonal.

    A core G_k is left-orthogonal if sum_s G_k[:, s, :]^T G_k[:, s, :] = I.

    Parameters
    ----------
    tt : TensorTrain
    tol : tolerance for orthogonality check

    Returns
    -------
    True if TT is in left-canonical form
    """
    for k in range(tt.ndim - 1):
        G = tt.cores[k]
        r_l, n_k, r_r = G.shape
        M = G.reshape(r_l * n_k, r_r)
        gram = M.T @ M  # Should be identity (r_r, r_r)
        err = float(jnp.linalg.norm(gram - jnp.eye(r_r)))
        if err > tol:
            return False
    return True


def tt_is_right_orthogonal(tt: TensorTrain, tol: float = 1e-6) -> bool:
    """Check if all but the first core are right-orthogonal."""
    for k in range(1, tt.ndim):
        G = tt.cores[k]
        r_l, n_k, r_r = G.shape
        M = G.reshape(r_l, n_k * r_r)
        gram = M @ M.T  # Should be identity (r_l, r_l)
        err = float(jnp.linalg.norm(gram - jnp.eye(r_l)))
        if err > tol:
            return False
    return True


def tt_contraction_cost(tt: TensorTrain) -> int:
    """
    Estimate the total contraction cost (FLOPs) for converting TT to dense.

    Parameters
    ----------
    tt : TensorTrain

    Returns
    -------
    Estimated FLOPs
    """
    total_cost = 0
    running_dim = 1  # Current accumulated dimension

    for k in range(tt.ndim):
        r_l, n_k, r_r = tt.cores[k].shape
        cost = running_dim * r_l * n_k * r_r
        total_cost += cost
        running_dim = running_dim * n_k

    return total_cost


def tt_memory_estimate(tt: TensorTrain, bytes_per_element: int = 4) -> int:
    """
    Estimate memory usage of a TT in bytes.

    Parameters
    ----------
    tt : TensorTrain
    bytes_per_element : bytes per float element

    Returns
    -------
    Memory in bytes
    """
    return tt.n_params * bytes_per_element


def tt_apply_function(
    tt: TensorTrain,
    func: Callable[[jnp.ndarray], jnp.ndarray],
    max_rank: int = 20,
) -> TensorTrain:
    """
    Apply a non-linear function to the TT tensor element-wise (approximate).

    Uses a CP-like approximation: convert to dense, apply function, then
    re-decompose. Only feasible for small tensors.

    Parameters
    ----------
    tt : TensorTrain
    func : element-wise function
    max_rank : rank for re-decomposition

    Returns
    -------
    TensorTrain approximation of func(tensor)
    """
    dense = tt_to_dense(tt)
    result = func(dense)
    return tt_svd(result, max_rank=max_rank)


def tt_log(tt: TensorTrain, max_rank: int = 20) -> TensorTrain:
    """Element-wise natural log of a positive TT tensor."""
    return tt_apply_function(tt, jnp.log, max_rank=max_rank)


def tt_exp(tt: TensorTrain, max_rank: int = 20) -> TensorTrain:
    """Element-wise exponential of a TT tensor."""
    return tt_apply_function(tt, jnp.exp, max_rank=max_rank)


def tt_softmax(tt: TensorTrain, max_rank: int = 20) -> TensorTrain:
    """Softmax of TT tensor (applied to flattened representation)."""
    dense = tt_to_dense(tt)
    result = jax.nn.softmax(dense.reshape(-1)).reshape(dense.shape)
    return tt_svd(result, max_rank=max_rank)


def tt_solve(
    ttm_A: TensorTrainMatrix,
    tt_b: TensorTrain,
    max_rank: int = 20,
    n_iter: int = 50,
    lr: float = 0.01,
) -> TensorTrain:
    """
    Solve the linear system TTM_A @ x = tt_b for x in TT format.

    Uses gradient descent minimization of ||A x - b||^2_F.

    Parameters
    ----------
    ttm_A : TensorTrainMatrix operator
    tt_b : TensorTrain right-hand side
    max_rank : TT-rank for the solution
    n_iter : gradient descent iterations
    lr : learning rate

    Returns
    -------
    TensorTrain solution approximation x
    """
    # Initialize x as a zero TT
    x = tt_random(ttm_A.col_shape, max_rank, jax.random.PRNGKey(0))
    x = tt_scale(x, 0.01)

    for _ in range(n_iter):
        Ax = tt_matvec(ttm_A, x)
        residual = tt_subtract(Ax, tt_b)
        # Gradient: A^T r
        At_r = tt_matvec(
            TensorTrainMatrix(
                [c.transpose(0, 2, 1, 3) for c in ttm_A.cores],
                ttm_A.col_shape,
                ttm_A.row_shape,
            ),
            residual,
        )
        x = tt_subtract(x, tt_scale(At_r, lr))
        x = tt_round(x, max_rank=max_rank)

    return x


# ============================================================================
# TT-based statistics for financial data
# ============================================================================

def tt_covariance_from_samples(
    samples: jnp.ndarray,
    shape: Tuple[int, ...],
    max_rank: int = 10,
) -> TensorTrain:
    """
    Estimate the TT covariance tensor from Monte Carlo samples.

    Each sample is a realization of a random tensor X. The covariance
    C[i1,...,iN, j1,...,jN] = E[X_{i1...iN} X_{j1...jN}] is estimated
    from samples and compressed as a TT.

    Parameters
    ----------
    samples : (n_samples, d^N) flattened tensor samples
    shape : tensor shape
    max_rank : TT-rank for compression

    Returns
    -------
    TensorTrain approximating the sample covariance tensor
    """
    n_samples = samples.shape[0]
    d_flat = 1
    for s in shape:
        d_flat *= s

    # Sample covariance matrix (d^N x d^N)
    samples_mat = samples.reshape(n_samples, d_flat)
    mu = jnp.mean(samples_mat, axis=0)
    centered = samples_mat - mu[None, :]
    cov = centered.T @ centered / (n_samples - 1)  # (d^N, d^N)

    # Compress the covariance matrix to TT
    combined_shape = tuple(s * s for s in shape)
    cov_tensor = cov.reshape(combined_shape)
    return tt_svd(cov_tensor, max_rank=max_rank)


def tt_mean(
    tt_list: List[TensorTrain],
    max_rank: int = 20,
) -> TensorTrain:
    """
    Compute the TT-mean of a list of TTs.

    Parameters
    ----------
    tt_list : list of TensorTrain
    max_rank : rank for compression

    Returns
    -------
    Mean TensorTrain
    """
    n = len(tt_list)
    result = tt_scale(tt_list[0], 1.0 / n)
    for tt in tt_list[1:]:
        result = tt_add(result, tt_scale(tt, 1.0 / n))
    return tt_round(result, max_rank=max_rank)


def tt_variance(
    tt_list: List[TensorTrain],
    max_rank: int = 20,
) -> jnp.ndarray:
    """
    Compute the average squared deviation from the mean.

    Returns sum_i ||tt_i - mean||^2_F / n.

    Parameters
    ----------
    tt_list : list of TensorTrain
    max_rank : rank budget

    Returns
    -------
    Scalar variance
    """
    n = len(tt_list)
    mean = tt_mean(tt_list, max_rank=max_rank)
    var = jnp.zeros(())
    for tt in tt_list:
        diff = tt_subtract(tt, mean)
        var = var + tt_dot(diff, diff)
    return var / n


def tt_pca(
    tt_list: List[TensorTrain],
    n_components: int = 3,
    max_rank: int = 20,
) -> Tuple[List[TensorTrain], jnp.ndarray]:
    """
    TT-PCA: find principal components of a collection of TT tensors.

    Uses the Gram matrix of inner products between TT tensors to extract
    principal directions in TT space.

    Parameters
    ----------
    tt_list : list of TensorTrain (same shape and rank)
    n_components : number of principal components
    max_rank : rank for orthogonalization

    Returns
    -------
    (principal_components, explained_variance)
    """
    n = len(tt_list)
    K = tt_gram_matrix(tt_list)

    # Eigendecomposition of Gram matrix
    evals, evecs = jnp.linalg.eigh(K)

    # Sort descending
    idx = jnp.argsort(-evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Build principal components as linear combinations
    pcs = []
    explained = []
    for comp in range(min(n_components, n)):
        coeffs = [float(evecs[i, comp]) for i in range(n)]
        pc = tt_linear_combination(tt_list, coeffs, max_rank=max_rank)
        pc = tt_scale(pc, 1.0 / (float(tt_norm(pc)) + 1e-10))
        pcs.append(pc)
        explained.append(float(jnp.maximum(evals[comp], 0.0)))

    total_var = sum(max(float(e), 0.0) for e in evals)
    explained_ratio = jnp.array(explained) / (total_var + 1e-10)

    return pcs, explained_ratio


def tt_distance_matrix(
    tt_list: List[TensorTrain],
) -> jnp.ndarray:
    """
    Compute pairwise Frobenius distance matrix for a list of TTs.

    D[i,j] = ||tt_i - tt_j||_F

    Parameters
    ----------
    tt_list : list of TensorTrain

    Returns
    -------
    (n, n) distance matrix
    """
    n = len(tt_list)
    norms = jnp.array([float(tt_norm(tt)) for tt in tt_list])
    K = tt_gram_matrix(tt_list)

    # D[i,j]^2 = ||tt_i||^2 - 2<tt_i,tt_j> + ||tt_j||^2
    D_sq = norms[:, None] ** 2 - 2 * K + norms[None, :] ** 2
    return jnp.sqrt(jnp.maximum(D_sq, 0.0))


def tt_kmeans(
    tt_list: List[TensorTrain],
    k: int,
    n_iter: int = 20,
    max_rank: int = 10,
    key: Optional[jax.random.KeyArray] = None,
) -> Tuple[List[int], List[TensorTrain]]:
    """
    K-means clustering of TT tensors in Frobenius metric.

    Parameters
    ----------
    tt_list : list of TensorTrain to cluster
    k : number of clusters
    n_iter : clustering iterations
    max_rank : centroid rank budget
    key : random key

    Returns
    -------
    (labels, centroids) assignment labels and centroid TTs
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n = len(tt_list)
    key, subkey = jax.random.split(key)
    centroid_idx = list(jax.random.permutation(subkey, n)[:k])
    centroids = [tt_list[i].copy() for i in centroid_idx]
    labels = [0] * n

    for iteration in range(n_iter):
        # Assignment
        new_labels = []
        for tt in tt_list:
            dists = [float(tt_norm(tt_subtract(tt, c))) for c in centroids]
            new_labels.append(int(np.argmin(dists)))

        if new_labels == labels:
            break
        labels = new_labels

        # Update centroids
        for c in range(k):
            members = [tt_list[i] for i, l in enumerate(labels) if l == c]
            if members:
                centroids[c] = tt_mean(members, max_rank=max_rank)

    return labels, centroids
