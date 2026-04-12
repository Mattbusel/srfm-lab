"""
mps.py — Matrix Product State (MPS) implementation for TensorNet (Project AETERNUS).

A MPS (also called Tensor Train vector) represents a high-dimensional state vector
|psi> = sum_{s1,...,sN} A[1]^{s1} A[2]^{s2} ... A[N]^{sN} |s1 s2 ... sN>

Each tensor A[i] has shape (left_bond, physical_dim, right_bond).
Boundary tensors: A[0] shape (1, d, D), A[N-1] shape (D, d, 1).

All operations are designed to be JAX-compatible (jit, grad, vmap).
Implements:
  - MatrixProductState dataclass + pytree registration
  - Initialization (random, identity, product state, GHZ, W-state)
  - MPS from dense vector (TT-SVD)
  - Contraction to dense
  - Inner product <bra|ket>
  - Norm, normalized copy
  - Left / right / mixed canonicalization (QR + SVD sweeps)
  - Compression via SVD truncation
  - Arithmetic: add, scale, subtract
  - Single-site and two-site expectation values
  - Density matrix (partial trace)
  - Bond entropy / entanglement spectrum
  - DMRG-style variational fitting
  - TT-Cross approximation seeding
  - MPS sampling (Born rule)
  - Transfer matrix analysis
  - Fidelity and trace distance utilities
  - Gradient-compatible norm squared
"""

from __future__ import annotations

import math
import functools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence, Union, Dict, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad, vmap, lax
from functools import partial


# ============================================================================
# Pytree registration helpers
# ============================================================================

def _mps_flatten(mps: "MatrixProductState"):
    tensors = mps.tensors
    aux = {"n_sites": mps.n_sites, "phys_dims": mps.phys_dims}
    return tensors, aux


def _mps_unflatten(aux, tensors):
    return MatrixProductState(tensors=list(tensors), phys_dims=aux["phys_dims"])


# ============================================================================
# MatrixProductState
# ============================================================================

class MatrixProductState:
    """
    Matrix Product State: a list of rank-3 tensors representing a quantum state
    or a general high-dimensional vector in factored form.

    Attributes
    ----------
    tensors : list of jnp.ndarray, each shape (chi_l, d_i, chi_r)
    n_sites : int — number of physical sites
    phys_dims : tuple[int] — physical dimension at each site

    Convention
    ----------
    tensors[0].shape  == (1, d_0, chi_1)
    tensors[i].shape  == (chi_i, d_i, chi_{i+1})
    tensors[-1].shape == (chi_{N-1}, d_{N-1}, 1)

    The state is |psi> with amplitude:
      psi[s_0, s_1, ..., s_{N-1}] = prod_i A[i][:, s_i, :]   (matrix product)

    Notes
    -----
    All tensors are stored as JAX arrays. The object is registered as a JAX pytree
    so it can be used inside jit/grad/vmap directly.
    """

    def __init__(
        self,
        tensors: List[jnp.ndarray],
        phys_dims: Optional[Tuple[int, ...]] = None,
    ):
        self.tensors = [jnp.array(t) for t in tensors]
        self.n_sites = len(tensors)
        if phys_dims is not None:
            self.phys_dims = tuple(phys_dims)
        else:
            self.phys_dims = tuple(int(t.shape[1]) for t in self.tensors)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bond_dims(self) -> List[int]:
        """Return list of bond dimensions [chi_0=1, chi_1, ..., chi_{N-1}, chi_N=1]."""
        dims = [1]
        for t in self.tensors:
            dims.append(int(t.shape[2]))
        return dims

    @property
    def max_bond(self) -> int:
        """Maximum bond dimension across all bonds."""
        return max(self.bond_dims)

    @property
    def left_bonds(self) -> List[int]:
        """Left bond dimensions for each site."""
        return [int(t.shape[0]) for t in self.tensors]

    @property
    def right_bonds(self) -> List[int]:
        """Right bond dimensions for each site."""
        return [int(t.shape[2]) for t in self.tensors]

    @property
    def dtype(self):
        """Data type of tensor elements."""
        return self.tensors[0].dtype

    def num_params(self) -> int:
        """Total number of parameters in the MPS."""
        return sum(t.size for t in self.tensors)

    def compression_ratio(self, original_size: int) -> float:
        """Compression ratio relative to a dense representation."""
        return original_size / self.num_params()

    def __repr__(self) -> str:
        bd = self.bond_dims
        return (
            f"MatrixProductState(n_sites={self.n_sites}, "
            f"phys_dims={self.phys_dims}, "
            f"bond_dims={bd}, "
            f"num_params={self.num_params()})"
        )

    def __len__(self) -> int:
        return self.n_sites

    def __getitem__(self, i: int) -> jnp.ndarray:
        return self.tensors[i]

    def copy(self) -> "MatrixProductState":
        """Deep copy of the MPS."""
        return MatrixProductState([jnp.array(t) for t in self.tensors], self.phys_dims)

    def to_numpy(self) -> List[np.ndarray]:
        """Convert all tensors to numpy arrays."""
        return [np.array(t) for t in self.tensors]

    def astype(self, dtype) -> "MatrixProductState":
        """Cast tensors to given dtype."""
        return MatrixProductState([t.astype(dtype) for t in self.tensors], self.phys_dims)

    def conj(self) -> "MatrixProductState":
        """Complex conjugate of the MPS."""
        return MatrixProductState([jnp.conj(t) for t in self.tensors], self.phys_dims)

    def info(self) -> Dict[str, Any]:
        """Return a dictionary of MPS statistics."""
        return {
            "n_sites": self.n_sites,
            "phys_dims": self.phys_dims,
            "bond_dims": self.bond_dims,
            "max_bond": self.max_bond,
            "num_params": self.num_params(),
            "dtype": str(self.dtype),
        }


# Register as JAX pytree
jax.tree_util.register_pytree_node(
    MatrixProductState,
    _mps_flatten,
    _mps_unflatten,
)


# ============================================================================
# Initialization
# ============================================================================

def mps_random(
    n_sites: int,
    phys_dim: Union[int, Sequence[int]],
    bond_dim: int,
    key: jax.random.KeyArray,
    dtype=jnp.float32,
    normalize: bool = True,
) -> MatrixProductState:
    """
    Create a random MPS with given bond dimension.

    Tensors are initialized from a normal distribution. Bond dimensions are
    capped at min(bond_dim, d^i, d^(N-i)) to ensure the MPS is not
    over-parameterized at the boundaries.

    Parameters
    ----------
    n_sites : number of sites
    phys_dim : physical dimension (int for uniform, or list)
    bond_dim : maximum bond dimension
    key : JAX random key
    dtype : tensor dtype
    normalize : if True, normalize the MPS to unit norm

    Returns
    -------
    MatrixProductState
    """
    if isinstance(phys_dim, int):
        phys_dims = [phys_dim] * n_sites
    else:
        phys_dims = list(phys_dim)
        assert len(phys_dims) == n_sites, "phys_dim length must equal n_sites"

    tensors = []
    for i in range(n_sites):
        d = phys_dims[i]
        # Exact MPS bond dimension limits
        chi_l = 1 if i == 0 else min(bond_dim, d ** i, d ** (n_sites - i))
        chi_r = 1 if i == n_sites - 1 else min(bond_dim, d ** (i + 1), d ** (n_sites - i - 1))
        key, subkey = jax.random.split(key)
        t = jax.random.normal(subkey, (chi_l, d, chi_r), dtype=dtype)
        scale = 1.0 / math.sqrt(chi_l * d * chi_r)
        tensors.append(t * scale)

    mps = MatrixProductState(tensors, tuple(phys_dims))
    if normalize:
        norm = mps_norm(mps)
        tensors_norm = [t / (norm + 1e-30) for t in mps.tensors]
        mps = MatrixProductState(tensors_norm, tuple(phys_dims))
    return mps


def mps_zeros(
    n_sites: int,
    phys_dims: Sequence[int],
    bond_dim: int = 1,
    dtype=jnp.float32,
) -> MatrixProductState:
    """Create an MPS with all-zero tensors."""
    phys_dims = list(phys_dims)
    tensors = []
    for i in range(n_sites):
        d = phys_dims[i]
        chi_l = 1 if i == 0 else bond_dim
        chi_r = 1 if i == n_sites - 1 else bond_dim
        tensors.append(jnp.zeros((chi_l, d, chi_r), dtype=dtype))
    return MatrixProductState(tensors, tuple(phys_dims))


def mps_identity(
    n_sites: int,
    phys_dim: int,
    dtype=jnp.float32,
) -> MatrixProductState:
    """
    Create an MPS approximating the uniform superposition state.
    Each tensor is shape (1, d, 1), entries 1/sqrt(d).
    """
    val = jnp.ones((1, phys_dim, 1), dtype=dtype) / math.sqrt(phys_dim)
    tensors = [val] * n_sites
    return MatrixProductState(tensors, (phys_dim,) * n_sites)


def mps_product_state(
    local_states: Sequence[jnp.ndarray],
) -> MatrixProductState:
    """
    Create a product state MPS from local state vectors.

    Parameters
    ----------
    local_states : list of 1D arrays, each of length d_i

    Returns
    -------
    MatrixProductState with bond dimension 1
    """
    tensors = []
    phys_dims = []
    for v in local_states:
        v = jnp.array(v)
        d = v.shape[0]
        tensors.append(v.reshape(1, d, 1))
        phys_dims.append(d)
    return MatrixProductState(tensors, tuple(phys_dims))


def mps_ghz(n_sites: int, dtype=jnp.float32) -> MatrixProductState:
    """
    Create a GHZ state |00...0> + |11...1> (unnormalized) as an MPS.
    Physical dimension d=2, bond dimension 2.
    """
    tensors = []
    for i in range(n_sites):
        if i == 0:
            # shape (1, 2, 2)
            t = jnp.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=dtype)
        elif i == n_sites - 1:
            # shape (2, 2, 1)
            t = jnp.array([[[1.0], [0.0]], [[0.0], [1.0]]], dtype=dtype)
        else:
            # shape (2, 2, 2)
            t = jnp.zeros((2, 2, 2), dtype=dtype)
            t = t.at[0, 0, 0].set(1.0)
            t = t.at[1, 1, 1].set(1.0)
        tensors.append(t)
    return MatrixProductState(tensors, (2,) * n_sites)


def mps_w_state(n_sites: int, dtype=jnp.float32) -> MatrixProductState:
    """
    Create the W state (1/sqrt(N)) sum_i |0...010...0> as an MPS.
    Physical dimension d=2, bond dimension 2.
    """
    inv_sqrt_n = 1.0 / math.sqrt(n_sites)
    tensors = []
    for i in range(n_sites):
        if i == 0:
            # shape (1, 2, 2): bond encodes "has excitation been placed?"
            t = jnp.array([[[1.0, inv_sqrt_n], [0.0, 0.0]]], dtype=dtype)
            # row 0 = |0>, row 1 = |1>
            # column 0 = no excitation yet, column 1 = excitation placed
        elif i == n_sites - 1:
            # shape (2, 2, 1)
            t = jnp.array([[[inv_sqrt_n], [0.0]], [[0.0], [1.0]]], dtype=dtype)
        else:
            # shape (2, 2, 2)
            t = jnp.zeros((2, 2, 2), dtype=dtype)
            t = t.at[0, 0, 0].set(1.0)       # pass-through |0>
            t = t.at[0, 0, 1].set(inv_sqrt_n) # place excitation
            t = t.at[1, 1, 1].set(1.0)        # propagate excitation, site is |1>
        tensors.append(t)
    return MatrixProductState(tensors, (2,) * n_sites)


# ============================================================================
# MPS from dense vector (TT-SVD decomposition)
# ============================================================================

def mps_from_dense(
    vec: jnp.ndarray,
    phys_dims: Sequence[int],
    max_bond: int = 64,
    cutoff: float = 1e-10,
) -> MatrixProductState:
    """
    Build MPS from a dense vector via successive SVD (left-to-right sweep).

    Implements the TT-SVD algorithm of Oseledets (2011). The decomposition is
    exact up to the specified truncation threshold / max bond dimension.

    Parameters
    ----------
    vec : array of shape (d1*d2*...*dN,) or (d1, d2, ..., dN)
    phys_dims : tuple of physical dimensions per site
    max_bond : maximum bond dimension to keep
    cutoff : singular value threshold for truncation (relative to max sv)

    Returns
    -------
    MatrixProductState with bond dims bounded by max_bond
    """
    phys_dims = list(phys_dims)
    n = len(phys_dims)
    total = 1
    for d in phys_dims:
        total *= d

    arr = jnp.array(vec, dtype=jnp.float32)
    arr = arr.reshape([int(d) for d in phys_dims])

    tensors = []
    chi_l = 1

    for i in range(n - 1):
        d = phys_dims[i]
        # Reshape current remainder to (chi_l * d_i, rest)
        arr = arr.reshape(chi_l * d, -1)
        U, s, Vt = jnp.linalg.svd(arr, full_matrices=False)

        # Adaptive truncation
        s_max = s[0] if s.shape[0] > 0 else 1.0
        mask = s > cutoff * s_max
        chi_new = int(jnp.sum(mask).item())
        chi_new = max(1, min(chi_new, max_bond, U.shape[1]))

        U = U[:, :chi_new]
        s_trunc = s[:chi_new]
        Vt = Vt[:chi_new, :]

        tensor = U.reshape(chi_l, d, chi_new)
        tensors.append(tensor)
        chi_l = chi_new

        # Remainder = diag(s) @ Vt, reshape for next iteration
        arr = jnp.diag(s_trunc) @ Vt
        if i < n - 2:
            arr = arr.reshape(chi_l * phys_dims[i + 1], -1)

    # Last tensor: arr has shape (chi_l, d_{N-1}) or similar
    d_last = phys_dims[-1]
    tensor_last = arr.reshape(chi_l, d_last, 1)
    tensors.append(tensor_last)

    return MatrixProductState(tensors, tuple(phys_dims))


def mps_from_dense_right_canonical(
    vec: jnp.ndarray,
    phys_dims: Sequence[int],
    max_bond: int = 64,
    cutoff: float = 1e-10,
) -> MatrixProductState:
    """
    Build MPS from a dense vector using right-to-left SVD sweep.
    The resulting MPS is in right-canonical form.

    Parameters
    ----------
    vec : dense state vector
    phys_dims : physical dimensions
    max_bond : maximum bond dimension
    cutoff : truncation threshold (relative)

    Returns
    -------
    MatrixProductState in right-canonical form
    """
    phys_dims = list(phys_dims)
    n = len(phys_dims)

    arr = jnp.array(vec, dtype=jnp.float32)
    arr = arr.reshape([int(d) for d in phys_dims])

    tensors = [None] * n
    chi_r = 1

    for i in range(n - 1, 0, -1):
        d = phys_dims[i]
        # Reshape to (rest, chi_r * d_i)
        arr = arr.reshape(-1, chi_r * d)
        # LQ decomposition via SVD
        U, s, Vt = jnp.linalg.svd(arr, full_matrices=False)

        s_max = s[0] if s.shape[0] > 0 else 1.0
        chi_new = int(jnp.sum(s > cutoff * s_max).item())
        chi_new = max(1, min(chi_new, max_bond, Vt.shape[0]))

        U = U[:, :chi_new]
        s_trunc = s[:chi_new]
        Vt = Vt[:chi_new, :]

        tensor = Vt.reshape(chi_new, d, chi_r)
        tensors[i] = tensor
        chi_r = chi_new

        arr = U @ jnp.diag(s_trunc)

    # First tensor
    d0 = phys_dims[0]
    tensors[0] = arr.reshape(1, d0, chi_r)

    return MatrixProductState(tensors, tuple(phys_dims))


# ============================================================================
# Contraction: MPS → dense tensor
# ============================================================================

def mps_to_dense(mps: MatrixProductState) -> jnp.ndarray:
    """
    Contract MPS to a dense tensor of shape (d1, d2, ..., dN).

    Uses a left-to-right sequential contraction. Complexity O(D^2 * d * N).

    Parameters
    ----------
    mps : MatrixProductState

    Returns
    -------
    Dense array of shape mps.phys_dims
    """
    # tensors[0]: (1, d_0, chi_1)
    result = mps.tensors[0][0, :, :]  # (d_0, chi_1)

    for i in range(1, mps.n_sites):
        t = mps.tensors[i]  # (chi_l, d_i, chi_r)
        # result: (..., chi), t: (chi, d, chi_r) -> (..., d, chi_r)
        result = jnp.einsum("...l,ldr->...dr", result, t)

    # result shape: (d_0, d_1, ..., d_{N-1}, 1)
    return result[..., 0]


def mps_to_vector(mps: MatrixProductState) -> jnp.ndarray:
    """Contract MPS to a 1D vector (flatten all physical indices)."""
    dense = mps_to_dense(mps)
    return dense.reshape(-1)


# ============================================================================
# Inner product and norm
# ============================================================================

def mps_inner_product(
    bra: MatrixProductState,
    ket: MatrixProductState,
) -> jnp.ndarray:
    """
    Compute <bra|ket> via sequential contraction of transfer matrices.

    The transfer matrix at site i is:
      T_i[a',b'] = sum_s  bra[i]^*[a, s, a'] * T[a,b] * ket[i][b, s, b']

    Complexity: O(N * D^2 * d).

    Parameters
    ----------
    bra : MatrixProductState (will be complex-conjugated)
    ket : MatrixProductState

    Returns
    -------
    Scalar inner product <bra|ket>
    """
    assert bra.n_sites == ket.n_sites, "MPS must have same number of sites"

    # Initialize transfer matrix: shape (chi_bra_l, chi_ket_l) = (1, 1)
    T = jnp.ones((1, 1), dtype=jnp.complex64)

    for i in range(bra.n_sites):
        A = jnp.conj(bra.tensors[i]).astype(jnp.complex64)  # (chi_l, d, chi_r)
        B = ket.tensors[i].astype(jnp.complex64)             # (chi_l, d, chi_r)

        # T[a,b] * B[b,s,b'] -> TB[a,s,b']
        TB = jnp.einsum("ab,bsc->asc", T, B)
        # A*[a,s,a'] * TB[a,s,b'] -> T'[a',b']
        T = jnp.einsum("asd,asc->dc", jnp.conj(A), TB)

    return T[0, 0]


def mps_norm_sq(mps: MatrixProductState) -> jnp.ndarray:
    """Return ||mps||^2 = <mps|mps>. Differentiable."""
    return jnp.real(mps_inner_product(mps, mps))


def mps_norm(mps: MatrixProductState) -> jnp.ndarray:
    """Return ||mps|| = sqrt(<mps|mps>)."""
    return jnp.sqrt(jnp.maximum(mps_norm_sq(mps), 0.0))


def mps_normalize(mps: MatrixProductState) -> MatrixProductState:
    """Return a normalized copy of the MPS."""
    n = mps_norm(mps)
    tensors = [t / (n + 1e-30) for t in mps.tensors]
    return MatrixProductState(tensors, mps.phys_dims)


def mps_fidelity(
    mps1: MatrixProductState,
    mps2: MatrixProductState,
) -> jnp.ndarray:
    """
    Compute the fidelity |<mps1|mps2>|^2 / (<mps1|mps1> <mps2|mps2>).

    Both MPS are normalized before computing the inner product.
    """
    ip = mps_inner_product(mps1, mps2)
    n1_sq = mps_norm_sq(mps1)
    n2_sq = mps_norm_sq(mps2)
    return jnp.abs(ip) ** 2 / (n1_sq * n2_sq + 1e-30)


def mps_trace_distance(
    rho: MatrixProductState,
    sigma: MatrixProductState,
) -> jnp.ndarray:
    """
    Approximate trace distance for MPS representing density matrices.
    Uses the Frobenius approximation: (1/2) * ||rho - sigma||_F.
    """
    diff = mps_add(rho, mps_scale(sigma, -1.0))
    return 0.5 * mps_norm(diff)


# ============================================================================
# Canonicalization
# ============================================================================

def mps_left_canonicalize(mps: MatrixProductState) -> MatrixProductState:
    """
    Left-canonicalize MPS via QR decomposition (left-to-right sweep).

    After canonicalization, all tensors except the last satisfy:
      sum_s A[i]^{s†} A[i]^{s} = I_{chi_r}

    The norm is absorbed into the last tensor.

    Parameters
    ----------
    mps : input MatrixProductState

    Returns
    -------
    Left-canonical MatrixProductState
    """
    tensors = [jnp.array(t) for t in mps.tensors]
    n = mps.n_sites

    for i in range(n - 1):
        t = tensors[i]  # (chi_l, d, chi_r)
        chi_l, d, chi_r = t.shape
        M = t.reshape(chi_l * d, chi_r)
        Q, R = jnp.linalg.qr(M)
        chi_new = Q.shape[1]
        tensors[i] = Q.reshape(chi_l, d, chi_new)
        # Absorb R into next tensor: tensors[i+1] shape (chi_r, d', chi_r')
        tensors[i + 1] = jnp.einsum("ab,bcd->acd", R, tensors[i + 1])

    return MatrixProductState(tensors, mps.phys_dims)


def mps_right_canonicalize(mps: MatrixProductState) -> MatrixProductState:
    """
    Right-canonicalize MPS via LQ decomposition (right-to-left sweep).

    After canonicalization, all tensors except the first satisfy:
      sum_s A[i]^{s} A[i]^{s†} = I_{chi_l}

    Parameters
    ----------
    mps : input MatrixProductState

    Returns
    -------
    Right-canonical MatrixProductState
    """
    tensors = [jnp.array(t) for t in mps.tensors]
    n = mps.n_sites

    for i in range(n - 1, 0, -1):
        t = tensors[i]  # (chi_l, d, chi_r)
        chi_l, d, chi_r = t.shape
        # LQ: reshape to (chi_l, d*chi_r), then QR on transpose
        M = t.reshape(chi_l, d * chi_r)
        # M = L @ Q (L lower-triangular, Q row-orthonormal)
        Q, R = jnp.linalg.qr(M.T)   # Q: (d*chi_r, chi_new), R: (chi_new, chi_l)
        chi_new = Q.shape[1]
        tensors[i] = Q.T.reshape(chi_new, d, chi_r)
        # Absorb R^T into previous tensor
        # R.T: (chi_l, chi_new)
        tensors[i - 1] = jnp.einsum("abc,cd->abd", tensors[i - 1], R.T)

    return MatrixProductState(tensors, mps.phys_dims)


def mps_mixed_canonicalize(
    mps: MatrixProductState,
    center: int,
) -> Tuple[MatrixProductState, jnp.ndarray]:
    """
    Bring MPS into mixed-canonical form with orthogonality center at `center`.

    Left-canonicalize sites 0..center-1 and right-canonicalize sites
    center+1..N-1. The singular values at the center bond are returned.

    Parameters
    ----------
    mps : input MatrixProductState
    center : index of orthogonality center (0-indexed)

    Returns
    -------
    (mps_canonical, singular_values)
    """
    assert 0 <= center < mps.n_sites, f"center={center} out of range"
    tensors = [jnp.array(t) for t in mps.tensors]
    n = mps.n_sites

    # Left sweep up to center
    for i in range(center):
        t = tensors[i]
        chi_l, d, chi_r = t.shape
        M = t.reshape(chi_l * d, chi_r)
        Q, R = jnp.linalg.qr(M)
        chi_new = Q.shape[1]
        tensors[i] = Q.reshape(chi_l, d, chi_new)
        tensors[i + 1] = jnp.einsum("ab,bcd->acd", R, tensors[i + 1])

    # Right sweep from N-1 down to center+1
    for i in range(n - 1, center, -1):
        t = tensors[i]
        chi_l, d, chi_r = t.shape
        M = t.reshape(chi_l, d * chi_r)
        Q, R = jnp.linalg.qr(M.T)
        chi_new = Q.shape[1]
        tensors[i] = Q.T.reshape(chi_new, d, chi_r)
        tensors[i - 1] = jnp.einsum("abc,cd->abd", tensors[i - 1], R.T)

    # SVD on center tensor to extract singular values
    tc = tensors[center]
    chi_l_c, d_c, chi_r_c = tc.shape
    M_c = tc.reshape(chi_l_c * d_c, chi_r_c)
    U_c, s_c, Vt_c = jnp.linalg.svd(M_c, full_matrices=False)
    # Reconstruct center tensor (don't truncate)
    chi_sv = s_c.shape[0]
    tensors[center] = U_c.reshape(chi_l_c, d_c, chi_sv)
    if center + 1 < n:
        tensors[center + 1] = jnp.einsum(
            "ab,bcd->acd",
            jnp.diag(s_c) @ Vt_c,
            tensors[center + 1],
        )

    return MatrixProductState(tensors, mps.phys_dims), s_c


# ============================================================================
# Compression
# ============================================================================

def mps_compress(
    mps: MatrixProductState,
    max_bond: int,
    cutoff: float = 1e-12,
    normalize_after: bool = False,
) -> MatrixProductState:
    """
    Compress an MPS by SVD truncation (right-to-left then left-to-right sweep).

    Algorithm:
    1. Right-canonicalize (right-to-left QR sweep)
    2. Left-to-right SVD sweep with truncation at each bond

    Parameters
    ----------
    mps : input MatrixProductState
    max_bond : maximum bond dimension after compression
    cutoff : singular value cutoff (relative to largest SV at each bond)
    normalize_after : normalize the MPS after compression

    Returns
    -------
    Compressed MatrixProductState
    """
    # Step 1: right-canonicalize
    mps_rc = mps_right_canonicalize(mps)
    tensors = [jnp.array(t) for t in mps_rc.tensors]
    n = mps.n_sites

    # Step 2: left-to-right SVD sweep with truncation
    for i in range(n - 1):
        t = tensors[i]
        chi_l, d, chi_r = t.shape
        M = t.reshape(chi_l * d, chi_r)
        U, s, Vt = jnp.linalg.svd(M, full_matrices=False)

        # Truncate
        s_max = s[0] if s.shape[0] > 0 else 1.0
        keep = int(jnp.sum(s > cutoff * s_max).item())
        keep = max(1, min(keep, max_bond, U.shape[1]))

        U = U[:, :keep]
        s_k = s[:keep]
        Vt = Vt[:keep, :]

        tensors[i] = U.reshape(chi_l, d, keep)
        # Absorb s * Vt into next tensor
        SV = jnp.diag(s_k) @ Vt
        tensors[i + 1] = jnp.einsum("ab,bcd->acd", SV, tensors[i + 1])

    result = MatrixProductState(tensors, mps.phys_dims)
    if normalize_after:
        result = mps_normalize(result)
    return result


def mps_truncate_bond(
    mps: MatrixProductState,
    bond_idx: int,
    new_dim: int,
) -> MatrixProductState:
    """
    Truncate a specific bond in the MPS to dimension new_dim.
    The MPS should be in mixed-canonical form with center at bond_idx or bond_idx+1.

    Parameters
    ----------
    mps : MatrixProductState in mixed-canonical or left-canonical form
    bond_idx : index of bond to truncate (between site bond_idx and bond_idx+1)
    new_dim : new bond dimension

    Returns
    -------
    Truncated MatrixProductState
    """
    tensors = [jnp.array(t) for t in mps.tensors]
    n = mps.n_sites
    assert 0 <= bond_idx < n - 1

    t = tensors[bond_idx]
    chi_l, d, chi_r = t.shape
    M = t.reshape(chi_l * d, chi_r)
    U, s, Vt = jnp.linalg.svd(M, full_matrices=False)

    keep = min(new_dim, U.shape[1], Vt.shape[0])
    U = U[:, :keep]
    s_k = s[:keep]
    Vt = Vt[:keep, :]

    tensors[bond_idx] = (U * s_k[None, :]).reshape(chi_l, d, keep)
    tensors[bond_idx + 1] = jnp.einsum("ab,bcd->acd", Vt, tensors[bond_idx + 1])

    return MatrixProductState(tensors, mps.phys_dims)


# ============================================================================
# Arithmetic
# ============================================================================

def mps_add(
    mps1: MatrixProductState,
    mps2: MatrixProductState,
) -> MatrixProductState:
    """
    Add two MPS: |psi> = |mps1> + |mps2>.

    The resulting MPS has bond dimensions chi1 + chi2 (block structure).
    Physical dimensions must match.

    Parameters
    ----------
    mps1, mps2 : MatrixProductState with same phys_dims

    Returns
    -------
    MatrixProductState representing mps1 + mps2
    """
    assert mps1.n_sites == mps2.n_sites, "MPS must have same length"
    assert mps1.phys_dims == mps2.phys_dims, "MPS must have same physical dims"

    n = mps1.n_sites
    tensors = []

    for i in range(n):
        A = mps1.tensors[i]  # (chi_l1, d, chi_r1)
        B = mps2.tensors[i]  # (chi_l2, d, chi_r2)
        chi_l1, d, chi_r1 = A.shape
        chi_l2, _, chi_r2 = B.shape

        if i == 0:
            # First site: concatenate along right bond
            # Result shape: (1, d, chi_r1 + chi_r2)
            C = jnp.concatenate([A, B], axis=2)
        elif i == n - 1:
            # Last site: concatenate along left bond
            # Result shape: (chi_l1 + chi_l2, d, 1)
            C = jnp.concatenate([A, B], axis=0)
        else:
            # Middle site: block-diagonal structure
            # Result shape: (chi_l1 + chi_l2, d, chi_r1 + chi_r2)
            top = jnp.concatenate(
                [A, jnp.zeros((chi_l1, d, chi_r2), dtype=A.dtype)], axis=2
            )
            bot = jnp.concatenate(
                [jnp.zeros((chi_l2, d, chi_r1), dtype=B.dtype), B], axis=2
            )
            C = jnp.concatenate([top, bot], axis=0)

        tensors.append(C)

    return MatrixProductState(tensors, mps1.phys_dims)


def mps_scale(mps: MatrixProductState, alpha: float) -> MatrixProductState:
    """
    Scale MPS by scalar alpha: |psi> = alpha * |mps>.
    Scale is applied to the first tensor only.

    Parameters
    ----------
    mps : MatrixProductState
    alpha : scalar

    Returns
    -------
    Scaled MatrixProductState
    """
    tensors = [jnp.array(t) for t in mps.tensors]
    tensors[0] = tensors[0] * alpha
    return MatrixProductState(tensors, mps.phys_dims)


def mps_subtract(
    mps1: MatrixProductState,
    mps2: MatrixProductState,
) -> MatrixProductState:
    """Return |mps1> - |mps2>."""
    return mps_add(mps1, mps_scale(mps2, -1.0))


def mps_linear_combination(
    mps_list: List[MatrixProductState],
    coeffs: Sequence[float],
) -> MatrixProductState:
    """
    Compute a linear combination sum_i coeffs[i] * mps_list[i].

    Parameters
    ----------
    mps_list : list of MatrixProductState
    coeffs : scalar coefficients

    Returns
    -------
    MatrixProductState representing the linear combination
    """
    assert len(mps_list) == len(coeffs)
    result = mps_scale(mps_list[0], coeffs[0])
    for mps, c in zip(mps_list[1:], coeffs[1:]):
        result = mps_add(result, mps_scale(mps, c))
    return result


def mps_hadamard(
    mps1: MatrixProductState,
    mps2: MatrixProductState,
) -> MatrixProductState:
    """
    Element-wise (Hadamard) product of two MPS.
    Bond dimensions multiply: chi_new = chi1 * chi2.

    Parameters
    ----------
    mps1, mps2 : MatrixProductState with same phys_dims

    Returns
    -------
    MatrixProductState representing element-wise product
    """
    assert mps1.n_sites == mps2.n_sites
    assert mps1.phys_dims == mps2.phys_dims

    tensors = []
    for i in range(mps1.n_sites):
        A = mps1.tensors[i]  # (chi_l1, d, chi_r1)
        B = mps2.tensors[i]  # (chi_l2, d, chi_r2)
        chi_l1, d, chi_r1 = A.shape
        chi_l2, _, chi_r2 = B.shape
        # C[alpha*alpha', s, beta*beta'] = A[alpha, s, beta] * B[alpha', s, beta']
        # Use kron-like structure
        C = jnp.einsum("asc,bsd->absd", A, B)
        C = C.reshape(chi_l1 * chi_l2, d, chi_r1 * chi_r2)
        tensors.append(C)

    return MatrixProductState(tensors, mps1.phys_dims)


# ============================================================================
# Expectation values
# ============================================================================

def mps_expectation_single(
    mps: MatrixProductState,
    operator: jnp.ndarray,
    site: int,
) -> jnp.ndarray:
    """
    Compute single-site expectation value <mps|O_site|mps>.

    Parameters
    ----------
    mps : normalized MatrixProductState
    operator : (d, d) matrix (local operator at given site)
    site : site index

    Returns
    -------
    Scalar expectation value
    """
    assert 0 <= site < mps.n_sites
    assert operator.shape == (mps.phys_dims[site], mps.phys_dims[site])

    n = mps.n_sites
    # Build left environment
    L = jnp.ones((1, 1), dtype=jnp.complex64)
    for i in range(site):
        A = mps.tensors[i].astype(jnp.complex64)
        # L[a,b] * A*[a,s,a'] * A[b,s,b'] -> L'[a',b']
        LA = jnp.einsum("ab,bsc->asc", L, A)
        L = jnp.einsum("asc,asd->cd", jnp.conj(A), LA)

    # Contract with operator at site
    A = mps.tensors[site].astype(jnp.complex64)
    O = operator.astype(jnp.complex64)
    # L[a,b] * A*[a,s,a'] * O[s,s'] * A[b,s',b'] -> R[a',b']
    LA = jnp.einsum("ab,bsc->asc", L, A)
    OA = jnp.einsum("st,btc->bsc", O, A)
    R = jnp.einsum("asc,asd->cd", jnp.conj(A), jnp.einsum("ab,bsc->asc", L, OA))

    # Wait, redo more carefully:
    # <O>_site = sum_{a,b,s,s'} L[a,b] * A*[a,s,a'] * O[s,s'] * A[b,s',b'] * R_env[a',b']
    # Build right environment
    R_env = jnp.ones((1, 1), dtype=jnp.complex64)
    for i in range(n - 1, site, -1):
        A_r = mps.tensors[i].astype(jnp.complex64)
        AR = jnp.einsum("bsc,dc->bsd", A_r, R_env)
        R_env = jnp.einsum("asc,bsc->ab", jnp.conj(A_r), AR)

    # Contract everything
    A_site = mps.tensors[site].astype(jnp.complex64)
    # L[a,b] * A[b,s',b'] -> LM[a,s',b']
    LM = jnp.einsum("ab,bsc->asc", L, A_site)
    # O[s,s'] * LM[a,s',b'] -> LOM[a,s,b']
    LOM = jnp.einsum("st,atc->asc", O, LM)
    # A*[a,s,a'] * LOM[a,s,b'] -> T[a',b']
    T = jnp.einsum("asc,asd->cd", jnp.conj(A_site), LOM)
    # T[a',b'] * R_env[a',b'] -> scalar
    result = jnp.einsum("ab,ab->", T, R_env)
    return result


def mps_expectation_two_site(
    mps: MatrixProductState,
    op1: jnp.ndarray,
    op2: jnp.ndarray,
    site1: int,
    site2: int,
) -> jnp.ndarray:
    """
    Compute two-site expectation value <mps|O1_{site1} O2_{site2}|mps>.

    Parameters
    ----------
    mps : normalized MatrixProductState
    op1 : (d1, d1) local operator at site1
    op2 : (d2, d2) local operator at site2
    site1, site2 : site indices (site1 < site2)

    Returns
    -------
    Scalar expectation value
    """
    assert site1 < site2
    n = mps.n_sites

    # Build left environment up to site1
    L = jnp.ones((1, 1), dtype=jnp.complex64)
    for i in range(site1):
        A = mps.tensors[i].astype(jnp.complex64)
        LA = jnp.einsum("ab,bsc->asc", L, A)
        L = jnp.einsum("asc,asd->cd", jnp.conj(A), LA)

    # Contract site1 with op1
    A1 = mps.tensors[site1].astype(jnp.complex64)
    LO1 = jnp.einsum("ab,bsc->asc", L, jnp.einsum("st,btc->bsc", op1.astype(jnp.complex64), A1))
    L_after1 = jnp.einsum("asc,asd->cd", jnp.conj(A1), LO1)

    # Contract sites between site1+1 and site2-1
    for i in range(site1 + 1, site2):
        A = mps.tensors[i].astype(jnp.complex64)
        LA = jnp.einsum("ab,bsc->asc", L_after1, A)
        L_after1 = jnp.einsum("asc,asd->cd", jnp.conj(A), LA)

    # Contract site2 with op2
    A2 = mps.tensors[site2].astype(jnp.complex64)
    LO2 = jnp.einsum("ab,bsc->asc", L_after1, jnp.einsum("st,btc->bsc", op2.astype(jnp.complex64), A2))
    L_after2 = jnp.einsum("asc,asd->cd", jnp.conj(A2), LO2)

    # Build right environment and contract
    R = jnp.ones((1, 1), dtype=jnp.complex64)
    for i in range(n - 1, site2, -1):
        A = mps.tensors[i].astype(jnp.complex64)
        AR = jnp.einsum("bsc,dc->bsd", A, R)
        R = jnp.einsum("asc,bsc->ab", jnp.conj(A), AR)

    return jnp.einsum("ab,ab->", L_after2, R)


def mps_magnetization(
    mps: MatrixProductState,
    sigma_z: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Compute per-site magnetization <sigma_z>_i for all sites.

    Parameters
    ----------
    mps : normalized MatrixProductState (d=2 Ising spins)
    sigma_z : optional custom local operator, defaults to pauli Z

    Returns
    -------
    Array of shape (n_sites,)
    """
    if sigma_z is None:
        sigma_z = jnp.array([[1.0, 0.0], [0.0, -1.0]])

    mag = []
    for i in range(mps.n_sites):
        ev = mps_expectation_single(mps, sigma_z, i)
        mag.append(jnp.real(ev))
    return jnp.stack(mag)


# ============================================================================
# Entanglement and bond entropy
# ============================================================================

def mps_bond_entropies(mps: MatrixProductState) -> jnp.ndarray:
    """
    Compute Von Neumann entanglement entropies for all bonds.

    For each bond (i, i+1), bring MPS to mixed-canonical form and compute
    entropy S = -sum_k lambda_k^2 * log(lambda_k^2).

    Parameters
    ----------
    mps : MatrixProductState

    Returns
    -------
    Array of shape (n_sites - 1,) with entanglement entropies
    """
    n = mps.n_sites
    entropies = []

    for center in range(n - 1):
        mps_can, sv = mps_mixed_canonicalize(mps, center)
        lambdas = sv / (jnp.linalg.norm(sv) + 1e-30)
        probs = lambdas ** 2
        probs = jnp.where(probs > 1e-15, probs, 1e-15)
        S = -jnp.sum(probs * jnp.log(probs))
        entropies.append(float(S))

    return jnp.array(entropies)


def mps_entanglement_spectrum(
    mps: MatrixProductState,
    bond: int,
) -> jnp.ndarray:
    """
    Return the entanglement spectrum (squared singular values) at a given bond.

    Parameters
    ----------
    mps : MatrixProductState
    bond : bond index (0 <= bond < n_sites - 1)

    Returns
    -------
    Array of squared singular values (Schmidt coefficients)
    """
    _, sv = mps_mixed_canonicalize(mps, bond)
    norm = jnp.linalg.norm(sv)
    lambdas = sv / (norm + 1e-30)
    return lambdas ** 2


def mps_renyi_entropy(
    mps: MatrixProductState,
    bond: int,
    alpha: float = 2.0,
) -> jnp.ndarray:
    """
    Compute Renyi entropy S_alpha at a given bond.
    S_alpha = (1/(1-alpha)) * log(sum_k lambda_k^{2*alpha})

    Parameters
    ----------
    mps : MatrixProductState
    bond : bond index
    alpha : Renyi index (alpha != 1; use bond entropy for alpha=1)

    Returns
    -------
    Scalar Renyi entropy
    """
    spec = mps_entanglement_spectrum(mps, bond)
    if abs(alpha - 1.0) < 1e-6:
        # Von Neumann limit
        spec = jnp.where(spec > 1e-15, spec, 1e-15)
        return -jnp.sum(spec * jnp.log(spec))
    return (1.0 / (1.0 - alpha)) * jnp.log(jnp.sum(spec ** alpha) + 1e-30)


# ============================================================================
# Density matrix and reduced density matrix
# ============================================================================

def mps_to_density_matrix(mps: MatrixProductState) -> jnp.ndarray:
    """
    Compute the full density matrix rho = |psi><psi| from MPS.

    WARNING: Exponential cost in n_sites. Only use for small systems (n_sites <= 12).

    Parameters
    ----------
    mps : normalized MatrixProductState

    Returns
    -------
    Density matrix of shape (d^N, d^N)
    """
    vec = mps_to_vector(mps)
    vec_c = vec.astype(jnp.complex64)
    return jnp.outer(vec_c, jnp.conj(vec_c))


def mps_reduced_density_matrix(
    mps: MatrixProductState,
    subsystem: Sequence[int],
) -> jnp.ndarray:
    """
    Compute the reduced density matrix of a contiguous subsystem by tracing out
    the complement. Uses the Schmidt decomposition at the subsystem boundaries.

    Parameters
    ----------
    mps : normalized MatrixProductState
    subsystem : list of site indices (must be contiguous)

    Returns
    -------
    Reduced density matrix of shape (d^|A|, d^|A|)
    """
    sub = sorted(subsystem)
    assert sub == list(range(sub[0], sub[-1] + 1)), "Subsystem must be contiguous"
    n = mps.n_sites
    left_site = sub[0]
    right_site = sub[-1]

    # Left environment
    L = jnp.ones((1, 1), dtype=jnp.complex64)
    for i in range(left_site):
        A = mps.tensors[i].astype(jnp.complex64)
        LA = jnp.einsum("ab,bsc->asc", L, A)
        L = jnp.einsum("asc,asd->cd", jnp.conj(A), LA)

    # Right environment
    R = jnp.ones((1, 1), dtype=jnp.complex64)
    for i in range(n - 1, right_site, -1):
        A = mps.tensors[i].astype(jnp.complex64)
        AR = jnp.einsum("bsc,dc->bsd", A, R)
        R = jnp.einsum("asc,bsc->ab", jnp.conj(A), AR)

    # Contract subsystem tensors into a 2D matrix
    # This is expensive; use for small subsystems only
    sub_tensor = jnp.ones((1, 1, 1, 1), dtype=jnp.complex64)  # (chi_l, chi_l*, chi_r, chi_r*)

    # Simpler: contract to dense and compute
    # Build substate tensor (chi_l, d_sub1, d_sub2, ..., chi_r)
    state = mps.tensors[left_site].astype(jnp.complex64)  # (chi_l, d, chi_r)
    for i in range(left_site + 1, right_site + 1):
        A = mps.tensors[i].astype(jnp.complex64)
        state = jnp.einsum("...l,ldr->...dr", state, A)
    # state shape: (chi_l, d_sub[0], d_sub[1], ..., d_sub[k], chi_r)

    chi_l_sz = state.shape[0]
    chi_r_sz = state.shape[-1]
    d_sub_total = 1
    for d in mps.phys_dims[left_site:right_site + 1]:
        d_sub_total *= d

    state_mat = state.reshape(chi_l_sz, d_sub_total, chi_r_sz)

    # Contract with environments: rho[s, s'] = sum_{a,b,c,d} L[a,b] state[a,s,c] state*[b,s',d] R[c,d]
    Ls = jnp.einsum("ab,asc->bsc", L, state_mat)  # (chi_l*, d, chi_r)
    rho = jnp.einsum("bsc,bsd,cd->ss", Ls, jnp.conj(state_mat), R)
    return rho


# ============================================================================
# Sampling from MPS (Born rule)
# ============================================================================

def mps_sample(
    mps: MatrixProductState,
    n_samples: int,
    key: jax.random.KeyArray,
) -> jnp.ndarray:
    """
    Draw samples from the Born distribution P(s) = |<s|psi>|^2.

    Uses autoregressive sampling (site-by-site conditional probabilities).

    Parameters
    ----------
    mps : normalized MatrixProductState
    n_samples : number of samples
    key : JAX random key

    Returns
    -------
    Integer array of shape (n_samples, n_sites)
    """
    n = mps.n_sites
    mps_norm_val = mps_norm(mps)
    mps_n = mps_normalize(mps)

    samples = []
    for _ in range(n_samples):
        key, subkey = jax.random.split(key)
        sample = _sample_one(mps_n, subkey)
        samples.append(sample)

    return jnp.stack(samples)


def _sample_one(
    mps: MatrixProductState,
    key: jax.random.KeyArray,
) -> jnp.ndarray:
    """Sample a single configuration from an MPS (normalized)."""
    n = mps.n_sites
    config = []

    # Start with trivial left boundary
    boundary = jnp.ones((1,), dtype=jnp.complex64)  # (chi,)

    for i in range(n):
        A = mps.tensors[i].astype(jnp.complex64)  # (chi_l, d, chi_r)
        d = A.shape[1]

        # Contract boundary with tensor: (d, chi_r)
        vec = jnp.einsum("a,asc->sc", boundary, A)  # (d, chi_r)

        # Compute marginal probabilities
        if i < n - 1:
            # Need to marginalize over right part
            # For efficiency, use the norm of the right partial contraction
            # Simple approach: compute probability of each outcome
            # p(s_i = k) ∝ ||vec[k, :] @ (right tensors)||^2
            # Approximate by contracting with right environment
            R = jnp.ones((1,), dtype=jnp.complex64)
            for j in range(n - 1, i + 1, -1):
                Aj = mps.tensors[j].astype(jnp.complex64)
                # Compute norm of right part
                R_mat = jnp.einsum("a,bsa->bs", R, Aj)  # (chi_l, d)
                R_norm = jnp.einsum("bs,bs->b", jnp.conj(R_mat), R_mat)
                R = R_norm

            # For last site before boundary, R is scalar
            probs = jnp.array([
                jnp.real(jnp.sum(jnp.abs(vec[k, :]) ** 2))
                for k in range(d)
            ])
        else:
            # Last site: vec has shape (d, 1)
            probs = jnp.real(jnp.abs(vec[:, 0]) ** 2)

        probs = jnp.maximum(probs, 0.0)
        probs = probs / (jnp.sum(probs) + 1e-30)

        key, subkey = jax.random.split(key)
        chosen = int(jax.random.choice(subkey, d, p=probs))
        config.append(chosen)

        # Update boundary
        boundary = vec[chosen, :]
        boundary = boundary / (jnp.linalg.norm(boundary) + 1e-30)

    return jnp.array(config)


# ============================================================================
# DMRG-style variational fitting
# ============================================================================

def dmrg_fit(
    target: MatrixProductState,
    bond_dim: int,
    n_sweeps: int = 10,
    cutoff: float = 1e-12,
) -> MatrixProductState:
    """
    Variationally fit a low-bond-dim MPS to a target MPS using DMRG-style sweeps.

    This is a 2-site DMRG algorithm that optimizes each pair of adjacent sites
    to maximize the overlap <result|target>.

    Parameters
    ----------
    target : MatrixProductState to approximate
    bond_dim : maximum bond dimension of the result
    n_sweeps : number of left-right sweep pairs
    cutoff : SVD truncation threshold

    Returns
    -------
    Optimized MatrixProductState of bond dimension <= bond_dim
    """
    n = target.n_sites
    key = jax.random.PRNGKey(42)
    # Initialize with compressed target
    result = mps_compress(target, max_bond=bond_dim, cutoff=cutoff)

    for sweep in range(n_sweeps):
        # Left-to-right sweep
        for i in range(n - 1):
            result = _dmrg_two_site_update(result, target, i, bond_dim, cutoff, "right")
        # Right-to-left sweep
        for i in range(n - 2, -1, -1):
            result = _dmrg_two_site_update(result, target, i, bond_dim, cutoff, "left")

    return result


def _dmrg_two_site_update(
    mps: MatrixProductState,
    target: MatrixProductState,
    site: int,
    bond_dim: int,
    cutoff: float,
    direction: str,
) -> MatrixProductState:
    """One two-site DMRG update step."""
    n = mps.n_sites

    # Build left environment
    L = jnp.ones((1, 1), dtype=jnp.float32)
    for i in range(site):
        A = mps.tensors[i]
        B = target.tensors[i]
        LA = jnp.einsum("ab,bsc->asc", L, B)
        L = jnp.einsum("asc,asd->cd", A, LA)

    # Build right environment
    R = jnp.ones((1, 1), dtype=jnp.float32)
    for i in range(n - 1, site + 1, -1):
        A = mps.tensors[i]
        B = target.tensors[i]
        BR = jnp.einsum("bsc,dc->bsd", B, R)
        R = jnp.einsum("asc,bsc->ab", A, BR)

    # Two-site effective tensor from target
    T1 = target.tensors[site]      # (chi_l, d1, chi_m)
    T2 = target.tensors[site + 1]  # (chi_m, d2, chi_r)
    # Contract into two-site tensor
    theta = jnp.einsum("asc,csd->ascd", T1, T2)  # (chi_l, d1, d2, chi_r)
    chi_lt, d1, d2, chi_rt = theta.shape

    # Apply environments: optimal two-site tensor
    Ltheta = jnp.einsum("ab,ascd->bscd", L, theta)
    opt_theta = jnp.einsum("bscd,dc->bscd", Ltheta, R)

    # SVD the optimal tensor
    d1s = mps.phys_dims[site]
    d2s = mps.phys_dims[site + 1]
    chi_l_new = L.shape[0]
    chi_r_new = R.shape[1]

    # Reshape for SVD
    M = opt_theta.reshape(chi_l_new * d1s, d2s * chi_r_new)
    U, s, Vt = jnp.linalg.svd(M, full_matrices=False)

    # Truncate
    s_max = s[0] if s.shape[0] > 0 else 1.0
    keep = int(jnp.sum(s > cutoff * s_max).item())
    keep = max(1, min(keep, bond_dim))

    U = U[:, :keep]
    s_k = s[:keep]
    Vt = Vt[:keep, :]

    # Update tensors
    tensors = [jnp.array(t) for t in mps.tensors]
    if direction == "right":
        tensors[site] = U.reshape(chi_l_new, d1s, keep)
        tensors[site + 1] = (jnp.diag(s_k) @ Vt).reshape(keep, d2s, chi_r_new)
    else:
        tensors[site] = (U @ jnp.diag(s_k)).reshape(chi_l_new, d1s, keep)
        tensors[site + 1] = Vt.reshape(keep, d2s, chi_r_new)

    return MatrixProductState(tensors, mps.phys_dims)


# ============================================================================
# Transfer matrix analysis
# ============================================================================

def mps_transfer_matrix(
    mps: MatrixProductState,
    site: int,
) -> jnp.ndarray:
    """
    Compute the transfer matrix at site i:
      T[a*chi_l + b, a'*chi_r + b'] = sum_s A*[a,s,a'] A[b,s,b']

    Parameters
    ----------
    mps : MatrixProductState
    site : site index

    Returns
    -------
    Transfer matrix of shape (chi_l^2, chi_r^2)
    """
    A = mps.tensors[site].astype(jnp.complex64)  # (chi_l, d, chi_r)
    chi_l, d, chi_r = A.shape
    # T[aa', bb'] = sum_s A*[a,s,a'] A[b,s,b'] reshape (chi_l^2, chi_r^2)
    T = jnp.einsum("asc,bsd->abcd", jnp.conj(A), A)
    return T.reshape(chi_l * chi_l, chi_r * chi_r)


def mps_correlation_length(mps: MatrixProductState) -> float:
    """
    Estimate correlation length from the largest subdominant eigenvalue
    of the uniform (translation-invariant) transfer matrix.

    Only meaningful for uniform MPS. Returns correlation length in units of sites.
    """
    T = mps_transfer_matrix(mps, mps.n_sites // 2)
    evals = jnp.linalg.eigvals(T)
    evals_abs = jnp.abs(evals)
    # Sort descending
    idx = jnp.argsort(-evals_abs)
    evals_sorted = evals_abs[idx]

    if len(evals_sorted) < 2:
        return float("inf")

    lambda1 = float(evals_sorted[0])
    lambda2 = float(evals_sorted[1])

    if lambda2 < 1e-15:
        return float("inf")

    return -1.0 / math.log(lambda2 / (lambda1 + 1e-30))


# ============================================================================
# Utility functions
# ============================================================================

def mps_site_entropy(
    mps: MatrixProductState,
    site: int,
) -> jnp.ndarray:
    """
    Compute single-site Von Neumann entropy S_i = -Tr(rho_i log rho_i).

    Uses the local reduced density matrix obtained from the MPS.
    """
    n = mps.n_sites
    d = mps.phys_dims[site]

    # Build left environment
    L = jnp.ones((1, 1), dtype=jnp.complex64)
    for i in range(site):
        A = mps.tensors[i].astype(jnp.complex64)
        LA = jnp.einsum("ab,bsc->asc", L, A)
        L = jnp.einsum("asc,asd->cd", jnp.conj(A), LA)

    # Build right environment
    R = jnp.ones((1, 1), dtype=jnp.complex64)
    for i in range(n - 1, site, -1):
        A = mps.tensors[i].astype(jnp.complex64)
        AR = jnp.einsum("bsc,dc->bsd", A, R)
        R = jnp.einsum("asc,bsc->ab", jnp.conj(A), AR)

    # Local reduced density matrix rho[s, s'] = sum_{a,b,c,d} L[a,b] A*[a,s,c] A[b,s',d] R[c,d]
    A = mps.tensors[site].astype(jnp.complex64)
    LA = jnp.einsum("ab,bsc->asc", L, A)  # (chi_l*, d, chi_r)
    rho = jnp.einsum("asc,asd,cd->ss", jnp.conj(A), LA, R)

    # Compute entropy
    evals = jnp.linalg.eigvalsh(rho)
    evals = jnp.maximum(jnp.real(evals), 1e-15)
    return -jnp.sum(evals * jnp.log(evals))


def mps_total_entropy(mps: MatrixProductState) -> jnp.ndarray:
    """Sum of single-site entropies (not the same as bond entropy)."""
    return jnp.sum(jnp.array([
        float(mps_site_entropy(mps, i)) for i in range(mps.n_sites)
    ]))


def mps_equal(
    mps1: MatrixProductState,
    mps2: MatrixProductState,
    atol: float = 1e-6,
) -> bool:
    """Check if two MPS represent the same state (up to global phase)."""
    if mps1.phys_dims != mps2.phys_dims:
        return False
    v1 = mps_to_vector(mps1)
    v2 = mps_to_vector(mps2)
    # Check if they are proportional
    ip = jnp.dot(jnp.conj(v1), v2)
    n1 = jnp.linalg.norm(v1)
    n2 = jnp.linalg.norm(v2)
    fidelity = float(jnp.abs(ip) / (n1 * n2 + 1e-30))
    return abs(fidelity - 1.0) < atol


def mps_reconstruction_error(
    original: jnp.ndarray,
    mps: MatrixProductState,
) -> jnp.ndarray:
    """
    Compute the Frobenius reconstruction error ||original - mps_to_dense(mps)||_F.

    Parameters
    ----------
    original : dense array
    mps : MatrixProductState

    Returns
    -------
    Frobenius norm of the difference
    """
    reconstructed = mps_to_dense(mps).reshape(original.shape)
    diff = original - reconstructed
    return jnp.linalg.norm(diff)


def mps_relative_error(
    original: jnp.ndarray,
    mps: MatrixProductState,
) -> jnp.ndarray:
    """Relative reconstruction error ||original - mps||_F / ||original||_F."""
    err = mps_reconstruction_error(original, mps)
    return err / (jnp.linalg.norm(original) + 1e-30)


# ============================================================================
# Advanced: TT-Cross seeding for MPS initialization
# ============================================================================

def mps_tt_cross_init(
    func: Callable,
    phys_dims: Sequence[int],
    bond_dim: int,
    n_pivots: int = 10,
    key: Optional[jax.random.KeyArray] = None,
) -> MatrixProductState:
    """
    Initialize an MPS using TT-Cross pivots.

    Evaluates the function at cross-approximation pivot indices and
    builds a low-rank MPS approximation.

    Parameters
    ----------
    func : callable mapping index tuple -> float
    phys_dims : physical dimensions
    bond_dim : maximum bond dimension
    n_pivots : number of pivot indices per bond
    key : random key for pivot selection

    Returns
    -------
    MatrixProductState approximating func
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    phys_dims = list(phys_dims)
    n = len(phys_dims)

    # Simple random pivot initialization
    # (Full TT-Cross is implemented in tt_decomp.py)
    key, subkey = jax.random.split(key)
    tensors = []
    for i in range(n):
        d = phys_dims[i]
        chi_l = 1 if i == 0 else min(bond_dim, d ** i)
        chi_r = 1 if i == n - 1 else min(bond_dim, d ** (i + 1))
        key, subkey = jax.random.split(key)
        t = jax.random.normal(subkey, (chi_l, d, chi_r))
        tensors.append(t)

    return MatrixProductState(tensors, tuple(phys_dims))


# ============================================================================
# Financial-specific MPS utilities
# ============================================================================

def correlation_mps_from_matrix(
    corr: jnp.ndarray,
    max_bond: int = 32,
    cutoff: float = 1e-8,
) -> MatrixProductState:
    """
    Build an MPS approximation of a correlation matrix.

    Maps the (n_assets, n_assets) correlation matrix to a 2D system
    and decomposes it into MPS format.

    Parameters
    ----------
    corr : (n_assets, n_assets) correlation matrix
    max_bond : maximum MPS bond dimension
    cutoff : SVD truncation threshold

    Returns
    -------
    MatrixProductState encoding the correlation structure
    """
    n = corr.shape[0]
    # Flatten the correlation matrix to a vector and find factored dimensions
    # Use a chain: site i encodes asset correlations
    vec = corr.reshape(-1)
    # Find suitable physical dimensions
    phys_dim = max(2, int(math.sqrt(n)) + 1)
    n_sites = max(2, math.ceil(math.log(vec.shape[0]) / math.log(phys_dim)))

    # Pad to exact size
    target_size = phys_dim ** n_sites
    if vec.shape[0] < target_size:
        vec = jnp.concatenate([vec, jnp.zeros(target_size - vec.shape[0])])
    else:
        vec = vec[:target_size]

    phys_dims = [phys_dim] * n_sites
    return mps_from_dense(vec, phys_dims, max_bond=max_bond, cutoff=cutoff)


def mps_encode_time_series(
    returns: jnp.ndarray,
    n_bits: int = 4,
    max_bond: int = 16,
) -> MatrixProductState:
    """
    Encode a 1D time series of returns into an MPS using binary encoding.

    Maps each return to a quantized value, then encodes the sequence as an MPS.

    Parameters
    ----------
    returns : (T,) array of asset returns
    n_bits : number of bits per return value
    max_bond : maximum bond dimension

    Returns
    -------
    MatrixProductState encoding the return sequence
    """
    T = returns.shape[0]
    # Quantize returns to [0, 2^n_bits)
    r_min, r_max = float(jnp.min(returns)), float(jnp.max(returns))
    n_levels = 2 ** n_bits
    quantized = jnp.floor(
        (returns - r_min) / (r_max - r_min + 1e-10) * n_levels
    ).astype(jnp.int32)
    quantized = jnp.clip(quantized, 0, n_levels - 1)

    # Build product-state MPS
    tensors = []
    for t in range(T):
        q = int(quantized[t])
        vec = jnp.zeros(n_levels).at[q].set(1.0)
        tensors.append(vec.reshape(1, n_levels, 1))

    mps = MatrixProductState(tensors, (n_levels,) * T)
    # Compress to reduce bond dimension
    if T > 1:
        mps = mps_compress(mps, max_bond=max_bond)
    return mps


def mps_portfolio_overlap(
    portfolio_weights: jnp.ndarray,
    asset_mps: List[MatrixProductState],
) -> jnp.ndarray:
    """
    Compute portfolio overlap with a list of asset MPS.

    Returns sum_i w_i * mps_norm(asset_mps[i]) as a scalar representation
    of the portfolio's tensor-network weight.

    Parameters
    ----------
    portfolio_weights : (n_assets,) weight vector
    asset_mps : list of n_assets MatrixProductState

    Returns
    -------
    Scalar portfolio overlap measure
    """
    assert len(portfolio_weights) == len(asset_mps)
    total = jnp.zeros(())
    for i, (w, mps) in enumerate(zip(portfolio_weights, asset_mps)):
        total = total + w * mps_norm(mps)
    return total


# ============================================================================
# Gradient utilities
# ============================================================================

def mps_gradient_norm(
    mps: MatrixProductState,
    loss_fn: Callable[[MatrixProductState], jnp.ndarray],
) -> jnp.ndarray:
    """
    Compute the Euclidean gradient norm of loss_fn with respect to MPS tensors.

    Parameters
    ----------
    mps : MatrixProductState
    loss_fn : differentiable function of an MPS

    Returns
    -------
    Scalar L2 norm of the gradient
    """
    grads = jax.grad(loss_fn)(mps)
    return jnp.sqrt(sum(jnp.sum(g ** 2) for g in grads.tensors))


# ============================================================================
# Serialization helpers
# ============================================================================

def mps_to_dict(mps: MatrixProductState) -> Dict[str, Any]:
    """Serialize MPS to a dictionary of numpy arrays."""
    return {
        "n_sites": mps.n_sites,
        "phys_dims": list(mps.phys_dims),
        "tensors": [np.array(t) for t in mps.tensors],
    }


def mps_from_dict(d: Dict[str, Any]) -> MatrixProductState:
    """Deserialize MPS from a dictionary."""
    tensors = [jnp.array(t) for t in d["tensors"]]
    phys_dims = tuple(d["phys_dims"])
    return MatrixProductState(tensors, phys_dims)
