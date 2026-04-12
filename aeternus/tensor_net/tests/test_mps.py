"""
test_mps.py — Tests for MatrixProductState operations.

Tests:
- MPS construction and basic properties
- MPS contraction (to dense) vs naive construction
- MPS inner product via transfer matrices
- MPS compression: error bounds, bond dimension reduction
- Canonicalization: left/right canonical gauge conditions
- MPS arithmetic: addition, scaling
- Expectation values
- Bond entropy
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor_net.mps import (
    MatrixProductState,
    mps_random,
    mps_identity,
    mps_from_dense,
    mps_to_dense,
    mps_inner_product,
    mps_norm,
    mps_left_canonicalize,
    mps_right_canonicalize,
    mps_compress,
    mps_add,
    mps_scale,
    mps_frobenius_error,
    mps_bond_entropies,
    mps_entanglement_spectrum,
    mps_expectation_single,
    mps_to_density_matrix,
    mps_compression_analysis,
)

# JAX configuration
jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_mps(rng):
    """Small 4-site MPS with phys_dim=2, bond_dim=2."""
    return mps_random(4, 2, 2, rng)


@pytest.fixture
def medium_mps(rng):
    """6-site MPS with phys_dim=3, bond_dim=4."""
    return mps_random(6, 3, 4, rng)


@pytest.fixture
def product_mps():
    """Product state: |0101> encoded as MPS (bond_dim=1)."""
    # Physical states: |0> = [1,0], |1> = [0,1]
    tensors = []
    for i in range(4):
        state = [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]
        t = jnp.array(state).reshape(1, 2, 1)
        tensors.append(t)
    return MatrixProductState(tensors, (2, 2, 2, 2))


# ---------------------------------------------------------------------------
# Test: Construction
# ---------------------------------------------------------------------------

class TestMPSConstruction:
    def test_random_mps_shape(self, rng):
        mps = mps_random(5, 2, 4, rng)
        assert mps.n_sites == 5
        assert mps.phys_dims == (2, 2, 2, 2, 2)
        assert mps.tensors[0].shape[0] == 1   # Left boundary
        assert mps.tensors[-1].shape[2] == 1  # Right boundary

    def test_random_mps_bond_dims_bounded(self, rng):
        max_bond = 4
        mps = mps_random(6, 2, max_bond, rng)
        for t in mps.tensors:
            assert t.shape[0] <= max_bond
            assert t.shape[2] <= max_bond

    def test_identity_mps(self):
        mps = mps_identity(4, 2)
        assert mps.n_sites == 4
        assert all(t.shape == (1, 2, 1) for t in mps.tensors)
        # Should be a product state
        dense = mps_to_dense(mps)
        assert dense.shape == (2, 2, 2, 2)

    def test_phys_dims_heterogeneous(self, rng):
        phys_dims = [2, 3, 4, 2]
        mps = mps_random(4, phys_dims, 3, rng)
        assert mps.phys_dims == tuple(phys_dims)
        for i, t in enumerate(mps.tensors):
            assert t.shape[1] == phys_dims[i]

    def test_mps_repr(self, small_mps):
        r = repr(small_mps)
        assert "n_sites=4" in r
        assert "phys_dims" in r

    def test_num_params(self, small_mps):
        n = small_mps.num_params()
        assert n > 0
        expected = sum(t.size for t in small_mps.tensors)
        assert n == expected

    def test_pytree_flatten_unflatten(self, small_mps):
        leaves, treedef = jax.tree_util.tree_flatten(small_mps)
        mps_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert mps_reconstructed.n_sites == small_mps.n_sites
        assert mps_reconstructed.phys_dims == small_mps.phys_dims


# ---------------------------------------------------------------------------
# Test: MPS from dense
# ---------------------------------------------------------------------------

class TestMPSFromDense:
    def test_from_dense_reconstruction(self):
        """MPS built from dense vector should reconstruct it exactly (large enough bond)."""
        np.random.seed(0)
        vec = np.random.randn(16).astype(np.float32)
        vec /= np.linalg.norm(vec)
        phys_dims = [2, 2, 2, 2]

        mps = mps_from_dense(jnp.array(vec), phys_dims, max_bond=8)
        reconstructed = np.array(mps_to_dense(mps)).reshape(-1)
        error = np.linalg.norm(vec - reconstructed) / np.linalg.norm(vec)
        assert error < 0.01, f"Reconstruction error too large: {error}"

    def test_from_dense_truncation_increases_error(self):
        """Smaller bond dim should give larger error."""
        np.random.seed(1)
        vec = np.random.randn(16).astype(np.float32)
        phys_dims = [2, 2, 2, 2]

        mps_full = mps_from_dense(jnp.array(vec), phys_dims, max_bond=8)
        mps_small = mps_from_dense(jnp.array(vec), phys_dims, max_bond=1)

        err_full = mps_frobenius_error(mps_full, jnp.array(vec).reshape(2, 2, 2, 2))
        err_small = mps_frobenius_error(mps_small, jnp.array(vec).reshape(2, 2, 2, 2))

        assert err_full <= err_small + 0.1, (
            f"Full bond error {err_full} should be <= small bond error {err_small}"
        )

    def test_from_dense_shape_preserved(self):
        np.random.seed(2)
        phys_dims = [2, 3, 2]
        total = 2 * 3 * 2
        vec = np.random.randn(total).astype(np.float32)
        mps = mps_from_dense(jnp.array(vec), phys_dims, max_bond=4)

        dense = mps_to_dense(mps)
        assert dense.shape == tuple(phys_dims)

    def test_from_dense_product_state(self):
        """Product state (rank-1) should be reconstructed with bond_dim=1."""
        # Create a product state |+>|+>|+>
        plus = np.array([1.0, 1.0]) / math.sqrt(2)
        vec = np.kron(np.kron(plus, plus), plus).astype(np.float32)
        phys_dims = [2, 2, 2]

        mps = mps_from_dense(jnp.array(vec), phys_dims, max_bond=1)
        reconstructed = np.array(mps_to_dense(mps)).reshape(-1)

        error = np.linalg.norm(vec - reconstructed) / (np.linalg.norm(vec) + 1e-12)
        assert error < 0.01, f"Product state error: {error}"


# ---------------------------------------------------------------------------
# Test: Contraction
# ---------------------------------------------------------------------------

class TestMPSContraction:
    def test_dense_shape(self, small_mps):
        dense = mps_to_dense(small_mps)
        assert dense.shape == (2, 2, 2, 2)

    def test_dense_shape_heterogeneous(self, rng):
        phys_dims = [2, 3, 4]
        mps = mps_random(3, phys_dims, 3, rng)
        dense = mps_to_dense(mps)
        assert dense.shape == tuple(phys_dims)

    def test_single_site_mps(self):
        t = jnp.array([[[1.0, 2.0, 3.0]]])  # (1, 3, 1)
        mps = MatrixProductState([t], (3,))
        dense = mps_to_dense(mps)
        assert dense.shape == (3,)
        np.testing.assert_allclose(np.array(dense), np.array([1.0, 2.0, 3.0]), rtol=1e-5)

    def test_bond_dim_1_product_state(self, product_mps):
        dense = mps_to_dense(product_mps)
        dense_flat = np.array(dense).reshape(-1)
        # |0101> — should have amplitude 1 at index 0b0101 = 5
        idx_0101 = 0 * 8 + 1 * 4 + 0 * 2 + 1 * 1  # = 5
        assert abs(dense_flat[idx_0101]) > 0.5, \
            f"Expected amplitude at index 5, got {dense_flat}"


# ---------------------------------------------------------------------------
# Test: Inner product
# ---------------------------------------------------------------------------

class TestMPSInnerProduct:
    def test_norm_positive(self, small_mps):
        norm = float(mps_norm(small_mps))
        assert norm > 0.0

    def test_self_inner_product_real(self, small_mps):
        ip = mps_inner_product(small_mps, small_mps)
        assert abs(float(jnp.imag(ip))) < 1e-5, "Self inner product should be real"
        assert float(jnp.real(ip)) > 0, "Self inner product should be positive"

    def test_inner_product_orthogonal_product_states(self):
        """<01|10> should be 0."""
        t0 = jnp.array([[[1.0, 0.0]]]).reshape(1, 2, 1)
        t1 = jnp.array([[[0.0, 1.0]]]).reshape(1, 2, 1)

        mps_01 = MatrixProductState([t0, t1], (2, 2))
        mps_10 = MatrixProductState([t1, t0], (2, 2))

        ip = float(jnp.real(mps_inner_product(mps_01, mps_10)))
        assert abs(ip) < 1e-5, f"<01|10> should be 0, got {ip}"

    def test_inner_product_identical_states(self):
        """<psi|psi> = ||psi||^2 >= 0."""
        key = jax.random.PRNGKey(5)
        mps = mps_random(4, 2, 3, key)
        ip = float(jnp.real(mps_inner_product(mps, mps)))
        assert ip > 0, f"<psi|psi> should be positive, got {ip}"

    def test_inner_product_bilinear(self):
        """<alpha*psi|phi> = conj(alpha) * <psi|phi>."""
        key = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(key)
        mps1 = mps_random(3, 2, 2, k1)
        mps2 = mps_random(3, 2, 2, k2)
        alpha = 2.5

        ip1 = float(jnp.real(mps_inner_product(mps_scale(mps1, alpha), mps2)))
        ip2 = alpha * float(jnp.real(mps_inner_product(mps1, mps2)))
        assert abs(ip1 - ip2) < 1e-4, f"Bilinearity failed: {ip1} vs {ip2}"


# ---------------------------------------------------------------------------
# Test: Compression
# ---------------------------------------------------------------------------

class TestMPSCompression:
    def test_compression_reduces_bond_dim(self):
        key = jax.random.PRNGKey(10)
        mps = mps_random(6, 2, 8, key)
        target_bond = 4
        mps_c, err = mps_compress(mps, target_bond)
        assert mps_c.max_bond <= target_bond

    def test_compression_error_nonnegative(self):
        key = jax.random.PRNGKey(11)
        mps = mps_random(5, 2, 4, key)
        _, trunc_err = mps_compress(mps, 2)
        assert trunc_err >= 0.0

    def test_compression_error_increases_with_smaller_bond(self):
        """Smaller bond dim → larger truncation error (generally)."""
        key = jax.random.PRNGKey(12)
        mps = mps_random(6, 2, 16, key)

        dense_orig = mps_to_dense(mps)
        errors = []
        for D in [8, 4, 2, 1]:
            mps_c, _ = mps_compress(mps, D)
            err = mps_frobenius_error(mps_c, dense_orig)
            errors.append(err)

        # Errors should be non-decreasing as bond dim decreases
        for i in range(len(errors) - 1):
            assert errors[i] <= errors[i + 1] + 0.1, \
                f"Error not monotone: D errors={list(zip([8,4,2,1], errors))}"

    def test_full_bond_compression_error_small(self):
        """Compressing with max_bond >= actual max_bond should have tiny error."""
        np.random.seed(0)
        vec = np.random.randn(8).astype(np.float32)
        phys_dims = [2, 2, 2]
        mps = mps_from_dense(jnp.array(vec), phys_dims, max_bond=4)
        dense_orig = mps_to_dense(mps)

        mps_c, _ = mps_compress(mps, 4)
        err = mps_frobenius_error(mps_c, dense_orig)
        assert err < 0.05, f"Full-bond compression error should be small: {err}"

    def test_compression_analysis_returns_valid(self):
        key = jax.random.PRNGKey(15)
        mps = mps_random(4, 2, 4, key)
        results = mps_compression_analysis(mps, [1, 2, 4])
        assert len(results["bond_dims"]) == 3
        assert len(results["errors"]) == 3
        assert all(e >= 0 for e in results["errors"])

    def test_compression_pytree_compatible(self):
        """Compression should work through jit."""
        key = jax.random.PRNGKey(20)
        mps = mps_random(4, 2, 4, key)

        @jax.jit
        def compress_and_norm(mps):
            mps_c, _ = mps_compress(mps, 2)
            return mps_norm(mps_c)

        # Just check it runs
        norm = compress_and_norm(mps)
        assert float(norm) > 0


# ---------------------------------------------------------------------------
# Test: Canonicalization
# ---------------------------------------------------------------------------

class TestMPSCanonicalization:
    def test_left_canonical_gauge(self):
        """
        After left canonicalization, each tensor A[i] should satisfy:
        sum_s A[i]^{s†} A[i]^{s} = I  (for i < N-1)
        """
        key = jax.random.PRNGKey(30)
        mps = mps_random(4, 2, 3, key)
        mps_lc = mps_left_canonicalize(mps)

        for i in range(mps.n_sites - 1):
            t = mps_lc.tensors[i]  # (chi_l, d, chi_r)
            chi_l, d, chi_r = t.shape
            M = t.reshape(chi_l * d, chi_r)
            # Should be column-isometric: M^T M ≈ I
            MtM = jnp.array(M.T @ M)
            identity = jnp.eye(chi_r)
            diff = jnp.linalg.norm(MtM - identity)
            assert float(diff) < 0.1, \
                f"Site {i} not left-canonical: ||M^T M - I|| = {diff:.4f}"

    def test_right_canonical_gauge(self):
        """
        After right canonicalization, each tensor A[i] (i > 0) should satisfy:
        sum_s A[i]^{s} A[i]^{s†} = I
        """
        key = jax.random.PRNGKey(31)
        mps = mps_random(4, 2, 3, key)
        mps_rc = mps_right_canonicalize(mps)

        for i in range(1, mps.n_sites):
            t = mps_rc.tensors[i]  # (chi_l, d, chi_r)
            chi_l, d, chi_r = t.shape
            M = t.reshape(chi_l, d * chi_r)
            # Should be row-isometric: M M^T ≈ I
            MMt = jnp.array(M @ M.T)
            identity = jnp.eye(chi_l)
            diff = jnp.linalg.norm(MMt - identity)
            assert float(diff) < 0.1, \
                f"Site {i} not right-canonical: ||M M^T - I|| = {diff:.4f}"

    def test_canonicalization_preserves_state(self):
        """Left/right canonicalization should not change the represented state."""
        key = jax.random.PRNGKey(35)
        mps = mps_random(4, 2, 3, key)

        dense_orig = np.array(mps_to_dense(mps)).reshape(-1)
        mps_lc = mps_left_canonicalize(mps)
        dense_lc = np.array(mps_to_dense(mps_lc)).reshape(-1)

        norm_orig = np.linalg.norm(dense_orig)
        norm_lc = np.linalg.norm(dense_lc)

        # After normalization, should be same state
        if norm_orig > 1e-10 and norm_lc > 1e-10:
            cosine = np.dot(dense_orig / norm_orig, dense_lc / norm_lc)
            assert abs(abs(cosine) - 1.0) < 0.1, \
                f"Canonicalization changed state: cosine similarity = {cosine}"

    def test_right_canonicalize_preserves_norm(self):
        key = jax.random.PRNGKey(36)
        mps = mps_random(5, 2, 4, key)

        norm_before = float(mps_norm(mps))
        mps_rc = mps_right_canonicalize(mps)
        norm_after = float(mps_norm(mps_rc))

        # Norms should be approximately equal
        if norm_before > 1e-10:
            ratio = norm_after / norm_before
            assert 0.5 < ratio < 2.0, \
                f"Right canonicalization changed norm by {ratio}x"


# ---------------------------------------------------------------------------
# Test: MPS arithmetic
# ---------------------------------------------------------------------------

class TestMPSArithmetic:
    def test_add_bond_dims(self):
        """After adding two MPS, bond dims should be sum of originals."""
        key = jax.random.PRNGKey(40)
        k1, k2 = jax.random.split(key)
        mps1 = mps_random(4, 2, 2, k1)
        mps2 = mps_random(4, 2, 3, k2)

        mps_sum = mps_add(mps1, mps2)
        assert mps_sum.n_sites == 4
        # Middle bond dims should be sum
        for i in range(1, mps_sum.n_sites - 1):
            bd_sum = mps_sum.tensors[i].shape[0]
            bd1 = mps1.tensors[i].shape[0]
            bd2 = mps2.tensors[i].shape[0]
            assert bd_sum == bd1 + bd2, \
                f"Site {i}: expected bond {bd1+bd2}, got {bd_sum}"

    def test_add_correctness(self):
        """MPS addition should produce the sum of states."""
        key = jax.random.PRNGKey(41)
        k1, k2 = jax.random.split(key)
        mps1 = mps_random(3, 2, 2, k1)
        mps2 = mps_random(3, 2, 2, k2)

        dense1 = np.array(mps_to_dense(mps1)).reshape(-1)
        dense2 = np.array(mps_to_dense(mps2)).reshape(-1)
        dense_sum = dense1 + dense2

        mps_s = mps_add(mps1, mps2)
        dense_mps_sum = np.array(mps_to_dense(mps_s)).reshape(-1)

        error = np.linalg.norm(dense_sum - dense_mps_sum)
        assert error < 0.01, f"MPS addition error: {error}"

    def test_scale(self):
        """Scaling MPS by scalar should scale all amplitudes."""
        key = jax.random.PRNGKey(42)
        mps = mps_random(3, 2, 2, key)
        alpha = 3.0

        dense_orig = np.array(mps_to_dense(mps)).reshape(-1)
        mps_scaled = mps_scale(mps, alpha)
        dense_scaled = np.array(mps_to_dense(mps_scaled)).reshape(-1)

        error = np.linalg.norm(dense_scaled - alpha * dense_orig)
        assert error < 0.01, f"Scale error: {error}"

    def test_add_commutative(self):
        """Addition should be approximately commutative (same dense vector)."""
        key = jax.random.PRNGKey(43)
        k1, k2 = jax.random.split(key)
        mps1 = mps_random(3, 2, 2, k1)
        mps2 = mps_random(3, 2, 2, k2)

        sum_12 = np.array(mps_to_dense(mps_add(mps1, mps2))).reshape(-1)
        sum_21 = np.array(mps_to_dense(mps_add(mps2, mps1))).reshape(-1)

        error = np.linalg.norm(sum_12 - sum_21)
        assert error < 0.01, f"Commutativity failed: {error}"


# ---------------------------------------------------------------------------
# Test: Expectation values
# ---------------------------------------------------------------------------

class TestExpectationValues:
    def test_identity_operator_gives_norm_sq(self):
        """<psi|I|psi> = <psi|psi> = ||psi||^2."""
        key = jax.random.PRNGKey(50)
        mps = mps_random(4, 2, 3, key)

        I_op = jnp.eye(2, dtype=jnp.float32)
        exp_val = float(mps_expectation_single(mps, I_op, 0))
        norm_sq = float(jnp.real(mps_inner_product(mps, mps)))

        assert abs(exp_val - norm_sq) < 0.1, \
            f"Identity expectation {exp_val} != norm^2 {norm_sq}"

    def test_pauli_z_product_state(self):
        """For |0>, <Z> = 1; for |1>, <Z> = -1."""
        # |0> state
        t0 = jnp.array([[[1.0, 0.0]]]).reshape(1, 2, 1)
        t1 = jnp.array([[[1.0, 0.0]]]).reshape(1, 2, 1)  # both |0>
        mps_00 = MatrixProductState([t0, t1], (2, 2))

        Z = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        exp_z0 = float(mps_expectation_single(mps_00, Z, 0))
        # For unnormalized |0>: <0|Z|0> = 1
        assert abs(exp_z0 - 1.0) < 0.1, f"<0|Z|0> should be 1, got {exp_z0}"


# ---------------------------------------------------------------------------
# Test: Entanglement entropy
# ---------------------------------------------------------------------------

class TestEntanglement:
    def test_product_state_zero_entropy(self):
        """Product state should have zero entanglement entropy."""
        key = jax.random.PRNGKey(60)
        mps = mps_identity(4, 2)
        entropies = np.array(mps_bond_entropies(mps))
        # Product state: all entropies should be ~0
        assert np.max(entropies) < 0.5, \
            f"Product state entropy should be low: {entropies}"

    def test_entropies_nonnegative(self):
        """Entanglement entropies should be non-negative."""
        key = jax.random.PRNGKey(61)
        mps = mps_random(5, 2, 4, key)
        entropies = np.array(mps_bond_entropies(mps))
        assert np.all(entropies >= -0.01), \
            f"Negative entropies found: {entropies}"

    def test_entropies_count(self):
        """Should have n_sites - 1 bond entropies."""
        key = jax.random.PRNGKey(62)
        mps = mps_random(5, 2, 3, key)
        entropies = mps_bond_entropies(mps)
        assert len(entropies) == 4, \
            f"Expected 4 entropies for 5-site MPS, got {len(entropies)}"

    def test_entanglement_spectrum_sorted(self):
        """Entanglement spectrum should be non-increasing."""
        key = jax.random.PRNGKey(63)
        mps = mps_random(5, 2, 4, key)
        spectrum = np.array(mps_entanglement_spectrum(mps, 2))
        # Check sorted descending
        assert np.all(np.diff(spectrum) <= 0.01), \
            f"Spectrum not sorted: {spectrum}"


# ---------------------------------------------------------------------------
# Test: Density matrix
# ---------------------------------------------------------------------------

class TestDensityMatrix:
    def test_density_matrix_shape(self):
        key = jax.random.PRNGKey(70)
        mps = mps_random(3, 2, 2, key)
        rho = mps_to_density_matrix(mps)
        assert rho.shape == (8, 8)

    def test_density_matrix_hermitian(self):
        key = jax.random.PRNGKey(71)
        mps = mps_random(3, 2, 2, key)
        rho = np.array(mps_to_density_matrix(mps))
        assert np.allclose(rho, rho.conj().T, atol=1e-5), "Density matrix not Hermitian"

    def test_density_matrix_trace_one(self):
        key = jax.random.PRNGKey(72)
        mps = mps_random(3, 2, 2, key)
        rho = np.array(mps_to_density_matrix(mps, normalize=True))
        trace = np.trace(rho)
        assert abs(trace - 1.0) < 0.01, f"Trace = {trace}, expected 1"

    def test_density_matrix_psd(self):
        """Density matrix should be positive semidefinite."""
        key = jax.random.PRNGKey(73)
        mps = mps_random(3, 2, 2, key)
        rho = np.array(mps_to_density_matrix(mps, normalize=True))
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-5), f"Negative eigenvalues: {eigvals.min()}"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_compress_and_recover(self):
        """
        Build an MPS, compress to smaller bond, then check that
        error decreases monotonically with bond dim.
        """
        np.random.seed(99)
        key = jax.random.PRNGKey(99)

        # Build high-bond MPS
        mps = mps_random(6, 2, 8, key)
        dense = mps_to_dense(mps)

        errors = []
        for D in [1, 2, 4, 8]:
            mps_c, _ = mps_compress(mps, D)
            err = mps_frobenius_error(mps_c, dense)
            errors.append(err)

        # Errors should be roughly non-increasing with D
        # (not strictly, due to numerical noise, but trend should hold)
        assert errors[0] >= errors[-1] - 0.2, \
            f"Errors not decreasing: {errors}"

    def test_add_compress_cycle(self):
        """Adding MPS doubles bond dims; compressing should recover compact form."""
        key = jax.random.PRNGKey(100)
        k1, k2 = jax.random.split(key)
        mps1 = mps_random(4, 2, 2, k1)
        mps2 = mps_random(4, 2, 2, k2)

        mps_sum = mps_add(mps1, mps2)
        # Bond dims should be doubled
        assert mps_sum.max_bond <= 4

        # Compress back
        mps_sum_c, _ = mps_compress(mps_sum, 4)
        assert mps_sum_c.max_bond <= 4

        # The represented states should be same
        dense_sum = np.array(mps_to_dense(mps_sum)).reshape(-1)
        dense_c = np.array(mps_to_dense(mps_sum_c)).reshape(-1)

        n1 = np.linalg.norm(dense_sum)
        n2 = np.linalg.norm(dense_c)
        if n1 > 1e-10 and n2 > 1e-10:
            cos = np.dot(dense_sum / n1, dense_c / n2)
            assert abs(abs(cos) - 1.0) < 0.2, f"Cosine similarity = {cos}"

    def test_gradient_through_mps(self):
        """JAX grad should work through MPS operations."""
        key = jax.random.PRNGKey(101)
        mps = mps_random(3, 2, 2, key)
        target = jnp.ones((2, 2, 2), dtype=jnp.float32)

        def loss(tensors):
            m = MatrixProductState(tensors, (2, 2, 2))
            return jnp.mean((mps_to_dense(m) - target) ** 2)

        grads = jax.grad(loss)(mps.tensors)
        assert len(grads) == 3
        assert all(g.shape == t.shape for g, t in zip(grads, mps.tensors))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
